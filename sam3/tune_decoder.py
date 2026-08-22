#!/usr/bin/env python3
"""
Fine-tune SAM3 decoder + bbox head only.
Backbone (vision + language) is frozen by default.
No mask GT required — only image, text, and normalized bbox annotations.

Single-GPU usage:
    python tune_decoder.py \
        --data    data.json \
        --ckpt    /path/to/sam3_checkpoint.pt \
        --output  ./output \
        --epochs  20 \
        --batch_size 28 \
        --lr 8e-5

Multi-GPU usage (DDP):
    torchrun --nproc_per_node=2 tune_decoder.py \
        --data    data.json \
        --ckpt    /path/to/sam3_checkpoint.pt \
        --output  ./output \
        --epochs  20 \
        --batch_size 28 \
        --lr 8e-5

Conda env: x-anylabeling-sam2
    python tune_decoder.py ...

NOTE: SAM3's vision backbone uses a fused CUDA kernel (addmm_act) that
physically raises an error when torch.is_grad_enabled().  For this reason
the backbone must always be run under torch.no_grad().  We split the forward
pass accordingly:
  1. backbone (no_grad)  →  backbone_out (all tensors detached)
  2. encoder + decoder   →  gradients flow here
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler  # kept for potential future use
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import torchvision.transforms.functional as TVF
import wandb

# ── SAM3 imports ──────────────────────────────────────────────────────────────
SAM3_ROOT = Path(__file__).parent
sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
)
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.geometry_encoders import Prompt
from sam3.model.model_misc import SAM3Output
from sam3.train.data.collator import packed_to_padded_naive
from sam3.train.loss.loss_fns import Boxes, IABCEMdetr
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher


# ── DDP helpers ──────────────────────────────────────────────────────────────

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return not is_dist() or dist.get_rank() == 0

def setup_ddp():
    """Initialise DDP if launched via torchrun. Returns local_rank."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return -1  # single-GPU mode
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank


# ── Image preprocessing ───────────────────────────────────────────────────────
IMG_SIZE   = 1008
IMG_MEAN   = [0.5, 0.5, 0.5]
IMG_STD    = [0.5, 0.5, 0.5]


def preprocess_image(path: str) -> torch.Tensor:
    """Load, resize (square pad), and normalise an image → [3, 1008, 1008]."""
    tensor, _ = preprocess_image_with_meta(path)
    return tensor


def preprocess_image_with_meta(path: str) -> tuple[torch.Tensor, dict]:
    """
    Load, resize (square pad), and normalise an image.
    Returns:
      tensor: [3, IMG_SIZE, IMG_SIZE]
      meta: geometry info needed to map boxes into padded-normalized coords.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # Resize longest side to IMG_SIZE while keeping aspect ratio
    scale = IMG_SIZE / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Pad to square
    tensor = TVF.to_tensor(img)                                  # [3, new_h, new_w]
    pad_bottom = IMG_SIZE - new_h
    pad_right  = IMG_SIZE - new_w
    tensor = torch.nn.functional.pad(tensor, (0, pad_right, 0, pad_bottom), value=0.0)

    # Normalise
    mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMG_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std                               # [3, 1008, 1008]
    meta = {
        "orig_w": float(w),
        "orig_h": float(h),
        "new_w": float(new_w),
        "new_h": float(new_h),
    }
    return tensor, meta


# ── Dataset ───────────────────────────────────────────────────────────────────

class BBoxDataset(Dataset):
    """
    Reads a JSON file of the form:
        [
          {
            "image": "path/to/image.jpg",
            "text":  "a description of the target object",
            "boxes": [[cx, cy, w, h], ...]   // normalised to [0, 1]
          },
          ...
        ]

    'boxes' must be in normalised centre-x, centre-y, width, height format.
    If your data is in normalised x1y1x2y2 format, pass --box_fmt xyxy.
    """

    MAX_BOXES = 10  # cap boxes per sample to avoid OOM on outliers

    def __init__(
        self,
        json_path: str,
        box_fmt: str = "cxcywh",
        boxes_reference: str = "original",
    ):
        with open(json_path) as f:
            self.samples = json.load(f)
        assert box_fmt in ("cxcywh", "xyxy"), f"Unknown box_fmt: {box_fmt}"
        assert boxes_reference in ("original", "padded"), (
            f"Unknown boxes_reference: {boxes_reference}"
        )
        self.box_fmt = box_fmt
        self.boxes_reference = boxes_reference

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image, img_meta = preprocess_image_with_meta(s["image"])  # [3, 1008, 1008]
        text   = s["text"]
        boxes  = torch.tensor(s["boxes"], dtype=torch.float32)   # [N, 4]
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        if boxes.shape[0] > self.MAX_BOXES:
            boxes = boxes[:self.MAX_BOXES]

        if boxes.numel() > 0:
            if self.box_fmt == "xyxy":
                x1, y1, x2, y2 = boxes.unbind(-1)
            else:
                cx, cy, bw, bh = boxes.unbind(-1)
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2

            # If boxes are normalized in original-image coordinates, map them
            # into padded-canvas normalized coordinates (IMG_SIZE x IMG_SIZE),
            # matching preprocess_image's resize + right/bottom pad.
            if self.boxes_reference == "original":
                fx = img_meta["new_w"] / IMG_SIZE
                fy = img_meta["new_h"] / IMG_SIZE
                x1 = x1 * fx
                x2 = x2 * fx
                y1 = y1 * fy
                y2 = y2 * fy

            x1 = x1.clamp(0.0, 1.0)
            y1 = y1.clamp(0.0, 1.0)
            x2 = x2.clamp(0.0, 1.0)
            y2 = y2.clamp(0.0, 1.0)

            # Back to normalized cxcywh expected by SAM3 training code.
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w  = (x2 - x1).clamp(min=0.0)
            h  = (y2 - y1).clamp(min=0.0)
            boxes = torch.stack([cx, cy, w, h], dim=-1)

        return {"image": image, "text": text, "boxes": boxes}


def collate_fn(batch):
    """Returns a list of dicts (one per sample) — batching is done in build_datapoint."""
    return batch


# ── BatchedDatapoint builder ──────────────────────────────────────────────────

def build_datapoint(batch, device: torch.device) -> BatchedDatapoint:
    """
    Convert a list of {'image', 'text', 'boxes'} dicts into a BatchedDatapoint
    ready for Sam3Image.forward().

    One FindStage is created with B queries (one per image).
    No interactive box/point prompts are used.
    """
    B = len(batch)

    # ── images ────────────────────────────────────────────────────────────────
    img_batch = torch.stack([s["image"] for s in batch]).to(device)   # [B, 3, H, W]

    # ── text (deduplicated list; text_ids maps each query to an entry) ────────
    find_text_batch: list[str] = []
    text_ids_list: list[int]   = []
    for s in batch:
        t = s["text"]
        if t not in find_text_batch:
            find_text_batch.append(t)
        text_ids_list.append(find_text_batch.index(t))

    # ── FindStage (no input box/point prompts) ────────────────────────────────
    img_ids          = torch.arange(B, dtype=torch.long,  device=device)
    text_ids         = torch.tensor(text_ids_list, dtype=torch.long, device=device)

    # Empty box prompts — shape convention: [N_boxes, B, 4]
    input_boxes      = torch.zeros(0, B, 4,  dtype=torch.float32, device=device)
    input_boxes_mask = torch.zeros(B, 0,     dtype=torch.bool,    device=device)
    input_boxes_label= torch.zeros(0, B,     dtype=torch.long,    device=device)

    # Empty point prompts — shape convention: [N_pts, B, 2]
    input_points      = torch.zeros(0, B, 2, dtype=torch.float32, device=device)
    input_points_mask = torch.zeros(B, 0,    dtype=torch.bool,    device=device)

    find_input = FindStage(
        img_ids           = img_ids,
        text_ids          = text_ids,
        input_boxes       = input_boxes,
        input_boxes_mask  = input_boxes_mask,
        input_boxes_label = input_boxes_label,
        input_points      = input_points,
        input_points_mask = input_points_mask,
        object_ids        = None,
    )

    # ── BatchedFindTarget ─────────────────────────────────────────────────────
    boxes_list = [s["boxes"].to(device) for s in batch]
    num_boxes  = torch.tensor([len(b) for b in boxes_list],
                               dtype=torch.long, device=device)

    if num_boxes.sum() > 0:
        boxes_packed = torch.cat(boxes_list, dim=0)    # [sum_N, 4] cxcywh
    else:
        boxes_packed = torch.zeros(0, 4, device=device)

    # padded representations
    if num_boxes.sum() > 0:
        boxes_padded = packed_to_padded_naive(boxes_packed, num_boxes)  # [B, Nmax, 4]
    else:
        boxes_padded = torch.zeros(B, 1, 4, device=device)

    object_ids = torch.arange(boxes_packed.shape[0],
                               dtype=torch.long, device=device)         # [sum_N]
    if num_boxes.sum() > 0:
        obj_ids_padded = packed_to_padded_naive(
            object_ids.unsqueeze(-1), num_boxes, fill_value=-1
        ).squeeze(-1)                                                    # [B, Nmax]
    else:
        obj_ids_padded = torch.full((B, 1), -1,
                                     dtype=torch.long, device=device)

    find_target = BatchedFindTarget(
        num_boxes          = num_boxes,
        boxes              = boxes_packed,
        boxes_padded       = boxes_padded,
        repeated_boxes     = torch.zeros(0, 4, dtype=torch.float32, device=device),
        segments           = None,
        semantic_segments  = None,
        is_valid_segment   = None,
        is_exhaustive      = torch.ones(B, dtype=torch.bool, device=device),
        object_ids         = object_ids,
        object_ids_padded  = obj_ids_padded,
    )

    # ── BatchedInferenceMetadata (dummy — not used during training) ───────────
    find_metadata = BatchedInferenceMetadata(
        coco_image_id       = torch.zeros(B, dtype=torch.long, device=device),
        original_image_id   = torch.zeros(B, dtype=torch.long, device=device),
        original_category_id= torch.zeros(B, dtype=torch.int,  device=device),
        original_size       = torch.tensor([[IMG_SIZE, IMG_SIZE]] * B,
                                            dtype=torch.long, device=device),
        object_id           = torch.zeros(B, dtype=torch.long, device=device),
        frame_index         = torch.zeros(B, dtype=torch.long, device=device),
        is_conditioning_only= [None] * B,
    )

    return BatchedDatapoint(
        img_batch      = img_batch,
        find_text_batch= find_text_batch,
        find_inputs    = [find_input],
        find_targets   = [find_target],
        find_metadatas = [find_metadata],
    )


# ── Parameter group helpers ───────────────────────────────────────────────────

def get_trainable_params(model, freeze_encoder: bool):
    """
    Always freeze: backbone (vision + language).
    Optionally freeze: transformer encoder.
    Always train: transformer decoder + box/class heads + geometry encoder.
    Returns a list of parameter dicts for the optimiser.
    """
    frozen_prefixes = ["backbone."]
    if freeze_encoder:
        frozen_prefixes.append("transformer.encoder.")

    frozen, trainable = [], []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in frozen_prefixes):
            param.requires_grad_(False)
            frozen.append(name)
        else:
            param.requires_grad_(True)
            trainable.append(name)

    print(f"Frozen   parameters : {len(frozen)}")
    print(f"Trainable parameters: {len(trainable)}")
    return [p for p in model.parameters() if p.requires_grad]


# ── Split forward (backbone no_grad / decoder with grad) ─────────────────────

def _deep_detach(obj):
    """Recursively detach all tensors in a nested dict/list structure."""
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    if isinstance(obj, dict):
        return {k: _deep_detach(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        detached = [_deep_detach(v) for v in obj]
        return type(obj)(detached)
    return obj


def _deep_to_float32(obj):
    """Recursively cast all floating-point tensors to fp32."""
    if isinstance(obj, torch.Tensor):
        return obj.float() if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: _deep_to_float32(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_deep_to_float32(v) for v in obj]
        return type(obj)(converted)
    return obj


def model_forward(model, dp: BatchedDatapoint, amp: bool = True) -> SAM3Output:
    """
    Two-phase forward pass:
      Phase 1 — backbone (vision + language) under torch.no_grad() + bf16.
                The backbone uses fused inference kernels that forbid grad.
      Phase 2 — encoder + decoder in fp32 (bf16 causes NaN in the matcher's
                GIoU / cost-matrix computation).
    """
    raw = model.module if isinstance(model, DDP) else model
    device = raw.device

    # ── Phase 1: backbone (no grad, bf16 for speed) ──────────────────────────
    with torch.no_grad(), torch.autocast(device_type=device.type,
                                          dtype=torch.bfloat16, enabled=amp):
        backbone_out: dict = {"img_batch_all_stages": dp.img_batch}
        backbone_out.update(raw.backbone.forward_image(dp.img_batch))
        backbone_out.update(raw.backbone.forward_text(dp.find_text_batch,
                                                       device=device))

    # Detach and cast backbone outputs to fp32 for the decoder
    backbone_out = _deep_detach(backbone_out)
    backbone_out = _deep_to_float32(backbone_out)

    # ── Phase 2: encoder + decoder in fp32 (no autocast) ─────────────────────
    find_input  = dp.find_inputs[0]
    find_target = dp.find_targets[0]

    geometric_prompt = Prompt(
        box_embeddings  = find_input.input_boxes,
        box_mask        = find_input.input_boxes_mask,
        box_labels      = find_input.input_boxes_label,
    )

    out = raw.forward_grounding(
        backbone_out     = backbone_out,
        find_input       = find_input,
        find_target      = find_target,
        geometric_prompt = geometric_prompt.clone(),
    )

    result = SAM3Output(iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE)
    result.append([out])
    return result


# ── Loss setup ────────────────────────────────────────────────────────────────

def build_loss(device):
    # O2O matcher (one prediction per GT)
    matcher = BinaryHungarianMatcherV2(
        focal=True,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        alpha=0.25,
        gamma=2,
        stable=False,
    )
    # O2M matcher (multiple predictions per GT, matches the pre-trained config)
    o2m_matcher = BinaryOneToManyMatcher(
        alpha=0.3,
        threshold=0.4,
        topk=4,
    )
    loss_fns = [
        Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
        IABCEMdetr(
            weak_loss=False,
            weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
            pos_weight=10.0,
            alpha=0.25,
            gamma=2,
            use_presence=True,
            pos_focal=False,
            pad_n_queries=200,
            pad_scale_pos=1.0,
        ),
    ]
    loss_wrapper = Sam3LossWrapper(
        loss_fns_find=loss_fns,
        normalization="local",           # "global" requires distributed training
        matcher=matcher,
        o2m_matcher=o2m_matcher,
        o2m_weight=2.0,
        use_o2m_matcher_on_o2m_aux=False,
        loss_fn_semantic_seg=None,
        scale_by_find_batch_size=False,
    ).to(device)
    return loss_wrapper


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, eval_dataloader, device, amp: bool):
    """
    Run eval on the full eval set.  Returns a dict of scalar metrics:
      mean_iou, iou@50, iou@75, mean_l1, mean_giou
    """
    raw = model.module if isinstance(model, DDP) else model
    was_training = raw.training
    raw.eval()

    all_ious, all_l1s, all_gious = [], [], []

    for batch in eval_dataloader:
        dp = build_datapoint(batch, device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                             enabled=amp):
            # backbone (no grad already via decorator)
            backbone_out: dict = {"img_batch_all_stages": dp.img_batch}
            backbone_out.update(raw.backbone.forward_image(dp.img_batch))
            backbone_out.update(raw.backbone.forward_text(dp.find_text_batch,
                                                           device=device))
            backbone_out = _deep_detach(backbone_out)

            find_input  = dp.find_inputs[0]
            find_target = dp.find_targets[0]

            geometric_prompt = Prompt(
                box_embeddings  = find_input.input_boxes,
                box_mask        = find_input.input_boxes_mask,
                box_labels      = find_input.input_boxes_label,
            )
            out = raw.forward_grounding(
                backbone_out     = backbone_out,
                find_input       = find_input,
                find_target      = find_target,
                geometric_prompt = geometric_prompt.clone(),
            )

        # pred_boxes: [B, num_queries, 4] cxcywh normalised
        pred_boxes = out["pred_boxes"].float()           # [B, Q, 4]
        pred_logits = out["pred_logits"].float()         # [B, Q, 1]

        # GT boxes per sample
        gt_boxes_list = [s["boxes"].to(device) for s in batch]

        B = pred_boxes.shape[0]
        for b in range(B):
            gt = gt_boxes_list[b]                        # [Ngt, 4] cxcywh
            if gt.shape[0] == 0:
                continue

            scores = pred_logits[b].squeeze(-1).sigmoid() # [Q]
            preds  = pred_boxes[b]                        # [Q, 4]

            # For each GT box, find the best matching prediction (highest IoU)
            gt_xyxy   = box_cxcywh_to_xyxy(gt)           # [Ngt, 4]
            pred_xyxy = box_cxcywh_to_xyxy(preds)        # [Q, 4]

            # Pairwise IoU: [Ngt, Q]
            iou_matrix = _pairwise_iou(gt_xyxy, pred_xyxy)

            # Best prediction per GT
            best_iou, best_idx = iou_matrix.max(dim=1)   # [Ngt]
            matched_preds = preds[best_idx]               # [Ngt, 4]

            # L1 distance
            l1 = (matched_preds - gt).abs().sum(dim=-1)  # [Ngt]

            # GIoU
            matched_xyxy = pred_xyxy[best_idx]
            giou = _diag_giou(gt_xyxy, matched_xyxy)     # [Ngt]

            all_ious.append(best_iou)
            all_l1s.append(l1)
            all_gious.append(giou)

    if was_training:
        raw.train()

    if not all_ious:
        return {}

    ious  = torch.cat(all_ious)
    l1s   = torch.cat(all_l1s)
    gious = torch.cat(all_gious)

    return {
        "eval/mean_iou":  ious.mean().item(),
        "eval/iou@50":    (ious >= 0.50).float().mean().item(),
        "eval/iou@75":    (ious >= 0.75).float().mean().item(),
        "eval/mean_l1":   l1s.mean().item(),
        "eval/mean_giou": gious.mean().item(),
    }


def _pairwise_iou(boxes1, boxes2):
    """Compute IoU between all pairs. boxes in xyxy format.
    Returns [N, M] IoU matrix."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)                                # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]                          # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _diag_giou(boxes1, boxes2):
    """GIoU between corresponding boxes (same count). xyxy format. Returns [N]."""
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter

    # Enclosing box
    enc_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    enc_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enc_area = (enc_rb[:, 0] - enc_lt[:, 0]) * (enc_rb[:, 1] - enc_lt[:, 1])

    iou = inter / union.clamp(min=1e-6)
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    # ── DDP setup ─────────────────────────────────────────────────────────────
    local_rank = setup_ddp()
    use_ddp = local_rank >= 0
    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print(f"Using device: {device}  (DDP={use_ddp}, "
              f"world_size={dist.get_world_size() if use_ddp else 1})")

    # ── wandb (main process only; disabled on other ranks) ──────────────────
    world_size = dist.get_world_size() if use_ddp else 1
    if is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "epochs": args.epochs,
                "batch_size_per_gpu": args.batch_size,
                "effective_batch_size": args.batch_size * world_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "box_fmt": args.box_fmt,
                "freeze_encoder": args.freeze_encoder,
                "amp": args.amp,
                "num_gpus": world_size,
            },
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"
        wandb.init(mode="disabled")

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    dataset = BBoxDataset(
        args.data,
        box_fmt=args.box_fmt,
        boxes_reference=args.boxes_reference,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    if is_main_process():
        print(f"Dataset: {len(dataset)} samples, "
              f"{len(dataloader)} batches/epoch (batch_size={args.batch_size} x "
              f"{world_size} GPUs = {args.batch_size * world_size} effective)")

    # ── Eval DataLoader (main process only, no DDP sampler needed) ────────────
    eval_dataloader = None
    if args.eval_data and is_main_process():
        eval_dataset = BBoxDataset(
            args.eval_data,
            box_fmt=args.box_fmt,
            boxes_reference=args.boxes_reference,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        print(f"Eval set: {len(eval_dataset)} samples, "
              f"{len(eval_dataloader)} batches")

    # ── Model ─────────────────────────────────────────────────────────────────
    if is_main_process():
        print("Loading SAM3 model…")
    model = build_sam3_image_model(
        device=str(device),
        eval_mode=False,
        checkpoint_path=args.ckpt,
        enable_segmentation=False,
    )
    model = model.to(device)
    model.train()

    trainable_params = get_trainable_params(model, freeze_encoder=args.freeze_encoder)

    if use_ddp:
        # NOTE: we bypass DDP's forward (calling raw.forward_grounding directly)
        # so DDP gradient sync would not fire. Instead, each rank trains on its
        # own data shard independently — effectively data-parallel with implicit
        # averaging via DistributedSampler. We still sync model weights at the
        # end of each epoch to keep ranks consistent.
        pass

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_wrapper = build_loss(device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(dataloader)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = min(args.warmup_epochs * steps_per_epoch, total_steps)
    cosine_steps    = total_steps - warmup_steps

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=args.lr * 0.05,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    if is_main_process():
        print(f"LR schedule: warmup {warmup_steps} steps → peak {args.lr} "
              f"→ cosine decay to {args.lr * 0.05:.1e} over {cosine_steps} steps")

    # ── Output dir ────────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume from most recent per-epoch checkpoint ─────────────────────────
    raw_model = model
    start_epoch = 0
    global_step = 0
    if args.resume:
        existing_ckpts = sorted(output_dir.glob("decoder_epoch*.pt"))
        if existing_ckpts:
            resume_path = existing_ckpts[-1]
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            global_step = ckpt["global_step"]
            if is_main_process():
                print(f"Resumed from {resume_path} — epoch {start_epoch}, "
                      f"global_step {global_step}")
            del ckpt

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            datapoint = build_datapoint(batch, device)
            # Bind names so the skip/except paths can unconditionally `del` them.
            model_output = back_converted = losses = loss = None

            try:
                # Backbone runs in bf16 inside model_forward; decoder + loss in fp32
                model_output   = model_forward(model, datapoint, amp=args.amp)
                back_converted = raw_model.back_convert(datapoint.find_targets[0])
                losses = loss_wrapper(model_output, [back_converted])
                loss   = losses["core_loss"]
            except (ValueError, RuntimeError) as e:
                if "invalid numeric" in str(e) or "nan" in str(e).lower():
                    print(f"[WARNING] NaN in forward/loss at step {step}, skipping")
                    del model_output, back_converted, losses, loss
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    scheduler.step()
                    global_step += 1
                    continue
                raise

            if not torch.isfinite(loss):
                print(f"[WARNING] Non-finite loss at step {step}, skipping batch")
                # CRITICAL: backward() wasn't called, so the autograd graph stays
                # pinned to these locals. Without explicit del, the graph lives
                # until the NEXT iter reassigns them — which happens *inside*
                # model_forward(), so peak memory ~2x and we OOM.
                del model_output, back_converted, losses, loss
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                scheduler.step()
                global_step += 1
                continue

            loss.backward()

            # Sanitize gradients BEFORE clipping: replace NaN/Inf elements with 0.
            # clip_grad_norm_ cannot handle NaN — one NaN element poisons
            # total_norm, the clip coefficient, and therefore every gradient.
            # Zeroing only the bad elements preserves the learning signal from
            # well-behaved params (usually the vast majority) instead of
            # discarding the entire batch.
            for p in trainable_params:
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

            total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)

            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1

            if is_main_process() and step % args.log_freq == 0:
                loss_parts = {}
                for k, v in losses.items():
                    if k == "core_loss":
                        continue
                    if isinstance(v, torch.Tensor):
                        loss_parts[k] = v.item()
                    elif isinstance(v, (int, float)):
                        loss_parts[k] = float(v)
                if global_step == 1:
                    print(f"   [debug] loss keys: {list(losses.keys())}")
                    print(f"   [debug] loss types: {{{', '.join(f'{k}: {type(v).__name__}' for k, v in losses.items())}}}")
                wandb.log({"train/loss": loss_val,
                           "train/lr": scheduler.get_last_lr()[0],
                           **{f"train/{k}": v for k, v in loss_parts.items()}},
                          step=global_step)
                loss_parts_str = {k: f"{v:.4f}" for k, v in loss_parts.items()}
                print(f"[epoch {epoch+1:3d}/{args.epochs} | step {step:4d}] "
                      f"loss={loss_val:.4f}  {loss_parts_str}")

            # ── Periodic evaluation ──────────────────────────────────────
            if (eval_dataloader is not None
                    and is_main_process()
                    and args.eval_freq > 0
                    and global_step % args.eval_freq == 0):
                torch.cuda.empty_cache()
                print(f"   Running eval at step {global_step}…")
                metrics = evaluate(model, eval_dataloader, device, args.amp)
                if metrics:
                    wandb.log(metrics, step=global_step)
                    print(f"   Eval: " + "  ".join(
                        f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items()))
                model.train()

        avg = epoch_loss / len(dataloader)
        if is_main_process():
            wandb.log({"train/epoch_avg_loss": avg, "epoch": epoch + 1},
                      step=global_step)
            print(f"── Epoch {epoch+1}/{args.epochs}  avg_loss={avg:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Free training memory before end-of-epoch ops
        import gc
        del datapoint
        gc.collect()
        torch.cuda.empty_cache()

        # ── Sync trainable weights across ranks (backbone is frozen, skip it) ─
        if use_ddp:
            for p in model.parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=0)

        # ── End-of-epoch eval ────────────────────────────────────────────────
        if eval_dataloader is not None and is_main_process():
            print(f"   Running end-of-epoch eval…")
            metrics = evaluate(model, eval_dataloader, device, args.amp)
            if metrics:
                wandb.log(metrics, step=global_step)
                print(f"   Eval: " + "  ".join(
                    f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items()))
            model.train()

        # Save checkpoints (main process only) — one distinct file per epoch,
        # never overwritten, via atomic rename.
        if is_main_process():
            ckpt_state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": {k: v.cpu() for k, v in raw_model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg,
            }
            ckpt_path = output_dir / f"decoder_epoch{epoch+1:04d}.pt"
            tmp_path = ckpt_path.with_suffix(".pt.tmp")
            torch.save(ckpt_state, tmp_path)
            tmp_path.rename(ckpt_path)
            del ckpt_state
            print(f"   Saved checkpoint → {ckpt_path}")

    if is_main_process():
        wandb.finish()
        print("Training complete.")
    if use_ddp:
        dist.destroy_process_group()


def _trainable_prefixes(freeze_encoder: bool) -> list[str]:
    """Returns model weight key prefixes that were trained (for selective saving)."""
    frozen = ["backbone."]
    if freeze_encoder:
        frozen.append("transformer.encoder.")
    # Save everything that is not frozen
    # (simplest approach: save full state dict and filter at load time)
    return [""]  # empty prefix = all keys


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune SAM3 decoder + bbox head (no mask GT needed)"
    )
    p.add_argument("--data",     required=True,
                   help="Path to training JSON file")
    p.add_argument("--eval_data", default=None,
                   help="Path to eval JSON file (same format as --data)")
    p.add_argument("--ckpt",     default=None,
                   help="Path to pre-trained SAM3 checkpoint (.pt). "
                        "If omitted, loads from HuggingFace.")
    p.add_argument("--output",   default="./output",
                   help="Directory for saving checkpoints")
    p.add_argument("--epochs",   type=int,   default=20)
    p.add_argument("--batch_size", type=int, default=20,
                   help="Batch size per GPU (profiled max ~22 on RTX 4090 with fp32 decoder)")
    p.add_argument("--lr",       type=float, default=8e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--log_freq",     type=int,   default=10,
                   help="Print loss every N steps")
    p.add_argument("--eval_freq",    type=int,   default=100,
                   help="Run eval every N training steps (0 = end-of-epoch only)")
    p.add_argument("--box_fmt",  default="cxcywh", choices=["cxcywh", "xyxy"],
                   help="Bounding-box format in the JSON file. "
                        "cxcywh = normalised centre-x, centre-y, w, h.  "
                        "xyxy   = normalised x1, y1, x2, y2.")
    p.add_argument("--boxes_reference", default="original",
                   choices=["original", "padded"],
                   help="Coordinate reference of normalized boxes in JSON. "
                        "'original' = normalized by original image W/H "
                        "(will be resized to padded canvas coords). "
                        "'padded'   = already normalized by 1008x1008 "
                        "preprocessed canvas.")
    p.add_argument("--warmup_epochs", type=int, default=1,
                   help="Linear LR warmup epochs (default: 1)")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Also freeze the transformer encoder (train decoder only).")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use bfloat16 AMP (default: on).")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the most recent decoder_epoch*.pt in --output dir")
    p.add_argument("--wandb_project", default="sam3-finetune",
                   help="Weights & Biases project name")
    p.add_argument("--wandb_run_name", default=None,
                   help="Weights & Biases run name (auto-generated if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
