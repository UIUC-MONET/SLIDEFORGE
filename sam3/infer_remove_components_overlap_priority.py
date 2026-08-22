#!/usr/bin/env python3
"""
Alternative component-removal strategy with overlap-aware priority rules.

Compared to infer_remove_components_from_classlist.py, this script changes the
acceptance and segmentation order logic:

1) Predict candidate boxes/masks first (do not segment immediately).
2) For same-label overlapping candidates, keep only the larger box.
3) For different-label overlapping candidates, segment smaller boxes first.
4) After each segmentation, remove its mask from the working image, and segment
   the remaining candidates on the residual image.

Input modes are the same as the classlist script:

1) Single class list applied to every image:
   python infer_remove_components_overlap_priority.py \
     --ckpt /path/to/sam3_slideforge.pt \
     --base_ckpt /path/to/sam3.pt \
     --class_list /data/.../sam3_text_types_38.json \
     --images /path/to/images_dir_or_list.json_or_single.png \
     --output_dir ./output/remove_components_overlap_priority

2) Per-image class list via manifest:
   python infer_remove_components_overlap_priority.py \
     --ckpt ... --base_ckpt ... \
     --manifest runs/<run_id>/manifest.json \
     --output_dir ./output/remove_components_overlap_priority
"""

import argparse
import heapq
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes

SAM3_ROOT = Path(__file__).parent
sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.geometry_encoders import Prompt
from tune_decoder import (
    preprocess_image_with_meta,
    build_datapoint,
    _deep_detach,
    _deep_to_float32,
    IMG_SIZE,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class Candidate:
    text_type: str
    score: float
    bbox_xyxy: list[float]
    area: float
    mask_small: torch.Tensor  # 2D tensor in decoder output space


def _safe_name(text: str) -> str:
    x = re.sub(r"[^0-9a-zA-Z_\-]+", "_", text.strip())
    return x.strip("_") or "unknown"


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _collect_images_from_input(images_arg: str) -> list[str]:
    """Accepts: single image path / directory / JSON file containing list of paths."""
    p = Path(images_arg)
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [str(p.resolve())]
    if p.is_dir():
        out = []
        for ext in IMG_EXTS:
            out.extend(sorted(p.rglob(f"*{ext}")))
        return [str(x.resolve()) for x in out]
    if p.is_file() and p.suffix.lower() == ".json":
        data = _load_json(p)
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict) and "image" in item:
                    out.append(item["image"])
            return out
    raise ValueError(f"Cannot interpret --images={images_arg} as image / dir / json list")


def _build_work_items(args) -> list[dict]:
    """Returns list of {"image": abs_path, "classes": [str, ...]}."""
    if args.manifest:
        data = _load_json(Path(args.manifest))
        items = []
        for row in data:
            img = row["image"]
            cls = [str(x).strip() for x in row["classes"] if str(x).strip()]
            items.append({"image": img, "classes": cls})
        return items

    if not args.class_list or not args.images:
        raise SystemExit("Provide either --manifest OR (--class_list AND --images).")
    class_list = _load_json(Path(args.class_list))
    if not isinstance(class_list, list):
        raise SystemExit("--class_list JSON must be a list of strings")
    class_list = [str(x).strip() for x in class_list if str(x).strip()]
    images = _collect_images_from_input(args.images)
    return [{"image": img, "classes": class_list} for img in images]


def _clamp_box(xyxy: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [x1, y1, x2, y2]


def _box_area(xyxy: list[float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(a: list[float], b: list[float]) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    return max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)


def _boxes_overlap(a: list[float], b: list[float], min_overlap_area: float) -> bool:
    return _intersection_area(a, b) >= min_overlap_area


def _box_to_mask(xyxy: list[float], height: int, width: int) -> torch.Tensor:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    bbox_mask = torch.zeros((height, width), dtype=torch.bool)
    bbox_mask[y1:y2, x1:x2] = True
    return bbox_mask


def _mask_tight_bbox_dilated(
    mask_2d: torch.Tensor,
    bbox_mask: torch.Tensor,
    dilation_px: int,
) -> list[float] | None:
    """Compute the tight axis-aligned bbox of ``mask_2d`` after a small
    dilation, hard-clipped to ``bbox_mask``.

    The dilation is a safety margin so character ascenders / descenders
    / anti-aliasing fringe are not chopped. The clip to ``bbox_mask``
    guarantees the tight bbox cannot escape the candidate's own region.

    Returns [x1, y1, x2, y2] in pixel coords, or None if the mask is
    empty after dilation + clip.
    """
    if not bool(mask_2d.any()):
        return None
    arr = mask_2d.cpu().numpy().astype(bool)
    if dilation_px > 0:
        arr = binary_dilation(arr, iterations=int(dilation_px))
    arr = arr & bbox_mask.cpu().numpy().astype(bool)
    if not arr.any():
        return None
    ys, xs = np.where(arr)
    return [
        float(int(xs.min())),
        float(int(ys.min())),
        float(int(xs.max()) + 1),
        float(int(ys.max()) + 1),
    ]


def _repair_mask_for_cleaning(
    residual_mask: torch.Tensor,
    bbox_mask: torch.Tensor,
    fill_holes: bool,
    closing_kernel: int,
    dilation_px: int,
) -> torch.Tensor:
    """Expand the predicted mask slightly before whitening so that anti-aliasing
    halos and small interior holes are cleaned up too. The expansion is hard-
    clipped back to the candidate's own bbox, so it can never reach into a
    neighbouring component.

    Order:
      1. binary_fill_holes  (fill mask-internal holes)
      2. binary_closing     (close thin cracks / dotted edges)
      3. binary_dilation    (eat anti-aliasing fringe)
      4. mask &= bbox_mask  (hard ceiling: never escape the bbox)
    """
    if not (fill_holes or closing_kernel > 0 or dilation_px > 0):
        return residual_mask
    arr = residual_mask.cpu().numpy().astype(bool)
    if fill_holes:
        arr = binary_fill_holes(arr)
    if closing_kernel > 0:
        k = int(closing_kernel)
        arr = binary_closing(arr, structure=np.ones((k, k), dtype=bool))
    if dilation_px > 0:
        arr = binary_dilation(arr, iterations=int(dilation_px))
    arr = arr & bbox_mask.cpu().numpy().astype(bool)
    return torch.from_numpy(arr.astype(np.bool_))


def _mask_to_original_coords(mask_2d: torch.Tensor, meta: dict, orig_h: int, orig_w: int) -> torch.Tensor:
    if mask_2d.ndim != 2:
        raise ValueError(f"Expected [H, W] mask, got shape={tuple(mask_2d.shape)}")
    m = mask_2d.float()[None, None]
    if m.shape[-2:] != (IMG_SIZE, IMG_SIZE):
        m = F.interpolate(m, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    new_h, new_w = int(meta["new_h"]), int(meta["new_w"])
    m = m[..., :new_h, :new_w]
    m = F.interpolate(m, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return (m[0, 0] > 0.0).cpu()


def _clamp_bbox(
    xyxy: list[float], h: int, w: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _crop_component_rgba_from_np(
    current_rgb_np: np.ndarray, mask_bool: torch.Tensor, xyxy: list[float]
) -> Image.Image | None:
    """RGBA crop where alpha = the predicted mask restricted to the bbox."""
    h, w = current_rgb_np.shape[:2]
    box = _clamp_bbox(xyxy, h, w)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop_mask = mask_bool[y1:y2, x1:x2]
    if crop_mask.numel() == 0 or int(crop_mask.sum().item()) == 0:
        return None
    crop_rgb = Image.fromarray(current_rgb_np[y1:y2, x1:x2], mode="RGB").convert("RGBA")
    alpha = (crop_mask.cpu().numpy().astype("uint8") * 255)
    alpha_img = Image.fromarray(alpha, mode="L")
    crop_rgba = crop_rgb.copy()
    crop_rgba.putalpha(alpha_img)
    return crop_rgba


def _crop_component_rgb_bbox_from_np(
    current_rgb_np: np.ndarray, xyxy: list[float]
) -> Image.Image | None:
    """Opaque RGB crop of the full bbox, using the current state of the image
    (i.e. anything already whitened by earlier components is visible as white).
    Used for lazy-bbox components: the crop stays a rectangle, but the image
    has only the predicted mask removed — not the whole bbox.
    """
    h, w = current_rgb_np.shape[:2]
    box = _clamp_bbox(xyxy, h, w)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return Image.fromarray(current_rgb_np[y1:y2, x1:x2], mode="RGB")


def _filter_same_label_overlap(
    candidates: list[Candidate],
    max_boxes_per_text: int,
    min_overlap_area: float,
) -> list[Candidate]:
    """
    Keep larger boxes for same-label overlaps.

    We sort by descending area (then score), and suppress any later candidate
    that overlaps with already-kept candidate(s) of the same label.
    """
    ordered = sorted(candidates, key=lambda c: (c.area, c.score), reverse=True)
    kept: list[Candidate] = []
    for cand in ordered:
        if any(_boxes_overlap(cand.bbox_xyxy, k.bbox_xyxy, min_overlap_area) for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= max_boxes_per_text:
            break
    return kept


def _suppress_part_of_via_agent(
    candidates: list["Candidate"],
    image_path: str,
    min_overlap_area: float,
    overlap_ratio_thresh: float,
    exchange_dir: Path,
    poll_sec: float = 0.5,
    timeout_sec: float = 3600.0,
) -> tuple[list["Candidate"], list[dict]]:
    """For each cross-label overlapping pair where A is smaller than B and
    intersection/A.area >= overlap_ratio_thresh, ask an external VLM (via
    file-exchange) whether A is a sub-part of B. If yes, suppress A.

    Returns (kept_candidates, decisions_log).
    """
    exchange_dir.mkdir(parents=True, exist_ok=True)
    n = len(candidates)
    dropped: set[int] = set()
    decisions: list[dict] = []
    for i in range(n):
        if i in dropped:
            continue
        ci = candidates[i]
        for j in range(n):
            if i == j or j in dropped:
                continue
            cj = candidates[j]
            if ci.text_type == cj.text_type:
                continue
            if not _boxes_overlap(ci.bbox_xyxy, cj.bbox_xyxy, min_overlap_area):
                continue
            # pick smaller = A, larger = B
            if ci.area < cj.area:
                A, B, a_idx = ci, cj, i
            elif cj.area < ci.area:
                A, B, a_idx = cj, ci, j
            else:
                continue  # equal area, skip
            inter = _intersection_area(A.bbox_xyxy, B.bbox_xyxy)
            denom = max(1.0, A.area)
            ratio = inter / denom
            if ratio < overlap_ratio_thresh:
                continue
            if a_idx in dropped:
                continue

            req_id = uuid.uuid4().hex[:10]
            req_path = exchange_dir / f"part_of_request_{req_id}.json"
            resp_path = exchange_dir / f"part_of_response_{req_id}.json"
            payload = {
                "request_id": req_id,
                "image_path": image_path,
                "A": {"class": A.text_type, "bbox_xyxy": A.bbox_xyxy, "area": A.area},
                "B": {"class": B.text_type, "bbox_xyxy": B.bbox_xyxy, "area": B.area},
                "overlap_ratio_over_A": ratio,
                "response_path": str(resp_path),
            }
            with req_path.open("w") as f:
                json.dump(payload, f, indent=2)
            print(f"[part_of] Wrote {req_path}, awaiting {resp_path} ...")
            start = time.time()
            while not resp_path.exists():
                if time.time() - start > timeout_sec:
                    raise TimeoutError(f"No part_of response at {resp_path} within {timeout_sec}s")
                time.sleep(poll_sec)
            with resp_path.open() as f:
                resp = json.load(f)
            is_part_of = bool(resp.get("is_part_of", False))
            reason = str(resp.get("reason", "")).strip()
            decisions.append({
                "request_id": req_id,
                "A_class": A.text_type, "A_bbox": A.bbox_xyxy,
                "B_class": B.text_type, "B_bbox": B.bbox_xyxy,
                "overlap_ratio_over_A": ratio,
                "is_part_of": is_part_of,
                "reason": reason,
            })
            if is_part_of:
                dropped.add(a_idx)
                if a_idx == i:
                    break  # A was ci; no need to keep comparing i
    kept = [c for idx, c in enumerate(candidates) if idx not in dropped]
    return kept, decisions


def _needs_mask_for_lazy(
    cand: "Candidate",
    others: list["Candidate"],
    min_overlap_area: float,
) -> bool:
    """Lazy-segment rule: candidate needs real mask iff it overlaps some
    different-label candidate AND is the smaller one in that overlap pair.
    Otherwise a bbox-only crop (intersected with the residual) is sufficient.
    """
    for o in others:
        if o is cand:
            continue
        if o.text_type == cand.text_type:
            continue
        if not _boxes_overlap(cand.bbox_xyxy, o.bbox_xyxy, min_overlap_area):
            continue
        if cand.area < o.area:
            return True
    return False


def _build_cross_label_priority_order(
    candidates: list[Candidate], min_overlap_area: float
) -> list[Candidate]:
    """
    Build segmentation order so that for different-label overlaps, smaller boxes
    are segmented first.
    """
    n = len(candidates)
    if n <= 1:
        return candidates

    edges = [set() for _ in range(n)]
    indeg = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = candidates[i], candidates[j]
            if ci.text_type == cj.text_type:
                continue
            if not _boxes_overlap(ci.bbox_xyxy, cj.bbox_xyxy, min_overlap_area):
                continue

            if ci.area < cj.area:
                src, dst = i, j
            elif cj.area < ci.area:
                src, dst = j, i
            elif ci.score > cj.score:
                src, dst = i, j
            elif cj.score > ci.score:
                src, dst = j, i
            else:
                src, dst = (i, j) if i < j else (j, i)

            if dst not in edges[src]:
                edges[src].add(dst)
                indeg[dst] += 1

    heap = []
    for idx in range(n):
        if indeg[idx] == 0:
            c = candidates[idx]
            heapq.heappush(heap, (c.area, -c.score, idx))

    ordered_indices = []
    while heap:
        _, _, idx = heapq.heappop(heap)
        ordered_indices.append(idx)
        for nxt in edges[idx]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                c = candidates[nxt]
                heapq.heappush(heap, (c.area, -c.score, nxt))

    if len(ordered_indices) != n:
        # Fallback to deterministic sort; this should rarely happen.
        ordered_indices = sorted(range(n), key=lambda i: (candidates[i].area, -candidates[i].score, i))

    return [candidates[i] for i in ordered_indices]


def _prefer_candidate(
    a: Candidate,
    b: Candidate,
    policy: str,
) -> bool:
    """Return True if candidate `a` is preferred over `b`."""
    if policy == "area_then_score":
        if a.area != b.area:
            return a.area > b.area
        if a.score != b.score:
            return a.score > b.score
        return a.text_type < b.text_type

    # Default: score_then_area
    if a.score != b.score:
        return a.score > b.score
    if a.area != b.area:
        return a.area > b.area
    return a.text_type < b.text_type


def _suppress_cross_label_near_duplicates(
    candidates: list[Candidate],
    min_overlap_area: float,
    overlap_ratio_thresh: float,
    conflict_policy: str,
) -> tuple[list[Candidate], int]:
    """
    For different-label pairs with very high overlap, keep only one candidate.

    Near-duplicate criterion:
      intersection / min(area_a, area_b) >= overlap_ratio_thresh
    """
    if overlap_ratio_thresh <= 0.0 or len(candidates) <= 1:
        return candidates, 0

    active = [True] * len(candidates)
    suppressed = 0

    changed = True
    while changed:
        changed = False
        for i in range(len(candidates)):
            if not active[i]:
                continue
            for j in range(i + 1, len(candidates)):
                if not active[j]:
                    continue

                ci = candidates[i]
                cj = candidates[j]
                if ci.text_type == cj.text_type:
                    continue

                inter = _intersection_area(ci.bbox_xyxy, cj.bbox_xyxy)
                if inter < min_overlap_area:
                    continue
                overlap_ratio = inter / max(1e-6, min(ci.area, cj.area))
                if overlap_ratio < overlap_ratio_thresh:
                    continue

                if _prefer_candidate(ci, cj, policy=conflict_policy):
                    active[j] = False
                else:
                    active[i] = False
                suppressed += 1
                changed = True
                break
            if changed:
                break

    kept = [candidates[idx] for idx, is_active in enumerate(active) if is_active]
    return kept, suppressed


def _forward_text_prompt(
    model,
    prep_tensor: torch.Tensor,
    text: str,
    device: torch.device,
):
    batch = [{
        "image": prep_tensor,
        "text": text,
        "boxes": torch.zeros(0, 4, dtype=torch.float32),
    }]
    dp = build_datapoint(batch, device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
        backbone_out = {"img_batch_all_stages": dp.img_batch}
        backbone_out.update(model.backbone.forward_image(dp.img_batch))
        backbone_out.update(model.backbone.forward_text(dp.find_text_batch, device=device))
    backbone_out = _deep_detach(backbone_out)
    backbone_out = _deep_to_float32(backbone_out)

    find_input = dp.find_inputs[0]
    geometric_prompt = Prompt(
        box_embeddings=find_input.input_boxes,
        box_mask=find_input.input_boxes_mask,
        box_labels=find_input.input_boxes_label,
    )
    out = model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_input,
        find_target=dp.find_targets[0],
        geometric_prompt=geometric_prompt.clone(),
    )
    return out


@torch.no_grad()
def run(args, model=None):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    work_items = _build_work_items(args)
    if args.max_images is not None:
        work_items = work_items[: args.max_images]
    print(f"Planned {len(work_items)} (image, class_list) work items.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "work_items.json").open("w") as f:
        json.dump(work_items, f, indent=2)

    # H8: a persistent worker (pipeline/src/sam3_worker.py) passes a pre-loaded
    # model; the CLI path (model=None) loads exactly as before.
    if model is None:
        print("Loading model...")
        model = build_sam3_image_model(
            device=str(device),
            eval_mode=True,
            checkpoint_path=args.base_ckpt,
            enable_segmentation=True,
        ).to(device)

        print(f"Loading fine-tuned checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print(f"  Checkpoint epoch={ckpt.get('epoch')} global_step={ckpt.get('global_step')}")
        model.eval()
        del ckpt

    for image_idx, item in enumerate(work_items):
        img_path = item["image"]
        text_types = item["classes"]
        if not text_types:
            print(f"[WARN] Empty class list for {img_path}, skipping.")
            continue
        try:
            orig = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skip unreadable image: {img_path}, err={e}")
            continue

        width, height = orig.size
        prep_tensor, meta = preprocess_image_with_meta(img_path)
        cleaned_np = np.array(orig, dtype=np.uint8).copy()
        removed_mask_global = torch.zeros((height, width), dtype=torch.bool)

        image_subdir_name = f"{image_idx:04d}_{Path(img_path).parent.name}_{Path(img_path).stem}"
        image_dir = output_root / image_subdir_name
        components_dir = image_dir / "components"
        components_dir.mkdir(parents=True, exist_ok=True)
        bbox_components_dir: Path | None = None
        if args.bbox_output_dir:
            bbox_components_dir = Path(args.bbox_output_dir) / image_subdir_name / "components"
            bbox_components_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "query_image_path": img_path,
            "query_image_size": {"width": width, "height": height},
            "classes_used": text_types,
            "selection_strategy": {
                "same_label_overlap": "keep_larger_box",
                "cross_label_overlap": "segment_smaller_box_first",
                "cross_label_near_duplicate_handling": "suppress_one_candidate",
                "cross_label_near_duplicate_policy": args.cross_label_conflict_policy,
                "cross_label_near_duplicate_overlap_ratio_thresh": float(args.cross_label_dedup_overlap_ratio),
                "segment_on_residual_image": True,
                "min_overlap_area": float(args.min_overlap_area),
                "min_residual_pixels": int(args.min_residual_pixels),
                "min_residual_ratio": float(args.min_residual_ratio),
                "lazy_segment": bool(args.lazy_segment),
                "lazy_segment_component_save": "bbox_crop_when_non_overlapping_else_mask_rgba",
                "removal_mode": "bbox_fill_for_seg_mode_bbox_else_mask_repair",
                "mask_repair_fill_holes": bool(args.mask_repair_fill_holes),
                "mask_repair_closing_kernel": int(args.mask_repair_closing_kernel),
                "mask_repair_dilation_px": int(args.mask_repair_dilation_px),
                "tight_bbox_dilation_px": int(args.tight_bbox_dilation_px),
                "part_of_check_enabled": bool(args.part_of_exchange_dir),
                "part_of_overlap_ratio_thresh": float(args.part_of_overlap_ratio),
                "bbox_components_saved": bool(args.bbox_output_dir),
                "bbox_components_dir": (
                    str(bbox_components_dir) if bbox_components_dir is not None else None
                ),
            },
            "components": [],
        }

        all_candidates: list[Candidate] = []
        num_raw_candidates = 0
        num_same_label_suppressed = 0
        num_cross_label_dedup_suppressed = 0
        num_skipped_tiny_residual_pixels = 0
        num_skipped_tiny_residual_ratio = 0
        num_candidates_after_same_label_filter = 0
        num_lazy_bbox_crops = 0
        num_mask_segments = 0

        # Stage 1: predict all candidates first (no segmentation yet).
        for text in text_types:
            out = _forward_text_prompt(model, prep_tensor, text, device)

            pred_boxes = out["pred_boxes"].float()[0]
            pred_logits = out["pred_logits"].float()[0, :, 0]
            pred_masks = out.get("pred_masks", None)
            if pred_masks is None:
                continue
            if pred_masks.ndim == 5 and pred_masks.shape[2] == 1:
                pred_masks = pred_masks[:, :, 0]
            if pred_masks.ndim != 4:
                print(f"[WARN] Unexpected pred_masks shape={tuple(pred_masks.shape)}, skip prompt={text}")
                continue
            pred_masks = pred_masks.float()[0]

            scores = pred_logits.sigmoid()
            pred_xyxy_norm = box_cxcywh_to_xyxy(pred_boxes)
            max_side = float(max(width, height))
            pred_xyxy = pred_xyxy_norm * max_side
            pred_xyxy[:, 0::2].clamp_(0, width)
            pred_xyxy[:, 1::2].clamp_(0, height)

            keep_indices = torch.where(scores >= args.score_thresh)[0]
            if keep_indices.numel() == 0:
                continue

            text_candidates: list[Candidate] = []
            for q_idx in keep_indices.tolist():
                box = _clamp_box(pred_xyxy[q_idx].cpu().tolist(), width, height)
                area = _box_area(box)
                if area <= 0.0:
                    continue
                text_candidates.append(
                    Candidate(
                        text_type=text,
                        score=float(scores[q_idx].item()),
                        bbox_xyxy=box,
                        area=area,
                        # Keep mask logits at half precision to reduce memory.
                        mask_small=pred_masks[q_idx].cpu().to(torch.float16),
                    )
                )

            if not text_candidates:
                continue

            num_raw_candidates += len(text_candidates)
            kept_text = _filter_same_label_overlap(
                text_candidates,
                max_boxes_per_text=args.max_boxes_per_text,
                min_overlap_area=args.min_overlap_area,
            )
            num_same_label_suppressed += max(0, len(text_candidates) - len(kept_text))
            all_candidates.extend(kept_text)

        num_candidates_after_same_label_filter = len(all_candidates)

        # Stage 2: optionally suppress near-duplicate cross-label candidates.
        all_candidates, num_cross_label_dedup_suppressed = _suppress_cross_label_near_duplicates(
            all_candidates,
            min_overlap_area=args.min_overlap_area,
            overlap_ratio_thresh=args.cross_label_dedup_overlap_ratio,
            conflict_policy=args.cross_label_conflict_policy,
        )

        # Stage 2.5: VLM-mediated "is A part of B?" suppression (optional).
        num_part_of_suppressed = 0
        part_of_decisions = []
        if args.part_of_exchange_dir:
            before_ct = len(all_candidates)
            all_candidates, part_of_decisions = _suppress_part_of_via_agent(
                all_candidates,
                image_path=img_path,
                min_overlap_area=args.min_overlap_area,
                overlap_ratio_thresh=float(args.part_of_overlap_ratio),
                exchange_dir=Path(args.part_of_exchange_dir),
            )
            num_part_of_suppressed = before_ct - len(all_candidates)

        # Stage 3: order by cross-label overlap priority (smaller first).
        ordered_candidates = _build_cross_label_priority_order(
            all_candidates, min_overlap_area=args.min_overlap_area
        )

        component_counter = 0
        for order_idx, cand in enumerate(ordered_candidates, start=1):
            bbox_mask = _box_to_mask(cand.bbox_xyxy, height, width)
            # Always compute the predicted mask. Removal from the cleaned
            # image is ALWAYS mask-based (so we don't whiten neighbouring
            # content), regardless of whether the component is saved as a
            # mask cutout or as an opaque bbox crop.
            full_mask = _mask_to_original_coords(cand.mask_small, meta, height, width)
            mask_in_bbox = full_mask & bbox_mask
            save_as_bbox_crop = args.lazy_segment and not _needs_mask_for_lazy(
                cand, ordered_candidates, args.min_overlap_area
            )
            seg_mode = "bbox" if save_as_bbox_crop else "mask"

            orig_pixels = int(mask_in_bbox.sum().item())
            if orig_pixels == 0:
                continue

            # Segment on residual image: do not reuse already removed pixels.
            residual_mask = mask_in_bbox & (~removed_mask_global)
            residual_pixels = int(residual_mask.sum().item())
            if residual_pixels == 0:
                continue
            if residual_pixels < args.min_residual_pixels:
                num_skipped_tiny_residual_pixels += 1
                continue
            residual_ratio = residual_pixels / max(1.0, float(orig_pixels))
            if residual_ratio < args.min_residual_ratio:
                num_skipped_tiny_residual_ratio += 1
                continue

            # v2.5: for `save_as_bbox_crop` candidates (lazy_segment marked
            # them safe — no cross-label neighbour overlaps their bbox),
            # always tighten the saved bbox to the dilated mask AND clean
            # the cleaned image by whitening the *entire tight bbox*. This
            # eliminates the mask-coverage residue (fragments of photos /
            # icons / text whose SAM3 mask had internal gaps the
            # `_repair_mask_for_cleaning` pipeline could not bridge).
            #
            # For `save_as_bbox_crop=False` (cross-label overlap exists):
            # the bbox could contain a neighbour, so we MUST stick with
            # mask-based cleanup. The previous behaviour is preserved.
            tight_bbox = _mask_tight_bbox_dilated(
                mask_in_bbox,
                bbox_mask,
                dilation_px=int(args.tight_bbox_dilation_px),
            )
            original_detection_bbox = [float(v) for v in cand.bbox_xyxy]
            bbox_was_tightened = False
            if (
                save_as_bbox_crop
                and tight_bbox is not None
                and bool(args.bbox_tighten)
            ):
                # Swap to the tight bbox and remember the loose detection
                # bbox for traceability.
                cand.bbox_xyxy = tight_bbox
                cand.area = (tight_bbox[2] - tight_bbox[0]) * (
                    tight_bbox[3] - tight_bbox[1]
                )
                bbox_was_tightened = True

            if save_as_bbox_crop:
                # Save the full rectangular crop at the (now tight) bbox.
                component_image = _crop_component_rgb_bbox_from_np(cleaned_np, cand.bbox_xyxy)
            else:
                # RGBA with alpha = predicted mask. bbox stays as detection.
                component_image = _crop_component_rgba_from_np(
                    cleaned_np, residual_mask, cand.bbox_xyxy
                )
            if component_image is None:
                continue

            component_counter += 1
            component_name = f"component_{component_counter:04d}_{_safe_name(cand.text_type)}.png"
            component_path = components_dir / component_name
            component_image.save(component_path)

            # Also save an opaque bbox crop (always, regardless of seg mode) into
            # bbox_components_dir so Agent F can review + OCR without mask edges.
            # Snapshot taken BEFORE removal so the component pixels are intact.
            bbox_rel: str | None = None
            if bbox_components_dir is not None:
                bbox_image = _crop_component_rgb_bbox_from_np(cleaned_np, cand.bbox_xyxy)
                if bbox_image is not None:
                    bbox_image.save(bbox_components_dir / component_name)
                    bbox_rel = f"components/{component_name}"

            # Cleanup:
            #   - bbox-mode (lazy_segment safe): whiten the entire tight
            #     bbox. By the lazy_segment contract, no cross-label
            #     neighbour overlaps this bbox, so bbox-fill cannot eat a
            #     neighbour. Clip to ~removed_mask_global so we don't
            #     accidentally re-touch already-removed pixels.
            #   - mask-mode (cross-label overlap exists): keep the v2.2
            #     mask-repair (fill_holes + close + dilate, clipped to the
            #     candidate's own bbox).
            if save_as_bbox_crop:
                tight_bbox_mask = _box_to_mask(cand.bbox_xyxy, height, width)
                cleanup_mask = tight_bbox_mask & (~removed_mask_global)
            else:
                cleanup_mask = _repair_mask_for_cleaning(
                    residual_mask,
                    bbox_mask,
                    fill_holes=bool(args.mask_repair_fill_holes),
                    closing_kernel=int(args.mask_repair_closing_kernel),
                    dilation_px=int(args.mask_repair_dilation_px),
                )
            removed_mask_global |= cleanup_mask
            cleaned_np[cleanup_mask.numpy()] = 255

            if save_as_bbox_crop:
                num_lazy_bbox_crops += 1
            else:
                num_mask_segments += 1

            component_entry = {
                "component_file": f"components/{component_name}",
                "text_type": cand.text_type,
                "score": float(cand.score),
                "bbox_xyxy": [float(v) for v in cand.bbox_xyxy],
                "bbox_area": float(cand.area),
                "segmentation_order": int(order_idx),
                "segmentation_mode": seg_mode,
                "query_image_size": {"width": width, "height": height},
                "tight_bbox_xyxy": tight_bbox,
                "bbox_was_tightened": bool(bbox_was_tightened),
            }
            if bbox_was_tightened:
                component_entry["original_bbox_xyxy"] = original_detection_bbox
            if bbox_rel is not None:
                component_entry["bbox_component_file"] = bbox_rel
            metadata["components"].append(component_entry)

        cleaned_img = Image.fromarray(cleaned_np, mode="RGB")
        cleaned_img.save(image_dir / "image_cleaned.png")

        metadata["num_raw_candidates"] = int(num_raw_candidates)
        metadata["num_candidates_after_same_label_filter"] = int(num_candidates_after_same_label_filter)
        metadata["num_candidates_after_cross_label_dedup"] = int(len(all_candidates))
        metadata["num_same_label_suppressed"] = int(num_same_label_suppressed)
        metadata["num_cross_label_dedup_suppressed"] = int(num_cross_label_dedup_suppressed)
        metadata["num_skipped_tiny_residual_pixels"] = int(num_skipped_tiny_residual_pixels)
        metadata["num_skipped_tiny_residual_ratio"] = int(num_skipped_tiny_residual_ratio)
        metadata["num_components"] = len(metadata["components"])
        metadata["num_lazy_bbox_crops"] = int(num_lazy_bbox_crops)
        metadata["num_mask_segments"] = int(num_mask_segments)
        metadata["num_part_of_suppressed"] = int(num_part_of_suppressed)
        metadata["part_of_decisions"] = part_of_decisions
        metadata["num_removed_pixels"] = int(removed_mask_global.sum().item())
        with (image_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        if (image_idx + 1) % 10 == 0 or (image_idx + 1) == len(work_items):
            print(f"Processed {image_idx + 1}/{len(work_items)} images")

    print(f"Done. Outputs saved to: {output_root}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--class_list", default=None, help="JSON list of class strings")
    p.add_argument("--images", default=None, help="Single image / dir / JSON list of paths")
    p.add_argument("--manifest", default=None, help="JSON: [{image, classes}, ...]")
    p.add_argument("--output_dir", default="output/remove_components_overlap_priority")
    p.add_argument("--score_thresh", type=float, default=0.3)
    p.add_argument(
        "--min_overlap_area",
        type=float,
        default=1.0,
        help="Minimum intersection area (pixels) to treat two boxes as overlapping.",
    )
    p.add_argument(
        "--cross_label_dedup_overlap_ratio",
        type=float,
        default=0.9,
        help=(
            "Suppress one candidate for different-label pairs when "
            "intersection/min(area_a, area_b) >= this value. "
            "Set <=0 to disable."
        ),
    )
    p.add_argument(
        "--cross_label_conflict_policy",
        choices=["score_then_area", "area_then_score"],
        default="score_then_area",
        help="Winner policy for cross-label near-duplicate suppression.",
    )
    p.add_argument(
        "--min_residual_pixels",
        type=int,
        default=32,
        help="Skip component export when residual mask has fewer pixels than this.",
    )
    p.add_argument(
        "--min_residual_ratio",
        type=float,
        default=0.1,
        help="Skip component export when residual/original-mask ratio is below this.",
    )
    p.add_argument("--max_boxes_per_text", type=int, default=5)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--lazy-segment",
        dest="lazy_segment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Only run mask segmentation for the smaller member of any cross-label "
            "overlapping bbox pair. Non-overlapping candidates (and the larger "
            "member of an overlap) use bbox-only crop against the residual image. "
            "Default: on. Pass --no-lazy-segment to disable."
        ),
    )
    p.add_argument(
        "--part_of_exchange_dir",
        default=None,
        help=(
            "If set, after cross-label near-duplicate dedup the script writes "
            "part_of_request_*.json into this directory for each smaller-in-larger "
            "cross-label overlap pair and blocks waiting for part_of_response_*.json "
            "from an external VLM (pipeline's Agent G). If the response says "
            "is_part_of=true, the smaller candidate is dropped before segmentation."
        ),
    )
    p.add_argument(
        "--part_of_overlap_ratio",
        type=float,
        default=0.3,
        help=(
            "Only ask the part-of VLM when intersection(A,B)/A.area >= this "
            "threshold (A is the smaller bbox). Lower values ask more often."
        ),
    )
    p.add_argument(
        "--mask_repair_fill_holes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before whitening, fill mask-internal holes (scipy.ndimage.binary_fill_holes). "
            "Helps clean up photographs / icons whose SAM3 mask has bright-pixel holes."
        ),
    )
    p.add_argument(
        "--mask_repair_closing_kernel",
        type=int,
        default=5,
        help=(
            "Square kernel size (px) for binary_closing applied before whitening. "
            "Closes thin cracks / dotted edges in the predicted mask. 0 disables."
        ),
    )
    p.add_argument(
        "--mask_repair_dilation_px",
        type=int,
        default=2,
        help=(
            "Dilation iterations (px, 4-connectivity) applied after fill+close, "
            "before whitening. Eats anti-aliasing fringe. 0 disables. The expanded "
            "mask is always hard-clipped back to the candidate's own bbox, so it "
            "can never reach a neighbouring component."
        ),
    )
    p.add_argument(
        "--tight_bbox_dilation_px",
        type=int,
        default=3,
        help=(
            "Dilation iterations (px) applied to the predicted mask BEFORE "
            "computing the tight bbox stored as `tight_bbox_xyxy` in metadata. "
            "Adds a small safety margin so character ascenders / descenders / "
            "anti-aliasing fringe are not chopped when the pipeline swaps "
            "bbox_xyxy <- tight_bbox_xyxy on Agent F's bbox-not-tight verdict."
        ),
    )
    p.add_argument(
        "--bbox_tighten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "v2.5 behaviour: for lazy-segment / bbox-crop components, swap the "
            "saved bbox to the (dilated) predicted-mask tight bbox at Agent E "
            "time. Pass --no-bbox_tighten to disable this swap (the saved bbox "
            "stays at SAM3's loose detection bbox, and the saved crop is the "
            "full original detection-bbox region). tight_bbox_xyxy is still "
            "computed and recorded in metadata for downstream use. Default on."
        ),
    )
    p.add_argument(
        "--bbox_output_dir",
        default=None,
        help=(
            "If set, additionally save an opaque bbox-crop (no masking) of every "
            "exported component into this directory, mirroring the per-image subdir "
            "layout of --output_dir. Used for Agent F review + OCR on a text-only "
            "component without interference from the mask boundary."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
