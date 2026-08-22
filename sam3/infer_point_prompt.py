#!/usr/bin/env python3
"""SAM3 point-prompt grounding for missed regions.

Reads a manifest of (image_path, list of point queries) and uses the
SAM3 SAM1-task interactive predictor (`SAM3InteractiveImagePredictor`)
to ground each point into a mask + bbox. Used by Agent H phase 1 when
``--missed_via_point_prompt`` is on: the VLM proposes (x, y) anchor
points instead of bboxes (which it estimates poorly), and SAM3
predicts the actual extent.

Input manifest (JSON, list of items):

    [
      {
        "image": "/abs/path/slide.png",
        "queries": [
          {"point": [x, y], "class": "photograph", "description": "...",
           "filename": "missed_0001_photograph.png"},
          ...
        ]
      },
      ...
    ]

Per-image output directory layout (mirrors the SAM3 text-prompt script
so downstream code can reuse `_find_cleaned_image` etc.):

    output_dir/<image_stem>/
        metadata.json              {components: [{component_file, text_type,
                                                  bbox_xyxy, bbox_area, ...}]}
        components/<filename>.png  RGBA mask-cut crop
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SAM3_ROOT = Path(__file__).parent
sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_tracker  # noqa: E402
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor  # noqa: E402


def _safe_stem(p: str) -> str:
    return Path(p).stem


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight bbox of a boolean (or 0/1) 2D mask. Returns None if empty."""
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _crop_rgba_from_mask(
    rgb: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]
) -> Image.Image:
    """RGB image + boolean mask + bbox → RGBA crop with alpha=255 inside mask, 0 outside."""
    x1, y1, x2, y2 = bbox
    crop_rgb = rgb[y1:y2, x1:x2]
    crop_alpha = (mask[y1:y2, x1:x2].astype(np.uint8)) * 255
    rgba = np.concatenate([crop_rgb, crop_alpha[..., None]], axis=2)
    return Image.fromarray(rgba, mode="RGBA")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Path to input manifest JSON.")
    ap.add_argument("--output_dir", required=True, help="Root output directory.")
    ap.add_argument("--base_ckpt", required=True, help="Path to sam3.pt base checkpoint.")
    ap.add_argument("--device", default=None, help="cuda / cuda:0 / cpu. Auto if omitted.")
    ap.add_argument("--multimask_output", action="store_true",
                    help="Ask SAM3 for multiple candidate masks and pick the highest-iou one.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer_point_prompt] device={device}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print("[infer_point_prompt] Building standalone SAM3 tracker + interactive predictor ...")
    # build_tracker with with_backbone=True gives the tracker its own visual
    # backbone (image encoder), which the interactive predictor needs to
    # call .set_image(rgb). build_sam3_image_model's default path builds the
    # tracker without its own backbone (it shares the main image-pipeline
    # backbone), so going through that path leaves the predictor unable to
    # encode arbitrary images standalone.
    tracker_base = build_tracker(
        apply_temporal_disambiguation=False, with_backbone=True
    )
    inst_predictor = SAM3InteractiveImagePredictor(tracker_base).to(device)

    # Load tracker + image-encoder weights from sam3.pt. The base ckpt has
    # ``tracker.*`` prefixed keys (309 of them in sam3.pt) plus
    # ``detector.*`` keys for the image pipeline; the tracker keys are what
    # we need here, with the prefix stripped to match `tracker_base`.
    print(f"[infer_point_prompt] Loading tracker weights from {args.base_ckpt}")
    ckpt = torch.load(args.base_ckpt, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    tracker_state = {
        k.replace("tracker.", ""): v for k, v in ckpt.items() if k.startswith("tracker.")
    }
    missing, unexpected = tracker_base.load_state_dict(tracker_state, strict=False)
    print(f"  tracker missing={len(missing)} unexpected={len(unexpected)}")

    # Load the detector's vision_backbone weights into the tracker's
    # backbone. In sam3.pt the visual backbone keys are
    # ``detector.backbone.vision_backbone.*`` and the tracker's analog is
    # ``backbone.visual.*``. Copy them.
    visual_state = {}
    for k, v in ckpt.items():
        if k.startswith("detector.backbone.vision_backbone."):
            visual_state[k.replace("detector.backbone.vision_backbone.", "")] = v
    if visual_state and tracker_base.backbone is not None:
        try:
            vb = tracker_base.backbone.vision_backbone
            vmissing, vunexp = vb.load_state_dict(visual_state, strict=False)
            print(
                f"  vision_backbone keys loaded: missing={len(vmissing)} "
                f"unexpected={len(vunexp)}"
            )
        except Exception as exc:
            print(f"  [warn] could not load vision backbone weights: {exc}")
    inst_predictor.eval()

    with open(args.manifest) as f:
        manifest = json.load(f)

    for item in manifest:
        img_path = item["image"]
        queries = item.get("queries") or []
        if not queries:
            continue
        stem = _safe_stem(img_path)
        out_subdir = output_root / stem
        comps_dir = out_subdir / "components"
        comps_dir.mkdir(parents=True, exist_ok=True)

        print(f"[infer_point_prompt] {img_path}  ({len(queries)} point query(s))")
        try:
            rgb_pil = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"  [warn] could not open image ({exc}); skipping.")
            continue
        rgb = np.array(rgb_pil)

        with torch.inference_mode():
            inst_predictor.set_image(rgb_pil)

            components_meta: list[dict] = []
            for qi, q in enumerate(queries, start=1):
                pt = q.get("point")
                cls = str(q.get("class") or "").strip() or "unknown"
                fname = q.get("filename") or f"missed_{qi:04d}_{re.sub(r'[^0-9A-Za-z_-]+','_',cls).strip('_') or 'unknown'}.png"
                if not pt or len(pt) != 2:
                    print(f"  [warn] q{qi} missing/bad 'point' field; skipping.")
                    continue
                try:
                    px = float(pt[0])
                    py = float(pt[1])
                except (TypeError, ValueError):
                    print(f"  [warn] q{qi} bad point coords; skipping.")
                    continue
                point_coords = np.array([[px, py]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)  # foreground
                try:
                    masks, iou_preds, _ = inst_predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=bool(args.multimask_output),
                        return_logits=False,
                        normalize_coords=True,
                    )
                except Exception as exc:
                    print(f"  [warn] q{qi} predict failed ({exc}); skipping.")
                    continue

                # `masks` is CxHxW with C∈{1,3}; pick best by IoU prediction.
                if masks.ndim == 3:
                    best_idx = int(np.argmax(iou_preds))
                    mask = masks[best_idx] > 0.0
                    iou = float(iou_preds[best_idx])
                else:
                    mask = masks > 0.0
                    iou = float(iou_preds.item() if hasattr(iou_preds, "item") else iou_preds)

                bbox = _bbox_from_mask(mask)
                if bbox is None:
                    print(f"  [warn] q{qi} predicted empty mask; skipping.")
                    continue
                rgba = _crop_rgba_from_mask(rgb, mask, bbox)
                rgba.save(str(comps_dir / fname))

                area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                components_meta.append({
                    "component_file": f"components/{fname}",
                    "text_type": cls,
                    "bbox_xyxy": [
                        float(bbox[0]), float(bbox[1]),
                        float(bbox[2]), float(bbox[3]),
                    ],
                    "bbox_area": area,
                    "prompt_point_xy": [px, py],
                    "iou_pred": iou,
                    "description": str(q.get("description", "")).strip(),
                })
                print(
                    f"  q{qi} pt=({px:.0f},{py:.0f}) class={cls!r} "
                    f"→ bbox={list(bbox)} iou={iou:.2f}"
                )

        meta = {
            "query_image_path": img_path,
            "components": components_meta,
        }
        with (out_subdir / "metadata.json").open("w") as f:
            json.dump(meta, f, indent=2)

    print("[infer_point_prompt] Done.")


if __name__ == "__main__":
    main()
