"""M2: SCAN-style Coverage + Non-overlap.

  R_c = |union(bboxes) ∩ content_mask| / |content_mask|
  R_o = 1 - (redundant_pixels / total_bbox_area)
        where redundant_pixels = sum_per_pixel(max(0, count - 1))
  combined = 0.9 * R_c + 0.1 * R_o    (per SCAN; arXiv:2505.14381)

Higher is better for all three.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import (
    SlideEvalSample,
    bbox_coverage_count,
    content_mask,
    load_original_rgb,
    union_bbox_mask,
)


def evaluate(sample: SlideEvalSample, bg_tolerance: int = 12) -> dict[str, Any]:
    original = load_original_rgb(sample)
    content = content_mask(original, bg_tolerance=bg_tolerance)
    union = union_bbox_mask(sample)
    counts = bbox_coverage_count(sample)

    content_px = int(content.sum())
    covered_content_px = int((content & union).sum())
    bbox_total_px = int(counts.sum())  # sum-of-bbox-areas
    bbox_union_px = int(union.sum())
    redundant_px = int(np.maximum(counts - 1, 0).sum())

    r_c = covered_content_px / content_px if content_px else 0.0
    overlap_frac = redundant_px / bbox_total_px if bbox_total_px else 0.0
    r_o = 1.0 - overlap_frac
    combined = 0.9 * r_c + 0.1 * r_o

    return {
        "coverage_ratio": r_c,
        "non_overlap_ratio": r_o,
        "overlap_fraction": overlap_frac,
        "combined_scan_score": combined,
        "content_pixels": content_px,
        "covered_content_pixels": covered_content_px,
        "bbox_union_pixels": bbox_union_px,
        "bbox_total_pixels": bbox_total_px,
        "redundant_pixels": redundant_px,
        "num_final_components": len(sample.final_components),
    }


def aggregate(slide_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not slide_results:
        return {}
    keys = ["coverage_ratio", "non_overlap_ratio", "overlap_fraction", "combined_scan_score"]
    agg: dict[str, Any] = {}
    for k in keys:
        vals = [r[k] for r in slide_results if r.get(k) is not None]
        if vals:
            agg[f"mean_{k}"] = float(np.mean(vals))
            agg[f"std_{k}"] = float(np.std(vals))
    return agg
