#!/usr/bin/env python3
"""Stage I of SlideForge — Deck State Graph construction.

Multi-agent perception-aligned decomposition: slide image -> slide-
conditioned component vocabulary -> SAM3 grounded segmentation -> validity
review -> layout review (missed regions + perceptual merge groups) ->
z-order assignment. The materialized ``final/`` state per slide is the
component layer of the Deck State Graph (see ``dsg/graph.py`` for the
graph interface); ``restyle/step0_ingest_dsg.py`` consumes it for
theme-preserving reconstruction.

The core loop is an Agent F review cycle that re-runs A-E on the cleaned
image while substantive components remain visible.

Inputs
------
--image     : single image path                               (mode 1)
--images    : directory OR JSON file with list of image paths (mode 2)

Backends
--------
--backend openai|gemini|claude|claude_code
--api_key  overrides env vars (OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY).
           Ignored for claude_code (uses file exchange).

Iteration
---------
For each input image:
  iter_00: A+B+C+D+E on the original image, then F reviews image_cleaned.png
  iter_01: if F says re-run, A+B+C+D+E on the previous iter's image_cleaned, then F again
  ... up to --max_review_iters iterations

Outputs
-------
run_dir/
  <image_stem>/
    iter_00/
      per_image.json
      manifest.json
      segmentation/
        0000_.../components/, image_cleaned.png, metadata.json
      exchange/     # claude_code only
      review.json   # Agent F verdict for this iter
    iter_01/ ...
    summary.json    # full history + final cleaned image path
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from decomposition.backends import build_backend  # noqa: E402
from decomposition.agents import (  # noqa: E402
    agent_a_describe,
    agent_b_map_to_classes,
    agent_c_direct_select,
    agent_d_union,
    agent_e_segment,
    agent_f_review,
    agent_f_validate_component,
    agent_h_find_missed_regions,
    agent_h_find_missed_regions_via_points,
    agent_h_merge_groups,
    agent_h_validate_missed_region,
    agent_m2_can_merge,
    agent_m3_overlap_arbitrate,
    agent_o_segmentation_layers,
    render_bbox_overlay,
)

from shapely.geometry import box as shapely_box, Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_CONDA_PY = "python"
DEFAULT_E_SCRIPT = REPO / "sam3" / "infer_remove_components_overlap_priority.py"

# H11 wave 1 (--agent_cascade): shared cheap screen backend for the per-item
# judgment agents (H-validate-missed, M2, M3). Stays None unless the flag is
# on, so every call site can pass it unconditionally.
AGENT_SCREEN_BACKEND = None


def _ensure_agent_screen_backend(args, backend_name, api_key, exchange_dir=None):
    global AGENT_SCREEN_BACKEND
    if getattr(args, "agent_cascade", False) and AGENT_SCREEN_BACKEND is None:
        AGENT_SCREEN_BACKEND = build_backend(
            backend_name, api_key=api_key, exchange_dir=exchange_dir,
            model=args.f_validity_screen_model,
        )
    return AGENT_SCREEN_BACKEND


# H11 wave 2 (--judgment_cascade): sonnet-tier screen for the judgment-heavy
# single-shot calls (F-cleanup, H-find-missed, H-merge). None unless flag on.
JUDGMENT_SCREEN_BACKEND = None


def _ensure_judgment_screen_backend(args, backend_name, api_key, exchange_dir=None):
    global JUDGMENT_SCREEN_BACKEND
    if getattr(args, "judgment_cascade", False) and JUDGMENT_SCREEN_BACKEND is None:
        JUDGMENT_SCREEN_BACKEND = build_backend(
            backend_name, api_key=api_key, exchange_dir=exchange_dir,
            model=args.judgment_screen_model,
        )
    return JUDGMENT_SCREEN_BACKEND


def _resolve_images(args) -> list[str]:
    if args.image and args.images:
        raise SystemExit("Use either --image or --images, not both.")
    if args.image:
        return [str(Path(args.image).resolve())]
    if args.images:
        p = Path(args.images)
        if p.is_dir():
            out = []
            for ext in IMG_EXTS:
                out.extend(sorted(p.rglob(f"*{ext}")))
            return [str(x.resolve()) for x in out]
        if p.is_file() and p.suffix.lower() == ".json":
            with p.open() as f:
                data = json.load(f)
            out: list[str] = []
            for item in data:
                if isinstance(item, str):
                    out.append(str(Path(item).resolve()))
                elif isinstance(item, dict) and "image" in item:
                    out.append(str(Path(item["image"]).resolve()))
            return out
    raise SystemExit("Provide --image PATH or --images DIR|JSON.")


def _e_extra_args(args) -> list[str]:
    script_name = Path(args.e_script).name
    if "overlap_priority" in script_name:
        extra = ["--min_overlap_area", str(args.min_overlap_area)]
        extra.append("--lazy-segment" if args.lazy_segment else "--no-lazy-segment")
        if args.part_of_check:
            extra += ["--part_of_overlap_ratio", str(args.part_of_overlap_ratio)]
        # Mask-repair (modification A): fill holes + close cracks + dilate
        # before whitening, hard-clipped to the candidate's own bbox.
        extra += [
            "--mask_repair_fill_holes" if args.mask_repair_fill_holes else "--no-mask_repair_fill_holes",
            "--mask_repair_closing_kernel", str(args.mask_repair_closing_kernel),
            "--mask_repair_dilation_px", str(args.mask_repair_dilation_px),
            "--tight_bbox_dilation_px", str(args.tight_bbox_dilation_px),
            "--bbox_tighten" if getattr(args, "bbox_tighten", True) else "--no-bbox_tighten",
        ]
        return extra
    return ["--nms_thresh", str(args.nms_thresh)]


def _find_cleaned_image(seg_dir: Path) -> Path | None:
    for child in seg_dir.iterdir():
        if child.is_dir():
            p = child / "image_cleaned.png"
            if p.is_file():
                return p
    return None


def _find_components_dir(cleaned_image_path: Path) -> Path:
    """The per-image seg subdir that contains ``image_cleaned.png`` also holds
    a ``components/`` dir with the per-component crops."""
    return cleaned_image_path.parent / "components"


def _find_bbox_components_dir(bbox_root: Path, cleaned_image_path: Path) -> Path:
    """Mirror of ``_find_components_dir`` but under the bbox output root.

    Segmentation layout: ``<seg_root>/<image_subdir>/components/...``.
    Bbox layout       : ``<bbox_root>/<image_subdir>/components/...``.
    """
    image_subdir = cleaned_image_path.parent.name
    return bbox_root / image_subdir / "components"


def _list_component_images(components_dir: Path) -> list[Path]:
    if not components_dir.is_dir():
        return []
    return sorted(components_dir.glob("component_*.png"))


def _extracted_classes_from_record(record: dict) -> set[str]:
    """Return lower-cased class names that produced at least one component
    crop in this iter (after F's per-component pruning).
    """
    out: set[str] = set()
    cleaned = Path(record.get("cleaned_image") or "")
    if not cleaned.parent.exists():
        return out
    meta_path = cleaned.parent / "metadata.json"
    if not meta_path.is_file():
        return out
    try:
        with meta_path.open() as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    for entry in meta.get("components") or []:
        t = str(entry.get("text_type", "")).strip().lower()
        if t:
            out.add(t)
    return out


def _retry_classes_from_prev(
    prev_record: dict,
    backend_name: str,
    api_key: str | None,
    model: str | None,
    class_list: list[str],
    iter_dir_for_backend: Path,
) -> list[str]:
    """Modification C: build the next iter's retry-class list from the
    previous iter — the union of:

    1. Classes that were proposed in prev.union_classes but produced no
       component crop (SAM3 grounded nothing above threshold).
    2. Taxonomy classes that Agent B maps prev.review.remaining_components
       text phrases onto.

    We deliberately go through Agent B (not the original A→B→C→D union) so
    that F's free-form residue descriptions ("bullet list partially intact",
    "Remote Sensing photo") are normalised onto the fixed taxonomy.
    """
    prev_per = prev_record.get("per_image") or {}
    prev_union = [str(c) for c in prev_per.get("union_classes") or []]
    extracted_lower = _extracted_classes_from_record(prev_record)
    missed = [c for c in prev_union if c.lower() not in extracted_lower]

    remaining_phrases = [
        str(x).strip()
        for x in (prev_record.get("review") or {}).get("remaining_components") or []
        if str(x).strip()
    ]
    mapped_remaining: list[str] = []
    if remaining_phrases:
        from decomposition.backends import build_backend  # local to avoid cycle
        try:
            backend = build_backend(
                backend_name,
                api_key=api_key,
                exchange_dir=str(iter_dir_for_backend / "exchange")
                if backend_name == "claude_code"
                else None,
                model=model,
            )
            mapped_remaining = agent_b_map_to_classes(
                backend, remaining_phrases, class_list
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] retry-class B-mapping failed: {exc}")
            mapped_remaining = []

    seen: set[str] = set()
    out: list[str] = []
    for c in [*missed, *mapped_remaining]:
        cl = c.lower()
        if cl in seen:
            continue
        seen.add(cl)
        out.append(c)
    return out


def _bbox_intersection_area(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _cross_iter_dedup(iter_records: list[dict], overlap_ratio_thresh: float) -> dict:
    """Drop any later-iter component whose bbox is largely contained in an
    earlier-iter kept component's bbox **of the same class**.

    Overlap metric: ``intersection / area_of_later_component``. If this is
    >= threshold AND the two components share the same ``text_type``
    (case-insensitive), the later component is treated as a near-duplicate.

    The class guard is what stops a retry-iter detection of one class
    (e.g. ``bullet list``) from being deleted because it spatially overlaps
    an earlier-iter detection of a different class (e.g. ``image
    collection``) that happens to cover the same region of the slide.
    Different classes in the same area = different components, even when
    their bboxes overlap heavily.

    Deletes the PNG and updates the later iter's ``metadata.json`` in place.
    Returns a summary dict listing every removal.
    """
    removed_entries: list[dict] = []
    if overlap_ratio_thresh <= 0 or len(iter_records) < 2:
        return {"overlap_ratio_thresh": overlap_ratio_thresh, "removed": []}

    iter_meta: list[tuple[Path, Path, Path, dict]] = []  # (metadata_path, components_dir, bbox_components_dir, meta)
    for rec in iter_records:
        cleaned = Path(rec["cleaned_image"])
        meta_path = cleaned.parent / "metadata.json"
        comp_dir = cleaned.parent / "components"
        iter_dir = Path(rec["iter_dir"])
        bbox_comp_dir = iter_dir / "bbox_components" / cleaned.parent.name / "components"
        if not meta_path.is_file():
            iter_meta.append((meta_path, comp_dir, bbox_comp_dir, {"components": []}))
            continue
        try:
            with meta_path.open() as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {"components": []}
        iter_meta.append((meta_path, comp_dir, bbox_comp_dir, meta))

    kept_prior: list[tuple[int, str, str, list[float]]] = []  # (iter_idx, filename, text_type_lower, bbox)
    for iter_idx, (meta_path, comp_dir, bbox_comp_dir, meta) in enumerate(iter_meta):
        components = meta.get("components") or []
        kept = []
        for entry in components:
            bbox = entry.get("bbox_xyxy")
            cf = str(entry.get("component_file", ""))
            filename = Path(cf).name
            text_type = str(entry.get("text_type", "")).strip().lower()
            if not bbox or not filename:
                kept.append(entry)
                continue
            area_later = max(1e-6, float(entry.get("bbox_area") or 0.0) or (
                (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1]))
            ))
            best_overlap = 0.0
            best_match: tuple[int, str, str] | None = None
            for prior_iter_idx, prior_name, prior_class, prior_bbox in kept_prior:
                # Class guard: only treat as duplicate when both sides agree
                # on the class. Different-class spatial overlap is allowed.
                if not text_type or prior_class != text_type:
                    continue
                inter = _bbox_intersection_area(bbox, prior_bbox)
                if inter <= 0:
                    continue
                ratio = inter / area_later
                if ratio > best_overlap:
                    best_overlap = ratio
                    best_match = (prior_iter_idx, prior_name, prior_class)
            if best_match is not None and best_overlap >= overlap_ratio_thresh:
                # Drop this later component from BOTH segmentation + bbox stores.
                p = comp_dir / filename
                if p.is_file():
                    try:
                        p.unlink()
                    except OSError:
                        pass
                bp = bbox_comp_dir / filename
                if bp.is_file():
                    try:
                        bp.unlink()
                    except OSError:
                        pass
                removed_entries.append({
                    "iter_idx": iter_idx,
                    "filename": filename,
                    "text_type": text_type,
                    "bbox_xyxy": [float(v) for v in bbox],
                    "overlap_ratio": float(best_overlap),
                    "matched_earlier": {
                        "iter_idx": best_match[0],
                        "filename": best_match[1],
                        "text_type": best_match[2],
                    },
                })
                continue
            kept.append(entry)
            kept_prior.append((iter_idx, filename, text_type, [float(v) for v in bbox]))
        meta["components"] = kept
        if removed_entries:
            with meta_path.open("w") as f:
                json.dump(meta, f, indent=2)
    return {
        "overlap_ratio_thresh": overlap_ratio_thresh,
        "removed": removed_entries,
    }


def _pixel_tight_bbox_within(
    orig_img_np: np.ndarray,
    bbox_xyxy: list[float],
    threshold: int = 235,
    pad_px: int = 2,
) -> list[float] | None:
    """Tight bbox of non-white pixels inside ``bbox_xyxy`` of
    ``orig_img_np``. Used as a fallback when SAM3's mask-derived
    ``tight_bbox_xyxy`` is not actually tight to the visible content
    (common for "region" classes like ``bullet list`` where the mask
    covers the whole region including empty margins).

    A pixel is considered "non-white" when at least one of its RGB
    channels is below ``threshold``. Returns axis-aligned bbox in
    ORIGINAL slide coords with a small ``pad_px`` safety margin, clamped
    to image bounds. Returns None if no non-white pixel is found.
    """
    h, w = orig_img_np.shape[:2]
    x1 = max(0, min(w, int(round(float(bbox_xyxy[0])))))
    y1 = max(0, min(h, int(round(float(bbox_xyxy[1])))))
    x2 = max(0, min(w, int(round(float(bbox_xyxy[2])))))
    y2 = max(0, min(h, int(round(float(bbox_xyxy[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = orig_img_np[y1:y2, x1:x2]
    if crop.ndim == 3:
        non_white = (crop[..., :3] < threshold).any(axis=-1)
    else:
        non_white = crop < threshold
    if not non_white.any():
        return None
    ys, xs = np.where(non_white)
    return [
        float(max(0, x1 + int(xs.min()) - pad_px)),
        float(max(0, y1 + int(ys.min()) - pad_px)),
        float(min(w, x1 + int(xs.max()) + 1 + pad_px)),
        float(min(h, y1 + int(ys.max()) + 1 + pad_px)),
    ]


def _build_overlap_arbitrate_overlay(
    original_image_path: str,
    bbox_a: list[float],
    bbox_b: list[float],
    out_path: Path,
) -> Path:
    """Render an overlay for Agent M3: original slide with A's bbox in
    red, B's bbox in blue, and the overlap region shaded yellow.
    """
    img = Image.open(original_image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    a = [float(v) for v in bbox_a]
    b = [float(v) for v in bbox_b]
    ox1 = max(a[0], b[0])
    oy1 = max(a[1], b[1])
    ox2 = min(a[2], b[2])
    oy2 = min(a[3], b[3])
    if ox2 > ox1 and oy2 > oy1:
        draw.rectangle([ox1, oy1, ox2, oy2], fill=(255, 220, 0, 90))
    draw.rectangle([a[0], a[1], a[2], a[3]], outline=(220, 30, 30, 255), width=5)
    draw.rectangle([b[0], b[1], b[2], b[3]], outline=(30, 80, 220, 255), width=5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    return out_path


def _apply_mask_overlap_carve(
    components_dir: Path,
    bbox_components_dir: Path | None,
    metadata_path: Path,
    original_image_path: str,
    backend=None,
    overlay_dir: Path | None = None,
    min_overlap_pixels: float = 100.0,
    # Legacy/unused — kept for backwards compat in case external callers
    # pass it; the carve now consults Agent M3 (VLM) regardless.
    mask_coverage_threshold: float = 0.05,
) -> dict:
    """Per-iter overlap carve (v2.9): Agent-M3-arbitrated.

    For every pair of fine components in this iter whose bboxes overlap
    by more than ``min_overlap_pixels``, ask Agent M3 (VLM) which side
    actually owns the overlap region:

      - If A's bbox lies (M3 says ``a_owns_overlap=False`` while
        ``b_owns_overlap=True``), A is carved: A.polygon = bbox(A) -
        bbox(B), bbox shrunk to polygon's bounds, bbox_components/ crop
        regenerated as RGBA polygon-cut.
      - Similarly for B.
      - TRUE/TRUE → no carve (genuine co-located content).
      - FALSE/FALSE → no carve (ambiguous; safer to leave bboxes as-is).

    Replaces the v2.7 mask-coverage heuristic which often mis-judged
    when SAM3 mask was loose / missing. The VLM has class context AND
    visual evidence so it makes more reliable decisions.

    The carve runs BEFORE Agent F validity, so Agent F sees the carved
    crop and can correctly identify text-only components and OCR them.

    Returns ``{"carved": [{"filename","carved_by","reason"}, ...]}``.
    """
    if not metadata_path.is_file() or bbox_components_dir is None:
        return {"carved": []}
    try:
        with metadata_path.open() as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"carved": []}
    components = meta.get("components") or []
    if len(components) < 2:
        return {"carved": []}

    items: list[dict] = []
    for entry in components:
        bb = entry.get("bbox_xyxy")
        if not bb or len(bb) != 4:
            continue
        cf = Path(str(entry.get("component_file", ""))).name
        bcf = Path(str(entry.get("bbox_component_file", entry.get("component_file", "")))).name
        items.append({
            "entry": entry,
            "bbox": [float(v) for v in bb],
            "seg_mode": str(entry.get("segmentation_mode", "bbox")),
            "seg_filename": cf,
            "bbox_filename": bcf,
        })
    if len(items) < 2:
        return {"carved": []}

    carved: list[dict] = []
    if backend is None:
        return {"carved": []}
    if overlay_dir is None:
        overlay_dir = metadata_path.parent / "overlap_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Track which (i, j) pairs have already been queried in this pass so
    # we don't ask Agent M3 about the same pair twice when iterating both
    # i→j and j→i. The carve action is asymmetric (A loses or B loses),
    # so we capture both decisions from a single query and apply each
    # time we see the relevant ordering.
    asked: dict[tuple[str, str], dict] = {}
    overlay_idx = 0
    n = len(items)
    for i in range(n):
        a = items[i]
        for j in range(n):
            if i == j:
                continue
            b = items[j]
            bbA = a["bbox"]
            bbB = b["bbox"]
            ox1 = max(bbA[0], bbB[0])
            oy1 = max(bbA[1], bbB[1])
            ox2 = min(bbA[2], bbB[2])
            oy2 = min(bbA[3], bbB[3])
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            overlap_area = (ox2 - ox1) * (oy2 - oy1)
            if overlap_area < min_overlap_pixels:
                continue

            # Look up (or query) the M3 decision for this unordered pair.
            pair_key = tuple(sorted([a["seg_filename"], b["seg_filename"]]))
            decision = asked.get(pair_key)
            if decision is None:
                # Use the canonical orientation (A=lower filename) for the
                # query; we'll reinterpret when applying.
                first_fn, second_fn = pair_key
                if first_fn == a["seg_filename"]:
                    qa, qb = a, b
                else:
                    qa, qb = b, a
                overlay_idx += 1
                overlay_path = overlay_dir / f"m3_overlay_{overlay_idx:04d}.png"
                _build_overlap_arbitrate_overlay(
                    original_image_path=original_image_path,
                    bbox_a=qa["bbox"],
                    bbox_b=qb["bbox"],
                    out_path=overlay_path,
                )
                a_class = str(qa["entry"].get("text_type", ""))
                b_class = str(qb["entry"].get("text_type", ""))
                try:
                    raw = agent_m3_overlap_arbitrate(
                        backend=backend,
                        screen_backend=AGENT_SCREEN_BACKEND,
                        original_image_path=original_image_path,
                        overlay_image_path=str(overlay_path),
                        class_a=a_class,
                        bbox_a=qa["bbox"],
                        class_b=b_class,
                        bbox_b=qb["bbox"],
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  [warn] agent_m3 failed for ({qa['seg_filename']} "
                        f"vs {qb['seg_filename']}): {exc}"
                    )
                    raw = {"a_owns_overlap": True, "b_owns_overlap": True, "reason": str(exc)}
                decision = {
                    "first_owns": bool(raw.get("a_owns_overlap", True)),
                    "second_owns": bool(raw.get("b_owns_overlap", True)),
                    "reason": raw.get("reason", ""),
                }
                asked[pair_key] = decision
                print(
                    f"  [m3] {first_fn} vs {second_fn}: "
                    f"first_owns={decision['first_owns']} "
                    f"second_owns={decision['second_owns']} "
                    f"({decision['reason'][:80]})"
                )
            # Translate decision back to (a, b) orientation in this loop.
            first_fn, _ = pair_key
            if first_fn == a["seg_filename"]:
                a_owns = decision["first_owns"]
                b_owns = decision["second_owns"]
            else:
                a_owns = decision["second_owns"]
                b_owns = decision["first_owns"]
            # Carve A only when A doesn't own the overlap and B does.
            if a_owns or not b_owns:
                continue

            # Carve A's bbox by B's bbox.
            geom_A = shapely_box(*bbA)
            geom_B = shapely_box(*bbB)
            carved_geom = geom_A.difference(geom_B)
            if carved_geom.is_empty:
                continue
            if isinstance(carved_geom, ShapelyMultiPolygon):
                # Pick the largest piece — A might split into disconnected
                # rectangles when B sits in the middle of A.
                carved_geom = max(carved_geom.geoms, key=lambda p: p.area)
            if not isinstance(carved_geom, ShapelyPolygon):
                continue
            if list(carved_geom.interiors):
                continue  # would create a hole — skip
            new_bb = list(carved_geom.bounds)
            if (new_bb[2] - new_bb[0]) < 1 or (new_bb[3] - new_bb[1]) < 1:
                continue
            vertices = _shapely_to_vertices(carved_geom)

            # Regenerate the bbox_components/ crop using the new bbox /
            # polygon. RGB if rectangular, RGBA polygon-alpha if not.
            try:
                img = Image.open(original_image_path).convert("RGB")
                W, H = img.size
                nb = [
                    max(0, min(float(W), new_bb[0])),
                    max(0, min(float(H), new_bb[1])),
                    max(0, min(float(W), new_bb[2])),
                    max(0, min(float(H), new_bb[3])),
                ]
                if nb[2] <= nb[0] or nb[3] <= nb[1]:
                    continue
                rect_crop = img.crop((int(nb[0]), int(nb[1]), int(nb[2]), int(nb[3])))
                bp = bbox_components_dir / a["bbox_filename"]
                if vertices:
                    cw, ch = rect_crop.size
                    alpha_mask = Image.new("L", (cw, ch), 0)
                    poly_local = [
                        (
                            max(0.0, min(float(cw), float(px) - nb[0])),
                            max(0.0, min(float(ch), float(py) - nb[1])),
                        )
                        for px, py in vertices
                    ]
                    ImageDraw.Draw(alpha_mask).polygon(poly_local, fill=255)
                    rgba_out = rect_crop.convert("RGBA")
                    rgba_out.putalpha(alpha_mask)
                    rgba_out.save(bp)
                else:
                    rect_crop.save(bp)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] could not regenerate carved crop {a['bbox_filename']}: {exc}")
                continue

            # Update the in-memory metadata entry.
            entry = a["entry"]
            entry.setdefault("original_bbox_xyxy", list(bbA))
            entry["bbox_xyxy"] = [float(v) for v in nb]
            entry["bbox_area"] = float((nb[2] - nb[0]) * (nb[3] - nb[1]))
            if vertices:
                entry["polygon_xyxy"] = vertices
            else:
                entry.pop("polygon_xyxy", None)
            entry["bbox_overlap_carved_by"] = b["bbox_filename"]
            # Clamp tight_bbox_xyxy so a downstream `_apply_bbox_tightening`
            # call cannot expand the bbox back into the carved-out region.
            tb = entry.get("tight_bbox_xyxy")
            if isinstance(tb, list) and len(tb) == 4:
                clamped = [
                    max(float(tb[0]), nb[0]),
                    max(float(tb[1]), nb[1]),
                    min(float(tb[2]), nb[2]),
                    min(float(tb[3]), nb[3]),
                ]
                if clamped[2] > clamped[0] and clamped[3] > clamped[1]:
                    entry["tight_bbox_xyxy"] = clamped
                else:
                    entry["tight_bbox_xyxy"] = list(nb)
            else:
                entry["tight_bbox_xyxy"] = list(nb)

            # Update local cache so subsequent pairs see the new geometry.
            a["bbox"] = list(nb)
            carved.append({
                "filename": a["bbox_filename"],
                "carved_by": b["bbox_filename"],
                "reason": str(decision.get("reason", ""))[:160],
            })
            # Re-evaluate this A against remaining Bs with the updated
            # bbox by breaking and restarting the inner sweep is
            # over-engineering — one carve per pass is sufficient for the
            # common case (chained overlaps are rare).
            break

    if carved:
        with metadata_path.open("w") as f:
            json.dump(meta, f, indent=2)
    return {"carved": carved}


def _apply_bbox_tightening(
    components_dir: Path,
    bbox_components_dir: Path | None,
    metadata_path: Path,
    original_image_path: str,
    component_verdicts: list[dict],
) -> dict:
    """For text-only components that Agent F flagged as ``bbox_is_tight=false``,
    swap ``bbox_xyxy`` to a tighter rectangle and re-crop the saved
    component PNGs.

    Two-stage tightening strategy (since SAM3 mask is sometimes loose for
    "region" classes like ``bullet list``):

    1. Try the mask-derived ``tight_bbox_xyxy`` first (already computed
       by Agent E from the dilated predicted mask). If it is meaningfully
       smaller than the current bbox, use it.
    2. If the mask tight bbox is essentially the current bbox (mask was
       loose), fall back to PIXEL-LEVEL tightening: scan non-white pixels
       inside the current bbox of the ORIGINAL slide and use their
       bounding box. This catches cases where SAM3's mask covers the whole
       semantic region including empty margins.

    - Mask-cut RGBA components (segmentation_mode='mask') have their alpha
      derived from the original predicted mask in the original bbox; we
      simply re-crop the cleaned image at the tight bbox and rebuild the
      RGBA with the mask intersected against the tight bbox region.
    - Bbox-rect components (segmentation_mode='bbox') get a clean opaque
      crop at the tight bbox.
    - The opaque ``bbox_components`` mirror is rewritten the same way.
    - metadata.json is updated: ``bbox_xyxy`` <- tight; new field
      ``bbox_was_tightened: true`` records that the swap happened, and
      ``original_bbox_xyxy`` preserves the original detection bbox.

    Returns a dict listing every tightened filename.
    """
    if not metadata_path.is_file():
        return {"tightened": []}
    try:
        with metadata_path.open() as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"tightened": []}
    components = meta.get("components") or []
    by_filename = {Path(str(e.get("component_file", ""))).name: e for e in components}
    verdict_by_filename = {v.get("filename"): v for v in component_verdicts}

    tightened: list[dict] = []
    try:
        orig_img = Image.open(original_image_path).convert("RGB")
        orig_arr = np.array(orig_img)
    except Exception:  # noqa: BLE001
        return {"tightened": []}

    def _bbox_essentially_same(a: list[float], b: list[float], tol: float = 1.0) -> bool:
        return (
            len(a) == 4
            and len(b) == 4
            and abs(float(a[0]) - float(b[0])) < tol
            and abs(float(a[1]) - float(b[1])) < tol
            and abs(float(a[2]) - float(b[2])) < tol
            and abs(float(a[3]) - float(b[3])) < tol
        )

    for fname, entry in by_filename.items():
        verdict = verdict_by_filename.get(fname) or {}
        if not verdict.get("is_text_only"):
            continue
        if not verdict.get("valid", True):
            continue
        if verdict.get("bbox_is_tight", True):
            continue
        current_bbox = list(entry.get("bbox_xyxy") or [])
        if len(current_bbox) != 4:
            continue
        w, h = orig_img.size

        # Stage 1: try the mask-derived tight bbox first.
        candidate: list[float] | None = None
        candidate_source = None
        mask_tight = entry.get("tight_bbox_xyxy")
        if isinstance(mask_tight, list) and len(mask_tight) == 4:
            mt = [
                max(0.0, min(float(w), float(mask_tight[0]))),
                max(0.0, min(float(h), float(mask_tight[1]))),
                max(0.0, min(float(w), float(mask_tight[2]))),
                max(0.0, min(float(h), float(mask_tight[3]))),
            ]
            if mt[2] > mt[0] and mt[3] > mt[1] and not _bbox_essentially_same(mt, current_bbox):
                candidate = mt
                candidate_source = "mask"

        # Stage 2: pixel-level fallback when mask tight ≈ current bbox.
        # SAM3's mask sometimes covers a "region" wider than the visible
        # text content (e.g. bullet_list whose mask spans the full
        # column width). Pixel-thresholding catches the truly-tight
        # bbox of the rendered glyphs.
        #
        # IMPORTANT: scan the SAVED bbox_components crop (which is the
        # iter's cleaned-image view at the current bbox, with prior
        # components already whitened), NOT the original slide. The
        # original may contain unrelated content at the bbox region —
        # e.g. iter_01's bullet_list bbox spatially overlaps iter_00's
        # photograph_grid, so original pixels there are photo content,
        # not bullets.
        if candidate is None and bbox_components_dir is not None:
            crop_path = bbox_components_dir / fname
            if crop_path.is_file():
                try:
                    saved_arr = np.array(Image.open(crop_path).convert("RGB"))
                except Exception:  # noqa: BLE001
                    saved_arr = None
                if saved_arr is not None and saved_arr.size > 0:
                    sh, sw = saved_arr.shape[:2]
                    non_white = (saved_arr[..., :3] < 235).any(axis=-1)
                    if non_white.any():
                        ys, xs = np.where(non_white)
                        pad = 2
                        local_x1 = max(0, int(xs.min()) - pad)
                        local_y1 = max(0, int(ys.min()) - pad)
                        local_x2 = min(sw, int(xs.max()) + 1 + pad)
                        local_y2 = min(sh, int(ys.max()) + 1 + pad)
                        # Convert local crop coords back to absolute slide coords.
                        cx1, cy1 = float(current_bbox[0]), float(current_bbox[1])
                        # Note: saved crop uses pixel rounding so the absolute
                        # mapping has up to 1 px error, that's fine.
                        pixel_tight = [
                            cx1 + float(local_x1),
                            cy1 + float(local_y1),
                            cx1 + float(local_x2),
                            cy1 + float(local_y2),
                        ]
                        if not _bbox_essentially_same(pixel_tight, current_bbox, tol=2.0):
                            candidate = pixel_tight
                            candidate_source = "pixel"

        if candidate is None:
            continue

        x1, y1, x2, y2 = candidate
        if x2 <= x1 or y2 <= y1:
            continue

        # Re-crop the original slide at the tight bbox. We use the ORIGINAL
        # slide pixels (not the cleaned image) because the cleaned image has
        # parts whitened by earlier components within the same iter and that
        # is undesirable for the saved component crop.
        crop = orig_img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Overwrite mask-cut version with the same opaque crop. Mask-cutting
        # a tightened bbox requires the original predicted mask which we no
        # longer have at this stage; opaque RGB at the tight bbox is the
        # right answer for text-only components anyway (the mask alpha was
        # only meaningful when the rectangle had non-text content around it).
        try:
            (components_dir / fname).parent.mkdir(parents=True, exist_ok=True)
            crop.save(components_dir / fname)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] tightening: could not rewrite {fname}: {exc}")
            continue
        if bbox_components_dir is not None:
            try:
                bbox_components_dir.mkdir(parents=True, exist_ok=True)
                crop.save(bbox_components_dir / fname)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] tightening: could not rewrite bbox {fname}: {exc}")

        new_area = (x2 - x1) * (y2 - y1)
        # Preserve the EARLIEST original detection bbox if Agent E already
        # tightened once (don't overwrite it with the now-loose mask bbox).
        if not entry.get("original_bbox_xyxy"):
            entry["original_bbox_xyxy"] = [float(v) for v in current_bbox]
        entry["bbox_xyxy"] = [x1, y1, x2, y2]
        entry["bbox_area"] = float(new_area)
        entry["bbox_was_tightened"] = True
        entry["bbox_tightening_source"] = candidate_source
        # After tightening, the saved crop is opaque — record the change.
        entry["segmentation_mode"] = "bbox"
        tightened.append({
            "filename": fname,
            "tightening_source": candidate_source,
            "previous_bbox_xyxy": [float(v) for v in current_bbox],
            "tight_bbox_xyxy": [x1, y1, x2, y2],
        })

    if tightened:
        with metadata_path.open("w") as f:
            json.dump(meta, f, indent=2)
    return {"tightened": tightened}


def _apply_bbox_expansion(
    components_dir: Path,
    bbox_components_dir: Path | None,
    metadata_path: Path,
    original_image_path: str,
    component_verdicts: list[dict],
    expand_pad_px: int = 30,
) -> dict:
    """For text-only components that Agent F flagged as ``bbox_is_too_small=true``,
    expand ``bbox_xyxy`` outward to capture clipped text glyphs and re-crop the
    saved component PNGs.

    This mirrors the pixel-level tightening logic: it looks at the ORIGINAL slide
    in a padded region around the current bbox, finds non-white pixels, and expands
    the bounding box to encompass them, ensuring we never shrink here.
    """
    if not metadata_path.is_file():
        return {"expanded": []}
    try:
        with metadata_path.open() as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"expanded": []}
    
    components = meta.get("components") or []
    by_filename = {Path(str(e.get("component_file", ""))).name: e for e in components}
    verdict_by_filename = {v.get("filename"): v for v in component_verdicts}

    expanded: list[dict] = []
    try:
        orig_img = Image.open(original_image_path).convert("RGB")
        w, h = orig_img.size
    except Exception:  # noqa: BLE001
        return {"expanded": []}

    for fname, entry in by_filename.items():
        verdict = verdict_by_filename.get(fname) or {}
        if not verdict.get("is_text_only"):
            continue
        if not verdict.get("valid", True):
            continue
        # Agent F must explicitly flag the bbox as too small
        if not verdict.get("bbox_is_too_small", False):
            continue
            
        current_bbox = list(entry.get("bbox_xyxy") or [])
        if len(current_bbox) != 4:
            continue

        cx1, cy1, cx2, cy2 = [float(v) for v in current_bbox]
        
        # Step 1: Create a padded search region clamped to image bounds
        px1 = max(0.0, cx1 - expand_pad_px)
        py1 = max(0.0, cy1 - expand_pad_px)
        px2 = min(float(w), cx2 + expand_pad_px)
        py2 = min(float(h), cy2 + expand_pad_px)
        
        if px2 <= px1 or py2 <= py1:
            continue

        # Step 2: Pixel-level scan on the ORIGINAL image
        crop = orig_img.crop((int(px1), int(py1), int(px2), int(py2)))
        crop_arr = np.array(crop)
        
        non_white = (crop_arr[..., :3] < 235).any(axis=-1)
        if not non_white.any():
            continue

        ys, xs = np.where(non_white)
        safe_pad = 5  # Internal padding for anti-aliased edges
        
        # Calculate new local bounds within the crop
        local_x1 = max(0, int(xs.min()) - safe_pad)
        local_y1 = max(0, int(ys.min()) - safe_pad)
        local_x2 = min(crop_arr.shape[1], int(xs.max()) + 1 + safe_pad)
        local_y2 = min(crop_arr.shape[0], int(ys.max()) + 1 + safe_pad)

        # Step 3: Convert back to global coordinates and merge
        global_x1 = px1 + float(local_x1)
        global_y1 = py1 + float(local_y1)
        global_x2 = px1 + float(local_x2)
        global_y2 = py1 + float(local_y2)

        # Merge strategy: Only expand, never shrink. 
        final_x1 = min(cx1, global_x1)
        final_y1 = min(cy1, global_y1)
        final_x2 = max(cx2, global_x2)
        final_y2 = max(cy2, global_y2)

        # If the expansion is trivial (< 1px), skip saving
        if (cx1 - final_x1 < 1.0) and (cy1 - final_y1 < 1.0) and \
           (final_x2 - cx2 < 1.0) and (final_y2 - cy2 < 1.0):
            continue

        # Step 4: Re-crop and save
        new_crop = orig_img.crop((int(final_x1), int(final_y1), int(final_x2), int(final_y2)))
        try:
            (components_dir / fname).parent.mkdir(parents=True, exist_ok=True)
            new_crop.save(components_dir / fname)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] expansion: could not rewrite {fname}: {exc}")
            continue
            
        if bbox_components_dir is not None:
            try:
                bbox_components_dir.mkdir(parents=True, exist_ok=True)
                new_crop.save(bbox_components_dir / fname)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] expansion: could not rewrite bbox {fname}: {exc}")

        # Step 5: Update Metadata
        new_area = (final_x2 - final_x1) * (final_y2 - final_y1)
        if not entry.get("original_bbox_xyxy"):
            entry["original_bbox_xyxy"] = [float(v) for v in current_bbox]
            
        entry["bbox_xyxy"] = [final_x1, final_y1, final_x2, final_y2]
        entry["bbox_area"] = float(new_area)
        entry["bbox_was_expanded"] = True
        entry["segmentation_mode"] = "bbox"

        expanded.append({
            "filename": fname,
            "previous_bbox_xyxy": [float(v) for v in current_bbox],
            "expanded_bbox_xyxy": [final_x1, final_y1, final_x2, final_y2],
        })

    if expanded:
        with metadata_path.open("w") as f:
            json.dump(meta, f, indent=2)
            
    return {"expanded": expanded}


def _prune_invalid_components(
    components_dir: Path,
    metadata_path: Path,
    invalid_basenames: list[str],
    bbox_components_dir: Path | None = None,
) -> dict:
    """Delete invalid component PNGs from BOTH the mask-cut
    ``components_dir`` AND (if provided) the ``bbox_components_dir``, and drop
    matching entries from ``metadata.json``. Returns a dict describing what
    was pruned.
    """
    removed_seg: list[str] = []
    removed_bbox: list[str] = []
    # Instrumentation (zero behavior change): archive judged-invalid crops before
    # deletion so model-cascade calibration (H3) has real negatives to replay.
    archive_dir = components_dir.parent / "pruned_crops"
    for name in invalid_basenames:
        p = components_dir / name
        if p.is_file():
            try:
                try:
                    archive_dir.mkdir(exist_ok=True)
                    shutil.copy2(p, archive_dir / f"seg_{name}")
                except OSError:
                    pass
                p.unlink()
                removed_seg.append(name)
            except OSError:
                pass
        if bbox_components_dir is not None:
            bp = bbox_components_dir / name
            if bp.is_file():
                try:
                    try:
                        archive_dir.mkdir(exist_ok=True)
                        shutil.copy2(bp, archive_dir / f"bbox_{name}")
                    except OSError:
                        pass
                    bp.unlink()
                    removed_bbox.append(name)
                except OSError:
                    pass
    removed_meta: list[str] = []
    if metadata_path.is_file() and invalid_basenames:
        try:
            with metadata_path.open() as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            meta = None
        if isinstance(meta, dict) and isinstance(meta.get("components"), list):
            bad = set(invalid_basenames)
            kept = []
            for entry in meta["components"]:
                cf = str(entry.get("component_file", ""))
                if Path(cf).name in bad:
                    removed_meta.append(Path(cf).name)
                else:
                    kept.append(entry)
            meta["components"] = kept
            with metadata_path.open("w") as f:
                json.dump(meta, f, indent=2)
    return {
        "removed_files": removed_seg,
        "removed_bbox_files": removed_bbox,
        "removed_metadata_entries": removed_meta,
    }


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(s: str) -> str:
    return _SAFE_NAME_RE.sub("_", s.strip()).strip("_") or "component"


def _collect_valid_components(
    iter_records: list[dict],
) -> list[dict]:
    """Walk every iter's metadata.json (post cross-iter-dedup) + review.json
    (post per-component validity pruning) and return one entry per surviving
    fine-grained component, with OCR / verdict fields merged in.

    Each returned entry has:
        iter_idx, original_filename, final_filename, text_type, score,
        bbox_xyxy, bbox_area, segmentation_mode, is_text_only, ocr_latex,
        valid_reason, seg_path (mask-cut RGBA on disk), bbox_path (opaque
        RGB on disk).
    """
    out: list[dict] = []
    for iter_idx, rec in enumerate(iter_records):
        cleaned = Path(rec.get("cleaned_image") or "")
        if not cleaned.parent.exists():
            continue
        meta_path = cleaned.parent / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            with meta_path.open() as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        review = rec.get("review") or {}
        invalid = set(review.get("invalid_components") or [])
        verdict_map = {
            v["filename"]: v for v in (review.get("component_verdicts") or [])
        }
        seg_dir = cleaned.parent / "components"
        iter_dir = Path(rec.get("iter_dir") or cleaned.parent.parent.parent)
        bbox_root = iter_dir / "bbox_components"
        bbox_dir = bbox_root / cleaned.parent.name / "components"
        for entry in meta.get("components") or []:
            cf = str(entry.get("component_file", ""))
            filename = Path(cf).name
            if not filename or filename in invalid:
                continue
            verdict = verdict_map.get(filename, {})
            out.append({
                "iter_idx": iter_idx,
                "original_filename": filename,
                "final_filename": f"iter{iter_idx:02d}_{filename}",
                "text_type": entry.get("text_type"),
                "score": entry.get("score"),
                "bbox_xyxy": entry.get("bbox_xyxy"),
                "bbox_area": entry.get("bbox_area"),
                "segmentation_mode": entry.get("segmentation_mode"),
                "is_text_only": bool(verdict.get("is_text_only", False)),
                "ocr_latex": str(verdict.get("ocr_latex", "")).strip(),
                "valid_reason": str(verdict.get("reason", "")).strip(),
                "bbox_was_tightened": bool(entry.get("bbox_was_tightened", False)),
                "bbox_tightening_source": entry.get("bbox_tightening_source"),
                "original_bbox_xyxy": entry.get("original_bbox_xyxy"),
                "tight_bbox_xyxy": entry.get("tight_bbox_xyxy"),
                "polygon_xyxy": entry.get("polygon_xyxy"),
                "bbox_overlap_carved_by": entry.get("bbox_overlap_carved_by"),
                "seg_path": str((seg_dir / filename).resolve()),
                "bbox_path": str((bbox_dir / filename).resolve()),
            })
    return out


def _build_consolidated_overlay(
    original_image_path: str,
    items: list[dict],
    out_path: Path,
    color_by_granularity: bool = True,
) -> Path:
    """Draw ALL valid bboxes onto the original slide at out_path.

    When ``color_by_granularity`` is True:
        red    = fine SAM3 component
        orange = layout_review missed_region
        blue   = merged-group bbox (drawn thicker)
    Otherwise everything is drawn red (used for the overlay we feed to H).
    """
    img = Image.open(original_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for it in items:
        bbox = it.get("bbox_xyxy") or it.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        gran = it.get("granularity", "fine")
        src = it.get("source", "sam3")
        if color_by_granularity:
            if gran == "merged":
                color, width = (0, 0, 255), 6
            elif src == "layout_review":
                color, width = (255, 140, 0), 5
            else:
                color, width = (255, 0, 0), 3
        else:
            color, width = (255, 0, 0), 4
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    return out_path


_TEXT_ONLY_PRONE_CLASSES = frozenset(
    s.lower()
    for s in [
        # Lists
        "bullet list", "bullet point", "bullet point text", "bullet marker",
        "numbered list", "list", "checklist", "icon list", "outline",
        # Plain text blocks
        "body text", "text", "text block", "text box", "text element",
        "text label", "text character", "text callout",
        # Titles / headings
        "slide title", "title slide", "title text",
        "heading", "heading text", "section title",
        "header", "header text", "header/title", "header/title text",
        "header/body text block", "header/footer", "header/label",
        # Captions / labels
        "caption/label", "label", "label text", "label/annotation",
        "label/header", "label/icon", "callout/label",
        "annotation", "diagram label", "shape label", "component label",
        "metric annotation",
        # Footers / page furniture
        "footer", "page number",
        # Author / affiliation
        "author affiliation", "author affiliation text",
        "author affiliation text block", "author affiliations",
        "author and affiliation", "author and affiliation text",
        "author and affiliation text block", "author attribution",
        "author biography", "author block", "author contact",
        "author information", "author names", "author profiles",
        "author text", "author/affiliation", "author/affiliation text",
        "author/affiliation text block", "author/presenter name",
        "affiliation list", "affiliation text", "presenter bio",
        "contact information",
        # Citations
        "citation/reference",
        # Math / code
        "equation", "mathematical notation", "mathematical symbol",
        "code block", "code example", "theorem statement",
        # Body text variants
        "definition box", "dialogue", "dialogue/conversation",
        "speech bubble", "question text", "question and answer",
        "question-answer block",
        # Numeric
        "numeric metric", "numeric text", "numeric value", "metric",
        "metric/statistic", "metric indicator", "performance indicator",
    ]
)


def _is_text_only_prone(class_name: str | None) -> bool:
    """Whether a taxonomy class tends to produce loose bboxes that extend
    into other components' regions. Used to bias polygon-refine carve
    direction: when a text-only-prone class overlaps with a non-text
    class, the text class is the one whose bbox is suspect."""
    if not class_name:
        return False
    return str(class_name).strip().lower() in _TEXT_ONLY_PRONE_CLASSES


def _bboxes_overlap_area(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _mask_polygon_in_slide_coords(
    seg_path: str | None,
    bbox: list[float] | None,
) -> "ShapelyPolygon | ShapelyMultiPolygon | None":
    """Return a shapely geometry of the candidate's actual pixel mask in
    SLIDE coordinates, derived from the RGBA crop at ``seg_path`` (alpha>0).

    Used by the L-shape carve in polygon refinement so the group is carved
    by what the candidate's pixels actually cover, not by the candidate's
    loose detection bbox (which often extends well past its visible
    content into another component's region).

    Returns None when:
      - seg_path is missing or unreadable
      - alpha mask is uniformly 0 / 255 (degenerate)
      - alpha-channel dims don't match the bbox (size mismatch)
      - resulting polygon has zero area
    """
    if not seg_path or not bbox or len(bbox) != 4:
        return None
    p = Path(str(seg_path))
    if not p.is_file():
        return None
    bx1 = int(round(float(bbox[0])))
    by1 = int(round(float(bbox[1])))
    bx2 = int(round(float(bbox[2])))
    by2 = int(round(float(bbox[3])))
    if bx2 <= bx1 or by2 <= by1:
        return None
    try:
        img = Image.open(str(p))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        alpha = np.array(img)[..., 3]
    except Exception:
        return None
    ah, aw = alpha.shape
    if aw != (bx2 - bx1) or ah != (by2 - by1):
        return None
    binary = (alpha > 0).astype(np.uint8) * 255
    if binary.sum() == 0:
        return None
    try:
        import cv2  # lazy — pipeline image deps already loaded by SAM3
    except Exception:
        return None
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    polys: list[ShapelyPolygon] = []
    for c in contours:
        pts = c.reshape(-1, 2)
        if len(pts) < 3:
            continue
        ring = [(float(x) + bx1, float(y) + by1) for x, y in pts]
        try:
            poly = ShapelyPolygon(ring)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area <= 1.0:
                continue
            polys.append(poly)
        except Exception:
            continue
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    try:
        return ShapelyMultiPolygon(polys)
    except Exception:
        return polys[0]


def _component_mask_in_region(
    seg_path: str | None,
    bbox: list[float] | None,
    rx1: int,
    ry1: int,
    rw: int,
    rh: int,
) -> np.ndarray:
    """Return a ``rh x rw`` boolean mask of the component's actual content,
    sliced to the slide-coords window ``[rx1, rx1+rw) x [ry1, ry1+rh)``.

    Loads the RGBA crop at ``seg_path`` and uses ``alpha > 0`` as the mask.
    Falls back to "the entire bbox rectangle is the mask" when the seg
    file is missing, unreadable, or its dimensions don't match the bbox
    (e.g. layout-review missed regions have no SAM3 mask at all).
    """
    out = np.zeros((rh, rw), dtype=bool)
    if not bbox or len(bbox) != 4 or rw <= 0 or rh <= 0:
        return out
    bx1 = int(round(float(bbox[0])))
    by1 = int(round(float(bbox[1])))
    bx2 = int(round(float(bbox[2])))
    by2 = int(round(float(bbox[3])))
    if bx2 <= bx1 or by2 <= by1:
        return out
    ix1 = max(rx1, bx1)
    iy1 = max(ry1, by1)
    ix2 = min(rx1 + rw, bx2)
    iy2 = min(ry1 + rh, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return out

    used_seg = False
    if seg_path and Path(str(seg_path)).is_file():
        try:
            img = Image.open(str(seg_path))
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            arr = np.array(img)
            alpha = arr[..., 3]
            ah, aw = alpha.shape
            if aw == (bx2 - bx1) and ah == (by2 - by1):
                out[iy1 - ry1: iy2 - ry1, ix1 - rx1: ix2 - rx1] = (
                    alpha[iy1 - by1: iy2 - by1, ix1 - bx1: ix2 - bx1] > 0
                )
                used_seg = True
        except Exception:
            used_seg = False
    if not used_seg:
        # Fallback: full bbox rectangle.
        out[iy1 - ry1: iy2 - ry1, ix1 - rx1: ix2 - rx1] = True
    return out


def _shapely_to_vertices(geom) -> list[list[float]] | None:
    """Convert a shapely geometry to a list of (x, y) vertices.

    Returns None when:
      - the geometry is the rectangular bounding box of itself (no need to
        store a polygon — caller falls back to rect bbox crop).
      - the geometry has interior holes or is disconnected
        (MultiPolygon) — caller falls back to rect bbox.
    """
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, ShapelyMultiPolygon):
        return None  # disconnected — fallback to rect
    if not isinstance(geom, ShapelyPolygon):
        return None
    if list(geom.interiors):
        return None  # has hole — fallback to rect
    coords = list(geom.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 4:
        return None
    # If it's just a rectangle (4 corners forming an axis-aligned rect),
    # return None — caller will use the bbox.
    if len(coords) == 4:
        xs = sorted({round(p[0], 3) for p in coords})
        ys = sorted({round(p[1], 3) for p in coords})
        if len(xs) == 2 and len(ys) == 2:
            return None
    return [[float(x), float(y)] for x, y in coords]


def _build_polygon_refinement_overlay(
    original_image_path: str,
    merged_polygon_geom,
    candidate_bbox: list[float],
    out_path: Path,
) -> Path:
    """Render an overlay for Agent M2: original slide with the current
    merged-group region outlined in green and the candidate bbox outlined
    in thick yellow.
    """
    img = Image.open(original_image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    # Green polygon (semi-transparent fill + opaque outline)
    if merged_polygon_geom is not None and not merged_polygon_geom.is_empty:
        polys = (
            list(merged_polygon_geom.geoms)
            if isinstance(merged_polygon_geom, ShapelyMultiPolygon)
            else [merged_polygon_geom]
        )
        for poly in polys:
            if not isinstance(poly, ShapelyPolygon):
                continue
            ext = [(float(x), float(y)) for x, y in poly.exterior.coords]
            if len(ext) >= 3:
                draw.polygon(ext, fill=(0, 200, 0, 60), outline=(0, 200, 0, 255))
                # Re-stroke with thicker outline (PIL polygon outline is 1 px).
                for i in range(len(ext)):
                    draw.line([ext[i], ext[(i + 1) % len(ext)]], fill=(0, 200, 0, 255), width=5)
    # Yellow candidate bbox
    if candidate_bbox and len(candidate_bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in candidate_bbox]
        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 200, 0, 255), width=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    return out_path


def _polygon_refine_merge_groups(
    backend,
    original_image_path: str,
    merge_overlay_dir: Path,
    merge_groups: list[dict],
    all_components: list[dict],
    seg_path_by_filename: dict[str, str | None] | None = None,
    max_rounds: int = 3,
) -> dict:
    """Post-merge polygon refinement (v2.6.1).

    For each merge_group, iteratively check non-member components whose
    bbox overlaps the group's current geometry. For each overlap, ask
    Agent M2 two things:

      1. ``should_merge`` — is the candidate part of the same entity?
      2. ``visually_overlaps`` — when not part of the same entity, do
         the actual pixels of candidate and group share screen space,
         or is the overlap just bbox-level "L-shape" coincidence?

    Routing:
      - ``should_merge=True`` → union the candidate's bbox into the
        group geometry, add it to ``member_filenames``.
      - ``should_merge=False`` and ``visually_overlaps=False`` →
        **L-shape**: carve group polygon by candidate bbox
        (``geom = geom.difference(cand_box)``). The candidate is
        untouched. Common when an L-shaped composite geometrically
        wraps around an unrelated photo / label that isn't actually
        behind it.
      - ``should_merge=False`` and ``visually_overlaps=True``:
          * If either side is text-prone → leave both unchanged.
            Slicing pixels out of a text glyph or putting a hole in a
            text region produces worse output than just keeping the
            rectangle.
          * Both non-text → schedule a *pixel-erase* (``pixel_erases``
            list): when the merged group's final crop is written, the
            candidate's mask pixels in the group crop will be painted
            to the background color (alpha=0 in the RGBA
            ``components/`` output, white pixels in the opaque
            ``bbox_components/`` output).

    ``seg_path_by_filename`` maps filenames to the RGBA mask-cut crop on
    disk. Used only for the pixel-erase execution step (not for the
    L-shape vs visual-overlap decision, which is now M2's job).

    Each merge_group is mutated in place with possibly extended
    ``member_filenames``, an updated ``merged_bbox`` (rectangular hull),
    and an optional ``polygon`` field (orthogonal polygon vertices) when
    the geometry deviates from a simple rectangle.

    Returns ``{"merge_groups": [...], "pixel_erases": [{merge_index,
    candidate_filename, candidate_class, candidate_bbox,
    candidate_seg_path, overlap_pixels, reason}, ...]}``.
    """
    if not merge_groups:
        return {"merge_groups": merge_groups, "pixel_erases": []}
    merge_overlay_dir.mkdir(parents=True, exist_ok=True)
    by_name = {c["filename"]: c for c in all_components if c.get("filename")}
    seg_path_by_filename = seg_path_by_filename or {}
    # Filenames already absorbed across all groups (avoid double-merging).
    absorbed_global: set[str] = set()
    for g in merge_groups:
        absorbed_global.update(g.get("member_filenames") or [])
    pixel_erases: list[dict] = []
    overlay_idx = 0
    for g_index, g in enumerate(merge_groups):
        member_set = set(g.get("member_filenames") or [])
        if not member_set:
            continue
        try:
            geom = shapely_box(*[float(v) for v in g["merged_bbox"]])
        except Exception:
            continue
        # Candidates already decided NO for this group in earlier rounds —
        # don't re-query them when an unrelated candidate changes geom.
        decided_no: set[str] = set()
        for round_idx in range(max_rounds):
            changed_this_round = False
            # Pre-snapshot: candidates this round = non-member components
            # whose bbox bbox-overlaps the current geom's bounding rect by
            # > 1 px^2 AND haven't already been decided this group.
            geom_minx, geom_miny, geom_maxx, geom_maxy = geom.bounds
            candidates: list[tuple[str, dict]] = []
            for fn, c in by_name.items():
                if fn in member_set or fn in absorbed_global or fn in decided_no:
                    continue
                bb = c.get("bbox")
                if not bb or len(bb) != 4:
                    continue
                if _bboxes_overlap_area(bb, [geom_minx, geom_miny, geom_maxx, geom_maxy]) <= 1.0:
                    continue
                # Check actual geom intersection (not just hull bbox).
                cand_box = shapely_box(*[float(v) for v in bb])
                inter = geom.intersection(cand_box)
                if inter.is_empty or inter.area <= 1.0:
                    continue
                candidates.append((fn, c))
            if not candidates:
                break
            for fn, c in candidates:
                bb = [float(v) for v in c["bbox"]]
                cand_box = shapely_box(*bb)
                if not geom.intersects(cand_box) or geom.intersection(cand_box).area <= 1.0:
                    # Geometry changed earlier in this round and now no
                    # longer overlaps — skip.
                    continue
                # Render overlay for M2 (uses current — possibly already
                # carved — geom).
                overlay_idx += 1
                overlay_path = merge_overlay_dir / f"m2_overlay_{overlay_idx:04d}.png"
                _build_polygon_refinement_overlay(
                    original_image_path=original_image_path,
                    merged_polygon_geom=geom,
                    candidate_bbox=bb,
                    out_path=overlay_path,
                )
                members_summary = [
                    {
                        "filename": m,
                        "class": str(by_name.get(m, {}).get("class", "")),
                        "bbox": by_name.get(m, {}).get("bbox"),
                    }
                    for m in sorted(member_set)
                ]
                try:
                    decision = agent_m2_can_merge(
                        backend=backend,
                        screen_backend=AGENT_SCREEN_BACKEND,
                        original_image_path=original_image_path,
                        overlay_image_path=str(overlay_path),
                        merged_class=str(g.get("merged_class", "")),
                        merged_members_summary=members_summary,
                        candidate_filename=fn,
                        candidate_class=str(c.get("class", "")),
                        candidate_bbox=bb,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [warn] agent_m2 failed for {fn}: {exc}")
                    decision = {"should_merge": False, "reason": str(exc)}

                if decision.get("should_merge"):
                    geom = geom.union(cand_box)
                    member_set.add(fn)
                    absorbed_global.add(fn)
                    changed_this_round = True
                    print(
                        f"    [m2] merged in {fn} ({c.get('class','?')}) "
                        f"into '{g.get('merged_class','?')}': "
                        f"{decision.get('reason','')[:80]}"
                    )
                    continue

                # M2 says NO. Trust M2's `visually_overlaps` verdict to
                # distinguish "L-shape" (just bbox overlap) from "real"
                # visual overlap — no mask-level pixel check.
                decided_no.add(fn)
                visually_overlaps = bool(decision.get("visually_overlaps", True))

                if not visually_overlaps:
                    # L-shape: bboxes overlap but the actual content
                    # doesn't. Carve the group polygon by the candidate's
                    # actual PIXEL MASK in slide coords (falls back to
                    # candidate bbox when the mask is unavailable). Using
                    # the mask avoids removing group-member pixels that
                    # happen to sit inside the candidate's loose detection
                    # rectangle but outside its visible content.
                    cand_seg_path = seg_path_by_filename.get(fn)
                    cand_mask_geom = _mask_polygon_in_slide_coords(
                        cand_seg_path, bb
                    )
                    if cand_mask_geom is None or cand_mask_geom.is_empty:
                        cut_geom = cand_box
                        cut_source = "bbox (no mask)"
                    else:
                        cut_geom = cand_mask_geom
                        cut_source = "mask"
                    new_geom = geom.difference(cut_geom)
                    if new_geom.is_empty:
                        print(
                            f"    [m2] L-shape carve of group by {fn} "
                            f"would leave group empty; skipping."
                        )
                        continue
                    if isinstance(new_geom, ShapelyMultiPolygon):
                        new_geom = max(new_geom.geoms, key=lambda p: p.area)
                    if not isinstance(new_geom, ShapelyPolygon):
                        continue
                    if list(new_geom.interiors):
                        print(
                            f"    [m2] L-shape carve of group by {fn} "
                            f"would create hole; skipping."
                        )
                        continue
                    # Over-shrink guard: if the carve would consume most
                    # of the group's area, M2's "no visual overlap"
                    # verdict is almost certainly wrong (it would mean
                    # the group's actual pixels happen to sit inside
                    # candidate's footprint, contradicting the verdict).
                    # Treat this as a real visual overlap instead and
                    # fall through to the pixel-erase path below.
                    geom_area = max(1e-6, geom.area)
                    new_area_ratio = new_geom.area / geom_area
                    if new_area_ratio < 0.15:
                        print(
                            f"    [m2] L-shape carve of group by {fn} "
                            f"would shrink group to "
                            f"{new_area_ratio*100:.1f}% of its area; "
                            f"M2's no-overlap verdict looks wrong "
                            f"— treating as real visual overlap "
                            f"(pixel-erase) instead."
                        )
                        # Fall through to the pixel-erase logic below.
                        visually_overlaps = True
                    else:
                        geom = new_geom
                        changed_this_round = True
                        print(
                            f"    [m2] L-shape: {fn} ({c.get('class','?')}) "
                            f"bbox-overlaps '{g.get('merged_class','?')}' but "
                            f"M2 says pixels don't visually overlap "
                            f"— carved group polygon by candidate {cut_source}."
                        )
                        continue

                # Real visual overlap. If either side is text-prone, do
                # not separate them — slicing through text glyphs or
                # putting a hole into a text region produces worse output
                # than just keeping the rectangle.
                cand_text = _is_text_only_prone(c.get("class"))
                group_text = _is_text_only_prone(g.get("merged_class"))
                if cand_text or group_text:
                    print(
                        f"    [m2] visual-overlap with text component "
                        f"({fn} class='{c.get('class','?')}', "
                        f"group='{g.get('merged_class','?')}') "
                        f"— not separating."
                    )
                    continue

                # Both non-text, real visual overlap, M2 says not the
                # same entity. Schedule a pixel-erase: in the final
                # group crop, the candidate's mask pixels will be
                # painted to the background color.
                pixel_erases.append({
                    "merge_index": g_index,
                    "merged_class": str(g.get("merged_class", "")),
                    "candidate_filename": fn,
                    "candidate_class": str(c.get("class", "")),
                    "candidate_bbox": list(bb),
                    "candidate_seg_path": str(seg_path_by_filename.get(fn) or ""),
                    "reason": str(decision.get("reason", ""))[:160],
                })
                # Don't set changed_this_round=True: geom hasn't changed,
                # so re-evaluating other candidates against geom would
                # produce the same M2 decisions.
                print(
                    f"    [m2] visual-overlap, both non-text: {fn} "
                    f"({c.get('class','?')}) and '{g.get('merged_class','?')}' "
                    f"are independent — candidate mask pixels will be "
                    f"erased from group's final crop."
                )
            if not changed_this_round:
                break
        # Tighten the merged geom to the actual union of member pixel masks.
        # M2 carves what's been removed, but the merge group's enclosing
        # rectangle is still the union of member detection bboxes — those
        # bboxes typically extend past the visible content (anti-aliased
        # borders, SAM3 over-shoot). Restrict the geom to a small padding
        # around the union of every member's RGBA-alpha mask polygon, so the
        # rendered merged crop hugs the real content.
        # IMPORTANT: members can be either SAM3 detections (have a seg_path
        # mask) or Agent H "missed regions" (no SAM3 mask, only a bbox).
        # For members without a mask, fall back to the bbox rectangle so
        # they're not silently dropped from the tight hull.
        member_geoms: list = []
        for mfn in member_set:
            mc = by_name.get(mfn)
            if not mc:
                continue
            mbb = mc.get("bbox")
            if not mbb:
                continue
            mseg = seg_path_by_filename.get(mfn)
            mg_geom = None
            if mseg:
                mg_geom = _mask_polygon_in_slide_coords(mseg, mbb)
            if mg_geom is None or mg_geom.is_empty:
                try:
                    mg_geom = shapely_box(*[float(v) for v in mbb])
                except Exception:
                    mg_geom = None
            if mg_geom is not None and not mg_geom.is_empty:
                member_geoms.append(mg_geom)
        if member_geoms:
            try:
                union_geom = member_geoms[0]
                for mg2 in member_geoms[1:]:
                    union_geom = union_geom.union(mg2)
                if not union_geom.is_empty:
                    umin_x, umin_y, umax_x, umax_y = union_geom.bounds
                    pad = 2.0  # tolerate anti-aliased mask edges
                    tight_rect = shapely_box(
                        umin_x - pad, umin_y - pad, umax_x + pad, umax_y + pad
                    )
                    tight_geom = geom.intersection(tight_rect)
                    if not tight_geom.is_empty:
                        if isinstance(tight_geom, ShapelyMultiPolygon):
                            tight_geom = max(tight_geom.geoms, key=lambda p: p.area)
                        if isinstance(tight_geom, ShapelyPolygon):
                            old_area = geom.area
                            geom = tight_geom
                            if old_area > 0:
                                shrunk = 1.0 - (geom.area / old_area)
                                if shrunk > 0.02:
                                    print(
                                        f"    [m2] tightened merged "
                                        f"'{g.get('merged_class','?')}' geom "
                                        f"to mask union (shrunk by "
                                        f"{shrunk*100:.1f}%)."
                                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"    [m2] tighten step failed for "
                    f"'{g.get('merged_class','?')}': {exc}"
                )
        # Persist results back into the merge_group.
        bounds = geom.bounds if not geom.is_empty else g["merged_bbox"]
        g["member_filenames"] = sorted(member_set)
        g["merged_bbox"] = [
            float(bounds[0]),
            float(bounds[1]),
            float(bounds[2]),
            float(bounds[3]),
        ]
        verts = _shapely_to_vertices(geom)
        if verts is not None:
            g["polygon"] = verts
        else:
            g.pop("polygon", None)
    return {"merge_groups": merge_groups, "pixel_erases": pixel_erases}


def _assign_z_indices(final_components: list[dict], comps_dir: Path) -> None:
    """Compute and assign z_index to each final component based on the
    dynamic overlap white-ratio priority (ported from common.py).
    """
    import collections
    import heapq
    from PIL import Image

    N = len(final_components)
    if N == 0:
        return

    # 1. Precompute area and integer bounding boxes
    areas = []
    bbox_ints = []
    for comp in final_components:
        bbox = comp["bbox_xyxy"]
        x0, y0, x1, y1 = int(round(bbox[0])), int(round(bbox[1])), int(round(bbox[2])), int(round(bbox[3]))
        bbox_ints.append((x0, y0, x1, y1))
        areas.append((x1 - x0) * (y1 - y0))

    # 2. Build Kahn's topological sort graph
    adj = collections.defaultdict(list)
    in_degree = [0] * N

    for i in range(N):
        x0_i, y0_i, x1_i, y1_i = bbox_ints[i]
        for j in range(i + 1, N):
            x0_j, y0_j, x1_j, y1_j = bbox_ints[j]

            # Calculate intersection box
            ox0 = max(x0_i, x0_j)
            oy0 = max(y0_i, y0_j)
            ox1 = min(x1_i, x1_j)
            oy1 = min(y1_i, y1_j)

            if ox1 > ox0 and oy1 > oy0:
                # Overlap exists! Load crops and compare white pixel ratio in the intersection.
                crop_path_i = comps_dir / final_components[i]["filename"]
                crop_path_j = comps_dir / final_components[j]["filename"]

                if crop_path_i.is_file() and crop_path_j.is_file():
                    try:
                        with Image.open(crop_path_i) as img_i, Image.open(crop_path_j) as img_j:
                            # Crop to local intersection
                            lx0_i, ly0_i = ox0 - x0_i, oy0 - y0_i
                            lx1_i, ly1_i = ox1 - x0_i, oy1 - y0_i
                            lx0_j, ly0_j = ox0 - x0_j, oy0 - y0_j
                            lx1_j, ly1_j = ox1 - x0_j, oy1 - y0_j

                            slice_i = img_i.crop((lx0_i, ly0_i, lx1_i, ly1_i))
                            slice_j = img_j.crop((lx0_j, ly0_j, lx1_j, ly1_j))

                            def get_white_ratio(sl: Image.Image) -> float:
                                arr = np.array(sl.convert("RGB"))
                                white_pixels = (arr[..., 0] >= 250) & (arr[..., 1] >= 250) & (arr[..., 2] >= 250)
                                if sl.mode == "RGBA":
                                    alpha = np.array(sl)[..., 3]
                                    opaque = alpha > 0
                                    total = int(opaque.sum())
                                    if total == 0:
                                        return 1.0
                                    return int((white_pixels & opaque).sum()) / total
                                else:
                                    total = arr.shape[0] * arr.shape[1]
                                    if total == 0:
                                        return 1.0
                                    return int(white_pixels.sum()) / total

                            ratio_i = get_white_ratio(slice_i)
                            ratio_j = get_white_ratio(slice_j)

                            # Component with LESS white ratio must be drawn ON TOP
                            if ratio_i < ratio_j - 0.02:
                                adj[j].append(i)
                                in_degree[i] += 1
                            elif ratio_j < ratio_i - 0.02:
                                adj[i].append(j)
                                in_degree[j] += 1
                    except Exception:
                        pass

    # Kahn's algorithm with priority queue
    pq = []
    for i in range(N):
        if in_degree[i] == 0:
            heapq.heappush(pq, (-areas[i], i))

    sorted_indices = []
    while pq:
        _, u = heapq.heappop(pq)
        sorted_indices.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(pq, (-areas[v], v))

    # Cycle fallback (safety net)
    if len(sorted_indices) < N:
        sorted_indices = sorted(range(N), key=lambda idx: areas[idx], reverse=True)

    # Assign z_index (0 to N-1) based on sorted order
    for z_idx, idx in enumerate(sorted_indices):
        final_components[idx]["z_index"] = z_idx


def _build_final_subdir(
    image_dir: Path,
    original_image_path: str,
    iters: list[dict],
    layout_review: dict | None,
    layout_review_raw: dict | None,
    backend=None,
    f_dual_image: bool = False,
) -> dict:
    """Materialise the per-image ``final/`` subdir that holds the
    consolidated, granularity-tagged output:

        final/
          components/                 (all valid masks/RGBA + merged + missed)
          bbox_components/            (opaque bbox crops, same filenames)
          overlay.png                 (all valid bboxes color-coded)
          metadata.json               (full granularity-tagged manifest)
          layout_review.json          (raw Agent H response, if any)
          review_overlay.png          (only-red overlay used to query H, if any)

    Returns the manifest dict.
    """
    final_dir = image_dir / "final"
    comps_dir = final_dir / "components"
    bbox_dir = final_dir / "bbox_components"
    comps_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fine SAM3 components (across all iters, post all pruning).
    fine = _collect_valid_components(iters)
    consolidated: list[dict] = []
    for c in fine:
        seg_src = Path(c["seg_path"])
        bbox_src = Path(c["bbox_path"])
        seg_dst = comps_dir / c["final_filename"]
        bbox_dst = bbox_dir / c["final_filename"]
        if seg_src.is_file():
            shutil.copy2(seg_src, seg_dst)
        if bbox_src.is_file():
            shutil.copy2(bbox_src, bbox_dst)
        consolidated.append({
            "filename": c["final_filename"],
            "granularity": "fine",
            "source": "sam3",
            "iter_idx": c["iter_idx"],
            "iter_original_filename": c["original_filename"],
            "text_type": c["text_type"],
            "score": c["score"],
            "bbox_xyxy": list(c["bbox_xyxy"]),
            "bbox_area": (c["bbox_xyxy"][2] - c["bbox_xyxy"][0]) * (c["bbox_xyxy"][3] - c["bbox_xyxy"][1]),
            "segmentation_mode": c["segmentation_mode"],
            "is_text_only": c["is_text_only"],
            "ocr_latex": c["ocr_latex"],
            "valid_reason": c["valid_reason"],
            "bbox_was_tightened": c.get("bbox_was_tightened", False),
            "bbox_tightening_source": c.get("bbox_tightening_source"),
            "original_bbox_xyxy": c.get("original_bbox_xyxy"),
            "tight_bbox_xyxy": c.get("tight_bbox_xyxy"),
            "polygon_xyxy": c.get("polygon_xyxy"),
            "bbox_overlap_carved_by": c.get("bbox_overlap_carved_by"),
            "merged_into": None,
        })

    # 2. Layout review missed_regions (synthetic fine-grain components).
    if layout_review:
        for idx, mr in enumerate(layout_review.get("missed_regions") or [], start=1):
            cls = str(mr["class"])
            bbox = [float(v) for v in mr["bbox"]]
            fname = f"missed_{idx:04d}_{_safe_name(cls)}.png"
            try:
                img = Image.open(original_image_path).convert("RGB")
                w, h = img.size
                x1 = max(0.0, min(float(w), bbox[0]))
                y1 = max(0.0, min(float(h), bbox[1]))
                x2 = max(0.0, min(float(w), bbox[2]))
                y2 = max(0.0, min(float(h), bbox[3]))
                if x2 > x1 and y2 > y1:
                    img.crop((int(x1), int(y1), int(x2), int(y2))).save(comps_dir / fname)
                    img.crop((int(x1), int(y1), int(x2), int(y2))).save(bbox_dir / fname)
                    consolidated.append({
                        "filename": fname,
                        "granularity": "fine",
                        "source": "layout_review",
                        "iter_idx": None,
                        "iter_original_filename": None,
                        "text_type": cls,
                        "score": None,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox_area": (x2 - x1) * (y2 - y1),
                        "segmentation_mode": "bbox",
                        "is_text_only": False,
                        "ocr_latex": "",
                        "valid_reason": "",
                        "description": str(mr.get("description", "")).strip(),
                        "merged_into": None,
                    })
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] could not save missed region {idx}: {exc}")

    # 3. Merged-group components (post-merge granularity).
    if layout_review:
        # Build a quick lookup from members iter_filename -> consolidated entry.
        original_to_final = {
            (c["iter_idx"], c["iter_original_filename"]): c["filename"]
            for c in consolidated
            if c["granularity"] == "fine" and c["source"] == "sam3"
        }
        # Also accept the final filename directly.
        final_index = {c["filename"]: c for c in consolidated}
        for idx, mg in enumerate(layout_review.get("merge_groups") or [], start=1):
            cls = str(mg["merged_class"])
            bbox = [float(v) for v in mg["merged_bbox"]]
            members_in: list[str] = list(mg.get("member_filenames") or [])
            # Resolve member references to final_filenames (accept either form).
            resolved_members: list[str] = []
            for m in members_in:
                if m in final_index:
                    resolved_members.append(m)
                    continue
                # try matching iter original filename — match against any iter
                matched = None
                for (ii, on), fn in original_to_final.items():
                    if on == m:
                        matched = fn
                        break
                if matched:
                    resolved_members.append(matched)
            if len(resolved_members) < 2:
                print(
                    f"  [warn] merge_group {idx} ({cls}): only "
                    f"{len(resolved_members)} resolvable members; skipping"
                )
                continue
            fname = f"merged_{idx:04d}_{_safe_name(cls)}.png"
            polygon_verts = mg.get("polygon")
            # Pixel-erases scheduled by polygon refinement for this merge
            # group: each entry's mask pixels get painted to the background
            # color inside the merged crop (independent components that
            # visually overlap the group but aren't part of it).
            pe_for_this = [
                pe for pe in (layout_review.get("pixel_erases") or [])
                if int(pe.get("merge_index", -1)) == idx - 1
            ]
            erased_pixel_count = 0
            try:
                img = Image.open(original_image_path).convert("RGB")
                w, h = img.size
                x1 = max(0.0, min(float(w), bbox[0]))
                y1 = max(0.0, min(float(h), bbox[1]))
                x2 = max(0.0, min(float(w), bbox[2]))
                y2 = max(0.0, min(float(h), bbox[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                rect_crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                cw, ch = rect_crop.size

                # Build the candidate erase mask (cw × ch bool) — union of
                # every scheduled candidate's mask, clipped to this crop.
                erase_mask = np.zeros((ch, cw), dtype=bool)
                for pe in pe_for_this:
                    cand_local = _component_mask_in_region(
                        pe.get("candidate_seg_path") or None,
                        pe.get("candidate_bbox"),
                        int(round(x1)),
                        int(round(y1)),
                        cw,
                        ch,
                    )
                    erase_mask |= cand_local
                if erase_mask.any():
                    erased_pixel_count = int(erase_mask.sum())
                    # Paint opaque RGB crop white where erase_mask is True.
                    bbox_arr = np.array(rect_crop)
                    bbox_arr[erase_mask] = (255, 255, 255)
                    rect_crop = Image.fromarray(bbox_arr)

                if polygon_verts and len(polygon_verts) >= 4:
                    # Polygon-aware crop: RGBA with alpha=255 inside polygon,
                    # 0 outside. bbox_components/ keeps the rectangular crop
                    # (downstream consumers that only support rectangular
                    # bboxes can fall back to it).
                    alpha_img = Image.new("L", (cw, ch), 0)
                    poly_local = [
                        (max(0, min(cw, float(px) - x1)), max(0, min(ch, float(py) - y1)))
                        for px, py in polygon_verts
                    ]
                    ImageDraw.Draw(alpha_img).polygon(poly_local, fill=255)
                    if erased_pixel_count:
                        alpha_arr = np.array(alpha_img)
                        alpha_arr[erase_mask] = 0
                        alpha_img = Image.fromarray(alpha_arr)
                    rgba = rect_crop.convert("RGBA")
                    rgba.putalpha(alpha_img)
                    rgba.save(comps_dir / fname)
                    rect_crop.save(bbox_dir / fname)
                else:
                    if erased_pixel_count:
                        # No polygon, but we still want components/ to
                        # encode the erased pixels as transparent so
                        # downstream RGBA-aware consumers see the hole.
                        alpha_arr = np.full((ch, cw), 255, dtype=np.uint8)
                        alpha_arr[erase_mask] = 0
                        rgba = rect_crop.convert("RGBA")
                        rgba.putalpha(Image.fromarray(alpha_arr))
                        rgba.save(comps_dir / fname)
                    else:
                        rect_crop.save(comps_dir / fname)
                    rect_crop.save(bbox_dir / fname)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] could not save merged crop {idx}: {exc}")
                continue
            consolidated.append({
                "filename": fname,
                "granularity": "merged",
                "source": "merged",
                "iter_idx": None,
                "iter_original_filename": None,
                "text_type": cls,
                "score": None,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_area": (x2 - x1) * (y2 - y1),
                "segmentation_mode": "polygon" if polygon_verts else "bbox",
                "polygon_xyxy": polygon_verts if polygon_verts else None,
                "is_text_only": False,
                "ocr_latex": "",
                "valid_reason": "",
                "merged_from": list(resolved_members),
                "reason": str(mg.get("reason", "")).strip(),
                "merged_into": None,
                "pixel_erased_candidates": [
                    {
                        "filename": pe.get("candidate_filename"),
                        "class": pe.get("candidate_class"),
                        "overlap_pixels": pe.get("overlap_pixels"),
                    }
                    for pe in pe_for_this
                ] if pe_for_this else None,
                "pixel_erased_total_px": erased_pixel_count or None,
            })
            # Annotate members with merged_into pointer.
            for fn in resolved_members:
                if fn in final_index:
                    final_index[fn]["merged_into"] = fname

    # 4. Color-coded overlay (red=fine, orange=missed, blue=merged).
    _build_consolidated_overlay(
        original_image_path,
        consolidated,
        final_dir / "overlay.png",
        color_by_granularity=True,
    )

    # 5. Post-merge re-OCR (v2.9): for each merge group whose ALL members
    # are text-only, re-run Agent F validity on the merged crop (RGBA
    # polygon-cut so only the merged content is visible) to get a single
    # OCR transcription that spans all members. Updates the merged
    # entry's ``is_text_only`` and ``ocr_latex`` in place.
    if backend is not None:
        by_filename = {c["filename"]: c for c in consolidated}
        for merged in consolidated:
            if merged.get("granularity") != "merged":
                continue
            members = merged.get("merged_from") or []
            if not members:
                continue
            mem_records = [by_filename.get(m) for m in members]
            if not all(
                isinstance(m, dict) and m.get("is_text_only") is True
                for m in mem_records
            ):
                continue
            merged_crop_path = comps_dir / merged["filename"]
            if not merged_crop_path.is_file():
                continue
            # Build a slide-context overlay for the merged crop so F can
            # see where the merged region sits on the full slide. Only
            # builds when --f_dual_image is on. Use a transient path
            # under the final/ subdir.
            merged_bb = merged.get("bbox_xyxy")
            context_path: str | None = None
            if (
                f_dual_image
                and isinstance(merged_bb, list)
                and len(merged_bb) == 4
            ):
                ctx_p = final_dir / "f_context_overlays" / merged["filename"]
                try:
                    render_bbox_overlay(
                        original_image_path,
                        [[float(v) for v in merged_bb]],
                        str(ctx_p),
                        width_px=6,
                    )
                    context_path = str(ctx_p)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [warn] could not render F context for merged {merged['filename']}: {exc}")
                    context_path = None
            try:
                verdict = agent_f_validate_component(
                    backend=backend,
                    component_path=str(merged_crop_path),
                    slide_context_path=context_path,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [warn] post-merge re-OCR failed for "
                    f"{merged['filename']}: {exc}"
                )
                continue
            merged["is_text_only"] = bool(verdict.get("is_text_only", False))
            merged["ocr_latex"] = str(verdict.get("ocr_latex", "")).strip()
            merged["valid_reason"] = str(verdict.get("reason", "")).strip()
            merged["post_merge_ocr"] = True
            print(
                f"  Post-merge re-OCR: {merged['filename']} "
                f"text_only={merged['is_text_only']} "
                f"ocr={('<%d chars>' % len(merged['ocr_latex'])) if merged['ocr_latex'] else '<empty>'}"
            )

    # 6. final_components view (v2.9): the post-merge effective list.
    # Skip fine components that were absorbed into a merged group; keep
    # the merged super-component instead. Each entry carries OCR if
    # text-only.
    final_components: list[dict] = []
    for c in consolidated:
        if c.get("granularity") == "fine" and c.get("merged_into"):
            continue  # absorbed into a merge group
        # Trim down to a clean record (still link back via filename to
        # the full entry in `components` for anyone who needs it).
        final_components.append({
            "filename": c["filename"],
            "granularity": c["granularity"],
            "source": c.get("source"),
            "text_type": c.get("text_type"),
            "bbox_xyxy": c.get("bbox_xyxy"),
            "polygon_xyxy": c.get("polygon_xyxy"),
            "segmentation_mode": c.get("segmentation_mode"),
            "is_text_only": c.get("is_text_only"),
            "ocr_latex": c.get("ocr_latex", "") if c.get("is_text_only") else "",
            "merged_from": c.get("merged_from"),
            "post_merge_ocr": c.get("post_merge_ocr", False),
        })

    # Assign z-indices to all final components based on dynamic overlap white-pixel priority
    _assign_z_indices(final_components, comps_dir)

    # 7. Metadata.
    manifest = {
        "image": original_image_path,
        "num_components": len(consolidated),
        "num_fine": sum(1 for c in consolidated if c["granularity"] == "fine"),
        "num_merged": sum(1 for c in consolidated if c["granularity"] == "merged"),
        "num_layout_review_missed": sum(
            1 for c in consolidated if c.get("source") == "layout_review"
        ),
        "num_bbox_tightened": sum(
            1 for c in consolidated if c.get("bbox_was_tightened")
        ),
        "num_final_components": len(final_components),
        "components": consolidated,
        "final_components": final_components,
    }
    with (final_dir / "metadata.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    if layout_review_raw is not None:
        with (final_dir / "layout_review.json").open("w") as f:
            json.dump(layout_review_raw, f, indent=2)

    return manifest


def _vlm_review_missed_regions(
    missed_regions: list[dict],
    original_image_path: str,
    iter_dir: Path,
    backend,
) -> list[dict]:
    """For each VLM-proposed missed region, do a per-proposal QA call:
    show the bbox crop + the slide-with-bbox-outlined, ask whether the
    proposal is a real, useful component. Drop the ones the VLM judges
    no. Persists the per-proposal artifacts under
    ``iter_dir/missed_vlm_review/``.
    """
    if not missed_regions:
        return missed_regions
    review_dir = iter_dir / "missed_vlm_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    try:
        slide = Image.open(original_image_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        print(f"  [missed-vlm-review] could not open slide ({exc}); keeping all.")
        return missed_regions

    sw, sh = slide.size
    keep: list[dict] = []
    for idx, mr in enumerate(missed_regions, start=1):
        bb = mr.get("bbox")
        cls = str(mr.get("class") or "").strip() or "unknown"
        desc = str(mr.get("description") or "")
        if not bb or len(bb) != 4:
            keep.append(mr)
            continue
        try:
            x1 = max(0, int(round(float(bb[0]))))
            y1 = max(0, int(round(float(bb[1]))))
            x2 = min(sw, int(round(float(bb[2]))))
            y2 = min(sh, int(round(float(bb[3]))))
        except (TypeError, ValueError):
            keep.append(mr)
            continue
        if x2 <= x1 or y2 <= y1:
            print(f"  [missed-vlm-review] {idx}: degenerate bbox, dropping.")
            continue

        crop_path = review_dir / f"missed_{idx:03d}_crop.png"
        slide.crop((x1, y1, x2, y2)).save(str(crop_path))
        context_path = review_dir / f"missed_{idx:03d}_context.png"
        try:
            render_bbox_overlay(
                original_image_path,
                [[float(x1), float(y1), float(x2), float(y2)]],
                str(context_path),
                width_px=6,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [missed-vlm-review] {idx}: overlay render failed ({exc}); keeping.")
            keep.append(mr)
            continue

        try:
            verdict = agent_h_validate_missed_region(
                backend=backend,
                screen_backend=AGENT_SCREEN_BACKEND,
                crop_path=str(crop_path),
                slide_context_path=str(context_path),
                class_name=cls,
                description=desc,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [missed-vlm-review] {idx} ({cls!r}): review failed ({exc}); "
                f"keeping by default."
            )
            keep.append(mr)
            continue

        if verdict.get("valid"):
            print(
                f"  [missed-vlm-review] {idx} ({cls!r}) bbox={[x1,y1,x2,y2]}: "
                f"KEEP — {verdict.get('reason','')[:120]}"
            )
            keep.append(mr)
        else:
            print(
                f"  [missed-vlm-review] {idx} ({cls!r}) bbox={[x1,y1,x2,y2]}: "
                f"DROP — {verdict.get('reason','')[:120]}"
            )
    return keep


def _ground_missed_points_with_sam3(
    point_queries: list[dict],
    original_image_path: str,
    args,
    iter_dir: Path,
) -> list[dict]:
    """Given VLM-proposed anchor points, run infer_point_prompt.py as a
    subprocess to ground each point into a real mask + bbox via SAM3's
    SAM1-task interactive predictor. Returns a list of
    ``{"bbox": [x1,y1,x2,y2], "class": str, "description": str, "seg_path": str}``
    that downstream code can treat just like the bbox-grounded path.

    If anything fails or VLM proposed no points, returns an empty list.
    """
    if not point_queries:
        return []
    point_dir = iter_dir / "missed_via_point_prompt"
    point_dir.mkdir(parents=True, exist_ok=True)
    # Tag queries with the filenames they'll be saved under, matching the
    # ``missed_NNNN_<class>.png`` scheme so downstream merge / final code
    # finds them at the expected paths.
    queries_with_names: list[dict] = []
    for qi, q in enumerate(point_queries, start=1):
        cls = str(q.get("class") or "").strip() or "unknown"
        fname = f"missed_{qi:04d}_{_safe_name(cls)}.png"
        queries_with_names.append({
            "point": q["point"],
            "class": cls,
            "description": str(q.get("description", "")).strip(),
            "filename": fname,
        })
    manifest = [{"image": original_image_path, "queries": queries_with_names}]
    manifest_path = point_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    sam3_output = point_dir / "sam3_output"
    point_script = REPO / "sam3" / "infer_point_prompt.py"
    cmd = [
        str(args.conda_python),
        str(point_script),
        "--manifest", str(manifest_path),
        "--output_dir", str(sam3_output),
        "--base_ckpt", str(args.base_ckpt),
    ]
    if args.device:
        cmd += ["--device", str(args.device)]
    print(f"  [missed-via-point] running: {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"  [missed-via-point] SAM3 subprocess returned {rc}; skipping.")
        return []

    # Read back the per-image metadata.
    img_stem = Path(original_image_path).stem
    out_meta_path = sam3_output / img_stem / "metadata.json"
    if not out_meta_path.is_file():
        print(f"  [missed-via-point] no metadata.json at {out_meta_path}; skipping.")
        return []
    try:
        with out_meta_path.open() as f:
            sam3_meta = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [missed-via-point] parse failure ({exc}); skipping.")
        return []

    grounded: list[dict] = []
    components_dir = sam3_output / img_stem / "components"
    for c in sam3_meta.get("components") or []:
        bb = c.get("bbox_xyxy")
        cls = str(c.get("text_type") or "").strip()
        if not bb or not cls:
            continue
        seg_rel = c.get("component_file") or ""
        seg_abs = str((components_dir / Path(seg_rel).name).resolve()) if seg_rel else ""
        grounded.append({
            "bbox": [float(v) for v in bb],
            "class": cls,
            "description": str(c.get("description", "")).strip(),
            "seg_path": seg_abs,
            "prompt_point_xy": c.get("prompt_point_xy"),
            "iou_pred": c.get("iou_pred"),
        })
    return grounded


def _drop_blank_missed_regions(
    missed_regions: list[dict],
    original_image_path: str,
    min_std: float = 8.0,
) -> list[dict]:
    """Drop Agent H phase 1 "missed" bboxes whose pixel content on the
    ORIGINAL slide is essentially uniform — VLM hallucinations like
    "the right portion of the lion photograph" pointing into empty
    whitespace.

    Heuristic: crop the original slide to the bbox, compute per-channel
    stddev, take the max across channels. If the max stddev is below
    ``min_std`` (default 8 on 0-255 RGB), call it blank.

    Returns the filtered list. Logs each drop.
    """
    if not missed_regions:
        return missed_regions
    try:
        img = Image.open(original_image_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        print(f"  [missed-content-check] could not open slide ({exc}); keeping all.")
        return missed_regions
    arr = np.array(img)
    H, W = arr.shape[:2]
    keep: list[dict] = []
    for mr in missed_regions:
        bb = mr.get("bbox")
        if not bb or len(bb) != 4:
            keep.append(mr)
            continue
        try:
            x1 = max(0, int(round(float(bb[0]))))
            y1 = max(0, int(round(float(bb[1]))))
            x2 = min(W, int(round(float(bb[2]))))
            y2 = min(H, int(round(float(bb[3]))))
        except (TypeError, ValueError):
            keep.append(mr)
            continue
        if x2 <= x1 or y2 <= y1:
            print(
                f"  [missed-content-check] drop {mr.get('class','?')!r} "
                f"bbox={bb} (degenerate)"
            )
            continue
        crop = arr[y1:y2, x1:x2, :]
        if crop.size == 0:
            continue
        try:
            channel_stds = crop.reshape(-1, 3).std(axis=0)
            max_std = float(channel_stds.max())
        except Exception:
            keep.append(mr)
            continue
        if max_std < min_std:
            print(
                f"  [missed-content-check] drop {mr.get('class','?')!r} "
                f"bbox={[round(float(v),0) for v in bb]} "
                f"std={max_std:.2f} < {min_std:.2f} "
                f"(desc: {str(mr.get('description',''))[:80]})"
            )
            continue
        keep.append(mr)
    return keep


def _drop_sparse_detections(
    metadata_path: Path,
    components_dir: Path,
    bbox_components_dir: Path,
    min_mask_density: float,
    min_bbox_area_for_check: float = 0.15,
    original_image_path: str | None = None,
) -> None:
    """Filter SAM3 detections whose RGBA-alpha mask is very SPARSE relative
    to their bbox — i.e. the bbox is way larger than the actual mask
    content. These are catch-all hallucinations (a "data visualization"
    bbox spanning the whole slide while the mask is just a thin curve).

    Two-stage gate:
      1. Only check detections whose bbox area >= ``min_bbox_area_for_check``
         fraction of the slide (default 15%). Small detections are
         trusted as-is — they can be sparse line drawings legitimately.
      2. Drop the detection if (mask alpha pixels) / (bbox area) is
         below ``min_mask_density``.

    Density-based filtering is robust to legit full-slide images
    (wallpaper photos, full-slide diagrams) which have DENSE masks, and
    to slim foreground content (axes, thin arrows) which have small
    bboxes that the first gate exempts.

    Edits ``metadata.json`` in place; removes corresponding crops on disk.
    """
    if not metadata_path.is_file():
        return
    try:
        with metadata_path.open() as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if original_image_path:
        try:
            with Image.open(original_image_path) as img:
                slide_w, slide_h = img.size
        except Exception:
            slide_w = slide_h = 0
    else:
        slide_w = slide_h = 0
    slide_area = float(slide_w) * float(slide_h) if (slide_w and slide_h) else 0.0
    min_check_area = (slide_area * min_bbox_area_for_check) if slide_area > 0 else 0.0
    components = meta.get("components") or []
    keep: list[dict] = []
    dropped: list[tuple[dict, float, float]] = []  # (comp, density, area_ratio)

    for c in components:
        bb = c.get("bbox_xyxy") or []
        ba = float(c.get("bbox_area") or 0.0)
        if ba <= 0 and len(bb) == 4:
            ba = max(0.0, float(bb[2]) - float(bb[0])) * max(0.0, float(bb[3]) - float(bb[1]))
        if ba <= 0 or ba < min_check_area:
            keep.append(c)
            continue
        seg_rel = c.get("component_file")
        if not seg_rel:
            keep.append(c)
            continue
        seg_path = components_dir / Path(str(seg_rel)).name
        if not seg_path.is_file():
            keep.append(c)
            continue
        try:
            with Image.open(str(seg_path)) as im:
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                arr = np.array(im)
            alpha = arr[..., 3]
            mask_pixels = int((alpha > 0).sum())
        except Exception:
            keep.append(c)
            continue
        density = mask_pixels / ba if ba > 0 else 1.0
        if density < min_mask_density:
            area_ratio = (ba / slide_area) if slide_area > 0 else 0.0
            dropped.append((c, density, area_ratio))
        else:
            keep.append(c)

    if not dropped:
        return
    for c, _density, _area_ratio in dropped:
        for key in ("component_file", "bbox_component_file"):
            rel = c.get(key)
            if not rel:
                continue
            fname = Path(str(rel)).name
            for d in (components_dir, bbox_components_dir):
                try:
                    target = d / fname
                    if target.is_file():
                        target.unlink()
                except OSError:
                    pass
    meta["components"] = keep
    try:
        with metadata_path.open("w") as f:
            json.dump(meta, f, indent=2)
    except OSError:
        return
    print(
        f"  [sparse-filter] dropped {len(dropped)} sparse-mask "
        f"detection(s) (mask density < {min_mask_density*100:.0f}% "
        f"AND bbox >= {min_bbox_area_for_check*100:.0f}% slide):"
    )
    for c, density, area_ratio in dropped:
        print(
            f"    - {Path(c.get('component_file','?')).name} "
            f"class='{c.get('text_type','?')}' "
            f"density={density*100:.1f}% area_ratio={area_ratio*100:.1f}%"
        )


def _layers_from_overlap_probe(
    *,
    image_path: str,
    union: list[str],
    iter_dir: Path,
    args,
    score_thresh_for_iter: float,
) -> tuple[list[list[str]], dict]:
    """Run a one-shot "probe" SAM3 pass on the original slide with the full
    union of classes, then partition the union into 2 layers based on the
    actual bbox-overlap structure:

      - Tier 1: classes whose probe bboxes do NOT overlap with any other
        class's probe bbox (cleanly isolated detections) + classes that
        produced no probe detections at all (segmenting them on a clean
        slide can't hurt and may pick them up).
      - Tier 2: classes whose probe bboxes overlap at least one other
        class's bbox (visually competing — segment these LAST, on the
        slide cleaned of the tier-1 masks).

    Returns ``(layers, probe_info)`` where probe_info captures the
    overlap graph + counts for persistence under iter_dir/.
    """
    probe_dir = iter_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_seg = probe_dir / "segmentation"
    probe_bbox = probe_dir / "bbox_components"
    probe_manifest_path = probe_dir / "manifest.json"
    with probe_manifest_path.open("w") as f:
        json.dump([{"image": image_path, "classes": union}], f, indent=2)

    print(f"  Agent E (overlap-probe pass over {len(union)} class(es)) ...")
    rc = agent_e_segment(
        manifest_path=probe_manifest_path,
        output_dir=probe_seg,
        ckpt=args.ckpt,
        base_ckpt=args.base_ckpt,
        conda_python=args.conda_python,
        script_path=Path(args.e_script),
        score_thresh=score_thresh_for_iter,
        max_boxes_per_text=args.max_boxes_per_text,
        device=args.device,
        extra_args=_e_extra_args(args),
        backend=None,  # probe never goes through claude_code part-of-check
        part_of_exchange_dir=None,
        bbox_output_dir=probe_bbox,
    )
    if rc != 0:
        print(f"  [overlap-probe] SAM3 returned {rc}; falling back to single layer.")
        return [list(union)], {"error": f"sam3_rc={rc}"}

    probe_cleaned = _find_cleaned_image(probe_seg)
    if probe_cleaned is None:
        print("  [overlap-probe] no probe metadata; falling back to single layer.")
        return [list(union)], {"error": "no_cleaned_image"}
    probe_meta_path = probe_cleaned.parent / "metadata.json"
    try:
        with probe_meta_path.open() as f:
            probe_meta = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [overlap-probe] could not parse probe metadata ({exc}); falling back.")
        return [list(union)], {"error": "parse_failure"}

    bboxes_per_class: dict[str, list[list[float]]] = {}
    for c in probe_meta.get("components") or []:
        cls = c.get("text_type")
        bb = c.get("bbox_xyxy")
        if cls and isinstance(bb, list) and len(bb) == 4:
            bboxes_per_class.setdefault(str(cls), []).append([float(v) for v in bb])

    # Build overlap graph at the class level.
    classes_with = list(bboxes_per_class.keys())
    conflict_classes: set[str] = set()
    overlap_pairs: list[tuple[str, str]] = []
    for i, ca in enumerate(classes_with):
        for cb in classes_with[i + 1:]:
            overlaps = False
            for ba in bboxes_per_class[ca]:
                if overlaps:
                    break
                for bbb in bboxes_per_class[cb]:
                    if _bboxes_overlap_area(ba, bbb) > 1.0:
                        overlaps = True
                        break
            if overlaps:
                conflict_classes.add(ca)
                conflict_classes.add(cb)
                overlap_pairs.append((ca, cb))

    # Partition union preserving input order.
    isolated_or_empty: list[str] = []
    conflicting: list[str] = []
    for cls in union:
        if cls in bboxes_per_class:
            if cls in conflict_classes:
                conflicting.append(cls)
            else:
                isolated_or_empty.append(cls)
        else:
            # No probe detection — put in tier 1 (cheap retry on clean slide).
            isolated_or_empty.append(cls)

    if not isolated_or_empty or not conflicting:
        # No useful split — keep one layer (everything together).
        layers: list[list[str]] = [list(union)]
    else:
        layers = [isolated_or_empty, conflicting]

    probe_info = {
        "probe_dir": str(probe_dir.relative_to(iter_dir)),
        "classes_with_detections": classes_with,
        "classes_without_detections": [
            cls for cls in union if cls not in bboxes_per_class
        ],
        "overlap_pairs": [list(p) for p in overlap_pairs],
        "conflict_classes": sorted(conflict_classes),
        "isolated_classes": sorted(set(isolated_or_empty) - {
            cls for cls in union if cls not in bboxes_per_class
        }),
    }
    return layers, probe_info


def _run_layered_segmentation(
    *,
    backend,
    image_path: str,
    union: list[str],
    iter_dir: Path,
    seg_dir: Path,
    bbox_dir: Path,
    part_of_dir: Path | None,
    args,
    score_thresh_for_iter: float,
    layers: list[list[str]] | None = None,
    layers_meta_file: str = "layers.json",
) -> tuple[Path, Path, Path, Path]:
    """Run SAM3 once per layer (top→bottom), chaining the cleaned image
    as input. Aggregate per-layer outputs into a single
    ``seg_dir/<image_stem>/`` so downstream code sees a normal SAM3 result.

    If ``layers`` is not provided, falls back to calling Agent O.

    Returns ``(cleaned_path, components_dir, bbox_components_dir, metadata_path)``
    matching the single-call branch.
    """
    if layers is None:
        print(f"  Agent O: ordering {len(union)} class(es) into segmentation layers ...")
        try:
            layers = agent_o_segmentation_layers(backend, image_path, union)
        except Exception as exc:  # noqa: BLE001
            print(f"  [agent_o] failure ({exc}); falling back to single layer.")
            layers = [list(union)]
        if not layers:
            layers = [list(union)]
    print(f"    -> {len(layers)} layer(s):")
    for li, classes in enumerate(layers, 1):
        print(f"       layer {li}: {classes}")

    try:
        with (iter_dir / layers_meta_file).open("w") as f:
            json.dump([{"layer": i + 1, "classes": cs} for i, cs in enumerate(layers)], f, indent=2)
    except OSError:
        pass

    layers_root = iter_dir / "layers"
    layers_root.mkdir(parents=True, exist_ok=True)

    per_layer_results: list[tuple[Path, Path, Path, Path, list[str]]] = []
    current_input = image_path

    for li, classes in enumerate(layers, 1):
        layer_dir = layers_root / f"layer_{li:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_manifest_path = layer_dir / "manifest.json"
        with layer_manifest_path.open("w") as f:
            json.dump([{"image": current_input, "classes": classes}], f, indent=2)

        layer_seg = layer_dir / "segmentation"
        layer_bbox = layer_dir / "bbox_components"
        layer_part_of = (layer_dir / "part_of_exchange") if part_of_dir is not None else None

        print(f"  Agent E (layer {li}/{len(layers)}, classes={classes})")
        rc = agent_e_segment(
            manifest_path=layer_manifest_path,
            output_dir=layer_seg,
            ckpt=args.ckpt,
            base_ckpt=args.base_ckpt,
            conda_python=args.conda_python,
            script_path=Path(args.e_script),
            score_thresh=score_thresh_for_iter,
            max_boxes_per_text=args.max_boxes_per_text,
            device=args.device,
            extra_args=_e_extra_args(args),
            backend=backend if args.part_of_check else None,
            part_of_exchange_dir=layer_part_of,
            bbox_output_dir=layer_bbox,
        )
        if rc != 0:
            raise SystemExit(
                f"Agent E (layer {li}) subprocess exited with code {rc} in {layer_dir}"
            )

        layer_cleaned = _find_cleaned_image(layer_seg)
        if layer_cleaned is None:
            raise SystemExit(
                f"Agent E (layer {li}) did not produce image_cleaned.png under {layer_seg}"
            )
        layer_components_dir = _find_components_dir(layer_cleaned)
        per_layer_results.append(
            (layer_seg, layer_bbox, layer_cleaned, layer_components_dir, classes)
        )
        # Next layer reads this layer's cleaned image.
        current_input = str(layer_cleaned)

    # Aggregate per-layer outputs into seg_dir/<image_stem>/.
    image_stem = Path(image_path).stem
    agg_subdir = seg_dir / image_stem
    agg_components_dir = agg_subdir / "components"
    agg_bbox_subdir = bbox_dir / image_stem
    agg_bbox_components_dir = agg_bbox_subdir / "components"
    agg_components_dir.mkdir(parents=True, exist_ok=True)
    agg_bbox_components_dir.mkdir(parents=True, exist_ok=True)

    aggregated_components: list[dict] = []
    accum_counts: dict[str, int | list] = {}
    next_idx = 1
    name_re = re.compile(r"^component_\d+_(.+)$")

    for li, (layer_seg, layer_bbox, layer_cleaned, layer_components_dir, _classes) in enumerate(
        per_layer_results, 1
    ):
        layer_subdir = layer_cleaned.parent.name
        layer_bbox_components_dir = layer_bbox / layer_subdir / "components"
        layer_meta_path = layer_cleaned.parent / "metadata.json"
        try:
            with layer_meta_path.open() as f:
                layer_meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            layer_meta = {"components": []}

        for k, v in layer_meta.items():
            if k in {"components", "query_image_path", "query_image_size"}:
                continue
            if isinstance(v, list):
                accum_counts[k] = list(accum_counts.get(k) or []) + list(v)
            elif isinstance(v, (int, float)):
                accum_counts[k] = (accum_counts.get(k) or 0) + v

        for comp in layer_meta.get("components") or []:
            orig_rel = comp.get("component_file", "")
            orig_base = Path(orig_rel).name
            orig_full = layer_components_dir / orig_base
            orig_bbox_rel = comp.get("bbox_component_file", "")
            orig_bbox_base = Path(orig_bbox_rel).name if orig_bbox_rel else orig_base
            orig_bbox_full = layer_bbox_components_dir / orig_bbox_base

            m = name_re.match(Path(orig_base).stem)
            class_suffix = m.group(1) if m else "x"
            new_base = f"component_{next_idx:04d}_{class_suffix}.png"
            next_idx += 1

            new_seg_path = agg_components_dir / new_base
            new_bbox_path = agg_bbox_components_dir / new_base
            if orig_full.is_file():
                shutil.copy2(orig_full, new_seg_path)
            if orig_bbox_full.is_file():
                shutil.copy2(orig_bbox_full, new_bbox_path)

            new_comp = dict(comp)
            new_comp["component_file"] = f"components/{new_base}"
            if orig_bbox_rel:
                new_comp["bbox_component_file"] = f"components/{new_base}"
            new_comp["seg_layer"] = li
            aggregated_components.append(new_comp)

    last_cleaned = per_layer_results[-1][2]
    agg_meta = {
        "query_image_path": image_path,
        "components": aggregated_components,
        "seg_layers": [
            {"layer": i + 1, "classes": cs} for i, cs in enumerate(layers)
        ],
        **accum_counts,
    }
    with (agg_subdir / "metadata.json").open("w") as f:
        json.dump(agg_meta, f, indent=2)
    shutil.copy2(last_cleaned, agg_subdir / "image_cleaned.png")

    seg_dir.mkdir(parents=True, exist_ok=True)
    with (seg_dir / "work_items.json").open("w") as f:
        json.dump(
            [{"image": image_path, "classes": union,
              "layered": [list(cs) for cs in layers]}],
            f, indent=2,
        )

    cleaned_agg = agg_subdir / "image_cleaned.png"
    return cleaned_agg, agg_components_dir, agg_bbox_components_dir, agg_subdir / "metadata.json"


def _run_one_iteration(
    backend_name: str,
    api_key: str | None,
    model: str | None,
    iter_dir: Path,
    image_path: str,
    class_list: list[str],
    args,
    retry_classes: list[str] | None = None,
    use_retry_score_thresh: bool = False,
) -> dict:
    """Run A-E-F on a single image, in its own iter_dir. Returns iter record.

    ``retry_classes`` (modification C): when non-empty, classes in this list
    are appended to the A+B+C union before E. This is how an iter that
    follows a missed-class iter explicitly re-asks SAM3 for the classes it
    failed to ground last time, plus any class names that Agent F's
    ``remaining_components`` text maps to.

    ``use_retry_score_thresh`` (modification C): when True, E is invoked with
    ``--score_thresh = args.score_thresh_retry`` (typically lower) so the
    re-prompted classes have a better chance of yielding candidates.
    """
    iter_dir.mkdir(parents=True, exist_ok=True)
    exchange_dir = iter_dir / "exchange"
    if backend_name == "claude_code":
        exchange_dir.mkdir(parents=True, exist_ok=True)
    backend = build_backend(
        backend_name,
        api_key=api_key,
        exchange_dir=str(exchange_dir) if backend_name == "claude_code" else None,
        model=model,
    )
    # H3 cascade: cheap screen model for F-validity (same backend transport).
    screen_backend = None
    if getattr(args, "f_validity_cascade", False):
        screen_backend = build_backend(
            backend_name,
            api_key=api_key,
            exchange_dir=str(exchange_dir) if backend_name == "claude_code" else None,
            model=args.f_validity_screen_model,
        )
    # H11 wave 1: shared screen for H-validate/M2/M3 (no-op unless --agent_cascade).
    _ensure_agent_screen_backend(
        args, backend_name, api_key,
        exchange_dir=str(exchange_dir) if backend_name == "claude_code" else None,
    )
    # H11 wave 2: judgment screen (F-cleanup/H) + cheap A/B/C backend.
    judgment_screen = _ensure_judgment_screen_backend(
        args, backend_name, api_key,
        exchange_dir=str(exchange_dir) if backend_name == "claude_code" else None,
    )
    abc_backend = backend
    if getattr(args, "abc_model", None):
        abc_backend = build_backend(
            backend_name, api_key=api_key,
            exchange_dir=str(exchange_dir) if backend_name == "claude_code" else None,
            model=args.abc_model,
        )

    print(f"  Agent A: describing components...")
    phrases = agent_a_describe(abc_backend, image_path)
    print(f"    -> {len(phrases)} phrases: {phrases}")

    print(f"  Agent B: mapping phrases to taxonomy...")
    if getattr(args, "fold_c_into_b", False):
        # H4: one image-grounded call does B's mapping + C's selection.
        from decomposition.agents import agent_bc_combined
        b_classes, c_classes = agent_bc_combined(backend, image_path, phrases, class_list)
        print(f"    -> B(mapped)={b_classes}")
        print(f"  Agent C (folded into B call): -> {c_classes}")
    else:
        b_classes = agent_b_map_to_classes(abc_backend, phrases, class_list)
        print(f"    -> {b_classes}")

        print(f"  Agent C: direct class selection...")
        c_classes = agent_c_direct_select(abc_backend, image_path, class_list)
        print(f"    -> {c_classes}")

    union = agent_d_union(b_classes, c_classes, class_list)
    print(f"  Agent D union: {union}")

    # Modification C: merge in retry classes (missed in prior iter +
    # B-mapped from prior F.remaining_components).
    retry_added: list[str] = []
    if retry_classes:
        union_lower = {c.lower() for c in union}
        valid_lower = {c.lower(): c for c in class_list}
        for c in retry_classes:
            cl = str(c).strip().lower()
            if not cl or cl in union_lower:
                continue
            canonical = valid_lower.get(cl)
            if canonical is None:
                continue
            union.append(canonical)
            union_lower.add(cl)
            retry_added.append(canonical)
        # Re-order to taxonomy order for stable output.
        union = agent_d_union(union, [], class_list)
        if retry_added:
            print(f"  Retry classes added to union: {retry_added}")

    per_image = {
        "input_image": image_path,
        "phrases": phrases,
        "b_classes": b_classes,
        "c_classes": c_classes,
        "union_classes": union,
        "retry_classes_added": retry_added,
        "score_thresh_used": (
            float(args.score_thresh_retry) if use_retry_score_thresh else float(args.score_thresh)
        ),
    }
    with (iter_dir / "per_image.json").open("w") as f:
        json.dump([per_image], f, indent=2)

    if not union:
        print("  Empty class union — skipping Agent E and treating iter as converged.")
        empty_review = {
            "needs_rerun": False,
            "reason": "empty class union: no taxonomy classes proposed for this iter",
            "remaining_components": [],
            "invalid_components": [],
            "component_verdicts": [],
            "ocr_results": [],
            "review_source": "skipped_empty_union",
            "pruned": {"removed_files": [], "removed_bbox_files": [], "removed_metadata_entries": []},
        }
        with (iter_dir / "review.json").open("w") as f:
            json.dump(empty_review, f, indent=2)
        return {
            "iter_dir": str(iter_dir),
            "input_image": image_path,
            "cleaned_image": image_path,
            "per_image": per_image,
            "review": empty_review,
        }

    seg_dir = iter_dir / "segmentation"
    bbox_dir = iter_dir / "bbox_components"
    part_of_dir = iter_dir / "part_of_exchange" if args.part_of_check else None
    score_thresh_for_iter = (
        float(args.score_thresh_retry) if use_retry_score_thresh else float(args.score_thresh)
    )

    use_overlap_probe = bool(getattr(args, "seg_overlap_probe", False)) and len(union) >= 2
    use_seg_layers = bool(getattr(args, "seg_layers", False)) and len(union) >= 2

    if use_overlap_probe:
        probe_layers, probe_info = _layers_from_overlap_probe(
            image_path=image_path,
            union=union,
            iter_dir=iter_dir,
            args=args,
            score_thresh_for_iter=score_thresh_for_iter,
        )
        try:
            with (iter_dir / "overlap_probe.json").open("w") as f:
                json.dump(probe_info, f, indent=2)
        except OSError:
            pass
        cleaned, components_dir, bbox_components_dir, metadata_path = (
            _run_layered_segmentation(
                backend=backend,
                image_path=image_path,
                union=union,
                iter_dir=iter_dir,
                seg_dir=seg_dir,
                bbox_dir=bbox_dir,
                part_of_dir=part_of_dir,
                args=args,
                score_thresh_for_iter=score_thresh_for_iter,
                layers=probe_layers,
                layers_meta_file="overlap_probe_layers.json",
            )
        )
    elif use_seg_layers:
        cleaned, components_dir, bbox_components_dir, metadata_path = (
            _run_layered_segmentation(
                backend=backend,
                image_path=image_path,
                union=union,
                iter_dir=iter_dir,
                seg_dir=seg_dir,
                bbox_dir=bbox_dir,
                part_of_dir=part_of_dir,
                args=args,
                score_thresh_for_iter=score_thresh_for_iter,
                layers_meta_file="agent_o_layers.json",
            )
        )
    else:
        manifest = [{"image": image_path, "classes": union}]
        manifest_path = iter_dir / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)

        rc = agent_e_segment(
            manifest_path=manifest_path,
            output_dir=seg_dir,
            ckpt=args.ckpt,
            base_ckpt=args.base_ckpt,
            conda_python=args.conda_python,
            script_path=Path(args.e_script),
            score_thresh=score_thresh_for_iter,
            max_boxes_per_text=args.max_boxes_per_text,
            device=args.device,
            extra_args=_e_extra_args(args),
            backend=backend if args.part_of_check else None,
            part_of_exchange_dir=part_of_dir,
            bbox_output_dir=bbox_dir,
        )
        if rc != 0:
            raise SystemExit(f"Agent E subprocess exited with code {rc} in {iter_dir}")

        cleaned = _find_cleaned_image(seg_dir)
        if cleaned is None:
            raise SystemExit(f"Agent E did not produce image_cleaned.png under {seg_dir}")

        components_dir = _find_components_dir(cleaned)
        bbox_components_dir = _find_bbox_components_dir(bbox_dir, cleaned)
        metadata_path = cleaned.parent / "metadata.json"

    # v2.9 safety net: drop SAM3 detections whose RGBA-alpha mask is
    # SPARSE relative to its bbox AND the bbox is large enough to matter.
    # Targets catch-all hallucinations (e.g. SAM3 detecting "data
    # visualization" with a 90%-of-slide bbox around a thin curve mask)
    # without harming legit full-slide dense images (wallpaper photos,
    # full-slide diagrams) whose masks are dense.
    min_density = float(getattr(args, "min_mask_density", 0.0))
    if min_density > 0.0:
        _drop_sparse_detections(
            metadata_path=metadata_path,
            components_dir=components_dir,
            bbox_components_dir=bbox_components_dir,
            min_mask_density=min_density,
            min_bbox_area_for_check=float(
                getattr(args, "min_bbox_area_for_density_check", 0.15)
            ),
            original_image_path=image_path,
        )

    # Mask-based overlap carve (v2.7): for every pair of fine components
    # in this iter whose bboxes overlap, use SAM3 mask coverage in the
    # overlap region to detect "lying" bboxes. A bbox extending into
    # another component's area without actual mask content is carved by
    # the other's bbox (polygon = bbox(A) - bbox(B), bbox shrunk to the
    # polygon's bounds, bbox_components/ crop regenerated as polygon-cut
    # RGBA when non-rectangular). Runs BEFORE Agent F so the validator
    # sees the carved view and can correctly identify text-only crops.
    if getattr(args, "mask_overlap_carve", True):
        carve_result = _apply_mask_overlap_carve(
            components_dir=components_dir,
            bbox_components_dir=bbox_components_dir,
            metadata_path=metadata_path,
            original_image_path=image_path,
            backend=backend,
            overlay_dir=iter_dir / "overlap_overlays",
            min_overlap_pixels=float(getattr(args, "mask_carve_min_overlap_pixels", 100.0)),
            mask_coverage_threshold=float(getattr(args, "mask_carve_threshold", 0.05)),
        )
        if carve_result["carved"]:
            print(
                f"  Overlap-arbitrate carve: rewrote {len(carve_result['carved'])} "
                f"bbox(es): "
                + ", ".join(
                    f"{c['filename']} ← {c['carved_by']}"
                    for c in carve_result["carved"]
                )
            )

    # Review is done on the BBOX version (opaque rectangle) so the VLM sees
    # the real pixels + can OCR text-only components. Fall back to the
    # mask-cut components only if bbox versions are missing.
    bbox_paths = _list_component_images(bbox_components_dir)
    seg_paths = _list_component_images(components_dir)
    review_source = "bbox" if bbox_paths else "segmentation"
    review_paths = bbox_paths if bbox_paths else seg_paths

    # Modification B: render an overlay (original slide + red rectangles
    # over every extracted bbox) so Agent F's cleanup verdict can tell
    # "missed component" apart from "mask-cleanup residue inside an
    # already-extracted region".
    overlay_path: str | None = None
    extracted_bboxes_for_overlay: list[list[float]] = []
    # v2.8: per-component slide-context overlay map (filename -> bbox).
    bbox_by_filename: dict[str, list[float]] = {}
    try:
        if metadata_path.is_file():
            with metadata_path.open() as f:
                meta_for_overlay = json.load(f)
            for entry in meta_for_overlay.get("components") or []:
                bb = entry.get("bbox_xyxy")
                if isinstance(bb, list) and len(bb) == 4:
                    bb_f = [float(v) for v in bb]
                    extracted_bboxes_for_overlay.append(bb_f)
                    cf = str(entry.get("component_file", ""))
                    if cf:
                        bbox_by_filename[Path(cf).name] = bb_f
    except (OSError, json.JSONDecodeError):
        extracted_bboxes_for_overlay = []
    if extracted_bboxes_for_overlay:
        overlay_path = str(iter_dir / "bbox_overlay.png")
        try:
            render_bbox_overlay(
                image_path, extracted_bboxes_for_overlay, overlay_path
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] could not render bbox overlay: {exc}")
            overlay_path = None

    # v2.8 (opt-in via --f_dual_image): per-component slide-context overlay
    # (one image per component, showing where its bbox sits on the full
    # slide). Disambiguates narrow / sparse character crops (brackets,
    # vertical bars, single mathematical letters) whose meaning isn't clear
    # from the bbox crop alone. Default off → single-image F validity.
    slide_context_paths: list[str | None] = []
    if getattr(args, "f_dual_image", False):
        f_context_dir = iter_dir / "f_context_overlays"
        for p in review_paths:
            fname = Path(str(p)).name
            bb = bbox_by_filename.get(fname)
            if not bb:
                slide_context_paths.append(None)
                continue
            out_p = f_context_dir / fname
            try:
                render_bbox_overlay(image_path, [bb], str(out_p), width_px=6)
                slide_context_paths.append(str(out_p))
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] could not render F context overlay for {fname}: {exc}")
                slide_context_paths.append(None)

    print(
        f"  Agent F: reviewing {cleaned} + {len(review_paths)} "
        f"{review_source}-crop component(s)"
        + (f" (with bbox overlay over {len(extracted_bboxes_for_overlay)} bboxes)" if overlay_path else "")
        + (f" + {sum(1 for c in slide_context_paths if c)} per-component context overlays" if any(slide_context_paths) else "")
        + " ..."
    )
    review = agent_f_review(
        backend,
        str(cleaned),
        component_paths=[str(p) for p in review_paths],
        bbox_overlay_path=overlay_path,
        slide_context_paths=slide_context_paths,
        screen_backend=screen_backend,
        cleanup_screen_backend=judgment_screen,
    )
    review["review_source"] = review_source
    if overlay_path:
        review["bbox_overlay_path"] = overlay_path
    print(
        f"    -> needs_rerun={review['needs_rerun']} "
        f"invalid={len(review.get('invalid_components', []))} "
        f"text_only_ocr={len(review.get('ocr_results', []))} "
        f"reason={review['reason']!r}"
    )

    pruned = {"removed_files": [], "removed_bbox_files": [], "removed_metadata_entries": []}
    invalid = review.get("invalid_components") or []
    if invalid:
        pruned = _prune_invalid_components(
            components_dir,
            metadata_path,
            invalid,
            bbox_components_dir=bbox_components_dir if bbox_paths else None,
        )
        print(
            f"    pruned {len(pruned['removed_files'])} segmentation crop(s) + "
            f"{len(pruned.get('removed_bbox_files', []))} bbox crop(s): "
            f"{pruned['removed_files']}"
        )
    review["pruned"] = pruned

    # Tighten any text-only crop Agent F flagged as bbox_is_tight=false.
    # Uses the tight_bbox_xyxy field that Agent E pre-computed from the
    # dilated predicted mask. Re-crops opaque RGB from the original slide.
    tighten_result = _apply_bbox_tightening(
        components_dir=components_dir,
        bbox_components_dir=bbox_components_dir if bbox_paths else None,
        metadata_path=metadata_path,
        original_image_path=image_path,
        component_verdicts=review.get("component_verdicts") or [],
    )
    if tighten_result["tightened"]:
        print(
            f"    tightened {len(tighten_result['tightened'])} text bbox(es): "
            f"{[t['filename'] for t in tighten_result['tightened']]}"
        )
    review["bbox_tightening"] = tighten_result

    # Expand any text-only crop Agent F flagged as bbox_is_too_small=true.
    expand_result = _apply_bbox_expansion(
        components_dir=components_dir,
        bbox_components_dir=bbox_components_dir if bbox_paths else None,
        metadata_path=metadata_path,
        original_image_path=image_path,
        component_verdicts=review.get("component_verdicts") or [],
    )
    if expand_result["expanded"]:
        print(
            f"    expanded {len(expand_result['expanded'])} text bbox(es): "
            f"{[e['filename'] for e in expand_result['expanded']]}"
        )
    review["bbox_expansion"] = expand_result

    adjustments_made = bool(tighten_result.get("tightened") or expand_result.get("expanded"))
    if adjustments_made:
        print(f" Agent F: re-reviewing after {len(tighten_result['tightened'])} tighten + {len(expand_result['expanded'])} expand...")
        review_initial = review

        bbox_paths2 = _list_component_images(bbox_components_dir)
        seg_paths2 = _list_component_images(components_dir)
        review_paths2 = bbox_paths2 if bbox_paths2 else seg_paths2

        extracted_bboxes2, bbox_by_name2 = [], {}
        try:
            with metadata_path.open() as f:
                meta2 = json.load(f)
            for e in meta2.get("components") or []:
                bb = e.get("bbox_xyxy")
                if isinstance(bb, list) and len(bb)==4:
                    bb_f = [float(v) for v in bb]
                    extracted_bboxes2.append(bb_f)
                    cf = Path(str(e.get("component_file",""))).name
                    if cf: bbox_by_name2[cf] = bb_f
        except: pass

        overlay2 = str(iter_dir / "bbox_overlay_post_adjust.png")
        if extracted_bboxes2:
            try: render_bbox_overlay(image_path, extracted_bboxes2, overlay2)
            except: overlay2 = None

        context2 = []
        if getattr(args, "f_dual_image", False):
            ctx_dir = iter_dir / "f_context_overlays_post"
            for p in review_paths2:
                fn = Path(str(p)).name
                bb = bbox_by_name2.get(fn)
                if bb:
                    out = ctx_dir / fn
                    try: render_bbox_overlay(image_path, [bb], str(out), width_px=6); context2.append(str(out))
                    except: context2.append(None)
                else: context2.append(None)

        review2 = agent_f_review(
            backend, str(cleaned),
            component_paths=[str(p) for p in review_paths2],
            bbox_overlay_path=overlay2,
            slide_context_paths=context2,
            screen_backend=screen_backend,
            cleanup_screen_backend=judgment_screen,
        )
        review2["review_source"] = review_source
        if overlay2: review2["bbox_overlay_path"] = overlay2

        invalid2 = review2.get("invalid_components") or []
        pruned2 = _prune_invalid_components(
            components_dir, metadata_path, invalid2,
            bbox_components_dir if bbox_paths2 else None
        ) if invalid2 else {"removed_files":[],"removed_bbox_files":[],"removed_metadata_entries":[]}
        review2["pruned"] = pruned2

        review2["bbox_tightening"] = tighten_result
        review2["bbox_expansion"] = expand_result
        review2["pre_adjustment_review"] = {
            "needs_rerun": review_initial.get("needs_rerun"),
            "reason": review_initial.get("reason"),
            "invalid_components": review_initial.get("invalid_components"),
        }
        review = review2
    else:
        review["bbox_tightening"] = tighten_result
        review["bbox_expansion"] = expand_result

    with (iter_dir / "review.json").open("w") as f:
        json.dump(review, f, indent=2)

    return {
        "iter_dir": str(iter_dir),
        "input_image": image_path,
        "cleaned_image": str(cleaned),
        "per_image": per_image,
        "review": review,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", help="Single image path.")
    ap.add_argument("--images", help="Directory of images OR JSON list of paths.")
    ap.add_argument("--class_list",
                    default=str(REPO / "data" / "sam3_text_types_306.json"),
                    help="Taxonomy JSON (defaults to the 306-class list).")
    ap.add_argument("--backend", required=True,
                    choices=["openai", "gemini", "claude", "claude_cli", "claude_code"])
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--max_review_iters", type=int, default=3,
                    help="Maximum A-E-F iterations per image (1 = no rerun).")

    # Agent E knobs
    ap.add_argument("--ckpt",
                    default=str(REPO / "sam3" / "checkpoints" / "sam3_slideforge.pt"),
                    help="Fine-tuned SlideForge component-detector decoder "
                         "checkpoint (see scripts/download_checkpoints.sh).")
    ap.add_argument("--base_ckpt",
                    default=str(REPO / "sam3" / "checkpoints" / "sam3.pt"),
                    help="Base SAM3 checkpoint (facebook/sam3 on Hugging Face).")
    ap.add_argument("--conda_python", default=DEFAULT_CONDA_PY)
    ap.add_argument("--e_script", default=str(DEFAULT_E_SCRIPT))
    ap.add_argument("--score_thresh", type=float, default=0.3)
    ap.add_argument("--score_thresh_retry", type=float, default=0.15,
                    help="Score threshold used for retry iters (modification C). "
                         "After iter 0, classes that previously yielded zero "
                         "components and classes inferred from F.remaining_components "
                         "are re-prompted with this lower threshold so SAM3 has a "
                         "better chance of grounding them. Default 0.15.")
    ap.add_argument("--nms_thresh", type=float, default=0.5,
                    help="Only used for infer_remove_components_from_classlist.py")
    ap.add_argument("--min_overlap_area", type=float, default=1.0,
                    help="Only used for infer_remove_components_overlap_priority.py")
    ap.add_argument("--max_boxes_per_text", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--lazy_segment", action=argparse.BooleanOptionalAction, default=True,
                    help="Only used for infer_remove_components_overlap_priority.py. "
                         "Default: on. Pass --no-lazy_segment to disable.")
    ap.add_argument("--part_of_check", action="store_true",
                    help="Enable Agent G: for cross-label overlaps where A is smaller than B, "
                         "ask the VLM backend whether A is a sub-part of B. If yes, suppress A. "
                         "Only supported by infer_remove_components_overlap_priority.py.")
    ap.add_argument("--part_of_overlap_ratio", type=float, default=0.3,
                    help="Only trigger Agent G when intersection(A,B)/A.area >= this threshold.")
    ap.add_argument("--cross_iter_dedup_ratio", type=float, default=0.5,
                    help="After the iteration loop, drop later-iter components whose "
                         "bbox has intersection/later_area >= this threshold with any "
                         "earlier-iter kept component. Set to 0 to disable.")
    # Mask-repair knobs (modification A in pipeline-v2.2): make image_cleaned
    # actually clean by repairing the predicted mask before whitening.
    ap.add_argument("--mask_repair_fill_holes",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Fill mask-internal holes before whitening (default on).")
    ap.add_argument("--mask_repair_closing_kernel", type=int, default=5,
                    help="binary_closing square kernel size in px (0 to disable). Default 5.")
    ap.add_argument("--mask_repair_dilation_px", type=int, default=2,
                    help="Dilation iterations in px (0 to disable). Default 2. The "
                         "expanded mask is hard-clipped to the candidate's own bbox.")
    ap.add_argument("--tight_bbox_dilation_px", type=int, default=3,
                    help="Dilation iterations in px applied to the predicted mask "
                         "before computing tight_bbox_xyxy. Used by the bbox-"
                         "tightening pass when Agent F flags a text crop as not "
                         "tight. Default 3.")
    ap.add_argument("--bbox_tighten",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="v2.5 behaviour at Agent E time: for lazy-segment / "
                         "bbox-crop components, swap the saved bbox to the "
                         "dilated mask's tight bbox (and recrop). Pass "
                         "--no-bbox_tighten to disable this — useful when the "
                         "tight swap shrinks the bbox so far that Agent F "
                         "judges the crop as 'empty frame outline' and "
                         "invalidates an otherwise valid component (e.g. "
                         "script math letters whose SAM3 mask is a thin "
                         "outline rather than a solid blob). Default on.")
    ap.add_argument("--merge_max_area_frac", type=float, default=1.0,
                    help="H10: reject Agent H merge groups whose merged bbox "
                         "exceeds this fraction of the slide area (members "
                         "stay as fine components). 1.0 = off (default).")
    ap.add_argument("--merge_max_members", type=int, default=0,
                    help="H10: reject Agent H merge groups with more than "
                         "this many members. 0 = off (default).")
    ap.add_argument("--merge_area_min_members", type=int, default=1,
                    help="H10 (stack-v2): the area cap only applies to groups "
                         "with at least this many members — protects legitimate "
                         "2-4-member big merges (diagram+caption) while "
                         "rejecting many-member giant absorbers. Default 1 "
                         "(area cap applies to all groups, v1 behavior).")
    ap.add_argument("--fold_c_into_b",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="H4: replace the separate B (phrase mapping) and C "
                         "(direct selection) calls with ONE image-grounded "
                         "combined call per iteration. Default off.")
    ap.add_argument("--resume",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Skip slides whose final/metadata.json already exists "
                         "in the run dir (crash recovery). Default off.")
    ap.add_argument("--judgment_cascade",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="H11 wave 2: route F-cleanup, H-find-missed and "
                         "H-merge through a mid-tier screen model; escalate "
                         "needs_rerun verdicts and low confidence to the main "
                         "model. Default off.")
    ap.add_argument("--judgment_screen_model", default="claude-sonnet-4-6",
                    help="Screen model for --judgment_cascade.")
    ap.add_argument("--abc_model", default=None,
                    help="H11 wave 2: model override for agents A/B/C "
                         "(mechanical describe/map/select). Default None = "
                         "main model.")
    ap.add_argument("--agent_cascade",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="H11 wave 1: route H-validate-missed, M2 and M3 "
                         "through the cheap screen model with opus escalation "
                         "on low confidence (H-validate additionally escalates "
                         "every 'invalid' verdict). Default off.")
    ap.add_argument("--f_validity_cascade",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="H3: route per-crop F-validity through a cheap screen "
                         "model first (interior-ink guard -> screen -> escalate "
                         "to the main model on invalid/low-confidence). "
                         "Default off (byte-identical to v2.5.2 behavior).")
    ap.add_argument("--f_validity_screen_model", default="claude-haiku-4-5-20251001",
                    help="Screen model for --f_validity_cascade.")
    ap.add_argument("--f_dual_image",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Send Agent F per-component validity TWO images: "
                         "(1) the bbox crop (as usual), (2) the full slide with "
                         "a red rectangle outlining where this bbox sits. The "
                         "second image disambiguates narrow / sparse crops "
                         "(brackets, vertical bars, single math symbols) whose "
                         "role is only obvious in slide context. Default off "
                         "(single image — backwards-compatible).")
    # Agent H layout review (v2.3): one VLM pass over all surviving bboxes
    # to merge fragments into one semantic component and surface missed regions.
    ap.add_argument("--layout_review",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Run Agent H to merge fragmented bbox groups + surface "
                         "missed regions. Output goes to <image_dir>/final/. "
                         "Default on. Pass --no-layout_review to disable.")
    # Polygon refinement (v2.6): per-overlap Agent M2 decision after merge
    ap.add_argument("--polygon_refine",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="After Agent H merge, for every non-member component "
                         "whose bbox overlaps a merged group, ask Agent M2 "
                         "whether to absorb (union polygon) or carve out "
                         "(diff polygon). Carve falls back to rect when it "
                         "would create a hole. Default on.")
    ap.add_argument("--polygon_refine_rounds", type=int, default=3,
                    help="Max rounds of polygon refinement per merged group "
                         "(fixed-point until no overlap changes). Default 3.")
    # Mask-based overlap carve (v2.7): per-iter pre-Agent-F bbox correction
    ap.add_argument("--mask_overlap_carve",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Per-iter: for every pair of fine components with "
                         "overlapping bboxes, use SAM3 mask coverage in the "
                         "overlap to detect bboxes that 'lie' about owning "
                         "another component's region. Carve those by the "
                         "other's bbox (shrunk bbox + optional polygon, "
                         "regenerated bbox crop as RGBA polygon-cut). Runs "
                         "BEFORE Agent F validity so the validator sees the "
                         "carved view. Default on.")
    ap.add_argument("--mask_carve_threshold", type=float, default=0.05,
                    help="Mask coverage threshold (0..1). If a fine "
                         "component's mask covers less than this fraction "
                         "of its overlap with another bbox, carve. "
                         "Default 0.05.")
    ap.add_argument("--max_seg_area_ratio", type=float, default=1.0,
                    help="Opt-in safety net: drop SAM3 detections whose "
                         "bbox covers more than this fraction of the "
                         "slide. Default 1.0 (off) since a legit full-"
                         "slide image / diagram is sometimes the whole "
                         "slide. Set e.g. 0.65 to defend against "
                         "obvious catch-all hallucinations.")
    ap.add_argument("--min_mask_density", type=float, default=0.0,
                    help="Drop SAM3 detections whose RGBA-alpha mask "
                         "covers less than this fraction of their bbox "
                         "AND whose bbox is at least "
                         "--min_bbox_area_for_density_check of the slide. "
                         "Defaults to 0.08 (8%%): catches catch-all "
                         "hallucinations where SAM3 draws a slide-wide "
                         "bbox around a thin curve / sparse mask. Real "
                         "components have dense masks (30%%+). Set 0 to "
                         "disable.")
    ap.add_argument("--missed_via_point_prompt",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Alternative Agent H phase 1 path: VLM proposes "
                         "anchor POINTS (x, y) instead of bboxes, then "
                         "SAM3's SAM1-task interactive predictor grounds "
                         "each point into a real mask + bbox. More robust "
                         "than VLM-estimated bboxes which often hallucinate "
                         "empty regions. Uses infer_point_prompt.py as a "
                         "subprocess. Default off.")
    ap.add_argument("--missed_vlm_review",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Per-proposal VLM QA check on every Agent H "
                         "phase 1 missed region: VLM is shown the bbox "
                         "crop + the slide with bbox outlined in red and "
                         "asked whether this is a real, useful component "
                         "or an inaccurate/useless proposal. Invalid "
                         "proposals are dropped. Default on. Pass "
                         "--no-missed_vlm_review to disable.")
    ap.add_argument("--missed_min_pixel_std", type=float, default=0.0,
                    help="Optional cheap pre-filter before VLM review: "
                         "drop Agent H phase 1 'missed regions' whose "
                         "pixel content on the original slide has max "
                         "per-channel stddev below this threshold (on "
                         "0-255 RGB). Default 0 (off) since the VLM "
                         "review is canonical. Set e.g. 8.0 to drop "
                         "obvious blanks before paying VLM cost.")
    ap.add_argument("--min_bbox_area_for_density_check", type=float, default=0.15,
                    help="Density check only fires on detections whose "
                         "bbox covers at least this fraction of the "
                         "slide. Default 0.15 (15%%). Small components "
                         "(thin axes, narrow arrows) are exempt — they "
                         "can be legit sparse-mask detections.")
    # Overlap-probe layered segmentation (v2.10, opt-in via --seg_overlap_probe).
    ap.add_argument("--seg_overlap_probe",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Empirical layered seg: run a one-shot probe "
                         "SAM3 pass with all union classes, compute "
                         "bbox overlap structure, partition into 2 "
                         "tiers (no-overlap classes first, conflicting "
                         "classes second), then re-run SAM3 layered "
                         "across the 2 tiers. No VLM call for layering "
                         "decisions (everything is derived from actual "
                         "SAM3 bbox geometry). Default off. Mutually "
                         "exclusive with --seg_layers (this one takes "
                         "precedence).")
    # Agent O layered segmentation (v2.9, opt-in via --seg_layers).
    ap.add_argument("--seg_layers",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Run an Agent O ordering pass before each iter's "
                         "SAM3 call, splitting the class union into 2-4 "
                         "ordered layers. SAM3 is called once per layer, "
                         "each layer running on the cleaned image after "
                         "the previous layer's masks are removed. Helps on "
                         "slides where a wide / loose-bordered class's "
                         "mask absorbs pixels of a smaller foreground "
                         "class. Default off — single SAM3 call per iter.")

    args = ap.parse_args()

    images = _resolve_images(args)
    if not images:
        raise SystemExit("No images resolved.")
    print(f"Pipeline will process {len(images)} image(s). max_review_iters={args.max_review_iters}")

    with open(args.class_list) as f:
        class_list = [str(x).strip() for x in json.load(f) if str(x).strip()]
    print(f"Taxonomy has {len(class_list)} classes.")

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Instrumentation: default the per-call VLM usage log into the run dir.
    if not os.environ.get("VLM_USAGE_LOG"):
        os.environ["VLM_USAGE_LOG"] = str((run_dir / "vlm_usage.jsonl").resolve())
    from backends import log_usage as _log_usage_marker

    overall_summary = []
    for img_idx, orig_image in enumerate(images):
        print(f"\n[{img_idx+1}/{len(images)}] {orig_image}")
        slide_t0 = time.time()
        _log_usage_marker({"marker": "slide_start", "image": str(orig_image), "idx": img_idx})
        stem = Path(orig_image).stem
        parent = Path(orig_image).parent.name
        safe = f"{img_idx:04d}_{parent}_{stem}"
        image_dir = run_dir / safe

        # --resume: skip slides already completed (final metadata exists), so a
        # crashed run (e.g. an extended CLI outage) can be relaunched without
        # redoing finished slides. Off by default.
        if getattr(args, "resume", False) and (image_dir / "final" / "metadata.json").is_file():
            print(f"  [resume] final/metadata.json exists — skipping {safe}")
            _log_usage_marker({"marker": "slide_skipped_resume", "image": str(orig_image), "idx": img_idx})
            continue

        iters = []
        current_image = orig_image
        for iter_i in range(args.max_review_iters):
            iter_dir = image_dir / f"iter_{iter_i:02d}"
            # Modification C: from iter 1 onward, build a retry-class list
            # from the previous iter's missed classes + F.remaining_components
            # and use the lower retry score_thresh.
            retry_classes: list[str] = []
            use_retry_thresh = False
            if iter_i > 0 and iters:
                retry_classes = _retry_classes_from_prev(
                    iters[-1],
                    backend_name=args.backend,
                    api_key=args.api_key,
                    model=args.model,
                    class_list=class_list,
                    iter_dir_for_backend=iter_dir,
                )
                use_retry_thresh = True
                print(
                    f"  Retry iter {iter_i}: prior-missed/remaining classes = "
                    f"{retry_classes} (score_thresh={args.score_thresh_retry})"
                )
            print(f"\n  --- iter {iter_i} (input={current_image}) ---")
            record = _run_one_iteration(
                backend_name=args.backend,
                api_key=args.api_key,
                model=args.model,
                iter_dir=iter_dir,
                image_path=current_image,
                class_list=class_list,
                args=args,
                retry_classes=retry_classes,
                use_retry_score_thresh=use_retry_thresh,
            )
            iters.append(record)
            if not record["review"]["needs_rerun"]:
                print(f"  Agent F: cleaned, stopping at iter {iter_i}.")
                break

            # Modification D: terminate the loop if this iter produced zero
            # surviving components (after F's per-component pruning). When
            # that happens we cannot make progress no matter how many more
            # iters we run — same SAM3, same image, same blind spot.
            num_components_this_iter = len(_extracted_classes_from_record(record))
            seg_dir_this_iter = Path(record.get("cleaned_image") or "").parent
            num_files_this_iter = 0
            if seg_dir_this_iter.exists():
                num_files_this_iter = len(
                    list((seg_dir_this_iter / "components").glob("component_*.png"))
                ) if (seg_dir_this_iter / "components").is_dir() else 0
            if iter_i > 0 and num_files_this_iter == 0:
                print(
                    f"  Termination guard (D): iter {iter_i} produced 0 components — "
                    "SAM3 cannot recover the missing content, stopping early."
                )
                break

            current_image = record["cleaned_image"]
            if iter_i == args.max_review_iters - 1:
                print(f"  Reached max_review_iters={args.max_review_iters} without 'clean' verdict.")

        cross_iter = _cross_iter_dedup(iters, args.cross_iter_dedup_ratio)
        if cross_iter["removed"]:
            print(
                f"  Cross-iter dedup: removed {len(cross_iter['removed'])} "
                f"later-iter component(s) overlapping earlier kept ones."
            )

        # Layout review (Agent H, v2.3): two-phase VLM pass over the
        # full extracted layout.
        #
        # Phase 1 — find_missed: H sees an overlay of only the SAM3-
        # extracted bboxes and lists substantive content that fell
        # OUTSIDE every red rectangle. No merge decisions in this call.
        #
        # Phase 2 (pipeline-side) — materialise: each missed region is
        # saved as a first-class fine component (missed_NNNN_<class>.png)
        # with a known filename in the final components directory.
        #
        # Phase 3 — merge: H sees a re-rendered overlay containing every
        # current fine component (SAM3 + the freshly materialised missed
        # regions, all drawn red) and decides which groups should be
        # combined into one semantic super-component. Members can be any
        # filename — there is no SAM3-vs-missed asymmetry at this stage.
        h_backend = None  # used by layout_review + post-merge re-OCR
        layout_review_result: dict | None = None
        materialised_missed: list[dict] = []
        if args.layout_review:
            valid_for_review = _collect_valid_components(iters)
            if valid_for_review:
                # Phase 1 overlay: SAM3-only bboxes.
                review_overlay_path = image_dir / "review_overlay.png"
                _build_consolidated_overlay(
                    orig_image,
                    [
                        {"bbox_xyxy": c["bbox_xyxy"], "granularity": "fine", "source": "sam3"}
                        for c in valid_for_review
                    ],
                    review_overlay_path,
                    color_by_granularity=False,
                )
                phase1_summary = [
                    {
                        "filename": f"iter{c['iter_idx']:02d}_{c['original_filename']}",
                        "class": c["text_type"],
                        "bbox": [round(float(v), 1) for v in (c["bbox_xyxy"] or [])],
                    }
                    for c in valid_for_review
                ]
                # Side map of seg-mask paths keyed by the same filename
                # used in phase1/phase3 summary. Kept OUT of the summary
                # passed to Agent H so file paths don't bloat the prompt.
                seg_path_by_filename: dict[str, str | None] = {
                    f"iter{c['iter_idx']:02d}_{c['original_filename']}": c.get("seg_path")
                    for c in valid_for_review
                }
                missed_regions: list[dict] = []
                merge_groups: list[dict] = []
                rejected_merge_groups: list[dict] = []
                try:
                    h_backend = build_backend(
                        args.backend,
                        api_key=args.api_key,
                        exchange_dir=str(image_dir / "layout_review_exchange")
                        if args.backend == "claude_code"
                        else None,
                        model=args.model,
                    )
                    # ---- Phase 1: find missed regions ----
                    print(
                        f"  Agent H phase 1 (find_missed) over "
                        f"{len(phase1_summary)} SAM3 bboxes ..."
                    )
                    if bool(getattr(args, "missed_via_point_prompt", False)):
                        # Point-prompt path: VLM proposes anchor points,
                        # SAM3 grounds masks/bboxes.
                        print("    (using --missed_via_point_prompt path)")
                        point_queries = agent_h_find_missed_regions_via_points(
                            backend=h_backend,
                            original_image_path=orig_image,
                            overlay_image_path=str(review_overlay_path),
                            components_summary=phase1_summary,
                            taxonomy=class_list,
                        )
                        print(f"    -> {len(point_queries)} anchor point(s) from VLM")
                        missed_regions = _ground_missed_points_with_sam3(
                            point_queries=point_queries,
                            original_image_path=orig_image,
                            args=args,
                            iter_dir=iter_dir,
                        )
                        print(f"    -> missed_regions={len(missed_regions)} (after SAM3 grounding)")
                    else:
                        missed_regions = agent_h_find_missed_regions(
                            screen_backend=JUDGMENT_SCREEN_BACKEND,
                            backend=h_backend,
                            original_image_path=orig_image,
                            overlay_image_path=str(review_overlay_path),
                            components_summary=phase1_summary,
                            taxonomy=class_list,
                        )
                        print(
                            f"    -> missed_regions={len(missed_regions)} (raw)"
                        )
                        # Cheap pixel-std pre-filter — only drops obvious
                        # blanks BEFORE paying the VLM cost. Default
                        # threshold is 0 (off) since the VLM review below
                        # is the canonical filter.
                        pre_std = float(getattr(args, "missed_min_pixel_std", 0.0))
                        if pre_std > 0:
                            missed_regions = _drop_blank_missed_regions(
                                missed_regions=missed_regions,
                                original_image_path=orig_image,
                                min_std=pre_std,
                            )
                            print(
                                f"    -> missed_regions={len(missed_regions)} "
                                f"(after pixel-std pre-filter)"
                            )
                        # VLM review (default on): for each remaining
                        # proposal, show the VLM the bbox crop + the slide
                        # context, ask whether it's a real component or a
                        # mistake. Drop anything it judges no.
                        if bool(getattr(args, "missed_vlm_review", True)) and missed_regions:
                            missed_regions = _vlm_review_missed_regions(
                                missed_regions=missed_regions,
                                original_image_path=orig_image,
                                iter_dir=iter_dir,
                                backend=h_backend,
                            )
                            print(
                                f"    -> missed_regions={len(missed_regions)} "
                                f"(after VLM review)"
                            )

                    # ---- Phase 2 (pipeline): predict missed filenames so
                    # they can appear in the Phase 3 overlay + summary
                    # alongside SAM3 components. Filename scheme matches
                    # _build_final_subdir so the final write step uses the
                    # same names. ----
                    for idx, mr in enumerate(missed_regions, start=1):
                        materialised_missed.append({
                            "filename": f"missed_{idx:04d}_{_safe_name(str(mr['class']))}.png",
                            "class": str(mr["class"]),
                            "bbox": [float(v) for v in mr["bbox"]],
                            "description": str(mr.get("description", "")).strip(),
                        })

                    # Phase 3 overlay: all fine components (SAM3 + missed),
                    # drawn uniformly red so H_merge sees no asymmetry.
                    merge_overlay_path = image_dir / "merge_overlay.png"
                    _build_consolidated_overlay(
                        orig_image,
                        [
                            {"bbox_xyxy": c["bbox_xyxy"], "granularity": "fine", "source": "sam3"}
                            for c in valid_for_review
                        ] + [
                            {"bbox_xyxy": mm["bbox"], "granularity": "fine", "source": "sam3"}
                            for mm in materialised_missed
                        ],
                        merge_overlay_path,
                        color_by_granularity=False,
                    )
                    phase3_summary = phase1_summary + [
                        {
                            "filename": mm["filename"],
                            "class": mm["class"],
                            "bbox": [round(float(v), 1) for v in mm["bbox"]],
                        }
                        for mm in materialised_missed
                    ]
                    # Layout-review missed regions have no SAM3 mask; their
                    # seg path is None and the mask probe falls back to the
                    # bbox rectangle.
                    for mm in materialised_missed:
                        seg_path_by_filename[mm["filename"]] = None

                    # ---- Phase 3: merge decisions over the unified set ----
                    print(
                        f"  Agent H phase 3 (merge) over "
                        f"{len(phase3_summary)} bboxes "
                        f"({len(phase1_summary)} SAM3 + "
                        f"{len(materialised_missed)} missed) ..."
                    )
                    merge_groups = agent_h_merge_groups(
                        backend=h_backend,
                        screen_backend=JUDGMENT_SCREEN_BACKEND,
                        original_image_path=orig_image,
                        overlay_image_path=str(merge_overlay_path),
                        components_summary=phase3_summary,
                        taxonomy=class_list,
                    )
                    print(
                        f"    -> merge_groups={len(merge_groups)}"
                    )

                    # H10 (opt-in): reject DEGENERATE merge groups — giant
                    # area or huge member count — keeping their members as
                    # fine components. Small legitimate merges pass through.
                    rejected_merge_groups: list[dict] = []
                    if merge_groups and (
                        args.merge_max_area_frac < 1.0 or args.merge_max_members > 0
                    ):
                        with Image.open(orig_image) as _im:
                            _W, _H = _im.size
                        _kept: list[dict] = []
                        for _g in merge_groups:
                            _bb = _g.get("merged_bbox") or [0.0, 0.0, 0.0, 0.0]
                            _frac = (
                                max(0.0, float(_bb[2]) - float(_bb[0]))
                                * max(0.0, float(_bb[3]) - float(_bb[1]))
                                / float(_W * _H)
                            )
                            _n = len(_g.get("member_filenames") or [])
                            # stack-v2 refinement: chart/plot/graph/table merges
                            # are exempt from the member cap — their fragments
                            # (axes, ticks, curves, cells) are not standalone
                            # components, unlike flowchart/diagram sub-elements.
                            # (Measured: capping an 11-member 'line chart' merge
                            # shattered the chart; Fable judge flagged it.)
                            _mc = str(_g.get("merged_class") or "").lower()
                            _chart_exempt = (
                                any(k in _mc for k in ("chart", "plot", "graph", "table"))
                                and not any(k in _mc for k in ("flow", "architecture", "diagram"))
                            )
                            if _chart_exempt:
                                _kept.append(_g)
                                continue
                            if (
                                args.merge_max_area_frac < 1.0
                                and _frac > args.merge_max_area_frac
                                and _n >= args.merge_area_min_members
                            ) or (
                                args.merge_max_members > 0
                                and _n > args.merge_max_members
                            ):
                                _g["rejected_reason"] = (
                                    f"area_frac={_frac:.3f} members={_n}"
                                )
                                rejected_merge_groups.append(_g)
                            else:
                                _kept.append(_g)
                        if rejected_merge_groups:
                            print(
                                f"    -> H10 merge-acceptance: rejected "
                                f"{len(rejected_merge_groups)} degenerate "
                                f"group(s), kept {len(_kept)}"
                            )
                        merge_groups = _kept

                    # Phase 4 (v2.6.1): post-merge polygon refinement.
                    # For every overlap between a merged group and a non-
                    # member fine component, ask Agent M2 whether the
                    # candidate is part of the same entity. M2 YES -> union.
                    # M2 NO -> either L-shape carve (masks don't share
                    # pixels) or schedule pixel-erase on the group's final
                    # crop (both non-text, masks do overlap), or leave
                    # alone (text on either side).
                    pixel_erases: list[dict] = []
                    if merge_groups and getattr(args, "polygon_refine", True):
                        print(
                            f"  Agent M2 (polygon refinement) over "
                            f"{len(merge_groups)} merge group(s), "
                            f"max_rounds={getattr(args, 'polygon_refine_rounds', 3)} ..."
                        )
                        try:
                            refine_result = _polygon_refine_merge_groups(
                                backend=h_backend,
                                original_image_path=orig_image,
                                merge_overlay_dir=image_dir / "polygon_refine_overlays",
                                merge_groups=merge_groups,
                                all_components=phase3_summary,
                                seg_path_by_filename=seg_path_by_filename,
                                max_rounds=int(getattr(args, "polygon_refine_rounds", 3)),
                            )
                            pixel_erases = refine_result.get("pixel_erases") or []
                            n_with_poly = sum(
                                1 for g in merge_groups if g.get("polygon")
                            )
                            n_extended = sum(
                                1
                                for g in merge_groups
                                if len(g.get("member_filenames") or []) > 0
                            )
                            # H10 post-refinement re-check (stack-v2 fix):
                            # M2 absorb decisions grow groups AFTER the pre-
                            # check, so caps must be re-enforced here — the
                            # calibration thresholds were measured on post-
                            # refinement member counts. Dropped groups keep
                            # their members as fine components; their
                            # scheduled pixel-erases are discarded.
                            if args.merge_max_area_frac < 1.0 or args.merge_max_members > 0:
                                with Image.open(orig_image) as _im2:
                                    _W2, _H2 = _im2.size
                                _kept2: list[dict] = []
                                for _gi, _g in enumerate(merge_groups):
                                    _bb = _g.get("merged_bbox") or [0.0, 0.0, 0.0, 0.0]
                                    _frac = (
                                        max(0.0, float(_bb[2]) - float(_bb[0]))
                                        * max(0.0, float(_bb[3]) - float(_bb[1]))
                                        / float(_W2 * _H2)
                                    )
                                    _n = len(_g.get("member_filenames") or [])
                                    _mc = str(_g.get("merged_class") or "").lower()
                                    _chart_exempt = (
                                        any(k in _mc for k in ("chart", "plot", "graph", "table"))
                                        and not any(k in _mc for k in ("flow", "architecture", "diagram"))
                                    )
                                    # Post-refine rejection fires on the AREA
                                    # cap only: member growth here comes from
                                    # individually opus-vetted M2 absorptions
                                    # (evidence of true belonging), and bulk
                                    # member-cap rejection at this stage
                                    # shattered a diagram's coverage
                                    # (LiMoSense R_c 1.0 -> 0.48). H's bulk
                                    # proposals are still member-capped at the
                                    # PRE-refine check.
                                    if not _chart_exempt and (
                                        args.merge_max_area_frac < 1.0
                                        and _frac > args.merge_max_area_frac
                                        and _n >= args.merge_area_min_members
                                    ):
                                        _g["rejected_reason"] = (
                                            f"post-refine area_frac={_frac:.3f} members={_n}"
                                        )
                                        rejected_merge_groups.append(_g)
                                        pixel_erases = [
                                            pe for pe in pixel_erases
                                            if pe.get("merge_index") != _gi
                                        ]
                                    else:
                                        _kept2.append(_g)
                                if len(_kept2) != len(merge_groups):
                                    print(
                                        f"    -> H10 post-refine: rejected "
                                        f"{len(merge_groups) - len(_kept2)} grown "
                                        f"group(s), kept {len(_kept2)}"
                                    )
                                merge_groups = _kept2
                            print(
                                f"    -> {n_with_poly} group(s) now have polygon, "
                                f"{n_extended} group(s) total, "
                                f"{len(pixel_erases)} pixel-erase(s) scheduled."
                            )
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"  [warn] polygon refinement failed: {exc}"
                            )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [warn] Agent H layout review failed: {exc}")
                    missed_regions = []
                    merge_groups = []
                    rejected_merge_groups = []
                layout_review_result = {
                    "missed_regions": missed_regions,
                    "merge_groups": merge_groups,
                    "rejected_merge_groups": rejected_merge_groups,
                    "pixel_erases": pixel_erases,
                }
            else:
                print("  Agent H: no valid components to review, skipping.")
        else:
            print("  Agent H: layout review disabled (--no-layout_review).")

        # Materialise the final/ subdir: every valid mask/bbox crop + a
        # color-coded overlay + a single granularity-tagged metadata.json.
        try:
            final_manifest = _build_final_subdir(
                image_dir=image_dir,
                original_image_path=orig_image,
                iters=iters,
                layout_review=layout_review_result,
                layout_review_raw=layout_review_result,
                backend=h_backend,
                f_dual_image=bool(getattr(args, "f_dual_image", False)),
            )
            print(
                f"  final/ built: {final_manifest['num_components']} components "
                f"({final_manifest['num_fine']} fine + "
                f"{final_manifest['num_merged']} merged + "
                f"{final_manifest['num_layout_review_missed']} layout-review missed)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] could not build final/ subdir: {exc}")
            final_manifest = None

        summary = {
            "original_image": orig_image,
            "wall_clock_sec": round(time.time() - slide_t0, 2),
            "num_iters_run": len(iters),
            "final_cleaned_image": iters[-1]["cleaned_image"] if iters else None,
            "final_verdict": iters[-1]["review"] if iters else None,
            "iters": iters,
            "cross_iter_dedup": cross_iter,
            "layout_review": layout_review_result,
            "final_manifest_summary": (
                {
                    k: final_manifest[k]
                    for k in (
                        "num_components",
                        "num_fine",
                        "num_merged",
                        "num_layout_review_missed",
                    )
                }
                if final_manifest
                else None
            ),
        }
        with (image_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        overall_summary.append(summary)
        _log_usage_marker({
            "marker": "slide_end",
            "image": str(orig_image),
            "idx": img_idx,
            "wall_clock_sec": summary["wall_clock_sec"],
        })

    with (run_dir / "summary.json").open("w") as f:
        json.dump(overall_summary, f, indent=2)
    print(f"\nDone. Overall summary -> {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
