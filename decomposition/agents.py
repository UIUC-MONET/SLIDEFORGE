"""Agents A-E for the slide component segmentation pipeline."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .backends import VLMBackend, log_usage
from .prompts import (
    AGENT_A_SYSTEM,
    AGENT_A_USER,
    AGENT_B_SYSTEM,
    AGENT_B_USER_TEMPLATE,
    AGENT_C_SYSTEM,
    AGENT_C_USER_TEMPLATE,
    AGENT_F_CLEANUP_SYSTEM,
    AGENT_F_CLEANUP_USER,
    AGENT_F_VALIDITY_SYSTEM,
    AGENT_F_VALIDITY_USER_TEMPLATE,
    AGENT_G_SYSTEM,
    AGENT_G_USER_TEMPLATE,
    AGENT_H_FIND_MISSED_SYSTEM,
    AGENT_H_FIND_MISSED_USER_TEMPLATE,
    AGENT_H_FIND_MISSED_POINTS_SYSTEM,
    AGENT_H_FIND_MISSED_POINTS_USER_TEMPLATE,
    AGENT_H_MISSED_VALIDATE_SYSTEM,
    AGENT_H_MISSED_VALIDATE_USER_TEMPLATE,
    AGENT_H_MERGE_SYSTEM,
    AGENT_H_MERGE_USER_TEMPLATE,
    AGENT_M2_SYSTEM,
    AGENT_M2_USER_TEMPLATE,
    AGENT_M3_SYSTEM,
    AGENT_M3_USER_TEMPLATE,
    AGENT_O_SEG_LAYERS_SYSTEM,
    AGENT_O_SEG_LAYERS_USER_TEMPLATE,
)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


def _parse_json(text: str) -> dict:
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Model sometimes emits a valid JSON object followed by trailing prose or a
    # second object ("Extra data") — take the FIRST complete object.
    start = t.find("{")
    if start == -1:
        raise json.JSONDecodeError("no JSON object found", t, 0)
    try:
        obj, _end = json.JSONDecoder().raw_decode(t[start:])
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Last resort: greedy braces match (previous behavior).
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        raise json.JSONDecodeError("no JSON object found", t, 0)
    return json.loads(m.group(0))


def agent_a_describe(backend: VLMBackend, image_path: str) -> list[str]:
    raw = backend.generate(AGENT_A_SYSTEM, AGENT_A_USER, images=[image_path])
    data = _parse_json(raw)
    phrases = [str(x).strip() for x in data.get("components", []) if str(x).strip()]
    # dedupe preserving order
    seen, out = set(), []
    for p in phrases:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def agent_b_map_to_classes(
    backend: VLMBackend, phrases: list[str], class_list: list[str]
) -> list[str]:
    if not phrases:
        return []
    user = AGENT_B_USER_TEMPLATE.format(
        class_list_json=json.dumps(class_list),
        phrases_json=json.dumps(phrases),
    )
    raw = backend.generate(AGENT_B_SYSTEM, user, images=None)
    data = _parse_json(raw)
    allowed = {c.lower(): c for c in class_list}
    picked: list[str] = []
    seen = set()
    for m in data.get("mappings", []):
        cls = m.get("class")
        if not cls:
            continue
        canonical = allowed.get(str(cls).strip().lower())
        if canonical and canonical not in seen:
            seen.add(canonical)
            picked.append(canonical)
    return picked


def agent_bc_combined(
    backend: VLMBackend, image_path: str, phrases: list[str], class_list: list[str]
) -> tuple[list[str], list[str]]:
    """H4 fold: B's phrase->class mapping + C's direct selection in ONE
    image-grounded call. Returns (b_classes, c_classes) with the same
    normalization as the separate agents."""
    from .prompts import AGENT_BC_SYSTEM, AGENT_BC_USER_TEMPLATE  # lazy: keep import surface stable
    user = AGENT_BC_USER_TEMPLATE.format(
        class_list_json=json.dumps(class_list),
        phrases_json=json.dumps(phrases),
    )
    raw = backend.generate(AGENT_BC_SYSTEM, user, images=[image_path])
    data = _parse_json(raw)
    allowed = {c.lower(): c for c in class_list}

    def _norm(values: Iterable[str]) -> list[str]:
        picked: list[str] = []
        seen = set()
        for cls in values:
            canonical = allowed.get(str(cls).strip().lower())
            if canonical and canonical not in seen:
                seen.add(canonical)
                picked.append(canonical)
        return picked

    b_classes = _norm(
        m.get("class") for m in data.get("mappings", []) if m.get("class")
    )
    c_classes = _norm(data.get("classes", []))
    return b_classes, c_classes


def agent_c_direct_select(
    backend: VLMBackend, image_path: str, class_list: list[str]
) -> list[str]:
    user = AGENT_C_USER_TEMPLATE.format(class_list_json=json.dumps(class_list))
    raw = backend.generate(AGENT_C_SYSTEM, user, images=[image_path])
    data = _parse_json(raw)
    allowed = {c.lower(): c for c in class_list}
    picked: list[str] = []
    seen = set()
    for cls in data.get("classes", []):
        canonical = allowed.get(str(cls).strip().lower())
        if canonical and canonical not in seen:
            seen.add(canonical)
            picked.append(canonical)
    return picked


def agent_g_part_of_check(
    backend: VLMBackend,
    image_path: str,
    bbox_a: list[float],
    class_a: str,
    bbox_b: list[float],
    class_b: str,
) -> dict:
    """Given the slide image and two overlapping bboxes (A smaller than B),
    decide whether A is a sub-part of B. Returns
    {"is_part_of": bool, "reason": str}.
    """
    user = AGENT_G_USER_TEMPLATE.format(
        class_a=class_a,
        class_b=class_b,
        bbox_a=[round(float(v), 2) for v in bbox_a],
        bbox_b=[round(float(v), 2) for v in bbox_b],
    )
    raw = backend.generate(AGENT_G_SYSTEM, user, images=[image_path])
    data = _parse_json(raw)
    return {
        "is_part_of": bool(data.get("is_part_of", False)),
        "reason": str(data.get("reason", "")).strip(),
    }


_COMPONENT_STEM_RE = re.compile(r"^component_\d+_(.+)$")


def _component_class_from_filename(name: str) -> str:
    stem = Path(name).stem
    m = _COMPONENT_STEM_RE.match(stem)
    if not m:
        return stem
    return m.group(1).replace("_", " ")


def render_bbox_overlay(
    original_image_path: str,
    bboxes_xyxy: list[list[float]],
    out_path: str,
    color: tuple[int, int, int] = (255, 0, 0),
    width_px: int = 4,
) -> str:
    """Draw the original slide with red rectangles over every extracted bbox,
    save it to ``out_path`` and return that path.

    Used as the second image to Agent F's cleanup call so F can distinguish
    "residue inside an already-extracted region" from "missed component".
    """
    img = Image.open(original_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in bboxes_xyxy:
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=color, width=int(width_px))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))
    return str(out)


def agent_f_cleanup_review(
    backend: VLMBackend,
    cleaned_image_path: str,
    bbox_overlay_path: str | None = None,
    screen_backend: VLMBackend | None = None,
) -> dict:
    """One VLM call: decide whether the cleaned slide needs another pass.

    When ``bbox_overlay_path`` is provided, F also sees the original slide
    with extracted-bbox rectangles drawn in red. The prompt instructs F to
    treat residue inside the red rectangles as mask-cleanup leftover (NOT a
    missed component) so it doesn't trigger spurious reruns.

    Returns ``{"needs_rerun": bool, "reason": str, "remaining_components": [..]}``.
    """
    images = [cleaned_image_path]
    if bbox_overlay_path:
        images.append(bbox_overlay_path)
    # H11 wave 2: sonnet screen; accept only high-confidence "clean" verdicts —
    # needs_rerun=true triggers a whole costly iteration, so that direction
    # (and any uncertainty) is confirmed by the expensive model.
    data, _tag = _screen_then_escalate(
        backend, screen_backend,
        AGENT_F_CLEANUP_SYSTEM, AGENT_F_CLEANUP_USER, images,
        accept=lambda d: _high_conf(d) and not bool(d.get("needs_rerun", False)),
    )
    return {
        "needs_rerun": bool(data.get("needs_rerun", False)),
        "reason": str(data.get("reason", "")).strip(),
        "remaining_components": [
            str(x).strip() for x in data.get("remaining_components", []) if str(x).strip()
        ],
        "cascade": data.get("cascade"),
    }


def _interior_ink_frac(path: str) -> float:
    """Fraction of non-background pixels in the crop's interior (border band
    excluded). Empty frame outlines — the known leniency-slip category for the
    cheap-model screen — score ~0 here. See experiments/h3-fvalidity-cascade."""
    try:
        im = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception:  # noqa: BLE001 — unreadable crop: treat as suspicious
        return 0.0
    ink = im < 230
    h, w = ink.shape
    b = max(2, int(0.06 * min(h, w)))
    if h <= 2 * b or w <= 2 * b:
        return 0.0
    return float(ink[b:h - b, b:w - b].mean())


_SCREEN_CONF_SUFFIX = (
    '\n\nAdditionally include a "confidence" field in your JSON: "high" if this '
    'verdict is unambiguous to you, "low" if you are at all unsure.'
)


def _screen_then_escalate(
    backend: VLMBackend,
    screen_backend: VLMBackend | None,
    system: str,
    user: str,
    images: list[str] | None,
    accept,
) -> tuple[dict, str | None]:
    """H11 wave-1 generic cascade: ask the cheap screen model first (with a
    mandatory confidence field); ``accept(data) -> bool`` decides whether the
    screen verdict stands. Anything else — including parse/transport errors —
    escalates to the expensive backend. Returns (data, cascade_tag)."""
    tag: str | None = None
    if screen_backend is not None:
        try:
            raw_s = screen_backend.generate(system, user + _SCREEN_CONF_SUFFIX, images=images)
            data_s = _parse_json(raw_s)
            if accept(data_s):
                data_s["cascade"] = "screen-accepted"
                return data_s, "screen-accepted"
            tag = "screen-escalated"
        except Exception:  # noqa: BLE001
            tag = "screen-error-escalated"
    raw = backend.generate(system, user, images=images)
    data = _parse_json(raw)
    if tag:
        data["cascade"] = tag
    return data, tag


def _high_conf(data: dict) -> bool:
    return str(data.get("confidence", "")).lower() == "high"


def agent_f_validate_component(
    backend: VLMBackend,
    component_path: str,
    slide_context_path: str | None = None,
    screen_backend: VLMBackend | None = None,
) -> dict:
    """One VLM call per component crop. Reviews the bbox-crop version and
    returns ``{"valid": bool, "is_text_only": bool, "ocr_latex": str,
    "bbox_is_tight": bool, "bbox_is_too_small": bool, "reason": str}``.

    ``bbox_is_tight`` and ``bbox_is_too_small`` are meaningful only for text-only 
    components — the pipeline uses them to decide whether to swap the SAM3 detection 
    bbox with the tight mask bbox stored in metadata, or to expand it outward.

    When ``slide_context_path`` is provided (v2.8: dual-image F validity),
    that file (the full slide with this component's bbox outlined in red)
    is sent as IMAGE 2. It lets the VLM disambiguate narrow / sparse
    crops (brackets, vertical bars, single mathematical symbols) whose
    role is only clear in slide context.
    """
    filename = Path(component_path).name
    class_name = _component_class_from_filename(filename)
    user = AGENT_F_VALIDITY_USER_TEMPLATE.format(
        class_name=class_name, filename=filename
    )
    images: list[str] = [component_path]
    if slide_context_path:
        images.append(slide_context_path)

    # H3 cascade (opt-in via screen_backend): geometric guard first — near-empty
    # interiors (the measured leniency-slip category) skip the cheap screen and
    # go straight to the expensive model. Otherwise the screen model answers,
    # and only valid+high-confidence verdicts are accepted without escalation.
    cascade = None
    if screen_backend is not None:
        if _interior_ink_frac(component_path) < 0.05:
            cascade = "guard-escalated"
        else:
            try:
                raw_s = screen_backend.generate(
                    AGENT_F_VALIDITY_SYSTEM, user + _SCREEN_CONF_SUFFIX, images=images
                )
                data_s = _parse_json(raw_s)
                conf = str(data_s.get("confidence", "")).lower()
                if bool(data_s.get("valid", False)) and conf == "high":
                    data_s["cascade"] = "screen-accepted"
                    data = data_s
                    cascade = "screen-accepted"
                    # H11: sampled audit of screen-accepts — the permanent
                    # leniency monitor (archived invalids can't measure the
                    # leniency direction once the cascade is live). Sampled
                    # crops get the expensive verdict (it's paid for) and an
                    # audit record; deterministic per-crop hash sampling.
                    try:
                        audit_frac = float(os.environ.get("CASCADE_AUDIT_FRAC", "0") or 0)
                    except ValueError:
                        audit_frac = 0.0
                    if audit_frac > 0:
                        import hashlib
                        h = int(hashlib.md5(component_path.encode()).hexdigest()[:8], 16)
                        if (h % 10_000) < audit_frac * 10_000:
                            raw_a = backend.generate(AGENT_F_VALIDITY_SYSTEM, user, images=images)
                            data_a = _parse_json(raw_a)
                            log_usage({
                                "audit": "f_validity",
                                "crop": Path(component_path).name,
                                "screen_valid": bool(data_s.get("valid", False)),
                                "expensive_valid": bool(data_a.get("valid", True)),
                                "agree": bool(data_s.get("valid", False)) == bool(data_a.get("valid", True)),
                            })
                            data_a["cascade"] = "screen-accepted-audited"
                            data = data_a
                            cascade = "screen-accepted"
            except Exception:  # noqa: BLE001 — screen failure: escalate silently
                cascade = "screen-error-escalated"

    if cascade != "screen-accepted":
        raw = backend.generate(
            AGENT_F_VALIDITY_SYSTEM, user, images=images
        )
        data = _parse_json(raw)
        if cascade:
            data["cascade"] = cascade
    valid = bool(data.get("valid", True))
    is_text_only = bool(data.get("is_text_only", False)) and valid
    ocr_latex = str(data.get("ocr_latex", "")).strip() if is_text_only else ""
    
    # Default to tight=True (no rewrite) unless the model says otherwise.
    # For non-text components, force bbox_is_tight=True regardless of model
    # output — we never rewrite non-text bboxes.
    bbox_is_tight_raw = data.get("bbox_is_tight", True)
    bbox_is_tight = bool(bbox_is_tight_raw) if is_text_only else True
    
    # Default to too_small=False unless the model says otherwise.
    # Likewise, only meaningful for text components.
    bbox_is_too_small_raw = data.get("bbox_is_too_small", False)
    bbox_is_too_small = bool(bbox_is_too_small_raw) if is_text_only else False

    return {
        "valid": valid,
        "is_text_only": is_text_only,
        "ocr_latex": ocr_latex,
        "bbox_is_tight": bbox_is_tight,
        "bbox_is_too_small": bbox_is_too_small,
        "reason": str(data.get("reason", "")).strip(),
        "cascade": data.get("cascade"),
    }


def agent_f_review(
    backend: VLMBackend,
    cleaned_image_path: str,
    component_paths: list[str] | None = None,
    bbox_overlay_path: str | None = None,
    slide_context_paths: list[str] | None = None,
    screen_backend: VLMBackend | None = None,
    cleanup_screen_backend: VLMBackend | None = None,
) -> dict:
    """Run one cleanup-verdict call plus one validity-per-component call.

    ``component_paths`` should point at the BBOX versions of the components
    (opaque rectangle crops), not the mask-cut segmentation versions, so the
    model sees the actual rendered region for OCR.

    ``bbox_overlay_path`` is an optional companion image — the original slide
    with extracted-bbox red rectangles drawn on top. When provided it is sent
    to the cleanup-verdict call so F can ignore residue inside already-
    extracted regions and only request a rerun for genuinely-missed content.

    ``slide_context_paths`` (v2.8) is an optional list parallel to
    ``component_paths``: for each component, a path to a slide-context
    overlay (full slide with that component's bbox outlined in red) that
    is passed as IMAGE 2 to the per-component validity call. Disambiguates
    narrow / sparse crops where IMAGE 1 alone is hard to interpret.

    Returns:
        {
          "needs_rerun": bool,
          "reason": str,
          "remaining_components": [..phrases..],
          "invalid_components": [..component basenames..],
          "component_verdicts": [
              {"filename": str, "valid": bool, "is_text_only": bool,
               "ocr_latex": str, "bbox_is_tight": bool, "bbox_is_too_small": bool, "reason": str}, ...
          ],
          "ocr_results": [
              {"filename": str, "latex": str}, ...   # text-only components only
          ],
        }
    """
    verdict = agent_f_cleanup_review(
        backend, cleaned_image_path, bbox_overlay_path=bbox_overlay_path,
        screen_backend=cleanup_screen_backend,
    )
    component_paths = list(component_paths or [])
    context_paths = list(slide_context_paths or [])
    # Allow callers to pass shorter/empty context list and still match.
    if context_paths and len(context_paths) != len(component_paths):
        # Best-effort alignment by index; missing entries → None.
        context_paths = context_paths + [None] * (len(component_paths) - len(context_paths))
    invalid: list[str] = []
    per_component: list[dict] = []
    ocr_results: list[dict] = []
    for idx, p in enumerate(component_paths):
        name = Path(p).name
        ctx = context_paths[idx] if idx < len(context_paths) else None
        try:
            decision = agent_f_validate_component(
                backend, p, slide_context_path=ctx, screen_backend=screen_backend
            )
        except Exception as exc:  # noqa: BLE001
            decision = {
                "valid": True,
                "is_text_only": False,
                "ocr_latex": "",
                "bbox_is_tight": True,
                "bbox_is_too_small": False,
                "reason": f"validity error: {exc}",
            }
        entry = {
            "filename": name,
            "valid": bool(decision.get("valid", True)),
            "is_text_only": bool(decision.get("is_text_only", False)),
            "ocr_latex": str(decision.get("ocr_latex", "")).strip(),
            "bbox_is_tight": bool(decision.get("bbox_is_tight", True)),
            "bbox_is_too_small": bool(decision.get("bbox_is_too_small", False)),
            "reason": str(decision.get("reason", "")).strip(),
            "cascade": decision.get("cascade"),
        }
        per_component.append(entry)
        if not entry["valid"]:
            invalid.append(name)
        elif entry["is_text_only"] and entry["ocr_latex"]:
            ocr_results.append({"filename": name, "latex": entry["ocr_latex"]})
    return {
        "needs_rerun": verdict["needs_rerun"],
        "reason": verdict["reason"],
        "remaining_components": verdict["remaining_components"],
        "invalid_components": invalid,
        "component_verdicts": per_component,
        "ocr_results": ocr_results,
    }


def agent_h_find_missed_regions(
    backend: VLMBackend,
    original_image_path: str,
    overlay_image_path: str,
    components_summary: list[dict],
    taxonomy: list[str],
    screen_backend: VLMBackend | None = None,
) -> list[dict]:
    """Phase 1 of layout review: identify substantive content that sits
    OUTSIDE every existing red rectangle and should become a new
    first-class component.

    ``components_summary`` is the existing bbox list (one entry per
    surviving SAM3 component across all iters); given as reference so the
    VLM doesn't double-report. The VLM is explicitly told NOT to make
    merge decisions in this call — those happen in a separate phase.

    Returns a list of ``{"bbox": [x1,y1,x2,y2], "class": str,
    "description": str}`` dicts.
    """
    user = AGENT_H_FIND_MISSED_USER_TEMPLATE.format(
        components_json=json.dumps(components_summary, indent=2),
        taxonomy_json=json.dumps(taxonomy),
    )
    # H11 wave 2: sonnet screen, accept high-confidence output; unsure escalates.
    data, _tag = _screen_then_escalate(
        backend, screen_backend,
        AGENT_H_FIND_MISSED_SYSTEM, user,
        [original_image_path, overlay_image_path],
        accept=_high_conf,
    )
    missed_regions: list[dict] = []
    for m in data.get("missed_regions") or []:
        bbox = m.get("bbox")
        cls = str(m.get("class", "")).strip()
        if not cls or not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        missed_regions.append({
            "bbox": [float(v) for v in bbox],
            "class": cls,
            "description": str(m.get("description", "")).strip(),
        })
    return missed_regions


def agent_h_validate_missed_region(
    backend: VLMBackend,
    crop_path: str,
    slide_context_path: str,
    class_name: str,
    description: str,
    bbox: list[float],
    screen_backend: VLMBackend | None = None,
) -> dict:
    """One VLM call to QA-check a single proposed missed region. Returns
    ``{"valid": bool, "reason": str}``.
    """
    user = AGENT_H_MISSED_VALIDATE_USER_TEMPLATE.format(
        class_name=class_name,
        description=description or "",
        bbox=[round(float(v), 1) for v in bbox],
    )
    # H11 wave 1: accept only high-confidence VALID screen verdicts; invalid
    # (drops a recovered region — destructive) or unsure escalates.
    data, _tag = _screen_then_escalate(
        backend, screen_backend,
        AGENT_H_MISSED_VALIDATE_SYSTEM, user,
        [crop_path, slide_context_path],
        accept=lambda d: _high_conf(d) and bool(d.get("valid", False)),
    )
    return {
        "valid": bool(data.get("valid", True)),
        "reason": str(data.get("reason", "")).strip(),
        "cascade": data.get("cascade"),
    }


def agent_h_find_missed_regions_via_points(
    backend: VLMBackend,
    original_image_path: str,
    overlay_image_path: str,
    components_summary: list[dict],
    taxonomy: list[str],
) -> list[dict]:
    """Alternative phase-1 path: VLM proposes anchor POINTS instead of
    bboxes. Downstream SAM3 (via infer_point_prompt.py) grounds each
    point into a real mask + bbox.

    Returns a list of ``{"point": [x, y], "class": str, "description": str}``.
    """
    user = AGENT_H_FIND_MISSED_POINTS_USER_TEMPLATE.format(
        components_json=json.dumps(components_summary, indent=2),
        taxonomy_json=json.dumps(taxonomy),
    )
    raw = backend.generate(
        AGENT_H_FIND_MISSED_POINTS_SYSTEM,
        user,
        images=[original_image_path, overlay_image_path],
    )
    data = _parse_json(raw)
    out: list[dict] = []
    for m in data.get("missed_regions") or []:
        pt = m.get("point")
        cls = str(m.get("class", "")).strip()
        if not cls or not (isinstance(pt, list) and len(pt) == 2):
            continue
        try:
            x, y = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        out.append({
            "point": [x, y],
            "class": cls,
            "description": str(m.get("description", "")).strip(),
        })
    return out


def agent_h_merge_groups(
    backend: VLMBackend,
    original_image_path: str,
    overlay_image_path: str,
    components_summary: list[dict],
    taxonomy: list[str],
    screen_backend: VLMBackend | None = None,
) -> list[dict]:
    """Phase 3 of layout review: decide which existing components should
    be merged into a single semantic super-component.

    ``components_summary`` should now contain BOTH the SAM3-extracted
    components AND the freshly-materialised missed-region components from
    Phase 1, treated uniformly. The VLM uses each ``filename`` to refer
    to bboxes in its output.

    Returns a list of ``{"member_filenames": [...], "merged_class": str,
    "merged_bbox": [x1,y1,x2,y2], "reason": str}`` dicts.
    """
    user = AGENT_H_MERGE_USER_TEMPLATE.format(
        components_json=json.dumps(components_summary, indent=2),
        taxonomy_json=json.dumps(taxonomy),
    )
    # H11 wave 2: sonnet screen, accept high-confidence output; unsure escalates.
    data, _tag = _screen_then_escalate(
        backend, screen_backend,
        AGENT_H_MERGE_SYSTEM, user,
        [original_image_path, overlay_image_path],
        accept=_high_conf,
    )
    merge_groups: list[dict] = []
    for g in data.get("merge_groups") or []:
        members = [
            str(m).strip()
            for m in (g.get("member_filenames") or [])
            if str(m).strip()
        ]
        cls = str(g.get("merged_class", "")).strip()
        bbox = g.get("merged_bbox")
        if (
            len(members) < 2
            or not cls
            or not (isinstance(bbox, list) and len(bbox) == 4)
        ):
            continue
        merge_groups.append({
            "member_filenames": members,
            "merged_class": cls,
            "merged_bbox": [float(v) for v in bbox],
            "reason": str(g.get("reason", "")).strip(),
        })
    return merge_groups


def agent_m2_can_merge(
    backend: VLMBackend,
    original_image_path: str,
    overlay_image_path: str,
    merged_class: str,
    merged_members_summary: list[dict],
    candidate_filename: str,
    candidate_class: str,
    candidate_bbox: list[float],
    screen_backend: VLMBackend | None = None,
) -> dict:
    """Per-candidate merge decision (post-merge polygon refinement step).

    Asks the VLM whether ``candidate`` should be absorbed into the existing
    merged group AND, when not, whether their actual pixels visually
    overlap or whether the bbox overlap is just "L-shape" (bbox-only)
    spatial coincidence.

    ``merged_members_summary`` is a list of
    ``{"filename", "class", "bbox"}`` dicts describing current members.

    Returns ``{"should_merge": bool, "visually_overlaps": bool,
    "reason": str}``. On parse failure, defaults to ``should_merge=False``
    + ``visually_overlaps=True`` (safer: do not merge if uncertain, and
    treat the overlap as real so the polygon-refine path either skips
    text components or applies a pixel-level erase rather than carving
    the group polygon).
    """
    user = AGENT_M2_USER_TEMPLATE.format(
        merged_class=merged_class,
        merged_members_json=json.dumps(merged_members_summary, indent=2),
        candidate_filename=candidate_filename,
        candidate_class=candidate_class,
        candidate_bbox=json.dumps(
            [round(float(v), 1) for v in candidate_bbox]
        ),
    )
    # H11 wave 1.1: the screen may only confirm the NULL action (leave alone:
    # no absorb, real visual overlap). Any geometry-altering verdict — absorb
    # or carve-enabling — escalates. (Wave 1's plain high-conf acceptance let
    # haiku perform surgery and was judged SIGNIFICANT-HARM on dense12.)
    try:
        data, _tag = _screen_then_escalate(
            backend, screen_backend,
            AGENT_M2_SYSTEM, user,
            [original_image_path, overlay_image_path],
            accept=lambda d: (
                _high_conf(d)
                and not bool(d.get("should_merge", False))
                and bool(d.get("visually_overlaps", True))
            ),
        )
    except (json.JSONDecodeError, ValueError):
        return {
            "should_merge": False,
            "visually_overlaps": True,
            "reason": "agent_m2: parse error",
        }
    return {
        "should_merge": bool(data.get("should_merge", False)),
        "visually_overlaps": bool(data.get("visually_overlaps", True)),
        "reason": str(data.get("reason", "")).strip(),
        "cascade": data.get("cascade"),
    }


def agent_m3_overlap_arbitrate(
    backend: VLMBackend,
    original_image_path: str,
    overlay_image_path: str,
    class_a: str,
    bbox_a: list[float],
    class_b: str,
    bbox_b: list[float],
    screen_backend: VLMBackend | None = None,
) -> dict:
    """Per-pair overlap-ownership decision (replaces the mask-coverage
    heuristic). Asks the VLM whether each side's actual content is in the
    overlap region.

    Returns ``{"a_owns_overlap": bool, "b_owns_overlap": bool, "reason": str}``.
    On parse error, returns both TRUE (no carve) for safety.
    """
    user = AGENT_M3_USER_TEMPLATE.format(
        class_a=class_a,
        bbox_a=json.dumps([round(float(v), 1) for v in bbox_a]),
        class_b=class_b,
        bbox_b=json.dumps([round(float(v), 1) for v in bbox_b]),
    )
    # H11 wave 1.1: the screen may only confirm the NULL action (both sides own
    # the overlap -> no carve). Any carve direction escalates.
    try:
        data, _tag = _screen_then_escalate(
            backend, screen_backend,
            AGENT_M3_SYSTEM, user,
            [original_image_path, overlay_image_path],
            accept=lambda d: (
                _high_conf(d)
                and bool(d.get("a_owns_overlap", True))
                and bool(d.get("b_owns_overlap", True))
            ),
        )
    except (json.JSONDecodeError, ValueError):
        return {
            "a_owns_overlap": True,
            "b_owns_overlap": True,
            "reason": "agent_m3: parse error (defaulting to no carve)",
        }
    return {
        "a_owns_overlap": bool(data.get("a_owns_overlap", True)),
        "b_owns_overlap": bool(data.get("b_owns_overlap", True)),
        "reason": str(data.get("reason", "")).strip(),
        "cascade": data.get("cascade"),
    }


def agent_d_union(b_classes: list[str], c_classes: list[str], class_list: list[str]) -> list[str]:
    """Preserve the taxonomy order in the final union."""
    chosen = {c.lower() for c in b_classes} | {c.lower() for c in c_classes}
    return [c for c in class_list if c.lower() in chosen]


def _poll_part_of_requests(
    proc: subprocess.Popen,
    exchange_dir: Path,
    backend: VLMBackend,
    poll_sec: float = 0.5,
) -> int:
    """While the Agent E subprocess is alive, handle any part_of_request_*.json
    files in exchange_dir by calling Agent G and writing part_of_response_*.json.
    Returns the subprocess exit code.
    """
    handled: set[str] = set()
    while True:
        rc = proc.poll()
        for req in sorted(exchange_dir.glob("part_of_request_*.json")):
            key = req.name
            if key in handled:
                continue
            resp_path = exchange_dir / req.name.replace("request_", "response_")
            if resp_path.exists():
                handled.add(key)
                continue
            try:
                with req.open() as f:
                    payload = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue  # request file still being written
            try:
                decision = agent_g_part_of_check(
                    backend=backend,
                    image_path=payload["image_path"],
                    bbox_a=payload["A"]["bbox_xyxy"],
                    class_a=payload["A"]["class"],
                    bbox_b=payload["B"]["bbox_xyxy"],
                    class_b=payload["B"]["class"],
                )
            except Exception as exc:
                decision = {"is_part_of": False, "reason": f"agent_g error: {exc}"}
            with resp_path.open("w") as f:
                json.dump(decision, f, indent=2)
            handled.add(key)
            print(f"[agent_g] {req.name} -> is_part_of={decision['is_part_of']} ({decision.get('reason','')[:80]})")
        if rc is not None:
            # Process exited; one last sweep handled above. Return.
            return rc
        time.sleep(poll_sec)


def agent_e_segment(
    manifest_path: Path,
    output_dir: Path,
    ckpt: str,
    base_ckpt: str,
    conda_python: str,
    script_path: Path,
    score_thresh: float = 0.3,
    max_boxes_per_text: int = 5,
    device: str | None = None,
    extra_args: list[str] | None = None,
    extra_env: dict | None = None,
    backend: VLMBackend | None = None,
    part_of_exchange_dir: Path | None = None,
    bbox_output_dir: Path | None = None,
) -> int:
    cmd = [
        conda_python, str(script_path),
        "--ckpt", ckpt,
        "--base_ckpt", base_ckpt,
        "--manifest", str(manifest_path),
        "--output_dir", str(output_dir),
        "--score_thresh", str(score_thresh),
        "--max_boxes_per_text", str(max_boxes_per_text),
    ]
    if extra_args:
        cmd += list(extra_args)
    if device:
        cmd += ["--device", device]
    if part_of_exchange_dir is not None:
        part_of_exchange_dir = Path(part_of_exchange_dir)
        part_of_exchange_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["--part_of_exchange_dir", str(part_of_exchange_dir)]
    if bbox_output_dir is not None:
        bbox_output_dir = Path(bbox_output_dir)
        bbox_output_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["--bbox_output_dir", str(bbox_output_dir)]
    env = None
    if extra_env:
        import os
        env = {**os.environ, **extra_env}
    print(f"[agent_e] Running: {' '.join(cmd)}")

    # H8 (opt-in via env SAM3_WORKER_DIR): enqueue the invocation to a
    # persistent worker instead of spawning a fresh subprocess (which reloads
    # 6.7 GB of checkpoints every time). The worker runs the SAME script code
    # via run(args, model=<preloaded>). Not supported for the part_of
    # interactive path (worker jobs are fire-and-wait).
    import os as _os
    worker_dir = _os.environ.get("SAM3_WORKER_DIR")
    if worker_dir and part_of_exchange_dir is None:
        import uuid as _uuid
        wq = Path(worker_dir)
        jid = _uuid.uuid4().hex[:12]
        job = {"id": jid, "argv": [str(c) for c in cmd[2:]]}
        tmp = wq / f".job_{jid}.tmp"
        tmp.write_text(json.dumps(job))
        tmp.rename(wq / f"job_{jid}.json")
        done = wq / f"done_{jid}.json"
        t0 = time.time()
        while not done.exists():
            if time.time() - t0 > 3600:
                raise TimeoutError(f"SAM3 worker: no result for job {jid} in 1h")
            time.sleep(0.2)
        result = json.loads(done.read_text())
        try:
            done.unlink()
        except OSError:
            pass
        rc = int(result.get("returncode", 1))
        if rc != 0:
            print(f"[agent_e] worker job {jid} failed rc={rc}: "
                  f"{str(result.get('error'))[:400]}")
        return rc

    if part_of_exchange_dir is not None and backend is not None:
        proc = subprocess.Popen(cmd, env=env)
        return _poll_part_of_requests(proc, part_of_exchange_dir, backend)
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def agent_o_segmentation_layers(
    backend: VLMBackend,
    image_path: str,
    class_list: list[str],
) -> list[list[str]]:
    """Order/partition ``class_list`` into 2-4 segmentation layers for the
    given slide. Layer 0 is segmented first; layer N runs after every
    earlier layer's masks are removed from the image.

    Returns a list-of-lists with the same set of classes (exactly once
    each, in tier order). On any parse / validation failure, falls back
    to a single-layer ordering equal to the input list.
    """
    if not class_list:
        return []
    if len(class_list) == 1:
        return [list(class_list)]
    user = AGENT_O_SEG_LAYERS_USER_TEMPLATE.format(
        class_list_json=json.dumps(list(class_list), ensure_ascii=False, indent=2)
    )
    try:
        raw = backend.generate(
            AGENT_O_SEG_LAYERS_SYSTEM, user, images=[image_path]
        )
        data = _parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"  [agent_o] parse failure ({exc}); falling back to single layer.")
        return [list(class_list)]

    raw_layers = data.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        print("  [agent_o] no 'layers' key in response; falling back to single layer.")
        return [list(class_list)]

    by_layer: list[tuple[int, list[str]]] = []
    for entry in raw_layers:
        if not isinstance(entry, dict):
            continue
        try:
            layer_idx = int(entry.get("layer", len(by_layer) + 1))
        except (TypeError, ValueError):
            layer_idx = len(by_layer) + 1
        classes = entry.get("classes") or []
        if not isinstance(classes, list):
            continue
        clean = [str(c).strip() for c in classes if str(c).strip()]
        if clean:
            by_layer.append((layer_idx, clean))
    by_layer.sort(key=lambda x: x[0])

    input_set = {str(c).strip() for c in class_list if str(c).strip()}
    seen: set[str] = set()
    canonical: list[list[str]] = []
    for _, classes in by_layer:
        clean_layer: list[str] = []
        for c in classes:
            if c in input_set and c not in seen:
                clean_layer.append(c)
                seen.add(c)
        if clean_layer:
            canonical.append(clean_layer)

    missing = [c for c in class_list if c.strip() and c.strip() not in seen]
    if missing:
        if canonical:
            canonical[-1].extend(missing)
        else:
            canonical.append(missing)
        print(
            f"  [agent_o] {len(missing)} class(es) missing from response; "
            f"appended to last layer: {missing}"
        )

    return canonical if canonical else [list(class_list)]

