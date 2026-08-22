"""Step 2: ask Claude (Opus) to classify every foreground component.

For each page under ``front_bg/<page>/`` we:
  1. Read the page's ``metadata.json`` to find the original slide image
     and the list of components.
  2. Send Claude one prompt per component (original slide + cropped
     component image + filename hint) asking for one of six classes.
  3. Save the per-page result to ``front_bg/<page>/categorization.json``.

Classes:
  0 - decorative; drop
  1 - text/formula, no background fill
  2 - text/formula with colored or shaped background
  3 - structured text that must stay editable (typically a table)
  4 - shape composition approximable by primitives (arrow + box, etc.)
  5 - raster-only content (photo / complex logo / screenshot / data chart)

Idempotent: if ``categorization.json`` already exists for a page it is
skipped. Delete the file to re-classify.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from common import (
    CLAUDE_OPUS,
    anthropic_client,
    call_claude,
    case_paths,
    encode_image_b64,
    list_front_bg_pages,
    parse_json_loose,
)


PROMPT = """You are analyzing ONE foreground element extracted from a PowerPoint slide.

The FIRST image you were shown is the ORIGINAL FULL SLIDE (before any segmentation). Use it as context to understand the role and semantic contribution of the element within the slide.
The SECOND image is the specific FOREGROUND ELEMENT to classify (transparent background, isolated from the slide).

Additional hint: the filename of this foreground element already contains a coarse category label produced by an upstream detector. The hint for THIS element is: "<<FILENAME_HINT>>". This hint is informative but NOT guaranteed to be accurate - treat it only as a reference and rely primarily on what you actually see in the images.

Assign the element to EXACTLY ONE of the following six classes:

- Class 0: SAFE-TO-DROP pure decoration. Use this ONLY for thin divider lines, plain edge/background bars, empty masks, subtle background patterns, or ornamental fragments that contain no logo, no icon, no readable text, no data mark, no arrow/flow meaning, and no callout / annotation meaning.
- Class 1: Pure text or formula WITHOUT any background color or shape; can be processed directly.
- Class 2: Pure text or formula that comes WITH a colored or shaped background.
- Class 3: Text in a special format that MUST remain editable (e.g., a table).
- Class 4: An object that can be approximated by a composition of multiple simple shapes (e.g., arrow + rectangle, flowchart, architecture diagram, icon made of simple shapes, callout group, shape-encoded chart, or other text+shape diagram). IMPORTANT: mixed text-and-shape content should be Class 4 when the visual structure can be represented by editable primitive PPTX shapes.
- Class 5: Raster-only content that carries strong semantics and CANNOT be approximated safely by editable primitive shapes. Use this for natural photographs, screenshots, complex logos/wordmarks/badges, dense bitmap illustrations, and matplotlib/raster data charts whose data marks, axes, legend, or text should stay image-like.

NEVER assign class 0 when the element is any of: logo, icon, photograph, chart/plot/legend, table, callout, annotation, highlighted marker, arrow that indicates flow, page number, section title, label, emoji used as emphasis, screenshot, or a component created by coverage audit.
If unsure between class 0 and any nonzero class, choose the nonzero class. False positives are safer than dropping slide content.

Respond with STRICT JSON only, no markdown, no commentary outside the JSON:
{
  "class": <integer 0-5>,
  "semantic_role": "safe_drop_decoration|text|label|formula|logo|icon|photo|screenshot|chart|legend|table|technical_diagram|shape_encoded_chart|data_diagram|flow_diagram|callout|annotation|page_number|other",
  "shape_approximable": <true|false>,
  "raster_required": <true|false>,
  "protected_content": <true|false>,
  "route_confidence": "high|medium|low",
  "drop_safe": <true|false>,
  "reason": "<one short sentence>"
}
"""


ROUTE_CRITIC_PROMPT = """You are the routing self-critic for a PowerPoint foreground component.

The FIRST image is the original full slide. The SECOND image is the isolated
component. A first-pass classifier proposed this route:

<<INITIAL_JSON>>

Review ONLY the route. Keep the route if it is correct, or revise it if the
component would be better handled by another pipeline class.

Routing rules:
- Prefer Class 4 for text+shape diagrams, architecture diagrams, flowcharts,
  callout groups, shape-encoded charts, and simple icons when editable PPTX
  primitive shapes can approximate the visual structure.
- Use Class 5 only when raster treatment is genuinely required: natural photo,
  screenshot, complex logo/wordmark/badge, dense bitmap illustration, or
  matplotlib/raster data chart.
- A mixed text+image component is NOT automatically Class 5. Ask whether Claude
  could reasonably reconstruct the visible structure with boxes, arrows, lines,
  simple shapes, and text. If yes, choose Class 4.
- Never allow Class 0 for readable text, labels, page numbers, data marks,
  callouts, arrows with flow meaning, charts, legends, tables, logos, icons, or
  anything semantically meaningful.

Output STRICT JSON only:
{
  "class": <integer 0-5>,
  "semantic_role": "safe_drop_decoration|text|label|formula|logo|icon|photo|screenshot|chart|legend|table|technical_diagram|shape_encoded_chart|data_diagram|flow_diagram|callout|annotation|page_number|other",
  "shape_approximable": <true|false>,
  "raster_required": <true|false>,
  "protected_content": <true|false>,
  "route_confidence": "high|medium|low",
  "drop_safe": <true|false>,
  "reason": "<one short sentence>"
}
"""


PROTECTED_HINT_WORDS = {
    "logo", "icon", "photo", "photograph", "chart", "plot", "legend",
    "table", "callout", "annotation", "label", "title", "page",
    "number", "footer", "emoji", "arrow", "diagram", "screenshot", "equation",
    "formula", "coverage", "missed",
}

PROTECTED_ROLES = {
    "text", "label", "formula", "logo", "icon", "photo", "screenshot",
    "chart", "legend", "table", "technical_diagram",
    "shape_encoded_chart", "data_diagram", "flow_diagram",
    "callout", "annotation", "page_number", "other",
}

TEXT_ROLES = {"text", "label", "formula", "page_number"}
TABLE_ROLES = {"table"}
SHAPE_ROLES = {
    "icon", "legend", "technical_diagram", "shape_encoded_chart",
    "data_diagram", "flow_diagram", "callout", "annotation", "chart",
}
RASTER_ROLES = {"logo", "photo", "screenshot"}


def filename_hint(component_path: Path) -> str:
    """``iter00_component_0001_logo_icon.png`` -> ``logo icon``."""
    stem = component_path.stem
    parts = stem.split("_")
    while parts and (
        parts[0].startswith("iter") or parts[0] == "component" or parts[0].isdigit()
    ):
        parts.pop(0)
    return " ".join(parts) if parts else stem


def classify_component(client, original_path: Path, component_path: Path):
    hint = filename_hint(component_path)
    content = [
        {"type": "text", "text": "Original slide (full image, before segmentation):"},
        encode_image_b64(original_path),
        {"type": "text", "text": f"Foreground element to classify (filename hint: \"{hint}\"):"},
        encode_image_b64(component_path),
        {"type": "text", "text": PROMPT.replace("<<FILENAME_HINT>>", hint)},
    ]
    raw = call_claude(client, content, model=CLAUDE_OPUS, max_tokens=768)
    return raw, parse_json_loose(raw)


def _valid_class(v) -> int | None:
    try:
        cls = int(v)
    except (TypeError, ValueError):
        return None
    return cls if 0 <= cls <= 5 else None


def _infer_role_from_hint(hint: str) -> str:
    hint_l = (hint or "").lower()
    if any(w in hint_l for w in ("photo", "photograph")):
        return "photo"
    if "screenshot" in hint_l:
        return "screenshot"
    if any(w in hint_l for w in ("logo", "wordmark")):
        return "logo"
    if any(w in hint_l for w in ("chart", "plot")):
        return "chart"
    if "table" in hint_l:
        return "table"
    if any(w in hint_l for w in ("diagram", "flowchart", "architecture")):
        return "technical_diagram"
    if any(w in hint_l for w in ("callout", "annotation", "speech")):
        return "callout"
    if any(w in hint_l for w in ("title", "label", "footer", "text", "number")):
        return "text"
    if "icon" in hint_l:
        return "icon"
    return "other"


def _fallback_class_from_role_hint(role: str, hint: str) -> int:
    role = (role or "").lower()
    hint_l = (hint or "").lower()
    if role in TEXT_ROLES or "footer" in hint_l:
        return 1
    if role in TABLE_ROLES:
        return 3
    if role in RASTER_ROLES:
        return 5
    if role in SHAPE_ROLES:
        if role == "chart" and any(w in hint_l for w in ("line", "bar", "plot", "chart")):
            return 5
        return 4
    if any(w in hint_l for w in ("photo", "screenshot", "logo", "wordmark")):
        return 5
    if any(w in hint_l for w in ("diagram", "flow", "arrow", "callout", "annotation", "icon", "legend")):
        return 4
    return 5


def coerce_parsed(raw: str | None, parsed, hint: str) -> dict:
    if isinstance(parsed, dict) and _valid_class(parsed.get("class")) is not None:
        return parsed
    raw_s = raw or ""
    out = dict(parsed) if isinstance(parsed, dict) else {}
    m = re.search(r'"?class"?\s*:\s*"?([0-5])"?', raw_s)
    if m:
        out["class"] = int(m.group(1))
    role = out.get("semantic_role")
    if not role:
        m = re.search(r'"?semantic_role"?\s*:\s*"?([A-Za-z0-9_ -]+)"?', raw_s)
        role = m.group(1).strip().lower().replace(" ", "_") if m else _infer_role_from_hint(hint)
        out["semantic_role"] = role
    if "drop_safe" not in out:
        m = re.search(r'"?drop_safe"?\s*:\s*(true|false)', raw_s, flags=re.I)
        if m:
            out["drop_safe"] = m.group(1).lower() == "true"
    if "shape_approximable" not in out:
        m = re.search(r'"?shape_approximable"?\s*:\s*(true|false)', raw_s, flags=re.I)
        if m:
            out["shape_approximable"] = m.group(1).lower() == "true"
    if "raster_required" not in out:
        m = re.search(r'"?raster_required"?\s*:\s*(true|false)', raw_s, flags=re.I)
        if m:
            out["raster_required"] = m.group(1).lower() == "true"
    if "protected_content" not in out:
        m = re.search(r'"?protected_content"?\s*:\s*(true|false)', raw_s, flags=re.I)
        if m:
            out["protected_content"] = m.group(1).lower() == "true"
    if not out.get("route_confidence"):
        m = re.search(r'"?route_confidence"?\s*:\s*"?([A-Za-z]+)"?', raw_s)
        if m:
            out["route_confidence"] = m.group(1).lower()
    if not out.get("reason"):
        m = re.search(r'"?reason"?\s*:\s*"([^"]+)"', raw_s)
        if m:
            out["reason"] = m.group(1)
    if _valid_class(out.get("class")) is None:
        out["class"] = _fallback_class_from_role_hint(str(out.get("semantic_role") or ""), hint)
        out["reason"] = (out.get("reason") or "classifier JSON parse failed; used safe hint fallback")
    return out


def _needs_route_critic(parsed: dict | None, hint: str) -> bool:
    if not isinstance(parsed, dict):
        return False
    cls = _valid_class(parsed.get("class"))
    role = str(parsed.get("semantic_role") or "").lower()
    hint_l = (hint or "").lower()
    confidence = str(parsed.get("route_confidence") or "").lower()
    if cls in (0, 5):
        return True
    if cls == 4 and role in {"chart", "shape_encoded_chart", "data_diagram"}:
        return True
    if confidence == "low":
        return True
    return any(w in hint_l for w in ("diagram", "flowchart", "callout", "chart", "plot", "logo", "photo", "screenshot"))


def route_self_critic(client, original_path: Path, component_path: Path,
                      initial: dict | None):
    if not isinstance(initial, dict):
        return None, None
    prompt = ROUTE_CRITIC_PROMPT.replace(
        "<<INITIAL_JSON>>",
        json.dumps(initial, ensure_ascii=False, indent=2),
    )
    content = [
        {"type": "text", "text": "Original slide (full image, before segmentation):"},
        encode_image_b64(original_path),
        {"type": "text", "text": "Foreground element to route-review:"},
        encode_image_b64(component_path),
        {"type": "text", "text": prompt},
    ]
    raw = call_claude(client, content, model=CLAUDE_OPUS, max_tokens=768)
    return raw, parse_json_loose(raw)


def _fallback_class_for_protected(hint: str, role: str,
                                  parsed: dict | None) -> int:
    role = (role or "").lower()
    hint_l = (hint or "").lower()
    shape_approximable = bool((parsed or {}).get("shape_approximable"))
    raster_required = bool((parsed or {}).get("raster_required"))
    if role in TEXT_ROLES:
        return 1
    if role in TABLE_ROLES:
        return 3
    if role in RASTER_ROLES or raster_required:
        return 5
    if role in SHAPE_ROLES or shape_approximable:
        return 4
    if any(w in hint_l for w in ("photo", "photograph", "screenshot", "logo", "wordmark")):
        return 5
    if any(w in hint_l for w in ("diagram", "flow", "arrow", "callout", "annotation", "icon", "legend")):
        return 4
    return 5


def guarded_class(raw_cls, hint: str, parsed: dict | None,
                  comp_meta: dict | None):
    cls = _valid_class(raw_cls)
    if cls is None:
        return raw_cls, []

    flags = []
    hint_l = (hint or "").lower()
    role = str((parsed or {}).get("semantic_role") or "").lower()
    drop_safe = (parsed or {}).get("drop_safe")
    protected = (
        any(w in hint_l for w in PROTECTED_HINT_WORDS)
        or role in PROTECTED_ROLES
        or bool((comp_meta or {}).get("coverage_audit_suggested_class"))
    )
    if cls == 0 and (protected or drop_safe is False):
        cls = _fallback_class_for_protected(hint, role, parsed)
        flags.append(f"class0_guard_forced_to_class{cls}")
    return cls, flags


def process_page(client, page_dir: Path) -> None:
    out_path = page_dir / "categorization.json"
    if out_path.exists():
        print(f"  [skip] {page_dir.name}: categorization.json already exists")
        return

    meta_path = page_dir / "metadata.json"
    if not meta_path.exists():
        print(f"  [skip] {page_dir.name}: no metadata.json")
        return
    meta = json.loads(meta_path.read_text())
    original_path = Path(meta["query_image_path"])
    if not original_path.exists():
        print(f"  [skip] {page_dir.name}: original missing: {original_path}")
        return

    components = sorted((page_dir / "components_segmented").glob("*.png"))
    print(f"\n=== Page {page_dir.name}: {len(components)} component(s) ===")
    bbox_lookup = {c.get("component_file"): c for c in meta.get("components", [])}

    results = []
    for comp in components:
        print(f"  -> {comp.name}")
        raw, parsed = classify_component(client, original_path, comp)
        hint = filename_hint(comp)
        parsed = coerce_parsed(raw, parsed, hint)
        comp_key = f"components_segmented/{comp.name}"
        comp_meta = bbox_lookup.get(comp_key)
        critic_raw = None
        critic_parsed = None
        if _needs_route_critic(parsed if isinstance(parsed, dict) else None, hint):
            try:
                critic_raw, critic_parsed = route_self_critic(
                    client, original_path, comp,
                    parsed if isinstance(parsed, dict) else None,
                )
                critic_parsed = coerce_parsed(critic_raw, critic_parsed, hint)
                if isinstance(critic_parsed, dict) and _valid_class(critic_parsed.get("class")) is not None:
                    parsed = critic_parsed
                    raw = raw + "\n\n[route_self_critic]\n" + (critic_raw or "")
            except Exception as e:
                critic_raw = f"[error] {type(e).__name__}: {e}"
        cls = parsed.get("class") if isinstance(parsed, dict) else None
        reason = parsed.get("reason") if isinstance(parsed, dict) else None
        cls, flags = guarded_class(cls, hint, parsed if isinstance(parsed, dict) else None,
                                   comp_meta)
        if flags:
            reason = f"{'; '.join(flags)}; {reason or ''}".strip()
        results.append({
            "component_file": comp_key,
            "filename_hint": hint,
            "class": cls,
            "semantic_role": parsed.get("semantic_role") if isinstance(parsed, dict) else None,
            "shape_approximable": parsed.get("shape_approximable") if isinstance(parsed, dict) else None,
            "raster_required": parsed.get("raster_required") if isinstance(parsed, dict) else None,
            "protected_content": parsed.get("protected_content") if isinstance(parsed, dict) else None,
            "route_confidence": parsed.get("route_confidence") if isinstance(parsed, dict) else None,
            "drop_safe": parsed.get("drop_safe") if isinstance(parsed, dict) else None,
            "class0_risk_flags": flags,
            "route_self_critic": critic_parsed if isinstance(critic_parsed, dict) else None,
            "route_self_critic_raw": critic_raw,
            "reason": reason,
            "raw_response": raw,
        })
        print(f"     class={cls} | {reason}")

    out_path.write_text(json.dumps({
        "page": page_dir.name,
        "original_image": str(original_path),
        "model": CLAUDE_OPUS,
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"  saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="override per-case output root (default: output/<case>/)")
    args = ap.parse_args()

    paths = case_paths(args.case, out_dir=args.out_dir)
    pages = list_front_bg_pages(paths)
    if not pages:
        print(f"no front_bg pages under {paths.front_bg_dir}", file=sys.stderr)
        return 1
    client = anthropic_client()
    for p in pages:
        process_page(client, p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
