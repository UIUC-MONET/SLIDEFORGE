"""Step 3 (class 5): regenerate every class-5 element (logo / photograph /
text-and-image / data chart) with the OpenAI gpt-image-2 image-edit API so
it harmonises with the new styled background.

Per class-5 component we run a two-call pipeline:

  1) Claude (Opus) inspects the original slide, the styled background, and
     the foreground element, then writes a tailored gpt-image-2 prompt that
     either (A) recolours the element's own background to match the new
     theme, (B) keeps the raster subject pixel-identical and adds a
     decorative rim, or (C) restyles only a chart's background/axis layer
     while locking data marks and text. Output: ``<stem>_gpt_prompt.txt``.

  2) gpt-image-2 image-edit is called with the cropped component PNG +
     the styled background as reference, and Claude's prompt as
     instruction. Output: ``<stem>_gpt.png`` (saved next to the original
     cropped PNG so step 5 can pick it up).

A copy of the original cropped component PNG is stashed alongside the
generated one because step 5 reads its pixel dimensions to resize the
regenerated image back to the original bbox footprint.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from pathlib import Path

from common import (
    CLAUDE_OPUS,
    OPENAI_IMAGE_MODEL,
    anthropic_client,
    call_claude,
    case_paths,
    encode_image_b64,
    list_front_bg_pages,
    load_api_keys,
    openai_client,
    parse_json_loose,
    styled_bg_for_role,
    page_role,
)


TARGET_CLASS = 5
MODEL = CLAUDE_OPUS


PROMPT_GEN = """You are preparing a single prompt that will be sent to gpt-image-2 to perform STYLE TRANSFER on a foreground element extracted from a PowerPoint slide.

Images:
  1. ORIGINAL FULL SLIDE - source slide for context.
  2. FOREGROUND ELEMENT - a class-5 element (pure logo / photograph /
     text-and-image composite / data chart) that gpt-image-2 will edit.
     Bbox on the slide: <<BBOX>>; slide is <<SLIDE_W>>x<<SLIDE_H>>px.
  3. NEW STYLED BACKGROUND - the new slide background; the foreground must
     visually harmonise with it.

New-style palette / mood:
<<PALETTE_SUMMARY>>

The element's semantic content (logo wordmark, photo subject, chart data,
axis labels, every coloured bar / line / dot, etc.) MUST be preserved
unchanged - gpt-image-2 must NOT regenerate, rewrite, recolour the
subject itself, or hallucinate any new content. The element will be
re-pasted onto the new background at the SAME bbox, so its outer
footprint must stay the same.

STEP 1: Classify the element into ONE of THREE categories:

  Category LOGO_OR_SOLID  (use Approach A)
    Logos, icons, wordmarks, badges, monograms, simple flat
    illustrations whose original background is a FLAT / SOLID colour
    fill (white, grey, light pastel, single-colour rectangle, etc.).
    The "background" area is clearly separate from the subject and can
    be repainted without touching the subject.

  Category NATURAL_IMAGE  (use Approach B)
    Photographs, microscopy / medical / satellite imagery, naturalistic
    illustrations, screenshots, and any image where the subject and
    background blend continuously (gradients, soft shadows, complex
    textures). Recolouring the background here would inevitably
    contaminate or destroy the subject, so it MUST be left untouched.

  Category DATA_CHART  (use Approach C)
    Histograms, bar charts, scatter / line / box plots, pie charts,
    heat maps, any visualisation that encodes data via colour or
    position. This includes matplotlib/seaborn/ggplot-style raster
    figures with large white figure or axes backgrounds.

STEP 2: Map the category to one of three approaches:

  Approach A - BACKGROUND COLOUR REPLACEMENT
    Recolour the element's own flat background fill so it harmonises with
    the new theme. Preserve the logo/icon/wordmark geometry, layout,
    wording, and identity exactly.

    IMPORTANT CONTRAST RULE:
    If replacing a light/white background with a dark theme colour would
    make black or dark logo text/strokes unreadable, do NOT leave dark
    text directly on the dark background. In that case, choose ONE safe
    option:
        1) keep a light neutral panel/card behind the logo while styling the
        surrounding background, OR
        2) recolour only the dark text/strokes to a high-contrast theme
        colour such as warm ivory, cream, or metallic gold, while
        preserving all shapes, glyphs, and layout.
    Never generate a prompt that says to place black/dark text unchanged
    on a near-black background.

    LAYOUT RULE:
    Keep the subject (logo wordmark / glyphs / icon) within the central 50%
    of the output canvas. The outer 25% on each side must be background-
    matching texture/fill only, with no critical glyphs or readable text, so
    downstream center-crop can remove only background slack.

  Approach B - MATCHING DECORATIVE BORDER
    Keep the element pixel-identical and ADD a thin decorative border /
    frame around its outer edge in the new background's visual
    language (materials, motifs, line weight, accent colour). The
    border must stay INSIDE the element's bbox - no bleed outside.

    HARD RULE - the border lives ONLY on the OUTER RIM. It MUST NOT
    extend, overlap, or bleed INTO the element's subject content
    (logo wordmark / glyphs, photo subject, chart plotting area, axes,
    bars, lines, data points, axis tick labels, legend swatches, etc.).
    Leave a clear inset margin between the inner edge of the border
    and the subject content so the subject remains fully visible and
    uncovered. If the original element has no margin around its
    subject, choose a thinner border rather than encroaching on the
    subject. Use this for NATURAL_IMAGE.

  Approach C - DATA CHART BACKGROUND RESTYLE WITH DATA LOCK
    The large white figure / axes background of a matplotlib-style chart
    is NOT fully protected; it is a transformable background layer. It may
    be changed to a theme-compatible panel colour. When changing the chart
    background, synchronously adjust axis lines, tick labels, axis labels,
    title, legend text, and grid lines so they remain readable.

    HARD DATA LOCK:
    Do NOT change the geometry, count, positions, sizes, labels, legend
    mapping, or colour identity of data marks (bars, lines, points, boxes,
    heatmap cells, pie slices). Do NOT rewrite any chart text. Do NOT add,
    remove, smooth, or hallucinate data. If separating the background/axis
    layer from data marks is unsafe, keep the chart interior pixel-identical
    and instead wrap it in a dark/light themed panel, card, or frame.

STEP 3: Write a single concrete, self-contained instruction prompt for
gpt-image-2 that performs the chosen approach. Be specific: name the
approach, cite concrete colours from the new palette, and call out the
hard preservation constraints listed above. If you chose Approach B,
the gpt_prompt MUST explicitly tell gpt-image-2 that the decorative
border sits only on the outer rim, must keep a clear inset margin from
the subject, and must NEVER extend into or overlap the subject content
(logo glyphs, photo subject, etc.). If you chose Approach C, the
gpt_prompt MUST explicitly distinguish transformable chart background /
axis / grid layers from locked data marks and legend mappings.

Output STRICT JSON only - no markdown, no commentary outside the JSON:
{
  "category": "LOGO_OR_SOLID" | "NATURAL_IMAGE" | "DATA_CHART",
  "approach": "A" or "B" or "C",
  "approach_name": "background_colour_replacement" or "matching_border" or "data_chart_background_restyle",
  "reason": "<short sentence linking category -> approach -> why>",
  "gpt_prompt": "<the full prompt, 80-200 words>"
}
"""


CHART_EDIT_CRITIC_PROMPT = """You are checking whether a style-transferred data chart preserved its data.

Image A is the ORIGINAL chart component. Image B is the edited chart.

The edit is allowed to change the figure/axes background, grid, panel/frame,
and axis/tick/label/title colours for readability. It is NOT allowed to
change data mark geometry, count, positions, sizes, colour-to-series mapping,
legend mapping, or any readable chart text.

Reply STRICT JSON only:
{
  "ok": <true|false>,
  "reason": "<one short sentence>",
  "violations": ["..."]
}
"""


def _palette_summary(palette: dict | None) -> str:
    if not isinstance(palette, dict):
        return "(no palette.json found; rely on the reference background image)"
    res = palette.get("result") or {}
    theme = res.get("theme_color") or {}
    theme_str = f"{theme.get('hex','')} ({theme.get('name','')})".strip()
    entries = []
    for p in res.get("theme_palette") or []:
        if isinstance(p, dict):
            entries.append(
                f"{p.get('hex','')} {p.get('name','')} [{p.get('role','')}]".strip()
            )
    notes = res.get("notes") or ""
    return (
        f"theme color: {theme_str}\n"
        f"palette: {'; '.join(entries) if entries else '(none)'}\n"
        f"notes: {notes}"
    )


def _load_palette(paths) -> dict | None:
    p = paths.styled_bg_dir / "palette.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def write_gpt_prompt(client, original_path, component_path, styled_bg_path,
                     palette, bbox, slide_w, slide_h):
    summary = _palette_summary(palette)
    txt = (
        PROMPT_GEN
        .replace("<<BBOX>>",
                 f"(x0={bbox[0]:.1f}, y0={bbox[1]:.1f}, x1={bbox[2]:.1f}, y1={bbox[3]:.1f})")
        .replace("<<SLIDE_W>>", str(slide_w))
        .replace("<<SLIDE_H>>", str(slide_h))
        .replace("<<PALETTE_SUMMARY>>", summary)
    )
    content = [
        {"type": "text", "text": "Original full slide:"},
        encode_image_b64(original_path),
        {"type": "text", "text": "Foreground element to style-transfer (class 5):"},
        encode_image_b64(component_path),
    ]
    if styled_bg_path is not None and styled_bg_path.exists():
        content.extend([
            {"type": "text", "text": "New styled background:"},
            encode_image_b64(styled_bg_path),
        ])
    else:
        content.append({"type": "text",
                        "text": "(no styled background; rely on palette text)"})
    content.append({"type": "text", "text": txt})
    raw = call_claude(client, content, model=MODEL, max_tokens=2048)
    parsed = parse_json_loose(raw)
    return raw, parsed


def _chart_edit_ok(client, original_component: Path, edited_component: Path) -> tuple[bool, dict | None]:
    content = [
        {"type": "text", "text": "Image A - original chart component:"},
        encode_image_b64(original_component),
        {"type": "text", "text": "Image B - edited chart component:"},
        encode_image_b64(edited_component),
        {"type": "text", "text": CHART_EDIT_CRITIC_PROMPT},
    ]
    raw = call_claude(client, content, model=MODEL, max_tokens=768)
    parsed = parse_json_loose(raw)
    if isinstance(parsed, dict):
        return bool(parsed.get("ok") is True), parsed
    return False, {"ok": False, "reason": "critic parse failed", "raw_response": raw}


# ---------------------------------------------------------------------------
# gpt-image-2 image-edit call
# ---------------------------------------------------------------------------

def call_gpt_image_edit(oa_client, prompt_text: str,
                        component_png: Path, styled_bg_path: Path | None,
                        out_path: Path) -> Path:
    """Single-image-output image edit, conditioned on the cropped component
    PNG (foreground to restyle) and the styled background (style reference)."""
    ins = [component_png]
    if styled_bg_path is not None and styled_bg_path.exists():
        ins.append(styled_bg_path)
    handles = [open(p, "rb") for p in ins]
    try:
        result = oa_client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=handles,
            prompt=prompt_text,
            n=1,
        )
    finally:
        for h in handles:
            h.close()
    if not result.data:
        raise RuntimeError("gpt-image-2 returned no image")
    out_path.write_bytes(base64.b64decode(result.data[0].b64_json))
    return out_path


def process_component(claude, oa_client, page_dir, slide_w, slide_h,
                      original_path, comp_meta, comp_entry, out_root,
                      styled_bg_path, palette):
    page_name = page_dir.name
    cf = comp_entry["component_file"]
    bbox = comp_meta["bbox_xyxy"]
    stem = Path(cf).stem
    out_dir = out_root / page_name
    out_dir.mkdir(parents=True, exist_ok=True)
    component_path = page_dir / cf

    print(f"--- [{page_name}] {stem} ---")

    # 1) Cache a copy of the original cropped PNG so step5 has it locally.
    cached_png = out_dir / f"{stem}.png"
    if not cached_png.exists():
        shutil.copyfile(component_path, cached_png)

    # 2) bbox sidecar for traceability
    (out_dir / f"{stem}_bbox.json").write_text(json.dumps({
        "page": page_name, "component_file": cf,
        "filename_hint": comp_entry.get("filename_hint"),
        "class": comp_entry.get("class"),
        "bbox_xyxy": bbox,
        "slide_size": {"width": slide_w, "height": slide_h},
        "original_slide_path": str(original_path),
    }, indent=2, ensure_ascii=False))

    # 3) Claude-generated gpt prompt
    prompt_txt = out_dir / f"{stem}_gpt_prompt.txt"
    prompt_meta = out_dir / f"{stem}_gpt_prompt.meta.json"
    if not (prompt_txt.exists() and prompt_meta.exists()):
        raw, parsed = write_gpt_prompt(
            claude, original_path, component_path,
            styled_bg_path, palette, bbox, slide_w, slide_h,
        )
        approach = approach_name = reason = gpt_prompt = category = None
        if isinstance(parsed, dict):
            approach = parsed.get("approach")
            approach_name = parsed.get("approach_name")
            reason = parsed.get("reason")
            gpt_prompt = parsed.get("gpt_prompt")
            category = parsed.get("category")
        if not isinstance(gpt_prompt, str) or not gpt_prompt.strip():
            print(f"  [warn] Claude prompt parse failed; using generic fallback")
            gpt_prompt = (
                "Perform style transfer on this image. Preserve the central "
                "subject (logo wordmark, photo subject, chart data) "
                "pixel-identical, keep the outer image footprint identical, "
                "and recolour or border the element so it harmonises with "
                f"the new slide style described as: {_palette_summary(palette)}"
            )
            approach = approach or "fallback"
            approach_name = approach_name or "fallback"
        prompt_txt.write_text(gpt_prompt.strip() + "\n")
        prompt_meta.write_text(json.dumps({
            "category": category,
            "approach": approach, "approach_name": approach_name,
            "reason": reason, "gpt_image_model": OPENAI_IMAGE_MODEL,
            "raw_response": raw,
        }, indent=2, ensure_ascii=False))
        print(f"  category={category} approach={approach}; saved {prompt_txt.name}")

    # 4) gpt-image-2 image edit
    out_png = out_dir / f"{stem}_gpt.png"
    if out_png.exists():
        print(f"  [skip] {out_png.name} exists")
        return out_png
    prompt = prompt_txt.read_text(encoding="utf-8").strip()
    try:
        call_gpt_image_edit(oa_client, prompt, cached_png, styled_bg_path, out_png)
        try:
            meta = json.loads(prompt_meta.read_text())
        except Exception:
            meta = {}
        if str(meta.get("category") or "").upper() == "DATA_CHART":
            ok, critic = _chart_edit_ok(claude, cached_png, out_png)
            meta["post_edit_critic"] = critic
            if not ok:
                shutil.copyfile(cached_png, out_png)
                meta["fallback"] = "original_chart_with_step5_panel"
                print(f"  [chart critic] fallback to original chart: {critic}")
            else:
                print("  [chart critic] OK")
            prompt_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"  saved: {out_png}")
    except Exception as e:
        print(f"  [error] gpt-image-2 edit failed: {e}")
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="override per-case output root (default: output/<case>/)")
    args = ap.parse_args()
    paths = case_paths(args.case, out_dir=args.out_dir)
    load_api_keys()
    paths.step3_class5_dir.mkdir(parents=True, exist_ok=True)

    palette = _load_palette(paths)
    if palette is None:
        print(f"  [warn] no palette.json under {paths.styled_bg_dir}; "
              f"prompt will rely on the reference background image only")

    claude = anthropic_client()
    oa_client = openai_client()

    pages = list_front_bg_pages(paths)
    if not pages:
        print(f"no front_bg pages under {paths.front_bg_dir}", file=sys.stderr)
        return 1

    n_pages = len(pages)
    for i, page in enumerate(pages):
        cat_path = page / "categorization.json"
        meta_path = page / "metadata.json"
        if not cat_path.exists() or not meta_path.exists():
            continue
        cat = json.loads(cat_path.read_text())
        meta = json.loads(meta_path.read_text())
        slide_w = int(meta["query_image_size"]["width"])
        slide_h = int(meta["query_image_size"]["height"])
        original_path = Path(meta["query_image_path"])
        bbox_lookup = {c["component_file"]: c for c in meta["components"]}
        targets = [r for r in cat["results"] if r.get("class") == TARGET_CLASS]
        if not targets:
            continue
        role = page_role(i, n_pages)
        styled_bg = styled_bg_for_role(paths, role)
        if not styled_bg.exists():
            # Fall back to the content bg if a role's file is missing
            styled_bg = paths.styled_bg_dir / "content.png"
        print(f"\n=== Page {page.name} (role={role}, bg={styled_bg.name}): "
              f"{len(targets)} class-5 ===")
        for r in targets:
            cm = bbox_lookup.get(r["component_file"])
            if cm is None:
                continue
            try:
                process_component(
                    claude, oa_client, page, slide_w, slide_h, original_path,
                    cm, r, paths.step3_class5_dir, styled_bg, palette,
                )
            except Exception as e:
                print(f"  [error] {r['component_file']}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
