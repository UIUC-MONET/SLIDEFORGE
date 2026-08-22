"""Step 1: generate styled backgrounds and derive a palette for one case.

Two sub-steps run back-to-back:

  1) gpt-image-2 image-edit call:
        Feed the (cleaned background, original render) pair for the
        opening page, one or more content pages, and the closing page,
        together with the user's ``style_prompt.txt``, and ask gpt-image-2
        to emit N restyled backgrounds (one per role: opening, content,
        closing). Output PNGs go to
            full_pipeline/output/<case>/styled_bg/{opening,content,closing}.png

  2) Claude palette extraction:
        Send all three styled backgrounds to Claude (Opus) in a single
        request and ask for a unified ``palette.json`` (theme_palette +
        font_palette) so downstream steps 3-5 can colour-restyle every
        element against a coherent target palette.

The page-role assignment is automatic: opening = page 0000, closing =
last page, content = a middle page (the one closest to the centre of
the deck). For very short decks (1-2 pages) the missing roles fall back
to the available ones.
"""

from __future__ import annotations

import argparse
import base64
import json
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
    load_style_prompt,
    openai_client,
    parse_json_loose,
)


# ---------------------------------------------------------------------------
# Page-role selection
# ---------------------------------------------------------------------------

CONTENT_PICKER_PROMPT = """You are choosing which MIDDLE page of a slide deck to use as the "content" exemplar that will be sent to a style-transfer image model alongside the opening and closing pages.

For each candidate above you saw:
  - the ORIGINAL slide image, and
  - the CLEANED background (foreground elements removed by an upstream segmenter).

Pick the candidate whose CLEANED background best satisfies BOTH:
  (a) cleanest reference - minimal residual foreground artifacts (faint logos, leftover text, bbox-removal halos), and
  (b) most representative of a typical content slide for THIS deck (not too sparse, not visually dominated by one big chart that the style model would confuse with the background).

Reply with STRICT JSON only, no markdown, no commentary:
{"pick_index": <1-N>, "reason": "<one short sentence>"}
"""


def _page_cleaned_bg(page_dir: Path) -> Path | None:
    """Latest-iter cleaned background PNG for a page."""
    candidates = sorted(page_dir.glob("cleaned_iter*.png"))
    return candidates[-1] if candidates else None


def _page_original(page_dir: Path) -> Path:
    return page_dir / "original.png"


def pick_content_page(claude_client, candidates: list[Path]) -> Path:
    """Ask Claude which middle page to use as the content reference.

    Falls back to the geometrically-middle candidate on any parsing
    failure. With 0 or 1 candidates, returns trivially.
    """
    if not candidates:
        raise ValueError("pick_content_page: no candidates")
    if len(candidates) == 1:
        print(f"  content page: only one candidate ({candidates[0].name}); picking it")
        return candidates[0]

    content: list = []
    for i, page in enumerate(candidates, 1):
        orig = _page_original(page)
        cleaned = _page_cleaned_bg(page)
        content.append({"type": "text",
                        "text": f"Candidate {i} (page {page.name}) - original slide:"})
        if orig.exists():
            content.append(encode_image_b64(orig))
        if cleaned is not None and cleaned.exists():
            content.append({"type": "text",
                            "text": f"Candidate {i} (page {page.name}) - cleaned background:"})
            content.append(encode_image_b64(cleaned))
    content.append({"type": "text", "text": CONTENT_PICKER_PROMPT})

    print(f"  asking Claude to pick best middle page from "
          f"{len(candidates)} candidate(s)...")
    raw = call_claude(claude_client, content, model=CLAUDE_OPUS, max_tokens=512)
    parsed = parse_json_loose(raw)
    pick = parsed.get("pick_index") if isinstance(parsed, dict) else None
    reason = parsed.get("reason") if isinstance(parsed, dict) else None
    if isinstance(pick, int) and 1 <= pick <= len(candidates):
        chosen = candidates[pick - 1]
        print(f"  -> Claude picked candidate {pick} ({chosen.name}): {reason}")
        return chosen
    fallback = candidates[len(candidates) // 2]
    print(f"  [warn] could not parse pick_index from Claude reply: {raw[:200]!r}; "
          f"falling back to {fallback.name}")
    return fallback


def select_role_pages(pages: list[Path], claude_client) -> dict[str, Path]:
    """Return {role: page_dir} for opening, content, closing.

      opening = pages[0]   (mandatory first page)
      closing = pages[-1]  (mandatory last page)
      content = Claude picks from pages[1:-1] (middle candidates)

    For very short decks (N <= 2) the missing roles fall back to the
    available ones.
    """
    if not pages:
        return {}
    n = len(pages)
    opening = pages[0]
    closing = pages[-1] if n > 1 else pages[0]
    middle_candidates = pages[1:-1] if n >= 3 else []
    if middle_candidates:
        content = pick_content_page(claude_client, middle_candidates)
    elif n == 2:
        content = pages[0]
    else:
        content = pages[0]
    return {"opening": opening, "content": content, "closing": closing}


# ---------------------------------------------------------------------------
# Step 1a: gpt-image-2 image-edit
# ---------------------------------------------------------------------------
#
# Sequential 3-call structure:
#
#   Call 1: opening = image.edit([opening_bg, opening_orig], n=1)
#   Call 2: content = image.edit([content_bg, content_orig, opening.png], n=1)
#   Call 3: closing = image.edit([closing_bg, closing_orig, opening.png], n=1)
#
# Why this layout:
#   * n=1 + a prompt that only describes ONE slide eliminates the multi-panel
#     composite behavior we observed with a single 6-in/3-out batched call.
#   * Call 2 and 3 receive the already-generated opening as a third input
#     and are explicitly told it is THE deck's style reference - enforcing
#     cross-page visual consistency without batching.
#   * Each call still receives the (cleaned_bg, original) pair so the
#     model knows where foreground bboxes will sit and can keep its
#     decorations out of those regions.

# 3:2 is the closest supported size to a 16:9 / 4:3 slide; gpt-image-2 will
# crop/letterbox internally and we resize on save.
_IMG_SIZE = "1536x1024"

# ---- Per-role prompt templates (IO scaffolding around <<STYLE>>) ----

_MATCH_FROM_OPENING_BULLETS = """  MATCH from Image 3 (the visual VOCABULARY) — your output must inherit:
    - the same color palette as Image 3 (read the actual tones off it)
    - the same materials, surface textures, and lighting / mood as Image 3
    - the same line weight and detail level as Image 3
    - the SAME FAMILY of decorative motifs and accent shapes used in Image 3.
      Do NOT invent motif families or subject matter that are absent from
      Image 3 and absent from the active Style block.
  DO NOT COPY from Image 3:
    - the specific positions of any motif or accent
    - the specific composition or framing
    - exact pixel content. The output MUST DIFFER from Image 3 in layout.
"""

_OPENING_PROMPT = """You are given 2 input images for the OPENING / TITLE page of a multi-page PowerPoint deck:
  Image 1: the CLEANED background of this page (upstream segmenter removed the foreground).
  Image 2: the ORIGINAL page (foreground intact). Use it ONLY to see where the foreground elements (title, authors, logos) will sit, so your generated decorations do NOT intrude into those bbox regions.

Style:
<<STYLE>>

Output requirements:
- Output EXACTLY ONE full-canvas styled background that fills the ENTIRE output image, edge to edge.
- Do NOT split the output into thumbnails, panels, grids, stripes, side-by-side variations, or "main image plus thumbnails" composites. The output IS the slide canvas - one slide, one output image.
- Decorations belong along the edges, corners, and margins. Leave the large central region visually quiet so the title / authors can be placed on top later.
- This is the opening/title page, so decoration can be on the more elaborate (hero) end.
- Image 1 may still contain residual logos or text that the upstream segmenter failed to remove - IGNORE them and do NOT preserve them in your output. (Residual color bars or solid color strips MAY be kept as a hint of the source deck's accent palette, but residual logos / text MUST be removed.)
"""


_CONTENT_PROMPT = """You are given 3 input images for a CONTENT / MIDDLE page of a multi-page PowerPoint deck:
  Image 1: the CLEANED background of the CURRENT page (foreground removed by an upstream segmenter; may still contain residual logos/text - ignore them).
  Image 2: the ORIGINAL CURRENT page (foreground intact). Use it ONLY to see where the foreground elements will sit, so your generated decorations do NOT intrude into those bbox regions.
  Image 3: the ALREADY-GENERATED OPENING page of THIS SAME deck. Image 3 is a STYLE-VOCABULARY reference, NOT a layout / pixel reference.

How to use Image 3:
<<MATCH_FROM_OPENING>>

This is a CONTENT / middle page, so its decoration must be NOTICEABLY LIGHTER than Image 3 (the opening page):
  - Use FEWER ornamental elements than Image 3; keep accents smaller, sparser, and lower-contrast.
  - Smaller / thinner decoration along edges; the central 70% of the canvas should be visually very quiet (mostly the base material with only subtle texture).
  - Move whichever decorations you keep to DIFFERENT positions than Image 3.

Decoration placement on this content page (HARD constraint):
  - Decorations must be confined to the TOP edge and/or the BOTTOM edge of the canvas only (within roughly the top 12% and the bottom 12% of the canvas height).
  - The LEFT and RIGHT edges of the canvas MUST be left CLEAN — no ornaments, strips, ribbons, or motifs along the left or right side. Content slides routinely place bullets / charts / diagrams that extend all the way to the left and right margins, and any side decoration would clash with them. A flat, plain extension of the base material along the left and right is correct.
  - The entire middle / central horizontal band of the canvas (between the top decoration strip and the bottom decoration strip) must be a quiet, near-empty extension of the base material — no motifs, no ornaments, no accent shapes intruding from the sides.
  - Image 2 shows where foreground elements (text, charts, diagrams) will be placed; treat every foreground bbox in Image 2 as STRICTLY OFF-LIMITS — your decorations must not overlap, intrude into, or even visually crowd those regions.

Style (context only - Image 3 is the canonical visual-vocabulary reference):
<<STYLE>>

Output requirements:
- Output EXACTLY ONE full-canvas styled background that fills the ENTIRE output image, edge to edge.
- Do NOT split the output into thumbnails, panels, grids, stripes, side-by-side variations, or "main image plus thumbnails" composites. The output IS the slide canvas — one slide, one output image.
- Do NOT replicate Image 3's composition. The composition / layout MUST differ.
- Leave a large open central region so content text / charts / tables can later be placed on top.
- Image 1 may still contain residual logos or text — IGNORE them and do NOT preserve them in your output. (Residual color bars or solid color strips MAY be kept as accent hint, but ONLY along the top or bottom edge — never on the left or right.)
""".replace("<<MATCH_FROM_OPENING>>", _MATCH_FROM_OPENING_BULLETS)


_CLOSING_PROMPT = """You are given 3 input images for the CLOSING / ENDING page of a multi-page PowerPoint deck:
  Image 1: the CLEANED background of this page (foreground removed; residual logos/text should be ignored).
  Image 2: the ORIGINAL page (foreground intact). Use it ONLY to see where the foreground elements will sit, so your generated decorations do NOT intrude into those bbox regions.
  Image 3: the ALREADY-GENERATED OPENING page of THIS SAME deck. Image 3 is a STYLE-VOCABULARY reference, NOT a layout / pixel reference.

How to use Image 3:
<<MATCH_FROM_OPENING>>

This is the CLOSING / ending page. Its decoration level can be similar to the opening (Image 3) — a hero-feel closer — but the SPECIFIC COMPOSITION must clearly differ from Image 3: rearrange motif placement, vary accent placement, and alter the framing. The viewer should be able to tell the two pages apart at a glance even though they read as the same deck.

Style (context only - Image 3 is the canonical visual-vocabulary reference):
<<STYLE>>

Output requirements:
- Output EXACTLY ONE full-canvas styled background that fills the ENTIRE output image, edge to edge.
- Do NOT split the output into thumbnails, panels, grids, stripes, side-by-side variations, or "main image plus thumbnails" composites. The output IS the slide canvas — one slide, one output image.
- Do NOT replicate Image 3's composition. The composition / layout MUST differ from Image 3.
- Leave the central region visually quiet so the closing message can later be placed on top.
- Image 1 may still contain residual logos or text — IGNORE them; do NOT preserve them in your output. (Residual color bars or strips MAY be kept as accent hint.)
""".replace("<<MATCH_FROM_OPENING>>", _MATCH_FROM_OPENING_BULLETS)


def _role_inputs(role: str, role_pages: dict[str, Path]) -> tuple[Path, Path]:
    page = role_pages[role]
    bg = _page_cleaned_bg(page)
    orig = _page_original(page)
    if bg is None:
        print(f"  [warn] no cleaned background for {role} ({page.name}); "
              f"falling back to original {orig.name}")
        bg = orig
    if not bg.exists():
        raise FileNotFoundError(bg)
    if not orig.exists():
        raise FileNotFoundError(orig)
    return bg, orig


def _call_image_edit(client, input_paths: list[Path], prompt: str,
                     out_path: Path) -> None:
    handles = [open(p, "rb") for p in input_paths]
    try:
        result = client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=handles,
            prompt=prompt,
            size=_IMG_SIZE,
            n=1,
        )
    finally:
        for h in handles:
            h.close()
    if not result.data:
        raise RuntimeError(f"gpt-image-2 returned no image for {out_path.name}")
    out_path.write_bytes(base64.b64decode(result.data[0].b64_json))


def generate_styled_backgrounds(paths, style_prompt: str,
                                role_pages: dict[str, Path]) -> dict[str, Path]:
    """Generate the 3 styled backgrounds via sequential dependent calls.

    Opening goes first, then opening's output is fed into the content and
    closing calls as a style-reference image. This gives single-canvas
    outputs (n=1 per call) while preserving cross-page style consistency."""
    paths.styled_bg_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {r: paths.styled_bg_dir / f"{r}.png"
                 for r in ("opening", "content", "closing")}

    style_block = style_prompt.strip()
    client = openai_client()

    # ---- Call 1: opening ----
    opening_out = out_paths["opening"]
    if opening_out.exists():
        print(f"  [skip] opening: {opening_out.name} already exists")
    else:
        bg, orig = _role_inputs("opening", role_pages)
        print(f"  [opening] inputs=[{bg.name}, {orig.name}], n=1...")
        prompt = _OPENING_PROMPT.replace("<<STYLE>>", style_block)
        _call_image_edit(client, [bg, orig], prompt, opening_out)
        print(f"    saved: {opening_out}")

    # ---- Call 2: content (with opening as style ref) ----
    content_out = out_paths["content"]
    if content_out.exists():
        print(f"  [skip] content: {content_out.name} already exists")
    else:
        bg, orig = _role_inputs("content", role_pages)
        print(f"  [content] inputs=[{bg.name}, {orig.name}, opening.png], n=1...")
        prompt = _CONTENT_PROMPT.replace("<<STYLE>>", style_block)
        _call_image_edit(client, [bg, orig, opening_out], prompt, content_out)
        print(f"    saved: {content_out}")

    # ---- Call 3: closing (with opening as style ref) ----
    closing_out = out_paths["closing"]
    if closing_out.exists():
        print(f"  [skip] closing: {closing_out.name} already exists")
    else:
        bg, orig = _role_inputs("closing", role_pages)
        print(f"  [closing] inputs=[{bg.name}, {orig.name}, opening.png], n=1...")
        prompt = _CLOSING_PROMPT.replace("<<STYLE>>", style_block)
        _call_image_edit(client, [bg, orig, opening_out], prompt, closing_out)
        print(f"    saved: {closing_out}")

    return out_paths


# ---------------------------------------------------------------------------
# Step 1b: Claude palette extraction
# ---------------------------------------------------------------------------

_PALETTE_PROMPT = """You are a senior visual designer analyzing the visual style of a slide deck.

You have just been shown THREE slide background images from the SAME deck, in this order:
  1. opening.png  -- the opening / title slide background
  2. content.png  -- a representative content slide background
  3. closing.png  -- the closing / ending slide background

Treat the three images as ONE coherent visual system. Derive a unified design palette that describes the deck as a whole (not each image individually).

Produce the following:

1. THEME COLOR
   - The single most representative dominant color of the deck (the color a viewer would name first when describing these slides).
   - Provide it as an uppercase hex string like "#1A2B3C", plus a short human-readable name (e.g. "deep navy").

2. THEME COLOR PALETTE  ("theme_palette")
   - An ordered list of 4-6 colors that together define the visual identity of the deck (background, primary accent, secondary accent, supporting tones, etc.).
   - Order them from MOST prominent / structurally important to LEAST.
   - For each entry give: hex, name, and a one-phrase role (e.g. "background", "primary accent", "supporting tone").

3. FONT COLOR PALETTE  ("font_palette")
   - An ordered list of recommended font (text) colors to use ON TOP OF the theme/background of this deck.
   - Each font color should be drawn from or harmonize with the theme palette.
   - Order them STRICTLY from HIGHEST contrast (most readable on the dominant background) to LOWEST contrast (most subtle / decorative text).
   - For each entry give: hex, name, an approximate WCAG contrast ratio against the theme background (a number like 12.4), and a one-phrase recommended use (e.g. "primary headline", "body text", "muted caption").

Respond with STRICT JSON only. No markdown fences, no commentary outside the JSON. Use this exact schema:

{
  "theme_color": {"hex": "#RRGGBB", "name": "..."},
  "theme_palette": [
    {"hex": "#RRGGBB", "name": "...", "role": "..."}
  ],
  "font_palette": [
    {"hex": "#RRGGBB", "name": "...", "contrast_ratio": 0.0, "use": "..."}
  ],
  "notes": "one short sentence summarizing the overall visual mood of the deck"
}
"""


def derive_palette(paths, claude_client=None) -> Path:
    """Ask Claude to derive a unified palette for the three styled bgs."""
    out_path = paths.styled_bg_dir / "palette.json"
    if out_path.exists():
        print(f"  palette.json already exists: {out_path}")
        return out_path

    image_paths = [paths.styled_bg_dir / f"{r}.png"
                   for r in ("opening", "content", "closing")]
    for p in image_paths:
        if not p.exists():
            raise FileNotFoundError(f"styled background missing: {p}")

    content = []
    for label, p in zip(("opening", "content", "closing"), image_paths):
        content.append({"type": "text", "text": f"Slide background ({label}.png):"})
        content.append(encode_image_b64(p))
    content.append({"type": "text", "text": _PALETTE_PROMPT})

    if claude_client is None:
        claude_client = anthropic_client()
    raw = call_claude(claude_client, content, model=CLAUDE_OPUS, max_tokens=2048)
    parsed = parse_json_loose(raw)
    payload = {
        "model": CLAUDE_OPUS,
        "image_dir": str(paths.styled_bg_dir),
        "images": [p.name for p in image_paths],
        "result": parsed,
        "raw_response": raw,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"  saved palette: {out_path}")
    if parsed is not None:
        theme = (parsed.get("theme_color") or {})
        print(f"  theme color: {theme.get('hex','?')} ({theme.get('name','?')})")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="override per-case output root (default: output/<case>/)")
    ap.add_argument("--skip-palette", action="store_true",
                    help="run only step 1a (gpt-image-2); skip 1b palette extraction")
    args = ap.parse_args()

    paths = case_paths(args.case, out_dir=args.out_dir)
    load_api_keys()
    style_prompt = load_style_prompt()

    pages = list_front_bg_pages(paths)
    if not pages:
        print(f"no front_bg pages under {paths.front_bg_dir}; "
              f"run step0_prepare_front_bg.py first", file=sys.stderr)
        return 1

    # Claude client is needed BEFORE the gpt-image-2 call because we use
    # Claude to pick the content (middle) page from the candidates.
    claude = anthropic_client()
    role_pages = select_role_pages(pages, claude)
    print("role assignments:")
    for r, p in role_pages.items():
        print(f"  {r:>7s} <- {p.name}")

    print(f"\nstyle prompt ({len(style_prompt)} chars), first line: "
          f"{style_prompt.splitlines()[0][:120]!r}")

    print("\n=== 1a. gpt-image-2 styled backgrounds ===")
    generate_styled_backgrounds(paths, style_prompt, role_pages)

    if args.skip_palette:
        print("\n[--skip-palette] skipping 1b Claude palette extraction")
        return 0

    print("\n=== 1b. Claude palette extraction ===")
    derive_palette(paths, claude)
    return 0


if __name__ == "__main__":
    sys.exit(main())
