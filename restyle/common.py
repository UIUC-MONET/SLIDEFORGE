"""Shared utilities for the unified slide-style-transfer pipeline.

Single entry point for:
  * Resolving per-case paths (input data + per-case output tree).
  * Loading API keys from ``api_keys.txt`` and exporting them as env vars
    that the openai / anthropic SDKs pick up automatically.
  * Loading the user-editable style prompt from ``style_prompt.txt``.
  * Tiny Claude / OpenAI client factories with retry, plus a few helpers
    used by every step (image base64 encoding, JSON parsing, hex
    normalisation, LibreOffice render).
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PIPELINE_DIR = Path(__file__).resolve().parent
ROOT = PIPELINE_DIR.parent  # SlideForge release root

API_KEYS_FILE = PIPELINE_DIR / "api_keys.txt"
STYLE_PROMPT_FILE = PIPELINE_DIR / "style_prompt.txt"

OUTPUT_ROOT = PIPELINE_DIR / "output"

DPI = 96
# Minimum readable on-slide font size. Below this the textbox is
# illegible at presentation distance even on a 1920x1440 canvas.
MIN_READABLE_PT = 10.0
EMU_PER_PX = 914400 // DPI  # 9525 EMU/px at 96 DPI

CLAUDE_OPUS = "claude-opus-4-7"
CLAUDE_SONNET = "claude-sonnet-4-6"

OPENAI_IMAGE_MODEL = "gpt-image-2"

LIBREOFFICE_BIN = "libreoffice"
LIBREOFFICE_TIMEOUT_S = 180


@dataclass(frozen=True)
class CasePaths:
    """All output paths for a single deck case."""

    case_id: str
    out_dir: Path                # output/<case>/
    front_bg_dir: Path           # output/<case>/front_bg/
    styled_bg_dir: Path          # output/<case>/styled_bg/
    step3_class1_dir: Path
    step3_class2_dir: Path
    step3_class3_dir: Path
    step3_class4_dir: Path
    step3_class5_dir: Path
    step4_dir: Path              # output/<case>/step4_recolor/
    final_pptx: Path             # output/<case>/final.pptx


def case_paths(case_id: str, out_dir: Path | str | None = None) -> CasePaths:
    """Resolve all per-case paths.

    By default the per-case output root is ``output/<case_id>/`` under the
    pipeline dir. Pass ``out_dir`` to redirect everything to a custom
    location (useful for one-off debugging runs)."""
    case_id = str(case_id)
    out = Path(out_dir) if out_dir is not None else OUTPUT_ROOT / case_id
    return CasePaths(
        case_id=case_id,
        out_dir=out,
        front_bg_dir=out / "front_bg",
        styled_bg_dir=out / "styled_bg",
        step3_class1_dir=out / "step3_class1",
        step3_class2_dir=out / "step3_class2",
        step3_class3_dir=out / "step3_class3",
        step3_class4_dir=out / "step3_class4",
        step3_class5_dir=out / "step3_class5",
        step4_dir=out / "step4_recolor",
        final_pptx=out / "final.pptx",
    )


# ---------------------------------------------------------------------------
# API keys + style prompt
# ---------------------------------------------------------------------------

def load_api_keys() -> dict[str, str]:
    """Populate os.environ with API keys and return the parsed dict.

    Keys are read from ``api_keys.txt`` next to this file when it exists
    (``KEY=value`` lines, ``#`` comments — see ``api_keys.txt.example``);
    keys already present in the environment are also accepted, so exporting
    ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` in the shell works without
    any file. The openai and anthropic SDKs pick both up from the
    environment automatically.
    """
    keys: dict[str, str] = {}
    if API_KEYS_FILE.exists():
        for line in API_KEYS_FILE.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v:
                keys[k] = v
                os.environ[k] = v
    required = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    # SKIP_OPENAI=1: run without the OpenAI-dependent steps (step1 background
    # generation and step3 class-5 repaint); the key is then not required.
    if os.environ.get("SKIP_OPENAI"):
        required.remove("OPENAI_API_KEY")
    # CLI backend uses a Claude Code subscription, not the API key.
    if os.environ.get("CLAUDE_BACKEND", "").lower() == "cli":
        required.remove("ANTHROPIC_API_KEY")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"missing API keys: {missing}. Export them in the environment "
            f"or add them to {API_KEYS_FILE} (see api_keys.txt.example)."
        )
    return keys


def load_style_prompt() -> str:
    # Allow a wrapping shell script to swap in a different prompt file
    # via env var, so the same pipeline can be driven through multiple
    # styles in one run without editing style_prompt.txt.
    override = os.environ.get("STYLE_PROMPT_FILE_OVERRIDE", "").strip()
    path = Path(override) if override else STYLE_PROMPT_FILE
    if not path.exists():
        raise FileNotFoundError(f"style prompt file missing: {path}")
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Claude client + retry
# ---------------------------------------------------------------------------

# Retry waits for transient errors: 408/429/5xx (incl. 529 Overloaded) and
# connection errors. Exhausted -> reraise.
_RETRY_DELAYS_S = (5, 15, 45, 90, 180)


def anthropic_client():
    load_api_keys()
    import anthropic
    return anthropic.Anthropic(max_retries=5)


def call_claude(client, content, model: str = CLAUDE_OPUS,
                max_tokens: int = 2048) -> str:
    """Send a single user message of ``content`` blocks to Claude.

    ``content`` is the list of {type:"text"|"image", ...} blocks. Returns
    the concatenated text of every text block in the response.

    ``CLAUDE_BACKEND=cli`` routes the call through the non-interactive
    ``claude -p`` CLI (Claude Code subscription) instead of the API; the
    ``client`` argument is ignored in that mode."""
    if os.environ.get("CLAUDE_BACKEND", "").lower() == "cli":
        return _call_claude_cli(content, model=model)
    import anthropic
    for attempt in range(len(_RETRY_DELAYS_S) + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            text_blocks = [
                b.text for b in resp.content
                if getattr(b, "type", None) == "text"
            ]
            return "\n".join(text_blocks).strip()
        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", None)
            retriable = status in (408, 429) or (status is not None and status >= 500)
            if not retriable or attempt >= len(_RETRY_DELAYS_S):
                raise
            delay = _RETRY_DELAYS_S[attempt]
            print(f"    [api {status}] {e.__class__.__name__}; "
                  f"retrying in {delay}s (attempt {attempt+1}/{len(_RETRY_DELAYS_S)})",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
        except anthropic.APIConnectionError as e:
            if attempt >= len(_RETRY_DELAYS_S):
                raise
            delay = _RETRY_DELAYS_S[attempt]
            print(f"    [conn {e.__class__.__name__}] retrying in {delay}s "
                  f"(attempt {attempt+1}/{len(_RETRY_DELAYS_S)})",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
    raise RuntimeError("call_claude: retry loop exited unexpectedly")


# ---------------------------------------------------------------------------
# claude -p CLI backend (Claude Code subscription — no API billing)
# ---------------------------------------------------------------------------

CLAUDE_CLI_BIN = os.environ.get("CLAUDE_CLI_BIN", "claude")
_CLI_SYSTEM_PROMPT = (
    "You are being used as a drop-in replacement for a raw model API call "
    "inside an automated pipeline. Read any referenced image files with the "
    "Read tool, then respond with the final answer only, in exactly the "
    "format the task asks for — no preamble, no commentary, no mention of "
    "reading files."
)
_CLI_MAX_ATTEMPTS = 7
_CLI_TIMEOUT_S = 600


def _content_to_cli_prompt(content, tmp_dir: Path) -> str:
    """Flatten anthropic content blocks into a single CLI prompt string.

    Image blocks are written to ``tmp_dir`` and replaced inline by a marker
    referencing the absolute path, preserving text/image ordering."""
    parts: list[str] = []
    img_n = 0
    for block in content:
        btype = block.get("type") if isinstance(block, dict) else None
        if btype == "text":
            parts.append(block["text"])
        elif btype == "image":
            img_n += 1
            src = block["source"]
            ext = ".jpg" if src.get("media_type") == "image/jpeg" else ".png"
            img_path = tmp_dir / f"image_{img_n:02d}{ext}"
            img_path.write_bytes(base64.standard_b64decode(src["data"]))
            parts.append(
                f"[image {img_n} — read this file with the Read tool: {img_path}]"
            )
        else:
            raise ValueError(f"unsupported content block for CLI backend: {btype}")
    return "\n".join(parts)


def _call_claude_cli(content, model: str = CLAUDE_OPUS) -> str:
    """Non-interactive ``claude -p`` subprocess call (subscription-billed).

    Mirrors the validated ClaudeCLIBackend in pipeline/backends.py: JSON
    output envelope, Read-tool image input, 7-attempt exponential backoff
    that survives multi-minute CLI outages."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="claude_cli_"))
    try:
        prompt = _content_to_cli_prompt(content, tmp_dir)
        cmd = [
            CLAUDE_CLI_BIN,
            "-p",
            "--model", model,
            "--output-format", "json",
            "--append-system-prompt", _CLI_SYSTEM_PROMPT,
            "--tools", "Read",
            "--permission-mode", "bypassPermissions",
            prompt,
        ]
        # Never let the CLI fall back to API-key billing: this backend exists
        # precisely to use the subscription account.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        last_error: Exception | None = None
        for attempt in range(_CLI_MAX_ATTEMPTS):
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=_CLI_TIMEOUT_S, check=False, env=env,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"claude CLI exit {proc.returncode}: "
                        f"stderr={proc.stderr[:600]!r}"
                    )
                envelope = json.loads(proc.stdout)
                result = envelope.get("result")
                if not isinstance(result, str) or not result.strip():
                    raise RuntimeError(
                        f"claude CLI returned no usable text. "
                        f"keys={list(envelope.keys())}"
                    )
                return result.strip()
            except (subprocess.TimeoutExpired, RuntimeError,
                    json.JSONDecodeError) as e:
                last_error = e
                if attempt < _CLI_MAX_ATTEMPTS - 1:
                    delay = min(300, 15 * (2 ** attempt))
                    print(
                        f"    [claude_cli] call failed "
                        f"(attempt {attempt+1}/{_CLI_MAX_ATTEMPTS}): {e}. "
                        f"Retrying in {delay}s...",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(delay)
        raise RuntimeError(
            f"claude CLI failed after {_CLI_MAX_ATTEMPTS} attempts: {last_error}"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Deterministic font sizing (H1+H2): easyocr ink prior + Carlito max-fit
# ---------------------------------------------------------------------------

# LibreOffice substitutes Carlito for Calibri (metric-compatible), so PIL
# measurements with Carlito match what the final render will do.
_CARLITO_DIR = Path("/usr/share/fonts/truetype/crosextra")
_CARLITO = {
    (False, False): _CARLITO_DIR / "Carlito-Regular.ttf",
    (True, False): _CARLITO_DIR / "Carlito-Bold.ttf",
    (False, True): _CARLITO_DIR / "Carlito-Italic.ttf",
    (True, True): _CARLITO_DIR / "Carlito-BoldItalic.ttf",
}

# easyocr detection-box height / font em size for Carlito renders, measured
# by measure_text_ink.py --calibrate (n=20, median 1.189, IQR 1.11-1.27).
# Targeting prior_em = ocr_box_h / this constant makes the *measured*
# rendered text height equal the *measured* original height (M1 -> 1.0)
# independent of the original font family.
EASYOCR_BOX_PER_EM = 1.19

# Interpreter of the environment that has easyocr (+ GPU torch) installed.
# OCR-based measurement runs in a subprocess under this interpreter so the
# main pipeline env stays light. Defaults to the current interpreter.
SLIDECODER_PYTHON = os.environ.get("SLIDECODER_PYTHON", sys.executable)


def font_sizing_mode() -> str:
    """"vlm" (baseline behavior) or "deterministic" (H1+H2)."""
    return os.environ.get("FONT_SIZING", "vlm").strip().lower()


def _load_carlito(em_px: int, bold: bool, italic: bool):
    from PIL import ImageFont
    path = _CARLITO.get((bool(bold), bool(italic)), _CARLITO[(False, False)])
    return ImageFont.truetype(str(path), em_px)


def _wrap_lines(text: str, font, max_w_px: float) -> tuple[list[str], bool]:
    """Greedy word-wrap each explicit line to ``max_w_px``. Returns
    (wrapped lines, fits_width) — fits_width False when a single word is
    wider than the box (no font size can be *wrapped* into fitting)."""
    out: list[str] = []
    fits = True
    for line in (text.split("\n") if text else [""]):
        cur = ""
        for w in line.split(" "):
            if font.getlength(w) > max_w_px:
                fits = False
            cand = w if not cur else cur + " " + w
            if not cur or font.getlength(cand) <= max_w_px:
                cur = cand
            else:
                out.append(cur)
                cur = w
        out.append(cur)
    return out, fits


def fit_font_size_pt(text: str, bbox_w_px: float, bbox_h_px: float,
                     bold: bool = False, italic: bool = False,
                     ink_h_px: float | None = None,
                     width_safety: float | None = None) -> tuple[float, dict]:
    """Deterministic font size for a textbox (H1+H2).

    * max-fit (H1): largest pt whose greedy word-wrap at the bbox width
      keeps every line within the width AND total height (lines x
      ascent+descent, matching LibreOffice single spacing) within the
      bbox height — binary search on em pixels with Carlito metrics.
    * ink prior (H2): when ``ink_h_px`` (median easyocr text-box height
      measured on the ORIGINAL component crop) is given, target
      em = ink_h_px / EASYOCR_BOX_PER_EM so the rendered text measures
      the same height as the original.

    final = min(prior, maxfit), floored at MIN_READABLE_PT.
    Returns (pt, debug dict)."""
    bbox_w_px = max(1.0, float(bbox_w_px))
    bbox_h_px = max(1.0, float(bbox_h_px))
    text = text or ""
    # R1 (refined): math/unicode-heavy glyphs render in fallback fonts whose
    # widths Carlito/PIL underestimates -> inflate measured widths instead of
    # abandoning deterministic sizing (the ink prior stays font-independent).
    if width_safety is None:
        width_safety = 1.45 if carlito_metrics_unreliable(text) else 1.0
    eff_w_px = bbox_w_px / max(1.0, width_safety)
    n_explicit_lines = len(text.split("\n")) if text else 1

    def fits(em_px: int) -> bool:
        font = _load_carlito(em_px, bold, italic)
        asc, desc = font.getmetrics()
        wrapped, fits_w = _wrap_lines(text, font, eff_w_px)
        if not fits_w:
            return False
        if len(wrapped) == 1:
            # Tight SAM bboxes hug the glyph INK, not the typographic line
            # box (asc+desc ≈ 1.22em) — requiring the full line box to fit
            # caps single lines ~20% small. LibreOffice renders the vertical
            # line-box overhang outside the frame without clipping, so ink
            # containment is the right constraint for one line.
            l, t, r, b = font.getbbox(wrapped[0])
            return (b - t) <= bbox_h_px
        # For safety-inflated text, forbid only wraps BEYOND the explicit
        # line count: a width misestimate must not add soft-wrapped lines,
        # but text that is multi-line by construction (bullet lists) must
        # still be sizable — forbidding all multi-line fits collapsed
        # maxfit to the floor (paper_9_2 p0005 M1 0.956->0.155).
        if width_safety > 1.0 and len(wrapped) > n_explicit_lines:
            return False
        return len(wrapped) * (asc + desc) <= bbox_h_px

    lo, hi = 4, 400  # em px
    if not fits(lo):
        maxfit_em = lo
    else:
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
        maxfit_em = lo
    maxfit_pt = maxfit_em * 72.0 / DPI

    prior_pt = None
    if ink_h_px and ink_h_px > 0:
        prior_em = float(ink_h_px) / EASYOCR_BOX_PER_EM
        prior_pt = prior_em * 72.0 / DPI

    final_pt = min(maxfit_pt, prior_pt) if prior_pt else maxfit_pt
    final_pt = max(MIN_READABLE_PT, final_pt)
    return final_pt, {
        "maxfit_pt": round(maxfit_pt, 2),
        "ink_prior_pt": round(prior_pt, 2) if prior_pt else None,
        "final_pt": round(final_pt, 2),
        "ink_h_px": round(float(ink_h_px), 1) if ink_h_px else None,
        "bbox_w_px": round(bbox_w_px, 1), "bbox_h_px": round(bbox_h_px, 1),
    }


_CARLITO_UNRELIABLE_RE = re.compile(
    "["
    "\u0300-\u036f"   # combining diacritics (f\u0302)
    "\u1d2c-\u1dbf"   # phonetic modifier sub/superscript letters
    "\u2016"           # double vertical bar
    "\u2070-\u209f"   # super/subscript digits and letters
    "\u2100-\u214f"   # letterlike symbols
    "\u2190-\u21ff"   # arrows
    "\u2200-\u22ff"   # mathematical operators
    "\u2300-\u23ff"   # misc technical
    "\u27c0-\u27ef"
    "\u2980-\u29ff"
    "\u2a00-\u2aff"
    "]")


def carlito_metrics_unreliable(text: str) -> bool:
    """R1: True when text contains glyphs whose rendered width the
    Carlito/PIL measurement cannot predict (combining marks, sub/
    superscripts, math operators). LibreOffice substitutes fallback
    fonts for these, so deterministic max-fit must not be trusted --
    callers fall back to the legacy VLM-estimate+clamp path."""
    return bool(_CARLITO_UNRELIABLE_RE.search(text or ""))


def load_text_metrics(page_dir: Path) -> dict:
    """Per-component easyocr ink measurements for a front_bg page dir.

    Reads ``<page_dir>/text_metrics.json``; if missing, generates it by
    running measure_text_ink.py under SLIDECODER_PYTHON (easyocr + GPU).
    Returns {} when measurement is impossible (caller falls back to
    bbox-only max-fit)."""
    page_dir = Path(page_dir)
    out = page_dir / "text_metrics.json"
    if not out.exists():
        script = PIPELINE_DIR / "measure_text_ink.py"
        try:
            subprocess.run(
                [SLIDECODER_PYTHON, str(script), str(page_dir)],
                check=True, timeout=600,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"  [warn] text_metrics generation failed for {page_dir.name}: {e}",
                  file=sys.stderr, flush=True)
            return {}
    try:
        return json.loads(out.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# OpenAI client (for gpt-image-2 image edit / generation)
# ---------------------------------------------------------------------------

def openai_client():
    load_api_keys()
    from openai import OpenAI
    return OpenAI()


# ---------------------------------------------------------------------------
# Image / JSON helpers
# ---------------------------------------------------------------------------

# Anthropic limits per-image base64 to 5 MB. Base64 inflates raw bytes by
# 4/3, so the raw payload must stay under ~3.75 MB to be safe.
_ANTHROPIC_B64_MAX_BYTES = 5 * 1024 * 1024
_RAW_SAFE_MAX_BYTES = _ANTHROPIC_B64_MAX_BYTES * 3 // 4 - 4096  # small margin


def encode_image_b64(path: Path) -> dict:
    """Return an anthropic ``image`` content block for ``path`` (PNG).

    Re-encodes in-memory (JPEG at decreasing quality / size) when the raw
    bytes would inflate past the Anthropic 5 MB per-image base64 cap. The
    original file on disk is never modified.
    """
    raw = Path(path).read_bytes()
    is_jpeg = str(path).lower().endswith((".jpg", ".jpeg"))
    media_type = "image/jpeg" if is_jpeg else "image/png"
    if len(raw) <= _RAW_SAFE_MAX_BYTES:
        data = base64.standard_b64encode(raw).decode("utf-8")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }

    # Oversized — re-encode in-memory at decreasing quality / size.
    from io import BytesIO
    with Image.open(path) as im:
        im = im.convert("RGB")
        max_w, max_h = 2048, 1536  # plenty for VLM judgment
        if im.width > max_w or im.height > max_h:
            im.thumbnail((max_w, max_h), Image.LANCZOS)
        for quality in (90, 80, 70, 60, 50):
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            blob = buf.getvalue()
            if len(blob) <= _RAW_SAFE_MAX_BYTES:
                data = base64.standard_b64encode(blob).decode("utf-8")
                return {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg",
                               "data": data},
                }
        # Last-ditch: shrink to 1024 max edge then quality 50.
        im.thumbnail((1024, 1024), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=50, optimize=True)
        blob = buf.getvalue()
        data = base64.standard_b64encode(blob).decode("utf-8")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": data},
        }


def parse_json_loose(raw: str):
    """Parse a JSON object out of ``raw``, even if it's wrapped in markdown
    fences or surrounded by stray prose. Returns ``None`` on failure."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        s = m.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _ascii_digits(s: str) -> str:
    """Convert any Unicode digit char (Devanagari ३, Arabic-Indic ٣, full-width
    ３, etc.) to its ASCII equivalent. Non-digit chars pass through."""
    out_chars = []
    for c in s:
        d = unicodedata.digit(c, -1)
        out_chars.append(str(d) if d >= 0 else c)
    return "".join(out_chars)


def hex_norm(s):
    """Normalize a hex colour string to 6 uppercase ASCII hex characters
    (without the leading ``#``). Returns ``None`` if the value cannot be
    interpreted as a valid hex colour.

    Defends against three classes of model output noise:
      * leading ``#`` and surrounding whitespace,
      * 3-char shorthand (``#abc`` -> ``aabbcc``),
      * non-ASCII look-alike characters: Devanagari / Arabic-Indic / full-width
        digits, plus full-width A-F letters, all of which Claude has been
        observed to emit occasionally and which would otherwise fail the
        ``[0-9A-F]`` regex.
    """
    if s is None:
        return None
    t = unicodedata.normalize("NFKC", str(s)).strip()
    t = _ascii_digits(t)
    t = t.lstrip("#").upper()
    if len(t) == 3:
        t = "".join(c * 2 for c in t)
    if len(t) != 6 or not re.fullmatch(r"[0-9A-F]{6}", t):
        return None
    return t


def hex_to_rgb(hx):
    t = hex_norm(hx)
    if t is None:
        return None
    return tuple(int(t[i:i + 2], 16) for i in (0, 2, 4))


def px_to_emu(p: float) -> int:
    return int(round(p * EMU_PER_PX))


# ---------------------------------------------------------------------------
# LibreOffice render
# ---------------------------------------------------------------------------

def render_pptx_to_png(pptx_path: Path, slide_w_px: int, slide_h_px: int) -> Path:
    """Headless LibreOffice convert pptx -> png, resized to slide pixels."""
    work = Path(tempfile.mkdtemp(prefix="pptx_render_"))
    profile = Path(tempfile.mkdtemp(prefix="lo_profile_"))
    try:
        # Anaconda exports LD_LIBRARY_PATH pointing at its own libs, which
        # makes the system libreoffice fail to load its own libs. Strip it.
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        subprocess.run(
            [
                LIBREOFFICE_BIN,
                "--headless",
                f"-env:UserInstallation=file://{profile}",
                "--convert-to", "png",
                "--outdir", str(work),
                str(pptx_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=LIBREOFFICE_TIMEOUT_S,
            env=env,
        )
        pngs = list(work.glob("*.png"))
        if not pngs:
            raise RuntimeError(f"libreoffice produced no PNG from {pptx_path}")
        img = Image.open(pngs[0]).convert("RGB").resize(
            (slide_w_px, slide_h_px), Image.LANCZOS
        )
        out_path = pptx_path.with_name(pptx_path.stem + "_rendered.png")
        img.save(out_path)
        return out_path
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(profile, ignore_errors=True)


def crop_bbox(image_path: Path, bbox, out_path: Path) -> Path:
    img = Image.open(image_path).convert("RGBA")
    x0, y0, x1, y1 = bbox
    x0i = max(0, int(x0))
    y0i = max(0, int(y0))
    x1i = min(img.width, int(round(x1)))
    y1i = min(img.height, int(round(y1)))
    img.crop((x0i, y0i, x1i, y1i)).save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------

def list_front_bg_pages(paths: CasePaths) -> list[Path]:
    if not paths.front_bg_dir.is_dir():
        return []
    return sorted(p for p in paths.front_bg_dir.iterdir() if p.is_dir())


def page_role(page_index: int, n_pages: int) -> str:
    """Return ``"opening" | "content" | "closing"`` for a 0-indexed page."""
    if n_pages <= 0:
        return "content"
    if page_index == 0:
        return "opening"
    if page_index == n_pages - 1:
        return "closing"
    return "content"


def styled_bg_for_role(paths: CasePaths, role: str) -> Path:
    """Path to the styled background PNG for a given page role."""
    return paths.styled_bg_dir / f"{role}.png"


__all__ = [
    "PIPELINE_DIR", "ROOT",
    "API_KEYS_FILE", "STYLE_PROMPT_FILE", "OUTPUT_ROOT",
    "carlito_metrics_unreliable",
    "DPI", "MIN_READABLE_PT", "EMU_PER_PX", "CLAUDE_OPUS", "CLAUDE_SONNET",
    "OPENAI_IMAGE_MODEL", "LIBREOFFICE_BIN", "LIBREOFFICE_TIMEOUT_S",
    "CasePaths", "case_paths",
    "load_api_keys", "load_style_prompt",
    "anthropic_client", "call_claude", "openai_client",
    "encode_image_b64", "parse_json_loose", "hex_norm", "hex_to_rgb",
    "px_to_emu", "render_pptx_to_png", "crop_bbox",
    "list_front_bg_pages", "page_role", "styled_bg_for_role",
]
