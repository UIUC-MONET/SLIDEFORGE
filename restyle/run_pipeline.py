"""End-to-end orchestrator for SlideForge theme-preserving reconstruction.

Runs Stage III (style adaptation) + Stage IV (verification & self-repair)
over the Deck State Graph state ingested by ``step0_ingest_dsg.py``.

Usage:
    # 1) ingest the Stage I Deck State Graph into a case workspace
    python step0_ingest_dsg.py --run-dir <decomposition run dir> \
        --out-dir output/<case>

    # 2) run the reconstruction pipeline (release configuration)
    export CLAUDE_BACKEND=cli FONT_SIZING=deterministic COVERAGE_GATE=hard
    python run_pipeline.py --case <case> [--out-dir output/<case>] \
        [--from-step 1] [--to-step 6]

Steps:
    1  step1_generate_backgrounds    (OpenAI gpt-image-2 + Claude palette
                                      -> the deck-level aesthetic contract)
    2  step2_categorize              (Claude: reconstruction-modality
                                      assignment, classes 0-5)
    P  measure_text_ink + class4_ocr_inventory   (easyocr priors; runs
                                      under SLIDECODER_PYTHON)
    3  step3_class1/2/3/4/5          (modality-aware reconstruction)
    4  step4_color_restyle           (palette-constrained recoloring)
    4b step4b_contrast_audit         (deterministic WCAG>=3.0 audit)
    5  step5_compose                 (deck assembly; deterministic layering)
    6  render + step5b_selfrepair    (rendered-state verification and
                                      OCR-diff self-repair) + re-render

Release configuration (matches the paper):
    CLAUDE_BACKEND=cli|api  FONT_SIZING=deterministic  COVERAGE_GATE=hard
    step5_compose runs with --no-critic --no-font-critic (bit-reproducible
    composition); self-repair only touches pages below 90% OCR coverage.
Set SKIP_OPENAI=1 to skip the gpt-image-2 dependent steps (step1, class5)
when reusing pre-generated backgrounds.

Each step is idempotent at the per-file level, so it's safe to re-run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent

SLIDECODER_PYTHON = os.environ.get("SLIDECODER_PYTHON", sys.executable)

# (step number, script, extra args, interpreter override)
STEPS = [
    (1, "step1_generate_backgrounds.py", [], None),
    (2, "step2_categorize.py", [], None),
    (3, "step3_class1.py", [], None),
    (3, "step3_class2.py", [], None),
    (3, "step3_class3.py", [], None),
    (3, "step3_class4.py", [], None),
    (3, "step3_class5.py", [], None),
    (4, "step4_color_restyle.py", [], None),
    (4, "step4b_contrast_audit.py", [], None),
    (5, "step5_compose.py", ["--no-critic", "--no-font-critic"], None),
]

OPENAI_STEPS = {"step1_generate_backgrounds.py", "step3_class5.py"}


def run_one(script: str, args: list[str], interpreter: str | None = None) -> int:
    cmd = [interpreter or sys.executable, str(PIPELINE_DIR / script), *args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def run_ocr_prereqs(out_dir: Path) -> int:
    """easyocr ink priors for deterministic sizing + the class-4 hard
    coverage gate. Runs once per front_bg page dir, under the easyocr
    interpreter (SLIDECODER_PYTHON)."""
    front_bg = out_dir / "front_bg"
    fails = 0
    for page_dir in sorted(p for p in front_bg.iterdir() if p.is_dir()):
        for script in ("measure_text_ink.py", "class4_ocr_inventory.py"):
            rc = run_one(script, [str(page_dir)], SLIDECODER_PYTHON)
            if rc != 0:
                print(f"[prereq] {script} failed on {page_dir.name} rc={rc}",
                      file=sys.stderr)
                fails += 1
    return fails


def run_verification(case: str, out_dir: Path) -> int:
    """Stage IV: render, OCR-diff self-repair, re-render, measure."""
    render_dir = out_dir / "render"
    final_pptx = out_dir / "final.pptx"
    render_script = str(PIPELINE_DIR / "metrics" / "render_pptx_pages.py")
    ocr_diff_script = str(PIPELINE_DIR / "metrics" / "ocr_diff.py")

    fails = 0
    for cmd in (
        [sys.executable, render_script, str(final_pptx), str(render_dir)],
        [SLIDECODER_PYTHON, str(PIPELINE_DIR / "step5b_selfrepair.py"),
         "--case", case, "--out-dir", str(out_dir)],
        # self-repair mutates final.pptx in place -> re-render before measuring
        ["rm", "-rf", str(render_dir)],
        [sys.executable, render_script, str(final_pptx), str(render_dir)],
        [SLIDECODER_PYTHON, ocr_diff_script, str(out_dir), str(render_dir),
         str(out_dir / "ocr_diff.json")],
    ):
        print(f"\n$ {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            fails += 1
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="case workspace (default: output/<case>/)")
    ap.add_argument("--from-step", type=int, default=1,
                    help="first step to run (inclusive); default 1")
    ap.add_argument("--to-step", type=int, default=6,
                    help="last step to run (inclusive); default 6")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir \
        else PIPELINE_DIR / "output" / args.case
    if not (out_dir / "front_bg").is_dir():
        print(f"front_bg/ missing under {out_dir} — run step0_ingest_dsg.py "
              f"first (see module docstring).", file=sys.stderr)
        return 1

    skip_openai = bool(os.environ.get("SKIP_OPENAI"))
    fail = 0
    prereqs_done = False
    for step, script, extra, interp in STEPS:
        if step < args.from_step or step > args.to_step:
            continue
        if skip_openai and script in OPENAI_STEPS:
            print(f"[skip] {script} (SKIP_OPENAI=1)")
            continue
        # easyocr priors must exist before the class reconstructions.
        if step == 3 and not prereqs_done:
            fail += run_ocr_prereqs(out_dir)
            prereqs_done = True
        rc = run_one(script, ["--case", args.case, "--out-dir", str(out_dir),
                              *extra], interp)
        if rc != 0:
            print(f"[step {step}] {script} failed with rc={rc}", file=sys.stderr)
            fail += 1
    if args.from_step <= 6 <= args.to_step:
        fail += run_verification(args.case, out_dir)

    if fail:
        print(f"\nPipeline finished with {fail} failing script(s).",
              file=sys.stderr)
        return 1
    print(f"\nPipeline finished OK for case={args.case}. "
          f"Final deck: {out_dir / 'final.pptx'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
