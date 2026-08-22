# Stage I — Deck State Graph construction

Perception-aligned slide decomposition: recovers the human-referable
component state of each rendered slide and grounds it for editing. The
output (`<run_dir>/<slide>/final/`) is the serialized component layer of
the **Deck State Graph** — build the full graph with
`python -m dsg.build_dsg --run-dir <run_dir>` from the repo root.

## How it works

Per slide, up to `--max_review_iters` iterations of:

| Agent | Role |
|---|---|
| **A** | VLM proposes short component phrases from the slide image |
| **B** | LLM maps phrases onto the fixed 306-class taxonomy |
| **C** | VLM selects taxonomy classes directly from the image |
| **D** | union of B and C → the slide-conditioned vocabulary Λ_p |
| **E** | SAM3 grounded segmentation, one text prompt per class (mask crops + cleaned background) |
| **M3** | mask-overlap probe + VLM arbitration; carves lying bboxes |
| **F** | per-crop validity verdicts (valid / text-only / OCR-LaTeX) and cleaned-image review that drives the loop |

Post-loop, per slide: cross-iteration dedup, layout review (**H**: find
missed regions, validate them, propose perceptual merge groups with
acceptance caps), polygon refinement, z-index assignment, and `final/`
materialization.

## Run

The one-command release configuration (persistent SAM3 worker + three-tier
model cascade + merge caps — the paper's headline config):

```bash
scripts/run_decomposition.sh <png dir | JSON list> <run_dir> [cuda:0]
```

Or drive `run_pipeline.py` directly (see `--help`; ~60 flags, all
research knobs default to the released configuration's values):

```bash
python decomposition/run_pipeline.py \
    --images <dir|json> --backend claude_cli \
    --f_validity_cascade --agent_cascade --judgment_cascade \
    --abc_model claude-haiku-4-5-20251001 \
    --merge_max_members 8 --merge_max_area_frac 0.35 --merge_area_min_members 6 \
    --run_dir runs/demo --device cuda:0
```

Backends: `claude` (Anthropic API), `claude_cli` (Claude Code
subscription), `openai`, `gemini`. Without the persistent worker
(`SAM3_WORKER_DIR` unset), each segmentation call reloads the 6.7 GB
checkpoints — start the worker for interactive use:

```bash
python decomposition/sam3_worker.py \
    --script sam3/infer_remove_components_overlap_priority.py \
    --ckpt sam3/checkpoints/sam3_slideforge.pt \
    --base_ckpt sam3/checkpoints/sam3.pt \
    --device cuda:0 --queue_dir /tmp/sam3_worker_queue
export SAM3_WORKER_DIR=/tmp/sam3_worker_queue
```

## Output

```
run_dir/<NNNN>_<slide>/
  iter_XX/                      per-iteration segmentation state
  final/
    metadata.json               component nodes: bbox, polygon, taxonomy
                                label, granularity, z_index, is_text_only,
                                ocr_latex
    components/*.png            mask-cut RGBA crops
    bbox_components/*.png       opaque bbox crops
    overlay.png                 visual summary
  summary.json
```

## Evaluation

`eval/run_eval.py` scores a run directory on the paper's state-recovery
paradigm: rendered reconstruction (MSE/PSNR/SSIM, optional LPIPS/CLIP),
SCAN coverage/redundancy, per-component VLM judgments (cohesion, label
fit), and holistic judgments (coverage, granularity, non-redundancy).
Ground truth for reconstruction metrics is extracted from source `.pptx`
files via `eval/decompose.py` (LibreOffice + poppler required).

`usage_report.py` aggregates the per-call `vlm_usage.jsonl` written by
every run into per-slide / per-agent cost tables.
