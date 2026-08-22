# Stage III/IV — Theme-preserving reconstruction & verification

Restyles a deck under a target visual theme while preserving content,
layout, and native PPTX editability. Operates on the Deck State Graph
state produced by Stage I: instead of regenerating slides as flat images,
every foreground component node is rebuilt with the document primitive
that best preserves its function.

## Steps

| Step | Script | Stage in paper |
|---|---|---|
| 0 | `step0_ingest_dsg.py` | flatten the DSG into per-page workspaces (`front_bg/<page>/`) |
| 1 | `step1_generate_backgrounds.py` | **aesthetic contract**: gpt-image-2 styled backgrounds (opening/content/closing) + Claude palette extraction with WCAG-annotated font palette |
| 2 | `step2_categorize.py` | **modality assignment** τ ∈ {0..5}: discardable ornament / editable textbox / text panel / native table / shape composition / styled raster |
| P | `measure_text_ink.py`, `class4_ocr_inventory.py` | easyocr priors for deterministic sizing + the class-4 coverage gate |
| 3 | `step3_class1..5.py` | **modality-aware reconstruction**, each with a render→verify→adjust loop |
| 4 | `step4_color_restyle.py` | **palette-constrained assembly**: functionally consistent color mapping (identical source colors → identical theme colors) |
| 4b | `step4b_contrast_audit.py` | deterministic local WCAG ≥ 3.0 contrast audit (zero VLM calls) |
| 5 | `step5_compose.py` | deck assembly at DSG anchors; deterministic layering (text topmost); bit-reproducible with critics off |
| 6 | `step5b_selfrepair.py` + `metrics/` | **rendered-state verification**: OCR-diff self-repair on pages below 90% coverage, then re-render and measure |

## Run

```bash
# ingest a Stage I run
python step0_ingest_dsg.py --run-dir ../runs/demo --out-dir output/demo

# pick a theme (or write your own style_prompt.txt)
cp style_prompt_industrial.txt style_prompt.txt

# release configuration
export CLAUDE_BACKEND=cli FONT_SIZING=deterministic COVERAGE_GATE=hard
python run_pipeline.py --case demo --out-dir output/demo
```

Output: `output/demo/final.pptx` (natively editable), `render/` (page
PNGs), `ocr_diff.json` (per-page OCR coverage, text height ratio, missing
words), `selfrepair.json` (what Stage IV repaired).

## Configuration surface

| Env var | Effect |
|---|---|
| `CLAUDE_BACKEND=cli` | route Claude calls through the `claude` CLI (subscription) instead of the API |
| `FONT_SIZING=deterministic` | Carlito max-fit + easyocr ink prior sizing (release default; `vlm` = legacy estimate) |
| `COVERAGE_GATE=hard` | class-4 reconstructions must pass an OCR coverage gate before acceptance |
| `SKIP_OPENAI=1` | skip gpt-image-2 steps (step1, class-5) — reuse existing `styled_bg/` |
| `STYLE_PROMPT_FILE_OVERRIDE` | swap the theme prompt without editing `style_prompt.txt` |
| `SLIDECODER_PYTHON` | interpreter with easyocr installed (defaults to the current one) |
| `CLAUDE_CLI_BIN` | path to the `claude` binary (default `claude`) |

API keys: export `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, or copy
`api_keys.txt.example` → `api_keys.txt` (gitignored) and fill it in.

## Notes

* All geometry stays in original-slide pixel space; the PPTX canvas is
  built at source pixels @ 96 DPI (9525 EMU/px), so bboxes map 1:1.
* `step5_compose.py` runs with `--no-critic --no-font-critic` in the
  release configuration: composition is then deterministic
  (bit-reproducible), with verification handled by the deterministic
  contrast audit (4b) and OCR-diff self-repair (5b).
* Carlito fonts (`fonts-crosextra-carlito`) are a hard requirement for
  deterministic sizing — the easyocr box-per-em calibration constant was
  measured against Carlito renders.
