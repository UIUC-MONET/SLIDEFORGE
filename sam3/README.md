# SAM3 — SlideForge component detector

Vendored copy of Meta's [SAM3](https://github.com/facebookresearch/sam3)
(the `sam3/` package, unmodified, under the [SAM License](LICENSE)),
plus the SlideForge-specific scripts that turn it into the Stage I
slide-component detector:

| File | Role |
|---|---|
| `infer_remove_components_overlap_priority.py` | Agent E: text-prompted grounded segmentation with overlap-priority post-processing; peels components off the slide and emits mask/bbox crops + a cleaned background |
| `infer_point_prompt.py` | point-prompt recovery path for layout-review missed regions |
| `tune_decoder.py` | decoder-only fine-tuning (30.4M trainable params, frozen VL backbone) on slide-component boxes labeled against the 306-class taxonomy in `../data/sam3_text_types_306.json` |
| `run_finetune.sh` | reference training config (2× RTX 4090, ~4.7 h, the released checkpoint's recipe) |

## Install

```bash
pip install -e .        # from this directory (torch must already be installed)
```

## Checkpoints

Two-checkpoint scheme, both placed in `checkpoints/` by
`../scripts/download_checkpoints.sh`:

* `sam3.pt` — base SAM3 ([facebook/sam3](https://huggingface.co/facebook/sam3))
* `sam3_slideforge.pt` — SlideForge fine-tuned decoder
  ([zoezheng126/slideforge-sam3-decoder](https://huggingface.co/zoezheng126/slideforge-sam3-decoder);
  mean IoU 0.873 on a slide-disjoint held-out split, 95.3% of
  predictions ≥ IoU 0.5)

The decomposition pipeline invokes this detector either as a subprocess
or through the persistent worker (`../decomposition/sam3_worker.py`),
which loads the ~6.7 GB of weights once and serves jobs from a file
queue.

Note: `sam3/agent/` (Meta's SAM3-Agent) is not used by SlideForge and is
retained only for upstream completeness.
