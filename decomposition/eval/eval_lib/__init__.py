"""Evaluation library for sam3 slide-decomposition pipeline output.

Implements the metrics recommended in metrics_survey.md:
  - reconstruction       (M1: LPIPS / CLIP / SSIM / MSE render-and-compare)
  - coverage             (M2: SCAN-style Coverage + Non-overlap)
  - component_judge      (M3: per-component VLM judge)
  - holistic_judge       (M4: holistic VLM judge on reconstruction)
"""
