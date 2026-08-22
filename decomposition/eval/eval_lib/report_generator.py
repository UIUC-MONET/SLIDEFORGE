"""Markdown report generator for slide-decomposition evaluation.

Converts the structural results.json summary into a beautifully formatted,
human-readable REPORT.md with aggregate tables and per-slide breakdown.
"""

from __future__ import annotations

import os
from datetime import datetime


def generate_report(summary: dict, output_dir: str) -> None:
    # Get general run info
    input_dir = summary.get("input", "Unknown")
    timestamp = summary.get("timestamp", "")
    try:
        # Convert timestamp like 20260514_123456 to human readable format
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        formatted_ts = dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        formatted_ts = timestamp

    vlm_provider = summary.get("vlm_provider") or "N/A"
    vlm_model = summary.get("vlm_model") or "default"
    num_slides = summary.get("num_slides", 0)
    input_base = os.path.basename(os.path.normpath(input_dir))

    # Compile aggregates
    aggs = summary.get("aggregates", {})
    recon_agg = aggs.get("reconstruction", {})
    cov_agg = aggs.get("coverage", {})
    comp_agg = aggs.get("component_judge", {})
    hol_agg = aggs.get("holistic_judge", {})

    def fmt_val(d, key, fmt="{:.4f}"):
        if not d:
            return "N/A"
        val = d.get(key)
        return fmt.format(val) if val is not None else "N/A"

    # Build the report
    md = []
    md.append(f"# Evaluation Report — `{input_base}`\n")
    md.append(f"**Run Date/Time**: {formatted_ts}  ")
    md.append(f"**Input Directory**: `{input_dir}`  ")
    md.append(f"**Total Slides Evaluated**: {num_slides}  ")
    if vlm_provider != "N/A":
        md.append(f"**VLM Judge**: `{vlm_provider}` (model: `{vlm_model}`)  ")
    md.append("\n## Aggregate Scores\n")
    md.append("| Metric | Mean | Std | Sample Count | Notes |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")

    # M1 Reconstruction
    md.append(f"| **M1 SSIM** | {fmt_val(recon_agg, 'mean_ssim')} | {fmt_val(recon_agg, 'std_ssim')} | {fmt_val(recon_agg, 'n_ssim', '{}')} | Higher is better |")
    md.append(f"| **M1 PSNR (dB)** | {fmt_val(recon_agg, 'mean_psnr', '{:.2f}')} | {fmt_val(recon_agg, 'std_psnr', '{:.2f}')} | {fmt_val(recon_agg, 'n_psnr', '{}')} | Higher is better |")
    md.append(f"| **M1 MSE** | {fmt_val(recon_agg, 'mean_mse', '{:.2f}')} | {fmt_val(recon_agg, 'std_mse', '{:.2f}')} | {fmt_val(recon_agg, 'n_mse', '{}')} | Lower is better |")
    md.append(f"| **M1 LPIPS** | {fmt_val(recon_agg, 'mean_lpips')} | {fmt_val(recon_agg, 'std_lpips')} | {fmt_val(recon_agg, 'n_lpips', '{}')} | Lower is better (optional) |")
    md.append(f"| **M1 CLIP Cosine** | {fmt_val(recon_agg, 'mean_clip_cosine')} | {fmt_val(recon_agg, 'std_clip_cosine')} | {fmt_val(recon_agg, 'n_clip_cosine', '{}')} | Higher is better (optional) |")

    # M2 Coverage
    md.append(f"| **M2 Coverage Ratio (R_c)** | {fmt_val(cov_agg, 'mean_coverage_ratio')} | {fmt_val(cov_agg, 'std_coverage_ratio')} | - | Higher is better |")
    md.append(f"| **M2 Non-overlap Ratio (R_o)** | {fmt_val(cov_agg, 'mean_non_overlap_ratio')} | {fmt_val(cov_agg, 'std_non_overlap_ratio')} | - | Higher is better |")
    md.append(f"| **M2 Combined SCAN Score** | {fmt_val(cov_agg, 'mean_combined_scan_score')} | {fmt_val(cov_agg, 'std_combined_scan_score')} | - | `0.9 * R_c + 0.1 * R_o` |")

    # M3 Component Judge
    md.append(f"| **M3 Cohesion (1-5)** | {fmt_val(comp_agg, 'mean_cohesion', '{:.2f}')} | {fmt_val(comp_agg, 'std_cohesion', '{:.2f}')} | - | Scale 1-5, higher is better |")
    md.append(f"| **M3 Label Fit (1-5)** | {fmt_val(comp_agg, 'mean_label_fit', '{:.2f}')} | {fmt_val(comp_agg, 'std_label_fit', '{:.2f}')} | - | Scale 1-5, higher is better |")

    # M4 Holistic Judge
    md.append(f"| **M4 Coverage (1-5)** | {fmt_val(hol_agg, 'mean_coverage', '{:.2f}')} | {fmt_val(hol_agg, 'std_coverage', '{:.2f}')} | - | Scale 1-5, higher is better |")
    md.append(f"| **M4 Granularity (1-5)** | {fmt_val(hol_agg, 'mean_granularity', '{:.2f}')} | {fmt_val(hol_agg, 'std_granularity', '{:.2f}')} | - | Scale 1-5, higher is better |")
    md.append(f"| **M4 Non-redundancy (1-5)** | {fmt_val(hol_agg, 'mean_non_redundancy', '{:.2f}')} | {fmt_val(hol_agg, 'std_non_redundancy', '{:.2f}')} | - | Scale 1-5, higher is better |")

    md.append("\n## Per-Slide Breakdown\n")

    per_slide = summary.get("per_slide", {})
    for slide_id, results in sorted(per_slide.items()):
        md.append(f"### `{slide_id}`")
        
        # Check if loading slide failed entirely
        if "_load_error" in results:
            md.append(f"\n> [!CAUTION]\n> **Failed to load slide**: {results['_load_error']}\n")
            continue

        recon = results.get("reconstruction", {})
        cov = results.get("coverage", {})
        comp = results.get("component_judge", {})
        holistic = results.get("holistic_judge", {})

        # Count final components
        num_components = "-"
        if "num_final_components" in recon:
            num_components = recon["num_final_components"]
        elif "num_final_components" in cov:
            num_components = cov["num_final_components"]
        elif "num_final_components" in holistic:
            num_components = holistic["num_final_components"]
        elif "num_judged" in comp:
            num_components = comp["num_judged"]

        md.append(f"\n*Components: {num_components} final components*\n")

        # Table header
        md.append("| Metric | Value | Details / Errors |")
        md.append("| :--- | :--- | :--- |")

        # SSIM / M1 metrics
        if recon:
            if "ssim" in recon:
                md.append(f"| **M1 SSIM** | {fmt_val(recon, 'ssim')} | PSNR: {fmt_val(recon, 'psnr', '{:.2f}')} dB, MSE: {fmt_val(recon, 'mse', '{:.2f}')} |")
            if "lpips" in recon:
                lpips_val = fmt_val(recon, 'lpips')
                lpips_skipped = recon.get("lpips_skipped")
                lpips_desc = f"Skipped: {lpips_skipped}" if lpips_skipped else "Computed"
                md.append(f"| **M1 LPIPS** | {lpips_val} | {lpips_desc} |")
            if "clip_cosine" in recon:
                clip_val = fmt_val(recon, 'clip_cosine')
                clip_skipped = recon.get("clip_skipped")
                clip_desc = f"Skipped: {clip_skipped}" if clip_skipped else "Computed"
                md.append(f"| **M1 CLIP Cosine** | {clip_val} | {clip_desc} |")

        # Coverage / SCAN / M2 metrics
        if cov and "coverage_ratio" in cov:
            md.append(f"| **M2 SCAN Score** | {fmt_val(cov, 'combined_scan_score')} | R_c (Coverage): {fmt_val(cov, 'coverage_ratio')}, R_o (Non-overlap): {fmt_val(cov, 'non_overlap_ratio')} |")

        # Component cohesion/fit / M3 metrics
        if comp and "mean_cohesion" in comp:
            md.append(f"| **M3 Component Judge** | Cohesion: {fmt_val(comp, 'mean_cohesion', '{:.2f}')}, Fit: {fmt_val(comp, 'mean_label_fit', '{:.2f}')} | Out of {comp.get('num_judged', 0)} components ({comp.get('num_failed', 0)} failed) |")

        # Holistic scores / M4 metrics
        if holistic and "coverage" in holistic:
            m4_scores = f"Cov: {fmt_val(holistic, 'coverage', '{}')} / Gran: {fmt_val(holistic, 'granularity', '{}')} / Non-Red: {fmt_val(holistic, 'non_redundancy', '{}')}"
            md.append(f"| **M4 Holistic Judge** | {m4_scores} | Scale 1-5 |")

        md.append("")  # empty line after table

        # M4 Reason
        if holistic and "reason" in holistic and holistic["reason"]:
            md.append(f"**M4 Reason**:")
            md.append(f"> {holistic['reason']}\n")

        # Individual metric level errors if any
        errors = []
        for m_name, m_res in results.items():
            if isinstance(m_res, dict) and "error" in m_res:
                errors.append(f"* **{m_name}** failed: `{m_res['error']}`")
        if errors:
            md.append("**Errors encountered during metrics evaluation**:")
            md.extend(errors)
            md.append("")

    md.append("\n## Output Directory Contents\n")
    md.append("* `results.json`: Complete raw metrics data and metadata.")
    md.append("* `reconstructions/`: Generated slide reconstructions (if `--save-reconstructions` was passed).")
    md.append("* `REPORT.md`: This summary report.")

    report_content = "\n".join(md)
    report_path = os.path.join(output_dir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
