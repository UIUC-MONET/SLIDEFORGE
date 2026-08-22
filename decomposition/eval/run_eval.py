#!/usr/bin/env python3
"""Top-level eval driver for the sam3 decomposition pipeline.

Example:
    python -m pipeline.eval.run_eval \
        --input /path/to/cost_measure_3slides \
        --metrics reconstruction coverage component_judge holistic_judge \
        --vlm-provider claude-code \
        --output pipeline/eval/results/run_2026-05-14

Metric names (use any subset, or `all`):
    reconstruction      M1 — render-and-compare (MSE/SSIM/PSNR; +LPIPS/CLIP if available)
    coverage            M2 — SCAN-style Coverage + Non-overlap
    component_judge     M3 — per-component VLM judge
    holistic_judge      M4 — holistic VLM judge

VLM providers (only used by component_judge / holistic_judge):
    claude-code         (default) — your Claude Code account via the `claude` CLI
    claude-api          Anthropic SDK; needs ANTHROPIC_API_KEY
    openai              OpenAI SDK; needs OPENAI_API_KEY
    gemini              Google GenAI SDK; needs GEMINI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

# Allow running as both a module and a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from eval_lib import common, reconstruction, coverage_redundancy, component_judge, holistic_judge
from eval_lib.vlm_clients import make_client
from eval_lib.report_generator import generate_report


METRIC_NAMES = ["reconstruction", "coverage", "component_judge", "holistic_judge"]
VLM_METRICS = {"component_judge", "holistic_judge"}


def parse_metrics(arg: list[str]) -> list[str]:
    if not arg or arg == ["all"]:
        return list(METRIC_NAMES)
    out: list[str] = []
    for m in arg:
        if m == "all":
            out.extend(METRIC_NAMES)
        elif m in METRIC_NAMES:
            if m not in out:
                out.append(m)
        else:
            raise SystemExit(
                f"Unknown metric '{m}'. Choose from {METRIC_NAMES} or 'all'."
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input", "-i", required=True,
        help="Root dir holding per-slide output dirs (each containing final/metadata.json).",
    )
    ap.add_argument(
        "--output", "-o", default=None,
        help="Output dir for results. Defaults to ./results/<input_basename>_<ts>/.",
    )
    ap.add_argument(
        "--metrics", "-m", nargs="+", default=["all"],
        help=f"Metrics to run. One of {METRIC_NAMES} or 'all'.",
    )
    ap.add_argument(
        "--vlm-provider", default="claude-code",
        choices=["claude-code", "claude-api", "openai", "gemini"],
        help="VLM provider for component_judge / holistic_judge.",
    )
    ap.add_argument(
        "--vlm-model", default=None,
        help="Override the default model for the chosen VLM provider.",
    )
    ap.add_argument(
        "--save-reconstructions", action="store_true",
        help="Save reconstructed canvases as PNGs alongside the results.",
    )
    ap.add_argument(
        "--workers", type=int, default=3,
        help="Number of parallel thread pool workers (default: 3).",
    )
    ap.add_argument(
        "--max-total-slides", type=int, default=None,
        help="Stop evaluation once total evaluated slides in results.json reaches this count.",
    )
    args = ap.parse_args()

    # Limit PyTorch CPU threads per worker to completely avoid CPU core thrashing/contention
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    metrics = parse_metrics(args.metrics)

    input_root = os.path.abspath(args.input)
    if not os.path.isdir(input_root):
        raise SystemExit(f"--input is not a directory: {input_root}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_dir = os.path.abspath(args.output)
    else:
        base = os.path.basename(os.path.normpath(input_root))
        out_dir = os.path.join(_THIS_DIR, "results", f"{base}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[eval] input  : {input_root}")
    print(f"[eval] output : {out_dir}")
    print(f"[eval] metrics: {metrics}")

    vlm_client = None
    if any(m in VLM_METRICS for m in metrics):
        print(f"[eval] vlm    : {args.vlm_provider} (model={args.vlm_model or 'default'})")
        try:
            vlm_client = make_client(args.vlm_provider, model=args.vlm_model)
        except Exception as e:
            print(f"[eval] WARN: VLM client init failed: {e}", file=sys.stderr)
            print(f"[eval] skipping {VLM_METRICS & set(metrics)}", file=sys.stderr)
            metrics = [m for m in metrics if m not in VLM_METRICS]

    slide_dirs = list(common.iter_slide_dirs(input_root))
    if not slide_dirs:
        raise SystemExit(f"No slide subdirs with final/metadata.json under {input_root}")
    print(f"[eval] slides : {len(slide_dirs)}")

    per_slide: dict[str, dict[str, dict]] = {}
    per_metric_lists: dict[str, list[dict]] = {m: [] for m in metrics}

    results_path = os.path.join(out_dir, "results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                existing = json.load(f)
            per_slide = existing.get("per_slide", {})
            for slide_id, slide_results in per_slide.items():
                for m, r in slide_results.items():
                    if m in per_metric_lists and isinstance(r, dict) and "error" not in r:
                        per_metric_lists[m].append(r)
            print(f"[eval] Loaded {len(per_slide)} existing slide results from {results_path}. Resuming progress.")
        except Exception as e:
            print(f"[eval] WARN: failed to load existing results at startup: {e}")

    # Pre-initialize LPIPS and CLIP models sequentially to prevent thread-safety issues and OOM crashes
    if "reconstruction" in metrics:
        print("[eval] Pre-initializing LPIPS and CLIP models to prevent concurrent loading race conditions...")
        reconstruction.pre_init_models()

    import threading
    from concurrent.futures import ThreadPoolExecutor

    write_lock = threading.Lock()
    print_lock = threading.Lock()
    # Serialize local compute (reconstruction/coverage) to 1 thread to prevent OOM;
    # API calls (component_judge/holistic_judge) run freely in parallel.
    local_compute_sem = threading.Semaphore(3)
    LOCAL_METRICS = {"reconstruction", "coverage"}

    def process_slide(slide_dir: str):
        slide_id = os.path.basename(os.path.normpath(slide_dir))
        with write_lock:
            if slide_id in per_slide:
                return
            if args.max_total_slides is not None and len(per_slide) >= args.max_total_slides:
                return

        lines = [f"\n[eval] === {slide_id} ==="]
        try:
            sample = common.load_slide_sample(slide_dir)
        except Exception as e:
            lines.append(f"[eval] load failed: {e}")
            with write_lock:
                per_slide[slide_id] = {"_load_error": str(e)}
            with print_lock:
                print("\n".join(lines), flush=True)
            return

        slide_results: dict[str, dict] = {}
        recon_save = (
            os.path.join(out_dir, f"{slide_id}_recon.png")
            if args.save_reconstructions
            else None
        )

        for m in metrics:
            t0 = time.time()
            try:
                is_local = m in LOCAL_METRICS
                if is_local:
                    local_compute_sem.acquire()
                try:
                    if m == "reconstruction":
                        r = reconstruction.evaluate(
                            sample,
                            save_reconstruction_to=recon_save,
                        )
                    elif m == "coverage":
                        r = coverage_redundancy.evaluate(sample)
                    elif m == "component_judge":
                        assert vlm_client is not None
                        r = component_judge.evaluate(sample, vlm_client)
                    elif m == "holistic_judge":
                        assert vlm_client is not None
                        r = holistic_judge.evaluate(
                            sample, vlm_client,
                            save_reconstruction_to=recon_save,
                        )
                    else:
                        continue
                finally:
                    if is_local:
                        local_compute_sem.release()
                r["_elapsed_s"] = round(time.time() - t0, 2)
                slide_results[m] = r
                head = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
                lines.append(f"  [{m}] {head}")
            except Exception as e:
                tb = traceback.format_exc(limit=3)
                slide_results[m] = {"error": f"{type(e).__name__}: {e}", "traceback": tb}
                lines.append(f"  [{m}] FAILED: {e}")

        with write_lock:
            per_slide[slide_id] = slide_results
            for m, r in slide_results.items():
                if m in per_metric_lists and isinstance(r, dict) and "error" not in r:
                    per_metric_lists[m].append(r)

            try:
                summary = {
                    "input": input_root,
                    "output": out_dir,
                    "timestamp": ts,
                    "metrics_run": metrics,
                    "vlm_provider": args.vlm_provider if vlm_client else None,
                    "vlm_model": args.vlm_model,
                    "num_slides": len(slide_dirs),
                    "per_slide": per_slide,
                    "aggregates": {
                        m: _aggregate(m, per_metric_lists[m]) for m in metrics
                    },
                }
                with open(results_path, "w") as f:
                    json.dump(summary, f, indent=2)
                generate_report(summary, out_dir)
            except Exception as e:
                lines.append(f"[eval] WARN: failed to save incremental progress for {slide_id}: {e}")

        with print_lock:
            print("\n".join(lines), flush=True)

    # Submit non-evaluated slides to thread pool for concurrent execution
    slides_to_run = [d for d in slide_dirs if os.path.basename(os.path.normpath(d)) not in per_slide]
    print(f"[eval] Running remaining {len(slides_to_run)} slides in parallel using ThreadPoolExecutor (max_workers={args.workers})")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(process_slide, slides_to_run))

    summary = {
        "input": input_root,
        "output": out_dir,
        "timestamp": ts,
        "metrics_run": metrics,
        "vlm_provider": args.vlm_provider if vlm_client else None,
        "vlm_model": args.vlm_model,
        "num_slides": len(slide_dirs),
        "per_slide": per_slide,
        "aggregates": {
            m: _aggregate(m, per_metric_lists[m]) for m in metrics
        },
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] wrote {results_path}")

    try:
        generate_report(summary, out_dir)
        print(f"[eval] wrote {os.path.join(out_dir, 'REPORT.md')}")
    except Exception as e:
        print(f"[eval] WARN: failed to generate REPORT.md: {e}", file=sys.stderr)

    _print_summary(summary)
    return 0


def _aggregate(metric: str, results: list[dict]) -> dict:
    if metric == "reconstruction":
        return reconstruction.aggregate(results)
    if metric == "coverage":
        return coverage_redundancy.aggregate(results)
    if metric == "component_judge":
        return component_judge.aggregate(results)
    if metric == "holistic_judge":
        return holistic_judge.aggregate(results)
    return {}


def _print_summary(summary: dict) -> None:
    print("\n========== AGGREGATE ==========")
    for m, agg in summary["aggregates"].items():
        if not agg:
            print(f"[{m}] (no successful slides)")
            continue
        print(f"[{m}]")
        for k, v in agg.items():
            if isinstance(v, float):
                print(f"  {k:25s} {v:.4f}")
            else:
                print(f"  {k:25s} {v}")


if __name__ == "__main__":
    raise SystemExit(main())
