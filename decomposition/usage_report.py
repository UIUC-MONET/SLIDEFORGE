"""Aggregate a VLM usage JSONL (written by backends.log_usage) into per-slide
and per-agent cost tables.

Usage:
    python src/usage_report.py <run_dir_or_jsonl> [--json OUT.json]

The log contains per-call records plus {"marker": "slide_start"/"slide_end"}
rows emitted by run_pipeline.py, which delimit slides.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    if path.is_dir():
        path = path / "vlm_usage.jsonl"
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def aggregate(recs: list[dict]) -> dict:
    slides: dict[str, dict] = {}
    per_agent = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                     "latency_sec": 0.0})
    current = None
    for r in recs:
        if r.get("marker") == "slide_start":
            current = r["image"]
            slides.setdefault(current, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                "vlm_latency_sec": 0.0, "wall_clock_sec": None,
                "agents": defaultdict(int),
            })
            continue
        if r.get("marker") == "slide_end":
            if r["image"] in slides:
                slides[r["image"]]["wall_clock_sec"] = r.get("wall_clock_sec")
            current = None if r.get("image") == current else current
            continue
        if "backend" not in r:
            continue
        agent = r.get("agent", "unknown")
        pa = per_agent[agent]
        pa["calls"] += 1
        pa["input_tokens"] += r.get("input_tokens", 0)
        pa["output_tokens"] += r.get("output_tokens", 0)
        pa["latency_sec"] += r.get("latency_sec", 0.0)
        if current and current in slides:
            s = slides[current]
            s["calls"] += 1
            s["input_tokens"] += r.get("input_tokens", 0)
            s["output_tokens"] += r.get("output_tokens", 0)
            s["cache_creation_input_tokens"] += r.get("cache_creation_input_tokens", 0)
            s["cache_read_input_tokens"] += r.get("cache_read_input_tokens", 0)
            s["vlm_latency_sec"] += r.get("latency_sec", 0.0)
            s["agents"][agent] += 1

    done = [s for s in slides.values() if s["wall_clock_sec"] is not None]
    n = len(done)
    totals = {
        "slides_completed": n,
        "slides_started": len(slides),
        "mean_calls_per_slide": round(sum(s["calls"] for s in done) / n, 2) if n else None,
        "mean_input_tokens_per_slide": round(sum(s["input_tokens"] for s in done) / n, 1) if n else None,
        "mean_output_tokens_per_slide": round(sum(s["output_tokens"] for s in done) / n, 1) if n else None,
        "mean_wall_clock_sec_per_slide": round(sum(s["wall_clock_sec"] for s in done) / n, 1) if n else None,
        "mean_vlm_latency_sec_per_slide": round(sum(s["vlm_latency_sec"] for s in done) / n, 1) if n else None,
    }
    for s in slides.values():
        s["agents"] = dict(s["agents"])
    return {
        "totals": totals,
        "per_agent": {k: dict(v) for k, v in sorted(per_agent.items())},
        "per_slide": slides,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="run dir containing vlm_usage.jsonl, or the jsonl itself")
    ap.add_argument("--json", dest="out_json", default=None)
    args = ap.parse_args()
    report = aggregate(load_records(Path(args.path)))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
    t = report["totals"]
    print(json.dumps(t, indent=2))
    print(f"{'agent':<40}{'calls':>8}{'in_tok':>12}{'out_tok':>10}{'lat_s':>10}")
    for agent, v in report["per_agent"].items():
        print(f"{agent:<40}{v['calls']:>8}{v['input_tokens']:>12}"
              f"{v['output_tokens']:>10}{round(v['latency_sec'], 1):>10}")


if __name__ == "__main__":
    main()
