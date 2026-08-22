"""Build a Deck State Graph JSON from a Stage I decomposition run.

Usage:
    python -m dsg.build_dsg --run-dir <decomposition run dir> [--out dsg.json]

    # optionally bind native PPTX objects extracted by
    # decomposition/eval/decompose.py:
    python -m dsg.build_dsg --run-dir runs/my_deck \
        --bind <slide_id>=<slide_components.json> --out dsg.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsg.graph import DeckStateGraph  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="Stage I run dir (decomposition/run_pipeline.py "
                         "--run_dir)")
    ap.add_argument("--out", default="dsg.json", help="output JSON path")
    ap.add_argument("--bind", action="append", default=[],
                    metavar="SLIDE_ID=PPTX_COMPONENTS_JSON",
                    help="bind native PPTX objects to a slide "
                         "(repeatable)")
    args = ap.parse_args()

    g = DeckStateGraph.from_decomposition_run(args.run_dir)
    for spec in args.bind:
        slide_id, _, json_path = spec.partition("=")
        n = g.bind_pptx_objects(slide_id, json_path)
        print(f"bound {n} component<->object edge(s) on {slide_id}")
    out = g.to_json(args.out)
    print(g.summary())
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
