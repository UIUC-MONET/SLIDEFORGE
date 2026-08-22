"""Step 0 — ingest the Deck State Graph produced by Stage I.

Bridges SlideForge's two stages: it flattens the component nodes and
cleaned-background state of the Deck State Graph (the ``final/`` state
written by ``decomposition/run_pipeline.py``) into the per-page
``front_bg/<page>/`` workspace that theme-preserving reconstruction
(step1..step5b) consumes.

Two input modes:

1. ``--run-dir <decomposition run dir>`` (recommended): the directory you
   passed as ``--run_dir`` to ``decomposition/run_pipeline.py``. Its
   ``<NNNN>_<name>/`` slide dirs are ingested directly; each slide's
   original render is resolved from ``final/metadata.json`` (falling back
   to ``--png-dir`` lookup by the ``slide_NNN`` suffix if the recorded
   path no longer exists).

2. ``--deck <dataset>/<deck> --png-root ... --decomp-root ...`` (legacy
   benchmark layout):
     png_root/<dataset>/<deck>/slide_<NNN>.png
     decomp_root/<dataset>/<deck>/<NNNN>_<deck>_slide_<NNN>/

Per slide dir the following DSG state is read:
      iter_00/segmentation/<inner>/image_cleaned.png      # cleaned background
      iter_00/segmentation/<inner>/metadata.json          # query_image_size
      final/metadata.json                                 # canonical components
      final/components/<filename>.png                     # mask-cropped fg
      final/bbox_components/<filename>.png                # bbox-cropped fg

The cleaned background comes from ``iter_00`` and the foreground
components from ``final/`` (the consolidated component set, including
layout-review "missed" components like logos that per-iteration
segmentation doesn't surface).

Output layout (per case):
  <out_dir>/front_bg/<NNNN>/
      metadata.json
      original.png
      cleaned_iter00.png
      components_segmented/<filename>.png
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


SLIDE_RE = re.compile(r"(slide_\d+)$")
PAGE_DIR_RE = re.compile(r"^(?P<idx>\d+)_(?P<rest>.+?_slide_\d+)$")
# --run-dir mode: any <NNNN>_<name> slide dir written by the Stage I pipeline.
RUN_PAGE_DIR_RE = re.compile(r"^(?P<idx>\d+)_(?P<rest>.+)$")


def _resolve_original(page_dir: Path, png_dir: Path | None) -> Path | None:
    """Resolve the original slide render for a Stage I slide dir.

    Prefers the absolute path recorded in ``final/metadata.json`` at
    decomposition time; falls back to matching the ``slide_NNN`` suffix
    (or the full dir stem) inside ``png_dir``."""
    meta_path = page_dir / "final" / "metadata.json"
    if meta_path.exists():
        try:
            recorded = json.loads(meta_path.read_text(encoding="utf-8")).get("image")
            if recorded and Path(recorded).exists():
                return Path(recorded)
        except Exception:
            pass
    if png_dir is not None:
        slide_m = SLIDE_RE.search(page_dir.name)
        candidates = []
        if slide_m:
            candidates.append(png_dir / f"{slide_m.group(1)}.png")
        m = RUN_PAGE_DIR_RE.match(page_dir.name)
        if m:
            candidates.append(png_dir / f"{m.group('rest')}.png")
        candidates.append(png_dir / f"{page_dir.name}.png")
        for c in candidates:
            if c.exists():
                return c
    return None


def _list_iter_dirs(page_dir: Path) -> list[Path]:
    iters = []
    for d in page_dir.iterdir():
        if d.is_dir() and re.fullmatch(r"iter_\d+", d.name):
            iters.append(d)
    iters.sort(key=lambda d: int(d.name.split("_")[1]))
    return iters


def _segmentation_inner(iter_dir: Path) -> Path | None:
    seg = iter_dir / "segmentation"
    if not seg.is_dir():
        return None
    candidates = [p for p in seg.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(p.name), reverse=True)
    return candidates[0]


def prepare_one_page(page_dir: Path, out_root: Path,
                     png_dir: Path | None) -> Path | None:
    m = RUN_PAGE_DIR_RE.match(page_dir.name)
    if not m:
        print(f"  [skip] {page_dir.name}: does not match NNNN_<name>")
        return None
    page_idx = m.group("idx")

    orig_render = _resolve_original(page_dir, png_dir)
    if orig_render is None:
        print(f"  [skip] {page_dir.name}: cannot resolve original PNG "
              f"(final/metadata.json 'image' missing on disk and no --png-dir match)")
        return None

    final_dir = page_dir / "final"
    final_meta_path = final_dir / "metadata.json"
    if not final_meta_path.exists():
        print(f"  [skip] {page_dir.name}: no final/metadata.json")
        return None
    final_meta = json.loads(final_meta_path.read_text(encoding="utf-8"))

    iter_dirs = _list_iter_dirs(page_dir)
    if not iter_dirs:
        print(f"  [skip] {page_dir.name}: no iter_* directories")
        return None
    iter0 = iter_dirs[0]
    seg_inner = _segmentation_inner(iter0)
    if seg_inner is None:
        print(f"  [skip] {page_dir.name}: no iter_00 segmentation inner dir")
        return None
    cleaned_src = seg_inner / "image_cleaned.png"
    if not cleaned_src.exists():
        print(f"  [skip] {page_dir.name}: missing {cleaned_src}")
        return None
    iter0_meta_path = seg_inner / "metadata.json"
    query_image_size = None
    if iter0_meta_path.exists():
        try:
            iter0_meta = json.loads(iter0_meta_path.read_text(encoding="utf-8"))
            query_image_size = iter0_meta.get("query_image_size")
        except Exception as e:
            print(f"  [warn] {page_dir.name}: cannot read iter_00 metadata: {e}")

    if query_image_size is None:
        from PIL import Image
        with Image.open(orig_render) as im:
            query_image_size = {"width": im.width, "height": im.height}

    out_dir = out_root / page_idx
    components_out = out_dir / "components_segmented"
    components_out.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(orig_render, out_dir / "original.png")
    shutil.copyfile(cleaned_src, out_dir / "cleaned_iter00.png")

    merged_components: list[dict] = []
    final_components_dir = final_dir / "components"
    final_bbox_dir = final_dir / "bbox_components"

    raw_components = final_meta.get("final_components") or final_meta.get("components") or []
    for comp in raw_components:
        fname = comp.get("filename")
        if not fname:
            continue
        src_png = final_components_dir / fname
        if not src_png.exists():
            alt = final_bbox_dir / fname
            if alt.exists():
                src_png = alt
        if not src_png.exists():
            print(f"    [warn] missing component image for {fname}")
            continue
        dst_png = components_out / fname
        shutil.copyfile(src_png, dst_png)

        new_comp = dict(comp)
        new_comp["component_file"] = f"components_segmented/{fname}"
        new_comp["bbox_component_file"] = f"components_segmented/{fname}"
        new_comp["source_iter"] = "final"
        # Older code in the pipeline reads tight_bbox_xyxy if present;
        # final/ entries don't carry it but downstream only consults
        # bbox_xyxy, so this is safe.
        merged_components.append(new_comp)

    if not merged_components:
        print(f"  [skip] {page_dir.name}: 0 usable components in final/")
        return out_dir

    out_meta = {
        "page": page_idx,
        "source_page_dir": str(page_dir),
        "query_image_path": str(out_dir / "original.png"),
        "query_image_size": query_image_size,
        "num_components": len(merged_components),
        "components": merged_components,
        "source_iters": [d.name for d in iter_dirs],
        "components_source": "final",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(out_meta, indent=2, ensure_ascii=False)
    )
    print(f"  [{page_idx}] {len(merged_components)} component(s) from final/")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="Stage I decomposition run dir (the --run_dir you "
                         "passed to decomposition/run_pipeline.py). "
                         "Recommended mode.")
    ap.add_argument("--png-dir", default=None,
                    help="Optional dir of original slide PNGs, used as a "
                         "fallback when the path recorded in the run's "
                         "final/metadata.json no longer exists.")
    ap.add_argument("--deck", default=None,
                    help="[legacy layout] <dataset>/<deck> relative path, "
                         "e.g. SIGCOMM2013_hotsdn/06")
    ap.add_argument("--png-root", default=None,
                    help="[legacy layout] root containing <dataset>/<deck>/slide_NNN.png")
    ap.add_argument("--decomp-root", default=None,
                    help="[legacy layout] root containing "
                         "<dataset>/<deck>/<NNNN>_<deck>_slide_NNN/")
    ap.add_argument("--out-dir", required=True,
                    help="per-case output root; front_bg/ written under it")
    args = ap.parse_args()

    if args.run_dir:
        decomp_dir = Path(args.run_dir)
        png_dir = Path(args.png_dir) if args.png_dir else None
        page_re = RUN_PAGE_DIR_RE
    elif args.deck and args.png_root and args.decomp_root:
        decomp_dir = Path(args.decomp_root) / args.deck
        png_dir = Path(args.png_root) / args.deck
        page_re = PAGE_DIR_RE
        if not png_dir.is_dir():
            print(f"png deck dir missing: {png_dir}", file=sys.stderr)
            return 1
    else:
        print("pass either --run-dir, or all of --deck/--png-root/--decomp-root",
              file=sys.stderr)
        return 1

    if not decomp_dir.is_dir():
        print(f"decomposition dir missing: {decomp_dir}", file=sys.stderr)
        return 1

    out_root = Path(args.out_dir) / "front_bg"
    out_root.mkdir(parents=True, exist_ok=True)
    page_dirs = sorted(
        p for p in decomp_dir.iterdir()
        if p.is_dir() and page_re.match(p.name) and (p / "final").is_dir()
    )
    print(f"Found {len(page_dirs)} page(s) under {decomp_dir}")
    n_ok = 0
    for pd in page_dirs:
        if prepare_one_page(pd, out_root, png_dir):
            n_ok += 1
    print(f"\nPrepared {n_ok}/{len(page_dirs)} page(s) under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
