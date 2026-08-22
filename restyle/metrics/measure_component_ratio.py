"""Per-component height-ratio measurement (H1+H2 P1 metric).

Renders each per-component pptx produced by step3_class1/class2, crops the
component's bbox, easyocr-measures text-box heights in both the rendered crop
and the original component crop, and reports rendered/original height ratios.

Run under the easyocr environment (SLIDECODER_PYTHON):
    $SLIDECODER_PYTHON src/measure_component_ratio.py \
        <step3_classN_dir>/<page> <front_bg_page_dir> <out_json>
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "full_pipeline_v2"))
from common import render_pptx_to_png, crop_bbox  # noqa: E402
from measure_text_ink import _boxes, _reader  # noqa: E402


def main() -> int:
    class_page_dir, fb_page_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    meta = json.loads((fb_page_dir / "metadata.json").read_text())
    slide_w = int(meta["query_image_size"]["width"])
    slide_h = int(meta["query_image_size"]["height"])
    bbox_lookup = {Path(c["component_file"]).stem: c["bbox_xyxy"]
                   for c in meta["components"]}
    reader = _reader()
    rows = []
    for pptx in sorted(class_page_dir.glob("*.pptx")):
        stem = pptx.stem
        bbox = bbox_lookup.get(stem)
        orig_png = fb_page_dir / "components_segmented" / f"{stem}.png"
        if bbox is None or not orig_png.exists():
            continue
        rendered = render_pptx_to_png(pptx, slide_w, slide_h)
        crop = pptx.with_name(f"{stem}_measure_crop.png")
        crop_bbox(rendered, bbox, crop)
        orig_h = [h for h, _ in _boxes(reader, orig_png)]
        rend_h = [h for h, _ in _boxes(reader, crop)]
        ratio = (statistics.median(rend_h) / statistics.median(orig_h)
                 if orig_h and rend_h else None)
        sizing = {}
        sj = pptx.with_name(f"{stem}.sizing.json")
        if sj.exists():
            sizing = json.loads(sj.read_text())
        rows.append({
            "stem": stem,
            "ratio": round(ratio, 3) if ratio else None,
            "orig_median_h": round(statistics.median(orig_h), 1) if orig_h else None,
            "rend_median_h": round(statistics.median(rend_h), 1) if rend_h else None,
            "n_orig_boxes": len(orig_h), "n_rend_boxes": len(rend_h),
            "sizing": sizing,
        })
        print(f"{stem}: ratio={rows[-1]['ratio']} "
              f"(orig {rows[-1]['orig_median_h']}px -> rend {rows[-1]['rend_median_h']}px) "
              f"final_pt={sizing.get('final_pt')} vlm_pt={sizing.get('vlm_pt')}")
    ratios = [r["ratio"] for r in rows if r["ratio"]]
    summary = {"median_ratio": round(statistics.median(ratios), 3) if ratios else None,
               "n": len(ratios), "rows": rows}
    print(f"MEDIAN RATIO: {summary['median_ratio']} over {summary['n']} components")
    Path(out_json).write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
