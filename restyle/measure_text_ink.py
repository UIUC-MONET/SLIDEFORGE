"""Measure per-component text-line heights with easyocr (H2 ink prior).

Runs under the easyocr environment (see SLIDECODER_PYTHON in common.py):
    $SLIDECODER_PYTHON measure_text_ink.py PAGE_DIR [...]

For each front_bg page dir, writes ``<page_dir>/text_metrics.json``:
    {"components_segmented/<name>.png":
        {"median_box_h_px": float, "n_boxes": int, "boxes": [[h, conf], ...]}}
Skips pages whose text_metrics.json already exists.

--calibrate: render Carlito text lines at known em sizes, OCR them, and print
the detection-box-height / em ratio (source of common.EASYOCR_BOX_PER_EM).
"""
import json
import statistics
import sys
from pathlib import Path

CONF_MIN = 0.4


def _reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=True, verbose=False)


def _flatten(img_path: Path, bg: int):
    """RGBA component crop composited over a solid background."""
    from PIL import Image
    img = Image.open(img_path).convert("RGBA")
    base = Image.new("RGB", img.size, (bg, bg, bg))
    base.paste(img, mask=img.split()[3])
    return base


def _boxes(reader, img_path: Path, conf_min: float = CONF_MIN):
    import numpy as np
    best = []
    for bg in (255, 0):
        res = reader.readtext(np.array(_flatten(img_path, bg)))
        boxes = [
            (float(max(p[1] for p in bbox) - min(p[1] for p in bbox)), float(conf))
            for bbox, _text, conf in res if conf >= conf_min
        ]
        if len(boxes) > len(best):
            best = boxes
    return best


def _boxes_with_fallback(reader, img_path: Path):
    """Confident boxes when available; otherwise low-confidence *detection*
    boxes (recognition may be garbage on mask-mangled crops, but the CRAFT
    detector's line heights are still roughly right). Returns (boxes, low_conf)."""
    boxes = _boxes(reader, img_path)
    if boxes:
        return boxes, False
    boxes = _boxes(reader, img_path, conf_min=0.0)
    return boxes, True


def measure_page(reader, page_dir: Path) -> None:
    """Priors are measured on the INTACT original page, one OCR pass per page,
    with word boxes assigned to components by bbox containment. This keeps the
    measurement frame identical to the M1 metric (full-page easyocr): tight
    per-component crops clip the detector's boxes ~20% and segmentation masks
    mangle glyphs, both of which systematically undershoot the prior.
    Components that catch no full-page boxes fall back to crop OCR."""
    out_path = page_dir / "text_metrics.json"
    if out_path.exists():
        print(f"[skip] {out_path} exists")
        return
    meta = json.loads((page_dir / "metadata.json").read_text())
    bboxes = {c["component_file"]: c.get("bbox_xyxy")
              for c in meta.get("components", [])}
    original = page_dir / "original.png"
    page_boxes = []
    if original.exists():
        for bbox, _text, conf in reader.readtext(str(original)):
            if conf < CONF_MIN:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            page_boxes.append((min(xs), min(ys), max(xs), max(ys), float(conf)))

    comp_dir = page_dir / "components_segmented"
    metrics = {}
    for png in sorted(comp_dir.glob("*.png")):
        key = f"components_segmented/{png.name}"
        cb = bboxes.get(key)
        heights: list[float] = []
        boxes_y_h: list[list[float]] = []
        source = "page"
        if cb:
            x0, y0, x1, y1 = cb
            # Full containment (center-only tests leaked neighboring
            # components' lines into overlapping bboxes), with tolerance
            # PROPORTIONAL to box height: detector boxes dilate with text
            # size, so a fixed ±12px excluded large titles' own boxes and
            # pushed them to the crop fallback (~20% under-measure:
            # paper_9_2 p0005 title 120px -> 96px). Leaked neighbor boxes
            # sit far outside and are still rejected.
            for bx0, by0, bx1, by1, _conf in page_boxes:
                tol = max(12.0, 0.25 * float(by1 - by0))
                if (bx0 >= x0 - tol and by0 >= y0 - tol
                        and bx1 <= x1 + tol and by1 <= y1 + tol):
                    heights.append(float(by1 - by0))
                    boxes_y_h.append([round(float(by0), 1),
                                      round(float(by1 - by0), 1)])
        low_conf = False
        if not heights:
            boxes, low_conf = _boxes_with_fallback(reader, png)
            heights = [h for h, _ in boxes]
            source = "crop"
        boxes_y_h.sort()
        metrics[key] = {
            "median_box_h_px": round(statistics.median(heights), 1) if heights else None,
            "n_boxes": len(heights),
            "boxes_y_h": boxes_y_h,   # y-sorted [top, height] (page source only)
            "source": source,
            "low_conf": low_conf,
        }
    out_path.write_text(json.dumps(metrics, indent=1))
    print(f"[saved] {out_path} ({len(metrics)} components)")


def calibrate() -> None:
    """Empirical easyocr-box-height / Carlito-em ratio."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    reader = _reader()
    samples = []
    texts = ["Key observation", "Watching a traditional video",
             "Tenant isolation SNAT request", "median height 42"]
    for em in (30, 45, 60, 90, 130):
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf", em)
        for text in texts:
            img = Image.new("RGB", (int(font.getlength(text)) + 60, em * 3),
                            (255, 255, 255))
            ImageDraw.Draw(img).text((30, em), text, font=font, fill=(0, 0, 0))
            for bbox, _t, conf in reader.readtext(np.array(img)):
                if conf < CONF_MIN:
                    continue
                h = max(p[1] for p in bbox) - min(p[1] for p in bbox)
                samples.append(h / em)
    print(f"n={len(samples)} median box/em = {statistics.median(samples):.3f} "
          f"(p25={statistics.quantiles(samples, n=4)[0]:.3f}, "
          f"p75={statistics.quantiles(samples, n=4)[2]:.3f})")


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--calibrate":
        calibrate()
        return 0
    if not args:
        print(__doc__)
        return 1
    reader = _reader()
    for page_dir in args:
        measure_page(reader, Path(page_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
