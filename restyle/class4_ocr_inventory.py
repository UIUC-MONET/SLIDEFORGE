"""Deterministic OCR inventory for class-4 components (H3).

For every class-4 component on a front_bg page, easyocr the INTACT original
page cropped to the component bbox (padded) and record each detected text line
with its slide-coordinate bbox. This is the ground-truth text list the hard
coverage gate enforces — independent of the VLM's (incomplete) inventory.

Runs under the easyocr environment (SLIDECODER_PYTHON):
    $SLIDECODER_PYTHON class4_ocr_inventory.py PAGE_DIR [...]

Writes <page_dir>/class4_ocr_inventory.json:
    {"components_segmented/<name>.png":
        {"lines": [{"text": str, "bbox_xyxy": [x0,y0,x1,y1], "conf": float,
                    "h_px": float}, ...]}}
"""
import json
import sys
from pathlib import Path

CONF_MIN = 0.4
PAD_PX = 12


def measure_page(reader, page_dir: Path) -> None:
    out_path = page_dir / "class4_ocr_inventory.json"
    if out_path.exists():
        print(f"[skip] {out_path} exists")
        return
    import numpy as np
    from PIL import Image
    meta = json.loads((page_dir / "metadata.json").read_text())
    cat_path = page_dir / "categorization.json"
    class4 = set()
    text_path_comps = set()
    if cat_path.exists():
        cat = json.loads(cat_path.read_text())
        class4 = {r["component_file"] for r in cat.get("results", [])
                  if r.get("class") == 4}
        # Components handled by the dedicated text path (class1/2). OCR lines
        # inside THEIR bboxes must not enter a class-4 inventory: overlapping
        # bboxes would make the gate inject a duplicate of text the class1/2
        # textbox already renders (seen: caption injected into an icon).
        text_path_comps = {r["component_file"] for r in cat.get("results", [])
                           if r.get("class") in (1, 2)}
    text_bboxes = [c["bbox_xyxy"] for c in meta.get("components", [])
                   if c.get("component_file") in text_path_comps
                   and c.get("bbox_xyxy")]
    original = Image.open(page_dir / "original.png").convert("RGB")
    out = {}
    for c in meta.get("components", []):
        cf = c.get("component_file")
        if cf not in class4 or not c.get("bbox_xyxy"):
            continue
        x0, y0, x1, y1 = c["bbox_xyxy"]
        cx0 = max(0, int(x0) - PAD_PX)
        cy0 = max(0, int(y0) - PAD_PX)
        cx1 = min(original.width, int(x1) + PAD_PX)
        cy1 = min(original.height, int(y1) + PAD_PX)
        crop = original.crop((cx0, cy0, cx1, cy1))
        lines = []
        for bbox, text, conf in reader.readtext(np.array(crop)):
            if conf < CONF_MIN or not str(text).strip():
                continue
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            lcx = (min(xs) + max(xs)) / 2 + cx0
            lcy = (min(ys) + max(ys)) / 2 + cy0
            if any(tb[0] <= lcx <= tb[2] and tb[1] <= lcy <= tb[3]
                   for tb in text_bboxes):
                continue  # owned by a class1/2 component
            lines.append({
                "text": str(text).strip(),
                "bbox_xyxy": [round(min(xs) + cx0, 1), round(min(ys) + cy0, 1),
                              round(max(xs) + cx0, 1), round(max(ys) + cy0, 1)],
                "conf": round(float(conf), 3),
                "h_px": round(max(ys) - min(ys), 1),
            })
        out[cf] = {"lines": lines}
        print(f"  {cf}: {len(lines)} OCR line(s)")
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[saved] {out_path}")


def main() -> int:
    import easyocr
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    for page_dir in sys.argv[1:]:
        measure_page(reader, Path(page_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
