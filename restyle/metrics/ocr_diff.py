"""OCR 差分:对比 original.png 与 final.pptx 渲染页的文字覆盖率 + 字高比."""
import sys, json, re
from pathlib import Path
import easyocr

reader = easyocr.Reader(['en'], gpu=True, verbose=False)

def norm_words(results):
    words = []
    for bbox, text, conf in results:
        if conf < 0.4: continue
        h = max(p[1] for p in bbox) - min(p[1] for p in bbox)
        for w in re.findall(r"[A-Za-z0-9]+", text.lower()):
            if len(w) >= 2: words.append((w, h))
    return words

def run_case(case_dir, render_dir, out_json):
    case_dir, render_dir = Path(case_dir), Path(render_dir)
    pages = sorted((case_dir/"front_bg").iterdir())
    report = []
    for i, p in enumerate(pages):
        orig = p/"original.png"
        rend = render_dir/f"page-{i+1}.png"
        if not orig.exists() or not rend.exists(): continue
        ow = norm_words(reader.readtext(str(orig)))
        rw = norm_words(reader.readtext(str(rend)))
        oset = {}
        for w,h in ow: oset.setdefault(w, []).append(h)
        rset = {}
        for w,h in rw: rset.setdefault(w, []).append(h)
        missing = sorted(set(oset) - set(rset))
        # 字高比:双方都有的词, median(render高)/median(orig高)
        import statistics
        ratios = []
        for w in set(oset) & set(rset):
            ratios.append(statistics.median(rset[w]) / statistics.median(oset[w]))
        report.append({
            "page": p.name,
            "orig_words": len(oset), "render_words": len(rset),
            "missing_words": missing,
            "coverage": 1 - len(missing)/max(len(oset),1),
            "median_height_ratio": round(statistics.median(ratios),3) if ratios else None,
        })
        print(f"  {p.name}: orig={len(oset)} render={len(rset)} coverage={report[-1]['coverage']:.0%} height_ratio={report[-1]['median_height_ratio']} missing={missing[:12]}")
    json.dump(report, open(out_json,"w"), indent=1)

if __name__ == "__main__":
    run_case(sys.argv[1], sys.argv[2], sys.argv[3])
