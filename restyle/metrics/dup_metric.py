"""M5 duplication metric (R3): count words the RENDER shows more often than
the ORIGINAL — the signature of double-rendered text (raster copy + rebuilt
textbox, or gate/self-repair double injection).

Per page:  M5 = sum over words of max(0, render_count - orig_count),
counting only words with orig_count >= 1 (pure hallucinations are rare and
OCR-noisy; words absent from the original are ignored here).
Fuzzy guard: a render word is matched to an original word when exact equal
or difflib ratio >= 0.85 (absorbs OCR misreads before counting).

Usage: dup_metric.py <case_dir> <render_dir> <out_json>
  <case_dir> needs front_bg/<page>/original.png; render_dir page-N.png
"""
import sys, re, json, difflib
from collections import Counter
from pathlib import Path
import easyocr

reader = easyocr.Reader(['en'], gpu=True, verbose=False)


def words_of(img_path):
    out = []
    for _bbox, text, conf in reader.readtext(str(img_path)):
        if conf < 0.4:
            continue
        for w in re.findall(r"[A-Za-z0-9]+", text.lower()):
            if len(w) >= 2:
                out.append(w)
    return Counter(out)


def fuzzy_fold(render_counts, orig_counts):
    """Map render words onto original vocabulary (exact or ratio>=0.85)."""
    folded = Counter()
    vocab = list(orig_counts)
    for w, n in render_counts.items():
        if w in orig_counts:
            folded[w] += n
            continue
        best = difflib.get_close_matches(w, vocab, n=1, cutoff=0.85)
        if best:
            folded[best[0]] += n
    return folded


def main():
    case_dir, render_dir, out_json = map(Path, sys.argv[1:4])
    report = []
    pages = sorted((case_dir / "front_bg").iterdir())
    for i, p in enumerate(pages):
        orig = p / "original.png"
        rend = render_dir / f"page-{i+1}.png"
        if not orig.exists() or not rend.exists():
            continue
        oc = words_of(orig)
        rc = fuzzy_fold(words_of(rend), oc)
        dups = {w: rc[w] - oc[w] for w in rc if rc[w] > oc[w]}
        m5 = sum(dups.values())
        report.append({"page": p.name, "M5": m5, "dup_words": dups})
        print(f"  {p.name}: M5={m5} dups={dict(sorted(dups.items(), key=lambda x:-x[1])[:8])}")
    total = sum(r["M5"] for r in report)
    print(f"case M5 total = {total}")
    json.dump({"M5_total": total, "pages": report}, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
