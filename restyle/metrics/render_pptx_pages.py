"""Render every page of a pptx to render_dir/page-N.png at source resolution.

soffice -> pdf -> pdftoppm -r 96 (canvas is built at source pixels @96dpi, so
-r 96 reproduces the original pixel size), matching the bench18 baseline
renders and src/ocr_diff.py's page-N.png convention.

usage: python3 render_pptx_pages.py final.pptx render_dir
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    pptx, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    with tempfile.TemporaryDirectory(prefix="lo_prof_") as prof, \
         tempfile.TemporaryDirectory(prefix="pptx_pdf_") as work:
        subprocess.run(
            ["soffice", "--headless", f"-env:UserInstallation=file://{prof}",
             "--convert-to", "pdf", "--outdir", work, str(pptx)],
            check=True, timeout=600, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        pdf = next(Path(work).glob("*.pdf"))
        subprocess.run(
            ["pdftoppm", "-png", "-r", "96", str(pdf), str(out_dir / "page")],
            check=True, timeout=600, env=env,
        )
    # pdftoppm names page-1.png / page-01.png depending on page count; ocr_diff
    # expects page-N.png without zero padding.
    for p in sorted(out_dir.glob("page-*.png")):
        n = p.stem.split("-")[1]
        if n.startswith("0"):
            p.rename(out_dir / f"page-{int(n)}.png")
    print(f"rendered {len(list(out_dir.glob('page-*.png')))} pages -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
