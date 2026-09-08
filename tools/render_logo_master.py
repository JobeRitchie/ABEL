"""Re-render abel_logo.png from the Illustrator original.

The .ai file is PDF-backed, so PyMuPDF rasterizes it directly — including the
soft-masked gradient sphere, which a plain SVG export drops. Needs PyMuPDF,
which is not a runtime dependency:

    pip install pymupdf
    python tools/render_logo_master.py "path/to/logo_design.ai"
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "abel" / "ui" / "assets"
LOGO_PATH = ASSETS / "abel_logo.png"

DEFAULT_SOURCE = Path(
    r"J:\Kash Lab\Jobe\ABEL Manuscript Files\logo_design.ai"
)

RENDER_DPI = 600
MASTER_SIZE = 2048
MARGIN = 0.03  # padding around the artwork when squaring the canvas


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not source.exists():
        raise SystemExit(f"Logo source not found: {source}")

    import pymupdf

    doc = pymupdf.open(source)
    pixmap = doc[0].get_pixmap(dpi=RENDER_DPI, alpha=True)
    art = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGBA")

    # Trim the page down to the artwork, then centre it on a square canvas so
    # the icon sizes stay aspect-correct.
    art = art.crop(art.getchannel("A").getbbox())
    canvas_side = int(round(max(art.size) / (1.0 - MARGIN)))
    canvas = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
    canvas.paste(art, ((canvas_side - art.width) // 2, (canvas_side - art.height) // 2))

    canvas.resize((MASTER_SIZE, MASTER_SIZE), Image.LANCZOS).save(LOGO_PATH)
    print(f"Wrote {LOGO_PATH} ({MASTER_SIZE} px, from {source})")
    print("Now run tools/make_icon.py to rebuild abel.ico / abel.png.")


if __name__ == "__main__":
    main()
