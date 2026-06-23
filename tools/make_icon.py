"""Rasterize abel_icon.svg into a multi-resolution abel.ico and abel.png.

Uses PySide6 (QtSvg) to render the vector source at several sizes and Pillow to
pack them into a Windows .ico, plus a standalone 256px .png for docs/non-Windows.

Run after editing the SVG:

    python tools/make_icon.py
"""

from __future__ import annotations

import io
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "abel" / "ui" / "assets"
SVG_PATH = ASSETS / "abel_icon.svg"
ICO_PATH = ASSETS / "abel.ico"
PNG_PATH = ASSETS / "abel.png"

SIZES = [16, 24, 32, 48, 64, 128, 256]


def _render(renderer: QSvgRenderer, size: int) -> Image.Image:
    """Render the SVG to a transparent PNG-backed PIL image at the given size."""
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    renderer.render(painter)
    painter.end()

    # Convert QImage -> PNG bytes -> PIL (avoids manual buffer/stride handling).
    from PySide6.QtCore import QBuffer, QByteArray

    buffer = io.BytesIO()

    qbytes = QByteArray()
    qbuffer = QBuffer(qbytes)
    qbuffer.open(QBuffer.OpenModeFlag.WriteOnly)
    image.save(qbuffer, "PNG")
    qbuffer.close()
    buffer.write(qbytes.data())
    buffer.seek(0)
    return Image.open(buffer).convert("RGBA")


def main() -> None:
    if not SVG_PATH.exists():
        raise SystemExit(f"SVG source not found: {SVG_PATH}")

    # A QApplication is required for QPainter/QImage rendering.
    app = QApplication.instance() or QApplication([])
    renderer = QSvgRenderer(str(SVG_PATH))
    if not renderer.isValid():
        raise SystemExit(f"Could not parse SVG: {SVG_PATH}")

    images = [_render(renderer, size) for size in SIZES]

    # Pillow packs all sizes into a single .ico.
    largest = images[-1]
    largest.save(ICO_PATH, format="ICO", sizes=[(s, s) for s in SIZES])
    largest.save(PNG_PATH, format="PNG")

    print(f"Wrote {ICO_PATH} ({', '.join(str(s) for s in SIZES)} px)")
    print(f"Wrote {PNG_PATH} (256 px)")

    del app  # keep linter happy; app lifetime spans rendering above


if __name__ == "__main__":
    main()
