"""Build the multi-resolution abel.ico and abel.png from the logo master.

The vector original lives outside the repo (``logo_design.ai`` in the manuscript
files). ``abel/ui/assets/abel_logo.png`` is a 2048px transparent render of it and
is the source of truth here; to refresh it after editing the .ai, re-render with
``tools/render_logo_master.py``.

The app icon composites the logo onto a white rounded-square plate: the logo's
arrows are black, so on Windows' dark taskbar and title bars an unplated icon
loses them entirely and reads as a notched blue blob.

Run after replacing the master:

    python tools/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parent.parent / "abel" / "ui" / "assets"
LOGO_PATH = ASSETS / "abel_logo.png"
ICO_PATH = ASSETS / "abel.ico"
PNG_PATH = ASSETS / "abel.png"

SIZES = [16, 24, 32, 48, 64, 128, 256]

MASTER = 1024
PLATE_COLOR = (255, 255, 255, 255)
CORNER_RADIUS = 0.22  # fraction of the plate side
LOGO_SCALE = 0.80  # logo width as a fraction of the plate, leaving a margin
SUPERSAMPLE = 4  # the rounded corners are drawn big and downsampled to smooth them


def _build_master(logo: Image.Image) -> Image.Image:
    """Composite the logo onto a white rounded-square plate at master size."""
    big = MASTER * SUPERSAMPLE
    plate = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(plate).rounded_rectangle(
        [0, 0, big - 1, big - 1],
        radius=int(big * CORNER_RADIUS),
        fill=PLATE_COLOR,
    )
    plate = plate.resize((MASTER, MASTER), Image.LANCZOS)

    side = int(MASTER * LOGO_SCALE)
    offset = (MASTER - side) // 2
    plate.alpha_composite(logo.resize((side, side), Image.LANCZOS), (offset, offset))
    return plate


def main() -> None:
    if not LOGO_PATH.exists():
        raise SystemExit(f"Logo master not found: {LOGO_PATH}")

    logo = Image.open(LOGO_PATH).convert("RGBA")
    master = _build_master(logo)

    # Downsample each size from the master rather than letting Pillow scale
    # internally, so the small sizes get a proper LANCZOS pass.
    images = [master.resize((s, s), Image.LANCZOS) for s in SIZES]

    images[-1].save(ICO_PATH, format="ICO", sizes=[(s, s) for s in SIZES])
    images[-1].save(PNG_PATH, format="PNG")

    print(f"Wrote {ICO_PATH} ({', '.join(str(s) for s in SIZES)} px)")
    print(f"Wrote {PNG_PATH} (256 px)")


if __name__ == "__main__":
    main()
