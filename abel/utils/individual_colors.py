"""Shared per-individual color palette.

Used by the Animal Identity dialog (swatches) and the clip renderer (overlay
dots + legend) so a given animal shows the *same* color everywhere — that's how
a reviewer maps a colored dot in a clip back to a named identity.
"""

from __future__ import annotations

# RGB tuples, indexed by individual position. High-contrast, kept in sync with
# the identity dialog swatches.
PALETTE: list[tuple[int, int, int]] = [
    (255, 80, 80), (80, 160, 255), (80, 220, 120),
    (240, 200, 60), (210, 110, 240), (90, 230, 230),
]


def color_for(idx: int) -> tuple[int, int, int]:
    """RGB color for an individual index (wraps around the palette)."""
    return PALETTE[idx % len(PALETTE)]


def color_for_bgr(idx: int) -> tuple[int, int, int]:
    """BGR color for the same index — for OpenCV drawing."""
    r, g, b = color_for(idx)
    return (b, g, r)
