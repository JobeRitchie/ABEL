"""Bundled UI assets (application icon, etc.)."""

from __future__ import annotations

from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent


def icon_path() -> Path:
    """Return the path to the ABEL application icon (.ico).

    Falls back to the .png if the .ico is missing (e.g. assets not yet built).
    """
    ico = _ASSETS_DIR / "abel.ico"
    if ico.exists():
        return ico
    return _ASSETS_DIR / "abel.png"
