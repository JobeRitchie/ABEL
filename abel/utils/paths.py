"""Path helpers for global app and project locations."""

from __future__ import annotations

from pathlib import Path

from abel.core.constants import GLOBAL_CONFIG_DIR, GLOBAL_LOG_DIR


def ensure_global_dirs() -> None:
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_LOG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
