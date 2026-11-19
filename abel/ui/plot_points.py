"""Individual data points overlaid on bar charts, plus the page toggle for them.

Every Analytics page that draws mean bars (Graphs, Session Sections,
Velocity, Motifs, Social) shows a "Show data points" checkbox built by
:func:`make_points_toggle`. The bars then call :func:`overlay_points` with the
values each bar averages, so a reader sees n and spread, not only the mean.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

import numpy as np
from PySide6.QtWidgets import QCheckBox, QWidget

POINTS_TOGGLE_LABEL = "Show data points"
POINTS_TOGGLE_TIP = (
    "Overlay each value behind a bar as a dot (one per session or subject),\n"
    "so the spread and n are visible, not only the mean."
)


def overlay_points(
    ax: Any,
    center: float,
    vals: Any,
    *,
    spread: float = 0.12,
    size: float = 22.0,
    horizontal: bool = False,
    facecolor: str = "white",
    edgecolor: str = "black",
    linewidth: float = 0.6,
    zorder: float = 5,
) -> None:
    """Scatter *vals* as jittered dots on the bar centered at *center*.

    Non-finite values are dropped. Jitter is seeded from the bar position so a
    redraw puts each dot in the same place. With ``horizontal`` the bar runs
    along x (``barh``) and the jitter goes on y.
    """
    arr = np.asarray(vals, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    seed = int(round(abs(float(center)) * 1000)) % (2**32)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-spread, spread, arr.size) if arr.size > 1 else np.zeros(1)
    pos = float(center) + jitter
    xs, ys = (arr, pos) if horizontal else (pos, arr)
    ax.scatter(
        xs, ys, s=size, facecolor=facecolor, edgecolors=edgecolor,
        linewidths=linewidth, zorder=zorder,
    )


def make_points_toggle(
    settings: MutableMapping[str, Any],
    key: str,
    on_change: Callable[[], Any],
    *,
    default: bool = True,
    parent: QWidget | None = None,
) -> QCheckBox:
    """Checkbox bound to ``settings[key]`` that redraws on change."""
    chk = QCheckBox(POINTS_TOGGLE_LABEL, parent)
    chk.setToolTip(POINTS_TOGGLE_TIP)
    settings.setdefault(key, default)
    chk.setChecked(bool(settings[key]))

    def _toggled(on: bool) -> None:
        settings[key] = bool(on)
        on_change()

    chk.toggled.connect(_toggled)
    return chk
