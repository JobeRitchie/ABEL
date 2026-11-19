"""Every Analytics page with mean bars has a "Show data points" toggle.

The toggle writes a per-page setting and redraws; the bar charts then overlay
one dot per value the bar averages. Pinned here: the shared helpers, that each
page carries its own toggle, and that real renders add or drop the dots.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.plot_points import (  # noqa: E402
    POINTS_TOGGLE_LABEL,
    make_points_toggle,
    overlay_points,
)
from abel.ui.tabs.behavior_analytics_tab import (  # noqa: E402
    BehaviorAnalyticsTab,
    _GraphsWidget,
    _VelocityWidget,
)


@pytest.fixture(scope="module")
def _app():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")


def _dots(ax) -> int:
    return sum(len(c.get_offsets()) for c in ax.collections if isinstance(c, PathCollection))


# ── helpers ──────────────────────────────────────────────────────────────


def test_overlay_drops_nonfinite_and_is_repeatable():
    fig = Figure()
    ax = fig.add_subplot(111)
    overlay_points(ax, 2, [1.0, np.nan, 3.0, np.inf, 5.0])
    overlay_points(ax, 2, [1.0, 3.0, 5.0])
    a, b = (c.get_offsets() for c in ax.collections)
    assert len(a) == 3
    np.testing.assert_allclose(a, b)  # same bar -> same jitter on redraw
    np.testing.assert_allclose(np.sort(a[:, 1]), [1.0, 3.0, 5.0])
    assert np.all(np.abs(a[:, 0] - 2) <= 0.12)


def test_overlay_horizontal_puts_values_on_x():
    fig = Figure()
    ax = fig.add_subplot(111)
    overlay_points(ax, 1, [4.0, 6.0], horizontal=True)
    off = ax.collections[0].get_offsets()
    np.testing.assert_allclose(np.sort(off[:, 0]), [4.0, 6.0])
    assert np.all(np.abs(off[:, 1] - 1) <= 0.12)


def test_overlay_empty_draws_nothing():
    fig = Figure()
    ax = fig.add_subplot(111)
    overlay_points(ax, 0, [np.nan])
    assert not ax.collections


def test_toggle_writes_setting_and_redraws(_app):
    settings: dict = {}
    calls: list[int] = []
    chk = make_points_toggle(settings, "k", lambda: calls.append(1), default=False)
    assert settings["k"] is False and not chk.isChecked()
    chk.setChecked(True)
    assert settings["k"] is True and calls == [1]


# ── one toggle per page ──────────────────────────────────────────────────


def test_every_bar_page_has_its_own_toggle(_app):
    tab = BehaviorAnalyticsTab()
    pages = {
        "graphs": (tab._graphs_tab, tab._graph_settings, "show_indiv_points"),
        "sections": (tab._sections_tab, tab._graph_settings, "sections_show_points"),
        "velocity": (tab._velocity_tab, tab._graph_settings, "velocity_show_points"),
        "motifs": (
            tab._relationships_tab,
            tab._relationships_tab._motif_graph_settings,
            "show_indiv_points",
        ),
        "social": (tab._social_tab, tab._social_tab._plot_opts, "show_points"),
    }
    for name, (page, settings, key) in pages.items():
        chk = page._points_chk
        assert chk.text() == POINTS_TOGGLE_LABEL, name
        assert chk.isChecked() == bool(settings[key]), name
        chk.setChecked(not chk.isChecked())  # redraw on empty data must not raise
        assert settings[key] == chk.isChecked(), name
    # Pages are independent: flipping Velocity left Sections alone.
    tab._velocity_tab._points_chk.setChecked(False)
    tab._sections_tab._points_chk.setChecked(True)
    assert tab._graph_settings["velocity_show_points"] is False
    assert tab._graph_settings["sections_show_points"] is True


# ── real renders ─────────────────────────────────────────────────────────


def _graphs_stub(show: bool):
    stub = SimpleNamespace()
    for name in ("_bar_groups_on_ax", "_session_aggregate", "_export_sessions",
                 "_missing_value_for_metric"):
        setattr(stub, name, getattr(_GraphsWidget, name).__get__(stub))
    gs = {"show_indiv_points": show, "show_stats": False, "error_style": "SEM",
          "tick_fontsize": 8, "axis_fontsize": 10, "title_fontsize": 12}
    stub._gs = lambda: gs
    stub._host = SimpleNamespace(
        _session_groups={"a1": "A", "a2": "A", "b1": "B", "b2": "B", "b3": "B"},
        _ordered_group_list=lambda names, *a: sorted(names),
        _group_color=lambda g, i: "#888888",
        _summary_tab=SimpleNamespace(_checked_subjects=lambda: {"a1", "a2", "b1", "b2", "b3"}),
        ordered_session_labels=lambda: ["a1", "a2", "b1", "b2", "b3"],
    )
    stub._export_sessions = lambda: ["a1", "a2", "b1", "b2", "b3"]
    stub._apply = lambda *a, **k: None
    return stub


@pytest.mark.parametrize("show, expected", [(True, 5), (False, 0)])
def test_graphs_group_bars_follow_toggle(show, expected):
    df = pd.DataFrame({
        "session_label": ["a1", "a2", "b1", "b2", "b3"],
        "behavior": ["Rear"] * 5,
        "bout_count": [3.0, 4.0, 7.0, 8.0, 9.0],
    })
    ax = Figure().add_subplot(111)
    _graphs_stub(show)._bar_groups_on_ax(ax, df, "bout_count", "Bouts", "sum")
    assert _dots(ax) == expected


@pytest.mark.parametrize("show, expected", [(True, 6), (False, 0)])
def test_velocity_first_last_bars_follow_toggle(show, expected):
    stub = SimpleNamespace()
    stub._draw_first_last = _VelocityWidget._draw_first_last.__get__(stub)
    fig = Figure()
    stub._figure = fig
    stub._gs = lambda: {"velocity_show_points": show}
    stub._n_bouts_spin = SimpleNamespace(value=lambda: 1)
    groups = {"a1": "A", "a2": "A", "b1": "B"}
    stub._host = SimpleNamespace(
        _session_groups_for_controls=lambda _c: groups,
        _ordered_group_list=lambda names, *a: sorted(names),
        _split_factors_for_controls=lambda _c: [],
        ordered_session_labels=lambda: list(groups),
    )
    stub._vel_facet_controls = None
    stub._vel_checked_groups = lambda: {"A", "B"}
    stub._behavior_combo = SimpleNamespace(currentText=lambda: "Rear")
    data = [
        {"session_label": s, "start_frame": f, "mean_vel": v, "peak_vel": v}
        for s in groups for f, v in ((0, 1.0), (10, 2.0))
    ]
    try:
        stub._draw_first_last(data, "mean", "group", "cm/s")
    except AttributeError:
        pass  # later styling helpers are not stubbed; the bars are drawn first
    # 3 sessions x (first, last) = 6 per-session means.
    assert _dots(fig.axes[0]) == expected
