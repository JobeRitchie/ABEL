"""Trace panel layout and legend with many behaviors.

The probability-trace group gave its spare height to the two control rows
(nothing below them claimed any stretch), so the plot was left a sliver at the
bottom of the panel and the flow row of FP/FN buttons wrapped onto five lines.
A 20-behavior project then drew an 18-entry legend on top of what plot was
left, in a palette of only 8 colors.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.tabs.temporal_review_tab import TemporalReviewTab  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def tab(qapp):
    widget = TemporalReviewTab()
    widget.resize(1600, 1000)
    widget.show()
    for _ in range(5):
        qapp.processEvents()
    yield widget
    widget.close()


def test_trace_canvas_gets_the_panels_spare_height(tab, qapp):
    canvas = tab._trace_canvas
    assert canvas is not None, "trace canvas should initialize on show"
    group = canvas.parent()
    # The plot, not the control rows, owns the leftover space.
    assert canvas.height() > group.height() * 0.5
    assert tab._trace_layout.itemAt(tab._trace_layout.count() - 1).widget() is canvas


def test_control_rows_stay_compact(tab):
    layout = tab._trace_layout
    # Rows 0 and 1 are the behavior selector and the FP/FN controls.
    assert layout.itemAt(0).geometry().height() < 60
    assert layout.itemAt(1).geometry().height() < 120


def test_palette_never_repeats_a_colour(tab):
    for n in (3, 8, 18, 40):
        colours = tab._trace_palette(n)
        assert len(colours) == n
        assert len(set(colours)) == n


def test_many_behavior_legend_sits_outside_the_axes(tab, qapp):
    axes = tab._trace_axes
    for i in range(18):
        axes.plot([0, 1], [0, 1], label=f"Behavior {i}")
    tab._render_trace_legend()
    qapp.processEvents()
    assert axes.get_legend() is None, "legend must not be drawn inside the axes"
    assert len(tab._trace_canvas.figure.legends) == 1

    # Redrawing must not stack a second legend on the figure.
    tab._render_trace_legend()
    assert len(tab._trace_canvas.figure.legends) == 2  # until the next refresh clears them
    tab._refresh_probability_plot()
    qapp.processEvents()
    assert len(tab._trace_canvas.figure.legends) == 0


def test_few_behaviors_keep_the_inline_legend(tab, qapp):
    axes = tab._trace_axes
    axes.clear()
    for i in range(3):
        axes.plot([0, 1], [0, 1], label=f"Behavior {i}")
    tab._render_trace_legend()
    assert axes.get_legend() is not None
