"""Dragging an existing ROI must move it, not start a new one.

Repositioning a zone used to mean redrawing it from scratch for every subject,
which is destructive for freehand polygons: the new trace is never the old
outline.  The canvas now hit-tests the press point first, so a press inside an
overlay grabs it and the drag translates the shape rigidly.  These tests drive
the real ``_ROICanvas`` mouse handlers, because the bug class they guard
against (press routed to the rubber-band path) only exists in that dispatch.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.tabs.roi_definition_tab import _ROICanvas  # noqa: E402
from abel.utils import roi_geometry  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(qapp):
    c = _ROICanvas()
    c.resize(640, 480)
    c.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    # A 1:1 frame keeps canvas pixels equal to image pixels, so the asserted
    # offsets below are the drag distances themselves.
    assert c._scale == 1.0 and (c._offset_x, c._offset_y) == (0, 0)
    return c


def _press(canvas, x, y, shift=False):
    mods = (
        Qt.KeyboardModifier.ShiftModifier if shift
        else Qt.KeyboardModifier.NoModifier
    )
    canvas.mousePressEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(x, y),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, mods))


def _move(canvas, x, y, held=True):
    buttons = Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton
    canvas.mouseMoveEvent(QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(x, y),
        Qt.MouseButton.NoButton, buttons, Qt.KeyboardModifier.NoModifier))


def _release(canvas, x, y):
    canvas.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease, QPointF(x, y),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier))


def _drag(canvas, x, y, dx, dy):
    _press(canvas, x, y)
    _move(canvas, x + dx, y + dy)
    _release(canvas, x + dx, y + dy)


def test_drag_moves_rect_without_resizing(canvas):
    canvas.set_draw_mode("roi_0")
    canvas.set_roi_at(0, {"x": 100, "y": 100, "w": 120, "h": 90})
    emitted = []
    canvas.roi_n_changed.connect(lambda i, r: emitted.append((i, dict(r))))

    _drag(canvas, 160, 145, 40, 25)

    assert roi_geometry.roi_bbox(canvas._rois[0]) == (140, 125, 120, 90)
    assert emitted and emitted[-1][0] == 0


def test_drag_translates_polygon_rigidly(canvas):
    poly = roi_geometry.normalize_roi(
        {"shape": "polygon", "points": [[300, 300], [400, 300], [380, 380], [310, 370]]}
    )
    canvas.set_n_rois(2)
    canvas.set_draw_mode("roi_1")
    canvas.set_roi_at(1, poly)

    _drag(canvas, 340, 335, -40, -20)

    moved = canvas._rois[1]
    assert roi_geometry.roi_shape(moved) == "polygon"
    deltas = {
        (round(n[0] - o[0]), round(n[1] - o[1]))
        for o, n in zip(poly["points"], moved["points"])
    }
    assert deltas == {(-40, -20)}


def test_press_on_empty_area_still_draws(canvas):
    canvas.set_draw_mode("roi_0")
    canvas.set_roi_at(0, {"x": 0, "y": 0, "w": 0, "h": 0})

    _press(canvas, 400, 60)
    assert canvas._move_target is None
    assert canvas._drag_origin is not None
    _release(canvas, 500, 140)

    assert roi_geometry.roi_bbox(canvas._rois[0]) == (400, 60, 100, 80)


def test_shift_press_redraws_over_an_existing_roi(canvas):
    canvas.set_draw_mode("roi_0")
    canvas.set_roi_at(0, {"x": 100, "y": 100, "w": 120, "h": 90})

    _press(canvas, 160, 145, shift=True)
    assert canvas._move_target is None
    _release(canvas, 200, 200)

    assert roi_geometry.roi_bbox(canvas._rois[0]) == (160, 145, 40, 55)


def test_drag_clamps_to_the_frame(canvas):
    canvas.set_draw_mode("roi_0")
    canvas.set_roi_at(0, {"x": 10, "y": 10, "w": 100, "h": 50})

    _drag(canvas, 60, 35, -500, -500)

    assert roi_geometry.roi_bbox(canvas._rois[0]) == (0, 0, 100, 50)


def test_subject_crop_is_draggable(canvas):
    canvas.set_draw_mode("subject_crop")
    canvas.set_crop({"x": 50, "y": 400, "w": 80, "h": 60})
    emitted = []
    canvas.crop_changed.connect(lambda r: emitted.append(dict(r)))

    _drag(canvas, 90, 430, 30, 0)

    assert roi_geometry.roi_bbox(canvas._crop) == (80, 400, 80, 60)
    assert emitted


def test_hover_cursor_marks_grabbable_rois(canvas):
    canvas.set_draw_mode("roi_0")
    canvas.set_roi_at(0, {"x": 100, "y": 100, "w": 120, "h": 90})

    _move(canvas, 110, 110, held=False)
    assert canvas.cursor().shape() == Qt.CursorShape.OpenHandCursor

    _move(canvas, 5, 5, held=False)
    assert canvas.cursor().shape() == Qt.CursorShape.CrossCursor
