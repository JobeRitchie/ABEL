"""Scrolling a page must not change the spin boxes / combos it passes over."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from abel.ui.wheel_guard import install_wheel_guard


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    install_wheel_guard(a)
    return a


def _wheel(widget, delta=-120):
    pos = QPointF(widget.rect().center())
    ev = QWheelEvent(
        pos,
        QPointF(widget.mapToGlobal(pos.toPoint())),
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, ev)
    return ev


def _page(app):
    area = QScrollArea()
    area.setWidgetResizable(True)
    inner = QWidget()
    lay = QVBoxLayout(inner)
    spin = QSpinBox()
    spin.setRange(0, 100)
    spin.setValue(50)
    combo = QComboBox()
    combo.addItems(["a", "b", "c"])
    combo.setCurrentIndex(1)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(50)
    for w in (spin, combo, slider):
        lay.addWidget(w)
    lay.addSpacing(3000)
    area.setWidget(inner)
    area.resize(300, 200)
    area.show()
    app.processEvents()
    return area, spin, combo, slider


def test_unfocused_controls_ignore_wheel_and_page_scrolls(app):
    area, spin, combo, slider = _page(app)
    area.setFocus()  # showing the window hands focus to the first control
    app.processEvents()
    for w in (spin, combo, slider):
        assert not w.hasFocus()
        # Left unaccepted, a real (spontaneous) wheel event propagates to the
        # scroll area; synthetic events never propagate, so check the flag.
        assert not _wheel(w).isAccepted()
    assert spin.value() == 50
    assert combo.currentIndex() == 1
    assert slider.value() == 50
    area.close()


def test_focused_control_still_takes_wheel(app):
    area, spin, _combo, _slider = _page(app)
    area.activateWindow()
    spin.setFocus()
    app.processEvents()
    if not spin.hasFocus():
        pytest.skip("platform did not grant focus")
    _wheel(spin, delta=120)
    assert spin.value() == 51
    area.close()


def test_wheel_focus_downgraded_to_strong_focus(app):
    area, spin, combo, _slider = _page(app)
    for w in (spin, combo):
        assert w.focusPolicy() != Qt.FocusPolicy.WheelFocus
    area.close()
