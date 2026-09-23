"""Stop the mouse wheel from changing settings while the user scrolls a page.

Qt spin boxes, combo boxes and sliders react to the wheel whenever the cursor
passes over them, so scrolling a settings panel silently edits whatever
control drifts under the pointer.  This application-wide filter makes those
controls ignore the wheel unless the user has clicked into them first; the
ignored event propagates to the enclosing scroll area, so the page scrolls.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QScrollBar,
)

_GUARDED = (QAbstractSpinBox, QComboBox, QAbstractSlider)


def _is_guarded(obj: QObject) -> bool:
    return isinstance(obj, _GUARDED) and not isinstance(obj, QScrollBar)


class WheelGuard(QObject):
    """Event filter installed on the QApplication (see ``install_wheel_guard``)."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        etype = event.type()
        if etype == QEvent.Type.Polish:
            # WheelFocus lets a wheel tick grab focus before the event is
            # delivered, which would defeat the hasFocus() check below.
            if _is_guarded(obj) and obj.focusPolicy() == Qt.FocusPolicy.WheelFocus:
                obj.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        elif etype == QEvent.Type.Wheel:
            if _is_guarded(obj) and not obj.hasFocus():
                event.ignore()
                return True
        return False


_guard: WheelGuard | None = None


def install_wheel_guard(app: QApplication) -> WheelGuard:
    global _guard
    if _guard is None:
        _guard = WheelGuard(app)
        app.installEventFilter(_guard)
    return _guard
