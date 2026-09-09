"""A modal "please wait" popup for operations that block the UI.

Opening a project and refreshing analytics both do enough file I/O on large
projects that the window looks frozen.  This is the one shared indicator for
that: an application-modal dialog with an indeterminate bar, no close button
and no Cancel, shown while the work runs and dismissed when it finishes.

Two design points matter:

* It is shown with ``show()``, never ``exec()``.  The blocking form would stop
  the event loop the background work needs, and under the offscreen Qt platform
  used by the test suite it would wait forever.  :func:`show_busy` is a no-op
  on those platforms for the same reason.
* ``closeEvent`` is deliberately *not* blocked. The titlebar has no close
  button, so the only way one arrives is the application shutting down — and a
  popup that refuses to close there would make a stuck background job into an
  unquittable app. Esc is swallowed by :meth:`BusyDialog.reject` instead.
* Sizing comes from font metrics, not fixed pixels, so the text does not clip
  under Windows display scaling.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


def _is_non_interactive() -> bool:
    """True when no human can see a dialog (headless test/benchmark runs)."""
    return os.environ.get("QT_QPA_PLATFORM", "").strip().lower() in {
        "offscreen", "minimal", "vnc",
    }


class BusyDialog(QDialog):
    """Indeterminate "working…" popup with no way to dismiss it by hand."""

    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        message: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        # No close button / context help: the dialog owns its own lifetime and
        # a half-loaded project behind a dismissed popup is worse than waiting.
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )

        layout = QVBoxLayout(self)
        fm = self.fontMetrics()
        margin = fm.height()
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(fm.height() // 2)

        self._label = QLabel(message, self)
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        bar = QProgressBar(self)
        bar.setRange(0, 0)  # indeterminate
        bar.setTextVisible(False)
        bar.setFixedHeight(fm.height())
        layout.addWidget(bar)

        # Wide enough for the longest message we set, measured rather than guessed.
        self.setMinimumWidth(max(fm.horizontalAdvance(message) + 6 * margin,
                                 fm.horizontalAdvance("W") * 34))

    def set_message(self, message: str) -> None:
        """Update the wait text without closing and reopening the popup."""
        self._label.setText(message)
        # A longer message wraps rather than clips; grow to fit it.
        self.adjustSize()

    def reject(self) -> None:  # noqa: D102 - Esc must not dismiss the popup
        return

    def finish(self) -> None:
        """Close the popup and release it for deletion."""
        super().reject()
        self.deleteLater()


def show_busy(parent: QWidget | None, title: str, message: str) -> BusyDialog | None:
    """Show a :class:`BusyDialog`, or return ``None`` when headless.

    Callers must pair every non-``None`` return with :meth:`BusyDialog.finish`,
    including on the error path.
    """
    if _is_non_interactive():
        return None
    dlg = BusyDialog(parent, title, message)
    dlg.show()
    # One forced paint so the popup is actually on screen before the caller
    # starts work that may not yield to the event loop for a while.
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415
    QApplication.processEvents()
    return dlg


def close_busy(dialog: BusyDialog | None) -> None:
    """Dismiss a popup from :func:`show_busy`; safe with ``None``."""
    if dialog is not None:
        dialog.finish()
