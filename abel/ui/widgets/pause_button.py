"""Pause / Resume button for a background job that has a :class:`JobFlag`."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QPushButton, QWidget

from abel.utils.cancellation import JobFlag

# Dingbat bars, not U+23F8: Windows draws that one as a color emoji.
_PAUSE_TEXT = "❚❚  Pause"
_RESUME_TEXT = "▶  Resume"


class PauseButton(QPushButton):
    """Toggles a job between running and paused.

    ``pause_requested`` fires on the click.  The job keeps going until its next
    stop point, so ``pause_reached`` fires once it is really waiting there.
    """

    pause_requested = Signal()
    pause_reached = Signal()
    resumed = Signal()

    def __init__(self, flag: JobFlag, parent: QWidget | None = None) -> None:
        super().__init__(_PAUSE_TEXT, parent)
        self._flag = flag
        self.setEnabled(False)
        self.setToolTip(
            "Pause the running job at its next safe point and free the CPU.\n"
            "Press Resume to carry on from the same place, or Stop to end the run.\n"
            "A paused job stays in memory, so keep ABEL open while it is paused."
        )
        self._poll = QTimer(self)
        self._poll.setInterval(150)
        self._poll.timeout.connect(self._check_reached)
        self.clicked.connect(self._toggle)

    def set_job_running(self, running: bool) -> None:
        """Call when a job starts or ends; a new job never starts out paused."""
        self._poll.stop()
        self._flag.resume()
        self.setText(_PAUSE_TEXT)
        self.setEnabled(bool(running))

    def _toggle(self) -> None:
        if self._flag.paused:
            self._poll.stop()
            self._flag.resume()
            self.setText(_PAUSE_TEXT)
            self.resumed.emit()
        else:
            self._flag.pause()
            self.setText(_RESUME_TEXT)
            self._poll.start()
            self.pause_requested.emit()

    def _check_reached(self) -> None:
        if self._flag.paused and self._flag.waiting > 0:
            self._poll.stop()
            self.pause_reached.emit()
