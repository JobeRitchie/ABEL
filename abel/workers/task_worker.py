"""Generic QRunnable helper for background jobs."""

from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from abel.utils.cancellation import OperationCancelled


class WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    line_emitted = Signal(str)  # live stdout/stderr lines from long-running subprocesses
    progress = Signal(object)  # structured progress events (dicts) from the task


_active_lock = threading.Lock()
_active_count = 0


def active_task_count() -> int:
    """Number of TaskWorker jobs currently running (for the close guard)."""
    with _active_lock:
        return _active_count


class TaskWorker(QRunnable):
    """Runs blocking callables without freezing the UI."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        global _active_count
        with _active_lock:
            _active_count += 1
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except (Exception, OperationCancelled):
            # A cancel reaches ``failed`` too; handlers tell it apart with
            # ``is_cancel_traceback``.
            self.signals.failed.emit(traceback.format_exc())
        finally:
            with _active_lock:
                _active_count -= 1
