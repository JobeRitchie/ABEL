"""Cooperative cancellation for long background jobs.

A Cancel / Stop button sets ``flag[0] = True`` on a one-element list shared
with the worker.  Long loops call :func:`check_cancel` (directly, or through a
progress callback wrapped by :func:`cancellable`) and unwind with
:class:`OperationCancelled`.

A Pause button uses the same stop points: with a :class:`JobFlag`, they block
until the job is resumed or stopped.

``OperationCancelled`` derives from ``BaseException`` on purpose: services wrap
optional steps (R3D features, caches, per-session extras) in broad
``except Exception`` handlers that log and carry on.  A cancel raised inside
one of those must not be mistaken for "this optional step failed", or the run
keeps going and writes a result the user asked to abandon.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

CANCEL_MARKER = "CANCELLED_BY_USER"


class OperationCancelled(BaseException):
    """The user asked the running job to stop."""

    def __init__(self, message: str = f"OPERATION_{CANCEL_MARKER}") -> None:
        super().__init__(message)


_shutdown = threading.Event()


def request_shutdown() -> None:
    """Ask every running job to stop, as if its Stop button were pressed.

    Used when the app is closing, so a job ends at one of its safe stop points
    (never between the writes of a model) instead of being killed mid-write.
    """
    _shutdown.set()


def shutdown_requested() -> bool:
    return _shutdown.is_set()


_PAUSE_POLL_SEC = 0.1


class JobFlag(list):
    """A cancel flag (``flag[0]``) whose job can also be paused.

    It is still the one-element list every service already takes, so nothing
    that reads ``flag[0]`` changes.  While paused, every stop point
    (:func:`check_cancel`, :func:`checkpoint`, a :func:`cancellable` progress
    callback) blocks instead of returning.  A paused job therefore always sits
    at a place where stopping is already safe, and a Stop pressed during the
    pause unwinds from that same place.
    """

    def __init__(self) -> None:
        super().__init__([False])
        self._lock = threading.Lock()
        self._resume = threading.Event()
        self._resume.set()
        self._paused_at: float | None = None
        self._paused_total = 0.0
        self._waiting = 0
        self._listeners: list[Callable[[bool], None]] = []

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    @property
    def waiting(self) -> int:
        """How many of the job's threads are blocked at a stop point right now."""
        return self._waiting

    def pause(self) -> None:
        with self._lock:
            if self.paused:
                return
            self._paused_at = time.monotonic()
            self._resume.clear()
            self._notify(True)

    def resume(self) -> None:
        with self._lock:
            if not self.paused:
                return
            self._paused_total += time.monotonic() - float(self._paused_at or 0.0)
            self._paused_at = None
            self._resume.set()
            self._notify(False)

    def clock(self) -> float:
        """``time.monotonic()`` with paused time taken out, for ETAs and timings."""
        with self._lock:
            now = self._paused_at if self._paused_at is not None else time.monotonic()
            return now - self._paused_total

    def add_pause_listener(self, listener: Callable[[bool], None]) -> None:
        """Call ``listener(paused)`` on every pause and resume, and once now."""
        with self._lock:
            self._listeners.append(listener)
            self._call(listener, self.paused)

    def remove_pause_listener(self, listener: Callable[[bool], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _notify(self, paused: bool) -> None:
        for listener in list(self._listeners):
            self._call(listener, paused)

    @staticmethod
    def _call(listener: Callable[[bool], None], paused: bool) -> None:
        try:
            listener(paused)
        except Exception:  # a listener must never break pause / resume
            pass


def is_paused(flag: "list[bool] | None") -> bool:
    return bool(getattr(flag, "paused", False))


def wait_if_paused(flag: "list[bool] | None") -> None:
    """Block while ``flag``'s job is paused.  Returns at once on resume, Stop or app close."""
    resume = getattr(flag, "_resume", None)
    if resume is None or resume.is_set():
        return
    with flag._lock:
        flag._waiting += 1
    try:
        while not resume.wait(_PAUSE_POLL_SEC):
            if _shutdown.is_set() or flag[0]:
                return
    finally:
        with flag._lock:
            flag._waiting -= 1


def is_cancelled(flag: "list[bool] | None") -> bool:
    return _shutdown.is_set() or (bool(flag) and bool(flag[0]))


def check_cancel(flag: "list[bool] | None", exc_type: type = OperationCancelled) -> None:
    wait_if_paused(flag)
    if is_cancelled(flag):
        raise exc_type()


def cancellable(
    callback: "Callable[..., Any] | None",
    flag: "list[bool] | None",
    exc_type: type = OperationCancelled,
) -> "Callable[..., Any] | None":
    """Wrap a progress callback so every call is also a cancel checkpoint.

    Progress callbacks fire from deep inside extraction, training and scoring
    loops, so this turns them into frequent stop points without threading the
    flag through every service.  Returns ``callback`` unchanged when there is
    no flag.
    """
    if flag is None:
        return callback

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        check_cancel(flag, exc_type)
        if callback is not None:
            return callback(*args, **kwargs)
        return None

    return _wrapped


_scope = threading.local()


@contextmanager
def cancel_scope(flag: "list[bool] | None", exc_type: type = OperationCancelled) -> Iterator[None]:
    """Bind ``flag`` to this thread so deep loops can call :func:`checkpoint`.

    Lets services with no cancel parameter (R3D decoding, dense scoring) stop
    between batches.  Threads started inside the scope do not inherit it, so
    work fanned out to an executor needs an explicit check instead.
    """
    prev = getattr(_scope, "binding", None)
    _scope.binding = (flag, exc_type)
    try:
        yield
    finally:
        _scope.binding = prev


def propagate_scope(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Carry this thread's cancel scope into ``fn`` when it runs on a pool thread."""
    binding = getattr(_scope, "binding", None)
    if binding is None:
        return fn

    def _run(*args: Any, **kwargs: Any) -> Any:
        with cancel_scope(*binding):
            return fn(*args, **kwargs)

    return _run


def checkpoint() -> None:
    """Raise if the job running on this thread was cancelled; else no-op."""
    binding = getattr(_scope, "binding", None)
    if binding is not None:
        check_cancel(binding[0], binding[1])
    elif _shutdown.is_set():
        raise OperationCancelled()


def is_cancel_traceback(traceback_text: str) -> bool:
    """True when a worker's failure text is a user cancel, not an error."""
    return CANCEL_MARKER in (traceback_text or "")
