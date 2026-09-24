"""Cooperative cancellation for long background jobs.

A Cancel / Stop button sets ``flag[0] = True`` on a one-element list shared
with the worker.  Long loops call :func:`check_cancel` (directly, or through a
progress callback wrapped by :func:`cancellable`) and unwind with
:class:`OperationCancelled`.

``OperationCancelled`` derives from ``BaseException`` on purpose: services wrap
optional steps (R3D features, caches, per-session extras) in broad
``except Exception`` handlers that log and carry on.  A cancel raised inside
one of those must not be mistaken for "this optional step failed", or the run
keeps going and writes a result the user asked to abandon.
"""

from __future__ import annotations

import threading
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


def is_cancelled(flag: "list[bool] | None") -> bool:
    return _shutdown.is_set() or (bool(flag) and bool(flag[0]))


def check_cancel(flag: "list[bool] | None", exc_type: type = OperationCancelled) -> None:
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
