"""Worker processes for context-feature frame chunks.

The per-frame work in a context chunk (crops, MOG2, small NumPy reductions) is
mostly Python-level, so chunk *threads* queue on the interpreter lock: measured
on 352x240 video, 4, 8 and 16 threads all ran about 2,400 frames/s while 8
processes ran about 8,200.  This module runs the same chunk functions, with the
same frame ranges, in a small pool of worker processes.

The pool is opt-in (feature extraction asks for it), shared by every session of
a run, and shut down when the run ends so its GPU contexts are released before
the next stage.  Anything that stops the pool from working (it cannot start, a
worker dies, an argument will not pickle) returns ``None`` and the caller runs
the chunks on threads exactly as before.
"""

from __future__ import annotations

import atexit
import concurrent.futures as cf
import logging
import multiprocessing as mp
import os
import pickle
import sys
import threading
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable

from abel.utils.cancellation import OperationCancelled, shutdown_requested

logger = logging.getLogger("abel")

# More than 8 workers gained under 10% in measurement and each costs memory.
MAX_WORKERS = 8
# A CUDA context plus optical-flow buffers per worker (measured about 550 MB
# each with 8 workers busy on 352x240 video); the reserve is left for the rest.
_WORKER_VRAM_MB = 600.0
_VRAM_RESERVE_MB = 1536.0
_POLL_SEC = 0.5

_state_lock = threading.Lock()
_pool: cf.ProcessPoolExecutor | None = None
_pool_cancel: Any = None
_pool_size = 0
_disabled_reason: str | None = None

# Set once in each worker by ``_init_worker``.
_worker_cancel: Any = None
_worker_gpu_lock: Any = None


def _main_module_is_safe() -> bool:
    """True when starting a worker will not re-run the program's entry script.

    A spawned worker re-imports the parent's ``__main__``.  The app
    (``python -m abel.main``) and pytest guard their entry points, so that is
    harmless.  A plain script without an ``if __name__ == "__main__"`` guard
    would run again, top to bottom, inside every worker.  Such launches stay on
    threads unless ``ABEL_CONTEXT_PROCESSES`` opts in explicitly.
    """
    main = sys.modules.get("__main__")
    name = str(getattr(getattr(main, "__spec__", None), "name", "") or "")
    return name.split(".")[0] in ("abel", "pytest")


def planned_workers(use_gpu: bool) -> int:
    """How many chunk processes to run; ``0`` means stay on threads."""
    env = os.environ.get("ABEL_CONTEXT_PROCESSES", "").strip()
    if env:
        try:
            return max(0, min(int(env), 32))
        except ValueError:
            pass
    if not _main_module_is_safe():
        return 0
    n = min(MAX_WORKERS, max(1, (os.cpu_count() or 1) - 1))
    if use_gpu:
        try:
            from abel.utils.gpu_optical_flow import gpu_vram_free_mb

            free = float(gpu_vram_free_mb())
        except Exception:  # pragma: no cover - probe is best-effort
            free = 0.0
        if free > 0:
            n = min(n, int((free - _VRAM_RESERVE_MB) // _WORKER_VRAM_MB))
    return n if n >= 2 else 0


def _init_worker(cancel_event: Any, gpu_lock: Any) -> None:
    global _worker_cancel, _worker_gpu_lock
    _worker_cancel = cancel_event
    _worker_gpu_lock = gpu_lock


def _worker_cancel_check() -> None:
    if _worker_cancel is not None and _worker_cancel.is_set():
        raise OperationCancelled()


def _run_chunk(kind: str, kwargs: dict, frame_start: int, frame_end: int) -> tuple[dict, list[str]]:
    """Run one frame chunk in a worker; returns its columns and any GPU warnings."""
    from abel.services.context_feature_service import ContextFeatureService

    warnings: list[str] = []
    if kind == "gpu":
        out = ContextFeatureService._process_video_chunk_gpu(
            frame_start=frame_start, frame_end=frame_end,
            gpu_flow_lock=_worker_gpu_lock, warning_cb=warnings.append,
            cancel_check=_worker_cancel_check, **kwargs,
        )
    else:
        out = ContextFeatureService._process_video_chunk(
            frame_start=frame_start, frame_end=frame_end,
            cancel_check=_worker_cancel_check, **kwargs,
        )
    return out, warnings


def _get_pool(use_gpu: bool) -> tuple[cf.ProcessPoolExecutor, Any] | None:
    global _pool, _pool_cancel, _pool_size, _disabled_reason
    with _state_lock:
        if _disabled_reason is not None:
            return None
        if _pool is not None:
            return _pool, _pool_cancel
        n = planned_workers(use_gpu)
        if n < 2:
            return None
        try:
            ctx = mp.get_context("spawn")
            cancel_event = ctx.Event()
            # One GPU optical-flow batch at a time across workers, as the
            # thread path does with its in-process lock.
            gpu_lock = ctx.Lock()
            _pool = cf.ProcessPoolExecutor(
                max_workers=n, mp_context=ctx,
                initializer=_init_worker, initargs=(cancel_event, gpu_lock),
            )
        except Exception as exc:
            _disabled_reason = str(exc)
            logger.warning("Context chunk processes unavailable (%s); using threads.", exc)
            return None
        _pool_cancel = cancel_event
        _pool_size = n
        logger.info("Context features: running frame chunks in %d worker process(es).", n)
        return _pool, _pool_cancel


def shutdown(cancel: bool = False) -> None:
    """Stop the pool (if any) and release its workers' GPU memory."""
    global _pool, _pool_cancel, _pool_size
    with _state_lock:
        pool, event = _pool, _pool_cancel
        _pool, _pool_cancel, _pool_size = None, None, 0
    if pool is None:
        return
    if cancel and event is not None:
        event.set()
    try:
        pool.shutdown(wait=not cancel, cancel_futures=True)
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        logger.debug("Context chunk pool shutdown: %s", exc)


atexit.register(shutdown, True)


def run_chunks(
    kind: str,
    chunk_kwargs: dict,
    chunk_boundaries: list[tuple[int, int]],
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    warning_cb: Callable[[str], None] | None = None,
) -> list[dict] | None:
    """Run every ``(start, end)`` chunk in the pool; results in chunk order.

    Returns ``None`` when the pool cannot be used, so the caller falls back to
    threads.  An error raised by a chunk itself, or a cancel, propagates.
    """
    global _disabled_reason
    got = _get_pool(use_gpu=(kind == "gpu"))
    if got is None:
        return None
    pool, cancel_event = got

    # Locks and callbacks stay in this process; workers get their own.
    kwargs = {
        k: v for k, v in chunk_kwargs.items()
        if k not in ("gpu_flow_lock", "warning_cb", "cancel_check")
    }
    n_chunks = len(chunk_boundaries)
    try:
        pickle.dumps(kwargs)
        futures = {
            pool.submit(_run_chunk, kind, kwargs, start, end): idx
            for idx, (start, end) in enumerate(chunk_boundaries)
        }
    except Exception as exc:
        logger.warning("Context chunk processes: could not submit work (%s); using threads.", exc)
        return None

    results: list[dict] = [{}] * n_chunks
    pending = set(futures)
    done_count = 0
    try:
        while pending:
            if cancel_check is not None:
                cancel_check()
            finished, pending = cf.wait(pending, timeout=_POLL_SEC, return_when=cf.FIRST_COMPLETED)
            for future in finished:
                idx = futures[future]
                data, warnings = future.result()
                results[idx] = data
                done_count += 1
                if warning_cb is not None:
                    for msg in warnings:
                        warning_cb(msg)
                if progress_cb is not None:
                    start, end = chunk_boundaries[idx]
                    progress_cb(
                        done_count, n_chunks,
                        f"chunk {done_count}/{n_chunks} done (frames {start}–{end - 1})",
                    )
    except BrokenProcessPool as exc:
        # A worker died (out of memory, killed).  Drop the pool for good this
        # session and let the caller redo these chunks on threads.
        with _state_lock:
            _disabled_reason = f"worker process died: {exc}"
        shutdown(cancel=True)
        logger.warning("Context chunk processes stopped (%s); using threads.", exc)
        return None
    except (cf.CancelledError, OperationCancelled):
        # The user cancelled, or the pool was stopped under this session.
        for future in pending:
            future.cancel()
        shutdown(cancel=True)
        if cancel_check is not None:
            cancel_check()  # re-raises the caller's own cancel type
        if shutdown_requested():
            raise
        logger.warning("Context chunk processes were stopped mid-session; using threads.")
        return None
    except BaseException:
        # A real error inside a chunk (for example an unreadable video).  Only
        # this session's remaining chunks are dropped; the pool stays up for
        # the other sessions.
        for future in pending:
            future.cancel()
        raise
    return results
