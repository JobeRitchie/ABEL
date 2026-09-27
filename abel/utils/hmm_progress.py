"""Per-iteration progress reporting for hmmlearn fits."""

from __future__ import annotations

from typing import Any, Callable


def reporting_monitor(monitor: Any, on_iter: Callable[[int, float], None]) -> Any:
    """Copy of hmmlearn's convergence monitor that reports each EM step.

    hmmlearn has no progress hook, but it documents replacing ``monitor_``;
    ``report`` runs once per iteration inside ``fit``.  ``on_iter`` gets the
    iteration number and that iteration's log-likelihood gain (NaN at first).
    """
    base = type(monitor)

    class _Reporting(base):  # type: ignore[misc, valid-type]
        def report(self, log_prob: float) -> None:
            super().report(log_prob)
            h = self.history
            gain = float(h[-1] - h[-2]) if len(h) >= 2 else float("nan")
            on_iter(int(self.iter), gain)

    return _Reporting(monitor.tol, monitor.n_iter, monitor.verbose)
