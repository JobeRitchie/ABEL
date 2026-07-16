"""ETA for a multi-item × multi-stage job whose stages have unequal cost.

A retrain-all run executes the same ordered *stages* for each *item* (behavior).
The stages have very different wall-clock costs — loading labels is near-instant
while evaluation reads parquet and cross-validates — so a naive "fraction of
stages completed" ETA runs ahead of (or behind) real time and oscillates on
every stage boundary.

``StageEtaEstimator`` instead learns each stage's typical duration from the
durations observed so far in this run, then estimates the remaining time as the
sum of the expected durations of all not-yet-run stages.  Because each stage
index is averaged against its own past samples, a consistently slow stage no
longer drags the estimate around when a fast stage completes.
"""

from __future__ import annotations

import time
from typing import Callable


def blend_whole_run_eta(
    hist_total: float | None,
    elapsed: float,
    live_remaining: float,
    frac: float,
    live_calibrated: bool = False,
) -> float:
    """Blend a measured whole-run total with the live per-stage estimate.

    ``hist_total`` is the whole run's expected wall time from prior completed
    runs of the same kind (``None`` when this run kind has never finished before);
    ``live_remaining`` is the current per-stage estimator's seconds-left;
    ``frac`` is the fraction of the whole run completed so far (0..1);
    ``live_calibrated`` is True once the per-stage estimator has measured roughly
    a full item's worth of stages on THIS run.

    Early in a run the per-stage estimator has measured little and its remaining
    sum swings, so we lean on the historical whole-run total. But that anchor is
    only "drift-free" while the workload is unchanged: after the dataset grows
    (e.g. new videos → more segments to score) the per-behavior mean can lag real
    time by *multiples*, and a plain ``frac``-weighted blend then keeps the ETA
    pinned near the stale-low anchor for most of the run — the classic
    "underestimates the whole run" symptom. Once the live estimator has calibrated
    it reflects this run/dataset's actual pace, so we hand it the majority of the
    weight immediately (``≥0.8``) instead of waiting for ``frac`` to climb. History
    then only lightly damps the early-calibration estimate; by the end the live
    figure is authoritative either way.
    """
    if hist_total is None or hist_total <= 0.0:
        return max(0.0, live_remaining)
    f = min(1.0, max(0.0, frac))
    live_total = elapsed + max(0.0, live_remaining)
    # Uncalibrated: original frac ramp (trust history while live still swings).
    # Calibrated: live is trustworthy now — give it ≥0.8 weight so a stale
    # historical anchor cannot suppress it, still ramping to fully-live by the end.
    w_live = max(f, 0.8) if live_calibrated else f
    blended_total = (1.0 - w_live) * hist_total + w_live * live_total
    return max(0.0, blended_total - elapsed)


class StageEtaEstimator:
    """Learn per-stage durations and estimate remaining time.

    Usage: call :meth:`update` once when *entering* each ``(item, stage)`` —
    item in ``[0, n_items)``, stage in ``[0, stages_per_item)`` — and use the
    returned seconds as the ETA.  ``stages_per_item`` is the count emitted as
    ``maximum`` by the inner task (5 or 6).
    """

    def __init__(
        self,
        n_items: int,
        stages_per_item: int,
        clock: Callable[[], float] = time.monotonic,
        seed_stage_seconds: float | None = None,
    ) -> None:
        self._n_items = max(1, int(n_items))
        self._stages = max(1, int(stages_per_item))
        self._clock = clock
        self._total = self._n_items * self._stages
        # Optional prior-run seed: a coarse per-stage wall-clock guess used as the
        # fallback for not-yet-measured stages, so the very first progress update
        # already reports a calibrated whole-run ETA instead of 0. Measured stages
        # always override it.
        self._seed = (
            float(seed_stage_seconds)
            if seed_stage_seconds and seed_stage_seconds > 0.0
            else None
        )
        # stage index -> running mean duration + sample count
        self._stage_mean: dict[int, float] = {}
        self._stage_n: dict[int, int] = {}
        self._last_key: int | None = None
        self._last_ts: float | None = None

    def _global_key(self, item: int, stage: int) -> int:
        item = min(max(0, item), self._n_items - 1)
        stage = min(max(0, stage), self._stages - 1)
        return item * self._stages + stage

    def _record(self, from_key: int, to_key: int, total_dur: float) -> None:
        """Attribute ``total_dur`` evenly across the stage slots just crossed."""
        crossed = to_key - from_key
        if crossed <= 0:
            return
        per = total_dur / crossed
        for k in range(from_key, to_key):
            s = k % self._stages
            n = self._stage_n.get(s, 0)
            mean = self._stage_mean.get(s, 0.0)
            self._stage_mean[s] = (mean * n + per) / (n + 1)
            self._stage_n[s] = n + 1

    def _expected(self, stage: int, fallback: float) -> float:
        return self._stage_mean.get(stage, fallback)

    def _overall_avg(self) -> float:
        if not self._stage_mean:
            return self._seed if self._seed is not None else 0.0
        return sum(self._stage_mean.values()) / len(self._stage_mean)

    def is_calibrated(self) -> bool:
        """True once the ETA rests on real data (or a prior-run seed).

        Until roughly a full item's worth of stage durations has been observed
        (and no seed is available), the estimate swings wildly, so callers should
        show a "calculating" placeholder rather than a misleading number.

        A stage's duration is only booked when the run *crosses into the next*
        stage, so a single item yields at most ``stages - 1`` measured stages —
        the final stage isn't timed until the following item begins (and for a
        one-item run, never).  Requiring the full ``stages`` samples therefore
        keeps single-item and just-finished-first-item runs stuck on
        "calculating" forever, so we calibrate once a nearly-complete item has
        been measured instead.
        """
        if self._seed is not None:
            return True
        return sum(self._stage_n.values()) >= max(1, self._stages - 1)

    def update(self, item: int, stage: int) -> float:
        """Record entry into ``(item, stage)``; return estimated seconds left."""
        now = self._clock()
        key = self._global_key(item, stage)

        if self._last_ts is not None and self._last_key is not None and key > self._last_key:
            self._record(self._last_key, key, now - self._last_ts)
        # Monotonic guard: never move the marker backwards.
        if self._last_key is None or key >= self._last_key:
            self._last_key = key
            self._last_ts = now

        fallback = self._overall_avg()
        remaining = 0.0
        for k in range(key, self._total):
            remaining += self._expected(k % self._stages, fallback)
        return max(0.0, remaining)
