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
    ) -> None:
        self._n_items = max(1, int(n_items))
        self._stages = max(1, int(stages_per_item))
        self._clock = clock
        self._total = self._n_items * self._stages
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
            return 0.0
        return sum(self._stage_mean.values()) / len(self._stage_mean)

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
