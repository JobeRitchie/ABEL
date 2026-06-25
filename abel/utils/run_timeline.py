"""Stage-aware run timeline with a self-refining time-to-completion estimate.

A :class:`RunTimeline` models a long-running job as an ordered list of named
:class:`Stage` objects.  As the job runs it records how long each stage actually
takes *on this machine, this run*, and continuously refines the estimate for the
remaining stages — so the predicted finish time gets more accurate as work
progresses instead of relying on a single fixed guess.

Design goals:
- **Qt-free and pure** so it can be unit-tested without a GUI (the GUI layer in
  this project cannot even import in headless CI).
- **Seeded** from the previous run's measured durations (persisted per project)
  so the very first progress update already shows a calibrated ETA.
- **Adaptive**: once a few stages finish, the ratio of their *actual* time to
  their *seeded* time scales the estimates of the not-yet-started stages, so a
  faster/slower-than-usual machine (or dataset) is accounted for live.

The class only tracks time and progress; rendering is the caller's job (see
``abel.ui.widgets.progress_panel``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Stage:
    """One unit of work in a run.

    ``weight`` is a relative expected-duration hint used only when no measured
    history exists yet (e.g. the first run ever).  ``total_units`` lets a stage
    report sub-progress — e.g. one unit per session for per-session
    preprocessing — so the bar advances smoothly within a long stage.
    """

    key: str
    label: str
    weight: float = 1.0
    total_units: int = 1

    # ── runtime state ────────────────────────────────────────────────
    started_at: float | None = None
    ended_at: float | None = None
    done_units: int = 0
    skipped: bool = False

    @property
    def is_running(self) -> bool:
        return self.started_at is not None and self.ended_at is None and not self.skipped

    @property
    def is_done(self) -> bool:
        return self.ended_at is not None or self.skipped

    def measured_seconds(self, now: float) -> float | None:
        """Wall seconds spent in this stage so far (or total if finished)."""
        if self.started_at is None or self.skipped:
            return None
        end = self.ended_at if self.ended_at is not None else now
        return max(0.0, end - self.started_at)


@dataclass
class StageView:
    """Immutable per-stage snapshot for rendering."""

    key: str
    label: str
    state: str  # "pending" | "running" | "done" | "skipped"
    done_units: int
    total_units: int
    elapsed_seconds: float | None
    estimate_seconds: float


@dataclass
class TimelineSnapshot:
    """Immutable view of the whole run for rendering."""

    fraction: float                 # overall completion in [0, 1]
    elapsed_seconds: float
    remaining_seconds: float
    total_estimate_seconds: float
    stages: list[StageView] = field(default_factory=list)
    active_stage_key: str | None = None


# Reasonable bounds so a single weird stage can't blow up the whole estimate.
_SCALE_MIN = 0.2
_SCALE_MAX = 5.0


class RunTimeline:
    """Track stage progress and produce a live, self-refining ETA."""

    def __init__(
        self,
        stages: list[Stage],
        *,
        history: dict | None = None,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if not stages:
            raise ValueError("RunTimeline requires at least one stage")
        self._stages = stages
        self._now_fn = now_fn
        self._started_at: float | None = None
        # history: {stage_key: {"seconds": float, "per_unit": float}}
        self._history = dict(history or {})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started_at is None:
            self._started_at = self._now_fn()

    def start_stage(self, key: str, *, total_units: int | None = None) -> None:
        self.start()
        st = self._stage(key)
        if total_units is not None:
            st.total_units = max(1, int(total_units))
        st.done_units = 0
        st.skipped = False
        st.ended_at = None
        st.started_at = self._now_fn()

    def advance(self, key: str, done_units: int | None = None, *, delta: int = 1) -> None:
        """Update sub-progress within a running stage."""
        st = self._stage(key)
        if st.started_at is None:
            self.start_stage(key)
        if done_units is not None:
            st.done_units = int(done_units)
        else:
            st.done_units += int(delta)
        st.done_units = max(0, min(st.done_units, st.total_units))

    def complete_stage(self, key: str) -> None:
        st = self._stage(key)
        if st.started_at is None:
            # A stage that completed without ever "starting" (instantaneous);
            # record a zero-length span so estimates treat it as done.
            st.started_at = self._now_fn()
        st.done_units = st.total_units
        st.ended_at = self._now_fn()

    def skip_stage(self, key: str) -> None:
        st = self._stage(key)
        st.skipped = True
        st.done_units = st.total_units
        if st.started_at is None:
            st.started_at = self._now_fn()
        st.ended_at = st.started_at

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def _seed_estimate(self, st: Stage, scale: float) -> float:
        """Estimated total seconds for a stage that hasn't finished."""
        hist = self._history.get(st.key) or {}
        per_unit = float(hist.get("per_unit", 0.0) or 0.0)
        if per_unit > 0.0 and st.total_units > 1:
            base = per_unit * st.total_units
        else:
            base = float(hist.get("seconds", 0.0) or 0.0)
        if base <= 0.0:
            # No history at all — fall back to the relative weight, anchored to
            # whatever per-unit rate we've learned this run (or a 1s default).
            base = max(0.0, st.weight) * self._default_unit_seconds()
        return base * scale

    def _default_unit_seconds(self) -> float:
        """A coarse seconds-per-weight anchor learned from finished stages."""
        finished = [
            (s, s.measured_seconds(self._now_fn()))
            for s in self._stages
            if s.is_done and not s.skipped
        ]
        total_w = sum(max(0.0, s.weight) for s, _ in finished)
        total_t = sum(t for _, t in finished if t is not None)
        if total_w > 0 and total_t > 0:
            return total_t / total_w
        return 1.0

    def _adaptive_scale(self) -> float:
        """Ratio of actual to seeded time across finished, seeded stages."""
        actual = 0.0
        seeded = 0.0
        for s in self._stages:
            if not s.is_done or s.skipped:
                continue
            hist = self._history.get(s.key) or {}
            seed = float(hist.get("seconds", 0.0) or 0.0)
            if seed <= 0.0 and float(hist.get("per_unit", 0.0) or 0.0) > 0:
                seed = float(hist["per_unit"]) * s.total_units
            t = s.measured_seconds(self._now_fn())
            if seed > 0.0 and t is not None and t > 0.0:
                actual += t
                seeded += seed
        if seeded > 0.0:
            return max(_SCALE_MIN, min(_SCALE_MAX, actual / seeded))
        return 1.0

    def _stage_estimate(self, st: Stage, scale: float) -> float:
        now = self._now_fn()
        if st.skipped:
            return 0.0
        if st.is_done:
            return st.measured_seconds(now) or 0.0
        if st.is_running:
            elapsed = st.measured_seconds(now) or 0.0
            seed = self._seed_estimate(st, scale)
            # Extrapolate from observed sub-progress when available; this is the
            # most reliable signal for a long per-unit stage mid-flight.
            if st.done_units > 0 and st.total_units > 0:
                projected = elapsed * st.total_units / st.done_units
                # Trust observation more as the stage progresses.
                frac = st.done_units / st.total_units
                blended = frac * projected + (1.0 - frac) * max(seed, projected)
                return max(blended, elapsed)
            return max(seed, elapsed)
        return self._seed_estimate(st, scale)

    def total_estimate_seconds(self) -> float:
        scale = self._adaptive_scale()
        return sum(self._stage_estimate(s, scale) for s in self._stages)

    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, self._now_fn() - self._started_at)

    def remaining_seconds(self) -> float:
        return max(0.0, self.total_estimate_seconds() - self.elapsed_seconds())

    def fraction(self) -> float:
        total = self.total_estimate_seconds()
        if total <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self.elapsed_seconds() / total))

    # ------------------------------------------------------------------
    # Rendering / persistence
    # ------------------------------------------------------------------

    def snapshot(self) -> TimelineSnapshot:
        now = self._now_fn()
        scale = self._adaptive_scale()
        views: list[StageView] = []
        active: str | None = None
        for s in self._stages:
            if s.skipped:
                state = "skipped"
            elif s.is_done:
                state = "done"
            elif s.is_running:
                state = "running"
                active = s.key
            else:
                state = "pending"
            views.append(
                StageView(
                    key=s.key,
                    label=s.label,
                    state=state,
                    done_units=s.done_units,
                    total_units=s.total_units,
                    elapsed_seconds=s.measured_seconds(now),
                    estimate_seconds=self._stage_estimate(s, scale),
                )
            )
        return TimelineSnapshot(
            fraction=self.fraction(),
            elapsed_seconds=self.elapsed_seconds(),
            remaining_seconds=self.remaining_seconds(),
            total_estimate_seconds=self.total_estimate_seconds(),
            stages=views,
            active_stage_key=active,
        )

    def to_history(self) -> dict:
        """Measured per-stage durations, suitable for seeding the next run."""
        now = self._now_fn()
        out: dict[str, dict] = {}
        for s in self._stages:
            if s.skipped:
                continue
            t = s.measured_seconds(now)
            if t is None or t <= 0.0:
                continue
            entry: dict[str, float] = {"seconds": float(t)}
            if s.total_units > 1 and s.done_units > 0:
                entry["per_unit"] = float(t / s.done_units)
            out[s.key] = entry
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _stage(self, key: str) -> Stage:
        for s in self._stages:
            if s.key == key:
                return s
        raise KeyError(f"Unknown stage: {key!r}")


def format_duration(seconds: float) -> str:
    """Human-friendly duration: '450 ms', '12.3 s', '4m 05s', '1h 02m'."""
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    if seconds < 3600.0:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s:02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m"
