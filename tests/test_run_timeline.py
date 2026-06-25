"""Tests for the stage-aware run timeline / ETA estimator."""

from __future__ import annotations

from abel.utils.run_timeline import RunTimeline, Stage, format_duration


class _Clock:
    """Deterministic monotonic clock for tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> None:
        self.t += dt


def _stages() -> list[Stage]:
    return [
        Stage("a", "Stage A", weight=1.0),
        Stage("b", "Stage B", weight=1.0, total_units=4),
        Stage("c", "Stage C", weight=2.0),
    ]


def test_fraction_and_completion_progress() -> None:
    clk = _Clock()
    tl = RunTimeline(_stages(), now_fn=clk)
    tl.start()
    assert tl.fraction() == 0.0

    tl.start_stage("a")
    clk.tick(10.0)
    tl.complete_stage("a")
    snap = tl.snapshot()
    assert snap.stages[0].state == "done"
    assert snap.stages[0].elapsed_seconds == 10.0
    # Some work done, fraction must have advanced.
    assert 0.0 < snap.fraction < 1.0


def test_running_stage_extrapolates_from_subprogress() -> None:
    clk = _Clock()
    tl = RunTimeline(_stages(), now_fn=clk)
    tl.start_stage("b", total_units=4)
    clk.tick(20.0)
    tl.advance("b", 2)  # half done in 20s -> projects ~40s total
    snap = tl.snapshot()
    b = next(s for s in snap.stages if s.key == "b")
    assert b.state == "running"
    assert b.estimate_seconds >= 38.0  # ~40s projection


def test_history_seeds_pending_estimate() -> None:
    clk = _Clock()
    history = {"c": {"seconds": 50.0}}
    tl = RunTimeline(_stages(), history=history, now_fn=clk)
    tl.start()
    c = next(s for s in tl.snapshot().stages if s.key == "c")
    assert abs(c.estimate_seconds - 50.0) < 1e-6


def test_adaptive_scale_adjusts_pending_from_actuals() -> None:
    clk = _Clock()
    # Seed says A=10s and C=100s. If A actually takes 20s (2x slower), the
    # pending estimate for C should scale up toward ~200s.
    history = {"a": {"seconds": 10.0}, "c": {"seconds": 100.0}}
    tl = RunTimeline(_stages(), history=history, now_fn=clk)
    tl.start_stage("a")
    clk.tick(20.0)
    tl.complete_stage("a")
    c = next(s for s in tl.snapshot().stages if s.key == "c")
    assert c.estimate_seconds > 150.0  # scaled up from 100s


def test_skip_stage_contributes_zero() -> None:
    clk = _Clock()
    tl = RunTimeline(_stages(), now_fn=clk)
    tl.start()
    tl.skip_stage("a")
    snap = tl.snapshot()
    a = next(s for s in snap.stages if s.key == "a")
    assert a.state == "skipped"
    assert a.estimate_seconds == 0.0


def test_to_history_roundtrip_records_per_unit() -> None:
    clk = _Clock()
    tl = RunTimeline(_stages(), now_fn=clk)
    tl.start_stage("b", total_units=4)
    clk.tick(40.0)
    tl.advance("b", 4)
    tl.complete_stage("b")
    hist = tl.to_history()
    assert "b" in hist
    assert abs(hist["b"]["per_unit"] - 10.0) < 1e-6


def test_format_duration_units() -> None:
    assert format_duration(0.25).endswith("ms")
    assert format_duration(12.3) == "12.3 s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3725) == "1h 02m"
