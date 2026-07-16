"""Per-project cross-run-type timing profile used to seed ETAs."""

from __future__ import annotations

from abel.utils.run_timing_profile import RunTimingProfile


def test_record_and_means():
    p = RunTimingProfile()
    assert not p.has_data()
    assert p.overall_mean() == 0.0

    # Scoring is the expensive phase; preparing is cheap.
    p.record("Preparing", 2.0)
    p.record("Training", 4.0)
    p.record("Scoring", 20.0)
    p.record("Evaluating", 4.0)

    assert p.has_data()
    assert p.phase_seconds("Scoring") == 20.0
    # Unknown phase falls back to the overall (sample-weighted) mean.
    assert p.phase_seconds("Nope") == p.overall_mean()
    # One item's total is the sum of the phase means.
    assert p.behavior_total() == 30.0
    # Per-step seed rescales to the run's step granularity.
    assert p.per_step_seed(6) == 5.0
    assert p.per_step_seed(10) == 3.0


def test_running_mean_is_adaptive():
    p = RunTimingProfile()
    for _ in range(5):
        p.record("Scoring", 10.0)
    assert abs(p.phase_seconds("Scoring") - 10.0) < 1e-9
    # A newer, slower observation moves the mean upward (not ignored).
    p.record("Scoring", 40.0)
    assert 10.0 < p.phase_seconds("Scoring") < 40.0


def test_cross_run_type_sharing():
    # A retrain run only ever saw Preparing/Training/Scoring/Evaluating; a later
    # pipeline run (more, finer steps) still gets a calibrated per-step seed from
    # that shared knowledge.
    p = RunTimingProfile()
    for phase, secs in [("Preparing", 2.0), ("Training", 5.0), ("Scoring", 18.0), ("Evaluating", 5.0)]:
        p.record(phase, secs)
    pipeline_seed = p.per_step_seed(15)   # pipeline reports ~15 steps/behavior
    assert pipeline_seed > 0
    assert abs(pipeline_seed - (30.0 / 15)) < 1e-9


def test_roundtrip_serialization():
    p = RunTimingProfile()
    p.record("Training", 3.0)
    p.record("Scoring", 12.0)
    restored = RunTimingProfile.from_dict(p.to_dict())
    assert restored.phase_seconds("Scoring") == 12.0
    assert restored.behavior_total() == p.behavior_total()
    assert restored.has_data()


def test_behavior_total_excludes_non_core_phases():
    # A manual whole-run UMAP ("Embedding") must not inflate the one-item total
    # used to seed per-behavior ETAs.
    p = RunTimingProfile()
    p.record("Training", 5.0)
    p.record("Scoring", 15.0)
    p.record("Embedding", 30.0)
    assert p.behavior_total() == 20.0  # Embedding excluded
    assert p.per_step_seed(4) == 5.0


def test_ignores_garbage():
    p = RunTimingProfile()
    p.record("", 5.0)
    p.record("Training", 0.0)
    p.record("Training", -3.0)
    p.record("Training", 10_000_000.0)  # absurd → ignored
    assert not p.has_data()


def test_run_total_normalises_per_behavior():
    # A completed 4-behavior run that took 240 s → 60 s per behavior. A future
    # run of a different behavior count scales linearly from that measured cost.
    p = RunTimingProfile()
    assert not p.has_run_total("pipeline")
    p.record_run_total("pipeline", 240.0, 4)
    assert p.has_run_total("pipeline")
    assert abs(p.run_total_per_behavior("pipeline") - 60.0) < 1e-9
    assert abs(p.run_total_seconds("pipeline", 3) - 180.0) < 1e-9


def test_run_total_is_kept_per_kind():
    # A scoring-only run-model pass is far cheaper per behavior than a full
    # pipeline; their whole-run totals must not be pooled.
    p = RunTimingProfile()
    p.record_run_total("pipeline", 100.0, 1)
    p.record_run_total("run_models", 5.0, 1)
    assert abs(p.run_total_per_behavior("pipeline") - 100.0) < 1e-9
    assert abs(p.run_total_per_behavior("run_models") - 5.0) < 1e-9
    # An unknown kind has no measured total (and is not confused with another).
    assert not p.has_run_total("retrain")
    assert p.run_total_per_behavior("retrain") == 0.0


def test_per_step_seed_prefers_run_total_over_phase_sum():
    # The phase-sum systematically drifts from real wall time; when a measured
    # whole-run total exists for the kind it wins as the per-step seed basis.
    p = RunTimingProfile()
    p.record("Training", 4.0)
    p.record("Scoring", 20.0)  # phase-sum behavior_total == 24 s
    p.record_run_total("pipeline", 60.0, 1)  # but the real run took 60 s
    assert abs(p.per_step_seed(6, "pipeline") - 10.0) < 1e-9   # 60 / 6
    # A kind without a run total falls back to the phase-sum.
    assert abs(p.per_step_seed(6, "retrain") - 4.0) < 1e-9     # 24 / 6
    # No kind given → phase-sum fallback (backward compatible).
    assert abs(p.per_step_seed(6) - 4.0) < 1e-9


def test_run_total_adaptive_and_roundtrips():
    p = RunTimingProfile()
    for _ in range(5):
        p.record_run_total("pipeline", 100.0, 1)
    p.record_run_total("pipeline", 200.0, 1)  # slower recent run nudges upward
    assert 100.0 < p.run_total_per_behavior("pipeline") < 200.0
    restored = RunTimingProfile.from_dict(p.to_dict())
    assert restored.has_run_total("pipeline")
    assert abs(
        restored.run_total_per_behavior("pipeline")
        - p.run_total_per_behavior("pipeline")
    ) < 1e-2  # to_dict rounds seconds to 3 dp


def test_run_total_ignores_garbage():
    p = RunTimingProfile()
    p.record_run_total("", 10.0, 1)        # no kind
    p.record_run_total("pipeline", 0.0, 1)  # non-positive
    p.record_run_total("pipeline", 10_000_000.0, 1)  # absurd
    assert not p.has_run_total("pipeline")
    assert not p.has_data()
