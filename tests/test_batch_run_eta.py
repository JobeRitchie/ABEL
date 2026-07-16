"""Whole-run ETA for batch (pipeline-all / retrain-all) runs.

These lock in the fix for the bug where a batch run's ETA reset to a single
behavior's worth at every behavior boundary instead of estimating the time to
finish *all* remaining behaviors. The estimator is driven exactly as the
Active-Learning pipeline-all wrapper drives it: ``update(behavior_index,
stage_index)`` on each progress event, timed by a wall clock.
"""

from __future__ import annotations

from abel.utils.eta_estimator import StageEtaEstimator, blend_whole_run_eta


def _clock():
    t = [0.0]
    return t, (lambda: t[0])


def test_batch_eta_spans_all_remaining_behaviors():
    # 4 behaviors × 5 stages, each stage ~1s of wall-clock time.
    t, clock = _clock()
    est = StageEtaEstimator(n_items=4, stages_per_item=5, clock=clock)

    # Behavior 0 runs fully: enter each stage one second apart.
    for stage in range(5):
        t[0] = float(stage)
        est.update(0, stage)

    # Enter behavior 1, stage 0 (behavior 0 took 5 s of wall time).
    t[0] = 5.0
    eta = est.update(1, 0)

    # Whole-run remaining spans behaviors 1, 2 and 3 — ~15 stages × 1 s ≈ 15 s.
    # The pre-fix per-behavior ETA would top out near one behavior (~5 s).
    assert eta > 12.0, f"ETA {eta:.1f}s should span all remaining behaviors, not one"


def test_batch_eta_seed_gives_calibrated_start():
    # With a prior-run seed, the very first progress event reports a full-run ETA
    # instead of 0 (3 behaviors × 4 stages × 2 s ≈ 24 s).
    t, clock = _clock()
    est = StageEtaEstimator(
        n_items=3, stages_per_item=4, seed_stage_seconds=2.0, clock=clock
    )
    eta = est.update(0, 0)
    assert 20.0 <= eta <= 28.0, f"seeded start ETA {eta:.1f}s out of range"


def test_eta_calibration_gate():
    # Without a seed, the estimator is "calculating" until a full item's worth of
    # stage durations has been observed; then it reports calibrated.
    t, clock = _clock()
    est = StageEtaEstimator(n_items=4, stages_per_item=3, clock=clock)
    assert not est.is_calibrated()
    for stage in range(3):          # cross all 3 stages of item 0
        t[0] = float(stage)
        est.update(0, stage)
    t[0] = 3.0
    est.update(1, 0)                # entering item 1 records item 0's last stage
    assert est.is_calibrated()

    # A prior-run seed makes it calibrated immediately.
    seeded = StageEtaEstimator(n_items=4, stages_per_item=3, seed_stage_seconds=2.0)
    assert seeded.is_calibrated()


def test_single_behavior_run_calibrates():
    # A one-behavior batch run never enters a second item, so the final stage is
    # never booked. Requiring a *full* item's samples would leave it stuck on
    # "calculating" for the whole run; a nearly-complete item must calibrate it.
    t, clock = _clock()
    est = StageEtaEstimator(n_items=1, stages_per_item=5, clock=clock)
    assert not est.is_calibrated()
    for stage in range(5):
        t[0] = float(stage)
        est.update(0, stage)
    # 4 of 5 stages measured (the last is never crossed) — calibrated anyway.
    assert est.is_calibrated()


def test_pipeline_all_preamble_does_not_reset_calibration():
    # Reproduces the pipeline-all wrapper's estimator management: each behavior's
    # pipeline opens with a transient backend-detection preamble reporting a tiny
    # maximum (steps=1). The wrapper must (re)build the estimator only when the
    # stage count *grows*, so the preamble can't wipe what earlier behaviors
    # measured and reset the ETA to "calculating…" at every behavior boundary.
    n_total = 3
    real_steps = 6
    t, clock = _clock()
    holder: dict[str, object] = {"est": None, "steps": None}

    def drive(beh_idx: int, max_scaled: int, val_scaled: int) -> None:
        steps = max(1, max_scaled // 100)
        est = holder["est"]
        if est is None or steps > (holder["steps"] or 0):  # grow-only
            est = StageEtaEstimator(n_total, steps, clock=clock)
            holder["est"] = est
            holder["steps"] = steps
        stage = min((holder["steps"] or 1) - 1, val_scaled // 100)
        est.update(beh_idx, stage)

    # Behavior 0: backend preamble (steps=1), then all real stages (steps=6).
    drive(0, 100, 0)  # preamble: maximum=1
    for stage in range(real_steps):
        t[0] += 1.0
        drive(0, real_steps * 100, stage * 100)

    # Behavior 1 opens with the same steps=1 preamble. Under the old
    # recreate-on-change logic this rebuilt the estimator and reset calibration.
    t[0] += 1.0
    drive(1, 100, 0)

    assert holder["steps"] == real_steps, "preamble must not shrink the stage count"
    assert holder["est"].is_calibrated(), (
        "estimator should stay calibrated across the behavior-1 preamble"
    )


def test_batch_eta_shrinks_toward_completion():
    # Uses wall-clock deltas between stage entries; ETA should be large early and
    # collapse toward ~one stage as the final behavior's last stage is entered.
    t, clock = _clock()
    est = StageEtaEstimator(n_items=3, stages_per_item=4, clock=clock)

    eta_after_first_behavior = None
    step = 0
    last_eta = None
    for item in range(3):
        for stage in range(4):
            t[0] = float(step)
            last_eta = est.update(item, stage)
            step += 1
            if item == 1 and stage == 0:
                eta_after_first_behavior = last_eta

    # Entering behavior 1 (2 of 3 behaviors remain) the ETA must exceed the
    # near-final ETA, which should be down to roughly one stage (~1 s).
    assert eta_after_first_behavior is not None
    assert eta_after_first_behavior > last_eta
    assert last_eta <= 2.0


def test_blend_falls_back_to_live_without_history():
    # No prior completed run of this kind → the blend is just the live estimate.
    assert blend_whole_run_eta(None, elapsed=10.0, live_remaining=42.0, frac=0.3) == 42.0
    assert blend_whole_run_eta(0.0, elapsed=10.0, live_remaining=42.0, frac=0.3) == 42.0


def test_blend_anchors_to_measured_total_at_start():
    # At the very start (frac≈0) the ETA equals the measured whole-run total,
    # regardless of a noisy early live estimate — the drift-free anchor.
    eta = blend_whole_run_eta(hist_total=120.0, elapsed=0.0, live_remaining=5.0, frac=0.0)
    assert abs(eta - 120.0) < 1e-9


def test_blend_shifts_to_live_as_run_progresses():
    # Same live estimate, growing progress fraction: weight moves from the
    # historical total toward the live measurement, so the two converge.
    hist = 100.0
    elapsed = 60.0
    live_remaining = 60.0  # live says total 120 s (machine slower than history)
    early = blend_whole_run_eta(hist, elapsed, live_remaining, frac=0.1)
    late = blend_whole_run_eta(hist, elapsed, live_remaining, frac=0.9)
    # Early trusts history (total≈100 → remaining≈40); late trusts live (≈60).
    assert early < late
    assert abs(late - 60.0) < abs(early - 60.0)


def test_blend_never_negative():
    # If we blow past the blended total, the remaining ETA clamps to zero.
    eta = blend_whole_run_eta(hist_total=30.0, elapsed=100.0, live_remaining=0.0, frac=0.2)
    assert eta == 0.0
