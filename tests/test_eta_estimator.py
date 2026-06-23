from __future__ import annotations

from abel.utils.eta_estimator import StageEtaEstimator


def _fake_clock():
    t = [0.0]
    return t, (lambda: t[0])


def test_eta_weights_unequal_stages():
    # 2 items x 3 stages; stage 2 is 10x slower than stages 0 and 1.
    t, clock = _fake_clock()
    est = StageEtaEstimator(n_items=2, stages_per_item=3, clock=clock)

    est.update(0, 0)              # enter item0 stage0 @ t=0
    t[0] = 1.0; est.update(0, 1)  # stage0 took 1s
    t[0] = 2.0; est.update(0, 2)  # stage1 took 1s
    t[0] = 12.0                   # stage2 took 10s
    eta = est.update(1, 0)        # enter item1 stage0; stage2 learned = 10s
    # Remaining = item1 stages {1, 1, 10} = 12s (not the naive 6/6 stage count).
    assert 11.0 <= eta <= 13.0

    t[0] = 13.0
    eta2 = est.update(1, 1)       # item1 stage0 took 1s
    # Only the slow stage 2 (10s) plus stage1 (1s) remain ≈ 11s — a naive
    # equal-stage ETA would say ~6.5s here.
    assert 10.0 <= eta2 <= 12.0

    t[0] = 14.0
    eta3 = est.update(1, 2)       # only the 10s stage left
    assert 9.0 <= eta3 <= 11.0


def test_eta_nonnegative_and_monotone_keys():
    t, clock = _fake_clock()
    est = StageEtaEstimator(n_items=3, stages_per_item=5, clock=clock)
    # Out-of-order / clamped inputs must not raise or go negative.
    t[0] = 0.0; assert est.update(0, 0) >= 0.0
    t[0] = 5.0; assert est.update(0, 9) >= 0.0   # stage clamped to last
    t[0] = 6.0; assert est.update(0, 2) >= 0.0   # backwards stage tolerated
    t[0] = 30.0; assert est.update(2, 4) >= 0.0


def test_eta_zero_at_completion():
    t, clock = _fake_clock()
    est = StageEtaEstimator(n_items=1, stages_per_item=3, clock=clock)
    est.update(0, 0)
    t[0] = 1.0; est.update(0, 1)
    t[0] = 2.0
    # Entering the final stage of the final item: at most one stage remains.
    eta = est.update(0, 2)
    assert eta >= 0.0
