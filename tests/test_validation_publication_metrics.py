"""Tests for the publication-grade validation metrics + biological-readout and
calibration analyses added to the external validation suite."""

from __future__ import annotations

import numpy as np

from abel.validation import metrics as vm
from abel.validation.analyses import calibration, time_budget
from abel.validation.analyses.generalization import HoldoutPredictions


# ── Imbalance-robust classification metrics ────────────────────────────────


def test_mcc_perfect_and_degenerate():
    y = np.array([0, 0, 1, 1, 0, 1])
    assert vm.matthews_corrcoef(y, y) == 1.0
    # Single predicted class → MCC denominator 0; sklearn's convention is 0.0
    # ("no correlation"), which we surface rather than crashing.
    assert vm.matthews_corrcoef(y, np.zeros_like(y)) == 0.0
    # Empty input is safe.
    assert np.isnan(vm.matthews_corrcoef(np.array([]), np.array([])))


def test_balanced_accuracy_and_specificity():
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 0])
    # sensitivity = 1/2, specificity = 3/4 -> balanced acc = 0.625
    assert abs(vm.balanced_accuracy(y_true, y_pred) - 0.625) < 1e-9
    assert abs(vm.specificity(y_true, y_pred) - 0.75) < 1e-9
    # Single-class truth: balanced accuracy undefined; specificity still defined.
    assert np.isnan(vm.balanced_accuracy(np.ones(4), np.array([1, 0, 1, 0])))


def test_roc_auc_separable_and_single_class():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert vm.roc_auc(y, score) == 1.0
    assert np.isnan(vm.roc_auc(np.ones(4), score))  # one class → NaN


# ── Agreement metrics ──────────────────────────────────────────────────────


def test_ccc_penalizes_scale_shift_but_pearson_does_not():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = x + 0.2  # perfectly correlated but shifted
    assert abs(vm.pearson_r(x, y) - 1.0) < 1e-9
    # CCC must drop below 1 because of the constant location shift.
    assert vm.concordance_ccc(x, y) < 0.95
    # Identical series → CCC == 1.
    assert abs(vm.concordance_ccc(x, x) - 1.0) < 1e-9


def test_bland_altman_bias_and_loa():
    true = np.array([0.10, 0.20, 0.30, 0.40])
    pred = true + 0.05  # constant +0.05 over-scoring
    ba = vm.bland_altman(true, pred)
    assert abs(ba.bias - 0.05) < 1e-9
    assert ba.sd_diff < 1e-9  # constant offset → no spread
    lo, hi = ba.bias_ci95()
    assert lo <= 0.05 <= hi


def test_pearson_guards_short_and_constant():
    assert np.isnan(vm.pearson_r([1.0, 2.0], [1.0, 2.0]))         # <3 pairs
    assert np.isnan(vm.pearson_r([1.0, 1.0, 1.0], [1, 2, 3]))     # constant x


# ── Calibration ────────────────────────────────────────────────────────────


def test_calibration_perfect_is_low_error():
    # Scores that exactly match empirical rates: 0.0 for negatives, 1.0 positives.
    y = np.array([0] * 50 + [1] * 50)
    score = np.array([0.0] * 50 + [1.0] * 50)
    curve = vm.calibration_curve(y, score, n_bins=10)
    assert curve.ece < 1e-9
    assert curve.brier < 1e-9


def test_calibration_miscalibrated_has_error():
    # Overconfident: always predicts 0.9 but only right half the time.
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.5).astype(int)
    score = np.full(200, 0.9)
    curve = vm.calibration_curve(y, score, n_bins=10)
    assert curve.ece > 0.3  # ~|0.5 − 0.9|
    assert 0.0 <= curve.brier <= 1.0


# ── Time-budget analysis ────────────────────────────────────────────────────


def _make_preds(fracs: dict[str, tuple[float, float]], seg_per_session: int = 20):
    """Build synthetic HoldoutPredictions with target true/pred time fractions.

    ``fracs`` maps session id -> (true_fraction, pred_fraction). Each session gets
    ``seg_per_session`` equal-length 1-frame windows; the first k positives encode
    the requested fraction. Frame bounds are laid out contiguously.
    """
    sess_ids, animals, sf, ef, yt, yp = [], [], [], [], [], []
    for sess, (tf, pf) in fracs.items():
        n_true = int(round(tf * seg_per_session))
        n_pred = int(round(pf * seg_per_session))
        for i in range(seg_per_session):
            sess_ids.append(sess)
            animals.append("")
            sf.append(i)
            ef.append(i)
            yt.append(1 if i < n_true else 0)
            yp.append(1 if i < n_pred else 0)
    return HoldoutPredictions(
        project_id="P", behavior_id="b", behavior_name="Freeze",
        session_ids=np.array(sess_ids, dtype=object),
        animal_ids=np.array(animals, dtype=object),
        start_frames=np.array(sf, dtype=np.int64),
        end_frames=np.array(ef, dtype=np.int64),
        y_true=np.array(yt, dtype=int),
        y_pred=np.array(yp, dtype=int),
        prob=np.array(yp, dtype=float),
    )


def test_time_budget_recovers_prevalence_and_correlates():
    preds = _make_preds({
        "s1": (0.10, 0.12),
        "s2": (0.30, 0.28),
        "s3": (0.50, 0.55),
        "s4": (0.70, 0.68),
    })
    res = time_budget.run_time_budget(preds)
    assert res is not None and not res.error
    assert res.n_units == 4
    # Prevalence recovered close to requested truth.
    assert np.allclose(sorted(res.true_prevalence), [0.1, 0.3, 0.5, 0.7], atol=1e-6)
    assert res.prev_pearson_r > 0.98
    assert res.prev_ccc > 0.95
    assert abs(res.prev_bias) < 0.05


def test_time_budget_needs_three_sessions():
    preds = _make_preds({"s1": (0.2, 0.2), "s2": (0.4, 0.4)})
    res = time_budget.run_time_budget(preds)
    assert res is not None and res.error  # too few sessions to correlate
    assert res.n_units == 2


def test_bout_count_requires_contiguous_segments():
    # Contiguous rows: alternating positives → each positive window is its own bout.
    assert time_budget._bout_count(np.array([1, 0, 1, 0, 1, 0]), contiguous=True) == 3
    assert time_budget._bout_count(np.array([0, 1, 1, 1, 0]), contiguous=True) == 1
    assert time_budget._bout_count(np.zeros(5, dtype=int), contiguous=True) == 0
    # Non-contiguous labeling (the real case: reviewed clips are minutes apart) →
    # a "bout" is undefined, and must be NaN rather than a fabricated integer.
    assert np.isnan(time_budget._bout_count(np.array([1, 0, 1]), contiguous=False))


def test_bouts_suppressed_when_segments_are_far_apart():
    """Reviewed clips that sit minutes apart must not be merged into one bout.

    Regression test for the real data: ABEL's labeled segments cover ~1.5% of a
    session and adjacent rows can be thousands of frames apart, so counting runs of
    adjacent positive ROWS invents bouts that never happened.
    """
    n = 6
    # Six 15-frame clips, each separated by a 1000-frame unlabeled gap.
    starts = np.arange(n, dtype=np.int64) * 1000
    preds = HoldoutPredictions(
        project_id="P", behavior_id="b", behavior_name="Freeze",
        session_ids=np.array(["s1"] * n, dtype=object),
        animal_ids=np.array([""] * n, dtype=object),
        start_frames=starts,
        end_frames=starts + 14,
        y_true=np.array([1, 1, 0, 1, 1, 0]),
        y_pred=np.array([1, 1, 0, 1, 1, 0]),
        prob=np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.1]),
    )
    units = time_budget._unit_series(preds)
    row = units.iloc[0]
    assert row["contiguity"] == 0.0                      # no adjacent pair touches
    assert np.isnan(row["true_bouts"])                   # so bouts are undefined
    assert row["coverage_frac"] < 0.05                   # ~90 of 5015 frames labeled
    # Prevalence is still well defined (4 of 6 clips positive).
    assert abs(row["true_prevalence"] - 4 / 6) < 1e-6


def test_time_budget_none_when_no_predictions():
    assert time_budget.run_time_budget(None) is None


# ── Small-sample confidence intervals ──────────────────────────────────────


def test_ci95_uses_t_not_1p96():
    """A 1.96 multiplier at 3 seeds is really an 81% interval — over-calls significance."""
    assert abs(vm.t_critical_95(3) - 4.303) < 1e-3   # df=2
    assert abs(vm.t_critical_95(2) - 12.706) < 1e-3  # df=1
    assert vm.t_critical_95(30) < 2.1                # converges toward 1.96

    vals = [0.10, 0.12, 0.14]
    sem = np.std(vals, ddof=1) / np.sqrt(3)
    assert abs(vm.ci95(vals) - 4.303 * sem) < 1e-6
    # Strictly wider than the old normal-approx interval it replaces.
    assert vm.ci95(vals) > 1.96 * sem
    # A single seed yields no spread → 0.0 (never "significant").
    assert vm.ci95([0.5]) == 0.0


def test_paired_p_matches_scipy_and_declines_degenerate_input():
    """The one paired t-test the suite shares — ablation, video-value and
    discrimination all call this, so it must not drift between them."""
    vals = [0.031, 0.028, 0.035, 0.030, 0.033]
    scipy_stats = __import__("scipy.stats", fromlist=["stats"])
    assert abs(vm.paired_p(vals)
               - float(scipy_stats.ttest_1samp(vals, 0.0).pvalue)) < 1e-12
    # A real, consistent gain is significant; noise straddling zero is not.
    assert vm.paired_p(vals) < 0.05
    assert vm.paired_p([0.01, -0.02, 0.03, -0.015, 0.005]) > 0.05

    # Undefined, not zero: one seed has no spread, and a constant difference is an
    # infinite t that must never be reported as p = 0.
    assert np.isnan(vm.paired_p([0.05]))
    assert np.isnan(vm.paired_p([0.05, 0.05, 0.05]))
    assert np.isnan(vm.paired_p([]))
    # NaN seeds are dropped, not propagated — a failed fit must not sink the test.
    assert abs(vm.paired_p([0.031, np.nan, 0.028, 0.035, 0.030, 0.033])
               - vm.paired_p(vals)) < 1e-12


def test_benjamini_hochberg_threshold_controls_the_discovery_rate():
    """A discrimination run tests ~40-100 pair x family combinations, so a bare
    p<0.05 line expects a handful of false positives by construction."""
    # 10 tests, 3 genuinely tiny p's. k=3 is the last rejection, so the critical
    # value is 3/10 * 0.05 — stricter than a bare 0.05, looser than Bonferroni.
    ps = [1e-6, 1e-5, 1e-4] + [0.30, 0.44, 0.51, 0.62, 0.73, 0.88, 0.95]
    thr = vm.benjamini_hochberg_threshold(ps)
    assert abs(thr - 3 / 10 * 0.05) < 1e-12
    assert 0.05 / len(ps) <= thr < 0.05      # between Bonferroni and uncorrected
    # It is a line ABOVE the rejected points, not one drawn through the last of them.
    rejected = [p for p in ps if p <= thr]
    assert rejected == [1e-6, 1e-5, 1e-4]
    assert max(rejected) < thr
    # Nothing survives → NaN, so the caller draws no line rather than an invented one.
    assert np.isnan(vm.benjamini_hochberg_threshold([0.4, 0.6, 0.9]))
    assert np.isnan(vm.benjamini_hochberg_threshold([]))
    # A single clearly-significant test is rejected, at the full alpha.
    assert vm.benjamini_hochberg_threshold([0.001]) == 0.05
    # NaN p-values (an undefined test) are excluded from the correction's m.
    assert abs(vm.benjamini_hochberg_threshold([0.001, float("nan")]) - 0.05) < 1e-12


# ── Calibration analysis wrapper ────────────────────────────────────────────


def test_run_calibration_from_predictions():
    y = np.array([0] * 30 + [1] * 30)
    prob = np.array([0.1] * 30 + [0.9] * 30)
    preds = HoldoutPredictions(
        project_id="P", behavior_id="b", behavior_name="Groom",
        session_ids=np.array(["s"] * 60, dtype=object),
        animal_ids=np.array([""] * 60, dtype=object),
        start_frames=np.arange(60, dtype=np.int64),
        end_frames=np.arange(60, dtype=np.int64),
        y_true=y, y_pred=(prob >= 0.5).astype(int), prob=prob,
    )
    res = calibration.run_calibration(preds)
    assert res is not None
    assert res.curve.n == 60
    assert 0.0 <= res.ece <= 0.2
    df = calibration.calibration_rows([res])
    assert list(df["behavior"]) == ["Groom"]
