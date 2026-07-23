"""Publication-grade evaluation metrics for the validation platform.

A single, defensively-written home for the classifier- and agreement-metrics the
rest of the suite reports.  Every function returns ``float('nan')`` (or an empty
structure) on degenerate input instead of raising, because these run across many
heterogeneous projects/behaviors where a held-out fold can legitimately be
single-class, tiny, or empty — one bad cell must never sink a whole run.

The metrics fall into three families, each motivated by the automated-behavior
literature:

* **Imbalanced-classification summaries** — Matthews correlation coefficient
  (MCC), balanced accuracy, specificity, ROC-AUC.  Behavior labels are heavily
  skewed (rare positives), where F1/accuracy alone mislead; MCC and balanced
  accuracy are the community-recommended robust summaries (Chicco & Jurman 2020),
  and DeepEthogram reports AUROC alongside precision/recall.
* **Biological-readout agreement** — Lin's concordance correlation coefficient
  (CCC), Pearson r, R², and Bland-Altman bias / limits-of-agreement.  These grade
  whether the model recovers the *scientific measure* (per-session time budget,
  bout counts) a human scorer would report, which is the validation reviewers
  actually care about — not just per-frame accuracy.
* **Probability calibration** — expected/maximum calibration error and the Brier
  score, so a project that turns on ABEL's probability calibration can show the
  predicted scores mean what they say (a reliability claim F1 cannot capture).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── Small-sample confidence intervals ──────────────────────────────────────

# Student-t 97.5th percentile by degrees of freedom (n − 1), for the small seed
# counts this suite actually uses. Falls back to scipy, then to the normal
# approximation, for anything larger.
_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042,
}


def t_critical_95(n: int) -> float:
    """Two-sided 95% t multiplier for ``n`` observations (df = n − 1).

    Using 1.96 here — the *normal* quantile — is a real and common error at these
    sample sizes: with the suite's default of 3 seeds, a "±1.96·SEM" interval is
    actually an **81%** interval (70% at 2 seeds), so it declares differences
    significant that a genuine 95% test would not. df=2 needs 4.303, not 1.96.
    """
    df = int(n) - 1
    if df < 1:
        return float("nan")
    if df in _T_975:
        return float(_T_975[df])
    try:
        from scipy import stats  # noqa: PLC0415

        return float(stats.t.ppf(0.975, df))
    except Exception:
        # Interpolate from the table's tail; converges to 1.96 for large df.
        return float(max(1.96, _T_975[30] - (df - 30) * 0.001)) if df > 30 else 1.96


def ci95(values) -> float:
    """Half-width of the 95% CI of the mean, using the t multiplier (not 1.96).

    Returns 0.0 for fewer than 2 finite values (no spread is estimable), matching
    the suite's convention that a single seed can never be called significant.
    """
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = vals.size
    if n < 2:
        return 0.0
    sem = float(np.std(vals, ddof=1) / np.sqrt(n))
    return float(t_critical_95(n) * sem)


def paired_p(deltas) -> float:
    """Two-sided paired t-test p on per-seed differences (H0: mean difference = 0).

    The companion to :func:`ci95`: every seeded analysis in the suite reports a
    *difference* (with-feature minus without), and a boolean "significant" is not
    something a manuscript can print — the exact p is.  Because both arms saw the
    same clips under the same seed, the paired form is the correct test.

    Returns NaN, never 0, when the test is undefined: fewer than 2 seeds, or
    differences that are constant across seeds (zero variance ⇒ an infinite t).
    NaN also comes back if scipy is absent, since the t survival function has no
    small table equivalent — callers must treat p as optional and fall back on
    ``|mean| > ci95`` (which needs no scipy) for the significance decision.

    The zero-variance guard is a *tolerance*, not ``sd == 0``.  ``np.std`` of three
    identical floats leaves ~1e-18 of dust, which sails past an exact comparison and
    lets scipy return p ≈ 1e-33 — a fabricated "overwhelming" result manufactured
    out of no variance at all, which then dominates any volcano it is plotted on.
    """
    vals = np.asarray([v for v in deltas if np.isfinite(v)], dtype=float)
    if vals.size < 2:
        return float("nan")
    sd = float(np.std(vals, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12 * max(1.0, float(np.abs(vals).max())):
        return float("nan")
    try:
        from scipy import stats  # noqa: PLC0415
    except ImportError:
        return float("nan")
    return float(stats.ttest_1samp(vals, 0.0).pvalue)


def benjamini_hochberg_threshold(pvalues, alpha: float = 0.05) -> float:
    """Benjamini-Hochberg critical value at ``alpha``: reject every p at or below it.

    A full discrimination run tests ~40-100 pair × feature-family combinations, so a
    bare p<0.05 line on a volcano expects a handful of false positives by
    construction.  This returns a second, honest reference line — NaN when nothing
    survives, so the caller draws no line rather than an invented one.

    Returns the critical value ``k/m · alpha`` of the last rejection, NOT the largest
    rejected p.  The two reject exactly the same tests (no observed p can fall
    between them — one there would itself have been rejected, contradicting ``k``),
    but a line drawn at the largest rejected p lands *on top of* that point, leaving
    a reader unable to tell whether it passed.  The critical value sits cleanly above
    every point it rejects.

    Deliberately dependency-free (BH is a sort and a scan), so the figure keeps its
    multiple-comparison line even where :func:`paired_p` had to fall back to NaN.
    """
    vals = np.sort(np.asarray([v for v in pvalues if np.isfinite(v)], dtype=float))
    m = vals.size
    if m == 0:
        return float("nan")
    # Largest k with p_(k) <= k/m * alpha; every p at or below p_(k) is rejected.
    critical = (np.arange(1, m + 1) / m) * float(alpha)
    passing = np.nonzero(vals <= critical)[0]
    return float(critical[passing[-1]]) if passing.size else float("nan")


# ── Imbalanced-classification summaries ────────────────────────────────────


def _finite_binary(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    n = min(len(yt), len(yp))
    return yt[:n], yp[:n]


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation coefficient (φ). NaN when a margin is empty.

    Robust to class imbalance: a high MCC requires the classifier to do well on
    both positives and negatives simultaneously, so it cannot be gamed by the
    majority class the way accuracy/F1 can.
    """
    yt, yp = _finite_binary(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    try:
        from sklearn.metrics import matthews_corrcoef as _mcc  # noqa: PLC0415

        val = float(_mcc(yt, yp))
        return val if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of sensitivity and specificity — the imbalance-corrected accuracy."""
    yt, yp = _finite_binary(y_true, y_pred)
    if yt.size == 0 or np.unique(yt).size < 2:
        return float("nan")
    try:
        from sklearn.metrics import balanced_accuracy_score  # noqa: PLC0415

        val = float(balanced_accuracy_score(yt, yp))
        return val if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True-negative rate TN / (TN + FP) — the recall of the *negative* class."""
    yt, yp = _finite_binary(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    denom = tn + fp
    return float(tn / denom) if denom else float("nan")


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve. NaN unless both classes are present."""
    yt = np.asarray(y_true).astype(int).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    n = min(len(yt), len(ys))
    yt, ys = yt[:n], ys[:n]
    good = np.isfinite(ys)
    yt, ys = yt[good], ys[good]
    if yt.size == 0 or np.unique(yt).size < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415

        val = float(roc_auc_score(yt, ys))
        return val if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


# ── Biological-readout agreement (predicted vs. observed measures) ──────────


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; NaN when <3 finite pairs or either side is constant."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    n = min(len(xa), len(ya))
    xa, ya = xa[:n], ya[:n]
    good = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[good], ya[good]
    if xa.size < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        r = float(np.corrcoef(xa, ya)[0, 1])
    return r if np.isfinite(r) else float("nan")


def concordance_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient between two measures.

    Unlike Pearson r (which rewards any linear relation), CCC also penalises
    location/scale shift, so it directly measures agreement with the *identity*
    line — exactly the "does the automated measure equal the manual one" question
    that behavior-scoring validations pose (Lin 1989).
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    n = min(len(xa), len(ya))
    xa, ya = xa[:n], ya[:n]
    good = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[good], ya[good]
    if xa.size < 3:
        return float("nan")
    mx, my = float(np.mean(xa)), float(np.mean(ya))
    vx, vy = float(np.var(xa)), float(np.var(ya))
    cov = float(np.mean((xa - mx) * (ya - my)))
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return float("nan")
    return float(2.0 * cov / denom)


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination = Pearson r squared (NaN-safe)."""
    r = pearson_r(x, y)
    return float(r * r) if np.isfinite(r) else float("nan")


@dataclass
class BlandAltman:
    """Bland-Altman agreement of ``pred`` against ``true`` (difference stats)."""

    bias: float = float("nan")          # mean(pred − true)
    sd_diff: float = float("nan")       # sd of the differences
    loa_lower: float = float("nan")     # bias − 1.96·sd
    loa_upper: float = float("nan")     # bias + 1.96·sd
    n: int = 0

    def bias_ci95(self) -> tuple[float, float]:
        """95% CI of the mean bias. Publication acceptance often asks that this
        interval include zero (no systematic over/under-scoring).

        Uses the t multiplier: this is a CI of a *mean*, at the small session
        counts a held-out split yields. (The limits of agreement above keep 1.96 —
        those are a population spread, ±1.96·SD, not a CI, so t does not apply.)
        """
        if self.n < 2 or not np.isfinite(self.sd_diff):
            return (float("nan"), float("nan"))
        sem = self.sd_diff / np.sqrt(self.n)
        t = t_critical_95(self.n)
        return (self.bias - t * sem, self.bias + t * sem)


def bland_altman(true: np.ndarray, pred: np.ndarray) -> BlandAltman:
    """Difference statistics of ``pred`` vs ``true`` (both same units)."""
    ta = np.asarray(true, dtype=float).ravel()
    pa = np.asarray(pred, dtype=float).ravel()
    n = min(len(ta), len(pa))
    ta, pa = ta[:n], pa[:n]
    good = np.isfinite(ta) & np.isfinite(pa)
    ta, pa = ta[good], pa[good]
    if ta.size == 0:
        return BlandAltman()
    diff = pa - ta
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    return BlandAltman(
        bias=bias, sd_diff=sd,
        loa_lower=bias - 1.96 * sd, loa_upper=bias + 1.96 * sd,
        n=int(diff.size),
    )


def mean_absolute_error(true: np.ndarray, pred: np.ndarray) -> float:
    ta = np.asarray(true, dtype=float).ravel()
    pa = np.asarray(pred, dtype=float).ravel()
    n = min(len(ta), len(pa))
    ta, pa = ta[:n], pa[:n]
    good = np.isfinite(ta) & np.isfinite(pa)
    ta, pa = ta[good], pa[good]
    if ta.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pa - ta)))


# ── Probability calibration ────────────────────────────────────────────────


@dataclass
class CalibrationCurve:
    """Reliability-diagram data + summary calibration errors.

    ``bin_center`` is the nominal midpoint of each populated bin, so a plot can lay
    the bins on a fixed ``[0,1]`` grid (comparable across behaviors) instead of
    inferring positions from ``bin_confidence``. ``bin_count`` matters as much as
    the values: behavior-model scores are strongly bimodal, so the middle bins can
    hold a handful of samples each — connecting them as if they were equal-weight
    points draws a violent zigzag that misrepresents a well-calibrated model.
    """

    bin_confidence: list[float] = field(default_factory=list)  # mean predicted prob per bin
    bin_accuracy: list[float] = field(default_factory=list)    # empirical positive rate per bin
    bin_count: list[int] = field(default_factory=list)         # samples per bin
    bin_center: list[float] = field(default_factory=list)      # nominal midpoint of the bin
    n_bins: int = 10
    ece: float = float("nan")     # expected calibration error (sample-weighted gap)
    mce: float = float("nan")     # maximum calibration error (worst bin gap)
    brier: float = float("nan")   # Brier score = mean((prob − y)^2)
    n: int = 0


def calibration_curve(
    y_true: np.ndarray, y_score: np.ndarray, *, n_bins: int = 10,
) -> CalibrationCurve:
    """Reliability curve + ECE/MCE/Brier for binary probabilities.

    Uses fixed-width ``[0,1]`` bins (the standard ECE definition). Empty bins are
    skipped so the curve only carries populated points. Returns an empty
    :class:`CalibrationCurve` when there is nothing scorable.
    """
    yt = np.asarray(y_true).astype(int).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    n = min(len(yt), len(ys))
    yt, ys = yt[:n], ys[:n]
    good = np.isfinite(ys)
    yt, ys = yt[good], np.clip(ys[good], 0.0, 1.0)
    out = CalibrationCurve(n=int(yt.size), n_bins=int(n_bins))
    if yt.size == 0:
        return out

    out.brier = float(np.mean((ys - yt) ** 2))

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # Right-closed bins; the top bin includes prob == 1.0.
    idx = np.clip(np.digitize(ys, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        sel = idx == b
        cnt = int(np.sum(sel))
        if cnt == 0:
            continue
        conf = float(np.mean(ys[sel]))
        acc = float(np.mean(yt[sel]))
        gap = abs(acc - conf)
        ece += (cnt / yt.size) * gap
        mce = max(mce, gap)
        out.bin_confidence.append(conf)
        out.bin_accuracy.append(acc)
        out.bin_count.append(cnt)
        out.bin_center.append(float((edges[b] + edges[b + 1]) / 2.0))
    out.ece = float(ece)
    out.mce = float(mce)
    return out
