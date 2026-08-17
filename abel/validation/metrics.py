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


# ── Clustered inference (observations nested in projects) ──────────────────


@dataclass
class ClusteredMeanTest:
    """Mean of a clustered sample, tested at the level where units are independent.

    Every field is reported rather than just the headline p, because the whole
    point of this test is that the naive number and the honest number differ and a
    reader must be able to see both.
    """

    estimate: float = float("nan")      # GLS mean gain (random-intercept weighted)
    se: float = float("nan")
    ci95: float = float("nan")          # half-width, t(df) based
    t_stat: float = float("nan")
    df: float = float("nan")            # Satterthwaite, or k−1 when unestimable
    p_value: float = float("nan")       # the primary: mixed model, df-corrected
    icc: float = float("nan")           # ICC(1) — between-cluster share of variance
    n_obs: int = 0
    n_clusters: int = 0
    design_effect: float = float("nan")
    n_effective: float = float("nan")
    p_naive: float = float("nan")       # one-sample t over all obs (clustering ignored)
    p_cluster_mean: float = float("nan")  # t over cluster means (fully aggregated)
    p_sign_flip: float = float("nan")   # exact cluster sign-flip randomisation test
    n_clusters_positive: int = 0
    df_method: str = ""                 # "satterthwaite" | "between-within"


def _reml_profile(y: np.ndarray, idx: np.ndarray, ns: np.ndarray, lam: float) -> float:
    """−2 REML log-likelihood of the intercept-only random-intercept model.

    Profiled over the residual variance, leaving one parameter: ``lam`` = τ²/σ².
    That parameterisation is what makes the fit robust at τ² = 0 — the boundary a
    log-scale optimiser cannot reach and where roughly a third of these rungs sit.
    """
    denom = 1.0 + ns * lam
    s = np.array([y[idx == j].sum() for j in range(len(ns))], dtype=float)
    yy = float(y @ y)
    sxx = float(np.sum(ns / denom))          # X'H⁻¹X
    sxy = float(np.sum(s / denom))           # X'H⁻¹y
    if sxx <= 0:
        return float("inf")
    mu = sxy / sxx
    yhy = yy - float(np.sum(lam * s * s / denom))
    q = yhy - mu * mu * sxx
    n = int(y.size)
    if q <= 0 or n <= 1:
        return float("inf")
    logdet_h = float(np.sum(np.log(denom)))
    return (n - 1) * np.log(q / (n - 1)) + logdet_h + np.log(sxx)


def _var_mu(tau2: float, sigma2: float, ns: np.ndarray) -> float:
    """Var(μ̂) for the random-intercept GLS mean, given the variance components."""
    w = 1.0 / (tau2 + sigma2 / ns)
    return float(1.0 / np.sum(w))


def _sign_flip_p(cluster_means: np.ndarray) -> float:
    """Exact two-sided cluster sign-flip randomisation p (Rademacher, enumerated).

    The assumption-light companion to the t-test: it asks only whether the observed
    arrangement of cluster means is extreme among all 2^k sign assignments, so a
    heavy-tailed cluster cannot sink it the way it inflates a t-test's SD.  With 8
    projects the floor is 2/256 = 0.0078, which is resolution enough to matter.
    Returns NaN above 22 clusters, where enumeration stops being free (and where the
    t-test no longer needs a companion anyway).
    """
    m = np.asarray([v for v in cluster_means if np.isfinite(v)], dtype=float)
    k = m.size
    if k < 2 or k > 22:
        return float("nan")
    sd = float(np.std(m, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12 * max(1.0, float(np.abs(m).max())):
        return float("nan")
    signs = ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1) * 2 - 1
    flipped = signs * m
    sds = flipped.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_all = flipped.mean(axis=1) / (sds / np.sqrt(k))
    t_obs = float(np.mean(m) / (sd / np.sqrt(k)))
    good = np.isfinite(t_all)
    if not good.any():
        return float("nan")
    return float(np.mean(np.abs(t_all[good]) >= abs(t_obs) - 1e-12))


def clustered_mean_test(values, clusters) -> ClusteredMeanTest:
    """Test whether a clustered sample's mean differs from zero, honestly.

    The suite's pooled tests average a per-observation gain (one per behavior, one
    per pair) and then run a one-sample t-test over them.  That treats the
    observations as independent, and they are not: behaviors within a project share
    animals, sessions, the holdout split and the negative pool.  Measured on the
    manuscript runs the intra-class correlation is ~0.3-0.4, which inflates the
    naive t by enough to move a headline p from 0.054 to 0.0002 — two orders of
    magnitude of significance manufactured purely by counting the same subjects
    repeatedly.

    So this fits an intercept-only **random-intercept model** (cluster = project) by
    REML and tests the GLS mean.  That keeps every observation's resolution and its
    contribution to the estimate, unlike collapsing to cluster means, while pricing
    the shared variance into the standard error.

    **The denominator df is the part that decides the answer, so it is not left to a
    default.**  A Wald z (df = ∞, what ``statsmodels`` MixedLM reports) is markedly
    anti-conservative at the ~8 clusters this suite has: on the manuscript ablation
    it returns p = 0.018 where the Satterthwaite df of 7.3 gives p = 0.049.  Since
    the fixed effect here is a pure *between-cluster* contrast, its df can never
    really exceed k − 1 by much, and the z-test's implicit claim of infinite df is
    the entire difference.  We therefore compute Satterthwaite df from the REML
    information matrix and fall back to between-within (k − 1, SAS ``ddfm=betwithin``)
    whenever τ̂² sits on the zero boundary and the information matrix is singular —
    the conservative direction, chosen deliberately because an unestimable clustering
    term is not evidence of no clustering.

    ``p_naive``, ``p_cluster_mean`` and ``p_sign_flip`` are carried alongside so the
    sensitivity of the conclusion to the inferential model is visible in the exported
    table rather than being a choice made silently here.
    """
    vals = np.asarray(values, dtype=float)
    keys = np.asarray(clusters)
    good = np.isfinite(vals)
    vals, keys = vals[good], keys[good]
    out = ClusteredMeanTest(n_obs=int(vals.size))
    if vals.size == 0:
        return out

    uniq, idx = np.unique(keys, return_inverse=True)
    k = int(uniq.size)
    n = int(vals.size)
    ns = np.array([int(np.sum(idx == j)) for j in range(k)], dtype=float)
    cluster_means = np.array([float(vals[idx == j].mean()) for j in range(k)])
    out.n_clusters = k
    out.n_clusters_positive = int(np.sum(cluster_means > 0))
    out.p_naive = paired_p(vals)
    out.p_cluster_mean = paired_p(cluster_means)
    out.p_sign_flip = _sign_flip_p(cluster_means)

    # Same zero-variance trap :func:`paired_p` guards: a constant sample leaves ~1e-18
    # of float dust, which divides into a colossal t and a p of ~1e-16 manufactured out
    # of no variance at all.  A tolerance, not ``== 0``.
    spread = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    if n < 2 or not np.isfinite(spread) or spread <= 1e-12 * max(1.0, float(np.abs(vals).max())):
        out.estimate = float(np.mean(vals))
        out.p_value = float("nan")
        out.df_method = "zero-variance"
        return out

    # A single cluster carries no between-cluster information; there is nothing to
    # correct and nothing honest to say beyond the naive test.
    if k < 2 or n <= k:
        out.estimate = float(np.mean(vals))
        out.p_value = out.p_naive
        out.df_method = "unclustered"
        return out

    try:
        from scipy import optimize, stats  # noqa: PLC0415
    except ImportError:
        out.estimate = float(np.mean(cluster_means))
        return out

    # REML fit over lam = tau^2/sigma^2, profiled; bounded so tau^2 = 0 is reachable.
    res = optimize.minimize_scalar(
        lambda lo: _reml_profile(vals, idx, ns, float(np.expm1(lo))),
        bounds=(0.0, 20.0), method="bounded", options={"xatol": 1e-10},
    )
    lam = float(np.expm1(float(res.x))) if res.success else 0.0
    denom = 1.0 + ns * lam
    s = np.array([vals[idx == j].sum() for j in range(k)], dtype=float)
    sxx = float(np.sum(ns / denom))
    mu = float(np.sum(s / denom) / sxx)
    q = float(vals @ vals) - float(np.sum(lam * s * s / denom)) - mu * mu * sxx
    sigma2 = max(q / (n - 1), 0.0)
    tau2 = lam * sigma2

    var_mu = _var_mu(tau2, sigma2, ns) if (tau2 + sigma2) > 0 else float("nan")
    se = float(np.sqrt(var_mu)) if np.isfinite(var_mu) and var_mu > 0 else float("nan")

    # ICC(1) and the design effect it implies — the plain-language version of why
    # the naive n is not the real n.
    icc = float(tau2 / (tau2 + sigma2)) if (tau2 + sigma2) > 0 else 0.0
    nbar = float(n / k)
    deff = 1.0 + (nbar - 1.0) * icc
    out.icc = icc
    out.design_effect = float(deff)
    out.n_effective = float(n / deff) if deff > 0 else float(n)

    df = float(k - 1)
    method = "between-within"
    if tau2 > 0:
        satt = _satterthwaite_df(vals, idx, ns, tau2, sigma2)
        if np.isfinite(satt) and satt > 0:
            # Cannot exceed the total df; never floored up to k−1, since an
            # unbalanced design can legitimately land just below it.
            df = float(min(satt, n - 1))
            method = "satterthwaite"

    out.estimate = mu
    out.se = se
    out.df = df
    out.df_method = method
    if np.isfinite(se) and se > 0:
        out.t_stat = float(mu / se)
        out.p_value = float(2.0 * stats.t.sf(abs(out.t_stat), df))
        out.ci95 = float(stats.t.ppf(0.975, df) * se)
    return out


def _satterthwaite_df(y: np.ndarray, idx: np.ndarray, ns: np.ndarray,
                      tau2: float, sigma2: float) -> float:
    """Satterthwaite denominator df for the GLS mean: 2·V² / (g' Cov(θ̂) g).

    ``Cov(θ̂)`` is the inverse REML information for (τ², σ²), obtained by numerically
    differentiating the REML objective — analytic traces for this model are a page of
    algebra that would buy nothing here, since the objective itself is closed-form and
    cheap.  Returns NaN on a singular information matrix so the caller falls back to
    the conservative between-within df.
    """
    def _obj(t2: float, s2: float) -> float:
        if t2 < 0 or s2 <= 0:
            return float("inf")
        # Re-express in the profiled parameterisation the fit used.
        return _reml_profile_unprofiled(y, idx, ns, t2, s2)

    h_t = max(abs(tau2) * 1e-4, 1e-10)
    h_s = max(abs(sigma2) * 1e-4, 1e-10)
    grad = np.zeros(2)
    grad[0] = (_var_mu(tau2 + h_t, sigma2, ns) - _var_mu(tau2 - h_t, sigma2, ns)) / (2 * h_t)
    grad[1] = (_var_mu(tau2, sigma2 + h_s, ns) - _var_mu(tau2, sigma2 - h_s, ns)) / (2 * h_s)

    steps = (h_t, h_s)
    hess = np.zeros((2, 2))
    base = (tau2, sigma2)
    for i in range(2):
        for j in range(2):
            acc = 0.0
            for si, sj, sign in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
                p = list(base)
                p[i] += si * steps[i]
                p[j] += sj * steps[j]
                v = _obj(p[0], p[1])
                if not np.isfinite(v):
                    return float("nan")
                acc += sign * v
            hess[i, j] = acc / (4 * steps[i] * steps[j])
    hess = 0.5 * hess  # objective is −2·loglik, information is the Hessian of −loglik
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return float("nan")
    var = _var_mu(tau2, sigma2, ns)
    denom = float(grad @ cov @ grad)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(2.0 * var * var / denom)


def _reml_profile_unprofiled(y: np.ndarray, idx: np.ndarray, ns: np.ndarray,
                             tau2: float, sigma2: float) -> float:
    """−2 REML log-likelihood at explicit (τ², σ²) — the un-profiled objective."""
    d = sigma2 + ns * tau2
    if np.any(d <= 0) or sigma2 <= 0:
        return float("inf")
    k, n = len(ns), int(y.size)
    s = np.array([y[idx == j].sum() for j in range(k)], dtype=float)
    yy = float(y @ y)
    sxx = float(np.sum(ns / d))
    if sxx <= 0:
        return float("inf")
    mu = float(np.sum(s / d) / sxx)
    # y'V⁻¹y via Sherman-Morrison on each block.
    yvy = yy / sigma2 - float(np.sum(tau2 * s * s / (sigma2 * d)))
    q = yvy - mu * mu * sxx
    logdet = float(np.sum((ns - 1) * np.log(sigma2) + np.log(d)))
    return logdet + np.log(sxx) + q


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


def benjamini_hochberg_adjust(pvalues) -> list[float]:
    """BH-adjusted p-values (q-values), aligned to the input order.

    The companion to :func:`benjamini_hochberg_threshold`: that returns one cutoff
    for drawing a line on a volcano, this returns a per-test number a table can
    print beside each raw p.  A q of 0.08 says "to call this significant you must
    accept an 8% false-discovery rate in this family", which is far more use to a
    reader than a bare reject/keep at one arbitrary alpha.

    Non-finite inputs stay non-finite in the output and are excluded from ``m``
    rather than counted as tests that happened to fail — a NaN p is a test that
    could not be run (zero variance, too few seeds), and inflating the family with
    them would penalise every other test for its absence.  Enforces the standard
    monotonicity (cumulative minimum from the largest p down) so a q can never come
    back smaller than that of a more significant test.
    """
    arr = np.asarray(list(pvalues), dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    good = np.nonzero(np.isfinite(arr))[0]
    m = good.size
    if m == 0:
        return out.tolist()
    order = good[np.argsort(arr[good], kind="stable")]
    ranked = arr[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out.tolist()


# ── Imbalanced-classification summaries ────────────────────────────────────


def is_degenerate_fit(tp: int, fp: int, fn: int, tn: int) -> bool:
    """True when a fit carries no information about the target, whatever its F1.

    The guardrail that has to exist once F1 is reported target-class rather than
    macro-averaged.  Macro-F1 scored an always-predict-target model correctly as
    broken (its ~0.50 floor was doing double duty as a collapse detector); target
    F1 scores the same model *well* — recall 1.0, specificity 0.0.  Measured across
    the manuscript runs, 438 cells are literal ``tn == 0 & fn == 0`` collapses and
    the worst of them roughly double their reported score (NSF Eat-vs-Freeze:
    macro 0.392, target 0.784).

    The criterion is **MCC ≤ 0**, computed here from the counts rather than read
    from a stored column so it works on any row.  MCC is the imbalance-robust
    summary this module already recommends (Chicco & Jurman 2020) and it is exactly
    zero for *any* constant classifier, at any prevalence.  That last property is
    what makes it the right test and a predicted-positive-fraction threshold the
    wrong one: the obvious rule — "flags ≥50% of the eval set as positive" — is
    correct for a rare-behavior detection curve but catastrophic for pairwise
    discrimination, where A-vs-B is ~50% positive *by construction*.  On the real
    cells that rule flags 1080 of 2000 discrimination fits (median F1 0.971 —
    healthy fits, every one); MCC ≤ 0 flags 9 (median F1 0.383).

    A held-out set that is single-class returns False: nothing can be concluded
    about the fit, and that condition already has its own ``degenerate`` flag.
    """
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    if tp + fp + fn + tn <= 0:
        return False
    if (tp + fn) == 0 or (tn + fp) == 0:
        return False  # truth has one class — a holdout problem, not a fit collapse
    denom = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    if denom <= 0:
        # A zero prediction margin: every row went to one class. MCC is 0 by
        # convention, which is the collapse this function exists to catch.
        return True
    mcc = (float(tp) * tn - float(fp) * fn) / np.sqrt(denom)
    return bool(mcc <= 0.0)


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
