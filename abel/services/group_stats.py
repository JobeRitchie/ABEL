"""Group-comparison statistics for the Analytics tab.

One engine behind every p-value the Analytics tab shows: the Statistics
dialog, the significance brackets drawn on graphs and the per-graph
Statistics report.  The choices mirror GraphPad Prism's analysis dialogs
(experimental design, Gaussian or not, equal SDs or not, then a multiple
comparisons method), so a result can be reproduced in Prism by making the
same choices there.

Every result carries the exact test, the n per group, and the method lines
needed to write it up; nothing is computed that the report does not name.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _st

from abel.services.group_stats_options import POSTHOC_LABELS, StatsOptions  # noqa: F401 (re-export)


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------


@dataclass
class GroupSummary:
    name: str
    n: int
    mean: float
    sd: float
    sem: float
    median: float


@dataclass
class Comparison:
    a: str
    b: str
    diff: float                 # mean(a) - mean(b); rank difference for rank tests
    stat_label: str
    stat: float
    df: float | None
    p_raw: float
    p_adj: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")


@dataclass
class EffectRow:
    source: str
    ss: float
    df_num: float
    df_den: float
    ms: float
    f: float
    p: float
    gg_epsilon: float | None = None


@dataclass
class StatsResult:
    test: str = ""
    stat_label: str = ""
    stat: float = float("nan")
    df: str = ""
    p: float = float("nan")
    groups: list[GroupSummary] = field(default_factory=list)
    effects: list[EffectRow] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    posthoc_method: str = ""
    method_lines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str = ""
    alpha: float = 0.05

    @property
    def ok(self) -> bool:
        return not self.error and (np.isfinite(self.p) or bool(self.effects))

    def significant_comparisons(self) -> list[Comparison]:
        return [c for c in self.comparisons if np.isfinite(c.p_adj) and c.p_adj < self.alpha]

    def headline(self) -> str:
        """One-line summary for a graph corner label."""
        if self.error:
            return ""
        if self.effects:
            return "; ".join(
                f"{e.source}: {p_stars(e.p)} {format_p(e.p)}" for e in self.effects
            )
        return f"{self.test}: {format_p(self.p)} {p_stars(self.p)}"

    def report(self, title: str = "") -> str:
        lines: list[str] = []
        if title:
            lines += [title, "=" * min(78, max(len(title), 20))]
        if self.error:
            lines.append(f"Not tested: {self.error}")
            lines += [f"Note: {n}" for n in self.notes]
            return "\n".join(lines)
        lines.append(f"Test: {self.test}")
        if self.groups:
            lines.append("")
            lines.append(f"{'Group':<28s} {'n':>4s} {'Mean':>11s} {'SD':>11s} {'SEM':>11s} {'Median':>11s}")
            for g in self.groups:
                lines.append(
                    f"{g.name[:28]:<28s} {g.n:4d} {_f(g.mean):>11s} {_f(g.sd):>11s} "
                    f"{_f(g.sem):>11s} {_f(g.median):>11s}"
                )
        lines.append("")
        if self.effects:
            lines.append(f"{'Source':<30s} {'SS':>11s} {'DFn':>7s} {'DFd':>7s} {'F':>9s} {'P':>10s}")
            for e in self.effects:
                lines.append(
                    f"{e.source[:30]:<30s} {_f(e.ss):>11s} {_dfs(e.df_num):>7s} "
                    f"{_dfs(e.df_den):>7s} {_f(e.f):>9s} {format_p(e.p, bare=True):>10s} {p_stars(e.p)}"
                )
                if e.gg_epsilon is not None:
                    lines.append(f"{'':<30s} Geisser-Greenhouse epsilon = {e.gg_epsilon:.4f}")
        else:
            df_txt = f" ({self.df})" if self.df else ""
            lines.append(
                f"{self.stat_label}{df_txt} = {_f(self.stat)}, {format_p(self.p)}  {p_stars(self.p)}"
            )
        if self.comparisons and not self.posthoc_method:
            c = self.comparisons[0]
            what = "Difference between medians" if c.stat_label in ("U", "W") else "Difference between means"
            ci = (
                f", 95% CI {_f(c.ci_low)} to {_f(c.ci_high)}"
                if np.isfinite(c.ci_low) and np.isfinite(c.ci_high) else ""
            )
            lines.append(f"{what} ({c.a} minus {c.b}): {_f(c.diff)}{ci}")
        elif self.comparisons:
            lines.append("")
            lines.append(f"Multiple comparisons: {self.posthoc_method}")
            lines.append(
                f"  {'Comparison':<40s} {'Diff':>10s} {'95% CI of diff':>24s} "
                f"{'P (raw)':>10s} {'P (adj)':>10s}"
            )
            for c in self.comparisons:
                ci = (
                    f"{_f(c.ci_low)} to {_f(c.ci_high)}"
                    if np.isfinite(c.ci_low) and np.isfinite(c.ci_high) else ""
                )
                lines.append(
                    f"  {(c.a + ' vs ' + c.b)[:40]:<40s} {_f(c.diff):>10s} {ci:>24s} "
                    f"{format_p(c.p_raw, bare=True):>10s} {format_p(c.p_adj, bare=True):>10s} "
                    f"{p_stars(c.p_adj)}"
                )
        elif self.posthoc_method:
            lines.append("")
            lines.append(f"Multiple comparisons: {self.posthoc_method}")
        if self.method_lines:
            lines.append("")
            lines.append("Method:")
            lines += [f"  {m}" for m in self.method_lines]
        if self.notes:
            lines.append("")
            lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def _f(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    ax = abs(x)
    if ax != 0 and (ax >= 1e5 or ax < 1e-3):
        return f"{x:.3g}"
    return f"{x:.4g}"


def _dfs(d: float) -> str:
    if d is None or not np.isfinite(d):
        return "-"
    return f"{d:.0f}" if abs(d - round(d)) < 1e-9 else f"{d:.2f}"


def format_p(p: float, bare: bool = False) -> str:
    """Prism-style P value text: "P<0.0001", "P=0.0123"."""
    if p is None or not np.isfinite(p):
        return "-" if bare else "P=-"
    if p < 0.0001:
        return "<0.0001" if bare else "P<0.0001"
    txt = f"{p:.4f}"
    return txt if bare else f"P={txt}"


def p_stars(p: float) -> str:
    """Prism's default GP style: ns, *, **, ***, ****."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def ci95_multiplier(n: Any) -> Any:
    """t(0.975, n-1): the SEM multiplier for a 95% CI of a mean of *n* values.

    Vectorized; 0 where n < 2 (no interval).  1.96 is only right for large n.
    """
    arr = np.asarray(n, dtype=float)
    out = np.where(arr >= 2, _st.t.ppf(0.975, np.maximum(arr - 1, 1)), 0.0)
    return float(out) if np.ndim(out) == 0 else out


# ----------------------------------------------------------------------
# Multiplicity adjustments
# ----------------------------------------------------------------------


def _adjust(p: np.ndarray, method: str) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    if method in ("sidak", "dunnett_approx"):
        return 1.0 - (1.0 - p) ** m
    if method == "holm_sidak":
        order = np.argsort(p)
        adj = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(order):
            val = 1.0 - (1.0 - p[idx]) ** (m - rank)
            running = max(running, val)
            adj[idx] = min(running, 1.0)
        return adj
    return p.copy()  # fisher / none


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _summaries(arrays: dict[str, np.ndarray]) -> list[GroupSummary]:
    out = []
    for name, a in arrays.items():
        n = len(a)
        sd = float(np.std(a, ddof=1)) if n > 1 else float("nan")
        out.append(GroupSummary(
            name=name, n=n,
            mean=float(np.mean(a)) if n else float("nan"),
            sd=sd,
            sem=sd / np.sqrt(n) if n > 1 else float("nan"),
            median=float(np.median(a)) if n else float("nan"),
        ))
    return out


def _pairs(names: list[str], posthoc: str, control: str) -> list[tuple[int, int]]:
    if posthoc == "dunnett":
        ci = names.index(control) if control in names else 0
        return [(i, ci) for i in range(len(names)) if i != ci]
    return list(combinations(range(len(names)), 2))


def _clean(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    return a[np.isfinite(a)]


# ----------------------------------------------------------------------
# One factor
# ----------------------------------------------------------------------


def one_factor(
    data: pd.DataFrame,
    order: Sequence[str] | None = None,
    opts: StatsOptions | None = None,
) -> StatsResult:
    """Compare the groups in ``data`` (columns ``value``, ``group``, optional ``subject``).

    Two groups get a t-test (or its rank / paired analog); three or more get
    a one-way ANOVA (or its analog) and the chosen multiple comparisons.
    """
    opts = opts or StatsOptions()
    res = StatsResult(alpha=opts.alpha)
    if data is None or data.empty:
        res.error = "no data."
        return res
    df = data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    names = [g for g in (order or sorted(df["group"].unique())) if g in set(df["group"])]
    n_nan = int((~np.isfinite(df["value"])).sum())
    if n_nan:
        res.notes.append(
            f"{n_nan} value(s) are undefined (e.g. mean bout duration of an animal "
            "with no bouts) and were left out, as in Prism's blank cells."
        )
    df = df[np.isfinite(df["value"])]
    if len(names) < 2:
        res.error = "need at least two groups."
        return res
    if opts.design == "paired":
        return _one_factor_paired(df, names, opts, res)
    arrays = {g: df.loc[df["group"] == g, "value"].to_numpy(float) for g in names}
    res.groups = _summaries(arrays)
    small = [g for g, a in arrays.items() if len(a) < 2]
    if small:
        res.error = (
            "every group needs at least 2 values; "
            + ", ".join(f"{g} has n={len(arrays[g])}" for g in small) + "."
        )
        return res
    if all(np.ptp(a) == 0 for a in arrays.values()):
        res.error = (
            "every group's values are identical within the group (no variability), "
            "so no test statistic is defined."
        )
        return res
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if len(names) == 2:
            _two_unpaired(arrays, names, opts, res)
        else:
            _k_unpaired(arrays, names, opts, res)
    return res


def _two_unpaired(arrays, names, opts, res) -> None:
    a, b = arrays[names[0]], arrays[names[1]]
    if not opts.gaussian:
        r = _st.mannwhitneyu(a, b, alternative="two-sided", method="auto")
        u = float(min(r.statistic, len(a) * len(b) - r.statistic))
        res.test, res.stat_label, res.stat, res.p = "Mann-Whitney test", "U", u, float(r.pvalue)
        exact = len(a) <= 8 or len(b) <= 8
        ties = len(np.unique(np.concatenate([a, b]))) < len(a) + len(b)
        res.method_lines.append(
            "Two-tailed Mann-Whitney U test (scipy.stats.mannwhitneyu); "
            + ("exact P." if exact and not ties else "normal approximation with tie and continuity correction.")
        )
        res.comparisons.append(Comparison(
            names[0], names[1], float(np.median(a) - np.median(b)), "U", u, None,
            res.p, res.p,
        ))
        return
    welch = not opts.equal_sd
    r = _st.ttest_ind(a, b, equal_var=not welch)
    va, vb, na, nb = np.var(a, ddof=1), np.var(b, ddof=1), len(a), len(b)
    if welch:
        se = np.sqrt(va / na + vb / nb)
        dof = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        res.test = "Unpaired t test with Welch's correction"
    else:
        sp = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
        se = np.sqrt(sp * (1 / na + 1 / nb))
        dof = na + nb - 2
        res.test = "Unpaired t test"
    diff = float(np.mean(a) - np.mean(b))
    tq = _st.t.ppf(0.975, dof)
    res.stat_label, res.stat, res.p = "t", float(r.statistic), float(r.pvalue)
    res.df = f"df={_dfs(dof)}"
    res.comparisons.append(Comparison(
        names[0], names[1], diff, "t", float(r.statistic), float(dof),
        res.p, res.p, diff - tq * se, diff + tq * se,
    ))
    res.method_lines.append(
        "Two-tailed " + ("Welch's t-test (unequal SDs)" if welch else "Student's t-test (pooled SD)")
        + " on per-animal values."
    )
    if not welch:
        f_stat = max(va, vb) / min(va, vb) if min(va, vb) > 0 else float("inf")
        if np.isfinite(f_stat):
            dfn = (na - 1) if va >= vb else (nb - 1)
            dfd = (nb - 1) if va >= vb else (na - 1)
            p_f = min(1.0, 2 * float(_st.f.sf(f_stat, dfn, dfd)))
            if p_f < 0.05:
                res.notes.append(
                    f"The SDs differ (F test P={p_f:.4f}). Consider Welch's correction "
                    "(Statistics options: 'Assume equal SDs' off)."
                )


def _k_unpaired(arrays, names, opts, res) -> None:
    k = len(names)
    N = sum(len(a) for a in arrays.values())
    seq = [arrays[g] for g in names]
    if not opts.gaussian:
        r = _st.kruskal(*seq)
        res.test, res.stat_label, res.stat, res.p = "Kruskal-Wallis test", "H", float(r.statistic), float(r.pvalue)
        res.df = f"df={k - 1}"
        res.method_lines.append("Kruskal-Wallis H test with tie correction (scipy.stats.kruskal).")
        _dunn_unpaired(arrays, names, opts, res)
        return
    if opts.equal_sd:
        r = _st.f_oneway(*seq)
        res.test, res.stat_label, res.stat, res.p = "Ordinary one-way ANOVA", "F", float(r.statistic), float(r.pvalue)
        res.df = f"{k - 1}, {N - k}"
        res.method_lines.append("Ordinary one-way ANOVA (scipy.stats.f_oneway).")
        mse = sum(((a - a.mean()) ** 2).sum() for a in seq) / (N - k)
        _posthoc_pooled(arrays, names, mse, N - k, opts, res)
    else:
        f_val, d1, d2, p = welch_anova(seq)
        res.test, res.stat_label, res.stat, res.p = "Welch's ANOVA", "W", f_val, p
        res.df = f"{_dfs(d1)}, {_dfs(d2)}"
        res.method_lines.append("Welch's one-way ANOVA (unequal SDs).")
        _posthoc_welch(arrays, names, opts, res)


def welch_anova(groups: Sequence[np.ndarray]) -> tuple[float, float, float, float]:
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])
    w = n / v
    W = w.sum()
    mw = (w * m).sum() / W
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    tmp = ((1 - w / W) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    f_val = a / b
    d2 = (k ** 2 - 1) / (3 * tmp)
    return float(f_val), float(k - 1), float(d2), float(_st.f.sf(f_val, k - 1, d2))


def _gate(opts: StatsOptions, res: StatsResult) -> bool:
    if opts.posthoc == "none":
        return False
    if opts.posthoc_only_if_significant and not (np.isfinite(res.p) and res.p < opts.alpha):
        res.posthoc_method = "not run (omnibus P is not below alpha; see Statistics options)"
        return False
    return True


def _posthoc_pooled(arrays, names, mse, dfe, opts, res) -> None:
    if not _gate(opts, res):
        return
    method = opts.posthoc
    k = len(names)
    means = {g: float(np.mean(arrays[g])) for g in names}
    ns = {g: len(arrays[g]) for g in names}
    pairs = _pairs(names, method, opts.control_group)
    if method == "tukey":
        tk = _st.tukey_hsd(*[arrays[g] for g in names])
        ci = tk.confidence_interval(0.95)
        for i, j in pairs:
            se = np.sqrt(mse / 2 * (1 / ns[names[i]] + 1 / ns[names[j]]))
            res.comparisons.append(Comparison(
                names[i], names[j], means[names[i]] - means[names[j]], "q",
                abs(means[names[i]] - means[names[j]]) / se, dfe,
                float("nan"), float(tk.pvalue[i, j]),
                float(ci.low[i, j]), float(ci.high[i, j]),
            ))
        res.posthoc_method = "Tukey's multiple comparisons test (Tukey-Kramer for unequal n; pooled SD)"
        return
    if method == "dunnett":
        ci_idx = names.index(opts.control_group) if opts.control_group in names else 0
        others = [i for i in range(k) if i != ci_idx]
        samples = [arrays[names[i]] for i in others]
        try:  # scipy >= 1.15 names the seed "rng"; the P is a seeded QMC integral
            dn = _st.dunnett(*samples, control=arrays[names[ci_idx]], rng=np.random.default_rng(0))
        except TypeError:
            dn = _st.dunnett(*samples, control=arrays[names[ci_idx]],
                             random_state=np.random.default_rng(0))
        dci = dn.confidence_interval(0.95)
        for pos, i in enumerate(others):
            d = means[names[i]] - means[names[ci_idx]]
            se = np.sqrt(mse * (1 / ns[names[i]] + 1 / ns[names[ci_idx]]))
            res.comparisons.append(Comparison(
                names[i], names[ci_idx], d, "q", abs(d) / se, dfe,
                float("nan"), float(dn.pvalue[pos]),
                float(dci.low[pos]), float(dci.high[pos]),
            ))
        res.posthoc_method = f"Dunnett's multiple comparisons test (control: {names[ci_idx]}; pooled SD)"
        return
    raw, rows = [], []
    for i, j in pairs:
        d = means[names[i]] - means[names[j]]
        se = np.sqrt(mse * (1 / ns[names[i]] + 1 / ns[names[j]]))
        t = d / se
        raw.append(2 * float(_st.t.sf(abs(t), dfe)))
        rows.append((names[i], names[j], d, t, se))
    _emit(rows, raw, dfe, method, res, "pooled SD from the ANOVA")


def _emit(rows, raw, dfe, method, res, basis: str) -> None:
    adj = _adjust(np.array(raw), method)
    m = len(raw)
    # Simultaneous CIs matching the adjustment (Bonferroni/Sidak) or plain 95%.
    if method == "bonferroni":
        conf = 1 - 0.05 / m
    elif method in ("sidak", "holm_sidak"):
        conf = (1 - 0.05) ** (1 / m)
    else:
        conf = 0.95
    for (a, b, d, t, se), pr, pa, dof in zip(rows, raw, adj, dfe if isinstance(dfe, list) else [dfe] * m):
        tq = _st.t.ppf(1 - (1 - conf) / 2, dof)
        res.comparisons.append(Comparison(a, b, d, "t", t, dof, pr, float(pa), d - tq * se, d + tq * se))
    res.posthoc_method = f"{POSTHOC_LABELS.get(method, method)} multiple comparisons ({basis})"


def _posthoc_welch(arrays, names, opts, res) -> None:
    if not _gate(opts, res):
        return
    k = len(names)
    method = opts.posthoc
    pairs = _pairs(names, method, opts.control_group)
    rows, raw, dfs = [], [], []
    gh = []
    for i, j in pairs:
        a, b = arrays[names[i]], arrays[names[j]]
        va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
        se = np.sqrt(va + vb)
        dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        d = float(np.mean(a) - np.mean(b))
        t = d / se
        rows.append((names[i], names[j], d, t, se))
        raw.append(2 * float(_st.t.sf(abs(t), dof)))
        dfs.append(float(dof))
        gh.append(float(_st.studentized_range.sf(abs(t) * np.sqrt(2), k, dof)))
    if method == "tukey":
        for (a, b, d, t, se), pr, p_gh, dof in zip(rows, raw, gh, dfs):
            qc = _st.studentized_range.ppf(0.95, k, dof) / np.sqrt(2)
            res.comparisons.append(Comparison(a, b, d, "t", t, dof, float("nan"), p_gh, d - qc * se, d + qc * se))
        res.posthoc_method = "Games-Howell multiple comparisons test (unequal SDs)"
        return
    use = "sidak" if method == "dunnett" else method
    _emit(rows, raw, dfs, use, res, "Welch t-tests, individual SDs")
    if method == "dunnett":
        res.posthoc_method = (
            f"Each vs control ({rows[0][1] if rows else ''}) with Welch t-tests, "
            "Šídák-corrected (Dunnett T3-style; unequal SDs)"
        )


def _dunn_unpaired(arrays, names, opts, res) -> None:
    if not _gate(opts, res):
        return
    allv = np.concatenate([arrays[g] for g in names])
    ranks = _st.rankdata(allv)
    N = len(allv)
    _, counts = np.unique(allv, return_counts=True)
    tie = float(((counts ** 3) - counts).sum())
    mean_rank, start = {}, 0
    for g in names:
        n = len(arrays[g])
        mean_rank[g] = float(ranks[start:start + n].mean())
        start += n
    pairs = _pairs(names, opts.posthoc, opts.control_group)
    raw, rows = [], []
    for i, j in pairs:
        gi, gj = names[i], names[j]
        se = np.sqrt((N * (N + 1) / 12 - tie / (12 * (N - 1))) * (1 / len(arrays[gi]) + 1 / len(arrays[gj])))
        z = (mean_rank[gi] - mean_rank[gj]) / se
        raw.append(2 * float(_st.norm.sf(abs(z))))
        rows.append((gi, gj, mean_rank[gi] - mean_rank[gj], z))
    method = opts.posthoc
    adj_method = {"tukey": "bonferroni", "dunnett": "bonferroni"}.get(method, method)
    adj = _adjust(np.array(raw), adj_method)
    for (a, b, d, z), pr, pa in zip(rows, raw, adj):
        res.comparisons.append(Comparison(a, b, d, "z", z, None, pr, float(pa)))
    label = "Dunn's multiple comparisons test" if adj_method == "bonferroni" else (
        f"Dunn's z tests, {POSTHOC_LABELS.get(adj_method, adj_method)}-adjusted")
    res.posthoc_method = label + " (mean-rank differences, tie-corrected)"


def _one_factor_paired(df, names, opts, res) -> StatsResult:
    if "subject" not in df.columns:
        res.error = "a paired design needs each value's subject."
        return res
    dup = df.groupby(["subject", "group"]).size()
    dup = dup[dup > 1]
    if not dup.empty:
        sub, grp = dup.index[0]
        res.error = (
            f"subject '{sub}' has {int(dup.iloc[0])} values in group '{grp}'. A paired design "
            "needs exactly one value per subject per group; check the factor assignments."
        )
        return res
    wide = df.pivot(index="subject", columns="group", values="value").reindex(columns=names)
    complete = wide.dropna()
    dropped = sorted(map(str, set(wide.index) - set(complete.index)))
    if dropped:
        res.notes.append(
            f"{len(dropped)} subject(s) lack a value in every group and were left out "
            f"of the paired analysis: {', '.join(dropped[:12])}{'...' if len(dropped) > 12 else ''}."
        )
    Y = complete.to_numpy(float)
    n, k = Y.shape
    res.groups = _summaries({g: Y[:, i] for i, g in enumerate(names)})
    if n < 2:
        res.error = f"need at least 2 subjects with a value in every group (have {n})."
        return res
    if k == 2:
        d = Y[:, 0] - Y[:, 1]
        if not opts.gaussian:
            r = _st.wilcoxon(Y[:, 0], Y[:, 1])
            res.test, res.stat_label, res.stat, res.p = (
                "Wilcoxon matched-pairs signed rank test", "W", float(r.statistic), float(r.pvalue))
            res.method_lines.append(
                "Two-tailed Wilcoxon signed-rank test (scipy.stats.wilcoxon; zero differences dropped)."
            )
            res.comparisons.append(Comparison(names[0], names[1], float(np.median(d)), "W",
                                              float(r.statistic), None, res.p, res.p))
            return res
        r = _st.ttest_rel(Y[:, 0], Y[:, 1])
        se = np.std(d, ddof=1) / np.sqrt(n)
        tq = _st.t.ppf(0.975, n - 1)
        res.test, res.stat_label, res.stat, res.p = "Paired t test", "t", float(r.statistic), float(r.pvalue)
        res.df = f"df={n - 1}"
        res.method_lines.append(f"Two-tailed paired t-test on {n} subjects matched by name.")
        res.comparisons.append(Comparison(names[0], names[1], float(d.mean()), "t", float(r.statistic),
                                          n - 1, res.p, res.p, float(d.mean() - tq * se),
                                          float(d.mean() + tq * se)))
        return res
    if not opts.gaussian:
        r = _st.friedmanchisquare(*[Y[:, i] for i in range(k)])
        res.test, res.stat_label, res.stat, res.p = "Friedman test", "Friedman statistic", float(r.statistic), float(r.pvalue)
        res.method_lines.append(f"Friedman test on {n} subjects matched by name.")
        if _gate(opts, res):
            R = np.apply_along_axis(_st.rankdata, 1, Y)
            mr = R.mean(axis=0)
            se = np.sqrt(k * (k + 1) / (6 * n))
            pairs = _pairs(names, opts.posthoc, opts.control_group)
            raw = [2 * float(_st.norm.sf(abs((mr[i] - mr[j]) / se))) for i, j in pairs]
            adj_method = {"tukey": "bonferroni", "dunnett": "bonferroni"}.get(opts.posthoc, opts.posthoc)
            adj = _adjust(np.array(raw), adj_method)
            for (i, j), pr, pa in zip(pairs, raw, adj):
                res.comparisons.append(Comparison(names[i], names[j], float(mr[i] - mr[j]), "z",
                                                  float((mr[i] - mr[j]) / se), None, pr, float(pa)))
            res.posthoc_method = "Dunn's multiple comparisons test (mean-rank differences)"
        return res
    # Repeated-measures one-way ANOVA.
    gm = Y.mean()
    ss_subj = k * ((Y.mean(axis=1) - gm) ** 2).sum()
    ss_cond = n * ((Y.mean(axis=0) - gm) ** 2).sum()
    ss_err = ((Y - gm) ** 2).sum() - ss_subj - ss_cond
    d1, d2 = k - 1, (n - 1) * (k - 1)
    f_val = (ss_cond / d1) / (ss_err / d2) if ss_err > 0 else float("inf")
    eps = gg_epsilon(Y)
    use_eps = eps if opts.geisser_greenhouse else 1.0
    p = float(_st.f.sf(f_val, d1 * use_eps, d2 * use_eps))
    res.test = "Repeated measures one-way ANOVA" + (" (Geisser-Greenhouse corrected)" if opts.geisser_greenhouse else "")
    res.stat_label, res.stat, res.p = "F", float(f_val), p
    res.df = f"{_dfs(d1 * use_eps)}, {_dfs(d2 * use_eps)}"
    res.effects = []
    res.method_lines.append(
        f"Repeated-measures one-way ANOVA on {n} subjects matched by name"
        + (f"; Geisser-Greenhouse epsilon = {eps:.4f}." if opts.geisser_greenhouse else "; sphericity assumed.")
    )
    if _gate(opts, res):
        _posthoc_paired(Y, names, opts, res, ss_err / d2, d2)
    return res


def gg_epsilon(Y: np.ndarray) -> float:
    """Geisser-Greenhouse epsilon from an n-subjects x k-levels matrix."""
    k = Y.shape[1]
    if k < 3 or Y.shape[0] < 2:
        return 1.0
    S = np.cov(Y, rowvar=False, ddof=1)
    H = np.eye(k) - np.ones((k, k)) / k
    Sc = H @ S @ H
    den = (k - 1) * float((Sc * Sc).sum())
    return float(np.clip(np.trace(Sc) ** 2 / den, 1.0 / (k - 1), 1.0)) if den > 0 else 1.0


def _posthoc_paired(Y, names, opts, res, mse_pooled, df_pooled) -> None:
    n, k = Y.shape
    method = opts.posthoc
    pairs = _pairs(names, method, opts.control_group)
    rows, raw, dfs, tuk = [], [], [], []
    for i, j in pairs:
        d = Y[:, i] - Y[:, j]
        if opts.geisser_greenhouse:
            se, dof = float(np.std(d, ddof=1) / np.sqrt(n)), n - 1
        else:
            se, dof = float(np.sqrt(2 * mse_pooled / n)), df_pooled
        t = float(d.mean() / se) if se > 0 else float("inf")
        rows.append((names[i], names[j], float(d.mean()), t, se))
        raw.append(2 * float(_st.t.sf(abs(t), dof)))
        dfs.append(float(dof))
        tuk.append(float(_st.studentized_range.sf(abs(t) * np.sqrt(2), k, dof)))
    basis = (
        "each comparison uses only the paired differences of its two groups (as Prism does with the Geisser-Greenhouse correction)"
        if opts.geisser_greenhouse else "pooled residual error of the RM ANOVA"
    )
    if method == "tukey":
        for (a, b, d, t, se), pr, pt, dof in zip(rows, raw, tuk, dfs):
            qc = _st.studentized_range.ppf(0.95, k, dof) / np.sqrt(2)
            res.comparisons.append(Comparison(a, b, d, "t", t, dof, float("nan"), pt, d - qc * se, d + qc * se))
        res.posthoc_method = f"Tukey's multiple comparisons test ({basis})"
        return
    use = "sidak" if method == "dunnett" else method
    _emit(rows, raw, dfs, use, res, basis)
    if method == "dunnett":
        res.posthoc_method = f"Each vs control, Šídák-corrected paired comparisons ({basis})"


# ----------------------------------------------------------------------
# Two factors
# ----------------------------------------------------------------------


def _sum_coding(levels: list[str], values: np.ndarray) -> np.ndarray:
    """Effect (sum-to-zero) coding: Prism's unweighted-means / Type III basis."""
    L = len(levels)
    out = np.zeros((len(values), L - 1))
    for c, lev in enumerate(levels[:-1]):
        out[values == lev, c] = 1.0
    out[values == levels[-1], :] = -1.0
    return out


def _ssr(X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[1] == 0:
        return float((y ** 2).sum())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


def _type3(y: np.ndarray, blocks: dict[str, np.ndarray]) -> tuple[dict[str, float], float, int]:
    """Type III SS for each named column block (blocks include 'Intercept')."""
    names = list(blocks)
    full = np.hstack([blocks[k] for k in names])
    ssr_full = _ssr(full, y)
    out = {}
    for k in names:
        reduced = np.hstack([blocks[o] for o in names if o != k]) if len(names) > 1 else np.zeros((len(y), 0))
        out[k] = _ssr(reduced, y) - ssr_full
    return out, ssr_full, int(np.linalg.matrix_rank(full))


def two_factor(
    data: pd.DataFrame,
    name_a: str,
    name_b: str,
    levels_a: Sequence[str],
    levels_b: Sequence[str],
    opts: StatsOptions | None = None,
    cell_label: Any = None,
) -> StatsResult:
    """Two-way ANOVA (Type III, as Prism) on columns ``value``, ``a``, ``b`` (+ ``subject``).

    ``opts.repeated_factor`` naming ``name_a`` or ``name_b`` makes it a mixed
    (split-plot) design matched by subject.  ``cell_label(a, b)`` names a cell
    in the comparison table (defaults to "a | b").
    """
    opts = opts or StatsOptions()
    res = StatsResult(alpha=opts.alpha)
    label = cell_label or (lambda a, b: f"{a} | {b}")
    df = data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    n_nan = int((~np.isfinite(df["value"])).sum())
    if n_nan:
        res.notes.append(f"{n_nan} undefined value(s) were left out.")
    df = df[np.isfinite(df["value"])]
    la = [l for l in levels_a if l in set(df["a"])]
    lb = [l for l in levels_b if l in set(df["b"])]
    if len(la) < 2 or len(lb) < 2:
        res.error = f"two-way ANOVA needs at least 2 levels of each factor ({name_a}: {len(la)}, {name_b}: {len(lb)})."
        return res
    if not opts.gaussian:
        res.notes.append("There is no nonparametric two-way ANOVA (Prism has none either); the parametric test was run.")
    cells = {(a, b): df.loc[(df["a"] == a) & (df["b"] == b), "value"].to_numpy(float) for a in la for b in lb}
    res.groups = _summaries({label(a, b): cells[(a, b)] for a in la for b in lb})
    empty = [label(a, b) for (a, b), v in cells.items() if len(v) == 0]
    if empty:
        res.error = "every cell needs at least one value; empty: " + ", ".join(empty) + "."
        return res
    rep = opts.repeated_factor
    if rep and rep in (name_a, name_b):
        if rep == name_a:
            sw = df.rename(columns={"a": "b", "b": "a"})
            out = _mixed(sw, name_b, name_a, lb, la, opts, res, lambda w, b: label(b, w))
            return out
        return _mixed(df, name_a, name_b, la, lb, opts, res, label)
    y = df["value"].to_numpy(float)
    A = _sum_coding(la, df["a"].to_numpy())
    B = _sum_coding(lb, df["b"].to_numpy())
    AB = np.hstack([A[:, [i]] * B[:, [j]] for i in range(A.shape[1]) for j in range(B.shape[1])])
    ss, ssr, rank = _type3(y, {"Intercept": np.ones((len(y), 1)), "A": A, "B": B, "AB": AB})
    dfe = len(y) - rank
    if dfe <= 0 or ssr <= 0:
        res.error = "no residual degrees of freedom (need more than one value in some cells)."
        return res
    mse = ssr / dfe
    res.test = "Ordinary two-way ANOVA (Type III sums of squares)"
    for src, key, d in (
        (f"{name_a} x {name_b}", "AB", (len(la) - 1) * (len(lb) - 1)),
        (name_a, "A", len(la) - 1),
        (name_b, "B", len(lb) - 1),
    ):
        f_val = (ss[key] / d) / mse
        res.effects.append(EffectRow(src, ss[key], d, dfe, ss[key] / d, f_val, float(_st.f.sf(f_val, d, dfe))))
    res.effects.append(EffectRow("Residual", ssr, dfe, float("nan"), mse, float("nan"), float("nan")))
    balanced = len({len(v) for v in cells.values()}) == 1
    res.method_lines.append(
        "Two-way ANOVA with interaction, fit by least squares with sum-to-zero coding "
        "(Type III SS, as Prism uses)" + ("." if not balanced else "; design is balanced, so Types I-III agree.")
    )
    res.p = res.effects[0].p
    if opts.posthoc != "none" and opts.twoway_compare != "none":
        _simple_effects_pooled(cells, la, lb, name_a, name_b, mse, dfe, opts, res, label)
    return res


def _simple_effects_pooled(cells, la, lb, name_a, name_b, mse, dfe, opts, res, label) -> None:
    rows, raw = [], []
    specs = []
    if opts.twoway_compare in ("first", "both"):
        specs += [((a1, b), (a2, b)) for b in lb for a1, a2 in combinations(la, 2)]
    if opts.twoway_compare in ("second", "both"):
        specs += [((a, b1), (a, b2)) for a in la for b1, b2 in combinations(lb, 2)]
    for c1, c2 in specs:
        v1, v2 = cells[c1], cells[c2]
        d = float(v1.mean() - v2.mean())
        se = float(np.sqrt(mse * (1 / len(v1) + 1 / len(v2))))
        t = d / se
        rows.append((label(*c1), label(*c2), d, t, se))
        raw.append(2 * float(_st.t.sf(abs(t), dfe)))
    method = _twoway_method(opts, res)
    _emit(rows, raw, dfe, method, res, "pooled residual SD of the two-way ANOVA; one family for all comparisons")


def _twoway_method(opts, res) -> str:
    m = opts.posthoc
    if m in ("tukey", "dunnett"):
        res.notes.append(
            f"{POSTHOC_LABELS[m]} is not used for two-way simple-effects comparisons here; "
            "Šídák was applied instead."
        )
        return "sidak"
    return m


def _mixed(df, name_a, name_b, la, lb, opts, res, label) -> StatsResult:
    """Split-plot ANOVA: ``a`` between subjects, ``b`` repeated within subject."""
    if "subject" not in df.columns:
        res.error = "a repeated-measures design needs each value's subject."
        return res
    dup = df.groupby(["subject", "b"]).size()
    dup = dup[dup > 1]
    if not dup.empty:
        sub, lev = dup.index[0]
        res.error = (
            f"subject '{sub}' has {int(dup.iloc[0])} values at {name_b}='{lev}'. A repeated "
            "design needs one value per subject per level; check the factor assignments."
        )
        return res
    between = df.groupby("subject")["a"].nunique()
    bad = list(between[between > 1].index)
    if bad:
        res.error = (
            f"subject '{bad[0]}' appears in more than one level of {name_a}, so {name_a} "
            f"cannot be a between-subjects factor. Mark {name_a} as the repeated factor instead."
        )
        return res
    wide = df.pivot(index="subject", columns="b", values="value").reindex(columns=lb)
    complete = wide.dropna()
    dropped = sorted(map(str, set(wide.index) - set(complete.index)))
    if dropped:
        res.notes.append(
            f"{len(dropped)} subject(s) lack a value at every level of {name_b} and were "
            f"left out: {', '.join(dropped[:12])}{'...' if len(dropped) > 12 else ''}."
        )
    group_of = df.groupby("subject")["a"].first().reindex(complete.index).to_numpy()
    Y = complete.to_numpy(float)
    n, kb = Y.shape
    present_a = [a for a in la if (group_of == a).sum() > 0]
    if len(present_a) < 2:
        res.error = f"need complete subjects in at least 2 levels of {name_a}."
        return res
    if n - len(present_a) <= 0:
        res.error = "no residual degrees of freedom between subjects."
        return res
    la = present_a
    res.groups = _summaries({label(a, b): Y[group_of == a, j] for a in la for j, b in enumerate(lb)})
    A = _sum_coding(la, group_of)
    one = np.ones((n, 1))
    # Between subjects: subject means.
    m_s = Y.mean(axis=1)
    ss_b, ssr_b, _ = _type3(m_s, {"Intercept": one, "A": A})
    ss_a = kb * ss_b["A"]
    ss_sa = kb * ssr_b
    d_a, d_sa = len(la) - 1, n - len(la)
    # Within subjects: orthonormal contrasts across the repeated levels.
    C = _orthonormal_contrasts(kb)
    Z = Y @ C
    ss_w = ss_ab = ss_e = 0.0
    resid = np.zeros_like(Z)
    for c in range(Z.shape[1]):
        z = Z[:, c]
        ssc, ssr_c, _ = _type3(z, {"Intercept": one, "A": A})
        ss_w += ssc["Intercept"]
        ss_ab += ssc["A"]
        ss_e += ssr_c
        for a in la:
            mask = group_of == a
            resid[mask, c] = z[mask] - z[mask].mean()
    d_w, d_ab, d_e = kb - 1, (len(la) - 1) * (kb - 1), (n - len(la)) * (kb - 1)
    S = resid.T @ resid / max(n - len(la), 1)
    eps = 1.0
    if kb > 2 and np.trace(S @ S) > 0:
        eps = float(np.clip(np.trace(S) ** 2 / ((kb - 1) * np.trace(S @ S)), 1.0 / (kb - 1), 1.0))
    e = eps if opts.geisser_greenhouse else 1.0
    mse_w = ss_e / d_e if d_e > 0 else float("nan")
    mse_b = ss_sa / d_sa
    res.test = (
        f"Two-way repeated-measures ANOVA, {name_b} within subjects "
        f"({n} subjects matched by name; Type III SS)"
    )

    def _row(src, ss, d, ms_err, dd, eps_used):
        f_val = (ss / d) / ms_err
        return EffectRow(src, ss, d * eps_used, dd * eps_used, ss / d, f_val,
                         float(_st.f.sf(f_val, d * eps_used, dd * eps_used)),
                         eps if (opts.geisser_greenhouse and kb > 2) else None)

    res.effects = [
        _row(f"{name_a} x {name_b}", ss_ab, d_ab, mse_w, d_e, e),
        _row(name_b, ss_w, d_w, mse_w, d_e, e),
        EffectRow(name_a, ss_a, d_a, d_sa, ss_a / d_a, (ss_a / d_a) / mse_b,
                  float(_st.f.sf((ss_a / d_a) / mse_b, d_a, d_sa))),
        EffectRow("Subject (between-subjects error)", ss_sa, d_sa, float("nan"), mse_b, float("nan"), float("nan")),
        EffectRow("Residual (within-subjects error)", ss_e, d_e, float("nan"), mse_w, float("nan"), float("nan")),
    ]
    res.p = res.effects[0].p
    res.method_lines.append(
        f"Mixed (split-plot) ANOVA: {name_a} between subjects, {name_b} repeated within subject; "
        "within-subject effects from orthonormal contrasts (GLM univariate tests)"
        + (f", Geisser-Greenhouse epsilon = {eps:.4f}." if opts.geisser_greenhouse and kb > 2 else ".")
    )
    if opts.posthoc == "none" or opts.twoway_compare == "none":
        return res
    rows, raw, dfs = [], [], []
    if opts.twoway_compare in ("first", "both"):
        # Between-subject comparisons at each repeated level: pooled SD of that level.
        for j, b in enumerate(lb):
            col = Y[:, j]
            mse_j = sum(((col[group_of == a] - col[group_of == a].mean()) ** 2).sum() for a in la) / (n - len(la))
            for a1, a2 in combinations(la, 2):
                v1, v2 = col[group_of == a1], col[group_of == a2]
                d = float(v1.mean() - v2.mean())
                se = float(np.sqrt(mse_j * (1 / len(v1) + 1 / len(v2))))
                t = d / se if se > 0 else float("inf")
                rows.append((label(a1, b), label(a2, b), d, t, se))
                raw.append(2 * float(_st.t.sf(abs(t), n - len(la))))
                dfs.append(float(n - len(la)))
    if opts.twoway_compare in ("second", "both"):
        # Repeated-level comparisons within each group: paired t-tests.
        for a in la:
            sub = Y[group_of == a]
            for j1, j2 in combinations(range(kb), 2):
                d = sub[:, j1] - sub[:, j2]
                if len(d) < 2:
                    continue
                se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
                t = float(d.mean() / se) if se > 0 else float("inf")
                rows.append((label(a, lb[j1]), label(a, lb[j2]), float(d.mean()), t, se))
                raw.append(2 * float(_st.t.sf(abs(t), len(d) - 1)))
                dfs.append(float(len(d) - 1))
    if rows:
        _emit(rows, raw, dfs, _twoway_method(opts, res), res,
              f"between-group tests use the pooled SD at that {name_b} level; "
              f"within-group tests are paired t-tests; one family for all comparisons")
    return res


def _orthonormal_contrasts(k: int) -> np.ndarray:
    """k x (k-1) matrix whose columns are orthonormal and sum to zero."""
    M = np.eye(k) - np.ones((k, k)) / k
    q, _ = np.linalg.qr(M)
    return q[:, : k - 1]
