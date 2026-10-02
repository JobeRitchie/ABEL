"""Reference checks for abel.services.group_stats against independent implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abel.services.group_stats import (
    StatsOptions,
    ci95_multiplier,
    format_p,
    gg_epsilon,
    one_factor,
    p_stars,
    two_factor,
    welch_anova,
)

rng = np.random.default_rng(11)


def _long(groups: dict[str, np.ndarray], subjects: bool = False) -> pd.DataFrame:
    rows = []
    for g, vals in groups.items():
        for i, v in enumerate(vals):
            row = {"group": g, "value": float(v)}
            if subjects:
                row["subject"] = f"m{i}"
            rows.append(row)
    return pd.DataFrame(rows)


def test_student_and_welch_match_scipy():
    a, b = rng.normal(10, 1, 5), rng.normal(13, 5, 9)
    d = _long({"A": a, "B": b})
    r = one_factor(d, ["A", "B"], StatsOptions())
    assert r.test == "Unpaired t test"
    assert r.p == pytest.approx(stats.ttest_ind(a, b).pvalue)
    r = one_factor(d, ["A", "B"], StatsOptions(equal_sd=False))
    assert "Welch" in r.test
    assert r.p == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)


def test_paired_t_matches_subjects_by_name_not_order():
    pre = rng.normal(10, 3, 8)
    post = pre + rng.normal(1.5, 0.5, 8)
    d = _long({"Pre": pre, "Post": post}, subjects=True).sample(frac=1, random_state=3)
    r = one_factor(d, ["Pre", "Post"], StatsOptions(design="paired"))
    assert r.test == "Paired t test"
    assert r.p == pytest.approx(stats.ttest_rel(pre, post).pvalue)


def test_paired_rejects_duplicate_subject_in_group():
    d = pd.DataFrame({"group": ["A", "A", "B"], "subject": ["m1", "m1", "m1"], "value": [1, 2, 3]})
    r = one_factor(d, ["A", "B"], StatsOptions(design="paired"))
    assert r.error and "m1" in r.error


def test_nonparametric_two_group():
    a, b = rng.normal(0, 1, 6), rng.normal(2, 1, 7)
    r = one_factor(_long({"A": a, "B": b}), ["A", "B"], StatsOptions(gaussian=False))
    assert r.p == pytest.approx(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    pre = rng.normal(0, 1, 9)
    post = pre + rng.normal(1, 1, 9)
    r = one_factor(_long({"P": pre, "Q": post}, subjects=True), ["P", "Q"],
                   StatsOptions(gaussian=False, design="paired"))
    assert r.p == pytest.approx(stats.wilcoxon(pre, post).pvalue)


def test_one_way_anova_and_tukey():
    g = {k: rng.normal(m, 2, n) for k, m, n in (("A", 10, 6), ("B", 13, 7), ("C", 11, 5))}
    r = one_factor(_long(g), ["A", "B", "C"], StatsOptions(posthoc="tukey"))
    assert r.p == pytest.approx(stats.f_oneway(*g.values()).pvalue)
    tk = stats.tukey_hsd(*g.values())
    got = {(c.a, c.b): c.p_adj for c in r.comparisons}
    assert got[("A", "B")] == pytest.approx(tk.pvalue[0, 1])
    assert got[("B", "C")] == pytest.approx(tk.pvalue[1, 2])


def test_posthoc_runs_without_significant_omnibus_by_default():
    g = {k: rng.normal(0, 1, 6) for k in "ABC"}
    r = one_factor(_long(g), list("ABC"), StatsOptions(posthoc="sidak"))
    assert len(r.comparisons) == 3
    gated = one_factor(_long(g), list("ABC"),
                       StatsOptions(posthoc="sidak", posthoc_only_if_significant=True))
    if gated.p >= 0.05:
        assert not gated.comparisons


def test_sidak_uses_pooled_sd():
    g = {"A": np.array([1.0, 2, 3, 4]), "B": np.array([3.0, 4, 5, 6]), "C": np.array([0.0, 5, 10, 15])}
    r = one_factor(_long(g), list("ABC"), StatsOptions(posthoc="sidak"))
    mse = sum(((v - v.mean()) ** 2).sum() for v in g.values()) / (12 - 3)
    t = (2.5 - 4.5) / np.sqrt(mse * 0.5)
    p = 2 * stats.t.sf(abs(t), 9)
    ab = next(c for c in r.comparisons if (c.a, c.b) == ("A", "B"))
    assert ab.p_raw == pytest.approx(p)
    assert ab.p_adj == pytest.approx(1 - (1 - p) ** 3)


def test_holm_sidak_step_down():
    from abel.services.group_stats import _adjust

    p = np.array([0.01, 0.04, 0.03])
    adj = _adjust(p, "holm_sidak")
    assert adj[0] == pytest.approx(1 - 0.99 ** 3)
    assert adj[2] == pytest.approx(max(adj[0], 1 - 0.97 ** 2))
    assert adj[1] == pytest.approx(max(adj[2], 0.04))


def test_dunnett_matches_scipy():
    g = {k: rng.normal(m, 2, 6) for k, m in (("Ctrl", 10), ("D1", 12), ("D2", 15))}
    r = one_factor(_long(g), list(g), StatsOptions(posthoc="dunnett", control_group="Ctrl"))
    assert {(c.a, c.b) for c in r.comparisons} == {("D1", "Ctrl"), ("D2", "Ctrl")}
    ref = stats.dunnett(g["D1"], g["D2"], control=g["Ctrl"])
    assert [c.p_adj for c in r.comparisons] == pytest.approx(list(ref.pvalue), abs=2e-3)


def test_welch_anova_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.oneway")
    g = [rng.normal(10, 1, 6), rng.normal(12, 4, 9), rng.normal(11, 8, 7)]
    f_val, d1, d2, p = welch_anova(g)
    ref = sm.anova_oneway(g, use_var="unequal")
    assert f_val == pytest.approx(ref.statistic)
    assert d2 == pytest.approx(ref.df[1])
    assert p == pytest.approx(ref.pvalue)


def test_kruskal_and_dunn():
    g = {"A": np.array([1.0, 2, 3, 4, 5]), "B": np.array([6.0, 7, 8, 9, 10]), "C": np.array([2.5, 3.5, 4.5, 11, 12])}
    r = one_factor(_long(g), list("ABC"), StatsOptions(gaussian=False, posthoc="tukey"))
    assert r.p == pytest.approx(stats.kruskal(*g.values()).pvalue)
    ab = next(c for c in r.comparisons if (c.a, c.b) == ("A", "B"))
    # No ties: z = (mean rank A - mean rank B) / sqrt(N(N+1)/12 * (1/5 + 1/5)); Bonferroni over 3.
    ranks = stats.rankdata(np.concatenate(list(g.values())))
    z = (ranks[:5].mean() - ranks[5:10].mean()) / np.sqrt(15 * 16 / 12 * 0.4)
    assert ab.p_adj == pytest.approx(min(1, 3 * 2 * stats.norm.sf(abs(z))))


def test_rm_anova_matches_statsmodels_when_sphericity_assumed():
    sm = pytest.importorskip("statsmodels.stats.anova")
    n, k = 8, 3
    base = rng.normal(10, 3, n)
    Y = base[:, None] + np.array([0, 1.0, 2.5]) + rng.normal(0, 1, (n, k))
    d = pd.DataFrame([
        {"subject": f"s{i}", "group": f"T{j}", "value": Y[i, j]} for i in range(n) for j in range(k)
    ])
    r = one_factor(d, ["T0", "T1", "T2"], StatsOptions(design="paired", geisser_greenhouse=False))
    ref = sm.AnovaRM(d, "value", "subject", within=["group"]).fit().anova_table
    assert r.stat == pytest.approx(ref["F Value"].iloc[0])
    assert r.p == pytest.approx(ref["Pr > F"].iloc[0])
    gg = one_factor(d, ["T0", "T1", "T2"], StatsOptions(design="paired"))
    eps = gg_epsilon(Y)
    assert gg.p == pytest.approx(stats.f.sf(r.stat, 2 * eps, 14 * eps))


def test_gg_epsilon_is_one_under_compound_symmetry():
    n = 400
    subj = rng.normal(0, 3, n)[:, None]
    Y = subj + rng.normal(0, 1, (n, 4))
    assert gg_epsilon(Y) == pytest.approx(1.0, abs=0.05)


def _two_way_df(cells: dict[tuple[str, str], int], seed: int = 5) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    rows = []
    for (a, b), n in cells.items():
        for v in r.normal(10 + 3 * (a == "Drug") + 2 * (b == "Post") + 4 * (a == "Drug") * (b == "Post"), 3, n):
            rows.append({"a": a, "b": b, "value": v})
    return pd.DataFrame(rows)


def test_two_way_type3_matches_statsmodels_unbalanced():
    smf = pytest.importorskip("statsmodels.formula.api")
    from statsmodels.stats.anova import anova_lm

    d = _two_way_df({("Ctrl", "Pre"): 5, ("Ctrl", "Post"): 9, ("Drug", "Pre"): 6, ("Drug", "Post"): 3})
    r = two_factor(d, "Drug", "Time", ["Ctrl", "Drug"], ["Pre", "Post"], StatsOptions())
    fit = smf.ols("value ~ C(a, Sum) * C(b, Sum)", data=d).fit()
    ref = anova_lm(fit, typ=3)
    eff = {e.source: e for e in r.effects}
    assert eff["Drug"].p == pytest.approx(ref.loc["C(a, Sum)", "PR(>F)"])
    assert eff["Time"].p == pytest.approx(ref.loc["C(b, Sum)", "PR(>F)"])
    assert eff["Drug x Time"].p == pytest.approx(ref.loc["C(a, Sum):C(b, Sum)", "PR(>F)"])
    assert eff["Residual"].ss == pytest.approx(fit.ssr)


def test_two_way_empty_cell_is_reported():
    d = _two_way_df({("Ctrl", "Pre"): 5, ("Ctrl", "Post"): 5, ("Drug", "Pre"): 5})
    r = two_factor(d, "Drug", "Time", ["Ctrl", "Drug"], ["Pre", "Post"], StatsOptions())
    assert "empty" in r.error


def _mixed_df(n_per: dict[str, int], k: int = 3, seed: int = 9) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    rows = []
    for g, n in n_per.items():
        for i in range(n):
            s = r.normal(0, 3)
            for j in range(k):
                rows.append({"subject": f"{g}{i}", "a": g, "b": f"T{j}",
                             "value": 10 + s + j * (1.5 if g == "Drug" else 0.3) + r.normal(0, 1)})
    return pd.DataFrame(rows)


def test_mixed_anova_balanced_matches_textbook():
    d = _mixed_df({"Ctrl": 6, "Drug": 6})
    opts = StatsOptions(repeated_factor="Time", geisser_greenhouse=False)
    r = two_factor(d, "Drug", "Time", ["Ctrl", "Drug"], ["T0", "T1", "T2"], opts)
    eff = {e.source: e for e in r.effects}
    # Textbook split-plot sums of squares (equal n).
    gm = d["value"].mean()
    k, n_s = 3, 12
    ss_a = sum(k * 6 * (d[d.a == g].value.mean() - gm) ** 2 for g in ("Ctrl", "Drug"))
    subj_means = d.groupby("subject").value.mean()
    grp_of = d.groupby("subject").a.first()
    ss_sa = k * sum((subj_means[s] - d[d.a == grp_of[s]].value.mean()) ** 2 for s in subj_means.index)
    ss_b = n_s * sum((d[d.b == t].value.mean() - gm) ** 2 for t in ("T0", "T1", "T2"))
    cell = d.groupby(["a", "b"]).value.mean()
    ss_cells = 6 * sum((cell[(g, t)] - gm) ** 2 for g in ("Ctrl", "Drug") for t in ("T0", "T1", "T2"))
    ss_ab = ss_cells - ss_a - ss_b
    ss_tot = ((d.value - gm) ** 2).sum()
    ss_err = ss_tot - ss_a - ss_sa - ss_b - ss_ab
    assert eff["Drug"].ss == pytest.approx(ss_a)
    assert eff["Time"].ss == pytest.approx(ss_b)
    assert eff["Drug x Time"].ss == pytest.approx(ss_ab)
    assert eff["Residual (within-subjects error)"].ss == pytest.approx(ss_err)
    f_ab = (ss_ab / 2) / (ss_err / (10 * 2))
    assert eff["Drug x Time"].p == pytest.approx(stats.f.sf(f_ab, 2, 20))


def test_mixed_rejects_subject_in_two_between_levels():
    d = _mixed_df({"Ctrl": 3, "Drug": 3})
    d.loc[(d.subject == "Ctrl0") & (d.b == "T1"), "a"] = "Drug"
    r = two_factor(d, "Drug", "Time", ["Ctrl", "Drug"], ["T0", "T1", "T2"],
                   StatsOptions(repeated_factor="Time"))
    assert "Ctrl0" in r.error


def test_undefined_values_are_dropped_with_note():
    d = _long({"A": np.array([1.0, 2.0, np.nan, 3.0]), "B": np.array([4.0, 5.0, 6.0])})
    r = one_factor(d, ["A", "B"], StatsOptions())
    assert r.groups[0].n == 3
    assert any("undefined" in n for n in r.notes)


def test_formatting_helpers():
    assert p_stars(0.00001) == "****"
    assert p_stars(0.2) == "ns"
    assert format_p(0.00001) == "P<0.0001"
    assert ci95_multiplier(6) == pytest.approx(stats.t.ppf(0.975, 5))
    assert ci95_multiplier(1) == 0.0
