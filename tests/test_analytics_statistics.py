"""The Analytics tab's statistics run on the per-animal values the user sees.

Pins the regressions found when ABEL's P values disagreed with Prism on the
same data: behaviors summed together, non-performers given latency 0, and
mean bout duration of 0 s for animals with no bouts.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abel.services.group_stats_options import StatsOptions
from abel.ui.tabs.behavior_analytics_tab import (
    _GraphsWidget,
    _SummaryStatsWidget,
    _aggregate_metric,
    _eb_val,
)


def test_aggregate_keeps_behaviors_and_undefined_values_apart() -> None:
    df = pd.DataFrame([
        {"session_label": "m1", "behavior": "Rear", "n_bouts": 2, "mean_bout_s": 1.5, "latency_s": 12.0},
        {"session_label": "m1", "behavior": "Groom", "n_bouts": 1, "mean_bout_s": 4.0, "latency_s": 40.0},
        {"session_label": "m2", "behavior": "Rear", "n_bouts": 0, "mean_bout_s": 0.0, "latency_s": np.nan},
    ])
    lat = _aggregate_metric(df, ["session_label", "behavior"], "latency_s")
    assert lat.set_index(["session_label", "behavior"]).loc[("m1", "Rear"), "latency_s"] == 12.0
    # An all-missing sum stays missing; it must not become a 0 s latency.
    assert np.isnan(lat.set_index(["session_label", "behavior"]).loc[("m2", "Rear"), "latency_s"])
    mb = _aggregate_metric(df, ["session_label", "behavior"], "mean_bout_s")
    assert np.isnan(mb.set_index(["session_label", "behavior"]).loc[("m2", "Rear"), "mean_bout_s"])


def test_ci_error_bar_uses_t_not_196() -> None:
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sem = vals.std(ddof=1) / np.sqrt(6)
    assert _eb_val(vals, "95% CI") == pytest.approx(stats.t.ppf(0.975, 5) * sem)


def _rows() -> list[dict]:
    # Rear: Ctrl m1-m3, Drug m4-m6 (m6 never reared: no row at all).
    # Groom rows must never leak into a Rear test.
    rear = {"m1": 10.0, "m2": 14.0, "m3": 12.0, "m4": 30.0, "m5": 26.0}
    rows = [
        {"session_label": s, "behavior": "Rear", "n_bouts": 3, "latency_s": v,
         "time_spent_s": 5.0, "mean_bout_s": 1.0}
        for s, v in rear.items()
    ]
    rows += [
        {"session_label": s, "behavior": "Groom", "n_bouts": 1, "latency_s": 100.0,
         "time_spent_s": 2.0, "mean_bout_s": 2.0}
        for s in rear
    ]
    return rows


def _summary_self() -> SimpleNamespace:
    graphs = SimpleNamespace(
        _graph_rows=lambda metric=None, group_filter=True: _rows(),
        _missing_value_for_metric=lambda metric, sess: 600.0 if metric == "latency_s" else 0.0,
        _is_data_range_active=lambda: False,
        _is_bout_filter_active=lambda: False,
    )
    factors = {f"m{i}": {"Treatment": "Ctrl" if i <= 3 else "Drug"} for i in range(1, 7)}
    host = SimpleNamespace(
        _graphs_tab=graphs,
        ordered_session_labels=lambda: [f"m{i}" for i in range(1, 7)],
        subject_for_label=lambda lbl: lbl,
        _session_factors=factors,
        _levels_by_factor=lambda: {"Treatment": ["Ctrl", "Drug"]},
    )
    me = SimpleNamespace(
        _host=host,
        _checked_subjects=lambda: {f"m{i}" for i in range(1, 7)},
        _STATS_METRICS=_SummaryStatsWidget._STATS_METRICS,
    )
    me._stats_value_table = types.MethodType(_SummaryStatsWidget._stats_value_table, me)
    return me


def test_dialog_tests_each_behavior_on_per_animal_values() -> None:
    me = _summary_self()
    text = _SummaryStatsWidget._compute_statistics(me, "latency_s", "Treatment", "", StatsOptions())
    assert "Rear: Latency to First (s)" in text and "Groom: Latency to First (s)" in text
    # m6 never reared: latency = time available (600 s), not 0 and not dropped.
    ctrl, drug = [10.0, 14.0, 12.0], [30.0, 26.0, 600.0]
    p = stats.ttest_ind(ctrl, drug).pvalue
    rear_block = text.split("Rear: Latency to First (s)")[1]
    assert f"P={p:.4f}" in rear_block
    assert "Unpaired t test" in rear_block


def _graph_self(split: list[str], factors: dict) -> SimpleNamespace:
    host = SimpleNamespace(
        _stats_options=StatsOptions(),
        subject_for_label=lambda lbl: lbl,
        _facet_split_factors=split,
        _session_factors=factors,
        _levels_by_factor=lambda: {"Drug": ["Saline", "Fent"], "Sex": ["M", "F"]},
    )
    return SimpleNamespace(_host=host, _stats_reports=[])


def test_graph_stats_use_the_plotted_values() -> None:
    me = _graph_self(["Drug"], {})
    sess_agg = pd.DataFrame({
        "session_label": [f"m{i}" for i in range(8)],
        "group": ["Saline"] * 4 + ["Fent"] * 4,
        "n_bouts": [3, 4, 5, 4, 9, 8, 10, 11],
    })
    res = _GraphsWidget._run_graph_stats(me, sess_agg, "n_bouts", ["Saline", "Fent"], "Bouts")
    assert res.p == pytest.approx(stats.ttest_ind([3, 4, 5, 4], [9, 8, 10, 11]).pvalue)
    assert me._stats_reports and me._stats_reports[0][0] == "Bouts"


def test_graph_two_split_factors_run_two_way_with_display_labels() -> None:
    rng = np.random.default_rng(3)
    rows, factors = [], {}
    for i in range(16):
        drug = "Saline" if i < 8 else "Fent"
        sex = "M" if i % 2 else "F"
        lbl = f"m{i}"
        factors[lbl] = {"Drug": drug, "Sex": sex}
        rows.append({"session_label": lbl, "group": f"{drug} × {sex}",
                     "n_bouts": rng.normal(10 + 5 * (drug == "Fent"), 1)})
    me = _graph_self(["Drug", "Sex"], factors)
    sess_agg = pd.DataFrame(rows)
    groups = ["Saline × M", "Saline × F", "Fent × M", "Fent × F"]
    res = _GraphsWidget._run_graph_stats(me, sess_agg, "n_bouts", groups, "Bouts")
    assert {e.source for e in res.effects} >= {"Drug", "Sex", "Drug x Sex"}
    # Comparison labels are the graph's own group labels, so brackets can be placed.
    assert {c.a for c in res.comparisons} <= set(groups)
    drug = next(e for e in res.effects if e.source == "Drug")
    assert drug.p < 0.001
