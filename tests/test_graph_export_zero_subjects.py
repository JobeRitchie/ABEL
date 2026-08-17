"""A subject that scores zero for every selected behavior must still be
exported from the Graphs tab.

Such a subject produces no summary rows and no binned bout rows, so every
export path that derived its subject list from the data itself dropped it
silently.  The exports now build their rows from the checked-subject roster.

The export helpers are exercised on a lightweight stub rather than a real
widget so the tests run headless.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from abel.ui.tabs.behavior_analytics_tab import _GraphsWidget


ZERO_SUBJECT = "M03_zero"
SUBJECTS = ["M01", "M02", ZERO_SUBJECT]
GROUPS = {"M01": "Control", "M02": "Control", ZERO_SUBJECT: "Control"}
BIN_SECONDS = 30


def _binned_rows() -> pd.DataFrame:
    """`_bin_bouts()`-shaped output — only M01/M02 ever performed the behavior."""
    return pd.DataFrame([
        {"session_label": "M01", "behavior_id": "b1", "behavior": "Rear",
         "time_bin_s": 0, "n_bouts": 2.0, "duration_s": 4.0, "distance_cm": 0.0,
         "mean_bout_s": 2.0},
        {"session_label": "M01", "behavior_id": "b1", "behavior": "Rear",
         "time_bin_s": 30, "n_bouts": 1.0, "duration_s": 3.0, "distance_cm": 0.0,
         "mean_bout_s": 3.0},
        {"session_label": "M02", "behavior_id": "b1", "behavior": "Rear",
         "time_bin_s": 0, "n_bouts": 4.0, "duration_s": 8.0, "distance_cm": 0.0,
         "mean_bout_s": 2.0},
    ])


def _summary_rows() -> list[dict]:
    """`_filtered_rows()`-shaped output — no row at all for the zero subject."""
    return [
        {"session_id": "s1", "subject": "M01", "session_label": "M01",
         "behavior_id": "b1", "behavior": "Rear", "n_bouts": 3.0,
         "time_spent_s": 7.0, "mean_bout_s": 2.333, "latency_s": 1.0,
         "distance_cm": 0.0},
        {"session_id": "s2", "subject": "M02", "session_label": "M02",
         "behavior_id": "b1", "behavior": "Rear", "n_bouts": 4.0,
         "time_spent_s": 8.0, "mean_bout_s": 2.0, "latency_s": 2.0,
         "distance_cm": 0.0},
    ]


def _make_widget(metric: str, style: str, mode: str = "individual"):
    """Build a stub carrying the real export helpers off ``_GraphsWidget``."""
    stub = SimpleNamespace()
    for name in (
        "_export_sessions", "_missing_value_for_metric", "_binned_session_grid",
        "_session_aggregate", "_build_wide_binned_df", "_collect_graph_data",
    ):
        setattr(stub, name, getattr(_GraphsWidget, name).__get__(stub))

    binned = _binned_rows()
    rows = _summary_rows()

    stub._host = SimpleNamespace(
        _session_groups=dict(GROUPS),
        _summary_rows=list(rows),
        _filtered_rows=lambda: list(rows),
        ordered_session_labels=lambda: list(SUBJECTS),
        _ordered_group_list=lambda names, *a: sorted({n for n in names if n}),
        _sessions_by_label={s: [s] for s in SUBJECTS},
        _summary_tab=SimpleNamespace(_checked_subjects=lambda: set(SUBJECTS)),
    )
    stub._bin_bouts = lambda: binned
    stub._get_metric = lambda: metric
    stub._get_style = lambda: style
    stub._get_mode = lambda: mode
    stub._checked_groups = lambda: set()
    stub._metric_label = lambda m: {"n_bouts": "Bout Count",
                                    "time_spent_s": "Time Spent (s)",
                                    "mean_bout_s": "Mean Bout (s)"}[m]
    stub._time_bin_spin = SimpleNamespace(value=lambda: BIN_SECONDS)
    stub._get_data_range_seconds = lambda: (None, None)
    stub._is_data_range_active = lambda: False
    stub._is_bout_filter_active = lambda: False
    stub._apply_latency_fallbacks = lambda r: r
    return stub


# -- wide binned export (Excel / time-binned CSV) ----------------------------

def test_wide_binned_export_includes_zero_subject():
    w = _make_widget("n_bouts", "overtime")
    wide = w._build_wide_binned_df()

    assert wide is not None
    assert ZERO_SUBJECT in set(wide["session_label"]), (
        "subject with no bouts was dropped from the binned export"
    )
    zero_row = wide[wide["session_label"] == ZERO_SUBJECT].iloc[0]
    assert zero_row["total"] == 0
    assert zero_row["0s"] == 0 and zero_row["30s"] == 0
    assert zero_row["group"] == "Control"


def test_wide_binned_export_keeps_real_values():
    w = _make_widget("n_bouts", "overtime")
    wide = w._build_wide_binned_df()

    m01 = wide[wide["session_label"] == "M01"].iloc[0]
    assert m01["0s"] == 2 and m01["30s"] == 1 and m01["total"] == 3


# -- bar / box / stacked / overview -----------------------------------------

def test_bar_export_includes_zero_subject():
    w = _make_widget("n_bouts", "bar")
    data = w._collect_graph_data(export_individual_sessions=True)

    assert set(data["Session"]) == set(SUBJECTS)
    zero = data[data["Session"] == ZERO_SUBJECT].iloc[0]
    assert zero["Bout Count"] == 0


def test_box_export_includes_zero_subject():
    w = _make_widget("time_spent_s", "box")
    data = w._collect_graph_data()

    assert ZERO_SUBJECT in set(data["session_label"])
    zero = data[data["session_label"] == ZERO_SUBJECT].iloc[0]
    assert zero["Time Spent (s)"] == 0


def test_stacked_export_includes_zero_subject():
    w = _make_widget("time_spent_s", "stacked")
    data = w._collect_graph_data()

    assert ZERO_SUBJECT in set(data["Session"])
    zero = data[data["Session"] == ZERO_SUBJECT].iloc[0]
    assert zero["Rear"] == 0


def test_overview_export_includes_zero_subject():
    w = _make_widget("n_bouts", "overview")
    data = w._collect_graph_data()

    assert ZERO_SUBJECT in set(data["Session"])
    zero = data[data["Session"] == ZERO_SUBJECT].iloc[0]
    assert zero["Bout Count"] == 0
    # Mean bout duration is undefined without bouts, not zero.
    assert math.isnan(float(zero["Mean Bout Duration (s)"]))


# -- group aggregates: the zero subject must count toward mean/SEM/N --------

def test_group_bar_counts_zero_subject_in_n_and_mean():
    w = _make_widget("n_bouts", "bar", mode="group")
    data = w._collect_graph_data()

    row = data[data["Behavior"] == "Rear"].iloc[0]
    assert row["N"] == 3, "zero subject missing from the group N"
    assert row["Bout Count"] == pytest.approx((3.0 + 4.0 + 0.0) / 3)


def test_group_timecourse_counts_zero_subject():
    w = _make_widget("n_bouts", "overtime", mode="group")
    data = w._collect_graph_data()

    first_bin = data[data["time_bin_s"] == 0].iloc[0]
    assert first_bin["N"] == 3
    assert first_bin["Bout Count"] == pytest.approx((2.0 + 4.0 + 0.0) / 3)
    # M02 has no bout in the second bin either — it must still count as zero.
    second_bin = data[data["time_bin_s"] == 30].iloc[0]
    assert second_bin["N"] == 3
    assert second_bin["Bout Count"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)


def test_mean_bout_is_blank_not_zero_for_zero_subject():
    """A subject that never performed the behavior has no mean bout length."""
    w = _make_widget("mean_bout_s", "bar")
    data = w._collect_graph_data(export_individual_sessions=True)

    zero = data[data["Session"] == ZERO_SUBJECT].iloc[0]
    assert math.isnan(float(zero["Mean Bout (s)"]))
