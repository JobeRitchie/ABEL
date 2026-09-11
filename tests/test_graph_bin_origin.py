"""Graphs-tab time bins can start at the Data Range 'from' or the session start.

With 60 s bins and Data Range 'from' = 100 s, "Start time bins at Data Range
'from'" gives bins 100-160, 160-220, ...; unchecked, bins stay aligned to the
session start (60-120, 120-180, ...) and the first bin holds only 100-120.

Exercised on a lightweight stub carrying the real methods so it runs headless.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from abel.ui.tabs.behavior_analytics_tab import BehaviorAnalyticsTab, _GraphsWidget

FPS = 10.0
BIN_SECONDS = 60
SUBJECT = "M01"


def _raw_bouts() -> dict[str, pd.DataFrame]:
    # A: 100-170 s.  B: 50-105 s (straddles the range start).
    return {"b1": pd.DataFrame({
        "session_id": ["s1", "s1"],
        "behavior": ["Rear", "Rear"],
        "start_frame": [1000, 500],
        "end_frame": [1699, 1049],
    })}


def _make_widget(lo: float | None, from_range: bool):
    stub = SimpleNamespace()
    for name in (
        "_bin_bouts", "_bin_grid", "_bin_origin_s", "_build_wide_binned_df",
        "_export_sessions", "_binned_session_grid",
    ):
        setattr(stub, name, getattr(_GraphsWidget, name).__get__(stub))
    stub._host = SimpleNamespace(
        _project_fps=lambda: FPS,
        _raw_bouts=_raw_bouts(),
        _summary_rows=[],
        _selected_behavior_ids=lambda: {"b1"},
        _session_label_by_session={"s1": SUBJECT},
        _session_groups={},
        ordered_session_labels=lambda: [SUBJECT],
        _ordered_group_list=lambda names, *a: sorted({n for n in names if n}),
        _summary_tab=SimpleNamespace(_checked_subjects=lambda: {SUBJECT}),
    )
    stub._time_bin_spin = SimpleNamespace(value=lambda: BIN_SECONDS)
    stub._bin_from_range_chk = SimpleNamespace(isChecked=lambda: from_range)
    stub._get_data_range_seconds = lambda: (lo, None)
    stub._get_metric = lambda: "time_spent_s"
    stub._get_mode = lambda: "individual"
    stub._checked_groups = lambda: set()
    stub._get_first_n_bouts = lambda: 0
    stub._get_until_behavior_id = lambda: None
    stub._cutoff_frames_for_until_behavior = lambda: {}
    stub._bin_cache = None
    stub._bin_cache_key = ()
    return stub


def _durations(w) -> dict:
    binned = w._bin_bouts()
    return dict(zip(binned["time_bin_s"], binned["duration_s"]))


def test_bins_start_at_range_from_when_checked():
    w = _make_widget(lo=100.0, from_range=True)
    assert _durations(w) == pytest.approx({100: 65.0, 160: 10.0})
    assert w._bin_grid(160) == [100, 160]


def test_bins_stay_session_aligned_when_unchecked():
    w = _make_widget(lo=100.0, from_range=False)
    # First bin holds only the in-range 100-120 s part.
    assert _durations(w) == pytest.approx({60: 25.0, 120: 50.0})
    # No zero-filled 0-60 bin lying wholly before the range.
    assert w._bin_grid(120) == [60, 120]


def test_checkbox_ignored_without_range_from():
    for from_range in (True, False):
        w = _make_widget(lo=None, from_range=from_range)
        assert _durations(w) == pytest.approx({0: 10.0, 60: 65.0, 120: 50.0})
        assert w._bin_grid(120) == [0, 60, 120]


def test_fractional_origin_keys_match_grid_and_export():
    w = _make_widget(lo=100.5, from_range=True)
    assert _durations(w) == pytest.approx({100.5: 64.5, 160.5: 9.5})
    assert w._bin_grid(160.5) == [100.5, 160.5]
    wide = w._build_wide_binned_df()
    row = wide.iloc[0]
    assert row["100.5s"] == pytest.approx(64.5)
    assert row["160.5s"] == pytest.approx(9.5)
    assert row["total"] == pytest.approx(74.0)


def test_wide_export_columns_follow_origin():
    wide = _make_widget(lo=100.0, from_range=True)._build_wide_binned_df()
    assert [c for c in wide.columns if c.endswith("s") and c[0].isdigit()] == ["100s", "160s"]
    wide = _make_widget(lo=100.0, from_range=False)._build_wide_binned_df()
    assert [c for c in wide.columns if c.endswith("s") and c[0].isdigit()] == ["60s", "120s"]


# -- Distance Traveled bins ---------------------------------------------------

def _distance_host():
    fps = 5.0  # equals the distance subsample rate -> every frame kept
    n = int(300 * fps)
    pose = SimpleNamespace(centroid_x=np.arange(n, dtype=float),
                           centroid_y=np.zeros(n))
    host = SimpleNamespace(
        _get_pose_for_session=lambda sid: pose,
        _project_fps=lambda: fps,
        _analysis_prechop_for_session=lambda sid: 0,
    )
    host._compute_session_distance_binned = (
        BehaviorAnalyticsTab._compute_session_distance_binned.__get__(host)
    )
    return host


def test_distance_bins_start_at_range_from():
    host = _distance_host()
    bins = dict(host._compute_session_distance_binned("s1", 60, 100.0, 100.0, None))
    assert bins[100] == pytest.approx(300.0)  # 60 s x 5 px/s
    assert min(bins) == 100


def test_distance_partial_first_bin_holds_only_in_range_movement():
    host = _distance_host()
    bins = dict(host._compute_session_distance_binned("s1", 60, 0.0, 100.0, 200.0))
    assert bins[60] == pytest.approx(100.0)   # 100-120 s only
    assert bins[180] == pytest.approx(100.0)  # 180-200 s only
    assert min(bins) == 60 and max(bins) == 180
