"""The "Scale by pre-behavior time" rate must not be exported as time bins.

"Bouts Until Behavior" + "Scale by pre-behavior time" divides each subject's
pre-target count by that subject's *own* pre-target interval, producing one
rate per subject (e.g. bouts / min).  The Excel export used to route it through
the time-binned path, which emitted raw unscaled per-bin counts plus a "total"
column -- neither the graphed rate nor the labeled unit.  It now exports one
value per subject per behavior instead.

The helpers are exercised on a lightweight stub rather than a real widget so
the tests run headless.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import abel.ui.tabs.behavior_analytics_tab as bat
from abel.ui.tabs.behavior_analytics_tab import _GraphsWidget


FPS = 30.0
SUBJECTS = ["M01", "M02", "M03_no_eat"]
GROUPS = {"M01": "Control", "M02": "Stress", "M03_no_eat": "Stress"}

REAR = "b_rear"
EAT = "b_eat"

# M01: 3 Rear bouts, first Eat at 120 s  -> 3 / 2 min = 1.5 /min
# M02: 2 Rear bouts, first Eat at  60 s  -> 2 / 1 min = 2.0 /min
# M03: 4 Rear bouts, never eats, session ends at 240 s -> 4 / 4 min = 1.0 /min
_REAR_STARTS = {"s1": [0, 300, 600], "s2": [0, 300], "s3": [0, 300, 600, 900]}
_EAT_STARTS = {"s1": [3600], "s2": [1800]}
_SESSION_END_S = 240.0

_LABEL_BY_SESSION = {"s1": "M01", "s2": "M02", "s3": "M03_no_eat"}


def _bout_df(starts_by_session: dict[str, list[int]], behavior: str) -> pd.DataFrame:
    rows = []
    for sid, starts in starts_by_session.items():
        for st in starts:
            rows.append({
                "session_id": sid, "behavior": behavior,
                "start_frame": st, "end_frame": st + 29,   # 1 s bouts
            })
    return pd.DataFrame(rows)


def _summary_rows() -> list[dict]:
    """Unscaled per-session totals, as _filtered_rows() would supply them."""
    return [
        {"session_id": sid, "subject": _LABEL_BY_SESSION[sid],
         "session_label": _LABEL_BY_SESSION[sid], "behavior_id": REAR,
         "behavior": "Rear", "n_bouts": float(len(starts)),
         "time_spent_s": float(len(starts)), "mean_bout_s": 1.0,
         "latency_s": 0.0, "distance_cm": 0.0}
        for sid, starts in _REAR_STARTS.items()
    ]


def _make_widget(scaling: bool = True, style: str = "bar", mode: str = "individual"):
    stub = SimpleNamespace()
    for name in (
        "_export_sessions", "_missing_value_for_metric", "_graph_rows",
        "_per_session_metric_table", "_scaled_rate_table",
        "_recompute_rows_for_first_n", "_cutoff_frames_for_until_behavior",
        "_is_until_scaling_active", "_is_until_behavior_active",
        "_is_bout_filter_active", "_is_first_n_active",
        "_get_until_behavior_id", "_get_until_scale_factor", "_metric_label",
        "_export_excel_scaled_rate", "_export_excel_data",
    ):
        setattr(stub, name, getattr(_GraphsWidget, name).__get__(stub))

    raw = {
        REAR: _bout_df(_REAR_STARTS, "Rear"),
        EAT: _bout_df(_EAT_STARTS, "Eat"),
    }
    rows = _summary_rows()

    stub._host = SimpleNamespace(
        _session_groups=dict(GROUPS),
        _summary_rows=list(rows),
        _filtered_rows=lambda: list(rows),
        _raw_bouts=raw,
        _project_fps=lambda: FPS,
        _session_label_by_session=dict(_LABEL_BY_SESSION),
        ordered_session_labels=lambda: list(SUBJECTS),
        _ordered_group_list=lambda names, *a: sorted({n for n in names if n}),
        _sessions_by_label={lbl: [sid] for sid, lbl in _LABEL_BY_SESSION.items()},
        _summary_tab=SimpleNamespace(_checked_subjects=lambda: set(SUBJECTS)),
        _status=SimpleNamespace(setText=lambda _t: None),
    )
    stub._get_metric = lambda: "n_bouts"
    stub._get_style = lambda: style
    stub._get_mode = lambda: mode
    stub._checked_groups = lambda: set()
    stub._get_data_range_seconds = lambda: (None, None)
    stub._is_data_range_active = lambda: False
    stub._apply_latency_fallbacks = lambda r: r
    stub._get_first_n_bouts = lambda: 0
    stub._session_analysis_end_s = lambda _sid: _SESSION_END_S
    stub._bout_filter_mode = SimpleNamespace(
        currentData=lambda: "until_behavior" if scaling else "none")
    stub._until_behavior_combo = SimpleNamespace(currentData=lambda: EAT)
    stub._until_scale_chk = SimpleNamespace(isChecked=lambda: scaling)
    stub._until_scale_spin = SimpleNamespace(value=lambda: 60.0)
    return stub


# -- the scaled table itself -------------------------------------------------

def test_scaled_table_is_one_rate_per_subject():
    w = _make_widget()
    tbl = w._scaled_rate_table()

    assert tbl is not None
    label = w._metric_label("n_bouts")
    assert label == "Bout Count (/ min)", label
    # One row per subject x behavior -- no bins, no total row.
    assert len(tbl) == len(SUBJECTS)
    assert set(tbl.columns) == {"Session", "Group", "Behavior", label}
    assert not any(str(c).lower() == "total" for c in tbl.columns)

    by_sess = dict(zip(tbl["Session"], tbl[label]))
    assert by_sess["M01"] == pytest.approx(1.5)
    assert by_sess["M02"] == pytest.approx(2.0)
    # Target never occurred -> denominator is the full session duration.
    assert by_sess["M03_no_eat"] == pytest.approx(1.0)


# -- routing -----------------------------------------------------------------

@pytest.mark.parametrize("style", ["bar", "box", "line", "overtime"])
def test_excel_export_routes_to_scaled_rate_for_every_style(style):
    """Even the time-course styles must not bin a pre-behavior rate."""
    w = _make_widget(style=style)
    called = []
    w._export_excel_scaled_rate = lambda: called.append(True)
    w._export_ethogram_excel = lambda: pytest.fail("ethogram path taken")
    w._export_excel_summary_metric = lambda: pytest.fail("summary path taken")
    w._build_wide_binned_df = lambda: pytest.fail("binned path taken")

    w._export_excel_data()
    assert called == [True]


def test_excel_export_keeps_binned_path_when_scaling_off(monkeypatch):
    w = _make_widget(scaling=False, style="overtime")
    seen = []
    w._export_excel_scaled_rate = lambda: pytest.fail("scaled path taken")
    w._export_ethogram_excel = lambda: pytest.fail("ethogram path taken")
    w._export_excel_summary_metric = lambda: seen.append("summary")
    w._bin_bouts = lambda: pd.DataFrame()
    monkeypatch.setattr(
        bat.QMessageBox, "information",
        staticmethod(lambda *a, **k: seen.append("no-data")),
    )

    w._export_excel_data()
    # overtime + n_bouts still goes down the binned path (which here finds no
    # data), NOT the scaled one.
    assert seen == ["no-data"]


def test_excel_export_unbins_bar_charts():
    """A bar chart shows one value per subject, so its Excel must too."""
    w = _make_widget(scaling=False, style="bar")
    seen = []
    w._export_excel_scaled_rate = lambda: pytest.fail("scaled path taken")
    w._export_ethogram_excel = lambda: pytest.fail("ethogram path taken")
    w._export_excel_summary_metric = lambda: seen.append("summary")
    w._build_wide_binned_df = lambda: pytest.fail("binned path taken")

    w._export_excel_data()
    assert seen == ["summary"]


# -- the workbook ------------------------------------------------------------

def test_workbook_is_subjects_by_behaviors(monkeypatch, tmp_path):
    pytest.importorskip("openpyxl")
    out = tmp_path / "rate.xlsx"
    w = _make_widget()
    monkeypatch.setattr(
        bat.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out), "")),
    )

    w._export_excel_scaled_rate()

    book = pd.read_excel(out, sheet_name=None)
    assert list(book) == ["Bout Count (per min)"], list(book)
    sheet = book["Bout Count (per min)"]
    assert list(sheet.columns) == ["Session", "Group", "Rear"]
    assert list(sheet["Session"]) == SUBJECTS
    assert list(sheet["Rear"]) == pytest.approx([1.5, 2.0, 1.0])
