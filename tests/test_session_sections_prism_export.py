"""Session Sections export: Prism-ready sheets and zero-filled subjects.

The export used to be a long CSV only (one row per subject x behavior x
section), which has to be hand-pivoted before it can be pasted into Prism.  The
workbook now opens on one sheet per metric laid out the way a Prism grouped
table ingests it: rows = sections, columns = subjects (blocked and padded by
group in group mode).

A scored subject with no bouts of a behavior used to produce no rows at all, so
it vanished from the chart and group means averaged only the subjects that did
the behavior.  Those pairs are now real zeros; never-scored pairs stay blank.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
openpyxl = pytest.importorskip("openpyxl")

from abel.ui.tabs.behavior_analytics_tab import (  # noqa: E402
    _SessionSectionsWidget as W,
)

SECTIONS = [("Baseline", 10.0), ("Tone 1", 10.0), ("ITI 1", 10.0)]


class _Stub:
    """Just enough of the widget for the export/compute code paths."""

    _EXPORT_METRICS = W._EXPORT_METRICS
    _section_pivot = staticmethod(W._section_pivot)
    _section_base_type = staticmethod(W._section_base_type)
    _prism_section_table = staticmethod(W._prism_section_table)
    _compute_section_data = W._compute_section_data
    _export_subject_slots = W._export_subject_slots
    _long_export_table = W._long_export_table
    _write_export_workbook = W._write_export_workbook
    _export_readme_lines = W._export_readme_lines

    def __init__(self, host, bids, metric="n_bouts"):
        self._host = host
        self._bids = bids
        self._metric = metric
        self._ss_facet_controls = {}
        self._bin_sections_chk = SimpleNamespace(isChecked=lambda: False)

    def _get_sections(self):
        return list(SECTIONS)

    def _checked_behavior_ids(self):
        return list(self._bids)

    def _get_metric(self):
        return self._metric

    def _get_aggregate(self):
        return "section"


def _host(raw_bouts, summary_rows, groups=None, order=None):
    labels = {r["session_id"]: r["session_label"] for r in summary_rows}
    return SimpleNamespace(
        _raw_bouts=raw_bouts,
        _summary_rows=summary_rows,
        _session_label_by_session=labels,
        _project_fps=lambda: 10.0,
        _summary_tab=SimpleNamespace(_checked_subjects=lambda: set()),
        _roi_scope_zone=0,
        _animal_scope="",
        roi_scope_label=lambda: "Whole arena",
        ordered_session_labels=lambda: list(order or sorted(set(labels.values()))),
        _ordered_group_list=lambda avail, _split=None: sorted(avail),
        _split_factors_for_controls=lambda _c: [],
        _session_groups=groups or {},
    )


def _bouts(bid, name, spans):
    """spans: list of (session_id, start_s, end_s) at 10 fps."""
    return pd.DataFrame({
        "session_id": [s for s, _a, _b in spans],
        "start_frame": [int(a * 10) for _s, a, _b in spans],
        "end_frame": [int(b * 10) for _s, _a, b in spans],
        "behavior": name,
        "behavior_id": bid,
    })


def _summary(pairs):
    return [
        {"session_id": sid, "session_label": f"m{sid}", "behavior_id": bid}
        for sid, bid in pairs
    ]


class TestZeroFill:
    def test_scored_subject_without_bouts_is_zero_not_missing(self):
        raw = {"b1": _bouts("b1", "Freeze", [("1", 12.0, 15.0)])}
        host = _host(raw, _summary([("1", "b1"), ("2", "b1")]))
        df = _Stub(host, ["b1"])._compute_section_data()
        m2 = df[df["session_label"] == "m2"]
        assert list(m2["section_name"]) == ["Baseline", "Tone 1", "ITI 1"]
        assert (m2[["n_bouts", "duration_s", "pct_time"]].to_numpy() == 0).all()
        # The subject with bouts is unchanged: 3 s of the 10 s tone.
        tone = df[(df["session_label"] == "m1") & (df["section_name"] == "Tone 1")]
        assert tone["duration_s"].item() == pytest.approx(3.0)
        assert tone["pct_time"].item() == pytest.approx(30.0)

    def test_unscored_pair_stays_absent(self):
        raw = {"b1": _bouts("b1", "Freeze", [("1", 12.0, 15.0)])}
        host = _host(raw, _summary([("1", "b1"), ("2", "b2")]))
        df = _Stub(host, ["b1"])._compute_section_data()
        assert set(df["session_label"]) == {"m1"}

    def test_zero_subject_now_lowers_the_group_mean(self):
        raw = {"b1": _bouts("b1", "Freeze", [("1", 12.0, 16.0)])}
        host = _host(raw, _summary([("1", "b1"), ("2", "b1")]))
        df = _Stub(host, ["b1"])._compute_section_data()
        piv = W._section_pivot(df, "duration_s", [n for n, _ in SECTIONS])
        assert piv["Tone 1"].dropna().mean() == pytest.approx(2.0)


class TestPrismTable:
    def test_rows_are_sections_columns_are_slots(self):
        df = pd.DataFrame({
            "session_label": ["a", "a", "b", "b"],
            "section_name": ["S1", "S2", "S1", "S2"],
            "n_bouts": [1, 2, 3, 4],
        })
        slots = [("", "b"), ("", "a"), ("", "ghost")]
        t = W._prism_section_table(df, "n_bouts", ["S1", "S2"], slots)
        assert list(t.index) == ["S1", "S2"]
        assert t[0].tolist() == [3, 4]
        assert t[1].tolist() == [1, 2]
        assert t[2].isna().all()

    def test_group_slots_are_blocked_and_padded(self):
        rows = []
        for lbl, grp in [("m1", "Ctrl"), ("m2", "Drug"), ("m3", "Ctrl"), ("m4", "Ctrl")]:
            rows.append({"session_label": lbl, "group": grp})
        host = _host({}, [], order=["m4", "m3", "m2", "m1"])
        slots = _Stub(host, [])._export_subject_slots(pd.DataFrame(rows))
        assert slots == [
            ("Ctrl", "m4"), ("Ctrl", "m3"), ("Ctrl", "m1"),
            ("Drug", "m2"), ("Drug", None), ("Drug", None),
        ]


class TestWorkbook:
    def _export(self, tmp_path, groups=None, metric="pct_time"):
        raw = {
            "b1": _bouts("b1", "Freeze", [("1", 12.0, 15.0), ("3", 0.0, 5.0)]),
            "b2": _bouts("b2", "Groom", [("2", 21.0, 30.0)]),
        }
        pairs = [(s, b) for s in ("1", "2", "3") for b in ("b1", "b2")]
        host = _host(raw, _summary(pairs), order=["m3", "m1", "m2"])
        stub = _Stub(host, ["b1", "b2"], metric=metric)
        df = stub._compute_section_data()
        if groups:
            df["group"] = df["session_label"].map(groups)
        path = tmp_path / "sections.xlsx"
        stub._write_export_workbook(str(path), df, SECTIONS, stub._long_export_table(df))
        return openpyxl.load_workbook(path)

    @staticmethod
    def _rows(ws):
        return [list(r) for r in ws.iter_rows(values_only=True)]

    def test_sheet_order_opens_on_chart_metric(self, tmp_path):
        wb = self._export(tmp_path, metric="pct_time")
        assert wb.sheetnames == [
            "Prism - % Time", "Prism - Bout Count", "Prism - Duration (s)",
            "Long Format", "README",
        ]

    def test_individual_block_layout(self, tmp_path):
        rows = self._rows(self._export(tmp_path)["Prism - Duration (s)"])
        assert rows[0][0] == "Freeze - Duration (s)"
        assert rows[1] == ["Section", "m3", "m1", "m2"]
        assert rows[2] == ["Baseline", 5.0, 0.0, 0.0]
        assert rows[3] == ["Tone 1", 0.0, 3.0, 0.0]
        assert rows[4] == ["ITI 1", 0.0, 0.0, 0.0]
        assert rows[5] == [None] * 4
        assert rows[6][0] == "Groom - Duration (s)"
        assert rows[10] == ["ITI 1", 0.0, 0.0, 9.0]

    def test_group_block_has_group_row_and_padding(self, tmp_path):
        # Drug is last, so its padding column is trailing blanks (trimmed by
        # openpyxl); put the short group first to see the padding in place.
        wb = self._export(tmp_path, groups={"m1": "Drug", "m2": "Ctrl", "m3": "Drug"})
        rows = self._rows(wb["Prism - Bout Count"])
        assert rows[1] == ["Group", "Ctrl", None, "Drug", None]
        assert rows[2] == ["Section", "m2", None, "m3", "m1"]
        assert rows[3] == ["Baseline", 0, None, 1, 0]
        assert rows[5] == ["ITI 1", 0, None, 0, 0]
        readme = [r[0] for r in self._rows(wb["README"])]
        assert any("padded to 2 columns" in (line or "") for line in readme)

    def test_long_sheet_carries_every_metric(self, tmp_path):
        rows = self._rows(self._export(tmp_path)["Long Format"])
        assert rows[0] == [
            "Session", "Behavior", "Section", "Section Type", "Section Order",
            "Avg Bout Count", "Avg Duration (s)", "Avg % Time In Section",
        ]
        assert len(rows) == 1 + 3 * 2 * 3
