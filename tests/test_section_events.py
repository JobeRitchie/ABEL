"""Session Sections with per-subject event times.

Hand-run assays (pinches, a splash, a scruff) drift by minutes between
subjects, so one shared set of fixed durations puts every boundary after
the first in the wrong place.  An event file gives each subject its own
section starts and ends; Session Sections then measures each subject inside
its own sections.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from abel.services.section_events import (
    SectionEventTable,
    load_section_events,
    mean_section_durations,
    parse_section_events,
    section_metrics,
)

FPS = 10.0


def _file_df() -> pd.DataFrame:
    # Frames at 10 fps.  A: home cage 0-100 s, stim 100-200 s, post 200 s-end.
    # B is shifted 50 s later.  Blank cells borrow from the neighbour.
    return pd.DataFrame({
        "Subject": ["A", "B", "Orphan"],
        "Notes": ["", "x", ""],
        "Home Cage Start": [0, None, 0],
        "Home Cage End": [1000, 1500, 10],
        "Stim Start": [1000, None, 10],
        "Stim End": [2000, 2500, 20],
        "Post Start": [2000, 2500, 20],
        "Post End": [None, None, 30],
    })


class TestParse:
    def test_sections_in_column_order_and_other_columns_ignored(self):
        t = parse_section_events(_file_df())
        assert t.section_names == ["Home Cage", "Stim", "Post"]
        assert set(t.subjects) == {"A", "B", "Orphan"}
        assert t.subjects["A"][0] == (0.0, 1000.0)

    def test_no_section_columns_is_an_error(self):
        with pytest.raises(ValueError, match="Start"):
            parse_section_events(pd.DataFrame({"Subject": ["A"], "Pinch": [5]}))

    def test_workbook_prefers_section_events_sheet(self, tmp_path):
        p = tmp_path / "ev.xlsx"
        with pd.ExcelWriter(p) as w:
            pd.DataFrame({"x": [1]}).to_excel(w, sheet_name="Notes", index=False)
            _file_df().to_excel(w, sheet_name="Section Events", index=False)
        t = load_section_events(p)
        assert t.section_names == ["Home Cage", "Stim", "Post"]

    def test_state_round_trip(self):
        t = parse_section_events(_file_df(), source="f.xlsx")
        back = SectionEventTable.from_state(t.to_state())
        assert back.section_names == t.section_names
        assert back.subjects == t.subjects
        assert back.source == "f.xlsx"


class TestBounds:
    def test_blanks_borrow_from_neighbours_and_last_runs_to_session_end(self):
        t = parse_section_events(_file_df())
        b = t.bounds_seconds("B", FPS, session_end_s=400.0)
        # Blank Home Cage Start -> 0; blank Stim Start -> Home Cage End;
        # blank Post End -> session end.
        assert b == [(0.0, 150.0), (150.0, 250.0), (250.0, 400.0)]

    def test_open_last_section_without_session_end_is_missing(self):
        t = parse_section_events(_file_df())
        assert t.bounds_seconds("A", FPS)[-1] is None

    def test_prechop_moves_bounds_onto_the_analysis_clock(self):
        t = parse_section_events(_file_df())
        b = t.bounds_seconds("A", FPS, prechop_frames=200, session_end_s=300.0)
        # Home cage 0-100 s raw becomes 0-80 s (the first 20 s were chopped).
        assert b[0] == (0.0, 80.0)
        assert b[1] == (80.0, 180.0)
        assert b[2] == (180.0, 300.0)

    def test_match_subject_is_case_insensitive(self):
        t = parse_section_events(_file_df())
        assert t.match_subject("a ") == "A"
        assert t.match_subject(None, "nope", "b") == "B"
        assert t.match_subject("C") is None

    def test_problems_flags_backwards_times(self):
        df = _file_df()
        df.loc[0, "Stim End"] = 500  # before Stim Start (1000)
        issues = parse_section_events(df).problems()
        assert any("A: Stim ends before it starts" in s for s in issues)

    def test_mean_durations(self):
        t = parse_section_events(_file_df())
        means = mean_section_durations(t, ["A", "B"], {"A": FPS, "B": FPS})
        assert means[0] == pytest.approx(125.0)  # (100 + 150) / 2
        assert means[1] == pytest.approx(100.0)
        assert means[2] == 0.0  # open-ended, no session end given


class TestMetrics:
    def test_overlap_count_seconds_pct_and_rate(self):
        starts = np.array([5.0, 50.0, 95.0])
        ends = np.array([15.0, 60.0, 130.0])
        n, dur, pct, rate = section_metrics(starts, ends, 0.0, 120.0)
        assert n == 3
        assert dur == pytest.approx(10 + 10 + 25)
        assert pct == pytest.approx(45 / 120 * 100)
        assert rate == pytest.approx(3 / 2.0)  # 3 bouts in 2 minutes


# ── Widget: per-subject sections drive the computed table ───────────────

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def widget_cls():
    pytest.importorskip("PySide6")
    from abel.ui.tabs.behavior_analytics_tab import _SessionSectionsWidget

    return _SessionSectionsWidget


def _stub(W, table: SectionEventTable, raw_bouts, summary_rows, subjects):
    labels = {r["session_id"]: r["session_label"] for r in summary_rows}

    class _Stub:
        _compute_section_data = W._compute_section_data
        _event_bounds_by_session = W._event_bounds_by_session
        _event_subject_for_session = W._event_subject_for_session
        _event_mean_durations = W._event_mean_durations
        _get_sections = W._get_sections

        def __init__(self):
            self._event_table = table
            self._host = SimpleNamespace(
                _raw_bouts=raw_bouts,
                _summary_rows=summary_rows,
                _session_label_by_session=labels,
                _subject_by_session=subjects,
                _project_fps=lambda: FPS,
                _session_fps=lambda _sid: FPS,
                _analysis_prechop_for_session=lambda _sid: 0,
                _graphs_tab=SimpleNamespace(_session_analysis_end_s=lambda _sid: 400.0),
                _summary_tab=SimpleNamespace(_checked_subjects=lambda: set()),
            )

        def _events_mode(self):
            return True

        def _checked_behavior_ids(self):
            return list(raw_bouts)

    return _Stub()


def _bouts(spans):
    """spans: (session_id, start_s, end_s) at 10 fps, end_frame inclusive."""
    return pd.DataFrame({
        "session_id": [s for s, _a, _b in spans],
        "start_frame": [int(a * FPS) for _s, a, _b in spans],
        "end_frame": [int(b * FPS) - 1 for _s, _a, b in spans],
        "behavior": "Groom",
        "behavior_id": "g",
    })


class TestWidgetCompute:
    def test_each_subject_measured_inside_its_own_sections(self, widget_cls):
        table = parse_section_events(_file_df())
        # A bout at 160-170 s is Stim for both A (100-200 s) and B (150-250 s).
        # A bout at 120-130 s is Stim for A but Home Cage for B, whose stim
        # starts 50 s later.
        raw = {"g": _bouts([("1", 120, 130), ("1", 160, 170),
                            ("2", 120, 130), ("2", 160, 170)])}
        summary = [
            {"session_id": "1", "session_label": "A", "behavior_id": "g"},
            {"session_id": "2", "session_label": "B", "behavior_id": "g"},
            {"session_id": "3", "session_label": "NoEvents", "behavior_id": "g"},
        ]
        w = _stub(widget_cls, table, raw, summary, {"1": "A", "2": "B", "3": "NoEvents"})
        df = w._compute_section_data()

        def val(label, sec, col):
            r = df[(df["session_label"] == label) & (df["section_name"] == sec)]
            return r[col].item()

        assert val("A", "Home Cage", "n_bouts") == 0
        assert val("A", "Stim", "n_bouts") == 2
        assert val("B", "Home Cage", "n_bouts") == 1
        assert val("B", "Stim", "n_bouts") == 1
        # % time uses each subject's own section length.
        assert val("B", "Home Cage", "pct_time") == pytest.approx(10 / 150 * 100)
        assert val("A", "Stim", "rate_per_min") == pytest.approx(2 / (100 / 60))
        # Last section runs to the session end (400 s).
        assert val("A", "Post", "section_s") == pytest.approx(200.0)
        # A subject with no event times is left out, never zero-filled.
        assert "NoEvents" not in set(df["session_label"])

    def test_sections_list_uses_file_names_and_mean_lengths(self, widget_cls):
        table = parse_section_events(_file_df())
        w = _stub(widget_cls, table, {}, [], {"1": "A", "2": "B"})
        secs = w._get_sections()
        assert [n for n, _ in secs] == ["Home Cage", "Stim", "Post"]
        assert secs[0][1] == pytest.approx(125.0)


class TestTemplate:
    def test_template_round_trips_through_the_loader(self, tmp_path):
        from abel.services.section_events import write_section_events_template

        p = write_section_events_template(
            tmp_path / "t.xlsx", ["Home Cage", "Scruff"], ["DG01", "CAB01"],
        )
        t = load_section_events(p)
        assert t.section_names == ["Home Cage", "Scruff"]
        assert list(t.subjects) == ["DG01", "CAB01"]
        assert t.subjects["DG01"] == [(None, None), (None, None)]
        assert "Instructions" in pd.ExcelFile(p).sheet_names

    def test_no_subjects_gives_an_example_row_and_csv_works(self, tmp_path):
        from abel.services.section_events import write_section_events_template

        p = write_section_events_template(tmp_path / "t.csv", [], [])
        t = load_section_events(p)
        assert t.section_names == ["Baseline", "Stimulus", "Post"]
        assert t.subjects["Example01"][0] == (0.0, 9000.0)
