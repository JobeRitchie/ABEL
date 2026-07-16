"""Regression test: "Send bouts for current behavior/subject to Clip Review".

The trace-behavior dropdown stores probability *column names* ("prob_<behavior_id>"),
but bout collection filters on *behavior IDs*.  The send handler used to pass the raw
column name straight through as a behavior filter, so it never matched any behavior and
the user always got "No bouts found for subject 'X' / prob_<uuid>".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.tabs.temporal_review_tab import CandidateWindow, TemporalReviewTab  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")
    return app


@dataclass
class _Behavior:
    behavior_id: str
    name: str


class _Behaviors:
    def __init__(self, behaviors: list[_Behavior]) -> None:
        self.behaviors = behaviors


BID = "3954934d-8fbf-41f6-9c9d-9b9dd3efb9c7"


def _tab(_app) -> TemporalReviewTab:
    tab = TemporalReviewTab()
    tab._behaviors = _Behaviors([_Behavior(behavior_id=BID, name="Rear")])
    return tab


def test_behavior_id_resolved_from_prob_column(_app) -> None:
    tab = _tab(_app)
    assert tab._behavior_id_from_col(f"prob_{BID}") == BID
    assert tab._behavior_id_from_col("prob_Rear") == BID
    assert tab._behavior_id_from_col("__all__") is None
    assert tab._behavior_id_from_col("prob_no_behavior") is None
    assert tab._behavior_id_from_col("prob_not_a_behavior") is None


def test_on_the_fly_bout_collection_does_not_crash(_app, tmp_path: Path) -> None:
    """The on-the-fly path called threshold_probabilities() with a stale 3rd arg.

    That raised TypeError inside the button's slot, so the click did nothing at all.
    Behaviors with no pre-computed postprocess parquet always land on this path.
    """
    import numpy as np
    import pandas as pd

    n = 600
    prob = np.zeros(n, dtype=float)
    prob[100:200] = 0.95  # one unambiguous bout
    prob[400:480] = 0.90  # and a second
    trace = tmp_path / "s1_trace.parquet"
    pd.DataFrame({"frame": np.arange(n), f"prob_{BID}": prob}).to_parquet(trace)

    tab = _tab(_app)
    tab._project_root = tmp_path
    tab._trace_paths = {"s1": str(trace)}
    tab._bout_paths_by_behavior = {}   # no precomputed parquets -> forces on-the-fly
    tab._bout_paths = {}

    candidates = tab._collect_all_bout_candidates(behavior_filter={BID})

    assert candidates, "on-the-fly bout collection produced no candidates"
    assert all(c.behavior_id == BID for c in candidates)
    assert all(c.session_id == "s1" for c in candidates)


def test_send_current_subject_filters_by_behavior_id_not_column(_app, tmp_path: Path) -> None:
    tab = _tab(_app)
    tab._project_root = tmp_path
    tab._trace_paths = {"s1": str(tmp_path / "s1.parquet")}
    tab._subject_by_session = {"s1": "CBMRE01"}

    tab._session.addItem("s1", userData="s1")
    tab._session.setCurrentIndex(tab._session.findData("s1"))
    tab._trace_behavior.addItem("Rear", userData=f"prob_{BID}")
    tab._trace_behavior.setCurrentIndex(tab._trace_behavior.findData(f"prob_{BID}"))

    seen_filters: list[set[str] | None] = []

    def _fake_collect(behavior_filter=None):
        seen_filters.append(behavior_filter)
        return [CandidateWindow(
            window_id="w1", session_id="s1", start_frame=0, end_frame=59,
            behavior_id=BID, total_score=0.9, source="temporal_bout_review",
        )]

    tab._collect_all_bout_candidates = _fake_collect  # type: ignore[method-assign]

    emitted: list[tuple[list, str]] = []
    tab.bout_candidates_append_requested.connect(
        lambda cands, label: emitted.append((cands, label))
    )

    tab._send_current_subject_behavior_bouts_to_clip_review()

    # The filter must carry the behavior ID, never the raw "prob_<id>" column name.
    assert seen_filters == [{BID}]
    assert len(emitted) == 1
    cands, label = emitted[0]
    assert [c.window_id for c in cands] == ["w1"]
    assert "CBMRE01" in label
