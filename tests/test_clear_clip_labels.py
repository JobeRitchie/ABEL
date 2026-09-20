"""Clearing a reviewed clip's labels: review-tab logic and the soundboard button.

A mislabeled clip has to be recoverable, the decision, the per-animal reviewer
label rows and the soundboard's round-trip payload all have to go, while the
candidate itself stays in the queue as unreviewed and neighboring clips keep
their labels.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from abel.models.schemas import ReviewDecisionType, ReviewerLabelRecord
from abel.services.behavior_service import NO_BEHAVIOR_ID
from abel.services.review_service import ReviewService
from abel.ui.behavior_soundboard import BehaviorSoundboard
from abel.ui.tabs.review_tab import ReviewTab


ANIMALS = [("sub_a:track_0", "black", (200, 0, 0)), ("sub_a:track_1", "green", (0, 200, 0))]


class _Candidate:
    def __init__(self, window_id: str, session_id: str, start: int, end: int) -> None:
        self.window_id = window_id
        self.session_id = session_id
        self.start_frame = start
        self.end_frame = end


class _Clearer:
    """The clearing methods without building a Qt widget."""

    _clear_label_segment_ids = ReviewTab._clear_label_segment_ids
    _clear_labels_for = ReviewTab._clear_labels_for

    def __init__(self, root: Path, animals=ANIMALS) -> None:
        self._project_root = root
        self._review_service = ReviewService()
        self._review_service.set_project(root)
        self._decision_by_clip_id: dict = {}
        self._structured_labels: dict = {}
        self._animals = list(animals)

    def _clip_animals_for(self, _cand):
        return self._animals


def _label_clip(rs: ReviewService, cand: _Candidate, animals=ANIMALS) -> None:
    """Review a clip the way the soundboard does: decision + per-animal rows."""
    rs.upsert_decision(
        clip_id=cand.window_id, reviewer="me", decision=ReviewDecisionType.ACCEPT,
        behavior_label="rearing", adjusted_start_frame=cand.start_frame,
        adjusted_end_frame=cand.end_frame,
    )
    for aid, _name, _rgb in animals:
        rs.append_segment_label(
            ReviewerLabelRecord(
                segment_id=f"seg_{aid}_{cand.session_id}_{cand.start_frame}_{cand.end_frame}",
                review_label="rearing", reviewer_id="me", notes="soundboard",
            )
        )
    rs.save_structured_labels(
        cand.window_id,
        [{"behavior_id": "rearing", "focal_animal_id": aid, "partner_animal_id": None}
         for aid, _n, _c in animals],
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    pytest.importorskip("pyarrow")
    return tmp_path


def test_clearing_removes_decision_labels_and_payload(project: Path) -> None:
    c = _Clearer(project)
    cand = _Candidate("win1", "sess_a", 100, 132)
    _label_clip(c._review_service, cand)

    removed = c._clear_labels_for([cand])

    assert removed == 3  # two animal-segment rows + the decision
    assert c._review_service.load_decisions() == []
    assert c._review_service.load_segment_labels() == []
    assert c._review_service.get_structured_labels("win1") == []


def test_clearing_leaves_other_clips_alone(project: Path) -> None:
    c = _Clearer(project)
    kept = _Candidate("win2", "sess_a", 200, 232)
    _label_clip(c._review_service, _Candidate("win1", "sess_a", 100, 132))
    _label_clip(c._review_service, kept)

    c._clear_labels_for([_Candidate("win1", "sess_a", 100, 132)])

    assert [d.clip_id for d in c._review_service.load_decisions()] == ["win2"]
    assert {r.segment_id for r in c._review_service.load_segment_labels()} == {
        f"seg_{aid}_sess_a_200_232" for aid, _n, _c in ANIMALS
    }
    assert len(c._review_service.get_structured_labels("win2")) == 2


def test_clearing_reaches_animals_only_named_in_the_saved_payload(project: Path) -> None:
    """An animal renamed out of the manifest still owns label rows."""
    c = _Clearer(project, animals=[])   # manifest no longer lists the individuals
    cand = _Candidate("win1", "sess_a", 100, 132)
    _label_clip(c._review_service, cand)

    ids = c._clear_label_segment_ids(cand)
    assert ids == {"win1"} | {f"seg_{aid}_sess_a_100_132" for aid, _n, _c in ANIMALS}

    c._clear_labels_for([cand])
    assert c._review_service.load_segment_labels() == []


def test_clearing_a_never_reviewed_clip_is_a_no_op(project: Path) -> None:
    c = _Clearer(project)
    assert c._clear_labels_for([_Candidate("win9", "sess_a", 1, 33)]) == 0
    assert c._clear_labels_for([]) == 0


def test_single_animal_clip_rows_are_keyed_by_window_id(project: Path) -> None:
    c = _Clearer(project, animals=[])
    cand = _Candidate("win1", "sess_a", 10, 42)
    rs = c._review_service
    rs.upsert_decision(clip_id="win1", reviewer="me", decision=ReviewDecisionType.ACCEPT,
                       behavior_label="rearing")
    rs.append_segment_label(
        ReviewerLabelRecord(segment_id="win1", review_label="rearing", reviewer_id="me")
    )

    assert c._clear_labels_for([cand]) == 2
    assert rs.load_segment_labels() == []
    assert rs.load_decisions() == []


# --- soundboard button -------------------------------------------------------


def _board(on_clear):
    QApplication.instance() or QApplication([])
    sb = BehaviorSoundboard()
    sb.configure(
        [("rearing", "Rearing", "r", False, "none"),
         (NO_BEHAVIOR_ID, "No Behavior", "n", False, "none")],
        lambda _b: None, {}, on_clear=on_clear,
    )
    return sb


def test_soundboard_clear_drops_staged_chips_and_calls_back():
    calls = []
    sb = _board(lambda: (calls.append(1), 3)[1])
    sb.set_animals(ANIMALS)
    sb._on_animal_clicked("sub_a:track_0")
    sb._on_behavior_clicked("rearing")
    assert sb._clip_labels

    sb._clear_clip_labels()

    assert sb._clip_labels == []
    assert len(calls) == 1
    assert "Cleared 3 saved labels" in sb._status.text()


def test_soundboard_clear_reports_when_nothing_was_stored():
    sb = _board(lambda: 0)
    sb.set_animals(ANIMALS)
    sb._clear_clip_labels()
    assert "Nothing to clear" in sb._status.text()

    sb._on_animal_clicked("sub_a:track_1")
    sb._on_behavior_clicked("rearing")
    sb._clear_clip_labels()
    assert "Discarded 1 staged label" in sb._status.text()


def test_soundboard_clear_survives_a_failing_callback():
    def boom():
        raise RuntimeError("parquet is locked")

    sb = _board(boom)
    sb.set_animals(ANIMALS)
    sb._on_animal_clicked("sub_a:track_0")
    sb._on_behavior_clicked("rearing")
    sb._clear_clip_labels()
    assert sb._clip_labels == []  # the window stays usable
