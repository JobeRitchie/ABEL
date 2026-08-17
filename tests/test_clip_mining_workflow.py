"""Targeted Clip Mining workflow: don't re-offer reviewed clips, spread the batch.

Three behaviours the mining loop depends on:

* matches that already carry a review decision are never handed back (and the
  check is live, so a clip judged while the modeless dialog is open drops out of
  the batch it is about to load);
* the capped batch takes turns between *subjects* rather than loading whichever
  animal happens to score highest;
* "Clear Unreviewed Clips" removes the mined windows themselves, not just their
  clip files — otherwise the mined queue keeps listing clips the user asked to
  clear.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import abel.ui.tabs.review_tab as review_tab_module  # noqa: E402
from abel.models.schemas import CandidateWindow, ReviewDecision  # noqa: E402
from abel.services.behavior_service import BehaviorService  # noqa: E402
from abel.services.candidate_service import CandidateGenerationService  # noqa: E402
from abel.services.clip_metrics_service import ClipRef  # noqa: E402
from abel.services.import_service import ImportService  # noqa: E402
from abel.services.review_service import ReviewService  # noqa: E402
from abel.ui.clip_mining_dialog import ClipMiningDialog  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")
    return app


def _dialog(tmp_path: Path, reviewed: set[str], applied: dict) -> ClipMiningDialog:
    """A mining dialog over a synthetic scored pool: 3 sessions × 6 windows.

    Sessions ``s1``/``s2`` are the same animal, ``s3`` another, so a batch balanced
    by *subject* must not treat the two ``s1``/``s2`` recordings as two subjects.
    Scores fall with session index, so an unbalanced cut drains ``s1`` first.
    """
    dlg = ClipMiningDialog(
        project_root=tmp_path,
        exemplar_provider=lambda: [],
        scope_label="test",
        on_apply=lambda refs, scores: applied.update(refs=refs, scores=scores),
        reviewed_provider=lambda: reviewed,
    )
    clips, rows = [], {}
    for si, sess in enumerate(["s1", "s2", "s3"]):
        for i in range(6):
            wid = f"{sess}_w{i}"
            clips.append(ClipRef(wid, sess, i * 30, i * 30 + 29))
            rows[wid] = {"centroid_speed_mean": 100.0 - si * 10 - i}
    dlg._clip_by_id = {c.window_id: c for c in clips}
    dlg._df = pd.DataFrame.from_dict(rows, orient="index")
    dlg._scored = True
    dlg._metrics._subject_by_session = {"s1": "mouseA", "s2": "mouseA", "s3": "mouseB"}
    dlg._clear_rows()
    dlg._add_row("centroid_speed_mean").set_range(0.0, None)  # matches everything
    dlg._update_count()
    return dlg


def _subject(dlg: ClipMiningDialog, window_id: str) -> str:
    return dlg._metrics.subject_for_session(dlg._clip_by_id[window_id].session_id)


def test_reviewed_clips_are_not_surfaced(_app, tmp_path) -> None:
    reviewed = {"s1_w0", "s1_w1", "s2_w0"}
    dlg = _dialog(tmp_path, reviewed, {})

    assert len(dlg._last_matches) == 15
    assert reviewed.isdisjoint(dlg._last_matches)
    assert dlg._skipped_reviewed == 3
    assert "already-reviewed" in dlg._count_label.text()


def test_unticking_the_skip_restores_reviewed_matches(_app, tmp_path) -> None:
    dlg = _dialog(tmp_path, {"s1_w0", "s1_w1", "s2_w0"}, {})

    dlg._skip_reviewed_chk.setChecked(False)

    assert len(dlg._last_matches) == 18
    assert dlg._skipped_reviewed == 0


def test_batch_is_spread_across_subjects(_app, tmp_path) -> None:
    dlg = _dialog(tmp_path, set(), {})

    picked = dlg._select_for_load(dlg._last_matches, 6)

    assert len(picked) == 6
    per_subject = {s: 0 for s in ("mouseA", "mouseB")}
    for wid in picked:
        per_subject[_subject(dlg, wid)] += 1
    assert per_subject == {"mouseA": 3, "mouseB": 3}


def test_unbalanced_batch_drains_the_top_scoring_session(_app, tmp_path) -> None:
    """Without the toggle it is a plain top-N cut — the behaviour being fixed."""
    dlg = _dialog(tmp_path, set(), {})
    dlg._balance_subjects_chk.setChecked(False)

    picked = dlg._select_for_load(dlg._last_matches, 6)

    assert all(_subject(dlg, w) == "mouseA" for w in picked)


def test_apply_drops_clips_reviewed_since_the_last_count(_app, tmp_path) -> None:
    """The dialog is modeless, so the batch is re-checked at Load time."""
    reviewed: set[str] = set()
    applied: dict = {}
    dlg = _dialog(tmp_path, reviewed, applied)
    dlg._cap_spin.setValue(6)

    reviewed.add("s3_w0")  # reviewed behind the open dialog
    dlg._apply()

    loaded = [r.window_id for r in applied["refs"]]
    assert loaded and "s3_w0" not in loaded


def _review_tab_with_mined_windows(tmp_path: Path):
    """A Review tab over a project holding 3 mined windows (one reviewed) + 1 AL window."""
    (tmp_path / "derived" / "review_tables").mkdir(parents=True, exist_ok=True)
    clips_dir = tmp_path / "derived" / "clips" / "s1"
    clips_dir.mkdir(parents=True, exist_ok=True)

    candidates = CandidateGenerationService()
    candidates.set_project(tmp_path)
    review = ReviewService()
    review.set_project(tmp_path)
    behaviors = BehaviorService()
    behaviors.set_project(tmp_path)

    windows = [
        CandidateWindow(
            window_id=f"m{i}", session_id="s1", start_frame=i * 30,
            end_frame=i * 30 + 29, source="clip_mining",
        )
        for i in range(3)
    ] + [
        CandidateWindow(
            window_id="al0", session_id="s1", start_frame=900, end_frame=929,
            source="active_learning_uncertainty",
        )
    ]
    candidates.upsert_external_window_candidates(windows)
    for w in windows:
        (clips_dir / f"{w.window_id}.mp4").write_bytes(b"stub")
    review.save_decisions([
        ReviewDecision(
            decision_id="d0", clip_id="m0", reviewer="tester",
            old_status="pending", new_status="accepted", decision="accept",
        )
    ])

    tab = review_tab_module.ReviewTab(review, candidates, ImportService(), behaviors)
    tab._project_root = tmp_path
    tab._refresh_candidates()
    tab._pending_mined_ids = {"m0", "m1", "m2"}
    tab._pending_mined_scores = {"m0": 1.0, "m1": 0.9, "m2": 0.8}
    tab._finalize_mined_view()
    return tab, candidates, clips_dir


def test_clear_unreviewed_clips_removes_mined_windows(_app, tmp_path, monkeypatch) -> None:
    tab, candidates, clips_dir = _review_tab_with_mined_windows(tmp_path)
    monkeypatch.setattr(
        review_tab_module.QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(
        review_tab_module.QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )

    tab._clear_unreviewed_clips()

    # Clip files: only the reviewed mined window keeps its clip.
    assert sorted(p.name for p in clips_dir.glob("*.mp4")) == ["m0.mp4"]
    # Candidate rows: unreviewed mined windows are gone; the reviewed one and the
    # active-learning window (a different source) stay.
    assert sorted(c.window_id for c in candidates.load_external_window_candidates()) == [
        "al0", "m0",
    ]
    # The mining filter forgets the windows it no longer has.
    assert tab._mined_ids == {"m0"}


def test_clearing_every_mined_window_drops_the_mining_filter(_app, tmp_path, monkeypatch) -> None:
    tab, _candidates, _clips = _review_tab_with_mined_windows(tmp_path)
    # No decisions at all → all three mined windows are unreviewed.
    tab._decision_by_clip_id = {}
    monkeypatch.setattr(
        review_tab_module.QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(
        review_tab_module.QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )

    tab._clear_unreviewed_clips()

    assert tab._mined_ids is None  # back to the normal review queue
