"""Coverage-gap clip mining: aim a hunt at subjects lacking examples of a behavior.

The coverage count must follow the LOSO "mice" unit (the session's subject, so
both tracks of a dyad are one subject), read the reviewer's live labels, split
co-occurring labels, and stop asking for a subject once enough gap-mined clips
were reviewed without a single example.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from abel.services.behavior_coverage_service import BehaviorCoverageService

DOM = "dom-uuid"
GROOM = "groom-uuid"


def _write_labels(root: Path, rows: list[tuple[str, str]]) -> None:
    path = root / "derived" / "review_labels" / "reviewer_labels.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["segment_id", "review_label"]).to_parquet(path)


def _service(root: Path, sessions: dict[str, str], segments: dict[str, str]) -> BehaviorCoverageService:
    svc = BehaviorCoverageService(root)
    svc._subject_by_session = dict(sessions)
    svc._session_by_segment = dict(segments)
    return svc


# Three subjects: A owns two sessions, B and C one each.
SESSIONS = {"s1": "A", "s2": "A", "s3": "B", "s4": "C"}
SEGMENTS = {
    "t0_s1_1": "s1", "t1_s1_1": "s1", "t0_s2_1": "s2",
    "t0_s3_1": "s3", "t1_s3_1": "s3",
    **{f"t0_s4_{i}": "s4" for i in range(40)},
}


def test_positives_count_per_subject_with_dyads_and_pipes(tmp_path) -> None:
    _write_labels(tmp_path, [
        ("t0_s1_1", DOM),
        ("t1_s1_1", DOM),            # partner track: same subject (session's)
        ("t0_s2_1", f"{GROOM}|{DOM}"),  # co-occurring label counts for both
        ("t0_s3_1", GROOM),
        ("unknown_seg", DOM),        # no session: ignored, not a crash
    ])
    rep = _service(tmp_path, SESSIONS, SEGMENTS).report(DOM, min_positives=3)

    assert rep.subjects["A"].positives == 3
    assert rep.subjects["B"].positives == 0
    assert rep.subjects["C"].positives == 0
    assert rep.gap_subjects() == {"B", "C"}
    assert rep.gap_sessions() == {"s3", "s4"}
    assert rep.n_covered() == 1


def test_threshold_moves_subjects_in_and_out(tmp_path) -> None:
    _write_labels(tmp_path, [("t0_s1_1", DOM), ("t1_s1_1", DOM), ("t0_s2_1", DOM)])
    svc = _service(tmp_path, SESSIONS, SEGMENTS)
    assert "A" not in svc.report(DOM, min_positives=3).gap_subjects()
    assert "A" in svc.report(DOM, min_positives=4).gap_subjects()


def test_subject_screened_without_a_hit_is_likely_absent(tmp_path) -> None:
    svc = _service(tmp_path, SESSIONS, SEGMENTS)
    mined = [f"t0_s4_{i}" for i in range(30)]
    assert svc.record_mined(DOM, mined) == 30
    # Logging again is idempotent.
    assert svc.record_mined(DOM, mined) == 0

    # Loaded but not yet reviewed: not screening evidence.
    _write_labels(tmp_path, [])
    rep = svc.report(DOM, min_positives=3, absent_after=30)
    assert rep.subjects["C"].screened == 0
    assert "C" in rep.gap_subjects()

    # All 30 reviewed, none showed the behavior.
    _write_labels(tmp_path, [(w, "no_behavior") for w in mined])
    rep = svc.report(DOM, min_positives=3, absent_after=30)
    assert rep.subjects["C"].screened == 30
    assert rep.likely_absent() == {"C"}
    assert "C" not in rep.gap_subjects()
    # The log is per behavior: another behavior still sees C as a gap.
    assert "C" in svc.report(GROOM, min_positives=3, absent_after=30).gap_subjects()


def test_one_example_keeps_a_subject_in_the_gap_set(tmp_path) -> None:
    """A subject that did it once does the behavior; misses never retire it."""
    svc = _service(tmp_path, SESSIONS, SEGMENTS)
    mined = [f"t0_s4_{i}" for i in range(35)]
    svc.record_mined(DOM, mined)
    rows = [(w, "no_behavior") for w in mined[:34]] + [("t0_s4_39", DOM)]
    _write_labels(tmp_path, rows)
    rep = svc.report(DOM, min_positives=3, absent_after=30)
    assert rep.subjects["C"].positives == 1
    assert rep.likely_absent() == set()
    assert "C" in rep.gap_subjects()


def test_screening_history_survives_a_subject_rename(tmp_path) -> None:
    svc = _service(tmp_path, SESSIONS, SEGMENTS)
    mined = [f"t0_s4_{i}" for i in range(30)]
    svc.record_mined(DOM, mined)
    _write_labels(tmp_path, [(w, "no_behavior") for w in mined])

    renamed = _service(tmp_path, {**SESSIONS, "s4": "C_renamed"}, SEGMENTS)
    rep = renamed.report(DOM, min_positives=3, absent_after=30)
    assert rep.subjects["C_renamed"].screened == 30
    assert rep.likely_absent() == {"C_renamed"}


def test_dominant_behavior_ignores_no_behavior(tmp_path) -> None:
    _write_labels(tmp_path, [
        ("t0_s1_1", DOM), ("t1_s1_1", DOM), ("t0_s2_1", GROOM),
        ("t0_s3_1", "no_behavior"), ("t1_s3_1", "no_behavior"),
    ])
    svc = _service(tmp_path, SESSIONS, SEGMENTS)
    picks = ["t0_s1_1", "t1_s1_1", "t0_s2_1", "t0_s3_1", "t1_s3_1"]
    assert svc.dominant_behavior(picks) == DOM
    assert svc.dominant_behavior([]) is None


# -- dialog ------------------------------------------------------------------

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.services.clip_metrics_service import ClipRef  # noqa: E402
from abel.ui.clip_mining_dialog import ClipMiningDialog  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")
    return app


def _dialog(tmp_path: Path, applied: dict, exemplars: list[ClipRef] | None = None) -> ClipMiningDialog:
    """A dialog over a scored pool: s1/s2 = mouse A, s3 = B, s4 = C, 6 windows each."""
    dlg = ClipMiningDialog(
        project_root=tmp_path,
        exemplar_provider=lambda: list(exemplars or []),
        scope_label="test",
        on_apply=lambda refs, scores: applied.update(refs=refs, scores=scores),
        reviewed_provider=lambda: set(),
    )
    clips, rows, segs = [], {}, {}
    for si, sess in enumerate(["s1", "s2", "s3", "s4"]):
        for i in range(6):
            wid = f"{sess}_w{i}"
            clips.append(ClipRef(wid, sess, i * 30, i * 30 + 29))
            rows[wid] = {"centroid_speed_mean": 100.0 - si * 10 - i}
            segs[wid] = sess
    dlg._clip_by_id = {c.window_id: c for c in clips}
    dlg._df = pd.DataFrame.from_dict(rows, orient="index")
    dlg._scored = True
    subjects = {"s1": "A", "s2": "A", "s3": "B", "s4": "C"}
    dlg._metrics._subject_by_session = dict(subjects)
    dlg._coverage._subject_by_session = dict(subjects)
    dlg._coverage._session_by_segment = segs
    # A tmp project has no behaviors of its own.
    dlg._gap_behavior.clear()
    dlg._behavior_names = {GROOM: "Groom", DOM: "Dominate"}
    dlg._gap_behavior.addItem("Groom", GROOM)
    dlg._gap_behavior.addItem("Dominate", DOM)
    dlg._gap_chk.setEnabled(True)
    dlg._clear_rows()
    dlg._add_row("centroid_speed_mean").set_range(0.0, None)  # matches everything
    dlg._update_count()
    return dlg


def _subjects_of(dlg: ClipMiningDialog, ids: list[str]) -> set[str]:
    return {dlg._metrics.subject_for_session(dlg._clip_by_id[w].session_id) for w in ids}


def test_gap_toggle_keeps_only_subjects_lacking_examples(_app, tmp_path) -> None:
    # A has 3 Dominate examples, B has 1, C has none.
    _write_labels(tmp_path, [
        ("s1_w0", DOM), ("s1_w1", DOM), ("s2_w0", DOM), ("s3_w0", DOM),
    ])
    dlg = _dialog(tmp_path, {})
    assert len(dlg._last_matches) == 24

    dlg._gap_behavior.setCurrentIndex(dlg._gap_behavior.findData(DOM))
    dlg._gap_chk.setChecked(True)

    assert _subjects_of(dlg, dlg._last_matches) == {"B", "C"}
    assert dlg._skipped_gap == 12
    assert "Dominate: 2 of 3 subjects" in dlg._gap_label.text()
    assert "Limited to 2 subject(s)" in dlg._count_label.text()

    # Raising the bar pulls A back in.
    dlg._gap_min.setValue(4)
    assert _subjects_of(dlg, dlg._last_matches) == {"A", "B", "C"}

    dlg._gap_chk.setChecked(False)
    assert len(dlg._last_matches) == 24
    assert not dlg._gap_label.isVisibleTo(dlg)


def test_gap_coverage_is_read_live(_app, tmp_path) -> None:
    """Labels written behind the modeless dialog change the gap set on the next count."""
    _write_labels(tmp_path, [("s1_w0", DOM), ("s1_w1", DOM), ("s2_w0", DOM)])
    dlg = _dialog(tmp_path, {})
    dlg._gap_behavior.setCurrentIndex(dlg._gap_behavior.findData(DOM))
    dlg._gap_chk.setChecked(True)
    assert _subjects_of(dlg, dlg._last_matches) == {"B", "C"}

    _write_labels(tmp_path, [
        ("s1_w0", DOM), ("s1_w1", DOM), ("s2_w0", DOM),
        ("s3_w0", DOM), ("s3_w1", DOM), ("s3_w2", DOM),
    ])
    dlg._update_count()
    assert _subjects_of(dlg, dlg._last_matches) == {"C"}


def test_toggle_defaults_to_the_exemplars_behavior(_app, tmp_path) -> None:
    _write_labels(tmp_path, [("s1_w0", DOM), ("s1_w1", DOM)])
    exemplars = [ClipRef("s1_w0", "s1", 0, 29), ClipRef("s1_w1", "s1", 30, 59)]
    dlg = _dialog(tmp_path, {}, exemplars)
    assert dlg._gap_behavior_id() == GROOM

    dlg._gap_chk.setChecked(True)

    assert dlg._gap_behavior_id() == DOM


def test_loading_a_gap_batch_logs_it_for_screening(_app, tmp_path) -> None:
    _write_labels(tmp_path, [("s1_w0", DOM), ("s1_w1", DOM), ("s2_w0", DOM)])
    applied: dict = {}
    dlg = _dialog(tmp_path, applied)
    dlg._gap_behavior.setCurrentIndex(dlg._gap_behavior.findData(DOM))
    dlg._gap_chk.setChecked(True)
    dlg._cap_spin.setValue(4)

    dlg._apply()

    loaded = [r.window_id for r in applied["refs"]]
    assert len(loaded) == 4
    assert _subjects_of(dlg, loaded) == {"B", "C"}
    log = BehaviorCoverageService(tmp_path)._load_log()
    assert set(log[DOM]) == set(loaded)


def test_plain_load_does_not_log(_app, tmp_path) -> None:
    _write_labels(tmp_path, [])
    applied: dict = {}
    dlg = _dialog(tmp_path, applied)
    dlg._cap_spin.setValue(4)
    dlg._apply()
    assert applied["refs"]
    assert BehaviorCoverageService(tmp_path)._load_log() == {}
