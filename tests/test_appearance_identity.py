"""Telling two tracks apart by coat brightness."""

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from abel.services.appearance_identity_service import (  # noqa: E402
    analyze_appearance_identity,
)
from abel.services.pose_processing_service import (  # noqa: E402
    MultiAnimalPoseData,
    PoseData,
)

W, H, N = 160, 80, 300
DARK, LIGHT = 20, 230


def _video(path, swap_at=None, same_colour=False):
    """Two blobs, one dark one light, optionally exchanging places mid-video."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert writer.isOpened()
    for i in range(N):
        frame = np.full((H, W, 3), 128, dtype=np.uint8)
        left, right = (DARK, LIGHT if not same_colour else DARK)
        if swap_at is not None and i >= swap_at:
            left, right = right, left
        cv2.circle(frame, (40, 40), 10, (left,) * 3, -1)
        cv2.circle(frame, (120, 40), 10, (right,) * 3, -1)
        writer.write(frame)
    writer.release()
    return path


def _multi():
    """Track A sits on the left blob, track B on the right, all session long."""
    per = {}
    for name, x in (("track_0", 40.0), ("track_1", 120.0)):
        per[name] = PoseData(
            body_parts=["nose"],
            x=pd.DataFrame({"nose": [x] * N}),
            y=pd.DataFrame({"nose": [40.0] * N}),
            likelihood=pd.DataFrame({"nose": [1.0] * N}),
            centroid_x=np.full(N, x), centroid_y=np.full(N, 40.0), n_frames=N,
        )
    return MultiAnimalPoseData(individuals=list(per), per_individual=per, n_frames=N)


def test_a_sustained_colour_exchange_is_reported_as_a_swap(tmp_path):
    res = analyze_appearance_identity(
        _video(tmp_path / "v.mp4", swap_at=150), _multi(), sample_every=5, min_run_samples=4,
    )
    assert res.usable
    assert len(res.corrections) == 1
    assert abs(res.corrections[0]["frame"] - 150) <= 10
    assert res.corrections[0]["a"] == "track_0" and res.corrections[0]["b"] == "track_1"
    assert res.consistency_after > res.consistency_before


def test_a_clean_session_gets_no_corrections(tmp_path):
    res = analyze_appearance_identity(
        _video(tmp_path / "v.mp4"), _multi(), sample_every=5, min_run_samples=4,
    )
    assert res.usable and res.corrections == []
    assert res.consistency_before == pytest.approx(1.0)


def test_lookalike_animals_are_refused_rather_than_guessed(tmp_path):
    res = analyze_appearance_identity(
        _video(tmp_path / "v.mp4", swap_at=150, same_colour=True), _multi(),
        sample_every=5, min_run_samples=4,
    )
    assert not res.usable
    assert res.corrections == []
    assert "look too similar" in res.message


def test_single_animal_sessions_are_refused(tmp_path):
    multi = _multi()
    one = MultiAnimalPoseData(
        individuals=["track_0"],
        per_individual={"track_0": multi.per_individual["track_0"]},
        n_frames=N,
    )
    res = analyze_appearance_identity(_video(tmp_path / "v.mp4"), one, sample_every=5)
    assert not res.usable and "two tracked animals" in res.message


def test_a_missing_video_is_reported_not_raised(tmp_path):
    res = analyze_appearance_identity(tmp_path / "nope.mp4", _multi())
    assert not res.usable and "Could not open" in res.message


# ── Batch scan ────────────────────────────────────────────────────────
def test_correction_sets_that_cancel_count_as_the_same_assignment():
    from abel.services.appearance_identity_service import _same_assignment

    # Two extra flips a few frames apart cancel out, so the assignment matches.
    assert _same_assignment([100, 500, 505], [100])
    # A genuinely different parity does not.
    assert not _same_assignment([100, 500], [100])
    # Slack for the sampling step.
    assert _same_assignment([100], [110])
    assert _same_assignment([], [])


def test_scan_reports_one_row_per_multi_animal_session(tmp_path):
    from abel.services.appearance_identity_service import scan_sessions

    class _Session:
        session_id = "s1"
        subject_id = "M1"
        individuals = ["track_0", "track_1"]
        identity_corrections = [{"frame": 150, "a": "track_0", "b": "track_1"}]

    class _Single(_Session):
        session_id = "s2"
        individuals = []

    class _Manifest:
        linked_sessions = [_Session(), _Single()]
        smoothing_settings = None

    class _Imports:
        def video_path_for_session(self, _m, _sid):
            return _video(tmp_path / "v.mp4", swap_at=150)

        def pose_path_for_session(self, _m, _sid):
            return tmp_path / "pose.h5"

    class _Poses:
        def load_and_clean_multi(self, _path, _settings=None):
            return _multi()

    rows = scan_sessions(
        _Manifest(), _Imports(), _Poses(), sample_every=5, min_run_samples=4,
    )
    assert len(rows) == 1  # the single-animal session is not scanned
    row = rows[0]
    assert row["subject"] == "M1" and row["usable"]
    assert row["agrees"] is True  # detected flip matches the saved correction


# ── Project probe ─────────────────────────────────────────────────────
def _probe_fixture(tmp_path, same_colour):
    from abel.services.appearance_identity_service import probe_project_appearance

    class _Session:
        session_id = "s1"
        subject_id = "M1"
        individuals = ["track_0", "track_1"]
        identity_corrections = []

    class _Manifest:
        linked_sessions = [_Session()]
        smoothing_settings = None

    class _Imports:
        def video_path_for_session(self, _m, _sid):
            return _video(tmp_path / "v.mp4", same_colour=same_colour)

        def pose_path_for_session(self, _m, _sid):
            return tmp_path / "pose.h5"

    class _Poses:
        def load_and_clean_multi(self, _path, _settings=None):
            return _multi()

    return probe_project_appearance(
        _Manifest(), _Imports(), _Poses(), sample_every=10,
    )


def test_probe_says_appearance_works_for_distinct_animals(tmp_path):
    out = _probe_fixture(tmp_path, same_colour=False)
    assert out["verdict"] == "usable"
    assert out["usable_fraction"] == 1.0
    assert out["median_contrast"] > 100


def test_probe_rules_appearance_out_for_lookalikes(tmp_path):
    out = _probe_fixture(tmp_path, same_colour=True)
    assert out["verdict"] == "unusable"
    assert "by eye" in out["message"]


def test_probe_handles_a_project_with_no_multi_animal_sessions():
    from abel.services.appearance_identity_service import probe_project_appearance

    class _Manifest:
        linked_sessions = []
        smoothing_settings = None

    out = probe_project_appearance(_Manifest(), object(), object())
    assert out["verdict"] == "none" and out["sessions"] == []
