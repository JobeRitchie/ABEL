"""Feature prep fails fast (and legibly) when session inputs are missing.

Regression: a session registered against a video that no longer existed used to
crash mid-run inside a worker thread with "Unable to open video file", after the
other sessions had already been processed.  Pose extraction never touched the
video, so the breakage stayed hidden until the feature cache was cleared.
"""

from __future__ import annotations

import traceback

import pytest

from abel.services.feature_prep_service import (
    FeaturePrepService,
    PrepConfig,
    PrepInputError,
    SessionJob,
)
from abel.utils.error_text import exception_message, format_task_error


@pytest.fixture()
def healthy_job(tmp_path):
    pose = tmp_path / "s1.csv"
    pose.write_text("x", encoding="utf-8")
    video = tmp_path / "s1.avi"
    video.write_bytes(b"\x00")
    return SessionJob("session_1", "M01", pose, video, 30.0)


def test_preflight_passes_when_inputs_exist(healthy_job):
    FeaturePrepService._preflight_inputs([healthy_job], PrepConfig(use_video_features=True))


def test_missing_video_raises_before_any_work(healthy_job, tmp_path):
    gone = SessionJob("session_2", "M02", healthy_job.pose_path, tmp_path / "gone.avi", 30.0)
    with pytest.raises(PrepInputError) as exc:
        FeaturePrepService._preflight_inputs([healthy_job, gone], PrepConfig(use_video_features=True))
    assert "gone.avi" in str(exc.value)
    assert "M02" in str(exc.value)


def test_missing_video_ignored_when_video_context_off(healthy_job, tmp_path):
    gone = SessionJob("session_2", "M02", healthy_job.pose_path, tmp_path / "gone.avi", 30.0)
    FeaturePrepService._preflight_inputs([healthy_job, gone], PrepConfig(use_video_features=False))


def test_missing_pose_raises_regardless_of_video_context(healthy_job, tmp_path):
    gone = SessionJob("session_2", "M02", tmp_path / "gone.csv", healthy_job.video_path, 30.0)
    with pytest.raises(PrepInputError, match="pose file not found"):
        FeaturePrepService._preflight_inputs([gone], PrepConfig(use_video_features=False))


def test_unlinked_video_raises_when_video_context_on(healthy_job):
    gone = SessionJob("session_2", "M02", healthy_job.pose_path, None, 30.0)
    with pytest.raises(PrepInputError, match="no video linked"):
        FeaturePrepService._preflight_inputs([gone], PrepConfig(use_video_features=True))


def test_every_missing_input_is_reported_not_just_the_first(healthy_job, tmp_path):
    """The whole point of the pre-flight: one pass, one complete list."""
    bad = [
        SessionJob(f"session_{i}", f"M{i:02d}", healthy_job.pose_path, tmp_path / f"gone{i}.avi", 30.0)
        for i in range(5)
    ]
    with pytest.raises(PrepInputError) as exc:
        FeaturePrepService._preflight_inputs([healthy_job, *bad], PrepConfig(use_video_features=True))
    msg = str(exc.value)
    assert msg.startswith("5 session input file(s) are missing")
    for i in range(5):
        assert f"gone{i}.avi" in msg


def test_error_message_survives_traceback_formatting(healthy_job, tmp_path):
    """The UI used to show tb[:600], which cut the message off the tail."""
    gone = SessionJob("session_2", "M02", healthy_job.pose_path, tmp_path / "gone.avi", 30.0)
    try:
        FeaturePrepService._preflight_inputs([gone], PrepConfig(use_video_features=True))
    except PrepInputError:
        tb = traceback.format_exc()
    assert "gone.avi" not in tb[:600], "test would be vacuous if the message fit in the old window"
    assert "gone.avi" in exception_message(tb)
    assert "gone.avi" in format_task_error(tb)


def test_format_task_error_puts_message_first_and_keeps_frames():
    tb = (
        'Traceback (most recent call last):\n'
        '  File "a.py", line 1, in <module>\n'
        '    boom()\n'
        'ValueError: it broke\n'
    )
    out = format_task_error(tb)
    assert out.startswith("ValueError: it broke")
    assert 'File "a.py"' in out


def test_format_task_error_handles_non_traceback_input():
    assert format_task_error("just a string") == "just a string"
    assert exception_message("") == ""
