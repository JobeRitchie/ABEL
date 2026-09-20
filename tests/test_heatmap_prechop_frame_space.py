"""Regression: spatial views must undo the prechop rebase before indexing pose.

Bug (found on the Sarah_nsf_ABEL project): the Analytics tab rebases
``_raw_bouts`` to the per-subject analysis prechop, so frame 0 becomes test
onset and latency/time-bin metrics are measured from there.  Pose arrays are
never rebased, they stay indexed by raw video frame.  The Spatial Heatmap and
Density Analysis views looked bout positions up in the pose using the rebased
frames, so every plotted coordinate came from 1000-4160 frames (33-139 s on
that project) before the behavior actually happened.  Every behavior then
smeared into the same generic arena-occupancy map: Eat, which happens only at
the center food pellet, scattered across the whole arena.

Worse, those views also re-applied ``_apply_prechop_to_bout_df(rebase=False)``
to the already-rebased frames, clamping starts up to the offset a second time
and silently dropping bouts.

The fix is ``_unrebase_bout_df_to_video_frames``, which adds the offset back,
the same ``+ pre`` correction ``_bout_velocity_records`` already applied.
"""

from __future__ import annotations

import types

import pandas as pd

from abel.ui.tabs.behavior_analytics_tab import BehaviorAnalyticsTab


def _host(prechop: dict[str, int], subject_by_session: dict[str, str]):
    host = types.SimpleNamespace()
    host._subject_prechop_frames = prechop
    host._session_prechop_overrides = {}
    host._subject_by_session = subject_by_session
    host._analysis_prechop_for_session = types.MethodType(
        BehaviorAnalyticsTab._analysis_prechop_for_session, host
    )
    return host


def _unrebase(host, df: pd.DataFrame) -> pd.DataFrame:
    return types.MethodType(
        BehaviorAnalyticsTab._unrebase_bout_df_to_video_frames, host
    )(df)


def test_unrebase_restores_raw_video_frames() -> None:
    host = _host({"m1": 1200, "m2": 4160}, {"s1": "m1", "s2": "m2"})
    # Frames as stored in _raw_bouts: already rebased to test onset.
    df = pd.DataFrame(
        {
            "session_id": ["s1", "s2"],
            "start_frame": [300, 500],
            "end_frame": [360, 590],
        }
    )
    out = _unrebase(host, df)
    assert out["start_frame"].tolist() == [1500, 4660]
    assert out["end_frame"].tolist() == [1560, 4750]
    # Durations are unchanged: only the origin moves.
    assert (out["end_frame"] - out["start_frame"]).tolist() == [60, 90]


def test_unrebase_round_trips_the_rebase() -> None:
    host = _host({"m1": 1200}, {"s1": "m1"})
    raw = pd.DataFrame(
        {"session_id": ["s1", "s1"], "start_frame": [1500, 9000], "end_frame": [1560, 9100]}
    )
    rebased = types.MethodType(BehaviorAnalyticsTab._apply_prechop_to_bout_df, host)(
        raw, rebase=True
    )
    restored = _unrebase(host, rebased)
    assert restored["start_frame"].tolist() == raw["start_frame"].tolist()
    assert restored["end_frame"].tolist() == raw["end_frame"].tolist()


def test_unrebase_keeps_every_bout() -> None:
    """The old double-prechop clamped starts and dropped rows; this must not."""
    host = _host({"m1": 1200}, {"s1": "m1"})
    # A bout early in the test window: rebased start (100) sits below the
    # offset, which the erroneous second prechop pass would have clamped to
    # 1200 (or dropped the row outright).
    df = pd.DataFrame({"session_id": ["s1"], "start_frame": [100], "end_frame": [160]})
    out = _unrebase(host, df)
    assert len(out) == 1
    assert out["start_frame"].tolist() == [1300]


def test_unrebase_is_a_noop_without_prechop() -> None:
    host = _host({}, {"s1": "m1"})
    df = pd.DataFrame({"session_id": ["s1"], "start_frame": [100], "end_frame": [160]})
    out = _unrebase(host, df)
    assert out["start_frame"].tolist() == [100]
    assert out["end_frame"].tolist() == [160]
