"""Sessions recorded at different frame rates share one window length in seconds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.services import roi_behavior_service
from abel.utils.gpu_feature_ops import build_segment_df_fast
from abel.utils.session_fps import (
    is_mixed,
    scaled_frames,
    session_fps,
    session_fps_scales,
    video_fps_by_session,
)


def _project(tmp_path: Path, rates: dict[str, float], source_fps: float = 30.0) -> Path:
    root = tmp_path / "proj"
    (root / "derived" / "review_tables").mkdir(parents=True)
    manifest = {
        "videos": [{"asset_id": f"v_{sid}", "fps": fps} for sid, fps in rates.items()],
        "linked_sessions": [
            {"session_id": sid, "video_asset_id": f"v_{sid}"} for sid in rates
        ],
    }
    (root / "derived" / "review_tables" / "import_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "project.yaml").write_text(
        f"feature_extraction:\n  source_fps: {source_fps}\n", encoding="utf-8"
    )
    return root


def test_single_rate_project_is_untouched(tmp_path):
    # Even when the Source FPS box disagrees with the videos, a project whose
    # videos all share a rate keeps its existing frame counts and segment ids.
    root = _project(tmp_path, {"a": 60.0, "b": 60.0}, source_fps=30.0)
    assert session_fps_scales(root) == {}


def test_ntsc_rate_counts_as_the_same(tmp_path):
    root = _project(tmp_path, {"a": 29.97, "b": 30.0})
    assert not is_mixed([29.97, 30.0])
    assert session_fps_scales(root) == {}


def test_mixed_project_scales_only_the_odd_sessions(tmp_path):
    root = _project(tmp_path, {"a": 30.0, "b": 60.0, "c": 15.0})
    scales = session_fps_scales(root)
    assert scales == {"b": 2.0, "c": 0.5}
    assert video_fps_by_session(root) == {"a": 30.0, "b": 60.0, "c": 15.0}
    assert scaled_frames(15, scales["b"]) == 30
    assert scaled_frames(15, scales.get("a")) == 15
    assert session_fps(30.0, scales["b"]) == 60.0
    assert session_fps(30.0, scales.get("a")) == 30.0


def _frames(fps: float, seconds: float = 8.0) -> pd.DataFrame:
    """One feature holding a 2 Hz sine plus a ramp, sampled at ``fps``."""
    t = np.arange(int(round(seconds * fps))) / fps
    return pd.DataFrame({
        "frame": np.arange(len(t)),
        "roi_dist": np.sin(2 * np.pi * 2.0 * t) + 0.5 * t,
    })


def test_same_motion_gives_matching_segments_at_two_rates():
    slow = build_segment_df_fast(
        _frames(30.0), ["roi_dist"], "m", "s30", 30, 15,
        include_posture_deltas=True,
    )
    fast = build_segment_df_fast(
        _frames(60.0), ["roi_dist"], "m", "s60", 60, 30,
        include_posture_deltas=True, time_scale=2.0,
    )
    assert len(slow) == len(fast)
    feature_cols = [c for c in slow.columns if c.startswith("roi_dist_")]
    assert any(c.endswith("_periodicity") for c in feature_cols)
    for col in feature_cols:
        np.testing.assert_allclose(
            fast[col].to_numpy(), slow[col].to_numpy(), rtol=0.08, atol=0.05,
            err_msg=col,
        )


def test_unscaled_periodicity_would_not_match():
    slow = build_segment_df_fast(_frames(30.0), ["roi_dist"], "m", "s30", 30, 15)
    fast = build_segment_df_fast(_frames(60.0), ["roi_dist"], "m", "s60", 60, 30)
    ratio = fast["roi_dist_periodicity"].to_numpy() / slow["roi_dist_periodicity"].to_numpy()
    assert np.all(ratio > 1.7)


def _write_pose_sources(root: Path, frames_by_session: dict[str, int]) -> Path:
    pose_dir = root / "derived" / "pose_features"
    (pose_dir / "sessions").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for sid, n in frames_by_session.items():
        pd.DataFrame({
            "frame": np.arange(n),
            "animal_id": "a1",
            "session_id": sid,
            "speed": rng.normal(size=n),
        }).to_parquet(pose_dir / "sessions" / f"{sid}.parquet", index=False)
    return pose_dir / "frame_pose.parquet"


def _segment_lengths(root: Path, pose_path: Path) -> dict[str, set[int]]:
    from abel.services.behavior_representation_service import (
        BehaviorRepresentationService,
        RepresentationConfig,
    )

    _frame, seg = BehaviorRepresentationService().build(
        project_root=root,
        frame_pose_path=pose_path,
        frame_context_path=None,
        config=RepresentationConfig(
            window_size_frames=30, window_stride_frames=15, use_r3d_features=False
        ),
    )
    lengths = seg["end_frame"] - seg["start_frame"] + 1
    return {str(sid): set(grp.tolist()) for sid, grp in lengths.groupby(seg["session_id"])}


def test_representation_windows_cover_the_same_seconds(tmp_path):
    root = _project(tmp_path, {"s30": 30.0, "s60": 60.0})
    pose_path = _write_pose_sources(root, {"s30": 300, "s60": 600})
    lengths = _segment_lengths(root, pose_path)
    assert lengths == {"s30": {30}, "s60": {60}}


def test_representation_windows_unchanged_for_single_rate(tmp_path):
    root = _project(tmp_path, {"s30": 60.0, "s60": 60.0})
    pose_path = _write_pose_sources(root, {"s30": 300, "s60": 600})
    lengths = _segment_lengths(root, pose_path)
    assert lengths == {"s30": {30}, "s60": {30}}


def test_summarize_bouts_uses_each_sessions_rate():
    bouts = pd.DataFrame({
        "session_id": ["a", "b"],
        "behavior_id": ["x", "x"],
        "start_frame": [30, 60],
        "end_frame": [59, 119],
    })
    stats = roi_behavior_service.summarize_bouts(
        bouts, 30.0, fps_by_session={"b": 60.0}
    )
    assert stats[("a", "x")][1] == pytest.approx(1.0)
    assert stats[("b", "x")][1] == pytest.approx(1.0)
    assert stats[("b", "x")][3] == pytest.approx(1.0)
