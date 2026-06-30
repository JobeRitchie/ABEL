"""P1: representation cache must be keyed on content/config, not file mtime.

Regression test for the "z-scoring pops up again on every rerun" bug: re-saving
the source pose/context parquet files with identical content (which bumps mtime)
must NOT invalidate the cached, z-scored representation.  Genuine changes
(content or representation config) must still invalidate it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from abel.services.behavior_representation_service import (
    BehaviorRepresentationService,
    RepresentationConfig,
)


def _make_sources(project_root: Path, *, n_frames: int = 150, seed: int = 0):
    """Write per-session pose + context parquet files and return their paths."""
    rng = np.random.default_rng(seed)
    pose_dir = project_root / "derived" / "pose_features"
    ctx_dir = project_root / "derived" / "context_features"
    (pose_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (ctx_dir / "sessions").mkdir(parents=True, exist_ok=True)

    for sid in ("s1", "s2"):
        frames = np.arange(n_frames)
        pose = pd.DataFrame(
            {
                "frame": frames,
                "animal_id": "a1",
                "session_id": sid,
                "speed": rng.normal(size=n_frames),
                "angle": rng.normal(size=n_frames),
            }
        )
        ctx = pd.DataFrame(
            {
                "frame": frames,
                "animal_id": "a1",
                "session_id": sid,
                "flow_mag": rng.normal(size=n_frames),
            }
        )
        pose.to_parquet(pose_dir / "sessions" / f"{sid}.parquet", index=False)
        ctx.to_parquet(ctx_dir / "sessions" / f"{sid}.parquet", index=False)

    return pose_dir / "frame_pose.parquet", ctx_dir / "frame_context.parquet"


def _build(project_root, pose_path, ctx_path, config):
    msgs: list[str] = []
    svc = BehaviorRepresentationService()
    frame, seg = svc.build(
        project_root=project_root,
        frame_pose_path=pose_path,
        frame_context_path=ctx_path,
        config=config,
        progress_cb=msgs.append,
    )
    return frame, seg, msgs


def _zscored(msgs):
    return any("z-scoring" in m for m in msgs)


def _cache_hit(msgs):
    return any("cache hit" in m for m in msgs)


def test_resave_identical_sources_does_not_invalidate(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    pose_path, ctx_path = _make_sources(project)
    cfg = RepresentationConfig(window_size_frames=30, window_stride_frames=15)

    # First build: full compute (z-scores).
    frame1, seg1, msgs1 = _build(project, pose_path, ctx_path, cfg)
    assert _zscored(msgs1), "first build should compute z-scores"

    # Re-save the SAME content (new mtime, identical data).
    time.sleep(0.01)
    pose_sess = pose_path.parent / "sessions"
    ctx_sess = ctx_path.parent / "sessions"
    for f in list(pose_sess.glob("*.parquet")) + list(ctx_sess.glob("*.parquet")):
        pd.read_parquet(f).to_parquet(f, index=False)

    # Second build: must be a cache hit, NO re-z-scoring.
    frame2, seg2, msgs2 = _build(project, pose_path, ctx_path, cfg)
    assert _cache_hit(msgs2), f"expected cache hit after identical re-save; got {msgs2}"
    assert not _zscored(msgs2), "z-scoring must NOT rerun on identical re-save"

    # Data parity.
    pd.testing.assert_frame_equal(
        frame1.reset_index(drop=True), frame2.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        seg1.reset_index(drop=True), seg2.reset_index(drop=True)
    )


def test_config_change_invalidates(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    pose_path, ctx_path = _make_sources(project)

    _, _, _ = _build(
        project, pose_path, ctx_path,
        RepresentationConfig(window_size_frames=30, window_stride_frames=15),
    )
    # Changing the window config must force a rebuild (re-z-score).
    _, _, msgs2 = _build(
        project, pose_path, ctx_path,
        RepresentationConfig(window_size_frames=60, window_stride_frames=15),
    )
    assert _zscored(msgs2), "changing representation config must rebuild the cache"


def test_content_change_invalidates(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    pose_path, ctx_path = _make_sources(project)
    cfg = RepresentationConfig(window_size_frames=30, window_stride_frames=15)

    _build(project, pose_path, ctx_path, cfg)

    # Add a brand-new session file -> content signature changes -> rebuild.
    extra = pd.DataFrame(
        {
            "frame": np.arange(150),
            "animal_id": "a1",
            "session_id": "s3",
            "speed": np.random.default_rng(9).normal(size=150),
            "angle": np.random.default_rng(8).normal(size=150),
        }
    )
    extra.to_parquet(pose_path.parent / "sessions" / "s3.parquet", index=False)
    ctx_extra = pd.DataFrame(
        {
            "frame": np.arange(150),
            "animal_id": "a1",
            "session_id": "s3",
            "flow_mag": np.random.default_rng(7).normal(size=150),
        }
    )
    ctx_extra.to_parquet(ctx_path.parent / "sessions" / "s3.parquet", index=False)

    _, seg2, msgs2 = _build(project, pose_path, ctx_path, cfg)
    assert _zscored(msgs2), "adding a session must rebuild the cache"
    assert "s3" in set(seg2["session_id"].astype(str)), "new session must appear"


def test_value_change_same_schema_invalidates(tmp_path: Path):
    """Re-extracting features with the SAME schema + row count but DIFFERENT
    values (e.g. a smoothing/units change or pose re-export) must invalidate the
    cache.  Row-count + column-names alone miss this; the footer statistics
    digest catches it."""
    project = tmp_path / "proj"
    project.mkdir()
    pose_path, ctx_path = _make_sources(project)
    cfg = RepresentationConfig(window_size_frames=30, window_stride_frames=15)

    frame1, _, msgs1 = _build(project, pose_path, ctx_path, cfg)
    assert _zscored(msgs1), "first build should compute z-scores"

    # Overwrite one session's pose values in place — identical columns, identical
    # row count, brand-new numbers.
    pose_sess = pose_path.parent / "sessions" / "s1.parquet"
    df = pd.read_parquet(pose_sess)
    df["speed"] = df["speed"].to_numpy() + 100.0  # shifts min/max -> footer stats change
    df.to_parquet(pose_sess, index=False)

    frame2, _, msgs2 = _build(project, pose_path, ctx_path, cfg)
    assert _zscored(msgs2), f"value change must rebuild the cache; got {msgs2}"
    assert not _cache_hit(msgs2), "stale cache must not be reused after a value change"
