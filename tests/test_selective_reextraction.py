"""Re-extracting only the sessions whose tracking was fixed."""

import json

import pandas as pd
import pytest

from abel.services.feature_prep_service import FeaturePrepService


def _project(tmp_path, sessions=("s1", "s2")):
    d = tmp_path / "derived"
    for sid in sessions:
        for family in ("pose_features", "context_features"):
            p = d / family / "sessions"
            p.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"frame": [0, 1]}).to_parquet(p / f"{sid}.parquet", index=False)
        (d / "pose_features" / f"{sid}.npz").write_bytes(b"x")
        (d / "r3d_features").mkdir(parents=True, exist_ok=True)
        (d / "r3d_features" / f"{sid}.parquet").write_bytes(b"x")
        dense = d / "r3d_features" / "dense_anchors"
        dense.mkdir(parents=True, exist_ok=True)
        (dense / f"{sid}__w16.parquet").write_bytes(b"x")
        (dense / f"{sid}__w16.json").write_text("{}", encoding="utf-8")
    reps = d / "representations"
    reps.mkdir(parents=True, exist_ok=True)
    (reps / "representations.manifest.json").write_text(
        json.dumps({"cache_signature": {}}), encoding="utf-8"
    )
    return tmp_path


def test_invalidating_one_session_leaves_the_others_cached(tmp_path):
    pytest.importorskip("pyarrow")
    root = _project(tmp_path)
    out = FeaturePrepService.invalidate_sessions(root, ["s1"])

    assert out["sessions"] == 1 and out["files"] >= 5
    assert FeaturePrepService.cached_pose_sessions(root) == {"s2"}
    assert FeaturePrepService.cached_context_sessions(root) == {"s2"}
    assert not (root / "derived" / "pose_features" / "s1.npz").exists()
    assert not (root / "derived" / "r3d_features" / "s1.parquet").exists()
    assert not (root / "derived" / "r3d_features" / "dense_anchors" / "s1__w16.parquet").exists()
    # s2 keeps everything, including its R3D embeddings.
    assert (root / "derived" / "r3d_features" / "s2.parquet").exists()
    assert (root / "derived" / "r3d_features" / "dense_anchors" / "s2__w16.json").exists()


def test_the_representation_manifest_is_dropped(tmp_path):
    # Re-extraction leaves row counts and columns unchanged, so without dropping
    # the manifest the segment cache would keep the pre-fix values.
    pytest.importorskip("pyarrow")
    root = _project(tmp_path)
    FeaturePrepService.invalidate_sessions(root, ["s1"])
    assert not (root / "derived" / "representations" / "representations.manifest.json").exists()


def test_nothing_is_touched_for_an_empty_list(tmp_path):
    pytest.importorskip("pyarrow")
    root = _project(tmp_path)
    out = FeaturePrepService.invalidate_sessions(root, [])
    assert out == {"sessions": 0, "files": 0}
    assert (root / "derived" / "representations" / "representations.manifest.json").exists()
    assert FeaturePrepService.cached_pose_sessions(root) == {"s1", "s2"}


@pytest.fixture
def _fresh_signature(monkeypatch):
    """Pretend the project-wide pose signature is current.

    ``sessions_needing_extraction`` also reports everything when the pose schema
    itself changed; these tests are about the per-session half of that answer.
    """
    monkeypatch.setattr(FeaturePrepService, "_pose_changed", classmethod(lambda *_a, **_k: False))


def test_sessions_needing_extraction_lists_the_invalidated_ones(tmp_path, _fresh_signature):
    pytest.importorskip("pyarrow")
    root = _project(tmp_path)
    FeaturePrepService.invalidate_sessions(root, ["s1"])
    needing = FeaturePrepService.sessions_needing_extraction(root, ["s1", "s2"])
    assert needing == ["s1"]


def test_context_cache_only_counts_when_video_features_are_on(tmp_path, _fresh_signature):
    pytest.importorskip("pyarrow")
    root = _project(tmp_path)
    (root / "derived" / "context_features" / "sessions" / "s2.parquet").unlink()
    assert FeaturePrepService.sessions_needing_extraction(
        root, ["s1", "s2"], use_video_features=True
    ) == ["s2"]
    assert FeaturePrepService.sessions_needing_extraction(
        root, ["s1", "s2"], use_video_features=False
    ) == []


def test_clip_refresh_is_a_no_op_without_sessions(tmp_path):
    from abel.services.preprocessing_service import regenerate_clips_for_sessions

    out = regenerate_clips_for_sessions(tmp_path, [])
    assert out["extracted"] == 0 and out["session_ids"] == []


def test_clip_refresh_says_so_when_a_session_has_no_clips(tmp_path):
    from abel.services.preprocessing_service import regenerate_clips_for_sessions

    out = regenerate_clips_for_sessions(tmp_path, ["s1"])
    assert out["extracted"] == 0
    assert any("no existing clips" in w.lower() for w in out["warnings"])
