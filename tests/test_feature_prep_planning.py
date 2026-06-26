"""Tests for the pure worker-planning logic in FeaturePrepService."""

from __future__ import annotations

from pathlib import Path

from abel.services.feature_prep_service import FeaturePrepService, plan_session_workers


def test_cpu_only_fills_cores() -> None:
    plan = plan_session_workers(10, gpu_info={"backend": "cpu"}, cpu_count=9)
    assert plan.max_workers == 8  # cpu_count - 1
    assert "auto" in plan.source


def test_cpu_only_capped_by_job_count() -> None:
    plan = plan_session_workers(3, gpu_info={"backend": "cpu"}, cpu_count=9)
    assert plan.max_workers == 3


def test_env_override_wins() -> None:
    plan = plan_session_workers(
        10, gpu_info={"backend": "torch", "total_mb": 24000}, cpu_count=16, env_workers="2"
    )
    assert plan.max_workers == 2
    assert plan.source == "environment override"


def test_gpu_flow_caps_workers_by_vram() -> None:
    plan = plan_session_workers(
        16, gpu_info={"backend": "torch", "total_mb": 4096, "name": "T4"}, cpu_count=16
    )
    assert plan.max_workers == 2  # <=4096 MB -> cap 2
    assert "GPU-adaptive" in plan.source


def test_gpu_flow_unknown_vram_safe_fallback() -> None:
    plan = plan_session_workers(
        16, gpu_info={"backend": "cv2_cuda", "total_mb": 0}, cpu_count=16
    )
    assert plan.max_workers == 2
    assert "GPU-safe fallback" in plan.source


def test_intra_session_workers_fill_remaining_cores() -> None:
    # 16 jobs, GPU caps sessions to 8 on a big GPU; remaining cores spread.
    plan = plan_session_workers(
        16, gpu_info={"backend": "torch", "total_mb": 24000}, cpu_count=16
    )
    assert plan.max_workers == 8
    # cpu_cap = 15, ceil(15/8) = 2
    assert plan.intra_session_workers == 2


# ── Keypoint-rename cache invalidation ────────────────────────────────


def _cfg(**kw):
    from abel.services.feature_prep_service import PrepConfig
    return PrepConfig(**kw)


def test_no_signature_and_no_cache_does_not_force_rebuild(tmp_path: Path) -> None:
    # Fresh project with nothing to rebuild.
    assert FeaturePrepService._pose_changed(tmp_path, {"a": "b"}) is False
    assert FeaturePrepService._context_changed(tmp_path, {"a": "b"}, _cfg()) is False


def test_legacy_cache_without_signature_rebuilds_pose_once(tmp_path: Path) -> None:
    # A pose cache with no signature file predates schema tracking and may use
    # the old (order-dependent) feature format, so it must rebuild once.
    sess = tmp_path / "derived" / "pose_features" / "sessions"
    sess.mkdir(parents=True)
    (sess / "s1.parquet").write_text("x", encoding="utf-8")
    assert FeaturePrepService._pose_changed(tmp_path, {}) is True
    # After recording the current signatures, it no longer rebuilds.
    FeaturePrepService._write_signatures(tmp_path, {}, _cfg())
    assert FeaturePrepService._pose_changed(tmp_path, {}) is False


def test_pose_signature_roundtrip_detects_rename(tmp_path: Path) -> None:
    FeaturePrepService._write_signatures(tmp_path, {"bodypart1": "nose"}, _cfg())
    assert FeaturePrepService._pose_changed(tmp_path, {"bodypart1": "nose"}) is False
    # A different rename map is detected.
    assert FeaturePrepService._pose_changed(tmp_path, {"bodypart1": "snout"}) is True
    # So is clearing the renames entirely.
    assert FeaturePrepService._pose_changed(tmp_path, {}) is True


def test_context_signature_detects_roi_change(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    roi = tmp_path / "config" / "environment_rois.yaml"
    roi.write_text("project_rois:\n  target_zones: [{x: 1, y: 1, w: 5, h: 5}]\n", encoding="utf-8")
    FeaturePrepService._write_signatures(tmp_path, {}, _cfg())
    assert FeaturePrepService._context_changed(tmp_path, {}, _cfg()) is False
    # Editing the ROI config invalidates the context cache (not pose).
    roi.write_text("project_rois:\n  target_zones: [{x: 9, y: 9, w: 5, h: 5}]\n", encoding="utf-8")
    assert FeaturePrepService._context_changed(tmp_path, {}, _cfg()) is True
    assert FeaturePrepService._pose_changed(tmp_path, {}) is False


def test_invalidate_forces_rebuild_for_any_map(tmp_path: Path) -> None:
    FeaturePrepService._write_signatures(tmp_path, {"bodypart1": "nose"}, _cfg())
    FeaturePrepService.invalidate_caches(tmp_path)
    assert FeaturePrepService._pose_changed(tmp_path, {"bodypart1": "nose"}) is True
    assert FeaturePrepService._context_changed(tmp_path, {"bodypart1": "nose"}, _cfg()) is True
    # Writing the real signatures again clears the stale state.
    FeaturePrepService._write_signatures(tmp_path, {"bodypart1": "nose"}, _cfg())
    assert FeaturePrepService._pose_changed(tmp_path, {"bodypart1": "nose"}) is False


def test_legacy_signature_uses_mtime_for_context(tmp_path: Path) -> None:
    import json as _json
    import os as _os
    import time as _time
    # Simulate an old-format (pose-only) signature file + a context cache.
    sig_dir = tmp_path / "derived" / "pose_features"
    sig_dir.mkdir(parents=True)
    (sig_dir / ".keypoint_alias_signature.json").write_text(
        _json.dumps({"signature": FeaturePrepService._alias_signature({})}),
        encoding="utf-8",
    )
    ctx_dir = tmp_path / "derived" / "context_features" / "sessions"
    ctx_dir.mkdir(parents=True)
    ctx_file = ctx_dir / "s1.parquet"
    ctx_file.write_text("x", encoding="utf-8")
    (tmp_path / "config").mkdir()
    roi = tmp_path / "config" / "environment_rois.yaml"
    roi.write_text("project_rois: {}\n", encoding="utf-8")

    # ROI older than the context cache -> not stale.
    old = _time.time() - 100
    _os.utime(roi, (old, old))
    assert FeaturePrepService._context_changed(tmp_path, {}, _cfg()) is False
    # ROI newer than the context cache -> stale (the NSF_Jess case).
    new = _time.time() + 100
    _os.utime(roi, (new, new))
    assert FeaturePrepService._context_changed(tmp_path, {}, _cfg()) is True
