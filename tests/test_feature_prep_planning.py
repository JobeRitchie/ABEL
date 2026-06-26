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


def test_no_signature_file_does_not_force_rebuild(tmp_path: Path) -> None:
    # Projects built before the guard existed keep their caches.
    assert FeaturePrepService._aliases_changed(tmp_path, {"a": "b"}) is False


def test_signature_roundtrip_detects_change(tmp_path: Path) -> None:
    FeaturePrepService._write_alias_signature(tmp_path, {"bodypart1": "nose"})
    assert FeaturePrepService._aliases_changed(tmp_path, {"bodypart1": "nose"}) is False
    # A different rename map is detected.
    assert FeaturePrepService._aliases_changed(tmp_path, {"bodypart1": "snout"}) is True
    # So is clearing the renames entirely.
    assert FeaturePrepService._aliases_changed(tmp_path, {}) is True


def test_invalidate_forces_rebuild_for_any_map(tmp_path: Path) -> None:
    # Even with a matching signature already written, invalidation wins.
    FeaturePrepService._write_alias_signature(tmp_path, {"bodypart1": "nose"})
    FeaturePrepService.invalidate_caches(tmp_path)
    assert FeaturePrepService._aliases_changed(tmp_path, {"bodypart1": "nose"}) is True
    assert FeaturePrepService._aliases_changed(tmp_path, {}) is True
    # Writing the real signature again clears the stale state.
    FeaturePrepService._write_alias_signature(tmp_path, {"bodypart1": "nose"})
    assert FeaturePrepService._aliases_changed(tmp_path, {"bodypart1": "nose"}) is False
