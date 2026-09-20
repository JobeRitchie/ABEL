"""BG-subtraction sensitivity is a real, per-project extraction setting.

It lives in environment_rois.yaml (motion.bg_var_threshold), is read by context
extraction, and, because the context-cache signature hashes that file,
changing it rebuilds context features.  A project that never touches it keeps a
byte-identical ROI file so its cache stays valid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abel.services.context_feature_service import ContextFeatureConfig, ContextFeatureService
from abel.services.feature_prep_service import FeaturePrepService, PrepConfig
from abel.services.roi_service import DEFAULT_BG_VAR_THRESHOLD, ROIService


def _project(tmp_path: Path) -> ROIService:
    svc = ROIService()
    svc.save(tmp_path, svc.default_config())
    return svc


def test_default_when_unset_and_key_not_written(tmp_path: Path) -> None:
    svc = _project(tmp_path)
    assert svc.bg_var_threshold(tmp_path) == DEFAULT_BG_VAR_THRESHOLD
    assert "bg_var_threshold" not in (tmp_path / ROIService.ROI_FILE).read_text(encoding="utf-8")


def test_set_persists_clamps_and_skips_noop_writes(tmp_path: Path) -> None:
    svc = _project(tmp_path)
    path = tmp_path / ROIService.ROI_FILE
    original = path.read_text(encoding="utf-8")

    assert svc.set_bg_var_threshold(tmp_path, DEFAULT_BG_VAR_THRESHOLD) is False
    assert path.read_text(encoding="utf-8") == original

    assert svc.set_bg_var_threshold(tmp_path, 9) is True
    assert ROIService().bg_var_threshold(tmp_path) == 9
    assert svc.set_bg_var_threshold(tmp_path, 9) is False

    svc.set_bg_var_threshold(tmp_path, 10_000)
    assert svc.bg_var_threshold(tmp_path) == 100
    svc.set_bg_var_threshold(tmp_path, 0)
    assert svc.bg_var_threshold(tmp_path) == 4

    # Back to the default restores the original file exactly.
    svc.set_bg_var_threshold(tmp_path, DEFAULT_BG_VAR_THRESHOLD)
    assert path.read_text(encoding="utf-8") == original


def test_other_motion_edits_keep_the_threshold(tmp_path: Path) -> None:
    svc = _project(tmp_path)
    svc.set_bg_var_threshold(tmp_path, 25)
    cfg = svc.load(tmp_path)
    cfg["motion"]["local_radius_px"] = 50
    svc.save(tmp_path, cfg)
    assert svc.bg_var_threshold(tmp_path) == 25
    assert svc.local_motion_radius(tmp_path) == 50


def test_changing_threshold_invalidates_context_cache(tmp_path: Path) -> None:
    svc = _project(tmp_path)
    FeaturePrepService._write_signatures(tmp_path, {}, PrepConfig())
    assert FeaturePrepService._context_changed(tmp_path, {}, PrepConfig()) is False

    svc.set_bg_var_threshold(tmp_path, 8)
    assert FeaturePrepService._context_changed(tmp_path, {}, PrepConfig()) is True
    assert FeaturePrepService._pose_changed(tmp_path, {}) is False


def _noisy_clip(tmp_path: Path, n: int = 90, w: int = 64, h: int = 64) -> Path:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    path = tmp_path / "noise.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for _ in range(n):
        frame = np.clip(128 + rng.normal(0, 12, (h, w)), 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def _energy(path: Path, threshold: int, n: int = 90) -> float:
    xs = np.full(n, 32.0)
    zeros = np.zeros(n)
    out = ContextFeatureService._process_video_chunk(
        video_path=path, frame_start=60, frame_end=n,
        body_x=xs, body_y=xs, paw_l_x=zeros, paw_l_y=zeros, paw_r_x=zeros, paw_r_y=zeros,
        nose_x=xs, nose_y=xs, target_roi={}, has_target=False, local_radius=12,
        config=ContextFeatureConfig(downsample_factor=1, prefer_gpu=False),
        mog2_var_threshold=threshold,
    )
    return float(np.mean(out["local_surface_energy"]))


def test_threshold_changes_extracted_energy(tmp_path: Path) -> None:
    path = _noisy_clip(tmp_path)
    assert _energy(path, 4) > 3 * _energy(path, 100)


def test_extraction_uses_the_project_threshold(tmp_path: Path, monkeypatch) -> None:
    import pandas as pd

    from abel.services.pose_processing_service import PoseData

    # Long enough that every one of the 8 progress chunks spans two flow strides.
    n = 200
    clip = _noisy_clip(tmp_path, n=n)
    parts = ["nose", "tail_base"]
    xs = pd.DataFrame({"nose": np.full(n, 34.0), "tail_base": np.full(n, 30.0)})
    lk = pd.DataFrame({"nose": np.ones(n), "tail_base": np.ones(n)})
    pose = PoseData(parts, xs, xs.copy(), lk, np.full(n, 32.0), np.full(n, 32.0), n)

    def _run(project: Path) -> tuple[list[int], float]:
        seen: list[int] = []
        real = ContextFeatureService._process_video_chunk

        def _spy(**kw):
            seen.append(kw["mog2_var_threshold"])
            return real(**kw)

        monkeypatch.setattr(ContextFeatureService, "_process_video_chunk", staticmethod(_spy))
        service = ContextFeatureService()
        monkeypatch.setattr(service._pose, "load_and_clean", lambda *a, **k: pose)
        df = service.compute_frame_context(
            project, clip, tmp_path / "unused.csv", "m1", "s1",
            config=ContextFeatureConfig(downsample_factor=1, prefer_gpu=False), save=False,
        )
        monkeypatch.setattr(ContextFeatureService, "_process_video_chunk", staticmethod(real))
        return seen, float(df["local_surface_motion_energy"].iloc[60:].mean())

    default_root = tmp_path / "default"
    tuned_root = tmp_path / "tuned"
    _project(default_root)
    _project(tuned_root).set_bg_var_threshold(tuned_root, 4)

    seen_default, energy_default = _run(default_root)
    seen_tuned, energy_tuned = _run(tuned_root)
    assert set(seen_default) == {DEFAULT_BG_VAR_THRESHOLD}
    assert set(seen_tuned) == {4}
    assert energy_tuned > energy_default
