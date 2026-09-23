"""The Pose & Features smoothing settings reach feature extraction and its caches."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from abel.models.schemas import PoseSmoothingSettings
from abel.services.feature_prep_service import FeaturePrepService, PrepConfig
from abel.services.pose_processing_service import PoseProcessingService


def _write_project(root: Path, **fx) -> None:
    (root / "project.yaml").write_text(
        yaml.safe_dump({"feature_extraction": {"window_duration_sec": 0.5, **fx}}),
        encoding="utf-8",
    )


class _Captured(Exception):
    pass


def test_load_from_project_reads_tab_values(tmp_path: Path) -> None:
    _write_project(tmp_path, smoothing_window=1, likelihood_threshold=0.4, interpolate_dropouts=False)
    s = PoseSmoothingSettings.load_from_project(tmp_path)
    assert (s.smoothing_window, s.likelihood_threshold, s.interpolate_dropouts) == (1, 0.4, False)
    # Settings the tab does not edit keep their defaults.
    assert s.absence_max_fill_frames == PoseSmoothingSettings().absence_max_fill_frames


def test_load_from_project_defaults_without_block(tmp_path: Path) -> None:
    assert PoseSmoothingSettings.load_from_project(tmp_path) == PoseSmoothingSettings()
    (tmp_path / "project.yaml").write_text("feature_extraction:\n  use_r3d_features: true\n", encoding="utf-8")
    assert PoseSmoothingSettings.load_from_project(tmp_path) == PoseSmoothingSettings()


def test_cache_signature_empty_at_defaults() -> None:
    assert PoseSmoothingSettings().cache_signature() == {}
    assert PoseSmoothingSettings(smoothing_window=1).cache_signature() == {"smoothing_window": 1}


@pytest.mark.parametrize("multi", [False, True])
def test_extraction_uses_project_smoothing(tmp_path: Path, monkeypatch, multi: bool) -> None:
    _write_project(tmp_path, smoothing_window=1)
    seen: list[PoseSmoothingSettings] = []

    def _capture(self, path, settings=None, **kw):
        seen.append(settings)
        raise _Captured

    svc = PoseProcessingService()
    if multi:
        monkeypatch.setattr(PoseProcessingService, "load_and_clean_multi", _capture)
        call = lambda **kw: svc.extract_and_save_frame_pose_features_multi(  # noqa: E731
            individual_animal_ids={}, **kw)
    else:
        monkeypatch.setattr(PoseProcessingService, "load_and_clean", _capture)
        call = lambda **kw: svc.extract_and_save_frame_pose_features(animal_id="a", **kw)  # noqa: E731
    common = dict(project_root=tmp_path, pose_path=tmp_path / "p.h5", fps=30.0,
                  session_id="s1", video_id="s1")

    with pytest.raises(_Captured):
        call(**common)
    assert seen[-1].smoothing_window == 1

    # An explicit value (batch inference in a scratch folder) wins.
    with pytest.raises(_Captured):
        call(smoothing=PoseSmoothingSettings(smoothing_window=7), **common)
    assert seen[-1].smoothing_window == 7


def test_default_smoothing_keeps_existing_signature(tmp_path: Path) -> None:
    # Caches built before extraction honored the setting were built at the
    # defaults, so a project still at the defaults must not be forced to rebuild.
    FeaturePrepService._write_signatures(tmp_path, {}, PrepConfig())
    _write_project(tmp_path, smoothing_window=5, likelihood_threshold=0.2)
    assert FeaturePrepService._pose_changed(tmp_path, {}) is False
    assert FeaturePrepService._context_changed(tmp_path, {}, PrepConfig()) is False


def test_smoothing_change_invalidates_pose_and_context(tmp_path: Path) -> None:
    FeaturePrepService._write_signatures(tmp_path, {}, PrepConfig())
    sess = tmp_path / "derived" / "pose_features" / "sessions"
    sess.mkdir(parents=True)
    (sess / "s1.parquet").write_text("x", encoding="utf-8")

    _write_project(tmp_path, smoothing_window=1)
    assert FeaturePrepService._pose_changed(tmp_path, {}) is True
    assert FeaturePrepService._context_changed(tmp_path, {}, PrepConfig()) is True
    assert FeaturePrepService.sessions_needing_extraction(
        tmp_path, ["s1"], use_video_features=False) == ["s1"]

    # Once rebuilt at the new setting, the caches are current again.
    FeaturePrepService._write_signatures(tmp_path, {}, PrepConfig())
    assert FeaturePrepService._pose_changed(tmp_path, {}) is False
