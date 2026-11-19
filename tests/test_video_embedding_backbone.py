"""The Video Appearance Embedding choice: R3D-18 or ResNet-18.

Each backbone writes its own columns and cache, the representation cache only
changes for a non-default backbone, dense inference follows each model's own
columns, and the Feature Extraction tab saves and restores the choice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.services import r3d_feature_service as rfs
from abel.services.behavior_representation_service import (
    BehaviorRepresentationService,
    RepresentationConfig,
)
from abel.storage.file_store import read_yaml, write_yaml


def test_backbone_names_and_columns() -> None:
    assert rfs.normalize_backbone("ResNet-18") == rfs.BACKBONE_RESNET18
    assert rfs.normalize_backbone("rn18") == rfs.BACKBONE_RESNET18
    assert rfs.normalize_backbone(None) == rfs.BACKBONE_R3D
    assert rfs.normalize_backbone("garbage") == rfs.BACKBONE_R3D
    rn = rfs.embedding_columns("resnet18")
    assert rn[0] == "rn18_000" and rn[-1] == "rn18_511" and len(rn) == 512
    assert rfs.r3d_columns()[0] == "r3d_000"
    assert rfs.is_video_embedding_column("rn18_007")
    assert rfs.is_video_embedding_column("r3d_007")
    assert not rfs.is_r3d_column("rn18_007")
    assert rfs.backbones_in_columns(["speed", "rn18_001"]) == ["resnet18"]
    assert rfs.backbones_in_columns(["r3d_001", "rn18_001"]) == ["r3d_18", "resnet18"]
    assert rfs.backbones_in_columns(["speed"]) == []


def test_project_backbone_resolution(tmp_path: Path) -> None:
    assert rfs.resolve_project_backbone(tmp_path) == "r3d_18"  # no project.yaml
    write_yaml(tmp_path / "project.yaml", {"feature_extraction": {"video_embedding_backbone": "resnet18"}})
    assert rfs.resolve_project_backbone(tmp_path) == "resnet18"


def test_config_signature_unchanged_for_r3d() -> None:
    """Existing (R3D) representation caches must stay valid."""
    sig = BehaviorRepresentationService._config_signature
    assert "video_embedding_backbone" not in sig(RepresentationConfig())
    assert "video_embedding_backbone" not in sig(RepresentationConfig(video_embedding_backbone="r3d_18"))
    rn = sig(RepresentationConfig(video_embedding_backbone="resnet18"))
    assert rn["video_embedding_backbone"] == "resnet18"
    # With the embedding off, the backbone does not matter.
    off = sig(RepresentationConfig(use_r3d_features=False, video_embedding_backbone="resnet18"))
    assert "video_embedding_backbone" not in off


def test_service_uses_own_columns_and_cache_dir(tmp_path: Path) -> None:
    svc = rfs.R3DFeatureService("resnet18")
    assert svc._cols[0] == "rn18_000"
    assert svc._cache_path(tmp_path, "s1") == tmp_path / "derived" / "rn18_features" / "s1.parquet"
    base, _ = svc._dense_cache_paths(tmp_path, "s1", 15)
    assert base.parent == tmp_path / "derived" / "rn18_features" / "dense_anchors"
    r3d = rfs.R3DFeatureService()
    assert r3d._cache_path(tmp_path, "s1") == tmp_path / "derived" / "r3d_features" / "s1.parquet"


def test_dense_signature_default_is_unchanged(tmp_path: Path) -> None:
    """R3D dense-anchor caches written before the backbone choice stay valid."""
    video = tmp_path / "v.avi"
    pose = tmp_path / "p.h5"
    video.write_bytes(b"x")
    pose.write_bytes(b"y")
    old = rfs.R3DFeatureService._dense_signature(None, video, pose)
    assert old["cache_version"] == "r3d_dense_v1"
    rn = rfs.R3DFeatureService._dense_signature(None, video, pose, cache_version="rn18_dense_v1")
    assert rn != old


def test_resnet18_forward_shape_and_frame_order_invariance(monkeypatch) -> None:
    """4 frames through a 2D ResNet, averaged: shape (n, 512), order-free."""
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None).eval()
    monkeypatch.setattr(rfs, "_get_model", lambda device, backbone=None: model)
    rng = np.random.default_rng(0)
    clips = rng.integers(0, 255, size=(3, 16, 112, 112, 3), dtype=np.uint8)
    out = rfs._forward_batch(clips, "cpu", "resnet18")
    assert out.shape == (3, 512) and out.dtype == np.float32
    # Reversing the clip samples the same 4 frames (0, 5, 10, 15) in reverse;
    # a frame-averaged 2D embedding cannot tell the difference.
    rev = rfs._forward_batch(clips[:, ::-1].copy(), "cpu", "resnet18")
    assert np.allclose(out, rev, atol=1e-4)


def test_attach_writes_backbone_columns(monkeypatch, tmp_path: Path) -> None:
    """``attach`` with ResNet-18 merges rn18_* columns, not r3d_*."""
    svc = rfs.R3DFeatureService("resnet18")

    class _Imports:
        def load_manifest(self, root):
            return object()

    svc._imports = _Imports()

    def fake_session(project_root, manifest, session_id, grp):
        df = pd.DataFrame(np.ones((len(grp), 512), dtype=np.float32), columns=svc._cols)
        df.insert(0, "segment_id", grp["segment_id"].astype(str).to_numpy())
        return df

    monkeypatch.setattr(svc, "_session_embeddings", fake_session)
    seg = pd.DataFrame({
        "segment_id": ["a", "b"], "session_id": ["s", "s"],
        "start_frame": [0, 15], "end_frame": [14, 29],
    })
    out = svc.attach(tmp_path, seg)
    assert "rn18_000" in out.columns
    assert not any(c.startswith("r3d_") for c in out.columns)


def test_validation_taxonomy_treats_rn18_as_video() -> None:
    from abel.validation.features import classify_modality, is_r3d_column, MODALITY_VIDEO

    assert classify_modality("rn18_042") == MODALITY_VIDEO
    assert is_r3d_column("rn18_042")


# ── Feature Extraction tab ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt unavailable: {exc}")


def _project(tmp_path: Path, **extraction) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    write_yaml(project / "project.yaml", {
        "schema_version": "0.3.0",
        "feature_extraction": {
            "window_duration_sec": 0.5,
            "stride_sec": 0.5,
            "source_fps": 30.0,
            "likelihood_threshold": 0.2,
            "interpolate_dropouts": True,
            "smoothing_window": 3,
            **extraction,
        },
    })
    return project


def _tab(project: Path):
    from abel.services.import_service import ImportService
    from abel.services.pose_features_service import PoseFeaturesService
    from abel.ui.tabs.pose_features_tab import PoseFeaturesTab

    tab = PoseFeaturesTab(PoseFeaturesService(), ImportService())
    tab._project_root = project
    tab._deferred_project_init(project)
    return tab


def test_tab_restores_and_saves_backbone(_app, tmp_path: Path) -> None:
    project = _project(tmp_path, use_video_features=True, video_embedding_backbone="resnet18")
    tab = _tab(project)
    assert tab._p_backbone_rn18.isChecked()
    assert tab._p_backbone_rn18.isEnabled()

    tab._p_backbone_r3d.setChecked(True)
    fx = read_yaml(project / "project.yaml", {})["feature_extraction"]
    assert fx["video_embedding_backbone"] == "r3d_18"


def test_tab_defaults_to_r3d_and_greys_out(_app, tmp_path: Path) -> None:
    project = _project(tmp_path, use_video_features=False)
    tab = _tab(project)
    assert tab._p_backbone_r3d.isChecked()
    assert not tab._p_backbone_r3d.isEnabled()
    tab._p_use_video.setChecked(True)
    assert tab._p_backbone_rn18.isEnabled()
    tab._p_use_r3d.setChecked(False)
    assert not tab._p_backbone_rn18.isEnabled()
