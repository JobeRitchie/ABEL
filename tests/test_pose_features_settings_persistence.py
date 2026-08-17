"""Regression test: the Features tab's settings must survive a project reload.

Restoring presets during project init fires ``valueChanged`` on the parameter
spinboxes, which used to persist the *default* settings (notably the "Include
video features" checkbox) over the project's saved values before they were read
back.  Loading a project must not clobber its own saved settings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.services.import_service import ImportService  # noqa: E402
from abel.services.pose_features_service import PoseFeaturesService  # noqa: E402
from abel.storage.file_store import read_yaml, write_yaml  # noqa: E402
from abel.ui.tabs.pose_features_tab import PoseFeaturesTab  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")
    return app


def test_use_video_features_persists_across_project_reload(_app, tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    # A saved project with video features enabled.
    write_yaml(project / "project.yaml", {
        "schema_version": "0.3.0",
        "feature_extraction": {
            "window_duration_sec": 2.0,
            "stride_sec": 1.0,
            "source_fps": 30.0,
            "likelihood_threshold": 0.2,
            "interpolate_dropouts": True,
            "smoothing_window": 5,
            "use_video_features": True,
        },
    })

    tab = PoseFeaturesTab(PoseFeaturesService(), ImportService())
    # Drive the load synchronously (set_project defers via QTimer).
    tab._project_root = project
    tab._deferred_project_init(project)

    # The checkbox reflects the saved value …
    assert tab._p_use_video.isChecked() is True
    # … and the load didn't rewrite project.yaml back to the default.
    raw = read_yaml(project / "project.yaml", {})
    assert raw["feature_extraction"]["use_video_features"] is True


def _project_with(tmp_path: Path, **extraction) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    write_yaml(project / "project.yaml", {
        "schema_version": "0.3.0",
        "feature_extraction": {
            "window_duration_sec": 2.0,
            "stride_sec": 1.0,
            "source_fps": 30.0,
            "likelihood_threshold": 0.2,
            "interpolate_dropouts": True,
            "smoothing_window": 5,
            **extraction,
        },
    })
    return project


def test_use_r3d_features_persists_across_project_reload(_app, tmp_path: Path) -> None:
    """Turning appearance embeddings off must survive a reload, like any setting."""
    project = _project_with(tmp_path, use_video_features=True, use_r3d_features=False)

    tab = PoseFeaturesTab(PoseFeaturesService(), ImportService())
    tab._project_root = project
    tab._deferred_project_init(project)

    assert tab._p_use_r3d.isChecked() is False
    assert read_yaml(project / "project.yaml", {})["feature_extraction"]["use_r3d_features"] is False


def test_r3d_defaults_on_for_projects_predating_the_setting(_app, tmp_path: Path) -> None:
    """A project saved before the checkbox existed keeps the shipped default."""
    project = _project_with(tmp_path, use_video_features=True)

    tab = PoseFeaturesTab(PoseFeaturesService(), ImportService())
    tab._project_root = project
    tab._deferred_project_init(project)

    assert tab._p_use_r3d.isChecked() is True


def test_partial_extraction_block_does_not_reset_legacy_settings(_app, tmp_path: Path) -> None:
    """The AL tab write-throughs one key here; that must not orphan the legacy block.

    A project.yaml that predates this tab keeps its settings under
    ``behavior_model``.  If a one-key ``feature_extraction`` block made the
    loader treat the project as "already migrated", every unwritten parameter
    would silently snap back to its default — including turning video features
    off for a project trained with them.
    """
    project = tmp_path / "proj"
    project.mkdir()
    write_yaml(project / "project.yaml", {
        "schema_version": "0.3.0",
        # Written by another tab's write-through, not by this one.
        "feature_extraction": {"use_r3d_features": False},
        "behavior_model": {"use_video_features": True, "advanced_roi_features": False},
    })

    tab = PoseFeaturesTab(PoseFeaturesService(), ImportService())
    tab._project_root = project
    tab._deferred_project_init(project)

    assert tab._p_use_video.isChecked() is True      # legacy value survived
    assert tab._p_advanced_roi.isChecked() is False  # legacy value survived
    assert tab._p_use_r3d.isChecked() is False       # partial block still honoured


def test_r3d_toggle_greys_out_without_video_features(_app, tmp_path: Path) -> None:
    """Appearance embeddings are pixel-derived: no video features, no toggle."""
    project = _project_with(tmp_path, use_video_features=False, use_r3d_features=True)

    tab = PoseFeaturesTab(PoseFeaturesService(), ImportService())
    tab._project_root = project
    tab._deferred_project_init(project)

    assert tab._p_use_video.isChecked() is False
    assert tab._p_use_r3d.isEnabled() is False
    # Enabling video features re-enables the row without changing its state.
    tab._p_use_video.setChecked(True)
    assert tab._p_use_r3d.isEnabled() is True
