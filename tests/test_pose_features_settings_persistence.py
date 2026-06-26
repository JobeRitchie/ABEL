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
