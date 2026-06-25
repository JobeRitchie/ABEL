"""Tests for the Body Part Rename feature.

The dialog collects free-text new names for body parts; the renames are stored
in the project's keypoint alias map and applied on pose load so every downstream
step uses the new names.  These tests cover the dialog's collect/duplicate/reset
logic and the rename round-trip through pose loading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.services.pose_processing_service import PoseProcessingService

# The dialog needs Qt; skip cleanly where it isn't importable/usable.
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.body_part_rename_dialog import BodyPartRenameDialog  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")
    return app


def test_collect_returns_only_changed_parts(_app) -> None:
    dlg = BodyPartRenameDialog(["bodypart1", "bodypart2", "nose"])
    dlg._edits["bodypart1"].setText("snout")
    dlg._edits["bodypart2"].setText("tail_base")
    # 'nose' left unchanged -> excluded.
    assert dlg._collect() == {"bodypart1": "snout", "bodypart2": "tail_base"}


def test_initial_map_prefills_new_names(_app) -> None:
    dlg = BodyPartRenameDialog(
        ["bodypart1", "nose"], initial_map={"bodypart1": "snout"}
    )
    assert dlg._edits["bodypart1"].text() == "snout"
    assert dlg._edits["nose"].text() == "nose"
    # An unchanged-from-original entry is not reported as a rename.
    assert dlg._collect() == {"bodypart1": "snout"}


def test_blank_new_name_is_ignored(_app) -> None:
    dlg = BodyPartRenameDialog(["nose"])
    dlg._edits["nose"].setText("   ")
    assert dlg._collect() == {}


def test_duplicate_names_flagged_in_status(_app) -> None:
    dlg = BodyPartRenameDialog(["a", "b"])
    dlg._edits["a"].setText("paw")
    dlg._edits["b"].setText("paw")
    dlg._update_status()
    assert "duplicate" in dlg._status.text().lower()


def test_reset_restores_originals(_app) -> None:
    dlg = BodyPartRenameDialog(["a", "b"])
    dlg._edits["a"].setText("x")
    dlg._reset()
    assert dlg._collect() == {}
    assert dlg._edits["a"].text() == "a"


def test_rename_applied_on_pose_load(tmp_path: Path) -> None:
    """The {original: new} map written by the dialog renames parts on load."""
    parts = ["bodypart1", "nose"]
    cols = [("S", p, c) for p in parts for c in ("x", "y", "likelihood")]
    df = pd.DataFrame(
        np.random.rand(5, len(cols)),
        columns=pd.MultiIndex.from_tuples(cols),
    )
    csv = tmp_path / "pose.csv"
    df.to_csv(csv)

    pose = PoseProcessingService().load(
        csv, keypoint_aliases={"bodypart1": "tail_base"}
    )
    assert "tail_base" in pose.body_parts
    assert "bodypart1" not in pose.body_parts
    assert "nose" in pose.body_parts
