"""Tests for keypoint alias mapping used by Direct Use.

Covers the auto-suggestion logic, the rename applied on pose load (which makes
derived feature columns line up with the model's expected names), and the
round-trip persistence helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abel.services import keypoint_mapping
from abel.services.pose_processing_service import PoseData, PoseProcessingService


def test_suggest_mapping_handles_synonyms_and_reorders() -> None:
    model = ["center_body", "left_body", "right_body", "left_ear", "right_ear",
             "nose", "tail_base"]
    data = ["back_left", "back_mid", "back_right", "ear_left", "ear_right",
            "nose", "tail_base"]
    m = keypoint_mapping.suggest_mapping(model, data)
    assert m == {
        "center_body": "back_mid",
        "left_body": "back_left",
        "right_body": "back_right",
        "left_ear": "ear_left",
        "right_ear": "ear_right",
        "nose": "nose",
        "tail_base": "tail_base",
    }


def test_suggest_mapping_leaves_unmatched_blank() -> None:
    m = keypoint_mapping.suggest_mapping(["nose", "whisker_tip"], ["nose", "elbow"])
    assert m["nose"] == "nose"
    assert m["whisker_tip"] == ""


def test_to_rename_map_drops_identity_and_empty() -> None:
    model_to_data = {
        "center_body": "back_mid",
        "nose": "nose",          # identity → dropped
        "left_ear": "",          # unmapped → dropped
    }
    assert keypoint_mapping.to_rename_map(model_to_data) == {"back_mid": "center_body"}


def _pose_with(parts: list[str], n: int = 30) -> PoseData:
    t = np.arange(n, dtype=float)
    x = pd.DataFrame({p: 10.0 + i + 0.5 * t for i, p in enumerate(parts)})
    y = pd.DataFrame({p: 5.0 + i + 0.3 * t for i, p in enumerate(parts)})
    likelihood = pd.DataFrame(1.0, index=range(n), columns=parts)
    return PoseData(
        body_parts=list(parts), x=x, y=y, likelihood=likelihood,
        centroid_x=np.asarray(x.mean(axis=1)), centroid_y=np.asarray(y.mean(axis=1)),
        n_frames=n,
    )


def test_apply_keypoint_aliases_renames_everywhere() -> None:
    pose = _pose_with(["back_mid", "nose", "tail_base"])
    renamed = PoseProcessingService._apply_keypoint_aliases(
        pose, {"back_mid": "center_body"},
    )
    assert "center_body" in renamed.body_parts
    assert "back_mid" not in renamed.body_parts
    assert "center_body" in renamed.x.columns
    assert "center_body" in renamed.y.columns
    assert "center_body" in renamed.likelihood.columns
    # nose / tail_base untouched.
    assert "nose" in renamed.body_parts and "tail_base" in renamed.body_parts


def test_apply_keypoint_aliases_skips_collision() -> None:
    # Renaming back_mid -> nose would collide with the existing nose; skip it.
    pose = _pose_with(["back_mid", "nose"])
    renamed = PoseProcessingService._apply_keypoint_aliases(
        pose, {"back_mid": "nose"},
    )
    assert "back_mid" in renamed.body_parts  # unchanged
    assert sorted(renamed.body_parts) == ["back_mid", "nose"]


def test_renamed_pose_produces_model_named_features() -> None:
    """The whole point: after aliasing, derived feature columns use model names."""
    svc = PoseProcessingService()
    data_pose = _pose_with(["back_mid", "back_left", "nose", "tail_base"])
    renamed = svc._apply_keypoint_aliases(data_pose, {
        "back_mid": "center_body", "back_left": "left_body",
    })
    feats = svc.compute_frame_pose_features(
        pose=renamed, fps=30.0, animal_id="a", session_id="s", video_id="v",
    )
    assert "center_body_speed" in feats.columns
    assert "left_body_speed" in feats.columns
    assert not any(c.startswith("back_mid_") for c in feats.columns)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "source"
    mapping = {"center_body": "back_mid", "nose": "nose"}
    keypoint_mapping.save(root, mapping)
    assert keypoint_mapping.load_saved(root) == mapping


def test_write_target_aliases(tmp_path: Path) -> None:
    target = tmp_path / "target"
    keypoint_mapping.write_target_aliases(target, {"back_mid": "center_body"})
    from abel.storage.file_store import read_json
    written = read_json(target / "config" / keypoint_mapping.ALIASES_FILENAME, {})
    assert written == {"back_mid": "center_body"}
