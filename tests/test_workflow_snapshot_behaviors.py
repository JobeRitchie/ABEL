"""Regression tests for workflow-snapshot behaviour-model resolution.

These lock in the Direct Use fix: a snapshot must capture *every* trained
behaviour (not just one), auto-discovering models from ``derived/models/``
when the Temporal Refinement tab was left on "auto" (empty explicit map).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from abel.services.workflow_snapshot_service import WorkflowSnapshotService
from abel.storage.file_store import write_json


def _make_model(
    project_root: Path,
    dir_name: str,
    target_behavior: str,
    use_video_features: bool | None = None,
) -> None:
    """Create a minimal trained-model directory on disk."""
    mdir = project_root / "derived" / "models" / dir_name
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model_state.pkl").write_bytes(b"\x00")  # presence is all that matters
    rs: dict = {"target_behavior": target_behavior}
    if use_video_features is not None:
        rs["behavior_model"] = {"use_video_features": use_video_features}
    write_json(mdir / "run_settings.json", rs)


def _write_behavior_defs(project_root: Path, defs: list[dict]) -> None:
    cfg = project_root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "behavior_definitions.yaml").write_text(
        yaml.dump({"behaviors": defs}, default_flow_style=False),
        encoding="utf-8",
    )


def _basic_project(tmp_path: Path) -> Path:
    """A project with 3 real behaviours + no_behavior, all trained."""
    root = tmp_path / "proj"
    _make_model(root, "behavior_model_Walk", "walk-id")
    _make_model(root, "behavior_model_Rear", "rear-id")
    _make_model(root, "behavior_model_Groom", "groom-id")
    _make_model(root, "behavior_model_No_Behavior", "no_behavior")
    _write_behavior_defs(root, [
        {"behavior_id": "no_behavior", "name": "No Behavior", "is_active": True},
        {"behavior_id": "rear-id", "name": "Rear", "is_active": True},
        {"behavior_id": "groom-id", "name": "Groom", "is_active": True},
        {"behavior_id": "walk-id", "name": "Walk", "is_active": True},
    ])
    return root


def test_auto_resolves_all_trained_behaviors(tmp_path: Path) -> None:
    root = _basic_project(tmp_path)
    snap = WorkflowSnapshotService().build_from_project(root)

    # All three real behaviours captured; no_behavior excluded as a competitor.
    assert snap.selected_behavior_models == {
        "walk-id": "behavior_model_Walk",
        "rear-id": "behavior_model_Rear",
        "groom-id": "behavior_model_Groom",
    }
    assert "no_behavior" not in snap.selected_behavior_models


def test_target_behavior_and_model_version_are_consistent(tmp_path: Path) -> None:
    root = _basic_project(tmp_path)
    snap = WorkflowSnapshotService().build_from_project(root)

    # target_behavior follows definition order (first active, non-no_behavior).
    assert snap.target_behavior == "rear-id"
    # model_version points at the chosen target behaviour's model.
    assert snap.model_version == "behavior_model_Rear"


def test_no_behavior_skipped_when_only_model(tmp_path: Path) -> None:
    """A no_behavior-only project must still raise (no usable competitor)."""
    root = tmp_path / "proj"
    _make_model(root, "behavior_model_No_Behavior", "no_behavior")
    _write_behavior_defs(root, [
        {"behavior_id": "no_behavior", "name": "No Behavior", "is_active": True},
    ])
    snap = WorkflowSnapshotService().build_from_project(root)
    assert snap.selected_behavior_models == {}


def test_explicit_tr_selection_overrides_but_keeps_new_models(tmp_path: Path) -> None:
    root = _basic_project(tmp_path)
    # User pinned Walk to a specific dir in the TR tab, but never revisited
    # after training Rear/Groom.  Explicit choice wins for Walk; the others
    # are still auto-captured.
    write_json(root / "config" / "temporal_refinement_settings.json", {
        "__all__": {},
        "by_behavior": {
            "target_behavior": {
                "selected_behavior_models": {"walk-id": "behavior_model_Walk"},
                "excluded_behavior_ids": [],
            }
        },
    })
    snap = WorkflowSnapshotService().build_from_project(root)
    assert snap.selected_behavior_models["walk-id"] == "behavior_model_Walk"
    assert "rear-id" in snap.selected_behavior_models
    assert "groom-id" in snap.selected_behavior_models


def test_use_video_features_captured_from_run_settings(tmp_path: Path) -> None:
    """The model's run_settings.json is authoritative for use_video_features."""
    root = tmp_path / "proj"
    _make_model(root, "behavior_model_Walk", "walk-id", use_video_features=True)
    _write_behavior_defs(root, [
        {"behavior_id": "walk-id", "name": "Walk", "is_active": True},
    ])
    snap = WorkflowSnapshotService().build_from_project(root)
    assert snap.use_video_features is True
    # Survives a round-trip through dict serialisation.
    from abel.services.workflow_snapshot_service import WorkflowSnapshot
    assert WorkflowSnapshot.from_dict(snap.to_dict()).use_video_features is True


def test_use_video_features_false_for_pose_only_model(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _make_model(root, "behavior_model_Walk", "walk-id", use_video_features=False)
    _write_behavior_defs(root, [
        {"behavior_id": "walk-id", "name": "Walk", "is_active": True},
    ])
    snap = WorkflowSnapshotService().build_from_project(root)
    assert snap.use_video_features is False


def test_direct_run_resolves_video_features_for_legacy_snapshot(tmp_path: Path) -> None:
    """Old snapshots (use_video_features defaulting False) must still recover
    the flag from the source model's run_settings.json."""
    from abel.services.direct_run_service import DirectRunService
    from abel.services.workflow_snapshot_service import WorkflowSnapshot

    root = tmp_path / "proj"
    _make_model(root, "behavior_model_Walk", "walk-id", use_video_features=True)

    legacy = WorkflowSnapshot(
        model_version="behavior_model_Walk",
        target_behavior="walk-id",
        use_video_features=False,  # legacy snapshot never captured it
    )
    assert DirectRunService._resolve_use_video_features(legacy, root) is True


def test_newest_model_wins_for_same_behavior(tmp_path: Path) -> None:
    import os
    import time

    root = tmp_path / "proj"
    _make_model(root, "behavior_model_Walk_v1", "walk-id")
    _make_model(root, "behavior_model_Walk_v2", "walk-id")
    _write_behavior_defs(root, [
        {"behavior_id": "walk-id", "name": "Walk", "is_active": True},
    ])
    # Make v2 the most recently modified directory.
    newer = time.time() + 10
    os.utime(root / "derived" / "models" / "behavior_model_Walk_v2", (newer, newer))

    snap = WorkflowSnapshotService().build_from_project(root)
    assert snap.selected_behavior_models == {"walk-id": "behavior_model_Walk_v2"}
