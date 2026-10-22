"""Free Up Disk Space removes only what a finished project does not need."""

from __future__ import annotations

from pathlib import Path

from abel.services.project_cleanup_service import (
    category_keys,
    cleanse_project,
    scan_project,
)


def _touch(path: Path, n: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * n)
    return path


def _project(root: Path) -> dict[str, Path]:
    d = root / "derived"
    files = {
        # Must survive every category.
        "raw_video": _touch(root / "raw" / "videos" / "a.mp4"),
        "project_yaml": _touch(root / "project.yaml"),
        "yaml_backup": _touch(root / "backups" / "project_20260101.yaml"),
        "training_set": _touch(d / "training_sets" / "training_set.parquet"),
        "newest_snapshot": _touch(d / "training_sets" / "snapshots" / "training_set_20260103_000000.parquet"),
        "pose_session": _touch(d / "pose_features" / "sessions" / "session_a.parquet"),
        "frame_pose": _touch(d / "pose_features" / "frame_pose.parquet"),
        "r3d": _touch(d / "r3d_features" / "session_a.parquet"),
        "enriched": _touch(d / "representations" / "enriched_segments.parquet"),
        "model": _touch(d / "models" / "behavior_model_Rear" / "model.pkl"),
        "model_parked": _touch(d / "model_backups" / "behavior_model_Rear" / "model.pkl"),
        "imported_clip": _touch(d / "clips" / "OtherProj__session_b" / "c.mp4"),
        "tr_run": _touch(d / "temporal_refinement" / "target_behavior" / "inference_x" / "p.parquet"),
        "diagnostics": _touch(d / "analysis" / "diagnostics" / "queue" / "q.json"),
        # Removable, one per category.
        "old_snapshot": _touch(d / "training_sets" / "snapshots" / "training_set_20260101_000000.parquet"),
        "repair_backup": _touch(d / "backups" / "fp_label_repair_1" / "r.parquet"),
        "script_backup": _touch(root / "backups" / "label_prune_1" / "t.parquet"),
        "ts_bak": _touch(d / "training_sets" / "training_set.parquet.bak_20260101"),
        "tab_cache": _touch(d / "temporal_refinement" / "tab_cache" / "frame_representations" / "f.parquet"),
        "own_clip": _touch(d / "clips" / "session_a" / "c.mp4"),
        "export_video": _touch(root / "exports" / "labeled_videos" / "v.mp4"),
        "segment_features": _touch(d / "representations" / "segment_features.parquet"),
        "frame_features": _touch(d / "representations" / "frame_features.parquet"),
    }
    return files


_REMOVABLE = {
    "old_snapshot", "repair_backup", "script_backup", "ts_bak",
    "tab_cache", "own_clip", "export_video", "segment_features", "frame_features",
}


def test_scan_is_read_only_and_measures(tmp_path: Path) -> None:
    files = _project(tmp_path)
    cats = {c.key: c for c in scan_project(tmp_path)}
    assert set(cats) == set(category_keys())
    assert all(p.exists() for p in files.values())
    assert cats["training_snapshots"].n_files == 1
    assert cats["old_backups"].n_files == 3
    assert cats["review_clips"].n_files == 1
    # Rebuildable caches and user exports are opt-in.
    assert not cats["representation_cache"].default_on
    assert not cats["labeled_video_exports"].default_on


def test_cleanse_everything_keeps_what_the_project_needs(tmp_path: Path) -> None:
    files = _project(tmp_path)
    result = cleanse_project(tmp_path, category_keys())
    assert result["errors"] == []
    for name, path in files.items():
        assert path.exists() == (name not in _REMOVABLE), name
    assert result["freed_files"] == len(_REMOVABLE)
    assert (tmp_path / "logs" / "disk_cleanup_log.json").exists()


def test_cleanse_only_selected(tmp_path: Path) -> None:
    files = _project(tmp_path)
    cleanse_project(tmp_path, ["old_backups"])
    assert not files["repair_backup"].exists()
    assert files["old_snapshot"].exists()
    assert files["segment_features"].exists()


def test_folder_scope_finds_and_cleans_every_project(tmp_path: Path) -> None:
    from abel.services.project_cleanup_service import (
        cleanse_projects,
        find_projects,
        scan_projects,
    )

    a = tmp_path / "A"
    b = tmp_path / "group" / "B"
    files_a, files_b = _project(a), _project(b)
    # Not a project: no project.yaml.
    stray = _touch(tmp_path / "notes" / "derived" / "training_sets" / "snapshots" / "x.parquet")

    roots = find_projects(tmp_path)
    assert sorted(roots) == sorted([a, b])
    cats = {c.key: c for c in scan_projects(roots)}
    assert cats["training_snapshots"].n_files == 2

    result = cleanse_projects(roots, ["training_snapshots"])
    assert result["projects"] == 2 and result["freed_files"] == 2
    assert not files_a["old_snapshot"].exists() and not files_b["old_snapshot"].exists()
    assert files_a["newest_snapshot"].exists() and files_b["newest_snapshot"].exists()
    assert stray.exists()
