"""Tests for the shared feature-exclusion source of truth."""

from __future__ import annotations

from pathlib import Path

from abel.storage.file_store import read_json, write_json
from abel.utils.feature_exclusions import (
    apply_feature_exclusions,
    columns_to_exclude,
    load_exclusion_spec,
    set_excluded_columns,
)


def _write(project_root: Path, data: dict) -> None:
    (project_root / "config").mkdir(parents=True, exist_ok=True)
    write_json(project_root / "config" / "feature_exclusions.json", data)


def test_no_file_returns_all_columns(tmp_path: Path) -> None:
    cols = ["a", "b", "c"]
    assert apply_feature_exclusions(tmp_path, cols) == cols
    assert load_exclusion_spec(tmp_path) == (set(), set())


def test_explicit_columns_excluded(tmp_path: Path) -> None:
    _write(tmp_path, {"excluded_feature_cols": ["b"]})
    assert apply_feature_exclusions(tmp_path, ["a", "b", "c"]) == ["a", "c"]


def test_group_marker_is_ignored_as_column(tmp_path: Path) -> None:
    _write(tmp_path, {"excluded_feature_cols": ["__feat_group:per_keypoint", "b"]})
    cols, groups = load_exclusion_spec(tmp_path)
    assert cols == {"b"}  # marker filtered out
    assert apply_feature_exclusions(tmp_path, ["a", "b"]) == ["a"]


def test_disabled_groups_expand_to_columns(tmp_path: Path) -> None:
    _write(tmp_path, {"disabled_feature_groups": ["orientation"]})
    cols = ["head_pitch_mean", "body_orientation_std", "speed_mean"]
    drop = columns_to_exclude(tmp_path, cols)
    assert "head_pitch_mean" in drop
    assert "body_orientation_std" in drop
    assert "speed_mean" not in drop


def test_set_excluded_columns_preserves_unshown(tmp_path: Path) -> None:
    # Feature Audit already excluded a column the AL dialog doesn't list.
    _write(tmp_path, {"excluded_feature_cols": ["audit_only"]})
    merged = set_excluded_columns(
        tmp_path,
        shown_columns=["a", "b", "c"],
        excluded_among_shown=["b"],
    )
    assert merged == {"audit_only", "b"}
    on_disk = set(read_json(tmp_path / "config" / "feature_exclusions.json", {})["excluded_feature_cols"])
    assert on_disk == {"audit_only", "b"}


def test_set_excluded_columns_unchecking_removes(tmp_path: Path) -> None:
    _write(tmp_path, {"excluded_feature_cols": ["a", "b"]})
    # Re-include "a" (no longer excluded among shown), keep "b" excluded.
    merged = set_excluded_columns(tmp_path, shown_columns=["a", "b"], excluded_among_shown=["b"])
    assert merged == {"b"}


def test_set_excluded_columns_preserves_group_markers(tmp_path: Path) -> None:
    _write(tmp_path, {"excluded_feature_cols": ["__feat_group:oscillation", "x"]})
    set_excluded_columns(tmp_path, shown_columns=["x"], excluded_among_shown=[])
    on_disk = read_json(tmp_path / "config" / "feature_exclusions.json", {})["excluded_feature_cols"]
    assert "__feat_group:oscillation" in on_disk
    assert "x" not in on_disk
