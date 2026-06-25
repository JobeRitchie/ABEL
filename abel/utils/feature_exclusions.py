"""Single source of truth for project-level feature exclusions.

Excluded features are stored in ``config/feature_exclusions.json`` and honoured by
**every** stage from Active Learning onward — training, inference (via the model's
persisted feature list), UMAP, evaluation and benchmarking — so a feature the user
turns off in one place stays off everywhere.

The file holds two keys:

- ``excluded_feature_cols`` — explicit column names (written by the Active
  Learning "Configure Features" dialog and the Feature Audit tab).
- ``disabled_feature_groups`` — coarse feature families toggled off in the
  Features tab; expanded to concrete columns via name patterns.

This module centralises the read + resolution logic that previously lived only
inside the trainer, so any consumer can apply the same exclusions with one call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from abel.storage.file_store import read_json, write_json

# Coarse feature-group → column-name patterns (mirrors the Features tab groups).
_GROUP_PATTERNS: dict[str, list[str]] = {
    "per_keypoint": ["_velocity_x", "_velocity_y", "_speed", "_acceleration", "_jerk"],
    "global_speed": [
        "centroid_velocity", "forepaw_speed", "forepaw_vertical_velocity",
        "nose_velocity", "nose_vertical_velocity",
    ],
    "oscillation": [
        "forepaw_oscillation_power", "nose_oscillation_power",
        "forepaw_autocorr_peak", "nose_autocorr_peak",
        "forepaw_movement_frequency", "nose_movement_frequency",
        "oscillation_energy", "nose_oscillation_energy",
    ],
    "orientation": ["head_pitch", "body_orientation"],
}


def _exclusions_path(project_root: Path) -> Path:
    return Path(project_root) / "config" / "feature_exclusions.json"


def load_exclusion_spec(project_root: Path) -> tuple[set[str], set[str]]:
    """Return ``(explicit_columns, disabled_groups)`` from the project config."""
    path = _exclusions_path(project_root)
    if not path.exists():
        return set(), set()
    try:
        data = read_json(path, {}) or {}
    except Exception:
        return set(), set()
    cols = {
        str(c) for c in data.get("excluded_feature_cols", [])
        if not str(c).startswith("__feat_group:")  # pattern markers, not columns
    }
    groups = {str(g) for g in data.get("disabled_feature_groups", [])}
    return cols, groups


def columns_to_exclude(project_root: Path, feature_cols: Iterable[str]) -> set[str]:
    """Resolve the concrete set of columns to drop from ``feature_cols``."""
    cols = list(feature_cols)
    explicit, groups = load_exclusion_spec(project_root)
    drop = {c for c in cols if c in explicit}
    for grp in groups:
        for pattern in _GROUP_PATTERNS.get(grp, []):
            for col in cols:
                if col == pattern or col.endswith(pattern) or pattern in col:
                    drop.add(col)
    return drop


def apply_feature_exclusions(project_root: Path, feature_cols: Iterable[str]) -> list[str]:
    """Return ``feature_cols`` with all project-excluded columns removed (order kept)."""
    cols = list(feature_cols)
    drop = columns_to_exclude(project_root, cols)
    if not drop:
        return cols
    return [c for c in cols if c not in drop]


def set_excluded_columns(
    project_root: Path,
    shown_columns: Iterable[str],
    excluded_among_shown: Iterable[str],
) -> set[str]:
    """Merge a dialog's selection into ``feature_exclusions.json`` and return the
    full excluded-column set.

    Columns *not* shown in the calling dialog are preserved as-is (e.g. Feature
    Audit exclusions for columns the Active Learning dialog didn't list); the
    shown columns are set exactly to ``excluded_among_shown``.  ``disabled_feature_groups``
    and other keys are left untouched.
    """
    path = _exclusions_path(project_root)
    data = {}
    if path.exists():
        try:
            data = read_json(path, {}) or {}
        except Exception:
            data = {}
    shown = {str(c) for c in shown_columns}
    chosen = {str(c) for c in excluded_among_shown}
    existing = {
        str(c) for c in data.get("excluded_feature_cols", [])
        if not str(c).startswith("__feat_group:")
    }
    preserved_markers = [
        str(c) for c in data.get("excluded_feature_cols", [])
        if str(c).startswith("__feat_group:")
    ]
    merged = (existing - shown) | chosen
    data["excluded_feature_cols"] = sorted(merged) + preserved_markers
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, data)
    return merged
