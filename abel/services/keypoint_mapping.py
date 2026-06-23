"""Keypoint (bodypart) alias mapping for Direct Use.

When a trained model is applied to new pose files whose DLC keypoints are
named differently (e.g. ``back_mid`` instead of ``center_body``), every
feature derived from those keypoints comes out under a different column name
and the model silently scores on zero-filled inputs.  This module helps the
user map the new files' keypoints onto the names the model was trained with,
*before* feature extraction, so all derived features line up.

The mapping is stored as ``{model_keypoint: data_keypoint}`` (what the model
expects -> what the new data calls it).  The pipeline inverts it to rename the
new data's keypoints to the model's names when loading pose.
"""

from __future__ import annotations

import logging
from pathlib import Path

from abel.services.pose_processing_service import normalize_bodypart_name
from abel.storage.file_store import read_json, write_json

logger = logging.getLogger("abel")

MAP_FILENAME = "direct_use_keypoint_map.json"
# Rename map ({data_keypoint: model_keypoint}) written into the target project
# for the Direct Use pipeline to consume.
ALIASES_FILENAME = "keypoint_aliases.json"

# Token canonicalisation for similarity scoring.  Anatomical synonyms collapse
# to a shared token so e.g. ``center_body`` and ``back_mid`` match.
_TOKEN_SYNONYMS: dict[str, str] = {
    "body": "trunk",
    "back": "trunk",
    "spine": "trunk",
    "dorsal": "trunk",
    "mid": "center",
    "middle": "center",
    "centre": "center",
    "centroid": "center",
    "snout": "nose",
    "tailbase": "tail_base",
    "l": "left",
    "r": "right",
}


def _canonical_tokens(name: str) -> frozenset[str]:
    """Split a (normalized) keypoint name into canonicalised tokens."""
    norm = normalize_bodypart_name(name)
    tokens = [t for t in norm.replace("-", "_").split("_") if t]
    return frozenset(_TOKEN_SYNONYMS.get(t, t) for t in tokens)


def _similarity(a: str, b: str) -> float:
    """Jaccard similarity of two keypoint names over canonical tokens."""
    ta, tb = _canonical_tokens(a), _canonical_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def suggest_mapping(
    model_keypoints: list[str],
    data_keypoints: list[str],
    min_score: float = 0.5,
) -> dict[str, str]:
    """Best-guess ``{model_keypoint: data_keypoint}`` mapping.

    Exact (normalized) matches win first; otherwise the highest token-overlap
    candidate above ``min_score`` is used.  Each data keypoint is assigned at
    most once.  Unmatched model keypoints map to ``""``.
    """
    data_norm = {normalize_bodypart_name(d): d for d in data_keypoints}
    used: set[str] = set()
    out: dict[str, str] = {}

    # Pass 1: exact normalized matches.
    for mk in model_keypoints:
        nm = normalize_bodypart_name(mk)
        if nm in data_norm and data_norm[nm] not in used:
            out[mk] = data_norm[nm]
            used.add(data_norm[nm])

    # Pass 2: best token-overlap for the remainder.
    for mk in model_keypoints:
        if mk in out:
            continue
        best, best_score = "", 0.0
        for dk in data_keypoints:
            if dk in used:
                continue
            score = _similarity(mk, dk)
            if score > best_score:
                best, best_score = dk, score
        if best and best_score >= min_score:
            out[mk] = best
            used.add(best)
        else:
            out[mk] = ""
    return out


def to_rename_map(model_to_data: dict[str, str]) -> dict[str, str]:
    """Invert ``{model_kp: data_kp}`` into the ``{data_kp: model_kp}`` rename
    map applied to freshly loaded pose data.  Skips empty / identity entries."""
    rename: dict[str, str] = {}
    for model_kp, data_kp in model_to_data.items():
        data_kp = str(data_kp or "").strip()
        model_kp = str(model_kp or "").strip()
        if data_kp and model_kp and data_kp != model_kp:
            rename[data_kp] = model_kp
    return rename


def load_saved(source_root: Path) -> dict[str, str]:
    """Load a previously saved ``{model_kp: data_kp}`` map for a source project."""
    path = source_root / "config" / MAP_FILENAME
    if not path.exists():
        return {}
    data = read_json(path, {})
    mapping = data.get("model_to_data", data) if isinstance(data, dict) else {}
    return {str(k): str(v) for k, v in mapping.items()}


def save(source_root: Path, model_to_data: dict[str, str]) -> None:
    """Persist a ``{model_kp: data_kp}`` map alongside the source project."""
    cfg = source_root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    write_json(cfg / MAP_FILENAME, {"model_to_data": dict(model_to_data)})


def write_target_aliases(target_root: Path, rename_map: dict[str, str]) -> None:
    """Write the ``{data_kp: model_kp}`` rename map into the target project so
    the Direct Use pipeline applies it on pose load."""
    cfg = target_root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    write_json(cfg / ALIASES_FILENAME, dict(rename_map))
