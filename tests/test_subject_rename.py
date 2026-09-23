"""Renaming subjects in Data Import must not detach anything from its sessions.

Fear Conditioning Redo was imported with a whole-stem subject regex, so every
session became its own subject (``m1_cond1``, ``m1_ext`` …).  Re-applying the
default regex merges them into real subjects (``m1``).  That rename must keep:

* segment/clip ids: extraction keys them by the frozen ``subject_key``;
* Analytics factors, session order and prechop: re-keyed on manifest save and
  on load (anchors);
* per-subject ROIs: re-keyed on manifest save;
* subject-level CV grouping: resolved to the *current* subject name.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from abel.models.schemas import ImportManifest, ImportNameSettings, LinkedSession
from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.services.import_service import ImportService
from abel.services.roi_service import ROIService
from abel.services.subject_rename_service import (
    ANCHORS_KEY,
    anchor_group_state,
    current_subjects,
    remap_group_state,
    remap_subject_rois,
    session_labels,
)

WHOLE_STEM = ImportNameSettings(subject_regex=r"^(.+?)(?=DLC|$)")
DEFAULT = ImportNameSettings()
POSE_SUFFIX = "DLC_Resnet50_alison_ltpJan28shuffle1_snapshot_040.csv"


def _fear_manifest(tmp_path: Path, stems: list[str]) -> ImportManifest:
    svc = ImportService()
    videos = [tmp_path / f"{s}.mp4" for s in stems]
    poses = [tmp_path / f"{s}{POSE_SUFFIX}" for s in stems]
    return svc.build_manifest(videos, poses, WHOLE_STEM)


def _sid(manifest: ImportManifest, stem: str) -> str:
    vid = next(v.asset_id for v in manifest.videos if Path(v.source_path).stem == stem)
    return next(s.session_id for s in manifest.linked_sessions if s.video_asset_id == vid)


def _project(tmp_path: Path, manifest: ImportManifest) -> Path:
    root = tmp_path / "proj"
    (root / "derived" / "review_tables").mkdir(parents=True)
    (root / "config").mkdir()
    ImportService().save_manifest(root, manifest)
    return root


# ----------------------------------------------------------------------
# subject_key
# ----------------------------------------------------------------------


def test_subject_key_backfills_from_subject_then_session() -> None:
    named = LinkedSession(session_id="session_1", video_asset_id="v", pose_asset_id="p", subject_id="m1_cond1")
    unnamed = LinkedSession(session_id="session_2", video_asset_id="v", pose_asset_id="p")
    kept = LinkedSession(
        session_id="session_3", video_asset_id="v", pose_asset_id="p", subject_id="m1", subject_key="m1_cond1"
    )
    assert named.subject_key == "m1_cond1"
    assert unnamed.subject_key == "session_2"
    assert kept.subject_key == "m1_cond1"


def test_regex_reapply_changes_subject_but_not_subject_key(tmp_path: Path) -> None:
    manifest = _fear_manifest(tmp_path, ["m20_recall", "m5_ext"])
    svc = ImportService()
    svc.apply_subject_name_settings(manifest, DEFAULT)

    by_stem = {
        Path(next(v for v in manifest.videos if v.asset_id == s.video_asset_id).source_path).stem: s
        for s in manifest.linked_sessions
    }
    assert by_stem["m20_recall"].subject_id == "m20"
    assert by_stem["m20_recall"].subject_key == "m20_recall"
    assert by_stem["m5_ext"].subject_id == "m5"
    assert by_stem["m5_ext"].subject_key == "m5_ext"
    # The key survives a save/load round trip too.
    root = _project(tmp_path, manifest)
    reloaded = svc.load_manifest(root)
    assert {s.subject_key for s in reloaded.linked_sessions} == {"m20_recall", "m5_ext"}


# ----------------------------------------------------------------------
# Manifest save carries Analytics + ROI state across the rename
# ----------------------------------------------------------------------


def test_save_manifest_rekeys_analytics_groups_and_rois(tmp_path: Path) -> None:
    stems = ["m1_cond1", "m1_ext", "m2_cond1", "m2_ext", "m3_cond1"]
    manifest = _fear_manifest(tmp_path, stems)
    root = _project(tmp_path, manifest)
    sids = {s: _sid(manifest, s) for s in stems}

    groups_path = root / "derived" / "analytics_groups.json"
    groups_path.write_text(json.dumps({
        "schema_version": "1.0",
        "factor_definitions": ["Session Type", "Sex"],
        "session_factors": {
            "m1_cond1": {"Session Type": "cond1", "Sex": "M"},
            "m1_ext": {"Session Type": "ext", "Sex": "M"},
            "m2_ext": {"Session Type": "ext", "Sex": "F"},
            "m3_cond1": {"Session Type": "cond1", "Sex": "F"},
        },
        "subject_order": ["m2_ext", "m1_cond1", "m1_ext", "m3_cond1"],
        "subject_prechop_frames": {"m3_cond1": 30},
    }), encoding="utf-8")
    rois = ROIService()
    cfg = rois.load(root)
    zone = {"x": 1, "y": 2, "w": 3, "h": 4}
    other = {"x": 9, "y": 9, "w": 9, "h": 9}
    cfg["subject_rois"] = {
        "m1_cond1": {"target_zones": [zone]},                       # merges into m1
        "m3_cond1": {"target_zones": [zone]},                       # pure rename to m3
        f"m2_ext::{sids['m2_ext']}": {"target_zones": [other]},    # per-session
    }
    rois.save(root, cfg)

    svc = ImportService()
    svc.apply_subject_name_settings(manifest, DEFAULT)
    notes = svc.save_manifest(root, manifest)
    assert notes

    state = json.loads(groups_path.read_text(encoding="utf-8"))
    assert state["session_factors"] == {
        "m1 – cond1": {"Session Type": "cond1", "Sex": "M"},
        "m1 – ext": {"Session Type": "ext", "Sex": "M"},
        "m2 – ext": {"Session Type": "ext", "Sex": "F"},
        "m3": {"Session Type": "cond1", "Sex": "F"},
    }
    assert state["subject_order"] == ["m2 – ext", "m1 – cond1", "m1 – ext", "m3"]
    assert state["subject_prechop_frames"] == {"m3": 30}
    assert state[ANCHORS_KEY]["labels"]["m1 – cond1"] == [sids["m1_cond1"]]
    assert list((root / "derived").glob("analytics_groups.backup-*.json"))

    subject_rois = ROIService().load(root)["subject_rois"]
    # m1 now also covers m1_ext, which had no ROI: the old block stays on its own session.
    assert set(subject_rois) == {f"m1::{sids['m1_cond1']}", "m3", f"m2::{sids['m2_ext']}"}
    assert subject_rois["m3"]["target_zones"][0]["w"] == 3
    assert subject_rois[f"m2::{sids['m2_ext']}"]["target_zones"][0]["w"] == 9

    # And back again: renaming to the whole-stem names restores the original keys.
    svc.apply_subject_name_settings(manifest, WHOLE_STEM)
    svc.save_manifest(root, manifest)
    state = json.loads(groups_path.read_text(encoding="utf-8"))
    assert set(state["session_factors"]) == {"m1_cond1", "m1_ext", "m2_ext", "m3_cond1"}
    assert state["subject_prechop_frames"] == {"m3_cond1": 30}


def test_save_without_rename_touches_nothing(tmp_path: Path) -> None:
    manifest = _fear_manifest(tmp_path, ["m1_cond1", "m2_cond1"])
    root = _project(tmp_path, manifest)
    groups_path = root / "derived" / "analytics_groups.json"
    groups_path.write_text(json.dumps({"session_factors": {"m1_cond1": {"Sex": "M"}}}), encoding="utf-8")
    before = groups_path.read_text(encoding="utf-8")

    assert ImportService().save_manifest(root, manifest) == []
    assert groups_path.read_text(encoding="utf-8") == before


def test_conflicting_prechop_is_reported_not_guessed(tmp_path: Path) -> None:
    manifest = _fear_manifest(tmp_path, ["m1_cond1", "m1_ext"])
    old = session_labels(manifest)
    ImportService().apply_subject_name_settings(manifest, DEFAULT)
    new = session_labels(manifest)

    state = {"subject_prechop_frames": {"m1_cond1": 30, "m1_ext": 45}}
    anchor_group_state(state, old)
    notes = remap_group_state(state, new)
    assert state["subject_prechop_frames"] == {}
    assert any("m1" in n and "different values" in n for n in notes)


def test_hand_typed_subject_keeps_one_label_per_session(tmp_path: Path) -> None:
    # "m01" is not a prefix of m1_cond1.mp4; the sessions used to collapse
    # onto a single "m01" label, and renaming back fanned one Session Type
    # out to all of them.
    stems = ["m1_cond1", "m1_cond2", "m1_ext", "m1_recall"]
    manifest = _fear_manifest(tmp_path, stems)
    ImportService().apply_subject_name_settings(manifest, DEFAULT)
    before = session_labels(manifest)
    state = {"session_factors": {
        before.label_by_session[_sid(manifest, s)]: {"Session Type": s.split("_")[1]} for s in stems
    }}
    anchor_group_state(state, before)

    for session in manifest.linked_sessions:
        session.subject_id = "m01"
    renamed = session_labels(manifest)
    assert sorted(renamed.label_by_session.values()) == [
        "m01 – cond1", "m01 – cond2", "m01 – ext", "m01 – recall",
    ]
    assert remap_group_state(state, renamed) == []

    for session in manifest.linked_sessions:
        session.subject_id = "m1"
    assert remap_group_state(state, session_labels(manifest)) == []
    assert state["session_factors"] == {
        f"m1 – {t}": {"Session Type": t} for t in ("cond1", "cond2", "ext", "recall")
    }


def test_unanchored_keys_are_left_alone() -> None:
    # Labels from merged projects never appear in this manifest's anchors.
    state = {"session_factors": {"other_project::m9": {"Sex": "F"}}, "subject_order": ["other_project::m9"]}
    notes = remap_group_state(state, session_labels(None))
    assert notes == []
    assert state["session_factors"] == {"other_project::m9": {"Sex": "F"}}
    assert state["subject_order"] == ["other_project::m9"]


def test_roi_split_moves_plain_key_to_each_new_subject() -> None:
    # One old subject split into two new names: each gets the plain block.
    out, notes = remap_subject_rois(
        {"cage1": {"target_zones": []}},
        old_subjects={"s1": "cage1", "s2": "cage1"},
        new_subjects={"s1": "mA", "s2": "mB"},
    )
    assert set(out) == {"mA", "mB"} and notes == []


# ----------------------------------------------------------------------
# Subject-level grouping follows the current name
# ----------------------------------------------------------------------


def test_current_subjects_maps_frozen_keys_to_current_names(tmp_path: Path) -> None:
    manifest = _fear_manifest(tmp_path, ["m1_cond1", "m1_ext"])
    ImportService().apply_subject_name_settings(manifest, DEFAULT)
    s1, s2 = _sid(manifest, "m1_cond1"), _sid(manifest, "m1_ext")
    df = pd.DataFrame({
        "animal_id": ["m1_cond1", "m1_ext", "m1_cond1:mouse1", "imported_x"],
        "session_id": [s1, s2, s1, "session_elsewhere"],
    })
    assert current_subjects(df, manifest).tolist() == ["m1", "m1", "m1", "imported_x"]


def test_current_subjects_groups_multi_animal_tracks_by_session(tmp_path: Path) -> None:
    # track_0/track_1 are per-video labels: each session's pair is its own unit,
    # never pooled with the same track label from another session.
    manifest = _fear_manifest(tmp_path, ["pairA", "pairB"])
    sa, sb = _sid(manifest, "pairA"), _sid(manifest, "pairB")
    df = pd.DataFrame({
        "animal_id": ["track_0", "track_1", "track_0", "track_1"],
        "session_id": [sa, sa, sb, sb],
    })
    out = current_subjects(df, manifest).tolist()
    assert out[0] == out[1] and out[2] == out[3] and out[0] != out[2]


def test_subject_split_keeps_a_subjects_sessions_together(tmp_path: Path) -> None:
    stems = [f"m{i}_{t}" for i in range(1, 7) for t in ("cond1", "ext")]
    manifest = _fear_manifest(tmp_path, stems)
    ImportService().apply_subject_name_settings(manifest, DEFAULT)
    root = _project(tmp_path, manifest)
    rows = [
        {"animal_id": s, "session_id": _sid(manifest, s), "f0": float(i)}
        for i, s in enumerate(stems) for _ in range(3)
    ]
    df = pd.DataFrame(rows)

    train_idx, val_idx = ActiveLearningTrainerService._split(
        df, "group_shuffle_subject", 0.34, 0, project_root=root
    )
    subject = df["animal_id"].str.split("_").str[0].to_numpy()
    assert not set(subject[train_idx]) & set(subject[val_idx])
    assert len(np.intersect1d(train_idx, val_idx)) == 0
