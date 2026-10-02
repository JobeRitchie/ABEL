"""Switching linked sessions to different pose tracking must not detach anything.

A new DLC/SLEAP run of the same recordings keeps the session (and so every
label, clip, ROI and analytics group keyed to it), carries animal ids onto the
new tracks, and clears only what was computed from the old file.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from abel.models.schemas import PoseAsset, VideoAsset
from abel.services.feature_prep_service import FeaturePrepService
from abel.services.import_service import ImportService


def _stub_video(path: Path, settings) -> VideoAsset:
    return VideoAsset(
        asset_id=f"vid_{uuid4().hex[:10]}",
        source_path=str(path),
        subject_id=ImportService.extract_subject_name(path, settings),
        session_id=ImportService.extract_session_type(path, settings),
    )


def _stub_pose(path: Path, settings) -> PoseAsset:
    return PoseAsset(
        asset_id=f"pose_{uuid4().hex[:10]}",
        source_path=str(path),
        format=path.suffix.lstrip("."),
        frame_count=100,
        subject_id=ImportService.extract_subject_name(path, settings),
        session_id=ImportService.extract_session_type(path, settings),
    )


@pytest.fixture()
def service(monkeypatch: pytest.MonkeyPatch) -> ImportService:
    monkeypatch.setattr(ImportService, "_video_asset", staticmethod(_stub_video))
    monkeypatch.setattr(ImportService, "_pose_asset", staticmethod(_stub_pose))
    return ImportService()


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _project(service: ImportService, tmp_path: Path, names=("m1_cond1", "m2_cond1")):
    videos = [_touch(tmp_path / "vid" / f"{n}.mp4") for n in names]
    poses = [_touch(tmp_path / "dlc1" / f"{n}DLC_resnet50_A_shuffle1_snapshot_100.h5") for n in names]
    return service.build_manifest(videos, poses)


def test_new_run_matches_existing_sessions_and_keeps_their_identity(service, tmp_path):
    manifest = _project(service, tmp_path)
    before = {s.session_id: (s.subject_key, s.subject_id) for s in manifest.linked_sessions}
    manifest.linked_sessions[0].subject_id = "renamed"  # subject_key stays frozen

    new_files = [
        _touch(tmp_path / "sleap" / "m1_cond1.mp4.predictions.sleap.h5"),
        _touch(tmp_path / "sleap" / "m2_cond1.mp4.predictions.sleap.h5"),
    ]
    known = {p.asset_id for p in manifest.poses}
    service.merge_new_files(manifest, [], new_files)
    new_ids = {p.asset_id for p in manifest.poses} - known
    # Videos are taken, so a plain merge leaves the new run unpaired...
    assert len(manifest.linked_sessions) == 2
    replacements = service.find_pose_replacements(manifest, pose_asset_ids=new_ids)
    assert set(replacements) == set(before)

    results = service.replace_session_poses(manifest, replacements)

    assert len(results) == 2
    assert {s.session_id for s in manifest.linked_sessions} == set(before)
    for s in manifest.linked_sessions:
        assert s.subject_key == before[s.session_id][0]
        pose = next(p for p in manifest.poses if p.asset_id == s.pose_asset_id)
        assert "sleap" in Path(pose.source_path).parts
    # Old assets are gone, so a later Auto Match cannot offer them back.
    assert len(manifest.poses) == 2
    assert service.find_pose_replacements(manifest) == {}


def test_csv_and_h5_of_the_same_run_are_not_a_replacement(service, tmp_path):
    manifest = _project(service, tmp_path, names=("m1_cond1",))
    twin = _touch(tmp_path / "dlc1" / "m1_cond1DLC_resnet50_A_shuffle1_snapshot_100.csv")
    service.merge_new_files(manifest, [], [twin])
    assert service.find_pose_replacements(manifest) == {}


def _multi_session(service, tmp_path, old_inds, corrections=()):
    manifest = _project(service, tmp_path, names=("pair1",))
    session = manifest.linked_sessions[0]
    session.individuals = list(old_inds)
    session.individual_subject_map = {i: i for i in old_inds}
    session.individual_subject_map[old_inds[0]] = "green"
    session.identity_corrections = list(corrections)
    return manifest, session


def _swap_to(service, manifest, session, new_inds, frame_count=100):
    new = PoseAsset(
        asset_id="pose_new", source_path="x/pair1.slp.h5", format="h5",
        frame_count=frame_count, individuals=list(new_inds),
    )
    manifest.poses.append(new)
    return service.replace_session_poses(manifest, {session.session_id: "pose_new"})[0]


def test_renamed_tracks_keep_the_old_animal_ids(service, tmp_path):
    manifest, session = _multi_session(
        service, tmp_path, ["individual1", "individual2"],
        corrections=[{"frame": 50, "a": "individual1", "b": "individual2"}],
    )
    result = _swap_to(service, manifest, session, ["track_0", "track_1"])

    assert session.individuals == ["track_0", "track_1"]
    assert session.individual_subject_map == {"track_0": "green", "track_1": "individual2"}
    assert session.identity_corrections == []
    assert any("swap correction" in n for n in result.notes)
    assert any("by position" in n for n in result.notes)


def test_same_track_names_keep_their_own_ids(service, tmp_path):
    manifest, session = _multi_session(service, tmp_path, ["track_0", "track_1"])
    result = _swap_to(service, manifest, session, ["track_1", "track_0", "track_2"])

    assert session.individual_subject_map == {
        "track_1": "track_1", "track_0": "green", "track_2": "track_2",
    }
    assert not any("by position" in n for n in result.notes)
    assert any("track_2" in n for n in result.notes)


def test_frame_count_change_is_reported(service, tmp_path):
    manifest = _project(service, tmp_path, names=("m1_cond1",))
    session = manifest.linked_sessions[0]
    result = _swap_to(service, manifest, session, [], frame_count=90)
    assert any("frame count" in n for n in result.notes)
    assert session.individuals == []


def test_invalidation_clears_only_the_replaced_session(tmp_path):
    pytest.importorskip("pyarrow")
    d = tmp_path / "derived"
    for sid in ("session_aa", "session_bb"):
        for family in ("pose_features", "context_features"):
            p = d / family / "sessions"
            p.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"frame": [0]}).to_parquet(p / f"{sid}.parquet", index=False)
        _touch(d / "r3d_features" / f"{sid}.parquet")
        _touch(d / "pose_clean" / f"{sid}.parquet")
    ts = d / "training_sets" / "training_set.parquet"
    ts.parent.mkdir(parents=True)
    pd.DataFrame({
        "segment_id": ["seg_m1_session_aa_0_15", "seg_m2_session_bb_0_15"],
        "session_id": ["session_aa", "session_bb"],
    }).to_parquet(ts, index=False)
    enr = d / "representations" / "enriched_segments.parquet"
    enr.parent.mkdir(parents=True)
    pd.DataFrame({"segment_id": ["seg_m1_session_aa_3_18", "seg_m2_session_bb_3_18"]}).to_parquet(enr, index=False)

    out = FeaturePrepService.invalidate_replaced_pose(tmp_path, ["session_aa"])

    assert out["rows"] == 2
    assert FeaturePrepService.cached_pose_sessions(tmp_path) == {"session_bb"}
    assert not (d / "r3d_features" / "session_aa.parquet").exists()
    assert not (d / "pose_clean" / "session_aa.parquet").exists()
    assert (d / "r3d_features" / "session_bb.parquet").exists()
    assert pd.read_parquet(ts)["session_id"].tolist() == ["session_bb"]
    assert pd.read_parquet(enr)["segment_id"].tolist() == ["seg_m2_session_bb_3_18"]
