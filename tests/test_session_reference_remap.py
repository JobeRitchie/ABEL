"""Review work must follow a recording when its session id changes.

A recording keeps its filename across re-imports but is minted a fresh
``session_id`` each time, and labels/decisions embed that id in their keys. The
Novel-object project de-duplicated 120 sessions down to 60 and left 273 reviewed
segments addressed to session ids the project no longer had: they matched no
feature row, were silently dropped from training, and were re-attempted (each
time re-reading the multi-GB frame-pose store) on every retrain.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from abel.models.schemas import ImportManifest, LinkedSession, PoseAsset, VideoAsset
from abel.services.import_service import ImportService
from abel.storage.file_store import read_json, write_json


@pytest.fixture()
def service() -> ImportService:
    return ImportService()


def _manifest(tmp_path: Path, session_ids: list[str], video_name: str = "CBMRE04_Acclimation.mp4") -> ImportManifest:
    """A manifest where every session points at the *same* recording."""
    video = VideoAsset(asset_id="vid_1", source_path=str(tmp_path / "raw" / "videos" / video_name))
    pose = PoseAsset(asset_id="pose_1", source_path=str(tmp_path / "raw" / "pose" / "CBMRE04_Acclimation.csv"), format="csv")
    return ImportManifest(
        videos=[video],
        poses=[pose],
        linked_sessions=[
            LinkedSession(session_id=sid, video_asset_id=video.asset_id, pose_asset_id=pose.asset_id)
            for sid in session_ids
        ],
    )


def _write_labels(root: Path, segment_ids: list[str]) -> Path:
    path = root / "derived" / "review_labels" / "reviewer_labels.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"segment_id": sid, "review_label": "Rear", "reviewer_id": "reviewer", "confidence": 1.0}
            for sid in segment_ids
        ]
    ).to_parquet(path, index=False)
    return path


def test_removing_a_duplicate_session_repoints_its_review_work(
    service: ImportService, tmp_path: Path
) -> None:
    manifest = _manifest(tmp_path, ["session_aaaaaaaa", "session_bbbbbbbb"])
    labels_path = _write_labels(
        tmp_path,
        [
            "bout_1234_session_bbbbbbbb_100_129",  # reviewed against the duplicate
            "session_aaaaaaaa_200_229",  # already on the canonical session
        ],
    )
    write_json(
        tmp_path / "derived" / "review_tables" / "review_decisions.json",
        {"decisions": [{"decision_id": "d1", "clip_id": "rand_session_bbbbbbbb_100_129"}]},
    )
    write_json(
        tmp_path / "config" / "seeds.json",
        {"seeds": [{"session_id": "session_bbbbbbbb", "start_frame": 10, "end_frame": 20}]},
    )

    summary = service.remove_sessions(tmp_path, manifest, ["session_bbbbbbbb"])

    labels = pd.read_parquet(labels_path)
    assert sorted(labels["segment_id"]) == [
        "bout_1234_session_aaaaaaaa_100_129",
        "session_aaaaaaaa_200_229",
    ], "a label on a de-duplicated session must follow the recording, not be orphaned"
    assert len(labels) == 2, "no review work may be lost"

    decisions = read_json(tmp_path / "derived" / "review_tables" / "review_decisions.json", {})
    assert decisions["decisions"][0]["clip_id"] == "rand_session_aaaaaaaa_100_129"

    seeds = read_json(tmp_path / "config" / "seeds.json", {})
    assert seeds["seeds"] == [
        {"session_id": "session_aaaaaaaa", "start_frame": 10, "end_frame": 20}
    ], "a seed on the duplicate is still valid work on the surviving session"

    assert summary["remapped"] == 3
    assert summary["sessions"] == 1


def test_removing_the_only_session_for_a_recording_prunes_its_review_work(
    service: ImportService, tmp_path: Path
) -> None:
    """Deleting a recording outright prunes its now-orphaned review work.

    There is nothing to re-point onto (the recording has left the project), and a
    duplicate is re-pointed *before* this — see
    :func:`test_removing_a_duplicate_session_repoints_its_review_work`. Left in
    place, the decisions/labels would linger in the review queue forever as a raw
    ``session_<hex>`` code with no subject and no extracted clip.
    """
    manifest = _manifest(tmp_path, ["session_aaaaaaaa"])
    labels_path = _write_labels(tmp_path, ["session_aaaaaaaa_100_129"])
    write_json(
        tmp_path / "derived" / "review_tables" / "review_decisions.json",
        {"decisions": [{"decision_id": "d1", "clip_id": "rand_session_aaaaaaaa_100_129"}]},
    )
    write_json(
        tmp_path / "derived" / "review_labels" / "soundboard_labels.json",
        {"windows": {"seg_M1_session_aaaaaaaa_100_129": [{"behavior_id": "b1"}]}},
    )

    summary = service.remove_sessions(tmp_path, manifest, ["session_aaaaaaaa"])

    assert summary["remapped"] == 0
    assert list(pd.read_parquet(labels_path)["segment_id"]) == []
    decisions = read_json(tmp_path / "derived" / "review_tables" / "review_decisions.json", {})
    assert decisions["decisions"] == []
    soundboard = read_json(tmp_path / "derived" / "review_labels" / "soundboard_labels.json", {})
    assert soundboard["windows"] == {}


def test_removing_a_session_keeps_review_work_for_the_sessions_that_remain(
    service: ImportService, tmp_path: Path
) -> None:
    """Pruning is scoped to the removed session — a co-existing session is untouched."""
    video_a = VideoAsset(asset_id="vid_a", source_path=str(tmp_path / "raw" / "videos" / "A.mp4"))
    video_b = VideoAsset(asset_id="vid_b", source_path=str(tmp_path / "raw" / "videos" / "B.mp4"))
    pose = PoseAsset(asset_id="pose_1", source_path=str(tmp_path / "raw" / "pose" / "p.csv"), format="csv")
    manifest = ImportManifest(
        videos=[video_a, video_b],
        poses=[pose],
        linked_sessions=[
            LinkedSession(session_id="session_aaaaaaaa", video_asset_id="vid_a", pose_asset_id="pose_1"),
            LinkedSession(session_id="session_bbbbbbbb", video_asset_id="vid_b", pose_asset_id="pose_1"),
        ],
    )
    labels_path = _write_labels(
        tmp_path, ["seg_M1_session_aaaaaaaa_10_39", "seg_M2_session_bbbbbbbb_10_39"]
    )
    write_json(
        tmp_path / "derived" / "review_tables" / "review_decisions.json",
        {
            "decisions": [
                {"decision_id": "d1", "clip_id": "rand_session_aaaaaaaa_10_39"},
                {"decision_id": "d2", "clip_id": "rand_session_bbbbbbbb_10_39"},
            ]
        },
    )

    service.remove_sessions(tmp_path, manifest, ["session_aaaaaaaa"])

    assert list(pd.read_parquet(labels_path)["segment_id"]) == ["seg_M2_session_bbbbbbbb_10_39"]
    decisions = read_json(tmp_path / "derived" / "review_tables" / "review_decisions.json", {})
    assert [d["clip_id"] for d in decisions["decisions"]] == ["rand_session_bbbbbbbb_10_39"]


def test_stale_session_remap_recovers_ids_from_the_registry(
    service: ImportService, tmp_path: Path
) -> None:
    """Sessions already dropped from the manifest resolve via the import registry."""
    manifest = _manifest(tmp_path, ["session_old00000"])
    service.save_manifest(tmp_path, manifest)  # registry now knows session_old00000

    # The recording is re-imported and minted a new id; the old one is gone from the
    # manifest but survives in the registry, keyed by the same video filename.
    manifest.linked_sessions[0].session_id = "session_new00000"
    service.save_manifest(tmp_path, manifest)

    assert service.stale_session_remap(tmp_path, manifest) == {"session_old00000": "session_new00000"}


def test_stale_remap_is_skipped_when_the_recording_is_ambiguous(
    service: ImportService, tmp_path: Path
) -> None:
    """Two live sessions for one video: no unique target, so leave the labels alone."""
    manifest = _manifest(tmp_path, ["session_old00000"])
    service.save_manifest(tmp_path, manifest)

    manifest = _manifest(tmp_path, ["session_dup10000", "session_dup20000"])
    service.save_manifest(tmp_path, manifest)

    assert service.stale_session_remap(tmp_path, manifest) == {}
