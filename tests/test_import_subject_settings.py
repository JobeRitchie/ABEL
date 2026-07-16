from pathlib import Path

import pytest

from abel.storage.file_store import read_json, write_json
from abel.models.schemas import ImportManifest, ImportNameSettings, LinkedSession, PoseAsset, VideoAsset
from abel.services.import_service import ImportService


def test_extract_subject_name_default_pattern() -> None:
    # The default regex captures the leading alphanumeric token up to the first
    # separator (``_``/``.``/``DLC``), so a separator-delimited name resolves to
    # the subject id.  Delimiter-less names need a custom regex (see below).
    subject = ImportService.extract_subject_name(
        Path("DG01_BehavioralCamera0.avi"),
        ImportNameSettings(),
    )
    assert subject == "DG01"


def test_extract_subject_name_custom_pattern() -> None:
    settings = ImportNameSettings(subject_regex=r"subject-([A-Z]{2}\d{2})", subject_group_index=1)
    subject = ImportService.extract_subject_name(Path("trial_subject-AB12_cam0.avi"), settings)
    assert subject == "AB12"


def test_build_manifest_sets_subject_ids() -> None:
    service = ImportService()
    manifest = service.build_manifest(
        [Path("DG01_BehavioralCamera0.avi")],
        [Path("DG01_BehavioralCamera0DLC_resnet50.csv")],
        subject_name_settings=ImportNameSettings(),
    )

    assert manifest.videos[0].subject_id == "DG01"
    assert manifest.poses[0].subject_id == "DG01"
    assert manifest.linked_sessions[0].subject_id == "DG01"


def test_auto_match_links_direct_dlc_suffix_names() -> None:
    service = ImportService()
    manifest = service.build_manifest(
        [Path("m1_cond1.mp4")],
        [Path("m1_cond1DLC_Resnet50_alison_ltpJan28shuffle1_snapshot_040.csv")],
        subject_name_settings=ImportNameSettings(),
    )

    assert len(manifest.linked_sessions) == 1


def test_auto_match_links_separator_before_dlc_suffix_names() -> None:
    service = ImportService()
    manifest = service.build_manifest(
        [Path("m1_cond1.mp4")],
        [Path("m1_cond1_DLC_Resnet50_alison_ltpJan28shuffle1_snapshot_040.csv")],
        subject_name_settings=ImportNameSettings(),
    )

    assert len(manifest.linked_sessions) == 1


def test_apply_subject_name_settings_updates_manifest() -> None:
    service = ImportService()
    manifest = ImportManifest(
        videos=[
            VideoAsset(
                asset_id="v1",
                source_path="DG01BehavioralCamera0.avi",
                subject_id="DG01",
            )
        ],
        poses=[
            PoseAsset(
                asset_id="p1",
                source_path="DG01BehavioralCamera0DLC_resnet50.csv",
                format="csv",
                subject_id="DG01",
            )
        ],
        linked_sessions=[
            LinkedSession(
                session_id="session_1",
                video_asset_id="v1",
                pose_asset_id="p1",
                subject_id="DG01",
            )
        ],
    )

    service.apply_subject_name_settings(
        manifest,
        ImportNameSettings(subject_regex=r"^(DG\d{2})Behavioral", subject_group_index=1),
    )
    service.update_session_subject(manifest, "session_1", "RatA")

    assert manifest.linked_sessions[0].subject_id == "RatA"
    assert manifest.videos[0].subject_id == "RatA"
    assert manifest.poses[0].subject_id == "RatA"


def test_remove_sessions_prunes_manifest_and_associated_data(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "derived" / "review_tables").mkdir(parents=True)
    (project_root / "derived" / "pose_features").mkdir(parents=True)
    (project_root / "derived" / "syllables").mkdir(parents=True)
    (project_root / "derived" / "motifs").mkdir(parents=True)
    (project_root / "config").mkdir(parents=True)

    manifest = ImportManifest(
        videos=[
            VideoAsset(asset_id="v1", source_path="DG01BehavioralCamera0.avi", subject_id="DG01"),
            VideoAsset(asset_id="v2", source_path="DG02BehavioralCamera0.avi", subject_id="DG02"),
        ],
        poses=[
            PoseAsset(asset_id="p1", source_path="DG01BehavioralCamera0DLC.csv", format="csv", subject_id="DG01"),
            PoseAsset(asset_id="p2", source_path="DG02BehavioralCamera0DLC.csv", format="csv", subject_id="DG02"),
        ],
        linked_sessions=[
            LinkedSession(session_id="session_keep", video_asset_id="v1", pose_asset_id="p1", subject_id="DG01"),
            LinkedSession(session_id="session_drop", video_asset_id="v2", pose_asset_id="p2", subject_id="DG02"),
        ],
    )

    (project_root / "derived" / "pose_features" / "session_drop.npz").write_bytes(b"x")
    (project_root / "derived" / "syllables" / "session_drop_syllables.npz").write_bytes(b"x")

    write_json(
        project_root / "config" / "seeds.json",
        {
            "seeds": [
                {"seed_id": "a", "behavior_id": "groom", "session_id": "session_drop", "start_frame": 1, "end_frame": 2},
                {"seed_id": "b", "behavior_id": "groom", "session_id": "session_keep", "start_frame": 1, "end_frame": 2},
            ]
        },
    )
    write_json(
        project_root / "derived" / "review_tables" / "candidate_windows.json",
        {
            "session_ids": ["session_keep", "session_drop"],
            "candidates": [
                {"window_id": "c1", "session_id": "session_drop", "start_frame": 0, "end_frame": 10},
                {"window_id": "c2", "session_id": "session_keep", "start_frame": 0, "end_frame": 10},
            ],
        },
    )
    # External window candidates (active-learning / temporal-bout review queues)
    # persist across tabs and must also be pruned so removed sessions do not leak
    # clips/windows into the review UI.
    write_json(
        project_root / "derived" / "review_tables" / "external_window_candidates.json",
        {
            "candidates": [
                {"window_id": "e1", "session_id": "session_drop", "start_frame": 0, "end_frame": 10},
                {"window_id": "e2", "session_id": "session_keep", "start_frame": 0, "end_frame": 10},
            ],
        },
    )

    service = ImportService()
    summary = service.remove_sessions(project_root, manifest, ["session_drop"])

    assert summary["sessions"] == 1
    assert len(manifest.linked_sessions) == 1
    assert manifest.linked_sessions[0].session_id == "session_keep"
    assert len(manifest.videos) == 1
    assert len(manifest.poses) == 1
    assert not (project_root / "derived" / "pose_features" / "session_drop.npz").exists()
    assert not (project_root / "derived" / "syllables" / "session_drop_syllables.npz").exists()

    external = read_json(
        project_root / "derived" / "review_tables" / "external_window_candidates.json", {}
    )
    kept_sessions = {row["session_id"] for row in external.get("candidates", [])}
    assert kept_sessions == {"session_keep"}


def test_remove_sessions_prunes_session_keyed_parquets(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    project_root = tmp_path / "project"
    (project_root / "derived" / "representations").mkdir(parents=True)
    (project_root / "derived" / "pose_features").mkdir(parents=True)

    manifest = ImportManifest(
        videos=[
            VideoAsset(asset_id="v1", source_path="DG01.avi", subject_id="DG01"),
            VideoAsset(asset_id="v2", source_path="DG02.avi", subject_id="DG02"),
        ],
        poses=[
            PoseAsset(asset_id="p1", source_path="DG01.csv", format="csv", subject_id="DG01"),
            PoseAsset(asset_id="p2", source_path="DG02.csv", format="csv", subject_id="DG02"),
        ],
        linked_sessions=[
            LinkedSession(session_id="session_keep", video_asset_id="v1", pose_asset_id="p1", subject_id="DG01"),
            LinkedSession(session_id="session_drop", video_asset_id="v2", pose_asset_id="p2", subject_id="DG02"),
        ],
    )

    frame_path = project_root / "derived" / "representations" / "frame_features.parquet"
    pd.DataFrame(
        {"session_id": ["session_keep", "session_keep", "session_drop"],
         "frame": [0, 1, 0], "feat": [1.0, 2.0, 3.0]}
    ).to_parquet(frame_path, index=False)

    pose_path = project_root / "derived" / "pose_features" / "frame_pose.parquet"
    pd.DataFrame(
        {"session_id": ["session_drop", "session_keep"], "frame": [0, 0], "x": [9.0, 8.0]}
    ).to_parquet(pose_path, index=False)

    service = ImportService()
    service.remove_sessions(project_root, manifest, ["session_drop"])

    frame_df = pd.read_parquet(frame_path)
    assert set(frame_df["session_id"]) == {"session_keep"}
    assert len(frame_df) == 2

    pose_df = pd.read_parquet(pose_path)
    assert set(pose_df["session_id"]) == {"session_keep"}


def test_update_session_pixels_per_mm_and_lookup() -> None:
    manifest = ImportManifest(
        videos=[VideoAsset(asset_id="v1", source_path="DG01BehavioralCamera0.avi", subject_id="DG01")],
        poses=[PoseAsset(asset_id="p1", source_path="DG01BehavioralCamera0DLC.csv", format="csv", subject_id="DG01")],
        linked_sessions=[
            LinkedSession(session_id="session_1", video_asset_id="v1", pose_asset_id="p1", subject_id="DG01")
        ],
    )

    service = ImportService()
    service.update_session_pixels_per_mm(manifest, "session_1", 3.5)

    assert manifest.linked_sessions[0].pixels_per_mm == 3.5
    assert manifest.videos[0].pixels_per_mm == 3.5
    assert service.pixels_per_mm_for_session(manifest, "session_1") == 3.5

    service.update_session_pixels_per_mm(manifest, "session_1", None)
    assert service.pixels_per_mm_for_session(manifest, "session_1") is None
