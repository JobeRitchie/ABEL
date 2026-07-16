"""Re-importing a recording from a moved folder must not duplicate its session.

Regression cover for the Novel-object project that ended up with 120 sessions for
60 recordings: the project folder moved, sources were relocated, and a re-run of
Auto-Match re-added every file because the dedup key was the absolute path string.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abel.services.import_service import ImportService


def _make_media(root: Path, names: list[str]) -> tuple[list[Path], list[Path]]:
    videos_dir = root / "raw" / "videos"
    poses_dir = root / "raw" / "pose"
    videos_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)
    videos, poses = [], []
    for name in names:
        video = videos_dir / f"{name}.mp4"
        pose = poses_dir / f"{name}DLC_Resnet50_shuffle2_snapshot_190.csv"
        video.write_bytes(b"")
        pose.write_text("")
        videos.append(video)
        poses.append(pose)
    return videos, poses


@pytest.fixture()
def service() -> ImportService:
    return ImportService()


def test_reimport_from_moved_folder_repoints_instead_of_duplicating(
    service: ImportService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ImportService, "_video_asset", staticmethod(_stub_video))
    monkeypatch.setattr(ImportService, "_pose_asset", staticmethod(_stub_pose))

    old_root = tmp_path / "Models For Manuscript" / "Novel object"
    videos, poses = _make_media(old_root, ["CBMRE01_Acclimation", "CBMRE02_Acclimation"])
    manifest = service.build_manifest(videos, poses)
    assert len(manifest.linked_sessions) == 2
    original_ids = [s.session_id for s in manifest.linked_sessions]

    # The project moves. The old location is gone; the same files now live elsewhere.
    new_root = tmp_path / "Novel object"
    new_videos, new_poses = _make_media(new_root, ["CBMRE01_Acclimation", "CBMRE02_Acclimation"])
    for path in videos + poses:
        path.unlink()

    service.merge_new_files(manifest, new_videos, new_poses)

    assert len(manifest.videos) == 2, "re-added videos must not duplicate existing assets"
    assert len(manifest.poses) == 2
    assert len(manifest.linked_sessions) == 2, "a moved recording must not gain a 2nd session"
    assert [s.session_id for s in manifest.linked_sessions] == original_ids, (
        "session ids must survive a relocation — labels and derived data hang off them"
    )
    assert {Path(v.source_path).parent for v in manifest.videos} == {new_root / "raw" / "videos"}


def test_readding_same_files_in_place_is_a_noop(
    service: ImportService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ImportService, "_video_asset", staticmethod(_stub_video))
    monkeypatch.setattr(ImportService, "_pose_asset", staticmethod(_stub_pose))

    videos, poses = _make_media(tmp_path / "proj", ["CBMRE01_Acclimation"])
    manifest = service.build_manifest(videos, poses)

    service.merge_new_files(manifest, videos, poses)

    assert len(manifest.videos) == 1
    assert len(manifest.linked_sessions) == 1


def test_live_source_path_is_not_swapped_by_a_stale_copy(
    service: ImportService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-named file elsewhere must not silently re-point a still-valid asset."""
    monkeypatch.setattr(ImportService, "_video_asset", staticmethod(_stub_video))
    monkeypatch.setattr(ImportService, "_pose_asset", staticmethod(_stub_pose))

    videos, poses = _make_media(tmp_path / "current", ["CBMRE01_Acclimation"])
    manifest = service.build_manifest(videos, poses)

    stale_videos, stale_poses = _make_media(tmp_path / "old_copy", ["CBMRE01_Acclimation"])
    service.merge_new_files(manifest, stale_videos, stale_poses)

    assert len(manifest.videos) == 1
    assert len(manifest.linked_sessions) == 1
    assert Path(manifest.videos[0].source_path) == videos[0], "live path must be kept"


def test_genuinely_new_recordings_still_import(
    service: ImportService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ImportService, "_video_asset", staticmethod(_stub_video))
    monkeypatch.setattr(ImportService, "_pose_asset", staticmethod(_stub_pose))

    videos, poses = _make_media(tmp_path / "proj", ["CBMRE01_Acclimation"])
    manifest = service.build_manifest(videos, poses)

    more_videos, more_poses = _make_media(tmp_path / "proj", ["CBMRE02_TestingDay2"])
    service.merge_new_files(manifest, more_videos, more_poses)

    assert len(manifest.videos) == 2
    assert len(manifest.linked_sessions) == 2


def test_find_duplicate_sessions_reports_canonical_first(
    service: ImportService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repair primitive must keep the first-linked session as canonical."""
    monkeypatch.setattr(ImportService, "_video_asset", staticmethod(_stub_video))
    monkeypatch.setattr(ImportService, "_pose_asset", staticmethod(_stub_pose))

    videos, poses = _make_media(tmp_path / "proj", ["CBMRE01_Acclimation"])
    manifest = service.build_manifest(videos, poses)
    assert service.find_duplicate_sessions(manifest) == {}

    # Simulate the corrupted state: a second asset + session for the same recording.
    dup_videos, dup_poses = _make_media(tmp_path / "elsewhere", ["CBMRE01_Acclimation"])
    dup_manifest = service.build_manifest(dup_videos, dup_poses)
    manifest.videos.extend(dup_manifest.videos)
    manifest.poses.extend(dup_manifest.poses)
    manifest.linked_sessions.extend(dup_manifest.linked_sessions)

    canonical = manifest.linked_sessions[0].session_id
    duplicate = manifest.linked_sessions[1].session_id
    assert service.find_duplicate_sessions(manifest) == {canonical: [duplicate]}


# ---------------------------------------------------------------------------
# The real asset builders probe the media with OpenCV/pandas; these fixtures are
# empty files, so stub the probing and keep the identity/linking logic under test.

def _stub_video(path: Path, settings) -> object:
    from uuid import uuid4

    from abel.models.schemas import VideoAsset

    return VideoAsset(
        asset_id=f"vid_{uuid4().hex[:10]}",
        source_path=str(path),
        local_path=None,
        subject_id=ImportService.extract_subject_name(path, settings),
        session_id=ImportService.extract_session_type(path, settings),
        pixels_per_mm=None,
    )


def _stub_pose(path: Path, settings) -> object:
    from uuid import uuid4

    from abel.models.schemas import PoseAsset

    return PoseAsset(
        asset_id=f"pose_{uuid4().hex[:10]}",
        source_path=str(path),
        local_path=None,
        format=path.suffix.lstrip("."),
        frame_count=0,
        body_parts=[],
        individuals=[],
        has_likelihood=True,
        subject_id=ImportService.extract_subject_name(path, settings),
        session_id=ImportService.extract_session_type(path, settings),
    )
