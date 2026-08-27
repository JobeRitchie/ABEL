"""ABEL position export for TRACY: file contract, subject naming, track point.

The load-bearing assertion is the *contract*: TRACY reads these files by column
name and keys its special case off the ``ABELposition`` filename marker, so the
header, the frame numbering and the filename all have to stay exactly as
written here.  ``test_tracy_reads_the_exported_file`` closes the loop by parsing
an exported file the same way TRACY's ``_read_abel_position_file`` does.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.models.schemas import (
    ImportManifest,
    LinkedSession,
    PoseAsset,
    VideoAsset,
)
from abel.services.export_service import ExportService
from abel.services.import_service import ImportService

BODY_PARTS = ("nose", "body_center", "tail_base")
N_FRAMES = 300
FPS = 30.0


def _write_pose(path: Path, offsets: dict[str, tuple[float, float]]) -> None:
    """DLC-style single-animal CSV: 3 header rows (scorer/bodyparts/coords).

    Each body part traces ``frame + dx`` / ``2 * frame + dy`` so the exported
    track identifies which part was used.
    """
    frames = np.arange(N_FRAMES, dtype=float)
    cols, data = [], {}
    for bp in BODY_PARTS:
        dx, dy = offsets[bp]
        for coord in ("x", "y", "likelihood"):
            cols.append(("DLC", bp, coord))
            if coord == "x":
                data[("DLC", bp, coord)] = frames + dx
            elif coord == "y":
                data[("DLC", bp, coord)] = 2.0 * frames + dy
            else:
                data[("DLC", bp, coord)] = np.full(N_FRAMES, 0.99)
    df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(cols))
    df.index.name = "coords"
    df.to_csv(path)


def _make_project(tmp_path: Path, sessions: list[tuple[str, str, str]]) -> Path:
    """Build a minimal project. *sessions* is [(session_id, subject, video_stem)]."""
    root = tmp_path / "project"
    (root / "derived" / "review_tables").mkdir(parents=True)
    (root / "raw").mkdir(parents=True, exist_ok=True)

    videos, poses, linked = [], [], []
    for i, (session_id, subject, stem) in enumerate(sessions):
        pose_path = root / "raw" / f"{stem}.csv"
        _write_pose(
            pose_path,
            {"nose": (10.0, 0.0), "body_center": (0.0, 0.0), "tail_base": (-10.0, 0.0)},
        )
        videos.append(
            VideoAsset(asset_id=f"v{i}", source_path=f"{stem}.avi",
                       subject_id=subject, fps=FPS, frame_count=N_FRAMES)
        )
        poses.append(
            PoseAsset(asset_id=f"p{i}", source_path=str(pose_path),
                      local_path=str(pose_path), format="csv", subject_id=subject)
        )
        linked.append(
            LinkedSession(session_id=session_id, video_asset_id=f"v{i}",
                          pose_asset_id=f"p{i}", subject_id=subject)
        )

    ImportService().save_manifest(
        root, ImportManifest(videos=videos, poses=poses, linked_sessions=linked)
    )
    return root


@pytest.fixture
def single_session_project(tmp_path: Path) -> Path:
    return _make_project(tmp_path, [("s1", "MS1", "MS1_OpenField")])


def test_writes_one_file_per_session_with_the_tracy_marker(single_session_project: Path) -> None:
    service = ExportService()
    service.set_project(single_session_project)
    out = service.export_abel_position_csv()

    assert out.success, out.warnings
    assert len(out.output_paths) == 1
    path = out.output_paths[0]
    # TRACY keys its special case off this exact marker in the filename.
    assert path.name == "MS1ABELposition.csv"
    assert path.parent == single_session_project / "exports" / "tracy"
    assert out.n_rows == N_FRAMES


def test_column_contract_and_frame_numbering(single_session_project: Path) -> None:
    service = ExportService()
    service.set_project(single_session_project)
    out = service.export_abel_position_csv()

    df = pd.read_csv(out.output_paths[0])
    # TRACY matches these names case-insensitively; changing them breaks import.
    assert list(df.columns) == ["frame", "timestamp", "X", "Y"]
    assert len(df) == N_FRAMES
    # Frames are 0-based video frames -- the same numbering as ABEL's boutframes.
    assert df["frame"].iloc[0] == 0
    assert df["frame"].iloc[-1] == N_FRAMES - 1
    # timestamp is video elapsed seconds, not a computer clock.
    assert np.allclose(df["timestamp"].to_numpy(), np.arange(N_FRAMES) / FPS, atol=1e-3)


def test_track_point_selects_the_named_body_part(single_session_project: Path) -> None:
    service = ExportService()
    service.set_project(single_session_project)

    nose = pd.read_csv(
        service.export_abel_position_csv(track_point="nose").output_paths[0]
    )
    tail = pd.read_csv(
        service.export_abel_position_csv(track_point="tail_base").output_paths[0]
    )
    centroid = pd.read_csv(
        service.export_abel_position_csv(track_point="centroid").output_paths[0]
    )

    # The project's pose smoothing is applied on load, so compare the interior
    # where the rolling window is full rather than the smoothed edges.
    interior = slice(10, N_FRAMES - 10)
    frames = np.arange(N_FRAMES, dtype=float)[interior]
    assert np.allclose(nose["X"].to_numpy()[interior], frames + 10.0, atol=1e-3)
    assert np.allclose(tail["X"].to_numpy()[interior], frames - 10.0, atol=1e-3)
    # Centroid is the mean over all three parts: +10, 0, -10 -> +0.
    assert np.allclose(centroid["X"].to_numpy()[interior], frames, atol=1e-3)


def test_unknown_body_part_falls_back_to_centroid_with_a_warning(
    single_session_project: Path,
) -> None:
    service = ExportService()
    service.set_project(single_session_project)
    out = service.export_abel_position_csv(track_point="left_forepaw")

    assert out.success
    assert any("left_forepaw" in w and "centroid" in w for w in out.warnings), out.warnings
    df = pd.read_csv(out.output_paths[0])
    interior = slice(10, N_FRAMES - 10)
    assert np.allclose(
        df["X"].to_numpy()[interior], np.arange(N_FRAMES, dtype=float)[interior], atol=1e-3
    )


def test_repeated_sessions_get_tracy_session_suffixed_subject_ids(tmp_path: Path) -> None:
    """A subject recorded twice must produce the IDs TRACY derives from FPData."""
    root = _make_project(
        tmp_path,
        [("s1", "CAB01", "CAB01_Alcohol"), ("s2", "CAB01", "CAB01_Fentanyl")],
    )
    service = ExportService()
    service.set_project(root)
    out = service.export_abel_position_csv()

    assert out.success, out.warnings
    names = sorted(p.name for p in out.output_paths)
    assert names == ["CAB01_AlcoholABELposition.csv", "CAB01_FentanylABELposition.csv"]


def test_session_filter_limits_the_export(tmp_path: Path) -> None:
    root = _make_project(
        tmp_path,
        [("s1", "CAB01", "CAB01_Alcohol"), ("s2", "CAB01", "CAB01_Fentanyl")],
    )
    service = ExportService()
    service.set_project(root)
    out = service.export_abel_position_csv(session_filter=["s2"])

    assert out.success
    assert [p.name for p in out.output_paths] == ["CAB01ABELposition.csv"]


def test_missing_pose_file_is_reported_not_raised(tmp_path: Path) -> None:
    root = _make_project(tmp_path, [("s1", "MS1", "MS1_OpenField")])
    (root / "raw" / "MS1_OpenField.csv").unlink()

    service = ExportService()
    service.set_project(root)
    out = service.export_abel_position_csv()

    assert not out.success
    assert any("pose file not found" in w for w in out.warnings)


def test_tracy_reads_the_exported_file(single_session_project: Path) -> None:
    """Parse an exported file the way TRACY's _read_abel_position_file does.

    Kept as an inline reimplementation rather than an import because TRACY is a
    separate application; if the two ever drift, this fails loudly here.
    """
    service = ExportService()
    service.set_project(single_session_project)
    out = service.export_abel_position_csv(track_point="body_center")
    path = out.output_paths[0]

    marker = ExportService.ABEL_POSITION_MARKER.lower()
    assert marker in path.name.lower(), "TRACY detects the file by this marker"

    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    for name in ("frame", "timestamp", "x", "y"):
        assert name in cols, f"TRACY looks up column '{name}' by name"

    frames = pd.to_numeric(df[cols["frame"]], errors="coerce").to_numpy(dtype=float)
    ts = pd.to_numeric(df[cols["timestamp"]], errors="coerce").to_numpy(dtype=float)
    assert np.isfinite(frames).all()
    assert np.all(np.diff(frames) > 0), "frames must be strictly increasing"

    # TRACY recovers the video frame rate from the timestamp column to
    # cross-check its 'Boutframes video FPS' setting.
    file_fps = 1.0 / float(np.median(np.diff(ts)))
    assert abs(file_fps - FPS) < 0.5, file_fps
