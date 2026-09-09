from pathlib import Path

from openpyxl import load_workbook

from abel.models.schemas import (
    CandidateWindow,
    ImportManifest,
    LinkedSession,
    ReviewDecision,
    ReviewDecisionType,
    VideoAsset,
)
from abel.services.export_service import ExportService
from abel.services.import_service import ImportService


def test_export_boutframes_handles_sheet_name_collisions_and_invalid_chars(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "derived" / "review_tables").mkdir(parents=True)

    # These subjects sanitize/truncate to the same prefix and include invalid Excel title chars.
    subject_a = "Subject:ABCDEF_123456789012345678901"
    subject_b = "Subject/ABCDEF_123456789012345678902"

    manifest = ImportManifest(
        videos=[
            VideoAsset(asset_id="v1", source_path="a.avi", subject_id=subject_a),
            VideoAsset(asset_id="v2", source_path="b.avi", subject_id=subject_b),
        ],
        linked_sessions=[
            LinkedSession(session_id="s1", video_asset_id="v1", pose_asset_id="p1", subject_id=subject_a),
            LinkedSession(session_id="s2", video_asset_id="v2", pose_asset_id="p2", subject_id=subject_b),
        ],
    )
    ImportService().save_manifest(project_root, manifest)

    candidates = [
        CandidateWindow(window_id="c1", session_id="s1", start_frame=10, end_frame=20, behavior_id="rear"),
        CandidateWindow(window_id="c2", session_id="s2", start_frame=30, end_frame=40, behavior_id="rear"),
    ]
    decisions = [
        ReviewDecision(
            decision_id="d1",
            clip_id="c1",
            reviewer="r",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.ACCEPT,
            adjusted_start_frame=11,
            adjusted_end_frame=19,
        ),
        ReviewDecision(
            decision_id="d2",
            clip_id="c2",
            reviewer="r",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.ACCEPT,
            adjusted_start_frame=31,
            adjusted_end_frame=39,
        ),
    ]

    service = ExportService()
    service.set_project(project_root)
    out = service.export_boutframes_xlsx(
        candidates,
        decisions,
        filename="boutframes.xlsx",
        include_end_frames=True,
    )

    assert out.success
    assert out.output_path is not None
    wb = load_workbook(out.output_path)

    assert "_bout_counts" in wb.sheetnames
    subject_sheets = [name for name in wb.sheetnames if name != "_bout_counts"]
    assert len(subject_sheets) == 2
    assert len(set(subject_sheets)) == 2

    for sheet_name in subject_sheets:
        assert len(sheet_name) <= 31
        for bad in [":", "\\", "/", "?", "*", "[", "]"]:
            assert bad not in sheet_name

    first_subject_rows = list(wb[subject_sheets[0]].iter_rows(min_row=1, max_row=2, values_only=True))
    assert first_subject_rows[0] == ("rear__start", "rear__end")


def _manifest_for(sessions: list[tuple[str, str, str]]) -> ImportManifest:
    """Build a manifest from ``(session_id, subject, video_stem)`` triples."""
    return ImportManifest(
        videos=[
            VideoAsset(asset_id=f"v_{sid}", source_path=f"{stem}.avi", subject_id=subject)
            for sid, subject, stem in sessions
        ],
        linked_sessions=[
            LinkedSession(
                session_id=sid,
                video_asset_id=f"v_{sid}",
                pose_asset_id=f"p_{sid}",
                subject_id=subject,
            )
            for sid, subject, _stem in sessions
        ],
    )


def _accepted(sessions: list[tuple[str, str, str]]):
    candidates = [
        CandidateWindow(
            window_id=f"c_{sid}", session_id=sid, start_frame=10, end_frame=20, behavior_id="rear"
        )
        for sid, _subject, _stem in sessions
    ]
    decisions = [
        ReviewDecision(
            decision_id=f"d_{sid}",
            clip_id=f"c_{sid}",
            reviewer="r",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.ACCEPT,
            adjusted_start_frame=11,
            adjusted_end_frame=19,
        )
        for sid, _subject, _stem in sessions
    ]
    return candidates, decisions


def test_one_duplicate_session_does_not_split_the_workbook(tmp_path: Path) -> None:
    """A lone multi-session subject must not shatter the export into per-subject files.

    Timestamped video filenames give every session its own derived session
    type, so the old "any subject has 2 sessions -> split by type" rule wrote
    one workbook per subject instead of the combined workbook TRACY reads.
    """
    project_root = tmp_path / "project"
    (project_root / "derived" / "review_tables").mkdir(parents=True)

    sessions = [
        ("s1", "m1", "m1_2024-01-07T11_15_57"),
        ("s2", "m2", "m2_2024-01-07T11_27_34"),
        ("s3", "m3", "m3_2024-01-07T11_43_50"),
        ("s4", "m4", "m4_2024-01-07T11_58_49"),
        ("s5", "m4", "m4_2024-01-07T12_13_14"),  # the one duplicate
    ]
    ImportService().save_manifest(project_root, _manifest_for(sessions))
    candidates, decisions = _accepted(sessions)

    service = ExportService()
    service.set_project(project_root)
    out = service.export_boutframes_xlsx(candidates, decisions, include_merged_projects=False)

    assert out.success
    assert len(out.output_paths) == 1
    wb = load_workbook(out.output_path)
    sheets = [name for name in wb.sheetnames if name != "_bout_counts"]
    # One sheet per session; the duplicated subject's two sessions stay apart.
    assert len(sheets) == 5
    assert "m1" in sheets and "m2" in sheets and "m3" in sheets
    assert sum(1 for name in sheets if name.startswith("m4 ")) == 2


def test_session_types_still_split_when_every_subject_has_them(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "derived" / "review_tables").mkdir(parents=True)

    sessions = [
        ("s1", "m1", "m1_Conditioning"),
        ("s2", "m1", "m1_Recall"),
        ("s3", "m2", "m2_Conditioning"),
        ("s4", "m2", "m2_Recall"),
    ]
    ImportService().save_manifest(project_root, _manifest_for(sessions))
    candidates, decisions = _accepted(sessions)

    service = ExportService()
    service.set_project(project_root)
    out = service.export_boutframes_xlsx(candidates, decisions, include_merged_projects=False)

    assert out.success
    names = sorted(p.name for p in out.output_paths)
    assert names == ["boutframes_Conditioning.xlsx", "boutframes_Recall.xlsx"]
    for path in out.output_paths:
        sheets = [n for n in load_workbook(path).sheetnames if n != "_bout_counts"]
        assert sorted(sheets) == ["m1", "m2"]
