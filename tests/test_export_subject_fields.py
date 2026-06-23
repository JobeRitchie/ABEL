import csv
import json
from pathlib import Path

import pandas as pd
import yaml

from abel.models.schemas import (
    CandidateWindow,
    ImportManifest,
    LinkedSession,
    ReviewDecision,
    ReviewDecisionType,
    VideoAsset,
)
from abel.services.behavior_service import BehaviorService
from abel.services.export_service import ExportService
from abel.services.import_service import ImportService


def test_export_csv_includes_subject_and_adjusted_frames(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "derived" / "review_tables").mkdir(parents=True)

    manifest = ImportManifest(
        videos=[VideoAsset(asset_id="v1", source_path="DG01BehavioralCamera0.avi", subject_id="DG01")],
        linked_sessions=[
            LinkedSession(
                session_id="session_alpha",
                video_asset_id="v1",
                pose_asset_id="p1",
                subject_id="DG01",
            )
        ],
    )
    ImportService().save_manifest(project_root, manifest)

    candidates = [
        CandidateWindow(
            window_id="cand_1",
            session_id="session_alpha",
            start_frame=100,
            end_frame=160,
            behavior_id="groom",
            total_score=0.9,
        )
    ]
    decisions = [
        ReviewDecision(
            decision_id="d1",
            clip_id="cand_1",
            reviewer="reviewer",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.ACCEPT,
            adjusted_start_frame=110,
            adjusted_end_frame=155,
        )
    ]

    service = ExportService()
    service.set_project(project_root)
    out = service.export_review_csv(candidates, decisions, filename="review_export.csv")

    assert out.success
    assert out.output_path is not None
    with out.output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["subject_id"] == "DG01"
    assert rows[0]["adjusted_start_frame"] == "110"
    assert rows[0]["adjusted_end_frame"] == "155"


def test_build_behaviogram_groups_accepted_by_subject(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "derived" / "review_tables").mkdir(parents=True)

    manifest = ImportManifest(
        videos=[VideoAsset(asset_id="v1", source_path="DG01BehavioralCamera0.avi", subject_id="DG01")],
        linked_sessions=[
            LinkedSession(
                session_id="session_alpha",
                video_asset_id="v1",
                pose_asset_id="p1",
                subject_id="DG01",
            )
        ],
    )
    ImportService().save_manifest(project_root, manifest)

    candidates = [
        CandidateWindow(window_id="a", session_id="session_alpha", start_frame=10, end_frame=40, behavior_id="groom"),
        CandidateWindow(window_id="b", session_id="session_alpha", start_frame=50, end_frame=70, behavior_id="rear"),
    ]
    decisions = [
        ReviewDecision(
            decision_id="d1",
            clip_id="a",
            reviewer="r",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.ACCEPT,
            adjusted_start_frame=12,
            adjusted_end_frame=35,
        ),
        ReviewDecision(
            decision_id="d2",
            clip_id="b",
            reviewer="r",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.REJECT,
        ),
    ]

    service = ExportService()
    service.set_project(project_root)
    data = service.build_behaviogram(candidates, decisions)

    assert "DG01" in data
    assert data["DG01"]["max_end_frame"] == 35
    assert data["DG01"]["behaviors"]["groom"] == [(12, 35)]
    assert "rear" not in data["DG01"]["behaviors"]


def test_list_available_subjects_from_manifest(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "derived" / "review_tables").mkdir(parents=True)

    manifest = ImportManifest(
        videos=[
            VideoAsset(asset_id="v1", source_path="S1_cam.avi", subject_id="S1"),
            VideoAsset(asset_id="v2", source_path="S2_cam.avi", subject_id="S2"),
        ],
        linked_sessions=[
            LinkedSession(session_id="session_a", video_asset_id="v1", pose_asset_id="p1", subject_id="S1"),
            LinkedSession(session_id="session_b", video_asset_id="v2", pose_asset_id="p2", subject_id="S2"),
        ],
    )
    ImportService().save_manifest(project_root, manifest)

    service = ExportService()
    service.set_project(project_root)

    assert service.list_available_subjects() == ["S1", "S2"]


def test_temporal_safe_token_resolves_to_behavior_name(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "config").mkdir(parents=True)
    (project_root / "derived" / "review_tables").mkdir(parents=True)

    # Behavior ID intentionally contains spaces to force safe-token mismatch.
    behavior_id = "grooming behavior"
    behavior_name = "Grooming"
    behavior_defs = {
        "behaviors": [
            {
                "behavior_id": behavior_id,
                "name": behavior_name,
                "short_name": "groom",
                "description": "",
                "operational_definition": "",
                "inclusion_criteria": "",
                "exclusion_criteria": "",
                "min_duration_sec": 0.0,
                "review_priority": 1,
                "color": "#4A90E2",
                "keyboard_shortcut": "g",
                "is_active": True,
                "notes": "",
                "version_history": [],
            }
        ]
    }
    (project_root / "config" / "behavior_definitions.yaml").write_text(
        yaml.safe_dump(behavior_defs),
        encoding="utf-8",
    )

    safe_token = "grooming_behavior"
    tr_root = project_root / "derived" / "temporal_refinement" / safe_token
    post_dir = tr_root / "postprocess_run"
    bouts_dir = post_dir / "bout_outputs"
    bouts_dir.mkdir(parents=True)

    bout_path = bouts_dir / "session_a_bouts.parquet"
    pd.DataFrame(
        [{"start_frame": 10, "end_frame": 20, "session_id": "session_a"}]
    ).to_parquet(bout_path, index=False)

    (tr_root / "latest.json").write_text(
        json.dumps({"postprocess_dir": str(post_dir)}),
        encoding="utf-8",
    )
    (post_dir / "postprocess_manifest.json").write_text(
        json.dumps({"bout_paths": {"session_a": str(bout_path)}}),
        encoding="utf-8",
    )

    service = ExportService()
    service.set_project(project_root)
    behaviors = BehaviorService()
    behaviors.set_project(project_root)
    service.set_behavior_service(behaviors)

    intervals = service._temporal_confirmed_intervals_by_session()
    assert "session_a" in intervals
    assert behavior_name in intervals["session_a"]
    assert intervals["session_a"][behavior_name] == [(10, 20)]


def test_canonicalize_behavior_intervals_merges_aliases(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "config").mkdir(parents=True)

    behavior_defs = {
        "behaviors": [
            {
                "behavior_id": "GROOM ID",
                "name": "Grooming",
                "short_name": "groom",
                "description": "",
                "operational_definition": "",
                "inclusion_criteria": "",
                "exclusion_criteria": "",
                "min_duration_sec": 0.0,
                "review_priority": 1,
                "color": "#4A90E2",
                "keyboard_shortcut": "g",
                "is_active": True,
                "notes": "",
                "version_history": [],
            }
        ]
    }
    (project_root / "config" / "behavior_definitions.yaml").write_text(
        yaml.safe_dump(behavior_defs),
        encoding="utf-8",
    )

    service = ExportService()
    service.set_project(project_root)
    behaviors = BehaviorService()
    behaviors.set_project(project_root)
    service.set_behavior_service(behaviors)

    merged = service._canonicalize_behavior_intervals(
        {
            "session_1": {
                "GROOM ID": [(1, 3)],
                "Grooming": [(5, 8)],
                "groom_id": [(10, 12)],
            }
        }
    )

    assert list(merged["session_1"].keys()) == ["Grooming"]
    assert merged["session_1"]["Grooming"] == [(1, 3), (5, 8), (10, 12)]


def test_temporal_generic_concept_id_falls_back_to_dir_token(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "config").mkdir(parents=True)

    behavior_defs = {
        "behaviors": [
            {
                "behavior_id": "Rear Behavior",
                "name": "Rearing",
                "short_name": "rear",
                "description": "",
                "operational_definition": "",
                "inclusion_criteria": "",
                "exclusion_criteria": "",
                "min_duration_sec": 0.0,
                "review_priority": 1,
                "color": "#7ED321",
                "keyboard_shortcut": "r",
                "is_active": True,
                "notes": "",
                "version_history": [],
            }
        ]
    }
    (project_root / "config" / "behavior_definitions.yaml").write_text(
        yaml.safe_dump(behavior_defs),
        encoding="utf-8",
    )

    safe_token = "Rear_Behavior"
    tr_root = project_root / "derived" / "temporal_refinement" / safe_token
    post_dir = tr_root / "postprocess_run"
    bouts_dir = post_dir / "bout_outputs"
    bouts_dir.mkdir(parents=True)

    bout_path = bouts_dir / "session_x_bouts.parquet"
    pd.DataFrame(
        [{"start_frame": 30, "end_frame": 45, "session_id": "session_x"}]
    ).to_parquet(bout_path, index=False)

    # Simulate older/generic manifest that doesn't carry behavior-specific concept id.
    (tr_root / "latest.json").write_text(
        json.dumps({"concept_id": "target_behavior", "postprocess_dir": str(post_dir)}),
        encoding="utf-8",
    )
    (post_dir / "postprocess_manifest.json").write_text(
        json.dumps({"bout_paths": {"session_x": str(bout_path)}}),
        encoding="utf-8",
    )

    service = ExportService()
    service.set_project(project_root)
    behaviors = BehaviorService()
    behaviors.set_project(project_root)
    service.set_behavior_service(behaviors)

    intervals = service._temporal_confirmed_intervals_by_session()
    assert intervals["session_x"]["Rearing"] == [(30, 45)]


def test_behavior_filter_matches_id_alias_tokens(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "config").mkdir(parents=True)

    behavior_defs = {
        "behaviors": [
            {
                "behavior_id": "rear id",
                "name": "Rearing",
                "short_name": "rear",
                "description": "",
                "operational_definition": "",
                "inclusion_criteria": "",
                "exclusion_criteria": "",
                "min_duration_sec": 0.0,
                "review_priority": 1,
                "color": "#7ED321",
                "keyboard_shortcut": "r",
                "is_active": True,
                "notes": "",
                "version_history": [],
            }
        ]
    }
    (project_root / "config" / "behavior_definitions.yaml").write_text(
        yaml.safe_dump(behavior_defs),
        encoding="utf-8",
    )

    service = ExportService()
    service.set_project(project_root)
    behaviors = BehaviorService()
    behaviors.set_project(project_root)
    service.set_behavior_service(behaviors)

    intervals = {
        "session_1": {
            "rear_id": [(1, 4)],
            "groom": [(10, 20)],
        }
    }
    filtered = {
        sid: {
            b: ivs
            for b, ivs in by_b.items()
            if service._normalize_behavior_token(b)
            in (
                set([service._normalize_behavior_token("Rearing")])
                | set(service._behavior_alias_tokens().get(service._normalize_behavior_token("Rearing"), set()))
            )
        }
        for sid, by_b in intervals.items()
    }

    assert "rear_id" in filtered["session_1"]
    assert "groom" not in filtered["session_1"]
