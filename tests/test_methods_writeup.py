"""Tests for the Methods write-up helper.

The helper's whole value is that it reports what the project actually says, so
the tests guard the two ways it could betray that: inventing a number the project
does not contain, and dropping the warning that the output is a draft.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import abel.ui.methods_writeup as mw

ALL_KEYS = [s.key for s in mw.SECTIONS]


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A minimal but realistic project tree: config, manifest, one model."""
    root = tmp_path / "TestProject"
    (root / "config").mkdir(parents=True)
    (root / "derived" / "review_tables").mkdir(parents=True)
    (root / "derived" / "models" / "behavior_model_Rear").mkdir(parents=True)

    (root / "project.yaml").write_text(
        yaml.safe_dump({
            "project_name": "TestProject",
            "assay_name": "OFT",
            "species": "mouse",
            "single_animal": True,
            "num_animals": 1,
            "default_fps": 30.0,
            "behavior_model": {
                "classifier_type": "xgboost",
                "calibration_method": "sigmoid",
                "segment_window_frames": 30,
                "segment_stride_frames": 10,
                "evaluation_split_strategy": "group_shuffle_session",
                "query_strategy": "uncertainty",
                "active_learning_query_size": 250,
                "use_video_features": True,
                "use_r3d_features": True,
                "allow_co_occurring_behaviors": False,
            },
        }),
        encoding="utf-8",
    )
    (root / "config" / "behavior_definitions.yaml").write_text(
        yaml.safe_dump({
            "behaviors": [
                {
                    "behavior_id": "no_behavior", "name": "No Behavior",
                    "is_active": True, "review_priority": 999,
                },
                {
                    "behavior_id": "beh-rear", "name": "Rear", "is_active": True,
                    "review_priority": 1, "operational_definition": "Both forepaws off the floor.",
                    "min_duration_sec": 0.5,
                },
                {
                    "behavior_id": "beh-old", "name": "Retired", "is_active": False,
                    "review_priority": 2,
                },
            ]
        }),
        encoding="utf-8",
    )
    (root / "derived" / "review_tables" / "import_manifest.json").write_text(
        json.dumps({
            "smoothing_settings": {
                "likelihood_threshold": 0.4, "interpolate_dropouts": True,
                "interpolate_max_gap": 10, "smoothing_window": 5,
            },
            "videos": [
                {"asset_id": "v1", "source_path": "a.mp4", "fps": 30.0,
                 "width": 640, "height": 480, "duration_sec": 600.0},
                {"asset_id": "v2", "source_path": "b.mp4", "fps": 30.0,
                 "width": 640, "height": 480, "duration_sec": 600.0},
            ],
            "poses": [{"asset_id": "p1", "source_path": "a.csv", "format": "csv",
                       "body_parts": ["nose", "tail_base"]}],
            "linked_sessions": [
                {"session_id": "s1", "video_asset_id": "v1", "pose_asset_id": "p1",
                 "subject_id": "M1"},
                {"session_id": "s2", "video_asset_id": "v2", "pose_asset_id": "p1",
                 "subject_id": "M2"},
            ],
        }),
        encoding="utf-8",
    )
    (root / "config" / "motif_settings.json").write_text(
        json.dumps({
            "hmm_n_states_mode": "manual", "hmm_n_states": 5, "hmm_n_iter": 500,
            "hmm_n_restarts": 8, "hmm_random_seed": 7, "hmm_occupancy_method": "posterior",
            "motif_method": "ngram", "ngram_min_n": 2, "ngram_max_n": 3,
            "n_permutations": 2000, "permutation_seed": 42,
            "transition_pval_correction": "fdr_bh",
        }),
        encoding="utf-8",
    )
    (root / "derived" / "models" / "behavior_model_Rear" / "model_card.yaml").write_text(
        yaml.safe_dump({
            "model_version": "behavior_model_Rear",
            "feature_columns": ["nose_velocity_mean", "nose_to_roi_1_dist", "flow_mag_mean"],
        }),
        encoding="utf-8",
    )
    return root


MODEL_ROWS = [
    {
        "behavior_id": "beh-rear", "behavior_name": "Rear",
        "model_version": "behavior_model_Rear", "frame_f1": 0.879,
        "frame_precision": 0.950, "frame_recall": 0.817, "pr_auc": 0.952,
        "n_train": 3893, "n_val": 1212, "calibration": "sigmoid",
        "last_trained": "2026-08-27T12:33:12", "n_positive_labels": 704,
        "metrics_basis": "target", "refined_f1": 0.746,
        "refined_settings": {
            "onset_threshold": 0.7, "min_bout_duration_frames": 15, "merge_gap_frames": 0,
        },
    },
    {
        "behavior_id": "no_behavior", "behavior_name": "No Behavior",
        "model_version": "behavior_model_No_Behavior", "frame_f1": 0.65,
        "n_positive_labels": 1519, "metrics_basis": "target",
    },
]


def test_sections_are_well_formed() -> None:
    keys = [s.key for s in mw.SECTIONS]
    assert len(keys) == len(set(keys)), "duplicate section keys"
    for spec in mw.SECTIONS:
        assert spec.label and spec.hint
        assert spec.group in mw.SECTION_GROUPS, f"{spec.key} has an unlisted group"
    # Every section except the citations list must have a renderer.
    for spec in mw.SECTIONS:
        if spec.key != "citations":
            assert spec.key in mw._SECTION_RENDERERS, f"{spec.key} has no renderer"
    assert mw.DEFAULT_KEYS, "no sections are on by default"


def test_section_refs_resolve_to_real_citations() -> None:
    from abel.ui.methods_content import REFERENCES

    known = {r.key for r in REFERENCES}
    cited = {k for keys in mw._SECTION_REFS.values() for k in keys}
    cited |= {"tran2018", "kay2017", "ke2017"}  # conditional citations
    assert cited <= known, f"unknown reference keys: {sorted(cited - known)}"


def test_gather_facts_reads_the_project(project: Path) -> None:
    facts = mw.gather_facts(project, model_rows=MODEL_ROWS)
    assert facts["project"]["project_name"] == "TestProject"
    assert facts["fps"] == 30.0
    assert facts["recordings"]["n_sessions"] == 2
    assert facts["recordings"]["n_subjects"] == 2
    assert facts["recordings"]["keypoints"] == ["nose", "tail_base"]
    assert facts["motif"]["hmm_n_states"] == 5
    assert facts["feature_modalities"]["n_features"] == 3


def test_no_behavior_and_inactive_behaviors_are_excluded(project: Path) -> None:
    """The negative class is not a behavior, and a retired one is not scored."""
    facts = mw.gather_facts(project, model_rows=MODEL_ROWS)
    names = [b["name"] for b in facts["behaviors"]]
    assert names == ["Rear"]

    text = mw.render_methods_text(facts, ["behaviors", "performance"], include_warning=False)
    assert "Rear" in text
    assert "Retired" not in text
    # "No Behavior" may be named as the negative class, but never counted or scored.
    assert "trained for 1 behavior" in text
    assert "behavior_model_No_Behavior" not in text


def test_metrics_come_from_the_model_rows(project: Path) -> None:
    facts = mw.gather_facts(project, model_rows=MODEL_ROWS)
    text = mw.render_methods_text(facts, ["performance"], include_warning=False)
    for value in ("0.950", "0.817", "0.879", "0.952", "704"):
        assert value in text, f"{value} missing from the performance table"


def test_hmm_sentence_reports_the_projects_settings(project: Path) -> None:
    facts = mw.gather_facts(project)
    text = mw.render_methods_text(facts, ["hmm"], include_warning=False)
    assert "a fixed 5 states" in text
    assert "8 random restarts" in text
    assert "500 iterations" in text
    assert "seed 7" in text
    assert "posterior" in text


def test_warning_is_present_by_default_and_removable(project: Path) -> None:
    facts = mw.gather_facts(project, model_rows=MODEL_ROWS)
    with_warning = mw.render_methods_text(facts, mw.DEFAULT_KEYS)
    assert mw.WARNING_TEXT in with_warning
    assert "DRAFT ONLY" in with_warning

    without = mw.render_methods_text(facts, mw.DEFAULT_KEYS, include_warning=False)
    assert "DRAFT ONLY" not in without
    # The footer restates it either way, so a pasted fragment still carries a flag.
    assert "Re-read the warning" in without


def test_unknowable_facts_become_placeholders(project: Path) -> None:
    """ABEL cannot know the tracker or the experimental design; it must not guess."""
    facts = mw.gather_facts(project, model_rows=MODEL_ROWS)
    text = mw.render_methods_text(facts, ALL_KEYS, include_warning=False)
    assert "[FILL IN:" in text
    assert "DeepLabCut 2.3.x or SLEAP 1.3.x" in text  # offered as an example, not asserted


def test_empty_project_degrades_without_inventing_numbers(tmp_path: Path) -> None:
    """A directory with no project files must not produce fabricated settings."""
    facts = mw.gather_facts(tmp_path)
    text = mw.render_methods_text(facts, ALL_KEYS, include_warning=False)
    assert "windows of 0 frames" not in text
    assert "target zone of interest" not in text  # no ROI was ever drawn
    assert "No trained behavior models were found" in text
    assert "[FILL IN:" in text


def test_every_section_renders_standalone(project: Path) -> None:
    """No section may raise or leak an error marker when rendered on its own."""
    facts = mw.gather_facts(project, model_rows=MODEL_ROWS)
    for key in ALL_KEYS:
        text = mw.render_methods_text(facts, [key], include_warning=False)
        assert "Could not build this section" not in text, f"{key} failed to render"


def test_render_is_stable_across_selection_order(project: Path) -> None:
    facts = mw.gather_facts(project, model_rows=MODEL_ROWS)
    forward = mw.render_methods_text(facts, ALL_KEYS, include_warning=False)
    backward = mw.render_methods_text(facts, list(reversed(ALL_KEYS)), include_warning=False)
    assert forward == backward
