import numpy as np
import pandas as pd
from abel.storage.file_store import write_json

from abel.services.candidate_service import CandidateGenerationService, SegmentCandidateGenerationConfig


def test_min_distance_to_reviewed_matches_bruteforce() -> None:
    rng = np.random.default_rng(7)
    feats = rng.normal(size=(512, 80)).astype(float)
    reviewed = rng.normal(size=(173, 80)).astype(float)

    got = CandidateGenerationService._min_distance_to_reviewed(feats, reviewed)

    sq = np.sum((feats[:, None, :] - reviewed[None, :, :]) ** 2, axis=2)
    expected = np.sqrt(np.min(sq, axis=1))

    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_segment_overlap_suppression_keeps_higher_probability() -> None:
    ranked = pd.DataFrame(
        [
            {
                "segment_id": "a",
                "session_id": "s1",
                "video_id": "v1",
                "start_frame": 0,
                "end_frame": 20,
                "prediction_prob": 0.95,
                "uncertainty_score": 0.6,
                "rank_score": 1.2,
            },
            {
                "segment_id": "b",
                "session_id": "s1",
                "video_id": "v1",
                "start_frame": 10,
                "end_frame": 25,
                "prediction_prob": 0.70,
                "uncertainty_score": 0.8,
                "rank_score": 1.5,
            },
            {
                "segment_id": "c",
                "session_id": "s1",
                "video_id": "v1",
                "start_frame": 25,
                "end_frame": 40,
                "prediction_prob": 0.80,
                "uncertainty_score": 0.5,
                "rank_score": 1.0,
            },
            {
                "segment_id": "d",
                "session_id": "s2",
                "video_id": "v2",
                "start_frame": 10,
                "end_frame": 30,
                "prediction_prob": 0.60,
                "uncertainty_score": 0.4,
                "rank_score": 0.8,
            },
        ]
    )

    out = CandidateGenerationService._suppress_overlapping_segments(ranked)

    kept = set(out["segment_id"].astype(str).tolist())
    assert "a" in kept
    assert "b" not in kept
    assert "c" in kept
    assert "d" in kept


def test_low_probability_mode_prioritizes_absent_windows() -> None:
    df = pd.DataFrame(
        [
            {"segment_id": "high_prob", "prediction_prob": 0.92, "uncertainty_score": 0.10},
            {"segment_id": "mid_prob", "prediction_prob": 0.40, "uncertainty_score": 0.20},
            {"segment_id": "low_prob", "prediction_prob": 0.06, "uncertainty_score": 0.05},
        ]
    )
    feedback = pd.DataFrame(columns=["segment_id", "review_label"])

    ranked = CandidateGenerationService._rank_segments(
        df,
        feedback=feedback,
        mode="low_probability",
        hard_negative_ratio=0.0,
        low_prob_max_prob=0.25,
    )

    assert ranked.iloc[0]["segment_id"] == "low_prob"


def test_random_absent_mode_samples_only_low_probability_when_available() -> None:
    df = pd.DataFrame(
        [
            {"segment_id": "s1", "prediction_prob": 0.10, "uncertainty_score": 0.2},
            {"segment_id": "s2", "prediction_prob": 0.20, "uncertainty_score": 0.3},
            {"segment_id": "s3", "prediction_prob": 0.85, "uncertainty_score": 0.1},
        ]
    )
    feedback = pd.DataFrame(columns=["segment_id", "review_label"])

    ranked = CandidateGenerationService._rank_segments(
        df,
        feedback=feedback,
        mode="random_absent",
        hard_negative_ratio=0.0,
        random_absent_max_prob=0.30,
        random_seed=7,
    )

    # High-probability rows should be ineligible and sorted after low-probability rows.
    top_two = set(ranked.head(2)["segment_id"].astype(str).tolist())
    assert top_two == {"s1", "s2"}


def test_fast_random_absent_generation_excludes_accepted_ranges(tmp_path) -> None:
    project_root = tmp_path
    (project_root / "derived" / "pose_features").mkdir(parents=True, exist_ok=True)
    (project_root / "derived" / "review_tables").mkdir(parents=True, exist_ok=True)
    (project_root / "derived" / "review_labels").mkdir(parents=True, exist_ok=True)

    write_json(
        project_root / "derived" / "pose_features" / "summaries.json",
        {
            "summaries": [
                {
                    "session_id": "session_abc12345",
                    "n_frames": 300,
                    "n_windows": 0,
                    "body_parts": [],
                    "fps": 30.0,
                    "feature_path": "",
                    "created_at": "2026-03-15T00:00:00",
                    "warnings": [],
                }
            ]
        },
    )
    write_json(
        project_root / "derived" / "review_tables" / "import_manifest.json",
        {
            "videos": [{"asset_id": "v1", "frame_count": 300}],
            "linked_sessions": [
                {
                    "session_id": "session_abc12345",
                    "video_asset_id": "v1",
                    "pose_asset_id": "p1",
                    "subject_id": "subj_1",
                }
            ],
        },
    )

    labels = pd.DataFrame(
        [
            {
                "segment_id": "seg_subj_1_session_abc12345_100_159",
                "review_label": "target_behavior",
                "reviewer_id": "r1",
                "confidence": 1.0,
                "notes": "",
                "timestamp": "2026-03-15T00:00:00",
            }
        ]
    )
    labels.to_parquet(project_root / "derived" / "review_labels" / "reviewer_labels.parquet", index=False)

    svc = CandidateGenerationService()
    svc.set_project(project_root)
    cfg = SegmentCandidateGenerationConfig(
        top_k=20,
        mode="random_absent",
        model_version="behavior_model_test",
        feature_version="representation_v1",
        sample_window_frames=60,
        random_seed=7,
    )
    result = svc.generate_random_absent_candidates(cfg)

    assert result.success
    assert result.n_segments_selected > 0
    for cand in result.candidates:
        assert cand.session_id == "session_abc12345"
        assert max(0, min(cand.end_frame, 159) - max(cand.start_frame, 100) + 1) == 0


def test_fast_random_absent_generation_balances_across_subjects(tmp_path) -> None:
    project_root = tmp_path
    (project_root / "derived" / "pose_features").mkdir(parents=True, exist_ok=True)
    (project_root / "derived" / "review_tables").mkdir(parents=True, exist_ok=True)

    write_json(
        project_root / "derived" / "pose_features" / "summaries.json",
        {
            "summaries": [
                {
                    "session_id": "s_a1",
                    "n_frames": 300,
                    "n_windows": 0,
                    "body_parts": [],
                    "fps": 30.0,
                    "feature_path": "",
                    "created_at": "2026-03-15T00:00:00",
                    "warnings": [],
                },
                {
                    "session_id": "s_a2",
                    "n_frames": 300,
                    "n_windows": 0,
                    "body_parts": [],
                    "fps": 30.0,
                    "feature_path": "",
                    "created_at": "2026-03-15T00:00:00",
                    "warnings": [],
                },
                {
                    "session_id": "s_a3",
                    "n_frames": 300,
                    "n_windows": 0,
                    "body_parts": [],
                    "fps": 30.0,
                    "feature_path": "",
                    "created_at": "2026-03-15T00:00:00",
                    "warnings": [],
                },
                {
                    "session_id": "s_b1",
                    "n_frames": 300,
                    "n_windows": 0,
                    "body_parts": [],
                    "fps": 30.0,
                    "feature_path": "",
                    "created_at": "2026-03-15T00:00:00",
                    "warnings": [],
                },
            ]
        },
    )
    write_json(
        project_root / "derived" / "review_tables" / "import_manifest.json",
        {
            "videos": [
                {"asset_id": "v_a1", "frame_count": 300},
                {"asset_id": "v_a2", "frame_count": 300},
                {"asset_id": "v_a3", "frame_count": 300},
                {"asset_id": "v_b1", "frame_count": 300},
            ],
            "linked_sessions": [
                {"session_id": "s_a1", "video_asset_id": "v_a1", "pose_asset_id": "p_a1", "subject_id": "subject_a"},
                {"session_id": "s_a2", "video_asset_id": "v_a2", "pose_asset_id": "p_a2", "subject_id": "subject_a"},
                {"session_id": "s_a3", "video_asset_id": "v_a3", "pose_asset_id": "p_a3", "subject_id": "subject_a"},
                {"session_id": "s_b1", "video_asset_id": "v_b1", "pose_asset_id": "p_b1", "subject_id": "subject_b"},
            ],
        },
    )

    svc = CandidateGenerationService()
    svc.set_project(project_root)
    cfg = SegmentCandidateGenerationConfig(
        top_k=8,
        mode="random_absent",
        model_version="behavior_model_test",
        feature_version="representation_v1",
        sample_window_frames=50,
        random_seed=11,
    )

    result = svc.generate_random_absent_candidates(cfg)
    assert result.success
    assert result.n_segments_selected == 8

    subject_counts: dict[str, int] = {}
    for cand in result.candidates:
        subject_counts[cand.animal_id] = int(subject_counts.get(cand.animal_id, 0) + 1)

    assert set(subject_counts.keys()) == {"subject_a", "subject_b"}
    assert subject_counts["subject_a"] >= 1
    assert subject_counts["subject_b"] >= 1


def test_generate_segment_candidates_top_k_zero_returns_all(tmp_path) -> None:
    project_root = tmp_path
    (project_root / "derived" / "representations").mkdir(parents=True, exist_ok=True)
    (project_root / "derived" / "models" / "m1").mkdir(parents=True, exist_ok=True)

    seg = pd.DataFrame(
        [
            {
                "segment_id": "s1",
                "start_frame": 0,
                "end_frame": 59,
                "video_id": "v1",
                "animal_id": "a1",
                "session_id": "session_1",
                "prediction_prob": 0.2,
                "uncertainty_score": 0.9,
            },
            {
                "segment_id": "s2",
                "start_frame": 60,
                "end_frame": 119,
                "video_id": "v1",
                "animal_id": "a1",
                "session_id": "session_1",
                "prediction_prob": 0.4,
                "uncertainty_score": 0.7,
            },
            {
                "segment_id": "s3",
                "start_frame": 120,
                "end_frame": 179,
                "video_id": "v1",
                "animal_id": "a1",
                "session_id": "session_1",
                "prediction_prob": 0.6,
                "uncertainty_score": 0.5,
            },
        ]
    )
    seg.to_parquet(project_root / "derived" / "representations" / "segment_features.parquet", index=False)

    pred = seg[["segment_id", "prediction_prob"]].copy()
    unc = seg[["segment_id", "uncertainty_score"]].copy()
    pred.to_parquet(project_root / "derived" / "models" / "m1" / "segment_predictions.parquet", index=False)
    unc.to_parquet(project_root / "derived" / "models" / "m1" / "segment_uncertainty.parquet", index=False)

    svc = CandidateGenerationService()
    svc.set_project(project_root)
    cfg = SegmentCandidateGenerationConfig(
        top_k=0,
        mode="uncertainty",
        model_version="m1",
        feature_version="representation_v1",
    )
    out = svc.generate_segment_candidates(cfg)

    assert out.success
    assert out.n_segments_selected == 3
