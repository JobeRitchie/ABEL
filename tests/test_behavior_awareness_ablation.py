"""Tests for the behavior-awareness ablation service.

These validate the core comparison logic without requiring a full project
on disk — we construct minimal DataFrames and mock file state as needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services.behavior_awareness_ablation_service import (
    AblationResult,
    BehaviorAwarenessAblationService,
)


# ------------------------------------------------------------------
# Mean Reciprocal Rank helper
# ------------------------------------------------------------------

def test_mrr_perfect_ranking() -> None:
    """All positives ranked first should give high MRR."""
    df = pd.DataFrame({
        "is_accepted": [True, True, False, False],
        "score": [0.9, 0.8, 0.3, 0.1],
    })
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    mrr = BehaviorAwarenessAblationService._mean_reciprocal_rank(df, df["is_accepted"].to_numpy())
    # Ranks 1 and 2 → MRR = (1/1 + 1/2) / 2 = 0.75
    assert abs(mrr - 0.75) < 1e-9


def test_mrr_worst_ranking() -> None:
    """Positives ranked last should give low MRR."""
    df = pd.DataFrame({
        "is_accepted": [False, False, True, True],
        "score": [0.9, 0.8, 0.3, 0.1],
    })
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    mrr = BehaviorAwarenessAblationService._mean_reciprocal_rank(df, df["is_accepted"].to_numpy())
    # ranks 3 and 4 → MRR = (1/3 + 1/4) / 2 ≈ 0.2917
    expected = (1.0 / 3 + 1.0 / 4) / 2
    assert abs(mrr - expected) < 1e-4


def test_mrr_no_positives() -> None:
    df = pd.DataFrame({"is_accepted": [False, False]})
    mrr = BehaviorAwarenessAblationService._mean_reciprocal_rank(df, df["is_accepted"].to_numpy())
    assert mrr == 0.0


# ------------------------------------------------------------------
# Verdict synthesis
# ------------------------------------------------------------------

def test_verdict_aware_wins() -> None:
    r = AblationResult()
    r.candidate_test_ran = True
    r.candidate_mrr_aware = 0.8
    r.candidate_mrr_unaware = 0.5
    r.candidate_detail = "test"
    r.temporal_test_ran = True
    r.temporal_f1_aware = 0.7
    r.temporal_f1_unaware = 0.6
    r.temporal_detail = "test"
    r.model_test_ran = True
    r.model_f1_aware = 0.85
    r.model_f1_unaware = 0.80
    r.model_detail = "test"

    verdict, summary = BehaviorAwarenessAblationService._synthesize_verdict(r)
    assert verdict == "aware_better"
    assert "wins 3/3" in summary


def test_verdict_unaware_wins() -> None:
    r = AblationResult()
    r.candidate_test_ran = True
    r.candidate_mrr_aware = 0.3
    r.candidate_mrr_unaware = 0.7
    r.candidate_detail = "test"
    r.temporal_test_ran = True
    r.temporal_f1_aware = 0.4
    r.temporal_f1_unaware = 0.6
    r.temporal_detail = "test"
    r.model_test_ran = False

    verdict, summary = BehaviorAwarenessAblationService._synthesize_verdict(r)
    assert verdict == "unaware_better"
    assert "Consider disabling" in summary


def test_verdict_inconclusive_no_tests() -> None:
    r = AblationResult()
    verdict, summary = BehaviorAwarenessAblationService._synthesize_verdict(r)
    assert verdict == "inconclusive"
    assert "No tests could be run" in summary


def test_verdict_inconclusive_tied() -> None:
    r = AblationResult()
    r.candidate_test_ran = True
    r.candidate_mrr_aware = 0.8
    r.candidate_mrr_unaware = 0.5
    r.candidate_detail = "test"
    r.temporal_test_ran = True
    r.temporal_f1_aware = 0.5
    r.temporal_f1_unaware = 0.7
    r.temporal_detail = "test"
    r.model_test_ran = False

    verdict, summary = BehaviorAwarenessAblationService._synthesize_verdict(r)
    assert verdict == "inconclusive"
    assert "More labeled data" in summary


# ------------------------------------------------------------------
# Peer feature column builder — empty when no peers
# ------------------------------------------------------------------

def test_build_peer_features_no_models_dir(tmp_path) -> None:
    svc = BehaviorAwarenessAblationService()
    df = pd.DataFrame({"segment_id": ["s1", "s2"], "val": [1.0, 2.0]})
    result = svc._build_peer_feature_columns(tmp_path, "dig", df)
    assert result.empty


def test_build_peer_features_with_peer(tmp_path) -> None:
    """When a peer model directory has predictions, features should be built."""
    svc = BehaviorAwarenessAblationService()
    models_root = tmp_path / "derived" / "models"

    # Target model — should be excluded
    target_dir = models_root / "behavior_model_dig"
    target_dir.mkdir(parents=True)
    target_pred = pd.DataFrame({"segment_id": ["s1", "s2"], "prediction_prob": [0.9, 0.1]})
    target_pred.to_parquet(target_dir / "segment_predictions.parquet", index=False)

    # Peer model
    peer_dir = models_root / "behavior_model_groom"
    peer_dir.mkdir(parents=True)
    peer_pred = pd.DataFrame({"segment_id": ["s1", "s2"], "prediction_prob": [0.3, 0.7]})
    peer_pred.to_parquet(peer_dir / "segment_predictions.parquet", index=False)

    df = pd.DataFrame({"segment_id": ["s1", "s2"], "val": [1.0, 2.0]})
    result = svc._build_peer_feature_columns(tmp_path, "dig", df)

    assert not result.empty
    assert "peer_prob_max" in result.columns
    assert "peer_prob_mean" in result.columns
    assert len(result) == 2
    # s1 should have groom prob 0.3, s2 should have 0.7
    np.testing.assert_allclose(result["peer_prob_max"].values, [0.3, 0.7], atol=1e-6)


# ------------------------------------------------------------------
# Full ablation on minimal project structure
# ------------------------------------------------------------------

def test_ablation_minimal_project_no_data(tmp_path) -> None:
    """Running ablation on a project with no data should not crash."""
    svc = BehaviorAwarenessAblationService()
    result = svc.run_ablation(tmp_path, "dig")
    assert result.verdict == "inconclusive"
    assert len(result.warnings) > 0
