"""Meta summary tables: the small, assay-scoped distillation for the manuscript figure.

The point of these tables is to collapse ~40 per-behavior CSVs into a handful a person
can plot. Each test pins one thing that would quietly corrupt the paper if it regressed:
behaviors must stay scoped to their assay (never pooled by name), the feature-value
table must count significant behaviors honestly, and the discrimination summary must be
taken over pairs that actually have headroom.
"""

from __future__ import annotations

import pandas as pd
import pytest

from abel.validation import meta_summary as ms


def _sources() -> dict[str, pd.DataFrame]:
    # Two assays that SHARE a behavior name ("Rear") — the collision the assay
    # scoping must survive.
    return {
        "publication_metrics": pd.DataFrame({
            "project_id": ["EPM", "OFT"], "project_name": ["EPM", "OFT"],
            "f1": [0.86, 0.94], "mcc": [0.73, 0.89], "balanced_accuracy": [0.84, 0.96],
            "roc_auc": [0.98, 0.99], "cohen_kappa": [0.72, 0.89],
        }),
        "accuracy_by_behavior": pd.DataFrame({
            "project_id": ["EPM", "EPM", "OFT"], "behavior_name": ["Rear", "Groom", "Rear"],
            "f1_mean": [0.80, 0.60, 0.90], "f1_ci": [0.02, 0.03, 0.01], "n": [5, 5, 5],
        }),
        "generalization": pd.DataFrame({
            "project": ["EPM", "EPM", "OFT"], "behavior": ["Rear", "Groom", "Rear"],
            "f1": [0.8, 0.6, 0.9], "cohen_kappa": [0.70, 0.50, 0.88],
        }),
        "calibration": pd.DataFrame({
            "project": ["EPM", "EPM", "OFT"], "behavior": ["Rear", "Groom", "Rear"],
            "ece": [0.01, 0.02, 0.015],
        }),
        "ablation": pd.DataFrame({
            "project": ["EPM", "EPM", "OFT", "EPM", "EPM", "OFT"],
            "behavior": ["Rear", "Groom", "Rear", "Rear", "Groom", "Rear"],
            "clip_budget": ["all", "all", "all", "all", "all", "all"],
            "config": ["baseline_none", "baseline_none", "baseline_none",
                       "all_features", "all_features", "all_features"],
            "label": ["Baseline (pose only)", "Baseline (pose only)", "Baseline (pose only)",
                      "All enhancements", "All enhancements", "All enhancements"],
            "gain_over_baseline": [0.0, 0.0, 0.0, 0.12, 0.03, 0.20],
            "significant": [False, False, False, True, False, True],
        }),
        "confusable_pairs": pd.DataFrame({
            "project": ["EPM", "EPM"], "pair": ["Rear vs Groom", "A vs B"],
            "pose_only_auc": [0.70, 0.999],      # second pair has no headroom
            "pose_context_auc": [0.95, 0.999], "pose_context_error_reduction": [0.80, 0.0],
            "pose_video_auc": [0.90, 0.999], "pose_video_error_reduction": [0.60, 0.0],
            "all_features_auc": [0.97, 0.999], "all_features_error_reduction": [0.90, 0.0],
        }),
    }


def test_per_behavior_keeps_same_name_in_two_assays_separate():
    out = ms.summary_per_behavior(_sources())
    # Three rows, and the two "Rear"s are distinct (EPM vs OFT), never merged.
    assert len(out) == 3
    rears = out[out["behavior"] == "Rear"].set_index("assay")
    assert set(rears.index) == {"EPM", "OFT"}
    assert rears.loc["EPM", "f1_holdout"] == 0.80
    assert rears.loc["OFT", "f1_holdout"] == 0.90
    # Joins landed on the right assay, not blended across the shared name.
    assert rears.loc["EPM", "generalization_kappa"] == 0.70
    assert rears.loc["OFT", "generalization_kappa"] == 0.88
    assert rears.loc["EPM", "all_enh_dF1"] == 0.12
    assert rears.loc["OFT", "all_enh_dF1"] == 0.20


def test_per_assay_reports_one_row_per_assay_with_derived_means():
    out = ms.summary_per_assay(_sources()).set_index("assay")
    assert set(out.index) == {"EPM", "OFT"}
    assert out.loc["EPM", "n_behaviors"] == 2      # Rear + Groom
    assert out.loc["OFT", "n_behaviors"] == 1
    assert out.loc["EPM", "mean_generalization_kappa"] == pytest.approx((0.70 + 0.50) / 2)
    assert out.loc["EPM", "mean_ece"] == pytest.approx((0.01 + 0.02) / 2)


def test_feature_value_counts_significant_behaviors():
    out = ms.summary_feature_value(_sources())
    row = out[out["enhancement"] == "All enhancements"].iloc[0]
    assert row["n_total"] == 3
    assert row["n_significant"] == 2               # EPM Rear + OFT Rear were significant
    # The pose-only baseline is never a feature-value row.
    assert "Baseline (pose only)" not in set(out["enhancement"])


def test_discrimination_only_uses_pairs_with_headroom():
    out = ms.summary_discrimination(_sources())
    # The near-perfect pose pair (AUC 0.999) is excluded; only the one hard pair counts.
    assert set(out["n_pairs"]) == {1}
    allf = out[out["feature_family"] == "All features"].iloc[0]
    assert allf["mean_auc_pose"] == pytest.approx(0.70)
    assert allf["mean_error_reduction"] == pytest.approx(0.90)
