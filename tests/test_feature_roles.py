"""Feature-role clustering: do extracted features play distinct roles per behavior?

Each test pins a claim the manuscript figure rests on: behaviors group by the modality
they rely on (one bar per feature type), the over-pose improvement is the ablation ΔF1
of context/video (and exactly 0 for pose/kinematics, which are already in the baseline),
and the whole thing joins ablation to importance on the assay-scoped behavior key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.validation.analyses import feature_roles as fr


def _shares() -> pd.DataFrame:
    # 6 behaviors, each clearly dominated by one modality.
    rows = []
    profile = {
        "Climb": {"Context (ROI / target)": 60, "Kinematics": 20, "Pose geometry": 10,
                  "Video (flow / appearance)": 10},
        "Sniff": {"Context (ROI / target)": 55, "Kinematics": 25, "Pose geometry": 10,
                  "Video (flow / appearance)": 10},
        "Approach": {"Context (ROI / target)": 10, "Kinematics": 25, "Pose geometry": 5,
                     "Video (flow / appearance)": 60},
        "Explore": {"Context (ROI / target)": 5, "Kinematics": 78, "Pose geometry": 14,
                    "Video (flow / appearance)": 3},
        "Groom": {"Context (ROI / target)": 17, "Kinematics": 16, "Pose geometry": 63,
                  "Video (flow / appearance)": 4},
        "Rear": {"Context (ROI / target)": 20, "Kinematics": 20, "Pose geometry": 55,
                 "Video (flow / appearance)": 5},
    }
    for beh, shares in profile.items():
        for mod, pct in shares.items():
            rows.append({"behavior": beh, "modality_label": mod, "percent": pct})
    return pd.DataFrame(rows)


def _ablation() -> pd.DataFrame:
    # Context-dominant behaviors gain from context; the video-dominant one from video;
    # pose/kinematics behaviors gain nothing meaningful (they're the baseline).
    gains = {
        "Climb": (0.20, 0.05), "Sniff": (0.15, 0.04), "Approach": (0.03, 0.12),
        "Explore": (0.00, 0.00), "Groom": (0.01, 0.00), "Rear": (0.01, 0.00),
    }
    rows = []
    for beh, (ctx, vid) in gains.items():
        rows.append({"project": "P", "behavior": beh, "clip_budget": "all",
                     "label": "Baseline (pose only)", "gain_over_baseline": 0.0})
        rows.append({"project": "P", "behavior": beh, "clip_budget": "all",
                     "label": "+ Environment / ROI context", "gain_over_baseline": ctx})
        rows.append({"project": "P", "behavior": beh, "clip_budget": "all",
                     "label": "+ Video features", "gain_over_baseline": vid})
    return pd.DataFrame(rows)


def test_modality_reliance_matrix_rows_sum_to_one():
    m = fr.modality_reliance_matrix(_shares())
    assert m.shape == (6, 4)
    np.testing.assert_allclose(m.sum(axis=1).to_numpy(), np.ones(6), atol=1e-9)


def test_modality_groups_one_group_per_dominant_modality():
    m = fr.modality_reliance_matrix(_shares())
    labels = fr.modality_groups(m)
    # 3 modalities are dominant across the 6 behaviors: context (2), video (1), etc.
    dom = m.idxmax(axis=1)
    assert set(dom) == {"Context (ROI / target)", "Video (flow / appearance)",
                        "Kinematics", "Pose geometry"}
    # One integer label per distinct dominant modality, context first (order priority).
    assert len(set(labels)) == 4
    assert labels[list(m.index).index("Climb")] == labels[list(m.index).index("Sniff")]


def test_improvement_is_gain_for_added_features_and_zero_for_baseline():
    m = fr.modality_reliance_matrix(_shares())
    gain = fr.ablation_gain_by_behavior(_ablation(), by="behavior")
    bars = fr.dominant_modality_improvement_bars(m, fr.modality_groups(m), gain)
    by_mod = bars.set_index("dominant_modality")
    # Context cluster (Climb+Sniff): mean of context gains 0.20, 0.15.
    assert by_mod.loc["Context (ROI / target)", "mean_improvement_over_pose"] == \
        pytest.approx((0.20 + 0.15) / 2)
    # Video cluster (Approach): its video gain 0.12.
    assert by_mod.loc["Video (flow / appearance)", "mean_improvement_over_pose"] == \
        pytest.approx(0.12)
    # Pose & kinematics clusters are baseline — exactly 0 over pose-only.
    assert by_mod.loc["Pose geometry", "mean_improvement_over_pose"] == 0.0
    assert by_mod.loc["Kinematics", "mean_improvement_over_pose"] == 0.0
    # Bars are ranked by descending improvement.
    assert list(bars["rank"]) == sorted(bars["rank"])
    assert bars.iloc[0]["mean_improvement_over_pose"] >= bars.iloc[-1]["mean_improvement_over_pose"]


def test_ablation_gain_key_is_assay_scoped_when_requested():
    g = fr.ablation_gain_by_behavior(_ablation(), by="assay")
    assert "P · Climb" in g.index          # never a bare "Climb" that could collide
    assert "Climb" not in g.index


def test_run_feature_roles_writes_bars_membership_and_dendrogram(tmp_path):
    written = fr.run_feature_roles(_shares(), _ablation(), tmp_path, k=4, scope="behavior")
    names = {p.name for p in written}
    assert {"feature_role_cluster_bars.csv", "feature_role_clusters.csv"} <= names
    bars = pd.read_csv(tmp_path / "feature_role_cluster_bars.csv")
    assert "kruskal_p_across_clusters" in bars.columns
    assert set(bars["dominant_modality"]) >= {"Context (ROI / target)", "Pose geometry"}
