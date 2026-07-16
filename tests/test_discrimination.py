"""Tests for the pairwise behavior-discrimination ablation.

Covers the pure logic (confusable-pair discovery, pair selection, matrix
assembly, significance) without training models; the end-to-end training path is
exercised by the validation run itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abel.validation.analyses import discrimination as disc
from abel.validation.datamodel import ProjectRef


def _project(tmp_path=None, **kw) -> ProjectRef:
    from pathlib import Path

    return ProjectRef(
        project_id="P", name="P", root=Path(tmp_path or "."),
        behavior_names={"a": "Freeze", "b": "Groom", "c": "Walk", "no_behavior": "No Behavior"},
        **kw,
    )


# ── Candidate ranking (training-free proximity pre-filter) ─────────────────


def _pool(n=30) -> pd.DataFrame:
    """Pool where behaviors a and b overlap in feature space and c sits far away."""
    rng = np.random.default_rng(0)
    rows = []
    for bid, centre in (("a", 0.0), ("b", 0.3), ("c", 12.0)):
        for _ in range(n):
            rows.append({
                "label": bid,
                "feat_x": centre + rng.normal(0, 0.1),
                "feat_y": centre + rng.normal(0, 0.1),
                "session_id": "s1",
            })
    return pd.DataFrame(rows)


def test_proximity_ranks_the_overlapping_pair_first():
    ranked = disc.rank_pairs_by_proximity(_pool(), ["a", "b", "c"])
    top = ranked.iloc[0]
    assert {top["behavior_a"], top["behavior_b"]} == {"a", "b"}  # nearly coincident
    # ...and the far-away behavior pairs rank last.
    assert "c" in {ranked.iloc[-1]["behavior_a"], ranked.iloc[-1]["behavior_b"]}


def test_proximity_excludes_no_behavior_and_handles_empty():
    ranked = disc.rank_pairs_by_proximity(_pool(), ["a", "no_behavior"])
    assert ranked.empty  # the only pair involved no_behavior
    assert disc.rank_pairs_by_proximity(pd.DataFrame(), ["a", "b"]).empty


# ── Pair selection ─────────────────────────────────────────────────────────


def test_select_pairs_all_when_under_cap():
    pairs = disc.select_pairs(["a", "b", "c"], max_pairs=15)
    assert len(pairs) == 3  # 3 choose 2
    assert ("a", "b") in pairs


def test_select_pairs_drops_no_behavior():
    pairs = disc.select_pairs(["a", "b", "no_behavior"], max_pairs=15)
    assert pairs == [("a", "b")]


def test_select_pairs_cap_prefers_the_ranked_candidates():
    # ranked is best-first (closest centroids => most likely to be confused).
    ranked = pd.DataFrame([
        {"behavior_a": "b", "behavior_b": "c", "centroid_distance": 0.1},
        {"behavior_a": "a", "behavior_b": "c", "centroid_distance": 5.0},
        {"behavior_a": "a", "behavior_b": "b", "centroid_distance": 9.0},
    ])
    pairs = disc.select_pairs(["a", "b", "c"], ranked=ranked, max_pairs=1)
    assert pairs == [("b", "c")]  # the closest pair survives the cap


# ── Feature-set ladder ─────────────────────────────────────────────────────


def test_feature_ladder_gates_on_available_families():
    p = _project(use_video_features=True)

    # Video only, no context: "all" would be a bit-exact duplicate of pose+video,
    # so it is NOT trained (it used to be, wasting a third of the compute and
    # drawing a duplicate bar series).
    names = [s.name for s in disc.build_feature_sets(
        p, has_social=False, has_video=True, has_context=False)]
    assert names == ["pose_only", "pose_video"]

    # A project with no add-on families at all → baseline only.
    p2 = _project(use_video_features=False)
    names2 = [s.name for s in disc.build_feature_sets(
        p2, has_social=False, has_video=False, has_context=False)]
    assert names2 == ["pose_only"]

    # Context is its own rung — never folded into the pose baseline.
    names3 = [s.name for s in disc.build_feature_sets(
        p, has_social=False, has_video=True, has_context=True)]
    assert names3 == ["pose_only", "pose_context", "pose_video", "all_features"]

    # Multi-animal project picks up the social rung.
    names4 = [s.name for s in disc.build_feature_sets(
        p, has_social=True, has_video=True, has_context=True)]
    assert "pose_social" in names4


def test_pose_baseline_excludes_environment_features():
    """The 'pose-only' baseline must be the ANIMAL, not the arena.

    Regression test for the bug that made "Sniff Novel vs Sniff Familiar" — the
    same motor act at two different objects — score AUC 1.000 from "pose alone":
    a single ``body_centroid_to_roi_2_dist`` column was sitting inside the pose
    baseline. True pose scores 0.71 on that pair; the environment carries it.
    """
    from abel.validation import features as feat

    df = pd.DataFrame({
        "nose_to_tail_dist_mean": [1.0, 2.0],
        "head_angular_velocity_std": [0.1, 0.2],
        "body_centroid_to_roi_2_dist_energy": [5.0, 6.0],   # environment!
        "nose_to_target_dist_mean": [3.0, 4.0],             # environment!
        "flow_mag_local_mean": [0.5, 0.6],                  # video
        "social_approach_velocity_mean": [0.7, 0.8],        # social
    })
    pose = feat.pose_only_cols(df)
    assert "body_centroid_to_roi_2_dist_energy" not in pose
    assert "nose_to_target_dist_mean" not in pose
    assert "flow_mag_local_mean" not in pose
    assert "social_approach_velocity_mean" not in pose
    assert set(pose) == {"nose_to_tail_dist_mean", "head_angular_velocity_std"}

    assert set(feat.context_only_cols(df)) == {
        "body_centroid_to_roi_2_dist_energy", "nose_to_target_dist_mean"}
    # An "_energy" aggregation suffix must not read as a video feature.
    assert not feat.is_video_feature("body_centroid_to_roi_2_dist_energy")


# ── PairResult significance + matrices ─────────────────────────────────────


def _pair(name_a, name_b, base_auc, video_auc, seeds=3, spread=0.0) -> disc.PairResult:
    r = disc.PairResult(project_id="P", behavior_a="x", behavior_b="y",
                        name_a=name_a, name_b=name_b)
    r.order = ["pose_only", "pose_video"]
    r.labels = {"pose_only": "Pose only", "pose_video": "+ Video"}
    base = [base_auc] * seeds
    vid = [video_auc + (i - seeds / 2) * spread for i in range(seeds)]
    r.auc_seeds = {"pose_only": base, "pose_video": vid}
    r.auc = {"pose_only": base_auc, "pose_video": float(np.mean(vid))}
    paired = [v - b for v, b in zip(vid, base)]
    r.gain = {"pose_video": float(np.mean(paired))}
    r.gain_ci = {"pose_video": disc._ci95(paired)}
    r.gain_n = {"pose_video": len(paired)}
    return r


def test_is_significant_requires_ci_excluding_zero():
    # Consistent +0.20 gain across seeds → CI is 0 → significant.
    solid = _pair("Freeze", "Groom", 0.60, 0.80, spread=0.0)
    assert solid.is_significant("pose_video")

    # A tiny gain swamped by seed spread → CI crosses 0 → not significant.
    noisy = _pair("Sniff", "Eat", 0.80, 0.802, spread=0.20)
    assert not noisy.is_significant("pose_video")

    # A single seed can never be significant.
    single = _pair("A", "B", 0.5, 0.9, seeds=1)
    assert not single.is_significant("pose_video")


def test_matrices_are_symmetric_and_named():
    results = [
        _pair("Freeze", "Groom", 0.62, 0.88),
        _pair("Sniff", "Eat", 0.71, 0.74),
    ]
    sep = disc.separability_matrix(results)
    assert sep.loc["Freeze", "Groom"] == sep.loc["Groom", "Freeze"] == 0.62
    assert np.isnan(sep.loc["Freeze", "Freeze"])  # diagonal undefined

    gain = disc.gain_matrix(results, feature_set="pose_video")
    assert abs(gain.loc["Freeze", "Groom"] - 0.26) < 1e-9
    assert abs(gain.loc["Eat", "Sniff"] - 0.03) < 1e-9


def test_discrimination_rows_flags_unscorable_pair():
    bad = disc.PairResult(project_id="P", behavior_a="x", behavior_b="y",
                          name_a="Rare1", name_b="Rare2",
                          error="need >=8 training clips of each behavior (have 2/3)")
    df = disc.discrimination_rows([bad, _pair("Freeze", "Groom", 0.6, 0.9)])
    assert df[df["pair"] == "Rare1 vs Rare2"]["error"].iloc[0].startswith("need >=8")
    good = df[(df["pair"] == "Freeze vs Groom") & (df["feature_set"] == "pose_video")]
    assert abs(float(good["auc_gain_vs_pose"].iloc[0]) - 0.30) < 1e-9


def test_error_reduction_normalizes_for_the_ceiling():
    # A raw +0.004 AUC gain looks like nothing, but at a 0.984 baseline it has
    # closed a quarter of the remaining gap to perfect separability.
    near_ceiling = _pair("Groom", "Freeze", 0.984, 0.988)
    assert abs(near_ceiling.error_reduction("pose_video") - 0.25) < 1e-6

    # The same raw gain from a low baseline is a much smaller share of the error.
    low_base = _pair("Sniff", "Eat", 0.50, 0.504)
    assert abs(low_base.error_reduction("pose_video") - 0.008) < 1e-6

    # An already-perfect baseline has no headroom → undefined, not a divide-by-zero.
    perfect = _pair("Rear", "Eat", 1.0, 1.0)
    assert np.isnan(perfect.error_reduction("pose_video"))


def test_error_reduction_suppressed_only_when_there_is_no_error_left():
    # A pair the pose baseline already solves (AUC 0.9998) leaves ~nothing to
    # remove; wiping out that sliver would otherwise score a bogus +100%.
    solved = _pair("Rear", "Sniff", 0.9998, 1.0)
    assert np.isnan(solved.error_reduction("pose_video"))

    # The floor is only a divide-by-zero guard, NOT a significance filter. It used
    # to be 0.005, which suppressed a *significant* 75% error reduction on a pair
    # with 0.0041 of headroom. Whether a gain is real is decided by is_significant().
    assert disc.MIN_HEADROOM == 0.002
    borderline = _pair("Approach Familiar", "Rear", 0.99587, 0.99897)
    er = borderline.error_reduction("pose_video")
    assert np.isfinite(er) and er > 0.7          # ~75% of the remaining error

    real = _pair("Approach", "Walk", 0.974, 0.982)
    assert abs(real.error_reduction("pose_video") - (0.008 / 0.026)) < 1e-6


def test_confusable_pairs_table_is_hardest_first():
    results = [
        _pair("Rear", "Eat", 0.999, 0.999),     # trivially separable
        _pair("Sniff", "Eat", 0.892, 0.887),    # the genuinely hard pair
        _pair("Groom", "Freeze", 0.984, 0.988),
    ]
    tbl = disc.confusable_pairs_table(results)
    assert list(tbl["pair"])[0] == "Sniff vs Eat"       # hardest first
    assert list(tbl["pair"])[-1] == "Rear vs Eat"       # easiest last
    assert "pose_video_error_reduction" in tbl.columns
    # A feature family that makes a hard pair *worse* shows a negative reduction.
    assert float(tbl.iloc[0]["pose_video_error_reduction"]) < 0
