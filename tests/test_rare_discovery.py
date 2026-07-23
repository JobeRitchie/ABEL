"""Tests for the rare-behavior discovery analysis (clip hunting).

Covers the pure logic — ranking, discovery curves, effort-to-N, cross-validation
bookkeeping, the essence/UMAP rankers on separable synthetic data, and the
Prism/tidy exporters — without training models or reading a project from disk
(the end-to-end training + metric passes are exercised by the validation run).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.validation.analyses import rare_discovery as rd
from abel.validation.datamodel import ProjectRef


def _project(root: Path) -> ProjectRef:
    return ProjectRef(
        project_id="HC", name="HC", root=root,
        behavior_names={"wds": "Wet dog shake", "rear": "Rear",
                        "no_behavior": "No Behavior"})


# ── discovery-curve primitives ─────────────────────────────────────────────


def test_discovered_curve_is_cumulative_positive_count():
    is_pos = np.array([0, 1, 0, 1, 1], dtype=bool)
    np.testing.assert_array_equal(
        rd._discovered_curve(is_pos), np.array([0, 1, 1, 2, 3], dtype=float))


def test_first_reach_returns_one_indexed_position_or_none():
    disc = np.array([0, 0, 1, 1, 2, 3], dtype=float)
    assert rd._first_reach(disc, 1) == 3     # first positive at index 2 → 3 reviewed
    assert rd._first_reach(disc, 3) == 6
    assert rd._first_reach(disc, 4) is None  # never reaches 4


def test_rank_by_score_pushes_nan_last_and_orders_by_direction():
    score = np.array([0.2, np.nan, 0.9, 0.5])
    desc = rd._rank_by_score(score, descending=True)
    assert desc[0] == 2 and desc[-1] == 1          # highest first, NaN last
    asc = rd._rank_by_score(score, descending=False)
    assert asc[0] == 0 and asc[-1] == 1            # lowest first, NaN last


def test_budget_grid_is_sorted_unique_and_in_range():
    grid = rd._budget_grid(400)
    assert grid == sorted(set(grid))
    assert grid[0] >= 1 and grid[-1] <= 400
    assert len(grid) <= 25
    assert rd._budget_grid(5) == [1, 2, 3, 4, 5]   # small n → every point


# ── cross-validation bookkeeping ───────────────────────────────────────────


def test_seed_positives_are_unique_and_drawn_from_positives():
    pos_idx = np.arange(10, 40)
    rng = np.random.default_rng(0)
    sel = rd._seed_positives(pos_idx, 8, rng)
    assert len(sel) == 8 and len(set(sel.tolist())) == 8
    assert set(sel.tolist()).issubset(set(pos_idx.tolist()))


def test_seed_positives_capped_by_available():
    sel = rd._seed_positives(np.arange(3), 8, np.random.default_rng(0))
    assert len(sel) == 3


# ── essence & UMAP rankers on separable synthetic data ─────────────────────


def _separable_metrics(n_pos=40, n_neg=400, seed=0):
    """Metrics where positives sit high on centroid_speed_mean, negatives low."""
    rng = np.random.default_rng(seed)
    pos = pd.DataFrame({
        "centroid_speed_mean": rng.normal(120, 8, n_pos),
        "nose_speed_max": rng.normal(200, 15, n_pos),
        "body_length_mean": rng.normal(60, 3, n_pos),
    })
    neg = pd.DataFrame({
        "centroid_speed_mean": rng.normal(30, 8, n_neg),
        "nose_speed_max": rng.normal(60, 15, n_neg),
        "body_length_mean": rng.normal(60, 3, n_neg),
    })
    df = pd.concat([pos, neg], ignore_index=True)
    is_pos = np.array([True] * n_pos + [False] * n_neg)
    return df, is_pos


def test_rank_essence_enriches_positives_at_the_top():
    metrics, is_pos = _separable_metrics()
    seed = metrics.iloc[np.where(is_pos)[0][:8]]      # 8 exemplars
    order = rd._rank_essence(metrics, seed)
    assert order is not None
    assert sorted(order.tolist()) == list(range(len(metrics)))  # a permutation
    # Top-40 ranked should be far denser in positives than the 8.3% base rate.
    top = is_pos[order[:40]]
    assert top.mean() > 0.5


def test_rank_umap_returns_permutation_and_favours_exemplar_neighbourhood():
    metrics, is_pos = _separable_metrics()
    feat = metrics.to_numpy(float)
    pos_pos = np.where(is_pos)[0][:8]
    order = rd._rank_umap(np.delete(feat, pos_pos, axis=0), feat[pos_pos], seed=0)
    n = len(feat) - len(pos_pos)
    assert sorted(order.tolist()) == list(range(n))
    is_pos_cand = np.delete(is_pos, pos_pos)
    assert is_pos_cand[order[:32]].mean() > is_pos_cand.mean()  # enriched early


# ── iterative (re-extracting / re-lassoing) reveal loops ───────────────────


def test_essence_discovery_is_full_permutation_and_enriches():
    metrics, is_pos = _separable_metrics(n_pos=40, n_neg=400)
    pos_pos = np.where(is_pos)[0][:8]
    seed = metrics.iloc[pos_pos]                              # held-out-excluded seeds
    cand = metrics.drop(index=pos_pos).reset_index(drop=True)
    is_pos_cand = np.delete(is_pos, pos_pos)
    order = rd._essence_discovery(
        cand, is_pos_cand, seed, seed=0, batch=25, refit_budget=200,
        log=lambda _m: None)
    assert order is not None
    assert sorted(order.tolist()) == list(range(len(cand)))   # full permutation
    assert is_pos_cand[order[:32]].mean() > 0.5              # enriched at the top


def test_essence_feature_frame_uses_shipped_features_not_clip_metrics():
    """Essence's substrate is the pool's shipped features, indexed by segment_id.

    Drops meta columns, all-NaN / constant columns, and columns too sparse to
    anchor a criterion — but keeps the real feature columns (incl. oscillation).
    """
    pool = pd.DataFrame({
        "segment_id": ["a", "b", "c", "d"],
        "session_id": ["s"] * 4,
        "label": ["wds", "x", "x", "x"],
        "nose_oscillation_power_mean": [1.0, 2.0, 3.0, 4.0],   # real feature — kept
        "head_angular_velocity_std": [0.1, 0.2, 0.3, 0.4],     # real feature — kept
        "dead_all_nan": [np.nan] * 4,                          # dropped
        "constant": [5.0, 5.0, 5.0, 5.0],                      # dropped (no spread)
        "too_sparse": [1.0, np.nan, np.nan, np.nan],           # dropped (<50% finite)
    })
    frame = rd._essence_feature_frame(pool)
    assert list(frame.index) == ["a", "b", "c", "d"]           # segment_id index
    assert set(frame.columns) == {
        "nose_oscillation_power_mean", "head_angular_velocity_std"}
    assert "label" not in frame.columns and "segment_id" not in frame.columns


def test_criteria_match_mask_is_and_of_active_bounds():
    """The AND-box: a row must clear every enabled bound (mirrors mine match_all)."""
    from abel.services.clip_metrics_service import Criterion
    df = pd.DataFrame({"a": [0.0, 5.0, 5.0, 9.0], "b": [0.0, 1.0, 9.0, 9.0]})
    crits = [Criterion(metric_id="a", low=4.0, high=6.0),
             Criterion(metric_id="b", low=None, high=2.0)]
    mask = rd._criteria_match_mask(df, crits)
    assert mask.tolist() == [False, True, False, False]   # only row 1 clears both
    # No active bounds → nothing is "matched" (so ordering falls back to score).
    assert not rd._criteria_match_mask(df, []).any()


def test_essence_ranked_order_puts_criteria_matches_first():
    """The fix: criteria-matched clips lead the ranking, not just high-score ones.

    Reproduces the shipped dialog (AND-box + ranker); ranking by the continuous
    score alone was collapsing essence toward random on the real WDS pool.
    """
    metrics, is_pos = _separable_metrics(n_pos=40, n_neg=400)
    seed = metrics.iloc[np.where(is_pos)[0][:8]]
    order = rd._essence_ranked_order(seed, metrics, metrics)
    assert order is not None
    assert sorted(order.tolist()) == list(range(len(metrics)))   # full permutation
    # The AND-box front-loads positives harder than the base rate.
    assert is_pos[order[:40]].mean() > 0.6


def test_essence_discovery_returns_none_without_separable_signal():
    # All-constant metrics → no essence → caller falls back to random.
    flat = pd.DataFrame({"a": np.ones(60), "b": np.ones(60)})
    is_pos = np.array([True] * 10 + [False] * 50)
    seed = flat.iloc[:5]
    out = rd._essence_discovery(flat.iloc[5:].reset_index(drop=True), is_pos[5:],
                                seed, seed=0, batch=10, refit_budget=40,
                                log=lambda _m: None)
    assert out is None


def test_umap_discovery_is_full_permutation_and_enriches():
    metrics, is_pos = _separable_metrics()
    emb = metrics.to_numpy(float)                             # separable stand-in embedding
    pos_pos = np.where(is_pos)[0][:8]
    cand_emb = np.delete(emb, pos_pos, axis=0)
    is_pos_cand = np.delete(is_pos, pos_pos)
    order = rd._umap_discovery(cand_emb, is_pos_cand, emb[pos_pos],
                               seed=0, batch=25, refit_budget=200)
    assert sorted(order.tolist()) == list(range(len(cand_emb)))
    assert is_pos_cand[order[:32]].mean() > is_pos_cand.mean()


def test_umap_discovery_beats_frozen_centroid_on_elongated_manifold():
    # Positives lie along an elongated manifold; seeds sit only at one end.  A
    # frozen single centroid ranks the far positives behind off-manifold negatives,
    # but re-lassoing (drifting the centroid as positives confirm) walks the whole
    # manifold — the compounding a frozen ranker cannot do.
    rng = np.random.default_rng(2)
    xs = np.linspace(0, 8, 40)
    pos = np.column_stack([xs, np.zeros(40)]) + rng.normal(0, 0.1, (40, 2))
    negs = rng.normal([4, 6], 0.6, size=(200, 2))            # 6 units off the line
    emb = np.vstack([pos, negs])
    is_pos = np.array([True] * 40 + [False] * 200)
    seed_idx = np.arange(6)                                   # the six lowest-x positives
    cand = np.delete(emb, seed_idx, axis=0)
    is_pos_cand = np.delete(is_pos, seed_idx)
    seed_emb = emb[seed_idx]

    it = rd._umap_discovery(cand, is_pos_cand, seed_emb, seed=0, batch=10,
                            refit_budget=len(cand))
    frozen = np.argsort(                                      # single, un-drifted centroid
        np.linalg.norm(cand - seed_emb.mean(axis=0), axis=1))
    it_eff = rd._first_reach(np.cumsum(is_pos_cand[it].astype(float)), 34)
    fr_eff = rd._first_reach(np.cumsum(is_pos_cand[frozen].astype(float)), 34)
    assert it_eff is not None and fr_eff is not None
    assert it_eff < fr_eff          # drifting the lasso reaches the far positives sooner


# ── aggregation (_assemble_result) ─────────────────────────────────────────


def _perfect_vs_random_arrays():
    """Essence finds a positive every clip; random finds them at ~10% rate."""
    ess = np.minimum(np.arange(1, 201), 50).astype(float)          # 1,2,...,50,50,...
    rnd = (np.arange(1, 201) * 0.1)                                # linear ~10%
    return {rd.STRATEGY_ESSENCE: [ess, ess], rd.STRATEGY_RANDOM: [rnd, rnd]}


def test_assemble_result_effort_and_enrichment(tmp_path):
    proj = _project(tmp_path)
    per_seed = _perfect_vs_random_arrays()
    res = rd._assemble_result(
        per_seed, (rd.STRATEGY_ESSENCE, rd.STRATEGY_RANDOM), proj, "wds",
        "Wet dog shake", pool_label="reviewed", n_pool=200, n_pos_pool=50,
        n_seed_pos=8, prevalence=0.1, sec_per_clip_review=4.0,
        effort_targets=(10, 25), display_budget=200)
    ess = res.curves[rd.STRATEGY_ESSENCE]
    rnd = res.curves[rd.STRATEGY_RANDOM]
    # Essence reaches 10 positives in 10 clips; random needs ~100.
    assert ess.effort_to_n[10] == 10
    assert rnd.effort_to_n[10] > ess.effort_to_n[10]
    # Discovery is monotonic non-decreasing.
    ys = [p.n_found_mean for p in ess.points]
    assert all(b >= a for a, b in zip(ys, ys[1:]))
    # Enrichment over random > 1 for essence early.
    assert res.enrichment_at(rd.STRATEGY_ESSENCE, 25) > 1.0
    # Per-seed effort replicates are retained for Prism.
    assert ess.effort_to_n_seeds[10] == [10.0, 10.0]
    # Cells are tagged with the pool + strategy.
    assert any(c.analysis == "rare_discovery" and c.config_name == "reviewed:essence"
               for c in res.cells)


# ── effort-to-a-good-MODEL (F1 / PR-AUC vs labeling effort) ─────────────────


def test_effort_to_threshold_finds_first_reaching_checkpoint():
    traj = [(20, 1, 0.30, 0.40), (45, 3, float("nan"), 0.70), (70, 8, 0.85, 0.90)]
    assert rd._effort_to_threshold(traj, 2, 0.80) == 70      # F1 col
    assert rd._effort_to_threshold(traj, 3, 0.65) == 45      # PR-AUC col
    assert rd._effort_to_threshold(traj, 2, 0.99) is None     # never reached
    # NaN checkpoints are skipped, not treated as a hit.
    assert rd._effort_to_threshold([(10, 0, float("nan"), float("nan"))], 2, 0.0) is None


def _quality_trajs():
    """AL climbs to good F1/PR-AUC fast; random lags — two identical seeds each."""
    al = [(20, 2, 0.30, 0.40), (45, 6, 0.60, 0.70),
          (70, 12, 0.85, 0.90), (95, 18, 0.90, 0.95)]
    rnd = [(20, 2, 0.20, 0.30), (45, 4, 0.35, 0.45),
           (70, 6, 0.50, 0.60), (95, 8, 0.60, 0.70)]
    return {rd.STRATEGY_AL: [al, al], rd.STRATEGY_RANDOM: [rnd, rnd]}


def test_assemble_quality_effort_targets_and_curves(tmp_path):
    proj = _project(tmp_path)
    res = rd._assemble_quality(
        _quality_trajs(), (rd.STRATEGY_AL, rd.STRATEGY_RANDOM), proj, "wds",
        "Wet dog shake", pool_label="reviewed", k0=20, seed_pos=5,
        f1_targets=(0.50, 0.80), pr_auc_targets=(0.90,), frac_targets=(0.90,),
        sec_per_clip_review=4.0)
    al = res.curves[rd.STRATEGY_AL]
    rnd = res.curves[rd.STRATEGY_RANDOM]
    # Best achievable feeds the fraction-of-best targets.
    assert res.best_f1 == 0.90 and res.best_pr_auc == 0.95
    # AL reaches F1≥0.80 at 70 clips; random never does within budget.
    assert al.effort["F1≥0.80"] == 70
    assert "F1≥0.80" not in rnd.effort
    # A target both reach → AL needs fewer clips (savings > 1).
    assert res.savings_vs_random(rd.STRATEGY_AL, "F1≥0.50") == 70 / 45
    # Fraction-of-best target exists and is met by AL.
    assert "F1≥90%max" in al.effort
    # Curve points aggregate per checkpoint; F1 rises monotonically for AL.
    ys = [p.f1_mean for p in al.points]
    assert ys[-1] == 0.90 and all(b >= a for a, b in zip(ys, ys[1:]))
    # Cells tagged for the aggregate store.
    assert any(c.analysis == "effort_to_quality"
               and c.config_name == "reviewed:active_learning" for c in res.cells)


def test_prism_quality_and_effort_tables(tmp_path):
    proj = _project(tmp_path)
    res = rd._assemble_quality(
        _quality_trajs(), (rd.STRATEGY_AL, rd.STRATEGY_RANDOM), proj, "wds",
        "Wet dog shake", pool_label="reviewed", k0=20, seed_pos=5,
        f1_targets=(0.80,), pr_auc_targets=(0.90,), frac_targets=(0.90,),
        sec_per_clip_review=4.0)
    # XY tables carry per-seed replicate subcolumns, not a mean: Prism cannot
    # ingest a CI half-width, so a mean-only curve could never get error bars.
    xy = rd.prism_quality(res, "f1")
    assert list(xy.columns)[0] == "Clips reviewed"
    assert any(c.startswith("Active Learning:") for c in xy.columns)
    assert any(c.startswith("Random clips:") for c in xy.columns)
    pr = rd.prism_quality(res, "pr_auc")
    assert any(c.startswith("Active Learning:") for c in pr.columns)
    eff = rd.prism_effort_to_quality(res)
    assert list(eff.columns)[0] == "Strategy"
    assert any(c.startswith("F1≥0.80:") for c in eff.columns)   # per-seed replicates


# ── whole-video reference ──────────────────────────────────────────────────


def test_whole_video_minutes_from_segment_extents(tmp_path):
    d = tmp_path / "derived" / "representations"
    d.mkdir(parents=True)
    # Two sessions, 18000 and 9000 frames @30fps → 10 + 5 = 15 minutes.
    pd.DataFrame({
        "session_id": ["s1"] * 3 + ["s2"] * 3,
        "end_frame": [100, 9000, 18000, 50, 4000, 9000],
    }).to_parquet(d / "segment_features.parquet")
    proj = _project(tmp_path)
    (tmp_path / "project.yaml").write_text("default_fps: 30.0\n", encoding="utf-8")
    assert abs(rd.whole_video_minutes(proj) - 15.0) < 1e-6


# ── auto-target: which behaviour is the rare one in this project? ──────────


def _project_with_bouts(root: Path, per_behavior_frames: dict[str, int]) -> ProjectRef:
    """A project whose dense bout detections give each behaviour a known prevalence."""
    reps = root / "derived" / "representations"
    reps.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"session_id": ["s1", "s2"],
                  "end_frame": [9000, 9000]}).to_parquet(
        reps / "segment_features.parquet")
    bouts = root / "derived" / "behavior_bouts"
    bouts.mkdir(parents=True, exist_ok=True)
    for bid, frames in per_behavior_frames.items():
        pd.DataFrame({
            "session_id": ["s1", "s2"],
            "start_frame": [100, 200],
            "duration_frames": [frames, frames],
        }).to_parquet(bouts / f"{bid}_bouts.parquet")
    (root / "project.yaml").write_text("default_fps: 30.0\n", encoding="utf-8")
    return _project(root)


def test_rank_behaviors_by_rarity_orders_rarest_first(tmp_path):
    proj = _project_with_bouts(tmp_path, {"wds": 30, "rear": 900})
    ranking = rd.rank_behaviors_by_rarity(proj, ["wds", "rear"])
    assert [bid for bid, _n, _v in ranking] == ["wds", "rear"]
    assert ranking[0][1] == "Wet dog shake"
    assert ranking[0][2] < ranking[1][2]


def test_rank_behaviors_by_rarity_drops_excluded_and_undetected(tmp_path):
    """A behaviour with no bouts file is absent, not ranked 'rarest' at zero."""
    proj = _project_with_bouts(tmp_path, {"rear": 900})     # no wds bouts file
    assert [b for b, _n, _v in rd.rank_behaviors_by_rarity(proj, ["wds", "rear"])] == ["rear"]
    assert rd.rank_behaviors_by_rarity(
        proj, ["wds", "rear"], exclude_behavior_ids=["rear"]) == []


def _add_traces(root: Path, per_frame: dict[str, int], *, sessions=("s1",)) -> None:
    """A competitive temporal-refinement run: per-frame winner + per-behaviour probs."""
    inf = root / "derived" / "temporal_refinement" / "target_behavior" / "inference_x"
    (inf / "probability_traces").mkdir(parents=True, exist_ok=True)
    (inf.parent / "latest.json").write_text(
        json.dumps({"inference_dir": str(inf)}), encoding="utf-8")
    pred: list[str] = []
    for bid, n in per_frame.items():
        pred += [bid] * n
    for sid in sessions:
        frame = {"frame": list(range(len(pred))), "predicted_behavior": pred}
        for bid in per_frame:
            frame[f"prob_{bid}"] = [1.0 if p == bid else 0.0 for p in pred]
        pd.DataFrame(frame).to_parquet(
            inf / "probability_traces" / f"{sid}_trace.parquet")


def test_dense_traces_are_preferred_over_stale_bouts(tmp_path):
    """A current competitive run outranks the exported bouts, which lose identity.

    Regression: fear conditioning had a complete 69-session run whose bouts were all
    stamped ``behavior_id = "target_behavior"``, so the only per-behaviour record was
    the traces.  Reading the stale per-behaviour bouts instead ranked freezing —
    49 % of frames — as the project's rarest behaviour.
    """
    proj = _project_with_bouts(tmp_path, {"wds": 900, "rear": 2})   # stale + inverted
    _add_traces(tmp_path, {"wds": 10, "rear": 90})
    assert rd.prevalence_source(proj, ["wds", "rear"]) == rd.PREVALENCE_SOURCE_TRACES
    ranking = rd.rank_behaviors_by_rarity(proj, ["wds", "rear"])
    assert [bid for bid, _n, _v in ranking] == ["wds", "rear"]
    assert ranking[0][2] == pytest.approx(0.10)
    assert ranking[1][2] == pytest.approx(0.90)


def test_single_behavior_traces_are_rejected(tmp_path):
    """``predicted_behavior`` is an argmax — one behaviour wins 100 % of frames.

    A non-competitive run must fall through to the bout detections rather than
    report the lone behaviour as occupying the whole session.
    """
    proj = _project_with_bouts(tmp_path, {"wds": 30, "rear": 900})
    _add_traces(tmp_path, {"wds": 100})          # single-behaviour inference
    assert rd.prevalence_source(proj, ["wds", "rear"]) == rd.PREVALENCE_SOURCE_BOUTS
    ranking = rd.rank_behaviors_by_rarity(proj, ["wds", "rear"])
    assert [bid for bid, _n, _v in ranking] == ["wds", "rear"]
    assert ranking[0][2] < 1.0


def test_session_gated_behavior_is_not_a_hunt_target(tmp_path):
    """Rare-because-impossible is not rare-because-hard.

    ``shock`` is abundant in a third of sessions and structurally absent from the
    rest (no shock delivered), so its project-wide mean looks rarest — but hunting
    it measures finding the shock *sessions*, not rare-behaviour discovery.  ``wds``
    is uniformly rare and is the real target.
    """
    sessions = [f"s{i}" for i in range(12)]
    proj = _project_with_bouts(tmp_path, {})
    (tmp_path / "derived" / "representations").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"session_id": sessions, "end_frame": [9000] * len(sessions)}
                 ).to_parquet(tmp_path / "derived" / "representations"
                              / "segment_features.parquet")
    inf = tmp_path / "derived" / "temporal_refinement" / "target_behavior" / "inference_x"
    (inf / "probability_traces").mkdir(parents=True, exist_ok=True)
    (inf.parent / "latest.json").write_text(
        json.dumps({"inference_dir": str(inf)}), encoding="utf-8")
    for i, sid in enumerate(sessions):
        # shock: 3% of frames in a third of sessions, none in the rest — so its
        # project-wide mean (1%) lands BELOW uniformly-rare wds (2%), while its
        # level where it can occur (3%) is higher.
        n_shock = 30 if i % 3 == 0 else 0
        pred = ["shock"] * n_shock + ["wds"] * 20 + ["other"] * (1000 - n_shock - 20)
        pd.DataFrame({
            "predicted_behavior": pred,
            "prob_shock": [0.0] * len(pred), "prob_wds": [0.0] * len(pred),
        }).to_parquet(inf / "probability_traces" / f"{sid}_trace.parquet")

    raw = rd.rank_behaviors_by_rarity(proj, ["shock", "wds"], skip_gated=False)
    assert raw[0][0] == "shock"                    # project-wide mean says "rarest"

    msgs: list[str] = []
    guarded = rd.rank_behaviors_by_rarity(proj, ["shock", "wds"], progress_cb=msgs.append)
    assert [bid for bid, _n, _v in guarded] == ["wds"]
    assert any("structurally gated" in m for m in msgs)


def test_uniformly_rare_behavior_survives_the_gate(tmp_path):
    """The guard must not eat a genuinely rare behaviour that occurs everywhere."""
    sessions = [f"s{i}" for i in range(12)]
    proj = _project_with_bouts(tmp_path, {})
    (tmp_path / "derived" / "representations").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"session_id": sessions, "end_frame": [9000] * len(sessions)}
                 ).to_parquet(tmp_path / "derived" / "representations"
                              / "segment_features.parquet")
    inf = tmp_path / "derived" / "temporal_refinement" / "target_behavior" / "inference_x"
    (inf / "probability_traces").mkdir(parents=True, exist_ok=True)
    (inf.parent / "latest.json").write_text(
        json.dumps({"inference_dir": str(inf)}), encoding="utf-8")
    for i, sid in enumerate(sessions):
        pred = ["wds"] * (4 + i % 3) + ["other"] * 1000       # rare, but always present
        pd.DataFrame({
            "predicted_behavior": pred,
            "prob_wds": [0.0] * len(pred), "prob_other": [0.0] * len(pred),
        }).to_parquet(inf / "probability_traces" / f"{sid}_trace.parquet")
    ranking = rd.rank_behaviors_by_rarity(proj, ["wds", "other"])
    assert ranking[0][0] == "wds"


def test_rank_behaviors_by_rarity_ignores_empty_bouts_file(tmp_path):
    """An EMPTY bouts file is unmeasurable, not 'rarest at zero'.

    Regression: fear conditioning shipped a zero-row ``Freeze_bouts.parquet``, so
    freezing averaged a time fraction of exactly 0.0 and won the rarity ranking
    outright — while being that project's single most abundant behaviour.
    """
    proj = _project_with_bouts(tmp_path, {"wds": 30, "rear": 900})
    empty = pd.DataFrame({"session_id": [], "start_frame": [], "duration_frames": []})
    empty.to_parquet(tmp_path / "derived" / "behavior_bouts" / "freeze_bouts.parquet")
    ranking = rd.rank_behaviors_by_rarity(proj, ["freeze", "wds", "rear"])
    assert [bid for bid, _n, _v in ranking] == ["wds", "rear"]


def test_prevalence_averages_only_over_run_covered_sessions(tmp_path):
    """Sessions no deployment run touched are not sessions with zero behaviour.

    Zero-filling every pool session penalised whichever behaviour happened to be
    deployed over fewer of them — a property of when the run was launched, not of
    rarity.  Here both behaviours run on s1 only, so adding an untouched s2 to the
    pool must not change their prevalence at all.
    """
    proj = _project_with_bouts(tmp_path, {})
    reps = tmp_path / "derived" / "representations"
    pd.DataFrame({"session_id": ["s1", "s2"], "end_frame": [9000, 9000]}).to_parquet(
        reps / "segment_features.parquet")
    bouts = tmp_path / "derived" / "behavior_bouts"
    for bid, frames in (("wds", 90), ("rear", 900)):
        pd.DataFrame({"session_id": ["s1"], "start_frame": [100],
                      "duration_frames": [frames]}).to_parquet(
            bouts / f"{bid}_bouts.parquet")
    ranking = dict((n, v) for _b, n, v in rd.rank_behaviors_by_rarity(proj, ["wds", "rear"]))
    # Measured on s1 alone — NOT halved by the untouched s2.
    assert ranking["Wet dog shake"] == pytest.approx(90 / 9000)
    assert ranking["Rear"] == pytest.approx(900 / 9000)


def test_stale_eval_bouts_fall_back_to_label_prevalence(tmp_path):
    """A handful of eval-split bouts is not a deployment read-out — use labels.

    Fear conditioning's ``behavior_bouts/`` held 133 bouts across 69 sessions
    (0.08 % of frames), written by the *evaluation* pipeline rather than a
    deployment run.  Ranking on that is meaningless, so the ranker falls back to
    label prevalence and says so.
    """
    proj = _project_with_bouts(tmp_path, {"wds": 2, "rear": 3})   # ~0.03 % of frames
    ts = tmp_path / "derived" / "training_sets"
    ts.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "segment_id": [f"s{i}" for i in range(10)],
        "session_id": ["s1"] * 10,
        "start_frame": list(range(0, 300, 30)),
        "end_frame": list(range(30, 330, 30)),
        "label": ["rear"] * 8 + ["wds"] * 2,
    }).to_parquet(ts / "training_set.parquet")

    msgs: list[str] = []
    ranking = rd.rank_behaviors_by_rarity(proj, ["wds", "rear"], progress_cb=msgs.append)
    assert rd.prevalence_source(proj, ["wds", "rear"]) == rd.PREVALENCE_SOURCE_LABELS
    assert [bid for bid, _n, _v in ranking] == ["wds", "rear"]
    assert any("stale evaluation artifact" in m for m in msgs)
    assert any(rd.LABEL_SOURCE_CAVEAT in m for m in msgs)


# ── preflight: enough examples to hunt, or go label more first? ────────────


def _project_with_bouts_and_labels(root: Path, per_behavior_frames: dict[str, int],
                                   labels: dict[str, int]) -> ProjectRef:
    """A project with known prevalence AND a known confirmed-example count."""
    proj = _project_with_bouts(root, per_behavior_frames)
    rows = []
    for bid, n in labels.items():
        for i in range(n):
            rows.append({"segment_id": f"{bid}-{i}", "label": bid,
                         "label_source": "review", "reviewer_confidence": 1.0,
                         # Several sessions so the holdout split has groups to draw
                         # from; the pool is what the counts are read off.
                         "session_id": f"s{i % 4}", "animal_id": f"m{i % 4}",
                         "feature_a": float(i)})
    d = root / "derived" / "training_sets"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "training_set.parquet")
    return proj


def test_preflight_blocks_a_behavior_with_too_few_examples(tmp_path):
    # Wet dog shake is by far the rarest, but only 6 clips of it were ever
    # confirmed — fewer than the 8 exemplars the definition would consume.
    proj = _project_with_bouts_and_labels(
        tmp_path, {"wds": 30, "rear": 900}, {"wds": 6, "rear": 200})
    pf = rd.preflight_project(proj, ["wds", "rear"], n_seed_pos=8)
    wds, rear = pf.behaviors[0], pf.behaviors[1]
    assert wds.behavior_name == "Wet dog shake" and wds.rank == 1
    assert wds.status == rd.PREFLIGHT_BLOCKED
    assert "Label more" in wds.note
    # The hunt falls back to the next-rarest behaviour that IS runnable.
    assert pf.target is rear
    assert "Wet dog shake" not in pf.blocked_note()[:20]  # names the problem first
    assert "Falling back to Rear" in pf.blocked_note()


def test_preflight_warns_when_evidence_is_thin_but_runnable(tmp_path):
    proj = _project_with_bouts_and_labels(
        tmp_path, {"wds": 30, "rear": 900}, {"wds": 20, "rear": 200})
    pf = rd.preflight_project(proj, ["wds", "rear"], n_seed_pos=8,
                              min_effort_target=10)
    wds = pf.behaviors[0]
    # ~15 confirmed in the pool − 8 exemplars: enough to cross-validate, but
    # fewer than the smallest effort target, so the bars would be empty.
    assert 2 <= wds.n_held_out < 10
    assert wds.status == rd.PREFLIGHT_WARN
    assert pf.target is wds                 # still the behaviour we'd hunt
    assert pf.blocked_note() == ""          # nothing is blocked


def test_preflight_passes_a_behavior_with_plenty_of_examples(tmp_path):
    proj = _project_with_bouts_and_labels(
        tmp_path, {"wds": 30, "rear": 900}, {"wds": 60, "rear": 200})
    pf = rd.preflight_project(proj, ["wds", "rear"], n_seed_pos=8)
    assert pf.behaviors[0].status == rd.PREFLIGHT_OK
    assert pf.target.behavior_name == "Wet dog shake"


def test_preflight_counts_only_the_hunting_pool_not_the_holdout(tmp_path):
    """The count shown must be the positives the hunt can actually use."""
    proj = _project_with_bouts_and_labels(
        tmp_path, {"wds": 30, "rear": 900}, {"wds": 40, "rear": 200})
    pf = rd.preflight_project(proj, ["wds", "rear"], n_seed_pos=8, test_size=0.25)
    # One of the four sessions is held out, so a quarter of the clips are not
    # available to hunt over.
    assert 0 < pf.behaviors[0].n_labeled < 40


def test_preflight_rows_flag_the_behavior_that_would_be_hunted(tmp_path):
    proj = _project_with_bouts_and_labels(
        tmp_path, {"wds": 30, "rear": 900}, {"wds": 6, "rear": 200})
    rows = rd.preflight_rows([rd.preflight_project(proj, ["wds", "rear"],
                                                   n_seed_pos=8)])
    hunted = [r for r in rows if r["would_be_hunted"]]
    assert [r["behavior"] for r in hunted] == ["Rear"]
    assert {r["status"] for r in rows} == {rd.PREFLIGHT_BLOCKED, rd.PREFLIGHT_OK}


def test_preflight_reports_an_unreadable_project_instead_of_raising(tmp_path):
    proj = _project(tmp_path)          # no bouts, no training set
    pf = rd.preflight_project(proj, ["wds"])
    assert pf.target is None
    assert pf.error and pf.blocked_note() == pf.error


# ── combined (cross-project) panels ────────────────────────────────────────


def _two_project_results(tmp_path):
    """The same essence-beats-random result from two different projects."""
    out = []
    for pid in ("HC", "OFT"):
        proj = ProjectRef(project_id=pid, name=pid, root=tmp_path,
                          behavior_names={"wds": "Wet dog shake"})
        out.append(rd._assemble_result(
            _perfect_vs_random_arrays(), (rd.STRATEGY_ESSENCE, rd.STRATEGY_RANDOM),
            proj, "wds", "Wet dog shake", pool_label="reviewed", n_pool=200,
            n_pos_pool=50, n_seed_pos=8, prevalence=0.1, sec_per_clip_review=4.0,
            effort_targets=(10, 25), display_budget=200))
    return out


def test_combined_effort_target_is_reached_by_every_arm_in_every_project(tmp_path):
    results = _two_project_results(tmp_path)
    # Essence reaches both 10 and 25; random only ever reaches 10 within the pool,
    # so 10 is the only target the bars can compare every arm at.
    assert rd.combined_effort_target(results) == 10
    # If one project's random arm never reaches 10 either, there is no honest
    # shared target left — better none than a silently unpaired comparison.
    results[1].curves[rd.STRATEGY_RANDOM].effort_to_n[10] = float("nan")
    assert rd.combined_effort_target(results) is None


def test_combined_rows_carry_enrichment_and_savings_per_project(tmp_path):
    results = _two_project_results(tmp_path)
    rows = rd.combined_rows(results, labels={"HC": "Home cage", "OFT": "Open field"})
    df = pd.DataFrame(rows)
    assert set(df["project"]) == {"Home cage", "Open field"}
    ess = df[df["strategy"] == "Essence Miner"]
    # Essence is enriched over prevalence and cheaper than random in both projects.
    assert (ess["enrichment_at_k"] > 1.0).all()
    assert (ess["fold_fewer_clips_than_random"] > 1.0).all()
    # Random is its own baseline: exactly 1× the random effort.
    rnd = df[df["strategy"] == "Random clips"]
    assert np.allclose(rnd["fold_fewer_clips_than_random"], 1.0)


def test_combined_figures_render(tmp_path):
    results = _two_project_results(tmp_path)
    labels = {"HC": "Home cage", "OFT": "Open field"}
    for fn, name in ((rd.plot_combined_discovery, "disc.png"),
                     (rd.plot_combined_enrichment, "enrich.png"),
                     (rd.plot_combined_savings, "saved.png")):
        p = tmp_path / name
        fn(results, p, labels=labels)
        assert p.exists() and p.stat().st_size > 0


def test_prism_combined_enrichment_is_strategy_by_project(tmp_path):
    results = _two_project_results(tmp_path)
    tbl = rd.prism_combined_enrichment(results, labels={"HC": "Home cage",
                                                        "OFT": "Open field"})
    assert list(tbl.columns)[0] == "Strategy"
    assert "Home cage · Wet dog shake" in tbl.columns
    assert set(tbl["Strategy"]) == {"Essence Miner", "Random clips"}


# ── Prism exporters (one row-title column first; replicate columns) ─────────


def test_prism_behavior_rarity_is_wide_sessions_by_behavior():
    per = pd.DataFrame({
        "session": ["s1", "s1", "s2", "s2"],
        "behavior": ["Wet dog shake", "Rear", "Wet dog shake", "Rear"],
        "time_fraction": [0.004, 0.02, 0.003, 0.018],
        "bout_rate": [0.4, 1.3, 0.3, 1.2], "n_bouts": [2, 8, 1, 7],
    })
    res = rd.BehaviorRarityResult(
        project_id="HC", target_name="Wet dog shake", measure="time_fraction",
        per_session=per, means={"Wet dog shake": 0.0035, "Rear": 0.019},
        target_rank=1, n_behaviors=2, n_sessions=2)
    tbl = rd.prism_behavior_rarity(res)
    assert list(tbl.columns)[0] == "Session"
    assert list(tbl.columns)[1:] == ["Wet dog shake", "Rear"]   # rarest first
    assert len(tbl) == 2                                         # one row per session


def test_prism_effort_has_per_seed_replicate_columns(tmp_path):
    proj = _project(tmp_path)
    res = rd._assemble_result(
        _perfect_vs_random_arrays(), (rd.STRATEGY_ESSENCE, rd.STRATEGY_RANDOM),
        proj, "wds", "Wet dog shake", pool_label="reviewed", n_pool=200,
        n_pos_pool=50, n_seed_pos=8, prevalence=0.1, sec_per_clip_review=4.0,
        effort_targets=(10,), display_budget=200)
    eff = rd.prism_effort(res)
    assert list(eff.columns)[0] == "Strategy"
    assert "N=10:1" in eff.columns and "N=10:2" in eff.columns  # 2 seed replicates


def test_prism_discovery_is_xy_with_one_column_per_strategy(tmp_path):
    proj = _project(tmp_path)
    res = rd._assemble_result(
        _perfect_vs_random_arrays(), (rd.STRATEGY_ESSENCE, rd.STRATEGY_RANDOM),
        proj, "wds", "Wet dog shake", pool_label="reviewed", n_pool=200,
        n_pos_pool=50, n_seed_pos=8, prevalence=0.1, sec_per_clip_review=4.0,
        effort_targets=(10,), display_budget=200)
    xy = rd.prism_discovery(res)
    assert list(xy.columns)[0] == "Clips reviewed"
    # One replicate subcolumn per seed per strategy, padded to a uniform width so
    # Prism's positional subcolumn paste lines the datasets up.
    ess = [c for c in xy.columns if c.startswith("Essence Miner:")]
    rand = [c for c in xy.columns if c.startswith("Random clips:")]
    assert ess and len(ess) == len(rand) == res.n_seeds
    # Replicate means must reproduce the mean the figure is drawn from.
    pt = next(p for p in res.curves[rd.STRATEGY_ESSENCE].points)
    row = xy.loc[xy["Clips reviewed"] == pt.n_reviewed, ess].iloc[0]
    assert row.mean() == pytest.approx(pt.n_found_mean)


def test_write_prism_emits_files_and_readme(tmp_path):
    proj = _project(tmp_path)
    res = rd._assemble_result(
        _perfect_vs_random_arrays(), (rd.STRATEGY_ESSENCE, rd.STRATEGY_RANDOM),
        proj, "wds", "Wet dog shake", pool_label="reviewed", n_pool=200,
        n_pos_pool=50, n_seed_pos=8, prevalence=0.1, sec_per_clip_review=4.0,
        effort_targets=(10,), display_budget=200)
    written = rd.write_prism(tmp_path, reviewed=res)
    names = {p.name for p in written}
    assert "prism_discovery_reviewed.csv" in names
    assert "prism_effort_reviewed.csv" in names
    assert "README_PRISM_rare_discovery.txt" in names


def test_write_prism_stem_keeps_projects_from_overwriting_each_other(tmp_path):
    proj = _project(tmp_path)
    res = rd._assemble_result(
        _perfect_vs_random_arrays(), (rd.STRATEGY_ESSENCE, rd.STRATEGY_RANDOM),
        proj, "wds", "Wet dog shake", pool_label="reviewed", n_pool=200,
        n_pos_pool=50, n_seed_pos=8, prevalence=0.1, sec_per_clip_review=4.0,
        effort_targets=(10,), display_budget=200)
    a = {p.name for p in rd.write_prism(tmp_path, reviewed=res, stem="HomeCage")}
    b = {p.name for p in rd.write_prism(tmp_path, reviewed=res, stem="OpenField")}
    assert "prism_discovery_reviewed__HomeCage.csv" in a
    assert "prism_discovery_reviewed__OpenField.csv" in b
    assert (tmp_path / "prism" / "prism_discovery_reviewed__HomeCage.csv").exists()


def test_write_prism_emits_quality_tables(tmp_path):
    """The effort-to-quality tables ship in the same Prism bundle."""
    proj = _project(tmp_path)
    q = rd._assemble_quality(
        _quality_trajs(), (rd.STRATEGY_AL, rd.STRATEGY_RANDOM), proj, "wds",
        "Wet dog shake", pool_label="reviewed", k0=20, seed_pos=5,
        f1_targets=(0.80,), pr_auc_targets=(0.90,), frac_targets=(0.90,),
        sec_per_clip_review=4.0)
    names = {p.name for p in rd.write_prism(tmp_path, quality=q)}
    assert "prism_quality_f1.csv" in names
    assert "prism_quality_prauc.csv" in names
    assert "prism_effort_to_quality.csv" in names


# ── a dropped arm must never look like a tested-and-lost arm ───────────────


def test_disabled_strategy_note_names_the_arm_and_the_reason(tmp_path):
    """An absent line reads as "we tested it and it did nothing" — say otherwise.

    This is the HomeCage failure: the Essence Miner arm was disabled because the
    pose drive was unmounted, and the figure showed three arms with no hint that
    a fourth had been silently dropped.
    """
    proj = _project(tmp_path)
    res = rd._assemble_result(
        _perfect_vs_random_arrays(), (rd.STRATEGY_ESSENCE, rd.STRATEGY_RANDOM),
        proj, "wds", "Wet dog shake", pool_label="reviewed", n_pool=200,
        n_pos_pool=50, n_seed_pos=8, prevalence=0.1, sec_per_clip_review=4.0,
        effort_targets=(10,), display_budget=200)
    assert rd._disabled_note(res) == ""          # nothing dropped → no footnote
    res.disabled_strategies = {rd.STRATEGY_ESSENCE: "raw pose unreadable."}
    note = rd._disabled_note(res)
    assert "NOT TESTED" in note
    assert "Essence Miner" in note and "raw pose unreadable." in note


def test_discovery_curve_figure_renders_with_a_disabled_arm(tmp_path):
    """The footnote must not break the figure it is stamped onto."""
    proj = _project(tmp_path)
    res = rd._assemble_result(
        _perfect_vs_random_arrays(), (rd.STRATEGY_ESSENCE, rd.STRATEGY_RANDOM),
        proj, "wds", "Wet dog shake", pool_label="reviewed", n_pool=200,
        n_pos_pool=50, n_seed_pos=8, prevalence=0.1, sec_per_clip_review=4.0,
        effort_targets=(10,), display_budget=200)
    res.disabled_strategies = {rd.STRATEGY_ESSENCE: "raw pose unreadable."}
    out = tmp_path / "curve.png"
    rd.plot_discovery_curve(res, out)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Full-pool label coverage guard
# ---------------------------------------------------------------------------

def _ids(session: str, starts, animal: str = "M1") -> list[str]:
    return [f"seg_{animal}_session_{session}_{s}_{s + 14}" for s in starts]


def test_coverage_keys_sessions_not_animal_session_pairs():
    """A multi-animal project whose sessions ARE extracted must not read as missing.

    Keying on animal+session reported one real project as 8% covered when the
    true session coverage was 43%.
    """
    pool = np.array(_ids("aaaa", [0, 15, 30], "M1") + _ids("aaaa", [0, 15], "M2"))
    ts = pd.DataFrame({"segment_id": _ids("aaaa", [0, 15], "M3"), "label": ["x", "x"]})
    cov = rd._label_coverage(ts, pool, set())
    assert cov.session_frac == 1.0          # session aaaa is present
    assert cov.n_missing_sessions == 0


def test_coverage_refuses_when_sessions_were_never_extracted():
    """Missing sessions deflate prevalence without bound → refuse, don't emit."""
    pool = np.array(_ids("aaaa", [0, 15, 30]))
    ts = pd.DataFrame({"segment_id": _ids("aaaa", [0]) + _ids("bbbb", [0]) + _ids("cccc", [0]),
                       "label": ["x", "x", "x"]})
    cov = rd._label_coverage(ts, pool, set())
    assert cov.blocking_reason and "feature extraction" in cov.blocking_reason


def test_coverage_warns_on_off_grid_ids_but_still_runs():
    """Off-grid clip ids cap coverage but are benign — warn, don't refuse."""
    pool = np.array(_ids("aaaa", list(range(0, 300, 15))))
    on_grid = _ids("aaaa", [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255])
    off_grid = _ids("aaaa", [7, 22])                      # not on the stride grid
    ts = pd.DataFrame({"segment_id": on_grid + off_grid,
                       "label": ["x"] * (len(on_grid) + len(off_grid))})
    cov = rd._label_coverage(ts, pool, set())
    assert not cov.blocking_reason
    assert cov.warning and "off-grid" in cov.warning


# ---------------------------------------------------------------------------
# Pool provenance on figures
# ---------------------------------------------------------------------------

def _result_with_pool(tmp_path, pool_label: str, prevalence: float) -> rd.RareDiscoveryResult:
    return rd._assemble_result(
        _perfect_vs_random_arrays(), (rd.STRATEGY_ESSENCE, rd.STRATEGY_RANDOM),
        _project(tmp_path), "wds", "Wet dog shake", pool_label=pool_label, n_pool=200,
        n_pos_pool=50, n_seed_pos=20, prevalence=prevalence,
        sec_per_clip_review=4.0, effort_targets=(10,), display_budget=200)


def test_provenance_flags_the_reviewed_pool_as_enriched(tmp_path):
    """A reviewed-pool number must never be mistaken for deployment rarity."""
    res = _result_with_pool(tmp_path, "reviewed", 0.104)
    line = res.provenance()
    assert "ENRICHED" in line and "10.40%" in line

    full = _result_with_pool(tmp_path, "full", 0.00835)
    assert "deployment rarity" in full.provenance()
    assert "ENRICHED" not in full.provenance()


def test_provenance_and_coverage_reach_the_rendered_figures(tmp_path):
    """Provenance is only useful if it survives onto the PNG titles."""
    res = _result_with_pool(tmp_path, "full", 0.00835)
    res.coverage_note = "Ground truth = 350 confirmed positives (68% of labels joined)."
    assert "68% of labels" in res.provenance()
    for fn, name in ((rd.plot_discovery_curve, "curve.png"),
                     (rd.plot_effort_to_n, "effort.png")):
        out = tmp_path / name
        fn(res, out)
        assert out.exists() and out.stat().st_size > 0
