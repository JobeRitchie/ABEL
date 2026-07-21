"""Contrastive essence extraction + graded ranking.

Guards the behaviour that fixes Targeted Clip Mining's over-selection: essence
must be inferred from how exemplars *differ from a background*, must ignore
degenerate/constant features, must stay far tighter than the legacy min-max
ranging, and must stay robust from a 2-clip selection up to hundreds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services.clip_metrics_service import (
    ClipMetricsService,
    Criterion,
    EssenceScorer,
)


def _make_pool(n_pos=60, n_bg=3000, seed=0):
    """A pool where positives sit high on ``good``/``good2`` but *overlap* the
    background there (so a naive min-max range leaks), are indistinguishable on
    ``noise``, and where ``const`` / ``duration_sec`` are degenerate."""
    rng = np.random.default_rng(seed)
    pos = pd.DataFrame({
        "good": rng.normal(3.0, 1.2, n_pos),
        "good2": rng.normal(3.0, 1.2, n_pos),
        "noise": rng.normal(0, 1.0, n_pos),
        "const": np.full(n_pos, 3.0),
        "duration_sec": np.full(n_pos, 0.5),
    }, index=[f"pos_{i}" for i in range(n_pos)])
    bg = pd.DataFrame({
        "good": rng.normal(0, 1.2, n_bg),
        "good2": rng.normal(0, 1.2, n_bg),
        "noise": rng.normal(0, 1.0, n_bg),
        "const": np.full(n_bg, 3.0),
        "duration_sec": np.full(n_bg, 0.5),
    }, index=[f"bg_{i}" for i in range(n_bg)])
    return pos, bg


def _match_frac(df, crits):
    res = ClipMetricsService.mine(df, crits, match_all=True)
    return len(res.matched_ids) / len(df)


def test_contrastive_essence_is_discriminative_and_tight():
    pos, bg = _make_pool()
    full = pd.concat([pos, bg])
    crits = ClipMetricsService.extract_contrastive_essence(pos, bg, k=5)
    assert crits, "expected at least one criterion"
    chosen = {c.metric_id for c in crits}

    # Degenerate features must never be chosen.
    assert "duration_sec" not in chosen
    assert "const" not in chosen
    # A genuinely discriminative feature must be chosen; pure noise must not.
    assert "good" in chosen or "good2" in chosen
    assert "noise" not in chosen

    # Keeps most exemplars...
    res_pos = ClipMetricsService.mine(pos, crits, match_all=True)
    assert len(res_pos.matched_ids) >= 0.75 * len(pos)
    # ...while excluding almost all background.
    assert _match_frac(bg, crits) < 0.15


def test_contrastive_beats_legacy_minmax_breadth():
    pos, bg = _make_pool()
    full = pd.concat([pos, bg])
    contrast = ClipMetricsService.extract_contrastive_essence(pos, bg, k=5)
    # The legacy path (no background → tightest-spread features ranged by min-max)
    # balloons when exemplars overlap the pool; the contrastive box does not.
    legacy = ClipMetricsService.extract_similar_essence(pos, None, k=3)
    assert _match_frac(full, contrast) < _match_frac(full, legacy)


def test_similar_essence_routes_to_contrastive_when_background_present():
    pos, bg = _make_pool()
    full = pd.concat([pos, bg])
    with_bg = ClipMetricsService.extract_similar_essence(pos, full, k=5)
    no_bg = ClipMetricsService.extract_similar_essence(pos, None, k=5)
    # With a background the routed result is far tighter than the background-free one.
    assert _match_frac(full, with_bg) < _match_frac(full, no_bg)


@pytest.mark.parametrize("n_pos", [2, 3, 5, 25, 200])
def test_robust_across_exemplar_count(n_pos):
    pos, bg = _make_pool(n_pos=n_pos)
    crits = ClipMetricsService.extract_contrastive_essence(pos, bg, k=5)
    assert crits  # never returns empty on separable data
    # Tighter than matching everything, and keeps the majority of exemplars.
    assert _match_frac(bg, crits) < 0.25
    res_pos = ClipMetricsService.mine(pos, crits, match_all=True)
    assert len(res_pos.matched_ids) >= 0.75 * n_pos


def test_essence_scorer_ranks_positives_above_background():
    pos, bg = _make_pool()
    full = pd.concat([pos, bg])
    scorer = ClipMetricsService.build_essence_scorer(pos, bg)
    assert scorer is not None
    s = scorer.score(full)
    assert s.loc[pos.index].mean() > s.loc[bg.index].mean()
    # Top-of-ranking is dominated by true positives.
    top = s.sort_values(ascending=False).index[: len(pos)]
    assert sum(1 for i in top if i.startswith("pos_")) >= 0.6 * len(pos)


def test_scorer_fallback_when_few_exemplars():
    # Below the logistic threshold it must still build a (stateless) scorer.
    pos, bg = _make_pool(n_pos=3)
    scorer = ClipMetricsService.build_essence_scorer(pos, bg)
    assert scorer is not None and scorer._model is None
    s = scorer.score(pd.concat([pos, bg]))
    assert s.loc[pos.index].mean() > s.loc[bg.index].mean()


def test_mine_orders_matches_by_rank_scores():
    pos, bg = _make_pool()
    full = pd.concat([pos, bg])
    crits = ClipMetricsService.extract_contrastive_essence(pos, bg, k=5)
    scorer = ClipMetricsService.build_essence_scorer(pos, bg)
    rank = scorer.score(full)
    res = ClipMetricsService.mine(full, crits, match_all=True, rank_scores=rank)
    ids = res.matched_ids
    assert ids, "expected matches"
    scores = [res.scores[w] for w in ids]
    assert scores == sorted(scores, reverse=True)  # best-first
    # The carried score is the ranker's, not the flat AND 1.0.
    assert res.scores[ids[0]] == pytest.approx(float(rank.loc[ids[0]]))
