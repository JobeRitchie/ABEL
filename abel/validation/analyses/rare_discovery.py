"""Rare-behavior discovery efficiency — the "targeting" figure.

How much faster do ABEL's clip-hunting tools surface a *rare* behavior than the
two things an ethologist would otherwise do: sample random clips, or scan whole
videos?  The exemplar rare behavior is **wet dog shake** in the home-cage
project (279 confirmed positives among 5,041 reviewed clips ≈ 5% prevalence, and
far rarer in the full 43k-window pool).

We compare four acquisition strategies, each faithful to a shipped ABEL tool:

- **essence**   — the contrastive Essence Miner (:mod:`clip_metrics_service`),
  reproduced exactly as the shipped Clip Mining dialog runs it: build a definition
  from a handful of exemplars — BOTH a hard AND-box of distinguishing feature
  ranges (``extract_similar_essence``) AND a graded likeness ranker
  (``build_essence_scorer``) — surface criteria-matched clips first (best-first
  within the match), and **re-extract** the definition as newly confirmed
  positives join the exemplars.  (The AND-box is the half that makes essence sharp
  for a rare behaviour; ranking the whole pool by the continuous score alone
  collapses toward random.)
- **active_learning** — warm-start a model on the seeds, then reveal the highest
  predicted-probability clips and retrain (ABEL's candidate loop).
- **umap**      — lasso the region of the embedding densest in the exemplars
  (rank by distance to the exemplar centroid) and **re-lasso** — recompute that
  centroid — as newly confirmed positives join the exemplars.
- **random**    — uniform clip sampling (the null baseline; analytic hypergeometric
  expectation available in closed form).

All three ABEL arms are **iterated identically** for a fair comparison: each grows
its exemplar/label set in equal ``batch`` steps, and every clip a strategy surfaces
that the (simulated) human confirms as a positive is folded back into that
strategy's definition — the essence is re-extracted, the UMAP region re-lassoed,
the model retrained — before the next batch is ranked.  This is what a real user
does (nobody freezes their essence at the first eight exemplars while an AL model
keeps learning), so the only thing that differs between arms is the *acquisition
signal*, not whether the tool is allowed to improve as evidence accrues.

Whole-video scanning is added to the *effort* figure only, expressed in minutes
a human must watch, because it lives on a different axis than "clips reviewed".

The metric is a **discovery (yield) curve**: clips reviewed → cumulative
*confirmed* positives found.  ABEL's tools front-load positives (steep early
curve); random is a straight diagonal at slope = prevalence.

── Several projects at once ───────────────────────────────────────────────────
With more than one project selected the runner does the *cheap* thing first:
:func:`rank_behaviors_by_rarity` reads each project's dense bout detections (no
model fitting) and the whole discovery/effort-to-quality budget then goes to that
project's rarest behaviour before moving on — so N projects cost N hunts, not
N × behaviours.  Each project keeps its own full set of figures, and the
``plot_combined_*`` panels put them on shared axes: every project contributes its
own rarest behaviour as one paired observation of the same four arms, normalised
(% of positives found, fold-enrichment, × fewer clips than that project's own
random arm) so projects of very different sizes can be averaged honestly.

── The circularity guard (read before trusting any number) ────────────────────
The 279 positives were themselves *found* using essence + temporal review, so
testing essence on the very clips it helped find would be rigged.  Every run is
therefore **cross-validated**: the essence/AL/UMAP definition is built from a
random ``n_seed_pos`` subset of the positives, and discovery is scored ONLY on
the *held-out* positives, which the definition never saw.  What this supports:
"given a few confirmed examples, ABEL recovers the remaining confirmed examples
with N× less review effort than random."  What it does NOT support: absolute
recall of *all* true wet-dog-shakes — behaviours no tool ever surfaced are
invisible here (no dense ground truth exists).  That caveat is printed on the
figures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.services.clip_metrics_service import ClipMetricsService
from abel.utils import xgb_predict
from abel.validation import metrics as vmetrics
from abel.validation import prism as _prism
from abel.validation.analyses import al_curve
from abel.validation.prism import _replicate_block
from abel.validation.datamodel import CellResult, ProjectRef
from abel.validation.holdout import HoldoutSplit
from abel.validation.holdout import split as holdout_split

STRATEGY_ESSENCE = "essence"
STRATEGY_AL = "active_learning"
STRATEGY_UMAP = "umap"
STRATEGY_RANDOM = "random"
STRATEGY_WHOLE_VIDEO = "whole_video"

STRATEGY_LABELS = {
    STRATEGY_ESSENCE: "Essence Miner",
    STRATEGY_AL: "Active Learning",
    STRATEGY_UMAP: "UMAP selection",
    STRATEGY_RANDOM: "Random clips",
    STRATEGY_WHOLE_VIDEO: "Whole-video scan",
}
STRATEGY_COLORS = {
    STRATEGY_ESSENCE: "#4C72B0",
    STRATEGY_AL: "#55A868",
    STRATEGY_UMAP: "#8172B3",
    STRATEGY_RANDOM: "#BBBBBB",
    STRATEGY_WHOLE_VIDEO: "#C44E52",
}

# Non-feature columns of the training set — never fed to the UMAP embedding.
_META_COLS = frozenset({
    "segment_id", "start_frame", "end_frame", "animal_id", "session_id",
    "label", "label_source", "reviewer_confidence", "overlap_allowed",
    "uncertainty_entropy", "uncertainty_margin", "density_outlier_score",
    "uncertainty_score", "prediction_prob", "prediction_prob_fused",
})


# ── result model ───────────────────────────────────────────────────────────


@dataclass
class DiscoveryPoint:
    n_reviewed: int
    n_found_mean: float       # confirmed held-out positives discovered by here
    n_found_ci: float         # t-based 95% half-width across seeds
    n_seeds: int
    # The per-seed counts the mean/CI were computed from. Kept so the Prism export
    # can emit replicate subcolumns: a 95% CI half-width is not an error format
    # Prism accepts, so summarising here would make the curve unplottable with
    # error bars. Defaulted -> older pickled results still load.
    n_found_seeds: list[float] = field(default_factory=list)


@dataclass
class StrategyCurve:
    strategy: str
    points: list[DiscoveryPoint] = field(default_factory=list)
    # clips reviewed to reach N confirmed positives (mean across seeds); NaN if
    # the strategy never reached N within the pool on some seed.
    effort_to_n: dict[int, float] = field(default_factory=dict)
    effort_to_n_ci: dict[int, float] = field(default_factory=dict)
    # Per-seed effort values behind each mean — retained so Prism can run the test.
    effort_to_n_seeds: dict[int, list[float]] = field(default_factory=dict)

    def label(self) -> str:
        return STRATEGY_LABELS.get(self.strategy, self.strategy)

    def color(self) -> str:
        return STRATEGY_COLORS.get(self.strategy, "#333333")


@dataclass
class RareDiscoveryResult:
    project_id: str
    behavior_id: str
    behavior_name: str
    pool_label: str            # "reviewed" | "full"
    n_pool: int                # candidate clips in the hunt (excl. seeds)
    n_pos_pool: int            # confirmed positives available to discover
    n_seed_pos: int
    prevalence: float          # n_pos_pool / n_pool
    curves: dict[str, StrategyCurve] = field(default_factory=dict)
    cells: list[CellResult] = field(default_factory=list)
    # Whole-video reference (effort figure only).
    total_video_minutes: float = float("nan")
    sec_per_clip_review: float = 4.0
    # Strategies that were REQUESTED but could not run, → why.  A dropped arm is
    # invisible on the figure (it is simply an absent line), which reads as "we
    # tested it and it did nothing" rather than "we could not test it".  Carrying
    # the reason on the result lets the report and the figure say so out loud.
    disabled_strategies: dict[str, str] = field(default_factory=dict)
    # Label→window join coverage for a full-pool run (see _label_coverage).  Empty
    # on the reviewed pool, where every clip is labelled by construction.
    coverage_note: str = ""
    # Configured effort targets (build order) and the seed count every arm ran, so
    # the Prism exports emit an identical column block for every project instead
    # of one derived from whichever targets happened to be reached.
    effort_targets: list[int] = field(default_factory=list)
    n_seeds: int = 0

    def provenance(self) -> str:
        """One-line pool description stamped on every figure.

        Discovery numbers are only interpretable against the prevalence they were
        measured at: the reviewed pool runs ~12x enriched for the target, which
        deflates fold-enrichment by a similar factor.  A figure that does not say
        which pool it came from invites exactly that comparison, so every plot
        carries this line.
        """
        pool = ("full segment pool (deployment rarity)" if self.pool_label == "full"
                else "reviewed clips only — ENRICHED, not deployment rarity")
        line = (f"Pool: {pool} — {self.n_pool:,} candidates, "
                f"{self.n_pos_pool} confirmed positives, prevalence {self.prevalence:.2%}")
        return line + (f"\n{self.coverage_note}" if self.coverage_note else "")

    def enrichment_at(self, strategy: str, k: int) -> float:
        """Precision@k ÷ prevalence — fold-enrichment over random at budget k."""
        cur = self.curves.get(strategy)
        if cur is None or self.prevalence <= 0:
            return float("nan")
        pt = min((p for p in cur.points if p.n_reviewed >= k),
                 key=lambda p: p.n_reviewed, default=None)
        if pt is None or pt.n_reviewed <= 0:
            return float("nan")
        return (pt.n_found_mean / pt.n_reviewed) / self.prevalence


# ── seed / ranking helpers ─────────────────────────────────────────────────


def _pos_mask(df: pd.DataFrame, target: str) -> np.ndarray:
    return (df["label"].astype(str).str.strip() == str(target).strip()).to_numpy()


def _seed_positives(pos_idx: np.ndarray, n_seed_pos: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Pick the ``n_seed_pos`` exemplar positives shared by every strategy."""
    k = min(int(n_seed_pos), len(pos_idx))
    if k <= 0:
        return np.asarray([], dtype=int)
    return np.sort(rng.choice(pos_idx, size=k, replace=False))


def _discovered_curve(is_pos_in_rank_order: np.ndarray) -> np.ndarray:
    """Cumulative confirmed positives as clips are reviewed top-of-ranking down."""
    return np.cumsum(is_pos_in_rank_order.astype(float))


def _rank_random(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n)


def _rank_by_score(score: np.ndarray, descending: bool = True) -> np.ndarray:
    """Argsort with NaNs pushed to the back (they are reviewed last)."""
    s = np.asarray(score, dtype=float)
    fill = -np.inf if descending else np.inf
    s = np.where(np.isfinite(s), s, fill)
    order = np.argsort(-s if descending else s, kind="mergesort")
    return order


# Essence-Miner settings — mirror the shipped Clip Mining dialog's "Extract
# essence" defaults so the validation measures the tool the user actually uses:
# k distinguishing features in the AND-box, "Balanced" breadth (0.80 exemplar
# recall).  See :meth:`ClipMetricsService.extract_similar_essence`.
_ESSENCE_CRITERIA_K = 5
_ESSENCE_RECALL_TARGET = 0.80
# The shipped feature set is ~1100 columns; the greedy criteria search is
# O(features) per re-extraction round, so ranking every column every batch is
# prohibitively slow.  Pre-restrict to the best-separated handful (the essence
# only ever commits ~5–8 of them anyway) and contrast against a bounded
# background sample — both keep the result within noise of the full search while
# making the iterated hunt tractable.
_ESSENCE_MAX_FEATURES = 40
_ESSENCE_BG_SAMPLE = 1200

# Random clips revealed before active learning's first model, so that model is not
# trained on positives alone.  Measured: 10 is enough to make the fit non-degenerate,
# and every clip beyond that is pure cost charged to AL's discovery curve.
_AL_WARM_FILL = 10


def _criteria_match_mask(df: pd.DataFrame, crits: list) -> np.ndarray:
    """Boolean mask of rows passing every active essence criterion (the AND-box).

    Same semantics as :meth:`ClipMetricsService.mine` with ``match_all=True`` — a
    row must clear every enabled bound — reproduced here as a positional mask so
    the discovery order can put matches first without round-tripping through the
    string-indexed ``mine`` result.  Returns all-False when no criterion is active.
    """
    keep = np.ones(len(df), dtype=bool)
    any_active = False
    for c in crits or []:
        if not getattr(c, "enabled", True):
            continue
        mid = getattr(c, "metric_id", None)
        low, high = getattr(c, "low", None), getattr(c, "high", None)
        if mid is None or mid not in df.columns or (low is None and high is None):
            continue
        any_active = True
        col = pd.to_numeric(df[mid], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(col)
        if low is not None:
            ok &= col >= float(low)
        if high is not None:
            ok &= col <= float(high)
        keep &= ok
    return keep if any_active else np.zeros(len(df), dtype=bool)


def _essence_ranked_order(
    exemplar_df: pd.DataFrame,
    background_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> "np.ndarray | None":
    """Order ``cand_df`` rows the way the shipped Clip Mining dialog does.

    The dialog's "Extract essence" builds BOTH a hard AND-box of distinguishing
    feature ranges (:meth:`~ClipMetricsService.extract_similar_essence`) AND a
    graded likeness ranker (:meth:`~ClipMetricsService.build_essence_scorer`), then
    loads criteria-matched clips first, best-first within the match
    (:meth:`~ClipMetricsService.mine`).  The criteria filter is the half that makes
    essence sharp for a rare behaviour — ranking the whole pool by the continuous
    score alone (what this analysis used to do) discards it and collapses toward
    random.  Here we reproduce the dialog exactly: matched rows first (ranked by
    essence-likeness), then the rest (also by likeness), as one full-length
    permutation.  Returns positional order into ``cand_df``, or ``None`` when there
    is no separable essence at all (caller falls back to random).
    """
    # Bound the definition-building cost on the wide shipped feature set: contrast
    # against a capped background sample and keep only the best-separated features
    # (the candidate set stays full — it is what we rank).
    bg = background_df
    if len(bg) > _ESSENCE_BG_SAMPLE:
        rs = np.random.default_rng(0)
        bg = bg.iloc[np.sort(rs.choice(len(bg), _ESSENCE_BG_SAMPLE, replace=False))]
    feats = ClipMetricsService._usable_essence_metrics(exemplar_df, bg)
    if not feats:
        return None
    feats = feats[:_ESSENCE_MAX_FEATURES]
    ex_f, bg_f, cand_f = exemplar_df[feats], bg[feats], cand_df[feats]

    scorer = ClipMetricsService.build_essence_scorer(ex_f, bg_f, feature_ids=feats)
    if scorer is None:
        return None
    score = scorer.score(cand_f).to_numpy(dtype=float)
    crits = ClipMetricsService.extract_similar_essence(
        ex_f, bg_f, k=_ESSENCE_CRITERIA_K, recall_target=_ESSENCE_RECALL_TARGET)
    matched = _criteria_match_mask(cand_f, crits)
    s = np.where(np.isfinite(score), score, -np.inf)
    # Primary key: matched (0) before unmatched (1); secondary: higher score first.
    return np.lexsort((-s, ~matched))


def _rank_essence(metrics_pool: pd.DataFrame, metrics_seed: pd.DataFrame) -> np.ndarray:
    """Rank candidates by Essence-Miner likeness to the seed exemplars.

    Uses the *shipped* essence path (criteria AND-box + graded ranker) with the
    candidate pool itself as the background — the same signal the Clip Mining
    dialog computes when you hit "extract essence".
    """
    order = _essence_ranked_order(metrics_seed, metrics_pool, metrics_pool)
    if order is not None:
        return order
    scorer = ClipMetricsService.build_essence_scorer(metrics_seed, metrics_pool)
    if scorer is None:
        return None  # no separable signal — caller falls back to random
    score = scorer.score(metrics_pool).to_numpy(dtype=float)
    return _rank_by_score(score, descending=True)


def _embed(feat: np.ndarray, seed: int, n_pca: int = 50) -> np.ndarray:
    """2-D embedding of the standardized feature matrix (UMAP if available).

    High-dimensional pose+kinematic vectors (>1000 cols) make UMAP very slow, so
    the matrix is PCA-reduced to ``n_pca`` components first — the standard, near
    lossless speed-up — before the 2-D embedding.  Falls back to a plain PCA-2D
    when ``umap-learn`` is absent.
    """
    import warnings

    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    with warnings.catch_warnings():
        # Cosmetic: all-NaN columns the imputer drops, and UMAP's single-thread
        # note when a random_state is set for reproducibility.
        warnings.simplefilter("ignore")
        X = SimpleImputer(strategy="median").fit_transform(feat)
        X = StandardScaler().fit_transform(X)
        if X.shape[1] > n_pca:
            X = PCA(n_components=n_pca, random_state=int(seed)).fit_transform(X)
        try:  # faithful to ABEL's own UMAP view when the dependency is present
            import umap  # type: ignore

            return umap.UMAP(n_components=2, random_state=int(seed)).fit_transform(X)
        except Exception:
            k = max(2, min(2, X.shape[1]))
            return PCA(n_components=k, random_state=int(seed)).fit_transform(X)


def _rank_umap_fixed(emb: np.ndarray, cand_idx: np.ndarray,
                     seed_sel: np.ndarray) -> np.ndarray:
    """Rank candidates by distance to the exemplar centroid in a *fixed* embedding.

    The 2-D embedding is unsupervised (never sees labels), so it is computed once
    over the whole pool and reused across every CV fold — an N×-per-seed speed-up
    over re-embedding, and more stable.  ``cand_idx`` / ``seed_sel`` index into the
    shared embedding.
    """
    centroid = np.nanmean(emb[seed_sel], axis=0)
    dist = np.linalg.norm(emb[cand_idx] - centroid, axis=1)
    return _rank_by_score(dist, descending=False)


def _rank_umap(feat_pool: np.ndarray, seed_rows: np.ndarray, seed: int) -> np.ndarray:
    """Rank candidates by distance to the exemplar centroid (single-shot helper).

    Embeds the pool + exemplars together and orders candidates by proximity to the
    exemplars' centroid.  ``run_rare_discovery`` uses the faster fixed-embedding
    path (:func:`_rank_umap_fixed`); this stays for standalone use and tests.
    """
    n_pool = feat_pool.shape[0]
    emb = _embed(np.vstack([feat_pool, seed_rows]), seed)
    centroid = np.nanmean(emb[n_pool:], axis=0)
    dist = np.linalg.norm(emb[:n_pool] - centroid, axis=1)
    return _rank_by_score(dist, descending=False)


def _al_discovery(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior: str,
    pool: pd.DataFrame,
    throwaway_holdout: pd.DataFrame,
    seed_rows: pd.DataFrame,
    *,
    seed: int,
    batch: int,
    max_reveal: int,
    log: Callable[[str], None],
) -> np.ndarray:
    """Reveal-order over the candidate pool produced by the AL loop.

    Warm-starts on the seed exemplars (+ a random negative fill), then repeatedly
    trains via the *real* trainer (:func:`al_curve._fit`), scores the unrevealed
    pool, and reveals the highest predicted-probability batch — ABEL's candidate
    generation.  Returns the order in which candidate rows are revealed, so the
    discovery curve is ``cumsum`` of their true-positive flags.  The loop stops
    once ``max_reveal`` candidates have been revealed — a realistic review budget
    (nobody hand-reviews the entire pool to find a rare behaviour), and the arm
    that would otherwise dominate the run's cost.  The ``throwaway_holdout`` only
    satisfies the trainer's eval split; its metrics are discarded.
    """
    rng = np.random.default_rng(seed + 101)
    n = len(pool)
    cap = min(int(max_reveal), n)
    revealed: list[int] = []
    remaining = list(range(n))
    # Warm negative fill so the first model is non-degenerate.  This is a FIXED
    # count, deliberately decoupled from the seed size: the fill is charged to the
    # discovery curve, but essence/UMAP pay nothing equivalent, so scaling it with
    # ``n_seed_pos`` handicapped AL more and more as the seed grew.  At a 20-clip
    # seed the old ``3 * len(seed_rows)`` spent AL's first 60 reveals on random
    # picks, which alone reversed the AL-vs-essence ranking at small budgets.
    fill = min(_AL_WARM_FILL, len(remaining))
    rng.shuffle(remaining)
    revealed.extend(remaining[:fill])
    remaining = remaining[fill:]

    while remaining and len(revealed) < cap:
        # Labeled training frame = seed exemplars (the warm-start positives) +
        # every candidate revealed so far, each carrying its TRUE label (the human
        # confirmed it on review).
        rev_idx = np.asarray(revealed, dtype=int)
        sub = pd.concat([seed_rows, pool.iloc[rev_idx]], ignore_index=True)
        try:
            res = al_curve._fit(trainer, project, behavior, sub, throwaway_holdout, seed)
            rem_df = pool.iloc[remaining]
            probs = xgb_predict.predict_proba(
                res.calibrated_model, rem_df[res.feature_cols].to_numpy(dtype=float))
            ti = int(res.target_idx)
            p_tar = probs[:, ti] if ti < probs.shape[1] else probs.max(axis=1)
            order = np.argsort(-p_tar)
        except Exception as exc:  # degenerate early set → random reveal this step
            log(f"AL seed {seed}: fallback ({type(exc).__name__})")
            order = rng.permutation(len(remaining))
        take = min(int(batch), len(remaining))
        chosen = [remaining[k] for k in order[:take]]
        revealed.extend(chosen)
        chosen_set = set(chosen)
        remaining = [i for i in remaining if i not in chosen_set]

    return np.asarray(revealed[:cap], dtype=int)


def _essence_discovery(
    metrics_pool: pd.DataFrame,
    is_pos: np.ndarray,
    seed_metrics: pd.DataFrame,
    *,
    seed: int,
    batch: int,
    refit_budget: int,
    log: Callable[[str], None],
) -> np.ndarray | None:
    """Iterative Essence-Miner reveal order — re-extracts as positives are confirmed.

    Mirrors what a real user does (and puts essence on equal footing with AL): build
    an essence from the seed exemplars against the pool background, review the top
    ``batch`` clips, fold any *confirmed* positives into the exemplar set, RE-EXTRACT,
    and re-rank the unreviewed pool.  Re-extraction runs while fewer than
    ``refit_budget`` clips have been reviewed — a realistic hunt budget and the
    window where the arms separate; the leftover pool is then appended in the *final*
    essence's order so the return stays a full-length permutation (needed so
    effort-to-N can reach deep targets).  ``metrics_pool``/``is_pos`` are aligned to
    the candidate set; ``seed_metrics`` are the held-out-excluded seed exemplars.
    Returns ``None`` if no separable essence exists at all (caller falls back to
    random).
    """
    n = len(metrics_pool)
    cap = min(int(refit_budget), n)
    revealed: list[int] = []
    remaining = list(range(n))
    exemplar_rows = [seed_metrics]          # grows with each confirmed positive
    any_scored = False

    def _order_remaining() -> "np.ndarray | None":
        # Faithful to the shipped dialog: criteria AND-box + graded ranker, matched
        # clips first.  Background is the full candidate pool (as in the dialog),
        # exemplars are the seed + every confirmed positive folded in so far.
        seed_df = pd.concat(exemplar_rows, ignore_index=True)
        rem = np.asarray(remaining, dtype=int)
        order_local = _essence_ranked_order(
            seed_df, metrics_pool, metrics_pool.iloc[rem])
        if order_local is None:
            return None
        return rem[order_local]                               # global candidate idx

    while remaining and len(revealed) < cap:
        ordered = _order_remaining()
        if ordered is None:
            break
        any_scored = True
        take = min(int(batch), len(remaining))
        chosen = [int(i) for i in ordered[:take]]
        newly_pos = [c for c in chosen if is_pos[c]]
        if newly_pos:
            exemplar_rows.append(metrics_pool.iloc[newly_pos])
        revealed.extend(chosen)
        chosen_set = set(chosen)
        remaining = [i for i in remaining if i not in chosen_set]

    if not any_scored:
        return None
    if remaining:  # append the tail in the final essence's order (user keeps hunting)
        tail = _order_remaining()
        revealed.extend(int(i) for i in (tail if tail is not None
                                         else np.asarray(remaining, dtype=int)))
    log(f"essence seed {seed}: revealed {min(cap, len(revealed))} with re-extraction")
    return np.asarray(revealed, dtype=int)


def _umap_discovery(
    emb: np.ndarray,
    is_pos: np.ndarray,
    seed_emb: np.ndarray,
    *,
    seed: int,
    batch: int,
    refit_budget: int,
) -> np.ndarray:
    """Iterative UMAP-selection reveal order — re-lassos as positives are confirmed.

    The unsupervised 2-D embedding is a property of the *data*, not the labels, so it
    is fixed (computed once); what a user actually updates each round is the selected
    region.  Here that region is the exemplar centroid, which drifts toward every
    newly confirmed positive, tightening the ranked neighbourhood around real
    instances round by round — the UMAP analogue of AL's retrain.  ``emb`` are the
    candidate embedding rows; ``seed_emb`` the seed exemplars' rows.  Same budget /
    full-permutation-tail contract as :func:`_essence_discovery`.
    """
    n = len(emb)
    cap = min(int(refit_budget), n)
    revealed: list[int] = []
    remaining = list(range(n))
    exemplar_pts = [np.asarray(seed_emb, dtype=float)]

    def _order_remaining() -> np.ndarray:
        centroid = np.nanmean(np.vstack(exemplar_pts), axis=0)
        rem = np.asarray(remaining, dtype=int)
        dist = np.linalg.norm(emb[rem] - centroid, axis=1)
        return rem[np.argsort(dist)]                          # nearest first

    while remaining and len(revealed) < cap:
        ordered = _order_remaining()
        take = min(int(batch), len(remaining))
        chosen = [int(i) for i in ordered[:take]]
        newly_pos = [c for c in chosen if is_pos[c]]
        if newly_pos:
            exemplar_pts.append(emb[newly_pos])
        revealed.extend(chosen)
        chosen_set = set(chosen)
        remaining = [i for i in remaining if i not in chosen_set]

    if remaining:
        revealed.extend(int(i) for i in _order_remaining())
    return np.asarray(revealed, dtype=int)


# ── orchestration (labeled reviewed pool) ──────────────────────────────────


def _essence_feature_frame(pool: pd.DataFrame, min_finite_frac: float = 0.5) -> pd.DataFrame:
    """The shipped per-window feature vectors, as the Essence Miner's substrate.

    The essence arm scores exemplar-likeness over the SAME features the shipped
    classifier is trained on — pose kinematics, the oscillation / periodicity /
    angular-velocity family, context and video features — read straight from the
    training pool, NOT the separate 22-metric :class:`ClipMetricsService` pose
    summaries the Clip Mining dialog recomputes.  This is the whole point of the
    validation: measure the tool on the app's real feature extraction.  Two
    consequences follow: (1) essence is compared to the AL and UMAP arms on an
    equal feature footing instead of being handicapped to a poorer set — decisive
    for a behaviour like wet-dog-shake whose signal lives in oscillation/angular
    features absent from the 22 metrics; and (2) the features are precomputed, so
    essence no longer needs the raw pose drive mounted (the old silent-degradation
    failure mode is gone).  Columns too sparse to anchor a criterion bound (finite
    in < ``min_finite_frac`` of rows) or constant are dropped.  Indexed by
    segment_id, matching the other essence helpers.
    """
    cols = [c for c in pool.columns
            if c not in _META_COLS
            and pd.api.types.is_numeric_dtype(pool[c])
            and float(pool[c].notna().mean()) >= min_finite_frac
            and float(pool[c].std(skipna=True) or 0.0) > 0.0]
    frame = pool[cols].copy()
    frame.index = pool["segment_id"].astype(str).to_numpy()
    return frame


@dataclass
class RareProjectCache:
    """Behaviour-independent work shared across a project's rare-discovery runs.

    The essence feature frame (shipped per-window features) and the UMAP embedding
    depend only on the *pool*, not the target behaviour, so computing them once and
    reusing them across every behaviour is **bit-identical** to recomputing — it
    just skips the redundant passes.  Rows align positionally with the project's
    ``holdout_split.train_pool`` (reset index).
    """

    metrics: pd.DataFrame | None = None      # essence feature frame (shipped features)
    embedding: np.ndarray | None = None      # 2-D UMAP embedding of the pool
    pose_reason: str | None = None           # retained for API compat; now always None


def prepare_project_cache(
    project: ProjectRef,
    holdout_split: HoldoutSplit,
    *,
    need_metrics: bool = True,
    need_embedding: bool = True,
    metrics_max_workers: int | None = 2,
    progress_cb: Callable[[str], None] | None = None,
) -> RareProjectCache:
    """Precompute the pool's essence feature frame + UMAP embedding for a project."""
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    pool = holdout_split.train_pool.reset_index(drop=True)
    cache = RareProjectCache()
    if need_metrics:
        # Essence runs on the shipped features (precomputed in the pool), so this
        # is a cheap column selection, not a pose recompute — see
        # :func:`_essence_feature_frame`.
        _log(f"{project.name}: gathering shipped features for essence "
             f"({len(pool)} clips, shared)…")
        cache.metrics = _essence_feature_frame(pool)
    if need_embedding:
        feat_cols = [c for c in pool.columns
                     if c not in _META_COLS
                     and pd.api.types.is_numeric_dtype(pool[c])
                     and bool(pool[c].notna().any())]
        if feat_cols:
            _log(f"{project.name}: embedding {len(pool)} clips for UMAP (shared)…")
            cache.embedding = _embed(pool[feat_cols].to_numpy(dtype=float), 9000)
    return cache


def run_rare_discovery(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_id: str,
    holdout_split: HoldoutSplit,
    *,
    strategies: tuple[str, ...] = (
        STRATEGY_ESSENCE, STRATEGY_AL, STRATEGY_UMAP, STRATEGY_RANDOM),
    n_seeds: int = 3,
    n_seed_pos: int = 20,
    batch: int = 25,
    al_max_budget: int = 500,
    display_budget: int | None = None,
    effort_targets: tuple[int, ...] = (10, 25, 50),
    sec_per_clip_review: float = 4.0,
    metrics_max_workers: int | None = 2,
    cache: RareProjectCache | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> RareDiscoveryResult:
    """Cross-validated rare-behavior discovery efficiency on the reviewed pool.

    The candidate hunt runs over ``holdout_split.train_pool`` (the held-out
    sessions are reserved only as the AL trainer's throwaway eval split).  Every
    strategy shares the *same* seed exemplars per seed, so the comparison
    isolates the acquisition rule.  All three ABEL arms (essence, UMAP, AL) are
    iterated identically: they grow their exemplar/label set in ``batch``-sized
    steps, refold every confirmed positive into their definition, and re-rank —
    for up to ``al_max_budget`` reviewed clips — so no arm is frozen at its seed
    definition while another keeps learning.  ``cache`` supplies the pool's essence
    feature frame + UMAP embedding (label-independent, so computed once and reused
    across every fold and refit; only the essence/centroid recompute per round) —
    see :func:`prepare_project_cache`.  Essence scores over the SAME shipped features
    as the AL/UMAP arms (:func:`_essence_feature_frame`), so no arm is handicapped to
    a poorer feature set and none depends on the raw pose drive.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    behavior_name = project.behavior_label(behavior_id)
    pool_all = holdout_split.train_pool.reset_index(drop=True)
    throwaway = holdout_split.holdout.reset_index(drop=True)
    strategies = list(strategies)
    disabled: dict[str, str] = {}

    pos_idx_all = np.where(_pos_mask(pool_all, behavior_id))[0]
    n_pos_all = len(pos_idx_all)
    if n_pos_all <= n_seed_pos + 1:
        raise ValueError(
            f"Too few '{behavior_name}' positives ({n_pos_all}) in the pool to "
            f"cross-validate with n_seed_pos={n_seed_pos}.")

    # Essence feature frame (the shipped per-window features; once for the whole
    # pool, reused across every seed).  No pose recompute and no drive dependency —
    # essence now runs on the same features as the AL/UMAP arms.
    metrics_all: pd.DataFrame | None = None
    if STRATEGY_ESSENCE in strategies:
        if cache is not None and cache.metrics is not None:
            metrics_all = cache.metrics
        else:
            metrics_all = _essence_feature_frame(pool_all)
        if metrics_all is None or metrics_all.shape[1] == 0:
            reason = "no usable shipped features in the training pool."
            _log(f"{behavior_name}: ESSENCE ARM DISABLED — {reason}")
            disabled[STRATEGY_ESSENCE] = reason
            strategies = [s for s in strategies if s != STRATEGY_ESSENCE]
            metrics_all = None

    if not strategies:
        raise ValueError(
            "No runnable discovery strategies (essence was disabled and nothing "
            "else was selected). Select AL/UMAP/random.")

    # UMAP feature matrix (once).
    feat_cols: list[str] = []
    feat_all: np.ndarray | None = None
    umap_emb: np.ndarray | None = None
    if STRATEGY_UMAP in strategies:
        if cache is not None and cache.embedding is not None:
            umap_emb = cache.embedding
        else:
            # Drop all-NaN feature columns (dead features that never populate for
            # this project) — feeding them to the imputer is wasteful and warns.
            feat_cols = [c for c in pool_all.columns
                         if c not in _META_COLS
                         and pd.api.types.is_numeric_dtype(pool_all[c])
                         and bool(pool_all[c].notna().any())]
            # Embed the whole pool ONCE (unsupervised → no leakage) and reuse
            # across every CV fold, instead of a fresh UMAP per seed (the cost).
            _log(f"{behavior_name}: embedding {len(pool_all)} clips (UMAP)…")
            umap_emb = _embed(pool_all[feat_cols].to_numpy(dtype=float), 9000)

    # Per-strategy, per-seed discovered arrays (aligned to a common length).
    per_seed: dict[str, list[np.ndarray]] = {s: [] for s in strategies}

    for rep in range(n_seeds):
        seed = 9000 + rep
        rng = np.random.default_rng(seed)
        seed_sel = _seed_positives(pos_idx_all, n_seed_pos, rng)
        seed_set = set(int(i) for i in seed_sel)
        cand_idx = np.asarray([i for i in range(len(pool_all)) if i not in seed_set],
                              dtype=int)
        cand = pool_all.iloc[cand_idx].reset_index(drop=True)
        is_pos = _pos_mask(cand, behavior_id)
        _log(f"{behavior_name}: seed {rep + 1}/{n_seeds} — "
             f"{len(cand)} candidates, {int(is_pos.sum())} held-out positives")

        for strat in strategies:
            if strat == STRATEGY_RANDOM:
                order = _rank_random(len(cand), rng)
            elif strat == STRATEGY_ESSENCE:
                # Iterated like AL: re-extract the essence as positives are confirmed.
                order = _essence_discovery(
                    metrics_all.iloc[cand_idx], is_pos, metrics_all.iloc[seed_sel],
                    seed=seed, batch=batch, refit_budget=al_max_budget, log=_log)
                if order is None:
                    order = _rank_random(len(cand), rng)
            elif strat == STRATEGY_UMAP:
                # Iterated like AL: re-lasso (recompute the exemplar centroid) as
                # positives are confirmed, in the fixed unsupervised embedding.
                order = _umap_discovery(
                    umap_emb[cand_idx], is_pos, umap_emb[seed_sel],
                    seed=seed, batch=batch, refit_budget=al_max_budget)
            elif strat == STRATEGY_AL:
                order = _al_discovery(
                    trainer, project, behavior_id, cand, throwaway,
                    pool_all.iloc[seed_sel], seed=seed, batch=batch,
                    max_reveal=al_max_budget, log=_log)
            else:
                continue
            disc = _discovered_curve(is_pos[order])
            per_seed[strat].append(disc)

    prevalence = float(n_pos_all - n_seed_pos) / max(1, len(pool_all) - n_seed_pos)
    result = _assemble_result(
        per_seed, strategies, project, behavior_id, behavior_name,
        pool_label="reviewed", n_pool=len(pool_all) - n_seed_pos,
        n_pos_pool=n_pos_all - n_seed_pos, n_seed_pos=n_seed_pos,
        prevalence=prevalence, sec_per_clip_review=sec_per_clip_review,
        effort_targets=effort_targets,
        display_budget=display_budget or al_max_budget)
    result.disabled_strategies = disabled
    return result


def _assemble_result(
    per_seed: dict[str, list[np.ndarray]],
    strategies: tuple[str, ...],
    project: ProjectRef,
    behavior_id: str,
    behavior_name: str,
    *,
    pool_label: str,
    n_pool: int,
    n_pos_pool: int,
    n_seed_pos: int,
    prevalence: float,
    sec_per_clip_review: float,
    effort_targets: tuple[int, ...],
    display_budget: int,
) -> RareDiscoveryResult:
    """Turn per-strategy per-seed discovered arrays into a plotted/exported result.

    The discovery curve zooms to ``display_budget`` (where the arms separate);
    effort-to-N is measured on the FULL-length arrays so a slow arm (random) can
    still reach a high target deep in the pool.
    """
    n_cand = min(min(len(a) for a in arrs) for arrs in per_seed.values() if arrs)
    grid = _budget_grid(min(int(display_budget), n_cand))

    result = RareDiscoveryResult(
        project_id=project.project_id, behavior_id=str(behavior_id),
        behavior_name=behavior_name, pool_label=pool_label,
        n_pool=int(n_pool), n_pos_pool=int(n_pos_pool),
        n_seed_pos=int(n_seed_pos), prevalence=float(prevalence),
        sec_per_clip_review=float(sec_per_clip_review),
        effort_targets=[int(t) for t in effort_targets],
        n_seeds=max((len(a) for a in per_seed.values() if a), default=0),
    )
    for strat in strategies:
        full = per_seed.get(strat) or []
        if not full:
            continue
        cur = StrategyCurve(strategy=strat)
        for k in grid:
            vals = [a[k - 1] for a in full if len(a) >= k]
            if not vals:
                continue
            cur.points.append(DiscoveryPoint(
                n_reviewed=int(k),
                n_found_mean=float(np.mean(vals)),
                n_found_ci=vmetrics.ci95(np.asarray(vals, dtype=float)),
                n_seeds=len(vals),
                n_found_seeds=[float(v) for v in vals]))
        for target in effort_targets:
            # One entry per seed, NaN where that seed never reached N -- see the
            # matching comment in _assemble_quality.
            raw = [_first_reach(a, target) for a in full]
            efforts = [float("nan") if e is None else float(e) for e in raw]
            reached = [e for e in efforts if np.isfinite(e)]
            cur.effort_to_n_seeds[int(target)] = efforts
            if reached:
                cur.effort_to_n[int(target)] = float(np.mean(reached))
                cur.effort_to_n_ci[int(target)] = vmetrics.ci95(
                    np.asarray(reached, dtype=float))
        result.curves[strat] = cur
        for pt in cur.points:
            result.cells.append(CellResult(
                project_id=project.project_id, project_name=project.name,
                behavior_id=str(behavior_id), behavior_name=behavior_name,
                analysis="rare_discovery", config_name=f"{pool_label}:{strat}",
                n_clips=pt.n_reviewed, seed=-1,
                n_pos_train=int(round(pt.n_found_mean)),
                n_neg_train=pt.n_reviewed - int(round(pt.n_found_mean))))

    result.total_video_minutes = whole_video_minutes(project)
    return result


# ── effort-to-a-good-MODEL (the "was the labeling worth it?" figure) ────────
# Discovery answers "how fast do I *find* positives"; this answers the question a
# user actually optimises: "how much labeling effort until the trained model is
# good — high F1 / PR-AUC on held-out data?"  Each strategy grows a training set
# from the SAME warm seed in equal batches, differing only in which clips it
# surfaces to label next; at every checkpoint a model is trained on the
# labels-so-far and scored on the fixed high-confidence holdout.  All ABEL arms
# iterate identically (essence re-extracted, UMAP re-lassoed, AL retrained), so
# the axis is real human effort and the win is reaching the target with fewer clips.


@dataclass
class QualityCurve:
    strategy: str
    points: list = field(default_factory=list)          # list[al_curve.ALPoint]
    # clips reviewed to reach a metric target (mean across seeds), keyed by a
    # human-readable target label (e.g. "F1≥0.80", "PR-AUC≥90%max").
    effort: dict[str, float] = field(default_factory=dict)
    effort_ci: dict[str, float] = field(default_factory=dict)
    effort_seeds: dict[str, list[float]] = field(default_factory=dict)

    def label(self) -> str:
        return STRATEGY_LABELS.get(self.strategy, self.strategy)

    def color(self) -> str:
        return STRATEGY_COLORS.get(self.strategy, "#333333")


@dataclass
class EffortToQualityResult:
    project_id: str
    behavior_id: str
    behavior_name: str
    pool_label: str
    k0: int                       # shared warm-seed size (clips) every arm starts from
    seed_pos: int                 # positives guaranteed in the warm seed
    sec_per_clip_review: float = 4.0
    best_f1: float = float("nan")
    best_pr_auc: float = float("nan")
    curves: dict[str, QualityCurve] = field(default_factory=dict)
    cells: list[CellResult] = field(default_factory=list)
    # Every CONFIGURED target label, in build order, and the seed count every arm
    # ran. Exports must key off these rather than off the labels that happen to
    # appear in `curves`: a target no seed reached is absent from effort_seeds, so
    # deriving the column set from observed keys gave each project a different set
    # of columns -- pasting two projects into one Prism table then shifted the
    # groups silently.
    target_labels: list[str] = field(default_factory=list)
    n_seeds: int = 0

    def clips_to(self, strategy: str, target_label: str) -> float | None:
        cur = self.curves.get(strategy)
        v = cur.effort.get(target_label) if cur else None
        return None if v is None else float(v)

    def savings_vs_random(self, strategy: str, target_label: str) -> float | None:
        """Random clips ÷ strategy clips to hit the target (× less labeling)."""
        s = self.clips_to(strategy, target_label)
        r = self.clips_to(STRATEGY_RANDOM, target_label)
        if not s or not r or s <= 0:
            return None
        return float(r / s)


def _pool_signals(
    project: ProjectRef,
    pool: pd.DataFrame,
    strategies: list[str],
    *,
    cache: "RareProjectCache | None",
    metrics_max_workers: int | None,
    log: Callable[[str], None],
) -> tuple["pd.DataFrame | None", "np.ndarray | None", list[str]]:
    """Resolve the essence feature frame + UMAP embedding the given strategies need.

    Both are shipped-feature-derived (label-independent), so a ``cache`` from
    :func:`prepare_project_cache` is reused verbatim.  Essence runs on the same
    shipped features as UMAP/AL (:func:`_essence_feature_frame`) — no pose recompute
    — and is dropped only if the pool has no usable feature columns at all.  Rows
    align positionally with ``pool``.
    """
    strategies = list(strategies)
    metrics_all: pd.DataFrame | None = None
    if STRATEGY_ESSENCE in strategies:
        if cache is not None and cache.metrics is not None:
            metrics_all = cache.metrics
        else:
            metrics_all = _essence_feature_frame(pool)
        if metrics_all is None or metrics_all.shape[1] == 0:
            log("ESSENCE ARM DISABLED — no usable shipped features in the pool.")
            strategies = [s for s in strategies if s != STRATEGY_ESSENCE]
            metrics_all = None
    umap_emb: np.ndarray | None = None
    if STRATEGY_UMAP in strategies:
        if cache is not None and cache.embedding is not None:
            umap_emb = cache.embedding
        else:
            feat_cols = [c for c in pool.columns
                         if c not in _META_COLS
                         and pd.api.types.is_numeric_dtype(pool[c])
                         and bool(pool[c].notna().any())]
            if feat_cols:
                log(f"embedding {len(pool)} clips (UMAP)…")
                umap_emb = _embed(pool[feat_cols].to_numpy(dtype=float), 9000)
        if umap_emb is None:
            strategies = [s for s in strategies if s != STRATEGY_UMAP]
    return metrics_all, umap_emb, strategies


def _acquire_next(
    strategy: str,
    remaining: list[int],
    pool: pd.DataFrame,
    labeled_pos: list[int],
    res,
    *,
    metrics_all: "pd.DataFrame | None",
    umap_emb: "np.ndarray | None",
    rng: np.random.Generator,
) -> np.ndarray:
    """Best-first order over ``remaining`` (positions into that list) for one batch.

    essence → re-extract from the *currently labeled* positives (background = pool),
    rank by likeness; umap → distance to the labeled-positive centroid; al → highest
    predicted target probability from the just-trained model; random / any fallback
    → shuffle.  Re-derives from the labels acquired so far, so every ABEL arm gets
    the same compounding AL enjoys.
    """
    if strategy == STRATEGY_ESSENCE and metrics_all is not None and labeled_pos:
        # Shipped essence path (criteria AND-box + ranker), matched clips first —
        # same as discovery, so both effort axes measure the tool as it ships.
        order = _essence_ranked_order(
            metrics_all.iloc[labeled_pos], metrics_all, metrics_all.iloc[remaining])
        if order is not None:
            return order
    elif strategy == STRATEGY_UMAP and umap_emb is not None and labeled_pos:
        centroid = np.nanmean(umap_emb[labeled_pos], axis=0)
        dist = np.linalg.norm(umap_emb[remaining] - centroid, axis=1)
        return np.argsort(dist)
    elif strategy == STRATEGY_AL and res is not None:
        try:
            rem_df = pool.iloc[remaining]
            probs = xgb_predict.predict_proba(
                res.calibrated_model, rem_df[res.feature_cols].to_numpy(dtype=float))
            ti = int(res.target_idx)
            p_tar = probs[:, ti] if ti < probs.shape[1] else probs.max(axis=1)
            return np.argsort(-p_tar)
        except Exception:
            pass
    return rng.permutation(len(remaining))


def _target_metrics(res, holdout: pd.DataFrame, behavior: str) -> tuple[float, float]:
    """Held-out (F1, PR-AUC) for the *target class only*.

    ``res.metrics["f1"]`` is macro-averaged over every class
    (``active_learning_trainer_service.py``), which is the right summary for the
    product but the wrong one here: on a rare behaviour the negative class is
    ~93 % of the holdout and trivially easy, so macro-F1 reads ~0.48 for a model
    that never once predicts the target.  Rare-behaviour quality has to be scored
    against the target itself, so we recompute from the probabilities rather than
    changing the trainer, whose metric other callers depend on.
    """
    ti = int(res.target_idx)
    probs = xgb_predict.predict_proba(
        res.calibrated_model, holdout[res.feature_cols].to_numpy(dtype=float))
    if ti >= probs.shape[1]:
        return float("nan"), float("nan")
    truth = (holdout["label"].astype(str).str.strip() == str(behavior).strip()).to_numpy()
    p_tar = probs[:, ti]
    pred = probs.argmax(axis=1) == ti
    tp = float(np.sum(pred & truth))
    denom = float(np.sum(pred) + np.sum(truth))
    f1 = (2.0 * tp / denom) if denom > 0 else float("nan")
    if not truth.any() or not np.isfinite(p_tar).all():
        return f1, float("nan")
    return f1, float(average_precision_score(truth, p_tar))


def _run_quality_strategy(
    trainer, project, behavior, pool, holdout, *,
    strategy: str, seed: int, k0: int, seed_pos: int, batch: int, max_budget: int,
    metrics_all, umap_emb, log: Callable[[str], None],
) -> list[tuple[int, int, float, float]]:
    """Train-eval trajectory for one strategy+seed: (n_clips, n_pos, f1, pr_auc)."""
    rng = np.random.default_rng(seed)
    pool = pool.reset_index(drop=True)
    n = len(pool)
    cap = min(int(max_budget), n)
    behavior_name = project.behavior_label(behavior)
    pos_arr = (pool["label"].astype(str).str.strip() == str(behavior).strip()).to_numpy()
    labeled = al_curve._seed_set(pool, behavior, k0, seed_pos, rng)
    traj: list[tuple[int, int, float, float]] = []

    while True:
        idx = sorted(labeled)
        sub = pool.iloc[idx]
        n_pos = int(pos_arr[idx].sum())
        f1 = pr = float("nan")
        res = None
        try:
            res = al_curve._fit(trainer, project, behavior, sub, holdout, seed)
            f1, pr = _target_metrics(res, holdout, behavior)
        except Exception as exc:  # noqa: BLE001 — tiny early sets can be degenerate
            log(f"{behavior_name}: {strategy} seed {seed} fit failed "
                f"({type(exc).__name__}) at {len(idx)} clips")
        traj.append((len(idx), n_pos, f1, pr))
        log(f"{behavior_name}: {strategy} seed {seed} — {len(idx)} clips "
            f"({n_pos} pos) F1={f1:.3f} PR-AUC={pr:.3f}")

        if len(labeled) >= cap:
            break
        remaining = [i for i in range(n) if i not in labeled]
        if not remaining:
            break
        n_choose = min(batch, len(remaining), cap - len(labeled))
        labeled_pos = [i for i in labeled if pos_arr[i]]
        order = _acquire_next(
            strategy, remaining, pool, labeled_pos, res,
            metrics_all=metrics_all, umap_emb=umap_emb, rng=rng)
        labeled.update(int(remaining[k]) for k in order[:n_choose])

    return traj


def _effort_to_threshold(traj, metric_pos: int, thr: float) -> int | None:
    """Smallest n_clips in one seed's trajectory whose metric first reaches ``thr``."""
    for row in sorted(traj, key=lambda r: r[0]):
        v = row[metric_pos]
        if np.isfinite(v) and v >= thr:
            return int(row[0])
    return None


def _assemble_quality(
    per_seed: dict[str, list[list[tuple[int, int, float, float]]]],
    strategies: tuple[str, ...],
    project: ProjectRef,
    behavior_id: str,
    behavior_name: str,
    *,
    pool_label: str,
    k0: int,
    seed_pos: int,
    f1_targets: tuple[float, ...],
    pr_auc_targets: tuple[float, ...],
    frac_targets: tuple[float, ...],
    sec_per_clip_review: float,
) -> EffortToQualityResult:
    """Per-seed train/eval trajectories → quality curves + clips-to-target efforts."""
    agg: dict[str, tuple[list, list]] = {}
    best_f1 = best_pr = float("-inf")
    for strat in strategies:
        trajs = per_seed.get(strat) or []
        pts = al_curve._aggregate(trajs)
        agg[strat] = (trajs, pts)
        for p in pts:
            if np.isfinite(p.f1_mean):
                best_f1 = max(best_f1, p.f1_mean)
            if np.isfinite(p.pr_auc_mean):
                best_pr = max(best_pr, p.pr_auc_mean)
    best_f1 = best_f1 if np.isfinite(best_f1) else float("nan")
    best_pr = best_pr if np.isfinite(best_pr) else float("nan")

    # (label, metric_pos, threshold): absolute targets + fraction-of-best targets.
    targets: list[tuple[str, int, float]] = []
    for t in f1_targets:
        targets.append((f"F1≥{t:.2f}", 2, float(t)))
    for t in pr_auc_targets:
        targets.append((f"PR-AUC≥{t:.2f}", 3, float(t)))
    for fr in frac_targets:
        pct = int(round(fr * 100))
        if np.isfinite(best_f1):
            targets.append((f"F1≥{pct}%max", 2, fr * best_f1))
        if np.isfinite(best_pr):
            targets.append((f"PR-AUC≥{pct}%max", 3, fr * best_pr))

    result = EffortToQualityResult(
        project_id=project.project_id, behavior_id=str(behavior_id),
        behavior_name=behavior_name, pool_label=pool_label, k0=int(k0),
        seed_pos=int(seed_pos), sec_per_clip_review=float(sec_per_clip_review),
        best_f1=float(best_f1), best_pr_auc=float(best_pr),
        target_labels=[lab for lab, _, _ in targets],
        n_seeds=max((len(per_seed.get(s) or []) for s in strategies), default=0))
    for strat in strategies:
        trajs, pts = agg[strat]
        cur = QualityCurve(strategy=strat, points=pts)
        for (label, mpos, thr) in targets:
            # Keep one entry PER SEED, NaN where that seed never reached the
            # target. Compacting the list (the old behaviour) both lost which
            # seed a value came from -- so replicate columns could not line up
            # across targets -- and hid the censoring: a mean over "the 2 of 5
            # seeds that got there" reads as a fast arm, when it is a rare one.
            raw = [_effort_to_threshold(t, mpos, thr) for t in trajs]
            efforts = [float("nan") if e is None else float(e) for e in raw]
            reached = [e for e in efforts if np.isfinite(e)]
            cur.effort_seeds[label] = efforts
            if reached:
                cur.effort[label] = float(np.mean(reached))
                cur.effort_ci[label] = vmetrics.ci95(np.asarray(reached, dtype=float))
        result.curves[strat] = cur
        for pt in pts:
            result.cells.append(CellResult(
                project_id=project.project_id, project_name=project.name,
                behavior_id=str(behavior_id), behavior_name=behavior_name,
                analysis="effort_to_quality", config_name=f"{pool_label}:{strat}",
                n_clips=pt.n_clips, seed=-1, f1=pt.f1_mean, pr_auc=pt.pr_auc_mean,
                n_pos_train=int(round(pt.n_pos_mean)),
                n_neg_train=pt.n_clips - int(round(pt.n_pos_mean))))
    return result


def run_effort_to_quality(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_id: str,
    holdout_split: HoldoutSplit,
    *,
    strategies: tuple[str, ...] = (
        STRATEGY_ESSENCE, STRATEGY_AL, STRATEGY_UMAP, STRATEGY_RANDOM),
    n_seeds: int = 3,
    k0: int = 20,
    seed_pos: int = 5,
    batch: int = 25,
    max_budget: int = 300,
    f1_targets: tuple[float, ...] = (0.70, 0.80),
    pr_auc_targets: tuple[float, ...] = (0.80, 0.90),
    frac_targets: tuple[float, ...] = (0.90, 0.95),
    sec_per_clip_review: float = 4.0,
    metrics_max_workers: int | None = 2,
    cache: RareProjectCache | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> EffortToQualityResult:
    """Labeling effort → model quality (F1 / PR-AUC), per acquisition strategy.

    Every strategy starts from the *same* warm seed (``k0`` clips, ``seed_pos``
    guaranteed positives) and grows the labeled set in ``batch`` steps up to
    ``max_budget`` reviewed clips, training on the labels-so-far and scoring on the
    fixed high-confidence holdout at each step.  They differ only in *which* clips
    each surfaces next (essence likeness / UMAP proximity / model probability /
    random) — and each ABEL arm re-derives that ranking from the labels acquired so
    far, so none is frozen while another learns.  Reports clips-to-reach absolute
    targets (``f1_targets`` "good/exceptional", ``pr_auc_targets``) and
    fraction-of-best targets (``frac_targets``, robust when a rare behaviour caps
    below the absolute bar).  ``cache`` supplies the shared essence features + embedding.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    behavior_name = project.behavior_label(behavior_id)
    pool = holdout_split.train_pool.reset_index(drop=True)
    holdout = holdout_split.holdout.reset_index(drop=True)

    metrics_all, umap_emb, strategies = _pool_signals(
        project, pool, list(strategies), cache=cache,
        metrics_max_workers=metrics_max_workers,
        log=lambda m: _log(f"{behavior_name}: {m}"))
    if not strategies:
        raise ValueError("No runnable strategies for effort-to-quality.")

    per_seed: dict[str, list] = {s: [] for s in strategies}
    for rep in range(n_seeds):
        seed = 6000 + rep
        for strat in strategies:
            traj = _run_quality_strategy(
                trainer, project, behavior_id, pool, holdout, strategy=strat,
                seed=seed, k0=k0, seed_pos=seed_pos, batch=batch,
                max_budget=max_budget, metrics_all=metrics_all, umap_emb=umap_emb,
                log=_log)
            per_seed[strat].append(traj)

    return _assemble_quality(
        per_seed, tuple(strategies), project, behavior_id, behavior_name,
        pool_label="reviewed", k0=k0, seed_pos=seed_pos, f1_targets=f1_targets,
        pr_auc_targets=pr_auc_targets, frac_targets=frac_targets,
        sec_per_clip_review=sec_per_clip_review)


def read_full_window_pool(project: ProjectRef, columns: "list[str] | None" = None) -> pd.DataFrame:
    """Every window the project has features for — the grid *and* the enrichment cache.

    ``segment_features.parquet`` holds only windows on the extraction stride grid.
    Review clips that do not land on that grid (bout-based, temporal-refinement and
    random-sampled clips) are featurised separately by the Active Learning tab and
    persisted to ``enriched_segments.parquet`` — with the same z-scoring and summary
    statistics as the representation builder, so the two are directly comparable.

    Reading only the grid file therefore drops a large, *labelled* slice of the pool:
    measured across the manuscript projects it read 3_chamber_social as 71 % covered
    and Novel-object as 26 %, when the union covers 100 % of both.  Prevalence is
    ``n_pos / n_pool``, so those missing rows silently inflate every enrichment ratio
    derived from it.  Always build the pool through here.
    """
    rep = project.root / "derived" / "representations"
    grid_path = rep / "segment_features.parquet"
    if not grid_path.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(grid_path, columns=columns)]

    enriched_path = rep / "enriched_segments.parquet"
    if enriched_path.exists():
        try:
            enriched = pd.read_parquet(enriched_path, columns=columns)
        except Exception:
            enriched = pd.DataFrame()
        if not enriched.empty:
            # The enrichment cache carries the label columns the training merge adds;
            # keep only what the grid table has so the pool stays one schema.
            shared = [c for c in frames[0].columns if c in enriched.columns]
            frames.append(enriched[shared])

    pool = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if "segment_id" in pool.columns:
        pool = pool.drop_duplicates(subset=["segment_id"], keep="first")
    return pool


def _read_training_labels(project: ProjectRef) -> pd.DataFrame:
    """Training-set labels plus ``label_source`` when the snapshot carries one.

    ``label_source`` is what marks cross-project imports, and older snapshots
    predate the column, so it is read optionally rather than required.
    """
    cols = ["segment_id", "label", "label_source"]
    try:
        return pd.read_parquet(project.training_set_path, columns=cols)
    except Exception:
        return pd.read_parquet(project.training_set_path, columns=cols[:2])


@dataclass
class _LabelCoverage:
    """How much of the training set is actually joinable to the full window pool."""
    id_frac: float
    session_frac: float
    n_missing_sessions: int
    blocking_reason: str = ""
    warning: str = ""


# A full-pool hunt joins labels to the window pool BY ID.  What is left after
# read_full_window_pool() folds in the enrichment cache:
#   1. cross-project imported rows (``label_source == "imported:*"``) — they name a
#      foreign project's sessions and are never in this project's pool, by design;
#   2. labels whose session was removed from the import manifest but whose rows were
#      left in the training snapshot (orphans — prune them, do not re-extract);
#   3. a non-standard window length from a different extraction config;
#   4. review clips whose features were never enriched, because the enrichment cache
#      was invalidated by a representation rebuild and no training run has re-run it.
# Prevalence is n_pos / n_pool, so any of these deflates the denominator's numerator
# and inflates enrichment without bound — one project reported "4901x".  Refuse
# rather than emit a number nobody can interpret.
# Prevalence is n_pos / n_pool, so any of these deflates the denominator's numerator
# and inflates enrichment without bound — one project reported "4901x".  Refuse
# rather than emit a number nobody can interpret.
_COVERAGE_MIN_ID = 0.70
_COVERAGE_MIN_SESSION = 0.90
_COVERAGE_WARN_ID = 0.95
_SESSION_RE = re.compile(r"session_([0-9a-fA-F]+)$")


def _label_coverage(ts: pd.DataFrame, seg_ids: np.ndarray,
                    pos_ids: set[str]) -> _LabelCoverage:
    """Fraction of labels / sessions that survive the join to the window pool.

    Cross-project imported rows are excluded from the denominator: they name the
    *source* project's sessions and can never appear in this project's window pool,
    so counting them reads a project that simply borrowed training examples as
    broken.  One manuscript project is half imports — 4,122 of 9,905 rows.
    """
    pool_ids = set(str(s) for s in seg_ids)
    if "label_source" in ts.columns:
        ts = ts[~ts["label_source"].astype(str).str.startswith("imported")]
    all_label_ids = ts["segment_id"].astype(str)
    id_frac = float(all_label_ids.isin(pool_ids).mean()) if len(all_label_ids) else 0.0

    def _session_of(sid: str) -> str:
        # seg_{animal}_session_{hash}_{start}_{end}.  Strip the trailing frame range
        # from the right (animal names contain underscores), then keep only the
        # session hash: extraction is per session, and keying on animal+session
        # instead reads a multi-animal project as ~92 % missing when the sessions
        # are in fact present.
        base = str(sid).rsplit("_", 2)
        base = base[0] if len(base) == 3 else str(sid)
        m = _SESSION_RE.search(base)
        return m.group(1) if m else base

    label_sessions = {_session_of(s) for s in all_label_ids}
    pool_sessions = {_session_of(s) for s in pool_ids}
    missing = label_sessions - pool_sessions
    session_frac = (1.0 - len(missing) / len(label_sessions)) if label_sessions else 0.0

    cov = _LabelCoverage(id_frac=id_frac, session_frac=session_frac,
                         n_missing_sessions=len(missing))
    if session_frac < _COVERAGE_MIN_SESSION:
        cov.blocking_reason = (
            f"{len(missing)} of {len(label_sessions)} labelled sessions have no rows "
            f"in segment_features.parquet ({session_frac:.0%} session coverage) — "
            f"re-run feature extraction for those sessions.")
    elif id_frac < _COVERAGE_MIN_ID:
        cov.blocking_reason = (
            f"only {id_frac:.0%} of labelled segments have a matching feature window "
            f"(need >={_COVERAGE_MIN_ID:.0%}).")
    elif id_frac < _COVERAGE_WARN_ID:
        cov.warning = (
            f"{1.0 - id_frac:.0%} of labelled segments have no matching feature "
            f"window (off-grid clip ids); full-pool prevalence is computed from "
            f"{len(pos_ids & pool_ids)} of {len(pos_ids)} confirmed positives.")
    return cov


def run_full_pool_supplement(
    project: ProjectRef,
    behavior_id: str,
    *,
    n_seeds: int = 3,
    n_seed_pos: int = 20,
    display_budget: int = 1500,
    effort_targets: tuple[int, ...] = (10, 25, 50),
    sec_per_clip_review: float = 4.0,
    metrics_max_workers: int | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> RareDiscoveryResult:
    """Essence-vs-random discovery over the FULL segment pool (realistic rarity).

    The reviewed-pool experiment has clean ground truth but an inflated ~5%
    prevalence (the reviewed set is positive-enriched).  Here the candidate pool
    is *every* extracted window (tens of thousands), so prevalence is the natural
    rarity of the behaviour.  Only the *confirmed* positives can be scored, so
    this reports how far down each ranking the known positives sit — recall of the
    confirmed set, not precision (unreviewed windows are unlabelled).  AL/UMAP are
    omitted: iterating a retrained model or embedding tens of thousands of windows
    per seed is disproportately costly for a supplement, and essence-vs-random is
    the cleanest realistic-rarity contrast.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    behavior_name = project.behavior_label(behavior_id)
    # Read the shipped per-window features straight from segment_features.parquet —
    # the same feature extraction the classifier uses, so the full-pool essence
    # matches the reviewed-pool essence (and needs no raw pose).
    _log(f"{behavior_name}: loading shipped features for the full pool…")
    seg_df = read_full_window_pool(project)
    if seg_df.empty:
        raise ValueError("Full segment pool is empty (no segment_features.parquet).")
    seg_ids = seg_df["segment_id"].astype(str).to_numpy()
    metrics = _essence_feature_frame(seg_df)
    if metrics.shape[1] == 0:
        raise ValueError(
            "Full-pool essence supplement found no usable shipped features in "
            "segment_features.parquet.")

    ts = _read_training_labels(project)
    pos_ids = set(ts.loc[ts["label"].astype(str).str.strip() == str(behavior_id).strip(),
                         "segment_id"].astype(str))
    is_known_pos = np.asarray([str(s) in pos_ids for s in seg_ids])
    n_known = int(is_known_pos.sum())
    coverage = _label_coverage(ts, seg_ids, pos_ids)
    if coverage.blocking_reason:
        raise ValueError(
            f"Full-pool discovery for '{behavior_name}' is not trustworthy on this "
            f"project: {coverage.blocking_reason}  Prevalence — and every enrichment "
            f"ratio derived from it — would be computed against a fraction of the "
            f"labels, which silently inflates the result.")
    if coverage.warning:
        _log(f"{behavior_name}: WARNING — {coverage.warning}")
    if n_known <= n_seed_pos + 1:
        raise ValueError(
            f"Too few confirmed '{behavior_name}' positives in the full pool "
            f"({n_known}) to cross-validate with n_seed_pos={n_seed_pos}.")
    pos_positions = np.where(is_known_pos)[0]

    strategies = (STRATEGY_ESSENCE, STRATEGY_RANDOM)
    per_seed: dict[str, list[np.ndarray]] = {s: [] for s in strategies}
    for rep in range(n_seeds):
        seed = 8000 + rep
        rng = np.random.default_rng(seed)
        seed_sel = np.sort(rng.choice(pos_positions,
                                      size=min(n_seed_pos, len(pos_positions)),
                                      replace=False))
        seed_set = set(int(i) for i in seed_sel)
        cand_idx = np.asarray([i for i in range(len(seg_ids)) if i not in seed_set],
                              dtype=int)
        is_pos = is_known_pos[cand_idx]
        _log(f"{behavior_name}: full-pool seed {rep + 1}/{n_seeds} — "
             f"{len(cand_idx)} windows, {int(is_pos.sum())} held-out positives")
        order = _rank_essence(metrics.iloc[cand_idx], metrics.iloc[seed_sel])
        if order is None:
            order = _rank_random(len(cand_idx), rng)
        per_seed[STRATEGY_ESSENCE].append(_discovered_curve(is_pos[order]))
        per_seed[STRATEGY_RANDOM].append(
            _discovered_curve(is_pos[_rank_random(len(cand_idx), rng)]))

    prevalence = float(n_known - n_seed_pos) / max(1, len(seg_ids) - n_seed_pos)
    res = _assemble_result(
        per_seed, strategies, project, behavior_id, behavior_name,
        pool_label="full", n_pool=len(seg_ids) - n_seed_pos,
        n_pos_pool=n_known - n_seed_pos, n_seed_pos=n_seed_pos,
        prevalence=prevalence, sec_per_clip_review=sec_per_clip_review,
        effort_targets=effort_targets, display_budget=display_budget)
    # Unreviewed windows count as negatives, so every arm's yield is a floor; and
    # any label that failed the id join is missing from the numerator.  Both belong
    # on the figure, not just in the log.
    res.coverage_note = (
        f"Ground truth = {n_known} confirmed positives ({coverage.id_frac:.0%} of "
        f"labels joined to a feature window); unreviewed windows scored as negatives.")
    return res


# ── rarity scaling (does the essence advantage grow as the behaviour gets rarer?) ──


@dataclass
class RarityPoint:
    prevalence: float
    essence_effort_mean: float   # clips reviewed to find `target` positives
    essence_effort_ci: float
    random_effort_mean: float
    random_effort_ci: float
    ratio_mean: float            # random ÷ essence (× fewer clips with targeting)
    n_pos_kept: int
    n_seeds: int
    # Per-seed clip counts behind the two means, for the Prism replicate export.
    essence_effort_seeds: list[float] = field(default_factory=list)
    random_effort_seeds: list[float] = field(default_factory=list)


@dataclass
class RarityScalingResult:
    project_id: str
    behavior_id: str
    behavior_name: str
    target: int                  # confirmed positives to collect
    sec_per_clip_review: float = 4.0
    points: list[RarityPoint] = field(default_factory=list)
    cells: list[CellResult] = field(default_factory=list)


def run_rarity_scaling(
    project: ProjectRef,
    behavior_id: str,
    holdout_split: HoldoutSplit,
    *,
    prevalences: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005),
    target: int = 10,
    n_seeds: int = 5,
    n_seed_pos: int = 20,
    sec_per_clip_review: float = 4.0,
    metrics_max_workers: int | None = 2,
    cache: RareProjectCache | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> RarityScalingResult:
    """Review effort to find ``target`` positives, essence vs random, as rarity falls.

    At each target prevalence the positives in the candidate pool are randomly
    down-sampled (all negatives kept) to hit that rarity, then both the Essence
    Miner and random sampling are run and we record the clips reviewed to reach
    ``target`` confirmed positives.  The honest rarity story is **divergence**:
    random's cost scales like ``target / prevalence`` and explodes as the
    behaviour gets rarer, while a targeted ranker grows far more slowly — so the
    clips (and hours) saved *grow* with rarity even when the fold-enrichment does
    not.  (Fixed-budget enrichment can actually fall as rarity drops, because the
    needle becomes absolutely scarcer; effort-to-target is the metric that holds.)
    Prevalences above the behaviour's natural rate are skipped, as are any that
    can't supply ``target`` positives to find.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    behavior_name = project.behavior_label(behavior_id)
    pool_all = holdout_split.train_pool.reset_index(drop=True)
    pos_idx = np.where(_pos_mask(pool_all, behavior_id))[0]
    neg_idx = np.where(~_pos_mask(pool_all, behavior_id))[0]
    n_neg = len(neg_idx)
    natural = len(pos_idx) / max(1, len(pool_all))

    if cache is not None and cache.metrics is not None:
        metrics_all = cache.metrics
    else:
        metrics_all = _essence_feature_frame(pool_all)
    if metrics_all is None or metrics_all.shape[1] == 0:
        raise ValueError(
            "Rarity scaling needs the Essence Miner, but the training pool has no "
            "usable shipped features.")

    result = RarityScalingResult(
        project_id=project.project_id, behavior_id=str(behavior_id),
        behavior_name=behavior_name, target=int(target),
        sec_per_clip_review=float(sec_per_clip_review))

    for p in prevalences:
        if p >= natural:
            _log(f"{behavior_name}: skipping prevalence {p:.1%} (≥ natural {natural:.1%})")
            continue
        # positives to keep so kept/(kept+n_neg)=p
        n_keep = int(round(p * n_neg / max(1e-9, 1.0 - p)))
        n_keep = min(n_keep, len(pos_idx) - n_seed_pos)
        if n_keep < target:
            _log(f"{behavior_name}: skipping prevalence {p:.1%} "
                 f"(only {n_keep} positives, need {target})")
            continue
        e_eff: list[int] = []
        r_eff: list[int] = []
        for rep in range(n_seeds):
            rng = np.random.default_rng(4000 + rep)
            seed_sel = np.sort(rng.choice(pos_idx, size=n_seed_pos, replace=False))
            remaining_pos = np.setdiff1d(pos_idx, seed_sel, assume_unique=False)
            kept_pos = rng.choice(remaining_pos, size=n_keep, replace=False)
            cand_idx = np.concatenate([kept_pos, neg_idx])
            rng.shuffle(cand_idx)
            is_pos = _pos_mask(pool_all.iloc[cand_idx], behavior_id)
            order = _rank_essence(metrics_all.iloc[cand_idx], metrics_all.iloc[seed_sel])
            if order is not None:
                e = _first_reach(_discovered_curve(is_pos[order]), target)
                if e is not None:
                    e_eff.append(e)
            r = _first_reach(_discovered_curve(is_pos[_rank_random(len(cand_idx), rng)]),
                             target)
            if r is not None:
                r_eff.append(r)
        if e_eff and r_eff:
            em, rm = float(np.mean(e_eff)), float(np.mean(r_eff))
            result.points.append(RarityPoint(
                prevalence=float(p),
                essence_effort_mean=em,
                essence_effort_ci=vmetrics.ci95(np.asarray(e_eff, dtype=float)),
                random_effort_mean=rm,
                random_effort_ci=vmetrics.ci95(np.asarray(r_eff, dtype=float)),
                ratio_mean=rm / em if em > 0 else float("nan"),
                n_pos_kept=int(n_keep), n_seeds=min(len(e_eff), len(r_eff)),
                essence_effort_seeds=[float(v) for v in e_eff],
                random_effort_seeds=[float(v) for v in r_eff]))
            _log(f"{behavior_name}: prevalence {p:.1%} → essence {em:.0f} vs "
                 f"random {rm:.0f} clips ({rm / max(1, em):.1f}× fewer)")
    result.points.sort(key=lambda pt: pt.prevalence, reverse=True)
    return result


# ── behavior rarity (how rare is the target behaviour vs the others?) ────────


@dataclass
class BehaviorRarityResult:
    project_id: str
    target_name: str
    measure: str                       # "time_fraction" | "bout_rate"
    per_session: pd.DataFrame          # long: session, behavior, time_fraction, bout_rate, n_bouts
    means: dict[str, float]            # behavior -> mean of `measure`, rarest first
    target_rank: int                   # 1 = rarest of all behaviours measured
    n_behaviors: int
    n_sessions: int
    kruskal_p: float = float("nan")    # across-behaviour omnibus
    target_vs_rest_p: float = float("nan")  # one-sided: target < pooled rest
    rarer_than_target: list[str] = field(default_factory=list)  # honesty: anything rarer
    excluded: list[str] = field(default_factory=list)  # behaviours left out (e.g. under-scored)
    # Where the prevalence came from: "bouts" (unbiased deployment detections) or
    # "labels" (the flagged fallback).  Never silently mixed — see
    # :func:`_prevalence_table`.
    source: str = "bouts"

    def target_mean(self) -> float:
        return float(self.means.get(self.target_name, float("nan")))

    def source_caveat(self) -> str:
        """Empty for the unbiased source; the loud caveat for the label fallback."""
        return LABEL_SOURCE_CAVEAT if self.source == PREVALENCE_SOURCE_LABELS else ""


def _session_frame_extents(project: ProjectRef) -> "pd.Series":
    """Per-session frame count (max end_frame) from the segment pool."""
    path = project.root / "derived" / "representations" / "segment_features.parquet"
    df = pd.read_parquet(path, columns=["session_id", "end_frame"])
    return df.groupby("session_id")["end_frame"].max()


def _project_fps(project: ProjectRef, fps: float | None = None) -> float:
    if fps is not None:
        return float(fps)
    import yaml  # noqa: PLC0415
    try:
        cfg = yaml.safe_load((project.root / "project.yaml").read_text(encoding="utf-8")) or {}
        return float(cfg.get("default_fps", 30.0)) or 30.0
    except Exception:
        return 30.0


# Below this pooled detected-time fraction (behaviour frames ÷ frames of the
# sessions the deployment run actually covered) the ``behavior_bouts`` parquets
# are not a deployment read-out at all — they are leftovers from
# ``evaluation_service.evaluate_and_save``, which writes the same filename for the
# handful of bouts in its *eval split*.  Measured across the eight manuscript
# projects the healthy ones sit at 1.7 %–11.9 % pooled; a stale-artifact project
# sits at 0.08 %, so anything under half a percent is degenerate, not rare.
_MIN_POOLED_DETECTED_FRACTION = 0.005

PREVALENCE_SOURCE_TRACES = "traces"
PREVALENCE_SOURCE_BOUTS = "bouts"
PREVALENCE_SOURCE_LABELS = "labels"

# Stamped on every figure/report built from the fallback, because a label share is
# NOT an unbiased prevalence: active learning deliberately over-samples the clips
# the model is unsure about, so it over-represents hard behaviours.
LABEL_SOURCE_CAVEAT = (
    "Prevalence from active-learning-selected LABELS (no usable deployment bout "
    "detections in this project) — biased sampling, not deployment rarity.")


# A behaviour is "silent" in a session when its prevalence there is under this
# fraction of its own active level (its 90th percentile across sessions) — i.e.
# effectively absent, rather than merely low.  Relative, not absolute, so it works
# for a behaviour occupying 40 % of a session and one occupying 0.5 %.
_SILENT_REL_FLOOR = 0.05
_SILENT_ACTIVE_QUANTILE = 0.90
# Above this share of silent sessions the behaviour is structurally gated by the
# design — it *cannot* occur in much of the dataset — rather than genuinely rare
# within the sessions where it can.  Measured across the eight manuscript projects:
# the gated cases (fear-conditioning Shocked, NSF Freeze) sit at 45 %; the real
# rare targets (Wet dog shake 16 %, open-field Freeze 15 %, Sniff TMT 10 %) sit
# well below, and 37 of 43 behaviours are under 10 %.
_MAX_SILENT_FRACTION = 0.30


def session_zero_inflation(per: pd.DataFrame, measure: str = "time_fraction") -> pd.DataFrame:
    """Per-behaviour silent-session share and prevalence *where it does occur*.

    Returns one row per behaviour: ``behavior, behavior_id, silent_fraction,
    prevalence_overall, prevalence_when_active, gated``.

    This distinguishes the two ways a behaviour can look rare on a project-wide
    average, which the discovery hunt must not confuse:

    * **Genuinely rare** — infrequent everywhere it is scored (wet dog shake:
      0.40 % overall, 0.47 % where active, silent in 16 % of sessions).  This is a
      real clip-hunting target; finding it is hard *because it is rare*.
    * **Structurally gated** — common in one kind of session and impossible in the
      rest, so the project-wide mean is dragged down by sessions the behaviour
      could never occur in.  Fear conditioning's *Shocked* is the archetype: no
      shock is delivered in most sessions, so it is silent in ~45 % of them while
      running 1.67 % where it is actually possible.  Hunting it does not measure
      rare-behaviour discovery, it measures whether the tool can find the shock
      sessions — and the "effort to find N" numbers are meaningless because most
      of the pool is ineligible by design.
    """
    rows: list[dict] = []
    for (bid, name), g in per.groupby(["behavior_id", "behavior"]):
        v = g[measure].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        active = float(np.quantile(v, _SILENT_ACTIVE_QUANTILE))
        cut = _SILENT_REL_FLOOR * active
        silent = float(np.mean(v < cut)) if active > 0 else 1.0
        live = v[v >= cut]
        rows.append({
            "behavior": str(name), "behavior_id": str(bid),
            "silent_fraction": silent,
            "prevalence_overall": float(v.mean()),
            "prevalence_when_active": float(live.mean()) if len(live) else float("nan"),
            "gated": bool(silent >= _MAX_SILENT_FRACTION),
        })
    return pd.DataFrame(rows)


def _trace_prevalence_table(
    project: ProjectRef,
    behavior_ids: list[str],
    *,
    fps: float,
    log: Callable[[str], None] = lambda _m: None,
) -> pd.DataFrame:
    """Prevalence from the dense temporal-refinement probability traces.

    The best source available, and the one a *current* run always leaves behind:
    competitive multi-behaviour inference writes one row per frame per session with
    a ``prob_<behavior_id>`` column for every behaviour and a ``predicted_behavior``
    winner, under ``temporal_refinement/target_behavior/<inference_dir>/
    probability_traces/``.  Prevalence is the winner's share of scored frames.

    Why this and not ``behavior_bouts/`` (:func:`_bout_prevalence_table`): the bout
    parquets exported alongside these traces are the *group-level* postprocess, so
    every row is stamped ``behavior_id = "target_behavior"`` — the per-behaviour
    identity survives only here, in the traces.  A project can therefore have a
    complete, current, 69-session deployment run and still show nothing but stale
    per-behaviour bouts, which is exactly how fear conditioning came to rank its
    most abundant behaviour (freezing, 49 % of frames here) as the rarest.

    One caveat, immaterial to *ranking*: the winner-take-all read-out does not apply
    the per-behaviour probability thresholds and minimum-bout durations from
    Temporal Review, so an individual prevalence can differ slightly from what the
    app reports as bouts.  The relative ordering — all this feeds — is unaffected.
    """
    latest_path = (project.root / "derived" / "temporal_refinement"
                   / "target_behavior" / "latest.json")
    if not latest_path.exists():
        return pd.DataFrame()
    try:
        import json  # noqa: PLC0415

        inference_dir = str(json.loads(
            latest_path.read_text(encoding="utf-8")).get("inference_dir", "") or "").strip()
    except Exception as exc:  # unreadable manifest — fall through to the next source
        log(f"rarity: cannot read temporal-refinement manifest ({type(exc).__name__})")
        return pd.DataFrame()
    traces_dir = Path(inference_dir) / "probability_traces"
    if not inference_dir or not traces_dir.is_dir():
        return pd.DataFrame()

    traces = sorted(traces_dir.glob("*.parquet"))
    if not traces:
        return pd.DataFrame()

    # Only a genuinely COMPETITIVE run can be read this way.  ``predicted_behavior``
    # is an argmax, so a single-behaviour inference makes that behaviour the winner
    # on every frame and reports it at 100 % — worse than useless for a rarity
    # ranking.  Require at least two of the requested behaviours to carry a
    # ``prob_`` column, else fall through to the bout detections.
    try:
        cols = set(pd.read_parquet(traces[0]).columns)
    except Exception:
        return pd.DataFrame()
    wanted = {str(b) for b in behavior_ids if f"prob_{b}" in cols}
    if len(wanted) < 2:
        log(f"rarity: temporal-refinement traces cover only {len(wanted)} of the "
            f"selected behaviours (not a competitive run) — not usable")
        return pd.DataFrame()
    uncovered = [project.behavior_label(b) for b in behavior_ids
                 if str(b) not in wanted]
    if uncovered:
        log(f"rarity: {', '.join(uncovered)} absent from the dense run — excluded")

    rows: list[dict] = []
    for tp in traces:
        try:
            d = pd.read_parquet(tp, columns=["predicted_behavior"])
        except Exception:
            continue
        if d.empty:
            continue
        # Session id is carried in the filename, not the columns.
        sid = re.sub(r"_(chunk_)?trace$", "", tp.stem)
        n_frames = len(d)
        sess_min = n_frames / max(1e-9, fps) / 60.0
        pred = d["predicted_behavior"].astype(str).to_numpy()
        # Bout count = contiguous runs of the behaviour, so bout_rate stays
        # comparable with the bouts-derived table.
        starts = np.concatenate([[True], pred[1:] != pred[:-1]])
        for bid in sorted(wanted):
            hit = pred == str(bid)
            rows.append({
                "session": sid, "behavior": project.behavior_label(bid),
                "behavior_id": str(bid),
                "n_bouts": int(np.sum(hit & starts)),
                "time_fraction": float(np.sum(hit)) / n_frames,
                "bout_rate": int(np.sum(hit & starts)) / max(1e-9, sess_min),
            })
    if not rows:
        return pd.DataFrame()
    per = pd.DataFrame(rows)
    measured = per.groupby("behavior_id")["time_fraction"].sum()
    dead = [project.behavior_label(b) for b, v in measured.items() if v <= 0.0]
    if dead:
        # Never scored anywhere in a dense run: unmeasurable, not rarest-at-zero.
        log(f"rarity: {', '.join(dead)} never predicted in the dense traces — "
            f"excluded (NOT ranked as rarest)")
        per = per[~per["behavior_id"].isin(
            [b for b, v in measured.items() if v <= 0.0])]
    log(f"rarity: using dense temporal-refinement traces "
        f"({per['session'].nunique()} sessions)")
    return per.reset_index(drop=True)


def _bout_prevalence_table(
    project: ProjectRef,
    behavior_ids: list[str],
    *,
    fps: float,
    log: Callable[[str], None] = lambda _m: None,
) -> pd.DataFrame:
    """Long per-(session, behaviour) prevalence from the dense bout detections.

    Columns: ``session, behavior, behavior_id, n_bouts, time_fraction, bout_rate``.
    Touches only the shipped ``behavior_bouts`` parquets — no model fitting, no pose.

    Two things this must not do, both of which silently manufacture a "rarest"
    behaviour out of missing data rather than out of biology:

    * **A behaviour with no detections is unmeasurable, not rare.**  An empty
      ``<id>_bouts.parquet`` (the file exists, zero rows) used to average to a
      time fraction of exactly 0.0 and therefore win the rarity ranking outright —
      which is how fear conditioning came to nominate *freezing*, its single most
      abundant behaviour, as the rarest.  Absent and empty are now treated alike:
      dropped from the table with a logged reason.
    * **Sessions no run covered are not sessions with zero behaviour.**  Prevalence
      is averaged over the *run-covered* sessions — the union of sessions appearing
      in any behaviour's bouts file — not over every session in the pool.  Zero-
      filling the rest penalised whichever behaviour happened to be deployed over
      fewer sessions (DG_EPM covers 21 of 57; TMT 21 of 29), which is a property of
      when the run was launched, not of how rare the behaviour is.  True zeros
      *within* the covered set are still counted, so a behaviour genuinely absent
      from a covered session is not flattered.
    """
    sess_len = _session_frame_extents(project)
    sess_len.index = sess_len.index.astype(str)
    sess_min = sess_len / fps / 60.0
    bouts_dir = project.root / "derived" / "behavior_bouts"

    # Pass 1: read what exists, and learn which sessions the run actually covered.
    detected: dict[str, tuple[str, pd.DataFrame]] = {}
    covered: set[str] = set()
    for bid in behavior_ids:
        name = project.behavior_label(bid)
        # bouts are filed by disk id; behavior_disk_name for renamed behaviours.
        path = bouts_dir / f"{bid}_bouts.parquet"
        if not path.exists():
            log(f"rarity: no bouts file for {name} — not measurable, excluded")
            continue
        d = pd.read_parquet(path)
        if d.empty:
            log(f"rarity: bouts file for {name} is empty (0 detections) — "
                f"not measurable, excluded (NOT ranked as rarest)")
            continue
        g = d.groupby(d["session_id"].astype(str)).agg(
            n_bouts=("start_frame", "size"), frames=("duration_frames", "sum"))
        detected[str(bid)] = (name, g)
        covered |= set(g.index)

    covered &= set(sess_len.index)
    if not detected or not covered:
        return pd.DataFrame()

    rows: list[dict] = []
    for bid, (name, g) in detected.items():
        for sid in sorted(covered):
            nb = int(g["n_bouts"].get(sid, 0))
            fr = int(g["frames"].get(sid, 0))
            rows.append({
                "session": sid, "behavior": name, "behavior_id": str(bid),
                "n_bouts": nb,
                "time_fraction": fr / max(1, int(sess_len[sid])),
                "bout_rate": nb / max(1e-9, float(sess_min.get(sid, np.nan))),
            })
    per = pd.DataFrame(rows)

    pooled = float(per["time_fraction"].sum() / max(1, len(covered)))
    if pooled < _MIN_POOLED_DETECTED_FRACTION:
        log(f"rarity: bout detections cover only {pooled:.3%} of the {len(covered)} "
            f"covered sessions — this is a stale evaluation artifact, not a "
            f"deployment read-out; bout-based rarity rejected")
        return pd.DataFrame()
    return per


def _label_prevalence_table(
    project: ProjectRef,
    behavior_ids: list[str],
    *,
    fps: float,
    log: Callable[[str], None] = lambda _m: None,
) -> pd.DataFrame:
    """Fallback prevalence from the confirmed labels, for projects with no bouts.

    Same long schema as :func:`_bout_prevalence_table` so every downstream figure,
    Prism export and report works unchanged — but built from
    ``training_sets/training_set.parquet``: ``time_fraction`` is the behaviour's
    share of *labeled* frames in a session and ``n_bouts`` its label count.

    This is a biased estimate and is always reported as such
    (:data:`LABEL_SOURCE_CAVEAT`): active learning picks clips the model is unsure
    about, so the label mix over-represents hard behaviours and under-represents
    the obvious ones.  It is used only to decide *which behaviour to hunt* when the
    unbiased source is unavailable — a biased ordering beats nominating whichever
    behaviour happens to have an empty parquet.
    """
    path = project.training_set_path
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=["session_id", "start_frame", "end_frame", "label"])
    df["session_id"] = df["session_id"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()
    df["frames"] = (df["end_frame"] - df["start_frame"]).clip(lower=1)

    sess_len = _session_frame_extents(project)
    sess_len.index = sess_len.index.astype(str)
    sess_min = sess_len / fps / 60.0
    labeled_frames = df.groupby("session_id")["frames"].sum()

    wanted = {str(b) for b in behavior_ids}
    present = wanted & set(df["label"].unique())
    if len(present) < 2:
        log("rarity: fewer than two selected behaviours appear in the labels — "
            "cannot rank")
        return pd.DataFrame()

    g = df[df["label"].isin(present)].groupby(["label", "session_id"]).agg(
        n=("frames", "size"), frames=("frames", "sum"))
    rows: list[dict] = []
    for bid in sorted(present):
        name = project.behavior_label(bid)
        for sid in labeled_frames.index:
            n = int(g["n"].get((bid, sid), 0))
            fr = int(g["frames"].get((bid, sid), 0))
            rows.append({
                "session": sid, "behavior": name, "behavior_id": str(bid),
                "n_bouts": n,
                "time_fraction": fr / max(1, int(labeled_frames[sid])),
                "bout_rate": n / max(1e-9, float(sess_min.get(sid, np.nan))),
            })
    missing = [project.behavior_label(b) for b in sorted(wanted - present)]
    if missing:
        log(f"rarity: no labels for {', '.join(missing)} — excluded")
    return pd.DataFrame(rows)


def _prevalence_table(
    project: ProjectRef,
    behavior_ids: list[str],
    *,
    fps: float,
    log: Callable[[str], None] = lambda _m: None,
) -> tuple[pd.DataFrame, str]:
    """Best available prevalence table, with the source it came from.

    In preference order: the dense temporal-refinement traces a current run leaves
    behind (:func:`_trace_prevalence_table`), then the exported per-behaviour bout
    detections (:func:`_bout_prevalence_table`, which rejects stale evaluation
    artifacts), then — flagged — the active-learning-selected labels.  Traces come
    first because they are the only per-behaviour record a competitive multi-
    behaviour run writes: its *bouts* are stamped ``target_behavior``, so a project
    can have a complete current run and stale per-behaviour bouts at the same time.
    """
    per = _trace_prevalence_table(project, behavior_ids, fps=fps, log=log)
    if not per.empty:
        return per, PREVALENCE_SOURCE_TRACES
    per = _bout_prevalence_table(project, behavior_ids, fps=fps, log=log)
    if not per.empty:
        return per, PREVALENCE_SOURCE_BOUTS
    log("rarity: falling back to LABEL prevalence — biased, not deployment rarity")
    return (_label_prevalence_table(project, behavior_ids, fps=fps, log=log),
            PREVALENCE_SOURCE_LABELS)


def rank_behaviors_by_rarity(
    project: ProjectRef,
    behavior_ids: list[str] | None = None,
    *,
    exclude_behavior_ids: list[str] | None = None,
    measure: str = "time_fraction",
    fps: float | None = None,
    skip_gated: bool = True,
    progress_cb: Callable[[str], None] | None = None,
) -> list[tuple[str, str, float]]:
    """Rank behaviours rarest→commonest: ``[(behavior_id, name, mean measure), …]``.

    ``skip_gated`` drops behaviours that are rare only because the design silences
    them in much of the dataset (:func:`session_zero_inflation`) — a shock response
    in the two-thirds of sessions with no shock is not a rare-behaviour hunting
    target.  Pass ``False`` to rank on the raw project-wide mean.

    The cheap first pass of a multi-project rare-discovery run: before spending
    hours on discovery/quality arms, read the dense bout detections and decide
    which behaviour in *this* project is actually the rare one worth hunting.
    Behaviours with no *usable* detections — no bouts file, or a file with zero
    rows — are absent from the ranking rather than silently ranked as "rarest" at
    zero.  When the project has no usable bout detections at all the ranking falls
    back to (biased) label prevalence rather than returning an arbitrary order;
    :func:`prevalence_source` reports which source was used.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    bids = list(behavior_ids or [b for b in project.behavior_names
                                 if str(b) != "no_behavior"])
    excl = {str(b) for b in (exclude_behavior_ids or [])}
    bids = [b for b in bids if str(b) not in excl]
    if not bids:
        return []
    per, source = _prevalence_table(
        project, bids, fps=_project_fps(project, fps), log=_log)
    if per.empty:
        return []
    if source == PREVALENCE_SOURCE_LABELS:
        _log(f"rarity: {LABEL_SOURCE_CAVEAT}")

    # Structurally gated behaviours are not rare in the sense the hunt measures —
    # see :func:`session_zero_inflation`.  Demote rather than delete: if EVERY
    # behaviour is gated, ranking them is still better than returning nothing.
    if skip_gated and source != PREVALENCE_SOURCE_LABELS:
        zi = session_zero_inflation(per, measure=measure)
        gated = zi[zi["gated"]] if not zi.empty else zi
        if not gated.empty and len(gated) < len(zi):
            for _i, r in gated.iterrows():
                _log(f"rarity: {r['behavior']} is silent in "
                     f"{r['silent_fraction']:.0%} of sessions "
                     f"({r['prevalence_overall']:.2%} overall but "
                     f"{r['prevalence_when_active']:.2%} where it occurs) — "
                     f"structurally gated by the design, not rare; not a hunt target")
            per = per[~per["behavior_id"].isin(set(gated["behavior_id"]))]
        elif not gated.empty:
            _log("rarity: every behaviour looks session-gated — ranking them all "
                 "rather than dropping the project")

    means = per.groupby(["behavior_id", "behavior"])[measure].mean().sort_values()
    return [(str(bid), str(name), float(val))
            for (bid, name), val in means.items()]


def prevalence_source(
    project: ProjectRef,
    behavior_ids: list[str] | None = None,
    *,
    fps: float | None = None,
) -> str:
    """Which prevalence source this project resolves to — ``"bouts"`` or ``"labels"``."""
    bids = list(behavior_ids or [b for b in project.behavior_names
                                 if str(b) != "no_behavior"])
    _per, source = _prevalence_table(project, bids, fps=_project_fps(project, fps))
    return source


def run_behavior_rarity(
    project: ProjectRef,
    target_behavior_id: str,
    *,
    behavior_ids: list[str] | None = None,
    exclude_behavior_ids: list[str] | None = None,
    measure: str = "time_fraction",
    fps: float | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> BehaviorRarityResult:
    """Per-session prevalence of every behaviour, to show how rare the target is.

    Prevalence is read from the shipped **deployed-model bout detections**
    (``derived/behavior_bouts/<id>_bouts.parquet``) — dense over every frame of
    every session the run covered, so unlike the active-learning-selected *labels*
    it is an unbiased picture of how often each behaviour actually occurs.  A
    project whose bouts are missing or are stale evaluation-split artifacts falls
    back to label prevalence with ``source == "labels"`` and the caveat stamped on
    the figure (:func:`_prevalence_table`).  ``measure`` is
    ``time_fraction`` (behaviour frames ÷ session frames) or ``bout_rate``
    (bouts per minute).  Each session is one observation, giving a distribution
    per behaviour that a reviewer can run an ANOVA / Kruskal–Wallis on directly.

    ``exclude_behavior_ids`` drops behaviours that are not validly measured in
    this dataset (e.g. one the model was never trained to detect well, or that is
    essentially absent) so they don't pollute the "rest" the target is tested
    against or distort the ranking.  The target is never excluded.  The dropped
    behaviours are recorded on the result and annotated on the figure.

    This describes the model's read-out of the data, not hand-verified ground
    truth — the honest and standard way to quantify relative rarity without dense
    manual scoring.  The result also lists any behaviour *rarer* than the target,
    so the figure never overclaims the target is the single rarest when it is not.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    fps = _project_fps(project, fps)

    bids = behavior_ids or [b for b in project.behavior_names if str(b) != "no_behavior"]
    # Drop excluded behaviours (never the target), so an under-scored behaviour
    # doesn't sit in the "rest" comparison or the ranking.
    excl = {str(b) for b in (exclude_behavior_ids or [])}
    excl.discard(str(target_behavior_id))
    excluded_names = [project.behavior_label(b) for b in bids if str(b) in excl]
    bids = [b for b in bids if str(b) not in excl]

    per, source = _prevalence_table(project, bids, fps=fps, log=_log)
    if per.empty:
        raise ValueError(
            "No usable prevalence source — neither deployment bout detections nor "
            "labels can measure rarity in this project.")
    if source == PREVALENCE_SOURCE_LABELS:
        _log(f"rarity: {LABEL_SOURCE_CAVEAT}")

    target_name = project.behavior_label(target_behavior_id)
    means = (per.groupby("behavior")[measure].mean()
             .sort_values())  # rarest first
    order = list(means.index)
    rarer = [b for b in order if means[b] < means.get(target_name, np.inf)
             and b != target_name]
    rank = order.index(target_name) + 1 if target_name in order else -1

    kruskal_p = tvr_p = float("nan")
    try:
        from scipy import stats  # noqa: PLC0415
        groups = [g[measure].to_numpy() for _, g in per.groupby("behavior")]
        if len(groups) >= 2:
            kruskal_p = float(stats.kruskal(*groups).pvalue)
        tgt = per.loc[per["behavior"] == target_name, measure].to_numpy()
        rest = per.loc[per["behavior"] != target_name, measure].to_numpy()
        if tgt.size and rest.size:
            tvr_p = float(stats.mannwhitneyu(tgt, rest, alternative="less").pvalue)
    except Exception as exc:  # scipy missing / degenerate — leave NaN
        _log(f"rarity: stats unavailable ({type(exc).__name__})")

    _log(f"{target_name}: rank {rank}/{len(order)} by {measure}; "
         f"Kruskal p={kruskal_p:.1e}, {target_name}<rest p={tvr_p:.1e}"
         + (f"; rarer still: {', '.join(rarer)}" if rarer else ""))

    return BehaviorRarityResult(
        project_id=project.project_id, target_name=target_name, measure=measure,
        per_session=per, means={b: float(means[b]) for b in order},
        target_rank=rank, n_behaviors=len(order),
        n_sessions=int(per["session"].nunique()),
        kruskal_p=kruskal_p, target_vs_rest_p=tvr_p, rarer_than_target=rarer,
        excluded=excluded_names, source=source)


# ── preflight: is there enough labeled evidence to hunt anything? ───────────
# Phase 1 of the two-phase workflow.  The hunt itself costs hours per project, and
# the single commonest way to waste them is starting it on a behaviour with eight
# confirmed examples: the cross-validation then has nothing left to score once the
# seed exemplars are taken, and the run dies (or worse, produces a curve out of
# three positives).  This pass reads only the bout detections and the *label
# columns* of the training set — seconds — and says, per project, which behaviour
# is rarest, how many confirmed examples it has, and whether that is enough.

PREFLIGHT_OK = "ok"
PREFLIGHT_WARN = "warn"
PREFLIGHT_BLOCKED = "blocked"

_PREFLIGHT_LABEL_COLS = ("label", "label_source", "reviewer_confidence",
                         "session_id", "animal_id")


@dataclass
class BehaviorPreflight:
    behavior_id: str
    behavior_name: str
    measure: str
    prevalence: float          # mean of `measure` from the dense bouts (NaN if none)
    rank: int                  # 1 = rarest of the behaviours checked
    n_labeled: int             # confirmed positives in the hunting pool
    n_seed_pos: int            # exemplars the run would spend defining the behaviour
    status: str = PREFLIGHT_OK
    note: str = ""

    @property
    def n_held_out(self) -> int:
        """Positives left to *discover* once the seed exemplars are taken."""
        return max(0, self.n_labeled - self.n_seed_pos)

    def runnable(self) -> bool:
        return self.status != PREFLIGHT_BLOCKED


@dataclass
class ProjectPreflight:
    project_id: str
    project_name: str
    behaviors: list[BehaviorPreflight] = field(default_factory=list)  # rarest first
    error: str = ""
    # Behaviour the USER picked instead of the automatic choice.  The ranking is a
    # heuristic over model output — it cannot know that a behaviour is gated by a
    # design fact no file records, or that the interesting rare behaviour is the
    # second one down.  An explicit pick always wins, and is never silently ignored.
    target_override: str = ""

    @property
    def target(self) -> "BehaviorPreflight | None":
        """The behaviour to hunt: the user's pick, else the rarest runnable one."""
        if self.target_override:
            picked = next((b for b in self.behaviors
                           if b.behavior_id == self.target_override), None)
            if picked is not None:
                return picked
        return next((b for b in self.behaviors if b.runnable()), None)

    def override_note(self) -> str:
        """Human-readable note when the automatic pick was overridden."""
        if not self.target_override:
            return ""
        auto = next((b for b in self.behaviors if b.runnable()), None)
        tgt = self.target
        if tgt is None or (auto is not None and auto.behavior_id == tgt.behavior_id):
            return ""
        return (f"{self.project_name}: hunting {tgt.behavior_name} "
                f"(chosen by you) instead of {auto.behavior_name}"
                if auto else f"{self.project_name}: hunting {tgt.behavior_name} (chosen by you)")

    @property
    def rarest(self) -> "BehaviorPreflight | None":
        return self.behaviors[0] if self.behaviors else None

    def blocked_note(self) -> str:
        """Why the rarest behaviour cannot be hunted, if it cannot."""
        r = self.rarest
        if self.error:
            return self.error
        if r is None:
            return "No behaviour could be measured (no bout detections)."
        if r.runnable():
            return ""
        tgt = self.target
        return (f"{r.note} " + (f"Falling back to {tgt.behavior_name}."
                                if tgt else "Nothing left to hunt in this project."))


def _preflight_pool_labels(
    project: ProjectRef,
    *,
    holdout_groups: "list[str] | None",
    min_confidence: float,
    test_size: float,
    seed: int,
) -> pd.Series:
    """Label counts of the hunting pool, read without loading the feature columns."""
    cols = None
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
        have = set(pq.ParquetFile(project.training_set_path).schema.names)
        cols = [c for c in _PREFLIGHT_LABEL_COLS if c in have]
    except Exception:  # pyarrow missing / odd file — fall back to a full read
        cols = None
    df = pd.read_parquet(project.training_set_path, columns=cols)
    # The SAME split the hunt runs on (same seed/test size), so the counts shown
    # are exactly the positives the discovery arms will have to work with.
    sp = holdout_split(project, holdout_groups=holdout_groups,
                       min_confidence=min_confidence, test_size=test_size,
                       seed=seed, df=df)
    return sp.train_pool["label"].astype(str).str.strip().value_counts()


def preflight_project(
    project: ProjectRef,
    behavior_ids: list[str],
    *,
    n_seed_pos: int = 8,
    min_effort_target: int = 10,
    measure: str = "time_fraction",
    exclude_behavior_ids: list[str] | None = None,
    holdout_groups: "list[str] | None" = None,
    min_confidence: float = 1.0,
    test_size: float = 0.25,
    seed: int = 42,
    progress_cb: Callable[[str], None] | None = None,
) -> ProjectPreflight:
    """Rank one project's behaviours by rarity and check each has enough examples."""
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    out = ProjectPreflight(project_id=project.project_id, project_name=project.name)
    _log(f"{project.name}: ranking behaviours by rarity…")
    try:
        ranking = rank_behaviors_by_rarity(
            project, behavior_ids, exclude_behavior_ids=exclude_behavior_ids,
            measure=measure, progress_cb=progress_cb)
    except Exception as exc:  # noqa: BLE001
        out.error = f"Rarity ranking failed: {type(exc).__name__}: {exc}"
        return out
    ranked_ids = {bid for bid, _n, _v in ranking}
    # Behaviours with no bout detections still get a row (rarity unknown), so the
    # user sees them rather than wondering where they went.
    ranking = list(ranking) + [(str(b), project.behavior_label(b), float("nan"))
                               for b in behavior_ids if str(b) not in ranked_ids]

    _log(f"{project.name}: counting confirmed examples…")
    try:
        counts = _preflight_pool_labels(
            project, holdout_groups=holdout_groups, min_confidence=min_confidence,
            test_size=test_size, seed=seed)
    except Exception as exc:  # noqa: BLE001
        out.error = f"Could not read the training set: {type(exc).__name__}: {exc}"
        return out

    for i, (bid, bname, val) in enumerate(ranking):
        n = int(counts.get(str(bid), 0))
        pf = BehaviorPreflight(
            behavior_id=str(bid), behavior_name=bname, measure=measure,
            prevalence=val, rank=i + 1, n_labeled=n, n_seed_pos=int(n_seed_pos))
        if pf.n_held_out < 2:
            pf.status = PREFLIGHT_BLOCKED
            pf.note = (f"Only {n} confirmed examples — at least "
                       f"{n_seed_pos + 2} are needed to define the behaviour from "
                       f"{n_seed_pos} exemplars and still have some held out to "
                       f"find. Label more before running the hunt.")
        elif pf.n_held_out < int(min_effort_target):
            pf.status = PREFLIGHT_WARN
            pf.note = (f"{n} confirmed examples leaves only {pf.n_held_out} to "
                       f"discover after the {n_seed_pos} exemplars — the curves "
                       f"will be noisy and the effort-to-{min_effort_target} bars "
                       f"empty. ~{n_seed_pos + min_effort_target}+ is comfortable.")
        out.behaviors.append(pf)
    return out


def preflight_rows(results: list[ProjectPreflight]) -> list[dict]:
    """Tidy rows for the preflight table / CSV export."""
    rows: list[dict] = []
    for pr in results:
        tgt = pr.target
        for b in pr.behaviors:
            rows.append({
                "project": pr.project_name,
                "project_id": pr.project_id,
                "behavior": b.behavior_name,
                "rarity_rank": b.rank,
                b.measure: b.prevalence,
                "confirmed_examples": b.n_labeled,
                "exemplars_needed": b.n_seed_pos,
                "left_to_discover": b.n_held_out,
                "status": b.status,
                "would_be_hunted": bool(tgt is not None
                                        and tgt.behavior_id == b.behavior_id),
                "note": b.note or pr.error,
            })
    return rows


def _budget_grid(n: int, n_points: int = 25) -> list[int]:
    """A review-budget grid: dense early (where the curves separate), sparse late."""
    if n <= n_points:
        return list(range(1, n + 1))
    lo = np.linspace(1, min(50, n), 10)
    hi = np.linspace(min(50, n), n, n_points - 10 + 1)[1:]
    grid = sorted(set(int(round(x)) for x in np.concatenate([lo, hi])))
    return [g for g in grid if 1 <= g <= n]


def _first_reach(discovered: np.ndarray, target: int) -> int | None:
    """Smallest #reviewed at which ``target`` confirmed positives are found."""
    hit = np.where(discovered >= target)[0]
    return int(hit[0] + 1) if hit.size else None


def whole_video_minutes(project: ProjectRef) -> float:
    """Total minutes of video across the project's reviewed sessions.

    Read from the segment pool's frame extents ÷ fps — the time a human would
    spend watching every recording end-to-end (the whole-video baseline).
    """
    import yaml

    fps = 30.0
    pj = project.root / "project.yaml"
    if pj.exists():
        try:
            cfg = yaml.safe_load(pj.read_text(encoding="utf-8")) or {}
            fps = float(cfg.get("default_fps", 30.0)) or 30.0
        except Exception:
            fps = 30.0
    path = project.root / "derived" / "representations" / "segment_features.parquet"
    if not path.exists():
        return float("nan")
    try:
        df = pd.read_parquet(path, columns=["session_id", "end_frame"])
    except Exception:
        return float("nan")
    per_session = df.groupby("session_id")["end_frame"].max()
    return float(per_session.sum() / fps / 60.0)


# ── figures (self-contained; no dependency on plots.py) ────────────────────

_CAVEAT = ("Ground truth = the confirmed positives only; behaviours no tool "
           "surfaced are invisible. Curves show relative enrichment, not absolute recall.")


def _fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _disabled_note(result: RareDiscoveryResult) -> str:
    """Figure footnote naming any arm that could not run, and why.

    Without this a dropped arm is just an absent line — the figure reads as a
    complete comparison in which that tool lost, which is the opposite of what
    happened.
    """
    if not getattr(result, "disabled_strategies", None):
        return ""
    bits = [f"{STRATEGY_LABELS.get(s, s)} — {why}"
            for s, why in result.disabled_strategies.items()]
    return "\nNOT TESTED in this run: " + "; ".join(bits)


def plot_discovery_curve(result: RareDiscoveryResult, out_path):
    """Mechanism panel: clips reviewed → cumulative confirmed positives found.

    Secondary to :func:`plot_quality_curve`.  Positives-found is the *mechanism*
    (the tools surface the behaviour far cheaper than random), not the outcome the
    user cares about — a run can acquire more positives and still train a worse
    model, which is measured, not hypothetical.  Keep it, but do not headline it.
    """
    plt = _fig()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for strat, cur in result.curves.items():
        xs = [p.n_reviewed for p in cur.points]
        ys = [p.n_found_mean for p in cur.points]
        cis = [p.n_found_ci for p in cur.points]
        ax.plot(xs, ys, "-o", ms=3, lw=2, color=cur.color(), label=cur.label(),
                zorder=3 if strat != STRATEGY_RANDOM else 1)
        lo = np.clip(np.array(ys) - np.array(cis), 0, None)  # counts can't go negative
        ax.fill_between(xs, lo, np.array(ys) + np.array(cis),
                        color=cur.color(), alpha=0.15, lw=0)
    ax.set_ylim(bottom=0)
    # Analytic random expectation (prevalence × reviewed) as a reference line.
    xmax = max(p.n_reviewed for c in result.curves.values() for p in c.points)
    ax.plot([0, xmax], [0, result.prevalence * xmax], "--", color="#888888",
            lw=1, zorder=0, label=f"random expectation ({result.prevalence:.1%})")
    ax.set_xlabel("Clips reviewed (labeling effort)")
    ax.set_ylabel(f"Confirmed {result.behavior_name} discovered")
    ax.set_title(f"Rare-behavior discovery — {result.behavior_name} "
                 f"({result.project_id})\n{result.provenance()}", fontsize=9)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    caveat = _CAVEAT + _disabled_note(result)
    fig.text(0.5, -0.02, caveat, ha="center", fontsize=6.5, color="#b03030", wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_effort_to_n(result: RareDiscoveryResult, out_path, target: int | None = None):
    """Secondary panel: minutes of human effort to collect N confirmed positives.

    Model arms are ``clips × sec_per_clip_review``; the whole-video bar is the
    expected watch time until the N-th positive appears (N/prevalence-in-time).
    Like :func:`plot_discovery_curve` this reports positives collected, not model
    quality — see that function's note.
    """
    plt = _fig()
    if target is None:
        # Largest target every model arm actually reached.
        reached = [t for t in sorted({k for c in result.curves.values()
                                      for k in c.effort_to_n})
                   if all(t in c.effort_to_n for s, c in result.curves.items())]
        target = reached[-1] if reached else None
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    labels, minutes, colors, errs = [], [], [], []
    spc = result.sec_per_clip_review
    for strat, cur in result.curves.items():
        clips = cur.effort_to_n.get(int(target)) if target else None
        if clips is None:
            continue
        labels.append(cur.label())
        minutes.append(clips * spc / 60.0)
        errs.append(cur.effort_to_n_ci.get(int(target), 0.0) * spc / 60.0)
        colors.append(cur.color())
    # Whole-video: watch until the N-th confirmed positive (uniform-in-time expectation).
    if target and np.isfinite(result.total_video_minutes) and result.n_pos_pool > 0:
        wv = result.total_video_minutes * (float(target) / result.n_pos_pool)
        labels.append(STRATEGY_LABELS[STRATEGY_WHOLE_VIDEO])
        minutes.append(wv)
        errs.append(0.0)
        colors.append(STRATEGY_COLORS[STRATEGY_WHOLE_VIDEO])
    order = np.argsort(minutes)
    labels = [labels[i] for i in order]
    minutes = [minutes[i] for i in order]
    errs = [errs[i] for i in order]
    colors = [colors[i] for i in order]
    bars = ax.barh(labels, minutes, xerr=errs, color=colors, alpha=0.9)
    for b, m in zip(bars, minutes):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                f" {m:.1f} min", va="center", fontsize=8)
    ax.set_xlabel("Human effort to find "
                  f"{target} confirmed {result.behavior_name} (minutes)")
    ax.set_title(f"Effort to target — {result.behavior_name} ({result.project_id})\n"
                 f"clip review @ {spc:.0f}s/clip\n{result.provenance()}", fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    fig.text(0.5, -0.02, _disabled_note(result).strip(), ha="center",
             fontsize=6.5, color="#b03030", wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quality_curve(result: EffortToQualityResult, out_path, metric: str = "f1"):
    """HEADLINE: clips reviewed → held-out model quality for the TARGET behaviour.

    This is the outcome the user actually buys with labeling effort, and it does
    not follow from positives-found: an arm can acquire more positives and train a
    worse model, because model quality also needs informative *negatives*.  Both
    metrics are target-class only (see :func:`_target_metrics`) — macro-averaged
    F1 reads ~0.5 for a model that never predicts a rare behaviour at all.
    """
    plt = _fig()
    is_f1 = metric == "f1"
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for strat, cur in result.curves.items():
        xs = [p.n_clips for p in cur.points]
        ys = [(p.f1_mean if is_f1 else p.pr_auc_mean) for p in cur.points]
        cis = [(p.f1_ci if is_f1 else p.pr_auc_ci) for p in cur.points]
        ax.plot(xs, ys, "-o", ms=3, lw=2, color=cur.color(), label=cur.label(),
                zorder=3 if strat != STRATEGY_RANDOM else 1)
        lo = np.clip(np.array(ys) - np.array(cis), 0, 1)
        ax.fill_between(xs, lo, np.clip(np.array(ys) + np.array(cis), 0, 1),
                        color=cur.color(), alpha=0.15, lw=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Clips reviewed (labeling effort)")
    ax.set_ylabel(f"Held-out {'F1' if is_f1 else 'PR-AUC'} "
                  f"({result.behavior_name} class only)")
    ax.set_title(f"Effort to a good model — {result.behavior_name} "
                 f"({result.project_id})\n"
                 f"acquisition pool: {result.pool_label}", fontsize=9)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_effort_to_quality(result: EffortToQualityResult, out_path,
                           target_label: str | None = None):
    """Bars: clips (and minutes) of labeling to reach a quality target, per strategy."""
    plt = _fig()
    if target_label is None:
        # Hardest target still reached by ≥2 strategies (fewest reachers = hardest),
        # so the bars compare a demanding bar an arm that never gets there is simply
        # absent from — not the easy target everyone clears.
        counts = {lab: sum(1 for c in result.curves.values() if lab in c.effort)
                  for lab in {l for c in result.curves.values() for l in c.effort}}
        cand = ([l for l, k in counts.items() if k >= 2]
                or [l for l, k in counts.items() if k >= 1])
        target_label = min(cand, key=lambda l: counts[l]) if cand else None
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    labels, clips, colors, errs = [], [], [], []
    for strat, cur in result.curves.items():
        c = cur.effort.get(target_label) if target_label else None
        if c is None:
            continue
        labels.append(cur.label())
        clips.append(c)
        errs.append(cur.effort_ci.get(target_label, 0.0))
        colors.append(cur.color())
    if not clips:
        plt.close(fig)
        return None
    order = np.argsort(clips)
    labels = [labels[i] for i in order]
    clips = [clips[i] for i in order]
    errs = [errs[i] for i in order]
    colors = [colors[i] for i in order]
    bars = ax.barh(labels, clips, xerr=errs, color=colors, alpha=0.9)
    spc = result.sec_per_clip_review
    for b, c in zip(bars, clips):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                f" {c:.0f} clips ≈ {c * spc / 60.0:.0f} min", va="center", fontsize=8)
    ax.set_xlabel(f"Clips reviewed to reach {target_label}")
    ax.set_title(f"Labeling effort to a good model — {result.behavior_name} "
                 f"({result.project_id})\n@ {spc:.0f}s/clip")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rarity_scaling(result: "RarityScalingResult", out_path):
    """Fig C: clips (and hours) to find N positives diverge as the behaviour rarer."""
    if not result.points:
        return None  # nothing to plot (every prevalence skipped/degenerate)
    plt = _fig()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    xs = [p.prevalence * 100 for p in result.points]
    e = [p.essence_effort_mean for p in result.points]
    r = [p.random_effort_mean for p in result.points]
    e_ci = [p.essence_effort_ci for p in result.points]
    r_ci = [p.random_effort_ci for p in result.points]
    ax.errorbar(xs, r, yerr=r_ci, fmt="-o", ms=5, lw=2,
                color=STRATEGY_COLORS[STRATEGY_RANDOM], capsize=3, label="Random clips")
    ax.errorbar(xs, e, yerr=e_ci, fmt="-o", ms=5, lw=2,
                color=STRATEGY_COLORS[STRATEGY_ESSENCE], capsize=3, label="Essence Miner")
    for x, pt in zip(xs, result.points):
        ax.annotate(f"{pt.ratio_mean:.1f}×", (x, pt.random_effort_mean),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=7, color="#444444")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()  # rarer → right
    ax.set_xlabel("Behaviour prevalence (%, log; rarer →)")
    ax.set_ylabel(f"Clips reviewed to find {result.target} confirmed positives (log)")
    ax.set_title(f"The rarer the behaviour, the more targeting saves\n"
                 f"{result.behavior_name} ({result.project_id})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_MEASURE_LABEL = {
    "time_fraction": "Time in behaviour (fraction of session)",
    "bout_rate": "Bouts per minute",
}
# The label fallback measures a different denominator (labeled frames, not session
# frames), so the axis must not claim the deployment quantity.
_MEASURE_LABEL_FROM_LABELS = {
    "time_fraction": "Share of LABELED frames (biased)",
    "bout_rate": "Labeled segments per minute (biased)",
}


def plot_behavior_rarity(result: BehaviorRarityResult, out_path):
    """Fig 3X: per-session prevalence per behaviour, target highlighted, sorted rarest→."""
    plt = _fig()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    order = list(result.means.keys())  # rarest first
    per = result.per_session
    for i, beh in enumerate(order):
        vals = per.loc[per["behavior"] == beh, result.measure].to_numpy()
        is_target = beh == result.target_name
        color = STRATEGY_COLORS[STRATEGY_WHOLE_VIDEO] if is_target else "#9AA0A6"
        # box + jittered points (each session is one observation).
        ax.boxplot([vals], positions=[i], widths=0.6, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color),
                   medianprops=dict(color=color), whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        jit = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.3
        ax.scatter(np.full(len(vals), i) + jit, vals, s=10, color=color,
                   alpha=0.6, zorder=3, edgecolors="none")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{b}\n(rank {i+1})" if b == result.target_name else b
                        for i, b in enumerate(order)], fontsize=8)
    measure_labels = (_MEASURE_LABEL_FROM_LABELS
                      if result.source == PREVALENCE_SOURCE_LABELS else _MEASURE_LABEL)
    ax.set_ylabel(measure_labels.get(result.measure, result.measure))
    sub = (f"Kruskal–Wallis p = {result.kruskal_p:.1e}   ·   "
           f"{result.target_name} < rest p = {result.target_vs_rest_p:.1e}")
    ax.set_title(f"How rare is {result.target_name}? — {result.project_id}\n{sub}",
                 fontsize=10)
    notes = []
    if result.source_caveat():
        # First, and in full: everything else on this panel is conditional on it.
        # Wrapped, because at this width the one-liner runs off the axes.
        import textwrap  # noqa: PLC0415

        notes.extend(textwrap.wrap(result.source_caveat(), width=62))
    if result.rarer_than_target:
        notes.append(f"rarer still: {', '.join(result.rarer_than_target)}")
    if result.excluded:
        notes.append(f"excluded (not validly scored): {', '.join(result.excluded)}")
    if notes:
        ax.text(0.99, 0.97, "\n".join(notes), transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#b03030")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── combined (cross-project) figures ───────────────────────────────────────
# One project answers "does clip hunting beat random for THIS rare behaviour?".
# The combined panels answer the question a reviewer actually asks: "does it beat
# random for rare behaviours *in general*?" — every project contributes its own
# rarest behaviour (see :func:`rank_behaviors_by_rarity`) as one observation, so
# each figure is a paired, within-project comparison of the same four arms.

_ARM_ORDER = (STRATEGY_ESSENCE, STRATEGY_AL, STRATEGY_UMAP, STRATEGY_RANDOM)


def _proj_label(result, labels: "dict[str, str] | None") -> str:
    name = (labels or {}).get(result.project_id, result.project_id)
    return f"{name}\n{result.behavior_name}"


def _grouped_bars(ax, groups: list[str], series: "dict[str, list[float]]",
                  colors: dict[str, str], *, fmt: str = "{:.1f}"):
    """Grouped bar chart: one x-group per project, one bar per strategy."""
    n_s = max(1, len(series))
    width = 0.8 / n_s
    x = np.arange(len(groups), dtype=float)
    for i, (strat, vals) in enumerate(series.items()):
        pos = x - 0.4 + width * (i + 0.5)
        v = np.asarray(vals, dtype=float)
        ax.bar(pos, np.nan_to_num(v, nan=0.0), width=width * 0.92,
               color=colors.get(strat, "#333333"), alpha=0.9,
               label=STRATEGY_LABELS.get(strat, strat))
        for px, pv in zip(pos, v):
            if np.isfinite(pv):
                ax.text(px, pv, " " + fmt.format(pv), ha="center", va="bottom",
                        fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=7.5)


def plot_combined_discovery(results: list[RareDiscoveryResult], out_path,
                            labels: "dict[str, str] | None" = None):
    """All projects on one axis: clips reviewed → **% of the rare behaviour found**.

    Counts are not comparable across projects (different pool sizes and positive
    counts), so the y-axis is normalised to the fraction of held-out positives
    recovered.  Thin lines are individual projects; the bold line is the mean
    across projects with a SEM band — the "does this generalise?" panel.
    """
    if not results:
        return
    plt = _fig()
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    xmax = min(max((p.n_reviewed for c in r.curves.values() for p in c.points),
                   default=0) for r in results)
    if xmax <= 0:
        plt.close(fig)
        return
    grid = np.unique(np.linspace(1, xmax, 60).astype(int))
    for strat in _ARM_ORDER:
        stack = []
        for r in results:
            cur = r.curves.get(strat)
            if cur is None or not cur.points or r.n_pos_pool <= 0:
                continue
            xs = np.array([p.n_reviewed for p in cur.points], dtype=float)
            ys = np.array([p.n_found_mean for p in cur.points], dtype=float) / r.n_pos_pool
            frac = np.interp(grid, xs, ys)
            stack.append(frac)
            ax.plot(grid, frac * 100, "-", lw=0.8, alpha=0.35,
                    color=STRATEGY_COLORS.get(strat, "#333333"), zorder=2)
        if not stack:
            continue
        arr = np.vstack(stack)
        mean = arr.mean(axis=0) * 100
        sem = (arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) * 100
               if len(arr) > 1 else np.zeros_like(mean))
        ax.plot(grid, mean, "-", lw=2.6, color=STRATEGY_COLORS.get(strat, "#333333"),
                label=f"{STRATEGY_LABELS.get(strat, strat)} (n={len(arr)})",
                zorder=4 if strat != STRATEGY_RANDOM else 3)
        ax.fill_between(grid, np.clip(mean - sem, 0, 100), np.clip(mean + sem, 0, 100),
                        color=STRATEGY_COLORS.get(strat, "#333333"), alpha=0.15, lw=0)
    ax.set_xlabel("Clips reviewed (labeling effort)")
    ax.set_ylabel("Held-out positives found (% of available)")
    ax.set_ylim(0, 100)
    names = ", ".join(_proj_label(r, labels).replace("\n", " · ") for r in results)
    ax.set_title("Rare-behavior discovery across projects — mean of "
                 f"{len(results)} projects\n{names}", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.text(0.5, -0.02, _CAVEAT, ha="center", fontsize=6.5, color="#b03030", wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def combined_enrichment_budget(results: list[RareDiscoveryResult]) -> int:
    """The largest review budget every project's curves actually reach."""
    per = [max((p.n_reviewed for c in r.curves.values() for p in c.points), default=0)
           for r in results]
    return int(min([p for p in per if p > 0] or [0]))


def plot_combined_enrichment(results: list[RareDiscoveryResult], out_path,
                             k: int | None = None,
                             labels: "dict[str, str] | None" = None):
    """Grouped bars: fold-enrichment over the behaviour's prevalence at budget ``k``.

    Enrichment (precision@k ÷ prevalence) is the one discovery number that IS
    comparable across projects — it is already normalised by how rare the target
    is in that project.  Random sits at 1.0 by construction (drawn as the
    reference line), so every bar reads directly as "× better than random".
    """
    if not results:
        return
    if k is None:
        k = min(100, combined_enrichment_budget(results))
    if not k:
        return
    plt = _fig()
    fig, ax = plt.subplots(figsize=(max(7.0, 1.9 * (len(results) + 1)), 4.8))
    arms = [s for s in _ARM_ORDER if s != STRATEGY_RANDOM]
    groups = [_proj_label(r, labels) for r in results] + ["Mean\n(all projects)"]
    series: dict[str, list[float]] = {}
    for strat in arms:
        vals = [r.enrichment_at(strat, int(k)) for r in results]
        finite = [v for v in vals if np.isfinite(v)]
        series[strat] = vals + [float(np.mean(finite)) if finite else float("nan")]
    _grouped_bars(ax, groups, series, STRATEGY_COLORS, fmt="{:.1f}×")
    ax.axhline(1.0, ls="--", lw=1.2, color=STRATEGY_COLORS[STRATEGY_RANDOM],
               zorder=1, label="Random clips (1.0×)")
    ax.set_ylabel(f"Fold-enrichment over random\n(precision@{int(k)} ÷ prevalence)")
    ax.set_title(f"How much better than random, at a {int(k)}-clip review budget?",
                 fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def combined_effort_target(results: list[RareDiscoveryResult]) -> int | None:
    """Largest positives-target every project reached on every non-random arm."""
    common: set[int] | None = None
    for r in results:
        per_arm = [{t for t, v in c.effort_to_n.items() if np.isfinite(v)}
                   for c in r.curves.values()]
        reached = set.intersection(*per_arm) if per_arm else set()
        common = reached if common is None else (common & reached)
    return max(common) if common else None


def plot_combined_savings(results: list[RareDiscoveryResult], out_path,
                          target: int | None = None,
                          labels: "dict[str, str] | None" = None):
    """Grouped bars: × fewer clips than random to collect N confirmed positives.

    The paired, within-project version of the effort figure — each project is its
    own control (its own random arm), so the projects can be averaged even though
    their absolute clip costs differ by an order of magnitude.
    """
    if not results:
        return
    if target is None:
        target = combined_effort_target(results)
    if not target:
        return
    plt = _fig()
    fig, ax = plt.subplots(figsize=(max(7.0, 1.9 * (len(results) + 1)), 4.8))
    arms = [s for s in _ARM_ORDER if s != STRATEGY_RANDOM]
    groups = [_proj_label(r, labels) for r in results] + ["Mean\n(all projects)"]
    series: dict[str, list[float]] = {}
    for strat in arms:
        vals = []
        for r in results:
            rnd = r.curves.get(STRATEGY_RANDOM)
            cur = r.curves.get(strat)
            a = cur.effort_to_n.get(int(target)) if cur else None
            b = rnd.effort_to_n.get(int(target)) if rnd else None
            vals.append(float(b / a) if a and b and a > 0 else float("nan"))
        finite = [v for v in vals if np.isfinite(v)]
        series[strat] = vals + [float(np.mean(finite)) if finite else float("nan")]
    _grouped_bars(ax, groups, series, STRATEGY_COLORS, fmt="{:.1f}×")
    ax.axhline(1.0, ls="--", lw=1.2, color=STRATEGY_COLORS[STRATEGY_RANDOM],
               zorder=1, label="Random clips (1.0×)")
    ax.set_ylabel(f"× fewer clips than random\nto find {int(target)} confirmed positives")
    ax.set_title(f"Review effort saved vs. random — {int(target)} confirmed positives",
                 fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_quality_savings(results: list[EffortToQualityResult], out_path,
                                  labels: "dict[str, str] | None" = None):
    """Grouped bars: × less labeling than random to reach a held-out quality target.

    The outcome version of :func:`plot_combined_savings` — positives found is the
    mechanism, a good model is the thing the labeling was for.  Each project uses
    the hardest target its own arms reached, named on the tick label, because a
    single fixed F1 is unreachable in one project and trivial in another.
    """
    if not results:
        return
    plt = _fig()
    fig, ax = plt.subplots(figsize=(max(7.0, 1.9 * (len(results) + 1)), 4.8))
    arms = [s for s in _ARM_ORDER if s != STRATEGY_RANDOM]
    groups, series = [], {s: [] for s in arms}
    for r in results:
        # Hardest target reached by ≥2 arms (so the comparison is a comparison).
        counts = {lab: sum(1 for c in r.curves.values() if lab in c.effort)
                  for lab in {l for c in r.curves.values() for l in c.effort}}
        cand = [l for l, n in counts.items() if n >= 2]
        if not cand:
            continue
        tgt = min(cand, key=lambda l: counts[l])
        groups.append(f"{_proj_label(r, labels)}\n{tgt}")
        for strat in arms:
            series[strat].append(r.savings_vs_random(strat, tgt) or float("nan"))
    if not groups:
        plt.close(fig)
        return
    groups.append("Mean\n(all projects)")
    for strat in arms:
        finite = [v for v in series[strat] if np.isfinite(v)]
        series[strat].append(float(np.mean(finite)) if finite else float("nan"))
    _grouped_bars(ax, groups, series, STRATEGY_COLORS, fmt="{:.1f}×")
    ax.axhline(1.0, ls="--", lw=1.2, color=STRATEGY_COLORS[STRATEGY_RANDOM],
               zorder=1, label="Random clips (1.0×)")
    ax.set_ylabel("× less labeling than random\nto reach the quality target")
    ax.set_title("Labeling saved to reach a good model — per project", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def combined_rows(results: list[RareDiscoveryResult], k: int | None = None,
                  target: int | None = None,
                  labels: "dict[str, str] | None" = None) -> list[dict]:
    """One row per (project, strategy): the two comparable cross-project numbers."""
    if not results:
        return []
    if k is None:
        k = min(100, combined_enrichment_budget(results)) or None
    if target is None:
        target = combined_effort_target(results)
    rows: list[dict] = []
    for r in results:
        rnd = r.curves.get(STRATEGY_RANDOM)
        for strat in _ARM_ORDER:
            cur = r.curves.get(strat)
            if cur is None:
                continue
            clips = cur.effort_to_n.get(int(target)) if target else None
            rclips = rnd.effort_to_n.get(int(target)) if (rnd and target) else None
            rows.append({
                "project": (labels or {}).get(r.project_id, r.project_id),
                "project_id": r.project_id,
                "behavior": r.behavior_name,
                "prevalence": r.prevalence,
                "n_pool": r.n_pool,
                "n_positives": r.n_pos_pool,
                "strategy": cur.label(),
                "budget_k": k,
                "enrichment_at_k": r.enrichment_at(strat, int(k)) if k else float("nan"),
                "effort_target_n": target,
                "clips_to_target": clips,
                "fold_fewer_clips_than_random": (
                    float(rclips / clips) if clips and rclips and clips > 0 else float("nan")),
            })
    return rows


def prism_combined_enrichment(results: list[RareDiscoveryResult],
                              k: int | None = None,
                              labels: "dict[str, str] | None" = None) -> pd.DataFrame:
    """Grouped table: rows = strategy, one replicate column per project."""
    if not results:
        return pd.DataFrame()
    if k is None:
        k = min(100, combined_enrichment_budget(results)) or 0
    rows = []
    for strat in _ARM_ORDER:
        cols = {_proj_label(r, labels).replace("\n", " · "):
                (r.enrichment_at(strat, int(k)) if k else np.nan)
                for r in results if strat in r.curves}
        if cols:  # an arm no project ran is absent, not a row of blanks
            rows.append({"Strategy": STRATEGY_LABELS.get(strat, strat), **cols})
    return pd.DataFrame(rows)


# ── tidy export rows ───────────────────────────────────────────────────────


def discovery_points_rows(result: RareDiscoveryResult) -> list[dict]:
    """Per-(strategy, budget) discovery rows for CSV export (Prism-ready)."""
    rows: list[dict] = []
    for strat, cur in result.curves.items():
        for pt in cur.points:
            rows.append({
                "project": result.project_id,
                "behavior": result.behavior_name,
                "pool": result.pool_label,
                "strategy": cur.label(),
                "clips_reviewed": pt.n_reviewed,
                "confirmed_found_mean": pt.n_found_mean,
                "confirmed_found_ci95": pt.n_found_ci,
                "n_seeds": pt.n_seeds,
            })
    return rows


def quality_points_rows(result: EffortToQualityResult) -> list[dict]:
    """Per-(strategy, budget) held-out F1/PR-AUC rows for CSV export."""
    rows: list[dict] = []
    for strat, cur in result.curves.items():
        for pt in cur.points:
            rows.append({
                "project": result.project_id,
                "behavior": result.behavior_name,
                "pool": result.pool_label,
                "strategy": cur.label(),
                "clips_reviewed": pt.n_clips,
                "n_positives_mean": pt.n_pos_mean,
                "f1_mean": pt.f1_mean, "f1_ci95": pt.f1_ci,
                "pr_auc_mean": pt.pr_auc_mean, "pr_auc_ci95": pt.pr_auc_ci,
                "n_seeds": pt.n_seeds,
            })
    return rows


def rarity_rows(result: "RarityScalingResult") -> list[dict]:
    return [{
        "project": result.project_id,
        "behavior": result.behavior_name,
        "prevalence": p.prevalence,
        "target_positives": result.target,
        "essence_clips_mean": p.essence_effort_mean,
        "essence_clips_ci95": p.essence_effort_ci,
        "random_clips_mean": p.random_effort_mean,
        "random_clips_ci95": p.random_effort_ci,
        "fewer_clips_ratio": p.ratio_mean,
        "n_pos_kept": p.n_pos_kept,
        "n_seeds": p.n_seeds,
    } for p in result.points]


# ── Prism-ready (pre-pivoted) tables ───────────────────────────────────────
# Same rules as abel/validation/prism.py: one row-title column first, replicates
# as consecutive columns (so Prism runs the test itself), no prose/JSON cells.


def prism_behavior_rarity(result: BehaviorRarityResult,
                          measure: str | None = None) -> pd.DataFrame:
    """Column table: rows = sessions, one column per behaviour, cell = the measure.

    This is the direct substrate for the rarity figure's stats — paste as a Prism
    Column table (groups = behaviours, each session a replicate) and run
    Analyze → ANOVA (or Kruskal–Wallis) → the WDS-vs-rest / omnibus p come out.

    ``measure`` defaults to the one the figure was drawn with; the per-session
    frame also carries ``bout_rate`` (the reviewer-facing "how often per minute"
    unit) and ``n_bouts``, neither of which was reachable from Prism before.
    """
    measure = measure or result.measure
    if measure not in result.per_session.columns:
        return pd.DataFrame()
    wide = result.per_session.pivot_table(
        index="session", columns="behavior", values=measure, aggfunc="first")
    # Columns rarest→common so the target sits where the figure shows it.
    order = [b for b in result.means if b in wide.columns]
    wide = wide.reindex(columns=order)
    wide.insert(0, "Session", wide.index)
    return wide.reset_index(drop=True)


def _ordered_curves(result):
    """``(label, curve)`` in canonical arm order, not dict-insertion order.

    Insertion order varies with which arms ran, so two projects could otherwise
    emit the same columns in different positions — invisible in the CSV and fatal
    when the user pastes both into one Prism table.
    """
    ordered = [(s, result.curves[s]) for s in _ARM_ORDER if s in result.curves]
    ordered += [(s, c) for s, c in result.curves.items() if s not in _ARM_ORDER]
    return [(c.label(), c) for _, c in ordered]


def prism_discovery(result: RareDiscoveryResult) -> pd.DataFrame:
    """XY table: X = clips reviewed, per-seed replicate subcolumns per strategy.

    Prism table type: XY, "Enter replicate values in side-by-side subcolumns".
    Emitting the seeds rather than the mean lets Prism draw the error bars and run
    the curve comparison itself; the previous mean-only export could only ever be
    plotted as a bare line, because a ci95 half-width is not an error format Prism
    accepts (see :func:`abel.validation.prism.sd_from_ci95`).
    """
    budgets = sorted({p.n_reviewed for c in result.curves.values() for p in c.points})
    n_rep = max((len(p.n_found_seeds) for c in result.curves.values()
                 for p in c.points), default=0)
    out = pd.DataFrame({"Clips reviewed": budgets})
    for label, cur in _ordered_curves(result):
        if n_rep:
            _replicate_block(out, label,
                             {p.n_reviewed: p.n_found_seeds for p in cur.points},
                             budgets, n_rep)
        else:  # older results carry no seeds; means keep the file pasteable
            m = {p.n_reviewed: p.n_found_mean for p in cur.points}
            out[label] = [m.get(b, np.nan) for b in budgets]
    return out


def prism_effort(result: RareDiscoveryResult) -> pd.DataFrame:
    """Grouped table: rows = strategy, per-seed clip-effort replicates per target N.

    Column set and replicate count come from the run's *configuration*, so every
    project emits the same block and the files stack.
    """
    targets = result.effort_targets or sorted(
        {t for c in result.curves.values() for t in c.effort_to_n_seeds})
    n_rep = result.n_seeds or max(
        (len(c.effort_to_n_seeds.get(t, []))
         for c in result.curves.values() for t in targets), default=0)
    rows = []
    for label, cur in _ordered_curves(result):
        row = {"Strategy": label}
        for t in targets:
            seeds = cur.effort_to_n_seeds.get(t, [])
            for i in range(n_rep):
                row[f"N={t}:{i + 1}"] = seeds[i] if i < len(seeds) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def prism_quality(result: EffortToQualityResult, metric: str = "f1") -> pd.DataFrame:
    """XY table: X = clips reviewed, per-seed held-out F1 (or PR-AUC) replicates.

    Prism table type: XY, "Enter replicate values in side-by-side subcolumns".
    """
    is_f1 = metric == "f1"
    budgets = sorted({p.n_clips for c in result.curves.values() for p in c.points})
    seeds_of = (lambda p: p.f1_seeds) if is_f1 else (lambda p: p.pr_auc_seeds)
    mean_of = (lambda p: p.f1_mean) if is_f1 else (lambda p: p.pr_auc_mean)
    n_rep = max((len(seeds_of(p)) for c in result.curves.values()
                 for p in c.points), default=0)
    out = pd.DataFrame({"Clips reviewed": budgets})
    for label, cur in _ordered_curves(result):
        if n_rep:
            _replicate_block(out, label,
                             {p.n_clips: seeds_of(p) for p in cur.points},
                             budgets, n_rep)
        else:
            m = {p.n_clips: mean_of(p) for p in cur.points}
            out[label] = [m.get(b, np.nan) for b in budgets]
    return out


def _quality_target_labels(result: EffortToQualityResult) -> list[str]:
    """Configured target labels in build order (falling back to observed keys)."""
    if result.target_labels:
        return list(result.target_labels)
    seen: list[str] = []
    for c in result.curves.values():
        for lab in c.effort_seeds:
            if lab not in seen:
                seen.append(lab)
    return seen


def prism_effort_to_quality(result: EffortToQualityResult) -> pd.DataFrame:
    """Grouped table: rows = strategy, per-seed clips-to-target replicate columns.

    A blank cell means that seed never reached the target within the clip budget
    -- real censoring, not a missing measurement. Pair with
    ``prism_effort_to_quality_reached`` before quoting a mean: an arm that reached
    a target on 2 of 5 seeds looks *faster* than one that reached it on all 5.
    """
    labels = _quality_target_labels(result)
    n_rep = result.n_seeds or max(
        (len(c.effort_seeds.get(lab, []))
         for c in result.curves.values() for lab in labels), default=0)
    rows = []
    for label, cur in _ordered_curves(result):
        row = {"Strategy": label}
        for lab in labels:
            seeds = cur.effort_seeds.get(lab, [])
            for i in range(n_rep):
                row[f"{lab}:{i + 1}"] = seeds[i] if i < len(seeds) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def prism_effort_to_quality_reached(result: EffortToQualityResult) -> pd.DataFrame:
    """Grouped table: rows = strategy, cols = target, cell = seeds that reached it.

    The denominator behind every mean in ``prism_effort_to_quality``. Without it
    the censoring is invisible and the slowest arms look best, because only their
    luckiest seeds contribute.
    """
    labels = _quality_target_labels(result)
    n_rep = result.n_seeds
    rows = []
    for label, cur in _ordered_curves(result):
        row = {"Strategy": label}
        for lab in labels:
            seeds = cur.effort_seeds.get(lab, [])
            row[lab] = int(sum(1 for v in seeds if np.isfinite(v)))
        if n_rep:
            row["Seeds run"] = int(n_rep)
        rows.append(row)
    return pd.DataFrame(rows)


def prism_rarity_scaling(result: "RarityScalingResult") -> pd.DataFrame:
    """XY table: X = prevalence, per-seed clips-to-target for essence vs random.

    Prism table type: XY, "Enter replicate values in side-by-side subcolumns".
    """
    xs = [p.prevalence for p in result.points]
    n_rep = max((max(len(p.essence_effort_seeds), len(p.random_effort_seeds))
                 for p in result.points), default=0)
    out = pd.DataFrame({"Prevalence": xs})
    if not n_rep:
        out["Essence Miner (clips)"] = [p.essence_effort_mean for p in result.points]
        out["Random clips"] = [p.random_effort_mean for p in result.points]
        return out
    for name, attr in (("Essence Miner", "essence_effort_seeds"),
                       ("Random clips", "random_effort_seeds")):
        _replicate_block(out, name,
                         {p.prevalence: getattr(p, attr) for p in result.points},
                         xs, n_rep)
    return out


def write_prism(
    out_dir,
    *,
    reviewed: RareDiscoveryResult | None = None,
    full: RareDiscoveryResult | None = None,
    rarity: "RarityScalingResult | None" = None,
    behavior_rarity: BehaviorRarityResult | None = None,
    quality: EffortToQualityResult | None = None,
    stem: str = "",
) -> list:
    """Write every available rare-discovery Prism table into ``out_dir/prism``.

    ``stem`` is appended to every filename (``…_reviewed__<stem>.csv``).  A
    multi-project run writes one set of tables *per project*, and without the
    suffix each project would silently overwrite the last one's tables.
    """
    out = Path(out_dir) / "prism"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    suffix = f"__{stem}" if stem else ""

    def _w(df, name):
        # Route through prism._write so these tables get the same treatment as the
        # rest of the bundle: ASCII headers (the effort targets carry a literal
        # ">=" that Excel/Prism would otherwise render as mojibake), a UTF-8 BOM,
        # 4-significant-figure rounding, and all-NaN column pruning.
        if df is not None and not df.empty:
            written.append(_prism._write(df, out / name.replace(".csv", f"{suffix}.csv")))

    if behavior_rarity is not None:
        # The figure's own measure keeps the plain filename; the other two ship
        # beside it so bout rate / bout counts are reachable without re-pivoting
        # the tidy CSV by hand.
        _w(prism_behavior_rarity(behavior_rarity), "prism_behavior_rarity.csv")
        for m in ("time_fraction", "bout_rate", "n_bouts"):
            if m != behavior_rarity.measure:
                _w(prism_behavior_rarity(behavior_rarity, m),
                   f"prism_behavior_rarity_{m}.csv")
    if reviewed is not None:
        _w(prism_discovery(reviewed), "prism_discovery_reviewed.csv")
        _w(prism_effort(reviewed), "prism_effort_reviewed.csv")
    if full is not None:
        _w(prism_discovery(full), "prism_discovery_fullpool.csv")
    if rarity is not None:
        _w(prism_rarity_scaling(rarity), "prism_rarity_scaling.csv")
    if quality is not None:
        _w(prism_quality(quality, "f1"), "prism_quality_f1.csv")
        _w(prism_quality(quality, "pr_auc"), "prism_quality_prauc.csv")
        _w(prism_effort_to_quality(quality), "prism_effort_to_quality.csv")
        _w(prism_effort_to_quality_reached(quality),
           "prism_effort_to_quality_reached.csv")

    if written:
        n_rep = 0
        for r in (reviewed, full):
            if r is not None:
                n_rep = max(n_rep, int(getattr(r, "n_seeds", 0) or 0))
        if quality is not None:
            n_rep = max(n_rep, int(getattr(quality, "n_seeds", 0) or 0))
        rep = n_rep or 1
        written.append(_prism.write_text(
            out / "README_PRISM_rare_discovery.txt",
            _RARE_README.format(suffix=suffix or "  (none: single-project run)",
                                rep=rep)))
    return written


# Every filename below carries the __<project> (and __<behavior>, where a project
# hunted more than one) suffix that write_prism appends -- the previous README
# named the unsuffixed stems, which matched no file actually on disk.
_RARE_README = """\
Rare-behavior discovery - Prism-ready tables
============================================
Filename suffix on this run: {suffix}
Replicate (seed) count:      {rep}

Each file is a direct paste into a new Prism table of the stated type. Blank cells
are real and mean "never reached within the clip budget" -- leave them blank; Prism
reads a blank as missing, which is what you want, and a 0 would be a lie.

prism_behavior_rarity__*.csv
    Column table; groups = behaviours, each session a replicate. Analyze ->
    Column statistics / ANOVA / Kruskal-Wallis for the "target < rest" p-value.
    Companions in the other two rarity units:
      prism_behavior_rarity_bout_rate__*.csv     bouts per minute
      prism_behavior_rarity_n_bouts__*.csv       raw bout counts
      prism_behavior_rarity_time_fraction__*.csv fraction of session time

prism_discovery_reviewed__*.csv / prism_discovery_fullpool__*.csv
    XY -> "Enter replicate values in side-by-side subcolumns", {rep} subcolumns.
    X = clips reviewed, one dataset per strategy, cells = per-seed positives found.
    Prism draws the error bars and runs the comparison from these.
    NOTE: the reviewed pool runs 4 arms and the full pool only 2 (essence vs
    random) BY DESIGN. They are different experiments -- do not paste them into
    one table.

prism_quality_f1__*.csv / prism_quality_prauc__*.csv
    XY -> replicate values, {rep} subcolumns. X = clips reviewed, cells = per-seed
    held-out F1 (or PR-AUC) of the model trained on the labels-so-far.

prism_rarity_scaling__*.csv
    XY -> replicate values, {rep} subcolumns. X = prevalence, essence vs random
    clips-to-target.

prism_effort_reviewed__*.csv
    Grouped -> replicate values. Rows = strategy, "N=<target>:<seed>" columns of
    clips-to-N. Analyze -> ANOVA / t tests across strategies.

prism_effort_to_quality__*.csv
    Grouped -> replicate values. Rows = strategy, "<target>:<seed>" columns of
    clips-to-reach-quality.
prism_effort_to_quality_reached__*.csv
    READ THIS BEFORE QUOTING A MEAN from the file above. Rows = strategy,
    cells = how many seeds ever reached that target. A mean over the 2 of {rep}
    seeds that got there makes a slow arm look fast; this is the denominator.

prism_combined_enrichment.csv
    Written once for the whole run (not per project): enrichment at the shared
    clip budget, all projects together.
"""
