"""Behaviorscape — the feature-landscape meta-analysis.

Trains one model per ``(project, behavior)`` (leakage-checked holdout split,
reusing the shared engine) and captures each model's per-feature importance
(XGBoost gain).  Importances are normalized per model, pooled across projects by
behavior name (with an optional alias map so slight naming drift is harmless),
and every surviving feature is tagged with one of five **data modalities**:

* ``pose``       — static pose geometry (angles, curvature, positions, pairwise
                   distances between body parts).
* ``kinematics`` — pose-derived motion (velocity, speed, acceleration, jerk).
* ``video``      — pixel-derived signal (optical flow, surface motion energy,
                   oscillation/appearance) — the value-add of clip-based features.
* ``context``    — relationship to the environment (distance/angle to ROIs,
                   targets, zones).
* ``social``     — inter-animal interaction (``social_*``: distance to nearest
                   animal, approach/radial velocity, heading alignment, contact) —
                   the value-add of multi-animal projects.

The resulting :class:`BehaviorscapeData` (a feature×behavior matrix + modality
map) feeds the four publication figures in :mod:`abel.validation.plots`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.validation import holdout, subsample
from abel.validation.datamodel import ProjectRef
from abel.validation.engine import run_one_config

# ── Modality taxonomy ───────────────────────────────────────────────────────
#
# Re-exported from :mod:`abel.validation.features`, which owns the single
# definition. Keeping a second copy here is exactly what let the two drift: this
# module already had a ``context`` modality while the ablation's "pose-only"
# baseline was quietly folding ROI/object features into pose (see the features
# module docstring for what that broke). One taxonomy, one place.

from abel.validation.features import (  # noqa: E402
    MODALITY_CONTEXT,
    MODALITY_KINEMATICS,
    MODALITY_ORDER,
    MODALITY_POSE,
    MODALITY_SOCIAL,
    MODALITY_VIDEO,
    classify_modality,
)

MODALITY_LABELS: dict[str, str] = {
    MODALITY_POSE: "Pose geometry",
    MODALITY_KINEMATICS: "Kinematics",
    MODALITY_VIDEO: "Video (flow / appearance)",
    MODALITY_CONTEXT: "Context (ROI / target)",
    MODALITY_SOCIAL: "Social (interaction)",
}

# Colour-blind-safe-ish, distinct in print.
MODALITY_COLORS: dict[str, str] = {
    MODALITY_POSE: "#4C72B0",        # blue
    MODALITY_KINEMATICS: "#55A868",  # green
    MODALITY_VIDEO: "#C44E52",       # red
    MODALITY_CONTEXT: "#8172B3",     # purple
    MODALITY_SOCIAL: "#DD8452",      # orange
}

# ── Per-(project, behavior) importance source ───────────────────────────────


@dataclass
class FeatureImportanceSource:
    """One model's per-feature importance (raw, pre-pooling)."""

    project_id: str
    behavior_id: str
    behavior_name: str  # raw project label (before alias mapping)
    importance: dict[str, float] = field(default_factory=dict)
    n_pos_train: int = 0
    n_features: int = 0
    degenerate: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.degenerate and not self.error and bool(self.importance)


def collect_feature_importance(
    trainer: ActiveLearningTrainerService,
    projects: list[ProjectRef],
    behaviors: dict[str, list[str]],
    *,
    min_confidence: float = 1.0,
    holdout_test_size: float = 0.25,
    holdout_seed: int = 42,
    holdout_groups: dict[str, list[str]] | None = None,
    seed: int = 4242,
    progress_cb: Callable[[str, float], None] | None = None,
) -> list[FeatureImportanceSource]:
    """Train one model per included (project, behavior) and capture importance.

    Mirrors the generalization analysis' leakage-checked holdout split so the
    importances reflect models fit exactly the way the product fits them.
    """
    holdout_groups = holdout_groups or {}
    sources: list[FeatureImportanceSource] = []

    # Total units for the progress fraction.
    total = sum(len(behaviors.get(p.project_id, [])) for p in projects) or 1
    done = 0

    for project in projects:
        bids = behaviors.get(project.project_id, [])
        if not bids:
            continue
        try:
            df = pd.read_parquet(project.training_set_path)
        except Exception as exc:  # noqa: BLE001
            for bid in bids:
                sources.append(FeatureImportanceSource(
                    project_id=project.project_id, behavior_id=str(bid),
                    behavior_name=project.behavior_label(bid),
                    error=f"load failed: {exc}",
                ))
                done += 1
            continue

        try:
            hsplit = holdout.split(
                project,
                holdout_groups=holdout_groups.get(project.project_id) or None,
                min_confidence=min_confidence,
                test_size=holdout_test_size,
                seed=holdout_seed,
                df=df,
            )
        except Exception as exc:  # noqa: BLE001
            for bid in bids:
                sources.append(FeatureImportanceSource(
                    project_id=project.project_id, behavior_id=str(bid),
                    behavior_name=project.behavior_label(bid),
                    error=f"holdout failed: {exc}",
                ))
                done += 1
            continue

        pool = hsplit.train_pool
        for bid in bids:
            name = project.behavior_label(bid)
            if progress_cb is not None:
                progress_cb(f"Feature importance — {project.project_id}: {name}",
                            done / total)
            n_pos = subsample.count_positives(pool, bid)
            n_neg = int(len(pool) - n_pos)
            res = run_one_config(
                trainer, project, str(bid), pool, hsplit.holdout,
                seed=seed, n_pos_train=n_pos, n_neg_train=n_neg,
                retain_importance=True,
            )
            sources.append(FeatureImportanceSource(
                project_id=project.project_id,
                behavior_id=str(bid),
                behavior_name=name,
                importance={k: float(v) for k, v in (res.feature_importance or {}).items()},
                n_pos_train=int(res.n_pos_train),
                n_features=int(res.n_features),
                degenerate=bool(res.degenerate),
                error=str(res.error or ""),
            ))
            done += 1

    if progress_cb is not None:
        progress_cb("Feature importance collected.", 1.0)
    return sources


# ── Pooled behaviorscape ────────────────────────────────────────────────────


def apply_alias(name: str, alias_map: dict[str, str] | None) -> str:
    """Resolve a raw behavior name to its pooled display name."""
    if not alias_map:
        return name
    return alias_map.get(name, alias_map.get(name.strip(), name))


@dataclass
class BehaviorscapeData:
    """Pooled, thresholded feature×behavior importance landscape."""

    matrix: pd.DataFrame                    # index = feature, columns = behavior
    modality: dict[str, str]                # feature -> modality
    sources: list[FeatureImportanceSource]
    pooled_members: dict[str, list[str]]    # behavior -> ["project:raw_name", ...]
    threshold: float = 0.0
    normalize: str = "fraction"
    n_features_total: int = 0
    n_features_kept: int = 0

    # Re-exported so figure code needn't import the module-level constants.
    modality_order: list[str] = field(default_factory=lambda: list(MODALITY_ORDER))
    modality_labels: dict[str, str] = field(default_factory=lambda: dict(MODALITY_LABELS))
    modality_colors: dict[str, str] = field(default_factory=lambda: dict(MODALITY_COLORS))

    @property
    def behaviors(self) -> list[str]:
        return list(self.matrix.columns)

    @property
    def features(self) -> list[str]:
        return list(self.matrix.index)

    def is_empty(self) -> bool:
        return self.matrix.empty

    @property
    def present_modalities(self) -> list[str]:
        """The modalities this landscape actually contains features for, in the
        canonical order.

        The taxonomy has five modalities, but a given set of projects need not use
        all of them — ``social`` exists only in multi-animal projects, so a run over
        single-animal projects has *zero* social features.  Figures and exports must
        key off this, not off ``MODALITY_ORDER``: a modality with no features behind
        it otherwise shows up as a phantom legend entry and an all-zero series,
        implying ABEL measured something it never measured.
        """
        used = {self.modality.get(f, MODALITY_POSE) for f in self.matrix.index}
        return [m for m in self.modality_order if m in used]

    def modality_fraction_by_behavior(self) -> pd.DataFrame:
        """Per-behavior fraction of total importance in each *present* modality."""
        present = self.present_modalities
        rows = {m: np.zeros(len(self.matrix.columns)) for m in present}
        for feat in self.matrix.index:
            m = self.modality.get(feat, MODALITY_POSE)
            if m in rows:
                rows[m] = rows[m] + self.matrix.loc[feat].to_numpy()
        out = pd.DataFrame(rows, index=self.matrix.columns)
        totals = out.sum(axis=1).replace(0.0, np.nan)
        out = out.div(totals, axis=0).fillna(0.0)
        return out

    def to_long_df(self) -> pd.DataFrame:
        """Tidy (feature, behavior, importance, modality) table for CSV export."""
        recs: list[dict] = []
        for feat in self.matrix.index:
            mod = self.modality.get(feat, MODALITY_POSE)
            for beh in self.matrix.columns:
                recs.append({
                    "feature": feat,
                    "modality": mod,
                    "behavior": beh,
                    "importance": float(self.matrix.loc[feat, beh]),
                })
        return pd.DataFrame.from_records(recs)

    def modality_fraction_long_df(self) -> pd.DataFrame:
        """Tidy per-behavior modality shares — the data behind the modality-bars figure.

        One row per (behavior, modality): ``importance_share`` is the fraction (0–1)
        of that behavior's total feature importance carried by the modality, plus a
        ``percent`` column for direct paste into a stacked-bar chart.
        """
        frac = self.modality_fraction_by_behavior()  # behaviors × modalities
        recs: list[dict] = []
        for beh in frac.index:
            for mod in MODALITY_ORDER:
                if mod not in frac.columns:
                    continue
                share = float(frac.loc[beh, mod])
                recs.append({
                    "behavior": beh,
                    "modality": mod,
                    "modality_label": MODALITY_LABELS[mod],
                    "importance_share": share,
                    "percent": round(share * 100.0, 4),
                })
        return pd.DataFrame.from_records(recs)

    def similarity_matrix_df(self) -> pd.DataFrame:
        """Behavior×behavior profile correlation — the data behind the similarity figure.

        Square matrix (behaviors as both rows and columns) of the correlation between
        behaviors' feature-importance vectors, clipped to [0, 1] exactly as plotted.
        """
        if self.matrix.empty or self.matrix.shape[1] < 2:
            return pd.DataFrame()
        behaviors = list(self.matrix.columns)
        corr = np.corrcoef(self.matrix.to_numpy(dtype=float).T)
        corr = np.clip(np.nan_to_num(corr, nan=0.0), 0.0, 1.0)
        return pd.DataFrame(corr, index=behaviors, columns=behaviors)


def distinctiveness_df(stats: "DistinctivenessStats | None") -> pd.DataFrame:
    """Tidy per-behavior distinctiveness table — the data behind the PERMANOVA figure."""
    if stats is None:
        return pd.DataFrame()
    order = sorted(stats.behaviors, key=lambda b: stats.distinctiveness[b], reverse=True)
    return pd.DataFrame.from_records([
        {
            "behavior": b,
            "distinctiveness_cosine": float(stats.distinctiveness[b]),
            "se": float(stats.err[b]),
            "n_replicates": int(stats.n_replicates[b]),
            "dominant_modality": stats.dominant_modality.get(b, ""),
        }
        for b in order
    ])


def _normalize_importance(imp: dict[str, float], how: str) -> dict[str, float]:
    if not imp:
        return {}
    vals = np.asarray(list(imp.values()), dtype=float)
    if how == "max":
        denom = float(np.nanmax(vals))
    else:  # "fraction" — share of the model's total gain
        denom = float(np.nansum(vals))
    if not np.isfinite(denom) or denom <= 0:
        return {k: 0.0 for k in imp}
    return {k: float(v) / denom for k, v in imp.items()}


def build_behaviorscape(
    sources: list[FeatureImportanceSource],
    *,
    threshold: float = 0.0,
    alias_map: dict[str, str] | None = None,
    normalize: str = "fraction",
    drop_excluded_behaviors: set[str] | None = None,
) -> BehaviorscapeData:
    """Pool importance sources into a thresholded feature×behavior matrix.

    Parameters
    ----------
    threshold:
        Keep a feature as long as it reaches ``threshold`` in **any single
        (project, behavior) model** — a feature that is dead in one project but
        important in another survives.  The test uses the per-model maximum
        importance, *not* the pooled mean (which dilution across projects could
        push below threshold).
    alias_map:
        Maps raw behavior names to a display name, harmonising label drift
        *within* a project.  Behaviors are NOT merged across projects: each
        column is scoped to its assay (``"<project> · <behavior>"``), so an
        assay's Rear and another assay's Rear stay two separate columns — they
        are independently-trained models and averaging them would invent a
        behavior no model represents.
    normalize:
        ``"fraction"`` (per-model share of total gain, default) or ``"max"``.
    drop_excluded_behaviors:
        Behavior names (aliased, un-prefixed) to omit entirely (e.g. user
        unchecked them) — applied before the assay prefix.
    """
    drop_excluded_behaviors = drop_excluded_behaviors or set()
    usable = [s for s in sources if s.ok]

    # "<project> · <behavior>" -> feature -> normalized importances (one per model)
    pooled: dict[str, dict[str, list[float]]] = {}
    members: dict[str, list[str]] = {}
    # feature -> highest normalized importance reached in ANY single model.  This
    # (not the pooled mean) drives the keep/drop threshold so a feature that is
    # dead in most projects but important in one is retained.
    source_max: dict[str, float] = {}
    for s in usable:
        beh_name = apply_alias(s.behavior_name, alias_map)
        if beh_name in drop_excluded_behaviors:
            continue
        beh = f"{s.project_id} · {beh_name}"   # assay-scoped: never pool across projects
        norm = _normalize_importance(s.importance, normalize)
        bucket = pooled.setdefault(beh, {})
        for feat, val in norm.items():
            bucket.setdefault(feat, []).append(val)
            if val > source_max.get(feat, 0.0):
                source_max[feat] = float(val)
        members.setdefault(beh, []).append(f"{s.project_id}:{s.behavior_name}")

    behaviors = sorted(pooled)
    all_features = sorted({f for bucket in pooled.values() for f in bucket})

    if not behaviors or not all_features:
        return BehaviorscapeData(
            matrix=pd.DataFrame(), modality={}, sources=sources,
            pooled_members=members, threshold=threshold, normalize=normalize,
        )

    # Mean across member models; absent feature in a behavior counts as 0.
    data = np.zeros((len(all_features), len(behaviors)), dtype=float)
    for j, beh in enumerate(behaviors):
        bucket = pooled[beh]
        for i, feat in enumerate(all_features):
            vals = bucket.get(feat)
            if vals:
                data[i, j] = float(np.mean(vals))
    matrix = pd.DataFrame(data, index=all_features, columns=behaviors)

    n_total = len(all_features)
    # Keep a feature if it clears the threshold in ANY single model (per-project
    # max), not the diluted pooled mean.
    keep = [f for f in all_features if source_max.get(f, 0.0) >= float(threshold)]
    matrix = matrix.loc[keep]
    # Drop any feature that is all-zero across the pooled view (e.g. threshold 0).
    matrix = matrix.loc[matrix.sum(axis=1) > 0]

    modality = {feat: classify_modality(feat) for feat in matrix.index}

    return BehaviorscapeData(
        matrix=matrix,
        modality=modality,
        sources=sources,
        pooled_members=members,
        threshold=float(threshold),
        normalize=normalize,
        n_features_total=n_total,
        n_features_kept=int(len(matrix.index)),
    )


# ── "Do behaviors rely on different features?" — significance testing ────────


@dataclass
class DistinctivenessStats:
    """Quantifies how distinct each behavior's feature-importance profile is.

    ``permanova`` (when present) is the headline test of the hypothesis that
    *different behaviors rely on different features*: it asks whether behavior
    identity explains the variance among the per-(project, behavior) importance
    vectors more than a random labelling would.
    """

    behaviors: list[str]
    distinctiveness: dict[str, float]      # mean cosine dist. of a behavior's reps to other centroids
    err: dict[str, float]                  # SE across that behavior's project replicates
    n_replicates: dict[str, int]
    dominant_modality: dict[str, str]
    mean_distinctiveness: float
    metric: str = "cosine"
    permanova: dict | None = None          # {pseudo_F, R2, p, n_groups, n_samples, n_perm}


def _cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance (1 − cosine similarity) over rows of ``x``."""
    norms = np.linalg.norm(x, axis=1)
    norms[norms == 0] = 1.0
    xn = x / norms[:, None]
    sim = np.clip(xn @ xn.T, -1.0, 1.0)
    return 1.0 - sim


def _permanova(dist: np.ndarray, labels: np.ndarray, *, n_perm: int, seed: int) -> dict | None:
    """One-way PERMANOVA (Anderson 2001) on a precomputed distance matrix.

    Returns the pseudo-F, R² (fraction of variance explained by the grouping),
    and a permutation p-value, or ``None`` if the design is degenerate.
    """
    n = len(labels)
    groups = list(dict.fromkeys(labels.tolist()))
    a = len(groups)
    if a < 2 or n - a < 1:
        return None
    d2 = dist ** 2
    tri = np.triu_indices(n, 1)
    sst = float(d2[tri].sum() / n)
    if sst <= 0:
        return None

    def _ssw(lab: np.ndarray) -> float:
        total = 0.0
        for g in groups:
            idx = np.where(lab == g)[0]
            ng = len(idx)
            if ng < 2:
                continue
            sub = d2[np.ix_(idx, idx)]
            total += float(sub[np.triu_indices(ng, 1)].sum() / ng)
        return total

    ssw = _ssw(labels)
    ssa = sst - ssw
    if ssw <= 0:
        return None
    f_obs = (ssa / (a - 1)) / (ssw / (n - a))

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        sw = _ssw(perm)
        if sw <= 0:
            ge += 1
            continue
        fp = ((sst - sw) / (a - 1)) / (sw / (n - a))
        if fp >= f_obs:
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {
        "pseudo_F": float(f_obs),
        "R2": float(ssa / sst),
        "p": float(p),
        "n_groups": int(a),
        "n_samples": int(n),
        "n_perm": int(n_perm),
    }


def behavior_distinctiveness_stats(
    data: BehaviorscapeData,
    *,
    n_perm: int = 999,
    seed: int = 0,
) -> DistinctivenessStats | None:
    """Rank how distinct each behavior's feature profile is.

    Uses the per-(assay, behavior) importance vectors (kept features only) as
    samples.  Per-behavior distinctiveness (cosine distance of a behavior's vector
    to every *other* behavior's centroid) ranks how specialised each behavior is.

    Behaviors are assay-scoped (never pooled across projects), so each behavior is
    a single model: replicate counts are 1, standard errors are 0, and the global
    PERMANOVA — which needs ≥2 replicates per group — does not run (``permanova`` is
    ``None``). The distinctiveness ranking remains a valid descriptive readout.
    """
    if data is None or data.is_empty():
        return None
    kept = list(data.matrix.index)
    member_to_pooled: dict[str, str] = {}
    for pooled, members in data.pooled_members.items():
        for m in members:
            member_to_pooled[m] = pooled

    by_beh: dict[str, list[np.ndarray]] = {}
    for s in data.sources:
        if not s.ok:
            continue
        pooled = member_to_pooled.get(f"{s.project_id}:{s.behavior_name}")
        if pooled is None or pooled not in data.matrix.columns:
            continue
        norm = _normalize_importance(s.importance, data.normalize)
        vec = np.array([norm.get(f, 0.0) for f in kept], dtype=float)
        if not np.any(vec > 0):
            continue
        by_beh.setdefault(pooled, []).append(vec)

    behaviors = [b for b in data.matrix.columns if b in by_beh]
    if len(behaviors) < 2:
        return None

    centroids = np.array([np.mean(np.array(by_beh[b]), axis=0) for b in behaviors])
    cnorm = np.linalg.norm(centroids, axis=1)
    cnorm[cnorm == 0] = 1.0
    centroids_unit = centroids / cnorm[:, None]

    distinctiveness: dict[str, float] = {}
    err: dict[str, float] = {}
    nrep: dict[str, int] = {}
    for bi, b in enumerate(behaviors):
        others = [j for j in range(len(behaviors)) if j != bi]
        rep_scores: list[float] = []
        for rep in by_beh[b]:
            rn = rep / (np.linalg.norm(rep) or 1.0)
            dists = [1.0 - float(np.clip(rn @ centroids_unit[j], -1.0, 1.0)) for j in others]
            rep_scores.append(float(np.mean(dists)) if dists else 0.0)
        distinctiveness[b] = float(np.mean(rep_scores))
        err[b] = (float(np.std(rep_scores, ddof=1) / np.sqrt(len(rep_scores)))
                  if len(rep_scores) > 1 else 0.0)
        nrep[b] = len(by_beh[b])

    dom = _behaviorscape_dominant_modality(data)

    permanova = None
    grouped = [b for b in behaviors if nrep[b] >= 2]
    if len(grouped) >= 2:
        xs: list[np.ndarray] = []
        labels: list[str] = []
        for b in grouped:
            for v in by_beh[b]:
                xs.append(v)
                labels.append(b)
        dist = _cosine_distance_matrix(np.array(xs))
        permanova = _permanova(dist, np.array(labels), n_perm=n_perm, seed=seed)

    return DistinctivenessStats(
        behaviors=behaviors,
        distinctiveness=distinctiveness,
        err=err,
        n_replicates=nrep,
        dominant_modality=dom,
        mean_distinctiveness=float(np.mean(list(distinctiveness.values()))),
        permanova=permanova,
    )


def _behaviorscape_dominant_modality(data: BehaviorscapeData) -> dict[str, str]:
    """Behavior -> modality with the largest importance share (for bar colouring)."""
    frac = data.modality_fraction_by_behavior()
    out: dict[str, str] = {}
    for beh in frac.index:
        row = frac.loc[beh]
        out[beh] = str(row.idxmax()) if float(row.max()) > 0 else MODALITY_ORDER[0]
    return out
