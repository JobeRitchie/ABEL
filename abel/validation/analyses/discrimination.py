"""Pairwise behavior discrimination — which feature families let ABEL tell two
*similar* behaviors apart?

The standard ablation (:mod:`abel.validation.analyses.ablation`) asks a
**detection** question: with feature family F on, how well is behavior X found
against everything else?  Because "everything else" is dominated by easy
negatives (no_behavior, locomotion, …), a feature family can look worthless there
while doing the one job that actually matters scientifically — separating the two
behaviors a human scorer would agonise over (Freeze vs. Groom; Sniff vs. Eat).

This module asks the **discrimination** question instead.  For every pair of
behaviors (A, B) in a project:

1. Restrict the training pool *and* the held-out set to clips labeled exactly A
   or exactly B (co-occurring "A|B" clips are excluded — they are not an
   either/or decision).
2. Train a binary A-vs-B classifier once per feature family, all families sharing
   the same seed and the same clips, so each family's effect is a **paired**
   difference.
3. Score separability on the held-out A/B clips with ROC-AUC (threshold-free —
   it measures how well the two classes are *ordered apart*, and is invariant to
   probability calibration), plus balanced accuracy and MCC.

The headline output is a **behavior × behavior separability matrix** and its
**Δ matrix** (e.g. what adding video features buys for each pair) — so the answer
to "does video disambiguate freezing from grooming?" is read straight off the
figure, per pair, rather than inferred from a single target-vs-rest bar.

**Why the ladder is feature families only.**  Calibration cannot change ROC-AUC
(it is a monotone transform of the scores), and augmentation / adaptive complexity
are keyed to the *target* class, so switching them on would inflate whichever
behavior was nominally the target and make an "A vs B" number asymmetric.  Every
fit here therefore uses :data:`engine.SYMMETRIC_FIT_OVERRIDES`, leaving the
feature set as the only thing that varies.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.validation import features
from abel.validation import metrics as vmetrics
from abel.validation.datamodel import CellResult, ProjectRef
from abel.validation.engine import SYMMETRIC_FIT_OVERRIDES, run_one_config
from abel.validation.holdout import HoldoutSplit

# A pair needs at least this many clips of *each* behavior to be scorable.
MIN_TRAIN_PER_CLASS = 8
MIN_HOLDOUT_PER_CLASS = 3

# Minimum pose-only error (1 − AUC) a pair must still carry before an
# error-reduction ratio is even defined: below this the baseline has essentially no
# misranked pairs left and the ratio is a division by noise. This is only a
# divide-by-zero guard, NOT a significance filter — a hard 0.005 cliff was doing
# the latter job badly, suppressing a *significant* 75% error reduction
# (Approach Familiar vs Rear, headroom 0.0041) and 60% of TMT's pairs on a purely
# arbitrary threshold. Whether a gain is real is decided by `is_significant()`
# (the paired-difference CI), which is what the figures gate on.
MIN_HEADROOM = 0.002

BASELINE_FEATURE_SET = "pose_only"
CONTEXT_FEATURE_SET = "pose_context"
VIDEO_FEATURE_SET = "pose_video"
SOCIAL_FEATURE_SET = "pose_social"
ALL_FEATURE_SET = "all_features"

# Display label per feature set.  The GUI names its figure views from this, so a
# renamed rung stays in step with the figures the run writes.
FEATURE_SET_LABELS = {
    BASELINE_FEATURE_SET: "Pose only",
    CONTEXT_FEATURE_SET: "+ Context / ROI",
    VIDEO_FEATURE_SET: "+ Video",
    SOCIAL_FEATURE_SET: "+ Social",
    ALL_FEATURE_SET: "All features",
}


@dataclass
class FeatureSetSpec:
    name: str        # cell/config key
    label: str       # bar / column label
    tag: str         # feature-family tag understood by features.select_feature_cols


def build_feature_sets(
    project: ProjectRef,
    *,
    has_social: bool = False,
    has_video: bool = True,
    has_context: bool = True,
) -> list[FeatureSetSpec]:
    """The feature-family ladder for the discrimination ablation.

    ``pose`` is the animal's body ALONE — environment/ROI features are their own
    rung. Without that separation the baseline can tell two objects apart using a
    single object-distance column and every pair involving object identity reports
    as trivially solved (see :mod:`abel.validation.features`).
    """
    def _spec(name: str, tag: str) -> FeatureSetSpec:
        return FeatureSetSpec(name, FEATURE_SET_LABELS[name], tag)

    specs = [_spec(BASELINE_FEATURE_SET, "pose")]
    if has_context:
        specs.append(_spec(CONTEXT_FEATURE_SET, "pose+context"))
    if project.use_video_features and has_video:
        specs.append(_spec(VIDEO_FEATURE_SET, "pose+video"))
    if has_social:
        specs.append(_spec(SOCIAL_FEATURE_SET, "pose+social"))
    # "All features" is only worth training when it is not a bit-exact duplicate of
    # a rung we already ran: with no social and no context, all == pose+video.
    extras = sum([bool(has_context),
                  bool(project.use_video_features and has_video),
                  bool(has_social)])
    if extras > 1:
        specs.append(_spec(ALL_FEATURE_SET, "all"))
    return specs


@dataclass
class PairResult:
    """A-vs-B separability under each feature family."""

    project_id: str
    behavior_a: str
    behavior_b: str
    name_a: str
    name_b: str
    n_train_a: int = 0
    n_train_b: int = 0
    n_hold_a: int = 0
    n_hold_b: int = 0

    order: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    auc: dict[str, float] = field(default_factory=dict)          # mean ROC-AUC
    auc_seeds: dict[str, list[float]] = field(default_factory=dict)
    balanced_accuracy: dict[str, float] = field(default_factory=dict)
    mcc: dict[str, float] = field(default_factory=dict)
    f1: dict[str, float] = field(default_factory=dict)

    gain: dict[str, float] = field(default_factory=dict)         # paired ΔAUC vs pose-only
    gain_ci: dict[str, float] = field(default_factory=dict)
    gain_n: dict[str, int] = field(default_factory=dict)
    # Exact two-sided paired-t p per family. `is_significant` gives a boolean, which
    # is not what a manuscript prints, and a volcano needs the p on its y-axis.
    gain_p: dict[str, float] = field(default_factory=dict)

    cells: list[CellResult] = field(default_factory=list)
    error: str = ""

    @property
    def pair_label(self) -> str:
        return f"{self.name_a} vs {self.name_b}"

    @property
    def project_name(self) -> str:
        """Assay display name, recovered from the cells this pair produced.

        `PairResult` is keyed by `project_id`; the readable name only ever reaches
        it through :class:`CellResult`.  Falls back to the id so a pair that errored
        before training (no cells) still labels itself.
        """
        for c in self.cells:
            if getattr(c, "project_name", ""):
                return str(c.project_name)
        return str(self.project_id)

    def best_single_family(self) -> str:
        """The single add-on family that removes the most pose-only error.

        Excludes ``all_features``: that rung is the *union* of the families, so
        attributing a pair's rescue to it answers "do more features help" rather
        than "WHICH modality disambiguates this pair" — the question the pairwise
        design exists to ask.  Returns "" when no single family scored.
        """
        best, best_er = "", float("-inf")
        for name in self.order:
            if name in (BASELINE_FEATURE_SET, ALL_FEATURE_SET):
                continue
            er = self.error_reduction(name)
            if np.isfinite(er) and er > best_er:
                best, best_er = name, er
        return best

    @property
    def baseline_auc(self) -> float:
        return self.auc.get(BASELINE_FEATURE_SET, float("nan"))

    def is_significant(self, name: str) -> bool:
        """Paired gain's 95% CI excludes zero (needs ≥2 seeds).

        Note this tests *consistency*, not importance: a tiny gain reproduced
        across every seed is "significant" here. Read it together with
        :meth:`error_reduction`, which carries the magnitude.
        """
        g = self.gain.get(name, float("nan"))
        ci = self.gain_ci.get(name, float("nan"))
        if self.gain_n.get(name, 0) < 2 or not (np.isfinite(g) and np.isfinite(ci)):
            return False
        return abs(g) > ci

    def error_reduction(self, name: str) -> float:
        """Share of the pose-only baseline's *remaining error* this family removes.

        ``(auc_f − auc_pose) / (1 − auc_pose)``. Pairwise AUC saturates near 1.0 for
        behaviors that barely resemble each other, and a raw ΔAUC of +0.004 reads as
        nothing there — yet at a baseline of 0.984 it has closed a quarter of the
        gap to perfect. This normalises the gain by the headroom that actually
        existed, so easy pairs stop drowning out the real effects.

        Returns NaN when the baseline leaves less than :data:`MIN_HEADROOM` of error
        to remove. Without that floor the ratio explodes on already-perfect pairs —
        wiping out 0.0002 of 0.0002 error scores a triumphant "+100%" that is pure
        noise, and it would dominate the figure. A pair the pose baseline already
        solves simply has no discrimination question left to ask.
        """
        base = self.baseline_auc
        cur = self.auc.get(name, float("nan"))
        if not (np.isfinite(base) and np.isfinite(cur)):
            return float("nan")
        headroom = 1.0 - base
        if headroom < MIN_HEADROOM:
            return float("nan")
        return float((cur - base) / headroom)


def _ci95(values) -> float:
    """95% CI half-width across seeds (t-based — see :func:`metrics.ci95`)."""
    return vmetrics.ci95(values)


def _exact_label_mask(df: pd.DataFrame, behavior_id: str) -> pd.Series:
    """Clips labeled *exactly* this behavior (excludes co-occurring "a|b" labels)."""
    return df["label"].astype(str).str.strip() == str(behavior_id).strip()


# ── Candidate ranking (training-free) ──────────────────────────────────────


def rank_pairs_by_proximity(
    pool: pd.DataFrame,
    behavior_ids: list[str],
    *,
    exclude: tuple[str, ...] = ("no_behavior",),
) -> pd.DataFrame:
    """Rank behavior pairs by how *close together* they sit in feature space.

    A training-free proxy for confusability: standardise the pose feature columns,
    take each behavior's centroid, and measure pairwise Euclidean distance.  Pairs
    whose centroids nearly coincide are the ones a classifier is most likely to
    mix up.

    This exists only to decide **which pairs survive the ``max_pairs`` cap** when a
    project has many behaviors — it is a cheap pre-filter, not a result.  The real,
    trained answer to "which behaviors does ABEL confuse" is the pose-only
    separability column of :func:`confusable_pairs_table`.
    """
    cols = ["behavior_a", "behavior_b", "centroid_distance"]
    ids = [str(b) for b in behavior_ids if str(b) not in exclude]
    if len(ids) < 2 or pool is None or pool.empty:
        return pd.DataFrame(columns=cols)
    feat = features.pose_only_cols(pool)
    if not feat:
        return pd.DataFrame(columns=cols)

    x = pool[feat].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    x = (x - x.mean(axis=0)) / sd

    labels = pool["label"].astype(str).str.strip().to_numpy()
    centroids: dict[str, np.ndarray] = {}
    for bid in ids:
        sel = labels == bid
        if sel.sum() > 0:
            centroids[bid] = x[sel].mean(axis=0)

    rows = []
    for a, b in itertools.combinations(sorted(centroids), 2):
        rows.append({
            "behavior_a": a, "behavior_b": b,
            "centroid_distance": float(np.linalg.norm(centroids[a] - centroids[b])),
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    # Closest first — those are the pairs worth spending training budget on.
    return pd.DataFrame(rows, columns=cols).sort_values(
        "centroid_distance"
    ).reset_index(drop=True)


def select_pairs(
    behavior_ids: list[str],
    *,
    ranked: pd.DataFrame | None = None,
    max_pairs: int = 15,
) -> list[tuple[str, str]]:
    """All pairs among ``behavior_ids``, capped at ``max_pairs``.

    When the cap bites, ``ranked`` (best-first, e.g. from
    :func:`rank_pairs_by_proximity`) decides which pairs survive.  Without it,
    pairs are taken in stable order.
    """
    ids = [str(b) for b in behavior_ids if str(b) != "no_behavior"]
    all_pairs = [(a, b) for a, b in itertools.combinations(ids, 2)]
    if len(all_pairs) <= max_pairs or ranked is None or ranked.empty:
        return all_pairs[:max_pairs]

    wanted = {frozenset(p) for p in all_pairs}
    ordered: list[tuple[str, str]] = []
    for _, row in ranked.iterrows():
        key = frozenset({str(row["behavior_a"]), str(row["behavior_b"])})
        if key in wanted and key not in {frozenset(p) for p in ordered}:
            ordered.append((str(row["behavior_a"]), str(row["behavior_b"])))
        if len(ordered) >= max_pairs:
            break
    for p in all_pairs:  # top up if discovery missed some
        if len(ordered) >= max_pairs:
            break
        if frozenset(p) not in {frozenset(q) for q in ordered}:
            ordered.append(p)
    return ordered[:max_pairs]


# ── The pairwise discrimination ablation ───────────────────────────────────


def run_pair_discrimination(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_a: str,
    behavior_b: str,
    holdout_split: HoldoutSplit,
    *,
    n_seeds: int = 3,
    progress_cb: Callable[[str], None] | None = None,
) -> PairResult:
    """Train A-vs-B under each feature family on the same clips; report separability."""
    name_a = project.behavior_label(behavior_a)
    name_b = project.behavior_label(behavior_b)
    res = PairResult(
        project_id=project.project_id,
        behavior_a=str(behavior_a), behavior_b=str(behavior_b),
        name_a=name_a, name_b=name_b,
    )

    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    pool, hold = holdout_split.train_pool, holdout_split.holdout
    pa, pb = _exact_label_mask(pool, behavior_a), _exact_label_mask(pool, behavior_b)
    ha, hb = _exact_label_mask(hold, behavior_a), _exact_label_mask(hold, behavior_b)
    res.n_train_a, res.n_train_b = int(pa.sum()), int(pb.sum())
    res.n_hold_a, res.n_hold_b = int(ha.sum()), int(hb.sum())

    if min(res.n_train_a, res.n_train_b) < MIN_TRAIN_PER_CLASS:
        res.error = (f"need >={MIN_TRAIN_PER_CLASS} training clips of each behavior "
                     f"(have {res.n_train_a}/{res.n_train_b})")
        return res
    if min(res.n_hold_a, res.n_hold_b) < MIN_HOLDOUT_PER_CLASS:
        res.error = (f"need >={MIN_HOLDOUT_PER_CLASS} held-out clips of each behavior "
                     f"(have {res.n_hold_a}/{res.n_hold_b})")
        return res

    pool_ab = pool.loc[pa | pb].reset_index(drop=True)
    hold_ab = hold.loc[ha | hb].reset_index(drop=True)

    has_social = bool(features.social_only_cols(pool_ab))
    has_video = bool(features.video_only_cols(pool_ab))
    has_context = bool(features.context_only_cols(pool_ab))
    specs = build_feature_sets(project, has_social=has_social, has_video=has_video,
                               has_context=has_context)

    def _cols(tag: str) -> list[str] | None:
        if tag == "all":
            return None  # trainer's full numeric set
        fams = set(tag.split("+"))
        return features.select_feature_cols(
            pool_ab,
            include_video="video" in fams,
            include_social="social" in fams,
            include_context="context" in fams,
        )

    for spec in specs:
        res.order.append(spec.name)
        res.labels[spec.name] = spec.label
        fco = _cols(spec.tag)
        aucs, bals, mccs, f1s = [], [], [], []
        for rep in range(n_seeds):
            seed = 3000 + rep
            _log(f"{name_a} vs {name_b}: {spec.name} seed {rep + 1}/{n_seeds}…")
            r = run_one_config(
                trainer, project, behavior_a, pool_ab, hold_ab,
                seed=seed, overrides=dict(SYMMETRIC_FIT_OVERRIDES),
                feature_cols_override=fco,
                n_pos_train=res.n_train_a, n_neg_train=res.n_train_b,
            )
            res.cells.append(CellResult(
                project_id=project.project_id,
                project_name=project.name,
                behavior_id=f"{behavior_a}|{behavior_b}",
                behavior_name=f"{name_a} vs {name_b}",
                analysis="discrimination",
                config_name=spec.name,
                n_clips=int(res.n_train_a + res.n_train_b),
                seed=int(seed),
                precision=r.precision, recall=r.recall, f1=r.f1,
                pr_auc=r.pr_auc, cohen_kappa=r.cohen_kappa,
                mcc=r.mcc, balanced_accuracy=r.balanced_accuracy,
                specificity=r.specificity, roc_auc=r.roc_auc,
                tp=r.tp, fp=r.fp, fn=r.fn, tn=r.tn,
                n_pos_train=r.n_pos_train, n_neg_train=r.n_neg_train,
                n_features=r.n_features,
                elapsed_sec_fit=r.elapsed_sec_fit,
                elapsed_sec_total=r.elapsed_sec_total,
                degenerate=r.degenerate, error=r.error,
            ))
            ok = not r.error
            # NaN keeps the per-seed lists aligned so gains stay paired.
            aucs.append(r.roc_auc if ok else float("nan"))
            bals.append(r.balanced_accuracy if ok else float("nan"))
            mccs.append(r.mcc if ok else float("nan"))
            f1s.append(r.f1 if ok else float("nan"))

        res.auc_seeds[spec.name] = aucs
        for store, vals in (
            (res.auc, aucs), (res.balanced_accuracy, bals), (res.mcc, mccs), (res.f1, f1s),
        ):
            finite = [v for v in vals if np.isfinite(v)]
            store[spec.name] = float(np.mean(finite)) if finite else float("nan")

    base = res.auc_seeds.get(BASELINE_FEATURE_SET, [])
    for name, vals in res.auc_seeds.items():
        if name == BASELINE_FEATURE_SET:
            continue
        paired = [v - b for v, b in zip(vals, base) if np.isfinite(v) and np.isfinite(b)]
        res.gain[name] = float(np.mean(paired)) if paired else float("nan")
        res.gain_ci[name] = _ci95(paired)
        res.gain_n[name] = len(paired)
        res.gain_p[name] = vmetrics.paired_p(paired)
    return res


def run_discrimination(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_ids: list[str],
    holdout_split: HoldoutSplit,
    *,
    n_seeds: int = 3,
    max_pairs: int = 15,
    progress_cb: Callable[[str], None] | None = None,
) -> list[PairResult]:
    """Run the discrimination ablation over all behavior pairs in one project."""
    ranked = rank_pairs_by_proximity(holdout_split.train_pool, behavior_ids)
    pairs = select_pairs(behavior_ids, ranked=ranked, max_pairs=max_pairs)
    return [
        run_pair_discrimination(
            trainer, project, a, b, holdout_split,
            n_seeds=n_seeds, progress_cb=progress_cb,
        )
        for a, b in pairs
    ]


# ── Tidy exports ───────────────────────────────────────────────────────────


def discrimination_rows(results: list[PairResult]) -> pd.DataFrame:
    """One row per (pair, feature family): separability + paired gain over pose-only.

    This is the single source both the per-project figures and the pooled
    :func:`abel.validation.plots.discrimination_landscape` read from, so it carries
    the pair-level context each row needs to be plotted on its own (``pose_only_auc``,
    ``headroom``, ``best_family``) rather than requiring a re-join back to the
    ``PairResult`` objects.
    """
    rows = []
    for r in results:
        if r.error:
            rows.append({
                "project": r.project_id, "project_name": r.project_name,
                "pair": r.pair_label,
                "behavior_a": r.name_a, "behavior_b": r.name_b,
                "feature_set": "", "label": "", "roc_auc": float("nan"),
                "balanced_accuracy": float("nan"), "mcc": float("nan"), "f1": float("nan"),
                "pose_only_auc": float("nan"), "headroom": float("nan"),
                "auc_gain_vs_pose": float("nan"), "auc_gain_ci95": float("nan"),
                "p_value": float("nan"), "n_seeds": 0,
                "error_reduction": float("nan"), "significant": "", "best_family": False,
                "n_train": r.n_train_a + r.n_train_b,
                "n_holdout": r.n_hold_a + r.n_hold_b,
                "error": r.error,
            })
            continue
        best = r.best_single_family()
        for name in r.order:
            is_base = name == BASELINE_FEATURE_SET
            rows.append({
                "project": r.project_id,
                "project_name": r.project_name,
                "pair": r.pair_label,
                "behavior_a": r.name_a,
                "behavior_b": r.name_b,
                "feature_set": name,
                "label": r.labels.get(name, name),
                "roc_auc": r.auc.get(name, float("nan")),
                "balanced_accuracy": r.balanced_accuracy.get(name, float("nan")),
                "mcc": r.mcc.get(name, float("nan")),
                "f1": r.f1.get(name, float("nan")),
                # Pair-level context, repeated on every row of the pair: the
                # difficulty this family was working against.
                "pose_only_auc": r.baseline_auc,
                "headroom": 1.0 - r.baseline_auc,
                "auc_gain_vs_pose": 0.0 if is_base else r.gain.get(name, float("nan")),
                "auc_gain_ci95": 0.0 if is_base else r.gain_ci.get(name, float("nan")),
                "p_value": float("nan") if is_base else r.gain_p.get(name, float("nan")),
                "n_seeds": len(r.auc_seeds.get(name, [])),
                # Fraction of the baseline's remaining error the family removes —
                # the honest magnitude once AUC is near ceiling.
                "error_reduction": 0.0 if is_base else r.error_reduction(name),
                "significant": "" if is_base else bool(r.is_significant(name)),
                # Marks the one single-family row per pair that the landscape panel
                # plots. False on the baseline and on all_features by construction.
                "best_family": bool(name == best and not is_base),
                "n_train": r.n_train_a + r.n_train_b,
                "n_holdout": r.n_hold_a + r.n_hold_b,
                "error": "",
            })
    return pd.DataFrame(rows)


def discrimination_seed_rows(results: list[PairResult]) -> pd.DataFrame:
    """One row per (pair, feature family, seed): the raw held-out ROC-AUC.

    The means and CIs above are summaries; these are the replicates behind them.
    Exported so Prism (or a reviewer) can re-run the paired test on the same numbers
    rather than taking our p on trust.
    """
    rows = []
    for r in results:
        if r.error:
            continue
        for name in r.order:
            for rep, auc in enumerate(r.auc_seeds.get(name, [])):
                rows.append({
                    "project": r.project_id,
                    "project_name": r.project_name,
                    "pair": r.pair_label,
                    "feature_set": name,
                    "label": r.labels.get(name, name),
                    "seed_index": rep + 1,
                    "roc_auc": float(auc),
                })
    return pd.DataFrame(rows)


def confusable_pairs_table(results: list[PairResult]) -> pd.DataFrame:
    """Behavior pairs ranked by how hard they are to tell apart (hardest first).

    The trained, honest confusability ranking: pose-only A-vs-B ROC-AUC, with what
    each feature family recovers on top of it.  A pair near 0.5 is a coin flip; a
    pair near 1.0 is trivially separable and can safely be ignored.
    """
    rows = []
    for r in results:
        if r.error:
            continue
        row = {
            "project": r.project_id,
            "pair": r.pair_label,
            "pose_only_auc": r.baseline_auc,
            "n_holdout": r.n_hold_a + r.n_hold_b,
        }
        for name in r.order:
            if name == BASELINE_FEATURE_SET:
                continue
            row[f"{name}_auc"] = r.auc.get(name, float("nan"))
            row[f"{name}_error_reduction"] = r.error_reduction(name)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("pose_only_auc").reset_index(drop=True)


def separability_matrix(
    results: list[PairResult], *, feature_set: str = BASELINE_FEATURE_SET,
) -> pd.DataFrame:
    """Symmetric behavior×behavior ROC-AUC matrix for one feature family."""
    return _matrix(results, lambda r: r.auc.get(feature_set, float("nan")))


def gain_matrix(
    results: list[PairResult], *, feature_set: str = VIDEO_FEATURE_SET,
) -> pd.DataFrame:
    """Symmetric behavior×behavior ΔROC-AUC matrix (feature family − pose-only)."""
    return _matrix(results, lambda r: r.gain.get(feature_set, float("nan")))


def error_reduction_matrix(
    results: list[PairResult], *, feature_set: str = VIDEO_FEATURE_SET,
) -> pd.DataFrame:
    """Symmetric behavior×behavior matrix of :meth:`PairResult.error_reduction`.

    The ceiling-corrected companion to :func:`gain_matrix`: on pairs the pose
    baseline already nails, a raw ΔAUC of +0.004 is invisible, but as a share of
    the error that was left it can be a 25-45% improvement.  This is the matrix
    worth plotting.
    """
    return _matrix(results, lambda r: r.error_reduction(feature_set))


def _matrix(results: list[PairResult], value_of) -> pd.DataFrame:
    names = sorted({n for r in results for n in (r.name_a, r.name_b)})
    mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for r in results:
        if r.error:
            continue
        v = value_of(r)
        mat.loc[r.name_a, r.name_b] = v
        mat.loc[r.name_b, r.name_a] = v
    return mat
