"""Feature/pipeline ablation — what does each ABEL feature add on its own?

Builds an *incremental* story rather than leave-one-out: a bare baseline (pose-only
features, no calibration, fixed model complexity, no augmentation, no co-occurring
labels), then one variant per enhancement added *on its own*, then a final variant
with every enhancement on.  All variants train on the full training pool and are
evaluated on the same fixed held-out set, so every bar's gain is measured against
the identical baseline.  Each toggle maps to a *real* config diff on ABEL's training
primitive (no re-implementation):

- video features  -> feature_cols_override = pose+video (vs pose-only baseline)
- social features -> feature_cols_override = pose+social (only if the pool has
                     ``social_*`` interaction columns, i.e. a multi-animal project)
- calibration     -> calibration_method = project's method (vs "none")
- adaptive cmplx. -> adaptive_complexity = True (vs False)
- augmentation    -> enable_feature_augmentation = True (vs False)
- co-occurring    -> allow_co_occurring_behaviors = True (only if project uses it)

Context padding and temporal refinement are representation/post-processing toggles
handled outside this in-engine pass and are not evaluated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.validation import features, subsample
from abel.validation.datamodel import CellResult, ProjectRef
from abel.validation.engine import run_one_config
from abel.validation.holdout import HoldoutSplit

# The pose-only baseline: every enhancement turned off.
BASE_OVERRIDES: dict[str, Any] = {
    "calibration_method": "none",
    "adaptive_complexity": False,
    "enable_feature_augmentation": False,
    "allow_co_occurring_behaviors": False,
}

BASELINE_CONFIG = "baseline_none"
ALL_FEATURES_CONFIG = "all_features"


@dataclass
class AblationConfig:
    name: str
    label: str                       # human-readable bar label
    description: str                 # plain-language explanation for the legend/notes
    # Feature families the config trains on: "pose" (baseline), "pose+video",
    # "pose+social", "pose+video+social", or "all" (the trainer's full numeric set).
    feature_set: str = "pose"
    overrides: dict[str, Any] = field(default_factory=dict)


def build_ablation_configs(
    project: ProjectRef, *, has_social: bool = False, has_context: bool = False,
) -> list[AblationConfig]:
    """Bare baseline + one enhancement-added-singly per feature + all enhancements.

    ``has_social`` (whether the training pool carries ``social_*`` interaction
    columns) adds the multi-animal social-feature bar; it is off for solo projects.
    ``has_context`` adds the environment/ROI bar — that family used to be folded
    into the "pose" baseline, which credited the environment's gains to pose (see
    :mod:`abel.validation.features`).
    """
    base = dict(BASE_OVERRIDES)
    configs: list[AblationConfig] = [
        AblationConfig(
            name=BASELINE_CONFIG,
            label="Baseline (pose only)",
            description="The animal's own body only — pose geometry and kinematics — "
                        "with no calibration, fixed model complexity, and no "
                        "augmentation. Excludes environment/ROI, video and social "
                        "features. The reference point every other bar is compared "
                        "against.",
            feature_set="pose",
            overrides=dict(base),
        )
    ]
    if has_context:
        configs.append(AblationConfig(
            name="add_context_features",
            label="+ Environment / ROI context",
            description="Adds environment features — ROI/zone occupancy, distance and "
                        "angle to objects/targets, arena walls — on top of the pose-only "
                        "baseline. In object-based assays this family carries the object "
                        "identity, so it must be scored separately from pose.",
            feature_set="pose+context",
            overrides=dict(base),
        ))
    if project.use_video_features:
        configs.append(AblationConfig(
            name="add_video_features",
            label="+ Video features",
            description="Adds video-derived context features (optical flow, substrate "
                        "motion, R3D embeddings) on top of the pose-only baseline.",
            feature_set="pose+video",
            overrides=dict(base),
        ))
    if has_social:
        configs.append(AblationConfig(
            name="add_social_features",
            label="+ Social features",
            description="Adds inter-animal interaction features (distance to the nearest "
                        "animal, approach velocity, heading alignment, contact state) on "
                        "top of the pose-only baseline — the multi-animal value-add.",
            feature_set="pose+social",
            overrides=dict(base),
        ))
    ov = dict(base); ov["calibration_method"] = project.calibration_method
    configs.append(AblationConfig(
        name="add_calibration",
        label="+ Probability calibration",
        description=f"Adds {project.calibration_method} probability calibration so "
                    "predicted scores better reflect true likelihoods.",
        feature_set="pose",
        overrides=ov,
    ))
    ov = dict(base); ov["adaptive_complexity"] = True
    configs.append(AblationConfig(
        name="add_adaptive_complexity",
        label="+ Adaptive model complexity",
        description="Lets the trainer scale model capacity to the amount of labeled "
                    "data instead of a fixed configuration.",
        feature_set="pose",
        overrides=ov,
    ))
    ov = dict(base); ov["enable_feature_augmentation"] = True
    configs.append(AblationConfig(
        name="add_augmentation",
        label="+ Feature augmentation",
        description="Adds augmented/perturbed training examples to improve robustness "
                    "to noise and tracking jitter.",
        feature_set="pose",
        overrides=ov,
    ))
    if project.allow_co_occurring_behaviors:
        ov = dict(base); ov["allow_co_occurring_behaviors"] = True
        configs.append(AblationConfig(
            name="add_co_occurring",
            label="+ Co-occurring behaviors",
            description="Allows clips to carry multiple simultaneous behavior labels "
                        "instead of forcing a single exclusive label.",
            feature_set="pose",
            overrides=ov,
        ))
    # Everything on = ABEL's full production pipeline for this project.
    all_ov = {
        "calibration_method": project.calibration_method,
        "adaptive_complexity": True,
        "enable_feature_augmentation": True,
        "allow_co_occurring_behaviors": project.allow_co_occurring_behaviors,
    }
    configs.append(AblationConfig(
        name=ALL_FEATURES_CONFIG,
        label="All enhancements",
        description="Every ABEL enhancement enabled together — the full production "
                    "pipeline.",
        feature_set=("all" if (project.use_video_features or has_social or has_context)
                     else "pose"),
        overrides=all_ov,
    ))
    return configs


@dataclass
class AblationResult:
    project_id: str
    behavior_id: str
    behavior_name: str
    clip_budget: int = subsample.ALL_CLIPS                     # positives trained on (-1 = all)
    order: list[str] = field(default_factory=list)            # config names, in build order
    f1_means: dict[str, float] = field(default_factory=dict)  # config -> mean F1
    f1_seeds: dict[str, list[float]] = field(default_factory=dict)  # config -> per-seed F1
    gain: dict[str, float] = field(default_factory=dict)      # config -> mean paired (F1 − baseline)
    gain_ci: dict[str, float] = field(default_factory=dict)   # config -> 95% half-width of paired gain
    gain_n: dict[str, int] = field(default_factory=dict)      # config -> # paired seeds behind the gain
    gain_p: dict[str, float] = field(default_factory=dict)    # config -> paired t-test p vs. baseline
    labels: dict[str, str] = field(default_factory=dict)      # config -> bar label
    descriptions: dict[str, str] = field(default_factory=dict)
    cells: list[CellResult] = field(default_factory=list)

    @property
    def baseline_f1(self) -> float:
        return self.f1_means.get(BASELINE_CONFIG, float("nan"))

    def is_significant(self, name: str) -> bool:
        """True when the gain's 95% CI excludes zero (distinguishable from baseline).

        Needs ≥2 paired seeds; ``|gain| > CI`` (a consistent zero-variance gain has
        CI = 0 and still counts, a single seed never does).
        """
        g = self.gain.get(name, float("nan"))
        ci = self.gain_ci.get(name, float("nan"))
        if self.gain_n.get(name, 0) < 2 or not (np.isfinite(g) and np.isfinite(ci)):
            return False
        return abs(g) > ci


def _ci95(values) -> float:
    """95% CI half-width across seeds (t-based — see :func:`metrics.ci95`).

    Previously used a 1.96 multiplier, which at the default 3 seeds is really an
    81% interval and over-called significance; the t multiplier (4.303 at df=2)
    is the honest one. Consistent with the paired t-test in :func:`_paired_p`.
    """
    from abel.validation import metrics as vmetrics  # noqa: PLC0415

    return vmetrics.ci95(values)


def _paired_p(deltas) -> float:
    """Two-sided paired t-test p on the per-seed gains — the number a manuscript
    reports, where ``is_significant`` only gives a boolean.  NaN when there are too
    few seeds or the gain is constant across them."""
    from abel.validation import metrics as vmetrics  # noqa: PLC0415

    return vmetrics.paired_p(deltas)


def budget_label(budget: int) -> str:
    """Short tag for a clip budget: ``all`` or ``n50``."""
    return "all" if budget == subsample.ALL_CLIPS else f"n{int(budget)}"


def run_ablation(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_id: str,
    holdout_split: HoldoutSplit,
    *,
    n_seeds: int = 3,
    clip_budget: int = subsample.ALL_CLIPS,
    neg_policy: str = "all",
    neg_per_pos: float = 3.0,
    progress_cb: Callable[[str], None] | None = None,
) -> AblationResult:
    """Run the ablation suite for one (project, behavior) at a given clip budget.

    ``clip_budget`` is the number of labeled positive clips to train every config on
    (``subsample.ALL_CLIPS`` = the full pool).  At a given budget all configs share the
    *same* per-seed subsample and seed, so each config's gain over the baseline is a
    paired difference — its 95% CI (across seeds) tells you whether the gain is real or
    within noise.
    """
    behavior_name = project.behavior_label(behavior_id)
    pool = holdout_split.train_pool
    group_col = holdout_split.group_col
    total_pos = subsample.count_positives(pool, behavior_id)

    has_social = bool(features.social_only_cols(pool))
    has_context = bool(features.context_only_cols(pool))

    def _resolve_feature_cols(feature_set: str) -> list[str] | None:
        """Map a config's feature-family tag to explicit override columns.

        ``"all"`` returns None so the trainer uses its full numeric set (matching
        production exactly); the family tags return an explicit pose ± video ±
        social column list so each add-on bar is cleanly isolated.
        """
        if feature_set == "all":
            return None
        fams = set(feature_set.split("+"))
        return features.select_feature_cols(
            pool,
            include_video="video" in fams,
            include_social="social" in fams,
            include_context="context" in fams,
        )

    result = AblationResult(
        project_id=project.project_id,
        behavior_id=str(behavior_id),
        behavior_name=behavior_name,
        clip_budget=int(clip_budget),
    )

    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    # Same training subsample for every config at a given (budget, seed) → paired gains.
    seed_subsets: dict[int, tuple] = {}
    for rep in range(n_seeds):
        seed = 1000 + rep
        if clip_budget == subsample.ALL_CLIPS or clip_budget >= total_pos:
            seed_subsets[rep] = (pool, int(total_pos), int(len(pool) - total_pos))
        else:
            sub, n_pos, n_neg = subsample.draw(
                pool, behavior_id, clip_budget, group_col=group_col, seed=seed,
                neg_policy=neg_policy, neg_per_pos=neg_per_pos,
            )
            seed_subsets[rep] = (sub, int(n_pos), int(n_neg))

    blabel = budget_label(clip_budget)
    for cfg in build_ablation_configs(project, has_social=has_social,
                                      has_context=has_context):
        result.order.append(cfg.name)
        result.labels[cfg.name] = cfg.label
        result.descriptions[cfg.name] = cfg.description
        fco = _resolve_feature_cols(cfg.feature_set)
        f1s: list[float] = []
        for rep in range(n_seeds):
            seed = 1000 + rep
            sub, n_pos, n_neg = seed_subsets[rep]
            _log(f"{behavior_name} [{blabel}]: ablation {cfg.name} seed {rep + 1}/{n_seeds}…")
            res = run_one_config(
                trainer, project, behavior_id, sub, holdout_split.holdout,
                seed=seed, overrides=cfg.overrides, feature_cols_override=fco,
                n_pos_train=n_pos, n_neg_train=n_neg,
            )
            result.cells.append(
                CellResult(
                    project_id=project.project_id,
                    project_name=project.name,
                    behavior_id=str(behavior_id),
                    behavior_name=behavior_name,
                    analysis="ablation",
                    config_name=cfg.name,
                    n_clips=int(n_pos),
                    seed=int(seed),
                    precision=res.precision, recall=res.recall, f1=res.f1,
                    pr_auc=res.pr_auc, cohen_kappa=res.cohen_kappa,
                    mcc=res.mcc, balanced_accuracy=res.balanced_accuracy,
                    specificity=res.specificity, roc_auc=res.roc_auc,
                    tp=res.tp, fp=res.fp, fn=res.fn, tn=res.tn,
                    n_pos_train=res.n_pos_train, n_neg_train=res.n_neg_train,
                    n_features=res.n_features,
                    elapsed_sec_fit=res.elapsed_sec_fit,
                    elapsed_sec_total=res.elapsed_sec_total,
                    degenerate=res.degenerate, error=res.error,
                )
            )
            # Use NaN for failed seeds so the per-seed lists stay aligned for pairing.
            f1s.append(res.f1 if (not res.error and np.isfinite(res.f1)) else float("nan"))
        result.f1_seeds[cfg.name] = f1s
        finite = [v for v in f1s if np.isfinite(v)]
        result.f1_means[cfg.name] = float(np.mean(finite)) if finite else float("nan")

    base_seeds = result.f1_seeds.get(BASELINE_CONFIG, [])
    for name, f1s in result.f1_seeds.items():
        if name == BASELINE_CONFIG:
            continue
        # Paired per-seed differences: config F1 − baseline F1 at the same seed/subsample.
        paired = [c - b for c, b in zip(f1s, base_seeds)
                  if np.isfinite(c) and np.isfinite(b)]
        result.gain[name] = float(np.mean(paired)) if paired else float("nan")
        result.gain_ci[name] = _ci95(paired)
        result.gain_n[name] = len(paired)
        result.gain_p[name] = _paired_p(paired)
    return result
