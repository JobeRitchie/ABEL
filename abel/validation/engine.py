"""The one shared train+evaluate primitive every analysis reuses.

Concatenates the (sub)sampled training pool with the held-out evaluation set,
hands the trainer a *precomputed split* (pool rows = train, held-out rows =
validation), and calls ABEL's real ``train_and_evaluate``.  Because held-out
rows are physically appended and flagged as validation, **no held-out row can
ever enter training** — the same mechanism guarantees zero leakage for
learning-curve, ablation, and generalization runs alike.

That guarantee covers the *model* but originally not the *calibrator*.  ABEL's
trainer fits its probability calibrator on the validation split (right for the
product — the calibrator must see unseen data, and the shipped model is refit
and CV-calibrated separately), and this engine hands it the held-out set as
that split.  Every calibrated cell was therefore scored on rows its calibrator
had been fit to.  Measured across 7 projects / 23 behaviors at a 50-clip
budget, that inflated the ablation's calibration gain from +0.064 to +0.073
(the enhancement is overwhelmingly real — the leak is ~12% of the bar, not the
bar itself) and understated held-out ECE by ~13%.  So we now carve a dedicated
**calibration slice** off the training
pool, split by group, and pass it as ``cal_idx``: the calibrator sees only pool
rows, and the held-out set stays untouched by every part of the pipeline.  The
slice comes out of the caller's budget rather than being free extra data, so
the reported ``n_pos_train`` still describes the labels the run actually
consumed.  When a pool is too small to carve a usable slice we disable
calibration for that cell instead of falling back to the leaky path.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from abel.services.active_learning_trainer_service import (
    ActiveLearningTrainerService,
    TrainingConfig,
)
from abel.validation import holdout
from abel.validation import metrics as vmetrics
from abel.validation.datamodel import ConfigEvalResult, ProjectRef

# Fraction of the training pool's groups reserved for fitting the probability
# calibrator, and the floor below which a slice isn't worth carving.
CAL_GROUP_FRAC = 0.2
CAL_MIN_ROWS = 20
CAL_MIN_POS = 3


def _carve_calibration_slice(
    pool: pd.DataFrame, group_col: str, seed: int, behavior_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``pool`` into (fit_rows, calibration_rows) along group boundaries.

    Splitting by group — not by row — keeps the calibrator off every session the
    base model trained on, so it measures the model's behaviour on genuinely
    unseen subjects the way the held-out set does.

    The slice is drawn from groups that actually contain the **target** behavior.
    ABEL's trainer is one-vs-rest: it collapses every other behavior into
    ``no_behavior``, so a slice full of varied non-target labels still arrives at
    the calibrator as a single class, and the calibrator then silently falls back
    to the validation split — reintroducing exactly the leak this exists to
    prevent.  Label *diversity* in the pool is therefore not the test; target
    positives are.

    Returns an empty calibration frame when the pool can't support one; the
    caller disables calibration rather than accepting that fallback.
    """
    empty = pool.iloc[0:0]
    if group_col not in pool.columns or len(pool) < 2 * CAL_MIN_ROWS:
        return pool, empty

    is_pos = pool["label"].astype(str).str.strip() == str(behavior_id).strip()
    groups = pool[group_col].astype(str)
    # Only groups carrying the target can seed a usable calibration slice, and we
    # must leave at least one behind so the model still sees positives.
    pos_groups = np.sort(groups[is_pos].unique())
    if len(pos_groups) < 2:
        return pool, empty

    n_cal = max(1, min(len(pos_groups) - 1, int(round(len(pos_groups) * CAL_GROUP_FRAC))))
    rng = np.random.default_rng(int(seed))
    cal_groups = set(rng.choice(pos_groups, size=n_cal, replace=False).tolist())
    in_cal = groups.isin(cal_groups)
    cal_df, fit_df = pool.loc[in_cal], pool.loc[~in_cal]

    # Both sides must remain usable: the calibrator needs enough rows and enough
    # positives to fit a sigmoid, the model needs positives left to learn from.
    if (
        len(cal_df) < CAL_MIN_ROWS
        or int(is_pos[in_cal].sum()) < CAL_MIN_POS
        or len(fit_df) < CAL_MIN_ROWS
        or int(is_pos[~in_cal].sum()) < CAL_MIN_POS
    ):
        return pool, empty
    return fit_df, cal_df


# TrainingConfig fields an ablation/learning-curve override may set.
_OVERRIDABLE = {
    "calibration_method",
    "allow_co_occurring_behaviors",
    "adaptive_complexity",
    "enable_feature_augmentation",
    "max_train_samples_per_class",
    "drop_zero_variance_features",
}


def build_config(
    project: ProjectRef,
    behavior_id: str,
    seed: int,
    overrides: dict[str, Any] | None = None,
) -> TrainingConfig:
    """Build a TrainingConfig mirroring the project, with ablation overrides."""
    overrides = overrides or {}
    cfg = TrainingConfig(
        classifier_family=project.classifier_type,
        calibration_method=project.calibration_method,
        split_strategy=project.split_strategy,
        target_label=str(behavior_id),
        random_state=int(seed),
        include_imported=False,  # validation = this project's own labels only
        allow_co_occurring_behaviors=project.allow_co_occurring_behaviors,
        model_version="__validation_tmp__",  # never written to disk by the engine
    )
    for key, val in overrides.items():
        if key in _OVERRIDABLE:
            setattr(cfg, key, val)
    return cfg


def run_one_config(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_id: str,
    train_pool_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    seed: int,
    overrides: dict[str, Any] | None = None,
    feature_cols_override: list[str] | None = None,
    n_pos_train: int = 0,
    n_neg_train: int = 0,
    retain_estimator: bool = False,
    retain_importance: bool = False,
) -> ConfigEvalResult:
    """Train on ``train_pool_df`` and evaluate on the fixed ``holdout_df``."""
    from sklearn.metrics import cohen_kappa_score

    n_pool = int(len(train_pool_df))
    n_hold = int(len(holdout_df))
    if n_pool == 0 or n_hold == 0:
        return ConfigEvalResult(
            project_id=project.project_id,
            behavior_id=str(behavior_id),
            n_pos_train=n_pos_train,
            n_neg_train=n_neg_train,
            n_features=0,
            degenerate=True,
            error="empty train pool or holdout",
        )

    cfg = build_config(project, behavior_id, seed, overrides)

    # Carve the calibrator's rows out of the training pool so it never sees the
    # held-out set (see the module docstring).  This happens for EVERY cell, not
    # just calibrated ones: the ablation compares a calibrated arm against an
    # uncalibrated baseline, and those arms are only a paired comparison if both
    # train on the same rows.  Carving only when calibrating would hand the
    # baseline 20% more training data and understate calibration's gain by as
    # much as the leak used to overstate it.  Uncalibrated cells simply leave
    # the slice unused.
    fit_df, cal_df = _carve_calibration_slice(
        train_pool_df, holdout._group_column(project.split_strategy),
        seed, str(behavior_id),
    )
    if cal_df.empty and str(cfg.calibration_method) in {"sigmoid", "isotonic"}:
        # No honest slice available — drop calibration for this cell rather than
        # let the trainer fall back to fitting it on the rows we then score.
        cfg.calibration_method = "none"

    n_fit, n_cal = int(len(fit_df)), int(len(cal_df))
    # The slice came out of the caller's budget, so the reported training counts
    # must describe the rows the base model actually saw — otherwise a learning
    # curve plots more labels than were used.
    if n_cal:
        is_pos = fit_df["label"].astype(str).str.strip() == str(behavior_id).strip()
        n_pos_train, n_neg_train = int(is_pos.sum()), int((~is_pos).sum())
    df = pd.concat([fit_df, holdout_df, cal_df], ignore_index=True)
    train_idx = np.arange(n_fit, dtype=int)
    val_idx = np.arange(n_fit, n_fit + n_hold, dtype=int)
    cal_idx = np.arange(n_fit + n_hold, n_fit + n_hold + n_cal, dtype=int)
    split = (train_idx, val_idx, cal_idx) if n_cal else (train_idx, val_idx)

    t0 = time.perf_counter()
    try:
        res = trainer.train_and_evaluate(
            df,
            cfg,
            project_root=project.root,
            precomputed_split=split,
            feature_cols_override=feature_cols_override,
        )
    except Exception as exc:  # noqa: BLE001 — surface as a degenerate cell
        return ConfigEvalResult(
            project_id=project.project_id,
            behavior_id=str(behavior_id),
            n_pos_train=n_pos_train,
            n_neg_train=n_neg_train,
            n_features=0,
            elapsed_sec_total=float(time.perf_counter() - t0),
            degenerate=True,
            error=f"{type(exc).__name__}: {exc}",
        )
    total = float(time.perf_counter() - t0)

    metrics = res.metrics
    ti = res.target_idx
    # Binary target-vs-rest arrays for PR curves / confusion / kappa.
    y_true = y_score = y_pred = None
    kappa = float("nan")
    mcc = bal_acc = spec = roc = float("nan")
    tp = fp = fn = tn = 0
    if ti is not None and 0 <= int(ti) < res.val_probs.shape[1]:
        y_true = (res.y_val == int(ti)).astype(int)
        y_pred = (res.val_preds == int(ti)).astype(int)
        y_score = res.val_probs[:, int(ti)]
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        if not res.degenerate_val and len(set(y_true.tolist())) > 1:
            try:
                kappa = float(cohen_kappa_score(y_true, y_pred))
            except Exception:
                kappa = float("nan")
        # Imbalance-robust classifier summaries (MCC / balanced accuracy /
        # specificity / ROC-AUC) reported alongside F1 for publication.
        mcc = vmetrics.matthews_corrcoef(y_true, y_pred)
        bal_acc = vmetrics.balanced_accuracy(y_true, y_pred)
        spec = vmetrics.specificity(y_true, y_pred)
        roc = vmetrics.roc_auc(y_true, y_score)

    out = ConfigEvalResult(
        project_id=project.project_id,
        behavior_id=str(behavior_id),
        n_pos_train=n_pos_train,
        n_neg_train=n_neg_train,
        n_features=int(metrics.get("n_features", 0)),
        precision=float(metrics.get("precision", float("nan"))),
        recall=float(metrics.get("recall", float("nan"))),
        f1=float(metrics.get("f1", float("nan"))),
        pr_auc=float(metrics.get("pr_auc", float("nan"))),
        cohen_kappa=kappa,
        mcc=mcc, balanced_accuracy=bal_acc, specificity=spec, roc_auc=roc,
        tp=tp, fp=fp, fn=fn, tn=tn,
        elapsed_sec_fit=float(res.elapsed_sec),
        elapsed_sec_total=total,
        degenerate=bool(res.degenerate_val),
        y_true=y_true,
        y_score=y_score,
        y_pred=y_pred,
        confusion_matrix=metrics.get("confusion_matrix"),
    )
    if retain_estimator:
        out.fitted_estimator = res.fitted_estimator
        out.val_meta = res.val_meta
    if retain_importance:
        imp = getattr(res, "feature_importance", None)
        out.feature_importance = dict(imp) if imp else {}
    return out


# Overrides that make a fit *symmetric* across the two classes: no target-biased
# feature augmentation (which injects synthetic copies of the target class) and no
# adaptive complexity/weighting keyed to the target. Required whenever a result is
# read as a statement about two behaviors *relative to each other* (an "A vs B
# separability" score), otherwise the answer tilts toward whichever behavior
# happened to be named the target.
#
# NOTE: ABEL's trainer is strictly one-vs-rest — when ``target_label`` is set it
# remaps every other behavior to ``no_behavior`` (see the "Collapse alternate-
# behavior labels into negatives" block in the trainer). There is therefore no
# multiclass model to read a confusion matrix off. Pairwise discrimination works
# *with* that design rather than against it: hand the trainer a frame containing
# only behaviors A and B, with A as the target, and B becomes the negative class —
# yielding a genuine binary A-vs-B model built by the shipped training code.
SYMMETRIC_FIT_OVERRIDES: dict[str, Any] = {
    "enable_feature_augmentation": False,
    "adaptive_complexity": False,
}
