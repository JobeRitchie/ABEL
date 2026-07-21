"""Leave-one-subject-out cross-validation for behavior models.

A single random 2-mouse holdout makes the reported number hostage to *which*
mice land in validation — if the hardest mouse is drawn, the behavior looks
broken even when it generalizes fine to the others. Leave-one-subject-out (LOSO)
CV removes that lottery: it trains N models, each holding out exactly one subject,
pools every subject's held-out predictions, and reports one stable, honest
generalization number per behavior (raw and after temporal refinement).

This reuses the validation suite's leakage-checked per-fold primitive
(:func:`abel.validation.engine.run_one_config`) and the shared refinement scorer
(:func:`abel.temporal_refinement.refined_eval.score_raw_and_refined`), so LOSO
numbers agree with the single-split Validation-tab math by construction.

Running this trains one model per subject per behavior, so it is compute-heavy —
intended as an on-demand analysis, not part of the standard pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.temporal_refinement.refined_eval import (
    _frames_from_segment_ids,
    load_temporal_settings,
    score_raw_and_refined,
)
from abel.validation.datamodel import ProjectRef
from abel.validation.engine import run_one_config

logger = logging.getLogger("abel.validation.loso")

# Labels that train the model but must never be evaluated on: temporal-review
# corrections and cross-project imports (see the trainer split for the rationale).
_REFINE_ONLY_EXACT = {"temporal_feedback"}
_REFINE_ONLY_PREFIX = ("imported:",)


def _is_refine_only(label_source: pd.Series) -> pd.Series:
    s = label_source.astype(str)
    mask = s.isin(_REFINE_ONLY_EXACT)
    for pref in _REFINE_ONLY_PREFIX:
        mask = mask | s.str.startswith(pref)
    return mask


def _mean_std_sem(values: list[float]) -> tuple[float, float, float]:
    """Mean, sample std (ddof=1) and SEM = s/sqrt(n) across folds, NaN-aware.

    SEM matches ABEL's benchmark convention (``abel/benchmark/runner.py``): the
    standard error of the mean over the CV folds, treating each held-out subject
    as one observation. With <2 finite values std/SEM collapse to 0.0.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0, 0.0
    std = float(np.std(arr, ddof=1))
    return mean, std, std / float(np.sqrt(n))


def leave_one_subject_out(
    project: ProjectRef,
    behavior_id: str,
    *,
    trainer: ActiveLearningTrainerService | None = None,
    seed: int = 42,
    df: pd.DataFrame | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run LOSO CV for one behavior and return pooled raw+refined metrics.

    Returns a dict with pooled macro P/R/F1 + TP/FP/FN for raw and refined, the
    per-fold F1s (for mean/std), and fold bookkeeping. ``error`` is set instead
    when the run cannot proceed (too few subjects, no target positives, etc.).
    """
    trainer = trainer or ActiveLearningTrainerService()
    if df is None:
        if not project.training_set_path.exists():
            return {"behavior_id": behavior_id, "error": "training_set.parquet not found"}
        df = pd.read_parquet(project.training_set_path)
    df = df.reset_index(drop=True)

    group_col = "animal_id" if "animal_id" in df.columns else "session_id"
    if group_col not in df.columns or "label" not in df.columns:
        return {"behavior_id": behavior_id, "error": f"training set missing '{group_col}'/'label'"}

    refine_only = (
        _is_refine_only(df["label_source"]) if "label_source" in df.columns
        else pd.Series(False, index=df.index)
    )
    target = str(behavior_id)
    subjects = sorted(df.loc[~refine_only, group_col].astype(str).unique())
    if len(subjects) < 2:
        return {"behavior_id": behavior_id, "error": f"need >=2 subjects, found {len(subjects)}"}

    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    sess_all: list[np.ndarray] = []
    sf_all: list[np.ndarray] = []
    ef_all: list[np.ndarray] = []
    per_fold: list[dict[str, Any]] = []
    used_subjects: list[str] = []

    grp = df[group_col].astype(str)
    for i, subj in enumerate(subjects, 1):
        if progress_cb:
            progress_cb(f"LOSO {project.behavior_label(target)}: fold {i}/{len(subjects)} (hold out {subj})")
        holdout = df[(grp == subj) & ~refine_only]
        pool = df[grp != subj]
        if holdout.empty or pool.empty:
            continue
        # Need at least one target-positive in the held-out subject to score recall.
        if (holdout["label"].astype(str) == target).sum() == 0:
            per_fold.append({"subject": subj, "skipped": "no target positives in holdout"})
            continue

        res = run_one_config(
            trainer, project, target, pool, holdout,
            seed=seed, retain_estimator=True,
        )
        if res.degenerate or res.y_true is None or res.y_score is None or res.val_meta is None:
            per_fold.append({"subject": subj, "skipped": res.error or "degenerate fold"})
            continue

        meta = res.val_meta.reset_index(drop=True)
        y = np.asarray(res.y_true, dtype=int)
        p = np.asarray(res.y_score, dtype=float)
        s, e = _frames_from_segment_ids(meta["segment_id"])
        # Namespace the session id by fold so the refinement frame-trace never
        # merges rows from two different held-out subjects.
        sess = np.array([f"{subj}::{sid}" for sid in meta["session_id"].astype(str)], dtype=object)

        valid = np.isfinite(p) & (s >= 0)
        if not valid.any():
            continue
        y_all.append(y[valid]); p_all.append(p[valid])
        sess_all.append(sess[valid]); sf_all.append(s[valid]); ef_all.append(e[valid])
        used_subjects.append(subj)

        from sklearn.metrics import average_precision_score, f1_score  # noqa: PLC0415
        yv, pv = y[valid], p[valid]
        # PR-AUC (average precision) is undefined when the held-out subject has a
        # single class — record NaN so it drops out of the mean/SEM cleanly.
        fold_prauc = (
            float(average_precision_score(yv, pv)) if np.unique(yv).size > 1 else float("nan")
        )
        per_fold.append({
            "subject": subj,
            "n_holdout": int(valid.sum()),
            "f1": float(f1_score(yv, (pv >= 0.5).astype(int), average="macro", zero_division=0)),
            "pr_auc": fold_prauc,
            "tp": int(res.tp), "fp": int(res.fp), "fn": int(res.fn),
        })

    if not y_all:
        return {"behavior_id": behavior_id, "error": "no scorable folds", "folds": per_fold}

    # Disk name, not behavior_label(): refinement settings are stored on disk under
    # the project's own behavior name, which a display rename must not follow.
    settings = load_temporal_settings(project.root, project.behavior_disk_name(target))
    pooled = score_raw_and_refined(
        y_true=np.concatenate(y_all),
        prob=np.concatenate(p_all),
        session_ids=np.concatenate(sess_all),
        start_frames=np.concatenate(sf_all),
        end_frames=np.concatenate(ef_all),
        settings=settings,
    )
    fold_f1s = [f["f1"] for f in per_fold if "f1" in f]
    fold_praucs = [f["pr_auc"] for f in per_fold if "pr_auc" in f]
    f1_mean, f1_std, f1_sem = _mean_std_sem(fold_f1s)
    pr_mean, pr_std, pr_sem = _mean_std_sem(fold_praucs)
    pooled.update({
        "behavior_id": behavior_id,
        "behavior_name": project.behavior_label(target),
        "method": "leave_one_subject_out",
        "n_subjects": len(used_subjects),
        "subjects": used_subjects,
        # Per-fold mean ± SEM (each held-out subject = one observation) is the
        # publication figure's basis; std kept for backward compatibility.
        "fold_f1_mean": f1_mean,
        "fold_f1_std": f1_std,
        "fold_f1_sem": f1_sem,
        "fold_prauc_mean": pr_mean,
        "fold_prauc_std": pr_std,
        "fold_prauc_sem": pr_sem,
        "folds": per_fold,
    })
    return pooled


def leave_one_subject_out_all(
    project: ProjectRef,
    *,
    behavior_ids: list[str] | None = None,
    seed: int = 42,
    progress_cb: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run LOSO CV for every (non-no_behavior) behavior. Loads the frame once."""
    trainer = ActiveLearningTrainerService()
    df = pd.read_parquet(project.training_set_path) if project.training_set_path.exists() else None
    bids = behavior_ids or [b for b in project.behavior_names if str(b) != "no_behavior"]
    out: list[dict[str, Any]] = []
    for bid in bids:
        out.append(
            leave_one_subject_out(
                project, bid, trainer=trainer, seed=seed, df=df, progress_cb=progress_cb
            )
        )
    return out
