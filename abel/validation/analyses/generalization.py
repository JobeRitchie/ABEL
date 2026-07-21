"""Generalization / human-agreement validation.

Trains on the training-pool subjects/sessions and evaluates on the *held-out*
subjects/sessions (a true cross-subject/session split).  Reports
precision/recall/F1 and Cohen's kappa of the model vs. the held-out reviewed
labels.  Where multiple human reviewers exist, the human inter-rater ceiling is
computed from ``validation_service`` and attached for plotting; otherwise it is
left as NaN (flagged).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.temporal_refinement.refined_eval import _frames_from_segment_ids
from abel.validation import subsample
from abel.validation.datamodel import CellResult, ConfigEvalResult, ProjectRef
from abel.validation.engine import run_one_config
from abel.validation.holdout import HoldoutSplit


@dataclass
class HoldoutPredictions:
    """Per-row held-out predictions from one representative generalization fit.

    Retained so the downstream biological-readout (time budget) and calibration
    analyses can be computed from the SAME held-out predictions the generalization
    metrics use — no extra model training. Arrays are aligned 1-D, target-vs-rest
    encoded (positive class == 1), with ``prob`` = P(target).
    """

    project_id: str
    behavior_id: str
    behavior_name: str
    session_ids: np.ndarray
    animal_ids: np.ndarray
    start_frames: np.ndarray
    end_frames: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    prob: np.ndarray


@dataclass
class GeneralizationResult:
    project_id: str
    behavior_id: str
    behavior_name: str
    f1_mean: float = float("nan")
    f1_ci: float = float("nan")       # 95% CI half-width across seeds
    kappa_mean: float = float("nan")
    kappa_ci: float = float("nan")
    n_seeds: int = 0
    human_ceiling_kappa: float = float("nan")  # NaN unless multi-reviewer data exists
    cells: list[CellResult] = field(default_factory=list)
    predictions: "HoldoutPredictions | None" = None


def run_generalization(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_id: str,
    holdout_split: HoldoutSplit,
    *,
    n_seeds: int = 3,
    human_ceiling_kappa: float = float("nan"),
    retain_predictions: bool = True,
    progress_cb: Callable[[str], None] | None = None,
) -> GeneralizationResult:
    """Held-out generalization metrics for one (project, behavior).

    When ``retain_predictions`` is set, the first seed's per-row held-out
    predictions (session/animal ids, segment frame bounds, true/pred labels,
    P(target)) are captured on ``result.predictions`` so the biological-readout
    and calibration analyses can reuse them without training again.
    """
    behavior_name = project.behavior_label(behavior_id)
    pool = holdout_split.train_pool
    n_pos = subsample.count_positives(pool, behavior_id)
    n_neg = int(len(pool) - n_pos)

    result = GeneralizationResult(
        project_id=project.project_id,
        behavior_id=str(behavior_id),
        behavior_name=behavior_name,
        human_ceiling_kappa=float(human_ceiling_kappa),
    )

    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    f1s: list[float] = []
    kappas: list[float] = []
    for rep in range(n_seeds):
        seed = 2000 + rep
        _log(f"{behavior_name}: generalization seed {rep + 1}/{n_seeds}…")
        # Retain the estimator/meta only on the first seed — that's all the
        # biological-readout + calibration analyses need (the folds share the
        # same pool + holdout, so any seed is representative).
        keep = retain_predictions and rep == 0
        res = run_one_config(
            trainer, project, behavior_id, pool, holdout_split.holdout,
            seed=seed, n_pos_train=n_pos, n_neg_train=n_neg,
            retain_estimator=keep,
        )
        if keep:
            result.predictions = _build_predictions(project, behavior_id, behavior_name, res)
        result.cells.append(
            CellResult(
                project_id=project.project_id,
                project_name=project.name,
                behavior_id=str(behavior_id),
                behavior_name=behavior_name,
                analysis="generalization",
                config_name="held_out_subjects",
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
        if not res.error and np.isfinite(res.f1):
            f1s.append(res.f1)
            kappas.append(res.cohen_kappa)

    from abel.validation import metrics as vmetrics  # noqa: PLC0415

    result.f1_mean = float(np.nanmean(f1s)) if f1s else float("nan")
    result.kappa_mean = float(np.nanmean(kappas)) if kappas else float("nan")
    # Seed spread was computed and thrown away — the figure needs error bars.
    result.f1_ci = vmetrics.ci95(f1s)
    result.kappa_ci = vmetrics.ci95(kappas)
    result.n_seeds = len(f1s)
    return result


def _build_predictions(
    project: ProjectRef,
    behavior_id: str,
    behavior_name: str,
    res: ConfigEvalResult,
) -> "HoldoutPredictions | None":
    """Assemble a HoldoutPredictions bundle from a retained-estimator result.

    Needs the target-vs-rest arrays plus ``val_meta`` (segment/session/animal ids).
    Frame bounds are parsed from the segment ids the same way the refinement scorer
    does. Returns ``None`` when the fold was degenerate or metadata is missing.
    """
    meta = getattr(res, "val_meta", None)
    if (
        res.y_true is None or res.y_pred is None or res.y_score is None
        or meta is None or "session_id" not in getattr(meta, "columns", [])
    ):
        return None
    meta = meta.reset_index(drop=True)
    n = min(len(meta), len(res.y_true))
    if n == 0:
        return None

    if "segment_id" in meta.columns:
        sf, ef = _frames_from_segment_ids(meta["segment_id"].iloc[:n])
    else:
        sf = np.full(n, -1, dtype=np.int64)
        ef = np.full(n, -1, dtype=np.int64)
    animal = (
        meta["animal_id"].astype(str).to_numpy()[:n]
        if "animal_id" in meta.columns
        else np.full(n, "", dtype=object)
    )
    return HoldoutPredictions(
        project_id=project.project_id,
        behavior_id=str(behavior_id),
        behavior_name=behavior_name,
        session_ids=meta["session_id"].astype(str).to_numpy()[:n],
        animal_ids=animal,
        start_frames=np.asarray(sf, dtype=np.int64)[:n],
        end_frames=np.asarray(ef, dtype=np.int64)[:n],
        y_true=np.asarray(res.y_true, dtype=int)[:n],
        y_pred=np.asarray(res.y_pred, dtype=int)[:n],
        prob=np.asarray(res.y_score, dtype=float)[:n],
    )
