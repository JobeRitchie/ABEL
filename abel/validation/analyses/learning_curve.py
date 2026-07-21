"""Optimal-clips learning curves (the headline data-efficiency analysis).

For one (project, behavior): subsample the training pool at an increasing
schedule of positive-clip counts, retrain via the shared engine, evaluate on the
fixed high-confidence held-out set, and trace F1 / PR-AUC vs. # clips.  Repeats
across seeds give a confidence band; a knee detector reports the "optimal clips"
saturation point.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.validation import subsample
from abel.validation.datamodel import CellResult, ProjectRef
from abel.validation.engine import run_one_config
from abel.validation.holdout import HoldoutSplit

DEFAULT_SIZES: list[int] = [10, 25, 50, 100, 200, subsample.ALL_CLIPS]


def derive_seed(project_id: str, behavior_id: str, size: int, repeat: int) -> int:
    """Deterministic per-cell seed independent across (project, behavior, size, repeat)."""
    key = f"{project_id}|{behavior_id}|{size}|{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


@dataclass
class LearningCurvePoint:
    requested_size: int          # the schedule entry (ALL_CLIPS == -1)
    n_clips_mean: float          # mean actual positive clips used
    f1_mean: float
    f1_ci: float                 # 95% half-width across seeds
    pr_auc_mean: float
    pr_auc_ci: float
    kappa_mean: float
    n_seeds: int
    precision_mean: float = float("nan")
    recall_mean: float = float("nan")
    tp_mean: float = float("nan")    # mean held-out target-vs-rest confusion counts
    fp_mean: float = float("nan")
    fn_mean: float = float("nan")


@dataclass
class LearningCurveResult:
    project_id: str
    behavior_id: str
    behavior_name: str
    points: list[LearningCurvePoint] = field(default_factory=list)
    knee_clips: float | None = None
    f1_max: float = float("nan")
    cells: list[CellResult] = field(default_factory=list)


def _ci95(values: np.ndarray) -> float:
    """95% CI half-width across seeds (t-based — see :func:`metrics.ci95`)."""
    from abel.validation import metrics as vmetrics  # noqa: PLC0415

    return vmetrics.ci95(values)


def detect_knee(points: list[LearningCurvePoint], eps: float = 0.02, delta: float = 0.01) -> float | None:
    """Smallest clip count where F1 ≥ (1−eps)·F1_max and the marginal gain < delta."""
    finite = [p for p in points if np.isfinite(p.f1_mean)]
    if len(finite) < 2:
        return None
    ordered = sorted(finite, key=lambda p: p.n_clips_mean)
    f1_max = max(p.f1_mean for p in ordered)
    if not np.isfinite(f1_max) or f1_max <= 0:
        return None
    threshold = (1.0 - eps) * f1_max
    for i, p in enumerate(ordered):
        if p.f1_mean < threshold:
            continue
        nxt_gain = (ordered[i + 1].f1_mean - p.f1_mean) if i + 1 < len(ordered) else 0.0
        if nxt_gain < delta:
            return float(p.n_clips_mean)
    return float(ordered[-1].n_clips_mean)


def run_learning_curve(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_id: str,
    holdout_split: HoldoutSplit,
    *,
    sizes: list[int] | None = None,
    n_seeds: int = 5,
    neg_policy: str = "all",
    neg_per_pos: float = 3.0,
    progress_cb: Callable[[str], None] | None = None,
) -> LearningCurveResult:
    """Run the full learning-curve sweep for one (project, behavior)."""
    sizes = sizes or list(DEFAULT_SIZES)
    behavior_name = project.behavior_label(behavior_id)
    pool = holdout_split.train_pool
    group_col = holdout_split.group_col
    total_pos = subsample.count_positives(pool, behavior_id)

    result = LearningCurveResult(
        project_id=project.project_id,
        behavior_id=str(behavior_id),
        behavior_name=behavior_name,
    )

    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    # Skip schedule entries that exceed the available positives (except ALL).
    eff_sizes = [
        s for s in sizes
        if s == subsample.ALL_CLIPS or s <= total_pos
    ]
    # De-duplicate the "all" point if a numeric size already equals total.
    seen_all = False
    cleaned: list[int] = []
    for s in eff_sizes:
        if s == subsample.ALL_CLIPS:
            if seen_all:
                continue
            seen_all = True
        cleaned.append(s)
    eff_sizes = cleaned

    for size in eff_sizes:
        size_label = "all" if size == subsample.ALL_CLIPS else str(size)
        f1s: list[float] = []
        praucs: list[float] = []
        kappas: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        tps: list[int] = []
        fps: list[int] = []
        fns: list[int] = []
        nclips: list[int] = []
        for rep in range(n_seeds):
            seed = derive_seed(project.project_id, str(behavior_id), size, rep)
            sub, n_pos, n_neg = subsample.draw(
                pool, behavior_id, size,
                group_col=group_col, seed=seed,
                neg_policy=neg_policy, neg_per_pos=neg_per_pos,
            )
            _log(f"{behavior_name}: n={size_label} seed {rep + 1}/{n_seeds} "
                 f"({n_pos} pos / {n_neg} neg)…")
            res = run_one_config(
                trainer, project, behavior_id, sub, holdout_split.holdout,
                seed=seed, n_pos_train=n_pos, n_neg_train=n_neg,
            )
            cell = CellResult(
                project_id=project.project_id,
                project_name=project.name,
                behavior_id=str(behavior_id),
                behavior_name=behavior_name,
                analysis="learning_curve",
                config_name=f"n={size_label}",
                n_clips=int(n_pos),
                seed=int(seed),
                precision=res.precision,
                recall=res.recall,
                f1=res.f1,
                pr_auc=res.pr_auc,
                cohen_kappa=res.cohen_kappa,
                mcc=res.mcc, balanced_accuracy=res.balanced_accuracy,
                specificity=res.specificity, roc_auc=res.roc_auc,
                tp=res.tp, fp=res.fp, fn=res.fn, tn=res.tn,
                n_pos_train=res.n_pos_train,
                n_neg_train=res.n_neg_train,
                n_features=res.n_features,
                elapsed_sec_fit=res.elapsed_sec_fit,
                elapsed_sec_total=res.elapsed_sec_total,
                degenerate=res.degenerate,
                error=res.error,
            )
            result.cells.append(cell)
            if not res.error:
                f1s.append(res.f1)
                praucs.append(res.pr_auc)
                kappas.append(res.cohen_kappa)
                precisions.append(res.precision)
                recalls.append(res.recall)
                tps.append(res.tp)
                fps.append(res.fp)
                fns.append(res.fn)
                nclips.append(n_pos)

        if nclips:
            result.points.append(
                LearningCurvePoint(
                    requested_size=size,
                    n_clips_mean=float(np.mean(nclips)),
                    f1_mean=float(np.nanmean(f1s)) if f1s else float("nan"),
                    f1_ci=_ci95(np.asarray(f1s)),
                    pr_auc_mean=float(np.nanmean(praucs)) if praucs else float("nan"),
                    pr_auc_ci=_ci95(np.asarray(praucs)),
                    kappa_mean=float(np.nanmean(kappas)) if kappas else float("nan"),
                    n_seeds=len(nclips),
                    precision_mean=float(np.nanmean(precisions)) if precisions else float("nan"),
                    recall_mean=float(np.nanmean(recalls)) if recalls else float("nan"),
                    tp_mean=float(np.mean(tps)) if tps else float("nan"),
                    fp_mean=float(np.mean(fps)) if fps else float("nan"),
                    fn_mean=float(np.mean(fns)) if fns else float("nan"),
                )
            )

    result.points.sort(key=lambda p: p.n_clips_mean)
    finite_f1 = [p.f1_mean for p in result.points if np.isfinite(p.f1_mean)]
    result.f1_max = float(max(finite_f1)) if finite_f1 else float("nan")
    result.knee_clips = detect_knee(result.points)
    return result


def average_curve(
    results: list[LearningCurveResult], project_label: str = "all projects",
) -> LearningCurveResult | None:
    """Mean learning curve across behaviors — the headline "best clip count in general".

    Points are grouped by their schedule step (``requested_size``) so every behavior
    contributes at matching clip counts; metrics are averaged across behaviors and the
    CI band reflects the spread *across behaviors* at each step.  The knee of this mean
    curve is the recommended general-purpose clip count.
    """
    from collections import defaultdict

    buckets: dict[int, list[LearningCurvePoint]] = defaultdict(list)
    for r in results:
        for p in r.points:
            buckets[p.requested_size].append(p)
    if not buckets:
        return None

    def _mean(ps: list[LearningCurvePoint], attr: str) -> float:
        vals = [getattr(p, attr) for p in ps if np.isfinite(getattr(p, attr))]
        return float(np.mean(vals)) if vals else float("nan")

    pts: list[LearningCurvePoint] = []
    for size, ps in buckets.items():
        pts.append(LearningCurvePoint(
            requested_size=size,
            n_clips_mean=float(np.mean([p.n_clips_mean for p in ps])),
            f1_mean=_mean(ps, "f1_mean"),
            f1_ci=_ci95(np.asarray([p.f1_mean for p in ps], dtype=float)),
            pr_auc_mean=_mean(ps, "pr_auc_mean"),
            pr_auc_ci=_ci95(np.asarray([p.pr_auc_mean for p in ps], dtype=float)),
            kappa_mean=_mean(ps, "kappa_mean"),
            n_seeds=sum(p.n_seeds for p in ps),
            precision_mean=_mean(ps, "precision_mean"),
            recall_mean=_mean(ps, "recall_mean"),
            tp_mean=_mean(ps, "tp_mean"),
            fp_mean=_mean(ps, "fp_mean"),
            fn_mean=_mean(ps, "fn_mean"),
        ))
    pts.sort(key=lambda p: p.n_clips_mean)

    n_beh = len(results)
    avg = LearningCurveResult(
        project_id=project_label,
        behavior_id="__average__",
        behavior_name=f"Average across {n_beh} behaviors",
        points=pts,
    )
    finite_f1 = [p.f1_mean for p in pts if np.isfinite(p.f1_mean)]
    avg.f1_max = float(max(finite_f1)) if finite_f1 else float("nan")
    avg.knee_clips = detect_knee(pts)
    return avg
