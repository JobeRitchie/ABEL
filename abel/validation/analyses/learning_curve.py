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
    # Same confusion counts as a percent of the held-out set (each count / n_val).
    # Averaging these per-behavior percents is fair across behaviors, unlike the raw
    # counts, which are dominated by behaviors with larger held-out sets and by the
    # shifting behavior mix as the clip budget grows.
    tp_pct: float = float("nan")
    fp_pct: float = float("nan")
    fn_pct: float = float("nan")
    # 95% CI half-widths for the percent-of-held-out confusion rates. Per behavior
    # these are the seed-to-seed spread; on the average curve (see average_curve)
    # they are the spread *across behaviors* — mirroring how f1_ci is built, so a
    # Prism error-rate graph carries the same error bars as the F1/PR-AUC one.
    tp_pct_ci: float = float("nan")
    fp_pct_ci: float = float("nan")
    fn_pct_ci: float = float("nan")
    # Imbalance-robust companions to F1, averaged over the same seeds. Reported
    # beside F1 everywhere it is plotted or exported: target-class F1 alone cannot
    # distinguish a good detector from an always-predict-target one, and these can.
    mcc_mean: float = float("nan")
    specificity_mean: float = float("nan")
    # Seeds at this budget whose fit collapsed to one class (metrics.is_degenerate_fit)
    # and seeds where probability calibration actually ran. A point that is mostly
    # degenerate is excluded from the knee and from the plotted F1 curve; a point
    # where calibration silently switched off is not comparable to one where it
    # did not, and both facts must survive into the exports.
    n_degenerate: int = 0
    n_calibrated: int = 0
    # Behaviors contributing to this point. 1 on a per-behavior curve; on an average
    # curve it is the real, per-x composition — which is NOT constant across x, and
    # was previously recoverable only by dividing n_seeds by the seed count.
    n_behaviors: int = 1

    @property
    def degenerate_frac(self) -> float:
        return (self.n_degenerate / self.n_seeds) if self.n_seeds else float("nan")

    @property
    def is_degenerate(self) -> bool:
        """Most seeds at this budget produced an uninformative fit."""
        return bool(self.n_seeds) and (self.n_degenerate * 2 > self.n_seeds)


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


#: Saturation criterion for :func:`detect_knee`, named so a figure legend can quote
#: it instead of guessing.  ``KNEE_EPS = 0.02`` means the knee is the first clip
#: count reaching **98%** of the curve's own maximum F1 — not 95%, which is the
#: looser threshold this is easy to mistake it for.
KNEE_EPS = 0.02
KNEE_DELTA = 0.01


def detect_knee(points: list[LearningCurvePoint], eps: float = KNEE_EPS,
                delta: float = KNEE_DELTA) -> float | None:
    """Smallest clip count where F1 ≥ (1−eps)·F1_max and the marginal gain < delta.

    Degenerate points are dropped first.  Under target-class F1 an always-predict-
    target collapse scores *well* (recall 1.0, specificity 0.0), and those live at
    the cold-start end of the curve — precisely where the knee is hunted.  Left in,
    they inflate ``f1_max`` and can plant the knee at the smallest budget on the
    schedule, reporting "20 clips is enough" from a model that had learned nothing.
    Points are kept if all of them are degenerate, so a fully-degenerate curve still
    returns something rather than silently reporting no knee.
    """
    usable = [p for p in points if not p.is_degenerate] or list(points)
    finite = [p for p in usable if np.isfinite(p.f1_mean)]
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
        tp_pcts: list[float] = []
        fp_pcts: list[float] = []
        fn_pcts: list[float] = []
        mccs: list[float] = []
        specs: list[float] = []
        nclips: list[int] = []
        n_degen = 0
        n_calib = 0
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
                precision_macro=res.precision_macro,
                recall_macro=res.recall_macro,
                f1_macro=res.f1_macro,
                pr_auc=res.pr_auc,
                cohen_kappa=res.cohen_kappa,
                mcc=res.mcc, balanced_accuracy=res.balanced_accuracy,
                specificity=res.specificity, roc_auc=res.roc_auc,
                tp=res.tp, fp=res.fp, fn=res.fn, tn=res.tn,
                confusion_measured=res.confusion_measured,
                n_pos_train=res.n_pos_train,
                n_neg_train=res.n_neg_train,
                n_features=res.n_features,
                elapsed_sec_fit=res.elapsed_sec_fit,
                elapsed_sec_total=res.elapsed_sec_total,
                degenerate=res.degenerate,
                degenerate_fit=res.degenerate_fit,
                calibration_applied=res.calibration_applied,
                calibration_method_used=res.calibration_method_used,
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
                mccs.append(res.mcc)
                specs.append(res.specificity)
                # Both collapse flags count. ``degenerate_fit`` is MCC <= 0 (the
                # model predicted one class); ``degenerate`` is the trainer's
                # single-class-validation flag, which is what a zero-positive
                # subsample raises: the fitted label map has no target class, the
                # target index falls back to ``no_behavior``, and the cell scores a
                # perfect target-class F1 off tp=<all negatives>, fp=fn=0. Counting
                # only the first left that point looking healthy, and detect_knee
                # then planted every behavior's knee at the smallest budget.
                n_degen += int(bool(res.degenerate_fit or res.degenerate))
                n_calib += int(bool(res.calibration_applied))
                n_val = res.tp + res.fp + res.fn + res.tn
                if n_val > 0:
                    tp_pcts.append(100.0 * res.tp / n_val)
                    fp_pcts.append(100.0 * res.fp / n_val)
                    fn_pcts.append(100.0 * res.fn / n_val)
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
                    tp_pct=float(np.mean(tp_pcts)) if tp_pcts else float("nan"),
                    fp_pct=float(np.mean(fp_pcts)) if fp_pcts else float("nan"),
                    fn_pct=float(np.mean(fn_pcts)) if fn_pcts else float("nan"),
                    tp_pct_ci=_ci95(np.asarray(tp_pcts, dtype=float)),
                    fp_pct_ci=_ci95(np.asarray(fp_pcts, dtype=float)),
                    fn_pct_ci=_ci95(np.asarray(fn_pcts, dtype=float)),
                    mcc_mean=(float(np.nanmean(mccs)) if mccs else float("nan")),
                    specificity_mean=(float(np.nanmean(specs)) if specs
                                      else float("nan")),
                    n_degenerate=int(n_degen),
                    n_calibrated=int(n_calib),
                )
            )

    result.points.sort(key=lambda p: p.n_clips_mean)
    # Degenerate points are excluded from the ceiling for the same reason
    # detect_knee drops them: a collapsed fit scores WELL on target-class F1, so
    # leaving one in sets f1_max to a value no real model reached and every
    # "% of peak" statement downstream is measured against it.  Kept only if
    # every point is degenerate, mirroring detect_knee's fallback.
    usable = [p for p in result.points if not p.is_degenerate] or list(result.points)
    finite_f1 = [p.f1_mean for p in usable if np.isfinite(p.f1_mean)]
    result.f1_max = float(max(finite_f1)) if finite_f1 else float("nan")
    result.knee_clips = detect_knee(result.points)
    return result


def average_curve(
    results: list[LearningCurveResult], project_label: str = "all projects",
    *, balanced: bool = False,
) -> LearningCurveResult | None:
    """Mean learning curve across behaviors — the headline "best clip count in general".

    Points are grouped by their schedule step (``requested_size``) so every behavior
    contributes at matching clip counts; metrics are averaged across behaviors and the
    CI band reflects the spread *across behaviors* at each step.  The knee of this mean
    curve is the recommended general-purpose clip count.

    **The composition of that mean is not constant across x.**  A behavior only has a
    point at budget n if it owns at least n positive clips, so rare behaviors drop out
    as the budget grows: on the manuscript run the curve goes from 43 behaviors at
    x=5 to 28 at x=400, a 35% attrition.  Since the behaviors that drop out are the
    rare, hard ones, part of the apparent right-hand *plateau* is the denominator
    losing its hardest members rather than the model ceasing to improve — measured at
    x=200, all-available scores 0.786 against 0.759 for a fixed cohort, a gap of
    +0.027.

    Two things follow.  ``n_behaviors`` is now recorded per point, so the composition
    is visible at every x instead of having to be inferred by dividing ``n_seeds`` by
    the seed count.  And ``balanced=True`` restricts the average to behaviors present
    at **every** budget, which is the comparison a plateau claim actually needs: a
    fixed cohort measured repeatedly, where a flat curve means flat performance and
    nothing else.  The balanced curve is smaller-n and noisier, which is the honest
    price of holding composition fixed.
    """
    from collections import defaultdict

    if balanced:
        # Hold the COHORT fixed, not the x-range. Intersecting the schedules instead
        # would delete the high-budget end — exactly the region a plateau claim is
        # about — so the deepest schedule that at least two behaviors reach is kept,
        # and only behaviors reaching all of it contribute. The schedule is a prefix
        # (a behavior with n positives has every smaller budget too), so "reached the
        # deepest budget" is the whole membership test.
        avail = [({p.requested_size for p in r.points}, r) for r in results if r.points]
        if not avail:
            return None
        all_sizes: set[int] = set().union(*(s for s, _ in avail))
        numeric = sorted(s for s in all_sizes if s != subsample.ALL_CLIPS)
        has_all = subsample.ALL_CLIPS in all_sizes
        cohort: list[LearningCurveResult] = []
        want: set[int] = set()
        for cut in range(len(numeric), 0, -1):
            want = set(numeric[:cut]) | ({subsample.ALL_CLIPS} if has_all else set())
            cohort = [r for sizes, r in avail if want <= sizes]
            if len(cohort) >= 2:
                break
        if len(cohort) < 2:
            return None
        results = [
            LearningCurveResult(
                project_id=r.project_id, behavior_id=r.behavior_id,
                behavior_name=r.behavior_name,
                points=[p for p in r.points if p.requested_size in want],
                knee_clips=r.knee_clips, f1_max=r.f1_max,
            )
            for r in cohort
        ]

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
            tp_pct=_mean(ps, "tp_pct"),
            fp_pct=_mean(ps, "fp_pct"),
            fn_pct=_mean(ps, "fn_pct"),
            tp_pct_ci=_ci95(np.asarray([p.tp_pct for p in ps], dtype=float)),
            fp_pct_ci=_ci95(np.asarray([p.fp_pct for p in ps], dtype=float)),
            fn_pct_ci=_ci95(np.asarray([p.fn_pct for p in ps], dtype=float)),
            mcc_mean=_mean(ps, "mcc_mean"),
            specificity_mean=_mean(ps, "specificity_mean"),
            n_degenerate=sum(p.n_degenerate for p in ps),
            n_calibrated=sum(p.n_calibrated for p in ps),
            n_behaviors=len(ps),
        ))
    pts.sort(key=lambda p: p.n_clips_mean)

    # Label from the real composition. A single "across N behaviors" claim is only
    # true when N does not move; when it does, the range is what is true.
    counts = {p.n_behaviors for p in pts}
    n_beh = len(results)
    if len(counts) == 1:
        comp = f"{n_beh} behaviors"
    else:
        comp = f"{min(counts)}-{max(counts)} behaviors (varies by clip budget)"
    avg = LearningCurveResult(
        project_id=project_label,
        behavior_id="__average_balanced__" if balanced else "__average__",
        behavior_name=(f"Average across {n_beh} behaviors present at every budget"
                       if balanced else f"Average across {comp}"),
        points=pts,
    )
    finite_f1 = [p.f1_mean for p in pts if np.isfinite(p.f1_mean)]
    avg.f1_max = float(max(finite_f1)) if finite_f1 else float("nan")
    avg.knee_clips = detect_knee(pts)
    return avg
