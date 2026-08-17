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
from collections.abc import Iterable, Sequence
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


def _group_column(df: pd.DataFrame) -> str | None:
    """The column that identifies a subject: ``animal_id``, else ``session_id``."""
    for col in ("animal_id", "session_id"):
        if col in df.columns:
            return col
    return None


def _select_subjects(
    available: list[str], requested: Iterable[str] | None
) -> tuple[list[str], list[str]]:
    """Intersect *requested* with *available*; returns (selected, unknown).

    Order follows ``available`` so folds run in the same (sorted) order whether or
    not the caller restricted the subject set.
    """
    if requested is None:
        return list(available), []
    want = {str(s).strip() for s in requested if str(s).strip()}
    selected = [s for s in available if s in want]
    unknown = sorted(want - set(available))
    return selected, unknown


def _read_subject_columns(path: Any) -> pd.DataFrame | None:
    """Read only the id/label columns of a training set.

    ``training_set.parquet`` carries >1000 feature columns, so listing subjects for
    a picker must not pull the whole frame into memory.
    """
    wanted = ["animal_id", "session_id", "label", "label_source"]
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415

        present = set(pq.ParquetFile(str(path)).schema.names)
        cols = [c for c in wanted if c in present]
        if cols:
            return pd.read_parquet(path, columns=cols)
    except Exception:  # pragma: no cover - falls back to a full read
        logger.debug("Column-projected parquet read failed; reading full frame", exc_info=True)
    try:
        return pd.read_parquet(path)
    except Exception:
        logger.exception("Could not read training set at %s", path)
        return None


def available_subjects(
    project: ProjectRef, *, df: pd.DataFrame | None = None
) -> list[dict[str, Any]]:
    """Subjects LOSO can hold out, with the label coverage each one contributes.

    Used to populate the mouse picker before a run. Each entry has ``subject``,
    ``n_windows`` (scorable labeled rows), ``n_labeled`` (rows carrying a real
    behavior, i.e. not ``no_behavior``) and ``n_sessions``. Refinement-only rows
    (temporal feedback / imported labels) are excluded, exactly as the run excludes
    them from evaluation, so the counts shown match what a fold can actually score.
    """
    if df is None:
        if not project.training_set_path.exists():
            return []
        df = _read_subject_columns(project.training_set_path)
    if df is None or df.empty:
        return []
    group_col = _group_column(df)
    if group_col is None or "label" not in df.columns:
        return []

    refine_only = (
        _is_refine_only(df["label_source"]) if "label_source" in df.columns
        else pd.Series(False, index=df.index)
    )
    scorable = df.loc[~refine_only.to_numpy()]
    if scorable.empty:
        return []
    labels = scorable["label"].astype(str)
    out: list[dict[str, Any]] = []
    for subj, idx in scorable.groupby(scorable[group_col].astype(str)).groups.items():
        rows = scorable.loc[idx]
        out.append({
            "subject": str(subj),
            "n_windows": int(len(rows)),
            "n_labeled": int((labels.loc[idx] != "no_behavior").sum()),
            "n_sessions": (
                int(rows["session_id"].astype(str).nunique())
                if "session_id" in rows.columns else 0
            ),
        })
    return sorted(out, key=lambda d: d["subject"])


def _mean_std_sem(values: list[float]) -> tuple[float, float, float]:
    """Mean, sample std (ddof=1) and SEM = s/sqrt(n) across folds, NaN-aware.

    DESCRIPTIVE ONLY — do not publish this SEM as an error bar. LOSO folds are
    not independent observations: any two folds share (N-2)/(N-1) of their
    training data, so their scores are strongly positively correlated and
    s/sqrt(n) understates the true uncertainty. Bengio & Grandvalet (2004,
    *JMLR* 5:1089-1105) proved that no unbiased estimator of k-fold CV variance
    exists, so there is no correction that rescues this number.

    The interval to publish is the subject-level bootstrap CI on the pooled
    held-out predictions (``boot_f1_target_lo/hi``, ``boot_prauc_lo/hi``), which
    resamples subjects and therefore respects the unit of analysis. Results also
    carry ``fold_sem_valid: False`` as a machine-readable flag.

    Kept because the validation GUI and ``loso_plot`` read ``fold_*_sem``; the
    spread it reports is still a useful sanity read on fold-to-fold variability.
    With <2 finite values std/SEM collapse to 0.0.
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


def _target_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Positive-class precision/recall/F1, ``zero_division=0``.

    ABEL trains one-vs-rest detectors at 3-8 % prevalence, so a macro average
    pairs the target's F1 with a ~0.97 "not this behavior" F1 and puts a ~0.50
    floor under a detector that never fires. The target class alone is the
    honest number.
    """
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _bootstrap_subject_ci(
    subject_y: list[np.ndarray],
    subject_p: list[np.ndarray],
    *,
    n_reps: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Percentile CIs for pooled target-class F1 and PR-AUC by resampling subjects.

    The unit of analysis is the animal, not the window: rows within a mouse are
    correlated, so resampling rows would produce an interval far too narrow.
    Each replicate draws ``n_subjects`` subjects **with replacement**, pools
    those subjects' held-out rows, and rescores. No model is refit, so this is
    cheap. Reported as the 2.5 / 97.5 percentiles over ``n_reps`` replicates.
    """
    from sklearn.metrics import average_precision_score  # noqa: PLC0415

    n = len(subject_y)
    out: dict[str, Any] = {
        "boot_n_reps": int(n_reps),
        "boot_seed": int(seed),
        "boot_n_subjects": int(n),
    }
    if n < 2:
        out.update({
            "boot_f1_target_lo": float("nan"), "boot_f1_target_hi": float("nan"),
            "boot_prauc_lo": float("nan"), "boot_prauc_hi": float("nan"),
        })
        return out

    rng = np.random.default_rng(seed)
    f1s = np.full(n_reps, np.nan, dtype=float)
    praucs = np.full(n_reps, np.nan, dtype=float)
    for r in range(n_reps):
        pick = rng.integers(0, n, size=n)
        y = np.concatenate([subject_y[i] for i in pick])
        p = np.concatenate([subject_p[i] for i in pick])
        f1s[r] = _target_prf(y, (p >= 0.5).astype(int))[2]
        # A replicate that happens to draw only negative subjects has no
        # defined average precision; it drops out via nanpercentile.
        if np.unique(y).size > 1:
            praucs[r] = float(average_precision_score(y, p))

    def _pct(arr: np.ndarray) -> tuple[float, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan"), float("nan")
        return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))

    f1_lo, f1_hi = _pct(f1s)
    pr_lo, pr_hi = _pct(praucs)
    out.update({
        "boot_f1_target_lo": f1_lo, "boot_f1_target_hi": f1_hi,
        "boot_prauc_lo": pr_lo, "boot_prauc_hi": pr_hi,
    })
    return out


def leave_one_subject_out(
    project: ProjectRef,
    behavior_id: str,
    *,
    trainer: ActiveLearningTrainerService | None = None,
    seed: int = 42,
    df: pd.DataFrame | None = None,
    subjects: Sequence[str] | None = None,
    n_boot: int = 2000,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run LOSO CV for one behavior and return pooled raw+refined metrics.

    ``subjects`` restricts the analysis to the given mice (``None`` = all). An
    unselected mouse is dropped from the analysis entirely — it is neither held out
    nor trained on — so the result is a clean LOSO over exactly the mice the user
    chose. Refinement-only rows are kept in the training pool regardless, since they
    are corrections rather than subjects in their own right.

    Returns a dict with pooled macro P/R/F1 + TP/FP/FN for raw and refined,
    pooled *target-class* P/R/F1 and PR-AUC, a subject-level bootstrap CI
    (``boot_*``, ``n_boot`` replicates — the interval to publish), a
    ``per_subject`` table, the per-fold F1s, and fold bookkeeping including how
    many folds were skipped and why. ``error`` is set instead when the run
    cannot proceed (too few subjects, no target positives, etc.).
    """
    trainer = trainer or ActiveLearningTrainerService()
    if df is None:
        if not project.training_set_path.exists():
            return {"behavior_id": behavior_id, "error": "training_set.parquet not found"}
        df = pd.read_parquet(project.training_set_path)
    df = df.reset_index(drop=True)

    group_col = _group_column(df)
    if group_col is None or "label" not in df.columns:
        return {
            "behavior_id": behavior_id,
            "error": "training set missing 'animal_id'/'session_id' or 'label'",
        }

    refine_only = (
        _is_refine_only(df["label_source"]) if "label_source" in df.columns
        else pd.Series(False, index=df.index)
    )
    target = str(behavior_id)
    grp = df[group_col].astype(str)
    all_subjects = sorted(grp[~refine_only].unique())
    selected, unknown = _select_subjects(all_subjects, subjects)
    if unknown:
        logger.warning(
            "LOSO: ignoring %d requested subject(s) not in the training set: %s",
            len(unknown), ", ".join(unknown),
        )
    excluded = [s for s in all_subjects if s not in set(selected)]
    if len(selected) < 2:
        scope = "selected" if subjects is not None else "found"
        return {
            "behavior_id": behavior_id,
            "behavior_name": project.behavior_label(target),
            "error": f"need >=2 subjects, {scope} {len(selected)}",
            "excluded_subjects": excluded,
        }
    if excluded:
        # Dropping the rows (not just the folds) keeps every fold's training pool
        # inside the user's chosen subject set.
        keep = (grp.isin(selected) | refine_only).to_numpy()
        df = df.loc[keep].reset_index(drop=True)
        refine_only = refine_only.loc[keep].reset_index(drop=True)
        grp = df[group_col].astype(str)

    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    sess_all: list[np.ndarray] = []
    sf_all: list[np.ndarray] = []
    ef_all: list[np.ndarray] = []
    per_fold: list[dict[str, Any]] = []
    per_subject: list[dict[str, Any]] = []
    used_subjects: list[str] = []

    for i, subj in enumerate(selected, 1):
        if progress_cb:
            progress_cb(f"LOSO {project.behavior_label(target)}: fold {i}/{len(selected)} (hold out {subj})")
        holdout = df[(grp == subj) & ~refine_only]
        pool = df[grp != subj]
        if holdout.empty or pool.empty:
            per_fold.append({"subject": subj, "skipped": "empty holdout or training pool"})
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
            per_fold.append({"subject": subj, "skipped": "no finite scores in holdout"})
            continue
        y_all.append(y[valid]); p_all.append(p[valid])
        sess_all.append(sess[valid]); sf_all.append(s[valid]); ef_all.append(e[valid])
        used_subjects.append(subj)

        from sklearn.metrics import average_precision_score, f1_score  # noqa: PLC0415
        yv, pv = y[valid], p[valid]
        predv = (pv >= 0.5).astype(int)
        # PR-AUC (average precision) is undefined when the held-out subject has a
        # single class — record NaN so it drops out of the mean/SEM cleanly.
        fold_prauc = (
            float(average_precision_score(yv, pv)) if np.unique(yv).size > 1 else float("nan")
        )
        prec_t, rec_t, f1_t = _target_prf(yv, predv)
        per_fold.append({
            "subject": subj,
            "n_holdout": int(valid.sum()),
            # Macro F1, kept unchanged for backward compatibility with the GUI and
            # loso_plot; it carries a ~0.50 floor at this prevalence. Use f1_target.
            "f1": float(f1_score(yv, predv, average="macro", zero_division=0)),
            "f1_target": f1_t,
            "precision_target": prec_t,
            "recall_target": rec_t,
            "pr_auc": fold_prauc,
            "tp": int(res.tp), "fp": int(res.fp), "fn": int(res.fn),
        })
        per_subject.append({
            "subject": subj,
            "n_rows": int(valid.sum()),
            "n_positives": int((yv == 1).sum()),
            "f1_target": f1_t,
            "pr_auc": fold_prauc,
            "tp": int(((yv == 1) & (predv == 1)).sum()),
            "fp": int(((yv == 0) & (predv == 1)).sum()),
            "fn": int(((yv == 1) & (predv == 0)).sum()),
            "tn": int(((yv == 0) & (predv == 0)).sum()),
        })

    if not y_all:
        return {
            "behavior_id": behavior_id,
            "behavior_name": project.behavior_label(target),
            "error": "no scorable folds",
            "folds": per_fold,
            "excluded_subjects": excluded,
            "n_folds_total": len(selected),
            "n_folds_scored": 0,
            "n_folds_skipped": len(selected),
        }

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
    fold_f1_targets = [f["f1_target"] for f in per_fold if "f1_target" in f]
    fold_praucs = [f["pr_auc"] for f in per_fold if "pr_auc" in f]
    f1_mean, f1_std, f1_sem = _mean_std_sem(fold_f1s)
    f1t_mean, f1t_std, f1t_sem = _mean_std_sem(fold_f1_targets)
    pr_mean, pr_std, pr_sem = _mean_std_sem(fold_praucs)

    # Pooled target-class P/R/F1 from the pooled counts score_raw_and_refined
    # already computed at the same 0.5 threshold, so the two never disagree.
    p_tp = float(pooled.get("raw_tp", 0)); p_fp = float(pooled.get("raw_fp", 0))
    p_fn = float(pooled.get("raw_fn", 0))
    pooled_prec = p_tp / (p_tp + p_fp) if (p_tp + p_fp) > 0 else 0.0
    pooled_rec = p_tp / (p_tp + p_fn) if (p_tp + p_fn) > 0 else 0.0
    pooled_f1t = (
        2.0 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec)
        if (pooled_prec + pooled_rec) > 0 else 0.0
    )

    from sklearn.metrics import average_precision_score  # noqa: PLC0415
    y_pooled, p_pooled = np.concatenate(y_all), np.concatenate(p_all)
    pooled_prauc = (
        float(average_precision_score(y_pooled, p_pooled))
        if np.unique(y_pooled).size > 1 else float("nan")
    )

    n_total = len(selected)
    n_scored = len(used_subjects)
    skip_reasons: dict[str, int] = {}
    for f in per_fold:
        if "skipped" in f:
            reason = str(f["skipped"])
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    pooled.update(_bootstrap_subject_ci(y_all, p_all, n_reps=n_boot, seed=seed))
    pooled.update({
        "behavior_id": behavior_id,
        "behavior_name": project.behavior_label(target),
        "method": "leave_one_subject_out",
        "n_subjects": len(used_subjects),
        "subjects": used_subjects,
        # Provenance of the mouse selection, so a restricted run is never mistaken
        # for a whole-cohort one in the figure, CSV or report.
        "selected_subjects": list(selected),
        "excluded_subjects": excluded,
        # Target-class metrics — what the manuscript reports. The macro raw_*/
        # refined_* keys above are retained unchanged for the GUI.
        "pooled_f1_target": pooled_f1t,
        "pooled_precision_target": pooled_prec,
        "pooled_recall_target": pooled_rec,
        "pooled_prauc": pooled_prauc,
        # Fold-level descriptive spread. NOT a valid error bar — see
        # _mean_std_sem's docstring and fold_sem_valid below.
        "fold_f1_mean": f1_mean,
        "fold_f1_std": f1_std,
        "fold_f1_sem": f1_sem,
        "fold_f1_target_mean": f1t_mean,
        "fold_f1_target_std": f1t_std,
        "fold_f1_target_sem": f1t_sem,
        "fold_prauc_mean": pr_mean,
        "fold_prauc_std": pr_std,
        "fold_prauc_sem": pr_sem,
        "fold_sem_valid": False,
        # Fold bookkeeping: a per-fold mean over only the scored folds is
        # survivorship-biased, so the drop count travels with the result.
        "n_folds_total": n_total,
        "n_folds_scored": n_scored,
        "n_folds_skipped": n_total - n_scored,
        "skip_reasons": skip_reasons,
        "per_subject": per_subject,
        "folds": per_fold,
    })
    if n_total > 0 and (n_total - n_scored) / n_total > 0.25:
        pooled["warning"] = "majority-skipped — per-fold mean is survivorship-biased"
    return pooled


def leave_one_subject_out_all(
    project: ProjectRef,
    *,
    behavior_ids: list[str] | None = None,
    seed: int = 42,
    subjects: Sequence[str] | None = None,
    n_boot: int = 2000,
    progress_cb: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run LOSO CV for every (non-no_behavior) behavior. Loads the frame once.

    ``subjects`` restricts every behavior's run to the same set of mice; ``None``
    uses all of them.
    """
    trainer = ActiveLearningTrainerService()
    df = pd.read_parquet(project.training_set_path) if project.training_set_path.exists() else None
    bids = behavior_ids or [b for b in project.behavior_names if str(b) != "no_behavior"]
    out: list[dict[str, Any]] = []
    for bid in bids:
        out.append(
            leave_one_subject_out(
                project, bid, trainer=trainer, seed=seed, df=df,
                subjects=subjects, n_boot=n_boot, progress_cb=progress_cb,
            )
        )
    return out
