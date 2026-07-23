"""Cross-project meta-analysis — pure aggregation over ``cells.parquet``.

No new training: consumes the tidy cell table produced by the other analyses and
rolls it up into per-project / per-behavior / cross-project summaries of
accuracy, training speed, and data efficiency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abel.validation.aggregate import summarize_by
from abel.validation.analyses.ablation import ALL_FEATURES_CONFIG


def _full_budget_only(abl_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the largest-clip-budget ablation rows per (project, behavior).

    Ablation may run at several clip budgets; for cross-project accuracy we want the
    full-data point (the largest ``n_clips``) so low-budget runs don't dilute it.
    """
    if abl_df.empty or "n_clips" not in abl_df.columns:
        return abl_df
    keys = ["project_id", "behavior_id"] if "behavior_id" in abl_df.columns else ["project_id"]
    maxn = abl_df.groupby(keys)["n_clips"].transform("max")
    return abl_df[abl_df["n_clips"] == maxn]


def accuracy_by_project(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Mean ± CI of a metric per project (using generalization cells if present,
    else the best available analysis)."""
    if df.empty:
        return pd.DataFrame()
    src = df[df["analysis"] == "generalization"]
    if src.empty:
        src = df[df["analysis"] == "ablation"]
        if not src.empty:
            src = src[src["config_name"] == ALL_FEATURES_CONFIG]
            src = _full_budget_only(src)
    if src.empty:
        src = df
    return summarize_by(src, ["project_id", "project_name"], metric=metric)


def _headline_cells(df: pd.DataFrame) -> pd.DataFrame:
    """The cells that constitute a behavior's *headline* held-out result.

    Generalization cells (the honest held-out split) plus the full-budget
    all-features ablation cells.  Factored out so the reported confusion counts
    and the reported F1 can never describe different fits — a table where
    "caught 191 of 214" sits next to an F1 computed over some other selection of
    cells is worse than no table.
    """
    if df.empty:
        return df
    src = df[df["analysis"].isin(["generalization", "ablation"])]
    if not src.empty and "config_name" in src.columns:
        abl = _full_budget_only(src[(src["analysis"] == "ablation")
                                    & (src["config_name"] == ALL_FEATURES_CONFIG)])
        src = pd.concat([src[src["analysis"] == "generalization"], abl], ignore_index=True)
    return src if not src.empty else df


def accuracy_by_behavior(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return summarize_by(_headline_cells(df), ["project_id", "behavior_name"], metric=metric)


#: Held-out confusion counts, in reporting order.
CONFUSION_COLS = ["tp", "fp", "fn", "tn"]

_CONFUSION_OUT_COLS = ["project_id", "behavior_name", *CONFUSION_COLS,
                       "n_val", "n_pos_val", "precision", "recall", "n_cells"]


def confusion_by_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out confusion counts per (project, behavior) — the tangible table.

    F1 and PR-AUC compress the result past the point a reader can picture it.
    "Of the 214 held-out windows the reviewer scored as rearing, the model found
    191 and missed 23, and flagged 17 the reviewer did not" is the same evidence
    and cannot be misread.  This builds that row.

    Three deliberate choices:

    * Counts are **averaged** across the cells behind a behavior's headline
      result (seeds × the configs :func:`_headline_cells` keeps), never summed.
      Every seed scores the same held-out pool, so summing would multiply one
      body of evidence by the seed count and advertise an ``n`` the study never
      had.  The mean is the per-fit count, which is what ``n_val`` describes.
    * ``precision``/``recall`` are recomputed **from these counts**, not copied
      from the per-cell metric means, so a reader who divides the columns gets
      the printed rate back.
    * The unit is one **held-out window scored by the reviewer** — never a bout.
      Bout-level counts are not identifiable from a sparse labeled subset (see
      :func:`abel.temporal_refinement.refined_eval._refined_bout_counts`), and
      count-framing is exactly the presentation that invites a reader to assume
      otherwise, so every consumer of this table must name the unit.

    ``tn`` is carried so the 2×2 closes, but it is the majority class under heavy
    imbalance: nothing downstream should derive a headline accuracy from it.
    """
    keys = ["project_id", "behavior_name"]
    empty = pd.DataFrame(columns=_CONFUSION_OUT_COLS)
    if df.empty or not set(CONFUSION_COLS) <= set(df.columns):
        return empty
    src = _headline_cells(df)
    if "error" in src.columns:
        src = src[~src["error"].astype(bool)]
    if src.empty or not set(keys) <= set(src.columns):
        return empty

    rows = []
    for key, grp in src.groupby(keys, dropna=False):
        c = {col: float(pd.to_numeric(grp[col], errors="coerce").mean())
             for col in CONFUSION_COLS}
        if not all(np.isfinite(v) for v in c.values()):
            continue
        n_pos = c["tp"] + c["fn"]
        pred_pos = c["tp"] + c["fp"]
        rows.append({
            **dict(zip(keys, key if isinstance(key, tuple) else (key,))),
            **{col: int(round(c[col])) for col in CONFUSION_COLS},
            "n_val": int(round(sum(c.values()))),
            "n_pos_val": int(round(n_pos)),
            "precision": float(c["tp"] / pred_pos) if pred_pos > 0 else float("nan"),
            "recall": float(c["tp"] / n_pos) if n_pos > 0 else float("nan"),
            "n_cells": int(len(grp)),
        })
    return pd.DataFrame(rows, columns=_CONFUSION_OUT_COLS) if rows else empty


#: Imbalance-robust summaries surfaced per project alongside F1 for publication.
PUBLICATION_METRICS = ["f1", "mcc", "balanced_accuracy", "roc_auc", "cohen_kappa"]


def publication_metrics_by_project(df: pd.DataFrame) -> pd.DataFrame:
    """Per-project mean of every publication metric (held-out generalization cells).

    One row per project, one column per metric — the compact "here is how the
    model does under every standard summary, not just F1" table reviewers expect.
    Prefers generalization cells (the honest held-out split); falls back to the
    full-budget all-features ablation cells when generalization was not run.
    """
    if df.empty:
        return pd.DataFrame()
    src = df[df["analysis"] == "generalization"]
    if src.empty:
        abl = df[df["analysis"] == "ablation"]
        if not abl.empty and "config_name" in abl.columns:
            abl = _full_budget_only(abl[abl["config_name"] == ALL_FEATURES_CONFIG])
        src = abl
    if src.empty:
        return pd.DataFrame()
    out = None
    for metric in PUBLICATION_METRICS:
        if metric not in src.columns:
            continue
        summ = summarize_by(src, ["project_id", "project_name"], metric=metric)
        summ = summ.rename(columns={f"{metric}_mean": metric})[
            ["project_id", "project_name", metric]
        ]
        out = summ if out is None else out.merge(summ, on=["project_id", "project_name"], how="outer")
    return out if out is not None else pd.DataFrame()


def training_speed_by_project(df: pd.DataFrame) -> pd.DataFrame:
    """Median wall-clock training time per project (and per-cell median)."""
    if df.empty or "elapsed_sec_total" not in df.columns:
        return pd.DataFrame(columns=["project_id", "project_name", "median_sec", "mean_sec", "n"])
    clean = df[~df["error"].astype(bool)]
    out = []
    for key, grp in clean.groupby(["project_id", "project_name"], dropna=False):
        t = pd.to_numeric(grp["elapsed_sec_total"], errors="coerce").dropna()
        out.append(
            {
                "project_id": key[0],
                "project_name": key[1],
                "median_sec": float(np.median(t)) if len(t) else float("nan"),
                "mean_sec": float(np.mean(t)) if len(t) else float("nan"),
                "n": int(len(t)),
            }
        )
    return pd.DataFrame(out)


def data_efficiency_summary(knees: list[dict]) -> pd.DataFrame:
    """Summarize per-(project,behavior) learning-curve knees into one table.

    ``knees`` items: {project_id, project_name, behavior_name, knee_clips, f1_max}.
    """
    if not knees:
        return pd.DataFrame(columns=["project_id", "behavior_name", "knee_clips", "f1_max"])
    return pd.DataFrame(knees)


def cross_project_overview(df: pd.DataFrame, metric: str = "f1") -> dict:
    """One-line cross-project descriptive summary for the dashboard header."""
    if df.empty:
        return {"n_projects": 0, "n_behaviors": 0, f"{metric}_mean": float("nan")}
    clean = df[~df["error"].astype(bool)] if "error" in df.columns else df
    vals = pd.to_numeric(clean[metric], errors="coerce").dropna()
    return {
        "n_projects": int(clean["project_id"].nunique()),
        "n_behaviors": int(clean["behavior_id"].nunique()),
        "n_cells": int(len(clean)),
        f"{metric}_mean": float(vals.mean()) if len(vals) else float("nan"),
        f"{metric}_min": float(vals.min()) if len(vals) else float("nan"),
        f"{metric}_max": float(vals.max()) if len(vals) else float("nan"),
    }
