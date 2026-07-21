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


def accuracy_by_behavior(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    src = df[df["analysis"].isin(["generalization", "ablation"])]
    if not src.empty and "config_name" in src.columns:
        abl = _full_budget_only(src[(src["analysis"] == "ablation")
                                    & (src["config_name"] == ALL_FEATURES_CONFIG)])
        src = pd.concat([src[src["analysis"] == "generalization"], abl], ignore_index=True)
    if src.empty:
        src = df
    return summarize_by(src, ["project_id", "behavior_name"], metric=metric)


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
