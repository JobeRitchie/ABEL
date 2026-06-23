"""Metric aggregation and comparison utilities for ablation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abel.benchmark.runner import RunResult


def load_behavior_names(project_root: str | Path) -> dict[str, str]:
    """Load behavior_id → short display name from behavior_definitions.yaml."""
    defs_path = Path(project_root) / "config" / "behavior_definitions.yaml"
    mapping: dict[str, str] = {}
    if not defs_path.exists():
        return mapping
    try:
        import yaml

        data = yaml.safe_load(defs_path.read_text(encoding="utf-8"))
        for beh in data.get("behaviors", []):
            bid = beh.get("behavior_id", "")
            short = beh.get("short_name", "") or beh.get("name", bid)
            if bid and bid != "no_behavior":
                mapping[bid] = short
    except Exception:
        pass
    return mapping


def apply_behavior_names(
    df: pd.DataFrame, names: dict[str, str],
) -> pd.DataFrame:
    """Replace behaviour IDs with short names in a DataFrame's Behavior column."""
    if "Behavior" in df.columns and names:
        df = df.copy()
        df["Behavior"] = df["Behavior"].map(lambda b: names.get(b, b))
    return df


def results_to_dataframe(results: list[RunResult]) -> pd.DataFrame:
    """Convert a list of RunResults to a summary DataFrame with mean±SEM."""
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append({
            "Run": r.run_name,
            "Behavior": r.behavior,
            "Precision": r.precision_mean,
            "Precision SEM": r.precision_sem,
            "Recall": r.recall_mean,
            "Recall SEM": r.recall_sem,
            "F1": r.f1_mean,
            "F1 SEM": r.f1_sem,
            "PR-AUC": r.pr_auc_mean,
            "PR-AUC SEM": r.pr_auc_sem,
            "Folds": r.n_folds,
            "Features": r.n_features,
            "Time (s)": round(r.elapsed_sec, 2),
            "Error": r.error,
        })
    return pd.DataFrame(rows)


def format_mean_sem(mean: float, sem: float, decimals: int = 3) -> str:
    """Format a metric as 'mean ± SEM'."""
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.{decimals}f} ± {sem:.{decimals}f}"


def compute_deltas(
    df: pd.DataFrame,
    baseline_name: str = "baseline_all_on",
    behavior: str | None = None,
) -> pd.DataFrame:
    """Compute metric deltas relative to the baseline run.

    If *behavior* is given, restrict to that behavior only.
    """
    sub = df if behavior is None else df[df["Behavior"] == behavior]
    metric_cols = ["Precision", "Recall", "F1", "PR-AUC"]
    baseline_row = sub[sub["Run"] == baseline_name]
    if baseline_row.empty:
        return pd.DataFrame()

    deltas: list[dict[str, Any]] = []
    baseline_vals = baseline_row.iloc[0]
    for _, row in sub.iterrows():
        if row["Run"] == baseline_name:
            continue
        d: dict[str, Any] = {"Run": row["Run"], "Behavior": row.get("Behavior", "")}
        for col in metric_cols:
            bv = float(baseline_vals[col])
            rv = float(row[col])
            if np.isnan(bv) or np.isnan(rv):
                d[f"Δ{col}"] = float("nan")
            else:
                d[f"Δ{col}"] = round(rv - bv, 4)
        deltas.append(d)
    return pd.DataFrame(deltas)


def rank_features_by_impact(df: pd.DataFrame, metric: str = "F1") -> pd.DataFrame:
    """Rank ablation toggles by their impact on the chosen metric.

    A larger negative delta means removing the feature hurts more (=more important).
    """
    deltas = compute_deltas(df)
    if deltas.empty:
        return deltas
    col = f"Δ{metric}"
    if col not in deltas.columns:
        return deltas
    ranked = deltas.sort_values(col, ascending=True).reset_index(drop=True)
    ranked["Rank"] = range(1, len(ranked) + 1)
    return ranked
