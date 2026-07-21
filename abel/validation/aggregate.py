"""Flatten CellResults into the tidy ``cells.parquet`` meta-analysis substrate.

One row per ``(project, behavior, analysis, config, n_clips, seed)`` atom.  Every
dashboard, learning curve, and ablation chart is a groupby over this single wide
table.  A :func:`to_long` helper provides the ``(..., metric, value)`` long form
when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from abel.validation.datamodel import CellResult

_METRIC_COLS = [
    "precision", "recall", "f1", "pr_auc", "cohen_kappa",
    "mcc", "balanced_accuracy", "specificity", "roc_auc",
]


def cells_to_frame(cells: Iterable[CellResult]) -> pd.DataFrame:
    """Wide tidy frame: one row per cell with all metric + bookkeeping columns."""
    rows = [c.to_row() for c in cells]
    if not rows:
        return pd.DataFrame(
            columns=[
                "project_id", "project_name", "behavior_id", "behavior_name",
                "analysis", "config_name", "n_clips", "seed",
                *_METRIC_COLS,
                "tp", "fp", "fn", "tn",
                "n_pos_train", "n_neg_train", "n_features",
                "elapsed_sec_fit", "elapsed_sec_total", "degenerate", "error",
                "arrays_ref",
            ]
        )
    return pd.DataFrame(rows)


def save_cells(cells: Iterable[CellResult], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cells_to_frame(cells).to_parquet(path, index=False)
    return path


def load_cells(path: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Melt the wide frame into ``[..., metric, value]`` long form."""
    id_cols = [
        "project_id", "project_name", "behavior_id", "behavior_name",
        "analysis", "config_name", "n_clips", "seed",
    ]
    keep = [c for c in id_cols if c in df.columns]
    value_cols = [c for c in _METRIC_COLS if c in df.columns]
    return df.melt(id_vars=keep, value_vars=value_cols, var_name="metric", value_name="value")


def _agg_mean_ci(series: pd.Series) -> tuple[float, float]:
    """Mean and 95% CI half-width (t-based — see :func:`metrics.ci95`)."""
    from abel.validation import metrics as vmetrics  # noqa: PLC0415

    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return float("nan"), 0.0
    if len(vals) < 2:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), vmetrics.ci95(vals)


def summarize_by(df: pd.DataFrame, by: list[str], metric: str = "f1") -> pd.DataFrame:
    """Mean ± 95% CI of a metric grouped by the given columns (errors excluded)."""
    if df.empty:
        return pd.DataFrame(columns=[*by, f"{metric}_mean", f"{metric}_ci", "n"])
    clean = df[~df["error"].astype(bool)] if "error" in df.columns else df
    out = []
    for key, grp in clean.groupby(by, dropna=False):
        mean, ci = _agg_mean_ci(grp[metric])
        key_tuple = key if isinstance(key, tuple) else (key,)
        out.append({**dict(zip(by, key_tuple)), f"{metric}_mean": mean, f"{metric}_ci": ci, "n": len(grp)})
    return pd.DataFrame(out)
