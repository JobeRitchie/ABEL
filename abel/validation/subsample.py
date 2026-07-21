"""Deterministic, group-aware subsampling of training clips for learning curves.

The unit of subsampling is a *labeled positive clip* of the target behavior, so
the learning-curve x-axis is "# positive clips labeled".  Positives are drawn
group-by-group (whole sessions/animals first) to mimic the realistic "you
labeled N clips across whatever sessions" scenario and avoid single-session
artefacts at small N.  Negatives follow a configurable policy.

Subsampling only ever draws from the *training pool* — the held-out evaluation
set is fixed across all sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALL_CLIPS = -1  # sentinel for "use all available positives"


def _positive_mask(df: pd.DataFrame, behavior_id: str) -> pd.Series:
    return df["label"].astype(str).str.strip() == str(behavior_id).strip()


def count_positives(df: pd.DataFrame, behavior_id: str) -> int:
    return int(_positive_mask(df, behavior_id).sum())


def draw(
    pool: pd.DataFrame,
    behavior_id: str,
    size: int,
    *,
    group_col: str = "session_id",
    seed: int = 0,
    neg_policy: str = "all",      # "all" | "ratio"
    neg_per_pos: float = 3.0,
) -> tuple[pd.DataFrame, int, int]:
    """Return a sub-pool with ``size`` positive clips (+ negatives per policy).

    Returns ``(subpool_df, n_pos, n_neg)``.  ``size == ALL_CLIPS`` (or ≥ available)
    uses every positive.  Guarantees ≥1 positive whenever any exist.
    """
    rng = np.random.default_rng(int(seed))
    pos_mask = _positive_mask(pool, behavior_id)
    pos_df = pool.loc[pos_mask]
    neg_df = pool.loc[~pos_mask]

    n_available = len(pos_df)
    if size == ALL_CLIPS or size >= n_available:
        chosen_pos = pos_df
    elif size <= 0:
        chosen_pos = pos_df.iloc[0:0]
    else:
        chosen_pos = _group_aware_take(pos_df, group_col, size, rng)
        if len(chosen_pos) == 0 and n_available > 0:
            # Always keep at least one positive at the smallest size.
            chosen_pos = pos_df.sample(n=1, random_state=int(seed))

    # Negative selection.
    if neg_policy == "ratio" and len(chosen_pos) > 0:
        cap = int(round(float(neg_per_pos) * len(chosen_pos)))
        if cap < len(neg_df):
            chosen_neg = _group_aware_take(neg_df, group_col, cap, rng)
        else:
            chosen_neg = neg_df
    else:
        chosen_neg = neg_df

    subpool = pd.concat([chosen_pos, chosen_neg], ignore_index=True)
    return subpool, int(len(chosen_pos)), int(len(chosen_neg))


def _group_aware_take(
    df: pd.DataFrame,
    group_col: str,
    n_target: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Accumulate rows group-by-group (shuffled) until ``n_target`` reached."""
    if group_col not in df.columns or len(df) == 0:
        if len(df) <= n_target:
            return df
        idx = rng.permutation(len(df))[:n_target]
        return df.iloc[np.sort(idx)]

    groups = list(df.groupby(group_col, sort=False))
    order = rng.permutation(len(groups))
    parts: list[pd.DataFrame] = []
    taken = 0
    for gi in order:
        _, gdf = groups[gi]
        if taken >= n_target:
            break
        remaining = n_target - taken
        if len(gdf) <= remaining:
            parts.append(gdf)
            taken += len(gdf)
        else:
            # Partial take from this group (deterministic via rng permutation).
            sub_idx = rng.permutation(len(gdf))[:remaining]
            parts.append(gdf.iloc[np.sort(sub_idx)])
            taken += remaining
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)
