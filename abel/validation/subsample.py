"""Deterministic, group-aware subsampling of training clips for learning curves.

The unit of subsampling is a *labeled positive clip* of the target behavior, so
the learning-curve x-axis is "# positive clips labeled".  Positives are drawn
group-by-group (whole sessions/animals first) to mimic the realistic "you
labeled N clips across whatever sessions" scenario and avoid single-session
artefacts at small N.  Negatives follow a configurable policy.

Subsampling only ever draws from the *training pool*, the held-out evaluation
set is fixed across all sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALL_CLIPS = -1  # sentinel for "use all available positives"


def positive_mask(df: pd.DataFrame, behavior_id: str, *, co_occurring: bool = False) -> pd.Series:
    """Rows the trainer will treat as positives for ``behavior_id``.

    With ``co_occurring`` (the project's ``allow_co_occurring_behaviors``) the
    trainer expands a pipe-joined label such as ``"Chase|Sniff"`` into one row per
    behavior, so the clip is a positive for each of them.  Multi-animal projects
    pool co-occurring labels this way.  Exact matching would count those clips as
    negatives and understate the positives a learning-curve point was trained on.
    """
    target = str(behavior_id).strip()
    labels = df["label"].astype(str)
    exact = labels.str.strip() == target
    if not co_occurring:
        return exact
    piped = labels.str.contains("|", regex=False)
    if not piped.any():
        return exact
    in_pipe = labels[piped].map(lambda s: target in {t.strip() for t in s.split("|")})
    return exact | in_pipe.reindex(df.index, fill_value=False).astype(bool)


def dropped_sibling_mask(df: pd.DataFrame, behavior_id: str, *, co_occurring: bool = False) -> pd.Series:
    """Pipe-joined rows the trainer DROPS when training ``behavior_id``.

    With co-occurring labels on, a clip labeled ``"B|C"`` is neither a positive nor
    a negative for ``A``: the trainer expands it and discards both halves rather
    than make it a negative.  Counting such rows as negatives overstates the
    negatives a cell trained on and lets a ratio draw spend its budget on them.
    """
    if not co_occurring:
        return pd.Series(False, index=df.index)
    piped = df["label"].astype(str).str.contains("|", regex=False)
    return piped & ~positive_mask(df, behavior_id, co_occurring=True)


# Kept for callers that predate the co-occurring flag.
_positive_mask = positive_mask


def count_positives(df: pd.DataFrame, behavior_id: str, *, co_occurring: bool = False) -> int:
    return int(positive_mask(df, behavior_id, co_occurring=co_occurring).sum())


def draw(
    pool: pd.DataFrame,
    behavior_id: str,
    size: int,
    *,
    group_col: str = "session_id",
    seed: int = 0,
    neg_policy: str = "all",      # "all" | "ratio"
    neg_per_pos: float = 3.0,
    co_occurring: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    """Return a sub-pool with ``size`` positive clips (+ negatives per policy).

    Returns ``(subpool_df, n_pos, n_neg)``.  ``size == ALL_CLIPS`` (or ≥ available)
    uses every positive.  Guarantees ≥1 positive whenever any exist.
    """
    rng = np.random.default_rng(int(seed))
    pos_mask = positive_mask(pool, behavior_id, co_occurring=co_occurring)
    pos_df = pool.loc[pos_mask]
    neg_df = pool.loc[~pos_mask & ~dropped_sibling_mask(pool, behavior_id, co_occurring=co_occurring)]

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
