"""Held-out subject/session selection — replaces any separate "gold" dataset.

Validation holds out a subset of subjects/sessions from training and evaluates
on *their already-reviewed/accepted clips* (the ground truth ABEL's pipeline
already produces).  The held-out evaluation set is filtered to **high-confidence
clips only** so we score against clean labels — never the model's own uncertain
candidates.  This filter applies ONLY to the held-out set; the training pool
keeps all its labels so learning-curve subsampling reflects real labeling effort.

Two filters here exist to defend the group-holdout guarantee against behaviour
*inside* the shipped trainer, and removing them silently reintroduces leakage:

**Refine-only rows must not reach the holdout frame.**  ``train_and_evaluate``
deliberately moves ``temporal_feedback`` (reviewer FP/FN corrections) and
``imported:*`` (other projects') rows OUT of the validation split and INTO
training — correct for the product, since those labels exist to *correct* the
model rather than to grade it.  But it does that *after* a ``precomputed_split``
is applied, so any such row we hand it inside the holdout frame becomes a
**training row belonging to a held-out session**.  Measured on real projects,
that leaked rows from 2 DG_FearConditioning and 2 DG_EPM sessions into training
while other rows from those same sessions were still being scored.  We therefore
drop refine-only rows from the held-out partition entirely: they cannot be
evaluated, and they must not be trained on either.  (``loso.py`` already does
this; this module did not.)

**Imported rows must not silently pad the training pool.**  ``TrainingConfig
.include_imported`` is only consumed by ``_load_training_frame``, which
``train_and_evaluate`` never calls — so the engine's ``include_imported=False``
was a no-op and 51% of DG_FearConditioning's pool was clips imported from another
project, counted by ``n_pos_train`` as this project's labeling effort.  We enforce
the intent here instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from abel.validation.datamodel import ProjectRef

# Label sources the trainer keeps for training but refuses to validate on.
_REFINE_ONLY_EXACT = {"temporal_feedback"}
_REFINE_ONLY_PREFIX = ("imported:",)


def median_clip_frames(project: ProjectRef,
                       df: pd.DataFrame | None = None) -> float:
    """Median labeled-clip length in frames, **measured** from the project's rows.

    The evaluated unit is one labeled clip, so any report of the confusion counts
    has to be able to say how long a clip is.  That must not be read from
    ``behavior_model.segment_window_frames``: its schema default is 60, but real
    projects set it from their own clip duration and most use ~0.5 s (~15 frames
    at 30 fps), so trusting the config default overstates the unit by ~4x.  The
    labeled rows carry their own frame bounds, so measure them.

    Returns NaN when the training set is missing or carries no frame bounds —
    callers must fall back to naming the unit without a duration rather than
    inventing one.
    """
    try:
        if df is None:
            if not project.training_set_path.exists():
                return float("nan")
            df = pd.read_parquet(project.training_set_path,
                                 columns=["start_frame", "end_frame"])
        if not {"start_frame", "end_frame"} <= set(df.columns) or df.empty:
            return float("nan")
        span = (pd.to_numeric(df["end_frame"], errors="coerce")
                - pd.to_numeric(df["start_frame"], errors="coerce") + 1).dropna()
        span = span[span > 0]
        return float(span.median()) if len(span) else float("nan")
    except Exception:  # noqa: BLE001 — a label for a figure must never sink a run
        return float("nan")


def clip_unit_label(frames: float, fps: float) -> str:
    """Human phrase for the evaluated unit, e.g. ``"labeled clips (~0.5 s)"``.

    Degrades to the bare unit when the length could not be measured, because a
    guessed duration is worse than none — it is the number a reader would quote.
    """
    if not np.isfinite(frames) or frames <= 0:
        return "labeled clips"
    if not np.isfinite(fps) or fps <= 0:
        return f"labeled clips (~{int(round(frames))} frames)"
    sec = float(frames) / float(fps)
    shown = f"{sec:.1f}" if sec >= 0.1 else f"{sec:.2f}"
    return f"labeled clips (~{shown} s)"


def _group_column(strategy: str) -> str:
    """Column to partition train vs. held-out by (mirrors trainer._split)."""
    return "animal_id" if str(strategy).endswith("subject") else "session_id"


def is_refine_only(df: pd.DataFrame) -> pd.Series:
    """Rows the trainer will move out of validation and into training."""
    if "label_source" not in df.columns:
        return pd.Series(False, index=df.index)
    s = df["label_source"].astype(str)
    mask = s.isin(_REFINE_ONLY_EXACT)
    for pref in _REFINE_ONLY_PREFIX:
        mask = mask | s.str.startswith(pref)
    return mask


def is_imported(df: pd.DataFrame) -> pd.Series:
    """Rows imported from another project (cross-project examples)."""
    if "label_source" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["label_source"].astype(str).str.startswith("imported:")


@dataclass
class HoldoutSplit:
    """A leakage-checked train-pool / held-out split for one project."""

    train_pool: pd.DataFrame
    holdout: pd.DataFrame
    group_col: str
    holdout_groups: list[str]
    train_groups: list[str]
    min_confidence: float
    # Rows removed so the trainer cannot smuggle held-out sessions into training,
    # and so imported clips cannot pad this project's labeling effort.
    n_refine_only_dropped: int = 0
    n_imported_dropped: int = 0
    excluded_imported: bool = True
    # Holdout groups left with nothing to score after filtering (they were made
    # entirely of refine-only rows) — advertising them as "held out" would be a lie.
    unevaluable_groups: list[str] = field(default_factory=list)

    def manifest(self, project: ProjectRef) -> dict:
        return {
            "project_id": project.project_id,
            "group_col": self.group_col,
            # Only groups that actually contribute scored rows. The pre-filter list
            # over-advertised the holdout by 21 of 40 groups on DG_FearConditioning.
            "holdout_groups": sorted(self.holdout_groups),
            "unevaluable_holdout_groups": sorted(self.unevaluable_groups),
            "train_groups": sorted(self.train_groups),
            "min_confidence": self.min_confidence,
            "n_train_pool_rows": int(len(self.train_pool)),
            "n_holdout_rows": int(len(self.holdout)),
            "n_refine_only_dropped": int(self.n_refine_only_dropped),
            "n_imported_dropped_from_pool": int(self.n_imported_dropped),
            "excluded_imported_from_pool": bool(self.excluded_imported),
        }


def choose_holdout_groups(
    df: pd.DataFrame,
    group_col: str,
    *,
    test_size: float = 0.25,
    seed: int = 42,
) -> list[str]:
    """Pick held-out groups via the same GroupShuffleSplit the product uses."""
    groups = df[group_col].astype(str)
    unique = sorted(groups.unique())
    if len(unique) < 2:
        # Degenerate: can't hold out a whole group; hold out none here and let
        # the caller decide (the engine would otherwise self-split).
        return []
    rng = np.random.default_rng(int(seed))
    n_hold = max(1, int(round(float(test_size) * len(unique))))
    n_hold = min(len(unique) - 1, n_hold)
    chosen = rng.choice(np.asarray(unique, dtype=object), size=n_hold, replace=False)
    return sorted(str(g) for g in chosen)


def split(
    project: ProjectRef,
    *,
    holdout_groups: Iterable[str] | None = None,
    min_confidence: float = 1.0,
    test_size: float = 0.25,
    seed: int = 42,
    df: pd.DataFrame | None = None,
    exclude_imported: bool = True,
) -> HoldoutSplit:
    """Partition a project's training set into (train_pool, high-conf holdout).

    Parameters
    ----------
    holdout_groups:
        Explicit sessions/animals to hold out.  When ``None`` they are drawn by
        a seeded GroupShuffleSplit over the project's split strategy.
    min_confidence:
        Held-out rows with ``reviewer_confidence`` below this are dropped from
        the evaluation set (default 1.0 = accepted/high-confidence only).
    df:
        Optional pre-loaded training frame (else read from disk).
    """
    if df is None:
        df = pd.read_parquet(project.training_set_path)
    df = df.reset_index(drop=True)

    group_col = _group_column(project.split_strategy)
    if group_col not in df.columns:
        raise ValueError(f"Training set has no '{group_col}' column for holdout split.")

    if holdout_groups is None:
        holdout_groups = choose_holdout_groups(
            df, group_col, test_size=test_size, seed=seed
        )
    holdout_set = {str(g) for g in holdout_groups}

    g = df[group_col].astype(str)
    holdout_mask = g.isin(holdout_set)
    train_pool = df.loc[~holdout_mask].reset_index(drop=True)
    holdout = df.loc[holdout_mask].reset_index(drop=True)

    # ── Defend the group guarantee against the trainer's refine-only reshuffle ──
    # Any temporal_feedback / imported row left in the holdout frame is silently
    # moved INTO training by train_and_evaluate — i.e. a held-out session's rows
    # would train the model. They cannot be scored either, so drop them outright:
    # a held-out session must contribute nothing at all.
    refine = is_refine_only(holdout)
    n_refine = int(refine.sum())
    if n_refine:
        holdout = holdout.loc[~refine].reset_index(drop=True)

    # High-confidence filter applies ONLY to the held-out evaluation set.
    if "reviewer_confidence" in holdout.columns and len(holdout):
        conf = pd.to_numeric(holdout["reviewer_confidence"], errors="coerce").fillna(0.0)
        holdout = holdout.loc[conf >= float(min_confidence)].reset_index(drop=True)

    # ── Enforce include_imported=False, which the trainer ignores ──
    # Cross-project clips would otherwise pad the pool (51% of DG_FearConditioning)
    # and be counted by n_pos_train as THIS project's labeling effort.
    n_imported = 0
    if exclude_imported and len(train_pool):
        imp = is_imported(train_pool)
        n_imported = int(imp.sum())
        if n_imported:
            train_pool = train_pool.loc[~imp].reset_index(drop=True)

    # Groups that survived selection but have nothing left to score.
    scored_groups = (
        set(holdout[group_col].astype(str)) if len(holdout) else set()
    )
    unevaluable = sorted(holdout_set - scored_groups)

    train_groups = sorted(set(train_pool[group_col].astype(str))) if len(train_pool) else []

    _assert_no_leakage(train_pool, holdout, group_col)

    return HoldoutSplit(
        train_pool=train_pool,
        holdout=holdout,
        group_col=group_col,
        # Advertise only the groups that actually contribute scored rows.
        holdout_groups=sorted(scored_groups),
        train_groups=train_groups,
        min_confidence=float(min_confidence),
        n_refine_only_dropped=n_refine,
        n_imported_dropped=n_imported,
        excluded_imported=bool(exclude_imported),
        unevaluable_groups=unevaluable,
    )


def _assert_no_leakage(train_pool: pd.DataFrame, holdout: pd.DataFrame, group_col: str) -> None:
    """Hard guard: no shared group, no shared segment_id, no trainer-smuggled rows.

    The refine-only check is the important one: those rows look perfectly innocent
    here, but ``train_and_evaluate`` will relocate them into the training split, so
    leaving even one in the holdout frame silently trains the model on a held-out
    session. Asserting on group/segment overlap alone gave false assurance.
    """
    if not len(train_pool) or not len(holdout):
        return

    smuggled = int(is_refine_only(holdout).sum())
    if smuggled:
        raise AssertionError(
            f"Holdout leakage: {smuggled} refine-only row(s) (temporal_feedback / "
            f"imported) remain in the holdout frame. The trainer moves these INTO "
            f"training, which would train on held-out sessions."
        )
    g_train = set(train_pool[group_col].astype(str))
    g_hold = set(holdout[group_col].astype(str))
    overlap_groups = g_train & g_hold
    if overlap_groups:
        raise AssertionError(
            f"Holdout leakage: {len(overlap_groups)} group(s) appear in both "
            f"train pool and holdout: {sorted(overlap_groups)[:5]}"
        )
    if "segment_id" in train_pool.columns and "segment_id" in holdout.columns:
        s_train = set(train_pool["segment_id"].astype(str))
        s_hold = set(holdout["segment_id"].astype(str))
        overlap_seg = s_train & s_hold
        if overlap_seg:
            raise AssertionError(
                f"Holdout leakage: {len(overlap_seg)} segment_id(s) shared "
                f"between train pool and holdout."
            )
