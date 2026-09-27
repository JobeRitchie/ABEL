"""Focused analysis: how much do the *social* (inter-animal) features buy us?

Multi-animal projects train on a ``social_*`` family (distance to the nearest
animal, approach and radial velocity, facing and heading alignment, contact state
and its duration).  The ablation already asks "does social help on top of pose
alone" (its add-one-in ``+ Social features`` rung).  This asks the question a user
of the shipped model cares about: **take the full shipped feature set and remove
only the social family; what is lost?**

For each (project, behavior) it trains ABEL's real classifier twice on the same
held-out split and the same per-seed subsample, once with every family the pool
carries except social and once with social added.  The per-seed F1 difference is
the paired value of the social family.  Solo-animal projects carry no informative
social columns and are skipped, so this layer only ever reports on the
multi-animal data.  The unit held out is the session (the dyad), never a track:
see :func:`abel.validation.holdout._group_column`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.validation import features, holdout, subsample, video_value
from abel.validation.datamodel import ProjectRef

ARM_NO_SOCIAL = "no_social"
ARM_WITH_SOCIAL = "with_social"

SocialValueResult = video_value.VideoValueResult


def has_social_features(pool: pd.DataFrame) -> bool:
    """True when the pool carries social columns that actually vary."""
    return bool(features.informative_cols(pool, features.social_only_cols(pool)))


def arm_columns(pool: pd.DataFrame, project: ProjectRef) -> tuple[list[str], list[str]]:
    """(without social, with social), everything else as the project ships it."""
    has_context = bool(features.informative_cols(pool, features.context_only_cols(pool)))
    common = dict(include_video=bool(project.use_video_features), include_context=has_context)
    return (features.select_feature_cols(pool, include_social=False, **common),
            features.select_feature_cols(pool, include_social=True, **common))


def run_social_value(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_id: str,
    holdout_split: holdout.HoldoutSplit,
    *,
    n_seeds: int = 5,
    progress_cb: Callable[[str], None] | None = None,
) -> SocialValueResult:
    """Train with vs. without the social family (paired per seed) for one behavior."""
    pool = holdout_split.train_pool
    res = SocialValueResult(
        project_id=project.project_id, behavior_id=str(behavior_id),
        behavior_name=project.behavior_label(behavior_id), n_seeds=int(n_seeds),
        n_pos_holdout=int(subsample.count_positives(
            holdout_split.holdout, behavior_id,
            co_occurring=project.allow_co_occurring_behaviors)),
    )
    if not has_social_features(pool):
        res.error = "project has no informative social features (single-animal)"
        return res
    cols_off, cols_on = arm_columns(pool, project)
    return video_value.run_paired_arms(
        trainer, project, behavior_id, holdout_split, res, cols_off, cols_on,
        n_seeds=n_seeds, arm_names=(ARM_NO_SOCIAL, ARM_WITH_SOCIAL),
        progress_cb=progress_cb)


def results_to_frame(results: list[SocialValueResult]) -> pd.DataFrame:
    """Tidy table with BH q-values, columns named for the social family."""
    df = video_value.results_to_frame(results)
    if df.empty:
        return df
    return df.rename(columns=lambda c: c.replace("no_video", "no_social")
                     .replace("with_video", "with_social"))


def plot_social_value(results: list[SocialValueResult], save_path: Path) -> Path:
    return video_value.plot_video_value(
        results, save_path,
        off_label="Shipped features minus social",
        on_label="+ Social (inter-animal) features",
        family="social features",
        title="Value of social features (paired: same split & subsample, "
              "social family on vs. off)")
