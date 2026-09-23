"""The held-out split gives validation a fair share of the target's positives.

A plain group shuffle picks subjects blind to the label, so with 30 dyads and a
10% split a rare behavior's validation set routinely held 0-3 positives and the
Model Overview scored it on nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abel.services.active_learning_trainer_service import (
    MIN_VAL_POSITIVES,
    ActiveLearningTrainerService as T,
)


def _df(n_groups: int = 30, per_group: int = 100, pos_groups: int = 6, pos_per: int = 8, seed: int = 0):
    rng = np.random.RandomState(seed)
    rows, pos = [], []
    for g in range(n_groups):
        for i in range(per_group):
            rows.append({"session_id": f"s{g}", "label": "x"})
            pos.append(g < pos_groups and i < pos_per)
    order = rng.permutation(len(rows))
    df = pd.DataFrame(rows).iloc[order].reset_index(drop=True)
    return df, np.asarray(pos)[order]


def test_every_seed_puts_positives_on_both_sides() -> None:
    df, pos = _df()
    for seed in range(20):
        tr, va = T._split(df, "group_shuffle_session", 0.1, seed, positive_mask=pos)
        assert pos[va].sum() > 0 and pos[tr].sum() > 0
        assert set(df.session_id.iloc[tr]).isdisjoint(df.session_id.iloc[va])


def test_rare_behavior_gets_enough_validation_positives() -> None:
    # 48 positives in 6 of 30 groups: a 10% split would hold ~5.
    df, pos = _df()
    tr, va = T._split(df, "group_shuffle_session", 0.1, 3, positive_mask=pos)
    assert pos[va].sum() >= MIN_VAL_POSITIVES - 2
    # Rows grow with the positives, so validation prevalence stays honest.
    assert abs(pos[va].mean() - pos.mean()) < 0.5 * pos.mean()


def test_common_behavior_keeps_the_configured_share() -> None:
    df, pos = _df(pos_groups=30, pos_per=20)
    _, va = T._split(df, "group_shuffle_session", 0.1, 1, positive_mask=pos)
    assert abs(len(va) / len(df) - 0.1) < 0.04
    assert abs(pos[va].sum() / pos.sum() - 0.1) < 0.04


def test_without_a_mask_the_split_is_unchanged() -> None:
    df, _ = _df()
    a = T._split(df, "group_shuffle_session", 0.1, 7)
    b = T._split(df, "group_shuffle_session", 0.1, 7, positive_mask=None)
    assert np.array_equal(a[1], b[1])
