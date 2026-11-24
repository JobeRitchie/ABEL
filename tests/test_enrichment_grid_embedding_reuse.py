"""Reviewed clips that cover a grid window reuse that window's video embedding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from abel.ui.tabs.active_learning_tab import ActiveLearningTab

COLS = ["rn18_000", "rn18_001"]


def _grid(animals=("A",)) -> pd.DataFrame:
    rows = []
    for a_i, animal in enumerate(animals):
        for k, start in enumerate((0, 15, 30)):
            rows.append({
                "segment_id": f"seg_{animal}_s1_{start}_{start + 14}",
                "session_id": "s1", "start_frame": start, "end_frame": start + 14,
                "animal_id": animal,
                "rn18_000": np.float32(1 + k + 10 * a_i), "rn18_001": np.float32(2 + k + 10 * a_i),
            })
    return pd.DataFrame(rows)


def _enriched(rows) -> pd.DataFrame:
    return pd.DataFrame([
        {"segment_id": f"quality_clip_x_s1_{s}_{e}", "session_id": "s1",
         "start_frame": s, "end_frame": e, "animal_id": a}
        for s, e, a in rows
    ])


def _run(enriched, grid):
    needs = pd.Series(True, index=enriched.index)
    return ActiveLearningTab._reuse_grid_embedding(enriched, grid, needs, COLS)


def test_matching_window_takes_the_grid_embedding():
    out, n = _run(_enriched([(15, 29, "A"), (18, 32, "A")]), _grid())
    assert n == 1
    assert out.loc[0, COLS].tolist() == [2.0, 3.0]
    assert out.loc[1, COLS].isna().all()  # shifted window: still needs the video


def test_two_animals_match_on_their_own_animal():
    out, n = _run(_enriched([(0, 14, "B"), (0, 14, "A")]), _grid(("A", "B")))
    assert n == 2
    assert out.loc[0, COLS].tolist() == [11.0, 12.0]
    assert out.loc[1, COLS].tolist() == [1.0, 2.0]


def test_two_animals_without_an_animal_on_the_clip_are_not_guessed():
    enriched = _enriched([(0, 14, "A")]).drop(columns="animal_id")
    assert _run(enriched, _grid(("A", "B"))) is None


def test_grid_rows_with_no_embedding_are_not_copied():
    grid = _grid()
    grid[COLS] = np.float32(0.0)
    assert _run(_enriched([(15, 29, "A")]), grid) is None
    grid[COLS] = np.nan
    assert _run(_enriched([(15, 29, "A")]), grid) is None
