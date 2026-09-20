"""Dense temporal inference on multi-animal sessions.

The representation frame table carries one row per (frame, animal).  Dense
inference used to treat those rows as the time axis, which in a two-animal
session halved every window's real duration, blended both animals into every
window, and stretched the probability trace over twice as many frames as the
video has -- the back half of every trace was flat zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abel.temporal_refinement.temporal_refinement_service import (
    TemporalRefinementService,
)


def _two_animal_frames(n_frames: int = 400, n_features: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for animal in ("track_0", "track_1"):
        df = pd.DataFrame(
            rng.normal(size=(n_frames, n_features)).astype(np.float32),
            columns=[f"feat_{i}" for i in range(n_features)],
        )
        df["frame"] = np.arange(n_frames, dtype=int)
        df["animal_id"] = animal
        df["session_id"] = "session_a"
        rows.append(df)
    # Interleaved by frame, which is how the representation table is sorted.
    return pd.concat(rows, ignore_index=True).sort_values("frame").reset_index(drop=True)


def test_session_frame_count_is_the_frame_axis_not_the_row_count() -> None:
    df = _two_animal_frames(n_frames=400)
    assert len(df) == 800
    assert TemporalRefinementService._session_frame_count(df) == 400


def test_session_frame_count_falls_back_to_row_count_without_a_frame_column() -> None:
    df = pd.DataFrame({"feat_0": [0.0, 1.0, 2.0]})
    assert TemporalRefinementService._session_frame_count(df) == 3


def test_per_animal_windows_span_the_full_window_duration() -> None:
    df = _two_animal_frames(n_frames=400)
    feature_cols = [f"feat_{i}" for i in range(6)]
    window_frames, step_frames = 16, 8

    mixed = TemporalRefinementService._build_dense_windows_for_session(
        df, feature_cols, window_frames, step_frames
    )
    # The mixed table is what the old code windowed: each window covers only
    # half the frames it was trained on, and is attributed to one animal.
    mixed_span = (mixed["end_frame"] - mixed["start_frame"]).median()
    assert mixed_span < window_frames - 1
    assert mixed["animal_id"].nunique() == 1

    spans = {}
    for animal_id, grp in df.groupby("animal_id"):
        windows = TemporalRefinementService._build_dense_windows_for_session(
            grp.reset_index(drop=True), feature_cols, window_frames, step_frames
        )
        assert set(windows["animal_id"]) == {animal_id}
        assert int(windows["end_frame"].max()) >= 400 - window_frames
        spans[animal_id] = (windows["end_frame"] - windows["start_frame"]).median()

    assert set(spans) == {"track_0", "track_1"}
    assert all(span == window_frames - 1 for span in spans.values())
