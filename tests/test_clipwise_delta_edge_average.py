"""Clip-wise _delta should be an edge-band average, robust to endpoint jitter.

Previously ``_delta`` was ``last_frame - first_frame``, so a single noisy frame
at either window boundary corrupted the net-displacement signal.  It is now
``mean(last k) - mean(first k)`` with k ~= 25% of the window.  These tests assert:

1. on a clean linear ramp the delta still reflects the true net change, and
2. a single-frame glitch at a boundary barely moves the new delta but would have
   wrecked the old last-minus-first delta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abel.utils.gpu_feature_ops import build_segment_df_fast


# A column name that _is_roi_spatial_col() recognises, so it gets _delta/_trend.
DELTA_COL = "nose_to_target_dist"


def _frame_df(values: np.ndarray) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "animal_id": "a1",
            "session_id": "s1",
            DELTA_COL: values.astype(float),
        }
    )


def _single_window_delta(values: np.ndarray) -> float:
    df = _frame_df(values)
    out = build_segment_df_fast(
        df, [DELTA_COL], "a1", "s1",
        window_size=len(values), stride=len(values),
        include_periodicity=False, include_posture_deltas=False,
    )
    assert len(out) == 1
    return float(out[f"{DELTA_COL}_delta"].iloc[0])


def test_linear_ramp_delta_is_reasonable():
    # Ramp 0..59: true end-to-start change is 59. Edge-averaged delta is a bit
    # smaller (it compares band centroids) but must stay strongly positive and
    # close to the span.
    vals = np.linspace(0.0, 59.0, 60)
    d = _single_window_delta(vals)
    assert d > 0
    assert 35.0 <= d <= 59.0  # edge bands centroids ~ (45 - 7.5) = ~44


def test_endpoint_glitch_barely_moves_edge_delta():
    n = 60
    clean = np.full(n, 10.0)
    clean_delta = _single_window_delta(clean)  # ~0, flat signal

    glitched = clean.copy()
    glitched[-1] = 1000.0  # one corrupted boundary frame

    new_delta = _single_window_delta(glitched)
    old_delta = glitched[-1] - glitched[0]  # what the previous impl returned

    # Old impl would report ~990; new impl spreads it over k frames (k=15) so
    # the corruption is attenuated by ~k and stays far smaller.
    assert abs(new_delta) < abs(old_delta) / 5
    assert abs(new_delta) < 100.0


def test_delta_sign_tracks_direction():
    approaching = np.linspace(50.0, 5.0, 60)   # distance shrinking
    retreating = np.linspace(5.0, 50.0, 60)    # distance growing
    assert _single_window_delta(approaching) < 0
    assert _single_window_delta(retreating) > 0
