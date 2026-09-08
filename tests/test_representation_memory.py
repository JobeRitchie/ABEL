"""Memory-frugal representation build: parity of the bincount z-score path.

The transform-based z-scoring allocated a full n_rows x n_features float64
frame per statistic and died with an 11.4 GiB MemoryError on a 9.4M-frame
project.  These tests pin the replacement to the semantics of the original
implementation — including the NaN, zero-variance and single-row edge cases —
and cover the float32 downcast the large-table path relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services.behavior_representation_service import BehaviorRepresentationService


def _transform_reference(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """The previous groupby(...).transform implementation, as ground truth."""
    out = df.copy()
    grp = out.groupby(["animal_id", "session_id"])[feature_cols]
    mu = grp.transform("mean")
    sigma = grp.transform("std").fillna(1.0).replace(0.0, 1.0)
    out[feature_cols] = (out[feature_cols] - mu) / sigma
    return out


def _frame_with_edge_cases() -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(17)
    rows = []
    for animal in ("a1", "a2"):
        for sid in ("s1", "s2", "s3"):
            for fr in range(int(rng.integers(1, 40))):  # includes single-row groups
                rows.append(
                    {
                        "animal_id": animal,
                        "session_id": sid,
                        "frame": fr,
                        "f0": float(rng.normal()),
                        "const": 5.0,  # zero variance
                        "sparse": float("nan") if rng.random() < 0.4 else float(rng.normal()),
                        "empty": float("nan"),  # entirely missing
                    }
                )
    return pd.DataFrame(rows), ["f0", "const", "sparse", "empty"]


def test_bincount_zscore_matches_transform_including_nans():
    df, cols = _frame_with_edge_cases()
    got, _ = BehaviorRepresentationService._zscore_by_group_with_stats(df, cols)
    pd.testing.assert_frame_equal(got, _transform_reference(df, cols), check_exact=False)


def test_zscore_leaves_input_untouched_by_default():
    df, cols = _frame_with_edge_cases()
    before = df.copy()
    BehaviorRepresentationService._zscore_by_group_with_stats(df, cols)
    pd.testing.assert_frame_equal(df, before)


def test_zscore_copy_false_scales_in_place():
    """build() relies on this: the frame table is too large to duplicate."""
    df, cols = _frame_with_edge_cases()
    expected = _transform_reference(df, cols)
    out, _ = BehaviorRepresentationService._zscore_by_group_with_stats(df, cols, copy=False)
    assert out is df
    pd.testing.assert_frame_equal(df, expected, check_exact=False)


def test_zscore_preserves_float32_columns():
    df, cols = _frame_with_edge_cases()
    f64 = _transform_reference(df, cols)
    for c in cols:
        df[c] = df[c].astype(np.float32)
    got, _ = BehaviorRepresentationService._zscore_by_group_with_stats(df, cols)
    assert all(got[c].dtype == np.float32 for c in cols)
    np.testing.assert_allclose(
        got["f0"].to_numpy(), f64["f0"].to_numpy(), rtol=1e-5, atol=1e-6
    )


def test_zscore_handles_rows_with_missing_group_key():
    df = pd.DataFrame(
        {
            "animal_id": ["a1", "a1", "a1", "a1", None, None],
            "session_id": ["s1", "s1", "s2", "s2", "s9", "s9"],
            "f0": [1.0, 3.0, 2.0, 6.0, 10.0, 20.0],
        }
    )
    got, stats = BehaviorRepresentationService._zscore_by_group_with_stats(df, ["f0"])
    pd.testing.assert_frame_equal(got, _transform_reference(df, ["f0"]), check_exact=False)
    assert len(stats) == 2  # the unkeyed rows form no group
    assert got["f0"].tail(2).isna().all()


def test_zscore_stats_match_pandas_aggregates():
    df, cols = _frame_with_edge_cases()
    _, stats = BehaviorRepresentationService._zscore_by_group_with_stats(df, cols)
    grp = df.groupby(["animal_id", "session_id"])[cols]
    expected = (
        grp.mean()
        .add_suffix("__mean")
        .join(grp.std().fillna(1.0).replace(0.0, 1.0).add_suffix("__std"))
        .reset_index()
    )
    pd.testing.assert_frame_equal(stats, expected, check_exact=False)


@pytest.mark.parametrize("triggered", [False, True])
def test_downcast_only_fires_on_large_tables(monkeypatch, triggered):
    df = pd.DataFrame(
        {
            "session_id": ["s1"] * 100,
            "a": np.arange(100, dtype=np.float64),
            "b": np.arange(100, dtype=np.float32),
        }
    )
    threshold = 10 if triggered else 10**12
    monkeypatch.setattr(
        BehaviorRepresentationService, "DOWNCAST_THRESHOLD_BYTES", threshold
    )
    msgs: list[str] = []
    out = BehaviorRepresentationService._downcast_large_float_table(df, "pose", msgs.append)
    assert out["a"].dtype == (np.float32 if triggered else np.float64)
    assert out["b"].dtype == np.float32  # already float32, untouched
    assert bool(msgs) is triggered
    np.testing.assert_allclose(out["a"].to_numpy(), np.arange(100), rtol=1e-6)
