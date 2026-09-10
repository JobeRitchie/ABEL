"""Segment summary statistics are stored as float32 (representation_v5).

The statistics were always *computed* in float32 — ``build_segment_df_fast``
reads its group block at that width — but the final assembly widened every
column back to float64.  That doubled the segment table (7.9 GiB for a
628k-window project) and, because a full-table copy consolidates every numeric
column into one contiguous block, made a routine append ask the allocator for a
single 9.2 GiB block and fail partway through a Pipeline-All run.

These tests pin the width at each producer, and pin the value parity that
justifies it: nothing here changes what the statistics measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services.behavior_representation_service import (
    BehaviorRepresentationService,
    RepresentationConfig,
)
from abel.utils.gpu_feature_ops import build_segment_df_fast, windowed_feature_summary


def _group(n_frames: int = 400, seed: int = 3) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    # A distance column so the ROI/target delta+trend branch is exercised too.
    feature_cols = ["speed", "nose_to_tail_dist", "center_to_target_dist"]
    df = pd.DataFrame(
        {
            "frame": np.arange(n_frames),
            "animal_id": "m1",
            "session_id": "s1",
            **{c: rng.normal(0.0, 1.0, n_frames) for c in feature_cols},
        }
    )
    return df, feature_cols


def test_segment_builder_emits_no_float64() -> None:
    df, cols = _group()
    seg = build_segment_df_fast(df, cols, "m1", "s1", 30, 15, True, True)

    assert not seg.empty
    float64_cols = [c for c in seg.columns if seg[c].dtype == np.float64]
    assert float64_cols == [], f"float64 leaked into the segment table: {float64_cols}"
    # The delta/trend branch must not be the odd one out.
    assert seg["center_to_target_dist_delta"].dtype == np.float32
    assert seg["center_to_target_dist_trend"].dtype == np.float32


def test_windowed_stats_are_float32_including_percentiles() -> None:
    """np.percentile returns float64 even from a float32 input; p10/p90 are cast."""
    rng = np.random.default_rng(11)
    data = rng.normal(0.0, 1.0, (200, 6)).astype(np.float32)
    stats = windowed_feature_summary(data, 30, 15, include_periodicity=True)

    assert set(stats) >= {"mean", "std", "median", "max", "p10", "p90", "energy"}
    for name, arr in stats.items():
        assert arr.dtype == np.float32, f"{name} is {arr.dtype}, not float32"


@pytest.mark.parametrize("window", [8, 30, 60])
def test_float32_storage_does_not_change_the_statistics(window: int) -> None:
    """Storing at float32 is lossless here: the stats are computed at that width."""
    rng = np.random.default_rng(5)
    data = np.concatenate(
        [rng.normal(0.0, 1.0, (600, 8)), rng.uniform(0.0, 900.0, (600, 4))], axis=1
    ).astype(np.float32)
    stats = windowed_feature_summary(data, window, window // 2, include_periodicity=True)

    for name, arr in stats.items():
        widened = arr.astype(np.float64)
        assert np.array_equal(
            widened.astype(np.float32), arr
        ), f"{name} does not survive the float32 round-trip"


def test_downcast_segment_features_leaves_non_float_columns_alone() -> None:
    df = pd.DataFrame(
        {
            "segment_id": ["a", "b"],
            "start_frame": np.array([0, 30], dtype=np.int64),
            "speed_mean": np.array([1.5, 2.5], dtype=np.float64),
            "speed_std": np.array([0.5, 0.25], dtype=np.float32),
        }
    )
    out = BehaviorRepresentationService.downcast_segment_features(df)

    assert out["speed_mean"].dtype == np.float32
    assert out["speed_std"].dtype == np.float32
    assert out["start_frame"].dtype == np.int64
    assert out["segment_id"].tolist() == ["a", "b"]
    assert out["speed_mean"].tolist() == [1.5, 2.5]


def test_feature_version_is_v5_so_old_caches_invalidate() -> None:
    """The v4 -> v5 bump is what rebuilds a float64 cache on the next run."""
    assert RepresentationConfig().feature_version == "representation_v5"


def test_legacy_cache_detector_flags_float64_and_ignores_float32(tmp_path) -> None:
    repr_dir = tmp_path / "derived" / "representations"
    repr_dir.mkdir(parents=True)
    seg_path = repr_dir / "segment_features.parquet"

    rng = np.random.default_rng(2)
    n_rows = 50
    old = pd.DataFrame(
        {
            "segment_id": [f"seg_{i}" for i in range(n_rows)],
            **{f"f{i}_mean": rng.normal(size=n_rows) for i in range(10)},
            **{
                f"r3d_{i:03d}": rng.normal(size=n_rows).astype(np.float32)
                for i in range(4)
            },
        }
    )
    old.to_parquet(seg_path, index=False)

    info = BehaviorRepresentationService.legacy_float64_segment_cache(tmp_path)
    assert info is not None
    assert info["n_rows"] == n_rows
    assert info["n_float64_cols"] == 10
    assert info["n_float32_cols"] == 4
    # Rebuilt is strictly smaller, and the reported peak is the all-float64 copy.
    assert info["rebuilt_gib"] < info["current_gib"] < info["peak_copy_gib"]

    BehaviorRepresentationService.downcast_segment_features(old).to_parquet(
        seg_path, index=False
    )
    assert BehaviorRepresentationService.legacy_float64_segment_cache(tmp_path) is None


def test_detector_ignores_float64_scoring_columns(tmp_path) -> None:
    # A v5 cache after a scored run: float32 features plus the float64
    # prediction/uncertainty columns written back by scoring. Not legacy.
    repr_dir = tmp_path / "derived" / "representations"
    repr_dir.mkdir(parents=True)
    rng = np.random.default_rng(4)
    n_rows = 30
    scored = pd.DataFrame(
        {
            "segment_id": [f"seg_{i}" for i in range(n_rows)],
            **{f"f{i}_mean": rng.normal(size=n_rows).astype(np.float32) for i in range(6)},
            **{c: rng.random(n_rows) for c in BehaviorRepresentationService.SCORING_COLUMNS},
        }
    )
    scored.to_parquet(repr_dir / "segment_features.parquet", index=False)
    assert BehaviorRepresentationService.legacy_float64_segment_cache(tmp_path) is None

    # A genuinely float64 feature column still trips it.
    scored["f0_mean"] = scored["f0_mean"].astype(np.float64)
    scored.to_parquet(repr_dir / "segment_features.parquet", index=False)
    info = BehaviorRepresentationService.legacy_float64_segment_cache(tmp_path)
    assert info is not None and info["n_float64_cols"] == 1


def test_detector_returns_none_without_a_cache(tmp_path) -> None:
    assert BehaviorRepresentationService.legacy_float64_segment_cache(tmp_path) is None


def test_merge_enriched_does_not_widen_the_segment_table() -> None:
    """Appending enriched rows must not consolidate the table into float64.

    ``reindex(..., fill_value=0.0)`` typed its pads from a Python float, so the
    padded frame arrived as float64 and the concat upcast every matching float32
    column with it — turning a routine append into a single contiguous float64
    block covering every numeric column.
    """
    from abel.ui.tabs.active_learning_tab import ActiveLearningTab

    df, cols = _group()
    segment_df = build_segment_df_fast(df, cols, "m1", "s1", 30, 15, True, True)

    # Enriched rows come from _segment_summary, i.e. Python floats -> float64,
    # and are missing any column that path cannot compute (e.g. r3d_*).
    enriched = segment_df.iloc[:3].astype(
        {c: np.float64 for c in segment_df.columns if segment_df[c].dtype == np.float32}
    )
    enriched["segment_id"] = ["enriched_0", "enriched_1", "enriched_2"]
    enriched = enriched.drop(columns=["speed_mean"])

    merged = ActiveLearningTab._merge_enriched(segment_df, enriched)

    widened = [c for c in merged.columns if merged[c].dtype == np.float64]
    assert widened == [], f"merge widened columns to float64: {widened[:5]}"
    assert len(merged) == len(segment_df) + 3
    # The absent column is still padded, at the target's width rather than float64.
    assert merged["speed_mean"].dtype == np.float32
    assert float(merged["speed_mean"].iloc[-1]) == 0.0
    # A single consolidated float64 block is exactly what blew up; assert the
    # float block that does exist is the narrow one.
    float_blocks = [b for b in merged._mgr.blocks if b.values.dtype.kind == "f"]
    assert all(b.values.dtype == np.float32 for b in float_blocks)


def test_merge_enriched_preserves_values_from_both_sides() -> None:
    from abel.ui.tabs.active_learning_tab import ActiveLearningTab

    df, cols = _group()
    segment_df = build_segment_df_fast(df, cols, "m1", "s1", 30, 15, True, True)
    enriched = segment_df.iloc[:2].copy()
    enriched["segment_id"] = ["enriched_0", "enriched_1"]
    enriched["speed_mean"] = np.array([4.25, -1.5], dtype=np.float32)

    merged = ActiveLearningTab._merge_enriched(segment_df, enriched)

    assert merged["segment_id"].tolist()[-2:] == ["enriched_0", "enriched_1"]
    assert merged["speed_mean"].tolist()[-2:] == [4.25, -1.5]
    # Original rows are untouched.
    pd.testing.assert_frame_equal(
        merged.iloc[: len(segment_df)].reset_index(drop=True), segment_df
    )
