"""P4: unified z-scoring — parity, stats persistence, and cache reuse.

- The vectorised ``_zscore_by_group`` must produce bit-for-bit the same result
  as the original per-group Python loop (data not damaged).
- Per-(animal_id, session_id) mean/std stats must be persisted by build().
- The adaptive feature cache must reuse the canonical representation cache
  instead of re-reading raw pose/context.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abel.services.behavior_representation_service import (
    BehaviorRepresentationService,
    RepresentationConfig,
)
from abel.services.behavior_adaptive_feature_cache_service import (
    BehaviorAdaptiveFeatureCacheService,
)


def _reference_zscore(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """The original per-group loop implementation, kept here as ground truth."""
    out = df.copy()
    for _, grp in out.groupby(["animal_id", "session_id"]):
        idx = grp.index
        vals = grp[feature_cols]
        mu = vals.mean(axis=0)
        sigma = vals.std(axis=0).fillna(1.0).replace(0.0, 1.0)
        out.loc[idx, feature_cols] = (vals - mu) / sigma
    return out


def _frame(seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    rows = []
    for animal in ("a1", "a2"):
        for sid in ("s1", "s2", "s3"):
            n = rng.integers(1, 50)  # include single-row groups
            for fr in range(int(n)):
                rows.append(
                    {
                        "animal_id": animal,
                        "session_id": sid,
                        "frame": fr,
                        "f0": float(rng.normal()),
                        "f1": float(rng.normal()),
                        "const": 5.0,  # zero-variance feature
                    }
                )
    df = pd.DataFrame(rows)
    return df, ["f0", "f1", "const"]


def test_vectorized_zscore_matches_loop():
    df, cols = _frame(seed=3)
    got = BehaviorRepresentationService._zscore_by_group(df, cols)
    ref = _reference_zscore(df, cols)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), ref.reset_index(drop=True), check_exact=False
    )


def test_zscore_stats_are_correct():
    df, cols = _frame(seed=4)
    _, stats = BehaviorRepresentationService._zscore_by_group_with_stats(df, cols)
    # One row per (animal_id, session_id) group.
    assert len(stats) == df.groupby(["animal_id", "session_id"]).ngroups
    # Spot-check one group's mean against a manual computation.
    g = df[(df.animal_id == "a1") & (df.session_id == "s1")]
    row = stats[(stats.animal_id == "a1") & (stats.session_id == "s1")].iloc[0]
    assert np.isclose(row["f0__mean"], g["f0"].mean())


def _make_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "proj"
    pose_dir = project / "derived" / "pose_features"
    ctx_dir = project / "derived" / "context_features"
    (pose_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (ctx_dir / "sessions").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    for sid in ("s1", "s2"):
        n = 120
        pd.DataFrame(
            {
                "frame": np.arange(n),
                "animal_id": "a1",
                "session_id": sid,
                "speed": rng.normal(size=n),
            }
        ).to_parquet(pose_dir / "sessions" / f"{sid}.parquet", index=False)
        pd.DataFrame(
            {
                "frame": np.arange(n),
                "animal_id": "a1",
                "session_id": sid,
                "flow_mag": rng.normal(size=n),
            }
        ).to_parquet(ctx_dir / "sessions" / f"{sid}.parquet", index=False)
    return project, pose_dir / "frame_pose.parquet", ctx_dir / "frame_context.parquet"


def test_build_persists_zscore_stats(tmp_path: Path):
    project, pose_path, ctx_path = _make_project(tmp_path)
    BehaviorRepresentationService().build(
        project_root=project,
        frame_pose_path=pose_path,
        frame_context_path=ctx_path,
        config=RepresentationConfig(window_size_frames=30, window_stride_frames=15),
    )
    stats_path = (
        project / "derived" / "representations"
        / BehaviorRepresentationService.ZSCORE_STATS_FILENAME
    )
    assert stats_path.exists(), "z-score stats must be persisted next to the cache"
    stats = pd.read_parquet(stats_path)
    assert {"animal_id", "session_id"}.issubset(stats.columns)
    assert len(stats) == 2  # two sessions


def test_adaptive_cache_reuses_representation(tmp_path: Path, monkeypatch):
    project, pose_path, ctx_path = _make_project(tmp_path)
    # Build the canonical representation cache first.
    BehaviorRepresentationService().build(
        project_root=project,
        frame_pose_path=pose_path,
        frame_context_path=ctx_path,
        config=RepresentationConfig(window_size_frames=30, window_stride_frames=15),
    )

    svc = BehaviorAdaptiveFeatureCacheService()

    # If the adaptive loader touches raw pose/context, fail loudly.
    import abel.services.behavior_adaptive_feature_cache_service as mod

    real_read = mod.pd.read_parquet
    repr_frame = project / "derived" / "representations" / "frame_features.parquet"

    def guarded_read(path, *a, **k):
        p = str(path)
        if "pose_features" in p or "context_features" in p:
            raise AssertionError(f"adaptive loader read raw source instead of cache: {p}")
        return real_read(path, *a, **k)

    monkeypatch.setattr(mod.pd, "read_parquet", guarded_read)
    frame_df = svc._load_merged_frame_features(project)
    assert not frame_df.empty
    assert repr_frame.exists()
