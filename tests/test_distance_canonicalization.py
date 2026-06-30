"""Pairwise-distance columns must collapse onto a canonical (sorted) name.

Mixed-order pose exports historically produced both ``dist_a_to_b`` and
``dist_b_to_a`` for the same symmetric distance.  Concatenating such sessions
left two half-populated columns per pair (complementary NaNs), each of which
could look "dead" downstream.  The representation builder now merges them onto
the canonical sorted name so the result is a single, fully-populated column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abel.services.behavior_representation_service import (
    BehaviorRepresentationService,
    RepresentationConfig,
)

_canon = BehaviorRepresentationService._canonicalize_distance_columns


def test_duplicate_orientations_merge_and_fill() -> None:
    # Two sessions: one names the pair nose->mid_back, the other mid_back->nose.
    # Each column is populated only for its own session (complementary NaNs).
    df = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2"],
            "dist_nose_to_mid_back": [1.0, 2.0, np.nan, np.nan],
            "dist_mid_back_to_nose": [np.nan, np.nan, 3.0, 4.0],
        }
    )
    out = _canon(df.copy())
    # Canonical sorted name is mid_back < nose -> dist_mid_back_to_nose.
    assert "dist_mid_back_to_nose" in out.columns
    assert "dist_nose_to_mid_back" not in out.columns
    # Fully populated after merge, values preserved per row.
    assert out["dist_mid_back_to_nose"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_norm_variant_canonicalized_independently() -> None:
    df = pd.DataFrame(
        {
            "dist_nose_to_mid_back_norm": [0.5, np.nan],
            "dist_mid_back_to_nose_norm": [np.nan, 0.9],
        }
    )
    out = _canon(df.copy())
    assert "dist_mid_back_to_nose_norm" in out.columns
    assert "dist_nose_to_mid_back_norm" not in out.columns
    assert out["dist_mid_back_to_nose_norm"].tolist() == [0.5, 0.9]


def test_canonical_present_takes_precedence_over_duplicate() -> None:
    # When both the canonical column and a duplicate carry a value on the same
    # row, the canonical value wins; the duplicate only fills gaps.
    df = pd.DataFrame(
        {
            "dist_mid_back_to_nose": [10.0, np.nan],
            "dist_nose_to_mid_back": [99.0, 7.0],
        }
    )
    out = _canon(df.copy())
    assert out["dist_mid_back_to_nose"].tolist() == [10.0, 7.0]
    assert "dist_nose_to_mid_back" not in out.columns


def test_non_distance_and_roi_columns_untouched() -> None:
    df = pd.DataFrame(
        {
            "nose_velocity": [1.0, 2.0],
            "dist_nose_to_target_dist": [3.0, 4.0],  # ROI/target distance, not a pair
            "body_centroid_to_roi_1_dist": [5.0, 6.0],
            "dist_centroid_to_centroid": [7.0, 8.0],  # symmetric, same part -> unchanged
        }
    )
    out = _canon(df.copy())
    assert list(out.columns) == list(df.columns)
    pd.testing.assert_frame_equal(out, df)


def test_already_canonical_unchanged() -> None:
    df = pd.DataFrame(
        {
            "dist_back_left_paw_to_nose": [1.0, 2.0],
            "dist_mid_back_to_nose": [3.0, 4.0],
        }
    )
    out = _canon(df.copy())
    # Both names are already sorted; nothing dropped or renamed.
    assert set(out.columns) == set(df.columns)
    pd.testing.assert_frame_equal(out, df)


def _make_mixed_order_sources(project_root: Path, n_frames: int = 120):
    """Two sessions whose pose files spell the same distance pair in opposite
    orders, to exercise canonicalization through the full build()."""
    rng = np.random.default_rng(0)
    pose_dir = project_root / "derived" / "pose_features"
    ctx_dir = project_root / "derived" / "context_features"
    (pose_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (ctx_dir / "sessions").mkdir(parents=True, exist_ok=True)

    frames = np.arange(n_frames)
    specs = {"s1": "dist_nose_to_mid_back", "s2": "dist_mid_back_to_nose"}
    for sid, dist_col in specs.items():
        pose = pd.DataFrame(
            {
                "frame": frames,
                "animal_id": "a1",
                "session_id": sid,
                "nose_velocity": rng.normal(size=n_frames),
                dist_col: rng.normal(size=n_frames).cumsum(),
            }
        )
        ctx = pd.DataFrame(
            {"frame": frames, "animal_id": "a1", "session_id": sid,
             "flow_mag": rng.normal(size=n_frames)}
        )
        pose.to_parquet(pose_dir / "sessions" / f"{sid}.parquet", index=False)
        ctx.to_parquet(ctx_dir / "sessions" / f"{sid}.parquet", index=False)
    return pose_dir / "frame_pose.parquet", ctx_dir / "frame_context.parquet"


def test_build_emits_single_canonical_distance_and_alive_delta(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    pose_path, ctx_path = _make_mixed_order_sources(project)
    # Enable clip-wise posture deltas so distance pairs get _delta/_trend columns
    # (mirrors the TMT_NewCam project configuration).
    (project / "config").mkdir(parents=True, exist_ok=True)
    (project / "config" / "experiment.yaml").write_text(
        "behavior_model:\n  invariant_features:\n    enable_clipwise_deltas: true\n",
        encoding="utf-8",
    )

    svc = BehaviorRepresentationService()
    frame, seg = svc.build(
        project_root=project,
        frame_pose_path=pose_path,
        frame_context_path=ctx_path,
        config=RepresentationConfig(window_size_frames=15, window_stride_frames=8),
    )
    # Only the canonical spelling survives, and it is fully populated (no NaNs
    # from the complementary-session split).
    assert "dist_mid_back_to_nose" in frame.columns
    assert "dist_nose_to_mid_back" not in frame.columns
    assert frame["dist_mid_back_to_nose"].notna().all()

    # Its clip-wise delta is a real, varying feature (not a dead all-zero column).
    delta = seg["dist_mid_back_to_nose_delta"].dropna()
    assert len(delta) > 0
    assert float(delta.std()) > 1e-9
