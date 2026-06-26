from pathlib import Path

import numpy as np
import pandas as pd

from abel.services.pose_processing_service import PoseData, PoseProcessingService


def _make_pose(order: list[str], n: int = 30) -> PoseData:
    rng = np.random.default_rng(0)
    x = pd.DataFrame({bp: rng.random(n) for bp in order})
    y = pd.DataFrame({bp: rng.random(n) for bp in order})
    lk = pd.DataFrame({bp: np.ones(n) for bp in order})
    return PoseData(
        x=x, y=y, likelihood=lk, body_parts=list(order),
        centroid_x=np.zeros(n), centroid_y=np.zeros(n), n_frames=n,
    )


def test_pairwise_distance_names_are_keypoint_order_independent() -> None:
    """Two projects with the same keypoints in a different DLC column order must
    produce identical dist_* feature columns (Direct Use cross-project reuse)."""
    from abel.models.schemas import InvariantFeatureConfig
    cfg = InvariantFeatureConfig(
        enable_egocentric_kinematics=False,
        enable_body_length_normalization=True,
        enable_relative_geometry=True,
        enable_head_direction=False,
        enable_joint_angles=False,
        enable_spine_curvature=False,
        enable_clipwise_deltas=False,
    )
    svc = PoseProcessingService()
    order_a = ["nose", "left_ear", "right_ear", "center_body", "tail_base"]
    order_b = ["center_body", "tail_base", "right_ear", "nose", "left_ear"]
    fa = svc.compute_frame_pose_features(_make_pose(order_a), 30.0, "a", "s", "v", invariant_config=cfg)
    fb = svc.compute_frame_pose_features(_make_pose(order_b), 30.0, "a", "s", "v", invariant_config=cfg)
    dist_a = {c for c in fa.columns if c.startswith("dist_")}
    dist_b = {c for c in fb.columns if c.startswith("dist_")}
    assert dist_a == dist_b
    # Names are canonically ordered (sorted pair).
    assert "dist_left_ear_to_nose" in dist_a
    assert "dist_nose_to_left_ear" not in dist_a


def test_compute_frame_pose_features_contains_expected_columns() -> None:
    n = 40
    t = np.arange(n, dtype=float)

    x = pd.DataFrame(
        {
            "nose": 10.0 + 0.5 * t,
            "paw_L": 5.0 + np.sin(t / 3.0),
            "paw_R": 6.0 + np.sin(t / 3.0 + 0.2),
            "tailbase": 3.0 + 0.2 * t,
        }
    )
    y = pd.DataFrame(
        {
            "nose": 8.0 + 0.3 * t,
            "paw_L": 4.0 + np.cos(t / 3.0),
            "paw_R": 4.5 + np.cos(t / 3.0 + 0.2),
            "tailbase": 2.0 + 0.1 * t,
        }
    )
    l = pd.DataFrame(1.0, index=range(n), columns=x.columns)

    pose = PoseData(
        body_parts=list(x.columns),
        x=x,
        y=y,
        likelihood=l,
        centroid_x=np.asarray(x.mean(axis=1), dtype=float),
        centroid_y=np.asarray(y.mean(axis=1), dtype=float),
        n_frames=n,
    )

    svc = PoseProcessingService()
    df = svc.compute_frame_pose_features(
        pose=pose,
        fps=30.0,
        animal_id="rat01",
        session_id="s1",
        video_id="v1",
    )

    assert len(df) == n
    assert "forepaw_speed" in df.columns
    assert "forepaw_movement_frequency" in df.columns
    assert "forepaw_autocorr_peak" in df.columns
    assert float(df["forepaw_speed"].mean()) > 0.0
