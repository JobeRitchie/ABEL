from pathlib import Path

import numpy as np
import pandas as pd

from abel.services.pose_processing_service import PoseData, PoseProcessingService


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
