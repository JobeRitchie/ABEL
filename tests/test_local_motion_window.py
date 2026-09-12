"""Local surface-motion windows keep a constant size.

Regression: the MOG2 background model ran on a crop that shrank whenever the
animal came within one Local radius of the frame edge (or collapsed to 1x1 on a
missing keypoint).  OpenCV's MOG2 re-initialises on any input-size change and
the next frame then reads as 100% foreground, so wall-hugging animals got
spurious local_surface_energy / nose_surface_energy spikes of ~1.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from abel.services.context_feature_service import ContextFeatureConfig, ContextFeatureService

cv2 = pytest.importorskip("cv2")


def test_local_crop_is_always_2r_square() -> None:
    frame = np.arange(40 * 50, dtype=np.uint8).reshape(40, 50)
    for x, y in [(25, 20), (1, 1), (49, 39), (-30, 5), (80, 90), (np.nan, 3)]:
        crop = ContextFeatureService._local_crop(frame, x, y, 8)
        assert crop.shape == (16, 16), (x, y)
    # Fully inside: an exact slice of the frame, not a resample.
    assert np.array_equal(ContextFeatureService._local_crop(frame, 25, 20, 8), frame[12:28, 17:33])


def test_hold_last_position_fills_dropouts() -> None:
    x = np.array([np.nan, 5.0, np.nan, np.nan, 9.0, np.nan])
    y = np.array([np.nan, 1.0, np.nan, np.nan, 2.0, np.nan])
    hx, hy = ContextFeatureService._hold_last_position(x, y)
    assert hx.tolist() == [5.0, 5.0, 5.0, 5.0, 9.0, 9.0]
    assert hy.tolist() == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    zx, _ = ContextFeatureService._hold_last_position(np.full(3, np.nan), np.full(3, np.nan))
    assert zx.tolist() == [0.0, 0.0, 0.0]


def test_no_foreground_spikes_at_frame_edge(tmp_path) -> None:
    # A featureless arena: whatever the window position, its content is the
    # same, so the true foreground fraction is 0 on every frame.
    n, w, h, r = 160, 64, 64, 12
    path = tmp_path / "edge.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    assert writer.isOpened()
    for _ in range(n):
        writer.write(np.full((h, w, 3), 128, dtype=np.uint8))
    writer.release()

    t = np.arange(n)
    # Centroid swings across the left edge (window clipped on some frames,
    # not others) and drops out now and then.
    body_x = 2.0 + 20.0 * (np.sin(t / 5.0) > 0)
    body_y = 32.0 + 10.0 * np.sin(t / 7.0)
    body_x[::13] = np.nan
    nose_x, nose_y = body_x + 3.0, body_y.copy()
    zeros = np.zeros(n)

    out = ContextFeatureService._process_video_chunk(
        video_path=path, frame_start=60, frame_end=n,
        body_x=body_x, body_y=body_y,
        paw_l_x=zeros, paw_l_y=zeros, paw_r_x=zeros, paw_r_y=zeros,
        nose_x=nose_x, nose_y=nose_y,
        target_roi={}, has_target=False, local_radius=r,
        config=ContextFeatureConfig(downsample_factor=1, prefer_gpu=False),
    )
    for key in ("local_surface_energy", "nose_surface_energy"):
        energy = np.asarray(out[key], dtype=float)
        assert len(energy) == n - 60
        assert energy.max() < 0.05, (key, energy.max())


def test_preview_nose_traces_equal_extracted_features(tmp_path) -> None:
    """The Smoothing Preview's nose traces are the extracted features, value for value."""
    import pandas as pd

    pytest.importorskip("PySide6")
    from abel.models.schemas import PoseSmoothingSettings
    from abel.services.pose_processing_service import PoseData
    from abel.ui.smoothing_preview_dialog import render_preview_frames

    n, w, h, r = 120, 96, 80, 10
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 255, (h, w), dtype=np.uint8)
    t = np.arange(n)
    nose_x = 48 + 40 * np.sin(t / 9.0)          # runs off both side edges
    nose_y = 40 + 30 * np.cos(t / 11.0)
    path = tmp_path / "blob.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for i in range(n):
        frame = bg.copy()
        cv2.circle(frame, (int(nose_x[i]), int(nose_y[i])), 6, 255, -1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()

    parts = ["nose", "tail_base"]
    xs = pd.DataFrame({"nose": nose_x, "tail_base": nose_x - 5})
    ys = pd.DataFrame({"nose": nose_y, "tail_base": nose_y})
    lk = pd.DataFrame({"nose": np.ones(n), "tail_base": np.ones(n)})
    pose = PoseData(parts, xs, ys, lk, xs.mean(axis=1).to_numpy(), ys.mean(axis=1).to_numpy(), n)

    start, count = 60, 40
    preview = render_preview_frames(
        video_path=path, raw_pose=pose, smooth_pose=pose,
        smoothing=PoseSmoothingSettings(), start_frame=start, n_frames=count,
        target_height=80, cancel_flag=[False], local_radius_px=r,
    )
    zeros = np.zeros(n)
    extracted = ContextFeatureService._process_video_chunk(
        video_path=path, frame_start=start, frame_end=start + count,
        body_x=pose.centroid_x, body_y=pose.centroid_y,
        paw_l_x=zeros, paw_l_y=zeros, paw_r_x=zeros, paw_r_y=zeros,
        nose_x=nose_x, nose_y=nose_y,
        target_roi={}, has_target=False, local_radius=r,
        config=ContextFeatureConfig(downsample_factor=0, prefer_gpu=False),
    )
    for trace, feature in (
        ("nose_bg_energy", "nose_surface_energy"),
        ("nose_px_change", "nose_surface_change"),
        ("nose_px_variance", "nose_surface_var"),
    ):
        got = np.asarray(preview.traces[trace], dtype=float)
        want = np.asarray(extracted[feature], dtype=float)
        assert got.std() > 0  # the blob actually moves through the window
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, err_msg=trace)
