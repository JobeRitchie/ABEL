"""A video that ends before its pose file must not break context extraction.

On the CPU optical-flow path a chunk used to return flow columns shorter than
its other columns when the video ran out a few frames into the chunk (one flow
sample, nothing to interpolate between), and whenever the flow stride was 1
(the first frame of a video has no previous frame).  The frame table then
failed with "All arrays must be of the same length".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from abel.services.context_feature_service import ContextFeatureConfig, ContextFeatureService

W, H, N_VIDEO, N_POSE = 96, 72, 60, 120


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("short_vid") / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    rng = np.random.default_rng(0)
    for _ in range(N_VIDEO):
        writer.write(rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8))
    writer.release()
    return path


def _chunk(video: Path, start: int, end: int, stride: int) -> dict:
    t = np.arange(N_POSE, dtype=float)
    x, y = 30.0 + (t % 40), 25.0 + (t % 20)
    return ContextFeatureService._process_video_chunk(
        video_path=video, frame_start=start, frame_end=end,
        body_x=x, body_y=y,
        paw_l_x=x - 4, paw_l_y=y + 3, paw_r_x=x + 4, paw_r_y=y + 3,
        nose_x=x + 8, nose_y=y - 2,
        target_roi={}, has_target=False, local_radius=16,
        config=ContextFeatureConfig(flow_temporal_stride=stride, downsample_factor=1),
        extra_rois=[{"shape": "rect", "x": 10, "y": 10, "w": 20, "h": 20}],
    )


@pytest.mark.parametrize("stride", [1, 5, 10])
@pytest.mark.parametrize("start", [0, 30, 45, 50, 55, 57, 58, 59, 60, 90])
def test_every_column_has_one_value_per_frame_read(video, start, stride):
    out = _chunk(video, start, N_POSE, stride)
    expected = max(0, N_VIDEO - start)
    assert {key: len(vals) for key, vals in out.items()} == dict.fromkeys(out, expected)


def test_full_length_chunks_are_unchanged_by_a_short_tail(video):
    """The frames a short video does have give the same numbers either way."""
    whole = _chunk(video, 30, N_VIDEO, stride=5)
    overrun = _chunk(video, 30, N_POSE, stride=5)
    for key in whole:
        np.testing.assert_allclose(whole[key], overrun[key], equal_nan=True)
