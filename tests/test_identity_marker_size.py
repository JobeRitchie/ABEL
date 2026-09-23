"""Identity-dot size setting: scales the dots, and the preview matches extraction."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from abel.models.schemas import CandidateWindow, PreprocessingPreset  # noqa: E402
from abel.services.preprocessing_service import (  # noqa: E402
    ClipExtractionConfig,
    ClipExtractionService,
)


def _video(path, n=40):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for _ in range(n):
        w.write(np.full((480, 640, 3), 128, np.uint8))
    w.release()


def _overlays(n=40):
    return [
        {"name": "A", "color": (0, 0, 255), "cx": np.full(n, 200.0), "cy": np.full(n, 240.0)},
        {"name": "B", "color": (255, 0, 0), "cx": np.full(n, 440.0), "cy": np.full(n, 240.0)},
    ]


def _preset(scale):
    return PreprocessingPreset(
        preset_id="p", name="p", crop_margin_px=100,
        resize_width=224, resize_height=224, identity_marker_scale=scale,
    )


def _blue_pixels(frame):
    # Animal B's dot only; the legend swatch is drawn in both colors, so it is constant.
    return int(((frame[..., 0] > 200) & (frame[..., 1] < 80) & (frame[..., 2] < 80)).sum())


def test_radius_scales_and_has_floor():
    assert ClipExtractionService.identity_marker_radius(224, 1.0) == 3.0
    assert ClipExtractionService.identity_marker_radius(224, 0.5) == 1.5
    assert ClipExtractionService.identity_marker_radius(700, 2.0) == 20.0
    assert ClipExtractionService.identity_marker_radius(224, 0.01) == 0.75
    assert ClipExtractionService.identity_marker_radius(224, float("nan")) == 3.0


def test_smaller_scale_draws_smaller_dots(tmp_path):
    vid = tmp_path / "v.mp4"
    _video(vid)
    svc = ClipExtractionService()
    sizes = [
        _blue_pixels(svc.render_preview_frame(vid, _preset(s), 5, 30, individual_overlays=_overlays()))
        for s in (0.5, 1.0, 2.0)
    ]
    assert sizes[0] < sizes[1] < sizes[2]


def test_preview_matches_extracted_clip(tmp_path):
    vid = tmp_path / "v.mp4"
    _video(vid)
    svc = ClipExtractionService()
    preset = _preset(0.5)
    cfg = ClipExtractionConfig(
        video_path=vid, session_id="s", preset=preset,
        output_dir=tmp_path, individual_overlays=_overlays(),
    )
    res = svc.extract_selected_clips(
        [CandidateWindow(window_id="w1", session_id="s", start_frame=5, end_frame=30)], cfg,
    )
    cap = cv2.VideoCapture(res.clips[0].processed_clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ok, extracted = cap.read()
    cap.release()
    assert ok
    preview = svc.render_preview_frame(vid, preset, 5, 30, individual_overlays=_overlays(), frame_offset=10)
    assert preview.shape == extracted.shape
    # Only mp4 compression separates them.
    assert float(np.abs(preview.astype(int) - extracted.astype(int)).mean()) < 6.0
