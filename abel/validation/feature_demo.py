"""Export a feature-demonstration clip (raw vs smoothed DLC + trace strip) to MP4.

Reuses the main GUI's preview renderer wholesale — :func:`render_preview_frames`
and :func:`_render_graph_strip` from
:mod:`abel.ui.smoothing_preview_dialog` — so the exported video matches what the
"Preview Video Settings" dialog shows, but written to disk for slides / figures
that demonstrate what a feature actually responds to.

The output frame is the raw|smoothed composite (from the renderer) stacked on top
of the per-frame trace strip with a playhead, written as an ``.mp4``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np

from abel.models.schemas import PoseSmoothingSettings


def export_feature_demo(
    video_path: Path,
    pose_path: Path,
    smoothing: PoseSmoothingSettings,
    out_path: Path,
    *,
    local_radius_px: int = 36,
    mog2_var_threshold: int = 16,
    duration_sec: float = 10.0,
    visible_traces: set[str] | None = None,
    start_frame: int | None = None,
    target_height: int = 340,
    cancel_flag: list[bool] | None = None,
    progress_cb: Callable[[str, float], None] | None = None,
) -> Path:
    """Render a demo clip to ``out_path`` (``.mp4``) and return the path.

    Raises ``RuntimeError`` if no frames render or the video writer can't open.
    """
    import cv2  # noqa: PLC0415

    from abel.services.pose_processing_service import PoseProcessingService  # noqa: PLC0415
    from abel.ui.smoothing_preview_dialog import (  # noqa: PLC0415
        _DEFAULT_TRACE_KEYS,
        _GRAPH_HEIGHT,
        _render_graph_strip,
        render_preview_frames,
    )

    cancel_flag = cancel_flag if cancel_flag is not None else [False]
    visible = set(visible_traces) if visible_traces is not None else set(_DEFAULT_TRACE_KEYS)

    def _emit(msg: str, frac: float) -> None:
        if progress_cb is not None:
            progress_cb(msg, frac)

    _emit("Loading + cleaning pose…", 0.02)
    pose_svc = PoseProcessingService()
    raw_pose = pose_svc.load(Path(pose_path))
    smooth_pose = pose_svc.clean_pose(
        raw_pose,
        likelihood_threshold=smoothing.likelihood_threshold,
        interpolate=smoothing.interpolate_dropouts,
        interpolate_max_gap=smoothing.interpolate_max_gap,
        smoothing_window=smoothing.smoothing_window,
    )

    # Probe fps + total frames so the clip window is valid.
    fps_source = 30.0
    total = raw_pose.n_frames
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = min(
            raw_pose.n_frames,
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or raw_pose.n_frames),
        )
    cap.release()

    n_frames = min(int(fps_source * duration_sec), total)
    if n_frames <= 0:
        raise RuntimeError("Clip duration resolves to zero frames.")
    if start_frame is None:
        max_start = max(0, total - n_frames)
        start_frame = random.randint(0, max_start) if max_start > 0 else 0

    _emit("Rendering frames…", 0.08)
    result = render_preview_frames(
        video_path=Path(video_path),
        raw_pose=raw_pose,
        smooth_pose=smooth_pose,
        smoothing=smoothing,
        start_frame=int(start_frame),
        n_frames=int(n_frames),
        target_height=int(target_height),
        cancel_flag=cancel_flag,
        local_radius_px=int(local_radius_px),
        fps=float(fps_source),
        mog2_var_threshold=int(mog2_var_threshold),
    )
    if not result.frames:
        raise RuntimeError(
            "No frames rendered — check the video/pose files are accessible and OpenCV is installed."
        )

    h0, w0 = result.frames[0].shape[:2]
    out_h = h0 + _GRAPH_HEIGHT
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps_source), (w0, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {out_path}.")

    n = len(result.frames)
    try:
        for i, frame in enumerate(result.frames):
            if cancel_flag[0]:
                break
            strip = _render_graph_strip(
                result.traces, visible, width=w0, height=_GRAPH_HEIGHT, playhead=i
            )
            if strip.shape[1] != w0:
                strip = cv2.resize(strip, (w0, _GRAPH_HEIGHT))
            composite = np.concatenate([frame, strip], axis=0)
            writer.write(composite)
            if i % 5 == 0:
                _emit(f"Writing frame {i + 1}/{n}…", 0.1 + 0.85 * (i + 1) / n)
    finally:
        writer.release()

    _emit(f"Saved {out_path.name}", 1.0)
    return out_path
