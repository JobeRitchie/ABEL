"""Rendering helpers for the Validation tab's Behavior Grid montage.

The Behavior Grid stitches 25 short clips — each a strong positive bout of one
behavior, drawn from across sessions — into a single 5×5 looping video with pose
keypoints overlaid.  This module holds the pure, OpenCV-only rendering primitives
so the heavy/IO-bound work is isolated from :class:`ValidationService` and can be
unit-tested without a Qt or project context.

Three primitives:

* :func:`draw_keypoints` — overlay pose dots on a full frame (original coords),
  ported from the Review tab's player so the look matches.
* :func:`render_cell` — decode one bout window, draw keypoints, crop a square
  region around the (smoothed) pose centroid, and write a square cell clip.
* :func:`stitch_grid` — tile per-cell clips into one square grid video, looping
  shorter cells so every tile always shows motion.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger("abel")

# Matches the Review tab player palette (BGR is applied via cv2 below, but these
# are stored RGB-style tuples exactly as in review_tab._KEYPOINT_PALETTE).
_KEYPOINT_PALETTE = [
    (60, 180, 255),
    (80, 220, 80),
    (255, 200, 70),
    (200, 120, 255),
    (255, 110, 110),
    (220, 220, 220),
    (255, 170, 0),
    (180, 255, 255),
]


def _scaled_crop_margin(
    base_margin_px: int,
    frame_width: int,
    frame_height: int,
    crop_area_scale: float = 1.25,
) -> int:
    """Scale the crop margin with source resolution (mirrors ClipExtractionService).

    Kept local so this module has no dependency on the preprocessing service; the
    formula is identical to ``ClipExtractionService._scaled_crop_margin`` so the
    grid crops match the Clip Review crops.
    """
    base = max(8, int(base_margin_px))
    short_edge = max(1, min(int(frame_width), int(frame_height)))
    reference_short_edge = 480
    scale = max(1.0, float(short_edge) / float(reference_short_edge))
    area_scale = max(0.1, float(crop_area_scale))
    linear_scale = float(math.sqrt(area_scale))
    scaled = int(round(base * scale * linear_scale))
    max_margin = max(8, short_edge // 2 - 1)
    return max(8, min(scaled, max_margin))


def draw_keypoints(
    bgr: np.ndarray,
    x_row: np.ndarray,
    y_row: np.ndarray,
    conf_row: np.ndarray,
    conf_thresh: float = 0.2,
) -> np.ndarray:
    """Return a copy of *bgr* with pose dots drawn for one frame's keypoints.

    *x_row*, *y_row*, *conf_row* are 1-D arrays of per-part coordinates (in the
    original video's pixel space) and confidences.  Low-confidence or non-finite
    parts are skipped.  Dot size scales with frame height to stay legible after
    the cell is downscaled.
    """
    import cv2  # noqa: PLC0415

    n_parts = min(len(x_row), len(y_row), len(conf_row))
    if n_parts <= 0:
        return bgr
    h = bgr.shape[0]
    radius = max(2, int(round(h / 250)))
    thickness = max(1, radius // 2)
    out = bgr.copy()
    for p in range(n_parts):
        conf = conf_row[p]
        if not np.isfinite(conf) or conf < conf_thresh:
            continue
        x = x_row[p]
        y = y_row[p]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        color = _KEYPOINT_PALETTE[p % len(_KEYPOINT_PALETTE)]
        center = (int(round(x)), int(round(y)))
        cv2.circle(out, center, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, center, radius, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
    return out


def render_cell(
    video_path: Path,
    pose_x: np.ndarray | None,
    pose_y: np.ndarray | None,
    pose_conf: np.ndarray | None,
    centroid_x: np.ndarray | None,
    centroid_y: np.ndarray | None,
    start_frame: int,
    end_frame: int,
    *,
    crop_margin_px: int,
    crop_area_scale: float,
    cell_px: int,
    show_keypoints: bool,
    out_path: Path,
    out_fps: float = 30.0,
    crop_scale: float = 1.0,
) -> bool:
    """Render one bout window to a square ``cell_px`` clip; return True on success.

    Frames ``[start_frame, end_frame]`` are decoded from *video_path*.  When pose
    centroids are available the crop follows the subject (smoothed, jump-limited —
    the same dynamic-centering used by Clip Review); otherwise a fixed centre crop
    is used.  Keypoints, when enabled, are drawn on the full frame *before*
    cropping so they stay pixel-aligned.

    *crop_scale* is a user-facing linear multiplier on the crop half-width: values
    above 1 show more surroundings (zoom out), below 1 tighten onto the subject.
    """
    import cv2  # noqa: PLC0415

    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame, int(end_frame))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Behavior grid: cannot open video %s", video_path)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out_path), fourcc, float(out_fps), (cell_px, cell_px))
    if not writer.isOpened():
        cap.release()
        logger.warning("Behavior grid: cannot create cell file %s", out_path)
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, probe = cap.read()
    if not ret:
        cap.release()
        writer.release()
        return False

    vid_h, vid_w = probe.shape[:2]
    m = _scaled_crop_margin(crop_margin_px, vid_w, vid_h, crop_area_scale=crop_area_scale)
    scale = max(0.1, float(crop_scale))
    if scale != 1.0:
        max_margin = max(8, min(vid_w, vid_h) // 2 - 1)
        m = max(8, min(int(round(m * scale)), max_margin))

    use_dynamic = (
        centroid_x is not None
        and centroid_y is not None
        and len(centroid_x) > 0
        and len(centroid_y) > 0
    )
    if use_dynamic:
        n_cent = min(len(centroid_x), len(centroid_y))
        seed = min(max(0, start_frame), n_cent - 1)
        last_cx = float(centroid_x[seed])
        last_cy = float(centroid_y[seed])
    else:
        last_cx, last_cy = vid_w / 2.0, vid_h / 2.0
    if not np.isfinite(last_cx) or not np.isfinite(last_cy):
        last_cx, last_cy = vid_w / 2.0, vid_h / 2.0

    smoothing_alpha = 0.2
    max_center_jump = max(4.0, float(m) * 0.35)

    have_kp = (
        show_keypoints
        and pose_x is not None
        and pose_y is not None
        and pose_conf is not None
    )

    n_to_write = end_frame - start_frame + 1
    frame = probe
    try:
        for idx in range(n_to_write):
            if idx > 0:
                ret, frame = cap.read()
                if not ret:
                    break
            fidx = start_frame + idx

            if have_kp and 0 <= fidx < pose_x.shape[0]:
                frame = draw_keypoints(
                    frame, pose_x[fidx], pose_y[fidx], pose_conf[fidx]
                )

            cx, cy = last_cx, last_cy
            if use_dynamic:
                ci = min(max(0, fidx), n_cent - 1)
                dyn_cx = float(centroid_x[ci])
                dyn_cy = float(centroid_y[ci])
                if np.isfinite(dyn_cx) and np.isfinite(dyn_cy):
                    dyn_cx = smoothing_alpha * dyn_cx + (1.0 - smoothing_alpha) * last_cx
                    dyn_cy = smoothing_alpha * dyn_cy + (1.0 - smoothing_alpha) * last_cy
                    dx, dy = dyn_cx - last_cx, dyn_cy - last_cy
                    jump = float(np.hypot(dx, dy))
                    if jump > max_center_jump:
                        s = max_center_jump / max(jump, 1e-9)
                        dyn_cx, dyn_cy = last_cx + dx * s, last_cy + dy * s
                    last_cx, last_cy = dyn_cx, dyn_cy
                    cx, cy = dyn_cx, dyn_cy

            x1 = max(0, int(cx - m))
            y1 = max(0, int(cy - m))
            x2 = min(vid_w, int(cx + m))
            y2 = min(vid_h, int(cy + m))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame
            crop = cv2.resize(crop, (cell_px, cell_px))
            writer.write(crop)
    finally:
        cap.release()
        writer.release()
    return True


def stitch_grid(
    cell_paths: list[Path | None],
    grid_px: int,
    out_path: Path,
    *,
    rows: int = 5,
    cols: int = 5,
    out_fps: float = 30.0,
) -> Path:
    """Tile per-cell clips into one square grid video, looping shorter cells.

    *cell_paths* is ``rows*cols`` long (``None`` or a missing file → black tile).
    The grid runs for as long as the longest cell; every other cell loops on a
    modulo of its own length so no tile freezes.  Returns *out_path*.
    """
    import cv2  # noqa: PLC0415

    cell_px = grid_px // cols
    grid_w = cell_px * cols
    grid_h = cell_px * rows

    caps: list[object | None] = []
    lengths: list[int] = []
    for p in cell_paths:
        cap = None
        n = 0
        if p is not None and Path(p).exists():
            c = cv2.VideoCapture(str(p))
            if c.isOpened():
                cap = c
                n = max(0, int(c.get(cv2.CAP_PROP_FRAME_COUNT)))
        caps.append(cap)
        lengths.append(n)

    total = rows * cols
    # Pad/truncate to exactly the grid size.
    caps = (caps + [None] * total)[:total]
    lengths = (lengths + [0] * total)[:total]

    max_len = max([n for n in lengths if n > 0], default=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out_path), fourcc, float(out_fps), (grid_w, grid_h))
    if not writer.isOpened():
        for c in caps:
            if c is not None:
                c.release()
        raise RuntimeError(f"Cannot create grid output {out_path}")

    black = np.zeros((cell_px, cell_px, 3), dtype=np.uint8)
    # Cache each cell's frames as we first read them so loops don't re-seek.
    cell_frames: list[list[np.ndarray]] = [[] for _ in range(total)]

    def _cell_frame(i: int, t: int) -> np.ndarray:
        n = lengths[i]
        cap = caps[i]
        if cap is None or n <= 0:
            return black
        want = t % n
        cached = cell_frames[i]
        while len(cached) <= want:
            ret, fr = cap.read()  # type: ignore[union-attr]
            if not ret:
                if cached:
                    break
                return black
            if fr.shape[0] != cell_px or fr.shape[1] != cell_px:
                fr = cv2.resize(fr, (cell_px, cell_px))
            cached.append(fr)
        return cached[min(want, len(cached) - 1)] if cached else black

    try:
        for t in range(max_len):
            canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            for i in range(total):
                r, c = divmod(i, cols)
                tile = _cell_frame(i, t)
                canvas[r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px] = tile
            writer.write(canvas)
    finally:
        writer.release()
        for c in caps:
            if c is not None:
                c.release()
    return out_path
