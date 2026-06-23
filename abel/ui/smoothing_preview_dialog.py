"""Side-by-side raw vs smoothed DLC tracking preview dialog.

Renders a random 10-second clip from a selected session with raw DLC
body-part dots on the left pane and smoothed dots on the right, allowing
the user to tune smoothing parameters and immediately see the effect.
A graph strip below the video shows user-selected per-frame traces
(context features, kinematics, global motion).  An inset on the smoothed
pane shows the background-subtracted zone around the nose.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, QThreadPool, QTimer, Slot
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from abel.models.schemas import ImportManifest, PoseSmoothingSettings
from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseData, PoseProcessingService
from abel.workers.task_worker import TaskWorker

logger = logging.getLogger("abel")

_PREVIEW_SEC = 10
_DISPLAY_HEIGHT = 340   # pixels tall per pane
_DIVIDER_W = 4          # separator between the two panes
_TRAIL_FRAMES = 10      # centroid trail length
_GRAPH_HEIGHT = 100     # height of the graph strip


# ---------------------------------------------------------------------------
# Trace catalog — every plottable trace the preview can compute
# ---------------------------------------------------------------------------

@dataclass
class TraceDef:
    """Metadata for one time-series trace available in the graph strip."""
    key: str            # dict key in PreviewResult.traces
    label: str          # legend text (ASCII only — OpenCV can't render Unicode)
    color: tuple[int, int, int]  # BGR
    category: str       # grouping shown in the selector dialog
    default_on: bool = True


# Ordered list — defines default draw order and selector list order.
TRACE_CATALOG: list[TraceDef] = [
    # ── Video / context features ─────────────────────────────────────
    TraceDef("nose_bg_energy",     "Nose BG energy",    (255, 200, 60),   "Video context"),
    TraceDef("nose_px_change",     "Nose px change",    (220, 120, 255),  "Video context"),
    TraceDef("nose_px_variance",   "Nose px variance",  (180, 180, 60),   "Video context", default_on=False),
    # ── Global motion ────────────────────────────────────────────────
    TraceDef("raw_centroid_speed", "Raw motion",         (80, 140, 255),  "Global motion"),
    TraceDef("smooth_centroid_speed", "Smooth motion",   (80, 255, 160),  "Global motion"),
    # ── Centroid kinematics ──────────────────────────────────────────
    TraceDef("centroid_speed",     "Centroid speed",     (255, 200, 60),  "Kinematics", default_on=False),
    TraceDef("centroid_accel",     "Centroid accel",     (60, 220, 255),  "Kinematics", default_on=False),
    TraceDef("body_angle",         "Body angle (nose-tail)", (220, 120, 255), "Kinematics", default_on=False),
    # ── Per-bodypart kinematics ──────────────────────────────────────
    TraceDef("nose_speed",         "Nose speed",         (255, 170, 0),   "Body-part speed", default_on=False),
    TraceDef("nose_accel",         "Nose accel",         (255, 100, 100), "Body-part speed", default_on=False),
    TraceDef("forelimb_l_speed",   "Forelimb L speed",   (80, 220, 80),   "Body-part speed", default_on=False),
    TraceDef("forelimb_r_speed",   "Forelimb R speed",   (200, 120, 255), "Body-part speed", default_on=False),
]

_TRACE_BY_KEY: dict[str, TraceDef] = {t.key: t for t in TRACE_CATALOG}
_DEFAULT_TRACE_KEYS: set[str] = {t.key for t in TRACE_CATALOG if t.default_on}


@dataclass
class PreviewResult:
    """Container returned by `render_preview_frames`."""
    frames: list[np.ndarray]
    traces: dict[str, np.ndarray] = field(default_factory=dict)
    fps: float = 30.0


# ---------------------------------------------------------------------------
# Rendering helpers (run in worker thread — no Qt calls allowed here)
# ---------------------------------------------------------------------------

def _part_color(part_idx: int) -> tuple[int, int, int]:
    """Distinct, deterministic BGR colour palette."""
    palette = [
        (60, 180, 255),
        (80, 220, 80),
        (255, 200, 70),
        (200, 120, 255),
        (255, 110, 110),
        (220, 220, 220),
        (255, 170, 0),
        (180, 255, 255),
    ]
    return palette[part_idx % len(palette)]


def _draw_label(
    img: np.ndarray,
    text: str,
    text_color: tuple[int, int, int],
) -> None:
    """Draw a small pill label in the top-left corner of *img* in-place."""
    import cv2  # noqa: PLC0415

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = 5
    cv2.rectangle(img, (0, 0), (tw + pad * 2, th + base + pad * 2), (15, 15, 15), -1)
    cv2.putText(img, text, (pad, th + pad), font, scale, text_color, thick, cv2.LINE_AA)


def _overlay_pose(
    canvas: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    lk_vals: np.ndarray,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    frame_idx: int,
    scale: float,
    trail_color: tuple[int, int, int],
    lk_threshold: float,
) -> None:
    """Draw body-part dots and centroid trail onto *canvas* in-place."""
    import cv2  # noqa: PLC0415

    h, w = canvas.shape[:2]
    n_parts = x_vals.shape[1]

    # Centroid trail (fades out toward the past)
    for back in range(1, _TRAIL_FRAMES + 1):
        prev = frame_idx - back
        if prev < 0:
            break
        alpha = 1.0 - back / (_TRAIL_FRAMES + 1)
        intensity = int(alpha * 180)
        cx = int(centroid_x[prev] * scale)
        cy = int(centroid_y[prev] * scale)
        if 0 <= cx < w and 0 <= cy < h:
            c = (
                int(trail_color[0] * alpha),
                int(trail_color[1] * alpha),
                int(trail_color[2] * alpha),
            )
            cv2.circle(canvas, (cx, cy), 2, c, -1)

    # Body-part dots
    for pi in range(n_parts):
        conf = float(lk_vals[frame_idx, pi]) if pi < lk_vals.shape[1] else 0.0
        bx = int(x_vals[frame_idx, pi] * scale)
        by = int(y_vals[frame_idx, pi] * scale)
        if not (0 <= bx < w and 0 <= by < h):
            continue
        color = _part_color(pi) if conf >= lk_threshold else (60, 60, 60)
        radius = 4 if conf >= lk_threshold else 2
        cv2.circle(canvas, (bx, by), radius, color, -1)
        if conf >= lk_threshold:
            cv2.circle(canvas, (bx, by), radius + 1, (0, 0, 0), 1)  # outline


def _crop_box_static(
    gray: np.ndarray, x: float, y: float, radius: int = 12,
) -> np.ndarray:
    """Extract a square crop from *gray* centred at (*x*, *y*)."""
    h, w = gray.shape[:2]
    x0 = max(0, int(x) - radius)
    x1 = min(w, int(x) + radius)
    y0 = max(0, int(y) - radius)
    y1 = min(h, int(y) + radius)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1), dtype=gray.dtype)
    return gray[y0:y1, x0:x1]


def _render_graph_strip(
    traces: dict[str, np.ndarray],
    visible_keys: set[str],
    width: int,
    height: int,
    playhead: int = -1,
) -> np.ndarray:
    """Render selected time-series traces as a BGR image.

    Only traces whose key is in *visible_keys* are drawn.
    A vertical playhead line marks the current frame.
    """
    import cv2  # noqa: PLC0415

    img = np.full((height, width, 3), 18, dtype=np.uint8)
    if width < 10:
        return img

    # Collect the traces to draw (preserving catalog order)
    to_draw: list[tuple[np.ndarray, tuple[int, int, int], str]] = []
    for tdef in TRACE_CATALOG:
        if tdef.key not in visible_keys:
            continue
        arr = traces.get(tdef.key)
        if arr is None or len(arr) < 2:
            continue
        to_draw.append((arr, tdef.color, tdef.label))

    if not to_draw:
        return img

    n = len(to_draw[0][0])
    pad_top, pad_bot = 14, 6
    plot_h = height - pad_top - pad_bot

    def _norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        if hi - lo < 1e-9:
            return np.full_like(arr, 0.5)
        return (arr - lo) / (hi - lo)

    xs = np.linspace(0, width - 1, n).astype(np.int32)

    for arr, color, _label in to_draw:
        normed = _norm(arr[:n])
        ys = (pad_top + plot_h - 1 - (normed * (plot_h - 1))).astype(np.int32)
        ys = np.clip(ys, pad_top, pad_top + plot_h - 1)
        pts = np.stack([xs[:len(ys)], ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

    # Legend in top-left
    x_off = 4
    for _, color, label in to_draw:
        cv2.putText(img, label, (x_off, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
        x_off += cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0] + 12

    # Playhead
    if 0 <= playhead < n:
        px = int(xs[playhead])
        cv2.line(img, (px, pad_top), (px, pad_top + plot_h), (255, 255, 255), 1)

    return img


def render_preview_frames(
    video_path: Path,
    raw_pose: PoseData,
    smooth_pose: PoseData,
    smoothing: PoseSmoothingSettings,
    start_frame: int,
    n_frames: int,
    target_height: int,
    cancel_flag: list[bool],
    local_radius_px: int = 0,
    fps: float = 30.0,
    mog2_var_threshold: int = 16,
) -> PreviewResult:
    """Build rendered frames + per-frame trace arrays (worker thread).

    Computes all traces defined in TRACE_CATALOG: video context features
    (MOG2, pixel change, variance), global motion (raw vs smoothed),
    centroid kinematics, and per-bodypart speeds.
    """
    import cv2  # noqa: PLC0415
    import re   # noqa: PLC0415

    _empty = PreviewResult(frames=[], traces={}, fps=fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return _empty

    raw_x = raw_pose.x.to_numpy(dtype=np.float32)
    raw_y = raw_pose.y.to_numpy(dtype=np.float32)
    raw_lk = raw_pose.likelihood.to_numpy(dtype=np.float32)

    sm_x = smooth_pose.x.to_numpy(dtype=np.float32)
    sm_y = smooth_pose.y.to_numpy(dtype=np.float32)

    # ── Resolve body-part indices by role ─────────────────────────────
    def _find_part(keywords: list[str]) -> int | None:
        for bi, bp in enumerate(smooth_pose.body_parts):
            lower = bp.lower()
            if any(kw in lower for kw in keywords):
                return bi
        return None

    nose_idx = _find_part(["nose", "snout", "rostrum"]) or 0

    # For paw detection use token matching like the context feature service
    part_tokens: dict[int, frozenset[str]] = {
        bi: frozenset(re.split(r"[_\-\s]+", bp.lower()))
        for bi, bp in enumerate(smooth_pose.body_parts)
    }
    fl_l_idx: int | None = None
    fl_r_idx: int | None = None
    tail_base_idx: int | None = None

    # Matching table: (required_tokens, target_variable_name)
    _match_table: list[tuple[tuple[str, ...], str]] = [
        # Forelimb / paw — left
        (("frontleg", "left"), "fl_l"),
        (("forepaw", "left"), "fl_l"),
        (("paw", "left"), "fl_l"),
        (("paw", "l"), "fl_l"),
        (("left", "paw"), "fl_l"),
        (("front", "left"), "fl_l"),
        # Forelimb / paw — right
        (("frontleg", "right"), "fl_r"),
        (("forepaw", "right"), "fl_r"),
        (("paw", "right"), "fl_r"),
        (("paw", "r"), "fl_r"),
        (("right", "paw"), "fl_r"),
        (("front", "right"), "fl_r"),
        # Tail base
        (("tail", "base"), "tb"),
        (("tailbase",), "tb"),
        (("tail", "root"), "tb"),
    ]
    _found: dict[str, int] = {}  # target_name -> body-part index
    for required, target in _match_table:
        if target in _found:
            continue
        for bi, tokens in part_tokens.items():
            if all(t in tokens for t in required):
                _found[target] = bi
                break
    fl_l_idx = _found.get("fl_l")
    fl_r_idx = _found.get("fl_r")
    tail_base_idx = _found.get("tb")

    radius = local_radius_px if local_radius_px > 0 else 36

    # ── Pre-compute centroid-based traces ─────────────────────────────
    raw_cx = np.asarray(raw_pose.centroid_x, dtype=np.float64)
    raw_cy = np.asarray(raw_pose.centroid_y, dtype=np.float64)
    sm_cx = np.asarray(smooth_pose.centroid_x, dtype=np.float64)
    sm_cy = np.asarray(smooth_pose.centroid_y, dtype=np.float64)
    end_frame = min(start_frame + n_frames, len(raw_cx))

    def _speed(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        seg_x = cx[start_frame:end_frame]
        seg_y = cy[start_frame:end_frame]
        dx = np.diff(seg_x, prepend=seg_x[0]) * fps
        dy = np.diff(seg_y, prepend=seg_y[0]) * fps
        return np.sqrt(dx ** 2 + dy ** 2)

    raw_speed = _speed(raw_cx, raw_cy)
    smooth_speed = _speed(sm_cx, sm_cy)

    # Smoothed centroid kinematics
    seg_cx = sm_cx[start_frame:end_frame]
    seg_cy = sm_cy[start_frame:end_frame]
    dx_c = np.diff(seg_cx, prepend=seg_cx[0]) * fps
    dy_c = np.diff(seg_cy, prepend=seg_cy[0]) * fps
    centroid_spd = np.sqrt(dx_c ** 2 + dy_c ** 2)
    ax_c = np.diff(dx_c, prepend=dx_c[0]) * fps
    ay_c = np.diff(dy_c, prepend=dy_c[0]) * fps
    centroid_acc = np.sqrt(ax_c ** 2 + ay_c ** 2)

    # Body angle: vector from tail-base to nose (or rear-most to nose fallback)
    if tail_base_idx is not None:
        rear_idx = tail_base_idx
    else:
        # Fallback: pick the body part with lowest mean y (most caudal in many setups)
        # or just use the last body part index as a rough proxy
        rear_idx = sm_x.shape[1] - 1
    nose_x_seg = sm_x[start_frame:end_frame, nose_idx].astype(np.float64)
    nose_y_seg = sm_y[start_frame:end_frame, nose_idx].astype(np.float64)
    rear_x_seg = sm_x[start_frame:end_frame, rear_idx].astype(np.float64)
    rear_y_seg = sm_y[start_frame:end_frame, rear_idx].astype(np.float64)
    body_ang = np.arctan2(nose_y_seg - rear_y_seg, nose_x_seg - rear_x_seg)

    # Per-bodypart speed helper (from smoothed coords)
    def _part_speed(pi: int) -> np.ndarray:
        px = sm_x[start_frame:end_frame, pi].astype(np.float64)
        py = sm_y[start_frame:end_frame, pi].astype(np.float64)
        dpx = np.diff(px, prepend=px[0]) * fps
        dpy = np.diff(py, prepend=py[0]) * fps
        return np.sqrt(dpx ** 2 + dpy ** 2)

    nose_spd = _part_speed(nose_idx)
    nose_acc_arr = np.abs(np.diff(nose_spd, prepend=nose_spd[0]) * fps)
    fl_l_spd = _part_speed(fl_l_idx) if fl_l_idx is not None else np.zeros(end_frame - start_frame)
    fl_r_spd = _part_speed(fl_r_idx) if fl_r_idx is not None else np.zeros(end_frame - start_frame)

    # ── MOG2 background subtractor (full-frame) ──────────────────────
    _DS = 2
    fg_sub = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=mog2_var_threshold, detectShadows=False,
    )

    _MOG2_WARMUP = 80
    warmup_start = max(0, start_frame - _MOG2_WARMUP)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    prev_gray: np.ndarray | None = None
    for wf in range(warmup_start, start_frame):
        ok, wframe = cap.read()
        if not ok or wf >= smooth_pose.n_frames:
            break
        wgray = cv2.cvtColor(wframe, cv2.COLOR_BGR2GRAY)
        wgray_ds = cv2.resize(
            wgray, (wgray.shape[1] // _DS, wgray.shape[0] // _DS),
            interpolation=cv2.INTER_AREA,
        )
        fg_sub.apply(wgray_ds, learningRate=0.01)
        prev_gray = wgray

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    label_smooth = (
        f"Smoothed  (w={smoothing.smoothing_window}fr, t={smoothing.likelihood_threshold:.2f})"
    )

    output_frames: list[np.ndarray] = []
    nose_energy_arr: list[float] = []
    nose_change_arr: list[float] = []
    nose_var_arr: list[float] = []
    _INSET = 80

    for i in range(n_frames):
        if cancel_flag[0]:
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        fi = start_frame + i
        if fi >= raw_pose.n_frames:
            break

        h_src, w_src = frame.shape[:2]
        scale = target_height / h_src if h_src > 0 else 1.0
        new_w = max(1, int(w_src * scale))
        scaled_radius = max(1, int(radius * scale)) if radius > 0 else 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_ds = cv2.resize(
            gray, (gray.shape[1] // _DS, gray.shape[0] // _DS),
            interpolation=cv2.INTER_AREA,
        )

        # ── Full-frame MOG2 then crop around nose ────────────────────
        fgmask_full = fg_sub.apply(gray_ds, learningRate=0.005)
        ncx, ncy = float(sm_x[fi, nose_idx]), float(sm_y[fi, nose_idx])
        ds_ncx, ds_ncy = ncx / _DS, ncy / _DS
        ds_radius = max(4, radius // _DS)
        nose_fgmask = _crop_box_static(fgmask_full, ds_ncx, ds_ncy, ds_radius)

        nose_energy_arr.append(float(np.mean(nose_fgmask.astype(np.float32)) / 255.0))

        # Pixel change and variance near the nose
        nose_crop_gray = _crop_box_static(gray, ncx, ncy, radius)
        nose_var_arr.append(float(np.var(nose_crop_gray.astype(np.float32))))

        if prev_gray is not None:
            prev_nose_crop = _crop_box_static(prev_gray, ncx, ncy, radius)
            hn = min(prev_nose_crop.shape[0], nose_crop_gray.shape[0])
            wn = min(prev_nose_crop.shape[1], nose_crop_gray.shape[1])
            if hn > 0 and wn > 0:
                nose_diff = cv2.absdiff(
                    nose_crop_gray[:hn, :wn].astype(np.float32),
                    prev_nose_crop[:hn, :wn].astype(np.float32),
                )
                nose_change_arr.append(float(np.mean(nose_diff) / 255.0))
            else:
                nose_change_arr.append(0.0)
        else:
            nose_change_arr.append(0.0)
        prev_gray = gray

        # ── Build display panes ──────────────────────────────────────
        left = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        right = left.copy()

        _overlay_pose(
            left, raw_x, raw_y, raw_lk,
            raw_pose.centroid_x, raw_pose.centroid_y,
            fi, scale,
            trail_color=(220, 220, 50),
            lk_threshold=smoothing.likelihood_threshold,
        )
        _overlay_pose(
            right, sm_x, sm_y, raw_lk,
            smooth_pose.centroid_x, smooth_pose.centroid_y,
            fi, scale,
            trail_color=(80, 255, 160),
            lk_threshold=smoothing.likelihood_threshold,
        )

        if scaled_radius > 0:
            h_r, w_r = right.shape[:2]
            overlay = right.copy()
            for pi in range(sm_x.shape[1]):
                conf = float(raw_lk[fi, pi]) if pi < raw_lk.shape[1] else 0.0
                if conf < smoothing.likelihood_threshold:
                    continue
                bx = int(sm_x[fi, pi] * scale)
                by = int(sm_y[fi, pi] * scale)
                if 0 <= bx < w_r and 0 <= by < h_r:
                    cv2.circle(overlay, (bx, by), scaled_radius, (120, 200, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5, right, 0.5, 0, right)

        # ── Background-subtraction inset ─────────────────────────────
        nx0 = max(0, int(ncy) - radius)
        nx1 = min(h_src, int(ncy) + radius)
        ny0 = max(0, int(ncx) - radius)
        ny1 = min(w_src, int(ncx) + radius)
        nose_crop_color = frame[nx0:nx1, ny0:ny1]
        if nose_crop_color.size > 0 and nose_fgmask.size > 0:
            fg_upscaled = cv2.resize(
                nose_fgmask,
                (nose_crop_color.shape[1], nose_crop_color.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            inset_color = cv2.resize(nose_crop_color, (_INSET, _INSET), interpolation=cv2.INTER_LINEAR)
            fg_resized = cv2.resize(fg_upscaled, (_INSET, _INSET), interpolation=cv2.INTER_NEAREST)
            fg_overlay = np.zeros((_INSET, _INSET, 3), dtype=np.uint8)
            fg_overlay[:, :, 1] = fg_resized
            fg_overlay[:, :, 2] = (fg_resized * 0.3).astype(np.uint8)
            inset = cv2.addWeighted(inset_color, 0.6, fg_overlay, 0.5, 0)
            cv2.rectangle(inset, (0, 0), (_INSET - 1, _INSET - 1), (80, 200, 80), 1)
            h_r, w_r = right.shape[:2]
            iy0 = h_r - _INSET - 6
            ix0 = w_r - _INSET - 6
            if iy0 >= 0 and ix0 >= 0:
                right[iy0:iy0 + _INSET, ix0:ix0 + _INSET] = inset
                cv2.putText(
                    right, "BG sub (nose)", (ix0, iy0 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 200, 80), 1, cv2.LINE_AA,
                )

        _draw_label(left, "Raw DLC tracking", (220, 220, 220))
        _draw_label(right, label_smooth, (80, 255, 160))

        divider = np.full((target_height, _DIVIDER_W, 3), 55, dtype=np.uint8)
        composite = np.concatenate([left, divider, right], axis=1)
        output_frames.append(composite)

    cap.release()
    n_out = len(output_frames)

    all_traces: dict[str, np.ndarray] = {
        "nose_bg_energy":       np.array(nose_energy_arr[:n_out], dtype=np.float64),
        "nose_px_change":       np.array(nose_change_arr[:n_out], dtype=np.float64),
        "nose_px_variance":     np.array(nose_var_arr[:n_out], dtype=np.float64),
        "raw_centroid_speed":   raw_speed[:n_out],
        "smooth_centroid_speed": smooth_speed[:n_out],
        "centroid_speed":       centroid_spd[:n_out],
        "centroid_accel":       centroid_acc[:n_out],
        "body_angle":           body_ang[:n_out],
        "nose_speed":           nose_spd[:n_out],
        "nose_accel":           nose_acc_arr[:n_out],
        "forelimb_l_speed":     fl_l_spd[:n_out],
        "forelimb_r_speed":     fl_r_spd[:n_out],
    }

    return PreviewResult(frames=output_frames, traces=all_traces, fps=fps)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class SmoothingPreviewDialog(QDialog):
    """Non-modal dialog: raw DLC tracking (left) vs smoothed (right).

    Parameters
    ----------
    import_service:
        Used to resolve video / pose paths from the manifest.
    manifest:
        The project's import manifest (sessions + smoothing settings).
    get_smoothing_fn:
        Zero-argument callable that returns the *current* ``PoseSmoothingSettings``
        from the parent UI, so the user can tweak settings and re-preview without
        closing the dialog.
    get_local_radius_fn:
        Zero-argument callable returning the current local-motion radius in pixels.
    project_root:
        Project directory for persisting the MOG2 threshold.
    """

    def __init__(
        self,
        import_service: ImportService,
        manifest: ImportManifest,
        get_smoothing_fn: Callable[[], PoseSmoothingSettings],
        get_local_radius_fn: Callable[[], int] | None = None,
        project_root: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preview Video Settings — Raw vs Smoothed DLC Tracking")
        self.setMinimumSize(860, 620)
        self.setModal(False)

        self._imports = import_service
        self._manifest = manifest
        self._get_smoothing = get_smoothing_fn
        self._get_local_radius = get_local_radius_fn or (lambda: 36)
        self._project_root = project_root
        self._pose = PoseProcessingService()
        self._pool = QThreadPool.globalInstance()

        self._frames: list[np.ndarray] = []
        self._current_frame_idx = 0
        self._cancel_flag: list[bool] = [False]

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 fps playback
        self._timer.timeout.connect(self._advance_frame)

        # --- Session selector row ---
        self._session_combo = QComboBox()
        self._session_combo.setMinimumWidth(300)
        for sess in manifest.linked_sessions:
            vid_name = ""
            for v in manifest.videos:
                if v.asset_id == sess.video_asset_id:
                    vid_name = Path(v.source_path).name
                    break
            self._session_combo.addItem(
                f"{sess.session_id}  ({vid_name})",
                userData=sess.session_id,
            )

        random_btn = QPushButton("Random")
        random_btn.setToolTip("Pick a random session")
        random_btn.clicked.connect(self._pick_random_session)

        self._generate_btn = QPushButton("▶  Generate Preview")
        self._generate_btn.setFixedHeight(34)
        self._generate_btn.clicked.connect(self._start_render)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Session:"))
        top_row.addWidget(self._session_combo, 1)
        top_row.addWidget(random_btn)
        top_row.addSpacing(8)
        top_row.addWidget(self._generate_btn)

        # --- Settings info bar ---
        self._settings_label = QLabel()
        self._settings_label.setWordWrap(True)
        self._settings_label.setStyleSheet("padding: 4px; background: #1e2a1e; border-radius: 3px;")

        # --- Optional tuning controls ---
        self._mog2_thresh = QSpinBox()
        self._mog2_thresh.setRange(4, 100)
        self._mog2_thresh.setValue(16)
        self._mog2_thresh.setToolTip(
            "MOG2 variance threshold \u2014 lower = more sensitive to subtle motion"
        )
        self._mog2_thresh.setSuffix("  var")
        self._mog2_thresh.valueChanged.connect(self._save_mog2_threshold)
        self._load_mog2_threshold()
        self._visible_traces: set[str] = set(_DEFAULT_TRACE_KEYS)

        trace_btn = QPushButton("Select Traces\u2026")
        trace_btn.setToolTip("Choose which dynamics traces appear in the graph strip")
        trace_btn.clicked.connect(self._open_trace_selector)

        tune_row = QHBoxLayout()
        tune_row.addWidget(QLabel("BG subtract sensitivity:"))
        tune_row.addWidget(self._mog2_thresh)
        tune_row.addSpacing(16)
        tune_row.addWidget(trace_btn)
        tune_row.addStretch()

        self._refresh_settings_label()

        # --- Frame display ---
        self._frame_label = QLabel("Click  ▶ Generate Preview  to render a 10-second clip.")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumHeight(_DISPLAY_HEIGHT)
        self._frame_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._frame_label.setStyleSheet("background: #0d0d0d; color: #777; font-size: 13px;")

        # --- Playback controls ---
        self._play_btn = QPushButton("⏸  Pause")
        self._play_btn.setFixedWidth(90)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_play)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_moved)

        self._frame_counter = QLabel("— / —")
        self._frame_counter.setFixedWidth(72)
        self._frame_counter.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(self._play_btn)
        ctrl_row.addWidget(self._slider, 1)
        ctrl_row.addWidget(self._frame_counter)

        # --- Progress bar (indeterminate, shown only while rendering) ---
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setFixedHeight(8)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)

        # --- Status label ---
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #aaa; font-size: 11px;")

        # --- Context-feature graph strip ---
        self._graph_label = QLabel()
        self._graph_label.setFixedHeight(_GRAPH_HEIGHT)
        self._graph_label.setStyleSheet("background: #121212;")
        self._graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._kinematic_data: PreviewResult | None = None

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.addLayout(top_row)
        layout.addWidget(self._settings_label)
        layout.addLayout(tune_row)
        layout.addWidget(self._progress)
        layout.addWidget(self._frame_label, 1)
        layout.addWidget(self._graph_label)
        layout.addLayout(ctrl_row)
        layout.addWidget(self._status)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_settings_label(self) -> None:
        s = self._get_smoothing()
        interp = f"yes (max gap {s.interpolate_max_gap} fr)" if s.interpolate_dropouts else "no"
        radius = self._get_local_radius()
        mog2 = self._mog2_thresh.value()
        self._settings_label.setText(
            f"<b>Current video settings</b> \u2014 "
            f"Smoothing: <b>{s.smoothing_window} frames</b>   |   "
            f"Likelihood: <b>{s.likelihood_threshold:.2f}</b>   |   "
            f"Interpolate: <b>{interp}</b>   |   "
            f"Local radius: <b>{radius} px</b>   |   "
            f"BG sensitivity: <b>{mog2}</b>"
        )

    def _pick_random_session(self) -> None:
        n = self._session_combo.count()
        if n > 0:
            self._session_combo.setCurrentIndex(random.randint(0, n - 1))

    def _current_session_id(self) -> str | None:
        idx = self._session_combo.currentIndex()
        return self._session_combo.itemData(idx) if idx >= 0 else None

    def _set_status(self, msg: str) -> None:
        self._status.setText(msg)

    def _save_mog2_threshold(self, _value: int = 0) -> None:
        """Persist MOG2 variance threshold to project.yaml."""
        if not self._project_root:
            return
        try:
            from abel.storage.file_store import read_yaml, write_yaml
            path = self._project_root / "project.yaml"
            raw = read_yaml(path, {})
            cfg = raw.setdefault("feature_extraction", {})
            cfg["mog2_var_threshold"] = self._mog2_thresh.value()
            write_yaml(path, raw)
        except Exception:
            pass

    def _load_mog2_threshold(self) -> None:
        """Restore MOG2 variance threshold from project.yaml."""
        if not self._project_root:
            return
        try:
            from abel.storage.file_store import read_yaml
            raw = read_yaml(self._project_root / "project.yaml", {})
            cfg = raw.get("feature_extraction") or {}
            val = cfg.get("mog2_var_threshold")
            if val is not None:
                self._mog2_thresh.blockSignals(True)
                self._mog2_thresh.setValue(int(val))
                self._mog2_thresh.blockSignals(False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _start_render(self) -> None:
        session_id = self._current_session_id()
        if not session_id:
            self._set_status("No session selected.")
            return

        video_path = self._imports.video_path_for_session(self._manifest, session_id)
        pose_path = self._imports.pose_path_for_session(self._manifest, session_id)

        if not video_path or not video_path.exists():
            self._set_status("Video file not found for this session — check import manifest.")
            return
        if not pose_path or not pose_path.exists():
            self._set_status("Pose file not found for this session — check import manifest.")
            return

        # Stop current playback
        self._timer.stop()
        self._frames = []
        self._cancel_flag[0] = False

        self._generate_btn.setEnabled(False)
        self._play_btn.setEnabled(False)
        self._slider.setEnabled(False)
        self._progress.setVisible(True)
        self._refresh_settings_label()
        self._set_status("Loading pose file…")
        self._frame_label.setText("Rendering… please wait.")

        smoothing = self._get_smoothing()
        local_radius = self._get_local_radius()
        mog2_thresh = self._mog2_thresh.value()

        # Capture variables for the closure
        _pose_svc = self._pose
        _cancel = self._cancel_flag

        def _work() -> PreviewResult:
            raw_pose = _pose_svc.load(pose_path)
            smooth_pose = _pose_svc.clean_pose(
                raw_pose,
                likelihood_threshold=smoothing.likelihood_threshold,
                interpolate=smoothing.interpolate_dropouts,
                interpolate_max_gap=smoothing.interpolate_max_gap,
                smoothing_window=smoothing.smoothing_window,
            )

            # Determine start frame using a random 10 s window
            fps_source = 30.0
            total_vid_frames = raw_pose.n_frames
            try:
                import cv2  # noqa: PLC0415
                cap_probe = cv2.VideoCapture(str(video_path))
                fps_source = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
                total_vid_frames = min(
                    raw_pose.n_frames,
                    int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or raw_pose.n_frames),
                )
                cap_probe.release()
            except Exception:
                pass

            n_preview = min(int(fps_source * _PREVIEW_SEC), total_vid_frames)
            max_start = max(0, total_vid_frames - n_preview)
            start_frame = random.randint(0, max_start) if max_start > 0 else 0

            return render_preview_frames(
                video_path=video_path,
                raw_pose=raw_pose,
                smooth_pose=smooth_pose,
                smoothing=smoothing,
                start_frame=start_frame,
                n_frames=n_preview,
                target_height=_DISPLAY_HEIGHT,
                cancel_flag=_cancel,
                local_radius_px=local_radius,
                fps=fps_source,
                mog2_var_threshold=mog2_thresh,
            )

        worker = TaskWorker(_work)
        worker.signals.finished.connect(self._on_render_done)
        worker.signals.failed.connect(self._on_render_failed)
        self._pool.start(worker)

    @Slot(object)
    def _on_render_done(self, rendered: object) -> None:
        result: PreviewResult = rendered  # type: ignore[assignment]
        self._progress.setVisible(False)
        self._generate_btn.setEnabled(True)

        if not result.frames:
            self._frame_label.setText("No frames rendered.")
            self._set_status(
                "Rendering produced no frames.  "
                "Ensure OpenCV is installed and the video file is accessible."
            )
            return

        self._frames = result.frames
        self._kinematic_data = result
        self._current_frame_idx = 0

        self._slider.setRange(0, len(result.frames) - 1)
        self._slider.setValue(0)
        self._slider.setEnabled(True)
        self._play_btn.setEnabled(True)
        self._play_btn.setText("⏸  Pause")

        self._show_frame(0)
        self._timer.start()
        self._set_status(
            f"Playing {len(result.frames)}-frame preview.  "
            "Left = raw DLC  |  Right = smoothed  |  "
            "Graph = context features near nose + global motion  |  "
            "Inset = background subtraction.  "
            "Adjust settings and click Generate to compare."
        )

    @Slot(str)
    def _on_render_failed(self, traceback_str: str) -> None:
        self._progress.setVisible(False)
        self._generate_btn.setEnabled(True)
        self._frame_label.setText("Rendering failed — see status below.")
        self._set_status(f"Error: {traceback_str.splitlines()[-1]}")
        logger.error("SmoothingPreviewDialog render error:\n%s", traceback_str)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._play_btn.setText("▶  Play")
        else:
            self._timer.start()
            self._play_btn.setText("⏸  Pause")

    def _advance_frame(self) -> None:
        if not self._frames:
            self._timer.stop()
            return
        next_idx = (self._current_frame_idx + 1) % len(self._frames)
        # Block slider signal temporarily to avoid double-advance
        self._slider.blockSignals(True)
        self._slider.setValue(next_idx)
        self._slider.blockSignals(False)
        self._current_frame_idx = next_idx
        self._show_frame(next_idx)

    def _on_slider_moved(self, value: int) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._play_btn.setText("▶  Play")
        self._current_frame_idx = value
        self._show_frame(value)

    def _show_frame(self, idx: int) -> None:
        if not self._frames or idx >= len(self._frames):
            return
        arr = self._frames[idx]
        h, w = arr.shape[:2]
        rgb = arr[:, :, ::-1].copy()  # BGR → RGB, ensure contiguous
        image = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        label_sz = self._frame_label.size()
        scaled = pixmap.scaled(
            label_sz,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._frame_label.setPixmap(scaled)
        self._frame_counter.setText(f"{idx + 1} / {len(self._frames)}")

        # Update kinematic graph with playhead
        self._update_graph(idx)

    def _update_graph(self, playhead: int) -> None:
        """Redraw the context-feature graph strip with the current playhead."""
        kd = self._kinematic_data
        if kd is None or not kd.traces:
            return
        graph_w = max(200, self._graph_label.width())
        graph = _render_graph_strip(
            kd.traces, self._visible_traces,
            width=graph_w, height=_GRAPH_HEIGHT, playhead=playhead,
        )
        rgb = graph[:, :, ::-1].copy()
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._graph_label.setPixmap(QPixmap.fromImage(image))

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._frames and self._current_frame_idx < len(self._frames):
            self._show_frame(self._current_frame_idx)

    # ------------------------------------------------------------------
    # Trace selector
    # ------------------------------------------------------------------

    def _open_trace_selector(self) -> None:
        """Open a popup checklist for choosing which traces appear in the graph."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Graph Traces")
        dlg.setMinimumWidth(300)

        vbox = QVBoxLayout(dlg)
        vbox.addWidget(QLabel("Check the traces you want displayed in the graph strip:"))

        _list = QListWidget()
        _list.setAlternatingRowColors(True)
        current_category: str | None = None
        for tdef in TRACE_CATALOG:
            if tdef.category != current_category:
                current_category = tdef.category
                header = QListWidgetItem(f"── {current_category} ──")
                header.setFlags(Qt.ItemFlag.NoItemFlags)  # non-interactive
                header.setForeground(QColor(150, 150, 150))
                _list.addItem(header)
            item = QListWidgetItem(tdef.label)
            item.setData(Qt.ItemDataRole.UserRole, tdef.key)
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(
                Qt.CheckState.Checked if tdef.key in self._visible_traces
                else Qt.CheckState.Unchecked
            )
            _list.addItem(item)

        vbox.addWidget(_list, 1)

        btn_row = QHBoxLayout()
        all_btn = QPushButton("All")
        none_btn = QPushButton("None")
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        btn_row.addWidget(all_btn)
        btn_row.addWidget(none_btn)
        btn_row.addStretch()
        btn_row.addWidget(ok_btn)
        vbox.addLayout(btn_row)

        def _set_all(state: Qt.CheckState) -> None:
            for r in range(_list.count()):
                it = _list.item(r)
                if it and it.data(Qt.ItemDataRole.UserRole) is not None:
                    it.setCheckState(state)

        all_btn.clicked.connect(lambda: _set_all(Qt.CheckState.Checked))
        none_btn.clicked.connect(lambda: _set_all(Qt.CheckState.Unchecked))
        ok_btn.clicked.connect(dlg.accept)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected: set[str] = set()
            for r in range(_list.count()):
                it = _list.item(r)
                if it is None:
                    continue
                key = it.data(Qt.ItemDataRole.UserRole)
                if key is not None and it.checkState() == Qt.CheckState.Checked:
                    selected.add(key)
            self._visible_traces = selected
            if self._frames and self._current_frame_idx < len(self._frames):
                self._update_graph(self._current_frame_idx)

    def closeEvent(self, event) -> None:
        self._cancel_flag[0] = True
        self._timer.stop()
        super().closeEvent(event)
