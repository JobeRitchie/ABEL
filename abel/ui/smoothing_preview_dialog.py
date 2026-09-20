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
from abel.services.roi_service import (
    BG_VAR_THRESHOLD_MAX,
    BG_VAR_THRESHOLD_MIN,
    DEFAULT_BG_VAR_THRESHOLD,
    ROIService,
)
from abel.workers.task_worker import TaskWorker

logger = logging.getLogger("abel")

BG_THRESHOLD_TOOLTIP = (
    "How different a pixel in the local motion window must be from its learned "
    "background (MOG2 variance threshold) before it counts as foreground. "
    "Lower = more sensitive: subtle movements and bedding shifts register, but "
    "so does video noise. Higher = only strong changes register. Default "
    f"{DEFAULT_BG_VAR_THRESHOLD}.\n\n"
    "Drives the local/nose surface-motion features (BG energy). This is a "
    "project setting used by feature extraction: changing it rebuilds context "
    "features at the next extraction."
)

_PREVIEW_SEC = 10
_DISPLAY_HEIGHT = 340   # pixels tall per pane
_DIVIDER_W = 4          # separator between the two panes
_TRAIL_FRAMES = 10      # centroid trail length
_GRAPH_HEIGHT = 100     # height of the graph strip


# ---------------------------------------------------------------------------
# Trace catalog: every plottable trace the preview can compute
# ---------------------------------------------------------------------------

@dataclass
class TraceDef:
    """Metadata for one time-series trace available in the graph strip."""
    key: str            # dict key in PreviewResult.traces
    label: str          # legend text (ASCII only: OpenCV can't render Unicode)
    color: tuple[int, int, int]  # BGR
    category: str       # grouping shown in the selector dialog
    default_on: bool = True


# Ordered list: defines default draw order and selector list order.
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
# Rendering helpers (run in worker thread: no Qt calls allowed here)
# ---------------------------------------------------------------------------

def _part_color(part_idx: int) -> tuple[int, int, int]:
    """Distinct, deterministic BGR color palette."""
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
    fixed_color: tuple[int, int, int] | None = None,
) -> None:
    """Draw body-part dots and centroid trail onto *canvas* in-place.

    When *fixed_color* is given, all of this animal's keypoints are drawn in that
    single color (used to tell apart multiple animals); otherwise each body part
    uses its own role color.
    """
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
        pcx, pcy = centroid_x[prev], centroid_y[prev]
        if not (np.isfinite(pcx) and np.isfinite(pcy)):
            continue  # animal absent at this past frame: no trail point
        cx = int(pcx * scale)
        cy = int(pcy * scale)
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
        px, py = x_vals[frame_idx, pi], y_vals[frame_idx, pi]
        if not (np.isfinite(px) and np.isfinite(py)):
            continue  # keypoint undetected (animal absent / dropout)
        bx = int(px * scale)
        by = int(py * scale)
        if not (0 <= bx < w and 0 <= by < h):
            continue
        if conf >= lk_threshold:
            color = fixed_color if fixed_color is not None else _part_color(pi)
        else:
            color = (60, 60, 60)
        radius = 4 if conf >= lk_threshold else 2
        cv2.circle(canvas, (bx, by), radius, color, -1)
        if conf >= lk_threshold:
            cv2.circle(canvas, (bx, by), radius + 1, (0, 0, 0), 1)  # outline


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
    extra_raw: list[PoseData] | None = None,
    extra_smooth: list[PoseData] | None = None,
    bg_var_threshold: int = 0,
) -> PreviewResult:
    """Build rendered frames + per-frame trace arrays (worker thread).

    Computes all traces defined in TRACE_CATALOG: video context features
    (MOG2, pixel change, variance), global motion (raw vs smoothed),
    centroid kinematics, and per-bodypart speeds.

    ``extra_raw``/``extra_smooth`` are additional animals (multi-animal sessions);
    their poses are overlaid on the raw/smoothed panes in distinct per-animal
    colors so the preview shows every tracked animal.  Traces remain focused on
    the primary animal (``raw_pose``/``smooth_pose``).
    """
    import cv2  # noqa: PLC0415
    import re   # noqa: PLC0415

    _empty = PreviewResult(frames=[], traces={}, fps=fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return _empty

    # Distinct BGR colors for additional animals (primary keeps role colors).
    _EXTRA_COLORS = [
        (255, 160, 80), (180, 90, 240), (90, 230, 230),
        (120, 255, 120), (80, 120, 255), (200, 200, 80),
    ]
    extra_raw = extra_raw or []
    extra_smooth = extra_smooth or []
    extras = []
    for k, ep in enumerate(extra_raw):
        es = extra_smooth[k] if k < len(extra_smooth) else ep
        extras.append({
            "raw_x": ep.x.to_numpy(dtype=np.float32),
            "raw_y": ep.y.to_numpy(dtype=np.float32),
            "raw_lk": ep.likelihood.to_numpy(dtype=np.float32),
            "sm_x": es.x.to_numpy(dtype=np.float32),
            "sm_y": es.y.to_numpy(dtype=np.float32),
            "raw_cx": np.asarray(ep.centroid_x, dtype=np.float64),
            "raw_cy": np.asarray(ep.centroid_y, dtype=np.float64),
            "sm_cx": np.asarray(es.centroid_x, dtype=np.float64),
            "sm_cy": np.asarray(es.centroid_y, dtype=np.float64),
            "color": _EXTRA_COLORS[k % len(_EXTRA_COLORS)],
            "n": ep.n_frames,
        })

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
        # Forelimb / paw: left
        (("frontleg", "left"), "fl_l"),
        (("forepaw", "left"), "fl_l"),
        (("paw", "left"), "fl_l"),
        (("paw", "l"), "fl_l"),
        (("left", "paw"), "fl_l"),
        (("front", "left"), "fl_l"),
        # Forelimb / paw: right
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

    # ── Nose-window video context, computed as extraction computes it ──
    # Same moving fixed-size window, spatial downsample and MOG2 settings as
    # ContextFeatureService, so these traces are the extracted
    # nose_surface_energy / _change / _var for the current smoothing.
    from abel.services.context_feature_service import (  # noqa: PLC0415
        MOG2_HISTORY,
        MOG2_VAR_THRESHOLD,
        MOG2_WARMUP_FRAMES,
        ContextFeatureConfig,
        ContextFeatureService,
    )

    try:
        _ds, _, _ = ContextFeatureService._resolve_downsample_factor(
            Path(video_path), ContextFeatureConfig()
        )
    except Exception:
        _ds = 1
    _ds = max(1, int(_ds))
    ds_radius = max(4, radius // _ds) if _ds > 1 else radius
    nose_lx, nose_ly = ContextFeatureService._hold_last_position(
        sm_x[:, nose_idx].astype(np.float64) / _ds,
        sm_y[:, nose_idx].astype(np.float64) / _ds,
    )

    def _ds_gray(g: np.ndarray) -> np.ndarray:
        if _ds <= 1:
            return g
        return cv2.resize(g, (g.shape[1] // _ds, g.shape[0] // _ds), interpolation=cv2.INTER_AREA)

    def _nose_window(g_ds: np.ndarray, fi: int) -> np.ndarray:
        return ContextFeatureService._local_crop(g_ds, nose_lx[fi], nose_ly[fi], ds_radius)

    fg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=bg_var_threshold if bg_var_threshold > 0 else MOG2_VAR_THRESHOLD,
        detectShadows=False,
    )
    warmup_start = max(0, start_frame - MOG2_WARMUP_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    prev_window: np.ndarray | None = None
    for wf in range(warmup_start, start_frame):
        ok, wframe = cap.read()
        if not ok or wf >= smooth_pose.n_frames:
            break
        prev_window = _nose_window(_ds_gray(cv2.cvtColor(wframe, cv2.COLOR_BGR2GRAY)), wf)
        fg_sub.apply(prev_window)

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

        # ── Nose window: MOG2 energy, pixel change, pixel variance ────
        window = _nose_window(_ds_gray(gray), fi)
        nose_fgmask = fg_sub.apply(window)
        window_f = window.astype(np.float32)
        nose_energy_arr.append(float(np.mean(nose_fgmask.astype(np.float32)) / 255.0))
        nose_var_arr.append(float(np.var(window_f)))
        if prev_window is None:
            nose_change_arr.append(0.0)
        else:
            nose_change_arr.append(
                float(np.mean(cv2.absdiff(window_f, prev_window.astype(np.float32))) / 255.0)
            )
        prev_window = window

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

        # ── Additional animals (multi-animal sessions) ───────────────
        for ex in extras:
            if fi >= ex["n"]:
                continue
            _overlay_pose(
                left, ex["raw_x"], ex["raw_y"], ex["raw_lk"],
                ex["raw_cx"], ex["raw_cy"], fi, scale,
                trail_color=ex["color"], lk_threshold=smoothing.likelihood_threshold,
                fixed_color=ex["color"],
            )
            _overlay_pose(
                right, ex["sm_x"], ex["sm_y"], ex["raw_lk"],
                ex["sm_cx"], ex["sm_cy"], fi, scale,
                trail_color=ex["color"], lk_threshold=smoothing.likelihood_threshold,
                fixed_color=ex["color"],
            )

        if scaled_radius > 0:
            # The two local-motion windows extraction samples: fixed squares
            # (side 2 x Local radius) on the body center and on the nose.
            overlay = right.copy()
            body_hx, body_hy = ContextFeatureService._hold_last_position(
                sm_cx[fi:fi + 1], sm_cy[fi:fi + 1]
            )
            for wx, wy in ((body_hx[0], body_hy[0]), (nose_lx[fi] * _ds, nose_ly[fi] * _ds)):
                if not (np.isfinite(wx) and np.isfinite(wy)):
                    continue
                bx, by = int(wx * scale), int(wy * scale)
                cv2.rectangle(
                    overlay,
                    (bx - scaled_radius, by - scaled_radius),
                    (bx + scaled_radius, by + scaled_radius),
                    (120, 200, 255), 1, cv2.LINE_AA,
                )
            cv2.addWeighted(overlay, 0.5, right, 0.5, 0, right)

        # ── Background-subtraction inset ─────────────────────────────
        # The same window MOG2 saw, at full resolution, so the mask lines up.
        nose_crop_color = ContextFeatureService._local_crop(
            frame, nose_lx[fi] * _ds, nose_ly[fi] * _ds, ds_radius * _ds
        )
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
        Project directory.  When given, the BG-subtraction threshold is read
        from and saved to the project's ROI config (the setting extraction uses).
    get_bg_threshold_fn / set_bg_threshold_fn:
        Optional callables bound to the parent UI's own threshold control, so
        the two stay in step.  They take precedence over *project_root*.
    """

    def __init__(
        self,
        import_service: ImportService,
        manifest: ImportManifest,
        get_smoothing_fn: Callable[[], PoseSmoothingSettings],
        get_local_radius_fn: Callable[[], int] | None = None,
        project_root: Path | None = None,
        parent: QWidget | None = None,
        get_bg_threshold_fn: Callable[[], int] | None = None,
        set_bg_threshold_fn: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preview Video Settings: Raw vs Smoothed DLC Tracking")
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
        self._visible_traces: set[str] = set(_DEFAULT_TRACE_KEYS)

        trace_btn = QPushButton("Select Traces\u2026")
        trace_btn.setToolTip("Choose which dynamics traces appear in the graph strip")
        trace_btn.clicked.connect(self._open_trace_selector)

        # The nose traces reproduce extraction exactly (same window, downsample
        # and MOG2 settings).  The threshold control edits the project setting
        # itself, never a preview-only copy.
        self._get_bg_threshold = get_bg_threshold_fn
        self._set_bg_threshold = set_bg_threshold_fn
        self._bg_threshold = QSpinBox()
        self._bg_threshold.setRange(BG_VAR_THRESHOLD_MIN, BG_VAR_THRESHOLD_MAX)
        self._bg_threshold.setValue(self._initial_bg_threshold())
        self._bg_threshold.setToolTip(BG_THRESHOLD_TOOLTIP)
        self._bg_threshold.valueChanged.connect(self._on_bg_threshold_changed)

        bg_note = QLabel(
            "Nose traces = the extracted nose-window features: background "
            "subtraction inside the square Local-radius window only. Changing "
            "the threshold rebuilds context features at the next extraction."
        )
        bg_note.setWordWrap(True)
        bg_note.setStyleSheet("color: #90A4AE; font-size: 11px;")

        tune_row = QHBoxLayout()
        tune_row.addWidget(trace_btn)
        tune_row.addSpacing(16)
        tune_row.addWidget(QLabel("BG sensitivity threshold:"))
        tune_row.addWidget(self._bg_threshold)
        tune_row.addSpacing(16)
        tune_row.addWidget(bg_note, 1)

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

        self._frame_counter = QLabel("- / -")
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
        self._settings_label.setText(
            f"<b>Current video settings:</b> "
            f"Smoothing: <b>{s.smoothing_window} frames</b>   |   "
            f"Likelihood: <b>{s.likelihood_threshold:.2f}</b>   |   "
            f"Interpolate: <b>{interp}</b>   |   "
            f"Local radius: <b>{radius} px</b> ({2 * radius}\u00d7{2 * radius} px windows)   |   "
            f"BG threshold: <b>{self._bg_threshold.value()}</b>"
        )

    def _initial_bg_threshold(self) -> int:
        try:
            if self._get_bg_threshold is not None:
                return int(self._get_bg_threshold())
            if self._project_root is not None:
                return ROIService().bg_var_threshold(self._project_root)
        except Exception:
            logger.debug("Could not read BG threshold", exc_info=True)
        return DEFAULT_BG_VAR_THRESHOLD

    def _on_bg_threshold_changed(self, value: int) -> None:
        try:
            if self._set_bg_threshold is not None:
                self._set_bg_threshold(int(value))
            elif self._project_root is not None:
                ROIService().set_bg_var_threshold(self._project_root, int(value))
        except Exception:
            logger.warning("Could not save BG threshold", exc_info=True)
        self._refresh_settings_label()

    def _pick_random_session(self) -> None:
        n = self._session_combo.count()
        if n > 0:
            self._session_combo.setCurrentIndex(random.randint(0, n - 1))

    def _current_session_id(self) -> str | None:
        idx = self._session_combo.currentIndex()
        return self._session_combo.itemData(idx) if idx >= 0 else None

    def _set_status(self, msg: str) -> None:
        self._status.setText(msg)


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
            self._set_status("Video file not found for this session: check import manifest.")
            return
        if not pose_path or not pose_path.exists():
            self._set_status("Pose file not found for this session: check import manifest.")
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
        bg_threshold = self._bg_threshold.value()

        # Capture variables for the closure
        _pose_svc = self._pose
        _cancel = self._cancel_flag

        def _work() -> PreviewResult:
            def _clean(p):
                return _pose_svc.clean_pose(
                    p,
                    likelihood_threshold=smoothing.likelihood_threshold,
                    interpolate=smoothing.interpolate_dropouts,
                    interpolate_max_gap=smoothing.interpolate_max_gap,
                    smoothing_window=smoothing.smoothing_window,
                    absence_max_fill_frames=smoothing.absence_max_fill_frames,
                )

            # Load every tracked individual; the first is the primary animal
            # (drives the traces/MOG2) and the rest are overlaid as extras.
            multi_raw = _pose_svc.load_multi(pose_path)
            inds = list(multi_raw.individuals)
            raw_pose = multi_raw.per_individual[inds[0]]
            smooth_pose = _clean(raw_pose)
            extra_raw = [multi_raw.per_individual[i] for i in inds[1:]]
            extra_smooth = [_clean(p) for p in extra_raw]

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

            # Prefer a window where at least one animal is actually present,
            # these videos have long stretches with no mouse, which would render
            # an empty (and previously crash-prone) preview.  Presence = any
            # individual has a finite centroid in the window.
            present = np.zeros(total_vid_frames, dtype=bool)
            for _ind in inds:
                p = multi_raw.per_individual[_ind]
                cx = np.asarray(p.centroid_x, dtype=float)[:total_vid_frames]
                cy = np.asarray(p.centroid_y, dtype=float)[:total_vid_frames]
                m = np.isfinite(cx) & np.isfinite(cy)
                present[: len(m)] |= m
            if max_start > 0 and present.any():
                # Use the windowed presence count to bias toward populated windows;
                # pick randomly among the best-covered candidates for variety.
                win = max(1, n_preview)
                csum = np.concatenate([[0], np.cumsum(present.astype(int))])
                coverage = csum[win : max_start + win + 1] - csum[0 : max_start + 1]
                best = int(coverage.max())
                if best > 0:
                    candidates = np.flatnonzero(coverage >= max(1, int(best * 0.75)))
                    start_frame = int(random.choice(candidates.tolist()))
                else:
                    start_frame = random.randint(0, max_start)
            else:
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
                extra_raw=extra_raw,
                extra_smooth=extra_smooth,
                bg_var_threshold=bg_threshold,
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
        self._frame_label.setText("Rendering failed: see status below.")
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
