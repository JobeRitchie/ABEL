"""Compute per-frame environment context descriptors for behavior modeling."""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import threading
from typing import Any, Callable

import numpy as np
import pandas as pd

from abel.services.pose_processing_service import PoseData, PoseProcessingService
from abel.services.provenance_service import ProvenanceService
from abel.services.roi_service import ROIService
from abel.storage.file_store import read_json, write_json


# threading is retained for the gpu_flow_lock parameter used in frame-chunk
# processing.  The module-level write lock that previously serialised session
# writes has been removed in favour of per-session parquet files.
logger = logging.getLogger("abel")

# Semantic role → ordered list of token-sets.  A body-part name is split on
# word boundaries (_, -, space) and lower-cased; a part matches the first
# token-set where *every* required token appears among the part's tokens.
# Patterns are listed most-specific first so a part like ``forepaw_left``
# is preferred over a generic ``left`` match.
_KEYPOINT_ROLE_TOKENS: dict[str, list[tuple[str, ...]]] = {
    # ── Nose / head tip ──────────────────────────────────────────────────────
    "nose": [
        ("nose",), ("snout",), ("tip",), ("rostrum",), ("muzzle",), ("head",),
    ],
    # ── Left forepaw / front limb ────────────────────────────────────────────
    "paw_l": [
        ("paw", "left"),     ("forepaw", "left"),  ("front", "left"),
        ("forelimb", "left"),("wrist", "left"),    ("hand", "left"),
        ("paw", "l"),        ("forepaw", "l"),     ("limb", "l"),
        ("left", "paw"),     ("left", "fore"),     ("left", "hand"),
        ("lateral", "left"), ("lateral_left",),
    ],
    # ── Right forepaw / front limb ───────────────────────────────────────────
    "paw_r": [
        ("paw", "right"),    ("forepaw", "right"), ("front", "right"),
        ("forelimb", "right"),("wrist", "right"),  ("hand", "right"),
        ("paw", "r"),        ("forepaw", "r"),     ("limb", "r"),
        ("right", "paw"),    ("right", "fore"),    ("right", "hand"),
        ("lateral", "right"), ("lateral_right",),
    ],
}


@dataclass
class ContextFeatureConfig:
    model_version: str = "behavior_context_v1"
    feature_version: str = "context_features_v1"
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 3
    farneback_winsize: int = 15
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2
    prefer_gpu: bool = True
    downsample_factor: int = 0  # 0 = auto (target ~512 px long edge), 1 = none, 2+ = factor
    flow_temporal_stride: int = 10  # compute optical flow every Nth frame; 1 = every frame, 10 = every 10th (interpolate between)
    # Optical flow is the dominant cost of context extraction (~73% on a 640x480
    # session).  The flow features are mean magnitude / direction over small
    # patches — inherently low-frequency — so computing flow at reduced spatial
    # resolution and fewer LK iterations preserves the *information* (Pearson
    # corr ~0.97 vs full-res on real footage) for a ~4x flow speedup.  The flow
    # field is upsampled back to working resolution internally, so downstream
    # patch extraction is unchanged.  flow_compute_downsample=1 disables this.
    flow_compute_downsample: int = 2  # extra spatial downsample applied to flow only (1 = off)
    flow_iterations: int = 2  # LK warp-and-solve iterations on the GPU flow path (was 3)


class ContextFeatureService:
    """Extract optical-flow, substrate motion, and spatial context features."""

    def __init__(self) -> None:
        self._pose = PoseProcessingService()
        self._provenance = ProvenanceService()
        self._rois = ROIService()
        self._flow_backend_cache: dict[bool, str] = {}
        self._warned_external_drive_pairs: set[tuple[str, str]] = set()

    @staticmethod
    def _pixels_per_mm_for_session(project_root: Path, session_id: str) -> float | None:
        manifest_path = project_root / "derived" / "review_tables" / "import_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            raw = read_json(manifest_path, {})
        except Exception:
            return None

        videos = {str(v.get("asset_id", "")): v for v in raw.get("videos", [])}
        for sess in raw.get("linked_sessions", []):
            if str(sess.get("session_id", "")).strip() != str(session_id):
                continue
            val = sess.get("pixels_per_mm", None)
            if val is None:
                video = videos.get(str(sess.get("video_asset_id", "")), {})
                val = video.get("pixels_per_mm", None)
            try:
                out = float(val)
            except Exception:
                return None
            return out if out > 0 else None
        return None

    @staticmethod
    def _get_session_day_label(project_root: Path, session_id: str) -> str:
        """Return the day label for *session_id* by parsing its video filename.

        Reads ``config/session_registry.json`` and strips the subject-id prefix,
        e.g. ``CBMRE01_Acclimation.mp4`` with subject ``CBMRE01`` → ``Acclimation``.
        Returns empty string on any failure.
        """
        try:
            import json
            reg_path = project_root / "config" / "session_registry.json"
            if not reg_path.exists():
                return ""
            with open(reg_path, encoding="utf-8") as f:
                reg = json.load(f)
            entries = reg.get("entries", {})
            if not isinstance(entries, dict):
                return ""
            entry = entries.get(session_id, {})
            if not isinstance(entry, dict):
                return ""
            fname = entry.get("video_filename", "")
            if not fname:
                return ""
            stem = Path(fname).stem  # strip extension
            subject_id = entry.get("subject_id", "")
            if subject_id and stem.startswith(subject_id + "_"):
                return stem[len(subject_id) + 1:]
            # Fallback: everything after the last underscore
            return stem.rsplit("_", 1)[-1] if "_" in stem else stem
        except Exception:
            return ""

    @staticmethod
    def _resolve_keypoint(pose: PoseData, role: str) -> str | None:
        """Return the body-part name that best matches *role*, or None if absent.

        Each body-part name is tokenised on word boundaries (_, -, space) and
        lower-cased.  The role's candidate patterns are checked in priority
        order; the first part whose token-set contains every required token is
        returned.  This makes the mapping fully data-driven: any DLC keypoint
        naming convention is handled without hardcoding exact names.
        """
        patterns = _KEYPOINT_ROLE_TOKENS.get(role, [])
        part_tokens: dict[str, frozenset[str]] = {
            bp: frozenset(re.split(r"[_\-\s]+", bp.lower()))
            for bp in pose.body_parts
        }
        for required in patterns:
            for bp, tokens in part_tokens.items():
                if all(t in tokens for t in required):
                    return bp
        return None

    @staticmethod
    def _keypoint_series(pose: PoseData, role: str, axis: str) -> np.ndarray:
        """Return the x or y coordinate series for the body part matching *role*.

        Falls back to an all-NaN array when no matching part is found so that
        XGBoost treats absent keypoints as truly missing (using its learned
        default-direction split) rather than as a coordinate of (0, 0), which
        would produce a constant but spurious distance to every ROI.
        """
        part = ContextFeatureService._resolve_keypoint(pose, role)
        if part is None:
            return np.full(pose.n_frames, np.nan, dtype=float)
        return np.asarray((pose.x if axis == "x" else pose.y)[part], dtype=float)

    @staticmethod
    def _angle_to_target(src_x: np.ndarray, src_y: np.ndarray, tgt_x: float, tgt_y: float) -> np.ndarray:
        return np.arctan2(tgt_y - src_y, tgt_x - src_x)

    @staticmethod
    def _crop_box(frame: np.ndarray, x: float, y: float, radius: int = 12) -> np.ndarray:
        # Guard against NaN keypoints (DLC low-confidence / missing detections).
        if not np.isfinite(x) or not np.isfinite(y):
            return np.zeros((1, 1), dtype=frame.dtype)
        h, w = frame.shape[:2]
        x0 = max(0, int(x) - radius)
        x1 = min(w, int(x) + radius)
        y0 = max(0, int(y) - radius)
        y1 = min(h, int(y) + radius)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((1, 1), dtype=frame.dtype)
        return frame[y0:y1, x0:x1]

    @staticmethod
    def _roi_crop(frame: np.ndarray, roi: dict[str, Any]) -> np.ndarray:
        h, w = frame.shape[:2]
        x0 = max(0, int(roi.get("x", 0) or 0))
        y0 = max(0, int(roi.get("y", 0) or 0))
        rw = max(1, int(roi.get("w", w) or w))
        rh = max(1, int(roi.get("h", h) or h))
        x1 = min(w, x0 + rw)
        y1 = min(h, y0 + rh)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((1, 1), dtype=frame.dtype)
        return frame[y0:y1, x0:x1]

    @staticmethod
    def _has_roi(roi: dict[str, Any]) -> bool:
        return int(roi.get("w", 0) or 0) > 0 and int(roi.get("h", 0) or 0) > 0

    @staticmethod
    def _roi_center(roi: dict[str, Any]) -> tuple[float, float]:
        return (
            float(roi.get("x", 0) or 0) + float(roi.get("w", 0) or 0) / 2.0,
            float(roi.get("y", 0) or 0) + float(roi.get("h", 0) or 0) / 2.0,
        )

    @staticmethod
    def _scale_roi_for_ds(roi: dict, inv: float) -> dict:
        """Return *roi* scaled by *inv* (= 1/ds) for spatial downsampling."""
        if inv >= 1.0:
            return roi
        return {
            "x": int(int(roi.get("x", 0) or 0) * inv),
            "y": int(int(roi.get("y", 0) or 0) * inv),
            "w": max(1, int(int(roi.get("w", 0) or 0) * inv)),
            "h": max(1, int(int(roi.get("h", 0) or 0) * inv)),
        }

    @staticmethod
    def _entropy(values: np.ndarray, bins: int = 16) -> float:
        hist, _ = np.histogram(values, bins=bins, density=True)
        hist = hist[hist > 0]
        if hist.size == 0:
            return 0.0
        return float(-(hist * np.log(hist)).sum())

    @staticmethod
    def _resolve_downsample_factor(
        video_path: Path, config: ContextFeatureConfig
    ) -> tuple[int, int, int]:
        """Return ``(factor, native_width, native_height)``.

        When ``config.downsample_factor`` is 0 (auto), the factor is chosen so
        the long edge of the downsampled frame is close to 512 px.  The factor
        is always a power of two (1, 2, 4, 8 …) for clean integer division.
        Width/height are 0 when the video could not be probed.
        """
        ds = config.downsample_factor
        if ds >= 1:
            return ds, 0, 0
        # Auto-detect: peek at video header to get native resolution.
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except Exception:
            return 1, 0, 0
        long_edge = max(w, h)
        target = 512
        if long_edge <= target:
            return 1, w, h
        # Largest power-of-two factor that keeps long_edge >= target.
        factor = 1
        while long_edge // (factor * 2) >= target:
            factor *= 2
        return factor, w, h

    @staticmethod
    def _apply_downsample(
        ds: int,
        body_x: np.ndarray,
        body_y: np.ndarray,
        paw_l_x: np.ndarray,
        paw_l_y: np.ndarray,
        paw_r_x: np.ndarray,
        paw_r_y: np.ndarray,
        nose_x: np.ndarray,
        nose_y: np.ndarray,
        target_roi: dict,
        local_radius: int,
    ) -> tuple:
        """Scale coordinate arrays, radii, and ROI for spatial downsampling.

        Returns (body_x, body_y, paw_l_x, …, nose_y, target_roi, local_radius, nose_radius).
        Original arrays are not mutated.
        """
        if ds <= 1:
            return (body_x, body_y, paw_l_x, paw_l_y, paw_r_x, paw_r_y,
                    nose_x, nose_y, target_roi, local_radius, 10)
        inv = 1.0 / ds
        body_x = body_x * inv
        body_y = body_y * inv
        paw_l_x = paw_l_x * inv
        paw_l_y = paw_l_y * inv
        paw_r_x = paw_r_x * inv
        paw_r_y = paw_r_y * inv
        nose_x = nose_x * inv
        nose_y = nose_y * inv
        local_radius = max(4, local_radius // ds)
        nose_radius = max(4, 10 // ds)
        target_roi = {
            "x": int(int(target_roi.get("x", 0) or 0) * inv),
            "y": int(int(target_roi.get("y", 0) or 0) * inv),
            "w": max(1, int(int(target_roi.get("w", 0) or 0) * inv)),
            "h": max(1, int(int(target_roi.get("h", 0) or 0) * inv)),
        }
        return (body_x, body_y, paw_l_x, paw_l_y, paw_r_x, paw_r_y,
                nose_x, nose_y, target_roi, local_radius, nose_radius)

    @staticmethod
    def _process_video_chunk(
        video_path: Path,
        frame_start: int,
        frame_end: int,
        body_x: np.ndarray,
        body_y: np.ndarray,
        paw_l_x: np.ndarray,
        paw_l_y: np.ndarray,
        paw_r_x: np.ndarray,
        paw_r_y: np.ndarray,
        nose_x: np.ndarray,
        nose_y: np.ndarray,
        target_roi: dict,
        has_target: bool,
        local_radius: int,
        config: ContextFeatureConfig,
        extra_rois: "list[dict] | None" = None,
        mog2_warmup_frames: int = 50,
        _cv2_cuda_algo: "Any | None" = None,
    ) -> dict[str, list]:
        """Process a contiguous range of video frames, returning per-column value lists.

        Each chunk opens its own ``cv2.VideoCapture`` so multiple chunks can run
        on different threads simultaneously.  The MOG2 background model is
        pre-warmed by replaying up to *mog2_warmup_frames* preceding frames so
        that local-surface-motion features are well-calibrated even at mid-video
        boundaries.  Optical-flow continuity across chunk boundaries is ensured
        by reading the frame immediately before *frame_start* as the initial
        ``prev_gray``.  OpenCV's Farneback implementation releases the GIL, so
        concurrent threads genuinely overlap their CPU work.
        """
        try:
            import cv2
        except Exception as exc:
            raise ImportError("opencv-python is required for ContextFeatureService") from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        try:
            # ── Spatial downsampling ──────────────────────────────────────
            ds = max(1, config.downsample_factor)
            (body_x, body_y, paw_l_x, paw_l_y, paw_r_x, paw_r_y,
             nose_x, nose_y, target_roi, local_radius, nose_radius,
             ) = ContextFeatureService._apply_downsample(
                ds, body_x, body_y, paw_l_x, paw_l_y,
                paw_r_x, paw_r_y, nose_x, nose_y, target_roi, local_radius,
            )

            def _ds_gray(g: np.ndarray) -> np.ndarray:
                if ds <= 1:
                    return g
                return cv2.resize(
                    g, (g.shape[1] // ds, g.shape[0] // ds),
                    interpolation=cv2.INTER_AREA,
                )

            fg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=16, detectShadows=False
            )
            nose_fg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=16, detectShadows=False
            )
            _inv_ds = 1.0 / ds if ds > 1 else 1.0
            _scaled_extra_rois: list[dict] = [
                ContextFeatureService._scale_roi_for_ds(r, _inv_ds)
                for r in (extra_rois or [])
            ]

            # Seek back up to mog2_warmup_frames before chunk start so the
            # background model has history before we begin recording output.
            # This also leaves prev_gray populated for the first output frame.
            warmup_start = max(0, frame_start - mog2_warmup_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)

            prev_gray: np.ndarray | None = None
            prev_nose_crop: np.ndarray | None = None
            prev_nose_surface_crop: np.ndarray | None = None
            for wf in range(warmup_start, frame_start):
                ok, frame = cap.read()
                if not ok:
                    break
                gray = _ds_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                surface_crop = ContextFeatureService._crop_box(
                    gray, body_x[wf], body_y[wf], radius=local_radius
                )
                fg_subtractor.apply(surface_crop)
                nose_surface_crop = ContextFeatureService._crop_box(
                    gray, nose_x[wf], nose_y[wf], radius=local_radius
                )
                nose_fg_subtractor.apply(nose_surface_crop)
                prev_nose_surface_crop = nose_surface_crop
                prev_nose_crop = ContextFeatureService._crop_box(
                    gray, nose_x[wf], nose_y[wf], radius=nose_radius
                )
                prev_gray = gray

            # Output accumulators for this chunk
            local_surface_energy: list[float] = []
            local_surface_var: list[float] = []
            local_surface_change: list[float] = []
            nose_surface_energy: list[float] = []
            nose_surface_var: list[float] = []
            nose_surface_change: list[float] = []
            nose_local_change: list[float] = []
            nose_local_variance: list[float] = []
            flow_mag_paw_l: list[float] = []
            flow_mag_paw_r: list[float] = []
            flow_mag_nose: list[float] = []
            flow_mag_tmt: list[float] = []
            flow_dir_paw: list[float] = []
            flow_entropy_local: list[float] = []

            temporal_stride = max(1, config.flow_temporal_stride)
            # Temporary lists for strided flow features (CPU path).
            _strided_flow_indices: list[int] = []
            _sf_mag_paw_l: list[float] = []
            _sf_mag_paw_r: list[float] = []
            _sf_mag_nose: list[float] = []
            _sf_mag_tmt: list[float] = []
            _sf_dir_paw: list[float] = []
            _sf_entropy: list[float] = []
            # Per-extra-ROI optical flow accumulators
            _n_extra = len(_scaled_extra_rois)
            _sf_mag_extras: list[list[float]] = [[] for _ in range(_n_extra)]
            flow_mag_extras: list[list[float]] = [[] for _ in range(_n_extra)]

            for frame_idx in range(frame_start, frame_end):
                ok, frame = cap.read()
                if not ok:
                    break
                gray = _ds_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

                surface_crop = ContextFeatureService._crop_box(
                    gray, body_x[frame_idx], body_y[frame_idx], radius=local_radius
                )
                fgmask = fg_subtractor.apply(surface_crop)
                fgmask_arr = np.asarray(fgmask, dtype=np.float32)
                surface_arr = np.asarray(surface_crop, dtype=np.float32)
                local_surface_energy.append(float(np.mean(fgmask_arr) / 255.0))
                local_surface_var.append(float(np.var(surface_arr)))

                # ── Nose-area surface crop (wide radius, MOG2) ────────────────
                nose_surface_crop = ContextFeatureService._crop_box(
                    gray, nose_x[frame_idx], nose_y[frame_idx], radius=local_radius
                )
                nose_fgmask = nose_fg_subtractor.apply(nose_surface_crop)
                nose_fgmask_arr = np.asarray(nose_fgmask, dtype=np.float32)
                nose_surface_arr = np.asarray(nose_surface_crop, dtype=np.float32)
                nose_surface_energy.append(float(np.mean(nose_fgmask_arr) / 255.0))
                nose_surface_var.append(float(np.var(nose_surface_arr)))
                if prev_nose_surface_crop is None:
                    nose_sdiff = np.zeros_like(nose_surface_arr)
                else:
                    pns = np.asarray(prev_nose_surface_crop, dtype=np.float32)
                    hs = min(pns.shape[0], nose_surface_arr.shape[0])
                    ws = min(pns.shape[1], nose_surface_arr.shape[1])
                    if hs <= 0 or ws <= 0:
                        nose_sdiff = np.zeros_like(nose_surface_arr)
                    else:
                        nose_sdiff = np.asarray(
                            cv2.absdiff(nose_surface_arr[:hs, :ws], pns[:hs, :ws]),
                            dtype=np.float32,
                        )
                nose_surface_change.append(float(np.mean(nose_sdiff) / 255.0))
                prev_nose_surface_crop = nose_surface_crop

                # ── Nose-area crop (tight radius) ─────────────────────────────
                nose_crop = ContextFeatureService._crop_box(
                    gray, nose_x[frame_idx], nose_y[frame_idx], radius=nose_radius
                )
                nose_arr = np.asarray(nose_crop, dtype=np.float32)
                nose_local_variance.append(float(np.var(nose_arr)))
                if prev_nose_crop is None:
                    nose_diff_arr = np.zeros_like(nose_arr)
                else:
                    pn = np.asarray(prev_nose_crop, dtype=np.float32)
                    hn = min(pn.shape[0], nose_arr.shape[0])
                    wn = min(pn.shape[1], nose_arr.shape[1])
                    if hn <= 0 or wn <= 0:
                        nose_diff_arr = np.zeros_like(nose_arr)
                    else:
                        nose_diff_arr = np.asarray(
                            cv2.absdiff(nose_arr[:hn, :wn], pn[:hn, :wn]), dtype=np.float32
                        )
                nose_local_change.append(float(np.mean(nose_diff_arr) / 255.0))
                prev_nose_crop = nose_crop

                if prev_gray is None:
                    diff = np.zeros_like(surface_crop, dtype=np.float32)
                    flow = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
                else:
                    prev_cx = body_x[frame_idx - 1] if frame_idx > 0 else body_x[frame_idx]
                    prev_cy = body_y[frame_idx - 1] if frame_idx > 0 else body_y[frame_idx]
                    prev_sub = ContextFeatureService._crop_box(
                        prev_gray, prev_cx, prev_cy, radius=local_radius
                    )
                    h = min(prev_sub.shape[0], surface_crop.shape[0])
                    w = min(prev_sub.shape[1], surface_crop.shape[1])
                    if h <= 0 or w <= 0:
                        diff = np.zeros_like(surface_crop, dtype=np.float32)
                    else:
                        diff = cv2.absdiff(surface_crop[:h, :w], prev_sub[:h, :w])

                diff_arr = np.asarray(diff, dtype=np.float32)
                local_surface_change.append(float(np.mean(diff_arr) / 255.0))

                # ── Optical flow (compute or skip based on temporal stride) ────
                local_frame_idx = frame_idx - frame_start
                compute_flow = (
                    prev_gray is not None
                    and (temporal_stride <= 1 or local_frame_idx % temporal_stride == 0)
                )

                if compute_flow:
                    init_flow = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
                    if _cv2_cuda_algo is not None:
                        from abel.utils.gpu_optical_flow import compute_flow_cv2_cuda
                        flow = compute_flow_cv2_cuda(_cv2_cuda_algo, prev_gray, gray)
                    else:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray,
                            gray,
                            init_flow,
                            config.farneback_pyr_scale,
                            config.farneback_levels,
                            config.farneback_winsize,
                            config.farneback_iterations,
                            config.farneback_poly_n,
                            config.farneback_poly_sigma,
                            0,
                        )

                    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    if ds > 1:
                        mag *= ds
                    l_patch = ContextFeatureService._crop_box(mag, paw_l_x[frame_idx], paw_l_y[frame_idx])
                    r_patch = ContextFeatureService._crop_box(mag, paw_r_x[frame_idx], paw_r_y[frame_idx])
                    n_patch = ContextFeatureService._crop_box(mag, nose_x[frame_idx], nose_y[frame_idx])
                    t_patch = (
                        ContextFeatureService._roi_crop(mag, target_roi)
                        if has_target
                        else np.full((1, 1), np.nan, dtype=np.float32)
                    )

                    _sf_mag_paw_l.append(float(np.mean(l_patch)))
                    _sf_mag_paw_r.append(float(np.mean(r_patch)))
                    _sf_mag_nose.append(float(np.mean(n_patch)))
                    _sf_mag_tmt.append(float(np.mean(t_patch)))
                    _sf_entropy.append(ContextFeatureService._entropy(n_patch.ravel()))
                    for _ei, _eroi in enumerate(_scaled_extra_rois):
                        _e_has = int(_eroi.get("w", 0) or 0) > 0 and int(_eroi.get("h", 0) or 0) > 0
                        _e_patch = (
                            ContextFeatureService._roi_crop(mag, _eroi)
                            if _e_has
                            else np.full((1, 1), np.nan, dtype=np.float32)
                        )
                        _sf_mag_extras[_ei].append(float(np.mean(_e_patch)))

                    l_vec = np.array([
                        np.mean(ContextFeatureService._crop_box(flow[..., 0], paw_l_x[frame_idx], paw_l_y[frame_idx])),
                        np.mean(ContextFeatureService._crop_box(flow[..., 1], paw_l_x[frame_idx], paw_l_y[frame_idx])),
                    ])
                    r_vec = np.array([
                        np.mean(ContextFeatureService._crop_box(flow[..., 0], paw_r_x[frame_idx], paw_r_y[frame_idx])),
                        np.mean(ContextFeatureService._crop_box(flow[..., 1], paw_r_x[frame_idx], paw_r_y[frame_idx])),
                    ])
                    denom = float(np.linalg.norm(l_vec) * np.linalg.norm(r_vec))
                    if denom <= 1e-9:
                        _sf_dir_paw.append(0.0)
                    else:
                        _sf_dir_paw.append(float(np.clip(np.dot(l_vec, r_vec) / denom, -1.0, 1.0)))

                    _strided_flow_indices.append(local_frame_idx)

                prev_gray = gray

            # Interpolate strided flow features back to per-frame resolution.
            n_output = len(local_surface_energy)
            if temporal_stride > 1 and len(_strided_flow_indices) > 1:
                x_strided = np.array(_strided_flow_indices, dtype=float)
                x_full = np.arange(n_output, dtype=float)
                flow_mag_paw_l = np.interp(x_full, x_strided, _sf_mag_paw_l).tolist()
                flow_mag_paw_r = np.interp(x_full, x_strided, _sf_mag_paw_r).tolist()
                flow_mag_nose = np.interp(x_full, x_strided, _sf_mag_nose).tolist()
                flow_mag_tmt = np.interp(x_full, x_strided, _sf_mag_tmt).tolist()
                flow_dir_paw = np.interp(x_full, x_strided, _sf_dir_paw).tolist()
                flow_entropy_local = np.interp(x_full, x_strided, _sf_entropy).tolist()
            elif _strided_flow_indices:
                flow_mag_paw_l = _sf_mag_paw_l
                flow_mag_paw_r = _sf_mag_paw_r
                flow_mag_nose = _sf_mag_nose
                flow_mag_tmt = _sf_mag_tmt
                flow_dir_paw = _sf_dir_paw
                flow_entropy_local = _sf_entropy
            else:
                # No flow computed at all (e.g. single frame chunk with no prev_gray).
                flow_mag_paw_l = [0.0] * n_output
                flow_mag_paw_r = [0.0] * n_output
                flow_mag_nose = [0.0] * n_output
                flow_mag_tmt = [float("nan")] * n_output
                flow_dir_paw = [0.0] * n_output
                flow_entropy_local = [0.0] * n_output
            # Extra-ROI flow interpolation
            for _ei in range(_n_extra):
                if _strided_flow_indices and len(_strided_flow_indices) > 1 and temporal_stride > 1:
                    _x_s = np.array(_strided_flow_indices, dtype=float)
                    _x_f = np.arange(n_output, dtype=float)
                    flow_mag_extras[_ei] = np.interp(_x_f, _x_s, _sf_mag_extras[_ei]).tolist()
                elif _strided_flow_indices:
                    flow_mag_extras[_ei] = _sf_mag_extras[_ei]
                else:
                    flow_mag_extras[_ei] = [float("nan")] * n_output

            return {
                "local_surface_energy": local_surface_energy,
                "local_surface_var": local_surface_var,
                "local_surface_change": local_surface_change,
                "nose_surface_energy": nose_surface_energy,
                "nose_surface_var": nose_surface_var,
                "nose_surface_change": nose_surface_change,
                "nose_local_change": nose_local_change,
                "nose_local_variance": nose_local_variance,
                "flow_mag_paw_l": flow_mag_paw_l,
                "flow_mag_paw_r": flow_mag_paw_r,
                "flow_mag_nose": flow_mag_nose,
                "flow_mag_tmt": flow_mag_tmt,
                "flow_dir_paw": flow_dir_paw,
                "flow_entropy_local": flow_entropy_local,
                **{f"flow_mag_extra_{_ei}": flow_mag_extras[_ei] for _ei in range(_n_extra)},
            }
        finally:
            cap.release()

    @staticmethod
    def _process_video_chunk_gpu(
        video_path: Path,
        frame_start: int,
        frame_end: int,
        body_x: np.ndarray,
        body_y: np.ndarray,
        paw_l_x: np.ndarray,
        paw_l_y: np.ndarray,
        paw_r_x: np.ndarray,
        paw_r_y: np.ndarray,
        nose_x: np.ndarray,
        nose_y: np.ndarray,
        target_roi: dict,
        has_target: bool,
        local_radius: int,
        config: ContextFeatureConfig,
        extra_rois: "list[dict] | None" = None,
        mog2_warmup_frames: int = 50,
        gpu_flow_lock: "threading.Lock | None" = None,
        gpu_batch_size: int = 0,
        gpu_lock_timeout: float = 120.0,
        warning_cb: "Callable[[str], None] | None" = None,
    ) -> dict[str, list]:
        """Process a video chunk using batched GPU optical flow.

        Splits work into sub-batches to bound memory.  Cheap per-frame
        operations (MOG2, nose/surface diffs) run on CPU; only dense optical
        flow is dispatched to the GPU in batches via pyramidal Lucas-Kanade.
        """
        try:
            import cv2
        except Exception as exc:
            raise ImportError("opencv-python is required for ContextFeatureService") from exc

        from abel.utils.gpu_optical_flow import (
            compute_flow_pairs_gpu,
            GPUFlowWarning,
        )

        SUB_BATCH = 128  # frames read per iteration (bounds resident memory)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        try:
            # ── Spatial downsampling ──────────────────────────────────────
            ds = max(1, config.downsample_factor)
            (body_x, body_y, paw_l_x, paw_l_y, paw_r_x, paw_r_y,
             nose_x, nose_y, target_roi, local_radius, nose_radius,
             ) = ContextFeatureService._apply_downsample(
                ds, body_x, body_y, paw_l_x, paw_l_y,
                paw_r_x, paw_r_y, nose_x, nose_y, target_roi, local_radius,
            )

            def _ds_gray(g: np.ndarray) -> np.ndarray:
                if ds <= 1:
                    return g
                return cv2.resize(
                    g, (g.shape[1] // ds, g.shape[0] // ds),
                    interpolation=cv2.INTER_AREA,
                )

            fg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=16, detectShadows=False
            )
            nose_fg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=16, detectShadows=False
            )
            _inv_ds = 1.0 / ds if ds > 1 else 1.0
            _scaled_extra_rois: list[dict] = [
                ContextFeatureService._scale_roi_for_ds(r, _inv_ds)
                for r in (extra_rois or [])
            ]

            # ── Warmup (identical to CPU path) ───────────────────────────────
            warmup_start = max(0, frame_start - mog2_warmup_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)

            prev_gray: np.ndarray | None = None
            prev_nose_crop: np.ndarray | None = None
            prev_nose_surface_crop: np.ndarray | None = None
            for wf in range(warmup_start, frame_start):
                ok, frame = cap.read()
                if not ok:
                    break
                gray = _ds_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                surface_crop = ContextFeatureService._crop_box(
                    gray, body_x[wf], body_y[wf], radius=local_radius
                )
                fg_subtractor.apply(surface_crop)
                nose_surface_crop = ContextFeatureService._crop_box(
                    gray, nose_x[wf], nose_y[wf], radius=local_radius
                )
                nose_fg_subtractor.apply(nose_surface_crop)
                prev_nose_surface_crop = nose_surface_crop
                prev_nose_crop = ContextFeatureService._crop_box(
                    gray, nose_x[wf], nose_y[wf], radius=nose_radius
                )
                prev_gray = gray

            # ── Output accumulators ──────────────────────────────────────────
            local_surface_energy: list[float] = []
            local_surface_var: list[float] = []
            local_surface_change: list[float] = []
            nose_surface_energy: list[float] = []
            nose_surface_var: list[float] = []
            nose_surface_change: list[float] = []
            nose_local_change: list[float] = []
            nose_local_variance: list[float] = []
            flow_mag_paw_l: list[float] = []
            flow_mag_paw_r: list[float] = []
            flow_mag_nose: list[float] = []
            flow_mag_tmt: list[float] = []
            flow_dir_paw: list[float] = []
            flow_entropy_local: list[float] = []
            # Per-extra-ROI optical flow accumulators
            _n_extra = len(_scaled_extra_rois)
            flow_mag_extras: list[list[float]] = [[] for _ in range(_n_extra)]

            temporal_stride = max(1, config.flow_temporal_stride)

            total_frames = frame_end - frame_start
            for batch_off in range(0, total_frames, SUB_BATCH):
                batch_n = min(SUB_BATCH, total_frames - batch_off)

                # Phase 1 — read frames ────────────────────────────────────────
                gray_frames: list[np.ndarray] = []
                for _ in range(batch_n):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    gray_frames.append(_ds_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                actual_n = len(gray_frames)
                if actual_n == 0:
                    break

                # Phase 2 — cheap per-frame CPU work ───────────────────────────
                for i in range(actual_n):
                    fi = frame_start + batch_off + i
                    gray = gray_frames[i]

                    # BG subtraction on body crop
                    surface_crop = ContextFeatureService._crop_box(
                        gray, body_x[fi], body_y[fi], radius=local_radius
                    )
                    fgmask = fg_subtractor.apply(surface_crop)
                    local_surface_energy.append(
                        float(np.mean(np.asarray(fgmask, dtype=np.float32)) / 255.0)
                    )
                    local_surface_var.append(
                        float(np.var(np.asarray(surface_crop, dtype=np.float32)))
                    )

                    # Nose surface crop (wide radius, MOG2)
                    nose_surface_crop = ContextFeatureService._crop_box(
                        gray, nose_x[fi], nose_y[fi], radius=local_radius
                    )
                    nose_fgmask = nose_fg_subtractor.apply(nose_surface_crop)
                    nose_fgmask_arr = np.asarray(nose_fgmask, dtype=np.float32)
                    nose_surface_arr = np.asarray(nose_surface_crop, dtype=np.float32)
                    nose_surface_energy.append(
                        float(np.mean(nose_fgmask_arr) / 255.0)
                    )
                    nose_surface_var.append(
                        float(np.var(nose_surface_arr))
                    )
                    if prev_nose_surface_crop is None:
                        nose_sdiff = np.zeros_like(nose_surface_arr)
                    else:
                        pns = np.asarray(prev_nose_surface_crop, dtype=np.float32)
                        hs = min(pns.shape[0], nose_surface_arr.shape[0])
                        ws = min(pns.shape[1], nose_surface_arr.shape[1])
                        if hs <= 0 or ws <= 0:
                            nose_sdiff = np.zeros_like(nose_surface_arr)
                        else:
                            nose_sdiff = np.asarray(
                                cv2.absdiff(nose_surface_arr[:hs, :ws], pns[:hs, :ws]),
                                dtype=np.float32,
                            )
                    nose_surface_change.append(
                        float(np.mean(nose_sdiff) / 255.0)
                    )
                    prev_nose_surface_crop = nose_surface_crop

                    # Nose crop (tight radius)
                    nose_crop = ContextFeatureService._crop_box(
                        gray, nose_x[fi], nose_y[fi], radius=nose_radius
                    )
                    nose_arr = np.asarray(nose_crop, dtype=np.float32)
                    nose_local_variance.append(float(np.var(nose_arr)))
                    if prev_nose_crop is None:
                        nose_diff_arr = np.zeros_like(nose_arr)
                    else:
                        pn = np.asarray(prev_nose_crop, dtype=np.float32)
                        hn = min(pn.shape[0], nose_arr.shape[0])
                        wn = min(pn.shape[1], nose_arr.shape[1])
                        if hn <= 0 or wn <= 0:
                            nose_diff_arr = np.zeros_like(nose_arr)
                        else:
                            nose_diff_arr = np.asarray(
                                cv2.absdiff(nose_arr[:hn, :wn], pn[:hn, :wn]),
                                dtype=np.float32,
                            )
                    nose_local_change.append(float(np.mean(nose_diff_arr) / 255.0))
                    prev_nose_crop = nose_crop

                    # Surface diff
                    pg = prev_gray if i == 0 else gray_frames[i - 1]
                    if pg is None:
                        diff = np.zeros_like(surface_crop, dtype=np.float32)
                    else:
                        fi_prev = fi - 1 if fi > 0 else fi
                        prev_sub = ContextFeatureService._crop_box(
                            pg, body_x[fi_prev], body_y[fi_prev], radius=local_radius
                        )
                        h = min(prev_sub.shape[0], surface_crop.shape[0])
                        w = min(prev_sub.shape[1], surface_crop.shape[1])
                        if h <= 0 or w <= 0:
                            diff = np.zeros_like(surface_crop, dtype=np.float32)
                        else:
                            diff = cv2.absdiff(
                                surface_crop[:h, :w], prev_sub[:h, :w]
                            )
                    local_surface_change.append(
                        float(np.mean(np.asarray(diff, dtype=np.float32)) / 255.0)
                    )

                # Phase 3 — GPU optical flow at anchor positions only
                # Select one frame every temporal_stride steps as an anchor.
                # Each anchor is paired with its *immediate* predecessor
                # (1-frame gap), keeping inter-frame displacement small enough
                # for pyramidal LK to converge reliably.  GPU work scales with
                # len(stride_indices) rather than actual_n, giving ~stride×
                # speed-up (e.g. ×10 at stride=10 → ~13 GPU pairs per 128
                # frames instead of 128).  Bedding-scale changes (0.25–0.5 s)
                # are fully captured; per-frame values are recovered in Phase 4
                # via linear interpolation between anchor positions.
                if temporal_stride > 1:
                    stride_indices = list(range(0, actual_n, temporal_stride))
                    # Always include the last frame for interpolation coverage.
                    if stride_indices[-1] != actual_n - 1:
                        stride_indices.append(actual_n - 1)
                else:
                    stride_indices = list(range(actual_n))

                # Build explicit (prev, curr) pairs for each anchor.
                # si=0: prev is prev_gray (None on first sub-batch → zero flow).
                _anchor_prev = [
                    gray_frames[si - 1] if si > 0 else prev_gray
                    for si in stride_indices
                ]
                _anchor_curr = [gray_frames[si] for si in stride_indices]

                _lock_acquired = False
                if gpu_flow_lock is not None:
                    _lock_acquired = gpu_flow_lock.acquire(timeout=gpu_lock_timeout)
                    if not _lock_acquired:
                        _timeout_msg = (
                            f"GPU lock timed out after {gpu_lock_timeout:.0f}s for "
                            f"frames {frame_start + batch_off}-{frame_start + batch_off + actual_n - 1}. "
                            "Falling back to CPU optical flow for this sub-batch."
                        )
                        logger.warning("context_feature_service: %s", _timeout_msg)
                        if warning_cb is not None:
                            warning_cb(_timeout_msg)
                else:
                    _lock_acquired = True  # no lock needed

                try:
                    if _lock_acquired:
                        _flow_warnings = GPUFlowWarning()
                        anchor_flows = compute_flow_pairs_gpu(
                            _anchor_prev, _anchor_curr,
                            gpu_batch_size=gpu_batch_size,
                            iterations=int(getattr(config, "flow_iterations", 3) or 3),
                            compute_downsample=int(getattr(config, "flow_compute_downsample", 1) or 1),
                            warnings_out=_flow_warnings,
                        )
                        if _flow_warnings.had_issues and warning_cb is not None:
                            for msg in _flow_warnings.messages:
                                warning_cb(msg)
                    else:
                        # CPU Farneback fallback when GPU lock timed out.
                        _zero_f: np.ndarray | None = None
                        anchor_flows = []
                        for _ap, _ac in zip(_anchor_prev, _anchor_curr):
                            if _ap is None:
                                if _zero_f is None:
                                    _zero_f = np.zeros((*_ac.shape[:2], 2), dtype=np.float32)
                                anchor_flows.append(_zero_f)
                            else:
                                anchor_flows.append(
                                    cv2.calcOpticalFlowFarneback(
                                        _ap, _ac, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                                    )
                                )
                finally:
                    if _lock_acquired and gpu_flow_lock is not None:
                        gpu_flow_lock.release()

                # Phase 4 — extract flow-patch features ────────────────────────
                # Extract features at strided positions.
                _s_mag_paw_l: list[float] = []
                _s_mag_paw_r: list[float] = []
                _s_mag_nose: list[float] = []
                _s_mag_tmt: list[float] = []
                _s_dir_paw: list[float] = []
                _s_entropy: list[float] = []
                _s_mag_extras: list[list[float]] = [[] for _ in range(_n_extra)]

                for si_idx, si in enumerate(stride_indices):
                    fi = frame_start + batch_off + si
                    flow = anchor_flows[si_idx]

                    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    if ds > 1:
                        mag *= ds  # restore to original-resolution pixel units
                    l_patch = ContextFeatureService._crop_box(
                        mag, paw_l_x[fi], paw_l_y[fi]
                    )
                    r_patch = ContextFeatureService._crop_box(
                        mag, paw_r_x[fi], paw_r_y[fi]
                    )
                    n_patch = ContextFeatureService._crop_box(
                        mag, nose_x[fi], nose_y[fi]
                    )
                    t_patch = (
                        ContextFeatureService._roi_crop(mag, target_roi)
                        if has_target
                        else np.full((1, 1), np.nan, dtype=np.float32)
                    )

                    _s_mag_paw_l.append(float(np.mean(l_patch)))
                    _s_mag_paw_r.append(float(np.mean(r_patch)))
                    _s_mag_nose.append(float(np.mean(n_patch)))
                    _s_mag_tmt.append(float(np.mean(t_patch)))
                    _s_entropy.append(
                        ContextFeatureService._entropy(n_patch.ravel())
                    )
                    for _ei, _eroi in enumerate(_scaled_extra_rois):
                        _e_has = int(_eroi.get("w", 0) or 0) > 0 and int(_eroi.get("h", 0) or 0) > 0
                        _e_patch = (
                            ContextFeatureService._roi_crop(mag, _eroi)
                            if _e_has
                            else np.full((1, 1), np.nan, dtype=np.float32)
                        )
                        _s_mag_extras[_ei].append(float(np.mean(_e_patch)))

                    l_vec = np.array([
                        np.mean(ContextFeatureService._crop_box(
                            flow[..., 0], paw_l_x[fi], paw_l_y[fi]
                        )),
                        np.mean(ContextFeatureService._crop_box(
                            flow[..., 1], paw_l_x[fi], paw_l_y[fi]
                        )),
                    ])
                    r_vec = np.array([
                        np.mean(ContextFeatureService._crop_box(
                            flow[..., 0], paw_r_x[fi], paw_r_y[fi]
                        )),
                        np.mean(ContextFeatureService._crop_box(
                            flow[..., 1], paw_r_x[fi], paw_r_y[fi]
                        )),
                    ])
                    denom = float(
                        np.linalg.norm(l_vec) * np.linalg.norm(r_vec)
                    )
                    if denom <= 1e-9:
                        _s_dir_paw.append(0.0)
                    else:
                        _s_dir_paw.append(
                            float(
                                np.clip(
                                    np.dot(l_vec, r_vec) / denom, -1.0, 1.0
                                )
                            )
                        )

                # Interpolate strided features back to every-frame resolution.
                if temporal_stride > 1 and len(stride_indices) > 1:
                    x_strided = np.array(stride_indices, dtype=float)
                    x_full = np.arange(actual_n, dtype=float)
                    flow_mag_paw_l.extend(np.interp(x_full, x_strided, _s_mag_paw_l).tolist())
                    flow_mag_paw_r.extend(np.interp(x_full, x_strided, _s_mag_paw_r).tolist())
                    flow_mag_nose.extend(np.interp(x_full, x_strided, _s_mag_nose).tolist())
                    flow_mag_tmt.extend(np.interp(x_full, x_strided, _s_mag_tmt).tolist())
                    flow_dir_paw.extend(np.interp(x_full, x_strided, _s_dir_paw).tolist())
                    flow_entropy_local.extend(np.interp(x_full, x_strided, _s_entropy).tolist())
                    for _ei in range(_n_extra):
                        flow_mag_extras[_ei].extend(np.interp(x_full, x_strided, _s_mag_extras[_ei]).tolist())
                else:
                    flow_mag_paw_l.extend(_s_mag_paw_l)
                    flow_mag_paw_r.extend(_s_mag_paw_r)
                    flow_mag_nose.extend(_s_mag_nose)
                    flow_mag_tmt.extend(_s_mag_tmt)
                    flow_dir_paw.extend(_s_dir_paw)
                    flow_entropy_local.extend(_s_entropy)
                    for _ei in range(_n_extra):
                        flow_mag_extras[_ei].extend(_s_mag_extras[_ei])

                # Carry forward for next sub-batch.
                prev_gray = gray_frames[-1]
                del gray_frames, anchor_flows

            return {
                "local_surface_energy": local_surface_energy,
                "local_surface_var": local_surface_var,
                "local_surface_change": local_surface_change,
                "nose_surface_energy": nose_surface_energy,
                "nose_surface_var": nose_surface_var,
                "nose_surface_change": nose_surface_change,
                "nose_local_change": nose_local_change,
                "nose_local_variance": nose_local_variance,
                "flow_mag_paw_l": flow_mag_paw_l,
                "flow_mag_paw_r": flow_mag_paw_r,
                "flow_mag_nose": flow_mag_nose,
                "flow_mag_tmt": flow_mag_tmt,
                "flow_dir_paw": flow_dir_paw,
                "flow_entropy_local": flow_entropy_local,
                **{f"flow_mag_extra_{_ei}": flow_mag_extras[_ei] for _ei in range(_n_extra)},
            }
        finally:
            cap.release()

    def compute_frame_context(
        self,
        project_root: Path,
        video_path: Path,
        pose_path: Path,
        animal_id: str,
        session_id: str,
        config: ContextFeatureConfig | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
        intra_session_workers: int = 1,
        warning_cb: Callable[[str], None] | None = None,
        keypoint_aliases: "dict[str, str] | None" = None,
    ) -> pd.DataFrame:
        """Compute per-frame context features, optionally across parallel frame chunks.

        Parameters
        ----------
        progress_cb:
            Called as ``progress_cb(chunks_done, chunks_total, message)`` each
            time a frame chunk finishes, enabling live progress updates in the UI
            even before the full session is complete.
        intra_session_workers:
            Number of parallel worker threads used to process this session's
            frames.  Each worker handles a contiguous chunk of frames with its
            own VideoCapture handle.  ``1`` (default) uses the existing
            sequential path.
        warning_cb:
            Called with a human-readable message when a GPU issue is encountered
            (OOM fallback, lock timeout).  Allows the UI layer to surface
            warnings to the user.
        """
        config = config or ContextFeatureConfig()

        try:
            import cv2  # noqa: F401 — validate availability before heavy work
        except Exception as exc:
            raise ImportError("opencv-python is required for ContextFeatureService") from exc

        pose = self._pose.load_and_clean(pose_path, keypoint_aliases=keypoint_aliases)
        target_rois = self._rois.resolve_target_rois(project_root, f"{animal_id}::{session_id}")
        # Apply day-label ROI exclusions (e.g. Acclimation sessions have no object present)
        excluded_days = self._rois.get_roi_excluded_days(project_root)
        if excluded_days:
            day_label = ContextFeatureService._get_session_day_label(project_root, session_id)
            if day_label and day_label.lower() in {d.lower() for d in excluded_days}:
                target_rois = []
        target_roi = target_rois[0] if target_rois else {}
        has_target = self._has_roi(target_roi)
        if has_target:
            target_x, target_y = self._roi_center(target_roi)
        else:
            target_x, target_y = float("nan"), float("nan")
        local_radius = self._rois.local_motion_radius(project_root)

        nose_x = self._keypoint_series(pose, "nose", "x")
        nose_y = self._keypoint_series(pose, "nose", "y")
        paw_l_x = self._keypoint_series(pose, "paw_l", "x")
        paw_l_y = self._keypoint_series(pose, "paw_l", "y")
        paw_r_x = self._keypoint_series(pose, "paw_r", "x")
        paw_r_y = self._keypoint_series(pose, "paw_r", "y")

        body_x = pose.centroid_x
        body_y = pose.centroid_y

        n_workers = max(1, intra_session_workers)
        # Decouple reporting granularity from parallelism: always create enough
        # chunks that the user sees frequent progress updates even when only one
        # worker is available. Using workers*2 (min 8) means each worker keeps
        # a chunk queued while reporting the previous one.
        # EXCEPTION: when the video is on a different drive from the project
        # (external/network drive), each chunk incurs a costly H.264 seek.
        # In that case use exactly one chunk per worker to minimize seek cost.
        try:
            _on_ext_drive = Path(video_path).drive.lower() != Path(project_root).drive.lower()
        except Exception:
            _on_ext_drive = False
        if _on_ext_drive:
            n_report_chunks = n_workers
        else:
            n_report_chunks = max(n_workers * 2, 8)
        frame_chunks = [
            c for c in np.array_split(np.arange(pose.n_frames), n_report_chunks) if len(c) > 0
        ]
        chunk_boundaries = [(int(c[0]), int(c[-1]) + 1) for c in frame_chunks]
        n_chunks = len(chunk_boundaries)

        # ── Detect GPU optical-flow backend ────────────────────────────────
        flow_backend = self._flow_backend_cache.get(bool(config.prefer_gpu), "cpu")
        gpu_flow_lock = None
        if config.prefer_gpu and flow_backend == "cpu":
            try:
                from abel.utils.gpu_optical_flow import (
                    detect_flow_backend,
                    get_flow_lock,
                )
                flow_backend = detect_flow_backend()
                self._flow_backend_cache[True] = flow_backend
                if flow_backend == "torch":
                    gpu_flow_lock = get_flow_lock()
            except Exception:
                flow_backend = "cpu"
                self._flow_backend_cache[True] = "cpu"
        elif flow_backend == "torch":
            try:
                from abel.utils.gpu_optical_flow import get_flow_lock

                gpu_flow_lock = get_flow_lock()
            except Exception:
                flow_backend = "cpu"
                self._flow_backend_cache[True] = "cpu"
        logger.info("Context features using optical-flow backend: %s", flow_backend)

        # -- External-drive I/O warning -----------------------------------------
        try:
            _vid_drive = Path(video_path).drive.lower()
            _proj_drive = Path(project_root).drive.lower()
            if _vid_drive and _vid_drive != _proj_drive:
                _pair = (_vid_drive, _proj_drive)
                if _pair not in self._warned_external_drive_pairs:
                    self._warned_external_drive_pairs.add(_pair)
                    _io_msg = (
                        f"Video is on drive {_vid_drive.upper()} while project is on "
                        f"{_proj_drive.upper()} — reading from an external or network "
                        "drive can be the main bottleneck. Copy videos to the project "
                        "raw/videos/ folder and re-import for much faster processing."
                    )
                    logger.warning("Context features: %s", _io_msg)
                    if progress_cb is not None:
                        progress_cb(0, 1, _io_msg)
        except Exception:
            pass

        # ── Resolve spatial downsample factor once for all chunks ─────────
        if config.downsample_factor < 1:
            resolved_ds, _vid_w, _vid_h = self._resolve_downsample_factor(video_path, config)
            from dataclasses import replace as _dc_replace
            config = _dc_replace(config, downsample_factor=resolved_ds)
        else:
            _vid_w, _vid_h = 0, 0
        if config.downsample_factor > 1:
            if _vid_w and _vid_h:
                ds_w = _vid_w // config.downsample_factor
                ds_h = _vid_h // config.downsample_factor
                _ds_msg = (
                    f"Detected {_vid_w}×{_vid_h} — downsampling {config.downsample_factor}× "
                    f"to {ds_w}×{ds_h} for optical flow"
                )
            else:
                _ds_msg = f"Applying {config.downsample_factor}× spatial downsample for optical flow"
            logger.info("Context features: %s (%s)", _ds_msg, video_path.name)
            if progress_cb is not None:
                progress_cb(0, 1, _ds_msg)

        # Keyword arguments shared by every chunk call.
        chunk_kwargs: dict = dict(
            video_path=video_path,
            body_x=body_x,
            body_y=body_y,
            paw_l_x=paw_l_x,
            paw_l_y=paw_l_y,
            paw_r_x=paw_r_x,
            paw_r_y=paw_r_y,
            nose_x=nose_x,
            nose_y=nose_y,
            target_roi=target_roi,
            has_target=has_target,
            local_radius=local_radius,
            config=config,
            extra_rois=target_rois[1:],
        )

        # Select chunk processor based on detected backend.
        if flow_backend == "torch":
            chunk_fn = ContextFeatureService._process_video_chunk_gpu
            chunk_kwargs["gpu_flow_lock"] = gpu_flow_lock
            chunk_kwargs["warning_cb"] = warning_cb
        elif flow_backend == "cv2_cuda":
            # cv2.cuda Farneback is a per-frame drop-in; integrate via the
            # standard chunk processor by injecting the algorithm object.
            try:
                from abel.utils.gpu_optical_flow import create_cv2_cuda_farneback
                chunk_kwargs["_cv2_cuda_algo"] = create_cv2_cuda_farneback(config)
            except Exception:
                pass
            chunk_fn = ContextFeatureService._process_video_chunk
        else:
            chunk_fn = ContextFeatureService._process_video_chunk

        # ordered_results[i] holds the dict returned by chunk i.
        ordered_results: list[dict[str, list]] = [{}] * n_chunks

        # Always use the executor — even max_workers=1 will queue the chunks
        # sequentially and fire a progress callback after every one of them,
        # giving live updates throughout the entire optical-flow loop.
        def _make_chunk_runner(start_f: int, end_f: int, chunk_idx: int):
            """Wrap chunk_fn to fire a start-of-chunk progress message."""
            def _run() -> dict:
                if progress_cb is not None:
                    progress_cb(
                        0,
                        n_chunks,
                        f"chunk {chunk_idx + 1}/{n_chunks}: starting frames {start_f}–{end_f - 1}…",
                    )
                return chunk_fn(frame_start=start_f, frame_end=end_f, **chunk_kwargs)
            return _run

        with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_chunk_idx = {
                executor.submit(_make_chunk_runner(start, end, idx)): idx
                for idx, (start, end) in enumerate(chunk_boundaries)
            }
            chunks_done = 0
            for future in cf.as_completed(future_to_chunk_idx):
                chunk_data = future.result()  # re-raises any exception from the chunk
                idx = future_to_chunk_idx[future]
                ordered_results[idx] = chunk_data
                chunks_done += 1
                if progress_cb is not None:
                    start_f, end_f = chunk_boundaries[idx]
                    progress_cb(
                        chunks_done,
                        n_chunks,
                        f"chunk {chunks_done}/{n_chunks} done (frames {start_f}–{end_f - 1})",
                    )

        # Concatenate chunk results in frame order.
        def _concat(key: str) -> list:
            out: list = []
            for r in ordered_results:
                out.extend(r.get(key, []))
            return out

        local_surface_energy = _concat("local_surface_energy")
        local_surface_var = _concat("local_surface_var")
        local_surface_change = _concat("local_surface_change")
        nose_surface_energy_vals = _concat("nose_surface_energy")
        nose_surface_var_vals = _concat("nose_surface_var")
        nose_surface_change_vals = _concat("nose_surface_change")
        nose_local_change_vals = _concat("nose_local_change")
        nose_local_variance_vals = _concat("nose_local_variance")
        flow_mag_paw_l = _concat("flow_mag_paw_l")
        flow_mag_paw_r = _concat("flow_mag_paw_r")
        flow_mag_nose = _concat("flow_mag_nose")
        flow_mag_tmt = _concat("flow_mag_tmt")
        flow_dir_paw = _concat("flow_dir_paw")
        flow_entropy_local = _concat("flow_entropy_local")
        # Extra-ROI optical flow (one list per extra ROI, ROI-2 indexed)
        _extra_flow_lists: list[list] = [
            _concat(f"flow_mag_extra_{_ei}") for _ei in range(len(target_rois) - 1)
        ]

        n = len(local_surface_energy)
        paw_centroid_x = (paw_l_x[:n] + paw_r_x[:n]) / 2.0
        paw_centroid_y = (paw_l_y[:n] + paw_r_y[:n]) / 2.0

        px_per_mm = self._pixels_per_mm_for_session(project_root, session_id)
        dist_scale = (1.0 / float(px_per_mm)) if px_per_mm is not None else 1.0
        if px_per_mm is None:
            logger.warning(
                "Context features for session %s are in pixel units; set px/mm in Data Import for physical scaling.",
                session_id,
            )

        df = pd.DataFrame(
            {
                "frame": np.arange(n),
                "video_id": session_id,
                "animal_id": animal_id,
                "session_id": session_id,
                # ── Optical-flow features (always present) ──────────────────────
                "flow_mag_paw_L": np.asarray(flow_mag_paw_l[:n]),
                "flow_mag_paw_R": np.asarray(flow_mag_paw_r[:n]),
                "flow_mag_near_nose": np.asarray(flow_mag_nose[:n]),
                "flow_directionality_paw": np.asarray(flow_dir_paw[:n]),
                "flow_entropy_local": np.asarray(flow_entropy_local[:n]),
                # ── Local surface / environment motion (body-centroid crop) ──────
                # Computed in a small window around the body centroid.
                # Name is assay-agnostic: meaningful on any floor surface.
                "local_surface_motion_energy": np.asarray(local_surface_energy[:n]),
                "local_surface_motion_variance": np.asarray(local_surface_var[:n]),
                "local_surface_change_rate": np.asarray(local_surface_change[:n]),
                # ── Nose-area surface motion (wide crop, MOG2 background model) ──
                # Parallel to body-centroid local_surface features but centered
                # on the nose.  Captures substrate disruption at the point of
                # contact — especially useful for overhead cameras where paw
                # keypoints are absent.
                "nose_surface_motion_energy": np.asarray(nose_surface_energy_vals[:n]),
                "nose_surface_motion_variance": np.asarray(nose_surface_var_vals[:n]),
                "nose_surface_change_rate": np.asarray(nose_surface_change_vals[:n]),
                # ── Nose-area local change (tight ~10 px crop around nose tip) ───
                # Captures background transitions under the nose, e.g. head
                # dipping in EPM, nose-poke events, sniffing at novel objects.
                "nose_local_change_rate": np.asarray(nose_local_change_vals[:n]),
                "nose_local_variance": np.asarray(nose_local_variance_vals[:n]),
                # ── ROI presence indicators ──────────────────────────────────────
                # Binary flag (1.0 = ROI configured for this session, 0.0 = absent).
                # Lets the model distinguish "no object present (acclimation)"
                # from "far from object (test day)" — which XGBoost cannot infer
                # from NaN alone when distance features are missing.
                "roi_1_present": np.full(n, 1.0 if has_target else 0.0),
                # ── Target-zone optical flow (NaN when no target configured) ─────
                "flow_mag_near_target": np.asarray(flow_mag_tmt[:n]),
                # ── Target-zone spatial features (NaN when no target configured) ─
                "nose_to_target_dist": np.sqrt(
                    (nose_x[:n] - target_x) ** 2 + (nose_y[:n] - target_y) ** 2
                ) * dist_scale,
                "forepaw_centroid_to_target_dist": np.sqrt(
                    (paw_centroid_x - target_x) ** 2 + (paw_centroid_y - target_y) ** 2
                ) * dist_scale,
                "body_centroid_to_target_dist": np.sqrt(
                    (body_x[:n] - target_x) ** 2 + (body_y[:n] - target_y) ** 2
                ) * dist_scale,
                "head_angle_to_target": self._angle_to_target(
                    nose_x[:n], nose_y[:n], target_x, target_y
                ),
                "body_angle_to_target": self._angle_to_target(
                    body_x[:n], body_y[:n], target_x, target_y
                ),
            }
        )

        # ── Symmetric roi_1 aliases — mirrors the roi_N naming for extra zones ─
        df["nose_to_roi_1_dist"] = df["nose_to_target_dist"]
        df["forepaw_centroid_to_roi_1_dist"] = df["forepaw_centroid_to_target_dist"]
        df["body_centroid_to_roi_1_dist"] = df["body_centroid_to_target_dist"]
        df["head_angle_to_roi_1"] = df["head_angle_to_target"]
        df["body_angle_to_roi_1"] = df["body_angle_to_target"]
        df["flow_mag_near_roi_1"] = df["flow_mag_near_target"]
        # Extra-ROI optical flow columns — one per ROI 2, 3, …
        for _ei, _ef_list in enumerate(_extra_flow_lists):
            _roi_idx = _ei + 2
            df[f"flow_mag_near_roi_{_roi_idx}"] = (
                np.asarray(_ef_list[:n]) if _ef_list else np.full(n, np.nan)
            )

        # ── Additional ROI spatial features (ROI 2+) ─────────────────────────
        # For each extra ROI we add a presence indicator plus distance/angle
        # columns when the ROI is configured.  The indicator is always written
        # (0.0 when absent) so that the merged feature matrix has a consistent
        # set of columns across acclimation and test-day sessions.
        for _roi_idx, _extra_roi in enumerate(target_rois[1:], start=2):
            _roi_is_valid = self._has_roi(_extra_roi)
            df[f"roi_{_roi_idx}_present"] = np.full(n, 1.0 if _roi_is_valid else 0.0)
            if not _roi_is_valid:
                continue
            _rx, _ry = self._roi_center(_extra_roi)
            _sfx = f"_roi_{_roi_idx}"
            df[f"nose_to{_sfx}_dist"] = np.sqrt(
                (nose_x[:n] - _rx) ** 2 + (nose_y[:n] - _ry) ** 2
            ) * dist_scale
            df[f"forepaw_centroid_to{_sfx}_dist"] = np.sqrt(
                (paw_centroid_x - _rx) ** 2 + (paw_centroid_y - _ry) ** 2
            ) * dist_scale
            df[f"body_centroid_to{_sfx}_dist"] = np.sqrt(
                (body_x[:n] - _rx) ** 2 + (body_y[:n] - _ry) ** 2
            ) * dist_scale
            df[f"head_angle_to{_sfx}"] = self._angle_to_target(
                nose_x[:n], nose_y[:n], _rx, _ry
            )
            df[f"body_angle_to{_sfx}"] = self._angle_to_target(
                body_x[:n], body_y[:n], _rx, _ry
            )

        out_dir = project_root / "derived" / "context_features"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write directly to a per-session parquet file (lock-free).
        # See PoseProcessingService.extract_and_save_frame_pose_features for
        # rationale.  Call consolidate_session_files() once after all sessions
        # have been processed to rebuild the canonical frame_context.parquet.
        session_out = out_dir / "sessions" / f"{session_id}.parquet"
        session_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(session_out, index=False)

        prov = self._provenance.make_provenance(
            project_root=project_root,
            model_version=config.model_version,
            feature_version=config.feature_version,
            config={"context_feature_config": config.__dict__, "video": str(video_path), "pose": str(pose_path)},
        )
        write_json(out_dir / "frame_context.manifest.json", {"provenance": prov.model_dump(mode="json"), "rows": int(len(df))})
        return df

    @staticmethod
    def consolidate_session_files(project_root: Path) -> Path | None:
        """Merge per-session parquet files into the canonical frame_context.parquet.

        Sessions already in the monolithic file that were *not* updated this
        run are preserved.  Per-session files are authoritative for any
        session_id they contain.

        Returns the output path on success, or None if there is nothing to
        consolidate.
        """
        sessions_dir = project_root / "derived" / "context_features" / "sessions"
        out_path = project_root / "derived" / "context_features" / "frame_context.parquet"
        per_session_files = sorted(sessions_dir.glob("*.parquet")) if sessions_dir.exists() else []
        if not per_session_files:
            return out_path if out_path.exists() else None

        new_session_ids = {f.stem for f in per_session_files}
        parts: list[pd.DataFrame] = []

        if out_path.exists():
            try:
                legacy = pd.read_parquet(out_path)
                legacy_kept = legacy[~legacy["session_id"].astype(str).isin(new_session_ids)]
                if not legacy_kept.empty:
                    parts.append(legacy_kept)
            except Exception:
                pass

        for f in per_session_files:
            try:
                parts.append(pd.read_parquet(f))
            except Exception:
                pass

        if not parts:
            return None

        combined = pd.concat(parts, ignore_index=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, index=False)
        return out_path
