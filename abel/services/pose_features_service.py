"""Pose feature extraction service.

Cleans DLC tracking data and computes per-window kinematic feature vectors
across the *entire* recording without any video decoding.  The resulting
.npz feature matrix is consumed by downstream behavior representation and
candidate ranking steps.

Pipeline position:
    Data Import → Behavior Definitions → Seed Examples
    → **Pose Features** ← here
    → Behavior Representations → Candidate Generation → Review
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from abel.models.schemas import PoseFeaturePreset, SessionFeatureSummary
from abel.services.pose_processing_service import PoseProcessingService
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml

logger = logging.getLogger("abel")

# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

DEFAULT_PRESETS: list[PoseFeaturePreset] = [
    PoseFeaturePreset(
        preset_id="standard",
        name="Standard — 0.5 s window, 0.5 s stride",
        window_duration_sec=0.5,
        stride_sec=0.5,
        source_fps=30.0,
        likelihood_threshold=0.2,
        interpolate_dropouts=True,
        smoothing_window=3,
    ),
    PoseFeaturePreset(
        preset_id="long_window",
        name="Long Window — 1 s window, 0.5 s stride",
        window_duration_sec=1.0,
        stride_sec=0.5,
        source_fps=30.0,
        likelihood_threshold=0.2,
        interpolate_dropouts=True,
        smoothing_window=5,
    ),
    PoseFeaturePreset(
        preset_id="high_res",
        name="High-Res — 0.25 s window, 0.5 s stride",
        window_duration_sec=0.25,
        stride_sec=0.5,
        source_fps=30.0,
        likelihood_threshold=0.2,
        interpolate_dropouts=True,
        smoothing_window=3,
    ),
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Feature vector layout — each window is described by these scalars.
# Order must match _build_feature_vector.
FEATURE_NAMES = [
    "speed_mean",          # mean units/second (mm/s when px/mm is provided)
    "speed_std",
    "speed_max",
    "disp_mean",           # mean per-frame displacement (mm when px/mm is provided)
    "disp_std",
    "axis_cos_mean",       # mean cos(body-axis angle) — rotation-invariant representation
    "axis_sin_mean",
    "axis_angle_std",      # angular variability
    "likelihood_mean",     # average pose confidence (quality proxy)
]


@dataclass
class PoseFeatureConfig:
    """Job spec for one session."""
    session_id: str
    pose_path: Path
    preset: PoseFeaturePreset


@dataclass
class PoseFeatureResult:
    """Outcome for one session."""
    session_id: str
    n_frames: int = 0
    n_windows: int = 0
    body_parts: list[str] = field(default_factory=list)
    feature_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    success: bool = False


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class PoseFeaturesService:
    """Extracts kinematic window features from DLC pose files.

    No video is decoded.  Outputs compressed NumPy .npz files
    (one per session) under derived/pose_features/.
    """

    def __init__(self) -> None:
        self._pose_service = PoseProcessingService()
        self._project_root: Path | None = None

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root

    def _keypoint_aliases(self) -> dict[str, str] | None:
        """Project-level ``{original: new}`` body-part rename map, or None.

        Written by Data Import (Keypoint Mapping / Rename Body Parts) and applied
        on pose load so the kinematic .npz windows use the project's chosen
        names, consistent with the parquet feature pipeline.
        """
        if not self._project_root:
            return None
        data = read_json(self._project_root / "config" / "keypoint_aliases.json", {})
        aliases = {str(k): str(v) for k, v in data.items() if str(k) and str(v)}
        return aliases or None

    def _pixels_per_mm_for_session(self, session_id: str) -> float | None:
        if not self._project_root:
            return None
        manifest_path = self._project_root / "derived" / "review_tables" / "import_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            raw = read_json(manifest_path, {})
        except Exception:
            return None

        sessions = raw.get("linked_sessions", [])
        videos = {str(v.get("asset_id", "")): v for v in raw.get("videos", [])}
        for sess in sessions:
            sid = str(sess.get("session_id", "")).strip()
            if sid != str(session_id):
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

    # ------------------------------------------------------------------
    # Preset management (mirrors BehaviorService / SeedService pattern)
    # ------------------------------------------------------------------

    @property
    def default_presets(self) -> list[PoseFeaturePreset]:
        return list(DEFAULT_PRESETS)

    def load_project_presets(self) -> list[PoseFeaturePreset]:
        if not self._project_root:
            return list(DEFAULT_PRESETS)
        path = self._project_root / "config" / "pose_features.yaml"
        raw = read_yaml(path, {})
        custom: list[PoseFeaturePreset] = []
        for item in raw.get("presets", []):
            try:
                custom.append(PoseFeaturePreset.model_validate(item))
            except Exception:
                pass
        custom_ids = {p.preset_id for p in custom}
        merged = list(custom)
        for p in DEFAULT_PRESETS:
            if p.preset_id not in custom_ids:
                merged.append(p)
        return merged

    def save_project_preset(self, preset: PoseFeaturePreset) -> None:
        if not self._project_root:
            return
        path = self._project_root / "config" / "pose_features.yaml"
        raw = read_yaml(path, {})
        presets = [p for p in raw.get("presets", []) if p.get("preset_id") != preset.preset_id]
        presets.append(preset.model_dump(mode="json"))
        write_yaml(path, {**raw, "presets": presets})

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(
        self,
        config: PoseFeatureConfig,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> PoseFeatureResult:
        """Clean pose, compute kinematic windows, save .npz.  No video touched."""
        result = PoseFeatureResult(session_id=config.session_id)
        preset = config.preset

        # --- Load (applying any project keypoint renames) ---
        try:
            pose = self._pose_service.load(
                config.pose_path, keypoint_aliases=self._keypoint_aliases()
            )
        except Exception as exc:
            result.warnings.append(f"Failed to load pose: {exc}")
            return result

        # --- Clean ---
        pose = self._pose_service.clean_pose(
            pose,
            likelihood_threshold=preset.likelihood_threshold,
            interpolate=preset.interpolate_dropouts,
            smoothing_window=preset.smoothing_window,
        )

        result.n_frames = pose.n_frames
        result.body_parts = list(pose.body_parts)
        fps = preset.source_fps

        px_per_mm = self._pixels_per_mm_for_session(config.session_id)
        if px_per_mm is None:
            result.warnings.append(
                "Pixels/mm not set for this session in Data Import; "
                "kinematic distance/speed features remain in pixel units."
            )

        # --- Compute per-frame kinematics ---
        if px_per_mm is not None:
            centroid_x = np.asarray(pose.centroid_x, dtype=float) / float(px_per_mm)
            centroid_y = np.asarray(pose.centroid_y, dtype=float) / float(px_per_mm)
        else:
            centroid_x = np.asarray(pose.centroid_x, dtype=float)
            centroid_y = np.asarray(pose.centroid_y, dtype=float)

        speed = self._pose_service.compute_speed(centroid_x, centroid_y, fps)
        try:
            axis_angle = self._pose_service.compute_body_axis_angle(pose)
        except Exception:
            axis_angle = np.zeros(pose.n_frames)

        dx = np.diff(centroid_x, prepend=centroid_x[0])
        dy = np.diff(centroid_y, prepend=centroid_y[0])
        disp = np.sqrt(dx ** 2 + dy ** 2)

        mean_likelihood = np.asarray(pose.likelihood.mean(axis=1), dtype=float)

        # --- Sliding windows ---
        win_frames = max(1, int(preset.window_duration_sec * fps))
        stride_frames = max(1, int(preset.stride_sec * fps))
        starts = list(range(0, pose.n_frames - win_frames + 1, stride_frames))

        if not starts:
            result.warnings.append(
                f"Recording too short for {preset.window_duration_sec}s windows "
                f"({pose.n_frames} frames at {fps} fps)."
            )
            return result

        n_windows = len(starts)
        features = np.zeros((n_windows, len(FEATURE_NAMES)), dtype=np.float32)
        window_frames = np.zeros((n_windows, 2), dtype=np.int32)

        for i, sf in enumerate(starts):
            if cancel_flag and cancel_flag[0]:
                result.warnings.append("Cancelled by user.")
                break
            ef = sf + win_frames
            s = speed[sf:ef]
            d = disp[sf:ef]
            a = axis_angle[sf:ef]
            lk = mean_likelihood[sf:ef]

            features[i] = [
                float(s.mean()), float(s.std()), float(s.max()),
                float(d.mean()), float(d.std()),
                float(np.cos(a).mean()), float(np.sin(a).mean()), float(a.std()),
                float(lk.mean()),
            ]
            window_frames[i] = [sf, ef]

            if progress_callback:
                progress_callback(i + 1, n_windows)

        result.n_windows = n_windows

        # --- Save .npz ---
        if self._project_root:
            out_dir = self._project_root / "derived" / "pose_features"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{config.session_id}.npz"
            np.savez_compressed(
                out_path,
                features=features,
                window_frames=window_frames,
                feature_names=np.array(FEATURE_NAMES),
                body_parts=np.array(pose.body_parts),
            )
            result.feature_path = out_path
            summary = SessionFeatureSummary(
                session_id=config.session_id,
                n_frames=pose.n_frames,
                n_windows=n_windows,
                body_parts=pose.body_parts,
                fps=fps,
                feature_path=str(out_path.relative_to(self._project_root)),
                warnings=result.warnings,
            )
            self._save_session_summary(summary)
            logger.info(
                "Pose features saved: %s — %d frames, %d windows, %d body parts",
                config.session_id, pose.n_frames, n_windows, len(pose.body_parts),
            )

        result.success = True
        return result

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def load_features(self, session_id: str) -> dict | None:
        """Load the .npz feature file for a session.  Returns None if missing."""
        if not self._project_root:
            return None
        path = self._project_root / "derived" / "pose_features" / f"{session_id}.npz"
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        except Exception as exc:
            logger.warning("Failed to load features for %s: %s", session_id, exc)
            return None

    # ------------------------------------------------------------------
    # Session summary persistence
    # ------------------------------------------------------------------

    def _summaries_path(self) -> Path:
        assert self._project_root is not None
        return self._project_root / "derived" / "pose_features" / "summaries.json"

    def _save_session_summary(self, summary: SessionFeatureSummary) -> None:
        path = self._summaries_path()
        raw = read_json(path, {"summaries": []})
        summaries = [s for s in raw["summaries"] if s.get("session_id") != summary.session_id]
        summaries.append(summary.model_dump(mode="json"))
        write_json(path, {"summaries": summaries})

    def load_all_summaries(self) -> list[SessionFeatureSummary]:
        if not self._project_root:
            return []
        try:
            raw = read_json(self._summaries_path(), {"summaries": []})
        except Exception:
            return []
        results = []
        for item in raw.get("summaries", []):
            try:
                results.append(SessionFeatureSummary.model_validate(item))
            except Exception:
                pass
        return results
