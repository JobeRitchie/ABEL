"""DLC pose data loading, cleaning, and kinematic feature extraction.

Handles both CSV (multi-index header) and H5 formats produced by DeepLabCut.
All heavy computation uses numpy/pandas which are Tier-1 dependencies.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from abel.models.schemas import InvariantFeatureConfig, PoseSmoothingSettings

logger = logging.getLogger("abel")


# ---------------------------------------------------------------------------
# Body-part name normalisation
# ---------------------------------------------------------------------------

_KNOWN_BODYPART_TOKENS: list[str] = [
    # directional / positional
    "left", "right", "fore", "hind", "front", "back", "mid",
    "upper", "lower", "dorsal", "ventral",
    # body parts (longest first within each group helps greedy matching)
    "shoulder", "elbow", "center", "centre", "snout",
    "muzzle", "rostrum", "wrist", "ankle", "spine", "shank",
    "trunk", "belly", "flank", "digit", "mouth", "rump",
    "limb", "nose", "head", "tail", "base", "body", "ear",
    "paw", "hip", "leg", "jaw", "eye", "tip", "neck",
    "hand", "foot", "knee", "toe",
]
# Sort longest-first so greedy matching prefers longer tokens.
_KNOWN_BODYPART_TOKENS.sort(key=len, reverse=True)


def _split_known_tokens(segment: str) -> str:
    """Greedily split a lowercase segment into known tokens separated by ``_``.

    Unrecognised remainders are kept as-is so novel body-part names pass
    through unchanged.
    """
    parts: list[str] = []
    remaining = segment
    while remaining:
        matched = False
        for token in _KNOWN_BODYPART_TOKENS:
            if remaining.startswith(token):
                parts.append(token)
                remaining = remaining[len(token):]
                matched = True
                break
        if not matched:
            # No known token matches — keep whatever is left as one chunk.
            parts.append(remaining)
            break
    return "_".join(parts)


def normalize_bodypart_name(name: str) -> str:
    """Map a DLC body-part name to a canonical lowercase, underscore-separated form.

    Handles camelCase (``LeftEar``), concatenated words (``leftear``),
    hyphens, spaces, and mixed digit boundaries (``spine1``) so that
    different DLC labelling conventions resolve to the same canonical key.

    Examples
    --------
    >>> normalize_bodypart_name("LeftEar")
    'left_ear'
    >>> normalize_bodypart_name("leftear")
    'left_ear'
    >>> normalize_bodypart_name("left_ear")
    'left_ear'
    >>> normalize_bodypart_name("TailBase")
    'tail_base'
    >>> normalize_bodypart_name("spine1")
    'spine_1'
    """
    # 1. Insert underscore at camelCase boundaries  (e.g. LeftEar → Left_Ear)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    # 2. Insert underscore between letters and digits  (spine1 → spine_1)
    s = re.sub(r"(?<=[a-zA-Z])(?=[0-9])", "_", s)
    s = re.sub(r"(?<=[0-9])(?=[a-zA-Z])", "_", s)
    # 3. Lowercase
    s = s.lower()
    # 4. Replace hyphens and spaces with underscores
    s = re.sub(r"[\s\-]+", "_", s)
    # 5. Collapse repeated underscores, strip leading/trailing
    s = re.sub(r"_+", "_", s).strip("_")
    # 6. For each segment, try to split concatenated known tokens
    #    (e.g. "leftear" → "left_ear", "forepaw" → "fore_paw")
    segments = s.split("_")
    expanded: list[str] = []
    for seg in segments:
        if len(seg) > 1 and not seg.isdigit():
            expanded.append(_split_known_tokens(seg))
        else:
            expanded.append(seg)
    return "_".join(expanded)


def _find_bp_col(
    norm_to_col: dict[str, str],
    bp_tokens: dict[str, frozenset[str]],
    candidates: list[str],
) -> str | None:
    """Return the first body-part column that matches any candidate name.

    Matching strategy (tried for each candidate in order):
    1. Exact match on the normalized name.
    2. Token-set subset match: all underscore-separated tokens in the
       candidate appear in the body-part's token set.  This makes
       'body_center' match 'center_body', 'tailbase' match 'tail_base', etc.

    Returns *None* if no candidate matches any tracked body part.
    """
    for cand in candidates:
        norm_cand = normalize_bodypart_name(cand)
        col = norm_to_col.get(norm_cand)
        if col:
            return col
        cand_toks = frozenset(re.split(r"[_\-\s]+", norm_cand))
        for norm_bp, bp_tok in bp_tokens.items():
            if cand_toks <= bp_tok:
                return norm_to_col[norm_bp]
    return None


class PoseData(NamedTuple):
    """Cleaned pose data with per-frame kinematic summary."""

    body_parts: list[str]
    x: pd.DataFrame           # (n_frames, n_parts)  — cleaned x coords
    y: pd.DataFrame           # (n_frames, n_parts)  — cleaned y coords
    likelihood: pd.DataFrame  # (n_frames, n_parts)  — raw likelihoods
    centroid_x: np.ndarray    # (n_frames,)
    centroid_y: np.ndarray    # (n_frames,)
    n_frames: int


class PoseProcessingService:
    """Loads and preprocesses DLC tracking files."""

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        path: Path,
        keypoint_aliases: "dict[str, str] | None" = None,
    ) -> PoseData:
        """Dispatch to the correct loader based on file extension.

        ``keypoint_aliases`` renames body parts immediately after loading
        (``{source_name_in_file: target_name}``).  Used by Direct Use to map a
        new project's keypoints onto the names a trained model expects.
        """
        suffix = path.suffix.lower()
        if suffix == ".csv":
            pose = self.load_csv(path)
        elif suffix in (".h5", ".hdf5"):
            pose = self.load_h5(path)
        else:
            raise ValueError(f"Unsupported pose format: {suffix}")
        if keypoint_aliases:
            pose = self._apply_keypoint_aliases(pose, keypoint_aliases)
        return pose

    @staticmethod
    def _apply_keypoint_aliases(
        pose: PoseData, aliases: dict[str, str],
    ) -> PoseData:
        """Rename body parts in a PoseData using ``{old: new}`` aliases.

        Matching is on normalized names so the caller can pass either the raw
        or normalized form.  Identity/empty entries and renames whose target
        already exists are skipped (the existing keypoint wins) to avoid
        silently dropping data through a collision.
        """
        norm_alias = {
            normalize_bodypart_name(k): normalize_bodypart_name(v)
            for k, v in aliases.items()
            if str(k).strip() and str(v).strip()
        }
        if not norm_alias:
            return pose

        existing = set(pose.body_parts)
        rename: dict[str, str] = {}
        for part in pose.body_parts:
            target = norm_alias.get(part)
            if not target or target == part:
                continue
            if target in existing and target not in rename.values():
                logger.warning(
                    "Keypoint alias '%s'->'%s' skipped: target already present.",
                    part, target,
                )
                continue
            rename[part] = target

        if not rename:
            return pose

        new_body_parts = [rename.get(p, p) for p in pose.body_parts]
        return pose._replace(
            body_parts=new_body_parts,
            x=pose.x.rename(columns=rename),
            y=pose.y.rename(columns=rename),
            likelihood=pose.likelihood.rename(columns=rename),
        )

    def load_csv(self, path: Path) -> PoseData:
        """Load a DLC CSV with 3-row MultiIndex header (scorer / bodypart / coord)."""
        raw = pd.read_csv(path, header=[0, 1, 2], index_col=0)
        assert isinstance(raw, pd.DataFrame), "pd.read_csv returned non-DataFrame"
        if raw.columns.nlevels == 3:
            raw.columns = raw.columns.droplevel(0)  # type: ignore[assignment]
        return self._parse_pose_df(raw, source=path)

    def load_h5(self, path: Path) -> PoseData:
        """Load a DLC H5 file."""
        try:
            raw_h5 = pd.read_hdf(str(path), key="df_with_missing")
        except Exception:
            raw_h5 = pd.read_hdf(str(path))
        assert isinstance(raw_h5, pd.DataFrame), "pd.read_hdf returned non-DataFrame"
        if raw_h5.columns.nlevels == 3:
            raw_h5.columns = raw_h5.columns.droplevel(0)  # type: ignore[assignment]
        return self._parse_pose_df(raw_h5, source=path)

    def load_and_clean(
        self,
        path: Path,
        settings: "PoseSmoothingSettings | None" = None,
        keypoint_aliases: "dict[str, str] | None" = None,
    ) -> PoseData:
        """Load a pose file and apply temporal smoothing with the given settings.

        This is the preferred entry point for all downstream services.  Pass
        ``manifest.smoothing_settings`` to apply the project-level smoothing
        that was configured at import time.  ``keypoint_aliases`` renames body
        parts on load (see :meth:`load`).
        """
        from abel.models.schemas import PoseSmoothingSettings as _S
        s = settings or _S()
        pose = self.load(path, keypoint_aliases=keypoint_aliases)
        return self.clean_pose(
            pose,
            likelihood_threshold=s.likelihood_threshold,
            interpolate=s.interpolate_dropouts,
            interpolate_max_gap=s.interpolate_max_gap,
            smoothing_window=s.smoothing_window,
        )

    def _parse_pose_df(self, df: pd.DataFrame, source: Path | None = None) -> PoseData:
        """Convert a (bodypart, coord) MultiIndex DataFrame into PoseData."""
        raw_parts = list(df.columns.get_level_values(0).unique())
        x_dict, y_dict, l_dict = {}, {}, {}

        seen_norms: dict[str, str] = {}  # normalized → original (collision check)
        for part in raw_parts:
            norm = normalize_bodypart_name(part)
            if norm in seen_norms and seen_norms[norm] != part:
                logger.warning(
                    "Body parts '%s' and '%s' both normalise to '%s' in %s; "
                    "keeping '%s'",
                    seen_norms[norm], part, norm, source, part,
                )
            seen_norms[norm] = part
            try:
                x_dict[norm] = df[part]["x"].astype(float).values
                y_dict[norm] = df[part]["y"].astype(float).values
                l_dict[norm] = df[part]["likelihood"].astype(float).values
            except Exception as exc:
                logger.warning("Skipping body part %s from %s: %s", part, source, exc)

        if not x_dict:
            raise ValueError(f"Could not extract any body parts from pose file: {source}")

        n = len(df)
        x_df = pd.DataFrame(x_dict, index=range(n), dtype=float)
        y_df = pd.DataFrame(y_dict, index=range(n), dtype=float)
        l_df = pd.DataFrame(l_dict, index=range(n), dtype=float)

        cx, cy = self._compute_centroid(x_df, y_df, l_df)
        return PoseData(
            body_parts=list(x_dict.keys()),
            x=x_df, y=y_df, likelihood=l_df,
            centroid_x=cx, centroid_y=cy,
            n_frames=n,
        )

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean_pose(
        self,
        pose: PoseData,
        likelihood_threshold: float = 0.2,
        interpolate: bool = True,
        interpolate_max_gap: int = 10,
        smoothing_window: int = 5,
    ) -> PoseData:
        """Apply likelihood masking, gap interpolation, and temporal smoothing."""
        x = pose.x.copy()
        y = pose.y.copy()

        # 1. Mask low-likelihood detections
        mask = pose.likelihood < likelihood_threshold
        x[mask] = np.nan
        y[mask] = np.nan

        # 2. Interpolate short gaps
        if interpolate and interpolate_max_gap > 0:
            x = x.interpolate(method="linear", limit=interpolate_max_gap)
            y = y.interpolate(method="linear", limit=interpolate_max_gap)

        # 3. Smooth
        if smoothing_window > 1:
            x = x.rolling(window=smoothing_window, center=True, min_periods=1).mean()
            y = y.rolling(window=smoothing_window, center=True, min_periods=1).mean()

        # 4. Fill any remaining NaN with forward/back fill then zero
        x = x.ffill().bfill().fillna(0.0)
        y = y.ffill().bfill().fillna(0.0)

        cx, cy = self._compute_centroid(x, y, pose.likelihood, threshold=likelihood_threshold)
        return PoseData(
            body_parts=pose.body_parts,
            x=x, y=y, likelihood=pose.likelihood,
            centroid_x=cx, centroid_y=cy,
            n_frames=pose.n_frames,
        )

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    @staticmethod
    def _keypoint_xy(pose: PoseData, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Resolve a body-part by name and return its (x, y) arrays.

        Matching strategy (tried in order for each candidate):
        1. Exact match on the normalized name.
        2. Token-set match: every underscore-separated token in the
           candidate appears somewhere in the body-part's tokens.

        If no candidate matches any body part, returns all-NaN so that
        downstream features (forepaw_speed, head_pitch, etc.) propagate NaN
        rather than a spurious constant zero.  NaN-valued columns are handled
        correctly by XGBoost (native missing-value path) and are flagged by
        the feature audit as truly absent rather than dead-constant.
        """
        import re as _re  # noqa: PLC0415

        norm_to_col = {normalize_bodypart_name(bp): bp for bp in pose.body_parts}

        # Build token sets once for all actual body parts
        bp_tokens: dict[str, frozenset[str]] = {
            norm: frozenset(_re.split(r"[_\-\s]+", norm))
            for norm in norm_to_col
        }

        for key in candidates:
            norm_key = normalize_bodypart_name(key)
            # Exact match first
            col = norm_to_col.get(norm_key)
            if col is not None:
                return np.asarray(pose.x[col], dtype=float), np.asarray(pose.y[col], dtype=float)
            # Token-set match: all tokens in the candidate present in a body part
            cand_tokens = set(_re.split(r"[_\-\s]+", norm_key))
            for norm_bp, bp_tok in bp_tokens.items():
                if cand_tokens <= bp_tok:  # subset check
                    col = norm_to_col[norm_bp]
                    return np.asarray(pose.x[col], dtype=float), np.asarray(pose.y[col], dtype=float)

        return np.full(pose.n_frames, np.nan, dtype=float), np.full(pose.n_frames, np.nan, dtype=float)

    @staticmethod
    def _finite_diff(arr: np.ndarray, fps: float) -> np.ndarray:
        return np.diff(arr, prepend=arr[0]) * fps

    @staticmethod
    def _window_autocorr_peak(arr: np.ndarray, window: int) -> np.ndarray:
        n = len(arr)
        if not np.any(np.isfinite(arr)):
            return np.full(n, np.nan, dtype=float)
        out = np.zeros(n, dtype=float)
        half = max(1, window // 2)
        for i in range(n):
            s = max(0, i - half)
            e = min(n, i + half + 1)
            x = arr[s:e]
            if len(x) < 4:
                continue
            x = x - x.mean()
            if np.var(x) <= 1e-9:
                continue
            ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
            if len(ac) > 1 and ac[0] > 1e-9:
                out[i] = float(np.max(ac[1:]) / ac[0])
        return out

    @staticmethod
    def _window_frequency(arr: np.ndarray, fps: float, window: int) -> np.ndarray:
        n = len(arr)
        if not np.any(np.isfinite(arr)):
            return np.full(n, np.nan, dtype=float)
        out = np.zeros(n, dtype=float)
        half = max(1, window // 2)
        for i in range(n):
            s = max(0, i - half)
            e = min(n, i + half + 1)
            x = arr[s:e]
            if len(x) < 8:
                continue
            x = x - x.mean()
            fft = np.fft.rfft(x)
            freqs = np.fft.rfftfreq(len(x), d=1.0 / fps)
            if len(freqs) < 2:
                continue
            idx = int(np.argmax(np.abs(fft[1:])) + 1)
            out[i] = float(freqs[idx])
        return out

    def compute_frame_pose_features(
        self,
        pose: PoseData,
        fps: float,
        animal_id: str,
        session_id: str,
        video_id: str,
        rhythmicity_window_sec: float = 1.0,
        invariant_config: "InvariantFeatureConfig | None" = None,
    ) -> pd.DataFrame:
        """Compute framewise kinematics and behavior-relevant rhythmicity descriptors.

        When *invariant_config* is provided (or defaulted), additional robustness
        features are computed alongside the existing absolute-frame kinematics:

        * Egocentric forward/lateral velocities per keypoint (body-centered frame)
        * Body-length-normalized pairwise inter-keypoint distances
        * Head direction angle and angular velocity
        * Joint angles from keypoint triplets
        """
        from abel.models.schemas import InvariantFeatureConfig as _IFC  # noqa: PLC0415
        cfg = invariant_config if invariant_config is not None else _IFC()

        n = pose.n_frames
        frame = np.arange(n, dtype=int)
        dt_window = max(4, int(round(rhythmicity_window_sec * fps)))

        paw_l_x, paw_l_y = self._keypoint_xy(pose, [
            "paw_L", "left_paw", "forepaw_left", "lateral_left", "front_left",
            "frontleg_left", "front_leg_left",
        ])
        paw_r_x, paw_r_y = self._keypoint_xy(pose, [
            "paw_R", "right_paw", "forepaw_right", "lateral_right", "front_right",
            "frontleg_right", "front_leg_right",
        ])
        nose_x, nose_y = self._keypoint_xy(pose, ["nose", "snout", "head"])
        ear_l_x, ear_l_y = self._keypoint_xy(pose, [
            "left_ear", "ear_left", "ear_l", "earL",
        ])
        ear_r_x, ear_r_y = self._keypoint_xy(pose, [
            "right_ear", "ear_right", "ear_r", "earR",
        ])

        paw_cx = (paw_l_x + paw_r_x) / 2.0
        paw_cy = (paw_l_y + paw_r_y) / 2.0

        # ── Body-axis angle (used both for existing features and egocentric transform) ──
        body_orientation = self.compute_body_axis_angle(pose)
        cos_orient = np.cos(body_orientation)
        sin_orient = np.sin(body_orientation)

        # ── Body length estimate (nose-to-tail per frame) ──────────────────
        body_length, body_length_pair = self._compute_body_length_with_pair(pose)
        safe_body_length = np.where(body_length > 1e-3, body_length, np.nan)
        # The distance pair that *defines* body length normalizes to a constant
        # 1.0, so its `_norm` column is dead by construction — skip it below.
        body_length_pair_set = frozenset(body_length_pair) if body_length_pair else frozenset()

        # ── Per-keypoint kinematics ─────────────────────────────────────────
        kp_cols: dict[str, np.ndarray] = {}
        for key in pose.body_parts:
            x = np.asarray(pose.x[key], dtype=float)
            y = np.asarray(pose.y[key], dtype=float)
            vx = self._finite_diff(x, fps)
            vy = self._finite_diff(y, fps)
            speed = np.sqrt(vx * vx + vy * vy)
            ax = self._finite_diff(vx, fps)
            ay = self._finite_diff(vy, fps)
            acc = np.sqrt(ax * ax + ay * ay)
            jerk = self._finite_diff(acc, fps)
            safe = key.replace(" ", "_")
            kp_cols[f"{safe}_velocity_x"] = vx
            kp_cols[f"{safe}_velocity_y"] = vy
            kp_cols[f"{safe}_speed"] = speed
            kp_cols[f"{safe}_acceleration"] = acc
            kp_cols[f"{safe}_jerk"] = jerk

            # ── Egocentric forward/lateral velocity (body-centred frame) ──
            if cfg.enable_egocentric_kinematics:
                # Rotate world-frame velocity into body frame:
                # forward = vx*cos(θ) + vy*sin(θ)
                # lateral = -vx*sin(θ) + vy*cos(θ)
                fwd_vel = vx * cos_orient + vy * sin_orient
                lat_vel = -vx * sin_orient + vy * cos_orient
                kp_cols[f"{safe}_forward_velocity"] = fwd_vel
                kp_cols[f"{safe}_lateral_velocity"] = lat_vel

        paw_vx = self._finite_diff(paw_cx, fps)
        paw_vy = self._finite_diff(paw_cy, fps)
        forepaw_speed = np.sqrt(paw_vx * paw_vx + paw_vy * paw_vy)
        nose_vx = self._finite_diff(nose_x, fps)
        nose_vy = self._finite_diff(nose_y, fps)
        nose_velocity = np.sqrt(nose_vx * nose_vx + nose_vy * nose_vy)

        # ── Head pitch: ear-midpoint → nose direction ──────────────────────
        # Priority: (1) midpoint of both ears, (2) whichever single ear is
        # present, (3) forepaw centroid, (4) NaN when nothing is available.
        # np.nanmean over a 2-row stack automatically handles the single-ear
        # case without extra branching.
        ear_ref_x = np.nanmean(np.stack([ear_l_x, ear_r_x], axis=0), axis=0)
        ear_ref_y = np.nanmean(np.stack([ear_l_y, ear_r_y], axis=0), axis=0)
        head_ref_x = np.where(np.isfinite(ear_ref_x), ear_ref_x, paw_cx)
        head_ref_y = np.where(np.isfinite(ear_ref_y), ear_ref_y, paw_cy)
        head_pitch = np.arctan2(nose_y - head_ref_y, nose_x - head_ref_x)
        cent_vx = self._finite_diff(pose.centroid_x, fps)
        cent_vy = self._finite_diff(pose.centroid_y, fps)
        centroid_velocity = np.sqrt(cent_vx * cent_vx + cent_vy * cent_vy)

        forepaw_oscillation_power = np.square(forepaw_speed - np.mean(forepaw_speed))
        forepaw_autocorr_peak = self._window_autocorr_peak(forepaw_speed, dt_window)
        paw_scrape_frequency = self._window_frequency(forepaw_speed, fps=fps, window=dt_window)
        oscillation_energy = np.convolve(forepaw_oscillation_power, np.ones(dt_window), mode="same") / float(dt_window)

        # ── Nose kinematics (parallel to forepaw features) ───────────────
        # Useful for overhead-camera setups where paw keypoints are absent
        # but nose movement against the substrate is still informative.
        nose_oscillation_power = np.square(nose_velocity - np.mean(nose_velocity))
        nose_autocorr_peak = self._window_autocorr_peak(nose_velocity, dt_window)
        nose_movement_frequency = self._window_frequency(nose_velocity, fps=fps, window=dt_window)
        nose_oscillation_energy = np.convolve(nose_oscillation_power, np.ones(dt_window), mode="same") / float(dt_window)

        base = {
            "frame": frame,
            "animal_id": animal_id,
            "session_id": session_id,
            "video_id": video_id,
            "forepaw_speed": forepaw_speed,
            "forepaw_vertical_velocity": paw_vy,
            "forepaw_oscillation_power": forepaw_oscillation_power,
            "nose_velocity": nose_velocity,
            "nose_vertical_velocity": nose_vy,
            "nose_oscillation_power": nose_oscillation_power,
            "nose_autocorr_peak": nose_autocorr_peak,
            "nose_movement_frequency": nose_movement_frequency,
            "nose_oscillation_energy": nose_oscillation_energy,
            "head_pitch": head_pitch,
            "body_orientation": body_orientation,
            "centroid_velocity": centroid_velocity,
            "forepaw_autocorr_peak": forepaw_autocorr_peak,
            "forepaw_movement_frequency": paw_scrape_frequency,
            "oscillation_energy": oscillation_energy,
        }
        base.update(kp_cols)

        # ── Body length (scale-invariant reference) ─────────────────────
        if cfg.enable_body_length_normalization:
            base["body_length_px"] = body_length

        # ── Pairwise inter-keypoint distances (normalized by body length) ──
        if cfg.enable_relative_geometry:
            parts = pose.body_parts
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    p_i = parts[i].replace(" ", "_")
                    p_j = parts[j].replace(" ", "_")
                    xi = np.asarray(pose.x[parts[i]], dtype=float)
                    yi = np.asarray(pose.y[parts[i]], dtype=float)
                    xj = np.asarray(pose.x[parts[j]], dtype=float)
                    yj = np.asarray(pose.y[parts[j]], dtype=float)
                    dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    # Canonical (sorted) pair name so the column is independent
                    # of the DLC keypoint column order.  Distance is symmetric,
                    # so two projects with the same keypoints listed in a
                    # different order would otherwise produce dist_A_to_B vs
                    # dist_B_to_A and become incompatible for cross-project
                    # (Direct Use) model reuse.
                    p_a, p_b = sorted((p_i, p_j))
                    col_name = f"dist_{p_a}_to_{p_b}"
                    base[col_name] = dist
                    # Skip the normalized variant for the body-length-defining
                    # pair: dist / body_length ≡ 1.0, a constant dead feature.
                    if (
                        cfg.enable_body_length_normalization
                        and frozenset((parts[i], parts[j])) != body_length_pair_set
                    ):
                        base[f"{col_name}_norm"] = dist / safe_body_length

        # ── Head direction features ─────────────────────────────────────
        if cfg.enable_head_direction:
            head_dir_cols = self._compute_head_direction_features(pose, fps)
            base.update(head_dir_cols)

        # ── Joint angles ─────────────────────────────────────────────────
        if cfg.enable_joint_angles:
            angle_cols = self._compute_joint_angle_features(pose)
            base.update(angle_cols)

        # ── Spine curvature ───────────────────────────────────────────────
        if cfg.enable_spine_curvature:
            curvature_cols = self._compute_spine_curvature_features(pose)
            base.update(curvature_cols)

        return pd.DataFrame(base)

    def extract_and_save_frame_pose_features(
        self,
        project_root: Path,
        pose_path: Path,
        fps: float,
        animal_id: str,
        session_id: str,
        video_id: str,
        invariant_config: "InvariantFeatureConfig | None" = None,
        keypoint_aliases: "dict[str, str] | None" = None,
    ) -> pd.DataFrame:
        pose = self.load_and_clean(pose_path, keypoint_aliases=keypoint_aliases)
        df = self.compute_frame_pose_features(
            pose=pose,
            fps=fps,
            animal_id=animal_id,
            session_id=session_id,
            video_id=video_id,
            invariant_config=invariant_config,
        )
        out_dir = project_root / "derived" / "pose_features"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write directly to a per-session parquet file.  The old pattern
        # (global lock + read-modify-write on a shared monolithic file)
        # serialised all parallel workers and forced an O(N²) write volume
        # as the file grew with each session — the last sessions had to
        # read/write the entire combined file.  Per-session files are
        # independent, so all workers can write concurrently with no lock.
        # Call consolidate_session_files() once after all sessions have
        # been processed to rebuild the canonical frame_pose.parquet.
        session_out = out_dir / "sessions" / f"{session_id}.parquet"
        session_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(session_out, index=False)
        return df

    @staticmethod
    def consolidate_session_files(project_root: Path) -> Path | None:
        """Merge per-session parquet files into the canonical frame_pose.parquet.

        Sessions already in the monolithic file that were *not* updated this
        run are preserved.  Per-session files are authoritative for any
        session_id they contain.

        Returns the output path on success, or None if there is nothing to
        consolidate.
        """
        sessions_dir = project_root / "derived" / "pose_features" / "sessions"
        out_path = project_root / "derived" / "pose_features" / "frame_pose.parquet"
        per_session_files = sorted(sessions_dir.glob("*.parquet")) if sessions_dir.exists() else []
        if not per_session_files:
            return out_path if out_path.exists() else None

        new_session_ids = {f.stem for f in per_session_files}
        parts: list[pd.DataFrame] = []

        # Preserve legacy sessions that were not re-extracted this run.
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

    @staticmethod
    def _compute_centroid(
        x: pd.DataFrame,
        y: pd.DataFrame,
        likelihood: pd.DataFrame,
        threshold: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean position of all body parts above the likelihood threshold."""
        if threshold > 0:
            mask = likelihood >= threshold
            cx = np.asarray(x.where(mask).mean(axis=1).fillna(x.mean(axis=1)), dtype=float)
            cy = np.asarray(y.where(mask).mean(axis=1).fillna(y.mean(axis=1)), dtype=float)
        else:
            cx = np.asarray(x.mean(axis=1), dtype=float)
            cy = np.asarray(y.mean(axis=1), dtype=float)
        return cx, cy

    @staticmethod
    def compute_speed(
        centroid_x: np.ndarray,
        centroid_y: np.ndarray,
        fps: float = 30.0,
    ) -> np.ndarray:
        """Per-frame displacement speed in pixels/second."""
        dx = np.diff(centroid_x, prepend=centroid_x[0])
        dy = np.diff(centroid_y, prepend=centroid_y[0])
        return np.sqrt(dx ** 2 + dy ** 2) * fps

    @staticmethod
    def compute_body_axis_angle(pose: PoseData) -> np.ndarray:
        """Angle (radians) of body axis per frame using available spine parts."""
        norm_to_col = {normalize_bodypart_name(bp): bp for bp in pose.body_parts}
        candidate_pairs = [
            ("tailbase", "nose"),
            ("tail_base", "nose"),
            ("spine1", "nose"),
            ("body_center", "nose"),
            ("center", "nose"),
            ("centre", "nose"),
        ]
        for tail_key, head_key in candidate_pairs:
            tail_col = norm_to_col.get(normalize_bodypart_name(tail_key))
            head_col = norm_to_col.get(normalize_bodypart_name(head_key))
            if tail_col is not None and head_col is not None:
                dx = np.asarray(pose.x[head_col], dtype=float) - np.asarray(pose.x[tail_col], dtype=float)
                dy = np.asarray(pose.y[head_col], dtype=float) - np.asarray(pose.y[tail_col], dtype=float)
                return np.arctan2(dy, dx)
        # Fallback: zero angle if spine parts unavailable
        return np.zeros(pose.n_frames)

    # ------------------------------------------------------------------
    # Robustness / invariance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_body_length(pose: PoseData) -> np.ndarray:
        """Estimate per-frame body length as the nose-to-tail-base distance.

        Falls back to the longest available spine pair if nose or tail_base are
        absent, and ultimately to all-NaN if no usable pair is found.

        Uses token-set matching so 'tailbase' matches 'tail_base', etc.
        """
        return PoseProcessingService._compute_body_length_with_pair(pose)[0]

    @staticmethod
    def _compute_body_length_with_pair(
        pose: PoseData,
    ) -> tuple[np.ndarray, tuple[str, str] | None]:
        """Body length array plus the (head_col, tail_col) pair that produced it.

        The pair is returned so callers can avoid degenerate self-normalization:
        the inter-keypoint distance for this exact pair, divided by the body
        length, is identically 1.0 and therefore a constant, information-free
        feature.  Returns ``(all-NaN array, None)`` when no usable pair exists.
        """
        norm_to_col = {normalize_bodypart_name(bp): bp for bp in pose.body_parts}
        bp_tokens: dict[str, frozenset[str]] = {
            norm: frozenset(re.split(r"[_\-\s]+", norm))
            for norm in norm_to_col
        }

        # Try head/tail pairs in order of anatomical distance
        candidate_pairs = [
            (["nose", "snout", "head"], ["tail_base", "tailbase", "tail"]),
            (["nose", "snout", "head"], ["tail_tip", "tail_end"]),
            (["nose", "snout", "head"], ["rump", "back"]),
            (["spine1", "back"], ["tail_base", "tailbase"]),
        ]
        for head_cands, tail_cands in candidate_pairs:
            hc = _find_bp_col(norm_to_col, bp_tokens, head_cands)
            tc = _find_bp_col(norm_to_col, bp_tokens, tail_cands)
            if hc is not None and tc is not None and hc != tc:
                dx = np.asarray(pose.x[hc], dtype=float) - np.asarray(pose.x[tc], dtype=float)
                dy = np.asarray(pose.y[hc], dtype=float) - np.asarray(pose.y[tc], dtype=float)
                bl = np.sqrt(dx ** 2 + dy ** 2)
                return np.where(bl > 1.0, bl, np.nan), (hc, tc)
        return np.full(pose.n_frames, np.nan, dtype=float), None

    def _compute_head_direction_features(
        self, pose: PoseData, fps: float
    ) -> dict[str, np.ndarray]:
        """Compute head direction angle and angular velocity.

        Uses ear midpoint → nose vector when ear keypoints are available;
        falls back to body axis when ears are absent.  Returns an empty dict
        if no usable keypoints are found.
        """
        norm_to_col = {normalize_bodypart_name(bp): bp for bp in pose.body_parts}

        # Try to find left and right ears
        ear_l_col = None
        ear_r_col = None
        for cand in ("ear_left", "left_ear", "ear_l", "earL"):
            c = norm_to_col.get(normalize_bodypart_name(cand))
            if c:
                ear_l_col = c
                break
        for cand in ("ear_right", "right_ear", "ear_r", "earR"):
            c = norm_to_col.get(normalize_bodypart_name(cand))
            if c:
                ear_r_col = c
                break

        nose_x, nose_y = self._keypoint_xy(pose, ["nose", "snout", "head"])
        n = pose.n_frames

        if ear_l_col is not None and ear_r_col is not None:
            # Head direction: ear-midpoint → nose
            ear_mx = (np.asarray(pose.x[ear_l_col], dtype=float) +
                      np.asarray(pose.x[ear_r_col], dtype=float)) / 2.0
            ear_my = (np.asarray(pose.y[ear_l_col], dtype=float) +
                      np.asarray(pose.y[ear_r_col], dtype=float)) / 2.0
            head_dir = np.arctan2(nose_y - ear_my, nose_x - ear_mx)
            # Ear spread: inter-ear distance (captures head posture / rearing)
            ear_spread = np.sqrt(
                (np.asarray(pose.x[ear_r_col], dtype=float) -
                 np.asarray(pose.x[ear_l_col], dtype=float)) ** 2 +
                (np.asarray(pose.y[ear_r_col], dtype=float) -
                 np.asarray(pose.y[ear_l_col], dtype=float)) ** 2
            )
            out: dict[str, np.ndarray] = {"ear_spread": ear_spread}
        else:
            # Fallback: use body axis angle as head direction proxy
            head_dir = self.compute_body_axis_angle(pose)
            out = {}

        # Angular velocity of head direction (unwrap to avoid 2π jumps)
        head_dir_unwrap = np.unwrap(head_dir)
        head_angular_velocity = self._finite_diff(head_dir_unwrap, fps)

        # Head forward/lateral velocity in head-direction frame
        head_cos = np.cos(head_dir)
        head_sin = np.sin(head_dir)
        nose_vx = self._finite_diff(nose_x, fps)
        nose_vy = self._finite_diff(nose_y, fps)
        head_forward_speed = nose_vx * head_cos + nose_vy * head_sin
        head_lateral_speed = -nose_vx * head_sin + nose_vy * head_cos

        out["head_direction_angle"] = head_dir
        out["head_angular_velocity"] = head_angular_velocity
        out["head_forward_speed"] = head_forward_speed
        out["head_lateral_speed"] = head_lateral_speed
        return out

    @staticmethod
    def _compute_joint_angle_features(pose: PoseData) -> dict[str, np.ndarray]:
        """Compute angles at anatomical joints from keypoint triplets.

        For each triplet (proximal, joint, distal), the angle at *joint* is
        the interior angle between the two limb vectors.  Returns only the
        triplets where all three keypoints are present.

        Uses token-set matching so 'center_body' matches 'body_center',
        'back' matches spine-like candidates, etc.
        """
        norm_to_col = {normalize_bodypart_name(bp): bp for bp in pose.body_parts}
        bp_tokens: dict[str, frozenset[str]] = {
            norm: frozenset(re.split(r"[_\-\s]+", norm))
            for norm in norm_to_col
        }

        def _get(candidates: list[str]) -> str | None:
            return _find_bp_col(norm_to_col, bp_tokens, candidates)

        def _angle_at_joint(
            ax: np.ndarray, ay: np.ndarray,
            bx: np.ndarray, by: np.ndarray,
            cx: np.ndarray, cy: np.ndarray,
        ) -> np.ndarray:
            """Angle at B between vectors BA and BC (radians)."""
            v1x, v1y = ax - bx, ay - by
            v2x, v2y = cx - bx, cy - by
            dot = v1x * v2x + v1y * v2y
            mag = np.sqrt(v1x ** 2 + v1y ** 2) * np.sqrt(v2x ** 2 + v2y ** 2)
            cos_angle = np.clip(dot / np.where(mag > 1e-9, mag, np.nan), -1.0, 1.0)
            return np.arccos(cos_angle)

        # ── Triplet definitions ─────────────────────────────────────────
        # Each tuple: (feature_name, proximal_candidates, joint_candidates, distal_candidates)
        # The angle is computed AT the joint keypoint.
        # Candidates use token-set matching: 'body_center' will match 'center_body', etc.
        #
        # Models covered:
        #   TMT-style:   nose, ear_left, ear_right, back, tailbase,
        #                frontleg_left/right, backleg_left/right
        #   NOvel-style: nose, left_ear, right_ear, left_body, center_body,
        #                right_body, tail_base
        #   EPM-style:   nose, ear_left, ear_right, body_left, body_mid,
        #                body_right, tail_base
        _SPINE_JOINT = [
            "body_center", "center_body", "back", "body_mid",
            "center", "spine2", "mid_back", "spine_mid",
        ]
        _BODY_CENTER_DISTAL = [
            "body_center", "center_body", "body_mid", "back",
            "center", "spine1",
        ]

        triplet_defs: list[tuple[str, list[str], list[str], list[str]]] = [
            # Spine bending: head-end → body-center → tail-end
            ("spine_flexion",
             ["nose", "snout", "head"],
             _SPINE_JOINT,
             ["tail_base", "tailbase", "tail"]),

            # Lateral torso: left → center → right body keypoints
            # Angle narrows during rearing (body compressed) and widens prone
            ("lateral_torso",
             ["body_left", "left_body"],
             ["body_mid", "center_body", "body_center", "center", "back"],
             ["body_right", "right_body"]),

            # Head-neck angle: how the head is oriented relative to the body
            # Uses ear as the "neck pivot" point when dedicated neck KP absent
            ("head_neck_angle",
             ["nose", "snout"],
             ["neck", "ear_left", "left_ear"],
             _BODY_CENTER_DISTAL),

            # Forelimb spread: angle at the shoulder/back between the two forepaws
            # Wide angle → raised forepaws (rearing or digging)
            ("forelimb_spread",
             ["frontleg_left", "front_leg_left", "forepaw_left", "front_left", "paw_l"],
             ["back", "body_center", "center_body", "center", "spine1", "shoulder"],
             ["frontleg_right", "front_leg_right", "forepaw_right", "front_right", "paw_r"]),

            # Hindlimb spread: angle at rump between the two hind paws
            ("hindlimb_spread",
             ["backleg_left", "back_leg_left", "hindpaw_left", "hind_left"],
             ["tail_base", "tailbase", "rump", "back"],
             ["backleg_right", "back_leg_right", "hindpaw_right", "hind_right"]),

            # Individual forelimb flexion angles (when elbow/shoulder tracked)
            ("fore_limb_left",
             ["paw_left", "paw_l", "forepaw_left", "front_left", "frontleg_left"],
             ["elbow_left", "shoulder_left"],
             _BODY_CENTER_DISTAL),
            ("fore_limb_right",
             ["paw_right", "paw_r", "forepaw_right", "front_right", "frontleg_right"],
             ["elbow_right", "shoulder_right"],
             _BODY_CENTER_DISTAL),

            # Individual hindlimb angles (when knee/hip tracked)
            ("hind_limb_left",
             ["hind_paw_left", "hindpaw_left", "backleg_left", "back_leg_left"],
             ["knee_left", "hip_left"],
             ["tail_base", "tailbase"]),
            ("hind_limb_right",
             ["hind_paw_right", "hindpaw_right", "backleg_right", "back_leg_right"],
             ["knee_right", "hip_right"],
             ["tail_base", "tailbase"]),
        ]

        out: dict[str, np.ndarray] = {}
        for feat_name, prox_cands, joint_cands, distal_cands in triplet_defs:
            prox_col = _get(prox_cands)
            joint_col = _get(joint_cands)
            distal_col = _get(distal_cands)
            if prox_col and joint_col and distal_col:
                # Avoid degenerate same-keypoint triplets (e.g. proximal == distal)
                if len({prox_col, joint_col, distal_col}) < 3:
                    continue
                angle = _angle_at_joint(
                    np.asarray(pose.x[prox_col], dtype=float),
                    np.asarray(pose.y[prox_col], dtype=float),
                    np.asarray(pose.x[joint_col], dtype=float),
                    np.asarray(pose.y[joint_col], dtype=float),
                    np.asarray(pose.x[distal_col], dtype=float),
                    np.asarray(pose.y[distal_col], dtype=float),
                )
                out[f"joint_angle_{feat_name}"] = angle
        return out

    @staticmethod
    def _compute_spine_curvature_features(pose: PoseData) -> dict[str, np.ndarray]:
        """Compute spine curvature from ordered midline keypoints.

        Curvature is the mean absolute angular change along the spine chain.
        Returns an empty dict when fewer than three midline keypoints are found.

        Uses token-set matching so 'center_body' is found via 'body_center',
        and 'back' is found as a standalone midline candidate.
        Covers TMT (nose, back, tailbase), NOvel/EPM (nose, center_body/body_mid, tail_base).
        """
        norm_to_col = {normalize_bodypart_name(bp): bp for bp in pose.body_parts}
        bp_tokens: dict[str, frozenset[str]] = {
            norm: frozenset(re.split(r"[_\-\s]+", norm))
            for norm in norm_to_col
        }

        def _find(candidates: list[str]) -> str | None:
            return _find_bp_col(norm_to_col, bp_tokens, candidates)

        # Ordered midline candidate slots: head → ... → tail
        # Each slot is a list of name variants; only the first matching one is used.
        # Token-set matching means 'body_center' also matches 'center_body', etc.
        midline_slots: list[list[str]] = [
            ["nose", "snout", "head"],
            ["neck"],
            ["spine_1", "spine1"],
            ["back", "back_mid"],                                           # TMT-style midpoint
            ["spine_2", "spine2"],
            ["spine_3", "spine3"],
            ["body_center", "center_body", "body_mid", "center", "mid_back"],  # NOvel/EPM-style
            ["tail_base", "tailbase", "tail"],
        ]

        midline_cols: list[str] = []
        for slot in midline_slots:
            col = _find(slot)
            # Avoid adding the same column twice (e.g. if 'back' and 'body_center' both resolve to same kp)
            if col and col not in midline_cols:
                midline_cols.append(col)

        if len(midline_cols) < 3:
            return {}

        curvatures: list[np.ndarray] = []
        for k in range(1, len(midline_cols) - 1):
            ax = np.asarray(pose.x[midline_cols[k - 1]], dtype=float)
            ay = np.asarray(pose.y[midline_cols[k - 1]], dtype=float)
            bx = np.asarray(pose.x[midline_cols[k]], dtype=float)
            by = np.asarray(pose.y[midline_cols[k]], dtype=float)
            cx = np.asarray(pose.x[midline_cols[k + 1]], dtype=float)
            cy = np.asarray(pose.y[midline_cols[k + 1]], dtype=float)
            v1x, v1y = bx - ax, by - ay
            v2x, v2y = cx - bx, cy - by
            a1 = np.arctan2(v1y, v1x)
            a2 = np.arctan2(v2y, v2x)
            curvatures.append(np.abs(np.angle(np.exp(1j * (a2 - a1)))))

        mean_curvature = np.nanmean(np.stack(curvatures, axis=1), axis=1)
        return {"spine_curvature": mean_curvature}

    @staticmethod
    def probe_metadata(path: Path) -> dict:
        """Return lightweight metadata (body parts, frame count) without full load."""
        meta: dict = {"path": str(path), "n_frames": 0, "body_parts": []}
        try:
            if path.suffix.lower() == ".csv":
                df_head = pd.read_csv(path, header=[0, 1, 2], index_col=0, nrows=0)
                if df_head.columns.nlevels == 3:
                    df_head.columns = df_head.columns.droplevel(0)
                meta["body_parts"] = [
                    normalize_bodypart_name(bp)
                    for bp in df_head.columns.get_level_values(0).unique()
                ]
                # Count rows (fast)
                meta["n_frames"] = sum(1 for _ in open(path)) - 3  # 3 header rows
            elif path.suffix.lower() in (".h5", ".hdf5"):
                try:
                    df_h5 = pd.read_hdf(str(path), key="df_with_missing", stop=0)
                except Exception:
                    df_h5 = pd.read_hdf(str(path), stop=0)
                if df_h5.columns.nlevels == 3:
                    df_h5.columns = df_h5.columns.droplevel(0)  # type: ignore[assignment]
                meta["body_parts"] = [
                    normalize_bodypart_name(bp)
                    for bp in df_h5.columns.get_level_values(0).unique()
                ]
        except Exception as exc:
            meta["error"] = str(exc)
        return meta
