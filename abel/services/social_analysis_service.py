"""Social-interaction analytics and a spatial-displacement dominance HMM.

This service consumes the per-frame *social* features written by
``PoseProcessingService.extract_and_save_frame_pose_features_multi`` (the
``social_*`` columns, present only when a project tracks more than one animal and
interaction features are enabled) and produces two things:

1. **Summary metrics** — per (subject, session) descriptors of the dyadic
   relationship: mean inter-animal distance, time in contact, contact bouts,
   net approach, orientation, and a directed advance/yield balance.

2. **A dominance HMM** — a Gaussian HMM fit over continuous social + movement
   features, *pooled across the whole cohort* so its latent states form one
   shared vocabulary of interaction modes (separated / approaching / contact …).
   States dominated by close proximity are flagged as *interaction* states, and
   within those a **spatial-displacement dominance score** is computed per
   subject: the animal that advances into the other's space while the other
   yields is scored as more dominant.  Subjects are then ranked within each
   session, surfacing who the dominant individual is.

The HMM fit requires ``hmmlearn`` (an optional dependency, like the existing
categorical behavior HMM); the summary and dominance-scoring logic are pure and
run without it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("abel")


# Continuous per-frame features the dominance HMM is fit on.  Only those present
# in the loaded frame table are used (older caches may lack the newer columns).
SOCIAL_HMM_FEATURES: tuple[str, ...] = (
    "social_dist_centroid_to_centroid_nearest_norm",
    "social_min_keypoint_dist_nearest_norm",
    "social_approach_velocity_nearest",
    "social_radial_velocity_toward_nearest",
    "social_facing_angle_nearest",
    "social_heading_alignment_nearest",
    "social_in_contact",
    "centroid_velocity",
)

# Column used to gauge how "close" (interaction-like) a state is.  Lower = the
# animals are nearer, so states with a low mean here are interaction states.
_PROXIMITY_COL = "social_dist_centroid_to_centroid_nearest_norm"


class SocialAnalysisService:
    """Compute social-interaction summaries and a dominance HMM."""

    # ── Data loading ─────────────────────────────────────────────────────

    @staticmethod
    def frame_pose_path(project_root: Path) -> Path:
        return project_root / "derived" / "pose_features" / "frame_pose.parquet"

    def load_social_frames(self, project_root: Path) -> pd.DataFrame | None:
        """Return the per-frame pose table if it carries social columns.

        Returns ``None`` when the table is missing or has no ``social_*``
        columns (single-animal project, or interaction features never
        extracted).
        """
        path = self.frame_pose_path(project_root)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except Exception:
            logger.exception("Failed to read %s", path)
            return None
        if not any(c.startswith("social_") for c in df.columns):
            return None
        return df

    @staticmethod
    def has_social_features(df: pd.DataFrame | None) -> bool:
        return df is not None and any(c.startswith("social_") for c in df.columns)

    # ── Per (subject, session) summary metrics ───────────────────────────

    def compute_social_summary(
        self, df: pd.DataFrame, fps: float, group_map: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """One row per (animal_id, session_id) of dyadic summary metrics.

        ``group_map`` optionally maps ``session_id`` → experimental group label.
        """
        if df is None or df.empty:
            return []
        group_map = group_map or {}
        rows: list[dict[str, Any]] = []
        for (animal_id, session_id), g in df.groupby(["animal_id", "session_id"], sort=True):
            rows.append(
                self._summary_for_group(str(animal_id), str(session_id), g, fps, group_map)
            )
        return rows

    def _summary_for_group(
        self,
        animal_id: str,
        session_id: str,
        g: pd.DataFrame,
        fps: float,
        group_map: dict[str, str],
    ) -> dict[str, Any]:
        def _mean(col: str) -> float:
            if col in g:
                v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
                return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            return float("nan")

        contact = (
            pd.to_numeric(g.get("social_in_contact"), errors="coerce").to_numpy(dtype=float)
            if "social_in_contact" in g
            else np.zeros(len(g))
        )
        contact_bool = np.nan_to_num(contact, nan=0.0) > 0.5
        n_contact = int(contact_bool.sum())
        contact_time_s = (n_contact / fps) if fps > 0 else 0.0
        # Contact bouts = rising edges of the contact mask.
        starts = contact_bool & ~np.concatenate(([False], contact_bool[:-1]))
        n_bouts = int(starts.sum())
        mean_bout_s = (contact_time_s / n_bouts) if n_bouts > 0 else 0.0

        radial = (
            pd.to_numeric(g.get("social_radial_velocity_toward_nearest"), errors="coerce")
            .to_numpy(dtype=float)
            if "social_radial_velocity_toward_nearest" in g
            else np.full(len(g), np.nan)
        )
        # Fraction of *detected* frames spent advancing. Filter NaNs first:
        # ``radial > 0`` maps NaN (undetected) to False, so nanmean over the
        # bool array would otherwise divide by the full frame count and deflate
        # the fraction for sparsely-tracked dyads.
        radial_finite = radial[np.isfinite(radial)]
        advance_frac = (
            float(np.mean(radial_finite > 0)) if radial_finite.size else float("nan")
        )

        return {
            "animal_id": animal_id,
            "session_id": session_id,
            "group": group_map.get(session_id, ""),
            "n_frames": int(len(g)),
            "mean_distance_norm": _mean("social_dist_centroid_to_centroid_nearest_norm"),
            "mean_distance_px": _mean("social_dist_centroid_to_centroid_nearest"),
            "contact_time_s": contact_time_s,
            "contact_fraction": (n_contact / len(g)) if len(g) else 0.0,
            "n_contact_bouts": n_bouts,
            "mean_contact_bout_s": mean_bout_s,
            "mean_approach_velocity": _mean("social_approach_velocity_nearest"),
            "mean_radial_velocity_toward": _mean("social_radial_velocity_toward_nearest"),
            "advance_fraction": advance_frac,
            "mean_facing_angle": _mean("social_facing_angle_nearest"),
            "mean_heading_alignment": _mean("social_heading_alignment_nearest"),
        }

    # ── Dominance HMM ────────────────────────────────────────────────────

    def available_hmm_features(self, df: pd.DataFrame) -> list[str]:
        return [c for c in SOCIAL_HMM_FEATURES if c in df.columns]

    def fit_dominance_hmm(
        self,
        df: pd.DataFrame,
        *,
        fps: float,
        n_states: int = 4,
        n_iter: int = 100,
        feature_cols: list[str] | None = None,
        group_map: dict[str, str] | None = None,
        random_state: int = 0,
    ) -> dict[str, Any]:
        """Fit a pooled Gaussian HMM and derive per-subject dominance scores.

        Returns a dict with ``error`` set on failure, otherwise:
        ``n_states``, ``feature_cols``, ``state_profiles`` (per-state mean of
        each feature), ``interaction_states`` (list of state ids), ``occupancy``
        (per (animal,session) state fractions), ``dominance`` (per
        (animal,session) score + within-session rank), and ``log_likelihood``.
        """
        try:
            from hmmlearn import hmm as hmmlearn_hmm  # type: ignore[import-untyped]
        except ImportError:
            return {
                "n_states": 0,
                "error": (
                    "hmmlearn is required for the dominance HMM.\n"
                    "Install it via the Dependencies tab (pip install hmmlearn)."
                ),
            }

        feats = feature_cols or self.available_hmm_features(df)
        if len(feats) < 2:
            return {"n_states": 0, "error": "Not enough social feature columns to fit an HMM."}

        seqs, keys = self._build_sequences(df, feats)
        if not seqs:
            return {"n_states": 0, "error": "No usable multi-animal frames found."}

        # Standardize globally so no single feature dominates the covariance.
        stacked = np.concatenate(seqs, axis=0)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        std = np.where(std > 1e-9, std, 1.0)

        def _norm(a: np.ndarray) -> np.ndarray:
            z = (a - mean) / std
            return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        norm_seqs = [_norm(s) for s in seqs]
        X = np.concatenate(norm_seqs, axis=0)
        lengths = [len(s) for s in norm_seqs]

        n_states = max(2, min(int(n_states), 12))
        model = hmmlearn_hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=int(n_iter),
            tol=1e-3,
            random_state=int(random_state),
            verbose=False,
        )
        try:
            model.fit(X, lengths)
            ll = float(model.score(X, lengths))
            states_all = model.predict(X, lengths)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Dominance HMM fit failed")
            return {"n_states": 0, "error": f"HMM fit failed: {exc}"}

        # Split the flat state vector back into per-(animal,session) sequences.
        state_seqs: dict[tuple[str, str], np.ndarray] = {}
        off = 0
        for key, ln in zip(keys, lengths):
            state_seqs[key] = states_all[off:off + ln]
            off += ln

        profiles = self._state_profiles(seqs, keys, states_all, lengths, feats, n_states)
        interaction_states = self._identify_interaction_states(profiles, feats)
        occupancy = self._state_occupancy(state_seqs, n_states)
        dominance = self.compute_displacement_dominance(
            df, state_seqs, interaction_states, fps, group_map or {}
        )

        return {
            "n_states": n_states,
            "feature_cols": feats,
            "state_profiles": profiles,          # {state_id: {feat: mean_value}}
            "interaction_states": interaction_states,
            "occupancy": occupancy,              # {(animal,session): [frac per state]}
            "dominance": dominance,              # list of per-subject dicts
            "log_likelihood": ll,
            "error": None,
        }

    # ── HMM helpers ──────────────────────────────────────────────────────

    def _build_sequences(
        self, df: pd.DataFrame, feats: list[str]
    ) -> tuple[list[np.ndarray], list[tuple[str, str]]]:
        """Return per-(animal,session) feature matrices ordered by frame."""
        seqs: list[np.ndarray] = []
        keys: list[tuple[str, str]] = []
        for (animal_id, session_id), g in df.groupby(["animal_id", "session_id"], sort=True):
            gg = g.sort_values("frame") if "frame" in g else g
            mat = gg[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            # Skip groups that are entirely undetected (all-NaN rows).
            if mat.shape[0] < 2 or not np.isfinite(mat).any():
                continue
            seqs.append(mat)
            keys.append((str(animal_id), str(session_id)))
        return seqs, keys

    @staticmethod
    def _state_profiles(
        seqs: list[np.ndarray],
        keys: list[tuple[str, str]],
        states_all: np.ndarray,
        lengths: list[int],
        feats: list[str],
        n_states: int,
    ) -> dict[int, dict[str, float]]:
        """Mean of each *raw* feature within each state (for interpretation)."""
        raw = np.concatenate(seqs, axis=0)
        profiles: dict[int, dict[str, float]] = {}
        for s in range(n_states):
            m = states_all == s
            if not m.any():
                profiles[s] = {f: float("nan") for f in feats}
                continue
            with np.errstate(invalid="ignore"):
                means = np.nanmean(raw[m], axis=0)
            profiles[s] = {f: float(v) for f, v in zip(feats, means)}
        return profiles

    @staticmethod
    def _identify_interaction_states(
        profiles: dict[int, dict[str, float]], feats: list[str]
    ) -> list[int]:
        """States whose mean proximity is below the median across states.

        Falls back to contact fraction, then to "all states" if neither signal
        is available, so downstream dominance scoring always has frames to use.
        """
        prox_key = _PROXIMITY_COL if _PROXIMITY_COL in feats else None
        if prox_key is not None:
            vals = {s: p.get(prox_key, np.nan) for s, p in profiles.items()}
            finite = {s: v for s, v in vals.items() if np.isfinite(v)}
            if finite:
                thresh = float(np.median(list(finite.values())))
                return sorted(s for s, v in finite.items() if v <= thresh)
        if "social_in_contact" in feats:
            vals = {s: p.get("social_in_contact", np.nan) for s, p in profiles.items()}
            finite = {s: v for s, v in vals.items() if np.isfinite(v)}
            if finite:
                thresh = float(np.median(list(finite.values())))
                return sorted(s for s, v in finite.items() if v >= thresh)
        return sorted(profiles.keys())

    @staticmethod
    def _state_occupancy(
        state_seqs: dict[tuple[str, str], np.ndarray], n_states: int
    ) -> dict[tuple[str, str], list[float]]:
        occ: dict[tuple[str, str], list[float]] = {}
        for key, seq in state_seqs.items():
            counts = np.bincount(seq, minlength=n_states).astype(float)
            total = counts.sum()
            occ[key] = (counts / total).tolist() if total > 0 else [0.0] * n_states
        return occ

    # ── Spatial-displacement dominance ───────────────────────────────────

    def compute_displacement_dominance(
        self,
        df: pd.DataFrame,
        state_seqs: dict[tuple[str, str], np.ndarray],
        interaction_states: list[int],
        fps: float,
        group_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Per (subject, session) spatial-displacement dominance, ranked/session.

        Within interaction-state frames, a subject that *advances* into the
        other's space (positive radial velocity toward it) while the other
        *yields* (retreats) is scored dominant.  The score is the mean radial
        velocity toward the other over interaction frames minus the fraction of
        those frames spent yielding — higher = more dominant.  Subjects are
        ranked within each session (rank 1 = most dominant).
        """
        inter = set(interaction_states)
        per_session: dict[str, list[dict[str, Any]]] = {}

        grouped = {
            (str(a), str(s)): g
            for (a, s), g in df.groupby(["animal_id", "session_id"], sort=True)
        }
        for key, seq in state_seqs.items():
            animal_id, session_id = key
            g = grouped.get(key)
            if g is None:
                continue
            gg = g.sort_values("frame") if "frame" in g else g
            radial = pd.to_numeric(
                gg.get("social_radial_velocity_toward_nearest"), errors="coerce"
            ).to_numpy(dtype=float) if "social_radial_velocity_toward_nearest" in gg else None
            n = min(len(seq), 0 if radial is None else len(radial))
            if radial is None or n == 0:
                continue
            in_inter = np.isin(seq[:n], list(inter)) if inter else np.ones(n, dtype=bool)
            r = radial[:n][in_inter]
            r = r[np.isfinite(r)]
            if r.size == 0:
                advance = 0.0
                yield_frac = 0.0
                inter_time_s = 0.0
                score = float("nan")
            else:
                advance = float(np.mean(r))
                yield_frac = float(np.mean(r < 0))
                inter_time_s = (r.size / fps) if fps > 0 else 0.0
                score = advance - yield_frac
            per_session.setdefault(session_id, []).append({
                "animal_id": animal_id,
                "session_id": session_id,
                "group": group_map.get(session_id, ""),
                "interaction_time_s": inter_time_s,
                "mean_advance": advance,
                "yield_fraction": yield_frac,
                "dominance_score": score,
            })

        # Rank within each session (highest score = rank 1 = most dominant).
        out: list[dict[str, Any]] = []
        for _sid, rows in per_session.items():
            ranked = sorted(
                rows,
                key=lambda r: (r["dominance_score"] if np.isfinite(r["dominance_score"]) else -np.inf),
                reverse=True,
            )
            for i, r in enumerate(ranked, start=1):
                r["dominance_rank"] = i
                r["is_dominant"] = i == 1 and len(ranked) > 1
                out.append(r)
        return out
