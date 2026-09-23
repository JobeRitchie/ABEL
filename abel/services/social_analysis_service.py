"""Social-interaction analytics and a spatial-displacement dominance HMM.

This service consumes the per-frame *social* features written by
``PoseProcessingService.extract_and_save_frame_pose_features_multi`` (the
``social_*`` columns, present only when a project tracks more than one animal and
interaction features are enabled) and produces two things:

1. **Summary metrics**: per (subject, session) descriptors of the dyadic
   relationship: mean inter-animal distance, time in contact, contact bouts,
   net approach, orientation, and a directed advance/yield balance.

2. **A dominance HMM**: a Gaussian HMM fit over continuous social + movement
   features, *pooled across the whole cohort* so its latent states form one
   shared vocabulary of interaction modes (contact / close / apart ...).
   States are numbered by proximity (state 0 = closest).  Frames where the
   partner is not tracked are left out of the fit and carry state ``-1``.
   States dominated by close proximity are flagged as *interaction* states,
   and within those a **displacement dominance index** is computed per
   subject.  In a dyad, a *displacement* is a stretch where one animal moves
   toward the other while the other moves away, both faster than a body-length
   threshold.  The index is ``(won - lost) / (won + lost)`` over those events,
   so it runs from -1 (always displaced) to +1 (always displaces).  Subjects
   are ranked within each session.

The HMM fit requires ``hmmlearn`` (an optional dependency, like the existing
categorical behavior HMM); the summary and dominance-scoring logic are pure and
run without it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Callable

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
_DIST_PX_COL = "social_dist_centroid_to_centroid_nearest"
_RADIAL_COL = "social_radial_velocity_toward_nearest"
_CONTACT_COL = "social_in_contact"
_SPEED_COL = "centroid_velocity"

# Features are clipped to these percentiles before standardizing.  Tracking
# jumps produce velocities of 10,000+ px/s; left in, they claim a state of
# their own and stretch every z-score.
_CLIP_PCT: tuple[float, float] = (0.5, 99.5)

# A displacement needs both animals moving at least this fast along the line
# between them (body lengths per second).  0.5 BL/s is roughly the 20th/80th
# percentile of radial speed in a 30-session home-cage dyad cohort.
DEFAULT_MOVE_THRESH_BL: float = 0.5
DEFAULT_MIN_EVENT_S: float = 0.1
DEFAULT_BIN_S: float = 120.0

_RESULT_FILE = "social_dominance_hmm.pkl"
_RESULT_VERSION = 1


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` half-open index ranges of the True stretches in *mask*."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _binom_p(k: int, n: int) -> float:
    """Two-sided binomial p against 50/50; NaN when there is nothing to test."""
    if n <= 0:
        return float("nan")
    try:
        from scipy.stats import binomtest
        return float(binomtest(int(k), int(n), 0.5).pvalue)
    except Exception:
        return float("nan")


class SocialAnalysisService:
    """Compute social-interaction summaries and a dominance HMM."""

    # ── Data loading ─────────────────────────────────────────────────────

    @staticmethod
    def frame_pose_path(project_root: Path) -> Path:
        return project_root / "derived" / "pose_features" / "frame_pose.parquet"

    def load_social_frames(self, project_root: Path) -> pd.DataFrame | None:
        """Return the per-frame social columns, or ``None`` if there are none.

        Only the id, ``social_*`` and centroid-speed columns are read: the full
        frame table of a multi-animal project runs to gigabytes.
        """
        path = self.frame_pose_path(project_root)
        if not path.exists():
            return None
        try:
            import pyarrow.parquet as pq
            names = list(pq.read_schema(path).names)
            cols = [
                c for c in names
                if c in ("frame", "animal_id", "session_id", _SPEED_COL)
                or c.startswith("social_")
            ]
            if not any(c.startswith("social_") for c in cols):
                return None
            return pd.read_parquet(path, columns=cols)
        except Exception:
            logger.exception("Failed to read %s", path)
            return None

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
            pd.to_numeric(g.get(_CONTACT_COL), errors="coerce").to_numpy(dtype=float)
            if _CONTACT_COL in g
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
            pd.to_numeric(g.get(_RADIAL_COL), errors="coerce").to_numpy(dtype=float)
            if _RADIAL_COL in g
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
            "mean_distance_norm": _mean(_PROXIMITY_COL),
            "mean_distance_px": _mean(_DIST_PX_COL),
            "contact_time_s": contact_time_s,
            "contact_fraction": (n_contact / len(g)) if len(g) else 0.0,
            "n_contact_bouts": n_bouts,
            "mean_contact_bout_s": mean_bout_s,
            "mean_approach_velocity": _mean("social_approach_velocity_nearest"),
            "mean_radial_velocity_toward": _mean(_RADIAL_COL),
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
        n_iter: int = 200,
        n_restarts: int = 10,
        feature_cols: list[str] | None = None,
        group_map: dict[str, str] | None = None,
        random_state: int = 0,
        move_thresh_bl: float = DEFAULT_MOVE_THRESH_BL,
        min_event_s: float = DEFAULT_MIN_EVENT_S,
        bin_s: float = DEFAULT_BIN_S,
        prechop_frames: dict[str, int] | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Fit a pooled Gaussian HMM and derive per-subject dominance scores.

        ``prechop_frames`` maps session id to the first analysed frame; frames
        before it (e.g. the experimenter's hand placing the intruder) are
        dropped from both the fit and the displacement counts.

        Returns a dict with ``error`` set on failure.  On success the keys are
        described in :meth:`_assemble_result`.
        """
        prechop = self.clean_prechop(prechop_frames)
        df = self.trim_prechop(df, prechop)
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

        def _say(msg: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(msg)
                except Exception:
                    pass

        feats = feature_cols or self.available_hmm_features(df)
        if len(feats) < 2:
            return {"n_states": 0, "error": "Not enough social feature columns to fit an HMM."}

        seqs, keys, frames = self._build_sequences(df, feats)
        if not seqs:
            return {"n_states": 0, "error": "No usable multi-animal frames found."}

        n_states = max(2, min(int(n_states), 12))
        valid = [self._valid_mask(s, feats) for s in seqs]
        raw_valid = np.concatenate([s[m] for s, m in zip(seqs, valid)], axis=0)
        if raw_valid.shape[0] < n_states * 20:
            return {
                "n_states": 0,
                "error": "Too few frames with both animals tracked to fit an HMM.",
            }

        # Clip, then standardize globally so no single feature dominates.
        with np.errstate(invalid="ignore"):
            lo, hi = np.nanpercentile(raw_valid, _CLIP_PCT, axis=0)
        lo = np.where(np.isfinite(lo), lo, -np.inf)
        hi = np.where(np.isfinite(hi), hi, np.inf)
        clipped_valid = np.clip(raw_valid, lo, hi)
        mean = np.nanmean(clipped_valid, axis=0)
        std = np.nanstd(clipped_valid, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)

        def _norm(a: np.ndarray) -> np.ndarray:
            z = (np.clip(a, lo, hi) - mean) / std
            return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        # Each unbroken stretch of tracked-partner frames is its own sequence,
        # so the model never learns a transition across a tracking gap.
        run_index: list[tuple[int, int, int]] = []
        pieces: list[np.ndarray] = []
        for i, (s, m) in enumerate(zip(seqs, valid)):
            for a, b in _runs(m):
                if b - a >= 2:
                    run_index.append((i, a, b))
                    pieces.append(_norm(s[a:b]))
        if not pieces:
            return {"n_states": 0, "error": "No unbroken stretches of tracked frames."}
        X = np.concatenate(pieces, axis=0)
        lengths = [len(p) for p in pieces]

        # hmmlearn's tol is on the total log-likelihood, so it has to scale
        # with the frame count or a million-frame fit never reports converged.
        tol = max(1e-2, 1e-5 * len(X))
        best_model = None
        best_ll = -np.inf
        restart_ll: list[float] = []
        n_restarts = max(1, int(n_restarts))
        for r in range(n_restarts):
            _say(f"Fitting dominance HMM: restart {r + 1} of {n_restarts}…")
            model = hmmlearn_hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=int(n_iter),
                tol=tol,
                random_state=int(random_state) + r,
                verbose=False,
            )
            try:
                model.fit(X, lengths)
                ll = float(model.score(X, lengths))
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Dominance HMM fit failed (restart %d)", r)
                if best_model is None and r == n_restarts - 1:
                    return {"n_states": 0, "error": f"HMM fit failed: {exc}"}
                continue
            restart_ll.append(ll)
            if ll > best_ll:
                best_ll, best_model = ll, model
        if best_model is None:
            return {"n_states": 0, "error": "HMM fit failed on every restart."}

        _say("Decoding states and scoring dominance…")
        decoded = best_model.predict(X, lengths)

        # Scatter decoded runs back onto each full sequence (-1 = no partner).
        state_seqs: dict[tuple[str, str], np.ndarray] = {
            k: np.full(len(s), -1, dtype=np.int16) for k, s in zip(keys, seqs)
        }
        off = 0
        for (i, a, b) in run_index:
            state_seqs[keys[i]][a:b] = decoded[off:off + (b - a)]
            off += b - a

        # Number states by proximity so state 0 is always the closest mode.
        raw_clipped = [np.clip(s, lo, hi) for s in seqs]
        profiles = self._state_profiles_from_seqs(raw_clipped, keys, state_seqs, feats, n_states)
        order = self._proximity_order(profiles, n_states)
        remap = np.empty(n_states, dtype=np.int16)
        remap[order] = np.arange(n_states, dtype=np.int16)
        for k, seq in state_seqs.items():
            ok = seq >= 0
            seq[ok] = remap[seq[ok]]
        profiles = {int(remap[s]): p for s, p in profiles.items()}

        z_means = np.asarray(best_model.means_, dtype=float)[order]
        transmat = np.asarray(best_model.transmat_, dtype=float)[np.ix_(order, order)]

        interaction_states = self._identify_interaction_states(profiles, feats)
        occupancy, no_partner = self._state_occupancy(state_seqs, n_states)
        dwell, switch = self._dwell_and_switching(state_seqs, n_states, fps)
        analysis = self.displacement_analysis(
            df, state_seqs, interaction_states, fps, group_map or {},
            move_thresh_bl=move_thresh_bl, min_event_s=min_event_s, bin_s=bin_s,
        )
        dominance = analysis["dominance"]

        monitor = getattr(best_model, "monitor_", None)
        return {
            "n_states": n_states,
            "feature_cols": list(feats),
            "state_profiles": profiles,          # {state_id: {feat: clipped mean}}
            "state_labels": self._state_labels(profiles, interaction_states),
            "z_means": z_means,                  # (n_states, n_feats) standardized
            "transmat": transmat,                # per-frame transition matrix
            "switch_matrix": switch,             # P(next state | leaving state)
            "dwell": dwell,                      # {state: {mean_s, median_s, n, samples}}
            "interaction_states": interaction_states,
            "occupancy": occupancy,              # {(animal,session): [frac per state]}
            "no_partner_fraction": no_partner,   # {(animal,session): frac}
            "state_seqs": state_seqs,            # {(animal,session): int16 per frame}
            "frames": frames,                    # {(animal,session): frame numbers}
            "dominance": dominance,              # list of per-subject dicts
            "dominance_bins": analysis["bins"],
            "displacement_events": analysis["events"],
            "identity_check": self.identity_rank_bias(dominance),
            "group_stats": self.group_steepness(dominance),
            "log_likelihood": best_ll,
            "fps": float(fps),
            "settings": {
                "n_states": n_states,
                "n_iter": int(n_iter),
                "n_restarts": n_restarts,
                "move_thresh_bl": float(move_thresh_bl),
                "min_event_s": float(min_event_s),
                "bin_s": float(bin_s),
                "prechop_frames": prechop,
            },
            "fit_info": {
                "converged": bool(getattr(monitor, "converged", False)),
                "n_iter_run": int(getattr(monitor, "iter", 0) or 0),
                "tol": float(tol),
                "restart_log_likelihoods": restart_ll,
                "n_frames_fit": int(len(X)),
                "n_sequences": int(len(lengths)),
                "clip_low": dict(zip(feats, map(float, lo))),
                "clip_high": dict(zip(feats, map(float, hi))),
            },
            "error": None,
        }

    # ── HMM helpers ──────────────────────────────────────────────────────

    def _build_sequences(
        self, df: pd.DataFrame, feats: list[str]
    ) -> tuple[list[np.ndarray], list[tuple[str, str]], dict[tuple[str, str], np.ndarray]]:
        """Return per-(animal,session) feature matrices ordered by frame."""
        seqs: list[np.ndarray] = []
        keys: list[tuple[str, str]] = []
        frames: dict[tuple[str, str], np.ndarray] = {}
        for (animal_id, session_id), g in df.groupby(["animal_id", "session_id"], sort=True):
            gg = g.sort_values("frame") if "frame" in g else g
            mat = gg[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            # Skip groups that are entirely undetected (all-NaN rows).
            if mat.shape[0] < 2 or not np.isfinite(mat).any():
                continue
            key = (str(animal_id), str(session_id))
            seqs.append(mat)
            keys.append(key)
            frames[key] = (
                gg["frame"].to_numpy(dtype=np.int32) if "frame" in gg
                else np.arange(len(gg), dtype=np.int32)
            )
        return seqs, keys, frames

    @staticmethod
    def _valid_mask(mat: np.ndarray, feats: list[str]) -> np.ndarray:
        """Frames where the partner is measurable (proximity is finite)."""
        if _PROXIMITY_COL in feats:
            return np.isfinite(mat[:, feats.index(_PROXIMITY_COL)])
        social = [i for i, f in enumerate(feats) if f.startswith("social_")]
        if social:
            return np.isfinite(mat[:, social]).any(axis=1)
        return np.isfinite(mat).any(axis=1)

    @staticmethod
    def _state_profiles_from_seqs(
        seqs: list[np.ndarray],
        keys: list[tuple[str, str]],
        state_seqs: dict[tuple[str, str], np.ndarray],
        feats: list[str],
        n_states: int,
    ) -> dict[int, dict[str, float]]:
        """Mean of each clipped raw feature within each state."""
        raw = np.concatenate(seqs, axis=0)
        states = np.concatenate([state_seqs[k] for k in keys])
        profiles: dict[int, dict[str, float]] = {}
        for s in range(n_states):
            m = states == s
            if not m.any():
                profiles[s] = {f: float("nan") for f in feats}
                continue
            with np.errstate(invalid="ignore"), _quiet_nanmean():
                means = np.nanmean(raw[m], axis=0)
            profiles[s] = {f: float(v) for f, v in zip(feats, means)}
        return profiles

    @staticmethod
    def _proximity_order(profiles: dict[int, dict[str, float]], n_states: int) -> np.ndarray:
        """Old state ids sorted closest-first (identity if proximity is absent)."""
        vals = np.array(
            [profiles.get(s, {}).get(_PROXIMITY_COL, np.nan) for s in range(n_states)],
            dtype=float,
        )
        if not np.isfinite(vals).any():
            return np.arange(n_states)
        return np.argsort(np.where(np.isfinite(vals), vals, np.inf), kind="stable")

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
        if _CONTACT_COL in feats:
            vals = {s: p.get(_CONTACT_COL, np.nan) for s, p in profiles.items()}
            finite = {s: v for s, v in vals.items() if np.isfinite(v)}
            if finite:
                thresh = float(np.median(list(finite.values())))
                return sorted(s for s, v in finite.items() if v >= thresh)
        return sorted(profiles.keys())

    @staticmethod
    def _state_labels(
        profiles: dict[int, dict[str, float]], interaction_states: list[int]
    ) -> dict[int, str]:
        """Short readable name per state from its proximity, contact and speed."""
        speeds = [p.get(_SPEED_COL, np.nan) for p in profiles.values()]
        finite_speeds = [v for v in speeds if np.isfinite(v)]
        med_speed = float(np.median(finite_speeds)) if finite_speeds else float("nan")
        labels: dict[int, str] = {}
        for s, p in sorted(profiles.items()):
            parts: list[str] = []
            contact = p.get(_CONTACT_COL, np.nan)
            dist = p.get(_PROXIMITY_COL, np.nan)
            if np.isfinite(contact) and contact >= 0.5:
                parts.append("contact")
            elif np.isfinite(dist):
                parts.append(f"{dist:.1f} BL apart")
            speed = p.get(_SPEED_COL, np.nan)
            if np.isfinite(speed) and np.isfinite(med_speed) and med_speed > 0:
                if speed >= 1.5 * med_speed:
                    parts.append("moving")
                elif speed <= 0.67 * med_speed:
                    parts.append("still")
            if s in interaction_states:
                parts.append("interaction")
            labels[s] = f"S{s}" + (f" ({', '.join(parts)})" if parts else "")
        return labels

    @staticmethod
    def _state_occupancy(
        state_seqs: dict[tuple[str, str], np.ndarray], n_states: int
    ) -> tuple[dict[tuple[str, str], list[float]], dict[tuple[str, str], float]]:
        """State fractions over tracked-partner frames, plus the untracked share."""
        occ: dict[tuple[str, str], list[float]] = {}
        absent: dict[tuple[str, str], float] = {}
        for key, seq in state_seqs.items():
            ok = seq >= 0
            counts = np.bincount(seq[ok].astype(np.int64), minlength=n_states).astype(float)
            total = counts.sum()
            occ[key] = (counts / total).tolist() if total > 0 else [0.0] * n_states
            absent[key] = float(1.0 - ok.mean()) if len(seq) else 0.0
        return occ, absent

    @staticmethod
    def _dwell_and_switching(
        state_seqs: dict[tuple[str, str], np.ndarray], n_states: int, fps: float,
        max_samples: int = 20000,
    ) -> tuple[dict[int, dict[str, Any]], np.ndarray]:
        """Per-state dwell times (s) and the state-switch probability matrix.

        A switch is a change of state between two adjacent tracked frames;
        runs broken by an untracked gap do not count as a switch.
        """
        period = 1.0 / fps if fps > 0 else 1.0
        dwells: dict[int, list[np.ndarray]] = {s: [] for s in range(n_states)}
        counts = np.zeros((n_states, n_states), dtype=float)
        for seq in state_seqs.values():
            if len(seq) == 0:
                continue
            change = np.flatnonzero(np.diff(seq) != 0) + 1
            starts = np.concatenate(([0], change))
            ends = np.concatenate((change, [len(seq)]))
            vals = seq[starts]
            for s in range(n_states):
                sel = vals == s
                if sel.any():
                    dwells[s].append((ends[sel] - starts[sel]) * period)
            a, b = vals[:-1], vals[1:]
            ok = (a >= 0) & (b >= 0)
            np.add.at(counts, (a[ok].astype(np.int64), b[ok].astype(np.int64)), 1.0)
        rng = np.random.default_rng(0)
        out: dict[int, dict[str, Any]] = {}
        for s in range(n_states):
            d = np.concatenate(dwells[s]) if dwells[s] else np.zeros(0)
            sample = d if d.size <= max_samples else rng.choice(d, max_samples, replace=False)
            out[s] = {
                "n": int(d.size),
                "mean_s": float(d.mean()) if d.size else float("nan"),
                "median_s": float(np.median(d)) if d.size else float("nan"),
                "samples": sample.astype(np.float32),
            }
        with np.errstate(invalid="ignore", divide="ignore"):
            row = counts.sum(axis=1, keepdims=True)
            switch = np.where(row > 0, counts / row, np.nan)
        return out, switch

    # ── Spatial-displacement dominance ───────────────────────────────────

    def compute_displacement_dominance(
        self,
        df: pd.DataFrame,
        state_seqs: dict[tuple[str, str], np.ndarray],
        interaction_states: list[int],
        fps: float,
        group_map: dict[str, str],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Per (subject, session) dominance rows, ranked within each session."""
        return self.displacement_analysis(
            df, state_seqs, interaction_states, fps, group_map, **kwargs
        )["dominance"]

    def displacement_analysis(
        self,
        df: pd.DataFrame,
        state_seqs: dict[tuple[str, str], np.ndarray],
        interaction_states: list[int],
        fps: float,
        group_map: dict[str, str],
        *,
        move_thresh_bl: float = DEFAULT_MOVE_THRESH_BL,
        min_event_s: float = DEFAULT_MIN_EVENT_S,
        bin_s: float = DEFAULT_BIN_S,
    ) -> dict[str, list[dict[str, Any]]]:
        """Displacement dominance per subject, per time bin, and per event.

        Radial speed (the subject's velocity along the line to its partner) is
        divided by the subject's median body length so the threshold is in
        body lengths per second.  Only interaction-state frames count.

        * **Dyads** (exactly two animals in the session): a displacement is a
          run of at least ``min_event_s`` where one animal advances faster than
          the threshold while the other retreats faster than it.  The score is
          the dominance index ``(won - lost) / (won + lost)``.
        * **Larger groups**: the partner is "nearest", whose identity the frame
          table does not keep, so events cannot be paired.  The score falls
          back to ``advance_fraction - yield_fraction`` of the subject alone.
        """
        inter = [int(s) for s in interaction_states]
        thr = float(move_thresh_bl)
        min_frames = max(1, int(round(float(min_event_s) * fps))) if fps > 0 else 1
        per_session: dict[str, dict[str, dict[str, Any]]] = {}

        for (a, s), g in df.groupby(["animal_id", "session_id"], sort=True):
            key = (str(a), str(s))
            seq = state_seqs.get(key)
            if seq is None:
                continue
            gg = g.sort_values("frame") if "frame" in g else g
            if _RADIAL_COL not in gg:
                continue
            n = min(len(seq), len(gg))
            if n == 0:
                continue
            radial = pd.to_numeric(gg[_RADIAL_COL], errors="coerce").to_numpy(dtype=float)[:n]
            bl = self._median_body_length(gg)
            frames = (
                gg["frame"].to_numpy(dtype=np.int64)[:n] if "frame" in gg
                else np.arange(n, dtype=np.int64)
            )
            per_session.setdefault(key[1], {})[key[0]] = {
                "frames": frames,
                "radial_bl": radial / bl if np.isfinite(bl) and bl > 0 else radial,
                "state": np.asarray(seq[:n]),
            }

        dominance: list[dict[str, Any]] = []
        bins: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for sid, animals in per_session.items():
            group = group_map.get(sid, "")
            rows: list[dict[str, Any]] = []
            for aid, d in animals.items():
                in_inter = np.isin(d["state"], inter) if inter else d["state"] >= 0
                r = d["radial_bl"][in_inter]
                r = r[np.isfinite(r)]
                rows.append({
                    "animal_id": aid,
                    "session_id": sid,
                    "group": group,
                    "interaction_time_s": (r.size / fps) if fps > 0 else 0.0,
                    "mean_advance": float(np.mean(np.clip(r, -5.0, 5.0))) if r.size else 0.0,
                    "advance_fraction": float(np.mean(r > thr)) if r.size else 0.0,
                    "yield_fraction": float(np.mean(r < -thr)) if r.size else 0.0,
                    "displacements_won": float("nan"),
                    "displacements_lost": float("nan"),
                    "displacement_p": float("nan"),
                    "method": "unpaired",
                    "dominance_score": (
                        float(np.mean(r > thr) - np.mean(r < -thr)) if r.size else float("nan")
                    ),
                })

            if len(animals) == 2:
                (a_id, da), (b_id, db) = sorted(animals.items())
                self._score_dyad(
                    sid, a_id, da, b_id, db, rows, bins, events,
                    inter=inter, thr=thr, min_frames=min_frames, fps=fps, bin_s=bin_s,
                )

            ranked = sorted(
                rows,
                key=lambda r: (r["dominance_score"] if np.isfinite(r["dominance_score"]) else -np.inf),
                reverse=True,
            )
            for i, r in enumerate(ranked, start=1):
                r["dominance_rank"] = i
                runner_up = ranked[1]["dominance_score"] if len(ranked) > 1 else float("nan")
                r["is_dominant"] = bool(
                    i == 1 and len(ranked) > 1 and np.isfinite(r["dominance_score"])
                    and (not np.isfinite(runner_up) or r["dominance_score"] > runner_up)
                )
                dominance.append(r)
        for b in bins:
            b["group"] = group_map.get(b["session_id"], "")
        return {"dominance": dominance, "bins": bins, "events": events}

    @staticmethod
    def _median_body_length(g: pd.DataFrame) -> float:
        """Subject's median body length in px, from the raw vs normalized distance."""
        if _DIST_PX_COL not in g or _PROXIMITY_COL not in g:
            return float("nan")
        px = pd.to_numeric(g[_DIST_PX_COL], errors="coerce").to_numpy(dtype=float)
        nm = pd.to_numeric(g[_PROXIMITY_COL], errors="coerce").to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            bl = px / nm
        bl = bl[np.isfinite(bl) & (bl > 0)]
        return float(np.median(bl)) if bl.size else float("nan")

    @staticmethod
    def _score_dyad(
        sid: str,
        a_id: str, da: dict[str, np.ndarray],
        b_id: str, db: dict[str, np.ndarray],
        rows: list[dict[str, Any]],
        bins: list[dict[str, Any]],
        events: list[dict[str, Any]],
        *, inter: list[int], thr: float, min_frames: int, fps: float, bin_s: float,
    ) -> None:
        """Pair the two animals frame by frame and count displacements."""
        common, ia, ib = np.intersect1d(da["frames"], db["frames"], return_indices=True)
        if common.size == 0:
            return
        ra, rb = da["radial_bl"][ia], db["radial_bl"][ib]
        sa, sb = da["state"][ia], db["state"][ib]
        if inter:
            in_inter = np.isin(sa, inter) | np.isin(sb, inter)
        else:
            in_inter = (sa >= 0) | (sb >= 0)
        with np.errstate(invalid="ignore"):
            a_wins = in_inter & (ra > thr) & (rb < -thr)
            b_wins = in_inter & (rb > thr) & (ra < -thr)

        period = 1.0 / fps if fps > 0 else 1.0
        first = int(common[0])
        bin_frames = max(1, int(round(bin_s * fps))) if fps > 0 and bin_s > 0 else 0
        won = {a_id: 0, b_id: 0}
        per_bin: dict[int, dict[str, int]] = {}
        for winner, loser, mask in ((a_id, b_id, a_wins), (b_id, a_id, b_wins)):
            for s, e in _runs(mask):
                if e - s < min_frames:
                    continue
                won[winner] += 1
                start_f = int(common[s])
                events.append({
                    "session_id": sid,
                    "winner": winner,
                    "loser": loser,
                    "start_frame": start_f,
                    "end_frame": int(common[e - 1]),
                    "start_s": start_f * period,
                    "duration_s": (e - s) * period,
                })
                if bin_frames:
                    bi = (start_f - first) // bin_frames
                    per_bin.setdefault(bi, {a_id: 0, b_id: 0})[winner] += 1

        total = won[a_id] + won[b_id]
        for r in rows:
            me = r["animal_id"]
            if me not in won:
                continue
            other = b_id if me == a_id else a_id
            r["displacements_won"] = float(won[me])
            r["displacements_lost"] = float(won[other])
            r["displacement_p"] = _binom_p(won[me], total)
            r["method"] = "paired"
            r["dominance_score"] = (
                (won[me] - won[other]) / total if total > 0 else float("nan")
            )

        if bin_frames:
            n_bins = int((int(common[-1]) - first) // bin_frames) + 1
            for bi in range(n_bins):
                c = per_bin.get(bi, {a_id: 0, b_id: 0})
                t = c[a_id] + c[b_id]
                for me, other in ((a_id, b_id), (b_id, a_id)):
                    bins.append({
                        "session_id": sid,
                        "animal_id": me,
                        "bin": bi,
                        "bin_start_s": bi * bin_frames * period,
                        "won": c[me],
                        "lost": c[other],
                        "dominance_index": (c[me] - c[other]) / t if t > 0 else float("nan"),
                    })

    # ── Cohort-level checks ──────────────────────────────────────────────

    @staticmethod
    def identity_rank_bias(dominance: list[dict[str, Any]]) -> dict[str, Any]:
        """How often each track id ranks first across dyad sessions.

        Track ids are assigned by the tracker, not by the experimenter, so a
        real hierarchy should not favor one id.  A lopsided count points at an
        identity-assignment artifact (or a design where the same id is always
        the resident).
        """
        sessions: dict[str, list[dict[str, Any]]] = {}
        for r in dominance:
            if r.get("method") == "paired":
                sessions.setdefault(str(r["session_id"]), []).append(r)
        ids = sorted({str(r["animal_id"]) for rows in sessions.values() for r in rows})
        counts = {i: 0 for i in ids}
        n = 0
        for rows in sessions.values():
            top = [r for r in rows if r.get("is_dominant")]
            if top:
                counts[str(top[0]["animal_id"])] += 1
                n += 1
        p = float("nan")
        if len(ids) == 2 and n > 0:
            p = _binom_p(counts[ids[0]], n)
        return {"counts": counts, "n_sessions": n, "p": p}

    @staticmethod
    def group_steepness(dominance: list[dict[str, Any]]) -> dict[str, Any]:
        """Hierarchy steepness (|dominance index| per dyad) compared across groups."""
        per_session: dict[str, tuple[str, float]] = {}
        for r in dominance:
            if r.get("method") != "paired" or not r.get("is_dominant"):
                continue
            v = float(r.get("dominance_score", np.nan))
            if np.isfinite(v):
                per_session[str(r["session_id"])] = (str(r.get("group", "")), abs(v))
        by_group: dict[str, list[float]] = {}
        for g, v in per_session.values():
            by_group.setdefault(g, []).append(v)
        summary = {
            g: {
                "n": len(v),
                "mean": float(np.mean(v)),
                "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"),
            }
            for g, v in by_group.items()
        }
        test, p = SocialAnalysisService.compare_groups(list(by_group.values()))
        return {"by_group": by_group, "summary": summary, "test": test, "p": p}

    @staticmethod
    def clean_prechop(prechop_frames: dict[str, int] | None) -> dict[str, int]:
        """Positive integer prechops keyed by session id, sorted for hashing."""
        out: dict[str, int] = {}
        for sid, v in (prechop_frames or {}).items():
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if n > 0:
                out[str(sid)] = n
        return dict(sorted(out.items()))

    @staticmethod
    def trim_prechop(df: pd.DataFrame, prechop: dict[str, int]) -> pd.DataFrame:
        """Drop each session's frames before its prechop."""
        if not prechop or df.empty or "frame" not in df or "session_id" not in df:
            return df
        start = df["session_id"].astype(str).map(prechop).fillna(0).to_numpy()
        keep = df["frame"].to_numpy() >= start
        return df if keep.all() else df.loc[keep].reset_index(drop=True)

    @staticmethod
    def compare_proportions(table: np.ndarray) -> tuple[str, float]:
        """Test a groups x outcomes count table: Fisher's exact for 2x2,
        chi-square otherwise. ``("", nan)`` if untestable."""
        t = np.asarray(table, dtype=float)
        t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0] if t.size else t
        if t.ndim != 2 or t.shape[0] < 2 or t.shape[1] < 2:
            return "", float("nan")
        try:
            from scipy.stats import chi2_contingency, fisher_exact
            if t.shape == (2, 2):
                return "Fisher's exact", float(fisher_exact(t.astype(int))[1])
            return "Chi-square", float(chi2_contingency(t)[1])
        except Exception:
            return "", float("nan")

    @staticmethod
    def compare_groups(samples: list[Any]) -> tuple[str, float]:
        """Mann-Whitney U (2 groups) or Kruskal-Wallis (3+); ``("", nan)`` if untestable."""
        usable = [np.asarray(v, dtype=float) for v in samples]
        if len(usable) < 2 or any(v.size < 2 for v in usable):
            return "", float("nan")
        try:
            from scipy.stats import kruskal, mannwhitneyu
            if len(usable) == 2:
                return "Mann-Whitney U", float(
                    mannwhitneyu(*usable, alternative="two-sided").pvalue
                )
            return "Kruskal-Wallis", float(kruskal(*usable).pvalue)
        except Exception:
            return "", float("nan")

    # ── Persistence ──────────────────────────────────────────────────────

    def input_fingerprint(self, project_root: Path, settings: dict[str, Any]) -> str:
        """Hash of the frame table's size/mtime and the fit settings."""
        path = self.frame_pose_path(project_root)
        try:
            st = path.stat()
            src = f"{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            src = "missing"
        blob = json.dumps({"src": src, "settings": settings}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def result_path(project_root: Path) -> Path:
        return project_root / "derived" / "analysis" / _RESULT_FILE

    def save_result(self, project_root: Path, result: dict[str, Any], fingerprint: str) -> None:
        path = self.result_path(project_root)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(
                    {"version": _RESULT_VERSION, "fingerprint": fingerprint, "result": result},
                    fh, protocol=pickle.HIGHEST_PROTOCOL,
                )
            tmp.replace(path)
        except Exception:
            logger.exception("Could not save the dominance HMM result to %s", path)

    def load_result(self, project_root: Path) -> tuple[dict[str, Any], str]:
        """Saved result and its fingerprint, or ``({}, "")``."""
        path = self.result_path(project_root)
        if not path.exists():
            return {}, ""
        try:
            with open(path, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("version") != _RESULT_VERSION:
                return {}, ""
            return dict(blob.get("result") or {}), str(blob.get("fingerprint", ""))
        except Exception:
            logger.exception("Could not read the saved dominance HMM result %s", path)
            return {}, ""


class _quiet_nanmean:
    """Silence numpy's all-NaN-slice RuntimeWarning for one block."""

    def __enter__(self) -> None:
        import warnings
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc: Any) -> None:
        self._cm.__exit__(*exc)
