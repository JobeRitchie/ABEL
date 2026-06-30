"""Service orchestration for temporal refinement via dense sliding-window inference.

Uses existing active-learning behavior models to score every frame at high
temporal resolution.  Overlapping window predictions are averaged per frame,
then a subtractive mutual-inhibition step penalises frames where multiple
behaviors are simultaneously likely.  The result is a calibrated probability
trace per behavior that feeds into the Temporal Review tab for threshold
tuning and bout extraction.

No model training occurs here — all models come from the active-learning
pipeline.
"""

from __future__ import annotations

import concurrent.futures as cf
import ctypes
import logging
import os
import pickle
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from abel.core.constants import APP_SCHEMA_VERSION
from abel.services.behavior_service import BehaviorService
from abel.services.behavior_representation_service import (
    BehaviorRepresentationService,
    canonical_distance_name,
)
from abel.services.import_service import ImportService
from abel.services.provenance_service import ProvenanceService
from abel.storage.file_store import read_json, write_json
from abel.temporal_refinement.bout_postprocess import (
    binary_trace_to_intervals,
    merge_close_bouts,
    remove_short_bouts,
    smooth_probabilities,
    threshold_probabilities,
)
from abel.temporal_refinement.temporal_metrics import (
    bout_level_metrics,
    frame_level_metrics,
    probability_histogram,
)

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    _HAS_XGB = False

logger = logging.getLogger("abel")


def _predict_via_dmatrix(model: Any, x: np.ndarray) -> np.ndarray:
    """Predict with a (possibly wrapped) XGBoost model using the DMatrix code path.

    Bypasses ``inplace_predict`` — the sklearn-wrapper path that triggers the
    "mismatched devices" warning and crashes on Windows when the booster lives on
    cuda:0 but the input array is a CPU numpy array.

    Supports:
    * Bare ``XGBClassifier`` (direct booster call).
    * ``CalibratedClassifierCV(cv='prefit', method='sigmoid'|'isotonic')``
      wrapping an ``XGBClassifier``.
    Falls back to ``model.predict_proba(x)`` for anything else.
    """
    cc_list = getattr(model, "calibrated_classifiers_", None)
    if cc_list is None:
        # Bare XGBClassifier
        booster = model.get_booster()
        dmat = xgb.DMatrix(x)  # type: ignore[union-attr]
        raw = booster.predict(dmat, output_margin=False)
        if raw.ndim == 1:
            raw = np.column_stack([1.0 - raw, raw])
        return raw

    # CalibratedClassifierCV — exactly 1 fold with cv='prefit'
    cc = cc_list[0]
    booster = cc.estimator.get_booster()
    dmat = xgb.DMatrix(x)  # type: ignore[union-attr]
    # DMatrix-based predict runs the GPU booster without inplace_predict;
    # avoids the device-mismatch fallback that crashes on Windows.
    raw_proba = booster.predict(dmat, output_margin=False)
    if raw_proba.ndim == 1:
        raw_proba = np.column_stack([1.0 - raw_proba, raw_proba])

    n_classes = raw_proba.shape[1]
    calibrated = np.zeros_like(raw_proba)

    if n_classes == 2 and len(cc.calibrators) == 1:
        # Binary classification: sklearn stores a single calibrator that maps
        # the positive-class (column 1) raw probability to a calibrated value;
        # the negative class (column 0) is the complement.
        calibrated[:, 1] = cc.calibrators[0].predict(raw_proba[:, 1])
        calibrated[:, 0] = 1.0 - calibrated[:, 1]
    else:
        for j, calibrator in enumerate(cc.calibrators):
            if j < n_classes:
                calibrated[:, j] = calibrator.predict(raw_proba[:, j])
        total = calibrated.sum(axis=1, keepdims=True)
        total = np.where(total == 0.0, 1.0, total)
        calibrated = calibrated / total

    return calibrated


def _contains_xgboost(estimator: Any) -> bool:
    """Return True if *estimator* is or wraps an XGBoost classifier.

    Handles CalibratedClassifierCV, Pipeline, and similar sklearn wrappers.
    """
    cls = type(estimator).__name__.lower()
    if "xgb" in cls or "xgboost" in cls:
        return True
    # CalibratedClassifierCV
    for attr in ("estimator", "base_estimator"):
        inner = getattr(estimator, attr, None)
        if inner is not None and _contains_xgboost(inner):
            return True
    # sklearn Pipeline
    steps = getattr(estimator, "steps", None)
    if isinstance(steps, list):
        for _name, step in steps:
            if step is not None and _contains_xgboost(step):
                return True
    return False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TemporalRefinementConfig:
    """Simplified config for inference-only temporal refinement.

    All model training happens in the active-learning pipeline.  This config
    controls only how those models are applied densely and how the resulting
    per-frame probabilities are shaped.
    """

    # -- Model selection --
    selected_behavior_models: dict[str, str] = field(default_factory=dict)
    excluded_behavior_ids: list[str] = field(default_factory=list)

    # -- Dense inference --
    inference_step_seconds: float = 0.10
    inference_warmup_seconds: float = 1.5
    inference_parallel_enabled: bool = True
    inference_max_workers: int = 0  # 0 = auto

    # -- Mutual inhibition --
    inhibition_weight: float = 0.20
    suppression_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-pair weights: {suppressed_id: {suppressor_id: weight}}.
    # When non-empty, replaces the global inhibition_weight for every pair
    # that has an explicit entry; pairs not listed fall back to 0.

    # -- Probability calibration --
    probability_temperature: float = 1.0

    # -- Postprocess (bout extraction) --
    smoothing_method: str = "moving_average"
    smoothing_window: int = 5
    onset_threshold: float = 0.50
    offset_threshold: float | None = None
    min_bout_duration_frames: int = 6
    merge_gap_frames: int = 3


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TemporalRefinementService:
    """Dense temporal refinement using existing active-learning models."""

    def __init__(self) -> None:
        self._project_root: Path | None = None
        self._imports = ImportService()
        self._behaviors = BehaviorService()
        self._provenance = ProvenanceService()

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._behaviors.set_project(project_root)

    def _require_project_root(self) -> Path:
        if self._project_root is None:
            raise ValueError("No project loaded")
        return self._project_root

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in str(value).strip()
        )
        return safe or "target_behavior"

    @staticmethod
    def _is_no_behavior_label(label: str) -> bool:
        token = re.sub(r"[^a-z0-9]+", "_", str(label or "").strip().lower()).strip("_")
        return token in {
            "no_behavior", "no_behaviour", "nobehavior", "nobehaviour",
        }

    @staticmethod
    def _emit(cb: Callable[[str], None] | None, msg: str) -> None:
        if cb is not None:
            cb(msg)

    @staticmethod
    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        cleaned = sorted(
            (min(int(s), int(e)), max(int(s), int(e))) for s, e in intervals
        )
        merged: list[tuple[int, int]] = [cleaned[0]]
        for s, e in cleaned[1:]:
            ps, pe = merged[-1]
            if s <= pe + 1:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    @staticmethod
    def _intervals_to_binary(
        intervals: list[tuple[int, int]], n_frames: int
    ) -> np.ndarray:
        out = np.zeros(max(0, int(n_frames)), dtype=int)
        for s, e in intervals:
            lo, hi = max(0, int(s)), min(len(out) - 1, int(e))
            if hi >= lo:
                out[lo : hi + 1] = 1
        return out

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fps_by_session(manifest) -> dict[str, float]:
        video_by_id = {v.asset_id: v for v in manifest.videos}
        out: dict[str, float] = {}
        for session in manifest.linked_sessions:
            sid = str(session.session_id)
            video = video_by_id.get(session.video_asset_id)
            fps = float(video.fps) if video and video.fps else 30.0
            out[sid] = max(1.0, fps)
        return out

    @staticmethod
    def _subject_by_session(manifest) -> dict[str, str]:
        video_by_id = {v.asset_id: v for v in manifest.videos}
        out: dict[str, str] = {}
        for session in manifest.linked_sessions:
            subject = (session.subject_id or "").strip()
            if not subject:
                video = video_by_id.get(session.video_asset_id)
                subject = (video.subject_id if video else "") or ""
            out[str(session.session_id)] = subject.strip() or str(session.session_id)
        return out

    # ------------------------------------------------------------------
    # Behavior model resolution
    # ------------------------------------------------------------------

    def _resolve_competition_model_versions(
        self, cfg: TemporalRefinementConfig
    ) -> dict[str, str]:
        project_root = self._require_project_root()
        models_root = project_root / "derived" / "models"
        selected_map = {
            str(k): str(v)
            for k, v in dict(cfg.selected_behavior_models or {}).items()
            if str(v).strip()
        }
        excluded = {
            str(v).strip()
            for v in list(cfg.excluded_behavior_ids or [])
            if str(v).strip()
        }

        def _auto_latest(bid: str) -> str:
            # Strategy 1: match by naming convention  behavior_model_{safe_bid}_v1*
            prefix = f"behavior_model_{self._safe_name(bid)}_v1"
            candidates: list[Path] = []
            if models_root.exists():
                exact = models_root / prefix
                if (exact / "model_state.pkl").exists():
                    candidates.append(exact)
                for p in models_root.glob(f"{prefix}_*"):
                    if p.is_dir() and (p / "model_state.pkl").exists():
                        candidates.append(p)
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return str(candidates[0].name)

            # Strategy 2: scan all model dirs for matching target_behavior
            # in run_settings.json (handles dirs named by behaviour name
            # rather than UUID, e.g. behavior_model_Freeze).
            if models_root.exists():
                for p in models_root.iterdir():
                    if not p.is_dir() or not p.name.startswith("behavior_model_"):
                        continue
                    if not (p / "model_state.pkl").exists():
                        continue
                    settings = read_json(p / "run_settings.json", {})
                    tb = str(settings.get("target_behavior", "")).strip()
                    if tb == bid:
                        candidates.append(p)
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    return str(candidates[0].name)
            return ""

        out: dict[str, str] = {}
        for behavior in self._behaviors.behaviors:
            bid = str(behavior.behavior_id or "").strip()
            if not bid or bid in excluded:
                continue
            # Skip no_behavior in auto-resolution unless explicitly selected.
            if self._is_no_behavior_label(bid) and bid not in selected_map:
                continue
            chosen = selected_map.get(bid, "").strip()
            if chosen:
                model_dir = models_root / chosen
                if (model_dir / "model_state.pkl").exists():
                    out[bid] = chosen
                    continue
            auto = _auto_latest(bid)
            if auto:
                out[bid] = auto

        for bid, model_version in selected_map.items():
            if bid in excluded or bid in out:
                continue
            model_dir = models_root / model_version
            if (model_dir / "model_state.pkl").exists():
                out[bid] = model_version
        return out

    def _infer_behavior_window_frames(self) -> int:
        project_root = self._require_project_root()
        seg_path = project_root / "derived" / "representations" / "segment_features.parquet"
        if not seg_path.exists():
            return 30
        seg_df = pd.read_parquet(seg_path, columns=["start_frame", "end_frame"])
        if seg_df.empty:
            return 30
        lengths = seg_df["end_frame"].to_numpy(dtype=int) - seg_df["start_frame"].to_numpy(dtype=int) + 1
        lengths = lengths[lengths > 0]
        return int(max(1, int(np.median(lengths)))) if lengths.size else 30

    # ------------------------------------------------------------------
    # Frame features
    # ------------------------------------------------------------------

    def _load_representation_frame_features(
        self,
        expected_session_ids: set[str] | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        project_root = self._require_project_root()
        pose_path = project_root / "derived" / "pose_features" / "frame_pose.parquet"
        ctx_path = project_root / "derived" / "context_features" / "frame_context.parquet"

        def _source_sig() -> dict[str, Any]:
            sig: dict[str, Any] = {}
            for key, path in (("pose", pose_path), ("context", ctx_path)):
                if not path.exists():
                    sig[key] = {"exists": False}
                    continue
                st = path.stat()
                sig[key] = {"exists": True, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
            return sig

        source_signature = _source_sig()
        cache_dir = self._temporal_tab_cache_root() / "frame_representations"
        cache_path = cache_dir / "frame_features.parquet"
        cache_meta_path = cache_dir / "meta.json"
        id_cols = {"frame", "session_id", "animal_id", "video_id"}

        def _to_grouped(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[str]]:
            if df.empty:
                return {}, []
            feat = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
            if not feat:
                return {}, []
            by_s: dict[str, pd.DataFrame] = {}
            for sid, grp in df.groupby("session_id"):
                by_s[str(sid)] = grp.sort_values("frame").reset_index(drop=True)
            return by_s, feat

        # Prefer the canonical representation cache first.  Since P1 it is keyed
        # on source *content* (not mtime), so it is stable across re-extracts and
        # is the same z-scored frame table the trainer uses.  Reusing it directly
        # avoids both a redundant raw reload + re-z-score and a duplicate copy of
        # the frame features under the temporal-refinement tab cache.
        frame_path = project_root / "derived" / "representations" / "frame_features.parquet"
        if frame_path.exists():
            frame_df = pd.read_parquet(frame_path)
            by_s, feat_cols = _to_grouped(frame_df)
            if by_s and feat_cols:
                if not expected_session_ids or expected_session_ids.issubset(set(by_s)):
                    self._emit(
                        progress_cb,
                        f"Loaded representation frame features: {len(by_s)} sessions, {len(feat_cols)} features",
                    )
                    return by_s, feat_cols

        if cache_path.exists() and cache_meta_path.exists():
            meta = read_json(cache_meta_path, {})
            if dict(meta.get("source_signature") or {}) == source_signature:
                cached_df = pd.read_parquet(cache_path)
                by_c, cols_c = _to_grouped(cached_df)
                if by_c and cols_c:
                    if not expected_session_ids or expected_session_ids.issubset(set(by_c)):
                        self._emit(progress_cb, f"Loaded cached frame features: {len(by_c)} sessions, {len(cols_c)} features")
                        return by_c, cols_c

        if not pose_path.exists():
            raise FileNotFoundError("Missing derived/pose_features/frame_pose.parquet. Run feature extraction first.")
        pose_df = pd.read_parquet(pose_path)
        ctx_df = pd.read_parquet(ctx_path) if ctx_path.exists() else None
        join_cols = ["frame", "session_id"]
        if "animal_id" in pose_df.columns and ctx_df is not None and "animal_id" in ctx_df.columns:
            join_cols.append("animal_id")
        merged = pose_df.merge(ctx_df, on=join_cols, how="inner", suffixes=("_pose", "_ctx")) if ctx_df is not None else pose_df.copy()
        feature_cols = [c for c in merged.columns if c not in id_cols and pd.api.types.is_numeric_dtype(merged[c])]
        if not feature_cols:
            raise ValueError("No numeric frame features available")

        repr_builder = BehaviorRepresentationService()
        merged = repr_builder._zscore_by_group(merged, feature_cols)

        by_session: dict[str, pd.DataFrame] = {}
        for sid, grp in merged.groupby("session_id"):
            by_session[str(sid)] = grp.sort_values("frame").reset_index(drop=True)
        if not by_session:
            raise ValueError("Frame features are empty")

        cache_dir.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_path, index=False)
        write_json(cache_meta_path, {
            "source_signature": source_signature,
            "n_sessions": len(by_session),
            "n_features": len(feature_cols),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        # Frame features are the inner join of pose and context, so any session
        # with pose but no context is dropped. Make that visible — a partial
        # context coverage is the usual reason temporal refinement "sees" fewer
        # sessions than expected.
        if ctx_df is not None:
            pose_sessions = (
                set(pose_df["session_id"].astype(str).unique())
                if "session_id" in pose_df.columns else set()
            )
            dropped = sorted(pose_sessions - set(by_session))
            if dropped:
                self._emit(
                    progress_cb,
                    f"Note: {len(dropped)} session(s) have pose but no context "
                    "features and were excluded (temporal refinement requires "
                    f"both): {self._fmt_session_list(dropped)}",
                )

        self._emit(progress_cb, f"Built frame feature cache: {len(by_session)} sessions, {len(feature_cols)} features")
        return by_session, feature_cols

    # ------------------------------------------------------------------
    # Session availability diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _parquet_session_ids(path: Path) -> set[str]:
        """Read just the session_id column from a parquet (empty set if absent)."""
        if not path.exists():
            return set()
        try:
            df = pd.read_parquet(path, columns=["session_id"])
        except Exception:
            return set()
        return set(df["session_id"].astype(str).unique())

    def _classify_requested_sessions(
        self, requested: list[str], frame_by_session: dict[str, pd.DataFrame],
    ) -> tuple[list[str], list[str], list[str]]:
        """Split requested session ids into (present, missing_context, missing_features).

        * present          — have usable frame features (pose ∩ context).
        * missing_context  — have pose but no context features (dropped by the join).
        * missing_features — have neither (no frame features at all).
        """
        project_root = self._require_project_root()
        pose_ids = self._parquet_session_ids(
            project_root / "derived" / "pose_features" / "frame_pose.parquet"
        )
        ctx_ids = self._parquet_session_ids(
            project_root / "derived" / "context_features" / "frame_context.parquet"
        )
        present: list[str] = []
        missing_context: list[str] = []
        missing_features: list[str] = []
        for s in requested:
            sid = str(s)
            if sid in frame_by_session:
                present.append(sid)
            elif sid in pose_ids and sid not in ctx_ids:
                missing_context.append(sid)
            else:
                missing_features.append(sid)
        return present, sorted(missing_context), sorted(missing_features)

    @staticmethod
    def _fmt_session_list(sids: list[str], limit: int = 10) -> str:
        shown = ", ".join(sids[:limit])
        if len(sids) > limit:
            shown += f", … (+{len(sids) - limit} more)"
        return shown

    def _no_target_sessions_message(
        self,
        had_selection: bool,
        frame_by_session: dict[str, pd.DataFrame],
        missing_context: list[str],
        missing_features: list[str],
    ) -> str:
        """Build an actionable error for the 'nothing to run' case."""
        if missing_context and not missing_features:
            return (
                "No sessions to run: every selected session is missing CONTEXT "
                "features. Temporal refinement scores frame-level pose AND context "
                "features, but these sessions have pose only. Re-run context-feature "
                "extraction (define the arena / zone ROIs for these videos), "
                "re-extract representations, then retry. Sessions missing context: "
                f"{self._fmt_session_list(missing_context, limit=20)}."
            )
        if missing_features and not missing_context:
            return (
                "No sessions to run: no frame (pose/context) features were found "
                "for the selected sessions. Run feature extraction for them first. "
                f"Sessions: {self._fmt_session_list(missing_features, limit=20)}."
            )
        if missing_context or missing_features:
            return (
                "No sessions to run: none of the selected sessions have usable "
                "frame features. Missing context: "
                f"{self._fmt_session_list(missing_context, limit=20)}; missing all "
                f"frame features: {self._fmt_session_list(missing_features, limit=20)}."
            )
        if not frame_by_session:
            return (
                "No sessions have both pose and context features available. Run "
                "pose and context feature extraction before temporal refinement."
            )
        return (
            "No target sessions to run. "
            f"{len(frame_by_session)} session(s) have frame features available; "
            "none matched the current selection."
        )

    # ------------------------------------------------------------------
    # Dense window construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dense_windows_for_session(
        session_df: pd.DataFrame,
        feature_cols: list[str],
        window_frames: int,
        step_frames: int,
        include_posture_deltas: bool = False,
    ) -> pd.DataFrame:
        n_frames = len(session_df)
        if n_frames <= 0:
            return pd.DataFrame()
        win = max(1, int(window_frames))
        step = max(1, int(step_frames))

        animal_id = str(session_df["animal_id"].iloc[0]) if "animal_id" in session_df.columns else ""
        session_id = str(session_df["session_id"].iloc[0]) if "session_id" in session_df.columns else ""

        # GPU-accelerated path: compute all windowed statistics in a single
        # batched tensor operation (same strategy as the active-learning
        # representation builder).
        from abel.utils.gpu_feature_ops import build_segment_df_fast

        result = build_segment_df_fast(
            session_df,
            feature_cols,
            animal_id=animal_id,
            session_id=session_id,
            window_size=win,
            stride=step,
            include_periodicity=True,
            include_posture_deltas=include_posture_deltas,
        )

        if result.empty:
            # Fallback for edge case where n_frames < window
            if n_frames <= win:
                from abel.services.behavior_representation_service import BehaviorRepresentationService as _BRS
                window = session_df.iloc[:n_frames].copy()
                window["animal_id"] = animal_id
                window["session_id"] = session_id
                seg_id = f"dense_{session_id}_0_{n_frames - 1}"
                summary = _BRS._segment_summary(window, feature_cols, seg_id)
                return pd.DataFrame([summary])
            return pd.DataFrame()

        # Rename segment_id prefix to match the dense naming convention
        result["segment_id"] = [
            f"dense_{session_id}_{s}_{e}"
            for s, e in zip(result["start_frame"], result["end_frame"])
        ]
        return result

    # ------------------------------------------------------------------
    # Worker concurrency
    # ------------------------------------------------------------------

    @staticmethod
    def _available_memory_bytes() -> int | None:
        try:
            if os.name == "nt":
                class _MEMSTATUS(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                status = _MEMSTATUS()
                status.dwLength = ctypes.sizeof(_MEMSTATUS)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    return int(status.ullAvailPhys)
        except Exception:
            pass
        return None

    def _resolve_inference_workers(
        self, cfg: TemporalRefinementConfig, n_sessions: int,
        progress_cb: Callable[[str], None] | None = None,
    ) -> int:
        if not cfg.inference_parallel_enabled or n_sessions <= 1:
            return 1
        cpu_cap = min(8, max(1, (os.cpu_count() or 4) - 1))
        requested = int(cfg.inference_max_workers or 0)
        workers = min(n_sessions, cpu_cap if requested <= 0 else max(1, requested))
        self._emit(progress_cb, f"Inference workers: {workers} (cpu_cap={cpu_cap}, sessions={n_sessions})")
        return max(1, workers)

    # ------------------------------------------------------------------
    # Artifact paths
    # ------------------------------------------------------------------

    def _artifact_root(self, concept_id: str) -> Path:
        root = self._require_project_root() / "derived" / "temporal_refinement" / self._safe_name(concept_id)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _temporal_tab_cache_root(self) -> Path:
        root = self._require_project_root() / "derived" / "temporal_refinement" / "tab_cache"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _latest_path(self, concept_id: str) -> Path:
        return self._artifact_root(concept_id) / "latest.json"

    def _write_latest(self, concept_id: str, payload: dict[str, Any]) -> None:
        write_json(self._latest_path(concept_id), payload)

    def _load_latest(self, concept_id: str) -> dict[str, Any]:
        return read_json(self._latest_path(concept_id), {})

    def _feedback_path(self, concept_id: str) -> Path:
        return self._artifact_root(concept_id) / "feedback_intervals.json"

    # ------------------------------------------------------------------
    # Feedback management
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_interval_map(raw: Any) -> dict[str, list[tuple[int, int]]]:
        out: dict[str, list[tuple[int, int]]] = {}
        if not isinstance(raw, dict):
            return out
        for sid, intervals in raw.items():
            if not isinstance(intervals, list):
                continue
            cleaned: list[tuple[int, int]] = []
            for item in intervals:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                try:
                    s, e = int(item[0]), int(item[1])
                except Exception:
                    continue
                if e < s:
                    s, e = e, s
                cleaned.append((s, e))
            if cleaned:
                out[str(sid)] = TemporalRefinementService._merge_intervals(cleaned)
        return out

    def load_temporal_feedback(self, concept_id: str) -> dict[str, Any]:
        payload = read_json(self._feedback_path(concept_id), {})
        fp = self._normalize_interval_map(payload.get("false_positive_intervals_by_session", {}))
        fn = self._normalize_interval_map(payload.get("false_negative_intervals_by_session", {}))
        return {
            "concept_id": str(concept_id),
            "false_positive_intervals_by_session": fp,
            "false_negative_intervals_by_session": fn,
            "n_false_positive_intervals": sum(len(v) for v in fp.values()),
            "n_false_negative_intervals": sum(len(v) for v in fn.values()),
            "path": str(self._feedback_path(concept_id)),
        }

    def add_temporal_feedback_interval(
        self, concept_id: str, session_id: str,
        start_frame: int, end_frame: int, feedback_type: str,
    ) -> dict[str, Any]:
        ftype = str(feedback_type or "").strip().lower()
        if ftype not in {"false_positive", "false_negative"}:
            raise ValueError("feedback_type must be 'false_positive' or 'false_negative'")
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        s, e = int(start_frame), int(end_frame)
        if e < s:
            s, e = e, s

        current = self.load_temporal_feedback(concept_id)
        fp = dict(current.get("false_positive_intervals_by_session", {}) or {})
        fn = dict(current.get("false_negative_intervals_by_session", {}) or {})
        target = fn if ftype == "false_negative" else fp
        existing = list(target.get(sid, []))
        existing.append((s, e))
        target[sid] = self._merge_intervals(existing)

        write_json(self._feedback_path(concept_id), {
            "concept_id": str(concept_id),
            "updated_utc": datetime.utcnow().isoformat(),
            "false_positive_intervals_by_session": {k: [[a, b] for a, b in v] for k, v in fp.items()},
            "false_negative_intervals_by_session": {k: [[a, b] for a, b in v] for k, v in fn.items()},
        })
        return self.load_temporal_feedback(concept_id)

    def remove_temporal_feedback_interval(
        self, concept_id: str, session_id: str,
        start_frame: int, end_frame: int, feedback_type: str,
    ) -> dict[str, Any]:
        """Remove a previously stored feedback interval (exact match)."""
        ftype = str(feedback_type or "").strip().lower()
        if ftype not in {"false_positive", "false_negative"}:
            raise ValueError("feedback_type must be 'false_positive' or 'false_negative'")
        sid = str(session_id or "").strip()
        s, e = int(start_frame), int(end_frame)
        if e < s:
            s, e = e, s

        current = self.load_temporal_feedback(concept_id)
        fp = dict(current.get("false_positive_intervals_by_session", {}) or {})
        fn = dict(current.get("false_negative_intervals_by_session", {}) or {})
        target = fn if ftype == "false_negative" else fp
        existing = list(target.get(sid, []))
        # Remove the interval that contains the requested range
        remaining = [iv for iv in existing if not (iv[0] == s and iv[1] == e)]
        if remaining:
            target[sid] = remaining
        else:
            target.pop(sid, None)

        write_json(self._feedback_path(concept_id), {
            "concept_id": str(concept_id),
            "updated_utc": datetime.utcnow().isoformat(),
            "false_positive_intervals_by_session": {k: [[a, b] for a, b in v] for k, v in fp.items()},
            "false_negative_intervals_by_session": {k: [[a, b] for a, b in v] for k, v in fn.items()},
        })
        return self.load_temporal_feedback(concept_id)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_temporal_tab_cache(
        self, concept_id: str | None = None, clear_run_artifacts: bool = False,
    ) -> dict[str, Any]:
        root = self._temporal_tab_cache_root()
        cleared_files = cleared_dirs = cleared_run_dirs = cleared_latest = 0
        if root.exists():
            for child in list(root.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                    cleared_dirs += 1
                else:
                    try:
                        child.unlink()
                        cleared_files += 1
                    except Exception:
                        pass

        if clear_run_artifacts:
            project_root = self._require_project_root()
            tr_root = project_root / "derived" / "temporal_refinement"
            targets: list[Path] = []
            if concept_id and str(concept_id).strip():
                targets = [self._artifact_root(str(concept_id).strip())]
            elif tr_root.exists():
                targets = [p for p in tr_root.iterdir() if p.is_dir() and p.name != "tab_cache"]
            for concept_root in targets:
                if not concept_root.exists():
                    continue
                for run_dir in list(concept_root.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    if run_dir.name.startswith("inference_") or run_dir.name.startswith("postprocess_"):
                        shutil.rmtree(run_dir, ignore_errors=True)
                        cleared_run_dirs += 1
                latest_p = concept_root / "latest.json"
                if latest_p.exists():
                    latest = read_json(latest_p, {})
                    latest["inference_dir"] = ""
                    latest["inference_parameter_hash"] = ""
                    latest["postprocess_dir"] = ""
                    latest["postprocess_parameter_hash"] = ""
                    write_json(latest_p, latest)
                    cleared_latest += 1

        return {
            "ok": True,
            "cache_root": str(root),
            "removed_dirs": cleared_dirs,
            "removed_files": cleared_files,
            "removed_run_dirs": cleared_run_dirs,
            "updated_latest_files": cleared_latest,
            "message": "Temporal refinement cache cleared.",
        }

    # ==================================================================
    # INFERENCE — dense sliding-window with mutual inhibition
    # ==================================================================

    def run_temporal_refinement_inference(
        self,
        concept_id: str,
        sessions: list[str] | None = None,
        mode: str = "dense",
        config: TemporalRefinementConfig | None = None,
        force: bool = False,
        max_sessions: int | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Run dense temporal inference over full sessions.

        For each session:
        1. Build dense overlapping windows at ``inference_step_seconds``.
        2. Score each window with every available behavior model.
        3. Average overlapping window scores per chunk (center-only assignment).
        4. Apply probability-temperature calibration.
        5. Apply subtractive mutual inhibition across behaviors.
        6. Write per-frame probability traces.
        """
        cfg = config or TemporalRefinementConfig()
        self._emit(progress_cb, "Dense temporal inference started.")

        manifest = self._imports.load_manifest(self._require_project_root())
        manifest_sids: set[str] = set()
        if manifest is not None:
            manifest_sids = {str(s.session_id) for s in list(getattr(manifest, "linked_sessions", []) or [])}

        frame_by_session, repr_feature_cols = self._load_representation_frame_features(
            expected_session_ids=manifest_sids if not sessions else set(str(s) for s in sessions),
            progress_cb=progress_cb,
        )

        if not sessions:
            target_sessions = sorted(frame_by_session.keys())
            missing_context: list[str] = []
            missing_features: list[str] = []
        else:
            requested = [str(s) for s in sessions]
            present, missing_context, missing_features = self._classify_requested_sessions(
                requested, frame_by_session,
            )
            target_sessions = sorted(present)

        if max_sessions and max_sessions > 0:
            target_sessions = target_sessions[:max_sessions]
            self._emit(progress_cb, f"Test mode: limiting to {max_sessions} session(s).")
        self._emit(progress_cb, f"Sessions: {len(target_sessions)} target, {len(frame_by_session)} available.")

        # Surface sessions that were requested but cannot be processed, so a
        # partial or empty run is never silently mistaken for success.
        if missing_context:
            self._emit(
                progress_cb,
                "WARNING: "
                f"{len(missing_context)} selected session(s) skipped — they have "
                "pose but no CONTEXT features. Temporal refinement needs both; "
                "run context-feature extraction (with arena/zone ROIs defined) for "
                f"these, then retry: {self._fmt_session_list(missing_context)}",
            )
        if missing_features:
            self._emit(
                progress_cb,
                "WARNING: "
                f"{len(missing_features)} selected session(s) skipped — no frame "
                "(pose/context) features were found for them. Run feature "
                f"extraction first: {self._fmt_session_list(missing_features)}",
            )

        if not target_sessions:
            raise ValueError(self._no_target_sessions_message(
                bool(sessions), frame_by_session, missing_context, missing_features,
            ))

        model_versions = self._resolve_competition_model_versions(cfg)
        if not model_versions:
            raise ValueError("No behavior models available. Train at least one model in the active-learning pipeline first.")
        behavior_ids = list(model_versions.keys())
        self._emit(progress_cb, f"Behavior models: {', '.join(f'{bid}={ver}' for bid, ver in model_versions.items())}")

        fps_by_session = self._fps_by_session(manifest) if manifest else {}
        window_frames = self._infer_behavior_window_frames()
        step_seconds = max(0.01, float(cfg.inference_step_seconds))
        warmup_seconds = max(0.0, float(cfg.inference_warmup_seconds))
        inhibition_weight = float(np.clip(float(cfg.inhibition_weight), 0.0, 0.5))
        suppression_matrix: dict[str, dict[str, float]] = dict(cfg.suppression_matrix or {})
        prob_temp = max(0.1, float(cfg.probability_temperature))

        # Load all model payloads once
        project_root = self._require_project_root()
        model_payloads: dict[str, dict[str, Any]] = {}
        gpu_or_xgb = False
        for bid, ver in model_versions.items():
            state_path = project_root / "derived" / "models" / ver / "model_state.pkl"
            if not state_path.exists():
                raise FileNotFoundError(f"Model missing: {state_path}")
            with open(state_path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                model_payloads[bid] = payload
                model_obj = payload.get("model")
                if model_obj is not None:
                    is_xgb_model = _contains_xgboost(model_obj)
                    payload["_is_xgb"] = is_xgb_model
                    if is_xgb_model:
                        gpu_or_xgb = True

        # FP feedback suppression
        fp_by_session: dict[str, list[tuple[int, int]]] = {}
        try:
            feedback = self.load_temporal_feedback(concept_id)
            fp_by_session = dict(feedback.get("false_positive_intervals_by_session", {}) or {})
        except Exception:
            pass

        # Artifact directory
        artifact_root = self._artifact_root(concept_id)
        cfg_payload = asdict(cfg)
        cfg_payload.update({"concept_id": concept_id, "mode": "dense_inhibition", "behavior_models": model_versions, "window_frames": window_frames})
        parameter_hash = self._provenance.config_hash(cfg_payload)
        run_dir = artifact_root / f"inference_{parameter_hash}"

        if run_dir.exists() and not force:
            latest = self._load_latest(concept_id)
            if str(latest.get("inference_dir", "")) == str(run_dir):
                existing_manifest = read_json(run_dir / "inference_manifest.json", {})
                existing_sessions = set(str(k) for k in (existing_manifest.get("trace_paths", {}) or {}))
                if set(target_sessions).issubset(existing_sessions):
                    self._emit(progress_cb, f"Skipped inference: artifacts exist at {run_dir}.")
                    return {"status": "skipped", "reason": "matching artifacts already exist", "inference_dir": str(run_dir)}

        run_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = run_dir / "probability_traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        chunk_traces_dir = run_dir / "chunk_probability_traces"
        chunk_traces_dir.mkdir(parents=True, exist_ok=True)

        workers = self._resolve_inference_workers(cfg, len(target_sessions), progress_cb)
        if gpu_or_xgb and workers > 1:
            self._emit(progress_cb, "XGBoost/GPU model detected; reducing to 1 worker.")
            workers = 1

        trace_paths: dict[str, str] = {}
        chunk_trace_paths: dict[str, str] = {}
        step_frames_by_session: dict[str, int] = {}
        total_windows = total_frames = 0
        t0 = time.perf_counter()

        # Clip-wise posture deltas must match the training-time setting so that
        # dense inference windows produce the same feature columns the model
        # was trained on.
        from abel.models.schemas import InvariantFeatureConfig
        posture_deltas = InvariantFeatureConfig.load_from_project(
            self._require_project_root()
        ).enable_clipwise_deltas

        # --- Per-session inference ---
        def _run_session(sid: str) -> dict[str, Any] | None:
            ts = time.perf_counter()
            session_df = frame_by_session.get(sid)
            if session_df is None or session_df.empty:
                return None
            n_frames = len(session_df)
            fps = max(1.0, float(fps_by_session.get(sid, 30.0) or 30.0))
            step_f = max(1, int(round(step_seconds * fps)))

            dense_segs = self._build_dense_windows_for_session(
                session_df, repr_feature_cols, window_frames, step_f,
                include_posture_deltas=posture_deltas,
            )
            if dense_segs.empty:
                return None
            n_windows = len(dense_segs)
            starts = dense_segs["start_frame"].to_numpy(dtype=int)
            ends = dense_segs["end_frame"].to_numpy(dtype=int)

            n_chunks = int(np.ceil(n_frames / step_f))
            chunk_sum = {bid: np.zeros(n_chunks, dtype=np.float64) for bid in behavior_ids}
            chunk_count = {bid: np.zeros(n_chunks, dtype=np.float64) for bid in behavior_ids}

            # Pre-build input arrays for all models (one pass per behavior).
            x_by_bid: dict[str, np.ndarray] = {}
            active_bids: list[str] = []
            for bid in behavior_ids:
                payload = model_payloads.get(bid, {})
                model = payload.get("model")
                model_cols = [str(c) for c in list(payload.get("feature_cols", []))]
                if model is None or not model_cols:
                    continue
                # Align the model's stored feature names onto the data's canonical
                # pairwise-distance spelling.  Models trained before distance
                # canonicalisation (v0.5.2) stored ``dist_b_to_a`` while freshly
                # extracted features now use the sorted ``dist_a_to_b``; without this
                # remap every such column reindexes to a missing name and is silently
                # zero-filled, which is what broke Direct Use for older models.  The
                # map is 1:1 (a training set can't hold both spellings), so feature
                # order is preserved.
                aligned_cols = [canonical_distance_name(c) for c in model_cols]
                x_by_bid[bid] = dense_segs.reindex(
                    columns=aligned_cols, fill_value=0.0
                ).to_numpy(dtype=np.float32)
                active_bids.append(bid)

            def _predict_bid(bid: str) -> tuple[str, np.ndarray]:
                """Score one behavior model using GPU-native DMatrix prediction."""
                payload = model_payloads[bid]
                model = payload["model"]
                x = x_by_bid[bid]
                is_xgb = payload.get("_is_xgb", False)

                if is_xgb and _HAS_XGB:
                    # Use DMatrix-based booster.predict — avoids inplace_predict
                    # device-mismatch crash on Windows + CUDA.
                    try:
                        probs_raw = _predict_via_dmatrix(model, x)
                    except Exception:
                        probs_raw = model.predict_proba(x)
                else:
                    probs_raw = model.predict_proba(x)

                label_map = payload.get("label_map", {})
                class_idx: int | None = None
                if isinstance(label_map, dict):
                    for k, v in label_map.items():
                        if str(v) == str(bid):
                            try:
                                class_idx = int(k)
                            except Exception:
                                pass
                            break
                probs_arr = np.asarray(probs_raw, dtype=float)
                if class_idx is not None and probs_arr.ndim == 2 and 0 <= class_idx < probs_arr.shape[1]:
                    pred_prob = np.clip(probs_arr[:, class_idx], 0.0, 1.0)
                else:
                    pred_prob = np.clip(np.max(probs_arr, axis=1), 0.0, 1.0)
                return bid, pred_prob

            # Run predictions sequentially — XGBoost CUDA is not thread-safe
            # for concurrent calls on Windows; GPU CUDA streams are already used
            # efficiently within each booster.predict(DMatrix) call.
            pred_results: dict[str, np.ndarray] = {}
            for bid in active_bids:
                bid_res, pred_prob = _predict_bid(bid)
                pred_results[bid_res] = pred_prob

            # Full-overlap chunk assignment: each window contributes to ALL
            # chunks it covers, so overlapping predictions are averaged.
            # With window_frames=15 and step_f=3, each window spans ~5
            # chunks, giving each interior chunk ~5 averaged predictions
            # instead of the single center-only value that caused a
            # blocky staircase pattern in the probability trace.
            c_starts = np.clip(starts.astype(np.int64) // step_f, 0, max(0, n_chunks - 1))
            c_ends = np.clip(ends.astype(np.int64) // step_f, 0, max(0, n_chunks - 1))
            max_span = int((c_ends - c_starts).max()) + 1 if len(c_starts) else 1
            for offset in range(max_span):
                c_idx = c_starts + offset
                valid = c_idx <= c_ends
                if not valid.any():
                    break
                for bid, pred_prob in pred_results.items():
                    np.add.at(chunk_sum[bid], c_idx[valid], pred_prob[valid])
                    np.add.at(chunk_count[bid], c_idx[valid], 1.0)

            # Average overlapping predictions
            per_behavior_raw: dict[str, np.ndarray] = {}
            for bid in behavior_ids:
                denom = np.maximum(chunk_count[bid], 1e-6)
                per_behavior_raw[bid] = np.clip(chunk_sum[bid] / denom, 0.0, 1.0)

            # Temperature calibration
            if abs(prob_temp - 1.0) > 1e-6:
                for bid in per_behavior_raw:
                    raw = np.clip(per_behavior_raw[bid], 1e-9, 1.0 - 1e-9)
                    logit = np.log(raw / (1.0 - raw))
                    scaled = logit / prob_temp
                    per_behavior_raw[bid] = 1.0 / (1.0 + np.exp(-scaled))

            # Subtractive mutual inhibition:
            # When a per-pair suppression_matrix is provided, each
            # (suppressed, suppressor) pair uses its own weight.
            # Otherwise fall back to the global inhibition_weight applied
            # uniformly to every pair.
            per_behavior_prob: dict[str, np.ndarray] = {}
            use_matrix = bool(suppression_matrix) and len(behavior_ids) > 1
            use_global = (not use_matrix) and inhibition_weight > 0 and len(behavior_ids) > 1

            if use_matrix:
                for bid in behavior_ids:
                    row = suppression_matrix.get(bid, {})
                    suppressed = per_behavior_raw[bid].copy()
                    for other_bid in behavior_ids:
                        if other_bid == bid:
                            continue
                        w = float(row.get(other_bid, 0.0))
                        if w > 0:
                            suppressed = suppressed - w * per_behavior_raw[other_bid]
                    per_behavior_prob[bid] = np.clip(suppressed, 0.0, 1.0)
            elif use_global:
                total_prob = np.zeros(n_chunks, dtype=np.float64)
                for bid in behavior_ids:
                    total_prob += per_behavior_raw[bid]
                for bid in behavior_ids:
                    others_sum = total_prob - per_behavior_raw[bid]
                    inhibited = per_behavior_raw[bid] - inhibition_weight * others_sum
                    per_behavior_prob[bid] = np.clip(inhibited, 0.0, 1.0)
            else:
                per_behavior_prob = dict(per_behavior_raw)

            # Warmup suppression
            warmup_chunks = int(np.ceil(warmup_seconds * fps / step_f)) if warmup_seconds > 0 else 0
            warmup_chunks = min(n_chunks, max(0, warmup_chunks))
            if warmup_chunks > 0:
                for bid in per_behavior_prob:
                    per_behavior_prob[bid][:warmup_chunks] = 0.0

            # FP feedback suppression
            session_fp = fp_by_session.get(sid, [])
            if session_fp:
                for fp_s, fp_e in session_fp:
                    c_lo = max(0, int(fp_s) // step_f)
                    c_hi = min(n_chunks - 1, int(fp_e) // step_f)
                    if c_lo <= c_hi:
                        for bid in per_behavior_prob:
                            if not self._is_no_behavior_label(bid):
                                per_behavior_prob[bid][c_lo : c_hi + 1] = 0.0

            # Determine winner per chunk
            competition_labels = list(per_behavior_prob.keys())
            if not competition_labels:
                return None

            prob_matrix = np.vstack([per_behavior_prob[bid] for bid in competition_labels]).T
            winner_idx = np.argmax(prob_matrix, axis=1)
            winner_labels = np.array([competition_labels[i] for i in winner_idx], dtype=object)

            # Target behavior probability
            non_no = [bid for bid in competition_labels if not self._is_no_behavior_label(bid)]
            if concept_id in competition_labels:
                target_prob = per_behavior_prob[concept_id]
            elif non_no:
                target_prob = np.max(np.vstack([per_behavior_prob[bid] for bid in non_no]), axis=0)
            else:
                target_prob = prob_matrix[:, 0]

            # Write chunk trace
            chunk_df = pd.DataFrame({
                "chunk_index": np.arange(n_chunks, dtype=int),
                "chunk_start_frame": np.arange(n_chunks, dtype=int) * step_f,
                "probability": target_prob.astype(float),
                "predicted_behavior": winner_labels,
            })
            for i, label in enumerate(competition_labels):
                chunk_df[f"prob_{self._safe_name(label)}"] = prob_matrix[:, i].astype(float)
            chunk_out = chunk_traces_dir / f"{self._safe_name(sid)}_chunk_trace.parquet"
            chunk_df.to_parquet(chunk_out, index=False)

            # Expand to frame-level trace
            frame_to_chunk = np.minimum(np.arange(n_frames, dtype=int) // step_f, max(0, n_chunks - 1))
            frame_prob = target_prob[frame_to_chunk] if len(target_prob) > 0 else np.zeros(n_frames, dtype=float)
            frame_winner = winner_labels[frame_to_chunk] if len(winner_labels) > 0 else np.full(n_frames, competition_labels[0], dtype=object)

            trace_df = pd.DataFrame({
                "frame": np.arange(n_frames, dtype=int),
                "probability": frame_prob.astype(float),
                "predicted_behavior": frame_winner,
                "chunk_index": frame_to_chunk.astype(int),
                "chunk_step_frames": step_f,
            })
            for i, label in enumerate(competition_labels):
                trace_df[f"prob_{self._safe_name(label)}"] = prob_matrix[frame_to_chunk, i].astype(float)
            trace_out = traces_dir / f"{self._safe_name(sid)}_trace.parquet"
            trace_df.to_parquet(trace_out, index=False)

            return {
                "sid": sid, "step_frames": step_f, "n_frames": n_frames,
                "n_windows": n_windows, "elapsed_sec": time.perf_counter() - ts,
                "trace_path": str(trace_out), "chunk_trace_path": str(chunk_out),
            }

        # --- Execute ---
        if workers <= 1:
            for idx, sid in enumerate(target_sessions, 1):
                self._emit(progress_cb, f"Inference session {idx}/{len(target_sessions)}: {sid}")
                result = _run_session(sid)
                if not result:
                    continue
                s_out = str(result["sid"])
                trace_paths[s_out] = result["trace_path"]
                chunk_trace_paths[s_out] = result["chunk_trace_path"]
                step_frames_by_session[s_out] = result["step_frames"]
                total_windows += result["n_windows"]
                total_frames += result["n_frames"]
                self._emit(progress_cb, f"  -> frames={result['n_frames']}, windows={result['n_windows']}, elapsed={result['elapsed_sec']:.1f}s")
        else:
            done = 0
            with cf.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="temporal-infer") as executor:
                futures = {executor.submit(_run_session, sid): sid for sid in target_sessions}
                for future in cf.as_completed(futures):
                    done += 1
                    sid = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        raise RuntimeError(f"Inference failed for session {sid}: {exc}") from exc
                    if not result:
                        continue
                    s_out = str(result["sid"])
                    trace_paths[s_out] = result["trace_path"]
                    chunk_trace_paths[s_out] = result["chunk_trace_path"]
                    step_frames_by_session[s_out] = result["step_frames"]
                    total_windows += result["n_windows"]
                    total_frames += result["n_frames"]
                    self._emit(progress_cb, f"Session {done}/{len(target_sessions)}: {s_out} ({result['n_frames']} frames, {result['elapsed_sec']:.1f}s)")

        elapsed_total = max(1e-9, time.perf_counter() - t0)
        self._emit(progress_cb, f"Inference done: {len(trace_paths)} sessions, {total_frames} frames, {total_windows} windows, {elapsed_total:.1f}s total.")

        inference_manifest = {
            "concept_id": concept_id,
            "window_frames": window_frames,
            "inference_step_seconds": step_seconds,
            "parameter_hash": parameter_hash,
            "software_version": APP_SCHEMA_VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "dense_inhibition",
            "competition": {
                "behavior_models": model_versions,
                "excluded_behavior_ids": sorted(str(v) for v in (cfg.excluded_behavior_ids or []) if str(v).strip()),
                "inhibition_weight": inhibition_weight,
                "suppression_matrix": suppression_matrix,
                "probability_temperature": prob_temp,
            },
            "step_frames_by_session": step_frames_by_session,
            "throughput": {
                "total_elapsed_seconds": elapsed_total,
                "sessions_processed": len(trace_paths),
                "frames_processed": total_frames,
                "windows_processed": total_windows,
            },
            "trace_paths": trace_paths,
            "chunk_trace_paths": chunk_trace_paths,
        }
        write_json(run_dir / "inference_manifest.json", inference_manifest)

        latest = self._load_latest(concept_id)
        latest.update({"inference_dir": str(run_dir), "inference_parameter_hash": parameter_hash, "concept_id": concept_id})
        self._write_latest(concept_id, latest)
        self._emit(progress_cb, "Dense inference completed successfully.")

        return {"status": "ok", "inference_dir": str(run_dir), "trace_paths": trace_paths, "parameter_hash": parameter_hash}

    # ==================================================================
    # POSTPROCESS — bouts from probability traces
    # ==================================================================

    def run_temporal_refinement_postprocess(
        self,
        concept_id: str,
        sessions: list[str] | None = None,
        config: TemporalRefinementConfig | None = None,
        force: bool = False,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        cfg = config or TemporalRefinementConfig()
        self._emit(progress_cb, f"Bout postprocess started for behavior={concept_id}.")

        latest = self._load_latest(concept_id)
        inference_dir_raw = str(latest.get("inference_dir", "")).strip()
        if not inference_dir_raw:
            raise FileNotFoundError("No inference artifacts found. Run inference first.")

        inference_dir = Path(inference_dir_raw)
        inf_manifest = read_json(inference_dir / "inference_manifest.json", {})
        trace_paths = {str(k): str(v) for k, v in inf_manifest.get("trace_paths", {}).items()}
        if sessions:
            allowed = set(sessions)
            trace_paths = {k: v for k, v in trace_paths.items() if k in allowed}
        self._emit(progress_cb, f"Loaded {len(trace_paths)} probability trace(s) for postprocessing.")

        trace_frames: dict[str, pd.DataFrame] = {}
        for sid, tp in trace_paths.items():
            trace_frames[sid] = pd.read_parquet(tp)

        # Select the probability signal for THIS behavior. When inference was run
        # in competition / Direct-Use mode (concept_id="target_behavior"), the
        # generic "probability" column holds the per-frame MAX across all behaviors
        # (see _run_inference_for_session), so thresholding it would emit a bout
        # wherever ANY behavior is active — inflating bout counts and collapsing
        # latency to near-zero. The per-behavior prob_{behavior} column is the
        # correct signal for a single behavior, matching what the Temporal Review
        # trace plot uses. Fall back to "probability" for legacy single-behavior
        # inference traces that have no per-behavior column.
        prob_col = f"prob_{self._safe_name(concept_id)}"
        smooth_by_session: dict[str, np.ndarray] = {}
        for sid, tdf in trace_frames.items():
            col = prob_col if prob_col in tdf.columns else "probability"
            probs = np.nan_to_num(tdf[col].to_numpy(dtype=np.float32), nan=0.0)
            smooth_by_session[sid] = smooth_probabilities(probs, cfg.smoothing_method, cfg.smoothing_window)

        cfg_payload = asdict(cfg)
        cfg_payload.update({"concept_id": concept_id, "source_inference": str(inference_dir)})
        parameter_hash = self._provenance.config_hash(cfg_payload)
        artifact_root = self._artifact_root(concept_id)
        run_dir = artifact_root / f"postprocess_{parameter_hash}"

        if run_dir.exists() and not force:
            if str(self._load_latest(concept_id).get("postprocess_dir", "")) == str(run_dir):
                self._emit(progress_cb, f"Skipped postprocess: artifacts exist at {run_dir}.")
                return {"status": "skipped", "postprocess_dir": str(run_dir)}
        run_dir.mkdir(parents=True, exist_ok=True)
        bouts_dir = run_dir / "bout_outputs"
        bouts_dir.mkdir(parents=True, exist_ok=True)

        manifest = self._imports.load_manifest(self._require_project_root())
        fps_by_session = self._fps_by_session(manifest) if manifest else {}

        metrics_by_session: dict[str, dict[str, Any]] = {}
        bout_paths: dict[str, str] = {}

        for idx, (sid, tp) in enumerate(trace_paths.items(), 1):
            self._emit(progress_cb, f"Postprocess session {idx}/{len(trace_paths)}: {sid}")
            smooth = smooth_by_session.get(sid, np.array([], dtype=float))
            if smooth.size == 0:
                continue

            binary = threshold_probabilities(smooth, onset_thresh=float(cfg.onset_threshold), offset_thresh=cfg.offset_threshold)
            binary = merge_close_bouts(binary, cfg.merge_gap_frames)
            binary = remove_short_bouts(binary, cfg.min_bout_duration_frames)
            intervals = binary_trace_to_intervals(binary)

            out_df = pd.DataFrame(intervals, columns=["start_frame", "end_frame"])
            out_df["session_id"] = sid
            out_df["behavior_id"] = concept_id
            out_path = bouts_dir / f"{self._safe_name(sid)}_bouts.parquet"
            out_df.to_parquet(out_path, index=False)
            bout_paths[sid] = str(out_path)

            bout_frames = sum(max(0, e - s + 1) for s, e in intervals)
            fps = max(1.0, float(fps_by_session.get(sid, 30.0) or 30.0))
            latency_s = float(intervals[0][0]) / fps if intervals else float("nan")
            metrics_by_session[sid] = {
                "frame_metrics": {},
                "bout_metrics": {},
                "boundary_metrics": {},
                "probability_histogram": probability_histogram(smooth),
                "n_bouts": len(intervals),
                "n_bout_frames": bout_frames,
                "fps": fps,
                "time_spent_seconds": bout_frames / fps,
                "latency_to_first_behavior_s": latency_s,
            }

        write_json(run_dir / "session_metrics.json", metrics_by_session)

        post_manifest = {
            "concept_id": concept_id,
            "window_frames": int(inf_manifest.get("window_frames", 30)),
            "parameter_hash": parameter_hash,
            "software_version": APP_SCHEMA_VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "postprocess": {
                "smoothing_method": cfg.smoothing_method,
                "smoothing_window": cfg.smoothing_window,
                "onset_threshold": float(cfg.onset_threshold),
                "offset_threshold": float(cfg.offset_threshold if cfg.offset_threshold is not None else cfg.onset_threshold),
                "min_bout_duration_frames": cfg.min_bout_duration_frames,
                "merge_gap_frames": cfg.merge_gap_frames,
            },
            "bout_paths": bout_paths,
        }
        write_json(run_dir / "postprocess_manifest.json", post_manifest)

        latest = self._load_latest(concept_id)
        latest.update({"postprocess_dir": str(run_dir), "postprocess_parameter_hash": parameter_hash})
        self._write_latest(concept_id, latest)
        self._emit(progress_cb, "Bout postprocessing completed successfully.")

        return {"status": "ok", "postprocess_dir": str(run_dir), "bout_paths": bout_paths, "parameter_hash": parameter_hash}

    # ------------------------------------------------------------------
    # Legacy stubs for backward compatibility
    # ------------------------------------------------------------------

    def run_temporal_refinement_training(
        self, concept_id: str, config: TemporalRefinementConfig | None = None,
        model_name: str | None = None, force: bool = False,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        self._emit(progress_cb, "Training is not needed: temporal refinement uses existing active-learning models directly. Run inference instead.")
        return {"status": "skipped", "reason": "training_removed"}

    def list_temporal_training_runs(self, concept_id: str) -> list[dict[str, Any]]:
        return []

    def set_active_temporal_training_run(self, concept_id: str, training_dir: str) -> dict[str, Any]:
        return {"status": "no_op", "reason": "training_removed"}
