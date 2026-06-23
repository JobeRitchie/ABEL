"""Ablation benchmark runner — trains models under different feature configs.

Supports repeated cross-validation (mean ± SEM) and per-behavior evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from abel.benchmark.configs import AblationSuite
from abel.temporal_refinement.bout_postprocess import (
    merge_close_bouts,
    remove_short_bouts,
    smooth_probabilities,
    threshold_probabilities,
)

logger = logging.getLogger("abel.benchmark")


@dataclass
class FoldMetrics:
    """Metrics from a single CV fold."""

    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    pr_auc: float = float("nan")
    train_rows: int = 0
    val_rows: int = 0

    # Per-fold arrays for aggregate PR curves / confusion matrices
    y_true: np.ndarray | None = None
    y_score: np.ndarray | None = None
    y_pred: np.ndarray | None = None


@dataclass
class RunResult:
    """Aggregated metrics from all CV folds for a single ablation config + behavior."""

    run_name: str
    behavior: str
    overrides: dict[str, Any]

    # Aggregated mean ± SEM
    precision_mean: float = float("nan")
    precision_sem: float = float("nan")
    recall_mean: float = float("nan")
    recall_sem: float = float("nan")
    f1_mean: float = float("nan")
    f1_sem: float = float("nan")
    pr_auc_mean: float = float("nan")
    pr_auc_sem: float = float("nan")

    n_folds: int = 0
    n_features: int = 0
    elapsed_sec: float = 0.0
    error: str = ""

    # Per-fold detail
    fold_metrics: list[FoldMetrics] = field(default_factory=list)

    # Aggregated across folds (concatenated y arrays for PR curve / CM)
    y_true: np.ndarray | None = None
    y_score: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    confusion_matrix: list[list[int]] | None = None
    label_map: dict[int, str] | None = None


@dataclass
class _FoldData:
    """Pre-computed per-fold arrays shared across configs with identical preprocessing."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    sample_weights: np.ndarray
    to_idx: dict[str, int]
    inv: dict[int, str]
    train_label_strs: list[str]   # for adaptive-complexity positive-count check
    val_session_ids: np.ndarray | None = None   # for temporal refinement ordering
    val_start_frames: np.ndarray | None = None  # for temporal refinement ordering
    val_end_frames: np.ndarray | None = None    # for temporal refinement frame traces


@dataclass
class _PrepCache:
    """Cached preprocessing result keyed by (behavior, co_occurring, use_video, video_only)."""

    feature_cols: list[str]
    folds: list[_FoldData]


class AblationRunner:
    """Execute ablation training runs and collect metrics."""

    def __init__(
        self,
        suite: AblationSuite,
        progress_cb: Callable[[str, float], None] | None = None,
    ) -> None:
        self.suite = suite
        self._progress_cb = progress_cb

    def _emit(self, msg: str, pct: float) -> None:
        if self._progress_cb:
            self._progress_cb(msg, pct)

    # ── Data loading ──────────────────────────────────────────────────

    @staticmethod
    def _load_training_data(project_root: Path) -> pd.DataFrame:
        """Load the project's training set parquet."""
        train_path = project_root / "derived" / "training_sets" / "training_set.parquet"
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training set not found at {train_path}. "
                "Run at least one active-learning cycle first."
            )
        return pd.read_parquet(train_path)

    @staticmethod
    def _expand_co_occurring(df: pd.DataFrame) -> pd.DataFrame:
        """Expand pipe-separated labels into separate rows.

        Rows produced by expansion are marked with _co_occurring_expanded=True so
        that _collapse_alternate_labels can drop sibling rows instead of remapping
        them to no_behavior (which would make the same clip both a positive and a
        negative for a given behavior model).
        """
        df = df.copy()
        df["_co_occurring_expanded"] = False
        multi_mask = df["label"].astype(str).str.contains(r"\|", na=False)
        if not multi_mask.any():
            return df
        single_rows = df.loc[~multi_mask].copy()
        single_rows["_co_occurring_expanded"] = False
        parts: list[pd.DataFrame] = [single_rows]
        for _, row in df.loc[multi_mask].iterrows():
            for sub in str(row["label"]).split("|"):
                sub = sub.strip()
                if sub:
                    new_row = row.copy()
                    new_row["label"] = sub
                    new_row["_co_occurring_expanded"] = True
                    parts.append(pd.DataFrame([new_row]))
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _collapse_alternate_labels(df: pd.DataFrame, target_label: str) -> pd.DataFrame:
        """Remap non-target behaviour labels to no_behavior.

        Rows that were produced by expanding a co-occurring (pipe-separated) label
        and do NOT match target_label are dropped rather than remapped — remapping
        them would make the same clip simultaneously a positive and a negative.
        """
        nb_tokens = {"no_behavior", "no_behaviour", "nobehavior", "nobehaviour"}
        _co_exp_col = df.columns.get_loc("_co_occurring_expanded") if "_co_occurring_expanded" in df.columns else None
        drop_indices: list[int] = []
        for i, lbl in enumerate(df["label"].astype(str)):
            lbl_clean = lbl.strip()
            lbl_lower = lbl_clean.lower().replace("_", "").replace(" ", "")
            if lbl_clean == target_label:
                continue
            if lbl_lower in nb_tokens:
                continue
            if _co_exp_col is not None and df.iat[i, _co_exp_col]:
                drop_indices.append(i)
                continue
            df.iat[i, df.columns.get_loc("label")] = "no_behavior"
        if drop_indices:
            df = df.drop(index=df.index[drop_indices]).reset_index(drop=True)
        if "_co_occurring_expanded" in df.columns:
            df = df.drop(columns=["_co_occurring_expanded"])
        return df

    @staticmethod
    def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
        ignore = {
            "segment_id", "label", "label_source", "reviewer_confidence",
            "animal_id", "session_id", "start_frame", "end_frame",
            "prediction_prob", "prediction_prob_fused", "prediction_variance",
            "density_outlier_score", "uncertainty_score", "uncertainty_entropy",
            "uncertainty_margin", "overlap_allowed", "overlap_allowed_x",
            "overlap_allowed_y", "label_true", "label_pred",
        }
        return [
            c for c in df.columns
            if c not in ignore
            and not str(c).startswith("uncertainty_")
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    @staticmethod
    def _drop_dead_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
        std = df[cols].std(axis=0)
        nan_frac = df[cols].isna().mean(axis=0)
        dead = (std.abs() < 1e-12) | std.isna() | (nan_frac > 0.99)
        return [c for c, is_dead in zip(cols, dead) if not is_dead]

    @staticmethod
    def _feature_families(feature_cols: list[str]) -> dict[str, list[str]]:
        """Categorise features into families for per-modality expert training."""
        pose_keys = (
            "nose", "paw", "forepaw", "body_orientation", "head_pitch",
            "centroid", "autocorr", "periodicity", "oscillation",
            "scrape", "angle",
        )
        context_keys = (
            "target", "tmt", "dist", "zone", "roi", "occup",
            "bedding", "substrate", "context",
        )
        motion_keys = ("velocity", "acceleration", "jerk", "speed", "flow", "motion")
        visual_keys = (
            "visual", "image", "clip", "embed", "r3d18", "video_", "cnn",
            "flow_mag", "flow_dir", "flow_entropy", "local_surface",
        )

        def _pick(keys: tuple[str, ...]) -> list[str]:
            return sorted({c for c in feature_cols if any(k in c.lower() for k in keys)})

        pose = _pick(pose_keys)
        context = _pick(context_keys)
        motion = _pick(motion_keys)
        visual = _pick(visual_keys)

        # Keep families disjoint — visual takes priority
        visual_set = set(visual)
        context = [c for c in context if c not in visual_set]
        motion = [c for c in motion if c not in visual_set]
        pose = [c for c in pose if c not in visual_set]

        families: dict[str, list[str]] = {}
        if pose:
            families["pose"] = pose
        if context:
            families["context"] = context
        if motion:
            families["motion"] = motion
        if visual:
            families["visual"] = visual
        return families

    @staticmethod
    def _load_temporal_settings(
        project_root: Path, target_behavior: str,
    ) -> dict[str, Any]:
        """Load per-behavior temporal refinement settings from the project.

        Returns a dict with onset_threshold, min_bout_duration_frames,
        merge_gap_frames.  Falls back to __all__ defaults if the behavior
        is not found.
        """
        defaults = {
            "onset_threshold": 0.65,
            "min_bout_duration_frames": 8,
            "merge_gap_frames": 4,
        }

        review_path = project_root / "config" / "temporal_review_settings.json"
        if not review_path.exists():
            return defaults

        try:
            with open(review_path, "r") as f:
                raw = json.load(f)
        except Exception:
            return defaults

        # Start with the __all__ baseline
        all_cfg = raw.get("__all__", {})
        settings = {**defaults, **all_cfg}

        # Try to find a per-behavior override.  Keys in by_behavior may be
        # UUIDs, so we need the behavior_definitions.yaml to resolve names.
        by_behavior = raw.get("by_behavior", {})
        uuid_to_name: dict[str, str] = {}
        defs_path = project_root / "config" / "behavior_definitions.yaml"
        if defs_path.exists():
            try:
                import yaml

                with open(defs_path, "r") as f:
                    defs = yaml.safe_load(f) or {}
                for b in defs.get("behaviors", []):
                    uuid_to_name[b["behavior_id"]] = b.get("name", "")
            except Exception:
                pass

        for uid, cfg in by_behavior.items():
            name = uuid_to_name.get(uid, uid)
            if name == target_behavior or uid == target_behavior:
                settings.update(cfg)
                break

        return settings

    @staticmethod
    def _apply_temporal_refinement(
        probs: np.ndarray,
        target_col: int,
        session_ids: np.ndarray,
        start_frames: np.ndarray,
        end_frames: np.ndarray,
        onset_threshold: float,
        min_bout_duration_frames: int,
        merge_gap_frames: int,
        smooth_window: int = 5,
    ) -> np.ndarray:
        """Apply the real bout-extraction pipeline and map results back to clips.

        For each session:
        1. Build a frame-level probability trace by assigning each clip's
           target-class probability to its [start_frame, end_frame] range,
           then linearly interpolating gaps between clips.
        2. Apply smoothing → hysteresis thresholding → merge close bouts →
           remove short bouts.
        3. Map the resulting binary bout trace back to clip predictions:
           a clip is predicted positive if the majority of its frames
           fall inside a predicted bout.

        Returns a new copy of *preds* (argmax indices) with temporal
        refinement applied.
        """
        preds = np.argmax(probs, axis=1).copy()

        for sid in np.unique(session_ids):
            mask = session_ids == sid
            idxs = np.where(mask)[0]
            if len(idxs) < 2:
                continue

            sf = start_frames[idxs].astype(int)
            ef = end_frames[idxs].astype(int)
            order = np.argsort(sf)
            idxs = idxs[order]
            sf = sf[order]
            ef = ef[order]

            # Target-class probabilities for these clips
            clip_probs = probs[idxs, target_col].astype(float)

            # Build frame-level trace
            trace_start = int(sf[0])
            trace_end = int(ef[-1])
            n_frames = trace_end - trace_start + 1
            if n_frames <= 0:
                continue

            frame_trace = np.full(n_frames, np.nan, dtype=np.float32)

            # Assign clip probabilities to their frame ranges
            for i in range(len(idxs)):
                local_s = int(sf[i]) - trace_start
                local_e = int(ef[i]) - trace_start
                local_e = min(local_e, n_frames - 1)
                frame_trace[local_s : local_e + 1] = clip_probs[i]

            # Interpolate NaN gaps (linear between last known and next known)
            nans = np.isnan(frame_trace)
            if nans.any() and not nans.all():
                known = np.where(~nans)[0]
                frame_trace = np.interp(
                    np.arange(n_frames), known, frame_trace[known],
                ).astype(np.float32)

            # ── Real bout postprocess pipeline ──
            # 1. Smooth
            frame_trace = smooth_probabilities(
                frame_trace, method="moving_average", window=smooth_window,
            )

            # 2. Hysteresis threshold (offset = onset * 0.7 — standard ratio)
            offset_thresh = onset_threshold * 0.7
            binary = threshold_probabilities(
                frame_trace, onset_threshold, offset_thresh,
            )

            # 3. Merge close bouts
            binary = merge_close_bouts(binary, merge_gap_frames)

            # 4. Remove short bouts
            binary = remove_short_bouts(binary, min_bout_duration_frames)

            # ── Map binary bout trace back to clip predictions ──
            for i in range(len(idxs)):
                local_s = int(sf[i]) - trace_start
                local_e = int(ef[i]) - trace_start
                local_e = min(local_e, n_frames - 1)
                clip_span = binary[local_s : local_e + 1]
                if len(clip_span) > 0 and clip_span.mean() >= 0.5:
                    preds[idxs[i]] = target_col
                else:
                    # Find the most likely non-target class
                    p = probs[idxs[i]].copy()
                    p[target_col] = -1.0
                    preds[idxs[i]] = int(np.argmax(p))

        return preds

    @staticmethod
    def _generate_cv_splits(
        df: pd.DataFrame,
        n_splits: int,
        test_size: float,
        random_state: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate n repeated group-shuffle splits by session."""
        from sklearn.model_selection import GroupShuffleSplit

        groups = df["session_id"].to_numpy() if "session_id" in df.columns else np.arange(len(df))
        n_unique = int(pd.Series(groups).nunique())
        n_rows = len(df)

        if n_rows <= 1:
            idx = np.arange(n_rows, dtype=int)
            return [(idx, idx)]

        if n_unique < 2:
            # Fallback: row-wise random splits with different seeds
            splits: list[tuple[np.ndarray, np.ndarray]] = []
            for k in range(n_splits):
                rng = np.random.RandomState(int(random_state) + k)
                perm = rng.permutation(n_rows)
                n_test = max(1, min(n_rows - 1, int(round(test_size * n_rows))))
                splits.append((perm[n_test:].astype(int), perm[:n_test].astype(int)))
            return splits

        splitter = GroupShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state,
        )
        return list(splitter.split(df, groups=groups))

    @staticmethod
    def _make_estimator(
        family: str,
        params: dict[str, Any],
        random_state: int,
        n_jobs: int = -1,
    ):
        """Create a classifier estimator matching the trainer service logic.

        Parameters
        ----------
        n_jobs : int
            Thread/process parallelism *inside* the estimator.  Use ``1`` when
            the caller already parallelises across runs (avoids deadlocks when
            multiple XGBoost / LightGBM instances each try to use all cores).
        """
        from sklearn.ensemble import HistGradientBoostingClassifier

        family = family.lower()
        if family == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                p = dict(params)
                p.setdefault("max_depth", 6)
                p.setdefault("learning_rate", 0.1)
                p.setdefault("n_estimators", 300)
                p.setdefault("subsample", 0.8)
                p.setdefault("colsample_bytree", 0.8)
                p.setdefault("reg_alpha", 0.1)
                p.setdefault("reg_lambda", 1.0)
                p.setdefault("min_child_samples", 10)
                return LGBMClassifier(
                    random_state=random_state, verbose=-1, n_jobs=n_jobs, **p,
                )
            except ImportError:
                pass
        if family == "xgboost":
            try:
                from xgboost import XGBClassifier
                p = dict(params)
                p.setdefault("tree_method", "hist")
                p.setdefault("max_depth", 6)
                p.setdefault("learning_rate", 0.1)
                p.setdefault("n_estimators", 300)
                p.setdefault("subsample", 0.8)
                p.setdefault("colsample_bytree", 0.8)
                p.setdefault("reg_alpha", 0.1)
                p.setdefault("reg_lambda", 1.0)
                p.setdefault("min_child_weight", 5)
                return XGBClassifier(
                    random_state=random_state,
                    eval_metric="mlogloss",
                    verbosity=0,
                    n_jobs=n_jobs,
                    **p,
                )
            except ImportError:
                pass
        if family in {"random_forest", "rf"}:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                random_state=random_state, n_estimators=300, n_jobs=n_jobs,
            )

        # Fallback: HistGradientBoosting
        return HistGradientBoostingClassifier(random_state=random_state, max_iter=300)

    @staticmethod
    def _detect_classifier(project_root: Path) -> str:
        """Read classifier_type from the project's experiment.yaml.

        Falls back to ``'xgboost'`` (the ABEL default) if the file
        is missing or the key is absent.
        """
        for cfg_name in ("experiment.yaml", "project.yaml"):
            cfg_path = project_root / "config" / cfg_name
            if not cfg_path.exists():
                continue
            try:
                import yaml
                with open(cfg_path, "r") as f:
                    raw = yaml.safe_load(f) or {}
                # experiment.yaml stores it flat; project.yaml nests under behavior_model
                ct = raw.get("classifier_type") or ""
                if not ct:
                    bm = raw.get("behavior_model") or {}
                    ct = bm.get("classifier_type") or ""
                if ct:
                    return str(ct).strip().lower()
            except Exception:
                continue
        return "xgboost"

    @staticmethod
    def _build_behavior_maps(project_root: Path) -> tuple[dict[str, str], dict[str, str]]:
        """Build UUID↔name lookup tables from behavior_definitions.yaml.

        Returns (name_to_uuid, uuid_to_name).
        """
        name_to_uuid: dict[str, str] = {}
        uuid_to_name: dict[str, str] = {}
        defs_path = project_root / "config" / "behavior_definitions.yaml"
        if defs_path.exists():
            try:
                import yaml
                with open(defs_path, "r") as f:
                    defs = yaml.safe_load(f) or {}
                for b in defs.get("behaviors", []):
                    uid = b.get("behavior_id", "")
                    name = b.get("name", "")
                    if uid and name:
                        name_to_uuid[name] = uid
                        uuid_to_name[uid] = name
            except Exception:
                pass
        return name_to_uuid, uuid_to_name

    @staticmethod
    def _resolve_behavior(
        target: str,
        labels: set[str],
        name_to_uuid: dict[str, str],
        uuid_to_name: dict[str, str],
    ) -> str:
        """Resolve a behavior target to the label format used in the data.

        If *target* is already present in *labels*, return it as-is.
        Otherwise try name→UUID and UUID→name lookups.
        """
        if target in labels:
            return target
        # Name → UUID
        uid = name_to_uuid.get(target, "")
        if uid and uid in labels:
            return uid
        # UUID → Name
        name = uuid_to_name.get(target, "")
        if name and name in labels:
            return name
        return target

    # ── Preprocessing cache builder ───────────────────────────────────

    def _build_prep_cache(
        self,
        raw_df: pd.DataFrame,
        target_behavior: str,
        allow_co_occurring: bool,
        use_video_features: bool = True,
        video_features_only: bool = False,
    ) -> _PrepCache:
        """Pre-compute the preprocessed DF, feature cols, CV splits, and
        per-fold numpy arrays so they can be reused across configs that
        share the same preprocessing flags.
        """
        df = raw_df.copy()

        if allow_co_occurring:
            df = self._expand_co_occurring(df)

        # Always collapse to binary (target vs no_behavior), matching the
        # main pipeline which never trains multi-class.
        if target_behavior:
            df = self._collapse_alternate_labels(df, target_behavior)

        df = df[~df["label"].astype(str).isin({"ambiguous", "boundary_error"})].copy()

        feature_cols = self._numeric_feature_cols(df)

        # ── Stage 1: project-level exclusions (feature_exclusions.json) ──
        excl_path = Path(self.suite.project_root) / "config" / "feature_exclusions.json"
        if excl_path.exists():
            try:
                with open(excl_path, "r") as f:
                    excl_data = json.load(f)
                excl_set = set(excl_data.get("excluded_feature_cols", []))
                if excl_set:
                    feature_cols = [c for c in feature_cols if c not in excl_set]
            except Exception:
                pass

        # ── Stage 2: drop dead features (zero-variance / all-NaN) ──
        feature_cols = self._drop_dead_features(df, feature_cols)

        # Video-derived feature keywords (optical flow, surface motion,
        # and CNN/embedding columns when present).
        video_kw = {
            "r3d18", "video_", "embedding", "cnn", "clip_emb", "visual_",
            "flow_mag", "flow_dir", "flow_entropy",
            "local_surface",
        }

        if video_features_only and feature_cols:
            # Keep ONLY video-derived features (standalone evaluation).
            feature_cols = [
                c for c in feature_cols
                if any(k in c.lower() for k in video_kw)
            ]
        elif not use_video_features and feature_cols:
            # Remove video-derived features.
            feature_cols = [
                c for c in feature_cols
                if not any(k in c.lower() for k in video_kw)
            ]

        if df.empty or not feature_cols:
            return _PrepCache(feature_cols=[], folds=[])

        splits = self._generate_cv_splits(
            df, self.suite.n_cv_folds, self.suite.test_size, self.suite.random_state,
        )

        folds: list[_FoldData] = []
        for train_idx, val_idx in splits:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            # Label encoding
            uniq = sorted(str(v) for v in train_df["label"].unique())
            to_idx = {v: i for i, v in enumerate(uniq)}
            y_train = np.array([to_idx[str(v)] for v in train_df["label"]], dtype=int)

            keep_val = val_df["label"].astype(str).isin(to_idx)
            val_df = val_df.loc[keep_val].copy()
            if val_df.empty:
                val_df = train_df.copy()
            y_val = np.array([to_idx[str(v)] for v in val_df["label"]], dtype=int)
            inv = {v: k for k, v in to_idx.items()}

            # Pre-convert to numpy
            x_train = train_df[feature_cols].to_numpy(dtype=float)
            x_val = val_df[feature_cols].to_numpy(dtype=float)

            # Sample weights
            nb_tokens = {"no_behavior", "no_behaviour", "nobehavior", "nobehaviour"}
            nb_mask = np.array([
                lbl.strip().lower().replace("_", "").replace(" ", "") in nb_tokens
                for lbl in train_df["label"].astype(str)
            ])
            sample_weights = np.ones(len(train_df), dtype=float)
            n_neg = int(nb_mask.sum())
            n_pos_w = len(train_df) - n_neg
            if n_pos_w > 0 and n_neg > 0:
                auto_weight = float(np.clip(n_pos_w / n_neg, 0.33, 3.0))
                sample_weights[nb_mask] = auto_weight

            folds.append(_FoldData(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                sample_weights=sample_weights,
                to_idx=to_idx,
                inv=inv,
                train_label_strs=[str(v) for v in train_df["label"]],
                val_session_ids=(
                    val_df["session_id"].to_numpy()
                    if "session_id" in val_df.columns else None
                ),
                val_start_frames=(
                    val_df["start_frame"].to_numpy(dtype=float)
                    if "start_frame" in val_df.columns else None
                ),
                val_end_frames=(
                    val_df["end_frame"].to_numpy(dtype=float)
                    if "end_frame" in val_df.columns else None
                ),
            ))

        return _PrepCache(feature_cols=feature_cols, folds=folds)

    # ── Single ablation run (all folds, one behavior) ─────────────────

    def _execute_single_run(
        self,
        run_cfg: dict[str, Any],
        prep: _PrepCache,
        target_behavior: str,
        *,
        n_jobs: int = -1,
    ) -> RunResult:
        """Train + evaluate across CV folds using pre-cached feature data.

        Matches the main pipeline (ActiveLearningTrainerService.train):
        - Always binary classification (target vs no_behavior)
        - Single model on all features (no expert ensemble)
        - Binary metrics evaluated on the target class
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import (
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        run_name = run_cfg.get("_run_name", "unknown")
        overrides = {k: v for k, v in run_cfg.items() if not k.startswith("_")}
        result = RunResult(run_name=run_name, behavior=target_behavior, overrides=overrides)
        result.n_features = len(prep.feature_cols)

        t0 = time.perf_counter()
        try:
            if not prep.feature_cols or not prep.folds:
                result.error = "No informative features after preprocessing"
                return result

            all_y_true: list[np.ndarray] = []
            all_y_score: list[np.ndarray] = []
            all_y_pred: list[np.ndarray] = []
            fold_results: list[FoldMetrics] = []
            last_inv: dict[int, str] = {}

            for fold_i, fd in enumerate(prep.folds):
                inv = fd.inv
                last_inv = inv

                # Resolve the target class index (always binary: 0 vs 1)
                target_idx = next(
                    (idx for idx, name in inv.items() if str(name) == target_behavior),
                    None,
                )
                if target_idx is None and len(inv) == 2:
                    target_idx = 1

                # ── Adaptive complexity ───────────────────────────
                params: dict[str, Any] = {}
                if run_cfg.get("adaptive_complexity", False):
                    nb_tokens = {"no_behavior", "no_behaviour", "nobehavior", "nobehaviour"}
                    n_pos = sum(
                        1 for lbl in fd.train_label_strs
                        if lbl.strip().lower().replace("_", "").replace(" ", "") not in nb_tokens
                    )
                    n_feat = len(prep.feature_cols)
                    ratio = n_pos / max(1, n_feat)
                    if ratio < 1.0:
                        params["max_depth"] = 4
                        params["n_estimators"] = max(50, min(150, int(n_pos * 0.5)))
                    elif ratio < 2.0:
                        params["max_depth"] = 5
                        params["n_estimators"] = max(100, min(200, int(n_pos * 0.6)))
                    else:
                        params["max_depth"] = 6
                        params["n_estimators"] = 300
                    if n_pos < 500:
                        params.setdefault("reg_alpha", 0.3)
                        params.setdefault("reg_lambda", 2.0)
                        params.setdefault("min_child_weight", max(5, int(n_pos * 0.03)))

                # ── Train (single model, all features) ────────────
                fold_seed = self.suite.random_state + fold_i
                est = self._make_estimator(
                    self.suite.classifier_family, params, fold_seed,
                    n_jobs=n_jobs,
                )
                est.fit(fd.x_train, fd.y_train,
                        sample_weight=fd.sample_weights)

                # ── Calibration ───────────────────────────────────
                clf = est
                cal_method = run_cfg.get("calibration_method", "none")
                if cal_method in {"sigmoid", "isotonic"}:
                    calibrated = CalibratedClassifierCV(
                        estimator=est, method=cal_method, cv="prefit",
                    )
                    calibrated.fit(fd.x_val, fd.y_val)
                    clf = calibrated
                probs = clf.predict_proba(fd.x_val)

                # ── Temporal refinement (bout postprocessing) ─────
                temporal_on = bool(
                    run_cfg.get("temporal_refinement_enabled", False)
                )

                if (
                    temporal_on
                    and target_idx is not None
                    and fd.val_session_ids is not None
                    and fd.val_start_frames is not None
                    and fd.val_end_frames is not None
                ):
                    t_settings = self._load_temporal_settings(
                        Path(self.suite.project_root), target_behavior,
                    )
                    preds = self._apply_temporal_refinement(
                        probs,
                        target_col=int(target_idx),
                        session_ids=fd.val_session_ids,
                        start_frames=fd.val_start_frames,
                        end_frames=fd.val_end_frames,
                        onset_threshold=t_settings["onset_threshold"],
                        min_bout_duration_frames=t_settings["min_bout_duration_frames"],
                        merge_gap_frames=t_settings["merge_gap_frames"],
                        smooth_window=int(run_cfg.get("smooth_window", 5)),
                    )
                else:
                    preds = np.argmax(probs, axis=1)

                # ── Evaluate this fold (binary metrics for target) ─
                if target_idx is not None:
                    y_true_bin = (fd.y_val == int(target_idx)).astype(int)
                    y_pred_bin = (preds == int(target_idx)).astype(int)

                    fold_p = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
                    fold_r = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
                    fold_f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
                    fold_score = probs[:, int(target_idx)]
                    if y_true_bin.sum() > 0:
                        fold_ap = float(average_precision_score(y_true_bin, fold_score))
                    else:
                        fold_ap = float("nan")
                else:
                    fold_p = float(precision_score(fd.y_val, preds, average="macro", zero_division=0))
                    fold_r = float(recall_score(fd.y_val, preds, average="macro", zero_division=0))
                    fold_f1 = float(f1_score(fd.y_val, preds, average="macro", zero_division=0))
                    fold_ap = float("nan")
                    fold_score = probs[:, -1]

                fm = FoldMetrics(
                    precision=fold_p, recall=fold_r, f1=fold_f1, pr_auc=fold_ap,
                    train_rows=len(fd.y_train), val_rows=len(fd.y_val),
                    y_true=fd.y_val, y_score=fold_score, y_pred=preds,
                )
                fold_results.append(fm)

                all_y_true.append(fd.y_val)
                all_y_score.append(fold_score)
                all_y_pred.append(preds)

            # ── Aggregate across folds ────────────────────────────
            result.fold_metrics = fold_results
            result.n_folds = len(fold_results)

            precisions = np.array([f.precision for f in fold_results])
            recalls = np.array([f.recall for f in fold_results])
            f1s = np.array([f.f1 for f in fold_results])
            aps = np.array([f.pr_auc for f in fold_results])

            n = len(fold_results)
            result.precision_mean = float(np.nanmean(precisions))
            result.precision_sem = float(np.nanstd(precisions, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            result.recall_mean = float(np.nanmean(recalls))
            result.recall_sem = float(np.nanstd(recalls, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            result.f1_mean = float(np.nanmean(f1s))
            result.f1_sem = float(np.nanstd(f1s, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            result.pr_auc_mean = float(np.nanmean(aps))
            result.pr_auc_sem = float(np.nanstd(aps, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

            # Concatenated arrays for aggregate PR curve / CM
            if all_y_true:
                result.y_true = np.concatenate(all_y_true)
                result.y_score = np.concatenate(all_y_score)
                result.y_pred = np.concatenate(all_y_pred)
                result.confusion_matrix = confusion_matrix(result.y_true, result.y_pred).tolist()
            result.label_map = last_inv

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Ablation run '%s' [%s] failed", run_name, target_behavior)

        result.elapsed_sec = time.perf_counter() - t0
        return result

    # ── Full ablation suite ───────────────────────────────────────────

    @staticmethod
    def _format_eta(elapsed: float, done: int, total: int) -> str:
        """Format an ETA string from completed / total counts."""
        if done < 1:
            return ""
        avg = elapsed / done
        remaining = avg * (total - done)
        if remaining < 60:
            return f"ETA {remaining:.0f}s"
        if remaining < 3600:
            return f"ETA {remaining / 60:.1f}m"
        return f"ETA {remaining / 3600:.1f}h"

    def run_all(self) -> list[RunResult]:
        """Execute all ablation configs × all target behaviors.

        Uses a preprocessing cache keyed by (behavior, co_occurring,
        use_video, video_only) to avoid redundant work, and runs
        independent training tasks concurrently.
        """
        project_root = Path(self.suite.project_root)

        # ── Auto-detect classifier from project config ────────────
        if not self.suite.classifier_family:
            self.suite.classifier_family = self._detect_classifier(project_root)

        self._emit("Loading training data…", 0.0)
        raw_df = self._load_training_data(project_root)

        # ── Resolve behavior names ↔ UUIDs ────────────────────────
        name_to_uuid, uuid_to_name = self._build_behavior_maps(project_root)
        data_labels = set(str(v) for v in raw_df["label"].unique())

        behaviors = self.suite.target_behaviors
        if not behaviors:
            skip = {"no_behavior", "no_behaviour", "ambiguous", "boundary_error"}
            all_labels = sorted(
                str(v) for v in raw_df["label"].unique() if str(v).lower() not in skip
            )
            behaviors = all_labels if all_labels else [""]

        # Resolve each behavior to the format actually present in the data,
        # and keep a display-name map for result labels.
        resolved: list[str] = []
        display_names: dict[str, str] = {}
        for beh in behaviors:
            r = self._resolve_behavior(beh, data_labels, name_to_uuid, uuid_to_name)
            resolved.append(r)
            # Human-readable display name
            display_names[r] = uuid_to_name.get(r, r)
        behaviors = resolved

        configs = self.suite.run_configs()

        # ── Identify unique preprocessing keys & build caches ─────
        CacheKey = tuple[str, bool, bool, bool]
        task_list: list[tuple[dict, str, CacheKey]] = []
        unique_keys: dict[CacheKey, None] = {}

        for behavior in behaviors:
            for cfg in configs:
                co_occur = bool(cfg.get("allow_co_occurring_behaviors", False))
                use_video = bool(cfg.get("use_video_features", True))
                vid_only = bool(cfg.get("video_features_only", False))
                key: CacheKey = (behavior, co_occur, use_video, vid_only)
                task_list.append((cfg, behavior, key))
                unique_keys.setdefault(key, None)

        n_caches = len(unique_keys)
        self._emit(
            f"Pre-computing {n_caches} unique feature set(s) "
            f"for {len(task_list)} runs…",
            0.02,
        )
        prep_cache: dict[CacheKey, _PrepCache] = {}
        for i, key in enumerate(unique_keys):
            beh, co_occur, use_video, vid_only = key
            prep_cache[key] = self._build_prep_cache(
                raw_df, beh, co_occur, use_video, vid_only,
            )
            self._emit(
                f"Cached feature set {i + 1}/{n_caches}",
                0.02 + 0.08 * ((i + 1) / n_caches),
            )

        # ── Execute runs (concurrently via ThreadPoolExecutor) ────
        n_total = len(task_list)
        results: list[RunResult] = []
        completed = 0
        t_start = time.perf_counter()
        max_workers = max(1, self.suite.max_workers) if self.suite.parallel else 1

        if max_workers > 1:
            self._emit(
                f"Running {n_total} tasks across {max_workers} threads…", 0.10,
            )
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {}
                for cfg, behavior, key in task_list:
                    prep = prep_cache[key]
                    fut = pool.submit(
                        self._execute_single_run, cfg, prep, behavior,
                        n_jobs=1,
                    )
                    futures[fut] = (cfg.get("_run_name", "?"), behavior)

                for fut in as_completed(futures):
                    run_name, behavior = futures[fut]
                    res = fut.result()
                    results.append(res)
                    completed += 1
                    elapsed = time.perf_counter() - t_start
                    pct = 0.10 + 0.85 * (completed / n_total)
                    beh_short = display_names.get(behavior, behavior)[:12]
                    f1_str = (
                        f"{res.f1_mean:.3f}±{res.f1_sem:.3f}"
                        if not res.error else "ERR"
                    )
                    eta = self._format_eta(elapsed, completed, n_total)
                    self._emit(
                        f"[{beh_short}] {run_name} F1={f1_str} "
                        f"({completed}/{n_total}) {eta}",
                        pct,
                    )
        else:
            for cfg, behavior, key in task_list:
                prep = prep_cache[key]
                run_name = cfg.get("_run_name", "unknown")
                completed += 1
                elapsed = time.perf_counter() - t_start
                eta = self._format_eta(elapsed, completed, n_total)
                pct = 0.10 + 0.85 * (completed / n_total)
                beh_short = display_names.get(behavior, behavior)[:12]
                self._emit(
                    f"[{beh_short}] Running: {run_name} "
                    f"({completed}/{n_total}) {eta}",
                    pct,
                )
                res = self._execute_single_run(cfg, prep, behavior)
                results.append(res)
                f1_str = (
                    f"{res.f1_mean:.3f}±{res.f1_sem:.3f}"
                    if not res.error else "ERR"
                )
                elapsed = time.perf_counter() - t_start
                eta = self._format_eta(elapsed, completed, n_total)
                self._emit(
                    f"[{beh_short}] {run_name} F1={f1_str} "
                    f"({completed}/{n_total}) {eta}",
                    pct,
                )

        # ── Sort & finish ─────────────────────────────────────────
        # Replace UUID behavior IDs with human-readable display names
        for r in results:
            r.behavior = display_names.get(r.behavior, r.behavior)

        def _sort_key(r: RunResult) -> tuple[str, int, str]:
            if r.run_name == "baseline_all_on":
                return (r.behavior, 0, "")
            if r.run_name == "baseline_all_off":
                return (r.behavior, 2, "")
            return (r.behavior, 1, r.run_name)

        results.sort(key=_sort_key)
        total_sec = time.perf_counter() - t_start
        self._emit(f"Ablation suite complete — {total_sec:.1f}s total.", 1.0)
        return results
