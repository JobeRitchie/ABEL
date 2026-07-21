"""Training and retraining service for closed-loop active learning."""

from __future__ import annotations

import logging
import platform
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abel.models.schemas import ModelCard
from abel.services.provenance_service import ProvenanceService
from abel.storage.file_store import write_json, write_yaml
from abel.utils import xgb_predict

logger = logging.getLogger("abel")


@dataclass
class TrainingConfig:
    classifier_family: str = "lightgbm"
    classifier_params: dict[str, Any] | None = None
    calibration_method: str = "sigmoid"
    split_strategy: str = "group_shuffle_session"
    test_size: float = 0.25
    random_state: int = 42
    target_label: str = ""
    model_version: str = "behavior_model_v1"
    feature_version: str = "representation_v1"
    include_imported: bool = True  # include cross-project imported examples in training
    require_gpu: bool = False
    max_train_samples_per_class: int = 0  # 0 = unlimited; caps each class for large-dataset speed
    no_behavior_sample_weight: float = 0.0  # 0 = auto (computed from class imbalance); >0 = manual override
    allow_co_occurring_behaviors: bool = False  # expand pipe-separated labels into per-behavior rows
    adaptive_complexity: bool = True  # auto-tune n_estimators/max_depth from data
    drop_zero_variance_features: bool = True  # remove features with zero variance before training
    # Per-run feature exclusions chosen in the Active Learning tab.  Applied at
    # training time (segment level) — NOT baked into the representation cache —
    # so toggling exclusions never forces a representation rebuild.
    excluded_feature_cols: tuple[str, ...] = ()
    enable_feature_augmentation: bool = True  # augment positive training examples with jitter + dropout
    augmentation_jitter_sigma: float = 0.05   # noise level as fraction of per-feature std
    augmentation_dropout_prob: float = 0.10   # fraction of features randomly zeroed per copy
    augmentation_copies: int = 3              # synthetic copies per positive example
    # After the held-out evaluation, refit the model that actually ships on ALL
    # labeled rows (train + held-out) so per-session corrections (e.g. temporal
    # false-positive feedback on a validation-fold mouse) reach inference.  The
    # reported metrics/split_manifest still come from the honest held-out split.
    # Disabled automatically for the validation/benchmark platform, which needs
    # the held-out model (it supplies a precomputed split).
    deploy_refit_on_all_data: bool = True
    # Very-low-sample guard.  When the number of trainable labeled rows is below
    # this threshold, reserving a hold-out test set wastes scarce examples and
    # yields meaningless metrics — so no rows are held out and the model trains
    # on ALL examples (reported metrics become in-sample, flagged in the log).
    # Never applied when the caller supplies a precomputed split.
    min_holdout_samples: int = 10


@dataclass
class TrainEvalResult:
    """In-memory result of a single train+evaluate run (no disk side effects).

    Returned by ``ActiveLearningTrainerService.train_and_evaluate`` and consumed
    by ``_write_model_artifacts`` and the validation/benchmark platform.
    """

    fitted_estimator: Any
    calibrated_model: Any
    feature_cols: list[str]
    label_map: dict[int, str]
    target_idx: int | None
    y_val: np.ndarray
    val_probs: np.ndarray
    val_preds: np.ndarray
    val_meta: pd.DataFrame
    metrics: dict[str, Any]
    feature_importance: dict[str, float]
    split_manifest: dict[str, Any]
    elapsed_sec: float
    degenerate_val: bool


class ActiveLearningTrainerService:
    """Maintains training snapshots, model state, and model cards."""

    def __init__(self) -> None:
        self._provenance = ProvenanceService()

    @staticmethod
    def _label_map(labels: pd.Series) -> tuple[np.ndarray, dict[int, str]]:
        uniq = sorted(str(v) for v in labels.unique())
        to_idx = {v: i for i, v in enumerate(uniq)}
        y = np.asarray([to_idx[str(v)] for v in labels], dtype=int)
        inv = {v: k for k, v in to_idx.items()}
        return y, inv

    @staticmethod
    def _imported_mask(frame: pd.DataFrame) -> pd.Series:
        """Rows that came from a cross-project import (label_source imported:*)."""
        if "label_source" in frame.columns:
            return frame["label_source"].astype(str).str.startswith("imported:")
        return pd.Series(False, index=frame.index)

    @classmethod
    def _scope_training_rows(
        cls,
        df: pd.DataFrame,
        session_ids: "set[str] | None",
        include_imported: bool,
    ) -> pd.DataFrame:
        """Apply session-scope and imported include/exclude to the training rows.

        Imported rows carry namespaced session ids that aren't part of this
        project's sessions, so the session-scope filter must not drop them — they
        are gated solely by ``include_imported``.  Raises ``ValueError`` if no
        rows remain after either step.
        """
        if session_ids and "session_id" in df.columns:
            in_scope = df["session_id"].astype(str).isin(session_ids) | cls._imported_mask(df)
            df = df[in_scope].reset_index(drop=True)
            if df.empty:
                raise ValueError(
                    "No training rows remain after filtering to the selected session scope."
                )
        if not include_imported:
            imported = cls._imported_mask(df)
            if bool(imported.any()):
                df = df[~imported].reset_index(drop=True)
                if df.empty:
                    raise ValueError(
                        "No training rows remain after excluding imported examples."
                    )
        return df

    @staticmethod
    def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
        ignore = {"segment_id", "label", "label_source", "reviewer_confidence", "animal_id", "session_id"}
        forbidden = {
            "start_frame",
            "end_frame",
            "prediction_prob",
            "prediction_prob_fused",
            "prediction_variance",
            "density_outlier_score",
            "uncertainty_score",
            "uncertainty_entropy",
            "uncertainty_margin",
            "overlap_allowed",
            "overlap_allowed_x",
            "overlap_allowed_y",
            "label_true",
            "label_pred",
        }
        cols = [
            c
            for c in df.columns
            if c not in ignore
            and c not in forbidden
            and not str(c).startswith("uncertainty_")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        return cols

    @staticmethod
    def _make_estimator(family: str, params: dict[str, Any], random_state: int):
        from sklearn.ensemble import HistGradientBoostingClassifier

        family = family.lower()
        if family == "lightgbm":
            try:
                from lightgbm import LGBMClassifier

                resolved_params = dict(params)
                # Regularization defaults to reduce overconfidence.
                resolved_params.setdefault("max_depth", 6)
                resolved_params.setdefault("learning_rate", 0.1)
                resolved_params.setdefault("n_estimators", 300)
                resolved_params.setdefault("subsample", 0.8)
                resolved_params.setdefault("colsample_bytree", 0.8)
                resolved_params.setdefault("reg_alpha", 0.1)
                resolved_params.setdefault("reg_lambda", 1.0)
                resolved_params.setdefault("min_child_samples", 10)
                return LGBMClassifier(random_state=random_state, **resolved_params)
            except Exception:
                resolved_params = dict(params)
                resolved_params.setdefault("max_depth", 6)
                resolved_params.setdefault("learning_rate", 0.1)
                # Translate n_estimators (LightGBM/XGBoost) to max_iter (HGB).
                resolved_params.setdefault("max_iter", resolved_params.pop("n_estimators", 300))
                resolved_params.setdefault("min_samples_leaf", 10)
                resolved_params.setdefault("l2_regularization", max(float(resolved_params.pop("reg_lambda", 1.0)), 1.0))
                # Remove keys that HGB doesn't accept.
                for _hgb_skip in ("tree_method", "n_estimators", "subsample", "colsample_bytree",
                                  "reg_alpha", "reg_lambda", "min_child_weight", "device"):
                    resolved_params.pop(_hgb_skip, None)
                return HistGradientBoostingClassifier(random_state=random_state, **resolved_params)
        if family == "xgboost":
            try:
                from xgboost import XGBClassifier

                resolved_params = dict(params)
                resolved_params.setdefault("tree_method", "hist")
                # Regularization defaults to reduce overconfidence.
                resolved_params.setdefault("max_depth", 6)
                resolved_params.setdefault("learning_rate", 0.1)
                resolved_params.setdefault("n_estimators", 300)
                resolved_params.setdefault("subsample", 0.8)
                resolved_params.setdefault("colsample_bytree", 0.8)
                resolved_params.setdefault("reg_alpha", 0.1)
                resolved_params.setdefault("reg_lambda", 1.0)
                resolved_params.setdefault("min_child_weight", 5)
                # On Windows, prefer CUDA when available and let training fallback to CPU.
                if platform.system().lower().startswith("win"):
                    resolved_params.setdefault("device", "cuda")

                return XGBClassifier(
                    random_state=random_state,
                    eval_metric="mlogloss",
                    **resolved_params,
                )
            except Exception:
                resolved_params = dict(params)
                resolved_params.setdefault("max_depth", 6)
                resolved_params.setdefault("learning_rate", 0.1)
                resolved_params.setdefault("max_iter", resolved_params.pop("n_estimators", 300))
                resolved_params.setdefault("min_samples_leaf", 10)
                resolved_params.setdefault("l2_regularization", max(float(resolved_params.pop("reg_lambda", 1.0)), 1.0))
                for _hgb_skip in ("tree_method", "n_estimators", "subsample", "colsample_bytree",
                                  "reg_alpha", "reg_lambda", "min_child_weight", "device"):
                    resolved_params.pop(_hgb_skip, None)
                return HistGradientBoostingClassifier(random_state=random_state, **resolved_params)
        if family in {"hist_gbdt", "histgradientboosting", "hgb"}:
            resolved_params = dict(params)
            resolved_params.setdefault("max_depth", 6)
            resolved_params.setdefault("learning_rate", 0.1)
            resolved_params.setdefault("max_iter", resolved_params.pop("n_estimators", 300))
            resolved_params.setdefault("min_samples_leaf", 10)
            resolved_params.setdefault("l2_regularization", max(float(resolved_params.pop("reg_lambda", 1.0)), 1.0))
            for _hgb_skip in ("tree_method", "n_estimators", "subsample", "colsample_bytree",
                              "reg_alpha", "reg_lambda", "min_child_weight", "device"):
                resolved_params.pop(_hgb_skip, None)
            return HistGradientBoostingClassifier(random_state=random_state, **resolved_params)
        if family in {"random_forest", "rf"}:
            from sklearn.ensemble import RandomForestClassifier

            rf_params = dict(params)
            rf_n = rf_params.pop("n_estimators", 300)
            for _rf_skip in ("tree_method", "subsample", "colsample_bytree",
                             "reg_alpha", "reg_lambda", "min_child_weight", "device"):
                rf_params.pop(_rf_skip, None)
            return RandomForestClassifier(random_state=random_state, n_estimators=rf_n, **rf_params)
        raise ValueError(f"Unsupported classifier_family: {family}")

    @staticmethod
    def _split(df: pd.DataFrame, strategy: str, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.model_selection import GroupShuffleSplit

        if strategy.endswith("subject"):
            groups = df["animal_id"].to_numpy()
        else:
            groups = df["session_id"].to_numpy()

        # Quick-test and sparse-label runs can collapse to a single group.
        # Fallback to a deterministic row-wise split instead of failing.
        n_rows = int(len(df))
        n_unique_groups = int(pd.Series(groups).nunique())
        if n_rows <= 1:
            idx = np.arange(n_rows, dtype=int)
            return idx, idx
        if n_unique_groups < 2:
            rng = np.random.RandomState(int(random_state))
            perm = rng.permutation(n_rows)
            n_test = max(1, int(round(float(test_size) * n_rows)))
            n_test = min(n_rows - 1, n_test)
            val_idx = perm[:n_test]
            train_idx = perm[n_test:]
            return train_idx.astype(int), val_idx.astype(int)

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, val_idx = next(splitter.split(df, groups=groups))
        return train_idx, val_idx

    @staticmethod
    def _augment_training_features(
        x: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        target_label_idx: int,
        jitter_sigma: float = 0.05,
        dropout_prob: float = 0.10,
        n_copies: int = 3,
        rng: "np.random.RandomState | None" = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create augmented copies of positive-class training examples.

        For each positive example (y == target_label_idx), creates *n_copies*
        synthetic variants by:
          1. Adding per-feature Gaussian noise scaled to jitter_sigma × feature_std.
          2. Randomly zeroing *dropout_prob* fraction of features (simulates
             tracking occlusions or absent keypoints).

        Augmented rows inherit the same sample weight as their source positive
        example.  Returns extended (x, y, sample_weights) arrays.
        """
        if rng is None:
            rng = np.random.RandomState(0)
        pos_mask = y == target_label_idx
        if not pos_mask.any() or n_copies < 1:
            return x, y, sample_weights

        x_pos = x[pos_mask]             # shape (n_pos, n_feat)
        w_pos = sample_weights[pos_mask]

        # Per-feature std for noise scaling (computed from all training data)
        feat_std = np.nanstd(x, axis=0)
        feat_std = np.where(feat_std > 1e-9, feat_std, 1e-9)

        aug_x_parts: list[np.ndarray] = []
        aug_y_parts: list[np.ndarray] = []
        aug_w_parts: list[np.ndarray] = []
        for _ in range(n_copies):
            noise = rng.randn(*x_pos.shape) * (jitter_sigma * feat_std)
            dropout_mask = rng.rand(*x_pos.shape) < dropout_prob
            copy_x = x_pos + noise
            copy_x[dropout_mask] = 0.0
            aug_x_parts.append(copy_x)
            aug_y_parts.append(np.full(len(x_pos), target_label_idx, dtype=y.dtype))
            aug_w_parts.append(w_pos.copy())

        x_aug = np.concatenate([x] + aug_x_parts, axis=0)
        y_aug = np.concatenate([y] + aug_y_parts, axis=0)
        w_aug = np.concatenate([sample_weights] + aug_w_parts, axis=0)
        return x_aug, y_aug, w_aug

    def merge_and_snapshot_training_set(
        self,
        project_root: Path,
        labeled_segments: pd.DataFrame,
    ) -> Path:
        out_dir = project_root / "derived" / "training_sets"
        snap_dir = out_dir / "snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_dir.mkdir(parents=True, exist_ok=True)

        current_path = out_dir / "training_set.parquet"
        if current_path.exists():
            current = pd.read_parquet(current_path)
            merged = pd.concat([current, labeled_segments], ignore_index=True)
            merged = merged.drop_duplicates(subset=["segment_id"], keep="last")
        else:
            merged = labeled_segments.drop_duplicates(subset=["segment_id"], keep="last")

        merged.to_parquet(current_path, index=False)
        snap_path = snap_dir / f"training_set_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        merged.to_parquet(snap_path, index=False)
        return snap_path

    def train(
        self,
        project_root: Path,
        config: TrainingConfig | None = None,
        session_ids: set[str] | None = None,
        progress_cb: "Callable[[str], None] | None" = None,
    ) -> dict[str, Any]:
        """Full train→evaluate→persist pipeline (unchanged public behavior).

        Decomposed into three reusable parts: load the training frame from disk,
        run the pure train+evaluate core, then write model artifacts.  The
        validation/benchmark platform reuses ``train_and_evaluate`` directly.
        """
        def _log(msg: str) -> None:
            if progress_cb is not None:
                progress_cb(msg)

        cfg = config or TrainingConfig()
        df = self._load_training_frame(project_root, cfg, session_ids, _log)
        result = self.train_and_evaluate(
            df, cfg, project_root=project_root, progress_cb=progress_cb
        )
        return self._write_model_artifacts(project_root, cfg, result, _log)

    def _load_training_frame(
        self,
        project_root: Path,
        cfg: TrainingConfig,
        session_ids: set[str] | None,
        _log: "Callable[[str], None]",
    ) -> pd.DataFrame:
        """Load + session-scope the training parquet and apply temporal feedback."""
        train_path = project_root / "derived" / "training_sets" / "training_set.parquet"
        if not train_path.exists():
            raise ValueError("Training set not found. Create labeled training data first.")

        _log("Loading training set…")
        df = pd.read_parquet(train_path)
        # Apply session scope and the imported-examples include/exclude toggle.
        n_imported = int(self._imported_mask(df).sum())
        df = self._scope_training_rows(df, session_ids, cfg.include_imported)
        if not cfg.include_imported and n_imported:
            _log(f"Excluded {n_imported} imported example(s) from training.")

        # ── Apply temporal false-positive feedback ────────────────────────
        # If the user flagged frame ranges as false positives in the temporal
        # review:
        # 1. Relabel any existing training segments whose center falls within
        #    those intervals to ``no_behavior``.
        # 2. Inject new hard-negative rows from segment_features.parquet for
        #    any overlapping segments not already in the training set, so the
        #    model sees the flagged regions as negatives even when no labeled
        #    training row existed there previously.
        target_label = str(cfg.target_label or "").strip()
        if target_label and "start_frame" in df.columns and "end_frame" in df.columns:
            try:
                from abel.storage.file_store import read_json
                fp_path = project_root / "derived" / "temporal_refinement" / target_label / "feedback_intervals.json"
                if fp_path.exists():
                    fb_raw = read_json(fp_path, {})
                    fp_map: dict[str, list] = fb_raw.get("false_positive_intervals_by_session", {})
                    fn_map: dict[str, list] = fb_raw.get("false_negative_intervals_by_session", {})
                    if fp_map or fn_map:
                        n_relabeled = 0
                        # The FP label means "the target behavior did NOT
                        # occur here" — NOT "no behavior occurred".  Use
                        # ``not_{target_label}`` so the label-collapse step
                        # treats it as a negative for this behavior without
                        # making claims about other behaviors.
                        fp_neg_label = f"not_{target_label}"

                        # Step 1: relabel existing training rows that fall in FP regions
                        if fp_map:
                            for i, row in df.iterrows():
                                sid = str(row.get("session_id", ""))
                                intervals = fp_map.get(sid, [])
                                if not intervals:
                                    continue
                                center = (int(row["start_frame"]) + int(row["end_frame"])) // 2
                                for interval in intervals:
                                    if len(interval) >= 2 and int(interval[0]) <= center <= int(interval[1]):
                                        if str(row.get("label", "")) == target_label:
                                            df.at[i, "label"] = fp_neg_label
                                            n_relabeled += 1
                                        break
                        if n_relabeled > 0:
                            logger.info(
                                "Relabeled %d existing segment(s) to no_behavior "
                                "based on temporal false-positive feedback for %s.",
                                n_relabeled,
                                target_label,
                            )

                        # Step 2: inject new hard-negative rows from features
                        # for FP intervals with no matching training segment.
                        features_path = project_root / "derived" / "representations" / "segment_features.parquet"
                        if features_path.exists():
                            try:
                                feat_df = pd.read_parquet(features_path)
                                if (
                                    not feat_df.empty
                                    and "session_id" in feat_df.columns
                                    and "start_frame" in feat_df.columns
                                    and "end_frame" in feat_df.columns
                                ):
                                    existing_ids = set(df["segment_id"].astype(str))
                                    injected_rows: list[pd.Series] = []
                                    # Inject hard negatives for FP intervals
                                    for sid, intervals in fp_map.items():
                                        sess_feat = feat_df[feat_df["session_id"].astype(str) == sid]
                                        for interval in intervals:
                                            if len(interval) < 2:
                                                continue
                                            iv_start, iv_end = int(interval[0]), int(interval[1])
                                            for _, frow in sess_feat.iterrows():
                                                seg_center = (int(frow["start_frame"]) + int(frow["end_frame"])) // 2
                                                seg_id = str(frow["segment_id"])
                                                if iv_start <= seg_center <= iv_end and seg_id not in existing_ids:
                                                    row_copy = frow.copy()
                                                    row_copy["label"] = fp_neg_label
                                                    row_copy["label_source"] = "temporal_feedback"
                                                    row_copy["reviewer_confidence"] = 1.0
                                                    injected_rows.append(row_copy)
                                                    existing_ids.add(seg_id)
                                    # Inject hard positives for FN intervals
                                    for sid, intervals in fn_map.items():
                                        sess_feat = feat_df[feat_df["session_id"].astype(str) == sid]
                                        for interval in intervals:
                                            if len(interval) < 2:
                                                continue
                                            iv_start, iv_end = int(interval[0]), int(interval[1])
                                            for _, frow in sess_feat.iterrows():
                                                seg_center = (int(frow["start_frame"]) + int(frow["end_frame"])) // 2
                                                seg_id = str(frow["segment_id"])
                                                if iv_start <= seg_center <= iv_end and seg_id not in existing_ids:
                                                    row_copy = frow.copy()
                                                    row_copy["label"] = target_label
                                                    row_copy["label_source"] = "temporal_feedback"
                                                    row_copy["reviewer_confidence"] = 1.0
                                                    injected_rows.append(row_copy)
                                                    existing_ids.add(seg_id)
                                    if injected_rows:
                                        inject_df = pd.DataFrame(injected_rows)
                                        # Align columns — only keep columns present in both
                                        shared_cols = [c for c in df.columns if c in inject_df.columns]
                                        df = pd.concat(
                                            [df, inject_df[shared_cols]],
                                            ignore_index=True,
                                        )
                                        n_fp_inject = sum(1 for r in injected_rows if r.get("label") == fp_neg_label)
                                        n_fn_inject = sum(1 for r in injected_rows if r.get("label") == target_label)
                                        logger.info(
                                            "Injected %d hard-negative + %d hard-positive "
                                            "segment(s) from temporal feedback for %s.",
                                            n_fp_inject,
                                            n_fn_inject,
                                            target_label,
                                        )
                                        _log(
                                            f"Temporal feedback: relabeled {n_relabeled}, "
                                            f"injected {n_fp_inject} hard-neg + {n_fn_inject} hard-pos"
                                        )
                            except Exception as exc:
                                logger.debug("Could not inject feedback segments: %s", exc)
            except Exception as exc:
                logger.debug("Could not apply temporal FP feedback: %s", exc)

        return df

    def _compute_sample_weights(
        self, fit_df: pd.DataFrame, cfg: "TrainingConfig"
    ) -> "tuple[np.ndarray, float, int, int, bool]":
        """Adaptive per-row sample weights from class imbalance.

        Returns ``(sample_weights, nb_weight, n_pos, n_neg, auto_applied)``.
        When ``no_behavior_sample_weight`` is 0 the negative weight is derived
        from the pos/neg ratio (clamped to [0.33, 3.0]); a manual value >0 is
        used directly.  The no_behavior model inverts which class is "negative".
        Shared by the held-out fit and the deploy refit so both weight identically.
        """
        sample_weights = np.ones(len(fit_df), dtype=float)
        labels_raw = fit_df["label"].astype(str)
        target_label_check = str(cfg.target_label or "").strip()
        _nb_tokens_sw = {"no_behavior", "no_behaviour", "nobehavior", "nobehaviour"}
        _target_is_nb_sw = (
            target_label_check.lower().replace("_", "").replace(" ", "") in _nb_tokens_sw
        )
        if _target_is_nb_sw:
            neg_mask = np.array([lbl.strip() == "has_behavior" for lbl in labels_raw])
        else:
            neg_mask = np.array([
                lbl.strip().lower().replace("_", "").replace(" ", "") in _nb_tokens_sw
                for lbl in labels_raw
            ])
        n_neg = int(neg_mask.sum())
        n_pos_w = len(fit_df) - n_neg
        nb_weight = float(cfg.no_behavior_sample_weight)
        auto_applied = False
        if nb_weight <= 0.0 and n_pos_w > 0 and n_neg > 0:
            nb_weight = float(np.clip(n_pos_w / n_neg, 0.33, 3.0))
            auto_applied = True
        elif nb_weight <= 0.0:
            nb_weight = 1.0
        if nb_weight != 1.0 and len(neg_mask):
            sample_weights[neg_mask] = nb_weight
        return sample_weights, nb_weight, n_pos_w, n_neg, auto_applied

    def _fit_with_gpu_fallback(
        self, est, params: "dict[str, Any]", x, y, sample_weights, cfg: "TrainingConfig"
    ):
        """Fit ``est`` with the XGBoost CUDA→CPU retry used across the service.

        Returns ``(est, used_cpu_fallback, model_device_requested,
        model_device_used, fallback_reason)``.  ``est`` may be replaced by a
        fresh CPU estimator on fallback.  Shared by the held-out fit and the
        deploy refit so both handle GPU failures identically.
        """
        used_cpu_fallback = False
        model_device_requested = str((params or {}).get("device", "")).strip().lower() or "auto"
        model_device_used = "cpu"
        fallback_reason = ""
        if cfg.classifier_family.lower().strip() == "xgboost":
            try:
                est_params = est.get_params() if hasattr(est, "get_params") else {}
                requested_from_est = str(est_params.get("device", model_device_requested)).strip().lower()
                if requested_from_est:
                    model_device_requested = requested_from_est
            except Exception:
                pass
        try:
            est.fit(x, y, sample_weight=sample_weights)
            if model_device_requested.startswith("cuda"):
                model_device_used = "gpu"
        except Exception as exc:
            family = cfg.classifier_family.lower().strip()
            requested_device = str((params or {}).get("device", "")).lower().strip()
            exc_msg = str(exc).lower()
            is_gpu_error = "cuda" in exc_msg or "gpu" in exc_msg or "invalid device" in exc_msg
            should_retry_xgb_cpu = (
                family == "xgboost"
                and (requested_device in {"cuda", "cuda:0"} or (requested_device == "" and is_gpu_error))
            )
            if not should_retry_xgb_cpu:
                raise
            logger.warning(
                "XGBoost CUDA training failed (%s). Retrying on CPU.",
                str(exc).splitlines()[0],
            )
            fallback_reason = str(exc).splitlines()[0]
            cpu_params = dict(params)
            cpu_params["device"] = "cpu"
            cpu_params.setdefault("tree_method", "hist")
            est = self._make_estimator("xgboost", cpu_params, cfg.random_state)
            est.fit(x, y, sample_weight=sample_weights)
            used_cpu_fallback = True
            model_device_used = "cpu"
        # Training is done; everything downstream (calibration, validation inference,
        # deployment, dense inference) feeds this model CPU numpy.  Leaving the booster
        # on cuda makes each of those copy the whole matrix host→device — ~7x slower
        # than just scoring on the CPU, and the source of XGBoost's "mismatched
        # devices" warning.  See abel.utils.xgb_predict.
        xgb_predict.ensure_cpu_prediction(est)
        return est, used_cpu_fallback, model_device_requested, model_device_used, fallback_reason

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        config: TrainingConfig | None = None,
        *,
        project_root: Path | None = None,
        precomputed_split: "tuple[np.ndarray, ...] | None" = None,
        feature_cols_override: "list[str] | None" = None,
        progress_cb: "Callable[[str], None] | None" = None,
    ) -> "TrainEvalResult":
        """Pure train+evaluate core shared by ``train`` and the validation engine.

        Operates on an in-memory training frame.  When ``precomputed_split`` is
        supplied the caller's (train_idx, val_idx) row positions are honoured
        instead of the internal group split — guaranteeing held-out rows never
        leak into training even across the preprocessing reordering below (the
        roles are tagged on a column that survives expansion/filtering).
        ``feature_cols_override`` intersects the engine-selected feature columns
        (used by ablation runs).  ``project_root`` is only read from (feature
        exclusions / trim report) and never written by this method.

        ``precomputed_split`` may carry an optional third index array,
        ``(train_idx, val_idx, cal_idx)``, naming rows used **only** to fit the
        probability calibrator.  Without it the calibrator falls back to fitting
        on the validation split — correct for the product (the calibrator must
        see the model's behaviour on unseen data, and the deployed model is
        refit and CV-calibrated anyway) but *not* for measurement: whoever
        scores those same validation rows is then reading a calibrator that was
        fit on them.  Measured across 7 projects / 23 behaviors at a 50-clip
        budget, that inflated calibration's apparent macro-F1 gain from +0.064
        to +0.073.  Callers who report metrics on the validation split should
        pass ``cal_idx``.
        """
        import time
        from sklearn.calibration import CalibratedClassifierCV, calibration_curve
        try:
            from sklearn.frozen import FrozenEstimator as _FrozenEstimator  # sklearn >=1.6
        except ImportError:
            _FrozenEstimator = None
        from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

        def _log(msg: str) -> None:
            if progress_cb is not None:
                progress_cb(msg)

        cfg = config or TrainingConfig()
        params = dict(cfg.classifier_params or {})
        fallback_reason = ""
        target_label = str(cfg.target_label or "").strip()
        _t_start = time.perf_counter()
        df = df.copy()

        # Tag caller-supplied split roles so they survive preprocessing reordering
        # (co-occurring expansion, untrainable filtering, label collapse, capping).
        _use_precomputed = precomputed_split is not None
        if _use_precomputed:
            _tr_idx0, _va_idx0 = precomputed_split[0], precomputed_split[1]
            _cal_idx0 = precomputed_split[2] if len(precomputed_split) > 2 else None
            _roles = np.array(["drop"] * len(df), dtype=object)
            _roles[np.asarray(_tr_idx0, dtype=int)] = "train"
            _roles[np.asarray(_va_idx0, dtype=int)] = "val"
            if _cal_idx0 is not None:
                _roles[np.asarray(_cal_idx0, dtype=int)] = "cal"
            df["_eval_split_role"] = _roles
            df = df[df["_eval_split_role"] != "drop"].reset_index(drop=True)

        # ── Expand co-occurring (pipe-separated) labels ───────────────────
        # When allow_co_occurring_behaviors is enabled, a single segment may
        # carry a label like "grooming|rearing".  Expand each such row into
        # one row per constituent label so that every behavior gets a positive
        # training example from the same feature vector.
        # _co_occurring_expanded marks rows that came from a pipe-separated clip so
        # that the label-collapse step below can DROP the sibling rows instead of
        # remapping them to no_behavior (which would make the same clip both a
        # positive AND a negative for a given behavior model).
        df["_co_occurring_expanded"] = False
        if cfg.allow_co_occurring_behaviors:
            multi_mask = df["label"].astype(str).str.contains(r"\|", na=False)
            if multi_mask.any():
                single_rows = df.loc[~multi_mask].copy()
                single_rows["_co_occurring_expanded"] = False
                expanded_parts: list[pd.DataFrame] = [single_rows]
                for idx_row, row in df.loc[multi_mask].iterrows():
                    for sub_label in str(row["label"]).split("|"):
                        sub_label = sub_label.strip()
                        if sub_label:
                            new_row = row.copy()
                            new_row["label"] = sub_label
                            new_row["_co_occurring_expanded"] = True
                            expanded_parts.append(pd.DataFrame([new_row]))
                df = pd.concat(expanded_parts, ignore_index=True)
                logger.info(
                    "Expanded %d co-occurring label rows into %d single-label rows.",
                    int(multi_mask.sum()),
                    len(df) - int((~multi_mask).sum()),
                )

        # Remove non-informative review outcomes before encoding labels.
        # "ambiguous" (reviewer disagreement) and "boundary_error" are not
        # valid training targets and would create a spurious third label class
        # that causes XGBoost binary:logistic to reject y as non-binary.
        _untrainable = {"ambiguous", "boundary_error"}
        trainable_mask = ~df["label"].astype(str).isin(_untrainable)
        df = df.loc[trainable_mask].copy()

        # ── Collapse alternate-behavior labels into negatives ─────────────
        # When training a binary classifier for target_label, any segment
        # labeled with a *different* behaviour name (e.g. "rearing" while
        # training "groom") should be treated as a negative example rather
        # than forming a spurious third class.  Remap all non-target,
        # non-negative labels to "no_behavior" so they contribute to the
        # negative pool.
        #
        # Special case: when target_label IS no_behavior, the roles are
        # inverted — segments labeled with any specific behaviour become
        # negatives (remapped to "has_behavior"), and no_behavior segments
        # remain as positives.
        #
        # Co-occurring expansion exception: rows produced by expanding a
        # pipe-separated multi-label clip (_co_occurring_expanded=True) that
        # do NOT match the current target should be DROPPED, not remapped to
        # no_behavior.  Remapping them would make the same clip simultaneously
        # a positive and a negative for a given behavior model.
        if target_label:
            _nb_tokens = {"no_behavior", "no_behaviour", "nobehavior", "nobehaviour"}
            _target_is_no_behavior = (
                target_label.strip().lower().replace("_", "").replace(" ", "")
                in _nb_tokens
            )
            _negative_prefixes = ("not_",)
            _co_exp_col = df.columns.get_loc("_co_occurring_expanded") if "_co_occurring_expanded" in df.columns else None
            labels_raw = df["label"].astype(str)
            n_remapped = 0
            drop_indices: list[int] = []

            if _target_is_no_behavior:
                # ── No-behavior model: specific behaviours → negative ──
                for i, lbl in enumerate(labels_raw):
                    lbl_clean = lbl.strip()
                    lbl_lower = lbl_clean.lower().replace("_", "").replace(" ", "")
                    if lbl_lower in _nb_tokens:
                        # Normalise all no_behavior variants to target_label
                        df.iat[i, df.columns.get_loc("label")] = target_label
                        continue  # positive — keep
                    # Co-occurring sibling: drop instead of remapping
                    if _co_exp_col is not None and df.iat[i, _co_exp_col]:
                        drop_indices.append(i)
                        continue
                    if any(lbl_clean.startswith(p) for p in _negative_prefixes):
                        # Explicit negative (not_xxx) — collapse to has_behavior
                        df.iat[i, df.columns.get_loc("label")] = "has_behavior"
                        n_remapped += 1
                        continue
                    # Specific behaviour → negative for no_behavior model
                    df.iat[i, df.columns.get_loc("label")] = "has_behavior"
                    n_remapped += 1
                if n_remapped > 0:
                    logger.info(
                        "No-behavior model: remapped %d specific-behavior label(s) "
                        "to has_behavior as negatives for target '%s'.",
                        n_remapped,
                        target_label,
                    )
            else:
                # ── Standard model: alternate behaviours → no_behavior ──
                for i, lbl in enumerate(labels_raw):
                    lbl_clean = lbl.strip()
                    lbl_lower = lbl_clean.lower().replace("_", "").replace(" ", "")
                    if lbl_clean == target_label:
                        continue  # positive — keep as-is
                    if lbl_lower in _nb_tokens:
                        continue  # already negative
                    # Co-occurring sibling that is NOT the target: drop, don't remap
                    if _co_exp_col is not None and df.iat[i, _co_exp_col]:
                        drop_indices.append(i)
                        continue
                    if any(lbl_clean.startswith(p) for p in _negative_prefixes):
                        # Explicit negative (not_xxx) — collapse to no_behavior
                        # so the binary encoder sees exactly two classes.
                        df.iat[i, df.columns.get_loc("label")] = "no_behavior"
                        n_remapped += 1
                        continue
                    # This is an alternate behaviour name → treat as negative
                    df.iat[i, df.columns.get_loc("label")] = "no_behavior"
                    n_remapped += 1
                if n_remapped > 0:
                    logger.info(
                        "Remapped %d alternate-behavior label(s) to no_behavior as "
                        "negatives for target '%s'.",
                        n_remapped,
                        target_label,
                    )

            if drop_indices:
                df = df.drop(index=df.index[drop_indices]).reset_index(drop=True)
                logger.info(
                    "Dropped %d co-occurring sibling row(s) for target '%s' "
                    "(same clip is a positive for a different behavior — not a negative).",
                    len(drop_indices),
                    target_label,
                )
        if df.empty:
            raise ValueError(
                "No trainable labeled rows remain after filtering ambiguous/boundary_error segments."
            )

        # Drop the internal tracking column before feature selection.
        if "_co_occurring_expanded" in df.columns:
            df = df.drop(columns=["_co_occurring_expanded"])

        # Cap samples per class for large-dataset runs.  Stratified per-class
        # sampling preserves label balance while reducing training time.
        if cfg.max_train_samples_per_class > 0:
            rng_cap = np.random.RandomState(int(cfg.random_state))
            parts: list[pd.DataFrame] = []
            for _lbl, _grp in df.groupby("label"):
                if len(_grp) > cfg.max_train_samples_per_class:
                    parts.append(
                        _grp.sample(
                            n=cfg.max_train_samples_per_class,
                            random_state=rng_cap.randint(0, 2**31),
                        )
                    )
                else:
                    parts.append(_grp)
            df = pd.concat(parts, ignore_index=True)
            logger.info(
                "Capped training set to max %d samples/class; %d total rows remain.",
                cfg.max_train_samples_per_class,
                len(df),
            )

        feature_cols = self._numeric_feature_cols(df)
        if not feature_cols:
            raise ValueError("No numeric feature columns in training set")

        # ── Apply project-level feature exclusions ────────────────
        # Read exclusions saved by the Feature Audit (config/feature_exclusions.json)
        # and feature-group selections from the Features tab.
        try:
            from abel.storage.file_store import read_json as _rj
            excl_path = project_root / "config" / "feature_exclusions.json"
            if excl_path.exists():
                excl_data = _rj(excl_path, {})
                excl_set = set(excl_data.get("excluded_feature_cols", []))
                # Remove managed prefixes — they are pattern markers, not column names
                excl_set = {e for e in excl_set if not e.startswith("__feat_group:")}
                if excl_set:
                    before = len(feature_cols)
                    feature_cols = [c for c in feature_cols if c not in excl_set]
                    n_project_excl = before - len(feature_cols)
                    if n_project_excl > 0:
                        logger.info(
                            "Applied %d project-level feature exclusion(s) from feature audit; "
                            "%d features remain.",
                            n_project_excl,
                            len(feature_cols),
                        )
                        _log(f"Applied {n_project_excl} project-level feature exclusions (feature audit).")

                # Apply feature-group exclusions from the Features tab
                disabled_groups = set(excl_data.get("disabled_feature_groups", []))
                if disabled_groups:
                    _GROUP_PATTERNS = {
                        "per_keypoint": ["_velocity_x", "_velocity_y", "_speed", "_acceleration", "_jerk"],
                        "global_speed": [
                            "centroid_velocity", "forepaw_speed", "forepaw_vertical_velocity",
                            "nose_velocity", "nose_vertical_velocity",
                        ],
                        "oscillation": [
                            "forepaw_oscillation_power", "nose_oscillation_power",
                            "forepaw_autocorr_peak", "nose_autocorr_peak",
                            "forepaw_movement_frequency", "nose_movement_frequency",
                            "oscillation_energy", "nose_oscillation_energy",
                        ],
                        "orientation": ["head_pitch", "body_orientation"],
                    }
                    before_group = len(feature_cols)
                    group_excl: set[str] = set()
                    for grp in disabled_groups:
                        for pattern in _GROUP_PATTERNS.get(grp, []):
                            for col in feature_cols:
                                if col == pattern or col.endswith(pattern) or pattern in col:
                                    group_excl.add(col)
                    if group_excl:
                        feature_cols = [c for c in feature_cols if c not in group_excl]
                        n_group_excl = before_group - len(feature_cols)
                        logger.info(
                            "Applied %d feature-group exclusion(s) from Features tab "
                            "(disabled groups: %s); %d features remain.",
                            n_group_excl, ", ".join(sorted(disabled_groups)),
                            len(feature_cols),
                        )
                        _log(
                            f"Applied {n_group_excl} feature-group exclusions "
                            f"(disabled: {', '.join(sorted(disabled_groups))})."
                        )
        except Exception as exc:
            logger.debug("Could not load feature exclusions: %s", exc)

        # ── Apply per-run (Active Learning tab) feature exclusions ──
        # These come from the in-tab feature-selection UI and are applied here
        # rather than in the representation so the cache is never invalidated.
        if cfg.excluded_feature_cols:
            _run_excl = set(cfg.excluded_feature_cols)
            before = len(feature_cols)
            feature_cols = [c for c in feature_cols if c not in _run_excl]
            n_run_excl = before - len(feature_cols)
            if n_run_excl > 0:
                logger.info(
                    "Applied %d per-run feature exclusion(s) from the Active Learning tab; "
                    "%d features remain.",
                    n_run_excl, len(feature_cols),
                )
                _log(f"Applied {n_run_excl} per-run feature exclusions (Active Learning selection).")
        if not feature_cols:
            raise ValueError("No numeric feature columns remain after feature exclusions.")

        # ── Drop dead features ─────────────────────────────────────
        # Columns that are constant, all-zero, or nearly all-NaN carry
        # no discriminative signal and inflate the effective feature space,
        # making overfitting more likely on small positive sets.
        if cfg.drop_zero_variance_features:
            col_std = df[feature_cols].std(axis=0)
            nan_frac = df[feature_cols].isna().mean(axis=0)
            nonzero_frac = (df[feature_cols].fillna(0.0) != 0).mean(axis=0)
            dead_mask = (col_std.abs() < 1e-12) | col_std.isna() | (nan_frac > 0.99)
            n_dead = int(dead_mask.sum())

            # Also flag columns that are >95% zero (weak signal).
            weak_mask = ~dead_mask & (nonzero_frac < 0.05)
            n_weak = int(weak_mask.sum())

            dropped_names = [c for c, is_dead in zip(feature_cols, dead_mask) if is_dead]
            weak_names = [c for c, is_weak in zip(feature_cols, weak_mask) if is_weak]

            if n_dead > 0:
                feature_cols = [c for c, is_dead in zip(feature_cols, dead_mask) if not is_dead]
                logger.info(
                    "Dropped %d dead feature(s) (zero variance / all NaN); %d informative features remain.",
                    n_dead,
                    len(feature_cols),
                )
                _log(f"Dropped {n_dead} dead features; {len(feature_cols)} remain.")
                if dropped_names:
                    logger.debug("Dead features dropped: %s", dropped_names[:20])

            if n_weak > 0:
                logger.info(
                    "%d weak feature(s) detected (>95%% zero). These are kept but may add noise: %s",
                    n_weak,
                    weak_names[:10],
                )
                _log(f"Warning: {n_weak} weak features detected (>95% zero). Consider reviewing in Feature Audit.")

            # Persist the trimmed features list for downstream inspection.
            try:
                trim_report = {
                    "dead_dropped": dropped_names,
                    "weak_kept": weak_names,
                    "n_features_after": len(feature_cols),
                }
                trim_dir = project_root / "derived" / "training_sets"
                trim_dir.mkdir(parents=True, exist_ok=True)
                trim_path = trim_dir / "feature_trim_report.json"
                from abel.storage.file_store import write_json as _wj
                _wj(trim_report, trim_path)
            except Exception:
                pass  # non-critical

            if not feature_cols:
                raise ValueError("All numeric feature columns have zero variance")

        # ── Ablation feature filtering ────────────────────────────
        # Intersect the engine-selected features with the caller's allow-list
        # (e.g. "video features off" / "video only" ablation runs).
        if feature_cols_override is not None:
            _ov = set(feature_cols_override)
            feature_cols = [c for c in feature_cols if c in _ov]
            if not feature_cols:
                raise ValueError("No feature columns remain after feature_cols_override.")
            _log(f"feature_cols_override: {len(feature_cols)} feature(s) retained.")

        # ── Very-low-sample guard: skip the hold-out split ────────────────
        # With only a handful of labeled clips, reserving a test set wastes
        # scarce positives and yields meaningless metrics.  Train on ALL rows
        # instead and flag the metrics as in-sample.  Honoured only for the
        # internal split — a caller-supplied precomputed split (validation /
        # benchmark platform) always keeps its held-out rows.
        _no_holdout = False
        cal_idx = np.empty(0, dtype=int)
        if _use_precomputed:
            _roles_final = df["_eval_split_role"].to_numpy()
            train_idx = np.where(_roles_final == "train")[0]
            val_idx = np.where(_roles_final == "val")[0]
            cal_idx = np.where(_roles_final == "cal")[0]
        elif len(df) < int(cfg.min_holdout_samples):
            _no_holdout = True
            train_idx = np.arange(len(df), dtype=int)
            val_idx = train_idx
            logger.warning(
                "Very low sample size (%d labeled row(s) < min_holdout_samples=%d): "
                "skipping the held-out test split. Training on ALL examples; "
                "reported metrics are in-sample (training data), not a "
                "generalization estimate.",
                len(df), int(cfg.min_holdout_samples),
            )
            _log(
                f"Very low sample size ({len(df)} labeled rows): not holding out any "
                f"examples — training on all data. Metrics are in-sample, not held-out."
            )
        else:
            train_idx, val_idx = self._split(df, cfg.split_strategy, cfg.test_size, cfg.random_state)
        if "_eval_split_role" in df.columns:
            df = df.drop(columns=["_eval_split_role"])
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        # Dedicated calibration rows (never trained on, never scored) — empty
        # unless the caller supplied cal_idx.
        cal_df = df.iloc[cal_idx]

        # Some labels are training aids, not held-out evaluation data, and must
        # never enter the validation split or they pollute the reported metrics:
        #   * temporal-review feedback (``temporal_feedback``) — FP/FN corrections
        #     a reviewer commits to CORRECT the model, not to grade it;
        #   * cross-project imported examples (``imported:*``) — they come from
        #     OTHER projects/subjects, so scoring them here isn't a clean
        #     held-out generalization test for THIS project.
        # Both are kept in training but moved out of validation.
        if "label_source" in val_df.columns and not val_df.empty:
            _src = val_df["label_source"].astype(str)
            _refine_only = _src.eq("temporal_feedback") | _src.str.startswith("imported:")
            _n_refine = int(_refine_only.sum())
            if _n_refine:
                # In no-holdout mode train_df already contains every row, so only
                # trim them out of validation — re-adding would duplicate them.
                if not _no_holdout:
                    train_df = pd.concat([train_df, val_df.loc[_refine_only]])
                val_df = val_df.loc[~_refine_only]
                logger.info(
                    "Excluded %d refinement-only row(s) (temporal_feedback / imported) "
                    "from validation; kept for training.",
                    _n_refine,
                )

        # Build label map from the training split so encoded classes remain
        # contiguous (0..K-1). Some estimators (notably XGBoost) reject sparse
        # encodings like [0, 2] when only two classes are present in-train.
        y_train, inv = self._label_map(train_df["label"])
        label_to_idx = {v: k for k, v in inv.items()}
        # Re-derive y_train for strict consistency with label_to_idx.
        y_train = np.asarray([label_to_idx[str(lbl)] for lbl in train_df["label"]], dtype=int)
        keep_val = val_df["label"].astype(str).isin(label_to_idx)
        val_df = val_df.loc[keep_val].copy()
        if val_df.empty:
            # Too few labeled rows to produce a valid hold-out split (e.g. only
            # one example of each class).  Fall back to using the training set
            # as the validation set so the pipeline can complete and generate
            # candidates for the next review round.
            logger.warning(
                "Validation split has no labels seen in training split "
                "(%d total labeled row(s)); using training set as validation "
                "for this bootstrap run.",
                len(df),
            )
            val_df = train_df.copy()
        y_val = np.asarray([label_to_idx[str(label)] for label in val_df["label"]], dtype=int)
        # Same label-map filtering for the calibration split: a calibrator can
        # only be fit on classes the model was actually trained to emit.
        if not cal_df.empty:
            cal_df = cal_df.loc[cal_df["label"].astype(str).isin(label_to_idx)].copy()
        y_cal = np.asarray(
            [label_to_idx[str(label)] for label in cal_df["label"]], dtype=int
        ) if not cal_df.empty else np.empty(0, dtype=int)

        # ── Adaptive model complexity ─────────────────────────────
        # When enabled, scale tree count and depth to the ratio of
        # positive examples vs features.  Small positive sets with many
        # features overfit rapidly with 300 deep trees.
        if cfg.adaptive_complexity and "n_estimators" not in (cfg.classifier_params or {}) and "max_depth" not in (cfg.classifier_params or {}):
            # Count positives: rows whose label matches target_label.
            # This works regardless of whether target is a specific
            # behaviour or no_behavior.
            _ac_target = str(cfg.target_label or "").strip()
            n_pos = int(sum(
                1 for lbl in train_df["label"].astype(str)
                if lbl.strip() == _ac_target
            ))
            n_feat = len(feature_cols)
            # Heuristic: the ratio of positive examples to features determines
            # how much capacity the model can safely use.
            ratio = n_pos / max(1, n_feat)
            if ratio < 1.0:
                adaptive_depth = 4
                adaptive_trees = max(50, min(150, int(n_pos * 0.5)))
            elif ratio < 2.0:
                adaptive_depth = 5
                adaptive_trees = max(100, min(200, int(n_pos * 0.6)))
            else:
                adaptive_depth = 6
                adaptive_trees = 300
            params["max_depth"] = adaptive_depth
            params["n_estimators"] = adaptive_trees
            # Also increase regularisation for small datasets
            if n_pos < 500:
                params.setdefault("reg_alpha", 0.3)
                params.setdefault("reg_lambda", 2.0)
                params.setdefault("min_child_weight", max(5, int(n_pos * 0.03)))
            logger.info(
                "Adaptive complexity: %d positives / %d features (ratio=%.2f) "
                "→ max_depth=%d, n_estimators=%d.",
                n_pos, n_feat, ratio, adaptive_depth, adaptive_trees,
            )
            _log(f"Adaptive complexity: depth={adaptive_depth}, trees={adaptive_trees} "
                 f"(positives={n_pos}, features={n_feat}, ratio={ratio:.2f})")

        _log(f"Prepared {len(train_df)} train / {len(val_df)} val rows with {len(feature_cols)} features.")
        est = self._make_estimator(cfg.classifier_family, params, cfg.random_state)
        x_train_np = train_df[feature_cols].to_numpy(dtype=float)

        # ── Adaptive sample weights from class imbalance ──────────────
        # When no_behavior_sample_weight is 0 (auto), compute the weight
        # from the actual positive/negative ratio so that the effective
        # class balance is approximately 1:1.  For manually specified
        # weights (>0), use the provided value directly.
        #
        # When training the no_behavior model, the positive class IS
        # no_behavior and negatives are "has_behavior".  We detect this
        # and invert the weighting direction so the minority class
        # (whichever it is) receives the upweight.
        sample_weights, nb_weight, n_pos_w, n_neg, _auto_weight_applied = (
            self._compute_sample_weights(train_df, cfg)
        )
        if _auto_weight_applied:
            logger.info(
                "Adaptive sample weight: %d positives / %d negatives → "
                "negative_weight=%.3f.",
                n_pos_w, n_neg, nb_weight,
            )
            _log(f"Adaptive sample weight: {n_pos_w} pos / {n_neg} neg → weight={nb_weight:.3f}")

        # ── Training-time feature augmentation ────────────────────────
        # Creates synthetic positive examples with Gaussian jitter + feature
        # dropout to improve robustness with small labeled sets and reduce
        # overfitting to tracking artefacts.
        if cfg.enable_feature_augmentation:
            target_label_str = str(cfg.target_label or "").strip()
            target_label_idx = label_to_idx.get(target_label_str)
            if target_label_idx is not None:
                rng_aug = np.random.RandomState(int(cfg.random_state) + 1)
                n_before = len(y_train)
                x_train_np, y_train, sample_weights = self._augment_training_features(
                    x_train_np, y_train, sample_weights,
                    target_label_idx=target_label_idx,
                    jitter_sigma=cfg.augmentation_jitter_sigma,
                    dropout_prob=cfg.augmentation_dropout_prob,
                    n_copies=cfg.augmentation_copies,
                    rng=rng_aug,
                )
                n_added = len(y_train) - n_before
                _log(
                    f"Feature augmentation: added {n_added} synthetic positives "
                    f"({cfg.augmentation_copies}× copies, "
                    f"jitter={cfg.augmentation_jitter_sigma}, "
                    f"dropout={cfg.augmentation_dropout_prob})"
                )

        _log(f"Fitting {cfg.classifier_family} (device={str(params.get('device', 'auto'))})…")
        est, used_cpu_fallback, model_device_requested, model_device_used, fallback_reason = (
            self._fit_with_gpu_fallback(est, params, x_train_np, y_train, sample_weights, cfg)
        )

        if cfg.require_gpu and model_device_used != "gpu":
            reason = fallback_reason or "Training backend did not execute on GPU."
            raise RuntimeError(f"Strict GPU mode enabled, but model training used CPU. Reason: {reason}")

        _log(f"Model fit complete (device={model_device_used}).")
        clf = est
        # Detect degenerate validation split early — needed to guard calibration.
        _degenerate_val_early = len(set(y_val.tolist())) < 2
        # Prefer a dedicated calibration split when the caller supplied one: it
        # keeps the calibrator off the rows the run is scored on.  Falls back to
        # the validation split (product behaviour) when absent or single-class.
        _cal_supplied = _use_precomputed and len(cal_idx) > 0
        _use_cal_split = len(y_cal) > 0 and len(set(y_cal.tolist())) > 1
        # A caller that supplied cal_idx is telling us it will score the
        # validation split, so calibrating on that split would be a leak.  If the
        # supplied slice turns out to be unusable, skip calibration entirely
        # rather than quietly falling back to it.
        _cal_unusable = _cal_supplied and not _use_cal_split
        if _cal_unusable:
            logger.warning(
                "Calibration split has only one class (%d row(s)); skipping "
                "calibration — refusing to fall back to the validation split, "
                "which the caller scores.",
                len(y_cal),
            )
        if cfg.calibration_method in {"sigmoid", "isotonic"} and not _cal_unusable:
            if _degenerate_val_early and not _use_cal_split:
                logger.warning(
                    "Skipping calibration: validation set has only one class (%d row(s)). "
                    "Metrics will be marked as unreliable.",
                    len(y_val),
                )
            else:
                _src_name = "dedicated calibration split" if _use_cal_split else "held-out validation data"
                _log(f"Calibrating predictions on {_src_name} ({cfg.calibration_method})…")
                method = "sigmoid" if cfg.calibration_method == "sigmoid" else "isotonic"
                if _FrozenEstimator is not None:
                    calibrated = CalibratedClassifierCV(estimator=_FrozenEstimator(est), method=method)
                else:
                    calibrated = CalibratedClassifierCV(estimator=est, method=method, cv="prefit")
                # Fit the calibrator on data the model has NOT been trained on, so
                # the sigmoid learns from its behaviour on unseen rows.  Fitting on
                # training data (the original approach) made the sigmoid learn the
                # overconfident training-time probability distribution, which then
                # compressed moderate predictions on novel subjects toward zero.
                # A dedicated calibration split is preferred when supplied; the
                # validation split is the product fallback.
                _cal_frame, _cal_y = (cal_df, y_cal) if _use_cal_split else (val_df, y_val)
                calibrated.fit(_cal_frame[feature_cols].to_numpy(dtype=float), _cal_y)
                clf = calibrated

        _log("Running validation inference…")
        probs = xgb_predict.predict_proba(clf, val_df[feature_cols].to_numpy(dtype=float))
        preds = np.argmax(probs, axis=1)

        precision = float(precision_score(y_val, preds, average="macro", zero_division=0))
        recall = float(recall_score(y_val, preds, average="macro", zero_division=0))
        f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        target_label = str(cfg.target_label or "").strip()
        target_idx = next((idx for idx, name in inv.items() if str(name) == target_label), None) if target_label else None
        if target_idx is None:
            target_idx = next(
                (idx for idx, name in inv.items() if not str(name).startswith("not_") and str(name) not in {"ambiguous", "boundary_error"}),
                None,
            )
        if target_idx is None and probs.shape[1] == 2:
            target_idx = 1
        if target_idx is None or int(target_idx) >= probs.shape[1]:
            pr_auc = float("nan")
        else:
            pr_auc = float(average_precision_score((y_val == int(target_idx)).astype(int), probs[:, int(target_idx)]))
        cm = confusion_matrix(y_val, preds).tolist()

        # Detect degenerate validation split — if the confusion matrix is 1×1,
        # only one class is present in validation and all metrics are trivially
        # perfect/zero.  Flag this clearly so the user re-trains with a better split.
        n_classes_in_val = len(set(y_val))
        degenerate_val = n_classes_in_val < 2
        if degenerate_val:
            _log(
                "WARNING: Validation set contains only one class — "
                "metrics (F1, PR-AUC) are unreliable.  "
                "Consider increasing the number of reviewed examples or "
                "using a different split strategy."
            )
            # Override metrics to signal invalidity.
            f1 = float("nan")
            pr_auc = float("nan")
            precision = float("nan")
            recall = float("nan")

        if probs.shape[1] == 2:
            cal_true, cal_pred = calibration_curve((y_val == 1).astype(int), probs[:, 1], n_bins=10)
            calibration = {"prob_true": cal_true.tolist(), "prob_pred": cal_pred.tolist()}
        else:
            calibration = {"prob_true": [], "prob_pred": []}

        _log(f"Validation: F1={f1:.3f}, PR-AUC={pr_auc:.3f}, precision={precision:.3f}, recall={recall:.3f}")

        # ── Extract feature importance from the raw estimator ─────────
        feature_importance: dict[str, float] = {}
        try:
            raw_est = est  # uncalibrated estimator
            if hasattr(raw_est, "feature_importances_"):
                importances = np.asarray(raw_est.feature_importances_, dtype=float)
                for col_name, imp in zip(feature_cols, importances):
                    feature_importance[col_name] = round(float(imp), 6)
            elif hasattr(raw_est, "get_booster"):
                booster = raw_est.get_booster()
                fscore = booster.get_score(importance_type="gain")
                for fidx_str, score in fscore.items():
                    idx = int(fidx_str.replace("f", ""))
                    if 0 <= idx < len(feature_cols):
                        feature_importance[feature_cols[idx]] = round(float(score), 6)
            if feature_importance:
                _log(f"Extracted feature importance for {len(feature_importance)} features.")
        except Exception as exc:
            logger.debug("Could not extract feature importance: %s", exc)

        # ── Refit the DEPLOYED model on ALL labeled rows ──────────────────
        # The metrics/split_manifest above stay honest (held-out split).  But
        # the model we persist should learn from every labeled row — including
        # the held-out sessions and any temporal-feedback corrections on them —
        # so per-mouse fixes actually reach inference.  Skipped when a caller
        # supplied a precomputed split (validation/benchmark platform needs the
        # held-out model) or when the split had no genuine hold-out.
        deploy_est = est
        deploy_clf = clf
        deploy_inv = inv
        deploy_target_idx = target_idx
        deployed_on = "train_split"
        n_deploy_train = int(len(train_df))
        _has_holdout = (
            precomputed_split is None
            and len(train_idx) > 0
            and len(val_idx) > 0
            and int(len(train_idx)) < int(len(df))
        )
        if cfg.deploy_refit_on_all_data and _has_holdout:
            try:
                _log("Refitting deployed model on all labeled data (train + held-out)…")
                d_y_full, deploy_inv = self._label_map(df["label"])
                d_label_to_idx = {v: k for k, v in deploy_inv.items()}
                d_x = df[feature_cols].to_numpy(dtype=float)
                d_weights_noaug, _, _, _, _ = self._compute_sample_weights(df, cfg)
                d_x_fit, d_y_fit, d_weights_fit = d_x, d_y_full, d_weights_noaug
                if cfg.enable_feature_augmentation:
                    d_target_idx_aug = d_label_to_idx.get(str(cfg.target_label or "").strip())
                    if d_target_idx_aug is not None:
                        rng_aug_d = np.random.RandomState(int(cfg.random_state) + 2)
                        d_x_fit, d_y_fit, d_weights_fit = self._augment_training_features(
                            d_x.copy(), d_y_full.copy(), d_weights_noaug.copy(),
                            target_label_idx=d_target_idx_aug,
                            jitter_sigma=cfg.augmentation_jitter_sigma,
                            dropout_prob=cfg.augmentation_dropout_prob,
                            n_copies=cfg.augmentation_copies,
                            rng=rng_aug_d,
                        )
                d_est = self._make_estimator(cfg.classifier_family, params, cfg.random_state)
                d_est, _, _, d_device_used, _ = self._fit_with_gpu_fallback(
                    d_est, params, d_x_fit, d_y_fit, d_weights_fit, cfg
                )
                deploy_est = d_est
                deploy_clf = d_est
                # Calibrate via internal CV — refitting on all data leaves no
                # separate hold-out, so cross-validated calibration is used
                # instead of the prefit-on-validation path.
                if cfg.calibration_method in {"sigmoid", "isotonic"}:
                    _, d_counts = np.unique(d_y_full, return_counts=True)
                    n_cv = int(max(2, min(3, int(d_counts.min())))) if len(d_counts) >= 2 else 0
                    if n_cv >= 2 and int(d_counts.min()) >= 2:
                        method = "sigmoid" if cfg.calibration_method == "sigmoid" else "isotonic"
                        try:
                            fresh = self._make_estimator(cfg.classifier_family, params, cfg.random_state)
                            d_cal = CalibratedClassifierCV(estimator=fresh, method=method, cv=n_cv)
                            d_cal.fit(d_x, d_y_full, sample_weight=d_weights_noaug)
                            deploy_clf = d_cal
                        except Exception as cal_exc:
                            logger.warning(
                                "Deploy CV calibration failed (%s); shipping uncalibrated all-data model.",
                                cal_exc,
                            )
                            deploy_clf = d_est
                deploy_target_idx = next(
                    (idx for idx, name in deploy_inv.items() if str(name) == str(cfg.target_label or "").strip()),
                    None,
                )
                if deploy_target_idx is None:
                    deploy_target_idx = next(
                        (idx for idx, name in deploy_inv.items()
                         if not str(name).startswith("not_") and str(name) not in {"ambiguous", "boundary_error"}),
                        target_idx,
                    )
                n_deploy_train = int(len(df))
                deployed_on = "all_data"
                _log(f"Deployed model refit on all {n_deploy_train} labeled rows (device={d_device_used}).")
                logger.info(
                    "Deployed model refit on all labeled data: %d rows "
                    "(reported metrics remain from the held-out split).",
                    n_deploy_train,
                )
            except Exception as exc:
                logger.warning(
                    "Deploy refit on all data failed (%s); shipping the held-out model instead.",
                    exc,
                )
                deploy_est, deploy_clf, deploy_inv, deploy_target_idx = est, clf, inv, target_idx
                deployed_on = "train_split"
                n_deploy_train = int(len(train_df))

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "calibration_curve": calibration,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_features": len(feature_cols),
            "holdout": (not _no_holdout),
            "evaluated_on": ("train_in_sample" if _no_holdout else "held_out"),
            "adaptive_complexity": bool(cfg.adaptive_complexity),
            "adaptive_sample_weight": float(nb_weight),
            "dropped_zero_variance_features": bool(cfg.drop_zero_variance_features),
            "calibration_fitted_on": "validation",
            "deployed_on": deployed_on,
            "n_deploy_train": int(n_deploy_train),
            "used_cpu_fallback": bool(used_cpu_fallback),
            "model_device_requested": model_device_requested,
            "model_device_used": model_device_used,
            "fallback_reason": fallback_reason,
        }
        split_manifest = {
            "strategy": cfg.split_strategy,
            "train_sessions": sorted(set(train_df["session_id"].astype(str))),
            "val_sessions": sorted(set(val_df["session_id"].astype(str))),
            "train_animals": sorted(set(train_df["animal_id"].astype(str))),
            "val_animals": sorted(set(val_df["animal_id"].astype(str))),
        }
        _meta_cols = ["segment_id", "animal_id", "session_id"]
        for _c in ("start_frame", "end_frame"):
            if _c in val_df.columns:
                _meta_cols.append(_c)
        val_meta = val_df[_meta_cols].copy()

        # Ship the deployed model (refit on all data when enabled); the
        # validation artifacts (y_val/val_probs/val_preds/metrics/split_manifest)
        # stay from the honest held-out evaluation above.
        return TrainEvalResult(
            fitted_estimator=deploy_est,
            calibrated_model=deploy_clf,
            feature_cols=feature_cols,
            label_map=deploy_inv,
            target_idx=(int(deploy_target_idx) if deploy_target_idx is not None else None),
            y_val=y_val,
            val_probs=probs,
            val_preds=preds,
            val_meta=val_meta,
            metrics=metrics,
            feature_importance=feature_importance,
            split_manifest=split_manifest,
            elapsed_sec=float(time.perf_counter() - _t_start),
            degenerate_val=bool(degenerate_val),
        )

    def _write_model_artifacts(
        self,
        project_root: Path,
        cfg: TrainingConfig,
        result: "TrainEvalResult",
        _log: "Callable[[str], None]",
    ) -> dict[str, Any]:
        """Persist model + metrics artifacts from a TrainEvalResult (unchanged layout)."""
        model_dir = project_root / "derived" / "models" / cfg.model_version
        model_dir.mkdir(parents=True, exist_ok=True)

        pred_df = result.val_meta[["segment_id", "animal_id", "session_id"]].copy()
        pred_df["label_true"] = result.y_val
        pred_df["label_pred"] = result.val_preds
        # Persist the held-out target probability so downstream tools can grade
        # the model honestly at a chosen threshold and after temporal refinement.
        # segment_predictions.parquet cannot be used for this: it is produced by
        # the deploy model, which is refit on ALL data (val included), so scoring
        # the held-out set with it leaks. These probabilities come from the
        # honest train/val split and are the only leak-free per-segment scores.
        if (
            result.val_probs is not None
            and result.target_idx is not None
            and getattr(result.val_probs, "ndim", 0) == 2
            and 0 <= int(result.target_idx) < result.val_probs.shape[1]
            and result.val_probs.shape[0] == len(pred_df)
        ):
            pred_df["prediction_prob"] = np.asarray(result.val_probs)[:, int(result.target_idx)].astype(float)
            pred_df["target_index"] = int(result.target_idx)
        pred_df.to_parquet(model_dir / "validation_predictions.parquet", index=False)

        _log("Saving model artifacts…")
        with open(model_dir / "model_state.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": result.calibrated_model,
                    "feature_cols": result.feature_cols,
                    "label_map": result.label_map,
                },
                f,
            )

        prov = self._provenance.make_provenance(
            project_root=project_root,
            model_version=cfg.model_version,
            feature_version=cfg.feature_version,
            config={"training_config": cfg.__dict__, "feature_cols": result.feature_cols},
        )

        metrics = result.metrics
        write_json(model_dir / "metrics.json", metrics)
        write_json(
            model_dir / "calibration_report.json",
            metrics.get("calibration_curve", {"prob_true": [], "prob_pred": []}),
        )
        if result.feature_importance:
            # Sort by importance descending for easy reading.
            sorted_importance = dict(
                sorted(result.feature_importance.items(), key=lambda kv: kv[1], reverse=True)
            )
            write_json(model_dir / "feature_importance.json", sorted_importance)
        write_json(model_dir / "split_manifest.json", result.split_manifest)

        card = ModelCard(
            model_version=cfg.model_version,
            classifier_family=cfg.classifier_family,
            calibration_method=cfg.calibration_method,
            training_split_strategy=cfg.split_strategy,
            labels=[result.label_map[i] for i in sorted(result.label_map)],
            feature_columns=result.feature_cols,
            metrics=metrics,
            provenance=prov,
        )
        write_yaml(model_dir / "model_card.yaml", card.model_dump(mode="json"))

        return {
            "model_dir": str(model_dir),
            "metrics": metrics,
            "feature_cols": result.feature_cols,
            "feature_importance": result.feature_importance,
            "model_device_requested": metrics.get("model_device_requested"),
            "model_device_used": metrics.get("model_device_used"),
            "used_cpu_fallback": bool(metrics.get("used_cpu_fallback")),
            "fallback_reason": metrics.get("fallback_reason", ""),
        }

    @staticmethod
    def predict_segments(model_dir: Path, segment_df: pd.DataFrame) -> pd.DataFrame:
        with open(model_dir / "model_state.pkl", "rb") as f:
            payload = pickle.load(f)

        clf = payload["model"]
        feature_cols: list[str] = payload["feature_cols"]
        probs = xgb_predict.predict_proba(clf, segment_df[feature_cols].to_numpy(dtype=float))
        max_prob = probs.max(axis=1)

        out = segment_df[["segment_id", "start_frame", "end_frame", "animal_id", "session_id"]].copy()
        out["prediction_prob"] = max_prob
        return out
