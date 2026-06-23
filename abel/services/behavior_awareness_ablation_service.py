"""Ablation service comparing behavior-aware vs. behavior-unaware pipeline modes.

Runs three lightweight tests on the current project data:

1. **Candidate ranking quality** — does cross-behavior competition scoring
   surface better candidates (higher rank for user-accepted segments)?
2. **Temporal refinement quality** — does mutual inhibition improve
   frame-level classification metrics on the validation set?
3. **Model feature ablation** — does augmenting training features with
   peer-behavior prediction scores improve cross-validated model quality?

All tests are non-destructive and do not modify project artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("abel")


@dataclass
class AblationResult:
    """Container for the three-part ablation comparison."""

    # Overall verdict: "aware_better", "unaware_better", "inconclusive"
    verdict: str = "inconclusive"
    summary: str = ""

    # Test 1 — candidate ranking
    candidate_test_ran: bool = False
    candidate_mrr_aware: float = float("nan")
    candidate_mrr_unaware: float = float("nan")
    candidate_detail: str = ""

    # Test 2 — temporal refinement (mutual inhibition)
    temporal_test_ran: bool = False
    temporal_f1_aware: float = float("nan")
    temporal_f1_unaware: float = float("nan")
    temporal_detail: str = ""

    # Test 3 — model feature ablation
    model_test_ran: bool = False
    model_f1_aware: float = float("nan")
    model_f1_unaware: float = float("nan")
    model_prauc_aware: float = float("nan")
    model_prauc_unaware: float = float("nan")
    model_detail: str = ""

    warnings: list[str] = field(default_factory=list)


class BehaviorAwarenessAblationService:
    """Runs a lightweight ablation comparing behavior-aware vs. unaware modes."""

    @staticmethod
    def _resolve_model_dir(project_root: Path, target_behavior: str) -> tuple[str, Path] | None:
        """Find the most recent model directory for *target_behavior*.

        Model directories are named ``behavior_model_{name}_{timestamp}`` or
        plain ``behavior_model_{name}``.  Returns ``(version_str, path)`` for
        the newest match, or ``None`` if nothing exists.
        """
        import re as _re

        models_root = project_root / "derived" / "models"
        if not models_root.exists():
            return None

        prefix = f"behavior_model_{target_behavior}"
        # Match exact name or name followed by _YYYYMMDD_HHMMSS timestamp
        pattern = _re.compile(
            rf"^{_re.escape(prefix)}(?:_\d{{8}}_\d{{6}})?$"
        )

        candidates: list[Path] = []
        for d in models_root.iterdir():
            if d.is_dir() and pattern.match(d.name):
                candidates.append(d)

        if not candidates:
            # Fallback: try behavior_model_v1
            fallback = models_root / "behavior_model_v1"
            if fallback.exists():
                return ("behavior_model_v1", fallback)
            return None

        # Latest by directory name (timestamp sorts lexicographically)
        best = sorted(candidates, key=lambda p: p.name)[-1]
        return (best.name, best)

    def _is_target_model_dir(self, dir_name: str, target_behavior: str) -> bool:
        """Return True if *dir_name* belongs to the target behavior."""
        import re as _re

        prefix = f"behavior_model_{target_behavior}"
        pattern = _re.compile(
            rf"^{_re.escape(prefix)}(?:_\d{{8}}_\d{{6}})?$"
        )
        return bool(pattern.match(dir_name))

    def run_ablation(
        self,
        project_root: Path,
        target_behavior: str,
        classifier_family: str = "lightgbm",
        n_folds: int = 3,
        progress_cb: Callable[[str], None] | None = None,
    ) -> AblationResult:
        def _log(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        result = AblationResult()
        _log("Starting behavior-awareness ablation study…")

        # --- Test 1: Candidate ranking quality ---
        try:
            _log("[1/3] Comparing candidate ranking strategies…")
            self._test_candidate_ranking(project_root, target_behavior, result, _log)
        except Exception as exc:
            result.warnings.append(f"Candidate ranking test failed: {exc}")
            logger.debug("Candidate ranking ablation error", exc_info=True)

        # --- Test 2: Temporal refinement (mutual inhibition) ---
        try:
            _log("[2/3] Comparing temporal refinement with/without mutual inhibition…")
            self._test_temporal_inhibition(project_root, target_behavior, result, _log)
        except Exception as exc:
            result.warnings.append(f"Temporal refinement test failed: {exc}")
            logger.debug("Temporal ablation error", exc_info=True)

        # --- Test 3: Model feature ablation ---
        try:
            _log("[3/3] Cross-validating model with/without peer-behavior features…")
            self._test_model_feature_ablation(
                project_root, target_behavior, classifier_family, n_folds, result, _log,
            )
        except Exception as exc:
            result.warnings.append(f"Model feature ablation failed: {exc}")
            logger.debug("Model feature ablation error", exc_info=True)

        # --- Synthesize verdict ---
        result.verdict, result.summary = self._synthesize_verdict(result)
        _log(f"Ablation complete. Verdict: {result.verdict}")
        _log(result.summary)
        return result

    # ------------------------------------------------------------------
    # Test 1 — Candidate ranking quality
    # ------------------------------------------------------------------

    def _test_candidate_ranking(
        self,
        project_root: Path,
        target_behavior: str,
        result: AblationResult,
        _log: Callable[[str], None],
    ) -> None:
        labels_path = project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if not labels_path.exists():
            result.warnings.append("No reviewer labels found — skipping candidate ranking test.")
            return

        labels = pd.read_parquet(labels_path)
        if labels.empty or "review_label" not in labels.columns:
            result.warnings.append("Reviewer labels empty or missing review_label column.")
            return

        # Determine which segments were accepted vs. rejected
        _positive_tokens = {target_behavior, target_behavior.lower()}
        _nb_tokens = {"no_behavior", "no_behaviour", "rejected", "false_positive"}
        labels["is_accepted"] = labels["review_label"].astype(str).str.strip().str.lower().isin(
            {t.lower() for t in _positive_tokens}
        )
        labels["is_rejected"] = labels["review_label"].astype(str).str.strip().str.lower().isin(_nb_tokens)
        reviewed = labels[labels["is_accepted"] | labels["is_rejected"]].copy()
        if reviewed.empty or not reviewed["is_accepted"].any():
            result.warnings.append("Not enough reviewed segments (need at least 1 accepted).")
            return

        # Load prediction + peer scores
        resolved = self._resolve_model_dir(project_root, target_behavior)
        if resolved is None:
            result.warnings.append("No segment predictions found — skipping candidate ranking test.")
            return
        model_version, model_dir = resolved
        pred_path = model_dir / "segment_predictions.parquet"
        if not pred_path.exists():
            result.warnings.append("No segment predictions found — skipping candidate ranking test.")
            return

        preds = pd.read_parquet(pred_path)
        merged = reviewed.merge(preds[["segment_id", "prediction_prob"]], on="segment_id", how="inner")
        if merged.empty:
            result.warnings.append("No overlap between reviewed segments and predictions.")
            return

        # Compute peer competition scores
        peer_max = self._compute_peer_max_prob(project_root, model_version, merged["segment_id"])

        # Ranking A: behavior-aware (predicted prob penalised by peer competition)
        merged["aware_score"] = merged["prediction_prob"].to_numpy(dtype=float) - 0.5 * peer_max
        # Ranking B: behavior-unaware (predicted prob only)
        merged["unaware_score"] = merged["prediction_prob"].to_numpy(dtype=float)

        mrr_aware = self._mean_reciprocal_rank(
            merged.sort_values("aware_score", ascending=False), merged["is_accepted"].to_numpy(dtype=bool),
        )
        mrr_unaware = self._mean_reciprocal_rank(
            merged.sort_values("unaware_score", ascending=False), merged["is_accepted"].to_numpy(dtype=bool),
        )

        result.candidate_test_ran = True
        result.candidate_mrr_aware = mrr_aware
        result.candidate_mrr_unaware = mrr_unaware
        winner = "behavior-aware" if mrr_aware > mrr_unaware else "behavior-unaware"
        result.candidate_detail = (
            f"Candidate ranking — MRR (behavior-aware): {mrr_aware:.3f}, "
            f"MRR (behavior-unaware): {mrr_unaware:.3f}. "
            f"Winner: {winner}."
        )
        _log(result.candidate_detail)

    @staticmethod
    def _mean_reciprocal_rank(sorted_df: pd.DataFrame, is_positive: np.ndarray) -> float:
        """Mean Reciprocal Rank of accepted (positive) segments in a ranked list."""
        reindexed = is_positive[sorted_df.index.to_numpy()] if hasattr(sorted_df, "index") else is_positive
        # After sort_values the df index may not be 0..N. Use positional.
        positives = sorted_df["is_accepted"].to_numpy(dtype=bool)
        if not positives.any():
            return 0.0
        reciprocal_ranks = []
        for rank_idx in range(len(positives)):
            if positives[rank_idx]:
                reciprocal_ranks.append(1.0 / (rank_idx + 1))
        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    def _compute_peer_max_prob(
        self,
        project_root: Path,
        current_model_version: str,
        segment_ids: pd.Series,
    ) -> np.ndarray:
        """Return per-segment max peer-behavior probability (0 if no peers)."""
        models_root = project_root / "derived" / "models"
        if not models_root.exists():
            return np.zeros(len(segment_ids), dtype=float)

        sid_set = set(segment_ids.astype(str))
        peer_probs: list[pd.Series] = []

        for model_dir in sorted(models_root.iterdir()):
            if not model_dir.is_dir():
                continue
            if not model_dir.name.startswith("behavior_model_"):
                continue
            # Skip all directories belonging to the *current* target behavior
            if model_dir.name == current_model_version or model_dir.name.startswith(current_model_version):
                continue
            pred_path = model_dir / "segment_predictions.parquet"
            if not pred_path.exists():
                continue
            try:
                peer_df = pd.read_parquet(pred_path, columns=["segment_id", "prediction_prob"])
                series = peer_df.set_index("segment_id")["prediction_prob"].rename(model_dir.name)
                peer_probs.append(series)
            except Exception:
                continue

        if not peer_probs:
            return np.zeros(len(segment_ids), dtype=float)

        peer_df = pd.concat(peer_probs, axis=1)
        # Align to the requested segment IDs
        aligned = peer_df.reindex(segment_ids.astype(str).values)
        vals = aligned.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            result = np.nanmax(vals, axis=1)
        return np.where(np.isnan(result), 0.0, result)

    # ------------------------------------------------------------------
    # Test 2 — Temporal refinement (mutual inhibition)
    # ------------------------------------------------------------------

    def _test_temporal_inhibition(
        self,
        project_root: Path,
        target_behavior: str,
        result: AblationResult,
        _log: Callable[[str], None],
    ) -> None:
        from sklearn.metrics import f1_score

        # Load validation predictions (frame-level or segment-level)
        resolved = self._resolve_model_dir(project_root, target_behavior)
        if resolved is None:
            result.warnings.append("No validation predictions — skipping temporal test.")
            return
        model_version, model_dir = resolved
        val_path = model_dir / "validation_predictions.parquet"
        if not val_path.exists():
            result.warnings.append("No validation predictions — skipping temporal test.")
            return

        val_df = pd.read_parquet(val_path)
        if val_df.empty or "label_true" not in val_df.columns:
            result.warnings.append("Validation predictions empty or missing labels.")
            return

        y_true = val_df["label_true"].to_numpy(dtype=int)
        y_pred_raw = val_df["label_pred"].to_numpy(dtype=int)

        # F1 without inhibition = raw model predictions
        f1_unaware = float(f1_score(y_true, y_pred_raw, average="macro", zero_division=0))

        # Simulate inhibition effect: load peer model predictions on validation segments
        # and adjust the score used for thresholding
        pred_probs_path = model_dir / "segment_predictions.parquet"
        if pred_probs_path.exists() and "segment_id" in val_df.columns:
            all_preds = pd.read_parquet(pred_probs_path)
            val_with_prob = val_df.merge(
                all_preds[["segment_id", "prediction_prob"]], on="segment_id", how="left",
            )
            if "prediction_prob" in val_with_prob.columns:
                raw_prob = val_with_prob["prediction_prob"].fillna(0.5).to_numpy(dtype=float)
            else:
                raw_prob = np.where(y_pred_raw == 1, 0.7, 0.3)
        else:
            raw_prob = np.where(y_pred_raw == 1, 0.7, 0.3)

        peer_max = self._compute_peer_max_prob(
            project_root, model_version,
            val_df["segment_id"] if "segment_id" in val_df.columns else pd.Series(range(len(val_df))),
        )

        # Apply inhibition: subtract scaled peer competition
        inhibition_weight = 0.20
        inhibited_prob = np.clip(raw_prob - inhibition_weight * peer_max, 0.0, 1.0)
        y_pred_inhibited = (inhibited_prob >= 0.5).astype(int)

        # Map predictions back to match y_true label space
        # y_true uses encoded labels (0/1), y_pred_inhibited is 0/1 for the positive class
        # We need to ensure we're comparing like-for-like
        # The positive class is typically index 1
        unique_true = set(y_true)
        if unique_true == {0, 1}:
            f1_aware = float(f1_score(y_true, y_pred_inhibited, average="macro", zero_division=0))
        else:
            # Multi-class — inhibition test doesn't apply cleanly
            result.warnings.append(
                "Multi-class validation set detected; temporal inhibition test uses binary simplification."
            )
            f1_aware = float(f1_score(
                (y_true > 0).astype(int), y_pred_inhibited, average="macro", zero_division=0,
            ))
            f1_unaware = float(f1_score(
                (y_true > 0).astype(int), y_pred_raw.clip(0, 1), average="macro", zero_division=0,
            ))

        result.temporal_test_ran = True
        result.temporal_f1_aware = f1_aware
        result.temporal_f1_unaware = f1_unaware
        winner = "behavior-aware (inhibition)" if f1_aware > f1_unaware else "behavior-unaware (no inhibition)"
        result.temporal_detail = (
            f"Temporal refinement — F1 with inhibition: {f1_aware:.3f}, "
            f"F1 without inhibition: {f1_unaware:.3f}. "
            f"Winner: {winner}."
        )
        _log(result.temporal_detail)

    # ------------------------------------------------------------------
    # Test 3 — Model feature ablation (peer features in training)
    # ------------------------------------------------------------------

    def _test_model_feature_ablation(
        self,
        project_root: Path,
        target_behavior: str,
        classifier_family: str,
        n_folds: int,
        result: AblationResult,
        _log: Callable[[str], None],
    ) -> None:
        from sklearn.metrics import average_precision_score, f1_score
        from sklearn.model_selection import StratifiedKFold

        from abel.services.active_learning_trainer_service import (
            ActiveLearningTrainerService,
        )

        train_path = project_root / "derived" / "training_sets" / "training_set.parquet"
        if not train_path.exists():
            result.warnings.append("No training set — skipping model feature ablation.")
            return

        df = pd.read_parquet(train_path)
        if df.empty or "label" not in df.columns:
            result.warnings.append("Training set empty or missing labels.")
            return

        # Remove non-trainable labels
        _untrainable = {"ambiguous", "boundary_error"}
        df = df[~df["label"].astype(str).isin(_untrainable)].copy()

        # Collapse alternates to no_behavior
        _nb_tokens = {"no_behavior", "no_behaviour", "nobehavior", "nobehaviour"}
        if target_behavior:
            for i, lbl in enumerate(df["label"].astype(str)):
                lbl_clean = lbl.strip()
                if lbl_clean == target_behavior:
                    continue
                if lbl_clean.lower().replace("_", "").replace(" ", "") in _nb_tokens:
                    continue
                df.iat[i, df.columns.get_loc("label")] = "no_behavior"

        # Encode labels
        y, inv = ActiveLearningTrainerService._label_map(df["label"])
        label_to_idx = {v: k for k, v in inv.items()}
        y = np.asarray([label_to_idx[str(lbl)] for lbl in df["label"]], dtype=int)

        if len(set(y)) < 2:
            result.warnings.append("Only one class in training set — cannot cross-validate.")
            return

        # Base feature columns
        base_feature_cols = ActiveLearningTrainerService._numeric_feature_cols(df)
        if not base_feature_cols:
            result.warnings.append("No numeric feature columns in training set.")
            return

        # Peer-behavior feature columns: for each peer model, get its prediction
        # score for each training segment and add as an extra feature
        peer_features = self._build_peer_feature_columns(project_root, target_behavior, df)
        augmented_feature_cols = base_feature_cols + list(peer_features.columns)

        if peer_features.empty:
            result.warnings.append(
                "No peer behavior models with predictions found — "
                "model ablation test cannot compare (requires 2+ trained behaviors)."
            )
            return

        # Merge peer features into df
        df_augmented = pd.concat([df, peer_features], axis=1)

        _log(
            f"  Base features: {len(base_feature_cols)}, "
            f"peer features: {len(peer_features.columns)}, "
            f"total augmented: {len(augmented_feature_cols)}"
        )

        # K-fold cross-validation
        n_folds = min(n_folds, len(set(y)))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        f1s_base: list[float] = []
        f1s_augmented: list[float] = []
        praucs_base: list[float] = []
        praucs_augmented: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y)):
            _log(f"  Fold {fold_idx + 1}/{n_folds}…")

            y_train, y_val = y[train_idx], y[val_idx]

            # --- Base model ---
            x_train_base = df.iloc[train_idx][base_feature_cols].to_numpy(dtype=float)
            x_val_base = df.iloc[val_idx][base_feature_cols].to_numpy(dtype=float)
            est_base = ActiveLearningTrainerService._make_estimator(
                classifier_family, {}, 42,
            )
            est_base.fit(x_train_base, y_train)
            probs_base = est_base.predict_proba(x_val_base)
            preds_base = np.argmax(probs_base, axis=1)

            f1_base = float(f1_score(y_val, preds_base, average="macro", zero_division=0))
            f1s_base.append(f1_base)

            # PR-AUC for the positive class
            target_idx = next(
                (idx for idx, name in inv.items() if str(name) == target_behavior), None,
            )
            if target_idx is None and probs_base.shape[1] == 2:
                target_idx = 1
            if target_idx is not None:
                prauc_base = float(average_precision_score(
                    (y_val == int(target_idx)).astype(int), probs_base[:, int(target_idx)],
                ))
            else:
                prauc_base = float("nan")
            praucs_base.append(prauc_base)

            # --- Augmented model (with peer-behavior features) ---
            x_train_aug = df_augmented.iloc[train_idx][augmented_feature_cols].to_numpy(dtype=float)
            x_val_aug = df_augmented.iloc[val_idx][augmented_feature_cols].to_numpy(dtype=float)
            est_aug = ActiveLearningTrainerService._make_estimator(
                classifier_family, {}, 42,
            )
            est_aug.fit(x_train_aug, y_train)
            probs_aug = est_aug.predict_proba(x_val_aug)
            preds_aug = np.argmax(probs_aug, axis=1)

            f1_aug = float(f1_score(y_val, preds_aug, average="macro", zero_division=0))
            f1s_augmented.append(f1_aug)

            if target_idx is not None:
                prauc_aug = float(average_precision_score(
                    (y_val == int(target_idx)).astype(int), probs_aug[:, int(target_idx)],
                ))
            else:
                prauc_aug = float("nan")
            praucs_augmented.append(prauc_aug)

            _log(
                f"    Fold {fold_idx + 1}: base F1={f1_base:.3f}, "
                f"augmented F1={f1_aug:.3f}"
            )

        mean_f1_base = float(np.mean(f1s_base))
        mean_f1_aug = float(np.mean(f1s_augmented))
        mean_prauc_base = float(np.nanmean(praucs_base))
        mean_prauc_aug = float(np.nanmean(praucs_augmented))

        result.model_test_ran = True
        result.model_f1_unaware = mean_f1_base
        result.model_f1_aware = mean_f1_aug
        result.model_prauc_unaware = mean_prauc_base
        result.model_prauc_aware = mean_prauc_aug

        # Statistical significance via Wilcoxon signed-rank test (if enough folds)
        sig_note = ""
        if n_folds >= 3:
            try:
                from scipy.stats import wilcoxon

                f1_diffs = np.array(f1s_augmented) - np.array(f1s_base)
                if not np.all(f1_diffs == 0):
                    _, p_value = wilcoxon(f1_diffs)
                    sig_note = f" (Wilcoxon p={p_value:.3f})"
                else:
                    sig_note = " (identical across folds)"
            except Exception:
                sig_note = ""

        f1_delta = mean_f1_aug - mean_f1_base
        winner = "behavior-aware (peer features)" if f1_delta > 0 else "behavior-unaware (base features)"
        result.model_detail = (
            f"Model ablation ({n_folds}-fold CV) — "
            f"base F1: {mean_f1_base:.3f}, augmented F1: {mean_f1_aug:.3f} "
            f"(Δ={f1_delta:+.3f}{sig_note}); "
            f"base PR-AUC: {mean_prauc_base:.3f}, augmented PR-AUC: {mean_prauc_aug:.3f}. "
            f"Winner: {winner}."
        )
        _log(result.model_detail)

    def _build_peer_feature_columns(
        self,
        project_root: Path,
        target_behavior: str,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create a DataFrame of peer-behavior prediction scores aligned to *df*."""
        models_root = project_root / "derived" / "models"
        if not models_root.exists() or "segment_id" not in df.columns:
            return pd.DataFrame(index=df.index)

        peer_cols: dict[str, np.ndarray] = {}

        for model_dir in sorted(models_root.iterdir()):
            if not model_dir.is_dir():
                continue
            if not model_dir.name.startswith("behavior_model_"):
                continue
            # Skip all model directories belonging to the target behavior
            if self._is_target_model_dir(model_dir.name, target_behavior):
                continue
            if model_dir.name == "behavior_model_v1":
                continue
            pred_path = model_dir / "segment_predictions.parquet"
            if not pred_path.exists():
                continue
            try:
                peer_df = pd.read_parquet(pred_path, columns=["segment_id", "prediction_prob"])
                lookup = peer_df.set_index("segment_id")["prediction_prob"].to_dict()
                col_name = f"peer_prob_{model_dir.name}"
                peer_cols[col_name] = np.array(
                    [float(lookup.get(str(sid), 0.0)) for sid in df["segment_id"].astype(str)],
                    dtype=float,
                )
            except Exception:
                continue

        if not peer_cols:
            return pd.DataFrame(index=df.index)

        peer_df = pd.DataFrame(peer_cols, index=df.index)

        # Add aggregated columns: max peer prob, mean peer prob
        vals = peer_df.to_numpy(dtype=float)
        peer_df["peer_prob_max"] = np.nanmax(vals, axis=1)
        peer_df["peer_prob_mean"] = np.nanmean(vals, axis=1)
        peer_df["exclusivity_margin"] = np.nan_to_num(
            np.nanmax(vals, axis=1), nan=0.0
        )

        return peer_df

    # ------------------------------------------------------------------
    # Verdict synthesis
    # ------------------------------------------------------------------

    @staticmethod
    def _synthesize_verdict(result: AblationResult) -> tuple[str, str]:
        """Combine the three test outcomes into an overall verdict and summary."""
        scores_aware = 0
        scores_unaware = 0
        tests_run = 0

        lines: list[str] = ["═══ Behavior-Awareness Ablation Report ═══", ""]

        if result.candidate_test_ran:
            tests_run += 1
            if result.candidate_mrr_aware > result.candidate_mrr_unaware:
                scores_aware += 1
                lines.append("✓ Candidate ranking: behavior-aware is better")
            elif result.candidate_mrr_aware < result.candidate_mrr_unaware:
                scores_unaware += 1
                lines.append("✗ Candidate ranking: behavior-unaware is better")
            else:
                lines.append("— Candidate ranking: tied")
            lines.append(f"  {result.candidate_detail}")
            lines.append("")

        if result.temporal_test_ran:
            tests_run += 1
            if result.temporal_f1_aware > result.temporal_f1_unaware:
                scores_aware += 1
                lines.append("✓ Temporal refinement: mutual inhibition helps")
            elif result.temporal_f1_aware < result.temporal_f1_unaware:
                scores_unaware += 1
                lines.append("✗ Temporal refinement: mutual inhibition hurts")
            else:
                lines.append("— Temporal refinement: tied")
            lines.append(f"  {result.temporal_detail}")
            lines.append("")

        if result.model_test_ran:
            tests_run += 1
            if result.model_f1_aware > result.model_f1_unaware:
                scores_aware += 1
                lines.append("✓ Model features: peer-behavior features improve model")
            elif result.model_f1_aware < result.model_f1_unaware:
                scores_unaware += 1
                lines.append("✗ Model features: peer-behavior features hurt model")
            else:
                lines.append("— Model features: tied")
            lines.append(f"  {result.model_detail}")
            lines.append("")

        if result.warnings:
            lines.append("Warnings:")
            for w in result.warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")

        if tests_run == 0:
            verdict = "inconclusive"
            lines.append(
                "No tests could be run. Ensure 2+ behaviors are trained "
                "and review labels exist."
            )
        elif scores_aware > scores_unaware:
            verdict = "aware_better"
            lines.append(
                f"VERDICT: Behavior-aware mode wins {scores_aware}/{tests_run} tests. "
                f"Keep all_behavior_aware enabled."
            )
        elif scores_unaware > scores_aware:
            verdict = "unaware_better"
            lines.append(
                f"VERDICT: Behavior-unaware mode wins {scores_unaware}/{tests_run} tests. "
                f"Consider disabling all_behavior_aware."
            )
        else:
            verdict = "inconclusive"
            lines.append(
                f"VERDICT: Inconclusive ({scores_aware} aware vs {scores_unaware} unaware "
                f"across {tests_run} tests). More labeled data may clarify."
            )

        return verdict, "\n".join(lines)
