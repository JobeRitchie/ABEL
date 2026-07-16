"""Phase 1 behavior-adaptive benchmarking and diagnostics orchestration.

Implements:
- behavior-adaptive expert benchmarking by feature family
- confound-aware analysis when non-target labels are sufficient
- baseline vs advanced comparator summaries
- publication-quality diagnostic plots (PNG + SVG)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from abel.services.behavior_adaptive_feature_cache_service import (
    BehaviorAdaptiveFeatureCacheService,
    MultiScaleFeatureCacheConfig,
)
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml
from abel.utils import xgb_predict


@dataclass
class BehaviorAdaptiveBenchmarkConfig:
    enabled: bool = False
    enable_modality_benchmarking: bool = True
    enable_confound_analysis: bool = True
    regenerate_diagnostics: bool = False
    export_high_resolution: bool = True
    save_artifacts: bool = True
    diagnostics_enabled: bool = True
    cache_features: bool = True
    primary_metric: str = "ap"
    min_examples_per_class: int = 12
    min_examples_for_learned_weights: int = 75
    use_gpu_if_available: bool = True
    quick_feature_test: bool = False
    subset_max_sessions: int = 6
    subset_max_segments_per_scale: int = 25000
    cpu_parallel_workers: int = 0
    cpu_use_process_pool: bool = True


class BehaviorAdaptiveBenchmarkService:
    """Run Phase 1 benchmarking and diagnostics in an opt-in, baseline-safe manner."""

    # Single segment window (seconds) used for all Phase 1 benchmarking.
    SEGMENT_SCALE_SEC = 0.5

    def __init__(self) -> None:
        self._feature_cache = BehaviorAdaptiveFeatureCacheService()

    @staticmethod
    def _settings_path(project_root: Path) -> Path:
        return project_root / "config" / "behavior_adaptive_settings.yaml"

    @staticmethod
    def _analysis_root(project_root: Path) -> Path:
        return project_root / "derived" / "analysis"

    def _benchmark_root(self, project_root: Path, behavior_id: str) -> Path:
        return self._analysis_root(project_root) / "benchmarks" / behavior_id

    def _diagnostic_root(self, project_root: Path, behavior_id: str) -> Path:
        return self._analysis_root(project_root) / "diagnostics" / behavior_id

    @staticmethod
    def _sanitize_behavior_id(behavior_id: str) -> str:
        clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(behavior_id).strip())
        return clean or "target_behavior"

    @staticmethod
    def load_or_init_settings(project_root: Path) -> dict[str, Any]:
        path = BehaviorAdaptiveBenchmarkService._settings_path(project_root)
        default: dict[str, Any] = {
            "phase1": {
                "enabled": False,
                "enable_modality_benchmarking": True,
                "enable_confound_analysis": True,
                "diagnostics_enabled": True,
                "cache_features": True,
                "regenerate_diagnostics": False,
                "export_high_resolution": True,
                "save_artifacts": True,
                "primary_metric": "ap",
                "min_examples_per_class": 12,
                "min_examples_for_learned_weights": 75,
                "use_gpu_if_available": True,
                "quick_feature_test": False,
                "subset_max_sessions": 6,
                "subset_max_segments_per_scale": 25000,
                "cpu_parallel_workers": 0,
                "cpu_use_process_pool": True,
            },
            "behavior_overrides": {},
        }
        if path.exists():
            raw = read_yaml(path, default)
            if not raw:
                raw = default
        else:
            raw = default
            path.parent.mkdir(parents=True, exist_ok=True)
            write_yaml(path, raw)
        return raw

    @staticmethod
    def save_settings(project_root: Path, settings: dict[str, Any]) -> None:
        path = BehaviorAdaptiveBenchmarkService._settings_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(path, settings)

    @staticmethod
    def _resolve_config(settings: dict[str, Any], behavior_id: str) -> BehaviorAdaptiveBenchmarkConfig:
        phase1 = dict(settings.get("phase1") or {})
        behavior_overrides = dict((settings.get("behavior_overrides") or {}).get(behavior_id) or {})

        merged = dict(phase1)
        merged.update(behavior_overrides)

        return BehaviorAdaptiveBenchmarkConfig(
            enabled=bool(merged.get("enabled", False)),
            enable_modality_benchmarking=bool(merged.get("enable_modality_benchmarking", True)),
            enable_confound_analysis=bool(merged.get("enable_confound_analysis", True)),
            regenerate_diagnostics=bool(merged.get("regenerate_diagnostics", False)),
            export_high_resolution=bool(merged.get("export_high_resolution", True)),
            save_artifacts=bool(merged.get("save_artifacts", True)),
            diagnostics_enabled=bool(merged.get("diagnostics_enabled", True)),
            cache_features=bool(merged.get("cache_features", True)),
            primary_metric=str(merged.get("primary_metric", "ap")),
            min_examples_per_class=int(merged.get("min_examples_per_class", 12)),
            min_examples_for_learned_weights=int(merged.get("min_examples_for_learned_weights", 75)),
            use_gpu_if_available=bool(merged.get("use_gpu_if_available", True)),
            quick_feature_test=bool(merged.get("quick_feature_test", False)),
            subset_max_sessions=int(merged.get("subset_max_sessions", 6)),
            subset_max_segments_per_scale=int(merged.get("subset_max_segments_per_scale", 25000)),
            cpu_parallel_workers=int(merged.get("cpu_parallel_workers", 0)),
            cpu_use_process_pool=bool(merged.get("cpu_use_process_pool", True)),
        )

    @staticmethod
    def _subset_sessions_from_reviews(review_intervals: pd.DataFrame, max_sessions: int) -> list[str]:
        if review_intervals.empty or max_sessions <= 0:
            return []
        counts = review_intervals["session_id"].astype(str).value_counts()
        return [str(s) for s in counts.index[: max(1, int(max_sessions))]]

    @staticmethod
    def _subset_sessions_from_pose_cache(project_root: Path, max_sessions: int) -> list[str]:
        if max_sessions <= 0:
            return []
        pose_path = project_root / "derived" / "pose_features" / "frame_pose.parquet"
        if not pose_path.exists():
            return []
        try:
            pose_df = pd.read_parquet(pose_path, columns=["session_id"])
            uniq = sorted(set(pose_df["session_id"].astype(str)))
            return uniq[: max(1, int(max_sessions))]
        except Exception:
            return []

    @staticmethod
    def _probe_xgboost_gpu() -> bool:
        try:
            from xgboost import XGBClassifier

            x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
            y = np.asarray([0, 1, 0, 1], dtype=int)
            model = XGBClassifier(
                n_estimators=1,
                max_depth=1,
                tree_method="hist",
                device="cuda",
                eval_metric="logloss",
                random_state=42,
            )
            model.fit(x, y)
            return True
        except Exception:
            return False

    @staticmethod
    def _make_phase1_estimator(
        *,
        prefer_gpu: bool,
        multiclass: bool,
        n_classes: int | None = None,
    ):
        # Prefer XGBoost on GPU when available; fall back to CPU-safe sklearn.
        try:
            from xgboost import XGBClassifier

            params: dict[str, Any] = {
                "n_estimators": 220,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "tree_method": "hist",
                "random_state": 42,
            }
            if multiclass:
                params["objective"] = "multi:softprob"
                params["eval_metric"] = "mlogloss"
                if n_classes is not None and int(n_classes) > 2:
                    params["num_class"] = int(n_classes)
            else:
                params["objective"] = "binary:logistic"
                params["eval_metric"] = "logloss"

            if prefer_gpu:
                params["device"] = "cuda"
                return XGBClassifier(**params), "gpu_xgboost", ""

            params["device"] = "cpu"
            return XGBClassifier(**params), "cpu_xgboost", ""
        except Exception as exc:
            from sklearn.ensemble import HistGradientBoostingClassifier

            note = str(exc).splitlines()[0] if str(exc) else "xgboost unavailable"
            return HistGradientBoostingClassifier(random_state=42), "cpu_hist_gbdt", note

    @staticmethod
    def _parse_segment_id_interval(segment_id: str) -> tuple[str, int, int] | None:
        text = str(segment_id or "").strip()
        if not text:
            return None
        parts = text.split("_")
        if len(parts) < 4:
            return None
        try:
            end = int(parts[-1])
            start = int(parts[-2])
        except ValueError:
            return None

        sid_idx = -1
        for i, token in enumerate(parts):
            if token == "session" and i + 1 < len(parts):
                sid_idx = i
                break
        if sid_idx < 0:
            return None
        sid = "_".join(parts[sid_idx : sid_idx + 2])
        if not sid.startswith("session_"):
            return None
        return sid, int(start), int(end)

    @staticmethod
    def _build_review_interval_table(project_root: Path) -> pd.DataFrame:
        path = project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["session_id", "start_frame", "end_frame", "review_label"])

        labels = pd.read_parquet(path)
        if labels.empty or "segment_id" not in labels.columns or "review_label" not in labels.columns:
            return pd.DataFrame(columns=["session_id", "start_frame", "end_frame", "review_label"])

        rows: list[dict[str, Any]] = []
        for rec in labels.to_dict(orient="records"):
            parsed = BehaviorAdaptiveBenchmarkService._parse_segment_id_interval(str(rec.get("segment_id", "")))
            if parsed is None:
                continue
            sid, start, end = parsed
            rows.append(
                {
                    "session_id": sid,
                    "start_frame": int(min(start, end)),
                    "end_frame": int(max(start, end)),
                    "review_label": str(rec.get("review_label", "")).strip(),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _derive_confound_map(
        review_intervals: pd.DataFrame,
        target_behavior: str,
        settings: dict[str, Any],
    ) -> dict[str, str]:
        behavior_overrides = settings.setdefault("behavior_overrides", {})
        behavior_cfg = behavior_overrides.setdefault(target_behavior, {})
        existing = dict(behavior_cfg.get("confound_label_map") or {})

        labels = sorted(set(review_intervals.get("review_label", pd.Series(dtype=str)).astype(str)))
        for label in labels:
            token = str(label).strip()
            if not token or token in {"ambiguous", "boundary_error"}:
                continue
            if token == target_behavior:
                continue
            if token.startswith("not_") or token == "no_behavior":
                continue
            existing.setdefault(token, token)

        behavior_cfg["confound_label_map"] = existing
        return existing

    @staticmethod
    def _majority_label(labels: list[str]) -> str:
        if not labels:
            return "ambiguous"
        values = pd.Series(labels).value_counts()
        top = str(values.index[0])
        return top

    @staticmethod
    def _assign_labels_to_segments(
        segments: pd.DataFrame,
        review_intervals: pd.DataFrame,
        target_behavior: str,
        confound_label_map: dict[str, str],
    ) -> pd.DataFrame:
        if segments.empty:
            return pd.DataFrame()
        if review_intervals.empty:
            return pd.DataFrame()

        out_rows: list[dict[str, Any]] = []
        by_session = {
            sid: grp[["start_frame", "end_frame", "review_label"]].to_dict(orient="records")
            for sid, grp in review_intervals.groupby("session_id")
        }

        for row in segments.itertuples(index=False):
            sid = str(getattr(row, "session_id"))
            start = int(getattr(row, "start_frame"))
            end = int(getattr(row, "end_frame"))
            overlaps: list[str] = []
            for interval in by_session.get(sid, []):
                s0 = int(interval["start_frame"])
                e0 = int(interval["end_frame"])
                if start <= e0 and end >= s0:
                    overlaps.append(str(interval["review_label"]).strip())
            if not overlaps:
                continue

            resolved = BehaviorAdaptiveBenchmarkService._majority_label(overlaps)
            if not resolved or resolved in {"ambiguous", "boundary_error"}:
                continue

            if resolved == target_behavior:
                class_label = target_behavior
                binary_label = 1
            elif resolved.startswith("not_") or resolved == "no_behavior":
                class_label = "no_behavior"
                binary_label = 0
            else:
                class_label = confound_label_map.get(resolved, resolved)
                binary_label = 0

            rec = {k: getattr(row, k) for k in segments.columns if hasattr(row, k)}
            rec["class_label"] = class_label
            rec["binary_label"] = int(binary_label)
            out_rows.append(rec)

        return pd.DataFrame(out_rows)

    @staticmethod
    def _feature_families(df: pd.DataFrame) -> dict[str, list[str]]:
        excluded = {
            "segment_id",
            "start_frame",
            "end_frame",
            "animal_id",
            "session_id",
            "class_label",
            "binary_label",
        }
        numeric = [
            c
            for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]

        pose_keys = (
            "nose",
            "paw",
            "forepaw",
            "body_orientation",
            "head_pitch",
            "centroid",
            "velocity",
            "acceleration",
            "jerk",
            "autocorr",
            "periodicity",
            "oscillation",
            "scrape",
            "angle",
        )
        context_keys = (
            "target",
            "tmt",
            "dist",
            "zone",
            "roi",
            "occup",
            "bedding",
            "substrate",
            "context",
        )
        motion_keys = ("flow", "motion", "velocity", "acceleration", "jerk", "substrate")
        visual_keys = ("visual", "image", "clip", "embed")

        def pick(keys: tuple[str, ...]) -> list[str]:
            out: list[str] = []
            for c in numeric:
                low = c.lower()
                if any(k in low for k in keys):
                    out.append(c)
            return sorted(set(out))

        pose = pick(pose_keys)
        context = pick(context_keys)
        motion = pick(motion_keys)
        visual = pick(visual_keys)

        # Keep families disjoint where possible for clarity.
        visual_set = set(visual)
        context = [c for c in context if c not in visual_set]
        pose = [c for c in pose if c not in visual_set]

        families = {
            "pose": pose,
            "visual": visual,
            "motion": motion,
            "context": context,
            "fused": numeric,
        }
        return families

    @staticmethod
    def _group_split(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.model_selection import GroupShuffleSplit

        groups = df["session_id"].astype(str).to_numpy()
        n_rows = len(df)
        n_unique = int(pd.Series(groups).nunique())
        if n_rows <= 1:
            idx = np.arange(n_rows, dtype=int)
            return idx, idx
        if n_unique < 2:
            rng = np.random.RandomState(random_state)
            perm = rng.permutation(n_rows)
            n_test = max(1, int(round(test_size * n_rows)))
            n_test = min(n_rows - 1, n_test)
            return perm[n_test:].astype(int), perm[:n_test].astype(int)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, val_idx = next(splitter.split(df, groups=groups))
        return train_idx.astype(int), val_idx.astype(int)

    @staticmethod
    def _metrics_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
        from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score

        # Choose threshold by F1 on validation by default when not provided.
        thresholds = np.linspace(0.05, 0.95, 37)
        best_thr = 0.5
        best_f1 = -1.0
        for thr in thresholds:
            pred = (y_prob >= thr).astype(int)
            f1 = float(f1_score(y_true, pred, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        chosen_thr = float(best_thr if threshold is None else threshold)
        y_pred = (y_prob >= chosen_thr).astype(int)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        ap = float(average_precision_score(y_true, y_prob))

        pr_p, pr_r, pr_t = precision_recall_curve(y_true, y_prob)

        return {
            "ap": ap,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": chosen_thr,
            "pr_curve": {
                "precision": pr_p.tolist(),
                "recall": pr_r.tolist(),
                "thresholds": pr_t.tolist(),
            },
        }

    @staticmethod
    def _reliability_with_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(y_prob, bins) - 1
        prob_pred: list[float] = []
        prob_true: list[float] = []
        counts: list[int] = []
        ece = 0.0
        n = max(1, len(y_true))
        for b in range(n_bins):
            mask = idx == b
            if not np.any(mask):
                prob_pred.append(float((bins[b] + bins[b + 1]) * 0.5))
                prob_true.append(float("nan"))
                counts.append(0)
                continue
            mean_pred = float(np.mean(y_prob[mask]))
            mean_true = float(np.mean(y_true[mask]))
            cnt = int(np.sum(mask))
            prob_pred.append(mean_pred)
            prob_true.append(mean_true)
            counts.append(cnt)
            ece += abs(mean_pred - mean_true) * (cnt / n)
        return {
            "prob_pred": prob_pred,
            "prob_true": prob_true,
            "counts": counts,
            "ece": float(ece),
        }

    @staticmethod
    def _recall_at_budgets(y_true: np.ndarray, y_prob: np.ndarray, budgets: tuple[int, ...] = (50, 100, 200)) -> dict[str, float]:
        order = np.argsort(-y_prob)
        positives = max(1, int(np.sum(y_true == 1)))
        out: dict[str, float] = {}
        for k in budgets:
            k_eff = min(len(order), int(k))
            top = order[:k_eff]
            rec = float(np.sum(y_true[top] == 1) / positives)
            out[f"top_{k}"] = rec
        return out

    @staticmethod
    def _train_binary_expert(
        df: pd.DataFrame,
        feature_cols: list[str],
        *,
        prefer_gpu: bool,
    ) -> dict[str, Any]:

        if not feature_cols:
            return {"status": "skipped", "reason": "no feature columns"}
        if len(df) < 40:
            return {"status": "skipped", "reason": "insufficient rows"}

        y = df["binary_label"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            return {"status": "skipped", "reason": "single class"}

        train_idx, val_idx = BehaviorAdaptiveBenchmarkService._group_split(df)
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]
        if train.empty or val.empty:
            return {"status": "skipped", "reason": "empty split"}

        x_train = train[feature_cols].to_numpy(dtype=float)
        y_train = train["binary_label"].to_numpy(dtype=int)
        x_val = val[feature_cols].to_numpy(dtype=float)
        y_val = val["binary_label"].to_numpy(dtype=int)

        model, backend, note = BehaviorAdaptiveBenchmarkService._make_phase1_estimator(
            prefer_gpu=prefer_gpu,
            multiclass=False,
        )
        fallback_reason = ""
        try:
            model.fit(x_train, y_train)
        except Exception as exc:
            # GPU path can fail at runtime on unsupported systems/drivers; retry on CPU.
            fallback_reason = str(exc).splitlines()[0] if str(exc) else "model fit failed"
            model, backend, note = BehaviorAdaptiveBenchmarkService._make_phase1_estimator(
                prefer_gpu=False,
                multiclass=False,
            )
            model.fit(x_train, y_train)
        y_prob = xgb_predict.predict_proba(model, x_val)[:, 1]

        metrics = BehaviorAdaptiveBenchmarkService._metrics_binary(y_val, y_prob)
        metrics["status"] = "ok"
        metrics["n_train"] = int(len(train))
        metrics["n_val"] = int(len(val))
        metrics["class_balance"] = {
            "train_pos": int(np.sum(y_train == 1)),
            "train_neg": int(np.sum(y_train == 0)),
            "val_pos": int(np.sum(y_val == 1)),
            "val_neg": int(np.sum(y_val == 0)),
        }
        metrics["reliability"] = BehaviorAdaptiveBenchmarkService._reliability_with_ece(y_val, y_prob)
        metrics["recall_at_budget"] = BehaviorAdaptiveBenchmarkService._recall_at_budgets(y_val, y_prob)
        metrics["val_probability"] = y_prob.tolist()
        metrics["val_target"] = y_val.tolist()
        metrics["val_segment_ids"] = val["segment_id"].astype(str).tolist()
        metrics["model_backend"] = backend
        metrics["estimator_note"] = note
        if fallback_reason:
            metrics["fallback_reason"] = fallback_reason
        return metrics

    @staticmethod
    def _compute_confound_analysis(
        df: pd.DataFrame,
        target_behavior: str,
        *,
        prefer_gpu: bool,
    ) -> dict[str, Any]:
        from sklearn.metrics import confusion_matrix

        out: dict[str, Any] = {
            "enabled": False,
            "top_confounds": [],
            "confusion_matrix": [],
            "labels": [],
            "margin_histogram": {},
            "pairwise": {},
        }

        class_counts = df["class_label"].astype(str).value_counts().to_dict()
        non_target = {k: int(v) for k, v in class_counts.items() if k not in {target_behavior, "no_behavior"}}
        if not non_target:
            out["reason"] = "No confound labels available."
            return out

        top_confounds = sorted(non_target.items(), key=lambda kv: kv[1], reverse=True)
        out["top_confounds"] = [{"label": k, "count": int(v)} for k, v in top_confounds[:6]]

        # Multiclass confusion when enough classes have support.
        use_labels = [target_behavior] + [k for k, v in top_confounds if v >= 8][:5]
        work = df[df["class_label"].astype(str).isin(use_labels)].copy()
        if work.empty or work["class_label"].nunique() < 2:
            out["reason"] = "Insufficient multiclass support for confound analysis."
            return out

        feature_cols = [
            c
            for c in work.columns
            if c
            not in {
                "segment_id",
                "start_frame",
                "end_frame",
                "animal_id",
                "session_id",
                "class_label",
                "binary_label",
            }
            and pd.api.types.is_numeric_dtype(work[c])
        ]
        if not feature_cols:
            out["reason"] = "No numeric features for confound analysis."
            return out

        train_idx, val_idx = BehaviorAdaptiveBenchmarkService._group_split(work)
        train = work.iloc[train_idx]
        val = work.iloc[val_idx]
        if train.empty or val.empty:
            out["reason"] = "Empty validation split for confound analysis."
            return out

        labels = sorted(train["class_label"].astype(str).unique())
        val = val[val["class_label"].astype(str).isin(labels)].copy()
        if val.empty:
            out["reason"] = "Validation labels not represented in train split."
            return out

        label_to_idx = {k: i for i, k in enumerate(labels)}
        y_train = np.asarray([label_to_idx[v] for v in train["class_label"].astype(str)], dtype=int)
        y_val = np.asarray([label_to_idx[v] for v in val["class_label"].astype(str)], dtype=int)

        clf, backend, note = BehaviorAdaptiveBenchmarkService._make_phase1_estimator(
            prefer_gpu=prefer_gpu,
            multiclass=True,
            n_classes=len(labels),
        )
        fallback_reason = ""
        try:
            clf.fit(train[feature_cols].to_numpy(dtype=float), y_train)
        except Exception as exc:
            fallback_reason = str(exc).splitlines()[0] if str(exc) else "confound model fit failed"
            clf, backend, note = BehaviorAdaptiveBenchmarkService._make_phase1_estimator(
                prefer_gpu=False,
                multiclass=True,
                n_classes=len(labels),
            )
            clf.fit(train[feature_cols].to_numpy(dtype=float), y_train)
        prob = xgb_predict.predict_proba(clf, val[feature_cols].to_numpy(dtype=float))
        pred = np.argmax(prob, axis=1)

        cm = confusion_matrix(y_val, pred, labels=np.arange(len(labels)))
        cm_norm = cm.astype(float)
        row_sum = cm_norm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm_norm = cm_norm / row_sum

        target_idx = label_to_idx.get(target_behavior)
        if target_idx is None:
            out["reason"] = "Target behavior absent from confound validation split."
            return out

        confound_scores: list[tuple[str, float]] = []
        for name, idx in label_to_idx.items():
            if name == target_behavior:
                continue
            fn_to_name = float(cm_norm[target_idx, idx])
            fp_from_name = float(cm_norm[idx, target_idx]) if idx < cm_norm.shape[0] else 0.0
            confound_scores.append((name, fn_to_name + fp_from_name))
        confound_scores.sort(key=lambda kv: kv[1], reverse=True)

        # Margin vs top confound for TP / FP / FN groups.
        top_label = confound_scores[0][0] if confound_scores else None
        margin_payload: dict[str, list[float]] = {"tp": [], "fp": [], "fn": []}
        if top_label and top_label in label_to_idx:
            top_idx = label_to_idx[top_label]
            target_prob = prob[:, target_idx]
            top_prob = prob[:, top_idx]
            margin = target_prob - top_prob
            y_true_bin = (y_val == target_idx).astype(int)
            y_pred_bin = (pred == target_idx).astype(int)
            tp_mask = (y_true_bin == 1) & (y_pred_bin == 1)
            fp_mask = (y_true_bin == 0) & (y_pred_bin == 1)
            fn_mask = (y_true_bin == 1) & (y_pred_bin == 0)
            margin_payload = {
                "tp": margin[tp_mask].tolist(),
                "fp": margin[fp_mask].tolist(),
                "fn": margin[fn_mask].tolist(),
            }

        out.update(
            {
                "enabled": True,
                "labels": labels,
                "confusion_matrix": cm.tolist(),
                "confusion_matrix_normalized": cm_norm.tolist(),
                "top_confounds": [{"label": k, "score": float(v)} for k, v in confound_scores[:6]],
                "margin_histogram": margin_payload,
                "pairwise": {
                    k: {
                        "support": int(class_counts.get(k, 0)),
                        "combined_confusion_score": float(v),
                    }
                    for k, v in confound_scores[:6]
                },
                "model_backend": backend,
                "estimator_note": note,
            }
        )
        if fallback_reason:
            out["fallback_reason"] = fallback_reason
        return out

    @staticmethod
    def _composite_weight_score(ap: float, ece: float, margin_stat: float) -> float:
        ap_term = float(np.clip(ap, 0.0, 1.0))
        cal_term = float(np.clip(1.0 - ece, 0.0, 1.0))
        margin_term = float(np.clip((margin_stat + 1.0) * 0.5, 0.0, 1.0))
        return 0.75 * ap_term + 0.15 * cal_term + 0.10 * margin_term

    @staticmethod
    def _estimate_margin_stat(confound: dict[str, Any]) -> float:
        margin = dict(confound.get("margin_histogram") or {})
        tp = np.asarray(margin.get("tp", []), dtype=float)
        fp = np.asarray(margin.get("fp", []), dtype=float)
        if tp.size == 0 and fp.size == 0:
            return 0.0
        return float(np.nanmean(tp) - np.nanmean(fp) if fp.size > 0 and tp.size > 0 else np.nanmean(tp))

    @staticmethod
    def _save_figure(fig, png_path: Path, svg_path: Path, dpi: int = 300) -> None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(svg_path, dpi=dpi, bbox_inches="tight")

    def _plot_diagnostics(
        self,
        behavior_id: str,
        run_diag_dir: Path,
        expert_results: dict[str, Any],
        scale_results: dict[str, Any],
        confound_results: dict[str, Any],
        baseline_metrics: dict[str, Any],
        export_high_res: bool,
    ) -> dict[str, str]:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return {}

        dpi = 300 if export_high_res else 180
        outputs: dict[str, str] = {}

        # 1) Feature-family comparison bar plot (AP, F1)
        valid_experts = [
            (name, row)
            for name, row in expert_results.items()
            if isinstance(row, dict) and row.get("status") == "ok"
        ]
        if valid_experts:
            names = [n for n, _ in valid_experts]
            ap_vals = [float(r.get("ap", float("nan"))) for _, r in valid_experts]
            f1_vals = [float(r.get("f1", float("nan"))) for _, r in valid_experts]

            x = np.arange(len(names))
            w = 0.35
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.bar(x - w / 2, ap_vals, width=w, label="AP")
            ax.bar(x + w / 2, f1_vals, width=w, label="F1")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Score")
            ax.set_title(f"Feature Family Comparison - {behavior_id}")
            ax.legend(frameon=False)
            fig.tight_layout()
            png = run_diag_dir / "feature_family_comparison.png"
            svg = run_diag_dir / "feature_family_comparison.svg"
            self._save_figure(fig, png, svg, dpi=dpi)
            plt.close(fig)
            outputs["feature_family_comparison"] = str(png)

        # 2) Confusion matrix
        cm = np.asarray(confound_results.get("confusion_matrix", []), dtype=float)
        labels = list(confound_results.get("labels", []))
        if cm.ndim == 2 and cm.size > 0:
            fig, ax = plt.subplots(figsize=(5.8, 5.4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(f"Confusion Matrix - {behavior_id}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            if labels and len(labels) == cm.shape[0]:
                ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
                ax.set_yticks(np.arange(len(labels)), labels)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            png = run_diag_dir / "confusion_matrix_phase1.png"
            svg = run_diag_dir / "confusion_matrix_phase1.svg"
            self._save_figure(fig, png, svg, dpi=dpi)
            plt.close(fig)
            outputs["confusion_matrix_phase1"] = str(png)

        # 3) Target-vs-top-confound margin histogram
        margin = dict(confound_results.get("margin_histogram") or {})
        has_margin = any(len(margin.get(k, [])) > 0 for k in ("tp", "fp", "fn"))
        if has_margin:
            fig, ax = plt.subplots(figsize=(8.2, 4.6))
            bins = np.linspace(-1.0, 1.0, 28)
            for key, color in (("tp", "#2ca02c"), ("fp", "#d62728"), ("fn", "#ff7f0e")):
                vals = np.asarray(margin.get(key, []), dtype=float)
                if vals.size > 0:
                    ax.hist(vals, bins=bins, alpha=0.4, label=key.upper(), color=color)
            ax.set_xlabel("Target score - top confound score")
            ax.set_ylabel("Count")
            ax.set_title(f"Target-vs-Confound Margin - {behavior_id}")
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            png = run_diag_dir / "target_confound_margin_histogram.png"
            svg = run_diag_dir / "target_confound_margin_histogram.svg"
            self._save_figure(fig, png, svg, dpi=dpi)
            plt.close(fig)
            outputs["target_confound_margin_histogram"] = str(png)

        # 4) Baseline vs best expert PR curve
        best_name = None
        best_ap = -1.0
        for name, row in expert_results.items():
            if isinstance(row, dict) and row.get("status") == "ok":
                ap = float(row.get("ap", -1.0))
                if ap > best_ap:
                    best_ap = ap
                    best_name = name
        if best_name:
            best = dict(expert_results.get(best_name) or {})
            pr = dict(best.get("pr_curve") or {})
            r = np.asarray(pr.get("recall", []), dtype=float)
            p = np.asarray(pr.get("precision", []), dtype=float)
            if r.size > 0 and p.size > 0:
                fig, ax = plt.subplots(figsize=(7.2, 5.0))
                ax.plot(r, p, label=f"Behavior-adaptive ({best_name}) AP={best_ap:.3f}", linewidth=2.0)
                base_ap = float((baseline_metrics.get("segment_level") or {}).get("pr_auc", float("nan")))
                if np.isfinite(base_ap):
                    ax.axhline(base_ap, linestyle="--", color="#6b7280", label=f"Baseline AP={base_ap:.3f}")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.02)
                ax.set_title(f"PR Comparison - {behavior_id}")
                ax.legend(frameon=False)
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                png = run_diag_dir / "pr_curve_comparison_phase1.png"
                svg = run_diag_dir / "pr_curve_comparison_phase1.svg"
                self._save_figure(fig, png, svg, dpi=dpi)
                plt.close(fig)
                outputs["pr_curve_comparison_phase1"] = str(png)

        # 5) Reliability plot for best expert
        if best_name:
            best = dict(expert_results.get(best_name) or {})
            rel = dict(best.get("reliability") or {})
            pred = np.asarray(rel.get("prob_pred", []), dtype=float)
            obs = np.asarray(rel.get("prob_true", []), dtype=float)
            mask = np.isfinite(pred) & np.isfinite(obs)
            if np.any(mask):
                fig, ax = plt.subplots(figsize=(5.8, 5.2))
                ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Perfect")
                ax.plot(pred[mask], obs[mask], marker="o", linewidth=2.0, label=f"{best_name}")
                ece = float(rel.get("ece", float("nan")))
                ax.set_xlabel("Predicted probability")
                ax.set_ylabel("Observed frequency")
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.set_title(f"Calibration (ECE={ece:.3f}) - {behavior_id}")
                ax.legend(frameon=False)
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                png = run_diag_dir / "calibration_reliability_phase1.png"
                svg = run_diag_dir / "calibration_reliability_phase1.svg"
                self._save_figure(fig, png, svg, dpi=dpi)
                plt.close(fig)
                outputs["calibration_reliability_phase1"] = str(png)

        return outputs

    @staticmethod
    def _read_baseline_metrics(project_root: Path) -> dict[str, Any]:
        path = project_root / "derived" / "evaluation" / "model_metrics.json"
        if path.exists():
            return read_json(path, {})
        return {}

    @staticmethod
    def _phase_card(summary: dict[str, Any]) -> list[str]:
        cards: list[str] = []
        cmp_row = dict(summary.get("baseline_vs_advanced") or {})
        baseline_ap = float(cmp_row.get("baseline_ap", float("nan")))
        best_ap = float(cmp_row.get("best_adaptive_ap", float("nan")))
        best_name = str(cmp_row.get("best_expert", ""))
        if np.isfinite(baseline_ap) and np.isfinite(best_ap):
            cards.append(f"Behavior-adaptive AP changed from {baseline_ap:.3f} to {best_ap:.3f} (best={best_name}).")
        confound = dict(summary.get("confound") or {})
        top = list(confound.get("top_confounds") or [])
        if top:
            main = top[0]
            label = str(main.get("label", "other_behavior"))
            cards.append(f"Top confound behavior was '{label}' for this target.")
        rec = dict(summary.get("recommendations") or {})
        best_family = rec.get("best_family")
        if best_family:
            cards.append(f"Recommended feature family was '{best_family}'.")
        return cards

    def run_phase1(
        self,
        project_root: Path,
        target_behavior: str,
        progress_cb: Callable[[str], None] | None = None,
        force: bool = False,
        session_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        safe_behavior = self._sanitize_behavior_id(target_behavior)
        settings = self.load_or_init_settings(project_root)
        cfg = self._resolve_config(settings, safe_behavior)
        if force:
            cfg.enabled = True

        if not cfg.enabled:
            return {
                "enabled": False,
                "reason": "Phase 1 behavior-adaptive benchmarking is disabled.",
            }

        def _progress(msg: str) -> None:
            if progress_cb is not None:
                progress_cb(msg)

        _progress("Phase 1: preparing feature caches.")
        gpu_available = bool(cfg.use_gpu_if_available) and self._probe_xgboost_gpu()
        _progress(
            "Phase 1: benchmark backend "
            + ("GPU-enabled (xgboost/cuda)." if gpu_available else "CPU fallback (gpu unavailable or disabled).")
        )
        _progress(
            "Phase 1 note: feature-cache extraction is CPU-bound; GPU acceleration starts during expert/confound model fitting."
        )
        _progress("Phase 1: loading reviewed labels and deriving confound map.")
        review_intervals = self._build_review_interval_table(project_root)
        confound_map = self._derive_confound_map(review_intervals, safe_behavior, settings)
        self.save_settings(project_root, settings)

        # Benchmarking runs at a single representative segment window.
        selected_scales = [float(self.SEGMENT_SCALE_SEC)]

        # Determine which sessions to build the Phase 1 cache for.
        # Priority order:
        #   1. session_ids passed by caller (from the active pipeline run — most precise)
        #   2. quick_feature_test subset derived from reviewed labels/pose cache
        #   3. all sessions (no filter)
        subset_sessions: list[str] = []
        if session_ids:
            subset_sessions = [str(s) for s in session_ids]
        elif cfg.quick_feature_test:
            subset_sessions = self._subset_sessions_from_reviews(review_intervals, cfg.subset_max_sessions)
            if not subset_sessions:
                subset_sessions = self._subset_sessions_from_pose_cache(project_root, cfg.subset_max_sessions)

        if session_ids:
            _progress(
                f"Phase 1: session filter from pipeline run — {len(subset_sessions)} session(s) selected."
            )
        elif cfg.quick_feature_test:
            _progress(
                "Phase 1: quick feature-test mode enabled "
                f"(scales={selected_scales}, sessions={len(subset_sessions) if subset_sessions else 'all'})."
            )

        # Cap parallel workers for Phase 1: the process pool serializes full
        # session DataFrames to subprocesses.  When sessions are large (many
        # frames) or there are many sessions, excessive workers cause more
        # serialization overhead than they save.  Cap at 8 unless the caller
        # explicitly configured a higher value via ABEL_PHASE1_WORKERS.
        n_sessions_for_phase1 = len(subset_sessions) if subset_sessions else 999
        _env_workers = os.environ.get("ABEL_PHASE1_WORKERS", "").strip()
        _env_cap = 0
        try:
            _env_cap = max(0, int(_env_workers)) if _env_workers else 0
        except Exception:
            pass
        effective_workers = _env_cap or min(
            int(cfg.cpu_parallel_workers) if cfg.cpu_parallel_workers > 0 else (os.cpu_count() or 2),
            max(1, min(8, n_sessions_for_phase1)),
        )

        _progress(
            "Phase 1: CPU parallel cache workers="
            f"{effective_workers} "
            f"(process_pool={'on' if cfg.cpu_use_process_pool else 'off'})."
        )

        cache = self._feature_cache.get_or_build_multiscale_cache(
            project_root,
            MultiScaleFeatureCacheConfig(
                scales_sec=list(selected_scales),
                fps=float(read_yaml(project_root / "project.yaml", {}).get("default_fps", 30.0) or 30.0),
                regenerate=bool(cfg.regenerate_diagnostics),
                session_ids=subset_sessions or None,
                parallel_workers=effective_workers,
                use_process_pool=bool(cfg.cpu_use_process_pool),
            ),
            progress_cb=_progress,
        )

        benchmark_root = self._benchmark_root(project_root, safe_behavior)
        diagnostic_root = self._diagnostic_root(project_root, safe_behavior)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_bench_dir = benchmark_root / "runs" / timestamp
        run_diag_dir = diagnostic_root / "runs" / timestamp
        run_bench_dir.mkdir(parents=True, exist_ok=True)
        run_diag_dir.mkdir(parents=True, exist_ok=True)

        baseline_metrics = self._read_baseline_metrics(project_root)

        # Phase 1 benchmarking runs at a single representative segment window
        # (see ``selected_scales`` above), so the feature cache holds exactly one
        # entry. The expert pass, modality benchmarking, and confound analysis
        # all operate on that one window's labeled segments — no multi-scale
        # selection heuristics are needed, and the labeled table is derived once
        # and reused for the confound step.
        expert_results: dict[str, Any] = {}
        scale_results: dict[str, Any] = {}
        scale_labeled: pd.DataFrame | None = None

        _progress("Phase 1: running expert benchmarks.")
        for scale_key, meta in (cache.get("scales") or {}).items():
            seg_path = Path(str(meta.get("segment_features", "")))
            if not seg_path.exists():
                continue
            seg_df = pd.read_parquet(seg_path)
            labeled = self._assign_labels_to_segments(seg_df, review_intervals, safe_behavior, confound_map)
            if cfg.quick_feature_test and len(labeled) > int(cfg.subset_max_segments_per_scale):
                labeled = labeled.sample(n=int(cfg.subset_max_segments_per_scale), random_state=42)
            # Reused below for the confound analysis (avoids re-reading the parquet
            # and re-deriving labels).
            scale_labeled = labeled
            if labeled.empty:
                scale_results[scale_key] = {
                    "status": "skipped",
                    "reason": "No overlap between reviewed labels and segment windows.",
                    "scale_sec": float(meta.get("scale_sec", 0.0)),
                }
                continue

            families = self._feature_families(labeled)
            fused_cols = families.get("fused", [])
            scale_out = self._train_binary_expert(
                labeled,
                fused_cols,
                prefer_gpu=gpu_available,
            )
            scale_out["scale_sec"] = float(meta.get("scale_sec", 0.0))
            scale_out["n_labeled_rows"] = int(len(labeled))
            scale_results[scale_key] = scale_out

            if cfg.enable_modality_benchmarking:
                for family_name, cols in families.items():
                    expert_results[family_name] = self._train_binary_expert(
                        labeled,
                        cols,
                        prefer_gpu=gpu_available,
                    )

        # Confound analysis reuses the labeled segments from the pass above.
        _progress("Phase 1: computing confound analysis.")
        confound_results: dict[str, Any] = {"enabled": False, "reason": "not run"}
        if cfg.enable_confound_analysis and scale_labeled is not None:
            confound_results = self._compute_confound_analysis(
                scale_labeled,
                safe_behavior,
                prefer_gpu=gpu_available,
            )

        # Recommendations and weights.
        margin_stat = self._estimate_margin_stat(confound_results)
        learned_scores: dict[str, float] = {}
        for family, row in expert_results.items():
            if row.get("status") != "ok":
                continue
            ece = float(((row.get("reliability") or {}).get("ece", 1.0)))
            learned_scores[family] = self._composite_weight_score(
                ap=float(row.get("ap", 0.0)),
                ece=ece,
                margin_stat=margin_stat,
            )

        if learned_scores:
            keys = sorted(learned_scores)
            values = np.asarray([learned_scores[k] for k in keys], dtype=float)
            values = np.maximum(values, 1e-6)
            weights = values / np.sum(values)
            min_examples_gate = int(cfg.min_examples_for_learned_weights)
            pos_count = 0
            neg_count = 0
            for row in expert_results.values():
                bal = dict(row.get("class_balance") or {})
                pos_count = max(pos_count, int(bal.get("train_pos", 0)))
                neg_count = max(neg_count, int(bal.get("train_neg", 0)))
            alpha = float(min(1.0, min(pos_count, neg_count) / max(1.0, min_examples_gate)))
            equal = np.full_like(weights, 1.0 / len(weights))
            soft = alpha * weights + (1.0 - alpha) * equal
            learned_weights = {k: float(v) for k, v in zip(keys, soft)}
        else:
            learned_weights = {}
            alpha = 0.0

        best_expert = ""
        best_ap = -1.0
        for name, row in expert_results.items():
            if row.get("status") != "ok":
                continue
            ap = float(row.get("ap", -1.0))
            if ap > best_ap:
                best_ap = ap
                best_expert = name

        baseline_ap = float((baseline_metrics.get("segment_level") or {}).get("pr_auc", float("nan")))

        summary: dict[str, Any] = {
            "phase": "phase1",
            "target_behavior": safe_behavior,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "config": {
                "enabled": cfg.enabled,
                "enable_modality_benchmarking": cfg.enable_modality_benchmarking,
                "enable_confound_analysis": cfg.enable_confound_analysis,
                "diagnostics_enabled": cfg.diagnostics_enabled,
                "segment_scale_sec": float(self.SEGMENT_SCALE_SEC),
                "primary_metric": cfg.primary_metric,
                "min_examples_for_learned_weights": cfg.min_examples_for_learned_weights,
                "use_gpu_if_available": cfg.use_gpu_if_available,
                "gpu_backend_active": gpu_available,
                "quick_feature_test": cfg.quick_feature_test,
                "subset_max_sessions": cfg.subset_max_sessions,
                "subset_max_segments_per_scale": cfg.subset_max_segments_per_scale,
                "cpu_parallel_workers": cfg.cpu_parallel_workers,
                "cpu_use_process_pool": cfg.cpu_use_process_pool,
            },
            "feature_cache": cache,
            "expert_results": expert_results,
            "scale_results": scale_results,
            "confound": confound_results,
            "baseline_metrics": baseline_metrics,
            "baseline_vs_advanced": {
                "baseline_ap": baseline_ap,
                "best_adaptive_ap": float(best_ap if best_ap >= 0 else float("nan")),
                "best_expert": best_expert,
                "delta_ap": float(best_ap - baseline_ap) if np.isfinite(baseline_ap) and best_ap >= 0 else float("nan"),
            },
            "recommendations": {
                "best_family": best_expert,
                "learned_family_weights": learned_weights,
                "soft_adaptation_alpha": float(alpha),
            },
            "confound_label_map": confound_map,
        }

        _progress("Phase 1: writing benchmark artifacts and diagnostics.")
        write_json(run_bench_dir / "phase1_benchmark_summary.json", summary)
        cards = self._phase_card(summary)
        write_json(run_bench_dir / "phase1_summary_cards.json", {"cards": cards})

        fig_outputs: dict[str, str] = {}
        if cfg.diagnostics_enabled:
            fig_outputs = self._plot_diagnostics(
                behavior_id=safe_behavior,
                run_diag_dir=run_diag_dir,
                expert_results=expert_results,
                scale_results=scale_results,
                confound_results=confound_results,
                baseline_metrics=baseline_metrics,
                export_high_res=cfg.export_high_resolution,
            )

        latest_bench = benchmark_root / "latest.json"
        latest_diag = diagnostic_root / "latest.json"
        write_json(
            latest_bench,
            {
                "target_behavior": safe_behavior,
                "latest_run": timestamp,
                "summary_path": str(run_bench_dir / "phase1_benchmark_summary.json"),
                "cards_path": str(run_bench_dir / "phase1_summary_cards.json"),
            },
        )
        write_json(
            latest_diag,
            {
                "target_behavior": safe_behavior,
                "latest_run": timestamp,
                "diagnostic_dir": str(run_diag_dir),
                "figures": fig_outputs,
            },
        )

        # Persist per-behavior recommendations in editable project config.
        behavior_overrides = settings.setdefault("behavior_overrides", {})
        behavior_cfg = behavior_overrides.setdefault(safe_behavior, {})
        behavior_cfg["latest_phase1_run"] = timestamp
        behavior_cfg["recommended_family"] = best_expert
        behavior_cfg["learned_family_weights"] = learned_weights
        behavior_cfg["soft_adaptation_alpha"] = float(alpha)
        self.save_settings(project_root, settings)

        result = {
            "enabled": True,
            "target_behavior": safe_behavior,
            "run_id": timestamp,
            "benchmark_summary_path": str(run_bench_dir / "phase1_benchmark_summary.json"),
            "summary_cards": cards,
            "figures": fig_outputs,
            "latest_benchmark_pointer": str(latest_bench),
            "latest_diagnostics_pointer": str(latest_diag),
        }
        write_json(run_bench_dir / "phase1_result_manifest.json", result)
        return result
