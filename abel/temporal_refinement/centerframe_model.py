"""Baseline center-frame classifier for temporal refinement."""

from __future__ import annotations

import pickle
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal
from typing import cast

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

from abel.utils import xgb_predict


@dataclass
class CenterFrameModelConfig:
    """Model/training config for the baseline temporal classifier."""

    hidden_layer_sizes: tuple[int, ...] = (128, 64)
    activation: str = "relu"
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    max_iter: int = 200
    random_state: int = 42
    prefer_gpu: bool = True


class CenterFrameModel:
    """MLP over flattened temporal windows (window -> center-frame probability)."""

    def __init__(self, config: CenterFrameModelConfig | None = None) -> None:
        self.config = config or CenterFrameModelConfig()
        self._clf: Any | None = None
        self.window_frames: int = 0
        self.n_features: int = 0

    @staticmethod
    def _flatten(batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 3:
            raise ValueError("Expected batch shape [n_samples, window_frames, n_features]")
        n, w, f = batch.shape
        return batch.reshape(n, w * f)

    @staticmethod
    def _balanced_resample(X: np.ndarray, y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X, y

        max_count = int(max(counts))
        rng = np.random.RandomState(int(random_state))
        take_indices: list[int] = []
        for cls, cls_count in zip(classes, counts):
            cls_idx = np.where(y == cls)[0]
            if int(cls_count) >= max_count:
                sampled = rng.choice(cls_idx, size=max_count, replace=False)
            else:
                sampled = rng.choice(cls_idx, size=max_count, replace=True)
            take_indices.extend(sampled.tolist())

        rng.shuffle(take_indices)
        idx = np.asarray(take_indices, dtype=int)
        return X[idx], y[idx]

    def fit(
        self,
        train_loader: tuple[np.ndarray, np.ndarray],
        val_loader: tuple[np.ndarray, np.ndarray] | None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fit classifier and return validation metrics when available."""
        X_train, y_train = train_loader
        if X_train.ndim != 3:
            raise ValueError("Expected training tensor shape [n, window_frames, n_features]")
        if len(X_train) == 0:
            raise ValueError("Training set is empty")

        cfg = dict(config or {})
        self.window_frames = int(X_train.shape[1])
        self.n_features = int(X_train.shape[2])

        X_flat = self._flatten(X_train)
        y_arr = np.asarray(y_train, dtype=int)
        X_bal, y_bal = self._balanced_resample(X_flat, y_arr, random_state=int(cfg.get("random_state", self.config.random_state)))

        model_cfg = CenterFrameModelConfig(
            hidden_layer_sizes=tuple(cfg.get("hidden_layer_sizes", self.config.hidden_layer_sizes)),
            activation=str(cfg.get("activation", self.config.activation)),
            alpha=float(cfg.get("alpha", self.config.alpha)),
            learning_rate_init=float(cfg.get("learning_rate_init", self.config.learning_rate_init)),
            max_iter=int(cfg.get("max_iter", self.config.max_iter)),
            random_state=int(cfg.get("random_state", self.config.random_state)),
            prefer_gpu=bool(cfg.get("prefer_gpu", self.config.prefer_gpu)),
        )
        self.config = model_cfg
        backend = "mlp_cpu"
        gpu_requested = bool(model_cfg.prefer_gpu)
        gpu_used = False
        gpu_fallback_reason = ""

        if gpu_requested and platform.system().lower().startswith("win"):
            try:
                from xgboost import XGBClassifier

                n_classes = int(len(np.unique(y_arr)))
                if n_classes <= 2:
                    objective = "binary:logistic"
                    eval_metric = "logloss"
                else:
                    objective = "multi:softprob"
                    eval_metric = "mlogloss"

                # scale_pos_weight < 0 means auto (n_neg / n_pos); 1.0 preserves
                # the natural class imbalance, making the model conservative
                # (higher precision). Increase toward n_neg/n_pos to boost recall.
                raw_spw = float(cfg.get("xgb_scale_pos_weight", 1.0))
                n_pos_train = max(1, int((y_arr == 1).sum()))
                n_neg_train = max(1, int((y_arr == 0).sum()))
                spw = float(n_neg_train) / float(n_pos_train) if raw_spw < 0 else raw_spw

                xgb = XGBClassifier(
                    objective=objective,
                    eval_metric=eval_metric,
                    n_estimators=int(cfg.get("xgb_n_estimators", 300)),
                    max_depth=int(cfg.get("xgb_max_depth", 6)),
                    learning_rate=float(cfg.get("xgb_learning_rate", 0.05)),
                    subsample=float(cfg.get("xgb_subsample", 0.9)),
                    colsample_bytree=float(cfg.get("xgb_colsample_bytree", 0.9)),
                    scale_pos_weight=spw if n_classes <= 2 else 1.0,
                    tree_method="hist",
                    device="cuda",
                    random_state=model_cfg.random_state,
                )
                fit_kwargs: dict[str, Any] = {}
                if val_loader is not None and len(val_loader[0]) > 0:
                    X_val, y_val = val_loader
                    fit_kwargs["eval_set"] = [(self._flatten(X_val), np.asarray(y_val, dtype=int))]
                    log_every_n = max(1, int(cfg.get("xgb_log_every_n", 50)))
                    fit_kwargs["verbose"] = log_every_n

                xgb.fit(X_flat, y_arr, **fit_kwargs)
                self._clf = xgb
                backend = "xgboost_cuda"
                gpu_used = True
            except Exception as exc:
                gpu_fallback_reason = str(exc).splitlines()[0]

        if self._clf is None:
            activation_name = model_cfg.activation.lower().strip()
            if activation_name == "identity":
                activation_literal: Literal["relu", "identity", "logistic", "tanh"] = "identity"
            elif activation_name == "logistic":
                activation_literal = "logistic"
            elif activation_name == "tanh":
                activation_literal = "tanh"
            else:
                activation_literal = "relu"

            self._clf = MLPClassifier(
                hidden_layer_sizes=model_cfg.hidden_layer_sizes,
                activation=activation_literal,
                alpha=model_cfg.alpha,
                learning_rate_init=model_cfg.learning_rate_init,
                max_iter=model_cfg.max_iter,
                random_state=model_cfg.random_state,
                verbose=bool(cfg.get("mlp_verbose", True)),
            )
            self._clf.fit(X_bal, y_bal)

        metrics: dict[str, Any] = {
            "train_rows": float(len(X_train)),
            "model_backend": backend,
            "gpu_requested": bool(gpu_requested),
            "gpu_used": bool(gpu_used),
            "gpu_fallback_reason": str(gpu_fallback_reason),
        }
        if val_loader is not None and len(val_loader[0]) > 0:
            X_val, y_val = val_loader
            val_prob = self.predict_proba(X_val)
            val_pred = (val_prob >= 0.5).astype(int)
            metrics.update(
                {
                    "val_precision": float(precision_score(y_val, val_pred, zero_division=0)),
                    "val_recall": float(recall_score(y_val, val_pred, zero_division=0)),
                    "val_f1": float(f1_score(y_val, val_pred, zero_division=0)),
                }
            )
        return metrics

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        """Predict positive-class probabilities."""
        if self._clf is None:
            raise RuntimeError("Model is not fitted")
        X_flat = self._flatten(batch)
        # Scores on the CPU: a GPU-trained booster would copy X_flat host→device on
        # every call, which is what used to raise the device-mismatch warning this
        # method suppressed.  See abel.utils.xgb_predict.
        probs = cast(np.ndarray, xgb_predict.predict_proba(self._clf, X_flat))
        if probs.shape[1] == 1:
            return np.asarray(probs[:, 0], dtype=np.float32)
        return np.asarray(probs[:, 1], dtype=np.float32)

    def save(self, path: Path) -> None:
        """Persist model with shape metadata."""
        if self._clf is None:
            raise RuntimeError("Model is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "window_frames": self.window_frames,
            "n_features": self.n_features,
            "classifier": self._clf,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: Path) -> "CenterFrameModel":
        """Load model from disk."""
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        model = cls(config=payload.get("config") or CenterFrameModelConfig())
        model.window_frames = int(payload.get("window_frames", 0))
        model.n_features = int(payload.get("n_features", 0))
        model._clf = payload.get("classifier")
        if model._clf is None:
            raise RuntimeError("Invalid model payload")
        return model
