"""Training a behavior with no labeled examples must fail, not ship a one-class model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services.active_learning_trainer_service import (
    ActiveLearningTrainerService,
    TrainingConfig,
)


def _frame(labels: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i, lbl in enumerate(labels):
        rows.append({
            "segment_id": f"seg_{i}",
            "session_id": f"s{i % 4}",
            "animal_id": f"a{i % 4}",
            "start_frame": i * 15,
            "end_frame": i * 15 + 14,
            "label": lbl,
            "label_source": "user_review",
            "reviewer_confidence": 1.0,
            "feat_a": float(rng.normal()),
            "feat_b": float(rng.normal()),
        })
    return pd.DataFrame(rows)


def _cfg(target: str) -> TrainingConfig:
    return TrainingConfig(
        classifier_family="hist_gbdt", calibration_method="none",
        target_label=target, random_state=0, adaptive_complexity=False,
        enable_feature_augmentation=False,
    )


def test_target_with_no_examples_raises():
    # Mirrors a project with "Rear" (labeled) and a duplicate "Rearing" (never labeled).
    df = _frame(["rear", "no_behavior"] * 40)
    with pytest.raises(ValueError, match="no labeled examples"):
        ActiveLearningTrainerService().train_and_evaluate(df, _cfg("rearing"))


def test_target_with_examples_still_trains():
    df = _frame(["rear", "no_behavior"] * 40)
    result = ActiveLearningTrainerService().train_and_evaluate(df, _cfg("rear"))
    assert result.metrics["n_train_pos"] > 0
