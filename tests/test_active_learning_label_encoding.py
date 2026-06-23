from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService, TrainingConfig


class ContiguousCheckingEstimator:
    def __init__(self, *args, **kwargs) -> None:
        self._n_classes = 2

    def get_params(self, deep: bool = True):
        return {}

    def fit(self, x, y, sample_weight=None):
        y_arr = np.asarray(y, dtype=int)
        uniq = np.unique(y_arr)
        expected = np.arange(len(uniq), dtype=int)
        if not np.array_equal(uniq, expected):
            raise ValueError(
                f"Invalid classes inferred from unique values of y. Expected {expected.tolist()}, got {uniq.tolist()}"
            )
        self._n_classes = int(len(uniq))
        return self

    def predict_proba(self, x):
        n = len(x)
        if self._n_classes <= 1:
            return np.ones((n, 1), dtype=float)
        p = np.full(n, 0.5, dtype=float)
        return np.column_stack([1.0 - p, p])


def _scope_df() -> pd.DataFrame:
    return pd.DataFrame({
        "segment_id": ["a", "b", "c", "d"],
        "session_id": ["s1", "s2", "X__s1", "X__s2"],
        "label_source": ["reviewer", "seed", "imported:X", "imported:X"],
        "label": ["walk", "no_behavior", "walk", "rear"],
    })


def test_scope_excludes_imported_when_toggled_off():
    df = _scope_df()
    out = ActiveLearningTrainerService._scope_training_rows(df, None, include_imported=False)
    assert set(out["segment_id"]) == {"a", "b"}  # imported X rows dropped


def test_scope_includes_imported_by_default():
    df = _scope_df()
    out = ActiveLearningTrainerService._scope_training_rows(df, None, include_imported=True)
    assert len(out) == 4


def test_scope_keeps_imported_through_session_filter():
    df = _scope_df()
    # Session scope = {s1} would drop s2; imported rows must survive regardless.
    out = ActiveLearningTrainerService._scope_training_rows(df, {"s1"}, include_imported=True)
    assert set(out["segment_id"]) == {"a", "c", "d"}
    # ...and are removed when the toggle is off, even within a session scope.
    out2 = ActiveLearningTrainerService._scope_training_rows(df, {"s1"}, include_imported=False)
    assert set(out2["segment_id"]) == {"a"}


def test_train_uses_contiguous_label_ids(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "project"
    train_dir = project_root / "derived" / "training_sets"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Full dataset has 3 labels, but train split below keeps only 2 labels.
    # The trainer must still encode train labels as contiguous [0, 1].
    df = pd.DataFrame(
        {
            "segment_id": ["seg_a", "seg_c", "seg_b"],
            "animal_id": ["animal_1", "animal_2", "animal_3"],
            "session_id": ["session_1", "session_2", "session_3"],
            "label": ["alpha", "charlie", "bravo"],
            "f0": [0.1, 0.2, 0.3],
            "f1": [1.0, 1.1, 1.2],
        }
    )
    df.to_parquet(train_dir / "training_set.parquet", index=False)

    trainer = ActiveLearningTrainerService()

    monkeypatch.setattr(
        ActiveLearningTrainerService,
        "_split",
        staticmethod(lambda _df, _strategy, _test_size, _random_state: (np.asarray([0, 1]), np.asarray([2]))),
    )
    monkeypatch.setattr(
        ActiveLearningTrainerService,
        "_make_estimator",
        staticmethod(lambda _family, _params, _random_state: ContiguousCheckingEstimator()),
    )

    result = trainer.train(
        project_root=project_root,
        config=TrainingConfig(
            classifier_family="xgboost",
            calibration_method="none",
            split_strategy="group_shuffle_session",
            model_version="test_model_v1",
        ),
    )

    metrics = result["metrics"]
    assert metrics["n_train"] == 2
    # Validation row has an unseen label and is dropped; trainer falls back to train rows.
    assert metrics["n_val"] == 2
