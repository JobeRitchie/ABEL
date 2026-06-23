import numpy as np
import pandas as pd

from abel.services.evaluation_service import EvaluationService


def test_segment_metrics_known_case() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])

    metrics = EvaluationService.segment_metrics(y_true, y_pred)

    assert abs(metrics["precision"] - 1.0) < 1e-9
    assert abs(metrics["recall"] - 0.5) < 1e-9
    assert abs(metrics["f1"] - (2.0 / 3.0)) < 1e-9


def test_frame_metrics_passthrough() -> None:
    frame = pd.DataFrame({"label_true": [1, 0, 1], "label_pred": [1, 0, 0]})
    metrics = EvaluationService.frame_metrics(frame)
    assert "precision" in metrics
    assert "recall" in metrics
