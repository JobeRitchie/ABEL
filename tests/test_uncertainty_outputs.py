import numpy as np
import pandas as pd

from abel.services.uncertainty_service import UncertaintyScoringService


def test_uncertainty_score_columns_and_range() -> None:
    n = 12
    df = pd.DataFrame(
        {
            "segment_id": [f"seg_{i}" for i in range(n)],
            "f1": np.linspace(0.0, 1.0, n),
            "f2": np.linspace(1.0, 0.0, n),
        }
    )
    probs = np.column_stack([np.linspace(0.2, 0.8, n), np.linspace(0.8, 0.2, n)])
    ens = [probs, probs * 0.98 + 0.01, probs * 1.01 - 0.005]

    svc = UncertaintyScoringService()
    out = svc.score_segments(
        segment_df=df,
        class_probs=probs,
        ensemble_probs=ens,
        feature_cols=["f1", "f2"],
    )

    assert "uncertainty_score" in out.columns
    assert out["uncertainty_score"].between(0.0, 1.0).all()
    assert out["prediction_variance"].ge(0.0).all()
