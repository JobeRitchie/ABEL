import numpy as np
import pandas as pd

from abel.temporal_refinement.centerframe_dataset import CenterFrameDataset
from abel.temporal_refinement.temporal_refinement_service import TemporalRefinementService


def test_phase3_strategy_comparison_soft_fusion_scores_between_baseline_and_best() -> None:
    summary = {
        "baseline_vs_advanced": {
            "baseline_ap": 0.42,
            "best_adaptive_ap": 0.61,
            "best_expert": "pose",
        }
    }

    out = TemporalRefinementService._compute_strategy_comparison(
        phase1_summary=summary,
        strategy="soft_fusion",
        soft_fusion_alpha=0.5,
    )

    assert out["strategy_requested"] == "soft_fusion"
    assert out["scores"]["baseline"] == 0.42
    assert out["scores"]["best_expert"] == 0.61
    assert 0.42 < out["scores"]["soft_fusion"] < 0.61


def test_phase3_weighted_resample_expands_rows() -> None:
    ds = CenterFrameDataset(
        X=np.zeros((3, 5, 2), dtype=np.float32),
        y=np.asarray([1, 0, 0], dtype=np.int32),
        metadata=pd.DataFrame(
            {
                "session_id": ["s1", "s1", "s1"],
                "subject_id": ["a", "a", "a"],
                "center_frame": [10, 11, 12],
                "source": ["positive", "negative", "hard_negative"],
            }
        ),
    )

    out, diag = TemporalRefinementService._weighted_resample_dataset(
        dataset=ds,
        sample_weights=np.asarray([1.0, 1.0, 3.0], dtype=float),
        random_state=42,
    )

    assert diag["enabled"] is True
    assert int(diag["output_rows"]) > int(diag["input_rows"])
    assert len(out) > len(ds)
