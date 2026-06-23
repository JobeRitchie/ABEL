from pathlib import Path

import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.temporal_refinement.temporal_refinement_service import TemporalRefinementConfig, TemporalRefinementService


def test_behavior_model_positive_mining_from_scored_segments(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path
    model_version = "behavior_model_test_v1"
    model_dir = project_root / "derived" / "models" / model_version
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_state.pkl").write_bytes(b"stub")

    seg_df = pd.DataFrame(
        {
            "segment_id": ["a", "b", "c"],
            "session_id": ["session_1", "session_1", "session_2"],
            "start_frame": [10, 20, 5],
            "end_frame": [15, 25, 9],
            "f1": [0.1, 0.2, 0.3],
        }
    )
    seg_path = project_root / "derived" / "representations" / "segment_features.parquet"
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    seg_df.to_parquet(seg_path, index=False)

    def _fake_predict(model_dir: Path, segment_df: pd.DataFrame) -> pd.DataFrame:
        _ = model_dir
        return pd.DataFrame(
            {
                "segment_id": segment_df["segment_id"],
                "start_frame": segment_df["start_frame"],
                "end_frame": segment_df["end_frame"],
                "animal_id": ["a1", "a1", "a2"],
                "session_id": segment_df["session_id"],
                "prediction_prob": [0.9, 0.81, 0.6],
            }
        )

    monkeypatch.setattr(ActiveLearningTrainerService, "predict_segments", staticmethod(_fake_predict))

    service = TemporalRefinementService()
    service.set_project(project_root)
    cfg = TemporalRefinementConfig(
        include_behavior_model_positives=True,
        behavior_model_positive_threshold=0.8,
        behavior_model_version=model_version,
    )

    intervals, resolved = service._behavior_model_positive_intervals(
        concept_id="abc",
        cfg=cfg,
        session_lengths={"session_1": 100, "session_2": 50},
    )

    assert resolved == model_version
    assert intervals == {"session_1": [(10, 15), (20, 25)]}
    assert (model_dir / "segment_predictions.parquet").exists()
