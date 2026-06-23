import numpy as np
import pandas as pd

from abel.temporal_refinement.temporal_refinement_service import TemporalRefinementConfig, TemporalRefinementService


def _count_negatives(samples) -> int:
    return sum(1 for s in samples if int(s.label) == 0)


def test_build_samples_falls_back_when_reviewed_negatives_absent() -> None:
    service = TemporalRefinementService()
    cfg = TemporalRefinementConfig(
        use_unreviewed_negatives=False,
        use_reviewed_windows_only=False,
        include_hard_negatives=False,
        training_stride_positive=1,
        training_stride_negative=5,
    )

    session_features = {"session_a": np.zeros((100, 3), dtype=np.float32)}
    subject_map = {"session_a": "subject_a"}
    reviewed_positive = {"session_a": [(10, 19)]}

    samples = service._build_samples(
        concept_id="target_behavior",
        cfg=cfg,
        session_features=session_features,
        subject_map=subject_map,
        reviewed_positive=reviewed_positive,
        hard_negative={},
        reviewed_negative={},
        labels_df=pd.DataFrame(),
        window_frames=15,
    )

    assert _count_negatives(samples) > 0


def test_build_samples_respects_reviewed_negative_intervals_when_present() -> None:
    service = TemporalRefinementService()
    cfg = TemporalRefinementConfig(
        use_unreviewed_negatives=False,
        use_reviewed_windows_only=False,
        include_hard_negatives=False,
        training_stride_positive=1,
        training_stride_negative=1,
    )

    session_features = {"session_a": np.zeros((80, 3), dtype=np.float32)}
    subject_map = {"session_a": "subject_a"}
    reviewed_positive = {"session_a": [(10, 14)]}
    reviewed_negative = {"session_a": [(30, 35)]}

    samples = service._build_samples(
        concept_id="target_behavior",
        cfg=cfg,
        session_features=session_features,
        subject_map=subject_map,
        reviewed_positive=reviewed_positive,
        hard_negative={},
        reviewed_negative=reviewed_negative,
        labels_df=pd.DataFrame(),
        window_frames=15,
    )

    negative_centers = {int(s.center_frame) for s in samples if int(s.label) == 0}
    assert negative_centers
    assert negative_centers.issubset({30, 31, 32, 33, 34, 35})


def test_build_samples_reviewed_only_mode_disables_background_negative_fallback() -> None:
    service = TemporalRefinementService()
    cfg = TemporalRefinementConfig(
        use_unreviewed_negatives=False,
        use_reviewed_windows_only=True,
        include_hard_negatives=False,
        training_stride_positive=1,
        training_stride_negative=1,
    )

    session_features = {"session_a": np.zeros((60, 3), dtype=np.float32)}
    subject_map = {"session_a": "subject_a"}
    reviewed_positive = {"session_a": [(10, 14)]}

    samples = service._build_samples(
        concept_id="target_behavior",
        cfg=cfg,
        session_features=session_features,
        subject_map=subject_map,
        reviewed_positive=reviewed_positive,
        hard_negative={},
        reviewed_negative={},
        labels_df=pd.DataFrame(),
        window_frames=15,
    )

    assert _count_negatives(samples) == 0


def test_temporal_refinement_defaults_to_reviewed_windows_only() -> None:
    cfg = TemporalRefinementConfig()
    assert cfg.use_unreviewed_negatives is False
    assert cfg.use_reviewed_windows_only is True
