import pandas as pd

from abel.services.evaluation_service import BoutMergeConfig, EvaluationService


def test_merge_bouts_respects_gap_and_min_duration() -> None:
    segments = pd.DataFrame(
        [
            {"animal_id": "a", "session_id": "s", "start_frame": 10, "end_frame": 20},
            {"animal_id": "a", "session_id": "s", "start_frame": 23, "end_frame": 35},
            {"animal_id": "a", "session_id": "s", "start_frame": 80, "end_frame": 86},
        ]
    )

    bouts = EvaluationService.merge_bouts(
        segments,
        BoutMergeConfig(max_gap_frames=3, min_bout_duration=10),
    )

    assert len(bouts) == 1
    assert int(bouts.iloc[0]["start_frame"]) == 10
    assert int(bouts.iloc[0]["end_frame"]) == 35
