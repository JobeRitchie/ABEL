from __future__ import annotations

import pandas as pd

from abel.ui.tabs.active_learning_tab import ActiveLearningTab


def _tab_stub() -> ActiveLearningTab:
    return ActiveLearningTab.__new__(ActiveLearningTab)


def test_remap_review_labels_falls_back_to_overlap_when_exact_window_missing() -> None:
    tab = _tab_stub()

    labels = pd.DataFrame(
        [
            {
                "segment_id": "rand_session_demo_100_129",
                "review_label": "behavior_A",
                "reviewer_id": "reviewer",
                "confidence": 1.0,
            }
        ]
    )
    segments = pd.DataFrame(
        [
            {"segment_id": "session_demo_96_125", "session_id": "session_demo", "start_frame": 96, "end_frame": 125},
            {"segment_id": "session_demo_126_155", "session_id": "session_demo", "start_frame": 126, "end_frame": 155},
        ]
    )

    remapped = tab._remap_review_labels_to_current_windows(labels, segments)
    assert not remapped.empty
    assert remapped.iloc[0]["segment_id"] == "session_demo_96_125"

    merged = tab._aggregate_reviewer_labels(segments, remapped)
    assert len(merged) == 1
    assert str(merged.iloc[0]["label"]) == "behavior_A"


def test_remap_review_labels_handles_shorter_clip_than_current_window() -> None:
    tab = _tab_stub()

    labels = pd.DataFrame(
        [
            {
                "segment_id": "rand_session_demo_100_129",
                "review_label": "behavior_B",
                "reviewer_id": "reviewer",
                "confidence": 1.0,
            }
        ]
    )
    segments = pd.DataFrame(
        [
            {"segment_id": "session_demo_90_149", "session_id": "session_demo", "start_frame": 90, "end_frame": 149},
            {"segment_id": "session_demo_150_209", "session_id": "session_demo", "start_frame": 150, "end_frame": 209},
        ]
    )

    # Current window length is 60, reviewed clip length is 30; remap should use overlap fallback.
    remapped = tab._remap_review_labels_to_current_windows(labels, segments)
    assert not remapped.empty
    assert remapped.iloc[0]["segment_id"] == "session_demo_90_149"

    merged = tab._aggregate_reviewer_labels(segments, remapped)
    assert len(merged) == 1
    assert str(merged.iloc[0]["label"]) == "behavior_B"
