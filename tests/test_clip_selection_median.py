from abel.models.schemas import CandidateWindow
from abel.ui.tabs.preprocessing_tab import ClipExtractionTab


def _cand(i: int, score: float) -> CandidateWindow:
    return CandidateWindow(
        window_id=f"w{i}",
        session_id="s1",
        start_frame=i * 10,
        end_frame=i * 10 + 9,
        behavior_id="b1",
        total_score=score,
    )


def test_select_top_bottom_and_median_includes_middle_without_duplicates() -> None:
    # Already sorted high->low, matching extraction flow.
    candidates = [_cand(i, score) for i, score in enumerate([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])]

    selected = ClipExtractionTab._select_top_bottom_and_median(
        candidates=candidates,
        top_n=1,
        bottom_n=1,
        median_n=3,
    )

    selected_ids = [c.window_id for c in selected]

    # top + bottom + center-centered median picks
    assert selected_ids == ["w0", "w6", "w3", "w4", "w2"]
    assert len(set(selected_ids)) == len(selected_ids)


def test_select_top_bottom_and_median_handles_overlap_between_buckets() -> None:
    candidates = [_cand(i, score) for i, score in enumerate([0.9, 0.8, 0.7])]

    selected = ClipExtractionTab._select_top_bottom_and_median(
        candidates=candidates,
        top_n=2,
        bottom_n=2,
        median_n=2,
    )

    selected_ids = [c.window_id for c in selected]

    # all candidates should appear at most once, even when buckets overlap.
    assert selected_ids == ["w0", "w1", "w2"]
    assert len(set(selected_ids)) == len(selected_ids)
