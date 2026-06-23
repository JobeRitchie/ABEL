from abel.models.schemas import CandidateWindow, ReviewDecision, ReviewDecisionType
from abel.services.export_service import ExportService


def test_accepted_behavior_intervals_use_adjusted_bounds() -> None:
    candidates = [
        CandidateWindow(window_id="a", session_id="s1", start_frame=10, end_frame=30, behavior_id="groom"),
        CandidateWindow(window_id="b", session_id="s1", start_frame=40, end_frame=80, behavior_id="rear"),
    ]
    decisions = [
        ReviewDecision(
            decision_id="d1",
            clip_id="a",
            reviewer="r",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.ACCEPT,
            adjusted_start_frame=12,
            adjusted_end_frame=25,
        ),
        ReviewDecision(
            decision_id="d2",
            clip_id="b",
            reviewer="r",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.REJECT,
        ),
    ]

    service = ExportService()
    intervals = service._accepted_behavior_intervals_by_session(candidates, decisions)

    assert intervals == {"s1": {"groom": [(12, 25)]}}


def test_accepted_behavior_intervals_prefer_decision_behavior_label() -> None:
    candidates = [
        CandidateWindow(window_id="a", session_id="s1", start_frame=10, end_frame=30, behavior_id="groom"),
    ]
    decisions = [
        ReviewDecision(
            decision_id="d1",
            clip_id="a",
            reviewer="r",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.ACCEPT,
            behavior_label="dig",
        ),
    ]

    service = ExportService()
    intervals = service._accepted_behavior_intervals_by_session(candidates, decisions)

    assert intervals == {"s1": {"dig": [(10, 30)]}}


def test_active_behaviors_for_frame() -> None:
    behavior_intervals = {
        "groom": [(10, 20)],
        "rear": [(15, 18), (30, 35)],
    }

    assert service_active(behavior_intervals, 9) == []
    assert service_active(behavior_intervals, 16) == ["groom", "rear"]
    assert service_active(behavior_intervals, 33) == ["rear"]


def service_active(behavior_intervals: dict[str, list[tuple[int, int]]], frame: int) -> list[str]:
    return ExportService._active_behaviors_for_frame(behavior_intervals, frame)


def test_overlay_supersample_scale_prefers_small_videos() -> None:
    assert ExportService._overlay_supersample_scale(480) == 3
    assert ExportService._overlay_supersample_scale(720) == 2
    assert ExportService._overlay_supersample_scale(1080) == 1


def test_overlay_supersample_scale_honors_explicit_override() -> None:
    assert ExportService._overlay_supersample_scale(360, {"overlay_supersample_scale": 4}) == 4
    assert ExportService._overlay_supersample_scale(360, {"overlay_supersample_scale": 99}) == 4
    assert ExportService._overlay_supersample_scale(360, {"overlay_supersample_scale": -1}) == 3
