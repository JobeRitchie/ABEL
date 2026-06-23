from __future__ import annotations

from abel.models.schemas import CandidateWindow, ReviewDecision, ReviewDecisionType
from abel.services.active_learning_scheduler import ActiveLearningScheduler


def test_hard_negative_selection_uses_rejected_status() -> None:
    scheduler = ActiveLearningScheduler()

    all_candidates = [
        CandidateWindow(
            window_id="c_rejected",
            session_id="session_a",
            start_frame=0,
            end_frame=10,
            behavior_id="target_behavior",
            total_score=0.8,
        ),
        CandidateWindow(
            window_id="c_unreviewed",
            session_id="session_a",
            start_frame=20,
            end_frame=30,
            behavior_id="target_behavior",
            total_score=0.75,
        ),
    ]

    reviewed_decisions = [
        ReviewDecision(
            decision_id="d1",
            clip_id="c_rejected",
            reviewer="tester",
            old_status="unscored",
            new_status="rejected",
            decision=ReviewDecisionType.REJECT,
        )
    ]

    hard_negatives = scheduler._find_hard_negatives(
        all_candidates=all_candidates,
        reviewed_decisions=reviewed_decisions,
        n_select=2,
    )

    assert [c.window_id for c in hard_negatives] == ["c_unreviewed"]
