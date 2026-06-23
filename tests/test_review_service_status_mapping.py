from __future__ import annotations

from abel.models.schemas import ReviewDecisionType
from abel.services.review_service import ReviewService


def test_upsert_decision_maps_bookmark_to_bookmarked_status(tmp_path) -> None:
    service = ReviewService()
    service.set_project(tmp_path)

    decision = service.upsert_decision(
        clip_id="clip_1",
        reviewer="tester",
        decision=ReviewDecisionType.BOOKMARK,
    )

    assert decision.new_status == "bookmarked"

    persisted = service.load_decisions()
    assert len(persisted) == 1
    assert persisted[0].new_status == "bookmarked"
