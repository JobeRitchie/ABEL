from __future__ import annotations

from abel.models.schemas import ReviewDecision, ReviewDecisionType, SeedExample
from abel.temporal_refinement.temporal_refinement_service import TemporalRefinementService


def test_temporal_feedback_persists_and_merges_into_training_intervals(tmp_path) -> None:
    service = TemporalRefinementService()
    service.set_project(tmp_path)

    concept_id = "Dig4"
    session_id = "session_a"

    # False-positive intervals should be merged and treated as reviewed negatives.
    service.add_temporal_feedback_interval(
        concept_id=concept_id,
        session_id=session_id,
        start_frame=20,
        end_frame=10,
        feedback_type="false_positive",
    )
    summary = service.add_temporal_feedback_interval(
        concept_id=concept_id,
        session_id=session_id,
        start_frame=18,
        end_frame=24,
        feedback_type="false_positive",
    )

    assert summary["n_false_positive_intervals"] == 1
    assert summary["false_positive_intervals_by_session"][session_id] == [(10, 24)]

    # False-negative intervals should be treated as positives.
    summary = service.add_temporal_feedback_interval(
        concept_id=concept_id,
        session_id=session_id,
        start_frame=100,
        end_frame=120,
        feedback_type="false_negative",
    )

    assert summary["n_false_negative_intervals"] == 1
    assert summary["false_negative_intervals_by_session"][session_id] == [(100, 120)]

    pos, uncertain, hard_neg, reviewed_neg = service._reviewed_intervals(concept_id=concept_id)

    assert uncertain == {}
    assert hard_neg == {}
    assert pos[session_id] == [(100, 120)]
    assert reviewed_neg[session_id] == [(10, 24)]


def test_temporal_feedback_overrides_conflicting_reviewed_bouts(tmp_path) -> None:
    service = TemporalRefinementService()
    service.set_project(tmp_path)

    concept_id = "Dig4"
    session_id = "session_a"

    decisions = [
        ReviewDecision(
            decision_id="d_pos",
            clip_id="clip_pos",
            reviewer="tester",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.ACCEPT,
        ),
        ReviewDecision(
            decision_id="d_neg",
            clip_id="clip_neg",
            reviewer="tester",
            old_status="unscored",
            new_status="rejected",
            decision=ReviewDecisionType.REJECT,
        ),
    ]

    service._review.load_decisions = lambda: decisions
    service._candidate_interval_index = lambda: {
        "clip_pos": {
            "session_id": session_id,
            "start_frame": 10,
            "end_frame": 30,
            "behavior_id": concept_id,
        },
        "clip_neg": {
            "session_id": session_id,
            "start_frame": 40,
            "end_frame": 60,
            "behavior_id": concept_id,
        },
    }

    service.add_temporal_feedback_interval(
        concept_id=concept_id,
        session_id=session_id,
        start_frame=20,
        end_frame=25,
        feedback_type="false_positive",
    )
    service.add_temporal_feedback_interval(
        concept_id=concept_id,
        session_id=session_id,
        start_frame=45,
        end_frame=50,
        feedback_type="false_negative",
    )

    pos, uncertain, hard_neg, reviewed_neg = service._reviewed_intervals(concept_id=concept_id)

    assert uncertain == {}
    assert hard_neg == {}
    assert pos[session_id] == [(10, 19), (26, 30), (45, 50)]
    assert reviewed_neg[session_id] == [(20, 25), (40, 44), (51, 60)]


def test_no_behavior_accept_is_negative_for_generic_temporal_training(tmp_path) -> None:
    service = TemporalRefinementService()
    service.set_project(tmp_path)

    session_id = "session_a"
    decisions = [
        ReviewDecision(
            decision_id="d_no_behavior",
            clip_id="clip_no_behavior",
            reviewer="tester",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.ACCEPT,
            behavior_label="no_behavior",
        )
    ]

    service._review.load_decisions = lambda: decisions
    service._candidate_interval_index = lambda: {
        "clip_no_behavior": {
            "session_id": session_id,
            "start_frame": 100,
            "end_frame": 110,
            "behavior_id": "",
        }
    }

    pos, uncertain, hard_neg, reviewed_neg = service._reviewed_intervals(concept_id="target_behavior")

    assert pos == {}
    assert uncertain == {}
    assert hard_neg == {}
    assert reviewed_neg == {session_id: [(100, 110)]}


def test_accept_of_other_behavior_is_negative_for_target_behavior(tmp_path) -> None:
    service = TemporalRefinementService()
    service.set_project(tmp_path)

    concept_id = "behavior_a"
    session_id = "session_a"
    decisions = [
        ReviewDecision(
            decision_id="d_accept_other",
            clip_id="clip_other",
            reviewer="tester",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.ACCEPT,
            behavior_label="behavior_b",
        )
    ]

    service._review.load_decisions = lambda: decisions
    service._candidate_interval_index = lambda: {
        "clip_other": {
            "session_id": session_id,
            "start_frame": 25,
            "end_frame": 35,
            "behavior_id": "behavior_b",
        }
    }

    pos, uncertain, hard_neg, reviewed_neg = service._reviewed_intervals(concept_id=concept_id)

    assert pos == {}
    assert uncertain == {}
    assert hard_neg == {}
    assert reviewed_neg == {session_id: [(25, 35)]}


def test_relabel_of_other_behavior_is_negative_for_target_behavior(tmp_path) -> None:
    service = TemporalRefinementService()
    service.set_project(tmp_path)

    concept_id = "behavior_a"
    session_id = "session_a"
    decisions = [
        ReviewDecision(
            decision_id="d_relabel_other",
            clip_id="clip_other",
            reviewer="tester",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.RELABEL,
            behavior_label="behavior_b",
        )
    ]

    service._review.load_decisions = lambda: decisions
    service._candidate_interval_index = lambda: {
        "clip_other": {
            "session_id": session_id,
            "start_frame": 60,
            "end_frame": 70,
            "behavior_id": "behavior_b",
        }
    }

    pos, uncertain, hard_neg, reviewed_neg = service._reviewed_intervals(concept_id=concept_id)

    assert pos == {}
    assert uncertain == {}
    assert hard_neg == {}
    assert reviewed_neg == {session_id: [(60, 70)]}


def test_relabel_no_behavior_is_negative_in_generic_temporal_training(tmp_path) -> None:
    service = TemporalRefinementService()
    service.set_project(tmp_path)

    session_id = "session_a"
    decisions = [
        ReviewDecision(
            decision_id="d_relabel_nb",
            clip_id="clip_nb",
            reviewer="tester",
            old_status="unscored",
            new_status="reviewed",
            decision=ReviewDecisionType.RELABEL,
            behavior_label="no_behavior",
        )
    ]

    service._review.load_decisions = lambda: decisions
    service._candidate_interval_index = lambda: {
        "clip_nb": {
            "session_id": session_id,
            "start_frame": 80,
            "end_frame": 90,
            "behavior_id": "",
        }
    }

    pos, uncertain, hard_neg, reviewed_neg = service._reviewed_intervals(concept_id="target_behavior")

    assert pos == {}
    assert uncertain == {}
    assert hard_neg == {}
    assert reviewed_neg == {session_id: [(80, 90)]}


def test_skip_decisions_are_excluded_from_temporal_intervals(tmp_path) -> None:
    service = TemporalRefinementService()
    service.set_project(tmp_path)

    concept_id = "behavior_a"
    session_id = "session_a"
    decisions = [
        ReviewDecision(
            decision_id="d_skip",
            clip_id="clip_skip",
            reviewer="tester",
            old_status="unscored",
            new_status="skipped",
            decision=ReviewDecisionType.SKIP,
            behavior_label="behavior_a",
        )
    ]

    service._review.load_decisions = lambda: decisions
    service._candidate_interval_index = lambda: {
        "clip_skip": {
            "session_id": session_id,
            "start_frame": 40,
            "end_frame": 50,
            "behavior_id": "behavior_a",
        }
    }

    pos, uncertain, hard_neg, reviewed_neg = service._reviewed_intervals(concept_id=concept_id)

    assert pos == {}
    assert uncertain == {}
    assert hard_neg == {}
    assert reviewed_neg == {}


def test_seed_examples_are_used_in_temporal_refinement_intervals(tmp_path) -> None:
    service = TemporalRefinementService()
    service.set_project(tmp_path)

    concept_id = "Dig4"
    session_id = "session_seed"

    service._seeds.add(
        SeedExample(
            seed_id="",
            behavior_id=concept_id,
            session_id=session_id,
            start_frame=10,
            end_frame=20,
            label_type="positive",
        )
    )
    service._seeds.add(
        SeedExample(
            seed_id="",
            behavior_id=concept_id,
            session_id=session_id,
            start_frame=30,
            end_frame=40,
            label_type="negative",
        )
    )
    service._seeds.add(
        SeedExample(
            seed_id="",
            behavior_id="OtherBehavior",
            session_id=session_id,
            start_frame=50,
            end_frame=60,
            label_type="positive",
        )
    )

    pos, uncertain, hard_neg, reviewed_neg = service._reviewed_intervals(concept_id=concept_id)

    assert uncertain == {}
    assert hard_neg == {}
    assert pos == {session_id: [(10, 20)]}
    assert reviewed_neg == {session_id: [(30, 40), (50, 60)]}


def test_reject_decisions_are_concept_scoped_for_temporal_negatives(tmp_path) -> None:
    service = TemporalRefinementService()
    service.set_project(tmp_path)

    concept_id = "behavior_a"
    session_id = "session_a"

    decisions = [
        ReviewDecision(
            decision_id="d_other",
            clip_id="clip_other",
            reviewer="tester",
            old_status="unscored",
            new_status="rejected",
            decision=ReviewDecisionType.REJECT,
        )
    ]

    service._review.load_decisions = lambda: decisions
    service._candidate_interval_index = lambda: {
        "clip_other": {
            "session_id": session_id,
            "start_frame": 10,
            "end_frame": 20,
            "behavior_id": "behavior_b",
        }
    }

    pos, uncertain, hard_neg, reviewed_neg = service._reviewed_intervals(concept_id=concept_id)

    assert pos == {}
    assert uncertain == {}
    assert hard_neg == {}
    assert reviewed_neg == {}

