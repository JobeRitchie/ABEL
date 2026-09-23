"""A window nominated by several behaviors keeps every nomination.

Batch runs (Retrain All / Pipeline All) generate candidates one behavior at a
time, and the queue holds one row per *segment* so a single extracted clip serves
every behavior that asked for it. The nominations therefore have to accumulate on
the row: storing a single ``behavior_id`` meant each behavior in the loop
overwrote the previous one, so a 15-model run left only the last behavior's
queue visible and nine behaviors looked as if they had generated nothing.
"""
from __future__ import annotations

from abel.models.schemas import CandidateWindow
from abel.services.candidate_service import CandidateGenerationService


def _win(bid: str, score: float = 0.5, clip: str | None = None) -> CandidateWindow:
    return CandidateWindow(
        window_id="seg_track_0_sessA_0_14",
        session_id="sessA",
        start_frame=0,
        end_frame=14,
        behavior_id=bid,
        total_score=score,
        clip_path=clip,
        source="active_learning_uncertainty",
    )


def test_behavior_id_backfills_the_list():
    w = _win("chase")
    assert w.behavior_ids == ["chase"]
    # And a list-built row keeps a primary id for older readers.
    w2 = CandidateWindow(
        window_id="w", session_id="s", start_frame=0, end_frame=1,
        behavior_ids=["attack", "chase"],
    )
    assert w2.behavior_id == "attack"


def test_upsert_merges_nominations_instead_of_replacing(tmp_path):
    svc = CandidateGenerationService()
    svc.set_project(tmp_path)
    (tmp_path / "derived" / "review_tables").mkdir(parents=True)

    added = svc.upsert_external_window_candidates([_win("chase", 0.4, clip="c.mp4")])
    assert added == 1
    # A later behavior in the batch nominates the same window.
    svc.upsert_external_window_candidates([_win("attack", 0.9)])
    svc.upsert_external_window_candidates([_win("boxing", 0.2)])

    rows = svc.load_external_window_candidates()
    assert len(rows) == 1, "one clip per window: nominations must not duplicate rows"
    assert set(rows[0].behavior_ids) == {"chase", "attack", "boxing"}
    assert rows[0].total_score == 0.9        # most uncertain nomination sets the rank
    assert rows[0].clip_path == "c.mp4"      # an already-rendered clip is not forgotten


def test_review_filter_matches_any_nomination():
    from abel.ui.tabs.review_tab import ReviewTab

    tab = type("T", (), {})()
    tab._decision_by_clip_id = {}
    tab._normalize_behavior_id = staticmethod(lambda b: str(b or "").strip())
    cand = _win("chase")
    cand = cand.model_copy(update={"behavior_ids": ["chase", "attack"]})
    ids = ReviewTab._nominating_behavior_ids(tab, cand)
    assert ids == {"chase", "attack"}


def _queue_tab(filtered: str, decisions: dict | None = None):
    from abel.ui.tabs.review_tab import ReviewTab

    tab = type("T", (), {})()
    tab._decision_by_clip_id = decisions or {}
    tab._normalize_behavior_id = staticmethod(lambda b: str(b or "").strip())
    tab._behavior_filter_combo = type("C", (), {"currentData": lambda self: filtered})()
    tab._effective_behavior_id = lambda c: ReviewTab._effective_behavior_id(tab, c)
    tab._nominating_behavior_ids = lambda c: ReviewTab._nominating_behavior_ids(tab, c)
    return tab, ReviewTab._queue_behavior_id


def test_shared_window_takes_the_filtered_behavior():
    # Primary nominator is allogroom, but the user is reviewing the attack queue:
    # the row must show and pre-label as attack, not allogroom.
    cand = _win("allogroom").model_copy(update={"behavior_ids": ["allogroom", "attack"]})
    tab, fn = _queue_tab("attack")
    assert fn(tab, cand) == "attack"
    tab, fn = _queue_tab("all")
    assert fn(tab, cand) == "allogroom"


def test_reviewed_window_keeps_its_saved_label():
    from types import SimpleNamespace

    cand = _win("allogroom").model_copy(update={"behavior_ids": ["allogroom", "attack"]})
    dec = SimpleNamespace(behavior_label="allogroom")
    tab, fn = _queue_tab("attack", {cand.window_id: dec})
    assert fn(tab, cand) == "allogroom"


def test_reviewed_clip_matches_every_per_animal_label():
    # Soundboard: track_0 attacks, track_1 submits. The decision keeps only the
    # first label (attack), but the clip must also appear under the submit filter.
    from types import SimpleNamespace

    from abel.ui.tabs.review_tab import ReviewTab

    cand = _win("attack")
    tab = type("T", (), {})()
    tab._normalize_behavior_id = staticmethod(lambda b: str(b or "").strip())
    tab._decision_by_clip_id = {cand.window_id: SimpleNamespace(behavior_label="attack")}
    tab._segment_labels_by_window = {("sessA", 0, 14): {"attack", "submit|groom"}}
    ids = ReviewTab._nominating_behavior_ids(tab, cand)
    assert ids == {"attack", "submit", "groom"}
