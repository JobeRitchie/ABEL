"""Multi-animal soundboard labels -> per-animal segment records -> pooled training.

Covers the "a mouse is a mouse" pooling design: solo behaviors label the focal
animal, directed social behaviors label only the actor, mutual social behaviors
label both animals, and multiple behaviors on one animal-segment merge into a
co-occurring pipe-joined label (rather than collapsing to ``ambiguous``). The
end-to-end join uses the real training-set aggregation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.models.schemas import BehaviorDefinition, ReviewDecisionType, ReviewerLabelRecord
from abel.services.behavior_service import BehaviorService
from abel.services.review_service import ReviewService
from abel.ui.tabs.active_learning_tab import ActiveLearningTab

SESS = "sessA"


def _bd(bid, name, social, direction):
    return BehaviorDefinition(
        behavior_id=bid, name=name, short_name=name[:3],
        is_social=social, directionality=direction,
    )


class _Svc(BehaviorService):
    def __init__(self, behaviors):
        self._behaviors = behaviors


@pytest.fixture()
def svc():
    return _Svc([
        _bd("rearing", "Rearing", False, "none"),
        _bd("grooming", "Grooming", False, "none"),
        _bd("fighting", "Fighting", True, "directed"),
        _bd("sniffing", "Sniffing", True, "mutual"),
    ])


def test_solo_labels_focal_only(svc):
    recs = svc.aggregate_clip_labels(
        [{"behavior_id": "rearing", "focal_animal_id": "track_0", "partner_animal_id": None}],
        SESS, 0, 29,
    )
    assert len(recs) == 1
    assert recs[0]["segment_id"] == f"seg_track_0_{SESS}_0_29"
    assert recs[0]["review_label"] == "rearing"
    assert recs[0]["fields"]["social_role"] == "none"


def test_directed_labels_actor_only(svc):
    recs = svc.aggregate_clip_labels(
        [{"behavior_id": "fighting", "focal_animal_id": "track_0", "partner_animal_id": "track_1"}],
        SESS, 0, 29,
    )
    segs = {r["segment_id"] for r in recs}
    assert segs == {f"seg_track_0_{SESS}_0_29"}  # actor only, not the recipient
    assert recs[0]["fields"]["social_role"] == "actor"


def test_mutual_labels_both_animals(svc):
    recs = svc.aggregate_clip_labels(
        [{"behavior_id": "sniffing", "focal_animal_id": "track_0", "partner_animal_id": "track_1"}],
        SESS, 0, 29,
    )
    segs = {r["segment_id"] for r in recs}
    assert segs == {f"seg_track_0_{SESS}_0_29", f"seg_track_1_{SESS}_0_29"}
    assert all(r["review_label"] == "sniffing" for r in recs)
    assert all(r["fields"]["social_role"] == "mutual" for r in recs)


def test_co_occurring_merge_not_ambiguous(svc):
    # One animal fights (directed) and sniffs (mutual) the same partner in one window.
    recs = svc.aggregate_clip_labels(
        [
            {"behavior_id": "fighting", "focal_animal_id": "track_0", "partner_animal_id": "track_1"},
            {"behavior_id": "sniffing", "focal_animal_id": "track_0", "partner_animal_id": "track_1"},
        ],
        SESS, 30, 59,
    )
    by_seg = {r["segment_id"]: r["review_label"] for r in recs}
    # Actor's segment carries both behaviors as one pipe-joined label.
    assert by_seg[f"seg_track_0_{SESS}_30_59"] == "fighting|sniffing"
    # The mutual sniff still reaches the partner's segment.
    assert by_seg[f"seg_track_1_{SESS}_30_59"] == "sniffing"
    # The merged record keeps the structured columns that are unambiguous. The
    # focal animal always is (the segment is keyed by it); the partner survives
    # here because both behaviors name the same one. The role does not, because
    # track_0 is the *actor* of the fight and a *mutual* sniffer, and the column
    # holds one value.
    merged = next(r for r in recs if r["review_label"] == "fighting|sniffing")
    assert merged["fields"]["focal_animal_id"] == "track_0"
    assert merged["fields"]["partner_animal_id"] is None
    assert merged["fields"]["social_role"] == "none"


def test_merged_fields_keep_partner_when_roles_agree(svc):
    """A social + solo pair leaves exactly one social role, so it is kept."""
    recs = svc.aggregate_clip_labels(
        [
            {"behavior_id": "fighting", "focal_animal_id": "track_0", "partner_animal_id": "track_1"},
            {"behavior_id": "rearing", "focal_animal_id": "track_0", "partner_animal_id": None},
        ],
        SESS, 30, 59,
    )
    merged = next(r for r in recs if "|" in r["review_label"])
    assert merged["review_label"] == "fighting|rearing"
    assert merged["fields"] == {
        "focal_animal_id": "track_0",
        "partner_animal_id": "track_1",
        "social_role": "actor",
    }


def test_directed_recipient_can_be_labeled_no_behavior(svc):
    """The groomee doing nothing is a legal pair, not a contradiction.

    A directed behavior writes no row for the recipient, so marking that animal
    as the universal negative must leave the actor's label alone. Regression:
    the soundboard dropped any staged label naming the selected animal as the
    *partner*, which deleted the directed label outright, so the clip committed
    as two negatives and the interaction was never seen by training.
    """
    from PySide6.QtWidgets import QApplication

    from abel.services.behavior_service import NO_BEHAVIOR_ID
    from abel.ui.behavior_soundboard import BehaviorSoundboard

    QApplication.instance() or QApplication([])
    sb = BehaviorSoundboard()
    sb.configure(
        [("fighting", "Fighting", "f", True, "directed"),
         ("sniffing", "Sniffing", "s", True, "mutual"),
         (NO_BEHAVIOR_ID, "No Behavior", "n", False, "none")],
        lambda _b: None, {},
    )
    sb.set_animals([("m0", "black", (200, 0, 0)), ("m1", "green", (0, 200, 0))])

    sb._on_behavior_clicked("fighting")
    sb._on_animal_clicked("m0")  # actor
    sb._on_animal_clicked("m1")  # recipient
    sb._selected_animal = "m1"
    sb._label_selected_no_behavior()
    staged = [(l["behavior_id"], l["focal_animal_id"]) for l in sb._clip_labels]
    assert ("fighting", "m0") in staged   # the actor's label survives
    assert (NO_BEHAVIOR_ID, "m1") in staged

    # A mutual behavior genuinely labels the partner's segment, so negating that
    # animal must still drop it.
    sb._clip_labels = []
    sb._on_behavior_clicked("sniffing")
    sb._on_animal_clicked("m0")
    sb._on_animal_clicked("m1")
    sb._selected_animal = "m1"
    sb._label_selected_no_behavior()
    assert [l["behavior_id"] for l in sb._clip_labels] == [NO_BEHAVIOR_ID]


def test_co_occurring_row_is_positive_not_negative():
    """A pipe-joined row is a positive for each behavior it names.

    Regression: the uncertainty ensemble compared the whole label string to the
    target, so "fighting|rearing" trained as a *negative* for both fighting and
    rearing, the exact opposite of what the reviewer recorded.
    """
    train_df = pd.DataFrame({
        "label": ["fighting|rearing", "grooming", "no_behavior"],
        "session_id": ["s1", "s1", "s1"],
        "start_frame": [0, 100, 200],
        "end_frame": [8, 108, 208],
    })
    keep, y = ActiveLearningTab._build_binary_target_with_overlap_guard(train_df, "fighting")
    assert dict(zip(train_df["label"].iloc[keep], y)) == {
        "fighting|rearing": 1, "grooming": 0, "no_behavior": 0,
    }
    keep, y = ActiveLearningTab._build_binary_target_with_overlap_guard(train_df, "rearing")
    assert dict(zip(train_df["label"].iloc[keep], y))["fighting|rearing"] == 1

    # A pipe row that does not name the target is dropped, not counted against
    # it: the same clip must not be a positive for one behavior and a negative
    # for another it also contains.
    keep, y = ActiveLearningTab._build_binary_target_with_overlap_guard(train_df, "grooming")
    assert "fighting|rearing" not in set(train_df["label"].iloc[keep])


def test_end_to_end_pooling_join(svc, tmp_path):
    """Persist via ReviewService, join to segment features, verify pooling."""
    # Synthetic per-animal segment features (with a relational social_* column).
    rows = []
    for animal in ("track_0", "track_1"):
        for (s, e) in ((0, 29), (30, 59)):
            rows.append({
                "segment_id": f"seg_{animal}_{SESS}_{s}_{e}",
                "animal_id": animal, "session_id": SESS,
                "start_frame": s, "end_frame": e,
                "posture_speed_mean": 0.5,
                "social_dist_centroid_to_centroid_nearest_norm_mean": 0.3,
            })
    segment_df = pd.DataFrame(rows)

    rsvc = ReviewService()
    rsvc.set_project(tmp_path)

    def commit(labels, start, end):
        for spec in svc.aggregate_clip_labels(labels, SESS, start, end):
            rsvc.append_segment_label(ReviewerLabelRecord(
                segment_id=spec["segment_id"], review_label=spec["review_label"],
                reviewer_id="rev", notes="soundboard", **spec["fields"],
            ))

    # Window 0-29: m1 rears, m2 grooms. Window 30-59: m1 fights+sniffs m2, m2 rears.
    commit([{"behavior_id": "rearing", "focal_animal_id": "track_0", "partner_animal_id": None},
            {"behavior_id": "grooming", "focal_animal_id": "track_1", "partner_animal_id": None}], 0, 29)
    commit([{"behavior_id": "fighting", "focal_animal_id": "track_0", "partner_animal_id": "track_1"},
            {"behavior_id": "sniffing", "focal_animal_id": "track_0", "partner_animal_id": "track_1"},
            {"behavior_id": "rearing", "focal_animal_id": "track_1", "partner_animal_id": None}], 30, 59)

    labels_df = pd.read_parquet(tmp_path / "derived" / "review_labels" / "reviewer_labels.parquet")
    merged = ActiveLearningTab._aggregate_reviewer_labels(segment_df, labels_df)

    # Expand co-occurring labels the way the trainer does.
    exp = []
    for r in merged.itertuples():
        for sub in str(r.label).split("|"):
            exp.append((r.animal_id, sub))
    exp_df = pd.DataFrame(exp, columns=["animal_id", "behavior"])

    def animals(behavior):
        return set(exp_df[exp_df.behavior == behavior]["animal_id"])

    assert animals("rearing") == {"track_0", "track_1"}   # pooled across animals
    assert animals("sniffing") == {"track_0", "track_1"}  # mutual -> both
    assert animals("fighting") == {"track_0"}             # directed -> actor only
    assert animals("grooming") == {"track_1"}
    # Relational (Phase-3) feature column survives the label join.
    assert "social_dist_centroid_to_centroid_nearest_norm_mean" in merged.columns


def test_structured_label_store_round_trip(tmp_path):
    """The exact soundboard payload round-trips for revisiting/editing a clip."""
    rs = ReviewService()
    rs.set_project(tmp_path)
    wid = f"seg_track_0_{SESS}_0_29"
    payload = [
        {"behavior_id": "rearing", "focal_animal_id": "sub:track_0", "partner_animal_id": None},
        {"behavior_id": "sniffing", "focal_animal_id": "sub:track_0", "partner_animal_id": "sub:track_1"},
    ]
    rs.save_structured_labels(wid, payload)
    assert rs.get_structured_labels(wid) == payload

    # Editing replaces the stored set (no accumulation).
    rs.save_structured_labels(wid, [{"behavior_id": "grooming", "focal_animal_id": "sub:track_1", "partner_animal_id": None}])
    assert [l["behavior_id"] for l in rs.get_structured_labels(wid)] == ["grooming"]

    # Empty payload clears the entry.
    rs.save_structured_labels(wid, [])
    assert rs.get_structured_labels(wid) == []
    # Unknown window -> empty.
    assert rs.get_structured_labels("nope") == []


def test_recommit_replaces_segment_labels(tmp_path):
    """Re-committing an edited clip purges prior label rows instead of duplicating."""
    rs = ReviewService()
    rs.set_project(tmp_path)
    seg = f"seg_track_0_{SESS}_0_29"

    rs.append_segment_label(ReviewerLabelRecord(segment_id=seg, review_label="rearing", reviewer_id="rev"))
    labels = rs.load_segment_labels()
    assert [r.review_label for r in labels] == ["rearing"]

    # Edit: purge this segment's rows, then write the new label.
    rs.remove_segment_labels([seg])
    rs.append_segment_label(ReviewerLabelRecord(segment_id=seg, review_label="grooming", reviewer_id="rev"))
    labels = rs.load_segment_labels()
    assert [r.review_label for r in labels] == ["grooming"]  # replaced, not duplicated


def test_load_labels_rebuilds_chips_without_reemitting():
    """Soundboard.load_labels rebuilds display chips and does not re-emit structured labels."""
    from PySide6.QtWidgets import QApplication
    from abel.ui.behavior_soundboard import BehaviorSoundboard

    app = QApplication.instance() or QApplication([])
    emitted = []
    sb = BehaviorSoundboard()
    sb.configure(
        [("rearing", "Rearing", "r", False, "none"),
         ("sniffing", "Sniffing", "s", True, "mutual"),
         ("fighting", "Fighting", "f", True, "directed")],
        lambda b: None, {},
        on_structured=lambda *a: emitted.append(a),
    )
    sb.set_animals([("sub:track_0", "black", (200, 0, 0)), ("sub:track_1", "green", (0, 200, 0))])
    sb.load_labels([
        {"behavior_id": "rearing", "focal_animal_id": "sub:track_0", "partner_animal_id": None},
        {"behavior_id": "sniffing", "focal_animal_id": "sub:track_0", "partner_animal_id": "sub:track_1"},
        {"behavior_id": "fighting", "focal_animal_id": "sub:track_1", "partner_animal_id": "sub:track_0"},
    ])
    assert len(sb._clip_labels) == 3
    assert emitted == []  # load must not re-emit
    displays = [l["display"] for l in sb._clip_labels]
    assert displays[0] == "Rearing: black"
    assert "⇄" in displays[1]  # mutual arrow, both names
    assert "→" in displays[2]  # directed arrow
    # Loaded labels feed the next commit payload.
    captured = []
    sb._on_commit = lambda L: captured.append(L)
    sb._commit()
    assert len(captured) == 1 and len(captured[0]) == 3


def test_commit_does_not_wipe_auto_advanced_labels():
    """After commit auto-advances and repopulates, the next clip's labels survive."""
    from PySide6.QtWidgets import QApplication
    from abel.ui.behavior_soundboard import BehaviorSoundboard

    app = QApplication.instance() or QApplication([])
    sb = BehaviorSoundboard()
    next_clip = [{"behavior_id": "rearing", "focal_animal_id": "sub:track_1", "partner_animal_id": None}]

    def on_commit(_payload):
        # Simulate the review tab auto-advancing to the next clip and loading
        # that clip's previously-committed labels back into the soundboard.
        sb.load_labels(next_clip)

    sb.configure(
        [("rearing", "Rearing", "r", False, "none"), ("sniffing", "Sniffing", "s", True, "mutual")],
        lambda b: None, {}, on_commit=on_commit,
    )
    sb.set_animals([("sub:track_0", "black", (200, 0, 0)), ("sub:track_1", "green", (0, 200, 0))])
    sb._on_animal_clicked("sub:track_0")
    sb._on_behavior_clicked("rearing")
    assert len(sb._clip_labels) == 1
    sb._commit()
    # The clear must happen before the callback, so the auto-loaded next-clip
    # labels are not wiped by the post-commit clear.
    assert len(sb._clip_labels) == 1
    assert sb._clip_labels[0]["focal_animal_id"] == "sub:track_1"


def test_review_label_display_is_animal_aware(svc):
    """The review-tab 'Active labels' tags name the mouse and social direction."""
    import re
    from abel.ui.tabs.review_tab import ReviewTab

    name_by_id = {"sub:track_0": "black female", "sub:track_1": "green female"}
    structured = [
        {"behavior_id": "rearing", "focal_animal_id": "sub:track_0", "partner_animal_id": None},
        {"behavior_id": "fighting", "focal_animal_id": "sub:track_0", "partner_animal_id": "sub:track_1"},
        {"behavior_id": "sniffing", "focal_animal_id": "sub:track_1", "partner_animal_id": "sub:track_0"},
    ]
    tags = ReviewTab._format_structured_tags(structured, name_by_id, svc)
    plain = [re.sub("<[^>]+>", "", t) for t in tags]
    assert plain[0] == "Rearing · black female"                       # solo -> animal named
    assert plain[1] == "Fighting · black female → green female"       # directed -> actor → recipient
    assert plain[2] == "Sniffing · green female ⇄ black female"       # mutual -> pair
    # Unknown behavior/animal degrade gracefully rather than crash.
    fallback = ReviewTab._format_structured_tags(
        [{"behavior_id": "???", "focal_animal_id": "ghost", "partner_animal_id": None}], {}, svc,
    )
    assert re.sub("<[^>]+>", "", fallback[0]) == "??? · ghost"


def test_saved_labels_row_renders_every_subject(svc, monkeypatch):
    """The review panel's "Saved labels" row shows one tag per subject.

    Regression: the panel only had the single-behavior "Review label" combo, so a
    two-mouse clip labeled "Fighting m0 -> m1" + "Rearing m1" displayed as a bare
    "Fighting", the second subject's behavior was invisible even though both rows
    were saved.
    """
    import re
    from types import SimpleNamespace

    from abel.ui.tabs.review_tab import ReviewTab

    cand = SimpleNamespace(
        window_id=f"seg_m0_{SESS}_0_29", session_id=SESS, start_frame=0, end_frame=29,
    )
    structured = [
        {"behavior_id": "fighting", "focal_animal_id": "m0", "partner_animal_id": "m1"},
        {"behavior_id": "rearing", "focal_animal_id": "m1", "partner_animal_id": None},
    ]
    tab = SimpleNamespace(
        _current_candidate_idx=0,
        _visible_candidates=[cand],
        _behavior_service=svc,
        _review_service=SimpleNamespace(get_structured_labels=lambda wid: structured),
        _decision_by_clip_id={},
        _saved_labels_display=SimpleNamespace(
            setText=lambda t: setattr(tab, "_text", t), setStyleSheet=lambda s: None,
        ),
        _clip_animals_for=lambda c: [("m0", "black", (0, 0, 0)), ("m1", "green", (0, 0, 0))],
        _normalize_behavior_id=lambda b: b,
        _decision_to_review_label=ReviewTab._decision_to_review_label,
    )
    tab._saved_labels_for = lambda c: ReviewTab._saved_labels_for(tab, c)
    tab._format_structured_tags = ReviewTab._format_structured_tags
    ReviewTab._update_saved_labels_display(tab)
    plain = re.sub("<[^>]+>", "", tab._text)
    assert "Fighting · black → green" in plain     # actor -> recipient, not a bare name
    assert "Rearing · green" in plain              # the other mouse is visible too
    assert "2 subjects labeled" in plain

    # A multi-animal clip with only a clip-level (combo) label says so, instead of
    # letting one behavior stand for both mice.
    tab._review_service = SimpleNamespace(get_structured_labels=lambda wid: [])
    tab._decision_by_clip_id = {
        cand.window_id: SimpleNamespace(
            decision=ReviewDecisionType.ACCEPT, behavior_label="fighting",
        )
    }
    ReviewTab._update_saved_labels_display(tab)
    plain = re.sub("<[^>]+>", "", tab._text)
    assert plain.startswith("Fighting")
    assert "· None" not in plain                   # no subject -> no bogus animal name
    assert "whole clip" in plain

    # Nothing saved yet -> the row points at the soundboard for per-subject labels.
    tab._decision_by_clip_id = {}
    ReviewTab._update_saved_labels_display(tab)
    assert "soundboard" in re.sub("<[^>]+>", "", tab._text)
