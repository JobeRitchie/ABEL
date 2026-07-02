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

from abel.models.schemas import BehaviorDefinition, ReviewerLabelRecord
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
    # Merged multi-behavior record drops per-behavior structured fields.
    merged = next(r for r in recs if r["review_label"] == "fighting|sniffing")
    assert merged["fields"] == {}


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
