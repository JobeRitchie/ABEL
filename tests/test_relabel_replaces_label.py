"""Relabeling a clip must leave one label, not a stale pair that trains as ambiguous."""
from __future__ import annotations

import pandas as pd

from abel.models.schemas import ReviewerLabelRecord
from abel.services.review_service import ReviewService
from abel.ui.tabs.active_learning_tab import ActiveLearningTab, _latest_label_per_reviewer


def _svc(tmp_path):
    svc = ReviewService()
    svc.set_project(tmp_path)
    return svc


def _labels(tmp_path):
    return pd.read_parquet(tmp_path / "derived" / "review_labels" / "reviewer_labels.parquet")


def test_upsert_replaces_same_reviewers_label(tmp_path):
    svc = _svc(tmp_path)
    svc.upsert_segment_label(ReviewerLabelRecord(segment_id="clip1", review_label="escape", reviewer_id="me"))
    svc.upsert_segment_label(ReviewerLabelRecord(segment_id="clip2", review_label="groom", reviewer_id="me"))
    svc.upsert_segment_label(ReviewerLabelRecord(segment_id="clip1", review_label="rear", reviewer_id="me"))
    df = _labels(tmp_path)
    assert df[df.segment_id == "clip1"].review_label.tolist() == ["rear"]
    assert df[df.segment_id == "clip2"].review_label.tolist() == ["groom"]


def test_upsert_keeps_other_reviewers_label(tmp_path):
    svc = _svc(tmp_path)
    svc.upsert_segment_label(ReviewerLabelRecord(segment_id="clip1", review_label="escape", reviewer_id="a"))
    svc.upsert_segment_label(ReviewerLabelRecord(segment_id="clip1", review_label="rear", reviewer_id="b"))
    assert sorted(_labels(tmp_path).review_label) == ["escape", "rear"]


def test_training_set_uses_latest_label_for_existing_double_rows():
    seg = pd.DataFrame({"segment_id": ["clip1", "clip2"], "session_id": ["s", "s"], "f0": [1.0, 2.0]})
    lbl = pd.DataFrame({
        "segment_id": ["clip1", "clip1", "clip2", "clip2"],
        "review_label": ["escape", "rear", "groom", "rear"],
        "reviewer_id": ["me", "me", "a", "b"],
        "confidence": [1.0] * 4,
        "timestamp": ["2026-11-19T15:40:06", "2026-11-19T15:40:07", "2026-11-19T15:40:00", "2026-11-19T15:41:00"],
    })
    out = ActiveLearningTab._aggregate_reviewer_labels(seg, lbl).set_index("segment_id")
    assert out.loc["clip1", "label"] == "rear"
    # Two different reviewers disagreeing is still ambiguous.
    assert out.loc["clip2", "label"] == "ambiguous"


def test_latest_label_orders_by_timestamp_and_keeps_feedback():
    lbl = pd.DataFrame({
        "segment_id": ["c", "c", "t", "t"],
        "review_label": ["rear", "escape", "x", "y"],
        "reviewer_id": ["me", "me", "temporal_feedback", "temporal_feedback"],
        "timestamp": ["2026-11-19T10:00:05", "2026-11-19T10:00:01", "2026-11-19T10:00:00", "2026-11-19T10:00:00"],
    })
    out = _latest_label_per_reviewer(lbl)
    assert out[out.segment_id == "c"].review_label.tolist() == ["rear"]
    assert sorted(out[out.segment_id == "t"].review_label) == ["x", "y"]
