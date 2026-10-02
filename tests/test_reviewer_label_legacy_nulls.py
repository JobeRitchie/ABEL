"""Reviewer labels saved before the multi-animal fields existed must still load."""

from __future__ import annotations

import pandas as pd

from abel.models.schemas import ReviewerLabelRecord
from abel.services.review_service import ReviewService


def _legacy_row(**overrides):
    row = {
        "segment_id": "seg_a_session_x_0_14",
        "review_label": "rear",
        "reviewer_id": "r1",
        "confidence": 1.0,
        "notes": "",
        "timestamp": "2026-01-01T00:00:00",
        "focal_animal_id": None,
        "partner_animal_id": None,
        "social_role": None,
    }
    row.update(overrides)
    return row


def test_null_social_role_and_notes_are_filled():
    rec = ReviewerLabelRecord.model_validate(
        _legacy_row(notes=None, focal_animal_id=float("nan"))
    )
    assert rec.social_role == "none"
    assert rec.notes == ""
    assert rec.focal_animal_id is None


def test_load_segment_labels_keeps_legacy_rows(tmp_path):
    labels_dir = tmp_path / "derived" / "review_labels"
    labels_dir.mkdir(parents=True)
    pd.DataFrame(
        [_legacy_row(), _legacy_row(segment_id="seg_b_session_x_14_28", social_role="none")]
    ).to_parquet(labels_dir / "reviewer_labels.parquet", index=False)

    svc = ReviewService()
    svc.set_project(tmp_path)
    assert len(svc.load_segment_labels()) == 2
