"""False-positive rejections are negatives for the rejected behavior only."""

from __future__ import annotations

import json

import pandas as pd

from abel.services.false_positive_label_repair import FalsePositiveLabelRepair
from abel.services.review_service import reject_review_label

A, B = "beh_a", "beh_b"


def test_reject_label_is_behavior_specific():
    assert reject_review_label(A) == f"not_{A}"
    assert reject_review_label("no_behavior") == "no_behavior"
    assert reject_review_label("") == "no_behavior"
    assert reject_review_label(None) == "no_behavior"
    assert reject_review_label(f"{A}|{B}") == "no_behavior"
    assert reject_review_label(f"not_{A}") == "no_behavior"


def _project(tmp_path):
    (tmp_path / "derived" / "review_tables").mkdir(parents=True)
    (tmp_path / "derived" / "review_labels").mkdir(parents=True)
    (tmp_path / "derived" / "training_sets").mkdir(parents=True)
    decisions = [
        {"clip_id": "s1", "decision": "reject", "behavior_label": A},          # FP of A -> fix
        {"clip_id": "s2", "decision": "reject", "behavior_label": "no_behavior"},  # true negative
        {"clip_id": "s3", "decision": "accept", "behavior_label": "no_behavior"},
        {"clip_id": "s4", "decision": "reject", "behavior_label": B},
        {"clip_id": "s4", "decision": "accept", "behavior_label": B},           # later accept wins
        {"clip_id": "s6", "decision": "reject", "behavior_label": "deleted_beh"},
    ]
    (tmp_path / "derived/review_tables/review_decisions.json").write_text(json.dumps({"decisions": decisions}))
    labels = pd.DataFrame({
        "segment_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
        "review_label": ["no_behavior", "no_behavior", "no_behavior", B, "no_behavior", "no_behavior"],
        "notes": ["", "", "", "", f"auto:no_behavior:{B}", ""],
    })
    labels.to_parquet(tmp_path / "derived/review_labels/reviewer_labels.parquet", index=False)
    ts = pd.DataFrame({"segment_id": ["s1", "s2", "s4", "s5"], "label": ["no_behavior", "no_behavior", B, "no_behavior"], "f": [1, 2, 3, 4]})
    ts.to_parquet(tmp_path / "derived/training_sets/training_set.parquet", index=False)
    return tmp_path


def test_repair_relabels_only_false_positives(tmp_path):
    root = _project(tmp_path)
    repair = FalsePositiveLabelRepair(root, ["no_behavior", A, B])
    plan = repair.plan()
    assert plan.relabel == {"s1": f"not_{A}", "s5": f"not_{B}"}
    assert (plan.reviewer_label_rows, plan.training_set_rows) == (2, 2)
    assert not plan.applied

    done = repair.apply()
    assert done.applied and done.backup_dir.is_dir()
    lbl = pd.read_parquet(root / "derived/review_labels/reviewer_labels.parquet").set_index("segment_id").review_label
    assert lbl.to_dict() == {"s1": f"not_{A}", "s2": "no_behavior", "s3": "no_behavior", "s4": B, "s5": f"not_{B}", "s6": "no_behavior"}
    ts = pd.read_parquet(root / "derived/training_sets/training_set.parquet").set_index("segment_id").label
    assert ts.to_dict() == {"s1": f"not_{A}", "s2": "no_behavior", "s4": B, "s5": f"not_{B}"}

    # Idempotent: nothing left to repair.
    assert not FalsePositiveLabelRepair(root, [A, B]).plan().needed
