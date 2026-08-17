"""Regression: temporal-review feedback must honour the training session scope.

Bug (found on TMT_NewCam): a user unticks held-out sessions in the training
selector, but any of those sessions that were reviewed in the Temporal Review tab
have false-positive/false-negative intervals saved to
``feedback_intervals.json``.  ``_load_training_frame`` scopes the labeled rows
correctly, then injects hard-negative / hard-positive rows from
``segment_features.parquet`` for *every* session named in the feedback file —
with no session-scope check.  That silently reintroduced 4 held-out mice into the
Dig model, leaking exactly the sessions being validated (the model was tuned by
the reviewer's own FP corrections on those mice).

The fix filters ``fp_map``/``fn_map`` to ``session_ids`` before any relabel or
injection runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from abel.services.active_learning_trainer_service import (
    ActiveLearningTrainerService,
    TrainingConfig,
)

TARGET = "beh"


def _build_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "derived" / "training_sets").mkdir(parents=True)
    (root / "derived" / "representations").mkdir(parents=True)
    (root / "derived" / "temporal_refinement" / TARGET).mkdir(parents=True)

    # Labeled corpus: one ticked (training) session, one unticked (held-out) one.
    # Both carry reviewer labels; the held-out session's labels are what the scope
    # filter should drop — the injection path is what this test really exercises.
    rows = []
    for sess in ("s_train", "s_hold"):
        for i in range(6):
            rows.append({
                "session_id": sess, "animal_id": sess,
                "segment_id": f"{sess}_seg{i}",
                "start_frame": i * 10, "end_frame": i * 10 + 9,
                "label": TARGET if i % 2 == 0 else "no_behavior",
                "label_source": "reviewer", "reviewer_confidence": 1.0,
                "feat": float(i),
            })
    pd.DataFrame(rows).to_parquet(
        root / "derived" / "training_sets" / "training_set.parquet"
    )

    # segment_features holds candidate segments for BOTH sessions in the FP window
    # (frames 900-909) — the injection would pull the held-out one if unguarded.
    feats = []
    for sess in ("s_train", "s_hold"):
        feats.append({
            "session_id": sess, "segment_id": f"{sess}_fp",
            "start_frame": 900, "end_frame": 909, "feat": 42.0,
        })
    pd.DataFrame(feats).to_parquet(
        root / "derived" / "representations" / "segment_features.parquet"
    )

    # The reviewer flagged an FP interval in BOTH sessions.
    fb = {
        "false_positive_intervals_by_session": {
            "s_train": [[900, 909]],
            "s_hold": [[900, 909]],
        },
        "false_negative_intervals_by_session": {},
    }
    (root / "derived" / "temporal_refinement" / TARGET / "feedback_intervals.json").write_text(
        json.dumps(fb)
    )
    return root


def test_feedback_from_unticked_session_is_excluded(tmp_path):
    root = _build_project(tmp_path)
    svc = ActiveLearningTrainerService()
    cfg = TrainingConfig(target_label=TARGET, include_imported=False)

    df = svc._load_training_frame(root, cfg, session_ids={"s_train"}, _log=lambda *_: None)

    # No data from the unticked session by ANY path (labeled rows or injected).
    assert (df["session_id"].astype(str) == "s_hold").sum() == 0
    # The ticked session's own FP injection still works (guard is scoped, not off).
    assert (df["segment_id"].astype(str) == "s_train_fp").sum() == 1


def test_feedback_applies_when_session_is_ticked(tmp_path):
    """Sanity: with the held-out session ticked back in, its injection returns."""
    root = _build_project(tmp_path)
    svc = ActiveLearningTrainerService()
    cfg = TrainingConfig(target_label=TARGET, include_imported=False)

    df = svc._load_training_frame(
        root, cfg, session_ids={"s_train", "s_hold"}, _log=lambda *_: None
    )
    assert (df["segment_id"].astype(str) == "s_hold_fp").sum() == 1
