"""Regression tests for the group-holdout guarantee.

These pin two defects found against real projects (DG_FearConditioning, DG_EPM):

1. ``train_and_evaluate`` moves ``temporal_feedback`` / ``imported:*`` rows OUT of
   the validation split and INTO training — *after* a ``precomputed_split`` is
   applied. Any such row left in the holdout frame therefore becomes a training
   row belonging to a held-out session. On real data this leaked rows from 2 FC
   and 2 EPM sessions into training while other rows of those same sessions were
   still being scored.
2. ``TrainingConfig.include_imported`` is only consumed by ``_load_training_frame``,
   which ``train_and_evaluate`` never calls — so the engine's ``include_imported=
   False`` was a no-op and 51% of DG_FearConditioning's training pool was clips
   imported from another project, counted as this project's labeling effort.
"""

from __future__ import annotations

import pandas as pd
import pytest

from abel.validation import holdout
from abel.validation.datamodel import ProjectRef


def _frame() -> pd.DataFrame:
    """Two sessions; each carries reviewed rows plus refine-only rows."""
    rows = []
    for sess in ("s_train", "s_hold"):
        for i in range(6):
            rows.append({
                "session_id": sess, "animal_id": sess,
                "segment_id": f"seg_{sess}_{i}_{i * 10}_{i * 10 + 9}",
                "label": "beh" if i % 2 == 0 else "no_behavior",
                "label_source": "manual",
                "reviewer_confidence": 1.0,
                "feat": float(i),
            })
        # Rows the trainer will relocate into training if we leave them in holdout.
        rows.append({
            "session_id": sess, "animal_id": sess,
            "segment_id": f"seg_{sess}_fb_900_909",
            "label": "beh", "label_source": "temporal_feedback",
            "reviewer_confidence": 1.0, "feat": 9.0,
        })
        rows.append({
            "session_id": sess, "animal_id": sess,
            "segment_id": f"seg_{sess}_imp_950_959",
            "label": "beh", "label_source": "imported:OtherProject",
            "reviewer_confidence": 1.0, "feat": 9.5,
        })
    return pd.DataFrame(rows)


def _project(tmp_path) -> ProjectRef:
    return ProjectRef(
        project_id="P", name="P", root=tmp_path,
        split_strategy="group_shuffle_session",
        behavior_names={"beh": "Beh", "no_behavior": "No Behavior"},
    )


def test_refine_only_rows_never_reach_the_holdout_frame(tmp_path):
    """They cannot be scored, and if left in they would be TRAINED on."""
    sp = holdout.split(_project(tmp_path), df=_frame(), holdout_groups=["s_hold"])

    assert holdout.is_refine_only(sp.holdout).sum() == 0
    # Both the temporal_feedback and the imported row of the held-out session go.
    assert sp.n_refine_only_dropped == 2
    # And they must not have been quietly rehomed into the training pool either —
    # a held-out session contributes NOTHING.
    assert (sp.train_pool["session_id"] == "s_hold").sum() == 0


def test_imported_rows_are_kept_out_of_the_training_pool(tmp_path):
    """include_imported=False is a no-op in the trainer, so enforce it at the split."""
    sp = holdout.split(_project(tmp_path), df=_frame(), holdout_groups=["s_hold"])
    assert holdout.is_imported(sp.train_pool).sum() == 0
    assert sp.n_imported_dropped == 1  # the training session's imported row
    # temporal_feedback in a TRAINING session is legitimate training data — keep it.
    assert (sp.train_pool["label_source"] == "temporal_feedback").sum() == 1

    # Opt out and the imported row stays (so the old behavior remains reachable).
    sp2 = holdout.split(_project(tmp_path), df=_frame(), holdout_groups=["s_hold"],
                        exclude_imported=False)
    assert holdout.is_imported(sp2.train_pool).sum() == 1


def test_leakage_guard_rejects_a_smuggled_refine_row(tmp_path):
    """The guard must catch refine-only rows, not just group/segment overlap.

    Asserting on group overlap alone gave false assurance: these rows look innocent
    but the trainer relocates them into the training split.
    """
    df = _frame()
    train_pool = df[df.session_id == "s_train"].reset_index(drop=True)
    bad_holdout = df[df.session_id == "s_hold"].reset_index(drop=True)  # still has refine rows
    with pytest.raises(AssertionError, match="refine-only"):
        holdout._assert_no_leakage(train_pool, bad_holdout, "session_id")


def test_manifest_does_not_advertise_unevaluable_groups(tmp_path):
    """A group made entirely of refine-only rows has nothing to score — say so."""
    df = _frame()
    # Make the held-out session refine-only end to end.
    df.loc[df.session_id == "s_hold", "label_source"] = "temporal_feedback"
    sp = holdout.split(_project(tmp_path), df=df, holdout_groups=["s_hold"])

    assert len(sp.holdout) == 0
    assert sp.holdout_groups == []                 # not advertised as held out
    assert sp.unevaluable_groups == ["s_hold"]     # reported instead
    man = sp.manifest(_project(tmp_path))
    assert man["unevaluable_holdout_groups"] == ["s_hold"]
    assert man["n_holdout_rows"] == 0
