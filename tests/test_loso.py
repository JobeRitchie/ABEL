"""Leave-one-subject-out CV aggregation tests.

Training is mocked (real per-fold training needs the full feature pipeline and
GPU): we monkeypatch ``run_one_config`` to return synthetic held-out scores, then
assert the LOSO loop holds out each subject in turn, pools every fold's held-out
predictions, and computes the shared raw+refined metrics correctly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import abel.validation.loso as loso
from abel.validation.datamodel import ConfigEvalResult, ProjectRef

TARGET = "appr-id"


def _project(tmp_path: Path) -> ProjectRef:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    return ProjectRef(
        project_id="p", name="p", root=tmp_path,
        behavior_names={TARGET: "Approach"},
    )


def _training_df() -> pd.DataFrame:
    rows = []
    # 4 subjects, each with 3 target positives + 7 negatives; contiguous frames.
    for subj in ("MS1", "MS2", "MS3", "MS4"):
        for k in range(10):
            label = TARGET if k < 3 else "no_behavior"
            start = k * 15
            rows.append({
                "segment_id": f"seg_{subj}_session_{subj}_{start}_{start + 14}",
                "animal_id": subj,
                "session_id": f"session_{subj}",
                "label": label,
                "label_source": "reviewer",
            })
    # Two temporal-feedback rows that must never be evaluated on.
    for k, subj in enumerate(("MS1", "MS2")):
        rows.append({
            "segment_id": f"seg_feedback_session_fb_{k}_{k + 14}",
            "animal_id": "feedback", "session_id": "session_fb",
            "label": "no_behavior", "label_source": "temporal_feedback",
        })
    return pd.DataFrame(rows)


def _fake_run_one_config(strong: bool):
    """Return a per-fold result builder. strong=True -> positives score high."""

    def _run(trainer, project, target, pool, holdout, *, seed, retain_estimator=False):
        y_true = (holdout["label"].astype(str) == str(target)).astype(int).to_numpy()
        rng = np.random.default_rng(0)
        if strong:
            score = np.where(y_true == 1, rng.uniform(0.7, 0.95, len(y_true)),
                             rng.uniform(0.02, 0.3, len(y_true)))
        else:  # weak: positives never cross 0.5 (the Approach-on-held-out case)
            score = np.where(y_true == 1, rng.uniform(0.2, 0.45, len(y_true)),
                             rng.uniform(0.02, 0.3, len(y_true)))
        pred = (score >= 0.5).astype(int)
        return ConfigEvalResult(
            project_id="p", behavior_id=str(target),
            n_pos_train=0, n_neg_train=0, n_features=1,
            y_true=y_true, y_score=score, y_pred=pred,
            tp=int(((y_true == 1) & (pred == 1)).sum()),
            fp=int(((y_true == 0) & (pred == 1)).sum()),
            fn=int(((y_true == 1) & (pred == 0)).sum()),
            val_meta=holdout[["segment_id", "session_id"]].reset_index(drop=True),
            degenerate=False,
        )

    return _run


def test_loso_holds_out_each_subject_and_pools(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loso, "run_one_config", _fake_run_one_config(strong=True))
    res = loso.leave_one_subject_out(_project(tmp_path), TARGET, df=_training_df())

    assert res.get("error") is None
    assert res["n_subjects"] == 4                       # one fold per subject
    assert res["method"] == "leave_one_subject_out"
    # 4 subjects x 3 positives = 12 held-out positives pooled, all caught (strong).
    assert res["raw_tp"] == 12
    assert res["raw_fn"] == 0
    assert res["raw_f1"] > 0.9
    # temporal_feedback rows never entered any holdout (never scored).
    assert all(f.get("subject") != "feedback" for f in res["folds"])


def test_loso_weak_model_reports_low_pooled_score(tmp_path, monkeypatch) -> None:
    # Positives never exceed 0.5 -> pooled raw catches none (the honest Approach story).
    monkeypatch.setattr(loso, "run_one_config", _fake_run_one_config(strong=False))
    res = loso.leave_one_subject_out(_project(tmp_path), TARGET, df=_training_df())
    assert res["raw_tp"] == 0
    assert res["raw_fn"] == 12


def test_loso_requires_two_subjects(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loso, "run_one_config", _fake_run_one_config(strong=True))
    df = _training_df()
    df = df[df["animal_id"].isin(["MS1", "feedback"])]  # only one real subject
    res = loso.leave_one_subject_out(_project(tmp_path), TARGET, df=df)
    assert "error" in res and "subject" in res["error"]


def test_loso_reports_per_fold_prauc_and_sem(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loso, "run_one_config", _fake_run_one_config(strong=True))
    res = loso.leave_one_subject_out(_project(tmp_path), TARGET, df=_training_df())

    # New aggregate keys the publication figure depends on.
    for key in ("fold_f1_mean", "fold_f1_sem", "fold_prauc_mean",
                "fold_prauc_std", "fold_prauc_sem"):
        assert key in res, f"missing {key}"

    # Strong model: high per-fold PR-AUC, SEM finite and non-negative.
    assert res["fold_prauc_mean"] > 0.9
    assert res["fold_prauc_sem"] >= 0.0
    assert res["fold_f1_sem"] >= 0.0
    # Every scored fold carries its own PR-AUC.
    scored = [f for f in res["folds"] if "pr_auc" in f]
    assert len(scored) == 4
    assert all(0.0 <= f["pr_auc"] <= 1.0 for f in scored)


def test_mean_std_sem_matches_formula() -> None:
    mean, std, sem = loso._mean_std_sem([0.2, 0.4, 0.6, 0.8])
    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(np.std([0.2, 0.4, 0.6, 0.8], ddof=1))
    assert sem == pytest.approx(std / np.sqrt(4))
    # NaNs are dropped; a single finite value gives 0 spread.
    m, s, e = loso._mean_std_sem([float("nan"), 0.7])
    assert m == pytest.approx(0.7) and s == 0.0 and e == 0.0
