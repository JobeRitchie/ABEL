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


def _training_df(n_rows: int = 10, n_pos: int = 3, pos_by_subject: dict | None = None) -> pd.DataFrame:
    rows = []
    # 4 subjects, each with 3 target positives + 7 negatives; contiguous frames.
    for subj in ("MS1", "MS2", "MS3", "MS4"):
        k_pos = n_pos if pos_by_subject is None else pos_by_subject.get(subj, n_pos)
        for k in range(n_rows):
            label = TARGET if k < k_pos else "no_behavior"
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


def _fake_run_one_config_by_subject(strong_subjects: set[str]):
    """Per-fold builder where only *strong_subjects* are detected.

    Gives the pooled score real subject-to-subject variance, which a bootstrap
    over subjects needs in order to produce a non-degenerate interval.
    """
    strong = _fake_run_one_config(strong=True)
    weak = _fake_run_one_config(strong=False)

    def _run(trainer, project, target, pool, holdout, *, seed, retain_estimator=False):
        subj = str(holdout["animal_id"].iloc[0])
        inner = strong if subj in strong_subjects else weak
        return inner(trainer, project, target, pool, holdout,
                     seed=seed, retain_estimator=retain_estimator)

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


def test_available_subjects_reports_counts(tmp_path) -> None:
    subs = loso.available_subjects(_project(tmp_path), df=_training_df())
    assert [s["subject"] for s in subs] == ["MS1", "MS2", "MS3", "MS4"]
    ms1 = subs[0]
    assert ms1["n_windows"] == 10 and ms1["n_labeled"] == 3 and ms1["n_sessions"] == 1
    # The refinement-only rows are not a selectable subject.
    assert all(s["subject"] != "feedback" for s in subs)


def test_loso_restricted_to_selected_subjects(tmp_path, monkeypatch) -> None:
    seen: list[set[str]] = []
    inner = _fake_run_one_config(strong=True)

    def _spy(trainer, project, target, pool, holdout, *, seed, retain_estimator=False):
        seen.append(set(pool["animal_id"].astype(str)))
        return inner(trainer, project, target, pool, holdout,
                     seed=seed, retain_estimator=retain_estimator)

    monkeypatch.setattr(loso, "run_one_config", _spy)
    res = loso.leave_one_subject_out(
        _project(tmp_path), TARGET, df=_training_df(), subjects=["MS1", "MS2", "MS3"]
    )

    assert res["n_subjects"] == 3
    assert res["subjects"] == ["MS1", "MS2", "MS3"]
    assert res["excluded_subjects"] == ["MS4"]
    # 3 subjects x 3 positives pooled — MS4 never scored.
    assert res["raw_tp"] == 9
    assert all(f.get("subject") != "MS4" for f in res["folds"])
    # An excluded mouse is out of every fold's TRAINING pool too; refinement-only
    # rows stay in.
    assert seen and all("MS4" not in pool for pool in seen)
    assert all("feedback" in pool for pool in seen)


def test_loso_rejects_selection_below_two_subjects(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loso, "run_one_config", _fake_run_one_config(strong=True))
    res = loso.leave_one_subject_out(
        _project(tmp_path), TARGET, df=_training_df(), subjects=["MS1"]
    )
    assert "selected 1" in res["error"]


def test_loso_ignores_unknown_selected_subjects(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loso, "run_one_config", _fake_run_one_config(strong=True))
    res = loso.leave_one_subject_out(
        _project(tmp_path), TARGET, df=_training_df(), subjects=["MS1", "MS2", "NOPE"]
    )
    assert res.get("error") is None
    assert res["subjects"] == ["MS1", "MS2"]


def test_bootstrap_ci_brackets_the_point_estimate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        loso, "run_one_config", _fake_run_one_config_by_subject({"MS1", "MS2"})
    )
    res = loso.leave_one_subject_out(
        _project(tmp_path), TARGET, df=_training_df(), n_boot=500
    )

    assert res["boot_n_reps"] == 500
    assert res["boot_seed"] == 42
    # Half the mice are detected, half are missed -> a real interval, not a point.
    assert res["boot_f1_target_lo"] < res["boot_f1_target_hi"]
    assert res["boot_f1_target_lo"] <= res["pooled_f1_target"] <= res["boot_f1_target_hi"]
    assert res["boot_prauc_lo"] <= res["pooled_prauc"] <= res["boot_prauc_hi"]
    assert 0.0 <= res["boot_f1_target_lo"] and res["boot_f1_target_hi"] <= 1.0


def test_bootstrap_is_reproducible_under_a_fixed_seed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        loso, "run_one_config", _fake_run_one_config_by_subject({"MS1", "MS2"})
    )
    keys = ("boot_f1_target_lo", "boot_f1_target_hi", "boot_prauc_lo", "boot_prauc_hi")
    a = loso.leave_one_subject_out(_project(tmp_path), TARGET, df=_training_df(), n_boot=300)
    b = loso.leave_one_subject_out(_project(tmp_path), TARGET, df=_training_df(), n_boot=300)
    assert [a[k] for k in keys] == [b[k] for k in keys]


def test_bootstrap_ci_responds_to_seed_and_to_subject_count() -> None:
    """Seed changes the draw; more subjects tighten the interval.

    Driven straight through the estimator: with only the 4 mice of the fixture
    the 2.5/97.5 percentiles land on the same few discrete values for every
    seed, which hides both effects.
    """
    rng = np.random.default_rng(0)
    ys, ps = [], []
    # Detection quality varies mouse to mouse on a fixed ladder, so the cohort
    # carries spread and any subset of it spans the same range.
    for skill in np.linspace(0.3, 0.9, 12):
        y = np.array([1] * 3 + [0] * 17)
        p = np.where(y == 1, rng.uniform(0.2, 0.2 + skill, 20), rng.uniform(0.0, 0.3, 20))
        ys.append(y); ps.append(p)

    a = loso._bootstrap_subject_ci(ys, ps, n_reps=400, seed=1)
    b = loso._bootstrap_subject_ci(ys, ps, n_reps=400, seed=1)
    c = loso._bootstrap_subject_ci(ys, ps, n_reps=400, seed=2)
    assert a == b                                            # same seed, same interval
    assert a["boot_prauc_lo"] != c["boot_prauc_lo"]          # different seed, different draw

    # Four mice spanning the same skill range -> a visibly wider interval.
    idx = [0, 3, 7, 11]
    wide = loso._bootstrap_subject_ci(
        [ys[i] for i in idx], [ps[i] for i in idx], n_reps=400, seed=1
    )
    assert (a["boot_prauc_hi"] - a["boot_prauc_lo"]) < (
        wide["boot_prauc_hi"] - wide["boot_prauc_lo"]
    )


def test_bootstrap_ci_is_degenerate_below_two_subjects() -> None:
    y = [np.array([1, 0, 0, 1])]
    p = [np.array([0.9, 0.1, 0.2, 0.8])]
    out = loso._bootstrap_subject_ci(y, p, n_reps=100, seed=42)
    assert np.isnan(out["boot_f1_target_lo"]) and np.isnan(out["boot_prauc_hi"])
    assert out["boot_n_subjects"] == 1


def test_never_fires_detector_scores_zero_on_target_but_half_on_macro(
    tmp_path, monkeypatch
) -> None:
    """The regression test for the macro-F1 floor.

    At ABEL's real prevalence a detector that never fires still earns macro
    F1 ~= 0.5 from the "not this behavior" class alone. Target-class F1 must
    report it as the zero it is.
    """
    monkeypatch.setattr(loso, "run_one_config", _fake_run_one_config(strong=False))
    # 1 positive in 25 rows per mouse = 4 % prevalence, ABEL's operating range.
    df = _training_df(n_rows=25, n_pos=1)
    res = loso.leave_one_subject_out(_project(tmp_path), TARGET, df=df, n_boot=200)

    assert res["raw_tp"] == 0                       # fires on nothing
    assert res["pooled_f1_target"] == 0.0
    assert res["pooled_recall_target"] == 0.0
    assert res["raw_f1"] > 0.45                     # macro's floor, from the negatives
    assert all(f["f1_target"] == 0.0 for f in res["folds"] if "f1_target" in f)
    assert all(f["f1"] > 0.45 for f in res["folds"] if "f1" in f)
    # The invalid error bar is flagged, not silently offered.
    assert res["fold_sem_valid"] is False


def test_skipped_folds_are_counted_and_explained(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loso, "run_one_config", _fake_run_one_config(strong=True))
    # MS4 has no target positives -> its fold cannot be scored.
    df = _training_df(pos_by_subject={"MS4": 0})
    res = loso.leave_one_subject_out(_project(tmp_path), TARGET, df=df, n_boot=200)

    assert res["n_folds_total"] == 4
    assert res["n_folds_scored"] == 3
    assert res["n_folds_skipped"] == 1
    assert res["skip_reasons"] == {"no target positives in holdout": 1}
    assert res["subjects"] == ["MS1", "MS2", "MS3"]
    # One dropped fold in four is under the 25 % bar, so no survivorship warning.
    assert "warning" not in res


def test_majority_skipped_folds_raise_a_warning(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loso, "run_one_config", _fake_run_one_config(strong=True))
    df = _training_df(pos_by_subject={"MS2": 0, "MS3": 0, "MS4": 0})
    res = loso.leave_one_subject_out(_project(tmp_path), TARGET, df=df, n_boot=200)

    assert res["n_folds_skipped"] == 3
    assert "survivorship" in res["warning"]


def test_per_subject_table_has_one_row_per_scored_fold(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loso, "run_one_config", _fake_run_one_config(strong=True))
    df = _training_df(pos_by_subject={"MS4": 0})
    res = loso.leave_one_subject_out(_project(tmp_path), TARGET, df=df, n_boot=200)

    rows = res["per_subject"]
    assert len(rows) == res["n_folds_scored"] == 3
    assert [r["subject"] for r in rows] == ["MS1", "MS2", "MS3"]
    for r in rows:
        assert set(r) == {"subject", "n_rows", "n_positives", "f1_target",
                          "pr_auc", "tp", "fp", "fn", "tn"}
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == r["n_rows"]
        assert r["n_positives"] == 3 and r["n_rows"] == 10
        assert r["f1_target"] == pytest.approx(1.0)   # strong model catches all 3
    # The pooled counts are the per-subject counts summed — no double counting.
    assert sum(r["tp"] for r in rows) == res["raw_tp"]
    assert sum(r["fp"] for r in rows) == res["raw_fp"]


def test_mean_std_sem_matches_formula() -> None:
    mean, std, sem = loso._mean_std_sem([0.2, 0.4, 0.6, 0.8])
    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(np.std([0.2, 0.4, 0.6, 0.8], ddof=1))
    assert sem == pytest.approx(std / np.sqrt(4))
    # NaNs are dropped; a single finite value gives 0 spread.
    m, s, e = loso._mean_std_sem([float("nan"), 0.7])
    assert m == pytest.approx(0.7) and s == 0.0 and e == 0.0
