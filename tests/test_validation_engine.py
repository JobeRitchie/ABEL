"""Tests for the validation platform: holdout split, subsampling, no-leakage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.validation import holdout, subsample
from abel.validation.datamodel import ProjectRef


def _synthetic_training_frame(n_per_session: int = 40, seed: int = 0) -> pd.DataFrame:
    """A tiny training-set-shaped frame: 4 sessions, 2 features, target 'B'."""
    rng = np.random.default_rng(seed)
    rows = []
    sid_n = 0
    for sess in ["s1", "s2", "s3", "s4"]:
        for i in range(n_per_session):
            is_pos = (i % 2 == 0)
            rows.append(
                {
                    "segment_id": f"seg_{sid_n}",
                    "session_id": sess,
                    "animal_id": "a1" if sess in ("s1", "s2") else "a2",
                    "start_frame": i * 10,
                    "end_frame": i * 10 + 9,
                    "label": "B" if is_pos else "no_behavior",
                    "label_source": "user_review",
                    "reviewer_confidence": 1.0 if i % 4 != 1 else 0.5,
                    "feat_a": float(rng.normal(1.0 if is_pos else 0.0)),
                    "feat_b": float(rng.normal()),
                }
            )
            sid_n += 1
    return pd.DataFrame(rows)


def _project_ref() -> ProjectRef:
    return ProjectRef(
        project_id="synthetic",
        name="synthetic",
        root=Path("."),
        classifier_type="hist_gbdt",
        split_strategy="group_shuffle_session",
        behavior_names={"B": "Behavior B"},
    )


def test_holdout_split_no_group_overlap():
    df = _synthetic_training_frame()
    proj = _project_ref()
    sp = holdout.split(proj, holdout_groups=["s4"], min_confidence=1.0, df=df)
    assert set(sp.holdout["session_id"]) == {"s4"}
    assert "s4" not in set(sp.train_pool["session_id"])
    # leakage guard finds nothing
    holdout._assert_no_leakage(sp.train_pool, sp.holdout, "session_id")


def test_holdout_high_confidence_filter():
    df = _synthetic_training_frame()
    proj = _project_ref()
    sp = holdout.split(proj, holdout_groups=["s4"], min_confidence=1.0, df=df)
    # every held-out row must be high confidence
    assert (sp.holdout["reviewer_confidence"] >= 1.0).all()
    # the 0.5-confidence rows were dropped from holdout only
    full_s4 = (df["session_id"] == "s4").sum()
    assert len(sp.holdout) < full_s4


def test_leakage_guard_raises_on_overlap():
    df = _synthetic_training_frame()
    pool = df[df["session_id"].isin(["s1", "s2", "s4"])]
    hold = df[df["session_id"].isin(["s4"])]  # s4 shared -> leakage
    with pytest.raises(AssertionError):
        holdout._assert_no_leakage(pool, hold, "session_id")


def test_subsample_respects_size_and_keeps_positive():
    df = _synthetic_training_frame()
    pool = df[df["session_id"].isin(["s1", "s2", "s3"])]
    sub, n_pos, n_neg = subsample.draw(pool, "B", size=5, group_col="session_id", seed=1)
    assert n_pos == 5
    assert n_pos == subsample.count_positives(sub, "B")
    # smallest size still yields >=1 positive
    sub1, n_pos1, _ = subsample.draw(pool, "B", size=1, group_col="session_id", seed=2)
    assert n_pos1 >= 1


def test_subsample_all_uses_every_positive():
    df = _synthetic_training_frame()
    pool = df[df["session_id"].isin(["s1", "s2", "s3"])]
    total_pos = subsample.count_positives(pool, "B")
    sub, n_pos, _ = subsample.draw(pool, "B", size=subsample.ALL_CLIPS, seed=0)
    assert n_pos == total_pos


def test_engine_never_trains_on_holdout_rows():
    """Over many random subsamples, no held-out segment_id reaches train rows."""
    import abel.validation.engine as engine

    df = _synthetic_training_frame()
    proj = _project_ref()
    sp = holdout.split(proj, holdout_groups=["s4"], min_confidence=1.0, df=df)
    holdout_segids = set(sp.holdout["segment_id"].astype(str))

    captured: dict = {}

    class _Spy:
        def train_and_evaluate(self, frame, cfg, *, project_root=None,
                               precomputed_split=None, feature_cols_override=None,
                               progress_cb=None):
            # The split may carry a third (calibration) index array.
            tr_idx, _va_idx = precomputed_split[0], precomputed_split[1]
            cal_idx = precomputed_split[2] if len(precomputed_split) > 2 else []
            train_segids = set(frame.iloc[np.asarray(tr_idx, dtype=int)]["segment_id"].astype(str))
            cal_segids = set(frame.iloc[np.asarray(cal_idx, dtype=int)]["segment_id"].astype(str))
            # Neither the model nor the calibrator may touch a held-out row, and
            # the calibrator's rows must not also be trained on.
            captured["leak"] = (train_segids | cal_segids) & holdout_segids
            captured["cal_in_train"] = cal_segids & train_segids
            raise RuntimeError("stop after capture")  # we only need the split

        # engine references project.classifier_type etc. via ProjectRef, not Spy
    spy = _Spy()

    for s in range(50):
        sub, n_pos, n_neg = subsample.draw(
            sp.train_pool, "B", size=int(1 + (s % 10)), group_col="session_id", seed=s
        )
        res = engine.run_one_config(
            spy, proj, "B", sub, sp.holdout, seed=s, n_pos_train=n_pos, n_neg_train=n_neg
        )
        # spy raised -> degenerate cell with our captured leak set
        assert captured.get("leak") == set(), f"leak at seed {s}: {captured.get('leak')}"
        assert captured.get("cal_in_train") == set(), f"cal/train overlap at seed {s}"
        assert res.degenerate  # spy raised, so it's surfaced as a degenerate cell


# ── Calibration slice: the calibrator never sees the evaluation rows ────────


def test_calibration_slice_is_group_disjoint_from_fit_rows():
    """The carved slice shares no session with the rows the model trains on."""
    from abel.validation.engine import _carve_calibration_slice

    df = _synthetic_training_frame(n_per_session=40)
    fit_df, cal_df = _carve_calibration_slice(df, "session_id", 0, "B")
    assert not cal_df.empty
    fit_g = set(fit_df["session_id"])
    cal_g = set(cal_df["session_id"])
    assert fit_g.isdisjoint(cal_g)
    assert fit_g | cal_g == set(df["session_id"])


def test_calibration_slice_carries_target_positives():
    """Both sides keep target positives — the one-vs-rest collapse trap.

    A slice full of *other* behaviors looks two-class in the pool but reaches the
    calibrator as a single class once the trainer collapses to target-vs-rest.
    """
    from abel.validation.engine import CAL_MIN_POS, _carve_calibration_slice

    df = _synthetic_training_frame(n_per_session=40)
    # s1 is the only session carrying the target; every other session holds a
    # different behavior, so label diversity is high but target positives are not.
    df.loc[(df["session_id"] != "s1") & (df["label"] == "B"), "label"] = "OTHER"
    fit_df, cal_df = _carve_calibration_slice(df, "session_id", 0, "B")
    # Only one group has positives -> no slice can leave positives on both sides.
    assert cal_df.empty

    # With positives spread across sessions, both sides keep enough of them.
    df2 = _synthetic_training_frame(n_per_session=40)
    fit_df2, cal_df2 = _carve_calibration_slice(df2, "session_id", 0, "B")
    n_pos = lambda d: int((d["label"].astype(str) == "B").sum())  # noqa: E731
    assert n_pos(cal_df2) >= CAL_MIN_POS and n_pos(fit_df2) >= CAL_MIN_POS


def test_calibration_slice_declines_when_pool_too_small():
    """Too few groups/rows -> no slice, so the caller can disable calibration."""
    from abel.validation.engine import _carve_calibration_slice

    tiny = _synthetic_training_frame(n_per_session=2)
    tiny = tiny[tiny["session_id"] == "s1"]          # single group
    fit_df, cal_df = _carve_calibration_slice(tiny, "session_id", 0, "B")
    assert cal_df.empty
    assert len(fit_df) == len(tiny)                  # pool returned intact


def test_engine_disables_calibration_when_no_slice_available():
    """Rather than fall back to calibrating on the eval rows, drop calibration."""
    import abel.validation.engine as engine

    df = _synthetic_training_frame(n_per_session=40)
    proj = _project_ref()
    proj.calibration_method = "sigmoid"
    sp = holdout.split(proj, holdout_groups=["s4"], min_confidence=1.0, df=df)
    seen: dict = {}

    class _Spy:
        def train_and_evaluate(self, frame, cfg, *, precomputed_split=None, **kw):
            seen["method"] = cfg.calibration_method
            seen["n_split"] = len(precomputed_split)
            raise RuntimeError("stop")

    # A pool with one session cannot yield a slice -> calibration must be off.
    one_group = sp.train_pool[sp.train_pool["session_id"] == "s1"]
    engine.run_one_config(_Spy(), proj, "B", one_group, sp.holdout, seed=0)
    assert seen["method"] == "none"
    assert seen["n_split"] == 2       # no cal_idx handed to the trainer

    # A pool with several sessions yields a slice -> calibration stays on.
    engine.run_one_config(_Spy(), proj, "B", sp.train_pool, sp.holdout, seed=0)
    assert seen["method"] == "sigmoid"
    assert seen["n_split"] == 3       # cal_idx present


def test_engine_reports_training_counts_after_carve():
    """n_pos_train describes the rows the model saw, not the pre-carve pool."""
    import abel.validation.engine as engine

    df = _synthetic_training_frame(n_per_session=40)
    proj = _project_ref()
    proj.calibration_method = "sigmoid"
    sp = holdout.split(proj, holdout_groups=["s4"], min_confidence=1.0, df=df)
    pool_pos = subsample.count_positives(sp.train_pool, "B")

    class _Spy:
        def train_and_evaluate(self, frame, cfg, *, precomputed_split=None, **kw):
            raise RuntimeError("stop")

    res = engine.run_one_config(
        _Spy(), proj, "B", sp.train_pool, sp.holdout, seed=0,
        n_pos_train=pool_pos, n_neg_train=0,
    )
    # The slice came out of the budget, so fewer positives were trained on than
    # the caller requested — the reported count must reflect that.
    assert 0 < res.n_pos_train < pool_pos


def test_trainer_calibrates_on_cal_split_not_val():
    """End-to-end: with cal_idx supplied, the calibrator is fit off the val rows.

    Guards the actual defect — a calibrator fit on the evaluation rows scores
    them optimistically, so identical models must differ once the calibration
    source moves.
    """
    from abel.services.active_learning_trainer_service import (
        ActiveLearningTrainerService, TrainingConfig,
    )

    df = _synthetic_training_frame(n_per_session=60, seed=3)
    train = df[df["session_id"].isin(["s1", "s2"])].reset_index(drop=True)
    val = df[df["session_id"] == "s3"].reset_index(drop=True)
    cal = df[df["session_id"] == "s4"].reset_index(drop=True)
    frame = pd.concat([train, val, cal], ignore_index=True)
    tr = np.arange(len(train))
    va = np.arange(len(train), len(train) + len(val))
    ca = np.arange(len(train) + len(val), len(frame))

    def _run(split, method="sigmoid"):
        cfg = TrainingConfig(
            classifier_family="hist_gbdt", calibration_method=method,
            target_label="B", random_state=0, adaptive_complexity=False,
            enable_feature_augmentation=False, deploy_refit_on_all_data=False,
        )
        return ActiveLearningTrainerService().train_and_evaluate(
            frame, cfg, precomputed_split=split,
        )

    r_val = _run((tr, va))          # calibrator fit on the scored rows (old path)
    r_cal = _run((tr, va, ca))      # calibrator fit on its own split (new path)
    r_none = _run((tr, va), method="none")   # no calibration at all

    # Same base model, same evaluation rows — only the calibration source moved.
    assert np.array_equal(r_val.y_val, r_cal.y_val)
    assert not np.allclose(r_val.val_probs, r_cal.val_probs), (
        "probabilities identical — cal_idx was ignored and the calibrator "
        "still fit on the validation split"
    )

    # A supplied-but-unusable slice must SKIP calibration outright, not fall back
    # to the validation split the caller scores.
    cal_neg_only = np.array(
        [i for i in ca if str(frame.iloc[i]["label"]) != "B"], dtype=int
    )
    r_bad = _run((tr, va, cal_neg_only))
    assert np.allclose(r_bad.val_probs, r_none.val_probs), (
        "unusable calibration slice fell back to calibrating on the eval rows"
    )


# ── Real-project smoke test (skipped when the project isn't present) ────────

_CIE = Path(r"c:/Users/jober/CIE_NSF")


@pytest.mark.skipif(
    not (_CIE / "derived" / "training_sets" / "training_set.parquet").exists(),
    reason="CIE_NSF project not available",
)
def test_real_project_learning_curve_point():
    import abel.validation.engine as engine
    from abel.services.active_learning_trainer_service import ActiveLearningTrainerService

    proj = ProjectRef.load(_CIE)
    assert proj.is_valid()
    behavior = "6e849017-f2ed-47a3-9707-69d0bda957e4"  # Rear

    sp = holdout.split(proj, min_confidence=1.0, test_size=0.25, seed=42)
    assert len(sp.holdout) > 0 and len(sp.train_pool) > 0

    sub, n_pos, n_neg = subsample.draw(
        sp.train_pool, behavior, size=50, group_col=sp.group_col, seed=0
    )
    assert n_pos == 50

    trainer = ActiveLearningTrainerService()
    res = engine.run_one_config(
        trainer, proj, behavior, sub, sp.holdout, seed=0,
        n_pos_train=n_pos, n_neg_train=n_neg,
    )
    assert not res.error, res.error
    assert np.isfinite(res.f1)
    # NOT 50: the honest calibration slice is carved out of the caller's budget,
    # so the reported count describes the positives the base model actually fit
    # on (see engine's module docstring). A learning curve must plot consumed
    # labels, not requested ones.
    assert 0 < res.n_pos_train <= 50
    assert res.n_neg_train > 0


@pytest.mark.skipif(
    not (_CIE / "derived" / "training_sets" / "training_set.parquet").exists(),
    reason="CIE_NSF project not available",
)
def test_al_vs_random_discovers_positives_faster():
    """Active learning (ABEL candidate ranking) should surface positives ≥ random."""
    from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
    from abel.validation.analyses.al_curve import run_al_vs_random

    proj = ProjectRef.load(_CIE)
    behavior = "138ed426-561b-4b66-8e5f-9dc1f9bcbcba"  # Groom (rarer → clearer AL win)
    sp = holdout.split(proj, min_confidence=1.0, seed=42)
    trainer = ActiveLearningTrainerService()

    res = run_al_vs_random(
        trainer, proj, behavior, sp,
        n_seeds=2, k0=20, batch=30, max_budget=110, seed_pos=5,
    )
    assert res.al_points and res.random_points
    # AL must never acquire a held-out clip: every cell's positives come from the pool,
    # and the held-out set is fixed — so no error and finite F1 once the model trains.
    assert all(not c.error for c in res.cells if c.n_clips >= 50)
    # Headline claim: AL discovers ≥ as many positives by the final budget.
    al_pos = res.al_points[-1].n_pos_mean
    rnd_pos = res.random_points[-1].n_pos_mean
    assert al_pos >= rnd_pos, f"AL discovered {al_pos} positives vs random {rnd_pos}"


@pytest.mark.skipif(
    not (_CIE / "derived" / "training_sets" / "training_set.parquet").exists(),
    reason="CIE_NSF project not available",
)
def test_gui_worker_completes_and_populates(tmp_path):
    """End-to-end GUI worker path: guards the worker-reference GC regression."""
    import os
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("PySide6 not available")

    app = QApplication.instance() or QApplication([])
    from abel.validation.gui import ValidationWindow
    from abel.validation.runner import ANALYSIS_GENERALIZATION

    win = ValidationWindow()
    proj = ProjectRef.load(_CIE)
    win._projects[proj.project_id] = proj
    win._selected[proj.project_id] = {"6e849017-f2ed-47a3-9707-69d0bda957e4"}
    win._output_dir = tmp_path
    win._gen_seeds.setValue(1)

    win._run([ANALYSIS_GENERALIZATION])
    t0 = time.time()
    while win._busy and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.03)

    assert not win._busy, "GUI run did not finish (worker signal not delivered?)"
    assert win._last_run is not None
    assert win._gen_panel._strip._grid.count() > 0
    assert win._last_run.report_path.exists()


# ── Ablation config builder (incremental add-one-in) ────────────────────────


def test_build_ablation_configs_incremental():
    from abel.validation.analyses.ablation import (
        ALL_FEATURES_CONFIG, BASELINE_CONFIG, build_ablation_configs,
    )

    # Video on, co-occurring off.
    proj = ProjectRef(project_id="P", name="P", root=Path("."),
                      calibration_method="sigmoid",
                      use_video_features=True, allow_co_occurring_behaviors=False)
    cfgs = build_ablation_configs(proj)
    names = [c.name for c in cfgs]
    assert names[0] == BASELINE_CONFIG
    assert names[-1] == ALL_FEATURES_CONFIG
    assert "add_video_features" in names
    assert "add_co_occurring" not in names           # gated off
    # Baseline turns every enhancement off; "all" turns them on.
    base = next(c for c in cfgs if c.name == BASELINE_CONFIG)
    assert base.feature_set == "pose"
    assert base.overrides["calibration_method"] == "none"
    assert base.overrides["adaptive_complexity"] is False
    allc = next(c for c in cfgs if c.name == ALL_FEATURES_CONFIG)
    assert allc.feature_set == "all"
    assert allc.overrides["adaptive_complexity"] is True
    assert allc.overrides["calibration_method"] == "sigmoid"

    # No video, co-occurring on.
    proj2 = ProjectRef(project_id="Q", name="Q", root=Path("."),
                       use_video_features=False, allow_co_occurring_behaviors=True)
    names2 = [c.name for c in build_ablation_configs(proj2)]
    assert "add_video_features" not in names2
    assert "add_co_occurring" in names2


def test_features_social_family_separation():
    """social_* columns are their own family, kept out of the pose baseline."""
    from abel.validation import features

    df = pd.DataFrame({
        "head_angle": [0.1, 0.2],
        "nose_velocity": [0.3, 0.4],
        "flow_mag_mean": [1.0, 2.0],                 # video
        "social_dist_nearest_mean": [3.0, 4.0],       # social
        "social_approach_velocity_nearest": [5.0, 6.0],
        "label": ["B", "no_behavior"],                # non-numeric, ignored
    })
    pose = set(features.pose_only_cols(df))
    video = set(features.video_only_cols(df))
    social = set(features.social_only_cols(df))

    assert social == {"social_dist_nearest_mean", "social_approach_velocity_nearest"}
    assert video == {"flow_mag_mean"}
    # The pose baseline must exclude BOTH video and social columns.
    assert pose.isdisjoint(social) and pose.isdisjoint(video)
    assert {"head_angle", "nose_velocity"} <= pose

    # select_feature_cols composes families additively.
    pv = features.select_feature_cols(df, include_video=True, include_social=False)
    ps = features.select_feature_cols(df, include_video=False, include_social=True)
    assert video <= set(pv) and social.isdisjoint(pv)
    assert social <= set(ps) and video.isdisjoint(ps)


def test_classify_modality_social():
    """social_* features are the social modality, not misread as kinematics."""
    from abel.validation.analyses.behaviorscape import (
        MODALITY_SOCIAL, classify_modality,
    )

    # Contains "velocity" but the social_ prefix wins.
    assert classify_modality("social_approach_velocity_nearest") == MODALITY_SOCIAL
    assert classify_modality("social_in_contact") == MODALITY_SOCIAL
    assert classify_modality("social_heading_alignment_mean") == MODALITY_SOCIAL
    # A non-social kinematic feature is unaffected.
    assert classify_modality("nose_velocity_mean") != MODALITY_SOCIAL


def test_build_ablation_configs_social_gated():
    """The social bar appears only when the pool carries social_* columns."""
    from abel.validation.analyses.ablation import build_ablation_configs

    proj = ProjectRef(project_id="M", name="M", root=Path("."),
                      use_video_features=True)
    solo = [c.name for c in build_ablation_configs(proj, has_social=False)]
    multi = {c.name: c for c in build_ablation_configs(proj, has_social=True)}
    assert "add_social_features" not in solo
    assert "add_social_features" in multi
    assert multi["add_social_features"].feature_set == "pose+social"


# ── Confusion counts + learning-curve point means (synthetic) ───────────────


def test_engine_fills_confusion_counts():
    import abel.validation.engine as engine
    from abel.services.active_learning_trainer_service import ActiveLearningTrainerService

    df = _synthetic_training_frame()
    proj = _project_ref()
    sp = holdout.split(proj, holdout_groups=["s4"], min_confidence=1.0, df=df)
    sub, n_pos, n_neg = subsample.draw(sp.train_pool, "B", size=subsample.ALL_CLIPS, seed=0)
    trainer = ActiveLearningTrainerService()
    res = engine.run_one_config(trainer, proj, "B", sub, sp.holdout, seed=0,
                                n_pos_train=n_pos, n_neg_train=n_neg)
    if res.error or res.y_true is None:
        pytest.skip(f"synthetic training unavailable: {res.error}")
    total = res.tp + res.fp + res.fn + res.tn
    assert total == len(res.y_true)
    # Internal consistency with the binary target-vs-rest arrays.
    assert res.tp + res.fn == int(res.y_true.sum())
    assert res.tp + res.fp == int(res.y_pred.sum())


def test_ablation_significance_guard():
    from abel.validation.analyses.ablation import AblationResult

    r = AblationResult(project_id="x", behavior_id="b", behavior_name="B")
    r.gain = {"a": 0.05, "c": 0.001}
    r.gain_ci = {"a": 0.0, "c": 0.02}
    r.gain_n = {"a": 2, "c": 5}
    assert r.is_significant("a")          # consistent gain, CI=0, ≥2 seeds → distinguishable
    assert not r.is_significant("c")      # |gain| < CI → within noise
    r.gain_n["a"] = 1
    assert not r.is_significant("a")      # a single seed is never significant


def test_ablation_budget_trains_on_subsample():
    from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
    from abel.validation.analyses import ablation

    df = _synthetic_training_frame()
    proj = _project_ref()
    sp = holdout.split(proj, holdout_groups=["s4"], min_confidence=1.0, df=df)
    trainer = ActiveLearningTrainerService()
    r = ablation.run_ablation(trainer, proj, "B", sp, n_seeds=2, clip_budget=8)
    assert r.clip_budget == 8
    assert r.order[0] == ablation.BASELINE_CONFIG
    assert r.order[-1] == ablation.ALL_FEATURES_CONFIG
    # Every config got paired gain bookkeeping (except the baseline itself).
    for name in r.order:
        if name == ablation.BASELINE_CONFIG:
            continue
        assert name in r.gain and name in r.gain_ci and name in r.gain_n
    # No config trained on more than the requested budget of positives.
    assert max(c.n_pos_train for c in r.cells) <= 8


def test_learning_curve_point_has_count_means():
    from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
    from abel.validation.analyses import learning_curve

    df = _synthetic_training_frame()
    proj = _project_ref()
    sp = holdout.split(proj, holdout_groups=["s4"], min_confidence=1.0, df=df)
    trainer = ActiveLearningTrainerService()
    lc = learning_curve.run_learning_curve(
        trainer, proj, "B", sp, sizes=[subsample.ALL_CLIPS], n_seeds=1,
    )
    if not lc.points:
        pytest.skip("synthetic learning curve produced no points")
    p = lc.points[0]
    # New per-point fields exist and counts are non-negative.
    for attr in ("precision_mean", "recall_mean", "tp_mean", "fp_mean", "fn_mean"):
        assert hasattr(p, attr)
    assert p.tp_mean >= 0 and p.fp_mean >= 0 and p.fn_mean >= 0


# ── Video-motion-feature value + throughput benchmark ───────────────────────


def _synthetic_video_frame(n_per_session: int = 40, seed: int = 0) -> pd.DataFrame:
    """Like _synthetic_training_frame but with a video-derived (flow_) column
    that carries the target signal, so 'with video' should beat 'without'."""
    rng = np.random.default_rng(seed)
    rows = []
    sid_n = 0
    for sess in ["s1", "s2", "s3", "s4", "s5", "s6"]:
        for i in range(n_per_session):
            is_pos = (i % 3 == 0)
            rows.append({
                "segment_id": f"seg_{sid_n}", "session_id": sess,
                "animal_id": "a1" if sess in ("s1", "s2", "s3") else "a2",
                "start_frame": i * 10, "end_frame": i * 10 + 9,
                "label": "Groom" if is_pos else "no_behavior",
                "label_source": "user_review", "reviewer_confidence": 1.0,
                "head_angle": float(rng.normal()),         # pose (no signal)
                "nose_speed": float(rng.normal()),          # kinematic (no signal)
                "flow_mag_mean": float(rng.normal(2.0 if is_pos else 0.0)),  # video: signal
            })
            sid_n += 1
    return pd.DataFrame(rows)


def _video_project() -> ProjectRef:
    return ProjectRef(project_id="vsyn", name="vsyn", root=Path("."),
                      classifier_type="hist_gbdt", use_video_features=True,
                      behavior_names={"Groom": "Groom"})


def test_video_value_isolates_video_features():
    from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
    from abel.validation import video_value

    df = _synthetic_video_frame()
    proj = _video_project()
    sp = holdout.split(proj, holdout_groups=["s6"], min_confidence=1.0, df=df)
    r = video_value.run_video_value(
        ActiveLearningTrainerService(), proj, "Groom", sp, n_seeds=2)
    if r.error:
        pytest.skip(f"synthetic training unavailable: {r.error}")
    # 'with video' uses exactly one more feature (the flow_ column) than 'without'.
    assert r.n_features_with_video == r.n_features_no_video + 1
    # The video feature carries the signal, so it should not hurt and normally helps.
    assert np.isfinite(r.gain)
    assert r.f1_with_video >= r.f1_no_video - 1e-6


def test_video_value_errors_without_video_cols():
    from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
    from abel.validation import video_value

    df = _synthetic_training_frame()  # feat_a/feat_b only — no video columns
    proj = _project_ref()
    sp = holdout.split(proj, holdout_groups=["s4"], min_confidence=1.0, df=df)
    r = video_value.run_video_value(
        ActiveLearningTrainerService(), proj, "B", sp, n_seeds=1)
    assert r.error and "video" in r.error.lower()


def test_benchmark_helpers_and_pick_session():
    from abel.validation import benchmark as bench

    # Normalization math.
    xr, ftr = bench._normalize(seconds=100.0, video_seconds=400.0)
    assert abs(xr - 0.25) < 1e-9 and abs(ftr - 4.0) < 1e-9
    assert all(np.isnan(v) for v in bench._normalize(0.0, 0.0))

    # A non-existent project yields no session (graceful), not a crash.
    assert bench.pick_session(Path("does_not_exist_xyz"), need_video=True) is None

    # An all-error result set still frames + tolerates plotting.
    rows = [bench.StageTiming(project_id="P", stage=bench.STAGE_EXTRACT, error="x")]
    dfr = bench.results_to_frame(rows)
    assert list(dfr["stage"]) == [bench.STAGE_EXTRACT]
