"""Regressions for the defects found in the 0.22 pre-release audit.

Each test pins one specific failure mode that was live in the working tree:
a crash, a fabricated value, or a label silently left behind.
"""

import ast
import json

import numpy as np
import pandas as pd
import pytest


def test_group_box_is_assigned_before_it_is_read() -> None:
    """``_write_clip`` read ``group_box`` 21 lines before assigning it.

    Only ``static_center=True`` short-circuiting kept it from raising, so the
    per-frame-follow crop path was dead on arrival with UnboundLocalError.
    """
    src = open("abel/services/preprocessing_service.py", encoding="utf-8").read()
    fn = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_write_clip"
    )
    names = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "group_box"]
    first_store = min(n.lineno for n in names if isinstance(n.ctx, ast.Store))
    first_load = min(n.lineno for n in names if isinstance(n.ctx, ast.Load))
    assert first_store < first_load


def test_a_partially_present_window_is_not_kept_with_all_nan_features() -> None:
    """The window summary is not NaN-aware, so partial presence means no data.

    At the old 0.5 threshold a window that was 83% present survived the gate
    with every feature NaN -- a row the classifier then routes down its
    missing-value branches and the density scorer fills with column means.
    """
    from abel.services.behavior_representation_service import (
        BehaviorRepresentationService,
        RepresentationConfig,
    )
    from abel.utils.gpu_feature_ops import build_segment_df_fast

    n = 120
    vals = np.arange(n, dtype=float)
    vals[50:90] = np.nan  # a 40-frame absence, past the 30-frame fill bound
    frame_df = pd.DataFrame({
        "frame": np.arange(n),
        "feat": vals,
        "pose_present": (~np.isnan(vals)).astype(float),
    })
    seg = build_segment_df_fast(
        frame_df, ["feat"], "a1", "s1", window_size=60, stride=15,
        include_periodicity=False, presence_cols=["pose_present"],
    )
    # Every window here touches the absence, so every one is all-NaN...
    assert seg["feat_mean"].isna().all()
    # ...including one that is 83% present, which the old 0.5 gate kept.
    assert (seg["pose_present_frac"] > 0.5).any()

    kept = BehaviorRepresentationService._drop_absent_segments(
        seg, RepresentationConfig().min_focal_presence
    )
    assert kept.empty, "a window with NaN features must not reach training"


def test_fully_present_windows_survive_the_gate() -> None:
    from abel.services.behavior_representation_service import (
        BehaviorRepresentationService,
        RepresentationConfig,
    )

    seg = pd.DataFrame({
        "segment_id": ["a", "b", "c"],
        "pose_present_frac": [1.0, 0.99, 0.0],
    })
    kept = BehaviorRepresentationService._drop_absent_segments(
        seg, RepresentationConfig().min_focal_presence
    )
    assert kept["segment_id"].tolist() == ["a"]


def test_absence_suppression_covers_the_whole_scoring_window() -> None:
    """A chunk's probability comes from ``window_frames``, not from its stride.

    Measuring presence over the stride slice left chunks whose own frames were
    tracked but whose scoring window reached into an absence -- so their
    features were NaN and their probability was still published.
    """
    from abel.temporal_refinement.temporal_refinement_service import (
        TemporalRefinementService,
    )

    # Frames 60+ are absent. With step 3 and a 30-frame window, chunk 11 spans
    # frames 33..62 and so reaches into the absence.
    present = np.concatenate([np.ones(60), np.zeros(60)])
    session_df = pd.DataFrame({"frame": np.arange(120), "pose_present": present})

    stride_only = TemporalRefinementService._absent_chunk_mask(
        session_df, n_chunks=40, step_frames=3
    )
    windowed = TemporalRefinementService._absent_chunk_mask(
        session_df, n_chunks=40, step_frames=3, window_frames=30
    )
    assert stride_only is not None and windowed is not None
    assert not stride_only[11], "the stride slice sees only tracked frames"
    assert windowed[11], "the scoring window reaches the absence"
    assert not windowed[:11].any(), "fully tracked windows stay scored"


def test_absence_suppression_is_none_for_legacy_frame_tables() -> None:
    from abel.temporal_refinement.temporal_refinement_service import (
        TemporalRefinementService,
    )

    legacy = pd.DataFrame({"speed": np.ones(90)})
    assert TemporalRefinementService._absent_chunk_mask(legacy, 30, 3) is None


def _plan(**kw):
    from abel.services.identity_remap_service import plan_identity_remap

    base = dict(
        session_id="s1", individuals=["track_0", "track_1"], subject_key="M1",
        old_map={"track_0": "track_0", "track_1": "track_1"},
        new_map={"track_0": "track_0", "track_1": "track_1"},
        old_corrections=[], new_corrections=[], segment_ids=[],
    )
    base.update(kw)
    return plan_identity_remap(**base)


def test_a_segment_whose_own_animal_is_unchanged_is_not_a_rename() -> None:
    """Renaming the *other* animal used to report this label as being moved.

    The confirmation dialog quotes ``n_changed``, so it claimed labels were
    being reassigned when the rewrite was a no-op onto the same id.
    """
    plan = _plan(
        new_map={"track_0": "track_0", "track_1": "white"},
        segment_ids=["seg_track_0_s1_10_25"],
    )
    assert plan.segment_renames == {}


def test_soundboard_labels_follow_their_renamed_window(tmp_path) -> None:
    """The soundboard store is keyed by window id, which *is* the segment id.

    Leaving the old key behind orphaned the per-subject payload while the
    reviewer label moved to the new one.
    """
    pytest.importorskip("pyarrow")
    from abel.services.identity_remap_service import apply_identity_remap

    labels_dir = tmp_path / "derived" / "review_labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "soundboard_labels.json").write_text(json.dumps({
        "windows": {
            "seg_track_0_s1_200_215": [
                {"focal_animal_id": "track_0", "behavior_id": "b"}
            ]
        }
    }))

    plan = _plan(
        new_corrections=[{"frame": 100, "a": "track_0", "b": "track_1"}],
        segment_ids=["seg_track_0_s1_200_215"],
    )
    assert plan.segment_renames == {
        "seg_track_0_s1_200_215": "seg_track_1_s1_200_215"
    }
    apply_identity_remap(tmp_path, plan)

    store = json.loads((labels_dir / "soundboard_labels.json").read_text())["windows"]
    assert list(store) == ["seg_track_1_s1_200_215"]
    row = store["seg_track_1_s1_200_215"][0]
    assert row["focal_animal_id"] == "track_1"
    # A solo label must keep its shape: no invented partner key.
    assert "partner_animal_id" not in row


def test_presence_columns_are_never_model_features() -> None:
    """They are float32 like every feature, so a dtype filter alone lets them in."""
    from abel.services.active_learning_trainer_service import ActiveLearningTrainerService

    df = pd.DataFrame({
        "segment_id": ["a"], "label": ["x"], "speed_mean": [1.0],
        "pose_present_frac": np.float32([1.0]),
        "partner_present_frac": np.float32([0.0]),
    })
    assert ActiveLearningTrainerService._numeric_feature_cols(df) == ["speed_mean"]


@pytest.mark.parametrize("module_path", [
    "abel/services/dissimilarity_service.py",
    "abel/services/feature_audit_service.py",
    "abel/services/candidate_service.py",
])
def test_presence_columns_are_excluded_from_every_other_feature_list(module_path) -> None:
    """Each of these picks feature columns by dtype, so the names must be listed."""
    src = open(module_path, encoding="utf-8").read()
    assert "pose_present_frac" in src, module_path
    assert "partner_present_frac" in src, module_path


def test_queue_diagnostics_describe_the_persisted_queue(tmp_path) -> None:
    """During a batch run the file holds every behavior, not just the last.

    Reporting the run's own candidates made n_selected and every
    reason_fraction describe one behavior while the queue held fifteen.
    """
    from abel.services.candidate_service import CandidateGenerationService

    svc = CandidateGenerationService()
    svc.set_project(tmp_path)
    rows = [
        {"segment_id": "a", "selection_reason": "uncertainty"},
        {"segment_id": "b", "selection_reason": "uncertainty"},
        {"segment_id": "c", "selection_reason": "diversity"},
    ]

    class _Cfg:
        target_behavior_id = "b1"

    svc._write_queue_composition_diagnostics(rows, _Cfg())
    latest = json.loads(
        (tmp_path / "derived" / "analysis" / "diagnostics" / "queue" / "latest.json")
        .read_text()
    )
    payload = json.loads(open(latest["queue_composition"], encoding="utf-8").read())
    assert payload["n_selected"] == 3
    assert payload["reason_counts"] == {"uncertainty": 2, "diversity": 1}
    assert payload["reason_fraction"]["diversity"] == pytest.approx(1 / 3)


def test_a_batch_run_does_not_inherit_the_previous_run_s_queue() -> None:
    """Merging unconditionally piled each Pipeline-All onto the last one.

    The first save of a batch must replace; only the rest merge.
    """
    from abel.ui.tabs.active_learning_tab import ActiveLearningTab

    tab = ActiveLearningTab.__new__(ActiveLearningTab)
    tab._pipeline_all_merge_candidates = True
    tab._pipeline_all_saved_any = False

    calls = [tab._merge_candidates_now() for _ in range(3)]
    assert calls == [False, True, True], "first save replaces, later ones merge"

    # A new batch re-arms, so it starts from a clean queue again.
    tab._pipeline_all_saved_any = False
    assert tab._merge_candidates_now() is False

    # Outside a batch nothing ever merges.
    tab._pipeline_all_merge_candidates = False
    assert tab._merge_candidates_now() is False
