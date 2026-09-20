"""An animal added to the cage part-way through a session must read as absent.

The failure this pins down: ``clean_pose`` used to back-fill a leading run of
undetected frames with the animal's first detected pose, so a mouse added 30 s
into a session appeared as a frozen phantom sitting at its entry point from
frame 0, which also fabricated social distance and contact for the animals
that really were there.
"""

import numpy as np
import pandas as pd

from abel.models.schemas import PoseSmoothingSettings
from abel.services.pose_processing_service import PoseData, PoseProcessingService

PARTS = ["nose", "left_ear", "right_ear", "center_body", "tail_base"]


def _pose(n: int, absent_until: int = 0, x0: float = 100.0) -> PoseData:
    """A pose that walks along x, undetected (NaN, likelihood 0) until *absent_until*."""
    xs = x0 + np.arange(n, dtype=float)
    x = pd.DataFrame({p: xs + i for i, p in enumerate(PARTS)})
    y = pd.DataFrame({p: np.full(n, 50.0 + i) for i, p in enumerate(PARTS)})
    lk = pd.DataFrame({p: np.ones(n) for p in PARTS})
    if absent_until > 0:
        x.iloc[:absent_until] = np.nan
        y.iloc[:absent_until] = np.nan
        lk.iloc[:absent_until] = 0.0
    return PoseData(
        body_parts=list(PARTS), x=x, y=y, likelihood=lk,
        centroid_x=np.zeros(n), centroid_y=np.zeros(n), n_frames=n,
    )


def test_long_leading_absence_stays_nan_instead_of_backfilling() -> None:
    svc = PoseProcessingService()
    cleaned = svc.clean_pose(_pose(300, absent_until=150), absence_max_fill_frames=30)

    assert np.isnan(cleaned.centroid_x[:150]).all(), "absent frames were back-filled"
    assert np.isfinite(cleaned.centroid_x[150:]).all(), "present frames went missing"
    assert cleaned.x.iloc[:150].isna().all().all()


def test_short_dropout_is_still_filled() -> None:
    """Ordinary tracking jitter must behave exactly as before the fix."""
    svc = PoseProcessingService()
    pose = _pose(300)
    pose.x.iloc[100:110] = np.nan
    pose.y.iloc[100:110] = np.nan
    pose.likelihood.iloc[100:110] = 0.0

    cleaned = svc.clean_pose(pose, absence_max_fill_frames=30)
    assert np.isfinite(cleaned.centroid_x).all()


def test_absence_bound_can_be_disabled() -> None:
    svc = PoseProcessingService()
    cleaned = svc.clean_pose(_pose(300, absent_until=150), absence_max_fill_frames=0)
    assert np.isfinite(cleaned.centroid_x).all()


def test_absence_mask_boundary_is_strictly_greater_than_the_bound() -> None:
    mask = PoseProcessingService._absence_mask(
        np.array([False] * 10 + [True] * 30 + [False] * 10), 30
    )
    assert not mask.any(), "a run of exactly max_fill_frames is a dropout, not an absence"
    mask = PoseProcessingService._absence_mask(
        np.array([False] * 10 + [True] * 31 + [False] * 10), 30
    )
    assert mask[10:41].all() and not mask[:10].any() and not mask[41:].any()


def test_settings_carry_the_bound_through_load_and_clean() -> None:
    assert PoseSmoothingSettings().absence_max_fill_frames == 30


def test_absent_partner_does_not_fabricate_social_contact() -> None:
    """The bug's real damage: the *resident* animal's social features."""
    svc = PoseProcessingService()
    resident = svc.clean_pose(_pose(300), absence_max_fill_frames=30)
    # The late arrival enters at frame 150, right on top of the resident, so a
    # back-filled phantom would score as contact for the whole session.
    late = svc.clean_pose(_pose(300, absent_until=150, x0=100.0), absence_max_fill_frames=30)

    social = svc.compute_frame_social_features(resident, {"late": late}, fps=30.0)
    pre = social.iloc[:150]
    post = social.iloc[150:]

    assert pre["social_dist_centroid_to_centroid_nearest"].isna().all()
    # Unmeasurable, not "measured 0% contact", a 0 would average into a
    # segment as a confident negative.
    assert pre["social_in_contact"].isna().all()
    assert pre["social_in_contact_duration_s"].isna().all()
    assert post["social_in_contact"].notna().all()


def test_presence_mask_and_frame_column() -> None:
    svc = PoseProcessingService()
    cleaned = svc.clean_pose(_pose(300, absent_until=150), absence_max_fill_frames=30)
    mask = svc.presence_mask(cleaned)
    assert not mask[:150].any() and mask[150:].all()

    df = svc.compute_frame_pose_features(cleaned, 30.0, "track_1", "s1", "v1")
    assert df["pose_present"].iloc[:150].eq(0.0).all()
    assert df["pose_present"].iloc[150:].eq(1.0).all()


def test_segment_builder_reports_presence_fraction_as_metadata() -> None:
    from abel.utils.gpu_feature_ops import build_segment_df_fast

    n = 100
    frame_df = pd.DataFrame({
        "frame": np.arange(n),
        "speed": np.ones(n),
        "pose_present": np.concatenate([np.zeros(50), np.ones(50)]),
    })
    seg = build_segment_df_fast(
        frame_df, ["speed"], "track_1", "s1", window_size=10, stride=10,
        presence_cols=["pose_present"],
    )
    assert "pose_present_frac" in seg.columns
    # Presence is bookkeeping: it must not be expanded into feature statistics.
    assert not any(c.startswith("pose_present_") and c.endswith("_mean") for c in seg.columns)
    assert seg["pose_present_frac"].iloc[:5].eq(0.0).all()
    assert seg["pose_present_frac"].iloc[5:].eq(1.0).all()


def test_absent_segments_are_dropped_from_the_representation() -> None:
    from abel.services.behavior_representation_service import BehaviorRepresentationService

    seg = pd.DataFrame({
        "segment_id": ["a", "b", "c", "d"],
        "pose_present_frac": [0.0, 0.4, 0.6, 1.0],
    })
    kept = BehaviorRepresentationService._drop_absent_segments(seg, 0.5)
    assert kept["segment_id"].tolist() == ["c", "d"]


def test_unknown_presence_is_kept_not_deleted() -> None:
    """A frame table extracted before presence tracking must not lose rows."""
    from abel.services.behavior_representation_service import BehaviorRepresentationService

    seg = pd.DataFrame({"segment_id": ["a", "b"], "speed_mean": [1.0, 2.0]})
    assert len(BehaviorRepresentationService._drop_absent_segments(seg, 0.5)) == 2

    seg_nan = pd.DataFrame({"segment_id": ["a"], "pose_present_frac": [np.nan]})
    assert len(BehaviorRepresentationService._drop_absent_segments(seg_nan, 0.5)) == 1


def test_presence_columns_are_never_model_features() -> None:
    from abel.services.active_learning_trainer_service import ActiveLearningTrainerService

    df = pd.DataFrame({
        "segment_id": ["a"], "label": ["x"],
        "speed_mean": [1.0], "pose_present_frac": [1.0], "partner_present_frac": [0.0],
    })
    cols = ActiveLearningTrainerService._numeric_feature_cols(df)
    assert cols == ["speed_mean"]


def test_dense_trace_is_blanked_while_the_animal_is_absent() -> None:
    """A behavior cannot be detected in a mouse that is not in the cage yet."""
    from abel.temporal_refinement.temporal_refinement_service import (
        TemporalRefinementService,
    )

    session_df = pd.DataFrame(
        {"pose_present": np.concatenate([np.zeros(90), np.ones(90)])}
    )
    mask = TemporalRefinementService._absent_chunk_mask(session_df, n_chunks=60, step_frames=3)
    assert mask is not None
    assert mask[:30].all() and not mask[30:].any()


def test_dense_trace_unchanged_for_frame_tables_without_presence() -> None:
    from abel.temporal_refinement.temporal_refinement_service import (
        TemporalRefinementService,
    )

    legacy = pd.DataFrame({"speed": np.ones(90)})
    assert TemporalRefinementService._absent_chunk_mask(legacy, 30, 3) is None


def test_multi_animal_extraction_emits_partner_presence(tmp_path, monkeypatch) -> None:
    """The resident's rows must record that it had nobody to be social with."""
    svc = PoseProcessingService()
    resident = svc.clean_pose(_pose(300), absence_max_fill_frames=30)
    late = svc.clean_pose(_pose(300, absent_until=150, x0=400.0), absence_max_fill_frames=30)

    from abel.services.pose_processing_service import MultiAnimalPoseData

    multi = MultiAnimalPoseData(
        individuals=["track_0", "track_1"],
        per_individual={"track_0": resident, "track_1": late},
        n_frames=300,
    )
    monkeypatch.setattr(svc, "load_and_clean_multi", lambda *a, **k: multi)

    out = svc.extract_and_save_frame_pose_features_multi(
        tmp_path, tmp_path / "unused.h5", 30.0, "s1", "v1",
        {"track_0": "track_0", "track_1": "track_1"},
        enable_social_features=True,
    )
    resident_rows = out[out.animal_id == "track_0"].sort_values("frame")
    assert resident_rows["partner_present"].iloc[:150].eq(0.0).all()
    assert resident_rows["partner_present"].iloc[150:].eq(1.0).all()
    # And its own presence is untouched: it was there the whole time.
    assert resident_rows["pose_present"].eq(1.0).all()
