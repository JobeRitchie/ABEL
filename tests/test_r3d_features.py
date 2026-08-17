"""R3D appearance features: modality classification, gating, and crop geometry.

These are the invariants that decide whether the feature family behaves like a
*video* feature everywhere it is consumed — the ablation arms, the validation
suite, and the benchmark harness all key off the column name.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from abel.models.schemas import BehaviorModelConfig
from abel.services.behavior_representation_service import RepresentationConfig
from abel.services.r3d_feature_service import (
    R3D_DIMS,
    R3DFeatureService,
    is_r3d_column,
    r3d_columns,
)
from abel.validation import features


def test_columns_are_canonical_and_recognised():
    cols = r3d_columns()
    assert len(cols) == R3D_DIMS
    assert cols[0] == "r3d_000" and cols[-1] == f"r3d_{R3D_DIMS - 1:03d}"
    assert all(is_r3d_column(c) for c in cols)
    assert not is_r3d_column("nose_speed_mean")


def test_classified_as_video_not_pose():
    """The whole point: an R3D column must never fall through to the pose baseline."""
    for col in ("r3d_000", "r3d_255", "r3d_511"):
        assert features.classify_modality(col) == features.MODALITY_VIDEO


def test_video_ablation_switches_them_off():
    df = pd.DataFrame(
        {
            "nose_speed_mean": [1.0, 2.0],
            "flow_mag_paw_L_mean": [0.3, 0.4],
            "r3d_000": [0.5, 0.7],
            "r3d_001": [0.1, 0.9],
            "label": ["a", "b"],
        }
    )
    with_video = features.select_feature_cols(df, include_video=True)
    without_video = features.select_feature_cols(df, include_video=False)
    assert {"r3d_000", "r3d_001"} <= set(with_video)
    assert not any(is_r3d_column(c) for c in without_video)


def test_benchmark_harness_treats_them_as_video():
    """The benchmark's keyword lists are separate from the validation taxonomy."""
    from abel.benchmark import runner as bench_runner

    src = bench_runner.__file__
    with open(src, encoding="utf8") as fh:
        text = fh.read()
    # Both video keyword sets must carry the r3d_ prefix, else "video off"
    # silently keeps the appearance features.
    assert text.count('"r3d_"') >= 2


def test_enabled_by_default():
    assert RepresentationConfig().use_r3d_features is True
    assert BehaviorModelConfig().use_r3d_features is True


def test_crop_geometry_tracks_the_animal():
    n = 50
    # Animal walks across the frame; two keypoints 40 px either side of centre.
    cx = np.linspace(100.0, 500.0, n)
    cy = np.full(n, 250.0)
    pose = SimpleNamespace(
        centroid_x=cx,
        centroid_y=cy,
        x=pd.DataFrame({"a": cx - 40.0, "b": cx + 40.0}),
        y=pd.DataFrame({"a": cy, "b": cy}),
    )
    out_cx, out_cy, side = R3DFeatureService._crop_geometry(pose)
    assert np.allclose(out_cx, cx) and np.allclose(out_cy, cy)
    # 2.5 x median extent (40 px) = 100 px.
    assert side == pytest.approx(100, abs=1)


def test_crop_geometry_survives_untracked_frames():
    """Dropped tracking must carry the last known centre, not collapse to (0, 0)."""
    cx = np.array([np.nan, 200.0, np.nan, 300.0, np.nan])
    cy = np.array([np.nan, 150.0, np.nan, 160.0, np.nan])
    pose = SimpleNamespace(
        centroid_x=cx,
        centroid_y=cy,
        x=pd.DataFrame({"a": cx - 30.0, "b": cx + 30.0}),
        y=pd.DataFrame({"a": cy, "b": cy}),
    )
    out_cx, out_cy, side = R3DFeatureService._crop_geometry(pose)
    assert np.isfinite(out_cx).all() and np.isfinite(out_cy).all()
    assert out_cx[0] == 200.0  # back-filled
    assert out_cx[-1] == 300.0  # forward-filled
    assert side >= 96


def test_toggle_invalidates_the_representation_cache():
    """Without this the checkbox is inert on any project that already has a cache."""
    from abel.services.behavior_representation_service import BehaviorRepresentationService

    on = BehaviorRepresentationService._config_signature(RepresentationConfig())
    off = BehaviorRepresentationService._config_signature(
        RepresentationConfig(use_r3d_features=False)
    )
    assert on != off


def test_multi_animal_segments_crop_around_their_own_animal():
    """Mouse2's segments must not be embedded from a crop around Mouse1."""
    n = 20
    a = SimpleNamespace(
        centroid_x=np.full(n, 100.0), centroid_y=np.full(n, 100.0),
        x=pd.DataFrame({"p": np.full(n, 80.0), "q": np.full(n, 120.0)}),
        y=pd.DataFrame({"p": np.full(n, 100.0), "q": np.full(n, 100.0)}),
    )
    b = SimpleNamespace(
        centroid_x=np.full(n, 400.0), centroid_y=np.full(n, 300.0),
        x=pd.DataFrame({"p": np.full(n, 380.0), "q": np.full(n, 420.0)}),
        y=pd.DataFrame({"p": np.full(n, 300.0), "q": np.full(n, 300.0)}),
    )
    session = SimpleNamespace(
        session_id="s1",
        individuals=["Mouse1", "Mouse2"],
        individual_subject_map={"Mouse1": "green", "Mouse2": "black"},
        subject_id="cage7",
        identity_corrections=[],
    )
    manifest = SimpleNamespace(linked_sessions=[session], smoothing_settings=None)
    grp = pd.DataFrame({
        "segment_id": ["s1", "s2", "s3"],
        "animal_id": ["black", "green", "black"],
    })

    svc = R3DFeatureService()
    svc._pose = SimpleNamespace(
        load_and_clean_multi=lambda *a_, **k: SimpleNamespace(
            per_individual={"Mouse1": a, "Mouse2": b}
        ),
    )
    tracks, animal_row = svc._session_geometry(manifest, "s1", __import__("pathlib").Path("p.h5"), grp)

    assert len(tracks) == 2
    assert list(animal_row) == [1, 0, 1]          # black → Mouse2, green → Mouse1
    assert tracks[animal_row[0]][0][0] == 400.0   # black crops around Mouse2
    assert tracks[animal_row[1]][0][0] == 100.0   # green crops around Mouse1


def test_unmapped_individuals_fall_back_to_the_subject_id():
    """Individuals with no subject mapping use ``{subject}:{individual}``."""
    n = 5
    pose = SimpleNamespace(
        centroid_x=np.full(n, 50.0), centroid_y=np.full(n, 50.0),
        x=pd.DataFrame({"p": np.full(n, 40.0)}),
        y=pd.DataFrame({"p": np.full(n, 50.0)}),
    )
    session = SimpleNamespace(
        session_id="s1", individuals=["Mouse1"], individual_subject_map={},
        subject_id="cage7", identity_corrections=[],
    )
    manifest = SimpleNamespace(linked_sessions=[session], smoothing_settings=None)
    grp = pd.DataFrame({"segment_id": ["s1"], "animal_id": ["cage7:Mouse1"]})

    svc = R3DFeatureService()
    svc._pose = SimpleNamespace(
        load_and_clean_multi=lambda *a_, **k: SimpleNamespace(per_individual={"Mouse1": pose}),
    )
    tracks, animal_row = svc._session_geometry(manifest, "s1", __import__("pathlib").Path("p.h5"), grp)
    assert len(tracks) == 1 and list(animal_row) == [0]


def test_single_animal_session_uses_project_smoothing_settings():
    """The crop must follow the same cleaned track the rest of the pipeline sees."""
    n = 5
    pose = SimpleNamespace(
        centroid_x=np.full(n, 10.0), centroid_y=np.full(n, 10.0),
        x=pd.DataFrame({"p": np.full(n, 5.0)}),
        y=pd.DataFrame({"p": np.full(n, 10.0)}),
    )
    smoothing = object()
    session = SimpleNamespace(
        session_id="s1", individuals=[], individual_subject_map={},
        subject_id="m1", identity_corrections=[],
    )
    manifest = SimpleNamespace(linked_sessions=[session], smoothing_settings=smoothing)
    seen: dict = {}

    def _load_and_clean(path, settings=None):
        seen["settings"] = settings
        return pose

    svc = R3DFeatureService()
    svc._pose = SimpleNamespace(load_and_clean=_load_and_clean)
    tracks, animal_row = svc._session_geometry(
        manifest, "s1", __import__("pathlib").Path("p.h5"),
        pd.DataFrame({"segment_id": ["a", "b"]}),
    )
    assert seen["settings"] is smoothing
    assert len(tracks) == 1 and list(animal_row) == [0, 0]


def test_deployment_honours_the_trained_setting():
    """Direct Use must not embed 512 dims for a model that was trained without them."""
    from abel.services.direct_run_service import DirectRunService

    snap_off = SimpleNamespace(
        feature_extraction_settings={"use_r3d_features": False},
        run_settings={}, model_version="behavior_model_Dig",
    )
    snap_on = SimpleNamespace(
        feature_extraction_settings={"use_r3d_features": True},
        run_settings={}, model_version="behavior_model_Dig",
    )
    root = __import__("pathlib").Path(".")
    assert DirectRunService._resolve_use_r3d_features(snap_off, root) is False
    assert DirectRunService._resolve_use_r3d_features(snap_on, root) is True


def test_deployment_falls_back_to_run_settings(tmp_path):
    """Snapshots predating the field recover the value the model was trained with."""
    import json

    from abel.services.direct_run_service import DirectRunService

    model_dir = tmp_path / "derived" / "models" / "behavior_model_Dig"
    model_dir.mkdir(parents=True)
    (model_dir / "run_settings.json").write_text(
        json.dumps({"use_r3d_features": True}), encoding="utf-8",
    )
    snap = SimpleNamespace(
        feature_extraction_settings={}, run_settings={},
        model_version="behavior_model_Dig",
    )
    assert DirectRunService._resolve_use_r3d_features(snap, tmp_path) is True


def test_attach_requires_segment_columns():
    svc = R3DFeatureService()
    with pytest.raises(ValueError, match="missing"):
        svc.attach(__import__("pathlib").Path("."), pd.DataFrame({"segment_id": ["s1"]}))


def test_attach_is_a_noop_without_a_manifest(tmp_path):
    """A project with no import manifest must pass its segments through untouched."""
    df = pd.DataFrame(
        {
            "segment_id": ["s1", "s2"],
            "session_id": ["a", "a"],
            "start_frame": [0, 10],
            "end_frame": [9, 19],
        }
    )
    out = R3DFeatureService().attach(tmp_path, df)
    assert list(out.columns) == list(df.columns)
    assert len(out) == 2


# ── Dense inference: anchor grid, cache reuse, graceful degradation ──────


def test_anchors_span_the_dense_grid_and_pin_both_ends():
    """No dense window may be extrapolated beyond the computed anchor range."""
    starts = np.arange(0, 300, 3)
    anchors = R3DFeatureService._anchor_frames(starts, window_frames=15, stride=15, n_pose=1000)
    assert anchors[0] == starts.min()
    assert anchors[-1] == starts.max()
    assert np.all(np.diff(anchors) > 0)
    # The training stride is 5x the dense stride, so anchors are 5x cheaper.
    assert len(anchors) <= len(starts) // 5 + 2


def test_anchors_never_run_past_tracked_pose():
    """A pose table shorter than the video must not produce anchors it can't crop."""
    starts = np.arange(0, 600, 3)
    anchors = R3DFeatureService._anchor_frames(starts, window_frames=15, stride=15, n_pose=200)
    assert anchors.max() <= 200 - 15


def test_anchor_grid_survives_a_single_window():
    anchors = R3DFeatureService._anchor_frames(np.array([7]), window_frames=15, stride=15, n_pose=100)
    assert list(anchors) == [7]


def test_cache_reuse_ignores_segments_of_a_different_length(tmp_path):
    """A 30-frame embedding is not a 15-frame one and must not be reused as an anchor."""
    reps = tmp_path / "derived" / "representations"
    reps.mkdir(parents=True)
    pd.DataFrame(
        {
            "segment_id": ["good", "toolong"],
            "session_id": ["s1", "s1"],
            "start_frame": [0, 30],
            "end_frame": [14, 59],
        }
    ).to_parquet(reps / "segment_features.parquet", index=False)

    cache_dir = tmp_path / "derived" / "r3d_features"
    cache_dir.mkdir(parents=True)
    rows = pd.DataFrame(np.arange(2 * R3D_DIMS, dtype=np.float32).reshape(2, R3D_DIMS),
                        columns=r3d_columns())
    rows.insert(0, "segment_id", ["good", "toolong"])
    rows.to_parquet(cache_dir / "s1.parquet", index=False)

    found = R3DFeatureService()._cached_by_start(tmp_path, "s1", window_frames=15)
    assert set(found) == {0}, "only the 15-frame segment is a valid anchor"


def test_cache_reuse_is_empty_without_a_cache(tmp_path):
    assert R3DFeatureService()._cached_by_start(tmp_path, "s1", window_frames=15) == {}


def test_attach_dense_leaves_windows_alone_when_the_video_is_unreachable(tmp_path):
    """An unreachable session degrades to the caller's zero fill, not a failed run."""
    df = pd.DataFrame({"segment_id": ["d1", "d2"], "session_id": ["s1", "s1"],
                       "start_frame": [0, 3], "end_frame": [14, 17]})
    out = R3DFeatureService().attach_dense(tmp_path, df, "s1", window_frames=15)
    assert list(out.columns) == list(df.columns)
    assert len(out) == 2


def test_attach_dense_is_a_noop_on_empty_windows(tmp_path):
    empty = pd.DataFrame(columns=["segment_id", "session_id", "start_frame", "end_frame"])
    out = R3DFeatureService().attach_dense(tmp_path, empty, "s1", window_frames=15)
    assert out.empty


def _dense_fixture(tmp_path):
    video = tmp_path / "s1.mp4"
    pose = tmp_path / "s1.h5"
    video.write_bytes(b"video")
    pose.write_bytes(b"pose")
    svc = R3DFeatureService()
    sig = svc._dense_signature(None, video, pose)
    anchors = np.array([0, 15, 30], dtype=int)
    emb = np.arange(3 * R3D_DIMS, dtype=np.float32).reshape(3, R3D_DIMS)
    return svc, video, pose, sig, anchors, emb


def test_dense_anchors_survive_a_round_trip(tmp_path):
    """Dense anchors are reusable across runs — that is the whole point."""
    svc, _video, _pose, sig, anchors, emb = _dense_fixture(tmp_path)
    svc._store_dense_anchors(tmp_path, "s1", 15, sig, anchors, emb, lambda _m: None)

    found = svc._load_dense_anchors(tmp_path, "s1", 15, sig)
    assert set(found) == {0, 15, 30}
    assert np.allclose(found[15], emb[1])


def test_dense_anchors_live_outside_the_temporal_cache(tmp_path):
    """Clearing the temporal cache must not cost GPU-hours of embedding."""
    svc, _video, _pose, sig, anchors, emb = _dense_fixture(tmp_path)
    svc._store_dense_anchors(tmp_path, "s1", 15, sig, anchors, emb, lambda _m: None)

    from abel.temporal_refinement.temporal_refinement_service import TemporalRefinementService

    temporal = TemporalRefinementService()
    temporal.set_project(tmp_path)
    temporal.clear_temporal_tab_cache(clear_run_artifacts=True)

    assert svc._load_dense_anchors(tmp_path, "s1", 15, sig), "R3D anchors were cleared"


def test_dense_anchors_are_dropped_when_the_video_changes(tmp_path):
    """The embeddings are a function of the pixels; new pixels invalidate them."""
    svc, video, pose, sig, anchors, emb = _dense_fixture(tmp_path)
    svc._store_dense_anchors(tmp_path, "s1", 15, sig, anchors, emb, lambda _m: None)

    video.write_bytes(b"a re-encoded video")
    assert svc._load_dense_anchors(tmp_path, "s1", 15, svc._dense_signature(None, video, pose)) == {}


def test_dense_anchors_of_another_window_length_are_not_reused(tmp_path):
    svc, _video, _pose, sig, anchors, emb = _dense_fixture(tmp_path)
    svc._store_dense_anchors(tmp_path, "s1", 15, sig, anchors, emb, lambda _m: None)
    assert svc._load_dense_anchors(tmp_path, "s1", 30, sig) == {}
