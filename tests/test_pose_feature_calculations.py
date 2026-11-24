from pathlib import Path

import numpy as np
import pandas as pd

from abel.services.pose_processing_service import PoseData, PoseProcessingService


def _make_pose(order: list[str], n: int = 30) -> PoseData:
    rng = np.random.default_rng(0)
    x = pd.DataFrame({bp: rng.random(n) for bp in order})
    y = pd.DataFrame({bp: rng.random(n) for bp in order})
    lk = pd.DataFrame({bp: np.ones(n) for bp in order})
    return PoseData(
        x=x, y=y, likelihood=lk, body_parts=list(order),
        centroid_x=np.zeros(n), centroid_y=np.zeros(n), n_frames=n,
    )


def test_pairwise_distance_names_are_keypoint_order_independent() -> None:
    """Two projects with the same keypoints in a different DLC column order must
    produce identical dist_* feature columns (Direct Use cross-project reuse)."""
    from abel.models.schemas import InvariantFeatureConfig
    cfg = InvariantFeatureConfig(
        enable_egocentric_kinematics=False,
        enable_body_length_normalization=True,
        enable_relative_geometry=True,
        enable_head_direction=False,
        enable_joint_angles=False,
        enable_spine_curvature=False,
        enable_clipwise_deltas=False,
    )
    svc = PoseProcessingService()
    order_a = ["nose", "left_ear", "right_ear", "center_body", "tail_base"]
    order_b = ["center_body", "tail_base", "right_ear", "nose", "left_ear"]
    fa = svc.compute_frame_pose_features(_make_pose(order_a), 30.0, "a", "s", "v", invariant_config=cfg)
    fb = svc.compute_frame_pose_features(_make_pose(order_b), 30.0, "a", "s", "v", invariant_config=cfg)
    dist_a = {c for c in fa.columns if c.startswith("dist_")}
    dist_b = {c for c in fb.columns if c.startswith("dist_")}
    assert dist_a == dist_b
    # Names are canonically ordered (sorted pair).
    assert "dist_left_ear_to_nose" in dist_a
    assert "dist_nose_to_left_ear" not in dist_a


def test_compute_frame_pose_features_contains_expected_columns() -> None:
    n = 40
    t = np.arange(n, dtype=float)

    x = pd.DataFrame(
        {
            "nose": 10.0 + 0.5 * t,
            "paw_L": 5.0 + np.sin(t / 3.0),
            "paw_R": 6.0 + np.sin(t / 3.0 + 0.2),
            "tailbase": 3.0 + 0.2 * t,
        }
    )
    y = pd.DataFrame(
        {
            "nose": 8.0 + 0.3 * t,
            "paw_L": 4.0 + np.cos(t / 3.0),
            "paw_R": 4.5 + np.cos(t / 3.0 + 0.2),
            "tailbase": 2.0 + 0.1 * t,
        }
    )
    l = pd.DataFrame(1.0, index=range(n), columns=x.columns)

    pose = PoseData(
        body_parts=list(x.columns),
        x=x,
        y=y,
        likelihood=l,
        centroid_x=np.asarray(x.mean(axis=1), dtype=float),
        centroid_y=np.asarray(y.mean(axis=1), dtype=float),
        n_frames=n,
    )

    svc = PoseProcessingService()
    df = svc.compute_frame_pose_features(
        pose=pose,
        fps=30.0,
        animal_id="rat01",
        session_id="s1",
        video_id="v1",
    )

    assert len(df) == n
    assert "forepaw_speed" in df.columns
    assert "forepaw_movement_frequency" in df.columns
    assert "forepaw_autocorr_peak" in df.columns
    assert float(df["forepaw_speed"].mean()) > 0.0


def test_pose_file_shorter_than_rhythmicity_window(tmp_path: Path) -> None:
    """A pose file with fewer frames than the 1 s rhythmicity window (an
    18-frame clip at 30 fps) must still extract.  ``np.convolve(mode="same")``
    returned window-length oscillation-energy columns, so the frame table
    failed with "All arrays must be of the same length"."""
    import pytest

    parts = ["nose", "left_ear", "right_ear", "neck", "center_body", "tail_base", "tail_tip"]
    cols = pd.MultiIndex.from_product(
        [["DLC_scorer"], parts, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    svc = PoseProcessingService()
    for n in (18, 29, 30, 45):
        rng = np.random.default_rng(n)
        data = np.empty((n, len(cols)))
        for k in range(len(parts)):
            data[:, 3 * k] = 100.0 + 10.0 * k + rng.normal(0, 2, n)
            data[:, 3 * k + 1] = 200.0 + rng.normal(0, 2, n)
            data[:, 3 * k + 2] = 0.99
        pose_path = tmp_path / f"pose_{n}.csv"
        pd.DataFrame(data, columns=cols).to_csv(pose_path)

        df = svc.extract_and_save_frame_pose_features(
            tmp_path, pose_path, 30.0, "MouseA0", f"session_{n}", f"video_{n}",
        )
        assert len(df) == n
        # No paw keypoints here, so the forepaw energy is NaN by design.
        assert np.isfinite(df["nose_oscillation_energy"]).all()

    # Sessions at least one window long keep the exact mode="same" values.
    for n, w in ((30, 30), (45, 30), (40, 7), (9, 4)):
        arr = np.random.default_rng(n).random(n)
        expected = np.convolve(arr, np.ones(w), mode="same") / float(w)
        assert PoseProcessingService._centered_window_mean(arr, w) == pytest.approx(expected)
    assert len(PoseProcessingService._centered_window_mean(np.zeros(0), 30)) == 0


def test_smooth_probabilities_keeps_length_of_short_trace() -> None:
    from abel.temporal_refinement.bout_postprocess import smooth_probabilities

    short = np.linspace(0.0, 1.0, 3, dtype=np.float32)
    assert len(smooth_probabilities(short, window=9)) == 3
    long = np.random.default_rng(0).random(50).astype(np.float32)
    expected = np.convolve(long, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    assert np.allclose(smooth_probabilities(long, window=5), expected)


def test_short_pose_sessions_are_reported_with_their_files(tmp_path: Path) -> None:
    """Prep flags pose files shorter than one second or one segment window,
    and the warning text names every affected file."""
    from abel.services.feature_prep_service import (
        FeaturePrepService, PrepConfig, SessionJob,
    )
    from abel.ui.tabs.pose_features_tab import short_pose_warning_text

    sessions = tmp_path / "derived" / "pose_features" / "sessions"
    sessions.mkdir(parents=True)
    rows = {"short": 18, "ok": 900, "pair": 40}  # "pair": 2 animals, 20 frames
    for sid, n in rows.items():
        pd.DataFrame({"frame": np.arange(n)}).to_parquet(sessions / f"{sid}.parquet", index=False)

    def _job(sid: str, individuals: list[str] | None = None) -> SessionJob:
        return SessionJob(
            session_id=sid, subject_id=f"Mouse_{sid}",
            pose_path=tmp_path / f"{sid}DLC.csv", video_path=None, fps=30.0,
            individuals=individuals or [],
        )

    jobs = [_job("short"), _job("ok"), _job("pair", ["a", "b"]), _job("not_extracted")]
    found = FeaturePrepService.find_short_pose_sessions(
        tmp_path, jobs, PrepConfig(segment_window_frames=15),
    )
    assert [(f["session_id"], f["n_frames"], f["min_frames"]) for f in found] == [
        ("short", 18, 30), ("pair", 20, 30),
    ]

    headline, body, detailed = short_pose_warning_text(found)
    assert headline.startswith("2 sessions")
    for sid in ("short", "pair"):
        assert f"{sid}DLC.csv" in body
        assert str(tmp_path / f"{sid}DLC.csv") in detailed
    assert "okDLC.csv" not in body
    assert "18 frames" in body

    many = [dict(found[0], session_id=f"s{i}") for i in range(12)]
    _, body_many, detailed_many = short_pose_warning_text(many)
    assert "and 4 more" in body_many
    assert detailed_many.count("\n") == 11
