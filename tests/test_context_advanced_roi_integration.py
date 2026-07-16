"""End-to-end: advanced ROI features flow through the real context pipeline.

Builds a throwaway project (pose CSV + video + a *two*-ROI config), runs
:meth:`ContextFeatureService.compute_frame_context`, and checks the emitted
frame table.  Covers the things unit tests on the geometry helpers cannot:
that the columns actually reach the feature table, that they are emitted for
*every* ROI rather than only the first, and that the toggle turns them off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services.context_feature_service import (
    ContextFeatureConfig,
    ContextFeatureService,
)
from abel.storage.file_store import write_json, write_yaml

cv2 = pytest.importorskip("cv2")

# Long enough that the service's 8-way frame chunking yields chunks comfortably
# larger than the optical-flow temporal stride, as in a real session.  Much
# shorter clips trip a pre-existing length mismatch in the flow interpolation
# that is unrelated to the ROI features under test here.
N_FRAMES = 240
W, H = 160, 120
BODY_PARTS = ("nose", "ear_left", "ear_right", "body_mid", "tail_base")

# ROI 1: a tall strip on the left (an "arm").  ROI 2: a small box on the right.
ROI_1 = {"shape": "rect", "x": 20, "y": 10, "w": 20, "h": 100}
ROI_2 = {"shape": "rect", "x": 120, "y": 50, "w": 20, "h": 20}


def _write_pose(path, xs, ys):
    """DLC-style single-animal CSV: 3 header rows (scorer/bodyparts/coords)."""
    cols, data = [], {}
    for bp in BODY_PARTS:
        for coord in ("x", "y", "likelihood"):
            cols.append(("DLC", bp, coord))
            if coord == "x":
                data[("DLC", bp, coord)] = xs
            elif coord == "y":
                data[("DLC", bp, coord)] = ys
            else:
                data[("DLC", bp, coord)] = np.full(len(xs), 0.99)
    df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(cols))
    df.index.name = "coords"
    df.to_csv(path)


def _write_video(path):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, (W, H))
    rng = np.random.default_rng(0)
    for _ in range(N_FRAMES):
        vw.write(rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
    vw.release()


@pytest.fixture
def project(tmp_path):
    """Animal walks straight down the middle of ROI 1, so it is inside it."""
    root = tmp_path / "proj"
    (root / "config").mkdir(parents=True)
    (root / "raw").mkdir(parents=True)

    xs = np.full(N_FRAMES, 30.0)                     # centre of ROI 1 (x 20-40)
    ys = np.linspace(15.0, 105.0, N_FRAMES)          # travels down the strip
    _write_pose(root / "raw" / "pose.csv", xs, ys)
    _write_video(root / "raw" / "video.avi")

    write_yaml(root / "config" / "environment_rois.yaml", {
        "schema_version": "0.3.0",
        "roi_count": 2,
        "project_rois": {
            "target_zones": [ROI_1, ROI_2],
            "subject_crop": {"x": 0, "y": 0, "w": 0, "h": 0},
        },
        "subject_rois": {},
        "motion": {"local_radius_px": 10},
        "roi_excluded_day_labels": [],
    })
    write_json(root / "config" / "session_registry.json",
               {"schema_version": "0.2.0", "entries": {}})
    return root


def _run(project, advanced: bool) -> pd.DataFrame:
    return ContextFeatureService().compute_frame_context(
        project_root=project,
        video_path=project / "raw" / "video.avi",
        pose_path=project / "raw" / "pose.csv",
        animal_id="A1",
        session_id="s1",
        config=ContextFeatureConfig(
            prefer_gpu=False,
            downsample_factor=1,
            advanced_roi_features=advanced,
        ),
        save=False,
    )


def test_advanced_roi_columns_are_emitted_by_default(project):
    df = _run(project, advanced=True)
    for pt in ("nose", "forepaw_centroid", "body_centroid"):
        assert f"in_roi_1_{pt}" in df.columns
        assert f"{pt}_to_roi_1_signed_dist" in df.columns
        assert f"{pt}_to_roi_1_edge_dist" in df.columns
        assert f"{pt}_to_roi_1_corner_dist" in df.columns
        assert f"{pt}_roi_1_axial" in df.columns
        assert f"{pt}_roi_1_lateral" in df.columns


def test_columns_are_emitted_for_every_roi_not_just_the_first(project):
    """The whole point of the dynamic loop: ROI 2 gets the same treatment."""
    df = _run(project, advanced=True)
    assert "in_roi_2_nose" in df.columns
    assert "nose_to_roi_2_signed_dist" in df.columns
    assert "nose_roi_2_axial" in df.columns
    # ...and ROI 3 was never configured, so it must not appear.
    assert not [c for c in df.columns if "roi_3" in c]


def test_values_reflect_real_geometry(project):
    df = _run(project, advanced=True)
    # The animal walks down the middle of ROI 1 -> inside it, outside ROI 2.
    assert df["in_roi_1_body_centroid"].mean() == pytest.approx(1.0)
    assert df["in_roi_2_body_centroid"].mean() == pytest.approx(0.0)
    # Inside ROI 1 -> signed distance positive; outside ROI 2 -> negative.
    assert (df["body_centroid_to_roi_1_signed_dist"] > 0).all()
    assert (df["body_centroid_to_roi_2_signed_dist"] < 0).all()
    # Travelling down the arm: axial position sweeps from negative to positive,
    # which is exactly the information distance-to-centre cannot express.
    axial = df["body_centroid_roi_1_axial"].to_numpy()
    assert axial[0] < -0.5 and axial[-1] > 0.5
    # Lateral stays ~0: the animal is on the arm's centreline throughout.
    assert np.abs(df["body_centroid_roi_1_lateral"]).max() < 0.2


def test_centre_distance_alone_cannot_separate_the_two_arm_ends(project):
    """Regression guard for the motivating failure case."""
    df = _run(project, advanced=True)
    d = df["body_centroid_to_roi_1_dist"].to_numpy()
    axial = df["body_centroid_roi_1_axial"].to_numpy()
    # First and last frame are near-equidistant from the ROI centre...
    assert abs(d[0] - d[-1]) < 0.15 * max(d[0], d[-1])
    # ...yet they are at opposite ends of the arm, and axial says so.
    assert np.sign(axial[0]) != np.sign(axial[-1])


def test_toggle_off_removes_the_columns_but_keeps_the_legacy_ones(project):
    df = _run(project, advanced=False)
    assert not [c for c in df.columns if c.startswith("in_roi_")]
    assert not [c for c in df.columns if "_signed_dist" in c or "_axial" in c]
    # The pre-existing centre-based ROI features are untouched.
    assert "nose_to_roi_1_dist" in df.columns
    assert "roi_1_present" in df.columns
