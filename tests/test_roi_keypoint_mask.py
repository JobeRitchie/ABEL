"""Keypoints outside the drawn zones are ignored, and long dropouts are held.

Reproduces the KORKO EPM failure: DLC put confident keypoints on a cable clamp
beside the maze, so some frames had half the body parts on the mouse and half
on the clamp, and the averaged centroid landed off the maze.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from abel.models.schemas import PoseSmoothingSettings
from abel.services.export_service import ExportService
from abel.services.pose_processing_service import PoseData, PoseProcessingService

sys.path.insert(0, str(Path(__file__).parent))
from test_export_abel_position_csv import BODY_PARTS, N_FRAMES, _make_project  # noqa: E402

# One horizontal arm: x 0..400, y 100..140.  The "clamp" sits well outside it.
ZONE = {"x": 0, "y": 100, "w": 400, "h": 40}
CLAMP = (350.0, 300.0)
MOUSE = (100.0, 120.0)


def _pose(n: int, split: slice, clamp_only: slice | None = None) -> PoseData:
    parts = ["nose", "body", "tail"]
    x = pd.DataFrame({p: np.full(n, MOUSE[0]) for p in parts})
    y = pd.DataFrame({p: np.full(n, MOUSE[1]) for p in parts})
    lik = pd.DataFrame({p: np.full(n, 0.9) for p in parts})
    # Two of three parts on the clamp, confidently.
    x.loc[x.index[split], ["nose", "body"]] = CLAMP[0]
    y.loc[y.index[split], ["nose", "body"]] = CLAMP[1]
    if clamp_only is not None:
        x.iloc[clamp_only] = CLAMP[0]
        y.iloc[clamp_only] = CLAMP[1]
    return PoseData(parts, x, y, lik, x.mean(axis=1).to_numpy(), y.mean(axis=1).to_numpy(), n)


def _clean(pose: PoseData, zones=None) -> PoseData:
    return PoseProcessingService().clean_pose(
        pose, likelihood_threshold=0.4, smoothing_window=1, keep_zones=zones,
    )


def test_split_frames_pull_the_centroid_off_the_zone_without_the_mask() -> None:
    cleaned = _clean(_pose(100, slice(40, 50)))
    assert cleaned.centroid_y[45] > 200  # between mouse and clamp


def test_mask_keeps_the_centroid_on_the_mouse() -> None:
    cleaned = _clean(_pose(100, slice(40, 50)), zones=[ZONE])
    np.testing.assert_allclose(cleaned.centroid_x, MOUSE[0])
    np.testing.assert_allclose(cleaned.centroid_y, MOUSE[1])
    # Masked parts read as missed detections downstream.
    assert (cleaned.likelihood.iloc[40:50][["nose", "body"]] == 0).all().all()
    assert (cleaned.likelihood.iloc[40:50]["tail"] == 0.9).all()


def test_clamp_only_stretch_is_held_not_blanked_as_absent() -> None:
    # 100 frames with every keypoint on the clamp: longer than the 30-frame
    # absence bound, but the animal was missed, not gone.
    cleaned = _clean(_pose(300, slice(0, 0), clamp_only=slice(100, 200)), zones=[ZONE])
    assert np.isfinite(cleaned.centroid_x).all()
    np.testing.assert_allclose(cleaned.centroid_y, MOUSE[1])


def test_margin_keeps_head_dips_just_past_the_edge() -> None:
    pose = _pose(20, slice(0, 0))
    pose.y.loc[5, "nose"] = 100 - 20  # 20 px over the edge, inside the 30 px margin
    cleaned = _clean(pose, zones=[ZONE])
    assert cleaned.y.loc[5, "nose"] == 80
    assert cleaned.likelihood.loc[5, "nose"] == 0.9


def test_long_gap_is_held_not_drawn_along_a_line() -> None:
    s = pd.DataFrame({"a": [0.0] + [np.nan] * 20 + [100.0] + [np.nan] * 3 + [200.0]})
    out = PoseProcessingService._interpolate_short_gaps(s, max_gap=10)["a"]
    assert out.iloc[1:21].isna().all()  # left for the hold step, no drift toward 100
    np.testing.assert_allclose(out.iloc[22:25], [125.0, 150.0, 175.0])


def test_setting_round_trips_and_changes_the_cache_signature(tmp_path: Path) -> None:
    (tmp_path / "project.yaml").write_text(
        yaml.safe_dump({"feature_extraction": {"roi_keypoint_mask": True}}), encoding="utf-8",
    )
    s = PoseSmoothingSettings.load_from_project(tmp_path)
    assert s.roi_keypoint_mask is True
    assert s.cache_signature() == {"roi_keypoint_mask": True}


def _write_clamp_pose(path: Path) -> None:
    """Every part on (x=frame, y=120) except frames 100-149, where two are on the clamp."""
    frames = np.arange(N_FRAMES, dtype=float)
    data = {}
    for bp in BODY_PARTS:
        x = frames.copy()
        y = np.full(N_FRAMES, 120.0)
        if bp != "tail_base":
            x[100:150] = CLAMP[0]
            y[100:150] = CLAMP[1]
        data[("DLC", bp, "x")] = x
        data[("DLC", bp, "y")] = y
        # Confidence 0.3: kept at the import default (0.2), dropped at 0.4.
        lik = np.full(N_FRAMES, 0.9)
        if bp == "nose":
            lik[200:210] = 0.3
            x[200:210] = 999.0
        data[("DLC", bp, "likelihood")] = lik
    df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(list(data)))
    df.index.name = "coords"
    df.to_csv(path)


def test_position_export_uses_project_settings_and_the_mask(tmp_path: Path) -> None:
    root = _make_project(tmp_path, [("s1", "MS1", "MS1_EPM")])
    _write_clamp_pose(root / "raw" / "MS1_EPM.csv")
    (root / "project.yaml").write_text(yaml.safe_dump({"feature_extraction": {
        "likelihood_threshold": 0.4, "smoothing_window": 1, "roi_keypoint_mask": True,
    }}), encoding="utf-8")
    (root / "config").mkdir(exist_ok=True)
    (root / "config" / "environment_rois.yaml").write_text(yaml.safe_dump({
        "roi_count": 1,
        "project_rois": {"target_zones": [{"x": 0, "y": 100, "w": 400, "h": 40}]},
    }), encoding="utf-8")

    svc = ExportService()
    svc.set_project(root)
    res = svc.export_abel_position_csv()
    df = pd.read_csv(res.output_paths[0])
    assert df["Y"].between(100, 140).all()
    # Low-confidence nose at x=999 was dropped by the project's 0.4 threshold.
    assert df["X"].iloc[200:210].max() < 300
