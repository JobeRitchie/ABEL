"""Tests for the missing-target-zone preflight on dense temporal inference.

A session with no ROI drawn resolves to the project default, which is a
zero-area box unless the project sets one.  Every ROI/target feature then comes
out all-NaN and the model emits a near-constant probability for the whole
session, so the run has to be refused rather than allowed to produce a flat
trace that reads as "no bouts".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from abel.services.roi_service import ROIService, is_roi_column
from abel.temporal_refinement.temporal_refinement_service import (
    TemporalRefinementService,
)


# ── column predicate ────────────────────────────────────────────────────────

@pytest.mark.parametrize("col", [
    "in_roi_1_nose",
    "nose_to_roi_1_signed_dist",
    "nose_to_roi_1_edge_dist_max",
    "nose_roi_1_axial_abs_p90",
    "body_centroid_roi_2_lateral_mean",
    "roi_1_present",
    "nose_to_target_dist",
    "body_centroid_to_target_dist_p90",
    "head_angle_to_target_mean",
])
def test_roi_columns_are_detected(col):
    assert is_roi_column(col)


@pytest.mark.parametrize("col", [
    "nose_speed_mean",
    "dist_back_mid_to_tail_base_median",
    "r3d_442",
    "back_left_jerk_energy",
    # "target" only counts as an ROI reference in the _to_target / angle_to_target
    # forms; a keypoint that merely contains the substring must not match.
    "target_pellet_speed_mean",
])
def test_non_roi_columns_are_not_detected(col):
    assert not is_roi_column(col)


# ── ROI coverage audit ──────────────────────────────────────────────────────

def _project(tmp_path: Path, subject_rois: dict, project_zone=None) -> Path:
    cfg = ROIService.default_config()
    if project_zone:
        cfg["project_rois"]["target_zones"] = [project_zone]
    cfg["subject_rois"] = subject_rois
    ROIService().save(tmp_path, cfg)
    return tmp_path


def test_subject_with_drawn_zone_is_covered(tmp_path):
    root = _project(tmp_path, {
        "MS1::s1": {"target_zones": [{"x": 10, "y": 10, "w": 30, "h": 20}]},
    })
    assert ROIService().subjects_without_target_area(root, ["MS1::s1"]) == []


def test_subject_with_no_entry_falls_back_to_zero_area_project_default(tmp_path):
    root = _project(tmp_path, {
        "MS1::s1": {"target_zones": [{"x": 10, "y": 10, "w": 30, "h": 20}]},
    })
    # nsf1 was never drawn — this is the NSF_LPT failure mode.
    assert ROIService().subjects_without_target_area(
        root, ["MS1::s1", "nsf1::s2"]
    ) == ["nsf1::s2"]


def test_project_default_zone_covers_undrawn_subjects(tmp_path):
    root = _project(
        tmp_path,
        {"MS1::s1": {"target_zones": [{"x": 10, "y": 10, "w": 30, "h": 20}]}},
        project_zone={"x": 5, "y": 5, "w": 40, "h": 40},
    )
    assert ROIService().subjects_without_target_area(root, ["nsf1::s2"]) == []


def test_explicit_zero_area_entry_is_uncovered(tmp_path):
    root = _project(tmp_path, {
        "MS1::s1": {"target_zones": [{"x": 0, "y": 0, "w": 0, "h": 0}]},
    })
    assert ROIService().subjects_without_target_area(root, ["MS1::s1"]) == ["MS1::s1"]


# ── the preflight ───────────────────────────────────────────────────────────

def _manifest(pairs: list[tuple[str, str]]):
    """Minimal stand-in exposing what ``_subject_by_session`` reads."""
    return SimpleNamespace(
        videos=[],
        linked_sessions=[
            SimpleNamespace(session_id=sid, subject_id=subj, video_asset_id="")
            for subj, sid in pairs
        ],
    )


def _service(root: Path) -> TemporalRefinementService:
    svc = TemporalRefinementService.__new__(TemporalRefinementService)
    svc._require_project_root = lambda: root  # type: ignore[method-assign]
    return svc


ROI_MODEL = {"feature_cols": ["nose_speed_mean", "nose_to_roi_1_edge_dist_max"]}
POSE_MODEL = {"feature_cols": ["nose_speed_mean", "dist_a_to_b_median"]}


def test_preflight_raises_when_roi_model_meets_undrawn_session(tmp_path):
    root = _project(tmp_path, {
        "MS1::s1": {"target_zones": [{"x": 10, "y": 10, "w": 30, "h": 20}]},
    })
    manifest = _manifest([("MS1", "s1"), ("nsf1", "s2")])
    with pytest.raises(ValueError) as err:
        _service(root)._preflight_target_rois(
            {"bid-a": ROI_MODEL}, ["s1", "s2"], manifest, None
        )
    msg = str(err.value)
    assert "no target zone drawn" in msg
    assert "nsf1 (s2)" in msg
    # The covered session must not be named as a culprit.
    assert "MS1 (s1)" not in msg


def test_preflight_passes_when_every_session_has_a_zone(tmp_path):
    root = _project(tmp_path, {
        "MS1::s1": {"target_zones": [{"x": 10, "y": 10, "w": 30, "h": 20}]},
        "nsf1::s2": {"target_zones": [{"x": 20, "y": 20, "w": 25, "h": 15}]},
    })
    manifest = _manifest([("MS1", "s1"), ("nsf1", "s2")])
    _service(root)._preflight_target_rois(
        {"bid-a": ROI_MODEL}, ["s1", "s2"], manifest, None
    )


def test_preflight_ignores_undrawn_sessions_when_no_model_uses_rois(tmp_path):
    """Pose-only models score a zone-less session perfectly well."""
    root = _project(tmp_path, {})
    manifest = _manifest([("nsf1", "s2")])
    _service(root)._preflight_target_rois(
        {"bid-a": POSE_MODEL}, ["s2"], manifest, None
    )


def test_preflight_only_flags_selected_sessions(tmp_path):
    """An undrawn session that is not being scored is not this run's problem."""
    root = _project(tmp_path, {
        "MS1::s1": {"target_zones": [{"x": 10, "y": 10, "w": 30, "h": 20}]},
    })
    manifest = _manifest([("MS1", "s1"), ("nsf1", "s2")])
    _service(root)._preflight_target_rois(
        {"bid-a": ROI_MODEL}, ["s1"], manifest, None
    )
