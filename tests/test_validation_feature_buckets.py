"""The feature-family taxonomy, pinned against the columns the pipeline really emits.

Every name below was taken from the services that write them — pose_processing_service
(frame_pose), context_feature_service (context_features) and the social features — not
invented. The ablation's credibility rests entirely on this mapping: a pixel- or
environment-derived column that lands in POSE silently hands the "pose only" baseline
information the animal's body never carried, which is how the context/ROI leak
described in ``abel.validation.features`` went unnoticed.
"""

from __future__ import annotations

import pytest

from abel.validation.features import (
    MODALITY_CONTEXT,
    MODALITY_KINEMATICS,
    MODALITY_POSE,
    MODALITY_SOCIAL,
    MODALITY_VIDEO,
    classify_modality,
    is_pose_feature,
)

# ── The animal's own body: geometry (POSE) and motion (KINEMATICS) ─────────
POSE_COLS = [
    "head_pitch", "body_orientation", "head_direction_angle", "body_length_px",
    "joint_angle_head_body",
    "dist_nose_to_tail_base", "dist_nose_to_tail_base_norm", "dist_forepaw_l_to_forepaw_r",
    "nose_oscillation_power", "forepaw_oscillation_power", "oscillation_energy",
    "nose_autocorr_peak", "forepaw_autocorr_peak", "nose_movement_frequency",
]
KINEMATIC_COLS = [
    "nose_speed", "nose_velocity", "nose_velocity_x", "nose_acceleration", "nose_jerk",
    "nose_forward_velocity", "nose_lateral_velocity", "nose_vertical_velocity",
    "forepaw_speed", "forepaw_vertical_velocity", "centroid_velocity",
    "head_angular_velocity", "head_forward_speed", "head_lateral_speed",
]

# ── Pixel-derived: optical flow + MOG2 background-subtracted surface motion ─
VIDEO_COLS = [
    "flow_mag_paw_l", "flow_mag_paw_r", "flow_mag_nose", "flow_dir_paw", "flow_entropy_local",
    "local_surface_motion_energy", "local_surface_motion_variance", "local_surface_change_rate",
    "nose_surface_motion_energy", "nose_surface_motion_variance", "nose_surface_change_rate",
    # Frame-differencing on a tight crop around the nose tip. These matched no key at
    # all and fell through to the POSE default — pixel signal inside the pose baseline.
    "nose_local_change_rate", "nose_local_variance",
    # ROI-anchored optical flow. Still optical flow: it exists only because a camera
    # saw pixels move, and it dies with the video. The ROI says where it was sampled,
    # not what it measures — so "video off" must drop it.
    "flow_mag_near_target", "flow_mag_near_roi_1", "flow_mag_near_roi_2",
]

# ── The environment, strictly its geometry: ROI distances, angles, presence ─
# Nothing pixel-derived belongs here — these are all computable from pose + the ROI
# definition alone, with no camera. (The ROI-anchored *flow* columns are VIDEO.)
CONTEXT_COLS = [
    "roi_1_present", "roi_2_present",
    "nose_to_target_dist", "forepaw_centroid_to_target_dist", "body_centroid_to_target_dist",
    "head_angle_to_target", "body_angle_to_target",
    "nose_to_roi_2_dist", "body_centroid_to_roi_2_dist",
    "head_angle_to_roi_2", "body_angle_to_roi_2",
]

SOCIAL_COLS = [
    "social_dist_centroid_to_centroid_nearest", "social_dist_centroid_to_centroid_nearest_norm",
    "social_approach_velocity_nearest", "social_radial_velocity_toward_nearest",
    "social_facing_angle_nearest", "social_heading_alignment_nearest",
    "social_in_contact", "social_in_contact_duration_s", "social_min_keypoint_dist_nearest",
]

_EXPECTED = (
    [(c, MODALITY_POSE) for c in POSE_COLS]
    + [(c, MODALITY_KINEMATICS) for c in KINEMATIC_COLS]
    + [(c, MODALITY_VIDEO) for c in VIDEO_COLS]
    + [(c, MODALITY_CONTEXT) for c in CONTEXT_COLS]
    + [(c, MODALITY_SOCIAL) for c in SOCIAL_COLS]
)


@pytest.mark.parametrize(("col", "expected"), _EXPECTED, ids=[c for c, _ in _EXPECTED])
def test_column_lands_in_the_right_bucket(col: str, expected: str) -> None:
    assert classify_modality(col) == expected


# The trainer aggregates every frame column over the segment; the suffix must not
# change what the feature *is*.
@pytest.mark.parametrize("suffix", ["_mean", "_std", "_max", "_median", "_p10", "_p90",
                                    "_energy", "_periodicity"])
@pytest.mark.parametrize(("col", "expected"), _EXPECTED, ids=[c for c, _ in _EXPECTED])
def test_aggregation_suffix_does_not_change_the_bucket(col: str, expected: str,
                                                       suffix: str) -> None:
    assert classify_modality(col + suffix) == expected


def test_pose_baseline_excludes_every_pixel_and_environment_column() -> None:
    """The ablation baseline must be the animal's body and nothing else.

    ``is_pose_feature`` is what selects the "pose only" columns, so a single video or
    context column answering True here silently credits the environment (or the
    camera) to pose — and the pairs that most need interrogating get reported as
    trivially solved.
    """
    leaked = [c for c in VIDEO_COLS + CONTEXT_COLS + SOCIAL_COLS if is_pose_feature(c)]
    assert not leaked, f"non-body columns inside the pose-only baseline: {leaked}"

    missing = [c for c in POSE_COLS + KINEMATIC_COLS if not is_pose_feature(c)]
    assert not missing, f"body columns missing from the pose-only baseline: {missing}"
