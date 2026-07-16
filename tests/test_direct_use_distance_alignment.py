"""Regression test for the Direct Use distance-column alignment fix.

v0.5.2 made pose-feature extraction emit canonical (sorted-endpoint) pairwise
distance column names (``dist_a_to_b``).  Inference aligns a model's stored
``feature_cols`` to the data by name with zero-fill for anything missing.  A
model trained *before* v0.5.2 stored the old, unsorted spelling (``dist_b_to_a``),
which no longer exists in freshly extracted features — so every such distance
column was silently zero-filled, killing predictions in Direct Use.

The fix canonicalises the model's distance feature names before reindexing, so an
old-ordered model lines up with the canonical data columns again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abel.services.behavior_representation_service import canonical_distance_name


def test_canonical_distance_name_sorts_endpoints() -> None:
    # Symmetric pair → sorted spelling, both orders collapse to the same name.
    assert canonical_distance_name("dist_b_to_a") == "dist_a_to_b"
    assert canonical_distance_name("dist_a_to_b") == "dist_a_to_b"
    # Multi-token keypoint names are handled (split on the ``_to_`` delimiter).
    assert canonical_distance_name("dist_tail_base_to_nose") == "dist_nose_to_tail_base"
    assert canonical_distance_name("dist_nose_to_tail_base") == "dist_nose_to_tail_base"
    # ``_norm`` variants keep their suffix.
    assert canonical_distance_name("dist_b_to_a_norm") == "dist_a_to_b_norm"
    # Non-distance columns are returned unchanged.
    assert canonical_distance_name("speed") == "speed"
    assert canonical_distance_name("nose_to_target_dist") == "nose_to_target_dist"


def test_old_model_columns_realign_to_canonical_data() -> None:
    """An old-ordered model's distance columns must resolve to real data values."""
    # Freshly extracted features use canonical (sorted) distance names.
    dense_segs = pd.DataFrame({
        "dist_a_to_b": [1.0, 2.0, 3.0],
        "dist_nose_to_tail_base": [4.0, 5.0, 6.0],
        "speed": [0.1, 0.2, 0.3],
    })
    # A model trained before v0.5.2 stored the unsorted spelling.
    model_cols = ["dist_b_to_a", "dist_tail_base_to_nose", "speed"]

    # Old behaviour: reindex by the raw model names → distance columns missing →
    # zero-filled, which is the bug.
    raw = dense_segs.reindex(columns=model_cols, fill_value=0.0).to_numpy(dtype=float)
    assert np.allclose(raw[:, 0], 0.0)  # dist_b_to_a not found
    assert np.allclose(raw[:, 1], 0.0)  # dist_tail_base_to_nose not found

    # Fixed behaviour: canonicalise the model names first.
    aligned_cols = [canonical_distance_name(c) for c in model_cols]
    fixed = dense_segs.reindex(columns=aligned_cols, fill_value=0.0).to_numpy(dtype=float)
    assert np.allclose(fixed[:, 0], [1.0, 2.0, 3.0])  # real dist values, in order
    assert np.allclose(fixed[:, 1], [4.0, 5.0, 6.0])
    assert np.allclose(fixed[:, 2], [0.1, 0.2, 0.3])
    # Column order (and therefore the model's feature order) is preserved.
    assert len(aligned_cols) == len(model_cols)


# ---------------------------------------------------------------------------
# Segment-level names carry a statistic suffix (dist_a_to_b_norm_mean). It must
# be split off before the endpoints are sorted, or it gets swept into the second
# endpoint — turning dist_nose_to_left_ear_mean into dist_left_ear_mean_to_nose,
# a name no table has, which then silently reindexes to a fill value.
# ---------------------------------------------------------------------------


def test_canonical_distance_name_handles_statistic_suffixes() -> None:
    from abel.services.behavior_representation_service import canonical_distance_name as cdn

    assert cdn("dist_nose_to_left_ear_mean") == "dist_left_ear_to_nose_mean"
    assert cdn("dist_left_ear_to_nose_mean") == "dist_left_ear_to_nose_mean"
    assert cdn("dist_nose_to_left_ear_norm_std") == "dist_left_ear_to_nose_norm_std"
    assert cdn("dist_nose_to_center_body_p90") == "dist_center_body_to_nose_p90"
    # Non-distance and ROI columns are untouched.
    assert cdn("nose_speed_mean") == "nose_speed_mean"
    assert cdn("nose_to_target_dist_mean") == "nose_to_target_dist_mean"


def test_aligner_reads_the_other_spelling_when_model_has_only_one() -> None:
    """The v0.5.2 case: model predates canonicalisation, data is canonical."""
    from abel.services.behavior_representation_service import align_model_feature_columns

    model_cols = ["dist_nose_to_left_ear_mean", "speed_mean"]
    data_cols = {"dist_left_ear_to_nose_mean", "speed_mean"}
    source, nan_fill = align_model_feature_columns(model_cols, data_cols)

    # Reads the real value under the canonical spelling — not a fill.
    assert source == ["dist_left_ear_to_nose_mean", "speed_mean"]
    assert nan_fill == [False, False]


def test_aligner_nan_fills_the_surplus_slot_of_a_double_named_pair() -> None:
    """Both spellings in one model: in training one held the value, one was NaN.

    Copying the value into both would present a combination the model never saw,
    and 0.0 would assert an average distance (the features are z-scored).
    """
    from abel.services.behavior_representation_service import align_model_feature_columns

    model_cols = ["dist_left_ear_to_nose_mean", "dist_nose_to_left_ear_mean"]
    data_cols = {"dist_left_ear_to_nose_mean"}
    source, nan_fill = align_model_feature_columns(model_cols, data_cols)

    # Canonical slot reads the value; the surplus legacy slot is NaN, not 0.0.
    assert source[0] == "dist_left_ear_to_nose_mean"
    assert nan_fill[0] is False
    assert nan_fill[1] is True


def test_aligner_zero_fills_genuinely_absent_columns() -> None:
    from abel.services.behavior_representation_service import align_model_feature_columns

    model_cols = ["flow_mag_paw_L_mean", "speed_mean"]
    data_cols = {"speed_mean"}
    source, nan_fill = align_model_feature_columns(model_cols, data_cols)

    assert source == ["flow_mag_paw_L_mean", "speed_mean"]
    # Not a pair-order case, so it stays a real gap: zero-filled by the caller.
    assert nan_fill == [False, False]
