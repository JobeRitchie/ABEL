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
