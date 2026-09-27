"""Social-feature layer + pipeline-parity fixes in the validation suite."""

from __future__ import annotations

import numpy as np
import pandas as pd

from abel.validation import holdout, social_value, subsample
from abel.validation.engine import _carve_calibration_slice


def _pool(n=60, social=True, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "segment_id": [f"s{i}" for i in range(n)],
        "session_id": [f"sess{i % 6}" for i in range(n)],
        "label": rng.choice(["Chase", "Flee", "no_behavior"], size=n),
        "nose_x_mean": rng.normal(size=n),
        "body_speed_mean": rng.normal(size=n),
        "flow_mag_mean": rng.normal(size=n),
    })
    if social:
        df["social_dist_centroid_to_centroid_nearest_mean"] = rng.normal(size=n)
    return df


def test_positive_mask_counts_pipe_labels_only_when_co_occurring():
    df = pd.DataFrame({"label": ["Chase", "Chase|Flee", "Flee", "no_behavior"]})
    assert subsample.count_positives(df, "Chase") == 1
    assert subsample.count_positives(df, "Chase", co_occurring=True) == 2
    assert subsample.count_positives(df, "Flee", co_occurring=True) == 2


def test_calibration_slice_sees_pipe_positives():
    df = pd.DataFrame({
        "session_id": [f"g{i // 10}" for i in range(100)],
        "label": ["A|B" if i % 3 == 0 else "B" for i in range(100)],
    })
    _, cal = _carve_calibration_slice(df, "session_id", 0, "A")
    assert cal.empty
    _, cal = _carve_calibration_slice(df, "session_id", 0, "A", co_occurring=True)
    assert not cal.empty


def test_social_layer_gates_on_informative_social_columns():
    assert social_value.has_social_features(_pool(social=True))
    assert not social_value.has_social_features(_pool(social=False))
    flat = _pool(social=True)
    flat["social_dist_centroid_to_centroid_nearest_mean"] = 0.0
    assert not social_value.has_social_features(flat)


def test_social_arms_differ_only_by_social_family():
    class P:
        use_video_features = True
    off, on = social_value.arm_columns(_pool(), P())
    assert set(on) - set(off) == {"social_dist_centroid_to_centroid_nearest_mean"}
    assert "flow_mag_mean" in off


def test_load_training_frame_canonicalizes_distance_spellings(tmp_path):
    class P:
        training_set_path = tmp_path / "training_set.parquet"
    raw = pd.DataFrame({
        "segment_id": ["a", "b"], "label": ["X", "Y"],
        "dist_nose_to_tail_base_mean": [1.0, np.nan],
        "dist_tail_base_to_nose_mean": [np.nan, 2.0],
    })
    raw.to_parquet(P.training_set_path)
    df = holdout.load_training_frame(P())
    dist = [c for c in df.columns if c.startswith("dist_")]
    assert len(dist) == 1 and df[dist[0]].notna().all()


def test_dropped_siblings_are_not_negatives():
    df = pd.DataFrame({"label": ["A", "A|B", "B|C", "no_behavior", "C"]})
    assert subsample.dropped_sibling_mask(df, "A", co_occurring=True).tolist() == [
        False, False, True, False, False]
    sub, n_pos, n_neg = subsample.draw(df, "A", subsample.ALL_CLIPS, co_occurring=True)
    assert (n_pos, n_neg) == (2, 2)
    assert not subsample.dropped_sibling_mask(df, "A").any()


def test_subject_split_keeps_dyad_together_without_manifest(tmp_path):
    class P:
        root = tmp_path
        split_strategy = "group_shuffle_subject"
        training_set_path = tmp_path / "none.parquet"
        project_id = "p"
    rows = [{"segment_id": f"{s}_{a}_{i}", "session_id": f"sess{s}", "animal_id": a,
             "label": "A" if i % 2 else "no_behavior", "reviewer_confidence": 1.0}
            for s in range(8) for a in ("track_0", "track_1") for i in range(4)]
    sp = holdout.split(P(), df=pd.DataFrame(rows), seed=1)
    assert not set(sp.train_pool["session_id"]) & set(sp.holdout["session_id"])
