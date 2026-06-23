"""Tests for folding imported examples into the unified UMAP.

The full ``generate_unified_umap`` render needs umap-learn; these target the two
new helpers that locate imported training rows and score them through each model
into the same prob-space, which is where all the new logic lives.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from abel.services.evaluation_service import EvaluationService


FEATS = [f"feat_{i}" for i in range(5)]


def _training_set(root: Path, rows: list[tuple[str, str, str]]) -> None:
    """rows: list of (segment_id, label, label_source)."""
    ts_dir = root / "derived" / "training_sets"
    ts_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    data: dict = {
        "segment_id": [r[0] for r in rows],
        "label": [r[1] for r in rows],
        "label_source": [r[2] for r in rows],
        "session_id": ["s"] * n,
        "start_frame": list(range(n)),
        "end_frame": [i + 14 for i in range(n)],
        "animal_id": ["a"] * n,
    }
    for j, c in enumerate(FEATS):
        data[c] = [float((i + j) % 7) for i in range(n)]
    pd.DataFrame(data).to_parquet(ts_dir / "training_set.parquet", index=False)


def _make_model_dir(root: Path, name: str) -> Path:
    """A real (tiny) binary sklearn model pickled in the expected payload shape."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, len(FEATS)))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    clf = LogisticRegression().fit(x, y)
    md = root / "derived" / "models" / f"behavior_model_{name}"
    md.mkdir(parents=True, exist_ok=True)
    with open(md / "model_state.pkl", "wb") as f:
        pickle.dump({"model": clf, "feature_cols": list(FEATS)}, f)
    return md


def test_load_imported_training_rows_filters_to_imports(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _training_set(root, [
        ("seg_own_1", "Freeze", "reviewer"),
        ("seg_own_2", "no_behavior", "seed"),
        ("CAB__seg_x", "Freeze", "imported:CAB"),
        ("CAB__seg_y", "no_behavior", "imported:CAB"),
    ])
    imp = EvaluationService._load_imported_training_rows(root)
    assert list(imp["segment_id"]) == ["CAB__seg_x", "CAB__seg_y"]
    assert set(imp["label_source"]) == {"imported:CAB"}


def test_load_imported_training_rows_empty_when_none(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _training_set(root, [("seg_own_1", "Freeze", "reviewer")])
    assert EvaluationService._load_imported_training_rows(root).empty
    # No training set at all → empty, no error.
    assert EvaluationService._load_imported_training_rows(tmp_path / "missing").empty


def test_embed_imported_segments_scores_into_prob_space(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    md_freeze = _make_model_dir(root, "Freeze")
    md_groom = _make_model_dir(root, "Groom")

    imported_df = pd.DataFrame({
        "segment_id": ["CAB__s1", "CAB__s2", "CAB__s3"],
        "label": ["bid-freeze", "bid-groom", "no_behavior"],
        "session_id": ["CAB__sess"] * 3,
        "start_frame": [0, 10, 20],
        "end_frame": [14, 24, 34],
        "animal_id": ["a"] * 3,
        **{c: [float(i) for i in range(3)] for c in FEATS},
    })
    meta_cols = ["segment_id", "session_id", "start_frame", "end_frame", "animal_id"]
    bid_list = ["Freeze", "Groom"]
    prob_cols = ["prob_Freeze", "prob_Groom"]
    latest = {"Freeze": md_freeze, "Groom": md_groom}

    out, label_map = EvaluationService._embed_imported_segments(
        imported_df, bid_list, latest, meta_cols, prob_cols,
    )
    # Shape: one row per imported segment, meta + prob columns present.
    assert len(out) == 3
    assert set(meta_cols + prob_cols).issubset(out.columns)
    # Probabilities are real model outputs in [0, 1], not the zero fallback.
    for col in prob_cols:
        vals = out[col].to_numpy(dtype=float)
        assert ((vals >= 0.0) & (vals <= 1.0)).all()
        assert np.any(vals > 0.0)
    # Label map carries each imported segment's label through verbatim.
    assert label_map == {
        "CAB__s1": "bid-freeze", "CAB__s2": "bid-groom", "CAB__s3": "no_behavior",
    }


def test_embed_imported_segments_missing_model_yields_zeros(tmp_path: Path) -> None:
    """A behaviour whose model dir is absent must not crash — its prob is 0."""
    root = tmp_path / "proj"
    md_freeze = _make_model_dir(root, "Freeze")
    imported_df = pd.DataFrame({
        "segment_id": ["CAB__s1", "CAB__s2"],
        "label": ["bid-freeze", "no_behavior"],
        "session_id": ["CAB__sess"] * 2,
        "start_frame": [0, 10],
        "end_frame": [14, 24],
        "animal_id": ["a"] * 2,
        **{c: [1.0, 2.0] for c in FEATS},
    })
    out, _ = EvaluationService._embed_imported_segments(
        imported_df,
        bid_list=["Freeze", "Missing"],
        latest_by_behavior={"Freeze": md_freeze},  # no "Missing"
        meta_cols=["segment_id", "session_id", "start_frame", "end_frame", "animal_id"],
        prob_cols=["prob_Freeze", "prob_Missing"],
    )
    assert (out["prob_Missing"].to_numpy(dtype=float) == 0.0).all()
    assert np.any(out["prob_Freeze"].to_numpy(dtype=float) > 0.0)
