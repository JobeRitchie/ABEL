"""Unified UMAP still builds when a behavior's model failed during training.

A failed behavior either has no model directory, or leaves a truncated
prediction file. The map must be built from the remaining behaviors and report
the ones left out. It must also work without the representation cache
(segment_features.parquet), which a session-subset run or disk cleanup removes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.services.evaluation_service import EvaluationService


def _seg_ids(n: int) -> list[str]:
    return [f"seg_m1_session_abc123_{i * 15}_{i * 15 + 14}" for i in range(n)]


def _model(root: Path, name: str, bid: str, probs: np.ndarray) -> Path:
    md = root / "derived" / "models" / f"behavior_model_{name}"
    md.mkdir(parents=True, exist_ok=True)
    (md / "model_state.pkl").write_bytes(b"x")
    (md / "run_settings.json").write_text(json.dumps({"target_behavior": bid}))
    pd.DataFrame({"segment_id": _seg_ids(len(probs)), "prediction_prob": probs}).to_parquet(
        md / "segment_predictions.parquet", index=False,
    )
    return md


def test_segment_meta_parsed_from_prediction_ids(tmp_path: Path) -> None:
    _model(tmp_path, "A", "bid-a", np.zeros(3))
    meta = EvaluationService._segment_meta_from_predictions(tmp_path / "derived" / "models")
    assert list(meta["session_id"]) == ["session_abc123"] * 3
    assert list(meta["animal_id"]) == ["m1"] * 3
    assert sorted(meta["start_frame"]) == [0, 15, 30]


def test_unified_umap_skips_failed_models(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n = 300
    third = n // 3
    names = {}
    for i, label in enumerate(("A", "B", "C")):
        p = rng.uniform(0.0, 0.2, n)
        p[i * third:(i + 1) * third] = rng.uniform(0.7, 1.0, third)
        _model(tmp_path, label, f"bid-{label.lower()}", p)
        names[f"bid-{label.lower()}"] = label
    # C's training failed mid-write: a truncated prediction file.
    pred_c = tmp_path / "derived" / "models" / "behavior_model_C" / "segment_predictions.parquet"
    pred_c.write_bytes(pred_c.read_bytes()[:200])
    # D never trained at all.
    names["bid-d"] = "D"

    res = EvaluationService().generate_unified_umap(tmp_path, behavior_names=names)

    assert not res.get("error"), res
    assert sorted(res["behaviors_used"]) == ["A", "B"]
    assert sorted(res["behaviors_skipped"]) == ["C", "D"]
    assert Path(res["out_path"]).exists()
