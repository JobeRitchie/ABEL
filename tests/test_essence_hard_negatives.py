"""Clip Mining folds clips the user rejected into the essence background."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from abel.services.clip_metrics_service import ClipMetricsService


def _project(tmp_path):
    lab = tmp_path / "derived" / "review_labels"
    lab.mkdir(parents=True)
    pd.DataFrame({
        "segment_id": ["ex1", "ex2", "m1", "m2", "m3", "m4", "al1"],
        "review_label": ["Rear", "Rear|Groom", "Walk", "Rear", "not_Rear", "ambiguous", "Walk"],
    }).to_parquet(lab / "reviewer_labels.parquet")
    rt = tmp_path / "derived" / "review_tables"
    rt.mkdir(parents=True)
    cands = [{"window_id": w, "source": "clip_mining"} for w in ("m1", "m2", "m3", "m4")]
    cands.append({"window_id": "al1", "source": "active_learning_uncertainty"})
    (rt / "external_window_candidates.json").write_text(json.dumps({"windows": cands}))
    return tmp_path


def test_rejected_mining_clips_are_mined_non_target_reviews(tmp_path):
    svc = ClipMetricsService()
    svc.set_project(_project(tmp_path))
    got = sorted(svc.rejected_mining_clips(["ex1", "ex2"]))
    # m2 is a hit, m4 only ambiguous, al1 did not come from mining
    assert got == ["m1", "m3"]


def test_with_hard_negatives_weights_and_caps():
    bg = pd.DataFrame(np.zeros((100, 2)), columns=["a", "b"])
    neg = pd.DataFrame(np.ones((5, 3)), columns=["a", "b", "extra"])
    out = ClipMetricsService.with_hard_negatives(bg, neg)
    assert list(out.columns) == ["a", "b"]
    assert len(out) == 100 + 5 * 5          # 100 // (4 * 5) = 5 repeats
    assert ClipMetricsService.with_hard_negatives(bg, None) is bg
