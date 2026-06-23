"""Tests for the Transfer Feedback service (Direct Use transfer quality)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from abel.services.transfer_feedback_service import TransferFeedbackService


def _write_cache(root: Path, rows: list[dict]) -> None:
    d = root / "derived" / "analytics_cache"
    d.mkdir(parents=True, exist_ok=True)
    (d / "analytics_cache.json").write_text(
        json.dumps({"version": 2, "summary_rows": rows}), encoding="utf-8"
    )


def _row(subject, behavior, n_bouts, time_s, sid=None):
    return {
        "session_id": sid or f"sess_{subject}",
        "subject": subject,
        "behavior": behavior,
        "behavior_id": behavior,
        "n_bouts": float(n_bouts),
        "time_spent_s": float(time_s),
        "mean_bout_s": float(time_s) / n_bouts if n_bouts else 0.0,
    }


def _normal_population(root: Path, n=20):
    """20 'normal' subjects with consistent per-behavior numbers + 1 outlier."""
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        s = f"N{i:02d}"
        rows.append(_row(s, "Walk", 45 + rng.integers(-3, 3), 250 + rng.integers(-10, 10)))
        rows.append(_row(s, "Rear", 48 + rng.integers(-3, 3), 70 + rng.integers(-5, 5)))
        rows.append(_row(s, "Groom", 16 + rng.integers(-2, 2), 60 + rng.integers(-5, 5)))
    # Clear outlier: almost no Walk/Rear.
    rows.append(_row("BAD1", "Walk", 1, 5))
    rows.append(_row("BAD1", "Rear", 0, 0))
    rows.append(_row("BAD1", "Groom", 16, 60))
    _write_cache(root, rows)


def test_no_analytics_warns(tmp_path: Path) -> None:
    rep = TransferFeedbackService().analyze(tmp_path / "empty")
    assert rep.subjects == []
    assert rep.warnings and "analytics" in rep.warnings[0].lower()


def test_outlier_ranks_worst_and_zero_bout_flagged(tmp_path: Path) -> None:
    root = tmp_path / "du"
    _normal_population(root)
    rep = TransferFeedbackService().analyze(root)
    assert rep.subjects
    worst = rep.subjects[0]
    assert worst.subject == "BAD1"
    assert worst.category in ("Poor", "Warning")
    assert worst.health_score < 80
    # zero/single-bout flags fire for Walk and Rear.
    assert any("Walk" in f for f in worst.flags)
    assert any("Rear" in f for f in worst.flags)
    # A normal subject is Good with no flags.
    normals = [s for s in rep.subjects if s.subject.startswith("N")]
    assert all(s.category == "Good" for s in normals)


def test_population_summary_counts(tmp_path: Path) -> None:
    root = tmp_path / "du"
    _normal_population(root)
    rep = TransferFeedbackService().analyze(root)
    pop = rep.population
    assert pop["n_subjects"] == 21
    assert pop["n_good"] >= 20
    assert pop["n_poor"] + pop["n_warning"] >= 1


def test_confidence_runs_from_traces(tmp_path: Path) -> None:
    """A stuck-high trace should produce a long high-confidence run + flag."""
    root = tmp_path / "du"
    _normal_population(root)
    # Build a trace for BAD1 that is stuck at prob 0.99 for 4000 frames (>60s @30fps).
    inf = root / "derived" / "temporal_refinement" / "_inf"
    inf.mkdir(parents=True, exist_ok=True)
    n = 4000
    trace = pd.DataFrame({
        "frame": np.arange(n),
        "prob_Walk": np.full(n, 0.99),
        "prob_Rear": np.zeros(n),
    })
    tp = inf / "BAD1.parquet"
    trace.to_parquet(tp, index=False)
    (inf / "inference_manifest.json").write_text(
        json.dumps({"trace_paths": {"sess_BAD1": str(tp)}}), encoding="utf-8"
    )
    tb = root / "derived" / "temporal_refinement" / "target_behavior"
    tb.mkdir(parents=True, exist_ok=True)
    (tb / "latest.json").write_text(json.dumps({"inference_dir": str(inf)}), encoding="utf-8")
    # Manifest mapping session→subject.
    rt = root / "derived" / "review_tables"
    rt.mkdir(parents=True, exist_ok=True)
    (rt / "import_manifest.json").write_text(
        json.dumps({"linked_sessions": [{"session_id": "sess_BAD1", "subject_id": "BAD1"}]}),
        encoding="utf-8",
    )

    rep = TransferFeedbackService().analyze(root, fps=30.0)
    assert rep.has_traces
    bad = [s for s in rep.subjects if s.subject == "BAD1"][0]
    assert bad.confidence["longest_high_run_s"] > 60
    assert any("stuck high" in f.lower() for f in bad.flags)
