"""Prism-ready exports + the CSV fixes that make the paired stats reproducible.

The point of these tables is that a user can paste them straight into GraphPad Prism
and re-run the test themselves.  That relies on three things the old exporters got
wrong, so each gets a test: the per-seed values must survive into the CSV, the tables
must be pre-pivoted (Prism cannot pivot on import), and a nested JSON blob must never
land inside a spreadsheet cell.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from abel.validation import prism
from abel.validation.benchmark import StageTiming
from abel.validation.video_value import VideoValueResult


# ── The seeds must reach the CSV ────────────────────────────────────────────


def test_video_value_row_keeps_per_seed_f1():
    """Without the seeds a reader cannot re-run the paired test the asterisks claim."""
    res = VideoValueResult(
        project_id="P", behavior_id="b1", behavior_name="Groom", n_seeds=3,
        f1_no_video_seeds=[0.80, 0.82, 0.81],
        f1_with_video_seeds=[0.85, 0.86, 0.84],
        f1_no_video=0.81, f1_with_video=0.85, gain=0.04,
    )
    row = res.to_row()
    assert [row[f"f1_no_video_seed{i}"] for i in (1, 2, 3)] == [0.80, 0.82, 0.81]
    assert [row[f"f1_with_video_seed{i}"] for i in (1, 2, 3)] == [0.85, 0.86, 0.84]
    # The raw list columns must not leak through as Python lists.
    assert "f1_no_video_seeds" not in row


def test_benchmark_row_expands_json_and_drops_reciprocal():
    """A JSON object inside a cell is unusable in Excel/Prism; x_realtime is just 1/x."""
    t = StageTiming(
        project_id="P", stage="extract", seconds=20.0, video_seconds=700.0,
        x_realtime=0.0286, faster_than_realtime=35.0,
        breakdown=json.dumps({"preprocess": 17.4, "consolidate": 0.2}),
    )
    row = t.to_row()
    assert row["breakdown_preprocess_sec"] == 17.4
    assert row["breakdown_consolidate_sec"] == 0.2
    assert "breakdown" not in row
    assert "x_realtime" not in row          # reciprocal of faster_than_realtime
    assert row["faster_than_realtime"] == 35.0


# ── The tables must be pre-pivoted ──────────────────────────────────────────


def _ablation_frame(with_seeds: bool) -> pd.DataFrame:
    rows = []
    for budget in ("n50", "all"):
        for proj, beh in (("P1", "Groom"), ("P1", "Rear"), ("P2", "Freeze")):
            for cfg, label in (("baseline_none", "Baseline (pose only)"),
                               ("add_video_features", "+ Video features")):
                row = {"project": proj, "behavior": beh, "clip_budget": budget,
                       "config": cfg, "label": label, "f1_mean": 0.8,
                       "gain_over_baseline": 0.0 if cfg == "baseline_none" else 0.05}
                if with_seeds:
                    row |= {"f1_seed1": 0.79, "f1_seed2": 0.81}
                rows.append(row)
    return pd.DataFrame(rows)


def test_prism_ablation_splits_by_budget_and_pivots_configs():
    """4 crossed factors can't be one Prism table — it must split into one per budget."""
    tables = prism.prism_ablation(_ablation_frame(with_seeds=False))
    assert set(tables) == {"n50", "all"}
    t = tables["n50"]
    # Rows = behaviors (one row-title column), columns = configs.
    assert list(t.columns) == ["Behavior", "Baseline (pose only)", "+ Video features"]
    assert list(t["Behavior"]) == ["P1 · Groom", "P1 · Rear", "P2 · Freeze"]
    assert len(t) == 3


def test_prism_ablation_emits_seeds_as_replicate_subcolumns():
    t = prism.prism_ablation(_ablation_frame(with_seeds=True))["all"]
    assert list(t.columns) == [
        "Behavior",
        "Baseline (pose only):1", "Baseline (pose only):2",
        "+ Video features:1", "+ Video features:2",
    ]


def test_prism_kappa_drops_an_all_empty_ceiling_column():
    """An all-NaN column becomes a phantom empty dataset in Prism's graph + legend."""
    df = pd.DataFrame({
        "project": ["P1", "P1"], "behavior": ["Groom", "Rear"],
        "f1": [0.9, 0.8], "cohen_kappa": [0.85, 0.70],
        "human_ceiling_kappa": [np.nan, np.nan],
    })
    out = prism.prism_kappa(df)
    assert "Human ceiling kappa" not in out.columns
    # Present when it is actually measured.
    df["human_ceiling_kappa"] = [0.95, np.nan]
    assert "Human ceiling kappa" in prism.prism_kappa(df).columns


def test_prism_video_value_prefers_seeds_and_falls_back_to_means():
    base = {"project_id": ["P"], "behavior_name": ["Groom"],
            "f1_no_video": [0.80], "f1_with_video": [0.85], "error": [np.nan]}
    # Legacy export (no seeds) → still pasteable, means only.
    means = prism.prism_video_value(pd.DataFrame(base))
    assert list(means.columns) == ["Behavior", "Pose only (mean)", "+Video (mean)"]

    # Current export → replicate subcolumns Prism can run a paired test on.
    seeded = prism.prism_video_value(pd.DataFrame(
        base | {"f1_no_video_seed1": [0.79], "f1_no_video_seed2": [0.81],
                "f1_with_video_seed1": [0.84], "f1_with_video_seed2": [0.86]}))
    assert list(seeded.columns) == ["Behavior", "Pose only:1", "Pose only:2",
                                    "+Video:1", "+Video:2"]


def test_seed_columns_order_numerically_not_lexically():
    """seed10 must not sort between seed1 and seed2."""
    df = pd.DataFrame({f"f1_seed{i}": [0.5] for i in range(1, 12)})
    assert prism._seed_cols(df, "f1_seed") == [f"f1_seed{i}" for i in range(1, 12)]


def test_write_all_emits_readme_and_tables(tmp_path):
    written = prism.write_all(tmp_path, ablation_df=_ablation_frame(with_seeds=True))
    names = {p.name for p in written}
    assert {"prism_ablation_n50.csv", "prism_ablation_all.csv",
            "README_PRISM.txt"} <= names
    readme = (tmp_path / "prism" / "README_PRISM.txt").read_text(encoding="utf-8")
    assert "Grouped" in readme  # the table type the user must pick before pasting


# ── Absent modalities must not appear anywhere ──────────────────────────────


def test_absent_modality_is_not_reported():
    """`social` features only exist in multi-animal projects.

    A single-animal run has zero of them, and must not emit a Social legend entry or
    an all-zero Social share row — that implies ABEL measured an interaction modality
    it never measured.
    """
    from abel.validation.analyses.behaviorscape import BehaviorscapeData

    matrix = pd.DataFrame(
        {"Groom": [0.6, 0.4], "Rear": [0.5, 0.5]},
        index=["nose_velocity_mean", "dist_nose_to_tail_base"],
    )
    data = BehaviorscapeData(
        matrix=matrix,
        modality={"nose_velocity_mean": "kinematics",
                  "dist_nose_to_tail_base": "pose"},
        sources=[], pooled_members={},
    )
    # The taxonomy still has all five...
    assert "social" in data.modality_order
    # ...but only the two with features behind them are reported.
    assert data.present_modalities == ["pose", "kinematics"]

    frac = data.modality_fraction_by_behavior()
    assert "social" not in frac.columns
    assert "video" not in frac.columns

    shares = data.modality_fraction_long_df()
    assert "Social (interaction)" not in set(shares["modality_label"])
