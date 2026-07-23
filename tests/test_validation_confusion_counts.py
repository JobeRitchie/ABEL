"""Held-out confusion counts as a first-class validation output.

F1 and PR-AUC are the comparable numbers; TP/FN/FP are the ones a reviewer can
check against their own scoring experience. Promoting them to a table, a figure,
a Prism export and a plain-language finding only helps if the counts are *right*,
so each test here pins one way a count table quietly lies:

* summing across seeds (advertising an ``n`` the study never had),
* counts and rates computed over different cells,
* letting deliberately handicapped cells into the headline,
* or losing the "these are windows, not bouts" caveat.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from abel.validation import findings as fnd
from abel.validation import holdout as vhold
from abel.validation import meta_summary as ms
from abel.validation import plots
from abel.validation import prism
from abel.validation.analyses import cross_project as cp
from abel.validation.datamodel import ProjectRef


def _cell(analysis: str, config: str, seed: int, *, tp: int, fp: int, fn: int,
          tn: int, f1: float = 0.9, behavior: str = "Rear", project: str = "EPM",
          n_clips: int = 100, error: str = "") -> dict:
    return {
        "project_id": project, "project_name": project,
        "behavior_id": f"b-{behavior}", "behavior_name": behavior,
        "analysis": analysis, "config_name": config, "n_clips": n_clips,
        "seed": seed, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "error": error,
    }


def _cells() -> pd.DataFrame:
    """Three seeds of one behavior, plus cells that must NOT reach the headline."""
    rows = [
        # The honest held-out split: same pool, three seeds.
        _cell("generalization", "production", 0, tp=190, fp=17, fn=24, tn=1200),
        _cell("generalization", "production", 1, tp=192, fp=17, fn=22, tn=1200),
        _cell("generalization", "production", 2, tp=191, fp=17, fn=23, tn=1200),
        # Deliberately handicapped: a pose-only baseline and a 10-clip learning
        # curve point. Both would drag the headline counts down if included.
        _cell("ablation", "baseline_none", 0, tp=40, fp=200, fn=174, tn=1000),
        _cell("learning_curve", "production", 0, tp=12, fp=90, fn=202, tn=1100,
              n_clips=10),
    ]
    return pd.DataFrame(rows)


# ── the aggregation ─────────────────────────────────────────────────────────


def test_counts_are_averaged_over_seeds_not_summed():
    """Every seed re-scores the same held-out pool; summing triples the evidence."""
    out = cp.confusion_by_behavior(_cells())
    assert len(out) == 1
    row = out.iloc[0]
    # mean(190, 192, 191) = 191 — not 573.
    assert row["tp"] == 191
    assert row["fn"] == 23
    assert row["fp"] == 17
    assert row["n_pos_val"] == 214
    assert row["n_cells"] == 3
    # n_val closes the 2x2 rather than being an independent guess.
    assert row["n_val"] == row["tp"] + row["fp"] + row["fn"] + row["tn"]


def test_handicapped_cells_are_excluded_from_headline_counts():
    """Counts must come from the same cells as the reported F1 (`_headline_cells`)."""
    full = cp.confusion_by_behavior(_cells())
    gen_only = cp.confusion_by_behavior(
        _cells()[lambda d: d["analysis"] == "generalization"])
    assert full.iloc[0]["tp"] == gen_only.iloc[0]["tp"]
    # And the same selection drives accuracy_by_behavior, so the two agree on n.
    acc = cp.accuracy_by_behavior(_cells())
    assert int(acc.iloc[0]["n"]) == int(full.iloc[0]["n_cells"])


def test_rates_are_recomputed_from_the_printed_counts():
    """A reader who divides the columns must get the printed rate back."""
    row = cp.confusion_by_behavior(_cells()).iloc[0]
    assert math.isclose(row["recall"], row["tp"] / (row["tp"] + row["fn"]),
                        rel_tol=1e-6)
    assert math.isclose(row["precision"], row["tp"] / (row["tp"] + row["fp"]),
                        rel_tol=1e-6)


def test_errored_cells_do_not_contribute():
    df = pd.concat([_cells(), pd.DataFrame([
        _cell("generalization", "production", 3, tp=0, fp=0, fn=0, tn=0,
              error="fit failed")])], ignore_index=True)
    assert cp.confusion_by_behavior(df).iloc[0]["tp"] == 191


def test_empty_and_countless_inputs_return_a_shaped_frame():
    """A run without counts must yield an empty *typed* frame, not an exception."""
    for df in (pd.DataFrame(),
               _cells().drop(columns=["tp", "fp", "fn", "tn"])):
        out = cp.confusion_by_behavior(df)
        assert out.empty
        assert {"tp", "fn", "fp", "tn", "n_val", "n_pos_val"} <= set(out.columns)


# ── the summary tables ──────────────────────────────────────────────────────


def _summary_sources() -> dict[str, pd.DataFrame]:
    conf = pd.DataFrame({
        "project_id": ["EPM", "EPM", "OFT"],
        "behavior_name": ["Rear", "Groom", "Rear"],
        "tp": [191, 40, 300], "fp": [17, 25, 10],
        "fn": [23, 60, 20], "tn": [1200, 900, 2000],
        "n_val": [1431, 1025, 2330], "n_pos_val": [214, 100, 320],
        "precision": [0.918, 0.615, 0.968], "recall": [0.893, 0.400, 0.938],
        "n_cells": [3, 3, 3],
    })
    return {
        "publication_metrics": pd.DataFrame({
            "project_id": ["EPM", "OFT"], "project_name": ["EPM", "OFT"],
            "f1": [0.86, 0.94], "cohen_kappa": [0.72, 0.89],
        }),
        "accuracy_by_behavior": pd.DataFrame({
            "project_id": ["EPM", "EPM", "OFT"],
            "behavior_name": ["Rear", "Groom", "Rear"],
            "f1_mean": [0.90, 0.48, 0.95],
        }),
        "confusion_by_behavior": conf,
    }


def test_per_behavior_table_carries_the_n_behind_each_f1():
    out = ms.summary_per_behavior(_summary_sources())
    assert {"n_val", "n_pos_val", "tp", "fn", "fp", "tn"} <= set(out.columns)
    # Assay scoping survives: EPM·Rear and OFT·Rear stay separate rows.
    epm_rear = out[(out["assay"] == "EPM") & (out["behavior"] == "Rear")].iloc[0]
    oft_rear = out[(out["assay"] == "OFT") & (out["behavior"] == "Rear")].iloc[0]
    assert epm_rear["tp"] == 191 and oft_rear["tp"] == 300


def test_per_assay_totals_sum_across_behaviors_and_omit_tn():
    """Summing across behaviors is meaningful (each brings its own positives);
    an assay-level accuracy off TN would be ~0.99 by imbalance alone."""
    out = ms.summary_per_assay(_summary_sources())
    epm = out[out["assay"] == "EPM"].iloc[0]
    assert epm["tp_total"] == 231          # 191 + 40
    assert epm["fn_total"] == 83           # 23 + 60
    assert epm["n_pos_val_total"] == 314
    assert not any(c.startswith("tn") for c in out.columns)


def test_summary_survives_a_run_without_the_confusion_table():
    src = _summary_sources()
    src.pop("confusion_by_behavior")
    assert "tp" not in ms.summary_per_behavior(src).columns
    assert not ms.summary_per_assay(src).empty


# ── the exports and the narrative ───────────────────────────────────────────


def test_prism_confusion_orders_columns_for_a_stacked_bar():
    t = prism.prism_confusion(_summary_sources()["confusion_by_behavior"])
    assert list(t.columns)[:4] == ["Behavior", "Found (TP)", "Missed (FN)",
                                   "False alarm (FP)"]
    # TN is last so it falls outside the plotted range of a 3-column stack.
    assert list(t.columns).index("True negative (TN)") > 3
    assert t.iloc[0]["Behavior"] == "EPM · Rear"


def test_findings_state_the_result_in_counts_and_name_the_unit():
    items = fnd._confusion_findings(fnd.FindingsInput(cells=_cells()))
    assert items, "counts must produce a finding when cells carry them"
    headline = items[0].headline
    assert "191" in headline and "214" in headline and "17" in headline

    caveats = [i for i in items if i.kind == fnd.KIND_CAVEAT]
    assert caveats, "the clips-not-bouts caveat is not optional"
    assert "bout" in caveats[0].headline.lower()
    # No hard-coded clip duration anywhere in the narrative: clip length is
    # per-project (most use ~0.5 s, the schema default is 60 frames), so a fixed
    # phrase would misdescribe most runs.
    text = " ".join(i.headline + i.detail for i in items)
    assert "15-frame" not in text and "2-second" not in text


def test_findings_are_silent_without_counts():
    assert fnd._confusion_findings(fnd.FindingsInput(cells=pd.DataFrame())) == []


# ── the clip length behind the counts ───────────────────────────────────────


def test_clip_length_is_measured_from_rows_not_read_from_config():
    """`segment_window_frames` defaults to 60 but most projects run ~0.5 s clips,
    so trusting the config default would overstate the unit ~4x."""
    df = pd.DataFrame({"start_frame": [0, 15, 30, 45],
                       "end_frame": [14, 29, 44, 59]})
    assert vhold.median_clip_frames(ProjectRef(project_id="P", name="P",
                                               root=Path("."), ), df) == 15.0


def test_clip_length_is_nan_when_unmeasurable():
    """A missing or bound-less training set must yield no duration at all —
    callers then name the unit without one rather than inventing a number."""
    proj = ProjectRef(project_id="P", name="P", root=Path("nonexistent"))
    assert math.isnan(vhold.median_clip_frames(proj))
    assert math.isnan(vhold.median_clip_frames(proj, pd.DataFrame({"a": [1]})))


def test_clip_unit_label_degrades_rather_than_guessing():
    assert vhold.clip_unit_label(15, 30) == "labeled clips (~0.5 s)"
    assert vhold.clip_unit_label(60, 30) == "labeled clips (~2.0 s)"
    # No fps → frames, which needs no assumption; no measurement → bare unit.
    assert vhold.clip_unit_label(15, 0) == "labeled clips (~15 frames)"
    assert vhold.clip_unit_label(float("nan"), 30) == "labeled clips"


def test_figure_prints_a_duration_only_when_the_run_agrees_on_one():
    """clip_sec is per-project; one assay's clip length must not be attributed
    to the whole figure."""
    base = pd.DataFrame({"project_id": ["EPM", "OFT"],
                         "behavior_name": ["Rear", "Rear"],
                         "tp": [10, 20], "fp": [1, 2], "fn": [3, 4],
                         "tn": [100, 200]})
    same = base.assign(clip_sec=[0.5, 0.5])
    mixed = base.assign(clip_sec=[0.5, 2.0])
    assert plots._clip_unit(same) == "clips (~0.5 s)"
    assert plots._clip_unit(mixed) == "clips"
    assert plots._clip_unit(base) == "clips"
