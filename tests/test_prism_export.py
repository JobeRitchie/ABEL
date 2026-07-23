"""Prism-ready exports + the CSV fixes that make the paired stats reproducible.

The point of these tables is that a user can paste them straight into GraphPad Prism
and re-run the test themselves.  That relies on three things the old exporters got
wrong, so each gets a test: the per-seed values must survive into the CSV, the tables
must be pre-pivoted (Prism cannot pivot on import), and a nested JSON blob must never
land inside a spreadsheet cell.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from abel.validation import metrics as vmetrics
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
    assert list(t["Behavior"]) == ["P1 - Groom", "P1 - Rear", "P2 - Freeze"]
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


# ── The rest of the analyses must pre-pivot too (no hand-reformatting) ───────


def test_sig_cleanup_kills_float_dust_and_negative_zero():
    """1e-17 CI noise and -0.0 must render as a clean 0; real small values survive."""
    assert prism._sig(1.9262369477246465e-17) == 0.0
    assert prism._sig(-0.0) == 0.0
    assert prism._sig(0.11223551643754803) == 0.1122      # 4 sig figs
    assert prism._sig(0.000019262) == 1.926e-05           # genuine small value kept


def test_prism_al_curves_share_one_x_and_split_by_metric():
    df = pd.DataFrame({
        "project_id": ["P", "P", "P", "P"],
        "behavior_name": ["Rear", "Rear", "Rear", "Rear"],
        "strategy": ["active_learning", "active_learning", "random", "random"],
        "n_clips_reviewed": [20, 50, 20, 50],
        "f1_mean": [0.4, 0.6, 0.3, 0.4],
        "pr_auc_mean": [0.1, 0.2, 0.1, 0.1],
        "pos_discovered_mean": [7.0, 14.0, 3.0, 4.0],
    })
    out = prism.prism_al_curves(df)
    assert set(out) == {"prism_al_curve_f1.csv", "prism_al_curve_pr_auc.csv",
                        "prism_al_curve_pos_discovered.csv"}
    f1 = out["prism_al_curve_f1.csv"]
    assert list(f1.columns) == ["Clips reviewed", "P - Rear - AL", "P - Rear - Random"]
    assert list(f1["Clips reviewed"]) == [20, 50]


def test_prism_calibration_and_time_budget_emit_paired_xy():
    rel = pd.DataFrame({
        "project": ["P", "P", "P"], "behavior": ["Rear", "Rear", "Rear"],
        "mean_confidence": [0.1, 0.5, 0.9], "empirical_accuracy": [0.12, 0.48, 0.91],
        "count": [10, 20, 30],
    })
    c = prism.prism_calibration(rel)
    assert list(c.columns) == ["P - Rear - confidence", "P - Rear - accuracy"]

    tb = pd.DataFrame({
        "project": ["P", "P"], "behavior": ["Rear", "Rear"], "session": ["s1", "s2"],
        "true_prevalence": [0.09, 0.03], "pred_prevalence": [0.10, 0.08],
    })
    t = prism.prism_time_budget(tb)
    assert list(t.columns) == ["P - Rear - true", "P - Rear - pred"]


def test_prism_discrimination_drops_zero_baseline_from_error_reduction():
    df = pd.DataFrame({
        "project": ["P", "P"], "pair": ["A vs B", "A vs B"],
        "label": ["Pose only", "+ Video"],
        "roc_auc": [0.80, 0.95], "error_reduction": [0.0, 0.75],
    })
    out = prism.prism_discrimination(df)
    roc = out["prism_discrimination_roc_auc.csv"]
    err = out["prism_discrimination_error_reduction.csv"]
    assert list(roc.columns) == ["Pair", "Pose only", "+ Video"]
    assert list(roc["Pair"]) == ["P · A vs B"]
    # Pose-only error reduction is 0 by definition — never emit that column.
    assert "Pose only" not in err.columns
    assert list(err.columns) == ["Pair", "+ Video"]


def _landscape_frame() -> pd.DataFrame:
    """Two pairs across two assays, as `discrimination_rows` emits them."""
    rows = []
    for pair, assay, pose, fam_er in (
        ("A vs B", "Assay A", 0.70, {"+ Context / ROI": 0.10, "+ Video": 0.72}),
        ("C vs D", "Assay B", 0.95, {"+ Context / ROI": 0.60, "+ Video": 0.05}),
    ):
        rows.append({
            "project": assay.lower(), "project_name": assay, "pair": pair,
            "feature_set": "pose_only", "label": "Pose only",
            "roc_auc": pose, "pose_only_auc": pose, "headroom": 1 - pose,
            "auc_gain_vs_pose": 0.0, "p_value": float("nan"),
            "error_reduction": 0.0, "significant": "", "best_family": False,
            "n_holdout": 200,
        })
        best = max(fam_er, key=fam_er.get)
        for label, er in fam_er.items():
            fs = "pose_video" if "Video" in label else "pose_context"
            rows.append({
                "project": assay.lower(), "project_name": assay, "pair": pair,
                "feature_set": fs, "label": label,
                "roc_auc": pose + er * (1 - pose), "pose_only_auc": pose,
                "headroom": 1 - pose, "auc_gain_vs_pose": er * (1 - pose),
                "p_value": 0.001 if label == best else 0.4,
                "error_reduction": er, "significant": label == best,
                "best_family": label == best, "n_holdout": 200,
            })
    return pd.DataFrame(rows)


def test_prism_landscape_is_one_row_per_pair_with_plottable_x_and_y():
    out = prism.prism_discrimination_landscape(_landscape_frame())
    assert len(out) == 2                       # one row per pair, not per family
    assert list(out["Pair"]) == ["Assay A - A vs B", "Assay B - C vs D"]
    # The x Prism plots is pre-computed: it has no calculated columns on a scatter.
    assert abs(float(out.loc[0, "PoseOnlyError"]) - 0.30) < 1e-12
    assert set(out["BestFamily"]) == {"Video", "Context / ROI"}
    # Numeric flag, not TRUE/FALSE text -- Prism groups on numbers.
    assert list(out["Significant"]) == [1, 1]
    assert out["Significant"].dtype.kind in "iu"
    # A CI half-width has no Prism input format and must never be exported.
    assert not [c for c in out.columns if "ci" in c.lower()]


def test_prism_volcano_keeps_every_family_and_drops_the_baseline():
    out = prism.prism_discrimination_volcano(_landscape_frame())
    assert len(out) == 4                       # 2 pairs x 2 add-on families
    assert "Pose only" not in set(out["FeatureFamily"])
    # The same pair appears once per family -- that is the point of the panel.
    assert list(out["Pair"]).count("Assay A - A vs B") == 2
    # y = -log10(p), pre-computed so the scatter is a straight paste.
    winner = out[(out["Pair"] == "Assay A - A vs B") & (out["FeatureFamily"] == "Video")]
    assert abs(float(winner["NegLog10P"].iloc[0]) - 3.0) < 1e-9
    loser = out[(out["Pair"] == "Assay A - A vs B")
                & (out["FeatureFamily"] == "Context / ROI")]
    assert float(loser["NegLog10P"].iloc[0]) < 1.0


def test_prism_seeds_pad_every_family_to_the_same_subcolumn_count():
    """Prism assigns replicate subcolumns positionally: a short block shifts
    every dataset after it one column left."""
    seed_df = pd.DataFrame([
        {"project": "p", "project_name": "Assay A", "pair": "A vs B",
         "feature_set": "pose_only", "label": "Pose only",
         "seed_index": i + 1, "roc_auc": 0.70 + i * 0.01} for i in range(3)
    ] + [
        {"project": "p", "project_name": "Assay A", "pair": "A vs B",
         "feature_set": "pose_video", "label": "+ Video",
         "seed_index": i + 1, "roc_auc": 0.90 + i * 0.01} for i in range(2)
    ])
    out = prism.prism_discrimination_seeds(seed_df)
    assert list(out.columns) == ["Pair", "Pose only:1", "Pose only:2", "Pose only:3",
                                 "Video:1", "Video:2", "Video:3"]
    assert math.isnan(float(out.loc[0, "Video:3"]))   # padded, not shifted
    assert prism.prism_discrimination_seeds(pd.DataFrame()).empty


def test_tiny_p_values_survive_the_dust_collapse(tmp_path):
    """_sig zeroes anything under 1e-9 as float dust. That is right for a CI
    half-width of 1e-17 and wrong for a p of 9e-10, which would export as "0" —
    a value a p-value can never take."""
    df = _landscape_frame()
    df.loc[df["best_family"], "p_value"] = 9.063e-10
    out = prism.prism_discrimination_landscape(df)
    path = prism._write(out, tmp_path / "land.csv")
    text = path.read_text(encoding="utf-8-sig")
    assert "9.063e-10" in text
    # The magnitude columns beside it are still dust-collapsed as before.
    dusty = pd.DataFrame({"Pair": ["A"], "ci95": [1e-17], "PValue": [1e-17]})
    cleaned = prism._clean(dusty)
    assert float(cleaned.loc[0, "ci95"]) == 0.0
    assert float(cleaned.loc[0, "PValue"]) == 1e-17


def test_prism_discrimination_scatters_decline_a_legacy_frame():
    """A run written before these columns existed must yield nothing, not raise."""
    old = pd.DataFrame({
        "project": ["P"], "pair": ["A vs B"], "label": ["+ Video"],
        "roc_auc": [0.9], "error_reduction": [0.5],
    })
    assert prism.prism_discrimination_landscape(old).empty
    assert prism.prism_discrimination_volcano(old).empty


def test_prism_accuracy_by_behavior_emits_one_row_title_and_a_plottable_sd():
    """Prism takes one row-title column and cannot plot a CI half-width.

    So project+behavior merge into a single title, and f1_ci is converted to the
    SD Prism actually draws error bars from. Pasting the raw CI into an SD
    subcolumn would overstate every bar by t(n)/sqrt(n).
    """
    df = pd.DataFrame({"project_id": ["P"], "behavior_name": ["Rear"],
                       "f1_mean": [0.85], "f1_ci": [0.02], "n": [10]})
    out = prism.prism_accuracy_by_behavior(df)
    assert list(out.columns) == ["Behavior", "F1", "SD", "N"]
    assert out["Behavior"][0] == "P - Rear"
    # Round-trip: the emitted SD must reproduce the input CI via metrics.ci95.
    sd = float(out["SD"][0])
    n = 10
    assert sd == pytest.approx(0.02 / vmetrics.t_critical_95(n) * math.sqrt(n))
    assert vmetrics.t_critical_95(n) * sd / math.sqrt(n) == pytest.approx(0.02)


def test_prism_sd_is_blank_for_a_single_seed():
    """One seed has no estimable spread; the SD must stay blank, not read as 0.0."""
    df = pd.DataFrame({"project_id": ["P"], "behavior_name": ["Rear"],
                       "f1_mean": [0.85], "f1_ci": [0.0], "n": [1]})
    out = prism.prism_accuracy_by_behavior(df)
    assert pd.isna(out["SD"][0])


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


# ── Nothing exported may turn to gibberish in Excel/Prism on Windows ─────────


def test_written_files_are_ascii_with_a_bom(tmp_path):
    """The mojibake guard, end to end.

    Prism and Excel on Windows decode a CSV with the ANSI code page unless it opens
    with a BOM, so a UTF-8 "≥" arrives as "â‰¥" and a header like "F1≥0.70:1" lands
    in the data table as gibberish. Two defences are asserted here: the symbols this
    package chooses are transliterated to ASCII, and the file carries a BOM so any
    non-ASCII the *user* typed still survives.
    """
    df = pd.DataFrame({
        "Behavior": ["P · Rear"],          # middle dot, from _row_title
        "F1≥0.80:1": [0.5],                # >=, from the effort-to-quality targets
        "ΔF1 — video": [0.25],             # delta + em dash
    })
    path = prism._write(df, tmp_path / "t.csv")

    raw = path.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf", "missing BOM: Windows will decode as cp1252"
    text = raw.decode("utf-8-sig")
    assert text.isascii(), [c for c in text if ord(c) > 127]
    header = text.splitlines()[0]
    assert "F1>=0.80:1" in header and "-" in header and "≥" not in header

    # Round-trips through the reader the rest of the suite uses.
    back = pd.read_csv(path, encoding="utf-8-sig")
    assert list(back.columns)[0] == "Behavior"


def test_user_typed_non_ascii_survives_via_the_bom(tmp_path):
    """We transliterate OUR symbols, never the user's text.

    A behavior named with an accent must come back intact rather than being
    mangled into ASCII -- that is what the BOM is for.
    """
    df = pd.DataFrame({"Behavior": ["Café approach"], "F1": [0.5]})
    path = prism._write(df, tmp_path / "u.csv")
    back = pd.read_csv(path, encoding="utf-8-sig")
    assert back["Behavior"][0] == "Café approach"


def test_readme_keeps_its_indentation_through_transliteration(tmp_path):
    """_ascii also runs over multi-line README prose; it must not touch whitespace."""
    text = "Title\n=====\nfile.csv\n    Table: XY — 3 subcolumns.\n"
    path = prism.write_text(tmp_path / "README.txt", text)
    out = path.read_text(encoding="utf-8-sig")
    assert out.splitlines()[3] == "    Table: XY - 3 subcolumns."
    assert out.isascii()
