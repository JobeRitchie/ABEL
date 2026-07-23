"""Meta-level summary tables — the spine of the one-figure manuscript story.

A full validation run emits ~189 figures and ~117 tidy CSVs: one learning curve,
one reliability diagram, one Bland-Altman per behavior per assay.  That granularity
is right for the archive and wrong for a paper, which needs a single multi-panel
figure backed by a handful of tables.  This module distils the exhaustive per-behavior
exports into five small tables — one row per assay, or per behavior, or per feature
family — that a person can plot directly:

    summary_per_assay.csv            8 rows  — headline accuracy + counts + κ/ECE/CCC
    summary_per_behavior.csv        43 rows  — the master supplementary table
    summary_feature_value.csv       14 rows  — ΔF1 per enhancement × clip budget
    summary_discrimination.csv       3 rows  — error removed by feature family
    summary_active_learning_curve.csv        — pooled positives-found curve

Each builder is defensive: a summary is emitted only from the source tables that are
present, and any absent join column is left NaN rather than sinking the whole export.
The tables are assay-scoped throughout — same-named behaviors from different assays
stay separate rows (see :func:`abel.validation.plots.pool_generalization_by_behavior`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abel.validation.metrics import ci95

# Logical name -> (run-dir subdir, filename). The bundle flattens these to
# ``<subdir>__<filename>`` in a single data/ folder; both loaders below use this map.
_SOURCES: dict[str, tuple[str, str]] = {
    "publication_metrics": ("cross_project", "publication_metrics.csv"),
    "accuracy_by_behavior": ("cross_project", "accuracy_by_behavior.csv"),
    "confusion_by_behavior": ("cross_project", "confusion_by_behavior.csv"),
    "generalization": ("generalization", "agreement.csv"),
    "calibration": ("calibration", "calibration.csv"),
    "time_budget": ("time_budget", "time_budget_agreement.csv"),
    "video_value": ("video_value", "video_value.csv"),
    "ablation": ("ablation", "ablation_results.csv"),
    "confusable_pairs": ("discrimination", "confusable_pairs.csv"),
    "al_summary": ("active_learning", "al_vs_random_summary.csv"),
    "al_points": ("active_learning", "al_vs_random_points.csv"),
}

# Full ablation config gets one label per enhancement; the pose-only reference has
# no ΔF1 of its own and is excluded from the feature-value table.
_BASELINE_LABEL = "Baseline (pose only)"
_ALL_ENH_CONFIG = "all_features"
# A pair the pose-only model already separates near-perfectly has no headroom for
# context/video to help; the discrimination summary is over pairs that do.
_HEADROOM_AUC = 0.998


# ── loaders ─────────────────────────────────────────────────────────────────


def _read(path: Path) -> pd.DataFrame | None:
    try:
        # utf-8-sig strips the BOM store.write_csv emits, and reads BOM-less
        # UTF-8 (older runs) unchanged -- otherwise the first column of every
        # source table would come back named "﻿project".
        return pd.read_csv(path, encoding="utf-8-sig") if path.is_file() else None
    except Exception:  # noqa: BLE001 — a corrupt source must not sink the summary
        return None


def load_run_dir(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the source tables from a run directory (``<run>/<subdir>/<name>.csv``)."""
    run_dir = Path(run_dir)
    out: dict[str, pd.DataFrame] = {}
    for key, (subdir, name) in _SOURCES.items():
        df = _read(run_dir / subdir / name)
        if df is not None:
            out[key] = df
    return out


def load_bundle(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load from a flattened bundle folder (``data/<subdir>__<name>.csv``)."""
    data_dir = Path(data_dir)
    out: dict[str, pd.DataFrame] = {}
    for key, (subdir, name) in _SOURCES.items():
        df = _read(data_dir / f"{subdir}__{name}")
        if df is not None:
            out[key] = df
    return out


# ── per-assay ───────────────────────────────────────────────────────────────


def _group_mean(df: pd.DataFrame | None, key: str, value: str,
                out_name: str) -> pd.DataFrame | None:
    if df is None or key not in df.columns or value not in df.columns:
        return None
    g = (df.groupby(key)[value].mean().reset_index()
         .rename(columns={key: "assay", value: out_name}))
    return g


def summary_per_assay(src: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per assay: the five headline metrics plus mean generalization κ,
    calibration ECE and prevalence CCC across that assay's behaviors."""
    pub = src.get("publication_metrics")
    if pub is None or pub.empty:
        return pd.DataFrame()
    base = pub.rename(columns={
        "project_id": "assay", "project_name": "assay_name",
        "cohen_kappa": "kappa"}).copy()
    keep = [c for c in ("assay", "assay_name", "f1", "mcc", "balanced_accuracy",
                        "roc_auc", "kappa") if c in base.columns]
    out = base[keep].copy()

    acc = src.get("accuracy_by_behavior")
    if acc is not None and "project_id" in acc.columns:
        nb = (acc.groupby("project_id").size().reset_index(name="n_behaviors")
              .rename(columns={"project_id": "assay"}))
        out = out.merge(nb, on="assay", how="left")

    for df, val, name in (
        (src.get("generalization"), "cohen_kappa", "mean_generalization_kappa"),
        (src.get("calibration"), "ece", "mean_ece"),
        (src.get("time_budget"), "prevalence_ccc", "mean_prevalence_ccc"),
    ):
        g = _group_mean(df, "project", val, name)
        if g is not None:
            out = out.merge(g, on="assay", how="left")

    # Assay-level totals of the tangible counts: "across this assay's behaviors,
    # the model recovered X of the Y windows the reviewer marked positive, with Z
    # false alarms." Summing across *behaviors* is the meaningful direction (each
    # behavior contributes its own positives); summing across seeds is not, and
    # confusion_by_behavior has already collapsed that axis by averaging.
    # TN is excluded on purpose — an assay-level accuracy off it would be ~0.99
    # by imbalance alone.
    conf = src.get("confusion_by_behavior")
    if conf is not None and {"project_id", "tp", "fn", "fp"} <= set(conf.columns):
        tot = (conf.groupby("project_id")[["tp", "fn", "fp"]].sum().reset_index()
               .rename(columns={"project_id": "assay", "tp": "tp_total",
                                "fn": "fn_total", "fp": "fp_total"}))
        tot["n_pos_val_total"] = tot["tp_total"] + tot["fn_total"]
        out = out.merge(tot, on="assay", how="left")

    order = ["assay", "assay_name", "n_behaviors", "f1", "mcc", "balanced_accuracy",
             "roc_auc", "kappa", "n_pos_val_total", "tp_total", "fn_total", "fp_total",
             "mean_generalization_kappa", "mean_ece", "mean_prevalence_ccc"]
    return out[[c for c in order if c in out.columns]]


# ── per-behavior (the master supplementary table) ───────────────────────────


def summary_per_behavior(src: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (assay, behavior): held-out F1 and the confusion counts behind
    it, generalization κ, ECE, prevalence CCC/bias, video ΔF1, all-enhancements
    ΔF1 and the AL positives-found ratio."""
    acc = src.get("accuracy_by_behavior")
    if acc is None or acc.empty:
        return pd.DataFrame()
    out = (acc.rename(columns={"project_id": "assay", "behavior_name": "behavior",
                               "f1_mean": "f1_holdout"})[["assay", "behavior",
                                                          "f1_holdout"]].copy())

    def _merge(df, cols_map, extra_key="project", extra_beh="behavior"):
        nonlocal out
        if df is None or df.empty:
            return
        need = [extra_key, extra_beh] + list(cols_map)
        if not all(c in df.columns for c in need):
            return
        part = df[need].rename(columns={extra_key: "assay", extra_beh: "behavior",
                                        **cols_map})
        out = out.merge(part, on=["assay", "behavior"], how="left")

    # Counts first, immediately beside F1: this is the master supplementary table,
    # and a rate is only checkable if the reader can see the n it came from.
    # Unit = one reviewer-scored held-out window (not a bout); counts are per fit,
    # averaged over seeds. See cross_project.confusion_by_behavior.
    _merge(src.get("confusion_by_behavior"),
           {"n_val": "n_val", "n_pos_val": "n_pos_val",
            "tp": "tp", "fn": "fn", "fp": "fp", "tn": "tn"},
           extra_key="project_id", extra_beh="behavior_name")
    _merge(src.get("generalization"), {"cohen_kappa": "generalization_kappa"})
    _merge(src.get("calibration"), {"ece": "ece"})
    _merge(src.get("time_budget"),
           {"prevalence_ccc": "prevalence_ccc", "prevalence_bias": "prevalence_bias"})
    _merge(src.get("video_value"), {"gain": "video_dF1", "significant": "video_sig"},
           extra_key="project_id", extra_beh="behavior_name")

    abl = src.get("ablation")
    if abl is not None and {"config", "clip_budget", "gain_over_baseline"} <= set(abl.columns):
        allenh = abl[(abl["config"] == _ALL_ENH_CONFIG) & (abl["clip_budget"] == "all")]
        if not allenh.empty:
            part = (allenh[["project", "behavior", "gain_over_baseline"]]
                    .rename(columns={"project": "assay",
                                     "gain_over_baseline": "all_enh_dF1"}))
            out = out.merge(part, on=["assay", "behavior"], how="left")

    al = src.get("al_summary")
    if al is not None and {"al_pos_discovered_end", "random_pos_discovered_end"} <= set(al.columns):
        r = al.copy()
        denom = r["random_pos_discovered_end"].replace(0, np.nan)
        r["al_pos_ratio"] = r["al_pos_discovered_end"] / denom
        part = r[["project", "behavior", "al_pos_ratio"]].rename(columns={"project": "assay"})
        out = out.merge(part, on=["assay", "behavior"], how="left")

    return out.sort_values(["assay", "behavior"], ignore_index=True)


# ── feature value (ablation, per enhancement × budget) ──────────────────────


def summary_feature_value(src: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """ΔF1 over the pose-only baseline for each enhancement, at each clip budget:
    mean, 95% CI across behaviors, and how many behaviors it helped significantly."""
    abl = src.get("ablation")
    if abl is None or abl.empty or "label" not in abl.columns:
        return pd.DataFrame()
    df = abl[abl["label"] != _BASELINE_LABEL].copy()
    if df.empty:
        return pd.DataFrame()
    # Preserve build order of enhancements and put n50 before all.
    label_order = list(dict.fromkeys(df["label"].astype(str)))
    budget_order = [b for b in ("n50", "all") if b in set(df["clip_budget"].astype(str))]
    rows = []
    for budget in budget_order:
        for label in label_order:
            grp = df[(df["clip_budget"].astype(str) == budget)
                     & (df["label"].astype(str) == label)]
            if grp.empty:
                continue
            gains = pd.to_numeric(grp["gain_over_baseline"], errors="coerce").dropna()
            # ``significant`` may round-trip as a real bool or the string "True";
            # count it without a fillna-downcast on the object column.
            sig = grp["significant"].tolist() if "significant" in grp.columns else []
            n_sig = sum(1 for v in sig if str(v).strip().lower() == "true")
            rows.append({
                "enhancement": label,
                "clip_budget": budget,
                "mean_dF1": float(gains.mean()) if len(gains) else float("nan"),
                "ci95": ci95(gains.tolist()) if len(gains) else 0.0,
                "n_significant": int(n_sig),
                "n_total": int(len(grp)),
            })
    return pd.DataFrame(rows)


# ── discrimination (error removed by feature family) ────────────────────────


def summary_discrimination(src: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Over confusable pairs the pose-only model can't already solve, the mean AUC
    a family reaches and the share of remaining error it removes."""
    cp = src.get("confusable_pairs")
    if cp is None or cp.empty or "pose_only_auc" not in cp.columns:
        return pd.DataFrame()
    head = cp[pd.to_numeric(cp["pose_only_auc"], errors="coerce") < _HEADROOM_AUC]
    if head.empty:
        return pd.DataFrame()
    pose = pd.to_numeric(head["pose_only_auc"], errors="coerce")
    fams = [
        ("+ Context / ROI", "pose_context_auc", "pose_context_error_reduction"),
        ("+ Video", "pose_video_auc", "pose_video_error_reduction"),
        ("All features", "all_features_auc", "all_features_error_reduction"),
    ]
    rows = []
    for name, auc_col, er_col in fams:
        if auc_col not in head.columns or er_col not in head.columns:
            continue
        rows.append({
            "feature_family": name,
            "n_pairs": int(len(head)),
            "mean_auc_pose": float(pose.mean()),
            "mean_auc_family": float(pd.to_numeric(head[auc_col], errors="coerce").mean()),
            "mean_error_reduction": float(pd.to_numeric(head[er_col], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


# ── active learning (pooled positives-found curve) ──────────────────────────


def summary_active_learning_curve(src: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Positives discovered vs clips reviewed, pooled across all behaviors, one row
    per (clips reviewed, strategy) — the meta AL-vs-random curve."""
    pts = src.get("al_points")
    if pts is None or pts.empty:
        return pd.DataFrame()
    need = {"n_clips_reviewed", "strategy", "pos_discovered_mean"}
    if not need <= set(pts.columns):
        return pd.DataFrame()
    rows = []
    for (n, strat), grp in pts.groupby(["n_clips_reviewed", "strategy"], sort=True):
        pos = pd.to_numeric(grp["pos_discovered_mean"], errors="coerce").dropna()
        row = {
            "n_clips_reviewed": int(n),
            "strategy": str(strat),
            "mean_pos_discovered": float(pos.mean()) if len(pos) else float("nan"),
            "ci95": ci95(pos.tolist()) if len(pos) else 0.0,
            "n_behaviors": int(len(grp)),
        }
        if "f1_mean" in grp.columns:
            f1 = pd.to_numeric(grp["f1_mean"], errors="coerce").dropna()
            row["mean_f1"] = float(f1.mean()) if len(f1) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ── orchestration ───────────────────────────────────────────────────────────

_BUILDERS = {
    "summary_per_assay.csv": summary_per_assay,
    "summary_per_behavior.csv": summary_per_behavior,
    "summary_feature_value.csv": summary_feature_value,
    "summary_discrimination.csv": summary_discrimination,
    "summary_active_learning_curve.csv": summary_active_learning_curve,
}

_README_HEAD = """\
Meta summary tables
===================
Distilled, assay-scoped summaries that back the one-figure manuscript story. Each is
small enough to plot directly; the exhaustive per-behavior CSVs remain next door for
the record.

These are READING tables: several carry two leading key columns and mix units in one
table, so they are not direct Prism pastes. The one-file-per-figure Prism tables live
in ../prism/ -- use those to plot, these to read.

Files in THIS directory (only the summaries this run could build are listed):

"""

# One blurb per builder, emitted only for the files actually written. The previous
# static string always advertised all five, so a run that could build one of them
# (the usual case -- each needs its own upstream analysis to have run) shipped a
# README describing four files that were not there.
_README_SECTIONS = {
    "summary_per_assay.csv":
        "summary_per_assay.csv            One row per assay: F1/MCC/balanced-acc/\n"
        "                                 ROC-AUC/kappa, plus mean generalization\n"
        "                                 kappa, ECE and prevalence CCC.\n",
    "summary_per_behavior.csv":
        "summary_per_behavior.csv         One row per (assay, behavior): the master\n"
        "                                 table joining held-out F1, generalization\n"
        "                                 kappa, ECE, prevalence CCC/bias, video dF1,\n"
        "                                 all-enhancements dF1, AL pos ratio.\n",
    "summary_feature_value.csv":
        "summary_feature_value.csv        dF1 over pose-only per enhancement x clip\n"
        "                                 budget: mean, 95% CI, and how many behaviors\n"
        "                                 it helped (sig).\n",
    "summary_discrimination.csv":
        "summary_discrimination.csv       Over confusable pairs with headroom, the AUC\n"
        "                                 each feature family reaches and the % of\n"
        "                                 remaining error removed.\n",
    "summary_active_learning_curve.csv":
        "summary_active_learning_curve.csv  Positives found vs clips reviewed, pooled\n"
        "                                 across behaviors, AL vs random.\n",
}

_README_TAIL = """
Behaviors are assay-scoped throughout: an assay's Rear and another assay's Rear are
different models and stay different rows.
"""


def build_summaries(src: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return ``{filename: DataFrame}`` for every summary buildable from ``src``."""
    out: dict[str, pd.DataFrame] = {}
    for fname, fn in _BUILDERS.items():
        try:
            df = fn(src)
        except Exception:  # noqa: BLE001 — one bad summary must not sink the rest
            df = pd.DataFrame()
        if df is not None and not df.empty:
            out[fname] = df
    return out


def write_all(out_dir: str | Path, src: dict[str, pd.DataFrame]) -> list[Path]:
    """Write every buildable summary into ``out_dir/summary`` + a README."""
    out_dir = Path(out_dir) / "summary"
    tables = build_summaries(src)
    if not tables:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    from abel.validation import prism

    for fname, df in tables.items():
        path = out_dir / fname
        # ASCII + BOM, same as the Prism bundle: these open in Excel on Windows.
        # Not prism._write -- that prunes all-NaN columns, which is right for a
        # plot-ready table and wrong here: "this metric was not computed" is
        # information a reading table should keep showing.
        prism._asciify(df.round(4)).to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    body = "".join(_README_SECTIONS.get(f, "") for f in tables)
    written.append(prism.write_text(out_dir / "README_SUMMARY.txt",
                                    _README_HEAD + body + _README_TAIL))
    return written
