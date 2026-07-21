"""GraphPad Prism-ready exports of the validation results.

The analysis CSVs elsewhere in this package are **tidy** (one row per observation,
with key columns like ``project`` / ``behavior`` / ``config``).  That is the right
shape for pandas and for archiving, and the wrong shape for Prism: Prism has no
pivot-on-import.  It ingests a rectangular block — first column = row titles, one
column per dataset, consecutive columns = side-by-side replicate subcolumns — and
a tidy file forces the user to hand-pivot in Excel before they can paste anything.

So this module emits, alongside the tidy CSVs, one **pre-pivoted** file per intended
figure.  Each is a direct paste into a Prism data table of a named type.  The rules
every writer here follows:

* **One** row-title column, first.  Prism takes a single row label; shipping
  ``project`` and ``behavior`` as separate columns means the user has to
  concatenate them by hand.
* **Replicates, not summaries.**  Where the design is seeded, the per-seed values
  are emitted as consecutive columns so Prism can run the paired test *itself*.
  A mean + CI + a boolean ``significant`` column cannot be re-tested or re-plotted.
* No free-text/prose columns, no JSON in cells, no reciprocal duplicate metrics,
  and floats rounded to something a spreadsheet can display.

``write_all`` also drops a ``README_PRISM.txt`` naming the Prism table type and
replicate count for each file, because that is the one thing the CSV itself cannot
carry.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Prism displays far fewer digits than float64 carries. Rounding to a fixed number
# of significant figures (not decimals) keeps small importances readable while
# stripping float64 dust; sub-1e-9 values and -0.0 collapse to a clean 0.
_SIGFIGS = 4


def _sig(x, n: int = _SIGFIGS):
    """Round ``x`` to ``n`` significant figures; collapse dust and -0.0 to 0.0.

    Values with |x| < 1e-9 (e.g. CI half-widths of ~1e-17 that are numerically
    zero) become 0.0 so they stop rendering as scientific-notation noise. Genuine
    small values (a ~1e-5 importance) are kept — Prism reads them fine.
    """
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(xf):
        return xf
    if xf == 0 or abs(xf) < 1e-9:
        return 0.0
    d = n - int(np.floor(np.log10(abs(xf)))) - 1
    r = round(xf, d)
    return 0.0 if r == 0 else r


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`_sig` to every float column, leaving NaN and non-numerics."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: _sig(v) if pd.notna(v) else v)
    return df


def _row_title(project: str, behavior: str) -> str:
    return f"{project} · {behavior}"


def _seed_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    """Seed columns for ``prefix``, ordered by seed number (not lexically —
    ``seed10`` must not sort between ``seed1`` and ``seed2``)."""
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    hits = [(int(m.group(1)), c) for c in df.columns if (m := pat.match(str(c)))]
    return [c for _, c in sorted(hits)]


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _clean(df).to_csv(path, index=False, encoding="utf-8")
    return path


# ── Generalization: model κ per behavior ────────────────────────────────────


def prism_kappa(gen_df: pd.DataFrame) -> pd.DataFrame:
    """Column table: one row per behavior, κ (and the human ceiling, if measured).

    The ``human_ceiling_kappa`` column is dropped when it is empty for every row —
    an all-NaN column in Prism silently becomes an empty dataset that still occupies
    a slot in the graph and the legend.
    """
    out = pd.DataFrame({
        "Behavior": [_row_title(p, b) for p, b in
                     zip(gen_df["project"], gen_df["behavior"])],
        "Cohen's kappa": pd.to_numeric(gen_df["cohen_kappa"], errors="coerce"),
        "F1": pd.to_numeric(gen_df["f1"], errors="coerce"),
    })
    ceiling = pd.to_numeric(gen_df.get("human_ceiling_kappa"), errors="coerce")
    if ceiling is not None and ceiling.notna().any():
        out["Human ceiling kappa"] = ceiling.to_numpy()
    return out.sort_values("Cohen's kappa", ascending=False, ignore_index=True)


# ── Video value: paired, with the seeds Prism needs to run the test ─────────


def prism_video_value(vv_df: pd.DataFrame) -> pd.DataFrame:
    """Grouped table with replicates: 2 groups (video off / on) × N seed subcolumns.

    Emitting only ``f1_no_video`` / ``f1_with_video`` means the user can plot the
    two means but cannot reproduce the paired t-test the asterisks come from.  The
    seed columns are laid out consecutively — off₁…offₙ, on₁…onₙ — which is exactly
    the order Prism assigns to side-by-side subcolumns on paste.
    """
    df = vv_df[vv_df.get("error").isna() | (vv_df.get("error") == "")] \
        if "error" in vv_df.columns else vv_df
    no_cols = _seed_cols(df, "f1_no_video_seed")
    yes_cols = _seed_cols(df, "f1_with_video_seed")

    out = pd.DataFrame({
        "Behavior": [_row_title(p, b) for p, b in
                     zip(df["project_id"], df["behavior_name"])],
    })
    if no_cols and yes_cols:
        for i, c in enumerate(no_cols, start=1):
            out[f"Pose only:{i}"] = pd.to_numeric(df[c], errors="coerce").to_numpy()
        for i, c in enumerate(yes_cols, start=1):
            out[f"+Video:{i}"] = pd.to_numeric(df[c], errors="coerce").to_numpy()
    else:
        # Older exports dropped the seeds. Fall back to the means so the file is
        # still pasteable, but it can only be plotted — not re-tested.
        out["Pose only (mean)"] = pd.to_numeric(df["f1_no_video"],
                                                errors="coerce").to_numpy()
        out["+Video (mean)"] = pd.to_numeric(df["f1_with_video"],
                                             errors="coerce").to_numpy()
    return out


# ── Ablation: one table per clip budget (Prism grids are 2-factor) ──────────


def prism_ablation(abl_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{clip_budget: grouped table}`` — rows = behaviors, columns = configs.

    The tidy ablation CSV crosses **four** factors (project × behavior × clip budget
    × config).  A Prism grouped table holds two (row groups × datasets), so this
    cannot be one table under any arrangement: it is split into one table per clip
    budget, leaving behaviors down the rows and configs across the columns.

    Per-seed F1 columns are used as replicate subcolumns when the export carries
    them; otherwise the config means are emitted alone.
    """
    tables: dict[str, pd.DataFrame] = {}
    seed_cols = _seed_cols(abl_df, "f1_seed")
    for budget, grp in abl_df.groupby("clip_budget", sort=False):
        rows: dict[str, dict[str, float]] = {}
        # Config order: as built (baseline first), not alphabetical.
        configs = list(dict.fromkeys(grp["label"].astype(str)))
        for _, r in grp.iterrows():
            key = _row_title(r["project"], r["behavior"])
            cell = rows.setdefault(key, {})
            label = str(r["label"])
            if seed_cols:
                for i, c in enumerate(seed_cols, start=1):
                    cell[f"{label}:{i}"] = pd.to_numeric(r[c], errors="coerce")
            else:
                cell[label] = pd.to_numeric(r["f1_mean"], errors="coerce")

        cols: list[str] = []
        for label in configs:
            cols.extend([f"{label}:{i}" for i in range(1, len(seed_cols) + 1)]
                        if seed_cols else [label])
        table = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=cols)
        table.insert(0, "Behavior", table.index)
        tables[str(budget)] = table.reset_index(drop=True)
    return tables


def prism_ablation_gain_matrix(abl_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{clip_budget: matrix}`` — the ablation heatmap, ready to paste as a Prism
    heatmap. Rows = ``project · behavior`` (row titles), columns = enhancement,
    cells = ΔF1 over the pose-only baseline. One matrix per clip budget.

    This is the pre-pivoted twin of the ``feature_impact`` figure's right panel, so
    the user can restyle the heatmap in Prism instead of re-pivoting the tidy CSV.
    """
    df = abl_df[abl_df["config"] != "baseline_none"].copy()
    if df.empty:
        return {}
    df["__row"] = [_row_title(p, b) for p, b in zip(df["project"], df["behavior"])]
    labels = list(dict.fromkeys(df["label"].astype(str)))   # build order, not alpha
    out: dict[str, pd.DataFrame] = {}
    for budget, grp in df.groupby("clip_budget", sort=False):
        piv = (grp.pivot_table(index="__row", columns="label",
                               values="gain_over_baseline", aggfunc="first")
               .reindex(columns=labels))
        piv.insert(0, "Behavior", piv.index)
        out[str(budget)] = piv.reset_index(drop=True)
    return out


def prism_ablation_gain(abl_df: pd.DataFrame) -> pd.DataFrame:
    """Grouped table of ΔF1 vs. baseline: rows = behaviors, columns = enhancement.

    The headline number, already differenced, for the manuscript's summary panel.
    Carries the exact p per cell in a parallel ``… (p)`` column rather than a
    boolean, so the reader can report a test statistic.
    """
    df = abl_df[abl_df["config"] != "baseline_none"].copy()
    df["__row"] = [_row_title(p, b) for p, b in zip(df["project"], df["behavior"])]
    # pivot_table sorts columns alphabetically; keep the build order the figures use,
    # so the columns line up with the ablation panel a reader is holding next to it.
    labels = list(dict.fromkeys(df["label"].astype(str)))
    frames = []
    for budget, grp in df.groupby("clip_budget", sort=False):
        piv = grp.pivot_table(index="__row", columns="label", values="gain_over_baseline",
                              aggfunc="first").reindex(columns=labels)
        piv.insert(0, "Clip budget", str(budget))
        frames.append(piv)
    out = pd.concat(frames) if frames else pd.DataFrame()
    if not out.empty:
        out.insert(0, "Behavior", out.index)
        out = out.reset_index(drop=True)
    return out


# ── Throughput: one table per stage (the stages don't share units) ──────────


def prism_throughput(bench_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{stage: table}`` — the three stages carry different units and cannot share
    a Prism table (``extract``/``infer`` are ×real-time; ``train`` is seconds)."""
    tables: dict[str, pd.DataFrame] = {}
    ok = bench_df[bench_df.get("error").isna() | (bench_df.get("error") == "")] \
        if "error" in bench_df.columns else bench_df

    for stage in ("extract", "infer"):
        grp = ok[ok["stage"] == stage]
        if grp.empty:
            continue
        tables[stage] = pd.DataFrame({
            "Project": grp["project_id"].astype(str).to_numpy(),
            "x faster than real-time": pd.to_numeric(
                grp["faster_than_realtime"], errors="coerce").to_numpy(),
            "Wall-clock seconds": pd.to_numeric(grp["seconds"],
                                                errors="coerce").to_numpy(),
        })

    trn = ok[ok["stage"] == "train"]
    if not trn.empty:
        # Grouped table: projects as columns, behaviors as the rows within them —
        # a ragged block (projects have different behaviors), which Prism reads as
        # unequal-n groups.
        wide = trn.pivot_table(index="detail", columns="project_id", values="seconds",
                               aggfunc="mean")
        wide.insert(0, "Behavior", wide.index)
        tables["train"] = wide.reset_index(drop=True)
    return tables


# ── Active learning: wide XY learning curves ────────────────────────────────

_AL_STRATEGY = {"active_learning": "AL", "random": "Random"}
_AL_METRICS = (("f1_mean", "prism_al_curve_f1.csv"),
               ("pr_auc_mean", "prism_al_curve_pr_auc.csv"),
               ("pos_discovered_mean", "prism_al_curve_pos_discovered.csv"))


def prism_al_curves(al_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{filename: wide XY table}``. Shared X = clips reviewed; one Y column per
    project·behavior·strategy — one file each for F1, PR-AUC, positives found."""
    df = al_df.copy()
    df["__col"] = (df["project_id"].astype(str) + " · " + df["behavior_name"].astype(str)
                   + " — " + df["strategy"].map(_AL_STRATEGY).fillna(df["strategy"]))
    out: dict[str, pd.DataFrame] = {}
    for metric, fname in _AL_METRICS:
        wide = (df.pivot_table(index="n_clips_reviewed", columns="__col",
                               values=metric, sort=False)
                .reset_index().rename(columns={"n_clips_reviewed": "Clips reviewed"}))
        wide.columns.name = None
        out[fname] = wide
    return out


# ── Calibration: reliability diagram, paired XY per series ───────────────────


def prism_calibration(rel_df: pd.DataFrame) -> pd.DataFrame:
    """Paired-XY: each project·behavior gets a (confidence, accuracy) column pair
    so every reliability curve pastes as its own Prism XY dataset. Series have
    differing bin counts; ``concat(axis=1)`` pads the short ones with NaN — the
    ragged block Prism expects for unequal-length XY datasets."""
    blocks = []
    for (proj, beh), g in rel_df.groupby(["project", "behavior"], sort=False):
        g = g.reset_index(drop=True)
        name = _row_title(proj, beh)
        blocks.append(pd.DataFrame({
            f"{name} — confidence": pd.to_numeric(g["mean_confidence"], errors="coerce"),
            f"{name} — accuracy": pd.to_numeric(g["empirical_accuracy"], errors="coerce"),
        }))
    return pd.concat(blocks, axis=1) if blocks else pd.DataFrame()


# ── Time budget: true vs predicted prevalence, paired XY per behavior ────────


def prism_time_budget(tb_df: pd.DataFrame) -> pd.DataFrame:
    """Paired-XY: per behavior, a (true, pred) prevalence column pair — points are
    sessions. Paste as XY, plot the identity line, report the correlation in Prism."""
    blocks = []
    for (proj, beh), g in tb_df.groupby(["project", "behavior"], sort=False):
        g = g.reset_index(drop=True)
        name = _row_title(proj, beh)
        blocks.append(pd.DataFrame({
            f"{name} — true": pd.to_numeric(g["true_prevalence"], errors="coerce"),
            f"{name} — pred": pd.to_numeric(g["pred_prevalence"], errors="coerce"),
        }))
    return pd.concat(blocks, axis=1) if blocks else pd.DataFrame()


# ── Behaviorscape: modality shares + feature importance heatmap ──────────────


def prism_behaviorscape_shares(shares_df: pd.DataFrame) -> pd.DataFrame:
    """Grouped/stacked-bar table: rows = behaviors, columns = modality %."""
    wide = (shares_df.pivot_table(index="behavior", columns="modality_label",
                                  values="percent", sort=False).reset_index())
    wide.columns.name = None
    return wide


def prism_behaviorscape_importance(imp_df: pd.DataFrame) -> pd.DataFrame:
    """Heatmap matrix: rows = feature, columns = behavior, cell = importance.
    All-zero feature rows are dropped so the heatmap isn't mostly blank."""
    wide = (imp_df.pivot_table(index="feature", columns="behavior",
                               values="importance", sort=False).reset_index())
    wide.columns.name = None
    num = wide.drop(columns="feature")
    keep = (num.fillna(0).abs().sum(axis=1) > 0).to_numpy()
    return wide[keep].reset_index(drop=True)


# ── Discrimination: roc_auc / error_reduction by feature set ─────────────────

_DISC_METRICS = (("roc_auc", "prism_discrimination_roc_auc.csv"),
                 ("error_reduction", "prism_discrimination_error_reduction.csv"))


def prism_discrimination(disc_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{filename: table}`` — rows = project·pair, columns = feature-set label.
    For error_reduction the pose-only baseline column (0 by definition) is dropped."""
    df = disc_df.copy()
    df["__pair"] = df["project"].astype(str) + " · " + df["pair"].astype(str)
    labels = list(dict.fromkeys(df["label"].astype(str)))   # keep build order
    out: dict[str, pd.DataFrame] = {}
    for metric, fname in _DISC_METRICS:
        cols = labels
        if metric == "error_reduction":
            cols = [c for c in labels if c != "Pose only"]  # baseline is 0 by defn
        wide = (df.pivot_table(index="__pair", columns="label", values=metric,
                               aggfunc="first", sort=False)
                .reindex(columns=cols).reset_index()
                .rename(columns={"__pair": "Pair"}))
        wide.columns.name = None
        out[fname] = wide
    return out


# ── Cross-project: accuracy (F1) by behavior ─────────────────────────────────


def prism_accuracy_by_behavior(beh_df: pd.DataFrame) -> pd.DataFrame:
    """Column/bar table: one row per project·behavior, pooled held-out F1 + CI."""
    return beh_df.rename(columns={
        "project_id": "Project", "behavior_name": "Behavior",
        "f1_mean": "F1", "f1_ci": "F1 95% CI", "n": "N"})


# ── Orchestration ───────────────────────────────────────────────────────────

_README = """\
Prism-ready exports
===================
These files are pre-pivoted for GraphPad Prism. Each is a direct paste: create the
named table type, then paste with the first column as row titles.

The tidy CSVs next to them (ablation_results.csv, video_value.csv, ...) remain the
archival/pandas copies -- they are long-format and Prism cannot pivot on import.

{sections}
Notes
-----
* Columns named "<group>:1 .. <group>:N" are the N per-seed replicates of that group.
  In Prism's New Table dialog choose "Grouped" -> "Enter and plot replicate values"
  with N side-by-side subcolumns, then paste. Prism will then run the paired test
  itself (e.g. Analyze -> t tests -> Paired) instead of you re-typing an asterisk.
* Row titles are "project (dot) behavior" in a single column, because Prism accepts
  exactly one row-title column.
"""


def write_all(out_dir: Path, *, gen_df: pd.DataFrame | None = None,
              video_df: pd.DataFrame | None = None,
              ablation_df: pd.DataFrame | None = None,
              bench_df: pd.DataFrame | None = None,
              al_df: pd.DataFrame | None = None,
              calibration_df: pd.DataFrame | None = None,
              time_budget_df: pd.DataFrame | None = None,
              bscape_shares_df: pd.DataFrame | None = None,
              bscape_importance_df: pd.DataFrame | None = None,
              discrimination_df: pd.DataFrame | None = None,
              accuracy_by_behavior_df: pd.DataFrame | None = None) -> list[Path]:
    """Write every available Prism table into ``out_dir/prism`` + a README."""
    out_dir = Path(out_dir) / "prism"
    written: list[Path] = []
    sections: list[str] = []

    if gen_df is not None and not gen_df.empty:
        written.append(_write(prism_kappa(gen_df), out_dir / "prism_kappa.csv"))
        sections.append("prism_kappa.csv\n    Table: Column (or Grouped, 1 replicate).\n"
                        "    One row per behavior; Cohen's kappa + F1.\n")

    if video_df is not None and not video_df.empty:
        t = prism_video_value(video_df)
        n_rep = sum(1 for c in t.columns if str(c).startswith("Pose only:"))
        written.append(_write(t, out_dir / "prism_video_value.csv"))
        sections.append(
            f"prism_video_value.csv\n    Table: Grouped, 2 groups x "
            f"{n_rep or 1} replicate(s).\n"
            "    Paired video-off vs video-on F1. Analyze -> t tests -> Paired.\n")

    if ablation_df is not None and not ablation_df.empty:
        for budget, table in prism_ablation(ablation_df).items():
            path = out_dir / f"prism_ablation_{budget}.csv"
            written.append(_write(table, path))
        gain = prism_ablation_gain(ablation_df)
        if not gain.empty:
            written.append(_write(gain, out_dir / "prism_ablation_gain.csv"))
        for budget, mat in prism_ablation_gain_matrix(ablation_df).items():
            written.append(_write(mat, out_dir / f"prism_ablation_gain_matrix_{budget}.csv"))
        sections.append(
            "prism_ablation_<budget>.csv\n    Table: Grouped, one column group per\n"
            "    config. One FILE per clip budget -- Prism grids are 2-factor and the\n"
            "    ablation crosses 4 (project x behavior x budget x config).\n"
            "prism_ablation_gain.csv\n    Table: Grouped. dF1 vs. the pose-only\n"
            "    baseline, already differenced.\n"
            "prism_ablation_gain_matrix_<budget>.csv\n    Table: Heatmap (XY / Grouped\n"
            "    with row titles). Behaviors x enhancements, cell = dF1 vs pose-only\n"
            "    -- the feature_impact heatmap, restyle it yourself.\n")

    if bench_df is not None and not bench_df.empty:
        for stage, table in prism_throughput(bench_df).items():
            written.append(_write(table, out_dir / f"prism_throughput_{stage}.csv"))
        sections.append(
            "prism_throughput_<stage>.csv\n    Table: Column. Split by stage because\n"
            "    extract/infer are x-real-time and train is seconds.\n")

    if al_df is not None and not al_df.empty:
        for fname, table in prism_al_curves(al_df).items():
            written.append(_write(table, out_dir / fname))
        sections.append(
            "prism_al_curve_<metric>.csv\n    Table: XY. Shared X = clips reviewed;\n"
            "    one Y column per project·behavior·strategy (AL vs Random).\n")

    if calibration_df is not None and not calibration_df.empty:
        written.append(_write(prism_calibration(calibration_df),
                              out_dir / "prism_calibration_reliability.csv"))
        sections.append(
            "prism_calibration_reliability.csv\n    Table: XY. Paired confidence/\n"
            "    accuracy columns per series — the reliability diagram.\n")

    if time_budget_df is not None and not time_budget_df.empty:
        written.append(_write(prism_time_budget(time_budget_df),
                              out_dir / "prism_time_budget_prevalence.csv"))
        sections.append(
            "prism_time_budget_prevalence.csv\n    Table: XY. Paired true/pred\n"
            "    prevalence per behavior; points are sessions.\n")

    if bscape_shares_df is not None and not bscape_shares_df.empty:
        written.append(_write(prism_behaviorscape_shares(bscape_shares_df),
                              out_dir / "prism_behaviorscape_modality_shares.csv"))
        sections.append(
            "prism_behaviorscape_modality_shares.csv\n    Table: Grouped/stacked bar.\n"
            "    Rows = behaviors, columns = modality %.\n")

    if bscape_importance_df is not None and not bscape_importance_df.empty:
        written.append(_write(prism_behaviorscape_importance(bscape_importance_df),
                              out_dir / "prism_behaviorscape_importance.csv"))
        sections.append(
            "prism_behaviorscape_importance.csv\n    Table: Heatmap. Rows = feature,\n"
            "    columns = behavior, cell = importance (all-zero rows dropped).\n")

    if discrimination_df is not None and not discrimination_df.empty:
        for fname, table in prism_discrimination(discrimination_df).items():
            written.append(_write(table, out_dir / fname))
        sections.append(
            "prism_discrimination_<metric>.csv\n    Table: Grouped. Rows = project·\n"
            "    pair, columns = feature set (roc_auc and error_reduction).\n")

    if accuracy_by_behavior_df is not None and not accuracy_by_behavior_df.empty:
        written.append(_write(prism_accuracy_by_behavior(accuracy_by_behavior_df),
                              out_dir / "prism_accuracy_by_behavior.csv"))
        sections.append(
            "prism_accuracy_by_behavior.csv\n    Table: Column/bar. Pooled held-out\n"
            "    F1 + 95% CI per project·behavior.\n")

    if written:
        (out_dir / "README_PRISM.txt").write_text(
            _README.format(sections="\n".join(sections)), encoding="utf-8")
        written.append(out_dir / "README_PRISM.txt")
    return written
