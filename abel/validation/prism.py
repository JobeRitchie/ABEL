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


# Columns where a tiny magnitude is the RESULT, not float dust. _sig collapses
# anything under 1e-9 to a clean zero, which is right for a CI half-width of 1e-17
# and wrong for a p-value of 9e-10 -- that exports as "0", a number a p-value can
# never take, and a reader cannot tell a real 9e-10 from a numerical artefact.
_NO_DUST_COLLAPSE = frozenset({"PValue"})


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`_sig` to every float column, leaving NaN and non-numerics."""
    df = df.copy()
    for c in df.columns:
        if not pd.api.types.is_float_dtype(df[c]):
            continue
        keep_small = str(c) in _NO_DUST_COLLAPSE
        df[c] = df[c].map(
            lambda v, keep=keep_small: (
                v if pd.isna(v) else (_sig_keep_small(v) if keep else _sig(v))))
    return df


def _sig_keep_small(x, n: int = _SIGFIGS):
    """:func:`_sig` without the dust collapse — for columns where small is real."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(xf) or xf == 0:
        return 0.0 if xf == 0 else xf
    return round(xf, n - int(np.floor(np.log10(abs(xf)))) - 1)


# ── ASCII-only export text ──────────────────────────────────────────────────
# Prism and Excel on Windows import CSV using the ANSI code page (cp1252), not
# UTF-8, unless the file opens with a BOM.  A UTF-8 "≥" then arrives as "â‰¥" —
# so a header like "F1≥0.70:1" lands in the user's data table as mojibake.
#
# Two independent defences, because either alone leaves a hole:
#   1. Transliterate the symbols *we* choose into ASCII (below).  Nothing here
#      carries meaning its ASCII spelling doesn't.
#   2. Write with a BOM (``utf-8-sig``).  Covers the text we do NOT control —
#      a project or behavior name the user typed with an accent or a µ.
#
# This is deliberately scoped to the CSV/TXT export boundary.  Plot labels keep
# their Unicode (Δ, κ, ×): a PNG renders them correctly and they read better.
# Each replacement is the same width in "spaces already around it" terms as the
# glyph it replaces, so substitution never disturbs the surrounding layout — the
# README strings are indented, multi-line, and must survive this untouched.
_ASCII_MAP = {
    "≥": ">=",   # effort-to-quality target labels -> "F1 >= 0.70"
    "≤": "<=",
    "·": "-",    # project·behavior row titles -> "OF - Freeze"
    "—": "-",    # em dash
    "–": "-",    # en dash
    "−": "-",    # true minus
    "±": "+/-",
    "×": "x",
    "÷": "/",
    "Δ": "d",    # dF1 -- "delta" would collide with column-name width elsewhere
    "κ": "kappa",
    "φ": "phi",
    "→": "->",
    "←": "<-",
    "≈": "~",
    "…": "...",
    "•": "*",
    "²": "^2",
    "√": "sqrt",
    "₁": "1", "ₙ": "n",
}


def _ascii(s):
    """Transliterate export text to ASCII; non-strings pass through unchanged.

    Whitespace is left exactly as found: this also runs over multi-line README
    text, where collapsing runs of spaces would destroy the indentation.

    Anything still outside ASCII after the mapping — an accent in a behavior name
    the *user* typed — is deliberately kept.  The BOM written by :func:`_write`
    carries it correctly; this map only spells out the symbols this package
    itself introduces.
    """
    if not isinstance(s, str):
        return s
    for bad, good in _ASCII_MAP.items():
        if bad in s:
            s = s.replace(bad, good)
    return s


def _asciify(df: pd.DataFrame) -> pd.DataFrame:
    """ASCII-transliterate column headers and every string cell of ``df``."""
    df = df.copy()
    df.columns = [_ascii(c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(_ascii)
    return df


def _row_title(project: str, behavior: str) -> str:
    return f"{project} - {behavior}"


# ── error bars ──────────────────────────────────────────────────────────────
# Prism's "Enter and plot error values" accepts Mean+SD+N or Mean+SEM+N. It has
# no input format for a confidence-interval half-width, so the ci95 columns the
# analyses produce cannot be pasted as error bars at all -- pasting one into an
# SD subcolumn silently inflates every bar by t(n)/sqrt(n).
#
# Prefer replicate subcolumns wherever the per-seed values survived (Prism then
# computes the error bars *and* can run the test). Use this only where they did
# not, inverting ci95 = t_crit_95(n) * sd / sqrt(n) exactly.


def sd_from_ci95(ci, n):
    """Recover the SD that produced a t-based 95% CI half-width over ``n`` seeds.

    Returns NaN when n < 2 (no spread is defined from a single seed) or when the
    inputs are missing, so the column stays blank in Prism rather than reading as
    a real zero-variance measurement.
    """
    from abel.validation.metrics import t_critical_95

    ci_a = pd.to_numeric(pd.Series(ci), errors="coerce").to_numpy(dtype=float)
    n_a = pd.to_numeric(pd.Series(n), errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.array([t_critical_95(int(v)) if np.isfinite(v) and v >= 2 else np.nan
                      for v in n_a], dtype=float)
        sd = ci_a / t * np.sqrt(n_a)
    return np.where(np.isfinite(sd), sd, np.nan)


def _mean_sd_n(out: pd.DataFrame, name: str, mean, ci, n) -> None:
    """Append a ``<name>:Mean``/``:SD``/``:N`` triple -- Prism's Mean, SD, N layout."""
    out[f"{name}:Mean"] = pd.to_numeric(pd.Series(mean), errors="coerce").to_numpy()
    out[f"{name}:SD"] = sd_from_ci95(ci, n)
    out[f"{name}:N"] = pd.to_numeric(pd.Series(n), errors="coerce").to_numpy()


def _replicate_block(out: pd.DataFrame, name: str, seeds_by_x: dict, xs: list,
                     n_rep: int) -> None:
    """Append ``<name>:1 .. <name>:n_rep`` replicate subcolumns.

    Every dataset in a file is padded to the same ``n_rep`` so all datasets have
    identical subcolumn counts -- Prism assigns subcolumns positionally on paste,
    so a short block would shift every dataset after it one column to the left.
    """
    for i in range(n_rep):
        out[f"{name}:{i + 1}"] = [
            (seeds_by_x.get(x)[i]
             if seeds_by_x.get(x) is not None and i < len(seeds_by_x[x])
             else np.nan)
            for x in xs
        ]


def _seed_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    """Seed columns for ``prefix``, ordered by seed number (not lexically —
    ``seed10`` must not sort between ``seed1`` and ``seed2``)."""
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    hits = [(int(m.group(1)), c) for c in df.columns if (m := pat.match(str(c)))]
    return [c for _, c in sorted(hits)]


def _drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all-NaN data columns, always keeping column 0 (the row titles).

    An all-NaN column pastes into Prism as a phantom dataset: it occupies a slot,
    claims a colour, and shows up in the legend with nothing plotted.  These arise
    wherever a writer reindexes onto a globally-collected label list (a config one
    project never built, a metric that was not computed).
    """
    if df.shape[1] < 2:
        return df
    keep = [df.columns[0]] + [c for c in df.columns[1:] if not df[c].isna().all()]
    return df[keep]


def _write(df: pd.DataFrame, path: Path) -> Path:
    """Write one Prism table: ASCII text, no phantom columns, BOM for Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _asciify(_drop_empty(_clean(df))).to_csv(
        path, index=False, encoding="utf-8-sig")
    return path


def write_text(path: Path, text: str) -> Path:
    """Write a README/sidecar as ASCII + BOM, matching the CSVs beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_ascii(text), encoding="utf-8-sig")
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


def prism_ablation_gain(abl_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{clip_budget: grouped table}`` of ΔF1 vs. baseline — rows = behaviors,
    columns = enhancement, each followed by its exact p.

    One file per clip budget.  The previous single-frame version carried both
    ``Behavior`` *and* ``Clip budget`` as leading columns; Prism accepts exactly
    one row-title column, so on paste the budget became a text "dataset" and every
    numeric column shifted one to the right.
    """
    df = abl_df[abl_df["config"] != "baseline_none"].copy()
    if df.empty:
        return {}
    df["__row"] = [_row_title(p, b) for p, b in zip(df["project"], df["behavior"])]
    # pivot_table sorts columns alphabetically; keep the build order the figures use,
    # so the columns line up with the ablation panel a reader is holding next to it.
    labels = list(dict.fromkeys(df["label"].astype(str)))
    has_p = "gain_p_value" in df.columns
    out: dict[str, pd.DataFrame] = {}
    for budget, grp in df.groupby("clip_budget", sort=False):
        piv = grp.pivot_table(index="__row", columns="label",
                              values="gain_over_baseline",
                              aggfunc="first").reindex(columns=labels)
        pv = (grp.pivot_table(index="__row", columns="label", values="gain_p_value",
                              aggfunc="first").reindex(columns=labels)
              if has_p else None)
        table = pd.DataFrame({"Behavior": piv.index.to_numpy()})
        for lab in labels:
            table[lab] = piv[lab].to_numpy()
            if pv is not None:
                # The exact p beside its gain, not a boolean `significant`: a
                # reader can report the former and cannot re-derive it from the
                # latter.
                table[f"{lab} (p)"] = pv[lab].to_numpy()
        out[str(budget)] = table.reset_index(drop=True)
    return out


def prism_ablation_gain_seeds(abl_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{clip_budget: grouped table}`` of **paired per-seed** ΔF1 vs. baseline.

    ``f1_seed{i}(config) - f1_seed{i}(baseline)`` within each project·behavior·
    budget — the same seed, so the same subsample and the same split.  This is the
    one ablation table that lets Prism run the test itself (Analyze -> t tests ->
    One sample t test vs 0) instead of reading a pre-computed p.
    """
    seed_cols = _seed_cols(abl_df, "f1_seed")
    if not seed_cols or abl_df.empty:
        return {}
    labels = [lab for lab, cfg in
              dict(zip(abl_df["label"].astype(str), abl_df["config"])).items()
              if cfg != "baseline_none"]
    out: dict[str, pd.DataFrame] = {}
    for budget, grp in abl_df.groupby("clip_budget", sort=False):
        rows: dict[str, dict[str, float]] = {}
        for (proj, beh), sub in grp.groupby(["project", "behavior"], sort=False):
            base = sub[sub["config"] == "baseline_none"]
            if base.empty:
                continue    # no paired reference -> a difference would be meaningless
            b = pd.to_numeric(base.iloc[0][seed_cols], errors="coerce").to_numpy()
            cell = rows.setdefault(_row_title(proj, beh), {})
            for _, r in sub[sub["config"] != "baseline_none"].iterrows():
                v = pd.to_numeric(r[seed_cols], errors="coerce").to_numpy()
                for i, d in enumerate(v - b, start=1):
                    cell[f"{r['label']}:{i}"] = d
        if not rows:
            continue
        cols = [f"{lab}:{i}" for lab in labels
                for i in range(1, len(seed_cols) + 1)]
        table = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=cols)
        table.insert(0, "Behavior", table.index)
        out[str(budget)] = table.reset_index(drop=True)
    return out


# ── Cross-project: the per-project summary tables ────────────────────────────


def prism_publication_metrics(pub_df: pd.DataFrame) -> pd.DataFrame:
    """Column table: the reviewer-facing metric set, one row per project."""
    cols = {"f1": "F1", "mcc": "MCC", "balanced_accuracy": "Balanced accuracy",
            "roc_auc": "ROC-AUC", "cohen_kappa": "Cohen's kappa"}
    out = pd.DataFrame({"Project": pub_df["project_id"].astype(str).to_numpy()})
    for src, dst in cols.items():
        if src in pub_df.columns:
            out[dst] = pd.to_numeric(pub_df[src], errors="coerce").to_numpy()
    return out


def prism_project_accuracy(acc_df: pd.DataFrame) -> pd.DataFrame:
    """Column table (Mean, SD, N): held-out F1 per project."""
    out = pd.DataFrame({"Project": acc_df["project_id"].astype(str).to_numpy()})
    _mean_sd_n(out, "F1", acc_df["f1_mean"], acc_df.get("f1_ci"), acc_df.get("n"))
    return out.rename(columns={"F1:Mean": "F1", "F1:SD": "SD", "F1:N": "N"})


def prism_training_speed(speed_df: pd.DataFrame) -> pd.DataFrame:
    """Column table: training seconds per project.

    Returns empty when every project reports 0 s — that means the run's cells
    carried no timing (e.g. only rare-discovery ran), and a bar chart of zeros
    would read as "training is instant" rather than "not measured".
    """
    med = pd.to_numeric(speed_df.get("median_sec"), errors="coerce")
    if med is None or not (med.fillna(0) > 0).any():
        return pd.DataFrame()
    return pd.DataFrame({
        "Project": speed_df["project_id"].astype(str).to_numpy(),
        "Median seconds": med.to_numpy(),
        "Mean seconds": pd.to_numeric(speed_df.get("mean_sec"),
                                      errors="coerce").to_numpy(),
        "N fits": pd.to_numeric(speed_df.get("n"), errors="coerce").to_numpy(),
    })


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
# (mean column, ci column, filename). pos_discovered has no CI in the tidy rows,
# so it stays a plain one-column-per-dataset XY table.
_AL_METRICS = (("f1_mean", "f1_ci", "prism_al_curve_f1.csv"),
               ("pr_auc_mean", "pr_auc_ci", "prism_al_curve_pr_auc.csv"),
               ("pos_discovered_mean", None, "prism_al_curve_pos_discovered.csv"))


def prism_al_curves(al_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{filename: wide XY table}``. Shared X = clips reviewed; one dataset per
    project·behavior·strategy — one file each for F1, PR-AUC, positives found.

    F1 and PR-AUC ship as ``:Mean``/``:SD``/``:N`` triples (Prism XY, "Enter and
    plot error values" -> Mean, SD, N).  The per-seed values do not survive into
    the tidy AL frame, so the SD is reconstructed from the t-based CI and N; the
    CI half-width itself is not something Prism can plot.
    """
    df = al_df.copy()
    df["__col"] = (df["project_id"].astype(str) + " - " + df["behavior_name"].astype(str)
                   + " - " + df["strategy"].map(_AL_STRATEGY).fillna(df["strategy"]))
    out: dict[str, pd.DataFrame] = {}
    for metric, ci_col, fname in _AL_METRICS:
        if metric not in df.columns:
            continue
        wide = (df.pivot_table(index="n_clips_reviewed", columns="__col",
                               values=metric, sort=False)
                .reset_index().rename(columns={"n_clips_reviewed": "Clips reviewed"}))
        wide.columns.name = None
        if ci_col and ci_col in df.columns and "n_seeds" in df.columns:
            ci = df.pivot_table(index="n_clips_reviewed", columns="__col",
                                values=ci_col, sort=False)
            n = df.pivot_table(index="n_clips_reviewed", columns="__col",
                               values="n_seeds", sort=False)
            table = pd.DataFrame({"Clips reviewed": wide["Clips reviewed"]})
            for name in [c for c in wide.columns if c != "Clips reviewed"]:
                _mean_sd_n(table, name, wide[name].to_numpy(),
                           ci.get(name), n.get(name))
            wide = table
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
            f"{name} - confidence": pd.to_numeric(g["mean_confidence"], errors="coerce"),
            f"{name} - accuracy": pd.to_numeric(g["empirical_accuracy"], errors="coerce"),
        }))
    return pd.concat(blocks, axis=1) if blocks else pd.DataFrame()


# ── Time budget: true vs predicted prevalence, paired XY per behavior ────────


def prism_time_budget_agreement(tb_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{filename: table}`` — the agreement statistics and the Bland-Altman bias.

    Two files because the units don't mix: correlation coefficients on one axis,
    prevalence differences on another.  ``median_labeled_coverage`` rides along
    with the CCC because these are prevalences over *reviewed* segments, and the
    number is meaningless without knowing how much of the session that was.
    """
    df = tb_df
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    if df.empty:
        return {}
    title = [_row_title(p, b) for p, b in zip(df["project"], df["behavior"])]
    out: dict[str, pd.DataFrame] = {}

    agree = pd.DataFrame({"Behavior": title})
    for src, dst in (("prevalence_ccc", "Prevalence CCC"),
                     ("prevalence_pearson_r", "Pearson r"),
                     ("prevalence_r2", "R squared"),
                     ("bout_ccc", "Bout-count CCC"),
                     ("bout_pearson_r", "Bout-count Pearson r"),
                     ("n_sessions", "Sessions (N)"),
                     ("median_labeled_coverage", "Median labeled coverage")):
        if src in df.columns:
            agree[dst] = pd.to_numeric(df[src], errors="coerce").to_numpy()
    out["prism_time_budget_agreement.csv"] = agree

    if {"prevalence_bias", "loa_lower", "loa_upper"} <= set(df.columns):
        lo = pd.to_numeric(df["loa_lower"], errors="coerce").to_numpy()
        hi = pd.to_numeric(df["loa_upper"], errors="coerce").to_numpy()
        bias = pd.DataFrame({
            "Behavior": title,
            "Bias (pred - true)": pd.to_numeric(df["prevalence_bias"],
                                                errors="coerce").to_numpy(),
            # LoA = bias +/- 1.96 SD, so the SD of the differences is the span/3.92.
            # This is the number Prism needs to redraw the limits itself.
            "SD of differences": (hi - lo) / 3.92,
            "N sessions": pd.to_numeric(df.get("n_sessions"),
                                        errors="coerce").to_numpy(),
            "LoA lower": lo, "LoA upper": hi,
        })
        out["prism_time_budget_bias.csv"] = bias
    return out


# ── Feature roles: which modality does each behavior lean on? ────────────────


def prism_feature_roles(memb_df: pd.DataFrame) -> pd.DataFrame:
    """Grouped table (unequal n): one column per dominant modality, each behavior's
    over-pose ΔF1 in its own group.

    Ragged by construction — a behavior belongs to exactly one modality — which is
    what a Prism grouped table with unequal group sizes expects.  Paste and run
    Analyze -> Nonparametric -> Kruskal-Wallis to reproduce the reported test.
    """
    if memb_df.empty or "improvement_over_pose" not in memb_df.columns:
        return pd.DataFrame()
    key = ("own_dominant_modality" if "own_dominant_modality" in memb_df.columns
           else "cluster")
    groups = list(dict.fromkeys(memb_df[key].astype(str)))
    out = pd.DataFrame({"Behavior": memb_df["behavior"].astype(str).to_numpy()})
    vals = pd.to_numeric(memb_df["improvement_over_pose"], errors="coerce")
    for g in groups:
        out[g] = np.where(memb_df[key].astype(str) == g, vals, np.nan)
    return out


def prism_feature_roles_bars(bars_df: pd.DataFrame) -> pd.DataFrame:
    """Column table (Mean, SD, N): mean over-pose ΔF1 per dominant modality.

    The Kruskal-Wallis columns in the tidy file repeat one run-level statistic on
    every row; they belong in the README, not in a data column Prism would try to
    plot, so they are dropped here.
    """
    if bars_df.empty:
        return pd.DataFrame()
    label = ("dominant_modality" if "dominant_modality" in bars_df.columns
             else "cluster")
    out = pd.DataFrame({"Dominant modality": bars_df[label].astype(str).to_numpy()})
    _mean_sd_n(out, "dF1 over pose-only", bars_df["mean_improvement_over_pose"],
               bars_df.get("ci95"), bars_df.get("n_behaviors"))
    out = out.rename(columns={"dF1 over pose-only:Mean": "Mean dF1 over pose-only",
                              "dF1 over pose-only:SD": "SD",
                              "dF1 over pose-only:N": "N behaviors"})
    if "p_value" in bars_df.columns:
        out["p (one-sample t vs 0)"] = pd.to_numeric(bars_df["p_value"],
                                                     errors="coerce").to_numpy()
    return out


def prism_time_budget(tb_df: pd.DataFrame) -> pd.DataFrame:
    """Paired-XY: per behavior, a (true, pred) prevalence column pair — points are
    sessions. Paste as XY, plot the identity line, report the correlation in Prism."""
    blocks = []
    for (proj, beh), g in tb_df.groupby(["project", "behavior"], sort=False):
        g = g.reset_index(drop=True)
        name = _row_title(proj, beh)
        blocks.append(pd.DataFrame({
            f"{name} - true": pd.to_numeric(g["true_prevalence"], errors="coerce"),
            f"{name} - pred": pd.to_numeric(g["pred_prevalence"], errors="coerce"),
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


# ── Discrimination: the pooled landscape + volcano, as Prism scatters ────────
#
# The two figures in `plots.discrimination_landscape` are scatters where each point
# carries several variables at once (an x, a y, a categorical colour, a categorical
# shape, a size).  That is Prism's **Multiple variables** table — rows are
# observations, columns are variables — not the pre-pivoted Grouped layout the rest
# of this module emits, so these two writers deliberately stay long.
#
# No ci95 column is exported here: Prism has no input format for a CI half-width
# (see the error-bar note above), and pasting one anywhere it fits misstates it.


def _pair_title(df: pd.DataFrame) -> pd.Series:
    """The single row-title column Prism allows: ``Assay - A vs B``."""
    name = df["project_name"] if "project_name" in df.columns else df["project"]
    return name.astype(str) + " - " + df["pair"].astype(str)


def _neg_log10_p(p) -> np.ndarray:
    """``-log10(p)`` for a volcano's y-axis; NaN stays NaN, 0 is floored not infinite."""
    arr = pd.to_numeric(pd.Series(p), errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.log10(np.maximum(arr, 1e-12))
    return np.where(np.isfinite(arr), out, np.nan)


def _significant_flag(df: pd.DataFrame) -> np.ndarray:
    """1/0 rather than TRUE/FALSE — Prism groups and plots on numbers, not text."""
    return df["significant"].astype(str).str.lower().eq("true").astype(int).to_numpy()


def prism_discrimination_landscape(disc_df: pd.DataFrame) -> pd.DataFrame:
    """Multiple-variables table for the landscape panel: one row per behavior pair.

    ``PoseOnlyError`` (x) vs ``ErrorRemoved`` (y), grouped by ``BestFamily`` for
    colour and ``Assay`` for symbol.  ``HeldOutClips`` is there to drive Prism's
    variable point size.
    """
    if "best_family" not in disc_df.columns or "pose_only_auc" not in disc_df.columns:
        return pd.DataFrame()
    df = disc_df[disc_df["best_family"].astype(bool)].copy()
    if df.empty:
        return pd.DataFrame()
    pose = pd.to_numeric(df["pose_only_auc"], errors="coerce")
    out = pd.DataFrame({
        "Pair": _pair_title(df).to_numpy(),
        "Assay": (df["project_name"] if "project_name" in df.columns
                  else df["project"]).astype(str).to_numpy(),
        "PoseOnlyAUC": pose.to_numpy(),
        # The plotted x. Kept as its own column rather than asking the user to
        # compute 1-AUC in Prism, which has no calculated columns on a scatter.
        "PoseOnlyError": (1.0 - pose).to_numpy(),
        "BestFamily": df["label"].astype(str).str.lstrip("+ ").str.strip().to_numpy(),
        "ErrorRemoved": pd.to_numeric(df["error_reduction"], errors="coerce").to_numpy(),
        "DeltaAUC": pd.to_numeric(df["auc_gain_vs_pose"], errors="coerce").to_numpy(),
        "PValue": pd.to_numeric(df["p_value"], errors="coerce").to_numpy(),
        "NegLog10P": _neg_log10_p(df["p_value"]),
        "Significant": _significant_flag(df),
        "HeldOutClips": pd.to_numeric(df["n_holdout"], errors="coerce").to_numpy(),
    })
    return out.sort_values("PoseOnlyError", ascending=False).reset_index(drop=True)


def prism_discrimination_volcano(disc_df: pd.DataFrame) -> pd.DataFrame:
    """Multiple-variables table for the volcano: one row per pair x feature family.

    Every add-on family is kept (the pose-only baseline has no gain of its own), so
    a pair rescued by two families appears twice — which is the point of the panel.
    """
    needed = {"p_value", "error_reduction", "feature_set"}
    if not needed.issubset(disc_df.columns):
        return pd.DataFrame()
    df = disc_df[(disc_df["feature_set"].astype(str) != "pose_only")
                 & (disc_df["feature_set"].astype(str) != "")].copy()
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Pair": _pair_title(df).to_numpy(),
        "Assay": (df["project_name"] if "project_name" in df.columns
                  else df["project"]).astype(str).to_numpy(),
        "FeatureFamily": df["label"].astype(str).str.lstrip("+ ").str.strip().to_numpy(),
        "ErrorRemoved": pd.to_numeric(df["error_reduction"], errors="coerce").to_numpy(),
        "DeltaAUC": pd.to_numeric(df["auc_gain_vs_pose"], errors="coerce").to_numpy(),
        "PValue": pd.to_numeric(df["p_value"], errors="coerce").to_numpy(),
        "NegLog10P": _neg_log10_p(df["p_value"]),
        "Significant": _significant_flag(df),
        "PoseOnlyAUC": pd.to_numeric(df["pose_only_auc"], errors="coerce").to_numpy(),
        "HeldOutClips": pd.to_numeric(df["n_holdout"], errors="coerce").to_numpy(),
    })
    return out.reset_index(drop=True)


def prism_discrimination_seeds(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Grouped table: rows = pair, one replicate block of held-out ROC-AUC per family.

    The tables above are summaries; this is the raw material behind them, so Prism
    can run the paired test itself instead of taking our ``PValue`` on trust — the
    rule this module follows wherever the per-seed values survived.
    """
    if seed_df is None or seed_df.empty:
        return pd.DataFrame()
    df = seed_df.copy()
    df["__pair"] = _pair_title(df)
    pairs = list(dict.fromkeys(df["__pair"]))
    labels = list(dict.fromkeys(df["label"].astype(str)))
    n_rep = int(pd.to_numeric(df["seed_index"], errors="coerce").max() or 0)
    if n_rep < 1:
        return pd.DataFrame()

    out = pd.DataFrame({"Pair": pairs})
    for label in labels:
        sub = df[df["label"].astype(str) == label]
        by_pair = {p: g.sort_values("seed_index")["roc_auc"].tolist()
                   for p, g in sub.groupby("__pair")}
        _replicate_block(out, label.lstrip("+ ").strip(), by_pair, pairs, n_rep)
    return out


# ── Cross-project: accuracy (F1) by behavior ─────────────────────────────────


def prism_accuracy_by_behavior(beh_df: pd.DataFrame) -> pd.DataFrame:
    """Column table (Mean, SD, N): one row per project·behavior, held-out F1.

    Two fixes over a plain rename: the project and behavior are merged into the
    single row-title column Prism allows, and the CI half-width is converted to
    the SD Prism actually plots.  Pasting a ``f1_ci`` column into an SD subcolumn
    overstates every error bar by ``t(n)/sqrt(n)`` — at 3 seeds, 2.5x.
    """
    out = pd.DataFrame({
        "Behavior": [_row_title(p, b) for p, b in
                     zip(beh_df["project_id"], beh_df["behavior_name"])],
    })
    _mean_sd_n(out, "F1", beh_df["f1_mean"], beh_df.get("f1_ci"), beh_df.get("n"))
    return out.rename(columns={"F1:Mean": "F1", "F1:SD": "SD", "F1:N": "N"})


# ── Learning curves: how much labeling does a behavior need? ─────────────────

_LC_METRICS = (("f1_mean", "f1_ci", "prism_learning_curve_f1.csv"),
               ("pr_auc_mean", "pr_auc_ci", "prism_learning_curve_pr_auc.csv"))


def prism_learning_curves(lc_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{filename: wide XY table}`` — shared X = clips labeled, one dataset per
    project·behavior as a ``:Mean``/``:SD``/``:N`` triple.

    The per-seed fits live in ``cells.parquet``, not in the points frame, so the
    SD is reconstructed from the t-based CI and N.  Any across-behavior average
    curve sorts first, so it lands as Prism's dataset A.
    """
    df = lc_df.copy()
    df["__col"] = [_row_title(p, b) for p, b in
                   zip(df["project_id"], df["behavior_name"])]
    # The pooled "Average across N behaviors" curve is the headline line; Prism
    # colours datasets in paste order, so it must come first.
    is_avg = df["behavior_name"].astype(str).str.startswith("Average across")
    names = (list(dict.fromkeys(df.loc[is_avg, "__col"]))
             + list(dict.fromkeys(df.loc[~is_avg, "__col"])))

    out: dict[str, pd.DataFrame] = {}
    for metric, ci_col, fname in _LC_METRICS:
        if metric not in df.columns:
            continue
        piv = df.pivot_table(index="n_clips_mean", columns="__col", values=metric)
        if piv.isna().all().all():
            continue    # PR-AUC is not always computed; skip the empty file
        ci = (df.pivot_table(index="n_clips_mean", columns="__col", values=ci_col)
              if ci_col in df.columns else None)
        n = (df.pivot_table(index="n_clips_mean", columns="__col", values="n_seeds")
             if "n_seeds" in df.columns else None)
        table = pd.DataFrame({"Clips labeled": piv.index.to_numpy()})
        for name in [c for c in names if c in piv.columns]:
            _mean_sd_n(table, name, piv[name].to_numpy(),
                       None if ci is None else ci.get(name),
                       None if n is None else n.get(name))
        out[fname] = table
    return out


def prism_learning_curve_knee(knee_df: pd.DataFrame) -> pd.DataFrame:
    """Column table: the saturation point (knee) and max F1 per behavior."""
    return pd.DataFrame({
        "Behavior": [_row_title(p, b) for p, b in
                     zip(knee_df["project_id"], knee_df["behavior_name"])],
        "Knee clips": pd.to_numeric(knee_df["knee_clips"], errors="coerce").to_numpy(),
        "Max F1": pd.to_numeric(knee_df["f1_max"], errors="coerce").to_numpy(),
    })


def prism_confusion(conf_df: pd.DataFrame) -> pd.DataFrame:
    """Grouped/stacked-bar table: held-out TP/FN/FP/TN per project·behavior.

    Column order is Found / Missed / False alarm so a stacked bar built straight
    from columns 1-3 reads left-to-right as agreement then the two error types.
    True negatives come last and are meant to be left out of the plot — under this
    imbalance they would flatten every other segment (see
    :func:`abel.validation.plots.confusion_counts_by_behavior`).
    """
    cols = {"tp": "Found (TP)", "fn": "Missed (FN)", "fp": "False alarm (FP)",
            "tn": "True negative (TN)", "n_pos_val": "Reviewer positives",
            "n_val": "Held-out clips", "precision": "Precision", "recall": "Recall",
            "clip_sec": "Clip length (s)"}
    out = conf_df.copy()
    out.insert(0, "Behavior", (out["project_id"].astype(str) + " · "
                               + out["behavior_name"].astype(str)))
    keep = ["Behavior"] + [c for c in cols if c in out.columns]
    return out[keep].rename(columns=cols)


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
  In Prism's New Table dialog choose "Grouped" (or "XY") -> "Enter and plot
  replicate values" with N side-by-side subcolumns, then paste. Prism will then run
  the test itself (e.g. Analyze -> t tests -> Paired) instead of you re-typing an
  asterisk, and will draw the error bars from the replicates.
* Where only summary statistics survived the analysis, tables carry
  "<group>:Mean / :SD / :N" triples instead -- choose "Enter and plot error values"
  -> "Mean, SD, N". The SD is reconstructed from the t-based 95% CI and N. Never
  paste a "95% CI" half-width into an SD subcolumn; Prism has no CI input format.
* Row titles are "project - behavior" in a single column, because Prism accepts
  exactly one row-title column.
* Blank cells are real: they mean "not measured / never reached", and Prism reads
  them as missing rather than zero. Leave them blank.
* Text is ASCII-only and every file is written UTF-8 with a BOM, so Windows Excel
  and Prism's CSV import read the headers correctly instead of showing mojibake.
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
              discrimination_seeds_df: pd.DataFrame | None = None,
              accuracy_by_behavior_df: pd.DataFrame | None = None,
              confusion_df: pd.DataFrame | None = None,
              lc_points_df: pd.DataFrame | None = None,
              lc_knee_df: pd.DataFrame | None = None,
              time_budget_agreement_df: pd.DataFrame | None = None,
              feature_roles_df: pd.DataFrame | None = None,
              feature_roles_bars_df: pd.DataFrame | None = None,
              publication_metrics_df: pd.DataFrame | None = None,
              project_accuracy_df: pd.DataFrame | None = None,
              training_speed_df: pd.DataFrame | None = None) -> list[Path]:
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
        for budget, table in prism_ablation_gain(ablation_df).items():
            written.append(_write(table, out_dir / f"prism_ablation_gain_{budget}.csv"))
        for budget, mat in prism_ablation_gain_matrix(ablation_df).items():
            written.append(_write(mat, out_dir / f"prism_ablation_gain_matrix_{budget}.csv"))
        n_abl_seeds = len(_seed_cols(ablation_df, "f1_seed"))
        for budget, table in prism_ablation_gain_seeds(ablation_df).items():
            written.append(_write(
                table, out_dir / f"prism_ablation_gain_seeds_{budget}.csv"))
        sections.append(
            "prism_ablation_<budget>.csv\n    Table: Grouped, one column group per\n"
            "    config. One FILE per clip budget -- Prism grids are 2-factor and the\n"
            "    ablation crosses 4 (project x behavior x budget x config).\n"
            "prism_ablation_gain_<budget>.csv\n    Table: Grouped. dF1 vs. the\n"
            "    pose-only baseline, already differenced, each column followed by its\n"
            "    exact p.\n"
            f"prism_ablation_gain_seeds_<budget>.csv\n    Table: Grouped, "
            f"{n_abl_seeds or 1} replicate(s),\n"
            "    'Enter and plot replicate values'. PAIRED per-seed dF1 (same seed,\n"
            "    same subsample). Analyze -> t tests -> One sample t test vs 0 to\n"
            "    reproduce the p-values yourself.\n"
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

        land = prism_discrimination_landscape(discrimination_df)
        if not land.empty:
            written.append(_write(land, out_dir / "prism_discrimination_landscape.csv"))
            sections.append(
                "prism_discrimination_landscape.csv\n    Table: Multiple variables.\n"
                "    One row per behavior pair. Plot PoseOnlyError (X, log scale) vs\n"
                "    ErrorRemoved (Y); colour by BestFamily, symbol by Assay, size by\n"
                "    HeldOutClips. Significant is 1/0.\n")

        vol = prism_discrimination_volcano(discrimination_df)
        if not vol.empty:
            written.append(_write(vol, out_dir / "prism_discrimination_volcano.csv"))
            sections.append(
                "prism_discrimination_volcano.csv\n    Table: Multiple variables.\n"
                "    One row per pair x feature family. Plot ErrorRemoved (X) vs\n"
                "    NegLog10P (Y); colour by FeatureFamily. A pair rescued by two\n"
                "    families appears once per family.\n")

    if discrimination_seeds_df is not None and not discrimination_seeds_df.empty:
        seeds = prism_discrimination_seeds(discrimination_seeds_df)
        if not seeds.empty:
            written.append(_write(seeds, out_dir / "prism_discrimination_seeds.csv"))
            sections.append(
                "prism_discrimination_seeds.csv\n    Table: Grouped, replicate values.\n"
                "    Rows = pair, one <family>:1..N block per feature family: the raw\n"
                "    per-seed held-out ROC-AUC. Use this to re-run the paired test in\n"
                "    Prism rather than pasting our PValue.\n")

    if accuracy_by_behavior_df is not None and not accuracy_by_behavior_df.empty:
        written.append(_write(prism_accuracy_by_behavior(accuracy_by_behavior_df),
                              out_dir / "prism_accuracy_by_behavior.csv"))
        sections.append(
            "prism_accuracy_by_behavior.csv\n    Table: Column/bar. Pooled held-out\n"
            "    F1 + 95% CI per project·behavior.\n")

    if confusion_df is not None and not confusion_df.empty:
        written.append(_write(prism_confusion(confusion_df),
                              out_dir / "prism_confusion.csv"))
        sections.append(
            "prism_confusion.csv\n    Table: Grouped, stacked bar. Rows = project\n"
            "    (dot) behavior; plot the first three columns (TP/FN/FP) stacked and\n"
            "    leave TN out -- it is 10-100x the others and flattens the plot.\n"
            "    Counts are per fit, averaged over seeds, and the unit is one\n"
            "    reviewer-scored CLIP (a fraction of a second in most projects),\n"
            "    not a bout.\n")

    if lc_points_df is not None and not lc_points_df.empty:
        for fname, table in prism_learning_curves(lc_points_df).items():
            written.append(_write(table, out_dir / fname))
        sections.append(
            "prism_learning_curve_<metric>.csv\n    Table: XY, 'Enter and plot error\n"
            "    values' -> Mean, SD, N (3 subcolumns per dataset). Shared X = clips\n"
            "    labeled; one dataset per project (dot) behavior, the across-behavior\n"
            "    average first. SD is reconstructed from the t-based 95% CI and N, so\n"
            "    Prism's error bars match the PNG.\n")

    if lc_knee_df is not None and not lc_knee_df.empty:
        written.append(_write(prism_learning_curve_knee(lc_knee_df),
                              out_dir / "prism_learning_curve_knee.csv"))
        sections.append(
            "prism_learning_curve_knee.csv\n    Table: Column. Saturation point\n"
            "    (knee) in clips and the max F1 reached, per behavior.\n")

    if time_budget_agreement_df is not None and not time_budget_agreement_df.empty:
        for fname, table in prism_time_budget_agreement(time_budget_agreement_df).items():
            written.append(_write(table, out_dir / fname))
        sections.append(
            "prism_time_budget_agreement.csv\n    Table: Column. Lin's CCC / r / R2\n"
            "    per behavior, with the median labeled coverage that must be quoted\n"
            "    beside them -- these are prevalences over REVIEWED segments.\n"
            "prism_time_budget_bias.csv\n    Table: Column, Mean + SD + N. Bland-\n"
            "    Altman bias and the SD of the per-session differences\n"
            "    (LoA = bias +/- 1.96 SD).\n")

    if feature_roles_df is not None and not feature_roles_df.empty:
        written.append(_write(prism_feature_roles(feature_roles_df),
                              out_dir / "prism_feature_roles.csv"))
        sections.append(
            "prism_feature_roles.csv\n    Table: Grouped, unequal n. One column per\n"
            "    dominant modality; each behavior's over-pose dF1 sits in its own\n"
            "    group. Analyze -> Nonparametric -> Kruskal-Wallis reproduces the test.\n")

    if feature_roles_bars_df is not None and not feature_roles_bars_df.empty:
        written.append(_write(prism_feature_roles_bars(feature_roles_bars_df),
                              out_dir / "prism_feature_roles_bars.csv"))
        sections.append(
            "prism_feature_roles_bars.csv\n    Table: Column, Mean + SD + N.\n"
            "    One bar per feature modality: mean dF1 over pose-only.\n")

    if publication_metrics_df is not None and not publication_metrics_df.empty:
        written.append(_write(prism_publication_metrics(publication_metrics_df),
                              out_dir / "prism_publication_metrics.csv"))
        sections.append(
            "prism_publication_metrics.csv\n    Table: Column. F1 / MCC / balanced\n"
            "    accuracy / ROC-AUC / kappa per project -- the reviewer summary.\n")

    if project_accuracy_df is not None and not project_accuracy_df.empty:
        written.append(_write(prism_project_accuracy(project_accuracy_df),
                              out_dir / "prism_accuracy_by_project.csv"))
        sections.append(
            "prism_accuracy_by_project.csv\n    Table: Column, Mean + SD + N.\n"
            "    Held-out F1 per project.\n")

    if training_speed_df is not None and not training_speed_df.empty:
        speed = prism_training_speed(training_speed_df)
        if not speed.empty:      # all-zero timings are "not measured", not "instant"
            written.append(_write(speed, out_dir / "prism_training_speed.csv"))
            sections.append(
                "prism_training_speed.csv\n    Table: Column. Median/mean training\n"
                "    seconds per project.\n")

    if written:
        written.append(write_text(out_dir / "README_PRISM.txt",
                                  _README.format(sections="\n".join(sections))))
    return written
