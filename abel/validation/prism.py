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
from contextlib import contextmanager
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
    # ``.get`` returns None when the column is absent, and to_numeric(None) raises —
    # which the export's error guard swallows, taking the whole panel with it.
    raw = gen_df["human_ceiling_kappa"] if "human_ceiling_kappa" in gen_df.columns \
        else None
    ceiling = pd.to_numeric(raw, errors="coerce") if raw is not None else None
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


def prism_video_gain(vv_df: pd.DataFrame) -> pd.DataFrame:
    """Column table: the video ΔF1 per behavior, sorted, with its own error bar.

    The paired table above is what you re-run the test from; this is the panel the
    figure actually draws — one bar per behavior, ascending, so the reader sees the
    distribution of the effect rather than two adjacent means.  SD is recovered
    from the stored CI half-width (see :func:`sd_from_ci95`); the exact p and the
    significance flag ride along so the asterisks are not retyped by hand.
    """
    if vv_df is None or vv_df.empty or "gain" not in vv_df.columns:
        return pd.DataFrame()
    df = vv_df[vv_df.get("error").isna() | (vv_df.get("error") == "")] \
        if "error" in vv_df.columns else vv_df
    if df.empty:
        return pd.DataFrame()
    n = pd.to_numeric(df.get("n_seeds"), errors="coerce")
    out = pd.DataFrame({
        "Behavior": [_row_title(p, b) for p, b in
                     zip(df["project_id"], df["behavior_name"])],
    })
    _mean_sd_n(out, "Video improvement (dF1)",
               pd.to_numeric(df["gain"], errors="coerce").to_numpy(),
               pd.to_numeric(df.get("gain_ci95"), errors="coerce").to_numpy(),
               n.to_numpy())
    if "p_value" in df.columns:
        out["PValue"] = pd.to_numeric(df["p_value"], errors="coerce").to_numpy()
    if "significant" in df.columns:
        out["Significant"] = df["significant"].astype(bool).astype(int).to_numpy()
    return out.sort_values("Video improvement (dF1):Mean", ignore_index=True)


# ── Per-behavior metrics: rarity landscape + the per-assay kappa inset ──────


#: Metric column stem -> the label the panel prints.  PR-AUC first: it is the
#: imbalance-sensitive axis, and macro F1 is deliberately absent (its ~0.5 floor
#: on a rare behavior is precisely the artefact a rarity panel would report as a
#: finding).
_BEHAVIOR_METRIC_LABELS = [
    ("pr_auc", "PR-AUC"),
    ("f1", "Target-class F1"),
    ("cohen_kappa", "Cohen's kappa"),
]


def prism_rarity_vs_performance(rarity_df: pd.DataFrame,
                                metrics_df: pd.DataFrame) -> pd.DataFrame:
    """XY Mean/SD/N: deployment prevalence (X) against every headline metric (Y).

    X is the percent of session time the behavior occupies, measured from dense
    bout detections over whole sessions — set the X axis to log scale, which is
    where two orders of magnitude of rarity become readable.

    The join is on (project, behavior), never on behavior name alone: the same
    name in two assays is two different behaviors with two different prevalences,
    and joining on the name would cross-multiply them.  Behaviors missing from
    either side are dropped rather than zero-filled — an unmeasured prevalence is
    not a prevalence of zero, and at the rare end that lie would land on the
    exact points the panel exists to show.
    """
    if (rarity_df is None or rarity_df.empty
            or metrics_df is None or metrics_df.empty):
        return pd.DataFrame()
    keys = ["project_id", "behavior_name"]
    if not (set(keys) <= set(rarity_df.columns)
            and set(keys) <= set(metrics_df.columns)):
        return pd.DataFrame()
    j = metrics_df.merge(rarity_df, on=keys, how="inner")
    j = j[pd.to_numeric(j["prevalence_pct"], errors="coerce").notna()]
    if j.empty:
        return pd.DataFrame()
    kept = set(map(tuple, j[keys].to_numpy()))
    dropped = sorted(_row_title(p, b) for p, b in
                     map(tuple, metrics_df[keys].to_numpy()) if (p, b) not in kept)
    j = j.sort_values("prevalence_pct", ignore_index=True)
    out = pd.DataFrame({
        "Prevalence (% of session time)": pd.to_numeric(j["prevalence_pct"],
                                                        errors="coerce").to_numpy(),
    })
    n = pd.to_numeric(j.get("n_seeds"), errors="coerce").to_numpy()
    for stem, label in _BEHAVIOR_METRIC_LABELS:
        if f"{stem}_mean" not in j.columns:
            continue
        out[f"{label}:Mean"] = pd.to_numeric(j[f"{stem}_mean"],
                                             errors="coerce").to_numpy()
        # Already an SD across seeds -- do NOT route it through sd_from_ci95.
        out[f"{label}:SD"] = pd.to_numeric(j.get(f"{stem}_sd"),
                                           errors="coerce").to_numpy()
        out[f"{label}:N"] = n
    # Carried so a point can be identified without re-joining anything, and so the
    # caption can state which prevalence source the axis came from.
    out["Behavior"] = [_row_title(p, b) for p, b in
                       zip(j["project_id"], j["behavior_name"])]
    out["Prevalence SD"] = pd.to_numeric(j.get("prevalence_sd"),
                                         errors="coerce").to_numpy()
    out["n sessions"] = pd.to_numeric(j.get("n_sessions"), errors="coerce").to_numpy()
    if "source" in j.columns:
        out["Prevalence source"] = j["source"].astype(str).to_numpy()
    # A behavior silently missing from a scatter is invisible — the reader counts
    # points and believes it is the whole set. Carried out so INDEX.txt can name it.
    out.attrs["dropped"] = dropped
    return out


def prism_metric_by_assay(metrics_df: pd.DataFrame,
                          metric: str = "cohen_kappa") -> pd.DataFrame:
    """Ragged Column table: one column per assay, one observation per behavior.

    The compact per-assay view — paste it and Prism runs the one-way ANOVA /
    Kruskal-Wallis across assays directly.  Columns are ragged by construction
    (assays have different behavior counts); the short ones are blank-padded,
    which Prism reads as missing rather than as zero.
    """
    col = f"{metric}_mean"
    if metrics_df is None or metrics_df.empty or col not in metrics_df.columns:
        return pd.DataFrame()
    df = metrics_df[["project_id", col]].copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df[col].notna()]
    if df.empty:
        return pd.DataFrame()
    cols = {}
    for assay, grp in df.groupby("project_id", dropna=False):
        cols[str(assay)] = sorted(grp[col].to_numpy(), reverse=True)
    width = max(len(v) for v in cols.values())
    return pd.DataFrame({k: list(v) + [np.nan] * (width - len(v))
                         for k, v in sorted(cols.items())})


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
        # Reindex the CI/N pivots onto the value pivot's index: pivot_table can drop
        # an all-NaN x from one metric but not another (e.g. a cold-start row with
        # NaN F1 but a real count), leaving the columns different lengths.
        ci = (df.pivot_table(index="n_clips_mean", columns="__col", values=ci_col)
              .reindex(piv.index) if ci_col in df.columns else None)
        n = (df.pivot_table(index="n_clips_mean", columns="__col", values="n_seeds")
             .reindex(piv.index) if "n_seeds" in df.columns else None)
        table = pd.DataFrame({"Clips labeled": piv.index.to_numpy()})
        for name in [c for c in names if c in piv.columns]:
            _mean_sd_n(table, name, piv[name].to_numpy(),
                       None if ci is None else ci.get(name),
                       None if n is None else n.get(name))
        out[fname] = table
    return out


# The companion to the F1/PR-AUC curves: the held-out error burden that shrinks
# as clips are added, as a percent of the fixed held-out set. False alarms (FP)
# and misses (FN) are the two error types the counts figure plots; true positives
# are omitted because TP = P - FN carries no information the misses don't.
_LC_ERROR_METRICS = (
    ("fp_pct", "fp_pct_ci", "False alarms (FP %)", "prism_learning_curve_error_fp.csv"),
    ("fn_pct", "fn_pct_ci", "Misses (FN %)", "prism_learning_curve_error_fn.csv"),
)


def prism_learning_curve_errors(lc_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{filename: wide XY table}`` for the held-out error-rate curves.

    Same shape as :func:`prism_learning_curves` — shared X = clips labeled, one
    ``:Mean``/``:SD``/``:N`` dataset per project·behavior with the across-behavior
    average first — but the Y is the confusion **rate** (percent of the held-out
    set): one file for false alarms (FP %), one for misses (FN %).  SD is rebuilt
    from the same 95% CI columns, so these bars match the F1/PR-AUC export.
    """
    df = lc_df.copy()
    df["__col"] = [_row_title(p, b) for p, b in
                   zip(df["project_id"], df["behavior_name"])]
    is_avg = df["behavior_name"].astype(str).str.startswith("Average across")
    names = (list(dict.fromkeys(df.loc[is_avg, "__col"]))
             + list(dict.fromkeys(df.loc[~is_avg, "__col"])))

    out: dict[str, pd.DataFrame] = {}
    for metric, ci_col, _label, fname in _LC_ERROR_METRICS:
        if metric not in df.columns:
            continue
        piv = df.pivot_table(index="n_clips_mean", columns="__col", values=metric)
        if piv.isna().all().all():
            continue
        # Align CI/N onto the value index (see prism_learning_curves) so a row
        # dropped from one metric's pivot cannot misalign the SD column.
        ci = (df.pivot_table(index="n_clips_mean", columns="__col", values=ci_col)
              .reindex(piv.index) if ci_col in df.columns else None)
        n = (df.pivot_table(index="n_clips_mean", columns="__col", values="n_seeds")
             .reindex(piv.index) if "n_seeds" in df.columns else None)
        table = pd.DataFrame({"Clips labeled": piv.index.to_numpy()})
        for name in [c for c in names if c in piv.columns]:
            _mean_sd_n(table, name, piv[name].to_numpy(),
                       None if ci is None else ci.get(name),
                       None if n is None else n.get(name))
        out[fname] = table

    # The direct companion to the headline F1/PR-AUC average graph: a single file
    # with the pooled curve's two error lines (FP % and FN %) side by side, so it
    # pastes as one XY table and plots as the two-line error-rate figure.
    avg = df[is_avg]
    if not avg.empty:
        combo = pd.DataFrame({"Clips labeled": sorted(avg["n_clips_mean"].unique())})
        for metric, ci_col, label, _fname in _LC_ERROR_METRICS:
            if metric not in avg.columns:
                continue
            a = (avg[["n_clips_mean", metric, ci_col, "n_seeds"]]
                 .drop_duplicates("n_clips_mean")
                 .set_index("n_clips_mean").reindex(combo["Clips labeled"]))
            _mean_sd_n(combo, label, a[metric].to_numpy(),
                       a[ci_col].to_numpy() if ci_col in a else None,
                       a["n_seeds"].to_numpy() if "n_seeds" in a else None)
        out["prism_learning_curve_error_rate_average.csv"] = combo
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


@contextmanager
def _guard(errors: list[str], label: str):
    """Isolate one table's build+write so a single bad pivot cannot sink the rest.

    Every export block runs inside ``with _guard(errors, "<name>"):`` — an
    exception is recorded and the export moves on to the next table, instead of
    aborting the whole run's Prism output (which is how a run "stops short" of
    exporting what it needs).  Skipped tables are listed at the end of the README.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 — one table's failure must not cascade
        errors.append(f"{label}: {type(exc).__name__}: {exc}")


# ── The curated per-panel set ───────────────────────────────────────────────
#
# ``write_all`` below is exhaustive on purpose: it is the archive, and on a
# multi-project run it emits several hundred files (one per project, per behavior,
# per feature family).  That is the right thing for a table you have to go looking
# for and the wrong thing for the ten you plot every time — nobody should have to
# work out which of 354 filenames is the headline curve.
#
# So the panels behind the manuscript figure are ALSO written, once, under
# ``prism/FIGURES/`` with names that say which panel they are.  Each is a single
# pre-pivoted block: paste it into a Prism table of the stated type and it draws.
# Nothing here is a new computation — every table is the same builder ``write_all``
# uses, so the curated copy and the archive copy can never disagree.

def _pooled_lc_panel(lc_df: pd.DataFrame) -> pd.DataFrame:
    """The across-behaviors learning curve alone: F1 and PR-AUC on one X.

    ``prism_learning_curves`` puts every behavior in the file with the average
    first; this keeps only the average, which is the line the figure plots, and
    pairs both metrics against the shared X so one paste draws the whole panel.

    The N column is the **behavior** count, not ``n_seeds``.  It used to be n_seeds
    summed across surviving behaviors (215 at the left edge falling to 140 at the
    right), which a reader had to divide by the seed count to recover the "28 of 43
    behaviors" that actually matters — a disclosure in the wrong units is not a
    disclosure.  A separate explicit column carries the composition so the attrition
    is legible without arithmetic.
    """
    is_avg = lc_df["behavior_name"].astype(str).str.startswith("Average across")
    avg = lc_df[is_avg]
    if avg.empty:
        return pd.DataFrame()
    xs = sorted(pd.to_numeric(avg["n_clips_mean"], errors="coerce").dropna().unique())
    out = pd.DataFrame({"Clips labeled": xs})
    a = (avg.drop_duplicates("n_clips_mean").set_index("n_clips_mean").reindex(xs))
    n_col = "n_behaviors" if "n_behaviors" in a.columns else "n_seeds"
    for metric, ci_col, label in (
        ("f1_mean", "f1_ci", "Target-class F1"),
        ("pr_auc_mean", "pr_auc_ci", "PR-AUC"),
    ):
        if metric not in a.columns:
            continue
        _mean_sd_n(out, label, a[metric].to_numpy(),
                   a[ci_col].to_numpy() if ci_col in a.columns else None,
                   a[n_col].to_numpy() if n_col in a.columns else None)
    # Composition and health, plainly named — a plateau read off the F1 column is
    # only a plateau if these hold still across the same rows.
    for src, label in (("n_behaviors", "n behaviors contributing"),
                       ("mcc_mean", "MCC"),
                       ("specificity_mean", "Specificity"),
                       ("n_degenerate", "n degenerate seeds"),
                       ("n_calibrated", "n calibrated seeds")):
        if src in a.columns:
            out[label] = pd.to_numeric(a[src], errors="coerce").to_numpy()
    return out


def _pooled_stat_panel(df: pd.DataFrame, title_col: str, keep: list[tuple[str, str]],
                       ) -> pd.DataFrame:
    """A pooled-statistics table as one row-title column plus named value columns."""
    if df is None or df.empty or title_col not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({title_col.replace("_", " ").title(): df[title_col].astype(str)})
    for src, label in keep:
        if src in df.columns:
            out[label] = pd.to_numeric(df[src], errors="coerce").to_numpy()
    return out


#: ``filename -> (Prism table type, what the panel shows)``, written into
#: ``FIGURES/INDEX.txt`` so the folder explains itself.
_PANEL_NOTES: dict[str, tuple[str, str]] = {
    "fig3_learning_curve_pooled.csv": (
        "XY, Mean/SD/N",
        "Data efficiency. X = clips labeled, Y = target-class F1 and PR-AUC "
        "averaged across behaviors. NOTE: the average is survivorship-biased at the "
        "high-clip end -- rare behaviors do not have that many labels and drop out "
        "of the mean -- so the composition CHANGES along the curve. N is the "
        "behavior count at each x ('n behaviors contributing'); part of any "
        "right-hand flattening is that denominator shrinking, not saturation. For a "
        "plateau claim use learning_curve_average_balanced.csv, which holds the "
        "cohort fixed. 'n degenerate seeds' counts collapsed fits, which score well "
        "on target-class F1 and are excluded from the plotted line and the knee."),
    "fig3_learning_curve_by_behavior.csv": (
        "XY, Mean/SD/N",
        "The same curve broken out per behavior; the pooled line is dataset A."),
    "fig3_learning_curve_knee.csv": (
        "Column",
        "Saturation point (knee clips) and max target-class F1 per behavior. The "
        "knee is the first clip count reaching 98% of that behavior's own maximum "
        "F1 with a marginal gain under 0.01 -- quote 98%, not 95%."),
    "fig3_ablation_gain_seeds.csv": (
        "Grouped, replicates, 'Enter and plot replicate values'",
        "Per-seed PAIRED dF1 over the pose-only baseline (same seed, same "
        "subsample). Analyze -> t tests -> One sample t test vs 0 reproduces the "
        "per-behavior p-values."),
    "fig3_ablation_pooled_by_behavior.csv": (
        "Column",
        "The manuscript statistic: each enhancement's dF1 pooled ACROSS BEHAVIORS "
        "(seeds averaged within behavior first), so n = behaviors, not seeds."),
    "fig3_discrimination_landscape.csv": (
        "XY / Volcano",
        "One point per behavior pair: pose-only separability against what each "
        "feature family adds."),
    "fig3_discrimination_pooled_by_pair.csv": (
        "Column",
        "Each feature family's effect pooled ACROSS PAIRS with headroom, so n = "
        "pairs. Pairs the pose baseline already solves are excluded -- they cannot "
        "move and would only dilute the mean."),
    "fig3_discrimination_volcano.csv": (
        "Multiple variables / Volcano",
        "One row per behavior pair x feature family. Plot ErrorRemoved (X) vs "
        "NegLog10P (Y), coloured by FeatureFamily. A pair rescued by two families "
        "appears once per family, so do not count rows as pairs."),
    "fig3_generalization_kappa.csv": (
        "Column",
        "Held-out agreement per behavior: Cohen's kappa and target-class F1."),
    "fig3_kappa_by_assay.csv": (
        "Column",
        "The same agreement collapsed to one column per assay, one observation "
        "per behavior. Ragged by design -- assays have different behavior counts, "
        "and the short columns are blank-padded, which Prism reads as missing. "
        "Analyze -> One-way ANOVA / Kruskal-Wallis compares assays."),
    "fig3_rarity_vs_performance.csv": (
        "XY, Mean/SD/N",
        "Performance against deployment rarity. X = percent of session time from "
        "dense bout detections over whole sessions -- SET THE X AXIS TO LOG. This "
        "is NOT the 'prevalence' in combined_across_projects.csv, which is the "
        "positive share of an enriched candidate pool. Behaviors with no measured "
        "prevalence are dropped, not zero-filled."),
    "fig3_video_value.csv": (
        "Column",
        "Video dF1 per behavior, sorted ascending, with SD and the exact paired "
        "p-value. The seed-level table for re-running the test yourself is "
        "prism_video_value.csv in the parent folder."),
    "fig3_modality_shares.csv": (
        "Grouped / stacked bar",
        "Share of each behavior's feature importance carried by each modality "
        "(pose geometry, kinematics, video, context). Rows sum to 100%."),
    "fig3_feature_importance_heatmap.csv": (
        "Heatmap",
        "Rows = feature, columns = behavior, cell = importance. All-zero feature "
        "rows are dropped so the map is not mostly blank."),
    "fig3_al_vs_random_f1.csv": (
        "XY, Mean/SD/N",
        "Active learning vs random: target-class F1 against clips reviewed."),
    "fig3_al_vs_random_positives.csv": (
        "XY, Mean/SD/N",
        "Active learning vs random: positives discovered against clips reviewed. "
        "This is the robust win -- lead with it, not with F1."),
    "fig3_rare_discovery_pooled.csv": (
        "XY, Mean/SD/N",
        "Clip hunting pooled over every model: positives found against clips "
        "reviewed, one dataset per acquisition strategy."),
    "fig3_rare_effort_minutes.csv": (
        "Grouped",
        "Minutes of human review each strategy costs to reach the target. Quote "
        "the assumed seconds-per-clip review rate in the legend."),
}


def write_figure_panels(out_dir: Path, **frames) -> list[Path]:
    """Write the curated per-panel Prism tables into ``out_dir/prism/FIGURES``.

    Accepts the same frames as :func:`write_all` and silently skips any panel whose
    source analysis did not run in this run.
    """
    fig_dir = Path(out_dir) / "prism" / "FIGURES"
    written: list[Path] = []
    errors: list[str] = []

    def _emit(name: str, table) -> None:
        if table is not None and not table.empty:
            written.append(_write(table, fig_dir / name))

    lc = frames.get("lc_points_df")
    if lc is not None and not lc.empty:
        with _guard(errors, "fig3 learning curve"):
            _emit("fig3_learning_curve_pooled.csv", _pooled_lc_panel(lc))
            curves = prism_learning_curves(lc)
            _emit("fig3_learning_curve_by_behavior.csv",
                  curves.get("prism_learning_curve_f1.csv"))
    knee = frames.get("lc_knee_df")
    if knee is not None and not knee.empty:
        with _guard(errors, "fig3 learning-curve knee"):
            _emit("fig3_learning_curve_knee.csv", prism_learning_curve_knee(knee))

    abl = frames.get("ablation_df")
    if abl is not None and not abl.empty:
        with _guard(errors, "fig3 ablation"):
            seeds = prism_ablation_gain_seeds(abl)
            # The full-data budget is the shipped-pipeline comparison; if a run used
            # only a low-data budget, fall back to whatever single budget it has.
            key = "all" if "all" in seeds else (next(iter(seeds), None))
            if key is not None:
                _emit("fig3_ablation_gain_seeds.csv", seeds[key])
    abl_pooled = frames.get("ablation_pooled_df")
    if abl_pooled is not None and not abl_pooled.empty:
        with _guard(errors, "fig3 ablation pooled"):
            _emit("fig3_ablation_pooled_by_behavior.csv", _pooled_stat_panel(
                abl_pooled, "label", [
                    ("clip_budget", "Clip budget"), ("n_behaviors", "n behaviors"),
                    ("n_projects", "n projects"),
                    ("n_untestable", "n behaviors untestable (excluded)"),
                    ("mean_gain", "Mean dF1"), ("median_gain", "Median dF1"),
                    ("estimate", "Mixed-model dF1"),
                    ("ci95", "95% CI half-width"),
                    ("df", "df"), ("p_value", "PValue"),
                    ("q_value", "QValue (BH)"),
                    ("icc_project", "ICC (project)"),
                    ("n_effective", "Effective n"),
                    ("p_naive_behavior", "PValue (behavior-as-unit, uncorrected)"),
                    ("p_project_mean", "PValue (project means)"),
                    ("p_sign_flip", "PValue (cluster sign-flip)"),
                    ("n_helped", "n behaviors improved"),
                    ("n_projects_helped", "n projects improved"),
                ]))

    disc = frames.get("discrimination_df")
    if disc is not None and not disc.empty:
        with _guard(errors, "fig3 discrimination"):
            _emit("fig3_discrimination_landscape.csv",
                  prism_discrimination_landscape(disc))
            _emit("fig3_discrimination_volcano.csv",
                  prism_discrimination_volcano(disc))
    disc_pooled = frames.get("discrimination_pooled_df")
    if disc_pooled is not None and not disc_pooled.empty:
        with _guard(errors, "fig3 discrimination pooled"):
            _emit("fig3_discrimination_pooled_by_pair.csv", _pooled_stat_panel(
                disc_pooled, "label", [
                    ("n_pairs", "n pairs"), ("n_projects", "n projects"),
                    ("mean_auc_gain", "Mean dAUC"),
                    ("estimate", "Mixed-model dAUC"),
                    ("ci95_auc_gain", "95% CI half-width"),
                    ("df", "df"), ("p_value", "PValue"),
                    ("q_value", "QValue (BH)"),
                    ("icc_project", "ICC (project)"),
                    ("p_naive_pair", "PValue (pair-as-unit, uncorrected)"),
                    ("p_project_mean", "PValue (project means)"),
                    ("p_sign_flip", "PValue (cluster sign-flip)"),
                    ("median_error_reduction", "Median error reduction"),
                    ("n_improved", "n pairs improved"),
                    # Headroom-selection exposure — see pooled_gain_by_pair.
                    ("n_pairs_excluded", "n pairs excluded (below headroom)"),
                    ("mean_gain_excluded", "Mean dAUC of excluded pairs"),
                    ("frac_near_cutoff", "Frac kept pairs near cutoff"),
                    ("r_headroom_gain", "r(headroom, gain)"),
                    ("mean_auc_gain_all_pairs", "Mean dAUC (no headroom filter)"),
                ]))

    gen = frames.get("gen_df")
    if gen is not None and not gen.empty:
        with _guard(errors, "fig3 generalization"):
            _emit("fig3_generalization_kappa.csv", prism_kappa(gen))

    rarity_dropped: list[str] = []
    beh_metrics = frames.get("behavior_metrics_df")
    if beh_metrics is not None and not beh_metrics.empty:
        with _guard(errors, "fig3 rarity vs performance"):
            rar = prism_rarity_vs_performance(frames.get("rarity_df"), beh_metrics)
            rarity_dropped = list(rar.attrs.get("dropped", [])) if not rar.empty else []
            _emit("fig3_rarity_vs_performance.csv", rar)
        with _guard(errors, "fig3 kappa by assay"):
            _emit("fig3_kappa_by_assay.csv", prism_metric_by_assay(beh_metrics))

    vv = frames.get("video_df")
    if vv is not None and not vv.empty:
        with _guard(errors, "fig3 video value"):
            _emit("fig3_video_value.csv", prism_video_gain(vv))

    shares = frames.get("bscape_shares_df")
    if shares is not None and not shares.empty:
        with _guard(errors, "fig3 modality shares"):
            _emit("fig3_modality_shares.csv", prism_behaviorscape_shares(shares))
    bs_imp = frames.get("bscape_importance_df")
    if bs_imp is not None and not bs_imp.empty:
        with _guard(errors, "fig3 feature importance"):
            _emit("fig3_feature_importance_heatmap.csv",
                  prism_behaviorscape_importance(bs_imp))

    al = frames.get("al_df")
    if al is not None and not al.empty:
        with _guard(errors, "fig3 active learning"):
            curves = prism_al_curves(al)
            for src, dst in (("prism_al_curve_f1.csv", "fig3_al_vs_random_f1.csv"),
                             ("prism_al_curve_pos_discovered.csv",
                              "fig3_al_vs_random_positives.csv")):
                _emit(dst, curves.get(src))

    _emit("fig3_rare_discovery_pooled.csv", frames.get("rare_pooled_df"))
    _emit("fig3_rare_effort_minutes.csv", frames.get("rare_effort_minutes_df"))

    # The effort target is chosen from the run (the largest N every arm reached in
    # every project), so it is not knowable from the filename. State it, or the
    # panel is a bar chart of minutes-to-an-unnamed-something.
    notes = dict(_PANEL_NOTES)
    if rarity_dropped:
        table_type, what = notes["fig3_rarity_vs_performance.csv"]
        notes["fig3_rarity_vs_performance.csv"] = (
            table_type,
            what + f" DROPPED for want of a measured prevalence, so absent from "
            f"the scatter ({len(rarity_dropped)}): " + ", ".join(rarity_dropped) + ".")
    target = frames.get("rare_effort_target")
    if target:
        table_type, what = notes["fig3_rare_effort_minutes.csv"]
        notes["fig3_rare_effort_minutes.csv"] = (
            table_type,
            f"Minutes of human review each strategy costs to find {int(target)} "
            "confirmed positives. One row = one behavior, so n = behaviors and the "
            "arms are paired within a row. Quote the assumed seconds-per-clip "
            "review rate in the legend. Companion tables at the other targets are "
            "prism_effort_to_find<N>_minutes_pooled.csv in the parent folder.")

    if written:
        lines = [
            "ABEL validation -- Prism tables for the manuscript figure panels.",
            "",
            "Each file is ONE pre-pivoted block: open a Prism data table of the",
            "stated type, click the first cell, paste. Column 1 is the row/X titles.",
            "The exhaustive per-project archive is in the parent prism/ folder.",
            "",
        ]
        for path in written:
            table_type, what = notes.get(path.name, ("", ""))
            lines.append(path.name)
            if table_type:
                lines.append(f"    Table: {table_type}.")
            for chunk in what.split(". "):
                if chunk.strip():
                    lines.append(f"    {chunk.strip().rstrip('.')}.")
            lines.append("")
        if errors:
            lines.append("Panels skipped (source table failed to build):")
            lines += [f"  - {e}" for e in errors]
        write_text(fig_dir / "INDEX.txt", "\n".join(lines))
    return written


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
    errors: list[str] = []   # tables that failed to build; surfaced in the README

    if gen_df is not None and not gen_df.empty:
      with _guard(errors, "kappa"):
        written.append(_write(prism_kappa(gen_df), out_dir / "prism_kappa.csv"))
        sections.append("prism_kappa.csv\n    Table: Column (or Grouped, 1 replicate).\n"
                        "    One row per behavior; Cohen's kappa + F1.\n")

    if video_df is not None and not video_df.empty:
      with _guard(errors, "video_value"):
        t = prism_video_value(video_df)
        n_rep = sum(1 for c in t.columns if str(c).startswith("Pose only:"))
        written.append(_write(t, out_dir / "prism_video_value.csv"))
        sections.append(
            f"prism_video_value.csv\n    Table: Grouped, 2 groups x "
            f"{n_rep or 1} replicate(s).\n"
            "    Paired video-off vs video-on F1. Analyze -> t tests -> Paired.\n")

    if ablation_df is not None and not ablation_df.empty:
      with _guard(errors, "ablation"):
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
      with _guard(errors, "throughput"):
        for stage, table in prism_throughput(bench_df).items():
            written.append(_write(table, out_dir / f"prism_throughput_{stage}.csv"))
        sections.append(
            "prism_throughput_<stage>.csv\n    Table: Column. Split by stage because\n"
            "    extract/infer are x-real-time and train is seconds.\n")

    if al_df is not None and not al_df.empty:
      with _guard(errors, "al_curves"):
        for fname, table in prism_al_curves(al_df).items():
            written.append(_write(table, out_dir / fname))
        sections.append(
            "prism_al_curve_<metric>.csv\n    Table: XY. Shared X = clips reviewed;\n"
            "    one Y column per project·behavior·strategy (AL vs Random).\n")

    if calibration_df is not None and not calibration_df.empty:
      with _guard(errors, "calibration"):
        written.append(_write(prism_calibration(calibration_df),
                              out_dir / "prism_calibration_reliability.csv"))
        sections.append(
            "prism_calibration_reliability.csv\n    Table: XY. Paired confidence/\n"
            "    accuracy columns per series — the reliability diagram.\n")

    if time_budget_df is not None and not time_budget_df.empty:
      with _guard(errors, "time_budget"):
        written.append(_write(prism_time_budget(time_budget_df),
                              out_dir / "prism_time_budget_prevalence.csv"))
        sections.append(
            "prism_time_budget_prevalence.csv\n    Table: XY. Paired true/pred\n"
            "    prevalence per behavior; points are sessions.\n")

    if bscape_shares_df is not None and not bscape_shares_df.empty:
      with _guard(errors, "behaviorscape_shares"):
        written.append(_write(prism_behaviorscape_shares(bscape_shares_df),
                              out_dir / "prism_behaviorscape_modality_shares.csv"))
        sections.append(
            "prism_behaviorscape_modality_shares.csv\n    Table: Grouped/stacked bar.\n"
            "    Rows = behaviors, columns = modality %.\n")

    if bscape_importance_df is not None and not bscape_importance_df.empty:
      with _guard(errors, "behaviorscape_importance"):
        written.append(_write(prism_behaviorscape_importance(bscape_importance_df),
                              out_dir / "prism_behaviorscape_importance.csv"))
        sections.append(
            "prism_behaviorscape_importance.csv\n    Table: Heatmap. Rows = feature,\n"
            "    columns = behavior, cell = importance (all-zero rows dropped).\n")

    if discrimination_df is not None and not discrimination_df.empty:
      with _guard(errors, "discrimination"):
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
      with _guard(errors, "discrimination_seeds"):
        seeds = prism_discrimination_seeds(discrimination_seeds_df)
        if not seeds.empty:
            written.append(_write(seeds, out_dir / "prism_discrimination_seeds.csv"))
            sections.append(
                "prism_discrimination_seeds.csv\n    Table: Grouped, replicate values.\n"
                "    Rows = pair, one <family>:1..N block per feature family: the raw\n"
                "    per-seed held-out ROC-AUC. Use this to re-run the paired test in\n"
                "    Prism rather than pasting our PValue.\n")

    if accuracy_by_behavior_df is not None and not accuracy_by_behavior_df.empty:
      with _guard(errors, "accuracy_by_behavior"):
        written.append(_write(prism_accuracy_by_behavior(accuracy_by_behavior_df),
                              out_dir / "prism_accuracy_by_behavior.csv"))
        sections.append(
            "prism_accuracy_by_behavior.csv\n    Table: Column/bar. Pooled held-out\n"
            "    F1 + 95% CI per project·behavior.\n")

    if confusion_df is not None and not confusion_df.empty:
      with _guard(errors, "confusion"):
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
        with _guard(errors, "learning_curve"):
            for fname, table in prism_learning_curves(lc_points_df).items():
                written.append(_write(table, out_dir / fname))
            sections.append(
                "prism_learning_curve_<metric>.csv\n    Table: XY, 'Enter and plot error\n"
                "    values' -> Mean, SD, N (3 subcolumns per dataset). Shared X = clips\n"
                "    labeled; one dataset per project (dot) behavior, the across-behavior\n"
                "    average first. SD is reconstructed from the t-based 95% CI and N, so\n"
                "    Prism's error bars match the PNG.\n")
        with _guard(errors, "learning_curve_errors"):
            err_tables = prism_learning_curve_errors(lc_points_df)
            for fname, table in err_tables.items():
                written.append(_write(table, out_dir / fname))
            if err_tables:
                sections.append(
                    "prism_learning_curve_error_fp.csv / _error_fn.csv\n"
                    "    Table: XY, 'Enter and plot error values' -> Mean, SD, N. The\n"
                    "    companion to the F1/PR-AUC curves: held-out FALSE ALARMS (FP)\n"
                    "    and MISSES (FN) as a percent of the held-out set vs. clips\n"
                    "    labeled. Same datasets/average/error bars as the F1 file.\n")

    if lc_knee_df is not None and not lc_knee_df.empty:
      with _guard(errors, "learning_curve_knee"):
        written.append(_write(prism_learning_curve_knee(lc_knee_df),
                              out_dir / "prism_learning_curve_knee.csv"))
        sections.append(
            "prism_learning_curve_knee.csv\n    Table: Column. Saturation point\n"
            "    (knee) in clips and the max F1 reached, per behavior.\n")

    if time_budget_agreement_df is not None and not time_budget_agreement_df.empty:
      with _guard(errors, "time_budget_agreement"):
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
      with _guard(errors, "feature_roles"):
        written.append(_write(prism_feature_roles(feature_roles_df),
                              out_dir / "prism_feature_roles.csv"))
        sections.append(
            "prism_feature_roles.csv\n    Table: Grouped, unequal n. One column per\n"
            "    dominant modality; each behavior's over-pose dF1 sits in its own\n"
            "    group. Analyze -> Nonparametric -> Kruskal-Wallis reproduces the test.\n")

    if feature_roles_bars_df is not None and not feature_roles_bars_df.empty:
      with _guard(errors, "feature_roles_bars"):
        written.append(_write(prism_feature_roles_bars(feature_roles_bars_df),
                              out_dir / "prism_feature_roles_bars.csv"))
        sections.append(
            "prism_feature_roles_bars.csv\n    Table: Column, Mean + SD + N.\n"
            "    One bar per feature modality: mean dF1 over pose-only.\n")

    if publication_metrics_df is not None and not publication_metrics_df.empty:
      with _guard(errors, "publication_metrics"):
        written.append(_write(prism_publication_metrics(publication_metrics_df),
                              out_dir / "prism_publication_metrics.csv"))
        sections.append(
            "prism_publication_metrics.csv\n    Table: Column. F1 / MCC / balanced\n"
            "    accuracy / ROC-AUC / kappa per project -- the reviewer summary.\n")

    if project_accuracy_df is not None and not project_accuracy_df.empty:
      with _guard(errors, "project_accuracy"):
        written.append(_write(prism_project_accuracy(project_accuracy_df),
                              out_dir / "prism_accuracy_by_project.csv"))
        sections.append(
            "prism_accuracy_by_project.csv\n    Table: Column, Mean + SD + N.\n"
            "    Held-out F1 per project.\n")

    if training_speed_df is not None and not training_speed_df.empty:
      with _guard(errors, "training_speed"):
        speed = prism_training_speed(training_speed_df)
        if not speed.empty:      # all-zero timings are "not measured", not "instant"
            written.append(_write(speed, out_dir / "prism_training_speed.csv"))
            sections.append(
                "prism_training_speed.csv\n    Table: Column. Median/mean training\n"
                "    seconds per project.\n")

    if errors:
        sections.append(
            "SKIPPED (these tables failed to build and were left out; the rest of\n"
            "the export continued):\n    " + "\n    ".join(errors) + "\n")
    if written:
        written.append(write_text(out_dir / "README_PRISM.txt",
                                  _README.format(sections="\n".join(sections))))
    return written
