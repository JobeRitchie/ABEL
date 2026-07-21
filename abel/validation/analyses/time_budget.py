"""Per-session agreement between the model and the reviewer.

**Read this before citing any number from this module.**

For each held-out ``(session, animal)`` unit it compares, model vs. reviewer, the
**prevalence of a behavior among that unit's reviewed segments** — and, only where
the labeling is dense enough to support it, the **bout count**.  Agreement is
summarised with the measures behavior-scoring validations report: Pearson r, Lin's
concordance CCC, R², and Bland-Altman bias with 95% limits of agreement (see
:mod:`abel.validation.metrics`).  It is a pure post-hoc computation on the
retained generalization predictions — no extra training.

What this is NOT
----------------
This is **not a time budget**, and it must never be labeled "% time freezing" or
"time in behavior".  ABEL's ``training_set.parquet`` holds only the segments a
reviewer actually looked at, and those are sparse: on DG_FearConditioning the
labeled segments cover a **median 1.5%** of each session's frame extent (min
0.55%).  They are also active-learning selected, so they are a *biased* sample
enriched for positives — not a uniform one.  A fraction computed over them
therefore answers "of the clips we reviewed in this session, what share were
freezing?", which is a per-session accuracy/prevalence measure, not the
biological quantity a stopwatch would produce.

Every unit carries a ``coverage_frac`` (labeled frames / session extent) so the
reader can see exactly how thin that sample is.

Getting the real time budget would require running dense inference over *all*
frames of the held-out sessions with the held-out model (the product's temporal
refinement pass), which this module deliberately does not do — the deploy-model
traces already on disk are trained on all data and would leak.

Why bout counts are gated
-------------------------
A bout is a run of contiguous *frames*.  Adjacent rows in the labeled set are
frequently not adjacent in time (on DG_FearConditioning only **47.5%** of
adjacent segment pairs are contiguous; gaps reach 19,331 frames), so counting
runs of adjacent positive *rows* would merge clips ten minutes apart into a
single "bout".  Bout counts are therefore only reported for units whose reviewed
segments are essentially contiguous (:data:`MIN_CONTIGUITY`); otherwise they are
NaN and drop out of the statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from abel.validation import metrics as vmetrics
from abel.validation.analyses.generalization import HoldoutPredictions

#: A unit's reviewed segments must be at least this contiguous (fraction of
#: adjacent pairs separated by <=1 frame) before its bout count means anything.
MIN_CONTIGUITY = 0.9

#: Correlating fewer than this many held-out units is not meaningful.
MIN_UNITS = 3


@dataclass
class TimeBudgetResult:
    project_id: str
    behavior_id: str
    behavior_name: str
    n_units: int = 0                       # held-out (session, animal) units compared

    # Per-unit paired series. ``prevalence`` = share of the unit's REVIEWED
    # segments in the behavior (NOT share of session time — see module docstring).
    true_prevalence: list[float] = field(default_factory=list)
    pred_prevalence: list[float] = field(default_factory=list)
    true_bouts: list[float] = field(default_factory=list)   # NaN where not contiguous
    pred_bouts: list[float] = field(default_factory=list)
    unit_labels: list[str] = field(default_factory=list)
    coverage_frac: list[float] = field(default_factory=list)  # labeled frames / session extent

    # Prevalence agreement.
    prev_pearson_r: float = float("nan")
    prev_ccc: float = float("nan")
    prev_r2: float = float("nan")
    prev_bias: float = float("nan")        # mean(pred − true)
    prev_bias_ci: tuple[float, float] = (float("nan"), float("nan"))
    prev_loa_lower: float = float("nan")
    prev_loa_upper: float = float("nan")
    prev_mae: float = float("nan")

    # Bout-count agreement (only over units with contiguous labeling).
    n_units_bouts: int = 0
    bout_pearson_r: float = float("nan")
    bout_ccc: float = float("nan")

    # Median labeled-frame coverage across units — the "how much of the session did
    # we actually look at" caveat that must travel with every number above.
    median_coverage: float = float("nan")

    error: str = ""

    @property
    def loa_width(self) -> float:
        """Width of the 95% limits of agreement — the per-unit usability number."""
        if not (np.isfinite(self.prev_loa_lower) and np.isfinite(self.prev_loa_upper)):
            return float("nan")
        return float(self.prev_loa_upper - self.prev_loa_lower)


def _bout_count(pos: np.ndarray, contiguous: bool) -> float:
    """Runs of consecutive positive segments — NaN unless the rows are contiguous.

    Only meaningful when adjacent rows are adjacent in time; see the module
    docstring for why that is usually false.
    """
    if not contiguous:
        return float("nan")
    p = np.asarray(pos, dtype=int)
    if p.size == 0:
        return 0.0
    return float(int(p[0] == 1) + int(np.sum((p[1:] == 1) & (p[:-1] == 0))))


def _unit_series(preds: HoldoutPredictions) -> pd.DataFrame:
    """One row per (session, animal): prevalence, coverage, and gated bout counts."""
    df = pd.DataFrame(
        {
            "session": preds.session_ids,
            "animal": preds.animal_ids,
            "start": np.asarray(preds.start_frames, dtype=np.int64),
            "end": np.asarray(preds.end_frames, dtype=np.int64),
            "y_true": np.asarray(preds.y_true, dtype=int),
            "y_pred": np.asarray(preds.y_pred, dtype=int),
        }
    )
    span = (df["end"] - df["start"] + 1).to_numpy(dtype=float)
    valid = np.isfinite(span) & (df["start"].to_numpy() >= 0) & (span > 0)
    df["weight"] = np.where(valid, span, 1.0)  # fallback: one frame per window

    rows = []
    for (sess, animal), grp in df.groupby(["session", "animal"], sort=False):
        grp = grp.sort_values("start", kind="stable")
        w = grp["weight"].to_numpy(dtype=float)
        total = float(w.sum())
        if total <= 0:
            continue
        yt = grp["y_true"].to_numpy(dtype=int)
        yp = grp["y_pred"].to_numpy(dtype=int)

        sf = grp["start"].to_numpy(dtype=np.int64)
        ef = grp["end"].to_numpy(dtype=np.int64)
        has_frames = bool((sf >= 0).all()) and len(sf) > 0
        if has_frames:
            extent = float(ef.max() - sf.min() + 1)
            coverage = float(total / extent) if extent > 0 else float("nan")
            if len(sf) > 1:
                gaps = sf[1:] - ef[:-1]
                contiguity = float(np.mean(gaps <= 1))
            else:
                contiguity = 1.0
        else:
            coverage, contiguity = float("nan"), 0.0
        contiguous = contiguity >= MIN_CONTIGUITY

        rows.append(
            {
                "unit": f"{sess}" + (f"/{animal}" if str(animal) else ""),
                "n_segments": int(len(grp)),
                "coverage_frac": coverage,
                "contiguity": contiguity,
                "true_prevalence": float((w * (yt == 1)).sum() / total),
                "pred_prevalence": float((w * (yp == 1)).sum() / total),
                "true_bouts": _bout_count(yt, contiguous),
                "pred_bouts": _bout_count(yp, contiguous),
            }
        )
    return pd.DataFrame(rows)


def run_time_budget(preds: HoldoutPredictions | None) -> TimeBudgetResult | None:
    """Per-session prevalence (and, where valid, bout) agreement for one behavior."""
    if preds is None:
        return None
    res = TimeBudgetResult(
        project_id=preds.project_id,
        behavior_id=preds.behavior_id,
        behavior_name=preds.behavior_name,
    )
    units = _unit_series(preds)
    res.n_units = int(len(units))
    if len(units) == 0:
        res.error = "no held-out units"
        return res

    res.unit_labels = units["unit"].tolist()
    res.true_prevalence = units["true_prevalence"].tolist()
    res.pred_prevalence = units["pred_prevalence"].tolist()
    res.true_bouts = units["true_bouts"].tolist()
    res.pred_bouts = units["pred_bouts"].tolist()
    res.coverage_frac = units["coverage_frac"].tolist()
    cov = pd.to_numeric(units["coverage_frac"], errors="coerce").dropna()
    res.median_coverage = float(cov.median()) if len(cov) else float("nan")

    if len(units) < MIN_UNITS:
        res.error = f"need >={MIN_UNITS} held-out sessions to correlate, found {len(units)}"
        return res

    tp = units["true_prevalence"].to_numpy(dtype=float)
    pp = units["pred_prevalence"].to_numpy(dtype=float)

    res.prev_pearson_r = vmetrics.pearson_r(tp, pp)
    res.prev_ccc = vmetrics.concordance_ccc(tp, pp)
    res.prev_r2 = vmetrics.r_squared(tp, pp)
    res.prev_mae = vmetrics.mean_absolute_error(tp, pp)
    ba = vmetrics.bland_altman(tp, pp)
    res.prev_bias = ba.bias
    res.prev_bias_ci = ba.bias_ci95()
    res.prev_loa_lower, res.prev_loa_upper = ba.loa_lower, ba.loa_upper

    # Bouts only over units whose labeling is dense enough to define them.
    bt = units["true_bouts"].to_numpy(dtype=float)
    bp = units["pred_bouts"].to_numpy(dtype=float)
    ok = np.isfinite(bt) & np.isfinite(bp)
    res.n_units_bouts = int(ok.sum())
    if res.n_units_bouts >= MIN_UNITS:
        res.bout_pearson_r = vmetrics.pearson_r(bt[ok], bp[ok])
        res.bout_ccc = vmetrics.concordance_ccc(bt[ok], bp[ok])
    return res


def time_budget_rows(results: list[TimeBudgetResult]) -> pd.DataFrame:
    """Tidy per-(project, behavior) agreement table for CSV export / the report."""
    rows = []
    for r in results:
        if r is None:
            continue
        lo, hi = r.prev_bias_ci
        rows.append(
            {
                "project": r.project_id,
                "behavior": r.behavior_name,
                "n_sessions": r.n_units,
                # The caveat that must travel with every number in this row.
                "median_labeled_coverage": r.median_coverage,
                "prevalence_pearson_r": r.prev_pearson_r,
                "prevalence_ccc": r.prev_ccc,
                "prevalence_r2": r.prev_r2,
                "prevalence_bias": r.prev_bias,
                "prevalence_bias_ci_lo": lo,
                "prevalence_bias_ci_hi": hi,
                "prevalence_loa_lower": r.prev_loa_lower,
                "prevalence_loa_upper": r.prev_loa_upper,
                "prevalence_loa_width": r.loa_width,
                "prevalence_mae": r.prev_mae,
                "n_sessions_with_bouts": r.n_units_bouts,
                "bout_pearson_r": r.bout_pearson_r,
                "bout_ccc": r.bout_ccc,
                "error": r.error,
            }
        )
    return pd.DataFrame(rows)


def time_budget_points(results: list[TimeBudgetResult]) -> pd.DataFrame:
    """Per-session paired points (model vs. reviewer) for scatter export."""
    rows = []
    for r in results:
        if r is None:
            continue
        for i, lab in enumerate(r.unit_labels):
            def _at(seq, idx=i):
                return seq[idx] if idx < len(seq) else float("nan")

            rows.append(
                {
                    "project": r.project_id,
                    "behavior": r.behavior_name,
                    "session": lab,
                    "labeled_coverage": _at(r.coverage_frac),
                    "true_prevalence": _at(r.true_prevalence),
                    "pred_prevalence": _at(r.pred_prevalence),
                    "true_bouts": _at(r.true_bouts),
                    "pred_bouts": _at(r.pred_bouts),
                }
            )
    return pd.DataFrame(rows)
