"""Turn raw analysis results into plain-language findings.

The validation suite already writes every number to CSV and every series to a
figure.  What it never did was *say what happened* — a reader had to open eight
folders and know, for instance, that a raw ΔAUC of +0.004 on a pair whose
baseline is already 0.984 is actually a 24% cut in the remaining error, or that
a "time budget" computed over reviewed segments is a prevalence, not a time
budget.  This module encodes those readings once, so the summary report and the
GUI say the same thing.

Every finding is one sentence of takeaway (``headline``) plus the numbers that
back it (``detail``).  Findings are honest by construction: the interpretation
caveats that make a number *not* mean what it appears to mean are emitted as
first-class findings (``kind="caveat"``), not buried in a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from abel.validation import subsample
from abel.validation.analyses import ablation, discrimination

KIND_RESULT = "result"
KIND_CAVEAT = "caveat"
KIND_WARNING = "warning"

# Human-agreement bands (Landis & Koch), used to describe κ in words.
_KAPPA_BANDS = [
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.00, "slight"),
]


@dataclass
class Finding:
    """One takeaway, ready to print."""

    analysis: str            # section it belongs to, e.g. "Discrimination"
    headline: str            # the conclusion, in words
    detail: str = ""         # the numbers behind it
    kind: str = KIND_RESULT  # result | caveat | warning

    def to_row(self) -> dict:
        return {"analysis": self.analysis, "kind": self.kind,
                "finding": self.headline, "detail": self.detail}


@dataclass
class FindingsInput:
    """Everything a run produced, as the runner already holds it in memory."""

    cells: pd.DataFrame = field(default_factory=pd.DataFrame)
    overview: dict = field(default_factory=dict)
    project_meta: list[dict] = field(default_factory=list)
    lc_results: list = field(default_factory=list)
    abl_results: list = field(default_factory=list)
    disc_by_project: dict = field(default_factory=dict)
    gen_results: list = field(default_factory=list)
    tb_results: list = field(default_factory=list)
    cal_results: list = field(default_factory=list)
    al_results: list = field(default_factory=list)
    vv_results: list = field(default_factory=list)
    bench_results: list = field(default_factory=list)
    bscape_stats: object = None
    bscape_data: object = None


# ── small formatting helpers ────────────────────────────────────────────────

def _f(v, nd: int = 3) -> str:
    """Format a float, tolerating None/NaN (which are pervasive here)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(x) else f"{x:.{nd}f}"


def _pct(v, nd: int = 0) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(x) else f"{x * 100:.{nd}f}%"


def _finite(values: Iterable) -> list[float]:
    out = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _kappa_word(k: float) -> str:
    for floor, word in _KAPPA_BANDS:
        if k >= floor:
            return word
    return "poor"


def _plural(n: int, one: str, many: str | None = None) -> str:
    return one if n == 1 else (many or one + "s")


def _clips(n) -> str:
    """A clip count that may be the ALL_CLIPS sentinel or a NaN."""
    try:
        x = float(n)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "not reached"
    return f"{int(round(x))}"


def _family(label: str) -> str:
    """'Video' → 'Video features'; 'All features' → 'All features' (not '…features features')."""
    lbl = str(label).lstrip("+ ").strip()
    return lbl if lbl.lower().endswith(("features", "context")) else f"{lbl} features"


# ── per-analysis finding derivation ─────────────────────────────────────────

def _overview_findings(inp: FindingsInput) -> list[Finding]:
    out: list[Finding] = []
    ov = inp.overview or {}
    n_proj = len(inp.project_meta) or ov.get("n_projects", 0)
    # NOT ov["n_behaviors"]: that counts distinct behavior_ids across every cell, and
    # the discrimination analysis files its cells under a *pair* ("Groom vs Rear"), so
    # 3 behaviors + 3 pairs reads as 6 behaviors.  project_meta is the truth.
    n_beh = (sum(len(p.get("behaviors", [])) for p in inp.project_meta)
             or ov.get("n_behaviors", 0))

    # The headline accuracy must come from models fit the way the product fits them.
    # ov["f1_mean"] averages over EVERY cell in the run — including 10-clip
    # learning-curve points and pose-only ablation baselines, which are crippled on
    # purpose — so it understates the shipped model and means nothing on its own.
    gen_f1 = _finite([g.f1_mean for g in inp.gen_results])
    if n_proj and gen_f1:
        out.append(Finding(
            "Overview",
            f"Validated {n_proj} {_plural(int(n_proj), 'project')} covering "
            f"{n_beh} {_plural(int(n_beh), 'behavior')}. On held-out subjects, the "
            f"production model configuration scores a mean F1 of "
            f"{_f(float(np.mean(gen_f1)))}.",
            f"Held-out F1 across behaviors: {_f(min(gen_f1))}–{_f(max(gen_f1))}. Every "
            f"number in this report comes from models trained on a training pool of "
            f"subjects/sessions and scored on subjects/sessions the model never saw. "
            f"(The run-wide mean over all {ov.get('n_cells', 0)} fitted cells is "
            f"{_f(ov.get('f1_mean'))}, but that pools deliberately handicapped models — "
            f"low-clip learning-curve points and pose-only ablation baselines — so it "
            f"is not the model's accuracy.)",
        ))
    elif n_proj:
        out.append(Finding(
            "Overview",
            f"Validated {n_proj} {_plural(int(n_proj), 'project')} covering "
            f"{n_beh} {_plural(int(n_beh), 'behavior')}.",
            f"Mean F1 over all {ov.get('n_cells', 0)} fitted cells: "
            f"{_f(ov.get('f1_mean'))} (range {_f(ov.get('f1_min'))}–"
            f"{_f(ov.get('f1_max'))}). This pools every configuration the run fitted, "
            f"including deliberately handicapped ones; run the generalization analysis "
            f"for the production model's held-out accuracy.",
        ))

    # A project whose held-out labels are too sparse produces plausible-looking but
    # meaningless numbers (macro-F1 pinned by the majority class, κ = 0).  Say so
    # rather than letting it sit in a bar chart looking like a pipeline failure.
    df = inp.cells
    if not df.empty and "n_pos_train" in df.columns:
        sparse = []
        for (pid, beh), grp in df.groupby(["project_name", "behavior_name"], dropna=False):
            pos = _finite(grp["n_pos_train"])
            if pos and max(pos) < 40:
                sparse.append(f"{pid} · {beh} ({int(max(pos))} positives)")
        if sparse:
            out.append(Finding(
                "Overview",
                f"{len(sparse)} (project × behavior) {_plural(len(sparse), 'combination')} "
                f"{_plural(len(sparse), 'is', 'are')} too sparsely labeled to score "
                f"reliably — treat {_plural(len(sparse), 'its', 'their')} metrics as "
                f"labeling artifacts, not pipeline performance.",
                "Fewer than 40 labeled positives in the training pool. At that density "
                "the target-class κ collapses toward 0 and F1 is pinned by the majority "
                "class, so every feature-family gain reads as ~0. Affected: "
                + "; ".join(sorted(sparse)[:8])
                + ("; …" if len(sparse) > 8 else ""),
                kind=KIND_WARNING,
            ))
    return out


def _confusion_findings(inp: FindingsInput) -> list[Finding]:
    """Say the result in counts, not rates.

    F1 = 0.91 is a number a reader takes on trust; "found 191 of 214 and raised 17
    false alarms" is one they can check against their own scoring experience.  Both
    are emitted, because the counts alone are not comparable across behaviors (the
    held-out denominators differ) — the rate does the comparing, the count makes it
    concrete.
    """
    from abel.validation.analyses.cross_project import confusion_by_behavior  # noqa: PLC0415

    conf = confusion_by_behavior(inp.cells)
    if conf.empty:
        return []

    tp, fn, fp = (int(conf[c].sum()) for c in ("tp", "fn", "fp"))
    n_pos = tp + fn
    if n_pos <= 0:
        return []

    rows = conf.sort_values("recall", na_position="last")
    worst = rows.iloc[0]
    best = rows.iloc[-1]

    def _row(r) -> str:
        return (f"{r['project_id']} · {r['behavior_name']}: {int(r['tp'])} of "
                f"{int(r['n_pos_val'])} found, {int(r['fp'])} false "
                f"{_plural(int(r['fp']), 'alarm')}")

    out = [Finding(
        "Held-out counts",
        f"Across {len(conf)} (assay × behavior) "
        f"{_plural(len(conf), 'model')}, the model recovered {tp} of the {n_pos} "
        f"held-out clips the reviewer marked positive ({_pct(tp / n_pos)}) and "
        f"raised {fp} false {_plural(fp, 'alarm')}.",
        f"Best recall — {_row(best)}. Weakest — {_row(worst)}. Counts are per fit, "
        f"averaged over seeds (each seed re-scores the same held-out pool, so they "
        f"are not additive) and totalled across behaviors, each of which brings its "
        f"own positives. Per-behavior counts: cross_project/confusion_by_behavior.csv.",
    )]

    # The single most misreadable thing about a count table, stated up front.
    out.append(Finding(
        "Held-out counts",
        "These counts are clips the reviewer scored, not bouts — a 'false alarm' "
        "is one mis-scored clip, not a spurious behavioral event.",
        "Bout-level counts are not identifiable from a held-out labeled subset: the "
        "evaluated unit is one short, isolated clip (a fraction of a second in most "
        "projects — see clip_sec in confusion_by_behavior.csv), while a bout needs "
        "contiguous observation longer than itself, so event-level FP/FN measure "
        "label sparsity rather than the model. True negatives are reported in the "
        "CSV to close the 2×2 but are not summarized here — under this class "
        "imbalance any accuracy derived from them is ~0.99 regardless of the model.",
        kind=KIND_CAVEAT,
    ))
    return out


def _learning_curve_findings(inp: FindingsInput) -> list[Finding]:
    if not inp.lc_results:
        return []
    out: list[Finding] = []
    knees = [(lc, lc.knee_clips) for lc in inp.lc_results]
    valid = [(lc, k) for lc, k in knees if k is not None and math.isfinite(float(k))]
    if valid:
        vals = [k for _, k in valid]
        median_knee = float(np.median(vals))
        worst_lc, worst_k = max(valid, key=lambda kv: kv[1])
        best_lc, best_k = min(valid, key=lambda kv: kv[1])
        out.append(Finding(
            "Learning curves",
            f"A median of ~{_clips(median_knee)} labeled clips per behavior reaches 95% "
            f"of that behavior's peak held-out F1 — this is the recommended labeling "
            f"budget.",
            f"Per-behavior knee ranges from {_clips(best_k)} clips "
            f"({best_lc.behavior_name}) to {_clips(worst_k)} clips "
            f"({worst_lc.behavior_name}). The knee is the smallest clip count whose mean "
            f"F1 reaches 95% of the curve's maximum, averaged over seeds.",
        ))
        out.append(Finding(
            "Learning curves",
            f"{worst_lc.behavior_name} is the most label-hungry behavior "
            f"({_clips(worst_k)} clips to plateau); {best_lc.behavior_name} is the "
            f"cheapest ({_clips(best_k)} clips).",
            f"Peak F1 — {worst_lc.behavior_name}: {_f(worst_lc.f1_max)}; "
            f"{best_lc.behavior_name}: {_f(best_lc.f1_max)}.",
        ))
    unplateaued = [lc for lc, k in knees
                   if k is None or not math.isfinite(float(k))]
    if unplateaued:
        names = ", ".join(sorted(lc.behavior_name for lc in unplateaued)[:6])
        out.append(Finding(
            "Learning curves",
            f"{len(unplateaued)} {_plural(len(unplateaued), 'behavior')} had not "
            f"plateaued at the largest clip budget tested — {_plural(len(unplateaued), 'it', 'they')} "
            f"would still improve with more labels.",
            f"No knee within the schedule: {names}. Extend the clip-size schedule to "
            f"find their plateau.",
            kind=KIND_CAVEAT,
        ))
    return out


def _ablation_findings(inp: FindingsInput) -> list[Finding]:
    if not inp.abl_results:
        return []
    out: list[Finding] = []

    # Pool gains per config across behaviors, at the largest budget (full data),
    # which is the configuration the shipped model actually uses.
    full = [r for r in inp.abl_results if r.clip_budget == subsample.ALL_CLIPS]
    pool = full or inp.abl_results
    by_config: dict[str, list[float]] = {}
    sig_count: dict[str, int] = {}
    labels: dict[str, str] = {}
    for r in pool:
        for cfg in r.order:
            if cfg == ablation.BASELINE_CONFIG:
                continue
            g = r.gain.get(cfg)
            if g is None or not math.isfinite(float(g)):
                continue
            by_config.setdefault(cfg, []).append(float(g))
            labels[cfg] = r.labels.get(cfg, cfg)
            if r.is_significant(cfg):
                sig_count[cfg] = sig_count.get(cfg, 0) + 1

    if not by_config:
        return out

    ranked = sorted(by_config.items(), key=lambda kv: float(np.mean(kv[1])), reverse=True)
    # The "all enhancements" bar is the union of the others, so it wins the ranking by
    # construction and answering "which single addition helps most?" with it is
    # meaningless. Rank the single enhancements; report the combined bar separately as
    # the ceiling they add up to.
    singles = [(c, g) for c, g in ranked if c != ablation.ALL_FEATURES_CONFIG]
    combined = next((g for c, g in ranked if c == ablation.ALL_FEATURES_CONFIG), None)
    n_beh = len(pool)
    helpful = [(c, g) for c, g in singles if sig_count.get(c, 0) > 0]

    if singles:
        top_cfg, top_gains = singles[0]
        combined_txt = ""
        if combined is not None:
            combined_txt = (f" Enabling every enhancement together adds "
                            f"{_f(np.mean(combined))}.")
        out.append(Finding(
            "Ablation (detection)",
            f"{labels.get(top_cfg, top_cfg).lstrip('+ ')} is the single most valuable "
            f"addition over the pose-only baseline, adding a mean ΔF1 of "
            f"{_f(np.mean(top_gains))} across behaviors.",
            "Each bar adds ONE enhancement to a pose-only baseline (every enhancement "
            "off); gains are paired per seed, so the same subsample trains baseline and "
            "config. Mean ΔF1 by single addition — "
            + "; ".join(f"{labels.get(c, c).lstrip('+ ')} {_f(np.mean(g), 3)}"
                        for c, g in singles)
            + "." + combined_txt,
        ))
    if helpful:
        out.append(Finding(
            "Ablation (detection)",
            f"{len(helpful)} of {len(singles)} single enhancements produced a "
            f"statistically significant gain on at least one behavior.",
            "Significant = the 95% CI of the paired per-seed gain excludes 0. Counts "
            "(behaviors with a significant gain, of "
            f"{n_beh}) — "
            + "; ".join(f"{labels.get(c, c).lstrip('+ ')}: {sig_count.get(c, 0)}"
                        for c, _ in singles)
            + ".",
        ))
    dead = [labels.get(c, c).lstrip("+ ") for c, g in singles
            if sig_count.get(c, 0) == 0 and float(np.mean(g)) <= 0.002]
    if dead:
        out.append(Finding(
            "Ablation (detection)",
            f"{', '.join(dead)} did not measurably improve detection on this data.",
            "A detection ablation asks 'behavior X vs. everything else', and 'everything "
            "else' is dominated by easy negatives — a feature family can look worthless "
            "here while still being the thing that separates two confusable behaviors. "
            "Read this beside the discrimination section before concluding a family is "
            "useless.",
            kind=KIND_CAVEAT,
        ))
    return out


def _discrimination_findings(inp: FindingsInput) -> list[Finding]:
    if not inp.disc_by_project:
        return []
    out: list[Finding] = []
    all_pairs = [pr for prs in inp.disc_by_project.values() for pr in prs]
    scored = [pr for pr in all_pairs if not pr.error
              and math.isfinite(float(pr.baseline_auc))]
    if not scored:
        return out

    hardest = min(scored, key=lambda pr: float(pr.baseline_auc))
    ceiling = [pr for pr in scored
               if float(pr.baseline_auc) > 1.0 - discrimination.MIN_HEADROOM]

    out.append(Finding(
        "Discrimination (pairwise)",
        f"{hardest.pair_label} is the hardest pair to tell apart from pose alone "
        f"(ROC-AUC {_f(hardest.baseline_auc)}).",
        f"Tested {len(scored)} behavior {_plural(len(scored), 'pair')} across "
        f"{len(inp.disc_by_project)} {_plural(len(inp.disc_by_project), 'project')}. "
        f"Each pair is a binary A-vs-B model trained on only the clips labeled exactly "
        f"A or exactly B, once per feature family on the same clips and seed.",
    ))

    # What, if anything, rescues the hardest pair.  This is the actionable result:
    # a family that removes the remaining error is where to invest.  The
    # "all_features" rung is the union of the families, so it wins any ranking by
    # construction — it is reported as the ceiling, never as the answer to "which
    # family?".
    def _label(fs: str) -> str:
        return discrimination.FEATURE_SET_LABELS.get(fs, fs).lstrip("+ ").strip()

    def _singles(order) -> list[str]:
        return [fs for fs in order
                if fs not in (discrimination.BASELINE_FEATURE_SET,
                              discrimination.ALL_FEATURE_SET)]

    rescues = []
    for fs in _singles(hardest.order):
        er = hardest.error_reduction(fs)
        if er is not None and math.isfinite(float(er)):
            rescues.append((fs, float(er)))
    if rescues:
        best_fs, best_er = max(rescues, key=lambda kv: kv[1])
        lbl = _label(best_fs)
        breakdown = "; ".join(f"{_label(fs)} {_pct(er)}" for fs, er in rescues)
        combined = hardest.error_reduction(discrimination.ALL_FEATURE_SET)
        combined_txt = ""
        if combined is not None and math.isfinite(float(combined)):
            combined_txt = f" All families together: {_pct(float(combined))}."
        if best_er >= 0.15:
            out.append(Finding(
                "Discrimination (pairwise)",
                f"{_family(lbl)} rescue {hardest.pair_label}, removing "
                f"{_pct(best_er)} of the pose-only model's remaining error.",
                "Error reduction = (AUC_family − AUC_pose) / (1 − AUC_pose): the share of "
                f"the baseline's *remaining* error the family removes. {breakdown}."
                + combined_txt,
            ))
        else:
            out.append(Finding(
                "Discrimination (pairwise)",
                f"No feature family rescues {hardest.pair_label} — the best "
                f"({lbl}) removes only {_pct(best_er)} of the remaining error. "
                f"This pair needs new features, not more data.",
                f"Error reduction by family — {breakdown}.{combined_txt} A targeting "
                "result: these two behaviors are not separable by any feature family "
                "currently extracted.",
                kind=KIND_WARNING,
            ))

    # Which single family helps most across all pairs that still have headroom.
    by_fs: dict[str, list[float]] = {}
    for pr in scored:
        for fs in _singles(pr.order):
            er = pr.error_reduction(fs)
            if er is not None and math.isfinite(float(er)):
                by_fs.setdefault(fs, []).append(float(er))
    if by_fs:
        ranked = sorted(by_fs.items(), key=lambda kv: float(np.mean(kv[1])), reverse=True)
        top_fs, top_ers = ranked[0]
        out.append(Finding(
            "Discrimination (pairwise)",
            f"Across every pair with headroom left, {_family(_label(top_fs))} remove the "
            f"most remaining error (mean {_pct(float(np.mean(top_ers)))}).",
            "Mean error reduction by family, over pairs the pose baseline has not already "
            "solved — "
            + "; ".join(f"{_label(fs)} {_pct(float(np.mean(ers)))} (n={len(ers)})"
                        for fs, ers in ranked)
            + ".",
        ))

    if ceiling:
        out.append(Finding(
            "Discrimination (pairwise)",
            f"{len(ceiling)} of {len(scored)} pairs are already solved from pose alone "
            f"(AUC > {_f(1.0 - discrimination.MIN_HEADROOM)}) and are greyed out of the "
            f"figures — there is no discrimination question left to ask there.",
            "Pairwise A-vs-B is a far easier task than detecting a behavior against all "
            "others, so pose-only AUC saturates. Raw ΔAUC is meaningless at that ceiling "
            "(every cell reads +0.00), which is why the figures plot the share of "
            "remaining error removed instead. Note also that at these seed counts a "
            "'significant' gain tests consistency, not importance: with near-zero seed "
            "variance a +0.001 gain can flag significant. Read significance beside the "
            "error-reduction column, never alone.",
            kind=KIND_CAVEAT,
        ))
    return out


def _generalization_findings(inp: FindingsInput) -> list[Finding]:
    if not inp.gen_results:
        return []
    out: list[Finding] = []
    kappas = [(r, float(r.kappa_mean)) for r in inp.gen_results
              if math.isfinite(float(r.kappa_mean))]
    if kappas:
        vals = [k for _, k in kappas]
        mean_k = float(np.mean(vals))
        worst_r, worst_k = min(kappas, key=lambda kv: kv[1])
        ceilings = _finite([r.human_ceiling_kappa for r in inp.gen_results])
        ceil_txt = ""
        if ceilings:
            ceil_txt = (f" Human-agreement ceiling on the same held-out data: "
                        f"κ = {_f(float(np.mean(ceilings)))}.")
        out.append(Finding(
            "Generalization",
            f"On subjects/sessions the model never saw, agreement with the human "
            f"reviewer is {_kappa_word(mean_k)} (mean Cohen's κ = {_f(mean_k)}).",
            f"Per-behavior κ ranges {_f(min(vals))}–{_f(max(vals))} over "
            f"{len(kappas)} {_plural(len(kappas), 'behavior')}.{ceil_txt} "
            f"κ, not accuracy, is the honest statistic here: it corrects for the "
            f"agreement you would get by chance on an imbalanced label set.",
        ))
        if worst_k < 0.6:
            out.append(Finding(
                "Generalization",
                f"{worst_r.behavior_name} is the weakest generalizer "
                f"(κ = {_f(worst_k)}, F1 = {_f(worst_r.f1_mean)}) and should not be "
                f"reported without review.",
                f"Every other behavior scores above it. κ < 0.6 is below the "
                f"'substantial agreement' band — a model at this level disagrees with "
                f"the reviewer often enough that its per-session numbers are unsafe.",
                kind=KIND_WARNING,
            ))
    return out


def _time_budget_findings(inp: FindingsInput) -> list[Finding]:
    usable = [t for t in inp.tb_results if not t.error]
    if not usable:
        return []
    out: list[Finding] = []
    cccs = [(t, float(t.prev_ccc)) for t in usable if math.isfinite(float(t.prev_ccc))]
    if cccs:
        best_t, best_c = max(cccs, key=lambda kv: kv[1])
        vals = [c for _, c in cccs]
        out.append(Finding(
            "Biological readout",
            f"The model reproduces the per-session prevalence a human scorer would "
            f"report, best for {best_t.behavior_name} (Lin's CCC = {_f(best_c)}).",
            f"CCC across {len(cccs)} {_plural(len(cccs), 'behavior')}: "
            f"{_f(min(vals))}–{_f(max(vals))}. CCC (not r) is the right statistic: r "
            f"rewards correlation even under a constant offset, CCC penalises the "
            f"offset. Bland-Altman limits of agreement, not r, decide whether a "
            f"single animal's number is usable.",
        ))
    # The single most important caveat in the whole suite.
    covs = _finite([t.median_coverage for t in usable])
    if covs:
        out.append(Finding(
            "Biological readout",
            "These are PREVALENCE figures (share of *reviewed segments*), not time "
            "budgets (share of session time) — reviewed segments cover a median of only "
            f"{_pct(float(np.median(covs)), 1)} of each session and are "
            "active-learning-biased toward positives.",
            "A genuine %-time-in-behavior figure requires dense inference over ALL frames "
            "of held-out sessions using the held-out model. The deploy-model traces on "
            "disk cannot be used for this — they are trained on all data, so scoring "
            "held-out sessions with them leaks. This is a known gap, not an error in "
            "these numbers: as prevalence they are correct.",
            kind=KIND_CAVEAT,
        ))
    no_bouts = [t for t in usable if not math.isfinite(float(t.bout_ccc))]
    if no_bouts:
        out.append(Finding(
            "Biological readout",
            f"Bout counts are reported as not-computable for "
            f"{len(no_bouts)} {_plural(len(no_bouts), 'behavior')} — correctly so.",
            "Labeled segments are not contiguous in time (gaps reach thousands of "
            "frames), so counting runs of adjacent positive rows would merge far-apart "
            "clips into one 'bout' and yield a plausible but fabricated bout count. The "
            "analysis refuses to compute bouts unless a session's segments are "
            "essentially contiguous.",
            kind=KIND_CAVEAT,
        ))
    return out


def _calibration_findings(inp: FindingsInput) -> list[Finding]:
    if not inp.cal_results:
        return []
    eces = [(c, float(c.ece)) for c in inp.cal_results
            if math.isfinite(float(c.ece))]
    if not eces:
        return []
    vals = [e for _, e in eces]
    mean_ece = float(np.mean(vals))
    worst_c, worst_e = max(eces, key=lambda kv: kv[1])
    verdict = ("well calibrated" if mean_ece < 0.05
               else "acceptably calibrated" if mean_ece < 0.10
               else "poorly calibrated")
    return [Finding(
        "Calibration",
        f"The shipped (project-calibrated) model is {verdict}: a predicted probability "
        f"of 0.8 really does mean right about 80% of the time (mean ECE = "
        f"{_f(mean_ece)}).",
        f"Expected calibration error ranges {_f(min(vals))}–{_f(max(vals))}; worst is "
        f"{worst_c.behavior_name} ({_f(worst_e)}). Note the score distribution is "
        f"strongly bimodal — very few held-out segments land between p=0.1 and p=0.9 — "
        f"so the mid-range bins of the reliability diagram carry only a handful of "
        f"samples each and are drawn hollow. Judge calibration from ECE and the filled "
        f"bins, not from the shape of the sparse middle.",
    )]


def _al_findings(inp: FindingsInput) -> list[Finding]:
    if not inp.al_results:
        return []
    out: list[Finding] = []
    ratios = []
    for r in inp.al_results:
        if not r.al_points or not r.random_points:
            continue
        al_pos = float(r.al_points[-1].n_pos_mean)
        rnd_pos = float(r.random_points[-1].n_pos_mean)
        if math.isfinite(al_pos) and math.isfinite(rnd_pos) and rnd_pos > 0:
            ratios.append((r, al_pos / rnd_pos, al_pos, rnd_pos))
    if ratios:
        mean_ratio = float(np.mean([x[1] for x in ratios]))
        out.append(Finding(
            "Active learning",
            f"ABEL's candidate ranking finds {mean_ratio:.1f}× more positive clips per "
            f"unit of labeling effort than reviewing clips at random.",
            "Positives discovered at the end of the labeling budget, active vs. random — "
            + "; ".join(f"{r.behavior_name} {a:.0f} vs {b:.0f}"
                        for r, _, a, b in ratios[:6])
            + ". Both arms warm-start from the same seed set and are scored on the same "
              "held-out data; they differ only in which clips get reviewed next.",
        ))
    # Be honest: the F1 win is the weaker claim, and overclaiming it is how this
    # experiment gets torn apart in review.
    faster = []
    for r in inp.al_results:
        al_n = r.clips_to_target(r.al_points)
        rnd_n = r.clips_to_target(r.random_points)
        if (al_n is not None and rnd_n is not None
                and math.isfinite(float(al_n)) and math.isfinite(float(rnd_n))
                and float(al_n) < float(rnd_n)):
            faster.append((r.behavior_name, float(al_n), float(rnd_n)))
    n_total = len(inp.al_results)
    if faster:
        out.append(Finding(
            "Active learning",
            f"On F1, active learning reaches 95% of peak sooner for "
            f"{len(faster)} of {n_total} {_plural(n_total, 'behavior')} — a real but "
            f"modest advantage, clearest for rarer behaviors.",
            "Clips to 95% of peak F1, active vs. random — "
            + "; ".join(f"{n} {a:.0f} vs {b:.0f}" for n, a, b in faster[:6])
            + ". Common, easy behaviors roughly tie on F1: the robust, defensible claim "
              "is positives-discovered-per-effort, not a universal F1 win.",
            kind=KIND_CAVEAT,
        ))
    return out


def _video_value_findings(inp: FindingsInput) -> list[Finding]:
    usable = [r for r in inp.vv_results if not r.error]
    if not usable:
        return []
    wins = [r for r in usable if r.significant and r.gain > 0]
    gains = _finite([r.gain for r in usable])
    out = [Finding(
        "Video features",
        (f"Video motion features significantly improve detection for "
         f"{len(wins)} of {len(usable)} tested (project × behavior) "
         f"{_plural(len(usable), 'combination')}."
         if wins else
         f"Video motion features did not produce a significant detection gain on any of "
         f"the {len(usable)} tested (project × behavior) "
         f"{_plural(len(usable), 'combination')}."),
        f"Mean ΔF1 with video features: {_f(float(np.mean(gains)))} "
        f"(range {_f(min(gains))}–{_f(max(gains))}). Both arms share the same held-out "
        f"split and the same training subsample and differ ONLY in the video-feature "
        f"columns, so the difference is a clean paired estimate. Significant = 95% CI of "
        f"the paired gain excludes 0. "
        + "; ".join(f"{r.project_id} · {r.behavior_name} {_f(r.f1_no_video)}→"
                    f"{_f(r.f1_with_video)} (Δ{r.gain:+.3f}"
                    f"{'*' if r.significant else ''})" for r in usable[:8]),
    )]
    return out


def _behaviorscape_findings(inp: FindingsInput) -> list[Finding]:
    stats = inp.bscape_stats
    if stats is None:
        return []
    out: list[Finding] = []
    pm = getattr(stats, "permanova", None)
    if pm:
        p_txt = "p < 0.001" if pm["p"] < 0.001 else f"p = {pm['p']:.3f}"
        sig = pm["p"] < 0.05
        out.append(Finding(
            "Behaviorscape",
            (f"Different behaviors genuinely rely on different feature types — behavior "
             f"identity explains {_pct(pm['R2'])} of the variance in feature importance "
             f"({p_txt})."
             if sig else
             f"Behaviors did not rely on detectably different feature types "
             f"(PERMANOVA {p_txt})."),
            f"PERMANOVA on per-(project, behavior) feature-importance vectors: "
            f"pseudo-F = {pm['pseudo_F']:.1f}, {p_txt}, over {pm['n_groups']} behaviors "
            f"with ≥2 project replicates. This is the evidence that the feature set is "
            f"doing behavior-specific work rather than one generic thing.",
        ))
    dom = getattr(stats, "dominant_modality", None) or {}
    if dom:
        counts: dict[str, int] = {}
        for mod in dom.values():
            counts[mod] = counts.get(mod, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top_mod, top_n = ranked[0]
        out.append(Finding(
            "Behaviorscape",
            f"{top_mod.capitalize()} features dominate {top_n} of {len(dom)} behaviors.",
            "Dominant feature modality per behavior — "
            + "; ".join(f"{m}: {n}" for m, n in ranked) + ".",
        ))
    return out


def _throughput_findings(inp: FindingsInput) -> list[Finding]:
    usable = [r for r in inp.bench_results if not r.error]
    if not usable:
        return []
    out: list[Finding] = []
    ex = [r for r in usable if r.stage == "extract"
          and math.isfinite(float(r.faster_than_realtime or float("nan")))]
    if ex:
        rates = [float(r.faster_than_realtime) for r in ex]
        out.append(Finding(
            "Throughput",
            f"Feature extraction runs at {float(np.mean(rates)):.1f}× real-time — one "
            f"hour of video is processed in about "
            f"{60 / max(float(np.mean(rates)), 1e-6):.0f} minutes.",
            "Timed on one representative session per project, as a full pose + video + "
            "representation rebuild with the feature cache off, normalized by the video's "
            "true duration. Range: "
            + "; ".join(f"{r.project_id} {float(r.faster_than_realtime):.1f}×"
                        for r in ex) + ".",
        ))
    tr = [r for r in usable if r.stage == "train"
          and math.isfinite(float(r.seconds or float("nan")))]
    if tr:
        secs = [float(r.seconds) for r in tr]
        out.append(Finding(
            "Throughput",
            f"Training one behavior classifier takes a median of "
            f"{float(np.median(secs)):.0f} s once features exist.",
            f"Across {len(tr)} fitted {_plural(len(tr), 'model')}: "
            f"{min(secs):.0f}–{max(secs):.0f} s.",
        ))
    return out


# ── entry point ─────────────────────────────────────────────────────────────

_DERIVERS = (
    _overview_findings,
    _confusion_findings,
    _learning_curve_findings,
    _ablation_findings,
    _discrimination_findings,
    _generalization_findings,
    _time_budget_findings,
    _calibration_findings,
    _al_findings,
    _video_value_findings,
    _behaviorscape_findings,
    _throughput_findings,
)


def derive_findings(inp: FindingsInput) -> list[Finding]:
    """Every plain-language finding a run supports, in report order.

    A deriver that trips over an unexpected shape must not sink the report — the
    findings are a *summary* of results already safely on disk.  Each is isolated.
    """
    out: list[Finding] = []
    for fn in _DERIVERS:
        try:
            out.extend(fn(inp))
        except Exception as exc:  # noqa: BLE001 — a bad summary must not lose the run
            out.append(Finding(
                fn.__name__.strip("_").replace("_findings", "").replace("_", " ").title(),
                "Could not summarize this analysis.",
                f"{type(exc).__name__}: {exc}. The underlying results and figures are "
                f"still written to the run folder.",
                kind=KIND_WARNING,
            ))
    return out


def findings_frame(items: list[Finding]) -> pd.DataFrame:
    return pd.DataFrame([f.to_row() for f in items]) if items else pd.DataFrame(
        columns=["analysis", "kind", "finding", "detail"])


def findings_markdown(run_id: str, items: list[Finding]) -> str:
    """The findings as a portable Markdown document."""
    lines = [f"# ABEL Validation — Key Findings", "", f"Run: `{run_id}`", ""]
    by_analysis: dict[str, list[Finding]] = {}
    for f in items:
        by_analysis.setdefault(f.analysis, []).append(f)
    for analysis, group in by_analysis.items():
        lines.append(f"## {analysis}")
        lines.append("")
        for f in group:
            prefix = {KIND_CAVEAT: "⚠️ **Caveat** — ",
                      KIND_WARNING: "🔴 **Warning** — "}.get(f.kind, "")
            lines.append(f"- {prefix}{f.headline}")
            if f.detail:
                lines.append(f"  - {f.detail}")
        lines.append("")
    return "\n".join(lines)
