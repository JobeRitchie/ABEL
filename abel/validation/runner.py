"""Top-level orchestrator: run enabled analyses across projects and persist.

This is the single entry point the GUI (and headless scripts) call.  It owns the
project loop, the holdout split per project, progress/ETA reporting, and writing
all artifacts into a :class:`ResultsStore` directory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.validation import (
    aggregate, benchmark, findings as findings_mod, holdout, meta_summary, pdf_report,
    plots, prism, report, subsample, video_value,
)
from abel.validation.analyses import (
    ablation, al_curve, behaviorscape, calibration, cross_project, discrimination,
    feature_roles, generalization, learning_curve, rare_discovery, time_budget,
)
from abel.validation.datamodel import CellResult, ProjectRef, RunManifest
from abel.validation.store import ResultsStore

ANALYSIS_LEARNING_CURVE = "learning_curve"
ANALYSIS_ABLATION = "ablation"
ANALYSIS_GENERALIZATION = "generalization"
ANALYSIS_DISCRIMINATION = "discrimination"
ANALYSIS_AL_CURVE = "al_curve"
ANALYSIS_BEHAVIORSCAPE = "behaviorscape"
ANALYSIS_VIDEO_VALUE = "video_value"
ANALYSIS_THROUGHPUT = "throughput"
ANALYSIS_RARE_DISCOVERY = "rare_discovery"

ALL_ANALYSES = [
    ANALYSIS_LEARNING_CURVE, ANALYSIS_ABLATION, ANALYSIS_GENERALIZATION,
    ANALYSIS_DISCRIMINATION,
]

# Everything the "Run Full Suite" button offers, in report order.  This is a
# superset of ALL_ANALYSES: the last three used to live only as their own GUI
# workers, outside the runner and outside the shared holdout split.
FULL_SUITE = [
    ANALYSIS_LEARNING_CURVE, ANALYSIS_ABLATION, ANALYSIS_DISCRIMINATION,
    ANALYSIS_GENERALIZATION, ANALYSIS_AL_CURVE, ANALYSIS_BEHAVIORSCAPE,
    ANALYSIS_VIDEO_VALUE, ANALYSIS_THROUGHPUT,
]

ANALYSIS_LABELS = {
    ANALYSIS_LEARNING_CURVE: "Learning curves (optimal clips)",
    ANALYSIS_ABLATION: "Feature / pipeline ablation",
    ANALYSIS_DISCRIMINATION: "Pairwise behavior discrimination",
    ANALYSIS_GENERALIZATION: "Generalization, biological readout & calibration",
    ANALYSIS_AL_CURVE: "Active learning vs. random",
    ANALYSIS_BEHAVIORSCAPE: "Behaviorscape (feature modalities)",
    ANALYSIS_VIDEO_VALUE: "Video-feature value (paired)",
    ANALYSIS_THROUGHPUT: "Pipeline throughput",
    ANALYSIS_RARE_DISCOVERY: "Rare-behavior discovery (clip hunting)",
}

ProgressCB = Callable[[str, float], None]


@dataclass
class ValidationRunConfig:
    analyses: list[str] = field(default_factory=lambda: list(ALL_ANALYSES))
    # learning curve
    sizes: list[int] = field(default_factory=lambda: list(learning_curve.DEFAULT_SIZES))
    n_seeds_lc: int = 5
    neg_policy: str = "all"
    neg_per_pos: float = 3.0
    # ablation / generalization
    n_seeds_ablation: int = 3
    ablation_budgets: list[int] = field(default_factory=lambda: [subsample.ALL_CLIPS])
    n_seeds_generalization: int = 3
    # pairwise discrimination (which features separate similar behaviors)
    n_seeds_discrimination: int = 3
    discrimination_max_pairs: int = 15
    # active-learning vs random
    n_seeds_al: int = 3
    al_k0: int = 20
    al_batch: int = 15
    al_max_budget: int = 200
    al_seed_pos: int = 5
    al_acquisition: str = "probability"
    # rare-behavior discovery (clip hunting: essence / AL / UMAP vs random / whole-video)
    # With several projects selected, hunting every checked behaviour in every one
    # of them costs hours and buries the result.  Auto-target runs the *cheap*
    # rarity pass first (dense bout detections, no fitting), then spends the whole
    # discovery/quality budget on that project's rarest behaviour — the one the
    # analysis is actually about — before moving to the next project.
    rare_auto_target: bool = True
    n_seeds_rare: int = 5
    # 20 exemplars, matching the real workflow: a user reaches the clip-hunting
    # tools after an initial random-hunting phase, not from a cold 8-clip start.
    # Measured on the full pool this cuts essence's effort-to-50 from 486 to 336
    # clips; on the (enriched) reviewed pool it is worth ~nothing, which is why the
    # effect was invisible before.  Do not raise much further — at 40 exemplars the
    # essence AND-box widens and purity regresses.
    rare_n_seed_pos: int = 20
    rare_al_budget: int = 400
    rare_effort_targets: list[int] = field(default_factory=lambda: [10, 25, 50])
    # Full segment pool at deployment rarity.  ON by default: the reviewed pool is
    # ~12x enriched for the target, which deflates fold-enrichment by a similar
    # factor and lets random look like a credible competitor when at true rarity it
    # trains a model with zero target F1.  Guarded by _label_coverage, which refuses
    # projects whose labels do not join cleanly to segment_features.parquet.
    rare_include_fullpool: bool = True
    rare_include_rarity_scaling: bool = True
    rare_measure: str = "time_fraction"    # behavior-rarity readout
    # Effort-to-quality: labeling effort → held-out target-class F1 / PR-AUC per
    # acquisition strategy ("how much review until the model is actually good?").
    # This is the PRIMARY rare-behaviour result — positives-found is the mechanism,
    # but the two can rank the arms differently, so the outcome metric leads.  Each
    # arm trains a model at every checkpoint; measured at ~0.5 s per fit, a full
    # behaviour is minutes, not hours.
    rare_include_quality: bool = True
    rare_quality_k0: int = 20
    rare_quality_seed_pos: int = 5
    rare_quality_batch: int = 25
    rare_quality_budget: int = 300
    rare_quality_f1_targets: list[float] = field(default_factory=lambda: [0.70, 0.80])
    rare_quality_pr_auc_targets: list[float] = field(default_factory=lambda: [0.80, 0.90])
    rare_quality_frac_targets: list[float] = field(default_factory=lambda: [0.90, 0.95])
    # Behaviours to EXCLUDE from the rarity comparison (names or ids) — e.g. one
    # that is not validly scored in this dataset and would pollute "the rest".
    rare_exclude_behaviors: list[str] = field(default_factory=list)
    # behaviorscape
    bscape_threshold: float = 0.010
    bscape_normalize: str = "fraction"
    bscape_alias_map: dict[str, str] = field(default_factory=dict)
    # video-feature value (paired with/without)
    n_seeds_video_value: int = 5
    # pipeline throughput.  Dense inference is OFF by default: unlike every other
    # analysis it has side effects on the real project (it recomputes that
    # session's temporal-refinement traces).
    throughput_stages: list[str] = field(
        default_factory=lambda: [benchmark.STAGE_EXTRACT, benchmark.STAGE_TRAIN])
    # holdout
    min_confidence: float = 1.0
    holdout_test_size: float = 0.25
    holdout_seed: int = 42
    # optional explicit holdout groups per project_id
    holdout_groups: dict[str, list[str]] = field(default_factory=dict)
    output_root: str = ""

    def to_dict(self) -> dict:
        return {
            "analyses": self.analyses,
            "sizes": self.sizes,
            "n_seeds_lc": self.n_seeds_lc,
            "neg_policy": self.neg_policy,
            "neg_per_pos": self.neg_per_pos,
            "n_seeds_ablation": self.n_seeds_ablation,
            "ablation_budgets": self.ablation_budgets,
            "n_seeds_generalization": self.n_seeds_generalization,
            "n_seeds_discrimination": self.n_seeds_discrimination,
            "discrimination_max_pairs": self.discrimination_max_pairs,
            "rare_auto_target": self.rare_auto_target,
            "n_seeds_rare": self.n_seeds_rare,
            "rare_n_seed_pos": self.rare_n_seed_pos,
            "rare_al_budget": self.rare_al_budget,
            "rare_include_fullpool": self.rare_include_fullpool,
            "rare_include_rarity_scaling": self.rare_include_rarity_scaling,
            "rare_include_quality": self.rare_include_quality,
            "rare_quality_k0": self.rare_quality_k0,
            "rare_quality_seed_pos": self.rare_quality_seed_pos,
            "rare_quality_batch": self.rare_quality_batch,
            "rare_quality_budget": self.rare_quality_budget,
            "rare_exclude_behaviors": self.rare_exclude_behaviors,
            "n_seeds_al": self.n_seeds_al,
            "al_k0": self.al_k0,
            "al_batch": self.al_batch,
            "al_max_budget": self.al_max_budget,
            "al_seed_pos": self.al_seed_pos,
            "al_acquisition": self.al_acquisition,
            "bscape_threshold": self.bscape_threshold,
            "bscape_normalize": self.bscape_normalize,
            "n_seeds_video_value": self.n_seeds_video_value,
            "throughput_stages": self.throughput_stages,
            "min_confidence": self.min_confidence,
            "holdout_test_size": self.holdout_test_size,
            "holdout_seed": self.holdout_seed,
            "holdout_groups": self.holdout_groups,
        }


# ── The publication preset ──────────────────────────────────────────────────
#
# One fixed, defensible setting per analysis, so "run everything properly" is a
# button and not a form.  The seed counts are the load-bearing choice: the 95%
# interval on a mean of n seeds uses the *t* quantile, which is 4.30 at 3 seeds
# and 2.78 at 5.  Three seeds therefore buys an interval so wide that real gains
# fail to reach significance — every analysis whose headline is a *difference*
# (ablation, discrimination, active learning, video value) gets 5.
PUBLICATION_SEEDS = 5


def publication_config(
    analyses: list[str] | None = None,
    output_root: str = "",
    **overrides,
) -> ValidationRunConfig:
    """The settings the suite should be run at for a publication-grade result."""
    cfg = ValidationRunConfig(
        analyses=list(analyses if analyses is not None else FULL_SUITE),
        # Data efficiency: a dense low-end schedule is what resolves the knee —
        # the interesting curvature is all below ~100 clips.
        sizes=[10, 25, 50, 75, 100, 150, 200, subsample.ALL_CLIPS],
        n_seeds_lc=PUBLICATION_SEEDS,
        neg_policy="all",
        n_seeds_ablation=PUBLICATION_SEEDS,
        # Two budgets: regularizers (calibration, augmentation) pay off in the
        # low-data regime and vanish at full data, so a full-data-only ablation
        # reports "no effect" for features that genuinely help where it matters.
        ablation_budgets=[50, subsample.ALL_CLIPS],
        n_seeds_generalization=PUBLICATION_SEEDS,
        n_seeds_discrimination=PUBLICATION_SEEDS,
        discrimination_max_pairs=15,
        n_seeds_al=PUBLICATION_SEEDS,
        al_k0=20,
        al_batch=15,
        al_max_budget=200,
        al_seed_pos=5,
        al_acquisition="probability",
        bscape_threshold=0.010,
        bscape_normalize="fraction",
        n_seeds_video_value=PUBLICATION_SEEDS,
        throughput_stages=[benchmark.STAGE_EXTRACT, benchmark.STAGE_TRAIN],
        # Score only against labels the reviewer was certain of.
        min_confidence=1.0,
        holdout_test_size=0.25,
        holdout_seed=42,
        output_root=output_root,
    )
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def preset_description(cfg: ValidationRunConfig) -> str:
    """Human-readable summary of what the preset will do (shown in the GUI)."""
    sizes = ", ".join("all" if s == subsample.ALL_CLIPS else str(s) for s in cfg.sizes)
    budgets = ", ".join("all" if b == subsample.ALL_CLIPS else str(b)
                        for b in cfg.ablation_budgets)
    return (
        f"{PUBLICATION_SEEDS} seeds per point (95% CI uses the t quantile — at 3 seeds "
        f"that is 4.30 and real effects miss significance; at 5 it is 2.78)\n"
        f"Learning-curve clip schedule: {sizes}\n"
        f"Ablation clip budgets: {budgets}  (low-data + full-data)\n"
        f"Discrimination: up to {cfg.discrimination_max_pairs} behavior pairs, "
        f"closest-centroid pairs kept first\n"
        f"Active learning: seed {cfg.al_k0} clips ({cfg.al_seed_pos} guaranteed "
        f"positives), +{cfg.al_batch}/step to {cfg.al_max_budget}\n"
        f"Held-out: {cfg.holdout_test_size:.0%} of subjects/sessions, reviewer "
        f"confidence ≥ {cfg.min_confidence:g}\n"
        f"Throughput stages: {', '.join(cfg.throughput_stages) or 'none'} "
        f"(dense inference excluded — it rewrites project traces)"
    )


@dataclass
class RunOutputs:
    run_dir: Path
    cells: pd.DataFrame
    report_path: Path
    manifest_path: Path
    # The consolidated summary: findings in words + headline figures.  The PDF is
    # rendered by the caller on the GUI thread (QtWebEngine needs an event loop),
    # so the runner produces the print-ready HTML and leaves ``pdf_path`` unset.
    summary_html: Path | None = None
    findings: list = field(default_factory=list)
    pdf_path: Path | None = None


def _default_root() -> Path:
    """Where a run lands when no output root was given.

    Runs belong to the validation workspace, not to any one project's parent
    folder — a run spans projects, and scattering results next to whichever
    project happened to be first made them impossible to find later.  A run
    launched from a saved session gets that session's ``runs/`` folder instead
    (the GUI passes it as ``output_root``); this is the unfiled fallback.
    """
    from abel.validation.workspace import workspace_root  # noqa: PLC0415

    return workspace_root() / "unfiled_runs"


def run_rarity_preflight(
    projects: list[ProjectRef],
    behaviors: dict[str, list[str]],
    config: ValidationRunConfig | None = None,
    progress_cb: ProgressCB | None = None,
) -> list:
    """Phase 1 of the rare-behaviour workflow: the cheap rarity + evidence check.

    Ranks each project's checked behaviours by measured rarity and reports how
    many confirmed examples the rarest ones actually have — seconds of parquet
    reads, no model fitting — so the user can go label more *before* committing
    hours to a hunt that would die on three positives.  Returns a list of
    :class:`rare_discovery.ProjectPreflight`, one per project.
    """
    config = config or ValidationRunConfig()
    excl_by_project: dict[str, list[str]] = {}
    for proj in projects:
        ids = list(proj.behavior_ids_matching(config.rare_exclude_behaviors))
        ids += [b for b in config.rare_exclude_behaviors
                if b in proj.behavior_names and b not in ids]
        excl_by_project[proj.project_id] = ids

    results = []
    todo = [p for p in projects if behaviors.get(p.project_id)]
    for i, proj in enumerate(todo):
        def _log(msg: str, _i=i) -> None:
            if progress_cb is not None:
                progress_cb(msg, _i / max(1, len(todo)))
        results.append(rare_discovery.preflight_project(
            proj, behaviors[proj.project_id],
            n_seed_pos=config.rare_n_seed_pos,
            min_effort_target=min(config.rare_effort_targets or [10]),
            measure=config.rare_measure,
            exclude_behavior_ids=excl_by_project[proj.project_id],
            holdout_groups=config.holdout_groups.get(proj.project_id),
            min_confidence=config.min_confidence,
            test_size=config.holdout_test_size,
            seed=config.holdout_seed,
            progress_cb=_log,
        ))
    if progress_cb is not None:
        progress_cb("Rarity check complete.", 1.0)
    return results


def run_validation(
    projects: list[ProjectRef],
    behaviors: dict[str, list[str]],
    config: ValidationRunConfig | None = None,
    progress_cb: ProgressCB | None = None,
) -> RunOutputs:
    """Run all enabled analyses for ``behaviors`` (per project) and write results."""
    config = config or ValidationRunConfig()
    root = Path(config.output_root) if config.output_root else _default_root()
    store = ResultsStore(root)
    trainer = ActiveLearningTrainerService()

    # Rough unit count for progress fraction.
    def _units_for(beh_count: int, rare_count: int) -> int:
        u = 0
        if ANALYSIS_LEARNING_CURVE in config.analyses:
            u += beh_count * len(config.sizes) * config.n_seeds_lc
        if ANALYSIS_ABLATION in config.analyses:
            # ~7 configs (baseline + singles + all), once per clip budget
            u += beh_count * 7 * config.n_seeds_ablation * max(1, len(config.ablation_budgets))
        if ANALYSIS_GENERALIZATION in config.analyses:
            u += beh_count * config.n_seeds_generalization
        if ANALYSIS_DISCRIMINATION in config.analyses:
            # pairs × ~3 feature families × seeds (+1 discovery fit)
            n_pairs = min(config.discrimination_max_pairs,
                          beh_count * max(0, beh_count - 1) // 2)
            u += n_pairs * 3 * config.n_seeds_discrimination + 1
        if ANALYSIS_AL_CURVE in config.analyses:
            steps = max(1, (config.al_max_budget - config.al_k0) // max(1, config.al_batch) + 1)
            u += beh_count * config.n_seeds_al * 2 * steps  # AL + random arms
        if ANALYSIS_VIDEO_VALUE in config.analyses:
            u += beh_count * config.n_seeds_video_value * 2  # with + without arms
        if ANALYSIS_BEHAVIORSCAPE in config.analyses:
            u += beh_count                                   # one fit per behavior
        if ANALYSIS_THROUGHPUT in config.analyses:
            u += len(config.throughput_stages)
        if ANALYSIS_RARE_DISCOVERY in config.analyses:
            # One coarse unit per completed sub-analysis (discovery, +rarity,
            # +full pool) per behavior, plus one behaviour-rarity readout/project.
            per_beh = 1 + (1 if config.rare_include_rarity_scaling else 0) \
                + (1 if config.rare_include_fullpool else 0)
            u += rare_count * per_beh + 1
            if config.rare_include_quality:
                # Effort-to-quality trains a model per checkpoint per arm, so it is
                # the one rare sub-analysis worth costing at fit granularity — a
                # coarse +1 would make the ETA stall badly here.
                steps = max(1, (config.rare_quality_budget - config.rare_quality_k0)
                            // max(1, config.rare_quality_batch) + 1)
                u += rare_count * config.n_seeds_rare * 4 * steps  # 4 acquisition arms
        return max(1, u)

    def _rare_count(beh_count: int) -> int:
        """Behaviours the rare analyses will actually hunt in one project."""
        return 1 if (config.rare_auto_target and beh_count > 1) else beh_count

    total_units = sum(
        _units_for(n := len(behaviors.get(p.project_id, [])), _rare_count(n))
        for p in projects)
    done = {"n": 0}
    _t_start = time.monotonic()

    def _eta_suffix(frac: float) -> str:
        """A wall-clock ETA appended to every progress message.

        Held behind a short warm-up gate ("calculating…") because the first few
        units are unrepresentative — the estimate uses whole-run throughput
        (elapsed ÷ fraction), which is the honest extrapolation when units take
        wildly different times (an AL fit ≫ a learning-curve point).
        """
        elapsed = time.monotonic() - _t_start
        if frac <= 0.02 or elapsed < 5.0:
            return " · ETA calculating…"
        remaining = elapsed * (1.0 - frac) / max(frac, 1e-9)
        return f" · ~{_fmt_dur(remaining)} left ({int(frac * 100)}%)"

    def _emit(msg: str) -> None:
        done["n"] += 1
        frac = min(1.0, done["n"] / total_units)
        if progress_cb:
            progress_cb(msg + _eta_suffix(frac), frac)

    def _emit_msg(msg: str) -> None:
        frac = min(1.0, done["n"] / total_units)
        if progress_cb:
            progress_cb(msg + _eta_suffix(frac), frac)

    all_cells: list[CellResult] = []
    lc_results: list = []
    abl_results: list = []
    gen_results: list = []
    al_results: list = []
    rare_reviewed: list = []
    rare_full: list = []
    rare_rarity: list = []
    rare_behavior: list = []
    rare_quality: list = []
    rare_target_rows: list[dict] = []      # which behaviour each project hunted, and why
    proj_names: dict[str, str] = {p.project_id: p.name for p in projects}
    vv_results: list = []
    bench_results: list = []
    tb_results: list = []
    cal_results: list = []
    disc_by_project: dict[str, list] = {}
    knees: list[dict] = []
    project_meta: list[dict] = []
    bscape_data = None
    bscape_stats = None

    for proj in projects:
        beh_ids = behaviors.get(proj.project_id, [])
        if not beh_ids:
            continue
        _emit_msg(f"[{proj.name}] loading + splitting holdout…")
        sp = holdout.split(
            proj,
            holdout_groups=config.holdout_groups.get(proj.project_id),
            min_confidence=config.min_confidence,
            test_size=config.holdout_test_size,
            seed=config.holdout_seed,
        )
        store.write_holdout_manifest(proj.project_id, sp.manifest(proj))
        project_meta.append({
            "project_id": proj.project_id, "name": proj.name,
            # Renames are display-only, so the manifest carries both names: a figure
            # labelled "Groom" must stay traceable to the project/behavior on disk.
            "source_name": proj.original_name,
            "renamed": proj.is_renamed,
            "root": str(proj.root),
            "classifier_type": proj.classifier_type,
            "n_holdout_rows": int(len(sp.holdout)),
            "n_train_pool_rows": int(len(sp.train_pool)),
            "behaviors": beh_ids,
            "behavior_detail": [
                {"behavior_id": bid,
                 "disk_name": proj.behavior_disk_name(bid),
                 "display_name": proj.behavior_label(bid)}
                for bid in beh_ids
            ],
        })

        # ── which behaviour does this project's rare-discovery budget go to? ──
        # The cheap pass first: rank every selected behaviour by how rare it
        # actually is (dense bout detections — no fitting, seconds not hours), then
        # hunt the rarest one.  The rest of the ranking is kept as a fallback chain:
        # the rarest behaviour is also the likeliest to have too few confirmed
        # positives to cross-validate, and dropping the project entirely over that
        # is worse than moving one rank down and saying so.
        rare_excl_ids: list[str] = []
        rare_targets: list[str] = []
        rare_auto = False
        if ANALYSIS_RARE_DISCOVERY in config.analyses:
            rare_excl_ids = list(proj.behavior_ids_matching(config.rare_exclude_behaviors))
            rare_excl_ids += [b for b in config.rare_exclude_behaviors
                              if b in proj.behavior_names and b not in rare_excl_ids]
            rare_auto = bool(config.rare_auto_target) and len(beh_ids) > 1
            rare_targets = list(beh_ids)
            if rare_auto:
                _emit_msg(f"[{proj.name}] ranking behaviours by rarity (cheap pass)…")
                try:
                    ranking = rare_discovery.rank_behaviors_by_rarity(
                        proj, beh_ids, exclude_behavior_ids=rare_excl_ids,
                        measure=config.rare_measure, progress_cb=_emit_msg)
                except Exception as exc:  # noqa: BLE001 — fall back to the selection order
                    _emit_msg(f"[{proj.name}] rarity ranking unavailable ({exc}); "
                              f"using the checked behaviours in order")
                    ranking = []
                if ranking:
                    rare_targets = [bid for bid, _n, _v in ranking]
                    _emit_msg(f"[{proj.name}] rarest → " + ", ".join(
                        f"{n} ({v:.4g})" for _b, n, v in ranking[:3]))

        # Behaviour-independent clip metrics + UMAP embedding, computed ONCE per
        # project and shared across every behaviour's discovery run (bit-identical
        # to recomputing — they depend only on the pool, not the target).
        rare_cache = None
        if rare_targets:
            _emit_msg(f"[{proj.name}] preparing shared clip metrics + embedding…")
            rare_cache = rare_discovery.prepare_project_cache(
                proj, sp, progress_cb=_emit_msg)

        for beh in beh_ids:
            name = proj.behavior_label(beh)

            if ANALYSIS_LEARNING_CURVE in config.analyses:
                lc = learning_curve.run_learning_curve(
                    trainer, proj, beh, sp,
                    sizes=config.sizes, n_seeds=config.n_seeds_lc,
                    neg_policy=config.neg_policy, neg_per_pos=config.neg_per_pos,
                    progress_cb=lambda m: _emit(m),
                )
                lc_results.append(lc)
                all_cells.extend(lc.cells)
                knees.append({
                    "project_id": proj.project_id, "project_name": proj.name,
                    "behavior_name": name, "knee_clips": lc.knee_clips, "f1_max": lc.f1_max,
                })
                lc_dir = store.sub("learning_curves")
                stem = f"{_tag(proj.project_id)}__{_tag(name)}"
                for view in plots.LEARNING_CURVE_VIEWS:
                    plots.learning_curve_plot(lc, lc_dir / f"{stem}__{view}.png", view=view)
                pd.DataFrame(_lc_points_rows(lc)).to_csv(lc_dir / f"{stem}__points.csv", index=False)
                plots.close_all()

            if ANALYSIS_ABLATION in config.analyses:
                for budget in (config.ablation_budgets or [subsample.ALL_CLIPS]):
                    ab = ablation.run_ablation(
                        trainer, proj, beh, sp, n_seeds=config.n_seeds_ablation,
                        clip_budget=budget, neg_policy=config.neg_policy,
                        neg_per_pos=config.neg_per_pos,
                        progress_cb=lambda m: _emit(m),
                    )
                    abl_results.append(ab)
                    all_cells.extend(ab.cells)

            if ANALYSIS_GENERALIZATION in config.analyses:
                gen = generalization.run_generalization(
                    trainer, proj, beh, sp, n_seeds=config.n_seeds_generalization,
                    progress_cb=lambda m: _emit(m),
                )
                gen_results.append(gen)
                all_cells.extend(gen.cells)

            if ANALYSIS_AL_CURVE in config.analyses:
                al = al_curve.run_al_vs_random(
                    trainer, proj, beh, sp,
                    n_seeds=config.n_seeds_al, k0=config.al_k0, batch=config.al_batch,
                    max_budget=config.al_max_budget, seed_pos=config.al_seed_pos,
                    acquisition=config.al_acquisition,
                    progress_cb=lambda m: _emit(m),
                )
                al_results.append(al)
                all_cells.extend(al.cells)

            if ANALYSIS_VIDEO_VALUE in config.analyses:
                # Shares this project's holdout split, so the paired with/without
                # comparison is scored on exactly the same held-out rows as
                # everything else in the run.
                vv = video_value.run_video_value(
                    trainer, proj, beh, sp, n_seeds=config.n_seeds_video_value,
                    progress_cb=lambda m: _emit(m),
                )
                vv_results.append(vv)

        # ── rare-behavior discovery (its own loop over this project's targets) ──
        # In auto-target mode ``rare_targets`` is the rarity ranking and we stop at
        # the first behaviour that is actually huntable; otherwise it is every
        # checked behaviour and all of them run.
        hunted: list[str] = []
        for beh in rare_targets:
            name = proj.behavior_label(beh)
            # Clip-hunting efficiency for this behaviour (essence / AL / UMAP vs
            # random / whole-video), cross-validated on the shared holdout pool.
            # Guarded: a behaviour with too few confirmed positives to
            # cross-validate is skipped rather than sinking the run.
            # Fine-grained logs route through _emit_msg (message + ETA, no unit
            # count); the unit counter advances once per completed sub-analysis
            # (below), so the coarse fraction stays honest and the ETA doesn't
            # overshoot to 100%.
            try:
                rd_res = rare_discovery.run_rare_discovery(
                    trainer, proj, beh, sp,
                    n_seeds=config.n_seeds_rare,
                    n_seed_pos=config.rare_n_seed_pos,
                    al_max_budget=config.rare_al_budget,
                    display_budget=config.rare_al_budget,
                    effort_targets=tuple(config.rare_effort_targets),
                    cache=rare_cache,
                    progress_cb=_emit_msg,
                )
                rare_reviewed.append(rd_res)
                all_cells.extend(rd_res.cells)
                hunted.append(beh)
                _emit(f"[{name}] clip-hunting discovery done")
                if config.rare_include_fullpool:
                    rare_full.append(rare_discovery.run_full_pool_supplement(
                        proj, beh, n_seeds=config.n_seeds_rare,
                        n_seed_pos=config.rare_n_seed_pos,
                        progress_cb=_emit_msg))
                    _emit(f"[{name}] full-pool supplement done")
                if config.rare_include_rarity_scaling:
                    rare_rarity.append(rare_discovery.run_rarity_scaling(
                        proj, beh, sp, n_seeds=config.n_seeds_rare,
                        cache=rare_cache, progress_cb=_emit_msg))
                    _emit(f"[{name}] rarity scaling done")
            except Exception as exc:  # noqa: BLE001 — skip un-huntable behaviours
                _emit(f"[{name}] rare-discovery skipped: {type(exc).__name__}: {exc}")
                if rare_auto:
                    # The rarest behaviour could not be cross-validated — drop one
                    # rank and try again rather than losing the whole project.
                    _emit_msg(f"[{proj.name}] trying the next-rarest behaviour…")
                    continue

            # Effort-to-quality: same acquisition arms, but the y-axis is model
            # quality on the fixed holdout rather than positives found — "how
            # much labeling until the model is good?".  Guarded separately from
            # the discovery block so a failure in one still leaves the other.
            # Each checkpoint fit emits a unit (see _units_for), so the ETA
            # tracks this the way it tracks the AL curve.
            if config.rare_include_quality:
                try:
                    q_res = rare_discovery.run_effort_to_quality(
                        trainer, proj, beh, sp,
                        n_seeds=config.n_seeds_rare,
                        k0=config.rare_quality_k0,
                        seed_pos=config.rare_quality_seed_pos,
                        batch=config.rare_quality_batch,
                        max_budget=config.rare_quality_budget,
                        f1_targets=tuple(config.rare_quality_f1_targets),
                        pr_auc_targets=tuple(config.rare_quality_pr_auc_targets),
                        frac_targets=tuple(config.rare_quality_frac_targets),
                        cache=rare_cache,
                        progress_cb=lambda m: _emit(m),
                    )
                    rare_quality.append(q_res)
                    all_cells.extend(q_res.cells)
                    _emit_msg(f"[{name}] effort-to-quality done")
                except Exception as exc:  # noqa: BLE001
                    _emit_msg(f"[{name}] effort-to-quality skipped: "
                              f"{type(exc).__name__}: {exc}")
            if rare_auto:
                break  # one behaviour per project — the rarest huntable one

        # ── behavior rarity (once per project, contextualising the hunted target) ──
        # Describes how rare each behaviour actually is (from dense bout detections),
        # the context that makes the rare-behaviour discovery story land.  The
        # highlighted behaviour is the one we actually hunted.
        if rare_targets:
            _emit_msg(f"[{proj.name}] behaviour rarity…")
            # Recorded even if the rarity readout below fails: the reader must be
            # able to see which behaviour the hunt was spent on regardless.
            for beh in hunted:
                rare_target_rows.append({
                    "project": proj.name, "project_id": proj.project_id,
                    "hunted_behavior": proj.behavior_label(beh),
                    "auto_selected": rare_auto,
                    "rarity_measure": config.rare_measure,
                    "rarity_value": float("nan"), "rarity_rank": -1,
                    "n_behaviors_compared": 0, "rarer_behaviors_not_hunted": "",
                })
            for beh in (hunted or rare_targets):
                try:
                    # Compare against EVERY project behaviour (not just the selected
                    # ones), else a single-behaviour selection has nothing to rank
                    # against and the stats are degenerate.
                    br = rare_discovery.run_behavior_rarity(
                        proj, beh, behavior_ids=None,
                        exclude_behavior_ids=rare_excl_ids,
                        measure=config.rare_measure, progress_cb=_emit_msg)
                    rare_behavior.append(br)
                    # Fill the measured rank into this project's placeholder row.
                    for row in rare_target_rows:
                        if (row["project_id"] == proj.project_id
                                and row["hunted_behavior"] == br.target_name):
                            row.update({
                                "rarity_measure": br.measure,
                                "rarity_value": br.target_mean(),
                                "rarity_rank": br.target_rank,
                                "n_behaviors_compared": br.n_behaviors,
                                "rarer_behaviors_not_hunted":
                                    ", ".join(br.rarer_than_target),
                            })
                    break  # one readout per project (all behaviours plotted together)
                except Exception as exc:  # noqa: BLE001
                    _emit_msg(f"[{proj.name}] behaviour rarity skipped: {exc}")
                    break
            _emit("behaviour rarity done")

        # ── pairwise discrimination (once per project, over all behavior pairs) ──
        # Complements the ablation: that one measures *detection* of each behavior
        # against everything else; this one measures whether the features can tell
        # two similar behaviors apart from each other.
        if ANALYSIS_DISCRIMINATION in config.analyses and len(beh_ids) >= 2:
            _emit_msg(f"[{proj.name}] pairwise discrimination…")
            pair_results = discrimination.run_discrimination(
                trainer, proj, beh_ids, sp,
                n_seeds=config.n_seeds_discrimination,
                max_pairs=config.discrimination_max_pairs,
                progress_cb=lambda m: _emit(m),
            )
            disc_by_project[proj.project_id] = pair_results
            for pr in pair_results:
                all_cells.extend(pr.cells)

    # ── behaviorscape (once, pooled across every project) ──
    # Runs outside the project loop because its unit of analysis is the pooled
    # behavior, not the project: it needs one importance vector per
    # (project, behavior) before it can ask whether behaviors differ.
    if ANALYSIS_BEHAVIORSCAPE in config.analyses:
        _emit_msg("Behaviorscape: collecting feature importance…")
        sources = behaviorscape.collect_feature_importance(
            trainer, projects, behaviors,
            min_confidence=config.min_confidence,
            holdout_test_size=config.holdout_test_size,
            holdout_seed=config.holdout_seed,
            holdout_groups=config.holdout_groups,
            progress_cb=lambda m, _f: _emit(m),
        )
        data = behaviorscape.build_behaviorscape(
            sources,
            threshold=config.bscape_threshold,
            alias_map=config.bscape_alias_map,
            normalize=config.bscape_normalize,
        )
        if not data.is_empty():
            bscape_data = data
            try:
                bscape_stats = behaviorscape.behavior_distinctiveness_stats(data)
            except Exception:  # noqa: BLE001 — the PERMANOVA is optional colour
                bscape_stats = None

    # ── pipeline throughput (once per project) ──
    if ANALYSIS_THROUGHPUT in config.analyses and config.throughput_stages:
        for proj in projects:
            names = [proj.behavior_label(b)
                     for b in behaviors.get(proj.project_id, [])] or None
            if benchmark.STAGE_EXTRACT in config.throughput_stages:
                _emit_msg(f"[{proj.name}] timing feature extraction…")
                bench_results.append(benchmark.time_extraction(proj.root, log=_emit_msg))
                _emit("extraction timed")
            if benchmark.STAGE_TRAIN in config.throughput_stages:
                _emit_msg(f"[{proj.name}] timing training…")
                bench_results.extend(
                    benchmark.time_training(proj.root, names, log=_emit_msg))
                _emit("training timed")
            if benchmark.STAGE_INFER in config.throughput_stages:
                _emit_msg(f"[{proj.name}] timing dense inference…")
                bench_results.append(benchmark.time_inference(proj.root, log=_emit_msg))
                _emit("inference timed")

    # ── persist tidy substrate ──
    cells_df = aggregate.cells_to_frame(all_cells)
    store.save_cells(cells_df)

    # ── plots + csvs ──
    sections: list[tuple[str, str]] = []
    # Kept in scope for the Prism export at the end of the run (each is only built
    # if its analysis actually ran).
    abl_df: pd.DataFrame | None = None
    gen_df: pd.DataFrame | None = None
    disc_df: pd.DataFrame | None = None
    vv_df: pd.DataFrame | None = None
    bench_df: pd.DataFrame | None = None

    if lc_results:
        knee_df = cross_project.data_efficiency_summary(knees)
        store.write_csv(knee_df, "optimal_clips_summary.csv", subdir="learning_curves")
        combined_rows = [row for lc in lc_results for row in _lc_points_rows(lc)]
        store.write_csv(pd.DataFrame(combined_rows), "learning_curve_points.csv",
                        subdir="learning_curves")
        # Mean curve across behaviors → recommended general-purpose clip count.
        if len(lc_results) >= 2:
            proj_ids = {lc.project_id for lc in lc_results}
            label = next(iter(proj_ids)) if len(proj_ids) == 1 else "all projects"
            avg = learning_curve.average_curve(lc_results, project_label=label)
            if avg is not None and avg.points:
                lc_dir = store.sub("learning_curves")
                for view in plots.LEARNING_CURVE_VIEWS:
                    # "0_AVERAGE" prefix sorts the mean curve first in each view.
                    plots.learning_curve_plot(avg, lc_dir / f"0_AVERAGE__{view}.png", view=view)
                store.write_csv(pd.DataFrame(_lc_points_rows(avg)),
                                "learning_curve_average.csv", subdir="learning_curves")
                plots.close_all()
        # Report shows the headline F1/PR-AUC view; the other views live alongside on disk.
        lc_imgs = sorted((store.sub("learning_curves")).glob("*__f1_prauc.png"))
        sections.append(("Learning curves (optimal clips)",
                         report.table_section(knee_df) + report.img_section(lc_imgs)))

    if abl_results:
        abl_dir = store.sub("ablation")
        # One chart per clip budget (low data first, full data last) → side-by-side compare.
        by_budget: dict[int, list] = {}
        for r in abl_results:
            by_budget.setdefault(r.clip_budget, []).append(r)
        ordered = sorted(by_budget.items(),
                         key=lambda kv: (kv[0] == subsample.ALL_CLIPS, kv[0]))
        budget_imgs = []
        for idx, (budget, group) in enumerate(ordered):
            blabel = ablation.budget_label(budget)
            title = "full data" if budget == subsample.ALL_CLIPS else f"{budget} clips"
            img = abl_dir / f"feature_impact__{idx}_{blabel}.png"
            plots.ablation_impact_plot(group, img, budget_title=title)
            budget_imgs.append(img)
        plots.close_all()
        abl_rows = []
        for r in abl_results:
            blabel = ablation.budget_label(r.clip_budget)
            for cfgname in r.order:
                is_base = cfgname == ablation.BASELINE_CONFIG
                row = {
                    "project": r.project_id,
                    "behavior": r.behavior_name,
                    "clip_budget": blabel,
                    "config": cfgname,
                    "label": r.labels.get(cfgname, cfgname),
                    "f1_mean": r.f1_means.get(cfgname, float("nan")),
                    "gain_over_baseline": 0.0 if is_base else r.gain.get(cfgname, float("nan")),
                    "gain_ci95": 0.0 if is_base else r.gain_ci.get(cfgname, float("nan")),
                    "gain_n_seeds": 0 if is_base else r.gain_n.get(cfgname, 0),
                    "gain_p_value": float("nan") if is_base
                                    else r.gain_p.get(cfgname, float("nan")),
                    "significant": "" if is_base else bool(r.is_significant(cfgname)),
                }
                # The per-seed F1 behind every mean/CI, so the paired test can be
                # re-run downstream (Prism, R, a reviewer) from the CSV alone —
                # a boolean `significant` column cannot be re-derived or re-plotted.
                for i, v in enumerate(r.f1_seeds.get(cfgname, []), start=1):
                    row[f"f1_seed{i}"] = v
                abl_rows.append(row)
        abl_df = pd.DataFrame(abl_rows)
        store.write_csv(abl_df, "ablation_results.csv", subdir="ablation")
        # The config descriptions are 7 unique sentences; inlining them on all ~900
        # rows bloats the CSV and puts quoted, comma-laden prose in a numeric table.
        # They belong in a lookup the figure legend is written from.
        desc_seen: dict[str, dict] = {}
        for r in abl_results:
            for cfgname in r.order:
                desc_seen.setdefault(cfgname, {
                    "config": cfgname,
                    "label": r.labels.get(cfgname, cfgname),
                    "description": r.descriptions.get(cfgname, ""),
                })
        store.write_csv(pd.DataFrame(list(desc_seen.values())),
                        "ablation_config_legend.csv", subdir="ablation")
        sections.append(("Feature / pipeline ablation impact (detection: behavior vs. rest)",
                         report.table_section(abl_df) + report.img_section(budget_imgs)))

    if disc_by_project:
        disc_dir = store.sub("discrimination")
        disc_imgs: list[Path] = []
        all_rows = []
        for pid, pair_results in disc_by_project.items():
            if not pair_results:
                continue
            tag = _tag(pid)
            # One matrix per add-on family — NOT just video. Defaulting the Δ panel to
            # video hid the actual result on object-based assays, where the ROI/context
            # family is what disambiguates the pairs and video does almost nothing.
            addons = [s for s in (pair_results[0].order if pair_results else [])
                      if s != discrimination.BASELINE_FEATURE_SET]
            for fs in addons:
                m_img = disc_dir / f"{tag}__separability_matrix__{fs}.png"
                if plots.discrimination_matrices(pair_results, m_img, feature_set=fs) is not None:
                    disc_imgs.append(m_img)
                er_mat = discrimination.error_reduction_matrix(pair_results, feature_set=fs)
                store.write_csv(er_mat.reset_index(names="behavior"),
                                f"{tag}__error_reduction__{fs}.csv", subdir="discrimination")
            g_img = disc_dir / f"{tag}__feature_gain_by_pair.png"
            if plots.discrimination_gain_plot(pair_results, g_img) is not None:
                disc_imgs.append(g_img)
            all_rows.append(discrimination.discrimination_rows(pair_results))
            base_mat = discrimination.separability_matrix(pair_results)
            store.write_csv(base_mat.reset_index(names="behavior"),
                            f"{tag}__separability_matrix.csv", subdir="discrimination")
        plots.close_all()
        disc_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        if not disc_df.empty:
            store.write_csv(disc_df, "discrimination_results.csv", subdir="discrimination")
        # Hardest-first ranking: the trained answer to "which behaviors does this
        # project actually conflate, and does any feature family fix them?"
        hard_df = pd.concat(
            [discrimination.confusable_pairs_table(pr) for pr in disc_by_project.values()],
            ignore_index=True,
        ) if disc_by_project else pd.DataFrame()
        if not hard_df.empty:
            store.write_csv(hard_df, "confusable_pairs.csv", subdir="discrimination")
        inner = ""
        if not hard_df.empty:
            inner += ("<h3>Hardest behavior pairs (lowest pose-only separability)</h3>"
                      + report.table_section(hard_df))
        inner += report.table_section(disc_df) + report.img_section(disc_imgs)
        sections.append(
            ("Behavior discrimination (can the features tell similar behaviors apart?)",
             inner)
        )

    if gen_results:
        plots.human_ceiling_plot(gen_results, store.sub("generalization") / "model_vs_human_kappa.png")
        plots.close_all()
        # Per project × behavior — the detail, kept so a pooled bar can be broken back down.
        gen_df = pd.DataFrame([
            {"project": r.project_id, "behavior": r.behavior_name,
             "f1": r.f1_mean, "cohen_kappa": r.kappa_mean,
             "human_ceiling_kappa": r.human_ceiling_kappa}
            for r in gen_results
        ])
        store.write_csv(gen_df, "agreement.csv", subdir="generalization")
        # Per (assay, behavior) with seed CI — the exact numbers behind the figure.
        pooled_df = plots.pool_generalization_by_behavior(gen_results)
        if not pooled_df.empty:
            store.write_csv(pooled_df, "agreement_pooled.csv", subdir="generalization")
        sections.append((
            "Generalization / human agreement",
            report.img_section([store.sub("generalization") / "model_vs_human_kappa.png"])
            + ("<h3>Per assay × behavior (with seed CI)</h3>"
               + report.table_section(pooled_df) if not pooled_df.empty else "")
            + "<h3>Per project × behavior</h3>" + report.table_section(gen_df)
        ))

        # ── Biological readout: time-budget & bout-count agreement ──
        # Reuses each generalization fit's retained held-out predictions, so it
        # adds no training. This is the "does the model recover the measure a
        # scorer would report" figure the manuscript leads its validation with.
        tb_results = [time_budget.run_time_budget(g.predictions) for g in gen_results]
        tb_results = [t for t in tb_results if t is not None]
        if tb_results:
            tb_dir = store.sub("time_budget")
            tb_imgs = []
            # Headline first: all behaviors on one panel (a reader should not have to
            # assemble N per-behavior files). "0_" prefixes sort them to the front.
            by_proj: dict[str, list] = {}
            for t in tb_results:
                if not t.error:
                    by_proj.setdefault(t.project_id, []).append(t)
            for pid, group in by_proj.items():
                for fn, stem in ((plots.time_budget_forest, "0_AGREEMENT_FOREST"),
                                 (plots.time_budget_grid, "0_PREVALENCE_GRID")):
                    img = tb_dir / f"{stem}__{_tag(pid)}.png"
                    if fn(group, img) is not None:
                        tb_imgs.append(img)
            # Per-behavior detail (outlier labels, full Bland-Altman) kept for drill-down.
            for t in tb_results:
                if t.error:
                    continue
                stem = f"{_tag(t.project_id)}__{_tag(t.behavior_name)}"
                img = tb_dir / f"{stem}.png"
                if plots.time_budget_plot(t, img) is not None:
                    tb_imgs.append(img)
            plots.close_all()
            tb_df = time_budget.time_budget_rows(tb_results)
            store.write_csv(tb_df, "time_budget_agreement.csv", subdir="time_budget")
            store.write_csv(time_budget.time_budget_points(tb_results),
                            "time_budget_points.csv", subdir="time_budget")
            sections.append(("Biological readout: time-budget & bout agreement",
                             report.table_section(tb_df) + report.img_section(tb_imgs)))

        # ── Probability calibration (reliability of the production model) ──
        cal_results = [calibration.run_calibration(g.predictions) for g in gen_results]
        cal_results = [c for c in cal_results if c is not None]
        if cal_results:
            cal_dir = store.sub("calibration")
            cal_imgs = []
            for c in cal_results:
                stem = f"{_tag(c.project_id)}__{_tag(c.behavior_name)}"
                img = cal_dir / f"{stem}.png"
                if plots.reliability_diagram(c, img) is not None:
                    cal_imgs.append(img)
            plots.close_all()
            cal_df = calibration.calibration_rows(cal_results)
            store.write_csv(cal_df, "calibration.csv", subdir="calibration")
            store.write_csv(calibration.reliability_points(cal_results),
                            "reliability_points.csv", subdir="calibration")
            sections.append(("Probability calibration (reliability)",
                             report.table_section(cal_df) + report.img_section(cal_imgs)))

    if al_results:
        al_dir = store.sub("active_learning")
        al_imgs = []
        al_rows = []
        for r in al_results:
            stem = f"{_tag(r.project_id)}__{_tag(r.behavior_name)}"
            img = al_dir / f"{stem}.png"
            plots.al_vs_random_plot(r, img)
            al_imgs.append(img)
            al_n = r.clips_to_target(r.al_points)
            rnd_n = r.clips_to_target(r.random_points)
            al_rows.append({
                "project": r.project_id, "behavior": r.behavior_name,
                "al_clips_to_95pct": al_n, "random_clips_to_95pct": rnd_n,
                "al_pos_discovered_end": (r.al_points[-1].n_pos_mean if r.al_points else float("nan")),
                "random_pos_discovered_end": (r.random_points[-1].n_pos_mean if r.random_points else float("nan")),
            })
        plots.close_all()
        store.write_csv(pd.DataFrame(al_rows), "al_vs_random_summary.csv", subdir="active_learning")
        # Per-step curves — the exact series drawn in the figure (F1 + positives
        # discovered vs. clips reviewed, for each arm), for paste into graphing tools.
        point_rows = [row for r in al_results for row in _al_points_rows(r)]
        store.write_csv(pd.DataFrame(point_rows), "al_vs_random_points.csv",
                        subdir="active_learning")
        sections.append(("Active learning vs. random selection",
                         report.table_section(pd.DataFrame(al_rows)) + report.img_section(al_imgs)))

    if rare_reviewed or rare_behavior or rare_quality:
        rd_dir = store.sub("rare_discovery")
        rd_imgs: list[Path] = []
        # Behaviour rarity first — it frames why the rest matters.
        for br in rare_behavior:
            img = rd_dir / f"0_behavior_rarity__{_tag(br.project_id)}.png"
            rare_discovery.plot_behavior_rarity(br, img)
            rd_imgs.append(img)
            br.per_session.to_csv(
                rd_dir / f"behavior_rarity__{_tag(br.project_id)}.csv", index=False)
        # Discovery curves + effort bars (reviewed pool, all four arms).
        rd_rows: list[dict] = []
        for r in rare_reviewed:
            stem = f"{_tag(r.project_id)}__{_tag(r.behavior_name)}"
            c_img = rd_dir / f"discovery_curve__{stem}.png"
            e_img = rd_dir / f"effort_to_n__{stem}.png"
            rare_discovery.plot_discovery_curve(r, c_img)
            rare_discovery.plot_effort_to_n(r, e_img)
            rd_imgs.extend([c_img, e_img])
            rd_rows.extend(rare_discovery.discovery_points_rows(r))
            for strat, cur in r.curves.items():
                for tgt, clips in cur.effort_to_n.items():
                    rd_rows.append({"project": r.project_id, "behavior": r.behavior_name,
                                    "pool": "reviewed", "strategy": cur.label(),
                                    "effort_target": tgt, "clips_to_target": clips})
        for r in rare_full:
            stem = f"{_tag(r.project_id)}__{_tag(r.behavior_name)}"
            f_img = rd_dir / f"discovery_curve_fullpool__{stem}.png"
            rare_discovery.plot_discovery_curve(r, f_img)
            rd_imgs.append(f_img)
            rd_rows.extend(rare_discovery.discovery_points_rows(r))
        # Effort-to-quality: F1 + PR-AUC trajectories and the clips-to-target bars.
        q_rows: list[dict] = []
        for r in rare_quality:
            stem = f"{_tag(r.project_id)}__{_tag(r.behavior_name)}"
            for metric in ("f1", "pr_auc"):
                q_img = rd_dir / f"quality_curve_{metric}__{stem}.png"
                rare_discovery.plot_quality_curve(r, q_img, metric=metric)
                rd_imgs.append(q_img)
            qe_img = rd_dir / f"effort_to_quality__{stem}.png"
            rare_discovery.plot_effort_to_quality(r, qe_img)
            rd_imgs.append(qe_img)
            q_rows.extend(rare_discovery.quality_points_rows(r))
            for strat, cur in r.curves.items():
                for target, clips in cur.effort.items():
                    q_rows.append({"project": r.project_id, "behavior": r.behavior_name,
                                   "pool": r.pool_label, "strategy": cur.label(),
                                   "quality_target": target, "clips_to_target": clips})
        rar_rows: list[dict] = []
        for r in rare_rarity:
            stem = f"{_tag(r.project_id)}__{_tag(r.behavior_name)}"
            img = rd_dir / f"rarity_scaling__{stem}.png"
            rare_discovery.plot_rarity_scaling(r, img)
            rd_imgs.append(img)
            rar_rows.extend(rare_discovery.rarity_rows(r))
        # ── combined (cross-project) panels ──
        # Only meaningful with ≥2 projects, where each contributes its own rarest
        # behaviour as one paired observation of the same four arms.  Prefixed 1_
        # so they sit right after the rarity context and ahead of the per-project
        # detail in the report and the GUI gallery.
        comb_rows: list[dict] = []
        if len({r.project_id for r in rare_reviewed}) >= 2:
            k = rare_discovery.combined_enrichment_budget(rare_reviewed)
            k = min(100, k) if k else None
            tgt = rare_discovery.combined_effort_target(rare_reviewed)
            c1 = rd_dir / "1_combined_discovery.png"
            c2 = rd_dir / "1_combined_enrichment.png"
            c3 = rd_dir / "1_combined_effort_saved.png"
            rare_discovery.plot_combined_discovery(rare_reviewed, c1, labels=proj_names)
            rare_discovery.plot_combined_enrichment(rare_reviewed, c2, k=k, labels=proj_names)
            rare_discovery.plot_combined_savings(rare_reviewed, c3, target=tgt,
                                                 labels=proj_names)
            rd_imgs.extend([c1, c2, c3])
            comb_rows = rare_discovery.combined_rows(rare_reviewed, k=k, target=tgt,
                                                     labels=proj_names)
            pc = rare_discovery.prism_combined_enrichment(rare_reviewed, k=k,
                                                          labels=proj_names)
            if not pc.empty:
                store.write_csv(pc.round(6), "prism_combined_enrichment.csv",
                                subdir="prism")
        if len({q.project_id for q in rare_quality}) >= 2:
            c4 = rd_dir / "1_combined_quality_saved.png"
            rare_discovery.plot_combined_quality_savings(rare_quality, c4,
                                                         labels=proj_names)
            rd_imgs.append(c4)
        plots.close_all()
        if comb_rows:
            store.write_csv(pd.DataFrame(comb_rows), "combined_across_projects.csv",
                            subdir="rare_discovery")
        if rare_target_rows:
            store.write_csv(pd.DataFrame(rare_target_rows), "hunted_targets.csv",
                            subdir="rare_discovery")
        if rd_rows:
            store.write_csv(pd.DataFrame(rd_rows), "discovery.csv", subdir="rare_discovery")
        if rar_rows:
            store.write_csv(pd.DataFrame(rar_rows), "rarity_scaling.csv",
                            subdir="rare_discovery")
        if q_rows:
            store.write_csv(pd.DataFrame(q_rows), "effort_to_quality.csv",
                            subdir="rare_discovery")
        # Prism-ready copies (one behaviour-rarity table, plus discovery/effort/
        # rarity/quality).  Quality results are matched to their discovery run by
        # behaviour so both land in the same Prism bundle.
        # With more than one project in the run every table is filed under a
        # per-project stem; a single-project run keeps the plain filenames.
        multi = len({r.project_id for r in
                     (rare_reviewed + rare_quality + rare_behavior)}) > 1

        def _prism_stem(res) -> str:
            return (f"{_tag(proj_names.get(res.project_id, res.project_id))}"
                    if multi else "")

        matched_quality = set()
        for r in rare_reviewed:
            q = next((x for x in rare_quality
                      if x.behavior_id == r.behavior_id
                      and x.project_id == r.project_id), None)
            if q is not None:
                matched_quality.add(id(q))
            rare_discovery.write_prism(
                store.run_dir, reviewed=r,
                # Matched on project AND behaviour: imported models reuse their
                # source project's behaviour ids, so id alone can collide.
                full=next((f for f in rare_full
                           if f.behavior_id == r.behavior_id
                           and f.project_id == r.project_id), None),
                rarity=next((x for x in rare_rarity
                             if x.behavior_id == r.behavior_id
                             and x.project_id == r.project_id), None),
                quality=q, stem=_prism_stem(r))
        # A quality run whose discovery arm was skipped still gets its tables.
        for q in rare_quality:
            if id(q) not in matched_quality:
                rare_discovery.write_prism(store.run_dir, quality=q,
                                           stem=_prism_stem(q))
        for br in rare_behavior:
            rare_discovery.write_prism(store.run_dir, behavior_rarity=br,
                                       stem=_prism_stem(br))
        rarity_note = ""
        # Say up front which behaviour each project's hunt was spent on — in
        # auto-target mode the reader never chose it, so the figures are
        # uninterpretable without it.
        if rare_target_rows:
            tdf = pd.DataFrame(rare_target_rows)[
                ["project", "hunted_behavior", "auto_selected", "rarity_measure",
                 "rarity_value", "rarity_rank", "n_behaviors_compared",
                 "rarer_behaviors_not_hunted"]]
            rarity_note += ("<p><b>Behaviour hunted per project</b> "
                            "(auto-selected = rarest by the cheap bout-detection "
                            "pass, before any model fitting):</p>"
                            + report.table_section(tdf))
        # Loudly flag any arm that could not run — an absent line on the figure
        # otherwise reads as "we tested it and it lost".
        for r in rare_reviewed:
            for strat, why in (r.disabled_strategies or {}).items():
                label = rare_discovery.STRATEGY_LABELS.get(strat, strat)
                rarity_note += (
                    f"<p style='color:#b03030;'><b>{label} was NOT tested</b> for "
                    f"{r.behavior_name}: {why} The figures below compare only the "
                    f"remaining strategies.</p>")
        for br in rare_behavior:
            if br.rarer_than_target:
                rarity_note += (f"<p><b>{br.target_name}</b> is rank {br.target_rank}"
                                f"/{br.n_behaviors} by {br.measure}; rarer still: "
                                f"{', '.join(br.rarer_than_target)} "
                                f"(Kruskal p={br.kruskal_p:.1e}, "
                                f"{br.target_name}&lt;rest p={br.target_vs_rest_p:.1e}).</p>")
        # Headline for the quality figures: labeling saved vs. random at the
        # strictest target each arm actually reached.
        for q in rare_quality:
            for strat in (rare_discovery.STRATEGY_ESSENCE, rare_discovery.STRATEGY_AL,
                          rare_discovery.STRATEGY_UMAP):
                cur = q.curves.get(strat)
                if not cur or not cur.effort:
                    continue
                target = max(cur.effort, key=lambda t: cur.effort[t])
                saved = q.savings_vs_random(strat, target)
                clips = q.clips_to(strat, target)
                if saved and clips:
                    rarity_note += (
                        f"<p><b>{q.behavior_name}</b> — {cur.label()} reached "
                        f"{target} in {clips:.0f} clips, {saved:.1f}× less labeling "
                        f"than random selection.</p>")
        sections.append(("Rare-behavior discovery (clip hunting)",
                         rarity_note + report.img_section(rd_imgs)))

    if bscape_data is not None:
        bs_dir = store.sub("behaviorscape")
        bs_imgs = plots.behaviorscape_figures(bscape_data, bs_dir)
        plots.close_all()
        # One tidy table per figure, so every plotted series can be re-drawn downstream.
        for df, fname, keep_index in (
            (bscape_data.to_long_df(), "behaviorscape_importance.csv", False),
            (bscape_data.modality_fraction_long_df(), "behaviorscape_modality_shares.csv", False),
            (behaviorscape.distinctiveness_df(bscape_stats),
             "behaviorscape_distinctiveness.csv", False),
            (bscape_data.similarity_matrix_df(), "behaviorscape_similarity_matrix.csv", True),
        ):
            if df is not None and not getattr(df, "empty", True):
                df.to_csv(bs_dir / fname, index=keep_index)
        sections.append((
            "Behaviorscape: which feature types drive which behaviors",
            report.table_section(behaviorscape.distinctiveness_df(bscape_stats))
            + report.img_section(bs_imgs)))

    if vv_results:
        vv_dir = store.sub("video_value")
        vv_df = video_value.results_to_frame(vv_results)
        store.write_csv(vv_df, "video_value.csv", subdir="video_value")
        vv_img = video_value.plot_video_value(vv_results, vv_dir / "video_value.png")
        plots.close_all()
        sections.append(("Video-feature value (paired with vs. without)",
                         report.table_section(vv_df) + report.img_section([vv_img])))

    if bench_results:
        bench_dir = store.sub("throughput")
        bench_df = benchmark.results_to_frame(bench_results)
        store.write_csv(bench_df, "benchmark.csv", subdir="throughput")
        bench_img = benchmark.plot_benchmark(bench_results, bench_dir / "benchmark.png")
        plots.close_all()
        sections.append(("Pipeline throughput",
                         report.table_section(bench_df) + report.img_section([bench_img])))

    # ── cross-project dashboard ──
    overview = cross_project.cross_project_overview(cells_df)
    acc_df = cross_project.accuracy_by_project(cells_df)
    speed_df = cross_project.training_speed_by_project(cells_df)
    if not acc_df.empty:
        plots.cross_project_bars(acc_df, "f1_mean", "f1_ci", "project_name",
                                 "Accuracy (F1) by project", "F1 (held-out)",
                                 store.sub("cross_project") / "accuracy_bars.png")
    if not speed_df.empty:
        plots.cross_project_bars(speed_df, "median_sec", None, "project_name",
                                 "Median training time by project", "seconds / run",
                                 store.sub("cross_project") / "speed_bars.png")
    plots.close_all()
    store.write_csv(acc_df, "dashboard.csv", subdir="cross_project")
    if not speed_df.empty:
        store.write_csv(speed_df, "training_speed.csv", subdir="cross_project")
    # Every publication summary metric per project (F1, MCC, balanced accuracy,
    # ROC-AUC, κ) — not just F1 — so reviewers see the imbalance-robust picture.
    pub_df = cross_project.publication_metrics_by_project(cells_df)
    if not pub_df.empty:
        store.write_csv(pub_df, "publication_metrics.csv", subdir="cross_project")
    # The headline meta-analysis figure. A bar per project is 3-6 near-identical
    # bars on a 0-1 axis and says nothing; the unit of evidence is the
    # (project × behavior) pair, with pooled estimates. "0_" sorts it first.
    beh_df = cross_project.accuracy_by_behavior(cells_df)
    if not beh_df.empty:
        ceiling = float("nan")
        if gen_results:
            ceilings = [r.human_ceiling_kappa for r in gen_results
                        if np.isfinite(r.human_ceiling_kappa)]
            ceiling = float(np.mean(ceilings)) if ceilings else float("nan")
        plots.cross_project_forest(
            beh_df, store.sub("cross_project") / "0_forest_by_behavior.png",
            metric="f1", ceiling=ceiling,
        )
        store.write_csv(beh_df, "accuracy_by_behavior.csv", subdir="cross_project")
        plots.close_all()
    cp_imgs = sorted(store.sub("cross_project").glob("*.png"))
    cp_inner = report.table_section(acc_df)
    if not pub_df.empty:
        cp_inner += ("<h3>Publication metrics by project</h3>"
                     + report.table_section(pub_df))
    sections.insert(0, ("Cross-project overview", cp_inner + report.img_section(cp_imgs)))

    # ── Prism-ready (pre-pivoted) copies ──
    # The tidy CSVs above are long-format; Prism cannot pivot on import, so it also
    # gets a directly-pasteable table per figure. See abel/validation/prism.py.
    try:
        al_pts = (pd.DataFrame([row for r in al_results for row in _al_points_rows(r)])
                  if al_results else None)
        cal_pts = calibration.reliability_points(cal_results) if cal_results else None
        tb_pts = time_budget.time_budget_points(tb_results) if tb_results else None
        bs_shares = (bscape_data.modality_fraction_long_df()
                     if bscape_data is not None else None)
        bs_imp = bscape_data.to_long_df() if bscape_data is not None else None
        prism_paths = prism.write_all(
            store.run_dir,
            gen_df=gen_df, ablation_df=abl_df, video_df=vv_df, bench_df=bench_df,
            al_df=al_pts, calibration_df=cal_pts, time_budget_df=tb_pts,
            bscape_shares_df=bs_shares, bscape_importance_df=bs_imp,
            discrimination_df=disc_df, accuracy_by_behavior_df=beh_df,
        )
        if prism_paths and progress_cb:
            progress_cb(f"Prism tables → {store.run_dir / 'prism'}", 0.99)
    except Exception as exc:  # noqa: BLE001 — a bad pivot must not sink the run
        if progress_cb:
            progress_cb(f"Prism export skipped: {type(exc).__name__}: {exc}", 0.99)

    # ── Meta summary tables (the one-figure manuscript spine) ──
    # Distil the per-behavior CSVs written above into 5 assay/behavior-level tables.
    try:
        summary_paths = meta_summary.write_all(
            store.run_dir, meta_summary.load_run_dir(store.run_dir))
        if summary_paths and progress_cb:
            progress_cb(f"Summary tables → {store.run_dir / 'summary'}", 0.995)
    except Exception as exc:  # noqa: BLE001 — a bad summary must not sink the run
        if progress_cb:
            progress_cb(f"Summary export skipped: {type(exc).__name__}: {exc}", 0.995)

    # ── Feature-role clustering (do extracted features play distinct roles?) ──
    # Needs both the behaviorscape modality profile and the ablation ΔF1; both are
    # assay-scoped, so they join 1:1 on "project · behavior".
    if bscape_data is not None and abl_df is not None and not abl_df.empty:
        try:
            fr_paths = feature_roles.run_feature_roles(
                bscape_data.modality_fraction_long_df(), abl_df,
                store.sub("feature_roles"), k=feature_roles.DEFAULT_K, scope="assay")
            if fr_paths and progress_cb:
                progress_cb(f"Feature roles → {store.run_dir / 'feature_roles'}", 0.997)
        except Exception as exc:  # noqa: BLE001 — must not sink the run
            if progress_cb:
                progress_cb(f"Feature-role export skipped: {type(exc).__name__}: {exc}",
                            0.997)

    # ── manifest + html ──
    manifest = RunManifest(
        run_id=store.run_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        analyses=config.analyses,
        projects=project_meta,
        config=config.to_dict(),
    ).to_dict()
    manifest_path = store.write_manifest(manifest)

    overview_disp = {"run": store.run_id, **{k: (round(v, 3) if isinstance(v, float) else v)
                                              for k, v in overview.items()}}
    report.build_html(store.run_id, overview_disp, sections, store.report_path)

    # ── consolidated findings + print-ready summary ──
    # The full report above is the exhaustive dump.  This is the thing a person
    # reads: what the run actually found, in words, with the caveats that stop a
    # number being over-read attached to the number itself.
    _emit_msg("Summarizing findings…")
    run_findings: list = []
    summary_html: Path | None = None
    try:
        run_findings = findings_mod.derive_findings(findings_mod.FindingsInput(
            cells=cells_df, overview=overview, project_meta=project_meta,
            lc_results=lc_results, abl_results=abl_results,
            disc_by_project=disc_by_project, gen_results=gen_results,
            tb_results=tb_results, cal_results=cal_results, al_results=al_results,
            vv_results=vv_results, bench_results=bench_results,
            bscape_stats=bscape_stats, bscape_data=bscape_data,
        ))
        store.write_csv(findings_mod.findings_frame(run_findings), "findings.csv")
        (store.run_dir / "FINDINGS.md").write_text(
            findings_mod.findings_markdown(store.run_id, run_findings), encoding="utf-8")
        summary_html = pdf_report.build_summary_html(
            store.run_id, store.run_dir, run_findings, overview, project_meta)
    except Exception as exc:  # noqa: BLE001 — a bad summary must not lose the results
        if progress_cb:
            progress_cb(f"Summary skipped: {type(exc).__name__}: {exc}", 0.99)

    if progress_cb:
        progress_cb(f"Done — results in {store.run_dir}", 1.0)

    return RunOutputs(
        run_dir=store.run_dir,
        cells=cells_df,
        report_path=store.report_path,
        manifest_path=manifest_path,
        summary_html=summary_html,
        findings=run_findings,
    )


def _fmt_dur(seconds: float) -> str:
    """Compact human duration for the ETA ("2m 30s", "1h 04m", "45s")."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def _tag(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name))[:32]


def _al_points_rows(al) -> list[dict]:
    """Tidy per-step rows for the active-learning vs. random curves (both arms)."""
    rows = []
    for arm, points in (("active_learning", al.al_points), ("random", al.random_points)):
        for p in points:
            rows.append({
                "project_id": al.project_id,
                "behavior_name": al.behavior_name,
                "strategy": arm,
                "n_clips_reviewed": p.n_clips,
                "n_seeds": p.n_seeds,
                "f1_mean": p.f1_mean, "f1_ci": p.f1_ci,
                "pr_auc_mean": p.pr_auc_mean, "pr_auc_ci": p.pr_auc_ci,
                "pos_discovered_mean": p.n_pos_mean,
            })
    return rows


def _lc_points_rows(lc) -> list[dict]:
    """Tidy per-point learning-curve rows (one row per clip-size step)."""
    rows = []
    for p in lc.points:
        rows.append({
            "project_id": lc.project_id,
            "behavior_name": lc.behavior_name,
            "n_clips_mean": round(p.n_clips_mean, 2),
            "n_seeds": p.n_seeds,
            "f1_mean": p.f1_mean, "f1_ci": p.f1_ci,
            "pr_auc_mean": p.pr_auc_mean, "pr_auc_ci": p.pr_auc_ci,
            "precision_mean": p.precision_mean, "recall_mean": p.recall_mean,
            "tp_mean": p.tp_mean, "fp_mean": p.fp_mean, "fn_mean": p.fn_mean,
            "kappa_mean": p.kappa_mean,
        })
    return rows
