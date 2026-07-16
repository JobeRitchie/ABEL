"""Tests for the one-button suite: findings, summary report, export bundle.

The load-bearing test here is ``test_no_deriver_silently_degrades``.  Every
finding deriver is wrapped in a try/except so a bad summary can never sink a run
that already has its results safely on disk — which is exactly what let a
property-vs-method mistake (``result.ece()`` on an ``@property``) ship as a
"Could not summarize this analysis" warning instead of a crash.  Every deriver is
therefore driven against REAL result dataclasses and asserted not to have
degraded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.validation import bundle, findings as fmod, pdf_report, subsample
from abel.validation.analyses.ablation import ALL_FEATURES_CONFIG, BASELINE_CONFIG, AblationResult
from abel.validation.analyses.al_curve import ALPoint, ALvsRandomResult
from abel.validation.analyses.calibration import CalibrationResult
from abel.validation.analyses.discrimination import (
    ALL_FEATURE_SET, BASELINE_FEATURE_SET, CONTEXT_FEATURE_SET, VIDEO_FEATURE_SET,
    PairResult,
)
from abel.validation.analyses.generalization import GeneralizationResult
from abel.validation.analyses.learning_curve import LearningCurvePoint, LearningCurveResult
from abel.validation.analyses.time_budget import TimeBudgetResult
from abel.validation.benchmark import STAGE_EXTRACT, STAGE_TRAIN, StageTiming
from abel.validation.metrics import CalibrationCurve
from abel.validation.runner import FULL_SUITE, PUBLICATION_SEEDS, publication_config
from abel.validation.video_value import VideoValueResult


# ── fixtures: real result objects, populated the way the analyses populate them ──

def _lc(name: str, knee: float, f1max: float) -> LearningCurveResult:
    pts = [LearningCurvePoint(requested_size=n, n_clips_mean=float(n), f1_mean=f1,
                              f1_ci=0.02, pr_auc_mean=f1, pr_auc_ci=0.02,
                              kappa_mean=f1 - 0.05, n_seeds=5)
           for n, f1 in ((10, 0.5), (50, 0.8), (100, f1max))]
    return LearningCurveResult(project_id="P1", behavior_id=name.lower(),
                               behavior_name=name, points=pts,
                               knee_clips=knee, f1_max=f1max)


def _ablation(behavior: str) -> AblationResult:
    order = [BASELINE_CONFIG, "add_video_features", "add_calibration", ALL_FEATURES_CONFIG]
    return AblationResult(
        project_id="P1", behavior_id=behavior.lower(), behavior_name=behavior,
        clip_budget=subsample.ALL_CLIPS,
        order=order,
        labels={BASELINE_CONFIG: "Baseline", "add_video_features": "+ Video features",
                "add_calibration": "+ Probability calibration",
                ALL_FEATURES_CONFIG: "All enhancements"},
        f1_means={BASELINE_CONFIG: 0.80, "add_video_features": 0.85,
                  "add_calibration": 0.80, ALL_FEATURES_CONFIG: 0.87},
        f1_seeds={c: [0.8, 0.81] for c in order},
        gain={"add_video_features": 0.05, "add_calibration": 0.000,
              ALL_FEATURES_CONFIG: 0.07},
        gain_ci={"add_video_features": 0.01, "add_calibration": 0.02,
                 ALL_FEATURES_CONFIG: 0.01},
        gain_n={"add_video_features": 5, "add_calibration": 5, ALL_FEATURES_CONFIG: 5},
        gain_p={"add_video_features": 0.001, "add_calibration": 0.9,
                ALL_FEATURES_CONFIG: 0.001},
        descriptions={c: f"desc {c}" for c in order},
    )


def _pair(a: str, b: str, base_auc: float, video_auc: float,
          ctx_auc: float, all_auc: float) -> PairResult:
    order = [BASELINE_FEATURE_SET, CONTEXT_FEATURE_SET, VIDEO_FEATURE_SET, ALL_FEATURE_SET]
    return PairResult(
        project_id="P1", behavior_a=a.lower(), behavior_b=b.lower(),
        name_a=a, name_b=b,
        n_train_a=50, n_train_b=50, n_hold_a=10, n_hold_b=10,
        order=order,
        auc={BASELINE_FEATURE_SET: base_auc, VIDEO_FEATURE_SET: video_auc,
             CONTEXT_FEATURE_SET: ctx_auc, ALL_FEATURE_SET: all_auc},
        gain={VIDEO_FEATURE_SET: video_auc - base_auc,
              CONTEXT_FEATURE_SET: ctx_auc - base_auc,
              ALL_FEATURE_SET: all_auc - base_auc},
        gain_ci={fs: 0.005 for fs in order},
        gain_n={fs: 5 for fs in order},
    )


def _generalization(name: str, kappa: float) -> GeneralizationResult:
    return GeneralizationResult(project_id="P1", behavior_id=name.lower(),
                                behavior_name=name, f1_mean=0.9, f1_ci=0.02,
                                kappa_mean=kappa, kappa_ci=0.03, n_seeds=5,
                                human_ceiling_kappa=0.92)


def _time_budget(name: str, ccc: float, bout_ccc: float = float("nan")) -> TimeBudgetResult:
    return TimeBudgetResult(
        project_id="P1", behavior_id=name.lower(), behavior_name=name,
        n_units=8, true_prevalence=np.array([0.1, 0.2]),
        pred_prevalence=np.array([0.12, 0.19]),
        prev_ccc=ccc, prev_pearson_r=0.9, prev_r2=0.8,
        bout_ccc=bout_ccc, median_coverage=0.015,
    )


def _calibration(name: str, ece: float) -> CalibrationResult:
    curve = CalibrationCurve(
        bin_confidence=np.array([0.1, 0.9]), bin_accuracy=np.array([0.1, 0.9]),
        bin_count=np.array([100, 100]), bin_center=np.array([0.1, 0.9]),
        n_bins=2, ece=ece, mce=ece * 2, brier=0.05, n=200,
    )
    return CalibrationResult(project_id="P1", behavior_id=name.lower(),
                             behavior_name=name, curve=curve)


def _al_point(n_clips: int, n_pos: float, f1: float) -> ALPoint:
    return ALPoint(n_clips=n_clips, n_pos_mean=n_pos, f1_mean=f1, f1_ci=0.02,
                   pr_auc_mean=f1, pr_auc_ci=0.02, n_seeds=5)


def _al(name: str) -> ALvsRandomResult:
    # Active learning discovers positives faster and plateaus sooner.
    al_pts = [_al_point(20, 10, 0.6), _al_point(50, 40, 0.9)]
    rnd_pts = [_al_point(20, 5, 0.5), _al_point(50, 15, 0.85)]
    return ALvsRandomResult(project_id="P1", behavior_id=name.lower(),
                            behavior_name=name, al_points=al_pts, random_points=rnd_pts)


def _video_value(name: str, gain: float, significant: bool) -> VideoValueResult:
    return VideoValueResult(
        project_id="P1", behavior_id=name.lower(), behavior_name=name, n_seeds=5,
        f1_no_video=0.80, f1_with_video=0.80 + gain, gain=gain, gain_ci95=0.01,
        significant=significant,
    )


def _bench() -> list[StageTiming]:
    return [
        StageTiming(project_id="P1", stage=STAGE_EXTRACT, seconds=60.0,
                    video_seconds=600.0, faster_than_realtime=10.0),
        StageTiming(project_id="P1", stage=STAGE_TRAIN, detail="Groom", seconds=12.0),
    ]


def _cells() -> pd.DataFrame:
    return pd.DataFrame([
        {"project_id": "P1", "project_name": "P1", "behavior_id": "groom",
         "behavior_name": "Groom", "analysis": "generalization", "f1": 0.9,
         "n_pos_train": 300, "error": ""},
        {"project_id": "P1", "project_name": "P1", "behavior_id": "rear",
         "behavior_name": "Rear", "analysis": "generalization", "f1": 0.85,
         "n_pos_train": 250, "error": ""},
    ])


@pytest.fixture
def full_input() -> fmod.FindingsInput:
    """One populated input touching every deriver."""
    return fmod.FindingsInput(
        cells=_cells(),
        overview={"n_projects": 1, "n_behaviors": 2, "f1_mean": 0.88,
                  "f1_min": 0.85, "f1_max": 0.90},
        project_meta=[{"project_id": "P1", "name": "P1", "behaviors": ["groom", "rear"]}],
        lc_results=[_lc("Groom", 100.0, 0.92), _lc("Rear", 250.0, 0.95)],
        abl_results=[_ablation("Groom"), _ablation("Rear")],
        disc_by_project={"P1": [
            # A hard pair with real headroom, and one the baseline already solves.
            _pair("Sniff", "Eat", 0.70, 0.75, 0.99, 0.99),
            _pair("Walk", "Freeze", 0.9995, 0.9996, 0.9997, 0.9998),
        ]},
        gen_results=[_generalization("Groom", 0.88), _generalization("Rear", 0.85)],
        tb_results=[_time_budget("Groom", 0.91), _time_budget("Rear", 0.80)],
        cal_results=[_calibration("Groom", 0.01), _calibration("Rear", 0.02)],
        al_results=[_al("Groom")],
        vv_results=[_video_value("Groom", 0.05, True),
                    _video_value("Rear", 0.001, False)],
        bench_results=_bench(),
    )


# ── the regression test that matters ────────────────────────────────────────

def test_no_deriver_silently_degrades(full_input):
    """Every deriver must produce real findings, not a 'Could not summarize' warning.

    The derivers swallow their own exceptions by design, so a broken one shows up
    as a bland warning rather than a failure. Assert none did.
    """
    items = fmod.derive_findings(full_input)
    degraded = [f for f in items if "Could not summarize" in f.headline]
    assert not degraded, (
        "deriver(s) failed: "
        + "; ".join(f"{f.analysis}: {f.detail}" for f in degraded))


def test_every_analysis_contributes_a_finding(full_input):
    items = fmod.derive_findings(full_input)
    got = {f.analysis for f in items}
    for expected in ("Overview", "Learning curves", "Ablation (detection)",
                     "Discrimination (pairwise)", "Generalization",
                     "Biological readout", "Calibration", "Active learning",
                     "Video features", "Throughput"):
        assert expected in got, f"no finding for {expected} (got {sorted(got)})"


def test_derive_findings_survives_an_empty_run():
    items = fmod.derive_findings(fmod.FindingsInput())
    assert not [f for f in items if "Could not summarize" in f.headline]


# ── the interpretation rules the findings exist to encode ───────────────────

def test_combined_config_is_not_reported_as_a_single_addition(full_input):
    """'All enhancements' is the union of the others and wins any ranking by
    construction — it must never be named the most valuable *single* addition."""
    items = fmod.derive_findings(full_input)
    single = next(f for f in items
                  if f.analysis == "Ablation (detection)"
                  and "single most valuable addition" in f.headline)
    assert "Video features" in f"{single.headline}"
    assert not single.headline.startswith("All enhancements")


def test_all_features_is_not_reported_as_the_rescuing_family(full_input):
    """Same rule for discrimination: 'all_features' is the ceiling, not an answer."""
    items = fmod.derive_findings(full_input)
    disc = [f for f in items if f.analysis == "Discrimination (pairwise)"]
    rescue = next(f for f in disc if "rescue" in f.headline)
    # Context (0.70 → 0.99) beats video (0.70 → 0.75); "All features" must not win.
    assert "Context" in rescue.headline
    assert "All features features" not in rescue.headline  # grammar guard


def test_hardest_pair_is_by_pose_only_auc(full_input):
    items = fmod.derive_findings(full_input)
    hardest = next(f for f in items if "hardest pair" in f.headline)
    assert "Sniff vs Eat" in hardest.headline


def test_ceiling_pairs_are_flagged_as_a_caveat(full_input):
    items = fmod.derive_findings(full_input)
    caveats = [f for f in items
               if f.analysis == "Discrimination (pairwise)" and f.kind == fmod.KIND_CAVEAT]
    assert caveats, "the pose-solved pair should raise the ceiling-effect caveat"
    assert "already solved" in caveats[0].headline


def test_prevalence_caveat_is_always_raised(full_input):
    """A time budget cannot be computed from sparse reviewed segments. If this
    caveat ever stops being emitted, the report starts making a false claim."""
    items = fmod.derive_findings(full_input)
    caveats = [f for f in items
               if f.analysis == "Biological readout" and f.kind == fmod.KIND_CAVEAT]
    assert any("PREVALENCE" in f.headline for f in caveats)


def test_uncomputable_bouts_are_flagged(full_input):
    items = fmod.derive_findings(full_input)
    assert any("Bout counts" in f.headline for f in items)


def test_behavior_count_excludes_discrimination_pairs(full_input):
    """Discrimination files its cells under a *pair* ('Groom vs Rear'), so counting
    distinct behavior_ids in the cell frame double-counts. The headline must report
    the behaviors the user actually selected."""
    cells = full_input.cells.copy()
    pair_row = cells.iloc[0].copy()
    pair_row["behavior_id"] = "groom|rear"
    pair_row["behavior_name"] = "Groom vs Rear"
    pair_row["analysis"] = "discrimination"
    full_input.cells = pd.concat([cells, pair_row.to_frame().T], ignore_index=True)
    full_input.overview = {**full_input.overview, "n_behaviors": 3}  # the wrong count

    items = fmod.derive_findings(full_input)
    headline = next(f for f in items if f.analysis == "Overview").headline
    # project_meta lists exactly 2 behaviors.
    assert "2 behaviors" in headline
    assert "3 behaviors" not in headline


def test_headline_f1_comes_from_the_production_config_not_all_cells(full_input):
    """ov['f1_mean'] averages over every fitted cell, including 10-clip learning-curve
    points and pose-only ablation baselines. Headlining it understates the model."""
    full_input.overview = {**full_input.overview, "f1_mean": 0.55, "n_cells": 400}
    items = fmod.derive_findings(full_input)
    headline = next(f for f in items if f.analysis == "Overview").headline
    # Generalization F1 is 0.9 for both behaviors; the pooled-cell 0.55 must not lead.
    assert "0.900" in headline
    assert "0.550" not in headline


def test_sparse_labeling_raises_a_warning():
    cells = pd.DataFrame([
        {"project_id": "P1", "project_name": "P1", "behavior_id": "attempt",
         "behavior_name": "Attempt", "analysis": "generalization", "f1": 0.49,
         "n_pos_train": 17, "error": ""},
    ])
    items = fmod.derive_findings(fmod.FindingsInput(cells=cells, overview={
        "n_projects": 1, "n_behaviors": 1, "f1_mean": 0.49}))
    warnings = [f for f in items if f.kind == fmod.KIND_WARNING]
    assert any("sparsely labeled" in f.headline for f in warnings)


# ── outputs ─────────────────────────────────────────────────────────────────

def test_findings_frame_and_markdown(full_input):
    items = fmod.derive_findings(full_input)
    df = fmod.findings_frame(items)
    assert list(df.columns) == ["analysis", "kind", "finding", "detail"]
    assert len(df) == len(items)
    md = fmod.findings_markdown("run_x", items)
    assert "# ABEL Validation — Key Findings" in md
    assert "run_x" in md


def test_build_summary_html_only_includes_analyses_that_ran(tmp_path):
    """A section with no findings, no figure and no table must not appear at all —
    an empty 'Throughput' heading in the PDF reads as a failed analysis."""
    items = [fmod.Finding("Generalization", "Agreement is substantial.", "κ = 0.8")]
    html_path = pdf_report.build_summary_html(
        "run_x", tmp_path, items, {"n_projects": 1},
        [{"name": "P1", "behaviors": ["a"]}])
    text = html_path.read_text(encoding="utf-8")
    assert "Generalization" in text
    assert "Pipeline throughput" not in text
    assert "Agreement is substantial." in text


def test_summary_html_escapes_content(tmp_path):
    items = [fmod.Finding("Generalization", "A <script>alert(1)</script> B", "x & y")]
    text = pdf_report.build_summary_html(
        "run_x", tmp_path, items, {}, []).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text


def test_headline_figures_follow_the_report_spec(tmp_path):
    (tmp_path / "cross_project").mkdir()
    (tmp_path / "cross_project" / "0_forest_by_behavior.png").write_bytes(b"x")
    (tmp_path / "generalization").mkdir()
    (tmp_path / "generalization" / "model_vs_human_kappa.png").write_bytes(b"x")
    (tmp_path / "generalization" / "not_a_headline.png").write_bytes(b"x")
    figs = [p.name for p in pdf_report.headline_figures(tmp_path)]
    assert figs == ["0_forest_by_behavior.png", "model_vs_human_kappa.png"]


def test_export_bundle_flattens_and_prefixes(tmp_path):
    run = tmp_path / "run_1"
    (run / "ablation").mkdir(parents=True)
    (run / "discrimination").mkdir()
    (run / "arrays").mkdir()
    (run / "ablation" / "results.csv").write_text("a,b\n1,2\n")
    (run / "ablation" / "impact.png").write_bytes(b"png")
    (run / "discrimination" / "results.csv").write_text("c,d\n3,4\n")
    (run / "arrays" / "intermediate.csv").write_text("junk\n")   # must be skipped
    (run / "run_manifest.json").write_text("{}")
    (run / "FINDINGS.md").write_text("# findings")

    res = bundle.export_bundle(run, tmp_path / "out")

    figs = sorted(p.name for p in (res.dest / "figures").iterdir())
    data = sorted(p.name for p in (res.dest / "data").iterdir())
    assert figs == ["ablation__impact.png"]
    # Same basename from two analyses must not collide.
    assert data == ["ablation__results.csv", "discrimination__results.csv"]
    assert "arrays__intermediate.csv" not in data
    assert (res.dest / "run_manifest.json").exists()
    assert (res.dest / "FINDINGS.md").exists()
    assert res.n_figures == 1 and res.n_tables == 2
    index = pd.read_csv(res.index_path)
    assert set(index["kind"]) == {"figure", "data", "report"}


# ── the preset ──────────────────────────────────────────────────────────────

def test_publication_preset_covers_the_whole_suite():
    cfg = publication_config()
    assert cfg.analyses == FULL_SUITE


def test_publication_preset_uses_five_seeds_everywhere():
    """At 3 seeds the 95% t-multiplier is 4.30 and real effects miss significance.
    Every analysis whose headline is a *difference* must not run under-powered."""
    cfg = publication_config()
    for attr in ("n_seeds_lc", "n_seeds_ablation", "n_seeds_generalization",
                 "n_seeds_discrimination", "n_seeds_al", "n_seeds_video_value"):
        assert getattr(cfg, attr) == PUBLICATION_SEEDS, attr


def test_publication_preset_excludes_dense_inference():
    """Dense inference rewrites the real project's traces — it must never be part
    of an unattended 'run everything' click."""
    assert "infer" not in publication_config().throughput_stages


def test_publication_preset_runs_ablation_at_a_low_budget_too():
    """Regularizers pay off in the low-data regime and vanish at full data; a
    full-data-only ablation reports 'no effect' for features that do help."""
    budgets = publication_config().ablation_budgets
    assert subsample.ALL_CLIPS in budgets
    assert any(b != subsample.ALL_CLIPS for b in budgets)


def test_preset_overrides_are_honoured():
    cfg = publication_config(output_root="/tmp/x", holdout_seed=7, min_confidence=0.5)
    assert cfg.holdout_seed == 7
    assert cfg.min_confidence == 0.5
    assert cfg.output_root == "/tmp/x"
