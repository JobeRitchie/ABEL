"""Variability for the learning-curve knee and F1 ceiling.

Both are read off the mean curve (a discrete argmin and a max), so they get a
bootstrap percentile interval rather than a standard error; see
:func:`abel.validation.analyses.learning_curve.bootstrap_knee_ci`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abel.validation import prism
from abel.validation.analyses import learning_curve as lcmod


def _unit_curve(f1_by_size: dict[int, float], degen_sizes: tuple[int, ...] = ()):
    """One seed's curve in bootstrap form: size -> (n_clips, f1, n_degen, n_seeds)."""
    return {size: (float(size), f1, int(size in degen_sizes), 1)
            for size, f1 in f1_by_size.items()}


def test_bootstrap_ci_brackets_the_point_estimate():
    # Five seeds that saturate at 100 clips with a little seed-to-seed jitter.
    rng = np.random.default_rng(0)
    sizes = [10, 25, 50, 100, 200]
    shape = {10: 0.30, 25: 0.55, 50: 0.75, 100: 0.90, 200: 0.91}
    units = [_unit_curve({s: shape[s] + rng.normal(0, 0.01) for s in sizes})
             for _ in range(5)]
    out = lcmod.bootstrap_knee_ci(units, n_reps=400, seed=1)

    assert out["boot_n_units"] == 5
    assert out["boot_unit"] == "seeds"
    assert out["knee_lo"] <= 100.0 <= out["knee_hi"]
    assert out["f1_max_lo"] <= out["f1_max_hi"]
    # The knee can only land on the schedule it was measured on.
    assert set(np.unique([out["knee_lo"], out["knee_hi"]])) <= set(float(s) for s in sizes)
    assert out["knee_undefined_frac"] == 0.0


def test_bootstrap_ci_widens_when_seeds_disagree():
    sizes = [10, 25, 50, 100, 200]
    agree = [_unit_curve({10: 0.3, 25: 0.6, 50: 0.9, 100: 0.91, 200: 0.91})
             for _ in range(5)]
    # Same curves except each seed saturates at a different budget.
    disagree = [_unit_curve({s: (0.9 if s >= knee else 0.3) for s in sizes})
                for knee in (10, 25, 50, 100, 200)]
    tight = lcmod.bootstrap_knee_ci(agree, n_reps=400, seed=1)
    wide = lcmod.bootstrap_knee_ci(disagree, n_reps=400, seed=1)
    assert tight["knee_hi"] - tight["knee_lo"] == 0.0
    assert wide["knee_hi"] - wide["knee_lo"] > 0.0


def test_bootstrap_needs_two_units_and_two_points():
    assert np.isnan(lcmod.bootstrap_knee_ci([])["knee_lo"])
    one = [_unit_curve({10: 0.3, 50: 0.9})]
    assert np.isnan(lcmod.bootstrap_knee_ci(one)["knee_lo"])
    # A unit with a single budget cannot define a knee and is dropped, not crashed on.
    stub = [_unit_curve({10: 0.3}), _unit_curve({10: 0.4})]
    assert lcmod.bootstrap_knee_ci(stub)["boot_n_units"] == 0


def test_degenerate_seeds_do_not_set_the_ceiling():
    # A collapsed fit scores a perfect target-class F1 at the smallest budget; the
    # bootstrap must drop it exactly as detect_knee does, or every replicate reports
    # "10 clips is enough" off a model that learned nothing.
    sizes = [10, 25, 50, 100]
    units = [_unit_curve({10: 1.0, 25: 0.5, 50: 0.8, 100: 0.82}, degen_sizes=(10,))
             for _ in range(5)]
    out = lcmod.bootstrap_knee_ci(units, n_reps=200, seed=3)
    assert out["f1_max_hi"] < 1.0
    assert out["knee_lo"] > 10.0
    assert max(sizes) >= out["knee_hi"]


def test_average_curve_bootstraps_over_behaviors():
    def _res(name, knee):
        pts = [lcmod.LearningCurvePoint(
            requested_size=s, n_clips_mean=float(s),
            f1_mean=(0.9 if s >= knee else 0.3), f1_ci=0.01,
            pr_auc_mean=0.5, pr_auc_ci=0.01, kappa_mean=0.4, n_seeds=5)
            for s in (10, 25, 50, 100, 200)]
        return lcmod.LearningCurveResult(project_id="P", behavior_id=name,
                                         behavior_name=name, points=pts)

    avg = lcmod.average_curve([_res("a", 25), _res("b", 100), _res("c", 200)])
    assert avg is not None
    assert avg.boot_unit == "behaviors"
    assert avg.boot_n_units == 3
    assert avg.knee_lo <= float(avg.knee_clips) <= avg.knee_hi


def test_prism_knee_table_carries_bounds_and_prism_deltas():
    df = pd.DataFrame([{
        "project_id": "EPM", "behavior_name": "Head Dip",
        "knee_clips": 300.0, "f1_max": 0.8739,
        "knee_lo": 200.0, "knee_hi": 400.0,
        "f1_max_lo": 0.83, "f1_max_hi": 0.90,
        "boot_n_units": 5, "boot_n_reps": 2000, "boot_unit": "seeds",
    }])
    out = prism.prism_learning_curve_knee(df)
    row = out.iloc[0]
    # Prism's asymmetric error format wants lengths from the mean, not endpoints.
    assert row["Knee -error"] == 100.0 and row["Knee +error"] == 100.0
    assert abs(row["Max F1 -error"] - 0.0439) < 1e-9
    assert row["Knee CI low"] == 200.0 and row["Knee CI high"] == 400.0
    assert row["N seeds"] == 5


def test_prism_knee_table_survives_a_run_without_intervals():
    # Runs predating the bootstrap (and figures re-rendered from saved means) have
    # no CI columns; the table must still export with blank error bars.
    df = pd.DataFrame([{"project_id": "OF", "behavior_name": "Walk",
                        "knee_clips": 150.0, "f1_max": 0.9589}])
    out = prism.prism_learning_curve_knee(df)
    assert out.iloc[0]["Knee clips"] == 150.0
    assert np.isnan(out.iloc[0]["Knee -error"])


def test_seed_unit_is_unpaired_and_behavior_unit_is_paired():
    # derive_seed keys on the budget, so repeat r at 50 clips and repeat r at 100
    # clips are unrelated fits: there is no seed to hold fixed down the schedule.
    # A behavior, by contrast, IS one object measured at every budget.
    units = [_unit_curve({10: 0.3, 25: 0.5, 50: 0.8, 100: 0.82}) for _ in range(4)]
    assert lcmod.bootstrap_knee_ci(units, unit="seeds")["boot_paired"] is False
    assert lcmod.bootstrap_knee_ci(units, unit="behaviors")["boot_paired"] is True


def test_paired_mode_handles_units_with_shorter_schedules():
    # Rare behaviors drop out of the high-clip budgets; a replicate that draws only
    # those must still produce a curve rather than a ragged aggregate or a crash.
    short = _unit_curve({10: 0.3, 25: 0.6})
    long_ = _unit_curve({10: 0.3, 25: 0.6, 50: 0.85, 100: 0.86})
    out = lcmod.bootstrap_knee_ci([short, long_, long_], unit="behaviors", n_reps=200)
    assert out["boot_n_units"] == 3
    assert np.isfinite(out["knee_lo"]) and np.isfinite(out["knee_hi"])
