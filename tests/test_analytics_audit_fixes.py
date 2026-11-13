"""Regressions for the defects found in the November data-accuracy audit.

Each test pins one number that was wrong: a comparison run on the wrong
factor, a duration one frame short, a mean biased by zero padding, a p-value
that could be zero, or two views of the same bouts that disagreed.
"""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from abel.services import group_stats
from abel.services.behavioral_motif_service import ngram_group_comparison
from abel.services.export_service import ExportService
from abel.services.group_stats_options import StatsOptions
from abel.temporal_refinement import bout_postprocess as bp
from abel.ui.tabs.behavior_analytics_tab import (
    BehaviorAnalyticsTab,
    _GraphsWidget,
    _smooth_trace,
)

# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------


def _mixed_table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sex in ("F", "M"):
        for i in range(6):
            base = rng.normal(10, 2)
            for day in ("D1", "D2", "D3"):
                rows.append({
                    "subject": f"{sex}{i}", "day": day, "sex": sex,
                    "value": base + rng.normal(0, 1),
                })
    return pd.DataFrame(rows)


def _pairs(res) -> set[tuple[str, str]]:
    return {(c.a, c.b) for c in res.comparisons}


def test_mixed_anova_compares_the_requested_factor_when_first_is_repeated():
    """'first' means first-factor levels within each second-factor level.

    With the first factor repeated, the factors trade places inside the mixed
    model and the comparison choice has to trade places with them; it used to
    compare the second factor instead.
    """
    df = _mixed_table().rename(columns={"day": "a", "sex": "b"})
    opts = StatsOptions(repeated_factor="Day", twoway_compare="first", posthoc="sidak")
    res = group_stats.two_factor(df, "Day", "Sex", ["D1", "D2", "D3"], ["F", "M"], opts)
    assert ("D1 | F", "D2 | F") in _pairs(res)
    assert ("D1 | F", "D1 | M") not in _pairs(res)

    opts = StatsOptions(repeated_factor="Day", twoway_compare="second", posthoc="sidak")
    res = group_stats.two_factor(df, "Day", "Sex", ["D1", "D2", "D3"], ["F", "M"], opts)
    assert _pairs(res) == {("D1 | F", "D1 | M"), ("D2 | F", "D2 | M"), ("D3 | F", "D3 | M")}


def test_mixed_anova_same_comparisons_whichever_factor_is_listed_first():
    df = _mixed_table()
    day_first = group_stats.two_factor(
        df.rename(columns={"day": "a", "sex": "b"}), "Day", "Sex",
        ["D1", "D2", "D3"], ["F", "M"],
        StatsOptions(repeated_factor="Day", twoway_compare="first", posthoc="sidak"),
    )
    sex_first = group_stats.two_factor(
        df.rename(columns={"sex": "a", "day": "b"}), "Sex", "Day",
        ["F", "M"], ["D1", "D2", "D3"],
        StatsOptions(repeated_factor="Day", twoway_compare="second", posthoc="sidak"),
    )
    # Same family of tests (Day levels within each Sex), so the same P values.
    assert sorted(round(c.p_adj, 10) for c in day_first.comparisons) == sorted(
        round(c.p_adj, 10) for c in sex_first.comparisons
    )


def test_ngram_permutation_p_is_never_zero():
    per = {f"a{i}": Counter({("x", "y"): 10}) for i in range(10)}
    per.update({f"b{i}": Counter({("y", "x"): 3}) for i in range(10)})
    grp = {**{f"a{i}": "A" for i in range(10)}, **{f"b{i}": "B" for i in range(10)}}
    res = ngram_group_comparison(per, grp, n_permutations=200)
    assert res
    for item in res:
        assert item["pval"] > 0
        assert min(item["pval_pairs"].values()) >= 1 / 201


def test_ngram_overall_p_is_corrected_for_the_number_of_pairs():
    rng = np.random.default_rng(1)
    per, grp = {}, {}
    for g in ("A", "B", "C"):
        for i in range(6):
            per[f"{g}{i}"] = Counter({("x", "y"): int(rng.integers(0, 6))})
            grp[f"{g}{i}"] = g
    item = ngram_group_comparison(per, grp, n_permutations=200)[0]
    assert len(item["pval_pairs"]) == 3
    assert item["pval"] == pytest.approx(min(1.0, 3 * min(item["pval_pairs"].values())))


# ----------------------------------------------------------------------
# Velocity smoothing
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 11, 29, 89])
def test_smoothing_does_not_lower_the_ends_of_a_trace(n):
    out = _smooth_trace(np.full(n, 10.0), 5)
    assert len(out) == n
    assert out == pytest.approx(np.full(n, 10.0))


def test_smoothing_matches_a_moving_average_away_from_the_ends():
    x = np.arange(40, dtype=float) ** 1.5
    ref = np.convolve(x, np.ones(5) / 5, mode="same")
    assert _smooth_trace(x, 5)[2:-2] == pytest.approx(ref[2:-2])


def test_smoothing_keeps_untracked_frames_out_of_their_neighbors():
    x = np.full(9, 4.0)
    x[4] = np.nan
    out = _smooth_trace(x, 5)
    assert np.isnan(out[4])
    assert out[[0, 1, 2, 3, 5, 6, 7, 8]] == pytest.approx(4.0)


# ----------------------------------------------------------------------
# Trace -> bouts: one implementation, shared defaults
# ----------------------------------------------------------------------


def test_trace_to_intervals_is_the_postprocess_pipeline():
    rng = np.random.default_rng(3)
    trace = np.clip(rng.normal(0.5, 0.3, 2000), 0, 1)
    smoothed = bp.smooth_probabilities(trace, "moving_average", 5)
    binary = bp.threshold_probabilities(smoothed, 0.6)
    binary = bp.merge_close_bouts(binary, 4)
    binary = bp.remove_short_bouts(binary, 8)
    assert bp.trace_to_intervals(
        trace, onset_threshold=0.6, min_bout_duration_frames=8, merge_gap_frames=4,
    ) == bp.binary_trace_to_intervals(binary)


def test_cleanup_frames_keep_their_duration_at_another_frame_rate():
    assert bp.scaled_cleanup_frames(12, 6, 5, None) == (12, 6, 5)
    assert bp.scaled_cleanup_frames(12, 6, 5, 2.0) == (24, 12, 10)
    # Smoothing switched off stays off; a zero merge gap stays zero.
    assert bp.scaled_cleanup_frames(12, 0, 1, 2.0) == (24, 0, 1)


def test_analytics_uses_the_review_defaults_without_a_settings_file(tmp_path):
    """No temporal_review_settings.json: Analytics, Review and export agree."""
    trace = np.zeros(400)
    trace[100:107] = 0.9   # 7 frames: shorter than the review default of 8
    trace[200:260] = 0.9
    path = tmp_path / "trace.parquet"
    pd.DataFrame({"frame": np.arange(400), "prob_b1": trace}).to_parquet(path)

    fake = SimpleNamespace(_project_root=tmp_path, _manifest=lambda: None)
    fake._load_temporal_review_thresholds = (
        lambda: BehaviorAnalyticsTab._load_temporal_review_thresholds(fake)
    )
    bouts = BehaviorAnalyticsTab._recompute_bouts_from_traces(
        fake, {"s1": str(path)}, {"b1": "Rear"},
    )["b1"]

    exporter = ExportService()
    exporter._project_root = tmp_path
    cfg = exporter._temporal_review_settings_for_token("b1")
    expected = bp.trace_to_intervals(
        trace, onset_threshold=cfg["onset_threshold"],
        min_bout_duration_frames=cfg["min_bout_duration_frames"],
        merge_gap_frames=cfg["merge_gap_frames"],
    )
    assert cfg == bp.review_default_settings()
    assert list(zip(bouts["start_frame"], bouts["end_frame"])) == expected
    assert len(expected) == 1  # the 7-frame run is below the default minimum


def test_partial_settings_file_falls_back_to_the_review_defaults(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "temporal_review_settings.json").write_text(
        json.dumps({"__all__": {"onset_threshold": 0.4}, "by_behavior": {"b1": {}}}),
        encoding="utf-8",
    )
    fake = SimpleNamespace(_project_root=tmp_path)
    out = BehaviorAnalyticsTab._load_temporal_review_thresholds(fake)
    assert out["b1"] == {
        "onset_threshold": 0.4,
        "min_bout_duration_frames": bp.REVIEW_DEFAULT_MIN_BOUT_FRAMES,
        "merge_gap_frames": bp.REVIEW_DEFAULT_MERGE_GAP_FRAMES,
    }


# ----------------------------------------------------------------------
# Graphs: time bins, ethogram, prechop
# ----------------------------------------------------------------------

FPS = 10.0


def _graphs(raw, metric="mean_bout_s", lo=None, hi=None, bin_s=60):
    stub = SimpleNamespace()
    for name in (
        "_bin_bouts", "_bin_grid", "_bin_origin_s", "_build_wide_binned_df",
        "_export_sessions", "_binned_session_grid", "_ethogram_bouts",
        "_ethogram_sessions", "_ethogram_roster", "_collect_graph_data",
    ):
        setattr(stub, name, getattr(_GraphsWidget, name).__get__(stub))
    stub._clip_bouts_to_range = _GraphsWidget._clip_bouts_to_range
    labels = {"s1": "M01", "s2": "M02"}
    stub._host = SimpleNamespace(
        _project_fps=lambda: FPS,
        _session_fps=lambda _sid: FPS,
        _raw_bouts=raw,
        _summary_rows=[],
        _selected_behavior_ids=lambda: {"b1"},
        _session_label_by_session=labels,
        _session_groups={},
        ordered_session_labels=lambda: ["M01", "M02"],
        _ordered_group_list=lambda names, *a: sorted({n for n in names if n}),
        _summary_tab=SimpleNamespace(_checked_subjects=lambda: {"M01", "M02"}),
        _behaviors=SimpleNamespace(behaviors=[
            SimpleNamespace(behavior_id="b1", name="Rear", color=None),
        ]),
        _facet_controls={},
        _session_factors={},
    )
    stub._time_bin_spin = SimpleNamespace(value=lambda: bin_s)
    stub._bin_from_range_chk = SimpleNamespace(isChecked=lambda: False)
    stub._get_data_range_seconds = lambda: (lo, hi)
    stub._get_metric = lambda: metric
    stub._get_mode = lambda: "individual"
    stub._get_style = lambda: "ethogram"
    stub._metric_label = lambda m: m
    stub._graph_rows = lambda: [{"session_label": "M01", "behavior": "Rear"}]
    stub._checked_groups = lambda: set()
    stub._checked_ethogram_sessions = lambda: set()
    stub._get_first_n_bouts = lambda: 0
    stub._get_until_behavior_id = lambda: None
    stub._cutoff_frames_for_until_behavior = lambda: {}
    stub._bin_cache = None
    stub._bin_cache_key = ()
    stub._ethogram_cache = None
    stub._ethogram_cache_key = None
    return stub


def _bouts(spans) -> dict[str, pd.DataFrame]:
    """spans: (session_id, first_frame, last_frame); the last frame is inclusive."""
    return {"b1": pd.DataFrame({
        "session_id": [s for s, _a, _b in spans],
        "behavior": "Rear",
        "start_frame": [a for _s, a, _b in spans],
        "end_frame": [b for _s, _a, b in spans],
    })}


def test_a_bout_across_a_bin_edge_is_one_bout_at_its_full_length():
    """A 10 s bout over the 60 s edge read as a 5 s bout in each bin."""
    w = _graphs(_bouts([("s1", 550, 649), ("s1", 100, 119)]))
    binned = w._bin_bouts().set_index("time_bin_s")
    # Bin 0-60: bouts of 2 s and 10 s start here.
    assert binned.loc[0, "n_bouts"] == 2
    assert binned.loc[0, "mean_bout_s"] == pytest.approx(6.0)
    # Bin 60-120: 5 s of behavior, but no bout starts, so no mean bout.
    assert binned.loc[60, "duration_s"] == pytest.approx(5.0)
    assert np.isnan(binned.loc[60, "mean_bout_s"])


def test_wide_mean_bout_export_blanks_empty_bins_and_reports_the_real_mean():
    w = _graphs(_bouts([("s1", 550, 649), ("s1", 100, 119), ("s2", 1300, 1339)]))
    wide = w._build_wide_binned_df().set_index("session_label")
    assert "total" not in wide.columns
    assert wide.loc["M01", "0s"] == pytest.approx(6.0)
    assert np.isnan(wide.loc["M01", "60s"]) and np.isnan(wide.loc["M01", "120s"])
    assert wide.loc["M01", "overall_mean"] == pytest.approx(6.0)
    assert np.isnan(wide.loc["M02", "0s"])
    assert wide.loc["M02", "overall_mean"] == pytest.approx(4.0)


def test_wide_count_export_still_zero_fills_and_totals():
    w = _graphs(_bouts([("s1", 550, 649), ("s2", 1300, 1339)]), metric="n_bouts")
    wide = w._build_wide_binned_df().set_index("session_label")
    assert wide.loc["M01", ["0s", "60s", "120s", "total"]].tolist() == [1, 0, 0, 1]
    assert wide.loc["M02", ["0s", "60s", "120s", "total"]].tolist() == [0, 0, 1, 1]


def test_ethogram_bout_durations_include_the_last_frame():
    """Frames 100..119 are 20 frames = 2.0 s, as the Summary reports it."""
    w = _graphs(_bouts([("s1", 100, 119), ("s1", 300, 300)]))
    bouts = w._ethogram_bouts()["b1"]["M01"]
    assert np.asarray(bouts) == pytest.approx(np.array([[10.0, 2.0], [30.0, 0.1]]))


def test_ethogram_csv_follows_the_data_range_like_the_plot():
    w = _graphs(_bouts([("s1", 100, 119), ("s1", 300, 349)]), lo=11.0, hi=32.0)
    table = w._collect_graph_data()
    assert table[["Start (s)", "End (s)", "Duration (s)"]].to_numpy().tolist() == [
        [11.0, 12.0, 1.0], [30.0, 32.0, 2.0],
    ]


def _prechop_host(rows, raw):
    host = SimpleNamespace(
        _summary_rows=rows, _raw_bouts=raw,
        _session_fps=lambda _sid: FPS,
        _analysis_prechop_for_session=lambda _sid: 600,
    )
    return host


def test_behavior_seen_only_before_the_prechop_counts_as_zero():
    """All of a session's bouts fall before test start: 0 bouts, not the old total."""
    rows = [
        {"session_id": "s1", "behavior_id": "b1", "n_bouts": 3.0,
         "time_spent_s": 9.0, "mean_bout_s": 3.0, "latency_s": 5.0},
        {"session_id": "s2", "behavior_id": "b1", "n_bouts": 1.0,
         "time_spent_s": 2.0, "mean_bout_s": 2.0, "latency_s": 70.0},
    ]
    # After the trim only s2 still has a bout (rebased to start at frame 100).
    raw = {"b1": pd.DataFrame({
        "session_id": ["s2"], "start_frame": [100], "end_frame": [119],
    })}
    host = _prechop_host(rows, raw)
    BehaviorAnalyticsTab._recompute_summary_stats_from_shifted_bouts(
        host, {("s1", "b1"), ("s2", "b1")},
    )
    s1, s2 = host._summary_rows
    assert (s1["n_bouts"], s1["time_spent_s"]) == (0.0, 0.0)
    assert np.isnan(s1["latency_s"])
    assert s2["time_spent_s"] == pytest.approx(2.0)
    assert s2["latency_s"] == pytest.approx(10.0)


def test_row_without_raw_bouts_keeps_its_values_and_shifts_latency():
    rows = [{"session_id": "s1", "behavior_id": "b9", "n_bouts": 2.0,
             "time_spent_s": 4.0, "mean_bout_s": 2.0, "latency_s": 80.0}]
    raw = {"b1": pd.DataFrame({
        "session_id": ["s2"], "start_frame": [100], "end_frame": [119],
    })}
    host = _prechop_host(rows, raw)
    BehaviorAnalyticsTab._recompute_summary_stats_from_shifted_bouts(host, set())
    assert host._summary_rows[0]["n_bouts"] == 2.0
    assert host._summary_rows[0]["latency_s"] == pytest.approx(20.0)


# ----------------------------------------------------------------------
# Distance, fps fallback, behaviogram
# ----------------------------------------------------------------------


def _distance_host(ppm=None):
    n = 3000
    pose = SimpleNamespace(
        centroid_x=np.arange(n, dtype=float), centroid_y=np.zeros(n), n_frames=n,
    )
    host = SimpleNamespace(
        _get_pose_for_session=lambda _sid: pose,
        _session_fps=lambda _sid: 30.0,
        _analysis_prechop_for_session=lambda _sid: 0,
        _pixels_per_mm_for_session=lambda _sid: ppm,
    )
    for name in ("_compute_session_distance", "_compute_session_distance_binned",
                 "_session_distance_in_range"):
        setattr(host, name, getattr(BehaviorAnalyticsTab, name).__get__(host))
    return host


def test_distance_in_a_data_range_is_that_part_of_the_path():
    host = _distance_host()
    whole = host._compute_session_distance("s1")
    assert host._session_distance_in_range("s1", None, None) == pytest.approx(whole)
    # 1 px per frame at 30 fps: 20 s to 50 s is 900 px.
    assert host._session_distance_in_range("s1", 20.0, 50.0) == pytest.approx(900.0, abs=6.0)
    # px -> mm -> cm when the session is calibrated.
    assert _distance_host(ppm=2.0)._session_distance_in_range(
        "s1", 20.0, 50.0) == pytest.approx(45.0, abs=0.3)


def test_distance_label_does_not_call_pixels_centimeters():
    rows = [{"session_id": s, "behavior_id": "__distance__"} for s in ("s1", "s2")]
    from abel.ui.tabs import behavior_analytics_tab as mod

    for r in rows:
        r["behavior_id"] = mod.DISTANCE_BEHAVIOR_ID
    ppm = {"s1": 2.0, "s2": None}
    host = SimpleNamespace(
        _summary_rows=rows, _pixels_per_mm_for_session=lambda sid: ppm[sid],
    )
    assert "px" in BehaviorAnalyticsTab.distance_unit(host)
    ppm["s2"] = 2.0
    assert BehaviorAnalyticsTab.distance_unit(host) == "cm"
    ppm["s1"] = ppm["s2"] = None
    assert BehaviorAnalyticsTab.distance_unit(host).startswith("px")


def test_project_fps_falls_back_to_the_project_source_fps(tmp_path):
    (tmp_path / "project.yaml").write_text(
        "feature_extraction:\n  source_fps: 25.0\n", encoding="utf-8",
    )
    host = SimpleNamespace(
        _fps_cache=None, _project_root=tmp_path,
        _manifest=lambda: SimpleNamespace(videos=[SimpleNamespace(fps=None)]),
    )
    assert BehaviorAnalyticsTab._project_fps(host) == pytest.approx(25.0)
    host = SimpleNamespace(
        _fps_cache=None, _project_root=tmp_path,
        _manifest=lambda: SimpleNamespace(videos=[SimpleNamespace(fps=60.0)]),
    )
    assert BehaviorAnalyticsTab._project_fps(host) == pytest.approx(60.0)


def test_behaviogram_keeps_a_subjects_sessions_on_separate_rows():
    ex = ExportService()
    ex._confirmed_intervals_by_session = lambda _c, _d: {
        "s_day1": {"Freeze": [(10, 50)]},
        "s_day2": {"Freeze": [(30, 90)]},
        "s_other": {"Freeze": [(5, 9)]},
    }
    ex._subject_by_session = lambda project_root=None: {
        "s_day1": "M1", "s_day2": "M1", "s_other": "M2",
    }
    ex._session_type_by_session = lambda project_root=None: {
        "s_day1": "Day1", "s_day2": "Day2", "s_other": "Day1",
    }
    ex._session_token_by_session = lambda project_root=None: {}
    out = ex.build_behaviogram([], [])
    assert set(out) == {"M1 Day1", "M1 Day2", "M2"}
    assert out["M1 Day1"]["behaviors"]["Freeze"] == [(10, 50)]
    assert out["M1 Day2"]["max_end_frame"] == 90


# ----------------------------------------------------------------------
# Clocks, HMM frame exports, Home tab metrics, pose features
# ----------------------------------------------------------------------


def test_session_end_is_on_the_analysis_clock():
    """A 10 min video with a 60 s prechop has 9 min of test, not 10."""
    host = SimpleNamespace(
        _session_fps=lambda _sid: 10.0, _session_end_s_overrides={},
        _get_pose_for_session=lambda _sid: SimpleNamespace(n_frames=6000),
        _analysis_prechop_for_session=lambda _sid: 600, _raw_bouts={},
    )
    w = SimpleNamespace(_host=host, _session_end_s_cache={})
    assert _GraphsWidget._session_analysis_end_s(w, "s1") == pytest.approx(540.0)

    # A merged project's raw-duration override is rebased the same way.
    host._session_end_s_overrides = {"s1": 600.0}
    w = SimpleNamespace(_host=host, _session_end_s_cache={})
    assert _GraphsWidget._session_analysis_end_s(w, "s1") == pytest.approx(540.0)


def test_hmm_state_frames_use_each_sessions_own_frame_rate():
    from abel.services.behavioral_motif_service import (
        state_bout_rows,
        state_bouts_to_frames,
        state_frame_matrix,
    )

    bouts = {
        "s30": [{"state": 0, "start_s": 10.0, "end_s": 12.0, "duration_s": 2.0, "n_bouts": 1}],
        "s60": [{"state": 0, "start_s": 10.0, "end_s": 12.0, "duration_s": 2.0, "n_bouts": 1}],
    }
    rates = {"s60": 60.0}
    frames = state_bouts_to_frames(bouts, 30.0, 1, fps_by_session=rates)
    assert frames["s30"]["HMM_State_0"] == [(300, 360)]
    assert frames["s60"]["HMM_State_0"] == [(600, 720)]

    mats = state_frame_matrix(bouts, 30.0, fps_by_session=rates)
    assert np.flatnonzero(mats["s60"] == 0)[[0, -1]].tolist() == [600, 720]
    rows = {r["session_id"]: r for r in state_bout_rows(bouts, 30.0, fps_by_session=rates)}
    assert (rows["s60"]["start_frame"], rows["s60"]["end_frame"]) == (600, 720)
    assert (rows["s30"]["start_frame"], rows["s30"]["end_frame"]) == (300, 360)


def _write_model(root, name, bid, metrics):
    mdir = root / "derived" / "models" / f"behavior_model_{name}"
    mdir.mkdir(parents=True)
    (mdir / "model_state.pkl").write_bytes(b"x")
    (mdir / "run_settings.json").write_text(json.dumps({"target_behavior": bid}), encoding="utf-8")
    (mdir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_home_tab_summary_is_held_out_and_target_class(tmp_path):
    from abel.services.model_performance_summary import held_out_summary, target_class_metrics

    groom = {
        "f1": 0.877, "precision": 0.882, "recall": 0.872, "pr_auc": 0.82,
        "confusion_matrix": [[56, 17], [15, 660]], "n_val_pos": 73,
        "holdout": True, "evaluated_on": "held_out",
    }
    rear = {
        "f1": 0.95, "precision": 0.95, "recall": 0.95, "pr_auc": 0.9,
        "confusion_matrix": [[500, 10], [10, 90]], "n_val_pos": 100,
        "holdout": True, "evaluated_on": "held_out",
    }
    _write_model(tmp_path, "Groom", "b_groom", groom)
    _write_model(tmp_path, "Rear", "b_rear", rear)
    # The old source: last behavior trained, scored on its own training labels.
    ev = tmp_path / "derived" / "evaluation"
    ev.mkdir(parents=True)
    (ev / "model_metrics.json").write_text(
        json.dumps({"frame_level": {"f1": 1.0}}), encoding="utf-8")

    target = target_class_metrics(groom)
    assert target["precision"] == pytest.approx(56 / 71)
    assert target["recall"] == pytest.approx(56 / 73)
    # The target row is found from n_val_pos whichever class index it has.
    assert target_class_metrics(rear)["recall"] == pytest.approx(0.9)

    out = held_out_summary(tmp_path, {"b_groom": "Groom", "b_rear": "Rear"})
    assert out["basis"] == "target" and out["n_models"] == 2
    assert out["lowest_behavior"] == "Groom"
    assert out["lowest_f1"] == pytest.approx(target["f1"])
    assert out["f1"] == pytest.approx((target["f1"] + 0.9) / 2)
    assert out["f1"] < 1.0


def test_home_tab_summary_falls_back_to_macro_and_says_so(tmp_path):
    from abel.services.model_performance_summary import held_out_summary

    _write_model(tmp_path, "Groom", "b_groom", {"f1": 0.8, "precision": 0.7, "recall": 0.9})
    out = held_out_summary(tmp_path)
    assert out["basis"] == "macro" and out["f1"] == pytest.approx(0.8)
    assert held_out_summary(tmp_path / "missing") is None


def test_late_entry_animal_keeps_its_oscillation_features():
    """NaN frames before an animal enters must not blank the whole session."""
    from abel.services.pose_processing_service import PoseData, PoseProcessingService

    n = 120
    t = np.arange(n, dtype=float)
    parts = ["nose", "paw_L", "paw_R", "tailbase"]
    x = pd.DataFrame({p: 10.0 + np.sin(t / 3.0 + i) for i, p in enumerate(parts)})
    y = pd.DataFrame({p: 5.0 + np.cos(t / 4.0 + i) for i, p in enumerate(parts)})
    x.iloc[:40] = np.nan
    y.iloc[:40] = np.nan
    pose = PoseData(
        x=x, y=y, likelihood=pd.DataFrame({p: np.ones(n) for p in parts}),
        body_parts=parts, centroid_x=x.mean(axis=1).to_numpy(),
        centroid_y=y.mean(axis=1).to_numpy(), n_frames=n,
    )
    feats = PoseProcessingService().compute_frame_pose_features(pose, 30.0, "a", "s", "v")
    for col in ("forepaw_oscillation_power", "nose_oscillation_power"):
        assert np.isfinite(feats[col].to_numpy()[45:]).all(), col
        assert np.isnan(feats[col].to_numpy()[:40]).all(), col


# ----------------------------------------------------------------------
# Density maps and Clip Mining
# ----------------------------------------------------------------------


def test_density_map_leaves_out_subjects_unticked_in_summary():
    """The map's points come from the same sessions its "n" counts."""
    from abel.ui.tabs.behavior_analytics_tab import _DensityAnalysisWidget as D

    def pose(x):
        return SimpleNamespace(
            n_frames=100, centroid_x=np.full(100, float(x)), centroid_y=np.full(100, float(x)),
        )

    poses = {"s1": pose(10), "s2": pose(500)}
    host = SimpleNamespace(
        _project_root="x", _session_label_by_session={"s1": "M1", "s2": "M2"},
        _session_groups={"M1": "Ctl", "M2": "Ctl"}, _session_factors={},
        _behaviors=SimpleNamespace(behaviors=[SimpleNamespace(behavior_id="b1")]),
        _raw_bouts={"b1": pd.DataFrame({
            "session_id": ["s1", "s2"], "start_frame": [0, 0], "end_frame": [9, 9],
        })},
        _unrebase_bout_df_to_video_frames=lambda df: df,
        _summary_tab=SimpleNamespace(_checked_subjects=lambda: {"M1"}),
    )
    w = SimpleNamespace(_host=host, _get_density_pose_for_session=lambda sid: poses[sid])
    w._get_session_ids_for_group = lambda g, f: D._get_session_ids_for_group(w, g, f)
    xs, _ys = D._collect_xy_for_group(w, "b1", "Ctl", None)["b1"]
    assert len(w._get_session_ids_for_group("Ctl", None)) == 1
    assert xs.size == 10 and (xs == 10).all()


def _clip_service(ppm_by_session, fps_by_session):
    from abel.services.clip_metrics_service import ClipMetricsService

    svc = ClipMetricsService()
    svc._manifest = SimpleNamespace(
        linked_sessions=[SimpleNamespace(session_id=s) for s in ppm_by_session],
    )
    svc._imports = SimpleNamespace(
        pixels_per_mm_for_session=lambda _m, sid: ppm_by_session.get(sid),
    )
    svc._fps = 30.0
    svc._fps_by_session = dict(fps_by_session)
    return svc


def test_clip_mining_uses_each_sessions_own_frame_rate():
    svc = _clip_service({"s30": 2.0, "s60": 2.0}, {"s60": 60.0})
    assert svc._fps_for("s30") == 30.0
    assert svc._fps_for("s60") == 60.0
    # 1 px per frame at 2 px/mm: 15 mm/s at 30 fps, 30 mm/s at 60 fps.
    x = np.arange(10, dtype=float)
    speed = svc._speed_mm_s(x, np.zeros(10), 2.0, svc._fps_for("s60"))
    assert speed == pytest.approx(np.full(9, 30.0))


def test_clip_mining_units_say_pixels_without_a_calibration():
    calibrated = _clip_service({"s1": 2.0}, {})
    assert calibrated.calibration_state() == "all"
    units = {m.id: m.unit for m in calibrated.available_metrics()}
    assert units["centroid_speed_mean"] == "mm/s"

    bare = _clip_service({"s1": None}, {})
    assert bare.calibration_state() == "none"
    units = {m.id: m.unit for m in bare.available_metrics()}
    assert units["centroid_speed_mean"] == "px/s"
    assert units["centroid_path_mm"] == "px"

    mixed = _clip_service({"s1": 2.0, "s2": None}, {})
    assert "px" in {m.id: m.unit for m in mixed.available_metrics()}["centroid_path_mm"]
