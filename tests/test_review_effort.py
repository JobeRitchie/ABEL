"""Tests for the human review-effort analysis (labeling-time ledger).

Covers the gap classification, the reviewer-channel split, the per-project and
pooled statistics, the volume denominators and the tidy/Prism exporters, against
synthetic decision logs written to a temp project — no real project, no training.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.validation import prism
from abel.validation.analyses import review_effort as re
from abel.validation.datamodel import ProjectRef

BASE = datetime(2026, 3, 1, 9, 0, 0)


def _project(root: Path, name: str = "P1", fps: float = 30.0) -> ProjectRef:
    return ProjectRef(project_id=name, name=name, root=Path(root), fps=fps,
                      behavior_names={"b1": "Rear", "no_behavior": "No Behavior"})


def _write_decisions(root: Path, rows: list[dict]) -> None:
    path = re.decisions_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"decisions": rows}), encoding="utf-8")


def _decision(offset_sec: float, *, reviewer: str = "reviewer",
              clip: str | None = None, start: int = 0, span: int = 15) -> dict:
    """One decision `offset_sec` after BASE, with real frame bounds."""
    return {
        "decision_id": f"d{offset_sec}",
        "clip_id": clip or f"rand_session_a_{start}_{start + span - 1}",
        "reviewer": reviewer,
        "timestamp": (BASE + timedelta(seconds=offset_sec)).isoformat(),
        "old_status": "unscored",
        "new_status": "reviewed",
        "decision": "accept",
        "behavior_label": "b1",
        "adjusted_start_frame": start,
        "adjusted_end_frame": start + span - 1,
    }


# ── gap classification ─────────────────────────────────────────────────────


def test_classify_gaps_splits_batch_active_and_break():
    stamps = [BASE + timedelta(seconds=s) for s in (0, 1.0, 1.01, 3.0, 400.0, 402.0)]
    timed, n_batch, n_breaks = re.classify_gaps(stamps)
    # 1.0 active, 0.01 batch, 1.99 active, 397 break, 2.0 active
    assert n_batch == 1
    assert n_breaks == 1
    np.testing.assert_allclose(sorted(timed), [1.0, 1.99, 2.0])


def test_classify_gaps_honours_custom_thresholds():
    stamps = [BASE + timedelta(seconds=s) for s in (0, 0.5, 10.0)]
    timed, n_batch, n_breaks = re.classify_gaps(stamps, break_sec=5.0, batch_sec=1.0)
    assert n_batch == 1 and n_breaks == 1        # 0.5 s is bulk, 9.5 s is a break
    assert timed.size == 0


def test_classify_gaps_on_fewer_than_two_stamps_is_empty():
    timed, n_batch, n_breaks = re.classify_gaps([BASE])
    assert timed.size == 0 and n_batch == 0 and n_breaks == 0


# ── reviewer channels ──────────────────────────────────────────────────────


def test_channel_separates_import_temporal_and_clip_review():
    assert re._channel({"reviewer": "imported:CIE_NSF"}) == "imported"
    assert re._channel({"reviewer": "temporal_feedback"}) == "temporal_feedback"
    assert re._channel({"reviewer": "reviewer"}) == "clip_review"
    assert re._channel({}) == "clip_review"


def test_imported_and_temporal_rows_are_counted_but_never_timed(tmp_path):
    """Only the clip-review channel may contribute to the per-clip rate.

    Both other channels write near-simultaneous bursts, so timing them would
    collapse the measured seconds-per-clip toward zero.
    """
    rows = [_decision(i * 2.0) for i in range(5)]
    rows += [_decision(100 + i * 0.001, reviewer="temporal_feedback",
                       clip=f"seg_feedback_a_{i}_{i}") for i in range(40)]
    rows += [_decision(200 + i * 0.001, reviewer="imported:Other",
                       clip=f"imp_{i}") for i in range(30)]
    _write_decisions(tmp_path, rows)

    result = re.measure_project(_project(tmp_path))
    assert not result.error
    assert result.n_decisions_total == 75
    assert result.n_clip_review == 5
    assert result.n_temporal_feedback == 40
    assert result.n_imported == 30
    assert result.n_timed == 4                    # four gaps between five clips
    assert result.median_sec == pytest.approx(2.0)


# ── per-project measurement ────────────────────────────────────────────────


def test_measure_project_reports_rate_hours_and_footage(tmp_path):
    # 10 clips, 3 s apart, 15 frames each at 30 fps => 0.5 s of footage per clip.
    rows = [_decision(i * 3.0, start=i * 100) for i in range(10)]
    _write_decisions(tmp_path, rows)

    result = re.measure_project(_project(tmp_path))
    assert result.n_timed == 9
    assert result.median_sec == pytest.approx(3.0)
    assert result.active_hours == pytest.approx(27.0 / 3600.0)
    assert result.clips_per_hour == pytest.approx(9 / (27.0 / 3600.0))
    assert result.footage_reviewed_hours == pytest.approx(10 * 0.5 / 3600.0)
    assert result.first_decision.startswith("2026-03-01T09:00:00")


def test_adjusted_hours_add_back_one_median_clip_per_sitting(tmp_path):
    """Each sitting's first clip has no measurable gap and must be added back."""
    rows = [_decision(t) for t in (0, 2.0, 4.0, 1000.0, 1002.0)]
    _write_decisions(tmp_path, rows)

    result = re.measure_project(_project(tmp_path))
    assert result.n_breaks == 1                   # two sittings
    assert result.active_hours == pytest.approx(6.0 / 3600.0)   # 2 + 2 + 2
    # + 2 sittings x the 2.0 s median
    assert result.active_hours_adjusted == pytest.approx(10.0 / 3600.0)
    assert result.active_hours_adjusted > result.active_hours


def test_missing_decisions_file_is_an_error_not_a_crash(tmp_path):
    result = re.measure_project(_project(tmp_path))
    assert result.error
    assert result.n_timed == 0


def test_single_decision_cannot_be_timed(tmp_path):
    _write_decisions(tmp_path, [_decision(0.0)])
    result = re.measure_project(_project(tmp_path))
    assert "at least 2" in result.error
    assert result.n_clip_review == 1


def test_all_bulk_actions_yields_no_timed_gaps(tmp_path):
    _write_decisions(tmp_path, [_decision(i * 0.001) for i in range(50)])
    result = re.measure_project(_project(tmp_path))
    assert result.error
    assert result.n_batch == 49 and result.n_timed == 0


def test_bare_list_decision_file_is_accepted(tmp_path):
    path = re.decisions_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_decision(0.0), _decision(2.0)]), encoding="utf-8")
    assert re.measure_project(_project(tmp_path)).n_timed == 1


def test_unparseable_timestamps_are_dropped_not_fatal(tmp_path):
    rows = [_decision(0.0), _decision(2.0)]
    rows.append({**_decision(4.0), "timestamp": "not-a-date"})
    rows.append({**_decision(6.0), "timestamp": None})
    _write_decisions(tmp_path, rows)
    result = re.measure_project(_project(tmp_path))
    assert result.n_timed == 1
    assert result.median_sec == pytest.approx(2.0)


def test_out_of_order_rows_are_sorted_before_gapping(tmp_path):
    """The file is not guaranteed ordered; unsorted rows would give negative gaps."""
    _write_decisions(tmp_path, [_decision(6.0), _decision(0.0), _decision(3.0)])
    result = re.measure_project(_project(tmp_path))
    assert result.n_timed == 2
    assert result.median_sec == pytest.approx(3.0)
    assert (result.timed_sec > 0).all()


# ── manual-scoring anchor ──────────────────────────────────────────────────


def test_saving_factor_compares_review_hours_to_full_manual_passes():
    result = re.ReviewEffortResult(project_id="P", active_hours=2.0, video_hours=40.0)
    assert result.manual_hours(1.0) == pytest.approx(40.0)
    assert result.manual_hours(3.0) == pytest.approx(120.0)
    assert result.saving_factor(1.0) == pytest.approx(20.0)
    assert result.saving_factor(3.0) == pytest.approx(60.0)


def test_saving_factor_is_nan_without_video_hours():
    result = re.ReviewEffortResult(project_id="P", active_hours=2.0)
    assert not np.isfinite(result.saving_factor(1.0))


# ── pooling ────────────────────────────────────────────────────────────────


def _result(pid: str, gaps: list[float], video_hours: float) -> re.ReviewEffortResult:
    arr = np.asarray(gaps, dtype=float)
    return re.ReviewEffortResult(
        project_id=pid, project_name=pid, n_clip_review=len(gaps) + 1,
        n_timed=arr.size, median_sec=float(np.median(arr)),
        active_hours=float(arr.sum() / 3600.0),
        active_hours_adjusted=float((arr.sum() + np.median(arr)) / 3600.0),
        video_hours=video_hours, footage_reviewed_hours=0.1, timed_sec=arr)


def test_pooled_median_weights_by_clip_not_by_project():
    """A project contributing 10x the clips must move the pooled median 10x as much."""
    big = _result("big", [1.0] * 100, 10.0)
    small = _result("small", [9.0] * 10, 10.0)
    pooled = re.pooled_summary([big, small])
    assert pooled["n_clips_timed"] == 110
    assert pooled["median_sec"] == pytest.approx(1.0)      # not the 5.0 of the medians
    assert pooled["active_hours"] == pytest.approx((100 * 1.0 + 10 * 9.0) / 3600.0)


def test_pooled_summary_carries_manual_anchors_and_volume():
    pooled = re.pooled_summary([_result("a", [2.0] * 1800, 20.0),
                                _result("b", [2.0] * 1800, 20.0)])
    assert pooled["video_hours"] == pytest.approx(40.0)
    assert pooled["active_hours"] == pytest.approx(2.0)    # 7200 s
    assert pooled["manual_hours_1x"] == pytest.approx(40.0)
    assert pooled["manual_hours_3x"] == pytest.approx(120.0)
    assert pooled["saving_factor_1x"] == pytest.approx(20.0)
    assert pooled["review_min_per_video_hour"] == pytest.approx(3.0)


def test_volume_ratios_pool_only_over_projects_with_measurable_video():
    """Both sides of every per-video-hour ratio must cover the same projects.

    Dividing all projects' review hours by only some projects' footage would
    inflate the rate by exactly the share of projects missing a segment pool.
    """
    measured = _result("has_video", [2.0] * 1800, 20.0)      # 1 h review, 20 h video
    unmeasured = _result("no_video", [2.0] * 1800, float("nan"))
    pooled = re.pooled_summary([measured, unmeasured])

    assert pooled["n_projects"] == 2
    assert pooled["n_projects_with_video"] == 1
    assert pooled["active_hours"] == pytest.approx(2.0)      # both projects
    assert pooled["video_active_hours"] == pytest.approx(1.0)  # only the measured one
    assert pooled["video_hours"] == pytest.approx(20.0)
    # 1 h of review against 20 h of footage — not 2 h, which would read as 2x worse.
    assert pooled["review_hours_per_video_hour"] == pytest.approx(0.05)
    assert pooled["saving_factor_1x"] == pytest.approx(20.0)


def test_volume_ratios_are_absent_when_no_project_has_video():
    pooled = re.pooled_summary([_result("a", [1.0] * 10, float("nan"))])
    assert pooled["n_projects_with_video"] == 0
    assert "review_hours_per_video_hour" not in pooled
    assert "saving_factor_1x" not in pooled
    assert pooled["active_hours"] > 0          # the effort itself is still reported


def test_findings_flag_a_partial_video_subset():
    from abel.validation import findings as findings_mod

    out = findings_mod.derive_findings(findings_mod.FindingsInput(
        effort_results=[_result("has_video", [2.0] * 1800, 20.0),
                        _result("no_video", [2.0] * 1800, float("nan"))]))
    detail = " ".join(f.detail for f in out if f.analysis == "Review effort")
    assert "1 of 2 projects" in detail


def test_pooled_summary_ignores_failed_projects():
    failed = re.ReviewEffortResult(project_id="bad", error="no review decisions on disk")
    pooled = re.pooled_summary([_result("a", [1.0] * 10, 5.0), failed])
    assert pooled["n_projects"] == 1


def test_pooled_summary_is_empty_when_nothing_measured():
    assert re.pooled_summary([re.ReviewEffortResult(project_id="x", error="nope")]) == {}
    assert re.pooled_to_frame([]).empty


# ── exports ────────────────────────────────────────────────────────────────


def test_results_to_frame_keeps_failed_projects_with_their_error():
    df = re.results_to_frame([_result("ok", [1.0] * 5, 2.0),
                              re.ReviewEffortResult(project_id="bad", error="boom")])
    assert list(df["project"]) == ["ok", "bad"]
    assert df.loc[1, "error"] == "boom"
    assert df.loc[1, "n_clips_timed"] == 0
    for passes in re.MANUAL_PASSES:
        assert f"manual_hours_{passes:g}x" in df.columns
        assert f"saving_vs_manual_{passes:g}x" in df.columns


def test_summary_text_reports_the_measured_rate_against_the_assumption():
    text = re.summary_text([_result("a", [2.0] * 100, 10.0)])
    assert "s per clip" in text
    assert f"{re.ASSUMED_SEC_PER_CLIP:.1f} s/clip" in text


def test_summary_text_explains_when_nothing_was_measurable():
    text = re.summary_text([re.ReviewEffortResult(project_id="x", error="no decisions")])
    assert "no decisions" in text


def test_prism_review_effort_builds_both_tables():
    df = re.results_to_frame([_result("a", [1.0] * 10, 4.0),
                              _result("b", [2.0] * 10, 6.0)])
    tables = prism.prism_review_effort(df)
    assert set(tables) == {"per_clip", "hours"}
    assert list(tables["per_clip"]["Project"]) == ["a", "b"]
    assert "Median s/clip" in tables["per_clip"].columns
    hours = tables["hours"]
    assert "ABEL clip review (h)" in hours.columns
    assert "Manual 1x real-time (h)" in hours.columns
    assert hours.loc[0, "Manual 1x real-time (h)"] == pytest.approx(4.0)


def test_prism_review_effort_drops_projects_with_nothing_timed():
    df = re.results_to_frame([_result("ok", [1.0] * 5, 2.0),
                              re.ReviewEffortResult(project_id="bad", error="boom")])
    tables = prism.prism_review_effort(df)
    assert list(tables["per_clip"]["Project"]) == ["ok"]


def test_prism_review_effort_is_empty_when_no_project_measured():
    df = re.results_to_frame([re.ReviewEffortResult(project_id="bad", error="boom")])
    assert prism.prism_review_effort(df) == {}


def test_plot_review_effort_writes_a_figure(tmp_path):
    out = re.plot_review_effort([_result("a", [1.0] * 20, 4.0),
                                 _result("b", [2.0] * 20, 6.0)],
                                tmp_path / "effort.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_review_effort_survives_having_nothing_to_plot(tmp_path):
    out = re.plot_review_effort([re.ReviewEffortResult(project_id="x", error="boom")],
                                tmp_path / "empty.png")
    assert out.exists()


# ── findings ───────────────────────────────────────────────────────────────


def test_findings_state_the_cost_and_both_caveats():
    from abel.validation import findings as findings_mod

    out = findings_mod.derive_findings(findings_mod.FindingsInput(
        effort_results=[_result("a", [2.0] * 1800, 20.0)]))
    effort = [f for f in out if f.analysis == "Review effort"]
    assert effort, "review-effort findings should be derived"
    assert any(f.kind == findings_mod.KIND_RESULT for f in effort)
    caveats = [f for f in effort if f.kind == findings_mod.KIND_CAVEAT]
    # The floor caveat and the 4 s/clip-assumption caveat must both be stated.
    assert len(caveats) == 2
    assert any("floor" in f.headline for f in caveats)
    assert any("rare-discovery" in f.headline for f in caveats)


def test_findings_are_absent_when_no_project_was_measured():
    from abel.validation import findings as findings_mod

    out = findings_mod.derive_findings(findings_mod.FindingsInput(
        effort_results=[re.ReviewEffortResult(project_id="x", error="boom")]))
    assert not [f for f in out if f.analysis == "Review effort"]
