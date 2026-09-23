"""Tests for social-interaction summaries and displacement-dominance scoring.

Most tests exercise the pure analytical logic (summary metrics,
interaction-state identification, displacement scoring given a state
assignment, dwell/switching, cohort checks), which runs without ``hmmlearn``.
The end-to-end fit is skipped when ``hmmlearn`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services.social_analysis_service import SocialAnalysisService, _runs

# Body length is 20 px in these frames (40 px apart = 2.0 BL), so a radial
# speed of 30 px/s is 1.5 BL/s: well above the 0.5 BL/s displacement threshold.
_BL_PX = 20.0


def _two_animal_df(n: int = 20, a_radial=30.0, b_radial=-30.0):
    """A→B: animal 'A' advances (radial toward > 0), 'B' yields (radial < 0)."""
    frames = np.arange(n)
    dist_px = np.linspace(40, 10, n)
    a = pd.DataFrame({
        "animal_id": "A",
        "session_id": "s1",
        "frame": frames,
        "social_dist_centroid_to_centroid_nearest": dist_px,
        "social_dist_centroid_to_centroid_nearest_norm": dist_px / _BL_PX,
        "social_radial_velocity_toward_nearest": np.broadcast_to(a_radial, n).astype(float),
        "social_approach_velocity_nearest": np.full(n, 2.0),
        "social_in_contact": (dist_px < 20).astype(float),
        "social_facing_angle_nearest": np.full(n, 0.2),
        "social_heading_alignment_nearest": np.full(n, 0.9),
        "centroid_velocity": np.full(n, 3.0),
    })
    b = a.copy()
    b["animal_id"] = "B"
    b["social_radial_velocity_toward_nearest"] = np.broadcast_to(b_radial, n).astype(float)
    return pd.concat([a, b], ignore_index=True)


def _all_interaction(n: int) -> dict:
    return {("A", "s1"): np.zeros(n, dtype=int), ("B", "s1"): np.zeros(n, dtype=int)}


def test_summary_metrics_per_subject():
    df = _two_animal_df()
    svc = SocialAnalysisService()
    rows = svc.compute_social_summary(df, fps=10.0)
    assert len(rows) == 2
    by_id = {r["animal_id"]: r for r in rows}
    # A advances (radial > 0 every frame), B yields.
    assert by_id["A"]["advance_fraction"] == 1.0
    assert by_id["B"]["advance_fraction"] == 0.0
    # Both share the same distance trajectory.
    assert abs(by_id["A"]["mean_distance_norm"] - by_id["B"]["mean_distance_norm"]) < 1e-9
    # Contact bouts detected once distance drops below threshold.
    assert by_id["A"]["n_contact_bouts"] >= 1
    assert by_id["A"]["contact_time_s"] > 0


def test_identify_interaction_states_by_proximity():
    svc = SocialAnalysisService()
    feats = ["social_dist_centroid_to_centroid_nearest_norm", "social_in_contact"]
    profiles = {
        0: {"social_dist_centroid_to_centroid_nearest_norm": 2.5, "social_in_contact": 0.0},
        1: {"social_dist_centroid_to_centroid_nearest_norm": 0.4, "social_in_contact": 0.9},
    }
    inter = svc._identify_interaction_states(profiles, feats)
    assert inter == [1]  # the close state


def test_displacement_dominance_ranks_advancer_first():
    df = _two_animal_df()
    svc = SocialAnalysisService()
    dom = svc.compute_displacement_dominance(
        df, _all_interaction(20), interaction_states=[0], fps=10.0, group_map={}
    )
    by_id = {r["animal_id"]: r for r in dom}
    assert by_id["A"]["dominance_rank"] == 1
    assert by_id["A"]["is_dominant"] is True
    assert by_id["B"]["dominance_rank"] == 2
    assert by_id["B"]["is_dominant"] is False
    # One unbroken displacement, won by A: index +1 / -1.
    assert by_id["A"]["method"] == "paired"
    assert by_id["A"]["displacements_won"] == 1.0
    assert by_id["A"]["displacements_lost"] == 0.0
    assert by_id["A"]["dominance_score"] == 1.0
    assert by_id["B"]["dominance_score"] == -1.0
    # A advances every frame → yield_fraction 0; B yields every frame → 1.
    assert by_id["A"]["yield_fraction"] == 0.0
    assert by_id["B"]["yield_fraction"] == 1.0


def test_dominance_restricted_to_interaction_states():
    df = _two_animal_df()
    svc = SocialAnalysisService()
    # State 1 = interaction only on the second half of frames.
    seq = np.array([0] * 10 + [1] * 10)
    dom = svc.compute_displacement_dominance(
        df, {("A", "s1"): seq, ("B", "s1"): seq}, interaction_states=[1],
        fps=10.0, group_map={},
    )
    by_id = {r["animal_id"]: r for r in dom}
    # Only 10 interaction frames counted.
    assert abs(by_id["A"]["interaction_time_s"] - 1.0) < 1e-9


def test_untracked_frames_never_count_as_interaction():
    df = _two_animal_df()
    svc = SocialAnalysisService()
    seq = np.array([-1] * 15 + [0] * 5)
    dom = svc.compute_displacement_dominance(
        df, {("A", "s1"): seq, ("B", "s1"): seq}, interaction_states=[0],
        fps=10.0, group_map={},
    )
    assert abs({r["animal_id"]: r for r in dom}["A"]["interaction_time_s"] - 0.5) < 1e-9


def test_paired_index_counts_events_both_ways():
    # A wins frames 0-4 and 10-14, B wins frames 20-24; gaps are still.
    n = 30
    a = np.zeros(n)
    b = np.zeros(n)
    a[0:5], b[0:5] = 30.0, -30.0
    a[10:15], b[10:15] = 30.0, -30.0
    a[20:25], b[20:25] = -30.0, 30.0
    df = _two_animal_df(n, a_radial=a, b_radial=b)
    svc = SocialAnalysisService()
    out = svc.displacement_analysis(
        df, _all_interaction(n), [0], fps=10.0, group_map={"s1": "G"}, bin_s=1.0,
    )
    by_id = {r["animal_id"]: r for r in out["dominance"]}
    assert by_id["A"]["displacements_won"] == 2.0
    assert by_id["B"]["displacements_won"] == 1.0
    assert by_id["A"]["dominance_score"] == pytest.approx(1 / 3)
    assert by_id["A"]["group"] == "G"
    assert len(out["events"]) == 3
    assert {e["winner"] for e in out["events"]} == {"A", "B"}
    # 1 s bins at 10 fps: three bins, one event each.
    a_bins = [b for b in out["bins"] if b["animal_id"] == "A"]
    assert [b["dominance_index"] for b in a_bins] == [1.0, 1.0, -1.0]


def test_events_shorter_than_min_duration_are_ignored():
    n = 20
    a = np.zeros(n)
    b = np.zeros(n)
    a[3:4], b[3:4] = 30.0, -30.0          # 1 frame = 0.1 s at 10 fps
    df = _two_animal_df(n, a_radial=a, b_radial=b)
    svc = SocialAnalysisService()
    out = svc.displacement_analysis(
        df, _all_interaction(n), [0], fps=10.0, group_map={}, min_event_s=0.3,
    )
    assert out["events"] == []
    assert all(not r["is_dominant"] for r in out["dominance"])


def test_threshold_is_in_body_lengths():
    # 8 px/s = 0.4 BL/s: below the default 0.5 BL/s, so no displacement.
    df = _two_animal_df(a_radial=8.0, b_radial=-8.0)
    svc = SocialAnalysisService()
    out = svc.displacement_analysis(df, _all_interaction(20), [0], fps=10.0, group_map={})
    assert out["events"] == []
    out = svc.displacement_analysis(
        df, _all_interaction(20), [0], fps=10.0, group_map={}, move_thresh_bl=0.3,
    )
    assert len(out["events"]) == 1


def test_runs_helper():
    assert _runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]
    assert _runs(np.zeros(0, dtype=bool)) == []


def test_dwell_and_switching_skip_untracked_gaps():
    svc = SocialAnalysisService()
    seqs = {("A", "s"): np.array([0, 0, 1, 1, 1, -1, 1, 0], dtype=np.int16)}
    dwell, switch = svc._dwell_and_switching(seqs, 2, fps=1.0)
    assert dwell[0]["n"] == 2 and dwell[1]["n"] == 2
    assert dwell[1]["mean_s"] == pytest.approx(2.0)
    # 0→1 once, 1→0 once; 1→(-1)→1 is not a switch.
    assert switch[0, 1] == 1.0
    assert switch[1, 0] == 1.0


def test_identity_rank_bias_and_group_steepness():
    rows = []
    for i in range(6):
        top = "track_0" if i < 5 else "track_1"
        other = "track_1" if top == "track_0" else "track_0"
        g = "ctrl" if i % 2 else "drug"
        rows.append({"session_id": f"s{i}", "animal_id": top, "method": "paired",
                     "is_dominant": True, "dominance_score": 0.2 + 0.1 * i, "group": g})
        rows.append({"session_id": f"s{i}", "animal_id": other, "method": "paired",
                     "is_dominant": False, "dominance_score": -(0.2 + 0.1 * i), "group": g})
    svc = SocialAnalysisService()
    idc = svc.identity_rank_bias(rows)
    assert idc["counts"] == {"track_0": 5, "track_1": 1}
    assert idc["n_sessions"] == 6
    assert 0 < idc["p"] < 1
    gs = svc.group_steepness(rows)
    assert sorted(gs["by_group"]) == ["ctrl", "drug"]
    assert gs["summary"]["drug"]["n"] == 3
    assert gs["test"] == "Mann-Whitney U"


def test_no_social_columns_returns_none_semantics():
    svc = SocialAnalysisService()
    plain = pd.DataFrame({"animal_id": ["A"], "session_id": ["s1"], "frame": [0]})
    assert not svc.has_social_features(plain)
    assert svc.has_social_features(_two_animal_df())


def _synthetic_cohort(n_sessions: int = 3, n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Dyads alternating between 'close' and 'apart' blocks, with a tracking gap."""
    rng = np.random.default_rng(seed)
    parts = []
    for si in range(n_sessions):
        close = (np.arange(n) // 60) % 2 == 0
        dist_bl = np.where(close, 0.8, 4.0) + rng.normal(0, 0.1, n)
        for aid, sign in (("track_0", 1.0), ("track_1", -1.0)):
            radial = np.where(close, sign * 40.0, 0.0) + rng.normal(0, 3, n)
            d = pd.DataFrame({
                "animal_id": aid,
                "session_id": f"s{si}",
                "frame": np.arange(n),
                "social_dist_centroid_to_centroid_nearest": dist_bl * _BL_PX,
                "social_dist_centroid_to_centroid_nearest_norm": dist_bl,
                "social_radial_velocity_toward_nearest": radial,
                "social_in_contact": close.astype(float),
                "centroid_velocity": np.abs(radial) + rng.normal(5, 1, n),
            })
            # Partner lost for 30 frames; one tracking jump.
            d.loc[300:329, [c for c in d.columns if c.startswith("social_")]] = np.nan
            d.loc[400, "centroid_velocity"] = 1e5
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_fit_end_to_end_orders_states_and_marks_gaps():
    pytest.importorskip("hmmlearn")
    df = _synthetic_cohort()
    svc = SocialAnalysisService()
    res = svc.fit_dominance_hmm(df, fps=30.0, n_states=2, n_restarts=2, bin_s=5.0)
    assert res["error"] is None
    # S0 is the closest state.
    d0 = res["state_profiles"][0]["social_dist_centroid_to_centroid_nearest_norm"]
    d1 = res["state_profiles"][1]["social_dist_centroid_to_centroid_nearest_norm"]
    assert d0 < d1
    assert res["interaction_states"] == [0]
    # Untracked frames carry -1 and are reported, not fit.
    seq = res["state_seqs"][("track_0", "s0")]
    assert (seq[300:330] == -1).all()
    assert (seq[:300] >= 0).all()
    assert res["no_partner_fraction"][("track_0", "s0")] == pytest.approx(30 / 600)
    # The planted advancer wins every session.
    top = {r["session_id"]: r["animal_id"] for r in res["dominance"] if r["is_dominant"]}
    assert top == {"s0": "track_0", "s1": "track_0", "s2": "track_0"}
    assert res["identity_check"]["counts"]["track_0"] == 3
    assert res["dominance_bins"] and res["displacement_events"]
    assert res["switch_matrix"].shape == (2, 2)
    assert len(res["fit_info"]["restart_log_likelihoods"]) == 2


def test_result_round_trip(tmp_path):
    svc = SocialAnalysisService()
    (tmp_path / "derived" / "pose_features").mkdir(parents=True)
    svc.frame_pose_path(tmp_path).write_bytes(b"x")
    settings = {"n_states": 2}
    fp = svc.input_fingerprint(tmp_path, settings)
    svc.save_result(tmp_path, {"n_states": 2, "settings": settings}, fp)
    res, saved_fp = svc.load_result(tmp_path)
    assert res["n_states"] == 2
    assert saved_fp == fp
    # Changing the frame table invalidates the fingerprint.
    svc.frame_pose_path(tmp_path).write_bytes(b"xy")
    assert svc.input_fingerprint(tmp_path, settings) != fp
