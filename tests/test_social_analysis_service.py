"""Tests for social-interaction summaries and displacement-dominance scoring.

The Gaussian HMM fit needs the optional ``hmmlearn`` dependency; these tests
exercise the pure analytical logic (summary metrics, interaction-state
identification, and dominance ranking given a state assignment) which runs
without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abel.services.social_analysis_service import SocialAnalysisService


def _two_animal_df():
    """A→B: animal 'A' advances (radial toward > 0), 'B' yields (radial < 0)."""
    n = 20
    frames = np.arange(n)
    a = pd.DataFrame({
        "animal_id": "A",
        "session_id": "s1",
        "frame": frames,
        "social_dist_centroid_to_centroid_nearest": np.linspace(40, 10, n),
        "social_dist_centroid_to_centroid_nearest_norm": np.linspace(2.0, 0.5, n),
        "social_radial_velocity_toward_nearest": np.full(n, 3.0),   # advancing
        "social_approach_velocity_nearest": np.full(n, 2.0),
        "social_in_contact": (np.linspace(40, 10, n) < 20).astype(float),
        "social_facing_angle_nearest": np.full(n, 0.2),
        "social_heading_alignment_nearest": np.full(n, 0.9),
        "centroid_velocity": np.full(n, 3.0),
    })
    b = a.copy()
    b["animal_id"] = "B"
    b["social_radial_velocity_toward_nearest"] = np.full(n, -3.0)   # yielding
    b["centroid_velocity"] = np.full(n, 3.0)
    return pd.concat([a, b], ignore_index=True)


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
    # Pretend every frame is interaction state 0.
    state_seqs = {
        ("A", "s1"): np.zeros(20, dtype=int),
        ("B", "s1"): np.zeros(20, dtype=int),
    }
    dom = svc.compute_displacement_dominance(
        df, state_seqs, interaction_states=[0], fps=10.0, group_map={}
    )
    by_id = {r["animal_id"]: r for r in dom}
    assert by_id["A"]["dominance_rank"] == 1
    assert by_id["A"]["is_dominant"] is True
    assert by_id["B"]["dominance_rank"] == 2
    assert by_id["A"]["dominance_score"] > by_id["B"]["dominance_score"]
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


def test_no_social_columns_returns_none_semantics():
    svc = SocialAnalysisService()
    plain = pd.DataFrame({"animal_id": ["A"], "session_id": ["s1"], "frame": [0]})
    assert not svc.has_social_features(plain)
    assert svc.has_social_features(_two_animal_df())
