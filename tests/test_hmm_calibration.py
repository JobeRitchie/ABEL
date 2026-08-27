"""HMM fitting and auto-calibration behaviour.

These cover the three things that were silently wrong before calibration
existed: EM truncation making information criteria incomparable, unseeded
restarts making the selected state count irreproducible, and hmmlearn's
``monitor_.converged`` reporting success when EM merely ran out of iterations.
"""

from __future__ import annotations

import numpy as np
import pytest

from abel.services.behavioral_motif_service import (
    MotifSettings,
    _fit_single_hmm,
    calibrate_hmm_settings,
    fit_hmm,
    hmm_free_params,
    load_motif_settings,
    save_motif_settings,
)

hmmlearn = pytest.importorskip("hmmlearn")


BEHAVIORS = [f"b{i}" for i in range(4)]


def _synthetic_sequences(n_sessions: int = 10, n_bouts: int = 60, seed: int = 7):
    """Two-regime sequences: a session alternates between a 'b0/b1' regime and a
    'b2/b3' regime, so a 2-state HMM is the right generative model.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[float, float, str]]] = {}
    for s in range(n_sessions):
        events: list[tuple[float, float, str]] = []
        t = 0.0
        regime = 0
        for _ in range(n_bouts):
            if rng.random() < 0.15:
                regime = 1 - regime
            pool = BEHAVIORS[:2] if regime == 0 else BEHAVIORS[2:]
            bid = pool[int(rng.integers(0, 2))]
            events.append((t, t + 1.0, bid))
            t += 1.0 + float(rng.random())
        out[f"session_{s}"] = events
    return out


def test_free_params_matches_categorical_hmm_parameterisation():
    # K(K-1) transition + K(F-1) emission + (K-1) start
    assert hmm_free_params(4, 7) == 12 + 24 + 3
    assert hmm_free_params(2, 2) == 2 + 2 + 1


def test_converged_flag_is_false_when_em_exhausts_iterations():
    """hmmlearn's own monitor_.converged returns True at the iteration cap.

    _fit_single_hmm must not inherit that, or model selection cannot tell a
    truncated fit from a converged one.
    """
    seqs = _synthetic_sequences()
    obs = [
        np.array([BEHAVIORS.index(e[2]) for e in evs], dtype=int)
        for evs in seqs.values()
    ]
    model, ll, converged, iters = _fit_single_hmm(obs, 4, 2, len(BEHAVIORS), seed=0)
    assert iters == 2
    assert converged is False
    # hmmlearn itself would have claimed success here:
    assert model.monitor_.converged is True
    assert np.isfinite(ll)


def test_seeded_restarts_make_selection_reproducible():
    seqs = _synthetic_sequences()
    settings = MotifSettings(
        hmm_n_states_mode="auto", hmm_n_states_min=2, hmm_n_states_max=4,
        hmm_n_iter=200, hmm_n_restarts=3, hmm_criterion="bic", hmm_random_seed=11,
    )
    a = fit_hmm(seqs, BEHAVIORS, settings)
    b = fit_hmm(seqs, BEHAVIORS, settings)
    assert a["error"] is None
    assert a["n_states"] == b["n_states"]
    assert a["log_likelihood"] == pytest.approx(b["log_likelihood"], abs=1e-9)
    np.testing.assert_allclose(a["transition_matrix"], b["transition_matrix"], atol=1e-9)


def test_fit_hmm_reports_all_criteria_and_convergence():
    seqs = _synthetic_sequences()
    settings = MotifSettings(
        hmm_n_states_mode="auto", hmm_n_states_min=2, hmm_n_states_max=4,
        hmm_n_iter=300, hmm_n_restarts=3, hmm_criterion="icl", hmm_random_seed=3,
    )
    res = fit_hmm(seqs, BEHAVIORS, settings)
    assert res["error"] is None
    assert res["criterion_used"] == "icl"
    for row in res["model_selection"]:
        for key in ("aic", "aicc", "bic", "icl", "n_free_params", "obs_per_param"):
            assert key in row
        # ICL >= BIC: the entropy term is non-negative.
        assert row["icl"] >= row["bic"] - 1e-9
    assert res["unconverged_state_counts"] == []


def test_truncated_em_breaks_loglikelihood_monotonicity():
    """The concrete failure mode calibration exists to catch.

    At the true maximum, adding a state cannot lower the log-likelihood. With a
    tight iteration cap it does, which invalidates every AIC/BIC comparison
    built on those numbers.
    """
    seqs = _synthetic_sequences(n_sessions=12, n_bouts=80)

    def lls(n_iter: int) -> list[float]:
        s = MotifSettings(
            hmm_n_states_mode="auto", hmm_n_states_min=2, hmm_n_states_max=6,
            hmm_n_iter=n_iter, hmm_n_restarts=3, hmm_random_seed=5,
        )
        return [r["log_likelihood"] for r in fit_hmm(seqs, BEHAVIORS, s)["model_selection"]]

    converged = lls(500)
    assert all(b >= a - 1e-6 for a, b in zip(converged, converged[1:])), (
        "converged fits should be monotone in K"
    )
    truncated = lls(2)
    assert max(converged) > max(truncated), "truncation should cost log-likelihood"


def test_calibration_recommends_settings_with_evidence():
    seqs = _synthetic_sequences()
    settings = MotifSettings(hmm_n_iter=20, hmm_n_restarts=2, hmm_n_states_min=4)
    res = calibrate_hmm_settings(seqs, BEHAVIORS, settings, time_budget_s=30.0)

    assert res["error"] is None
    prop = res["proposed"]
    assert prop["hmm_n_iter"] >= 100
    assert prop["hmm_n_restarts"] >= 5
    assert prop["hmm_criterion"] == "icl"
    assert prop["hmm_n_states_mode"] == "manual"
    assert 2 <= prop["hmm_n_states"] <= prop["hmm_n_states_max"]
    # The search floor is not inherited from the user's setting: a min of 4
    # would make it impossible to recommend the 2-state generative truth.
    assert prop["hmm_n_states_min"] == 2

    # n_iter=20 truncates EM here, and that must be surfaced, not swallowed.
    assert any("n_iter" in w for w in res["warnings"])
    assert any("STAGE 2" in line for line in res["report"])
    assert any("Pohle" in line for line in res["report"])
    for row in res["table"]:
        assert row["obs_per_param"] >= 10.0


def test_calibration_bounds_states_by_available_data():
    """Two short sessions cannot support a large state space."""
    tiny = {
        "s0": [(float(i), float(i) + 0.5, BEHAVIORS[i % 4]) for i in range(20)],
        "s1": [(float(i), float(i) + 0.5, BEHAVIORS[(i + 1) % 4]) for i in range(20)],
    }
    settings = MotifSettings(hmm_n_states_max=12)
    res = calibrate_hmm_settings(tiny, BEHAVIORS, settings, time_budget_s=20.0)
    assert res["error"] is None
    assert res["proposed"]["hmm_n_states_max"] <= 4
    assert any("bouts across" in w for w in res["warnings"])


def test_progress_callback_is_driven_and_bounded():
    seqs = _synthetic_sequences(n_sessions=6, n_bouts=40)
    seen: list[tuple[str, float]] = []
    calibrate_hmm_settings(
        seqs, BEHAVIORS, MotifSettings(), time_budget_s=15.0,
        progress_cb=lambda m, f: seen.append((m, f)),
    )
    assert seen
    assert all(0.0 <= f <= 1.0 for _m, f in seen)
    assert seen[-1][1] == pytest.approx(1.0)


def test_random_seed_round_trips_through_settings_file(tmp_path):
    s = MotifSettings(hmm_random_seed=1234, hmm_criterion="icl")
    save_motif_settings(tmp_path, s)
    back = load_motif_settings(tmp_path)
    assert back.hmm_random_seed == 1234
    assert back.hmm_criterion == "icl"


def test_legacy_settings_file_without_seed_still_loads(tmp_path):
    """Existing projects were written before hmm_random_seed existed."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "motif_settings.json").write_text(
        '{"hmm_n_iter": 20, "hmm_criterion": "bic"}', encoding="utf-8"
    )
    back = load_motif_settings(tmp_path)
    assert back.hmm_n_iter == 20
    assert back.hmm_random_seed == 0


def test_permutation_pvalues_are_never_exactly_zero():
    """A p-value of 0 is not valid and survives FDR correction as 'infinitely
    significant' (Phipson & Smyth 2010). The floor is 1/(n_permutations+1).
    """
    from abel.services.behavioral_motif_service import permutation_test_transition

    # Perfectly separated groups: every permutation statistic is below the
    # observed one, which is exactly the case that used to produce p = 0.
    a = [np.full((3, 3), 10.0) for _ in range(5)]
    b = [np.zeros((3, 3)) for _ in range(5)]
    p = permutation_test_transition(a, b, n_permutations=200, seed=1)
    assert p.shape == (3, 3)
    assert np.all(p > 0.0)
    # Permutations that happen to recover the original split (or its complement)
    # tie the observed statistic and legitimately count, so p sits a little above
    # the floor rather than exactly on it.
    assert np.all(p >= 1.0 / 201.0)
    assert np.all(p < 0.05)


def test_permutation_pvalue_is_uniform_ish_under_the_null():
    from abel.services.behavioral_motif_service import permutation_test_transition

    rng = np.random.default_rng(0)
    mats = [rng.normal(size=(2, 2)) for _ in range(12)]
    p = permutation_test_transition(mats[:6], mats[6:], n_permutations=500, seed=2)
    assert np.all(p >= 1.0 / 501.0)
    assert np.all(p <= 1.0)
