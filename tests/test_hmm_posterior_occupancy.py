"""Posterior (soft) state occupancy for the behavioral-motif HMM.

The point of the posterior measure is that Viterbi occupancy is a *thresholded*
readout: entering a state for one or two bouts costs two transition penalties,
so a subject who performed a rare behavior a handful of times can report exactly
0.000 occupancy in that behavior's state.  These tests pin the shape and the
normalisation of the soft measure, the fallback when it is unavailable, and the
behaviour that motivated it.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from abel.services.behavioral_motif_service import (
    MotifSettings,
    fit_hmm,
    resolve_state_occupancy,
    state_occupancy,
    state_occupancy_from_posteriors,
)


def _seq(sid_events):
    """[(start, end, bid), ...] with 1 s bouts spaced 2 s apart."""
    return [(float(2 * i), float(2 * i + 1), bid) for i, bid in enumerate(sid_events)]


# ---------------------------------------------------------------------------
# state_occupancy_from_posteriors
# ---------------------------------------------------------------------------

def test_posteriors_average_within_each_session_block():
    # Two sessions of 2 and 3 observations over 2 states.
    post = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0.5, 0.5],
        [1.0, 0.0],
    ])
    out = state_occupancy_from_posteriors(post, [2, 3], ["a", "b"], 2)
    assert out["a"] == pytest.approx([0.5, 0.5])
    assert out["b"] == pytest.approx([2.0 / 3.0, 1.0 / 3.0])


def test_rows_sum_to_one():
    rng = np.random.default_rng(0)
    post = rng.random((7, 3))
    post /= post.sum(axis=1, keepdims=True)
    out = state_occupancy_from_posteriors(post, [4, 3], ["a", "b"], 3)
    for fracs in out.values():
        assert sum(fracs) == pytest.approx(1.0)


def test_empty_session_yields_zeros_not_nan():
    post = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = state_occupancy_from_posteriors(post, [2, 0], ["a", "b"], 2)
    assert out["b"] == [0.0, 0.0]
    assert not any(np.isnan(out["a"]))


# ---------------------------------------------------------------------------
# resolve_state_occupancy
# ---------------------------------------------------------------------------

def test_resolve_defaults_to_viterbi():
    res = {
        "n_states": 2,
        "state_sequences": {"a": [0, 0, 1, 1]},
        "posterior_occupancy": {"a": [0.9, 0.1]},
    }
    occ, method = resolve_state_occupancy(res, "viterbi")
    assert method == "viterbi"
    assert occ["a"] == pytest.approx([0.5, 0.5])


def test_resolve_returns_posterior_when_asked():
    res = {
        "n_states": 2,
        "state_sequences": {"a": [0, 0, 1, 1]},
        "posterior_occupancy": {"a": [0.9, 0.1]},
    }
    occ, method = resolve_state_occupancy(res, "posterior")
    assert method == "posterior"
    assert occ["a"] == pytest.approx([0.9, 0.1])


def test_resolve_falls_back_and_says_so_when_posteriors_missing():
    # A caller must never label Viterbi numbers as posterior ones.
    res = {"n_states": 2, "state_sequences": {"a": [0, 0, 1, 1]}, "posterior_occupancy": {}}
    occ, method = resolve_state_occupancy(res, "posterior")
    assert method == "viterbi"
    assert occ["a"] == pytest.approx([0.5, 0.5])


# ---------------------------------------------------------------------------
# fit_hmm integration
# ---------------------------------------------------------------------------

pytestmark_fit = pytest.mark.skipif(
    importlib.util.find_spec("hmmlearn") is None,
    reason="hmmlearn not installed",
)


def _rare_behavior_dataset():
    """Sessions over 2 behaviors where 'R' is rare and always isolated.

    Most sessions are pure 'C'; a few carry a single interleaved 'R'; one is a
    long run of 'R'.  The run makes the fit allocate a state to 'R', and the
    isolated singletons are exactly the case Viterbi refuses to visit it for.
    """
    seqs = {}
    for i in range(6):
        seqs[f"pure_{i}"] = _seq(["C"] * 14)
    for i in range(4):
        seqs[f"single_{i}"] = _seq(["C"] * 7 + ["R"] + ["C"] * 6)
    seqs["burst"] = _seq(["C"] * 3 + ["R"] * 10 + ["C"] * 3)
    return seqs


@pytestmark_fit
def test_fit_hmm_returns_posterior_occupancy_shaped_like_viterbi():
    seqs = _rare_behavior_dataset()
    settings = MotifSettings(
        hmm_n_states_mode="manual", hmm_n_states=2,
        hmm_n_iter=200, hmm_n_restarts=3, hmm_random_seed=0,
    )
    res = fit_hmm(seqs, ["C", "R"], settings)
    assert res.get("error") is None

    post = res["posterior_occupancy"]
    vit = state_occupancy(res["state_sequences"], res["n_states"])
    assert set(post) == set(vit)
    for sid in post:
        assert len(post[sid]) == res["n_states"]
        assert sum(post[sid]) == pytest.approx(1.0)


@pytestmark_fit
def test_posterior_gives_isolated_rare_bouts_nonzero_weight():
    """The reason the option exists.

    A session holding one isolated 'R' gets zero Viterbi occupancy in the
    R-state -- the path will not pay two transition costs for one observation --
    while its posterior occupancy is small but strictly positive, and strictly
    larger than that of a session with no 'R' at all.
    """
    seqs = _rare_behavior_dataset()
    settings = MotifSettings(
        hmm_n_states_mode="manual", hmm_n_states=2,
        hmm_n_iter=200, hmm_n_restarts=3, hmm_random_seed=0,
    )
    res = fit_hmm(seqs, ["C", "R"], settings)
    assert res.get("error") is None

    emis = np.array(res["emission_matrix"])
    r_state = int(np.argmax(emis[:, 1]))
    if emis[r_state, 1] < 0.5:
        pytest.skip("fit did not allocate a state to the rare behavior")

    vit = state_occupancy(res["state_sequences"], res["n_states"])
    post = res["posterior_occupancy"]

    singles = [s for s in seqs if s.startswith("single_")]
    pures = [s for s in seqs if s.startswith("pure_")]

    assert all(vit[s][r_state] == 0.0 for s in singles), (
        "expected Viterbi to refuse the rare state for isolated bouts; "
        "the dataset no longer exercises the case this option is for"
    )
    for s in singles:
        assert post[s][r_state] > 0.0
        for p in pures:
            assert post[s][r_state] > post[p][r_state]
