"""Interval sets that give every subject something to align a signal to.

Viterbi state bouts can be empty for a subject, which is a missing value rather
than a zero and drops that subject out of a within-subject comparison.  Two
alternatives fill the gap, and each has a property that must not silently
regress: the top-N posterior selection has to apply the *same* rule to every
subject (or a group difference is confounded with which rule each got), and the
unclaimed stretches have to be exactly the complement of the state bouts.
"""

from __future__ import annotations

import numpy as np
import pytest

from abel.services.behavioral_motif_service import (
    UNCLAIMED_STATE,
    default_state_label,
    posterior_selection_summary,
    posterior_state_intervals,
    state_bouts_to_frames,
    unclaimed_intervals,
)


def _events(labels, step=2.0, dur=1.0):
    return [(i * step, i * step + dur, b) for i, b in enumerate(labels)]


# ---------------------------------------------------------------------------
# posterior_state_intervals
# ---------------------------------------------------------------------------

def test_every_session_gets_intervals_for_every_state():
    """The whole point: no subject is missing from any state."""
    seqs = {
        "has_it": _events(["a", "b", "b", "a"]),
        "hasnt": _events(["a", "a", "a", "a"]),
    }
    post = {
        "has_it": [[0.9, 0.1], [0.1, 0.9], [0.05, 0.95], [0.8, 0.2]],
        "hasnt": [[0.99, 0.01], [0.98, 0.02], [0.97, 0.03], [0.99, 0.01]],
    }
    out = posterior_state_intervals(seqs, ["a", "b"], post, n_states=2, top_n=2)
    for sid in seqs:
        states = {b["state"] for b in out[sid]}
        assert states == {0, 1}, f"{sid} is missing a state"


def test_same_number_of_bouts_selected_for_every_subject():
    """One rule for everyone, or the comparison is confounded with the rule."""
    seqs = {s: _events(["a", "b", "a", "b", "a"]) for s in ("s1", "s2", "s3")}
    rng = np.random.default_rng(3)
    post = {}
    for s in seqs:
        m = rng.random((5, 2))
        post[s] = (m / m.sum(axis=1, keepdims=True)).tolist()

    out = posterior_state_intervals(seqs, ["a", "b"], post, n_states=2, top_n=3)
    for state in (0, 1):
        counts = {
            sid: sum(b["n_bouts"] for b in out[sid] if b["state"] == state)
            for sid in seqs
        }
        assert len(set(counts.values())) == 1, counts


def test_selection_takes_the_highest_posterior_bouts():
    seqs = {"s": _events(["a", "b", "a", "b"])}
    post = {"s": [[0.1, 0.9], [0.2, 0.8], [0.95, 0.05], [0.99, 0.01]]}
    out = posterior_state_intervals(seqs, ["a", "b"], post, n_states=2, top_n=2)
    s0 = [b for b in out["s"] if b["state"] == 0]
    picked = sorted(i for b in s0 for i in range(b["first_bout_index"],
                                                 b["last_bout_index"] + 1))
    assert picked == [2, 3]


def test_consecutive_picks_merge_into_one_interval():
    seqs = {"s": _events(["a", "a", "a", "a"])}
    post = {"s": [[0.1, 0.9], [0.9, 0.1], [0.95, 0.05], [0.2, 0.8]]}
    out = posterior_state_intervals(seqs, ["a"], post, n_states=2, top_n=2)
    s0 = [b for b in out["s"] if b["state"] == 0]
    assert len(s0) == 1
    assert s0[0]["n_bouts"] == 2
    assert s0[0]["start_s"] == pytest.approx(2.0)
    assert s0[0]["end_s"] == pytest.approx(5.0)


def test_intervals_carry_the_evidence_that_qualifies_them():
    """A non-empty selection must never look like evidence on its own."""
    seqs = {"never": _events(["a", "a", "a"])}
    post = {"never": [[0.999, 0.001], [0.998, 0.002], [0.999, 0.001]]}
    out = posterior_state_intervals(seqs, ["a"], post, n_states=2, top_n=2)
    rare = [b for b in out["never"] if b["state"] == 1]
    assert rare, "the subject should still be represented"
    assert all(b["posterior_mean"] < 0.01 for b in rare)
    assert all(b["expected_bouts"] < 0.01 for b in rare)
    assert all(b["source"] == "posterior_topn" for b in rare)


def test_summary_reports_expected_bouts_per_session_and_state():
    seqs = {"s": _events(["a", "b", "a"])}
    post = {"s": [[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]]}
    out = posterior_state_intervals(seqs, ["a", "b"], post, n_states=2, top_n=2)
    rows = posterior_selection_summary(out, 2)
    assert {r["state"] for r in rows} == {0, 1}
    by_state = {r["state"]: r for r in rows}
    assert by_state[1]["expected_bouts"] == pytest.approx(1.5)
    assert by_state[0]["expected_bouts"] == pytest.approx(1.5)


def test_top_n_of_zero_selects_nothing():
    seqs = {"s": _events(["a", "b"])}
    post = {"s": [[0.5, 0.5], [0.5, 0.5]]}
    assert posterior_state_intervals(seqs, ["a", "b"], post, 2, top_n=0) == {}


# ---------------------------------------------------------------------------
# unclaimed_intervals
# ---------------------------------------------------------------------------

def _bout(state, start, end):
    return {"state": state, "start_s": start, "end_s": end,
            "duration_s": end - start, "n_bouts": 1, "behavior_ids": []}


def test_unclaimed_covers_head_gaps_and_tail():
    bouts = {"s": [_bout(0, 10.0, 20.0), _bout(1, 30.0, 40.0)]}
    out = unclaimed_intervals(bouts, session_end_s={"s": 60.0})
    spans = [(b["start_s"], b["end_s"]) for b in out["s"]]
    assert spans == [(0.0, 10.0), (20.0, 30.0), (40.0, 60.0)]
    assert all(b["state"] == UNCLAIMED_STATE for b in out["s"])
    assert all(b["source"] == "unclaimed" for b in out["s"])


def test_unclaimed_is_the_exact_complement_of_the_state_bouts():
    bouts = {"s": [_bout(0, 5.0, 12.0), _bout(2, 12.0, 19.0), _bout(1, 25.0, 30.0)]}
    end = 44.0
    out = unclaimed_intervals(bouts, session_end_s={"s": end})
    covered = sum(b["end_s"] - b["start_s"] for b in bouts["s"])
    gap = sum(b["duration_s"] for b in out["s"])
    assert covered + gap == pytest.approx(end)


def test_overlapping_state_bouts_do_not_invent_negative_gaps():
    # A max_gap_s setting can leave two state bouts overlapping in time.
    bouts = {"s": [_bout(0, 0.0, 30.0), _bout(1, 10.0, 20.0)]}
    out = unclaimed_intervals(bouts, session_end_s={"s": 40.0})
    assert [(b["start_s"], b["end_s"]) for b in out["s"]] == [(30.0, 40.0)]


def test_short_slivers_are_dropped_by_min_duration():
    bouts = {"s": [_bout(0, 0.0, 10.0), _bout(1, 10.2, 20.0), _bout(0, 25.0, 30.0)]}
    out = unclaimed_intervals(bouts, session_end_s={"s": 30.0}, min_duration_s=1.0)
    assert [(b["start_s"], b["end_s"]) for b in out["s"]] == [(20.0, 25.0)]


def test_no_session_end_means_no_tail_interval():
    bouts = {"s": [_bout(0, 5.0, 10.0)]}
    out = unclaimed_intervals(bouts)
    assert [(b["start_s"], b["end_s"]) for b in out["s"]] == [(0.0, 5.0)]


# ---------------------------------------------------------------------------
# export shape
# ---------------------------------------------------------------------------

def test_null_column_is_seeded_on_every_session():
    """Same columns on every sheet, including a session with no gaps at all."""
    bouts = {
        "gappy": [_bout(0, 10.0, 20.0), _bout(1, 30.0, 40.0)],
        "solid": [_bout(0, 0.0, 40.0)],
    }
    null = unclaimed_intervals(bouts, session_end_s={"gappy": 40.0, "solid": 40.0})
    merged = {s: bouts[s] + null[s] for s in bouts}
    frames = state_bouts_to_frames(merged, fps=10.0, n_states=2, include_unclaimed=True)
    key = default_state_label(UNCLAIMED_STATE)
    assert key == "HMM_NoState"
    for sid in bouts:
        assert key in frames[sid]
    assert frames["solid"][key] == []
    assert frames["gappy"][key] == [(0, 100), (200, 300)]


def test_unclaimed_column_absent_unless_requested():
    bouts = {"s": [_bout(0, 0.0, 1.0)]}
    frames = state_bouts_to_frames(bouts, fps=10.0, n_states=2)
    assert default_state_label(UNCLAIMED_STATE) not in frames["s"]
