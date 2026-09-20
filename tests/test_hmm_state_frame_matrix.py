"""Per-frame HMM state matrix: the ethogram CSV export's data layer.

The matrix is what a user reads frame by frame, so the two things that must
hold are (a) a frame is only given a state when a state bout actually covers
it, and (b) a frame number here means the same frame it means in the state
boutframes workbook.
"""

from __future__ import annotations

import numpy as np

from abel.services.behavioral_motif_service import (
    state_bouts_to_frames,
    state_frame_matrix,
)

FPS = 10.0


def _bouts(items: list[tuple[float, float, int]], sid: str = "s1") -> dict[str, list[dict]]:
    return {
        sid: [
            {
                "state": int(st),
                "start_s": float(a),
                "end_s": float(b),
                "duration_s": float(b) - float(a),
                "n_bouts": 1,
                "behavior_ids": ["b0"],
                "first_bout_index": i,
                "last_bout_index": i,
            }
            for i, (a, b, st) in enumerate(items)
        ]
    }


def test_covered_frames_carry_the_state_and_gaps_stay_nan():
    mat = state_frame_matrix(_bouts([(0.0, 0.5, 0), (1.0, 1.2, 2)]), fps=FPS)
    arr = mat["s1"]

    assert len(arr) == 13, "runs to the last covered frame inclusive"
    assert np.array_equal(arr[0:6], np.zeros(6)), "0.0-0.5 s is state 0"
    assert np.all(np.isnan(arr[6:10])), "the stretch between state bouts is unmodeled"
    assert np.array_equal(arr[10:13], np.full(3, 2.0)), "1.0-1.2 s is state 2"


def test_frames_match_the_boutframes_export_exactly():
    """A frame in the CSV must be the frame the TRACY workbook calls the same."""
    bouts = _bouts([(0.3, 0.7, 1), (2.0, 2.4, 0)])
    offsets = {"s1": 45}

    mat = state_frame_matrix(bouts, fps=FPS, frame_offsets=offsets, raw_video_frames=True)
    intervals = state_bouts_to_frames(bouts, fps=FPS, n_states=2, frame_offsets=offsets)

    arr = mat["s1"]
    for state, spans in intervals["s1"].items():
        st = int(state.rsplit("_", 1)[1])
        for start, end in spans:
            assert np.all(arr[start:end + 1] == st), f"{state} {start}-{end}"


def test_raw_video_mode_blanks_the_prechopped_head():
    mat = state_frame_matrix(
        _bouts([(0.0, 0.2, 1)]), fps=FPS,
        frame_offsets={"s1": 30}, raw_video_frames=True,
    )
    arr = mat["s1"]
    assert np.all(np.isnan(arr[:30])), "pre-assay video frames have no state"
    assert np.array_equal(arr[30:33], np.ones(3))


def test_assay_aligned_mode_ignores_the_offset():
    """Aligned columns must share a clock, so the prechop must not shift them."""
    bouts = {**_bouts([(0.0, 0.2, 1)], sid="a"), **_bouts([(0.0, 0.2, 1)], sid="b")}
    mat = state_frame_matrix(bouts, fps=FPS, frame_offsets={"a": 0, "b": 300})
    assert np.array_equal(mat["a"], mat["b"])


def test_recording_length_pads_past_the_last_bout():
    mat = state_frame_matrix(
        _bouts([(0.0, 0.2, 0)]), fps=FPS, session_end_s={"s1": 6.0},
    )
    arr = mat["s1"]
    assert len(arr) == 60, "the column stops where the recording stops"
    assert np.all(np.isnan(arr[3:])), "unobserved behavior is not a state"


def test_a_bout_running_past_the_recorded_end_is_not_truncated():
    mat = state_frame_matrix(
        _bouts([(0.0, 3.0, 1)]), fps=FPS, session_end_s={"s1": 1.0},
    )
    assert len(mat["s1"]) == 31
    assert np.all(mat["s1"] == 1.0)


def test_sessions_with_no_bouts_still_span_their_recording():
    mat = state_frame_matrix({"s1": []}, fps=FPS, session_end_s={"s1": 2.0})
    assert len(mat["s1"]) == 20
    assert np.all(np.isnan(mat["s1"]))


def test_zero_fps_is_refused_rather_than_silently_wrong():
    import pytest

    with pytest.raises(ValueError):
        state_frame_matrix(_bouts([(0.0, 1.0, 0)]), fps=0.0)


# ---------------------------------------------------------------------------
# The "no state" code: an explicit number for the unmodeled stretches
# ---------------------------------------------------------------------------

def test_no_state_value_fills_the_gaps_and_leaves_states_alone():
    mat = state_frame_matrix(
        _bouts([(0.0, 0.5, 0), (1.0, 1.2, 2)]), fps=FPS,
        session_end_s={"s1": 2.0}, no_state_value=4,
    )
    arr = mat["s1"]

    assert not np.isnan(arr).any(), "nothing is left blank once a code is asked for"
    assert np.array_equal(arr[0:6], np.zeros(6))
    assert np.array_equal(arr[6:10], np.full(4, 4.0)), "the gap carries the code"
    assert np.array_equal(arr[10:13], np.full(3, 2.0))
    assert np.array_equal(arr[13:20], np.full(7, 4.0)), "the tail of the recording too"


def test_no_state_code_does_not_collide_with_a_real_state():
    """n_states is the next free number: states 0..3 leave 4 unused."""
    n_states = 4
    mat = state_frame_matrix(
        _bouts([(0.0, 0.2, 3)]), fps=FPS, session_end_s={"s1": 1.0},
        no_state_value=n_states,
    )
    assert set(np.unique(mat["s1"])) == {3.0, float(n_states)}


def test_prechopped_head_stays_blank_even_when_gaps_are_coded():
    """The HMM never saw the pre-assay frames, so they are not 'no state'."""
    mat = state_frame_matrix(
        _bouts([(0.0, 0.2, 1)]), fps=FPS, session_end_s={"s1": 1.0},
        frame_offsets={"s1": 30}, raw_video_frames=True, no_state_value=2,
    )
    arr = mat["s1"]
    assert np.all(np.isnan(arr[:30]))
    assert np.array_equal(arr[30:33], np.ones(3))
    assert np.all(arr[33:] == 2.0)


def test_default_still_leaves_the_gaps_blank():
    mat = state_frame_matrix(_bouts([(0.0, 0.2, 0)]), fps=FPS, session_end_s={"s1": 1.0})
    assert np.isnan(mat["s1"][3:]).all()
