"""Tests for restricting behavior bouts to an ROI (roi_behavior_service)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services import roi_behavior_service as rbs
from abel.utils.roi_geometry import debounce_bool


def _bouts(rows: list[tuple[str, int, int]], bid: str = "b1") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"session_id": s, "start_frame": a, "end_frame": b,
             "behavior_id": bid, "behavior": "Groom"}
            for s, a, b in rows
        ]
    )


def _mask(n: int, inside_spans: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in inside_spans:
        m[a:b + 1] = True
    return m


# ----------------------------------------------------------------------
# debounce_bool
# ----------------------------------------------------------------------

def test_debounce_merges_short_runs_into_predecessor():
    m = np.array([True] * 10 + [False] * 2 + [True] * 10)
    out = debounce_bool(m, min_run=5)
    assert out.all()


def test_debounce_leaves_first_run_alone():
    m = np.array([True] * 2 + [False] * 10)
    out = debounce_bool(m, min_run=5)
    assert out[0] and out[1]
    assert not out[2:].any()


def test_debounce_noop_below_threshold():
    m = np.array([True, False, True])
    assert np.array_equal(debounce_bool(m, 1), m)


# ----------------------------------------------------------------------
# Attribution: overlap
# ----------------------------------------------------------------------

def test_overlap_splits_a_straddling_bout():
    bouts = _bouts([("s1", 0, 99)])
    masks = {"s1": _mask(200, [(0, 49)])}  # first half of the bout is inside

    out, missing = rbs.scope_bouts_to_roi(
        bouts, masks, attribution=rbs.ATTRIBUTION_OVERLAP
    )

    assert not missing
    assert len(out) == 1
    assert int(out.iloc[0]["start_frame"]) == 0
    assert int(out.iloc[0]["end_frame"]) == 49


def test_overlap_emits_one_fragment_per_visit():
    bouts = _bouts([("s1", 0, 99)])
    masks = {"s1": _mask(200, [(10, 19), (60, 79)])}

    out, _ = rbs.scope_bouts_to_roi(bouts, masks, attribution=rbs.ATTRIBUTION_OVERLAP)

    assert list(zip(out["start_frame"], out["end_frame"])) == [(10, 19), (60, 79)]


def test_overlap_drops_fragments_below_min_frames():
    bouts = _bouts([("s1", 0, 99)])
    masks = {"s1": _mask(200, [(10, 12), (60, 79)])}

    out, _ = rbs.scope_bouts_to_roi(
        bouts, masks, attribution=rbs.ATTRIBUTION_OVERLAP, min_frames=5
    )

    assert list(zip(out["start_frame"], out["end_frame"])) == [(60, 79)]


def test_overlap_carries_other_columns_through():
    bouts = _bouts([("s1", 0, 99)])
    masks = {"s1": _mask(200, [(0, 49)])}

    out, _ = rbs.scope_bouts_to_roi(bouts, masks)

    assert out.iloc[0]["behavior_id"] == "b1"
    assert out.iloc[0]["behavior"] == "Groom"
    assert list(out.columns) == list(bouts.columns)


# ----------------------------------------------------------------------
# Attribution: onset / majority
# ----------------------------------------------------------------------

def test_onset_keeps_whole_bout_that_started_inside():
    bouts = _bouts([("s1", 0, 99)])
    masks = {"s1": _mask(200, [(0, 9)])}  # only the first 10 frames are inside

    out, _ = rbs.scope_bouts_to_roi(bouts, masks, attribution=rbs.ATTRIBUTION_ONSET)

    assert len(out) == 1
    assert int(out.iloc[0]["end_frame"]) == 99  # not truncated


def test_onset_drops_a_bout_that_entered_late():
    bouts = _bouts([("s1", 0, 99)])
    masks = {"s1": _mask(200, [(90, 99)])}

    out, _ = rbs.scope_bouts_to_roi(bouts, masks, attribution=rbs.ATTRIBUTION_ONSET)

    assert out.empty


def test_majority_needs_more_than_half_the_frames():
    bouts = _bouts([("s1", 0, 99), ("s1", 200, 299)])
    masks = {"s1": _mask(400, [(0, 60), (200, 240)])}  # 61% vs 41%

    out, _ = rbs.scope_bouts_to_roi(bouts, masks, attribution=rbs.ATTRIBUTION_MAJORITY)

    assert list(out["start_frame"]) == [0]


def test_majority_rejects_an_exact_half():
    bouts = _bouts([("s1", 0, 99)])
    masks = {"s1": _mask(200, [(0, 49)])}  # exactly 50 of 100 frames

    out, _ = rbs.scope_bouts_to_roi(bouts, masks, attribution=rbs.ATTRIBUTION_MAJORITY)

    assert out.empty


# ----------------------------------------------------------------------
# Inside vs outside
# ----------------------------------------------------------------------

def test_inside_and_outside_partition_the_bout():
    bouts = _bouts([("s1", 0, 99)])
    masks = {"s1": _mask(200, [(0, 49)])}

    inside, _ = rbs.scope_bouts_to_roi(bouts, masks, inside=True)
    outside, _ = rbs.scope_bouts_to_roi(bouts, masks, inside=False)

    dur = lambda df: int((df["end_frame"] - df["start_frame"] + 1).sum())  # noqa: E731
    assert dur(inside) == 50
    assert dur(outside) == 50
    assert dur(inside) + dur(outside) == 100


# ----------------------------------------------------------------------
# Prechop offsets — bouts are rebased, masks are not
# ----------------------------------------------------------------------

def test_prechop_offset_aligns_bouts_with_the_mask():
    # Bout at rebased frames 0..49 is really video frames 100..149.
    bouts = _bouts([("s1", 0, 49)])
    masks = {"s1": _mask(300, [(100, 149)])}

    with_off, _ = rbs.scope_bouts_to_roi(bouts, masks, {"s1": 100})
    without_off, _ = rbs.scope_bouts_to_roi(bouts, masks, {"s1": 0})

    assert len(with_off) == 1
    assert int(with_off.iloc[0]["start_frame"]) == 0   # returned still rebased
    assert int(with_off.iloc[0]["end_frame"]) == 49
    assert without_off.empty  # ignoring the offset finds nothing — the bug this guards


# ----------------------------------------------------------------------
# Unscopable sessions
# ----------------------------------------------------------------------

def test_session_without_a_mask_is_reported_not_silently_kept():
    bouts = _bouts([("s1", 0, 49), ("s2", 0, 49)])
    masks = {"s1": _mask(100, [(0, 49)]), "s2": None}

    out, missing = rbs.scope_bouts_to_roi(bouts, masks)

    assert missing == {"s2"}
    assert set(out["session_id"]) == {"s1"}


def test_bout_past_the_end_of_the_pose_track_is_dropped():
    bouts = _bouts([("s1", 500, 599)])
    masks = {"s1": _mask(100, [(0, 99)])}

    out, missing = rbs.scope_bouts_to_roi(bouts, masks)

    assert out.empty
    assert not missing


def test_empty_input_returns_empty():
    out, missing = rbs.scope_bouts_to_roi(pd.DataFrame(), {})
    assert out.empty
    assert not missing


# ----------------------------------------------------------------------
# roi_inside_mask
# ----------------------------------------------------------------------

class _FakePose:
    def __init__(self, xs, ys):
        self.centroid_x = np.asarray(xs, dtype=float)
        self.centroid_y = np.asarray(ys, dtype=float)


def test_inside_mask_respects_rect_geometry():
    roi = {"shape": "rect", "x": 0, "y": 0, "w": 10, "h": 10}
    pose = _FakePose([1, 1, 50, 50], [1, 1, 50, 50])

    mask = rbs.roi_inside_mask(pose, roi, fps=0.0)

    assert list(mask) == [True, True, False, False]


def test_inside_mask_debounces_single_frame_flicker():
    roi = {"shape": "rect", "x": 0, "y": 0, "w": 10, "h": 10}
    # One frame of tracking jitter outside the zone in the middle of a visit.
    xs = [1] * 10 + [50] + [1] * 10
    pose = _FakePose(xs, [1] * 21)

    mask = rbs.roi_inside_mask(pose, roi, fps=30.0)  # min_run = 6 frames

    assert mask.all()


def test_inside_mask_none_for_zero_area_roi():
    roi = {"shape": "rect", "x": 0, "y": 0, "w": 0, "h": 0}
    assert rbs.roi_inside_mask(_FakePose([1], [1]), roi, 30.0) is None


def test_inside_mask_none_without_pose():
    roi = {"shape": "rect", "x": 0, "y": 0, "w": 10, "h": 10}
    assert rbs.roi_inside_mask(None, roi, 30.0) is None


def test_inside_mask_supports_circles():
    roi = {"shape": "circle", "cx": 50.0, "cy": 50.0, "r": 10.0}
    pose = _FakePose([50, 50, 50], [50, 55, 90])

    mask = rbs.roi_inside_mask(pose, roi, fps=0.0)

    assert list(mask) == [True, True, False]


# ----------------------------------------------------------------------
# summarize_bouts
# ----------------------------------------------------------------------

def test_summarize_bouts_matches_hand_computed_metrics():
    bouts = _bouts([("s1", 30, 59), ("s1", 120, 149)])  # two 30-frame bouts

    stats = rbs.summarize_bouts(bouts, fps=30.0)

    n, time_s, mean_s, lat_s = stats[("s1", "b1")]
    assert n == 2
    assert time_s == pytest.approx(2.0)
    assert mean_s == pytest.approx(1.0)
    assert lat_s == pytest.approx(1.0)


def test_summarize_bouts_separates_behaviors():
    df = pd.concat([_bouts([("s1", 0, 29)], "b1"), _bouts([("s1", 0, 59)], "b2")])

    stats = rbs.summarize_bouts(df, fps=30.0)

    assert stats[("s1", "b1")][1] == pytest.approx(1.0)
    assert stats[("s1", "b2")][1] == pytest.approx(2.0)


def test_summarize_bouts_empty():
    assert rbs.summarize_bouts(pd.DataFrame(), 30.0) == {}
