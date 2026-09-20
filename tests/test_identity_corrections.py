"""Identity-swap corrections: track exchange, swap events, and label remapping."""

import numpy as np
import pandas as pd
import pytest

from abel.services.identity_remap_service import (
    IdentityRemapPlan,
    apply_identity_remap,
    parse_segment_id,
    permutation_at,
    plan_identity_remap,
)
from abel.services.pose_processing_service import (
    MultiAnimalPoseData,
    PoseData,
    PoseProcessingService,
)


def _pose(xs, ys=None, n=None):
    n = n if n is not None else len(xs)
    ys = ys if ys is not None else [0.0] * n
    x = pd.DataFrame({"nose": [float(v) for v in xs]})
    y = pd.DataFrame({"nose": [float(v) for v in ys]})
    return PoseData(
        body_parts=["nose"], x=x, y=y,
        likelihood=pd.DataFrame({"nose": [1.0] * n}),
        # Deliberately *not* the mean of the body parts: a correction must move
        # the centroids it was given, not recompute them under a different rule.
        centroid_x=np.asarray(xs, dtype=float) + 1000.0,
        centroid_y=np.asarray(ys, dtype=float),
        n_frames=n,
    )


def _multi(a_xs, b_xs):
    per = {"track_0": _pose(a_xs), "track_1": _pose(b_xs)}
    return MultiAnimalPoseData(
        individuals=list(per), per_individual=per, n_frames=len(a_xs)
    )


def test_correction_exchanges_tracks_from_the_frame_onward():
    multi = _multi([0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15])
    out = PoseProcessingService.apply_identity_corrections(
        multi, [{"frame": 3, "a": "track_0", "b": "track_1"}]
    )
    assert list(out.per_individual["track_0"].x["nose"]) == [0, 1, 2, 13, 14, 15]
    assert list(out.per_individual["track_1"].x["nose"]) == [10, 11, 12, 3, 4, 5]


def test_frames_before_the_correction_are_untouched():
    multi = _multi([0, 1, 2, 3], [10, 11, 12, 13])
    out = PoseProcessingService.apply_identity_corrections(
        multi, [{"frame": 2, "a": "track_0", "b": "track_1"}]
    )
    before = np.asarray(multi.per_individual["track_0"].centroid_x)[:2]
    after = np.asarray(out.per_individual["track_0"].centroid_x)[:2]
    assert np.array_equal(before, after)


def test_centroids_are_carried_over_not_recomputed():
    multi = _multi([0, 1, 2, 3], [10, 11, 12, 13])
    out = PoseProcessingService.apply_identity_corrections(
        multi, [{"frame": 2, "a": "track_0", "b": "track_1"}]
    )
    # 1000 + the *other* track's x, i.e. the value the source track carried.
    assert list(out.per_individual["track_0"].centroid_x) == [1000, 1001, 1012, 1013]


def test_a_frame_zero_correction_swaps_the_whole_session():
    multi = _multi([0, 1, 2], [10, 11, 12])
    out = PoseProcessingService.apply_identity_corrections(
        multi, [{"frame": 0, "a": "track_0", "b": "track_1"}]
    )
    assert list(out.per_individual["track_0"].x["nose"]) == [10, 11, 12]


def test_two_corrections_compose_back_to_the_original():
    multi = _multi([0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15])
    out = PoseProcessingService.apply_identity_corrections(
        multi,
        [
            {"frame": 2, "a": "track_0", "b": "track_1"},
            {"frame": 4, "a": "track_0", "b": "track_1"},
        ],
    )
    assert list(out.per_individual["track_0"].x["nose"]) == [0, 1, 12, 13, 4, 5]


def test_a_short_track_does_not_shift_the_other_identity():
    per = {"track_0": _pose([0, 1, 2, 3]), "track_1": _pose([10, 11])}
    multi = MultiAnimalPoseData(individuals=list(per), per_individual=per, n_frames=4)
    out = PoseProcessingService.apply_identity_corrections(
        multi, [{"frame": 2, "a": "track_0", "b": "track_1"}]
    )
    assert out.per_individual["track_0"].n_frames == 4
    assert out.per_individual["track_1"].n_frames == 4
    # track_1's own frames 0-1 survive; the exchanged tail is the missing part.
    assert list(out.per_individual["track_1"].x["nose"])[:2] == [10, 11]


# ── Swap events ───────────────────────────────────────────────────────
def test_swap_events_group_flips_and_report_net_parity():
    # Two animals walking apart, then their positions exchanged from frame 6.
    a = list(np.arange(0, 6) * 10.0) + list(np.arange(6, 12) * -10.0 + 200.0)
    b = list(np.arange(0, 6) * -10.0 + 200.0) + list(np.arange(6, 12) * 10.0)
    multi = _multi(a, b)
    out = PoseProcessingService.analyze_identity_swaps(multi)
    assert out["n_events"] >= 1
    ev = out["events"][0]
    assert ev["frame"] == 6
    assert ev["a"], ev["b"]
    assert ev["net"] is True
    assert set(ev) >= {"n_flips", "overlapping", "separation_px"}


def test_analysis_is_empty_without_swaps():
    multi = _multi(list(np.arange(20) * 3.0), list(np.arange(20) * 3.0 + 300.0))
    assert PoseProcessingService.analyze_identity_swaps(multi)["n_events"] == 0


# ── Label remapping ───────────────────────────────────────────────────
def test_parse_segment_id_keeps_underscored_animal_ids():
    assert parse_segment_id("seg_track_0_session_ab12_10_25", "session_ab12") == (
        "track_0", 10, 25,
    )
    assert parse_segment_id("seg_x_other_1_2", "session_ab12") is None


def test_permutation_only_counts_corrections_at_or_before_the_frame():
    inds = ["track_0", "track_1"]
    corr = [{"frame": 100, "a": "track_0", "b": "track_1"}]
    assert permutation_at(inds, corr, 99) == {"track_0": "track_0", "track_1": "track_1"}
    assert permutation_at(inds, corr, 100) == {"track_0": "track_1", "track_1": "track_0"}


def _plan(segs, old_corr=(), new_corr=(), old_map=None, new_map=None):
    return plan_identity_remap(
        session_id="session_ab12",
        individuals=["track_0", "track_1"],
        subject_key="M1",
        old_map=old_map if old_map is not None else {"track_0": "track_0", "track_1": "track_1"},
        new_map=new_map if new_map is not None else {"track_0": "track_0", "track_1": "track_1"},
        old_corrections=list(old_corr),
        new_corrections=list(new_corr),
        segment_ids=list(segs),
    )


def test_a_new_correction_moves_only_the_labels_after_its_frame():
    plan = _plan(
        ["seg_track_0_session_ab12_10_25", "seg_track_0_session_ab12_200_215"],
        new_corr=[{"frame": 100, "a": "track_0", "b": "track_1"}],
    )
    assert plan.n_labels == 2
    assert plan.segment_renames == {
        "seg_track_0_session_ab12_200_215": "seg_track_1_session_ab12_200_215"
    }


def test_renaming_an_individual_moves_all_of_its_labels():
    plan = _plan(
        ["seg_track_0_session_ab12_10_25"],
        new_map={"track_0": "green", "track_1": "black"},
    )
    assert plan.segment_renames == {
        "seg_track_0_session_ab12_10_25": "seg_green_session_ab12_10_25"
    }
    assert plan.animal_renames["seg_track_0_session_ab12_10_25"]["track_1"] == "black"


def test_a_window_containing_the_correction_frame_is_flagged():
    plan = _plan(
        ["seg_track_0_session_ab12_95_110"],
        new_corr=[{"frame": 100, "a": "track_0", "b": "track_1"}],
    )
    assert plan.straddling == ["seg_track_0_session_ab12_95_110"]


def test_removing_a_correction_moves_the_labels_back():
    plan = _plan(
        ["seg_track_0_session_ab12_200_215"],
        old_corr=[{"frame": 100, "a": "track_0", "b": "track_1"}],
        new_corr=[],
    )
    assert plan.segment_renames == {
        "seg_track_0_session_ab12_200_215": "seg_track_1_session_ab12_200_215"
    }


def test_no_change_means_no_plan():
    plan = _plan(["seg_track_0_session_ab12_10_25"])
    assert not plan
    assert plan.n_changed == 0


def test_apply_rewrites_the_label_store(tmp_path):
    pytest.importorskip("pyarrow")
    labels = tmp_path / "derived" / "review_labels"
    labels.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "segment_id": "seg_track_0_session_ab12_200_215",
                "review_label": "sniff",
                "reviewer_id": "me",
                "focal_animal_id": "track_0",
                "partner_animal_id": "track_1",
            },
            {
                "segment_id": "seg_track_0_session_ab12_10_25",
                "review_label": "rear",
                "reviewer_id": "me",
                "focal_animal_id": "track_0",
                "partner_animal_id": None,
            },
        ]
    ).to_parquet(labels / "reviewer_labels.parquet", index=False)

    plan: IdentityRemapPlan = _plan(
        [
            "seg_track_0_session_ab12_200_215",
            "seg_track_0_session_ab12_10_25",
        ],
        new_corr=[{"frame": 100, "a": "track_0", "b": "track_1"}],
    )
    counts = apply_identity_remap(tmp_path, plan)
    assert counts["labels"] == 1

    out = pd.read_parquet(labels / "reviewer_labels.parquet")
    moved = out[out["review_label"] == "sniff"].iloc[0]
    kept = out[out["review_label"] == "rear"].iloc[0]
    assert moved["segment_id"] == "seg_track_1_session_ab12_200_215"
    assert moved["focal_animal_id"] == "track_1"
    assert moved["partner_animal_id"] == "track_0"
    assert kept["segment_id"] == "seg_track_0_session_ab12_10_25"


def test_scattered_keypoints_mark_an_event_unreliable():
    # The tracks exchange positions at frame 6 (so the detector flags it), but
    # one animal's keypoints have flown apart there: a lost track, not a swap.
    n = 12
    centres_a = list(np.arange(0, 6) * 10.0) + list(np.arange(6, 12) * -10.0 + 200.0)
    centres_b = list(np.arange(0, 6) * -10.0 + 200.0) + list(np.arange(6, 12) * 10.0)
    per = {}
    for name, centres in (("track_0", centres_a), ("track_1", centres_b)):
        half = [5.0] * n
        if name == "track_0":
            half[6] = 400.0  # keypoints scattered across the whole arena
        x = pd.DataFrame({
            "nose": [c - h for c, h in zip(centres, half)],
            "tail": [c + h for c, h in zip(centres, half)],
        })
        y = pd.DataFrame({"nose": [0.0] * n, "tail": [0.0] * n})
        per[name] = PoseData(
            body_parts=["nose", "tail"], x=x, y=y,
            likelihood=pd.DataFrame({"nose": [1.0] * n, "tail": [1.0] * n}),
            centroid_x=np.asarray(centres, dtype=float),
            centroid_y=np.zeros(n),
            n_frames=n,
        )
    multi = MultiAnimalPoseData(individuals=list(per), per_individual=per, n_frames=n)
    events = PoseProcessingService.analyze_identity_swaps(multi)["events"]
    assert events and any(e["unreliable"] for e in events)


def test_overlays_follow_the_corrections(monkeypatch):
    from abel.services.preprocessing_service import ClipExtractionService

    multi = _multi([0, 1, 2, 3], [10, 11, 12, 13])

    class _Svc:
        def load_and_clean_multi(self, path, settings=None, identity_corrections=None):
            return PoseProcessingService.apply_identity_corrections(
                multi, identity_corrections or []
            )

    overlays = ClipExtractionService.build_individual_overlays(
        _Svc(), "pose.h5", None, {"track_0": "green", "track_1": "black"},
        [{"frame": 2, "a": "track_0", "b": "track_1"}],
    )
    green = next(o for o in overlays if o["name"] == "green")
    assert list(green["cx"]) == [1000, 1001, 1012, 1013]
