"""HMM state bouts: decoding, merging, frame recovery, latency and export.

The HMM is fitted over the *bout index*, so its decoded states carry no clock.
Everything here guards the step that puts the clock back on — an off-by-one in
that mapping would silently attribute every state to the wrong stretch of the
session, and nothing downstream (ethogram, latency, TRACY workbook) could
detect it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.services.behavioral_motif_service import (
    MotifSettings,
    build_sequences,
    decode_state_spans,
    default_state_label,
    fit_hmm,
    merge_state_bouts,
    retained_events,
    state_bout_rows,
    state_bout_summary,
    state_bout_transition_counts,
    state_bouts_to_frames,
    state_entry_latency,
)
from abel.services.export_service import ExportService

FPS = 30.0


def _spans(items: list[tuple[float, float, int]], sid: str = "s1") -> dict[str, list[dict]]:
    """Build a state-span dict from (start_s, end_s, state) triples."""
    return {
        sid: [
            {
                "bout_index": i,
                "start_s": float(a),
                "end_s": float(b),
                "behavior_id": "b0",
                "state": int(st),
            }
            for i, (a, b, st) in enumerate(items)
        ]
    }


# ---------------------------------------------------------------------------
# Decoding: states must land on the bouts they came from
# ---------------------------------------------------------------------------

def test_retained_events_drops_unselected_behaviors_and_empty_sessions():
    sequences = {
        "s1": [(0.0, 1.0, "b0"), (2.0, 3.0, "bX"), (4.0, 5.0, "b1")],
        "s2": [(0.0, 1.0, "bX")],
    }
    kept = retained_events(sequences, ["b0", "b1"])
    assert list(kept) == ["s1"]
    assert [e[2] for e in kept["s1"]] == ["b0", "b1"]


def test_decode_state_spans_aligns_states_with_bout_times():
    """A behavior outside the fit must not shift the states that follow it.

    This is the regression that motivated routing both the encoder and the
    decoder through ``retained_events``: the HMM never saw ``bX``, so state 1
    belongs to the bout at t=4, not the one at t=2.
    """
    sequences = {"s1": [(0.0, 1.0, "b0"), (2.0, 3.0, "bX"), (4.0, 5.0, "b1")]}
    spans = decode_state_spans(sequences, ["b0", "b1"], {"s1": [0, 1]})
    assert [(s["start_s"], s["behavior_id"], s["state"]) for s in spans["s1"]] == [
        (0.0, "b0", 0),
        (4.0, "b1", 1),
    ]


def test_decode_state_spans_truncates_on_length_mismatch(caplog):
    sequences = {"s1": [(0.0, 1.0, "b0"), (2.0, 3.0, "b0"), (4.0, 5.0, "b0")]}
    spans = decode_state_spans(sequences, ["b0"], {"s1": [0, 1]})
    assert len(spans["s1"]) == 2


# ---------------------------------------------------------------------------
# Merging into state bouts
# ---------------------------------------------------------------------------

def test_merge_collapses_same_state_runs_and_spans_the_gaps():
    spans = _spans([(0.0, 1.0, 0), (5.0, 6.0, 0), (10.0, 11.0, 1)])
    bouts = merge_state_bouts(spans)["s1"]
    assert [b["state"] for b in bouts] == [0, 1]
    # The state bout runs first-start to last-end, so the 4 s silence between
    # the two state-0 bouts is inside it: duration is not the members' sum.
    assert bouts[0]["start_s"] == 0.0 and bouts[0]["end_s"] == 6.0
    assert bouts[0]["duration_s"] == 6.0
    assert bouts[0]["n_bouts"] == 2


def test_merge_max_gap_breaks_a_run_of_one_state():
    spans = _spans([(0.0, 1.0, 0), (100.0, 101.0, 0)])
    assert len(merge_state_bouts(spans)["s1"]) == 1
    broken = merge_state_bouts(spans, max_gap_s=10.0)["s1"]
    assert len(broken) == 2
    assert [b["state"] for b in broken] == [0, 0]


def test_merge_preserves_every_bout():
    spans = _spans([(0.0, 1.0, 0), (2.0, 3.0, 1), (4.0, 5.0, 1), (6.0, 7.0, 0)])
    bouts = merge_state_bouts(spans)["s1"]
    assert sum(b["n_bouts"] for b in bouts) == 4


def test_state_bout_transitions_have_a_zero_diagonal():
    """Distinct from the model's transition matrix, which is diagonal-heavy."""
    spans = _spans([(0.0, 1.0, 0), (2.0, 3.0, 1), (4.0, 5.0, 0)])
    mat = state_bout_transition_counts(merge_state_bouts(spans), 2)["s1"]
    assert np.trace(mat) == 0
    assert mat[0, 1] == 1 and mat[1, 0] == 1


def test_state_bout_summary_fractions_use_covered_time():
    spans = _spans([(0.0, 10.0, 0), (20.0, 25.0, 1)])
    summ = state_bout_summary(merge_state_bouts(spans), 2)["s1"]
    assert summ["total_s"] == [10.0, 5.0]
    assert summ["covered_s"] == 15.0
    # The 10 s silence between the state bouts belongs to no state and is not
    # charged to either.
    assert summ["time_frac"] == pytest.approx([10 / 15, 5 / 15])


# ---------------------------------------------------------------------------
# Frame recovery — what TRACY reads
# ---------------------------------------------------------------------------

def test_frames_round_trip_exactly_and_cover_every_state():
    spans = _spans([(1800 / FPS, 1830 / FPS, 0), (2400 / FPS, 2460 / FPS, 1)])
    frames = state_bouts_to_frames(merge_state_bouts(spans), FPS, n_states=3)
    assert set(frames["s1"]) == {default_state_label(i) for i in range(3)}
    assert frames["s1"][default_state_label(0)] == [(1800, 1830)]
    assert frames["s1"][default_state_label(1)] == [(2400, 2460)]
    # A state with no bouts still gets a column, so sheets stay comparable.
    assert frames["s1"][default_state_label(2)] == []


def test_frame_offsets_undo_the_analysis_prechop():
    """Exports must be in raw video frames — the numbering TRACY aligns on."""
    spans = _spans([(0.0, 1.0, 0)])
    bouts = merge_state_bouts(spans)
    plain = state_bouts_to_frames(bouts, FPS, 1)["s1"][default_state_label(0)]
    shifted = state_bouts_to_frames(
        bouts, FPS, 1, frame_offsets={"s1": 900}
    )["s1"][default_state_label(0)]
    assert plain == [(0, 30)]
    assert shifted == [(900, 930)]


def test_zero_fps_is_refused_rather_than_producing_frame_zero():
    with pytest.raises(ValueError):
        state_bouts_to_frames(merge_state_bouts(_spans([(0.0, 1.0, 0)])), 0.0, 1)


def test_state_bout_rows_carry_both_seconds_and_frames():
    bouts = merge_state_bouts(_spans([(2.0, 4.0, 1)]))
    row = state_bout_rows(bouts, FPS, frame_offsets={"s1": 60},
                          session_labels={"s1": "M1"},
                          session_groups={"M1": "CTRL"})[0]
    assert row["session_label"] == "M1" and row["group"] == "CTRL"
    assert row["start_s"] == 2.0 and row["start_frame"] == 120
    assert row["end_frame"] == 180


# ---------------------------------------------------------------------------
# Latency to enter a state
# ---------------------------------------------------------------------------

def test_latency_is_measured_to_the_first_qualifying_state_bout():
    spans = _spans([(5.0, 6.0, 1), (20.0, 60.0, 1)])
    # Without a floor the 1 s flicker at t=5 sets the latency...
    loose = state_entry_latency(merge_state_bouts(spans, max_gap_s=5.0), 2)["s1"]
    assert loose["latency_s"][1] == 5.0
    # ...with one, the state counts as entered only at the 40 s bout.
    strict = state_entry_latency(
        merge_state_bouts(spans, max_gap_s=5.0), 2, min_dwell_s=10.0
    )["s1"]
    assert strict["latency_s"][1] == 20.0
    assert strict["n_entries"][1] == 1


def test_latency_is_nan_for_a_state_never_entered():
    """Never-entered must not be censored to the session length.

    Substituting the session end would turn an unobserved latency into a
    measured one and bias any group mean toward whoever was recorded longest.
    """
    lat = state_entry_latency(merge_state_bouts(_spans([(0.0, 1.0, 0)])), 2,
                              censor_at_s={"s1": 600.0})["s1"]
    assert np.isnan(lat["latency_s"][1])
    assert lat["entered"] == [True, False]
    assert lat["observed_s"] == 600.0


def test_latency_is_relative_to_the_assay_start():
    lat = state_entry_latency(merge_state_bouts(_spans([(70.0, 80.0, 0)])), 1,
                              session_start_s={"s1": 60.0})["s1"]
    assert lat["latency_s"][0] == 10.0


def test_a_stricter_dwell_floor_never_reports_an_earlier_entry():
    spans = _spans([(1.0, 2.0, 0), (10.0, 40.0, 0), (50.0, 51.0, 1), (60.0, 90.0, 1)])
    bouts = merge_state_bouts(spans, max_gap_s=5.0)
    loose = state_entry_latency(bouts, 2)["s1"]["latency_s"]
    strict = state_entry_latency(bouts, 2, min_dwell_s=20.0)["s1"]["latency_s"]
    for a, b in zip(loose, strict):
        assert np.isnan(b) or b >= a


# ---------------------------------------------------------------------------
# Export — must be the same workbook shape TRACY already reads
# ---------------------------------------------------------------------------

@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "exports").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _service(project: Path) -> ExportService:
    svc = ExportService()
    svc.set_project(project)
    return svc


def test_state_boutframes_workbook_has_one_sheet_per_subject(project: Path):
    frames = {
        "s1": {"HMM_State_0": [(0, 30)], "HMM_State_1": [(60, 90)]},
        "s2": {"HMM_State_0": [(10, 40)], "HMM_State_1": []},
    }
    out = _service(project).export_state_boutframes_xlsx(
        frames, column_order=["HMM_State_0", "HMM_State_1"]
    )
    assert out.success, out.warnings
    book = pd.read_excel(out.output_path, sheet_name=None)
    assert {"s1", "s2", "_bout_counts"} <= set(book)
    assert list(book["s1"].columns) == ["HMM_State_0", "HMM_State_1"]
    assert book["s1"]["HMM_State_0"].tolist() == [0]


def test_state_boutframes_matches_the_behavior_export_column_shape(project: Path):
    """The two exports share one writer, so TRACY reads them identically."""
    frames = {"s1": {"HMM_State_0": [(0, 30)]}}
    out = _service(project).export_state_boutframes_xlsx(
        frames, column_order=["HMM_State_0"], include_end_frames=True
    )
    df = pd.read_excel(out.output_path, sheet_name="s1")
    assert list(df.columns) == ["HMM_State_0__start", "HMM_State_0__end"]
    assert df.iloc[0].tolist() == [0, 30]


def test_state_boutframes_binary_mode_is_one_row_per_frame(project: Path):
    frames = {"s1": {"HMM_State_0": [(0, 2)], "HMM_State_1": [(4, 5)]}}
    out = _service(project).export_state_boutframes_xlsx(
        frames, column_order=["HMM_State_0", "HMM_State_1"], binary_mode=True
    )
    df = pd.read_excel(out.output_path, sheet_name="s1")
    assert list(df.columns) == ["frame", "HMM_State_0", "HMM_State_1"]
    assert df["HMM_State_0"].tolist() == [1, 1, 1, 0, 0, 0]
    assert df["HMM_State_1"].tolist() == [0, 0, 0, 0, 1, 1]


def test_state_boutframes_reports_empty_input_instead_of_writing_a_blank_book(project: Path):
    out = _service(project).export_state_boutframes_xlsx({})
    assert not out.success
    assert "No state bouts" in " ".join(out.warnings)


def test_behavior_boutframes_still_writes_after_the_writer_refactor(project: Path):
    """The shared writer must not have changed the existing export's output."""
    from abel.models.schemas import CandidateWindow, ReviewDecision, ReviewDecisionType

    cands = [
        CandidateWindow(
            window_id="c1", session_id="s1", start_frame=10, end_frame=20,
            behavior_id="Rear",
        )
    ]
    decs = [
        ReviewDecision(
            decision_id="d1", clip_id="c1", reviewer="tester",
            old_status="pending", new_status="accepted",
            decision=ReviewDecisionType.ACCEPT, behavior_label="Rear",
            adjusted_start_frame=10, adjusted_end_frame=20,
        )
    ]
    out = _service(project).export_boutframes_xlsx(
        cands, decs, include_merged_projects=False
    )
    assert out.success, out.warnings
    df = pd.read_excel(out.output_path, sheet_name="s1")
    assert df.iloc[0].tolist() == [10]


# ---------------------------------------------------------------------------
# Whole chain, on a real fit
# ---------------------------------------------------------------------------

def test_full_chain_from_fitted_hmm_to_workbook(project: Path):
    pytest.importorskip("hmmlearn")
    rng = np.random.default_rng(3)
    bids = [f"b{i}" for i in range(4)]
    rows: dict[str, list[dict]] = {b: [] for b in bids}
    for s in range(5):
        frame = 300
        for i in range(80):
            b = bids[int(rng.choice((0, 1) if i < 40 else (2, 3)))]
            dur = int(rng.integers(15, 45))
            rows[b].append(
                {"session_id": f"sess{s}", "start_frame": frame, "end_frame": frame + dur}
            )
            frame += dur + int(rng.integers(10, 60))

    sequences = build_sequences({b: pd.DataFrame(v) for b, v in rows.items()}, FPS)
    res = fit_hmm(
        sequences, bids,
        MotifSettings(hmm_n_states_mode="manual", hmm_n_states=2,
                      hmm_n_restarts=2, hmm_random_seed=0),
    )
    assert not res.get("error"), res.get("error")

    spans = decode_state_spans(sequences, bids, res["state_sequences"])
    for sid, evs in sequences.items():
        assert len(spans[sid]) == len(evs)

    bouts = merge_state_bouts(spans)
    frames = state_bouts_to_frames(bouts, FPS, res["n_states"])
    starts = {int(round(e[0] * FPS)) for e in sequences["sess0"]}
    for ivs in frames["sess0"].values():
        for start, end in ivs:
            assert start in starts
            assert end >= start

    out = _service(project).export_state_boutframes_xlsx(
        frames, column_order=[default_state_label(i) for i in range(res["n_states"])]
    )
    assert out.success, out.warnings
    assert len(pd.read_excel(out.output_path, sheet_name=None)) == len(bouts) + 1


def test_state_boutframes_honours_an_explicit_output_directory(tmp_path: Path, project: Path):
    """The save dialog's location must be where the workbook lands."""
    elsewhere = tmp_path / "picked"
    elsewhere.mkdir()
    out = _service(project).export_state_boutframes_xlsx(
        {"s1": {"HMM_State_0": [(0, 5)]}},
        filename="picked.xlsx",
        column_order=["HMM_State_0"],
        out_dir=elsewhere,
    )
    assert out.success, out.warnings
    assert out.output_path == elsewhere / "picked.xlsx"
    assert not (project / "exports" / "reports" / "picked.xlsx").exists()


# ---------------------------------------------------------------------------
# Persistence — the fit is cached so it is not re-run every session
# ---------------------------------------------------------------------------

def _fake_result() -> dict:
    return {
        "n_states": 2,
        "transition_matrix": [[0.9, 0.1], [0.2, 0.8]],
        "aic": float("nan"),
        "log_likelihood": -12.5,
        "state_bouts": {"s1": [{"state": 0, "start_s": 0.0, "end_s": 4.0,
                               "duration_s": 4.0, "n_bouts": 2,
                               "behavior_ids": ["b0", "b0"],
                               "first_bout_index": 0, "last_bout_index": 1}]},
        "group_occ_mean": {"CTRL": np.array([0.6, 0.4])},
        "input_fingerprint": "abc123",
    }


def test_saved_fit_round_trips_including_numpy_and_nan(tmp_path: Path):
    from abel.services.behavioral_motif_service import load_hmm_result, save_hmm_result

    assert save_hmm_result(tmp_path, _fake_result()) is not None
    back = load_hmm_result(tmp_path)
    assert back["n_states"] == 2
    assert back["transition_matrix"] == [[0.9, 0.1], [0.2, 0.8]]
    # numpy arrays the UI adds must survive as plain lists...
    assert back["group_occ_mean"]["CTRL"] == [0.6, 0.4]
    # ...and NaN must stay NaN, not become None: the stats panel formats it
    # with :.2f and None would raise there on every restored fit.
    assert np.isnan(back["aic"])
    assert back["state_bouts"]["s1"][0]["duration_s"] == 4.0
    assert back["restored_from_cache"] is True
    assert back["saved_at"]


def test_loading_returns_empty_when_there_is_no_cache(tmp_path: Path):
    from abel.services.behavioral_motif_service import load_hmm_result

    assert load_hmm_result(tmp_path) == {}


def test_a_corrupt_cache_is_ignored_rather_than_raising(tmp_path: Path):
    from abel.services.behavioral_motif_service import hmm_result_path, load_hmm_result

    path = hmm_result_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_hmm_result(tmp_path) == {}


def test_clear_removes_the_cache_and_tolerates_a_missing_one(tmp_path: Path):
    from abel.services.behavioral_motif_service import (
        clear_hmm_result, hmm_result_path, load_hmm_result, save_hmm_result,
    )

    save_hmm_result(tmp_path, _fake_result())
    clear_hmm_result(tmp_path)
    assert not hmm_result_path(tmp_path).exists()
    clear_hmm_result(tmp_path)  # no-op, must not raise
    assert load_hmm_result(tmp_path) == {}


def test_fingerprint_is_stable_and_reacts_to_each_input():
    from abel.services.behavioral_motif_service import hmm_input_fingerprint

    seq = {"s1": [(0.0, 1.0, "b0"), (2.0, 3.0, "b1")]}
    st = MotifSettings(hmm_n_states=3)
    base = hmm_input_fingerprint(seq, ["b0", "b1"], st)

    assert base == hmm_input_fingerprint(seq, ["b1", "b0"], st), "behavior order is not data"
    assert base != hmm_input_fingerprint(
        {"s1": [(0.0, 1.0, "b0"), (2.5, 3.0, "b1")]}, ["b0", "b1"], st
    ), "a shifted bout must invalidate the cache"
    assert base != hmm_input_fingerprint(seq, ["b0"], st)
    assert base != hmm_input_fingerprint(seq, ["b0", "b1"], MotifSettings(hmm_n_states=4))
    assert base != hmm_input_fingerprint(
        seq, ["b0", "b1"], st, session_groups={"s1": "CTRL"}
    )
    assert base != hmm_input_fingerprint(
        {"s1": seq["s1"], "s2": [(0.0, 1.0, "b0")]}, ["b0", "b1"], st
    )


def test_display_only_settings_do_not_invalidate_a_fit():
    """Changing an error-bar style must not force a re-fit."""
    from abel.services.behavioral_motif_service import hmm_input_fingerprint

    seq = {"s1": [(0.0, 1.0, "b0")]}
    a = hmm_input_fingerprint(seq, ["b0"], MotifSettings(ngram_top_k=15, umap_n_neighbors=10))
    b = hmm_input_fingerprint(seq, ["b0"], MotifSettings(ngram_top_k=30, umap_n_neighbors=25))
    assert a == b
