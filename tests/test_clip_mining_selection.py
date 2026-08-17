"""Subject-balanced selection of targeted clip-mining matches."""

from __future__ import annotations

from abel.services.clip_metrics_service import select_even_by_group


def _subject_of(mapping: dict[str, str]):
    return lambda wid: mapping[wid]


def test_even_split_across_subjects() -> None:
    """A cap smaller than the match count draws equally from every subject."""
    ids = [f"m{s}_{i}" for s in range(4) for i in range(50)]
    subj = {w: w.split("_")[0] for w in ids}
    # Score order alone would take all 50 of m0 before touching m1.
    ordered = sorted(ids, key=lambda w: (w.split("_")[0], int(w.split("_")[1])))

    out = select_even_by_group(ordered, _subject_of(subj), 20)

    assert len(out) == 20
    counts = {s: sum(1 for w in out if subj[w] == s) for s in {"m0", "m1", "m2", "m3"}}
    assert set(counts.values()) == {5}


def test_short_subjects_give_their_slots_back() -> None:
    """A subject with few matches never costs the batch its cap."""
    ids = ["a1", "a2", "a3", "a4", "a5", "b1"]
    subj = {w: w[0] for w in ids}

    out = select_even_by_group(ids, _subject_of(subj), 5)

    assert len(out) == 5
    assert out[:2] == ["a1", "b1"]  # round-robin starts even
    assert sum(1 for w in out if subj[w] == "a") == 4


def test_best_match_stays_first_and_order_holds_within_subject() -> None:
    """Within a subject the score order is preserved; the top match leads."""
    ids = ["a1", "b1", "a2", "b2", "a3"]
    subj = {w: w[0] for w in ids}

    out = select_even_by_group(ids, _subject_of(subj), 4)

    assert out[0] == "a1"
    assert [w for w in out if subj[w] == "a"] == ["a1", "a2"]
    assert [w for w in out if subj[w] == "b"] == ["b1", "b2"]


def test_single_subject_is_a_plain_top_n_cut() -> None:
    ids = [f"a{i}" for i in range(10)]
    out = select_even_by_group(ids, lambda _w: "a", 3)
    assert out == ["a0", "a1", "a2"]


def test_cap_at_or_above_match_count_returns_everything() -> None:
    ids = ["a1", "b1", "a2"]
    subj = {w: w[0] for w in ids}
    out = select_even_by_group(ids, _subject_of(subj), 99)
    assert sorted(out) == sorted(ids)


def test_empty_inputs() -> None:
    assert select_even_by_group([], lambda w: "a", 5) == []
    assert select_even_by_group(["a1"], lambda w: "a", 0) == []
