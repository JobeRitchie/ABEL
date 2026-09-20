"""Review-tab clip resolution must not stat the filesystem per candidate.

An AL run persists every window it ranked, not just the ones it cut clips for,
so the candidate list runs to hundreds of thousands of rows while only a few
hundred have clips.  Resolving each row with ``Path.exists()`` probes made a
single filter pass cost ~40 s of blocking stats on a real project.  These tests
pin the index-backed behavior: same answers, one directory scan.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from abel.services.candidate_service import CandidateGenerationService
from abel.ui.tabs.review_tab import ReviewTab


class _Candidate:
    """The three attributes the resolver reads."""

    def __init__(self, window_id: str, session_id: str, clip_path: str | None = None) -> None:
        self.window_id = window_id
        self.session_id = session_id
        self.clip_path = clip_path


class _Resolver:
    """The resolver's methods without building a Qt widget."""

    _clips_root = ReviewTab._clips_root
    _ensure_clip_index = ReviewTab._ensure_clip_index
    _clip_index_key = staticmethod(ReviewTab._clip_index_key)
    _candidate_clip_path = ReviewTab._candidate_clip_path

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._clip_index: set[str] | None = None


@pytest.fixture
def project(tmp_path: Path) -> Path:
    clips = tmp_path / "derived" / "clips"
    (clips / "session_aaa").mkdir(parents=True)
    (clips / "session_bbb").mkdir(parents=True)
    (clips / "session_aaa" / "win1.mp4").write_bytes(b"x")
    (clips / "session_bbb" / "win2.mp4").write_bytes(b"x")
    return tmp_path


def test_resolves_clip_by_window_id(project: Path) -> None:
    r = _Resolver(project)
    got = r._candidate_clip_path(_Candidate("win1", "session_aaa"))
    assert got is not None
    assert Path(got).name == "win1.mp4"


def test_missing_clip_resolves_to_none(project: Path) -> None:
    r = _Resolver(project)
    assert r._candidate_clip_path(_Candidate("nope", "session_aaa")) is None
    assert r._candidate_clip_path(_Candidate("win1", "session_missing")) is None


def test_explicit_in_tree_clip_path_is_honoured(project: Path) -> None:
    r = _Resolver(project)
    explicit = str(project / "derived" / "clips" / "session_bbb" / "win2.mp4")
    assert r._candidate_clip_path(_Candidate("win2", "session_bbb", explicit)) == explicit


def test_explicit_path_outside_clips_tree_falls_back_to_stat(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "clip.mp4"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"x")
    r = _Resolver(project)
    assert r._candidate_clip_path(_Candidate("w", "session_aaa", str(outside))) == str(outside)
    gone = str(tmp_path / "elsewhere" / "not_there.mp4")
    assert r._candidate_clip_path(_Candidate("w", "session_aaa", gone)) is None


def test_bulk_resolution_scans_each_directory_once(project: Path, monkeypatch) -> None:
    """The whole point: candidate count must not drive filesystem calls."""
    calls: list[str] = []
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: calls.append(str(self)) or real_exists(self))

    r = _Resolver(project)
    rows = [_Candidate(f"w{i}", "session_aaa") for i in range(500)]
    resolved = [c for c in rows if r._candidate_clip_path(c)]

    assert len(resolved) == 0  # none of w0..w499 exist
    # Without the index this would be one or more exists() per row.
    assert not calls, f"expected no per-row exists() probes, saw {len(calls)}"


def test_verify_finds_a_clip_created_after_the_index_was_built(project: Path) -> None:
    r = _Resolver(project)
    r._ensure_clip_index()  # snapshot before the clip appears
    late = project / "derived" / "clips" / "session_aaa" / "late.mp4"
    late.write_bytes(b"x")

    assert r._candidate_clip_path(_Candidate("late", "session_aaa")) is None
    assert r._candidate_clip_path(_Candidate("late", "session_aaa"), verify=True) == str(late)
    # A verify hit drops the stale snapshot so the next bulk pass is correct.
    assert r._clip_index is None


def test_missing_clips_directory_is_not_an_error(tmp_path: Path) -> None:
    r = _Resolver(tmp_path)
    assert r._candidate_clip_path(_Candidate("w", "session_aaa")) is None
    assert r._clip_index == set()


def test_external_candidates_are_not_revalidated_when_unchanged(tmp_path: Path) -> None:
    """Repeated Review-tab refreshes must not re-parse the whole queue."""
    import json

    tables = tmp_path / "derived" / "review_tables"
    tables.mkdir(parents=True)
    rows = [
        {
            "window_id": f"seg_{i}",
            "session_id": "session_aaa",
            "start_frame": i,
            "end_frame": i + 10,
        }
        for i in range(50)
    ]
    path = tables / "external_window_candidates.json"
    path.write_text(json.dumps({"candidates": rows}), encoding="utf-8")
    # The loader refuses to cache a file written in the last couple of seconds,
    # because exFAT mtimes are second-granular and a same-second rewrite would
    # be invisible.  Backdate it so the caching path is the one under test.
    old_time = time.time() - 60
    os.utime(path, (old_time, old_time))

    svc = CandidateGenerationService()
    svc.set_project(tmp_path)
    first = svc.load_external_window_candidates()
    assert len(first) == 50

    parses = {"n": 0}
    real_validate = type(first[0]).model_validate

    def counting_validate(cls_arg):  # pragma: no cover - only called on a miss
        parses["n"] += 1
        return real_validate(cls_arg)

    import abel.services.candidate_service as cs

    monkey = getattr(cs, "CandidateWindow")
    original = monkey.model_validate
    monkey.model_validate = staticmethod(counting_validate)
    try:
        again = svc.load_external_window_candidates()
        assert len(again) == 50
        assert parses["n"] == 0, "cached load should not re-validate any row"

        # A write through the service must drop the cache.
        svc.remove_external_window_candidates(["seg_0"])
        after = svc.load_external_window_candidates()
        assert len(after) == 49
        assert parses["n"] > 0, "a changed file must be re-read"
    finally:
        monkey.model_validate = original


def test_freshly_written_queue_is_not_cached(tmp_path: Path) -> None:
    """exFAT mtimes are second-granular, so a just-written file is re-read."""
    import json

    tables = tmp_path / "derived" / "review_tables"
    tables.mkdir(parents=True)
    path = tables / "external_window_candidates.json"
    path.write_text(
        json.dumps({"candidates": [{"window_id": "a", "session_id": "s", "start_frame": 0, "end_frame": 1}]}),
        encoding="utf-8",
    )

    svc = CandidateGenerationService()
    svc.set_project(tmp_path)
    assert len(svc.load_external_window_candidates()) == 1
    # Written moments ago, so nothing was cached and a rewrite is picked up
    # even though the mtime second and the row count are unchanged.
    assert svc._external_cache_stamp is None

    path.write_text(
        json.dumps({"candidates": [{"window_id": "b", "session_id": "s", "start_frame": 0, "end_frame": 1}]}),
        encoding="utf-8",
    )
    assert [c.window_id for c in svc.load_external_window_candidates()] == ["b"]
