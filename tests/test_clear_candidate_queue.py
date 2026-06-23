"""Clear Candidates must clear only the pending queue, preserving reviewed work.

The Clip tab's "Clear Candidates" button calls
``CandidateGenerationService.clear_candidate_queue(preserve_reviewed=True)``.
It must:
  * remove pending (unreviewed) windows from the candidate queue,
  * keep windows that have already been reviewed (present in reviewer_labels),
  * never delete reviewer labels or rendered clip files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from abel.services.candidate_service import CandidateGenerationService
from abel.storage.file_store import read_json, write_json


def _candidate(seg_id: str) -> dict:
    return {
        "segment_id": seg_id,
        "start_frame": 0,
        "end_frame": 30,
        "video_id": "v1",
        "animal_id": "a1",
        "session_id": "s1",
    }


def _setup(tmp_path: Path, reviewed: list[str], queue: list[str]):
    project = tmp_path / "proj"
    (project / "derived" / "review_tables").mkdir(parents=True)
    (project / "derived" / "review_labels").mkdir(parents=True)
    clips = project / "derived" / "clips" / "s1"
    clips.mkdir(parents=True)
    clip_file = clips / "reviewed_clip.mp4"
    clip_file.write_bytes(b"fake-clip")

    write_json(
        project / "derived" / "review_tables" / "candidate_segments.json",
        {"candidates": [_candidate(s) for s in queue], "config": {}},
    )
    if reviewed:
        pd.DataFrame(
            {"segment_id": reviewed, "review_label": ["accept"] * len(reviewed)}
        ).to_parquet(
            project / "derived" / "review_labels" / "reviewer_labels.parquet",
            index=False,
        )

    svc = CandidateGenerationService()
    svc.set_project(project)
    return project, svc, clip_file


def test_keeps_reviewed_removes_pending(tmp_path: Path):
    reviewed = ["s1_0_30", "s1_60_90"]
    pending = ["s1_120_150", "s1_180_210"]
    project, svc, clip_file = _setup(tmp_path, reviewed, reviewed + pending)

    summary = svc.clear_candidate_queue(preserve_reviewed=True)

    assert summary["removed"] == 2
    assert summary["kept_reviewed"] == 2

    remaining = {
        c["segment_id"]
        for c in read_json(
            project / "derived" / "review_tables" / "candidate_segments.json",
            {"candidates": []},
        )["candidates"]
    }
    assert remaining == set(reviewed)

    # Reviewer labels and clip files are untouched.
    assert (project / "derived" / "review_labels" / "reviewer_labels.parquet").exists()
    assert clip_file.exists() and clip_file.read_bytes() == b"fake-clip"


def test_all_pending_removes_queue_file(tmp_path: Path):
    project, svc, _ = _setup(tmp_path, reviewed=[], queue=["s1_0_30", "s1_60_90"])
    summary = svc.clear_candidate_queue(preserve_reviewed=True)
    assert summary["removed"] == 2
    assert summary["kept_reviewed"] == 0
    # With nothing reviewed to keep, the queue file is removed entirely.
    assert not (
        project / "derived" / "review_tables" / "candidate_segments.json"
    ).exists()


def test_reviewed_segment_ids_reads_labels(tmp_path: Path):
    project, svc, _ = _setup(
        tmp_path, reviewed=["s1_0_30"], queue=["s1_0_30", "s1_60_90"]
    )
    assert svc.reviewed_segment_ids() == {"s1_0_30"}
