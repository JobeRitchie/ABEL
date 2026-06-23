"""Tests for the Behavior Grid montage: bout selection + grid stitching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from abel.services.validation_service import ValidationService

cv2 = pytest.importorskip("cv2")


# ---------------------------------------------------------------------------
# Synthetic two-session project
# ---------------------------------------------------------------------------
def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "config").mkdir(parents=True)
    (root / "derived" / "temporal_refinement" / "groom").mkdir(parents=True)
    inf_dir = root / "derived" / "inference"
    inf_dir.mkdir(parents=True)

    (root / "config" / "behavior_definitions.yaml").write_text(
        yaml.safe_dump(
            {"behaviors": [{
                "behavior_id": "groom", "name": "Grooming",
                "short_name": "groom", "is_active": True,
            }]}
        ),
        encoding="utf-8",
    )
    (root / "config" / "temporal_review_settings.json").write_text(
        json.dumps({"__all__": {
            "onset_threshold": 0.6, "min_bout_duration_frames": 3, "merge_gap_frames": 1,
        }}),
        encoding="utf-8",
    )

    trace_paths: dict[str, str] = {}
    # session_a: a strong bout (0.95) and a weaker one (0.7).
    probs_a = [0.1] * 60
    for f in range(10, 21):
        probs_a[f] = 0.95
    for f in range(30, 41):
        probs_a[f] = 0.70
    # session_b: a mid-strength bout (0.85).
    probs_b = [0.1] * 60
    for f in range(15, 26):
        probs_b[f] = 0.85
    for sid, probs in (("session_a", probs_a), ("session_b", probs_b)):
        df = pd.DataFrame({"frame": list(range(60)), "prob_groom": probs})
        p = inf_dir / f"{sid}__trace.parquet"
        df.to_parquet(p, index=False)
        trace_paths[sid] = str(p)

    (inf_dir / "inference_manifest.json").write_text(
        json.dumps({"trace_paths": trace_paths}), encoding="utf-8"
    )
    (root / "derived" / "temporal_refinement" / "groom" / "latest.json").write_text(
        json.dumps({"inference_dir": str(inf_dir), "postprocess_dir": ""}), encoding="utf-8"
    )
    return root


def _service(root: Path) -> ValidationService:
    svc = ValidationService()
    svc.set_project(root)
    return svc


# ---------------------------------------------------------------------------
# Bout selection
# ---------------------------------------------------------------------------
def test_select_grid_bouts_prefers_distinct_sessions(tmp_path: Path) -> None:
    svc = _service(_make_project(tmp_path))
    specs = svc.select_grid_bouts("groom", n_cells=2, top_fraction=1.0)
    assert len(specs) == 2
    # With two cells and two sessions, each session contributes once.
    assert {s.session_id for s in specs} == {"session_a", "session_b"}


def test_select_grid_bouts_top_fraction_keeps_confident(tmp_path: Path) -> None:
    svc = _service(_make_project(tmp_path))
    # Top 40% by mean prob: weakest bout (0.70 in session_a) should be excluded.
    specs = svc.select_grid_bouts("groom", n_cells=25, top_fraction=0.4)
    assert specs
    assert all(s.mean_prob >= 0.8 for s in specs)


def test_select_grid_bouts_no_same_session_overlap(tmp_path: Path) -> None:
    svc = _service(_make_project(tmp_path))
    specs = svc.select_grid_bouts("groom", n_cells=25, top_fraction=1.0)
    by_session: dict[str, list[tuple[int, int]]] = {}
    for s in specs:
        by_session.setdefault(s.session_id, []).append((s.bout_start, s.bout_end))
    for spans in by_session.values():
        spans.sort()
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            assert e1 < s2  # no frame overlap within a session


def test_select_grid_bouts_empty_behavior(tmp_path: Path) -> None:
    svc = _service(_make_project(tmp_path))
    assert svc.select_grid_bouts("not_a_behavior") == []


# ---------------------------------------------------------------------------
# Grid stitching
# ---------------------------------------------------------------------------
def _write_clip(path: Path, n_frames: int, cell_px: int, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 30.0, (cell_px, cell_px))
    for _ in range(n_frames):
        w.write(np.full((cell_px, cell_px, 3), value, dtype=np.uint8))
    w.release()


def test_stitch_grid_runs_to_longest_cell(tmp_path: Path) -> None:
    from abel.services import behavior_grid_render as render

    cell_px = 40
    grid_px = cell_px * 5
    paths: list[Path | None] = []
    # Two clips of differing length; the rest of the 25 cells are black.
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _write_clip(a, 5, cell_px, 200)
    _write_clip(b, 12, cell_px, 100)
    paths = [a, b] + [None] * 23

    out = tmp_path / "grid.mp4"
    render.stitch_grid(paths, grid_px, out)
    assert out.exists()

    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    assert n == 12  # length of the longest cell
    assert (w, h) == (grid_px, grid_px)
