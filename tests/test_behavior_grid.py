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


def test_select_grid_bouts_prefers_confident_then_backfills(tmp_path: Path) -> None:
    svc = _service(_make_project(tmp_path))
    # 3 detected bouts: session_a (0.95, 0.70), session_b (0.85). top_fraction=0.4
    # keeps only 0.95 as "confident", but the grid backfills with the next-
    # strongest bouts until full rather than leaving cells blank — so all three
    # are used and the weakest (0.70) appears only after the strong ones.
    specs = svc.select_grid_bouts("groom", n_cells=25, top_fraction=0.4)
    assert len(specs) == 3
    probs = [round(s.mean_prob, 2) for s in specs]
    assert min(probs) == pytest.approx(0.70, abs=1e-3)  # weakest backfilled in
    # Within session_a the strong (0.95) bout is chosen before the weak (0.70).
    a_probs = [round(s.mean_prob, 2) for s in specs if s.session_id == "session_a"]
    assert a_probs == [0.95, 0.70]


def test_select_grid_bouts_top_fraction_orders_confident_first(tmp_path: Path) -> None:
    svc = _service(_make_project(tmp_path))
    # With a single confident bout requested, only the strongest is returned.
    specs = svc.select_grid_bouts("groom", n_cells=1, top_fraction=0.4)
    assert len(specs) == 1
    assert specs[0].mean_prob >= 0.9


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


def test_select_grid_bouts_banded_orders_rows_high_to_low(tmp_path: Path) -> None:
    svc = _service(_make_project(tmp_path))
    # Three detected bouts at 0.95, 0.85, 0.70. With a 3-row × 1-col grid each row
    # is a distinct probability band, ordered top (highest) → bottom (lowest).
    specs = svc.select_grid_bouts("groom", layout="bands", rows=3, cols=1)
    assert len(specs) == 3
    probs = [round(s.mean_prob, 2) for s in specs]
    assert probs == sorted(probs, reverse=True)
    assert probs[0] == pytest.approx(0.95, abs=1e-3)
    assert probs[-1] == pytest.approx(0.70, abs=1e-3)


def _add_subject_manifest(root: Path, mapping: dict[str, str]) -> None:
    """Write a minimal import manifest mapping session_id -> subject_id."""
    path = root / "derived" / "review_tables" / "import_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    sessions = [
        {
            "session_id": sid,
            "video_asset_id": f"v_{sid}",
            "pose_asset_id": f"p_{sid}",
            "subject_id": subject,
        }
        for sid, subject in mapping.items()
    ]
    path.write_text(json.dumps({"linked_sessions": sessions}), encoding="utf-8")


def _make_three_session_project(tmp_path: Path) -> Path:
    """Three sessions, each with one strong bout (no top-fraction filtering needed)."""
    root = _make_project(tmp_path)
    inf_dir = root / "derived" / "inference"
    manifest = json.loads((inf_dir / "inference_manifest.json").read_text())
    trace_paths = manifest["trace_paths"]
    # session_a/b already exist; overwrite b to a single strong bout and add c.
    for sid in ("session_b", "session_c"):
        probs = [0.1] * 60
        for f in range(15, 26):
            probs[f] = 0.9
        df = pd.DataFrame({"frame": list(range(60)), "prob_groom": probs})
        p = inf_dir / f"{sid}__trace.parquet"
        df.to_parquet(p, index=False)
        trace_paths[sid] = str(p)
    # Trim session_a's weaker second bout so each session has exactly one bout.
    probs_a = [0.1] * 60
    for f in range(10, 21):
        probs_a[f] = 0.95
    pd.DataFrame({"frame": list(range(60)), "prob_groom": probs_a}).to_parquet(
        inf_dir / "session_a__trace.parquet", index=False
    )
    (inf_dir / "inference_manifest.json").write_text(
        json.dumps({"trace_paths": trace_paths}), encoding="utf-8"
    )
    return root


def test_select_grid_bouts_prefers_unique_subjects(tmp_path: Path) -> None:
    root = _make_three_session_project(tmp_path)
    # session_a and session_b are the same subject; session_c is a second subject.
    _add_subject_manifest(
        root, {"session_a": "m1", "session_b": "m1", "session_c": "m2"}
    )
    svc = _service(root)
    # Two cells across two subjects: the lone m2 session must always be chosen,
    # plus exactly one of the m1 sessions — never two m1 clips.
    for _ in range(15):
        specs = svc.select_grid_bouts("groom", n_cells=2, top_fraction=1.0)
        assert len(specs) == 2
        sessions = {s.session_id for s in specs}
        assert "session_c" in sessions
        assert len(sessions & {"session_a", "session_b"}) == 1


def test_select_grid_bouts_reuses_subject_when_short(tmp_path: Path) -> None:
    root = _make_three_session_project(tmp_path)
    # All three sessions belong to one subject; with only one unique subject the
    # grid must still fill multiple cells from its distinct (non-overlapping) bouts.
    _add_subject_manifest(
        root, {"session_a": "m1", "session_b": "m1", "session_c": "m1"}
    )
    svc = _service(root)
    specs = svc.select_grid_bouts("groom", n_cells=3, top_fraction=1.0)
    assert len(specs) == 3
    assert {s.session_id for s in specs} == {"session_a", "session_b", "session_c"}


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
