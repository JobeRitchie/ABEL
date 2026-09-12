"""Seed examples appear in the Review tab as reviewed, read-only rows.

Seeds train the model straight from config/seeds.json, so the Review tab lists
them next to reviewed clips for clarity but must never write a review decision
or reviewer label for them (that would label a phantom segment id).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

import abel.ui.tabs.review_tab as review_tab_module  # noqa: E402
from abel.models.schemas import CandidateWindow, ReviewDecisionType, SeedExample  # noqa: E402
from abel.services.behavior_service import BehaviorService  # noqa: E402
from abel.services.candidate_service import CandidateGenerationService  # noqa: E402
from abel.services.import_service import ImportService  # noqa: E402
from abel.services.review_service import ReviewService  # noqa: E402
from abel.services.seed_service import SeedService  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")
    return app


def _tab(tmp_path: Path, monkeypatch):
    (tmp_path / "derived" / "review_tables").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    clips_dir = tmp_path / "derived" / "clips" / "s1"
    clips_dir.mkdir(parents=True, exist_ok=True)

    seeds = SeedService()
    seeds.set_project(tmp_path)
    seeds.add(SeedExample(seed_id="abc", behavior_id="rear", session_id="s1",
                          start_frame=100, end_frame=160))
    seeds.add(SeedExample(seed_id="neg", behavior_id="rear", session_id="s1",
                          start_frame=400, end_frame=460, label_type="negative"))

    candidates = CandidateGenerationService()
    candidates.set_project(tmp_path)
    candidates.upsert_external_window_candidates([
        CandidateWindow(window_id="al0", session_id="s1", start_frame=900, end_frame=929,
                        source="active_learning_uncertainty"),
    ])
    (clips_dir / "al0.mp4").write_bytes(b"stub")
    review = ReviewService()
    review.set_project(tmp_path)
    behaviors = BehaviorService()
    behaviors.set_project(tmp_path)

    shown: list[str] = []
    monkeypatch.setattr(
        review_tab_module.QMessageBox, "information",
        staticmethod(lambda _p, title, *_a, **_k: shown.append(title)),
    )
    tab = review_tab_module.ReviewTab(review, candidates, ImportService(), behaviors)
    tab._project_root = tmp_path
    tab._refresh_candidates()
    return tab, review, shown


def _ids(tab) -> list[str]:
    return [c.window_id for c in tab._visible_candidates]


def test_seeds_listed_only_with_show_reviewed(_app, tmp_path, monkeypatch) -> None:
    tab, _review, _shown = _tab(tmp_path, monkeypatch)
    assert _ids(tab) == ["al0"]

    tab._show_reviewed_chk.setChecked(True)
    assert set(_ids(tab)) == {"al0", "seed_abc", "seed_neg"}
    rows = {c.window_id: r for r, c in enumerate(tab._visible_candidates)}
    assert tab._candidate_table.item(rows["seed_abc"], 7).text() == "seed"
    assert tab._candidate_table.item(rows["seed_neg"], 7).text() == "seed (negative)"
    assert tab._candidate_table.item(rows["seed_abc"], 4).text() == "Seed"


def test_seed_rows_never_write_decisions(_app, tmp_path, monkeypatch) -> None:
    tab, review, shown = _tab(tmp_path, monkeypatch)
    tab._show_reviewed_chk.setChecked(True)
    idx = _ids(tab).index("seed_abc")
    tab._load_candidate(idx, select_row=True)

    tab._save_with_decision(ReviewDecisionType.ACCEPT)
    tab._apply_batch_decision(ReviewDecisionType.REJECT)

    assert not any(d.clip_id.startswith("seed_") for d in review.load_decisions())
    assert shown and set(shown) == {"Seed Examples"}
