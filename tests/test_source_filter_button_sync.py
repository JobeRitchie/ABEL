"""Regression: the 'Filter Sources' button count must match the number of
source checkboxes the dialog shows.

Stale keys in ``_source_filter_enabled`` (sources whose candidates were cleared
or extracted away) previously left the button reading e.g. "(4/5)" while the
dialog only listed 4 source types — "5 things to filter but only 4 options".
``_sync_source_filter_button`` prunes the dict to the currently-present source
types so the count stays honest.
"""

from types import SimpleNamespace

from abel.models.schemas import CandidateWindow
from abel.ui.tabs.preprocessing_tab import ClipExtractionTab


class _FakeButton:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:  # noqa: N802 (Qt API name)
        self.text = text


def _cand(i: int, source: str) -> CandidateWindow:
    return CandidateWindow(
        window_id=f"w{i}",
        session_id="s1",
        start_frame=i * 10,
        end_frame=i * 10 + 9,
        behavior_id="b1",
        total_score=1.0,
        source=source,
        selection_reason=source,
    )


def _make_fake(candidates: list[CandidateWindow], enabled: dict[str, bool]):
    fake = SimpleNamespace()
    fake._candidates_by_session = {"s1": list(candidates)}
    fake._source_filter_enabled = dict(enabled)
    fake._filter_sources_btn = _FakeButton()
    fake._discover_source_types = lambda: ClipExtractionTab._discover_source_types(fake)
    return fake


def test_sync_prunes_stale_source_keys() -> None:
    # Two sources present, but the dict carries a third stale key.
    cands = [_cand(0, "uncertainty"), _cand(1, "umap_selection")]
    fake = _make_fake(
        cands,
        {"uncertainty": True, "umap_selection": True, "hard_negative": False},
    )

    ClipExtractionTab._sync_source_filter_button(fake)

    assert set(fake._source_filter_enabled) == {"uncertainty", "umap_selection"}
    # All present sources enabled → no count suffix.
    assert fake._filter_sources_btn.text == "Filter Sources…"


def test_sync_count_matches_present_sources() -> None:
    cands = [_cand(0, "uncertainty"), _cand(1, "umap_selection"), _cand(2, "diversity")]
    # One present source disabled, plus a stale key that must not inflate n_total.
    fake = _make_fake(
        cands,
        {
            "uncertainty": True,
            "umap_selection": False,
            "diversity": True,
            "hard_negative": True,  # stale — no candidate uses it
        },
    )

    ClipExtractionTab._sync_source_filter_button(fake)

    # 2 of 3 present sources enabled — the stale 4th key is gone.
    assert fake._filter_sources_btn.text == "Filter Sources (2/3)"
