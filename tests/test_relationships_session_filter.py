"""The Relationships panel can run its analyses on one level of a factor."""

from __future__ import annotations

from types import SimpleNamespace

from abel.services import behavioral_motif_service as bms
from abel.services.behavioral_motif_service import MotifSettings
from abel.ui.tabs.behavior_analytics_tab import _BehaviorMotifWidget as W


def _panel(session_filter: dict[str, str]) -> SimpleNamespace:
    session_factors = {
        "M1 cond": {"Phase": "Conditioning", "Sex": "F"},
        "M1 ext": {"Phase": "Extinction", "Sex": "F"},
        "M2 cond": {"Phase": "Conditioning", "Sex": "M"},
        "M2 ext": {"Phase": "Extinction", "Sex": "M"},
    }

    def levels() -> dict[str, list[str]]:
        return {
            "Phase": ["Conditioning", "Extinction"],
            "Sex": ["F", "M"],
        }

    host = SimpleNamespace(
        _session_factors=session_factors,
        _levels_by_factor=levels,
        _summary_tab=SimpleNamespace(_checked_subjects=lambda: set(session_factors)),
        _project_fps=lambda: 30.0,
        _session_fps=lambda _sid: 30.0,
        _fps_by_session=lambda: {},
        _raw_bouts=[1],
        _session_label_by_session={
            "s1": "M1 cond", "s2": "M1 ext", "s3": "M2 cond", "s4": "M2 ext",
        },
    )
    settings = MotifSettings(session_filter=session_filter)
    p = SimpleNamespace(_host=host, _settings=settings)
    p._session_filter = lambda: W._session_filter(p)
    p._session_passes_filter = lambda lbl, flt=None: W._session_passes_filter(p, lbl, flt)
    p._get_behavior_ids_and_names = lambda: (["b"], ["B"], {"b": "B"})
    return p


def _sessions(p, monkeypatch) -> set[str]:
    monkeypatch.setattr(
        bms, "build_sequences",
        lambda *_a, **_k: {s: [(0.0, 1.0, "b")] for s in ("s1", "s2", "s3", "s4")},
    )
    return set(W._get_sequences_for_analysis(p))


def test_no_filter_keeps_every_checked_session(monkeypatch) -> None:
    assert _sessions(_panel({}), monkeypatch) == {"s1", "s2", "s3", "s4"}


def test_one_level_keeps_only_those_sessions(monkeypatch) -> None:
    p = _panel({"Phase": "Conditioning"})
    assert _sessions(p, monkeypatch) == {"s1", "s3"}


def test_filters_on_two_factors_combine(monkeypatch) -> None:
    p = _panel({"Phase": "Extinction", "Sex": "M"})
    assert _sessions(p, monkeypatch) == {"s4"}


def test_stale_level_is_ignored_not_excluding_everything(monkeypatch) -> None:
    p = _panel({"Phase": "Renamed level", "Gone factor": "x"})
    assert W._session_filter(p) == {}
    assert _sessions(p, monkeypatch) == {"s1", "s2", "s3", "s4"}


def test_filter_round_trips_through_settings_json(tmp_path) -> None:
    bms.save_motif_settings(tmp_path, MotifSettings(session_filter={"Phase": "Conditioning"}))
    assert bms.load_motif_settings(tmp_path).session_filter == {"Phase": "Conditioning"}
