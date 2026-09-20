"""Analytics group state follows a subject rename, even one it saved over.

The tab keeps factors in memory keyed by session label.  A rename in Data
Import while the tab is open used to leave those keys stale, and the tab's
next save wrote them back.  Saved anchors (the session ids behind each key) now
let both a refresh and a later load re-key the state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.models.schemas import ImportNameSettings  # noqa: E402
from abel.services.import_service import ImportService  # noqa: E402
from abel.ui.tabs import behavior_analytics_tab as analytics_mod  # noqa: E402
from abel.ui.tabs.behavior_analytics_tab import BehaviorAnalyticsTab  # noqa: E402

STEMS = ["m1_cond1", "m1_ext", "m2_cond1"]


@pytest.fixture(scope="module")
def _app():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")


def _project(tmp_path: Path):
    root = tmp_path / "proj"
    (root / "derived" / "review_tables").mkdir(parents=True)
    svc = ImportService()
    manifest = svc.build_manifest(
        [tmp_path / f"{s}.mp4" for s in STEMS],
        [tmp_path / f"{s}DLC_resnet50.csv" for s in STEMS],
        ImportNameSettings(subject_regex=r"^(.+?)(?=DLC|$)"),
    )
    svc.save_manifest(root, manifest)
    (root / "derived" / "analytics_groups.json").write_text(json.dumps({
        "factor_definitions": ["Sex"],
        "session_factors": {s: {"Sex": "M" if s.startswith("m1") else "F"} for s in STEMS},
        "subject_order": list(reversed(STEMS)),
    }), encoding="utf-8")
    return root, svc, manifest


def _open(root: Path) -> BehaviorAnalyticsTab:
    tab = BehaviorAnalyticsTab()
    tab.set_project(root)
    tab._subject_by_session = tab._build_subject_map()
    tab._load_group_state()
    return tab


def test_stale_save_after_rename_heals_on_next_load(_app, tmp_path: Path) -> None:
    root, svc, manifest = _project(tmp_path)
    tab = _open(root)  # loaded under the old labels

    svc.apply_subject_name_settings(manifest, ImportNameSettings())
    svc.save_manifest(root, manifest)
    tab._save_group_state()  # the open tab writes its old-label state back

    healed = _open(root)
    assert healed._session_factors == {
        "m1 – cond1": {"Sex": "M"},
        "m1 – ext": {"Sex": "M"},
        "m2": {"Sex": "F"},
    }
    assert healed._subject_order == ["m2", "m1 – ext", "m1 – cond1"]


def test_refresh_remaps_in_memory_state(_app, tmp_path: Path) -> None:
    root, svc, manifest = _project(tmp_path)
    tab = _open(root)
    tab._save_group_state()  # writes anchors for the old labels

    svc.apply_subject_name_settings(manifest, ImportNameSettings())
    svc.save_manifest(root, manifest)
    previous = tab._session_labels
    tab._manifest_cache = analytics_mod._MANIFEST_UNSET  # what _refresh does
    tab._subject_by_session = tab._build_subject_map()
    tab._remap_group_state_to_labels(previous)

    assert set(tab._session_factors) == {"m1 – cond1", "m1 – ext", "m2"}
    on_disk = json.loads((root / "derived" / "analytics_groups.json").read_text(encoding="utf-8"))
    assert set(on_disk["session_factors"]) == {"m1 – cond1", "m1 – ext", "m2"}


def test_cached_distance_rows_take_current_labels(_app, tmp_path: Path) -> None:
    # The distance/ROI cache is keyed by pose files, not names, so a rename
    # hits it: its rows must not bring the old labels back as extra sessions.
    root, svc, manifest = _project(tmp_path)
    old_tab = _open(root)
    cached = old_tab._build_distance_rows(sorted(old_tab._subject_by_session))
    assert {r["session_label"] for r in cached} == set(STEMS)

    svc.apply_subject_name_settings(manifest, ImportNameSettings())
    svc.save_manifest(root, manifest)
    tab = _open(root)
    rows = tab._relabel_to_current_sessions(cached + [
        {"session_id": "merged::s9", "subject": "x", "session_label": "other::m9", "session_type": ""},
    ])
    assert {r["session_label"] for r in rows} == {"m1 – cond1", "m1 – ext", "m2", "other::m9"}
    assert {r["subject"] for r in rows} == {"m1", "m2", "x"}
