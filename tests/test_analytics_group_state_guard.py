"""A save must never write the cleared factor state over analytics_groups.json.

``set_project`` clears the in-memory factors and defers reading the file to the
next event-loop turn.  A save in between -- loading a section preset is enough --
used to write that empty state to disk, wiping every hand-typed factor
assignment (a project lost its Sex factor this way while its section preset
survived).  These tests pin the guard and the backup that now protect the file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.tabs.behavior_analytics_tab import BehaviorAnalyticsTab  # noqa: E402

STATE = {
    "schema_version": "1.0",
    "factor_definitions": ["Session Type", "Sex"],
    "session_factors": {
        "m10_cond1": {"Session Type": "cond1", "Sex": "M"},
        "m11_cond1": {"Session Type": "cond1", "Sex": "F"},
    },
    "facet_controls": {"Session Type": "__combine__", "Sex": "__split__"},
    "section_definitions": [{"name": "Baseline", "duration": 180}],
}


@pytest.fixture(scope="module")
def _app():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")


def _project(tmp_path: Path) -> Path:
    (tmp_path / "derived").mkdir()
    (tmp_path / "derived" / "analytics_groups.json").write_text(
        json.dumps(STATE), encoding="utf-8",
    )
    return tmp_path


def _on_disk(root: Path) -> dict:
    return json.loads(
        (root / "derived" / "analytics_groups.json").read_text(encoding="utf-8")
    )


def test_save_before_load_leaves_file_untouched(_app, tmp_path: Path) -> None:
    root = _project(tmp_path)
    tab = BehaviorAnalyticsTab()
    tab.set_project(root)  # clears factors; the read is still deferred
    tab._sections_tab.set_sections_state(
        [{"name": f"Tone {i}", "duration": 15} for i in range(1, 51)]
    )
    tab._save_group_state()  # what _load_selected_preset does

    saved = _on_disk(root)
    assert saved["factor_definitions"] == ["Session Type", "Sex"]
    assert saved["session_factors"]["m11_cond1"]["Sex"] == "F"


def test_save_after_load_keeps_factors(_app, tmp_path: Path) -> None:
    root = _project(tmp_path)
    tab = BehaviorAnalyticsTab()
    tab.set_project(root)
    tab._load_group_state()
    tab._save_group_state()

    saved = _on_disk(root)
    assert saved["factor_definitions"] == ["Session Type", "Sex"]
    assert saved["session_factors"]["m10_cond1"] == {"Session Type": "cond1", "Sex": "M"}


def test_dropping_a_factor_keeps_a_backup(_app, tmp_path: Path) -> None:
    root = _project(tmp_path)
    tab = BehaviorAnalyticsTab()
    tab.set_project(root)
    tab._load_group_state()
    tab._factor_definitions.remove("Sex")
    for facs in tab._session_factors.values():
        facs.pop("Sex", None)
    tab._save_group_state()

    assert _on_disk(root)["factor_definitions"] == ["Session Type"]
    backups = list((root / "derived").glob("analytics_groups.backup-*.json"))
    assert len(backups) == 1
    kept = json.loads(backups[0].read_text(encoding="utf-8"))
    assert kept["session_factors"]["m11_cond1"]["Sex"] == "F"


def test_ordinary_save_makes_no_backup(_app, tmp_path: Path) -> None:
    root = _project(tmp_path)
    tab = BehaviorAnalyticsTab()
    tab.set_project(root)
    tab._load_group_state()
    tab._session_factors["m10_cond1"]["Sex"] = "F"
    tab._save_group_state()

    assert not list((root / "derived").glob("analytics_groups.backup-*.json"))


def test_fresh_project_without_a_file_can_save(_app, tmp_path: Path) -> None:
    (tmp_path / "derived").mkdir()
    tab = BehaviorAnalyticsTab()
    tab.set_project(tmp_path)
    tab._load_group_state()
    tab._factor_definitions.append("Sex")
    tab._save_group_state()

    assert _on_disk(tmp_path)["factor_definitions"] == ["Sex"]
