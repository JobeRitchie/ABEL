"""Sessions unchecked in Analytics > Summary stay unchecked after a restart.

The check boxes used to live only in the table, so every launch re-checked
every session.  The exclusions now persist in analytics_groups.json, follow a
subject rename, and leave sessions the user never touched checked.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.models.schemas import ImportNameSettings  # noqa: E402
from abel.services.import_service import ImportService  # noqa: E402
from abel.ui.tabs.behavior_analytics_tab import BehaviorAnalyticsTab  # noqa: E402

LABELS = ["m1", "m2", "m3"]


@pytest.fixture(scope="module")
def _app():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")


def _open(root: Path, labels: list[str]) -> BehaviorAnalyticsTab:
    tab = BehaviorAnalyticsTab()
    tab.set_project(root)
    tab._load_group_state()
    tab._summary_rows = [{"session_label": label} for label in labels]
    tab._summary_tab._refresh_session_table()
    return tab


def _row_states(tab: BehaviorAnalyticsTab) -> dict[str, bool]:
    table = tab._summary_tab._session_table
    return {
        table.item(i, 0).text(): table.item(i, 0).checkState() == Qt.CheckState.Checked
        for i in range(table.rowCount())
    }


def _uncheck(tab: BehaviorAnalyticsTab, label: str) -> None:
    table = tab._summary_tab._session_table
    for i in range(table.rowCount()):
        if table.item(i, 0).text() == label:
            table.item(i, 0).setCheckState(Qt.CheckState.Unchecked)  # fires cellChanged


def test_unchecked_session_survives_reopen(_app, tmp_path: Path) -> None:
    (tmp_path / "derived").mkdir()
    tab = _open(tmp_path, LABELS)
    assert all(_row_states(tab).values())

    _uncheck(tab, "m2")
    saved = json.loads((tmp_path / "derived" / "analytics_groups.json").read_text("utf-8"))
    assert saved["unchecked_subjects"] == ["m2"]

    reopened = _open(tmp_path, [*LABELS, "m4"])
    assert _row_states(reopened) == {"m1": True, "m2": False, "m3": True, "m4": True}
    assert reopened._summary_tab._checked_subjects() == {"m1", "m3", "m4"}


def test_hidden_session_keeps_its_exclusion(_app, tmp_path: Path) -> None:
    (tmp_path / "derived").mkdir()
    tab = _open(tmp_path, LABELS)
    _uncheck(tab, "m2")

    # m2 drops out of the table (filter/merge change), then a toggle saves.
    tab._summary_rows = [{"session_label": label} for label in ("m1", "m3")]
    tab._summary_tab._refresh_session_table()
    _uncheck(tab, "m3")
    assert tab._unchecked_subjects == {"m2", "m3"}

    tab._summary_tab._check_all()
    assert tab._unchecked_subjects == {"m2"}


def test_exclusion_follows_subject_rename(_app, tmp_path: Path) -> None:
    stems = ["m1_cond1", "m2_cond1"]
    root = tmp_path / "proj"
    (root / "derived" / "review_tables").mkdir(parents=True)
    svc = ImportService()
    manifest = svc.build_manifest(
        [tmp_path / f"{s}.mp4" for s in stems],
        [tmp_path / f"{s}DLC_resnet50.csv" for s in stems],
        ImportNameSettings(subject_regex=r"^(.+?)(?=DLC|$)"),
    )
    svc.save_manifest(root, manifest)
    (root / "derived" / "analytics_groups.json").write_text(
        json.dumps({"unchecked_subjects": ["m1_cond1"]}), encoding="utf-8",
    )

    tab = BehaviorAnalyticsTab()
    tab.set_project(root)
    tab._subject_by_session = tab._build_subject_map()
    tab._load_group_state()
    tab._save_group_state()  # anchors the exclusion to its session

    svc.apply_subject_name_settings(manifest, ImportNameSettings())
    svc.save_manifest(root, manifest)

    renamed = BehaviorAnalyticsTab()
    renamed.set_project(root)
    renamed._subject_by_session = renamed._build_subject_map()
    renamed._load_group_state()
    assert renamed._unchecked_subjects == {"m1"}
