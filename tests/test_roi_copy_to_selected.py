"""Copy Current ROI to Selected Subjects writes only the ticked entries."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QDialog, QDialogButtonBox  # noqa: E402

from abel.services.roi_service import ROIService, clear_roi_config_cache  # noqa: E402
from abel.ui.tabs import roi_definition_tab as mod  # noqa: E402
from test_roi_subject_list import _project, _subject_tab, qapp  # noqa: E402,F401

ZONE = {"x": 5, "y": 6, "w": 7, "h": 8}


def _zones(root):
    clear_roi_config_cache()
    return ROIService().load(root).get("subject_rois", {})


def _dialog(entries):
    return mod._SubjectPickDialog(entries, "M1")


def test_only_ticked_entries_and_the_source_are_written(qapp, tmp_path, monkeypatch):
    root = _project(tmp_path, ["M1", "M2", "M3", "M4"])
    tab = _subject_tab(qapp, root)
    tab._subject_list.setCurrentRow(0)
    tab._load_target_zones([ZONE])

    seen = {}

    class FakeDialog:
        def __init__(self, entries, source_label, parent=None):
            seen["keys"] = [k for k, _l, _h in entries]

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_keys(self):
            return ["M2::session_1", "M4::session_3"]

    monkeypatch.setattr(mod, "_SubjectPickDialog", FakeDialog)
    tab._copy_to_selected_subjects()

    # The entry on screen is the source, so it is not offered as a target.
    assert seen["keys"] == ["M2::session_1", "M3::session_2", "M4::session_3"]
    saved = _zones(root)
    assert set(saved) == {"M1::session_0", "M2::session_1", "M4::session_3"}
    for key in saved:
        assert saved[key]["target_zones"][0] == ZONE
    assert tab._current_subject_id() == "M1::session_0"


def test_cancel_writes_nothing(qapp, tmp_path, monkeypatch):
    root = _project(tmp_path, ["M1", "M2"])
    tab = _subject_tab(qapp, root)
    tab._subject_list.setCurrentRow(0)
    tab._load_target_zones([ZONE])

    class FakeDialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

        def selected_keys(self):
            return ["M2::session_1"]

    monkeypatch.setattr(mod, "_SubjectPickDialog", FakeDialog)
    tab._copy_to_selected_subjects()
    assert _zones(root) == {}


def test_dialog_starts_empty_and_reports_overwrites(qapp):
    dlg = _dialog([("a", "M2 / s1", True), ("b", "M3 / s2", False)])
    ok = dlg._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert dlg.selected_keys() == []
    assert not ok.isEnabled()

    dlg._list.item(0).setCheckState(Qt.CheckState.Checked)
    assert dlg.selected_keys() == ["a"]
    assert ok.isEnabled()
    assert "overwritten" in dlg._summary.text()


def test_tick_buttons_respect_the_filter(qapp):
    dlg = _dialog([("a", "M2 / s1", True), ("b", "M3 / s2", False), ("c", "M4 / s3", False)])
    dlg._set_visible(lambda it: not it.data(Qt.ItemDataRole.UserRole + 1))
    assert dlg.selected_keys() == ["b", "c"]

    dlg._set_visible(lambda _it: False)
    dlg._filter.setText("m3")
    dlg._set_visible(lambda _it: True)
    assert dlg.selected_keys() == ["b"]


def test_ticking_one_row_of_a_multi_selection_ticks_them_all(qapp):
    dlg = _dialog([("a", "M2", False), ("b", "M3", False), ("c", "M4", False)])
    dlg._list.item(0).setSelected(True)
    dlg._list.item(1).setSelected(True)
    dlg._list.item(1).setCheckState(Qt.CheckState.Checked)
    assert dlg.selected_keys() == ["a", "b"]
