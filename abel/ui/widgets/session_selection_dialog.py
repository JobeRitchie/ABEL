"""Reusable 'Choose Sessions' dialog with session-type filtering.

Both the Active Learning and Temporal Refinement tabs let the user restrict a
run to a subset of linked sessions.  This dialog centralises that UI so the two
tabs stay in sync and share the session-type selector: a dropdown of the session
types present in the project plus a 'Check all of type' button that adds every
session of the chosen type to the current selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class SessionOption:
    """One selectable linked session.

    ``session_id`` is the stable unique id used for the returned selection;
    ``subject`` and ``session_type`` drive the row label and the type filter.
    """

    session_id: str
    subject: str = ""
    session_type: str = ""

    def label(self) -> str:
        parts = [self.session_id]
        if self.subject:
            parts.append(f"subject: {self.subject}")
        if self.session_type:
            parts.append(f"type: {self.session_type}")
        return "  |  ".join(parts)


_ALL_TYPES = "All types"
_UNTYPED = "(no type)"


def session_type_matches(chosen: str, session_type: str) -> bool:
    """Return True if a session of ``session_type`` matches the dropdown choice.

    ``_ALL_TYPES`` matches everything, ``_UNTYPED`` matches sessions with no
    type, and any other value matches on exact type equality.
    """
    if chosen == _ALL_TYPES:
        return True
    if chosen == _UNTYPED:
        return not session_type
    return session_type == chosen


def choose_sessions(
    parent: QWidget,
    *,
    title: str,
    info: str,
    options: list[SessionOption],
    current_selected: set[str],
) -> list[str] | None:
    """Show the Choose Sessions dialog and return the selected session ids.

    ``current_selected`` seeds the initial check state.  Returns the list of
    checked session ids on OK, or ``None`` if the dialog was cancelled.  The
    caller is responsible for interpreting an empty / full selection (e.g.
    collapsing 'all selected' to a None scope).
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(560, 640)

    info_label = QLabel(info, dlg)
    info_label.setWordWrap(True)

    list_widget = QListWidget(dlg)
    for opt in options:
        item = QListWidgetItem(opt.label())
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.ItemDataRole.UserRole, opt.session_id)
        item.setData(Qt.ItemDataRole.UserRole + 1, opt.session_type)
        item.setCheckState(
            Qt.CheckState.Checked
            if opt.session_id in current_selected
            else Qt.CheckState.Unchecked
        )
        list_widget.addItem(item)

    def _set_all(state: Qt.CheckState) -> None:
        for i in range(list_widget.count()):
            list_widget.item(i).setCheckState(state)

    select_all_btn = QPushButton("Select All", dlg)
    deselect_all_btn = QPushButton("Deselect All", dlg)
    select_all_btn.clicked.connect(lambda: _set_all(Qt.CheckState.Checked))
    deselect_all_btn.clicked.connect(lambda: _set_all(Qt.CheckState.Unchecked))

    # ── Session-type filter: dropdown of present types + 'Check all of type' ──
    type_combo = QComboBox(dlg)
    type_combo.addItem(_ALL_TYPES)
    present_types = sorted({opt.session_type for opt in options if opt.session_type})
    for t in present_types:
        type_combo.addItem(t)
    if any(not opt.session_type for opt in options):
        type_combo.addItem(_UNTYPED)

    check_type_btn = QPushButton("Check all of type", dlg)
    check_type_btn.setToolTip(
        "Add every session of the selected type to the current selection.\n"
        "Existing checks are left in place, so you can combine multiple types."
    )

    def _check_all_of_type() -> None:
        chosen = type_combo.currentText()
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            stype = str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
            if session_type_matches(chosen, stype):
                item.setCheckState(Qt.CheckState.Checked)

    check_type_btn.clicked.connect(_check_all_of_type)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        parent=dlg,
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)

    top_btn_row = QHBoxLayout()
    top_btn_row.addWidget(select_all_btn)
    top_btn_row.addWidget(deselect_all_btn)
    top_btn_row.addStretch(1)

    type_row = QHBoxLayout()
    type_row.addWidget(QLabel("Session type:", dlg))
    type_row.addWidget(type_combo, 1)
    type_row.addWidget(check_type_btn)

    layout = QVBoxLayout(dlg)
    layout.addWidget(info_label)
    layout.addLayout(top_btn_row)
    layout.addLayout(type_row)
    layout.addWidget(list_widget, 1)
    layout.addWidget(buttons)

    if dlg.exec() != int(QDialog.DialogCode.Accepted):
        return None

    selected_ids: list[str] = []
    for i in range(list_widget.count()):
        item = list_widget.item(i)
        if item.checkState() == Qt.CheckState.Checked:
            sid = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
            if sid:
                selected_ids.append(sid)
    return selected_ids
