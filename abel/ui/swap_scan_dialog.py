"""Results of an appearance-based swap scan across a project's sessions.

Shows, per session, what the saved corrections say against what the pixels say,
and lets the user apply the ones that disagree in a single step.  Sessions whose
animals are not visually distinct are listed as "can't tell" rather than being
silently treated as clean.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

_OK = "#A5D6A7"
_DISAGREE = "#FFB74D"
_UNUSABLE = "#90A4AE"


def _frames(values) -> str:
    if not values:
        return "none"
    shown = ", ".join(str(v) for v in values[:4])
    return shown if len(values) <= 4 else f"{shown}, +{len(values) - 4} more"


class SwapScanDialog(QDialog):
    """Review and apply the corrections an appearance scan proposes.

    ``rows`` are the dicts returned by
    :func:`abel.services.appearance_identity_service.scan_sessions`.
    :attr:`selected_rows` holds the rows the user chose to apply.
    """

    def __init__(self, rows: "list[dict]", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Appearance swap scan")
        self._rows = list(rows or [])
        self.selected_rows: list[dict] = []

        disagree = [r for r in self._rows if r.get("usable") and not r.get("agrees")]
        unusable = [r for r in self._rows if not r.get("usable")]

        layout = QVBoxLayout(self)
        summary = QLabel(
            f"Checked {len(self._rows)} session(s): "
            f"<b>{len(disagree)}</b> disagree with their saved corrections, "
            f"{len(self._rows) - len(disagree) - len(unusable)} already match, "
            f"{len(unusable)} could not be judged from appearance."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self._table = QTableWidget(len(self._rows), 6)
        self._table.setHorizontalHeaderLabels(
            ["Apply", "Subject", "Saved", "Detected", "Identity now → after", "Notes"]
        )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        for row, data in enumerate(self._rows):
            applicable = bool(data.get("usable")) and not data.get("agrees")
            check = QTableWidgetItem()
            check.setFlags(
                (Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                if applicable else Qt.ItemFlag.NoItemFlags
            )
            check.setCheckState(
                Qt.CheckState.Checked if applicable else Qt.CheckState.Unchecked
            )
            self._table.setItem(row, 0, check)
            self._table.setItem(row, 1, QTableWidgetItem(str(data.get("subject", ""))))
            self._table.setItem(row, 2, QTableWidgetItem(_frames(data.get("saved"))))
            self._table.setItem(row, 3, QTableWidgetItem(
                _frames(data.get("detected")) if data.get("usable") else "-"
            ))
            self._table.setItem(row, 4, QTableWidgetItem(
                f"{data.get('consistency_before', 0.0):.0%} → {data.get('consistency_after', 0.0):.0%}"
                if data.get("usable") else "-"
            ))
            note = (
                "agrees with your corrections" if data.get("usable") and data.get("agrees")
                else (data.get("message") or "")
            )
            self._table.setItem(row, 5, QTableWidgetItem(note))
            colour = QColor(
                _UNUSABLE if not data.get("usable")
                else (_OK if data.get("agrees") else _DISAGREE)
            )
            self._table.item(row, 1).setForeground(colour)
        self._table.resizeColumnsToContents()
        layout.addWidget(self._table)

        self._remap_chk = QCheckBox("Move labels already committed on these sessions")
        self._remap_chk.setChecked(True)
        self._remap_chk.setToolTip(
            "Labels are keyed by animal, so applying a swap moves which animal a "
            "label describes. Leave this on unless you plan to re-review them."
        )
        layout.addWidget(self._remap_chk)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close
        )
        apply_btn = buttons.button(QDialogButtonBox.StandardButton.Apply)
        apply_btn.setEnabled(bool(disagree))
        apply_btn.setText("Apply checked")
        apply_btn.clicked.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(buttons)
        layout.addLayout(row)

    @property
    def remap_labels(self) -> bool:
        return self._remap_chk.isChecked()

    def _on_apply(self) -> None:
        self.selected_rows = [
            data for row, data in enumerate(self._rows)
            if (item := self._table.item(row, 0)) is not None
            and item.checkState() == Qt.CheckState.Checked
        ]
        self.accept()
