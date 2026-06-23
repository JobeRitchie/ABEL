"""Reusable dialog for mapping keypoint (bodypart) names.

Presents the *expected* keypoints (rows) and lets the user choose which
*found* keypoint each corresponds to.  Returns an ``{expected: found}`` map.
Used by Data Import to reconcile pose files whose keypoints are named
differently from the project's canonical scheme.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from abel.services import keypoint_mapping

_NONE = "(none)"


class KeypointMappingDialog(QDialog):
    """Map ``expected_keypoints`` onto ``found_keypoints``."""

    def __init__(
        self,
        expected_keypoints: list[str],
        found_keypoints: list[str],
        initial_map: dict[str, str] | None = None,
        expected_label: str = "Canonical Keypoint",
        found_label: str = "Found in File(s)",
        explainer: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Map Keypoints")
        self.setMinimumWidth(480)
        self._expected = list(expected_keypoints)
        self._found = list(found_keypoints)
        self._combos: dict[str, QComboBox] = {}
        self.result_map: dict[str, str] = {}

        layout = QVBoxLayout(self)
        if explainer:
            lbl = QLabel(explainer)
            lbl.setWordWrap(True)
            layout.addWidget(lbl)

        auto_row = QHBoxLayout()
        auto_btn = QPushButton("Auto-map")
        auto_btn.setToolTip("Re-run automatic name matching.")
        auto_btn.clicked.connect(self._auto_map)
        self._status = QLabel("")
        auto_row.addWidget(auto_btn)
        auto_row.addWidget(self._status, 1)
        layout.addLayout(auto_row)

        self._table = QTableWidget(len(self._expected), 2)
        self._table.setHorizontalHeaderLabels([expected_label, found_label])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, hdr.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, hdr.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        mapping = dict(initial_map or {})
        if not mapping:
            mapping = keypoint_mapping.suggest_mapping(self._expected, self._found)
        options = [_NONE] + self._found
        for r, kp in enumerate(self._expected):
            item = QTableWidgetItem(kp)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 0, item)
            combo = QComboBox()
            combo.addItems(options)
            sel = mapping.get(kp, "")
            idx = combo.findText(sel) if sel else -1
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.currentIndexChanged.connect(self._update_status)
            self._combos[kp] = combo
            self._table.setCellWidget(r, 1, combo)
        layout.addWidget(self._table)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_status()

    def _auto_map(self) -> None:
        mapping = keypoint_mapping.suggest_mapping(self._expected, self._found)
        for kp, combo in self._combos.items():
            sel = mapping.get(kp, "")
            idx = combo.findText(sel) if sel else -1
            combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._update_status()

    def _collect(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for kp, combo in self._combos.items():
            val = combo.currentText()
            if val and val != _NONE:
                out[kp] = val
        return out

    def _update_status(self) -> None:
        mapping = self._collect()
        vals = list(mapping.values())
        dupes = len(vals) != len(set(vals))
        msg = f"{len(mapping)}/{len(self._expected)} mapped"
        if dupes:
            msg += "   ⚠ duplicate assignments"
        self._status.setText(msg)

    def _on_accept(self) -> None:
        self.result_map = self._collect()
        self.accept()
