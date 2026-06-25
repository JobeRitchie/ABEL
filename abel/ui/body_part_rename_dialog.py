"""Dialog for renaming body parts (keypoints) to new project-wide names.

Unlike Keypoint Mapping — which reconciles differently-named pose files onto the
project's *existing* keypoint scheme — this lets the user give body parts brand
new names of their own choosing.  The renames are stored in the project's
keypoint alias map (``config/keypoint_aliases.json``) and applied during pose
loading, so every downstream step (feature extraction, context features, trained
models) sees the new names.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class BodyPartRenameDialog(QDialog):
    """Give each body part a new name.

    ``result_map`` holds ``{original_name: new_name}`` for the parts the user
    actually changed (identity/blank entries are omitted).
    """

    def __init__(
        self,
        body_parts: list[str],
        initial_map: dict[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rename Body Parts")
        self.setMinimumWidth(460)
        self._parts = list(body_parts)
        self._edits: dict[str, QLineEdit] = {}
        self.result_map: dict[str, str] = {}

        initial = dict(initial_map or {})

        layout = QVBoxLayout(self)
        explainer = QLabel(
            "Give body parts new names. The new names replace the originals "
            "everywhere downstream — feature extraction, context features and "
            "any models you train on this project. Leave a name unchanged to "
            "keep it.\n\nThis is different from Keypoint Mapping, which only "
            "aligns differently-named pose files onto the project's existing "
            "keypoints."
        )
        explainer.setWordWrap(True)
        layout.addWidget(explainer)

        self._table = QTableWidget(len(self._parts), 2)
        self._table.setHorizontalHeaderLabels(["Original Name", "New Name"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, hdr.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, hdr.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        for r, part in enumerate(self._parts):
            item = QTableWidgetItem(part)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 0, item)
            edit = QLineEdit(str(initial.get(part, part)))
            edit.setPlaceholderText(part)
            edit.textChanged.connect(self._update_status)
            self._edits[part] = edit
            self._table.setCellWidget(r, 1, edit)
        layout.addWidget(self._table)

        action_row = QHBoxLayout()
        reset_btn = QPushButton("Reset to Original")
        reset_btn.setToolTip("Restore every body part to its original name.")
        reset_btn.clicked.connect(self._reset)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        action_row.addWidget(reset_btn)
        action_row.addWidget(self._status, 1)
        layout.addLayout(action_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_status()

    # ── Internals ─────────────────────────────────────────────────────

    def _reset(self) -> None:
        for part, edit in self._edits.items():
            edit.setText(part)
        self._update_status()

    def _collect(self) -> dict[str, str]:
        """Return ``{original: new}`` for the parts the user actually renamed."""
        out: dict[str, str] = {}
        for part, edit in self._edits.items():
            new = edit.text().strip()
            if new and new != part:
                out[part] = new
        return out

    def _final_names(self) -> list[str]:
        """The effective name for every part (its new name, or the original)."""
        return [
            (edit.text().strip() or part) for part, edit in self._edits.items()
        ]

    def _update_status(self) -> None:
        finals = self._final_names()
        dupes = sorted({n for n in finals if finals.count(n) > 1})
        n_renamed = len(self._collect())
        msg = f"{n_renamed} body part(s) renamed"
        if dupes:
            msg += f"   ⚠ duplicate names: {', '.join(dupes)}"
            self._status.setStyleSheet("color: #EF9A9A;")
        else:
            self._status.setStyleSheet("")
        self._status.setText(msg)

    def _on_accept(self) -> None:
        self.result_map = self._collect()
        self.accept()
