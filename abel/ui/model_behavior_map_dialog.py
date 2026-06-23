"""Behaviour-mapping helper for importing another project's models.

Each model predicts a behaviour that may or may not already exist in this
project.  This dialog lets the user, per source behaviour, either map it onto an
existing project behaviour, auto-create it (carrying over the source project's
name/definition), or skip the model.  The result is a
``{source_behavior_id: decision}`` map where ``decision`` is a host behaviour id,
``AUTO_CREATE_BEHAVIOR``, or ``SKIP_BEHAVIOR``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from abel.services.model_refinement_service import (
    AUTO_CREATE_BEHAVIOR,
    SKIP_BEHAVIOR,
)


class ModelBehaviorMapDialog(QDialog):
    """Map each imported model's behaviour onto this project.

    Parameters
    ----------
    source_tag:
        Display name of the source project.
    rows:
        ``[(source_behavior_id, source_name, suggested_host_id)]`` — one per
        source behaviour being imported.  ``suggested_host_id`` is "" when the
        behaviour has no existing match (it then defaults to auto-create).
    host_behaviors:
        ``[(host_behavior_id, host_name)]`` available as map targets.
    """

    def __init__(
        self,
        source_tag: str,
        rows: list[tuple[str, str, str]],
        host_behaviors: list[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Map Model Behaviours — {source_tag}")
        self.resize(640, 420)

        self._host_behaviors = list(host_behaviors)
        self._combos: list[tuple[str, QComboBox]] = []  # (source_behavior_id, combo)

        intro = QLabel(
            f"Each model from <b>{source_tag}</b> predicts a behaviour. Map it to "
            "a behaviour in this project, or <i>Auto-create</i> it to add this "
            "project a copy of the source's behaviour definition (same name). "
            "Choose <i>skip</i> to not import that model."
        )
        intro.setWordWrap(True)

        self._table = QTableWidget(len(rows), 2)
        self._table.setHorizontalHeaderLabels(
            ["Model behaviour (source)", "Apply as"]
        )
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        for r, (src_bid, src_name, suggested_host_id) in enumerate(rows):
            name_item = QTableWidgetItem(src_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 0, name_item)

            combo = QComboBox()
            # Option 0: auto-create, carrying the source behaviour name through.
            combo.addItem(f"Auto-create “{src_name}”", AUTO_CREATE_BEHAVIOR)
            for host_id, host_name in self._host_behaviors:
                combo.addItem(f"Map to: {host_name}", host_id)
            combo.addItem("— skip (don't import) —", SKIP_BEHAVIOR)

            # Pre-select the suggested existing behaviour, else auto-create.
            if suggested_host_id:
                idx = combo.findData(suggested_host_id)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                combo.setCurrentIndex(0)
            self._table.setCellWidget(r, 1, combo)
            self._combos.append((src_bid, combo))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self._table, 1)
        layout.addWidget(buttons)

    def decisions(self) -> dict[str, str]:
        """Return ``{source_behavior_id: host_id | AUTO_CREATE_BEHAVIOR | SKIP_BEHAVIOR}``."""
        out: dict[str, str] = {}
        for src_bid, combo in self._combos:
            out[src_bid] = str(combo.currentData())
        return out
