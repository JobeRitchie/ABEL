"""Behaviour name remapping helper for the Model Refinement tab.

Different projects sometimes give the same behaviour different names — one
labels it "Dip", another "Head Dip".  Because cross-project import matches
behaviours by name, those examples would be silently skipped.  This dialog lets
the user map each unmatched source behaviour onto an existing host behaviour,
with an auto-suggested best guess pre-selected.  The resulting aliases are
returned as ``{source_name_lower: host_name}`` and persisted by the caller.
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

# Sentinel combo entry meaning "leave unmapped / don't import this behaviour".
_SKIP = "— skip (don't import) —"


class BehaviorRemapDialog(QDialog):
    """Map differently named source behaviours onto host behaviours.

    Parameters
    ----------
    source_tag:
        Display name of the source project (for the title/intro).
    rows:
        ``[(source_name, example_count, suggested_host_name)]`` — one per source
        behaviour that does not already match a host behaviour by exact name.
        ``suggested_host_name`` is "" when no good guess exists.
    host_names:
        All host behaviour names available as remap targets.
    current_aliases:
        Existing ``{source_name_lower: host_name}`` mappings to pre-select.
    """

    def __init__(
        self,
        source_tag: str,
        rows: list[tuple[str, int, str]],
        host_names: list[str],
        current_aliases: dict[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Remap Behaviour Names — {source_tag}")
        self.resize(620, 420)

        self._host_names = list(host_names)
        self._current = {
            str(k).lower(): str(v) for k, v in (current_aliases or {}).items()
        }
        self._combos: list[tuple[str, QComboBox]] = []  # (source_name, combo)

        intro = QLabel(
            "These behaviours from <b>{tag}</b> have no behaviour of the same "
            "name in this project, so their examples are skipped on import. "
            "Map each one to the host behaviour it really is (e.g. "
            "<i>Head Dip → Dip</i>). Leave it as <i>skip</i> to ignore it. "
            "A best-guess match is pre-selected where one was found.".format(
                tag=source_tag
            )
        )
        intro.setWordWrap(True)

        self._table = QTableWidget(len(rows), 3)
        self._table.setHorizontalHeaderLabels(
            ["Source behaviour", "Examples", "Map to host behaviour"]
        )
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        for r, (source_name, count, suggested) in enumerate(rows):
            name_item = QTableWidgetItem(source_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            count_item = QTableWidgetItem(str(count))
            count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(r, 0, name_item)
            self._table.setItem(r, 1, count_item)

            combo = QComboBox()
            combo.addItem(_SKIP)
            for hn in self._host_names:
                combo.addItem(hn)
            # Pre-select: existing alias > auto-suggestion > skip.
            preset = self._current.get(source_name.lower(), "") or suggested
            if preset:
                idx = combo.findText(preset, Qt.MatchFlag.MatchFixedString)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    if not self._current.get(source_name.lower()) and suggested:
                        combo.setToolTip(f"Auto-suggested match: {suggested}")
            self._table.setCellWidget(r, 2, combo)
            self._combos.append((source_name, combo))

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

    def aliases(self) -> dict[str, str]:
        """Return the chosen ``{source_name_lower: host_name}`` mappings.

        Source behaviours left on *skip* are omitted.
        """
        out: dict[str, str] = {}
        for source_name, combo in self._combos:
            choice = combo.currentText()
            if choice and choice != _SKIP:
                out[source_name.lower()] = choice
        return out
