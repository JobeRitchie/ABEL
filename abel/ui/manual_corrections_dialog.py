"""Dialog for entering hand-scored values in place of ABEL's Analytics values."""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from abel.services.manual_corrections_service import METRICS, ManualCorrections

_BLANK = "-"


def _fmt(metric: str, value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return _BLANK
    return f"{value:.0f}" if metric == "n_bouts" else f"{value:.2f}"


class ManualCorrectionsDialog(QDialog):
    """Pick a behavior, then type over any session's value.

    Edits go to a working copy; nothing is saved unless the dialog is accepted.
    """

    def __init__(
        self,
        parent: QWidget | None,
        behavior_names: dict[str, str],
        sessions: list[dict[str, Any]],
        base_rows: list[dict[str, Any]],
        corrections: ManualCorrections,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manual Corrections")
        self._sessions = sessions
        self._working = corrections.copy()
        self._metrics = list(METRICS)
        self._filling = False
        self._abel: dict[tuple[str, str], dict[str, float | None]] = {}
        for row in base_rows:
            key = (str(row.get("behavior_id", "")), str(row.get("session_id", "")))
            self._abel[key] = {m: self._as_float(row.get(m)) for m in self._metrics}

        intro = QLabel(
            "Type a hand-scored value over ABEL's value for any session. "
            "Corrected values are shown in bold. Clear a cell to go back to "
            "ABEL's value. Latency is measured in analysis time (after any "
            "prechop)."
        )
        intro.setWordWrap(True)

        warning = QLabel(
            "Any manual correction is flagged on the project Home screen and "
            "on the Validation tab, with the number of corrected values per "
            "behavior. The ABEL value each one replaced is kept in the project."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #FFB74D; font-weight: 600;")

        scope = QLabel(
            "Corrections replace whole-session values in the Summary table, "
            "graphs, statistics and exports. A corrected latency also sets the "
            "cutoff for 'Bouts until behavior'. Time-binned graphs, ethograms, "
            "the Data Range and the other bout filters still use ABEL's bouts."
        )
        scope.setWordWrap(True)
        scope.setStyleSheet("color: #90A4AE;")

        self._behavior_combo = QComboBox(self)
        for bid, name in sorted(behavior_names.items(), key=lambda kv: kv[1].lower()):
            self._behavior_combo.addItem(name, userData=bid)
        self._behavior_combo.currentIndexChanged.connect(self._fill_table)
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Behavior:"))
        combo_row.addWidget(self._behavior_combo, 1)

        self._table = QTableWidget(len(sessions), 1 + len(self._metrics), self)
        self._table.setHorizontalHeaderLabels(["Session", *METRICS.values()])
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, self._table.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemChanged.connect(self._on_item_changed)

        self._message = QLabel("")
        self._message.setWordWrap(True)

        reset_sel_btn = QPushButton("Reset Selected Cells")
        reset_sel_btn.clicked.connect(self._reset_selected)
        reset_all_btn = QPushButton("Reset This Behavior")
        reset_all_btn.clicked.connect(self._reset_behavior)
        paste_btn = QPushButton("Paste")
        paste_btn.setToolTip(
            "Paste a column or block copied from a spreadsheet, starting at the "
            "selected cell (Ctrl+V)."
        )
        paste_btn.clicked.connect(self._paste)
        paste_btn.setShortcut(QKeySequence.StandardKey.Paste)
        btn_row = QHBoxLayout()
        btn_row.addWidget(paste_btn)
        btn_row.addWidget(reset_sel_btn)
        btn_row.addWidget(reset_all_btn)
        btn_row.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        # Enter in the table must not save and close the dialog.
        for button in buttons.buttons():
            button.setAutoDefault(False)
            button.setDefault(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(warning)
        layout.addWidget(scope)
        layout.addLayout(combo_row)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._message)
        layout.addLayout(btn_row)
        layout.addWidget(buttons)

        self._fill_table()
        # Sized from the font so it scales with the display, not fixed pixels.
        fm = self.fontMetrics()
        self.resize(fm.averageCharWidth() * 110, fm.height() * 34)

    # -- data ----------------------------------------------------------

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    def corrections(self) -> ManualCorrections:
        return self._working

    def _behavior_id(self) -> str:
        return str(self._behavior_combo.currentData() or "")

    def _abel_value(self, bid: str, sid: str, metric: str) -> float | None:
        found = self._abel.get((bid, sid))
        if found is None:
            # ABEL scored no bouts for this session.
            return None if metric == "latency_s" else 0.0
        return found.get(metric)

    # -- table ---------------------------------------------------------

    def _fill_table(self) -> None:
        bid = self._behavior_id()
        self._filling = True
        try:
            for r, session in enumerate(self._sessions):
                sid = str(session["session_id"])
                name_item = QTableWidgetItem(str(session.get("session_label", sid)))
                name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self._table.setItem(r, 0, name_item)
                for c, metric in enumerate(self._metrics, start=1):
                    self._table.setItem(r, c, QTableWidgetItem())
                    self._show_cell(r, c, bid, sid, metric)
        finally:
            self._filling = False
        self._update_count()

    def _show_cell(self, row: int, col: int, bid: str, sid: str, metric: str) -> None:
        item = self._table.item(row, col)
        manual = self._working.get(bid, sid, metric)
        abel = self._abel_value(bid, sid, metric)
        was_filling = self._filling
        self._filling = True
        try:
            item.setText(_fmt(metric, abel if manual is None else manual))
            font = item.font()
            font.setBold(manual is not None)
            item.setFont(font)
            item.setToolTip(
                "" if manual is None
                else "Manually corrected. ABEL value: "
                + ("none" if abel is None else _fmt(metric, abel))
            )
        finally:
            self._filling = was_filling

    def _apply_text(self, row: int, col: int, text: str) -> str:
        """Store one typed/pasted cell; return an error message or ''."""
        bid = self._behavior_id()
        sid = str(self._sessions[row]["session_id"])
        metric = self._metrics[col - 1]
        abel = self._abel_value(bid, sid, metric)
        text = text.strip()
        error = ""
        if text in ("", _BLANK) or text == _fmt(metric, abel):
            self._working.clear(bid, sid, metric)
        else:
            try:
                self._working.set(bid, sid, metric, float(text), abel)
            except ValueError:
                error = f"'{text}' is not a number of 0 or more. Value not changed."
        self._show_cell(row, col, bid, sid, metric)
        return error

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() == 0:
            return
        self._message.setText(self._apply_text(item.row(), item.column(), item.text()))
        self._update_count()

    def _update_count(self) -> None:
        if self._message.text():
            return
        n = self._working.counts().get(self._behavior_id(), 0)
        self._message.setText(
            f"{n} corrected value(s) for this behavior." if n else ""
        )

    def _reset_selected(self) -> None:
        self._message.setText("")
        for item in self._table.selectedItems():
            if item.column() > 0:
                self._apply_text(item.row(), item.column(), "")
        self._update_count()

    def _reset_behavior(self) -> None:
        self._message.setText("")
        self._working.clear(self._behavior_id())
        self._fill_table()

    def _paste(self) -> None:
        text = QGuiApplication.clipboard().text()
        row0, col0 = self._table.currentRow(), self._table.currentColumn()
        if not text.strip() or row0 < 0:
            return
        col0 = max(1, col0)
        errors = 0
        for dr, line in enumerate(text.rstrip("\r\n").splitlines()):
            for dc, cell in enumerate(line.split("\t")):
                r, c = row0 + dr, col0 + dc
                if r >= self._table.rowCount() or c >= self._table.columnCount():
                    continue
                if self._apply_text(r, c, cell):
                    errors += 1
        self._message.setText(
            f"{errors} pasted cell(s) were not numbers and were skipped." if errors else ""
        )
        self._update_count()
