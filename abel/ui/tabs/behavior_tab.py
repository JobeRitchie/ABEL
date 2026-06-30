"""Behavior Definitions tab — full CRUD editor for operational behavior definitions."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.models.schemas import BehaviorDefinition
from abel.services.behavior_service import BehaviorService
from abel.storage.file_store import read_yaml, write_yaml

logger = logging.getLogger("abel")

_SWATCH_CSS = "background-color: {color}; border: 1px solid #1565C0; border-radius: 4px;"


class BehaviorTab(QWidget):
    """Full CRUD editor for behavior definitions."""

    # Emitted whenever allow_co_occurring_behaviors is toggled so the Review tab
    # can update its co-occurring UI state without requiring a manual Refresh.
    co_occurring_changed = Signal()

    # Emitted whenever the set of behavior definitions changes (add/edit/delete/
    # import) so other tabs—e.g. the Active Learning target-behavior dropdown—can
    # refresh their behavior-derived options without a manual Refresh.
    behaviors_changed = Signal()

    def __init__(self, service: BehaviorService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = service
        self._selected_id: str | None = None
        self._current_color: str = "#4A90E2"
        self._project_root: Path | None = None

        # ── Left panel: table + actions ─────────────────────────────────
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Name", "Short", "Color", "Key", "Active"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setColumnWidth(0, 160)
        self._table.setColumnWidth(1, 70)
        self._table.setColumnWidth(2, 50)
        self._table.setColumnWidth(3, 40)
        self._table.setColumnWidth(4, 50)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

        new_btn = QPushButton("＋ New")
        self._delete_btn = QPushButton("🗑 Delete")
        export_btn = QPushButton("Export…")
        import_btn = QPushButton("Import…")

        new_btn.clicked.connect(self._new_behavior)
        self._delete_btn.clicked.connect(self._delete_selected)
        export_btn.clicked.connect(self._export)
        import_btn.clicked.connect(self._import)

        btn_bar = QHBoxLayout()
        for b in (new_btn, self._delete_btn, export_btn, import_btn):
            btn_bar.addWidget(b)
        btn_bar.addStretch()

        self._co_occurring_chk = QCheckBox("Allow co-occurring behavior labels (multi-label per clip)")
        self._co_occurring_chk.setToolTip(
            "When enabled, the review and seed tabs allow assigning more than one behavior\n"
            "label to the same clip window. Labels are stored as pipe-separated strings\n"
            "(e.g. grooming|rearing) and each is expanded into its own training row."
        )
        self._co_occurring_chk.toggled.connect(self._on_co_occurring_toggled)

        left = QVBoxLayout()
        left.addLayout(btn_bar)
        left.addWidget(self._table)
        left.addWidget(self._co_occurring_chk)

        left_widget = QWidget()
        left_widget.setLayout(left)

        # ── Right panel: edit form ───────────────────────────────────────
        right_widget = self._build_form_panel()

        # ── Splitter ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([380, 560])

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(splitter)

        self._set_form_enabled(False)

    # ------------------------------------------------------------------
    # Form construction
    # ------------------------------------------------------------------

    def _build_form_panel(self) -> QWidget:
        """Scrollable form with all BehaviorDefinition fields."""
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Core identity
        self._f_name = QLineEdit()
        self._f_name.setPlaceholderText("e.g. Grooming")
        self._f_short = QLineEdit()
        self._f_short.setPlaceholderText("e.g. groom")
        self._f_short.setMaximumWidth(120)

        self._f_color_btn = QPushButton()
        self._f_color_btn.setFixedSize(80, 26)
        self._f_color_btn.clicked.connect(self._pick_color)
        self._set_color_btn(self._current_color)

        self._f_shortcut = QLineEdit()
        self._f_shortcut.setMaximumWidth(50)
        self._f_shortcut.setMaxLength(1)
        self._f_shortcut.setPlaceholderText("g")

        self._f_min_dur = QDoubleSpinBox()
        self._f_min_dur.setRange(0.0, 600.0)
        self._f_min_dur.setSingleStep(0.1)
        self._f_min_dur.setSuffix(" s")
        self._f_min_dur.setMaximumWidth(100)

        self._f_active = QCheckBox("Behavior is active")
        self._f_active.setChecked(True)

        # Social / interaction behavior (multi-animal projects only). Hidden
        # unless the project tracks more than one animal.
        self._f_social = QCheckBox("Social / interaction behavior")
        self._f_social.setToolTip(
            "Mark this as an interaction behavior. It is trained on the focal\n"
            "animal's timeline using inter-animal (social_*) features. Requires a\n"
            "multi-animal project with interaction features enabled."
        )
        self._f_directionality = QComboBox()
        self._f_directionality.addItems(["none", "directed", "mutual"])
        self._f_directionality.setToolTip(
            "directed: labels the focal actor (e.g. the animal that displaces another).\n"
            "mutual: labels both interacting animals' overlapping segments.\n"
            "none: solo behavior."
        )
        self._f_social.toggled.connect(self._f_directionality.setEnabled)
        self._f_directionality.setEnabled(False)

        form_layout.addRow("Name *", self._f_name)

        short_row = QHBoxLayout()
        short_row.addWidget(self._f_short)
        short_row.addStretch()
        form_layout.addRow("Short name", short_row)

        meta_row = QHBoxLayout()
        meta_row.addWidget(QLabel("Color:"))
        meta_row.addWidget(self._f_color_btn)
        meta_row.addSpacing(12)
        meta_row.addWidget(QLabel("Shortcut:"))
        meta_row.addWidget(self._f_shortcut)
        meta_row.addStretch()
        form_layout.addRow("", meta_row)

        dur_row = QHBoxLayout()
        dur_row.addWidget(self._f_min_dur)
        dur_row.addStretch()
        form_layout.addRow("Min duration", dur_row)
        form_layout.addRow("", self._f_active)

        self._social_row = QWidget()
        social_row = QHBoxLayout(self._social_row)
        social_row.setContentsMargins(0, 0, 0, 0)
        social_row.addWidget(self._f_social)
        social_row.addSpacing(12)
        social_row.addWidget(QLabel("Directionality:"))
        social_row.addWidget(self._f_directionality)
        social_row.addStretch()
        self._social_form_row = self._social_row
        form_layout.addRow("", self._social_row)
        self._social_row.setVisible(False)  # shown only for multi-animal projects

        # Behavior definition
        self._f_description = QTextEdit()
        self._f_description.setFixedHeight(72)
        self._f_description.setPlaceholderText("Describe this behavior so reviewers know what to look for.")

        # Save / Cancel
        self._save_btn = QPushButton("💾 Save Behavior")
        self._cancel_btn = QPushButton("Cancel")
        self._save_btn.clicked.connect(self._save_form)
        self._cancel_btn.clicked.connect(self._cancel_edit)
        save_row = QHBoxLayout()
        save_row.addWidget(self._save_btn)
        save_row.addWidget(self._cancel_btn)
        save_row.addStretch()

        # Full form container
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.addWidget(form_widget)
        container_layout.addWidget(QLabel("Behavior Definition:"))
        container_layout.addWidget(self._f_description)
        container_layout.addLayout(save_row)
        container_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        return scroll

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        self._table.setRowCount(0)
        for b in self._service.behaviors:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(b.name))
            self._table.setItem(row, 1, QTableWidgetItem(b.short_name))

            swatch = QTableWidgetItem("  ")
            swatch.setBackground(QColor(b.color or "#4A90E2"))
            swatch.setData(Qt.ItemDataRole.UserRole, b.behavior_id)
            self._table.setItem(row, 2, swatch)

            self._table.setItem(row, 3, QTableWidgetItem(b.keyboard_shortcut or ""))
            self._table.setItem(row, 4, QTableWidgetItem("✓" if b.is_active else "✗"))

        # Notify dependent tabs (e.g. Active Learning) that behavior options changed.
        self.behaviors_changed.emit()

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._service.set_project(project_root)
        self._load_co_occurring_setting()
        self._apply_multi_animal_visibility()
        self.refresh()
        self._clear_form()
        self._set_form_enabled(False)

    def _apply_multi_animal_visibility(self) -> None:
        """Show social-behavior fields only when the project tracks >1 animal."""
        multi = False
        if self._project_root:
            cfg_path = self._project_root / "project.yaml"
            if cfg_path.exists():
                raw = read_yaml(cfg_path, {})
                multi = int(raw.get("num_animals", 1) or 1) > 1 or not bool(
                    raw.get("single_animal", True)
                )
        self._social_row.setVisible(multi)

    def _load_co_occurring_setting(self) -> None:
        if not self._project_root:
            return
        cfg_path = self._project_root / "project.yaml"
        if not cfg_path.exists():
            return
        raw = read_yaml(cfg_path, {})
        model = raw.get("behavior_model") or {}
        enabled = bool(model.get("allow_co_occurring_behaviors", False))
        self._co_occurring_chk.blockSignals(True)
        self._co_occurring_chk.setChecked(enabled)
        self._co_occurring_chk.blockSignals(False)

    def _on_co_occurring_toggled(self, checked: bool) -> None:
        if not self._project_root:
            return
        cfg_path = self._project_root / "project.yaml"
        raw = read_yaml(cfg_path, {})
        if "behavior_model" not in raw or raw["behavior_model"] is None:
            raw["behavior_model"] = {}
        raw["behavior_model"]["allow_co_occurring_behaviors"] = bool(checked)
        write_yaml(cfg_path, raw)
        logger.info("allow_co_occurring_behaviors set to %s", checked)
        self.co_occurring_changed.emit()

    # ------------------------------------------------------------------
    # Selection / form fill
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        rows = self._table.selectedItems()
        if not rows:
            self._selected_id = None
            self._set_form_enabled(False)
            return
        swatch_item = self._table.item(self._table.currentRow(), 2)
        if not swatch_item:
            return
        bid = swatch_item.data(Qt.ItemDataRole.UserRole)
        self._selected_id = bid
        behavior = self._service.get(bid)
        if behavior:
            self._fill_form(behavior)
            self._set_form_enabled(True)

    def _fill_form(self, b: BehaviorDefinition) -> None:
        self._current_color = b.color or "#4A90E2"
        self._f_name.setText(b.name)
        self._f_short.setText(b.short_name)
        self._set_color_btn(self._current_color)
        self._f_shortcut.setText(b.keyboard_shortcut or "")
        self._f_min_dur.setValue(b.min_duration_sec)
        self._f_active.setChecked(b.is_active)
        self._f_social.setChecked(bool(getattr(b, "is_social", False)))
        direction = getattr(b, "directionality", "none") or "none"
        idx = self._f_directionality.findText(direction)
        self._f_directionality.setCurrentIndex(idx if idx >= 0 else 0)
        self._f_description.setPlainText(b.description)

    def _clear_form(self) -> None:
        self._current_color = "#4A90E2"
        for widget in (
            self._f_name, self._f_short, self._f_shortcut,
            self._f_description,
        ):
            if isinstance(widget, QLineEdit):
                widget.clear()
            else:
                widget.clear()
        self._f_min_dur.setValue(0.5)
        self._f_active.setChecked(True)
        self._f_social.setChecked(False)
        self._f_directionality.setCurrentIndex(0)
        self._set_color_btn(self._current_color)

    def _set_form_enabled(self, enabled: bool) -> None:
        for w in (
            self._f_name, self._f_short, self._f_color_btn, self._f_shortcut,
            self._f_min_dur, self._f_active, self._f_social, self._f_directionality,
            self._f_description,
            self._save_btn, self._cancel_btn, self._delete_btn,
        ):
            w.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _new_behavior(self) -> None:
        self._selected_id = None
        self._clear_form()
        self._set_form_enabled(True)
        self._table.clearSelection()
        self._f_name.setFocus()

    def _save_form(self) -> None:
        name = self._f_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Required", "Behavior name is required.")
            return

        b = BehaviorDefinition(
            behavior_id=self._selected_id or "",
            name=name,
            short_name=self._f_short.text().strip() or name[:6].lower(),
            color=self._current_color,
            keyboard_shortcut=self._f_shortcut.text().strip() or None,
            min_duration_sec=self._f_min_dur.value(),
            is_active=self._f_active.isChecked(),
            is_social=self._f_social.isChecked(),
            directionality=self._f_directionality.currentText() if self._f_social.isChecked() else "none",
            description=self._f_description.toPlainText().strip(),
        )

        if self._selected_id:
            self._service.update(self._selected_id, b)
        else:
            b = self._service.add(b)
            self._selected_id = b.behavior_id
        self.refresh()
        logger.info("Behavior saved: %s", name)

    def _cancel_edit(self) -> None:
        if self._selected_id:
            behavior = self._service.get(self._selected_id)
            if behavior:
                self._fill_form(behavior)
        else:
            self._clear_form()
            self._set_form_enabled(False)

    def _delete_selected(self) -> None:
        if not self._selected_id:
            return
        behavior = self._service.get(self._selected_id)
        if not behavior:
            return
        result = QMessageBox.question(
            self, "Delete Behavior",
            f"Delete '{behavior.name}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return
        self._service.delete(self._selected_id)
        self._selected_id = None
        self._clear_form()
        self._set_form_enabled(False)
        self.refresh()

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._current_color), self, "Pick Behavior Color")
        if color.isValid():
            self._current_color = color.name()
            self._set_color_btn(self._current_color)

    def _set_color_btn(self, color: str) -> None:
        self._f_color_btn.setStyleSheet(
            f"background-color: {color}; border: 1px solid #1565C0; border-radius: 4px;"
        )
        self._f_color_btn.setText(color)

    def _show_template_dialog(self) -> None:
        pass  # Templates removed

    def _export(self) -> None:
        if not self._service.behaviors:
            QMessageBox.information(self, "Export", "No behaviors defined yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Behavior Definitions", "", "YAML files (*.yaml *.yml)"
        )
        if path:
            self._service.export_definitions(Path(path))
            QMessageBox.information(self, "Export", f"Saved to {path}")

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Behavior Definitions", "", "YAML files (*.yaml *.yml)"
        )
        if path:
            added = self._service.import_definitions(Path(path))
            self.refresh()
            QMessageBox.information(self, "Import", f"Imported {added} behavior(s).")
