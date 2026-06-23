"""Apply Models tab — run another project's trained models in this project.

This is Direct Use without creating a new project or re-extracting features: it
copies a source project's trained behaviour models into the *current* project so
they can score this project's already-extracted features.  Once imported, the
models appear wherever models are selected (inference, Visualize, …).

Only models whose feature schema this project's features cover can be imported;
each model's behaviour is mapped onto a project behaviour (or auto-created from
the source's definition) before import.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from abel.services.model_refinement_service import (
    AUTO_CREATE_BEHAVIOR,
    SKIP_BEHAVIOR,
    ModelImportPreview,
    ModelRefinementService,
)
from abel.ui.model_behavior_map_dialog import ModelBehaviorMapDialog

logger = logging.getLogger("abel")

_BTN = (
    "QPushButton { background: #1A2A3A; color: #B0BEC5; font-size: 12px;"
    " border: 1px solid #2A4060; border-radius: 4px; padding: 4px 12px; }"
    "QPushButton:hover { background: #1E3A5F; }"
    "QPushButton:disabled { color: #546E7A; border-color: #1A2A3A; }"
)
_BTN_PRIMARY = (
    "QPushButton { background: #1565C0; color: white; font-size: 13px;"
    " font-weight: 700; border: none; border-radius: 5px; padding: 8px 20px; }"
    "QPushButton:hover { background: #1976D2; }"
    "QPushButton:disabled { background: #263238; color: #546E7A; }"
)


class _ModelImportWorker(QThread):
    """Runs the (I/O-heavy) model copy + rewrite off the UI thread."""

    # Named ``done`` (not ``finished``) so it doesn't shadow QThread's built-in
    # ``finished`` signal used for thread-lifecycle management.
    done = Signal(dict)

    def __init__(
        self,
        host_root: Path,
        source_root: Path,
        model_dirs: list[str],
        decisions: dict[str, str],
        aliases: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._host_root = host_root
        self._source_root = source_root
        self._model_dirs = model_dirs
        self._decisions = decisions
        self._aliases = aliases or {}
        self._svc = ModelRefinementService()

    def run(self) -> None:
        try:
            res = self._svc.import_models(
                self._host_root, self._source_root, self._model_dirs,
                behavior_decisions=self._decisions, name_overrides=self._aliases,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Model import failed for %s", self._source_root)
            res = {"status": "error", "error": str(exc), "imported": [], "skipped": []}
        self.done.emit(res)


class ApplyModelsTab(QWidget):
    """Import another project's trained models into the current project."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._svc = ModelRefinementService()
        self._host_root: Path | None = None
        self._source_root: Path | None = None
        self._preview: ModelImportPreview | None = None
        self._decisions: dict[str, str] = {}  # source_behavior_id -> decision
        self._aliases: dict[str, str] = {}
        self._worker: _ModelImportWorker | None = None

        header = QLabel("Apply Models From Another Project")
        header.setStyleSheet("font-size: 16px; font-weight: 800; color: #90CAF9;")
        desc = QLabel(
            "Run another project's trained models on this project without "
            "re-extracting features or creating a new project. Pick a source "
            "project, map each model's behaviour onto this project (or auto-create "
            "it), and import. Imported models become selectable everywhere models "
            "are used. Only models whose features this project covers can be imported."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #607D8B; padding-bottom: 4px;")

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("+ Select Source Project…")
        self._add_btn.setStyleSheet(_BTN)
        self._add_btn.clicked.connect(self._select_source)
        self._map_btn = QPushButton("Map Behaviours…")
        self._map_btn.setStyleSheet(_BTN)
        self._map_btn.clicked.connect(self._map_behaviours)
        self._map_btn.setEnabled(False)
        self._remove_btn = QPushButton("Remove Imported…")
        self._remove_btn.setStyleSheet(_BTN)
        self._remove_btn.clicked.connect(self._remove_import)
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._map_btn)
        btn_row.addWidget(self._remove_btn)
        btn_row.addStretch(1)
        self._import_btn = QPushButton("Import Selected Models")
        self._import_btn.setStyleSheet(_BTN_PRIMARY)
        self._import_btn.clicked.connect(self._import)
        self._import_btn.setEnabled(False)
        btn_row.addWidget(self._import_btn)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Model", "Behaviour", "Feature match", "Apply as", "Status"]
        )
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, hdr.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, hdr.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, hdr.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, hdr.ResizeMode.Stretch)
        hdr.setSectionResizeMode(4, hdr.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { background: #0A1929; color: #B0BEC5; font-size: 11px;"
            " gridline-color: #1E3A5F; alternate-background-color: #0D2137;"
            " border: 1px solid #1E3A5F; }"
            "QHeaderView::section { background: #102A43; color: #90CAF9;"
            " border: none; padding: 4px; font-weight: 600; }"
        )

        self._detail = QLabel("Select a source project to see its models and compatibility.")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("font-size: 11px; color: #90A4AE; padding-top: 4px;")

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 11px; color: #78909C; padding-top: 2px;")

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.hide()

        layout = QVBoxLayout(self)
        layout.addWidget(header)
        layout.addWidget(desc)
        layout.addLayout(btn_row)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._detail)
        layout.addWidget(self._status)
        layout.addWidget(self._progress)

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------

    def set_project(self, project_root: Path) -> None:
        self._host_root = project_root
        self._aliases = self._svc.load_aliases(project_root)
        self._source_root = None
        self._preview = None
        self._decisions = {}
        self._rebuild_table()

    # ------------------------------------------------------------------
    # Source selection + preview
    # ------------------------------------------------------------------

    def _select_source(self) -> None:
        if self._host_root is None:
            QMessageBox.information(self, "Apply Models", "Open a project first.")
            return
        path = QFileDialog.getExistingDirectory(self, "Select Source ABEL Project")
        if not path:
            return
        src = Path(path)
        if src.resolve() == self._host_root.resolve():
            QMessageBox.information(self, "Apply Models", "That is the current project.")
            return
        try:
            pv = self._svc.preview_model_import(
                self._host_root, src, name_overrides=self._aliases,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Apply Models", f"Could not read project:\n{exc}")
            return
        if not pv.items:
            QMessageBox.information(
                self, "Apply Models",
                f"'{pv.tag}' has no trained behaviour models to import.",
            )
            return
        self._source_root = src
        self._preview = pv
        # Seed decisions from the name/alias match; unmatched default to auto-create.
        self._decisions = {
            it.model.behavior_id: (it.host_behavior_id or AUTO_CREATE_BEHAVIOR)
            for it in pv.items
        }
        self._map_btn.setEnabled(True)
        self._rebuild_table()

    def _map_behaviours(self) -> None:
        if self._preview is None or self._host_root is None:
            return
        host_behaviors = self._svc.list_host_behaviors(self._host_root)
        # One row per distinct source behaviour among the compatible models.
        seen: set[str] = set()
        rows: list[tuple[str, str, str]] = []
        for it in self._preview.items:
            if not it.compatible or it.model.behavior_id in seen:
                continue
            seen.add(it.model.behavior_id)
            rows.append((it.model.behavior_id, it.model.behavior_name, it.host_behavior_id))
        if not rows:
            QMessageBox.information(self, "Apply Models", "No compatible models to map.")
            return
        dlg = ModelBehaviorMapDialog(self._preview.tag, rows, host_behaviors, self)
        if dlg.exec():
            self._decisions.update(dlg.decisions())
            self._rebuild_table()

    # ------------------------------------------------------------------
    # Import + removal
    # ------------------------------------------------------------------

    def _import(self) -> None:
        if self._preview is None or self._host_root is None or self._source_root is None:
            return
        if self._worker is not None:
            return
        model_dirs = [
            it.model.model_dir for it in self._preview.items
            if it.compatible and self._decisions.get(it.model.behavior_id) != SKIP_BEHAVIOR
        ]
        if not model_dirs:
            QMessageBox.information(
                self, "Apply Models",
                "No compatible, non-skipped models selected to import.",
            )
            return
        self._set_busy(True, f"Importing {len(model_dirs)} model(s) from '{self._preview.tag}'…")
        self._worker = _ModelImportWorker(
            self._host_root, self._source_root, model_dirs, dict(self._decisions), self._aliases,
        )
        self._worker.done.connect(self._on_import_done)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_import_done(self, result: dict) -> None:
        self._set_busy(False)
        self._worker = None
        imported = result.get("imported", [])
        skipped = result.get("skipped", [])
        if result.get("status") == "success":
            self._status.setStyleSheet("font-size: 11px; color: #66BB6A; padding-top: 2px;")
            msg = f"Imported {len(imported)} model(s) from '{result.get('tag', '')}'."
            if skipped:
                msg += f" Skipped {len(skipped)}."
            msg += " They're now selectable wherever models are used."
            self._status.setText(msg)
        else:
            self._status.setStyleSheet("font-size: 11px; color: #EF5350; padding-top: 2px;")
            self._status.setText(result.get("error", "Import failed.") or "Import failed.")
        # Refresh against the now-recorded imports; clear the pending preview.
        self._preview = None
        self._source_root = None
        self._decisions = {}
        self._map_btn.setEnabled(False)
        self._rebuild_table()

    def _remove_import(self) -> None:
        if self._host_root is None or self._worker is not None:
            return
        imports = self._svc.list_model_imports(self._host_root)
        if not imports:
            QMessageBox.information(self, "Apply Models", "No imported models to remove.")
            return
        # Remove the import the selected row belongs to.
        row = self._table.currentRow()
        tag = ""
        if row >= 0:
            item = self._table.item(row, 0)
            if item is not None:
                tag = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not tag:
            tag = str(imports[0].get("tag", ""))
        if not tag:
            return
        reply = QMessageBox.question(
            self, "Remove Imported Models",
            f"Remove all models imported from '{tag}'?\n\n"
            "This deletes the copied model directories. Behaviours that were "
            "auto-created stay defined. This project's own models are untouched.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            res = self._svc.remove_model_import(self._host_root, tag)
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.warning(self, "Apply Models", f"Could not remove import:\n{exc}")
            return
        self._status.setStyleSheet("font-size: 11px; color: #66BB6A; padding-top: 2px;")
        self._status.setText(f"Removed {res.get('removed_models', 0)} imported model(s) from '{tag}'.")
        self._rebuild_table()

    # ------------------------------------------------------------------
    # Table rendering
    # ------------------------------------------------------------------

    def _apply_as_text(self, source_behavior_id: str, matched_name: str) -> str:
        decision = self._decisions.get(source_behavior_id) or (matched_name and "match") or AUTO_CREATE_BEHAVIOR
        if decision == AUTO_CREATE_BEHAVIOR:
            return "Auto-create"
        if decision == SKIP_BEHAVIOR:
            return "Skip"
        # Otherwise it's a host behaviour id — show its name.
        if self._host_root is not None:
            for hid, hname in self._svc.list_host_behaviors(self._host_root):
                if hid == decision:
                    return f"→ {hname}"
        return f"→ {matched_name}" if matched_name else "Auto-create"

    def _rebuild_table(self) -> None:
        host_imports = (
            self._svc.list_model_imports(self._host_root) if self._host_root else []
        )
        imported_rows = [
            (rec.get("tag", ""), m) for rec in host_imports for m in rec.get("models", [])
        ]
        pending = self._preview.items if self._preview else []

        self._table.setRowCount(len(pending) + len(imported_rows))
        r = 0
        any_importable = False
        for it in pending:
            tag = self._preview.tag if self._preview else ""
            name_item = QTableWidgetItem(it.model.behavior_name)
            name_item.setData(Qt.ItemDataRole.UserRole, tag)
            beh_item = QTableWidgetItem(it.model.behavior_name)
            cov_item = QTableWidgetItem(f"{it.coverage:.0%}")
            apply_item = QTableWidgetItem(
                self._apply_as_text(it.model.behavior_id, it.host_behavior_name)
            )
            if it.compatible:
                status = "Ready" if self._decisions.get(it.model.behavior_id) != SKIP_BEHAVIOR else "Skipped"
                if status == "Ready":
                    any_importable = True
                status_item = QTableWidgetItem(f"✓ {status}" if status == "Ready" else status)
                status_item.setForeground(QColor("#66BB6A" if status == "Ready" else "#90A4AE"))
            else:
                status_item = QTableWidgetItem(
                    f"Blocked — {it.missing_features} feature(s) missing"
                )
                status_item.setForeground(QColor("#EF5350"))
            for c, item in enumerate((name_item, beh_item, cov_item, apply_item, status_item)):
                self._table.setItem(r, c, item)
            r += 1

        for tag, m in imported_rows:
            name_item = QTableWidgetItem(str(m.get("behavior_name", m.get("model_dir", ""))))
            name_item.setData(Qt.ItemDataRole.UserRole, tag)
            beh_item = QTableWidgetItem(str(m.get("behavior_name", "")))
            cov_item = QTableWidgetItem("imported")
            apply_item = QTableWidgetItem(str(tag))
            status_item = QTableWidgetItem("● Imported")
            status_item.setForeground(QColor("#42A5F5"))
            for c, item in enumerate((name_item, beh_item, cov_item, apply_item, status_item)):
                self._table.setItem(r, c, item)
            r += 1

        self._import_btn.setEnabled(any_importable and self._worker is None)
        self._update_detail()

    def _update_detail(self) -> None:
        if self._preview is None:
            n_imported = sum(
                len(rec.get("models", []))
                for rec in (self._svc.list_model_imports(self._host_root) if self._host_root else [])
            )
            if n_imported:
                self._detail.setText(
                    f"{n_imported} imported model(s) active in this project. "
                    "Select a source project to import more."
                )
            else:
                self._detail.setText(
                    "Select a source project to see its models and compatibility."
                )
            return
        pv = self._preview
        compatible = [i for i in pv.items if i.compatible]
        lines = [
            f"<b>{pv.tag}</b>: {len(compatible)}/{len(pv.items)} model(s) compatible "
            f"with this project's {pv.host_feature_count} feature columns."
        ]
        if pv.keypoint_renames:
            lines.append(
                f"Keypoint names realigned ({len(pv.keypoint_renames)} renamed) so the "
                "models read this project's columns."
            )
        d = pv.diagnostics
        if d is not None and d.config_mismatches:
            lines.append("⚠ Extraction settings differ: " + "; ".join(d.config_mismatches))
        self._detail.setText("<br>".join(lines))

    # ------------------------------------------------------------------
    # Busy state
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._add_btn.setEnabled(not busy)
        self._map_btn.setEnabled(not busy and self._preview is not None)
        self._import_btn.setEnabled(not busy)
        self._remove_btn.setEnabled(not busy)
        self._progress.setVisible(busy)
        if message:
            self._status.setStyleSheet("font-size: 11px; color: #78909C; padding-top: 2px;")
            self._status.setText(message)
