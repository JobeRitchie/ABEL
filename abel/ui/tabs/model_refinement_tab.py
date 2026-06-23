"""Model Refinement tab — refine this project's models with labeled examples
imported from other ABEL projects.

This tab merges *labeled training examples* (segment features + their reviewer
labels) from one or more source projects into the current project's training
set.  After importing, retrain in the Active Learning tab to produce refined
models that have learned from the larger, more diverse dataset.

Only projects that share this project's pose-keypoint / feature schema can be
imported; incompatible projects are detected and blocked with an explanation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
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

from abel.ui.behavior_remap_dialog import BehaviorRemapDialog
from abel.services.model_refinement_service import (
    COMPAT_THRESHOLD,
    ImportRecord,
    ModelRefinementService,
    RefinementPreview,
)

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


class _ImportWorker(QThread):
    """Runs the (I/O-heavy) example import off the UI thread."""

    # Named ``done`` (not ``finished``) so it doesn't shadow QThread's built-in
    # ``finished`` signal, which Qt uses for thread-lifecycle management.
    done = Signal(list)  # list[dict] results

    def __init__(
        self,
        host_root: Path,
        sources: list[Path],
        aliases: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._host_root = host_root
        self._sources = sources
        self._aliases = aliases or {}
        self._svc = ModelRefinementService()

    def run(self) -> None:
        results: list[dict] = []
        for src in self._sources:
            try:
                res = self._svc.import_examples(
                    self._host_root, src, name_overrides=self._aliases
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Import failed for %s", src)
                res = {"status": "error", "error": str(exc), "tag": src.name}
            res.setdefault("tag", src.name)
            res.setdefault("source_root", str(src))
            results.append(res)
        self.done.emit(results)


class ModelRefinementTab(QWidget):
    """Import labeled examples from other projects, then retrain."""

    retrain_requested = Signal()  # ask MainWindow to switch to Active Learning

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._svc = ModelRefinementService()
        self._host_root: Path | None = None
        self._previews: dict[str, RefinementPreview] = {}  # str(path) -> preview
        self._imports_by_key: dict[str, ImportRecord] = {}  # "import::<tag>" -> record
        self._worker: _ImportWorker | None = None
        self._aliases: dict[str, str] = {}  # {source_name_lower: host_name}

        header = QLabel("Model Refinement")
        header.setStyleSheet("font-size: 16px; font-weight: 800; color: #90CAF9;")
        desc = QLabel(
            "Refine this project's models by importing labeled examples from "
            "other ABEL projects. The imported segments' features and behaviour "
            "labels are added to this project's training set; retrain in the "
            "Active Learning tab to produce improved models.\n\n"
            "Only projects that use the same pose keypoints and ROI layout are "
            "compatible — others are detected and blocked automatically."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #607D8B; padding-bottom: 4px;")

        # ── Source list ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("+ Add Source Project…")
        self._add_btn.setStyleSheet(_BTN)
        self._add_btn.clicked.connect(self._add_source)
        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.setStyleSheet(_BTN)
        self._remove_btn.clicked.connect(self._remove_selected)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setStyleSheet(_BTN)
        self._clear_btn.clicked.connect(self._clear_sources)
        self._remove_import_btn = QPushButton("Remove Import")
        self._remove_import_btn.setStyleSheet(_BTN)
        self._remove_import_btn.setToolTip(
            "Cleanly un-import the selected already-imported project: delete its "
            "imported training rows, review clips, and review entries."
        )
        self._remove_import_btn.setEnabled(False)
        self._remove_import_btn.clicked.connect(self._remove_import)
        self._remap_btn = QPushButton("Map Behaviour Names…")
        self._remap_btn.setStyleSheet(_BTN)
        self._remap_btn.setToolTip(
            "Match behaviours that have different names across projects but are "
            "really the same (e.g. Head Dip → Dip) so their examples import."
        )
        self._remap_btn.setEnabled(False)
        self._remap_btn.clicked.connect(self._open_remap_dialog)
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._remove_btn)
        btn_row.addWidget(self._clear_btn)
        btn_row.addWidget(self._remove_import_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._remap_btn)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Source Project", "Schema Match", "Importable Examples", "Status"]
        )
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, hdr.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, hdr.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, hdr.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, hdr.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { background: #0A1929; color: #B0BEC5; font-size: 11px;"
            " gridline-color: #1E3A5F; alternate-background-color: #0D2137;"
            " border: 1px solid #1E3A5F; }"
            "QHeaderView::section { background: #0F2744; color: #78909C; font-size: 11px;"
            " font-weight: 600; padding: 3px 6px; border-bottom: 1px solid #1E3A5F; }"
        )
        self._table.currentCellChanged.connect(lambda *_: self._refresh_detail())

        # ── Detail / mapping panel ────────────────────────────────────
        self._detail = QLabel("Add a source project to see its compatibility and "
                              "behaviour mapping.")
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._detail.setStyleSheet(
            "font-size: 11px; color: #B0BEC5; background: #0A1929;"
            " border: 1px solid #1E3A5F; border-radius: 4px; padding: 8px;"
        )
        self._detail.setMinimumHeight(120)
        self._detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # ── Actions ───────────────────────────────────────────────────
        action_row = QHBoxLayout()
        self._import_btn = QPushButton("Import Compatible Examples")
        self._import_btn.setStyleSheet(_BTN_PRIMARY)
        self._import_btn.setEnabled(False)
        self._import_btn.clicked.connect(self._start_import)
        self._retrain_btn = QPushButton("Go to Active Learning to Retrain →")
        self._retrain_btn.setStyleSheet(_BTN)
        self._retrain_btn.clicked.connect(self.retrain_requested)
        action_row.addWidget(self._import_btn)
        action_row.addStretch(1)
        action_row.addWidget(self._retrain_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 11px; color: #78909C; padding-top: 2px;")

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(8)
        root.addWidget(header)
        root.addWidget(desc)
        root.addLayout(btn_row)
        root.addWidget(self._table, 1)
        root.addWidget(QLabel("Details:"))
        root.addWidget(self._detail)
        root.addLayout(action_row)
        root.addWidget(self._progress)
        root.addWidget(self._status)

    # ── Public API ─────────────────────────────────────────────────────

    def set_project(self, project_root: Path) -> None:
        self._host_root = project_root
        self._aliases = self._svc.load_aliases(project_root)
        self._previews.clear()
        self._reload_imports()

    def _reload_imports(self) -> None:
        """Load the persisted list of already-imported sources for this project."""
        self._imports_by_key = {}
        if self._host_root is not None:
            for rec in self._svc.list_imports(self._host_root):
                self._imports_by_key[f"import::{rec.tag}"] = rec
        self._rebuild_table()

    # ── Source management ───────────────────────────────────────────────

    def _add_source(self) -> None:
        if self._host_root is None:
            QMessageBox.information(self, "Model Refinement", "Open a project first.")
            return
        path = QFileDialog.getExistingDirectory(self, "Select Source ABEL Project")
        if not path:
            return
        src = Path(path)
        if src.resolve() == self._host_root.resolve():
            QMessageBox.information(
                self, "Model Refinement", "That is the current project.")
            return
        if str(src) in self._previews:
            return
        try:
            pv = self._svc.preview(
                self._host_root, src, name_overrides=self._aliases
            )
        except Exception as exc:
            QMessageBox.warning(self, "Model Refinement", f"Could not read project:\n{exc}")
            return
        self._previews[str(src)] = pv
        self._rebuild_table()

    def _remove_selected(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        item = self._table.item(row, 0)
        if item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        self._previews.pop(str(key), None)
        self._rebuild_table()

    def _clear_sources(self) -> None:
        # Clears only the pending (not-yet-imported) sources; already-imported
        # sources persist and stay listed.
        self._previews.clear()
        self._rebuild_table()
        self._detail.setText("Add a source project to see its compatibility and "
                             "behaviour mapping.")
        self._status.setText("")

    def _remove_import(self) -> None:
        if self._host_root is None or self._worker is not None:
            return
        row = self._table.currentRow()
        if row < 0:
            return
        item = self._table.item(row, 0)
        if item is None:
            return
        rec = self._imports_by_key.get(str(item.data(Qt.ItemDataRole.UserRole)))
        if rec is None:
            return

        reply = QMessageBox.question(
            self, "Remove Import",
            f"Remove all imported data from '{rec.tag}'?\n\n"
            f"This deletes {rec.imported_rows} imported training row(s), their "
            "copied review clips, and their review entries. This project's own "
            "data is untouched. Retrain afterwards to drop them from the model.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        from PySide6.QtWidgets import QApplication
        self._remove_import_btn.setEnabled(False)
        self._status.setStyleSheet("font-size: 11px; color: #78909C; padding-top: 2px;")
        self._status.setText(f"Removing import '{rec.tag}'…")
        res = None
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            res = self._svc.remove_import(self._host_root, rec.tag)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Remove import failed for %s", rec.tag)
            QMessageBox.warning(
                self, "Remove Import", f"Could not remove import:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

        if res and res.get("status") == "success":
            self._status.setStyleSheet("font-size: 11px; color: #66BB6A; padding-top: 2px;")
            self._status.setText(
                f"Removed '{rec.tag}': {res.get('removed_rows', 0)} training row(s), "
                f"{res.get('removed_clips', 0)} clip(s), "
                f"{res.get('removed_decisions', 0)} review entr(ies). Retrain to apply."
            )
        self._reload_imports()

    def _rebuild_table(self) -> None:
        self._table.setRowCount(len(self._previews) + len(self._imports_by_key))
        r = 0
        for key, pv in self._previews.items():
            name_item = QTableWidgetItem(pv.tag)
            name_item.setData(Qt.ItemDataRole.UserRole, key)
            name_item.setToolTip(str(key))

            match_item = QTableWidgetItem(f"{pv.coverage:.0%}")
            count_item = QTableWidgetItem(str(pv.importable_labeled))
            if pv.compatible:
                status_item = QTableWidgetItem("✓ Ready")
                status_item.setForeground(QColor("#66BB6A"))
            else:
                status_item = QTableWidgetItem("✕ Blocked")
                status_item.setForeground(QColor("#EF5350"))
                status_item.setToolTip(pv.reason)
            for it in (name_item, match_item, count_item, status_item):
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 0, name_item)
            self._table.setItem(r, 1, match_item)
            self._table.setItem(r, 2, count_item)
            self._table.setItem(r, 3, status_item)
            r += 1

        # Already-imported sources (persisted), shown below pending sources.
        for key, rec in self._imports_by_key.items():
            name_item = QTableWidgetItem(rec.tag)
            name_item.setData(Qt.ItemDataRole.UserRole, key)
            name_item.setToolTip(rec.source_root or rec.tag)
            match_item = QTableWidgetItem("—")
            count_item = QTableWidgetItem(str(rec.imported_rows))
            status_item = QTableWidgetItem("✓ Imported")
            status_item.setForeground(QColor("#42A5F5"))
            for it in (name_item, match_item, count_item, status_item):
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 0, name_item)
            self._table.setItem(r, 1, match_item)
            self._table.setItem(r, 2, count_item)
            self._table.setItem(r, 3, status_item)
            r += 1

        n_ready = sum(1 for pv in self._previews.values() if pv.compatible)
        self._import_btn.setEnabled(n_ready > 0 and self._worker is None)
        if self._table.rowCount() > 0 and self._table.currentRow() < 0:
            self._table.setCurrentCell(0, 0)
        self._refresh_detail()

    def _refresh_detail(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= self._table.rowCount():
            self._remap_btn.setEnabled(False)
            self._remove_import_btn.setEnabled(False)
            return
        item = self._table.item(row, 0)
        if item is None:
            self._remap_btn.setEnabled(False)
            self._remove_import_btn.setEnabled(False)
            return
        key = str(item.data(Qt.ItemDataRole.UserRole))

        # Already-imported source: show its record + enable clean removal.
        rec = self._imports_by_key.get(key)
        if rec is not None:
            self._remap_btn.setEnabled(False)
            self._remove_import_btn.setEnabled(self._worker is None)
            lines = [
                f"Imported source: {rec.tag}",
                f"From: {rec.source_root or '(unknown)'}",
                f"Training rows: {rec.imported_rows}",
                f"Review clips registered: {rec.review_registered}",
            ]
            if rec.imported_at:
                lines.append(f"Imported: {rec.imported_at}")
            if rec.behaviors:
                lines.append(
                    "Behaviours: "
                    + ", ".join(f"{n} ({c})" for n, c in rec.behaviors.items())
                )
            lines.append(
                "\nUse “Remove Import” to cleanly delete this source's imported "
                "training rows, copied review clips, and review entries. Retrain "
                "afterwards to drop them from the model."
            )
            self._detail.setText("\n".join(lines))
            return

        self._remove_import_btn.setEnabled(False)
        pv = self._previews.get(key)
        if pv is None:
            self._remap_btn.setEnabled(False)
            return
        # Remapping only helps when the feature schema is compatible; otherwise
        # the import is blocked regardless of behaviour names.
        schema_ok = pv.coverage >= COMPAT_THRESHOLD and pv.host_feature_count > 0
        self._remap_btn.setEnabled(schema_ok and self._worker is None)

        lines: list[str] = []
        lines.append(
            f"Feature schema: {pv.shared_feature_count}/{pv.host_feature_count} "
            f"of this project's columns present in source ({pv.coverage:.0%})."
        )
        if pv.keypoint_renames:
            pairs = ", ".join(
                f"{src}→{host}" for src, host in sorted(pv.keypoint_renames.items())
            )
            lines.append(
                f"Keypoints remapped onto this project's scheme: {pairs}."
            )
        lines.extend(self._diagnostics_lines(pv.diagnostics))
        if not pv.compatible:
            lines.append(f"\n⚠  {pv.reason}")
        matched = pv.matched_behaviors
        if matched:
            lines.append("\nBehaviours that will be imported:")
            for m in matched:
                via = "  (remapped)" if m.remapped else ""
                lines.append(
                    f"   • {m.host_name}: {m.example_count} example(s){via}"
                )
        unmatched = pv.unmatched_behaviors
        if unmatched:
            names = ", ".join(
                f"{m.source_name} ({m.example_count})" for m in unmatched
            )
            lines.append(
                f"\nSkipped (no matching behaviour in this project): {names}"
            )
            if schema_ok:
                lines.append(
                    "Use “Map Behaviour Names…” to match any of these to a "
                    "host behaviour."
                )
        self._detail.setText("\n".join(lines))

    @staticmethod
    def _diagnostics_lines(diag) -> list[str]:
        """Render the value-level project-similarity diagnostics, if present.

        These never block an import — they help judge whether merging is
        *scientifically* sound, beyond the schema matching mechanically.
        """
        if diag is None:
            return []
        lines = ["\nProject comparison (informational):"]

        # Net feature-value shift — the headline "are values comparable" signal.
        if diag.feature_shift_median is not None:
            band = (
                "low" if diag.feature_shift_median < 0.25
                else "moderate" if diag.feature_shift_median < 0.5
                else "high"
            )
            base = diag.within_host_shift_median
            base_txt = f"; within-project baseline {base:.2f}" if base is not None else ""
            frac = diag.feature_shift_frac_gt_half
            frac_txt = f", {frac:.0%} of columns >0.5" if frac is not None else ""
            lines.append(
                f"   • Feature-value shift: {diag.feature_shift_median:.2f} IQR "
                f"({band}){frac_txt}{base_txt}."
            )

        # Spatial calibration — expected to differ with camera/resolution setup;
        # body-length-normalised features are scale-invariant, so this is context.
        if diag.host_px_per_mm is not None and diag.source_px_per_mm is not None:
            pct = diag.px_per_mm_pct_diff or 0.0
            note = (
                "  (likely a camera/resolution difference; normalised features "
                "are scale-invariant)" if pct >= 5 else ""
            )
            lines.append(
                f"   • Calibration: {diag.host_px_per_mm:.3f} vs "
                f"{diag.source_px_per_mm:.3f} px/mm ({pct:.0f}% diff){note}"
            )

        # Pose model — same DLC network == most directly comparable keypoints.
        if diag.host_pose_models or diag.source_pose_models:
            if diag.pose_models_match:
                lines.append(
                    f"   • Pose model: same network "
                    f"({', '.join(diag.host_pose_models)})."
                )
            else:
                lines.append(
                    f"   • Pose model: differs — host {diag.host_pose_models or ['?']} "
                    f"vs source {diag.source_pose_models or ['?']}. Same keypoints, "
                    "but a different DLC network can shift point placement slightly."
                )

        # Feature-extraction settings.
        if diag.config_mismatches:
            lines.append(
                "   • ⚠  Extraction settings differ: "
                + "; ".join(diag.config_mismatches)
                + ". Features may not be directly comparable."
            )
        else:
            lines.append("   • Extraction settings: aligned.")
        return lines

    # ── Behaviour name remapping ────────────────────────────────────────

    def _open_remap_dialog(self) -> None:
        if self._host_root is None:
            return
        row = self._table.currentRow()
        if row < 0 or row >= self._table.rowCount():
            return
        item = self._table.item(row, 0)
        if item is None:
            return
        pv = self._previews.get(str(item.data(Qt.ItemDataRole.UserRole)))
        if pv is None:
            return

        host_behaviors = self._svc.list_host_behaviors(self._host_root)
        host_names = [name for _, name in host_behaviors]
        if not host_names:
            QMessageBox.information(
                self, "Map Behaviour Names",
                "This project has no behaviours defined to map onto.",
            )
            return

        # Rows = source behaviours not already matched by exact name
        # (unmatched, or previously remapped so the user can adjust them).
        rows: list[tuple[str, int, str]] = []
        for m in pv.behavior_mappings:
            if m.source_behavior_id in {"no_behavior"}:
                continue
            exact = m.matched and not m.remapped
            if exact or m.example_count <= 0:
                continue
            suggested = self._svc.suggest_host_match(m.source_name, host_names)
            rows.append((m.source_name, m.example_count, suggested))

        if not rows:
            QMessageBox.information(
                self, "Map Behaviour Names",
                f"Every behaviour in '{pv.tag}' already matches a behaviour in "
                "this project — nothing to remap.",
            )
            return

        dlg = BehaviorRemapDialog(
            pv.tag, rows, host_names, current_aliases=self._aliases, parent=self,
        )
        if dlg.exec() != BehaviorRemapDialog.DialogCode.Accepted:
            return

        chosen = dlg.aliases()
        # Replace mappings for the source behaviours shown in this dialog
        # (so de-selecting one clears it), keep aliases for other projects.
        shown = {name.lower() for name, _, _ in rows}
        self._aliases = {
            k: v for k, v in self._aliases.items() if k not in shown
        }
        self._aliases.update(chosen)
        try:
            self._svc.save_aliases(self._host_root, self._aliases)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to save behavior aliases")
            QMessageBox.warning(
                self, "Map Behaviour Names",
                f"Mappings applied for this session but could not be saved:\n{exc}",
            )
        self._repreview_all()
        n = len(chosen)
        self._status.setStyleSheet(
            "font-size: 11px; color: #66BB6A; padding-top: 2px;"
        )
        self._status.setText(
            f"Applied {n} behaviour remapping(s)." if n else
            "Behaviour remappings cleared."
        )

    def _repreview_all(self) -> None:
        """Recompute every source preview with the current alias table."""
        if self._host_root is None:
            return
        for key in list(self._previews):
            try:
                self._previews[key] = self._svc.preview(
                    self._host_root, Path(key), name_overrides=self._aliases,
                )
            except Exception:
                logger.exception("Re-preview failed for %s", key)
        self._rebuild_table()

    # ── Import ──────────────────────────────────────────────────────────

    def _start_import(self) -> None:
        if self._host_root is None or self._worker is not None:
            return
        ready = [Path(k) for k, pv in self._previews.items() if pv.compatible]
        if not ready:
            return
        total = sum(
            pv.importable_labeled for pv in self._previews.values() if pv.compatible
        )
        reply = QMessageBox.question(
            self, "Import Examples",
            f"Import {total} labeled example(s) from {len(ready)} project(s) "
            "into this project's training set?\n\n"
            "Existing training data is preserved; retrain afterwards to apply.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._import_btn.setEnabled(False)
        self._add_btn.setEnabled(False)
        self._remap_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._status.setText("Importing examples…")
        self._status.setStyleSheet("font-size: 11px; color: #78909C; padding-top: 2px;")

        self._worker = _ImportWorker(self._host_root, ready, aliases=self._aliases)
        self._worker.done.connect(self._on_import_finished)
        self._worker.start()

    def _on_import_finished(self, results: list[dict]) -> None:
        if self._worker is not None:
            self._worker.wait()
            self._worker.deleteLater()
            self._worker = None
        self._progress.setVisible(False)
        self._add_btn.setEnabled(True)

        total_imported = sum(
            int(r.get("imported_rows", 0)) for r in results if r.get("status") == "success"
        )
        errors = [r for r in results if r.get("status") != "success"]
        if total_imported > 0:
            msg = f"Imported {total_imported} example(s)."
            if errors:
                msg += f" {len(errors)} project(s) failed."
            msg += " Retrain in Active Learning to apply."
            self._status.setStyleSheet("font-size: 11px; color: #66BB6A; padding-top: 2px;")
            self._status.setText(msg)
        else:
            reason = errors[0].get("error", "Unknown error") if errors else "Nothing imported."
            self._status.setStyleSheet("font-size: 11px; color: #EF5350; padding-top: 2px;")
            self._status.setText(f"Import failed: {reason}")

        # Successfully imported sources move into the persisted "Imported" list;
        # drop their pending previews and reload from the manifest.
        if total_imported > 0:
            for r in results:
                if r.get("status") == "success":
                    self._previews.pop(str(r.get("source_root", "")), None)
        self._reload_imports()
