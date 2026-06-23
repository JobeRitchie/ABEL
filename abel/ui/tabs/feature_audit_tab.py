"""Feature audit tab — detect bodyparts and dead/weak features before training."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, Signal


class FeatureAuditTab(QWidget):
    """Scan pose files and derived features to identify dead/weak columns."""

    _audit_finished_signal = Signal(object)  # FeatureAuditResult | Exception

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger("abel")
        self._project_root: Path | None = None
        self._audit_result = None
        self._bp_checkboxes: dict[str, QCheckBox] = {}

        self._build_ui()
        self._audit_finished_signal.connect(self._on_audit_finished)

    def set_project(self, project_root: Path | None) -> None:
        self._project_root = project_root

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        help_label = QLabel(
            "Scan pose files to identify which body parts are detected and which "
            "derived features carry real signal.\nDead features (all-zero or all-NaN) "
            "are automatically excluded during training. You can also manually omit body "
            "parts or features here."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        # ── Buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._audit_run_btn = QPushButton("Run Feature Audit")
        self._audit_run_btn.setToolTip(
            "Scan all pose files and the training set to detect body parts,\n"
            "identify dead/weak features, and recommend exclusions."
        )
        self._audit_run_btn.clicked.connect(self._run_feature_audit)

        self._audit_apply_btn = QPushButton("Apply Recommended Exclusions")
        self._audit_apply_btn.setToolTip(
            "Auto-exclude dead features from future training runs.\n"
            "This updates the excluded feature list used by the active learning pipeline."
        )
        self._audit_apply_btn.clicked.connect(self._apply_audit_exclusions)
        self._audit_apply_btn.setEnabled(False)

        btn_row.addWidget(self._audit_run_btn)
        btn_row.addWidget(self._audit_apply_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._audit_status_label = QLabel("")
        self._audit_status_label.setWordWrap(True)
        layout.addWidget(self._audit_status_label)

        # ── Body Parts table ──────────────────────────────────────
        bp_group = QGroupBox("Detected Body Parts")
        bp_layout = QVBoxLayout(bp_group)
        bp_help = QLabel(
            "Body parts found across pose files. Uncheck to exclude all features "
            "derived from a body part."
        )
        bp_help.setWordWrap(True)
        bp_layout.addWidget(bp_help)

        self._bp_table = QTableWidget(0, 6)
        self._bp_table.setHorizontalHeaderLabels([
            "Include", "Body Part", "Sessions", "Coverage", "Mean Likelihood", "Low Quality %",
        ])
        self._bp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._bp_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        bp_layout.addWidget(self._bp_table)
        layout.addWidget(bp_group)

        # ── Feature Health table ──────────────────────────────────
        feat_group = QGroupBox("Feature Health Summary")
        feat_layout = QVBoxLayout(feat_group)

        self._feat_summary_label = QLabel("")
        self._feat_summary_label.setWordWrap(True)
        feat_layout.addWidget(self._feat_summary_label)

        self._feat_table = QTableWidget(0, 7)
        self._feat_table.setHorizontalHeaderLabels([
            "Feature", "Family", "Body Part", "Nonzero %", "NaN %", "Status", "Importance",
        ])
        self._feat_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._feat_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._feat_table.setSortingEnabled(True)
        feat_layout.addWidget(self._feat_table)
        layout.addWidget(feat_group)

        # ── Feature Importance table ──────────────────────────────
        imp_group = QGroupBox("Feature Importance (from trained models)")
        imp_layout = QVBoxLayout(imp_group)

        self._imp_summary_label = QLabel(
            "Run a training round first to populate feature importance scores. "
            "Shows per-model importance (XGBoost gain). Click any column header to sort."
        )
        self._imp_summary_label.setWordWrap(True)
        imp_layout.addWidget(self._imp_summary_label)

        self._imp_table = QTableWidget(0, 1)
        self._imp_table.setHorizontalHeaderLabels(["Feature"])
        self._imp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._imp_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._imp_table.setSortingEnabled(True)
        imp_layout.addWidget(self._imp_table)

        self._imp_refresh_btn = QPushButton("Refresh Importance Scores")
        self._imp_refresh_btn.setToolTip(
            "Load feature importance from the latest trained models.\n"
            "Requires at least one completed training run."
        )
        self._imp_refresh_btn.clicked.connect(self._refresh_feature_importance)
        imp_layout.addWidget(self._imp_refresh_btn)
        layout.addWidget(imp_group)

        # ── Log panel ─────────────────────────────────────────────
        self._log_panel = QTextEdit()
        self._log_panel.setReadOnly(True)
        self._log_panel.setMaximumHeight(120)
        layout.addWidget(QLabel("Audit log"))
        layout.addWidget(self._log_panel)

    # ------------------------------------------------------------------
    # Audit execution
    # ------------------------------------------------------------------

    def _run_feature_audit(self) -> None:
        """Run the feature audit in a background thread."""
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return

        self._audit_run_btn.setEnabled(False)
        self._audit_status_label.setText("Running feature audit — scanning pose files and features…")
        self._log_panel.append("Feature audit started…")

        project_root = self._project_root

        def _do_audit():
            try:
                from abel.services.feature_audit_service import FeatureAuditService
                svc = FeatureAuditService()
                result = svc.audit_features(project_root)
                if not result.bodyparts:
                    result.bodyparts = svc.detect_bodyparts(project_root)
                svc.save_audit_report(project_root, result)
                self._audit_finished_signal.emit(result)
            except Exception as exc:
                self._audit_finished_signal.emit(exc)

        t = threading.Thread(target=_do_audit, daemon=True)
        t.start()

    def _on_audit_finished(self, result: object) -> None:
        """Handle audit completion on the GUI thread."""
        self._audit_run_btn.setEnabled(True)

        if isinstance(result, Exception):
            self._audit_status_label.setText(f"Feature audit failed: {result}")
            self._log_panel.append(f"Feature audit error: {result}")
            return

        from abel.services.feature_audit_service import FeatureAuditResult
        if not isinstance(result, FeatureAuditResult):
            return

        self._audit_result = result
        self._audit_apply_btn.setEnabled(bool(result.recommended_exclusions))

        # ── Populate body parts table ─────────────────────────────
        self._bp_checkboxes.clear()
        self._bp_table.setRowCount(len(result.bodyparts))
        for row, bp in enumerate(result.bodyparts):
            cb = QCheckBox()
            cb.setChecked(True)
            self._bp_checkboxes[bp.name] = cb
            self._bp_table.setCellWidget(row, 0, cb)
            self._bp_table.setItem(row, 1, QTableWidgetItem(bp.name))
            self._bp_table.setItem(row, 2, QTableWidgetItem(
                f"{bp.sessions_present}/{bp.sessions_total}"
            ))
            self._bp_table.setItem(row, 3, QTableWidgetItem(f"{bp.coverage:.0%}"))
            self._bp_table.setItem(row, 4, QTableWidgetItem(f"{bp.mean_likelihood:.3f}"))
            self._bp_table.setItem(row, 5, QTableWidgetItem(f"{bp.low_likelihood_fraction:.1%}"))

            if bp.coverage < 0.5 or bp.mean_likelihood < 0.15:
                cb.setChecked(False)

        # ── Populate feature health table ─────────────────────────
        n_dead = len(result.dead_feature_names)
        n_weak = len(result.weak_feature_names)
        n_total = len(result.features)
        n_healthy = n_total - n_dead - n_weak

        self._feat_summary_label.setText(
            f"<b>{n_total}</b> features scanned: "
            f"<span style='color: green;'>{n_healthy} healthy</span>, "
            f"<span style='color: orange;'>{n_weak} weak (&gt;95% zero)</span>, "
            f"<span style='color: red;'>{n_dead} dead (no signal)</span>"
        )

        # Load feature importance if available.
        avg_importance: dict[str, float] = {}
        if self._project_root:
            try:
                from abel.services.feature_audit_service import FeatureAuditService as _FAS
                per_model = _FAS.load_feature_importance(self._project_root)
                avg_importance = _FAS.aggregate_feature_importance(per_model)
            except Exception:
                pass

        dead_set = set(result.dead_feature_names)
        weak_set = set(result.weak_feature_names)
        sorted_features = sorted(
            result.features,
            key=lambda f: (0 if f.name in dead_set else (1 if f.name in weak_set else 2), f.name),
        )

        self._feat_table.setSortingEnabled(False)
        self._feat_table.setRowCount(len(sorted_features))
        for row, feat in enumerate(sorted_features):
            self._feat_table.setItem(row, 0, QTableWidgetItem(feat.name))
            self._feat_table.setItem(row, 1, QTableWidgetItem(feat.family))
            self._feat_table.setItem(row, 2, QTableWidgetItem(feat.source_bodypart or "—"))
            self._feat_table.setItem(row, 3, QTableWidgetItem(f"{feat.nonzero_fraction:.1%}"))
            self._feat_table.setItem(row, 4, QTableWidgetItem(f"{feat.nan_fraction:.1%}"))

            if feat.name in dead_set:
                status = "DEAD"
            elif feat.name in weak_set:
                status = "WEAK"
            else:
                status = "OK"
            status_item = QTableWidgetItem(status)
            if status == "DEAD":
                status_item.setForeground(Qt.GlobalColor.red)
            elif status == "WEAK":
                status_item.setForeground(Qt.GlobalColor.darkYellow)
            self._feat_table.setItem(row, 5, status_item)

            imp_val = avg_importance.get(feat.name)
            if imp_val is not None:
                imp_item = QTableWidgetItem()
                imp_item.setData(Qt.ItemDataRole.DisplayRole, round(imp_val, 4))
                self._feat_table.setItem(row, 6, imp_item)
            else:
                self._feat_table.setItem(row, 6, QTableWidgetItem("—"))
        self._feat_table.setSortingEnabled(True)

        self._audit_status_label.setText(
            f"Audit complete: {len(result.bodyparts)} body parts, "
            f"{n_total} features ({n_dead} dead, {n_weak} weak). "
            + (f"{n_dead} recommended exclusions ready to apply."
               if n_dead else "No dead features found.")
        )
        self._log_panel.append(
            f"Feature audit complete: {n_dead} dead, {n_weak} weak out of {n_total} features."
        )

        # Auto-refresh the importance table as well.
        self._refresh_feature_importance()

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def _refresh_feature_importance(self) -> None:
        """Load and display per-model feature importance."""
        if not self._project_root:
            self._imp_summary_label.setText("No project loaded.")
            return

        from abel.services.feature_audit_service import FeatureAuditService

        per_model = FeatureAuditService.load_feature_importance(self._project_root)
        if not per_model:
            self._imp_summary_label.setText(
                "No feature importance data found. Train at least one model first."
            )
            self._imp_table.setRowCount(0)
            return

        # Skip dead models whose total importance is zero.
        live_models: dict[str, dict[str, float]] = {}
        skipped: list[str] = []
        for model_name, imp in per_model.items():
            if sum(imp.values()) > 0:
                live_models[model_name] = imp
            else:
                skipped.append(model_name)

        if not live_models:
            self._imp_summary_label.setText(
                "All models have zero feature importance (no splits learned). "
                "Retrain models with more labelled data."
            )
            self._imp_table.setRowCount(0)
            return

        # Collect the union of all feature names.
        all_features: set[str] = set()
        for imp in live_models.values():
            all_features.update(imp.keys())

        # Short display names: strip "behavior_model_" prefix.
        model_display: list[tuple[str, str]] = []
        for m in sorted(live_models):
            short = m.replace("behavior_model_", "") if m.startswith("behavior_model_") else m
            model_display.append((m, short))

        n_cols = 1 + len(model_display)  # Feature + one col per model
        headers = ["Feature"] + [short for _, short in model_display]

        self._imp_table.setSortingEnabled(False)
        self._imp_table.setColumnCount(n_cols)
        self._imp_table.setHorizontalHeaderLabels(headers)

        sorted_features = sorted(all_features)
        self._imp_table.setRowCount(len(sorted_features))

        for row, feat_name in enumerate(sorted_features):
            self._imp_table.setItem(row, 0, QTableWidgetItem(feat_name))
            for col_idx, (model_key, _short) in enumerate(model_display, start=1):
                val = live_models[model_key].get(feat_name, 0.0)
                item = QTableWidgetItem()
                item.setData(Qt.ItemDataRole.DisplayRole, round(val, 6))
                self._imp_table.setItem(row, col_idx, item)

        self._imp_table.setSortingEnabled(True)
        self._imp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        skipped_msg = ""
        if skipped:
            short_skipped = [s.replace("behavior_model_", "") for s in skipped]
            skipped_msg = f" Skipped {len(skipped)} model(s) with no learned splits: {', '.join(short_skipped)}."

        self._imp_summary_label.setText(
            f"<b>{len(all_features)}</b> features across "
            f"<b>{len(live_models)}</b> model(s). "
            f"Click any column header to sort.{skipped_msg}"
        )

        self._log_panel.append(
            f"Feature importance loaded: {len(all_features)} features from "
            f"{len(live_models)} model(s) ({len(skipped)} skipped — no splits)."
        )

    # ------------------------------------------------------------------
    # Apply exclusions
    # ------------------------------------------------------------------

    def _apply_audit_exclusions(self) -> None:
        """Apply recommended exclusions + unchecked bodyparts to the project config."""
        if not self._audit_result or not self._project_root:
            return

        exclusions: set[str] = set(self._audit_result.recommended_exclusions)

        unchecked_bodyparts: list[str] = []
        for bp_name, cb in self._bp_checkboxes.items():
            if not cb.isChecked():
                unchecked_bodyparts.append(bp_name)

        if unchecked_bodyparts:
            for feat in self._audit_result.features:
                for bp in unchecked_bodyparts:
                    if feat.name.startswith(bp) or f"_{bp}" in feat.name:
                        exclusions.add(feat.name)
                        break

        if not exclusions:
            QMessageBox.information(self, "No Exclusions", "No features to exclude.")
            return

        try:
            from abel.storage.file_store import read_json, write_json
            config_dir = self._project_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            excl_path = config_dir / "feature_exclusions.json"
            existing_data = read_json(excl_path, {}) if excl_path.exists() else {}
            existing_set = set(existing_data.get("excluded_feature_cols", []))
            combined = sorted(existing_set | exclusions)

            write_json(excl_path, {
                "excluded_feature_cols": combined,
                "excluded_bodyparts": sorted(set(
                    existing_data.get("excluded_bodyparts", []) + unchecked_bodyparts
                )),
                "auto_generated": True,
            })

            n_new = len(exclusions - existing_set)
            self._log_panel.append(
                f"Applied {n_new} new exclusion(s) ({len(combined)} total excluded features). "
                f"Excluded bodyparts: {unchecked_bodyparts or 'none'}. "
                f"Saved to config/feature_exclusions.json."
            )
            self._audit_status_label.setText(
                f"Exclusions applied: {len(combined)} features excluded. "
                "These will be automatically skipped in future training runs."
            )
        except Exception as exc:
            self._log_panel.append(f"Failed to apply exclusions: {exc}")
            QMessageBox.warning(self, "Error", f"Could not save exclusions: {exc}")
