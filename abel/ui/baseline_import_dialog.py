"""Detection + mapping dialog for importing another project as a baseline.

Shows whether the host is a new project or already has behaviours/models, lists
each source behaviour with its labeled-example count and trained-model coverage,
and lets the user map each one onto an existing host behaviour, add it as a new
behaviour, or skip it.  The import is gated behind an explicit Accept.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from abel.services.model_refinement_service import (
    AUTO_CREATE_BEHAVIOR,
    SKIP_BEHAVIOR,
    BaselinePreview,
    CoverageDiagnosis,
)


def _format_diagnosis_html(diag: CoverageDiagnosis) -> str:
    """Render a :class:`CoverageDiagnosis` as a self-contained HTML report."""
    def _esc(s: str) -> str:
        return (
            str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    parts: list[str] = []
    parts.append(
        f"<p><b>{diag.models_blocked} of {diag.models_total}</b> trained "
        f"model(s) can’t be imported into this project "
        f"(lowest coverage <b>{diag.worst_coverage:.0%}</b>; models need "
        f"≥98% of their feature columns to exist here).</p>"
    )

    if diag.missing_groups:
        parts.append(
            f"<p><b>Missing feature columns ({diag.missing_total})</b><br>"
            "These columns exist in the source models but not in this project’s "
            "extracted features:</p><ul>"
        )
        for label, count in diag.missing_groups:
            parts.append(f"<li>{_esc(label)} — <b>{count}</b></li>")
        parts.append("</ul>")

    if diag.causes:
        parts.append("<p><b>Likely cause</b></p><ul>")
        for c in diag.causes:
            parts.append(f"<li>{_esc(c)}</li>")
        parts.append("</ul>")

    if diag.fixes:
        parts.append("<p><b>How to fix</b></p><ol>")
        for f in diag.fixes:
            parts.append(f"<li>{_esc(f)}</li>")
        parts.append("</ol>")

    if diag.sample_missing:
        sample = ", ".join(_esc(s) for s in diag.sample_missing)
        parts.append(
            "<p style='color:#90A4AE;font-size:11px;'><b>Example missing "
            f"columns:</b> {sample}…</p>"
        )
    return "".join(parts)


class BaselineDiagnosisDialog(QDialog):
    """Read-only helper explaining why baseline models are blocked + how to fix."""

    def __init__(self, diagnosis: CoverageDiagnosis, tag: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Model coverage — {tag}")
        self.resize(620, 520)
        layout = QVBoxLayout(self)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(False)
        browser.setHtml(_format_diagnosis_html(diagnosis))
        layout.addWidget(browser, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class BaselineImportDialog(QDialog):
    """Confirm + map a baseline import (clips + feature rows + models)."""

    def __init__(
        self,
        preview: BaselinePreview,
        host_behaviors: list[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pv = preview
        self._host_behaviors = list(host_behaviors)
        self._combos: list[tuple[str, QComboBox]] = []  # (source_behavior_id, combo)

        self.setWindowTitle(f"Import Baseline — {preview.tag}")
        self.resize(720, 480)

        layout = QVBoxLayout(self)

        # ── Detection banner ──────────────────────────────────────────
        if preview.host_is_new:
            banner = (
                "<b>New project.</b> This project has extracted features but no "
                "behaviours, models, or training set yet — importing will seed it "
                f"from <b>{preview.tag}</b>."
            )
        else:
            banner = (
                "<b>Existing project.</b> This project already has behaviours/models. "
                f"Imported behaviours from <b>{preview.tag}</b> will be added as new "
                "or mapped onto your existing ones below."
            )
        banner_lbl = QLabel(banner)
        banner_lbl.setWordWrap(True)
        layout.addWidget(banner_lbl)

        summary = QLabel(
            f"Feature schema coverage: {preview.coverage:.0%} · "
            f"{preview.total_examples} labeled example(s) · "
            f"{preview.model_count} trained model(s)."
        )
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #90A4AE; font-size: 11px;")
        layout.addWidget(summary)

        # Schema / diagnostics warnings
        warn_lines: list[str] = []
        if not preview.schema_ok:
            warn_lines.append(
                "⚠ " + (preview.reason or "Incompatible feature schema.")
                + " Import is blocked until the projects share a pose/feature schema."
            )
        d = preview.diagnostics
        if d is not None and d.config_mismatches:
            warn_lines.append(
                "⚠ Feature-extraction settings differ: " + "; ".join(d.config_mismatches)
                + ". Imported features may not be directly comparable."
            )
        if preview.keypoint_renames:
            warn_lines.append(
                f"Keypoint names realigned onto this project's scheme "
                f"({len(preview.keypoint_renames)} renamed)."
            )
        diag = preview.coverage_diagnosis
        if diag is not None and diag.has_blocked_models:
            warn_lines.append(
                f"⚠ {diag.models_blocked} of {diag.models_total} trained model(s) "
                f"can't be imported — this project is missing "
                f"{diag.missing_total} feature column(s) they were trained on. "
                "Examples still import; the models won't be copied."
            )

        if warn_lines:
            warn = QLabel("\n".join(warn_lines))
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #FFB74D; font-size: 11px;")
            layout.addWidget(warn)

        # Offer the coverage-diagnosis helper whenever models are blocked.
        if diag is not None and diag.has_blocked_models:
            diag_row = QHBoxLayout()
            diag_btn = QPushButton("Diagnose models — why & how to fix")
            diag_btn.clicked.connect(
                lambda: BaselineDiagnosisDialog(diag, preview.tag, self).exec()
            )
            diag_row.addWidget(diag_btn)
            diag_row.addStretch(1)
            layout.addLayout(diag_row)

        intro = QLabel(
            "For each behaviour below choose how to apply it: <i>Auto-create</i> "
            "adds it as a new behaviour (carrying the source's definition), "
            "<i>Map to</i> folds it onto an existing behaviour, and <i>skip</i> "
            "imports neither its examples nor its model."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ── Per-behaviour table ───────────────────────────────────────
        self._table = QTableWidget(len(preview.rows), 4)
        self._table.setHorizontalHeaderLabels(
            ["Behaviour (source)", "Examples", "Model", "Apply as"]
        )
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        for r, row in enumerate(preview.rows):
            name_item = QTableWidgetItem(row.source_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 0, name_item)

            ex_item = QTableWidgetItem(str(row.example_count))
            ex_item.setFlags(ex_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 1, ex_item)

            if not row.has_model:
                model_txt = "—"
            elif row.model_compatible:
                model_txt = f"✓ {row.model_coverage:.0%}"
            else:
                model_txt = f"✕ {row.model_coverage:.0%}"
            model_item = QTableWidgetItem(model_txt)
            model_item.setFlags(model_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if row.has_model and not row.model_compatible:
                model_item.setForeground(QColor("#EF5350"))
                model_item.setToolTip(
                    "The model's features aren't fully covered by this project; "
                    "its examples can still be imported but the model won't be copied."
                )
            self._table.setItem(r, 2, model_item)

            combo = QComboBox()
            combo.addItem(f"Auto-create “{row.source_name}”", AUTO_CREATE_BEHAVIOR)
            for host_id, host_name in self._host_behaviors:
                combo.addItem(f"Map to: {host_name}", host_id)
            combo.addItem("— skip (don't import) —", SKIP_BEHAVIOR)
            # Default to the detected match, else auto-create.
            if row.matched_host_id:
                idx = combo.findData(row.matched_host_id)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                combo.setCurrentIndex(0)
            self._table.setCellWidget(r, 3, combo)
            self._combos.append((row.source_behavior_id, combo))

        layout.addWidget(self._table, 1)

        # ── Buttons (Accept gated on schema compatibility) ────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Accept && Import")
        ok_btn.setEnabled(preview.schema_ok and bool(preview.rows))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def decisions(self) -> dict[str, str]:
        """Return ``{source_behavior_id: host_id | AUTO_CREATE | SKIP}``."""
        return {src_bid: str(combo.currentData()) for src_bid, combo in self._combos}
