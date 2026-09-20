"""Methods Write-up Helper subtab: draft a methods section from the open project.

The user ticks which facts they want covered; ABEL reads them out of the project
and renders a plain-text draft they can edit in place, copy, or save.

Two deliberate design points:

* The warning is not a footnote.  It is a red banner above the controls, it is
  prepended to the generated text, and it is repeated when the text is saved to a
  file.  Generated prose reads authoritative whether or not it is right, and a
  methods section is a claim about what was actually done.
* Gathering runs on a worker thread.  Per-behavior metrics come from
  :meth:`ValidationService.model_overview`, which reads every model directory and
  the saved probability traces, a second or more on a real project, and longer
  when validation quizzes are included.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.ui.methods_writeup import (
    DEFAULT_KEYS,
    SECTION_GROUPS,
    SECTIONS,
    WARNING_TEXT,
    gather_facts,
    render_methods_text,
    summarize_validation_runs,
)

logger = logging.getLogger("abel")

_PLACEHOLDER = (
    "Tick the facts you want covered on the left, then click "
    "“Generate Draft”.\n\n"
    "Everything ABEL can read from this project: model metrics, the behavior "
    "list, window sizes, bout thresholds, HMM settings, is filled in from the "
    "project itself. Everything it cannot know is left as a [FILL IN: ...] "
    "placeholder for you to replace.\n\n"
    "The result is a starting point for your own writing, not text to paste into "
    "a manuscript."
)


class _GatherWorker(QThread):
    """Collects project facts off the UI thread."""

    done = Signal(object, str)  # facts dict (or None), error message

    def __init__(self, project_root: Path, include_reliability: bool) -> None:
        super().__init__()
        self._root = Path(project_root)
        self._include_reliability = include_reliability

    def run(self) -> None:
        try:
            from abel.services.validation_service import ValidationService  # noqa: PLC0415

            svc = ValidationService()
            svc.set_project(self._root)
            try:
                model_rows = svc.model_overview()
            except Exception as exc:
                logger.warning("Methods write-up: model overview unavailable (%s)", exc)
                model_rows = []
            reliability = summarize_validation_runs(svc) if self._include_reliability else []
            facts = gather_facts(self._root, model_rows=model_rows, reliability=reliability)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Methods write-up: could not gather project facts")
            self.done.emit(None, str(exc))
            return
        self.done.emit(facts, "")


class MethodsWriteupTab(QWidget):
    """Checkbox-driven draft generator for a manuscript methods section."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_root: Path | None = None
        self._worker: _GatherWorker | None = None
        self._checks: dict[str, QCheckBox] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        root.addWidget(self._build_warning_banner())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_options_panel())
        splitter.addWidget(self._build_output_panel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

    # ── Warning banner ─────────────────────────────────────────────────

    def _build_warning_banner(self) -> QWidget:
        box = QWidget()
        box.setStyleSheet(
            "background: #3B1616; border: 2px solid #EF5350; border-radius: 6px;"
        )
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        title = QLabel("⚠  THIS IS A DRAFT: DO NOT COPY IT INTO A MANUSCRIPT")
        title.setStyleSheet("font-size: 15px; font-weight: 800; color: #FF8A80;")
        title.setWordWrap(True)

        body = QLabel(
            "This tool assembles a suggested write-up from the settings and results "
            "stored in this project. It is a guide and a checklist, not your methods "
            "section. Rewrite every sentence in your own words, verify every number "
            "against your own records, replace every [FILL IN: ...] placeholder, and "
            "delete anything you did not actually do. You are responsible for the "
            "accuracy of what you publish."
        )
        body.setStyleSheet("font-size: 12px; color: #FFCDD2;")
        body.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(body)
        return box

    # ── Options panel ──────────────────────────────────────────────────

    def _build_options_panel(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        header = QLabel("What should the write-up cover?")
        header.setStyleSheet("font-size: 14px; font-weight: 800; color: #90CAF9;")
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # No horizontal scrolling: the per-section hints are word-wrapped labels, so
        # they must reflow to the panel's width rather than set a wide minimum.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #1E3A5F; }")
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(8, 8, 8, 8)
        inner_layout.setSpacing(8)

        for group in SECTION_GROUPS:
            specs = [s for s in SECTIONS if s.group == group]
            if not specs:
                continue
            box = QGroupBox(group)
            box.setStyleSheet(
                "QGroupBox { font-weight: 700; color: #B0BEC5; border: 1px solid #1E3A5F;"
                " border-radius: 4px; margin-top: 8px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            )
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(10, 10, 10, 8)
            box_layout.setSpacing(2)
            for spec in specs:
                cb = QCheckBox(spec.label)
                cb.setChecked(spec.key in DEFAULT_KEYS)
                cb.setToolTip(spec.hint)
                self._checks[spec.key] = cb
                hint = QLabel(spec.hint)
                hint.setWordWrap(True)
                hint.setStyleSheet(
                    "font-size: 11px; color: #78909C; margin-left: 20px;"
                )
                box_layout.addWidget(cb)
                box_layout.addWidget(hint)
                box_layout.addSpacing(4)
            inner_layout.addWidget(box)

        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        select_row = QHBoxLayout()
        all_btn = QPushButton("Select All")
        all_btn.clicked.connect(lambda: self._set_all(True))
        none_btn = QPushButton("Select None")
        none_btn.clicked.connect(lambda: self._set_all(False))
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        for btn in (all_btn, none_btn, reset_btn):
            select_row.addWidget(btn)
        select_row.addStretch(1)
        outer.addLayout(select_row)

        self._warning_check = QCheckBox("Keep the draft warning at the top of the text")
        self._warning_check.setChecked(True)
        self._warning_check.setToolTip(
            "Leaves the warning block in the generated text, so it travels with the "
            "draft if you paste or share it."
        )
        outer.addWidget(self._warning_check)

        self._generate_btn = QPushButton("Generate Draft")
        self._generate_btn.setStyleSheet("font-weight: 700; padding: 6px;")
        self._generate_btn.clicked.connect(self._generate)
        outer.addWidget(self._generate_btn)

        self._status = QLabel("No project loaded.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 11px; color: #78909C;")
        outer.addWidget(self._status)
        return panel

    # ── Output panel ───────────────────────────────────────────────────

    def _build_output_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("Suggested write-up (editable: change it before you use it)")
        header.setStyleSheet("font-size: 14px; font-weight: 800; color: #90CAF9;")
        layout.addWidget(header)

        self._output = QTextEdit()
        self._output.setAcceptRichText(False)
        self._output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        # Monospace: the performance and threshold tables are column-aligned text.
        self._output.setFont(QFont("Consolas", 10))
        self._output.setStyleSheet(
            "QTextEdit { background: #0A1929; color: #CFD8DC;"
            " border: 1px solid #1E3A5F; }"
        )
        self._output.setPlaceholderText(_PLACEHOLDER)
        layout.addWidget(self._output, 1)

        btn_row = QHBoxLayout()
        self._copy_btn = QPushButton("Copy to Clipboard")
        self._copy_btn.clicked.connect(self._copy)
        self._save_btn = QPushButton("Save as .txt…")
        self._save_btn.clicked.connect(self._save)
        for btn in (self._copy_btn, self._save_btn):
            btn.setEnabled(False)
            btn_row.addWidget(btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        return panel

    # ── Project wiring ─────────────────────────────────────────────────

    def set_project(self, project_root: Path) -> None:
        self._project_root = Path(project_root)
        self._output.clear()
        self._copy_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._status.setText(
            "Ready. The draft is built from this project's current settings and its "
            "most recently trained models."
        )

    # ── Selection helpers ──────────────────────────────────────────────

    def _set_all(self, checked: bool) -> None:
        for cb in self._checks.values():
            cb.setChecked(checked)

    def _reset_defaults(self) -> None:
        for key, cb in self._checks.items():
            cb.setChecked(key in DEFAULT_KEYS)

    def _selected_keys(self) -> list[str]:
        return [key for key, cb in self._checks.items() if cb.isChecked()]

    # ── Generate ───────────────────────────────────────────────────────

    def _generate(self) -> None:
        if self._project_root is None:
            QMessageBox.information(
                self, "No Project", "Open a project before drafting a methods section."
            )
            return
        if self._worker is not None:
            return
        keys = self._selected_keys()
        if not keys:
            QMessageBox.information(
                self, "Nothing Selected",
                "Tick at least one section to include in the write-up.",
            )
            return

        self._generate_btn.setEnabled(False)
        self._status.setText("Reading project settings and model metrics…")
        self._worker = _GatherWorker(self._project_root, "reliability" in keys)
        self._worker.done.connect(self._on_gathered)
        self._worker.start()

    def _on_gathered(self, facts: dict[str, Any] | None, error: str) -> None:
        if self._worker is not None:
            self._worker.wait()
            self._worker.deleteLater()
            self._worker = None
        self._generate_btn.setEnabled(True)

        if facts is None:
            self._status.setText(f"Could not read the project: {error}")
            QMessageBox.critical(
                self, "Write-up Failed",
                f"ABEL could not read this project's settings:\n{error}",
            )
            return

        text = render_methods_text(
            facts, self._selected_keys(), include_warning=self._warning_check.isChecked()
        )
        self._output.setPlainText(text)
        self._copy_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        n_models = len([
            r for r in facts.get("model_rows", [])
            if str(r.get("model_version") or "-") != "-"
        ])
        self._status.setText(
            f"Draft built from {n_models} trained model(s). Edit it here, then copy or "
            f"save. Verify every number before you use it."
        )

    # ── Output actions ─────────────────────────────────────────────────

    def _copy(self) -> None:
        text = self._output.toPlainText()
        if not text.strip():
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self._status.setText(
            "Copied. Remember: this is a draft to edit, not text to paste as-is."
        )

    def _save(self) -> None:
        text = self._output.toPlainText()
        if not text.strip():
            return
        default_name = "methods_draft.txt"
        if self._project_root is not None:
            default_name = str(self._project_root / "exports" / "methods_draft.txt")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Methods Draft", default_name, "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        # The warning travels with the file even when it was toggled off in the
        # editor: a .txt on a shared drive outlives the context it was made in.
        payload = text if WARNING_TEXT in text else WARNING_TEXT + "\n\n" + text
        try:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload, encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not write the file:\n{exc}")
            return
        self._status.setText(f"Saved to {path}")
