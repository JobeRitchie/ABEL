"""Validation tab: model overview, interactive reviewer quiz, and suggestions.

Three subtabs share one :class:`ValidationService` and one assembled
:class:`ValidationRun`:

* **Overview** — per-behavior model quality, label counts, and bout counts.
* **Validation Quiz** — a blind labeling test with hotkey assignment, looping,
  auto-advance, and an "Unsure" option.  Answers are stored per named reviewer.
* **Results & Suggestions** — user-vs-machine + inter-rater metrics and
  rule-based guidance, with an opt-in write-back into training labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from abel.models.schemas import ValidationAnswerRecord, ValidationRun, ValidationSettings
from abel.services.behavior_service import BehaviorService
from abel.services.validation_service import NO_BEHAVIOR_ID, ValidationService
from abel.ui.widgets.clip_player import ClipPlayer
from abel.workers.task_worker import TaskWorker

logger = logging.getLogger("abel")

_QUALITY_COLORS = {
    "good": "#2E7D32",
    "fair": "#F9A825",
    "poor": "#C62828",
    "unknown": "#546E7A",
}


def _fmt(value: object, pct: bool = False) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v:.0%}" if pct else f"{v:.3f}"


# ===========================================================================
# Overview panel
# ===========================================================================
class ValidationOverviewPanel(QWidget):
    """Dashboard of every behavior model's quality and data coverage."""

    _COLUMNS = [
        "Behavior", "Model", "Quality", "F1", "Precision", "Recall",
        "PR-AUC", "Train", "Val", "Pos labels", "Neg labels", "Bouts", "Overlap",
    ]

    def __init__(self, service: ValidationService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = service

        header = QLabel("Model Overview")
        header.setStyleSheet("font-size: 16px; font-weight: 700; color: #ECEFF1;")
        subtitle = QLabel(
            "Quality metrics, label coverage, detected bouts, and behavior overlap for each model."
        )
        subtitle.setStyleSheet("color: #90A4AE; font-size: 12px;")

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)

        top = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_box.addWidget(header)
        title_box.addWidget(subtitle)
        top.addLayout(title_box)
        top.addStretch()
        top.addWidget(self._refresh_btn)

        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        overlap_header = self._table.horizontalHeaderItem(self._COLUMNS.index("Overlap"))
        if overlap_header is not None:
            overlap_header.setToolTip(
                "Fraction of this behavior's flagged frames where another behavior is also flagged. "
                "High overlap suggests thresholds are too lax or behavior inhibition is too weak."
            )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

        self._empty = QLabel("Open a project with trained models to see the overview.")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet("color: #546E7A; font-size: 13px; padding: 20px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addLayout(top)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._empty)
        self._empty.hide()

    def refresh(self) -> None:
        try:
            rows = self._service.model_overview()
        except Exception:
            logger.exception("Validation overview refresh failed")
            rows = []
        self._table.setRowCount(0)
        if not rows:
            self._table.hide()
            self._empty.show()
            return
        self._empty.hide()
        self._table.show()
        for data in rows:
            r = self._table.rowCount()
            self._table.insertRow(r)
            quality = data.get("quality", "unknown")
            overlap = data.get("overlap_fraction")
            overlap_text = "—" if overlap is None else f"{overlap:.0%}"
            cells = [
                data.get("behavior_name", "?"),
                data.get("model_version", "—"),
                quality.capitalize(),
                _fmt(data.get("frame_f1")),
                _fmt(data.get("frame_precision")),
                _fmt(data.get("frame_recall")),
                _fmt(data.get("pr_auc")),
                str(data.get("n_train") if data.get("n_train") is not None else "—"),
                str(data.get("n_val") if data.get("n_val") is not None else "—"),
                str(data.get("n_positive_labels", 0)),
                str(data.get("n_negative_labels", 0)),
                str(data.get("n_bouts", 0)),
                overlap_text,
            ]
            overlap_col = len(cells) - 1
            for c, val in enumerate(cells):
                item = QTableWidgetItem(str(val))
                if c == 2:  # quality badge
                    item.setForeground(QColor("#FFFFFF"))
                    item.setBackground(QColor(_QUALITY_COLORS.get(quality, "#546E7A")))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                elif c == overlap_col and overlap is not None:
                    # Amber/red as overlap rises — high overlap means weak inhibition.
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if overlap >= 0.15:
                        item.setForeground(QColor("#EF9A9A"))
                    elif overlap >= 0.05:
                        item.setForeground(QColor("#FFCC80"))
                self._table.setItem(r, c, item)


# ===========================================================================
# Quiz panel
# ===========================================================================
class ValidationQuizPanel(QWidget):
    """Blind labeling quiz with hotkeys, looping, auto-advance, and Unsure."""

    run_changed = Signal()
    answers_changed = Signal()

    def __init__(
        self,
        service: ValidationService,
        behavior_service: BehaviorService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._behaviors = behavior_service
        self._pool = QThreadPool.globalInstance()
        self._project_root: Path | None = None
        self._run: ValidationRun | None = None
        self._idx = -1
        self._answers: dict[str, ValidationAnswerRecord] = {}
        self._behavior_shortcuts: list[QShortcut] = []
        self._busy = False

        self._build_ui()
        self._build_settings_popup()
        self._install_navigation_shortcuts()

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        # Top bar
        self._reviewer_input = QLineEdit("reviewer")
        self._reviewer_input.setMaximumWidth(180)
        self._reviewer_input.setPlaceholderText("Reviewer name")
        self._reviewer_input.editingFinished.connect(self._on_reviewer_changed)

        self._settings_btn = QPushButton("Settings")
        self._settings_btn.clicked.connect(self._toggle_settings)

        self._generate_btn = QPushButton("Generate Test")
        self._generate_btn.setToolTip(
            "Assemble a new randomized test using the current settings. Adjust Settings first."
        )
        self._generate_btn.clicked.connect(self._generate)

        self._resume_btn = QPushButton("Resume Saved Test")
        self._resume_btn.setToolTip("Reload your most recent saved test without generating a new one.")
        self._resume_btn.clicked.connect(self._resume)

        self._progress_label = QLabel("No test loaded")
        self._progress_label.setStyleSheet("font-weight: 600; color: #CFD8DC;")

        top = QHBoxLayout()
        top.addWidget(QLabel("Reviewer:"))
        top.addWidget(self._reviewer_input)
        top.addSpacing(12)
        top.addWidget(self._settings_btn)
        top.addWidget(self._generate_btn)
        top.addWidget(self._resume_btn)
        top.addStretch()
        top.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)

        # Player
        self._player = ClipPlayer()

        # Decision controls
        self._labels_row = QHBoxLayout()
        self._labels_row.setSpacing(8)
        self._label_buttons_container = QWidget()
        self._label_buttons_container.setLayout(self._labels_row)

        self._prev_btn = QPushButton("◀ Previous")
        self._prev_btn.clicked.connect(self._prev)
        self._next_btn = QPushButton("Next ▶")
        self._next_btn.clicked.connect(self._next)
        self._autoadvance_chk = QCheckBox("Auto-advance")
        self._autoadvance_chk.setChecked(True)
        self._autoadvance_chk.setToolTip("Automatically move to the next clip after labeling.")

        self._current_answer_label = QLabel("")
        self._current_answer_label.setStyleSheet("font-size: 12px; color: #80CBC4; font-weight: 600;")

        nav = QHBoxLayout()
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._next_btn)
        nav.addWidget(self._autoadvance_chk)
        nav.addSpacing(12)
        nav.addWidget(self._current_answer_label)
        nav.addStretch()

        hint = QLabel(
            "Watch the clip and assign a behavior with the on-screen buttons or hotkeys. "
            "Use No Behavior when no target behavior is present, or Unsure if you cannot tell."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #90A4AE; font-size: 11px;")

        self._empty = QLabel(
            "Adjust Settings, then click “Generate Test” to begin. Clips are drawn blind from "
            "prior-accepted, model-positive, no-behavior, and borderline categories."
        )
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet("color: #546E7A; font-size: 13px; padding: 30px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addLayout(top)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._player, 1)
        layout.addWidget(self._label_buttons_container)
        layout.addLayout(nav)
        layout.addWidget(hint)
        layout.addWidget(self._empty)

        self._set_quiz_widgets_visible(False)

    def _build_settings_popup(self) -> None:
        self._settings_panel = QFrame(self, Qt.WindowType.Popup)
        self._settings_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self._settings_panel.setStyleSheet(
            "QFrame { background: #263238; border: 1px solid #546E7A; border-radius: 6px; }"
        )
        v = QVBoxLayout(self._settings_panel)
        v.setContentsMargins(12, 10, 12, 12)
        v.setSpacing(8)

        title = QLabel("Test Settings")
        title.setStyleSheet("font-weight: 700; font-size: 13px; color: #ECEFF1;")
        v.addWidget(title)

        self._spin_total = QSpinBox()
        self._spin_total.setRange(4, 1000)
        self._spin_total.setSingleStep(5)

        self._spin_prior = QDoubleSpinBox()
        self._spin_unrev = QDoubleSpinBox()
        self._spin_neg = QDoubleSpinBox()
        self._spin_fringe = QDoubleSpinBox()
        for sp in (self._spin_prior, self._spin_unrev, self._spin_neg, self._spin_fringe):
            sp.setRange(0.0, 1.0)
            sp.setSingleStep(0.05)
            sp.setDecimals(2)

        self._spin_clip_sec = QDoubleSpinBox()
        self._spin_clip_sec.setRange(0.5, 10.0)
        self._spin_clip_sec.setSingleStep(0.5)
        self._spin_clip_sec.setDecimals(1)

        self._spin_fringe_hw = QDoubleSpinBox()
        self._spin_fringe_hw.setRange(0.01, 0.4)
        self._spin_fringe_hw.setSingleStep(0.01)
        self._spin_fringe_hw.setDecimals(2)

        self._chk_balance = QCheckBox("Balance clips across behaviors")
        self._chk_loop = QCheckBox("Loop clips by default")
        self._chk_autoadv = QCheckBox("Auto-advance by default")

        def _row(label: str, w: QWidget) -> None:
            row = QHBoxLayout()
            lab = QLabel(label)
            lab.setStyleSheet("color: #CFD8DC; font-size: 12px;")
            lab.setMinimumWidth(200)
            row.addWidget(lab)
            row.addWidget(w)
            row.addStretch()
            v.addLayout(row)

        _row("Total clips", self._spin_total)
        _row("Proportion: prior-accepted", self._spin_prior)
        _row("Proportion: unreviewed positive", self._spin_unrev)
        _row("Proportion: negative", self._spin_neg)
        _row("Proportion: fringe (borderline)", self._spin_fringe)
        _row("Clip length (seconds)", self._spin_clip_sec)
        _row("Fringe half-width", self._spin_fringe_hw)
        v.addWidget(self._chk_balance)
        v.addWidget(self._chk_loop)
        v.addWidget(self._chk_autoadv)

        note = QLabel("Proportions are relative weights; they are normalized when building.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #78909C; font-size: 11px;")
        v.addWidget(note)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_settings_from_popup)
        btn_row.addWidget(save_btn)
        v.addLayout(btn_row)
        self._settings_panel.adjustSize()
        self._settings_panel.hide()

    # ---------------------------------------------------------------- project
    def set_project(self, project_root: Path) -> None:
        self._project_root = Path(project_root)
        self._behaviors.set_project(project_root)
        self._rebuild_label_buttons()
        self._load_settings_into_popup()
        # Resume an in-progress run if one exists, but never auto-generate one —
        # the user decides when to build a test via "Generate Test".
        self.reload()

    def reload(self) -> None:
        """Reload the active saved run if present; otherwise show the empty state."""
        run = self._service.load_active_run()
        if run is not None and run.clips:
            self._set_run(run)
        else:
            self._run = None
            self._idx = -1
            self._player.close_clip()
            self._set_quiz_widgets_visible(False)
            self._progress_label.setText("No test loaded")

    # ---------------------------------------------------------------- settings popup
    def _toggle_settings(self) -> None:
        if self._settings_panel.isVisible():
            self._settings_panel.hide()
            return
        self._load_settings_into_popup()
        pos = self._settings_btn.mapToGlobal(self._settings_btn.rect().bottomLeft())
        self._settings_panel.move(pos)
        self._settings_panel.show()

    def _load_settings_into_popup(self) -> None:
        s = self._service.load_settings()
        self._spin_total.setValue(s.n_total_clips)
        self._spin_prior.setValue(s.prop_prior_accepted)
        self._spin_unrev.setValue(s.prop_unreviewed_positive)
        self._spin_neg.setValue(s.prop_negative)
        self._spin_fringe.setValue(s.prop_fringe)
        self._spin_clip_sec.setValue(s.clip_seconds)
        self._spin_fringe_hw.setValue(s.fringe_half_width)
        self._chk_balance.setChecked(s.balance_across_behaviors)
        self._chk_loop.setChecked(s.loop_default)
        self._chk_autoadv.setChecked(s.autoadvance_default)

    def _settings_from_popup(self) -> ValidationSettings:
        s = self._service.load_settings()
        s.n_total_clips = self._spin_total.value()
        s.prop_prior_accepted = self._spin_prior.value()
        s.prop_unreviewed_positive = self._spin_unrev.value()
        s.prop_negative = self._spin_neg.value()
        s.prop_fringe = self._spin_fringe.value()
        s.clip_seconds = self._spin_clip_sec.value()
        s.fringe_half_width = self._spin_fringe_hw.value()
        s.balance_across_behaviors = self._chk_balance.isChecked()
        s.loop_default = self._chk_loop.isChecked()
        s.autoadvance_default = self._chk_autoadv.isChecked()
        return s

    def _save_settings_from_popup(self) -> None:
        self._service.save_settings(self._settings_from_popup())
        self._autoadvance_chk.setChecked(self._chk_autoadv.isChecked())
        self._player.set_loop(self._chk_loop.isChecked())
        self._settings_panel.hide()

    # ---------------------------------------------------------------- build
    def _resume(self) -> None:
        run = self._service.load_active_run()
        if run is not None and run.clips:
            self._set_run(run)
        else:
            QMessageBox.information(
                self,
                "Validation",
                "No saved test to resume. Use “Generate Test” to build one "
                "with your current settings.",
            )

    def _generate(self) -> None:
        if self._busy:
            return
        # Only confirm when an existing test would be superseded — a first test
        # generates immediately so nothing is wasted.
        existing = self._service.load_active_run()
        if existing is not None and existing.clips:
            resp = QMessageBox.question(
                self,
                "Generate Test",
                "Build a new randomized test with the current settings? Your existing "
                "test stays saved and can be reopened from the Results tab.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
        self._assemble()

    def _assemble(self) -> None:
        if self._busy:
            return
        settings = self._service.load_settings()
        self._busy = True
        self._set_busy_ui(True)

        worker = TaskWorker(self._service.assemble_run, settings)
        worker.signals.finished.connect(self._on_assembled)
        worker.signals.failed.connect(self._on_assemble_failed)
        self._pool.start(worker)

    def _set_busy_ui(self, busy: bool) -> None:
        self._progress_bar.setVisible(busy)
        self._progress_bar.setRange(0, 0 if busy else 100)
        self._generate_btn.setEnabled(not busy)
        self._resume_btn.setEnabled(not busy)
        if busy:
            self._progress_label.setText("Building test — extracting clips…")

    def _on_assembled(self, run: ValidationRun) -> None:
        self._busy = False
        self._set_busy_ui(False)
        if not run.clips:
            QMessageBox.information(
                self,
                "Validation",
                "No clips could be assembled. Make sure temporal inference has been run and "
                "source videos are available.",
            )
            self._progress_label.setText("No test loaded")
            return
        self._set_run(run)
        self.run_changed.emit()

    def _on_assemble_failed(self, tb: str) -> None:
        self._busy = False
        self._set_busy_ui(False)
        logger.error("Validation test assembly failed:\n%s", tb)
        QMessageBox.critical(self, "Validation", f"Failed to build test:\n\n{tb.strip().splitlines()[-1]}")
        self._progress_label.setText("No test loaded")

    # ---------------------------------------------------------------- run / navigation
    def _set_run(self, run: ValidationRun) -> None:
        self._run = run
        self._answers = self._service.load_answers(run.run_id, self._reviewer())
        self._set_quiz_widgets_visible(True)
        self._player.set_loop(self._service.load_settings().loop_default)
        # Resume at first unanswered clip.
        self._idx = self._first_unanswered()
        self._load_current()

    def _first_unanswered(self) -> int:
        if not self._run:
            return -1
        for i, clip in enumerate(self._run.clips):
            if clip.clip_id not in self._answers:
                return i
        return 0

    def _set_quiz_widgets_visible(self, visible: bool) -> None:
        self._empty.setVisible(not visible)
        for w in (
            self._player, self._label_buttons_container, self._prev_btn,
            self._next_btn, self._autoadvance_chk, self._current_answer_label,
        ):
            w.setVisible(visible)

    def _reviewer(self) -> str:
        return self._reviewer_input.text().strip() or "reviewer"

    def _on_reviewer_changed(self) -> None:
        if self._run is None:
            return
        self._answers = self._service.load_answers(self._run.run_id, self._reviewer())
        self._update_progress()
        self._update_current_answer_label()

    def _load_current(self) -> None:
        if not self._run or not self._run.clips:
            return
        self._idx = max(0, min(self._idx, len(self._run.clips) - 1))
        clip = self._run.clips[self._idx]
        if clip.clip_path and Path(clip.clip_path).exists():
            self._player.load_clip(clip.clip_path, autoplay=True)
        else:
            self._player.close_clip()
        self._update_progress()
        self._update_current_answer_label()

    def _update_progress(self) -> None:
        if not self._run:
            self._progress_label.setText("No test loaded")
            return
        n = len(self._run.clips)
        answered = sum(1 for c in self._run.clips if c.clip_id in self._answers)
        self._progress_label.setText(
            f"Clip {self._idx + 1} / {n}   •   {answered}/{n} labeled   •   reviewer: {self._reviewer()}"
        )

    def _update_current_answer_label(self) -> None:
        if not self._run:
            self._current_answer_label.setText("")
            return
        clip = self._run.clips[self._idx]
        ans = self._answers.get(clip.clip_id)
        if ans is None:
            self._current_answer_label.setText("Not labeled")
            return
        if ans.is_unsure:
            self._current_answer_label.setText("Your label: Unsure")
        elif ans.label == NO_BEHAVIOR_ID:
            self._current_answer_label.setText("Your label: No Behavior")
        else:
            self._current_answer_label.setText(f"Your label: {self._behavior_name(ans.label)}")

    def _prev(self) -> None:
        if not self._run:
            return
        self._idx = max(0, self._idx - 1)
        self._load_current()

    def _next(self) -> None:
        if not self._run:
            return
        if self._idx >= len(self._run.clips) - 1:
            self._update_progress()
            return
        self._idx += 1
        self._load_current()

    # ---------------------------------------------------------------- labeling
    def _rebuild_label_buttons(self) -> None:
        while self._labels_row.count():
            item = self._labels_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for b in self._behaviors.behaviors:
            bid = str(b.behavior_id or "").strip()
            if not bid or bid == NO_BEHAVIOR_ID:
                continue
            key = str(b.keyboard_shortcut or "").strip()
            label = b.name + (f"  [{key}]" if key else "")
            btn = QPushButton(label)
            color = b.color or "#4A90E2"
            btn.setStyleSheet(
                f"QPushButton {{ background: {color}; color: white; font-weight: 600; "
                f"padding: 8px 14px; border-radius: 4px; }}"
                f"QPushButton:hover {{ border: 2px solid #ECEFF1; }}"
            )
            btn.clicked.connect(lambda _=False, x=bid: self._answer_behavior(x))
            self._labels_row.addWidget(btn)

        neg_btn = QPushButton("No Behavior  [n]")
        neg_btn.setStyleSheet(
            "QPushButton { background: #455A64; color: white; font-weight: 600; "
            "padding: 8px 14px; border-radius: 4px; }"
            "QPushButton:hover { border: 2px solid #ECEFF1; }"
        )
        neg_btn.clicked.connect(lambda: self._answer_behavior(NO_BEHAVIOR_ID))
        self._labels_row.addWidget(neg_btn)

        unsure_btn = QPushButton("Unsure  [u]")
        unsure_btn.setStyleSheet(
            "QPushButton { background: #6A1B9A; color: white; font-weight: 600; "
            "padding: 8px 14px; border-radius: 4px; }"
            "QPushButton:hover { border: 2px solid #ECEFF1; }"
        )
        unsure_btn.clicked.connect(self._answer_unsure)
        self._labels_row.addWidget(unsure_btn)
        self._labels_row.addStretch()

        self._register_behavior_shortcuts()

    def _register_behavior_shortcuts(self) -> None:
        for sc in self._behavior_shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self._behavior_shortcuts.clear()
        used: set[str] = set()
        for b in self._behaviors.behaviors:
            bid = str(b.behavior_id or "").strip()
            key = str(b.keyboard_shortcut or "").strip()
            if not bid or bid == NO_BEHAVIOR_ID or not key:
                continue
            if key.lower() in used:
                continue
            used.add(key.lower())
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda x=bid: self._answer_behavior(x))
            self._behavior_shortcuts.append(sc)
        # Fixed keys for Negative and Unsure (only if not already claimed).
        for key, handler in (("n", lambda: self._answer_behavior(NO_BEHAVIOR_ID)), ("u", self._answer_unsure)):
            if key in used:
                continue
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(handler)
            self._behavior_shortcuts.append(sc)

    def _install_navigation_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self._player.toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self._prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self._next)

    def _answer_behavior(self, behavior_id: str) -> None:
        self._record_answer(label=behavior_id, is_unsure=False)

    def _answer_unsure(self) -> None:
        self._record_answer(label=NO_BEHAVIOR_ID, is_unsure=True)

    def _record_answer(self, label: str, is_unsure: bool) -> None:
        if not self._run or self._idx < 0:
            return
        clip = self._run.clips[self._idx]
        answer = ValidationAnswerRecord(
            clip_id=clip.clip_id,
            reviewer_id=self._reviewer(),
            label=label,
            is_unsure=is_unsure,
        )
        self._service.save_answer(self._run.run_id, answer)
        self._answers[clip.clip_id] = answer
        self._update_current_answer_label()
        self.answers_changed.emit()
        if self._autoadvance_chk.isChecked():
            self._next()
        else:
            self._update_progress()

    def _behavior_name(self, bid: str) -> str:
        for b in self._behaviors.behaviors:
            if str(b.behavior_id) == bid:
                return b.name
        return bid


# ===========================================================================
# Results panel
# ===========================================================================
class ValidationResultsPanel(QWidget):
    """User-vs-machine and inter-rater metrics, plus model-improvement suggestions."""

    run_deleted = Signal()

    def __init__(self, service: ValidationService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = service

        header = QLabel("Results & Suggestions")
        header.setStyleSheet("font-size: 16px; font-weight: 700; color: #ECEFF1;")

        self._run_combo = QComboBox()
        self._run_combo.setMinimumWidth(220)
        self._run_combo.currentIndexChanged.connect(lambda _i: self.refresh())

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)

        self._export_btn = QPushButton("Export to Excel…")
        self._export_btn.setToolTip("Export all validation tests to an Excel workbook (one sheet per test).")
        self._export_btn.clicked.connect(self._export_excel)

        self._delete_btn = QPushButton("Delete Test")
        self._delete_btn.setToolTip("Permanently delete the selected test and all of its reviewer answers.")
        self._delete_btn.setStyleSheet("QPushButton { color: #EF9A9A; }")
        self._delete_btn.clicked.connect(self._delete)

        self._commit_reviewer = QComboBox()
        self._commit_reviewer.setMinimumWidth(140)
        self._commit_btn = QPushButton("Commit labels to training")
        self._commit_btn.setToolTip(
            "Write this reviewer's (non-unsure) answers into reviewer_labels for future training."
        )
        self._commit_btn.clicked.connect(self._commit)

        self._intra_reviewer = QComboBox()
        self._intra_reviewer.setMinimumWidth(140)
        self._intra_reviewer.setToolTip(
            "Reviewer to evaluate for intra-rater reliability (their answers vs their own "
            "previously-accepted labels)."
        )
        self._intra_reviewer.currentIndexChanged.connect(lambda _i: self._rerender())

        top = QHBoxLayout()
        top.addWidget(header)
        top.addStretch()
        top.addWidget(QLabel("Test:"))
        top.addWidget(self._run_combo)
        top.addWidget(self._refresh_btn)
        top.addWidget(self._export_btn)
        top.addWidget(self._delete_btn)

        sel_row = QHBoxLayout()
        sel_row.addStretch()
        sel_row.addWidget(QLabel("Intra-rater reviewer:"))
        sel_row.addWidget(self._intra_reviewer)

        commit_row = QHBoxLayout()
        commit_row.addStretch()
        commit_row.addWidget(QLabel("Write-back reviewer:"))
        commit_row.addWidget(self._commit_reviewer)
        commit_row.addWidget(self._commit_btn)

        self._view = QTextBrowser()
        self._view.setOpenExternalLinks(False)
        self._view.setOpenLinks(False)  # we handle action links ourselves
        self._view.anchorClicked.connect(self._on_anchor)
        self._view.setStyleSheet("QTextBrowser { background: #1c2530; border: none; }")

        # Cache of the last computed metrics so the reviewer selector can re-render
        # without recomputing.
        self._last_run: ValidationRun | None = None
        self._last_metrics: dict | None = None
        self._last_suggestions: list[dict] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addLayout(top)
        layout.addLayout(sel_row)
        layout.addWidget(self._view, 1)
        layout.addLayout(commit_row)

    def refresh(self) -> None:
        self._reload_run_list()
        run = self._selected_run()
        self._delete_btn.setEnabled(run is not None)
        if run is None:
            self._last_run = self._last_metrics = self._last_suggestions = None
            self._view.setHtml(self._wrap("<p style='color:#90A4AE'>No test results yet. Generate a test and label some clips in the Validation Quiz subtab.</p>"))
            self._commit_reviewer.clear()
            self._intra_reviewer.clear()
            return
        answers = self._service.load_all_answers(run.run_id)
        self._commit_reviewer.blockSignals(True)
        self._commit_reviewer.clear()
        for rid in answers:
            self._commit_reviewer.addItem(rid)
        self._commit_reviewer.blockSignals(False)

        prev_intra = self._intra_reviewer.currentText()
        self._intra_reviewer.blockSignals(True)
        self._intra_reviewer.clear()
        for rid in answers:
            self._intra_reviewer.addItem(rid)
        idx = self._intra_reviewer.findText(prev_intra)
        if idx >= 0:
            self._intra_reviewer.setCurrentIndex(idx)
        self._intra_reviewer.blockSignals(False)

        try:
            metrics = self._service.compute_metrics(run, answers)
            suggestions = self._service.suggestions(metrics)
        except Exception:
            logger.exception("Validation metrics failed")
            self._view.setHtml(self._wrap("<p style='color:#C62828'>Failed to compute metrics.</p>"))
            return
        self._last_run, self._last_metrics, self._last_suggestions = run, metrics, suggestions
        self._view.setHtml(self._render(run, metrics, suggestions))

    def _rerender(self) -> None:
        """Re-render from cached metrics (e.g. intra-rater reviewer changed)."""
        if self._last_run and self._last_metrics is not None:
            self._view.setHtml(self._render(self._last_run, self._last_metrics, self._last_suggestions or []))

    def _reload_run_list(self) -> None:
        current = self._run_combo.currentData()
        runs = self._service.list_runs()
        runs.sort(key=lambda r: r.created_at, reverse=True)
        active = self._service.load_active_run()
        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        for r in runs:
            stamp = r.created_at.strftime("%Y-%m-%d %H:%M")
            tag = " (active)" if active and active.run_id == r.run_id else ""
            self._run_combo.addItem(f"{stamp} — {len(r.clips)} clips{tag}", userData=r.run_id)
        if current is not None:
            idx = self._run_combo.findData(current)
            if idx >= 0:
                self._run_combo.setCurrentIndex(idx)
        elif active is not None:
            idx = self._run_combo.findData(active.run_id)
            if idx >= 0:
                self._run_combo.setCurrentIndex(idx)
        self._run_combo.blockSignals(False)

    def _selected_run(self) -> ValidationRun | None:
        run_id = self._run_combo.currentData()
        if not run_id:
            return None
        return self._service.load_run(str(run_id))

    def _delete(self) -> None:
        run_id = self._run_combo.currentData()
        run = self._selected_run()
        if not run_id or run is None:
            return
        stamp = run.created_at.strftime("%Y-%m-%d %H:%M")
        n_rev = len(self._service.list_reviewers(str(run_id)))
        resp = QMessageBox.question(
            self,
            "Delete Test",
            f"Permanently delete the test from {stamp} ({len(run.clips)} clips, "
            f"{n_rev} reviewer(s))?\n\nAll answers for this test will be removed. "
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        self._service.delete_run(str(run_id))
        self.refresh()
        # The quiz panel may have had this run open — let it reload/clear.
        self.run_deleted.emit()

    def _export_excel(self) -> None:
        if not self._service.list_runs():
            QMessageBox.information(self, "Export", "No validation tests to export yet.")
            return
        from datetime import datetime  # noqa: PLC0415

        default = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Validation Results", default, "Excel Workbook (*.xlsx)"
        )
        if not path:
            return
        try:
            n = self._service.export_results_xlsx(Path(path))
        except Exception as exc:
            logger.exception("Validation Excel export failed")
            QMessageBox.critical(self, "Export Failed", f"Could not export results:\n\n{exc}")
            return
        QMessageBox.information(
            self, "Export Complete", f"Exported {n} test(s) to:\n{path}"
        )

    def _on_anchor(self, url) -> None:
        """Handle in-document action links (e.g. apply behavior inhibition)."""
        if url.scheme() != "applyinhibit":
            return
        # url.path() is percent-decoded (toString() would encode the '|' as %7C).
        payload = url.path()
        if "|" not in payload:
            return
        a, b = payload.split("|", 1)
        na, nb = self._service._behavior_name(a), self._service._behavior_name(b)
        resp = QMessageBox.question(
            self,
            "Apply Behavior Inhibition",
            f"Add mutual inhibition between {na} and {nb}?\n\n"
            "This writes a suppression weight into the Temporal Refinement settings. "
            "Re-run Temporal Refinement (dense inhibition) for it to take effect.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        weight = self._service.apply_inhibition(a, b)
        QMessageBox.information(
            self,
            "Inhibition Applied",
            f"Set mutual inhibition between {na} and {nb} to {weight:.2f}.\n\n"
            "Open the Temporal tab → Refinement and re-run dense inhibition to apply it.",
        )

    def _commit(self) -> None:
        run = self._selected_run()
        rid = self._commit_reviewer.currentText().strip()
        if run is None or not rid:
            return
        resp = QMessageBox.question(
            self,
            "Commit Labels",
            f"Write {rid}'s non-unsure answers into training labels (reviewer_labels)?\n\n"
            "This will influence future model training.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        n = self._service.commit_answers_to_training(run.run_id, rid)
        QMessageBox.information(self, "Commit Labels", f"Committed {n} label(s) to training.")

    # ---------------------------------------------------------------- rendering
    @staticmethod
    def _wrap(body: str) -> str:
        return (
            "<div style='font-family: Segoe UI, sans-serif; color:#CFD8DC; font-size:13px;'>"
            + body
            + "</div>"
        )

    @staticmethod
    def _bar(value: float | None, color: str) -> str:
        """Render a horizontal metric bar (0–1) as an HTML cell."""
        if value is None:
            return "<span style='color:#607D8B;'>—</span>"
        pct = max(0.0, min(1.0, float(value))) * 100.0
        return (
            "<table cellpadding='0' cellspacing='0' style='width:130px; border-collapse:collapse;'><tr>"
            f"<td style='width:90px; background:#2b3947;'>"
            f"<div style='background:{color}; height:13px; width:{pct:.0f}%;'></div></td>"
            f"<td style='padding-left:6px; color:#CFD8DC;'>{value:.0%}</td></tr></table>"
        )

    @staticmethod
    def _metric_color(value: float | None) -> str:
        if value is None:
            return "#546E7A"
        if value >= 0.8:
            return "#2E7D32"
        if value >= 0.6:
            return "#F9A825"
        return "#C62828"

    def _render(self, run: ValidationRun, metrics: dict, suggestions: list[dict]) -> str:
        parts: list[str] = []
        n_rev = len(metrics["per_reviewer"])
        parts.append("<h2 style='color:#ECEFF1; margin-bottom:2px;'>Test summary</h2>")
        parts.append(
            f"<p style='color:#90A4AE;'>{metrics['n_clips']} clips • {n_rev} reviewer(s) • "
            f"{len(metrics['behaviors'])} behavior(s)</p>"
        )

        # ── Per-reviewer user-vs-machine (with metric bars) ──────────────
        parts.append("<h3 style='color:#80CBC4;'>User vs. machine</h3>")
        if not metrics["per_reviewer"]:
            parts.append("<p style='color:#90A4AE;'>No answers recorded yet.</p>")
        for rid, rdata in metrics["per_reviewer"].items():
            agree = rdata.get("agreement")
            parts.append(
                f"<p style='margin-bottom:2px;'><b style='color:#ECEFF1;'>{rid}</b> — "
                f"agreement with model: <b style='color:{self._metric_color(agree)};'>"
                f"{_fmt(agree, pct=True)}</b> • unsure: {_fmt(rdata.get('unsure_rate'), pct=True)} • "
                f"answered {rdata.get('n_answered', 0)}</p>"
            )
            n_overlap = rdata.get("n_overlap", 0)
            if n_overlap:
                matched = rdata.get("n_overlap_matched", 0)
                parts.append(
                    f"<p style='margin:0 0 6px 0; color:#FFCC80; font-size:12px;'>"
                    f"⚠ {n_overlap} clip(s) had two or more behaviors flagged at once and were "
                    f"excluded from scoring (not counted against {rid}). You matched one of the "
                    f"flagged behaviors on {matched}/{n_overlap}.</p>"
                )
            parts.append(
                "<table cellpadding='5' cellspacing='0' style='border-collapse:collapse; margin:4px 0 14px 0;'>"
                "<tr style='color:#B0BEC5;'>"
                "<th align='left'>Behavior</th><th align='left'>Precision</th>"
                "<th align='left'>Recall</th><th align='left'>F1</th>"
                "<th>TP</th><th>FP</th><th>FN</th></tr>"
            )
            for b, pb in (rdata.get("per_behavior") or {}).items():
                parts.append(
                    "<tr style='border-top:1px solid #37474F;'>"
                    f"<td style='color:#ECEFF1;'>{self._behavior_label(b)}</td>"
                    f"<td>{self._bar(pb.get('precision'), self._metric_color(pb.get('precision')))}</td>"
                    f"<td>{self._bar(pb.get('recall'), self._metric_color(pb.get('recall')))}</td>"
                    f"<td>{self._bar(pb.get('f1'), self._metric_color(pb.get('f1')))}</td>"
                    f"<td align='center'>{pb.get('tp', 0)}</td>"
                    f"<td align='center' style='color:#EF9A9A;'>{pb.get('fp', 0)}</td>"
                    f"<td align='center' style='color:#EF9A9A;'>{pb.get('fn', 0)}</td>"
                    "</tr>"
                )
            parts.append("</table>")

        # ── Where & why they disagree (confusion) ────────────────────────
        parts.append(self._render_confusion(metrics.get("confusion", {})))

        # ── Inter-rater ──────────────────────────────────────────────────
        inter = metrics.get("inter_rater", {})
        parts.append("<h3 style='color:#80CBC4;'>User vs. user (inter-rater)</h3>")
        if inter.get("n_reviewers", 0) < 2 or inter.get("shared_clips", 0) == 0:
            parts.append(
                "<p style='color:#90A4AE;'>Needs at least two reviewers with overlapping answers "
                "on the same test. Have another reviewer complete this test.</p>"
            )
        else:
            kappa = inter.get("kappa")
            parts.append(
                f"<p>{inter['n_reviewers']} reviewers • {inter['shared_clips']} shared clips • "
                f"agreement <b>{_fmt(inter.get('agreement'), pct=True)}</b> • "
                f"κ (kappa) <b style='color:{self._metric_color(kappa)};'>{_fmt(kappa)}</b> "
                f"<span style='color:#78909C;'>({self._kappa_label(kappa)})</span></p>"
            )

        # ── Intra-rater (test-retest vs the reviewer's own prior labels) ──
        parts.append(self._render_intra_rater(metrics.get("intra_rater", {})))

        # ── Behavior overlap ─────────────────────────────────────────────
        parts.append(self._render_overlap(metrics.get("overlap", {})))

        # ── Suggestions ──────────────────────────────────────────────────
        parts.append("<h3 style='color:#80CBC4;'>Suggestions</h3>")
        sev_color = {"high": "#C62828", "medium": "#F9A825", "ok": "#2E7D32"}
        parts.append("<ul style='margin-top:4px;'>")
        for s in suggestions:
            color = sev_color.get(s.get("severity", "medium"), "#F9A825")
            action_html = ""
            if s.get("action") == "apply_inhibit" and s.get("pair_a") and s.get("pair_b"):
                href = f"applyinhibit:{s['pair_a']}|{s['pair_b']}"
                action_html = (
                    f" &nbsp;<a href='{href}' style='color:#4FC3F7; text-decoration:none; "
                    f"font-weight:700;'>[Apply inhibition]</a>"
                )
            parts.append(
                f"<li style='margin-bottom:6px;'>"
                f"<span style='color:{color}; font-weight:700;'>● </span>{s.get('message', '')}"
                f"{action_html}</li>"
            )
        parts.append("</ul>")

        return self._wrap("".join(parts))

    def _render_overlap(self, overlap: dict) -> str:
        parts: list[str] = ["<h3 style='color:#80CBC4;'>Behavior overlap (simultaneous detections)</h3>"]
        pairs = overlap.get("pairs") or []
        rate = overlap.get("overall_overlap_rate")
        if not pairs:
            parts.append(
                "<p style='color:#90A4AE;'>No simultaneous behavior detections found, or temporal "
                "inference hasn't been run. ABEL allows multiple behaviors at once, so this checks "
                "how often bouts overlap.</p>"
            )
            return "".join(parts)
        parts.append(
            f"<p style='color:#B0BEC5;'>Overall, <b>{_fmt(rate, pct=True)}</b> of all flagged frames have "
            "two or more behaviors active at once. High overlap can mean thresholds are too lax or "
            "behavior inhibition is too weak.</p>"
        )
        parts.append(
            "<table cellpadding='5' cellspacing='0' style='border-collapse:collapse; margin:4px 0 12px 0;'>"
            "<tr style='color:#B0BEC5;'><th align='left'>Behavior pair</th>"
            "<th>Shared frames</th><th>% of each</th><th></th></tr>"
        )
        for pair in pairs[:8]:
            a, b = pair["a"], pair["b"]
            na, nb = self._behavior_label(a), self._behavior_label(b)
            fa, fb = pair.get("frac_a"), pair.get("frac_b")
            worst = max(fa or 0.0, fb or 0.0)
            color = "#EF9A9A" if worst >= 0.35 else ("#FFCC80" if worst >= 0.15 else "#CFD8DC")
            href = f"applyinhibit:{a}|{b}"
            parts.append(
                "<tr style='border-top:1px solid #37474F;'>"
                f"<td style='color:#ECEFF1;'>{na} ↔ {nb}</td>"
                f"<td align='center'>{pair.get('overlap_frames', 0)}</td>"
                f"<td align='center' style='color:{color};'>{_fmt(fa, pct=True)} / {_fmt(fb, pct=True)}</td>"
                f"<td align='center'><a href='{href}' style='color:#4FC3F7; text-decoration:none; "
                f"font-weight:700;'>Apply inhibition</a></td></tr>"
            )
        parts.append("</table>")
        return "".join(parts)

    def _render_intra_rater(self, intra: dict) -> str:
        parts: list[str] = [
            "<h3 style='color:#80CBC4;'>Intra-rater reliability (self-consistency)</h3>"
        ]
        if not intra:
            parts.append(
                "<p style='color:#90A4AE;'>No reviewer answers yet. This compares a reviewer's "
                "quiz answers against the behaviors they previously accepted (the prior-accepted "
                "clips), measuring how consistently they agree with their own past judgments.</p>"
            )
            return "".join(parts)

        rid = self._intra_reviewer.currentText().strip()
        data = intra.get(rid)
        if data is None and intra:
            # Fall back to the first reviewer if the selection isn't available.
            rid = next(iter(intra))
            data = intra[rid]
        if not data or data.get("n", 0) == 0:
            parts.append(
                f"<p style='color:#90A4AE;'>“{rid or '—'}” has not labeled any prior-accepted clips "
                "in this test yet. Increase the prior-accepted proportion in Settings, or have this "
                "reviewer complete more of the test.</p>"
            )
            return "".join(parts)

        agree = data.get("agreement")
        kappa = data.get("kappa")
        parts.append(
            f"<p><b style='color:#ECEFF1;'>{rid}</b> vs. their own prior-accepted labels — "
            f"{data['n']} clip(s)"
            + (f", {data['n_unsure']} unsure" if data.get("n_unsure") else "")
            + f" • self-agreement <b style='color:{self._metric_color(agree)};'>{_fmt(agree, pct=True)}</b>"
            f" • κ <b style='color:{self._metric_color(kappa)};'>{_fmt(kappa)}</b> "
            f"<span style='color:#78909C;'>({self._kappa_label(kappa)})</span></p>"
        )
        per_behavior = data.get("per_behavior") or {}
        counts = data.get("per_behavior_counts") or {}
        if per_behavior:
            parts.append(
                "<table cellpadding='5' cellspacing='0' style='border-collapse:collapse; margin:4px 0 12px 0;'>"
                "<tr style='color:#B0BEC5;'><th align='left'>Behavior</th>"
                "<th align='left'>Self-agreement</th><th>Clips</th></tr>"
            )
            for b, acc in sorted(per_behavior.items(), key=lambda kv: (kv[1] is None, kv[1] or 0)):
                parts.append(
                    "<tr style='border-top:1px solid #37474F;'>"
                    f"<td style='color:#ECEFF1;'>{self._behavior_label(b)}</td>"
                    f"<td>{self._bar(acc, self._metric_color(acc))}</td>"
                    f"<td align='center'>{counts.get(b, 0)}</td></tr>"
                )
            parts.append("</table>")
        return "".join(parts)

    def _render_confusion(self, confusion: dict) -> str:
        labels = confusion.get("labels") or []
        matrix = confusion.get("matrix") or {}
        if not labels or confusion.get("n_total", 0) == 0:
            return ""
        parts: list[str] = ["<h3 style='color:#80CBC4;'>Where & why they disagree</h3>"]

        n_dis = confusion.get("n_disagreements", 0)
        n_fr = confusion.get("n_fringe_disagreements", 0)
        n_clear = confusion.get("n_clear_disagreements", 0)
        if n_dis:
            parts.append(
                f"<p style='color:#B0BEC5;'>{n_dis} disagreement(s): "
                f"<span style='color:#FFB74D;'>{n_fr} borderline</span> (near threshold) • "
                f"<span style='color:#EF9A9A;'>{n_clear} clear-cut</span> (likely model error).</p>"
            )

        # Top confusions list.
        top = confusion.get("top_confusions") or []
        if top:
            parts.append("<ul style='margin:4px 0 10px 0;'>")
            for conf in top[:5]:
                m = self._service._label_display(conf.get("machine", ""))
                u = self._service._label_display(conf.get("user", ""))
                cnt = conf.get("count", 0)
                fr = conf.get("fringe_count", 0)
                tag = f" <span style='color:#FFB74D;'>({fr} borderline)</span>" if fr else ""
                parts.append(
                    f"<li style='margin-bottom:3px;'>Model said <b style='color:#90CAF9;'>{m}</b>, "
                    f"reviewers said <b style='color:#A5D6A7;'>{u}</b> — {cnt}×{tag}</li>"
                )
            parts.append("</ul>")

        # Confusion matrix heatmap (rows = model, cols = reviewers).
        max_off = 1
        for m in labels:
            for u in labels:
                if m != u:
                    max_off = max(max_off, matrix.get(m, {}).get(u, 0))
        parts.append(
            "<p style='color:#78909C; margin-bottom:2px;'>Confusion matrix — rows: model label, "
            "columns: reviewer label. Off-diagonal cells are disagreements.</p>"
        )
        parts.append("<table cellpadding='6' cellspacing='0' style='border-collapse:collapse; margin-bottom:12px;'>")
        parts.append("<tr><td></td>")
        for u in labels:
            parts.append(
                f"<td align='center' style='color:#B0BEC5; font-size:11px;'>"
                f"{self._service._label_display(u)}</td>"
            )
        parts.append("</tr>")
        for m in labels:
            parts.append(
                f"<tr><td style='color:#B0BEC5; font-size:11px;'>{self._service._label_display(m)}</td>"
            )
            for u in labels:
                val = matrix.get(m, {}).get(u, 0)
                if m == u:
                    bg = "#1B5E20" if val else "#263238"
                else:
                    intensity = val / max_off if max_off else 0.0
                    # Red shading for disagreements scaled by frequency.
                    r = int(40 + 150 * intensity)
                    bg = f"rgb({r},40,40)" if val else "#263238"
                color = "#ECEFF1" if val else "#455A64"
                parts.append(
                    f"<td align='center' style='background:{bg}; color:{color}; "
                    f"border:1px solid #1c2530;'>{val}</td>"
                )
            parts.append("</tr>")
        parts.append("</table>")
        return "".join(parts)

    @staticmethod
    def _kappa_label(kappa: float | None) -> str:
        if kappa is None:
            return "n/a"
        if kappa >= 0.8:
            return "almost perfect"
        if kappa >= 0.6:
            return "substantial"
        if kappa >= 0.4:
            return "moderate"
        if kappa >= 0.2:
            return "fair"
        return "poor"

    def _behavior_label(self, bid: str) -> str:
        return self._service._label_display(bid)


# ===========================================================================
# Behavior grid panel
# ===========================================================================
_GRID_RESOLUTIONS = [("720 px", 720), ("1080 px", 1080), ("1440 px", 1440)]


class BehaviorGridPanel(QWidget):
    """Builds a 5×5 looping montage of strong positive bouts for one behavior."""

    def __init__(self, service: ValidationService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = service
        self._pool = QThreadPool.globalInstance()
        self._project_root: Path | None = None
        self._preview_path: Path | None = None
        self._busy = False
        self._build_ui()

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        header = QLabel("Behavior Grid")
        header.setStyleSheet("font-size: 16px; font-weight: 700; color: #ECEFF1;")
        subtitle = QLabel(
            "A 5×5 montage of strong positive bouts of one behavior, sampled across "
            "sessions, with pose keypoints overlaid. Generate, preview, and export."
        )
        subtitle.setStyleSheet("color: #90A4AE; font-size: 12px;")
        subtitle.setWordWrap(True)

        self._behavior_combo = QComboBox()
        self._behavior_combo.setMinimumWidth(160)

        self._spin_pre = QDoubleSpinBox()
        self._spin_post = QDoubleSpinBox()
        for sp in (self._spin_pre, self._spin_post):
            sp.setRange(0.0, 10.0)
            sp.setSingleStep(0.5)
            sp.setDecimals(1)
            sp.setValue(0.5)
            sp.setSuffix(" s")
            sp.setFixedWidth(80)

        self._spin_crop = QDoubleSpinBox()
        self._spin_crop.setRange(0.4, 8.0)
        self._spin_crop.setSingleStep(0.1)
        self._spin_crop.setDecimals(1)
        self._spin_crop.setValue(1.0)
        self._spin_crop.setSuffix("×")
        self._spin_crop.setFixedWidth(80)
        self._spin_crop.setToolTip(
            "Crop size around each animal. Above 1× shows more surroundings (zoom "
            "out); below 1× tightens onto the subject. (The crop is capped at half "
            "the source frame, so very large values stop zooming out at the full frame.)"
        )
        self._spin_crop.valueChanged.connect(self._on_crop_changed)

        self._spin_kp_size = QDoubleSpinBox()
        self._spin_kp_size.setRange(0.3, 5.0)
        self._spin_kp_size.setSingleStep(0.1)
        self._spin_kp_size.setDecimals(1)
        self._spin_kp_size.setValue(1.0)
        self._spin_kp_size.setSuffix("×")
        self._spin_kp_size.setFixedWidth(80)
        self._spin_kp_size.setToolTip(
            "Size of the overlaid pose-tracking dots. Above 1× draws larger dots, "
            "below 1× smaller."
        )
        self._spin_kp_size.valueChanged.connect(self._on_kp_size_changed)

        self._res_combo = QComboBox()
        for label, _px in _GRID_RESOLUTIONS:
            self._res_combo.addItem(label)
        self._res_combo.setCurrentIndex(1)  # 1080
        self._res_combo.setToolTip("Total (square) resolution of the stitched grid video.")

        self._keypoints_chk = QCheckBox("Show keypoints")
        self._keypoints_chk.setChecked(True)
        self._keypoints_chk.toggled.connect(self._spin_kp_size.setEnabled)

        self._generate_btn = QPushButton("Generate Grid")
        self._generate_btn.setToolTip("Build a fresh random montage with the current settings.")
        self._generate_btn.clicked.connect(self._generate)

        self._export_btn = QPushButton("Export…")
        self._export_btn.setToolTip("Save the current grid video to a file.")
        self._export_btn.clicked.connect(self._export)
        self._export_btn.setEnabled(False)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Behavior:"))
        controls.addWidget(self._behavior_combo)
        controls.addSpacing(10)
        controls.addWidget(QLabel("Before:"))
        controls.addWidget(self._spin_pre)
        controls.addWidget(QLabel("After:"))
        controls.addWidget(self._spin_post)
        controls.addSpacing(10)
        controls.addWidget(QLabel("Crop:"))
        controls.addWidget(self._spin_crop)
        controls.addSpacing(10)
        controls.addWidget(QLabel("Resolution:"))
        controls.addWidget(self._res_combo)
        controls.addSpacing(10)
        controls.addWidget(self._keypoints_chk)
        controls.addWidget(QLabel("Dot size:"))
        controls.addWidget(self._spin_kp_size)
        controls.addStretch()
        controls.addWidget(self._generate_btn)
        controls.addWidget(self._export_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)

        self._player = ClipPlayer()
        self._player.set_loop(True)

        self._empty = QLabel(
            "Pick a behavior and click “Generate Grid” to build a montage of its "
            "strongest detected bouts from across your sessions."
        )
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet("color: #546E7A; font-size: 13px; padding: 30px;")

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_box.addWidget(header)
        title_box.addWidget(subtitle)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addLayout(title_box)
        layout.addLayout(controls)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._player, 1)
        layout.addWidget(self._empty)
        self._player.hide()

    # ---------------------------------------------------------------- project
    def set_project(self, project_root: Path) -> None:
        self._project_root = Path(project_root)
        self._preview_path = None
        self._export_btn.setEnabled(False)
        self._player.close_clip()
        self._player.hide()
        self._empty.show()
        self._restore_crop_scale()
        self._reload_behaviors()

    def _restore_crop_scale(self) -> None:
        """Load persisted Crop × and Dot-size × values for this project."""
        try:
            settings = self._service.load_settings()
            crop = float(settings.behavior_grid_crop_scale)
            kp = float(getattr(settings, "behavior_grid_keypoint_scale", 1.0))
        except Exception:
            crop, kp = 1.0, 1.0
        self._spin_crop.blockSignals(True)
        self._spin_crop.setValue(crop)
        self._spin_crop.blockSignals(False)
        self._spin_kp_size.blockSignals(True)
        self._spin_kp_size.setValue(kp)
        self._spin_kp_size.blockSignals(False)
        self._spin_kp_size.setEnabled(self._keypoints_chk.isChecked())

    def _on_crop_changed(self, value: float) -> None:
        """Persist the Crop × value so it survives project reloads / new grids."""
        if self._project_root is None:
            return
        try:
            settings = self._service.load_settings()
            settings.behavior_grid_crop_scale = float(value)
            self._service.save_settings(settings)
        except Exception:
            logger.exception("Behavior grid: failed to persist crop scale")

    def _on_kp_size_changed(self, value: float) -> None:
        """Persist the keypoint dot-size multiplier across reloads / new grids."""
        if self._project_root is None:
            return
        try:
            settings = self._service.load_settings()
            settings.behavior_grid_keypoint_scale = float(value)
            self._service.save_settings(settings)
        except Exception:
            logger.exception("Behavior grid: failed to persist keypoint scale")

    def _reload_behaviors(self) -> None:
        self._behavior_combo.blockSignals(True)
        self._behavior_combo.clear()
        try:
            behaviors = self._service.behavior_grid_behaviors()
        except Exception:
            logger.exception("Behavior grid: failed to list behaviors")
            behaviors = []
        for bid, name in behaviors:
            self._behavior_combo.addItem(name, userData=bid)
        self._behavior_combo.blockSignals(False)
        if not behaviors:
            self._empty.setText(
                "No behaviors with detected bouts are available. Define behaviors and run "
                "temporal inference first."
            )

    # ---------------------------------------------------------------- generate
    def _generate(self) -> None:
        if self._busy:
            return
        bid = self._behavior_combo.currentData()
        if not bid:
            QMessageBox.information(self, "Behavior Grid", "Select a behavior first.")
            return
        out_path = self._service.behavior_grid_preview_path()
        grid_px = _GRID_RESOLUTIONS[max(0, self._res_combo.currentIndex())][1]
        self._busy = True
        self._set_busy_ui(True)
        worker = TaskWorker(
            self._service.render_behavior_grid,
            str(bid),
            float(self._spin_pre.value()),
            float(self._spin_post.value()),
            int(grid_px),
            bool(self._keypoints_chk.isChecked()),
            Path(out_path),
            crop_scale=float(self._spin_crop.value()),
            keypoint_scale=float(self._spin_kp_size.value()),
        )
        worker.signals.finished.connect(self._on_generated)
        worker.signals.failed.connect(self._on_generate_failed)
        self._pool.start(worker)

    def _set_busy_ui(self, busy: bool) -> None:
        self._progress_bar.setVisible(busy)
        self._progress_bar.setRange(0, 0 if busy else 100)
        self._generate_btn.setEnabled(not busy)
        if busy:
            self._generate_btn.setText("Generating…")
        else:
            self._generate_btn.setText("Generate Grid")

    def _on_generated(self, path: object) -> None:
        self._busy = False
        self._set_busy_ui(False)
        out = Path(str(path))
        if not out.exists():
            QMessageBox.information(self, "Behavior Grid", "No grid video was produced.")
            return
        self._preview_path = out
        self._export_btn.setEnabled(True)
        self._empty.hide()
        self._player.show()
        self._player.load_clip(str(out), autoplay=True)

    def _on_generate_failed(self, tb: str) -> None:
        self._busy = False
        self._set_busy_ui(False)
        logger.error("Behavior grid generation failed:\n%s", tb)
        msg = tb.strip().splitlines()[-1] if tb.strip() else "Unknown error."
        QMessageBox.critical(self, "Behavior Grid", f"Failed to build grid:\n\n{msg}")

    # ---------------------------------------------------------------- export
    def _export(self) -> None:
        if not self._preview_path or not self._preview_path.exists():
            QMessageBox.information(self, "Export", "Generate a grid first.")
            return
        from datetime import datetime  # noqa: PLC0415

        bid = self._behavior_combo.currentText().strip() or "behavior"
        default = f"behavior_grid_{bid}_{datetime.now().strftime('%Y%m%d_%H%M')}.mp4"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Behavior Grid", default, "MP4 Video (*.mp4)"
        )
        if not path:
            return
        import shutil  # noqa: PLC0415

        try:
            shutil.copyfile(self._preview_path, path)
        except Exception as exc:
            logger.exception("Behavior grid export failed")
            QMessageBox.critical(self, "Export Failed", f"Could not export grid:\n\n{exc}")
            return
        QMessageBox.information(self, "Export Complete", f"Saved grid to:\n{path}")


# ===========================================================================
# Container
# ===========================================================================
class ValidationTab(QWidget):
    """Top-level Validation tab hosting Overview / Quiz / Results subtabs."""

    def __init__(
        self,
        service: ValidationService,
        behavior_service: BehaviorService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._behaviors = behavior_service
        self._project_root: Path | None = None

        self.overview_panel = ValidationOverviewPanel(service)
        self.quiz_panel = ValidationQuizPanel(service, behavior_service)
        self.results_panel = ValidationResultsPanel(service)
        self.behavior_grid_panel = BehaviorGridPanel(service)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._tabs.addTab(self.overview_panel, "Overview")
        self._tabs.addTab(self.quiz_panel, "Validation Quiz")
        self._tabs.addTab(self.results_panel, "Results & Suggestions")
        self._tabs.addTab(self.behavior_grid_panel, "Behavior Grid")
        self._tabs.currentChanged.connect(self._on_sub_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs)

        self.quiz_panel.run_changed.connect(self.results_panel.refresh)
        self.quiz_panel.answers_changed.connect(self.results_panel.refresh)
        # If a test is deleted from Results, the quiz may have it open — reload it.
        self.results_panel.run_deleted.connect(self.quiz_panel.reload)

    def set_project(self, project_root: Path) -> None:
        self._project_root = Path(project_root)
        self._service.set_project(self._project_root)
        self.overview_panel.refresh()
        self.quiz_panel.set_project(self._project_root)
        self.results_panel.refresh()
        self.behavior_grid_panel.set_project(self._project_root)

    def _on_sub_changed(self, index: int) -> None:
        widget = self._tabs.widget(index)
        if widget is self.overview_panel:
            self.overview_panel.refresh()
        elif widget is self.results_panel:
            self.results_panel.refresh()
