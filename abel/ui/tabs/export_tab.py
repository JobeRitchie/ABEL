"""Export tab for writing reviewed outputs."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re
import time
from typing import Any

from PySide6.QtCore import Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.services.behavior_service import BehaviorService
from abel.services.candidate_service import CandidateGenerationService
from abel.services.export_service import ExportService
from abel.services.review_service import ReviewService
from abel.workers.task_worker import TaskWorker


_SESSION_PICKER_ROWS_PER_COLUMN = 10


class BehaviogramDialog(QDialog):
    """Interactive per-subject behaviogram viewer."""

    def __init__(self, data: dict[str, dict[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Behaviogram Viewer")
        self.resize(980, 640)
        self._data = data

        self._subject_combo = QComboBox()
        self._subject_combo.addItems(sorted(data.keys()))
        self._subject_combo.currentIndexChanged.connect(self._render_selected_subject)

        self._summary = QLabel("No subject selected.")
        self._summary.setWordWrap(True)

        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._canvas)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Subject:"))
        top_row.addWidget(self._subject_combo)
        top_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addWidget(self._summary)
        layout.addWidget(scroll, 1)

        self._render_selected_subject()

    def _render_selected_subject(self) -> None:
        subject = self._subject_combo.currentText().strip()
        if not subject or subject not in self._data:
            self._summary.setText("No behaviogram data available.")
            self._canvas.clear()
            return

        subject_block = self._data[subject]
        max_end_frame = int(subject_block.get("max_end_frame", -1))
        behaviors = subject_block.get("behaviors", {})
        if not isinstance(behaviors, dict) or not behaviors:
            self._summary.setText(f"Subject {subject}: no accepted behavior clips.")
            self._canvas.clear()
            return

        behavior_names = sorted(str(b) for b in behaviors.keys())
        frame_span = max(1, max_end_frame + 1)
        left_pad = 180
        row_h = 26
        graph_w = 760
        graph_h = max(80, row_h * len(behavior_names))
        width = left_pad + graph_w + 16
        height = graph_h + 48

        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(QColor("#0e1620"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#90a4ae")))
        painter.drawRect(left_pad, 20, graph_w, graph_h)

        for idx, behavior in enumerate(behavior_names):
            y0 = 24 + idx * row_h
            y_mid = y0 + (row_h // 2)

            painter.setPen(QPen(QColor("#cfd8dc")))
            painter.drawText(8, y_mid + 4, behavior)

            painter.setPen(QPen(QColor("#263238")))
            painter.drawLine(left_pad, y0 + row_h - 2, left_pad + graph_w, y0 + row_h - 2)

            intervals = behaviors.get(behavior, [])
            if not isinstance(intervals, list):
                continue
            color = QColor.fromHsv((idx * 43) % 360, 190, 220)
            for interval in intervals:
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    continue
                start, end = int(interval[0]), int(interval[1])
                x0 = left_pad + int(graph_w * (max(0, start) / frame_span))
                x1 = left_pad + int(graph_w * (max(0, end) / frame_span))
                if x1 < x0:
                    x1 = x0
                x1 = min(left_pad + graph_w - 1, x1)
                painter.fillRect(x0, y0 + 2, max(1, x1 - x0 + 1), row_h - 6, color)

        painter.setPen(QPen(QColor("#90a4ae")))
        painter.drawText(left_pad, graph_h + 40, "0")
        painter.drawText(left_pad + graph_w - 40, graph_h + 40, str(max_end_frame))
        painter.end()

        pix = QPixmap.fromImage(image)
        self._canvas.setPixmap(pix)
        self._canvas.setMinimumSize(width, height)
        self._summary.setText(
            f"Subject {subject} | Behaviors: {len(behavior_names)} | Timeline: 0-{max_end_frame} frames"
        )


class ExportTab(QWidget):
    progress_update_requested = Signal(int, int, str)

    def __init__(
        self,
        export_service: ExportService,
        candidate_service: CandidateGenerationService,
        review_service: ReviewService,
        behavior_service: BehaviorService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = export_service
        self._candidates = candidate_service
        self._review = review_service
        self._behavior_service = behavior_service
        self._project_root: Path | None = None
        self._pool = QThreadPool.globalInstance()
        self._exporting_labeled_videos = False
        self._running_simple_export = False
        self._labeled_export_started_at: float | None = None
        self._labeled_export_first_segment_started_at: float | None = None
        self._labeled_export_first_segment_seconds: float | None = None
        self._labeled_export_total_segments: int = 0
        self._labeled_export_last_segment: int = 0
        self._labeled_export_global_total_frames: int = 0
        self._labeled_export_segment_done_frames: dict[int, int] = {}
        self._labeled_export_segment_total_frames: dict[int, int] = {}
        self.progress_update_requested.connect(self._on_progress_update)

        self._csv_filename = QLineEdit("review_export.csv")
        self._docx_filename = QLineEdit("behavior_presence.docx")
        self._boutframes_filename = QLineEdit("boutframes.xlsx")
        self._boutframes_include_end = QCheckBox("Include end-frame columns")
        self._boutframes_include_end.setChecked(False)
        self._boutframes_binary_mode = QCheckBox("Binary frame labels (0/1)")
        self._boutframes_binary_mode.setChecked(False)
        self._boutframes_binary_mode.setToolTip(
            "Export one row per frame with a 0/1 column per behavior instead of bout start-frame lists."
        )
        self._boutframes_behavior_filter: list[str] | None = None  # None = all behaviors
        self._boutframes_select_btn = QPushButton("Select Behaviors\u2026")
        self._boutframes_select_btn.setToolTip("Choose which behaviors to include in the export.")
        self._boutframes_select_btn.clicked.connect(self._pick_boutframe_behaviors)
        self._status = QLabel("No export run yet.")
        self._export_log = QTextEdit()
        self._export_log.setReadOnly(True)
        self._export_log.setMinimumHeight(140)
        self._export_log.setPlaceholderText("Export progress and debug messages will appear here.")

        self._export_csv_btn = QPushButton("Export CSV")
        self._export_csv_btn.clicked.connect(self._export_csv)
        self._export_docx_btn = QPushButton("Export Word Behavior Matrix")
        self._export_docx_btn.clicked.connect(self._export_docx)
        self._export_bout_btn = QPushButton("Export Boutframes Workbook")
        self._export_bout_btn.clicked.connect(self._export_boutframes)
        self._export_labeled_video_btn = QPushButton("Export Labeled Tracking Videos")
        self._export_labeled_video_btn.clicked.connect(self._export_labeled_videos)
        self._customize_overlay_btn = QPushButton("Customize Overlay…")
        self._customize_overlay_btn.setToolTip("Preview and adjust overlay appearance before exporting")
        self._customize_overlay_btn.clicked.connect(self._open_overlay_settings)
        view_behaviogram_btn = QPushButton("View Behaviogram")
        view_behaviogram_btn.clicked.connect(self._view_behaviogram)

        row_csv = QHBoxLayout()
        row_csv.addWidget(QLabel("CSV filename:"))
        row_csv.addWidget(self._csv_filename, 1)
        row_csv.addWidget(self._export_csv_btn)

        row_docx = QHBoxLayout()
        row_docx.addWidget(QLabel("Word filename:"))
        row_docx.addWidget(self._docx_filename, 1)
        row_docx.addWidget(self._export_docx_btn)

        row_bout = QHBoxLayout()
        row_bout.addWidget(QLabel("Boutframes filename:"))
        row_bout.addWidget(self._boutframes_filename, 1)
        row_bout.addWidget(self._boutframes_include_end)
        row_bout.addWidget(self._boutframes_binary_mode)
        row_bout.addWidget(self._boutframes_select_btn)
        row_bout.addWidget(self._export_bout_btn)

        row_view = QHBoxLayout()
        row_view.addWidget(QLabel("Accepted clips visualization:"))
        row_view.addStretch(1)
        row_view.addWidget(view_behaviogram_btn)

        row_labeled = QHBoxLayout()
        row_labeled.addWidget(QLabel("Whole-video overlays:"))
        row_labeled.addStretch(1)
        row_labeled.addWidget(self._customize_overlay_btn)
        row_labeled.addWidget(self._export_labeled_video_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Idle")
        self._progress.setTextVisible(True)
        self._progress.setMinimumHeight(22)
        self._progress.setStyleSheet(
            "QProgressBar {"
            "  border: 1px solid #37474F;"
            "  border-radius: 6px;"
            "  background: #1A2027;"
            "  color: #ECEFF1;"
            "  font-size: 11px;"
            "  text-align: center;"
            "}"
            "QProgressBar::chunk {"
            "  border-radius: 5px;"
            "  background: qlineargradient("
            "    x1:0, y1:0, x2:1, y2:0,"
            "    stop:0 #1565C0, stop:0.5 #42A5F5, stop:1 #1565C0"
            "  );"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.addLayout(row_csv)
        layout.addLayout(row_docx)
        layout.addLayout(row_bout)
        layout.addLayout(row_labeled)
        layout.addLayout(row_view)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        layout.addWidget(QLabel("Export Debug Log"))
        layout.addWidget(self._export_log)
        layout.addStretch()

    def _append_export_log(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        self._export_log.append(text)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 1.0:
            return f"{seconds * 1000.0:.0f} ms"
        if seconds < 60.0:
            return f"{seconds:.1f} s"
        mins = int(seconds // 60)
        rem = int(seconds % 60)
        return f"{mins}m {rem:02d}s"

    @staticmethod
    def _ascii_bar(frac: float, width: int = 20) -> str:
        f = min(1.0, max(0.0, float(frac)))
        filled = int(round(f * width))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    @staticmethod
    def _parse_segment_progress(message: str) -> tuple[int, int, int | None, int | None] | None:
        text = str(message or "")
        m = re.search(r"segment\s+(\d+)/(\d+)", text)
        if not m:
            return None
        seg_idx = max(1, int(m.group(1)))
        seg_total = max(seg_idx, int(m.group(2)))
        frame_m = re.search(r"frame\s+([\d,]+)/([\d,]+)", text)
        if frame_m:
            frame_done = max(0, int(str(frame_m.group(1)).replace(",", "")))
            frame_total = max(1, int(str(frame_m.group(2)).replace(",", "")))
            frame_done = min(frame_done, frame_total)
            return seg_idx, seg_total, frame_done, frame_total
        return seg_idx, seg_total, None, None

    @staticmethod
    def _parse_total_export_frames(message: str) -> int | None:
        text = str(message or "")
        m = re.search(r"\(([\d,]+)\s+frames\s+total\)", text)
        if not m:
            return None
        try:
            return max(0, int(str(m.group(1)).replace(",", "")))
        except Exception:
            return None

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._service.set_project(project_root)
        self._candidates.set_project(project_root)
        self._review.set_project(project_root)
        if self._behavior_service is not None:
            self._behavior_service.set_project(project_root)

    def _load_inputs(self):
        return self._candidates.load_candidates(), self._review.load_decisions()

    def _export_csv(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        candidates, decisions = self._load_inputs()
        out = self._service.export_review_csv(
            candidates=candidates,
            decisions=decisions,
            filename=self._csv_filename.text().strip() or "review_export.csv",
        )
        if not out.success:
            QMessageBox.warning(self, "Export failed", "\n".join(out.warnings) or "Unknown error")
            return
        self._status.setText(f"Exported {out.n_rows} rows to {out.output_path}")

    def _export_docx(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return
        if self._running_simple_export:
            return

        candidates, decisions = self._load_inputs()
        self._running_simple_export = True
        self._set_export_buttons_enabled(False)
        self._progress.setRange(0, 0)
        self._progress.setFormat("Working...")
        self._status.setText("Exporting Word behavior matrix...")

        filename = self._docx_filename.text().strip() or "behavior_presence.docx"

        worker = TaskWorker(
            self._service.export_behavior_presence_docx,
            candidates,
            decisions,
            filename,
        )
        worker.signals.finished.connect(self._on_docx_export_finished)
        worker.signals.failed.connect(self._on_simple_export_failed)
        self._pool.start(worker)

    def _pick_boutframe_behaviors(self) -> None:
        """Open a dialog to choose which behaviors to include in boutframe exports."""
        # Use behavior definitions directly — avoids heavy I/O on the main thread.
        behavior_defs = self._behavior_service.behaviors if self._behavior_service else []
        available = sorted(b.name for b in behavior_defs if b.is_active)
        if not available:
            QMessageBox.information(self, "No Behaviors", "No active behaviors defined.")
            return

        current = set(self._boutframes_behavior_filter) if self._boutframes_behavior_filter is not None else set(available)

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Behaviors for Export")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Choose which behaviors to include:"))

        checks: list[QCheckBox] = []
        for name in available:
            cb = QCheckBox(name)
            cb.setChecked(name in current)
            layout.addWidget(cb)
            checks.append(cb)

        sel_row = QHBoxLayout()
        sel_all = QPushButton("Select All")
        desel_all = QPushButton("Deselect All")
        sel_all.clicked.connect(lambda: [cb.setChecked(True) for cb in checks])
        desel_all.clicked.connect(lambda: [cb.setChecked(False) for cb in checks])
        sel_row.addWidget(sel_all)
        sel_row.addWidget(desel_all)
        sel_row.addStretch(1)
        layout.addLayout(sel_row)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = [name for cb, name in zip(checks, available) if cb.isChecked()]
        if not selected:
            QMessageBox.information(self, "No Behaviors Selected", "Select at least one behavior.")
            return
        self._boutframes_behavior_filter = selected if len(selected) < len(available) else None
        n = len(selected)
        total = len(available)
        self._boutframes_select_btn.setText(
            f"Select Behaviors\u2026 ({n}/{total})" if self._boutframes_behavior_filter is not None
            else "Select Behaviors\u2026"
        )

    def _export_boutframes(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        binary_mode = bool(self._boutframes_binary_mode.isChecked())
        candidates, decisions = self._load_inputs()
        out = self._service.export_boutframes_xlsx(
            candidates=candidates,
            decisions=decisions,
            filename=self._boutframes_filename.text().strip() or "boutframes.xlsx",
            include_end_frames=bool(self._boutframes_include_end.isChecked()) and not binary_mode,
            behavior_filter=self._boutframes_behavior_filter,
            binary_mode=binary_mode,
        )
        if not out.success:
            QMessageBox.warning(self, "Export failed", "\n".join(out.warnings) or "Unknown error")
            return
        if len(out.output_paths) > 1:
            self._status.setText(
                f"Exported {len(out.output_paths)} boutframes workbooks to "
                f"{out.output_paths[0].parent}"
            )
        else:
            self._status.setText(f"Exported boutframes workbook to {out.output_path}")

    def _view_behaviogram(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        candidates, decisions = self._load_inputs()
        data = self._service.build_behaviogram(candidates, decisions)
        if not data:
            QMessageBox.information(
                self,
                "No Data",
                "No accepted behavior clips are available for behaviogram viewing.",
            )
            return

        dialog = BehaviogramDialog(data, self)
        dialog.exec()

    def _open_overlay_settings(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        behavior_info = self._service._get_behavior_overlay_info()

        from abel.ui.overlay_settings_dialog import OverlaySettingsDialog  # noqa: PLC0415

        dlg = OverlaySettingsDialog(self._project_root, behavior_info, parent=self)
        dlg.exec()

    def _pick_behaviors(self) -> list[str] | None:
        """Show a dialog to choose which behaviors to annotate. Returns selected names, or None if cancelled."""
        behavior_defs = self._behavior_service.behaviors if self._behavior_service else []
        active_names = [b.name for b in behavior_defs if b.is_active]

        if not active_names:
            return []  # No behaviors defined; export all (empty = no filter)

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Behaviors to Annotate")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Choose which behaviors to include in the video overlay:"))

        checks: list[QCheckBox] = []
        for name in active_names:
            cb = QCheckBox(name)
            cb.setChecked(name.strip().lower() not in {"no behavior", "no_behaviour"})
            layout.addWidget(cb)
            checks.append(cb)

        select_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        select_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb in checks])
        deselect_all_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in checks])
        select_row.addWidget(select_all_btn)
        select_row.addWidget(deselect_all_btn)
        select_row.addStretch(1)
        layout.addLayout(select_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        selected = [name for cb, name in zip(checks, active_names) if cb.isChecked()]
        if not selected:
            QMessageBox.information(
                self,
                "No Behaviors Selected",
                "Select at least one behavior to include in the annotation.",
            )
            return None
        return selected

    def _pick_subjects(self) -> list[str] | None:
        """Show a dialog to choose which sessions to export. Returns selected session IDs, or None if cancelled."""
        sessions = self._service.list_available_sessions()
        if not sessions:
            return []

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Sessions to Export")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Choose which sessions to include in labeled video export:"))

        checks: list[tuple[QCheckBox, str]] = []
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        checks_widget = QWidget()
        checks_layout = QGridLayout(checks_widget)
        checks_layout.setContentsMargins(4, 4, 4, 4)
        checks_layout.setHorizontalSpacing(18)
        checks_layout.setVerticalSpacing(6)

        for index, (label, session_id) in enumerate(sessions):
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setToolTip(f"Session ID: {session_id}")
            row = index % _SESSION_PICKER_ROWS_PER_COLUMN
            col = index // _SESSION_PICKER_ROWS_PER_COLUMN
            checks_layout.addWidget(cb, row, col)
            checks.append((cb, session_id))

        checks_layout.setRowStretch(_SESSION_PICKER_ROWS_PER_COLUMN, 1)
        checks_layout.setColumnStretch(max(1, len(sessions) // _SESSION_PICKER_ROWS_PER_COLUMN), 1)
        scroll.setWidget(checks_widget)
        layout.addWidget(scroll, 1)

        approx_columns = max(1, (len(sessions) + _SESSION_PICKER_ROWS_PER_COLUMN - 1) // _SESSION_PICKER_ROWS_PER_COLUMN)
        dlg.resize(min(960, 320 + max(0, approx_columns - 1) * 220), 520)

        select_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        select_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb, _ in checks])
        deselect_all_btn.clicked.connect(lambda: [cb.setChecked(False) for cb, _ in checks])
        select_row.addWidget(select_all_btn)
        select_row.addWidget(deselect_all_btn)
        select_row.addStretch(1)
        layout.addLayout(select_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        selected = [session_id for cb, session_id in checks if cb.isChecked()]
        if not selected:
            QMessageBox.information(
                self,
                "No Sessions Selected",
                "Select at least one session to export labeled videos.",
            )
            return None
        return selected

    def _pick_overlay_mode(self) -> str | None:
        """Show a dialog to choose between Basic and Advanced overlay modes."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Overlay Mode")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Choose the video overlay style:"))

        combo = QComboBox()
        combo.addItem("Basic – active behavior label", "basic")
        combo.addItem("Advanced – live probabilities & cumulative durations", "advanced")
        layout.addWidget(combo)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return str(combo.currentData())

    def _export_labeled_videos(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return
        if self._exporting_labeled_videos:
            return

        session_filter = self._pick_subjects()
        if session_filter is None:
            return  # User cancelled
        behavior_filter = self._pick_behaviors()
        if behavior_filter is None:
            return  # User cancelled
        overlay_mode = self._pick_overlay_mode()
        if overlay_mode is None:
            return  # User cancelled

        candidates, decisions = self._load_inputs()
        self._exporting_labeled_videos = True
        self._labeled_export_started_at = time.monotonic()
        self._labeled_export_first_segment_started_at = None
        self._labeled_export_first_segment_seconds = None
        self._labeled_export_total_segments = 0
        self._labeled_export_last_segment = 0
        self._labeled_export_global_total_frames = 0
        self._labeled_export_segment_done_frames = {}
        self._labeled_export_segment_total_frames = {}
        self._export_labeled_video_btn.setEnabled(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Starting...")
        self._status.setText("Exporting labeled tracking videos...")
        self._export_log.clear()
        self._append_export_log(
            "Starting labeled video export "
            f"(behaviors={behavior_filter if behavior_filter else 'all'}, "
            f"sessions={session_filter if session_filter else 'all'}, "
            f"overlay={overlay_mode})."
        )

        def _task():
            return self._service.export_labeled_tracking_videos(
                candidates,
                decisions,
                progress_callback=lambda done, total, msg: self.progress_update_requested.emit(done, total, msg),
                behavior_filter=behavior_filter if behavior_filter else None,
                session_filter=session_filter if session_filter else None,
                overlay_mode=overlay_mode,
            )

        worker = TaskWorker(_task)
        worker.signals.finished.connect(self._on_labeled_export_finished)
        worker.signals.failed.connect(self._on_labeled_export_failed)
        self._pool.start(worker)

    def _set_export_buttons_enabled(self, enabled: bool) -> None:
        self._export_csv_btn.setEnabled(enabled)
        self._export_docx_btn.setEnabled(enabled)
        self._export_bout_btn.setEnabled(enabled)
        self._export_labeled_video_btn.setEnabled(enabled and not self._exporting_labeled_videos)

    @Slot(object)
    def _on_docx_export_finished(self, out) -> None:
        self._running_simple_export = False
        self._set_export_buttons_enabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(1 if out and out.success else 0)
        self._progress.setFormat("Done" if out and out.success else "Failed")

        if not out or not out.success:
            QMessageBox.warning(self, "Export failed", "\n".join(out.warnings) if out else "Unknown error")
            return

        if out.warnings:
            self._status.setText(
                f"Exported Word report ({out.n_rows} rows) to {out.output_path} with {len(out.warnings)} warning(s)."
            )
            QMessageBox.information(self, "Export completed with warnings", "\n".join(out.warnings))
        else:
            self._status.setText(f"Exported Word report ({out.n_rows} frame rows) to {out.output_path}")

    @Slot(str)
    def _on_simple_export_failed(self, traceback_text: str) -> None:
        self._running_simple_export = False
        self._set_export_buttons_enabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Error")
        self._status.setText("Export failed.")
        QMessageBox.warning(self, "Export failed", traceback_text[:800])

    @Slot(object)
    def _on_labeled_export_finished(self, out) -> None:
        self._exporting_labeled_videos = False
        self._export_labeled_video_btn.setEnabled(True)
        self._labeled_export_first_segment_started_at = None
        self._labeled_export_first_segment_seconds = None
        self._labeled_export_total_segments = 0
        self._labeled_export_last_segment = 0
        self._labeled_export_global_total_frames = 0
        self._labeled_export_segment_done_frames = {}
        self._labeled_export_segment_total_frames = {}
        if not out.success:
            self._progress.setFormat("Failed")
            self._append_export_log("Export failed: " + ("; ".join(out.warnings) if out and out.warnings else "Unknown error"))
            QMessageBox.warning(self, "Export failed", "\n".join(out.warnings) or "Unknown error")
            return

        self._progress.setValue(self._progress.maximum())
        self._progress.setFormat("Done")
        self._status.setText(
            f"Exported labeled tracking videos ({out.n_rows} annotated frames) to {out.output_path}"
        )
        self._append_export_log(self._status.text())

    @Slot(str)
    def _on_labeled_export_failed(self, traceback_text: str) -> None:
        self._exporting_labeled_videos = False
        self._export_labeled_video_btn.setEnabled(True)
        self._labeled_export_first_segment_started_at = None
        self._labeled_export_first_segment_seconds = None
        self._labeled_export_total_segments = 0
        self._labeled_export_last_segment = 0
        self._labeled_export_global_total_frames = 0
        self._labeled_export_segment_done_frames = {}
        self._labeled_export_segment_total_frames = {}
        self._progress.setFormat("Error")
        self._status.setText("Labeled video export failed.")
        self._append_export_log("Labeled video export failed.")
        self._append_export_log(traceback_text[:800])
        QMessageBox.warning(self, "Export failed", traceback_text[:800])

    @Slot(int, int, str)
    def _on_progress_update(self, done: int, total: int, message: str) -> None:
        parsed = self._parse_segment_progress(message)
        now = time.monotonic()

        if parsed is None:
            self._progress.setRange(0, max(1, total))
            self._progress.setValue(min(max(0, done), max(1, total)))
            self._progress.setFormat(f"{done}/{total} sessions")
            self._status.setText(message)
            self._append_export_log(message)
            return

        seg_idx, seg_total, frame_done, frame_total = parsed
        self._labeled_export_total_segments = max(self._labeled_export_total_segments, seg_total)
        parsed_total_export_frames = self._parse_total_export_frames(message)
        if parsed_total_export_frames is not None:
            self._labeled_export_global_total_frames = max(
                self._labeled_export_global_total_frames,
                parsed_total_export_frames,
            )

        self._labeled_export_last_segment = max(self._labeled_export_last_segment, seg_idx)

        sub_frac = 0.0
        if frame_done is not None and frame_total is not None and frame_total > 0:
            self._labeled_export_segment_total_frames[seg_idx] = max(
                self._labeled_export_segment_total_frames.get(seg_idx, 0),
                frame_total,
            )
            self._labeled_export_segment_done_frames[seg_idx] = max(
                self._labeled_export_segment_done_frames.get(seg_idx, 0),
                frame_done,
            )
            sub_frac = min(1.0, max(0.0, frame_done / frame_total))

        total_segments = max(1, self._labeled_export_total_segments)
        done_total_frames = 0
        if self._labeled_export_global_total_frames > 0:
            for _idx, _done in self._labeled_export_segment_done_frames.items():
                _seg_total = self._labeled_export_segment_total_frames.get(_idx, 0)
                if _seg_total > 0:
                    done_total_frames += min(_done, _seg_total)
                else:
                    done_total_frames += max(0, _done)
            done_total_frames = max(0, min(done_total_frames, self._labeled_export_global_total_frames))

            self._progress.setRange(0, self._labeled_export_global_total_frames)
            self._progress.setValue(done_total_frames)
            frac_total = done_total_frames / max(1, self._labeled_export_global_total_frames)
        else:
            value_scaled = int(round(((seg_idx - 1) + sub_frac) * 100.0))
            max_scaled = total_segments * 100
            self._progress.setRange(0, max_scaled)
            self._progress.setValue(min(max(0, value_scaled), max_scaled))
            frac_total = min(1.0, max(0.0, ((seg_idx - 1) + sub_frac) / total_segments))
        bar_text = self._ascii_bar(frac_total)

        elapsed = 0.0
        if self._labeled_export_started_at is not None:
            elapsed = max(0.0, now - self._labeled_export_started_at)

        if elapsed > 0.0 and frac_total > 0.0:
            eta_seconds = max(0.0, elapsed * (1.0 - frac_total) / max(frac_total, 1e-6))
            eta_local = datetime.now() + timedelta(seconds=eta_seconds)
            timing_text = (
                f"elapsed {self._fmt_duration(elapsed)} | "
                f"ETA {self._fmt_duration(eta_seconds)} | "
                f"finish ~ {eta_local.strftime('%H:%M:%S')}"
            )
        else:
            timing_text = f"elapsed {self._fmt_duration(elapsed)} | ETA --"

        if self._labeled_export_global_total_frames > 0:
            seg_text = f"{done_total_frames:,} / {self._labeled_export_global_total_frames:,} frames"
        elif frame_done is not None and frame_total is not None:
            seg_text = f"segment {seg_idx}/{total_segments} frame {frame_done:,}/{frame_total:,}"
        else:
            seg_text = f"segment {seg_idx}/{total_segments}"

        progress_text = f"{seg_text} {bar_text} {frac_total * 100.0:5.1f}%"
        self._progress.setFormat(progress_text)

        status_text = f"{message} | {timing_text}"
        self._status.setText(status_text)
        self._append_export_log(status_text)
