"""Dedicated review tab for temporal refinement outputs."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from abel.core.project_manager import ProjectManager
from abel.services.behavior_service import BehaviorService
from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseProcessingService
from abel.services.review_service import ReviewService
from abel.models.schemas import CandidateWindow, ReviewDecisionType, ReviewerLabelRecord
from abel.temporal_refinement.bout_postprocess import (
    binary_trace_to_intervals,
    merge_close_bouts,
    remove_short_bouts,
    smooth_probabilities,
    threshold_probabilities,
)
from abel.temporal_refinement.temporal_refinement_service import TemporalRefinementConfig
from abel.ui.tabs.review_tab import CandidateVideoPlayer
from abel.workers.task_worker import TaskWorker
from abel.utils.error_text import format_task_error


class TemporalReviewTab(QWidget):
    """Review temporal-refinement metrics and bout outputs by behavior/session."""

    bout_candidates_requested = Signal(list, str)
    bout_candidates_append_requested = Signal(list, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_root: Path | None = None
        self._manager: ProjectManager | None = None
        self._behaviors = BehaviorService()
        self._imports = ImportService()
        self._pose = PoseProcessingService()
        # Cache of (x, y, conf) numpy arrays per session for the keypoint overlay.
        self._session_pose_cache: dict[str, tuple] = {}
        self._subject_by_session: dict[str, str] = {}
        self._loaded_session_video_id: str | None = None
        self._trace_paths: dict[str, str] = {}
        self._trace_probability_lookup: dict[str, pd.DataFrame] = {}
        self._competition_models: dict[str, str] = {}
        self._competition_excluded: list[str] = []
        self._bout_paths: dict[str, str] = {}
        self._bout_paths_by_behavior: dict[str, dict[str, str]] = {}
        self._session_rows: list[dict[str, Any]] = []
        self._review_settings: dict[str, dict[str, Any]] = {}
        self._pool = QThreadPool.globalInstance()
        self._is_refreshing = False
        self._trace_init_scheduled = False
        self._review_service = ReviewService()

        self._status = QLabel("Open a project to review temporal refinement outputs.")
        self._status.setWordWrap(True)

        self._behavior = QComboBox()
        self._behavior.addItem("All reviewed positives", userData="target_behavior")
        self._behavior.currentIndexChanged.connect(self._on_behavior_changed)

        self._sort_by = QComboBox()
        self._sort_by.addItem("Worst Frame F1", userData="f1_asc")
        self._sort_by.addItem("Best Frame F1", userData="f1_desc")
        self._sort_by.addItem("Worst Bout IoU", userData="iou_asc")
        self._sort_by.addItem("Best Bout IoU", userData="iou_desc")
        self._sort_by.addItem("Highest Onset Error", userData="onset_desc")
        self._sort_by.addItem("Highest Offset Error", userData="offset_desc")
        self._sort_by.addItem("Most Bouts", userData="nbouts_desc")
        self._sort_by.addItem("Fewest Bouts", userData="nbouts_asc")
        self._sort_by.currentIndexChanged.connect(self._rebuild_metrics_table)
        self._sort_by.setVisible(False)

        self._session = QComboBox()
        self._session.currentIndexChanged.connect(self._refresh_session_bouts)

        self._per_behavior_thresholds_btn = QPushButton("Per-Behavior Thresholds")
        self._per_behavior_thresholds_btn.clicked.connect(self._open_per_behavior_thresholds_dialog)

        self._refresh_btn = QPushButton("Refresh (Apply Settings)")
        self._refresh_btn.clicked.connect(self._run_refresh)

        # Read-only label showing the saved settings for the selected behavior.
        # Edited via Per-Behavior Thresholds dialog only.
        self._trace_settings_label = QLabel("")

        self._session_quality_btn = QPushButton("Session Quality…")
        self._session_quality_btn.setToolTip(
            "Detect sessions with abnormally low model confidence or unusual bout patterns "
            "by comparing each session against others of the same type "
            "(e.g. acclimation vs acclimation, testing vs testing)."
        )
        self._session_quality_btn.clicked.connect(self._open_session_quality_dialog)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Behavior:"))
        top_row.addWidget(self._behavior, 1)
        top_row.addWidget(self._per_behavior_thresholds_btn)
        top_row.addWidget(self._refresh_btn)
        top_row.addWidget(self._session_quality_btn)

        self._metrics_table = QTableWidget(0, 9)
        self._metrics_table.setHorizontalHeaderLabels(
            [
                "Subject",
                "N Bouts",
            "Time Spent (s)",
                "Frame F1",
                "Frame Precision",
                "Frame Recall",
                "Bout IoU",
                "Onset Err",
                "Offset Err",
            ]
        )
        self._metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._metrics_table.setVisible(False)

        self._bouts_table = QTableWidget(0, 2)
        self._bouts_table.setHorizontalHeaderLabels(["Start Frame", "End Frame"])
        self._bouts_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._bouts_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._bouts_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        session_row = QHBoxLayout()
        session_row.addWidget(QLabel("Subject:"))
        session_row.addWidget(self._session, 1)

        preview_row = QHBoxLayout()
        self._preview_btn = QPushButton("Seek Bout")
        self._preview_btn.setToolTip("Jump the video player to the start of the selected bout.")
        self._preview_btn.clicked.connect(self._preview_selected_bout)
        self._mark_fp_btn = QPushButton("Add False Positive Interval")
        self._mark_fp_btn.clicked.connect(self._mark_selected_bout_false_positive)
        self._mark_fn_btn = QPushButton("Add False Negative Interval")
        self._mark_fn_btn.clicked.connect(self._add_false_negative_interval)
        self._send_bouts_btn = QPushButton("Send All Bouts to Clip Review")
        self._send_bouts_btn.setToolTip(
            "Convert every detected bout (across all behaviors and sessions) into "
            "fixed-length clip windows and send them to the Clip Extraction tab for review."
        )
        self._send_bouts_btn.clicked.connect(self._send_bouts_to_clip_review)
        self._send_subject_bouts_btn = QPushButton("Send Current Behavior/Subject Bouts")
        self._send_subject_bouts_btn.setToolTip(
            "Send bouts for the currently selected behavior and current subject to Clip Review "
            "without clearing any clips already there — so you can accumulate clips from "
            "different subjects one at a time."
        )
        self._send_subject_bouts_btn.clicked.connect(self._send_current_subject_behavior_bouts_to_clip_review)
        preview_row.addWidget(self._preview_btn)
        preview_row.addWidget(self._mark_fp_btn)
        preview_row.addWidget(self._mark_fn_btn)
        preview_row.addWidget(self._send_bouts_btn)
        preview_row.addWidget(self._send_subject_bouts_btn)
        preview_row.addStretch(1)

        self._player = CandidateVideoPlayer(self)
        self._player.setMinimumHeight(260)

        preview_group = QGroupBox("Session Video Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addLayout(preview_row)
        preview_layout.addWidget(self._player)

        # Row 1: behavior selector + read-only settings display
        trace_row1 = QHBoxLayout()
        trace_row1.addWidget(QLabel("Behavior:"))
        self._trace_behavior = QComboBox()
        self._trace_behavior.addItem("All behaviors", userData="__all__")
        self._trace_behavior.currentIndexChanged.connect(self._on_trace_behavior_changed)
        trace_row1.addWidget(self._trace_behavior)
        trace_row1.addWidget(self._trace_settings_label, 1)

        # Row 2: probability readout + FP/FN toggle buttons
        trace_row2 = QHBoxLayout()
        self._selected_probability = QLabel("Selected frame probability: --")
        trace_row2.addWidget(self._selected_probability)
        trace_row2.addStretch(1)

        self._fp_flag_btn = QPushButton("Flag FP (click bout)")
        self._fp_flag_btn.setCheckable(True)
        self._fp_flag_btn.setToolTip(
            "Toggle FP flagging mode. When active, left-click any detected bout "
            "on the trace to instantly flag it as a false positive (hard negative). "
            "Right-click a red region to unflag."
        )
        self._fp_flag_btn.toggled.connect(self._toggle_fp_mode)
        trace_row2.addWidget(self._fp_flag_btn)

        self._fn_flag_btn = QPushButton("Flag FN (drag range)")
        self._fn_flag_btn.setCheckable(True)
        self._fn_flag_btn.setToolTip(
            "Toggle FN flagging mode. When active, click and drag on the trace "
            "to highlight a frame range and flag it as a false negative. "
            "Right-click a blue region to unflag."
        )
        self._fn_flag_btn.toggled.connect(self._toggle_fn_mode)
        trace_row2.addWidget(self._fn_flag_btn)

        self._staged_count_label = QLabel("")
        trace_row2.addWidget(self._staged_count_label)

        self._commit_flags_btn = QPushButton("Commit Flags")
        self._commit_flags_btn.setToolTip("Persist all staged FP/FN flags. Until committed, flags are only visual.")
        self._commit_flags_btn.clicked.connect(self._commit_staged_flags)
        self._commit_flags_btn.setEnabled(False)
        trace_row2.addWidget(self._commit_flags_btn)

        self._clear_staged_btn = QPushButton("Clear Staged")
        self._clear_staged_btn.setToolTip("Discard all staged (uncommitted) flags.")
        self._clear_staged_btn.clicked.connect(self._clear_staged_flags)
        self._clear_staged_btn.setEnabled(False)
        trace_row2.addWidget(self._clear_staged_btn)

        # -- State for interactive FP/FN flagging --
        self._fp_flag_active = False
        self._fn_flag_active = False
        self._fp_drag_start: float | None = None  # rubber-band drag for FP mode
        self._fp_drag_rect = None
        self._fn_drag_start: float | None = None
        self._fn_drag_rect = None  # matplotlib Rectangle patch for rubber-band
        self._right_drag_start: float | None = None  # right-click drag to bulk-remove
        self._right_drag_rect = None
        self._staged_fp_by_key: dict[tuple[str, str], list[tuple[int, int]]] = {}
        self._staged_fn_by_key: dict[tuple[str, str], list[tuple[int, int]]] = {}
        self._current_bouts: list[tuple[int, int]] = []
        self._current_fp_intervals: list[tuple[int, int]] = []
        self._current_fn_intervals: list[tuple[int, int]] = []

        trace_group = QGroupBox("Probability Trace (Zoom + Click To Seek)")
        trace_layout = QVBoxLayout(trace_group)
        trace_layout.addLayout(trace_row1)
        trace_layout.addLayout(trace_row2)
        self._trace_canvas = None
        self._trace_axes = None
        self._trace_click_cid = None
        self._trace_toolbar = None
        self._trace_placeholder = QLabel("Probability trace will appear after loading inference artifacts.")
        self._trace_placeholder.setWordWrap(True)
        trace_layout.addWidget(self._trace_placeholder)
        # Defer matplotlib canvas creation until first use so that Qt DPI
        # information is available and the canvas does not trigger
        # QFont::setPointSize <= 0 warnings during app startup.
        self._trace_layout: QVBoxLayout | None = trace_layout

        lower_split = QSplitter()
        lower_split.setChildrenCollapsible(False)
        lower_split.addWidget(preview_group)
        lower_split.addWidget(trace_group)
        lower_split.setSizes([500, 800])

        # Keep the bouts table instantiated (hidden) so that _selected_bout_interval
        # and _preview_selected_bout still function when called programmatically.
        self._bouts_table.hide()

        review_group = QGroupBox("Temporal Results")
        review_layout = QVBoxLayout(review_group)
        review_layout.addLayout(session_row)
        review_layout.addWidget(lower_split, 1)

        root = QVBoxLayout(self)
        root.addLayout(top_row)
        root.addWidget(self._status)
        root.addWidget(review_group, 1)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._trace_axes is None and self._trace_layout is not None:
            QTimer.singleShot(0, self._refresh_probability_plot)

    def _can_init_trace_plot(self) -> bool:
        if not self.isVisible():
            return False
        screen = self.screen()
        if screen is None:
            return False
        try:
            return float(screen.logicalDotsPerInch()) > 0.0
        except Exception:
            return False

    def _schedule_trace_plot_retry(self) -> None:
        if self._trace_init_scheduled:
            return
        self._trace_init_scheduled = True

        def _retry() -> None:
            self._trace_init_scheduled = False
            self._refresh_probability_plot()

        QTimer.singleShot(50, _retry)

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._manager = ProjectManager(project_root)
        # Defer I/O to avoid blocking the tab switch.
        from PySide6.QtCore import QTimer  # noqa: PLC0415
        QTimer.singleShot(0, lambda: self._deferred_project_init(project_root))

    def _deferred_project_init(self, project_root: Path) -> None:
        if self._project_root != project_root:
            return
        self._behaviors.set_project(project_root)
        self._review_service.set_project(project_root)
        self._subject_by_session = self._build_subject_map()
        self._review_settings = self._load_settings()
        self._refresh_behaviors()
        self._apply_behavior_settings_to_controls()
        self._refresh()

    def _build_subject_map(self) -> dict[str, str]:
        if self._project_root is None:
            return {}
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return {}
        video_by_id = {v.asset_id: v for v in manifest.videos}
        out: dict[str, str] = {}
        for session in manifest.linked_sessions:
            sid = str(session.session_id)
            subject = (session.subject_id or "").strip()
            if not subject:
                video = video_by_id.get(session.video_asset_id)
                subject = (video.subject_id or "").strip() if video else ""
            out[sid] = subject or sid
        return out

    def _subject_display(self, session_id: str, used: set[str] | None = None) -> str:
        subject = self._subject_by_session.get(session_id, session_id)
        if used is not None and subject in used:
            return f"{subject} ({session_id})"
        return subject

    def _refresh_behaviors(self) -> None:
        current = str(self._behavior.currentData() or "target_behavior")
        self._behavior.blockSignals(True)
        self._behavior.clear()
        self._behavior.addItem("All reviewed positives", userData="target_behavior")
        for behavior in self._behaviors.behaviors:
            self._behavior.addItem(behavior.name, userData=str(behavior.behavior_id))
        idx = self._behavior.findData(current)
        if idx < 0 and self._behaviors.behaviors:
            idx = self._behavior.findData(str(self._behaviors.behaviors[0].behavior_id))
        self._behavior.setCurrentIndex(idx if idx >= 0 else 0)
        self._behavior.blockSignals(False)

    def _behavior_id_from_trace_col(self, col: str) -> str | None:
        """Reverse-map a prob_<token> column to its behavior_id, or None."""
        if not str(col).startswith("prob_"):
            return None
        token = str(col).removeprefix("prob_")
        for behavior in self._behaviors.behaviors:
            bid = str(behavior.behavior_id or "").strip()
            if not bid:
                continue
            if token == bid or token == self._safe_name(bid):
                return bid
        return None

    def _on_trace_behavior_changed(self, _index: int) -> None:
        """Update the settings display when the trace-view behavior selector changes."""
        col = str(self._trace_behavior.currentData() or "__all__")
        if col == "__all__":
            self._trace_settings_label.setText("")
        else:
            bid = self._behavior_id_from_trace_col(col)
            if bid:
                cfg = self._settings_for_behavior(bid)
                self._update_trace_settings_label(cfg)
            else:
                self._trace_settings_label.setText("")
        self._refresh_probability_plot()

    # ── Interactive FP / FN flagging modes ───────────────────────────────

    def _set_trace_navigation_enabled(self, enabled: bool) -> None:
        toolbar = self._trace_toolbar
        canvas = self._trace_canvas
        if toolbar is not None and not enabled:
            mode = getattr(toolbar, "mode", None)
            mode_name = getattr(mode, "name", str(mode)).upper()
            if "PAN" in mode_name:
                toolbar.pan()
            elif "ZOOM" in mode_name:
                toolbar.zoom()
        if toolbar is not None:
            toolbar.setEnabled(enabled)
        if self._trace_axes is not None:
            self._trace_axes.set_navigate(enabled)
        if canvas is not None:
            canvas.draw_idle()

    def _update_trace_navigation_state(self) -> None:
        self._set_trace_navigation_enabled(not (self._fp_flag_active or self._fn_flag_active))

    def _toggle_fp_mode(self, checked: bool) -> None:
        self._fp_flag_active = checked
        if checked:
            self._fn_flag_btn.setChecked(False)
            self._fp_flag_btn.setStyleSheet("background-color: #EF9A9A; font-weight: bold;")
            self._status.setText("FP mode — click bouts to stage as false positive. Right-click staged/committed to remove. Commit when done.")
        else:
            self._fp_flag_btn.setStyleSheet("")
            if not self._fn_flag_active:
                self._status.setText("")
        self._update_trace_navigation_state()

    def _toggle_fn_mode(self, checked: bool) -> None:
        self._fn_flag_active = checked
        if checked:
            self._fp_flag_btn.setChecked(False)
            self._fn_flag_btn.setStyleSheet("background-color: #90CAF9; font-weight: bold;")
            self._status.setText("FN mode — drag on trace to stage missed regions. Right-click staged/committed to remove. Commit when done.")
        else:
            self._fn_flag_btn.setStyleSheet("")
            self._fn_drag_start = None
            if self._fn_drag_rect is not None:
                self._fn_drag_rect.remove()
                self._fn_drag_rect = None
                if self._trace_canvas:
                    self._trace_canvas.draw_idle()
            if not self._fp_flag_active:
                self._status.setText("")
        self._update_trace_navigation_state()

    def _find_bout_at_frame(self, frame_idx: int) -> tuple[int, int] | None:
        """Return the (start, end) bout interval that contains *frame_idx*, or None."""
        for s, e in self._current_bouts:
            if s <= frame_idx <= e:
                return s, e
        return None

    def _find_flagged_interval_at_frame(self, frame_idx: int) -> tuple[str, tuple[int, int], bool] | None:
        """Return ('false_positive'|'false_negative', (start, end), is_staged) for a flagged region at *frame_idx*.

        Checks staged (uncommitted) intervals first, then committed ones.
        """
        staged_fp, staged_fn = self._current_staged_lists()
        for s, e in staged_fp:
            if s <= frame_idx <= e:
                return "false_positive", (s, e), True
        for s, e in staged_fn:
            if s <= frame_idx <= e:
                return "false_negative", (s, e), True
        for s, e in self._current_fp_intervals:
            if s <= frame_idx <= e:
                return "false_positive", (s, e), False
        for s, e in self._current_fn_intervals:
            if s <= frame_idx <= e:
                return "false_negative", (s, e), False
        return None

    def _stage_fp(self, start: int, end: int) -> None:
        """Stage a bout as FP (visual only — not persisted until commit)."""
        staged_fp, _ = self._current_staged_lists()
        if (start, end) not in staged_fp:
            staged_fp.append((start, end))
        self._update_staged_ui()
        self._draw_staged_overlays()

    def _stage_fn(self, start: int, end: int) -> None:
        """Stage a range as FN (visual only — not persisted until commit)."""
        _, staged_fn = self._current_staged_lists()
        if (start, end) not in staged_fn:
            staged_fn.append((start, end))
        self._update_staged_ui()
        self._draw_staged_overlays()

    def _unstage(self, feedback_type: str, start: int, end: int) -> None:
        """Remove a staged interval."""
        staged_fp, staged_fn = self._current_staged_lists()
        target = staged_fp if feedback_type == "false_positive" else staged_fn
        try:
            target.remove((start, end))
        except ValueError:
            pass
        self._update_staged_ui()
        self._draw_staged_overlays()

    def _stage_key(self, concept_id: str, session_id: str) -> tuple[str, str]:
        return (str(concept_id).strip(), str(session_id).strip())

    def _current_stage_key(self) -> tuple[str, str] | None:
        sid = str(self._session.currentData() or "").strip()
        if not sid:
            return None
        return self._stage_key(self._resolve_trace_concept_id(), sid)

    def _current_staged_lists(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        key = self._current_stage_key()
        if key is None:
            return [], []
        staged_fp = self._staged_fp_by_key.setdefault(key, [])
        staged_fn = self._staged_fn_by_key.setdefault(key, [])
        return staged_fp, staged_fn

    def _update_staged_ui(self) -> None:
        staged_fp, staged_fn = self._current_staged_lists()
        active_total = len(staged_fp) + len(staged_fn)
        all_fp = sum(len(v) for v in self._staged_fp_by_key.values())
        all_fn = sum(len(v) for v in self._staged_fn_by_key.values())
        all_total = all_fp + all_fn

        if active_total:
            parts: list[str] = []
            if staged_fp:
                parts.append(f"{len(staged_fp)} FP")
            if staged_fn:
                parts.append(f"{len(staged_fn)} FN")
            if all_total > active_total:
                self._staged_count_label.setText(
                    f"Staged here: {', '.join(parts)}  |  All sessions pending: {all_total}"
                )
            else:
                self._staged_count_label.setText(f"Staged: {', '.join(parts)}")
        elif all_total:
            self._staged_count_label.setText(f"No staged flags for this subject. All sessions pending: {all_total}")
        else:
            self._staged_count_label.setText("")
        self._commit_flags_btn.setEnabled(all_total > 0)
        self._clear_staged_btn.setEnabled(active_total > 0)

    def _draw_staged_overlays(self) -> None:
        """Redraw only the staged (uncommitted) overlays on the existing trace."""
        if self._trace_axes is None or self._trace_canvas is None:
            return
        # Remove previous staged patches (tagged with '_staged')
        for patch in list(self._trace_axes.patches):
            if getattr(patch, '_staged', False):
                patch.remove()
        import matplotlib.patches as mpatches  # noqa: PLC0415
        staged_fp, staged_fn = self._current_staged_lists()
        for s, e in staged_fp:
            rect = mpatches.Rectangle(
                (s, 0), e - s, 1.0,
                linewidth=2, edgecolor="#C62828", facecolor="#EF5350", alpha=0.25,
                linestyle="--",
            )
            rect._staged = True  # type: ignore[attr-defined]
            self._trace_axes.add_patch(rect)
        for s, e in staged_fn:
            rect = mpatches.Rectangle(
                (s, 0), e - s, 1.0,
                linewidth=2, edgecolor="#1565C0", facecolor="#42A5F5", alpha=0.25,
                linestyle="--",
            )
            rect._staged = True  # type: ignore[attr-defined]
            self._trace_axes.add_patch(rect)
        self._trace_canvas.draw_idle()

    def _commit_staged_flags(self) -> None:
        """Persist all staged FP/FN flags."""
        manager = self._manager
        if manager is None:
            self._status.setText("Cannot commit: no project loaded.")
            return
        total_before = sum(len(v) for v in self._staged_fp_by_key.values()) + sum(len(v) for v in self._staged_fn_by_key.values())
        if total_before <= 0:
            self._status.setText("Nothing staged to commit.")
            return

        n_fp = 0
        n_fn = 0
        try:
            for (concept_id, sid), intervals in list(self._staged_fp_by_key.items()):
                for start, end in intervals:
                    manager.add_temporal_feedback_interval(
                        concept_id=concept_id, session_id=sid,
                        start_frame=start, end_frame=end, feedback_type="false_positive",
                    )
                    self._write_temporal_review_decisions(
                        session_id=sid, start=start, end=end,
                        review_label="no_behavior", concept_id=concept_id,
                        decision_type=ReviewDecisionType.REJECT, behavior_label=concept_id,
                    )
                    n_fp += 1
            for (concept_id, sid), intervals in list(self._staged_fn_by_key.items()):
                for start, end in intervals:
                    manager.add_temporal_feedback_interval(
                        concept_id=concept_id, session_id=sid,
                        start_frame=start, end_frame=end, feedback_type="false_negative",
                    )
                    self._write_temporal_review_decisions(
                        session_id=sid, start=start, end=end,
                        review_label=concept_id, concept_id=concept_id,
                    )
                    n_fn += 1
        except Exception as exc:
            import traceback
            err = traceback.format_exc()
            self._status.setText(f"Commit failed: {exc}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Commit Error", f"Failed to commit flags:\n\n{err}")
            return

        self._staged_fp_by_key.clear()
        self._staged_fn_by_key.clear()
        self._update_staged_ui()
        self._status.setText(f"Committed {n_fp} FP + {n_fn} FN flags across staged subjects.")
        self._refresh_probability_plot()

    def _clear_staged_flags(self) -> None:
        """Discard all staged (uncommitted) flags."""
        key = self._current_stage_key()
        if key is not None:
            self._staged_fp_by_key.pop(key, None)
            self._staged_fn_by_key.pop(key, None)
        self._update_staged_ui()
        self._draw_staged_overlays()
        self._status.setText("Cleared all staged flags.")

    def _unflag_interval(self, feedback_type: str, start: int, end: int) -> None:
        """Remove a previously flagged FP or FN interval."""
        manager = self._manager
        if manager is None:
            return
        sid = str(self._session.currentData() or "").strip()
        if not sid:
            return
        concept_id = self._resolve_trace_concept_id()
        feedback = manager.remove_temporal_feedback_interval(
            concept_id=concept_id,
            session_id=sid,
            start_frame=start,
            end_frame=end,
            feedback_type=feedback_type,
        )
        label = "FP" if feedback_type == "false_positive" else "FN"
        self._status.setText(
            f"Unflagged {label}: frames {start}–{end}.  "
            f"FP={feedback.get('n_false_positive_intervals', 0)}, "
            f"FN={feedback.get('n_false_negative_intervals', 0)}."
        )
        self._refresh_probability_plot()

    def _resolve_trace_concept_id(self) -> str:
        """Get the concept_id for the currently viewed trace behavior."""
        col = str(self._trace_behavior.currentData() or "__all__")
        if col and col != "__all__":
            return col.removeprefix("prob_")
        return self._concept_id()

    def _on_behavior_changed(self, _index: int) -> None:
        self._apply_behavior_settings_to_controls()
        self._refresh()

    def _concept_id(self) -> str:
        return str(self._behavior.currentData() or "target_behavior").strip() or "target_behavior"

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
        return safe or "target_behavior"

    @staticmethod
    def _is_no_behavior_token(token: str) -> bool:
        norm = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(token or "").strip()).strip("_")
        return norm in {"no_behavior", "no_behaviour", "nobehavior", "nobehaviour"}

    @staticmethod
    def _fmt(value) -> str:
        try:
            if value is None:
                return ""
            num = float(value)
            if pd.isna(num):
                return ""
            return f"{num:.3f}"
        except Exception:
            return str(value)

    def _settings_path(self) -> Path | None:
        if self._project_root is None:
            return None
        return self._project_root / "config" / "temporal_review_settings.json"

    @staticmethod
    def _default_review_settings() -> dict[str, Any]:
        return {
            "onset_threshold": 0.65,
            "min_bout_duration_frames": 8,
            "merge_gap_frames": 4,
        }

    def _load_settings(self) -> dict[str, dict[str, Any]]:
        path = self._settings_path()
        default = self._default_review_settings()
        if path is None or not path.exists():
            return {"__all__": dict(default)}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"__all__": dict(default)}

        out: dict[str, dict[str, Any]] = {}
        out["__all__"] = {**default, **dict(raw.get("__all__", {}) or {})}
        by_behavior = dict(raw.get("by_behavior", {}) or {})
        for behavior_id, values in by_behavior.items():
            if not isinstance(values, dict):
                continue
            out[str(behavior_id)] = {**out["__all__"], **values}
        return out

    def _save_settings(self) -> None:
        path = self._settings_path()
        if path is None:
            return
        payload = {
            "__all__": dict(self._review_settings.get("__all__", self._default_review_settings())),
            "by_behavior": {
                key: value
                for key, value in self._review_settings.items()
                if key != "__all__"
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _settings_for_behavior(self, behavior_id: str) -> dict[str, Any]:
        global_defaults = dict(self._review_settings.get("__all__", self._default_review_settings()))
        return {**global_defaults, **dict(self._review_settings.get(behavior_id, {}) or {})}

    def _load_run_postprocess_config_for_behavior(self, behavior_id: str) -> dict[str, Any]:
        """Read the threshold/bout settings actually used in the last postprocess run."""
        if self._project_root is None:
            return {}
        token = self._safe_name(behavior_id)
        latest_path = (
            self._project_root / "derived" / "temporal_refinement" / token / "latest.json"
        )
        if not latest_path.exists():
            return {}
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            post_dir = str(latest.get("postprocess_dir", "") or "").strip()
            if not post_dir:
                return {}
            manifest_path = Path(post_dir) / "postprocess_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return dict(manifest.get("postprocess", {}) or {})
        except Exception:
            return {}

    def _update_trace_settings_label(self, cfg: dict) -> None:
        """Refresh the read-only settings display from a config dict."""
        onset = float(cfg.get("onset_threshold", 0.65))
        min_bout = int(cfg.get("min_bout_duration_frames", 8))
        merge_gap = int(cfg.get("merge_gap_frames", 4))
        self._trace_settings_label.setText(
            f"Threshold: {onset:.3f}  |  Min Bout: {min_bout} frames  |  Merge Gap: {merge_gap} frames"
        )

    def _apply_behavior_settings_to_controls(self) -> None:
        """Update the read-only settings display from saved per-behavior settings."""
        col = str(self._trace_behavior.currentData() or "__all__")
        if col == "__all__":
            self._trace_settings_label.setText("")
            return
        bid = self._behavior_id_from_trace_col(col) or self._concept_id()
        cfg = self._settings_for_behavior(bid)
        self._update_trace_settings_label(cfg)

    def _collect_control_settings(self) -> dict[str, Any]:
        """Return current settings for the active behavior, read from saved review settings."""
        col = str(self._trace_behavior.currentData() or "__all__")
        bid = (self._behavior_id_from_trace_col(col) if col != "__all__" else None) or self._concept_id()
        cfg = self._settings_for_behavior(bid)
        return {
            "onset_threshold": float(cfg.get("onset_threshold", 0.65)),
            "min_bout_duration_frames": int(cfg.get("min_bout_duration_frames", 8)),
            "merge_gap_frames": int(cfg.get("merge_gap_frames", 4)),
        }

    def _persist_current_behavior_settings(self) -> None:
        self._review_settings[self._concept_id()] = self._collect_control_settings()
        self._save_settings()

    def _persist_global_settings(self) -> None:
        payload = self._collect_control_settings()
        self._review_settings["__all__"] = dict(payload)
        for behavior in self._behaviors.behaviors:
            behavior_id = str(behavior.behavior_id)
            self._review_settings[behavior_id] = dict(payload)
        self._review_settings["target_behavior"] = dict(payload)
        self._save_settings()

    def _review_config(self) -> TemporalRefinementConfig:
        cfg = self._collect_control_settings()
        onset = float(cfg["onset_threshold"])
        return TemporalRefinementConfig(
            onset_threshold=onset,
            min_bout_duration_frames=int(cfg["min_bout_duration_frames"]),
            merge_gap_frames=int(cfg["merge_gap_frames"]),
        )

    def _latest_path(self) -> Path | None:
        if self._project_root is None:
            return None
        return (
            self._project_root
            / "derived"
            / "temporal_refinement"
            / self._safe_name(self._concept_id())
            / "latest.json"
        )

    def _set_refresh_busy(self, busy: bool) -> None:
        self._is_refreshing = busy
        self._refresh_btn.setEnabled(not busy)
        self._per_behavior_thresholds_btn.setEnabled(not busy)
        self._behavior.setEnabled(not busy)

    def _training_settings_path(self) -> Path | None:
        if self._project_root is None:
            return None
        return self._project_root / "config" / "temporal_refinement_settings.json"

    def _load_training_config_for_current_behavior(self) -> TemporalRefinementConfig:
        path = self._training_settings_path()
        behavior_id = self._concept_id()
        if path is None or not path.exists():
            return TemporalRefinementConfig()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return TemporalRefinementConfig()

        import dataclasses
        defaults = dict(raw.get("__all__", {}) or {})
        by_behavior = dict(raw.get("by_behavior", {}) or {})
        merged = {**defaults, **dict(by_behavior.get(behavior_id, {}) or {})}
        valid_fields = {f.name for f in dataclasses.fields(TemporalRefinementConfig)}
        merged = {k: v for k, v in merged.items() if k in valid_fields}
        return TemporalRefinementConfig(**merged)

    def _inject_fp_fn_as_reviewer_labels(self, concept_id: str, progress_cb=None) -> int:
        """Tile FP/FN feedback intervals into non-overlapping AL-sized windows and write
        them to the reviewer label store.

        False-positive windows are labelled ``"no_behavior"`` (hard negative).
        False-negative windows are labelled with *concept_id* (positive).

        Returns the number of new label records written.
        """
        from abel.models.schemas import ReviewerLabelRecord

        project_root = self._project_root
        manager = self._manager
        if project_root is None or manager is None:
            return 0

        feedback = manager.load_temporal_feedback(concept_id)
        fp_by_session: dict = dict(feedback.get("false_positive_intervals_by_session", {}) or {})
        fn_by_session: dict = dict(feedback.get("false_negative_intervals_by_session", {}) or {})
        if not fp_by_session and not fn_by_session:
            return 0

        # Resolve the window size in use by the active learning pipeline.
        window_size = self._resolve_window_size()

        # Load the existing temporal-feedback entries so we don't duplicate them.
        label_path = project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        existing_ids: set[str] = set()
        if label_path.exists():
            try:
                _df = pd.read_parquet(label_path, columns=["segment_id", "reviewer_id"])
                mask = _df["reviewer_id"].astype(str) == "temporal_feedback"
                existing_ids = set(_df.loc[mask, "segment_id"].astype(str).tolist())
            except Exception:
                pass

        new_records: list[dict] = []

        def _tile_interval(session_id: str, start: int, end: int, review_label: str) -> None:
            for tile_start in range(int(start), int(end), window_size):
                tile_end = tile_start + window_size - 1
                seg_id = f"seg_feedback_{session_id}_{tile_start}_{tile_end}"
                if seg_id in existing_ids:
                    continue
                rec = ReviewerLabelRecord(
                    segment_id=seg_id,
                    review_label=review_label,
                    reviewer_id="temporal_feedback",
                    confidence=1.0,
                    notes=f"auto:{review_label}:{concept_id}",
                )
                new_records.append(rec.model_dump(mode="json"))
                existing_ids.add(seg_id)

        for session_id, intervals in fp_by_session.items():
            for (interval_start, interval_end) in intervals:
                # FP tiles: parquet label = "no_behavior" (not a positive example
                # of the concept for AL training).
                _tile_interval(session_id, interval_start, interval_end, "no_behavior")

        for session_id, intervals in fn_by_session.items():
            for (interval_start, interval_end) in intervals:
                _tile_interval(session_id, interval_start, interval_end, str(concept_id))

        if not new_records:
            return 0

        # Batch-write all new records in a single parquet round-trip.
        label_path.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(new_records)
        if label_path.exists():
            existing_full = pd.read_parquet(label_path)
            merged = pd.concat([existing_full, new_df], ignore_index=True)
        else:
            merged = new_df
        merged.to_parquet(label_path, index=False)

        return len(new_records)

    def _run_refresh(self) -> None:
        if self._manager is None:
            QMessageBox.warning(self, "Temporal Review", "Open a project first.")
            return
        if self._is_refreshing:
            return

        self._persist_current_behavior_settings()
        self._set_refresh_busy(True)
        self._status.setText("Applying review settings and regenerating bout outputs...")

        worker = TaskWorker(self._refresh_task)
        worker.kwargs["progress_cb"] = worker.signals.line_emitted.emit
        worker.signals.line_emitted.connect(self._on_refresh_progress)
        worker.signals.finished.connect(self._on_refresh_finished)
        worker.signals.failed.connect(self._on_refresh_failed)
        self._pool.start(worker)

    def _refresh_task(self, progress_cb=None) -> dict[str, Any]:
        manager = self._manager
        if manager is None:
            raise ValueError("No project loaded")
        return manager.run_temporal_refinement_postprocess(
            concept_id=self._concept_id(),
            config=self._review_config(),
            progress_cb=progress_cb,
        )

    def _on_refresh_progress(self, line: str) -> None:
        text = str(line or "").strip()
        if text:
            self._status.setText(text)

    def _on_refresh_finished(self, _result: dict[str, Any]) -> None:
        self._set_refresh_busy(False)
        self._status.setText("Temporal review outputs refreshed with current settings.")
        self._refresh()

    def _on_refresh_failed(self, traceback_text: str) -> None:
        self._set_refresh_busy(False)
        self._status.setText("Failed to refresh temporal review outputs.")
        QMessageBox.warning(self, "Temporal Review", format_task_error(traceback_text))

    def _refresh(self) -> None:
        self._metrics_table.setRowCount(0)
        self._session.blockSignals(True)
        self._session.clear()
        self._session.blockSignals(False)
        self._bouts_table.setRowCount(0)
        self._player.close_clip()
        self._loaded_session_video_id = None
        self._trace_paths = {}
        self._trace_probability_lookup = {}
        self._competition_models = {}
        self._competition_excluded = []
        self._bout_paths = {}
        self._bout_paths_by_behavior = {}
        self._session_rows = []
        self._trace_behavior.blockSignals(True)
        self._trace_behavior.clear()
        self._trace_behavior.addItem("All behaviors", userData="__all__")
        self._trace_behavior.blockSignals(False)
        self._set_selected_probability(None)
        feedback = self._load_feedback_summary()

        latest_path = self._latest_path()
        if latest_path is None or not latest_path.exists():
            self._status.setText("No temporal refinement outputs yet for selected behavior.")
            self._refresh_probability_plot()
            return

        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._status.setText("Could not read temporal refinement outputs.")
            self._refresh_probability_plot()
            return

        post_dir = str(latest.get("postprocess_dir", "")).strip()
        inference_dir = str(latest.get("inference_dir", "")).strip()
        cfg = self._collect_control_settings()
        lines = [
            f"Behavior: {self._behavior.currentText()}",
            f"Training: {latest.get('training_dir', 'not available')}",
            f"Inference: {inference_dir or 'not available'}",
            f"Postprocess: {post_dir or 'not available'}",
            (
                "Feedback intervals: "
                f"false_positive={feedback.get('n_false_positive_intervals', 0)}, "
                f"false_negative={feedback.get('n_false_negative_intervals', 0)}"
            ),
            (
                "Review settings: "
                f"onset={cfg['onset_threshold']:.2f}, "
                f"min_bout={cfg['min_bout_duration_frames']}f, "
                f"merge_gap={cfg['merge_gap_frames']}f"
            ),
        ]
        if inference_dir:
            self._load_inference_trace_paths(Path(inference_dir))
        if self._competition_models:
            lines.append(f"Competition models: {len(self._competition_models)} behavior(s)")
        if self._competition_excluded:
            lines.append("Excluded from competition: " + ", ".join(self._competition_excluded))
        self._status.setText("Temporal outputs loaded." if post_dir else "No bout outputs yet.")

        if not post_dir:
            self._status.setText("No bout outputs yet. Click Refresh (Apply Settings) to generate them.")
            self._refresh_probability_plot()
            return

        self._status.setText("Temporal outputs loaded.")
        self._load_postprocess(Path(post_dir))

    def _load_feedback_summary(self) -> dict[str, Any]:
        if self._manager is None:
            return {}
        try:
            return self._manager.load_temporal_feedback(concept_id=self._concept_id())
        except Exception:
            return {}

    def _selected_bout_interval(self) -> tuple[int, int] | None:
        if self._bouts_table.rowCount() <= 0:
            return None
        selected = self._bouts_table.selectionModel().selectedRows()
        row_idx = int(selected[0].row()) if selected else 0
        start_item = self._bouts_table.item(row_idx, 0)
        end_item = self._bouts_table.item(row_idx, 1)
        if start_item is None or end_item is None:
            return None
        try:
            s = int(start_item.text())
            e = int(end_item.text())
            if e < s:
                s, e = e, s
            return s, e
        except Exception:
            return None

    def _mark_selected_bout_false_positive(self) -> None:
        manager = self._manager
        if manager is None:
            QMessageBox.warning(self, "Temporal Review", "Open a project first.")
            return
        sid = str(self._session.currentData() or "").strip()
        if not sid:
            QMessageBox.information(self, "Temporal Review", "Choose a subject/session first.")
            return
        interval = self._selected_bout_interval()
        if interval is not None:
            default_start, default_end = interval
        else:
            center = int(self._player.current_frame)
            default_start = max(0, center - 15)
            default_end = max(default_start, center + 15)

        dlg = QDialog(self)
        dlg.setWindowTitle("Add False Positive Interval")
        start_spin = QSpinBox(dlg)
        start_spin.setRange(0, 10_000_000)
        start_spin.setValue(int(default_start))
        end_spin = QSpinBox(dlg)
        end_spin.setRange(0, 10_000_000)
        end_spin.setValue(int(default_end))

        # Behavior selection: what is the clip *actually* showing?
        # Default = concept-specific hard negative (the model fired but this
        # behavior did not occur here; not the same as "no behavior at all").
        behavior_combo = QComboBox(dlg)
        _trace_data = str(self._trace_behavior.currentData() or "").strip()
        if _trace_data and _trace_data != "__all__":
            # Column names are "prob_{behavior_id}" — strip the prob_ prefix to get
            # the actual behavior ID used as concept_id.
            concept_id_fp = _trace_data.removeprefix("prob_")
            concept_name_fp = self._trace_behavior.currentText().strip() or concept_id_fp
        else:
            concept_id_fp = self._concept_id()
            concept_name_fp = self._behavior.currentText().strip() or concept_id_fp
        behavior_combo.addItem(
            f"Hard negative (\u2018{concept_name_fp}\u2019 did not occur)",
            userData=concept_id_fp,  # REJECT will be stored for this behavior
        )
        behavior_combo.addItem("Hard negative (no behavior at all)", userData="no_behavior")
        for b in self._behaviors.behaviors:
            behavior_combo.addItem(f"Correct behavior is: {b.name}", userData=str(b.behavior_id))

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Start frame:"))
        frame_row.addWidget(start_spin)
        frame_row.addWidget(QLabel("End frame:"))
        frame_row.addWidget(end_spin)

        beh_row = QHBoxLayout()
        beh_row.addWidget(QLabel("Clip actually shows:"))
        beh_row.addWidget(behavior_combo, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout(dlg)
        layout.addLayout(frame_row)
        layout.addLayout(beh_row)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        start = int(start_spin.value())
        end = int(end_spin.value())
        selected = str(behavior_combo.currentData() or "no_behavior")

        # Determine decision type and labels:
        # - concept selected (default): REJECT for that behavior specifically.
        #   Parquet label = "no_behavior" so AL training doesn't treat it as a positive.
        # - "no_behavior": ACCEPT with no_behavior — universal hard negative.
        # - other behavior: ACCEPT for that behavior — was actually something else.
        if selected == concept_id_fp:
            fp_decision = ReviewDecisionType.REJECT
            fp_behavior_label = concept_id_fp   # decisions JSON: REJECT for Dig
            fp_parquet_label = "no_behavior"    # parquet: not a positive example
        else:
            fp_decision = ReviewDecisionType.ACCEPT
            fp_behavior_label = selected        # "no_behavior" or other behavior id
            fp_parquet_label = selected

        feedback = manager.add_temporal_feedback_interval(
            concept_id=concept_id_fp,
            session_id=sid,
            start_frame=start,
            end_frame=end,
            feedback_type="false_positive",
        )
        # Write a finalized review decision so this interval is treated as
        # already-reviewed — no additional approval needed in the review tab.
        self._write_temporal_review_decisions(
            session_id=sid, start=start, end=end,
            review_label=fp_parquet_label,
            concept_id=concept_id_fp,
            decision_type=fp_decision,
            behavior_label=fp_behavior_label,
        )
        self._status.setText(
            f"Saved false-positive interval {start}-{end} (‘{behavior_combo.currentText()}’). "
            f"FP={feedback.get('n_false_positive_intervals', 0)}, FN={feedback.get('n_false_negative_intervals', 0)}."
        )
        self._refresh()

    def _add_false_negative_interval(self) -> None:
        manager = self._manager
        if manager is None:
            QMessageBox.warning(self, "Temporal Review", "Open a project first.")
            return
        sid = str(self._session.currentData() or "").strip()
        if not sid:
            QMessageBox.information(self, "Temporal Review", "Choose a subject/session first.")
            return

        center = int(self._player.current_frame)
        default_start = max(0, center - 15)
        default_end = max(default_start, center + 15)

        dlg = QDialog(self)
        dlg.setWindowTitle("Add False Negative Interval")
        start_spin = QSpinBox(dlg)
        start_spin.setRange(0, 10_000_000)
        start_spin.setValue(default_start)
        end_spin = QSpinBox(dlg)
        end_spin.setRange(0, 10_000_000)
        end_spin.setValue(default_end)

        # Behavior selection: what behavior is actually present?
        # Pre-fills with the current concept (the one being reviewed).
        concept_id = self._concept_id()
        behavior_combo = QComboBox(dlg)
        for b in self._behaviors.behaviors:
            behavior_combo.addItem(b.name, userData=str(b.behavior_id))
        pre_idx = behavior_combo.findData(concept_id)
        if pre_idx >= 0:
            behavior_combo.setCurrentIndex(pre_idx)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Start frame:"))
        frame_row.addWidget(start_spin)
        frame_row.addWidget(QLabel("End frame:"))
        frame_row.addWidget(end_spin)

        beh_row = QHBoxLayout()
        beh_row.addWidget(QLabel("Behavior present:"))
        beh_row.addWidget(behavior_combo, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout(dlg)
        layout.addLayout(frame_row)
        layout.addLayout(beh_row)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        start = int(start_spin.value())
        end = int(end_spin.value())
        correct_label = str(behavior_combo.currentData() or concept_id)

        feedback = manager.add_temporal_feedback_interval(
            concept_id=concept_id,
            session_id=sid,
            start_frame=start,
            end_frame=end,
            feedback_type="false_negative",
        )
        # Write a finalized review decision so this interval is treated as
        # already-reviewed — no additional approval needed in the review tab.
        self._write_temporal_review_decisions(
            session_id=sid, start=start, end=end,
            review_label=correct_label, concept_id=concept_id,
        )
        self._status.setText(
            f"Saved false-negative interval {start}-{end} (‘{behavior_combo.currentText()}’). "
            f"FP={feedback.get('n_false_positive_intervals', 0)}, FN={feedback.get('n_false_negative_intervals', 0)}."
        )
        self._refresh()

    def _write_temporal_review_decisions(
        self,
        session_id: str,
        start: int,
        end: int,
        review_label: str,
        concept_id: str,
        decision_type: ReviewDecisionType = ReviewDecisionType.ACCEPT,
        behavior_label: str | None = None,
    ) -> None:
        """Tile the interval into AL-sized windows and write both a reviewer label
        row (parquet) and a review decision (JSON) for each tile so that the review
        tab treats these intervals as already-reviewed after a model run.

        ``review_label`` is written to the parquet store (used by AL training).
        ``behavior_label`` is written to the decisions JSON (used by temporal training);
        defaults to ``review_label`` when not supplied.
        ``decision_type`` controls the decision stored: ACCEPT for positives / universal
        negatives, REJECT for concept-specific negatives (false positives of that behavior).
        """
        project_root = self._project_root
        if project_root is None:
            return
        window_size = self._resolve_window_size()

        for tile_start in range(int(start), int(end), window_size):
            tile_end = tile_start + window_size - 1
            seg_id = f"seg_feedback_{session_id}_{tile_start}_{tile_end}"

            _decision_behavior = behavior_label if behavior_label is not None else review_label
            # Reviewer label (parquet) — timestamp now = after any prediction file.
            self._review_service.append_segment_label(
                ReviewerLabelRecord(
                    segment_id=seg_id,
                    review_label=review_label,
                    reviewer_id="temporal_feedback",
                    confidence=1.0,
                    notes=f"temporal:{review_label}:{concept_id}",
                )
            )
            # Review decision (JSON) — marks the tile as fully reviewed so it
            # never surfaces as needing approval in the review tab.
            self._review_service.upsert_decision(
                clip_id=seg_id,
                reviewer="temporal_feedback",
                decision=decision_type,
                behavior_label=_decision_behavior,
                notes=f"temporal:{_decision_behavior}:{concept_id}",
                adjusted_start_frame=tile_start,
                adjusted_end_frame=tile_end,
            )

    # ── Send all detected bouts to clip extraction ───────────────────────────

    def _resolve_bout_behavior_id(self, concept: str) -> str:
        """Return the real behavior_id to tag bout-derived windows with.

        If *concept* is already a specific behavior id (not the generic
        ``target_behavior`` sentinel), return it as-is.  When the concept IS
        ``target_behavior`` (i.e. 'All reviewed positives'), try to resolve to
        an actual behavior: if exactly one non-no_behavior behavior is active
        in this project we can unambiguously tag the bouts with it.
        """
        if concept and concept != "target_behavior":
            return concept
        active = [
            b for b in self._behaviors.behaviors
            if str(b.behavior_id or "").strip()
            and not self._is_no_behavior_token(str(b.behavior_id or "").strip())
        ]
        if len(active) == 1:
            return str(active[0].behavior_id).strip()
        return concept

    def _resolve_window_size(self) -> int:
        """Read the segment window size, checking multiple sources.

        Resolution order:
        1. representations.manifest.json  (provenance.config.representation_config.window_size_frames)
        2. workflow_snapshot.json          (segment_window_frames)
        3. project.yaml                   (behavior_model.segment_window_frames)
        4. Hardcoded fallback of 60 frames
        """
        window_size = 60
        if self._project_root is None:
            return window_size

        # 1. Representations manifest
        manifest_path = self._project_root / "derived" / "representations" / "representations.manifest.json"
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                rep_cfg = (
                    (manifest_data.get("provenance") or {})
                    .get("config", {})
                    .get("representation_config", {})
                )
                w = int(rep_cfg.get("window_size_frames") or 0)
                if w >= 8:
                    return w
            except Exception:
                pass

        # 2. Workflow snapshot (records what was actually used for training)
        snapshot_path = self._project_root / "derived" / "workflow_snapshot.json"
        if snapshot_path.exists():
            try:
                snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
                w = int(snap.get("segment_window_frames") or 0)
                if w >= 8:
                    return w
            except Exception:
                pass

        # 3. Project config
        project_yaml = self._project_root / "project.yaml"
        if project_yaml.exists():
            try:
                import yaml
                proj = yaml.safe_load(project_yaml.read_text(encoding="utf-8")) or {}
                bm = proj.get("behavior_model") or {}
                w = int(bm.get("segment_window_frames") or 0)
                if w >= 8:
                    return w
            except Exception:
                pass

        return window_size

    def _collect_all_bout_candidates(
        self,
        behavior_filter: set[str] | None = None,
    ) -> list[CandidateWindow]:
        """Convert every detected bout across all behaviors into fixed-length CandidateWindows.

        Uses per-behavior probability columns and per-behavior threshold settings
        so that each behavior is evaluated independently.  Pre-computed per-behavior
        postprocess parquets are preferred when available; otherwise bouts are
        computed on-the-fly from the behaviour-specific probability column in the
        inference trace.

        Parameters
        ----------
        behavior_filter:
            If provided, only include bouts for behavior IDs in the set.
        """
        if self._project_root is None:
            return []

        window_size = self._resolve_window_size()
        candidates: list[CandidateWindow] = []
        seen_ids: set[str] = set()

        # Identify all behaviours to process.
        behavior_ids: list[str] = []
        for behavior in self._behaviors.behaviors:
            bid = str(behavior.behavior_id or "").strip()
            if not bid or self._is_no_behavior_token(bid):
                continue
            if behavior_filter is not None and bid not in behavior_filter:
                continue
            behavior_ids.append(bid)

        if not behavior_ids:
            return []

        # Pre-load full trace DataFrames (all columns) for on-the-fly bout
        # computation and scoring from per-behavior probability columns.
        trace_cache: dict[str, pd.DataFrame | None] = {}
        for sid, tp in self._trace_paths.items():
            if tp and Path(tp).exists():
                try:
                    trace_cache[sid] = pd.read_parquet(tp)
                except Exception:
                    trace_cache[sid] = None
            else:
                trace_cache[sid] = None

        # Helper: tile a single bout into CandidateWindow(s).
        def _tile_bout(
            bout_bid: str, session_id: str,
            bout_start: int, bout_end: int, bout_score: float,
        ) -> None:
            bout_len = bout_end - bout_start + 1
            if bout_len <= 0:
                return
            if bout_len <= window_size:
                mid = (bout_start + bout_end) // 2
                w_start = max(0, mid - window_size // 2)
                w_end = w_start + window_size - 1
                wid = f"bout_{bout_bid}_{session_id}_{w_start}_{w_end}"
                if wid not in seen_ids:
                    seen_ids.add(wid)
                    candidates.append(CandidateWindow(
                        window_id=wid, session_id=session_id,
                        start_frame=w_start, end_frame=w_end,
                        behavior_id=bout_bid, total_score=bout_score,
                        source="temporal_bout_review",
                    ))
            else:
                for tile_start in range(bout_start, bout_end + 1, window_size):
                    tile_end = tile_start + window_size - 1
                    if tile_end > bout_end:
                        tile_start = max(bout_start, bout_end - window_size + 1)
                        tile_end = tile_start + window_size - 1
                    wid = f"bout_{bout_bid}_{session_id}_{tile_start}_{tile_end}"
                    if wid not in seen_ids:
                        seen_ids.add(wid)
                        candidates.append(CandidateWindow(
                            window_id=wid, session_id=session_id,
                            start_frame=tile_start, end_frame=tile_end,
                            behavior_id=bout_bid, total_score=bout_score,
                            source="temporal_bout_review",
                        ))

        concept = self._concept_id() or "target_behavior"

        for bid in behavior_ids:
            cfg = self._settings_for_behavior(bid)
            onset = float(cfg.get("onset_threshold", 0.65))
            min_bout = int(cfg.get("min_bout_duration_frames", 8))
            merge_gap = int(cfg.get("merge_gap_frames", 4))

            sessions_covered: set[str] = set()

            # ── 1. Per-behavior postprocess parquets (generated with correct
            #       per-behavior settings via Per-Behavior Thresholds dialog).
            bout_paths = self._bout_paths_by_behavior.get(bid, {})
            if not bout_paths:
                bout_paths = self._bout_paths_by_behavior.get(self._safe_name(bid), {})
            for session_id, parquet_path_str in bout_paths.items():
                parquet_path = Path(parquet_path_str)
                if not parquet_path.exists():
                    continue
                try:
                    df = pd.read_parquet(parquet_path)
                except Exception:
                    continue
                if df.empty or not {"start_frame", "end_frame"}.issubset(set(df.columns)):
                    continue
                sessions_covered.add(session_id)
                trace_df = trace_cache.get(session_id)
                prob_col = f"prob_{bid}"
                for row in df[["start_frame", "end_frame"]].itertuples(index=False):
                    bout_start = int(row.start_frame)
                    bout_end = int(row.end_frame)
                    bout_score = 0.0
                    if trace_df is not None:
                        score_col = prob_col if prob_col in trace_df.columns else "probability"
                        bout_slice = trace_df[
                            (trace_df["frame"] >= bout_start)
                            & (trace_df["frame"] <= bout_end)
                        ]
                        if not bout_slice.empty and score_col in bout_slice.columns:
                            mean_val = bout_slice[score_col].mean()
                            if pd.notna(mean_val):
                                bout_score = round(float(mean_val), 4)
                    _tile_bout(bid, session_id, bout_start, bout_end, bout_score)

            # ── 2. Non-competition concept bouts (main dropdown = this specific
            #       behavior, single-behavior inference where "probability" is
            #       behaviour-specific).
            if concept == bid and self._bout_paths:
                for session_id, parquet_path_str in self._bout_paths.items():
                    if session_id in sessions_covered:
                        continue
                    parquet_path = Path(parquet_path_str)
                    if not parquet_path.exists():
                        continue
                    try:
                        df = pd.read_parquet(parquet_path)
                    except Exception:
                        continue
                    if df.empty or not {"start_frame", "end_frame"}.issubset(set(df.columns)):
                        continue
                    sessions_covered.add(session_id)
                    trace_df = trace_cache.get(session_id)
                    for row in df[["start_frame", "end_frame"]].itertuples(index=False):
                        bout_start = int(row.start_frame)
                        bout_end = int(row.end_frame)
                        bout_score = 0.0
                        if trace_df is not None and "probability" in trace_df.columns:
                            bout_slice = trace_df[
                                (trace_df["frame"] >= bout_start)
                                & (trace_df["frame"] <= bout_end)
                            ]
                            if not bout_slice.empty:
                                mean_val = bout_slice["probability"].mean()
                                if pd.notna(mean_val):
                                    bout_score = round(float(mean_val), 4)
                        _tile_bout(bid, session_id, bout_start, bout_end, bout_score)

            # ── 3. On-the-fly: for sessions not covered above, compute bouts
            #       from the per-behavior probability column using this
            #       behavior's specific threshold settings.
            for session_id, _trace_path_str in self._trace_paths.items():
                if session_id in sessions_covered:
                    continue
                trace_df = trace_cache.get(session_id)
                if trace_df is None or trace_df.empty:
                    continue

                prob_col = f"prob_{bid}"
                if prob_col not in trace_df.columns:
                    # Non-competition trace: use "probability" only if this
                    # behavior matches the loaded concept.
                    if concept == bid and "probability" in trace_df.columns:
                        prob_col = "probability"
                    else:
                        continue

                frame_arr = trace_df["frame"].to_numpy(dtype=int)
                prob_arr = pd.to_numeric(
                    trace_df[prob_col], errors="coerce"
                ).fillna(0.0).to_numpy(dtype=float)
                smoothed = smooth_probabilities(prob_arr)
                binary = threshold_probabilities(smoothed, onset_thresh=onset)
                binary = merge_close_bouts(binary, merge_gap)
                binary = remove_short_bouts(binary, min_bout)
                intervals = binary_trace_to_intervals(binary)

                for idx_start, idx_end in intervals:
                    bout_start = int(frame_arr[idx_start])
                    bout_end = int(frame_arr[min(idx_end, len(frame_arr) - 1)])
                    bout_score = 0.0
                    fb_mean = trace_df.iloc[idx_start:idx_end + 1][prob_col].mean()
                    if pd.notna(fb_mean):
                        bout_score = round(float(fb_mean), 4)
                    _tile_bout(bid, session_id, bout_start, bout_end, bout_score)

        return candidates

    # ── Session quality inspector ─────────────────────────────────────────

    @staticmethod
    def _infer_session_type(session_id: str, name_hint: str = "") -> str:
        """Infer a session-phase label from the session name or ID.

        *name_hint* should be the video filename stem (e.g.
        ``CBMRE01_AcclimationDLC_Resnet50_...``); it is tried first so that
        opaque UUID-based session IDs are resolved correctly.  The label is
        used to group sessions together so that acclimation days are only
        compared to other acclimation days, etc.
        """
        # Prefer the human-readable filename stem over the internal session ID.
        raw = str(name_hint or session_id or "")
        # Split CamelCase BEFORE lowercasing so "TestingDay2" → "Testing Day2".
        # Only insert a space at lowercase→uppercase transitions (e.g. g→D in
        # "TestingDay") to avoid mangling all-caps acronyms like "DLC".
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw)
        text = text.replace("_", " ").replace("-", " ").lower()
        # Strip known DLC suffixes so they don't confuse the keyword search.
        text = re.sub(r"\bdlc\b.*", "", text)
        # Priority-ordered keyword patterns → label.
        # Numbered phase+day patterns must come BEFORE their generic equivalents
        # so "testing day2" → "testing_day2" rather than just "testing".
        patterns = [
            ("acclimation",   r"\bacclim"),
            ("habituation",   r"\bhabitu"),
            ("baseline",      r"\bbaseline|\bbase\b"),
            ("open field",    r"\bopen.?field"),
            ("epm",           r"\bepm\b|\belevated.?plus"),
            ("fst",           r"\bfst\b|\bforced.?swim"),
            ("tmt",           r"\btmt\b"),
            ("novel object",  r"\bnovel(?:.obj)?|\bnop\b"),
            ("probe",         r"\bprobe\b"),
            # ── numbered phase+day combinations (before bare phase keywords) ──
            ("testing_day1",  r"\btest(?:ing)?\b.*\bday\s*1"),
            ("testing_day2",  r"\btest(?:ing)?\b.*\bday\s*2"),
            ("testing_day3",  r"\btest(?:ing)?\b.*\bday\s*3"),
            ("training_day1", r"\btrain(?:ing)?\b.*\bday\s*1"),
            ("training_day2", r"\btrain(?:ing)?\b.*\bday\s*2"),
            ("training_day3", r"\btrain(?:ing)?\b.*\bday\s*3"),
            # ── generic phase keywords ──
            ("testing",       r"\btesting\b"),
            ("training",      r"\btraining\b|\btrain\b"),
            ("test",          r"\btest\b"),
        ]
        for label, pattern in patterns:
            if re.search(pattern, text):
                return label
        # Only use numeric day/session fallbacks on non-UUID strings.
        # UUID-style IDs look like "session_7e59b632" — we skip them.
        is_uuid_like = bool(re.search(r"\b[0-9a-f]{6,}\b", text))
        if not is_uuid_like:
            m = re.search(r"\bday\s*(\d+)", text)
            if m:
                return f"day{m.group(1)}"
            m = re.search(r"\bsession\s*(\d+)\b", text)
            if m:
                return f"session{m.group(1)}"
        return "other"

    def _compute_session_quality_stats(
        self,
        behavior_ids: list[str],
        z_threshold: float = 2.0,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Compute per-session quality metrics for each selected behavior.

        Groups sessions by their inferred session type (resolved via the video
        filename stem so UUID-based session IDs are handled correctly) and
        detects outliers using modified z-scores (Iglewicz & Hoaglin, 1993)
        relative to the group.  Absolute confidence floor checks supplement
        the relative z-scores so sessions are flagged even when the entire
        group has poor model confidence (MAD would otherwise be 0).

        Returns ``(rows, group_counts)`` where *rows* is a list of metric dicts
        (one per session × behavior pair) and *group_counts* maps
        ``session_type`` → number of unique sessions in that type group.
        """
        if self._project_root is None:
            return [], {}

        # ── 0. Build session_id → video filename stem map from manifest ───
        session_name_map: dict[str, str] = {}
        try:
            manifest = self._imports.load_manifest(self._project_root)
            if manifest is not None:
                video_by_id = {v.asset_id: v for v in manifest.videos}
                for session in manifest.linked_sessions:
                    sid = str(session.session_id)
                    video = video_by_id.get(session.video_asset_id)
                    if video and video.source_path:
                        stem = Path(video.source_path).stem
                        session_name_map[sid] = stem
        except Exception:
            pass

        # ── 1. Load trace DataFrames ──────────────────────────────────────
        trace_cache: dict[str, pd.DataFrame | None] = {}
        for sid, tp in self._trace_paths.items():
            if tp and Path(tp).exists():
                try:
                    trace_cache[sid] = pd.read_parquet(tp)
                except Exception:
                    trace_cache[sid] = None
            else:
                trace_cache[sid] = None

        # ── 2. Load per-behavior bout parquets ────────────────────────────
        bout_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
        for bid in behavior_ids:
            bout_paths = (
                self._bout_paths_by_behavior.get(bid)
                or self._bout_paths_by_behavior.get(self._safe_name(bid), {})
            )
            for sid, p in (bout_paths or {}).items():
                if p and Path(p).exists():
                    try:
                        bout_cache[(bid, sid)] = pd.read_parquet(p)
                    except Exception:
                        bout_cache[(bid, sid)] = None
                else:
                    bout_cache[(bid, sid)] = None

        concept = self._concept_id() or "target_behavior"
        all_session_ids = sorted(
            set(self._trace_paths.keys()) | set(self._subject_by_session.keys())
        )

        # ── 3. Collect raw metrics per (behavior, session) ────────────────
        raw_rows: list[dict[str, Any]] = []
        for bid in behavior_ids:
            cfg = self._settings_for_behavior(bid)
            onset = float(cfg.get("onset_threshold", 0.65))
            min_bout = int(cfg.get("min_bout_duration_frames", 8))
            merge_gap = int(cfg.get("merge_gap_frames", 4))

            bname = bid
            for beh in self._behaviors.behaviors:
                if str(beh.behavior_id or "").strip() == bid:
                    bname = str(beh.name or bid).strip() or bid
                    break

            for sid in all_session_ids:
                trace_df = trace_cache.get(sid)
                if trace_df is None or trace_df.empty or "frame" not in trace_df.columns:
                    continue

                prob_col = f"prob_{bid}"
                if prob_col not in trace_df.columns:
                    if concept == bid and "probability" in trace_df.columns:
                        prob_col = "probability"
                    else:
                        continue

                prob_arr = (
                    pd.to_numeric(trace_df[prob_col], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                n_frames = len(prob_arr)
                if n_frames == 0:
                    continue

                max_conf = float(np.max(prob_arr))
                p95_conf = float(np.percentile(prob_arr, 95))
                mean_conf = float(np.mean(prob_arr))
                frac_active = float(np.mean(prob_arr >= onset))

                # Bout metrics — prefer pre-computed parquets, fall back to on-the-fly.
                bout_df = bout_cache.get((bid, sid))
                if (
                    bout_df is not None
                    and not bout_df.empty
                    and {"start_frame", "end_frame"}.issubset(set(bout_df.columns))
                ):
                    n_bouts = len(bout_df)
                    durs = (bout_df["end_frame"] - bout_df["start_frame"]).clip(lower=0).to_numpy(dtype=float)
                    mean_bout_dur = float(np.mean(durs)) if n_bouts > 0 else float("nan")
                else:
                    smoothed = smooth_probabilities(prob_arr)
                    binary = threshold_probabilities(smoothed, onset_thresh=onset)
                    binary = merge_close_bouts(binary, merge_gap)
                    binary = remove_short_bouts(binary, min_bout)
                    intervals = binary_trace_to_intervals(binary)
                    n_bouts = len(intervals)
                    if n_bouts > 0:
                        durs = np.array([float(e - s + 1) for s, e in intervals])
                        mean_bout_dur = float(np.mean(durs))
                    else:
                        mean_bout_dur = float("nan")

                bout_density = (n_bouts / n_frames) * 1000.0 if n_frames > 0 else 0.0

                raw_rows.append({
                    "session_id": sid,
                    "subject": self._subject_by_session.get(sid, sid),
                    "session_type": self._infer_session_type(
                        sid, name_hint=session_name_map.get(sid, "")
                    ),
                    "behavior_id": bid,
                    "behavior_name": bname,
                    "n_frames": n_frames,
                    "max_conf": max_conf,
                    "p95_conf": p95_conf,
                    "mean_conf": mean_conf,
                    "frac_active": frac_active,
                    "n_bouts": n_bouts,
                    "bout_density": bout_density,
                    "mean_bout_dur": mean_bout_dur,
                })

        if not raw_rows:
            return [], {}

        # ── 4. Group by (behavior, session_type), compute modified z-scores ──
        from collections import defaultdict  # noqa: PLC0415

        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in raw_rows:
            groups[(row["behavior_id"], row["session_type"])].append(row)

        session_type_counts: dict[str, set[str]] = defaultdict(set)
        for row in raw_rows:
            session_type_counts[row["session_type"]].add(row["session_id"])
        group_counts = {k: len(v) for k, v in session_type_counts.items()}

        def _modified_z(value: float, arr: list[float]) -> float:
            """Modified z-score using median + MAD (robust to outliers)."""
            if len(arr) < 2:
                return 0.0
            a = np.array(arr, dtype=float)
            median = float(np.median(a))
            mad = float(np.median(np.abs(a - median)))
            if mad < 1e-9:
                std = float(np.std(a))
                if std < 1e-9:
                    return 0.0
                return float(0.6745 * (value - median) / std)
            return float(0.6745 * (value - median) / mad)

        out_rows: list[dict[str, Any]] = []
        for (_bid, _stype), group_rows in groups.items():
            max_confs = [r["max_conf"] for r in group_rows]
            p95_confs = [r["p95_conf"] for r in group_rows]
            mean_confs = [r["mean_conf"] for r in group_rows]
            n_bouts_vals = [float(r["n_bouts"]) for r in group_rows]
            densities = [r["bout_density"] for r in group_rows]
            frac_vals = [r["frac_active"] for r in group_rows]

            # Absolute floor thresholds — applied independently of z-score so
            # that sessions are still flagged when the entire group has poor
            # confidence (MAD collapses to 0 and relative z-scores are all 0).
            # These values are empirically conservative for rodent-behavior models.
            ABS_MAX_CONF_POOR   = 0.25   # max probability never exceeded 25 %
            ABS_MAX_CONF_WARN   = 0.50   # max probability never exceeded 50 %
            ABS_MEAN_CONF_FLOOR = 0.01   # mean probability < 1 % (model never activated)
            # Group-level medians help distinguish "this behavior is just rare
            # in this cohort" from "this session looks wrong vs peers."
            grp_median_max = float(np.median(max_confs))
            grp_median_n   = float(np.median(n_bouts_vals))

            for row in group_rows:
                # Confidence metrics: unusually LOW → bad (anomaly_z = how far below median)
                z_max  = max(0.0, -_modified_z(row["max_conf"],  max_confs))
                z_p95  = max(0.0, -_modified_z(row["p95_conf"],  p95_confs))
                z_mean = max(0.0, -_modified_z(row["mean_conf"], mean_confs))
                # Count / density: deviation in either direction is suspicious
                z_nbouts  = abs(_modified_z(float(row["n_bouts"]),    n_bouts_vals))
                z_density = abs(_modified_z(row["bout_density"],       densities))
                z_frac    = abs(_modified_z(row["frac_active"],        frac_vals))

                # Composite = worst single-metric anomaly across all six signals.
                composite_z = max(z_max, z_p95, z_mean, z_nbouts, z_density, z_frac)

                # ── Absolute floor checks (supplement relative z-scores) ──
                # 1. Model never meaningfully activated for this behavior.
                abs_flag_hard = (
                    row["max_conf"] < ABS_MAX_CONF_POOR
                    and row["mean_conf"] < ABS_MEAN_CONF_FLOOR
                )
                # 2. Session has notably lower peak confidence than group median
                #    (catches sessions dragged into "OK" by a uniformly bad cohort).
                abs_flag_vs_group = (
                    grp_median_max >= 0.40                   # group is normally confident
                    and row["max_conf"] < ABS_MAX_CONF_WARN  # this session is not
                )
                # 3. Zero bouts when the group median is ≥ 1 bout.
                abs_flag_no_bouts = (
                    int(row["n_bouts"]) == 0
                    and grp_median_n >= 1.0
                )

                abs_flagged = abs_flag_hard or abs_flag_vs_group or abs_flag_no_bouts

                out_row = dict(row)
                out_row["z_max_conf"]    = round(z_max, 2)
                out_row["z_p95_conf"]    = round(z_p95, 2)
                out_row["z_mean_conf"]   = round(z_mean, 2)
                out_row["z_n_bouts"]     = round(z_nbouts, 2)
                out_row["z_density"]     = round(z_density, 2)
                out_row["z_frac_active"] = round(z_frac, 2)
                out_row["composite_z"]   = round(composite_z, 2)
                out_row["group_size"]    = len(group_rows)
                out_row["abs_flagged"]   = abs_flagged
                out_row["is_flagged"] = (
                    abs_flagged
                    or (composite_z > z_threshold and len(group_rows) >= 2)
                )
                out_row["is_borderline"] = (
                    not out_row["is_flagged"]
                    and (
                        (composite_z > 1.5 and len(group_rows) >= 2)
                        or (
                            row["max_conf"] < ABS_MAX_CONF_WARN
                            and not abs_flagged
                        )
                    )
                )
                out_rows.append(out_row)

        # Sort: flagged → borderline → by composite_z desc
        out_rows.sort(
            key=lambda r: (-int(r["is_flagged"]), -int(r["is_borderline"]), -r["composite_z"])
        )
        return out_rows, group_counts

    def _collect_top_prob_clips(
        self,
        pairs: list[tuple[str, str]],
        top_n: int = 5,
    ) -> list[CandidateWindow]:
        """For each (session_id, behavior_id) pair extract the *top_n* highest-mean-
        probability non-overlapping windows from the inference trace.

        These windows are returned even when the probability never crosses the
        bout threshold — the intent is to surface the *best available evidence*
        for a behavior in sessions where the model struggled, so the user can
        label them and feed them back as supplemental training data.

        Parameters
        ----------
        pairs:
            List of ``(session_id, behavior_id)`` to extract clips for.
        top_n:
            Number of clip windows to extract per pair.
        """
        if self._project_root is None:
            return []

        window_size = self._resolve_window_size()
        concept = self._concept_id() or "target_behavior"
        candidates: list[CandidateWindow] = []
        seen_ids: set[str] = set()

        # Load trace DataFrames — cache to avoid re-reading the same file.
        trace_cache: dict[str, pd.DataFrame | None] = {}
        for sid, tp in self._trace_paths.items():
            if tp and Path(tp).exists():
                try:
                    trace_cache[sid] = pd.read_parquet(tp)
                except Exception:
                    trace_cache[sid] = None
            else:
                trace_cache[sid] = None

        for session_id, behavior_id in pairs:
            trace_df = trace_cache.get(session_id)
            if trace_df is None or trace_df.empty or "frame" not in trace_df.columns:
                continue

            # Resolve probability column.
            prob_col = f"prob_{behavior_id}"
            if prob_col not in trace_df.columns:
                if concept == behavior_id and "probability" in trace_df.columns:
                    prob_col = "probability"
                else:
                    continue

            frame_arr = trace_df["frame"].to_numpy(dtype=int)
            prob_arr = (
                pd.to_numeric(trace_df[prob_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            n = len(frame_arr)
            if n < window_size:
                continue

            # Score every possible window by its mean probability, then pick
            # the top-N using a greedy non-overlapping selection so we don't
            # return ten windows all centred on the same peak.
            # Use a sliding sum for efficiency.
            window_half = window_size // 2
            win_scores: list[tuple[float, int]] = []  # (score, frame_idx_start)
            step = max(1, window_size // 4)  # stride = quarter-window for fine coverage
            for idx in range(0, n - window_size + 1, step):
                w_mean = float(np.mean(prob_arr[idx: idx + window_size]))
                win_scores.append((w_mean, idx))

            # Descending by mean probability.
            win_scores.sort(key=lambda x: -x[0])

            picked: list[tuple[int, int, float]] = []  # (start_frame, end_frame, score)
            for score, idx_start in win_scores:
                if len(picked) >= top_n:
                    break
                f_start = int(frame_arr[idx_start])
                f_end   = int(frame_arr[min(idx_start + window_size - 1, n - 1)])
                # Non-overlap check: reject if this window overlaps any already-picked one.
                overlap = False
                for ps, pe, _ in picked:
                    if f_start <= pe and f_end >= ps:
                        overlap = True
                        break
                if overlap:
                    continue
                picked.append((f_start, f_end, score))

            for f_start, f_end, score in picked:
                wid = f"quality_clip_{behavior_id}_{session_id}_{f_start}_{f_end}"
                if wid in seen_ids:
                    continue
                seen_ids.add(wid)
                candidates.append(CandidateWindow(
                    window_id=wid,
                    session_id=session_id,
                    start_frame=f_start,
                    end_frame=f_end,
                    behavior_id=behavior_id,
                    total_score=round(score, 4),
                    source="quality_check",
                    selection_reason="top_prob_window",
                ))

        return candidates

    def _open_session_quality_dialog(self) -> None:
        """Open the Session Quality Inspector dialog."""
        if self._project_root is None:
            QMessageBox.warning(self, "Session Quality", "Open a project first.")
            return
        if not self._trace_paths:
            QMessageBox.information(
                self, "Session Quality",
                "No inference traces are loaded. Run inference and apply thresholds first.",
            )
            return

        # Gather behaviors available for analysis.
        behavior_rows: list[tuple[str, str, float]] = []
        for beh in self._behaviors.behaviors:
            bid = str(beh.behavior_id or "").strip()
            bname = str(beh.name or bid).strip() or bid
            if not bid or self._is_no_behavior_token(bid) or self._is_no_behavior_token(bname):
                continue
            cfg = self._settings_for_behavior(bid)
            threshold = float(cfg.get("onset_threshold", 0.65))
            behavior_rows.append((bid, bname, threshold))

        if not behavior_rows:
            concept = self._concept_id()
            if concept and not self._is_no_behavior_token(concept):
                behavior_rows.append((concept, concept, 0.65))

        if not behavior_rows:
            QMessageBox.information(self, "Session Quality", "No behaviors are configured.")
            return

        # ── Build dialog ──────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Session Quality Inspector")
        dlg.resize(1050, 640)

        # — Behavior selection —
        beh_group = QGroupBox("Behaviors to analyze")
        beh_vlayout = QVBoxLayout(beh_group)
        checkboxes: list[tuple[str, QCheckBox]] = []
        for bid, bname, thr in behavior_rows:
            cb = QCheckBox(f"{bname}  (threshold: {thr:.3f})")
            cb.setChecked(True)
            beh_vlayout.addWidget(cb)
            checkboxes.append((bid, cb))
        beh_vlayout.addStretch()

        # — Settings panel —
        settings_group = QGroupBox("Outlier settings")
        settings_vlayout = QVBoxLayout(settings_group)
        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z-threshold:"))
        z_spin = QDoubleSpinBox()
        z_spin.setRange(0.5, 10.0)
        z_spin.setSingleStep(0.5)
        z_spin.setValue(2.0)
        z_spin.setDecimals(1)
        z_spin.setToolTip(
            "Sessions whose composite anomaly z-score exceeds this value are flagged.\n"
            "A value of 2.0 means the session deviates more than 2\u03c3 from its\n"
            "session-type group median on at least one key metric."
        )
        z_row.addWidget(z_spin)
        z_row.addStretch()
        settings_vlayout.addLayout(z_row)

        top_n_row = QHBoxLayout()
        top_n_row.addWidget(QLabel("Top clips / pair:"))
        top_n_spin = QSpinBox()
        top_n_spin.setRange(1, 50)
        top_n_spin.setValue(5)
        top_n_spin.setToolTip(
            "How many highest-confidence clip windows to extract per "
            "session \u00d7 behavior pair when sending to Clip Review."
        )
        top_n_row.addWidget(top_n_spin)
        top_n_row.addStretch()
        settings_vlayout.addLayout(top_n_row)

        hint_label = QLabel(
            "Sessions are compared only against others of the same\n"
            "type (e.g. acclimation vs acclimation, test vs test),\n"
            "inferred automatically from the session name."
        )
        hint_label.setWordWrap(True)
        settings_vlayout.addWidget(hint_label)
        settings_vlayout.addStretch()

        analyze_btn = QPushButton("Analyze")
        analyze_btn.setMinimumWidth(100)

        top_panel = QHBoxLayout()
        top_panel.addWidget(beh_group, 1)
        top_panel.addWidget(settings_group)
        top_panel.addWidget(analyze_btn, 0, Qt.AlignmentFlag.AlignTop)

        # — Results table —
        COL_HEADERS = [
            "Subject", "Sess. Type", "Behavior", "Group N",
            "Max Conf", "P95 Conf", "Mean Conf",
            "N Bouts", "Dens./1k",
            "Composite Z", "Status",
        ]
        results_table = QTableWidget(0, len(COL_HEADERS))
        results_table.setHorizontalHeaderLabels(COL_HEADERS)
        results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        results_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        results_table.setSortingEnabled(True)
        results_table.setAlternatingRowColors(False)

        # — Legend —
        legend_label = QLabel(
            "\u25a0 Flagged (>\u03c3 threshold)"
            "    \u25a0 Borderline (1.5\u2013threshold)"
            "    \u25a0 OK"
            "    \u25a0 N/A (group < 2)"
        )
        legend_label.setStyleSheet(
            "color: #555; font-size: 11px;"
        )

        summary_label = QLabel("Select behaviors and click \u2018Analyze\u2019 to run.")
        summary_label.setWordWrap(True)

        export_btn = QPushButton("Export CSV\u2026")
        export_btn.setEnabled(False)
        send_clips_btn = QPushButton("Send Selected to Clip Review\u2026")
        send_clips_btn.setEnabled(False)
        send_clips_btn.setToolTip(
            "Extract the top-confidence clip windows for each selected row\n"
            "(or all flagged/borderline rows if nothing is selected) and\n"
            "append them to Clip Review for supplemental labelling."
        )
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(legend_label)
        bottom_row.addStretch()
        bottom_row.addWidget(send_clips_btn)
        bottom_row.addWidget(export_btn)
        bottom_row.addWidget(close_btn)

        layout = QVBoxLayout(dlg)
        layout.addLayout(top_panel)
        layout.addWidget(results_table, 1)
        layout.addWidget(summary_label)
        layout.addLayout(bottom_row)

        # ── Colors (defined once) ─────────────────────────────────────────
        COLOR_FLAGGED    = QColor(255, 190, 190)   # red
        COLOR_BORDERLINE = QColor(255, 237, 150)   # yellow
        COLOR_OK         = QColor(210, 245, 210)   # green
        COLOR_NA         = QColor(230, 230, 230)   # grey

        # ── Cached rows for export ────────────────────────────────────────
        _cached_rows: list[dict] = []

        def _run_analysis() -> None:
            nonlocal _cached_rows
            selected_bids = [bid for bid, cb in checkboxes if cb.isChecked()]
            if not selected_bids:
                QMessageBox.information(dlg, "Session Quality", "Select at least one behavior.")
                return

            threshold = float(z_spin.value())
            analyze_btn.setEnabled(False)
            summary_label.setText("Computing quality metrics\u2026")
            dlg.repaint()

            try:
                rows, group_counts = self._compute_session_quality_stats(
                    selected_bids, z_threshold=threshold
                )
            except Exception as exc:
                QMessageBox.warning(dlg, "Session Quality", f"Analysis failed:\n{exc}")
                analyze_btn.setEnabled(True)
                return

            _cached_rows = rows

            results_table.setSortingEnabled(False)
            results_table.setRowCount(0)

            for row in rows:
                group_n = int(row.get("group_size", 1))
                composite_z = float(row.get("composite_z", 0.0))
                is_flagged    = bool(row.get("is_flagged", False))
                is_borderline = bool(row.get("is_borderline", False))

                if group_n < 2:
                    row_color  = COLOR_NA
                    status_txt = "N/A"
                elif is_flagged:
                    row_color  = COLOR_FLAGGED
                    status_txt = "\u26a0 FLAGGED"
                    if row.get("abs_flagged"):
                        status_txt = "\u26a0 FLAGGED (abs)"
                elif is_borderline:
                    row_color  = COLOR_BORDERLINE
                    status_txt = "\u007e borderline"
                else:
                    row_color  = COLOR_OK
                    status_txt = "\u2713 OK"

                mean_dur = row.get("mean_bout_dur", float("nan"))
                mean_dur_str = (
                    f"{mean_dur:.1f} fr"
                    if isinstance(mean_dur, float) and not np.isnan(mean_dur)
                    else "—"
                )
                tooltip = (
                    f"Session: {row['session_id']}\n"
                    f"Frames: {row['n_frames']:,}  |  "
                    f"Frac active: {row['frac_active']:.3f}\n"
                    f"Mean bout duration: {mean_dur_str}\n"
                    f"Z scores (anomaly direction):\n"
                    f"  max conf : {row['z_max_conf']:.2f}  "
                    f"  p95 conf : {row['z_p95_conf']:.2f}  "
                    f"  mean conf: {row['z_mean_conf']:.2f}\n"
                    f"  n bouts  : {row['z_n_bouts']:.2f}  "
                    f"  density  : {row['z_density']:.2f}  "
                    f"  frac act.: {row['z_frac_active']:.2f}"
                )

                cells = [
                    str(row["subject"]),
                    str(row["session_type"]),
                    str(row["behavior_name"]),
                    str(group_n),
                    f"{row['max_conf']:.3f}",
                    f"{row['p95_conf']:.3f}",
                    f"{row['mean_conf']:.3f}",
                    str(int(row["n_bouts"])),
                    f"{row['bout_density']:.2f}",
                    f"{composite_z:.2f}",
                    status_txt,
                ]

                r_idx = results_table.rowCount()
                results_table.insertRow(r_idx)
                for c_idx, cell_text in enumerate(cells):
                    item = QTableWidgetItem(cell_text)
                    item.setBackground(row_color)
                    item.setForeground(QColor(20, 20, 20))
                    item.setToolTip(tooltip)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    results_table.setItem(r_idx, c_idx, item)

            results_table.setSortingEnabled(True)

            n_flagged    = sum(1 for r in rows if r.get("is_flagged"))
            n_borderline = sum(1 for r in rows if r.get("is_borderline"))
            group_summary = ", ".join(
                f"{t}: {c}" for t, c in sorted(group_counts.items())
            )
            summary_label.setText(
                f"{n_flagged} flagged, {n_borderline} borderline "
                f"out of {len(rows)} session\u00d7behavior pair(s).  "
                f"Session-type groups: {group_summary or 'N/A'}"
            )
            export_btn.setEnabled(bool(rows))
            send_clips_btn.setEnabled(bool(rows))
            analyze_btn.setEnabled(True)

        def _export_csv() -> None:
            if not _cached_rows:
                return
            path, _ = QFileDialog.getSaveFileName(
                dlg, "Export Session Quality Report", "", "CSV files (*.csv)"
            )
            if not path:
                return
            try:
                df = pd.DataFrame(_cached_rows)
                export_cols = [
                    "subject", "session_id", "session_type", "behavior_name",
                    "n_frames", "max_conf", "p95_conf", "mean_conf", "frac_active",
                    "n_bouts", "bout_density", "mean_bout_dur",
                    "z_max_conf", "z_p95_conf", "z_mean_conf",
                    "z_n_bouts", "z_density", "z_frac_active",
                    "composite_z", "is_flagged", "is_borderline", "group_size",
                ]
                cols_present = [c for c in export_cols if c in df.columns]
                df[cols_present].to_csv(path, index=False)
                QMessageBox.information(dlg, "Export", f"Saved to:\n{path}")
            except Exception as exc:
                QMessageBox.warning(dlg, "Export failed", str(exc))

        def _send_selected_to_clip_review() -> None:
            if not _cached_rows:
                QMessageBox.information(dlg, "Session Quality", "Run analysis first.")
                return

            top_n = int(top_n_spin.value())

            # Determine which rows to use: selected rows first, then fall back
            # to all flagged + borderline rows if nothing is explicitly selected.
            selected_model_rows = {
                idx.row() for idx in results_table.selectionModel().selectedRows()
            }

            if selected_model_rows:
                # results_table may be sorted; map visual row → cached row via
                # (subject, session_type, behavior_name) composite key.
                target_pairs: list[tuple[str, str]] = []
                for vis_row in sorted(selected_model_rows):
                    subj_item = results_table.item(vis_row, 0)
                    stype_item = results_table.item(vis_row, 1)
                    beh_item   = results_table.item(vis_row, 2)
                    if not subj_item:
                        continue
                    subj  = subj_item.text()
                    stype = stype_item.text() if stype_item else ""
                    bname = beh_item.text()   if beh_item   else ""
                    for row in _cached_rows:
                        if (
                            row["subject"] == subj
                            and row["session_type"] == stype
                            and row["behavior_name"] == bname
                        ):
                            target_pairs.append((row["session_id"], row["behavior_id"]))
                            break
            else:
                target_pairs = [
                    (r["session_id"], r["behavior_id"])
                    for r in _cached_rows
                    if r.get("is_flagged") or r.get("is_borderline")
                ]

            if not target_pairs:
                QMessageBox.information(
                    dlg, "Session Quality",
                    "No flagged/borderline sessions found, and nothing is selected.\n"
                    "Select rows manually or run the analysis so flagged rows appear."
                )
                return

            send_clips_btn.setEnabled(False)
            summary_label.setText("Extracting top-confidence clips\u2026")
            dlg.repaint()

            try:
                candidates = self._collect_top_prob_clips(
                    target_pairs, top_n=top_n
                )
            except Exception as exc:
                QMessageBox.warning(dlg, "Clip extraction failed", str(exc))
                send_clips_btn.setEnabled(True)
                summary_label.setText("Clip extraction failed.")
                return

            if not candidates:
                QMessageBox.information(
                    dlg, "Session Quality",
                    "No probability data found for the selected session/behavior pairs.\n"
                    "Make sure inference traces are loaded."
                )
                send_clips_btn.setEnabled(True)
                return

            # Build a human-readable label from the unique behaviors in the batch.
            beh_names = sorted({
                next(
                    (r["behavior_name"] for r in _cached_rows
                     if r["session_id"] == c.session_id and r["behavior_id"] == c.behavior_id),
                    str(c.behavior_id)
                )
                for c in candidates
            })
            label = "Quality-check top clips – " + ", ".join(beh_names[:3])
            if len(beh_names) > 3:
                label += f" +{len(beh_names) - 3} more"

            self.bout_candidates_append_requested.emit(candidates, label)
            n_sessions = len({c.session_id for c in candidates})
            summary_label.setText(
                f"Sent {len(candidates)} clip window(s) from {n_sessions} session(s) "
                f"to Clip Review."
            )
            send_clips_btn.setEnabled(True)

        analyze_btn.clicked.connect(_run_analysis)
        export_btn.clicked.connect(_export_csv)
        send_clips_btn.clicked.connect(_send_selected_to_clip_review)

        dlg.exec()

    def _send_current_subject_behavior_bouts_to_clip_review(self) -> None:
        """Send bouts for the current behavior and current subject to Clip Review (appending)."""
        if self._project_root is None:
            QMessageBox.warning(self, "Temporal Review", "Open a project first.")
            return

        if not self._bout_paths_by_behavior and not self._bout_paths and not self._trace_paths:
            QMessageBox.information(
                self, "Temporal Review",
                "No bout outputs are loaded. Run inference and apply thresholds first.",
            )
            return

        # Determine the current behavior filter.  The dropdown stores probability
        # column names ('prob_<behavior_id>'), so translate back to a behavior ID.
        trace_col = str(self._trace_behavior.currentData() or "__all__").strip()
        behavior_filter: set[str] | None = None
        behavior_label = "all behaviors"
        if trace_col and trace_col != "__all__":
            trace_bid = self._behavior_id_from_col(trace_col)
            if not trace_bid:
                QMessageBox.information(
                    self, "Temporal Review",
                    f"Could not match the selected trace '{self._trace_behavior.currentText()}' "
                    "to a behavior in this project.",
                )
                return
            behavior_filter = {trace_bid}
            behavior_label = self._behavior_label_from_col(trace_col)

        # Determine the current subject and all sessions belonging to it.
        current_sid = str(self._session.currentData() or "").strip()
        if not current_sid:
            QMessageBox.information(
                self, "Temporal Review",
                "No session is selected. Select a session first.",
            )
            return
        current_subject = self._subject_by_session.get(current_sid, current_sid)
        # Collect every session that belongs to the same subject.
        subject_sessions: set[str] = {
            sid for sid, subj in self._subject_by_session.items()
            if subj == current_subject
        }
        if not subject_sessions:
            subject_sessions = {current_sid}

        # Collect all bout candidates (optionally filtered by behavior), then
        # narrow down to the subject's sessions.
        all_candidates = self._collect_all_bout_candidates(behavior_filter=behavior_filter)
        candidates = [c for c in all_candidates if c.session_id in subject_sessions]

        if not candidates:
            QMessageBox.information(
                self, "Temporal Review",
                f"No bouts found for subject '{current_subject}' / {behavior_label}.",
            )
            return

        n_behaviors = len({c.behavior_id for c in candidates if c.behavior_id})
        self.bout_candidates_append_requested.emit(
            candidates,
            f"Temporal bout review – {current_subject}",
        )
        self._status.setText(
            f"Appended {len(candidates)} clip window(s) for subject '{current_subject}' "
            f"({n_behaviors} behavior(s)) to Clip Extraction."
        )

    def _send_bouts_to_clip_review(self) -> None:
        """Show a behavior selection dialog, then collect and emit bouts."""
        if self._project_root is None:
            QMessageBox.warning(self, "Temporal Review", "Open a project first.")
            return

        if not self._bout_paths_by_behavior and not self._bout_paths and not self._trace_paths:
            QMessageBox.information(
                self, "Temporal Review",
                "No bout outputs are loaded. Run inference and apply thresholds first.",
            )
            return

        # Gather available behaviors.
        behavior_rows: list[tuple[str, str]] = []
        for behavior in self._behaviors.behaviors:
            bid = str(behavior.behavior_id or "").strip()
            bname = str(behavior.name or bid).strip() or bid
            if not bid or self._is_no_behavior_token(bid) or self._is_no_behavior_token(bname):
                continue
            behavior_rows.append((bid, bname))

        if not behavior_rows:
            QMessageBox.information(
                self, "Temporal Review",
                "No behaviors available to send.",
            )
            return

        # ── Behavior selection dialog ────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Send Bouts to Clip Extraction")
        dlg.resize(400, 320)

        hint = QLabel(
            "Select which behaviors to send to clip extraction. "
            "Bouts will be detected using each behavior's own threshold settings."
        )
        hint.setWordWrap(True)

        scroll = QScrollArea(dlg)
        scroll.setWidgetResizable(True)
        check_container = QWidget()
        check_layout = QVBoxLayout(check_container)
        check_layout.setContentsMargins(6, 6, 6, 6)

        checkboxes: list[tuple[str, QCheckBox]] = []
        for bid, bname in behavior_rows:
            cfg = self._settings_for_behavior(bid)
            threshold = float(cfg.get("onset_threshold", 0.65))
            cb = QCheckBox(f"{bname}  (threshold {threshold:.3f})")
            cb.setChecked(True)
            check_layout.addWidget(cb)
            checkboxes.append((bid, cb))

        check_layout.addStretch()
        scroll.setWidget(check_container)

        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        sel_row = QHBoxLayout()
        sel_row.addWidget(select_all_btn)
        sel_row.addWidget(deselect_all_btn)
        sel_row.addStretch()

        def _select_all() -> None:
            for _, cb in checkboxes:
                cb.setChecked(True)

        def _deselect_all() -> None:
            for _, cb in checkboxes:
                cb.setChecked(False)

        select_all_btn.clicked.connect(_select_all)
        deselect_all_btn.clicked.connect(_deselect_all)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout(dlg)
        layout.addWidget(hint)
        layout.addLayout(sel_row)
        layout.addWidget(scroll)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        selected_ids: set[str] = set()
        for bid, cb in checkboxes:
            if cb.isChecked():
                selected_ids.add(bid)

        if not selected_ids:
            QMessageBox.information(
                self, "Temporal Review",
                "No behaviors selected.",
            )
            return

        candidates = self._collect_all_bout_candidates(behavior_filter=selected_ids)
        if not candidates:
            QMessageBox.information(
                self, "Temporal Review",
                "No bouts were found for the selected behavior(s).",
            )
            return

        n_behaviors = len({c.behavior_id for c in candidates if c.behavior_id})
        n_sessions = len({c.session_id for c in candidates})
        self.bout_candidates_requested.emit(candidates, "Temporal bout review")
        self._status.setText(
            f"Sent {len(candidates)} clip window(s) from {n_behaviors} behavior(s) "
            f"across {n_sessions} session(s) to Clip Extraction."
        )

    def _load_postprocess(self, post_dir: Path) -> None:
        metrics_path = post_dir / "session_metrics.json"
        manifest_path = post_dir / "postprocess_manifest.json"
        if not metrics_path.exists() or not manifest_path.exists():
            self._status.setText("Postprocess artifacts incomplete.")
            self._refresh_probability_plot()
            return

        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            self._status.setText("Failed to parse postprocess artifacts.")
            self._refresh_probability_plot()
            return

        self._session_rows = []
        # Build rows from all known session sources so review always lists every
        # processed session, even when postprocess metrics were generated for a subset.
        metrics_by_session = {
            str(k): dict(v or {}) for k, v in dict(metrics or {}).items()
        }
        manifest_bout_paths = {
            str(k): str(v)
            for k, v in dict(manifest.get("bout_paths", {}) or {}).items()
        }
        session_ids = sorted(
            set(metrics_by_session.keys())
            | set(manifest_bout_paths.keys())
            | set(self._trace_paths.keys())
        )
        used_subjects: set[str] = set()
        for sid in session_ids:
            item = metrics_by_session.get(sid, {})
            frame_m = item.get("frame_metrics", {}) or {}
            bout_m = item.get("bout_metrics", {}) or {}
            subject = self._subject_display(sid, used=used_subjects)
            used_subjects.add(str(self._subject_by_session.get(sid, sid) or sid))
            self._session_rows.append(
                {
                    "session": sid,
                    "subject": subject,
                    "n_bouts": float(item.get("n_bouts", 0) or 0),
                    "time_spent_s": float(item.get("time_spent_seconds", 0.0) or 0.0),
                    "f1": float(frame_m.get("f1", float("nan")) or float("nan")),
                    "precision": float(frame_m.get("precision", float("nan")) or float("nan")),
                    "recall": float(frame_m.get("recall", float("nan")) or float("nan")),
                    "iou": float(bout_m.get("mean_iou", float("nan")) or float("nan")),
                    "onset_err": float(bout_m.get("onset_error_frames", float("nan")) or float("nan")),
                    "offset_err": float(bout_m.get("offset_error_frames", float("nan")) or float("nan")),
                }
            )

        self._bout_paths = manifest_bout_paths
        self._load_all_behavior_bout_paths()
        self._rebuild_metrics_table()

    def _load_all_behavior_bout_paths(self) -> None:
        """Populate _bout_paths_by_behavior for every behavior that has a postprocess run."""
        if self._project_root is None:
            return
        tr_root = self._project_root / "derived" / "temporal_refinement"
        self._bout_paths_by_behavior = {}
        for behavior in self._behaviors.behaviors:
            bid = str(behavior.behavior_id or "").strip()
            if not bid:
                continue
            token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in bid.strip())
            latest_path = tr_root / token / "latest.json"
            if not latest_path.exists():
                continue
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            post_dir_raw = str(latest.get("postprocess_dir", "") or "").strip()
            if not post_dir_raw:
                continue
            manifest_path = Path(post_dir_raw) / "postprocess_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                post_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            paths = {str(k): str(v) for k, v in dict(post_manifest.get("bout_paths", {}) or {}).items()}
            # Store under both the raw behavior_id and the safe token so lookup always succeeds
            self._bout_paths_by_behavior[bid] = paths
            self._bout_paths_by_behavior[token] = paths
            bname = str(behavior.name or "").strip()
            if bname:
                self._bout_paths_by_behavior[bname] = paths
                self._bout_paths_by_behavior[self._safe_name(bname)] = paths

    def _load_inference_trace_paths(self, inference_dir: Path) -> None:
        manifest_path = inference_dir / "inference_manifest.json"
        if not manifest_path.exists():
            self._trace_paths = {}
            self._competition_models = {}
            self._competition_excluded = []
            return
        try:
            inf = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            self._trace_paths = {}
            self._competition_models = {}
            self._competition_excluded = []
            return
        self._trace_paths = {str(k): str(v) for k, v in (inf.get("trace_paths", {}) or {}).items()}
        comp = dict(inf.get("competition") or {})
        self._competition_models = {str(k): str(v) for k, v in dict(comp.get("behavior_models", {}) or {}).items()}
        self._competition_excluded = [str(v) for v in list(comp.get("excluded_behavior_ids", []) or []) if str(v).strip()]

    def _sort_key(self, row: dict, mode: str) -> tuple[float, str]:
        def safe(v: float, nan_replacement: float) -> float:
            return nan_replacement if pd.isna(v) else float(v)

        if mode == "f1_desc":
            return (-safe(row["f1"], -1e9), row["subject"])
        if mode == "iou_asc":
            return (safe(row["iou"], 1e9), row["subject"])
        if mode == "iou_desc":
            return (-safe(row["iou"], -1e9), row["subject"])
        if mode == "onset_desc":
            return (-safe(row["onset_err"], -1e9), row["subject"])
        if mode == "offset_desc":
            return (-safe(row["offset_err"], -1e9), row["subject"])
        if mode == "nbouts_desc":
            return (-safe(row["n_bouts"], -1e9), row["subject"])
        if mode == "nbouts_asc":
            return (safe(row["n_bouts"], 1e9), row["subject"])
        return (safe(row["f1"], 1e9), row["subject"])

    def _rebuild_metrics_table(self) -> None:
        rows = list(self._session_rows)
        mode = str(self._sort_by.currentData() or "f1_asc")
        rows.sort(key=lambda r: self._sort_key(r, mode))

        self._metrics_table.setRowCount(0)
        self._session.blockSignals(True)
        self._session.clear()
        self._session.blockSignals(False)

        for row_data in rows:
            row = self._metrics_table.rowCount()
            self._metrics_table.insertRow(row)
            vals = [
                str(row_data["subject"]),
                self._fmt(row_data["n_bouts"]),
                self._fmt(row_data.get("time_spent_s", 0.0)),
                self._fmt(row_data["f1"]),
                self._fmt(row_data["precision"]),
                self._fmt(row_data["recall"]),
                self._fmt(row_data["iou"]),
                self._fmt(row_data["onset_err"]),
                self._fmt(row_data["offset_err"]),
            ]
            for col, val in enumerate(vals):
                self._metrics_table.setItem(row, col, QTableWidgetItem(val))

        self._session.blockSignals(True)
        for row_data in rows:
            sid = str(row_data["session"])
            self._session.addItem(str(row_data["subject"]), userData=sid)
        self._session.blockSignals(False)

        if self._session.count() > 0:
            self._session.setCurrentIndex(0)
            self._refresh_session_bouts()
        else:
            self._refresh_probability_plot()

    def _refresh_session_bouts(self) -> None:
        self._bouts_table.setRowCount(0)
        self._player.close_clip()
        self._loaded_session_video_id = None
        self._set_selected_probability(None)
        sid = str(self._session.currentData() or "").strip()
        if not sid:
            self._refresh_probability_plot()
            return
        path_raw = str(self._bout_paths.get(sid, "")).strip()
        if not path_raw:
            self._refresh_probability_plot()
            return
        path = Path(path_raw)
        if not path.exists():
            self._refresh_probability_plot()
            return

        try:
            df = pd.read_parquet(path)
        except Exception:
            self._refresh_probability_plot()
            return

        for row_data in df[["start_frame", "end_frame"]].itertuples(index=False):
            row = self._bouts_table.rowCount()
            self._bouts_table.insertRow(row)
            self._bouts_table.setItem(row, 0, QTableWidgetItem(str(int(row_data.start_frame))))
            self._bouts_table.setItem(row, 1, QTableWidgetItem(str(int(row_data.end_frame))))

        self._refresh_probability_plot()
        self._update_staged_ui()

    def _open_per_behavior_thresholds_dialog(self) -> None:
        if self._manager is None:
            QMessageBox.warning(self, "Temporal Review", "Open a project first.")
            return
        if self._is_refreshing:
            return

        behavior_rows: list[tuple[str, str]] = []
        for behavior in self._behaviors.behaviors:
            bid = str(behavior.behavior_id or "").strip()
            bname = str(behavior.name or bid).strip() or bid
            if not bid or self._is_no_behavior_token(bid) or self._is_no_behavior_token(bname):
                continue
            behavior_rows.append((bid, bname))

        if not behavior_rows:
            QMessageBox.information(self, "Temporal Review", "No behaviors available for threshold setup.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Per-Behavior Thresholds")
        dlg.resize(620, 460)

        table = QTableWidget(len(behavior_rows), 4, dlg)
        table.setHorizontalHeaderLabels(
            [
                "Behavior",
                "Positive Threshold",
                "Min Bout (frames)",
                "Merge Gap (frames)",
            ]
        )
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for row, (bid, bname) in enumerate(behavior_rows):
            name_item = QTableWidgetItem(bname)
            name_item.setData(0x0100, bid)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, name_item)

            cfg = self._settings_for_behavior(bid)
            threshold = float(cfg.get("onset_threshold", 0.65))
            min_bout = int(cfg.get("min_bout_duration_frames", 8))
            merge_gap = int(cfg.get("merge_gap_frames", 4))

            th = QDoubleSpinBox(dlg)
            th.setRange(0.0, 1.0)
            th.setSingleStep(0.001)
            th.setDecimals(3)
            th.setValue(threshold)
            table.setCellWidget(row, 1, th)

            mb = QSpinBox(dlg)
            mb.setRange(1, 1200)
            mb.setValue(min_bout)
            table.setCellWidget(row, 2, mb)

            mg = QSpinBox(dlg)
            mg.setRange(0, 1200)
            mg.setValue(merge_gap)
            table.setCellWidget(row, 3, mg)

        hint = QLabel(
            "Set per-behavior positive threshold, minimum bout length, and merge gap. "
            "Apply + Process regenerates bout outputs for all listed behaviors."
        )
        hint.setWordWrap(True)

        apply_btn = QPushButton("Apply + Process All Behaviors", dlg)
        close_btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dlg)
        close_btns.rejected.connect(dlg.reject)
        close_btns.accepted.connect(dlg.accept)

        def _collect_settings() -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item is None:
                    continue
                bid = str(item.data(0x0100) or "").strip()
                if not bid:
                    continue
                th_widget = table.cellWidget(row, 1)
                mb_widget = table.cellWidget(row, 2)
                mg_widget = table.cellWidget(row, 3)
                if (
                    not isinstance(th_widget, QDoubleSpinBox)
                    or not isinstance(mb_widget, QSpinBox)
                    or not isinstance(mg_widget, QSpinBox)
                ):
                    continue
                base = dict(self._settings_for_behavior(bid))
                thr = float(th_widget.value())
                base["onset_threshold"] = thr
                base["min_bout_duration_frames"] = int(mb_widget.value())
                base["merge_gap_frames"] = int(mg_widget.value())
                out[bid] = base
            return out

        def _apply_and_run() -> None:
            settings_map = _collect_settings()
            if not settings_map:
                QMessageBox.information(self, "Temporal Review", "No behavior settings to apply.")
                return

            for bid, cfg in settings_map.items():
                self._review_settings[bid] = cfg
            self._save_settings()
            self._apply_behavior_settings_to_controls()
            self._set_refresh_busy(True)
            self._status.setText("Applying per-behavior thresholds and regenerating temporal outputs...")

            worker = TaskWorker(self._batch_refresh_task, settings_map)
            worker.kwargs["progress_cb"] = worker.signals.line_emitted.emit
            worker.signals.line_emitted.connect(self._on_refresh_progress)
            worker.signals.finished.connect(self._on_batch_refresh_finished)
            worker.signals.failed.connect(self._on_refresh_failed)
            self._pool.start(worker)
            dlg.accept()

        apply_btn.clicked.connect(_apply_and_run)

        layout = QVBoxLayout(dlg)
        layout.addWidget(hint)
        layout.addWidget(table)
        layout.addWidget(apply_btn)
        layout.addWidget(close_btns)
        dlg.exec()

    def _batch_refresh_task(
        self,
        settings_map: dict[str, dict[str, Any]],
        progress_cb=None,
    ) -> dict[str, Any]:
        manager = self._manager
        if manager is None:
            raise ValueError("No project loaded")

        processed: list[str] = []
        skipped: list[str] = []
        for idx, (bid, cfg_vals) in enumerate(settings_map.items(), start=1):
            latest_path = (
                self._project_root
                / "derived"
                / "temporal_refinement"
                / self._safe_name(bid)
                / "latest.json"
            ) if self._project_root is not None else None
            if latest_path is None or (not latest_path.exists()):
                skipped.append(bid)
                if progress_cb is not None:
                    progress_cb(f"Skipping {bid}: no temporal inference artifacts found.")
                continue

            onset = float(cfg_vals.get("onset_threshold", 0.65))
            cfg = TemporalRefinementConfig(
                onset_threshold=onset,
                min_bout_duration_frames=int(cfg_vals.get("min_bout_duration_frames", 8)),
                merge_gap_frames=int(cfg_vals.get("merge_gap_frames", 4)),
            )
            if progress_cb is not None:
                progress_cb(f"Batch postprocess {idx}/{len(settings_map)}: {bid}")
            manager.run_temporal_refinement_postprocess(
                concept_id=bid,
                config=cfg,
                progress_cb=progress_cb,
            )
            processed.append(bid)

        return {
            "status": "ok",
            "processed": processed,
            "skipped": skipped,
        }

    def _on_batch_refresh_finished(self, result: dict[str, Any]) -> None:
        self._set_refresh_busy(False)
        processed = list(result.get("processed", []) or [])
        skipped = list(result.get("skipped", []) or [])
        self._status.setText(
            f"Per-behavior processing complete: processed={len(processed)}, skipped={len(skipped)}"
        )
        self._refresh()

    def _init_trace_plot(self, trace_layout: QVBoxLayout) -> None:
        try:
            import matplotlib.figure as mfig  # noqa: PLC0415
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: PLC0415
            from matplotlib.backends.backend_qt import NavigationToolbar2QT  # noqa: PLC0415
        except Exception:
            self._trace_placeholder.setText(
                "Matplotlib is not available. Install matplotlib to use probability trace plotting."
            )
            return

        fig = mfig.Figure(figsize=(7, 2.5), tight_layout=True)
        axes = fig.add_subplot(111)
        axes.set_title("Probability across time")
        axes.set_xlabel("Frame")
        axes.set_ylabel("Probability")
        axes.set_ylim(0.0, 1.0)
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(230)
        toolbar = NavigationToolbar2QT(canvas, self)
        toolbar.setMovable(False)

        trace_layout.addWidget(toolbar)
        trace_layout.addWidget(canvas)

        self._trace_canvas = canvas
        self._trace_axes = axes
        self._trace_toolbar = toolbar
        self._trace_click_cid = canvas.mpl_connect("button_press_event", self._on_trace_click)
        canvas.mpl_connect("motion_notify_event", self._on_trace_motion)
        canvas.mpl_connect("button_release_event", self._on_trace_release)
        self._update_trace_navigation_state()
        self._trace_placeholder.hide()

    def _refresh_probability_plot(self) -> None:
        # Lazy-initialize the matplotlib canvas on first call so that proper
        # screen DPI is available (avoiding QFont::setPointSize <= 0 warnings).
        if self._trace_axes is None and self._trace_layout is not None:
            if not self._can_init_trace_plot():
                self._schedule_trace_plot_retry()
                return
            self._init_trace_plot(self._trace_layout)
        if self._trace_axes is None or self._trace_canvas is None:
            return

        sid = str(self._session.currentData() or "").strip()
        self._update_staged_ui()
        self._trace_axes.clear()
        # axes.clear() removes all patches — make sure the rubber-band rect
        # references are nulled out so subsequent drag attempts don't try to
        # call .remove() on already-detached patch objects.
        self._fp_drag_rect = None
        self._fn_drag_rect = None
        self._right_drag_rect = None
        self._trace_axes.set_title("Probability across time")
        self._trace_axes.set_xlabel("Frame")
        self._trace_axes.set_ylabel("Probability")
        self._trace_axes.set_ylim(0.0, 1.0)

        # Reset interactive state
        self._current_bouts = []
        self._current_fp_intervals = []
        self._current_fn_intervals = []

        if not sid:
            self._set_selected_probability(None)
            self._trace_axes.text(0.5, 0.5, "Choose a subject/session.", ha="center", va="center")
            self._trace_canvas.draw_idle()
            return

        trace_path = Path(str(self._trace_paths.get(sid, "")).strip())
        if not trace_path.exists():
            self._set_selected_probability(None)
            self._trace_axes.text(0.5, 0.5, "No probability trace for selected subject/session.", ha="center", va="center")
            self._trace_canvas.draw_idle()
            return

        try:
            trace_df = pd.read_parquet(trace_path)
        except Exception:
            self._set_selected_probability(None)
            self._trace_axes.text(0.5, 0.5, "Failed to load probability trace.", ha="center", va="center")
            self._trace_canvas.draw_idle()
            return

        if trace_df.empty or "frame" not in trace_df.columns:
            self._set_selected_probability(None)
            self._trace_axes.text(0.5, 0.5, "Trace is empty or malformed.", ha="center", va="center")
            self._trace_canvas.draw_idle()
            return

        prob_cols = [c for c in trace_df.columns if str(c) == "probability" or str(c).startswith("prob_")]
        if not prob_cols:
            self._set_selected_probability(None)
            self._trace_axes.text(0.5, 0.5, "Trace has no probability columns.", ha="center", va="center")
            self._trace_canvas.draw_idle()
            return

        self._trace_probability_lookup[sid] = trace_df[["frame", *prob_cols]].copy()
        self._refresh_trace_behavior_options(trace_df)
        selected_col = str(self._trace_behavior.currentData() or "__all__")

        frame = trace_df["frame"].to_numpy(dtype=float)
        stride = max(1, int(len(frame) // 5000))
        # Build frame_plot to match _maxpool_downsample block structure: one entry
        # per block of `stride` frames, using the first frame of each block as x.
        # This guarantees len(frame_plot) == len(y) for every plotted series.
        n_blocks = (len(frame) + stride - 1) // stride
        frame_plot = frame[: n_blocks * stride : stride]

        cfg = self._review_config()
        _PALETTE = ["#0D47A1", "#C62828", "#2E7D32", "#EF6C00", "#6A1B9A", "#00838F", "#F57F17", "#37474F"]
        multi_cols = [c for c in prob_cols if str(c).startswith("prob_")] or [c for c in prob_cols if c == "probability"]

        if selected_col == "__all__":
            # All behaviors: just overlay the traces — no behavior-specific thresholds or bouts
            for i, col in enumerate(multi_cols):
                raw = pd.to_numeric(trace_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                smoothed = smooth_probabilities(raw, method=cfg.smoothing_method, window=cfg.smoothing_window)
                y = self._maxpool_downsample(smoothed, stride)
                self._trace_axes.plot(frame_plot, y, color=_PALETTE[i % len(_PALETTE)], linewidth=1.15,
                                      label=self._behavior_label_from_col(col))
        else:
            # Single behavior: plot its trace, threshold lines, and correct bout intervals
            col = selected_col if selected_col in trace_df.columns else ("probability" if "probability" in trace_df.columns else str(prob_cols[0]))
            raw_full = pd.to_numeric(trace_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            smoothed_full = smooth_probabilities(raw_full, method=cfg.smoothing_method, window=cfg.smoothing_window)
            y = self._maxpool_downsample(smoothed_full, stride)
            self._trace_axes.plot(frame_plot, y, color="#0D47A1", linewidth=1.15,
                                  label=self._behavior_label_from_col(col))

            onset = float(cfg.onset_threshold)
            self._trace_axes.axhline(onset, color="#555", linestyle="-", linewidth=1.0, label=f"Threshold {onset:.2f}")

            prob_plot = self._maxpool_downsample(smoothed_full, stride)

            # Always compute bouts on-the-fly from the current UI threshold settings
            # so that highlighted bout regions are guaranteed to match the threshold
            # line shown on the plot.  Pre-computed parquet files may have been
            # generated with different settings and must not be used here — doing so
            # was the source of sub-threshold bouts being highlighted as positive.
            bouts = pd.DataFrame()
            frame_arr = trace_df["frame"].to_numpy(dtype=int)
            binary = threshold_probabilities(smoothed_full, onset_thresh=onset)
            binary = merge_close_bouts(binary, max_gap_frames=int(cfg.merge_gap_frames))
            binary = remove_short_bouts(binary, min_duration_frames=int(cfg.min_bout_duration_frames))
            intervals = binary_trace_to_intervals(binary)
            if intervals:
                rows_data = [
                    (int(frame_arr[s]), int(frame_arr[min(e, len(frame_arr) - 1)]))
                    for s, e in intervals
                ]
                bouts = pd.DataFrame(rows_data, columns=["start_frame", "end_frame"])

            if not bouts.empty and {"start_frame", "end_frame"}.issubset(set(bouts.columns)):
                bout_mask = [False] * len(frame_plot)
                bout_intervals: list[tuple[int, int]] = []
                for row in bouts[["start_frame", "end_frame"]].itertuples(index=False):
                    start = int(row.start_frame)
                    end = int(row.end_frame)
                    bout_intervals.append((start, end))
                    self._trace_axes.axvspan(start, end, color="#00A86B", alpha=0.22)
                    for i, fval in enumerate(frame_plot):
                        if start <= fval <= end:
                            bout_mask[i] = True
                self._current_bouts = bout_intervals
                self._trace_axes.fill_between(
                    frame_plot,
                    0.0,
                    prob_plot,
                    where=bout_mask,
                    interpolate=True,
                    color="#00875A",
                    alpha=0.42,
                    label="Positive bout area",
                )
            else:
                self._current_bouts = []

        # ── Draw FP / FN feedback overlays ────────────────────────────
        self._current_fp_intervals = []
        self._current_fn_intervals = []
        if self._manager is not None:
            try:
                feedback = self._manager.load_temporal_feedback(
                    concept_id=self._resolve_trace_concept_id(),
                )
            except Exception:
                feedback = {}
        else:
            feedback = {}
        if feedback:
            fp_by_session = feedback.get("false_positive_intervals_by_session", {})
            fn_by_session = feedback.get("false_negative_intervals_by_session", {})
            fp_intervals = list(fp_by_session.get(sid, []))
            fn_intervals = list(fn_by_session.get(sid, []))
            self._current_fp_intervals = [(int(iv[0]), int(iv[1])) for iv in fp_intervals if len(iv) >= 2]
            self._current_fn_intervals = [(int(iv[0]), int(iv[1])) for iv in fn_intervals if len(iv) >= 2]
            _fp_labeled = False
            for s, e in self._current_fp_intervals:
                self._trace_axes.axvspan(
                    s, e, color="#E53935", alpha=0.30,
                    label="Flagged FP" if not _fp_labeled else None,
                )
                _fp_labeled = True
            _fn_labeled = False
            for s, e in self._current_fn_intervals:
                self._trace_axes.axvspan(
                    s, e, color="#1E88E5", alpha=0.30,
                    label="Flagged FN" if not _fn_labeled else None,
                )
                _fn_labeled = True

        self._trace_axes.legend(loc="upper right", fontsize=8)
        self._trace_canvas.draw_idle()

    def _resolve_bout_paths_for_trace_column(self, col: str) -> dict[str, str]:
        col_str = str(col)
        if col_str == "probability":
            return self._bout_paths

        if not col_str.startswith("prob_"):
            return {}

        token = col_str.removeprefix("prob_")
        if not self._bout_paths_by_behavior:
            self._load_all_behavior_bout_paths()
        candidates: list[str] = [token, token.lower()]

        # Match both behavior-id based and display-name based safe tokens.
        for behavior in self._behaviors.behaviors:
            bid = str(behavior.behavior_id or "").strip()
            bname = str(behavior.name or "").strip()
            if not bid and not bname:
                continue
            safe_bid = self._safe_name(bid) if bid else ""
            safe_name = self._safe_name(bname) if bname else ""
            if token in {bid, bname, safe_bid, safe_name}:
                for key in (bid, safe_bid, bname, safe_name):
                    if key:
                        candidates.extend([key, key.lower()])

        seen: set[str] = set()
        for key in candidates:
            norm = str(key).strip()
            if not norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            paths = self._bout_paths_by_behavior.get(norm)
            if paths:
                return paths

        # Final relaxed match: compare normalized safe tokens across known keys.
        token_safe = self._safe_name(token).lower()
        for key, paths in self._bout_paths_by_behavior.items():
            if self._safe_name(str(key)).lower() == token_safe:
                return paths

        # If the selected trace column actually corresponds to the target concept,
        # allow concept-level fallback for backward compatibility.
        concept_id = self._concept_id()
        if token == self._safe_name(concept_id):
            return self._bout_paths

        return {}

    @staticmethod
    def _maxpool_downsample(arr: np.ndarray, stride: int) -> np.ndarray:
        """Downsample by taking the max within each block of `stride` samples.

        Unlike simple stride sampling (arr[::stride]), this guarantees that any
        peak in the full-resolution data is reflected in the downsampled result.
        That means threshold crossings are never hidden from the display even
        when stride > 1.
        """
        if stride <= 1:
            return arr.astype(float)
        n = len(arr)
        n_blocks = (n + stride - 1) // stride
        out = np.empty(n_blocks, dtype=float)
        for i in range(n_blocks):
            s = i * stride
            e = min(s + stride, n)
            out[i] = float(np.max(arr[s:e]))
        return out

    def _behavior_label_from_col(self, col: str) -> str:
        """Resolve a human-readable name for a prob_<token> column."""
        token = str(col).removeprefix("prob_") if str(col).startswith("prob_") else str(col)
        if not token or token == "probability":
            return "Target"
        low = token.lower()
        if low == "no_behavior":
            return "No Behavior"
        for behavior in self._behaviors.behaviors:
            bid = str(behavior.behavior_id or "").strip()
            safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in bid.strip())
            name = str(behavior.name or "").strip() or bid
            if token == bid or token == safe:
                return name
        return token

    def _behavior_id_from_col(self, col: str) -> str | None:
        """Resolve the behavior_id behind a trace column ('prob_<token>' / 'probability').

        The trace-behavior dropdown stores probability *column names*, while bout
        collection filters on behavior IDs; the two must be translated here.
        """
        raw = str(col or "").strip()
        if not raw or raw == "__all__":
            return None
        token = raw.removeprefix("prob_") if raw.startswith("prob_") else raw
        if not token or token == "probability":
            # Single-behavior inference: 'probability' belongs to the selected concept.
            return self._concept_id() or None
        if self._is_no_behavior_token(token):
            return None
        for behavior in self._behaviors.behaviors:
            bid = str(behavior.behavior_id or "").strip()
            if not bid:
                continue
            name = str(behavior.name or "").strip()
            candidates = {bid, self._safe_name(bid)}
            if name:
                candidates.update({name, self._safe_name(name)})
            if token in candidates:
                return bid
        return None

    def _refresh_trace_behavior_options(self, trace_df: pd.DataFrame) -> None:
        multi_cols = [c for c in trace_df.columns if str(c).startswith("prob_")]
        # sentinel value __all__ = show all behaviors together
        options: list[tuple[str, str]] = [("All behaviors", "__all__")]
        for col in multi_cols:
            label = self._behavior_label_from_col(col)
            options.append((label, col))
        # Fall back to raw probability column when no prob_* columns exist
        if not multi_cols and "probability" in trace_df.columns:
            options.append(("Target", "probability"))

        current = str(self._trace_behavior.currentData() or "__all__")
        self._trace_behavior.blockSignals(True)
        self._trace_behavior.clear()
        for label, value in options:
            self._trace_behavior.addItem(label, userData=value)
        idx = self._trace_behavior.findData(current)
        self._trace_behavior.setCurrentIndex(idx if idx >= 0 else 0)
        self._trace_behavior.blockSignals(False)

    def _set_selected_probability(self, probability: float | None) -> None:
        if probability is None:
            self._selected_probability.setText("Selected frame probability: --")
            return
        self._selected_probability.setText(f"Selected frame probability: {probability:.3f}")

    def _probability_for_frame(self, sid: str, frame_idx: int) -> float | None:
        trace_df = self._trace_probability_lookup.get(sid)
        if trace_df is None or trace_df.empty:
            return None
        try:
            frame_values = pd.to_numeric(trace_df["frame"], errors="coerce")
            selected_col = str(self._trace_behavior.currentData() or "__all__")
            if selected_col == "__all__":
                # Return max probability across all behavior columns for clicked frame
                selected_col = next((c for c in trace_df.columns if str(c).startswith("prob_")), "probability")
            if selected_col not in trace_df.columns:
                selected_col = "probability" if "probability" in trace_df.columns else next((c for c in trace_df.columns if str(c).startswith("prob_")), "")
                if not selected_col:
                    return None
            prob_values = pd.to_numeric(trace_df[selected_col], errors="coerce")
            nearest_idx = (frame_values - float(frame_idx)).abs().idxmin()
            value = prob_values.loc[nearest_idx]
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    def _on_trace_click(self, event) -> None:
        if self._trace_axes is None:
            return
        if event is None or event.inaxes != self._trace_axes or event.xdata is None:
            return

        sid = str(self._session.currentData() or "").strip()
        if not sid:
            self._status.setText("Choose a subject/session first.")
            return

        frame_idx = int(max(0, round(float(event.xdata))))

        # Right-click → start drag-to-remove; a bare right-click (no drag,
        # handled on release) removes the single interval under the cursor.
        if event.button == 3:
            self._right_drag_start = float(event.xdata)
            import matplotlib.patches as mpatches  # noqa: PLC0415
            if self._right_drag_rect is not None:
                try:
                    self._right_drag_rect.remove()
                except ValueError:
                    pass
                self._right_drag_rect = None
            self._right_drag_rect = mpatches.Rectangle(
                (self._right_drag_start, 0), 0, 1.0,
                linewidth=1.5, edgecolor="#757575", facecolor="#BDBDBD", alpha=0.30,
                transform=self._trace_axes.get_xaxis_transform(), clip_on=True,
                linestyle="dotted",
            )
            self._trace_axes.add_patch(self._right_drag_rect)
            if self._trace_canvas:
                self._trace_canvas.draw_idle()
            return

        # FP mode: left-click-and-drag stages all bouts in the dragged range;
        # a bare click (no drag) still stages the single bout under the cursor.
        if self._fp_flag_active and event.button == 1:
            self._fp_drag_start = float(event.xdata)
            import matplotlib.patches as mpatches  # noqa: PLC0415
            if self._fp_drag_rect is not None:
                self._fp_drag_rect.remove()
            self._fp_drag_rect = mpatches.Rectangle(
                (self._fp_drag_start, 0), 0, 1.0,
                linewidth=1.5, edgecolor="#B71C1C", facecolor="#EF5350", alpha=0.30,
                transform=self._trace_axes.get_xaxis_transform(), clip_on=True,
            )
            self._trace_axes.add_patch(self._fp_drag_rect)
            canvas = self._trace_canvas
            if canvas is not None:
                canvas.draw_idle()
            return

        # FN mode: left-click starts rubber-band drag
        if self._fn_flag_active and event.button == 1:
            self._fn_drag_start = float(event.xdata)
            import matplotlib.patches as mpatches  # noqa: PLC0415
            if self._fn_drag_rect is not None:
                self._fn_drag_rect.remove()
            self._fn_drag_rect = mpatches.Rectangle(
                (self._fn_drag_start, 0), 0, 1.0,
                linewidth=1.5, edgecolor="#1565C0", facecolor="#42A5F5", alpha=0.35,
            )
            self._trace_axes.add_patch(self._fn_drag_rect)
            canvas = self._trace_canvas
            if canvas is not None:
                canvas.draw_idle()
            return

        # Default: seek video
        self._jump_to_session_frame(sid=sid, frame_idx=frame_idx)

    def _on_trace_motion(self, event) -> None:
        """Update the rubber-band rectangle while dragging in FP, FN, or right-click mode."""
        if self._trace_axes is None or event is None or event.inaxes != self._trace_axes or event.xdata is None:
            return
        if self._fp_flag_active and self._fp_drag_start is not None and self._fp_drag_rect is not None:
            x0 = self._fp_drag_start
            x1 = float(event.xdata)
            self._fp_drag_rect.set_x(min(x0, x1))
            self._fp_drag_rect.set_width(abs(x1 - x0))
            if self._trace_canvas:
                self._trace_canvas.draw_idle()
            return
        if self._right_drag_start is not None and self._right_drag_rect is not None:
            x0 = self._right_drag_start
            x1 = float(event.xdata)
            self._right_drag_rect.set_x(min(x0, x1))
            self._right_drag_rect.set_width(abs(x1 - x0))
            if self._trace_canvas:
                self._trace_canvas.draw_idle()
            return
        if not self._fn_flag_active or self._fn_drag_start is None:
            return
        if self._fn_drag_rect is None:
            return
        x0 = self._fn_drag_start
        x1 = float(event.xdata)
        self._fn_drag_rect.set_x(min(x0, x1))
        self._fn_drag_rect.set_width(abs(x1 - x0))
        if self._trace_canvas:
            self._trace_canvas.draw_idle()

    def _on_trace_release(self, event) -> None:
        """Finish rubber-band selection in FP, FN, or right-click-remove mode."""
        # ── Right-click drag release → bulk-remove flagged intervals ────────
        if self._right_drag_start is not None:
            if self._right_drag_rect is not None:
                try:
                    self._right_drag_rect.remove()
                except ValueError:
                    pass
                self._right_drag_rect = None
            x_end = (
                float(event.xdata)
                if (event is not None and event.xdata is not None
                    and self._trace_axes is not None
                    and event.inaxes == self._trace_axes)
                else self._right_drag_start
            )
            drag_start = int(max(0, round(min(self._right_drag_start, x_end))))
            drag_end   = int(max(0, round(max(self._right_drag_start, x_end))))
            self._right_drag_start = None

            if drag_end - drag_start < 2:
                # Bare right-click — remove the single interval under cursor
                frame_idx = drag_start
                hit = self._find_flagged_interval_at_frame(frame_idx)
                if hit:
                    ftype, (s, e), is_staged = hit
                    if is_staged:
                        self._unstage(ftype, s, e)
                    else:
                        self._unflag_interval(ftype, s, e)
            else:
                # Drag — remove all intervals overlapping [drag_start, drag_end]
                removed = 0
                concept_id = self._resolve_trace_concept_id()
                sid = str(self._session.currentData() or "").strip()
                manager = self._manager
                # Remove staged intervals in range for the current session only
                # (staged items only exist for the visible session's concept)
                for ftype, key_dict in (
                    ("false_positive", self._staged_fp_by_key),
                    ("false_negative", self._staged_fn_by_key),
                ):
                    for key, intervals in list(key_dict.items()):
                        to_remove = [(s, e) for s, e in intervals
                                     if s <= drag_end and e >= drag_start]
                        for pair in to_remove:
                            intervals.remove(pair)
                            removed += 1
                # Remove committed intervals in range
                committed_fp = [(s, e) for s, e in self._current_fp_intervals
                                if s <= drag_end and e >= drag_start]
                committed_fn = [(s, e) for s, e in self._current_fn_intervals
                                if s <= drag_end and e >= drag_start]
                if (committed_fp or committed_fn) and manager and sid:
                    for s, e in committed_fp:
                        manager.remove_temporal_feedback_interval(
                            concept_id=concept_id, session_id=sid,
                            start_frame=s, end_frame=e, feedback_type="false_positive",
                        )
                        removed += 1
                    for s, e in committed_fn:
                        manager.remove_temporal_feedback_interval(
                            concept_id=concept_id, session_id=sid,
                            start_frame=s, end_frame=e, feedback_type="false_negative",
                        )
                        removed += 1
                    self._refresh_probability_plot()
                else:
                    self._update_staged_ui()
                    self._draw_staged_overlays()
                if removed:
                    self._status.setText(f"Removed {removed} flag(s) in selected range.")
                else:
                    self._status.setText("No flags found in selected range.")
            if self._trace_canvas:
                self._trace_canvas.draw_idle()
            return

        # ── FP drag release ──────────────────────────────────────────────────
        if self._fp_flag_active and self._fp_drag_start is not None:
            if self._fp_drag_rect is not None:
                self._fp_drag_rect.remove()
                self._fp_drag_rect = None
            x_end = (
                float(event.xdata)
                if (event is not None and event.xdata is not None
                    and self._trace_axes is not None
                    and event.inaxes == self._trace_axes)
                else self._fp_drag_start
            )
            drag_start = int(max(0, round(min(self._fp_drag_start, x_end))))
            drag_end   = int(max(0, round(max(self._fp_drag_start, x_end))))
            self._fp_drag_start = None
            if drag_end - drag_start < 2:
                # Bare click — original single-bout behaviour
                bout = self._find_bout_at_frame(drag_start)
                if bout:
                    self._stage_fp(bout[0], bout[1])
                else:
                    self._status.setText("No detected bout at this frame. Click on a green bout region.")
            else:
                # Drag — stage every bout that overlaps the selection
                bouts_hit = [(s, e) for s, e in self._current_bouts
                             if s <= drag_end and e >= drag_start]
                if bouts_hit:
                    for s, e in bouts_hit:
                        self._stage_fp(s, e)
                    self._status.setText(f"Staged {len(bouts_hit)} bout(s) in range as false positive.")
                else:
                    self._status.setText("No detected bouts in the selected range.")
            if self._trace_canvas:
                self._trace_canvas.draw_idle()
            return

        # ── FN drag release ──────────────────────────────────────────────────
        if not self._fn_flag_active or self._fn_drag_start is None:
            return
        if event is None or event.button != 1:
            return
        # Clean up the rubber-band rectangle
        if self._fn_drag_rect is not None:
            self._fn_drag_rect.remove()
            self._fn_drag_rect = None

        x_end = float(event.xdata) if (event.xdata is not None and self._trace_axes is not None and event.inaxes == self._trace_axes) else self._fn_drag_start
        start = int(max(0, round(min(self._fn_drag_start, x_end))))
        end = int(max(0, round(max(self._fn_drag_start, x_end))))
        self._fn_drag_start = None

        if end - start < 2:
            self._status.setText("Drag a wider range to flag as false negative.")
            if self._trace_canvas:
                self._trace_canvas.draw_idle()
            return

        self._stage_fn(start, end)

    def _jump_to_session_frame(self, sid: str, frame_idx: int) -> None:
        video_path, expected_str = self._session_video_path_with_info(sid)
        if video_path is None or not video_path.exists():
            self._set_selected_probability(None)
            self._status.setText("Video source not found for selected subject/session.")
            self._show_video_not_found_dialog(sid, expected_str)
            return

        if self._loaded_session_video_id != sid:
            loaded = self._player.load_clip(str(video_path))
            if not loaded:
                self._set_selected_probability(None)
                self._status.setText("Failed to load session video.")
                self._show_video_load_failed_dialog(sid, str(video_path))
                return
            self._loaded_session_video_id = sid
            self._attach_session_pose_overlay(sid)

        clamped = int(max(0, min(int(frame_idx), max(0, self._player.n_frames - 1))))
        self._player.seek(clamped)
        subject = self._subject_by_session.get(sid, sid)
        probability = self._probability_for_frame(sid=sid, frame_idx=clamped)
        self._set_selected_probability(probability)
        if probability is None:
            self._status.setText(f"Jumped {subject} to frame {clamped}.")
        else:
            self._status.setText(f"Jumped {subject} to frame {clamped} (p={probability:.3f}).")

    def _attach_session_pose_overlay(self, sid: str) -> None:
        """Load the session's pose and feed it to the player's keypoint overlay.

        The player shows the full session video, so pose rows are indexed by the
        same absolute frame number (frame_offset=0).  Failures are non-fatal —
        the keypoint button simply stays inert when no pose is available.
        """
        try:
            x_vals, y_vals, conf_vals = self._load_session_pose(sid)
        except Exception:
            self._player.clear_pose_overlay()
            return
        if x_vals is None:
            self._player.clear_pose_overlay()
            return
        self._player.set_pose_overlay(x_vals, y_vals, conf_vals, frame_offset=0)

    def _load_session_pose(self, sid: str):
        """Return ``(x, y, conf)`` arrays for *sid*, or ``(None, None, None)``."""
        if sid in self._session_pose_cache:
            return self._session_pose_cache[sid]
        result = (None, None, None)
        if self._project_root is not None:
            manifest = self._imports.load_manifest(self._project_root)
            pose_path = (
                self._imports.pose_path_for_session(manifest, sid)
                if manifest is not None else None
            )
            if pose_path is not None and pose_path.exists():
                smoothing = getattr(manifest, "smoothing_settings", None)
                pose = self._pose.load_and_clean(pose_path, smoothing)
                result = (
                    pose.x.to_numpy(dtype=float),
                    pose.y.to_numpy(dtype=float),
                    pose.likelihood.to_numpy(dtype=float),
                )
        self._session_pose_cache[sid] = result
        return result

    def _session_video_path(self, session_id: str) -> Path | None:
        path, _ = self._session_video_path_with_info(session_id)
        return path

    def _session_video_path_with_info(self, session_id: str) -> tuple[Path | None, str]:
        """Return *(resolved_path_or_None, expected_path_str)*.

        *expected_path_str* is a human-readable hint about which path was
        looked up, used to populate the "video not found" dialog so the user
        knows exactly which file needs to be re-linked.
        """
        if self._project_root is None:
            return None, "(project not loaded)"
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return None, "(import manifest missing — re-open the project)"
        # Find the session entry
        session = next(
            (s for s in manifest.linked_sessions if s.session_id == session_id), None
        )
        if session is None:
            return None, f"(session {session_id!r} not found in manifest)"
        # Find the matching video asset
        video = next(
            (v for v in manifest.videos if v.asset_id == session.video_asset_id), None
        )
        if video is None:
            return None, f"(no video asset registered for session {session_id!r})"
        # Collect candidates in preference order
        candidates: list[str] = [
            c for c in (video.local_path, video.source_path) if c
        ]
        for raw in candidates:
            p = Path(raw)
            if p.exists():
                return p, raw
        # Return the last known path as the hint even though it doesn't exist
        hint = candidates[-1] if candidates else "(no path recorded)"
        return Path(hint), hint

    def _show_video_not_found_dialog(self, session_id: str, expected_path: str) -> None:
        """Show an informative dialog when the video file for a session cannot be found."""
        subject = self._subject_by_session.get(session_id, session_id)
        msg = QMessageBox(self)
        msg.setWindowTitle("Video File Not Found")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(
            f"The video for <b>{subject}</b> could not be found.<br><br>"
            f"<b>Expected path:</b><br><code>{expected_path}</code><br><br>"
            "This usually happens when video files have been moved, renamed, or "
            "are on a drive that is not currently connected.<br><br>"
            "<b>How to fix:</b><br>"
            "1. Go to the <b>Data Import</b> tab.<br>"
            "2. Remove the existing video entry for this subject.<br>"
            "3. Re-add the video file from its current location using "
            "<i>Add Videos</i>.<br>"
            "4. Click <b>Link &amp; Save</b> to register the new path.<br>"
            "5. Return here — clicking the timeline should now open the video."
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def _show_video_load_failed_dialog(self, session_id: str, video_path: str) -> None:
        """Show an informative dialog when the video file exists but cannot be decoded."""
        subject = self._subject_by_session.get(session_id, session_id)
        msg = QMessageBox(self)
        msg.setWindowTitle("Video Could Not Be Opened")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(
            f"The video for <b>{subject}</b> was found but could not be opened:<br><br>"
            f"<code>{video_path}</code><br><br>"
            "This can happen when:<br>"
            "\u2022 The video codec (e.g. H.265/HEVC) is not supported by the installed "
            "version of OpenCV.<br>"
            "\u2022 The video file is corrupted or partially written.<br>"
            "\u2022 The file extension does not match the actual container format.<br><br>"
            "<b>Possible fixes:</b><br>"
            "1. Re-encode the video to H.264 MP4 (e.g. using FFmpeg or HandBrake).<br>"
            "2. Ensure <code>opencv-python</code> or <code>opencv-python-headless</code> "
            "is up to date (<code>pip install -U opencv-python</code>).<br>"
            "3. Re-import the re-encoded file via the <b>Data Import</b> tab."
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def _preview_selected_bout(self) -> None:
        sid = str(self._session.currentData() or "").strip()
        if not sid:
            self._status.setText("Choose a subject/session first.")
            return

        selected = self._bouts_table.selectionModel().selectedRows()
        if not selected and self._bouts_table.rowCount() > 0:
            row_idx = 0
        elif not selected:
            self._status.setText("No bouts available for preview.")
            return
        else:
            row_idx = int(selected[0].row())

        try:
            start_item = self._bouts_table.item(row_idx, 0)
            end_item = self._bouts_table.item(row_idx, 1)
            if start_item is None or end_item is None:
                self._status.setText("Selected bout row is invalid.")
                return
            start = int(start_item.text())
            end = int(end_item.text())
        except Exception:
            self._status.setText("Selected bout row is invalid.")
            return

        _ = end
        self._jump_to_session_frame(sid=sid, frame_idx=start)
