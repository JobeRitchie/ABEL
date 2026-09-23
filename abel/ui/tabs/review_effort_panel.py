"""Review Effort subtab: how much human time the clip review in this project took.

A thin front end over :mod:`abel.validation.analyses.review_effort`, the same
measurement the Validation Platform's "Review Effort" tab runs across projects.
It only reads ``review_decisions.json``; nothing is written back.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from abel.workers.task_worker import TaskWorker

logger = logging.getLogger("abel")


def _num(value: float, fmt: str = "{:.1f}") -> str:
    try:
        return fmt.format(value) if math.isfinite(float(value)) else "-"
    except (TypeError, ValueError):
        return "-"


def _hours(value: float) -> str:
    """Hours as 'Xh Ym' so short projects do not read as 0.0 h."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(v):
        return "-"
    total_min = int(round(v * 60))
    return f"{total_min // 60}h {total_min % 60:02d}m"


class ReviewEffortPanel(QWidget):
    """Total annotation time and seconds per clip, read from the review timestamps."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_root: Path | None = None
        self._pool = QThreadPool.globalInstance()
        self._worker: TaskWorker | None = None
        self._stale = True
        self._last: tuple | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        from abel.validation.analyses import review_effort  # noqa: PLC0415

        layout = QVBoxLayout(self)

        intro = QLabel(
            "How much time did labeling take? Each clip's review time is the gap to the "
            "previous review decision. Gaps shorter than the bulk-action threshold are one "
            "UI action writing many labels, and gaps longer than the break threshold are "
            "the reviewer stepping away; neither is counted. Hours are therefore a floor; "
            "the adjusted total adds the first clip of each sitting back at the median "
            "rate. Imported labels are excluded and Temporal Review corrections are "
            "counted but not timed. Nothing is written to the project."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        box = QGroupBox("Gap thresholds")
        row = QHBoxLayout(box)
        self._batch = QDoubleSpinBox()
        self._batch.setRange(0.0, 5.0)
        self._batch.setSingleStep(0.01)
        self._batch.setDecimals(2)
        self._batch.setSuffix(" s")
        self._batch.setValue(review_effort.BATCH_SEC)
        self._batch.setToolTip("Gaps shorter than this are one bulk UI action, not a clip review.")
        self._break = QDoubleSpinBox()
        self._break.setRange(5.0, 3600.0)
        self._break.setSingleStep(30.0)
        self._break.setDecimals(0)
        self._break.setSuffix(" s")
        self._break.setValue(review_effort.BREAK_SEC)
        self._break.setToolTip("Gaps longer than this count as a break and are not charged to any clip.")
        row.addWidget(QLabel("Bulk-action below:"))
        row.addWidget(self._batch)
        row.addSpacing(16)
        row.addWidget(QLabel("Break above:"))
        row.addWidget(self._break)
        row.addStretch(1)
        self._run_btn = QPushButton("Measure Review Effort")
        self._run_btn.clicked.connect(self.measure)
        row.addWidget(self._run_btn)
        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        row.addWidget(self._export_btn)
        layout.addWidget(box)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._stats = QTableWidget(0, 2)
        self._stats.setHorizontalHeaderLabels(["Metric", "Value"])
        self._stats.verticalHeader().setVisible(False)
        self._stats.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._stats.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._stats.horizontalHeader().setStretchLastSection(True)

        self._daily = QTableWidget(0, 7)
        self._daily.setHorizontalHeaderLabels(
            ["Date", "Clips timed", "Active time", "Median s/clip", "Sittings",
             "First", "Last"])
        self._daily.verticalHeader().setVisible(False)
        self._daily.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._daily.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._daily.horizontalHeader().setStretchLastSection(True)

        tables = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Summary</b>"))
        left.addWidget(self._stats)
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>By day</b> (local time)"))
        right.addWidget(self._daily)
        tables.addLayout(left, 1)
        tables.addLayout(right, 1)
        layout.addLayout(tables, 1)

    # ------------------------------------------------------------------
    def set_project(self, project_root: Path | None) -> None:
        self._project_root = Path(project_root) if project_root else None
        self._stale = True
        self._last = None
        self._stats.setRowCount(0)
        self._daily.setRowCount(0)
        self._export_btn.setEnabled(False)
        self._status.setText("")

    def refresh_if_stale(self) -> None:
        """Measure on first view of a project so the numbers are there without a click."""
        if self._stale and self._project_root is not None and self._worker is None:
            self.measure()

    def measure(self) -> None:
        if self._project_root is None:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return
        if self._worker is not None:
            return
        root = self._project_root
        break_sec = float(self._break.value())
        batch_sec = float(self._batch.value())

        def _task() -> tuple:
            from abel.validation.analyses import review_effort  # noqa: PLC0415
            from abel.validation.datamodel import ProjectRef  # noqa: PLC0415

            result = review_effort.measure_project(
                ProjectRef.load(root), break_sec=break_sec, batch_sec=batch_sec)
            daily = review_effort.daily_breakdown(
                root, break_sec=break_sec, batch_sec=batch_sec)
            return result, daily

        self._stale = False
        self._run_btn.setEnabled(False)
        self._status.setText("Reading review decisions and measuring project video (can take ~20 s)…")
        worker = TaskWorker(_task)
        worker.signals.finished.connect(self._on_done)
        worker.signals.failed.connect(self._on_failed)
        self._worker = worker
        self._pool.start(worker)

    def _on_failed(self, msg: str) -> None:
        self._worker = None
        self._run_btn.setEnabled(True)
        logger.error("Review effort failed: %s", msg)
        self._status.setText("Could not measure review effort. See the log for details.")

    def _on_done(self, payload: object) -> None:
        self._worker = None
        self._run_btn.setEnabled(True)
        result, daily = payload  # type: ignore[misc]
        self._last = (result, daily)
        self._export_btn.setEnabled(not result.error)
        if result.error:
            self._status.setText(f"Nothing to measure: {result.error}.")
            self._stats.setRowCount(0)
            self._daily.setRowCount(0)
            return
        self._status.setText(
            f"{_hours(result.active_hours)} of active clip review, "
            f"{_num(result.median_sec)} s per clip (median), "
            f"{result.n_timed:,} clips timed.")
        self._fill_stats(result)
        self._fill_daily(daily)

    def _fill_stats(self, r) -> None:
        from abel.validation.analyses import review_effort  # noqa: PLC0415

        rows: list[tuple[str, str]] = [
            ("Active review time", _hours(r.active_hours)),
            ("Active review time (adjusted)", _hours(r.active_hours_adjusted)),
            ("Median time per clip", f"{_num(r.median_sec)} s"),
            ("Mean time per clip", f"{_num(r.mean_sec)} s"),
            ("IQR (25th-75th pct)", f"{_num(r.p25_sec)} - {_num(r.p75_sec)} s"),
            ("90th percentile", f"{_num(r.p90_sec)} s"),
            ("Clips per hour (while working)", _num(r.clips_per_hour, "{:.0f}")),
            ("Clips timed", f"{r.n_timed:,}"),
            ("Bulk-action decisions (not timed)", f"{r.n_batch:,}"),
            ("Breaks", f"{r.n_breaks:,}"),
            ("Clip-review decisions", f"{r.n_clip_review:,}"),
            ("Temporal Review corrections", f"{r.n_temporal_feedback:,}"),
            ("Imported labels (excluded)", f"{r.n_imported:,}"),
            ("Video in project", _hours(r.video_hours)),
            ("Footage actually watched", _hours(r.footage_reviewed_hours)),
            ("Share of footage watched",
             f"{_num(r.footage_reviewed_frac * 100, '{:.2f}')} %"
             if math.isfinite(r.footage_reviewed_frac) else "-"),
            ("Review minutes per video hour",
             _num(r.review_hours_per_video_hour * 60)),
        ]
        for passes in review_effort.MANUAL_PASSES:
            factor = r.saving_factor(passes)
            if math.isfinite(factor):
                rows.append((f"vs. manual scoring at {passes:g}x real-time",
                             f"{factor:.0f}x less time ({_hours(r.manual_hours(passes))})"))
        rows.append(("First decision (UTC)", r.first_decision.replace("T", " ")))
        rows.append(("Last decision (UTC)", r.last_decision.replace("T", " ")))

        self._stats.setRowCount(len(rows))
        for i, (name, value) in enumerate(rows):
            self._stats.setItem(i, 0, QTableWidgetItem(name))
            self._stats.setItem(i, 1, QTableWidgetItem(value))

    def _fill_daily(self, daily) -> None:
        self._daily.setRowCount(len(daily))
        for i, rec in enumerate(daily.itertuples(index=False)):
            cells = [
                rec.date,
                f"{int(rec.clips_timed):,}",
                _hours(rec.active_min / 60.0),
                _num(rec.median_sec),
                str(int(rec.sittings)),
                rec.first_local,
                rec.last_local,
            ]
            for j, text in enumerate(cells):
                self._daily.setItem(i, j, QTableWidgetItem(text))

    def _export(self) -> None:
        if self._last is None:
            return
        from abel.validation.analyses import review_effort  # noqa: PLC0415

        result, daily = self._last
        start = str(self._project_root or "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export review effort", str(Path(start) / "review_effort.csv"),
            "CSV files (*.csv)")
        if not path:
            return
        try:
            review_effort.results_to_frame([result]).to_csv(path, index=False)
            daily_path = Path(path).with_name(Path(path).stem + "_by_day.csv")
            daily.to_csv(daily_path, index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._status.setText(f"Exported to {path} and {daily_path.name}.")
