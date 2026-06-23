"""Transfer Feedback subtab — judge how well a Direct Use run transferred.

Shows a per-subject health score with red flags and a population summary, so
the user can spot subjects whose results look untrustworthy and deep-dive on
them.  Requires the Direct Use output project's analytics to have been
refreshed first (it reads derived/analytics_cache).
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from abel.services.transfer_feedback_service import (
    SubjectFeedback,
    TransferFeedbackService,
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
    " font-weight: 700; border: none; border-radius: 5px; padding: 8px 18px; }"
    "QPushButton:hover { background: #1976D2; }"
    "QPushButton:disabled { background: #263238; color: #546E7A; }"
)

_CAT_COLOR = {"Good": "#66BB6A", "Warning": "#FFB74D", "Poor": "#EF5350"}


class TransferFeedbackTab(QWidget):
    """Assess Direct Use transfer quality per subject and across the population."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._svc = TransferFeedbackService()
        self._target_root: Path | None = None
        self._source_root: Path | None = None
        self._report = None

        header = QLabel("Transfer Feedback")
        header.setStyleSheet("font-size: 16px; font-weight: 800; color: #90CAF9;")
        desc = QLabel(
            "Estimate how well a Direct Use run transferred to the new data. "
            "Pick a Direct Use output project, then Analyze. Subjects are scored "
            "and sorted worst-first with red flags; select one to deep-dive.\n\n"
            "Tip: refresh analytics for the Direct Use project first (open it and "
            "use the Analytics tab) so the numbers are current."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #607D8B;")

        # ── Project selection row ─────────────────────────────────────
        row = QHBoxLayout()
        self._browse_btn = QPushButton("Select Direct Use Project…")
        self._browse_btn.setStyleSheet(_BTN)
        self._browse_btn.clicked.connect(self._browse_target)
        self._target_lbl = QLabel("No project selected.")
        self._target_lbl.setStyleSheet("font-size: 11px; color: #78909C;")
        self._target_lbl.setWordWrap(True)
        self._analyze_btn = QPushButton("Analyze Transfer")
        self._analyze_btn.setStyleSheet(_BTN_PRIMARY)
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.clicked.connect(self._analyze)
        row.addWidget(self._browse_btn)
        row.addWidget(self._target_lbl, 1)
        row.addWidget(self._analyze_btn)

        # ── Population summary ────────────────────────────────────────
        self._pop_lbl = QLabel("")
        self._pop_lbl.setStyleSheet(
            "font-size: 12px; color: #B0BEC5; padding: 6px 0;"
        )
        self._pop_lbl.setWordWrap(True)

        # ── Subject table ─────────────────────────────────────────────
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Subject", "Status", "Health", "Flags"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, hdr.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, hdr.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, hdr.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, hdr.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
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

        # ── Detail panel ──────────────────────────────────────────────
        self._detail = QLabel("Select a subject to see its deep-dive.")
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._detail.setStyleSheet(
            "font-size: 11px; color: #B0BEC5; background: #0A1929;"
            " border: 1px solid #1E3A5F; border-radius: 4px; padding: 8px;"
        )
        self._detail.setMinimumHeight(160)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 11px; color: #78909C;")

        body = QHBoxLayout()
        body.addWidget(self._table, 3)
        body.addWidget(self._detail, 2)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(8)
        root.addWidget(header)
        root.addWidget(desc)
        root.addLayout(row)
        root.addWidget(self._pop_lbl)
        root.addLayout(body, 1)
        root.addWidget(self._status)

    # ── Public API ─────────────────────────────────────────────────────

    def set_project(self, project_root: Path) -> None:
        """No-op: the feedback target is chosen explicitly (a DU output)."""
        pass

    def set_target_project(self, target_root: Path) -> None:
        """Pre-fill the target from a just-completed Direct Use run."""
        self._set_target(Path(target_root))

    # ── Target selection ────────────────────────────────────────────────

    def _browse_target(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Direct Use Output Project")
        if path:
            self._set_target(Path(path))

    def _set_target(self, target_root: Path) -> None:
        self._target_root = target_root
        self._source_root = self._read_source_project(target_root)
        src_txt = f"  ·  base: {self._source_root.name}" if self._source_root else ""
        self._target_lbl.setText(f"{target_root.name}{src_txt}")
        self._analyze_btn.setEnabled(True)
        self._status.setText("")

    @staticmethod
    def _read_source_project(target_root: Path) -> Path | None:
        proj = target_root / "project.yaml"
        if not proj.exists():
            return None
        try:
            import yaml  # noqa: PLC0415
            data = yaml.safe_load(proj.read_text(encoding="utf-8")) or {}
            src = str(data.get("source_project", "") or "").strip()
            if src and Path(src).exists():
                return Path(src)
        except Exception:
            pass
        return None

    # ── Analysis ────────────────────────────────────────────────────────

    def _analyze(self) -> None:
        if self._target_root is None:
            return
        cache = self._target_root / "derived" / "analytics_cache" / "analytics_cache.json"
        if not cache.exists():
            self._status.setStyleSheet("font-size: 11px; color: #FFB74D;")
            self._status.setText(
                "No analytics found. Open this project and refresh analytics in the "
                "Analytics tab first, then Analyze again."
            )
            return
        fps = self._read_fps(self._target_root)
        try:
            report = self._svc.analyze(self._target_root, self._source_root, fps=fps)
        except Exception as exc:
            logger.exception("Transfer feedback failed")
            self._status.setStyleSheet("font-size: 11px; color: #EF5350;")
            self._status.setText(f"Analysis failed: {exc}")
            return
        self._report = report
        if report.warnings:
            self._status.setStyleSheet("font-size: 11px; color: #FFB74D;")
            self._status.setText("  ".join(report.warnings))
        else:
            self._status.setStyleSheet("font-size: 11px; color: #78909C;")
            self._status.setText("")
        self._populate(report)

    @staticmethod
    def _read_fps(target_root: Path) -> float:
        proj = target_root / "project.yaml"
        if proj.exists():
            try:
                import yaml  # noqa: PLC0415
                data = yaml.safe_load(proj.read_text(encoding="utf-8")) or {}
                fps = data.get("default_fps")
                if fps:
                    return float(fps)
            except Exception:
                pass
        return 30.0

    def _populate(self, report) -> None:
        pop = report.population
        trace_note = "" if report.has_traces else "  (no probability traces — confidence checks skipped)"
        self._pop_lbl.setText(
            f"{pop.get('n_subjects', 0)} subjects  ·  "
            f"<span style='color:#EF5350'>{pop.get('n_poor', 0)} poor</span>  ·  "
            f"<span style='color:#FFB74D'>{pop.get('n_warning', 0)} warning</span>  ·  "
            f"<span style='color:#66BB6A'>{pop.get('n_good', 0)} good</span>  ·  "
            f"mean health {pop.get('mean_health', 0):.0f}/100{trace_note}"
        )
        self._table.setRowCount(len(report.subjects))
        for r, s in enumerate(report.subjects):
            name = QTableWidgetItem(s.subject)
            name.setData(Qt.ItemDataRole.UserRole, r)
            cat = QTableWidgetItem(s.category)
            cat.setForeground(QColor(_CAT_COLOR.get(s.category, "#B0BEC5")))
            health = QTableWidgetItem(f"{s.health_score:.0f}")
            flags = QTableWidgetItem(str(len(s.flags)))
            for it in (name, cat, health, flags):
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 0, name)
            self._table.setItem(r, 1, cat)
            self._table.setItem(r, 2, health)
            self._table.setItem(r, 3, flags)
        if report.subjects:
            self._table.setCurrentCell(0, 0)
        else:
            self._detail.setText("No subjects found.")

    def _refresh_detail(self) -> None:
        if self._report is None:
            return
        row = self._table.currentRow()
        if row < 0 or row >= len(self._report.subjects):
            return
        s: SubjectFeedback = self._report.subjects[row]
        lines: list[str] = []
        color = _CAT_COLOR.get(s.category, "#B0BEC5")
        lines.append(
            f"<b style='font-size:13px'>{s.subject}</b> — "
            f"<span style='color:{color}'>{s.category}</span> "
            f"(health {s.health_score:.0f}/100)"
        )
        lines.append(f"Sessions: {len(s.sessions)}")

        if s.flags:
            lines.append("<br><b>Red flags:</b>")
            for f in s.flags:
                lines.append(f"&nbsp;&nbsp;⚠ {f}")
        else:
            lines.append("<br>No red flags — looks consistent with the rest of the run.")

        c = s.confidence
        if c:
            conf_bits = []
            if "mean_conf" in c:
                conf_bits.append(f"mean confidence {c['mean_conf']:.2f}")
            if "longest_high_run_s" in c:
                conf_bits.append(f"longest high-confidence run {c['longest_high_run_s']:.0f}s")
            if "longest_low_run_s" in c:
                conf_bits.append(f"longest low-confidence run {c['longest_low_run_s']:.0f}s")
            if conf_bits:
                lines.append("<br><b>Confidence:</b> " + ", ".join(conf_bits))

        if s.behavior_metrics:
            lines.append("<br><b>Per-behaviour:</b>")
            for beh in sorted(s.behavior_metrics):
                m = s.behavior_metrics[beh]
                lines.append(
                    f"&nbsp;&nbsp;{beh}: {int(m['n_bouts'])} bouts, "
                    f"{m['time_spent_s']:.0f}s total, "
                    f"{m['mean_bout_s']:.1f}s/bout"
                )
        self._detail.setText("<br>".join(lines))
