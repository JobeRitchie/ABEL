
from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Heavy library imports are deferred to first use for faster tab switching.
# Each accessor checks the module-level sentinel and imports on demand.
FigureCanvas: Any = None
NavigationToolbar: Any = None
Figure: Any = None
_MPL_OK: bool | None = None  # None = not yet checked

ttest_ind: Any = None
f_oneway: Any = None
_SCIPY_OK: bool | None = None

cv2: Any = None
_CV2_OK: bool | None = None


def _ensure_matplotlib() -> bool:
    """Import matplotlib on first call; subsequent calls are instant."""
    global FigureCanvas, NavigationToolbar, Figure, _MPL_OK  # noqa: PLW0603
    if _MPL_OK is not None:
        return _MPL_OK
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qt import NavigationToolbar2QT
        from matplotlib.figure import Figure as _Fig
        FigureCanvas = FigureCanvasQTAgg
        NavigationToolbar = NavigationToolbar2QT
        Figure = _Fig
        _MPL_OK = True
    except Exception:
        _MPL_OK = False
    return _MPL_OK


def _ensure_scipy() -> bool:
    global ttest_ind, f_oneway, _SCIPY_OK  # noqa: PLW0603
    if _SCIPY_OK is not None:
        return _SCIPY_OK
    try:
        from scipy.stats import ttest_ind as _t, f_oneway as _f
        ttest_ind = _t
        f_oneway = _f
        _SCIPY_OK = True
    except Exception:
        _SCIPY_OK = False
    return _SCIPY_OK


def _ensure_cv2() -> bool:
    global cv2, _CV2_OK  # noqa: PLW0603
    if _CV2_OK is not None:
        return _CV2_OK
    try:
        import cv2 as _cv2
        cv2 = _cv2
        _CV2_OK = True
    except Exception:
        _CV2_OK = False
    return _CV2_OK

from PySide6.QtCore import Qt, QObject, QThreadPool, QTimer, QMimeData, QEvent, Signal
from PySide6.QtGui import QAction, QDrag, QKeySequence, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.core.project_manager import ProjectManager
from abel.services.behavior_service import BehaviorService
from abel.services.behavioral_motif_service import (
    MotifSettings,
    load_motif_settings,
    save_motif_settings,
)
from abel.services.import_service import ImportService
from abel.services.project_merge_service import ProjectMergeService
from abel.services.pose_processing_service import PoseProcessingService
from abel.services.roi_service import ROIService
from abel.workers.task_worker import TaskWorker

logger = logging.getLogger("abel")

NO_BEHAVIOR_ID = "no_behavior"
DISTANCE_BEHAVIOR_ID = "__distance_traveled__"
DISTANCE_BEHAVIOR_NAME = "Distance Traveled"

# ROI occupancy pseudo-behaviors.  Like Distance Traveled, these are synthetic
# rows that don't come from behavior inference; they reuse the standard summary
# columns (time_spent_s = time in zone, n_bouts = entries, mean_bout_s = mean
# visit duration, latency_s = time to first entry).  One per configured zone.
ROI_BEHAVIOR_PREFIX = "__roi_in_zone__"


def roi_behavior_id(zone_index: int) -> str:
    """Pseudo-behavior id for a 1-based ROI zone index."""
    return f"{ROI_BEHAVIOR_PREFIX}{zone_index}"


def roi_behavior_name(zone_index: int, roi_count: int) -> str:
    """Display name for an ROI pseudo-behavior."""
    return "Time in ROI" if roi_count <= 1 else f"Time in ROI {zone_index}"


def is_roi_behavior_id(bid: str) -> bool:
    return str(bid).startswith(ROI_BEHAVIOR_PREFIX)


def is_pseudo_behavior_id(bid: str) -> bool:
    """True for synthetic rows (distance / ROI) that have no raw bout data."""
    return bid == DISTANCE_BEHAVIOR_ID or is_roi_behavior_id(bid)


def _debounce_bool(mask: "np.ndarray", min_run: int) -> "np.ndarray":
    """Merge runs shorter than *min_run* frames into the preceding run.

    Suppresses single-frame flicker at an ROI boundary so tracking jitter
    doesn't inflate the entry count.  Processes runs left-to-right so merges
    propagate; the first run is left untouched (nothing precedes it).
    """
    if min_run <= 1 or mask.size == 0:
        return mask
    out = mask.copy()
    change_idx = np.flatnonzero(np.diff(out.astype(np.int8))) + 1
    bounds = np.concatenate(([0], change_idx, [out.size]))
    for i in range(len(bounds) - 1):
        s, e = int(bounds[i]), int(bounds[i + 1])
        if (e - s) < min_run and s > 0:
            out[s:e] = out[s - 1]
    return out

# Sentinel for distinguishing "not yet cached" from "cached value of None".
_MANIFEST_UNSET = object()

# Subsampling rate for distance calculation.  Using every frame over-counts
# distance due to per-frame tracking jitter.  5 Hz is a well-established
# default for rodent locomotion analysis (cf. Mathis et al. 2018) that
# effectively filters high-frequency noise while preserving real displacement.
_DISTANCE_SUBSAMPLE_HZ: float = 5.0

class _SubjectSelectorDialog(QDialog):
    """Dialog for robust subject selection with checkboxes and group fields."""
    def __init__(self, parent, subjects, checked_subjects, subject_groups):
        super().__init__(parent)
        self.setWindowTitle("Select Subjects for Heatmap")
        self.resize(400, 500)
        self._subject_widgets = {}
        layout = QVBoxLayout(self)
        hint = QLabel("Check subjects to include in the heatmap and composite background. Optionally assign group labels.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        vbox = QVBoxLayout(inner)
        for subj in sorted(subjects):
            row = QHBoxLayout()
            cb = QCheckBox(subj)
            cb.setChecked(subj in checked_subjects)
            group_edit = QLineEdit()
            group_edit.setPlaceholderText("Group\u2026")
            group_edit.setMaximumWidth(120)
            group_edit.setText(subject_groups.get(subj, ""))
            row.addWidget(cb)
            row.addStretch(1)
            row.addWidget(group_edit)
            w = QWidget()
            w.setLayout(row)
            vbox.addWidget(w)
            self._subject_widgets[subj] = (cb, group_edit)
        vbox.addStretch(1)
        inner.setLayout(vbox)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)
        btn_row = QHBoxLayout()
        check_all_btn = QPushButton("Check All")
        check_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb, _ in self._subject_widgets.values()])
        uncheck_all_btn = QPushButton("Uncheck All")
        uncheck_all_btn.clicked.connect(lambda: [cb.setChecked(False) for cb, _ in self._subject_widgets.values()])
        btn_row.addWidget(check_all_btn)
        btn_row.addWidget(uncheck_all_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selection(self):
        checked = set()
        groups = {}
        for subj, (cb, group_edit) in self._subject_widgets.items():
            if cb.isChecked():
                checked.add(subj)
            group = group_edit.text().strip()
            if group:
                groups[subj] = group
        return checked, groups


class _SubjectPrechopDialog(QDialog):
    """Dialog for configuring per-subject analysis prechop frame offsets."""

    def __init__(self, parent: QWidget, subjects: list[str], current: dict[str, int]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Per-Subject Prechop")
        self.resize(500, 460)
        self._spin_by_subject: dict[str, QSpinBox] = {}

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Set how many leading frames to ignore for each subject in analytics.\n"
            "Frame and time axes are shifted so analysis starts after this prechop.\n"
            "Boutframe export and labeled-video export are not affected."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._table = QTableWidget(len(subjects), 2, self)
        self._table.setHorizontalHeaderLabels(["Subject", "Prechop Frames"])
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        for row_idx, subject in enumerate(subjects):
            subj_item = QTableWidgetItem(subject)
            subj_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._table.setItem(row_idx, 0, subj_item)

            spin = QSpinBox(self._table)
            spin.setRange(0, 10_000_000)
            spin.setSingleStep(10)
            spin.setValue(max(0, int(current.get(subject, 0))))
            self._table.setCellWidget(row_idx, 1, spin)
            self._spin_by_subject[subject] = spin

        layout.addWidget(self._table, 1)

        btn_row = QHBoxLayout()
        zero_all_btn = QPushButton("Reset All to 0")
        zero_all_btn.clicked.connect(self._reset_all)
        btn_row.addWidget(zero_all_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _reset_all(self) -> None:
        for spin in self._spin_by_subject.values():
            spin.setValue(0)

    def offsets(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for subject, spin in self._spin_by_subject.items():
            val = int(spin.value())
            if val > 0:
                out[subject] = val
        return out

# ======================================================================
# Palette used across sub-tabs
# ======================================================================
_PALETTE = [
    "#5b9bd5", "#ed7d31", "#70ad47", "#ffc000", "#a855f7",
    "#06b6d4", "#f43f5e", "#6366f1", "#84cc16", "#ec4899",
]


# ======================================================================
# Faceted grouping controls (FecalFinder-style)
# ======================================================================
# Per-factor dropdowns that each combine / split / filter-to-a-level, driving
# how sessions are split into plot series. Replaces the older single
# "Group by" combo + "Groups" checklist. The control values below double as
# the userData stored on each combo item.
FACET_COMBINE = "__combine__"   # pool across this factor (ignore it)
FACET_SPLIT = "__split__"       # one series per level of this factor
FACET_COMBINE_LABEL = "— combine —"
FACET_SPLIT_LABEL = "— split —"


def facet_session_labels(
    session_factors: dict[str, dict[str, str]],
    factor_definitions: list[str],
    controls: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Map each session label to a series label per the facet *controls*.

    ``controls`` maps a factor name to ``FACET_COMBINE`` (pool), ``FACET_SPLIT``
    (one series per level), or a specific level string (filter to that level).
    The series label is the ``×``-joined cross-product of the SPLIT factors'
    levels (in ``factor_definitions`` order), or ``"All"`` when nothing is split.

    Returns ``(session_label -> series_label, split_factor_names)``. Sessions
    that fail a specific-level filter, or that lack a level for a SPLIT factor,
    are omitted from the mapping.
    """
    split = [f for f in factor_definitions if controls.get(f) == FACET_SPLIT]
    out: dict[str, str] = {}
    for label, facs in session_factors.items():
        keep = True
        for factor, mode in controls.items():
            if mode in (FACET_COMBINE, FACET_SPLIT):
                continue
            if (facs.get(factor, "") or "") != mode:  # specific-level filter
                keep = False
                break
        if not keep:
            continue
        if split:
            parts = [facs.get(f, "") for f in split]
            if not all(parts):
                continue  # session missing a level for a split factor
            out[label] = " × ".join(parts)
        else:
            out[label] = "All"
    return out, split


class _FacetControls(QWidget):
    """A compact column of per-factor dropdowns (combine / split / level).

    Emits :attr:`changed` whenever any dropdown changes. Selections survive a
    :meth:`rebuild` when the factor and its levels still exist.
    """

    changed = Signal()

    def __init__(self, header: str = "Group by:") -> None:
        super().__init__()
        self._vbox = QVBoxLayout(self)
        self._vbox.setSpacing(2)
        self._vbox.setContentsMargins(0, 0, 0, 0)
        self._combos: dict[str, QComboBox] = {}
        self._sig: tuple = ()  # (factors, levels) signature to skip no-op rebuilds
        self._header = QLabel(header)
        self._header.setStyleSheet("color:#90a4ae;font-size:10px;")
        self._vbox.addWidget(self._header)
        self._empty = QLabel("Define factors in the Summary tab to group.")
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet("color:#607d8b;font-size:10px;")
        self._vbox.addWidget(self._empty)

    def rebuild(
        self,
        factors: list[str],
        levels_by_factor: dict[str, list[str]],
        force: bool = False,
    ) -> None:
        """Rebuild one dropdown per factor, restoring prior selections.

        Skips the teardown when the factor/level structure is unchanged so it
        can be called on every redraw without churning the widgets.
        """
        sig = tuple((f, tuple(levels_by_factor.get(f, []))) for f in factors)
        if not force and sig == self._sig and self._combos:
            return
        self._sig = sig
        prev = self.state()
        # Tear down existing factor rows (keep header + empty hint).
        while self._vbox.count() > 2:
            item = self._vbox.takeAt(2)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            else:
                lay = item.layout()
                if lay is not None:
                    while lay.count():
                        sub = lay.takeAt(0)
                        sw = sub.widget()
                        if sw is not None:
                            sw.deleteLater()
        self._combos = {}
        self._empty.setVisible(not factors)

        for factor in factors:
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl = QLabel(f"{factor}:")
            lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
            lbl.setMinimumWidth(60)
            combo = QComboBox()
            combo.addItem(FACET_COMBINE_LABEL, FACET_COMBINE)
            combo.addItem(FACET_SPLIT_LABEL, FACET_SPLIT)
            for lvl in levels_by_factor.get(factor, []):
                combo.addItem(lvl, lvl)
            combo.setStyleSheet(
                "QComboBox{background:#0A1929;border:1px solid #1E3A5F;"
                "border-radius:3px;color:#cfd8dc;font-size:10px;padding:1px 3px;}"
            )
            # Keep the box a modest, fixed width instead of stretching to fill
            # the whole control panel (which pushed the right edge under the plot
            # area and got clipped). The popup list still expands to show the full
            # level names, so long labels stay readable.
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(8)
            combo.setMaximumWidth(190)
            want = prev.get(factor, FACET_COMBINE)
            idx = combo.findData(want)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
            row.addWidget(lbl)
            row.addWidget(combo)
            row.addStretch(1)
            self._combos[factor] = combo
            self._vbox.addLayout(row)

    def state(self) -> dict[str, str]:
        return {f: str(c.currentData()) for f, c in self._combos.items()}

    def set_state(self, controls: dict[str, str]) -> None:
        for factor, combo in self._combos.items():
            want = controls.get(factor, FACET_COMBINE)
            combo.blockSignals(True)
            idx = combo.findData(want)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)


# ======================================================================
# Dynamic canvas resize helper
# ======================================================================
class _ViewportResizeFilter(QObject):
    """Qt event filter installed on a QScrollArea viewport.

    When the viewport receives a Resize event the *callback* is invoked
    with ``(new_width, new_height)`` in device pixels.  A zero-delay
    ``QTimer.singleShot`` is used so the callback fires after Qt has
    finished the resize pass, avoiding recursive layout updates.
    """

    def __init__(self, callback: "Any", parent: "QObject | None" = None) -> None:
        super().__init__(parent)
        self._cb = callback
        self._pending = False

    def eventFilter(self, obj: "QObject", event: "QEvent") -> bool:  # type: ignore[override]
        from PySide6.QtCore import QEvent as _QE
        if event.type() == _QE.Type.Resize and not self._pending:
            self._pending = True
            QTimer.singleShot(0, self._fire)
        return False

    def _fire(self) -> None:
        self._pending = False
        try:
            self._cb()
        except Exception:
            pass


def _legend_right_margin(labels: list, fig_width_px: int = 700) -> tuple:
    """Compute (rect_right, legend_x) so an external legend fits within figure bounds.

    rect_right — right edge for tight_layout's rect parameter (axes stay left of this).
    legend_x   — x-anchor for figure.legend bbox_to_anchor (figure coordinates).

    The legend zone is fixed at a physical pixel width so it does not shrink
    when the figure is wide — the axes simply get more room.
    """
    max_len = max((len(str(s)) for s in labels), default=0)
    # Approximate legend width in pixels: swatch (22px) + char width + padding.
    # At 8pt font, average char width ~5.5px, plus 24px padding.
    legend_px = max(80, int(22 + max_len * 5.5 + 24))
    # Reserve a fixed pixel strip on the right; clamp so axes always get ≥ 40%
    legend_frac = min(0.45, legend_px / max(fig_width_px, 200))
    legend_frac = max(0.12, legend_frac)
    rect_right  = round(1.0 - legend_frac - 0.01, 3)
    rect_right  = max(0.45, rect_right)
    legend_x    = rect_right + 0.01
    return rect_right, legend_x


def _eb_val(vals: "np.ndarray", style: str) -> float:
    """Return the error bar half-height for *vals* according to *style*.

    style: "SEM" | "SD" | "95% CI" | "None"
    Returns 0.0 when style is "None" or there is only one value.
    """
    if style == "None" or len(vals) < 2:
        return 0.0
    std = float(np.array(vals, dtype=float).std(ddof=1))
    n   = len(vals)
    if style == "SD":
        return std
    if style == "95% CI":
        return 1.96 * std / np.sqrt(n)
    # Default: SEM
    return std / np.sqrt(n)


def _force_fit_canvas(canvas: Any, fig: Any, max_w: int, max_h: int, dpi: int = 100) -> None:
    """Resize *canvas* to tightly match the rendered content bounding box,
    bounded by *max_w* × *max_h*.  Call after drawing and tight_layout."""
    try:
        renderer = canvas.get_renderer()
        bbox = fig.get_tightbbox(renderer)   # in inches
        if bbox is not None:
            fit_w = max(80,  min(max_w, int(bbox.width  * dpi) + 12))
            fit_h = max(60,  min(max_h, int(bbox.height * dpi) + 12))
            canvas.setFixedSize(fit_w, fit_h)
            fig.set_size_inches(fit_w / dpi, fit_h / dpi)
    except Exception:
        pass


def _vbar_extent(scroll: Any) -> int:
    """Vertical-scrollbar thickness in pixels for *scroll* (style-dependent)."""
    try:
        sb = scroll.verticalScrollBar()
        w = sb.sizeHint().width() if sb is not None else 0
        if w <= 0:
            from PySide6.QtWidgets import QStyle
            w = scroll.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        return max(0, int(w))
    except Exception:
        return 16


def _stable_fill_width(scroll: Any, content_h_at: Any, min_w: int = 200) -> int:
    """Return a fill-width for *scroll*'s viewport that will not oscillate.

    ``viewport().width()`` shrinks by the scrollbar thickness whenever a vertical
    scrollbar toggles; pairing that with an aspect-locked height makes the canvas
    flip-flop forever at the threshold where the scrollbar appears (the "can't
    decide if it wants a scrollbar" shimmer).  We instead read
    ``maximumViewportSize()`` — the viewport size assuming *no* scrollbars,
    independent of the current scrollbar state — and reserve the scrollbar gutter
    up-front, deterministically, when the content would overflow the viewport.
    Because the decision is made against the scrollbar-free width it is stable:
    one pass settles instead of toggling forever.

    *content_h_at* is a callable mapping a candidate width to the content height
    it would produce; it decides whether a vertical scrollbar is needed.
    """
    try:
        mvs = scroll.maximumViewportSize()
        full_w = max(min_w, int(mvs.width()))
        full_h = max(1, int(mvs.height()))
        if int(content_h_at(full_w)) > full_h:
            return max(min_w, full_w - _vbar_extent(scroll))
        return full_w
    except Exception:
        try:
            return max(min_w, scroll.viewport().width())
        except Exception:
            return min_w


def _autofill_canvas(
    canvas_scroll: Any,
    canvas: Any,
    figure: Any,
    *,
    dpi: int = 100,
    min_w: int = 200,
    preserve_aspect: bool = True,
    min_h: int = 120,
    max_h: int | None = None,
) -> None:
    """Scale a matplotlib *canvas* to fill the width of its enclosing
    *canvas_scroll* viewport.

    Generalises ``_GraphsWidget._sync_canvas_to_viewport`` so every analytics
    subtab can share one dynamic-resize behaviour.  The content aspect ratio is
    taken from the canvas's *current* width/height (each draw routine sets that
    from the figure content), so spatial heatmaps and grid plots keep their
    proportions instead of being stretched.  When the resulting height exceeds
    the viewport the scroll area shows a vertical scrollbar.

    No-op when matplotlib is unavailable or the widgets are not yet built.
    """
    if canvas_scroll is None or canvas is None or figure is None:
        return
    try:
        if preserve_aspect:
            # Aspect comes from the figure's current size — draw routines set
            # this to the content-appropriate dimensions, so the proportion is
            # honoured whether we are called right after a draw or on a plain
            # viewport resize (the figure keeps its last-synced aspect).
            fig_w_in = float(figure.get_figwidth()) or 1.0
            fig_h_in = float(figure.get_figheight()) or 1.0
            aspect = fig_h_in / fig_w_in

            def _content_h(w: float) -> float:
                h = w * aspect
                if max_h is not None and max_h > min_h:
                    h = min(h, float(max_h))
                return h
        else:
            # Fill width but keep the figure's *designed* pixel height, so a wide
            # viewport stretches width without amplifying height (used by tall,
            # row-based figures like the engram view).
            _fixed_h = max(min_h, int(round(float(figure.get_figheight()) * dpi)))
            if max_h is not None and max_h > min_h:
                _fixed_h = min(_fixed_h, int(max_h))

            def _content_h(w: float) -> float:
                return _fixed_h
        # Reserve the scrollbar gutter deterministically (see _stable_fill_width)
        # so the canvas settles instead of oscillating at the scrollbar threshold.
        new_w = _stable_fill_width(canvas_scroll, _content_h, min_w=min_w)
        new_h = max(min_h, int(round(_content_h(new_w))))
        figure.set_size_inches(new_w / dpi, new_h / dpi)
        canvas.setFixedSize(new_w, new_h)
        canvas.updateGeometry()
        # Re-run layout at the final size so axis/tick labels reflow instead of
        # overlapping (set_size_inches alone invalidates the earlier tight_layout).
        try:
            figure.tight_layout()
        except Exception:
            pass
    except Exception:
        pass


# ======================================================================
# Top-level host widget
# ======================================================================

class BehaviorAnalyticsTab(QWidget):
    """Summarise bout counts / durations, visualise graphs, and render
    spatial heatmaps — organised across three sub-tabs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # -- shared state ---------------------------------------------
        self._project_root: Path | None = None
        self._manager: ProjectManager | None = None
        self._behaviors = BehaviorService()
        self._imports = ImportService()
        self._pose = PoseProcessingService()
        self._roi_service = ROIService()
        self._subject_by_session: dict[str, str] = {}
        self._session_by_subject: dict[str, list[str]] = {}
        self._session_type_by_session: dict[str, str] = {}
        self._session_label_by_session: dict[str, str] = {}
        self._sessions_by_label: dict[str, list[str]] = {}
        self._summary_rows: list[dict[str, Any]] = []
        self._session_groups: dict[str, str] = {}
        # Multi-factor group assignment: each session can have multiple factors
        self._factor_definitions: list[str] = []   # ordered factor names
        self._session_factors: dict[str, dict[str, str]] = {}  # session_label → {factor: level}
        self._active_grouping_factor: str = ""  # derived: single split factor / __interaction__
        # FecalFinder-style facet controls: factor -> FACET_COMBINE / FACET_SPLIT / level.
        self._facet_controls: dict[str, str] = {}
        self._facet_split_factors: list[str] = []  # factors currently set to split
        self._subject_order: list[str] = []  # user-defined subject/session order
        self._subject_prechop_frames: dict[str, int] = {}  # subject -> analysis prechop frames
        self._session_prechop_overrides: dict[str, int] = {}  # session_id -> analysis prechop frames
        self._session_end_s_overrides: dict[str, float] = {}  # session_id -> end seconds (merged)
        self._group_order: list[str] = []  # user-defined group display order
        self._factor_level_order: dict[str, list[str]] = {}  # factor → ordered levels
        self._group_colors: dict[str, str] = {}  # group name → hex color override
        self._raw_bouts: dict[str, pd.DataFrame] = {}
        self._last_stats_result: dict[str, Any] = {}  # populated by stats dialog
        # Merged external projects
        self._merge_service = ProjectMergeService()
        self._pose_cache: dict[str, Any] = {}  # session_id → PoseData
        self._pose_vel_cache: dict[str, "np.ndarray"] = {}  # session_id → centroid_velocity array
        # Per-refresh caches — invalidated at the start of each _refresh call.
        self._manifest_cache: Any = _MANIFEST_UNSET  # ImportManifest | None
        self._fps_cache: float | None = None
        self._tr_bouts_cache: dict[str, pd.DataFrame] | None = None
        self._graph_settings: dict[str, Any] = {
            "title_fontsize": 12,
            "axis_fontsize": 10,
            "tick_fontsize": 8,
            "legend_fontsize": "small",
            "legend_loc": "best",
            "dpi": 150,
            "fig_bg": "#ffffff",
            "error_style": "SEM",       # "SEM" | "SD" | "95% CI" | "None"
            "bar_spacing": 1.0,          # bar-width multiplier (0.3 – 2.0)
            "eb_capsize": 4,             # error bar cap width in points
            "eb_linewidth": 1.0,         # error bar line thickness in points
            "force_fit": True,           # auto-resize canvas to content
            "show_indiv_points": True,
            "show_stats": True,          # overlay significance brackets/stars on charts
            "max_w": 700,
            "max_h": 420,
            "scale": 100,               # canvas scale % (50-200)
        }

        # -- status bar (shared) --------------------------------------
        self._status = QLabel("Open a project and run temporal refinement to view analytics.")
        self._status.setWordWrap(True)

        # -- global controls ------------------------------------------
        self._refresh_btn = QPushButton("Refresh Analytics")
        self._refresh_btn.clicked.connect(self._refresh)

        self._clear_btn = QPushButton("Clear Display")
        self._clear_btn.setToolTip(
            "Clear displayed analytics without reloading from disk.\n"
            "Use between test runs or after switching sessions."
        )
        self._clear_btn.clicked.connect(self._clear_display)

        self._behavior_filter_btn = QPushButton("All behaviors ▾")
        self._behavior_filter_btn.setToolTip("Select which behaviors to include in analytics.")
        self._behavior_filter_menu = QMenu(self)
        self._behavior_filter_btn.setMenu(self._behavior_filter_menu)
        self._behavior_filter_actions: list[tuple[str, str, QAction]] = []  # (bid, label, action)

        self._subject_prechop_btn = QPushButton("Per-Subject Prechop...")
        self._subject_prechop_btn.setToolTip(
            "Set per-subject frame offsets so analytics starts after an initial pre-period."
        )
        self._subject_prechop_btn.clicked.connect(self._open_subject_prechop_dialog)

        top_row = QHBoxLayout()
        top_row.addWidget(self._refresh_btn)
        top_row.addWidget(self._clear_btn)
        top_row.addWidget(QLabel("Behaviors:"))
        top_row.addWidget(self._behavior_filter_btn)
        top_row.addWidget(self._subject_prechop_btn)

        # -- Merge Projects button (opens a popup panel) --------------
        self._merge_btn = QPushButton("\u229e Merge Projects\u2026")
        self._merge_btn.setToolTip(
            "Import bout data from other ABEL projects to analyse\n"
            "them together with the current project (e.g. across cohorts)."
        )
        self._merge_btn.setCheckable(True)
        self._merge_btn.toggled.connect(self._toggle_merge_panel)
        top_row.addWidget(self._merge_btn)
        top_row.addStretch(1)

        # Merge panel (hidden by default; shown below top_row when button toggled)
        self._merge_panel = QWidget()
        self._merge_panel.setVisible(False)
        self._merge_panel.setStyleSheet(
            "QWidget{background:#0d1b2a;border:1px solid #1E3A5F;border-radius:4px;}"
        )
        merge_inner = QVBoxLayout(self._merge_panel)
        merge_inner.setSpacing(4)
        merge_inner.setContentsMargins(8, 6, 8, 6)

        merge_btn_row = QHBoxLayout()
        self._merge_add_btn = QPushButton("Add Project\u2026")
        self._merge_add_btn.setToolTip("Pick an ABEL project folder to merge in.")
        self._merge_add_btn.clicked.connect(self._merge_add_project)
        self._merge_remove_btn = QPushButton("Remove Selected")
        self._merge_remove_btn.clicked.connect(self._merge_remove_selected)
        merge_btn_row.addWidget(self._merge_add_btn)
        merge_btn_row.addWidget(self._merge_remove_btn)
        merge_btn_row.addStretch(1)
        merge_inner.addLayout(merge_btn_row)

        self._merge_list = QListWidget()
        self._merge_list.setMaximumHeight(80)
        self._merge_list.setToolTip(
            "Each entry shows: Tag  |  Path  |  Group override (if any).\n"
            "Double-click a row to edit its tag or group override."
        )
        self._merge_list.itemDoubleClicked.connect(self._merge_edit_item)
        merge_inner.addWidget(self._merge_list)

        merge_note = QLabel(
            "Sessions from merged projects are prefixed with their tag. "
            "Set a \u2018Group override\u2019 to assign all imported sessions to a specific group."
        )
        merge_note.setWordWrap(True)
        merge_note.setStyleSheet(
            "color:#90a4ae;font-size:10px;background:transparent;border:none;"
        )
        merge_inner.addWidget(merge_note)

        self._merge_use_cache_chk = QCheckBox("Use cached analytics from merged projects (if available)")
        self._merge_use_cache_chk.setChecked(True)
        self._merge_use_cache_chk.setToolTip(
            "When checked, each merged project's pre-computed analytics cache\n"
            "(derived/analytics_cache/) is used directly, skipping the slower\n"
            "bout re-computation from inference traces.\n"
            "Uncheck if a merged project's temporal settings have changed since\n"
            "its cache was last generated."
        )
        merge_inner.addWidget(self._merge_use_cache_chk)


        self._tabs = QTabWidget()
        self._summary_tab = _SummaryStatsWidget(self)
        self._graphs_tab = _GraphsWidget(self)
        self._heatmap_tab = _HeatmapWidget(self)
        self._density_tab = _DensityAnalysisWidget(self)
        self._relationships_tab = _BehaviorMotifWidget(self)
        self._sections_tab = _SessionSectionsWidget(self)
        self._velocity_tab = _VelocityWidget(self)
        self._social_tab = _SocialInteractionWidget(self)
        self._tabs.addTab(self._summary_tab, "Summary && Statistics")
        self._tabs.addTab(self._graphs_tab, "Graphs")
        self._tabs.addTab(self._heatmap_tab, "Spatial Heatmap")
        self._tabs.addTab(self._density_tab, "Density Analysis")
        self._tabs.addTab(self._relationships_tab, "Behavior Relationships")
        self._tabs.addTab(self._sections_tab, "Session Sections")
        self._tabs.addTab(self._velocity_tab, "Velocity")
        self._tabs.addTab(self._social_tab, "Social Interaction")

        root = QVBoxLayout(self)
        root.setSpacing(2)
        root.addLayout(top_row)
        root.addWidget(self._merge_panel)
        root.addWidget(self._status)
        root.addWidget(self._tabs, 1)

    def _toggle_merge_panel(self, checked: bool) -> None:
        self._merge_panel.setVisible(checked)
        self._merge_btn.setText(
            "\u229f Merge Projects\u2026" if checked else "\u229e Merge Projects\u2026"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._manager = ProjectManager(project_root)
        self._session_groups.clear()
        self._factor_definitions.clear()
        self._session_factors.clear()
        self._active_grouping_factor = ""
        self._facet_controls.clear()
        self._facet_split_factors.clear()
        self._subject_order.clear()
        self._subject_prechop_frames.clear()
        self._session_prechop_overrides.clear()
        self._session_end_s_overrides.clear()
        self._group_order.clear()
        self._factor_level_order.clear()
        self._group_colors.clear()
        self._pose_cache.clear()
        self._pose_vel_cache.clear()
        self._manifest_cache = _MANIFEST_UNSET
        self._fps_cache = None
        self._tr_bouts_cache = None
        # Persist previous merged-project list then reload for new project
        self._merge_service.clear()
        self._status.setText("Loading analytics\u2026")
        # Defer ALL I/O to avoid blocking the UI thread during tab switch.
        QTimer.singleShot(0, self._deferred_project_init)

    def _deferred_project_init(self) -> None:
        """Run all project I/O off the initial paint cycle."""
        if self._project_root is None:
            return
        self._behaviors.set_project(self._project_root)
        self._subject_by_session = self._build_subject_map()
        self._session_by_subject = self._invert_subject_map()
        self._load_group_state()
        self._graphs_tab._refresh_factor_selector()
        self._graphs_tab._refresh_until_behavior_combo()
        self._refresh_behavior_filter()
        self._heatmap_tab._refresh_lists()
        self._density_tab.refresh_selectors()
        self._relationships_tab.set_project(self._project_root)
        self._social_tab.on_project_reloaded()
        # Restore any previously merged projects for this project folder
        self._merge_service.load(self._project_root)
        self._rebuild_merge_list_widget()
        self._status.setText(
            "Project loaded. Click \u201cRefresh Analytics\u201d to load data."
        )

    # ------------------------------------------------------------------
    # Group state persistence
    # ------------------------------------------------------------------

    def _save_group_state(self) -> None:
        """Persist group assignments to {project_root}/derived/analytics_groups.json."""
        if self._project_root is None:
            return
        state = {
            "schema_version": "1.0",
            "factor_definitions": list(self._factor_definitions),
            "session_factors": {k: dict(v) for k, v in self._session_factors.items()},
            "active_grouping_factor": self._active_grouping_factor,
            "facet_controls": dict(self._facet_controls),
            "subject_order": list(self._subject_order),
            "subject_prechop_frames": dict(self._subject_prechop_frames),
            "group_order": list(self._group_order),
            "factor_level_order": {k: list(v) for k, v in self._factor_level_order.items()},
            "group_colors": dict(self._group_colors),
            "section_definitions": self._sections_tab.get_sections_state(),
            "section_custom_presets": self._sections_tab.get_custom_presets(),
        }
        out_path = self._project_root / "derived" / "analytics_groups.json"
        try:
            out_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("Failed to save analytics group state.")

    def _load_group_state(self) -> None:
        """Restore group assignments from {project_root}/derived/analytics_groups.json."""
        if self._project_root is None:
            return
        state_path = self._project_root / "derived" / "analytics_groups.json"
        if not state_path.exists():
            return
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load analytics group state.")
            return
        self._factor_definitions[:] = [str(f) for f in (state.get("factor_definitions") or [])]
        self._session_factors.clear()
        for label, facs in (state.get("session_factors") or {}).items():
            if isinstance(facs, dict):
                self._session_factors[str(label)] = {str(k): str(v) for k, v in facs.items()}
        self._active_grouping_factor = str(state.get("active_grouping_factor") or "")
        self._facet_controls.clear()
        loaded_controls = state.get("facet_controls")
        if isinstance(loaded_controls, dict):
            self._facet_controls = {str(k): str(v) for k, v in loaded_controls.items()}
        elif self._active_grouping_factor:
            # Migrate older projects that only stored a single active factor.
            self._facet_controls = self._session_groups_factor_to_controls(
                self._active_grouping_factor
            )
        self._subject_order[:] = [str(s) for s in (state.get("subject_order") or [])]
        self._subject_prechop_frames.clear()
        for subject, frames in (state.get("subject_prechop_frames") or {}).items():
            try:
                val = max(0, int(frames))
            except Exception:
                continue
            if val > 0:
                self._subject_prechop_frames[str(subject)] = val
        self._group_order[:] = [str(g) for g in (state.get("group_order") or [])]
        self._factor_level_order.clear()
        for f, levels in (state.get("factor_level_order") or {}).items():
            if isinstance(levels, list):
                self._factor_level_order[str(f)] = [str(l) for l in levels]
        self._group_colors.clear()
        for g, c in (state.get("group_colors") or {}).items():
            self._group_colors[str(g)] = str(c)
        self._sync_session_groups()
        sec_defs = state.get("section_definitions")
        if sec_defs:
            self._sections_tab.set_sections_state(sec_defs)
        custom_presets = state.get("section_custom_presets")
        if custom_presets:
            self._sections_tab.set_custom_presets(custom_presets)

    def _open_level_order_dialog(self) -> None:
        """Open a dialog to set the display order of levels within each factor."""
        if not self._factor_definitions:
            QMessageBox.information(
                self, "No Factors",
                "No factors are defined yet. Add factors in the Summary tab first.",
            )
            return

        # Collect all unique levels per factor from session assignments
        all_levels: dict[str, set[str]] = {f: set() for f in self._factor_definitions}
        for facs in self._session_factors.values():
            for f, level in facs.items():
                if f in all_levels and level:
                    all_levels[f].add(level)

        if not any(all_levels.values()):
            QMessageBox.information(
                self, "No Levels",
                "No factor levels have been assigned to any session yet.",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Factor Level Order")
        dlg.resize(380, min(100 + 150 * len(self._factor_definitions), 640))
        main_vbox = QVBoxLayout(dlg)
        hint = QLabel(
            "Set the display order of levels within each factor. "
            "This order applies to all charts — including the 'All Factors (interaction)' mode, "
            "where groups are ordered as the cross-product of each factor's levels "
            "(e.g. Drug: Water→Fentanyl, Sex: Male→Female → Water Male, Water Female, Fentanyl Male, Fentanyl Female)."
        )
        hint.setWordWrap(True)
        main_vbox.addWidget(hint)

        list_widgets: dict[str, "QListWidget"] = {}

        for factor in self._factor_definitions:
            levels_in_factor = all_levels.get(factor, set())
            if not levels_in_factor:
                continue

            grp = QGroupBox(factor)
            grp_lay = QHBoxLayout(grp)

            lw = QListWidget()
            lw.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            lw.setMaximumHeight(130)

            # Populate using existing order, appending unknown levels alphabetically
            saved_order = self._factor_level_order.get(factor, [])
            ordered_levels = [l for l in saved_order if l in levels_in_factor]
            remaining_levels = sorted(l for l in levels_in_factor if l not in ordered_levels)
            for level in ordered_levels + remaining_levels:
                lw.addItem(QListWidgetItem(level))

            list_widgets[factor] = lw

            btn_vbox = QVBoxLayout()
            up_btn = QPushButton("\u25b2")
            down_btn = QPushButton("\u25bc")
            up_btn.setFixedSize(28, 26)
            down_btn.setFixedSize(28, 26)
            up_btn.setToolTip("Move selected level up")
            down_btn.setToolTip("Move selected level down")

            def _move(lw_ref: "QListWidget", delta: int) -> None:
                idx = lw_ref.currentRow()
                if idx < 0:
                    return
                new_idx = idx + delta
                if new_idx < 0 or new_idx >= lw_ref.count():
                    return
                item = lw_ref.takeItem(idx)
                lw_ref.insertItem(new_idx, item)
                lw_ref.setCurrentRow(new_idx)

            up_btn.clicked.connect(lambda _, lw_r=lw: _move(lw_r, -1))
            down_btn.clicked.connect(lambda _, lw_r=lw: _move(lw_r, 1))

            btn_vbox.addWidget(up_btn)
            btn_vbox.addWidget(down_btn)
            btn_vbox.addStretch(1)
            grp_lay.addWidget(lw, 1)
            grp_lay.addLayout(btn_vbox)
            main_vbox.addWidget(grp)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        main_vbox.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Save the new level orders
        for factor, lw in list_widgets.items():
            self._factor_level_order[factor] = [
                lw.item(i).text() for i in range(lw.count())
            ]

        self._save_group_state()
        # Refresh all sub-tabs that use group ordering
        self._graphs_tab.update_graph()
        self._sections_tab._update_plot()

    def _open_subject_prechop_dialog(self) -> None:
        if self._project_root is None:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return
        subjects = sorted({str(s) for s in self._subject_by_session.values() if str(s).strip()})
        if not subjects:
            QMessageBox.information(self, "No Subjects", "No subject/session mappings are available yet.")
            return

        dlg = _SubjectPrechopDialog(self, subjects, self._subject_prechop_frames)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_offsets = dlg.offsets()
        if new_offsets == self._subject_prechop_frames:
            return

        self._subject_prechop_frames = new_offsets
        self._save_group_state()
        self._status.setText("Updated per-subject prechop. Refreshing analytics...")
        self._refresh()

    def _analysis_prechop_for_session(self, session_id: str) -> int:
        try:
            override = max(0, int(self._session_prechop_overrides.get(str(session_id), 0)))
        except Exception:
            override = 0
        if override > 0:
            return override
        subject = self._subject_by_session.get(str(session_id), str(session_id))
        try:
            return max(0, int(self._subject_prechop_frames.get(subject, 0)))
        except Exception:
            return 0

    def _apply_prechop_to_bout_df(self, bout_df: pd.DataFrame, *, rebase: bool) -> pd.DataFrame:
        """Trim bouts before per-subject prechop and optionally rebase frame indices."""
        if bout_df.empty or not {"session_id", "start_frame", "end_frame"}.issubset(bout_df.columns):
            return bout_df
        if not self._subject_prechop_frames and not self._session_prechop_overrides:
            return bout_df

        df = bout_df.copy()
        sid_arr = df["session_id"].astype(str).to_numpy()
        offsets = np.array([self._analysis_prechop_for_session(sid) for sid in sid_arr], dtype=np.int64)
        if offsets.size == 0 or int(offsets.max()) <= 0:
            return df

        starts = pd.to_numeric(df["start_frame"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        ends = pd.to_numeric(df["end_frame"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)

        starts = np.maximum(starts, offsets)
        valid = ends >= starts
        if not np.any(valid):
            return df.iloc[0:0].copy()

        df = df.loc[valid].copy()
        starts = starts[valid]
        ends = ends[valid]
        offsets = offsets[valid]

        if rebase:
            starts = starts - offsets
            ends = ends - offsets

        df["start_frame"] = starts.astype(np.int64)
        df["end_frame"] = ends.astype(np.int64)
        return df.reset_index(drop=True)

    def _recompute_summary_stats_from_shifted_bouts(self) -> None:
        """Recompute summary metrics from prechopped/rebased raw bouts when available."""
        if not self._summary_rows or not self._raw_bouts:
            return

        fps = self._project_fps()
        stats_by_key: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        for bid, bdf in self._raw_bouts.items():
            if bdf.empty or not {"session_id", "start_frame", "end_frame"}.issubset(bdf.columns):
                continue
            for sid, grp in bdf.groupby("session_id"):
                sid_str = str(sid)
                n_b = float(len(grp))
                total_frames = float((grp["end_frame"] - grp["start_frame"] + 1).sum())
                time_s = total_frames / fps
                mean_s = time_s / n_b if n_b > 0 else 0.0
                lat_s = float(grp["start_frame"].min()) / fps if n_b > 0 else float("nan")
                stats_by_key[(sid_str, str(bid))] = (n_b, time_s, mean_s, lat_s)

        adjusted_rows: list[dict[str, Any]] = []
        for row in self._summary_rows:
            bid = str(row.get("behavior_id", ""))
            if is_pseudo_behavior_id(bid):
                adjusted_rows.append(row)
                continue

            sid = str(row.get("session_id", ""))
            key = (sid, bid)
            if key in stats_by_key:
                n_b, time_s, mean_s, lat_s = stats_by_key[key]
                updated = dict(row)
                updated["n_bouts"] = n_b
                updated["time_spent_s"] = time_s
                updated["mean_bout_s"] = mean_s
                updated["latency_s"] = lat_s
                adjusted_rows.append(updated)
                continue

            # If no raw bouts are available for this row, keep legacy values but
            # still shift latency to keep the analysis timeline aligned.
            pre = self._analysis_prechop_for_session(sid)
            if pre > 0:
                updated = dict(row)
                try:
                    lat = float(updated.get("latency_s", float("nan")))
                    if np.isfinite(lat):
                        updated["latency_s"] = max(0.0, lat - (pre / fps))
                except Exception:
                    pass
                adjusted_rows.append(updated)
            else:
                adjusted_rows.append(row)

        self._summary_rows = adjusted_rows

    # ------------------------------------------------------------------
    # Manifest accessor (cached per refresh cycle)
    # ------------------------------------------------------------------

    def _manifest(self) -> Any:
        """Return the cached ImportManifest, loading it at most once per refresh."""
        if self._manifest_cache is _MANIFEST_UNSET:
            self._manifest_cache = (
                self._imports.load_manifest(self._project_root)
                if self._project_root is not None
                else None
            )
        return self._manifest_cache

    # ------------------------------------------------------------------
    # Subject / session mapping
    # ------------------------------------------------------------------

    def _build_subject_map(self) -> dict[str, str]:
        if self._project_root is None:
            return {}
        manifest = self._manifest()
        if manifest is None:
            return {}
        video_by_id = {v.asset_id: v for v in manifest.videos}
        out: dict[str, str] = {}
        session_types: dict[str, str] = {}
        for session in manifest.linked_sessions:
            sid = str(session.session_id)
            subject = (session.subject_id or "").strip()
            if not subject:
                video = video_by_id.get(session.video_asset_id)
                subject = (video.subject_id or "").strip() if video else ""
            out[sid] = subject or sid
            # Extract session type from video filename first, then fall back to
            # splitting the subject label on the first "_" (e.g. "m10_cond1"
            # → session type "cond1").
            video = video_by_id.get(session.video_asset_id)
            stype = ""
            if video:
                stem = Path(video.source_path).stem
                subj = out[sid]
                if subj and stem.startswith(subj):
                    remainder = stem[len(subj):].lstrip("_- ")
                    # Strip DLC suffix if present (e.g. "ConditioningDLC_...")
                    if remainder and not remainder.upper().startswith("DLC"):
                        stype = remainder
            # Fallback: parse "{subject}_{session_type}" convention from the
            # subject label itself (e.g. "m10_cond1" → "cond1").
            if not stype and "_" in out[sid]:
                stype = out[sid].split("_", 1)[1]
            session_types[sid] = stype
        self._session_type_by_session = session_types
        # Determine if any subject has multiple sessions
        subject_session_count: dict[str, int] = {}
        for sid, subj in out.items():
            subject_session_count[subj] = subject_session_count.get(subj, 0) + 1
        # Build session labels
        labels: dict[str, str] = {}
        for sid, subj in out.items():
            stype = session_types.get(sid, "")
            if subject_session_count.get(subj, 1) > 1 and stype:
                labels[sid] = f"{subj} \u2013 {stype}"
            else:
                labels[sid] = subj
        self._session_label_by_session = labels
        # Build reverse map: label → [session_ids]
        sessions_by_label: dict[str, list[str]] = {}
        for sid, label in labels.items():
            sessions_by_label.setdefault(label, []).append(sid)
        self._sessions_by_label = sessions_by_label
        return out

    def _invert_subject_map(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for sid, subject in self._subject_by_session.items():
            out.setdefault(subject, []).append(sid)
        return out

    def ordered_session_labels(self) -> list[str]:
        """Return session labels in user-defined order, falling back to
        alphabetical for any labels not yet in the custom order list."""
        all_labels = sorted({r["session_label"] for r in self._summary_rows})
        if not self._subject_order:
            return all_labels
        ordered: list[str] = [l for l in self._subject_order if l in all_labels]
        remaining = [l for l in all_labels if l not in self._subject_order]
        return ordered + remaining

    def _session_groups_factor_to_controls(self, factor: str) -> dict[str, str]:
        """Translate a legacy single active factor into facet controls."""
        if factor == "__interaction__":
            return {f: FACET_SPLIT for f in self._factor_definitions}
        controls = {f: FACET_COMBINE for f in self._factor_definitions}
        if factor in controls:
            controls[factor] = FACET_SPLIT
        return controls

    def _default_facet_controls(self) -> dict[str, str]:
        """Default facet state: split on the first factor, combine the rest."""
        controls = {f: FACET_COMBINE for f in self._factor_definitions}
        if self._factor_definitions:
            controls[self._factor_definitions[0]] = FACET_SPLIT
        return controls

    def _normalize_facet_controls(
        self, controls: dict[str, str] | None
    ) -> dict[str, str]:
        """Drop stale factors and add missing ones (defaulting to combine).

        Keeps the facet controls coherent with the current factor definitions.
        All-combine is a valid state — it pools every session into one "All"
        series; fresh projects instead seed a split via
        :meth:`_default_facet_controls`.
        """
        defs = self._factor_definitions
        if not defs:
            return {}
        src = dict(controls or {})
        out: dict[str, str] = {}
        for f in defs:
            val = src.get(f, FACET_COMBINE)
            # A specific-level filter is only valid if that level still exists.
            if val not in (FACET_COMBINE, FACET_SPLIT):
                levels = {
                    facs.get(f, "") for facs in self._session_factors.values()
                }
                if val not in levels:
                    val = FACET_COMBINE
            out[f] = val
        return out

    def _sync_session_groups(self) -> None:
        """Recompute ``_session_groups`` (and derived state) from facet controls."""
        self._facet_controls = self._normalize_facet_controls(self._facet_controls)
        mapping, split = facet_session_labels(
            self._session_factors, self._factor_definitions, self._facet_controls
        )
        self._session_groups.clear()
        self._session_groups.update(mapping)
        self._facet_split_factors = split
        # Derive a legacy single "active factor" for stats dialogs / labels.
        if len(split) == 1:
            self._active_grouping_factor = split[0]
        elif len(split) >= 2:
            self._active_grouping_factor = "__interaction__"
        else:
            self._active_grouping_factor = ""

    def _refresh_group_selectors(self) -> None:
        """Rebuild the factor/group selectors on every group-aware sub-tab.

        Called whenever factor definitions or per-session factor assignments
        change. Without this, only the Graphs tab was refreshed, leaving the
        Velocity and Session Sections selectors empty until a full data reload.
        """
        self._graphs_tab._refresh_factor_selector()
        self._sections_tab.on_groups_updated()
        self._velocity_tab.on_groups_updated()

    def _session_groups_for_controls(
        self, controls: dict[str, str]
    ) -> dict[str, str]:
        """session_label→series mapping for *controls* without mutating state."""
        norm = self._normalize_facet_controls(controls)
        mapping, _split = facet_session_labels(
            self._session_factors, self._factor_definitions, norm
        )
        return mapping

    def _split_factors_for_controls(self, controls: dict[str, str]) -> list[str]:
        norm = self._normalize_facet_controls(controls)
        return [f for f in self._factor_definitions if norm.get(f) == FACET_SPLIT]

    def _session_groups_for_factor(self, factor: str) -> dict[str, str]:
        """Back-compat: session_label→group mapping splitting on a single *factor*.

        Retained for any callers that still think in terms of one factor. With
        ``"__interaction__"`` it splits on every defined factor.
        """
        if factor == "__interaction__":
            controls = {f: FACET_SPLIT for f in self._factor_definitions}
        elif factor:
            controls = {f: FACET_COMBINE for f in self._factor_definitions}
            controls[factor] = FACET_SPLIT
        else:
            controls = {}
        return self._session_groups_for_controls(controls)

    def _ordered_group_list(
        self,
        available: list[str] | set[str],
        split_factors: list[str] | None = None,
    ) -> list[str]:
        """Return *available* groups respecting user-defined order.

        Priority:
        1. Multiple split factors: cross-product of per-factor level orders
           (e.g. Drug × Sex → Water Male, Water Female, Fentanyl Male…).
        2. A single split factor with a saved level order: that order.
        3. Fall back to the legacy ``_group_order`` manual list.

        ``split_factors`` lets a tab order by its own facet selection; when
        omitted the main tab's current split factors are used.
        """
        import itertools as _itertools

        avail_set = set(available)
        # Order the interaction cross-product by the factors actually split.
        if split_factors is None:
            split_factors = self._facet_split_factors
        interaction_factors = split_factors or self._factor_definitions
        if len(split_factors) == 1:
            factor = split_factors[0]
        elif len(split_factors) >= 2:
            factor = "__interaction__"
        else:
            factor = self._active_grouping_factor or (
                self._factor_definitions[0] if self._factor_definitions else ""
            )

        if factor == "__interaction__" and interaction_factors:
            level_lists: list[list[str]] = []
            for f in interaction_factors:
                all_f: set[str] = set()
                for facs in self._session_factors.values():
                    lvl = facs.get(f, "")
                    if lvl:
                        all_f.add(lvl)
                level_lists.append(self._ordered_levels_for_factor(f, all_f))
            if level_lists:
                ordered: list[str] = []
                for combo in _itertools.product(*level_lists):
                    name = " \u00d7 ".join(combo)
                    if name in avail_set and name not in ordered:
                        ordered.append(name)
                remaining = sorted(g for g in avail_set if g not in ordered)
                return ordered + remaining

        if factor and factor != "__interaction__" and self._factor_level_order.get(factor):
            level_order = self._factor_level_order[factor]
            ordered2 = [g for g in level_order if g in avail_set]
            remaining2 = sorted(g for g in avail_set if g not in level_order)
            return ordered2 + remaining2

        # Legacy manual group order
        ordered3 = [g for g in self._group_order if g in avail_set]
        remaining3 = sorted(g for g in avail_set if g not in self._group_order)
        return ordered3 + remaining3

    def _ordered_levels_for_factor(self, factor: str, available: set[str]) -> list[str]:
        """Return levels for *factor* in user-defined order; unknown levels appended alphabetically."""
        order = self._factor_level_order.get(factor, [])
        ordered = [l for l in order if l in available]
        remaining = sorted(l for l in available if l not in order)
        return ordered + remaining

    def _levels_by_factor(self) -> dict[str, list[str]]:
        """Distinct non-empty levels present for each defined factor (ordered)."""
        out: dict[str, list[str]] = {}
        for f in self._factor_definitions:
            levels = {
                facs.get(f, "") for facs in self._session_factors.values()
            }
            out[f] = self._ordered_levels_for_factor(f, {l for l in levels if l})
        return out

    def _group_color(self, group_name: str, index: int) -> str:
        """Return the colour for *group_name*, falling back to the palette."""
        return self._group_colors.get(group_name, _PALETTE[index % len(_PALETTE)])

    def _detected_session_types(self) -> list[str]:
        """Return sorted list of unique session types found in the project."""
        types = sorted({
            t for t in self._session_type_by_session.values() if t
        })
        return types

    # Behavior filter

    def _refresh_behavior_filter(self) -> None:
        # Remember which behavior IDs were previously checked
        prev_checked = {bid for bid, _lbl, act in self._behavior_filter_actions if act.isChecked()}
        self._behavior_filter_menu.clear()
        self._behavior_filter_actions.clear()

        # Convenience: Select All / Select None
        select_all_act = self._behavior_filter_menu.addAction("Select All")
        select_none_act = self._behavior_filter_menu.addAction("Select None")
        self._behavior_filter_menu.addSeparator()

        # Populate behavior checkboxes
        for b in self._behaviors.behaviors:
            if str(b.behavior_id) == NO_BEHAVIOR_ID:
                continue
            bid = str(b.behavior_id)
            label = str(b.name or b.behavior_id or "")
            action = self._behavior_filter_menu.addAction(label)
            action.setCheckable(True)
            # Default: check if it was previously checked, or check all if first population
            action.setChecked(bid in prev_checked if prev_checked else True)
            action.toggled.connect(self._on_filter_changed)
            self._behavior_filter_actions.append((bid, label, action))

        # Distance pseudo-behavior
        dist_act = self._behavior_filter_menu.addAction(DISTANCE_BEHAVIOR_NAME)
        dist_act.setCheckable(True)
        dist_act.setChecked(DISTANCE_BEHAVIOR_ID in prev_checked if prev_checked else False)
        dist_act.toggled.connect(self._on_filter_changed)
        self._behavior_filter_actions.append((DISTANCE_BEHAVIOR_ID, DISTANCE_BEHAVIOR_NAME, dist_act))

        # ROI occupancy pseudo-behaviors (one per configured zone, if any).
        roi_count = self._configured_roi_count()
        for zi in range(1, roi_count + 1):
            rid = roi_behavior_id(zi)
            rname = roi_behavior_name(zi, roi_count)
            roi_act = self._behavior_filter_menu.addAction(rname)
            roi_act.setCheckable(True)
            roi_act.setChecked(rid in prev_checked if prev_checked else False)
            roi_act.toggled.connect(self._on_filter_changed)
            self._behavior_filter_actions.append((rid, rname, roi_act))

        # Wire up Select All / Select None
        def _check_all():
            for _bid, _lbl, act in self._behavior_filter_actions:
                act.blockSignals(True)
                act.setChecked(True)
                act.blockSignals(False)
            self._on_filter_changed()

        def _check_none():
            for _bid, _lbl, act in self._behavior_filter_actions:
                act.blockSignals(True)
                act.setChecked(False)
                act.blockSignals(False)
            self._on_filter_changed()

        select_all_act.triggered.connect(_check_all)
        select_none_act.triggered.connect(_check_none)

        self._update_behavior_filter_label()

    def _on_filter_changed(self, _checked: bool = False) -> None:
        self._update_behavior_filter_label()
        self._summary_tab.rebuild()
        self._graphs_tab.update_graph()
        self._relationships_tab.on_data_loaded()

    def _update_behavior_filter_label(self) -> None:
        """Update the button text to reflect the current selection."""
        selected = self._selected_behavior_ids()
        total = len(self._behavior_filter_actions)
        # Don't count the pseudo-behaviors (distance / ROI) in the "real" total
        real_total = sum(1 for bid, _, _ in self._behavior_filter_actions if not is_pseudo_behavior_id(bid))
        real_selected = sum(1 for bid in selected if not is_pseudo_behavior_id(bid))
        has_pseudo_selected = any(is_pseudo_behavior_id(bid) for bid in selected)
        if real_selected == real_total and not has_pseudo_selected:
            self._behavior_filter_btn.setText("All behaviors ▾")
        elif not selected:
            self._behavior_filter_btn.setText("(none) ▾")
        elif len(selected) == 1:
            lbl = next((l for b, l, _ in self._behavior_filter_actions if b in selected), "?")
            self._behavior_filter_btn.setText(f"{lbl} ▾")
        else:
            self._behavior_filter_btn.setText(f"{len(selected)} selected ▾")

    def _selected_behavior_ids(self) -> set[str]:
        """Return the set of behavior IDs currently checked in the filter menu."""
        return {bid for bid, _lbl, act in self._behavior_filter_actions if act.isChecked()}

    # ------------------------------------------------------------------
    # Raw bout data for time-binned graphs
    # ------------------------------------------------------------------

    def _load_raw_bouts(self) -> dict[str, pd.DataFrame]:
        if self._project_root is None:
            return {}

        bid_name_map: dict[str, str] = {}
        behavior_list = [
            b for b in self._behaviors.behaviors
            if str(b.behavior_id) != NO_BEHAVIOR_ID
        ]
        for b in behavior_list:
            bid_name_map[str(b.behavior_id)] = str(b.name or b.behavior_id)

        result: dict[str, pd.DataFrame] = {}

        # --- HIGHEST PRIORITY: per-behavior temporal refinement bouts -----
        # These come from dense frame-level inference and postprocessing,
        # matching the source used by the summary tab.
        tr_root = self._project_root / "derived" / "temporal_refinement"
        active_tb_inference = self._get_active_target_behavior_inference_dir()
        if tr_root.exists():
            for behavior in behavior_list:
                bid = str(behavior.behavior_id or "").strip()
                if not bid:
                    continue
                bname = str(behavior.name or bid)
                token = self._safe_name(bid)
                latest_path = tr_root / token / "latest.json"
                if not latest_path.exists():
                    continue
                try:
                    latest = json.loads(latest_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                # Skip stale postprocess artifacts (inference no longer exists
                # or was superseded by a newer target_behavior inference run).
                inf_dir_raw = str(latest.get("inference_dir", "") or "").strip()
                if inf_dir_raw and not Path(inf_dir_raw).exists():
                    continue
                if (
                    active_tb_inference
                    and inf_dir_raw
                    and inf_dir_raw != active_tb_inference
                ):
                    continue
                post_dir_raw = str(latest.get("postprocess_dir", "") or "").strip()
                if not post_dir_raw:
                    continue
                manifest_path = Path(post_dir_raw) / "postprocess_manifest.json"
                if not manifest_path.exists():
                    continue
                try:
                    pm = json.loads(manifest_path.read_text(encoding="utf-8"))
                    bout_paths_map = {
                        str(k): str(v)
                        for k, v in (pm.get("bout_paths", {}) or {}).items()
                    }
                except Exception:
                    continue
                tr_rows: list[dict] = []
                for sid, bp_str in bout_paths_map.items():
                    bp_path = Path(bp_str)
                    if not bp_path.exists():
                        continue
                    try:
                        bout_df = pd.read_parquet(bp_path)
                    except Exception:
                        continue
                    if bout_df.empty or not {"start_frame", "end_frame"}.issubset(
                        bout_df.columns
                    ):
                        continue
                    for _, bout in bout_df.iterrows():
                        tr_rows.append({
                            "session_id": str(bout.get("session_id", sid)),
                            "start_frame": int(bout["start_frame"]),
                            "end_frame": int(bout["end_frame"]),
                            "behavior_id": bid,
                            "behavior": bname,
                        })
                if tr_rows:
                    result[bid] = pd.DataFrame(tr_rows)

        # --- target_behavior TR: recompute bouts from probability traces ----
        # Use per-behavior thresholds from temporal_review_settings.json so
        # that the graphs tab matches what the temporal review tab shows.
        # This runs before the behavior_bouts/ fallback so recomputed data
        # takes priority over stale evaluation-pipeline bouts.
        # Reuse the result cached by _load_from_target_behavior_tr when
        # available — avoids re-reading parquet files and re-running smoothing.
        if self._tr_bouts_cache is not None:
            recomputed = self._tr_bouts_cache
        else:
            trace_paths, sm_method, sm_window = self._get_tr_trace_paths_and_smoothing()
            recomputed = (
                self._recompute_bouts_from_traces(
                    trace_paths, bid_name_map,
                    smoothing_method=sm_method,
                    smoothing_window=sm_window,
                )
                if trace_paths else {}
            )
        if recomputed:
            for bid, bout_df in recomputed.items():
                if bid in result:
                    merged = pd.concat([result[bid], bout_df], ignore_index=True)
                    merged = merged.drop_duplicates(
                        subset=["session_id", "start_frame", "end_frame"],
                        keep="first",
                    )
                    result[bid] = merged
                else:
                    result[bid] = bout_df.reset_index(drop=True)

        # --- per-behavior parquet files in behavior_bouts/ ----------------
        # These originate from the evaluation pipeline (window-level
        # classification) and serve as a fallback when TR bouts are absent.
        bouts_dir = self._project_root / "derived" / "behavior_bouts"
        for behavior in behavior_list:
            bid = str(behavior.behavior_id or "").strip()
            if not bid or bid in result:
                continue
            bout_path = bouts_dir / f"{bid}_bouts.parquet"
            if bout_path.exists():
                try:
                    df = pd.read_parquet(bout_path)
                    if not df.empty and {"start_frame", "end_frame", "session_id"}.issubset(df.columns):
                        df = df.copy()
                        df["behavior_id"] = bid
                        df["behavior"] = str(behavior.name or bid)
                        result[bid] = df
                except Exception:
                    pass

        return result

    # ------------------------------------------------------------------
    # Analytics cache helpers
    # ------------------------------------------------------------------

    def _analytics_cache_dir(self, project_root: Path) -> Path:
        return project_root / "derived" / "analytics_cache"

    def _compute_source_fingerprint(
        self, project_root: Path, behavior_list: list,
    ) -> str:
        """Hash of mtime+size for all source files read during a refresh.

        When this hash matches the stored cache the full I/O pass is skipped.
        """
        parts: list[str] = []
        derived = project_root / "derived"
        tr_root = derived / "temporal_refinement"
        bouts_dir = derived / "behavior_bouts"

        # Include temporal review settings so that threshold/bout-param changes
        # in the Temporal Review tab invalidate the analytics cache even when
        # the underlying derived files have not been regenerated yet.
        tr_settings_path = project_root / "config" / "temporal_review_settings.json"
        if tr_settings_path.exists():
            try:
                st = tr_settings_path.stat()
                parts.append(f"{tr_settings_path}:{st.st_mtime:.3f}:{st.st_size}")
            except OSError:
                pass

        if tr_root.exists():
            for jf in tr_root.glob("*/latest.json"):
                try:
                    st = jf.stat()
                    parts.append(f"{jf}:{st.st_mtime:.3f}:{st.st_size}")
                except OSError:
                    pass

        if bouts_dir.exists():
            for behavior in behavior_list:
                p = bouts_dir / f"{str(behavior.behavior_id)}_bouts.parquet"
                if p.exists():
                    try:
                        st = p.stat()
                        parts.append(f"{p}:{st.st_mtime:.3f}:{st.st_size}")
                    except OSError:
                        pass

        import_manifests = sorted(derived.glob("*.import_manifest*.json"))
        if not import_manifests:
            import_manifests = sorted(derived.glob("**/*.import_manifest*.json"))
        for p in import_manifests[:1]:
            try:
                st = p.stat()
                parts.append(f"{p}:{st.st_mtime:.3f}:{st.st_size}")
            except OSError:
                pass

        raw = "|".join(sorted(parts))
        return hashlib.md5(raw.encode()).hexdigest()

    def _try_load_analytics_cache(
        self, project_root: Path, fingerprint: str, behavior_list: list,
    ) -> dict | None:
        """Return cached result bundle if fingerprint matches, else None."""
        cache_dir = self._analytics_cache_dir(project_root)
        meta_path = cache_dir / "analytics_cache.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if meta.get("fingerprint") != fingerprint or meta.get("version") != 2:
            return None
        summary_rows: list[dict] = meta.get("summary_rows", [])
        if not summary_rows:
            return None
        raw_bouts: dict[str, pd.DataFrame] = {}
        for behavior in behavior_list:
            bid = str(behavior.behavior_id or "").strip()
            if not bid:
                continue
            p = cache_dir / f"bouts_{self._safe_name(bid)}.parquet"
            if p.exists():
                try:
                    raw_bouts[bid] = pd.read_parquet(p)
                except Exception:
                    pass
        return {
            "summary_rows": summary_rows,
            "raw_bouts": raw_bouts,
            "from_cache": True,
            "extra_labels": {},
            "extra_groups": {},
        }

    def _save_analytics_cache(
        self,
        project_root: Path,
        fingerprint: str,
        summary_rows: list[dict],
        raw_bouts: dict[str, pd.DataFrame],
    ) -> None:
        """Persist summary_rows and raw_bouts to derived/analytics_cache/."""
        try:
            import math
            cache_dir = self._analytics_cache_dir(project_root)
            cache_dir.mkdir(parents=True, exist_ok=True)
            for bid, df in raw_bouts.items():
                if df.empty:
                    continue
                p = cache_dir / f"bouts_{self._safe_name(bid)}.parquet"
                df.to_parquet(p, index=False)
            clean_rows = [
                {k: (None if isinstance(v, float) and math.isnan(v) else v)
                 for k, v in row.items()}
                for row in summary_rows
            ]
            meta = {
                "version": 2,
                "fingerprint": fingerprint,
                "summary_rows": clean_rows,
            }
            (cache_dir / "analytics_cache.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to save analytics cache: %s", exc)

    # ------------------------------------------------------------------
    # Data loading  (async)
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Start an asynchronous analytics refresh (non-blocking)."""
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.setText("Refreshing\u2026")
        self._status.setText("Loading analytics data \u2013 please wait\u2026")
        self._manifest_cache = _MANIFEST_UNSET
        self._fps_cache = None
        self._tr_bouts_cache = None
        # Rebuild subject/session maps on the main thread (fast JSON read).
        # The background worker snapshots these maps before doing file I/O.
        if self._project_root is not None:
            self._subject_by_session = self._build_subject_map()
            self._session_by_subject = self._invert_subject_map()
            self._heatmap_tab._refresh_lists()
            self._density_tab.refresh_selectors()
        worker = TaskWorker(self._run_refresh_background)
        worker.signals.finished.connect(self._on_refresh_done)
        worker.signals.failed.connect(self._on_refresh_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_refresh_done(self, result: dict) -> None:
        """Main-thread callback: apply worker results and refresh the UI."""
        self._refresh_btn.setText("Refresh Analytics")
        self._refresh_btn.setEnabled(True)

        if result.get("error") == "no_project":
            self._status.setText("No project loaded.")
            self._summary_tab.rebuild()
            self._graphs_tab.update_graph()
            return
        if result.get("error") == "no_behaviors":
            self._status.setText("No behaviors defined.")
            self._summary_tab.rebuild()
            self._graphs_tab.update_graph()
            return

        self._summary_rows = result["summary_rows"]
        self._raw_bouts = result["raw_bouts"]
        self._graphs_tab._session_end_s_cache.clear()
        self._session_prechop_overrides.clear()
        self._session_end_s_overrides.clear()
        for sid, end_s in (result.get("extra_session_end_s") or {}).items():
            try:
                e = float(end_s)
            except Exception:
                continue
            if e > 0:
                self._session_end_s_overrides[str(sid)] = e
        for sid, frames in (result.get("extra_session_prechop") or {}).items():
            try:
                val = max(0, int(frames))
            except Exception:
                continue
            if val > 0:
                self._session_prechop_overrides[str(sid)] = val

        if self._subject_prechop_frames or self._session_prechop_overrides:
            self._raw_bouts = {
                bid: self._apply_prechop_to_bout_df(df, rebase=True)
                for bid, df in self._raw_bouts.items()
            }
            self._recompute_summary_stats_from_shifted_bouts()

        extra_labels: dict = result.get("extra_labels", {})
        if extra_labels:
            self._session_label_by_session.update(extra_labels)
        for label, group in result.get("extra_groups", {}).items():
            self._session_groups[label] = group

        # Apply factor assignments imported from merged projects' analytics_groups.json.
        # Only fills in empty slots — manual assignments from the user are preserved.
        extra_factor_assignments: dict = result.get("extra_factor_assignments", {})
        if extra_factor_assignments:
            for label, factors in extra_factor_assignments.items():
                for factor_name, level in factors.items():
                    if factor_name not in self._factor_definitions:
                        self._factor_definitions.append(factor_name)
                    sess_facs = self._session_factors.setdefault(label, {})
                    if not sess_facs.get(factor_name, ""):
                        sess_facs[factor_name] = level
            self._save_group_state()
            self._sync_session_groups()

        rows_loaded = len(self._summary_rows)
        merged_sessions = result.get("merged_sessions", 0)
        from_cache = result.get("from_cache", False)
        cache_tag = " (cached)" if from_cache else ""
        if merged_sessions:
            self._status.setText(
                f"Loaded {rows_loaded} row(s) + {merged_sessions} merged "
                f"session(s) from {result.get('n_merged_projects', 0)} "
                f"external project(s).{cache_tag}"
            )
        else:
            self._status.setText(
                f"Loaded {rows_loaded} row(s) across "
                f"{result.get('n_behaviors', 0)} behavior(s).{cache_tag}"
            )

        self._summary_tab.rebuild()
        self._graphs_tab.update_graph()
        self._relationships_tab.on_data_loaded()
        self._density_tab.refresh_selectors()
        self._sections_tab.on_data_loaded()
        self._velocity_tab.on_data_loaded()
        QTimer.singleShot(0, self._compute_and_add_distance_rows)
        QTimer.singleShot(0, self._compute_and_add_roi_rows)

    def _on_refresh_failed(self, traceback_str: str) -> None:
        """Main-thread callback: surface errors and re-enable the button."""
        self._refresh_btn.setText("Refresh Analytics")
        self._refresh_btn.setEnabled(True)
        self._status.setText("Error loading analytics \u2014 see log for details.")
        logger.error("Analytics refresh failed:\n%s", traceback_str)
        # Surface the top-level exception in the UI so failures are not silent.
        last_line = ""
        try:
            lines = [ln.strip() for ln in str(traceback_str).splitlines() if ln.strip()]
            if lines:
                last_line = lines[-1]
        except Exception:
            last_line = ""
        detail = f"\n\n{last_line}" if last_line else ""
        QMessageBox.critical(
            self,
            "Refresh Analytics Failed",
            "Analytics refresh failed. See project log for details." + detail,
        )

    def _run_refresh_background(self) -> dict:
        """Background-thread worker: load all analytics data.

        Builds summary_rows and raw_bouts in a single I/O pass, eliminating
        the previous double-read.  Per-behavior file reads are parallelised
        with ThreadPoolExecutor so large projects load much faster.
        Results are returned as a plain dict; _on_refresh_done applies them
        to self.* on the main thread.
        """
        project_root = self._project_root
        if project_root is None:
            return {"error": "no_project"}

        behavior_list = [
            b for b in self._behaviors.behaviors
            if str(b.behavior_id) != NO_BEHAVIOR_ID
        ]
        if not behavior_list:
            return {"error": "no_behaviors"}

        # Snapshot read-only maps built on the main thread just before this
        # worker was started.  We never write to self.* from here.
        subject_by_session: dict[str, str] = dict(self._subject_by_session)
        session_label_by_session: dict[str, str] = dict(self._session_label_by_session)
        session_type_by_session: dict[str, str] = dict(self._session_type_by_session)

        bid_name_map: dict[str, str] = {
            str(b.behavior_id): str(b.name or b.behavior_id)
            for b in behavior_list
        }
        fps = self._project_fps()
        active_tb_inference = self._get_active_target_behavior_inference_dir()

        # ── Try disk cache ─────────────────────────────────────────────────
        fingerprint = self._compute_source_fingerprint(project_root, behavior_list)
        cached = self._try_load_analytics_cache(project_root, fingerprint, behavior_list)
        if cached is not None:
            cached["n_behaviors"] = len(behavior_list)
            logger.debug("Analytics loaded from cache (fp %s).", fingerprint[:8])
            # Even on a cache hit we must still run Source 4 so that any
            # registered external merge projects are included.  The cache
            # only stores host-project rows; merged rows are never cached.
            if not self._merge_service.is_empty():
                extra_labels: dict = {}
                extra_groups: dict = {}
                extra_factor_assignments: dict = {}
                extra_session_prechop: dict[str, int] = {}
                extra_session_end_s: dict[str, float] = {}
                try:
                    (
                        ext_summary,
                        ext_raw_rows,
                        ext_label_map,
                        ext_tag_groups,
                        ext_factor_map,
                        ext_session_prechop_map,
                        ext_session_end_s_map,
                    ) = (
                        self._merge_service.load_merged_bouts(
                            bid_name_map, fps,
                            use_cached=self._merge_use_cache_chk.isChecked(),
                        )
                    )
                    extra_labels.update(ext_label_map)
                    extra_groups.update(ext_tag_groups)
                    extra_factor_assignments.update(ext_factor_map)
                    extra_session_prechop.update(ext_session_prechop_map)
                    extra_session_end_s.update(ext_session_end_s_map)
                    cached["summary_rows"].extend(ext_summary)
                    for _bid, _bout_rows in ext_raw_rows.items():
                        _new_df = pd.DataFrame(_bout_rows)
                        if _new_df.empty:
                            continue
                        if _bid in cached["raw_bouts"] and not cached["raw_bouts"][_bid].empty:
                            _merged_df = pd.concat([cached["raw_bouts"][_bid], _new_df], ignore_index=True)
                        else:
                            _merged_df = _new_df
                        if {"session_id", "start_frame", "end_frame"}.issubset(_merged_df.columns):
                            _merged_df = _merged_df.drop_duplicates(
                                subset=["session_id", "start_frame", "end_frame"],
                                keep="first",
                            )
                        cached["raw_bouts"][_bid] = _merged_df.reset_index(drop=True)
                    merged_sessions = len({r["session_id"] for r in ext_summary})
                except Exception:
                    merged_sessions = 0
                cached["extra_labels"] = extra_labels
                cached["extra_groups"] = extra_groups
                cached["extra_factor_assignments"] = extra_factor_assignments
                cached["extra_session_prechop"] = extra_session_prechop
                cached["extra_session_end_s"] = extra_session_end_s
                cached["merged_sessions"] = merged_sessions
                cached["n_merged_projects"] = len(self._merge_service.entries)
            return cached

        summary_rows: list[dict] = []
        raw_bouts: dict[str, pd.DataFrame] = {}
        loaded_keys: set[tuple[str, str]] = set()
        n_workers = min(8, (os.cpu_count() or 2) + 2)

        # ── Shared stats helper ────────────────────────────────────────────
        def _stats(grp: pd.DataFrame, sid_str: str, bid: str, bname: str) -> dict:
            n_b = len(grp)
            if "duration_frames" in grp.columns:
                total_f = float(grp["duration_frames"].sum())
            elif {"start_frame", "end_frame"}.issubset(grp.columns):
                total_f = float((grp["end_frame"] - grp["start_frame"] + 1).sum())
            else:
                total_f = 0.0
            time_s = total_f / fps
            mean_dur = time_s / n_b if n_b > 0 else 0.0
            lat = (
                float(grp["start_frame"].min()) / fps
                if n_b > 0 and "start_frame" in grp.columns
                else float("nan")
            )
            subj = subject_by_session.get(sid_str, sid_str)
            return {
                "session_id": sid_str,
                "subject": subj,
                "session_label": session_label_by_session.get(sid_str, subj),
                "session_type": session_type_by_session.get(sid_str, ""),
                "behavior_id": bid,
                "behavior": bname,
                "n_bouts": float(n_b),
                "time_spent_s": time_s,
                "mean_bout_s": mean_dur,
                "latency_s": lat,
                "distance_cm": 0.0,
            }

        # ── Source 1: behavior-specific TR folders (parallelised) ──────────
        tr_root = project_root / "derived" / "temporal_refinement"

        def _load_tr_behavior(behavior) -> tuple[list[dict], list[dict], str]:
            """Returns (summary_rows, raw_rows, bid)."""
            bid = str(behavior.behavior_id or "").strip()
            bname = str(behavior.name or bid)
            if not bid or not tr_root.exists():
                return [], [], bid
            latest_path = tr_root / self._safe_name(bid) / "latest.json"
            if not latest_path.exists():
                return [], [], bid
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
            except Exception:
                return [], [], bid
            inf_dir = str(latest.get("inference_dir", "") or "").strip()
            if inf_dir and not Path(inf_dir).exists():
                return [], [], bid
            if active_tb_inference and inf_dir and inf_dir != active_tb_inference:
                return [], [], bid
            post_dir = str(latest.get("postprocess_dir", "") or "").strip()
            if not post_dir:
                return [], [], bid

            metrics_path = Path(post_dir) / "session_metrics.json"
            manifest_path = Path(post_dir) / "postprocess_manifest.json"
            out_summary: list[dict] = []
            out_raw: list[dict] = []

            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception:
                    metrics = {}
                for sid, item in metrics.items():
                    n_b = float(item.get("n_bouts", 0) or 0)
                    time_s = float(item.get("time_spent_seconds", 0.0) or 0.0)
                    mean_dur = time_s / n_b if n_b > 0 else 0.0
                    lat = float(item.get("latency_to_first_behavior_s", float("nan")))
                    subj = subject_by_session.get(str(sid), str(sid))
                    out_summary.append({
                        "session_id": str(sid), "subject": subj,
                        "session_label": session_label_by_session.get(str(sid), subj),
                        "session_type": session_type_by_session.get(str(sid), ""),
                        "behavior_id": bid, "behavior": bname,
                        "n_bouts": n_b, "time_spent_s": time_s,
                        "mean_bout_s": mean_dur, "latency_s": lat, "distance_cm": 0.0,
                    })
                # Also pull raw bouts for motif analysis
                if manifest_path.exists():
                    try:
                        pm = json.loads(manifest_path.read_text(encoding="utf-8"))
                        bps = {str(k): str(v) for k, v in (pm.get("bout_paths", {}) or {}).items()}
                    except Exception:
                        bps = {}
                    for sid, bp_str in bps.items():
                        bp = Path(bp_str)
                        if not bp.exists():
                            continue
                        try:
                            bdf = pd.read_parquet(bp)
                        except Exception:
                            continue
                        if bdf.empty or not {"start_frame", "end_frame"}.issubset(bdf.columns):
                            continue
                        for _, row in bdf.iterrows():
                            out_raw.append({
                                "session_id": str(row.get("session_id", sid)),
                                "start_frame": int(row["start_frame"]),
                                "end_frame": int(row["end_frame"]),
                                "behavior_id": bid, "behavior": bname,
                            })
                if out_summary:
                    return out_summary, out_raw, bid

            # Fallback: derive summary stats from manifest bout parquets
            if manifest_path.exists():
                try:
                    pm = json.loads(manifest_path.read_text(encoding="utf-8"))
                    bps = {str(k): str(v) for k, v in (pm.get("bout_paths", {}) or {}).items()}
                except Exception:
                    bps = {}
                for sid, bp_str in bps.items():
                    bp = Path(bp_str)
                    if not bp.exists():
                        continue
                    try:
                        bdf = pd.read_parquet(bp)
                    except Exception:
                        continue
                    if bdf.empty or not {"start_frame", "end_frame"}.issubset(bdf.columns):
                        continue
                    out_summary.append(_stats(bdf, str(sid), bid, bname))
                    for _, row in bdf.iterrows():
                        out_raw.append({
                            "session_id": str(row.get("session_id", sid)),
                            "start_frame": int(row["start_frame"]),
                            "end_frame": int(row["end_frame"]),
                            "behavior_id": bid, "behavior": bname,
                        })
            return out_summary, out_raw, bid

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_load_tr_behavior, b): b for b in behavior_list}
            for fut in as_completed(futures):
                s_rows, r_rows, bid = fut.result()
                for r in s_rows:
                    loaded_keys.add((bid, r["session_id"]))
                summary_rows.extend(s_rows)
                if r_rows:
                    raw_bouts[bid] = pd.DataFrame(r_rows)

        # ── Source 2: target_behavior TR (competition/trace mode) ──────────
        trace_paths, sm_method, sm_window = self._get_tr_trace_paths_and_smoothing()
        if trace_paths:
            recomputed = self._recompute_bouts_from_traces(
                trace_paths, bid_name_map,
                smoothing_method=sm_method,
                smoothing_window=sm_window,
            )
            self._tr_bouts_cache = recomputed  # keep for legacy code
            for bid, bout_df in recomputed.items():
                if bid in raw_bouts:
                    combined = pd.concat([raw_bouts[bid], bout_df], ignore_index=True)
                    raw_bouts[bid] = combined.drop_duplicates(
                        subset=["session_id", "start_frame", "end_frame"], keep="first"
                    )
                else:
                    raw_bouts[bid] = bout_df.reset_index(drop=True)
                for sid, grp in bout_df.groupby("session_id"):
                    sid_str = str(sid)
                    if (bid, sid_str) in loaded_keys:
                        continue
                    summary_rows.append(_stats(grp, sid_str, bid, bid_name_map.get(bid, bid)))
                    loaded_keys.add((bid, sid_str))

        # ── Source 3: behavior_bouts parquets fallback (parallelised) ──────
        bouts_dir = project_root / "derived" / "behavior_bouts"

        def _load_bouts_fallback(behavior) -> tuple[list[dict], pd.DataFrame | None, str]:
            bid = str(behavior.behavior_id or "").strip()
            bname = str(behavior.name or bid)
            if not bid or bid in raw_bouts:
                return [], None, bid
            bp = bouts_dir / f"{bid}_bouts.parquet"
            if not bp.exists():
                return [], None, bid
            try:
                df = pd.read_parquet(bp)
            except Exception:
                return [], None, bid
            if df.empty:
                return [], None, bid
            out_rows = []
            for sid, grp in df.groupby("session_id"):
                if (bid, str(sid)) not in loaded_keys:
                    out_rows.append(_stats(grp, str(sid), bid, bname))
            raw_df = df.copy()
            raw_df["behavior_id"] = bid
            raw_df["behavior"] = bname
            return out_rows, raw_df, bid

        if bouts_dir.exists():
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_load_bouts_fallback, b): b for b in behavior_list}
                for fut in as_completed(futures):
                    s_rows, raw_df, bid = fut.result()
                    for r in s_rows:
                        loaded_keys.add((bid, r["session_id"]))
                    summary_rows.extend(s_rows)
                    if raw_df is not None and not raw_df.empty and bid not in raw_bouts:
                        raw_bouts[bid] = raw_df

        # ── Ensure all known sessions appear (zero rows for absent data) ───
        for sid, subject in subject_by_session.items():
            for behavior in behavior_list:
                bid = str(behavior.behavior_id or "").strip()
                if not bid or (bid, sid) in loaded_keys:
                    continue
                bname = str(behavior.name or bid)
                slbl = session_label_by_session.get(sid, subject)
                summary_rows.append({
                    "session_id": sid, "subject": subject,
                    "session_label": slbl,
                    "session_type": session_type_by_session.get(sid, ""),
                    "behavior_id": bid, "behavior": bname,
                    "n_bouts": 0.0, "time_spent_s": 0.0,
                    "mean_bout_s": 0.0, "latency_s": float("nan"), "distance_cm": 0.0,
                })
                loaded_keys.add((bid, sid))

        # ── Source 4: merged external projects ─────────────────────────────
        extra_labels: dict = {}
        extra_groups: dict = {}
        extra_factor_assignments: dict = {}
        extra_session_prechop: dict[str, int] = {}
        extra_session_end_s: dict[str, float] = {}
        merged_sessions = 0
        if not self._merge_service.is_empty():
            (
                ext_summary,
                ext_raw_rows,
                ext_label_map,
                ext_tag_groups,
                ext_factor_map,
                ext_session_prechop_map,
                ext_session_end_s_map,
            ) = (
                self._merge_service.load_merged_bouts(
                    bid_name_map, fps,
                    use_cached=self._merge_use_cache_chk.isChecked(),
                )
            )
            extra_labels.update(ext_label_map)
            extra_groups.update(ext_tag_groups)
            extra_factor_assignments.update(ext_factor_map)
            extra_session_prechop.update(ext_session_prechop_map)
            extra_session_end_s.update(ext_session_end_s_map)
            summary_rows.extend(ext_summary)
            for bid, bout_rows in ext_raw_rows.items():
                new_df = pd.DataFrame(bout_rows)
                if new_df.empty:
                    continue
                if bid in raw_bouts and not raw_bouts[bid].empty:
                    merged_df = pd.concat([raw_bouts[bid], new_df], ignore_index=True)
                else:
                    merged_df = new_df
                if {"session_id", "start_frame", "end_frame"}.issubset(merged_df.columns):
                    merged_df = merged_df.drop_duplicates(
                        subset=["session_id", "start_frame", "end_frame"],
                        keep="first",
                    )
                raw_bouts[bid] = merged_df.reset_index(drop=True)
            merged_sessions = len({r["session_id"] for r in ext_summary})

        # ── Save cache (host rows only; merged project rows are excluded) ───
        host_rows = [r for r in summary_rows if "::" not in str(r.get("session_id", ""))]
        host_bouts: dict[str, pd.DataFrame] = {}
        for bid, df in raw_bouts.items():
            if df is None or df.empty:
                continue
            if "session_id" not in df.columns:
                host_bouts[bid] = df
                continue
            host_only = df.loc[~df["session_id"].astype(str).str.contains("::", na=False)].copy()
            if not host_only.empty:
                host_bouts[bid] = host_only.reset_index(drop=True)
        self._save_analytics_cache(project_root, fingerprint, host_rows, host_bouts)

        return {
            "summary_rows": summary_rows,
            "raw_bouts": raw_bouts,
            "extra_labels": extra_labels,
            "extra_groups": extra_groups,
            "extra_factor_assignments": extra_factor_assignments,
            "extra_session_prechop": extra_session_prechop,
            "extra_session_end_s": extra_session_end_s,
            "merged_sessions": merged_sessions,
            "n_merged_projects": len(self._merge_service.entries),
            "n_behaviors": len(behavior_list),
            "from_cache": False,
        }



    def _compute_and_add_distance_rows(self) -> None:
        """Compute session-level distance and add pseudo-behavior rows.

        Called via QTimer.singleShot so the UI shows behavior data
        immediately while pose files are loaded in the background.
        """
        existing_sids = {r["session_id"] for r in self._summary_rows}
        if not existing_sids:
            return
        # Remove any stale distance rows from a previous run
        self._summary_rows[:] = [
            r for r in self._summary_rows
            if r["behavior_id"] != DISTANCE_BEHAVIOR_ID
        ]
        for sid in sorted(existing_sids):
            dist_px = self._compute_session_distance(sid)
            ppm = self._pixels_per_mm_for_session(sid)
            # ppm is pixels-per-mm; divide by ppm to get mm, then by 10 for cm
            dist_cm = (dist_px / ppm / 10.0) if ppm and ppm > 0 else dist_px
            subject = self._subject_by_session.get(sid, sid)
            session_label = self._session_label_by_session.get(sid, subject)
            session_type = self._session_type_by_session.get(sid, "")
            self._summary_rows.append({
                "session_id": sid,
                "subject": subject,
                "session_label": session_label,
                "session_type": session_type,
                "behavior_id": DISTANCE_BEHAVIOR_ID,
                "behavior": DISTANCE_BEHAVIOR_NAME,
                "n_bouts": 0.0,
                "time_spent_s": 0.0,
                "mean_bout_s": 0.0,
                "latency_s": float("nan"),
                "distance_cm": dist_cm,
            })
        self._summary_tab.rebuild()
        self._graphs_tab.update_graph()

    def _configured_roi_count(self) -> int:
        """Number of ROI zones to expose as pseudo-behaviors (0 if none defined).

        Returns the project ``roi_count`` only when at least one zone (project
        default or any per-subject override) has a non-zero area; otherwise
        ROIs were never set up for this project and we expose nothing.
        """
        if self._project_root is None:
            return 0
        try:
            cfg = self._roi_service.load(self._project_root)
        except Exception:
            return 0
        count = max(0, int(cfg.get("roi_count", 0) or 0))
        if count <= 0:
            return 0

        def _has_area(z: Any) -> bool:
            return (
                isinstance(z, dict)
                and int(z.get("w", 0) or 0) > 0
                and int(z.get("h", 0) or 0) > 0
            )

        proj_zones = cfg.get("project_rois", {}).get("target_zones", []) or []
        if any(_has_area(z) for z in proj_zones):
            return count
        for s_block in (cfg.get("subject_rois", {}) or {}).values():
            if any(_has_area(z) for z in (s_block.get("target_zones", []) or [])):
                return count
        return 0

    def _compute_and_add_roi_rows(self) -> None:
        """Compute per-session ROI occupancy and add pseudo-behavior rows.

        One synthetic behavior per configured zone; the standard summary
        columns carry the ROI metrics (time in zone, entries, mean visit
        duration, latency to first entry).  Called via QTimer.singleShot so
        behavior data renders before pose files are loaded.
        """
        if self._project_root is None:
            return
        existing_sids = {r["session_id"] for r in self._summary_rows}
        if not existing_sids:
            return
        # Drop any stale ROI rows from a previous run.
        self._summary_rows[:] = [
            r for r in self._summary_rows
            if not is_roi_behavior_id(r["behavior_id"])
        ]
        roi_count = self._configured_roi_count()
        if roi_count <= 0:
            self._summary_tab.rebuild()
            return

        fps = self._project_fps()
        for sid in sorted(existing_sids):
            subject = self._subject_by_session.get(sid, sid)
            session_label = self._session_label_by_session.get(sid, subject)
            session_type = self._session_type_by_session.get(sid, "")
            try:
                rois = self._roi_service.resolve_target_rois(
                    self._project_root, f"{subject}::{sid}"
                )
            except Exception:
                rois = []
            pose = self._get_pose_for_session(sid)
            for zi in range(roi_count):
                roi = rois[zi] if zi < len(rois) else None
                stats = self._compute_session_roi_stats(sid, pose, roi, fps)
                if stats is None:
                    continue
                time_s, n_entries, mean_s, latency_s = stats
                self._summary_rows.append({
                    "session_id": sid,
                    "subject": subject,
                    "session_label": session_label,
                    "session_type": session_type,
                    "behavior_id": roi_behavior_id(zi + 1),
                    "behavior": roi_behavior_name(zi + 1, roi_count),
                    "n_bouts": float(n_entries),
                    "time_spent_s": time_s,
                    "mean_bout_s": mean_s,
                    "latency_s": latency_s,
                    "distance_cm": 0.0,
                })
        self._summary_tab.rebuild()
        self._graphs_tab.update_graph()

    def _compute_session_roi_stats(
        self, session_id: str, pose: Any, roi: dict | None, fps: float,
    ) -> tuple[float, int, float, float] | None:
        """Return ``(time_in_s, n_entries, mean_visit_s, latency_s)`` for one ROI.

        A frame counts as "inside" when the body centroid falls within the ROI
        rectangle.  Boundary/tracking flicker is debounced so jitter doesn't
        inflate the entry count.  Returns None when the ROI is undefined
        (zero area) or pose data is unavailable.
        """
        if pose is None or not roi:
            return None
        from abel.utils import roi_geometry
        if not roi_geometry.roi_has_area(roi):
            return None

        start_idx = self._analysis_prechop_for_session(session_id)
        cx = np.asarray(pose.centroid_x, dtype=np.float64)[start_idx:]
        cy = np.asarray(pose.centroid_y, dtype=np.float64)[start_idx:]
        if cx.size == 0:
            return None

        # "Inside" respects the ROI's true shape (rect/circle/polygon), so
        # occupancy for a circular or freehand zone matches what was drawn.
        inside = roi_geometry.roi_contains(roi, cx, cy)
        # Debounce ~0.2 s of boundary flicker so a jittery frame isn't an entry.
        min_run = max(1, int(round(0.2 * fps))) if fps > 0 else 1
        inside = _debounce_bool(inside, min_run)

        n_inside = int(inside.sum())
        time_s = (n_inside / fps) if fps > 0 else 0.0
        # Entries = out→in transitions; an initial inside run counts as one.
        entries_mask = inside & ~np.concatenate(([False], inside[:-1]))
        n_entries = int(entries_mask.sum())
        mean_s = (time_s / n_entries) if n_entries > 0 else 0.0
        if n_entries > 0 and fps > 0:
            latency_s = float(int(np.argmax(inside))) / fps
        else:
            latency_s = float("nan")
        return time_s, n_entries, mean_s, latency_s

    def _clear_display(self) -> None:
        self._summary_rows.clear()
        self._raw_bouts.clear()
        self._last_stats_result.clear()
        self._pose_cache.clear()
        self._pose_vel_cache.clear()
        self._summary_tab.rebuild()
        self._graphs_tab.update_graph()
        if self._heatmap_tab._figure is not None:
            self._heatmap_tab._figure.clear()
            if self._heatmap_tab._canvas is not None:
                self._heatmap_tab._canvas.draw_idle()
        self._relationships_tab.on_data_loaded()
        self._velocity_tab._vel_cache.clear()
        self._velocity_tab.on_data_loaded()
        self._status.setText("Analytics display cleared.")

    # ------------------------------------------------------------------
    # Merge-projects helpers
    # ------------------------------------------------------------------

    def _rebuild_merge_list_widget(self) -> None:
        self._merge_list.clear()
        for e in self._merge_service.entries:
            grp_txt = f"  →  group: {e.group_override}" if e.group_override else ""
            self._merge_list.addItem(f"[{e.tag}]  {e.project_root}{grp_txt}")
        n = len(self._merge_service.entries)
        if n:
            self._merge_btn.setText(f"\u229f Merge Projects ({n})\u2026")
        else:
            state = "\u229f" if self._merge_btn.isChecked() else "\u229e"
            self._merge_btn.setText(f"{state} Merge Projects\u2026")

    def _merge_add_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select ABEL Project Folder", "",
        )
        if not folder:
            return
        p = Path(folder)
        if not (p / "project.yaml").exists() and not (p / "config").exists():
            QMessageBox.warning(
                self, "Invalid Project",
                f"The selected folder does not look like an ABEL project:\n{folder}\n\n"
                "Expected to find project.yaml or a config/ sub-folder.",
            )
            return
        # Ask for tag and optional group override
        tag, ok = QInputDialog.getText(
            self, "Project Tag",
            "Enter a short tag for this project (used as session-ID prefix):",
            text=p.name,
        )
        if not ok or not tag.strip():
            return
        tag = tag.strip()
        grp, ok2 = QInputDialog.getText(
            self, "Group Override (optional)",
            "Assign all imported sessions to a group name.\n"
            "Leave blank to use the imported project's own group definitions:",
        )
        grp = grp.strip() if ok2 else ""
        self._merge_service.add_project(p, tag=tag, group_override=grp)
        if self._project_root:
            self._merge_service.save(self._project_root)
        self._rebuild_merge_list_widget()
        self._status.setText(
            f"Added merged project '{tag}'. Click Refresh Analytics to load data."
        )

    def _merge_remove_selected(self) -> None:
        row = self._merge_list.currentRow()
        if row < 0 or row >= len(self._merge_service.entries):
            return
        entry = self._merge_service.entries[row]
        self._merge_service.remove_project(entry.project_root)
        if self._project_root:
            self._merge_service.save(self._project_root)
        self._rebuild_merge_list_widget()
        self._status.setText(
            "Merged project removed. Click Refresh Analytics to reload."
        )

    def _merge_edit_item(self) -> None:
        row = self._merge_list.currentRow()
        if row < 0 or row >= len(self._merge_service.entries):
            return
        entry = self._merge_service.entries[row]
        tag, ok = QInputDialog.getText(
            self, "Edit Tag", "Tag:", text=entry.tag,
        )
        if not ok:
            return
        grp, ok2 = QInputDialog.getText(
            self, "Edit Group Override",
            "Group override (blank = use imported project's own groups):",
            text=entry.group_override,
        )
        entry.tag = tag.strip() or entry.tag
        entry.group_override = grp.strip() if ok2 else entry.group_override
        if self._project_root:
            self._merge_service.save(self._project_root)
        self._rebuild_merge_list_widget()


        """Return the inference_dir currently active for target_behavior, or ''."""
        if self._project_root is None:
            return ""
        tb_latest = (
            self._project_root / "derived" / "temporal_refinement"
            / "target_behavior" / "latest.json"
        )
        if not tb_latest.exists():
            return ""
        try:
            tb = json.loads(tb_latest.read_text(encoding="utf-8"))
            return str(tb.get("inference_dir", "") or "").strip()
        except Exception:
            return ""

    def _load_from_temporal_refinement(
        self, tr_root: Path, bid: str, bname: str,
    ) -> int:
        """Load summary rows from a behavior-specific temporal-refinement folder.

        Only the folder derived from safe_name(bid) is checked.  The generic
        ``target_behavior`` token is deliberately excluded: using it as a
        wildcard would shadow every behavior with the same single TR run.
        """
        rows = 0
        fps = self._project_fps()
        active_tb_inference = self._get_active_target_behavior_inference_dir()
        for token in (self._safe_name(bid),):
            latest_path = tr_root / token / "latest.json"
            if not latest_path.exists():
                continue
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            # Guard: if the inference this postprocess was built from no longer
            # exists, the artifacts are stale.  Skip them so the caller falls
            # through to _load_from_target_behavior_tr which recomputes bouts
            # from the current (live) inference traces.
            inference_dir_raw = str(latest.get("inference_dir", "") or "").strip()
            if inference_dir_raw and not Path(inference_dir_raw).exists():
                continue
            # Also skip when the behavior-specific postprocess was generated from
            # a different inference run than the currently active target_behavior
            # inference (i.e. a new model was trained since the last postprocess).
            if (
                active_tb_inference
                and inference_dir_raw
                and inference_dir_raw != active_tb_inference
            ):
                continue

            post_dir_raw = str(latest.get("postprocess_dir", "") or "").strip()
            if not post_dir_raw:
                continue
            manifest_path = Path(post_dir_raw) / "postprocess_manifest.json"
            metrics_path = Path(post_dir_raw) / "session_metrics.json"

            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception:
                    metrics = {}
                for sid, item in metrics.items():
                    n_bouts = float(item.get("n_bouts", 0) or 0)
                    time_s = float(item.get("time_spent_seconds", 0.0) or 0.0)
                    mean_dur = time_s / n_bouts if n_bouts > 0 else 0.0
                    latency_s = float(item.get("latency_to_first_behavior_s", float("nan")))
                    subject = self._subject_by_session.get(str(sid), str(sid))
                    session_label = self._session_label_by_session.get(str(sid), subject)
                    session_type = self._session_type_by_session.get(str(sid), "")
                    self._summary_rows.append({
                        "session_id": str(sid),
                        "subject": subject,
                        "session_label": session_label,
                        "session_type": session_type,
                        "behavior_id": bid,
                        "behavior": bname,
                        "n_bouts": n_bouts,
                        "time_spent_s": time_s,
                        "mean_bout_s": mean_dur,
                        "latency_s": latency_s,
                        "distance_cm": 0.0,
                    })
                    rows += 1
                if rows > 0:
                    return rows

            if manifest_path.exists():
                try:
                    pm = json.loads(manifest_path.read_text(encoding="utf-8"))
                    bout_paths = {str(k): str(v) for k, v in (pm.get("bout_paths", {}) or {}).items()}
                except Exception:
                    bout_paths = {}
                for sid, bp in bout_paths.items():
                    bp_path = Path(bp)
                    if not bp_path.exists():
                        continue
                    try:
                        bout_df = pd.read_parquet(bp_path)
                    except Exception:
                        continue
                    n_bouts = len(bout_df)
                    if "duration_frames" in bout_df.columns:
                        total_frames = float(bout_df["duration_frames"].sum())
                    elif {"start_frame", "end_frame"}.issubset(bout_df.columns):
                        total_frames = float((bout_df["end_frame"] - bout_df["start_frame"] + 1).sum())
                    else:
                        total_frames = 0.0
                    time_s = total_frames / fps
                    mean_dur = time_s / n_bouts if n_bouts > 0 else 0.0
                    if n_bouts > 0 and "start_frame" in bout_df.columns:
                        latency_s = float(bout_df["start_frame"].min()) / fps
                    else:
                        latency_s = float("nan")
                    subject = self._subject_by_session.get(str(sid), str(sid))
                    session_label = self._session_label_by_session.get(str(sid), subject)
                    session_type = self._session_type_by_session.get(str(sid), "")
                    self._summary_rows.append({
                        "session_id": str(sid),
                        "subject": subject,
                        "session_label": session_label,
                        "session_type": session_type,
                        "behavior_id": bid,
                        "behavior": bname,
                        "n_bouts": float(n_bouts),
                        "time_spent_s": time_s,
                        "mean_bout_s": mean_dur,
                        "latency_s": latency_s,
                        "distance_cm": 0.0,
                    })
                    rows += 1
                if rows > 0:
                    return rows
        return rows

    # ------------------------------------------------------------------
    # target_behavior TR with per-bout behavior resolution
    # ------------------------------------------------------------------

    def _load_from_target_behavior_tr(
        self,
        fps: float,
        bid_name_map: dict[str, str],
        loaded_keys: set[tuple[str, str]],
    ) -> int:
        """Resolve the combined ``target_behavior`` TR run into per-behavior stats.

        Recomputes bout intervals from probability traces using per-behavior
        thresholds from temporal_review_settings.json, ensuring consistency
        with what the temporal review tab displays.
        """
        trace_paths, sm_method, sm_window = self._get_tr_trace_paths_and_smoothing()
        if not trace_paths:
            return 0

        recomputed = self._recompute_bouts_from_traces(
            trace_paths, bid_name_map,
            smoothing_method=sm_method,
            smoothing_window=sm_window,
        )
        # Cache for reuse in _load_raw_bouts (same refresh cycle).
        self._tr_bouts_cache = recomputed
        if not recomputed:
            return 0

        rows_added = 0
        for bid, bout_df in recomputed.items():
            for sid, grp in bout_df.groupby("session_id"):
                sid_str = str(sid)
                if (bid, sid_str) in loaded_keys:
                    continue
                n_bouts = len(grp)
                total_frames = float((grp["end_frame"] - grp["start_frame"] + 1).sum())
                time_s = total_frames / fps
                mean_dur = time_s / n_bouts if n_bouts > 0 else 0.0
                if n_bouts > 0 and "start_frame" in grp.columns:
                    latency_s = float(grp["start_frame"].min()) / fps
                else:
                    latency_s = float("nan")
                subject = self._subject_by_session.get(sid_str, sid_str)
                session_label = self._session_label_by_session.get(sid_str, subject)
                session_type = self._session_type_by_session.get(sid_str, "")
                self._summary_rows.append({
                    "session_id": sid_str,
                    "subject": subject,
                    "session_label": session_label,
                    "session_type": session_type,
                    "behavior_id": bid,
                    "behavior": bid_name_map.get(bid, bid),
                    "n_bouts": float(n_bouts),
                    "time_spent_s": time_s,
                    "mean_bout_s": mean_dur,
                    "latency_s": latency_s,
                    "distance_cm": 0.0,
                })
                loaded_keys.add((bid, sid_str))
                rows_added += 1

        return rows_added

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _project_fps(self) -> float:
        if self._fps_cache is not None:
            return self._fps_cache
        if self._project_root is None:
            return 30.0
        manifest = self._manifest()
        fps = 30.0
        if manifest is not None:
            for v in manifest.videos:
                if v.fps and v.fps > 0:
                    fps = float(v.fps)
                    break
        self._fps_cache = fps
        return fps

    def _get_pose_for_session(self, session_id: str) -> Any:
        """Return PoseData for a session, using a cache to avoid repeated loads."""
        if session_id in self._pose_cache:
            return self._pose_cache[session_id]
        if self._project_root is None:
            return None
        manifest = self._manifest()
        if manifest is None:
            return None
        pose_by_id = {p.asset_id: p for p in manifest.poses}
        session = next(
            (s for s in manifest.linked_sessions if str(s.session_id) == session_id),
            None,
        )
        if session is None:
            return None
        pa = pose_by_id.get(session.pose_asset_id)
        if pa is None:
            return None
        pp = Path(pa.source_path)
        if pa.local_path:
            lp = Path(pa.local_path)
            if lp.exists():
                pp = lp
        if not pp.exists():
            return None
        try:
            pose = self._pose.load(pp)
            self._pose_cache[session_id] = pose
            return pose
        except Exception:
            return None

    def _pixels_per_mm_for_session(self, session_id: str) -> float | None:
        """Return pixels-per-mm calibration for a session, or None."""
        if self._project_root is None:
            return None
        manifest = self._manifest()
        if manifest is None:
            return None
        return self._imports.pixels_per_mm_for_session(manifest, session_id)

    def _compute_bout_distance(
        self, session_id: str, intervals: list[tuple[int, int]],
    ) -> float:
        """Compute total centroid displacement (pixels) across bout intervals.

        Uses subsampling at ``_DISTANCE_SUBSAMPLE_HZ`` to reduce jitter-
        induced inflation of the distance estimate.
        """
        pose = self._get_pose_for_session(session_id)
        if pose is None or not intervals:
            return 0.0
        fps = self._project_fps()
        step = max(1, round(fps / _DISTANCE_SUBSAMPLE_HZ))
        cx = np.asarray(pose.centroid_x, dtype=np.float64)
        cy = np.asarray(pose.centroid_y, dtype=np.float64)
        n = len(cx)
        total = 0.0
        for s, e in intervals:
            s = max(0, min(s, n - 1))
            e = max(s, min(e, n - 1))
            if e <= s:
                continue
            seg_cx = cx[s:e + 1:step]
            seg_cy = cy[s:e + 1:step]
            if len(seg_cx) < 2:
                continue
            dx = np.diff(seg_cx)
            dy = np.diff(seg_cy)
            total += float(np.sum(np.sqrt(dx * dx + dy * dy)))
        return total

    def _compute_session_distance(self, session_id: str) -> float:
        """Compute total centroid distance (pixels) across the *entire* session.

        Subsamples at ``_DISTANCE_SUBSAMPLE_HZ`` (default 5 Hz) to prevent
        tracking jitter from inflating the estimate.
        """
        pose = self._get_pose_for_session(session_id)
        if pose is None:
            return 0.0
        fps = self._project_fps()
        step = max(1, round(fps / _DISTANCE_SUBSAMPLE_HZ))
        start_idx = self._analysis_prechop_for_session(session_id)
        cx = np.asarray(pose.centroid_x, dtype=np.float64)[start_idx::step]
        cy = np.asarray(pose.centroid_y, dtype=np.float64)[start_idx::step]
        if len(cx) < 2:
            return 0.0
        dx = np.diff(cx)
        dy = np.diff(cy)
        return float(np.sum(np.sqrt(dx * dx + dy * dy)))

    def _compute_session_distance_binned(
        self, session_id: str, bin_seconds: float,
    ) -> list[tuple[float, float]]:
        """Compute distance (pixels) per time-bin across the whole session.

        Returns a sorted list of ``(bin_start_s, distance_px)`` tuples.
        """
        pose = self._get_pose_for_session(session_id)
        if pose is None:
            return []
        fps = self._project_fps()
        step = max(1, round(fps / _DISTANCE_SUBSAMPLE_HZ))
        cx = np.asarray(pose.centroid_x, dtype=np.float64)
        cy = np.asarray(pose.centroid_y, dtype=np.float64)
        n = len(cx)
        start_idx = self._analysis_prechop_for_session(session_id)
        indices = np.arange(start_idx, n, step)
        cx_sub = cx[indices]
        cy_sub = cy[indices]
        if len(cx_sub) < 2:
            return []
        times_s = (indices.astype(np.float64) - float(start_idx)) / fps
        dx = np.diff(cx_sub)
        dy = np.diff(cy_sub)
        segment_dist = np.sqrt(dx * dx + dy * dy)
        segment_times = times_s[:-1]
        bin_starts = (segment_times // bin_seconds) * bin_seconds
        result: dict[float, float] = {}
        for t, d in zip(bin_starts, segment_dist):
            t_key = float(t)
            result[t_key] = result.get(t_key, 0.0) + float(d)
        return sorted(result.items())

    def _distance_for_bouts_df(
        self, session_id: str, bout_df: pd.DataFrame,
    ) -> float:
        """Compute distance (cm) for a DataFrame of bouts with start_frame/end_frame."""
        if bout_df.empty or not {"start_frame", "end_frame"}.issubset(bout_df.columns):
            return 0.0
        pre = self._analysis_prechop_for_session(session_id)
        intervals = list(zip(
            (bout_df["start_frame"].astype(int) + pre).tolist(),
            (bout_df["end_frame"].astype(int) + pre).tolist(),
        ))
        dist_px = self._compute_bout_distance(session_id, intervals)
        ppm = self._pixels_per_mm_for_session(session_id)
        if ppm and ppm > 0:
            return dist_px / ppm / 10.0  # px -> mm -> cm
        return dist_px

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in str(value).strip()
        ) or "target_behavior"

    def _get_active_target_behavior_inference_dir(self) -> str:
        """Return the inference_dir currently active for target_behavior, or ''.

        This is used to detect stale per-behavior postprocess artifacts: if
        the artifact's inference_dir doesn't match the currently active
        target_behavior inference run, the artifact should be skipped.
        """
        if self._project_root is None:
            return ""
        tb_latest = (
            self._project_root / "derived" / "temporal_refinement"
            / "target_behavior" / "latest.json"
        )
        if not tb_latest.exists():
            return ""
        try:
            tb = json.loads(tb_latest.read_text(encoding="utf-8"))
            return str(tb.get("inference_dir", "") or "").strip()
        except Exception:
            return ""



    def _load_temporal_review_thresholds(self) -> dict[str, dict[str, float]]:
        """Load per-behavior bout detection thresholds from temporal_review_settings."""
        if self._project_root is None:
            return {}
        settings_path = self._project_root / "config" / "temporal_review_settings.json"
        if not settings_path.exists():
            return {}
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        defaults = data.get("__all__", {})
        by_behavior = data.get("by_behavior", {})
        result: dict[str, dict[str, float]] = {}
        for key, vals in by_behavior.items():
            if key == "target_behavior":
                continue
            result[key] = {
                "onset_threshold": float(vals.get("onset_threshold", defaults.get("onset_threshold", 0.5))),
                "min_bout_duration_frames": int(vals.get("min_bout_duration_frames", defaults.get("min_bout_duration_frames", 6))),
                "merge_gap_frames": int(vals.get("merge_gap_frames", defaults.get("merge_gap_frames", 3))),
            }
        result["__defaults__"] = {
            "onset_threshold": float(defaults.get("onset_threshold", 0.5)),
            "min_bout_duration_frames": int(defaults.get("min_bout_duration_frames", 6)),
            "merge_gap_frames": int(defaults.get("merge_gap_frames", 3)),
        }
        return result

    def _recompute_bouts_from_traces(
        self,
        trace_paths: dict[str, str],
        bid_name_map: dict[str, str],
        smoothing_method: str = "moving_average",
        smoothing_window: int = 5,
    ) -> dict[str, pd.DataFrame]:
        """Recompute bout intervals from probability traces using per-behavior thresholds.

        Returns dict mapping behavior_id to DataFrame with columns
        [session_id, start_frame, end_frame, behavior_id, behavior].
        """
        from abel.temporal_refinement.bout_postprocess import (
            smooth_probabilities,
            threshold_probabilities,
            merge_close_bouts,
            remove_short_bouts,
            binary_trace_to_intervals,
        )

        all_thresholds = self._load_temporal_review_thresholds()
        defaults = all_thresholds.pop("__defaults__", {
            "onset_threshold": 0.5,
            "min_bout_duration_frames": 6,
            "merge_gap_frames": 3,
        })

        result: dict[str, list[dict]] = {bid: [] for bid in bid_name_map}

        for sid, tp_str in trace_paths.items():
            if not tp_str or not Path(tp_str).exists():
                continue
            try:
                trace_df = pd.read_parquet(tp_str)
            except Exception:
                continue
            if trace_df.empty:
                continue

            frame_arr = (
                trace_df["frame"].to_numpy(dtype=int)
                if "frame" in trace_df.columns
                else np.arange(len(trace_df))
            )

            for bid, bname in bid_name_map.items():
                prob_col = f"prob_{bid}"
                if prob_col not in trace_df.columns:
                    continue

                raw = pd.to_numeric(
                    trace_df[prob_col], errors="coerce",
                ).fillna(0.0).to_numpy(dtype=float)
                smoothed = smooth_probabilities(
                    raw, method=smoothing_method, window=smoothing_window,
                )

                params = all_thresholds.get(bid, defaults)
                onset = float(params.get("onset_threshold", defaults["onset_threshold"]))
                min_bout = int(params.get("min_bout_duration_frames", defaults["min_bout_duration_frames"]))
                merge_gap = int(params.get("merge_gap_frames", defaults["merge_gap_frames"]))

                binary = threshold_probabilities(smoothed, onset_thresh=onset, offset_thresh=onset)
                binary = merge_close_bouts(binary, max_gap_frames=merge_gap)
                binary = remove_short_bouts(binary, min_duration_frames=min_bout)
                intervals = binary_trace_to_intervals(binary)

                for s_idx, e_idx in intervals:
                    sf = int(frame_arr[s_idx]) if s_idx < len(frame_arr) else s_idx
                    ef = int(frame_arr[min(e_idx, len(frame_arr) - 1)])
                    result[bid].append({
                        "session_id": sid,
                        "start_frame": sf,
                        "end_frame": ef,
                        "behavior_id": bid,
                        "behavior": bname,
                    })

        return {bid: pd.DataFrame(rows) for bid, rows in result.items() if rows}

    def _get_tr_trace_paths_and_smoothing(self) -> tuple[dict[str, str], str, int]:
        """Load trace paths and smoothing params from target_behavior TR manifests.

        Returns (trace_paths, smoothing_method, smoothing_window).
        """
        trace_paths: dict[str, str] = {}
        smoothing_method = "moving_average"
        smoothing_window = 5
        if self._project_root is None:
            return trace_paths, smoothing_method, smoothing_window

        tr_root = self._project_root / "derived" / "temporal_refinement"
        latest_path = tr_root / "target_behavior" / "latest.json"
        if not latest_path.exists():
            return trace_paths, smoothing_method, smoothing_window
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            return trace_paths, smoothing_method, smoothing_window

        post_dir_raw = str(latest.get("postprocess_dir", "") or "").strip()
        inference_dir_raw = str(latest.get("inference_dir", "") or "").strip()

        if inference_dir_raw:
            ip = Path(inference_dir_raw) / "inference_manifest.json"
            if ip.exists():
                try:
                    im = json.loads(ip.read_text(encoding="utf-8"))
                    trace_paths = {
                        str(k): str(v)
                        for k, v in (im.get("trace_paths", {}) or {}).items()
                    }
                except Exception:
                    pass

        if post_dir_raw:
            mp = Path(post_dir_raw) / "postprocess_manifest.json"
            if mp.exists():
                try:
                    pm = json.loads(mp.read_text(encoding="utf-8"))
                    smoothing_method = str(pm.get("smoothing_method", smoothing_method))
                    smoothing_window = int(pm.get("smoothing_window", smoothing_window))
                except Exception:
                    pass

        return trace_paths, smoothing_method, smoothing_window

    # ------------------------------------------------------------------
    # Velocity computation helpers
    # ------------------------------------------------------------------

    def _compute_bout_velocities(
        self,
        session_id: str,
        start_frame: int,
        end_frame: int,
        smooth_window: int = 5,
    ) -> "np.ndarray":
        """Compute per-frame velocity (cm/s or px/s) for a single bout.

        Args:
            session_id: session to load pose data for.
            start_frame: inclusive first frame of the bout (prechop-adjusted).
            end_frame: inclusive last frame of the bout.
            smooth_window: moving-average kernel size (frames, must be odd).

        Returns:
            1-D float64 array of velocity values; empty array if no pose data.
        """
        pose = self._get_pose_for_session(session_id)
        fps = self._project_fps()
        ppm = self._pixels_per_mm_for_session(session_id)
        if pose is None:
            # Fallback: load pre-computed centroid_velocity from the local
            # pose_features/sessions/ parquet (px/s, same origin as pose.centroid_x/y).
            vel = self._load_centroid_velocity_from_features(
                session_id, start_frame, end_frame, fps, ppm
            )
            if len(vel) == 0:
                return np.array([], dtype=np.float64)
            k = max(1, int(smooth_window))
            if k > 1 and len(vel) >= k:
                kernel = np.ones(k, dtype=np.float64) / k
                vel = np.convolve(vel, kernel, mode="same")
            return vel
        cx = np.asarray(pose.centroid_x, dtype=np.float64)
        cy = np.asarray(pose.centroid_y, dtype=np.float64)
        n = len(cx)
        s = max(0, min(start_frame, n - 1))
        e = max(s, min(end_frame, n - 1))
        if e <= s:
            return np.array([0.0], dtype=np.float64)
        seg_cx = cx[s : e + 1]
        seg_cy = cy[s : e + 1]
        dx = np.diff(seg_cx)
        dy = np.diff(seg_cy)
        vel_px_frame = np.sqrt(dx * dx + dy * dy)
        # Convert px/frame → cm/s (or px/s if uncalibrated)
        if ppm and ppm > 0:
            vel = vel_px_frame / ppm / 10.0 * fps   # px → mm → cm, × fps → /s
        else:
            vel = vel_px_frame * fps                 # px/s (uncalibrated)
        # Moving-average smoothing
        k = max(1, int(smooth_window))
        if k > 1 and len(vel) >= k:
            kernel = np.ones(k, dtype=np.float64) / k
            vel = np.convolve(vel, kernel, mode="same")
        return vel

    def _load_centroid_velocity_from_features(
        self,
        session_id: str,
        start_frame: int,
        end_frame: int,
        fps: float,
        ppm: float | None,
    ) -> "np.ndarray":
        """Load pre-computed centroid velocity from the local pose_features parquet.

        The ``centroid_velocity`` column is stored in px/s.  This method
        converts it to the same units used by ``_compute_bout_velocities``
        (cm/s when calibrated, px/s otherwise) and returns the bout slice.
        Returns an empty array when the parquet cannot be found or read.
        """
        if self._project_root is None:
            return np.array([], dtype=np.float64)
        # Return cached array if available (avoids re-reading the parquet each bout)
        if session_id in self._pose_vel_cache:
            vel_arr = self._pose_vel_cache[session_id]
            n = len(vel_arr)
            s = max(0, min(start_frame, n - 1))
            e = max(s, min(end_frame, n - 1))
            if e <= s:
                return np.array([0.0], dtype=np.float64)
            vel_pxs = vel_arr[s : e + 1]
            if ppm and ppm > 0:
                return vel_pxs / ppm / 10.0
            return vel_pxs
        pq_path = (
            self._project_root
            / "derived"
            / "pose_features"
            / "sessions"
            / f"{session_id}.parquet"
        )
        if not pq_path.exists():
            return np.array([], dtype=np.float64)
        try:
            import pandas as _pd
            df = _pd.read_parquet(pq_path, columns=["frame", "centroid_velocity"])
        except Exception:
            return np.array([], dtype=np.float64)
        if df.empty or "centroid_velocity" not in df.columns:
            return np.array([], dtype=np.float64)
        # Sort by frame so positional slice == frame-indexed slice
        df = df.sort_values("frame").reset_index(drop=True)
        vel_arr = df["centroid_velocity"].to_numpy(dtype=np.float64)
        self._pose_vel_cache[session_id] = vel_arr
        n = len(vel_arr)
        s = max(0, min(start_frame, n - 1))
        e = max(s, min(end_frame, n - 1))
        if e <= s:
            return np.array([0.0], dtype=np.float64)
        vel_pxs = vel_arr[s : e + 1]
        if ppm and ppm > 0:
            return vel_pxs / ppm / 10.0   # px/s → mm/s → cm/s
        return vel_pxs

    def _collect_velocity_data_for_behavior(
        self,
        behavior_id: str,
        smooth_window: int = 5,
    ) -> "list[dict]":
        """Return a list of per-bout velocity records for *behavior_id*.

        Each record contains:
          session_id, session_label, group, start_frame, end_frame,
          duration_frames, duration_s, mean_vel, peak_vel, velocity_trace.
        """
        bout_df = self._raw_bouts.get(str(behavior_id))
        if bout_df is None or bout_df.empty:
            return []
        if not {"session_id", "start_frame", "end_frame"}.issubset(bout_df.columns):
            return []
        fps = max(1e-9, self._project_fps())
        records: list[dict] = []
        for _, row in bout_df.iterrows():
            sid = str(row.get("session_id", ""))
            pre = self._analysis_prechop_for_session(sid)
            sf = int(row["start_frame"]) + pre
            ef = int(row["end_frame"]) + pre
            vel = self._compute_bout_velocities(sid, sf, ef, smooth_window)
            if len(vel) == 0:
                continue
            slbl = self._session_label_by_session.get(sid, sid)
            grp = self._session_groups.get(slbl, "")
            dur_frames = ef - sf + 1
            records.append({
                "session_id": sid,
                "session_label": slbl,
                "group": grp,
                "start_frame": sf,
                "end_frame": ef,
                "duration_frames": dur_frames,
                "duration_s": dur_frames / fps,
                "mean_vel": float(np.mean(vel)),
                "peak_vel": float(np.max(vel)),
                "velocity_trace": vel,
            })
        return records

    def _filtered_rows(self) -> list[dict[str, Any]]:
        selected = self._selected_behavior_ids()
        if not selected:
            return []
        return [r for r in self._summary_rows if r["behavior_id"] in selected]


# ======================================================================
# Sub-tab 1: Summary & Statistics
# ======================================================================

class _SummaryStatsWidget(QWidget):
    """Table, inline multi-factor group assignment, and statistical tests."""

    _FIXED_COLS = 2  # columns before factor columns: [Session, Session Type]

    def __init__(self, host: BehaviorAnalyticsTab) -> None:
        super().__init__()
        self._host = host
        self._rebuilding = False  # guard to prevent recursive rebuilds
        self._drag_start_row: int | None = None  # for drag-and-drop reorder

        # -- session / factor table -----------------------------------
        self._session_table = QTableWidget(0, 2)
        self._session_table.setHorizontalHeaderLabels(["Session", "Session Type"])
        self._session_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._session_table.horizontalHeader().setSortIndicatorShown(True)
        self._session_table.horizontalHeader().sectionClicked.connect(
            self._on_header_sort_clicked
        )
        self._session_table.verticalHeader().setVisible(False)
        # Enable extended selection (Ctrl+click, Shift+click)
        self._session_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._session_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self._session_table.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self._session_table.setDragEnabled(True)
        self._session_table.setAcceptDrops(True)
        self._session_table.setDropIndicatorShown(True)
        self._session_table.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._session_table.cellChanged.connect(self._on_factor_cell_changed)
        self._session_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._session_table.customContextMenuRequested.connect(
            self._show_session_context_menu
        )
        self._session_table.installEventFilter(self)
        # Sync order when drag-drop completes (if supported by the model)
        try:
            self._session_table.model().rowsMoved.connect(self._on_rows_moved)
        except AttributeError:
            pass  # signal not available in this PySide6 version

        self._sort_column: int | None = None
        self._sort_ascending: bool = True

        group_box = QGroupBox("Sessions && Factors")
        gb_layout = QVBoxLayout(group_box)
        hint = QLabel(
            "Check sessions to include in statistics and graphs. "
            "Add factors (columns) and type a level for each session "
            "to enable group comparisons and multi-factor ANOVA.\n"
            "Ctrl+click or Shift+click to select multiple cells, then right-click "
            "to assign a factor level to all selected rows. "
            "Ctrl+C / Ctrl+V to copy and paste cell values. Drag rows to reorder."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #90a4ae; font-size: 10px;")
        gb_layout.addWidget(hint)

        # Factor management row
        factor_row = QHBoxLayout()
        self._add_factor_btn = QPushButton("Add Factor")
        self._add_factor_btn.setToolTip(
            "Add a new factor column (e.g. Sex, Treatment, Genotype).\n"
            "Sessions can then be assigned a level for each factor."
        )
        self._add_factor_btn.clicked.connect(self._add_factor)
        self._remove_factor_btn = QPushButton("Remove Factor")
        self._remove_factor_btn.setToolTip("Remove the last factor column.")
        self._remove_factor_btn.clicked.connect(self._remove_factor)

        # Sort selector
        self._sort_combo = QComboBox()
        self._sort_combo.addItem("Custom Order", userData="custom")
        self._sort_combo.addItem("Session Name (A\u2192Z)", userData="name_asc")
        self._sort_combo.addItem("Session Name (Z\u2192A)", userData="name_desc")
        self._sort_combo.addItem("Session Type", userData="type")
        self._sort_combo.setToolTip("Sort sessions in the table")
        self._sort_combo.currentIndexChanged.connect(self._on_sort_combo_changed)

        factor_row.addWidget(self._add_factor_btn)
        factor_row.addWidget(self._remove_factor_btn)
        factor_row.addWidget(QLabel("Sort:"))
        factor_row.addWidget(self._sort_combo)
        factor_row.addStretch(1)
        gb_layout.addLayout(factor_row)

        gb_layout.addWidget(self._session_table, 1)

        sel_row = QHBoxLayout()
        check_all_btn = QPushButton("Check All")
        check_all_btn.clicked.connect(self._check_all)
        uncheck_all_btn = QPushButton("Uncheck All")
        uncheck_all_btn.clicked.connect(self._uncheck_all)
        self._auto_group_btn = QPushButton("Auto-Group by Session Type")
        self._auto_group_btn.setToolTip(
            "Automatically assign group labels based on detected session types\n"
            "(e.g. Conditioning, Extinction, Recall) into a new factor column."
        )
        self._auto_group_btn.clicked.connect(self._auto_group_by_session_type)
        self._assign_btn = QPushButton("Assign Factor to Selected\u2026")
        self._assign_btn.setToolTip(
            "Assign a factor level to all selected (highlighted) sessions.\n"
            "Select multiple sessions with Ctrl+click or Shift+click first."
        )
        self._assign_btn.clicked.connect(self._assign_factor_to_selected)
        sel_row.addWidget(check_all_btn)
        sel_row.addWidget(uncheck_all_btn)
        sel_row.addWidget(self._auto_group_btn)
        sel_row.addWidget(self._assign_btn)
        sel_row.addStretch(1)
        gb_layout.addLayout(sel_row)

        # Move up/down buttons
        move_row = QHBoxLayout()
        self._move_up_btn = QPushButton("\u25B2 Move Up")
        self._move_up_btn.setToolTip("Move selected sessions up in the list.")
        self._move_up_btn.clicked.connect(self._move_selected_up)
        self._move_down_btn = QPushButton("\u25BC Move Down")
        self._move_down_btn.setToolTip("Move selected sessions down in the list.")
        self._move_down_btn.clicked.connect(self._move_selected_down)
        move_row.addWidget(self._move_up_btn)
        move_row.addWidget(self._move_down_btn)
        move_row.addStretch(1)
        gb_layout.addLayout(move_row)

        # -- summary data table ---------------------------------------
        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["Session", "Session Type", "Group", "Behavior", "N Bouts",
             "Total Duration (s)", "Mean Bout Duration (s)", "Distance Traveled"]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        # -- statistics -----------------------------------------------
        stats_row = QHBoxLayout()
        self._stats_btn = QPushButton("Run Statistics\u2026")
        self._stats_btn.setToolTip("Run t-test, one-way, or two-way ANOVA on grouped session data.")
        self._stats_btn.clicked.connect(self._run_statistics_dialog)
        self._export_csv_btn = QPushButton("Export Table to CSV\u2026")
        self._export_csv_btn.clicked.connect(self._export_csv)
        stats_row.addWidget(self._stats_btn)
        stats_row.addWidget(self._export_csv_btn)
        stats_row.addStretch(1)

        self._stats_output_text = ""
        self._stats_view_btn = QPushButton("\U0001f4ca View Results\u2026")
        self._stats_view_btn.setToolTip("Show the last statistics results in a popup.")
        self._stats_view_btn.clicked.connect(self._show_summary_stats_popup)
        self._stats_view_btn.setEnabled(False)
        stats_row.addWidget(self._stats_view_btn)

        # -- layout ---------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.addWidget(group_box, 1)
        left_l.addLayout(stats_row)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.addWidget(self._table, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([380, 520])

        root = QVBoxLayout(self)
        root.addWidget(splitter, 1)

    # -- session table -------------------------------------------------

    def _ordered_session_labels(self) -> list[str]:
        """Return session labels in current sort/custom order."""
        all_labels = self._host.ordered_session_labels()
        sort_mode = str(self._sort_combo.currentData() or "custom")
        if sort_mode == "name_asc":
            return sorted(all_labels)
        elif sort_mode == "name_desc":
            return sorted(all_labels, reverse=True)
        elif sort_mode == "type":
            def _type_key(label: str) -> str:
                for sid, lbl in self._host._session_label_by_session.items():
                    if lbl == label:
                        return self._host._session_type_by_session.get(sid, "")
                return ""
            return sorted(all_labels, key=lambda l: (_type_key(l), l))
        # "custom" — respect user-defined order
        return all_labels

    def _refresh_session_table(self) -> None:
        """Rebuild the session/factor table from current data."""
        self._session_table.blockSignals(True)
        try:
            session_labels = self._ordered_session_labels()
            factors = self._host._factor_definitions

            n_cols = self._FIXED_COLS + len(factors)
            self._session_table.setColumnCount(n_cols)
            headers = ["Session", "Session Type"] + list(factors)
            self._session_table.setHorizontalHeaderLabels(headers)

            # Preserve check state
            prev_checked = self._checked_subjects()
            # Track which labels were already in the table so we can
            # distinguish user-unchecked rows from brand-new rows.
            prev_labels: set[str] = set()
            for _i in range(self._session_table.rowCount()):
                _it = self._session_table.item(_i, 0)
                if _it is not None:
                    prev_labels.add(_it.text())
            prev_had_rows = bool(prev_labels)

            self._session_table.setRowCount(len(session_labels))
            for row_idx, label in enumerate(session_labels):
                # Column 0: checkbox + session label (selectable for multi-select)
                item0 = QTableWidgetItem(label)
                item0.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsDragEnabled
                )
                if prev_had_rows and label in prev_labels:
                    # Label was in the table before — preserve user's choice
                    item0.setCheckState(
                        Qt.CheckState.Checked if label in prev_checked
                        else Qt.CheckState.Unchecked
                    )
                else:
                    # Brand-new label (or first population) — default to Checked
                    item0.setCheckState(Qt.CheckState.Checked)
                self._session_table.setItem(row_idx, 0, item0)

                # Column 1: session type (read-only but selectable)
                stype = ""
                for sid, lbl in self._host._session_label_by_session.items():
                    if lbl == label:
                        stype = self._host._session_type_by_session.get(sid, "")
                        break
                item1 = QTableWidgetItem(stype)
                item1.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                self._session_table.setItem(row_idx, 1, item1)

                # Factor columns (editable + selectable)
                session_facs = self._host._session_factors.get(label, {})
                for fi, fname in enumerate(factors):
                    col = self._FIXED_COLS + fi
                    level = session_facs.get(fname, "")
                    item_f = QTableWidgetItem(level)
                    item_f.setFlags(
                        Qt.ItemFlag.ItemIsEnabled
                        | Qt.ItemFlag.ItemIsEditable
                        | Qt.ItemFlag.ItemIsSelectable
                    )
                    self._session_table.setItem(row_idx, col, item_f)

            self._session_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )
            # Sync host subject order from current table order
            self._sync_order_from_table()
        finally:
            self._session_table.blockSignals(False)

    def _on_factor_cell_changed(self, row: int, col: int) -> None:
        """Handle edits in the session/factor table."""
        if self._rebuilding:
            return
        if col == 0:
            # Checkbox toggled — update graphs
            self.rebuild()
            self._host._graphs_tab.update_graph()
            return
        if col < self._FIXED_COLS:
            return  # session type column is read-only
        # Factor column edited
        fi = col - self._FIXED_COLS
        factors = self._host._factor_definitions
        if fi >= len(factors):
            return
        fname = factors[fi]
        item_label = self._session_table.item(row, 0)
        item_val = self._session_table.item(row, col)
        if item_label is None or item_val is None:
            return
        label = item_label.text()
        level = item_val.text().strip()
        if label not in self._host._session_factors:
            self._host._session_factors[label] = {}
        if level:
            self._host._session_factors[label][fname] = level
        else:
            self._host._session_factors[label].pop(fname, None)
        self._host._sync_session_groups()
        self.rebuild()
        self._host._refresh_group_selectors()
        self._host._graphs_tab.update_graph()
        self._host._save_group_state()

    def _on_subject_toggled(self) -> None:
        self.rebuild()
        self._host._graphs_tab.update_graph()

    def _checked_subjects(self) -> set[str]:
        out: set[str] = set()
        for i in range(self._session_table.rowCount()):
            item = self._session_table.item(i, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                out.add(item.text())
        return out

    def _check_all(self) -> None:
        self._session_table.blockSignals(True)
        for i in range(self._session_table.rowCount()):
            item = self._session_table.item(i, 0)
            if item is not None:
                item.setCheckState(Qt.CheckState.Checked)
        self._session_table.blockSignals(False)
        self.rebuild()
        self._host._graphs_tab.update_graph()

    def _uncheck_all(self) -> None:
        self._session_table.blockSignals(True)
        for i in range(self._session_table.rowCount()):
            item = self._session_table.item(i, 0)
            if item is not None:
                item.setCheckState(Qt.CheckState.Unchecked)
        self._session_table.blockSignals(False)
        self.rebuild()
        self._host._graphs_tab.update_graph()

    # -- sorting -------------------------------------------------------

    def _on_header_sort_clicked(self, logical_index: int) -> None:
        """Sort the session table when a column header is clicked."""
        if self._sort_column == logical_index:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_column = logical_index
            self._sort_ascending = True
        self._sort_table_by_column(logical_index, self._sort_ascending)

    def _on_sort_combo_changed(self) -> None:
        """Re-sort via the sort dropdown."""
        self._refresh_session_table()
        self.rebuild()
        self._host._graphs_tab.update_graph()

    def _sort_table_by_column(self, col: int, ascending: bool) -> None:
        """Sort session rows in-place by the given column, then persist order."""
        n = self._session_table.rowCount()
        if n < 2:
            return
        # Collect current row data
        rows_data: list[tuple[str, list[Any]]] = []
        for i in range(n):
            label_item = self._session_table.item(i, 0)
            sort_item = self._session_table.item(i, col)
            sort_key = sort_item.text() if sort_item else ""
            rows_data.append((sort_key, self._snapshot_row(i)))
        rows_data.sort(key=lambda x: x[0].lower(), reverse=not ascending)
        self._session_table.blockSignals(True)
        for i, (_, row_snap) in enumerate(rows_data):
            self._restore_row(i, row_snap)
        self._session_table.blockSignals(False)
        self._sync_order_from_table()
        self._sort_combo.blockSignals(True)
        self._sort_combo.setCurrentIndex(0)  # "Custom Order" since user manually sorted
        self._sort_combo.blockSignals(False)

    def _snapshot_row(self, row: int) -> list[Any]:
        """Capture all cell data from a row for reorder operations."""
        n_cols = self._session_table.columnCount()
        snap: list[Any] = []
        for c in range(n_cols):
            item = self._session_table.item(row, c)
            if item is None:
                snap.append(("", None, Qt.ItemFlag(0)))
            else:
                snap.append((item.text(), item.checkState() if c == 0 else None, item.flags()))
        return snap

    def _restore_row(self, row: int, snap: list[Any]) -> None:
        """Write a row snapshot back into the table."""
        for c, (text, check, flags) in enumerate(snap):
            item = QTableWidgetItem(text)
            item.setFlags(flags)
            if c == 0 and check is not None:
                item.setCheckState(check)
            self._session_table.setItem(row, c, item)

    def _sync_order_from_table(self) -> None:
        """Update host._subject_order from current table row order."""
        order: list[str] = []
        for i in range(self._session_table.rowCount()):
            item = self._session_table.item(i, 0)
            if item is not None:
                order.append(item.text())
        self._host._subject_order = order
        self._host._save_group_state()

    # -- drag-and-drop / move ----------------------------------------

    def _on_rows_moved(self) -> None:
        """Called after the QTableWidget internal row move completes."""
        self._sync_order_from_table()
        self.rebuild()
        self._host._graphs_tab.update_graph()

    def _selected_row_indices(self) -> list[int]:
        """Return sorted list of currently selected (highlighted) row indices."""
        rows = sorted({idx.row() for idx in self._session_table.selectedIndexes()})
        return rows

    def _move_selected_up(self) -> None:
        """Move all selected rows up by one position."""
        selected = self._selected_row_indices()
        if not selected or selected[0] == 0:
            return
        self._session_table.blockSignals(True)
        for row in selected:
            above = self._snapshot_row(row - 1)
            current = self._snapshot_row(row)
            self._restore_row(row - 1, current)
            self._restore_row(row, above)
        self._session_table.blockSignals(False)
        # Re-select the moved rows
        self._session_table.clearSelection()
        for row in selected:
            new_row = row - 1
            for c in range(self._session_table.columnCount()):
                item = self._session_table.item(new_row, c)
                if item is not None:
                    item.setSelected(True)
        self._sync_order_from_table()
        self._sort_combo.blockSignals(True)
        self._sort_combo.setCurrentIndex(0)
        self._sort_combo.blockSignals(False)
        self.rebuild()
        self._host._graphs_tab.update_graph()

    def _move_selected_down(self) -> None:
        """Move all selected rows down by one position."""
        selected = self._selected_row_indices()
        n = self._session_table.rowCount()
        if not selected or selected[-1] >= n - 1:
            return
        self._session_table.blockSignals(True)
        for row in reversed(selected):
            below = self._snapshot_row(row + 1)
            current = self._snapshot_row(row)
            self._restore_row(row + 1, current)
            self._restore_row(row, below)
        self._session_table.blockSignals(False)
        # Re-select the moved rows
        self._session_table.clearSelection()
        for row in selected:
            new_row = row + 1
            for c in range(self._session_table.columnCount()):
                item = self._session_table.item(new_row, c)
                if item is not None:
                    item.setSelected(True)
        self._sync_order_from_table()
        self._sort_combo.blockSignals(True)
        self._sort_combo.setCurrentIndex(0)
        self._sort_combo.blockSignals(False)
        self.rebuild()
        self._host._graphs_tab.update_graph()

    # -- multi-select context menu / batch assign ---------------------

    def eventFilter(self, obj, event) -> bool:
        """Handle Ctrl+C / Ctrl+V on the session/factor table."""
        if obj is self._session_table and event.type() == QEvent.Type.KeyPress:
            if event.matches(QKeySequence.StandardKey.Copy):
                self._copy_selection()
                return True
            if event.matches(QKeySequence.StandardKey.Paste):
                self._paste_to_selection()
                return True
        return super().eventFilter(obj, event)

    def _copy_selection(self) -> None:
        """Copy selected cells to the clipboard as tab-separated text."""
        indexes = self._session_table.selectedIndexes()
        if not indexes:
            return
        rows = sorted({idx.row() for idx in indexes})
        cols = sorted({idx.column() for idx in indexes})
        selected_set = {(idx.row(), idx.column()) for idx in indexes}
        lines: list[str] = []
        for r in rows:
            parts: list[str] = []
            for c in cols:
                if (r, c) in selected_set:
                    item = self._session_table.item(r, c)
                    parts.append(item.text() if item is not None else "")
                else:
                    parts.append("")
            lines.append("\t".join(parts))
        QGuiApplication.clipboard().setText("\n".join(lines))

    def _paste_to_selection(self) -> None:
        """Paste clipboard text into the selected editable cells."""
        text = QGuiApplication.clipboard().text()
        if not text:
            return
        # Parse clipboard into a grid (rows split by \n, cells by \t)
        paste_rows = [row.split("\t") for row in text.splitlines()]
        if not paste_rows:
            return

        indexes = self._session_table.selectedIndexes()
        if not indexes:
            return

        # Anchor to the top-left selected cell
        min_row = min(idx.row() for idx in indexes)
        min_col = min(idx.column() for idx in indexes)

        single_value = len(paste_rows) == 1 and len(paste_rows[0]) == 1
        selected_set = {(idx.row(), idx.column()) for idx in indexes}

        self._session_table.blockSignals(True)
        try:
            if single_value:
                # Fill all selected cells with the single pasted value
                value = paste_rows[0][0]
                for r, c in selected_set:
                    self._write_cell(r, c, value)
            else:
                # Paste grid starting at anchor; only touch selected cells when
                # there's an active selection, otherwise paste freely from anchor.
                for pr, paste_row_vals in enumerate(paste_rows):
                    dest_row = min_row + pr
                    if dest_row >= self._session_table.rowCount():
                        break
                    for pc, value in enumerate(paste_row_vals):
                        dest_col = min_col + pc
                        if dest_col >= self._session_table.columnCount():
                            break
                        # Only write into cells that are part of the selection
                        # (when user pre-selected a target range) or all cells
                        # when the anchor alone was selected.
                        if len(indexes) <= 1 or (dest_row, dest_col) in selected_set:
                            self._write_cell(dest_row, dest_col, value)
        finally:
            self._session_table.blockSignals(False)

        self._host._sync_session_groups()
        self.rebuild()
        self._host._graphs_tab.update_graph()
        self._host._save_group_state()

    def _write_cell(self, row: int, col: int, value: str) -> None:
        """Write a value to an editable cell, persisting factor changes."""
        item = self._session_table.item(row, col)
        if item is None:
            return
        flags = item.flags()
        if not (flags & Qt.ItemFlag.ItemIsEditable):
            return  # skip read-only columns (Session name, Session Type)
        fi = col - self._FIXED_COLS
        factors = self._host._factor_definitions
        if fi < 0 or fi >= len(factors):
            return
        fname = factors[fi]
        label_item = self._session_table.item(row, 0)
        if label_item is None:
            return
        label = label_item.text()
        value = value.strip()
        if label not in self._host._session_factors:
            self._host._session_factors[label] = {}
        if value:
            self._host._session_factors[label][fname] = value
        else:
            self._host._session_factors[label].pop(fname, None)
        item.setText(value)

    def _show_session_context_menu(self, pos) -> None:
        """Right-click context menu for batch operations on selected rows."""
        selected = self._selected_row_indices()
        if not selected:
            return
        menu = QMenu(self)
        check_action = menu.addAction(f"Check {len(selected)} selected")
        uncheck_action = menu.addAction(f"Uncheck {len(selected)} selected")
        menu.addSeparator()

        factors = self._host._factor_definitions
        factor_actions: list[tuple] = []
        if factors:
            for fname in factors:
                action = menu.addAction(f"Set '{fname}' for {len(selected)} selected\u2026")
                factor_actions.append((action, fname))

        chosen = menu.exec(self._session_table.viewport().mapToGlobal(pos))
        if chosen is None:
            return

        if chosen is check_action:
            self._session_table.blockSignals(True)
            for row in selected:
                item = self._session_table.item(row, 0)
                if item:
                    item.setCheckState(Qt.CheckState.Checked)
            self._session_table.blockSignals(False)
            self.rebuild()
            self._host._graphs_tab.update_graph()
            return

        if chosen is uncheck_action:
            self._session_table.blockSignals(True)
            for row in selected:
                item = self._session_table.item(row, 0)
                if item:
                    item.setCheckState(Qt.CheckState.Unchecked)
            self._session_table.blockSignals(False)
            self.rebuild()
            self._host._graphs_tab.update_graph()
            return

        for action, fname in factor_actions:
            if chosen is action:
                self._batch_assign_factor(selected, fname)
                return

    def _assign_factor_to_selected(self) -> None:
        """Toolbar button: assign a factor level to all selected rows."""
        selected = self._selected_row_indices()
        if not selected:
            QMessageBox.information(
                self, "Assign Factor",
                "Select one or more sessions first (Ctrl+click or Shift+click).",
            )
            return
        factors = self._host._factor_definitions
        if not factors:
            QMessageBox.information(
                self, "Assign Factor",
                "Add at least one factor column first.",
            )
            return
        if len(factors) == 1:
            fname = factors[0]
        else:
            fname, ok = QInputDialog.getItem(
                self, "Select Factor",
                "Which factor to assign?",
                factors, 0, False,
            )
            if not ok:
                return
        self._batch_assign_factor(selected, fname)

    def _batch_assign_factor(self, rows: list[int], factor_name: str) -> None:
        """Prompt for a level and assign it to the given rows."""
        # Collect existing levels for auto-complete hints
        existing_levels = sorted({
            facs.get(factor_name, "")
            for facs in self._host._session_factors.values()
            if facs.get(factor_name, "")
        })
        if existing_levels:
            level, ok = QInputDialog.getItem(
                self, f"Set '{factor_name}'",
                f"Level for {len(rows)} session(s).\n"
                f"Choose an existing level or type a new one:",
                existing_levels, 0, True,
            )
        else:
            level, ok = QInputDialog.getText(
                self, f"Set '{factor_name}'",
                f"Level for {len(rows)} session(s):",
            )
        if not ok:
            return
        level = level.strip()
        # Find the factor column index
        factors = self._host._factor_definitions
        if factor_name not in factors:
            return
        col = self._FIXED_COLS + factors.index(factor_name)
        self._session_table.blockSignals(True)
        for row in rows:
            label_item = self._session_table.item(row, 0)
            if label_item is None:
                continue
            label = label_item.text()
            if label not in self._host._session_factors:
                self._host._session_factors[label] = {}
            if level:
                self._host._session_factors[label][factor_name] = level
            else:
                self._host._session_factors[label].pop(factor_name, None)
            cell_item = self._session_table.item(row, col)
            if cell_item is not None:
                cell_item.setText(level)
        self._session_table.blockSignals(False)
        self._host._sync_session_groups()
        self.rebuild()
        self._host._refresh_group_selectors()
        self._host._graphs_tab.update_graph()
        self._host._save_group_state()

    # -- factor management --------------------------------------------

    def _add_factor(self) -> None:
        """Prompt for a factor name and add a new column."""
        name, ok = QInputDialog.getText(
            self, "Add Factor",
            "Factor name (e.g. Sex, Treatment, Genotype):",
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self._host._factor_definitions:
            QMessageBox.information(
                self, "Add Factor", f"Factor '{name}' already exists."
            )
            return
        self._host._factor_definitions.append(name)
        # New factor defaults to splitting if it's the first one, else combine.
        self._host._facet_controls[name] = (
            FACET_SPLIT if len(self._host._factor_definitions) == 1 else FACET_COMBINE
        )
        self._host._sync_session_groups()
        self._refresh_session_table()
        self._host._refresh_group_selectors()
        self._host._save_group_state()

    def _remove_factor(self) -> None:
        """Remove the last factor column."""
        if not self._host._factor_definitions:
            QMessageBox.information(
                self, "Remove Factor", "No factors defined."
            )
            return
        removed = self._host._factor_definitions.pop()
        # Clean up session_factors
        for facs in self._host._session_factors.values():
            facs.pop(removed, None)
        # Drop the removed factor's facet control; sync re-derives the rest.
        self._host._facet_controls.pop(removed, None)
        self._host._sync_session_groups()
        self._refresh_session_table()
        self.rebuild()
        self._host._refresh_group_selectors()
        self._host._graphs_tab.update_graph()
        self._host._save_group_state()

    # -- table --------------------------------------------------------

    def rebuild(self) -> None:
        if self._rebuilding:
            return
        self._rebuilding = True
        try:
            self._refresh_session_table_if_changed()
            checked = self._checked_subjects()
            rows = [
                r for r in self._host._filtered_rows()
                if r["session_label"] in checked
            ]
            # Sort by user-defined subject order, then by behavior name
            ordered_labels = self._host.ordered_session_labels()
            label_rank = {lbl: i for i, lbl in enumerate(ordered_labels)}
            rows.sort(key=lambda r: (label_rank.get(r["session_label"], 9999), r["behavior"]))
            self._table.setRowCount(0)
            # Update column count to include all factors
            factors = self._host._factor_definitions
            n_data_cols = 5 + len(factors)  # Session, Type, [factors...], Behavior, N Bouts, Duration, Mean, Dist
            base_headers = ["Session", "Session Type"]
            factor_headers = list(factors)
            metric_headers = ["Behavior", "N Bouts",
                              "Total Duration (s)", "Mean Bout Duration (s)",
                              "Latency to First (s)",
                              "Distance Traveled (cm)"]
            self._table.setColumnCount(len(base_headers) + len(factor_headers) + len(metric_headers))
            self._table.setHorizontalHeaderLabels(
                base_headers + factor_headers + metric_headers
            )
            for row_data in rows:
                r = self._table.rowCount()
                self._table.insertRow(r)
                label = str(row_data["session_label"])
                stype = str(row_data.get("session_type", ""))
                session_facs = self._host._session_factors.get(label, {})
                factor_vals = [session_facs.get(f, "") for f in factors]
                vals = (
                    [label, stype]
                    + factor_vals
                    + [
                        str(row_data["behavior"]),
                        f"{row_data['n_bouts']:.0f}",
                        f"{row_data['time_spent_s']:.2f}",
                        f"{row_data['mean_bout_s']:.2f}",
                        f"{(float(row_data['latency_s']) if row_data.get('latency_s') is not None else float('nan')):.2f}",
                        f"{row_data.get('distance_cm', 0.0):.1f}",
                    ]
                )
                for c, v in enumerate(vals):
                    self._table.setItem(r, c, QTableWidgetItem(v))
        finally:
            self._rebuilding = False

    def _refresh_session_table_if_changed(self) -> None:
        current_labels = set(r["session_label"] for r in self._host._summary_rows)
        existing: list[str] = []
        for i in range(self._session_table.rowCount()):
            item = self._session_table.item(i, 0)
            if item is not None:
                existing.append(item.text())
        current_factor_count = len(self._host._factor_definitions)
        table_factor_count = max(0, self._session_table.columnCount() - self._FIXED_COLS)
        # Rebuild if labels changed or factor count changed
        if set(existing) != current_labels or current_factor_count != table_factor_count:
            self._refresh_session_table()

    # -- statistics ---------------------------------------------------

    def _run_statistics_dialog(self) -> None:
        if not _ensure_scipy():
            QMessageBox.warning(
                self, "Analytics",
                "scipy is not installed. Install it with: pip install scipy",
            )
            return

        factors = self._host._factor_definitions
        if not factors:
            msg = (
                "Add at least one factor and assign levels to sessions first.\n\n"
                "Use 'Add Factor' to create a grouping variable (e.g. Treatment, "
                "Session Type), then assign levels to each session in the table.\n"
                "Or click 'Auto-Group by Session Type' to populate groups automatically."
            )
            self._stats_output_text = msg
            self._stats_view_btn.setEnabled(True)
            QMessageBox.information(self, "Analytics", msg)
            return

        # Check that at least one factor has ≥2 levels assigned
        has_valid_factor = False
        for fname in factors:
            levels = {
                facs.get(fname, "")
                for facs in self._host._session_factors.values()
                if facs.get(fname, "")
            }
            if len(levels) >= 2:
                has_valid_factor = True
                break
        if not has_valid_factor:
            msg = (
                "At least one factor must have 2 or more levels assigned.\n\n"
                "Assign levels by typing into the factor column for each session "
                "in the table, or use 'Auto-Group by Session Type'."
            )
            self._stats_output_text = msg
            self._stats_view_btn.setEnabled(True)
            QMessageBox.information(self, "Analytics", msg)
            return

        checked = self._checked_subjects()
        rows = [r for r in self._host._filtered_rows() if r["session_label"] in checked]
        # Apply data range filter from the Graphs tab if active
        graphs_tab = self._host._graphs_tab
        if graphs_tab is not None and graphs_tab._is_data_range_active():
            rows = graphs_tab._recompute_rows_for_range(rows)
        # Apply bout filter from the Graphs tab if active
        if graphs_tab is not None and graphs_tab._is_bout_filter_active():
            rows = graphs_tab._recompute_rows_for_first_n(rows)
        if not rows:
            msg = "No data loaded or no sessions checked."
            self._stats_output_text = msg
            self._stats_view_btn.setEnabled(True)
            QMessageBox.information(self, "Analytics", msg)
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Statistical Tests")
        dlg.resize(440, 300)

        metric_combo = QComboBox(dlg)
        metric_combo.addItem("Bout Count", userData="n_bouts")
        metric_combo.addItem("Total Duration (s)", userData="time_spent_s")
        metric_combo.addItem("Mean Bout Duration (s)", userData="mean_bout_s")
        metric_combo.addItem("Latency to First (s)", userData="latency_s")
        metric_combo.addItem("Distance Traveled (cm)", userData="distance_cm")

        # Factor selection
        factor1_combo = QComboBox(dlg)
        for f in factors:
            factor1_combo.addItem(f)

        factor2_combo = QComboBox(dlg)
        factor2_combo.addItem("(none \u2014 one-way design)", userData="__none__")
        for f in factors:
            factor2_combo.addItem(f, userData=f)

        test_combo = QComboBox(dlg)
        test_combo.addItem("Auto (choose best test)", userData="auto")
        test_combo.addItem("Independent t-test", userData="ttest")
        test_combo.addItem("One-way ANOVA", userData="anova")
        test_combo.addItem("Two-way ANOVA", userData="anova2")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Metric:"))
        layout.addWidget(metric_combo)
        layout.addWidget(QLabel("Primary factor:"))
        layout.addWidget(factor1_combo)
        layout.addWidget(QLabel("Second factor (for two-way ANOVA):"))
        layout.addWidget(factor2_combo)
        layout.addWidget(QLabel("Test:"))
        layout.addWidget(test_combo)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        metric = str(metric_combo.currentData())
        test = str(test_combo.currentData())
        factor1_name = factor1_combo.currentText()
        factor2_data = str(factor2_combo.currentData() or "__none__")
        factor2_name = factor2_data if factor2_data != "__none__" else ""

        if factor2_name and factor2_name == factor1_name:
            QMessageBox.information(
                self, "Analytics",
                "Primary and second factors must be different.",
            )
            return

        df = pd.DataFrame(rows)
        agg_fn = "mean" if metric == "mean_bout_s" else "sum"
        sess_agg = df.groupby("session_label")[metric].agg(agg_fn).reset_index()

        # Map factor levels onto sessions
        sess_agg["_factor1"] = sess_agg["session_label"].map(
            lambda lbl: self._host._session_factors.get(lbl, {}).get(factor1_name, "")
        )
        sess_agg = sess_agg[sess_agg["_factor1"] != ""]
        if factor2_name:
            sess_agg["_factor2"] = sess_agg["session_label"].map(
                lambda lbl: self._host._session_factors.get(lbl, {}).get(factor2_name, "")
            )
            sess_agg = sess_agg[sess_agg["_factor2"] != ""]

        if sess_agg.empty:
            self._stats_output_text = "No sessions have levels assigned for the selected factor(s)."
            self._stats_view_btn.setEnabled(True)
            return

        # Determine the actual test to run
        group_names_1 = sorted(sess_agg["_factor1"].unique())
        if test == "auto":
            if factor2_name:
                test = "anova2"
            elif len(group_names_1) == 2:
                test = "ttest"
            else:
                test = "anova"
        elif test == "anova2" and not factor2_name:
            self._stats_output_text = "Two-way ANOVA requires a second factor. Select one and try again."
            self._stats_view_btn.setEnabled(True)
            return

        # \u2500\u2500 Two-way ANOVA \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if test == "anova2" and factor2_name:
            self._run_two_way_anova(
                sess_agg, metric, metric_combo.currentText(),
                factor1_name, factor2_name,
            )
            return

        # \u2500\u2500 One-way tests (t-test / one-way ANOVA) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        group_arrays = [
            sess_agg.loc[sess_agg["_factor1"] == g, metric].to_numpy(dtype=float)
            for g in group_names_1
        ]

        min_n = min(len(a) for a in group_arrays) if group_arrays else 0
        if min_n < 2:
            self._stats_output_text = (
                "Each group must have at least 2 sessions.\n"
                f"Group sizes: {', '.join(f'{g}={len(a)}' for g, a in zip(group_names_1, group_arrays))}"
            )
            self._stats_view_btn.setEnabled(True)
            return

        lines: list[str] = []
        lines.append(f"Metric: {metric_combo.currentText()}")
        lines.append(f"Factor: {factor1_name}")
        lines.append(f"Groups: {', '.join(f'{g} (n={len(a)})' for g, a in zip(group_names_1, group_arrays))}")
        for g, a in zip(group_names_1, group_arrays):
            lines.append(f"  {g}: mean={np.mean(a):.3f}, std={np.std(a, ddof=1):.3f}")
        lines.append("")

        stat, pval = float("nan"), float("nan")
        if test == "ttest" and len(group_arrays) == 2 and ttest_ind is not None:
            _res = ttest_ind(group_arrays[0], group_arrays[1])
            stat, pval = float(_res[0]), float(_res[1])
            lines.append(f"Independent t-test: t={stat:.4f}, p={pval:.6f}")
            lines.append(_significance_label(pval))
        elif test == "anova" and f_oneway is not None:
            _res = f_oneway(*group_arrays)
            stat, pval = float(_res[0]), float(_res[1])
            lines.append(f"One-way ANOVA: F={stat:.4f}, p={pval:.6f}")
            lines.append(_significance_label(pval))

            # Automatic post-hoc pairwise comparisons (Sidak correction)
            if pval < 0.05 and len(group_names_1) > 2:
                lines.append("")
                lines.append("Post-hoc pairwise comparisons (Sidak correction):")
                lines.append("-" * 50)
                n_comparisons = len(group_names_1) * (len(group_names_1) - 1) // 2
                for i in range(len(group_names_1)):
                    for j in range(i + 1, len(group_names_1)):
                        pw_res = ttest_ind(group_arrays[i], group_arrays[j])
                        pw_p = float(pw_res[1])
                        pw_p_adj = 1.0 - (1.0 - pw_p) ** n_comparisons
                        pw_p_adj = min(pw_p_adj, 1.0)
                        sig = "***" if pw_p_adj < 0.001 else "**" if pw_p_adj < 0.01 else "*" if pw_p_adj < 0.05 else "ns"
                        lines.append(
                            f"  {group_names_1[i]} vs {group_names_1[j]}: "
                            f"t={float(pw_res[0]):.4f}, p={pw_p:.6f}, "
                            f"p(adj)={pw_p_adj:.6f} {sig}"
                        )

        self._stats_output_text = "\n".join(lines)
        self._stats_view_btn.setEnabled(True)

        # Store for graphs significance overlay
        self._host._last_stats_result = {
            "metric": metric,
            "test": test,
            "stat": stat,
            "pval": pval,
            "groups": group_names_1,
        }
        self._host._graphs_tab.update_graph()

    def _run_two_way_anova(
        self,
        sess_agg: "pd.DataFrame",
        metric: str,
        metric_label: str,
        factor1_name: str,
        factor2_name: str,
    ) -> None:
        """Run a Type-I two-way ANOVA with interaction term."""
        from scipy.stats import f as f_dist  # type: ignore[import-untyped]

        values = sess_agg[metric].to_numpy(dtype=float)
        fa = sess_agg["_factor1"].to_numpy()
        fb = sess_agg["_factor2"].to_numpy()
        n_total = len(values)
        grand_mean = float(np.mean(values))

        levels_a = sorted(set(fa))
        levels_b = sorted(set(fb))

        # SS factor A (main effect)
        ss_a = 0.0
        for a in levels_a:
            mask = fa == a
            n_a = int(np.sum(mask))
            if n_a > 0:
                ss_a += n_a * (float(np.mean(values[mask])) - grand_mean) ** 2

        # SS factor B (main effect)
        ss_b = 0.0
        for b in levels_b:
            mask = fb == b
            n_b = int(np.sum(mask))
            if n_b > 0:
                ss_b += n_b * (float(np.mean(values[mask])) - grand_mean) ** 2

        # SS interaction
        ss_ab = 0.0
        for a in levels_a:
            for b in levels_b:
                mask = (fa == a) & (fb == b)
                n_cell = int(np.sum(mask))
                if n_cell > 0:
                    cell_mean = float(np.mean(values[mask]))
                    a_mean = float(np.mean(values[fa == a]))
                    b_mean = float(np.mean(values[fb == b]))
                    ss_ab += n_cell * (cell_mean - a_mean - b_mean + grand_mean) ** 2

        ss_total = float(np.sum((values - grand_mean) ** 2))
        ss_error = ss_total - ss_a - ss_b - ss_ab

        # DF
        df_a = len(levels_a) - 1
        df_b = len(levels_b) - 1
        df_ab = df_a * df_b
        n_cells = len(levels_a) * len(levels_b)
        df_error = n_total - n_cells

        if df_error <= 0 or ss_error <= 0:
            self._stats_output_text = (
                "Two-way ANOVA cannot be computed:\n"
                "not enough observations per cell (need at least 1 replicate per cell\n"
                "and residual degrees of freedom > 0).\n\n"
                f"Design: {len(levels_a)} x {len(levels_b)} ({n_cells} cells), n={n_total}"
            )
            self._stats_view_btn.setEnabled(True)
            return

        ms_a = ss_a / df_a if df_a > 0 else 0
        ms_b = ss_b / df_b if df_b > 0 else 0
        ms_ab = ss_ab / df_ab if df_ab > 0 else 0
        ms_error = ss_error / df_error

        lines: list[str] = []
        lines.append(f"Two-way ANOVA \u2014 Metric: {metric_label}")
        lines.append(f"  Factor A: {factor1_name} ({', '.join(levels_a)})")
        lines.append(f"  Factor B: {factor2_name} ({', '.join(levels_b)})")
        lines.append(f"  Design: {len(levels_a)} x {len(levels_b)}, n={n_total}")
        lines.append("")
        lines.append(f"{'Source':<24s} {'SS':>10s} {'df':>4s} {'MS':>10s} {'F':>10s} {'p':>10s}")
        lines.append("-" * 74)

        def _row(label: str, ss: float, df: int, ms: float) -> str:
            if df > 0 and ms_error > 0:
                f_val = ms / ms_error
                p_val = 1.0 - float(f_dist.cdf(f_val, df, df_error))
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                return (
                    f"{label:<24s} {ss:10.3f} {df:4d} {ms:10.3f} {f_val:10.4f} {p_val:10.6f}  {sig}"
                )
            _dash = "\u2014"
            return f"{label:<24s} {ss:10.3f} {df:4d} {ms:10.3f} {_dash:>10s} {_dash:>10s}"

        lines.append(_row(factor1_name, ss_a, df_a, ms_a))
        lines.append(_row(factor2_name, ss_b, df_b, ms_b))
        lines.append(_row(f"{factor1_name} x {factor2_name}", ss_ab, df_ab, ms_ab))
        lines.append(f"{'Residual':<24s} {ss_error:10.3f} {df_error:4d} {ms_error:10.3f}")
        lines.append("-" * 74)
        lines.append(f"{'Total':<24s} {ss_total:10.3f} {n_total - 1:4d}")

        # Cell means summary
        lines.append("")
        lines.append("Cell means:")
        header = f"{'':>16s}"
        for b in levels_b:
            header += f"  {b:>12s}"
        lines.append(header)
        for a in levels_a:
            row_str = f"{a:>16s}"
            for b in levels_b:
                mask = (fa == a) & (fb == b)
                n_cell = int(np.sum(mask))
                if n_cell > 0:
                    cell_mean = float(np.mean(values[mask]))
                    row_str += f"  {cell_mean:10.3f}({n_cell})"
                else:
                    row_str += "  " + "\u2014".rjust(12)
            lines.append(row_str)

        self._stats_output_text = "\n".join(lines)
        self._stats_view_btn.setEnabled(True)

        # Store primary factor's result for graph overlay
        if df_a > 0 and ms_error > 0:
            f_val = ms_a / ms_error
            p_val = 1.0 - float(f_dist.cdf(f_val, df_a, df_error))
        else:
            f_val, p_val = float("nan"), float("nan")
        self._host._last_stats_result = {
            "metric": metric,
            "test": "anova2",
            "stat": f_val,
            "pval": p_val,
            "groups": levels_a,
        }
        self._host._graphs_tab.update_graph()

    def _show_summary_stats_popup(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Statistics Results")
        dlg.resize(600, 440)
        layout = QVBoxLayout(dlg)
        te = QTextEdit(dlg)
        te.setReadOnly(True)
        te.setPlainText(self._stats_output_text or "(No results yet — run statistics first.)")
        te.setStyleSheet(
            "QTextEdit{background:#0A1929;color:#cfd8dc;font-family:Consolas,monospace;"
            "font-size:11px;border:1px solid #1E3A5F;border-radius:4px;}"
        )
        layout.addWidget(te, 1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        dlg.exec()

    def _export_csv(self) -> None:
        checked = self._checked_subjects()
        rows = [r for r in self._host._filtered_rows() if r["session_label"] in checked]
        if not rows:
            QMessageBox.information(self, "Export", "No data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary to CSV", "", "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        df = pd.DataFrame(rows)
        # Add all factor columns
        for fname in self._host._factor_definitions:
            df[fname] = df["session_label"].map(
                lambda lbl, fn=fname: self._host._session_factors.get(lbl, {}).get(fn, "")
            )
        # Keep backward-compat "group" column from active factor
        df["group"] = df["session_label"].map(self._host._session_groups)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        self._host._status.setText(f"Exported summary CSV to {path}")

    # -- auto-group by session type -----------------------------------

    def _auto_group_by_session_type(self) -> None:
        """Assign session-type labels into a 'Session Type' factor column."""
        types = self._host._detected_session_types()
        if not types:
            QMessageBox.information(
                self, "Auto-Group",
                "No distinct session types detected.\n"
                "Session types are derived from video filenames\n"
                "(e.g. Subject_Conditioning.mp4 \u2192 Conditioning).",
            )
            return
        # Ensure a "Session Type" factor column exists
        factor_name = "Session Type"
        if factor_name not in self._host._factor_definitions:
            self._host._factor_definitions.append(factor_name)
        # Split on Session Type unless the user already split another factor.
        if not any(
            v == FACET_SPLIT for v in self._host._facet_controls.values()
        ):
            self._host._facet_controls[factor_name] = FACET_SPLIT
        # Populate the factor
        for label in sorted({r["session_label"] for r in self._host._summary_rows}):
            for sid, lbl in self._host._session_label_by_session.items():
                if lbl == label:
                    stype = self._host._session_type_by_session.get(sid, "")
                    if stype:
                        if label not in self._host._session_factors:
                            self._host._session_factors[label] = {}
                        self._host._session_factors[label][factor_name] = stype
                    break
        self._host._sync_session_groups()
        self._refresh_session_table()
        self.rebuild()
        self._host._refresh_group_selectors()
        self._host._graphs_tab.update_graph()
        self._host._status.setText(
            f"Auto-grouped sessions by type: {', '.join(types)}"
        )
        self._host._save_group_state()


def _significance_label(pval: float) -> str:
    if pval < 0.001:
        return "Result: Highly significant (p < 0.001)"
    if pval < 0.05:
        return "Result: Significant (p < 0.05)"
    return "Result: Not significant (p >= 0.05)"


# ======================================================================
# Sub-tab 2: Graphs
# ======================================================================

class _GraphsWidget(QWidget):
    """All chart visualisation types."""

    # toggle-button style sheet
    _BTN_STYLE = (
        "QPushButton{padding:3px 9px;border:1px solid #37474f;border-radius:3px;"
        "background:#1a2027;color:#cfd8dc;}"
        "QPushButton:checked{background:#1565c0;border-color:#1565c0;color:#fff;}"
        "QPushButton:hover:!checked{background:#263238;}"
    )

    def __init__(self, host: BehaviorAnalyticsTab) -> None:
        super().__init__()
        self._host = host
        self._updating = False  # guard against recursive updates
        self._session_end_s_cache: dict[str, float] = {}
        self._bin_cache: pd.DataFrame | None = None  # cached _bin_bouts result
        self._bin_cache_key: tuple[Any, ...] = ()  # key to invalidate cache
        # Ethogram cache: pre-processed {bid: {sess_label: [(start_s, dur_s), ...]}}
        # Does NOT include session filter — session changes just re-render from cache.
        self._ethogram_cache: dict[str, dict[str, list[tuple[float, float]]]] | None = None
        self._ethogram_cache_key: tuple[Any, ...] = ()

        def _toggle_row(options: list[tuple[str, str]], default_idx: int = 0):
            """Return (layout, button_group, {key: QPushButton})."""
            grp = QButtonGroup(self)
            grp.setExclusive(True)
            row = QHBoxLayout()
            row.setSpacing(3)
            btns: dict[str, QPushButton] = {}
            for i, (label, key) in enumerate(options):
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.setChecked(i == default_idx)
                btn.setStyleSheet(self._BTN_STYLE)
                btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                btn.setMaximumHeight(26)
                grp.addButton(btn, i)
                row.addWidget(btn)
                btns[key] = btn
            row.addStretch(1)
            return row, grp, btns

        # Chart style — two rows so buttons fit in a narrow left panel
        self._style_grp = QButtonGroup(self)
        self._style_grp.setExclusive(True)
        self._style_btns: dict[str, QPushButton] = {}
        _style_opts_r1 = [("Bar", "bar"), ("Stacked", "stacked"), ("Line", "line"), ("Box", "box"), ("Over Time", "overtime")]
        _style_opts_r2 = [("Ethogram", "ethogram"), ("Pie", "pie"), ("Overview", "overview")]
        style_row1 = QHBoxLayout(); style_row1.setSpacing(3)
        style_row2 = QHBoxLayout(); style_row2.setSpacing(3)
        for _sopts, _srow in [(_style_opts_r1, style_row1), (_style_opts_r2, style_row2)]:
            for _slbl, _skey in _sopts:
                _sbtn = QPushButton(_slbl)
                _sbtn.setCheckable(True)
                _sbtn.setChecked(_skey == "bar")
                _sbtn.setStyleSheet(self._BTN_STYLE)
                _sbtn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                _sbtn.setMaximumHeight(26)
                self._style_grp.addButton(_sbtn)
                self._style_btns[_skey] = _sbtn
                _srow.addWidget(_sbtn)
        style_row1.addStretch(1)
        style_row2.addStretch(1)

        # Metric — two rows
        self._metric_grp = QButtonGroup(self)
        self._metric_grp.setExclusive(True)
        self._metric_btns: dict[str, QPushButton] = {}
        _metric_opts_r1 = [("Bout Count", "n_bouts"), ("Duration (s)", "time_spent_s"), ("Mean Dur. (s)", "mean_bout_s")]
        _metric_opts_r2 = [("Latency (s)", "latency_s"), ("Distance (cm)", "distance_cm")]
        metric_row1 = QHBoxLayout(); metric_row1.setSpacing(3)
        metric_row2 = QHBoxLayout(); metric_row2.setSpacing(3)
        for _mopts, _mrow in [(_metric_opts_r1, metric_row1), (_metric_opts_r2, metric_row2)]:
            for _mlbl, _mkey in _mopts:
                _mbtn = QPushButton(_mlbl)
                _mbtn.setCheckable(True)
                _mbtn.setChecked(_mkey == "n_bouts")
                _mbtn.setStyleSheet(self._BTN_STYLE)
                _mbtn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                _mbtn.setMaximumHeight(26)
                self._metric_grp.addButton(_mbtn)
                self._metric_btns[_mkey] = _mbtn
                _mrow.addWidget(_mbtn)
        metric_row1.addStretch(1)
        metric_row2.addStretch(1)

        # Row 3 — individual vs group
        _mode_opts = [("Individual Sessions", "individual"), ("By Group", "group")]
        mode_row, self._mode_grp, self._mode_btns = _toggle_row(_mode_opts)

        self._style_grp.idClicked.connect(lambda _: self.update_graph())
        self._metric_grp.idClicked.connect(lambda _: self.update_graph())
        self._mode_grp.idClicked.connect(lambda _: self.update_graph())

        # Row 4 — secondary options
        self._time_bin_spin = QSpinBox()
        self._time_bin_spin.setRange(10, 3600)
        self._time_bin_spin.setValue(300)
        self._time_bin_spin.setSuffix(" s")
        self._time_bin_spin.setToolTip("Time-bin size in seconds (used for time-course charts).")
        self._time_bin_spin.editingFinished.connect(self.update_graph)

        self._settings_btn = QPushButton("Settings…")
        self._settings_btn.setToolTip("Adjust fonts, DPI, legend placement.")
        self._settings_btn.clicked.connect(self._open_settings_dialog)

        self._export_btn = QPushButton("Export…")
        self._export_btn.setToolTip("Save graph to PNG, SVG, or PDF.")
        self._export_btn.clicked.connect(self._export_figure)

        self._export_data_btn = QPushButton("Export Data…")
        self._export_data_btn.setToolTip("Export the underlying graph data as CSV.")
        self._export_data_btn.clicked.connect(self._export_graph_data)

        self._export_excel_btn = QPushButton("Export Excel…")
        self._export_excel_btn.setToolTip(
            "Export binned data to Excel (one sheet per behavior, columns per subject)."
        )
        self._export_excel_btn.clicked.connect(self._export_excel_data)

        self._groups_btn = QPushButton("Groups…")
        self._groups_btn.setToolTip("Reorder groups and customise group colours.")
        self._groups_btn.clicked.connect(self._open_groups_dialog)

        self._level_order_btn = QPushButton("Level Order…")
        self._level_order_btn.setToolTip(
            "Set the display order of levels within each factor. "
            "Applies to all charts including factor interactions."
        )
        self._level_order_btn.clicked.connect(self._host._open_level_order_dialog)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setToolTip("Redraw the graph with the current settings.")
        self._apply_btn.clicked.connect(self.update_graph)
        self._apply_btn.setStyleSheet(
            "QPushButton{padding:3px 14px;background:#1565c0;border:none;"
            "border-radius:3px;color:#fff;font-weight:bold;}"
            "QPushButton:hover{background:#1976d2;}"
        )

        # Row 5 — axis range overrides
        self._x_min = QDoubleSpinBox()
        self._x_min.setRange(-1e6, 1e6)
        self._x_min.setSpecialValueText("auto")
        self._x_min.setValue(self._x_min.minimum())  # "auto"
        self._x_min.setDecimals(1)
        self._x_min.setMinimumWidth(90)

        self._x_max = QDoubleSpinBox()
        self._x_max.setRange(-1e6, 1e6)
        self._x_max.setSpecialValueText("auto")
        self._x_max.setValue(self._x_max.minimum())
        self._x_max.setDecimals(1)
        self._x_max.setMinimumWidth(90)

        self._y_min = QDoubleSpinBox()
        self._y_min.setRange(-1e6, 1e6)
        self._y_min.setSpecialValueText("auto")
        self._y_min.setValue(self._y_min.minimum())
        self._y_min.setDecimals(1)
        self._y_min.setMinimumWidth(90)

        self._y_max = QDoubleSpinBox()
        self._y_max.setRange(-1e6, 1e6)
        self._y_max.setSpecialValueText("auto")
        self._y_max.setValue(self._y_max.minimum())
        self._y_max.setDecimals(1)
        self._y_max.setMinimumWidth(90)

        # Row 6 — data range (seconds) — filters which bouts contribute
        self._data_min_s = QDoubleSpinBox()
        self._data_min_s.setRange(-1e6, 1e6)
        self._data_min_s.setSpecialValueText("auto")
        self._data_min_s.setValue(self._data_min_s.minimum())
        self._data_min_s.setDecimals(1)
        self._data_min_s.setSuffix(" s")
        self._data_min_s.setMinimumWidth(100)
        self._data_min_s.setToolTip(
            "Include only bouts starting at or after this time (seconds). "
            "Affects bar graphs, statistics, and all aggregated metrics."
        )

        self._data_max_s = QDoubleSpinBox()
        self._data_max_s.setRange(-1e6, 1e6)
        self._data_max_s.setSpecialValueText("auto")
        self._data_max_s.setValue(self._data_max_s.minimum())
        self._data_max_s.setDecimals(1)
        self._data_max_s.setSuffix(" s")
        self._data_max_s.setMinimumWidth(100)
        self._data_max_s.setToolTip(
            "Include only bouts ending at or before this time (seconds). "
            "Affects bar graphs, statistics, and all aggregated metrics."
        )

        # Row 4 — faceted grouping: one combine/split/level dropdown per factor.
        self._facet = _FacetControls("Group by:")
        self._facet.setToolTip(
            "For each factor choose:\n"
            "  • — combine —  pool across that factor\n"
            "  • — split —    one series per level\n"
            "  • a level      keep only that level\n"
            "Split two or more factors to plot their interaction."
        )
        self._facet.changed.connect(self._on_facets_changed)

        # -- Ethogram session filter (shown only when Ethogram style is active) --
        self._ethogram_session_list = QListWidget()
        self._ethogram_session_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._ethogram_session_list.setMaximumHeight(120)
        self._ethogram_session_list.setStyleSheet(
            "QListWidget{background:#0A1929;border:1px solid #1E3A5F;border-radius:3px;"
            "color:#cfd8dc;font-size:10px;}"
        )
        self._ethogram_check_all_btn = QPushButton("All")
        self._ethogram_check_all_btn.setMaximumWidth(40)
        self._ethogram_check_all_btn.setStyleSheet(self._BTN_STYLE)
        self._ethogram_check_all_btn.clicked.connect(self._check_all_ethogram_sessions)
        self._ethogram_check_none_btn = QPushButton("None")
        self._ethogram_check_none_btn.setMaximumWidth(48)
        self._ethogram_check_none_btn.setStyleSheet(self._BTN_STYLE)
        self._ethogram_check_none_btn.clicked.connect(self._uncheck_all_ethogram_sessions)

        _ethogram_sess_btns = QVBoxLayout()
        _ethogram_sess_btns.addWidget(self._ethogram_check_all_btn)
        _ethogram_sess_btns.addWidget(self._ethogram_check_none_btn)
        _ethogram_sess_btns.addStretch(1)
        _ethogram_sess_lbl = QLabel("Sessions:")
        _ethogram_sess_lbl.setMinimumWidth(52)
        _ethogram_sess_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        _ethogram_sess_row = QHBoxLayout()
        _ethogram_sess_row.setSpacing(4)
        _ethogram_sess_row.addWidget(_ethogram_sess_lbl)
        _ethogram_sess_row.addWidget(self._ethogram_session_list, 1)
        _ethogram_sess_row.addLayout(_ethogram_sess_btns)

        self._ethogram_filter_widget = QWidget()
        _ef_vbox = QVBoxLayout(self._ethogram_filter_widget)
        _ef_vbox.setSpacing(0)
        _ef_vbox.setContentsMargins(0, 0, 0, 0)
        _ef_vbox.addLayout(_ethogram_sess_row)
        self._ethogram_filter_widget.setVisible(False)

        action_row1 = QHBoxLayout()
        action_row1.setSpacing(6)
        _bin_lbl_g = QLabel("Bin:")
        _bin_lbl_g.setStyleSheet("color:#90a4ae;font-size:10px;")
        action_row1.addWidget(_bin_lbl_g)
        action_row1.addWidget(self._time_bin_spin)
        action_row1.addWidget(self._settings_btn)
        action_row1.addWidget(self._groups_btn)
        action_row1.addWidget(self._level_order_btn)
        action_row1.addStretch(1)

        action_row2 = QHBoxLayout()
        action_row2.setSpacing(6)
        action_row2.addWidget(self._export_btn)
        action_row2.addWidget(self._export_data_btn)
        action_row2.addWidget(self._export_excel_btn)
        action_row2.addWidget(self._apply_btn)
        action_row2.addStretch(1)

        x_axis_row = QHBoxLayout()
        x_axis_row.setSpacing(6)
        x_axis_row.addWidget(QLabel("min"))
        x_axis_row.addWidget(self._x_min)
        x_axis_row.addWidget(QLabel("max"))
        x_axis_row.addWidget(self._x_max)
        x_axis_row.addStretch(1)

        y_axis_row = QHBoxLayout()
        y_axis_row.setSpacing(6)
        y_axis_row.addWidget(QLabel("min"))
        y_axis_row.addWidget(self._y_min)
        y_axis_row.addWidget(QLabel("max"))
        y_axis_row.addWidget(self._y_max)
        y_axis_row.addStretch(1)

        data_range_row = QHBoxLayout()
        data_range_row.setSpacing(6)
        data_range_row.addWidget(QLabel("min"))
        data_range_row.addWidget(self._data_min_s)
        data_range_row.addWidget(QLabel("max"))
        data_range_row.addWidget(self._data_max_s)
        data_range_row.addStretch(1)

        # Row 7 — bout filter (First N / Bouts Until Behavior)
        self._bout_filter_mode = QComboBox()
        self._bout_filter_mode.addItem("Disabled", userData="disabled")
        self._bout_filter_mode.addItem("First N Bouts", userData="first_n")
        self._bout_filter_mode.addItem("Bouts Until Behavior", userData="until_behavior")
        self._bout_filter_mode.setMinimumWidth(140)
        self._bout_filter_mode.setToolTip(
            "Disabled: include all bouts.\n"
            "First N Bouts: keep only the first N bouts per behavior per session.\n"
            "Bouts Until Behavior: include all bouts of each behavior that occur "
            "before the first occurrence of a selected target behavior per session."
        )

        self._first_n_bouts = QSpinBox()
        self._first_n_bouts.setRange(1, 100000)
        self._first_n_bouts.setValue(5)
        self._first_n_bouts.setMinimumWidth(60)
        self._first_n_bouts.setToolTip("Number of bouts to include per behavior per session.")
        self._first_n_bouts.setVisible(False)
        self._first_n_bouts.editingFinished.connect(self.update_graph)

        self._until_behavior_combo = QComboBox()
        self._until_behavior_combo.setMinimumWidth(120)
        self._until_behavior_combo.setToolTip(
            "Include all bouts of each behavior that occur before the first "
            "bout of this target behavior in each session."
        )
        self._until_behavior_combo.setVisible(False)
        self._until_behavior_combo.currentIndexChanged.connect(
            lambda _: self.update_graph()
        )

        self._until_scale_chk = QCheckBox("Scale by pre-behavior time")
        self._until_scale_chk.setVisible(False)
        self._until_scale_chk.setToolTip(
            "Divide the metric (Bout Count or Total Duration) by the time elapsed "
            "before the target behavior's first occurrence, normalized to the chosen "
            "time base (e.g. 60\u202fs\u202f=\u202fper minute).\n"
            "If the target behavior never occurred, the full session duration (or "
            "the Data Range max, if set) is used as the denominator."
        )

        self._until_scale_spin = QDoubleSpinBox()
        self._until_scale_spin.setRange(1.0, 86400.0)
        self._until_scale_spin.setValue(60.0)
        self._until_scale_spin.setSuffix(" s")
        self._until_scale_spin.setDecimals(1)
        self._until_scale_spin.setMinimumWidth(70)
        self._until_scale_spin.setToolTip(
            "Time base for normalization.\n60 = per minute | 3600 = per hour"
        )
        self._until_scale_spin.setVisible(False)
        self._until_scale_spin.editingFinished.connect(self.update_graph)

        def _on_until_scale_toggled(checked: bool) -> None:
            self._until_scale_spin.setVisible(checked)
            self.update_graph()

        self._until_scale_chk.toggled.connect(_on_until_scale_toggled)

        def _on_bout_filter_mode_changed(_idx: int) -> None:
            mode = self._bout_filter_mode.currentData()
            is_until = mode == "until_behavior"
            self._first_n_bouts.setVisible(mode == "first_n")
            self._until_behavior_combo.setVisible(is_until)
            self._until_scale_chk.setVisible(is_until)
            self._until_scale_spin.setVisible(is_until and self._until_scale_chk.isChecked())
            self.update_graph()

        self._bout_filter_mode.currentIndexChanged.connect(_on_bout_filter_mode_changed)

        bout_filter_row = QHBoxLayout()
        bout_filter_row.setSpacing(6)
        bout_filter_row.addWidget(self._bout_filter_mode)
        bout_filter_row.addWidget(self._first_n_bouts)
        bout_filter_row.addWidget(self._until_behavior_combo)
        bout_filter_row.addStretch(1)

        _until_scale_row = QHBoxLayout()
        _until_scale_row.setSpacing(6)
        _until_scale_row.setContentsMargins(8, 0, 0, 0)
        _until_scale_row.addWidget(self._until_scale_chk)
        _until_scale_row.addWidget(self._until_scale_spin)
        _until_scale_row.addStretch(1)

        def _labeled_row(label: str, row_layout: QHBoxLayout) -> QHBoxLayout:
            outer = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(52)
            lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
            outer.addWidget(lbl)
            outer.addLayout(row_layout)
            return outer

        controls_widget = QWidget()
        ctrl_vbox = QVBoxLayout(controls_widget)
        ctrl_vbox.setSpacing(2)
        ctrl_vbox.setContentsMargins(4, 4, 4, 4)
        _chart_hdr = QLabel("Chart:")
        _chart_hdr.setStyleSheet("color:#90a4ae;font-size:10px;")
        ctrl_vbox.addWidget(_chart_hdr)
        ctrl_vbox.addLayout(style_row1)
        ctrl_vbox.addLayout(style_row2)
        _metric_hdr = QLabel("Metric:")
        _metric_hdr.setStyleSheet("color:#90a4ae;font-size:10px;")
        ctrl_vbox.addWidget(_metric_hdr)
        ctrl_vbox.addLayout(metric_row1)
        ctrl_vbox.addLayout(metric_row2)
        ctrl_vbox.addLayout(_labeled_row("View:", mode_row))
        ctrl_vbox.addWidget(self._facet)
        ctrl_vbox.addStretch(1)
        ctrl_vbox.addWidget(self._ethogram_filter_widget)
        ctrl_vbox.addLayout(action_row1)
        ctrl_vbox.addLayout(action_row2)
        # X/Y axis range overrides live in the Settings… dialog to keep the
        # panel uncluttered; x_axis_row/y_axis_row hold the persistent spinboxes.
        ctrl_vbox.addLayout(_labeled_row("Data Range:", data_range_row))
        ctrl_vbox.addLayout(_labeled_row("Bout Filter:", bout_filter_row))
        ctrl_vbox.addLayout(_until_scale_row)

        # -- canvas ---------------------------------------------------
        self._figure: Any = None
        self._canvas: Any = None
        self._toolbar: Any = None
        self._placeholder = QLabel("Graph will appear after loading analytics data.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setMinimumHeight(260)
        self._placeholder.setStyleSheet(
            "border: 1px solid #1A2027; background: #0A1929; "
            "border-radius: 4px; color: #546E7A;"
        )
        self._canvas_scroll: Any = None
        if _ensure_matplotlib() and Figure is not None and FigureCanvas is not None and NavigationToolbar is not None:
            _dpi = int(self._host._graph_settings.get("dpi", 100))
            _init_px_w = int(self._host._graph_settings.get("max_w", 700))
            _init_px_h = int(self._host._graph_settings.get("max_h", 420))
            self._figure = Figure(figsize=(_init_px_w / _dpi, _init_px_h / _dpi))
            self._canvas = FigureCanvas(self._figure)
            # Fixed size: canvas renders at exactly (max_w x max_h) pixels.
            # The enclosing scroll area expands to fill available GUI space and
            # provides scrollbars only when the canvas is taller than the viewport
            # (e.g. many-panel ethograms).
            self._canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._canvas.setFixedSize(_init_px_w, _init_px_h)
            self._toolbar = NavigationToolbar(self._canvas, self)
            self._placeholder.setVisible(False)
            from PySide6.QtWidgets import QScrollArea as _QScrollArea
            self._canvas_scroll = _QScrollArea()
            self._canvas_scroll.setWidget(self._canvas)
            # widgetResizable=False: canvas keeps its explicit fixed size;
            # the scroll area shows scrollbars when the canvas is larger.
            self._canvas_scroll.setWidgetResizable(False)
            self._canvas_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._canvas_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            # Dynamic resize: canvas fills the available viewport width
            self._graphs_resize_filter = _ViewportResizeFilter(
                self._sync_canvas_to_viewport, self
            )
            self._canvas_scroll.viewport().installEventFilter(self._graphs_resize_filter)

        # -- layout: splitter (controls left, canvas right) -----------
        from PySide6.QtWidgets import QSplitter as _QSplitterG
        from PySide6.QtWidgets import QScrollArea as _QScrollAreaG
        _left_g = _QScrollAreaG()
        _left_g.setWidget(controls_widget)
        _left_g.setWidgetResizable(True)
        # Min width keeps the control buttons (Export…, Settings…, Bin, etc.)
        # fully visible even when the user drags the splitter left to enlarge
        # the plot; a horizontal scrollbar appears only below this width.
        _left_g.setMinimumWidth(340)
        # No maximum width: the splitter must be draggable in both directions so
        # the plot area can be shrunk as well as grown.
        _left_g.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        _right_g = QWidget()
        _right_vbox_g = QVBoxLayout(_right_g)
        _right_vbox_g.setContentsMargins(0, 0, 0, 0)
        _right_vbox_g.setSpacing(2)
        if self._toolbar is not None:
            _right_vbox_g.addWidget(self._toolbar)
        if self._canvas_scroll is not None:
            _right_vbox_g.addWidget(self._canvas_scroll, 1)
        _right_vbox_g.addWidget(self._placeholder, 1)

        _splitter_g = _QSplitterG(Qt.Orientation.Horizontal)
        _splitter_g.setChildrenCollapsible(False)
        _splitter_g.addWidget(_left_g)
        _splitter_g.addWidget(_right_g)
        _splitter_g.setStretchFactor(0, 0)
        _splitter_g.setStretchFactor(1, 1)
        _splitter_g.setSizes([360, 1000])
        # Kept as an attribute so showEvent can re-apply proportional sizes once
        # the widget has real geometry — at construction time the left scroll
        # area's wide content gives it an oversized hint, so the splitter would
        # otherwise hand most of the width to the controls and leave the plot
        # small until the user dragged the handle.
        self._splitter_g = _splitter_g
        self._splitter_g_init = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(_splitter_g, 1)

    # -- graph dispatch -----------------------------------------------

    def _auto_resize_figure(
        self,
        n_behaviors: int,
        n_groups: int,
        style: str,
        mode: str,
    ) -> None:
        """Resize the matplotlib figure and the canvas widget to fit the content.

        Calculates an appropriate figure size based on the number of
        sub-panels (behaviors × groups for multi-panel views) and
        updates the scroll area so the user can scroll when the figure
        is taller than the visible area.
        """
        if self._figure is None or self._canvas is None:
            return

        # Per-cell size targets (inches)
        gs = self._host._graph_settings
        dpi = int(gs.get("dpi", 100))
        max_w = int(gs.get("max_w", 700))
        max_h = int(gs.get("max_h", 420))

        # Natural figure dimensions in inches, computed from content.
        # Use the max_w as the reference width so cells scale properly.
        ref_w_in = max_w / dpi  # reference width in inches
        CELL_W = ref_w_in / 2.5  # ~2-3 columns fill max_w nicely
        CELL_H = CELL_W * 0.85   # slightly taller than wide
        MIN_W, MIN_H = ref_w_in * 0.8, ref_w_in * 0.5

        if style in ("bar", "line", "box", "overview") and n_behaviors > 1 and mode == "group":
            ncols = min(n_behaviors, 3)
            nrows = (n_behaviors + ncols - 1) // ncols
            fig_w = max(MIN_W, ncols * CELL_W)
            fig_h = max(MIN_H, nrows * CELL_H)
        elif style in ("bar", "line", "box") and mode == "individual":
            ncols = min(n_behaviors, 3)
            nrows = (n_behaviors + ncols - 1) // ncols
            fig_w = max(MIN_W, ncols * CELL_W)
            fig_h = max(MIN_H, nrows * CELL_H)
        elif style == "overtime":
            fig_w = max(MIN_W, min(n_behaviors, 3) * CELL_W)
            fig_h = max(MIN_H, ((n_behaviors + 2) // 3) * CELL_H)
        elif style == "ethogram":
            # Ethogram height grows with behavior count — allow vertical scroll
            fig_h = max(MIN_H, n_behaviors * 0.55 + 0.8)
            fig_w = ref_w_in
        else:
            fig_w, fig_h = ref_w_in, ref_w_in * 0.58

        # Extra width for many groups in single-behavior group mode
        if mode == "group" and n_behaviors == 1 and n_groups > 4:
            fig_w = max(fig_w, n_groups * 0.6 + 0.8)

        # Use the available viewport width when it is larger than max_w.
        if self._canvas_scroll is not None:
            vp_w = self._canvas_scroll.viewport().width()
            effective_max_w = max(max_w, vp_w)
        else:
            effective_max_w = max_w

        # Cap width at effective_max_w; allow height to exceed max_h (scroll
        # area will show a vertical scrollbar for very tall multi-panel charts).
        px_w = min(int(fig_w * dpi), effective_max_w)
        px_h = int(fig_h * dpi)   # may exceed max_h for tall content

        self._figure.set_size_inches(px_w / dpi, px_h / dpi)

        # Commit the canvas to exactly (px_w x px_h) device pixels so the
        # scroll area provides scrollbars when px_h > visible height.
        if self._canvas is not None:
            self._canvas.setFixedSize(px_w, px_h)

    def _sync_canvas_to_viewport(self) -> None:
        """Expand canvas width to fill the available scroll area viewport.

        Called when the viewport is resized (window resize or splitter move).
        Height is kept proportional to max_h / max_w so the aspect ratio is
        maintained.  Multi-panel charts (ethogram, over-time) may still
        exceed max_h and will be scrollable.
        """
        if self._canvas_scroll is None or self._canvas is None or self._figure is None:
            return
        gs = self._host._graph_settings
        dpi = int(gs.get("dpi", 100))
        base_w = int(gs.get("max_w", 700))
        base_h = int(gs.get("max_h", 420))
        # Current canvas height — preserve it (may be taller for multi-panel charts)
        cur_h = self._canvas.height()
        # Scale width to viewport; scale height proportionally only if it
        # matches the default ratio (i.e. not an over-sized ethogram).
        ratio = base_h / max(base_w, 1)
        default_h = int(base_w * ratio)
        if abs(cur_h - default_h) <= 2:
            eff_ratio = ratio
        else:
            # Multi-panel chart: keep the current height/width proportion
            eff_ratio = cur_h / max(self._canvas.width(), 1)
        # Stable fill width: reserve the scrollbar gutter up-front so the canvas
        # doesn't shimmer between scrollbar-shown/hidden states.
        vp_w = _stable_fill_width(
            self._canvas_scroll, lambda w: w * eff_ratio, min_w=200
        )
        new_h = max(120, int(vp_w * eff_ratio))
        try:
            self._figure.set_size_inches(vp_w / dpi, new_h / dpi)
            self._canvas.setFixedSize(vp_w, new_h)
            self._canvas.updateGeometry()
        except Exception:
            pass

    def update_graph(self) -> None:
        if self._figure is None or self._updating:
            return
        self._updating = True
        try:
            self._refresh_factor_selector()
            style = self._get_style()
            self._ethogram_filter_widget.setVisible(style == "ethogram")
            if style == "ethogram":
                self._refresh_ethogram_session_filter()
            checked = self._host._summary_tab._checked_subjects()
            selected_bids = self._host._selected_behavior_ids()
            metric = self._get_metric()

            # When viewing distance (either via the metric toggle or the
            # behavior filter), pull the Distance Traveled pseudo-behavior
            # rows directly and force the metric to distance_cm.
            if metric == "distance_cm" or (selected_bids == {DISTANCE_BEHAVIOR_ID}):
                metric = "distance_cm"
                rows = [
                    r for r in self._host._summary_rows
                    if r["behavior_id"] == DISTANCE_BEHAVIOR_ID
                    and r["session_label"] in checked
                ]
            else:
                rows = [
                    r for r in self._host._filtered_rows()
                    if r["session_label"] in checked
                ]
            # Apply group filter when in group mode
            checked_groups = self._checked_groups()
            mode = self._get_mode()
            if mode == "group" and checked_groups:
                groups_map = self._host._session_groups
                rows = [
                    r for r in rows
                    if groups_map.get(r["session_label"], "") in checked_groups
                ]
            # Apply data range filter — recompute aggregated metrics from
            # raw bouts clipped to the user-specified time window.
            if self._is_data_range_active():
                rows = self._recompute_rows_for_range(rows)
            # Apply bout filter — first N or bouts-until-behavior.
            if self._is_bout_filter_active():
                rows = self._recompute_rows_for_first_n(rows)
            rows = self._apply_latency_fallbacks(rows)
            if not rows:
                self._figure.clear()
                ax = self._figure.add_subplot(111)
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                self._canvas.draw_idle()
                return

            df = pd.DataFrame(rows)
            style = self._get_style()
            mode = self._get_mode()
            mlabel = self._metric_label(metric)
            agg_fn = "mean" if metric == "mean_bout_s" else "sum"

            behaviors = sorted(df["behavior"].unique())
            multi_behavior = len(behaviors) > 1
            _subplot_styles = {"bar", "line", "box", "overview"}

            # Resize the figure and scroll canvas to fit the content
            n_groups = len({
                self._host._session_groups.get(r["session_label"], "")
                for r in rows if self._host._session_groups.get(r["session_label"], "")
            }) or 1
            self._auto_resize_figure(len(behaviors), n_groups, style, mode)
            self._figure.clear()
            _subplot_right_ratio = 0.82  # updated by legend placement below
            _use_rect_layout = False

            if multi_behavior and style in _subplot_styles and mode == "group":
                n_beh = len(behaviors)
                ncols = min(n_beh, 3)
                nrows = (n_beh + ncols - 1) // ncols
                for bi, bname in enumerate(behaviors):
                    ax_i = self._figure.add_subplot(nrows, ncols, bi + 1)
                    bdf = df[df["behavior"] == bname]
                    if style == "bar":
                        self._bar_groups_on_ax(ax_i, bdf, metric, mlabel, agg_fn, title=str(bname))
                    elif style == "line":
                        self._line_timecourse_on_ax(ax_i, bdf, metric, mlabel)
                    elif style == "box":
                        self._box_plot_on_ax(ax_i, bdf, metric, mlabel, agg_fn)
                    elif style == "overview":
                        self._overview_on_ax(ax_i, bdf, str(bname))
                # Collect handles/labels from first subplot, remove all per-ax
                # legends, then place a single shared legend outside the grid.
                _shared_handles: list[Any] = []
                _shared_labels: list[str] = []
                for _ax in self._figure.axes:
                    _ax_leg = _ax.get_legend()
                    if _ax_leg is not None:
                        if not _shared_handles:
                            try:
                                _leg_h = list(_ax_leg.legend_handles)
                            except AttributeError:
                                _leg_h = list(getattr(_ax_leg, "legendHandles", []))
                            _leg_l = [_t.get_text() for _t in _ax_leg.get_texts()]
                            if _leg_h and _leg_l:
                                _shared_handles, _shared_labels = _leg_h, _leg_l
                        _ax_leg.remove()
                    if not _shared_handles:
                        _h, _l = _ax.get_legend_handles_labels()
                        if _h:
                            _shared_handles, _shared_labels = _h, _l
                if _shared_handles:
                    _gs_leg = self._gs()
                    _fig_w_px = self._canvas.width() if self._canvas else int(self._host._graph_settings.get("max_w", 700))
                    _rr, _lx = _legend_right_margin(_shared_labels, _fig_w_px)
                    # Hide x-tick labels from all subplots — the right-side
                    # legend already maps colours to group names.
                    for _ax in self._figure.axes:
                        _ax.set_xticklabels([])
                    self._figure.legend(
                        _shared_handles, _shared_labels,
                        fontsize=_gs_leg.get("legend_fontsize", "small"),
                        loc="upper left",
                        bbox_to_anchor=(_lx, 0.98),
                        bbox_transform=self._figure.transFigure,
                        frameon=True,
                        framealpha=0.9,
                    )
                    _subplot_right_ratio = _rr
                    _use_rect_layout = True
            elif style == "bar":
                if mode == "individual":
                    self._bar_individual(df, metric, mlabel, agg_fn)
                else:
                    self._bar_groups(df, metric, mlabel, agg_fn)
                    # Move group legend to the right side (same as multi-behavior mode)
                    _bar_side_handles: list[Any] = []
                    _bar_side_labels: list[str] = []
                    for _ax in self._figure.axes:
                        _ax_leg = _ax.get_legend()
                        if _ax_leg is not None:
                            if not _bar_side_handles:
                                try:
                                    _leg_h = list(_ax_leg.legend_handles)
                                except AttributeError:
                                    _leg_h = list(getattr(_ax_leg, "legendHandles", []))
                                _leg_l = [_t.get_text() for _t in _ax_leg.get_texts()]
                                if _leg_h and _leg_l:
                                    _bar_side_handles, _bar_side_labels = _leg_h, _leg_l
                            _ax_leg.remove()
                        if not _bar_side_handles:
                            _h, _l = _ax.get_legend_handles_labels()
                            if _h:
                                _bar_side_handles, _bar_side_labels = _h, _l
                    if _bar_side_handles:
                        _gs_leg = self._gs()
                        _fig_w_px = self._canvas.width() if self._canvas else int(self._host._graph_settings.get("max_w", 700))
                        _rr, _lx = _legend_right_margin(_bar_side_labels, _fig_w_px)
                        self._figure.legend(
                            _bar_side_handles, _bar_side_labels,
                            fontsize=_gs_leg.get("legend_fontsize", "small"),
                            loc="upper left",
                            bbox_to_anchor=(_lx, 0.98),
                            bbox_transform=self._figure.transFigure,
                            frameon=True,
                            framealpha=0.9,
                        )
                        _subplot_right_ratio = _rr
                        _use_rect_layout = True
            elif style == "stacked":
                title_sfx = "by Group" if mode == "group" else "by Session"
                self._stacked_bar(df, metric, mlabel, f"{mlabel} {title_sfx} (Stacked)")
            elif style == "line":
                self._line_timecourse(df)
            elif style == "box":
                self._box_plot(df)
            elif style == "overtime":
                if mode == "individual":
                    self._time_binned_chart(df)
                else:
                    self._time_binned_group_chart(df)
            elif style == "ethogram":
                self._ethogram(df)
            elif style == "pie":
                self._pie_chart(df)
            elif style == "overview":
                self._combined_overview(df)
            else:
                ax = self._figure.add_subplot(111)
                ax.text(0.5, 0.5, "Unknown chart type", ha="center", va="center",
                        transform=ax.transAxes)
            try:
                if _use_rect_layout:
                    self._figure.tight_layout(pad=1.2, rect=[0, 0, _subplot_right_ratio, 1])
                else:
                    self._figure.tight_layout(pad=1.2)
            except Exception:
                pass
            if self._host._graph_settings.get("force_fit", False):
                gs_ff = self._gs()

                def _fit() -> None:
                    _force_fit_canvas(
                        self._canvas, self._figure,
                        int(gs_ff.get("max_w", 700)), int(gs_ff.get("max_h", 420)),
                        int(gs_ff.get("dpi", 100)),
                    )
                    # Tight-fitting caps the canvas at max_w×max_h, which leaves
                    # the plot small in a wide window. Expand it to fill the
                    # available viewport width (preserving the just-computed
                    # aspect) so the initial render uses the full plot area —
                    # matching what a manual splitter drag produces. Without this
                    # the plot stayed capped until a resize event happened to fire.
                    self._sync_canvas_to_viewport()
                # Draw first so get_tightbbox measures the realised figure — on
                # the first render the renderer isn't ready yet, so an inline-only
                # fit collapses to the minimum size (canvas appears miniaturised
                # until a manual resize). Re-fit on the next layout pass.
                self._canvas.draw_idle()
                _fit()
                QTimer.singleShot(0, lambda: (_fit(), self._canvas.draw_idle()))
            else:
                # Fill the freshly-drawn figure to the viewport width so the
                # initial render uses the full plot area (no manual drag needed).
                # The deferred call re-runs after Qt finalises the layout pass —
                # on the first render the viewport width is not yet realised, so
                # the inline call alone would leave the canvas under-sized.
                self._sync_canvas_to_viewport()
                QTimer.singleShot(0, self._sync_canvas_to_viewport)
                self._canvas.draw_idle()
        finally:
            self._updating = False

    def showEvent(self, event: Any) -> None:  # type: ignore[override]
        """Re-fill the canvas when the tab becomes visible.

        ``update_graph`` may run while this tab is hidden (e.g. data loaded
        from cache while another sub-tab is active), when the viewport reports a
        provisional width.  Re-syncing on show guarantees the plot fills the
        real viewport once the user navigates here.
        """
        super().showEvent(event)

        def _settle() -> None:
            # First real show: the splitter now has true geometry, so pin the
            # controls to a fixed width and give the rest to the plot.  Done
            # once so we don't clobber a deliberate user drag on later shows.
            if not self._splitter_g_init:
                splitter = getattr(self, "_splitter_g", None)
                if splitter is not None:
                    total = splitter.width()
                    if total > 400:
                        left = 360
                        self._splitter_g_init = True
                        splitter.setSizes([left, max(400, total - left)])
                        # setSizes lands on the next layout pass; sync the
                        # canvas after Qt re-flows so it reads the true width.
                        QTimer.singleShot(0, self._sync_canvas_to_viewport)
            self._sync_canvas_to_viewport()

        QTimer.singleShot(0, _settle)

    # -- style / metric / mode accessors ------------------------------

    def _get_style(self) -> str:
        for key, btn in self._style_btns.items():
            if btn.isChecked():
                return key
        return "bar"

    def _get_metric(self) -> str:
        for key, btn in self._metric_btns.items():
            if btn.isChecked():
                return key
        return "n_bouts"

    def _get_mode(self) -> str:
        for key, btn in self._mode_btns.items():
            if btn.isChecked():
                return key
        return "individual"

    def _metric_label(self, metric: str) -> str:
        base = {
            "n_bouts": "Bout Count",
            "time_spent_s": "Total Duration (s)",
            "mean_bout_s": "Mean Bout Duration (s)",
            "latency_s": "Latency to First (s)",
            "distance_cm": "Distance Traveled (cm)",
        }.get(metric, metric)
        if self._is_until_scaling_active() and metric in ("n_bouts", "time_spent_s"):
            sf = self._get_until_scale_factor()
            if sf == 60.0:
                rate_sfx = "/ min"
            elif sf == 3600.0:
                rate_sfx = "/ hr"
            else:
                rate_sfx = f"/ {sf:.0f}s"
            base = base.replace(" (s)", "") + f" ({rate_sfx})"
        return base

    # -- faceted grouping controls -------------------------------------

    def _refresh_factor_selector(self) -> None:
        """Rebuild the per-factor facet dropdowns when factors change."""
        self._facet.blockSignals(True)
        self._facet.rebuild(
            list(self._host._factor_definitions), self._host._levels_by_factor()
        )
        # Seed controls if the host has none yet (first load / fresh project).
        if not self._host._facet_controls and self._host._factor_definitions:
            self._host._facet_controls = self._host._default_facet_controls()
        # Keep groups + normalized controls consistent with the current data.
        self._host._sync_session_groups()
        self._facet.set_state(self._host._facet_controls)
        self._facet.blockSignals(False)

    def _refresh_group_filter(self) -> None:
        """No-op retained for callers: facets filter via the dropdowns."""
        return

    def _on_facets_changed(self) -> None:
        """User changed a facet dropdown — recompute groups and redraw."""
        if self._updating:
            return
        self._host._facet_controls = self._facet.state()
        self._host._sync_session_groups()
        self.update_graph()
        self._host._save_group_state()

    # Back-compat alias for the old combo signal name.
    _on_factor_selector_changed = _on_facets_changed

    def _checked_groups(self) -> set[str]:
        """All series the facet controls currently produce (no separate filter)."""
        return {g for g in self._host._session_groups.values() if g}

    # -- ethogram session filter --------------------------------------

    def _refresh_ethogram_session_filter(self) -> None:
        """Rebuild the ethogram session checkboxes when sessions change."""
        all_sessions = sorted(self._host._summary_tab._checked_subjects())
        existing: list[str] = []
        for i in range(self._ethogram_session_list.count()):
            cb = self._ethogram_session_list.itemWidget(self._ethogram_session_list.item(i))
            if isinstance(cb, QCheckBox):
                existing.append(cb.text())
        if existing == all_sessions:
            return
        prev_checked = self._checked_ethogram_sessions()
        self._ethogram_session_list.blockSignals(True)
        self._ethogram_session_list.clear()
        for sess in all_sessions:
            item = QListWidgetItem()
            cb = QCheckBox(sess)
            cb.setChecked(not prev_checked or sess in prev_checked)
            cb.stateChanged.connect(lambda _, s=self: QTimer.singleShot(0, s.update_graph))
            item.setSizeHint(cb.sizeHint())
            self._ethogram_session_list.addItem(item)
            self._ethogram_session_list.setItemWidget(item, cb)
        self._ethogram_session_list.blockSignals(False)

    def _checked_ethogram_sessions(self) -> set[str]:
        """Return the set of session labels checked in the ethogram filter."""
        out: set[str] = set()
        for i in range(self._ethogram_session_list.count()):
            item = self._ethogram_session_list.item(i)
            if item is None:
                continue
            cb = self._ethogram_session_list.itemWidget(item)
            if isinstance(cb, QCheckBox) and cb.isChecked():
                out.add(cb.text())
        return out

    def _check_all_ethogram_sessions(self) -> None:
        for i in range(self._ethogram_session_list.count()):
            cb = self._ethogram_session_list.itemWidget(self._ethogram_session_list.item(i))
            if isinstance(cb, QCheckBox):
                cb.setChecked(True)

    def _uncheck_all_ethogram_sessions(self) -> None:
        for i in range(self._ethogram_session_list.count()):
            cb = self._ethogram_session_list.itemWidget(self._ethogram_session_list.item(i))
            if isinstance(cb, QCheckBox):
                cb.setChecked(False)

    # -- helpers ------------------------------------------------------

    def _gs(self) -> dict[str, Any]:
        """Return graph settings with fonts scaled to the current figure size.

        Font sizes are stored at a *reference* figure width (7 in) and scaled
        linearly so that text remains legible whether the graph is small or
        large.  All plot methods call this helper, so scaling is automatic.
        """
        gs = self._host._graph_settings
        if self._figure is not None:
            w_in = self._figure.get_figwidth()
            REFERENCE_W = 7.0  # inches at which the stored font sizes were chosen
            scale = max(0.55, min(1.6, w_in / REFERENCE_W))
            return {
                **gs,
                "title_fontsize": max(7, int(gs.get("title_fontsize", 12) * scale)),
                "axis_fontsize": max(7, int(gs.get("axis_fontsize", 10) * scale)),
                "tick_fontsize": max(6, int(gs.get("tick_fontsize", 8) * scale)),
            }
        return gs

    def _get_data_range_seconds(self) -> tuple[float | None, float | None]:
        """Return (min_seconds, max_seconds) from the Data Range controls.

        Returns None for either end when set to 'auto' (the sentinel minimum).
        """
        lo = self._data_min_s.value()
        hi = self._data_max_s.value()
        lo_val = None if lo == self._data_min_s.minimum() else lo
        hi_val = None if hi == self._data_max_s.minimum() else hi
        return lo_val, hi_val

    def _session_analysis_end_s(self, session_id: str) -> float:
        """Return analysis timeline end (seconds) for a session."""
        sid = str(session_id)
        cached = self._session_end_s_cache.get(sid)
        if cached is not None:
            return cached

        fps = max(1e-9, float(self._host._project_fps()))
        end_s = 0.0

        # Prefer imported merged-session end-time overrides when available.
        override_end = self._host._session_end_s_overrides.get(sid)
        if override_end is not None:
            try:
                ov = float(override_end)
                if ov > 0:
                    self._session_end_s_cache[sid] = ov
                    return ov
            except Exception:
                pass

        # Preferred source: pose length for true session duration.
        try:
            pose = self._host._get_pose_for_session(sid)
            if pose is not None and int(getattr(pose, "n_frames", 0)) > 0:
                n_frames = int(getattr(pose, "n_frames", 0))
                end_s = max(0.0, float(n_frames) / fps)
        except Exception:
            end_s = 0.0

        # Fallback: latest observed bout endpoint across all behaviors.
        if end_s <= 0.0:
            max_end_frame = -1
            for bdf in self._host._raw_bouts.values():
                if bdf.empty or not {"session_id", "end_frame"}.issubset(bdf.columns):
                    continue
                try:
                    grp = bdf[bdf["session_id"].astype(str) == sid]
                    if grp.empty:
                        continue
                    local_max = int(pd.to_numeric(grp["end_frame"], errors="coerce").max())
                    if local_max > max_end_frame:
                        max_end_frame = local_max
                except Exception:
                    continue
            if max_end_frame >= 0:
                end_s = float(max_end_frame + 1) / fps

        self._session_end_s_cache[sid] = float(end_s)
        return float(end_s)

    def _latency_fallback_seconds(
        self,
        session_id: str,
        lo_s: float | None,
        hi_s: float | None,
    ) -> float:
        """Fallback latency when the behavior never occurs."""
        if hi_s is not None:
            return float(hi_s)
        end_s = self._session_analysis_end_s(session_id)
        if lo_s is not None:
            return max(float(lo_s), float(end_s))
        return float(end_s)

    def _apply_latency_fallbacks(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace missing latency with max-possible time for graph/export paths."""
        if self._get_metric() != "latency_s":
            return rows

        lo_s, hi_s = self._get_data_range_seconds()
        changed = False
        out: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("behavior_id", "")) == DISTANCE_BEHAVIOR_ID:
                out.append(row)
                continue

            n_bouts = float(row.get("n_bouts", 0.0) or 0.0)
            lat_raw = row.get("latency_s", float("nan"))
            try:
                lat_val = float(lat_raw)
            except Exception:
                lat_val = float("nan")

            if n_bouts <= 0.0 or not np.isfinite(lat_val):
                sid = str(row.get("session_id", ""))
                updated = dict(row)
                updated["latency_s"] = self._latency_fallback_seconds(sid, lo_s, hi_s)
                out.append(updated)
                changed = True
            else:
                out.append(row)
        return out if changed else rows

    def _is_data_range_active(self) -> bool:
        lo, hi = self._get_data_range_seconds()
        return lo is not None or hi is not None

    def _get_first_n_bouts(self) -> int:
        """Return the first-N-bouts limit, or 0 when inactive."""
        if self._bout_filter_mode.currentData() == "first_n":
            return int(self._first_n_bouts.value())
        return 0

    def _is_first_n_active(self) -> bool:
        return self._bout_filter_mode.currentData() == "first_n"

    def _get_until_behavior_id(self) -> str | None:
        """Return the target behavior ID for 'bouts until behavior' mode, or None."""
        if self._bout_filter_mode.currentData() != "until_behavior":
            return None
        bid = self._until_behavior_combo.currentData()
        return bid if bid else None

    def _is_until_behavior_active(self) -> bool:
        return self._get_until_behavior_id() is not None

    def _is_until_scaling_active(self) -> bool:
        """Return True when 'Scale by pre-behavior time' is enabled in until-behavior mode."""
        return (
            self._bout_filter_mode.currentData() == "until_behavior"
            and self._until_scale_chk.isChecked()
        )

    def _get_until_scale_factor(self) -> float:
        """Return the normalization time base in seconds (e.g. 60 for per-minute)."""
        return max(1.0, float(self._until_scale_spin.value()))

    def _is_bout_filter_active(self) -> bool:
        """Return True if any bout-filter mode is active."""
        return self._is_first_n_active() or self._is_until_behavior_active()

    def _refresh_until_behavior_combo(self) -> None:
        """Populate the 'until behavior' combo from the host's behavior list."""
        prev = self._until_behavior_combo.currentData()
        self._until_behavior_combo.blockSignals(True)
        self._until_behavior_combo.clear()
        for b in self._host._behaviors.behaviors:
            if str(b.behavior_id) == NO_BEHAVIOR_ID:
                continue
            bid = str(b.behavior_id)
            label = str(b.name or bid)
            self._until_behavior_combo.addItem(label, userData=bid)
        # Restore previous selection if still valid
        if prev:
            idx = self._until_behavior_combo.findData(prev)
            if idx >= 0:
                self._until_behavior_combo.setCurrentIndex(idx)
        self._until_behavior_combo.blockSignals(False)

    def _cutoff_frames_for_until_behavior(self) -> dict[str, float]:
        """Return {session_id: first_start_frame} for the target behavior.

        For each session, finds the start_frame of the earliest bout of the
        target behavior. Bouts of any behavior whose start_frame is >=
        this cutoff should be excluded.
        """
        target_bid = self._get_until_behavior_id()
        if not target_bid:
            return {}
        raw = self._host._raw_bouts
        target_df = raw.get(target_bid)
        if target_df is None or target_df.empty:
            return {}
        cutoffs: dict[str, float] = {}
        for sid, grp in target_df.groupby("session_id"):
            cutoffs[str(sid)] = float(grp["start_frame"].min())
        return cutoffs

    def _recompute_rows_for_first_n(
        self, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Re-aggregate summary rows keeping only filtered bouts per session/behavior.

        Supports two modes:
        - First N: keep only the earliest N bouts per session/behavior.
        - Until Behavior: keep only bouts that start before the first
          occurrence of a target behavior in each session.
        """
        n_limit = self._get_first_n_bouts()         # >0 in first_n mode
        until_bid = self._get_until_behavior_id()    # non-None in until mode
        if n_limit <= 0 and until_bid is None:
            return rows

        fps = self._host._project_fps()
        raw = self._host._raw_bouts
        if not raw:
            return rows

        # Pre-compute per-session cutoff frames for "until behavior" mode
        cutoffs: dict[str, float] = {}
        if until_bid is not None:
            cutoffs = self._cutoff_frames_for_until_behavior()

        wanted: dict[tuple[str, str], dict[str, Any]] = {}
        for r in rows:
            key = (r["session_label"], r["behavior_id"])
            wanted[key] = r

        result: list[dict[str, Any]] = []
        processed: set[tuple[str, str]] = set()

        for bid, bdf in raw.items():
            if bdf.empty or not {"start_frame", "end_frame", "session_id"}.issubset(bdf.columns):
                continue
            for sid, grp in bdf.groupby("session_id"):
                slbl = self._host._session_label_by_session.get(str(sid), str(sid))
                key = (slbl, bid)
                if key not in wanted:
                    continue
                template = wanted[key]

                sorted_grp = grp.sort_values("start_frame")
                if n_limit > 0:
                    sorted_grp = sorted_grp.head(n_limit)
                elif until_bid is not None:
                    cutoff = cutoffs.get(str(sid))
                    if cutoff is not None:
                        sorted_grp = sorted_grp[sorted_grp["start_frame"] < cutoff]
                    # If no cutoff (target never occurred), keep all bouts

                starts = sorted_grp["start_frame"].to_numpy(dtype=np.float64) / fps
                ends = (sorted_grp["end_frame"].to_numpy(dtype=np.float64) + 1) / fps
                durations = ends - starts
                valid = durations > 0
                starts = starts[valid]
                durations = durations[valid]

                n_bouts = float(len(durations))
                time_s = float(durations.sum())
                mean_dur = time_s / n_bouts if n_bouts > 0 else 0.0
                latency_s = float(starts.min()) if n_bouts > 0 else float("nan")

                # Per-time scaling for "until behavior" mode
                if until_bid is not None and self._is_until_scaling_active():
                    scale_factor = self._get_until_scale_factor()
                    cutoff_f = cutoffs.get(str(sid))
                    if cutoff_f is not None:
                        pre_time_s = float(cutoff_f) / fps
                    else:
                        # Target never occurred: use data-range max or full session duration
                        _lo, _hi = self._get_data_range_seconds()
                        pre_time_s = float(_hi) if _hi is not None else self._session_analysis_end_s(str(sid))
                    if pre_time_s > 0:
                        n_bouts = n_bouts / pre_time_s * scale_factor
                        time_s = time_s / pre_time_s * scale_factor
                    else:
                        n_bouts = float("nan")
                        time_s = float("nan")

                result.append({
                    **template,
                    "n_bouts": n_bouts,
                    "time_spent_s": time_s,
                    "mean_bout_s": mean_dur,
                    "latency_s": latency_s,
                })
                processed.add(key)

        for key, template in wanted.items():
            if key not in processed:
                if is_pseudo_behavior_id(template.get("behavior_id", "")):
                    result.append(template)
                else:
                    result.append({
                        **template,
                        "n_bouts": 0.0,
                        "time_spent_s": 0.0,
                        "mean_bout_s": 0.0,
                        "latency_s": float("nan"),
                    })

        return result

    def _recompute_rows_for_range(
        self, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Re-aggregate summary rows from raw bouts using the Data Range filter.

        When a data range is active, the pre-aggregated summary rows (which
        cover the entire session) are replaced by values computed only from
        bouts that overlap the specified time window.
        """
        lo_s, hi_s = self._get_data_range_seconds()
        if lo_s is None and hi_s is None:
            return rows  # no range filter — use original rows

        fps = self._host._project_fps()
        raw = self._host._raw_bouts
        if not raw:
            return rows  # no raw data to recompute from

        # Build a set of (session_label, behavior_id) pairs present in rows
        wanted: dict[tuple[str, str], dict[str, Any]] = {}
        for r in rows:
            key = (r["session_label"], r["behavior_id"])
            wanted[key] = r  # keep last for template fields

        result: list[dict[str, Any]] = []
        processed: set[tuple[str, str]] = set()

        for bid, bdf in raw.items():
            if bdf.empty or not {"start_frame", "end_frame", "session_id"}.issubset(bdf.columns):
                continue
            for sid, grp in bdf.groupby("session_id"):
                slbl = self._host._session_label_by_session.get(str(sid), str(sid))
                key = (slbl, bid)
                if key not in wanted:
                    continue
                template = wanted[key]

                starts = grp["start_frame"].to_numpy(dtype=np.float64) / fps
                ends = (grp["end_frame"].to_numpy(dtype=np.float64) + 1) / fps

                # Clip bouts to the data range
                if lo_s is not None:
                    mask = ends > lo_s
                    starts = starts[mask]
                    ends = ends[mask]
                    starts = np.maximum(starts, lo_s)
                if hi_s is not None:
                    mask = starts < hi_s
                    starts = starts[mask]
                    ends = ends[mask]
                    ends = np.minimum(ends, hi_s)

                durations = ends - starts
                valid = durations > 0
                starts = starts[valid]
                ends = ends[valid]
                durations = durations[valid]

                n_bouts = float(len(durations))
                time_s = float(durations.sum())
                mean_dur = time_s / n_bouts if n_bouts > 0 else 0.0
                latency_s = float(starts.min()) if n_bouts > 0 else float("nan")

                result.append({
                    **template,
                    "n_bouts": n_bouts,
                    "time_spent_s": time_s,
                    "mean_bout_s": mean_dur,
                    "latency_s": latency_s,
                })
                processed.add(key)

        # Keep rows that had no raw bout data (e.g. zero-rows for subjects
        # with no bouts, or the distance pseudo-behavior)
        for key, template in wanted.items():
            if key not in processed:
                if is_pseudo_behavior_id(template.get("behavior_id", "")):
                    # Pseudo-behavior rows (distance / ROI) have no raw bouts —
                    # keep their whole-session values as-is.
                    result.append(template)
                else:
                    result.append({
                        **template,
                        "n_bouts": 0.0,
                        "time_spent_s": 0.0,
                        "mean_bout_s": 0.0,
                        "latency_s": float("nan"),
                    })

        return result

    def _apply(self, ax: Any, title: str = "", ylabel: str = "", xlabel: str = "") -> None:
        gs = self._gs()
        if title:
            ax.set_title(title, fontsize=gs["title_fontsize"])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=gs["axis_fontsize"])
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=gs["axis_fontsize"])
        ax.tick_params(axis="both", labelsize=gs["tick_fontsize"])
        # User axis-range overrides (auto when at minimum sentinel)
        if self._x_min.value() != self._x_min.minimum():
            ax.set_xlim(left=self._x_min.value())
        if self._x_max.value() != self._x_max.minimum():
            ax.set_xlim(right=self._x_max.value())
        if self._y_min.value() != self._y_min.minimum():
            ax.set_ylim(bottom=self._y_min.value())
        if self._y_max.value() != self._y_max.minimum():
            ax.set_ylim(top=self._y_max.value())

    def _annotate_stats(self, ax: Any, x_positions: list[float], y_top: float) -> None:
        """Draw a significance bracket above two bars if stats results are available."""
        if not self._gs().get("show_stats", True):
            return
        ls = self._host._last_stats_result
        if not ls or len(x_positions) != 2:
            return
        pval = ls.get("pval")
        if pval is None or np.isnan(pval):
            return
        # Only annotate if the metric matches
        if ls.get("metric") != self._get_metric():
            return
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        pad = max(y_top * 0.08, 0.10)
        y_line = y_top + pad
        ax.plot(x_positions, [y_line, y_line], color="black", linewidth=1.2, zorder=6)
        ax.plot([x_positions[0], x_positions[0]], [y_top, y_line], color="black",
                linewidth=1.2, zorder=6)
        ax.plot([x_positions[1], x_positions[1]], [y_top, y_line], color="black",
                linewidth=1.2, zorder=6)
        ax.text(
            (x_positions[0] + x_positions[1]) / 2, y_line + pad * 0.3,
            f"p={pval:.4f}  {sig}",
            ha="center", va="bottom", fontsize=self._gs()["tick_fontsize"],
            color="black", fontweight="bold", zorder=7,
        )

    # -- chart types --------------------------------------------------

    def _bar_individual(self, df: pd.DataFrame, metric: str, ylabel: str,
                        agg_fn: str) -> None:
        """Clustered bar chart — one group of bars per session, one bar per behavior."""
        ax = self._figure.add_subplot(111)
        gs = self._gs()
        behaviors = sorted(df["behavior"].unique())
        sessions = sorted(df["session_label"].unique())
        n_b = len(behaviors)
        x = np.arange(len(sessions))
        width = 0.75 / max(n_b, 1)
        for i, bname in enumerate(behaviors):
            sub_df = df[df["behavior"] == bname]
            vals = sub_df.groupby("session_label")[metric].agg(agg_fn).reindex(sessions, fill_value=0)
            offset = (i - n_b / 2 + 0.5) * width
            color = _PALETTE[i % len(_PALETTE)]
            bars = ax.bar(x + offset, vals.values, width=width * 0.92,
                          label=str(bname), color=color, alpha=0.85)
            if gs.get("show_indiv_points", True) and len(behaviors) == 1:
                for xi, v in zip(x, vals.values):
                    if v > 0:
                        ax.text(xi + offset, v * 1.01 + 0.002,
                                f"{v:.1f}", ha="center", va="bottom",
                                fontsize=max(6, gs["tick_fontsize"] - 1))
        ax.set_xticks(x)
        ax.set_xticklabels(sessions, rotation=45, ha="right",
                            fontsize=gs["tick_fontsize"])
        title = f"{ylabel} by Session"
        if n_b > 1:
            ax.legend(fontsize=gs["legend_fontsize"], loc=gs["legend_loc"])
        self._apply(ax, title=title, ylabel=ylabel)

    def _bar_groups(self, df: pd.DataFrame, metric: str, ylabel: str,
                    agg_fn: str) -> None:
        """Group comparison bar ± SEM with jittered individual session overlay."""
        ax = self._figure.add_subplot(111)
        gs = self._gs()
        groups = self._host._session_groups
        if not groups:
            ax.text(0.5, 0.5, "Assign groups in the Summary tab first.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=gs["axis_fontsize"], color="#90a4ae")
            return
        dfc = df.copy()
        dfc["group"] = dfc["session_label"].map(groups)
        dfc = dfc.dropna(subset=["group"])
        if dfc.empty:
            ax.text(0.5, 0.5, "No sessions with group assignments.",
                    ha="center", va="center", transform=ax.transAxes)
            return
        sess_agg = dfc.groupby(["session_label", "group"])[metric].agg(agg_fn).reset_index()
        group_names = self._host._ordered_group_list(sess_agg["group"].unique())
        error_style = gs.get("error_style", "SEM")
        bar_spacing  = float(gs.get("bar_spacing", 1.0))
        _capsize = int(gs.get("eb_capsize", 4)) if error_style != "None" else 0
        _eblw    = float(gs.get("eb_linewidth", 1.0))
        g_means = [sess_agg.loc[sess_agg["group"] == g, metric].mean() for g in group_names]
        g_ebs   = [
            _eb_val(sess_agg.loc[sess_agg["group"] == g, metric].to_numpy(), error_style)
            for g in group_names
        ]
        colors = [self._host._group_color(g, i) for i, g in enumerate(group_names)]
        x = np.arange(len(group_names))
        bar_w = min(0.82, 0.65 * bar_spacing)
        yerr_kw = {"elinewidth": _eblw, "capthick": _eblw} if error_style != "None" else {}
        ax.bar(x, g_means,
               yerr=(g_ebs if error_style != "None" else None),
               capsize=_capsize,
               width=bar_w, color=colors, alpha=0.82,
               error_kw=yerr_kw, zorder=3)
        # Individual data points
        if gs.get("show_indiv_points", True):
            rng = np.random.default_rng(1)
            for gi, gname in enumerate(group_names):
                vals = sess_agg.loc[sess_agg["group"] == gname, metric].to_numpy()
                jitter = rng.uniform(-0.14, 0.14, len(vals)) * bar_spacing
                ax.scatter(np.full(len(vals), gi) + jitter, vals,
                           color="white", edgecolors="black", linewidths=0.7,
                           s=40, zorder=5)
        # Value labels above bars
        for i, (m, eb) in enumerate(zip(g_means, g_ebs)):
            ax.text(i, m + eb + float(np.max(g_means)) * 0.02, f"{m:.1f}",
                    ha="center", va="bottom", fontsize=max(6, gs["tick_fontsize"] - 1))
        # Stats annotation
        if len(group_names) == 2:
            y_top = float(max(m + eb for m, eb in zip(g_means, g_ebs)))
            self._annotate_stats(ax, [0.0, 1.0], y_top)
        elif len(group_names) > 2 and gs.get("show_stats", True):
            ls = self._host._last_stats_result
            if ls and ls.get("metric") == self._get_metric():
                _pv = ls.get("pval")
                if _pv is not None and not np.isnan(float(_pv)) and _pv < 0.05:
                    _sig = "***" if _pv < 0.001 else "**" if _pv < 0.01 else "*"
                    ax.text(0.98, 0.98, f"{_sig}  p={_pv:.4f}",
                            ha="right", va="top", transform=ax.transAxes,
                            fontsize=gs["tick_fontsize"], color="black", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        # Add labelled patches so update_graph can place a shared side legend.
        from matplotlib.patches import Patch as _Patch
        _leg_handles = [_Patch(facecolor=c, label=g) for c, g in zip(colors, group_names)]
        ax.legend(handles=_leg_handles)  # collected and replaced by side legend
        ax.set_xticks(x)
        ax.set_xticklabels([])  # groups are identified by the side legend
        eb_lbl = error_style if error_style != "None" else ""
        ax.set_ylabel(
            f"{ylabel} (mean \u00b1 {eb_lbl})" if eb_lbl else ylabel,
            fontsize=gs["axis_fontsize"]
        )
        ax.set_title(f"Group Comparison: {ylabel}", fontsize=gs["title_fontsize"])

    def _stacked_bar(self, df: pd.DataFrame, col: str, ylabel: str, title: str) -> None:
        pivot = df.pivot_table(index="session_label", columns="behavior", values=col,
                               aggfunc="sum", fill_value=0)
        ax = self._figure.add_subplot(111)
        pivot.plot.bar(stacked=True, ax=ax)
        self._apply(ax, title=title, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=self._gs()["legend_fontsize"], loc=self._gs()["legend_loc"])

    def _line_timecourse(self, df: pd.DataFrame) -> None:
        """Time-course line graph.

        Individual mode: one trace per subject (all subjects on same axes).
        Group mode:      mean ± SEM across subjects within each group.
        Both modes collapse across the session timeline using time-binned bouts.
        """
        ax = self._figure.add_subplot(111)
        gs = self._gs()
        binned = self._bin_bouts()
        if binned.empty:
            ax.text(0.5, 0.5, "No time-binned data (need raw bout files)",
                    ha="center", va="center", transform=ax.transAxes)
            return
        metric = self._get_metric()
        metric_label = self._metric_label(metric)
        mode = self._get_mode()
        bin_seconds = max(10, int(self._time_bin_spin.value()))
        col = metric if metric != "time_spent_s" else "duration_s"

        if mode == "group":
            groups = self._host._session_groups
            if not groups:
                ax.text(0.5, 0.5, "Assign groups in the Summary tab first",
                        ha="center", va="center", transform=ax.transAxes)
                return
            binned["group"] = binned["session_label"].map(groups)
            binned = binned.dropna(subset=["group"])
            # Respect checked group filter
            checked_groups = self._checked_groups()
            if checked_groups:
                binned = binned[binned["group"].isin(checked_groups)]
            if binned.empty:
                ax.text(0.5, 0.5, "No grouped data", ha="center", va="center",
                        transform=ax.transAxes)
                return
            if metric == "mean_bout_s":
                sess_bin = binned.groupby(["group", "session_label", "time_bin_s"])[col].mean().reset_index()
            else:
                sess_bin = binned.groupby(["group", "session_label", "time_bin_s"])[col].sum().reset_index()
            group_list = self._host._ordered_group_list(sess_bin["group"].unique())
            error_style = gs.get("error_style", "SEM")
            for gi, g in enumerate(group_list):
                gdf = sess_bin[sess_bin["group"] == g]
                stats = gdf.groupby("time_bin_s")[col].agg(["mean", "sem"]).reset_index()
                stats["sem"] = stats["sem"].fillna(0)
                minutes = stats["time_bin_s"] / 60.0
                color = self._host._group_color(g, gi)
                ax.plot(minutes, stats["mean"], marker=".", label=str(g), color=color)
                if error_style != "None":
                    eb = stats["sem"].to_numpy() * (1.0 if error_style == "SEM" else
                                                     (stats["mean"].count()**0.5 if error_style == "SD" else 1.96))
                    if error_style == "SD":
                        # recompute proper SD from SEM
                        n_grp = gdf.groupby("time_bin_s")[col].count().reindex(stats["time_bin_s"]).fillna(1).to_numpy()
                        eb = stats["sem"].to_numpy() * np.sqrt(np.maximum(n_grp, 1))
                    elif error_style == "95% CI":
                        eb = stats["sem"].to_numpy() * 1.96
                    else:
                        eb = stats["sem"].to_numpy()
                    ax.fill_between(minutes, stats["mean"] - eb,
                                    stats["mean"] + eb, alpha=0.2, color=color)
            eb_lbl = f" \u00b1 {error_style}" if error_style != "None" else ""
            self._apply(ax, title=f"Time Course: {metric_label} by Group",
                        ylabel=f"{metric_label}{eb_lbl}",
                        xlabel=f"Time (min, bin={bin_seconds}s)")
            ax.legend(fontsize=gs["legend_fontsize"], loc=gs["legend_loc"])
        else:
            # Collapse across all sessions → mean ± SEM
            if metric == "mean_bout_s":
                sess_bin = binned.groupby(["session_label", "behavior", "time_bin_s"])[col].mean().reset_index()
            else:
                sess_bin = binned.groupby(["session_label", "behavior", "time_bin_s"])[col].sum().reset_index()
            error_style = gs.get("error_style", "SEM")
            for bi, bname in enumerate(sorted(sess_bin["behavior"].unique())):
                bdf = sess_bin[sess_bin["behavior"] == bname]
                stats = bdf.groupby("time_bin_s")[col].agg(["mean", "sem"]).reset_index()
                stats["sem"] = stats["sem"].fillna(0)
                minutes = stats["time_bin_s"] / 60.0
                color = _PALETTE[bi % len(_PALETTE)]
                ax.plot(minutes, stats["mean"], marker=".", label=str(bname), color=color)
                if error_style != "None":
                    eb = stats["sem"].to_numpy() * (1.96 if error_style == "95% CI" else 1.0)
                    ax.fill_between(minutes, stats["mean"] - eb,
                                    stats["mean"] + eb, alpha=0.2, color=color)
            eb_lbl = f" \u00b1 {error_style}" if error_style != "None" else ""
            self._apply(ax, title=f"Time Course: {metric_label}{eb_lbl} across sessions",
                        ylabel=f"{metric_label}{eb_lbl}",
                        xlabel=f"Time (min, bin={bin_seconds}s)")
            if sess_bin["behavior"].nunique() > 1:
                ax.legend(fontsize=gs["legend_fontsize"], loc=gs["legend_loc"])

    def _box_plot(self, df: pd.DataFrame) -> None:
        ax = self._figure.add_subplot(111)
        gs = self._gs()
        groups = self._host._session_groups
        if not groups:
            ax.text(0.5, 0.5, "Assign groups in the Summary tab first",
                    ha="center", va="center", transform=ax.transAxes)
            return
        metric = self._get_metric()
        metric_label = self._metric_label(metric)
        agg_fn = "mean" if metric == "mean_bout_s" else "sum"
        dfc = df.copy()
        dfc["group"] = dfc["session_label"].map(groups)
        dfc = dfc.dropna(subset=["group"])
        if dfc.empty:
            ax.text(0.5, 0.5, "No grouped data", ha="center", va="center",
                    transform=ax.transAxes)
            return
        sess_agg = dfc.groupby(["session_label", "group"])[metric].agg(agg_fn).reset_index()
        group_names = self._host._ordered_group_list(sess_agg["group"].unique())
        data = [sess_agg.loc[sess_agg["group"] == g, metric].tolist() for g in group_names]
        bp = ax.boxplot(data, labels=group_names, patch_artist=True, widths=0.5,
                        medianprops={"color": "white", "linewidth": 1.5})
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(self._host._group_color(group_names[i], i))
            patch.set_alpha(0.7)
        _max_lbl_bp = max((len(str(g)) for g in group_names), default=0)
        _rot_bp = 45 if (len(group_names) > 4 or _max_lbl_bp > 6) else 0
        if _rot_bp:
            ax.set_xticklabels(group_names, rotation=_rot_bp, ha="right",
                               fontsize=gs["tick_fontsize"])
        rng = np.random.default_rng(42)
        for i, vals in enumerate(data):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                       color="white", s=25, zorder=3, edgecolors="black", linewidths=0.5)
        # Stats annotation
        if len(group_names) == 2:
            y_top = float(max(max(d) for d in data if d))
            self._annotate_stats(ax, [1.0, 2.0], y_top)
        self._apply(ax, title=f"Box Plot: {metric_label}", ylabel=metric_label)

    def _ethogram(self, df: pd.DataFrame) -> None:
        """Raster/ethogram showing behaviour bouts across time for each subject.

        Expensive DataFrame processing is cached keyed on raw bout data + settings.
        Changing only the session filter re-renders from the cache without
        re-scanning the parquets, so toggling sessions is near-instant.
        """
        ax = self._figure.add_subplot(111)
        raw = self._host._raw_bouts
        if not raw:
            ax.text(0.5, 0.5, "No raw bout data available (need bout parquet files)",
                    ha="center", va="center", transform=ax.transAxes)
            return
        fps = self._host._project_fps()
        checked = self._host._summary_tab._checked_subjects()

        # -- Build color/name lookups (fast, no iteration) ------------------------
        bid_to_name: dict[str, str] = {}
        bid_to_color: dict[str, str] = {}
        _DEFAULT_COLOR = "#4A90E2"
        all_default = all(
            (b.color or _DEFAULT_COLOR) == _DEFAULT_COLOR
            for b in self._host._behaviors.behaviors
        )
        for idx, b in enumerate(self._host._behaviors.behaviors):
            bid = str(b.behavior_id)
            bid_to_name[bid] = str(b.name or bid)
            if all_default or not b.color or b.color == _DEFAULT_COLOR:
                bid_to_color[bid] = _PALETTE[idx % len(_PALETTE)]
            else:
                bid_to_color[bid] = str(b.color)

        selected_bids = self._host._selected_behavior_ids()
        first_n = self._get_first_n_bouts()
        until_cutoffs = self._cutoff_frames_for_until_behavior()

        # -- Cache key: everything EXCEPT the session filter ----------------------
        _total_rows = sum(len(v) for v in raw.values())
        _cache_key: tuple[Any, ...] = (
            id(raw), _total_rows, fps,
            frozenset(selected_bids), first_n,
            tuple(sorted(until_cutoffs.items())) if until_cutoffs else (),
        )

        if self._ethogram_cache is None or self._ethogram_cache_key != _cache_key:
            # Build cache: {bid: {sess_label: [(start_s, dur_s), ...]}}
            # Uses vectorized pandas ops — no Python-level iterrows.
            _label_map = self._host._session_label_by_session
            new_cache: dict[str, dict[str, list[tuple[float, float]]]] = {}
            for bid, bdf in raw.items():
                if bid not in selected_bids:
                    continue
                if bdf.empty or "start_frame" not in bdf.columns:
                    new_cache[bid] = {}
                    continue
                # Apply bout filter
                bdf_f = bdf
                if first_n > 0 and "session_id" in bdf.columns:
                    bdf_f = bdf.sort_values("start_frame").groupby("session_id").head(first_n)
                elif until_cutoffs and "session_id" in bdf.columns:
                    parts = []
                    for sid_uc, grp_uc in bdf.groupby("session_id"):
                        cutoff = until_cutoffs.get(str(sid_uc))
                        if cutoff is not None:
                            parts.append(grp_uc[grp_uc["start_frame"] < cutoff])
                        else:
                            parts.append(grp_uc)
                    bdf_f = pd.concat(parts, ignore_index=True) if parts else bdf.iloc[:0]
                # Vectorized conversion to seconds
                starts = bdf_f["start_frame"].astype(float) / fps
                durs = (bdf_f["end_frame"].astype(float) / fps - starts).clip(lower=0)
                labels = bdf_f["session_id"].astype(str).map(
                    lambda s: _label_map.get(s, s)  # noqa: B023
                )
                # Group by session label
                sess_bouts: dict[str, list[tuple[float, float]]] = {}
                tmp = pd.DataFrame({"sess": labels, "s": starts, "d": durs})
                for sess_lbl, grp in tmp.groupby("sess"):
                    sess_bouts[str(sess_lbl)] = list(zip(grp["s"], grp["d"]))
                new_cache[bid] = sess_bouts
            self._ethogram_cache = new_cache
            self._ethogram_cache_key = _cache_key

        # -- Render from cache (fast path, even on session filter changes) --------
        sessions = sorted(self._checked_ethogram_sessions() or checked)
        session_idx = {s: i for i, s in enumerate(sessions)}

        for bid, sess_bouts in self._ethogram_cache.items():
            color = bid_to_color.get(bid, _PALETTE[0])
            label = bid_to_name.get(bid, bid)
            label_used = False
            for sess_label, bouts in sess_bouts.items():
                if sess_label not in session_idx or not bouts:
                    continue
                y = session_idx[sess_label]
                # broken_barh draws all segments as one PatchCollection — much
                # faster than individual barh calls for large bout counts.
                ax.broken_barh(
                    bouts, (y - 0.3, 0.6),
                    facecolors=color, edgecolors="none", alpha=0.8,
                    label=label if not label_used else "",
                )
                label_used = True

        ax.set_yticks(range(len(sessions)))
        ax.set_yticklabels(sessions)
        ax.invert_yaxis()
        self._apply(ax, title="Ethogram / Raster", xlabel="Time (s)", ylabel="Session")
        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels_, fontsize=self._gs()["legend_fontsize"],
                      loc=self._gs()["legend_loc"])

    def _pie_chart(self, df: pd.DataFrame) -> None:
        ax = self._figure.add_subplot(111)
        agg = df.groupby("behavior")["time_spent_s"].sum()
        if agg.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        colors = _PALETTE[:len(agg)]
        wedges, texts, autotexts = ax.pie(
            agg.values, labels=agg.index, autopct="%1.1f%%",
            colors=colors, startangle=90, pctdistance=0.75,
        )
        for t in autotexts:
            t.set_fontsize(self._gs()["tick_fontsize"])
        ax.set_title("Behavior Proportion (by total duration)",
                     fontsize=self._gs()["title_fontsize"])

    # -- time-binned --------------------------------------------------

    def _bin_bouts(self) -> pd.DataFrame:
        fps = self._host._project_fps()
        bin_seconds = max(10, int(self._time_bin_spin.value()))
        selected_bids = frozenset(self._host._selected_behavior_ids())
        checked = frozenset(self._host._summary_tab._checked_subjects())
        metric = self._get_metric()
        # Distance is needed when the metric toggle is set to distance OR
        # the behavior filter selects the distance pseudo-behavior.
        need_distance = metric == "distance_cm" or DISTANCE_BEHAVIOR_ID in selected_bids
        data_range = self._get_data_range_seconds()
        first_n = self._get_first_n_bouts()
        until_bid = self._get_until_behavior_id()

        # Check cache validity
        cache_key = (
            id(self._host._raw_bouts), len(self._host._raw_bouts),
            fps, bin_seconds, selected_bids, checked, need_distance, data_range,
            first_n, until_bid,
        )
        if self._bin_cache is not None and self._bin_cache_key == cache_key:
            return self._bin_cache

        rows: list[dict[str, Any]] = []

        # -- Session-level distance bins (the "Distance Traveled" pseudo-behavior) --
        # Generate distance rows whenever distance is the active metric or the
        # distance behavior is explicitly selected in the behavior filter.
        want_distance_rows = need_distance
        if want_distance_rows:
            _seen_sessions: set[str] = set()
            for r in self._host._summary_rows:
                sid = r["session_id"]
                if sid in _seen_sessions:
                    continue
                slbl = r["session_label"]
                if slbl not in checked:
                    continue
                _seen_sessions.add(sid)
                ppm = self._host._pixels_per_mm_for_session(sid)
                for t_bin, d_px in self._host._compute_session_distance_binned(sid, bin_seconds):
                    # Apply data range filter to distance bins
                    bin_end = t_bin + bin_seconds
                    if data_range[0] is not None and bin_end <= data_range[0]:
                        continue
                    if data_range[1] is not None and t_bin >= data_range[1]:
                        continue
                    d_cm = (d_px / ppm / 10.0) if ppm and ppm > 0 else d_px
                    rows.append({
                        "session_label": slbl,
                        "behavior_id": DISTANCE_BEHAVIOR_ID,
                        "behavior": DISTANCE_BEHAVIOR_NAME,
                        "time_bin_s": t_bin,
                        "n_bouts": 0,
                        "duration_s": 0.0,
                        "distance_cm": d_cm,
                    })

        # -- Regular (behavior-based) bout bins --
        # When the distance metric is selected, the only meaningful rows come
        # from the session-level distance bins above; skip bout bins to avoid
        # cluttering the output with zeros from regular behavior bouts.
        skip_bout_bins = metric == "distance_cm" or (want_distance_rows and len(selected_bids) == 1)
        until_cutoffs = self._cutoff_frames_for_until_behavior() if until_bid else {}
        if not skip_bout_bins:
            for bid, bdf in self._host._raw_bouts.items():
                if bid not in selected_bids:
                    continue
                if bid == DISTANCE_BEHAVIOR_ID:
                    continue  # handled above
                bname = bdf["behavior"].iloc[0] if len(bdf) > 0 else bid
                for sid, grp in bdf.groupby("session_id"):
                    session_label = self._host._session_label_by_session.get(str(sid), str(sid))
                    if session_label not in checked:
                        continue
                    # Apply bout filter
                    if first_n > 0:
                        grp = grp.sort_values("start_frame").head(first_n)
                    elif until_cutoffs:
                        cutoff = until_cutoffs.get(str(sid))
                        if cutoff is not None:
                            grp = grp[grp["start_frame"] < cutoff]
                    starts = grp["start_frame"].to_numpy(dtype=np.int64)
                    ends = grp["end_frame"].to_numpy(dtype=np.int64)
                    start_s = starts.astype(np.float64) / fps
                    end_s = (ends + 1).astype(np.float64) / fps

                    # Clip bouts to the data range
                    if data_range[0] is not None:
                        lo_s = data_range[0]
                        mask = end_s > lo_s
                        start_s = start_s[mask]
                        end_s = end_s[mask]
                        start_s = np.maximum(start_s, lo_s)
                    if data_range[1] is not None:
                        hi_s = data_range[1]
                        mask = start_s < hi_s
                        start_s = start_s[mask]
                        end_s = end_s[mask]
                        end_s = np.minimum(end_s, hi_s)

                    for k in range(len(start_s)):
                        b_start = float(start_s[k])
                        b_end = float(end_s[k])
                        if b_end <= b_start:
                            continue
                        cursor = b_start
                        is_first_slice = True
                        while cursor < b_end:
                            cur_bin = int(cursor // bin_seconds) * bin_seconds
                            bin_edge = cur_bin + bin_seconds
                            slice_end = min(b_end, float(bin_edge))
                            slice_dur = slice_end - cursor

                            rows.append({
                                "session_label": session_label, "behavior_id": bid,
                                "behavior": bname, "time_bin_s": cur_bin,
                                "n_bouts": 1 if is_first_slice else 0,
                                "duration_s": slice_dur,
                                "distance_cm": 0.0,
                            })
                            cursor = slice_end
                            is_first_slice = False

        if not rows:
            self._bin_cache = pd.DataFrame()
            self._bin_cache_key = cache_key
            return self._bin_cache
        df = pd.DataFrame(rows)
        agg = df.groupby(["session_label", "behavior_id", "behavior", "time_bin_s"]).agg(
            n_bouts=("n_bouts", "sum"), duration_s=("duration_s", "sum"),
            distance_cm=("distance_cm", "sum"),
        ).reset_index()
        agg["mean_bout_s"] = agg["duration_s"] / agg["n_bouts"].clip(lower=1)
        self._bin_cache = agg
        self._bin_cache_key = cache_key
        return agg

    def _time_binned_chart(self, df: pd.DataFrame) -> None:
        ax = self._figure.add_subplot(111)
        binned = self._bin_bouts()
        if binned.empty:
            ax.text(0.5, 0.5, "No time-binned data (need raw bout files)",
                    ha="center", va="center", transform=ax.transAxes)
            return
        sel_bids = self._host._selected_behavior_ids()
        metric = "distance_cm" if sel_bids == {DISTANCE_BEHAVIOR_ID} else self._get_metric()
        metric_label = self._metric_label(metric)
        bin_seconds = max(10, int(self._time_bin_spin.value()))
        col = metric if metric != "time_spent_s" else "duration_s"
        if metric == "mean_bout_s":
            agg = binned.groupby(["behavior", "time_bin_s"])[col].mean().reset_index()
        else:
            agg = binned.groupby(["behavior", "time_bin_s"])[col].sum().reset_index()
        for bname, grp in agg.groupby("behavior"):
            ordered = grp.sort_values("time_bin_s")
            minutes = ordered["time_bin_s"] / 60.0
            ax.plot(minutes, ordered[col], marker=".", label=str(bname))
            ax.fill_between(minutes, 0, ordered[col], alpha=0.15)
        self._apply(ax, title=f"Time-Binned {metric_label} (Checked Subjects)",
                    ylabel=metric_label, xlabel=f"Time (min, bin={bin_seconds}s)")
        if binned["behavior"].nunique() > 1:
            ax.legend(fontsize=self._gs()["legend_fontsize"], loc=self._gs()["legend_loc"])

    def _time_binned_group_chart(self, df: pd.DataFrame) -> None:
        ax = self._figure.add_subplot(111)
        groups = self._host._session_groups
        if not groups:
            ax.text(0.5, 0.5, "Assign groups in the Summary tab first",
                    ha="center", va="center", transform=ax.transAxes)
            return
        binned = self._bin_bouts()
        if binned.empty:
            ax.text(0.5, 0.5, "No time-binned data", ha="center", va="center",
                    transform=ax.transAxes)
            return
        sel_bids = self._host._selected_behavior_ids()
        metric = "distance_cm" if sel_bids == {DISTANCE_BEHAVIOR_ID} else self._get_metric()
        metric_label = self._metric_label(metric)
        bin_seconds = max(10, int(self._time_bin_spin.value()))
        col = metric if metric != "time_spent_s" else "duration_s"
        if metric == "mean_bout_s":
            col = metric
        binned["group"] = binned["session_label"].map(groups)
        binned = binned.dropna(subset=["group"])
        # Respect checked group filter
        checked_groups = self._checked_groups()
        if checked_groups:
            binned = binned[binned["group"].isin(checked_groups)]
        if binned.empty:
            ax.text(0.5, 0.5, "No grouped data", ha="center", va="center", transform=ax.transAxes)
            return
        if metric == "mean_bout_s":
            sess_bin = binned.groupby(["group", "session_label", "time_bin_s"])[col].mean().reset_index()
        else:
            sess_bin = binned.groupby(["group", "session_label", "time_bin_s"])[col].sum().reset_index()
        group_list = self._host._ordered_group_list(sess_bin["group"].unique())
        error_style = self._gs().get("error_style", "SEM")
        for gi, g in enumerate(group_list):
            gdf = sess_bin[sess_bin["group"] == g]
            stats = gdf.groupby("time_bin_s")[col].agg(["mean", "sem"]).reset_index()
            stats["sem"] = stats["sem"].fillna(0)
            minutes = stats["time_bin_s"] / 60.0
            color = self._host._group_color(g, gi)
            ax.plot(minutes, stats["mean"], marker=".", label=str(g), color=color)
            if error_style != "None":
                eb = stats["sem"].to_numpy() * (1.96 if error_style == "95% CI" else 1.0)
                ax.fill_between(minutes, stats["mean"] - eb,
                                stats["mean"] + eb, alpha=0.2, color=color)
        eb_lbl = f" \u00b1 {error_style}" if error_style != "None" else ""
        self._apply(ax, title=f"Time-Binned {metric_label} by Group",
                    ylabel=f"{metric_label}{eb_lbl}",
                    xlabel=f"Time (min, bin={bin_seconds}s)")
        ax.legend(fontsize=self._gs()["legend_fontsize"], loc=self._gs()["legend_loc"])

    def _combined_overview(self, df: pd.DataFrame) -> None:
        ax1 = self._figure.add_subplot(111)
        gs = self._gs()
        count_df = df.groupby("session_label", sort=True)["n_bouts"].sum().reset_index()
        dur_df = df.groupby("session_label", sort=True)["mean_bout_s"].mean().reset_index()
        sessions = count_df["session_label"].tolist()
        x = np.arange(len(sessions))
        width = 0.35
        ax1.bar(x - width / 2, count_df["n_bouts"], width, color="#5b9bd5",
                label="Bout Count", alpha=0.85)
        ax1.set_ylabel("Bout Count", fontsize=gs["axis_fontsize"], color="#5b9bd5")
        ax1.tick_params(axis="y", labelcolor="#5b9bd5", labelsize=gs["tick_fontsize"])
        ax1.set_xticks(x)
        ax1.set_xticklabels(sessions, rotation=45, ha="right", fontsize=gs["tick_fontsize"])
        ax2 = ax1.twinx()
        ax2.bar(x + width / 2, dur_df["mean_bout_s"], width, color="#ed7d31",
                label="Mean Bout Duration (s)", alpha=0.85)
        ax2.set_ylabel("Mean Bout Duration (s)", fontsize=gs["axis_fontsize"], color="#ed7d31")
        ax2.tick_params(axis="y", labelcolor="#ed7d31", labelsize=gs["tick_fontsize"])
        ax1.set_title("Combined Overview: Bout Count vs Mean Duration",
                      fontsize=gs["title_fontsize"])
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=gs["legend_fontsize"], loc=gs["legend_loc"])

    # -- per-behavior subplot helpers ---------------------------------

    def _bar_groups_on_ax(self, ax: Any, df: pd.DataFrame, metric: str,
                          ylabel: str, agg_fn: str, title: str = "") -> None:
        """Bar chart for a single behavior on a given axes (for multi-behavior subplots)."""
        gs = self._gs()
        groups = self._host._session_groups
        if not groups:
            ax.text(0.5, 0.5, "No groups", ha="center", va="center", transform=ax.transAxes)
            return
        dfc = df.copy()
        dfc["group"] = dfc["session_label"].map(groups)
        dfc = dfc.dropna(subset=["group"])
        if dfc.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        sess_agg = dfc.groupby(["session_label", "group"])[metric].agg(agg_fn).reset_index()
        group_names = self._host._ordered_group_list(sess_agg["group"].unique())
        error_style = gs.get("error_style", "SEM")
        bar_spacing  = float(gs.get("bar_spacing", 1.0))
        _capsize = int(gs.get("eb_capsize", 4)) if error_style != "None" else 0
        _eblw    = float(gs.get("eb_linewidth", 1.0))
        g_means = [sess_agg.loc[sess_agg["group"] == g, metric].mean() for g in group_names]
        g_ebs   = [
            _eb_val(sess_agg.loc[sess_agg["group"] == g, metric].to_numpy(), error_style)
            for g in group_names
        ]
        colors = [self._host._group_color(g, i) for i, g in enumerate(group_names)]
        x = np.arange(len(group_names))
        bar_w = min(0.80, 0.60 * bar_spacing)
        ax.bar(x, g_means,
               yerr=(g_ebs if error_style != "None" else None),
               capsize=_capsize,
               width=bar_w, color=colors, alpha=0.82,
               error_kw={"elinewidth": _eblw, "capthick": _eblw} if error_style != "None" else {},
               zorder=3)
        if gs.get("show_indiv_points", True):
            rng = np.random.default_rng(1)
            for gi, gname in enumerate(group_names):
                vals = sess_agg.loc[sess_agg["group"] == gname, metric].to_numpy()
                jitter = rng.uniform(-0.12, 0.12, len(vals)) * bar_spacing
                ax.scatter(np.full(len(vals), gi) + jitter, vals,
                           color="white", edgecolors="black", linewidths=0.5, s=25, zorder=5)
        # Inline significance annotation (independent of the stats panel)
        _show_stats = self._gs().get("show_stats", True)
        if _show_stats and len(group_names) == 2:
            try:
                from scipy.stats import ttest_ind as _ttest_on_ax  # type: ignore[import-untyped]
                _a0 = sess_agg.loc[sess_agg["group"] == group_names[0], metric].to_numpy()
                _a1 = sess_agg.loc[sess_agg["group"] == group_names[1], metric].to_numpy()
                if len(_a0) >= 2 and len(_a1) >= 2:
                    _, _pv = _ttest_on_ax(_a0, _a1, equal_var=False)
                    _sig = "***" if _pv < 0.001 else "**" if _pv < 0.01 else "*" if _pv < 0.05 else ""
                    if _sig:
                        _ytop = float(max(m + e for m, e in zip(g_means, g_ebs)))
                        _pad = max(_ytop * 0.08, 0.10)
                        _yline = _ytop + _pad
                        ax.plot([0, 1], [_yline, _yline], color="black", linewidth=1.0, zorder=6)
                        ax.plot([0, 0], [_ytop, _yline], color="black", linewidth=1.0, zorder=6)
                        ax.plot([1, 1], [_ytop, _yline], color="black", linewidth=1.0, zorder=6)
                        ax.text(0.5, _yline + _pad * 0.3, f"p={_pv:.3f}  {_sig}",
                                ha="center", va="bottom",
                                fontsize=max(6, gs["tick_fontsize"] - 1),
                                color="black", fontweight="bold", zorder=7)
            except Exception:
                pass
        elif _show_stats and len(group_names) > 2:
            try:
                from scipy.stats import f_oneway as _fow_on_ax  # type: ignore[import-untyped]
                _arrs = [sess_agg.loc[sess_agg["group"] == g, metric].to_numpy()
                         for g in group_names]
                _arrs = [a for a in _arrs if len(a) >= 2]
                if len(_arrs) >= 2:
                    _, _pv = _fow_on_ax(*_arrs)
                    _sig = "***" if _pv < 0.001 else "**" if _pv < 0.01 else "*" if _pv < 0.05 else ""
                    if _sig:
                        ax.text(0.98, 0.98, f"ANOVA {_sig}  p={_pv:.4f}",
                                ha="right", va="top", transform=ax.transAxes,
                                fontsize=max(6, gs["tick_fontsize"] - 1),
                                color="black", fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                          ec="black", alpha=0.8))
            except Exception:
                pass
        # Add labelled patches so the parent update_graph can collect handles
        # and place a shared side legend (hiding these x-tick labels).
        from matplotlib.patches import Patch as _Patch
        _leg_handles = [_Patch(facecolor=c, label=g) for c, g in zip(colors, group_names)]
        ax.legend(handles=_leg_handles)  # collected and replaced by side legend
        ax.set_xticks(x)
        ax.set_xticklabels([])  # groups are identified by the side legend
        self._apply(ax, title=title, ylabel=ylabel)

    def _line_timecourse_on_ax(self, ax: Any, df: pd.DataFrame,
                                metric: str, ylabel: str) -> None:
        """Line time-course for a single behavior on a given axes."""
        gs = self._gs()
        groups = self._host._session_groups
        if not groups:
            ax.text(0.5, 0.5, "No groups", ha="center", va="center", transform=ax.transAxes)
            return
        binned = self._bin_bouts()
        if binned.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        beh_name = df["behavior"].iloc[0] if len(df) > 0 else ""
        binned = binned[binned["behavior"] == beh_name]
        if binned.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        bin_seconds = max(10, int(self._time_bin_spin.value()))
        col = metric if metric != "time_spent_s" else "duration_s"
        binned["group"] = binned["session_label"].map(groups)
        binned = binned.dropna(subset=["group"])
        # Respect checked group filter
        checked_groups = self._checked_groups()
        if checked_groups:
            binned = binned[binned["group"].isin(checked_groups)]
        if binned.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        if metric == "mean_bout_s":
            sess_bin = binned.groupby(["group", "session_label", "time_bin_s"])[col].mean().reset_index()
        else:
            sess_bin = binned.groupby(["group", "session_label", "time_bin_s"])[col].sum().reset_index()
        group_list = self._host._ordered_group_list(sess_bin["group"].unique())
        for gi, g in enumerate(group_list):
            gdf = sess_bin[sess_bin["group"] == g]
            stats = gdf.groupby("time_bin_s")[col].agg(["mean", "sem"]).reset_index()
            stats["sem"] = stats["sem"].fillna(0)
            minutes = stats["time_bin_s"] / 60.0
            color = self._host._group_color(g, gi)
            ax.plot(minutes, stats["mean"], marker=".", label=str(g), color=color, markersize=4)
            error_style = gs.get("error_style", "SEM")
            if error_style != "None":
                eb = stats["sem"].to_numpy() * (1.96 if error_style == "95% CI" else 1.0)
                ax.fill_between(minutes, stats["mean"] - eb,
                                stats["mean"] + eb, alpha=0.2, color=color)
        eb_lbl = f" \u00b1 {gs.get('error_style', 'SEM')}" if gs.get("error_style", "SEM") != "None" else ""
        self._apply(ax, title=str(beh_name), ylabel=f"{ylabel}{eb_lbl}",
                    xlabel=f"Time (min, bin={bin_seconds}s)")
        ax.legend(fontsize="x-small", loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                  frameon=True, framealpha=0.9)

    def _box_plot_on_ax(self, ax: Any, df: pd.DataFrame, metric: str,
                        ylabel: str, agg_fn: str) -> None:
        """Box plot for a single behavior on a given axes."""
        gs = self._gs()
        groups = self._host._session_groups
        if not groups:
            ax.text(0.5, 0.5, "No groups", ha="center", va="center", transform=ax.transAxes)
            return
        dfc = df.copy()
        dfc["group"] = dfc["session_label"].map(groups)
        dfc = dfc.dropna(subset=["group"])
        if dfc.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        sess_agg = dfc.groupby(["session_label", "group"])[metric].agg(agg_fn).reset_index()
        group_names = self._host._ordered_group_list(sess_agg["group"].unique())
        data = [sess_agg.loc[sess_agg["group"] == g, metric].tolist() for g in group_names]
        bp = ax.boxplot(data, labels=group_names, patch_artist=True, widths=0.5,
                        medianprops={"color": "white", "linewidth": 1.2})
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(self._host._group_color(group_names[i], i))
            patch.set_alpha(0.7)
        _max_lbl_bp = max((len(str(g)) for g in group_names), default=0)
        _rot_bp = 45 if (len(group_names) > 4 or _max_lbl_bp > 6) else 0
        if _rot_bp:
            ax.set_xticklabels(group_names, rotation=_rot_bp, ha="right",
                               fontsize=max(7, gs["tick_fontsize"] - 1))
        beh_name = df["behavior"].iloc[0] if len(df) > 0 else ""
        self._apply(ax, title=str(beh_name), ylabel=ylabel)

    def _overview_on_ax(self, ax: Any, df: pd.DataFrame, title: str) -> None:
        """Overview chart for a single behavior on a given axes."""
        gs = self._gs()
        count_df = df.groupby("session_label", sort=True)["n_bouts"].sum().reset_index()
        dur_df = df.groupby("session_label", sort=True)["mean_bout_s"].mean().reset_index()
        sessions = count_df["session_label"].tolist()
        x = np.arange(len(sessions))
        width = 0.35
        ax.bar(x - width / 2, count_df["n_bouts"], width, color="#5b9bd5",
               label="Bouts", alpha=0.85)
        ax.set_ylabel("Bout Count", fontsize=max(7, gs["axis_fontsize"] - 2), color="#5b9bd5")
        ax.set_xticks(x)
        ax.set_xticklabels(sessions, rotation=45, ha="right", fontsize=max(6, gs["tick_fontsize"] - 1))
        ax2 = ax.twinx()
        ax2.bar(x + width / 2, dur_df["mean_bout_s"], width, color="#ed7d31",
                label="Duration", alpha=0.85)
        ax2.set_ylabel("Mean Duration (s)", fontsize=max(7, gs["axis_fontsize"] - 2), color="#ed7d31")
        ax.set_title(title, fontsize=max(8, gs["title_fontsize"] - 2))

    # -- settings dialog ----------------------------------------------

    def _open_settings_dialog(self) -> None:
        # Bind to the real settings dict (not self._gs(), which returns a
        # throwaway font-scaled copy when a figure is on screen — writes to it
        # would be discarded and never persist).
        gs = self._host._graph_settings
        dlg = QDialog(self)
        dlg.setWindowTitle("Graph Settings")
        dlg.resize(380, 380)
        form = QFormLayout()

        title_fs = QSpinBox(dlg)
        title_fs.setRange(6, 32)
        title_fs.setValue(int(gs["title_fontsize"]))

        axis_fs = QSpinBox(dlg)
        axis_fs.setRange(6, 28)
        axis_fs.setValue(int(gs["axis_fontsize"]))

        tick_fs = QSpinBox(dlg)
        tick_fs.setRange(4, 24)
        tick_fs.setValue(int(gs["tick_fontsize"]))

        dpi_spin = QSpinBox(dlg)
        dpi_spin.setRange(72, 600)
        dpi_spin.setSingleStep(50)
        dpi_spin.setValue(int(gs["dpi"]))

        legend_combo = QComboBox(dlg)
        for loc in ["best", "upper right", "upper left", "lower right", "lower left",
                     "center left", "center right", "upper center", "lower center"]:
            legend_combo.addItem(loc)
        legend_combo.setCurrentText(str(gs["legend_loc"]))

        error_style_combo = QComboBox(dlg)
        for es in ("SEM", "SD", "95% CI", "None"):
            error_style_combo.addItem(es, userData=es)
        error_style_combo.setCurrentText(gs.get("error_style", "SEM"))
        error_style_combo.setToolTip(
            "Error bar / shaded-band style shown on bar and line charts.\n"
            "SEM = standard error of the mean (default)\n"
            "SD = standard deviation\n"
            "95% CI = 1.96 \u00d7 SEM\n"
            "None = no error bars"
        )

        bar_spacing_spin = QDoubleSpinBox(dlg)
        bar_spacing_spin.setRange(0.3, 2.0)
        bar_spacing_spin.setSingleStep(0.1)
        bar_spacing_spin.setDecimals(1)
        bar_spacing_spin.setValue(float(gs.get("bar_spacing", 1.0)))
        bar_spacing_spin.setToolTip(
            "Multiplier for bar widths. 1.0 = default spacing.\n"
            "< 1.0 = narrower bars with more gap between groups.\n"
            "> 1.0 = wider bars, closer together."
        )

        eb_capsize_spin = QSpinBox(dlg)
        eb_capsize_spin.setRange(0, 20)
        eb_capsize_spin.setValue(int(gs.get("eb_capsize", 4)))
        eb_capsize_spin.setSuffix(" pt")
        eb_capsize_spin.setToolTip("Width of the horizontal caps at the top and bottom of each error bar.")

        eb_lw_spin = QDoubleSpinBox(dlg)
        eb_lw_spin.setRange(0.2, 6.0)
        eb_lw_spin.setSingleStep(0.2)
        eb_lw_spin.setDecimals(1)
        eb_lw_spin.setValue(float(gs.get("eb_linewidth", 1.0)))
        eb_lw_spin.setSuffix(" pt")
        eb_lw_spin.setToolTip("Thickness of the vertical error bar lines.")

        indiv_points_check = QCheckBox("Show individual data points", dlg)
        indiv_points_check.setChecked(gs.get("show_indiv_points", False))
        indiv_points_check.setToolTip(
            "Overlay individual subject data points on top of bar/line charts."
        )

        show_stats_check = QCheckBox("Show statistics on graph", dlg)
        show_stats_check.setChecked(gs.get("show_stats", True))
        show_stats_check.setToolTip(
            "Overlay significance brackets / stars (p-values) on comparison charts.\n"
            "Uncheck to hide them; the stats panel/popup is unaffected."
        )

        force_fit_check = QCheckBox("Force fit to canvas size", dlg)
        force_fit_check.setChecked(gs.get("force_fit", False))
        force_fit_check.setToolTip(
            "After rendering, auto-resize the canvas to match the content's\n"
            "tight bounding box, up to the width/height limits below.\n"
            "Eliminates excess whitespace around the plot."
        )

        max_w_spin = QSpinBox(dlg)
        max_w_spin.setRange(200, 3840)
        max_w_spin.setSingleStep(50)
        max_w_spin.setSuffix(" px")
        max_w_spin.setToolTip("Maximum display width of the graph panel in pixels.")
        max_w_spin.setValue(int(gs.get("max_w", 800)))

        max_h_spin = QSpinBox(dlg)
        max_h_spin.setRange(150, 2160)
        max_h_spin.setSingleStep(50)
        max_h_spin.setSuffix(" px")
        max_h_spin.setToolTip("Maximum display height of the graph panel in pixels.")
        max_h_spin.setValue(int(gs.get("max_h", 500)))

        # Axis range overrides (moved here from the control panel to declutter).
        # The persistent _x_min/_x_max/_y_min/_y_max spinboxes remain the source
        # of truth; these dialog copies seed from and write back to them.
        _axis_holders = {
            "x_min": self._x_min, "x_max": self._x_max,
            "y_min": self._y_min, "y_max": self._y_max,
        }
        _axis_dlg_spins: dict[str, QDoubleSpinBox] = {}
        for _akey, _aholder in _axis_holders.items():
            _asp = QDoubleSpinBox(dlg)
            _asp.setRange(-1e6, 1e6)
            _asp.setDecimals(1)
            _asp.setSpecialValueText("auto")
            _asp.setValue(_aholder.value())
            _asp.setToolTip("Leave at 'auto' to let matplotlib choose the limit.")
            _axis_dlg_spins[_akey] = _asp

        form.addRow("Title font size:", title_fs)
        form.addRow("Axis font size:", axis_fs)
        form.addRow("Tick font size:", tick_fs)
        form.addRow("Export DPI:", dpi_spin)
        form.addRow("Legend position:", legend_combo)
        form.addRow("Error bar style:", error_style_combo)
        form.addRow("Bar spacing:", bar_spacing_spin)
        form.addRow("Error bar cap width:", eb_capsize_spin)
        form.addRow("Error bar line thickness:", eb_lw_spin)
        form.addRow(indiv_points_check)
        form.addRow(show_stats_check)
        form.addRow(force_fit_check)
        form.addRow("Max display width:", max_w_spin)
        form.addRow("Max display height:", max_h_spin)
        form.addRow("X axis min:", _axis_dlg_spins["x_min"])
        form.addRow("X axis max:", _axis_dlg_spins["x_max"])
        form.addRow("Y axis min:", _axis_dlg_spins["y_min"])
        form.addRow("Y axis max:", _axis_dlg_spins["y_max"])

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout(dlg)
        layout.addLayout(form)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        gs["title_fontsize"] = title_fs.value()
        gs["axis_fontsize"] = axis_fs.value()
        gs["tick_fontsize"] = tick_fs.value()
        gs["dpi"] = dpi_spin.value()
        gs["legend_loc"] = legend_combo.currentText()
        gs["error_style"] = str(error_style_combo.currentData() or "SEM")
        gs["bar_spacing"] = bar_spacing_spin.value()
        gs["eb_capsize"] = eb_capsize_spin.value()
        gs["eb_linewidth"] = eb_lw_spin.value()
        gs["show_indiv_points"] = indiv_points_check.isChecked()
        gs["show_stats"] = show_stats_check.isChecked()
        gs["force_fit"] = force_fit_check.isChecked()
        gs["max_w"] = max_w_spin.value()
        gs["max_h"] = max_h_spin.value()
        self._x_min.setValue(_axis_dlg_spins["x_min"].value())
        self._x_max.setValue(_axis_dlg_spins["x_max"].value())
        self._y_min.setValue(_axis_dlg_spins["y_min"].value())
        self._y_max.setValue(_axis_dlg_spins["y_max"].value())
        # Immediately resize canvas to new max dimensions so the user sees
        # the change without needing to regenerate data.
        if self._canvas is not None:
            dpi = int(gs.get("dpi", 100))
            self._canvas.setFixedSize(gs["max_w"], gs["max_h"])
            self._figure.set_size_inches(gs["max_w"] / dpi, gs["max_h"] / dpi)
        self.update_graph()

    def _export_figure(self) -> None:
        if self._figure is None:
            QMessageBox.information(self, "Export", "No figure to export.")
            return
        dpi = int(self._host._graph_settings.get("dpi", 150))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Graph", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All Files (*)",
        )
        if not path:
            return
        try:
            self._figure.savefig(path, dpi=dpi, bbox_inches="tight",
                                 facecolor=self._figure.get_facecolor())
            self._host._status.setText(f"Exported graph to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    # -- data export --------------------------------------------------

    def _collect_graph_data(self, export_individual_sessions: bool = False) -> pd.DataFrame | None:
        """Collect the data underlying the current graph as a DataFrame.

        When ``export_individual_sessions`` is True, group mode exports
        per-session rows (with group labels) instead of group mean/SEM tables.
        """
        checked = self._host._summary_tab._checked_subjects()
        metric = self._get_metric()

        # Distance Traveled is a pseudo-behavior stored under a special ID.
        # _filtered_rows() only returns normal behavior rows (distance_cm = 0
        # on those), so we must pull the distance rows directly — mirroring the
        # logic in update_graph.
        if metric == "distance_cm":
            rows = [
                r for r in self._host._summary_rows
                if r["behavior_id"] == DISTANCE_BEHAVIOR_ID
                and r["session_label"] in checked
            ]
        else:
            rows = [
                r for r in self._host._filtered_rows()
                if r["session_label"] in checked
            ]
        checked_groups = self._checked_groups()
        mode = self._get_mode()
        groups_map = self._host._session_groups
        if mode == "group" and checked_groups:
            rows = [
                r for r in rows
                if groups_map.get(r["session_label"], "") in checked_groups
            ]
        # Apply data range filter
        if self._is_data_range_active():
            rows = self._recompute_rows_for_range(rows)
        # Apply bout filter
        if self._is_bout_filter_active():
            rows = self._recompute_rows_for_first_n(rows)
        rows = self._apply_latency_fallbacks(rows)
        if not rows:
            return None

        df = pd.DataFrame(rows)
        style = self._get_style()
        metric_label = self._metric_label(metric)
        agg_fn = "mean" if metric == "mean_bout_s" else "sum"

        if style in ("line", "overtime"):
            binned = self._bin_bouts()
            if binned.empty:
                return None
            bin_seconds = max(10, int(self._time_bin_spin.value()))
            col = metric if metric != "time_spent_s" else "duration_s"
            # Build complete bin index so empty bins appear as zero
            max_bin = int(binned["time_bin_s"].max())
            all_bins = list(range(0, max_bin + bin_seconds, bin_seconds))

            if mode == "group" and groups_map:
                binned = binned.copy()
                binned["group"] = binned["session_label"].map(groups_map)
                binned = binned.dropna(subset=["group"])
                if checked_groups:
                    binned = binned[binned["group"].isin(checked_groups)]
                if binned.empty:
                    return None
                if metric == "mean_bout_s":
                    sess_bin = binned.groupby(["group", "session_label", "time_bin_s"])[col].mean().reset_index()
                else:
                    sess_bin = binned.groupby(["group", "session_label", "time_bin_s"])[col].sum().reset_index()
                group_list = self._host._ordered_group_list(sess_bin["group"].unique())
                out_rows = []
                for g in group_list:
                    gdf = sess_bin[sess_bin["group"] == g]
                    stats = gdf.groupby("time_bin_s")[col].agg(["mean", "sem", "count"]).reset_index()
                    stats = stats.set_index("time_bin_s").reindex(all_bins, fill_value=0).reset_index()
                    stats.rename(columns={"index": "time_bin_s"}, inplace=True)
                    stats["sem"] = stats["sem"].fillna(0)
                    stats["time_min"] = stats["time_bin_s"] / 60.0
                    stats["group"] = g
                    stats.rename(columns={"mean": metric_label, "sem": "SEM", "count": "N"}, inplace=True)
                    out_rows.append(stats[["group", "time_min", "time_bin_s", metric_label, "SEM", "N"]])
                return pd.concat(out_rows, ignore_index=True) if out_rows else None
            else:
                if metric == "mean_bout_s":
                    sess_bin = binned.groupby(["session_label", "behavior", "time_bin_s"])[col].mean().reset_index()
                else:
                    sess_bin = binned.groupby(["session_label", "behavior", "time_bin_s"])[col].sum().reset_index()
                stats = sess_bin.groupby(["behavior", "time_bin_s"])[col].agg(["mean", "sem", "count"]).reset_index()
                # Reindex each behavior to include all bins
                parts = []
                for beh in stats["behavior"].unique():
                    beh_df = stats[stats["behavior"] == beh].set_index("time_bin_s")
                    beh_df = beh_df.reindex(all_bins, fill_value=0).reset_index()
                    beh_df.rename(columns={"index": "time_bin_s"}, inplace=True)
                    beh_df["behavior"] = beh
                    parts.append(beh_df)
                stats = pd.concat(parts, ignore_index=True)
                stats["sem"] = stats["sem"].fillna(0)
                stats["time_min"] = stats["time_bin_s"] / 60.0
                stats.rename(columns={"mean": metric_label, "sem": "SEM", "count": "N"}, inplace=True)
                return stats[["behavior", "time_min", "time_bin_s", metric_label, "SEM", "N"]]

        elif style == "bar":
            if mode == "group" and groups_map and not export_individual_sessions:
                dfc = df.copy()
                dfc["group"] = dfc["session_label"].map(groups_map)
                dfc = dfc.dropna(subset=["group"])
                # Include "behavior" in the groupby so each behavior gets its
                # own rows.  Without it, all behaviors are collapsed per session
                # (e.g. latencies summed across every behavior → one inflated
                # number per group with no behavior breakdown).
                sess_agg = dfc.groupby(["session_label", "behavior", "group"])[metric].agg(agg_fn).reset_index()
                group_names = self._host._ordered_group_list(sess_agg["group"].unique())
                behaviors_out = sorted(sess_agg["behavior"].unique())
                out_rows = []
                for bname in behaviors_out:
                    for g in group_names:
                        mask = (sess_agg["behavior"] == bname) & (sess_agg["group"] == g)
                        gvals = sess_agg.loc[mask, metric]
                        if gvals.empty:
                            continue
                        out_rows.append({
                            "Behavior": bname, "Group": g, metric_label: gvals.mean(),
                            "SEM": gvals.sem() if len(gvals) > 1 else 0.0, "N": len(gvals),
                        })
                return pd.DataFrame(out_rows)
            else:
                behaviors = sorted(df["behavior"].unique())
                ordered_checked = [
                    s for s in self._host.ordered_session_labels()
                    if s in checked
                ]
                if mode == "group" and checked_groups:
                    ordered_checked = [
                        s for s in ordered_checked
                        if groups_map.get(s, "") in checked_groups
                    ]

                # In group mode, keep subjects ordered by group first,
                # then by the user-defined/alphabetical session order.
                if mode == "group" and groups_map:
                    nonempty_groups = [groups_map.get(s, "") for s in ordered_checked if groups_map.get(s, "")]
                    ordered_groups = self._host._ordered_group_list(nonempty_groups)
                    sessions: list[str] = []
                    for g in ordered_groups:
                        sessions.extend([s for s in ordered_checked if groups_map.get(s, "") == g])
                    sessions.extend([s for s in ordered_checked if not groups_map.get(s, "")])
                else:
                    sessions = list(ordered_checked)

                # Fallback if ordered list is empty for any reason.
                if not sessions:
                    sessions = sorted(df["session_label"].unique())

                groups_by_session = {sess: groups_map.get(sess, "") for sess in sessions}
                lo_s, hi_s = self._get_data_range_seconds()
                out_rows = []
                for bname in behaviors:
                    bdf = df[df["behavior"] == bname]
                    vals_by_session = bdf.groupby("session_label")[metric].agg(agg_fn).to_dict()
                    for sess in sessions:
                        if sess in vals_by_session:
                            v = vals_by_session[sess]
                        elif metric == "latency_s":
                            sid_candidates = self._host._sessions_by_label.get(sess, [])
                            if sid_candidates:
                                # If multiple session_ids map to one label, use the largest fallback latency.
                                v = max(
                                    self._latency_fallback_seconds(str(sid), lo_s, hi_s)
                                    for sid in sid_candidates
                                )
                            else:
                                v = float(hi_s) if hi_s is not None else 0.0
                        else:
                            v = 0
                        out_rows.append({
                            "Session": sess,
                            "Group": groups_by_session.get(sess, ""),
                            "Behavior": bname,
                            metric_label: v,
                        })
                return pd.DataFrame(out_rows)

        elif style == "box":
            if groups_map:
                dfc = df.copy()
                dfc["group"] = dfc["session_label"].map(groups_map)
                dfc = dfc.dropna(subset=["group"])
                sess_agg = dfc.groupby(["session_label", "group"])[metric].agg(agg_fn).reset_index()
                sess_agg.rename(columns={metric: metric_label}, inplace=True)
                return sess_agg[["session_label", "group", metric_label]]
            return None

        elif style == "stacked":
            pivot = df.pivot_table(index="session_label", columns="behavior",
                                   values=metric, aggfunc="sum", fill_value=0)
            pivot.index.name = "Session"
            return pivot.reset_index()

        elif style == "pie":
            agg = df.groupby("behavior")["time_spent_s"].sum().reset_index()
            agg.columns = ["Behavior", "Total Duration (s)"]
            agg["Percent"] = (agg["Total Duration (s)"] / agg["Total Duration (s)"].sum() * 100).round(2)
            return agg

        elif style == "ethogram":
            raw = self._host._raw_bouts
            if not raw:
                return None
            fps = self._host._project_fps()
            selected_bids = self._host._selected_behavior_ids()
            eth_checked = self._checked_ethogram_sessions() or checked
            out_rows = []
            for bid, bdf in raw.items():
                if bid not in selected_bids:
                    continue
                for _, bout in bdf.iterrows():
                    sid = str(bout.get("session_id", ""))
                    sess_label = self._host._session_label_by_session.get(sid, sid)
                    if sess_label not in eth_checked:
                        continue
                    start_s = float(bout["start_frame"]) / fps
                    end_s = float(bout["end_frame"]) / fps
                    out_rows.append({
                        "Session": sess_label, "Behavior": str(bout.get("behavior", bid)),
                        "Start (s)": round(start_s, 3), "End (s)": round(end_s, 3),
                        "Duration (s)": round(max(0, end_s - start_s), 3),
                    })
            return pd.DataFrame(out_rows) if out_rows else None

        elif style == "overview":
            count_df = df.groupby("session_label")["n_bouts"].sum().reset_index()
            dur_df = df.groupby("session_label")["mean_bout_s"].mean().reset_index()
            merged = count_df.merge(dur_df, on="session_label")
            merged.columns = ["Session", "Bout Count", "Mean Bout Duration (s)"]
            return merged

        # Fallback: summary rows
        return df

    def _build_wide_binned_df(self) -> "pd.DataFrame | None":
        """Build a wide per-session DataFrame for time-binned metrics.

        Returns a DataFrame with columns::

            session_label | group | behavior | 0s | 30s | 60s | … | total

        Rows are sorted by group order then session label.
        Returns *None* if no data is available.
        """
        binned = self._bin_bouts()
        if binned.empty:
            return None

        metric = self._get_metric()
        col = metric if metric != "time_spent_s" else "duration_s"
        bin_seconds = max(10, int(self._time_bin_spin.value()))
        groups_map = self._host._session_groups
        checked = self._host._summary_tab._checked_subjects()
        checked_groups = self._checked_groups()
        mode = self._get_mode()

        agg_fn = "mean" if metric == "mean_bout_s" else "sum"
        agg = binned.groupby(
            ["session_label", "behavior", "time_bin_s"]
        )[col].agg(agg_fn).reset_index()

        agg["group"] = agg["session_label"].map(groups_map).fillna("")
        agg = agg[agg["session_label"].isin(checked)]
        if mode == "group" and checked_groups:
            agg = agg[agg["group"].isin(checked_groups)]
        if agg.empty:
            return None

        max_bin = int(agg["time_bin_s"].max())
        all_bins = list(range(0, max_bin + bin_seconds, bin_seconds))
        bin_col_names = [f"{b}s" for b in all_bins]
        behaviors = sorted(agg["behavior"].unique())

        def _sort_key(label: str) -> tuple:
            grp = groups_map.get(label, "")
            ordered = self._host._ordered_group_list(
                sorted({g for g in groups_map.values() if g})
            )
            idx = ordered.index(grp) if grp in ordered else len(ordered)
            return (idx, label)

        sessions = sorted(agg["session_label"].unique(), key=_sort_key)

        parts = []
        for beh in behaviors:
            beh_data = agg[agg["behavior"] == beh]
            pivot = beh_data.pivot_table(
                index="session_label", columns="time_bin_s",
                values=col, aggfunc="first",
            )
            pivot = pivot.reindex(index=sessions, columns=all_bins, fill_value=0).fillna(0)
            pivot.columns = bin_col_names
            pivot = pivot.reset_index()
            pivot.insert(1, "group", pivot["session_label"].map(groups_map).fillna(""))
            pivot.insert(2, "behavior", beh)
            pivot["total"] = pivot[bin_col_names].sum(axis=1)
            parts.append(pivot)

        return pd.concat(parts, ignore_index=True) if parts else None

    def _export_graph_data(self) -> None:
        """Export the data underlying the current graph to a clean CSV.

        Columns are sanitized to ASCII-only so the file pastes cleanly into
        graphing software (Prism, Excel, R, etc.).  Output is ordered
        logically: grouping columns first, then metric columns.
        """
        _TIMEBINNED_METRICS = {"n_bouts", "time_spent_s", "mean_bout_s", "distance_cm"}
        style = self._get_style()
        metric = self._get_metric()

        if style in ("line", "overtime") and metric in _TIMEBINNED_METRICS:
            data = self._build_wide_binned_df()
            if data is None or data.empty:
                QMessageBox.information(self, "Export Data", "No data to export for the current graph.")
                return
        else:
            data = self._collect_graph_data(
                export_individual_sessions=(metric not in _TIMEBINNED_METRICS)
            )
            if data is None or data.empty:
                QMessageBox.information(self, "Export Data", "No data to export for the current graph.")
                return

        # Sanitize column names: replace non-ASCII and problematic chars
        def _sanitize_col(col: str) -> str:
            import re
            # Replace special unicode chars with ASCII equivalents
            col = str(col)
            col = col.replace("\u2013", "-").replace("\u2014", "-").replace("\u00b1", "+-")
            col = col.replace("(", "").replace(")", "").replace(" ", "_")
            # Remove any remaining non-ASCII
            col = re.sub(r"[^\x00-\x7F]", "", col)
            # Collapse repeated underscores/spaces
            col = re.sub(r"_+", "_", col).strip("_")
            return col

        data = data.rename(columns={c: _sanitize_col(str(c)) for c in data.columns})

        # Reorder: put identifier columns first (session, group, behavior, time)
        # then metric columns
        _id_priority = ["session", "session_label", "group", "behavior", "Behavior",
                        "time_min", "time_bin_s", "Time_min"]
        present_ids = [c for c in _id_priority if c in data.columns]
        other_cols = [c for c in data.columns if c not in present_ids]
        data = data[present_ids + other_cols]

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Graph Data", "",
            "CSV (*.csv);;TSV (*.tsv);;All Files (*)",
        )
        if not path:
            return
        try:
            sep = "\t" if path.lower().endswith(".tsv") else ","
            data.to_csv(path, index=False, sep=sep, encoding="utf-8")
            self._host._status.setText(f"Exported graph data to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))


    # -- Excel export -------------------------------------------------

    def _export_excel_data(self) -> None:
        """Export data to an Excel workbook.

        When Ethogram style is active: exports a binary (0/1) presence matrix,
        one sheet per behavior, rows = seconds, columns = selected sessions.
        For session-level summary metrics (e.g. Latency to First) that are not
        present in the time-binned bout data: exports summary table, one sheet
        per behavior.
        Otherwise: exports time-binned continuous data.
        """
        if self._get_style() == "ethogram":
            self._export_ethogram_excel()
            return

        # Metrics that live only as session-level summaries and are NOT columns
        # in _bin_bouts() output (which only contains n_bouts, duration_s,
        # distance_cm).  Routing these through the time-bin path causes a
        # KeyError before the file dialog opens, making the button appear broken.
        _TIMEBINNED_METRICS = {"n_bouts", "time_spent_s", "mean_bout_s", "distance_cm"}
        metric = self._get_metric()
        if metric not in _TIMEBINNED_METRICS:
            self._export_excel_summary_metric()
            return

        binned = self._bin_bouts()
        if binned.empty:
            QMessageBox.information(
                self, "Export Excel",
                "No time-binned data available. Ensure bout files exist.",
            )
            return

        wide = self._build_wide_binned_df()
        if wide is None or wide.empty:
            QMessageBox.information(
                self, "Export Excel",
                "No data for the selected sessions/groups.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Excel", "",
            "Excel Workbook (*.xlsx);;All Files (*)",
        )
        if not path:
            return

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for beh in sorted(wide["behavior"].unique()):
                    beh_df = wide[wide["behavior"] == beh].drop(columns=["behavior"]).reset_index(drop=True)
                    sheet = str(beh)[:31].replace("/", "-").replace("\\", "-")
                    beh_df.to_excel(writer, sheet_name=sheet, index=False)
            self._host._status.setText(f"Exported Excel workbook to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    def _export_excel_summary_metric(self) -> None:
        """Export session-level summary metrics (e.g. Latency to First) to Excel.

        Produces one sheet per behavior.  In group mode each sheet contains
        Group / Mean / SEM / N.  In individual mode each sheet contains
        Session / Value columns.
        """
        data = self._collect_graph_data(export_individual_sessions=True)
        if data is None or data.empty:
            QMessageBox.information(
                self, "Export Excel",
                "No data to export for the current graph.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Excel", "",
            "Excel Workbook (*.xlsx);;All Files (*)",
        )
        if not path:
            return

        metric_label = self._metric_label(self._get_metric())
        mode = self._get_mode()
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                if "Behavior" in data.columns:
                    for bname in sorted(data["Behavior"].unique()):
                        beh_data = data[data["Behavior"] == bname].drop(columns=["Behavior"])
                        sheet = str(bname)[:31].replace("/", "-").replace("\\", "-")
                        beh_data.to_excel(writer, sheet_name=sheet, index=False)
                else:
                    data.to_excel(writer, sheet_name=metric_label[:31], index=False)
            self._host._status.setText(f"Exported Excel workbook to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    def _export_ethogram_excel(self) -> None:
        """Export ethogram as binary (0/1) presence matrix.

        One sheet per behavior. Rows = integer seconds (0 … max_t).
        Columns = one per selected session. Value is 1 if the behavior
        is occurring at that second, 0 otherwise. Easy to copy-paste.
        """
        raw = self._host._raw_bouts
        if not raw:
            QMessageBox.information(
                self, "Export Ethogram Excel",
                "No raw bout data available (need bout parquet files).",
            )
            return

        fps = self._host._project_fps()
        selected_bids = self._host._selected_behavior_ids()
        checked = self._host._summary_tab._checked_subjects()
        eth_checked = self._checked_ethogram_sessions() or checked
        bid_to_name = {
            str(b.behavior_id): str(b.name or b.behavior_id)
            for b in self._host._behaviors.behaviors
        }

        # Collect bout intervals per behavior per session
        # {behavior_name: {session_label: [(start_s, end_s), ...]}}
        beh_sess_bouts: dict[str, dict[str, list[tuple[float, float]]]] = {}
        max_end_s = 0.0

        for bid, bdf in raw.items():
            if bid not in selected_bids:
                continue
            bname = bid_to_name.get(bid, bid)
            if bname not in beh_sess_bouts:
                beh_sess_bouts[bname] = {}
            for _, bout in bdf.iterrows():
                sid = str(bout.get("session_id", ""))
                sess_label = self._host._session_label_by_session.get(sid, sid)
                if sess_label not in eth_checked:
                    continue
                start_s = float(bout["start_frame"]) / fps
                end_s = float(bout["end_frame"]) / fps
                max_end_s = max(max_end_s, end_s)
                beh_sess_bouts[bname].setdefault(sess_label, []).append((start_s, end_s))

        if not beh_sess_bouts:
            QMessageBox.information(
                self, "Export Ethogram Excel",
                "No data for the selected behaviors/sessions.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Ethogram Excel", "",
            "Excel Workbook (*.xlsx);;All Files (*)",
        )
        if not path:
            return

        try:
            sessions_sorted = sorted(eth_checked)
            max_t = int(np.ceil(max_end_s)) + 1  # inclusive last second
            time_index = list(range(max_t))       # [0, 1, 2, ...]

            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for bname in sorted(beh_sess_bouts.keys()):
                    sess_bouts = beh_sess_bouts[bname]
                    # Build a binary column per session
                    data: dict[str, list[int]] = {}
                    for sess in sessions_sorted:
                        arr = np.zeros(max_t, dtype=np.int8)
                        for start_s, end_s in sess_bouts.get(sess, []):
                            t0 = max(0, int(start_s))
                            t1 = min(max_t, int(np.ceil(end_s)))
                            arr[t0:t1] = 1
                        data[sess] = arr.tolist()
                    df_out = pd.DataFrame(data, index=time_index)
                    df_out.index.name = "Time (s)"
                    sheet = str(bname)[:31].replace("/", "-").replace("\\", "-")
                    df_out.to_excel(writer, sheet_name=sheet)

            self._host._status.setText(f"Exported ethogram Excel to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    # -- groups dialog ------------------------------------------------

    def _open_groups_dialog(self) -> None:
        """Open dialog to reorder groups and set custom colours."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor

        groups = sorted({g for g in self._host._session_groups.values() if g})
        if not groups:
            QMessageBox.information(
                self, "Groups",
                "No groups defined yet. Assign factor levels in the Summary tab.",
            )
            return

        ordered = self._host._ordered_group_list(groups)

        dlg = QDialog(self)
        dlg.setWindowTitle("Group Order && Colours")
        dlg.resize(360, 300)
        layout = QVBoxLayout(dlg)
        hint = QLabel("Drag or use arrows to reorder. Click the colour swatch to change.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        list_w = QListWidget(dlg)
        list_w.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

        color_buttons: dict[str, QPushButton] = {}
        for gi, g in enumerate(ordered):
            item = QListWidgetItem(list_w)
            item.setData(Qt.ItemDataRole.UserRole, g)
            row_w = QWidget()
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(4, 2, 4, 2)
            lbl = QLabel(g)
            lbl.setMinimumWidth(120)
            cur_color = self._host._group_colors.get(g, _PALETTE[gi % len(_PALETTE)])
            cbtn = QPushButton()
            cbtn.setFixedSize(28, 22)
            cbtn.setStyleSheet(f"background:{cur_color};border:1px solid #555;border-radius:3px;")
            cbtn._color = cur_color  # type: ignore[attr-defined]
            cbtn._group = g  # type: ignore[attr-defined]

            def _pick(btn: QPushButton = cbtn) -> None:
                c = QColorDialog.getColor(
                    QColor(btn._color), dlg,  # type: ignore[attr-defined]
                    f"Colour for {btn._group}",  # type: ignore[attr-defined]
                )
                if c.isValid():
                    btn._color = c.name()  # type: ignore[attr-defined]
                    btn.setStyleSheet(
                        f"background:{c.name()};border:1px solid #555;border-radius:3px;"
                    )

            cbtn.clicked.connect(_pick)
            row_lay.addWidget(lbl)
            row_lay.addStretch(1)
            row_lay.addWidget(cbtn)
            color_buttons[g] = cbtn
            item.setSizeHint(row_w.sizeHint())
            list_w.addItem(item)
            list_w.setItemWidget(item, row_w)

        layout.addWidget(list_w, 1)

        # Up / Down buttons
        arrow_row = QHBoxLayout()
        up_btn = QPushButton("â–² Up")
        down_btn = QPushButton("â–¼ Down")

        def _move(delta: int) -> None:
            idx = list_w.currentRow()
            if idx < 0:
                return
            new_idx = idx + delta
            if new_idx < 0 or new_idx >= list_w.count():
                return
            item = list_w.takeItem(idx)
            widget = None
            # Re-create row widget after take (Qt destroys the old one)
            g = item.data(Qt.ItemDataRole.UserRole)
            row_w2 = QWidget()
            rl2 = QHBoxLayout(row_w2)
            rl2.setContentsMargins(4, 2, 4, 2)
            rl2.addWidget(QLabel(g))
            rl2.addStretch(1)
            cbtn2 = color_buttons[g]
            rl2.addWidget(cbtn2)
            item.setSizeHint(row_w2.sizeHint())
            list_w.insertItem(new_idx, item)
            list_w.setItemWidget(item, row_w2)
            list_w.setCurrentRow(new_idx)

        up_btn.clicked.connect(lambda: _move(-1))
        down_btn.clicked.connect(lambda: _move(1))
        arrow_row.addWidget(up_btn)
        arrow_row.addWidget(down_btn)
        arrow_row.addStretch(1)
        layout.addLayout(arrow_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Save order
        new_order: list[str] = []
        for i in range(list_w.count()):
            item = list_w.item(i)
            if item is None:
                continue
            g = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if g:
                new_order.append(g)
        if new_order:
            self._host._group_order = list(new_order)

        # Save colors
        for g, cbtn in color_buttons.items():
            c = str(getattr(cbtn, "_color", "") or "").strip()
            if c:
                self._host._group_colors[g] = c

        self._host._save_group_state()
        self._refresh_group_filter()
        self.update_graph()


# ======================================================================
# Sub-tab 3: Spatial Heatmap
# ======================================================================

class _HeatmapWidget(QWidget):
    def __init__(self, host: "BehaviorAnalyticsTab") -> None:
        super().__init__()
        self._host = host
        self._heatmap_graph_settings: dict[str, Any] = {"max_w": 700, "max_h": 550}

        # -- shared state -------------------------------------------------
        self._selected_subjects: set[str] = set()
        self._subject_groups: dict[str, str] = {}
        self._custom_bg_path: str | None = None  # user-imported image

        # -- subject / behavior selectors -----------------------------
        self._select_subjects_btn = QPushButton("Select Subjects…")
        self._select_subjects_btn.clicked.connect(self._open_subject_selector)

        self._behavior_filter_btn = QPushButton("All behaviors ▾")
        self._behavior_filter_btn.setToolTip("Select which behaviors to include in the heatmap.")
        self._behavior_filter_menu = QMenu(self)
        self._behavior_filter_btn.setMenu(self._behavior_filter_menu)
        self._behavior_filter_actions: list[tuple[str, str, QAction]] = []  # (bid, label, action)

        # -- action buttons (top) -------------------------------------
        self._generate_btn = QPushButton("Generate Heatmap")
        self._generate_btn.clicked.connect(self._generate)
        self._export_btn = QPushButton("Export…")
        self._export_btn.clicked.connect(self._export_figure)
        self._color_assign_btn = QPushButton("Colors…")
        self._color_assign_btn.setToolTip("Assign custom colors to each behavior.")
        self._color_assign_btn.clicked.connect(self._open_color_assignment_dialog)
        self._graph_size_btn = QPushButton("Graph Size…")
        self._graph_size_btn.setToolTip("Set maximum display width and height for the heatmap canvas.")
        self._graph_size_btn.clicked.connect(self._open_graph_size_dialog)

        top_btns = QHBoxLayout()
        top_btns.setSpacing(4)
        top_btns.addWidget(self._generate_btn)
        top_btns.addWidget(self._export_btn)
        top_btns.addWidget(self._color_assign_btn)
        top_btns.addWidget(self._graph_size_btn)

        # ── Scatter group ────────────────────────────────────────────
        self._colormap_combo = QComboBox()
        for cm_name in ["Per-behavior color", "viridis", "plasma", "inferno",
                        "magma", "cividis", "hot", "coolwarm", "YlOrRd", "Blues"]:
            self._colormap_combo.addItem(cm_name)

        self._point_size_spin = QDoubleSpinBox()
        self._point_size_spin.setRange(0.5, 30.0)
        self._point_size_spin.setValue(4.0)
        self._point_size_spin.setSingleStep(0.5)

        self._point_alpha = QSlider(Qt.Orientation.Horizontal)
        self._point_alpha.setRange(5, 100)
        self._point_alpha.setValue(35)

        self._show_points = QCheckBox("Show scatter points")
        self._show_points.setChecked(True)

        scatter_grp = QGroupBox("Scatter")
        scatter_grp.setCheckable(True)
        scatter_grp.setChecked(True)
        sf = QFormLayout(scatter_grp)
        sf.setContentsMargins(4, 2, 4, 2)
        sf.setSpacing(3)
        sf.addRow("Colour map:", self._colormap_combo)
        sf.addRow("Point size:", self._point_size_spin)
        sf.addRow("Opacity:", self._point_alpha)
        sf.addRow(self._show_points)

        # ── Density contours group ───────────────────────────────────
        self._kde_bw = QDoubleSpinBox()
        self._kde_bw.setRange(0.01, 2.0)
        self._kde_bw.setValue(0.25)
        self._kde_bw.setSingleStep(0.05)
        self._kde_bw.setDecimals(2)
        self._kde_bw.setToolTip("Kernel bandwidth. Smaller = tighter clusters.")

        self._kde_alpha = QSlider(Qt.Orientation.Horizontal)
        self._kde_alpha.setRange(5, 100)
        self._kde_alpha.setValue(55)

        self._threshold_spin = QSpinBox()
        self._threshold_spin.setRange(1, 50)
        self._threshold_spin.setValue(15)
        self._threshold_spin.setSuffix("% of peak")

        self._show_colorbar = QCheckBox("Show colour bar")
        self._show_colorbar.setChecked(False)

        self._kde_check = QCheckBox()  # kept for API compat; group.isChecked() now drives it
        self._kde_check.setChecked(True)
        self._kde_check.setVisible(False)

        density_grp = QGroupBox("Density Gradient")
        density_grp.setCheckable(True)
        density_grp.setChecked(True)
        density_grp.toggled.connect(self._kde_check.setChecked)
        df = QFormLayout(density_grp)
        df.setContentsMargins(4, 2, 4, 2)
        df.setSpacing(3)
        df.addRow("Bandwidth:", self._kde_bw)
        df.addRow("Opacity:", self._kde_alpha)
        df.addRow("Threshold:", self._threshold_spin)
        df.addRow(self._show_colorbar)

        # ── Background group ─────────────────────────────────────────
        self._bg_mode_combo = QComboBox()
        self._bg_mode_combo.addItem("Auto (from video)", userData="auto")
        self._bg_mode_combo.addItem("Custom image…", userData="custom")
        self._bg_mode_combo.addItem("None (blank)", userData="none")
        self._bg_mode_combo.currentIndexChanged.connect(self._on_bg_mode_changed)

        self._bg_alpha = QSlider(Qt.Orientation.Horizontal)
        self._bg_alpha.setRange(0, 100)
        self._bg_alpha.setValue(30)

        self._pp_contrast = QSlider(Qt.Orientation.Horizontal)
        self._pp_contrast.setRange(50, 300)
        self._pp_contrast.setValue(130)
        self._pp_brightness = QSlider(Qt.Orientation.Horizontal)
        self._pp_brightness.setRange(-80, 80)
        self._pp_brightness.setValue(0)
        self._pp_sharpness = QSlider(Qt.Orientation.Horizontal)
        self._pp_sharpness.setRange(0, 300)
        self._pp_sharpness.setValue(100)
        self._pp_blur = QSlider(Qt.Orientation.Horizontal)
        self._pp_blur.setRange(0, 10)
        self._pp_blur.setValue(0)

        # Fine image adjustments live in a separate, non-modal dialog so the
        # main panel stays compact.  The sliders below are reparented into the
        # dialog (which persists on self), so reading their values still works.
        self._image_adjust_dialog = QDialog(self)
        self._image_adjust_dialog.setWindowTitle("Image Adjustments")
        self._image_adjust_dialog.resize(360, 200)
        _iaf = QFormLayout(self._image_adjust_dialog)
        _iaf.addRow("Contrast:", self._pp_contrast)
        _iaf.addRow("Brightness:", self._pp_brightness)
        _iaf.addRow("Sharpness:", self._pp_sharpness)
        _iaf.addRow("Blur:", self._pp_blur)
        _ia_btn_row = QHBoxLayout()
        _ia_regen_btn = QPushButton("Regenerate")
        _ia_regen_btn.setToolTip("Re-render the heatmap with the current adjustments.")
        _ia_regen_btn.clicked.connect(self._generate)
        _ia_close_btn = QPushButton("Close")
        _ia_close_btn.clicked.connect(self._image_adjust_dialog.hide)
        _ia_btn_row.addStretch(1)
        _ia_btn_row.addWidget(_ia_regen_btn)
        _ia_btn_row.addWidget(_ia_close_btn)
        _iaf.addRow(_ia_btn_row)

        self._image_adjust_btn = QPushButton("Image Adjust…")
        self._image_adjust_btn.setToolTip(
            "Open contrast / brightness / sharpness / blur controls for the "
            "background image."
        )

        def _show_image_adjust() -> None:
            self._image_adjust_dialog.show()
            self._image_adjust_dialog.raise_()
            self._image_adjust_dialog.activateWindow()

        self._image_adjust_btn.clicked.connect(_show_image_adjust)

        bg_grp = QGroupBox("Background Image")
        bf = QFormLayout(bg_grp)
        bf.setContentsMargins(4, 2, 4, 2)
        bf.setSpacing(3)
        bf.addRow("Source:", self._bg_mode_combo)
        bf.addRow("Opacity:", self._bg_alpha)
        bf.addRow(self._image_adjust_btn)

        # ── Display group ────────────────────────────────────────────
        self._show_legend = QCheckBox("Show behavior legend")
        self._show_legend.setChecked(True)

        self._export_transparent = QCheckBox("Transparent background on export")
        self._export_transparent.setChecked(False)
        self._export_transparent.setToolTip(
            "When exporting, save with a transparent background\n"
            "instead of the dark canvas colour."
        )

        display_grp = QGroupBox("Display / Export")
        disp_f = QFormLayout(display_grp)
        disp_f.setContentsMargins(4, 2, 4, 2)
        disp_f.setSpacing(3)
        disp_f.addRow(self._show_legend)
        disp_f.addRow(self._export_transparent)

        # -- controls layout ------------------------------------------
        controls = QWidget()
        ctrl_vbox = QVBoxLayout(controls)
        ctrl_vbox.setContentsMargins(4, 4, 4, 4)
        ctrl_vbox.setSpacing(4)

        sel_row = QHBoxLayout()
        sel_row.addWidget(self._select_subjects_btn, 1)
        beh_row = QHBoxLayout()
        beh_row.addWidget(QLabel("Behaviors:"))
        beh_row.addWidget(self._behavior_filter_btn, 1)

        ctrl_vbox.addLayout(sel_row)
        ctrl_vbox.addLayout(beh_row)
        ctrl_vbox.addLayout(top_btns)
        ctrl_vbox.addWidget(scatter_grp)
        ctrl_vbox.addWidget(density_grp)
        ctrl_vbox.addWidget(bg_grp)
        ctrl_vbox.addWidget(display_grp)
        ctrl_vbox.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        # No maximum width: keep the splitter draggable in both directions.

        # -- canvas ---------------------------------------------------
        self._figure: Any = None
        self._canvas: Any = None
        self._toolbar: Any = None
        self._canvas_scroll: Any = None
        if _ensure_matplotlib() and Figure is not None and FigureCanvas is not None and NavigationToolbar is not None:
            _hm_w = int(self._heatmap_graph_settings.get("max_w", 700))
            _hm_h = int(self._heatmap_graph_settings.get("max_h", 550))
            self._figure = Figure(figsize=(_hm_w / 100, _hm_h / 100), tight_layout=True)
            self._canvas = FigureCanvas(self._figure)
            # Fixed size keeps the canvas compact; scroll area handles overflow.
            self._canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._canvas.setFixedSize(_hm_w, _hm_h)
            self._toolbar = NavigationToolbar(self._canvas, self)
            self._canvas_scroll = QScrollArea()
            self._canvas_scroll.setWidget(self._canvas)
            self._canvas_scroll.setWidgetResizable(False)
            self._canvas_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._canvas_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            # Dynamic resize: canvas grows to fill the available viewport width
            self._hm_resize_filter = _ViewportResizeFilter(
                self._sync_canvas_to_viewport, self
            )
            self._canvas_scroll.viewport().installEventFilter(self._hm_resize_filter)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        if self._toolbar is not None:
            right_l.addWidget(self._toolbar)
        if self._canvas_scroll is not None:
            right_l.addWidget(self._canvas_scroll, 1)
        else:
            right_l.addWidget(QLabel("Matplotlib required for spatial heatmaps."))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1000])

        root = QVBoxLayout(self)
        root.addWidget(splitter, 1)

    def _sync_canvas_to_viewport(self) -> None:
        """Fill the canvas to the scroll viewport width, preserving aspect."""
        _autofill_canvas(self._canvas_scroll, self._canvas, self._figure, dpi=100)

    def showEvent(self, event: Any) -> None:  # type: ignore[override]
        """Re-fill the canvas when this sub-tab becomes visible.

        The resize filter only fires on an actual viewport resize, so a canvas
        drawn while the tab was hidden stays at its provisional size until the
        user wiggles the splitter.  Re-syncing on show fills it immediately.
        """
        super().showEvent(event)
        QTimer.singleShot(0, self._sync_canvas_to_viewport)

    # -- public -------------------------------------------------------

    def _refresh_lists(self) -> None:
        # Update available sessions for the selector dialog
        manifest = (
            self._host._imports.load_manifest(self._host._project_root)
            if self._host._project_root else None
        )
        if manifest is None:
            return
        all_labels = set()
        for s in manifest.linked_sessions:
            label = self._host._session_label_by_session.get(str(s.session_id), str(s.session_id))
            all_labels.add(label)
        # Remove any no-longer-available sessions from selection
        self._selected_subjects &= all_labels
        self._subject_groups = {k: v for k, v in self._subject_groups.items() if k in all_labels}
        # Update behavior checkbox menu
        prev_checked = {bid for bid, _lbl, act in self._behavior_filter_actions if act.isChecked()}
        self._behavior_filter_menu.clear()
        self._behavior_filter_actions.clear()

        select_all_act = self._behavior_filter_menu.addAction("Select All")
        select_none_act = self._behavior_filter_menu.addAction("Select None")
        self._behavior_filter_menu.addSeparator()

        for b in self._host._behaviors.behaviors:
            if str(b.behavior_id) == NO_BEHAVIOR_ID:
                continue
            bid = str(b.behavior_id)
            label = str(b.name)
            action = self._behavior_filter_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(bid in prev_checked if prev_checked else True)
            self._behavior_filter_actions.append((bid, label, action))

        def _check_all():
            for _bid, _lbl, act in self._behavior_filter_actions:
                act.blockSignals(True)
                act.setChecked(True)
                act.blockSignals(False)
            self._update_behavior_filter_label()

        def _check_none():
            for _bid, _lbl, act in self._behavior_filter_actions:
                act.blockSignals(True)
                act.setChecked(False)
                act.blockSignals(False)
            self._update_behavior_filter_label()

        select_all_act.triggered.connect(_check_all)
        select_none_act.triggered.connect(_check_none)
        self._update_behavior_filter_label()

    def _selected_heatmap_behavior_ids(self) -> set[str]:
        """Return the set of behavior IDs currently checked in the heatmap filter menu."""
        return {bid for bid, _lbl, act in self._behavior_filter_actions if act.isChecked()}

    def _update_behavior_filter_label(self) -> None:
        """Update the heatmap behavior button text to reflect the current selection."""
        selected = self._selected_heatmap_behavior_ids()
        total = len(self._behavior_filter_actions)
        if len(selected) == total and total > 0:
            self._behavior_filter_btn.setText("All behaviors \u25be")
        elif not selected:
            self._behavior_filter_btn.setText("(none) \u25be")
        elif len(selected) == 1:
            lbl = next((l for b, l, _ in self._behavior_filter_actions if b in selected), "?")
            self._behavior_filter_btn.setText(f"{lbl} \u25be")
        else:
            self._behavior_filter_btn.setText(f"{len(selected)} selected \u25be")

    def _open_subject_selector(self):
        # Get all sessions from the project
        manifest = (
            self._host._imports.load_manifest(self._host._project_root)
            if self._host._project_root else None
        )
        if manifest is None:
            return
        all_labels = {self._host._session_label_by_session.get(str(s.session_id), str(s.session_id)) for s in manifest.linked_sessions}
        # Pre-populate: if no heatmap selection yet, use Summary tab's
        # checked subjects and host's active-factor groups.
        if not self._selected_subjects:
            self._selected_subjects = self._host._summary_tab._checked_subjects() & all_labels
        merged_groups = dict(self._host._session_groups)
        merged_groups.update(self._subject_groups)
        dlg = _SubjectSelectorDialog(self, all_labels, self._selected_subjects, merged_groups)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            checked, groups = dlg.get_selection()
            self._selected_subjects = checked
            self._subject_groups = groups
            self._host._status.setText(f"Selected {len(checked)} subject(s) for heatmap.")

    def _on_bg_mode_changed(self, idx: int) -> None:
        mode = str(self._bg_mode_combo.currentData())
        if mode == "custom":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Background Image", "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
            )
            if path:
                self._custom_bg_path = path
                self._host._status.setText(f"Custom background: {path}")
            else:
                # Revert to auto if they cancelled
                self._bg_mode_combo.blockSignals(True)
                self._bg_mode_combo.setCurrentIndex(0)
                self._bg_mode_combo.blockSignals(False)

    def _get_selected_session_ids(self, manifest):
        # Map selected session labels to session IDs
        target_sids: list[str] = []
        for label in self._selected_subjects:
            sids = self._host._sessions_by_label.get(label, [])
            target_sids.extend(sids)
        return target_sids

    def _open_color_assignment_dialog(self):
        from PySide6.QtWidgets import QColorDialog, QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
        from PySide6.QtGui import QColor
        dlg = QDialog(self)
        dlg.setWindowTitle("Assign Behavior Colors")
        layout = QVBoxLayout(dlg)
        color_buttons = []
        behaviors = [b for b in self._host._behaviors.behaviors if str(b.behavior_id) != NO_BEHAVIOR_ID]
        for b in behaviors:
            row = QHBoxLayout()
            label = QLabel(str(b.name))
            btn = QPushButton()
            # Set initial color
            color_val = str(b.color or _PALETTE[behaviors.index(b) % len(_PALETTE)])
            btn.setStyleSheet(f"background-color: {color_val};")
            btn._color = color_val
            def make_picker(button, behavior):
                def pick():
                    col = QColorDialog.getColor(QColor(button._color), dlg, f"Pick color for {behavior.name}")
                    if col.isValid():
                        button._color = col.name()
                        button.setStyleSheet(f"background-color: {col.name()};")
                return pick
            btn.clicked.connect(make_picker(btn, b))
            row.addWidget(label)
            row.addWidget(btn)
            layout.addLayout(row)
            color_buttons.append((b, btn))
        save_btn = QPushButton("Save Colors")
        save_btn.clicked.connect(dlg.accept)
        layout.addWidget(save_btn)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Update color and persist to disk
            for b, btn in color_buttons:
                b.color = btn._color
                # Persist color change
                if hasattr(self._host._behaviors, 'update'):
                    self._host._behaviors.update(b.behavior_id, b)
            self._host._refresh_behavior_filter()
            self._refresh_lists()
            self._generate()

    def _generate(self) -> None:
        if self._host._project_root is None:
            QMessageBox.warning(self, "Heatmap", "Open a project first.")
            return
        if not _ensure_matplotlib() or self._figure is None:
            QMessageBox.warning(self, "Heatmap", "Matplotlib is required for heatmaps.")
            return
        manifest = self._host._imports.load_manifest(self._host._project_root)
        if manifest is None:
            QMessageBox.warning(self, "Heatmap", "No project manifest found.")
            return

        self._refresh_lists()

        # Always restrict selection to Summary tab's checked sessions
        summary_checked = self._host._summary_tab._checked_subjects()
        all_labels = {
            self._host._session_label_by_session.get(str(s.session_id), str(s.session_id))
            for s in manifest.linked_sessions
        }
        # Auto-populate from Summary tab's checked sessions if nothing selected
        if not self._selected_subjects:
            self._selected_subjects = summary_checked & all_labels
        else:
            # Enforce Summary tab exclusions on existing selection
            self._selected_subjects &= (summary_checked & all_labels)
        if not self._selected_subjects:
            QMessageBox.information(self, "Heatmap", "Select at least one subject (via 'Select Subjects…').")
            return

        target_sids = self._get_selected_session_ids(manifest)
        if not target_sids:
            QMessageBox.information(self, "Heatmap", "No sessions matched.")
            return

        # Use the first selected subject's first session as background
        bg_sid = target_sids[0]
        behavior_filter = self._selected_heatmap_behavior_ids()
        behavior_list = [
            b for b in self._host._behaviors.behaviors
            if str(b.behavior_id) != NO_BEHAVIOR_ID
        ]
        self._host._status.setText("Generating spatial heatmap…")
        QTimer.singleShot(50, lambda: self._render(
            manifest, bg_sid, behavior_filter, behavior_list, target_sids,
        ))

    # -- render -------------------------------------------------------

    def _render(
        self,
        manifest: Any,
        bg_session_id: str,
        behavior_filter: set[str],
        behavior_list: list[Any],
        target_sids: list[str],
    ) -> None:
        try:
            self._render_inner(manifest, bg_session_id, behavior_filter, behavior_list, target_sids)
        except Exception as exc:
            logger.exception("Heatmap render error: %s", exc)
            self._host._status.setText(f"Heatmap error: {exc}")

    def _render_inner(
        self,
        manifest: Any,
        bg_session_id: str,
        behavior_filter: set[str],
        behavior_list: list[Any],
        target_sids: list[str],
    ) -> None:
        project_root = self._host._project_root
        assert project_root is not None

        video_by_id = {v.asset_id: v for v in manifest.videos}
        pose_by_id = {p.asset_id: p for p in manifest.poses}
        session_by_id = {str(s.session_id): s for s in manifest.linked_sessions}

        bg_session = session_by_id.get(bg_session_id)
        if bg_session is None:
            self._host._status.setText("Background session not found.")
            return

        pose_asset = pose_by_id.get(bg_session.pose_asset_id)
        video_asset = video_by_id.get(bg_session.video_asset_id)
        if pose_asset is None:
            self._host._status.setText("No pose data for selected subject.")
            return

        pose_path = Path(pose_asset.source_path)
        if pose_asset.local_path:
            lp = Path(pose_asset.local_path)
            if lp.exists():
                pose_path = lp
        if not pose_path.exists():
            self._host._status.setText(f"Pose file not found: {pose_path}")
            return
        try:
            bg_pose = self._host._pose.load(pose_path)
        except Exception as exc:
            self._host._status.setText(f"Failed to load pose: {exc}")
            return

        # Composite background image
        bg_image = None
        _orig_video_w: int = 0
        _orig_video_h: int = 0
        _comp_target_size: tuple[int, int] | None = None
        COMPOSITE_SCALE_W = 480
        MAX_COMPOSITE_VIDEOS = 10
        bg_mode = str(self._bg_mode_combo.currentData())

        # Always read original video dims for axis scaling even if bg is off
        if _ensure_cv2() and len(target_sids) > 0:
            for sid in target_sids[:1]:
                sess = session_by_id.get(sid)
                va = video_by_id.get(sess.video_asset_id) if sess else None
                if va is not None:
                    vp = Path(va.local_path) if va.local_path and Path(va.local_path).exists() else Path(va.source_path)
                    if vp.exists():
                        cap = cv2.VideoCapture(str(vp))
                        _orig_video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                        _orig_video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                        cap.release()

        if bg_mode == "custom" and self._custom_bg_path and _CV2_OK:
            # Load user-provided image and deform to fit the arena
            try:
                raw = cv2.imread(self._custom_bg_path, cv2.IMREAD_COLOR)
                if raw is not None:
                    bg_image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    if _orig_video_w > 0 and _orig_video_h > 0:
                        bg_image = cv2.resize(bg_image, (_orig_video_w, _orig_video_h),
                                              interpolation=cv2.INTER_LINEAR)
            except Exception as exc:
                self._host._status.setText(f"Failed to load custom image: {exc}")

        elif bg_mode == "auto" and _CV2_OK and len(target_sids) > 0:
            import traceback
            frames: list[np.ndarray] = []
            errors: list[str] = []
            for sid in target_sids[:MAX_COMPOSITE_VIDEOS]:
                try:
                    sess = session_by_id.get(sid)
                    if sess is None:
                        continue
                    va = video_by_id.get(sess.video_asset_id)
                    if va is None:
                        continue
                    vp = Path(va.local_path) if va.local_path and Path(va.local_path).exists() else Path(va.source_path)
                    if not vp.exists():
                        continue
                    cap = cv2.VideoCapture(str(vp))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, bgr = cap.read()
                    cap.release()
                    if not ret or bgr is None:
                        continue
                    if _orig_video_h == 0:
                        _orig_video_h, _orig_video_w = bgr.shape[:2]
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if _comp_target_size is None:
                        _scale = COMPOSITE_SCALE_W / max(rgb.shape[1], 1)
                        _comp_target_size = (COMPOSITE_SCALE_W, max(1, int(rgb.shape[0] * _scale)))
                    rgb_small = cv2.resize(rgb, _comp_target_size, interpolation=cv2.INTER_AREA)
                    frames.append(rgb_small.astype("float32"))
                except Exception as exc:
                    errors.append(str(exc))
            if frames:
                bg_image = np.mean(frames, axis=0).astype("uint8")
                if _orig_video_w > 0 and _orig_video_h > 0:
                    bg_image = cv2.resize(bg_image, (_orig_video_w, _orig_video_h),
                                          interpolation=cv2.INTER_LINEAR)

        # Apply preprocessing to bg_image (auto or custom)
        if bg_image is not None and _CV2_OK:
            try:
                contrast_scale = self._pp_contrast.value() / 100.0
                brightness_offset = float(self._pp_brightness.value())
                img_f = bg_image.astype(np.float32)
                img_f = (img_f - 128.0) * contrast_scale + 128.0 + brightness_offset
                bg_image = np.clip(img_f, 0, 255).astype(np.uint8)
                sharp_pct = self._pp_sharpness.value()
                if sharp_pct != 100:
                    strength = (sharp_pct - 100) / 100.0
                    if strength > 0:
                        blurred = cv2.GaussianBlur(bg_image, (0, 0), 3)
                        bg_image = cv2.addWeighted(bg_image, 1.0 + strength, blurred, -strength, 0)
                    else:
                        sigma = max(0.5, abs(strength) * 5)
                        ksize = int(sigma * 3) | 1
                        bg_image = cv2.GaussianBlur(bg_image, (ksize, ksize), sigma)
                blur_sigma = self._pp_blur.value()
                if blur_sigma > 0:
                    ksize = blur_sigma * 2 + 1
                    bg_image = cv2.GaussianBlur(bg_image, (ksize, ksize), blur_sigma)
                lab = cv2.cvtColor(bg_image, cv2.COLOR_RGB2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l_ch)
                bg_image = cv2.cvtColor(cv2.merge((cl, a_ch, b_ch)), cv2.COLOR_LAB2RGB)
            except Exception as exc:
                self._host._status.setText(f"Background preprocessing failed: {exc}")

        # pose look-up per session
        poses_by_sid: dict[str, Any] = {}
        for sid in target_sids:
            sess = session_by_id.get(sid)
            if sess is None:
                continue
            pa = pose_by_id.get(sess.pose_asset_id)
            if pa is None:
                continue
            pp = Path(pa.source_path)
            if pa.local_path:
                lp = Path(pa.local_path)
                if lp.exists():
                    pp = lp
            if not pp.exists():
                continue
            try:
                poses_by_sid[sid] = self._host._pose.load(pp)
            except Exception:
                pass
        if bg_session_id not in poses_by_sid:
            poses_by_sid[bg_session_id] = bg_pose

        # behaviour colour map
        bid_to_name: dict[str, str] = {}
        bid_to_color: dict[str, str] = {}
        for i_b, b in enumerate(behavior_list):
            bid = str(b.behavior_id)
            bid_to_name[bid] = str(b.name)
            bid_to_color[bid] = str(b.color or _PALETTE[i_b % len(_PALETTE)])

        # bout intervals
        bouts_dir = project_root / "derived" / "behavior_bouts"
        behavior_bouts: dict[str, pd.DataFrame] = {}
        for b in behavior_list:
            bid = str(b.behavior_id)
            if behavior_filter and bid not in behavior_filter:
                continue
            frames: list[pd.DataFrame] = []
            bout_path = bouts_dir / f"{bid}_bouts.parquet"
            parquet_exists = bout_path.exists()
            if parquet_exists:
                # Parquet is the authoritative source for this behavior.
                # If it exists but has no rows for the selected sessions, that
                # correctly means the animal showed no instances of this behavior
                # in those sessions — do NOT fall back to any other data source,
                # which could mix in bouts from a different behavior entirely.
                try:
                    bdf = pd.read_parquet(bout_path)
                    bdf = bdf[bdf["session_id"].astype(str).isin(target_sids)].reset_index(drop=True)
                    bdf = self._host._apply_prechop_to_bout_df(bdf, rebase=False)
                    if not bdf.empty:
                        frames.append(bdf)
                except Exception:
                    parquet_exists = False  # treat corrupt file as absent
            if not parquet_exists:
                # No exported parquet at all — try temporal-refinement per-session bouts
                for sid in target_sids:
                    tr_bouts = self._load_tr_bouts(bid, sid)
                    if tr_bouts is not None and not tr_bouts.empty:
                        frames.append(tr_bouts)
            if frames:
                behavior_bouts[bid] = pd.concat(frames, ignore_index=True)

        # -- draw -----------------------------------------------------
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#121212")

        bg_alpha_val = self._bg_alpha.value() / 100.0
        if bg_image is not None and bg_alpha_val > 0:
            ax.imshow(bg_image, aspect="auto", alpha=bg_alpha_val)

        all_x_parts: list[np.ndarray] = []
        all_y_parts: list[np.ndarray] = []
        per_bid_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for bid, bdf in behavior_bouts.items():
            xs_parts: list[np.ndarray] = []
            ys_parts: list[np.ndarray] = []
            # group by session to avoid repeated dict lookups per row
            for sid_val, grp in bdf.groupby("session_id", sort=False):
                p = poses_by_sid.get(str(sid_val))
                if p is None:
                    continue
                n_frames = p.n_frames
                starts = np.clip(grp["start_frame"].to_numpy(dtype=np.int64), 0, n_frames - 1)
                ends = np.clip(grp["end_frame"].to_numpy(dtype=np.int64), 0, n_frames - 1)
                ends = np.maximum(starts, ends)
                for s, e in zip(starts, ends):
                    xs_parts.append(p.centroid_x[s:e + 1])
                    ys_parts.append(p.centroid_y[s:e + 1])
            if xs_parts:
                xs_arr = np.concatenate(xs_parts)
                ys_arr = np.concatenate(ys_parts)
                per_bid_xy[bid] = (xs_arr, ys_arr)
                all_x_parts.append(xs_arr)
                all_y_parts.append(ys_arr)

        all_x = np.concatenate(all_x_parts) if all_x_parts else np.empty(0)
        all_y = np.concatenate(all_y_parts) if all_y_parts else np.empty(0)

        pt_size = float(self._point_size_spin.value())
        pt_alpha = self._point_alpha.value() / 100.0
        show_pts = self._show_points.isChecked()
        show_kde = self._kde_check.isChecked()
        use_cmap = self._colormap_combo.currentText()
        is_per_beh = use_cmap == "Per-behavior color"

        # scatter points
        if show_pts and per_bid_xy:
            for bid, (bxs, bys) in per_bid_xy.items():
                kw: dict[str, Any] = {
                    "s": pt_size, "alpha": pt_alpha, "edgecolors": "none",
                    "label": bid_to_name.get(bid, bid),
                }
                if is_per_beh:
                    kw["color"] = bid_to_color.get(bid, _PALETTE[0])
                else:
                    kw["c"] = bys
                    kw["cmap"] = use_cmap
                ax.scatter(bxs, bys, **kw)

        # Density contours — fast histogram2d + Gaussian blur (GPU-accelerated if
        # CuPy is available, otherwise NumPy CPU path).  Replaces the old
        # gaussian_kde approach which scaled as O(n²) in the number of points.
        if show_kde and per_bid_xy:
            try:
                from matplotlib.colors import to_rgba
                from scipy.ndimage import gaussian_filter

                # Try GPU (CuPy) first; fall back to NumPy silently.
                try:
                    import cupy as cp  # type: ignore[import-untyped]
                    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter  # type: ignore[import-untyped]
                    _USE_GPU = True
                except Exception:
                    cp = None  # type: ignore[assignment]
                    _USE_GPU = False

                if bg_image is not None:
                    h, w = bg_image.shape[:2]
                elif _orig_video_w > 0 and _orig_video_h > 0:
                    w, h = _orig_video_w, _orig_video_h
                else:
                    w = int(float(all_x.max())) + 10 if all_x.size else 100
                    h = int(float(all_y.max())) + 10 if all_y.size else 100
                grid_res = min(400, max(w, h))
                kde_bw = float(self._kde_bw.value())
                kde_alpha_val = self._kde_alpha.value() / 100.0
                thresh_frac = self._threshold_spin.value() / 100.0
                # Sigma in grid pixels: scale bandwidth to grid resolution
                sigma = max(1.0, kde_bw * grid_res / 10.0)

                _last_cmap_obj = None

                for bid, (bxs, bys) in per_bid_xy.items():
                    if len(bxs) < 10:
                        continue
                    # 2-D histogram (O(n) binning) then smooth
                    H, _, _ = np.histogram2d(
                        bxs, bys,
                        bins=grid_res,
                        range=[[0, w], [0, h]],
                    )
                    H = H.T.astype(np.float32)  # align to image coords
                    if _USE_GPU and cp is not None:
                        H_gpu = cp.asarray(H)
                        Z_gpu = gpu_gaussian_filter(H_gpu, sigma=sigma)
                        Z = cp.asnumpy(Z_gpu).astype(np.float64)
                    else:
                        Z = gaussian_filter(H, sigma=sigma).astype(np.float64)

                    z_max = float(Z.max())
                    if z_max <= 0:
                        continue

                    # Remap: below threshold → alpha 0, at peak → kde_alpha_val.
                    # This produces a smooth radiant gradient that fades to
                    # transparent at the edges and glows at density epicenters.
                    z_thresh = z_max * thresh_frac
                    Z_norm = np.clip(
                        (Z - z_thresh) / (z_max - z_thresh + 1e-12), 0.0, 1.0
                    )

                    # Build per-pixel RGBA float32 array
                    if is_per_beh:
                        base = to_rgba(bid_to_color.get(bid, _PALETTE[0]))
                        rgba = np.zeros((*Z_norm.shape, 4), dtype=np.float32)
                        rgba[..., 0] = base[0]
                        rgba[..., 1] = base[1]
                        rgba[..., 2] = base[2]
                        rgba[..., 3] = Z_norm * kde_alpha_val
                    else:
                        import matplotlib.pyplot as plt
                        cmap_obj = plt.get_cmap(use_cmap)
                        _last_cmap_obj = cmap_obj
                        rgba = cmap_obj(Z_norm).astype(np.float32)
                        rgba[..., 3] = Z_norm * kde_alpha_val

                    # extent=[left, right, bottom, top] matches pose pixel coords
                    ax.imshow(
                        rgba,
                        extent=[0, w, h, 0],
                        origin="upper",
                        aspect="auto",
                        interpolation="bilinear",
                        zorder=2,
                    )

                if (self._show_colorbar.isChecked()
                        and not is_per_beh
                        and _last_cmap_obj is not None):
                    import matplotlib.cm as _cm
                    sm = _cm.ScalarMappable(cmap=_last_cmap_obj)
                    sm.set_array([])
                    self._figure.colorbar(sm, ax=ax, label="Density (norm.)", shrink=0.7)
            except Exception as exc:
                logger.warning("Density overlay failed: %s", exc)
        # ax.imshow() already inverts the y-axis (origin='upper') so that
        # row 0 is at the top, matching pose pixel coordinates.  Only invert
        # manually when there is no background image.
        if bg_image is None:
            ax.invert_yaxis()

        # Always pin axes to the full arena extent so that switching between
        # one behavior and all behaviors never changes the spatial scale.
        _ax_w = _orig_video_w if _orig_video_w > 0 else (bg_image.shape[1] if bg_image is not None else 0)
        _ax_h = _orig_video_h if _orig_video_h > 0 else (bg_image.shape[0] if bg_image is not None else 0)
        if _ax_w > 0 and _ax_h > 0:
            ax.set_xlim(0, _ax_w)
            ax.set_ylim(_ax_h, 0)  # y=0 at top, matching pose/image pixel coords

        # Behavior legend
        if self._show_legend.isChecked() and per_bid_xy:
            from matplotlib.patches import Patch
            legend_handles = []
            for bid in per_bid_xy:
                color = bid_to_color.get(bid, _PALETTE[0])
                label = bid_to_name.get(bid, bid)
                legend_handles.append(Patch(facecolor=color, edgecolor="none", label=label))
            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    loc="upper right",
                    fontsize=9,
                    framealpha=0.6,
                    facecolor="#1a2027",
                    edgecolor="#37474f",
                    labelcolor="white",
                )

        n_sess = len(target_sids)
        self._sync_canvas_to_viewport()
        self._canvas.draw_idle()
        self._host._status.setText(
            f"Heatmap rendered: {len(all_x):,} points from "
            f"{len(behavior_bouts)} behavior(s) across {n_sess} session(s)."
        )

    # -- helpers ------------------------------------------------------

    def _load_tr_bouts(self, bid: str, session_id: str) -> pd.DataFrame | None:
        project_root = self._host._project_root
        if project_root is None:
            return None
        tr_root = project_root / "derived" / "temporal_refinement"
        for token in (self._host._safe_name(bid), "target_behavior"):
            latest_path = tr_root / token / "latest.json"
            if not latest_path.exists():
                continue
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            post_dir = str(latest.get("postprocess_dir", "") or "").strip()
            if not post_dir:
                continue
            manifest_path = Path(post_dir) / "postprocess_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                pm = json.loads(manifest_path.read_text(encoding="utf-8"))
                bout_paths = {
                    str(k): str(v) for k, v in
                    (pm.get("bout_paths", {}) or {}).items()
                }
            except Exception:
                continue
            bp = bout_paths.get(session_id, "")
            if bp and Path(bp).exists():
                try:
                    df = pd.read_parquet(Path(bp))
                except Exception:
                    continue
                # The "target_behavior" run contains bouts for ALL
                # behaviors combined.  We must filter to the requested
                # behavior to avoid misassigning bouts/colours.
                if token == "target_behavior" and not df.empty:
                    if "behavior_id" in df.columns:
                        df = df[df["behavior_id"].astype(str) == bid]
                    elif "predicted_behavior" in df.columns:
                        df = df[df["predicted_behavior"].astype(str) == bid]
                    else:
                        # Cannot attribute bouts to a specific behavior
                        continue
                df = self._host._apply_prechop_to_bout_df(df, rebase=False)
                if not df.empty:
                    return df
        return None

    def _open_graph_size_dialog(self) -> None:
        """Open a small dialog to set max display width/height for the heatmap canvas."""
        gs = self._heatmap_graph_settings
        dlg = QDialog(self)
        dlg.setWindowTitle("Heatmap Graph Size")
        dlg.resize(300, 130)
        form = QFormLayout()

        max_w_spin = QSpinBox(dlg)
        max_w_spin.setRange(200, 3840)
        max_w_spin.setSingleStep(50)
        max_w_spin.setSuffix(" px")
        max_w_spin.setToolTip("Maximum display width of the heatmap canvas in pixels.")
        max_w_spin.setValue(int(gs.get("max_w", 800)))

        max_h_spin = QSpinBox(dlg)
        max_h_spin.setRange(150, 2160)
        max_h_spin.setSingleStep(50)
        max_h_spin.setSuffix(" px")
        max_h_spin.setToolTip("Maximum display height of the heatmap canvas in pixels.")
        max_h_spin.setValue(int(gs.get("max_h", 600)))

        form.addRow("Max width:", max_w_spin)
        form.addRow("Max height:", max_h_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout = QVBoxLayout(dlg)
        layout.addLayout(form)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        gs["max_w"] = max_w_spin.value()
        gs["max_h"] = max_h_spin.value()
        if self._canvas is not None:
            self._canvas.setFixedSize(gs["max_w"], gs["max_h"])
            self._figure.set_size_inches(gs["max_w"] / 100, gs["max_h"] / 100)

    def _export_figure(self) -> None:
        if self._figure is None:
            QMessageBox.information(self, "Export", "No heatmap to export.")
            return
        dpi = int(self._host._graph_settings.get("dpi", 150))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Heatmap", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All Files (*)",
        )
        if not path:
            return
        try:
            transparent = self._export_transparent.isChecked()
            fc = "none" if transparent else self._figure.get_facecolor()
            self._figure.savefig(
                path, dpi=dpi, bbox_inches="tight",
                facecolor=fc, transparent=transparent,
            )
            self._host._status.setText(f"Exported heatmap to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))


# ======================================================================
# Sub-tab 4 (new): Density Analysis — normalized density maps + group diff
# ======================================================================

import math as _math


def _make_diverging_transparent_center_cmap(base_cmap_name: str) -> Any:
    """Build a two-colour diverging colormap: colour_A → transparent → colour_B.

    The root problem with standard diverging colormaps (RdBu_r, coolwarm …)
    is that their RGB values are *desaturated / white* near the centre.  Even
    at low alpha those pale colours produce grey blobs over a dark background.

    This version avoids that by **only ever using the fully-saturated endpoint
    colours** for RGB.  The left half of the LUT always uses colour_A's RGB
    (sampled at t=0.02) and the right half always uses colour_B's RGB (sampled
    at t=0.98).  The alpha channel follows a squared V-shape so only
    genuinely large differences are opaque; near-zero differences are
    completely invisible regardless of the underlying colormap.

    Result: red (or whatever colour_A is) fades cleanly to *transparent* at the
    midpoint and then reappears as blue (colour_B) on the other side — no white,
    no grey, no desaturated haze.
    """
    try:
        import matplotlib.pyplot as _plt
        import matplotlib.colors as _mc

        base = _plt.get_cmap(base_cmap_name)
        N = 512
        t = np.linspace(0.0, 1.0, N)

        # Sample only the outer 2 % of the original cmap to get pure,
        # fully-saturated colours uncontaminated by the white centre.
        c_left  = np.array(base(0.02)[:3], dtype=np.float64)  # e.g. deep red
        c_right = np.array(base(0.98)[:3], dtype=np.float64)  # e.g. deep blue

        rgba = np.ones((N, 4), dtype=np.float64)
        mid = N // 2
        rgba[:mid, :3] = c_left    # left half  → colour A (constant RGB)
        rgba[mid:, :3] = c_right   # right half → colour B (constant RGB)

        # V-shape alpha: exactly 0 at t=0.5, exactly 1 at t=0 and t=1.
        # Power 0.7 keeps moderate differences clearly visible; the curve
        # rises steeply from zero so even subtle differences appear.
        rgba[:, 3] = np.abs(2.0 * t - 1.0) ** 0.7

        return _mc.ListedColormap(rgba, name=f"{base_cmap_name}_tc")
    except Exception:
        # Fallback: return original cmap unchanged
        try:
            import matplotlib.pyplot as _plt
            return _plt.get_cmap(base_cmap_name)
        except Exception:
            return None


def _compute_auto_bandwidth(
    video_w: int,
    video_h: int,
    grid_res: int,
    subject_span_px: float | None = None,
    multiplier: float = 1.0,
) -> float:
    """Return a Gaussian sigma (in grid pixels) suited to the arena geometry.

    The base bandwidth is estimated as ~3 % of the arena diagonal, which
    typically corresponds to ~1–1.5 animal body lengths for standard rodent
    arenas recorded at common resolutions.  When *subject_span_px* is
    provided (the median keypoint span in pixels) the base is derived from
    the actual subject size instead.

    The resulting sigma is scaled to grid coordinates and multiplied by the
    user-configurable *multiplier* (default 1.0 = no adjustment).
    """
    diag_px = _math.sqrt(max(video_w, 1) ** 2 + max(video_h, 1) ** 2)
    if subject_span_px is not None and subject_span_px > 4:
        base_px = float(subject_span_px) * 1.2
    else:
        base_px = diag_px * 0.03  # 3 % of diagonal

    # Convert from pixel space to grid space
    longest_side = max(video_w, video_h, 1)
    scale = grid_res / longest_side
    sigma = base_px * scale * max(0.05, float(multiplier))
    return max(1.0, sigma)


def _estimate_subject_span(pose_data: Any, max_frames: int = 2000) -> float | None:
    """Return the median keypoint bounding-box diagonal in pixels.

    Uses up to *max_frames* uniformly sampled frames to keep the operation
    fast.  Returns *None* when pose data lacks coordinate arrays.
    """
    try:
        xs = getattr(pose_data, "xs", None)   # (n_frames, n_kps)
        ys = getattr(pose_data, "ys", None)
        if xs is None or ys is None:
            return None
        n = xs.shape[0]
        step = max(1, n // max_frames)
        xs_s = xs[::step]
        ys_s = ys[::step]
        spans: list[float] = []
        for i in range(xs_s.shape[0]):
            x_row = xs_s[i]
            y_row = ys_s[i]
            valid = np.isfinite(x_row) & np.isfinite(y_row)
            if valid.sum() < 2:
                continue
            dx = float(x_row[valid].max() - x_row[valid].min())
            dy = float(y_row[valid].max() - y_row[valid].min())
            spans.append(_math.sqrt(dx ** 2 + dy ** 2))
        return float(np.median(spans)) if spans else None
    except Exception:
        return None


class _DensityAnalysisWidget(QWidget):
    """Density-map analysis with per-group heatmaps and group-difference maps.

    Offers two sub-tabs:
      • **Density Maps** – normalised KDE density maps, one subplot per
        behavior, with optional per-group overlay mode.
      • **Group Comparison** – signed or normalised difference map between
        two user-selected groups (or factor-combination groups), rendered
        with a diverging colormap.

    Bandwidth is estimated automatically from the arena resolution and the
    detected subject size, with a user-controllable multiplier for fine-
    tuning.
    """

    # Configurable defaults persisted in the widget (not to disk).
    _DEFAULT_SETTINGS: dict[str, Any] = {
        "auto_bw": True,
        "bw_multiplier": 1.0,
        "manual_bw": 0.05,          # fraction of arena diagonal
        "grid_res": 256,
        "threshold_pct": 2,          # bottom % trimmed before normalizing
        "colormap_density": "viridis",
        "colormap_diff": "RdBu_r",
        "bg_mode": "auto",
        "bg_alpha": 30,
        "overlay_alpha": 90,
        "show_colorbar": True,
        "show_subject_count": True,
        "normalize_mode": "peak",    # "peak" | "sum"
        "diff_metric": "signed",     # "signed" | "normalized" | "log2ratio"
        "diff_mask_threshold": 0.05, # mask diff where both groups < this fraction of peak
        "show_group_contours": False, # overlay individual group contours on diff map
        "max_w": 800,
        "max_h": 600,
        "export_dpi": 200,
    }

    def __init__(self, host: "BehaviorAnalyticsTab") -> None:
        super().__init__()
        self._host = host
        self._settings: dict[str, Any] = dict(self._DEFAULT_SETTINGS)

        # ── shared rendering state ────────────────────────────────────
        self._density_figure: Any = None
        self._density_canvas: Any = None
        self._density_toolbar: Any = None
        self._diff_figure: Any = None
        self._diff_canvas: Any = None
        self._diff_toolbar: Any = None

        # ── data caches for instant re-render on display-setting changes ──
        # Changing colormap/alpha/metric/mask re-draws from cache (fast).
        # Changing groups/bandwidth/grid-res invalidates the cache (slow).
        self._diff_cache: dict[str, Any] = {}       # bid → {Z_a, Z_b, n_a, n_b}
        self._diff_cache_meta: dict[str, Any] = {}  # key, video dims, bg_image, sids
        self._dm_cache: dict[str, Any] = {}         # (bid, grp_label) → {Z, n_pts, n_sess}
        self._dm_cache_meta: dict[str, Any] = {}
        self._density_manifest_cache: dict[str, Any] = {}
        self._density_session_source_cache: dict[str, dict[str, Any] | None] = {}

        # ── build inner sub-tabs ──────────────────────────────────────
        self._inner = QTabWidget()
        self._density_panel = self._build_density_panel()
        self._diff_panel = self._build_diff_panel()
        self._inner.addTab(self._density_panel, "Density Maps")
        self._inner.addTab(self._diff_panel, "Group Comparison")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._inner, 1)

    # ──────────────────────────────────────────────────────────────────
    # Panel builders
    # ──────────────────────────────────────────────────────────────────

    def _build_density_panel(self) -> QWidget:
        """Build the Density Maps panel (controls + canvas)."""
        w = QWidget()

        # ── Controls ──────────────────────────────────────────────────
        # Behavior selector
        self._dm_behavior_combo = QComboBox()
        self._dm_behavior_combo.setToolTip(
            "Select a single behavior or 'All behaviors' to render\n"
            "one subplot per behavior in the same figure."
        )

        # Session / group selector
        self._dm_group_mode_combo = QComboBox()
        self._dm_group_mode_combo.addItem("All sessions (combined)", userData="all")
        self._dm_group_mode_combo.addItem("Per group (subplots)", userData="per_group")
        self._dm_group_mode_combo.addItem("Specific group…", userData="specific")
        self._dm_group_mode_combo.currentIndexChanged.connect(self._on_dm_group_mode_changed)
        self._dm_group_mode_combo.setToolTip(
            "All sessions: pool every loaded session into one density map.\n"
            "Per group: render one column of subplots per group.\n"
            "Specific group: show only sessions from one group."
        )

        self._dm_specific_group_combo = QComboBox()
        self._dm_specific_group_combo.setVisible(False)
        self._dm_specific_group_combo.setToolTip("Group to display when 'Specific group…' is selected.")

        sel_grp = QGroupBox("Data Selection")
        sel_grp.setCheckable(False)
        sel_f = QFormLayout(sel_grp)
        sel_f.setContentsMargins(6, 4, 6, 4)
        sel_f.setSpacing(4)
        sel_f.addRow("Behavior:", self._dm_behavior_combo)
        sel_f.addRow("Mode:", self._dm_group_mode_combo)
        sel_f.addRow("Group:", self._dm_specific_group_combo)

        # Bandwidth group
        self._dm_auto_bw = QCheckBox("Auto-detect bandwidth")
        self._dm_auto_bw.setChecked(bool(self._settings["auto_bw"]))
        self._dm_auto_bw.setToolTip(
            "Estimate the optimal KDE bandwidth from the arena resolution\n"
            "and measured subject size.  Uncheck to set manually."
        )
        self._dm_auto_bw.toggled.connect(self._on_auto_bw_toggled)

        self._dm_bw_multiplier = QDoubleSpinBox()
        self._dm_bw_multiplier.setRange(0.1, 8.0)
        self._dm_bw_multiplier.setValue(float(self._settings["bw_multiplier"]))
        self._dm_bw_multiplier.setSingleStep(0.1)
        self._dm_bw_multiplier.setDecimals(2)
        self._dm_bw_multiplier.setToolTip(
            "Multiply the auto-detected bandwidth by this factor.\n"
            "< 1 = sharper (tighter clusters), > 1 = smoother (broader spread)."
        )

        self._dm_manual_bw = QDoubleSpinBox()
        self._dm_manual_bw.setRange(0.001, 1.0)
        self._dm_manual_bw.setValue(float(self._settings["manual_bw"]))
        self._dm_manual_bw.setSingleStep(0.005)
        self._dm_manual_bw.setDecimals(4)
        self._dm_manual_bw.setSuffix(" × diag")
        self._dm_manual_bw.setToolTip(
            "Bandwidth as a fraction of the arena diagonal (pixel space).\n"
            "0.03 ≈ 3 % of diagonal, suitable for whole-body-sized density blobs."
        )
        self._dm_manual_bw.setEnabled(not bool(self._settings["auto_bw"]))

        self._dm_grid_res_combo = QComboBox()
        for res_label, res_val in [("128 (fast)", 128), ("256 (balanced)", 256),
                                    ("512 (high detail)", 512)]:
            self._dm_grid_res_combo.addItem(res_label, userData=res_val)
        self._dm_grid_res_combo.setCurrentIndex(1)
        self._dm_grid_res_combo.setToolTip(
            "Grid resolution for the density map.\n"
            "Higher = more detail but slower rendering."
        )

        bw_grp = QGroupBox("Bandwidth / Resolution")
        bw_f = QFormLayout(bw_grp)
        bw_f.setContentsMargins(6, 4, 6, 4)
        bw_f.setSpacing(4)
        bw_f.addRow(self._dm_auto_bw)
        bw_f.addRow("Multiplier:", self._dm_bw_multiplier)
        bw_f.addRow("Manual BW:", self._dm_manual_bw)
        bw_f.addRow("Grid res:", self._dm_grid_res_combo)

        # Display group
        self._dm_colormap_combo = QComboBox()
        self._dm_colormap_combo.setEditable(True)
        self._dm_colormap_combo.setToolTip(
            "Colormap for the density heatmap.\n"
            "Select from the list or type any valid matplotlib colormap name."
        )
        for cm in ["inferno", "plasma", "viridis", "magma", "hot",
                   "YlOrRd", "OrRd", "Blues", "Greens", "cividis",
                   "turbo", "jet"]:
            self._dm_colormap_combo.addItem(cm)
        idx_cm = self._dm_colormap_combo.findText(str(self._settings["colormap_density"]))
        if idx_cm < 0:
            self._dm_colormap_combo.setCurrentText(str(self._settings["colormap_density"]))
        else:
            self._dm_colormap_combo.setCurrentIndex(idx_cm)

        self._dm_normalize_combo = QComboBox()
        self._dm_normalize_combo.addItem("Peak = 1 (relative)", userData="peak")
        self._dm_normalize_combo.addItem("Density sum (absolute)", userData="sum")
        self._dm_normalize_combo.setToolTip(
            "Peak = 1: each map rescaled so peak = 1 (comparable pattern shape).\n"
            "Density sum: maps reflect relative time/bout count differences."
        )

        self._dm_threshold_spin = QSpinBox()
        self._dm_threshold_spin.setRange(0, 20)
        self._dm_threshold_spin.setValue(int(self._settings["threshold_pct"]))
        self._dm_threshold_spin.setSuffix(" %")
        self._dm_threshold_spin.setToolTip(
            "Trim the lowest N% of density values to zero before rendering.\n"
            "Reduces visual noise from sparse, infrequent visits."
        )

        self._dm_overlay_alpha = QSlider(Qt.Orientation.Horizontal)
        self._dm_overlay_alpha.setRange(10, 100)
        self._dm_overlay_alpha.setValue(int(self._settings["overlay_alpha"]))
        self._dm_overlay_alpha_lbl = QLabel(f"{self._settings['overlay_alpha']}%")
        self._dm_overlay_alpha.valueChanged.connect(
            lambda v: self._dm_overlay_alpha_lbl.setText(f"{v}%")
        )

        self._dm_bg_mode_combo = QComboBox()
        self._dm_bg_mode_combo.addItem("Auto (from video)", userData="auto")
        self._dm_bg_mode_combo.addItem("None (dark)", userData="none")
        self._dm_bg_mode_combo.setCurrentIndex(
            0 if self._settings["bg_mode"] == "auto" else 1
        )

        self._dm_bg_alpha = QSlider(Qt.Orientation.Horizontal)
        self._dm_bg_alpha.setRange(0, 100)
        self._dm_bg_alpha.setValue(int(self._settings["bg_alpha"]))
        self._dm_bg_alpha_lbl = QLabel(f"{self._settings['bg_alpha']}%")
        self._dm_bg_alpha.valueChanged.connect(
            lambda v: self._dm_bg_alpha_lbl.setText(f"{v}%")
        )

        self._dm_show_colorbar = QCheckBox("Show colorbar")
        self._dm_show_colorbar.setChecked(bool(self._settings["show_colorbar"]))

        self._dm_show_count = QCheckBox("Show subject/session count")
        self._dm_show_count.setChecked(bool(self._settings["show_subject_count"]))
        self._dm_show_count.setToolTip("Annotate each subplot with n = <session count>.")

        disp_grp = QGroupBox("Display")
        disp_f = QFormLayout(disp_grp)
        disp_f.setContentsMargins(6, 4, 6, 4)
        disp_f.setSpacing(4)
        disp_f.addRow("Colormap:", self._dm_colormap_combo)
        disp_f.addRow("Normalize:", self._dm_normalize_combo)
        disp_f.addRow("Low-trim:", self._dm_threshold_spin)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(self._dm_overlay_alpha)
        alpha_row.addWidget(self._dm_overlay_alpha_lbl)
        disp_f.addRow("Overlay:", alpha_row)

        bg_alpha_row = QHBoxLayout()
        bg_alpha_row.addWidget(self._dm_bg_alpha)
        bg_alpha_row.addWidget(self._dm_bg_alpha_lbl)
        disp_f.addRow("BG source:", self._dm_bg_mode_combo)
        disp_f.addRow("BG opacity:", bg_alpha_row)
        disp_f.addRow(self._dm_show_colorbar)
        disp_f.addRow(self._dm_show_count)

        # Action buttons
        self._dm_generate_btn = QPushButton("Generate Density Map")
        self._dm_generate_btn.setToolTip("Compute density grids from raw data and render.")
        self._dm_generate_btn.clicked.connect(self._generate_density)

        self._dm_export_btn = QPushButton("Export…")
        self._dm_export_btn.setToolTip("Save the current density map as PNG / SVG / PDF.")
        self._dm_export_btn.clicked.connect(lambda: self._export_figure(self._density_figure))

        self._dm_settings_btn = QPushButton("Settings…")
        self._dm_settings_btn.setToolTip("Configure density analysis settings.")
        self._dm_settings_btn.clicked.connect(self._open_settings_dialog)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        btn_row.addWidget(self._dm_generate_btn)
        btn_row.addWidget(self._dm_export_btn)
        btn_row.addWidget(self._dm_settings_btn)
        btn_row.addStretch(1)

        self._dm_stale_lbl = QLabel("⚠ Data changed — click Generate to recompute.")
        self._dm_stale_lbl.setStyleSheet("color:#ffb300; font-size:9px; padding:1px 4px;")
        self._dm_stale_lbl.setVisible(False)

        # Assemble controls
        ctrl_inner = QWidget()
        ctrl_vbox = QVBoxLayout(ctrl_inner)
        ctrl_vbox.setContentsMargins(4, 4, 4, 4)
        ctrl_vbox.setSpacing(6)
        ctrl_vbox.addLayout(btn_row)
        ctrl_vbox.addWidget(self._dm_stale_lbl)
        ctrl_vbox.addWidget(sel_grp)
        ctrl_vbox.addWidget(bw_grp)
        ctrl_vbox.addWidget(disp_grp)
        ctrl_vbox.addStretch(1)

        # Wire: data-changing controls → stale warning
        for _sig in [
            self._dm_behavior_combo.currentIndexChanged,
            self._dm_group_mode_combo.currentIndexChanged,
            self._dm_specific_group_combo.currentIndexChanged,
            self._dm_grid_res_combo.currentIndexChanged,
            self._dm_auto_bw.toggled,
            self._dm_bw_multiplier.valueChanged,
            self._dm_manual_bw.valueChanged,
            self._dm_normalize_combo.currentIndexChanged,
            self._dm_threshold_spin.valueChanged,
        ]:
            _sig.connect(self._mark_dm_stale)

        # Wire: display-only controls → instant redraw from cache
        for _sig in [
            self._dm_colormap_combo.currentTextChanged,
            self._dm_overlay_alpha.valueChanged,
            self._dm_bg_alpha.valueChanged,
            self._dm_bg_mode_combo.currentIndexChanged,
            self._dm_show_colorbar.toggled,
            self._dm_show_count.toggled,
        ]:
            _sig.connect(self._dm_auto_redraw)

        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidget(ctrl_inner)
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(270)
        # No maximum width: keep the splitter draggable in both directions.
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Canvas
        self._density_canvas_scroll: Any = None
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(0, 0, 0, 0)
        if _ensure_matplotlib() and Figure is not None and FigureCanvas is not None:
            mw = int(self._settings["max_w"])
            mh = int(self._settings["max_h"])
            self._density_figure = Figure(figsize=(mw / 100, mh / 100))
            self._density_canvas = FigureCanvas(self._density_figure)
            self._density_canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._density_canvas.setFixedSize(mw, mh)
            self._density_toolbar = NavigationToolbar(self._density_canvas, w)
            right_l.addWidget(self._density_toolbar)
            _cv_scroll = QScrollArea()
            _cv_scroll.setWidget(self._density_canvas)
            _cv_scroll.setWidgetResizable(False)
            _cv_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            _cv_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._density_canvas_scroll = _cv_scroll
            # Dynamic resize: canvas grows to fill the available viewport width
            self._density_resize_filter = _ViewportResizeFilter(
                self._sync_density_canvas_to_viewport, self
            )
            _cv_scroll.viewport().installEventFilter(self._density_resize_filter)
            right_l.addWidget(_cv_scroll, 1)
        else:
            right_l.addWidget(QLabel("Matplotlib is required for density maps."))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(ctrl_scroll)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1000])

        root_l = QVBoxLayout(w)
        root_l.setContentsMargins(0, 0, 0, 0)
        root_l.addWidget(splitter, 1)
        return w

    def _build_diff_panel(self) -> QWidget:
        """Build the Group Comparison (difference heatmap) panel."""
        w = QWidget()

        # ── Group / behavior selectors ────────────────────────────────
        self._diff_factor_combo = QComboBox()
        self._diff_factor_combo.setToolTip(
            "Select which grouping factor to use for labelling groups.\n"
            "Leave blank to use the top-level 'Group' assignment."
        )
        self._diff_factor_combo.addItem("(top-level group)", userData="")
        self._diff_factor_combo.currentIndexChanged.connect(self._on_diff_factor_changed)

        self._diff_group_a_combo = QComboBox()
        self._diff_group_a_combo.setToolTip("Reference group (shown in red on the difference map).")
        self._diff_group_b_combo = QComboBox()
        self._diff_group_b_combo.setToolTip("Comparison group (shown in blue on the difference map).")

        self._diff_behavior_combo = QComboBox()
        self._diff_behavior_combo.setToolTip("Behavior to compare between groups.")

        grp_sel = QGroupBox("Group Selection")
        grp_sel_f = QFormLayout(grp_sel)
        grp_sel_f.setContentsMargins(6, 4, 6, 4)
        grp_sel_f.setSpacing(4)
        grp_sel_f.addRow("Factor:", self._diff_factor_combo)
        grp_sel_f.addRow("Group A (ref):", self._diff_group_a_combo)
        grp_sel_f.addRow("Group B (comp):", self._diff_group_b_combo)
        grp_sel_f.addRow("Behavior:", self._diff_behavior_combo)

        # ── Comparison metric ─────────────────────────────────────────
        self._diff_metric_combo = QComboBox()
        self._diff_metric_combo.addItem("Signed difference  (A − B)", userData="signed")
        self._diff_metric_combo.addItem("Normalised diff  (A−B)/(A+B)", userData="normalized")
        self._diff_metric_combo.addItem("Log₂ ratio  log₂(A/B)", userData="log2ratio")
        self._diff_metric_combo.setToolTip(
            "How to compute the comparison between groups:\n"
            "• Signed: raw difference, values in [−1, +1] after peak-norm.\n"
            "• Normalised: (A−B)/(A+B+ε), insensitive to absolute density.\n"
            "• Log₂ ratio: highlights relative enrichment (0 = equal)."
        )
        idx_dm = self._diff_metric_combo.findData(self._settings["diff_metric"])
        if idx_dm >= 0:
            self._diff_metric_combo.setCurrentIndex(idx_dm)

        self._diff_mask_spin = QDoubleSpinBox()
        self._diff_mask_spin.setRange(0.0, 0.5)
        self._diff_mask_spin.setValue(float(self._settings["diff_mask_threshold"]))
        self._diff_mask_spin.setSingleStep(0.01)
        self._diff_mask_spin.setDecimals(3)
        self._diff_mask_spin.setToolTip(
            "Mask regions where both groups have density < this fraction of their\n"
            "respective peaks.  Prevents noise amplification in unvisited areas."
        )

        self._diff_show_contours = QCheckBox("Overlay group contours")
        self._diff_show_contours.setChecked(bool(self._settings["show_group_contours"]))
        self._diff_show_contours.setToolTip(
            "Draw contour lines for Group A (solid) and Group B (dashed)\n"
            "on top of the difference map to help interpret the comparison."
        )

        metric_grp = QGroupBox("Comparison Metric")
        metric_f = QFormLayout(metric_grp)
        metric_f.setContentsMargins(6, 4, 6, 4)
        metric_f.setSpacing(4)
        metric_f.addRow("Metric:", self._diff_metric_combo)
        metric_f.addRow("Mask threshold:", self._diff_mask_spin)
        metric_f.addRow(self._diff_show_contours)

        # ── Shared bandwidth / display ────────────────────────────────
        self._diff_auto_bw = QCheckBox("Auto-detect bandwidth")
        self._diff_auto_bw.setChecked(True)
        self._diff_auto_bw.toggled.connect(self._on_diff_auto_bw_toggled)

        self._diff_bw_multiplier = QDoubleSpinBox()
        self._diff_bw_multiplier.setRange(0.1, 8.0)
        self._diff_bw_multiplier.setValue(float(self._settings["bw_multiplier"]))
        self._diff_bw_multiplier.setSingleStep(0.1)
        self._diff_bw_multiplier.setDecimals(2)
        self._diff_bw_multiplier.setToolTip("Scale the auto-detected bandwidth by this multiplier.")

        self._diff_manual_bw = QDoubleSpinBox()
        self._diff_manual_bw.setRange(0.001, 1.0)
        self._diff_manual_bw.setValue(float(self._settings["manual_bw"]))
        self._diff_manual_bw.setSingleStep(0.005)
        self._diff_manual_bw.setDecimals(4)
        self._diff_manual_bw.setSuffix(" × diag")
        self._diff_manual_bw.setEnabled(False)

        self._diff_grid_res_combo = QComboBox()
        for res_label, res_val in [("128 (fast)", 128), ("256 (balanced)", 256),
                                    ("512 (high detail)", 512)]:
            self._diff_grid_res_combo.addItem(res_label, userData=res_val)
        self._diff_grid_res_combo.setCurrentIndex(1)

        self._diff_colormap_combo = QComboBox()
        self._diff_colormap_combo.setEditable(True)
        self._diff_colormap_combo.setToolTip(
            "Diverging colormap for the difference map.\n"
            "Select from the list or type any valid matplotlib colormap name."
        )
        for cm in ["RdBu_r", "coolwarm", "bwr", "seismic",
                   "PiYG", "PRGn", "RdYlBu", "RdYlGn", "Spectral_r",
                   "twilight_shifted"]:
            self._diff_colormap_combo.addItem(cm)
        idx_dcm = self._diff_colormap_combo.findText(str(self._settings["colormap_diff"]))
        if idx_dcm < 0:
            self._diff_colormap_combo.setCurrentText(str(self._settings["colormap_diff"]))
        else:
            self._diff_colormap_combo.setCurrentIndex(idx_dcm)

        self._diff_overlay_alpha = QSlider(Qt.Orientation.Horizontal)
        self._diff_overlay_alpha.setRange(10, 100)
        self._diff_overlay_alpha.setValue(90)
        self._diff_overlay_alpha_lbl = QLabel("90%")
        self._diff_overlay_alpha.valueChanged.connect(
            lambda v: self._diff_overlay_alpha_lbl.setText(f"{v}%")
        )

        self._diff_bg_mode_combo = QComboBox()
        self._diff_bg_mode_combo.addItem("Auto (from video)", userData="auto")
        self._diff_bg_mode_combo.addItem("None (dark)", userData="none")

        self._diff_bg_alpha = QSlider(Qt.Orientation.Horizontal)
        self._diff_bg_alpha.setRange(0, 100)
        self._diff_bg_alpha.setValue(int(self._settings["bg_alpha"]))
        self._diff_bg_alpha_lbl = QLabel(f"{self._settings['bg_alpha']}%")
        self._diff_bg_alpha.valueChanged.connect(
            lambda v: self._diff_bg_alpha_lbl.setText(f"{v}%")
        )

        self._diff_show_colorbar = QCheckBox("Show colorbar")
        self._diff_show_colorbar.setChecked(True)

        self._diff_show_count = QCheckBox("Show session count")
        self._diff_show_count.setChecked(True)

        diff_disp_grp = QGroupBox("Bandwidth / Display")
        diff_disp_f = QFormLayout(diff_disp_grp)
        diff_disp_f.setContentsMargins(6, 4, 6, 4)
        diff_disp_f.setSpacing(4)
        diff_disp_f.addRow(self._diff_auto_bw)
        diff_disp_f.addRow("Multiplier:", self._diff_bw_multiplier)
        diff_disp_f.addRow("Manual BW:", self._diff_manual_bw)
        diff_disp_f.addRow("Grid res:", self._diff_grid_res_combo)
        diff_disp_f.addRow("Colormap:", self._diff_colormap_combo)

        _diff_oa_row = QHBoxLayout()
        _diff_oa_row.addWidget(self._diff_overlay_alpha)
        _diff_oa_row.addWidget(self._diff_overlay_alpha_lbl)
        diff_disp_f.addRow("Overlay α:", _diff_oa_row)

        diff_bg_alpha_row = QHBoxLayout()
        diff_bg_alpha_row.addWidget(self._diff_bg_alpha)
        diff_bg_alpha_row.addWidget(self._diff_bg_alpha_lbl)
        diff_disp_f.addRow("BG source:", self._diff_bg_mode_combo)
        diff_disp_f.addRow("BG opacity:", diff_bg_alpha_row)
        diff_disp_f.addRow(self._diff_show_colorbar)
        diff_disp_f.addRow(self._diff_show_count)

        # Action buttons
        self._diff_generate_btn = QPushButton("Generate Comparison")
        self._diff_generate_btn.setToolTip("Compute density grids from raw data and render.")
        self._diff_generate_btn.clicked.connect(self._generate_diff)

        self._diff_export_btn = QPushButton("Export…")
        self._diff_export_btn.clicked.connect(lambda: self._export_figure(self._diff_figure))

        self._diff_settings_btn = QPushButton("Settings…")
        self._diff_settings_btn.clicked.connect(self._open_settings_dialog)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        btn_row.addWidget(self._diff_generate_btn)
        btn_row.addWidget(self._diff_export_btn)
        btn_row.addWidget(self._diff_settings_btn)
        btn_row.addStretch(1)

        self._diff_stale_lbl = QLabel("⚠ Data changed — click Generate to recompute.")
        self._diff_stale_lbl.setStyleSheet("color:#ffb300; font-size:9px; padding:1px 4px;")
        self._diff_stale_lbl.setVisible(False)

        # Assemble
        ctrl_inner = QWidget()
        ctrl_vbox = QVBoxLayout(ctrl_inner)
        ctrl_vbox.setContentsMargins(4, 4, 4, 4)
        ctrl_vbox.setSpacing(6)
        ctrl_vbox.addLayout(btn_row)
        ctrl_vbox.addWidget(self._diff_stale_lbl)
        ctrl_vbox.addWidget(grp_sel)
        ctrl_vbox.addWidget(metric_grp)
        ctrl_vbox.addWidget(diff_disp_grp)

        # Wire: data-changing controls → stale warning
        for _sig in [
            self._diff_group_a_combo.currentIndexChanged,
            self._diff_group_b_combo.currentIndexChanged,
            self._diff_behavior_combo.currentIndexChanged,
            self._diff_grid_res_combo.currentIndexChanged,
            self._diff_auto_bw.toggled,
            self._diff_bw_multiplier.valueChanged,
            self._diff_manual_bw.valueChanged,
        ]:
            _sig.connect(self._mark_diff_stale)
        self._diff_factor_combo.currentIndexChanged.connect(self._mark_diff_stale)

        # Wire: display-only controls → instant redraw from cache
        for _sig in [
            self._diff_metric_combo.currentIndexChanged,
            self._diff_mask_spin.valueChanged,
            self._diff_show_contours.toggled,
            self._diff_colormap_combo.currentTextChanged,
            self._diff_overlay_alpha.valueChanged,
            self._diff_bg_alpha.valueChanged,
            self._diff_bg_mode_combo.currentIndexChanged,
            self._diff_show_colorbar.toggled,
            self._diff_show_count.toggled,
        ]:
            _sig.connect(self._diff_auto_redraw)

        # Legend / help note
        legend_lbl = QLabel(
            "<span style='color:#ef5350;'>■</span> Red = Group A higher &nbsp;"
            "<span style='color:#42a5f5;'>■</span> Blue = Group B higher &nbsp;"
            "<span style='color:#9e9e9e;'>■</span> Masked / equal"
        )
        legend_lbl.setTextFormat(Qt.TextFormat.RichText)
        legend_lbl.setWordWrap(True)
        legend_lbl.setStyleSheet("font-size:10px; color:#90a4ae; padding: 2px 4px;")
        ctrl_vbox.addWidget(legend_lbl)
        ctrl_vbox.addStretch(1)

        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidget(ctrl_inner)
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(270)
        # No maximum width: keep the splitter draggable in both directions.
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Canvas
        self._diff_canvas_scroll: Any = None
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(0, 0, 0, 0)
        if _ensure_matplotlib() and Figure is not None and FigureCanvas is not None:
            mw = int(self._settings["max_w"])
            mh = int(self._settings["max_h"])
            self._diff_figure = Figure(figsize=(mw / 100, mh / 100))
            self._diff_canvas = FigureCanvas(self._diff_figure)
            self._diff_canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._diff_canvas.setFixedSize(mw, mh)
            self._diff_toolbar = NavigationToolbar(self._diff_canvas, w)
            right_l.addWidget(self._diff_toolbar)
            _cv_scroll = QScrollArea()
            _cv_scroll.setWidget(self._diff_canvas)
            _cv_scroll.setWidgetResizable(False)
            _cv_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            _cv_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._diff_canvas_scroll = _cv_scroll
            # Dynamic resize: canvas grows to fill the available viewport width
            self._diff_resize_filter = _ViewportResizeFilter(
                self._sync_diff_canvas_to_viewport, self
            )
            _cv_scroll.viewport().installEventFilter(self._diff_resize_filter)
            right_l.addWidget(_cv_scroll, 1)
        else:
            right_l.addWidget(QLabel("Matplotlib is required for density analysis."))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(ctrl_scroll)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1000])

        root_l = QVBoxLayout(w)
        root_l.setContentsMargins(0, 0, 0, 0)
        root_l.addWidget(splitter, 1)
        return w

    def _sync_density_canvas_to_viewport(self) -> None:
        """Fill the density canvas to its scroll viewport width, keeping aspect."""
        _autofill_canvas(
            self._density_canvas_scroll,
            getattr(self, "_density_canvas", None),
            getattr(self, "_density_figure", None),
            dpi=100,
        )

    def _sync_diff_canvas_to_viewport(self) -> None:
        """Fill the difference canvas to its scroll viewport width, keeping aspect."""
        _autofill_canvas(
            self._diff_canvas_scroll,
            getattr(self, "_diff_canvas", None),
            getattr(self, "_diff_figure", None),
            dpi=100,
        )

    def showEvent(self, event: Any) -> None:  # type: ignore[override]
        """Re-fill both canvases when this sub-tab becomes visible (the resize
        filter only fires on an actual viewport resize, so a figure drawn while
        hidden would otherwise stay small until the splitter is wiggled)."""
        super().showEvent(event)

        def _resync() -> None:
            self._sync_density_canvas_to_viewport()
            self._sync_diff_canvas_to_viewport()

        QTimer.singleShot(0, _resync)

    # ──────────────────────────────────────────────────────────────────
    # Public refresh (called by host when project is loaded / data reloaded)
    # ──────────────────────────────────────────────────────────────────

    def refresh_selectors(self) -> None:
        """Populate behavior and group drop-downs from the current project state."""
        host = self._host
        self._density_manifest_cache.clear()
        self._density_session_source_cache.clear()

        # Behavior combos ──────────────────────────────────────────────
        behaviors = [b for b in host._behaviors.behaviors
                     if str(b.behavior_id) != NO_BEHAVIOR_ID]

        for combo in (self._dm_behavior_combo, self._diff_behavior_combo):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All behaviors", userData=None)
            for b in behaviors:
                combo.addItem(str(b.name), userData=str(b.behavior_id))
            idx = combo.findText(prev)
            combo.setCurrentIndex(max(0, idx))
            combo.blockSignals(False)

        # Factor / group combos ────────────────────────────────────────
        factors = host._factor_definitions or []
        prev_factor = self._diff_factor_combo.currentData()
        self._diff_factor_combo.blockSignals(True)
        self._diff_factor_combo.clear()
        self._diff_factor_combo.addItem("(top-level group)", userData="")
        for f in factors:
            self._diff_factor_combo.addItem(f, userData=f)
        idx_f = self._diff_factor_combo.findData(prev_factor)
        self._diff_factor_combo.setCurrentIndex(max(0, idx_f))
        self._diff_factor_combo.blockSignals(False)

        self._refresh_group_combos()

        # Density map group selector ───────────────────────────────────
        groups = self._get_available_groups()
        prev_dm_grp = self._dm_specific_group_combo.currentText()
        self._dm_specific_group_combo.blockSignals(True)
        self._dm_specific_group_combo.clear()
        for g in sorted(groups):
            self._dm_specific_group_combo.addItem(g)
        idx_g = self._dm_specific_group_combo.findText(prev_dm_grp)
        if idx_g >= 0:
            self._dm_specific_group_combo.setCurrentIndex(idx_g)
        self._dm_specific_group_combo.blockSignals(False)

    def _refresh_group_combos(self) -> None:
        """Repopulate Group A / B combos from the selected factor."""
        groups = sorted(self._get_available_groups())
        for combo in (self._diff_group_a_combo, self._diff_group_b_combo):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            for g in groups:
                combo.addItem(g)
            idx = combo.findText(prev)
            combo.setCurrentIndex(max(0, idx))
            combo.blockSignals(False)
        # Ensure A and B default to different groups if possible
        if self._diff_group_a_combo.count() > 1 and (
            self._diff_group_a_combo.currentText() == self._diff_group_b_combo.currentText()
        ):
            self._diff_group_b_combo.setCurrentIndex(1)

    def _get_available_groups(self) -> list[str]:
        """Return sorted unique group labels for the currently selected factor."""
        host = self._host
        factor = self._diff_factor_combo.currentData() if hasattr(self, "_diff_factor_combo") else ""
        groups: set[str] = set()
        if factor:
            for _sid, factors in host._session_factors.items():
                g = factors.get(factor, "")
                if g:
                    groups.add(g)
        else:
            for g in host._session_groups.values():
                if g:
                    groups.add(g)
        if not groups:
            groups = {"(all sessions)"}
        return sorted(groups)

    # ──────────────────────────────────────────────────────────────────
    # Slot helpers
    # ──────────────────────────────────────────────────────────────────

    def _on_dm_group_mode_changed(self, _idx: int) -> None:
        mode = self._dm_group_mode_combo.currentData()
        self._dm_specific_group_combo.setVisible(mode == "specific")

    def _on_auto_bw_toggled(self, checked: bool) -> None:
        self._dm_manual_bw.setEnabled(not checked)

    def _on_diff_auto_bw_toggled(self, checked: bool) -> None:
        self._diff_manual_bw.setEnabled(not checked)

    def _on_diff_factor_changed(self, _idx: int) -> None:
        self._refresh_group_combos()

    def _mark_diff_stale(self) -> None:
        """Show stale warning when data-affecting diff settings change."""
        if hasattr(self, "_diff_stale_lbl"):
            self._diff_stale_lbl.setVisible(True)

    def _mark_dm_stale(self) -> None:
        """Show stale warning when data-affecting density settings change."""
        if hasattr(self, "_dm_stale_lbl"):
            self._dm_stale_lbl.setVisible(True)

    def _diff_auto_redraw(self) -> None:
        """Re-render the diff map from cached grids when only display settings changed."""
        if self._diff_cache and self._diff_figure is not None:
            QTimer.singleShot(60, self._render_diff)

    def _dm_auto_redraw(self) -> None:
        """Re-render the density map from cached grids when only display settings changed."""
        if self._dm_cache and self._density_figure is not None:
            QTimer.singleShot(60, self._render_density)

    # ──────────────────────────────────────────────────────────────────
    # Core density computation
    # ──────────────────────────────────────────────────────────────────

    def _split_namespaced_session_id(self, session_id: str) -> tuple[str | None, str]:
        sid = str(session_id).strip()
        if "::" not in sid:
            return None, sid
        tag, local_sid = sid.split("::", 1)
        return (tag or None), local_sid

    def _get_manifest_for_project(self, project_root: Path) -> Any:
        key = str(project_root.resolve())
        if key in self._density_manifest_cache:
            return self._density_manifest_cache[key]
        try:
            manifest = self._host._imports.load_manifest(project_root)
        except Exception:
            manifest = None
        self._density_manifest_cache[key] = manifest
        return manifest

    def _resolve_density_session_source(self, session_id: str) -> dict[str, Any] | None:
        sid = str(session_id).strip()
        if sid in self._density_session_source_cache:
            return self._density_session_source_cache[sid]

        tag, local_sid = self._split_namespaced_session_id(sid)
        project_root = self._host._project_root
        if project_root is None:
            self._density_session_source_cache[sid] = None
            return None
        if tag:
            entry = next((e for e in self._host._merge_service.entries if e.tag == tag), None)
            if entry is None:
                self._density_session_source_cache[sid] = None
                return None
            project_root = entry.project_root

        manifest = self._get_manifest_for_project(project_root)
        if manifest is None:
            self._density_session_source_cache[sid] = None
            return None

        linked_sessions = list(getattr(manifest, "linked_sessions", []) or [])
        session_obj = next(
            (s for s in linked_sessions if str(getattr(s, "session_id", "")) == local_sid),
            None,
        )
        if session_obj is None:
            self._density_session_source_cache[sid] = None
            return None

        pose_by_id = {getattr(p, "asset_id", ""): p for p in list(getattr(manifest, "poses", []) or [])}
        video_by_id = {getattr(v, "asset_id", ""): v for v in list(getattr(manifest, "videos", []) or [])}
        pose_asset = pose_by_id.get(str(getattr(session_obj, "pose_asset_id", "")))
        video_asset = video_by_id.get(str(getattr(session_obj, "video_asset_id", "")))

        def _choose_path(asset: Any) -> Path | None:
            if asset is None:
                return None
            local_path = str(getattr(asset, "local_path", "") or "").strip()
            if local_path:
                lp = Path(local_path)
                if lp.exists():
                    return lp
            source_path = str(getattr(asset, "source_path", "") or "").strip()
            if source_path:
                sp = Path(source_path)
                if sp.exists():
                    return sp
            return None

        out: dict[str, Any] = {
            "project_root": project_root,
            "pose_path": _choose_path(pose_asset),
            "video_path": _choose_path(video_asset),
            "video_w": int(getattr(video_asset, "width", 0) or 0) if video_asset is not None else 0,
            "video_h": int(getattr(video_asset, "height", 0) or 0) if video_asset is not None else 0,
        }
        self._density_session_source_cache[sid] = out
        return out

    def _get_density_pose_for_session(self, session_id: str) -> Any:
        sid = str(session_id)
        if sid in self._host._pose_cache:
            return self._host._pose_cache[sid]
        source = self._resolve_density_session_source(sid)
        pose_path = source.get("pose_path") if source else None
        if pose_path is None:
            return None
        try:
            pose = self._host._pose.load(pose_path)
        except Exception:
            return None
        self._host._pose_cache[sid] = pose
        return pose

    def _collect_xy_for_group(
        self,
        behavior_id: str | None,
        group_name: str | None,
        factor: str | None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return {behavior_id: (xs, ys)} pooled across all sessions in *group_name*.

        When *behavior_id* is None, return data for all behaviors.
        When *group_name* is None / empty, use all sessions.
        """
        host = self._host
        if host._project_root is None:
            return {}

        # Determine target session IDs
        target_sids: list[str] = []
        candidate_sids = list(host._session_label_by_session.keys())
        if not candidate_sids:
            manifest = host._imports.load_manifest(host._project_root)
            if manifest is not None:
                candidate_sids = [str(s.session_id) for s in manifest.linked_sessions]
        for sid in candidate_sids:
            if group_name and group_name not in ("(all sessions)", "(ungrouped)"):
                if factor:
                    label = host._session_label_by_session.get(sid, sid)
                    grp = host._session_factors.get(label, {}).get(factor, "")
                else:
                    label = host._session_label_by_session.get(sid, sid)
                    grp = host._session_groups.get(label, "")
                if grp != group_name:
                    continue
            target_sids.append(sid)

        if not target_sids:
            return {}

        # Load poses
        poses_by_sid: dict[str, Any] = {}
        for sid in target_sids:
            pose = self._get_density_pose_for_session(sid)
            if pose is not None:
                poses_by_sid[sid] = pose

        if not poses_by_sid:
            return {}

        # Determine which behaviors to collect
        behaviors = [b for b in host._behaviors.behaviors
                     if str(b.behavior_id) != NO_BEHAVIOR_ID]
        if behavior_id is not None:
            behaviors = [b for b in behaviors if str(b.behavior_id) == behavior_id]
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for b in behaviors:
            bid = str(b.behavior_id)
            xs_all: list[np.ndarray] = []
            ys_all: list[np.ndarray] = []

            bdf: pd.DataFrame | None = None
            raw_df = host._raw_bouts.get(bid)
            if raw_df is not None and not raw_df.empty and "session_id" in raw_df.columns:
                try:
                    bdf = raw_df[raw_df["session_id"].astype(str).isin(target_sids)].copy()
                    if not bdf.empty:
                        bdf = host._apply_prechop_to_bout_df(bdf, rebase=False)
                        if bdf.empty:
                            bdf = None
                    else:
                        bdf = None
                except Exception:
                    bdf = None

            if bdf is None or bdf.empty:
                continue

            for sid_val, grp in bdf.groupby("session_id", sort=False):
                p = poses_by_sid.get(str(sid_val))
                if p is None:
                    continue
                n_frames = p.n_frames
                starts = np.clip(grp["start_frame"].to_numpy(np.int64), 0, n_frames - 1)
                ends = np.clip(grp["end_frame"].to_numpy(np.int64), 0, n_frames - 1)
                ends = np.maximum(starts, ends)
                for s, e in zip(starts, ends):
                    xs_all.append(p.centroid_x[s:e + 1])
                    ys_all.append(p.centroid_y[s:e + 1])

            if xs_all:
                result[bid] = (np.concatenate(xs_all), np.concatenate(ys_all))

        return result

    def _build_density_grid(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        video_w: int,
        video_h: int,
        grid_res: int,
        sigma: float,
        normalize: str,
        threshold_pct: int,
    ) -> np.ndarray:
        """Compute a normalised 2-D density grid using histogram2d + Gaussian blur.

        Returns a float64 array of shape (grid_res_h, grid_res_w) with values
        in [0, 1], or zero-filled on failure.
        """
        from scipy.ndimage import gaussian_filter

        try:
            import cupy as cp
            from cupyx.scipy.ndimage import gaussian_filter as _gpu_gf
            _USE_GPU = True
        except Exception:
            cp = None
            _USE_GPU = False

        grid_w = grid_res
        grid_h = max(1, int(round(grid_res * video_h / max(video_w, 1))))

        H, _, _ = np.histogram2d(
            np.clip(xs, 0, video_w),
            np.clip(ys, 0, video_h),
            bins=[grid_w, grid_h],
            range=[[0, video_w], [0, video_h]],
        )
        H = H.T.astype(np.float32)   # (grid_h, grid_w) to match image convention

        if _USE_GPU and cp is not None:
            H_gpu = cp.asarray(H)
            Z = cp.asnumpy(_gpu_gf(H_gpu, sigma=sigma)).astype(np.float64)
        else:
            Z = gaussian_filter(H, sigma=sigma).astype(np.float64)

        # Low-percentile threshold
        if threshold_pct > 0 and Z.max() > 0:
            thresh_val = np.percentile(Z[Z > 0], threshold_pct) if (Z > 0).any() else 0.0
            Z = np.clip(Z - thresh_val, 0.0, None)

        z_max = float(Z.max())
        if z_max <= 0:
            return np.zeros_like(Z)

        if normalize == "peak":
            Z = Z / z_max
        else:  # "sum"
            z_sum = float(Z.sum())
            Z = Z / z_sum if z_sum > 0 else Z

        return Z

    def _get_video_dims(self, session_ids: list[str]) -> tuple[int, int]:
        """Return (width, height) of the first available video in *session_ids*."""
        if not session_ids:
            return (640, 480)
        for sid in session_ids:
            source = self._resolve_density_session_source(sid)
            if not source:
                continue
            w = int(source.get("video_w") or 0)
            h = int(source.get("video_h") or 0)
            if w > 0 and h > 0:
                return (w, h)
            vp = source.get("video_path")
            if vp is not None and vp.exists() and _ensure_cv2():
                cap = cv2.VideoCapture(str(vp))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                if w > 0 and h > 0:
                    return (w, h)
        return (640, 480)

    def _load_bg_image(
        self, session_ids: list[str], bg_mode: str,
        video_w: int, video_h: int,
    ) -> np.ndarray | None:
        """Composite background from up to 8 sessions.  Returns RGB uint8 or None."""
        if bg_mode == "none" or not _ensure_cv2():
            return None
        frames: list[np.ndarray] = []
        for sid in session_ids[:8]:
            source = self._resolve_density_session_source(sid)
            vp = source.get("video_path") if source else None
            if vp is None or not vp.exists():
                continue
            try:
                cap = cv2.VideoCapture(str(vp))
                ret, bgr = cap.read()
                cap.release()
                if ret and bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (video_w, video_h), interpolation=cv2.INTER_AREA)
                    frames.append(rgb.astype("float32"))
            except Exception:
                pass
        if not frames:
            return None
        return np.mean(frames, axis=0).astype("uint8")

    def _estimate_subject_size(self, session_ids: list[str]) -> float | None:
        """Return the estimated subject keypoint span in pixels."""
        for sid in session_ids[:3]:
            pd_obj = self._get_density_pose_for_session(sid)
            if pd_obj is None:
                continue
            try:
                span = _estimate_subject_span(pd_obj)
                if span is not None and span > 4:
                    return span
            except Exception:
                pass
        return None

    def _get_session_ids_for_group(self, group_name: str | None, factor: str | None) -> list[str]:
        """Return session IDs matching *group_name* for *factor*, restricted to
        sessions checked in the Summary & Statistics tab."""
        host = self._host
        session_ids = list(host._session_label_by_session.keys())
        if not session_ids and host._project_root is not None:
            manifest = host._imports.load_manifest(host._project_root)
            if manifest is not None:
                session_ids = [str(s.session_id) for s in manifest.linked_sessions]
        checked_sessions = host._summary_tab._checked_subjects()
        result: list[str] = []
        for sid in session_ids:
            label = host._session_label_by_session.get(sid, sid)
            if checked_sessions and label not in checked_sessions:
                continue
            if group_name and group_name not in ("(all sessions)", "(ungrouped)"):
                if factor:
                    grp = host._session_factors.get(label, {}).get(factor, "")
                else:
                    grp = host._session_groups.get(label, "")
                if grp != group_name:
                    continue
            result.append(sid)
        return result

    # ──────────────────────────────────────────────────────────────────
    # Density Maps generation
    # ──────────────────────────────────────────────────────────────────

    def _generate_density(self) -> None:
        if self._host._project_root is None:
            QMessageBox.warning(self, "Density Map", "Open a project first.")
            return
        if not _ensure_matplotlib() or self._density_figure is None:
            QMessageBox.warning(self, "Density Map", "Matplotlib is required.")
            return
        self._host._status.setText("Computing density maps — please wait…")
        QTimer.singleShot(30, self._render_density)

    def _render_density(self) -> None:
        try:
            self._render_density_inner()
        except Exception as exc:
            logger.exception("Density map render error: %s", exc)
            self._host._status.setText(f"Density map error: {exc}")

    def _render_density_inner(self) -> None:
        import matplotlib.pyplot as _plt

        host = self._host
        fig = self._density_figure
        fig.clear()

        behavior_id   = self._dm_behavior_combo.currentData()
        mode          = self._dm_group_mode_combo.currentData()
        grid_res      = int(self._dm_grid_res_combo.currentData() or 256)
        normalize     = str(self._dm_normalize_combo.currentData() or "peak")
        threshold_pct = int(self._dm_threshold_spin.value())
        auto_bw       = bool(self._dm_auto_bw.isChecked())
        bw_mult       = round(float(self._dm_bw_multiplier.value()), 3)
        manual_bw     = round(float(self._dm_manual_bw.value()), 5)
        # Display-only settings (no recompute needed)
        overlay_alpha = self._dm_overlay_alpha.value() / 100.0
        bg_mode       = str(self._dm_bg_mode_combo.currentData() or "auto")
        bg_alpha      = self._dm_bg_alpha.value() / 100.0
        show_colorbar = self._dm_show_colorbar.isChecked()
        show_count    = self._dm_show_count.isChecked()
        colormap      = str(self._dm_colormap_combo.currentText())

        factor = self._diff_factor_combo.currentData() if hasattr(self, "_diff_factor_combo") else ""
        if mode == "all":
            groups_to_render: list[tuple[str | None, str]] = [(None, "All sessions")]
        elif mode == "specific":
            g = self._dm_specific_group_combo.currentText().strip()
            groups_to_render = [(g if g else None, g or "All sessions")]
        else:  # per_group
            groups_to_render = [(g, g) for g in sorted(self._get_available_groups())]

        behaviors = [b for b in host._behaviors.behaviors
                     if str(b.behavior_id) != NO_BEHAVIOR_ID]
        if behavior_id is not None:
            behaviors = [b for b in behaviors if str(b.behavior_id) == behavior_id]
        if not behaviors:
            self._host._status.setText("No behaviors selected.")
            return

        n_beh  = len(behaviors)
        n_grp  = len(groups_to_render)
        n_cols = n_grp
        n_rows = n_beh

        # ── Cache check ───────────────────────────────────────────────
        grp_repr = tuple(sorted(g for g, _ in groups_to_render if g is not None))
        data_key = (behavior_id or "", mode, grp_repr, grid_res,
                    auto_bw, bw_mult, manual_bw, normalize, threshold_pct)
        use_cache = (bool(self._dm_cache)
                     and self._dm_cache_meta.get("key") == data_key)

        if use_cache:
            video_w  = self._dm_cache_meta["video_w"]
            video_h  = self._dm_cache_meta["video_h"]
            bg_image = self._dm_cache_meta.get("bg_image")
        else:
            first_sids   = self._get_session_ids_for_group(groups_to_render[0][0], factor or None)
            video_w, video_h = self._get_video_dims(first_sids or [])
            bg_image     = self._load_bg_image(first_sids[:3], "auto", video_w, video_h) if first_sids else None
            subject_span = self._estimate_subject_size(first_sids[:3]) if first_sids else None

            if auto_bw:
                sigma = _compute_auto_bandwidth(video_w, video_h, grid_res,
                                                subject_span_px=subject_span,
                                                multiplier=bw_mult)
            else:
                diag  = _math.sqrt(video_w ** 2 + video_h ** 2)
                sigma = max(1.0, manual_bw * diag * grid_res / max(video_w, video_h))

            grid_h_loc = max(1, int(round(grid_res * video_h / max(video_w, 1))))
            new_cache: dict[str, Any] = {}
            for b in behaviors:
                bid = str(b.behavior_id)
                for (grp_name, grp_label) in groups_to_render:
                    xy_data       = self._collect_xy_for_group(bid, grp_name, factor or None)
                    xs_arr, ys_arr = xy_data.get(bid, (np.empty(0), np.empty(0)))
                    n_pts  = xs_arr.size
                    n_sess = len(self._get_session_ids_for_group(grp_name, factor or None))
                    Z = (self._build_density_grid(xs_arr, ys_arr, video_w, video_h,
                                                  grid_res, sigma, normalize, threshold_pct)
                         if n_pts >= 10 else np.zeros((grid_h_loc, grid_res)))
                    new_cache[(bid, grp_label)] = {"Z": Z, "n_pts": n_pts, "n_sess": n_sess}

            self._dm_cache      = new_cache
            self._dm_cache_meta = {
                "key": data_key,
                "video_w": video_w, "video_h": video_h,
                "bg_image": bg_image,
            }
            if hasattr(self, "_dm_stale_lbl"):
                self._dm_stale_lbl.setVisible(False)

        # ── Figure layout ─────────────────────────────────────────────
        aspect = video_h / max(video_w, 1)
        MAX_TOTAL_W = 13.0
        cell_w = max(min(MAX_TOTAL_W / max(n_cols, 1) - 0.25, 5.0), 3.0)
        cell_h = cell_w * aspect
        fig.set_size_inches(n_cols * cell_w + 1.0, n_rows * cell_h + 0.8)
        fig.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")

        try:
            cmap_obj = _plt.get_cmap(colormap)
        except Exception:
            cmap_obj = _plt.get_cmap("inferno")

        status_parts: list[str] = []
        grid_h_fb = max(1, int(round(grid_res * video_h / max(video_w, 1))))

        for r_idx, b in enumerate(behaviors):
            bid    = str(b.behavior_id)
            b_name = str(b.name)

            for c_idx, (grp_name, grp_label) in enumerate(groups_to_render):
                cached_c = self._dm_cache.get((bid, grp_label), {})
                Z      = cached_c.get("Z", np.zeros((grid_h_fb, grid_res)))
                n_pts  = cached_c.get("n_pts", 0)
                n_sess = cached_c.get("n_sess", 0)

                ax = fig.add_subplot(n_rows, n_cols, r_idx * n_cols + c_idx + 1)
                ax.set_facecolor("#121212")

                if bg_image is not None and bg_alpha > 0 and bg_mode != "none":
                    ax.imshow(bg_image, aspect="auto", alpha=bg_alpha,
                              extent=[0, video_w, video_h, 0], origin="upper")

                if n_pts >= 10:
                    rgba = cmap_obj(Z)
                    rgba[..., 3] = (Z ** 0.45) * overlay_alpha
                    ax.imshow(rgba, extent=[0, video_w, video_h, 0],
                              origin="upper", aspect="auto",
                              interpolation="bilinear", zorder=2)
                    if show_colorbar:
                        import matplotlib.cm as _cm
                        sm = _cm.ScalarMappable(cmap=cmap_obj, norm=_plt.Normalize(0, 1))
                        sm.set_array([])
                        cb = fig.colorbar(sm, ax=ax, label="Density", shrink=0.7, pad=0.02)
                        cb.ax.yaxis.label.set_color("#cfd8dc")
                        cb.ax.tick_params(colors="#cfd8dc", labelsize=7)
                        cb.outline.set_edgecolor("#37474f")
                    status_parts.append(f"{b_name}/{grp_label}: {n_pts:,} pts")
                else:
                    ax.text(0.5, 0.5, "Insufficient data\n(< 10 points)",
                            transform=ax.transAxes, ha="center", va="center",
                            color="#607d8b", fontsize=9)

                ax.set_xlim(0, video_w)
                ax.set_ylim(video_h, 0)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(length=0)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#37474f")
                    spine.set_linewidth(0.5)

                title_parts = [b_name]
                if n_grp > 1:
                    title_parts.append(grp_label)
                if show_count and n_pts > 0:
                    title_parts.append(f"n={n_sess}")
                ax.set_title("  |  ".join(title_parts), fontsize=8, color="#cfd8dc", pad=3)

        try:
            fig.tight_layout(pad=0.8, h_pad=0.6, w_pad=0.5)
        except Exception:
            pass
        _dpi = fig.get_dpi() or 100
        _fig_px_w = max(200, int(fig.get_figwidth() * _dpi))
        _fig_px_h = max(150, int(fig.get_figheight() * _dpi))
        self._density_canvas.setFixedSize(_fig_px_w, _fig_px_h)
        self._sync_density_canvas_to_viewport()
        self._density_canvas.draw_idle()
        self._host._status.setText(
            "Density maps rendered.  " + "  |  ".join(status_parts) if status_parts
            else "Density maps rendered (no data)."
        )

    # ──────────────────────────────────────────────────────────────────
    # Group Comparison (difference heatmap) generation
    # ──────────────────────────────────────────────────────────────────

    def _generate_diff(self) -> None:
        if self._host._project_root is None:
            QMessageBox.warning(self, "Group Comparison", "Open a project first.")
            return
        if not _ensure_matplotlib() or self._diff_figure is None:
            QMessageBox.warning(self, "Group Comparison", "Matplotlib is required.")
            return
        grp_a = self._diff_group_a_combo.currentText().strip()
        grp_b = self._diff_group_b_combo.currentText().strip()
        if not grp_a or not grp_b:
            QMessageBox.warning(self, "Group Comparison", "Select both Group A and Group B.")
            return
        if grp_a == grp_b:
            QMessageBox.warning(self, "Group Comparison",
                                "Group A and Group B must be different.")
            return
        self._host._status.setText("Computing group-comparison density map — please wait…")
        QTimer.singleShot(30, self._render_diff)

    def _render_diff(self) -> None:
        try:
            self._render_diff_inner()
        except Exception as exc:
            logger.exception("Group comparison render error: %s", exc)
            self._host._status.setText(f"Group comparison error: {exc}")

    def _render_diff_inner(self) -> None:
        import matplotlib.pyplot as _plt
        import matplotlib.colors as _mcolors

        host = self._host
        fig = self._diff_figure
        fig.clear()

        grp_a         = self._diff_group_a_combo.currentText().strip()
        grp_b         = self._diff_group_b_combo.currentText().strip()
        factor        = str(self._diff_factor_combo.currentData() or "")
        behavior_id   = self._diff_behavior_combo.currentData()
        grid_res      = int(self._diff_grid_res_combo.currentData() or 256)
        auto_bw       = bool(self._diff_auto_bw.isChecked())
        bw_mult       = round(float(self._diff_bw_multiplier.value()), 3)
        manual_bw     = round(float(self._diff_manual_bw.value()), 5)
        # Display-only (no recompute needed)
        metric        = str(self._diff_metric_combo.currentData() or "signed")
        mask_threshold = float(self._diff_mask_spin.value())
        show_contours = self._diff_show_contours.isChecked()
        colormap      = str(self._diff_colormap_combo.currentText())
        bg_mode       = str(self._diff_bg_mode_combo.currentData() or "auto")
        bg_alpha      = self._diff_bg_alpha.value() / 100.0
        overlay_alpha = self._diff_overlay_alpha.value() / 100.0
        show_colorbar = self._diff_show_colorbar.isChecked()
        show_count    = self._diff_show_count.isChecked()

        behaviors = [b for b in host._behaviors.behaviors
                     if str(b.behavior_id) != NO_BEHAVIOR_ID]
        if behavior_id is not None:
            behaviors = [b for b in behaviors if str(b.behavior_id) == behavior_id]
        if not behaviors:
            self._host._status.setText("No behaviors selected.")
            return

        # ── Cache check ───────────────────────────────────────────────
        # Key covers every setting that affects the raw density grids.
        # Display-only settings (metric, colormap, alpha…) are NOT in the key.
        data_key = (grp_a, grp_b, factor, behavior_id or "",
                    grid_res, auto_bw, bw_mult, manual_bw)
        use_cache = (bool(self._diff_cache)
                     and self._diff_cache_meta.get("key") == data_key)

        if use_cache:
            video_w  = self._diff_cache_meta["video_w"]
            video_h  = self._diff_cache_meta["video_h"]
            bg_image = self._diff_cache_meta.get("bg_image")
            sids_a   = self._diff_cache_meta.get("sids_a", [])
            sids_b   = self._diff_cache_meta.get("sids_b", [])
        else:
            # Full computation: read parquets + build Gaussian grids
            sids_a   = self._get_session_ids_for_group(grp_a, factor or None)
            sids_b   = self._get_session_ids_for_group(grp_b, factor or None)
            all_sids = list(dict.fromkeys(sids_a + sids_b))
            video_w, video_h = self._get_video_dims(all_sids)
            # Always load background so toggling bg_mode later works instantly
            bg_image = self._load_bg_image(all_sids[:3], "auto", video_w, video_h) if all_sids else None
            subject_span = self._estimate_subject_size(all_sids[:3]) if all_sids else None
            # Sigma is identical for all behaviors (arena / subject-size based)
            if auto_bw:
                sigma = _compute_auto_bandwidth(
                    video_w, video_h, grid_res,
                    subject_span_px=subject_span,
                    multiplier=bw_mult,
                )
            else:
                diag  = _math.sqrt(video_w ** 2 + video_h ** 2)
                sigma = max(1.0, manual_bw * diag * grid_res / max(video_w, video_h))

            grid_h = max(1, int(round(grid_res * video_h / max(video_w, 1))))
            new_cache: dict[str, Any] = {}
            for b in behaviors:
                bid = str(b.behavior_id)
                xy_a = self._collect_xy_for_group(bid, grp_a, factor or None)
                xy_b = self._collect_xy_for_group(bid, grp_b, factor or None)
                xs_a, ys_a = xy_a.get(bid, (np.empty(0), np.empty(0)))
                xs_b, ys_b = xy_b.get(bid, (np.empty(0), np.empty(0)))
                n_a, n_b   = xs_a.size, xs_b.size
                Z_a = (self._build_density_grid(xs_a, ys_a, video_w, video_h, grid_res, sigma, "peak", 2)
                       if n_a >= 10 else np.zeros((grid_h, grid_res)))
                Z_b = (self._build_density_grid(xs_b, ys_b, video_w, video_h, grid_res, sigma, "peak", 2)
                       if n_b >= 10 else np.zeros_like(Z_a))
                new_cache[bid] = {"Z_a": Z_a, "Z_b": Z_b, "n_a": n_a, "n_b": n_b}

            self._diff_cache      = new_cache
            self._diff_cache_meta = {
                "key": data_key,
                "video_w": video_w, "video_h": video_h,
                "bg_image": bg_image,
                "sids_a": sids_a, "sids_b": sids_b,
            }
            if hasattr(self, "_diff_stale_lbl"):
                self._diff_stale_lbl.setVisible(False)

        # ── Figure layout ─────────────────────────────────────────────
        n_beh = len(behaviors)
        aspect = video_h / max(video_w, 1)
        MAX_TOTAL_W = 13.0
        cell_w = max(min(MAX_TOTAL_W / max(n_beh, 1) - 0.3, 5.0), 3.0)
        cell_h = cell_w * aspect
        fig.set_size_inches(cell_w * n_beh + 1.5, cell_h + 1.2)
        fig.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")

        cmap_obj = _make_diverging_transparent_center_cmap(colormap) or _plt.get_cmap(colormap)
        status_parts: list[str] = []
        grid_h_fb = max(1, int(round(grid_res * video_h / max(video_w, 1))))

        for col_idx, b in enumerate(behaviors):
            bid    = str(b.behavior_id)
            b_name = str(b.name)
            cached_b = self._diff_cache.get(bid, {})
            Z_a  = cached_b.get("Z_a", np.zeros((grid_h_fb, grid_res)))
            Z_b  = cached_b.get("Z_b", np.zeros_like(Z_a))
            n_a  = cached_b.get("n_a", 0)
            n_b  = cached_b.get("n_b", 0)

            ax = fig.add_subplot(1, n_beh, col_idx + 1)
            ax.set_facecolor("#121212")

            if bg_image is not None and bg_alpha > 0 and bg_mode != "none":
                ax.imshow(bg_image, aspect="auto", alpha=bg_alpha,
                          extent=[0, video_w, video_h, 0], origin="upper")

            if n_a < 10 and n_b < 10:
                ax.text(0.5, 0.5, "Insufficient data\nfor both groups",
                        transform=ax.transAxes, ha="center", va="center",
                        color="#607d8b", fontsize=9)
            else:
                eps = 1e-9
                if metric == "signed":
                    Z_diff     = Z_a - Z_b
                    cbar_label = f"{grp_a} − {grp_b}  (peak-norm)"
                elif metric == "normalized":
                    Z_diff     = (Z_a - Z_b) / (Z_a + Z_b + eps)
                    cbar_label = f"(A−B)/(A+B)  [{grp_a} vs {grp_b}]"
                else:  # log2ratio
                    Z_diff     = np.log2((Z_a + eps) / (Z_b + eps))
                    cbar_label = f"log₂({grp_a}/{grp_b})"

                mask = (Z_a < mask_threshold) & (Z_b < mask_threshold)

                visible_diff = Z_diff[~mask]
                if visible_diff.size > 0 and np.abs(visible_diff).max() > eps:
                    peak = float(np.abs(visible_diff).max())
                    if metric == "log2ratio":
                        peak = min(peak, 3.0)
                    vmin, vmax = -peak, peak
                else:
                    vmin, vmax = -1.0, 1.0
                cbar_label += f"  [±{abs(vmax):.2f}]"

                norm_obj = _mcolors.Normalize(vmin=vmin, vmax=vmax)
                rgba = cmap_obj(norm_obj(Z_diff)).astype(np.float32)
                rgba[mask, 3] = 0.0
                if overlay_alpha < 1.0:
                    rgba[~mask, 3] *= overlay_alpha

                ax.imshow(
                    rgba,
                    extent=[0, video_w, video_h, 0],
                    origin="upper",
                    aspect="auto",
                    interpolation="bilinear",
                    zorder=2,
                )

                if show_contours:
                    contour_y = np.linspace(0, video_h, Z_a.shape[0])
                    contour_x = np.linspace(0, video_w, Z_a.shape[1])
                    if n_a >= 10 and Z_a.max() > 0:
                        ax.contour(contour_x, contour_y, Z_a,
                                   levels=[0.25, 0.5, 0.75],
                                   colors=["#ef9a9a", "#ef5350", "#b71c1c"],
                                   linewidths=[0.7, 1.0, 1.2], linestyles="solid", zorder=3)
                    if n_b >= 10 and Z_b.max() > 0:
                        ax.contour(contour_x, contour_y, Z_b,
                                   levels=[0.25, 0.5, 0.75],
                                   colors=["#90caf9", "#42a5f5", "#0d47a1"],
                                   linewidths=[0.7, 1.0, 1.2], linestyles="dashed", zorder=3)

                if show_colorbar:
                    sm = _plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
                    sm.set_array([])
                    cb = fig.colorbar(sm, ax=ax, label=cbar_label, shrink=0.8, pad=0.02)
                    cb.ax.yaxis.label.set_color("#cfd8dc")
                    cb.ax.tick_params(colors="#cfd8dc", labelsize=7)
                    cb.outline.set_edgecolor("#37474f")

                status_parts.append(
                    f"{b_name}: A={n_a:,} pts ({len(sids_a)} sess), "
                    f"B={n_b:,} pts ({len(sids_b)} sess)"
                )

            ax.set_xlim(0, video_w)
            ax.set_ylim(video_h, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_edgecolor("#37474f")
                spine.set_linewidth(0.5)

            title_parts = [b_name]
            if show_count:
                title_parts.append(f"{grp_a} (n={len(sids_a)}) vs {grp_b} (n={len(sids_b)})")
            ax.set_title("  |  ".join(title_parts), fontsize=8, color="#cfd8dc", pad=3)

        try:
            fig.tight_layout(pad=0.8, h_pad=0.5, w_pad=0.6)
        except Exception:
            pass
        _dpi = fig.get_dpi() or 100
        _fig_px_w = max(200, int(fig.get_figwidth() * _dpi))
        _fig_px_h = max(150, int(fig.get_figheight() * _dpi))
        self._diff_canvas.setFixedSize(_fig_px_w, _fig_px_h)
        self._sync_diff_canvas_to_viewport()
        self._diff_canvas.draw_idle()
        self._host._status.setText(
            "Group comparison rendered.  " + "  |  ".join(status_parts)
            if status_parts else
            "Group comparison rendered (no data for selected behaviors/groups)."
        )

    # ──────────────────────────────────────────────────────────────────
    # Settings dialog
    # ──────────────────────────────────────────────────────────────────

    def _open_settings_dialog(self) -> None:
        """Configurable settings for the density analysis panel."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Density Analysis Settings")
        dlg.resize(480, 580)

        def _desc(txt: str) -> QLabel:
            lbl = QLabel(f"<i>{txt}</i>")
            lbl.setStyleSheet("color:#78909c; font-size:9px; padding-left:2px;")
            lbl.setWordWrap(True)
            return lbl

        # ── Canvas & Export ──────────────────────────────────────────
        max_w_spin = QSpinBox(dlg)
        max_w_spin.setRange(200, 3840)
        max_w_spin.setSingleStep(50)
        max_w_spin.setSuffix(" px")
        max_w_spin.setValue(int(self._settings["max_w"]))

        max_h_spin = QSpinBox(dlg)
        max_h_spin.setRange(150, 2160)
        max_h_spin.setSingleStep(50)
        max_h_spin.setSuffix(" px")
        max_h_spin.setValue(int(self._settings["max_h"]))

        export_dpi_spin = QSpinBox(dlg)
        export_dpi_spin.setRange(72, 600)
        export_dpi_spin.setSingleStep(50)
        export_dpi_spin.setSuffix(" dpi")
        export_dpi_spin.setValue(int(self._settings["export_dpi"]))

        canvas_grp = QGroupBox("Canvas & Export")
        canvas_f = QFormLayout(canvas_grp)
        canvas_f.setSpacing(5)
        canvas_f.addRow("Initial width:", max_w_spin)
        canvas_f.addRow("Initial height:", max_h_spin)
        canvas_f.addRow(_desc(
            "Starting canvas size in pixels.  The canvas is automatically resized "
            "after plotting to match the rendered figure — these values only affect "
            "the blank canvas shown before the first plot."
        ))
        canvas_f.addRow("Export DPI:", export_dpi_spin)
        canvas_f.addRow(_desc(
            "Resolution when saving via Export.  "
            "72 = screen quality  ·  150 = good print  ·  300 = publication."
        ))

        # ── Bandwidth ────────────────────────────────────────────────
        default_bw_spin = QDoubleSpinBox(dlg)
        default_bw_spin.setRange(0.1, 8.0)
        default_bw_spin.setSingleStep(0.1)
        default_bw_spin.setDecimals(2)
        default_bw_spin.setValue(float(self._settings["bw_multiplier"]))

        default_manual_bw = QDoubleSpinBox(dlg)
        default_manual_bw.setRange(0.001, 1.0)
        default_manual_bw.setSingleStep(0.005)
        default_manual_bw.setDecimals(4)
        default_manual_bw.setSuffix(" × diag")
        default_manual_bw.setValue(float(self._settings["manual_bw"]))

        bw_grp = QGroupBox("Bandwidth (Gaussian Spread)")
        bw_f = QFormLayout(bw_grp)
        bw_f.setSpacing(5)
        bw_f.addRow("Auto BW multiplier:", default_bw_spin)
        bw_f.addRow(_desc(
            "Scales the automatically detected spread (sigma) by this factor.  "
            "1.0 = use measured subject body size directly.  "
            "< 1.0 = sharper, tighter hotspots.  > 1.0 = smoother, broader fields.  "
            "Try 0.5–2.0 to adjust without disabling auto mode."
        ))
        bw_f.addRow("Manual bandwidth:", default_manual_bw)
        bw_f.addRow(_desc(
            "Used when 'Auto-detect' is turned off.  "
            "Expressed as a fraction of the arena diagonal.  "
            "0.03 = 3% of diagonal ≈ one animal body length for typical arenas."
        ))

        # ── Difference map ───────────────────────────────────────────
        mask_spin = QDoubleSpinBox(dlg)
        mask_spin.setRange(0.0, 0.5)
        mask_spin.setSingleStep(0.01)
        mask_spin.setDecimals(3)
        mask_spin.setValue(float(self._settings["diff_mask_threshold"]))

        diff_grp = QGroupBox("Group Comparison Map")
        diff_f = QFormLayout(diff_grp)
        diff_f.setSpacing(5)
        diff_f.addRow("Mask threshold:", mask_spin)
        diff_f.addRow(_desc(
            "Pixels where BOTH groups have density below this fraction of their peak "
            "are hidden (shown as transparent).  "
            "Increase (e.g. 0.10) to hide low-traffic background noise.  "
            "Decrease (e.g. 0.01) to reveal subtle differences in rarely visited areas."
        ))

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        root_l = QVBoxLayout(dlg)
        root_l.setSpacing(8)
        root_l.setContentsMargins(10, 10, 10, 10)
        root_l.addWidget(canvas_grp)
        root_l.addWidget(bw_grp)
        root_l.addWidget(diff_grp)
        root_l.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._settings["max_w"] = max_w_spin.value()
        self._settings["max_h"] = max_h_spin.value()
        self._settings["export_dpi"] = export_dpi_spin.value()
        self._settings["bw_multiplier"] = default_bw_spin.value()
        self._settings["manual_bw"] = default_manual_bw.value()
        self._settings["diff_mask_threshold"] = mask_spin.value()

        # Apply to controls
        self._dm_bw_multiplier.setValue(float(self._settings["bw_multiplier"]))
        self._dm_manual_bw.setValue(float(self._settings["manual_bw"]))
        self._diff_bw_multiplier.setValue(float(self._settings["bw_multiplier"]))
        self._diff_manual_bw.setValue(float(self._settings["manual_bw"]))
        self._diff_mask_spin.setValue(float(self._settings["diff_mask_threshold"]))

    # ──────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────

    def _export_figure(self, figure: Any) -> None:
        if figure is None:
            QMessageBox.information(self, "Export", "No figure to export.")
            return
        dpi = int(self._settings.get("export_dpi", 200))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Density Analysis", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All Files (*)",
        )
        if not path:
            return
        try:
            figure.savefig(path, dpi=dpi, bbox_inches="tight",
                           facecolor=figure.get_facecolor())
            self._host._status.setText(f"Exported to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))


# ======================================================================
# Sub-tab 5: Behavioral Motif Discovery
# ======================================================================

# Deferred imports for motif analysis libraries
_HMM_OK: bool | None = None


def _ensure_hmmlearn() -> bool:
    global _HMM_OK  # noqa: PLW0603
    if _HMM_OK is not None:
        return _HMM_OK
    try:
        import hmmlearn  # noqa: F401
        _HMM_OK = True
    except ImportError:
        _HMM_OK = False
    return _HMM_OK


class _BehaviorMotifWidget(QWidget):
    """Three-panel behavioral sequence analysis: transition matrices,
    N-gram / cluster motif discovery, and Hidden Markov Model analysis.

    All heavy computation runs in background threads (QThreadPool).
    Settings are persisted to ``{project_root}/config/motif_settings.json``.
    """

    def __init__(self, host: "BehaviorAnalyticsTab") -> None:
        super().__init__()
        self._host = host
        self._project_root: Path | None = None
        self._settings = MotifSettings()
        self._pool = QThreadPool.globalInstance()

        # ── graph size settings ──────────────────────────────────────
        self._motif_graph_settings: dict[str, Any] = {
            "max_w": 700,
            "max_h": 450,
            # Pixel size per subplot cell in per-group Network / chord view.
            # Canvas is auto-sized to (ncols * net_cell_px) x (nrows * net_cell_px).
            "net_cell_px": 380,
            "error_style": "SEM",       # "SEM" | "SD" | "95% CI" | "None"
            "bar_spacing": 1.0,
            "eb_capsize": 4,             # error bar cap width in points
            "eb_linewidth": 1.0,         # error bar line thickness in points
            "force_fit": False,
            "show_indiv_points": False,
            "show_stats": True,          # overlay significance stars on charts
        }

        # ── per-panel result caches ───────────────────────────────────
        self._transition_result: dict[str, Any] = {}
        self._motif_result: dict[str, Any] = {}
        self._hmm_result: dict[str, Any] = {}

        # ── comparison filter (None = show all significant pairs) ────
        self._motif_selected_comparisons: set[str] | None = None

        # ── shared status label ──────────────────────────────────────
        self._status_lbl = QLabel("Load analytics data (Refresh Analytics), then run an analysis.")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")

        self._settings_btn = QPushButton("Settings\u2026")
        self._settings_btn.setToolTip("Configure motif analysis settings (saved with project).")
        self._settings_btn.clicked.connect(self._open_settings_dialog)

        self._graph_size_btn = QPushButton("Graph Size\u2026")
        self._graph_size_btn.setToolTip("Set maximum display width and height for motif analysis graphs.")
        self._graph_size_btn.clicked.connect(self._open_motif_graph_size_dialog)

        self._motif_redraw_btn = QPushButton("Redraw")
        self._motif_redraw_btn.setToolTip(
            "Re-render the current panel with the latest graph settings\n"
            "without re-running the analysis."
        )
        self._motif_redraw_btn.clicked.connect(self._redraw_current_motif_panel)

        self._motif_comparisons_btn = QPushButton("Comparisons\u2026")
        self._motif_comparisons_btn.setToolTip(
            "Choose which pairwise group comparisons to show as significance markers on the charts.\n"
            "Options are auto-populated from the available group pairs after running an analysis."
        )
        self._motif_comparisons_btn.clicked.connect(self._open_comparisons_dialog)

        top_row = QHBoxLayout()
        top_row.addWidget(self._status_lbl, 1)
        top_row.addWidget(self._motif_comparisons_btn)
        top_row.addWidget(self._motif_redraw_btn)
        top_row.addWidget(self._graph_size_btn)
        top_row.addWidget(self._settings_btn)

        # ── inner sub-tabs ───────────────────────────────────────────
        self._inner_tabs = QTabWidget()
        self._trans_widget = self._build_transition_panel()
        self._motif_widget = self._build_motif_panel()
        self._hmm_widget   = self._build_hmm_panel()
        self._inner_tabs.addTab(self._trans_widget,  "Transition Matrix")
        self._inner_tabs.addTab(self._motif_widget,  "Behavior Relationships")
        self._inner_tabs.addTab(self._hmm_widget,    "HMM Analysis")

        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.addLayout(top_row)
        root.addWidget(self._inner_tabs, 1)

    # ── Panel builders ────────────────────────────────────────────────

    def _build_transition_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(4)

        # Row 1: gap + normalize
        ctrl1 = QHBoxLayout()
        self._tr_gap_spin = QDoubleSpinBox()
        self._tr_gap_spin.setRange(0.1, 300.0)
        self._tr_gap_spin.setValue(self._settings.max_transition_gap_s)
        self._tr_gap_spin.setSuffix(" s")
        self._tr_gap_spin.setDecimals(1)
        self._tr_gap_spin.setToolTip("Max gap between end of bout A and start of bout B.")
        self._tr_gap_spin.editingFinished.connect(self._sync_gap_to_settings)
        self._tr_norm_cb = QCheckBox("Probabilities")
        self._tr_norm_cb.setChecked(self._settings.normalize_rows)
        self._tr_norm_cb.setToolTip("Row-normalised probabilities (checked) or raw counts (unchecked).")
        self._tr_norm_cb.toggled.connect(lambda _=None: self._render_transition_from_cache())
        self._tr_run_btn = QPushButton("Run")
        self._tr_run_btn.clicked.connect(self._run_transition)
        ctrl1.addWidget(QLabel("Max gap:"))
        ctrl1.addWidget(self._tr_gap_spin)
        ctrl1.addWidget(self._tr_norm_cb)
        ctrl1.addWidget(self._tr_run_btn)
        ctrl1.addStretch(1)

        # Row 2: group selectors + view selector + export
        ctrl2 = QHBoxLayout()
        self._tr_group_a_combo = QComboBox()
        self._tr_group_a_combo.setToolTip("Group A (reference)")
        self._tr_group_a_combo.setMinimumWidth(120)
        self._tr_group_a_combo.currentIndexChanged.connect(lambda _=None: self._render_transition_from_cache())
        self._tr_group_b_combo = QComboBox()
        self._tr_group_b_combo.setToolTip("Group B (comparison)")
        self._tr_group_b_combo.setMinimumWidth(120)
        self._tr_group_b_combo.currentIndexChanged.connect(lambda _=None: self._render_transition_from_cache())
        self._tr_view_combo = QComboBox()
        self._tr_view_combo.addItem("Heatmap per group",     userData="heatmap_groups")
        self._tr_view_combo.addItem("Delta heatmap (A \u2212 B)", userData="delta")
        self._tr_view_combo.addItem("Grouped bar chart",     userData="bar")
        self._tr_view_combo.addItem("Network / chord",       userData="network")
        self._tr_view_combo.setToolTip("Choose visualization style.")
        self._tr_view_combo.currentIndexChanged.connect(self._on_tr_view_changed)
        self._tr_export_fig_btn = QPushButton("Export Figure\u2026")
        self._tr_export_fig_btn.clicked.connect(lambda: self._export_figure("transition"))
        self._tr_export_csv_btn = QPushButton("Export CSV\u2026")
        self._tr_export_csv_btn.clicked.connect(lambda: self._export_data_csv("transition"))
        ctrl2.addWidget(QLabel("Group A:"))
        ctrl2.addWidget(self._tr_group_a_combo)
        ctrl2.addWidget(QLabel("Group B:"))
        ctrl2.addWidget(self._tr_group_b_combo)
        ctrl2.addWidget(QLabel("View:"))
        ctrl2.addWidget(self._tr_view_combo)
        ctrl2.addWidget(self._tr_export_fig_btn)
        ctrl2.addWidget(self._tr_export_csv_btn)
        ctrl2.addStretch(1)
        layout.addLayout(ctrl1)
        layout.addLayout(ctrl2)

        # Row 3: Network-style options (only visible when "Network / chord" is selected)
        self._tr_net_style_row = QWidget()
        _tns_layout = QHBoxLayout(self._tr_net_style_row)
        _tns_layout.setContentsMargins(0, 0, 0, 0)
        _tns_layout.setSpacing(6)
        _tns_lbl = QLabel("Network style:")
        _tns_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        _tns_layout.addWidget(_tns_lbl)
        self._tr_net_style_combo = QComboBox()
        self._tr_net_style_combo.addItem("Pooled (all sessions)", userData="pooled")
        self._tr_net_style_combo.addItem("Per group",             userData="per_group")
        self._tr_net_style_combo.setToolTip(
            "Pooled: single diagram averaged across all sessions.\n"
            "Per group: one diagram per group side-by-side."
        )
        self._tr_net_style_combo.currentIndexChanged.connect(self._on_tr_net_style_changed)
        _tns_layout.addWidget(self._tr_net_style_combo)
        # Per-group filter sub-widget
        self._tr_net_group_filter_widget = QWidget()
        _tng_layout = QHBoxLayout(self._tr_net_group_filter_widget)
        _tng_layout.setContentsMargins(0, 0, 0, 0)
        _tng_layout.setSpacing(4)
        _tng_glbl = QLabel("Groups:")
        _tng_glbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        _tng_layout.addWidget(_tng_glbl)
        self._tr_net_group_list = QListWidget()
        self._tr_net_group_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._tr_net_group_list.setMaximumHeight(56)
        self._tr_net_group_list.setStyleSheet(
            "QListWidget{background:#0A1929;border:1px solid #1E3A5F;"
            "border-radius:3px;color:#cfd8dc;font-size:10px;}"
        )
        _tng_layout.addWidget(self._tr_net_group_list, 1)
        _tng_all = QPushButton("All")
        _tng_all.setMaximumWidth(40)
        _tng_all.clicked.connect(self._tr_net_group_check_all)
        _tng_none = QPushButton("None")
        _tng_none.setMaximumWidth(48)
        _tng_none.clicked.connect(self._tr_net_group_check_none)
        _tng_layout.addWidget(_tng_all)
        _tng_layout.addWidget(_tng_none)
        self._tr_net_group_filter_widget.setVisible(False)
        _tns_layout.addWidget(self._tr_net_group_filter_widget, 1)
        self._tr_net_style_row.setVisible(False)
        layout.addWidget(self._tr_net_style_row)

        self._tr_fig: Any = None
        self._tr_canvas: Any = None
        self._tr_toolbar: Any = None
        self._tr_canvas_scroll: Any = None
        if (
            _ensure_matplotlib()
            and Figure is not None
            and FigureCanvas is not None
            and NavigationToolbar is not None
        ):
            _tr_w = int(self._motif_graph_settings.get("max_w", 700))
            _tr_h = int(self._motif_graph_settings.get("max_h", 450))
            self._tr_fig = Figure(figsize=(_tr_w / 100, _tr_h / 100))
            self._tr_canvas = FigureCanvas(self._tr_fig)
            self._tr_canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._tr_canvas.setFixedSize(_tr_w, _tr_h)
            self._tr_toolbar = NavigationToolbar(self._tr_canvas, w)
            self._tr_canvas_scroll = QScrollArea()
            self._tr_canvas_scroll.setWidget(self._tr_canvas)
            self._tr_canvas_scroll.setWidgetResizable(False)
            self._tr_canvas_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._tr_canvas_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            # Dynamic resize: canvas grows to fill the available viewport width
            self._tr_resize_filter = _ViewportResizeFilter(
                self._sync_tr_canvas_to_viewport, self
            )
            self._tr_canvas_scroll.viewport().installEventFilter(self._tr_resize_filter)
            layout.addWidget(self._tr_toolbar)
            layout.addWidget(self._tr_canvas_scroll, 1)
        else:
            layout.addWidget(QLabel("Matplotlib is required."))

        self._tr_stats_text = ""
        self._tr_stats_btn = QPushButton("\U0001f4ca View Stats\u2026")
        self._tr_stats_btn.setToolTip("Show transition statistics and permutation-test p-values.")
        self._tr_stats_btn.clicked.connect(self._show_tr_stats_popup)
        self._tr_stats_btn.setEnabled(False)
        _tr_btn_row = QHBoxLayout()
        _tr_btn_row.addWidget(self._tr_stats_btn)
        _tr_btn_row.addStretch(1)
        layout.addLayout(_tr_btn_row)
        return w

    # ── Transition network group-filter helpers ─────────────────────

    def _on_tr_view_changed(self) -> None:
        view = str(self._tr_view_combo.currentData() or "")
        self._tr_net_style_row.setVisible(view == "network")
        self._render_transition_from_cache()

    def _on_tr_net_style_changed(self) -> None:
        style = str(self._tr_net_style_combo.currentData() or "pooled")
        self._tr_net_group_filter_widget.setVisible(style == "per_group")
        self._render_transition_from_cache()

    def _populate_tr_net_group_filter(self, groups: list[str]) -> None:
        self._tr_net_group_list.blockSignals(True)
        self._tr_net_group_list.clear()
        for g in sorted(groups):
            item = QListWidgetItem(g)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._tr_net_group_list.addItem(item)
        self._tr_net_group_list.blockSignals(False)

    def _tr_net_checked_groups(self) -> list[str]:
        out: list[str] = []
        for i in range(self._tr_net_group_list.count()):
            item = self._tr_net_group_list.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def _tr_net_group_check_all(self) -> None:
        self._tr_net_group_list.blockSignals(True)
        for i in range(self._tr_net_group_list.count()):
            item = self._tr_net_group_list.item(i)
            if item:
                item.setCheckState(Qt.CheckState.Checked)
        self._tr_net_group_list.blockSignals(False)
        self._render_transition_from_cache()

    def _tr_net_group_check_none(self) -> None:
        self._tr_net_group_list.blockSignals(True)
        for i in range(self._tr_net_group_list.count()):
            item = self._tr_net_group_list.item(i)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)
        self._tr_net_group_list.blockSignals(False)
        self._render_transition_from_cache()

    def _build_motif_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(4)

        ctrl1 = QHBoxLayout()
        self._mo_method_combo = QComboBox()
        self._mo_method_combo.addItem("N-gram Frequency",    userData="ngram")
        self._mo_method_combo.addItem("Sequence Clustering", userData="sequence_clustering")
        self._mo_method_combo.addItem("Both",                userData="both")
        idx = {"ngram": 0, "sequence_clustering": 1, "both": 2}.get(
            self._settings.motif_method, 2
        )
        self._mo_method_combo.setCurrentIndex(idx)
        self._mo_method_combo.setToolTip(
            "N-gram: most frequent multi-behavior sequences.\n"
            "Sequence Clustering: cluster sessions by motif profile."
        )
        self._mo_dedup_cb = QCheckBox("Skip consecutive same")
        self._mo_dedup_cb.setChecked(True)
        self._mo_dedup_cb.setToolTip(
            "Collapse runs of the same behavior into a single event before\n"
            "computing n-grams, avoiding uninformative A\u2192A\u2192A motifs."
        )
        self._mo_run_btn = QPushButton("Run")
        self._mo_run_btn.clicked.connect(self._run_motifs)
        ctrl1.addWidget(QLabel("Method:"))
        ctrl1.addWidget(self._mo_method_combo)
        ctrl1.addWidget(self._mo_dedup_cb)
        ctrl1.addWidget(self._mo_run_btn)
        ctrl1.addStretch(1)

        ctrl2 = QHBoxLayout()
        self._mo_view_combo = QComboBox()
        self._mo_view_combo.addItem("Top motifs (bar)",                  userData="bar")
        self._mo_view_combo.addItem("Grouped bars with error bars",      userData="grouped_bar")
        self._mo_view_combo.addItem("Engram sequence view",              userData="engram")
        self._mo_view_combo.addItem("Behavior correlation heatmap",       userData="corr_heatmap")
        self._mo_view_combo.addItem("Behavior correlation network",       userData="corr_network")
        self._mo_view_combo.addItem("Cluster map (UMAP)",                userData="cluster")
        self._mo_view_combo.setToolTip(
            "bar: ranked horizontal bars.\n"
            "grouped_bar: per-group mean ± SEM bars.\n"
            "engram: colored block sequences with group abundance bars.\n"
            "corr_heatmap: pooled + per-group behavior correlations.\n"
            "corr_network: graph view of the strongest behavior links.\n"
            "cluster: UMAP session scatter."
        )
        self._mo_view_combo.currentIndexChanged.connect(lambda _=None: self._on_mo_view_changed())
        self._mo_export_fig_btn = QPushButton("Export Figure\u2026")
        self._mo_export_fig_btn.clicked.connect(lambda: self._export_figure("motif"))
        self._mo_export_csv_btn = QPushButton("Export CSV\u2026")
        self._mo_export_csv_btn.clicked.connect(lambda: self._export_data_csv("motif"))
        ctrl2.addWidget(QLabel("View:"))
        ctrl2.addWidget(self._mo_view_combo)
        ctrl2.addWidget(self._mo_export_fig_btn)
        ctrl2.addWidget(self._mo_export_csv_btn)
        ctrl2.addStretch(1)

        # UMAP group filter row (only visible when cluster view is active)
        self._umap_group_filter_row = QWidget()
        _ugf_layout = QHBoxLayout(self._umap_group_filter_row)
        _ugf_layout.setContentsMargins(0, 0, 0, 0)
        _ugf_layout.setSpacing(6)
        _ugf_lbl = QLabel("UMAP Groups:")
        _ugf_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        _ugf_layout.addWidget(_ugf_lbl)
        self._umap_group_list = QListWidget()
        self._umap_group_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._umap_group_list.setMaximumHeight(60)
        self._umap_group_list.setStyleSheet(
            "QListWidget{background:#0A1929;border:1px solid #1E3A5F;"
            "border-radius:3px;color:#cfd8dc;font-size:10px;}"
        )
        _ugf_layout.addWidget(self._umap_group_list, 1)
        _ugf_all = QPushButton("All")
        _ugf_all.setMaximumWidth(40)
        _ugf_all.clicked.connect(self._umap_check_all_groups)
        _ugf_none = QPushButton("None")
        _ugf_none.setMaximumWidth(48)
        _ugf_none.clicked.connect(self._umap_uncheck_all_groups)
        _ugf_layout.addWidget(_ugf_all)
        _ugf_layout.addWidget(_ugf_none)
        _ugf_apply = QPushButton("Apply \u25ba")
        _ugf_apply.setMaximumWidth(60)
        _ugf_apply.setToolTip("Re-run UMAP using only the checked groups.")
        _ugf_apply.clicked.connect(self._run_motifs)
        _ugf_layout.addWidget(_ugf_apply)
        self._umap_group_filter_row.setVisible(False)

        layout.addLayout(ctrl1)
        layout.addLayout(ctrl2)
        layout.addWidget(self._umap_group_filter_row)

        self._mo_fig: Any = None
        self._mo_canvas: Any = None
        self._mo_toolbar: Any = None
        self._mo_canvas_scroll: Any = None
        if (
            _ensure_matplotlib()
            and Figure is not None
            and FigureCanvas is not None
            and NavigationToolbar is not None
        ):
            _mo_w = int(self._motif_graph_settings.get("max_w", 700))
            _mo_h = int(self._motif_graph_settings.get("max_h", 450))
            self._mo_fig = Figure(figsize=(_mo_w / 100, _mo_h / 100))
            self._mo_canvas = FigureCanvas(self._mo_fig)
            self._mo_canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._mo_canvas.setFixedSize(_mo_w, _mo_h)
            self._mo_toolbar = NavigationToolbar(self._mo_canvas, w)
            self._mo_canvas_scroll = QScrollArea()
            self._mo_canvas_scroll.setWidget(self._mo_canvas)
            self._mo_canvas_scroll.setWidgetResizable(False)
            self._mo_canvas_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._mo_canvas_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            # Dynamic resize: canvas grows to fill the available viewport width
            self._mo_resize_filter = _ViewportResizeFilter(
                self._sync_mo_canvas_to_viewport, self
            )
            self._mo_canvas_scroll.viewport().installEventFilter(self._mo_resize_filter)
            layout.addWidget(self._mo_toolbar)
            layout.addWidget(self._mo_canvas_scroll, 1)
        else:
            layout.addWidget(QLabel("Matplotlib is required."))

        self._mo_stats_text = ""
        self._mo_stats_btn = QPushButton("\U0001f4ca View Stats\u2026")
        self._mo_stats_btn.setToolTip("Show motif statistics and group comparison results.")
        self._mo_stats_btn.clicked.connect(self._show_mo_stats_popup)
        self._mo_stats_btn.setEnabled(False)
        _mo_btn_row = QHBoxLayout()
        _mo_btn_row.addWidget(self._mo_stats_btn)
        _mo_btn_row.addStretch(1)
        layout.addLayout(_mo_btn_row)
        return w

    def _build_hmm_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(4)

        ctrl1 = QHBoxLayout()
        self._hmm_mode_combo = QComboBox()
        self._hmm_mode_combo.addItem("Auto (AIC/BIC)", userData="auto")
        self._hmm_mode_combo.addItem("Manual",         userData="manual")
        self._hmm_mode_combo.setCurrentIndex(
            0 if self._settings.hmm_n_states_mode == "auto" else 1
        )
        self._hmm_mode_combo.currentIndexChanged.connect(self._sync_hmm_mode)
        self._hmm_n_states_spin = QSpinBox()
        self._hmm_n_states_spin.setRange(2, 20)
        self._hmm_n_states_spin.setValue(self._settings.hmm_n_states)
        self._hmm_n_states_spin.setVisible(self._settings.hmm_n_states_mode == "manual")
        self._hmm_run_btn = QPushButton("Run HMM")
        self._hmm_run_btn.clicked.connect(self._run_hmm)
        ctrl1.addWidget(QLabel("Mode:"))
        ctrl1.addWidget(self._hmm_mode_combo)
        ctrl1.addWidget(QLabel("N states:"))
        ctrl1.addWidget(self._hmm_n_states_spin)
        ctrl1.addWidget(self._hmm_run_btn)
        ctrl1.addStretch(1)

        ctrl2 = QHBoxLayout()
        self._hmm_view_combo = QComboBox()
        self._hmm_view_combo.addItem("Model selection (AIC/BIC)",              userData="model_sel")
        self._hmm_view_combo.addItem("Emission heatmap",                       userData="emission")
        self._hmm_view_combo.addItem("State occupancy per group",      userData="occupancy")
        self._hmm_view_combo.addItem("State-to-state transition heatmap",      userData="trans_hmm")
        self._hmm_view_combo.setToolTip(
            "Choose which aspect of the HMM to visualize."
        )
        self._hmm_view_combo.currentIndexChanged.connect(lambda _=None: self._render_hmm())
        self._hmm_export_fig_btn = QPushButton("Export Figure\u2026")
        self._hmm_export_fig_btn.clicked.connect(lambda: self._export_figure("hmm"))
        self._hmm_export_csv_btn = QPushButton("Export CSV\u2026")
        self._hmm_export_csv_btn.clicked.connect(lambda: self._export_data_csv("hmm"))
        ctrl2.addWidget(QLabel("View:"))
        ctrl2.addWidget(self._hmm_view_combo)
        ctrl2.addWidget(self._hmm_export_fig_btn)
        ctrl2.addWidget(self._hmm_export_csv_btn)
        ctrl2.addStretch(1)
        layout.addLayout(ctrl1)
        layout.addLayout(ctrl2)

        self._hmm_sel_fig: Any = None
        self._hmm_sel_canvas: Any = None
        self._hmm_emit_fig: Any = None
        self._hmm_emit_canvas: Any = None
        self._hmm_canvas_scroll: Any = None

        if _ensure_matplotlib() and Figure is not None and FigureCanvas is not None:
            _hmm_w = int(self._motif_graph_settings.get("max_w", 700))
            _hmm_h = int(self._motif_graph_settings.get("max_h", 450))
            self._hmm_sel_fig = Figure(figsize=(_hmm_w / 100, _hmm_h / 100))
            self._hmm_sel_canvas = FigureCanvas(self._hmm_sel_fig)
            self._hmm_sel_canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._hmm_sel_canvas.setFixedSize(_hmm_w, _hmm_h)
            self._hmm_canvas_scroll = QScrollArea()
            self._hmm_canvas_scroll.setWidget(self._hmm_sel_canvas)
            self._hmm_canvas_scroll.setWidgetResizable(False)
            self._hmm_canvas_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._hmm_canvas_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            # Dynamic resize: canvas grows to fill the available viewport width
            self._hmm_resize_filter = _ViewportResizeFilter(
                self._sync_hmm_canvas_to_viewport, self
            )
            self._hmm_canvas_scroll.viewport().installEventFilter(self._hmm_resize_filter)
            layout.addWidget(NavigationToolbar(self._hmm_sel_canvas, w))
            layout.addWidget(self._hmm_canvas_scroll, 1)
        else:
            layout.addWidget(QLabel("Matplotlib required."))

        self._hmm_stats_text = ""
        self._hmm_stats_btn = QPushButton("\U0001f4ca View HMM Stats\u2026")
        self._hmm_stats_btn.setToolTip("Show HMM model parameters, AIC/BIC, and per-group state occupancy.")
        self._hmm_stats_btn.clicked.connect(self._show_hmm_stats_popup)
        self._hmm_stats_btn.setEnabled(False)
        _hmm_btn_row = QHBoxLayout()
        _hmm_btn_row.addWidget(self._hmm_stats_btn)
        _hmm_btn_row.addStretch(1)
        layout.addLayout(_hmm_btn_row)
        return w

    # ── Canvas auto-fill ──────────────────────────────────────────────

    def _sync_tr_canvas_to_viewport(self) -> None:
        """Fill the transition canvas to its scroll viewport width, keeping aspect."""
        _autofill_canvas(
            getattr(self, "_tr_canvas_scroll", None),
            getattr(self, "_tr_canvas", None),
            getattr(self, "_tr_fig", None),
            dpi=100,
        )

    def _sync_mo_canvas_to_viewport(self) -> None:
        """Fill the relationships canvas to its scroll viewport width, keeping aspect."""
        _autofill_canvas(
            getattr(self, "_mo_canvas_scroll", None),
            getattr(self, "_mo_canvas", None),
            getattr(self, "_mo_fig", None),
            dpi=100,
        )

    def _sync_hmm_canvas_to_viewport(self) -> None:
        """Fill the HMM canvas to its scroll viewport width, keeping aspect."""
        _autofill_canvas(
            getattr(self, "_hmm_canvas_scroll", None),
            getattr(self, "_hmm_sel_canvas", None),
            getattr(self, "_hmm_sel_fig", None),
            dpi=100,
        )

    def showEvent(self, event: Any) -> None:  # type: ignore[override]
        """Re-fill the canvases when this sub-tab becomes visible (the resize
        filter only fires on an actual viewport resize, so a figure drawn while
        hidden would otherwise stay small until the splitter is wiggled)."""
        super().showEvent(event)

        def _resync() -> None:
            self._sync_tr_canvas_to_viewport()
            self._sync_mo_canvas_to_viewport()
            self._sync_hmm_canvas_to_viewport()

        QTimer.singleShot(0, _resync)

    # ── Project / data hooks ──────────────────────────────────────────

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._settings = load_motif_settings(project_root)
        self._sync_settings_to_ui()
        self._transition_result.clear()
        self._motif_result.clear()
        self._hmm_result.clear()

    def on_data_loaded(self) -> None:
        self._transition_result.clear()
        self._motif_result.clear()
        self._hmm_result.clear()
        for fig in (self._tr_fig, self._mo_fig, self._hmm_sel_fig, self._hmm_emit_fig):
            if fig is not None:
                fig.clear()
        for canvas in (self._tr_canvas, self._mo_canvas, self._hmm_sel_canvas, self._hmm_emit_canvas):
            if canvas is not None:
                canvas.draw_idle()
        self._tr_stats_text = ""
        self._mo_stats_text = ""
        self._hmm_stats_text = ""
        self._tr_stats_btn.setEnabled(False)
        self._mo_stats_btn.setEnabled(False)
        self._hmm_stats_btn.setEnabled(False)
        # Populate group pickers from newly loaded data
        self._refresh_group_combos()
        self._status_lbl.setText("Data refreshed. Run an analysis in each sub-tab.")

    def _refresh_group_combos(self) -> None:
        """Populate the Group A / Group B dropdowns with current group names."""
        groups = sorted({g for g in self._host._session_groups.values() if g})
        for combo in (self._tr_group_a_combo, self._tr_group_b_combo):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(all sessions)", userData="")
            for g in groups:
                combo.addItem(g, userData=g)
            # restore previous selection if still present
            idx = combo.findText(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif combo is self._tr_group_b_combo and len(groups) >= 2:
                combo.setCurrentIndex(2)  # pick second group by default
            combo.blockSignals(False)

    # ── Settings helpers ──────────────────────────────────────────────

    def _sync_gap_to_settings(self) -> None:
        self._settings.max_transition_gap_s = float(self._tr_gap_spin.value())
        self._save_settings()

    def _sync_hmm_mode(self) -> None:
        mode = str(self._hmm_mode_combo.currentData() or "auto")
        self._settings.hmm_n_states_mode = mode
        self._hmm_n_states_spin.setVisible(mode == "manual")
        self._save_settings()

    def _sync_settings_to_ui(self) -> None:
        self._tr_gap_spin.setValue(self._settings.max_transition_gap_s)
        self._tr_norm_cb.setChecked(self._settings.normalize_rows)
        idx = {"ngram": 0, "sequence_clustering": 1, "both": 2}.get(self._settings.motif_method, 0)
        self._mo_method_combo.setCurrentIndex(idx)
        hmm_idx = 0 if self._settings.hmm_n_states_mode == "auto" else 1
        self._hmm_mode_combo.setCurrentIndex(hmm_idx)
        self._hmm_n_states_spin.setValue(self._settings.hmm_n_states)
        self._hmm_n_states_spin.setVisible(self._settings.hmm_n_states_mode == "manual")

    def _save_settings(self) -> None:
        if self._project_root is not None:
            save_motif_settings(self._project_root, self._settings)

    # ── Shared helpers ────────────────────────────────────────────────

    def _get_behavior_ids_and_names(
        self,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        bids: list[str] = []
        bnames: list[str] = []
        for b in self._host._behaviors.behaviors:
            bid = str(getattr(b, "behavior_id", "") or "").strip()
            if not bid or bid in {NO_BEHAVIOR_ID, DISTANCE_BEHAVIOR_ID}:
                continue
            if bid not in self._host._selected_behavior_ids():
                continue
            bids.append(bid)
            bnames.append(str(getattr(b, "name", "") or bid))
        return bids, bnames, dict(zip(bids, bnames))

    def _get_sequences_for_analysis(
        self,
    ) -> dict[str, list[tuple[float, float, str]]]:
        from abel.services.behavioral_motif_service import build_sequences
        checked = self._host._summary_tab._checked_subjects()
        bids, _, _ = self._get_behavior_ids_and_names()
        fps = self._host._project_fps()
        raw = build_sequences(self._host._raw_bouts, fps, selected_bids=set(bids))
        label_map = self._host._session_label_by_session
        return {
            sid: events
            for sid, events in raw.items()
            if label_map.get(sid, sid) in checked
        }

    def _build_session_group_map(
        self, sequences: dict[str, list]
    ) -> dict[str, str]:
        groups = self._host._session_groups
        label_map = self._host._session_label_by_session
        return {
            sid: grp
            for sid in sequences
            if (grp := groups.get(label_map.get(sid, sid), ""))
        }

    @staticmethod
    def _dedup_consecutive(
        events: list[tuple[float, float, str]],
    ) -> list[tuple[float, float, str]]:
        """Collapse consecutive runs of the same behavior into one event."""
        if not events:
            return []
        out: list[tuple[float, float, str]] = [events[0]]
        for ev in events[1:]:
            if ev[2] != out[-1][2]:
                out.append(ev)
        return out

    @staticmethod
    def _group_sem(
        per_session: dict[str, Any], session_to_group: dict[str, str]
    ) -> dict[str, tuple[float, float]]:
        """Return {group: (mean, sem)} for scalar per_session values."""
        from collections import defaultdict
        grouped: dict[str, list[float]] = defaultdict(list)
        for sid, val in per_session.items():
            grp = session_to_group.get(sid, "")
            grouped[grp].append(float(val))
        out: dict[str, tuple[float, float]] = {}
        for grp, vals in grouped.items():
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            mean = float(arr.mean())
            sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            out[grp] = (mean, sem)
        return out

    def _transition_pval_correction_mode(self) -> str:
        mode = str(getattr(self._settings, "transition_pval_correction", "fdr_bh") or "fdr_bh").strip().lower()
        return mode if mode in {"none", "fdr_bh"} else "fdr_bh"

    @staticmethod
    def _fdr_bh_adjust(pmat: np.ndarray) -> np.ndarray:
        """Return Benjamini-Hochberg FDR-adjusted q-values for a p-value matrix."""
        arr = np.asarray(pmat, dtype=float)
        flat = arr.ravel()
        valid = np.isfinite(flat)
        if not valid.any():
            return arr.copy()

        pv = np.clip(flat[valid], 0.0, 1.0)
        m = pv.size
        order = np.argsort(pv)
        ranked = pv[order]
        q_ranked = ranked * m / np.arange(1, m + 1, dtype=float)
        q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
        q_ranked = np.clip(q_ranked, 0.0, 1.0)

        qvals = np.empty_like(pv)
        qvals[order] = q_ranked
        out = flat.copy()
        out[valid] = qvals
        return out.reshape(arr.shape)

    def _apply_transition_pval_correction(self, pmat: np.ndarray) -> np.ndarray:
        mode = self._transition_pval_correction_mode()
        arr = np.asarray(pmat, dtype=float)
        if mode == "none":
            return arr
        return self._fdr_bh_adjust(arr)

    # ── Transition Matrix panel ───────────────────────────────────────

    def _run_transition(self) -> None:
        from abel.services.behavioral_motif_service import (
            compute_transition_matrix,
            group_mean_matrix,
            normalize_transition_matrix,
            permutation_test_transition,
        )

        if not self._host._raw_bouts:
            self._tr_stats_text = "Load analytics data first (Refresh Analytics)."
            return
        bids, bnames, bid_to_name = self._get_behavior_ids_and_names()
        if len(bids) < 2:
            self._tr_stats_text = "Select at least 2 behaviors."
            return

        gap_s = float(self._tr_gap_spin.value())
        self._settings.max_transition_gap_s = gap_s
        self._settings.normalize_rows = self._tr_norm_cb.isChecked()
        self._save_settings()

        sequences = self._get_sequences_for_analysis()
        if not sequences:
            self._tr_stats_text = "No session data found. Run temporal refinement first."
            return

        session_group_map = self._build_session_group_map(sequences)
        n_perms = self._settings.n_permutations
        seed = self._settings.permutation_seed
        include_self = self._settings.include_self_transitions

        self._tr_run_btn.setEnabled(False)
        self._tr_run_btn.setText("Running\u2026")
        self._status_lbl.setText("Computing transition matrices\u2026")

        def _compute() -> dict[str, Any]:
            per_session = compute_transition_matrix(
                sequences, bids, max_gap_s=gap_s, include_self=include_self,
                overlap_tolerance_s=self._settings.bout_overlap_tolerance_s,
            )
            groups = sorted({g for g in session_group_map.values() if g})
            group_matrices = group_mean_matrix(per_session, session_group_map)
            # Per-group lists of matrices for SEM
            per_group_mats: dict[str, list[np.ndarray]] = {}
            for sid, mat in per_session.items():
                grp = session_group_map.get(sid, "")
                if grp:
                    per_group_mats.setdefault(grp, []).append(mat)
            group_sem_mats: dict[str, np.ndarray] = {}
            for grp, mats in per_group_mats.items():
                if len(mats) > 1:
                    group_sem_mats[grp] = np.stack(mats).std(axis=0, ddof=1) / np.sqrt(len(mats))
                else:
                    group_sem_mats[grp] = np.zeros_like(mats[0])
            # Build per-session probability matrices so permutation tests can
            # match the displayed/exported metric (count vs probability).
            per_group_prob_mats: dict[str, list[np.ndarray]] = {
                grp: [normalize_transition_matrix(m) for m in mats]
                for grp, mats in per_group_mats.items()
            }

            # Permutation test for every pair of groups.
            pval_mats_count: dict[tuple[str, str], np.ndarray] = {}
            pval_mats_prob: dict[tuple[str, str], np.ndarray] = {}
            for g1, g2 in combinations(groups, 2):
                m1_count = per_group_mats.get(g1, [])
                m2_count = per_group_mats.get(g2, [])
                if m1_count and m2_count:
                    pval_mats_count[(g1, g2)] = permutation_test_transition(
                        m1_count, m2_count, n_perms, seed
                    )

                m1_prob = per_group_prob_mats.get(g1, [])
                m2_prob = per_group_prob_mats.get(g2, [])
                if m1_prob and m2_prob:
                    pval_mats_prob[(g1, g2)] = permutation_test_transition(
                        m1_prob, m2_prob, n_perms, seed
                    )
            return {
                "per_session": per_session,
                "group_matrices": group_matrices,
                "group_sem_mats": group_sem_mats,
                "per_group_mats": per_group_mats,
                # Backward-compatible alias used by older code paths.
                "pval_mats": pval_mats_count,
                "pval_mats_count": pval_mats_count,
                "pval_mats_prob": pval_mats_prob,
                "bids": bids,
                "bnames": bnames,
                "gap_s": gap_s,
                "groups": groups,
            }

        worker = TaskWorker(_compute)
        worker.signals.finished.connect(self._on_transition_done)
        worker.signals.failed.connect(self._on_worker_failed)
        self._pool.start(worker)

    def _on_transition_done(self, result: dict[str, Any]) -> None:
        self._tr_run_btn.setEnabled(True)
        self._tr_run_btn.setText("Run")
        self._transition_result = result
        # Repopulate group pickers with result groups
        groups = result.get("groups", [])
        for combo in (self._tr_group_a_combo, self._tr_group_b_combo):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(all sessions)", userData="")
            for g in groups:
                combo.addItem(g, userData=g)
            idx = combo.findText(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        if not self._tr_group_a_combo.currentData() and groups:
            self._tr_group_a_combo.setCurrentIndex(1)
        if not self._tr_group_b_combo.currentData() and len(groups) >= 2:
            self._tr_group_b_combo.setCurrentIndex(2)
        self._populate_tr_net_group_filter(groups)
        self._render_transition_from_cache()
        self._status_lbl.setText("Transition matrix complete.")

    def _render_transition_from_cache(self) -> None:
        if self._tr_fig is None or not self._transition_result:
            return
        result = self._transition_result
        bnames: list[str] = result["bnames"]
        group_matrices: dict[str, Any] = result.get("group_matrices", {})
        per_session: dict[str, Any] = result.get("per_session", {})
        gap_s: float = result.get("gap_s", self._settings.max_transition_gap_s)
        group_sem_mats: dict[str, Any] = result.get("group_sem_mats", {})
        normalize = self._tr_norm_cb.isChecked()
        pval_mats_count: dict = result.get("pval_mats_count", result.get("pval_mats", {}))
        pval_mats_prob: dict = result.get("pval_mats_prob", pval_mats_count)
        pval_mats_raw: dict = pval_mats_prob if normalize else pval_mats_count
        pval_mats: dict = {
            k: self._apply_transition_pval_correction(v)
            for k, v in pval_mats_raw.items()
        }
        pval_mode = self._transition_pval_correction_mode()
        view = str(self._tr_view_combo.currentData() or "heatmap_groups")
        grp_a = str(self._tr_group_a_combo.currentData() or "")
        grp_b = str(self._tr_group_b_combo.currentData() or "")
        n = len(bnames)

        self._tr_fig.clear()
        if n == 0:
            self._tr_canvas.draw_idle()
            return

        def _display_mat(raw_mat: np.ndarray) -> tuple[np.ndarray, str, float]:
            if normalize:
                from abel.services.behavioral_motif_service import normalize_transition_matrix
                dm = normalize_transition_matrix(raw_mat)
                return dm, "Transition Probability", 1.0
            return raw_mat, "Count", float(raw_mat.max()) or 1.0

        def _annotate(ax: Any, mat: np.ndarray, vmax: float, fontsize: int = 8) -> None:
            for i in range(n):
                for j in range(n):
                    v = float(mat[i, j])
                    if abs(v) > 1e-6:
                        tc = "white" if abs(v) > vmax * 0.6 else "black"
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=fontsize, color=tc)

        gs = self._host._graph_settings

        if view == "heatmap_groups":
            groups_to_show: list[str] = (
                [g for g in result.get("groups", []) if not grp_a or g == grp_a or g == grp_b]
                if (grp_a or grp_b)
                else result.get("groups", [])
            )
            if not groups_to_show and group_matrices:
                groups_to_show = list(group_matrices.keys())
            if not groups_to_show:
                # No groups — pool everything
                if per_session:
                    stacked = np.stack(list(per_session.values()), axis=0)
                    mean_mat = stacked.mean(axis=0)
                else:
                    mean_mat = np.zeros((n, n))
                dm, clabel, vmax = _display_mat(mean_mat)
                ax = self._tr_fig.add_subplot(111)
                im = ax.imshow(dm, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
                self._tr_fig.colorbar(im, ax=ax, label=clabel, shrink=0.8)
                ax.set_xticks(range(n)); ax.set_xticklabels(bnames, rotation=45, ha="right", fontsize=9)
                ax.set_yticks(range(n)); ax.set_yticklabels(bnames, fontsize=9)
                ax.set_xlabel("Second behavior (to)", fontsize=9)
                ax.set_ylabel("First behavior (from)", fontsize=9)
                ax.set_title(f"Transition Matrix \u2014 Pooled (max gap={gap_s:.1f}s)", fontsize=11)
                _annotate(ax, dm, vmax)
            else:
                ncols = min(len(groups_to_show), 4)
                nrows = (len(groups_to_show) + ncols - 1) // ncols
                for gi, gname in enumerate(groups_to_show):
                    ax = self._tr_fig.add_subplot(nrows, ncols, gi + 1)
                    raw = group_matrices.get(gname, np.zeros((n, n)))
                    dm, clabel, vmax = _display_mat(raw)
                    im = ax.imshow(dm, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
                    self._tr_fig.colorbar(im, ax=ax, label=clabel, shrink=0.7)
                    ax.set_xticks(range(n)); ax.set_xticklabels(bnames, rotation=45, ha="right", fontsize=8)
                    ax.set_yticks(range(n)); ax.set_yticklabels(bnames, fontsize=8)
                    ax.set_xlabel("Second behavior (to)", fontsize=8)
                    ax.set_ylabel("First behavior (from)", fontsize=8)
                    ax.set_title(str(gname), fontsize=10)
                    _annotate(ax, dm, vmax, 7)

        elif view == "delta":
            # Delta heatmap: Group A − Group B
            mat_a = group_matrices.get(grp_a, np.zeros((n, n)))
            mat_b = group_matrices.get(grp_b, np.zeros((n, n)))
            if normalize:
                from abel.services.behavioral_motif_service import normalize_transition_matrix
                mat_a = normalize_transition_matrix(mat_a)
                mat_b = normalize_transition_matrix(mat_b)
            delta = mat_a - mat_b
            vabs = float(np.abs(delta).max()) or 0.05

            ax_delta = self._tr_fig.add_subplot(1, 2, 1) if pval_mats else self._tr_fig.add_subplot(111)
            im = ax_delta.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
            self._tr_fig.colorbar(im, ax=ax_delta, label=f"\u0394 ({'prob' if normalize else 'count'})", shrink=0.8)
            ax_delta.set_xticks(range(n)); ax_delta.set_xticklabels(bnames, rotation=45, ha="right", fontsize=9)
            ax_delta.set_yticks(range(n)); ax_delta.set_yticklabels(bnames, fontsize=9)
            ax_delta.set_xlabel("Second behavior (to)", fontsize=9)
            ax_delta.set_ylabel("First behavior (from)", fontsize=9)
            ax_delta.set_title(
                f"\u0394 Transition ({grp_a or 'A'} \u2212 {grp_b or 'B'})\nmax gap={gap_s:.1f}s",
                fontsize=11,
            )
            for i in range(n):
                for j in range(n):
                    v = float(delta[i, j])
                    if abs(v) > 1e-4:
                        tc = "white" if abs(v) > vabs * 0.6 else "black"
                        ax_delta.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color=tc)

            # Right panel: p-value heatmap if available
            pair_key = (grp_a, grp_b) if (grp_a, grp_b) in pval_mats else (grp_b, grp_a) if (grp_b, grp_a) in pval_mats else None
            if pair_key and pval_mats:
                ax_p = self._tr_fig.add_subplot(1, 2, 2)
                pmat = pval_mats[pair_key]
                im2 = ax_p.imshow(pmat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.1)
                self._tr_fig.colorbar(im2, ax=ax_p, label="p-value (permutation)", shrink=0.8)
                ax_p.set_xticks(range(n)); ax_p.set_xticklabels(bnames, rotation=45, ha="right", fontsize=9)
                ax_p.set_yticks(range(n)); ax_p.set_yticklabels(bnames, fontsize=9)
                ax_p.set_xlabel("Second behavior (to)", fontsize=9)
                ax_p.set_ylabel("First behavior (from)", fontsize=9)
                test_basis = "probabilities" if normalize else "counts"
                corr_basis = "FDR-BH corrected" if pval_mode == "fdr_bh" else "raw"
                ax_p.set_title(
                    f"Permutation p-value ({test_basis}, {corr_basis})\n({self._settings.n_permutations} perms)",
                    fontsize=11,
                )
                sig_label = "q" if pval_mode == "fdr_bh" else "p"
                for i in range(n):
                    for j in range(n):
                        p = float(pmat[i, j])
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        if sig:
                            ax_p.text(j, i, f"{sig_label}={p:.3f}\n{sig}", ha="center", va="center", fontsize=7, color="black")

        elif view == "bar":
            # Grouped bar chart for each transition pair (from, to) with error bars
            pairs: list[tuple[int, int]] = [
                (i, j) for i in range(n) for j in range(n)
                if i != j or self._settings.include_self_transitions
            ]
            if not pairs:
                self._tr_fig.text(0.5, 0.5, "No transitions.", ha="center", va="center")
                self._tr_canvas.draw_idle()
                return
            mgs = self._motif_graph_settings
            error_style = mgs.get("error_style", "SEM")
            bar_spacing  = float(mgs.get("bar_spacing", 1.0))
            groups = result.get("groups", [])
            per_group_mats: dict[str, list[Any]] = result.get("per_group_mats", {})
            ax = self._tr_fig.add_subplot(111)
            x = np.arange(len(pairs))
            width = min(0.80, 0.70 * bar_spacing) / max(len(groups), 1)
            xlabels = [f"{bnames[i][:6]}\u2192{bnames[j][:6]}" for i, j in pairs]
            _bar_tops_tr = np.zeros(len(pairs))
            for gi, grp in enumerate(groups):
                mats = per_group_mats.get(grp, [])
                if not mats:
                    continue
                vals = np.array([[_display_mat(m)[0][i, j] for i, j in pairs] for m in mats])
                means = vals.mean(axis=0)
                ebs = np.array([
                    _eb_val(vals[:, p_idx], error_style) for p_idx in range(len(pairs))
                ])
                for _xi in range(len(pairs)):
                    _top = means[_xi] + (ebs[_xi] if error_style != "None" else 0.0)
                    if _top > _bar_tops_tr[_xi]:
                        _bar_tops_tr[_xi] = _top
                _mgs_cs = int(mgs.get("eb_capsize", 4)) if error_style != "None" else 0
                _mgs_lw = float(mgs.get("eb_linewidth", 1.0))
                offset = (gi - len(groups) / 2 + 0.5) * width * bar_spacing
                ax.bar(x + offset, means,
                       width=width * 0.88 * bar_spacing,
                       yerr=(ebs if error_style != "None" else None),
                       label=grp, color=_PALETTE[gi % len(_PALETTE)],
                       alpha=0.85,
                       capsize=_mgs_cs,
                       error_kw={"elinewidth": _mgs_lw, "capthick": _mgs_lw} if error_style != "None" else {})
            # Significance stars from permutation test (filtered by selected comparisons)
            active_pval_mats = self._pval_mat_filtered(pval_mats)
            if active_pval_mats and mgs.get("show_stats", True):
                for _xi, (_pi, _pj) in enumerate(pairs):
                    _min_p = min(float(_pm[_pi, _pj]) for _pm in active_pval_mats.values())
                    if _min_p < 0.05:
                        _sig = "***" if _min_p < 0.001 else "**" if _min_p < 0.01 else "*"
                        _top = float(_bar_tops_tr[_xi])
                        _gap = _top * 0.04 + 0.05
                        ax.text(x[_xi], _top + _gap, _sig,
                                ha="center", va="bottom", fontsize=9,
                                color="black", fontweight="bold", zorder=6)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=60, ha="right",
                               fontsize=max(5, gs["tick_fontsize"] - 2))
            eb_lbl = f" \u00b1 {error_style}" if error_style != "None" else ""
            ax.set_ylabel("Transition probability" if normalize else "Mean count",
                          fontsize=gs["axis_fontsize"])
            ax.set_title(f"Transition Rates per Group (mean{eb_lbl})", fontsize=gs["title_fontsize"])
            if groups:
                ax.legend(fontsize="x-small")

        elif view == "network":
            net_style = str(self._tr_net_style_combo.currentData() or "pooled")
            net_groups = self._tr_net_checked_groups() if net_style == "per_group" else []
            self._render_transition_network(
                ax_root=None, result=result,
                normalize=normalize, gap_s=gap_s,
                style=net_style, selected_groups=net_groups,
            )
            self._sync_tr_canvas_to_viewport()
            self._tr_canvas.draw_idle()
            lines = self._build_tr_stats_lines(result, normalize)
            self._tr_stats_text = "\n".join(lines)
            self._tr_stats_btn.setEnabled(True)
            return

        try:
            self._tr_fig.tight_layout(pad=1.2)
        except Exception:
            pass
        mgs_ff = self._motif_graph_settings
        if mgs_ff.get("force_fit", False):
            max_w = int(mgs_ff.get("max_w", 700))
            max_h = int(mgs_ff.get("max_h", 450))
            _force_fit_canvas(self._tr_canvas, self._tr_fig, max_w, max_h)
        elif view == "bar":
            # A grouped bar chart shouldn't grow taller than the visible area —
            # cap the height to the viewport so a wide window doesn't balloon it
            # into a scroll-only figure. (Matrix/heatmap views keep their aspect.)
            _autofill_canvas(
                getattr(self, "_tr_canvas_scroll", None), self._tr_canvas, self._tr_fig,
                dpi=100, max_h=self._tr_canvas_scroll.viewport().height(),
            )
        else:
            self._sync_tr_canvas_to_viewport()
        self._tr_canvas.draw_idle()
        lines = self._build_tr_stats_lines(result, normalize)
        self._tr_stats_text = "\n".join(lines)
        self._tr_stats_btn.setEnabled(True)

    def _render_transition_network(
        self,
        ax_root: Any,
        result: dict[str, Any],
        normalize: bool,
        gap_s: float,
        style: str = "pooled",
        selected_groups: list[str] | None = None,
    ) -> None:
        """Draw directed network/chord diagrams — pooled or one per group."""
        bnames: list[str] = result["bnames"]
        n = len(bnames)
        per_session: dict[str, Any] = result.get("per_session", {})
        if not per_session:
            return

        import matplotlib.colors as _mcolors  # type: ignore[import-untyped]
        from abel.services.behavioral_motif_service import normalize_transition_matrix

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = np.cos(angles)
        ys = np.sin(angles)
        _LABEL_R = 1.30

        def _prep_mat(raw: np.ndarray) -> np.ndarray:
            return normalize_transition_matrix(raw) if normalize else raw

        def _draw_one(ax: Any, mat: np.ndarray, title: str, vmax_ref: float = 0.0) -> None:
            ax.set_aspect("equal")
            vmax = vmax_ref if vmax_ref > 0 else (float(mat.max()) or 1.0)
            # Use source-node color for each arc so every transition is visible
            # regardless of its strength value (fixes the RdYlBu_r yellow-on-
            # white invisibility problem for mid-range probabilities ~0.3-0.6).
            _MIN_V = 0.005  # skip truly zero transitions
            # Collect and sort edges weakest-first so strong arcs render on top.
            edges = [
                (mat[i, j], i, j)
                for i in range(n) for j in range(n)
                if i != j and mat[i, j] >= _MIN_V
            ]
            edges.sort(key=lambda t: t[0])
            for v, i, j in edges:
                # Alpha and lw scale with strength; floor ensures weak arcs visible.
                alpha = min(0.92, max(0.30, 0.20 + 0.70 * v / vmax))
                lw = max(0.7, 0.5 + 4.0 * v / vmax)
                # Color = source node color (clearly visible on any background).
                src_color = _PALETTE[i % len(_PALETTE)]
                # Slightly darken weak arcs so thin lines don't look washed out.
                rgba = _mcolors.to_rgba(src_color)
                darkened = tuple(max(0, c * (0.55 + 0.45 * v / vmax)) for c in rgba[:3]) + (alpha,)
                # Arc curvature: always +0.40.  For any pair (A→B, B→A),
                # using the same sign for both means the CCW-perpendicular for
                # A→B and for B→A point to OPPOSITE sides of the chord (since
                # the direction vectors are reversed), so the two arcs physically
                # separate onto different geometric paths.  Using i<j to negate
                # the sign is a mathematical no-op: it produces the same Bezier
                # control point and the arcs overlap.
                rad = 0.40
                ax.annotate(
                    "",
                    xy=(xs[j] * 0.90, ys[j] * 0.90),
                    xytext=(xs[i] * 0.90, ys[i] * 0.90),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=darkened,
                        lw=lw,
                        connectionstyle=f"arc3,rad={rad}",
                        mutation_scale=12,
                    ),
                    zorder=2,
                )
            for i, (bname, x, y) in enumerate(zip(bnames, xs, ys)):
                ax.scatter([x], [y], s=240, color=_PALETTE[i % len(_PALETTE)], zorder=5,
                           edgecolors="white", linewidths=0.9)
                lx, ly = x * _LABEL_R, y * _LABEL_R
                ha = "left" if x > 0.1 else ("right" if x < -0.1 else "center")
                va = "bottom" if y > 0.1 else ("top" if y < -0.1 else "center")
                short = bname[:14] if len(bname) > 14 else bname
                ax.text(lx, ly, short, ha=ha, va=va, fontsize=8,
                        fontweight="bold", color=_PALETTE[i % len(_PALETTE)])
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.axis("off")
            ax.set_title(title, fontsize=10)
            # Direction key: small text in lower-right corner
            ax.text(
                1.55, -1.55,
                "→ source to destination",
                ha="right", va="bottom", fontsize=6.5,
                color="#888888", style="italic",
                transform=ax.transData,
            )

        if style == "per_group":
            group_matrices: dict[str, np.ndarray] = result.get("group_matrices", {})
            groups_to_draw = [g for g in (selected_groups or sorted(group_matrices.keys()))
                              if g in group_matrices]
            if not groups_to_draw:
                style = "pooled"  # fall back gracefully

        if style == "per_group":
            ng = len(groups_to_draw)
            ncols = min(ng, 3)
            nrows = (ng + ncols - 1) // ncols
            # --- Dynamic canvas resize so each cell has enough room ----------
            cell_px = int(self._motif_graph_settings.get("net_cell_px", 380))
            ideal_px_w = ncols * cell_px
            ideal_px_h = nrows * cell_px
            # Resize canvas; QScrollArea handles any overflow.
            if self._tr_canvas is not None:
                self._tr_canvas.setFixedSize(ideal_px_w, ideal_px_h)
            # Also set figure size explicitly (resize_event may not fire yet)
            self._tr_fig.set_size_inches(ideal_px_w / 100.0, ideal_px_h / 100.0)
            # Consistent vmax across all subplots for fair color comparison
            all_mats = [_prep_mat(group_matrices[g]) for g in groups_to_draw]
            vmax_global = float(max(m.max() for m in all_mats)) or 1.0
            for gi, grp in enumerate(groups_to_draw):
                ax = self._tr_fig.add_subplot(nrows, ncols, gi + 1)
                label_kind = "probability" if normalize else "count"
                _draw_one(ax, all_mats[gi], f"{grp}\n({label_kind})", vmax_ref=vmax_global)
            try:
                self._tr_fig.tight_layout(pad=0.8)
            except Exception:
                pass
        else:
            # Pooled: restore canvas to configured max_w × max_h
            default_w = int(self._motif_graph_settings.get("max_w", 700))
            default_h = int(self._motif_graph_settings.get("max_h", 450))
            if self._tr_canvas is not None:
                self._tr_canvas.setFixedSize(default_w, default_h)
            self._tr_fig.set_size_inches(default_w / 100.0, default_h / 100.0)
            stacked = np.stack(list(per_session.values()), axis=0)
            mat = _prep_mat(stacked.mean(axis=0))
            ax = self._tr_fig.add_subplot(111) if ax_root is None else ax_root
            _draw_one(ax, mat, f"Transition Network (max gap={gap_s:.1f}s)")
            try:
                self._tr_fig.tight_layout(pad=1.2)
            except Exception:
                pass

    def _build_tr_stats_lines(self, result: dict[str, Any], normalize: bool) -> list[str]:
        bnames = result.get("bnames", [])
        n = len(bnames)
        per_session = result.get("per_session", {})
        gap_s = result.get("gap_s", 0.0)
        pval_mats_count = result.get("pval_mats_count", result.get("pval_mats", {}))
        pval_mats_prob = result.get("pval_mats_prob", pval_mats_count)
        pval_mats_raw = pval_mats_prob if normalize else pval_mats_count
        pval_mats = {
            k: self._apply_transition_pval_correction(v)
            for k, v in pval_mats_raw.items()
        }
        pval_mode = self._transition_pval_correction_mode()
        group_matrices = result.get("group_matrices", {})
        metric_label = "row-normalized probabilities" if normalize else "raw transition counts"
        corr_label = "FDR-BH" if pval_mode == "fdr_bh" else "None (raw p-values)"
        stat_label = "q" if pval_mode == "fdr_bh" else "p"
        lines: list[str] = [
            f"Transition analysis  |  max gap={gap_s:.1f}s  |  sessions={len(per_session)}",
            f"Behaviors: {', '.join(bnames)}",
            "Axes: rows = first behavior (from), columns = second behavior (to)",
            f"Permutation test per cell (two-sided): |mean_A - mean_B| on session-level {metric_label}.",
            f"Multiple-comparison correction: {corr_label}",
        ]
        for (g1, g2), pmat in pval_mats.items():
            lines.append(f"\nPermutation test ({self._settings.n_permutations} perms): {g1} vs {g2}")
            for i, bna in enumerate(bnames):
                for j, bnb in enumerate(bnames):
                    p = float(pmat[i, j])
                    if p < 0.05:
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                        lines.append(f"  {bna} → {bnb}  {stat_label}={p:.4f} {sig}")
        if not pval_mats and group_matrices:
            lines.append(f"\n{len(group_matrices)} group(s) detected; assign groups in Summary tab to enable permutation tests.")
        return lines

    # ── Motif Discovery panel ────────────────────────────────────────

    def _run_motifs(self) -> None:
        from abel.services.behavioral_motif_service import (
            aggregate_ngrams,
            cluster_sessions,
            extract_ngrams_from_sequences,
            filter_overlapping_events,
            ngram_group_comparison,
        )
        if not self._host._raw_bouts:
            self._mo_stats_text = "Load analytics data first (Refresh Analytics)."
            return
        bids, bnames, bid_to_name = self._get_behavior_ids_and_names()
        if not bids:
            self._mo_stats_text = "No behaviors selected."
            return
        sequences = self._get_sequences_for_analysis()
        if not sequences:
            self._mo_stats_text = "No session data found."
            return

        method = str(self._mo_method_combo.currentData() or "ngram")
        dedup = self._mo_dedup_cb.isChecked()
        self._settings.motif_method = method
        self._save_settings()

        session_group_map = self._build_session_group_map(sequences)
        settings_snap = MotifSettings(
            motif_method=method,
            ngram_min_n=self._settings.ngram_min_n,
            ngram_max_n=self._settings.ngram_max_n,
            ngram_top_k=self._settings.ngram_top_k,
            min_ngram_count=self._settings.min_ngram_count,
            umap_n_components=self._settings.umap_n_components,
            umap_n_neighbors=self._settings.umap_n_neighbors,
            umap_min_dist=self._settings.umap_min_dist,
            hdbscan_min_cluster_size=self._settings.hdbscan_min_cluster_size,
            hdbscan_min_samples=self._settings.hdbscan_min_samples,
            cluster_ngram_n=self._settings.cluster_ngram_n,
            n_permutations=self._settings.n_permutations,
            permutation_seed=self._settings.permutation_seed,
            bout_overlap_tolerance_s=self._settings.bout_overlap_tolerance_s,
        )

        # Apply overlap filter first, then optional dedup per session
        overlap_tol = self._settings.bout_overlap_tolerance_s
        if dedup:
            dedup_sequences = {
                sid: self._dedup_consecutive(
                    filter_overlapping_events(evs, overlap_tol)
                )
                for sid, evs in sequences.items()
            }
        else:
            dedup_sequences = {
                sid: filter_overlapping_events(evs, overlap_tol)
                for sid, evs in sequences.items()
            }

        self._mo_run_btn.setEnabled(False)
        self._mo_run_btn.setText("Running\u2026")
        self._status_lbl.setText("Computing motif analysis\u2026")

        # Capture UMAP group filter state on main thread before worker starts
        checked_umap_groups_snap = self._checked_umap_groups()

        def _compute() -> dict[str, Any]:
            out: dict[str, Any] = {
                "method": method,
                "bid_to_name": bid_to_name,
                "bnames": bnames,
                "session_group_map": session_group_map,
            }

            def _corr_from_vectors(vectors: list[list[float]], n_behaviors: int) -> np.ndarray:
                if n_behaviors <= 0:
                    return np.zeros((0, 0), dtype=float)
                if len(vectors) < 2:
                    return np.eye(n_behaviors, dtype=float)
                arr = np.asarray(vectors, dtype=float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = np.corrcoef(arr, rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                np.fill_diagonal(corr, 1.0)
                return corr

            behavior_vectors: dict[str, list[float]] = {}
            for sid, events in dedup_sequences.items():
                counts: dict[str, int] = {bid: 0 for bid in bids}
                total = 0
                for _st, _et, bid in events:
                    if bid in counts:
                        counts[bid] += 1
                        total += 1
                if total <= 0:
                    behavior_vectors[sid] = [0.0 for _ in bids]
                else:
                    behavior_vectors[sid] = [counts[bid] / total for bid in bids]

            pooled_vectors = list(behavior_vectors.values())
            per_group_vectors: dict[str, list[list[float]]] = {}
            for sid, vec in behavior_vectors.items():
                grp = session_group_map.get(sid, "")
                if grp:
                    per_group_vectors.setdefault(grp, []).append(vec)

            out["behavior_correlation"] = {
                "bids": bids,
                "bnames": bnames,
                "per_session_vectors": behavior_vectors,
                "session_group_map": session_group_map,
                "pooled": _corr_from_vectors(pooled_vectors, len(bids)).tolist(),
                "per_group": {
                    grp: _corr_from_vectors(vectors, len(bids)).tolist()
                    for grp, vectors in per_group_vectors.items()
                },
                "group_sizes": {
                    grp: len(vectors) for grp, vectors in per_group_vectors.items()
                },
                "n_sessions": len(behavior_vectors),
            }

            has_groups = bool(session_group_map)
            if method in ("ngram", "both"):
                ngrams_by_session: dict[int, Any] = {}
                for n_val in range(settings_snap.ngram_min_n, settings_snap.ngram_max_n + 1):
                    ngrams_by_session[n_val] = extract_ngrams_from_sequences(
                        dedup_sequences, n_val
                    )
                if has_groups:
                    group_results: dict[int, list] = {}
                    for n_val, ngrams_n in ngrams_by_session.items():
                        group_results[n_val] = ngram_group_comparison(
                            ngrams_n, session_group_map,
                            top_k=settings_snap.ngram_top_k,
                            min_count=settings_snap.min_ngram_count,
                            n_permutations=settings_snap.n_permutations,
                            seed=settings_snap.permutation_seed,
                            behavior_names=bid_to_name,
                        )
                    out["ngram_group_results"] = group_results
                else:
                    agg_results: dict[int, list] = {}
                    for n_val, ngrams_n in ngrams_by_session.items():
                        agg_results[n_val] = aggregate_ngrams(
                            ngrams_n,
                            top_k=settings_snap.ngram_top_k,
                            min_count=settings_snap.min_ngram_count,
                            behavior_names=bid_to_name,
                        )
                    out["ngram_agg_results"] = agg_results
                out["ngrams_by_session"] = ngrams_by_session

            if method in ("sequence_clustering", "both"):
                # Apply UMAP group filter: if the user has pre-filtered groups,
                # only pass those sessions to cluster_sessions.
                cluster_seqs = dedup_sequences
                checked_umap = checked_umap_groups_snap
                if checked_umap:
                    cluster_seqs = {
                        sid: evs for sid, evs in dedup_sequences.items()
                        if session_group_map.get(sid, "") in checked_umap
                        or not session_group_map.get(sid, "")
                    }
                cluster_result = cluster_sessions(cluster_seqs or dedup_sequences, settings_snap)
                cluster_result["session_groups"] = {
                    sid: session_group_map.get(sid, "")
                    for sid in cluster_result["session_ids"]
                }
                out["cluster_result"] = cluster_result
            return out

        worker = TaskWorker(_compute)
        worker.signals.finished.connect(self._on_motifs_done)
        worker.signals.failed.connect(self._on_worker_failed)
        self._pool.start(worker)

    def _on_mo_view_changed(self) -> None:
        """Toggle UMAP group filter visibility and refresh the motif view."""
        view = str(self._mo_view_combo.currentData() or "bar")
        self._umap_group_filter_row.setVisible(view == "cluster")
        self._render_motifs()

    def _populate_umap_group_filter(self, groups: list[str]) -> None:
        """Rebuild the UMAP group checklist; preserve existing check states."""
        prev_checked: set[str] = set()
        for i in range(self._umap_group_list.count()):
            it = self._umap_group_list.item(i)
            if it and it.checkState() == Qt.CheckState.Checked:
                prev_checked.add(it.text())
        prev_had = self._umap_group_list.count() > 0

        self._umap_group_list.blockSignals(True)
        self._umap_group_list.clear()
        for g in sorted(groups):
            it = QListWidgetItem(g)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if prev_had and g in prev_checked:
                it.setCheckState(Qt.CheckState.Checked)
            elif prev_had:
                it.setCheckState(Qt.CheckState.Unchecked)
            else:
                it.setCheckState(Qt.CheckState.Checked)
            self._umap_group_list.addItem(it)
        self._umap_group_list.blockSignals(False)

    def _checked_umap_groups(self) -> set[str]:
        out: set[str] = set()
        for i in range(self._umap_group_list.count()):
            it = self._umap_group_list.item(i)
            if it and it.checkState() == Qt.CheckState.Checked:
                out.add(it.text())
        return out

    def _umap_check_all_groups(self) -> None:
        for i in range(self._umap_group_list.count()):
            it = self._umap_group_list.item(i)
            if it:
                it.setCheckState(Qt.CheckState.Checked)
        self._render_motifs()

    def _umap_uncheck_all_groups(self) -> None:
        for i in range(self._umap_group_list.count()):
            it = self._umap_group_list.item(i)
            if it:
                it.setCheckState(Qt.CheckState.Unchecked)
        self._render_motifs()

    def _on_motifs_done(self, result: dict[str, Any]) -> None:
        self._mo_run_btn.setEnabled(True)
        self._mo_run_btn.setText("Run")
        self._motif_result = result
        # Populate UMAP group filter from cluster result
        cluster_result = result.get("cluster_result")
        if cluster_result:
            groups = sorted({
                g for g in cluster_result.get("session_groups", {}).values() if g
            })
            if groups:
                self._populate_umap_group_filter(groups)
        self._render_motifs()
        self._status_lbl.setText("Motif analysis complete.")

    def _render_motifs(self) -> None:
        if self._mo_fig is None or not self._motif_result:
            return
        result = self._motif_result
        view = str(self._mo_view_combo.currentData() or "bar")

        # Route cluster/UMAP view
        if view == "cluster":
            self._mo_fig.clear()
            ax = self._mo_fig.add_subplot(111)
            self._render_cluster_on_ax(ax, result)
            try:
                self._mo_fig.tight_layout(pad=1.2)
            except Exception:
                pass
            self._sync_mo_canvas_to_viewport()
            self._mo_canvas.draw_idle()
            self._build_motif_stats(result)
            return

        if view == "corr_heatmap":
            self._mo_fig.clear()
            self._render_behavior_correlation_heatmap(result)
            try:
                self._mo_fig.tight_layout(pad=1.2)
            except Exception:
                pass
            self._sync_mo_canvas_to_viewport()
            self._mo_canvas.draw_idle()
            self._build_motif_stats(result)
            return

        if view == "corr_network":
            self._mo_fig.clear()
            self._render_behavior_correlation_network(result)
            try:
                self._mo_fig.tight_layout(pad=1.2)
            except Exception:
                pass
            self._sync_mo_canvas_to_viewport()
            self._mo_canvas.draw_idle()
            self._build_motif_stats(result)
            return

        # Pull n-gram data
        items_all = self._get_all_ngram_items(result)
        if not items_all:
            self._mo_fig.clear()
            self._mo_fig.text(0.5, 0.5, "No motif data.\nRun analysis first.",
                              ha="center", va="center", fontsize=10)
            self._mo_canvas.draw_idle()
            self._build_motif_stats(result)
            return

        self._mo_fig.clear()
        if view == "bar":
            ax = self._mo_fig.add_subplot(111)
            self._render_ngram_bar(ax, items_all, result)
        elif view == "grouped_bar":
            ax = self._mo_fig.add_subplot(111)
            self._render_ngram_grouped_bar(ax, items_all, result)
        elif view == "engram":
            self._render_engram_view(items_all, result)
        try:
            self._mo_fig.tight_layout(pad=1.2)
        except Exception:
            pass
        if self._motif_graph_settings.get("force_fit", False):
            mgs = self._motif_graph_settings
            _force_fit_canvas(
                self._mo_canvas, self._mo_fig,
                int(mgs.get("max_w", 700)), int(mgs.get("max_h", 450)),
            )
        elif view in ("bar", "grouped_bar"):
            # Bar views fit the viewport height instead of ballooning.
            _autofill_canvas(
                getattr(self, "_mo_canvas_scroll", None), self._mo_canvas, self._mo_fig,
                dpi=100, max_h=self._mo_canvas_scroll.viewport().height(),
            )
        elif view == "engram":
            # One row per motif: fill width but keep the designed per-row height
            # (don't let the width-fill amplify it into a giant scroll figure).
            _autofill_canvas(
                getattr(self, "_mo_canvas_scroll", None), self._mo_canvas, self._mo_fig,
                dpi=100, preserve_aspect=False,
            )
        else:
            self._sync_mo_canvas_to_viewport()
        self._mo_canvas.draw_idle()
        self._build_motif_stats(result)

    def _get_all_ngram_items(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten ngram_agg_results or ngram_group_results into a single sorted list."""
        items_all: list[dict[str, Any]] = []
        for src in ("ngram_agg_results", "ngram_group_results"):
            data_dict = result.get(src, {})
            for n_val, items in sorted(data_dict.items()):
                items_all.extend(items)
        if items_all:
            items_all.sort(key=lambda x: -x.get("total", 0))
        return items_all[: self._settings.ngram_top_k]

    def _render_behavior_correlation_heatmap(self, result: dict[str, Any]) -> None:
        gs = self._host._graph_settings
        corr_data = result.get("behavior_correlation") or {}
        bnames: list[str] = list(corr_data.get("bnames") or [])
        if not bnames:
            ax = self._mo_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No behavior-correlation data.\nRun analysis first.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.axis("off")
            return

        pooled = np.asarray(corr_data.get("pooled") or [], dtype=float)
        group_mats = {
            str(g): np.asarray(mat, dtype=float)
            for g, mat in (corr_data.get("per_group") or {}).items()
        }
        group_sizes: dict[str, int] = {
            str(g): int(n) for g, n in (corr_data.get("group_sizes") or {}).items()
        }

        mats: list[tuple[str, np.ndarray]] = []
        if pooled.size:
            mats.append((f"Pooled (n={int(corr_data.get('n_sessions', 0))})", pooled))
        for grp in sorted(group_mats):
            gmat = group_mats[grp]
            mats.append((f"{grp} (n={group_sizes.get(grp, 0)})", gmat))

        if not mats:
            ax = self._mo_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No correlation matrices available.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.axis("off")
            return

        n_panels = len(mats)
        ncols = min(3, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        short_labels = [name[:14] if len(name) > 14 else name for name in bnames]

        for idx, (title, mat) in enumerate(mats):
            ax = self._mo_fig.add_subplot(nrows, ncols, idx + 1)
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            self._mo_fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(len(short_labels)))
            ax.set_yticks(range(len(short_labels)))
            ax.set_xticklabels(short_labels, rotation=45, ha="right",
                               fontsize=max(6, gs["tick_fontsize"] - 2))
            ax.set_yticklabels(short_labels, fontsize=max(6, gs["tick_fontsize"] - 2))
            ax.set_title(title, fontsize=max(8, gs["title_fontsize"] - 1))

            if len(short_labels) <= 10:
                for i in range(len(short_labels)):
                    for j in range(len(short_labels)):
                        v = float(mat[i, j])
                        if abs(v) >= 0.35 or i == j:
                            tc = "white" if abs(v) > 0.6 else "black"
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                    fontsize=7, color=tc)

        self._mo_fig.suptitle(
            "Behavior Correlation Across Sessions and Groups",
            fontsize=gs["title_fontsize"], y=0.995,
        )

    def _render_behavior_correlation_network(self, result: dict[str, Any]) -> None:
        gs = self._host._graph_settings
        corr_data = result.get("behavior_correlation") or {}
        bnames: list[str] = list(corr_data.get("bnames") or [])
        if len(bnames) < 2:
            ax = self._mo_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Need at least 2 behaviors for a correlation network.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.axis("off")
            return

        pooled = np.asarray(corr_data.get("pooled") or [], dtype=float)
        if pooled.size == 0:
            ax = self._mo_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No pooled correlation data.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.axis("off")
            return

        group_mats = {
            str(g): np.asarray(mat, dtype=float)
            for g, mat in (corr_data.get("per_group") or {}).items()
        }
        panel_mats: list[tuple[str, np.ndarray]] = [("Pooled", pooled)]
        for grp in sorted(group_mats):
            panel_mats.append((grp, group_mats[grp]))

        threshold = 0.35
        n_panels = len(panel_mats)
        ncols = min(3, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        angles = np.linspace(0.0, 2.0 * np.pi, len(bnames), endpoint=False)
        xs = np.cos(angles)
        ys = np.sin(angles)

        for idx, (title, mat) in enumerate(panel_mats):
            ax = self._mo_fig.add_subplot(nrows, ncols, idx + 1)
            ax.set_aspect("equal")

            # Draw weaker edges first so stronger links remain visible on top.
            edges: list[tuple[float, int, int]] = []
            for i in range(len(bnames)):
                for j in range(i + 1, len(bnames)):
                    corr = float(mat[i, j])
                    if abs(corr) >= threshold:
                        edges.append((abs(corr), i, j))
            edges.sort()

            for _w, i, j in edges:
                corr = float(mat[i, j])
                color = "#2e7d32" if corr >= 0 else "#c62828"
                lw = 0.8 + 3.2 * abs(corr)
                alpha = min(0.9, 0.2 + abs(corr))
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=color, linewidth=lw, alpha=alpha, zorder=1)

            for i, bname in enumerate(bnames):
                ax.scatter([xs[i]], [ys[i]], s=240, color=_PALETTE[i % len(_PALETTE)],
                           edgecolors="white", linewidths=0.9, zorder=3)
                lx, ly = xs[i] * 1.2, ys[i] * 1.2
                ha = "left" if xs[i] > 0.1 else ("right" if xs[i] < -0.1 else "center")
                va = "bottom" if ys[i] > 0.1 else ("top" if ys[i] < -0.1 else "center")
                ax.text(lx, ly, bname[:14], ha=ha, va=va,
                        fontsize=max(6, gs["tick_fontsize"] - 2), color="#e0e0e0")

            ax.set_xlim(-1.35, 1.35)
            ax.set_ylim(-1.35, 1.35)
            ax.axis("off")
            ax.set_title(f"{title} | |r| >= {threshold:.2f}",
                         fontsize=max(8, gs["title_fontsize"] - 1))

        self._mo_fig.suptitle(
            "Behavior Relationship Network (green=positive, red=negative)",
            fontsize=gs["title_fontsize"], y=0.995,
        )

    def _render_ngram_bar(self, ax: Any, items: list[dict[str, Any]], result: dict[str, Any]) -> None:
        """Simple horizontal bar chart of top motif counts."""
        gs = self._host._graph_settings
        labels = [it["motif_label"] for it in items]
        counts = [it["total"] for it in items]
        n_val_colors = [len(it.get("motif", ())) for it in items]
        n_min = min(n_val_colors) if n_val_colors else 2
        colors = [_PALETTE[(nv - n_min) % len(_PALETTE)] for nv in n_val_colors]
        y = np.arange(len(labels))
        ax.barh(y, counts, color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=max(6, gs["tick_fontsize"] - 1))
        ax.invert_yaxis()
        ax.set_xlabel("Occurrence Count", fontsize=gs["axis_fontsize"])
        ax.set_title(
            f"Top Behavior Motifs "
            f"({self._settings.ngram_min_n}\u2013{self._settings.ngram_max_n}-grams, pooled)",
            fontsize=gs["title_fontsize"],
        )

    def _render_ngram_grouped_bar(
        self, ax: Any, items: list[dict[str, Any]], result: dict[str, Any]
    ) -> None:
        """Grouped bar chart with mean ± configurable error bars per group."""
        gs_host = self._host._graph_settings
        mgs = self._motif_graph_settings
        session_group_map: dict[str, str] = result.get("session_group_map", {})
        groups = sorted({g for g in session_group_map.values() if g})
        if not groups:
            # Fall back to simple bar if no groups
            self._render_ngram_bar(ax, items, result)
            return

        error_style = mgs.get("error_style", "SEM")
        bar_spacing  = float(mgs.get("bar_spacing", 1.0))
        show_pts     = mgs.get("show_indiv_points", False)
        _capsize     = int(mgs.get("eb_capsize", 4)) if error_style != "None" else 0
        _eblw        = float(mgs.get("eb_linewidth", 1.0))

        labels = [it["motif_label"] for it in items]
        x = np.arange(len(labels))
        width = min(0.80, 0.70 * bar_spacing) / max(len(groups), 1)

        rng = np.random.default_rng(1)
        # Track the highest rendered point (bar + error bar) per item index so
        # significance stars can be placed safely above everything.
        bar_tops: dict[int, float] = {}
        for gi, grp in enumerate(groups):
            means: list[float] = []
            ebs:   list[float] = []
            all_vals: list[np.ndarray] = []
            for it in items:
                per_sess = it.get("per_session", {})
                grp_vals = np.array([float(v) for sid, v in per_sess.items()
                                     if session_group_map.get(sid) == grp], dtype=float)
                if len(grp_vals):
                    means.append(float(grp_vals.mean()))
                    ebs.append(_eb_val(grp_vals, error_style))
                else:
                    means.append(0.0)
                    ebs.append(0.0)
                all_vals.append(grp_vals)
            for xi in range(len(items)):
                top = means[xi] + (ebs[xi] if error_style != "None" else 0.0)
                if top > bar_tops.get(xi, 0.0):
                    bar_tops[xi] = top
            offset = (gi - len(groups) / 2 + 0.5) * width
            color = _PALETTE[gi % len(_PALETTE)]
            ax.bar(
                x + offset, means,
                width=width * 0.9,
                yerr=(ebs if error_style != "None" else None),
                capsize=_capsize,
                label=grp, color=color,
                alpha=0.85,
                error_kw={"elinewidth": _eblw, "capthick": _eblw} if error_style != "None" else {},
            )
            # Individual session dots
            if show_pts:
                for xi, vals in enumerate(all_vals):
                    if len(vals):
                        jitter = rng.uniform(-0.10, 0.10, len(vals)) * bar_spacing
                        ax.scatter(np.full(len(vals), x[xi] + offset) + jitter, vals,
                                   color="white", edgecolors=color, linewidths=0.8,
                                   s=22, zorder=5)

        # Significance stars — drawn above the tallest bar+error across all groups
        _show_stats = mgs.get("show_stats", True)
        for xi, it in enumerate(items):
            if not _show_stats:
                break
            pval_pairs = it.get("pval_pairs") or {}
            if pval_pairs:
                sig = self._sig_for_pairs(pval_pairs)
            else:
                # Fallback: use legacy single pval
                pval = it.get("pval", 1.0)
                sig = ("***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                       if pval < 0.05 else "")
            if sig:
                top = bar_tops.get(xi, 0.0)
                gap = top * 0.04 + 0.10
                ax.text(x[xi], top + gap, sig,
                        ha="center", va="bottom",
                        fontsize=10, color="black", fontweight="bold", zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           fontsize=max(6, gs_host["tick_fontsize"] - 2))
        eb_lbl = f" \u00b1 {error_style}" if error_style != "None" else ""
        ax.set_ylabel(f"Mean Count per Session{eb_lbl}", fontsize=gs_host["axis_fontsize"])
        ax.set_title("Motif Frequency per Group (mean \u00b1 SEM)" if error_style == "SEM" else
                     f"Motif Frequency per Group (mean{eb_lbl})", fontsize=gs_host["title_fontsize"])
        ax.legend(fontsize="x-small")

    def _render_engram_view(
        self, items: list[dict[str, Any]], result: dict[str, Any]
    ) -> None:
        """Engram sequence view.

        Each row = one motif: colored behavior-block strip on the left +
        horizontal grouped bar chart (mean ± configurable error bars per group) on the right.
        """
        gs = self._host._graph_settings
        mgs = self._motif_graph_settings
        _engram_error_style = mgs.get("error_style", "SEM")
        session_group_map: dict[str, str] = result.get("session_group_map", {})
        groups = sorted({g for g in session_group_map.values() if g})
        bid_to_name: dict[str, str] = result.get("bid_to_name", {})

        # Stable per-behavior color
        all_bids: list[str] = list({b for it in items for b in it.get("motif", ())})
        bid_color: dict[str, str] = {bid: _PALETTE[i % len(_PALETTE)] for i, bid in enumerate(all_bids)}

        n_rows = len(items)
        has_groups = bool(groups)

        # Scale figure height so rows don't overlap
        row_h = 0.55  # inches per row
        fig_h = max(3.5, n_rows * row_h + 0.8)
        self._mo_fig.set_size_inches(self._mo_fig.get_figwidth(), fig_h)

        from matplotlib.gridspec import GridSpec  # type: ignore[import-untyped]
        import matplotlib.patches as mpatches  # type: ignore[import-untyped]

        ncols = 2 if has_groups else 1
        gs_layout = GridSpec(
            n_rows, ncols, figure=self._mo_fig,
            width_ratios=[0.38, 0.62] if has_groups else [1.0],
            hspace=0.06, wspace=0.04,
            left=0.01, right=0.97, top=0.96, bottom=0.04,
        )

        # Find a shared x-max across all bars for a uniform scale
        x_max = 0.1
        if has_groups:
            for it in items:
                per_sess = it.get("per_session", {})
                for grp in groups:
                    vals = [float(v) for sid, v in per_sess.items()
                            if session_group_map.get(sid) == grp]
                    if vals:
                        arr = np.array(vals)
                        x_max = max(x_max, float(arr.mean()) + (arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0))
            x_max *= 1.25  # headroom

        for row_i, it in enumerate(items):
            motif = it.get("motif", ())
            is_last = row_i == n_rows - 1

            # ── block panel ────────────────────────────────────────────
            ax_block = self._mo_fig.add_subplot(gs_layout[row_i, 0])
            block_w = 1.0 / max(len(motif), 1)
            for col_i, bid in enumerate(motif):
                color = bid_color.get(bid, "#888888")
                rect = mpatches.FancyBboxPatch(
                    (col_i * block_w + 0.01, 0.08), block_w - 0.02, 0.84,
                    boxstyle="round,pad=0.02",
                    facecolor=color, edgecolor="none", alpha=0.9,
                )
                ax_block.add_patch(rect)
                label_txt = bid_to_name.get(bid, bid)[:10]
                ax_block.text(
                    (col_i + 0.5) * block_w, 0.5, label_txt,
                    ha="center", va="center",
                    fontsize=max(5, gs["tick_fontsize"] - 2),
                    color="white", fontweight="bold",
                )
                if col_i < len(motif) - 1:
                    ax_block.text(
                        (col_i + 1) * block_w - 0.005, 0.5, "\u2192",
                        ha="center", va="center", fontsize=7, color="#cccccc",
                    )
            ax_block.set_xlim(0, 1)
            ax_block.set_ylim(0, 1)
            ax_block.axis("off")
            if row_i == 0:
                ax_block.set_title(
                    "Motif Sequence", fontsize=gs["tick_fontsize"],
                    pad=2, loc="left",
                )

            # ── abundance panel ────────────────────────────────────────
            if has_groups:
                ax_bar = self._mo_fig.add_subplot(gs_layout[row_i, 1])
                means_g: list[float] = []
                sems_g: list[float] = []
                for grp in groups:
                    per_sess = it.get("per_session", {})
                    vals = [float(v) for sid, v in per_sess.items()
                            if session_group_map.get(sid) == grp]
                    if vals:
                        arr = np.array(vals)
                        means_g.append(float(arr.mean()))
                        sems_g.append(_eb_val(arr, _engram_error_style))
                    else:
                        means_g.append(0.0)
                        sems_g.append(0.0)

                colors_g = [_PALETTE[gi % len(_PALETTE)] for gi in range(len(groups))]
                y_pos = np.arange(len(groups))
                _eng_cs = int(mgs.get("eb_capsize", 4)) if _engram_error_style != "None" else 0
                _eng_lw = float(mgs.get("eb_linewidth", 1.0))
                ax_bar.barh(
                    y_pos, means_g,
                    xerr=(sems_g if _engram_error_style != "None" else None),
                    color=colors_g, alpha=0.85,
                    height=0.65,
                    capsize=_eng_cs,
                    error_kw={"elinewidth": _eng_lw, "capthick": _eng_lw} if _engram_error_style != "None" else {},
                )
                ax_bar.set_xlim(0, x_max)
                ax_bar.set_yticks(y_pos)
                ax_bar.set_yticklabels(
                    [g[:10] for g in groups],
                    fontsize=max(5, gs["tick_fontsize"] - 2),
                )
                ax_bar.invert_yaxis()
                # x ticks: only show on the last row
                if is_last:
                    ax_bar.tick_params(axis="x", labelsize=max(5, gs["tick_fontsize"] - 2))
                    ax_bar.set_xlabel("Mean count / session", fontsize=max(6, gs["axis_fontsize"] - 2))
                else:
                    ax_bar.tick_params(axis="x", labelbottom=False)
                ax_bar.spines["top"].set_visible(False)
                ax_bar.spines["right"].set_visible(False)
                if row_i == 0:
                    ax_bar.set_title(
                        "Abundance per Group (mean ± SEM)",
                        fontsize=gs["tick_fontsize"], pad=2, loc="left",
                    )
                # significance star at right edge if p < 0.05
                pval_pairs = it.get("pval_pairs") or {}
                if pval_pairs:
                    sig = self._sig_for_pairs(pval_pairs)
                else:
                    pval = it.get("pval", 1.0)
                    sig = ("***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                           if pval < 0.05 else "")
                if sig and self._motif_graph_settings.get("show_stats", True):
                    ax_bar.text(
                        1.01, 0.5, sig, transform=ax_bar.transAxes,
                        ha="left", va="center", fontsize=9, color="black",
                        fontweight="bold", clip_on=False,
                    )
            else:
                # No groups — just show the total count as text overlay on block panel
                ax_block.text(
                    0.99, 0.08, f"\u00d7{it['total']}",
                    ha="right", va="bottom", transform=ax_block.transAxes,
                    fontsize=max(6, gs["tick_fontsize"] - 1), color="#90a4ae",
                )

    def _build_motif_stats(self, result: dict[str, Any]) -> None:
        lines: list[str] = []
        for src in ("ngram_agg_results", "ngram_group_results"):
            data_dict = result.get(src, {})
            for n_val, items in sorted(data_dict.items()):
                lines.append(f"\u2014\u2014 {n_val}-grams \u2014\u2014")
                for it in items[:20]:
                    sig = "*" if it.get("pval", 1.0) < 0.05 else ""
                    means_str = "  ".join(
                        f"{g}={v:.2f}" for g, v in (it.get("group_means") or {}).items()
                    )
                    lines.append(
                        f"  {it['motif_label']}  \u00d7{it['total']}"
                        + (f"  {means_str}" if means_str else "")
                        + (f"  p={it.get('pval', float('nan')):.3f}{sig}" if "pval" in it else "")
                    )
        if "cluster_result" in result:
            cr = result["cluster_result"]
            if cr.get("error"):
                lines.append(f"Clustering: {cr['error']}")
            else:
                lines.append(
                    f"Sequence clustering: {cr.get('n_clusters', '?')} clusters, "
                    f"{len(cr.get('session_ids', []))} sessions"
                )
        self._mo_stats_text = "\n".join(lines) if lines else "No results."
        self._mo_stats_btn.setEnabled(True)

    def _render_cluster_on_ax(self, ax: Any, result: dict[str, Any]) -> None:
        cr = result.get("cluster_result")
        gs = self._host._graph_settings
        if cr is None:
            ax.text(0.5, 0.5, "No clustering data.\nRun with method 'Both' or 'Sequence Clustering'.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            return
        if cr.get("error"):
            ax.text(0.5, 0.5, cr["error"], ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, wrap=True)
            return
        embedding = np.array(cr["embedding"])
        labels = np.array(cr["labels"])
        session_ids: list[str] = cr["session_ids"]
        session_groups: dict[str, str] = cr.get("session_groups", {})
        label_map = self._host._session_label_by_session

        if embedding.shape[1] < 2:
            ax.text(0.5, 0.5, "Not enough dimensions for 2-D plot.",
                    ha="center", va="center", transform=ax.transAxes)
            return

        # Apply UMAP group filter (if group filter list has items)
        checked_umap_groups = self._checked_umap_groups()
        if checked_umap_groups:
            keep_indices = [
                i for i, sid in enumerate(session_ids)
                if session_groups.get(sid, "") in checked_umap_groups
                or not session_groups.get(sid, "")  # keep ungrouped sessions
            ]
            if keep_indices:
                embedding = embedding[keep_indices]
                labels = labels[keep_indices]
                session_ids = [session_ids[i] for i in keep_indices]

        # Colour by group first; fall back to cluster label if no groups
        groups = sorted({g for g in session_groups.values() if g})
        group_color: dict[str, str] = {g: _PALETTE[gi % len(_PALETTE)] for gi, g in enumerate(groups)}
        cluster_color: dict[int, str] = {}
        for li, lbl in enumerate(sorted(set(labels.tolist()))):
            cluster_color[lbl] = _PALETTE[li % len(_PALETTE)] if lbl >= 0 else "#888888"

        # Plot each point
        plotted_groups: set[str] = set()
        plotted_clusters: set[int] = set()
        for i, sid in enumerate(session_ids):
            x, y = float(embedding[i, 0]), float(embedding[i, 1])
            grp = session_groups.get(sid, "")
            lbl = int(labels[i])
            if groups:
                color = group_color.get(grp, "#888888")
                marker = "o" if lbl >= 0 else "x"
                leg_label = grp if grp not in plotted_groups else None
                plotted_groups.add(grp)
            else:
                color = cluster_color.get(lbl, "#888888")
                marker = "o"
                leg_label = (f"Cluster {lbl}" if lbl >= 0 else "Noise") if lbl not in plotted_clusters else None
                plotted_clusters.add(lbl)
            ax.scatter(x, y, c=color, marker=marker, s=70, alpha=0.85,
                       edgecolors="white", linewidths=0.5,
                       label=leg_label)
            display_name = label_map.get(sid, sid)
            ax.annotate(
                display_name,
                (x, y),
                fontsize=max(5, gs["tick_fontsize"] - 3),
                xytext=(4, 3), textcoords="offset points", alpha=0.75,
            )

        # Draw cluster convex hulls as background patches (if >2 points per cluster)
        if len(session_ids) >= 4:
            try:
                from scipy.spatial import ConvexHull  # type: ignore[import-untyped]
                import matplotlib.patches as mpatches  # type: ignore[import-untyped]
                from matplotlib.path import Path as MPath  # type: ignore[import-untyped]
                for lbl_val in sorted(set(labels.tolist())):
                    if lbl_val < 0:
                        continue
                    pts = embedding[labels == lbl_val]
                    if len(pts) < 3:
                        continue
                    try:
                        hull = ConvexHull(pts)
                        hull_pts = pts[hull.vertices]
                        poly = mpatches.Polygon(
                            hull_pts, closed=True,
                            facecolor=cluster_color.get(lbl_val, "#888888"),
                            alpha=0.08, edgecolor=cluster_color.get(lbl_val, "#888888"),
                            linewidth=1.0, linestyle="--",
                        )
                        ax.add_patch(poly)
                    except Exception:
                        pass
            except ImportError:
                pass

        ax.set_xlabel("UMAP 1", fontsize=gs["axis_fontsize"])
        ax.set_ylabel("UMAP 2", fontsize=gs["axis_fontsize"])
        n_cl = cr.get("n_clusters", "?")
        dim_note = "UMAP" if embedding.shape[0] > 4 else "PCA"
        ax.set_title(
            f"Session Motif Landscape \u2014 {dim_note} ({n_cl} density clusters)\n"
            f"Each point = one session. Points close together share similar motif profiles.",
            fontsize=gs["title_fontsize"],
        )
        ax.legend(fontsize="x-small", loc="best", framealpha=0.4)

        # Add note if too few sessions for meaningful UMAP
        if len(session_ids) < 8:
            ax.text(
                0.01, 0.01,
                f"Note: only {len(session_ids)} sessions — UMAP may not be meaningful with so few points. "
                "Add more sessions or use n-gram views instead.",
                transform=ax.transAxes,
                fontsize=7, color="#ffcc80", va="bottom", wrap=True,
            )

    # ── HMM panel ────────────────────────────────────────────────────

    def _run_hmm(self) -> None:
        from abel.services.behavioral_motif_service import (
            filter_overlapping_events,
            fit_hmm,
        )

        if not _ensure_hmmlearn():
            self._hmm_stats_text = (
                "hmmlearn is not installed.\n"
                "Go to the Dependencies tab and install it (pip install hmmlearn)."
            )
            self._status_lbl.setText(
                "hmmlearn is not installed. Install it via the Dependencies tab (pip install hmmlearn)."
            )
            return
        if not self._host._raw_bouts:
            self._hmm_stats_text = "Load analytics data first (Refresh Analytics)."
            self._status_lbl.setText("No analytics data loaded. Click Refresh Analytics first.")
            return
        bids, bnames, bid_to_name = self._get_behavior_ids_and_names()
        if not bids:
            self._hmm_stats_text = "No behaviors selected."
            self._status_lbl.setText("No behaviors selected. Use the behavior filter to select at least one.")
            return
        sequences = self._get_sequences_for_analysis()
        if not sequences:
            self._hmm_stats_text = "No session data found."
            self._status_lbl.setText("No session data found. Ensure sessions are checked in the Summary tab.")
            return

        self._settings.hmm_n_states_mode = str(self._hmm_mode_combo.currentData() or "auto")
        self._settings.hmm_n_states = int(self._hmm_n_states_spin.value())
        self._save_settings()

        settings_snap = MotifSettings(
            hmm_n_states_mode=self._settings.hmm_n_states_mode,
            hmm_n_states=self._settings.hmm_n_states,
            hmm_n_states_min=self._settings.hmm_n_states_min,
            hmm_n_states_max=self._settings.hmm_n_states_max,
            hmm_n_iter=self._settings.hmm_n_iter,
            hmm_n_restarts=self._settings.hmm_n_restarts,
            hmm_criterion=self._settings.hmm_criterion,
            bout_overlap_tolerance_s=self._settings.bout_overlap_tolerance_s,
        )
        session_group_map = self._build_session_group_map(sequences)

        # Filter out deeply overlapping concurrent bouts before HMM encoding
        overlap_tol = self._settings.bout_overlap_tolerance_s
        sequences = {
            sid: filter_overlapping_events(evs, overlap_tol)
            for sid, evs in sequences.items()
        }

        self._hmm_run_btn.setEnabled(False)
        self._hmm_run_btn.setText("Running\u2026")
        self._status_lbl.setText("Fitting HMM\u2026 (this may take a moment)")

        def _compute() -> dict[str, Any]:
            hmm_res = fit_hmm(sequences, bids, settings_snap)
            if hmm_res.get("error"):
                return hmm_res
            from abel.services.behavioral_motif_service import state_occupancy

            occ = state_occupancy(hmm_res.get("state_sequences", {}), hmm_res["n_states"])
            n_states = hmm_res["n_states"]
            # Per-group occupancy with SEM
            per_group_occ: dict[str, list[list[float]]] = {}
            for sid, fracs in occ.items():
                grp = session_group_map.get(sid, "")
                if grp:
                    per_group_occ.setdefault(grp, []).append(list(fracs))
            group_occ_mean: dict[str, list[float]] = {}
            group_occ_sem: dict[str, list[float]] = {}
            for grp, all_fracs in per_group_occ.items():
                arr = np.array(all_fracs)
                group_occ_mean[grp] = arr.mean(axis=0).tolist()
                if arr.shape[0] > 1:
                    group_occ_sem[grp] = (arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])).tolist()
                else:
                    group_occ_sem[grp] = [0.0] * n_states
            # Permutation test: compare ALL pairwise group combinations
            groups = sorted(per_group_occ.keys())
            pval_occ: dict[tuple[str, str], np.ndarray] = {}
            from itertools import combinations as _hmm_combos
            for g1, g2 in _hmm_combos(groups, 2):
                arr1 = np.array(per_group_occ[g1])
                arr2 = np.array(per_group_occ[g2])
                pvals = np.ones(n_states)
                rng2 = np.random.default_rng(settings_snap.permutation_seed)
                for st in range(n_states):
                    v1, v2 = arr1[:, st], arr2[:, st]
                    obs = abs(v1.mean() - v2.mean())
                    all_v = np.concatenate([v1, v2])
                    n1 = len(v1)
                    count = sum(
                        1 for _ in range(settings_snap.n_permutations)
                        if abs(
                            all_v[p := rng2.permutation(len(all_v))][:n1].mean()
                            - all_v[p][n1:].mean()
                        ) >= obs
                    )
                    pvals[st] = count / settings_snap.n_permutations
                pval_occ[(g1, g2)] = pvals
            hmm_res["session_occupancy"] = occ
            hmm_res["group_occ_mean"] = group_occ_mean
            hmm_res["group_occ_sem"] = group_occ_sem
            hmm_res["pval_occ"] = {str(k): v.tolist() for k, v in pval_occ.items()}
            hmm_res["bnames"] = bnames
            hmm_res["session_groups"] = session_group_map
            hmm_res["groups"] = groups
            return hmm_res

        worker = TaskWorker(_compute)
        worker.signals.finished.connect(self._on_hmm_done)
        worker.signals.failed.connect(self._on_worker_failed)
        self._pool.start(worker)

    def _on_hmm_done(self, result: dict[str, Any]) -> None:
        self._hmm_run_btn.setEnabled(True)
        self._hmm_run_btn.setText("Run HMM")
        self._hmm_result = result
        if result.get("error"):
            self._hmm_stats_text = f"HMM Error:\n{result['error']}"
            self._hmm_stats_btn.setEnabled(True)
            self._status_lbl.setText("HMM failed.")
            return
        self._render_hmm()
        self._status_lbl.setText(
            f"HMM complete ({result.get('n_states', '?')} hidden states)."
        )

    def _render_hmm(self) -> None:
        result = self._hmm_result
        if not result or result.get("error"):
            return
        if self._hmm_sel_fig is None:
            return

        n_states = result["n_states"]
        bnames: list[str] = result.get("bnames", [])
        gs = self._host._graph_settings
        view = str(self._hmm_view_combo.currentData() or "model_sel")

        self._hmm_sel_fig.clear()

        if view == "model_sel":
            model_sel = result.get("model_selection", [])
            ax = self._hmm_sel_fig.add_subplot(111)
            if model_sel and len(model_sel) > 1:
                ns_vals  = [r["n_states"]         for r in model_sel]
                aic_vals = [r["aic"]               for r in model_sel]
                bic_vals = [r["bic"]               for r in model_sel]
                ax.plot(ns_vals, aic_vals, "o-", label="AIC", color=_PALETTE[0])
                ax.plot(ns_vals, bic_vals, "s-", label="BIC", color=_PALETTE[1])
                ax.axvline(n_states, color="white", linestyle="--",
                           alpha=0.6, label=f"Selected: {n_states}")
                ax.set_xlabel("N states", fontsize=gs["axis_fontsize"])
                ax.set_ylabel("Information Criterion", fontsize=gs["axis_fontsize"])
                ax.set_title("HMM Model Selection", fontsize=gs["title_fontsize"])
                ax.legend(fontsize="x-small")
                ax.tick_params(labelsize=gs["tick_fontsize"])
            else:
                r = model_sel[0] if model_sel else {}
                ax.text(0.5, 0.5,
                        f"N states = {r.get('n_states', n_states)}\n"
                        f"LL = {r.get('log_likelihood', float('nan')):.1f}\n"
                        f"AIC = {r.get('aic', float('nan')):.1f}\n"
                        f"BIC = {r.get('bic', float('nan')):.1f}",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_title("HMM Model (manual / single)", fontsize=gs["title_fontsize"])

        elif view == "emission":
            emit_mat = result.get("emission_matrix")
            ax = self._hmm_sel_fig.add_subplot(111)
            if emit_mat is not None and bnames:
                emit_arr = np.array(emit_mat)
                im = ax.imshow(emit_arr, aspect="auto", cmap="Blues", vmin=0, vmax=1)
                self._hmm_sel_fig.colorbar(im, ax=ax, label="P(behavior | state)", shrink=0.8)
                ax.set_xticks(range(len(bnames)))
                ax.set_xticklabels(bnames, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(n_states))
                ax.set_yticklabels([f"State {i}" for i in range(n_states)], fontsize=8)
                ax.set_title("Emission Probabilities", fontsize=gs["title_fontsize"])
                for i in range(n_states):
                    for j in range(len(bnames)):
                        v = float(emit_arr[i, j])
                        if v > 0.05:
                            tc = "white" if v > 0.6 else "black"
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                    fontsize=7, color=tc)
            else:
                ax.text(0.5, 0.5, "No emission data", ha="center", va="center",
                        transform=ax.transAxes)

        elif view == "occupancy":
            # Grouped bar with configurable error bars + significance markers
            _hmm_mgs = self._motif_graph_settings
            _hmm_error_style = _hmm_mgs.get("error_style", "SEM")
            _hmm_bar_spacing = float(_hmm_mgs.get("bar_spacing", 1.0))
            group_occ_mean: dict[str, list[float]] = result.get("group_occ_mean", {})
            session_occ: dict[str, list[float]] = result.get("session_occupancy", {})
            session_grps_occ: dict[str, str] = result.get("session_groups", {})
            pval_occ_raw: dict[str, Any] = result.get("pval_occ", {})
            groups = result.get("groups", sorted(group_occ_mean.keys()))
            ax = self._hmm_sel_fig.add_subplot(111)
            x = np.arange(n_states)
            width = min(0.80, 0.70 * _hmm_bar_spacing) / max(len(groups), 1)
            # Build per-group raw occupancy arrays for _eb_val
            grp_occ_arrays: dict[str, np.ndarray] = {}
            for grp in groups:
                rows = [v for sid, v in session_occ.items()
                        if session_grps_occ.get(sid) == grp]
                grp_occ_arrays[grp] = np.array(rows) if rows else np.zeros((0, n_states))
            for gi, grp in enumerate(groups):
                means = np.array(group_occ_mean.get(grp, [0.0] * n_states))
                occ_arr = grp_occ_arrays.get(grp, np.zeros((0, n_states)))
                ebs = np.array([
                    _eb_val(occ_arr[:, st] if occ_arr.shape[0] > 0 else np.array([0.0]),
                            _hmm_error_style)
                    for st in range(n_states)
                ])
                _hmm_cs = int(_hmm_mgs.get("eb_capsize", 4)) if _hmm_error_style != "None" else 0
                _hmm_lw = float(_hmm_mgs.get("eb_linewidth", 1.0))
                offset = (gi - len(groups) / 2 + 0.5) * width * _hmm_bar_spacing
                ax.bar(x + offset, means,
                       width=width * 0.9 * _hmm_bar_spacing,
                       yerr=(ebs if _hmm_error_style != "None" else None),
                       label=grp, color=_PALETTE[gi % len(_PALETTE)],
                       alpha=0.85,
                       capsize=_hmm_cs,
                       error_kw={"elinewidth": _hmm_lw, "capthick": _hmm_lw} if _hmm_error_style != "None" else {})
            # Add significance stars if pval_occ available (filtered by selected comparisons)
            active_pval_occ = self._pval_mat_filtered(
                {k: np.array(v) for k, v in pval_occ_raw.items()}
            )
            _hmm_show_stats = self._motif_graph_settings.get("show_stats", True)
            for pair_key, pvals_arr in active_pval_occ.items():
                if not _hmm_show_stats:
                    break
                for st in range(n_states):
                    p = float(pvals_arr[st])
                    if p < 0.05:
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                        max_bar = max(
                            (group_occ_mean.get(g, [0.0]*n_states)[st] +
                             _eb_val(
                                 grp_occ_arrays.get(g, np.zeros((0, n_states)))[:, st]
                                 if grp_occ_arrays.get(g, np.zeros((0, n_states))).shape[0] > 0
                                 else np.array([0.0]),
                                 _hmm_error_style
                             ))
                            for g in groups
                        ) if groups else 0.0
                        _hmm_sig_gap = max_bar * 0.04 + 0.02
                        ax.text(st, max_bar + _hmm_sig_gap, sig,
                                ha="center", va="bottom",
                                fontsize=10, color="black", fontweight="bold", zorder=6)
            ax.set_xticks(x)
            ax.set_xticklabels([f"State {i}" for i in range(n_states)],
                               fontsize=gs["tick_fontsize"])
            ax.set_ylabel("Mean Occupancy Fraction", fontsize=gs["axis_fontsize"])
            ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.2 + 0.05))
            _hmm_eb_lbl = f" \u00b1 {_hmm_error_style}" if _hmm_error_style != "None" else ""
            ax.set_title(f"State Occupancy per Group (mean{_hmm_eb_lbl})",
                         fontsize=gs["title_fontsize"])
            if groups:
                ax.legend(fontsize="x-small")

        elif view == "trans_hmm":
            trans_mat = result.get("transition_matrix")
            ax = self._hmm_sel_fig.add_subplot(111)
            if trans_mat is not None:
                tm = np.array(trans_mat)
                im = ax.imshow(tm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
                self._hmm_sel_fig.colorbar(im, ax=ax, label="Transition Probability", shrink=0.8)
                state_labels = [f"S{i}" for i in range(n_states)]
                ax.set_xticks(range(n_states)); ax.set_xticklabels(state_labels, fontsize=9)
                ax.set_yticks(range(n_states)); ax.set_yticklabels(state_labels, fontsize=9)
                ax.set_title("HMM State Transition Matrix", fontsize=gs["title_fontsize"])
                for i in range(n_states):
                    for j in range(n_states):
                        v = float(tm[i, j])
                        if v > 0.05:
                            tc = "white" if v > 0.6 else "black"
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                    fontsize=9, color=tc)
            else:
                ax.text(0.5, 0.5, "No transition matrix data", ha="center", va="center",
                        transform=ax.transAxes)

        try:
            self._hmm_sel_fig.tight_layout(pad=1.0)
        except Exception:
            pass
        # Bar/line views shouldn't grow taller than the viewport; heatmaps
        # (emission, state-transition) keep their aspect ratio.
        if view in ("occupancy", "model_sel"):
            _autofill_canvas(
                getattr(self, "_hmm_canvas_scroll", None),
                self._hmm_sel_canvas, self._hmm_sel_fig,
                dpi=100, max_h=self._hmm_canvas_scroll.viewport().height(),
            )
        else:
            self._sync_hmm_canvas_to_viewport()
        self._hmm_sel_canvas.draw_idle()

        # Stats text
        model_sel = result.get("model_selection", [])
        sel_criterion = self._settings.hmm_criterion.upper()
        criterion_reason = ""
        if model_sel and len(model_sel) > 1:
            crit_vals = [r[self._settings.hmm_criterion] for r in model_sel if self._settings.hmm_criterion in r]
            if crit_vals:
                best_idx = int(np.argmin(crit_vals))
                best_n = model_sel[best_idx]["n_states"]
                criterion_reason = (
                    f" \u2014 selected N={best_n} because it has the lowest {sel_criterion} "
                    f"({crit_vals[best_idx]:.1f}). Lower {sel_criterion} = better balance of "
                    f"model fit vs. complexity."
                )

        lines: list[str] = [
            f"HMM  |  {n_states} hidden states (auto-selected by {sel_criterion}){criterion_reason}",
            f"Log-likelihood={result.get('log_likelihood', float('nan')):.2f}  "
            f"AIC={result.get('aic', float('nan')):.2f}  "
            f"BIC={result.get('bic', float('nan')):.2f}",
            "",
            "AIC/BIC graph: Each line shows the information criterion at different state "
            "counts. Lower = better model. The dashed vertical line marks the selected N. "
            "AIC rewards fit more; BIC penalises complexity more heavily. "
            "Use 'Emission heatmap' to see what each hidden state represents "
            "(which behaviors it emits). Use 'State occupancy' to compare how much "
            "time each group spends in each state.",
            "",
        ]
        pval_occ_raw = result.get("pval_occ", {})
        for pair_str, pvals_lst in pval_occ_raw.items():
            pvals = np.array(pvals_lst)
            lines.append(f"Occupancy permutation test ({pair_str}):")
            for st in range(n_states):
                p = float(pvals[st])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                lines.append(f"  State {st}: p={p:.4f} ({sig})")
        group_occ_mean: dict[str, list[float]] = result.get("group_occ_mean", {})
        if group_occ_mean:
            lines.append("\nGroup mean occupancy:")
            for grp, fracs in sorted(group_occ_mean.items()):
                frac_s = "  ".join(f"S{i}:{fracs[i]:.3f}" for i in range(len(fracs)))
                lines.append(f"  {grp}: {frac_s}")
        self._hmm_stats_text = "\n".join(lines)
        self._hmm_stats_btn.setEnabled(True)

    # ── Stats popup dialogs ───────────────────────────────────────────

    def _show_stats_popup(self, title: str, text: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(580, 400)
        layout = QVBoxLayout(dlg)
        te = QTextEdit(dlg)
        te.setReadOnly(True)
        te.setPlainText(text or "(No results yet — run analysis first.)")
        te.setStyleSheet(
            "QTextEdit{background:#0A1929;color:#cfd8dc;font-family:Consolas,monospace;"
            "font-size:11px;border:1px solid #1E3A5F;border-radius:4px;}"
        )
        layout.addWidget(te, 1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        dlg.exec()

    def _show_tr_stats_popup(self) -> None:
        self._show_stats_popup("Transition Statistics", self._tr_stats_text)

    def _show_mo_stats_popup(self) -> None:
        self._show_stats_popup("Motif Statistics", self._mo_stats_text)

    def _show_hmm_stats_popup(self) -> None:
        self._show_stats_popup("HMM Statistics", self._hmm_stats_text)

    # ── Graph size dialog ─────────────────────────────────────────────

    def _open_motif_graph_size_dialog(self) -> None:
        """Open a dialog to set display size and chart style settings for motif analysis canvases."""
        gs = self._motif_graph_settings
        dlg = QDialog(self)
        dlg.setWindowTitle("Motif Graph Settings")
        dlg.resize(340, 400)
        form = QFormLayout()

        max_w_spin = QSpinBox(dlg)
        max_w_spin.setRange(200, 3840)
        max_w_spin.setSingleStep(50)
        max_w_spin.setSuffix(" px")
        max_w_spin.setToolTip(
            "Default canvas width used by Motif Discovery and HMM Analysis panels,\n"
            "and the Transition Matrix when viewing a single (pooled) network."
        )
        max_w_spin.setValue(int(gs.get("max_w", 700)))

        max_h_spin = QSpinBox(dlg)
        max_h_spin.setRange(150, 2160)
        max_h_spin.setSingleStep(50)
        max_h_spin.setSuffix(" px")
        max_h_spin.setToolTip("Default canvas height for the canvases above.")
        max_h_spin.setValue(int(gs.get("max_h", 450)))

        cell_spin = QSpinBox(dlg)
        cell_spin.setRange(200, 1200)
        cell_spin.setSingleStep(20)
        cell_spin.setSuffix(" px")
        cell_spin.setToolTip(
            "Pixel size of each individual subplot cell when the Transition Matrix\n"
            "Network / chord view is displayed Per-group.\n"
            "The canvas is automatically sized to (cols \u00d7 cell) \u00d7 (rows \u00d7 cell)\n"
            "and the scroll area handles any overflow."
        )
        cell_spin.setValue(int(gs.get("net_cell_px", 380)))

        error_style_combo = QComboBox(dlg)
        for es in ("SEM", "SD", "95% CI", "None"):
            error_style_combo.addItem(es, userData=es)
        error_style_combo.setCurrentText(gs.get("error_style", "SEM"))
        error_style_combo.setToolTip(
            "Error bar style for Motif Discovery grouped bar charts.\n"
            "SEM = standard error, SD = standard deviation,\n"
            "95% CI = 1.96 \u00d7 SEM, None = no error bars."
        )

        bar_spacing_spin = QDoubleSpinBox(dlg)
        bar_spacing_spin.setRange(0.3, 2.0)
        bar_spacing_spin.setSingleStep(0.1)
        bar_spacing_spin.setDecimals(1)
        bar_spacing_spin.setValue(float(gs.get("bar_spacing", 1.0)))
        bar_spacing_spin.setToolTip("Multiplier for bar widths (1.0 = default).")

        eb_capsize_spin = QSpinBox(dlg)
        eb_capsize_spin.setRange(0, 20)
        eb_capsize_spin.setValue(int(gs.get("eb_capsize", 4)))
        eb_capsize_spin.setSuffix(" pt")
        eb_capsize_spin.setToolTip("Width of the horizontal caps at the top and bottom of each error bar.")

        eb_lw_spin = QDoubleSpinBox(dlg)
        eb_lw_spin.setRange(0.2, 6.0)
        eb_lw_spin.setSingleStep(0.2)
        eb_lw_spin.setDecimals(1)
        eb_lw_spin.setValue(float(gs.get("eb_linewidth", 1.0)))
        eb_lw_spin.setSuffix(" pt")
        eb_lw_spin.setToolTip("Thickness of the vertical error bar lines.")

        indiv_points_check = QCheckBox("Overlay individual subject points", dlg)
        indiv_points_check.setChecked(gs.get("show_indiv_points", False))
        indiv_points_check.setToolTip("Jittered scatter dots showing each session's value on motif bar charts.")

        show_stats_check = QCheckBox("Show statistics on graph", dlg)
        show_stats_check.setChecked(gs.get("show_stats", True))
        show_stats_check.setToolTip(
            "Overlay significance stars (p-values) on transition, motif and HMM\n"
            "bar charts. Uncheck to hide them; the stats popups are unaffected."
        )

        force_fit_check = QCheckBox("Force fit to canvas size", dlg)
        force_fit_check.setChecked(gs.get("force_fit", False))
        force_fit_check.setToolTip(
            "After rendering, auto-resize the canvas to match the content's\n"
            "tight bounding box (up to the width/height limits above)."
        )

        form.addRow("Canvas width (pooled):", max_w_spin)
        form.addRow("Canvas height (pooled):", max_h_spin)
        form.addRow("Network cell size:", cell_spin)
        form.addRow("Error bar style:", error_style_combo)
        form.addRow("Bar spacing:", bar_spacing_spin)
        form.addRow("Error bar cap width:", eb_capsize_spin)
        form.addRow("Error bar line thickness:", eb_lw_spin)
        form.addRow(indiv_points_check)
        form.addRow(show_stats_check)
        form.addRow(force_fit_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout = QVBoxLayout(dlg)
        layout.addLayout(form)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        gs["max_w"] = max_w_spin.value()
        gs["max_h"] = max_h_spin.value()
        gs["net_cell_px"] = cell_spin.value()
        gs["error_style"] = str(error_style_combo.currentData() or "SEM")
        gs["bar_spacing"] = bar_spacing_spin.value()
        gs["eb_capsize"] = eb_capsize_spin.value()
        gs["eb_linewidth"] = eb_lw_spin.value()
        gs["show_indiv_points"] = indiv_points_check.isChecked()
        gs["show_stats"] = show_stats_check.isChecked()
        gs["force_fit"] = force_fit_check.isChecked()
        # Apply the default size to heatmap/motif/HMM canvases.
        # The transition network canvas is resized on next render based on cell_px.
        for canvas, fig in (
            (getattr(self, "_tr_canvas", None),      getattr(self, "_tr_fig", None)),
            (getattr(self, "_mo_canvas", None),      getattr(self, "_mo_fig", None)),
            (getattr(self, "_hmm_sel_canvas", None), getattr(self, "_hmm_sel_fig", None)),
        ):
            if canvas is not None:
                canvas.setFixedSize(gs["max_w"], gs["max_h"])
            if fig is not None:
                fig.set_size_inches(gs["max_w"] / 100, gs["max_h"] / 100)
                if canvas is not None:
                    canvas.draw_idle()
        # Re-render transition network if it is showing the network view
        view = str(self._tr_view_combo.currentData() or "")
        if view == "network":
            self._render_transition_from_cache()
        # Always redraw the currently visible panel so graph settings take effect
        self._redraw_current_motif_panel()

    def _all_available_comparisons(self) -> list[str]:
        """Return sorted list of all comparison keys available across all results."""
        keys: set[str] = set()
        # From transition result pval_mats (keys are tuple stored as str or tuple)
        for k in self._transition_result.get("pval_mats", {}):
            if isinstance(k, tuple):
                keys.add(f"{k[0]} vs {k[1]}")
            else:
                keys.add(str(k))
        # From motif result pval_pairs (per item)
        for src in ("ngram_group_results", "ngram_agg_results"):
            for n_val, items in (self._motif_result.get(src) or {}).items():
                for it in items:
                    for k in (it.get("pval_pairs") or {}):
                        keys.add(str(k))
                break  # one n_val is enough to get all pair keys
        # From HMM result pval_occ (keys stored as "('g1', 'g2')" strings)
        for k in self._hmm_result.get("pval_occ", {}):
            # Key is stored as str repr of tuple: "('g1', 'g2')"
            ks = str(k).strip("()").replace("'", "")
            parts = [p.strip() for p in ks.split(",", 1)]
            if len(parts) == 2:
                keys.add(f"{parts[0]} vs {parts[1]}")
            else:
                keys.add(ks)
        return sorted(keys)

    def _sig_for_pairs(self, pval_pairs: dict[str, float]) -> str:
        """Return significance label for the most significant *selected* pair, or ''."""
        sel = self._motif_selected_comparisons  # None = all pairs
        min_p = 1.0
        for key, pv in pval_pairs.items():
            if sel is None or key in sel:
                if pv < min_p:
                    min_p = pv
        if min_p < 0.001:
            return "***"
        if min_p < 0.01:
            return "**"
        if min_p < 0.05:
            return "*"
        return ""

    def _pval_mat_filtered(self, pval_mats: dict) -> dict:
        """Return only pval_mat entries matching the selected comparisons."""
        sel = self._motif_selected_comparisons
        if sel is None:
            return pval_mats
        out: dict = {}
        for k, v in pval_mats.items():
            if isinstance(k, tuple):
                key_str = f"{k[0]} vs {k[1]}"
            else:
                ks = str(k).strip("()").replace("'", "")
                parts = [p.strip() for p in ks.split(",", 1)]
                key_str = f"{parts[0]} vs {parts[1]}" if len(parts) == 2 else ks
            if key_str in sel:
                out[k] = v
        return out

    def _open_comparisons_dialog(self) -> None:
        """Dialog to choose which pairwise comparisons to display as significance markers."""
        all_pairs = self._all_available_comparisons()
        if not all_pairs:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "No Comparisons Available",
                "Run at least one analysis first to populate the available group comparisons."
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Choose Significance Comparisons")
        dlg.resize(360, 60 + len(all_pairs) * 30)
        layout = QVBoxLayout(dlg)

        lbl = QLabel(
            "Check which pairwise comparisons to show as significance markers\n"
            "on the charts. Unchecked pairs are still tested but not displayed."
        )
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        checks: list[QCheckBox] = []
        for pair in all_pairs:
            cb = QCheckBox(pair)
            # Default: checked (show all), or preserve previous selection
            if self._motif_selected_comparisons is None:
                cb.setChecked(True)
            else:
                cb.setChecked(pair in self._motif_selected_comparisons)
            layout.addWidget(cb)
            checks.append(cb)

        # Select all / none buttons
        sel_row = QHBoxLayout()
        sel_all_btn = QPushButton("Select All")
        sel_none_btn = QPushButton("Select None")
        sel_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb in checks])
        sel_none_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in checks])
        sel_row.addWidget(sel_all_btn)
        sel_row.addWidget(sel_none_btn)
        sel_row.addStretch(1)
        layout.addLayout(sel_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        selected = {cb.text() for cb in checks if cb.isChecked()}
        # If all are checked, treat as "show all" (None)
        self._motif_selected_comparisons = None if selected == set(all_pairs) else selected
        # Redraw current panel immediately
        self._redraw_current_motif_panel()

    def _redraw_current_motif_panel(self) -> None:
        """Re-render the active inner sub-tab using cached results (no recomputation)."""
        idx = self._inner_tabs.currentIndex()
        if idx == 0:
            self._render_transition_from_cache()
        elif idx == 1:
            self._render_motifs_from_cache()
        elif idx == 2:
            self._render_hmm()

    def _render_motifs_from_cache(self) -> None:
        """Re-render the motif panel from ``self._motif_result`` without rerunning."""
        if self._motif_result:
            self._on_motifs_done(self._motif_result)

    # ── Settings dialog ───────────────────────────────────────────────

    def _open_settings_dialog(self) -> None:
        s = self._settings
        dlg = QDialog(self)
        dlg.setWindowTitle("Motif Analysis Settings")
        dlg.resize(440, 580)
        form = QFormLayout()

        gap_spin = QDoubleSpinBox(dlg); gap_spin.setRange(0.1, 300.0); gap_spin.setValue(s.max_transition_gap_s); gap_spin.setSuffix(" s"); gap_spin.setDecimals(1)
        gap_spin.setToolTip(
            "Max time between end of bout A and start of bout B for A\u2192B to be counted."
        )
        form.addRow("Max transition gap:", gap_spin)

        overlap_tol_spin = QDoubleSpinBox(dlg)
        overlap_tol_spin.setRange(0.0, 10.0)
        overlap_tol_spin.setSingleStep(0.1)
        overlap_tol_spin.setDecimals(2)
        overlap_tol_spin.setSuffix(" s")
        overlap_tol_spin.setValue(s.bout_overlap_tolerance_s)
        overlap_tol_spin.setToolTip(
            "Temporal-refinement models run independently and can produce bouts\n"
            "that overlap by a fraction of a second at behavioral transitions.\n"
            "This tolerance allows bout B to start up to this many seconds BEFORE\n"
            "bout A ends and still be counted as a transition A\u2192B.\n"
            "The same threshold is used to filter concurrent bouts from N-gram\n"
            "motif sequences and HMM state encoding (bouts that overlap more\n"
            "than this are treated as concurrent and excluded from the sequence).\n"
            "0 = strict (B must start after A ends).  Default: 1.0 s."
        )
        form.addRow("Bout overlap tolerance:", overlap_tol_spin)

        self_trans_cb = QCheckBox("Include self-transitions (A\u2192A)", dlg); self_trans_cb.setChecked(s.include_self_transitions)
        form.addRow(self_trans_cb)
        n_perm_spin = QSpinBox(dlg); n_perm_spin.setRange(100, 10000); n_perm_spin.setSingleStep(100); n_perm_spin.setValue(s.n_permutations)
        form.addRow("Permutations:", n_perm_spin)
        corr_combo = QComboBox(dlg)
        corr_combo.addItem("FDR (Benjamini-Hochberg)", userData="fdr_bh")
        corr_combo.addItem("None (raw p-values)", userData="none")
        corr_mode = str(getattr(s, "transition_pval_correction", "fdr_bh") or "fdr_bh").strip().lower()
        corr_combo.setCurrentIndex(0 if corr_mode == "fdr_bh" else 1)
        corr_combo.setToolTip(
            "Multiple-comparison correction for transition-matrix cell-wise\n"
            "permutation p-values used by significance markers."
        )
        form.addRow("Transition p-value correction:", corr_combo)

        form.addRow(QLabel("\u2014\u2014 N-gram \u2014\u2014"))
        min_n_spin = QSpinBox(dlg); min_n_spin.setRange(2, 10); min_n_spin.setValue(s.ngram_min_n)
        form.addRow("Min N:", min_n_spin)
        max_n_spin = QSpinBox(dlg); max_n_spin.setRange(2, 10); max_n_spin.setValue(s.ngram_max_n)
        form.addRow("Max N:", max_n_spin)
        top_k_spin = QSpinBox(dlg); top_k_spin.setRange(3, 100); top_k_spin.setValue(s.ngram_top_k)
        form.addRow("Top K motifs:", top_k_spin)
        min_count_spin = QSpinBox(dlg); min_count_spin.setRange(1, 100); min_count_spin.setValue(s.min_ngram_count)
        form.addRow("Min motif count:", min_count_spin)

        form.addRow(QLabel("\u2014\u2014 Sequence Clustering \u2014\u2014"))
        cluster_n_spin = QSpinBox(dlg); cluster_n_spin.setRange(2, 8); cluster_n_spin.setValue(s.cluster_ngram_n)
        form.addRow("Cluster n-gram size:", cluster_n_spin)
        umap_nn_spin = QSpinBox(dlg); umap_nn_spin.setRange(2, 50); umap_nn_spin.setValue(s.umap_n_neighbors)
        form.addRow("UMAP n_neighbors:", umap_nn_spin)
        hdbscan_mcs_spin = QSpinBox(dlg); hdbscan_mcs_spin.setRange(2, 20); hdbscan_mcs_spin.setValue(s.hdbscan_min_cluster_size)
        form.addRow("HDBSCAN min cluster:", hdbscan_mcs_spin)

        form.addRow(QLabel("\u2014\u2014 HMM \u2014\u2014"))
        hmm_min_spin = QSpinBox(dlg); hmm_min_spin.setRange(2, 20); hmm_min_spin.setValue(s.hmm_n_states_min)
        form.addRow("N states min:", hmm_min_spin)
        hmm_max_spin = QSpinBox(dlg); hmm_max_spin.setRange(2, 20); hmm_max_spin.setValue(s.hmm_n_states_max)
        form.addRow("N states max:", hmm_max_spin)
        hmm_iter_spin = QSpinBox(dlg); hmm_iter_spin.setRange(10, 2000); hmm_iter_spin.setValue(s.hmm_n_iter)
        form.addRow("HMM iterations:", hmm_iter_spin)
        hmm_restart_spin = QSpinBox(dlg); hmm_restart_spin.setRange(1, 20); hmm_restart_spin.setValue(s.hmm_n_restarts)
        form.addRow("HMM restarts:", hmm_restart_spin)
        hmm_crit_combo = QComboBox(dlg); hmm_crit_combo.addItem("BIC", userData="bic"); hmm_crit_combo.addItem("AIC", userData="aic")
        hmm_crit_combo.setCurrentIndex(0 if s.hmm_criterion == "bic" else 1)
        form.addRow("Selection criterion:", hmm_crit_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        vl = QVBoxLayout(dlg)
        scr = QScrollArea(dlg); scr.setWidgetResizable(True)
        inner = QWidget(); inner.setLayout(form); scr.setWidget(inner)
        vl.addWidget(scr, 1); vl.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        s.max_transition_gap_s    = gap_spin.value()
        s.bout_overlap_tolerance_s = overlap_tol_spin.value()
        s.include_self_transitions = self_trans_cb.isChecked()
        s.n_permutations          = n_perm_spin.value()
        s.transition_pval_correction = str(corr_combo.currentData() or "fdr_bh")
        s.ngram_min_n             = min_n_spin.value()
        s.ngram_max_n             = max(max_n_spin.value(), min_n_spin.value())
        s.ngram_top_k             = top_k_spin.value()
        s.min_ngram_count         = min_count_spin.value()
        s.cluster_ngram_n         = cluster_n_spin.value()
        s.umap_n_neighbors        = umap_nn_spin.value()
        s.hdbscan_min_cluster_size = hdbscan_mcs_spin.value()
        s.hmm_n_states_min        = hmm_min_spin.value()
        s.hmm_n_states_max        = max(hmm_max_spin.value(), hmm_min_spin.value())
        s.hmm_n_iter              = hmm_iter_spin.value()
        s.hmm_n_restarts          = hmm_restart_spin.value()
        s.hmm_criterion           = str(hmm_crit_combo.currentData() or "bic")
        self._tr_gap_spin.setValue(s.max_transition_gap_s)
        self._tr_norm_cb.setChecked(s.normalize_rows)
        self._save_settings()

    # ── Export helpers ────────────────────────────────────────────────

    def _export_figure(self, panel: str) -> None:
        fig_map = {"transition": self._tr_fig, "motif": self._mo_fig, "hmm": self._hmm_sel_fig}
        fig = fig_map.get(panel)
        if fig is None:
            QMessageBox.information(self, "Export", "No figure to export.")
            return
        dpi = int(self._host._graph_settings.get("dpi", 150))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All Files (*)",
        )
        if not path:
            return
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
            self._host._status.setText(f"Exported figure to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    def _select_relationships_for_export(
        self,
        relationships: list[str],
        *,
        title: str,
        message: str,
    ) -> set[str] | None:
        """Prompt the user to choose which relationship rows to export.

        Returns selected relationship labels, or ``None`` if cancelled.
        """
        rels = sorted({str(r) for r in relationships if str(r).strip()})
        if not rels:
            QMessageBox.information(self, "Export", "No relationships available to export.")
            return set()

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(520, 520)
        layout = QVBoxLayout(dlg)

        lbl = QLabel(message)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        list_w = QListWidget(dlg)
        list_w.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        checks: list[QListWidgetItem] = []
        for rel in rels:
            it = QListWidgetItem(rel)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            list_w.addItem(it)
            checks.append(it)
        layout.addWidget(list_w, 1)

        row = QHBoxLayout()
        all_btn = QPushButton("Select All")
        none_btn = QPushButton("Select None")

        def _set_all(state: Qt.CheckState) -> None:
            for it in checks:
                it.setCheckState(state)

        all_btn.clicked.connect(lambda: _set_all(Qt.CheckState.Checked))
        none_btn.clicked.connect(lambda: _set_all(Qt.CheckState.Unchecked))
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch(1)
        layout.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        selected = {
            it.text() for it in checks
            if it.checkState() == Qt.CheckState.Checked
        }
        return selected

    @staticmethod
    def _corr_relationship_label(a: str, b: str) -> str:
        a2, b2 = sorted((str(a), str(b)))
        return f"Correlation: {a2} <-> {b2}"

    def _export_data_csv(self, panel: str) -> None:
        if panel == "transition":
            result = self._transition_result
            if not result:
                QMessageBox.information(self, "Export", "Run transition analysis first.")
                return

            bnames = result.get("bnames", [])
            include_self = bool(self._settings.include_self_transitions)
            relationships = [
                f"{bna} -> {bnb}"
                for i, bna in enumerate(bnames)
                for j, bnb in enumerate(bnames)
                if include_self or i != j
            ]
            selected_rels = self._select_relationships_for_export(
                relationships,
                title="Export Transition Relationships",
                message="Choose which behavior-to-behavior transitions to include in the CSV export.",
            )
            if selected_rels is None:
                return
            if not selected_rels:
                QMessageBox.information(self, "Export", "No relationships selected.")
                return

            path, _ = QFileDialog.getSaveFileName(self, "Export Transition Data", "", "CSV (*.csv);;All Files (*)")
            if not path:
                return
            per_session: dict[str, Any] = result.get("per_session", {})
            normalize = self._tr_norm_cb.isChecked()
            rows: list[dict[str, Any]] = []
            for sid, mat in per_session.items():
                if normalize:
                    from abel.services.behavioral_motif_service import normalize_transition_matrix
                    dm = normalize_transition_matrix(mat)
                else:
                    dm = mat
                lm = self._host._session_label_by_session
                grp = self._host._session_groups.get(lm.get(sid, sid), "")
                for i, bna in enumerate(bnames):
                    for j, bnb in enumerate(bnames):
                        rel = f"{bna} -> {bnb}"
                        if rel not in selected_rels:
                            continue
                        rows.append({"session_id": sid, "session_label": lm.get(sid, sid),
                                     "group": grp, "from_behavior": bna, "to_behavior": bnb,
                                     "value": float(dm[i, j]),
                                     "metric": "probability" if normalize else "count"})
            if not rows:
                QMessageBox.information(self, "Export", "No transition rows matched the selected relationships.")
                return
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            self._host._status.setText(f"Exported to {path}")

        elif panel == "motif":
            result = self._motif_result
            if not result:
                QMessageBox.information(self, "Export", "Run motif analysis first.")
                return

            motif_rels: set[str] = set()
            for src in ("ngram_agg_results", "ngram_group_results"):
                for _n_val, items in result.get(src, {}).items():
                    for it in items:
                        motif_rels.add(f"Motif: {it.get('motif_label', '')}")

            corr_data = result.get("behavior_correlation") or {}
            corr_bnames: list[str] = list(corr_data.get("bnames") or [])
            corr_rels: set[str] = set()
            for i, b1 in enumerate(corr_bnames):
                for j, b2 in enumerate(corr_bnames):
                    if i >= j:
                        continue
                    corr_rels.add(self._corr_relationship_label(b1, b2))

            selected_rels = self._select_relationships_for_export(
                sorted(motif_rels | corr_rels),
                title="Export Behavior Relationships",
                message=(
                    "Choose which relationships to include.\n"
                    "Motif items filter n-gram rows; Correlation items filter correlation rows."
                ),
            )
            if selected_rels is None:
                return
            if not selected_rels:
                QMessageBox.information(self, "Export", "No relationships selected.")
                return

            path, _ = QFileDialog.getSaveFileName(self, "Export Motif Data", "", "CSV (*.csv);;All Files (*)")
            if not path:
                return
            rows2 = []
            for src in ("ngram_agg_results", "ngram_group_results"):
                for n_val, items in result.get(src, {}).items():
                    for it in items:
                        motif_key = f"Motif: {it.get('motif_label', '')}"
                        if motif_key not in selected_rels:
                            continue
                        row: dict[str, Any] = {"n": n_val, "motif": " > ".join(it["motif"]),
                                                "motif_label": it["motif_label"], "total": it["total"]}
                        for sid2, cnt in it.get("per_session", {}).items():
                            row[f"session_{sid2}"] = cnt
                        for g2, mv in (it.get("group_means") or {}).items():
                            row[f"group_mean_{g2}"] = mv
                        if "pval" in it:
                            row["pval"] = it["pval"]
                        rows2.append(row)

            pooled = np.asarray(corr_data.get("pooled") or [], dtype=float)
            if corr_bnames and pooled.size:
                for i, b1 in enumerate(corr_bnames):
                    for j, b2 in enumerate(corr_bnames):
                        if i == j:
                            continue
                        rel = self._corr_relationship_label(b1, b2)
                        if rel not in selected_rels:
                            continue
                        rows2.append({
                            "record_type": "behavior_correlation",
                            "matrix": "pooled",
                            "behavior_a": b1,
                            "behavior_b": b2,
                            "correlation_r": float(pooled[i, j]),
                        })
                for grp, mat in sorted((corr_data.get("per_group") or {}).items()):
                    gmat = np.asarray(mat, dtype=float)
                    if gmat.size == 0:
                        continue
                    for i, b1 in enumerate(corr_bnames):
                        for j, b2 in enumerate(corr_bnames):
                            if i == j:
                                continue
                            rel = self._corr_relationship_label(b1, b2)
                            if rel not in selected_rels:
                                continue
                            rows2.append({
                                "record_type": "behavior_correlation",
                                "matrix": str(grp),
                                "behavior_a": b1,
                                "behavior_b": b2,
                                "correlation_r": float(gmat[i, j]),
                            })

            if not rows2:
                QMessageBox.information(self, "Export", "No motif data.")
                return
            pd.DataFrame(rows2).to_csv(path, index=False, encoding="utf-8-sig")
            self._host._status.setText(f"Exported to {path}")

        elif panel == "hmm":
            result = self._hmm_result
            if not result or result.get("error"):
                QMessageBox.information(self, "Export", "Run HMM analysis first.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export HMM Data", "", "CSV (*.csv);;All Files (*)")
            if not path:
                return
            n_states = result["n_states"]
            bnames = result.get("bnames", [])
            rows3: list[dict[str, Any]] = []
            emit_mat = result.get("emission_matrix", [])
            for i in range(n_states):
                r: dict[str, Any] = {"state": i, "type": "emission"}
                for j, bn in enumerate(bnames):
                    r[bn] = emit_mat[i][j] if emit_mat and i < len(emit_mat) else 0.0
                rows3.append(r)
            trans_mat = result.get("transition_matrix", [])
            for i in range(n_states):
                r2: dict[str, Any] = {"state": i, "type": "hmm_transition"}
                for j in range(n_states):
                    r2[f"to_{j}"] = trans_mat[i][j] if trans_mat and i < len(trans_mat) else 0.0
                rows3.append(r2)
            for sid, fracs in result.get("session_occupancy", {}).items():
                lm = self._host._session_label_by_session
                r3: dict[str, Any] = {"state": -1, "type": "occupancy",
                                       "session_id": sid, "session_label": lm.get(sid, sid),
                                       "group": self._host._session_groups.get(lm.get(sid, sid), "")}
                for st, frac in enumerate(fracs):
                    r3[f"state_{st}_frac"] = frac
                rows3.append(r3)
            pd.DataFrame(rows3).to_csv(path, index=False, encoding="utf-8-sig")
            self._host._status.setText(f"Exported to {path}")

    # ── Error handler ────────────────────────────────────────────────

    def _on_worker_failed(self, tb: str) -> None:
        logger.error("Motif analysis worker failed:\n%s", tb)
        for btn, default in [
            (self._tr_run_btn, "Run"),
            (self._mo_run_btn, "Run"),
            (self._hmm_run_btn, "Run HMM"),
        ]:
            btn.setEnabled(True)
            btn.setText(default)
        short = tb.strip().splitlines()[-1] if tb.strip() else "Unknown error"
        self._status_lbl.setText(f"Analysis failed: {short}")
        QMessageBox.warning(
            self, "Analysis Error",
            f"The analysis failed:\n\n{short}\n\nSee the application log for details.",
        )


# ======================================================================
# Sub-tab 6: Session Sections
# ======================================================================


class _SessionSectionsWidget(QWidget):
    """Divide each session into user-defined named time sections and
    analyse behavior count / duration within each section.

    Sections are defined by a name and a duration in seconds and are
    applied sequentially from t=0 of the (prechop-adjusted) session.

    Two chart styles are available:
    • By Section (bar)  – average count/duration per section, one
                          subplot per behavior.
    • Across Sections (line) – trend line showing how each behavior
                               changes from section to section.
    """

    _BTN_STYLE = (
        "QPushButton{padding:3px 9px;border:1px solid #37474f;border-radius:3px;"
        "background:#1a2027;color:#cfd8dc;}"
        "QPushButton:checked{background:#1565c0;border-color:#1565c0;color:#fff;}"
        "QPushButton:hover:!checked{background:#263238;}"
    )
    _HEATMAP_BINS_PER_SECTION = 20

    # ── Built-in section presets ──────────────────────────────────────
    _BUILTIN_PRESETS: list[dict] = [
        {
            "name": "Trace Fear Conditioning",
            "sections": [
                {"name": "Baseline",  "duration": 180},
                {"name": "Tone 1",    "duration": 20},
                {"name": "Trace 1",   "duration": 20},
                {"name": "Shock 1",   "duration": 2},
                {"name": "ITI 1",     "duration": 210},
                {"name": "Tone 2",    "duration": 20},
                {"name": "Trace 2",   "duration": 20},
                {"name": "Shock 2",   "duration": 2},
                {"name": "ITI 2",     "duration": 210},
                {"name": "Tone 3",    "duration": 20},
                {"name": "Trace 3",   "duration": 20},
                {"name": "Shock 3",   "duration": 2},
                {"name": "ITI 3",     "duration": 210},
                {"name": "Tone 4",    "duration": 20},
                {"name": "Trace 4",   "duration": 20},
                {"name": "Shock 4",   "duration": 2},
                {"name": "ITI 4",     "duration": 210},
                {"name": "Tone 5",    "duration": 20},
                {"name": "Trace 5",   "duration": 20},
                {"name": "Shock 5",   "duration": 2},
                {"name": "ITI 5",     "duration": 210},
            ],
        },
        {
            "name": "Trace Fear Extinction",
            "sections": [
                {"name": "Baseline (Context B)", "duration": 120},
                {"name": "Tone 1", "duration": 20},
                {"name": "ITI 1", "duration": 60},
                {"name": "Tone 2", "duration": 20},
                {"name": "ITI 2", "duration": 60},
                {"name": "Tone 3", "duration": 20},
                {"name": "ITI 3", "duration": 60},
                {"name": "Tone 4", "duration": 20},
                {"name": "ITI 4", "duration": 60},
                {"name": "Tone 5", "duration": 20},
                {"name": "ITI 5", "duration": 60},
                {"name": "Tone 6", "duration": 20},
                {"name": "ITI 6", "duration": 60},
                {"name": "Tone 7", "duration": 20},
                {"name": "ITI 7", "duration": 60},
                {"name": "Tone 8", "duration": 20},
                {"name": "ITI 8", "duration": 60},
                {"name": "Tone 9", "duration": 20},
                {"name": "ITI 9", "duration": 60},
                {"name": "Tone 10", "duration": 20},
                {"name": "ITI 10", "duration": 60},
                {"name": "Tone 11", "duration": 20},
                {"name": "ITI 11", "duration": 60},
                {"name": "Tone 12", "duration": 20},
                {"name": "ITI 12", "duration": 60},
                {"name": "Tone 13", "duration": 20},
                {"name": "ITI 13", "duration": 60},
                {"name": "Tone 14", "duration": 20},
                {"name": "ITI 14", "duration": 60},
                {"name": "Tone 15", "duration": 20},
                {"name": "ITI 15", "duration": 60},
                {"name": "Tone 16", "duration": 20},
                {"name": "ITI 16", "duration": 60},
                {"name": "Tone 17", "duration": 20},
                {"name": "ITI 17", "duration": 60},
                {"name": "Tone 18", "duration": 20},
                {"name": "ITI 18", "duration": 60},
                {"name": "Tone 19", "duration": 20},
                {"name": "ITI 19", "duration": 60},
                {"name": "Tone 20", "duration": 20},
                {"name": "ITI 20", "duration": 60},
            ],
        },
        {
            "name": "Delay Fear Conditioning",
            "sections": [
                {"name": "Baseline",  "duration": 180},
                {"name": "Tone 1",    "duration": 20},
                {"name": "Shock 1",   "duration": 2},
                {"name": "ITI 1",     "duration": 90},
                {"name": "Tone 2",    "duration": 20},
                {"name": "Shock 2",   "duration": 2},
                {"name": "ITI 2",     "duration": 90},
                {"name": "Tone 3",    "duration": 20},
                {"name": "Shock 3",   "duration": 2},
                {"name": "ITI 3",     "duration": 90},
                {"name": "Tone 4",    "duration": 20},
                {"name": "Shock 4",   "duration": 2},
                {"name": "ITI 4",     "duration": 90},
                {"name": "Tone 5",    "duration": 20},
                {"name": "Shock 5",   "duration": 2},
                {"name": "ITI 5",     "duration": 90},
            ],
        },
        {
            "name": "Delay Fear Extinction",
            "sections": [
                {"name": "Baseline", "duration": 180},
                {"name": "Tone 1", "duration": 20},
                {"name": "ITI 1", "duration": 60},
                {"name": "Tone 2", "duration": 20},
                {"name": "ITI 2", "duration": 60},
                {"name": "Tone 3", "duration": 20},
                {"name": "ITI 3", "duration": 60},
                {"name": "Tone 4", "duration": 20},
                {"name": "ITI 4", "duration": 60},
                {"name": "Tone 5", "duration": 20},
                {"name": "ITI 5", "duration": 60},
                {"name": "Tone 6", "duration": 20},
                {"name": "ITI 6", "duration": 60},
                {"name": "Tone 7", "duration": 20},
                {"name": "ITI 7", "duration": 60},
                {"name": "Tone 8", "duration": 20},
                {"name": "ITI 8", "duration": 60},
                {"name": "Tone 9", "duration": 20},
                {"name": "ITI 9", "duration": 60},
                {"name": "Tone 10", "duration": 20},
                {"name": "ITI 10", "duration": 60},
                {"name": "Tone 11", "duration": 20},
                {"name": "ITI 11", "duration": 60},
                {"name": "Tone 12", "duration": 20},
                {"name": "ITI 12", "duration": 60},
                {"name": "Tone 13", "duration": 20},
                {"name": "ITI 13", "duration": 60},
                {"name": "Tone 14", "duration": 20},
                {"name": "ITI 14", "duration": 60},
                {"name": "Tone 15", "duration": 20},
                {"name": "ITI 15", "duration": 60},
                {"name": "Tone 16", "duration": 20},
                {"name": "ITI 16", "duration": 60},
                {"name": "Tone 17", "duration": 20},
                {"name": "ITI 17", "duration": 60},
                {"name": "Tone 18", "duration": 20},
                {"name": "ITI 18", "duration": 60},
                {"name": "Tone 19", "duration": 20},
                {"name": "ITI 19", "duration": 60},
                {"name": "Tone 20", "duration": 20},
                {"name": "ITI 20", "duration": 60},
            ],
        },
    ]

    def __init__(self, host: "BehaviorAnalyticsTab") -> None:
        super().__init__()
        self._host = host
        self._updating = False
        self._custom_presets: list[dict] = []  # persisted via host group state

        # ── Section-definition table ──────────────────────────────────
        self._section_table = QTableWidget(0, 2)
        self._section_table.setHorizontalHeaderLabels(["Section Name", "Duration (s)"])
        self._section_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._section_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._section_table.verticalHeader().setVisible(False)
        self._section_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self._section_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._section_table.setMinimumHeight(80)
        self._section_table.setMaximumHeight(220)
        self._section_table.cellChanged.connect(self._on_section_changed)
        self._section_table.installEventFilter(self)

        self._add_section_btn = QPushButton("Add Section")
        self._add_section_btn.clicked.connect(self._add_section)
        self._remove_section_btn = QPushButton("Remove Selected")
        self._remove_section_btn.clicked.connect(self._remove_selected_sections)

        self._total_dur_label = QLabel("Total: 0 s")
        self._total_dur_label.setStyleSheet("color:#90a4ae;font-size:10px;")

        sec_btn_row = QHBoxLayout()
        sec_btn_row.addWidget(self._add_section_btn)
        sec_btn_row.addWidget(self._remove_section_btn)
        sec_btn_row.addStretch(1)

        # ── Preset selector ───────────────────────────────────────────
        self._preset_combo = QComboBox()
        self._preset_combo.setToolTip("Select a section preset to load.")
        self._preset_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._load_preset_btn = QPushButton("Load")
        self._load_preset_btn.setStyleSheet(self._BTN_STYLE)
        self._load_preset_btn.setMaximumWidth(52)
        self._load_preset_btn.setToolTip("Replace current sections with the selected preset.")
        self._load_preset_btn.clicked.connect(self._load_selected_preset)
        self._save_preset_btn = QPushButton("Save as Preset\u2026")
        self._save_preset_btn.setStyleSheet(self._BTN_STYLE)
        self._save_preset_btn.setToolTip(
            "Save the current section definitions as a named custom preset."
        )
        self._save_preset_btn.clicked.connect(self._save_current_as_preset)
        self._delete_preset_btn = QPushButton("Delete")
        self._delete_preset_btn.setStyleSheet(self._BTN_STYLE)
        self._delete_preset_btn.setMaximumWidth(54)
        self._delete_preset_btn.setToolTip("Delete the selected custom preset (built-ins cannot be deleted).")
        self._delete_preset_btn.clicked.connect(self._delete_selected_preset)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        preset_row.addWidget(self._preset_combo, 1)
        preset_row.addWidget(self._load_preset_btn)
        preset_row.addWidget(self._save_preset_btn)
        preset_row.addWidget(self._delete_preset_btn)

        self._rebuild_preset_combo()  # populate combo

        section_box = QGroupBox("Section Definitions")
        sb_layout = QVBoxLayout(section_box)
        _hint = QLabel(
            "Add named sections to divide each session from t=0.\n"
            "Each section is applied sequentially; only bouts whose start "
            "time falls within a section are counted for that section."
        )
        _hint.setWordWrap(True)
        _hint.setStyleSheet("color:#90a4ae;font-size:10px;")
        sb_layout.addLayout(preset_row)
        sb_layout.addWidget(_hint)
        sb_layout.addWidget(self._section_table)
        sb_layout.addLayout(sec_btn_row)
        sb_layout.addWidget(self._total_dur_label)

        # ── Chart-type / metric / mode toggles ────────────────────────
        def _toggle_row(
            options: list[tuple[str, str]], default_idx: int = 0
        ) -> tuple[QHBoxLayout, QButtonGroup, dict[str, QPushButton]]:
            grp = QButtonGroup(self)
            grp.setExclusive(True)
            row = QHBoxLayout()
            row.setSpacing(3)
            btns: dict[str, QPushButton] = {}
            for i, (label, key) in enumerate(options):
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.setChecked(i == default_idx)
                btn.setStyleSheet(self._BTN_STYLE)
                btn.setSizePolicy(
                    QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
                )
                btn.setMaximumHeight(26)
                grp.addButton(btn, i)
                row.addWidget(btn)
                btns[key] = btn
            row.addStretch(1)
            return row, grp, btns

        chart_row, self._chart_grp, self._chart_btns = _toggle_row(
            [("By Section (Bar)", "bar"), ("Across Sections (Line)", "line"), ("Heatmap", "heatmap")]
        )
        metric_row, self._metric_grp, self._metric_btns = _toggle_row(
            [("Bout Count", "n_bouts"), ("Duration (s)", "duration_s"), ("% Time", "pct_time")]
        )
        mode_row, self._mode_grp, self._mode_btns = _toggle_row(
            [("Individual Sessions", "individual"), ("By Group", "group")]
        )
        self._mode_grp.idClicked.connect(lambda _: self._on_mode_toggled())

        # ── Faceted grouping controls (visible in group mode only) ─────
        self._ss_facet_controls: dict[str, str] = {}
        self._facet = _FacetControls("Group by:")
        self._facet.setToolTip(
            "For each factor choose — combine —, — split —, or a specific level.\n"
            "Split two or more factors to plot their interaction."
        )
        self._facet.changed.connect(self._on_facets_changed)

        self._group_controls = QWidget()
        _gc_vbox = QVBoxLayout(self._group_controls)
        _gc_vbox.setSpacing(3)
        _gc_vbox.setContentsMargins(0, 0, 0, 0)
        _gc_vbox.addWidget(self._facet)
        self._group_controls.setVisible(False)

        # ── Behavior selector ─────────────────────────────────────────
        self._behavior_list = QListWidget()
        self._behavior_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._behavior_list.setMinimumHeight(90)
        self._behavior_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._behavior_list.setStyleSheet(
            "QListWidget{background:#0A1929;border:1px solid #1E3A5F;"
            "border-radius:3px;color:#cfd8dc;font-size:10px;}"
        )
        _beh_all = QPushButton("All")
        _beh_all.setMaximumWidth(40)
        _beh_all.setStyleSheet(self._BTN_STYLE)
        _beh_all.clicked.connect(self._check_all_behaviors)
        _beh_none = QPushButton("None")
        _beh_none.setMaximumWidth(48)
        _beh_none.setStyleSheet(self._BTN_STYLE)
        _beh_none.clicked.connect(self._uncheck_all_behaviors)

        _beh_btn_vbox = QVBoxLayout()
        _beh_btn_vbox.addWidget(_beh_all)
        _beh_btn_vbox.addWidget(_beh_none)
        _beh_btn_vbox.addStretch(1)

        beh_row = QHBoxLayout()
        _beh_lbl = QLabel("Behaviors:")
        _beh_lbl.setMinimumWidth(60)
        _beh_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        beh_row.addWidget(_beh_lbl)
        beh_row.addWidget(self._behavior_list, 1)
        beh_row.addLayout(_beh_btn_vbox)

        # ── Section filter ────────────────────────────────────────────
        self._section_filter_list = QListWidget()
        self._section_filter_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._section_filter_list.setMinimumHeight(90)
        self._section_filter_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._section_filter_list.setStyleSheet(
            "QListWidget{background:#0A1929;border:1px solid #1E3A5F;"
            "border-radius:3px;color:#cfd8dc;font-size:10px;}"
        )
        _sec_all = QPushButton("All")
        _sec_all.setMaximumWidth(40)
        _sec_all.setStyleSheet(self._BTN_STYLE)
        _sec_all.clicked.connect(self._check_all_sections)
        _sec_none = QPushButton("None")
        _sec_none.setMaximumWidth(48)
        _sec_none.setStyleSheet(self._BTN_STYLE)
        _sec_none.clicked.connect(self._uncheck_all_sections)

        _sec_btn_vbox = QVBoxLayout()
        _sec_btn_vbox.addWidget(_sec_all)
        _sec_btn_vbox.addWidget(_sec_none)
        _sec_btn_vbox.addStretch(1)

        sec_filter_row = QHBoxLayout()
        _sec_lbl = QLabel("Sections:")
        _sec_lbl.setMinimumWidth(60)
        _sec_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        sec_filter_row.addWidget(_sec_lbl)
        sec_filter_row.addWidget(self._section_filter_list, 1)
        sec_filter_row.addLayout(_sec_btn_vbox)

        # Quick section-type selection (e.g., Tone / ITI / Baseline).
        self._section_type_combo = QComboBox()
        self._section_type_combo.setToolTip(
            "Select a section type to quickly check matching sections."
        )
        self._section_type_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._section_type_add_btn = QPushButton("Select Type")
        self._section_type_add_btn.setMaximumWidth(84)
        self._section_type_add_btn.setStyleSheet(self._BTN_STYLE)
        self._section_type_add_btn.setToolTip(
            "Check all sections of the selected type while preserving existing checks."
        )
        self._section_type_add_btn.clicked.connect(
            lambda: self._select_sections_by_type(only=False)
        )
        self._section_type_only_btn = QPushButton("Only Type")
        self._section_type_only_btn.setMaximumWidth(74)
        self._section_type_only_btn.setStyleSheet(self._BTN_STYLE)
        self._section_type_only_btn.setToolTip(
            "Check only sections of the selected type and uncheck all others."
        )
        self._section_type_only_btn.clicked.connect(
            lambda: self._select_sections_by_type(only=True)
        )
        _stype_row = QHBoxLayout()
        _stype_lbl = QLabel("Type:")
        _stype_lbl.setMinimumWidth(60)
        _stype_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        _stype_row.setSpacing(4)
        _stype_row.addWidget(_stype_lbl)
        _stype_row.addWidget(self._section_type_combo, 1)
        _stype_row.addWidget(self._section_type_add_btn)
        _stype_row.addWidget(self._section_type_only_btn)

        # ── Action buttons ────────────────────────────────────────────
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._update_plot)
        self._apply_btn.setStyleSheet(
            "QPushButton{padding:3px 14px;background:#1565c0;border:none;"
            "border-radius:3px;color:#fff;font-weight:bold;}"
            "QPushButton:hover{background:#1976d2;}"
        )
        self._export_btn = QPushButton("Export Chart\u2026")
        self._export_btn.clicked.connect(self._export_figure)
        self._export_data_btn = QPushButton("Export Data\u2026")
        self._export_data_btn.clicked.connect(self._export_data)
        self._graph_settings_btn = QPushButton("\u2699 Settings")
        self._graph_settings_btn.setToolTip(
            "Edit graph appearance: fonts, error bars, DPI, figure size."
        )
        self._graph_settings_btn.setStyleSheet(self._BTN_STYLE)
        self._graph_settings_btn.clicked.connect(self._open_graph_settings)

        self._level_order_btn_sec = QPushButton("Level Order\u2026")
        self._level_order_btn_sec.setToolTip(
            "Set the display order of levels within each factor. "
            "Applies to all charts including factor interactions."
        )
        self._level_order_btn_sec.clicked.connect(self._host._open_level_order_dialog)

        action_row = QHBoxLayout()
        action_row.addWidget(self._apply_btn)
        action_row.addWidget(self._export_btn)
        action_row.addWidget(self._export_data_btn)
        action_row.addWidget(self._graph_settings_btn)
        action_row.addWidget(self._level_order_btn_sec)
        action_row.addStretch(1)

        # ── Aggregate toggle ──────────────────────────────────────────
        agg_row, self._agg_grp, self._agg_btns = _toggle_row(
            [("By Section", "section"), ("By Trial Type", "type")]
        )
        self._agg_grp.idClicked.connect(lambda _: self._refresh_section_filter())

        # Optional averaging binning for repeated selected sections
        self._bin_sections_chk = QCheckBox("Average repeated selected sections")
        self._bin_sections_chk.setStyleSheet("color:#cfd8dc;font-size:10px;")
        self._bin_sections_chk.setToolTip(
            "When enabled, repeated selected sections of the same type are averaged in fixed-size bins.\n"
            "Example: ITI 1..20 with bin size 5 -> four averaged ITI bins.\n"
            "Changes take effect when you click Apply."
        )

        self._bin_size_spin = QSpinBox()
        self._bin_size_spin.setRange(2, 100)
        self._bin_size_spin.setValue(5)
        self._bin_size_spin.setToolTip(
            "Minimum number of repeated selected sections required to apply binning.\n"
            "Changes take effect when you click Apply."
        )

        bin_row = QHBoxLayout()
        bin_row.setSpacing(4)
        bin_row.addWidget(self._bin_sections_chk)
        _bin_lbl = QLabel("Bin size:")
        _bin_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        bin_row.addWidget(_bin_lbl)
        bin_row.addWidget(self._bin_size_spin)
        bin_row.addStretch(1)

        def _labeled_row(label: str, row_layout: QHBoxLayout) -> QHBoxLayout:
            outer = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(60)
            lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
            outer.addWidget(lbl)
            outer.addLayout(row_layout)
            return outer

        controls_widget = QWidget()
        ctrl_vbox = QVBoxLayout(controls_widget)
        ctrl_vbox.setSpacing(3)
        ctrl_vbox.setContentsMargins(4, 4, 4, 4)
        ctrl_vbox.addLayout(_labeled_row("Chart:", chart_row))
        ctrl_vbox.addLayout(_labeled_row("Metric:", metric_row))
        ctrl_vbox.addLayout(_labeled_row("Aggregate:", agg_row))
        ctrl_vbox.addLayout(_labeled_row("Binning:", bin_row))
        ctrl_vbox.addLayout(_labeled_row("View:", mode_row))
        ctrl_vbox.addWidget(self._group_controls)
        # Stretch lets the Behaviors / Sections lists share the spare vertical
        # space so all entries are visible without scrolling.
        ctrl_vbox.addLayout(beh_row, 1)
        ctrl_vbox.addLayout(sec_filter_row, 1)
        ctrl_vbox.addLayout(_stype_row)
        ctrl_vbox.addLayout(action_row)

        # ── Canvas ────────────────────────────────────────────────────
        self._figure: Any = None
        self._canvas: Any = None
        self._toolbar: Any = None
        self._placeholder = QLabel(
            "Define sections on the left and click Apply to view the chart."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setMinimumHeight(200)
        self._placeholder.setStyleSheet(
            "border: 1px solid #1A2027; background: #0A1929; "
            "border-radius: 4px; color: #546E7A;"
        )
        self._canvas_scroll: Any = None

        if (
            _ensure_matplotlib()
            and Figure is not None
            and FigureCanvas is not None
            and NavigationToolbar is not None
        ):
            _dpi = int(self._host._graph_settings.get("dpi", 100))
            _pw = int(self._host._graph_settings.get("max_w", 700))
            _ph = int(self._host._graph_settings.get("max_h", 420))
            self._figure = Figure(figsize=(_pw / _dpi, _ph / _dpi))
            self._canvas = FigureCanvas(self._figure)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
            )
            self._canvas.setFixedSize(_pw, _ph)
            self._toolbar = NavigationToolbar(self._canvas, self)
            self._placeholder.setVisible(False)
            from PySide6.QtWidgets import QScrollArea as _QScrollArea2
            self._canvas_scroll = _QScrollArea2()
            self._canvas_scroll.setWidget(self._canvas)
            self._canvas_scroll.setWidgetResizable(False)
            self._canvas_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._canvas_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            # Dynamic resize: canvas grows to fill the available viewport width
            self._sec_resize_filter = _ViewportResizeFilter(
                self._sync_canvas_to_viewport, self
            )
            self._canvas_scroll.viewport().installEventFilter(self._sec_resize_filter)

        # ── Layout ────────────────────────────────────────────────────
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.addWidget(section_box)
        # Stretch factor 1 lets the controls block (with its expanding lists)
        # grow into the space below the section-definitions table.
        left_l.addWidget(controls_widget, 1)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(280)
        # No maximum width: keep the splitter draggable in both directions.
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        if self._toolbar is not None:
            right_l.addWidget(self._toolbar)
        if self._canvas_scroll is not None:
            right_l.addWidget(self._canvas_scroll, 1)
        right_l.addWidget(self._placeholder)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1000])
        # Kept as attributes so showEvent can re-pin the controls width once the
        # widget has real geometry — at construction the hard-coded 340 px clips
        # the wider control rows (preset Save/Delete, Heatmap toggle, the
        # Duration column, the All/None buttons) behind a horizontal scrollbar.
        self._sec_splitter = splitter
        self._sec_left_inner = left
        self._sec_splitter_init = False

        root = QVBoxLayout(self)
        root.addWidget(splitter, 1)

    def _sync_canvas_to_viewport(self) -> None:
        """Fill the canvas to the scroll viewport width, preserving aspect."""
        _dpi = int(self._host._graph_settings.get("dpi", 100))
        _autofill_canvas(self._canvas_scroll, self._canvas, self._figure, dpi=_dpi)

    def showEvent(self, event: Any) -> None:  # type: ignore[override]
        """Re-fill the canvas when this sub-tab becomes visible (the resize
        filter only fires on an actual viewport resize, so a figure drawn while
        hidden would otherwise stay small until the splitter is wiggled)."""
        super().showEvent(event)

        def _settle() -> None:
            # First real show: the splitter now has true geometry, so pin the
            # controls to the width their content actually needs (so no buttons
            # are clipped) and give the rest to the plot.  Capped to half the
            # tab so the chart keeps a usable area.  Done once so we don't
            # clobber a deliberate user drag on later shows.
            if not self._sec_splitter_init:
                splitter = getattr(self, "_sec_splitter", None)
                if splitter is not None:
                    total = splitter.width()
                    if total > 400:
                        natural = self._sec_left_inner.sizeHint().width() + 24
                        left = max(360, min(natural, total // 2))
                        self._sec_splitter_init = True
                        splitter.setSizes([left, max(400, total - left)])
                        QTimer.singleShot(0, self._sync_canvas_to_viewport)
            self._sync_canvas_to_viewport()

        QTimer.singleShot(0, _settle)

    # ── Preset management ─────────────────────────────────────────────

    def _rebuild_preset_combo(self) -> None:
        """Repopulate the preset combo from built-ins + custom presets."""
        self._preset_combo.blockSignals(True)
        current = self._preset_combo.currentText()
        self._preset_combo.clear()
        self._preset_combo.addItem("— select a preset —")
        for p in self._BUILTIN_PRESETS:
            self._preset_combo.addItem(p["name"])
        for p in self._custom_presets:
            self._preset_combo.addItem(f"[Custom] {p['name']}")
        idx = self._preset_combo.findText(current)
        self._preset_combo.setCurrentIndex(max(0, idx))
        self._preset_combo.blockSignals(False)
        self._update_delete_btn_state()

    def _update_delete_btn_state(self) -> None:
        idx = self._preset_combo.currentIndex()
        n_builtin = len(self._BUILTIN_PRESETS)
        # Index 0 = placeholder; 1..n_builtin = built-ins; beyond = custom
        self._delete_preset_btn.setEnabled(idx > n_builtin)

    def _load_selected_preset(self) -> None:
        idx = self._preset_combo.currentIndex()
        if idx <= 0:
            return
        n_builtin = len(self._BUILTIN_PRESETS)
        if idx <= n_builtin:
            sections = self._BUILTIN_PRESETS[idx - 1]["sections"]
        else:
            sections = self._custom_presets[idx - n_builtin - 1]["sections"]
        self.set_sections_state(sections)
        self._host._save_group_state()

    def _save_current_as_preset(self) -> None:
        current_sections = self.get_sections_state()
        if not current_sections:
            QMessageBox.warning(self, "No Sections", "Add at least one section before saving a preset.")
            return
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:", text="My Preset"
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        # Overwrite if same name already exists in custom presets
        for p in self._custom_presets:
            if p["name"] == name:
                p["sections"] = current_sections
                self._rebuild_preset_combo()
                self._host._save_group_state()
                return
        self._custom_presets.append({"name": name, "sections": current_sections})
        self._rebuild_preset_combo()
        # Select the newly saved preset
        target = f"[Custom] {name}"
        idx = self._preset_combo.findText(target)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        self._host._save_group_state()

    def _delete_selected_preset(self) -> None:
        idx = self._preset_combo.currentIndex()
        n_builtin = len(self._BUILTIN_PRESETS)
        if idx <= n_builtin:
            return  # built-in, not deletable
        custom_idx = idx - n_builtin - 1
        name = self._custom_presets[custom_idx]["name"]
        reply = QMessageBox.question(
            self, "Delete Preset",
            f'Delete custom preset "{name}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._custom_presets.pop(custom_idx)
        self._rebuild_preset_combo()
        self._preset_combo.setCurrentIndex(0)
        self._host._save_group_state()

    def get_custom_presets(self) -> list[dict]:
        return list(self._custom_presets)

    def set_custom_presets(self, presets: list[dict]) -> None:
        self._custom_presets = [
            {"name": str(p.get("name", "")), "sections": list(p.get("sections", []))}
            for p in (presets or [])
            if p.get("name") and p.get("sections")
        ]
        self._rebuild_preset_combo()

    # ── Section management ────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        """Enable Ctrl+C / Ctrl+V copy-paste for the section definition table."""
        if obj is self._section_table and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            mods = event.modifiers()
            ctrl = Qt.KeyboardModifier.ControlModifier
            if mods & ctrl and key == Qt.Key.Key_C:
                item = self._section_table.currentItem()
                if item is not None:
                    QGuiApplication.clipboard().setText(item.text())
                return True
            if mods & ctrl and key == Qt.Key.Key_V:
                text = QGuiApplication.clipboard().text().strip()
                if not text:
                    return True
                item = self._section_table.currentItem()
                if item is not None:
                    self._section_table.blockSignals(True)
                    item.setText(text)
                    self._section_table.blockSignals(False)
                    self._on_section_changed()
                return True
        return super().eventFilter(obj, event)

    def _add_section(self) -> None:
        self._section_table.blockSignals(True)
        row = self._section_table.rowCount()
        self._section_table.insertRow(row)
        name_item = QTableWidgetItem(f"Section {row + 1}")
        dur_item = QTableWidgetItem("60")
        dur_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._section_table.setItem(row, 0, name_item)
        self._section_table.setItem(row, 1, dur_item)
        self._section_table.blockSignals(False)
        self._update_total_label()
        self._host._save_group_state()

    def get_sections_state(self) -> list[dict]:
        """Return section definitions as a list of dicts for persistence."""
        out: list[dict] = []
        for row in range(self._section_table.rowCount()):
            name_item = self._section_table.item(row, 0)
            out.append({
                "name": name_item.text() if name_item else f"Section {row + 1}",
                "duration": self._get_section_duration(row),
            })
        return out

    def set_sections_state(self, sections: list[dict]) -> None:
        """Repopulate the section table from persisted state."""
        self._section_table.blockSignals(True)
        self._section_table.setRowCount(0)
        for i, sec in enumerate(sections):
            row = self._section_table.rowCount()
            self._section_table.insertRow(row)
            name_item = QTableWidgetItem(str(sec.get("name") or f"Section {i + 1}"))
            dur_item = QTableWidgetItem(str(sec.get("duration", 60)))
            dur_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._section_table.setItem(row, 0, name_item)
            self._section_table.setItem(row, 1, dur_item)
        self._section_table.blockSignals(False)
        self._update_total_label()
        self._refresh_section_filter()

    def _remove_selected_sections(self) -> None:
        rows = sorted(
            {idx.row() for idx in self._section_table.selectedIndexes()},
            reverse=True,
        )
        for r in rows:
            self._section_table.removeRow(r)
        self._update_total_label()
        self._host._save_group_state()

    def _on_section_changed(self) -> None:
        self._update_total_label()
        self._refresh_section_filter()
        self._host._save_group_state()

    def _update_total_label(self) -> None:
        total = sum(
            self._get_section_duration(r)
            for r in range(self._section_table.rowCount())
        )
        self._total_dur_label.setText(f"Total: {total:.1f} s")

    def _get_section_duration(self, row: int) -> float:
        item = self._section_table.item(row, 1)
        if item is None:
            return 0.0
        try:
            return max(0.0, float(item.text()))
        except ValueError:
            return 0.0

    def _get_sections(self) -> list[tuple[str, float]]:
        """Return [(name, duration_s), ...] for each valid defined section."""
        out: list[tuple[str, float]] = []
        for row in range(self._section_table.rowCount()):
            name_item = self._section_table.item(row, 0)
            name = name_item.text().strip() if name_item else f"Section {row + 1}"
            dur = self._get_section_duration(row)
            if dur > 0:
                out.append((name or f"Section {row + 1}", dur))
        return out

    # ── Behavior selector ─────────────────────────────────────────────

    def _refresh_behavior_selector(self) -> None:
        prev_checked: set[str] = set()
        for i in range(self._behavior_list.count()):
            item = self._behavior_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                prev_checked.add(item.data(Qt.ItemDataRole.UserRole))

        self._behavior_list.clear()
        for bid, df in self._host._raw_bouts.items():
            if df.empty:
                continue
            bname = bid
            if "behavior" in df.columns and len(df) > 0:
                bname = str(df["behavior"].iloc[0])
            item = QListWidgetItem(str(bname))
            item.setData(Qt.ItemDataRole.UserRole, bid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if (not prev_checked or bid in prev_checked)
                else Qt.CheckState.Unchecked
            )
            self._behavior_list.addItem(item)

    def _checked_behavior_ids(self) -> list[str]:
        out: list[str] = []
        for i in range(self._behavior_list.count()):
            item = self._behavior_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def _check_all_behaviors(self) -> None:
        for i in range(self._behavior_list.count()):
            self._behavior_list.item(i).setCheckState(Qt.CheckState.Checked)

    def _uncheck_all_behaviors(self) -> None:
        for i in range(self._behavior_list.count()):
            self._behavior_list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _refresh_section_filter(self) -> None:
        """Rebuild the section filter list from current sections / trial types."""
        # Determine names to show based on aggregate mode
        raw_sections = self._get_sections()
        if self._get_aggregate() == "type":
            seen: dict[str, None] = {}
            for name, _ in raw_sections:
                seen.setdefault(self._section_base_type(name), None)
            names = list(seen.keys())
        else:
            names = [name for name, _ in raw_sections]

        # Preserve checked state for names that existed before
        prev_checked: set[str] = set()
        for i in range(self._section_filter_list.count()):
            item = self._section_filter_list.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                prev_checked.add(item.text())

        self._section_filter_list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if (not prev_checked or name in prev_checked)
                else Qt.CheckState.Unchecked
            )
            self._section_filter_list.addItem(item)
        self._refresh_section_type_selector(names)

    def _refresh_section_type_selector(self, names: list[str]) -> None:
        """Populate section-type combo from currently visible section names."""
        prev_type = str(self._section_type_combo.currentData() or "")
        seen: dict[str, None] = {}
        for name in names:
            base = self._section_base_type(str(name))
            if base:
                seen.setdefault(base, None)
        type_names = list(seen.keys())

        self._section_type_combo.blockSignals(True)
        self._section_type_combo.clear()
        self._section_type_combo.addItem("-- section type --", userData="")
        for tname in type_names:
            self._section_type_combo.addItem(tname, userData=tname)

        if prev_type:
            for i in range(self._section_type_combo.count()):
                if str(self._section_type_combo.itemData(i) or "") == prev_type:
                    self._section_type_combo.setCurrentIndex(i)
                    break
        self._section_type_combo.blockSignals(False)

        has_types = bool(type_names)
        self._section_type_combo.setEnabled(has_types)
        self._section_type_add_btn.setEnabled(has_types)
        self._section_type_only_btn.setEnabled(has_types)

    def _select_sections_by_type(self, only: bool) -> None:
        """Bulk-check visible sections whose base type matches the combo selection."""
        selected_type = str(self._section_type_combo.currentData() or "").strip()
        if not selected_type:
            return

        for i in range(self._section_filter_list.count()):
            item = self._section_filter_list.item(i)
            if item is None:
                continue
            name = str(item.text())
            item_type = self._section_base_type(name)
            is_match = item_type == selected_type
            if only:
                item.setCheckState(
                    Qt.CheckState.Checked if is_match else Qt.CheckState.Unchecked
                )
            elif is_match:
                item.setCheckState(Qt.CheckState.Checked)

    def _checked_section_names(self) -> set[str]:
        out: set[str] = set()
        for i in range(self._section_filter_list.count()):
            item = self._section_filter_list.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                out.add(item.text())
        return out

    def _check_all_sections(self) -> None:
        for i in range(self._section_filter_list.count()):
            self._section_filter_list.item(i).setCheckState(Qt.CheckState.Checked)

    def _uncheck_all_sections(self) -> None:
        for i in range(self._section_filter_list.count()):
            self._section_filter_list.item(i).setCheckState(Qt.CheckState.Unchecked)

    # ── Data computation ──────────────────────────────────────────────

    def _compute_section_data(self) -> "pd.DataFrame":
        """Compute per-session per-section bout count and duration.

        Returns a DataFrame with columns:
            session_id, session_label, behavior_id, behavior,
            section_idx, section_name, n_bouts, duration_s
        """
        sections = self._get_sections()
        if not sections:
            return pd.DataFrame()

        raw_bouts = self._host._raw_bouts
        if not raw_bouts:
            return pd.DataFrame()

        fps = self._host._project_fps()
        if fps <= 0:
            fps = 25.0

        # Section boundaries as [start_s, end_s)
        boundaries: list[tuple[float, float, str, int]] = []
        t = 0.0
        for idx, (sec_name, dur) in enumerate(sections):
            boundaries.append((t, t + dur, sec_name, idx))
            t += dur

        checked_bids = set(self._checked_behavior_ids())
        checked_sessions = self._host._summary_tab._checked_subjects()
        rows: list[dict] = []

        for bid, df in raw_bouts.items():
            if bid not in checked_bids:
                continue
            if df.empty or not {"session_id", "start_frame", "end_frame"}.issubset(
                df.columns
            ):
                continue

            bname = bid
            if "behavior" in df.columns and len(df) > 0:
                bname = str(df["behavior"].iloc[0])

            for sid, grp in df.groupby("session_id"):
                sid_str = str(sid)
                sess_label = self._host._session_label_by_session.get(
                    sid_str, sid_str
                )
                if checked_sessions and sess_label not in checked_sessions:
                    continue
                start_s = grp["start_frame"].to_numpy(dtype=float) / fps
                end_s = grp["end_frame"].to_numpy(dtype=float) / fps

                for sec_start, sec_end, sec_name, sec_idx in boundaries:
                    overlap = np.maximum(
                        0.0,
                        np.minimum(end_s, sec_end) - np.maximum(start_s, sec_start),
                    )
                    n = int((overlap > 0.0).sum())
                    dur_s = float(overlap.sum())
                    sec_dur = sec_end - sec_start
                    pct_time = (dur_s / sec_dur * 100.0) if sec_dur > 0 else 0.0
                    rows.append(
                        {
                            "session_id": sid_str,
                            "session_label": sess_label,
                            "behavior_id": bid,
                            "behavior": bname,
                            "section_idx": sec_idx,
                            "section_name": sec_name,
                            "n_bouts": n,
                            "duration_s": dur_s,
                            "pct_time": pct_time,
                        }
                    )

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Chart helpers ─────────────────────────────────────────────────

    def _gs(self) -> dict:
        return self._host._graph_settings

    def _get_chart_style(self) -> str:
        for key, btn in self._chart_btns.items():
            if btn.isChecked():
                return key
        return "bar"

    def _get_metric(self) -> str:
        for key, btn in self._metric_btns.items():
            if btn.isChecked():
                return key
        return "n_bouts"

    def _get_mode(self) -> str:
        for key, btn in self._mode_btns.items():
            if btn.isChecked():
                return key
        return "individual"

    def _get_aggregate(self) -> str:
        for key, btn in self._agg_btns.items():
            if btn.isChecked():
                return key
        return "section"

    @staticmethod
    def _section_base_type(name: str) -> str:
        """Strip trailing number from a section name to get the trial type.

        'Tone 1' → 'Tone',  'ITI 3' → 'ITI',  'Baseline' → 'Baseline'
        """
        import re
        stripped = re.sub(r"\s*\d+\s*$", "", name).strip()
        return stripped if stripped else name

    def _style_ax(self, ax: Any) -> None:
        ax.set_facecolor(self._host._graph_settings.get("fig_bg", "#ffffff"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Public API ────────────────────────────────────────────────────

    def on_data_loaded(self) -> None:
        """Called when new analytics data is loaded."""
        self._refresh_behavior_selector()
        self._refresh_section_filter()
        self._refresh_factor_selector()
        self._refresh_group_filter()

    def on_groups_updated(self) -> None:
        """Called when the host's group assignments change externally."""
        self._refresh_factor_selector()
        self._refresh_group_filter()

    # ── Faceted grouping helpers ──────────────────────────────────────

    def _on_mode_toggled(self) -> None:
        mode = self._get_mode()
        self._group_controls.setVisible(mode == "group")

    def _refresh_factor_selector(self) -> None:
        """Rebuild the per-factor facet dropdowns when factors change."""
        self._facet.blockSignals(True)
        self._facet.rebuild(
            list(self._host._factor_definitions), self._host._levels_by_factor()
        )
        if not self._ss_facet_controls and self._host._factor_definitions:
            self._ss_facet_controls = self._host._default_facet_controls()
        self._facet.set_state(self._ss_facet_controls)
        self._facet.blockSignals(False)

    def _refresh_group_filter(self) -> None:
        """No-op retained for callers: facets filter via the dropdowns."""
        return

    def _checked_groups(self) -> set[str]:
        """All series the facet controls currently produce."""
        return {
            g for g in self._host._session_groups_for_controls(
                self._ss_facet_controls
            ).values()
            if g
        }

    def _on_facets_changed(self) -> None:
        self._ss_facet_controls = self._facet.state()
        self._update_plot()

    # ── Plot dispatch ─────────────────────────────────────────────────

    def _aggregate_to_types(
        self, df: "pd.DataFrame", sections: list[tuple[str, float]]
    ) -> tuple["pd.DataFrame", list[tuple[str, float]]]:
        """Average section-level rows into trial-type rows.

        Sections named 'Tone 1', 'Tone 2', ... are averaged per session
        into a single 'Tone' row.  The returned section list preserves the
        original order of first occurrence of each type.
        """
        # Build ordered list of unique base types (preserving encounter order)
        seen: dict[str, None] = {}
        for name, dur in sections:
            bt = self._section_base_type(name)
            seen.setdefault(bt, None)
        type_order = list(seen.keys())

        if df.empty:
            type_sections = [(t, 0.0) for t in type_order]
            return df, type_sections

        df = df.copy()
        df["trial_type"] = df["section_name"].apply(self._section_base_type)

        agg_rows = []
        for (sid, slbl, bid, beh, trial_type), grp in df.groupby(
            ["session_id", "session_label", "behavior_id", "behavior", "trial_type"],
            sort=False,
        ):
            agg_rows.append({
                "session_id": sid,
                "session_label": slbl,
                "behavior_id": bid,
                "behavior": beh,
                "section_idx": type_order.index(trial_type),
                "section_name": trial_type,
                "n_bouts": float(grp["n_bouts"].mean()),
                "duration_s": float(grp["duration_s"].mean()),
                "pct_time": float(grp["pct_time"].mean()) if "pct_time" in grp.columns else 0.0,
            })

        type_sections = [(t, 0.0) for t in type_order]
        return (
            pd.DataFrame(agg_rows) if agg_rows else pd.DataFrame(),
            type_sections,
        )

    @staticmethod
    def _trailing_int(name: str) -> int | None:
        import re
        m = re.search(r"(\d+)\s*$", str(name))
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _apply_section_binning(
        self,
        df: "pd.DataFrame",
        sections: list[tuple[str, float]],
    ) -> tuple["pd.DataFrame", list[tuple[str, float]]]:
        """Average repeated selected sections into fixed-size bins by base type.

        Binning is applied per base type (e.g., ITI, Tone) only when the
        count of selected sections for that type is >= bin size.
        """
        if df.empty or not sections:
            return df, sections
        if not bool(self._bin_sections_chk.isChecked()):
            return df, sections

        bin_size = int(self._bin_size_spin.value())
        if bin_size < 2:
            return df, sections

        sec_names = [n for n, _ in sections]
        sec_durs = [float(d) for _, d in sections]
        by_type: dict[str, list[int]] = {}
        for idx, name in enumerate(sec_names):
            bt = self._section_base_type(name)
            by_type.setdefault(bt, []).append(idx)

        # Build source-section -> binned-section mapping while preserving order.
        source_to_label: dict[str, str] = {}
        label_to_src_names: dict[str, list[str]] = {}
        label_to_avg_dur: dict[str, float] = {}
        idx_to_label: dict[int, str] = {}
        idx_to_is_rep: dict[int, bool] = {}

        for bt, idxs in by_type.items():
            if len(idxs) < bin_size:
                for idx in idxs:
                    nm = sec_names[idx]
                    idx_to_label[idx] = nm
                    idx_to_is_rep[idx] = True
                    source_to_label[nm] = nm
                    label_to_src_names[nm] = [nm]
                    label_to_avg_dur[nm] = sec_durs[idx]
                continue

            bin_no = 1
            pos = 0
            while pos < len(idxs):
                chunk = idxs[pos:pos + bin_size]
                # Average any full bin, and also any trailing remainder of 2+
                # into a final partial bin (e.g. 19 ITIs at bin 5 ->
                # 1-5, 6-10, 11-15, 16-19) rather than splitting it out. A
                # lone straggler (len 1) can't be averaged, so it stays as-is.
                if len(chunk) >= 2:
                    first_name = sec_names[chunk[0]]
                    last_name = sec_names[chunk[-1]]
                    n0 = self._trailing_int(first_name)
                    n1 = self._trailing_int(last_name)
                    if n0 is not None and n1 is not None:
                        label = f"{bt} {n0}-{n1}"
                    else:
                        label = f"{bt} Bin {bin_no}"
                    bin_no += 1

                    src_names = [sec_names[i] for i in chunk]
                    avg_dur = float(np.mean([sec_durs[i] for i in chunk]))

                    for j, idx in enumerate(chunk):
                        idx_to_label[idx] = label
                        idx_to_is_rep[idx] = j == 0
                        source_to_label[sec_names[idx]] = label

                    label_to_src_names[label] = src_names
                    label_to_avg_dur[label] = avg_dur
                else:
                    # Lone straggler (single section) can't be averaged;
                    # it remains unbinned under its own name.
                    for idx in chunk:
                        nm = sec_names[idx]
                        idx_to_label[idx] = nm
                        idx_to_is_rep[idx] = True
                        source_to_label[nm] = nm
                        label_to_src_names[nm] = [nm]
                        label_to_avg_dur[nm] = sec_durs[idx]
                pos += bin_size

        # Preserve timeline order using representative indices.
        binned_sections: list[tuple[str, float]] = []
        seen_labels: set[str] = set()
        for idx in range(len(sec_names)):
            if not idx_to_is_rep.get(idx, False):
                continue
            label = idx_to_label.get(idx, sec_names[idx])
            if label in seen_labels:
                continue
            seen_labels.add(label)
            binned_sections.append((label, float(label_to_avg_dur.get(label, sec_durs[idx]))))

        # If no effective change, return original.
        if len(binned_sections) == len(sections) and all(
            a[0] == b[0] for a, b in zip(binned_sections, sections)
        ):
            return df, sections

        work = df.copy()
        work["section_name"] = work["section_name"].map(
            lambda n: source_to_label.get(str(n), str(n))
        )

        grouped = (
            work.groupby(
                [
                    "session_id",
                    "session_label",
                    "behavior_id",
                    "behavior",
                    "section_name",
                ],
                sort=False,
                dropna=False,
            )
            .agg(
                n_bouts=("n_bouts", "mean"),
                duration_s=("duration_s", "mean"),
                pct_time=("pct_time", "mean"),
            )
            .reset_index()
        )

        sec_order = {name: i for i, (name, _d) in enumerate(binned_sections)}
        grouped["section_idx"] = grouped["section_name"].map(
            lambda n: int(sec_order.get(str(n), 0))
        )

        return grouped, binned_sections

    def _update_plot(self) -> None:
        if self._figure is None or self._updating:
            return
        self._updating = True
        try:
            sections = self._get_sections()
            if not sections:
                self._figure.clear()
                ax = self._figure.add_subplot(111)
                ax.text(
                    0.5, 0.5,
                    "No sections defined.\nAdd sections in the panel on the left.",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#90a4ae", fontsize=11,
                )
                ax.axis("off")
                self._figure.set_facecolor(
                    self._host._graph_settings.get("fig_bg", "#ffffff")
                )
                self._canvas.draw()
                return

            df = self._compute_section_data()
            if df.empty:
                self._figure.clear()
                ax = self._figure.add_subplot(111)
                ax.text(
                    0.5, 0.5,
                    "No bout data found for the selected behaviors.",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#90a4ae", fontsize=11,
                )
                ax.axis("off")
                self._figure.set_facecolor(
                    self._host._graph_settings.get("fig_bg", "#ffffff")
                )
                self._canvas.draw()
                return

            # Apply trial-type aggregation if requested
            if self._get_aggregate() == "type":
                df, sections = self._aggregate_to_types(df, sections)

            # Filter to only checked sections
            checked_secs = self._checked_section_names()
            if checked_secs:
                sections = [(n, d) for n, d in sections if n in checked_secs]
                if not df.empty and "section_name" in df.columns:
                    df = df[df["section_name"].isin(checked_secs)]

            # Optional averaging bins for repeated selected section types.
            # Applies to By Section mode in bar/line charts.
            style = self._get_chart_style()
            if self._get_aggregate() == "section" and style in {"bar", "line"}:
                df, sections = self._apply_section_binning(df, sections)

            if not sections or df.empty:
                self._figure.clear()
                ax = self._figure.add_subplot(111)
                ax.text(
                    0.5, 0.5,
                    "No sections selected.\nCheck at least one section in the filter list.",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#90a4ae", fontsize=11,
                )
                ax.axis("off")
                self._figure.set_facecolor(
                    self._host._graph_settings.get("fig_bg", "#ffffff")
                )
                self._canvas.draw()
                return

            metric = self._get_metric()
            mode = self._get_mode()

            if style == "bar":
                self._draw_bar_chart(df, metric, mode, sections)
            elif style == "line":
                self._draw_line_chart(df, metric, mode, sections)
            else:
                self._draw_heatmap_chart(df, metric, mode, sections)
            # Scale the freshly-drawn figure to fill the viewport width.
            self._sync_canvas_to_viewport()
        finally:
            self._updating = False

    def _figure_layout(
        self, n_beh: int
    ) -> tuple[int, int, int, int, int, int]:
        """Return (ncols, nrows, dpi, max_w, px_w, px_h) for n_beh subplots."""
        gs = self._host._graph_settings
        dpi = int(gs.get("dpi", 100))
        max_w = int(gs.get("max_w", 700))
        ncols = min(n_beh, 3)
        nrows = max(1, (n_beh + ncols - 1) // ncols)
        ref_w = max_w / dpi
        cell_w = ref_w / max(ncols, 1.5)
        cell_h = cell_w * 0.85
        fig_w = max(ref_w * 0.8, ncols * cell_w)
        fig_h = max(ref_w * 0.5, nrows * cell_h)
        px_w = min(int(fig_w * dpi), max_w)
        px_h = int(fig_h * dpi)
        return ncols, nrows, dpi, max_w, px_w, px_h

    def _place_group_legend(
        self,
        shared_h: list,
        shared_l: list,
        gs: dict,
        n_beh: int,
        ncols: int,
        nrows: int,
    ) -> None:
        """Add the shared group legend and run tight_layout.

        When the subplot grid leaves empty cells (e.g. 4 behaviours in a 3×2
        grid), drop the legend into that genuinely empty bottom-right region —
        ``tight_layout`` never expands axes into empty grid cells, so the legend
        cannot be overrun (unlike the reserved-right-margin strip, which
        ``_autofill_canvas``'s re-run of plain ``tight_layout`` would overlap).
        Only when the grid is full do we fall back to the reserved-margin strip.
        """
        if not shared_h:
            try:
                self._figure.tight_layout(pad=1.5)
            except Exception:
                pass
            return

        filled_last_row = n_beh - (nrows - 1) * ncols
        empty_cells = ncols - filled_last_row if nrows >= 1 else 0

        if empty_cells >= 1:
            # Centre of the empty cells in the last (bottom) row, figure coords.
            x_center = (filled_last_row + ncols) / 2.0 / ncols
            y_center = (0.5 / nrows)
            self._figure.legend(
                shared_h, shared_l,
                fontsize=gs.get("legend_fontsize", "small"),
                loc="center",
                bbox_to_anchor=(x_center, y_center),
                bbox_transform=self._figure.transFigure,
                frameon=True,
                framealpha=0.9,
            )
            try:
                self._figure.tight_layout(pad=1.5, rect=[0, 0, 1, 0.97])
            except Exception:
                pass
        else:
            _fig_w_px = (
                self._canvas.width() if self._canvas
                else int(self._host._graph_settings.get("max_w", 700))
            )
            _rr, _lx = _legend_right_margin(shared_l, _fig_w_px)
            self._figure.legend(
                shared_h, shared_l,
                fontsize=gs.get("legend_fontsize", "small"),
                loc="upper left",
                bbox_to_anchor=(_lx, 0.98),
                bbox_transform=self._figure.transFigure,
                frameon=True,
                framealpha=0.9,
            )
            try:
                self._figure.tight_layout(pad=1.5, rect=[0, 0, _rr, 0.97])
            except Exception:
                pass

    # ── Bar chart ─────────────────────────────────────────────────────

    def _draw_bar_chart(
        self,
        df: "pd.DataFrame",
        metric: str,
        mode: str,
        sections: list[tuple[str, float]],
    ) -> None:
        gs = self._host._graph_settings
        fig_bg = gs.get("fig_bg", "#ffffff")
        error_style = gs.get("error_style", "SEM")
        section_names = [s[0] for s in sections]
        behaviors = sorted(df["behavior"].unique().tolist())
        n_beh = len(behaviors)
        ncols, nrows, dpi, max_w, px_w, px_h = self._figure_layout(n_beh)

        self._figure.set_size_inches(px_w / dpi, px_h / dpi)
        if self._canvas is not None:
            self._canvas.setFixedSize(px_w, px_h)
        self._figure.clear()
        self._figure.set_facecolor(fig_bg)

        metric_label = (
            "Bout Count" if metric == "n_bouts"
            else "% Time in Section" if metric == "pct_time"
            else "Duration (s)"
        )

        # Use the tab-local factor for group mode
        host_groups = (
            self._host._session_groups_for_controls(self._ss_facet_controls)
            if mode == "group" else {}
        )
        checked_grps = self._checked_groups() if mode == "group" else set()

        for beh_idx, beh_name in enumerate(behaviors):
            ax = self._figure.add_subplot(nrows, ncols, beh_idx + 1)
            beh_df = df[df["behavior"] == beh_name]
            x = np.arange(len(section_names))

            if mode == "individual":
                sess_order = self._host.ordered_session_labels()
                sessions_here = [
                    s for s in sess_order
                    if s in beh_df["session_label"].unique()
                ]
                bar_w = 0.8 / max(len(sessions_here), 1)
                for s_idx, sess in enumerate(sessions_here):
                    sess_df = beh_df[beh_df["session_label"] == sess]
                    vals = [
                        float(
                            sess_df[sess_df["section_name"] == sn][metric].sum()
                        )
                        if sn in sess_df["section_name"].values
                        else 0.0
                        for sn in section_names
                    ]
                    offset = (s_idx - len(sessions_here) / 2 + 0.5) * bar_w
                    color = _PALETTE[s_idx % len(_PALETTE)]
                    ax.bar(
                        x + offset, vals, bar_w * 0.9,
                        color=color, alpha=0.85, label=sess,
                    )
            else:
                if not host_groups:
                    vals = []
                    errs = []
                    for sn in section_names:
                        sec_vals = (
                            beh_df[beh_df["section_name"] == sn]
                            .groupby("session_label")[metric]
                            .sum()
                            .to_numpy(float)
                        )
                        vals.append(
                            float(sec_vals.mean()) if len(sec_vals) > 0 else 0.0
                        )
                        errs.append(_eb_val(sec_vals, error_style))
                    ax.bar(
                        x, vals, 0.6, color=_PALETTE[0], alpha=0.85,
                        yerr=errs,
                        capsize=gs.get("eb_capsize", 4),
                        error_kw={
                            "elinewidth": gs.get("eb_linewidth", 1.0),
                            "capthick": gs.get("eb_linewidth", 1.0),
                        },
                    )
                else:
                    all_groups = self._host._ordered_group_list(
                        set(host_groups.values()),
                        self._host._split_factors_for_controls(self._ss_facet_controls),
                    )
                    groups = [g for g in all_groups if g in checked_grps]
                    bar_w = 0.8 / max(len(groups), 1)
                    for g_idx, grp_name in enumerate(groups):
                        grp_sessions = {
                            lbl
                            for lbl, g in host_groups.items()
                            if g == grp_name
                        }
                        grp_df = beh_df[
                            beh_df["session_label"].isin(grp_sessions)
                        ]
                        vals = []
                        errs = []
                        for sn in section_names:
                            sec_vals = (
                                grp_df[grp_df["section_name"] == sn]
                                .groupby("session_label")[metric]
                                .sum()
                                .to_numpy(float)
                            )
                            vals.append(
                                float(sec_vals.mean())
                                if len(sec_vals) > 0
                                else 0.0
                            )
                            errs.append(_eb_val(sec_vals, error_style))
                        offset = (g_idx - len(groups) / 2 + 0.5) * bar_w
                        color = self._host._group_color(grp_name, g_idx)
                        ax.bar(
                            x + offset, vals, bar_w * 0.9,
                            color=color, alpha=0.85,
                            yerr=errs,
                            capsize=gs.get("eb_capsize", 4),
                            error_kw={
                                "elinewidth": gs.get("eb_linewidth", 1.0),
                                "capthick": gs.get("eb_linewidth", 1.0),
                            },
                            label=grp_name,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(
                section_names, rotation=30, ha="right",
                fontsize=gs.get("tick_fontsize", 8),
            )
            ax.set_title(str(beh_name), fontsize=gs.get("title_fontsize", 12))
            ax.set_ylabel(metric_label, fontsize=gs.get("axis_fontsize", 10))
            ax.tick_params(labelsize=gs.get("tick_fontsize", 8))
            self._style_ax(ax)

        # Single shared legend for group mode; per-axes legend for individual mode
        if mode == "group":
            _shared_h: list[Any] = []
            _shared_l: list[str] = []
            for _a in self._figure.axes:
                _leg = _a.get_legend()
                if _leg is not None:
                    _leg.remove()
                _h, _ll = _a.get_legend_handles_labels()
                if _h and not _shared_h:
                    _shared_h, _shared_l = _h, _ll
            self._place_group_legend(_shared_h, _shared_l, gs, n_beh, ncols, nrows)
        else:
            for _a in self._figure.axes:
                _h, _ll = _a.get_legend_handles_labels()
                if _h:
                    _a.legend(
                        fontsize=gs.get("legend_fontsize", "small"),
                        loc=gs.get("legend_loc", "best"),
                    )
            try:
                self._figure.tight_layout(pad=1.5)
            except Exception:
                pass
        self._canvas.draw()

    # ── Line chart ────────────────────────────────────────────────────

    def _draw_line_chart(
        self,
        df: "pd.DataFrame",
        metric: str,
        mode: str,
        sections: list[tuple[str, float]],
    ) -> None:
        gs = self._host._graph_settings
        fig_bg = gs.get("fig_bg", "#ffffff")
        error_style = gs.get("error_style", "SEM")
        section_names = [s[0] for s in sections]
        behaviors = sorted(df["behavior"].unique().tolist())
        n_beh = len(behaviors)
        ncols, nrows, dpi, max_w, px_w, px_h = self._figure_layout(n_beh)

        self._figure.set_size_inches(px_w / dpi, px_h / dpi)
        if self._canvas is not None:
            self._canvas.setFixedSize(px_w, px_h)
        self._figure.clear()
        self._figure.set_facecolor(fig_bg)

        metric_label = (
            "Bout Count" if metric == "n_bouts"
            else "% Time in Section" if metric == "pct_time"
            else "Duration (s)"
        )
        show_pts = gs.get("show_indiv_points", True)
        x = np.arange(len(section_names))

        host_groups = (
            self._host._session_groups_for_controls(self._ss_facet_controls)
            if mode == "group" else {}
        )
        checked_grps = self._checked_groups() if mode == "group" else set()

        for beh_idx, beh_name in enumerate(behaviors):
            ax = self._figure.add_subplot(nrows, ncols, beh_idx + 1)
            beh_df = df[df["behavior"] == beh_name]

            if mode == "individual":
                sess_order = self._host.ordered_session_labels()
                sessions_here = [
                    s for s in sess_order
                    if s in beh_df["session_label"].unique()
                ]
                for s_idx, sess in enumerate(sessions_here):
                    sess_df = beh_df[beh_df["session_label"] == sess]
                    vals = [
                        float(
                            sess_df[sess_df["section_name"] == sn][metric].sum()
                        )
                        if sn in sess_df["section_name"].values
                        else 0.0
                        for sn in section_names
                    ]
                    color = _PALETTE[s_idx % len(_PALETTE)]
                    ax.plot(
                        x, vals, color=color, linewidth=1.8,
                        marker="o" if show_pts else None,
                        markersize=4, label=sess, alpha=0.85,
                    )
            else:
                if not host_groups:
                    vals = []
                    errs = []
                    for sn in section_names:
                        sec_vals = (
                            beh_df[beh_df["section_name"] == sn]
                            .groupby("session_label")[metric]
                            .sum()
                            .to_numpy(float)
                        )
                        vals.append(
                            float(sec_vals.mean()) if len(sec_vals) > 0 else 0.0
                        )
                        errs.append(_eb_val(sec_vals, error_style))
                    ax.errorbar(
                        x, vals, yerr=errs, color=_PALETTE[0], linewidth=2,
                        marker="o", markersize=5,
                        capsize=gs.get("eb_capsize", 4),
                        elinewidth=gs.get("eb_linewidth", 1.0),
                    )
                else:
                    all_groups = self._host._ordered_group_list(
                        set(host_groups.values()),
                        self._host._split_factors_for_controls(self._ss_facet_controls),
                    )
                    groups = [g for g in all_groups if g in checked_grps]
                    for g_idx, grp_name in enumerate(groups):
                        grp_sessions = {
                            lbl
                            for lbl, g in host_groups.items()
                            if g == grp_name
                        }
                        grp_df = beh_df[
                            beh_df["session_label"].isin(grp_sessions)
                        ]
                        vals = []
                        errs = []
                        for sn in section_names:
                            sec_vals = (
                                grp_df[grp_df["section_name"] == sn]
                                .groupby("session_label")[metric]
                                .sum()
                                .to_numpy(float)
                            )
                            vals.append(
                                float(sec_vals.mean())
                                if len(sec_vals) > 0
                                else 0.0
                            )
                            errs.append(_eb_val(sec_vals, error_style))
                        color = self._host._group_color(grp_name, g_idx)
                        ax.errorbar(
                            x, vals, yerr=errs, color=color, linewidth=2,
                            marker="o", markersize=5,
                            capsize=gs.get("eb_capsize", 4),
                            elinewidth=gs.get("eb_linewidth", 1.0),
                            label=grp_name,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(
                section_names, rotation=30, ha="right",
                fontsize=gs.get("tick_fontsize", 8),
            )
            ax.set_title(str(beh_name), fontsize=gs.get("title_fontsize", 12))
            ax.set_ylabel(metric_label, fontsize=gs.get("axis_fontsize", 10))
            ax.tick_params(labelsize=gs.get("tick_fontsize", 8))
            self._style_ax(ax)

        # Single shared legend outside the subplots for group mode
        if mode == "group":
            _shared_h: list[Any] = []
            _shared_l: list[str] = []
            for _a in self._figure.axes:
                _leg = _a.get_legend()
                if _leg is not None:
                    _leg.remove()
                _h, _ll = _a.get_legend_handles_labels()
                if _h and not _shared_h:
                    _shared_h, _shared_l = _h, _ll
            self._place_group_legend(_shared_h, _shared_l, gs, n_beh, ncols, nrows)
        else:
            for _a in self._figure.axes:
                _h, _ll = _a.get_legend_handles_labels()
                if _h:
                    _a.legend(
                        fontsize=gs.get("legend_fontsize", "small"),
                        loc=gs.get("legend_loc", "best"),
                    )
            try:
                self._figure.tight_layout(pad=1.5)
            except Exception:
                pass
        self._canvas.draw()

    # ── Heatmap chart ─────────────────────────────────────────────────

    def _draw_heatmap_chart(
        self,
        df: "pd.DataFrame",
        metric: str,
        mode: str,
        sections: list[tuple[str, float]],
    ) -> None:
        """Draw a behavior heatmap with time bins across selected sections.

        Individual Sessions mode (uncollapsed):
            One subplot per behavior. Rows = subjects, columns = time bins
            spanning the selected section timeline.

        By Group mode (collapsed):
            One subplot per behavior. Rows = groups (or one overall row),
            columns = the same time bins, values averaged across subjects.
        """
        gs = self._host._graph_settings
        fig_bg = gs.get("fig_bg", "#ffffff")
        section_names = [s[0] for s in sections]

        checked_bids = self._checked_behavior_ids()
        bid_to_name: dict[str, str] = {}
        for bid in checked_bids:
            bname = bid
            bdf = self._host._raw_bouts.get(bid)
            if bdf is not None and not bdf.empty and "behavior" in bdf.columns:
                bname = str(bdf["behavior"].iloc[0])
            bid_to_name[bid] = bname
        behaviors = [bid_to_name[bid] for bid in checked_bids]
        n_beh = len(behaviors)
        if n_beh == 0:
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No behaviors selected.",
                ha="center", va="center", transform=ax.transAxes,
                color="#90a4ae", fontsize=11,
            )
            ax.axis("off")
            self._figure.set_facecolor(fig_bg)
            self._canvas.draw()
            return

        metric_label = (
            "Bout Count" if metric == "n_bouts"
            else "% Time in Section" if metric == "pct_time"
            else "Duration (s)"
        )

        # Build timeline intervals from the original section definitions.
        # This keeps heatmaps time-resolved even when aggregate mode is
        # "By Trial Type" (where aggregated section durations can be 0).
        selected_names = [s[0] for s in sections]
        raw_sections = self._get_sections()
        raw_intervals: list[tuple[str, str, float, float]] = []
        t_cur = 0.0
        for raw_name, raw_dur in raw_sections:
            d = max(0.0, float(raw_dur))
            s0 = t_cur
            s1 = t_cur + d
            raw_intervals.append((raw_name, self._section_base_type(raw_name), s0, s1))
            t_cur = s1

        raw_name_set = {n for n, _ in raw_sections}
        select_by_exact_name = bool(selected_names) and all(
            name in raw_name_set for name in selected_names
        )

        timeline_sections: list[tuple[str, float, float]] = []
        if select_by_exact_name:
            sel_set = set(selected_names)
            for raw_name, raw_base, s0, s1 in raw_intervals:
                if raw_name in sel_set and s1 > s0:
                    timeline_sections.append((raw_name, s0, s1))
        else:
            sel_set = set(selected_names)
            for raw_name, raw_base, s0, s1 in raw_intervals:
                if raw_base in sel_set and s1 > s0:
                    timeline_sections.append((raw_name, s0, s1))

        if not timeline_sections:
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No sections selected.",
                ha="center", va="center", transform=ax.transAxes,
                color="#90a4ae", fontsize=11,
            )
            ax.axis("off")
            self._figure.set_facecolor(fig_bg)
            self._canvas.draw()
            return

        # Create equal-width time bins for each selected timeline segment.
        # Important: build bins per segment directly so no inter-segment gaps
        # can appear as artificial first-column data in later sections.
        bins_per_sec = max(4, int(self._HEATMAP_BINS_PER_SECTION))
        bin_ranges: list[tuple[float, float]] = []
        section_boundaries_idx: list[int] = []
        section_centers_idx: list[float] = []
        timeline_labels: list[str] = []
        total_bins = 0
        for sec_name, s0, s1 in timeline_sections:
            edges = np.linspace(s0, s1, bins_per_sec + 1)
            for i in range(bins_per_sec):
                bin_ranges.append((float(edges[i]), float(edges[i + 1])))
            section_boundaries_idx.append(total_bins)
            section_centers_idx.append(total_bins + (bins_per_sec - 1) / 2)
            timeline_labels.append(sec_name)
            total_bins += bins_per_sec
        section_boundaries_idx.append(total_bins)

        bin_starts = np.array([b0 for b0, _ in bin_ranges], dtype=float)
        bin_ends = np.array([b1 for _, b1 in bin_ranges], dtype=float)

        # Session ordering and filtering shared across behaviors.
        checked_sessions = self._host._summary_tab._checked_subjects()
        sess_order = self._host.ordered_session_labels()
        session_labels = [
            s for s in sess_order if (not checked_sessions or s in checked_sessions)
        ]

        if not session_labels:
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No sessions selected.",
                ha="center", va="center", transform=ax.transAxes,
                color="#90a4ae", fontsize=11,
            )
            ax.axis("off")
            self._figure.set_facecolor(fig_bg)
            self._canvas.draw()
            return

        sid_to_label = self._host._session_label_by_session
        label_to_sid = {lbl: sid for sid, lbl in sid_to_label.items()}

        def _subject_time_vector(bid: str, session_label: str) -> np.ndarray:
            out = np.zeros(total_bins, dtype=float)
            sid = label_to_sid.get(session_label)
            if sid is None:
                return out
            bdf = self._host._raw_bouts.get(bid)
            if bdf is None or bdf.empty:
                return out
            if not {"session_id", "start_frame", "end_frame"}.issubset(bdf.columns):
                return out
            sdf = bdf[bdf["session_id"].astype(str) == str(sid)]
            if sdf.empty:
                return out

            fps = self._host._project_fps()
            if fps <= 0:
                fps = 25.0
            starts = sdf["start_frame"].to_numpy(dtype=float) / fps
            ends = sdf["end_frame"].to_numpy(dtype=float) / fps

            for i, (b0, b1) in enumerate(zip(bin_starts, bin_ends, strict=False)):
                overlap = np.maximum(
                    0.0,
                    np.minimum(ends, b1) - np.maximum(starts, b0),
                )
                if metric == "n_bouts":
                    out[i] = float((overlap > 0.0).sum())
                else:
                    dur = float(overlap.sum())
                    if metric == "pct_time":
                        bin_dur = max(b1 - b0, 1e-9)
                        out[i] = dur / bin_dur * 100.0
                    else:
                        out[i] = dur
            return out

        # Choose a colourmap that works on both light and dark backgrounds
        _bg = fig_bg.strip().lstrip("#")
        try:
            _r = int(_bg[0:2], 16)
            _g = int(_bg[2:4], 16)
            _b = int(_bg[4:6], 16)
            _luminance = (0.299 * _r + 0.587 * _g + 0.114 * _b) / 255
        except Exception:
            _luminance = 1.0
        cmap_name = "YlOrRd" if _luminance > 0.5 else "plasma"

        host_groups = (
            self._host._session_groups_for_controls(self._ss_facet_controls)
            if mode == "group" else {}
        )
        checked_grps = self._checked_groups() if mode == "group" else set()

        if mode == "individual":
            # ── Uncollapsed: one subplot per behavior ─────────────────
            # Rows = subjects, columns = time bins across sections
            sess_order = self._host.ordered_session_labels()
            ncols, nrows, dpi, max_w, px_w, px_h = self._figure_layout(n_beh)

            self._figure.set_size_inches(px_w / dpi, px_h / dpi)
            if self._canvas is not None:
                self._canvas.setFixedSize(px_w, px_h)
            self._figure.clear()
            self._figure.set_facecolor(fig_bg)

            for beh_idx, bid in enumerate(checked_bids):
                beh_name = bid_to_name.get(bid, bid)
                ax = self._figure.add_subplot(nrows, ncols, beh_idx + 1)
                sessions_here = [s for s in sess_order if s in session_labels]
                if not sessions_here:
                    ax.axis("off")
                    continue

                mat = np.zeros((len(sessions_here), total_bins))
                for s_idx, sess in enumerate(sessions_here):
                    mat[s_idx, :] = _subject_time_vector(bid, sess)

                vmax = mat.max() if mat.max() > 0 else 1.0
                im = ax.imshow(
                    mat, aspect="auto", cmap=cmap_name,
                    interpolation="nearest", vmin=0, vmax=vmax,
                )
                self._figure.colorbar(
                    im, ax=ax, label=metric_label,
                    pad=0.02, fraction=0.046,
                )
                ax.set_xticks(np.array(section_centers_idx, dtype=float))
                ax.set_xticklabels(
                    timeline_labels, rotation=30, ha="right",
                    fontsize=gs.get("tick_fontsize", 8),
                )
                for x_idx in section_boundaries_idx[1:-1]:
                    ax.axvline(x_idx - 0.5, color="#90a4ae", linewidth=0.7, alpha=0.6)
                ax.set_yticks(np.arange(len(sessions_here)))
                ax.set_yticklabels(
                    sessions_here, fontsize=gs.get("tick_fontsize", 8)
                )
                ax.set_title(
                    str(beh_name), fontsize=gs.get("title_fontsize", 12)
                )
                ax.set_xlabel(
                    "Time Across Selected Sections", fontsize=gs.get("axis_fontsize", 10)
                )
                ax.set_ylabel(
                    "Subject", fontsize=gs.get("axis_fontsize", 10)
                )
                self._style_ax(ax)

        else:
            # ── Collapsed: one subplot per behavior; rows=groups, cols=time bins ──
            all_groups = self._host._ordered_group_list(
                set(host_groups.values()),
                self._host._split_factors_for_controls(self._ss_facet_controls),
            ) if host_groups else []
            groups = [g for g in all_groups if g in checked_grps] if host_groups else []
            if not groups:
                groups = ["All Subjects"]

            n_plots = n_beh
            ncols, nrows, dpi, max_w, px_w, px_h = self._figure_layout(n_plots)
            self._figure.set_size_inches(px_w / dpi, px_h / dpi)
            if self._canvas is not None:
                self._canvas.setFixedSize(px_w, px_h)
            self._figure.clear()
            self._figure.set_facecolor(fig_bg)

            for beh_idx, bid in enumerate(checked_bids):
                beh_name = bid_to_name.get(bid, bid)
                ax = self._figure.add_subplot(nrows, ncols, beh_idx + 1)

                row_labels = list(groups)
                mat = np.zeros((len(row_labels), total_bins), dtype=float)
                for r_idx, grp_name in enumerate(row_labels):
                    if grp_name == "All Subjects":
                        grp_sessions = list(session_labels)
                    else:
                        grp_sessions = [
                            s for s in session_labels
                            if host_groups.get(s, "") == grp_name
                        ]
                    if not grp_sessions:
                        continue
                    subj_mat = np.vstack([
                        _subject_time_vector(bid, sess) for sess in grp_sessions
                    ])
                    mat[r_idx, :] = subj_mat.mean(axis=0)

                vmax = mat.max() if mat.max() > 0 else 1.0
                im = ax.imshow(
                    mat, aspect="auto", cmap=cmap_name,
                    interpolation="nearest", vmin=0, vmax=vmax,
                )
                self._figure.colorbar(
                    im, ax=ax, label=metric_label,
                    pad=0.02, fraction=0.046,
                )
                ax.set_xticks(np.array(section_centers_idx, dtype=float))
                ax.set_xticklabels(
                    timeline_labels, rotation=30, ha="right",
                    fontsize=gs.get("tick_fontsize", 8),
                )
                for x_idx in section_boundaries_idx[1:-1]:
                    ax.axvline(x_idx - 0.5, color="#90a4ae", linewidth=0.7, alpha=0.6)
                ax.set_yticks(np.arange(len(row_labels)))
                ax.set_yticklabels(row_labels, fontsize=gs.get("tick_fontsize", 8))
                ax.set_title(str(beh_name), fontsize=gs.get("title_fontsize", 12))
                ax.set_xlabel(
                    "Time Across Selected Sections", fontsize=gs.get("axis_fontsize", 10)
                )
                ax.set_ylabel(
                    "Group", fontsize=gs.get("axis_fontsize", 10)
                )
                self._style_ax(ax)

        try:
            self._figure.tight_layout(pad=1.5)
        except Exception:
            pass
        self._canvas.draw()

    # ── Graph settings ────────────────────────────────────────────────

    def _open_graph_settings(self) -> None:
        gs = self._gs()
        dlg = QDialog(self)
        dlg.setWindowTitle("Section Sections Graph Settings")
        dlg.setMinimumWidth(360)
        form = QFormLayout()
        form.setSpacing(8)

        title_fs = QSpinBox(dlg)
        title_fs.setRange(6, 32)
        title_fs.setValue(int(gs.get("title_fontsize", 11)))
        form.addRow("Title font size:", title_fs)

        axis_fs = QSpinBox(dlg)
        axis_fs.setRange(4, 28)
        axis_fs.setValue(int(gs.get("axis_fontsize", 9)))
        form.addRow("Axis label font size:", axis_fs)

        tick_fs = QSpinBox(dlg)
        tick_fs.setRange(4, 24)
        tick_fs.setValue(int(gs.get("tick_fontsize", 8)))
        form.addRow("Tick font size:", tick_fs)

        error_cb = QComboBox(dlg)
        for _es in ("SEM", "SD", "95% CI", "None"):
            error_cb.addItem(_es, userData=_es)
        error_cb.setCurrentText(str(gs.get("error_style", "SEM")))
        error_cb.setToolTip(
            "Error bar / band style.\n"
            "SEM = standard error of the mean\n"
            "SD = standard deviation\n"
            "95% CI = 1.96 \u00d7 SEM\n"
            "None = no error bars"
        )
        form.addRow("Error bar style:", error_cb)

        indiv_chk = QCheckBox("Show individual data points", dlg)
        indiv_chk.setChecked(bool(gs.get("show_indiv_points", True)))
        form.addRow("", indiv_chk)

        eb_cap = QSpinBox(dlg)
        eb_cap.setRange(0, 20)
        eb_cap.setValue(int(gs.get("eb_capsize", 4)))
        eb_cap.setSuffix(" pt")
        form.addRow("Error bar cap width:", eb_cap)

        eb_lw = QDoubleSpinBox(dlg)
        eb_lw.setRange(0.2, 6.0)
        eb_lw.setSingleStep(0.2)
        eb_lw.setDecimals(1)
        eb_lw.setValue(float(gs.get("eb_linewidth", 1.0)))
        eb_lw.setSuffix(" pt")
        form.addRow("Error bar line thickness:", eb_lw)

        dpi_spin = QSpinBox(dlg)
        dpi_spin.setRange(50, 600)
        dpi_spin.setSingleStep(50)
        dpi_spin.setValue(int(gs.get("dpi", 150)))
        form.addRow("Export DPI:", dpi_spin)

        max_w_spin = QSpinBox(dlg)
        max_w_spin.setRange(200, 3840)
        max_w_spin.setSingleStep(50)
        max_w_spin.setSuffix(" px")
        max_w_spin.setValue(int(gs.get("max_w", 700)))
        form.addRow("Max display width:", max_w_spin)

        max_h_spin = QSpinBox(dlg)
        max_h_spin.setRange(150, 2160)
        max_h_spin.setSingleStep(50)
        max_h_spin.setSuffix(" px")
        max_h_spin.setValue(int(gs.get("max_h", 420)))
        form.addRow("Max display height:", max_h_spin)

        force_fit_chk = QCheckBox("Force fit to canvas size", dlg)
        force_fit_chk.setChecked(bool(gs.get("force_fit", True)))
        form.addRow("", force_fit_chk)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout(dlg)
        layout.addLayout(form)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        gs["title_fontsize"] = title_fs.value()
        gs["axis_fontsize"] = axis_fs.value()
        gs["tick_fontsize"] = tick_fs.value()
        gs["error_style"] = str(error_cb.currentData() or "SEM")
        gs["show_indiv_points"] = indiv_chk.isChecked()
        gs["eb_capsize"] = eb_cap.value()
        gs["eb_linewidth"] = eb_lw.value()
        gs["dpi"] = dpi_spin.value()
        gs["max_w"] = max_w_spin.value()
        gs["max_h"] = max_h_spin.value()
        gs["force_fit"] = force_fit_chk.isChecked()
        self._update_plot()

    # ── Export ────────────────────────────────────────────────────────

    def _export_figure(self) -> None:
        if self._figure is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Section Chart", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        dpi = int(self._host._graph_settings.get("dpi", 150))
        try:
            self._figure.savefig(path, dpi=dpi, bbox_inches="tight")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))

    def _export_data(self) -> None:
        sections = self._get_sections()
        df = self._compute_section_data()
        if df.empty or not sections:
            QMessageBox.information(self, "No Data", "No section data to export.")
            return

        # Keep export aligned with the current chart view.
        if self._get_aggregate() == "type":
            df, sections = self._aggregate_to_types(df, sections)

        checked_secs = self._checked_section_names()
        if checked_secs:
            sections = [(n, d) for n, d in sections if n in checked_secs]
            df = df[df["section_name"].isin(checked_secs)]

        if self._get_aggregate() == "section":
            df, sections = self._apply_section_binning(df, sections)

        if self._get_mode() == "group":
            host_groups = self._host._session_groups_for_controls(self._ss_facet_controls)
            if host_groups:
                df = df.copy()
                df["group"] = df["session_label"].map(host_groups)
                checked_grps = self._checked_groups()
                if checked_grps:
                    df = df[df["group"].isin(checked_grps)]

        if df.empty:
            QMessageBox.information(
                self,
                "No Data",
                "No rows remain after applying current section/group filters.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Section Data", "", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            out = df.copy()

            # Human-friendly fields for copy/paste workflows.
            out["section_type"] = out["section_name"].apply(self._section_base_type)
            out["section_order"] = out.get("section_idx", 0).astype(int) + 1

            keep_cols = [
                "session_label",
                "group",
                "behavior",
                "section_name",
                "section_type",
                "section_order",
                "n_bouts",
                "duration_s",
                "pct_time",
            ]
            keep_cols = [c for c in keep_cols if c in out.columns]
            out = out[keep_cols]

            # Stable sort for spreadsheet readability.
            sort_cols = [c for c in ["group", "session_label", "behavior", "section_order"] if c in out.columns]
            if sort_cols:
                out = out.sort_values(sort_cols, kind="stable")

            # Rename to spreadsheet-friendly headers.
            out = out.rename(columns={
                "session_label": "Session",
                "group": "Group",
                "behavior": "Behavior",
                "section_name": "Section",
                "section_type": "Section Type",
                "section_order": "Section Order",
                "n_bouts": "Avg Bout Count",
                "duration_s": "Avg Duration (s)",
                "pct_time": "Avg % Time In Section",
            })

            # Light rounding keeps precision while avoiding noisy floats.
            for col in ["Avg Bout Count", "Avg Duration (s)", "Avg % Time In Section"]:
                if col in out.columns:
                    out[col] = out[col].astype(float).round(4)

            out.to_csv(path, index=False)
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))


# ======================================================================
# Sub-tab 7: Velocity During Behavior
# ======================================================================

# Number of normalised time-points used for the Profile chart.
_N_NORM_POINTS: int = 50


class _SocialInteractionWidget(QWidget):
    """Social-interaction analytics sub-tab (multi-animal projects).

    Two views over the per-frame ``social_*`` features:

    * **Summary** — per (subject, session) dyadic metrics: mean inter-animal
      distance, time in contact, contact bouts, net approach, advance/yield
      balance, and orientation.
    * **Dominance (HMM)** — a Gaussian HMM fit over continuous social + movement
      features (pooled across the cohort so states are comparable), from which a
      spatial-displacement dominance score is derived per subject: the animal
      that advances into the other's space while the other yields ranks as more
      dominant.  Subjects are ranked within each session.

    All computation is on-demand via the buttons; nothing runs until the user
    asks, so opening the tab is cheap.
    """

    _SUMMARY_COLS = [
        ("Subject", "animal_id"),
        ("Session", "session_id"),
        ("Group", "group"),
        ("Frames", "n_frames"),
        ("Mean dist (norm)", "mean_distance_norm"),
        ("Contact %", "contact_fraction"),
        ("Contact bouts", "n_contact_bouts"),
        ("Mean bout (s)", "mean_contact_bout_s"),
        ("Mean approach", "mean_approach_velocity"),
        ("Advance %", "advance_fraction"),
        ("Heading align", "mean_heading_alignment"),
    ]

    _DOM_COLS = [
        ("Session", "session_id"),
        ("Subject", "animal_id"),
        ("Group", "group"),
        ("Rank", "dominance_rank"),
        ("Dominance score", "dominance_score"),
        ("Mean advance", "mean_advance"),
        ("Yield %", "yield_fraction"),
        ("Interaction (s)", "interaction_time_s"),
        ("Dominant?", "is_dominant"),
    ]

    def __init__(self, host: "BehaviorAnalyticsTab") -> None:
        super().__init__()
        self._host = host
        self._frames_cache = None  # cached per-frame social DataFrame

        from abel.services.social_analysis_service import SocialAnalysisService
        self._svc = SocialAnalysisService()

        # ── Controls ──────────────────────────────────────────────────────
        self._status = QLabel("Open a multi-animal project, then click Compute.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#90a4ae;font-size:11px;")

        self._compute_btn = QPushButton("Compute Social Metrics")
        self._compute_btn.setToolTip(
            "Load the per-frame social features and summarize the dyadic "
            "relationship for every subject / session."
        )
        self._compute_btn.clicked.connect(self._compute_summary)

        self._n_states_spin = QSpinBox()
        self._n_states_spin.setRange(2, 8)
        self._n_states_spin.setValue(4)
        self._n_states_spin.setToolTip("Number of latent interaction states for the dominance HMM.")

        self._hmm_btn = QPushButton("Run Dominance HMM")
        self._hmm_btn.setToolTip(
            "Fit a Gaussian HMM over social + movement features (pooled across "
            "the cohort) and rank subjects by spatial-displacement dominance."
        )
        self._hmm_btn.clicked.connect(self._run_dominance_hmm)

        ctrl = QHBoxLayout()
        ctrl.addWidget(self._compute_btn)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("States:"))
        ctrl.addWidget(self._n_states_spin)
        ctrl.addWidget(self._hmm_btn)
        ctrl.addStretch(1)

        # ── Summary table ─────────────────────────────────────────────────
        self._summary_table = self._make_table([c[0] for c in self._SUMMARY_COLS])

        # ── Dominance table + state-profile text ──────────────────────────
        self._dom_table = self._make_table([c[0] for c in self._DOM_COLS])
        self._profile_text = QTextEdit()
        self._profile_text.setReadOnly(True)
        self._profile_text.setPlaceholderText(
            "Run the dominance HMM to see latent-state feature profiles, which "
            "states count as interaction, and the dominance ranking rationale."
        )

        dom_split = QSplitter(Qt.Orientation.Horizontal)
        dom_split.addWidget(self._dom_table)
        dom_split.addWidget(self._profile_text)
        dom_split.setStretchFactor(0, 3)
        dom_split.setStretchFactor(1, 2)

        inner = QTabWidget()
        inner.addTab(self._summary_table, "Summary")
        _dom_holder = QWidget()
        _dom_v = QVBoxLayout(_dom_holder)
        _dom_v.setContentsMargins(0, 0, 0, 0)
        _dom_v.addWidget(dom_split, 1)
        inner.addTab(_dom_holder, "Dominance (HMM)")

        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.addLayout(ctrl)
        root.addWidget(self._status)
        root.addWidget(inner, 1)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_table(headers: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    def _group_map(self) -> dict[str, str]:
        # Best-effort session_id → group label; missing keys fall back to "".
        try:
            return dict(getattr(self._host, "_session_groups", {}) or {})
        except Exception:
            return {}

    def on_project_reloaded(self) -> None:
        """Clear cached results when the host switches projects."""
        self._frames_cache = None
        self._summary_table.setRowCount(0)
        self._dom_table.setRowCount(0)
        self._profile_text.clear()
        self._status.setText("Open a multi-animal project, then click Compute.")

    def _load_frames(self):
        if self._frames_cache is not None:
            return self._frames_cache
        root = getattr(self._host, "_project_root", None)
        if root is None:
            return None
        df = self._svc.load_social_frames(root)
        self._frames_cache = df
        return df

    @staticmethod
    def _fmt(value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else ""
        if isinstance(value, float):
            if not np.isfinite(value):
                return "—"
            return f"{value:.3f}"
        return str(value)

    def _fill_table(self, table: QTableWidget, cols, rows: list[dict]) -> None:
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, (_label, key) in enumerate(cols):
                val = row.get(key, "")
                # Percent-style columns stored as fractions.
                if key in ("contact_fraction", "advance_fraction", "yield_fraction") and isinstance(val, (int, float)) and np.isfinite(val):
                    text = f"{val * 100:.1f}%"
                else:
                    text = self._fmt(val)
                item = QTableWidgetItem(text)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    item.setData(Qt.ItemDataRole.UserRole, float(val) if np.isfinite(val) else float("nan"))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    # ── Actions ─────────────────────────────────────────────────────────────

    def _compute_summary(self) -> None:
        df = self._load_frames()
        if df is None:
            self._status.setText(
                "No social features found. This tab needs a multi-animal project "
                "with 'Interaction features' extracted in the Pose Features tab."
            )
            return
        fps = self._host._project_fps()
        rows = self._svc.compute_social_summary(df, fps, self._group_map())
        self._fill_table(self._summary_table, self._SUMMARY_COLS, rows)
        n_subj = len({(r["animal_id"], r["session_id"]) for r in rows})
        self._status.setText(
            f"Summarized {n_subj} subject/session record(s) from "
            f"{len(df):,} frames. Run the dominance HMM for latent-state ranking."
        )

    def _run_dominance_hmm(self) -> None:
        df = self._load_frames()
        if df is None:
            self._status.setText(
                "No social features found. Extract 'Interaction features' first."
            )
            return
        fps = self._host._project_fps()
        self._hmm_btn.setEnabled(False)
        self._hmm_btn.setText("Fitting…")
        self._status.setText("Fitting dominance HMM… (this may take a moment)")
        QGuiApplication.processEvents()
        try:
            res = self._svc.fit_dominance_hmm(
                df, fps=fps, n_states=int(self._n_states_spin.value()),
                group_map=self._group_map(),
            )
        finally:
            self._hmm_btn.setEnabled(True)
            self._hmm_btn.setText("Run Dominance HMM")

        if res.get("error"):
            self._status.setText(res["error"])
            return

        self._fill_table(self._dom_table, self._DOM_COLS, res.get("dominance", []))
        self._profile_text.setPlainText(self._format_profiles(res))
        n_dom = sum(1 for r in res.get("dominance", []) if r.get("is_dominant"))
        self._status.setText(
            f"Fit {res['n_states']}-state HMM over {len(res['feature_cols'])} features; "
            f"identified dominant subject in {n_dom} session(s). "
            f"log-likelihood {res.get('log_likelihood', float('nan')):.0f}."
        )

    @staticmethod
    def _format_profiles(res: dict) -> str:
        lines: list[str] = []
        inter = set(res.get("interaction_states", []))
        lines.append("Latent-state feature profiles (raw means):")
        lines.append(f"Interaction states (close proximity): {sorted(inter) or '—'}")
        lines.append("")
        feats = res.get("feature_cols", [])
        for state, prof in sorted(res.get("state_profiles", {}).items()):
            tag = "  [interaction]" if state in inter else ""
            lines.append(f"State {state}{tag}:")
            for f in feats:
                v = prof.get(f, float("nan"))
                vs = f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else "—"
                lines.append(f"    {f}: {vs}")
            lines.append("")
        lines.append(
            "Dominance = mean radial velocity toward the other during interaction "
            "states minus the fraction of those frames spent yielding. Higher = the "
            "subject advances into contested space while the other gives ground."
        )
        return "\n".join(lines)


class _VelocityWidget(QWidget):
    """Velocity-during-behavior analytics sub-tab.

    Five chart types are available via toggle buttons:

    Summary      — Bar chart of mean/peak velocity per session or group.
    Profile      — Time-normalised velocity profile (0–100 % of bout
                   duration), averaged across bouts ± SEM/SD/CI.
                   Reveals whether animals accelerate or decelerate.
    Bout Sequence— Velocity vs sequential bout index within each session.
                   First-N bouts are highlighted in blue, last-N in red.
    Distribution — Box plot of per-bout mean/peak velocities.
    First vs Last— Paired bar chart: first-N vs last-N bouts per
                   session/group, for detecting habituation or
                   sensitisation.
    """

    _BTN_STYLE = (
        "QPushButton{padding:3px 9px;border:1px solid #37474f;border-radius:3px;"
        "background:#1a2027;color:#cfd8dc;}"
        "QPushButton:checked{background:#1565c0;border-color:#1565c0;color:#fff;}"
        "QPushButton:hover:!checked{background:#263238;}"
    )

    def __init__(self, host: "BehaviorAnalyticsTab") -> None:
        super().__init__()
        self._host = host
        self._updating = False
        # Per-behavior cache of velocity records, keyed by (bid, smooth_window).
        self._vel_cache: dict[tuple[str, int], list[dict]] = {}

        def _toggle_row(
            options: list[tuple[str, str]], default_idx: int = 0
        ) -> tuple[QHBoxLayout, QButtonGroup, dict[str, QPushButton]]:
            grp = QButtonGroup(self)
            grp.setExclusive(True)
            row = QHBoxLayout()
            row.setSpacing(3)
            btns: dict[str, QPushButton] = {}
            for i, (label, key) in enumerate(options):
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.setChecked(i == default_idx)
                btn.setStyleSheet(self._BTN_STYLE)
                btn.setSizePolicy(
                    QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
                )
                btn.setMaximumHeight(26)
                grp.addButton(btn, i)
                row.addWidget(btn)
                btns[key] = btn
            row.addStretch(1)
            return row, grp, btns

        # ── Row 1: chart type (two rows so labels fit narrow panel) ──
        self._chart_grp = QButtonGroup(self)
        self._chart_grp.setExclusive(True)
        self._chart_btns: dict[str, QPushButton] = {}
        _all_chart_opts = [
            ("Summary",       "summary"),
            ("Profile",       "profile"),
            ("Distrib.",      "distribution"),
            ("Bout Seq.",     "sequence"),
            ("First vs Last", "firstlast"),
        ]
        chart_row1 = QHBoxLayout()
        chart_row1.setSpacing(3)
        chart_row2 = QHBoxLayout()
        chart_row2.setSpacing(3)
        for _ci, (_clbl, _ckey) in enumerate(_all_chart_opts):
            _cbtn = QPushButton(_clbl)
            _cbtn.setCheckable(True)
            _cbtn.setChecked(_ckey == "summary")
            _cbtn.setStyleSheet(self._BTN_STYLE)
            _cbtn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            _cbtn.setMaximumHeight(26)
            self._chart_grp.addButton(_cbtn)
            self._chart_btns[_ckey] = _cbtn
            if _ci < 3:
                chart_row1.addWidget(_cbtn)
            else:
                chart_row2.addWidget(_cbtn)
        chart_row1.addStretch(1)
        chart_row2.addStretch(1)
        self._chart_grp.idClicked.connect(lambda _: self._update())
        self._chart_grp.idClicked.connect(lambda _: self._on_chart_type_toggled())

        # ── Profile normalisation toggle (shown only for Profile chart) ─
        norm_opts = [
            ("Time-Normalised", "normalized"),
            ("Absolute Time", "absolute"),
        ]
        norm_row, self._norm_grp, self._norm_btns = _toggle_row(norm_opts)
        self._norm_grp.idClicked.connect(lambda _: self._update())

        self._profile_norm_widget = QWidget()
        _pnw_vbox = QVBoxLayout(self._profile_norm_widget)
        _pnw_vbox.setSpacing(2)
        _pnw_vbox.setContentsMargins(0, 0, 0, 0)
        _pnw_lbl = QLabel("Profile:")
        _pnw_lbl.setMinimumWidth(64)
        _pnw_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        _pnw_row = QHBoxLayout()
        _pnw_row.addWidget(_pnw_lbl)
        _pnw_row.addLayout(norm_row)
        _pnw_vbox.addLayout(_pnw_row)
        self._profile_norm_widget.setVisible(False)

        # ── Pre/post context toggle (shown only for Profile chart) ─────
        self._context_chk = QCheckBox("Pre/post context")
        self._context_chk.setStyleSheet("color:#cfd8dc;font-size:10px;")
        self._context_chk.setToolTip(
            "Show velocity before and after each bout, with the bout region shaded.\n"
            "The x-axis is normalized: 0% = bout start, 100% = bout end."
        )
        self._context_chk.toggled.connect(lambda _: self._update())
        self._context_s_spin = QDoubleSpinBox()
        self._context_s_spin.setRange(0.1, 30.0)
        self._context_s_spin.setValue(2.0)
        self._context_s_spin.setSuffix(" s")
        self._context_s_spin.setSingleStep(0.5)
        self._context_s_spin.setDecimals(1)
        self._context_s_spin.setMaximumWidth(72)
        self._context_s_spin.setToolTip("Duration of context window shown before and after each bout.")
        self._context_s_spin.editingFinished.connect(self._update)
        _ctx_inner = QHBoxLayout()
        _ctx_inner.setSpacing(4)
        _ctx_inner.setContentsMargins(0, 0, 0, 0)
        _ctx_inner.addWidget(self._context_chk)
        _ctx_inner.addWidget(self._context_s_spin)
        _ctx_inner.addStretch(1)
        self._context_widget = QWidget()
        _ctx_vbox = QVBoxLayout(self._context_widget)
        _ctx_vbox.setSpacing(2)
        _ctx_vbox.setContentsMargins(0, 0, 0, 0)
        _ctx_lbl = QLabel("Context:")
        _ctx_lbl.setMinimumWidth(64)
        _ctx_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        _ctx_outer_row = QHBoxLayout()
        _ctx_outer_row.addWidget(_ctx_lbl)
        _ctx_outer_row.addLayout(_ctx_inner)
        _ctx_vbox.addLayout(_ctx_outer_row)
        self._context_widget.setVisible(False)

        # ── Axis limits widget (all chart types) ──────────────────────
        _AL_SPIN_W = 70
        _AL_DEC = 3
        _AL_RANGE = (-1e5, 1e5)
        self._xlim_chk = QCheckBox("X lim:")
        self._xlim_chk.setStyleSheet("color:#cfd8dc;font-size:10px;")
        self._xlim_chk.setToolTip("Fix the x-axis range.")
        self._xlim_min_spin = QDoubleSpinBox()
        self._xlim_min_spin.setRange(*_AL_RANGE)
        self._xlim_min_spin.setDecimals(_AL_DEC)
        self._xlim_min_spin.setValue(0.0)
        self._xlim_min_spin.setMaximumWidth(_AL_SPIN_W)
        self._xlim_min_spin.setEnabled(False)
        self._xlim_max_spin = QDoubleSpinBox()
        self._xlim_max_spin.setRange(*_AL_RANGE)
        self._xlim_max_spin.setDecimals(_AL_DEC)
        self._xlim_max_spin.setValue(100.0)
        self._xlim_max_spin.setMaximumWidth(_AL_SPIN_W)
        self._xlim_max_spin.setEnabled(False)
        self._ylim_chk = QCheckBox("Y lim:")
        self._ylim_chk.setStyleSheet("color:#cfd8dc;font-size:10px;")
        self._ylim_chk.setToolTip("Fix the y-axis range.")
        self._ylim_min_spin = QDoubleSpinBox()
        self._ylim_min_spin.setRange(*_AL_RANGE)
        self._ylim_min_spin.setDecimals(_AL_DEC)
        self._ylim_min_spin.setValue(0.0)
        self._ylim_min_spin.setMaximumWidth(_AL_SPIN_W)
        self._ylim_min_spin.setEnabled(False)
        self._ylim_max_spin = QDoubleSpinBox()
        self._ylim_max_spin.setRange(*_AL_RANGE)
        self._ylim_max_spin.setDecimals(_AL_DEC)
        self._ylim_max_spin.setValue(50.0)
        self._ylim_max_spin.setMaximumWidth(_AL_SPIN_W)
        self._ylim_max_spin.setEnabled(False)
        _dash_x = QLabel("–")
        _dash_x.setStyleSheet("color:#90a4ae;")
        _dash_y = QLabel("–")
        _dash_y.setStyleSheet("color:#90a4ae;")
        def _toggle_lim_spins(chk, mn, mx):
            chk.toggled.connect(lambda v: (mn.setEnabled(v), mx.setEnabled(v), self._update()))
            for sp in (mn, mx):
                sp.editingFinished.connect(self._update)
        _toggle_lim_spins(self._xlim_chk, self._xlim_min_spin, self._xlim_max_spin)
        _toggle_lim_spins(self._ylim_chk, self._ylim_min_spin, self._ylim_max_spin)
        _xl_row = QHBoxLayout()
        _xl_row.setSpacing(3)
        _xl_row.setContentsMargins(0, 0, 0, 0)
        _xl_row.addWidget(self._xlim_chk)
        _xl_row.addWidget(self._xlim_min_spin)
        _xl_row.addWidget(_dash_x)
        _xl_row.addWidget(self._xlim_max_spin)
        _xl_row.addStretch(1)
        _yl_row = QHBoxLayout()
        _yl_row.setSpacing(3)
        _yl_row.setContentsMargins(0, 0, 0, 0)
        _yl_row.addWidget(self._ylim_chk)
        _yl_row.addWidget(self._ylim_min_spin)
        _yl_row.addWidget(_dash_y)
        _yl_row.addWidget(self._ylim_max_spin)
        _yl_row.addStretch(1)
        self._axlim_widget = QWidget()
        _alw_vbox = QVBoxLayout(self._axlim_widget)
        _alw_vbox.setSpacing(2)
        _alw_vbox.setContentsMargins(0, 0, 0, 0)
        _alw_hdr = QLabel("Axis limits:")
        _alw_hdr.setStyleSheet("color:#90a4ae;font-size:10px;")
        _alw_vbox.addWidget(_alw_hdr)
        _alw_vbox.addLayout(_xl_row)
        _alw_vbox.addLayout(_yl_row)

        bout_avg_opts = [
            ("All Bouts", "all"),
            ("Per Session", "per_session"),
        ]
        bout_avg_row, self._bout_avg_grp, self._bout_avg_btns = _toggle_row(bout_avg_opts)
        self._bout_avg_grp.idClicked.connect(lambda _: self._update())

        self._bout_avg_widget = QWidget()
        _baw_vbox = QVBoxLayout(self._bout_avg_widget)
        _baw_vbox.setSpacing(2)
        _baw_vbox.setContentsMargins(0, 0, 0, 0)
        _baw_lbl = QLabel("Bouts:")
        _baw_lbl.setMinimumWidth(64)
        _baw_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        _baw_row = QHBoxLayout()
        _baw_row.addWidget(_baw_lbl)
        _baw_row.addLayout(bout_avg_row)
        _baw_vbox.addLayout(_baw_row)
        self._bout_avg_widget.setVisible(True)  # visible for default "summary" chart

        # ── Row 2: metric ─────────────────────────────────────────────
        metric_opts = [
            ("Mean Velocity", "mean"),
            ("Peak Velocity", "peak"),
        ]
        metric_row, self._metric_grp, self._metric_btns = _toggle_row(metric_opts)
        self._metric_grp.idClicked.connect(lambda _: self._update())

        # ── Row 3: view mode ──────────────────────────────────────────
        mode_opts = [
            ("Individual Sessions", "individual"),
            ("By Group", "group"),
        ]
        mode_row, self._mode_grp, self._mode_btns = _toggle_row(mode_opts)
        self._mode_grp.idClicked.connect(lambda _: self._update())
        self._mode_grp.idClicked.connect(lambda _: self._on_vel_mode_toggled())

        # ── Faceted grouping controls (visible in group mode only) ─────
        self._vel_facet_controls: dict[str, str] = {}
        self._vel_facet = _FacetControls("Group by:")
        self._vel_facet.setToolTip(
            "For each factor choose — combine —, — split —, or a specific level.\n"
            "Split two or more factors to plot their interaction."
        )
        self._vel_facet.changed.connect(self._on_vel_facets_changed)

        self._vel_group_controls = QWidget()
        _vgc_vbox = QVBoxLayout(self._vel_group_controls)
        _vgc_vbox.setSpacing(3)
        _vgc_vbox.setContentsMargins(0, 0, 0, 0)
        _vgc_vbox.addWidget(self._vel_facet)
        self._vel_group_controls.setVisible(False)

        # ── Behavior selector ─────────────────────────────────────────
        self._behavior_combo = QComboBox()
        self._behavior_combo.setToolTip(
            "Select the behavior whose bouts will be analysed for velocity."
        )
        self._behavior_combo.currentIndexChanged.connect(lambda _: self._update())

        # ── Smoothing kernel ──────────────────────────────────────────
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(1, 31)
        self._smooth_spin.setValue(5)
        self._smooth_spin.setSingleStep(2)
        self._smooth_spin.setToolTip(
            "Moving-average kernel size for per-frame velocity (frames).\n"
            "Odd values are recommended.  Larger → smoother but blurs\n"
            "fast transients.  Value of 1 = no smoothing."
        )
        self._smooth_spin.editingFinished.connect(self._on_smooth_changed)

        # ── N-bouts spinner (First / Last) ────────────────────────────
        self._n_bouts_spin = QSpinBox()
        self._n_bouts_spin.setRange(1, 10000)
        self._n_bouts_spin.setValue(3)
        self._n_bouts_spin.setToolTip(
            "N used in 'First vs Last' and 'Bout Sequence' charts.\n"
            "Sets how many bouts are highlighted at the start and end."
        )
        self._n_bouts_spin.editingFinished.connect(lambda: self._update())

        # ── Outlier detection ─────────────────────────────────────────
        self._outlier_chk = QCheckBox("Outlier Detection")
        self._outlier_chk.setStyleSheet("color:#cfd8dc;font-size:10px;font-weight:bold;")
        self._outlier_chk.setToolTip(
            "Detect and handle bouts whose velocity is an outlier relative "
            "to all bouts across all sessions.\n"
            "Exclude: removes outlier bouts from the chart and export.\n"
            "Winsorize: clips outlier values to the nearest fence value."
        )

        self._outlier_method_combo = QComboBox()
        self._outlier_method_combo.addItem("IQR (Tukey)", "iqr")
        self._outlier_method_combo.addItem("Z-score", "zscore")
        self._outlier_method_combo.setToolTip(
            "IQR: flags bouts outside Q1 - k*IQR ... Q3 + k*IQR.\n"
            "Z-score: flags bouts with |z| > threshold."
        )
        self._outlier_method_combo.setMaximumWidth(120)

        self._outlier_thresh_spin = QDoubleSpinBox()
        self._outlier_thresh_spin.setRange(0.5, 10.0)
        self._outlier_thresh_spin.setValue(1.5)
        self._outlier_thresh_spin.setSingleStep(0.5)
        self._outlier_thresh_spin.setDecimals(1)
        self._outlier_thresh_spin.setMaximumWidth(64)
        self._outlier_thresh_spin.setToolTip(
            "Threshold multiplier.\n"
            "IQR: 1.5 = mild outliers, 3.0 = extreme outliers.\n"
            "Z-score: typical threshold is 2.5-3.0."
        )

        self._outlier_action_combo = QComboBox()
        self._outlier_action_combo.addItem("Exclude", "exclude")
        self._outlier_action_combo.addItem("Winsorize", "winsorize")
        self._outlier_action_combo.setToolTip(
            "Exclude: outlier bouts are removed entirely.\n"
            "Winsorize: outlier values are clamped to the fence value."
        )
        self._outlier_action_combo.setMaximumWidth(100)

        def _on_outlier_method_changed(idx: int) -> None:
            method = self._outlier_method_combo.currentData()
            if method == "iqr":
                self._outlier_thresh_spin.setValue(1.5)
                self._outlier_thresh_spin.setRange(0.5, 10.0)
            else:
                self._outlier_thresh_spin.setValue(2.5)
                self._outlier_thresh_spin.setRange(0.5, 10.0)

        self._outlier_method_combo.currentIndexChanged.connect(_on_outlier_method_changed)

        for _w in (self._outlier_method_combo, self._outlier_thresh_spin, self._outlier_action_combo):
            if hasattr(_w, "currentIndexChanged"):
                _w.currentIndexChanged.connect(lambda _: self._update())
            elif hasattr(_w, "editingFinished"):
                _w.editingFinished.connect(self._update)

        _od_row1 = QHBoxLayout()
        _od_row1.setSpacing(4)
        _od_row1.setContentsMargins(0, 0, 0, 0)
        _od_row1.addWidget(self._outlier_chk)
        _od_row1.addStretch(1)

        _od_row2 = QHBoxLayout()
        _od_row2.setSpacing(4)
        _od_row2.setContentsMargins(16, 0, 0, 0)
        _od_meth_lbl = QLabel("Method:")
        _od_meth_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        _od_row2.addWidget(_od_meth_lbl)
        _od_row2.addWidget(self._outlier_method_combo)
        _od_thr_lbl = QLabel("k:")
        _od_thr_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        _od_row2.addWidget(_od_thr_lbl)
        _od_row2.addWidget(self._outlier_thresh_spin)
        _od_row2.addStretch(1)

        _od_row3 = QHBoxLayout()
        _od_row3.setSpacing(4)
        _od_row3.setContentsMargins(16, 0, 0, 0)
        _od_act_lbl = QLabel("Action:")
        _od_act_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        _od_row3.addWidget(_od_act_lbl)
        _od_row3.addWidget(self._outlier_action_combo)
        _od_row3.addStretch(1)

        self._outlier_controls_widget = QWidget()
        _oc_vbox = QVBoxLayout(self._outlier_controls_widget)
        _oc_vbox.setSpacing(2)
        _oc_vbox.setContentsMargins(0, 0, 0, 0)
        _oc_vbox.addLayout(_od_row2)
        _oc_vbox.addLayout(_od_row3)
        self._outlier_controls_widget.setVisible(False)

        self._outlier_chk.toggled.connect(self._outlier_controls_widget.setVisible)
        self._outlier_chk.toggled.connect(lambda _: self._update())

        _outlier_section = QVBoxLayout()
        _outlier_section.setSpacing(2)
        _outlier_section.setContentsMargins(0, 0, 0, 0)
        _outlier_section.addLayout(_od_row1)
        _outlier_section.addWidget(self._outlier_controls_widget)

        # ── Session filter ────────────────────────────────────────────
        self._session_list = QListWidget()
        self._session_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        # Grow into spare vertical space so many sessions are visible at once.
        self._session_list.setMinimumHeight(72)
        self._session_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._session_list.setStyleSheet(
            "QListWidget{background:#0A1929;border:1px solid #1E3A5F;"
            "border-radius:3px;color:#cfd8dc;font-size:10px;}"
        )
        self._sess_all_btn = QPushButton("All")
        self._sess_all_btn.setMaximumWidth(40)
        self._sess_all_btn.setStyleSheet(self._BTN_STYLE)
        self._sess_all_btn.clicked.connect(self._check_all_sessions)
        self._sess_none_btn = QPushButton("None")
        self._sess_none_btn.setMaximumWidth(48)
        self._sess_none_btn.setStyleSheet(self._BTN_STYLE)
        self._sess_none_btn.clicked.connect(self._uncheck_all_sessions)

        sess_btns_vbox = QVBoxLayout()
        sess_btns_vbox.addWidget(self._sess_all_btn)
        sess_btns_vbox.addWidget(self._sess_none_btn)
        sess_btns_vbox.addStretch(1)

        sess_lbl = QLabel("Sessions:")
        sess_lbl.setMinimumWidth(64)
        sess_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        sess_row_layout = QHBoxLayout()
        sess_row_layout.setSpacing(4)
        sess_row_layout.addWidget(sess_lbl)
        sess_row_layout.addWidget(self._session_list, 1)
        sess_row_layout.addLayout(sess_btns_vbox)

        # ── Action row ────────────────────────────────────────────────
        self._graph_settings_btn = QPushButton("⚙ Settings")
        self._graph_settings_btn.setToolTip("Edit graph appearance: fonts, error bars, DPI, figure size.")
        self._graph_settings_btn.clicked.connect(self._open_graph_settings)

        self._export_btn = QPushButton("Export\u2026")
        self._export_btn.setToolTip("Save the current chart as PNG / SVG / PDF.")
        self._export_btn.clicked.connect(self._export_figure)

        self._export_data_btn = QPushButton("Export Data\u2026")
        self._export_data_btn.setToolTip(
            "Export per-bout velocity records (mean, peak, duration) as CSV."
        )
        self._export_data_btn.clicked.connect(self._export_data)

        self._export_profile_btn = QPushButton("Export Profile Data\u2026")
        self._export_profile_btn.setToolTip(
            "Export the mean \u00b1 error band data used to draw the velocity "
            "profile graph.  Opens a copy-paste friendly dialog with CSV "
            "preview, clipboard copy, and file save options."
        )
        self._export_profile_btn.clicked.connect(self._export_profile_data)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._update)
        self._apply_btn.setStyleSheet(
            "QPushButton{padding:3px 14px;background:#1565c0;border:none;"
            "border-radius:3px;color:#fff;font-weight:bold;}"
            "QPushButton:hover{background:#1976d2;}"
        )

        self._scale_spin = QSpinBox()
        self._scale_spin.setRange(25, 300)
        self._scale_spin.setValue(int(self._host._graph_settings.get("scale", 100)))
        self._scale_spin.setSuffix("%")
        self._scale_spin.setSingleStep(10)
        self._scale_spin.setToolTip("Scale the figure size as a percentage of the base max_w × max_h.")
        self._scale_spin.setMaximumWidth(72)
        self._scale_spin.editingFinished.connect(self._on_scale_changed)

        action_row1 = QHBoxLayout()
        action_row1.setSpacing(6)
        _sm_lbl = QLabel("Smooth:")
        _sm_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        action_row1.addWidget(_sm_lbl)
        action_row1.addWidget(self._smooth_spin)
        _nb_lbl = QLabel("N bouts:")
        _nb_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        action_row1.addWidget(_nb_lbl)
        action_row1.addWidget(self._n_bouts_spin)
        action_row1.addStretch(1)

        action_row2 = QHBoxLayout()
        action_row2.setSpacing(6)
        _sc_lbl = QLabel("Scale:")
        _sc_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
        action_row2.addWidget(_sc_lbl)
        action_row2.addWidget(self._scale_spin)
        action_row2.addWidget(self._graph_settings_btn)
        action_row2.addStretch(1)

        action_row3 = QHBoxLayout()
        action_row3.setSpacing(6)
        action_row3.addWidget(self._export_btn)
        action_row3.addWidget(self._export_data_btn)
        action_row3.addWidget(self._export_profile_btn)
        action_row3.addWidget(self._apply_btn)
        action_row3.addStretch(1)

        beh_inner = QHBoxLayout()
        beh_inner.addWidget(self._behavior_combo, 1)
        beh_inner.addStretch()

        def _labeled_row(label: str, inner: QHBoxLayout) -> QHBoxLayout:
            outer = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(64)
            lbl.setStyleSheet("color:#90a4ae;font-size:10px;")
            outer.addWidget(lbl)
            outer.addLayout(inner)
            return outer

        controls_widget = QWidget()
        ctrl_vbox = QVBoxLayout(controls_widget)
        ctrl_vbox.setSpacing(2)
        ctrl_vbox.setContentsMargins(4, 4, 4, 4)
        _chart_hdr = QLabel("Chart:")
        _chart_hdr.setStyleSheet("color:#90a4ae;font-size:10px;")
        ctrl_vbox.addWidget(_chart_hdr)
        ctrl_vbox.addLayout(chart_row1)
        ctrl_vbox.addLayout(chart_row2)
        ctrl_vbox.addWidget(self._profile_norm_widget)
        ctrl_vbox.addWidget(self._context_widget)
        ctrl_vbox.addWidget(self._axlim_widget)
        ctrl_vbox.addWidget(self._bout_avg_widget)
        ctrl_vbox.addLayout(_labeled_row("Metric:", metric_row))
        ctrl_vbox.addLayout(_labeled_row("View:", mode_row))
        ctrl_vbox.addWidget(self._vel_group_controls)
        ctrl_vbox.addLayout(_labeled_row("Behavior:", beh_inner))
        # Stretch factor 1 lets the Sessions list expand into the spare vertical
        # space so many sessions are visible without scrolling.
        ctrl_vbox.addLayout(sess_row_layout, 1)
        ctrl_vbox.addLayout(_outlier_section)
        ctrl_vbox.addLayout(action_row1)
        ctrl_vbox.addLayout(action_row2)
        ctrl_vbox.addLayout(action_row3)

        # ── Status label ──────────────────────────────────────────────
        self._status_lbl = QLabel(
            "Refresh analytics to load velocity data."
        )
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color:#90a4ae;font-size:10px;")

        # ── Matplotlib canvas ─────────────────────────────────────────
        self._figure: Any = None
        self._canvas: Any = None
        self._toolbar: Any = None
        self._placeholder = QLabel(
            "Velocity data will appear here after loading analytics.\n"
            "Pose files must be loaded and temporal refinement run."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setMinimumHeight(260)
        self._placeholder.setStyleSheet(
            "border:1px solid #1A2027;background:#0A1929;"
            "border-radius:4px;color:#546E7A;"
        )
        self._canvas_scroll: Any = None
        if (
            _ensure_matplotlib()
            and Figure is not None
            and FigureCanvas is not None
            and NavigationToolbar is not None
        ):
            _dpi = int(self._host._graph_settings.get("dpi", 150))
            _pw = int(self._host._graph_settings.get("max_w", 700))
            _ph = int(self._host._graph_settings.get("max_h", 420))
            self._figure = Figure(figsize=(_pw / _dpi, _ph / _dpi), dpi=_dpi)
            self._canvas = FigureCanvas(self._figure)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
            )
            self._canvas.setFixedSize(_pw, _ph)
            self._toolbar = NavigationToolbar(self._canvas, self)
            self._placeholder.setVisible(False)
            from PySide6.QtWidgets import QScrollArea as _SA
            self._canvas_scroll = _SA()
            self._canvas_scroll.setWidget(self._canvas)
            self._canvas_scroll.setWidgetResizable(False)
            self._canvas_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._canvas_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            # Dynamic resize: canvas grows to fill the available viewport width
            self._vel_resize_filter = _ViewportResizeFilter(
                self._sync_canvas_to_viewport, self
            )
            self._canvas_scroll.viewport().installEventFilter(self._vel_resize_filter)

        # ── Splitter layout: controls left, canvas right ──────────────
        from PySide6.QtWidgets import QScrollArea as _QScrollAreaV
        left_widget = _QScrollAreaV()
        left_widget.setWidget(controls_widget)
        left_widget.setWidgetResizable(True)
        left_widget.setMinimumWidth(260)
        # No maximum width: keep the splitter draggable in both directions.
        left_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        right_widget = QWidget()
        right_vbox = QVBoxLayout(right_widget)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(2)
        if self._toolbar is not None:
            right_vbox.addWidget(self._toolbar)
        if self._canvas_scroll is not None:
            right_vbox.addWidget(self._canvas_scroll, 1)
        right_vbox.addWidget(self._placeholder, 1)
        right_vbox.addWidget(self._status_lbl)

        from PySide6.QtWidgets import QSplitter as _QSplitter
        splitter = _QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1000])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter, 1)

    # ── Public interface ──────────────────────────────────────────────

    def on_data_loaded(self) -> None:
        """Called by the host after each analytics refresh."""
        self._vel_cache.clear()
        self._rebuild_behavior_combo()
        self._rebuild_session_list()
        self._refresh_vel_factor_selector()
        self._refresh_vel_group_filter()
        self._on_chart_type_toggled()
        self._update()

    def on_groups_updated(self) -> None:
        """Called by the host when group/factor definitions change."""
        self._refresh_vel_factor_selector()
        self._refresh_vel_group_filter()

    # ── Control helpers ───────────────────────────────────────────────

    def _on_vel_mode_toggled(self) -> None:
        mode = self._get_mode()
        self._vel_group_controls.setVisible(mode == "group")

    def _on_chart_type_toggled(self) -> None:
        chart = self._get_chart()
        self._profile_norm_widget.setVisible(chart == "profile")
        self._context_widget.setVisible(chart == "profile")
        self._bout_avg_widget.setVisible(chart in ("summary", "distribution", "profile"))

    def _get_bout_mode(self) -> str:
        btn = self._bout_avg_grp.checkedButton()
        if btn is None:
            return "all"
        for key, b in self._bout_avg_btns.items():
            if b is btn:
                return key
        return "all"

    def _collapse_bouts(self, data: list[dict]) -> list[dict]:
        """Average all bouts within a session into a single record per session."""
        by_sess: dict[str, list[dict]] = {}
        for d in data:
            by_sess.setdefault(d["session_label"], []).append(d)
        collapsed: list[dict] = []
        for sess_label, bouts in by_sess.items():
            if not bouts:
                continue
            mean_vel_vals = [b["mean_vel"] for b in bouts if b.get("mean_vel") is not None]
            peak_vel_vals = [b["peak_vel"] for b in bouts if b.get("peak_vel") is not None]
            avg = dict(bouts[0])  # copy first bout as base for session-level metadata
            avg["mean_vel"] = float(np.mean(mean_vel_vals)) if mean_vel_vals else 0.0
            avg["peak_vel"] = float(np.mean(peak_vel_vals)) if peak_vel_vals else 0.0
            avg["session_label"] = sess_label
            collapsed.append(avg)
        return collapsed

    def _refresh_vel_factor_selector(self) -> None:
        """Rebuild the per-factor facet dropdowns when factors change."""
        factors = list(getattr(self._host, "_factor_definitions", []))
        self._vel_facet.blockSignals(True)
        self._vel_facet.rebuild(factors, self._host._levels_by_factor())
        if not self._vel_facet_controls and factors:
            self._vel_facet_controls = self._host._default_facet_controls()
        self._vel_facet.set_state(self._vel_facet_controls)
        self._vel_facet.blockSignals(False)

    def _refresh_vel_group_filter(self) -> None:
        """No-op retained for callers: facets filter via the dropdowns."""
        return

    def _vel_checked_groups(self) -> set[str]:
        """All series the facet controls currently produce."""
        return {
            g for g in self._host._session_groups_for_controls(
                self._vel_facet_controls
            ).values()
            if g
        }

    def _on_vel_facets_changed(self) -> None:
        self._vel_facet_controls = self._vel_facet.state()
        self._update()

    def _get_norm_mode(self) -> str:
        btn = self._norm_grp.checkedButton()
        if btn is None:
            return "normalized"
        for key, b in self._norm_btns.items():
            if b is btn:
                return key
        return "normalized"

    def _rebuild_behavior_combo(self) -> None:
        prev = self._behavior_combo.currentData()
        self._behavior_combo.blockSignals(True)
        self._behavior_combo.clear()
        for b in self._host._behaviors.behaviors:
            bid = str(b.behavior_id)
            if bid == NO_BEHAVIOR_ID:
                continue
            label = str(b.name or bid)
            self._behavior_combo.addItem(label, userData=bid)
        for i in range(self._behavior_combo.count()):
            if self._behavior_combo.itemData(i) == prev:
                self._behavior_combo.setCurrentIndex(i)
                break
        self._behavior_combo.blockSignals(False)

    def _rebuild_session_list(self) -> None:
        summary_checked = self._host._summary_tab._checked_subjects()
        all_labels = sorted({
            r["session_label"]
            for r in self._host._summary_rows
            if r.get("behavior_id") != DISTANCE_BEHAVIOR_ID
            and (not summary_checked or r["session_label"] in summary_checked)
        })
        prev_checked = self._checked_sessions()
        self._session_list.blockSignals(True)
        self._session_list.clear()
        for lbl in all_labels:
            item = QListWidgetItem()
            cb = QCheckBox(lbl)
            cb.setChecked(not prev_checked or lbl in prev_checked)
            cb.stateChanged.connect(
                lambda _, s=self: QTimer.singleShot(0, s._update)
            )
            item.setSizeHint(cb.sizeHint())
            self._session_list.addItem(item)
            self._session_list.setItemWidget(item, cb)
        self._session_list.blockSignals(False)

    def _checked_sessions(self) -> set[str]:
        out: set[str] = set()
        for i in range(self._session_list.count()):
            item = self._session_list.item(i)
            if item is None:
                continue
            cb = self._session_list.itemWidget(item)
            if isinstance(cb, QCheckBox) and cb.isChecked():
                out.add(cb.text())
        return out

    def _check_all_sessions(self) -> None:
        for i in range(self._session_list.count()):
            cb = self._session_list.itemWidget(self._session_list.item(i))
            if isinstance(cb, QCheckBox):
                cb.setChecked(True)

    def _uncheck_all_sessions(self) -> None:
        for i in range(self._session_list.count()):
            cb = self._session_list.itemWidget(self._session_list.item(i))
            if isinstance(cb, QCheckBox):
                cb.setChecked(False)

    def _on_smooth_changed(self) -> None:
        """Invalidate velocity cache when the smoothing window changes."""
        self._vel_cache.clear()
        self._update()

    def _on_scale_changed(self) -> None:
        """Apply new scale value from the spinbox."""
        self._host._graph_settings["scale"] = self._scale_spin.value()
        self._update()

    def _get_chart(self) -> str:
        for key, btn in self._chart_btns.items():
            if btn.isChecked():
                return key
        return "summary"

    def _get_metric(self) -> str:
        for key, btn in self._metric_btns.items():
            if btn.isChecked():
                return key
        return "mean"

    def _get_mode(self) -> str:
        for key, btn in self._mode_btns.items():
            if btn.isChecked():
                return key
        return "individual"

    def _vel_unit_label(self) -> str:
        """Return 'cm/s' when pixel calibration is available, else 'px/s'."""
        for row in self._host._summary_rows[:10]:
            sid = row.get("session_id", "")
            ppm = self._host._pixels_per_mm_for_session(str(sid))
            if ppm and ppm > 0:
                return "cm/s"
        return "px/s"

    # ── Velocity data ─────────────────────────────────────────────────

    def _get_vel_data(self, behavior_id: str) -> list[dict]:
        """Return cached or freshly computed per-bout velocity records."""
        smooth = self._smooth_spin.value()
        key = (behavior_id, smooth)
        if key not in self._vel_cache:
            self._vel_cache[key] = self._host._collect_velocity_data_for_behavior(
                behavior_id, smooth_window=smooth
            )
        return self._vel_cache[key]

    def _apply_outlier_filter(
        self, data: list[dict], metric_key: str
    ) -> "tuple[list[dict], int]":
        """Apply outlier detection to *data* using the current UI settings.

        Returns ``(processed_data, n_excluded)`` where *processed_data* has
        outliers either excluded or winsorized and *n_excluded* counts how
        many bouts were affected.

        Outlier detection is performed across ALL sessions jointly so that
        a single extremely fast or slow bout in any session is flagged.
        The velocity field used is ``mean_vel`` for "mean" metric and
        ``peak_vel`` for "peak" metric.
        """
        if not self._outlier_chk.isChecked() or not data:
            return data, 0

        vel_field = "mean_vel" if metric_key == "mean" else "peak_vel"
        vals = np.array([d[vel_field] for d in data], dtype=float)

        method = self._outlier_method_combo.currentData() or "iqr"
        thresh = float(self._outlier_thresh_spin.value())

        if method == "iqr":
            q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            iqr = q3 - q1
            lo, hi = q1 - thresh * iqr, q3 + thresh * iqr
        else:
            mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1))
            if std == 0:
                return data, 0
            lo, hi = mean - thresh * std, mean + thresh * std

        action = self._outlier_action_combo.currentData() or "exclude"
        is_outlier = (vals < lo) | (vals > hi)
        n_excluded = int(is_outlier.sum())

        if n_excluded == 0:
            return data, 0

        if action == "exclude":
            result = [d for d, flag in zip(data, is_outlier) if not flag]
        else:
            # Winsorize: clamp to fence values in a copy of the records
            result = []
            for d, flag in zip(data, is_outlier):
                if flag:
                    rec = dict(d)
                    rec[vel_field] = float(np.clip(d[vel_field], lo, hi))
                    # Also clamp the velocity_trace if present
                    tr = rec.get("velocity_trace")
                    if tr is not None:
                        rec["velocity_trace"] = np.clip(np.asarray(tr, dtype=float), lo, hi)
                    rec["_outlier_winsorized"] = True
                    result.append(rec)
                else:
                    result.append(d)

        return result, n_excluded

    def _update(self) -> None:
        """Recompute velocity data and redraw the current chart."""
        if self._figure is None or self._updating:
            return
        self._updating = True
        try:
            bid = self._behavior_combo.currentData()
            if not bid:
                self._show_message("No behavior selected.")
                return
            chart = self._get_chart()
            metric_key = self._get_metric()
            mode = self._get_mode()
            checked_sessions = self._checked_sessions()

            data = self._get_vel_data(bid)
            if not data:
                self._show_message(
                    "No pose data available for this behavior.\n"
                    "Make sure pose files are linked and temporal refinement "
                    "has been run at least once."
                )
                return

            if checked_sessions:
                data = [d for d in data if d["session_label"] in checked_sessions]
            if not data:
                self._show_message("No data for the selected sessions.")
                return

            # Apply outlier detection before drawing
            data, n_outliers = self._apply_outlier_filter(data, metric_key)
            if not data:
                self._show_message("All bouts were flagged as outliers. Reduce the threshold or disable outlier detection.")
                return

            vel_unit = self._vel_unit_label()
            self._figure.clear()
            self._figure.patch.set_facecolor(
                self._host._graph_settings.get("fig_bg", "#ffffff")
            )

            # Optionally collapse bouts to per-session means
            draw_data = data
            bout_mode = self._get_bout_mode()
            if chart in ("summary", "distribution") and bout_mode == "per_session":
                draw_data = self._collapse_bouts(data)

            if chart == "summary":
                self._draw_summary(draw_data, metric_key, mode, vel_unit)
            elif chart == "profile":
                self._draw_profile(data, mode, vel_unit, per_session=bout_mode == "per_session")
            elif chart == "sequence":
                self._draw_sequence(data, metric_key, mode, vel_unit)
            elif chart == "distribution":
                self._draw_distribution(draw_data, metric_key, mode, vel_unit)
            elif chart == "firstlast":
                self._draw_first_last(data, metric_key, mode, vel_unit)

            # Apply manual axis limits (overrides tight_layout auto-scaling)
            if self._figure and self._figure.axes:
                _ax0 = self._figure.axes[0]
                if self._xlim_chk.isChecked():
                    _ax0.set_xlim(self._xlim_min_spin.value(), self._xlim_max_spin.value())
                if self._ylim_chk.isChecked():
                    _ax0.set_ylim(self._ylim_min_spin.value(), self._ylim_max_spin.value())

            # Fill the freshly-drawn figure to the viewport width on first render.
            # Deferred call re-runs after the layout pass (viewport width is not
            # yet realised on the very first render).
            self._sync_canvas_to_viewport()
            QTimer.singleShot(0, self._sync_canvas_to_viewport)
            self._canvas.draw_idle()

            beh_label = self._behavior_combo.currentText()
            n_sess = len({d["session_label"] for d in data})
            bout_note = " (averaged per session)" if (
                chart in ("summary", "distribution", "profile") and bout_mode == "per_session"
            ) else ""
            _outlier_note = ""
            if n_outliers > 0:
                _od_action = self._outlier_action_combo.currentData() or "exclude"
                if _od_action == "exclude":
                    _outlier_note = f"  [{n_outliers} outlier bout(s) excluded]"
                else:
                    _outlier_note = f"  [{n_outliers} outlier bout(s) winsorized]"
            self._status_lbl.setText(
                f"{beh_label}: {len(data)} bout(s) across {n_sess} session(s){bout_note}."
                f"  Units: {vel_unit}{_outlier_note}"
            )
        finally:
            self._updating = False

    def _show_message(self, msg: str) -> None:
        if self._figure is not None:
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5, msg,
                ha="center", va="center",
                transform=ax.transAxes,
                color="#546E7A", fontsize=10, wrap=True,
            )
            ax.set_axis_off()
            self._canvas.draw_idle()
        self._status_lbl.setText(msg)

    def _sync_canvas_to_viewport(self) -> None:
        """Resize the matplotlib canvas to fill the available scroll area viewport.

        Called automatically when the viewport is resized (window resize or
        splitter move).  The figure height stays proportional to the
        max_h / max_w ratio set in graph settings so aspect ratio is preserved.
        """
        if self._canvas_scroll is None or self._canvas is None or self._figure is None:
            return
        gs = self._host._graph_settings
        dpi = int(gs.get("dpi", 150))
        scale = int(gs.get("scale", 100)) / 100.0
        base_w = int(gs.get("max_w", 700))
        base_h = int(gs.get("max_h", 420))
        # Maintain aspect ratio: height scales proportionally with width
        ratio = base_h / max(base_w, 1)
        # Stable fill width: reserve the scrollbar gutter up-front so the canvas
        # doesn't shimmer back and forth when its height sits at the threshold
        # where a vertical scrollbar would appear.
        avail_w = _stable_fill_width(
            self._canvas_scroll, lambda w: w * ratio * scale, min_w=200
        )
        new_h = max(120, int(avail_w * ratio * scale))
        new_w = max(200, int(avail_w * scale))
        try:
            self._figure.set_size_inches(new_w / dpi, new_h / dpi)
            self._canvas.setFixedSize(new_w, new_h)
            self._canvas.updateGeometry()
        except Exception:
            pass

    def showEvent(self, event: Any) -> None:  # type: ignore[override]
        """Re-fill the canvas when the tab becomes visible (the viewport width
        is provisional if data was rendered while this tab was hidden)."""
        super().showEvent(event)
        QTimer.singleShot(0, self._sync_canvas_to_viewport)


    # ── Shared axis styling ───────────────────────────────────────────

    def _gs(self) -> dict[str, Any]:
        return self._host._graph_settings

    def _style_ax(self, ax: Any) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)

    # ── Chart: Summary bar ────────────────────────────────────────────


    def _canvas_width_for_labels(self, labels: list) -> int:
        """Return the pixel width needed to show axes + external legend without clipping.

        The returned width is always at least as wide as the current viewport so
        the canvas fills all available space even when the legend is narrow.
        """
        dpi = int(self._host._graph_settings.get("dpi", 150))
        vp_w = (self._canvas_scroll.viewport().width()
                if self._canvas_scroll else 0)
        if not labels:
            return max(int(self._host._graph_settings.get("max_w", 700)), vp_w)
        max_chars = max(len(str(s)) for s in labels)
        # matplotlib 'small' font ≈ 8.33 pt; avg char width ≈ 0.55 em
        px_per_char = 8.33 * 0.55 * dpi / 72.0
        legend_px = int(22 + max_chars * 5.5 + 24)  # swatch + text + padding
        min_axes_px = 460
        margins_px = 85   # y-axis label + ticks + right gap before legend
        needed = margins_px + min_axes_px + 10 + legend_px
        base = max(needed, int(self._host._graph_settings.get("max_w", 700)))
        return max(base, vp_w)

    def _resize_canvas(self, new_w: int) -> None:
        """Resize canvas to fill the available viewport width (ignoring *new_w* cap).

        *new_w* is kept as a parameter for backward-compatibility but the
        actual width used is the greater of the viewport and *new_w*, so the
        canvas always expands to fill available space rather than being capped.
        """
        if self._canvas is None or self._figure is None:
            return
        gs = self._host._graph_settings
        dpi = int(gs.get("dpi", 150))
        scale = int(gs.get("scale", 100)) / 100.0
        # Use the viewport width when it is larger than the requested width
        if self._canvas_scroll is not None:
            vp_w = self._canvas_scroll.viewport().width()
            effective_w = max(new_w, vp_w)
        else:
            effective_w = new_w
        pw = max(200, int(effective_w * scale))
        base_h = int(gs.get("max_h", 420))
        base_w = int(gs.get("max_w", 700))
        ratio = base_h / max(base_w, 1)
        ph = max(120, int(pw * ratio))
        self._figure.set_size_inches(pw / dpi, ph / dpi)
        self._canvas.setFixedSize(pw, ph)
        self._canvas.updateGeometry()

    def _draw_summary(
        self,
        data: list[dict],
        metric_key: str,
        mode: str,
        vel_unit: str,
    ) -> None:
        """Mean/peak velocity per session (individual) or group (group mode)."""
        ax = self._figure.add_subplot(111)
        gs = self._gs()
        mf = "mean_vel" if metric_key == "mean" else "peak_vel"
        ml = f"{'Mean' if metric_key == 'mean' else 'Peak'} Velocity ({vel_unit})"
        error_style = gs.get("error_style", "SEM")
        cs = int(gs.get("eb_capsize", 4))
        lw = float(gs.get("eb_linewidth", 1.0))

        if mode == "individual":
            by_sess: dict[str, list[float]] = {}
            for d in data:
                by_sess.setdefault(d["session_label"], []).append(d[mf])
            labels = self._host.ordered_session_labels()
            labels = [lb for lb in labels if lb in by_sess]
            vals = [float(np.mean(by_sess[lb])) for lb in labels]
            errs = [_eb_val(np.array(by_sess[lb]), error_style) for lb in labels]
            colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
            x = np.arange(len(labels))
            ax.bar(
                x, vals, yerr=errs, color=colors, alpha=0.85,
                capsize=cs, error_kw={"elinewidth": lw, "capthick": lw},
            )
            if gs.get("show_indiv_points", True):
                for i, lb in enumerate(labels):
                    for v in by_sess[lb]:
                        ax.scatter(i, v, color="black", s=18, zorder=4, alpha=0.55)
            ax.set_xticks(x)
            ax.set_xticklabels(
                labels, rotation=35, ha="right",
                fontsize=gs.get("tick_fontsize", 8),
            )
        else:
            groups_map = self._host._session_groups_for_controls(self._vel_facet_controls)
            checked_grps = self._vel_checked_groups()
            by_grp: dict[str, list[float]] = {}
            for d in data:
                grp = groups_map.get(d["session_label"], "") or d["session_label"]
                if grp not in checked_grps:
                    continue
                by_grp.setdefault(grp, []).append(d[mf])
            grp_list = self._host._ordered_group_list(by_grp.keys(), self._host._split_factors_for_controls(self._vel_facet_controls))
            vals = [float(np.mean(by_grp[g])) for g in grp_list]
            errs = [_eb_val(np.array(by_grp[g]), error_style) for g in grp_list]
            colors = [self._host._group_color(g, i) for i, g in enumerate(grp_list)]
            x = np.arange(len(grp_list))
            ax.bar(
                x, vals, yerr=errs, color=colors, alpha=0.85,
                capsize=cs, error_kw={"elinewidth": lw, "capthick": lw},
            )
            if gs.get("show_indiv_points", True):
                for i, g in enumerate(grp_list):
                    for v in by_grp[g]:
                        ax.scatter(i, v, color="black", s=18, zorder=4, alpha=0.55)
            ax.set_xticks(x)
            ax.set_xticklabels(
                grp_list, rotation=35, ha="right",
                fontsize=gs.get("tick_fontsize", 8),
            )

        beh_label = self._behavior_combo.currentText()
        ax.set_title(
            f"Velocity During \u201c{beh_label}\u201d",
            fontsize=gs.get("title_fontsize", 11), fontweight="bold",
        )
        ax.set_ylabel(ml, fontsize=gs.get("axis_fontsize", 9), fontweight="bold")
        self._resize_canvas(int(self._host._graph_settings.get("max_w", 700)))
        try:
            self._figure.tight_layout(pad=1.5)
        except Exception:
            pass
        self._style_ax(ax)

    # ── Chart: Normalised / Absolute profile ─────────────────────────

    def _draw_profile(
        self,
        data: list[dict],
        mode: str,
        vel_unit: str,
        per_session: bool = False,
    ) -> None:
        """Velocity profile – time-normalised (0–100 %) or absolute time (s).

        When *per_session* is True, bouts within each session are averaged
        first so every session contributes equally regardless of bout count.
        """
        ax = self._figure.add_subplot(111)
        gs = self._gs()
        error_style = gs.get("error_style", "SEM")
        norm_mode = self._get_norm_mode()

        use_context = (hasattr(self, "_context_chk") and self._context_chk.isChecked())
        context_s = self._context_s_spin.value() if use_context else 0.0

        pct_axis = np.linspace(0, 100, _N_NORM_POINTS)

        def _stack_normalised(records: list[dict]) -> np.ndarray:
            traces = []
            for d in records:
                tr = d.get("velocity_trace")
                if tr is None or len(tr) < 2:
                    continue
                xp = np.linspace(0, 100, len(tr))
                traces.append(np.interp(pct_axis, xp, tr))
            return np.array(traces) if traces else np.empty((0, _N_NORM_POINTS))

        def _stack_absolute(records: list[dict], max_frames: int) -> np.ndarray:
            mat = np.full((len(records), max_frames), np.nan)
            valid_rows = 0
            for si, d in enumerate(records):
                tr = d.get("velocity_trace")
                if tr is None or len(tr) < 2:
                    continue
                n = min(len(tr), max_frames)
                mat[valid_rows, :n] = tr[:n]
                valid_rows += 1
            return mat[:valid_rows] if valid_rows else np.empty((0, max_frames))

        def _get_fps() -> float:
            fps = 25.0
            if hasattr(self._host, "_project_fps"):
                try:
                    fps = float(self._host._project_fps()) or 25.0
                except Exception:
                    pass
            return fps

        # ── Context-mode helpers ──────────────────────────────────────
        def _fetch_context_trace(d: dict) -> "tuple[np.ndarray, int, int]":
            """Return (extended_trace, actual_pre_frames, bout_frames).

            Fetches velocity from (start_frame - context_frames) to
            (end_frame + context_frames) using the host's compute method.
            Returns the trace, how many pre-bout frames were actually
            obtained (may be less than context_frames at session start),
            and the bout length in frames.
            """
            sid = d.get("session_id", "")
            sf = int(d.get("start_frame", 0))
            ef = int(d.get("end_frame", sf))
            ctx_f = int(context_s * _get_fps())
            ext_sf = max(0, sf - ctx_f)
            actual_pre = sf - ext_sf
            smooth = self._smooth_spin.value()
            trace = self._host._compute_bout_velocities(sid, ext_sf, ef + ctx_f, smooth)
            return trace, actual_pre, ef - sf + 1

        def _stack_context_norm(records: list[dict]) -> "tuple[np.ndarray, np.ndarray]":
            """Normalise all context traces to a common %-axis.

            Bout occupies 0–100 %; pre/post are expressed as fractions of the
            average bout length so all traces align at 0 and 100.
            """
            ctx_f = int(context_s * _get_fps())
            bout_lens = []
            for d in records:
                ef = int(d.get("end_frame", 0))
                sf = int(d.get("start_frame", 0))
                bl = ef - sf + 1
                if bl > 0:
                    bout_lens.append(bl)
            avg_bl = max(1, int(np.mean(bout_lens))) if bout_lens else 1
            pre_pct = 100.0 * ctx_f / avg_bl
            post_pct = 100.0 * ctx_f / avg_bl
            x_common = np.linspace(-pre_pct, 100.0 + post_pct, _N_NORM_POINTS)
            rows = []
            for d in records:
                trace, actual_pre, bout_f = _fetch_context_trace(d)
                if len(trace) < 2 or bout_f <= 0:
                    continue
                post_f = len(trace) - actual_pre - bout_f
                tr_pre_pct = 100.0 * actual_pre / bout_f
                tr_post_pct = 100.0 * max(0, post_f) / bout_f
                xp = np.linspace(-tr_pre_pct, 100.0 + tr_post_pct, len(trace))
                rows.append(np.interp(x_common, xp, trace))
            if not rows:
                return np.empty((0, _N_NORM_POINTS)), x_common
            return np.array(rows), x_common

        def _stack_context_abs(records: list[dict]) -> "tuple[np.ndarray, np.ndarray, float]":
            """Stack traces aligned at bout start (t=0) for absolute-time context.

            Returns (mat, x_axis, p95_bout_dur_s):
              mat              – rows aligned at t=0; NaN-padded beyond bout end
              x_axis           – time in seconds; x=0 is bout start
              p95_bout_dur_s   – 95th-percentile bout duration (seconds), for shading
            """
            fps = _get_fps()
            ctx_f = int(context_s * fps)
            bout_lens = []
            for d in records:
                bl = int(d.get("end_frame", 0)) - int(d.get("start_frame", 0)) + 1
                if bl > 0:
                    bout_lens.append(bl)
            if not bout_lens:
                return np.empty((0, 1)), np.array([0.0]), 0.0
            p95_f = int(np.percentile(bout_lens, 95))
            total_f = ctx_f + p95_f + ctx_f
            x_axis = (np.arange(total_f) - ctx_f) / fps
            rows_abs: list[np.ndarray] = []
            for d in records:
                trace, actual_pre, _ = _fetch_context_trace(d)
                if len(trace) < 2:
                    continue
                row = np.full(total_f, np.nan)
                # trace[actual_pre] corresponds to t=0; row[ctx_f] = t=0
                t0_row = ctx_f
                t0_trace = actual_pre
                before = min(t0_trace, t0_row)
                after_n = min(len(trace) - t0_trace, total_f - t0_row)
                if before > 0:
                    row[t0_row - before:t0_row] = trace[t0_trace - before:t0_trace]
                if after_n > 0:
                    row[t0_row:t0_row + after_n] = trace[t0_trace:t0_trace + after_n]
                rows_abs.append(row)
            if not rows_abs:
                return np.empty((0, total_f)), x_axis, p95_f / fps
            return np.array(rows_abs), x_axis, p95_f / fps

        def _plot_traces_context(records: list[dict], color: str, label: str) -> None:
            if norm_mode == "normalized":
                # ── Normalised: bout occupies 0–100 % ─────────────────
                if per_session:
                    by_s: dict[str, list[dict]] = {}
                    for d in records:
                        by_s.setdefault(d["session_label"], []).append(d)
                    sess_rows: list[np.ndarray] = []
                    x_common_ref: "np.ndarray | None" = None
                    for sess_recs in by_s.values():
                        mat_s, xc = _stack_context_norm(sess_recs)
                        if mat_s.shape[0] > 0:
                            sess_rows.append(mat_s.mean(axis=0))
                            x_common_ref = xc
                    if not sess_rows or x_common_ref is None:
                        return
                    mat_ctx = np.array(sess_rows)
                    x_axis_ctx = x_common_ref
                else:
                    mat_ctx, x_axis_ctx = _stack_context_norm(records)
                    if mat_ctx.shape[0] == 0:
                        return
                mean_v = mat_ctx.mean(axis=0)
                err_v = np.array([_eb_val(mat_ctx[:, j], error_style) for j in range(len(x_axis_ctx))])
            else:
                # ── Absolute time: aligned at bout start (t=0) ────────
                if per_session:
                    by_s_a: dict[str, list[dict]] = {}
                    for d in records:
                        by_s_a.setdefault(d["session_label"], []).append(d)
                    sess_rows_a: list[np.ndarray] = []
                    x_ref_a: "np.ndarray | None" = None
                    for sess_recs in by_s_a.values():
                        mat_s_a, xc_a, _ = _stack_context_abs(sess_recs)
                        if mat_s_a.shape[0] > 0:
                            sess_rows_a.append(np.nanmean(mat_s_a, axis=0))
                            x_ref_a = xc_a
                    if not sess_rows_a or x_ref_a is None:
                        return
                    mat_ctx = np.array(sess_rows_a)
                    x_axis_ctx = x_ref_a
                else:
                    mat_ctx, x_axis_ctx, _ = _stack_context_abs(records)
                    if mat_ctx.shape[0] == 0:
                        return
                mean_v = np.nanmean(mat_ctx, axis=0)
                n_valid = np.sum(~np.isnan(mat_ctx), axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    std_v = np.nanstd(mat_ctx, axis=0, ddof=1)
                    if error_style == "SD":
                        err_v = std_v
                    elif error_style == "95% CI":
                        err_v = np.where(n_valid > 1, 1.96 * std_v / np.sqrt(np.maximum(n_valid, 1)), 0.0)
                    else:  # SEM
                        err_v = np.where(n_valid > 1, std_v / np.sqrt(np.maximum(n_valid, 1)), 0.0)
            ax.plot(x_axis_ctx, mean_v, color=color, label=label, linewidth=1.5)
            ax.fill_between(x_axis_ctx, mean_v - err_v, mean_v + err_v, color=color, alpha=0.18)

        def _per_session_matrix_norm(records: list[dict]) -> np.ndarray:
            """Return one row per session (mean of that session's bouts)."""
            by_s: dict[str, list[dict]] = {}
            for d in records:
                by_s.setdefault(d["session_label"], []).append(d)
            rows = []
            for sess_recs in by_s.values():
                m = _stack_normalised(sess_recs)
                if m.shape[0] > 0:
                    rows.append(m.mean(axis=0))
            return np.array(rows) if rows else np.empty((0, _N_NORM_POINTS))

        def _per_session_matrix_abs(records: list[dict], max_frames: int) -> tuple:
            """Return (mat, p95) with one row per session for absolute time."""
            by_s: dict[str, list[dict]] = {}
            for d in records:
                by_s.setdefault(d["session_label"], []).append(d)
            rows = []
            sess_p95s = []
            for sess_recs in by_s.values():
                lens = [len(d["velocity_trace"]) for d in sess_recs
                        if d.get("velocity_trace") is not None and len(d.get("velocity_trace", [])) >= 2]
                if not lens:
                    continue
                p95_s = int(np.percentile(lens, 95))
                sess_p95s.append(p95_s)
                m = _stack_absolute(sess_recs, max_frames)
                if m.shape[0] == 0:
                    continue
                col = np.nanmean(m[:, :p95_s], axis=0) if p95_s > 0 else np.array([])
                row = np.full(max_frames, np.nan)
                row[:len(col)] = col
                rows.append(row)
            overall_p95 = int(np.percentile(sess_p95s, 95)) if sess_p95s else 0
            return (np.array(rows) if rows else np.empty((0, max_frames))), overall_p95

        def _plot_traces(records: list[dict], color: str, label: str) -> None:
            if norm_mode == "normalized":
                if per_session:
                    mat = _per_session_matrix_norm(records)
                else:
                    mat = _stack_normalised(records)
                if mat.shape[0] == 0:
                    return
                n_pts = _N_NORM_POINTS
                mean_v = mat.mean(axis=0)
                err_v = np.array([_eb_val(mat[:, j], error_style) for j in range(n_pts)])
                x_axis = pct_axis
            else:
                lengths = [len(d["velocity_trace"]) for d in records
                           if d.get("velocity_trace") is not None and len(d.get("velocity_trace", [])) >= 2]
                if not lengths:
                    return
                fps = _get_fps()
                max_frames = max(lengths)
                if per_session:
                    mat, p95 = _per_session_matrix_abs(records, max_frames)
                else:
                    p95 = int(np.percentile(lengths, 95))
                    mat = _stack_absolute(records, max_frames)
                if mat.shape[0] == 0:
                    return
                mean_v = np.nanmean(mat[:, :p95], axis=0)
                n_valid = np.sum(~np.isnan(mat[:, :p95]), axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    std_v = np.nanstd(mat[:, :p95], axis=0)
                    err_v = np.where(n_valid > 1, std_v / np.sqrt(np.maximum(n_valid, 1)), 0.0)
                x_axis = np.arange(p95) / fps

            ax.plot(x_axis, mean_v, color=color, label=label, linewidth=1.5)
            ax.fill_between(x_axis, mean_v - err_v, mean_v + err_v, color=color, alpha=0.18)

        if use_context:
            # Compute p95 bout duration across all data for abs-mode shading
            _ctx_fps = _get_fps()
            _ctx_bls = [
                int(d.get("end_frame", 0)) - int(d.get("start_frame", 0)) + 1
                for d in data
                if int(d.get("end_frame", 0)) - int(d.get("start_frame", 0)) + 1 > 0
            ]
            _p95_dur_s = float(np.percentile(_ctx_bls, 95)) / _ctx_fps if _ctx_bls else 0.0
            if mode == "individual":
                by_sess_ctx: dict[str, list[dict]] = {}
                for d in data:
                    by_sess_ctx.setdefault(d["session_label"], []).append(d)
                labels_ctx = self._host.ordered_session_labels()
                labels_ctx = [lb for lb in labels_ctx if lb in by_sess_ctx]
                for idx, lb in enumerate(labels_ctx):
                    _plot_traces_context(by_sess_ctx[lb], _PALETTE[idx % len(_PALETTE)], lb)
            else:
                groups_map_ctx = self._host._session_groups_for_controls(self._vel_facet_controls)
                checked_grps_ctx = self._vel_checked_groups()
                by_grp_ctx: dict[str, list[dict]] = {}
                for d in data:
                    grp_ctx = groups_map_ctx.get(d["session_label"], "") or d["session_label"]
                    if grp_ctx not in checked_grps_ctx:
                        continue
                    by_grp_ctx.setdefault(grp_ctx, []).append(d)
                grp_list_ctx = self._host._ordered_group_list(by_grp_ctx.keys(), self._host._split_factors_for_controls(self._vel_facet_controls))
                for idx, grp in enumerate(grp_list_ctx):
                    _plot_traces_context(by_grp_ctx[grp], self._host._group_color(grp, idx), grp)
            # Shade the bout region
            _shade_r = 100.0 if norm_mode == "normalized" else _p95_dur_s
            ax.axvspan(0, _shade_r, color="steelblue", alpha=0.07, zorder=0)
            ax.axvline(0, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.45)
            ax.axvline(_shade_r, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.45)
        else:
            if mode == "individual":
                by_sess: dict[str, list[dict]] = {}
                for d in data:
                    by_sess.setdefault(d["session_label"], []).append(d)
                labels = self._host.ordered_session_labels()
                labels = [lb for lb in labels if lb in by_sess]
                for idx, lb in enumerate(labels):
                    _plot_traces(by_sess[lb], _PALETTE[idx % len(_PALETTE)], lb)
            else:
                groups_map = self._host._session_groups_for_controls(self._vel_facet_controls)
                checked_grps = self._vel_checked_groups()
                by_grp: dict[str, list[dict]] = {}
                for d in data:
                    grp = groups_map.get(d["session_label"], "") or d["session_label"]
                    if grp not in checked_grps:
                        continue
                    by_grp.setdefault(grp, []).append(d)
                grp_list = self._host._ordered_group_list(by_grp.keys(), self._host._split_factors_for_controls(self._vel_facet_controls))
                for idx, grp in enumerate(grp_list):
                    _plot_traces(by_grp[grp], self._host._group_color(grp, idx), grp)

        beh_label = self._behavior_combo.currentText()
        per_sfx = " — per-session mean" if per_session else ""
        if use_context:
            if norm_mode == "normalized":
                norm_suffix = "(normalised, with pre/post context)"
                _xlabel = "Bout Time (%, 0–100 = during bout)"
            else:
                norm_suffix = "(absolute time, with pre/post context)"
                _xlabel = "Time relative to bout start (s)"
        else:
            norm_suffix = "(time-normalised)" if norm_mode == "normalized" else "(absolute time)"
            _xlabel = "Bout Duration (%)" if norm_mode == "normalized" else "Time (s)"
        _title_str = f"Velocity Profile During \u201c{beh_label}\u201d {norm_suffix}{per_sfx}"
        ax.set_xlabel(
            _xlabel,
            fontsize=gs.get("axis_fontsize", 9), fontweight="bold",
        )
        ax.set_ylabel(f"Velocity ({vel_unit})", fontsize=gs.get("axis_fontsize", 9), fontweight="bold")
        if not use_context and norm_mode == "normalized":
            ax.set_xlim(0, 100)
        handles, _labs = ax.get_legend_handles_labels()
        if handles:
            _fig_w_px = self._canvas.width() if self._canvas else int(self._host._graph_settings.get("max_w", 700))
            _rr, _lx = _legend_right_margin(_labs, _fig_w_px)
            _new_w = self._canvas_width_for_labels(_labs)
            self._resize_canvas(_new_w)
            try:
                self._figure.tight_layout(pad=1.5)
                self._figure.subplots_adjust(right=_rr, top=0.88)
            except Exception:
                pass
            self._figure.legend(
                handles, _labs,
                loc="upper left",
                bbox_to_anchor=(_lx, 0.88),
                bbox_transform=self._figure.transFigure,
                borderaxespad=0,
                fontsize=gs.get("legend_fontsize", "small"),
            )
        else:
            self._resize_canvas(int(self._host._graph_settings.get("max_w", 700)))
            try:
                self._figure.tight_layout(pad=1.5)
                self._figure.subplots_adjust(top=0.88)
            except Exception:
                pass
        self._figure.text(
            0.5, 0.97, _title_str,
            ha="center", va="top",
            fontsize=gs.get("title_fontsize", 11), fontweight="bold",
            transform=self._figure.transFigure, clip_on=False,
        )
        self._style_ax(ax)

    # ── Chart: Bout sequence ──────────────────────────────────────────

    def _draw_sequence(
        self,
        data: list[dict],
        metric_key: str,
        mode: str,
        vel_unit: str,
    ) -> None:
        """Velocity vs sequential bout index within each session.

        First-N bouts are highlighted in blue circles, last-N in red squares.
        Individual mode: one scatter+line per session.
        Group mode: mean ± SEM band at each bout index across sessions.
        """
        ax = self._figure.add_subplot(111)
        gs = self._gs()
        mf = "mean_vel" if metric_key == "mean" else "peak_vel"
        ml = f"{'Mean' if metric_key == 'mean' else 'Peak'} Velocity ({vel_unit})"
        n_highlight = self._n_bouts_spin.value()
        error_style = gs.get("error_style", "SEM")

        # Sort bouts within each session chronologically
        by_sess: dict[str, list[dict]] = {}
        for d in data:
            by_sess.setdefault(d["session_label"], []).append(d)
        for bouts in by_sess.values():
            bouts.sort(key=lambda d: d["start_frame"])

        if mode == "individual":
            for sess_idx, (lb, bouts) in enumerate(by_sess.items()):
                color = _PALETTE[sess_idx % len(_PALETTE)]
                xs = list(range(1, len(bouts) + 1))
                ys = [b[mf] for b in bouts]
                ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.4)
                n = len(xs)
                fn = min(n_highlight, n)
                # First-N dots
                ax.scatter(
                    xs[:fn], ys[:fn],
                    color="#2196F3", s=45, zorder=5,
                    label=f"First {n_highlight}" if sess_idx == 0 else "",
                )
                # Last-N squares (non-overlapping with first)
                last_start = max(fn, n - n_highlight)
                if last_start < n:
                    ax.scatter(
                        xs[last_start:], ys[last_start:],
                        color="#F44336", s=45, zorder=5, marker="s",
                        label=f"Last {n_highlight}" if sess_idx == 0 else "",
                    )
        else:
            groups_map = self._host._session_groups_for_controls(self._vel_facet_controls)
            checked_grps = self._vel_checked_groups()
            by_grp_sess: dict[str, dict[str, list[dict]]] = {}
            for lb, bouts in by_sess.items():
                grp = groups_map.get(lb, "") or lb
                if grp not in checked_grps:
                    continue
                by_grp_sess.setdefault(grp, {})[lb] = bouts
            grp_list = self._host._ordered_group_list(by_grp_sess.keys(), self._host._split_factors_for_controls(self._vel_facet_controls))
            for grp_idx, grp in enumerate(grp_list):
                color = self._host._group_color(grp, grp_idx)
                all_sess_bouts = list(by_grp_sess[grp].values())
                max_b = max((len(b) for b in all_sess_bouts), default=0)
                if max_b == 0:
                    continue
                n_sess = len(all_sess_bouts)
                mat = np.full((n_sess, max_b), np.nan)
                for si, bouts in enumerate(all_sess_bouts):
                    for bi, d in enumerate(bouts):
                        mat[si, bi] = d[mf]
                mean_v = np.nanmean(mat, axis=0)
                valid = ~np.isnan(mean_v)
                xs = np.where(valid)[0] + 1
                ys = mean_v[valid]
                errs = np.array([
                    _eb_val(mat[~np.isnan(mat[:, i]), i], error_style)
                    for i in range(max_b) if not np.isnan(mean_v[i])
                ])
                ax.plot(xs, ys, color=color, linewidth=1.8, label=grp)
                ax.fill_between(
                    xs, ys - errs, ys + errs, color=color, alpha=0.2,
                )

        beh_label = self._behavior_combo.currentText()
        _title_str = f"Velocity Across Bout Sequence \u2014 \u201c{beh_label}\u201d"
        ax.set_xlabel("Bout Number", fontsize=gs.get("axis_fontsize", 9), fontweight="bold")
        ax.set_ylabel(ml, fontsize=gs.get("axis_fontsize", 9), fontweight="bold")
        handles, labs_seq = ax.get_legend_handles_labels()
        if handles:
            _fig_w_px = self._canvas.width() if self._canvas else int(self._host._graph_settings.get("max_w", 700))
            _rr, _lx = _legend_right_margin(labs_seq, _fig_w_px)
            _new_w = self._canvas_width_for_labels(labs_seq)
            self._resize_canvas(_new_w)
            try:
                self._figure.tight_layout(pad=1.5)
                self._figure.subplots_adjust(right=_rr, top=0.88)
            except Exception:
                pass
            self._figure.legend(
                handles, labs_seq,
                loc="upper left",
                bbox_to_anchor=(_lx, 0.88),
                bbox_transform=self._figure.transFigure,
                borderaxespad=0,
                fontsize=gs.get("legend_fontsize", "small"),
            )
        else:
            self._resize_canvas(int(self._host._graph_settings.get("max_w", 700)))
            try:
                self._figure.tight_layout(pad=1.5)
                self._figure.subplots_adjust(top=0.88)
            except Exception:
                pass
        self._figure.text(
            0.5, 0.97, _title_str,
            ha="center", va="top",
            fontsize=gs.get("title_fontsize", 11), fontweight="bold",
            transform=self._figure.transFigure, clip_on=False,
        )
        self._style_ax(ax)

    # ── Chart: Distribution ───────────────────────────────────────────

    def _draw_distribution(
        self,
        data: list[dict],
        metric_key: str,
        mode: str,
        vel_unit: str,
    ) -> None:
        """Box plot with individual points overlaid."""
        ax = self._figure.add_subplot(111)
        gs = self._gs()
        mf = "mean_vel" if metric_key == "mean" else "peak_vel"
        ml = f"{'Mean' if metric_key == 'mean' else 'Peak'} Velocity ({vel_unit})"

        if mode == "individual":
            by_sess: dict[str, list[float]] = {}
            for d in data:
                by_sess.setdefault(d["session_label"], []).append(d[mf])
            labels = self._host.ordered_session_labels()
            labels = [lb for lb in labels if lb in by_sess]
            plot_data = [by_sess[lb] for lb in labels]
            colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
        else:
            groups_map = self._host._session_groups_for_controls(self._vel_facet_controls)
            checked_grps = self._vel_checked_groups()
            by_grp: dict[str, list[float]] = {}
            for d in data:
                grp = groups_map.get(d["session_label"], "") or d["session_label"]
                if grp not in checked_grps:
                    continue
                by_grp.setdefault(grp, []).append(d[mf])
            labels = self._host._ordered_group_list(by_grp.keys(), self._host._split_factors_for_controls(self._vel_facet_controls))
            plot_data = [by_grp[lb] for lb in labels]
            colors = [
                self._host._group_color(g, i) for i, g in enumerate(labels)
            ]

        if not plot_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return

        bp = ax.boxplot(
            plot_data, patch_artist=True, notch=False,
            medianprops={"color": "white", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        rng = np.random.default_rng(seed=42)
        for i, vals in enumerate(plot_data):
            jitter = rng.uniform(-0.14, 0.14, len(vals))
            ax.scatter(
                np.full(len(vals), i + 1) + jitter, vals,
                color="black", s=14, alpha=0.5, zorder=4,
            )

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(
            labels, rotation=35, ha="right",
            fontsize=gs.get("tick_fontsize", 8),
        )
        beh_label = self._behavior_combo.currentText()
        ax.set_title(
            f"Velocity Distribution During \u201c{beh_label}\u201d",
            fontsize=gs.get("title_fontsize", 11), fontweight="bold",
        )
        ax.set_ylabel(ml, fontsize=gs.get("axis_fontsize", 9), fontweight="bold")
        self._resize_canvas(int(self._host._graph_settings.get("max_w", 700)))
        try:
            self._figure.tight_layout(pad=1.5)
        except Exception:
            pass
        self._style_ax(ax)

    # ── Chart: First vs Last N ────────────────────────────────────────

    def _draw_first_last(
        self,
        data: list[dict],
        metric_key: str,
        mode: str,
        vel_unit: str,
    ) -> None:
        """Paired bar chart: mean velocity of first-N vs last-N bouts.

        In individual mode each session gets a pair of bars linked by a
        thin grey line so within-subject change is immediately visible.
        In group mode error bars show SEM/SD/CI across sessions.
        """
        ax = self._figure.add_subplot(111)
        gs = self._gs()
        mf = "mean_vel" if metric_key == "mean" else "peak_vel"
        ml = f"{'Mean' if metric_key == 'mean' else 'Peak'} Velocity ({vel_unit})"
        n_bouts = self._n_bouts_spin.value()
        error_style = gs.get("error_style", "SEM")
        cs = int(gs.get("eb_capsize", 4))
        lw_eb = float(gs.get("eb_linewidth", 1.0))

        # Build chronologically sorted bouts per session
        by_sess: dict[str, list[dict]] = {}
        for d in data:
            by_sess.setdefault(d["session_label"], []).append(d)
        for bouts in by_sess.values():
            bouts.sort(key=lambda d: d["start_frame"])

        def _first_mean(bouts: list[dict]) -> float:
            fn = min(n_bouts, len(bouts))
            return float(np.mean([b[mf] for b in bouts[:fn]])) if fn else 0.0

        def _last_mean(bouts: list[dict]) -> float:
            n = len(bouts)
            fn = min(n_bouts, n)
            last_start = max(fn, n - n_bouts)
            sub = bouts[last_start:]
            return float(np.mean([b[mf] for b in sub])) if sub else 0.0

        w = 0.35

        if mode == "individual":
            labels = self._host.ordered_session_labels()
            labels = [lb for lb in labels if lb in by_sess]
            x = np.arange(len(labels))
            first_means = [_first_mean(by_sess[lb]) for lb in labels]
            last_means = [_last_mean(by_sess[lb]) for lb in labels]
            ax.bar(
                x - w / 2, first_means, w,
                color="#2196F3", alpha=0.85, label=f"First {n_bouts}",
            )
            ax.bar(
                x + w / 2, last_means, w,
                color="#F44336", alpha=0.85, label=f"Last {n_bouts}",
            )
            for i in range(len(labels)):
                ax.plot(
                    [x[i] - w / 2, x[i] + w / 2],
                    [first_means[i], last_means[i]],
                    color="gray", linewidth=0.9, alpha=0.55,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(
                labels, rotation=35, ha="right",
                fontsize=gs.get("tick_fontsize", 8),
            )
        else:
            groups_map = self._host._session_groups_for_controls(self._vel_facet_controls)
            checked_grps = self._vel_checked_groups()
            by_grp: dict[str, dict[str, list[dict]]] = {}
            for lb, bouts in by_sess.items():
                grp = groups_map.get(lb, "") or lb
                if grp not in checked_grps:
                    continue
                by_grp.setdefault(grp, {})[lb] = bouts
            grp_list = self._host._ordered_group_list(by_grp.keys(), self._host._split_factors_for_controls(self._vel_facet_controls))
            x = np.arange(len(grp_list))
            first_means, last_means = [], []
            first_errs, last_errs = [], []
            for grp in grp_list:
                fv = [_first_mean(b) for b in by_grp[grp].values()]
                lv = [_last_mean(b) for b in by_grp[grp].values()]
                first_means.append(float(np.mean(fv)) if fv else 0.0)
                last_means.append(float(np.mean(lv)) if lv else 0.0)
                first_errs.append(_eb_val(np.array(fv), error_style) if fv else 0.0)
                last_errs.append(_eb_val(np.array(lv), error_style) if lv else 0.0)
            ax.bar(
                x - w / 2, first_means, w,
                color="#2196F3", alpha=0.85,
                yerr=first_errs, capsize=cs,
                error_kw={"elinewidth": lw_eb, "capthick": lw_eb},
                label=f"First {n_bouts}",
            )
            ax.bar(
                x + w / 2, last_means, w,
                color="#F44336", alpha=0.85,
                yerr=last_errs, capsize=cs,
                error_kw={"elinewidth": lw_eb, "capthick": lw_eb},
                label=f"Last {n_bouts}",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                grp_list, rotation=35, ha="right",
                fontsize=gs.get("tick_fontsize", 8),
            )

        beh_label = self._behavior_combo.currentText()
        _title_str = f"First {n_bouts} vs Last {n_bouts} Bouts \u2014 \u201c{beh_label}\u201d"
        ax.set_ylabel(ml, fontsize=gs.get("axis_fontsize", 9), fontweight="bold")
        handles_fl, labs_fl = ax.get_legend_handles_labels()
        if handles_fl:
            _fig_w_px = self._canvas.width() if self._canvas else int(self._host._graph_settings.get("max_w", 700))
            _rr_fl, _lx_fl = _legend_right_margin(labs_fl, _fig_w_px)
            _new_w = self._canvas_width_for_labels(labs_fl)
            self._resize_canvas(_new_w)
            try:
                self._figure.tight_layout(pad=1.5)
                self._figure.subplots_adjust(right=_rr_fl, top=0.88)
            except Exception:
                pass
            self._figure.legend(
                handles_fl, labs_fl,
                loc="upper left",
                bbox_to_anchor=(_lx_fl, 0.88),
                bbox_transform=self._figure.transFigure,
                borderaxespad=0,
                fontsize=gs.get("legend_fontsize", "small"),
            )
        else:
            self._resize_canvas(int(self._host._graph_settings.get("max_w", 700)))
            try:
                self._figure.tight_layout(pad=1.5)
                self._figure.subplots_adjust(top=0.88)
            except Exception:
                pass
        self._figure.text(
            0.5, 0.97, _title_str,
            ha="center", va="top",
            fontsize=gs.get("title_fontsize", 11), fontweight="bold",
            transform=self._figure.transFigure, clip_on=False,
        )
        self._style_ax(ax)

    # ── Settings ──────────────────────────────────────────────────────

    def _open_graph_settings(self) -> None:
        gs = self._gs()
        dlg = QDialog(self)
        dlg.setWindowTitle("Velocity Graph Settings")
        dlg.setMinimumWidth(360)
        form = QFormLayout()
        form.setSpacing(8)

        title_fs = QSpinBox(dlg)
        title_fs.setRange(6, 32)
        title_fs.setValue(int(gs.get("title_fontsize", 11)))
        form.addRow("Title font size:", title_fs)

        axis_fs = QSpinBox(dlg)
        axis_fs.setRange(4, 28)
        axis_fs.setValue(int(gs.get("axis_fontsize", 9)))
        form.addRow("Axis label font size:", axis_fs)

        tick_fs = QSpinBox(dlg)
        tick_fs.setRange(4, 24)
        tick_fs.setValue(int(gs.get("tick_fontsize", 8)))
        form.addRow("Tick font size:", tick_fs)

        error_cb = QComboBox(dlg)
        for es in ("SEM", "SD", "95% CI", "None"):
            error_cb.addItem(es, userData=es)
        error_cb.setCurrentText(str(gs.get("error_style", "SEM")))
        error_cb.setToolTip(
            "Error band style on line charts.\n"
            "SEM = standard error of the mean\n"
            "SD = standard deviation\n"
            "95% CI = 1.96 \u00d7 SEM\n"
            "None = no shading"
        )
        form.addRow("Error band style:", error_cb)

        indiv_chk = QCheckBox("Show individual data points", dlg)
        indiv_chk.setChecked(bool(gs.get("show_indiv_points", True)))
        form.addRow("", indiv_chk)

        dpi_spin = QSpinBox(dlg)
        dpi_spin.setRange(50, 600)
        dpi_spin.setSingleStep(50)
        dpi_spin.setValue(int(gs.get("dpi", 150)))
        form.addRow("Export DPI:", dpi_spin)

        max_w_spin = QSpinBox(dlg)
        max_w_spin.setRange(200, 3840)
        max_w_spin.setSingleStep(50)
        max_w_spin.setSuffix(" px")
        max_w_spin.setValue(int(gs.get("max_w", 700)))
        form.addRow("Max figure width:", max_w_spin)

        max_h_spin = QSpinBox(dlg)
        max_h_spin.setRange(150, 2160)
        max_h_spin.setSingleStep(50)
        max_h_spin.setSuffix(" px")
        max_h_spin.setValue(int(gs.get("max_h", 420)))
        form.addRow("Max figure height:", max_h_spin)

        eb_capsize = QSpinBox(dlg)
        eb_capsize.setRange(0, 20)
        eb_capsize.setValue(int(gs.get("eb_capsize", 4)))
        eb_capsize.setSuffix(" pt")
        form.addRow("Error bar cap size:", eb_capsize)

        eb_lw = QDoubleSpinBox(dlg)
        eb_lw.setRange(0.2, 6.0)
        eb_lw.setSingleStep(0.2)
        eb_lw.setDecimals(1)
        eb_lw.setValue(float(gs.get("eb_linewidth", 1.0)))
        eb_lw.setSuffix(" pt")
        form.addRow("Error bar line width:", eb_lw)

        scale_spin = QSpinBox(dlg)
        scale_spin.setRange(25, 300)
        scale_spin.setSingleStep(10)
        scale_spin.setSuffix("%")
        scale_spin.setValue(int(gs.get("scale", 100)))
        scale_spin.setToolTip("Scale the figure canvas relative to max_w × max_h.")
        form.addRow("Figure scale:", scale_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout(dlg)
        layout.addLayout(form)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        gs["title_fontsize"] = title_fs.value()
        gs["axis_fontsize"] = axis_fs.value()
        gs["tick_fontsize"] = tick_fs.value()
        gs["error_style"] = str(error_cb.currentData() or "SEM")
        gs["show_indiv_points"] = indiv_chk.isChecked()
        gs["dpi"] = dpi_spin.value()
        gs["max_w"] = max_w_spin.value()
        gs["max_h"] = max_h_spin.value()
        gs["eb_capsize"] = eb_capsize.value()
        gs["eb_linewidth"] = eb_lw.value()
        gs["scale"] = scale_spin.value()
        self._scale_spin.setValue(scale_spin.value())
        self._update()

    # ── Export ────────────────────────────────────────────────────────

    def _export_figure(self) -> None:
        if self._figure is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Velocity Chart", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        dpi = int(self._host._graph_settings.get("dpi", 150))
        try:
            self._figure.savefig(path, dpi=dpi, bbox_inches="tight")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))

    def _export_data(self) -> None:
        """Export per-bout velocity records as a clean, paste-friendly CSV.

        Columns (in order):
            session  group  bout_number  duration_s
            mean_vel_[unit]  peak_vel_[unit]
            outlier_excluded (if outlier detection is active)
        The file is sorted by session then bout order for easy copy-paste
        into graphing software.  All column headers use ASCII only.
        """
        bid = self._behavior_combo.currentData()
        if not bid:
            return

        metric_key = self._get_metric()
        all_data = self._get_vel_data(bid)
        if not all_data:
            QMessageBox.information(self, "No Data", "No velocity data to export.")
            return

        # Apply session filter (match what is shown on screen)
        checked_sessions = self._checked_sessions()
        if checked_sessions:
            all_data = [d for d in all_data if d["session_label"] in checked_sessions]

        vel_unit = self._vel_unit_label()
        # Determine unit suffix for column names (ASCII only)
        unit_sfx = "cm_s" if vel_unit == "cm/s" else "px_s"
        mean_col = f"mean_vel_{unit_sfx}"
        peak_col = f"peak_vel_{unit_sfx}"

        # Apply outlier filter and track which bouts are outliers when
        # "Exclude" mode is off (Winsorize) so we can flag them.
        outlier_active = self._outlier_chk.isChecked()
        outlier_action = self._outlier_action_combo.currentData() or "exclude"

        if outlier_active:
            filtered_data, _ = self._apply_outlier_filter(all_data, metric_key)
            excluded_set: set[int] = set()
            if outlier_action == "exclude":
                # identify excluded bouts by their position in all_data
                filtered_ids = {id(d) for d in filtered_data}
                excluded_set = {i for i, d in enumerate(all_data) if id(d) not in filtered_ids}
                export_data = all_data  # export all, with flag
            else:
                export_data = filtered_data  # winsorized values
        else:
            export_data = all_data
            excluded_set = set()

        # Build per-session bout counters
        sess_bout_counter: dict[str, int] = {}
        rows = []
        for i, d in enumerate(export_data):
            sess = str(d.get("session_label") or d.get("session_id") or "")
            # Strip any non-ASCII characters that could cause paste issues
            sess_clean = sess.encode("ascii", "replace").decode("ascii")
            # Re-read the group label live from _session_groups so that
            # interaction / multi-factor labels ("Male × fentanyl") are always
            # current even when the vel_cache was built under a different factor.
            grp = str(self._host._session_groups.get(sess, d.get("group") or ""))
            sess_bout_counter[sess] = sess_bout_counter.get(sess, 0) + 1
            bout_num = sess_bout_counter[sess]
            row: dict = {
                "session": sess_clean,
                "group": grp,
                "bout_number": bout_num,
                "duration_s": round(float(d.get("duration_s", 0)), 4),
                mean_col: round(float(d.get("mean_vel", 0)), 4),
                peak_col: round(float(d.get("peak_vel", 0)), 4),
            }
            if outlier_active and outlier_action == "exclude":
                row["outlier"] = 1 if i in excluded_set else 0
            rows.append(row)

        if not rows:
            QMessageBox.information(self, "No Data", "No data rows to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Velocity Data", "", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False, encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))

    # ── Export profile data (copy-paste friendly dialog) ──────────────

    def _export_profile_data(self) -> None:
        """Build the mean ± error band data for the velocity profile chart and
        show a copy-paste friendly dialog.

        The dialog displays a live CSV preview, a "Copy to Clipboard" button
        for instant paste into Excel / Prism / R, and a "Save CSV…" button
        for file output.  Two formats are offered:

        Tidy (long) — one row per (label × x-point):
            x | label | mean_velocity | error | n

        Wide — one column-pair per label:
            x | <label>_mean | <label>_error | <label>_n | …
        """
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QTextEdit,
            QPushButton as _QPB, QLabel as _QL, QRadioButton,
            QButtonGroup as _BG, QDialogButtonBox,
        )
        from PySide6.QtGui import QClipboard
        from PySide6.QtCore import QCoreApplication

        bid = self._behavior_combo.currentData()
        if not bid:
            QMessageBox.information(self, "No Data", "No behavior selected.")
            return

        metric_key = self._get_metric()
        all_data = self._get_vel_data(bid)
        if not all_data:
            QMessageBox.information(self, "No Data", "No velocity data available.")
            return

        checked_sessions = self._checked_sessions()
        if checked_sessions:
            all_data = [d for d in all_data if d["session_label"] in checked_sessions]
        if not all_data:
            QMessageBox.information(self, "No Data", "No data for the selected sessions.")
            return

        all_data, _ = self._apply_outlier_filter(all_data, metric_key)
        if not all_data:
            QMessageBox.information(self, "No Data", "All bouts were excluded by the outlier filter.")
            return

        vel_unit = self._vel_unit_label()
        unit_sfx = "cm_s" if vel_unit == "cm/s" else "px_s"
        norm_mode = self._get_norm_mode()
        mode = self._get_mode()
        bout_mode = self._get_bout_mode()
        per_session = (bout_mode == "per_session")
        use_context = (hasattr(self, "_context_chk") and self._context_chk.isChecked())
        context_s = self._context_s_spin.value() if use_context else 0.0
        error_style = self._gs().get("error_style", "SEM")
        beh_label = self._behavior_combo.currentText()

        # ── Helpers (mirrors _draw_profile logic) ─────────────────────
        n_pts = _N_NORM_POINTS
        pct_axis = np.linspace(0, 100, n_pts)

        def _get_fps() -> float:
            fps = 25.0
            if hasattr(self._host, "_project_fps"):
                try:
                    fps = float(self._host._project_fps()) or 25.0
                except Exception:
                    pass
            return fps

        def _stack_norm(records: list[dict]) -> np.ndarray:
            traces = []
            for d in records:
                tr = d.get("velocity_trace")
                if tr is None or len(tr) < 2:
                    continue
                xp = np.linspace(0, 100, len(tr))
                traces.append(np.interp(pct_axis, xp, tr))
            return np.array(traces) if traces else np.empty((0, n_pts))

        def _stack_abs(records: list[dict], max_frames: int) -> np.ndarray:
            mat = np.full((len(records), max_frames), np.nan)
            vr = 0
            for d in records:
                tr = d.get("velocity_trace")
                if tr is None or len(tr) < 2:
                    continue
                n = min(len(tr), max_frames)
                mat[vr, :n] = tr[:n]
                vr += 1
            return mat[:vr] if vr else np.empty((0, max_frames))

        def _eb(arr: np.ndarray) -> float:
            return float(_eb_val(arr, error_style))

        def _compute_series_norm(records: list[dict], ps: bool) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
            """Return (x_axis, mean, err, n_per_point) for normalised mode."""
            if ps:
                by_s: dict[str, list[dict]] = {}
                for d in records:
                    by_s.setdefault(d["session_label"], []).append(d)
                rows = []
                for sr in by_s.values():
                    m = _stack_norm(sr)
                    if m.shape[0] > 0:
                        rows.append(m.mean(axis=0))
                mat = np.array(rows) if rows else np.empty((0, n_pts))
            else:
                mat = _stack_norm(records)
            if mat.shape[0] == 0:
                return pct_axis, np.full(n_pts, np.nan), np.full(n_pts, np.nan), np.zeros(n_pts)
            mean_v = mat.mean(axis=0)
            err_v = np.array([_eb(mat[:, j]) for j in range(n_pts)])
            n_v = np.full(n_pts, float(mat.shape[0]))
            return pct_axis, mean_v, err_v, n_v

        def _compute_series_abs(records: list[dict], ps: bool) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
            """Return (x_axis_s, mean, err, n) for absolute-time mode."""
            fps = _get_fps()
            lengths = [len(d["velocity_trace"]) for d in records
                       if d.get("velocity_trace") is not None and len(d.get("velocity_trace", [])) >= 2]
            if not lengths:
                return np.array([0.0]), np.array([np.nan]), np.array([np.nan]), np.array([0.0])
            max_frames = max(lengths)
            if ps:
                by_s: dict[str, list[dict]] = {}
                for d in records:
                    by_s.setdefault(d["session_label"], []).append(d)
                rows, sess_p95s = [], []
                for sr in by_s.values():
                    lens_s = [len(d["velocity_trace"]) for d in sr
                              if d.get("velocity_trace") is not None and len(d.get("velocity_trace", [])) >= 2]
                    if not lens_s:
                        continue
                    p95_s = int(np.percentile(lens_s, 95))
                    sess_p95s.append(p95_s)
                    m = _stack_abs(sr, max_frames)
                    if m.shape[0] == 0:
                        continue
                    col = np.nanmean(m[:, :p95_s], axis=0) if p95_s > 0 else np.array([])
                    row = np.full(max_frames, np.nan)
                    row[:len(col)] = col
                    rows.append(row)
                mat = np.array(rows) if rows else np.empty((0, max_frames))
                p95 = int(np.percentile(sess_p95s, 95)) if sess_p95s else 0
            else:
                p95 = int(np.percentile(lengths, 95))
                mat = _stack_abs(records, max_frames)
            if mat.shape[0] == 0 or p95 == 0:
                return np.array([0.0]), np.array([np.nan]), np.array([np.nan]), np.array([0.0])
            mean_v = np.nanmean(mat[:, :p95], axis=0)
            n_v = np.sum(~np.isnan(mat[:, :p95]), axis=0).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                std_v = np.nanstd(mat[:, :p95], axis=0)
                if error_style == "SD":
                    err_v = std_v
                elif error_style == "95% CI":
                    err_v = np.where(n_v > 1, 1.96 * std_v / np.sqrt(np.maximum(n_v, 1)), 0.0)
                else:
                    err_v = np.where(n_v > 1, std_v / np.sqrt(np.maximum(n_v, 1)), 0.0)
            x_axis = np.arange(p95) / fps
            return x_axis, mean_v, err_v, n_v

        def _compute_series_ctx_norm(records: list[dict], ps: bool) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
            fps = _get_fps()
            ctx_f = int(context_s * fps)
            bout_lens = []
            for d in records:
                bl = int(d.get("end_frame", 0)) - int(d.get("start_frame", 0)) + 1
                if bl > 0:
                    bout_lens.append(bl)
            avg_bl = max(1, int(np.mean(bout_lens))) if bout_lens else 1
            pre_pct = 100.0 * ctx_f / avg_bl
            post_pct = 100.0 * ctx_f / avg_bl
            x_common = np.linspace(-pre_pct, 100.0 + post_pct, n_pts)
            smooth = self._smooth_spin.value()

            def _fetch(d: dict) -> "tuple[np.ndarray, int, int]":
                sid = d.get("session_id", "")
                sf = int(d.get("start_frame", 0))
                ef = int(d.get("end_frame", sf))
                ext_sf = max(0, sf - ctx_f)
                actual_pre = sf - ext_sf
                tr = self._host._compute_bout_velocities(sid, ext_sf, ef + ctx_f, smooth)
                return tr, actual_pre, ef - sf + 1

            def _stack_ctx(recs: list[dict]) -> np.ndarray:
                rows = []
                for d in recs:
                    tr, actual_pre, bout_f = _fetch(d)
                    if len(tr) < 2 or bout_f <= 0:
                        continue
                    post_f = len(tr) - actual_pre - bout_f
                    tr_pre_pct = 100.0 * actual_pre / bout_f
                    tr_post_pct = 100.0 * max(0, post_f) / bout_f
                    xp = np.linspace(-tr_pre_pct, 100.0 + tr_post_pct, len(tr))
                    rows.append(np.interp(x_common, xp, tr))
                return np.array(rows) if rows else np.empty((0, n_pts))

            if ps:
                by_s: dict[str, list[dict]] = {}
                for d in records:
                    by_s.setdefault(d["session_label"], []).append(d)
                sess_rows = []
                for sr in by_s.values():
                    m = _stack_ctx(sr)
                    if m.shape[0] > 0:
                        sess_rows.append(m.mean(axis=0))
                mat = np.array(sess_rows) if sess_rows else np.empty((0, n_pts))
            else:
                mat = _stack_ctx(records)
            if mat.shape[0] == 0:
                return x_common, np.full(n_pts, np.nan), np.full(n_pts, np.nan), np.zeros(n_pts)
            mean_v = mat.mean(axis=0)
            err_v = np.array([_eb(mat[:, j]) for j in range(n_pts)])
            n_v = np.full(n_pts, float(mat.shape[0]))
            return x_common, mean_v, err_v, n_v

        def _compute_series_ctx_abs(records: list[dict], ps: bool) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
            fps = _get_fps()
            ctx_f = int(context_s * fps)
            bout_lens = []
            for d in records:
                bl = int(d.get("end_frame", 0)) - int(d.get("start_frame", 0)) + 1
                if bl > 0:
                    bout_lens.append(bl)
            if not bout_lens:
                return np.array([0.0]), np.array([np.nan]), np.array([np.nan]), np.array([0.0])
            p95_f = int(np.percentile(bout_lens, 95))
            total_f = ctx_f + p95_f + ctx_f
            x_axis = (np.arange(total_f) - ctx_f) / fps
            smooth = self._smooth_spin.value()

            def _fetch(d: dict) -> "tuple[np.ndarray, int]":
                sid = d.get("session_id", "")
                sf = int(d.get("start_frame", 0))
                ef = int(d.get("end_frame", sf))
                ext_sf = max(0, sf - ctx_f)
                actual_pre = sf - ext_sf
                tr = self._host._compute_bout_velocities(sid, ext_sf, ef + ctx_f, smooth)
                return tr, actual_pre

            def _stack_ctx_a(recs: list[dict]) -> np.ndarray:
                rows = []
                for d in recs:
                    tr, actual_pre = _fetch(d)
                    if len(tr) < 2:
                        continue
                    row = np.full(total_f, np.nan)
                    t0_row, t0_tr = ctx_f, actual_pre
                    before = min(t0_tr, t0_row)
                    after_n = min(len(tr) - t0_tr, total_f - t0_row)
                    if before > 0:
                        row[t0_row - before:t0_row] = tr[t0_tr - before:t0_tr]
                    if after_n > 0:
                        row[t0_row:t0_row + after_n] = tr[t0_tr:t0_tr + after_n]
                    rows.append(row)
                return np.array(rows) if rows else np.empty((0, total_f))

            if ps:
                by_s: dict[str, list[dict]] = {}
                for d in records:
                    by_s.setdefault(d["session_label"], []).append(d)
                sess_rows = []
                for sr in by_s.values():
                    m = _stack_ctx_a(sr)
                    if m.shape[0] > 0:
                        sess_rows.append(np.nanmean(m, axis=0))
                mat = np.array(sess_rows) if sess_rows else np.empty((0, total_f))
            else:
                mat = _stack_ctx_a(records)
            if mat.shape[0] == 0:
                return x_axis, np.full(total_f, np.nan), np.full(total_f, np.nan), np.zeros(total_f)
            mean_v = np.nanmean(mat, axis=0)
            n_v = np.sum(~np.isnan(mat), axis=0).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                std_v = np.nanstd(mat, axis=0, ddof=1)
                if error_style == "SD":
                    err_v = std_v
                elif error_style == "95% CI":
                    err_v = np.where(n_v > 1, 1.96 * std_v / np.sqrt(np.maximum(n_v, 1)), 0.0)
                else:
                    err_v = np.where(n_v > 1, std_v / np.sqrt(np.maximum(n_v, 1)), 0.0)
            return x_axis, mean_v, err_v, n_v

        # ── Collect series per group/session ──────────────────────────
        series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        # Each entry: (label, x_axis, mean_v, err_v, n_v)

        if mode == "individual":
            by_sess: dict[str, list[dict]] = {}
            for d in all_data:
                by_sess.setdefault(d["session_label"], []).append(d)
            labels_ord = self._host.ordered_session_labels()
            labels_ord = [lb for lb in labels_ord if lb in by_sess]
            for lb in labels_ord:
                recs = by_sess[lb]
                if use_context:
                    if norm_mode == "normalized":
                        x, mv, ev, nv = _compute_series_ctx_norm(recs, per_session)
                    else:
                        x, mv, ev, nv = _compute_series_ctx_abs(recs, per_session)
                else:
                    if norm_mode == "normalized":
                        x, mv, ev, nv = _compute_series_norm(recs, per_session)
                    else:
                        x, mv, ev, nv = _compute_series_abs(recs, per_session)
                series.append((lb, x, mv, ev, nv))
        else:
            groups_map = self._host._session_groups_for_controls(self._vel_facet_controls)
            checked_grps = self._vel_checked_groups()
            by_grp: dict[str, list[dict]] = {}
            for d in all_data:
                grp = groups_map.get(d["session_label"], "") or d["session_label"]
                if grp not in checked_grps:
                    continue
                by_grp.setdefault(grp, []).append(d)
            grp_list = self._host._ordered_group_list(by_grp.keys(), self._host._split_factors_for_controls(self._vel_facet_controls))
            for grp in grp_list:
                recs = by_grp[grp]
                if use_context:
                    if norm_mode == "normalized":
                        x, mv, ev, nv = _compute_series_ctx_norm(recs, per_session)
                    else:
                        x, mv, ev, nv = _compute_series_ctx_abs(recs, per_session)
                else:
                    if norm_mode == "normalized":
                        x, mv, ev, nv = _compute_series_norm(recs, per_session)
                    else:
                        x, mv, ev, nv = _compute_series_abs(recs, per_session)
                series.append((grp, x, mv, ev, nv))

        if not series:
            QMessageBox.information(self, "No Data", "No profile data could be computed.")
            return

        x_lbl = "x_pct" if (norm_mode == "normalized" and not use_context) else "x_s"
        mean_col = f"mean_{unit_sfx}"
        err_col = f"{error_style.replace(' ', '_').replace('%', 'pct')}_{unit_sfx}"
        err_col = err_col.replace("/", "_")

        def _build_tidy(srs: list) -> str:
            lines = [f"behavior,{x_lbl},label,{mean_col},{err_col},n"]
            beh_clean = beh_label.encode("ascii", "replace").decode("ascii")
            for lbl, x_arr, mv, ev, nv in srs:
                lbl_clean = str(lbl)
                for xi, (xv, m, e, n) in enumerate(zip(x_arr, mv, ev, nv)):
                    lines.append(
                        f"{beh_clean},{xv:.4f},{lbl_clean},{m:.4f},{e:.4f},{int(n)}"
                    )
            return "\n".join(lines)

        def _build_wide(srs: list) -> str:
            # All series share x_axis (use the first one as reference)
            x_ref = srs[0][1]
            header_parts = [x_lbl]
            for lbl, *_ in srs:
                lbl_clean = str(lbl).replace(",", "_").replace("\u00d7", "x")
                header_parts += [
                    f"{lbl_clean}_{mean_col}",
                    f"{lbl_clean}_{err_col}",
                    f"{lbl_clean}_n",
                ]
            lines = [",".join(header_parts)]
            max_len = max(len(s[1]) for s in srs)
            for i in range(max_len):
                xv = x_ref[i] if i < len(x_ref) else ""
                row_parts = [f"{xv:.4f}" if isinstance(xv, float) else str(xv)]
                for _lbl, x_arr, mv, ev, nv in srs:
                    if i < len(mv):
                        row_parts += [f"{mv[i]:.4f}", f"{ev[i]:.4f}", str(int(nv[i]))]
                    else:
                        row_parts += ["", "", ""]
                lines.append(",".join(row_parts))
            return "\n".join(lines)

        # ── Dialog ────────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Export Profile Data \u2014 \u201c{beh_label}\u201d")
        dlg.setMinimumSize(620, 440)
        vbox = QVBoxLayout(dlg)
        vbox.setSpacing(8)

        info_lbl = _QL(
            f"<b>Behavior:</b> {beh_label} &nbsp;|&nbsp; "
            f"<b>Mode:</b> {'Normalised (%)' if norm_mode == 'normalized' else 'Absolute time (s)'} &nbsp;|&nbsp; "
            f"<b>Error:</b> {error_style} &nbsp;|&nbsp; "
            f"<b>Series:</b> {', '.join(s[0] for s in series)}"
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("font-size:11px;")
        vbox.addWidget(info_lbl)

        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(12)
        fmt_lbl = _QL("Format:")
        fmt_lbl.setStyleSheet("font-size:11px;")
        fmt_row.addWidget(fmt_lbl)
        rb_tidy = QRadioButton("Tidy (long) — one row per data point")
        rb_wide = QRadioButton("Wide — one column-pair per series")
        rb_tidy.setChecked(True)
        fmt_grp = _BG(dlg)
        fmt_grp.addButton(rb_tidy, 0)
        fmt_grp.addButton(rb_wide, 1)
        fmt_row.addWidget(rb_tidy)
        fmt_row.addWidget(rb_wide)
        fmt_row.addStretch(1)
        vbox.addLayout(fmt_row)

        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setFontFamily("Courier New, Consolas, monospace")
        preview.setFontPointSize(9)
        preview.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        vbox.addWidget(preview, 1)

        def _refresh_preview() -> None:
            if rb_tidy.isChecked():
                preview.setPlainText(_build_tidy(series))
            else:
                preview.setPlainText(_build_wide(series))

        rb_tidy.toggled.connect(lambda _: _refresh_preview())
        _refresh_preview()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        copy_btn = _QPB("Copy to Clipboard")
        copy_btn.setToolTip("Copy the CSV text above directly to the system clipboard.")
        copy_btn.setStyleSheet(
            "QPushButton{padding:4px 14px;background:#1565c0;border:none;"
            "border-radius:3px;color:#fff;font-weight:bold;}"
            "QPushButton:hover{background:#1976d2;}"
        )

        def _copy() -> None:
            cb = QCoreApplication.instance().clipboard()
            cb.setText(preview.toPlainText())
            copy_btn.setText("\u2713 Copied!")
            from PySide6.QtCore import QTimer as _QT
            _QT.singleShot(1800, lambda: copy_btn.setText("Copy to Clipboard"))

        copy_btn.clicked.connect(_copy)
        btn_row.addWidget(copy_btn)

        save_btn = _QPB("Save CSV\u2026")
        save_btn.setToolTip("Save the CSV to a file.")

        def _save() -> None:
            path, _ = QFileDialog.getSaveFileName(
                dlg, "Save Profile Data", "", "CSV (*.csv)"
            )
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(preview.toPlainText())
                save_btn.setText("\u2713 Saved!")
                from PySide6.QtCore import QTimer as _QT
                _QT.singleShot(1800, lambda: save_btn.setText("Save CSV\u2026"))
            except Exception as exc:
                QMessageBox.warning(dlg, "Save Failed", str(exc))

        save_btn.clicked.connect(_save)
        btn_row.addWidget(save_btn)

        btn_row.addStretch(1)
        close_btn = _QPB("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        vbox.addLayout(btn_row)

        dlg.exec()

