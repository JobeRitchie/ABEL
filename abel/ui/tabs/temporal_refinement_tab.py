"""UI for dense temporal refinement using existing active-learning models.



No model training occurs here — all models come from the active-learning

pipeline.  Overlapping window predictions are averaged per frame, then a

subtractive mutual-inhibition step penalises frames where multiple behaviors

are simultaneously likely.

"""



from __future__ import annotations



import json

from pathlib import Path

from typing import Any, Callable



import numpy as np

import pandas as pd


# Heavy library imports deferred to first use for faster tab switching.
FigureCanvas = None
NavigationToolbar = None
Figure = None
mpimg = None
_MATPLOTLIB_QT_OK: bool | None = None


def _ensure_matplotlib_tr() -> bool:
    global FigureCanvas, NavigationToolbar, Figure, mpimg, _MATPLOTLIB_QT_OK  # noqa: PLW0603
    if _MATPLOTLIB_QT_OK is not None:
        return _MATPLOTLIB_QT_OK
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qt import NavigationToolbar2QT
        from matplotlib.figure import Figure as _Fig
        import matplotlib.image as _mpimg
        FigureCanvas = FigureCanvasQTAgg
        NavigationToolbar = NavigationToolbar2QT
        Figure = _Fig
        mpimg = _mpimg
        _MATPLOTLIB_QT_OK = True
    except Exception:
        _MATPLOTLIB_QT_OK = False
    return _MATPLOTLIB_QT_OK



from PySide6.QtCore import QThreadPool, Qt

from PySide6.QtGui import QPixmap

from PySide6.QtWidgets import (

    QCheckBox,

    QComboBox,

    QDialog,

    QDialogButtonBox,

    QDoubleSpinBox,

    QFormLayout,

    QGridLayout,

    QGroupBox,

    QHBoxLayout,

    QLabel,

    QListWidget,

    QListWidgetItem,

    QMessageBox,

    QPushButton,

    QSpinBox,

    QTextEdit,

    QVBoxLayout,

    QWidget,

    QSizePolicy,

)



from abel.core.project_manager import ProjectManager

from abel.services.behavior_service import BehaviorService

from abel.services.import_service import ImportService

from abel.temporal_refinement.temporal_refinement_service import TemporalRefinementConfig

from abel.ui.suppression_helper_dialog import SuppressionHelperDialog

from abel.workers.task_worker import TaskWorker





class TemporalRefinementTab(QWidget):

    """Dense temporal refinement via existing active-learning models."""



    def __init__(self, parent: QWidget | None = None) -> None:

        super().__init__(parent)

        self._project_root: Path | None = None

        self._manager: ProjectManager | None = None

        self._imports = ImportService()

        self._behaviors = BehaviorService()

        self._pool = QThreadPool.globalInstance()

        self._active_job: str | None = None

        self._settings_by_behavior: dict[str, dict[str, Any]] = {}

        self._selected_behavior_models: dict[str, str] = {}

        self._excluded_behavior_ids: set[str] = set()

        self._suppression_matrix: dict[str, dict[str, float]] = {}

        self._selected_session_ids: set[str] | None = None

        self._current_worker: TaskWorker | None = None  # keep alive until job finishes

        self._viz_worker: TaskWorker | None = None  # keep graph-gen worker alive

        # ── Graph size settings ───────────────────────────────────
        self._tr_graph_settings: dict[str, Any] = {"max_w": 900, "max_h": 400}



        # ── Status ────────────────────────────────────────────────

        self._status = QLabel(

            "Open a project to run temporal refinement with existing active-learning models."

        )

        self._status.setWordWrap(True)



        # ── Concept (fixed to all-behavior competition) ───────────

        self._concept = QComboBox()

        self._concept.addItem(

            "All included behaviors (full-session competition)",

            userData="target_behavior",

        )

        self._concept.setEnabled(False)



        # ── Model selection ───────────────────────────────────────

        self._configure_models_btn = QPushButton("Select Behavior Models")

        self._configure_models_btn.clicked.connect(self._open_behavior_model_mapping_dialog)

        self._model_selection_summary = QLabel("Behavior models: auto")

        self._model_selection_summary.setWordWrap(True)



        # ── Inference controls ────────────────────────────────────

        self._infer_step_seconds = QDoubleSpinBox()

        self._infer_step_seconds.setRange(0.01, 2.0)

        self._infer_step_seconds.setSingleStep(0.01)

        self._infer_step_seconds.setDecimals(2)

        self._infer_step_seconds.setValue(0.10)

        _tip_step = (

            "How far to advance the sliding window between each prediction (in seconds).\n\n"

            "Smaller = more predictions per second → finer temporal resolution and smoother\n"

            "probability traces, but longer compute time.\n\n"

            "Larger = faster inference with coarser resolution. If your behavior bouts\n"

            "are at least 1–2 seconds long, 0.10–0.20 s is usually sufficient.\n\n"

            "Typical range: 0.05 – 0.50 s   |   Default: 0.10 s"

        )

        self._infer_step_seconds.setToolTip(_tip_step)



        self._infer_warmup = QDoubleSpinBox()

        self._infer_warmup.setRange(0.0, 10.0)

        self._infer_warmup.setSingleStep(0.1)

        self._infer_warmup.setDecimals(2)

        self._infer_warmup.setValue(1.50)

        _tip_warmup = (

            "Number of seconds to suppress all behavior predictions at the start of\n"

            "each session.\n\n"

            "Videos often begin mid-movement or before the animal has settled, causing\n"

            "spurious high-confidence detections in the first frames. This setting zeros\n"

            "out those predictions so they never generate bouts.\n\n"

            "Set to 0 to disable suppression entirely.\n\n"

            "Typical range: 0.0 – 3.0 s   |   Default: 1.50 s"

        )

        self._infer_warmup.setToolTip(_tip_warmup)



        # ── Mutual inhibition ────────────────────────────────────

        self._inhibition_weight = QDoubleSpinBox()

        self._inhibition_weight.setRange(0.0, 0.50)

        self._inhibition_weight.setSingleStep(0.01)

        self._inhibition_weight.setDecimals(2)

        self._inhibition_weight.setValue(0.20)

        _tip_inhibition = (

            "Subtractive mutual inhibition between competing behaviors.\n\n"

            "For each behavior, its probability is reduced by:\n"

            "   weight × (sum of all other behaviors' probabilities)\n\n"

            "This penalises frames where multiple behaviors fire simultaneously,\n"

            "helping the dominant behavior stand out without distorting its absolute\n"

            "probability when it is the only behavior present.\n\n"

            "Higher values → stronger suppression of ambiguous frames.\n"

            "Lower values → behaviors compete less; useful when behaviors are distinct.\n"

            "0.0 → no inhibition applied at all.\n\n"

            "Keep modest (0.10 – 0.25) because many behaviors share postural similarity.\n\n"

            "Typical range: 0.0 – 0.30   |   Default: 0.20"

        )

        self._inhibition_weight.setToolTip(_tip_inhibition)



        # ── Probability temperature ──────────────────────────────

        self._probability_temperature = QDoubleSpinBox()

        self._probability_temperature.setRange(0.1, 10.0)

        self._probability_temperature.setSingleStep(0.1)

        self._probability_temperature.setDecimals(2)

        self._probability_temperature.setValue(1.0)

        _tip_temp = (

            "Probability calibration temperature applied after scoring.\n\n"

            "Rescales each raw probability through its log-odds:\n"

            "   scaled_prob = sigmoid(logit(p) / T)\n\n"

            "T = 1.0  → no change (default)\n"

            "T > 1.0  → softens overconfident predictions toward 0.5. Use this\n"

            "           when the model produces many near-1.0 false positives.\n"

            "           Try 1.5 – 3.0 as a starting point.\n"

            "T < 1.0  → sharpens predictions, pushing probabilities further toward\n"

            "           0 or 1. Use cautiously — can amplify noise.\n\n"

            "This affects probability traces and bout detection thresholds.\n\n"

            "Typical range: 1.0 – 2.0   |   Default: 1.0"

        )

        self._probability_temperature.setToolTip(_tip_temp)



        # ── Parallel inference ────────────────────────────────────

        self._parallel_enabled = QCheckBox("Parallel inference")

        self._parallel_enabled.setChecked(True)

        self._parallel_enabled.setToolTip(

            "Run inference on multiple sessions in parallel using threads. "

            "Disable if you encounter memory pressure."

        )



        # ── Test mode ─────────────────────────────────────────────

        self._test_single_session = QCheckBox("Test mode (1 session)")

        self._test_single_session.setToolTip(

            "When checked, inference runs on only the first available session. "

            "Use to quickly verify settings before running the full dataset."

        )



        # ── Session scope ─────────────────────────────────────────

        self._select_sessions_btn = QPushButton("Choose Sessions…")

        self._select_sessions_btn.clicked.connect(self._open_session_selection_dialog)

        self._session_scope_summary = QLabel("Session scope: all linked sessions")

        self._session_scope_summary.setWordWrap(True)



        # ── Action buttons ────────────────────────────────────────

        self._infer_btn = QPushButton("Run Inference")

        self._refine_btn = QPushButton("Generate Bouts")

        self._refresh_results_btn = QPushButton("Refresh Results View")

        self._clear_cache_btn = QPushButton("Clear Temporal Cache")



        self._infer_btn.clicked.connect(self._run_infer)

        self._refine_btn.clicked.connect(self._run_refine)

        self._refresh_results_btn.clicked.connect(self._refresh_results_view)

        self._clear_cache_btn.clicked.connect(self._clear_temporal_cache)



        # ── Log ───────────────────────────────────────────────────

        self._log = QTextEdit()

        self._log.setReadOnly(True)

        self._log.setMinimumHeight(70)

        self._log.setMaximumHeight(100)



        self._artifact_summary = QTextEdit()

        self._artifact_summary.setReadOnly(True)

        self._artifact_summary.setMaximumHeight(100)



        # ── Visualization ─────────────────────────────────────────

        self._viz_title = QLabel("Temporal precision preview")

        self._viz_selector = QComboBox()

        self._viz_selector.addItem("Auto", userData="auto")

        self._viz_selector.addItem("Winning Behavior Timeline", userData="timeline")

        self._viz_selector.addItem("Behavior Probability Traces", userData="traces")

        self._viz_selector.addItem("Behavior Probability Heatmap", userData="heatmap")

        # Default to the heatmap view.
        self._viz_selector.setCurrentIndex(self._viz_selector.findData("heatmap"))

        self._viz_selector.currentIndexChanged.connect(self._on_viz_selection_changed)

        self._viz_session_selector = QComboBox()

        self._viz_session_selector.setMinimumWidth(160)

        self._viz_session_selector.setToolTip(

            "Select which session to display in the graph."

        )

        self._viz_session_selector.currentIndexChanged.connect(self._on_viz_selection_changed)

        self._viz_trace_paths: dict[str, str] = {}

        self._viz_preview = QLabel("Run inference to populate this graph area.")

        self._viz_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._viz_preview.setMinimumHeight(220)

        self._viz_preview.setMaximumWidth(
            int(self._tr_graph_settings.get("max_w", 900)),
        )

        self._viz_preview.setStyleSheet(

            "border: 1px solid #cccccc; background: #f7f7f7;"

        )

        self._viz_pixmap_original: QPixmap | None = None

        self._viz_figure: Any = None

        self._viz_canvas: Any = None

        self._viz_toolbar: Any = None

        if _ensure_matplotlib_tr() and Figure is not None and FigureCanvas is not None and NavigationToolbar is not None:

            self._viz_figure = Figure(figsize=(10.5, 4.0), tight_layout=True)

            self._viz_canvas = FigureCanvas(self._viz_figure)

            self._viz_canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
            )

            self._viz_canvas.setFixedSize(
                int(self._tr_graph_settings.get("max_w", 900)),
                int(self._tr_graph_settings.get("max_h", 400)),
            )

            self._viz_toolbar = NavigationToolbar(self._viz_canvas, self)

            self._viz_toolbar.setVisible(False)

            self._viz_canvas.setVisible(False)



        # ── Layout ────────────────────────────────────────────────

        config_group = QGroupBox("Temporal Refinement (Dense Inference)")

        config_layout = QVBoxLayout(config_group)



        top_row = QHBoxLayout()

        top_row.addWidget(QLabel("Scope:"))

        top_row.addWidget(self._concept, 1)

        config_layout.addLayout(top_row)



        model_row = QHBoxLayout()

        model_row.addWidget(self._configure_models_btn)

        model_row.addWidget(self._model_selection_summary, 1)

        config_layout.addLayout(model_row)



        session_row = QHBoxLayout()

        session_row.addWidget(self._select_sessions_btn)

        session_row.addWidget(self._session_scope_summary, 1)

        config_layout.addLayout(session_row)



        # Numeric parameters laid out as a 2-column grid so they use the available
        # horizontal space instead of being squished into a single narrow column.
        self._suppression_btn = QPushButton("Configure Suppression…")
        self._suppression_btn.setToolTip(
            "Open the suppression helper to set per-behavior suppression weights "
            "and preview the effect on synthetic waveforms."
        )
        self._suppression_btn.clicked.connect(self._open_suppression_helper)

        # Give the spin boxes real breathing room (the global stylesheet sets no
        # min-height, so without this the digits clip vertically).
        for _spin in (
            self._infer_step_seconds,
            self._infer_warmup,
            self._inhibition_weight,
            self._probability_temperature,
        ):
            _spin.setMinimumHeight(30)
            _spin.setMinimumWidth(110)
            _spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._suppression_btn.setMinimumHeight(30)

        # Grid columns: [label1][field1][gap][label2][field2][gap]
        params_grid = QGridLayout()
        params_grid.setHorizontalSpacing(12)
        params_grid.setVerticalSpacing(10)
        params_grid.setContentsMargins(0, 4, 0, 4)
        _LBL = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        def _grid_field(row: int, pair: int, text: str, widget, tip: str) -> None:
            base = pair * 3  # each pair occupies 3 columns: label, field, gap
            lbl = QLabel(text)
            lbl.setToolTip(tip)
            widget.setToolTip(tip)
            params_grid.addWidget(lbl, row, base, _LBL)
            params_grid.addWidget(widget, row, base + 1)

        _grid_field(0, 0, "Inference step (seconds):", self._infer_step_seconds, _tip_step)
        _grid_field(0, 1, "Warmup suppress (seconds):", self._infer_warmup, _tip_warmup)
        _grid_field(1, 0, "Inhibition weight:", self._inhibition_weight, _tip_inhibition)
        _grid_field(1, 1, "Probability temperature:", self._probability_temperature, _tip_temp)

        # Suppression helper sits under the inhibition field; parallel toggle under
        # the temperature field.
        params_grid.addWidget(self._suppression_btn, 2, 1)
        params_grid.addWidget(self._parallel_enabled, 2, 4)

        # Spacer columns (2 and 5) absorb the extra width so the two pairs spread
        # out instead of crowding the left edge, and fields keep their natural size.
        params_grid.setColumnStretch(2, 1)
        params_grid.setColumnStretch(5, 1)

        config_layout.addLayout(params_grid)



        self._config_summary = QLabel("")

        self._config_summary.setWordWrap(True)

        config_layout.addWidget(self._config_summary)



        self._tips = QLabel(

            "Uses existing active-learning models to score every frame with dense "

            "overlapping windows.  Overlapping predictions are averaged, then "

            "subtractive mutual inhibition reduces ambiguous frames.\n"

            "Bout thresholding and review are handled in the Temporal Review tab."

        )

        self._tips.setWordWrap(True)

        self._tips.setStyleSheet("color: #616161; font-size: 11px;")

        config_layout.addWidget(self._tips)



        btn_row = QHBoxLayout()

        btn_row.addWidget(self._infer_btn)

        btn_row.addWidget(self._refine_btn)

        btn_row.addWidget(self._test_single_session)

        btn_row.addWidget(self._refresh_results_btn)

        btn_row.addWidget(self._clear_cache_btn)

        btn_row.addStretch(1)



        root = QVBoxLayout(self)

        root.addWidget(config_group)

        root.addLayout(btn_row)

        root.addWidget(self._status)

        root.addWidget(QLabel("Status / artifacts"))

        root.addWidget(self._log, 1)

        root.addWidget(self._artifact_summary)

        viz_head = QHBoxLayout()

        viz_head.addWidget(self._viz_title)

        viz_head.addStretch(1)

        viz_head.addWidget(QLabel("Session:"))

        viz_head.addWidget(self._viz_session_selector)

        viz_head.addWidget(QLabel("Graph:"))

        viz_head.addWidget(self._viz_selector)

        _tr_graph_size_btn = QPushButton("Graph Size\u2026")
        _tr_graph_size_btn.setToolTip("Set maximum display width and height for the temporal refinement graph.")
        _tr_graph_size_btn.clicked.connect(self._open_tr_graph_size_dialog)
        viz_head.addWidget(_tr_graph_size_btn)

        root.addLayout(viz_head)

        if self._viz_toolbar is not None:

            root.addWidget(self._viz_toolbar)

        if self._viz_canvas is not None:

            root.addWidget(self._viz_canvas)

        root.addWidget(self._viz_preview)



    # ==================================================================

    # Project lifecycle

    # ==================================================================



    def set_project(self, project_root: Path) -> None:

        self._project_root = project_root

        self._manager = ProjectManager(project_root)

        self._status.setText(f"Loading project\u2026")

        # Defer all I/O so the tab switch paint completes immediately.
        from PySide6.QtCore import QTimer  # noqa: PLC0415

        QTimer.singleShot(0, self._deferred_project_init)

    def _open_tr_graph_size_dialog(self) -> None:
        """Open a dialog to set max display width/height for the temporal refinement graph."""
        from PySide6.QtWidgets import (  # noqa: PLC0415
            QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QVBoxLayout,
        )
        gs = self._tr_graph_settings
        dlg = QDialog(self)
        dlg.setWindowTitle("Graph Size")
        dlg.resize(300, 130)
        form = QFormLayout()

        max_w_spin = QSpinBox(dlg)
        max_w_spin.setRange(200, 3840)
        max_w_spin.setSingleStep(50)
        max_w_spin.setSuffix(" px")
        max_w_spin.setToolTip("Maximum display width of the graph/preview area in pixels.")
        max_w_spin.setValue(int(gs.get("max_w", 900)))

        max_h_spin = QSpinBox(dlg)
        max_h_spin.setRange(150, 2160)
        max_h_spin.setSingleStep(50)
        max_h_spin.setSuffix(" px")
        max_h_spin.setToolTip("Maximum display height of the graph/preview area in pixels.")
        max_h_spin.setValue(int(gs.get("max_h", 400)))

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
        if self._viz_canvas is not None:
            self._viz_canvas.setFixedSize(gs["max_w"], gs["max_h"])
            if self._viz_figure is not None:
                self._viz_figure.set_size_inches(gs["max_w"] / 100, gs["max_h"] / 100)
                self._viz_canvas.draw_idle()
        if self._viz_preview is not None:
            self._viz_preview.setMaximumWidth(gs["max_w"])

    def _deferred_project_init(self) -> None:
        if self._project_root is None:
            return
        self._behaviors.set_project(self._project_root)

        self._settings_by_behavior = self._load_settings()

        self._apply_settings_to_widgets(

            self._settings_for_behavior("target_behavior")

        )

        self._refresh_config_summary()

        self._refresh_session_scope_summary()

        self._status.setText(f"Project ready: {self._project_root}")

        self._refresh_results_view()

        self._refresh_visualization_preview()

        # Regenerate preview graphs asynchronously on project load so
        # pre-existing inference artifacts are displayed without blocking
        # the GUI thread.
        self._generate_preview_graphs_async()


    # ==================================================================

    # Session scope

    # ==================================================================



    def _session_display_map(self) -> dict[str, str]:

        """Map session_id -> human-readable subject name (falling back to a short code).

        Sessions are shown by their subject name; if a subject appears more than
        once, the short session code is appended to keep entries unique.
        """

        if self._project_root is None:

            return {}

        manifest = self._imports.load_manifest(self._project_root)

        if manifest is None:

            return {}

        video_by_id = {v.asset_id: v for v in manifest.videos}

        subject_by_sid: dict[str, str] = {}

        for s in manifest.linked_sessions:

            sid = str(s.session_id)

            subject = str(s.subject_id or "").strip()

            if not subject:

                video = video_by_id.get(s.video_asset_id)

                subject = str(video.subject_id or "").strip() if video else ""

            subject_by_sid[sid] = subject

        subject_counts: dict[str, int] = {}

        for subject in subject_by_sid.values():

            if subject:

                subject_counts[subject] = subject_counts.get(subject, 0) + 1

        out: dict[str, str] = {}

        for sid, subject in subject_by_sid.items():

            short = sid[8:16] if sid.startswith("session_") and len(sid) > 8 else sid

            if not subject:

                out[sid] = short

            elif subject_counts.get(subject, 0) > 1:

                out[sid] = f"{subject} ({short})"

            else:

                out[sid] = subject

        return out



    def _session_options_from_manifest(self) -> list[tuple[str, str, str]]:

        if self._project_root is None:

            return []

        manifest = self._imports.load_manifest(self._project_root)

        if manifest is None or not manifest.linked_sessions:

            return []

        rows: list[tuple[str, str, str]] = []

        for linked in manifest.linked_sessions:

            sid = str(linked.session_id)

            subject = str(linked.subject_id or "")

            label = f"{sid}"

            if subject:

                label += f"  |  subject: {subject}"

            rows.append((sid, subject, label))

        rows.sort(key=lambda x: (x[1], x[0]))

        return rows



    def _refresh_session_scope_summary(self) -> None:

        options = self._session_options_from_manifest()

        total = len(options)

        if total <= 0:

            self._session_scope_summary.setText("Session scope: no linked sessions")

            self._select_sessions_btn.setEnabled(False)

            return

        self._select_sessions_btn.setEnabled(True)

        all_ids = {sid for sid, _subj, _label in options}

        if not self._selected_session_ids:

            self._session_scope_summary.setText(

                f"Session scope: all linked sessions ({total})"

            )

            return

        valid_selected = {

            sid for sid in self._selected_session_ids if sid in all_ids

        }

        if valid_selected != self._selected_session_ids:

            self._selected_session_ids = valid_selected or None

        selected_n = (

            len(self._selected_session_ids) if self._selected_session_ids else 0

        )

        if selected_n <= 0:

            self._session_scope_summary.setText(

                "Session scope: all linked sessions (none explicitly selected)"

            )

            self._selected_session_ids = None

            return

        self._session_scope_summary.setText(

            f"Session scope: {selected_n}/{total} session(s) selected"

        )



    def _open_session_selection_dialog(self) -> None:

        if self._project_root is None:

            QMessageBox.warning(self, "Temporal Refinement", "Open a project first.")

            return

        options = self._session_options_from_manifest()

        if not options:

            QMessageBox.information(

                self, "Temporal Refinement", "No linked sessions available."

            )

            return

        all_ids = {sid for sid, _subj, _label in options}

        current = (

            set(self._selected_session_ids)

            if self._selected_session_ids

            else set(all_ids)

        )



        dlg = QDialog(self)

        dlg.setWindowTitle("Select Sessions For Temporal Refinement")

        dlg.resize(560, 640)

        info = QLabel(

            "Only selected sessions will be processed during inference and postprocessing.",

            dlg,

        )

        info.setWordWrap(True)

        list_widget = QListWidget(dlg)

        for sid, _subject, label in options:

            item = QListWidgetItem(label)

            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

            item.setData(Qt.ItemDataRole.UserRole, sid)

            item.setCheckState(

                Qt.CheckState.Checked if sid in current else Qt.CheckState.Unchecked

            )

            list_widget.addItem(item)



        select_all_btn = QPushButton("Select All", dlg)

        deselect_all_btn = QPushButton("Deselect All", dlg)



        def _set_all(state: Qt.CheckState) -> None:

            for i in range(list_widget.count()):

                list_widget.item(i).setCheckState(state)



        select_all_btn.clicked.connect(lambda: _set_all(Qt.CheckState.Checked))

        deselect_all_btn.clicked.connect(lambda: _set_all(Qt.CheckState.Unchecked))

        buttons = QDialogButtonBox(

            QDialogButtonBox.StandardButton.Ok

            | QDialogButtonBox.StandardButton.Cancel,

            parent=dlg,

        )

        buttons.accepted.connect(dlg.accept)

        buttons.rejected.connect(dlg.reject)

        top_btn_row = QHBoxLayout()

        top_btn_row.addWidget(select_all_btn)

        top_btn_row.addWidget(deselect_all_btn)

        top_btn_row.addStretch(1)

        layout = QVBoxLayout(dlg)

        layout.addWidget(info)

        layout.addLayout(top_btn_row)

        layout.addWidget(list_widget, 1)

        layout.addWidget(buttons)



        if dlg.exec() != int(QDialog.DialogCode.Accepted):

            return

        selected_ids: list[str] = []

        for i in range(list_widget.count()):

            item = list_widget.item(i)

            if item.checkState() == Qt.CheckState.Checked:

                sid = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()

                if sid:

                    selected_ids.append(sid)

        if not selected_ids:

            QMessageBox.warning(

                self, "Temporal Refinement", "Select at least one session."

            )

            return

        if set(selected_ids) == all_ids:

            self._selected_session_ids = None

        else:

            self._selected_session_ids = set(selected_ids)

        self._refresh_session_scope_summary()

        self._status.setText(

            f"Session scope updated: {len(selected_ids)} session(s) selected."

        )



    # ==================================================================

    # Settings persistence

    # ==================================================================



    def _settings_path(self) -> Path | None:

        if self._project_root is None:

            return None

        return self._project_root / "config" / "temporal_refinement_settings.json"



    @staticmethod

    def _default_settings() -> dict[str, Any]:

        return {

            "inference_step_seconds": 0.10,

            "inference_warmup_seconds": 1.50,

            "inhibition_weight": 0.20,

            "suppression_matrix": {},

            "probability_temperature": 1.0,

            "inference_parallel_enabled": True,

            "selected_behavior_models": {},

            "excluded_behavior_ids": [],

        }



    def _load_settings(self) -> dict[str, dict[str, Any]]:

        path = self._settings_path()

        default = self._default_settings()

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

            "__all__": dict(

                self._settings_by_behavior.get("__all__", self._default_settings())

            ),

            "by_behavior": {

                key: value

                for key, value in self._settings_by_behavior.items()

                if key != "__all__"

            },

        }

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



    def _current_behavior_id(self) -> str:

        return str(self._concept.currentData() or "target_behavior").strip() or "target_behavior"



    def _settings_for_behavior(self, behavior_id: str) -> dict[str, Any]:

        global_defaults = dict(

            self._settings_by_behavior.get("__all__", self._default_settings())

        )

        return {**global_defaults, **dict(self._settings_by_behavior.get(behavior_id, {}) or {})}



    def _apply_settings_to_widgets(self, settings: dict[str, Any]) -> None:

        self._infer_step_seconds.setValue(

            float(settings.get("inference_step_seconds", 0.10))

        )

        self._infer_warmup.setValue(

            float(settings.get("inference_warmup_seconds", 1.50))

        )

        self._inhibition_weight.setValue(

            float(settings.get("inhibition_weight", 0.20))

        )

        self._probability_temperature.setValue(

            float(settings.get("probability_temperature", 1.0))

        )

        self._parallel_enabled.setChecked(

            bool(settings.get("inference_parallel_enabled", True))

        )

        self._selected_behavior_models = {

            str(k): str(v)

            for k, v in dict(settings.get("selected_behavior_models", {}) or {}).items()

            if str(v).strip()

        }

        self._excluded_behavior_ids = {

            str(v).strip()

            for v in list(settings.get("excluded_behavior_ids", []) or [])

            if str(v).strip()

        }

        raw_matrix = settings.get("suppression_matrix", {}) or {}

        self._suppression_matrix = {

            str(k): {str(k2): float(v2) for k2, v2 in dict(v).items()}

            for k, v in dict(raw_matrix).items()

            if isinstance(v, dict)

        }



    def _collect_widget_settings(self) -> dict[str, Any]:

        return {

            "inference_step_seconds": float(self._infer_step_seconds.value()),

            "inference_warmup_seconds": float(self._infer_warmup.value()),

            "inhibition_weight": float(self._inhibition_weight.value()),

            "suppression_matrix": dict(self._suppression_matrix),

            "probability_temperature": float(self._probability_temperature.value()),

            "inference_parallel_enabled": bool(self._parallel_enabled.isChecked()),

            "selected_behavior_models": dict(self._selected_behavior_models),

            "excluded_behavior_ids": sorted(self._excluded_behavior_ids),

        }



    def _persist_settings(self) -> None:

        behavior_id = self._current_behavior_id()

        self._settings_by_behavior[behavior_id] = self._collect_widget_settings()

        self._save_settings()



    def _refresh_config_summary(self) -> None:

        n_pairs = sum(len(v) for v in self._suppression_matrix.values())

        inhib_text = (

            f"suppression matrix ({n_pairs} pairs)"

            if n_pairs > 0

            else f"inhibition {self._inhibition_weight.value():.2f}"

        )

        parts = [

            f"step {self._infer_step_seconds.value():.2f}s",

            f"warmup {self._infer_warmup.value():.2f}s",

            inhib_text,

            f"temperature {self._probability_temperature.value():.2f}",

        ]

        self._config_summary.setText(" | ".join(parts))

        self._model_selection_summary.setText(

            self._behavior_model_selection_summary_text()

        )



    # ==================================================================

    # Model mapping dialog

    # ==================================================================



    def _available_behavior_models(self) -> list[str]:

        if self._project_root is None:

            return []

        models_root = self._project_root / "derived" / "models"

        if not models_root.exists():

            return []

        rows: list[Path] = []

        for p in models_root.iterdir():

            if not p.is_dir():

                continue

            if not (p / "model_state.pkl").exists():

                continue

            if not p.name.startswith("behavior_model_"):

                continue

            rows.append(p)

        rows.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return [p.name for p in rows]



    def _behavior_model_selection_summary_text(self) -> str:

        if not self._selected_behavior_models and not self._excluded_behavior_ids:

            return "Behavior models: auto"

        shown = sorted(

            (k, v)

            for k, v in self._selected_behavior_models.items()

            if k not in self._excluded_behavior_ids

        )

        parts = [f"{bid} -> {model}" for bid, model in shown[:3]]

        excluded = sorted(self._excluded_behavior_ids)

        if excluded:

            preview = ", ".join(excluded[:2]) + ("..." if len(excluded) > 2 else "")

            parts.append(f"excluded: {preview}")

        if len(shown) > 3:

            parts.append(f"+{len(shown) - 3} more")

        return "Behavior models: " + "; ".join(parts)



    def _open_suppression_helper(self) -> None:

        names: list[str] = []

        ids: list[str] = []

        seen: set[str] = set()

        for b in self._behaviors.behaviors:

            bid = str(b.behavior_id)

            if bid in self._excluded_behavior_ids:

                continue

            names.append(b.name)

            ids.append(bid)

            seen.add(bid)

        if "no_behavior" not in seen and "no_behavior" not in self._excluded_behavior_ids:

            names.append("No Behavior")

            ids.append("no_behavior")

        if len(names) < 2:

            QMessageBox.information(

                self,

                "Suppression Helper",

                "At least two active (non-excluded) behaviors are needed to "

                "configure suppression relationships.",

            )

            return

        dlg = SuppressionHelperDialog(

            behavior_names=names,

            behavior_ids=ids,

            initial_matrix=dict(self._suppression_matrix),

            initial_temperature=float(self._probability_temperature.value()),

            global_inhibition=float(self._inhibition_weight.value()),

            parent=self,

        )

        if dlg.exec() != int(QDialog.DialogCode.Accepted):

            return

        self._suppression_matrix = dlg.suppression_matrix()

        self._persist_settings()

        n_pairs = sum(len(v) for v in self._suppression_matrix.values())

        self._status.setText(

            f"Suppression matrix updated: {n_pairs} directed pair(s) configured."

        )

        self._refresh_config_summary()



    def _open_behavior_model_mapping_dialog(self) -> None:

        versions = self._available_behavior_models()

        if not versions:

            QMessageBox.information(

                self,

                "Temporal Refinement",

                "No behavior models found. Train at least one model in the "

                "active-learning pipeline first.",

            )

            return

        # Build a map of behavior_id → best (latest) model version by reading
        # run_settings.json from each model directory.
        auto_resolved: dict[str, str] = {}
        if self._project_root is not None:
            models_root = self._project_root / "derived" / "models"
            if models_root.exists():
                from abel.storage.file_store import read_json as _rj
                # Iterate newest-first so first match per behavior wins.
                for version in versions:
                    model_dir = models_root / version
                    settings = _rj(model_dir / "run_settings.json", {})
                    tb = str(settings.get("target_behavior", "")).strip()
                    if tb and tb not in auto_resolved:
                        auto_resolved[tb] = version

        dlg = QDialog(self)

        dlg.setWindowTitle("Select Behavior Models")

        form = QFormLayout(dlg)

        selectors: dict[str, QComboBox] = {}



        # Collect behaviors including a virtual "No Behavior" entry so
        # the user can assign a no-behavior model to the competition.
        _entries: list[tuple[str, str]] = []  # (bid, display_name)
        _seen_bids: set[str] = set()
        for behavior in self._behaviors.behaviors:
            bid = str(behavior.behavior_id)
            _entries.append((bid, behavior.name))
            _seen_bids.add(bid)
        if "no_behavior" not in _seen_bids:
            _entries.append(("no_behavior", "No Behavior"))

        for bid, display_name in _entries:

            combo = QComboBox(dlg)

            combo.addItem("Auto (latest for this behavior)", userData="")

            combo.addItem("Exclude from competition", userData="__exclude__")

            for version in versions:

                combo.addItem(version, userData=version)

            current = (

                "__exclude__"

                if bid in self._excluded_behavior_ids

                else str(self._selected_behavior_models.get(bid, "") or "")

            )

            # Auto-resolve: if no explicit selection yet, pre-select best model.
            if not current and bid in auto_resolved:
                current = auto_resolved[bid]

            idx = combo.findData(current)

            combo.setCurrentIndex(idx if idx >= 0 else 0)

            selectors[bid] = combo

            form.addRow(f"{display_name} ({bid}):", combo)

        # Auto-assign button to populate all combos with best detected model.
        def _auto_assign_all() -> None:
            for bid, combo in selectors.items():
                if bid in auto_resolved:
                    idx = combo.findData(auto_resolved[bid])
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

        auto_btn = QPushButton("Auto-Assign All")
        auto_btn.setToolTip(
            "Automatically assign the latest trained model to each behavior "
            "based on the target_behavior recorded in each model's run settings."
        )
        auto_btn.clicked.connect(_auto_assign_all)
        form.addRow(auto_btn)

        buttons = QDialogButtonBox(

            QDialogButtonBox.StandardButton.Ok

            | QDialogButtonBox.StandardButton.Cancel,

            dlg,

        )

        form.addRow(buttons)

        buttons.accepted.connect(dlg.accept)

        buttons.rejected.connect(dlg.reject)



        if dlg.exec() != int(QDialog.DialogCode.Accepted):

            return

        updated: dict[str, str] = {}

        excluded: set[str] = set()

        for bid, combo in selectors.items():

            chosen = str(combo.currentData() or "").strip()

            if chosen == "__exclude__":

                excluded.add(bid)

                continue

            if chosen:

                updated[bid] = chosen

        self._selected_behavior_models = updated

        self._excluded_behavior_ids = excluded

        self._persist_settings()

        self._refresh_config_summary()

        self._status.setText("Updated behavior model mapping.")



    # ==================================================================

    # Config construction

    # ==================================================================



    def _config(self) -> TemporalRefinementConfig:

        return TemporalRefinementConfig(

            selected_behavior_models=dict(self._selected_behavior_models),

            excluded_behavior_ids=sorted(self._excluded_behavior_ids),

            inference_step_seconds=float(self._infer_step_seconds.value()),

            inference_warmup_seconds=float(self._infer_warmup.value()),

            inference_parallel_enabled=bool(self._parallel_enabled.isChecked()),

            inhibition_weight=float(self._inhibition_weight.value()),

            suppression_matrix=dict(self._suppression_matrix),

            probability_temperature=float(self._probability_temperature.value()),

        )



    # ==================================================================

    # Busy state

    # ==================================================================



    def _set_busy(self, busy: bool) -> None:

        self._infer_btn.setEnabled(not busy)

        self._refine_btn.setEnabled(not busy)

        self._refresh_results_btn.setEnabled(not busy)

        self._clear_cache_btn.setEnabled(not busy)



    # ==================================================================

    # Actions

    # ==================================================================



    def _run_infer(self) -> None:

        if self._manager is None:

            QMessageBox.warning(self, "Temporal Refinement", "Open a project first.")

            return

        self._persist_settings()

        self._refresh_config_summary()

        self._set_busy(True)

        self._active_job = "infer"

        test_mode = self._test_single_session.isChecked()

        label = " (test: 1 session)" if test_mode else ""

        self._status.setText(f"Running dense temporal inference{label}...")

        self._append_log(f"Starting dense inference{label}...")

        # Capture all widget-derived values on the GUI thread so that the
        # worker never reads Qt widgets from a background thread (which can
        # deadlock on Windows).
        _concept_id = self._current_behavior_id()
        _config = self._config()
        _sessions = (
            sorted(self._selected_session_ids) if self._selected_session_ids else None
        )

        worker = TaskWorker(self._infer_task)

        worker.kwargs["progress_cb"] = worker.signals.line_emitted.emit

        worker.kwargs["max_sessions"] = 1 if test_mode else None

        worker.kwargs["concept_id"] = _concept_id

        worker.kwargs["config"] = _config

        worker.kwargs["sessions"] = _sessions

        worker.signals.line_emitted.connect(self._on_progress_line)

        worker.signals.finished.connect(

            lambda result: self._on_finished("Inference completed.", result)

        )

        worker.signals.failed.connect(self._on_failed)

        self._current_worker = worker  # prevents GC before signals fire

        self._pool.start(worker)



    def _run_refine(self) -> None:

        if self._manager is None:

            QMessageBox.warning(self, "Temporal Refinement", "Open a project first.")

            return

        self._persist_settings()

        self._refresh_config_summary()

        self._set_busy(True)

        self._active_job = "refine"

        self._status.setText("Generating bout calls from frame probabilities...")

        self._append_log("Starting bout postprocess...")

        _concept_id = self._current_behavior_id()
        _config = self._config()
        _sessions = (
            sorted(self._selected_session_ids) if self._selected_session_ids else None
        )

        worker = TaskWorker(self._refine_task)

        worker.kwargs["progress_cb"] = worker.signals.line_emitted.emit

        worker.kwargs["concept_id"] = _concept_id

        worker.kwargs["config"] = _config

        worker.kwargs["sessions"] = _sessions

        worker.signals.line_emitted.connect(self._on_progress_line)

        worker.signals.finished.connect(

            lambda result: self._on_finished("Bout extraction completed.", result)

        )

        worker.signals.failed.connect(self._on_failed)

        self._current_worker = worker  # prevents GC before signals fire

        self._pool.start(worker)



    def _infer_task(

        self,

        progress_cb: Callable[[str], None] | None = None,

        max_sessions: int | None = None,

        concept_id: str | None = None,

        config: "TemporalRefinementConfig | None" = None,

        sessions: list[str] | None = None,

    ) -> dict[str, Any]:

        manager = self._require_manager()

        return manager.run_temporal_refinement_inference(

            concept_id=concept_id or self._current_behavior_id(),

            sessions=sessions,

            config=config or self._config(),

            mode="dense",

            max_sessions=max_sessions,

            progress_cb=progress_cb,

        )



    def _refine_task(

        self,

        progress_cb: Callable[[str], None] | None = None,

        concept_id: str | None = None,

        config: "TemporalRefinementConfig | None" = None,

        sessions: list[str] | None = None,

    ) -> dict[str, Any]:

        manager = self._require_manager()

        return manager.run_temporal_refinement_postprocess(

            concept_id=concept_id or self._current_behavior_id(),

            sessions=sessions,

            config=config or self._config(),

            progress_cb=progress_cb,

        )



    def _clear_temporal_cache(self) -> None:

        manager = self._manager

        if manager is None:

            QMessageBox.warning(self, "Temporal Refinement", "Open a project first.")

            return

        answer = QMessageBox.question(

            self,

            "Clear Temporal Cache",

            "Clear all temporal refinement cache and inference artifacts "

            "for the current behavior scope?\n\n"

            "This forces the next run to recompute inference outputs.",

            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,

            QMessageBox.StandardButton.No,

        )

        if answer != QMessageBox.StandardButton.Yes:

            return

        result = manager.clear_temporal_refinement_cache(

            concept_id=self._current_behavior_id(),

            clear_run_artifacts=True,

        )

        self._append_log(

            f"Cache cleared: files={result.get('removed_files', 0)}, "

            f"dirs={result.get('removed_dirs', 0)}, "

            f"run_dirs={result.get('removed_run_dirs', 0)}"

        )

        self._status.setText("Temporal cache cleared.")



    # ==================================================================

    # Callbacks

    # ==================================================================



    def _append_log(self, text: str) -> None:

        self._log.append(text)



    def _require_manager(self) -> ProjectManager:

        if self._manager is None:

            raise ValueError("No project loaded")

        return self._manager



    def _on_progress_line(self, text: str) -> None:

        self._append_log(text)



    def _on_finished(self, status_text: str, result: dict[str, Any]) -> None:

        self._set_busy(False)

        self._active_job = None

        self._current_worker = None

        self._status.setText(status_text)

        self._append_log(status_text)

        if isinstance(result, dict):

            for key in ("status", "inference_dir", "postprocess_dir", "parameter_hash", "reason"):

                value = result.get(key)

                if value not in {None, ""}:

                    self._append_log(f"  {key}: {value}")

        self._refresh_results_view()

        # Generate preview graphs in the background so the main thread stays
        # responsive.  _refresh_visualization_preview calls
        # _ensure_competition_preview_graphs which does heavy matplotlib
        # rendering + parquet I/O — doing that on the GUI thread was causing
        # "not responding" on Windows.
        self._generate_preview_graphs_async()



    def _on_failed(self, traceback_text: str) -> None:

        self._set_busy(False)

        self._active_job = None

        self._current_worker = None

        self._status.setText("Temporal refinement failed.")

        self._append_log("ERROR: temporal refinement failed.")

        self._append_log(traceback_text)

        QMessageBox.critical(

            self,

            "Temporal Refinement",

            "Temporal refinement failed. Check logs for details.",

        )

    def _generate_preview_graphs_async(self) -> None:
        """Generate matplotlib preview graphs in a background thread, then
        refresh the visualization widget on the GUI thread when done."""
        latest_path = self._latest_path()
        if latest_path is None or not latest_path.exists():
            self._refresh_visualization_preview()
            return
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            self._refresh_visualization_preview()
            return
        inference_dir = Path(str(latest.get("inference_dir", "") or "").strip())
        if not str(inference_dir) or not inference_dir.exists():
            self._refresh_visualization_preview()
            return

        def _gen_graphs() -> None:
            self._ensure_competition_preview_graphs(inference_dir)

        def _on_done(_: Any) -> None:
            self._viz_worker = None
            self._refresh_visualization_preview()

        worker = TaskWorker(_gen_graphs)
        worker.signals.finished.connect(_on_done)
        worker.signals.failed.connect(_on_done)
        self._viz_worker = worker  # prevent GC before signals fire
        self._pool.start(worker)



    # ==================================================================

    # Results & visualization

    # ==================================================================



    @staticmethod

    def _safe_name(value: str) -> str:

        safe = "".join(

            ch if ch.isalnum() or ch in {"_", "-"} else "_"

            for ch in str(value).strip()

        )

        return safe or "target_behavior"



    def _behavior_label(self, token: str) -> str:

        raw = str(token or "").strip()

        if not raw:

            return raw

        low = raw.lower()

        if low == "no_behavior":

            return "No Behavior"

        if low == "target_behavior":

            return "Target Behavior"

        for behavior in self._behaviors.behaviors:

            bid = str(behavior.behavior_id or "").strip()

            name = str(behavior.name or "").strip() or bid

            if raw == bid or raw == self._safe_name(bid):

                return name

        return raw



    def _latest_path(self) -> Path | None:

        if self._project_root is None:

            return None

        concept = self._current_behavior_id()

        return (

            self._project_root

            / "derived"

            / "temporal_refinement"

            / self._safe_name(concept)

            / "latest.json"

        )



    def _refresh_results_view(self) -> None:

        latest_path = self._latest_path()

        self._artifact_summary.clear()

        if latest_path is None or not latest_path.exists():

            self._artifact_summary.setPlainText(

                "No temporal refinement artifacts found yet."

            )

            return

        try:

            latest = json.loads(latest_path.read_text(encoding="utf-8"))

        except Exception as exc:

            self._artifact_summary.setPlainText(f"Failed to read latest: {exc}")

            return

        inference_dir = str(latest.get("inference_dir", ""))

        post_dir = str(latest.get("postprocess_dir", ""))

        lines = [

            f"Behavior scope: {self._concept.currentText()}",

            f"Inference run: {inference_dir or 'not available'}",

            f"Bout postprocess: {post_dir or 'not available'}",

            "Use Temporal Review to inspect and fine-tune thresholds.",

        ]

        self._artifact_summary.setPlainText("\n".join(lines))



    def _on_viz_selection_changed(self, _index: int) -> None:

        self._refresh_visualization_preview()



    def _refresh_viz_session_list(self, inference_dir: Path) -> None:

        manifest_path = inference_dir / "inference_manifest.json"

        if not manifest_path.exists():

            return

        try:

            mf = json.loads(manifest_path.read_text(encoding="utf-8"))

        except Exception:

            return

        trace_map: dict[str, str] = {

            str(k): str(v) for k, v in mf.get("trace_paths", {}).items()

        }

        session_ids = sorted(trace_map.keys())

        if not session_ids:

            return

        self._viz_trace_paths = trace_map

        self._viz_session_selector.blockSignals(True)

        current_sid = str(self._viz_session_selector.currentData() or "")

        self._viz_session_selector.clear()

        select_idx = 0

        display_map = self._session_display_map()

        for i, sid in enumerate(session_ids):

            display = display_map.get(sid) or (

                sid[8:16] if sid.startswith("session_") and len(sid) > 8 else sid

            )

            self._viz_session_selector.addItem(display, userData=sid)

            if sid == current_sid:

                select_idx = i

        self._viz_session_selector.setCurrentIndex(select_idx)

        self._viz_session_selector.blockSignals(False)



    def _refresh_visualization_preview(self) -> None:

        if self._project_root is None:

            if self._viz_canvas is not None:

                self._viz_canvas.setVisible(False)

            if self._viz_toolbar is not None:

                self._viz_toolbar.setVisible(False)

            self._viz_pixmap_original = None

            self._viz_preview.setPixmap(QPixmap())

            self._viz_preview.setText(

                "Open a project to view temporal refinement previews."

            )

            self._viz_preview.setVisible(True)

            return



        latest_path = self._latest_path()

        latest: dict[str, Any] = {}

        if latest_path is not None and latest_path.exists():

            try:

                latest = json.loads(latest_path.read_text(encoding="utf-8"))

            except Exception:

                latest = {}



        inference_dir = (

            Path(str(latest.get("inference_dir", "") or "").strip())

            if latest

            else Path()

        )



        if str(inference_dir) and inference_dir.exists():

            # Do NOT call _ensure_competition_preview_graphs here — it does
            # heavy matplotlib rendering + parquet I/O and blocks the GUI thread,
            # causing "not responding" on Windows.  Graph generation is handled
            # exclusively by _generate_preview_graphs_async (called from
            # _on_finished and on project load).
            self._refresh_viz_session_list(inference_dir)



        selected = str(self._viz_selector.currentData() or "auto")

        path_map: dict[str, list[Path]] = {

            "timeline": [inference_dir / "winning_behavior_timeline.png"],

            "traces": [inference_dir / "behavior_probability_traces.png"],

            "heatmap": [inference_dir / "behavior_probability_heatmap.png"],

            "auto": [

                inference_dir / "winning_behavior_timeline.png",

                inference_dir / "behavior_probability_traces.png",

                inference_dir / "behavior_probability_heatmap.png",

            ],

        }

        candidates = path_map.get(selected, path_map["auto"])

        image_path = next(

            (p for p in candidates if str(p) and p.exists()), None

        )



        if self._render_interactive_visualization(selected, inference_dir, image_path):

            return



        if image_path is None:

            if self._viz_canvas is not None:

                self._viz_canvas.setVisible(False)

            if self._viz_toolbar is not None:

                self._viz_toolbar.setVisible(False)

            self._viz_pixmap_original = None

            self._viz_preview.setPixmap(QPixmap())

            self._viz_preview.setText(

                "No graph artifacts available yet. Run inference first."

            )

            self._viz_preview.setVisible(True)

            return



        pix = QPixmap(str(image_path))

        if pix.isNull():

            if self._viz_canvas is not None:

                self._viz_canvas.setVisible(False)

            if self._viz_toolbar is not None:

                self._viz_toolbar.setVisible(False)

            self._viz_pixmap_original = None

            self._viz_preview.setPixmap(QPixmap())

            self._viz_preview.setText(

                f"Could not load graph image: {image_path.name}"

            )

            self._viz_preview.setVisible(True)

            return



        if self._viz_canvas is not None:

            self._viz_canvas.setVisible(False)

        if self._viz_toolbar is not None:

            self._viz_toolbar.setVisible(False)

        self._viz_pixmap_original = pix

        self._viz_preview.setText("")

        self._viz_preview.setVisible(True)

        self._render_visualization_pixmap()



    def _render_interactive_visualization(

        self,

        selected: str,

        inference_dir: Path,

        image_path: Path | None,

    ) -> bool:

        if self._viz_canvas is None or self._viz_figure is None:

            return False



        fig = self._viz_figure

        fig.clear()

        ax = fig.add_subplot(111)

        rendered = False



        selected_sid = str(self._viz_session_selector.currentData() or "")

        try:

            traces_dir = inference_dir / "probability_traces"

            trace_candidates = (

                sorted(traces_dir.glob("*_trace.parquet"))

                if traces_dir.exists()

                else []

            )

            _trace_file: Path | None = None

            if selected_sid and selected_sid in self._viz_trace_paths:

                _tp = Path(self._viz_trace_paths[selected_sid])

                _trace_file = _tp if _tp.exists() else None

            if _trace_file is None:

                _trace_file = trace_candidates[0] if trace_candidates else None

            trace_df = (

                pd.read_parquet(_trace_file)

                if _trace_file is not None

                else pd.DataFrame()

            )

        except Exception:

            trace_df = pd.DataFrame()



        can_plot_trace = not trace_df.empty and "frame" in trace_df.columns

        prob_cols = (

            [c for c in trace_df.columns if str(c).startswith("prob_")]

            if can_plot_trace

            else []

        )



        if can_plot_trace and selected in {"timeline", "traces", "heatmap", "auto"}:

            session_name = str(

                self._viz_session_selector.currentText() or "session"

            )



            if selected in {"timeline", "auto"} and "predicted_behavior" in trace_df.columns:

                labels = [

                    self._behavior_label(str(x))

                    for x in trace_df["predicted_behavior"].tolist()

                ]

                uniq = sorted(set(labels))

                idx_map = {lab: i for i, lab in enumerate(uniq)}

                y = np.asarray([idx_map[lab] for lab in labels], dtype=float)

                x_full = trace_df["frame"].to_numpy(dtype=int)

                stride = max(1, int(len(x_full) // 20000))

                ax.plot(

                    x_full[::stride], y[::stride], linewidth=1.0, color="#0f766e"

                )

                ax.set_yticks(np.arange(len(uniq), dtype=int))

                ax.set_yticklabels(uniq, fontsize=8)

                ax.set_xlabel("Frame")

                ax.set_title(f"Winning Behavior Timeline ({session_name})")

                ax.grid(alpha=0.2)

                rendered = True



            if (

                not rendered

                and selected in {"traces", "auto"}

                and prob_cols

            ):

                x = trace_df["frame"].to_numpy(dtype=int)

                stride = max(1, int(len(x) // 20000))

                x_plot = x[::stride]

                for col in prob_cols:

                    y = (

                        pd.to_numeric(trace_df[col], errors="coerce")

                        .fillna(0.0)

                        .to_numpy(dtype=float)[::stride]

                    )

                    ax.plot(

                        x_plot,

                        y,

                        linewidth=1.1,

                        label=self._behavior_label(col.replace("prob_", "")),

                    )

                ax.set_ylim(0.0, 1.0)

                ax.set_xlabel("Frame")

                ax.set_ylabel("Probability")

                ax.set_title(f"Behavior Probability Traces ({session_name})")

                ax.grid(alpha=0.2)

                ax.legend(loc="upper right", frameon=False, fontsize=8)

                rendered = True



            if (

                not rendered

                and selected in {"heatmap", "auto"}

                and prob_cols

            ):

                arr = np.vstack(

                    [

                        pd.to_numeric(trace_df[c], errors="coerce")

                        .fillna(0.0)

                        .to_numpy(dtype=float)

                        for c in prob_cols

                    ]

                )

                stride_hm = max(1, int(arr.shape[1] // 4000))

                arr = arr[:, ::stride_hm]

                im = ax.imshow(

                    arr,

                    aspect="auto",

                    interpolation="nearest",

                    vmin=0.0,

                    vmax=1.0,

                    cmap="viridis",

                )

                ax.set_yticks(np.arange(len(prob_cols), dtype=int))

                ax.set_yticklabels(

                    [

                        self._behavior_label(c.replace("prob_", ""))

                        for c in prob_cols

                    ],

                    fontsize=8,

                )

                ax.set_xlabel("Frame index (windowed view)")

                ax.set_title(

                    f"Behavior Probability Heatmap ({session_name})"

                )

                fig.colorbar(

                    im, ax=ax, fraction=0.025, pad=0.02, label="Probability"

                )

                rendered = True



        if not rendered and image_path is not None and mpimg is not None:

            try:

                arr = mpimg.imread(str(image_path))

                ax.imshow(arr)

                ax.set_axis_off()

                ax.set_title(image_path.name)

                rendered = True

            except Exception:

                rendered = False



        if not rendered:

            fig.clear()

            return False



        fig.tight_layout()

        self._viz_canvas.draw_idle()

        self._viz_canvas.setVisible(True)

        if self._viz_toolbar is not None:

            self._viz_toolbar.setVisible(True)

        self._viz_preview.setVisible(False)

        return True



    def _ensure_competition_preview_graphs(self, inference_dir: Path) -> None:

        traces_dir = inference_dir / "probability_traces"

        if not traces_dir.exists():

            return

        candidates = sorted(traces_dir.glob("*_trace.parquet"))

        if not candidates:

            return



        timeline_path = inference_dir / "winning_behavior_timeline.png"

        traces_path = inference_dir / "behavior_probability_traces.png"

        heatmap_path = inference_dir / "behavior_probability_heatmap.png"



        try:

            import matplotlib.pyplot as plt

        except Exception:

            return



        try:

            trace_df = pd.read_parquet(candidates[0])

        except Exception:

            return

        if trace_df.empty or "frame" not in trace_df.columns:

            return



        session_name = candidates[0].name.replace("_trace.parquet", "")

        view = trace_df.copy()

        prob_cols = [c for c in view.columns if str(c).startswith("prob_")]

        if not prob_cols:

            return



        # Winner timeline

        if "predicted_behavior" in view.columns and not timeline_path.exists():

            labels = [

                self._behavior_label(str(x))

                for x in view["predicted_behavior"].tolist()

            ]

            uniq = sorted(set(labels))

            idx_map = {lab: i for i, lab in enumerate(uniq)}

            y = np.asarray([idx_map[lab] for lab in labels], dtype=float)

            fig_t, ax_t = plt.subplots(figsize=(10.5, 2.8))

            x_full = view["frame"].to_numpy(dtype=int)

            stride = max(1, int(len(x_full) // 6000))

            ax_t.plot(x_full[::stride], y[::stride], linewidth=1.0, color="#0f766e")

            ax_t.set_yticks(np.arange(len(uniq), dtype=int))

            ax_t.set_yticklabels(uniq, fontsize=8)

            ax_t.set_xlabel("Frame")

            ax_t.set_title(f"Winning Behavior Timeline ({session_name})")

            ax_t.grid(alpha=0.2)

            fig_t.tight_layout()

            fig_t.savefig(timeline_path, dpi=180, bbox_inches="tight")

            plt.close(fig_t)



        # Probability traces

        if prob_cols and not traces_path.exists():

            fig_p, ax_p = plt.subplots(figsize=(10.5, 3.8))

            x = view["frame"].to_numpy(dtype=int)

            stride = max(1, int(len(x) // 6000))

            x_plot = x[::stride]

            for col in prob_cols:

                y = (

                    pd.to_numeric(view[col], errors="coerce")

                    .fillna(0.0)

                    .to_numpy(dtype=float)[::stride]

                )

                ax_p.plot(

                    x_plot,

                    y,

                    linewidth=1.1,

                    label=self._behavior_label(col.replace("prob_", "")),

                )

            ax_p.set_ylim(0.0, 1.0)

            ax_p.set_xlabel("Frame")

            ax_p.set_ylabel("Probability")

            ax_p.set_title(f"Behavior Probability Traces ({session_name})")

            ax_p.grid(alpha=0.2)

            ax_p.legend(loc="upper right", frameon=False, fontsize=8)

            fig_p.tight_layout()

            fig_p.savefig(traces_path, dpi=180, bbox_inches="tight")

            plt.close(fig_p)



        # Probability heatmap

        if prob_cols and not heatmap_path.exists():

            arr = np.vstack(

                [

                    pd.to_numeric(view[c], errors="coerce")

                    .fillna(0.0)

                    .to_numpy(dtype=float)

                    for c in prob_cols

                ]

            )

            stride_hm = max(1, int(arr.shape[1] // 1200))

            arr = arr[:, ::stride_hm]

            fig_h, ax_h = plt.subplots(figsize=(10.5, 4.0))

            im = ax_h.imshow(

                arr,

                aspect="auto",

                interpolation="nearest",

                vmin=0.0,

                vmax=1.0,

                cmap="viridis",

            )

            ax_h.set_yticks(np.arange(len(prob_cols), dtype=int))

            ax_h.set_yticklabels(

                [

                    self._behavior_label(c.replace("prob_", ""))

                    for c in prob_cols

                ],

                fontsize=8,

            )

            ax_h.set_xlabel("Frame index (windowed view)")

            ax_h.set_title(f"Behavior Probability Heatmap ({session_name})")

            cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02)

            cbar.set_label("Probability")

            fig_h.tight_layout()

            fig_h.savefig(heatmap_path, dpi=180, bbox_inches="tight")

            plt.close(fig_h)



    def _render_visualization_pixmap(self) -> None:

        pix = self._viz_pixmap_original

        if pix is None or pix.isNull():

            return

        target = self._viz_preview.size()

        if target.width() <= 8 or target.height() <= 8:

            return

        scaled = pix.scaled(

            target,

            Qt.AspectRatioMode.KeepAspectRatio,

            Qt.TransformationMode.SmoothTransformation,

        )

        self._viz_preview.setPixmap(scaled)



    def resizeEvent(self, event) -> None:

        super().resizeEvent(event)

        self._render_visualization_pixmap()

