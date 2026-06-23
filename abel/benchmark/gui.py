"""Standalone PySide6 GUI for running ABEL ablation benchmarks."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.benchmark.configs import ALL_TOGGLES, AblationSuite, AblationToggle
from abel.benchmark.metrics import (
    apply_behavior_names,
    compute_deltas,
    load_behavior_names,
    rank_features_by_impact,
    results_to_dataframe,
)
from abel.benchmark.plots import (
    confusion_matrix_grid,
    delta_impact_chart,
    metric_bar_chart,
    pr_curves_overlay,
    save_all_plots,
)
from abel.benchmark.report import export_csv, export_html
from abel.benchmark.runner import AblationRunner, RunResult

logger = logging.getLogger("abel.benchmark")


@dataclass
class _ModelInfo:
    """Metadata for a discovered trained model."""
    dir_name: str          # e.g. "behavior_model_Dig"
    display_name: str      # e.g. "Dig"
    behavior_id: str       # UUID from model_card labels
    classifier: str        # e.g. "xgboost"
    f1: float              # current model F1 from metrics.json
    n_features: int        # number of feature columns

# ── Dark stylesheet (matches ABEL main app) ──────────────────────

_DARK_STYLE = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 13px;
}
QMainWindow { background-color: #1e1e2e; }
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #89b4fa; }
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 7px 18px;
    color: #cdd6f4;
    min-height: 22px;
}
QPushButton:hover { background-color: #45475a; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled { color: #6c7086; }
QPushButton#runBtn {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
    font-size: 14px;
    padding: 10px 32px;
}
QPushButton#runBtn:hover { background-color: #74c7ec; }
QPushButton#runBtn:disabled { background-color: #45475a; color: #6c7086; }
QLineEdit, QComboBox, QSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #cdd6f4;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background-color: #313244; selection-background-color: #45475a; }
QTabWidget::pane { border: 1px solid #45475a; border-radius: 4px; }
QTabBar::tab {
    background-color: #313244;
    border: 1px solid #45475a;
    padding: 8px 18px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected { background-color: #45475a; color: #89b4fa; border-bottom-color: #45475a; }
QTabBar::tab:hover { background-color: #45475a; }
QTableWidget {
    gridline-color: #45475a;
    background-color: #1e1e2e;
    alternate-background-color: #181825;
}
QTableWidget::item { padding: 4px 8px; }
QTableWidget::item:selected { background-color: #45475a; }
QHeaderView::section {
    background-color: #313244;
    border: 1px solid #45475a;
    padding: 5px 8px;
    font-weight: bold;
    color: #89b4fa;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    background-color: #313244;
    text-align: center;
    color: #cdd6f4;
    height: 22px;
}
QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
QTextEdit {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 4px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 12px;
}
QScrollArea { border: none; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #45475a; }
QCheckBox::indicator:checked { background-color: #89b4fa; border-color: #89b4fa; }
QCheckBox::indicator:unchecked { background-color: #313244; }
QLabel#sectionHeader { font-size: 15px; font-weight: bold; color: #89b4fa; }
QStatusBar { background-color: #181825; color: #a6adc8; }
"""


# ── Worker signal relay ───────────────────────────────────────────────

class _WorkerSignals(QObject):
    progress = Signal(str, float)   # message, 0-1
    finished = Signal(list)         # list[RunResult]
    error = Signal(str)


class _BenchmarkWorker(QRunnable):
    """Runs the ablation suite in a background thread."""

    def __init__(self, suite: AblationSuite) -> None:
        super().__init__()
        self.suite = suite
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            runner = AblationRunner(
                self.suite,
                progress_cb=lambda msg, pct: self.signals.progress.emit(msg, pct),
            )
            results = runner.run_all()
            self.signals.finished.emit(results)
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ── Sortable table item ───────────────────────────────────────────────

class _NumericTableItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically when a float value is stored."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        my_val = self.data(Qt.ItemDataRole.UserRole)
        other_val = other.data(Qt.ItemDataRole.UserRole) if other else None
        if my_val is not None and other_val is not None:
            try:
                return float(my_val) < float(other_val)
            except (TypeError, ValueError):
                pass
        return super().__lt__(other)


# ── Main GUI window ──────────────────────────────────────────────────

class BenchmarkWindow(QMainWindow):
    """Standalone benchmark/ablation GUI for ABEL."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ABEL — Ablation Benchmark Suite")
        self.resize(1100, 780)
        self._results: list[RunResult] = []
        self._output_dir: Path | None = None
        self._metrics_df: "pd.DataFrame | None" = None  # canonical metrics data
        self._toggle_checks: dict[str, QCheckBox] = {}
        self._behavior_names: dict[str, str] = {}  # behavior_id → short name

        self._build_ui()
        self.setStyleSheet(_DARK_STYLE)

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 8)

        # Title
        title = QLabel("ABEL Ablation Benchmark")
        title.setObjectName("sectionHeader")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        root.addWidget(title)

        subtitle = QLabel(
            "Systematically evaluate the impact of each pipeline feature on model performance."
        )
        subtitle.setStyleSheet("color: #a6adc8; margin-bottom: 8px;")
        root.addWidget(subtitle)

        # Main tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, stretch=1)

        self._tabs.addTab(self._build_config_tab(), "Configuration")
        self._tabs.addTab(self._build_results_tab(), "Results")
        self._tabs.addTab(self._build_plots_tab(), "Visualizations")
        self._tabs.addTab(self._build_log_tab(), "Log")

        # Bottom bar
        bottom = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        bottom.addWidget(self._progress, stretch=1)

        self._run_btn = QPushButton("Run Benchmark")
        self._run_btn.setObjectName("runBtn")
        self._run_btn.setToolTip("Start the ablation suite — trains models for every config × behavior × fold combination")
        self._run_btn.clicked.connect(self._on_run)
        bottom.addWidget(self._run_btn)

        root.addLayout(bottom)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready. Select a project and configure toggles.")

        self.setCentralWidget(central)

    # ── Config tab ────────────────────────────────────────────────

    def _build_config_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        # Project selection
        proj_group = QGroupBox("Project")
        proj_layout = QHBoxLayout(proj_group)
        self._proj_label = QLabel("No project selected")
        self._proj_label.setStyleSheet("color: #f9e2af;")
        proj_layout.addWidget(self._proj_label, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.setToolTip("Select an ABEL project folder that contains a training set")
        browse_btn.clicked.connect(self._browse_project)
        proj_layout.addWidget(browse_btn)
        layout.addWidget(proj_group)

        # Model selector (multi-select)
        model_group = QGroupBox("Models (select which to include in analysis)")
        model_layout = QVBoxLayout(model_group)
        model_btn_row = QHBoxLayout()
        sel_all_mod = QPushButton("Select All")
        sel_all_mod.setToolTip("Include all discovered models in the ablation analysis")
        sel_all_mod.clicked.connect(lambda: self._set_all_models(True))
        sel_none_mod = QPushButton("Deselect All")
        sel_none_mod.setToolTip("Exclude all models from the analysis")
        sel_none_mod.clicked.connect(lambda: self._set_all_models(False))
        model_btn_row.addWidget(sel_all_mod)
        model_btn_row.addWidget(sel_none_mod)
        model_btn_row.addStretch()
        model_layout.addLayout(model_btn_row)
        self._model_list = QListWidget()
        self._model_list.setToolTip(
            "Each checked model will be independently evaluated across all ablation configs.\n"
            "Shows: display name, current F1, classifier, and behavior ID."
        )
        self._model_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._model_list.setMaximumHeight(160)
        model_layout.addWidget(self._model_list)
        layout.addWidget(model_group)

        # Toggle checkboxes
        toggle_group = QGroupBox("Feature Toggles (checked = included in ablation)")
        toggle_layout = QVBoxLayout(toggle_group)

        select_row = QHBoxLayout()
        sel_all = QPushButton("Select All")
        sel_all.setToolTip("Include all pipeline features in the ablation study")
        sel_all.clicked.connect(lambda: self._set_all_toggles(True))
        sel_none = QPushButton("Deselect All")
        sel_none.setToolTip("Exclude all pipeline features from the ablation study")
        sel_none.clicked.connect(lambda: self._set_all_toggles(False))
        select_row.addWidget(sel_all)
        select_row.addWidget(sel_none)
        select_row.addStretch()
        toggle_layout.addLayout(select_row)

        for toggle in ALL_TOGGLES:
            cb = QCheckBox(f"{toggle.label}")
            cb.setToolTip(toggle.description)
            cb.setChecked(True)
            self._toggle_checks[toggle.key] = cb
            toggle_layout.addWidget(cb)

        scroll = QScrollArea()
        scroll.setWidget(toggle_group)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, stretch=1)

        # Settings row
        settings_group = QGroupBox("Run Settings")
        settings_form = QFormLayout(settings_group)

        self._classifier_combo = QComboBox()
        self._classifier_combo.setToolTip("Classification algorithm used for each ablation run")
        self._classifier_combo.addItems(["lightgbm", "xgboost", "random_forest"])
        settings_form.addRow("Classifier:", self._classifier_combo)

        self._test_size_spin = QSpinBox()
        self._test_size_spin.setToolTip("Percentage of sessions held out for validation in each CV fold")
        self._test_size_spin.setRange(10, 50)
        self._test_size_spin.setValue(25)
        self._test_size_spin.setSuffix("%")
        settings_form.addRow("Test split:", self._test_size_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setToolTip("Base random seed for reproducibility; each fold offsets from this value")
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(42)
        settings_form.addRow("Random seed:", self._seed_spin)

        self._folds_spin = QSpinBox()
        self._folds_spin.setRange(2, 20)
        self._folds_spin.setValue(5)
        self._folds_spin.setToolTip("Number of repeated cross-validation folds (mean ± SEM)")
        settings_form.addRow("CV folds:", self._folds_spin)

        exec_row = QHBoxLayout()
        self._seq_radio = QRadioButton("Sequential")
        self._seq_radio.setToolTip("Run each ablation config one at a time (safer, lower memory)")
        self._par_radio = QRadioButton("Parallel")
        self._par_radio.setToolTip("Run multiple ablation configs concurrently using threads")
        self._seq_radio.setChecked(True)
        exec_row.addWidget(self._seq_radio)
        exec_row.addWidget(self._par_radio)

        self._workers_spin = QSpinBox()
        self._workers_spin.setToolTip("Maximum number of concurrent threads when running in parallel mode")
        self._workers_spin.setRange(1, 16)
        self._workers_spin.setValue(4)
        self._workers_spin.setPrefix("Workers: ")
        exec_row.addWidget(self._workers_spin)
        exec_row.addStretch()
        settings_form.addRow("Execution:", exec_row)

        layout.addWidget(settings_group)
        return page

    # ── Results tab ───────────────────────────────────────────────

    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # Summary card
        self._summary_label = QLabel("Run a benchmark to see results.")
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet(
            "background-color: #313244; border-radius: 8px; padding: 12px; "
            "border-left: 4px solid #89b4fa;"
        )
        layout.addWidget(self._summary_label)

        # Metric table
        self._metric_table = QTableWidget()
        self._metric_table.setAlternatingRowColors(True)
        self._metric_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._metric_table, stretch=1)

        # Delta table
        delta_label = QLabel("Feature Impact (Δ vs. Baseline)")
        delta_label.setObjectName("sectionHeader")
        layout.addWidget(delta_label)

        self._delta_table = QTableWidget()
        self._delta_table.setAlternatingRowColors(True)
        self._delta_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._delta_table, stretch=1)

        # Export buttons
        export_row = QHBoxLayout()
        csv_btn = QPushButton("Export CSV")
        csv_btn.setToolTip("Save the raw metric table (mean, SEM, per behavior) as a CSV file")
        csv_btn.clicked.connect(self._export_csv)
        export_row.addWidget(csv_btn)
        html_btn = QPushButton("Export HTML Report")
        html_btn.setToolTip("Generate a self-contained HTML report with embedded plots and per-behavior tables")
        html_btn.clicked.connect(self._export_html)
        export_row.addWidget(html_btn)
        open_btn = QPushButton("Open Output Folder")
        open_btn.setToolTip("Open the benchmark output directory in your file explorer")
        open_btn.clicked.connect(self._open_output_dir)
        export_row.addWidget(open_btn)
        export_row.addStretch()
        layout.addLayout(export_row)

        return page

    # ── Plots tab ─────────────────────────────────────────────────

    def _build_plots_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self._plot_tabs = QTabWidget()
        layout.addWidget(self._plot_tabs)

        placeholder = "Run a benchmark to generate plots."

        # Each sub-tab: (widget_attr, tab_name, export_data_key)
        self._plot_tab_defs = [
            ("_bar_plot_widget", "Metric Bars", "metric_bars"),
            ("_impact_plot_widget", "Feature Impact", "feature_impact"),
            ("_heatmap_plot_widget", "ΔF1 Heatmap", "delta_heatmap"),
            ("_pr_plot_widget", "PR Curves", "pr_curves"),
            ("_cm_plot_widget", "Confusion Matrices", "confusion_matrices"),
        ]

        for attr, tab_name, data_key in self._plot_tab_defs:
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)

            label = QLabel(placeholder)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            setattr(self, attr, label)
            vbox.addWidget(label, stretch=1)

            export_btn = QPushButton(f"Copy Table Data")
            export_btn.setToolTip(f"Copy the raw data behind this plot to the clipboard as a tab-separated table")
            export_btn.setMaximumWidth(180)
            export_btn.clicked.connect(lambda checked=False, key=data_key: self._copy_plot_data(key))
            export_btn.setVisible(False)
            setattr(self, f"_export_btn_{data_key}", export_btn)

            btn_row = QHBoxLayout()
            btn_row.addStretch()
            btn_row.addWidget(export_btn)
            vbox.addLayout(btn_row)

            self._plot_tabs.addTab(container, tab_name)

        return page

    # ── Log tab ───────────────────────────────────────────────────

    def _build_log_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        layout.addWidget(self._log_text)
        return page

    # ── Helpers ───────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        logger.info(msg)

    def _set_all_toggles(self, checked: bool) -> None:
        for cb in self._toggle_checks.values():
            cb.setChecked(checked)

    def _browse_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select ABEL Project Folder")
        if not folder:
            return
        root = Path(folder)
        # Validate it looks like an ABEL project
        train_path = root / "derived" / "training_sets" / "training_set.parquet"
        if not train_path.exists():
            QMessageBox.warning(
                self,
                "Invalid Project",
                f"No training set found at:\n{train_path}\n\n"
                "Run at least one active-learning training cycle in ABEL first.",
            )
            return
        self._proj_label.setText(str(root))
        self._proj_label.setStyleSheet("color: #a6e3a1;")
        self._behavior_names = load_behavior_names(root)
        self._discover_and_populate_models(root)
        self._load_previous_results(root)
        self._log(f"Project loaded: {root}")

    @staticmethod
    def _find_latest_run(bench_dir: Path) -> Path | None:
        """Return the most recent run_* subdirectory, or bench_dir itself for legacy layout."""
        if not bench_dir.is_dir():
            return None
        # Look for timestamped run subdirectories
        run_dirs = sorted(
            (d for d in bench_dir.iterdir()
             if d.is_dir() and d.name.startswith("run_") and (d / "ablation_results.csv").exists()),
            key=lambda d: d.name,
            reverse=True,
        )
        if run_dirs:
            return run_dirs[0]
        # Backward compat: flat layout with CSV directly in bench_dir
        if (bench_dir / "ablation_results.csv").exists():
            return bench_dir
        return None

    def _load_previous_results(self, project_root: Path) -> None:
        """Load the most recent benchmark results from disk if available."""
        bench_dir = project_root / "derived" / "benchmark"
        latest = self._find_latest_run(bench_dir)
        if latest is None:
            return
        csv_path = latest / "ablation_results.csv"

        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            if df.empty:
                return

            self._output_dir = latest
            self._metrics_df = df
            self._fill_table(self._metric_table, df)

            deltas = compute_deltas(df)
            self._fill_table(self._delta_table, deltas)

            # Build summary from the loaded dataframe
            from abel.benchmark.metrics import format_mean_sem
            behaviors = sorted(df["Behavior"].unique().tolist())
            lines: list[str] = []
            for behavior in behaviors:
                bdf = df[df["Behavior"] == behavior]
                beh_label = self._behavior_names.get(behavior, behavior[:20]) if behavior else "(all)"
                baseline = bdf[bdf["Run"] == "baseline_all_on"]
                all_off = bdf[bdf["Run"] == "baseline_all_off"]

                lines.append(f"<b style='color:#89b4fa;'>Behavior: {beh_label}</b>")
                if not baseline.empty:
                    b = baseline.iloc[0]
                    lines.append(
                        f"&nbsp;&nbsp;All ON: F1={format_mean_sem(b['F1'], b['F1 SEM'])}"
                        f"  PR-AUC={format_mean_sem(b['PR-AUC'], b['PR-AUC SEM'])}"
                    )
                if not all_off.empty:
                    o = all_off.iloc[0]
                    lines.append(
                        f"&nbsp;&nbsp;All OFF: F1={format_mean_sem(o['F1'], o['F1 SEM'])}"
                        f"  PR-AUC={format_mean_sem(o['PR-AUC'], o['PR-AUC SEM'])}"
                    )
                if not baseline.empty and not all_off.empty:
                    diff = float(baseline.iloc[0]["F1"]) - float(all_off.iloc[0]["F1"])
                    colour = "#a6e3a1" if diff > 0 else "#f38ba8"
                    lines.append(
                        f"&nbsp;&nbsp;Net impact: <span style='color:{colour}'>ΔF1 = {diff:+.4f}</span>"
                    )
            self._summary_label.setText("<br>".join(lines) if lines else "No results.")

            self._embed_plots()
            n_runs = len(df)
            self._log(f"Loaded {n_runs} previous benchmark results from {csv_path}")
            self._status.showMessage(f"Loaded {n_runs} previous results from disk.")
        except Exception as exc:
            self._log(f"Could not load previous results: {exc}")

    def _set_all_models(self, checked: bool) -> None:
        for i in range(self._model_list.count()):
            item = self._model_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) is not None:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    @staticmethod
    def _discover_models(project_root: Path) -> list[_ModelInfo]:
        """Scan derived/models/ for trained behavior models."""
        models_dir = project_root / "derived" / "models"
        results: list[_ModelInfo] = []
        if not models_dir.is_dir():
            return results

        for d in sorted(models_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("behavior_model_"):
                continue
            card_path = d / "model_card.yaml"
            metrics_path = d / "metrics.json"
            if not card_path.exists():
                continue

            try:
                import yaml
                card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            labels = card.get("labels", [])
            # The target behavior ID is the first non-no_behavior label
            behavior_id = next(
                (lbl for lbl in labels if lbl != "no_behavior"), ""
            )
            if not behavior_id:
                continue  # skip no_behavior-only models

            display_name = d.name.replace("behavior_model_", "").replace("_", " ")
            classifier = card.get("classifier_family", "unknown")
            n_features = len(card.get("feature_columns", []))

            f1 = float("nan")
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    f1 = float(metrics.get("f1", float("nan")))
                except Exception:
                    pass

            results.append(_ModelInfo(
                dir_name=d.name,
                display_name=display_name,
                behavior_id=behavior_id,
                classifier=classifier,
                f1=f1,
                n_features=n_features,
            ))

        return results

    def _discover_and_populate_models(self, project_root: Path) -> None:
        """Discover trained models and populate the model selector list."""
        self._model_list.clear()
        models = self._discover_models(project_root)

        if models:
            for m in models:
                f1_str = f"F1={m.f1:.3f}" if not (m.f1 != m.f1) else "F1=n/a"  # NaN check
                short = self._behavior_names.get(m.behavior_id, m.display_name)
                label = f"{short}  —  {f1_str}  |  {m.classifier}  |  {m.n_features} features"
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                item.setData(Qt.ItemDataRole.UserRole, m.behavior_id)
                item.setToolTip(
                    f"Model: {m.dir_name}\n"
                    f"Behavior ID: {m.behavior_id}\n"
                    f"Classifier: {m.classifier}\n"
                    f"Current F1: {m.f1:.4f}\n"
                    f"Features: {m.n_features}"
                )
                self._model_list.addItem(item)
            self._log(f"Discovered {len(models)} trained model(s)")
        else:
            # Fallback: detect behaviors from training set labels
            self._log("No trained models found — falling back to behavior detection from training data")
            self._fallback_detect_behaviors(project_root)

    def _fallback_detect_behaviors(self, project_root: Path) -> None:
        """Detect behaviors from training set when no models exist yet."""
        behaviors: list[str] = []
        train_path = project_root / "derived" / "training_sets" / "training_set.parquet"
        if train_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(train_path, columns=["label"])
                nb_tokens = {"no_behavior", "no_behaviour", "ambiguous", "boundary_error"}
                for lbl in sorted(df["label"].unique()):
                    lbl_str = str(lbl).strip()
                    if lbl_str.lower().replace("_", "") not in {t.replace("_", "") for t in nb_tokens}:
                        if lbl_str not in behaviors:
                            behaviors.append(lbl_str)
            except Exception:
                pass

        for beh in behaviors:
            item = QListWidgetItem(beh)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, beh)
            self._model_list.addItem(item)

        if not behaviors:
            item = QListWidgetItem("(no models or behaviors detected)")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            self._model_list.addItem(item)

    # ── Run ablation ──────────────────────────────────────────────

    def _on_run(self) -> None:
        project_path = self._proj_label.text()
        if not project_path or "No project" in project_path:
            QMessageBox.warning(self, "No Project", "Select a project folder first.")
            return

        # Collect selected behaviors from model list
        selected_behaviors: list[str] = []
        for i in range(self._model_list.count()):
            item = self._model_list.item(i)
            beh_id = item.data(Qt.ItemDataRole.UserRole)
            if beh_id and item.checkState() == Qt.CheckState.Checked:
                selected_behaviors.append(beh_id)

        if not selected_behaviors:
            QMessageBox.warning(self, "No Models Selected", "Select at least one model to include in the analysis.")
            return

        # Collect enabled toggles
        selected_toggles: list[AblationToggle] = []
        for toggle in ALL_TOGGLES:
            if self._toggle_checks.get(toggle.key, QCheckBox()).isChecked():
                selected_toggles.append(toggle)

        if not selected_toggles:
            QMessageBox.warning(self, "No Toggles", "Enable at least one feature toggle.")
            return

        project_root = Path(project_path)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._output_dir = project_root / "derived" / "benchmark" / f"run_{stamp}"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        suite = AblationSuite(
            project_root=str(project_root),
            target_behaviors=selected_behaviors,
            toggles=selected_toggles,
            classifier_family=self._classifier_combo.currentText(),
            test_size=self._test_size_spin.value() / 100.0,
            random_state=self._seed_spin.value(),
            n_cv_folds=self._folds_spin.value(),
            parallel=self._par_radio.isChecked(),
            max_workers=self._workers_spin.value(),
            output_dir=str(self._output_dir),
        )

        self._run_btn.setEnabled(False)
        self._progress.setValue(0)
        self._progress.setFormat("0%")
        self._log_text.clear()
        self._log(f"Starting ablation suite: {len(selected_toggles)} toggles, "
                  f"{len(selected_behaviors)} behavior(s), "
                  f"{suite.n_cv_folds} CV folds, classifier={suite.classifier_family}")
        self._log(f"Execution mode: {'parallel' if suite.parallel else 'sequential'}")
        self._status.showMessage("Running ablation benchmark…")

        worker = _BenchmarkWorker(suite)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        QThreadPool.globalInstance().start(worker)

    @Slot(str, float)
    def _on_progress(self, msg: str, pct: float) -> None:
        self._progress.setValue(int(pct * 1000))
        # Show ETA on the progress bar itself
        eta_part = ""
        if "ETA " in msg:
            for token in msg.split():
                if token.startswith("ETA"):
                    continue
                if token.endswith(("s", "m", "h")) and any(c.isdigit() for c in token):
                    eta_part = f"ETA {token}"
                    break
            if not eta_part:
                idx = msg.index("ETA ")
                eta_part = msg[idx:].strip()
        pct_text = f"{pct * 100:.0f}%"
        self._progress.setFormat(f"{pct_text}  {eta_part}" if eta_part else pct_text)
        self._status.showMessage(msg)
        self._log(msg)

    @Slot(list)
    def _on_finished(self, results: list[RunResult]) -> None:
        self._results = results
        self._metrics_df = results_to_dataframe(results)
        self._run_btn.setEnabled(True)
        self._progress.setValue(1000)
        self._progress.setFormat("100% — Complete")
        self._status.showMessage(f"Complete — {len(results)} runs finished.")
        self._log(f"\nBenchmark complete: {len(results)} configurations evaluated.")

        # Populate tables
        self._populate_metric_table()
        self._populate_delta_table()
        self._populate_summary()

        # Auto-export CSV first so results persist even if plots fail
        if self._output_dir:
            try:
                csv_path = export_csv(results, self._output_dir / "ablation_results.csv")
                self._log(f"CSV exported: {csv_path}")
            except Exception as exc:
                self._log(f"CSV export failed: {exc}")

        # Generate plots
        if self._output_dir:
            try:
                saved = save_all_plots(
                    results, self._output_dir,
                    behavior_names=self._behavior_names,
                )
                self._log(f"Saved {len(saved)} plot(s) to {self._output_dir}")
            except Exception as exc:
                self._log(f"Plot generation failed: {exc}")
            self._embed_plots()

        # Switch to results tab
        self._tabs.setCurrentIndex(1)

    @Slot(str)
    def _on_error(self, err: str) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setValue(0)
        self._status.showMessage("Error!")
        self._log(f"ERROR: {err}")
        QMessageBox.critical(self, "Benchmark Error", err)

    # ── Populate tables ───────────────────────────────────────────

    def _populate_metric_table(self) -> None:
        df = results_to_dataframe(self._results)
        self._fill_table(self._metric_table, df)

    def _populate_delta_table(self) -> None:
        df = results_to_dataframe(self._results)
        deltas = compute_deltas(df)
        self._fill_table(self._delta_table, deltas)

    def _populate_summary(self) -> None:
        from abel.benchmark.metrics import format_mean_sem

        df = results_to_dataframe(self._results)
        behaviors = sorted(df["Behavior"].unique().tolist())
        lines: list[str] = []

        for behavior in behaviors:
            bdf = df[df["Behavior"] == behavior]
            beh_label = self._behavior_names.get(behavior, behavior[:20]) if behavior else "(all)"
            baseline = bdf[bdf["Run"] == "baseline_all_on"]
            all_off = bdf[bdf["Run"] == "baseline_all_off"]

            lines.append(f"<b style='color:#89b4fa;'>Behavior: {beh_label}</b>")
            if not baseline.empty:
                b = baseline.iloc[0]
                lines.append(
                    f"&nbsp;&nbsp;All ON: F1={format_mean_sem(b['F1'], b['F1 SEM'])}"
                    f"  PR-AUC={format_mean_sem(b['PR-AUC'], b['PR-AUC SEM'])}"
                )
            if not all_off.empty:
                o = all_off.iloc[0]
                lines.append(
                    f"&nbsp;&nbsp;All OFF: F1={format_mean_sem(o['F1'], o['F1 SEM'])}"
                    f"  PR-AUC={format_mean_sem(o['PR-AUC'], o['PR-AUC SEM'])}"
                )
            if not baseline.empty and not all_off.empty:
                diff = float(baseline.iloc[0]["F1"]) - float(all_off.iloc[0]["F1"])
                colour = "#a6e3a1" if diff > 0 else "#f38ba8"
                lines.append(
                    f"&nbsp;&nbsp;Net impact: <span style='color:{colour}'>ΔF1 = {diff:+.4f}</span>"
                )

        self._summary_label.setText("<br>".join(lines) if lines else "No results.")

    def _fill_table(self, table: QTableWidget, df: "pd.DataFrame") -> None:
        if df.empty:
            table.clear()
            return
        # Map behaviour IDs → short names for display
        df = apply_behavior_names(df, self._behavior_names)
        table.setSortingEnabled(False)
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(list(df.columns))

        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(df.columns):
                val = row[col]
                if isinstance(val, float):
                    if col.startswith("Δ"):
                        text = f"{val:+.4f}"
                    else:
                        text = f"{val:.4f}" if abs(val) < 100 else f"{val:.1f}"
                else:
                    text = str(val)
                item = _NumericTableItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                # Store numeric value for proper sort ordering
                if isinstance(val, (int, float)):
                    item.setData(Qt.ItemDataRole.UserRole, float(val))

                # Colour deltas
                if isinstance(val, float) and col.startswith("Δ"):
                    if val < -0.001:
                        item.setForeground(Qt.GlobalColor.red)
                    elif val > 0.001:
                        item.setForeground(Qt.GlobalColor.green)

                table.setItem(r, c, item)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        if table.columnCount() > 0:
            table.horizontalHeader().setSectionResizeMode(
                table.columnCount() - 1, QHeaderView.ResizeMode.Stretch
            )
        table.setSortingEnabled(True)

    # ── Embed plots ───────────────────────────────────────────────

    def _embed_plots(self) -> None:
        """Load saved PNG plots into the visualization tab."""
        if not self._output_dir:
            return

        try:
            from PySide6.QtGui import QPixmap

            plots = [
                ("_bar_plot_widget", "ablation_metric_bars.png", "metric_bars"),
                ("_impact_plot_widget", "ablation_feature_impact.png", "feature_impact"),
                ("_heatmap_plot_widget", "ablation_delta_heatmap.png", "delta_heatmap"),
            ]

            # PR curves and confusion matrices are saved per-behavior;
            # pick the first match so the tab isn't empty.
            for prefix, attr, data_key in [
                ("ablation_pr_curves_", "_pr_plot_widget", "pr_curves"),
                ("ablation_confusion_matrices_", "_cm_plot_widget", "confusion_matrices"),
            ]:
                matches = sorted(self._output_dir.glob(f"{prefix}*.png"))
                if matches:
                    plots.append((attr, matches[0].name, data_key))

            for attr, filename, data_key in plots:
                path = self._output_dir / filename
                if path.exists():
                    pixmap = QPixmap(str(path))
                    if not pixmap.isNull():
                        label = QLabel()
                        label.setPixmap(pixmap.scaledToWidth(
                            min(900, pixmap.width()),
                            Qt.TransformationMode.SmoothTransformation,
                        ))
                        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                        scroll = QScrollArea()
                        scroll.setWidget(label)
                        scroll.setWidgetResizable(True)

                        # Replace the placeholder label inside the container
                        old_widget = getattr(self, attr, None)
                        if old_widget is not None:
                            parent_layout = old_widget.parentWidget().layout() if old_widget.parentWidget() else None
                            if parent_layout:
                                idx = parent_layout.indexOf(old_widget)
                                if idx >= 0:
                                    parent_layout.removeWidget(old_widget)
                                    old_widget.deleteLater()
                                    parent_layout.insertWidget(idx, scroll, stretch=1)
                                    setattr(self, attr, scroll)

                        # Show the export button
                        export_btn = getattr(self, f"_export_btn_{data_key}", None)
                        if export_btn:
                            export_btn.setVisible(True)

        except Exception as exc:
            self._log(f"Could not embed plots: {exc}")

    def _get_plot_dataframe(self, data_key: str) -> "pd.DataFrame | None":
        """Build the raw data table behind a given plot tab."""
        import pandas as pd

        if self._metrics_df is None or self._metrics_df.empty:
            return None
        df = self._metrics_df.copy()
        if self._behavior_names:
            df = apply_behavior_names(df, self._behavior_names)

        if data_key == "metric_bars":
            # Raw metric table: Run, Behavior, Precision ± SEM, Recall ± SEM, F1 ± SEM, PR-AUC ± SEM
            cols = ["Run", "Behavior", "Precision", "Precision SEM",
                    "Recall", "Recall SEM", "F1", "F1 SEM", "PR-AUC", "PR-AUC SEM"]
            return df[[c for c in cols if c in df.columns]]

        elif data_key == "feature_impact":
            # ΔF1 vs all_on per behavior (cost of removing each feature)
            rows = []
            for behavior in sorted(df["Behavior"].unique()):
                deltas = compute_deltas(df, baseline_name="baseline_all_on", behavior=behavior)
                if deltas.empty:
                    continue
                if self._behavior_names:
                    deltas = apply_behavior_names(deltas, self._behavior_names)
                rows.append(deltas)
            return pd.concat(rows, ignore_index=True) if rows else None

        elif data_key == "delta_heatmap":
            # Pivot: Toggle × Behavior → ΔF1 vs all_on (cost of removal)
            rows = []
            for behavior in sorted(df["Behavior"].unique()):
                deltas = compute_deltas(df, baseline_name="baseline_all_on", behavior=behavior)
                if deltas.empty or "ΔF1" not in deltas.columns:
                    continue
                if self._behavior_names:
                    deltas = apply_behavior_names(deltas, self._behavior_names)
                mask = deltas["Run"].str.startswith("without_")
                sub = deltas[mask].copy()
                sub["Toggle"] = (
                    sub["Run"].str.replace("without_", "", n=1)
                    .str.replace("_", " ").str.title()
                )
                rows.append(sub[["Toggle", "Behavior", "ΔF1"]])
            if not rows:
                return None
            long = pd.concat(rows, ignore_index=True)
            return long.pivot(index="Toggle", columns="Behavior", values="ΔF1").reset_index()

        elif data_key == "pr_curves":
            # PR-AUC summary per run/behavior
            cols = ["Run", "Behavior", "PR-AUC", "PR-AUC SEM"]
            return df[[c for c in cols if c in df.columns]]

        elif data_key == "confusion_matrices":
            # Metric summary relevant to confusion matrices
            cols = ["Run", "Behavior", "Precision", "Recall", "F1"]
            return df[[c for c in cols if c in df.columns]]

        return None

    def _copy_plot_data(self, data_key: str) -> None:
        """Copy the raw data behind a plot to the clipboard as tab-separated text."""
        df = self._get_plot_dataframe(data_key)
        if df is None or df.empty:
            self._status.showMessage("No data available to copy.")
            return

        text = df.to_csv(sep="\t", index=False)
        QApplication.clipboard().setText(text)
        self._status.showMessage(f"Copied {len(df)} rows to clipboard.")
        self._log(f"Copied {data_key} data ({len(df)} rows) to clipboard.")

    # ── Export actions ────────────────────────────────────────────

    def _export_csv(self) -> None:
        if not self._results:
            QMessageBox.information(self, "No Results", "Run a benchmark first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "ablation_results.csv", "CSV Files (*.csv)"
        )
        if path:
            export_csv(self._results, Path(path))
            self._log(f"CSV exported: {path}")
            self._status.showMessage(f"CSV saved to {path}")

    def _export_html(self) -> None:
        if not self._results:
            QMessageBox.information(self, "No Results", "Run a benchmark first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HTML Report", "ablation_report.html", "HTML Files (*.html)"
        )
        if path:
            project_name = Path(self._proj_label.text()).name if self._proj_label.text() else ""
            export_html(
                self._results,
                Path(path),
                plot_dir=self._output_dir,
                project_name=project_name,
                behavior_names=self._behavior_names,
            )
            self._log(f"HTML report exported: {path}")
            self._status.showMessage(f"Report saved to {path}")

    def _open_output_dir(self) -> None:
        if self._output_dir and self._output_dir.exists():
            if sys.platform == "win32":
                os.startfile(str(self._output_dir))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(self._output_dir)], check=False)
            else:
                subprocess.run(["xdg-open", str(self._output_dir)], check=False)
        else:
            QMessageBox.information(self, "No Output", "Run a benchmark first to create output.")


# ── Entry point ───────────────────────────────────────────────────────

def launch_benchmark_gui() -> None:
    """Launch the standalone benchmark GUI."""
    existing_app = QApplication.instance()
    app = existing_app or QApplication(sys.argv)
    window = BenchmarkWindow()
    window.show()

    # Only run the event loop if we created the app (standalone mode)
    if not existing_app:
        sys.exit(app.exec())


if __name__ == "__main__":
    launch_benchmark_gui()
