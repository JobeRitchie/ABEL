"""PySide6 GUI for the ABEL Validation / Meta-Analysis Platform.

One tab per validation analysis, each surfacing its own relevant settings.
Projects + behaviors + holdout settings are shared across tabs.  Each analysis
runs ABEL's *real* training primitive off the UI thread via QThreadPool.
"""

from __future__ import annotations

import csv
import html
import io
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit,
    QListView, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
    QMenu, QPushButton, QScrollArea, QSpinBox, QSplitter, QStatusBar, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextEdit, QTreeView, QVBoxLayout, QWidget,
)

from abel.ui.raw_data_warning import confirm_run_with_missing_raw_data
from abel.validation import bundle, pdf_report, subsample
from abel.validation.analyses.discrimination import FEATURE_SET_LABELS
from abel.validation.analyses.learning_curve import DEFAULT_SIZES
from abel.validation.datamodel import ProjectRef
from abel.validation.findings import KIND_CAVEAT, KIND_WARNING
from abel.validation.plots import LEARNING_CURVE_VIEWS
from abel.validation.runner import (
    ANALYSIS_ABLATION, ANALYSIS_AL_CURVE, ANALYSIS_BEHAVIORSCAPE,
    ANALYSIS_DISCRIMINATION, ANALYSIS_GENERALIZATION, ANALYSIS_LABELS,
    ANALYSIS_LEARNING_CURVE, ANALYSIS_RARE_DISCOVERY, ANALYSIS_THROUGHPUT,
    ANALYSIS_VIDEO_VALUE, FULL_SUITE,
    RunOutputs, ValidationRunConfig, preset_description, publication_config,
    run_rarity_preflight, run_validation,
)
from abel.validation.workspace import (
    RUNS_DIRNAME, SessionRecord, SessionStore, workspace_root,
)

_DARK_STYLE = """
QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', system-ui, sans-serif; font-size: 13px; }
QMainWindow { background-color: #1e1e2e; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 14px; padding-top: 18px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #89b4fa; }
QPushButton { background-color: #313244; border: 1px solid #45475a; border-radius: 5px; padding: 7px 18px; color: #cdd6f4; min-height: 22px; }
QPushButton:hover { background-color: #45475a; }
QPushButton:disabled { color: #6c7086; }
QPushButton#runBtn { background-color: #89b4fa; color: #1e1e2e; font-weight: bold; font-size: 14px; padding: 10px 28px; }
QPushButton#runBtn:hover { background-color: #74c7ec; }
QPushButton#runBtn:disabled { background-color: #45475a; color: #6c7086; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 4px 8px; color: #cdd6f4; }
QTabWidget::pane { border: 1px solid #45475a; border-radius: 4px; }
QTabBar::tab { background-color: #313244; border: 1px solid #45475a; padding: 8px 18px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background-color: #45475a; color: #89b4fa; }
QTableWidget { gridline-color: #45475a; background-color: #1e1e2e; alternate-background-color: #181825; }
QHeaderView::section { background-color: #313244; border: 1px solid #45475a; padding: 5px 8px; font-weight: bold; color: #89b4fa; }
QProgressBar { border: 1px solid #45475a; border-radius: 4px; background-color: #313244; text-align: center; color: #cdd6f4; height: 22px; }
QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
QTextEdit { background-color: #181825; border: 1px solid #45475a; border-radius: 4px; font-family: 'Cascadia Code','Consolas',monospace; font-size: 12px; }
QScrollArea { border: none; }
QListWidget { background-color: #181825; border: 1px solid #45475a; border-radius: 4px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #45475a; }
QCheckBox::indicator:checked { background-color: #89b4fa; border-color: #89b4fa; }
QStatusBar { background-color: #181825; color: #a6adc8; }
"""


# ── Background worker ───────────────────────────────────────────────────────

class _Signals(QObject):
    progress = Signal(str, float)
    finished = Signal(object)   # RunOutputs
    error = Signal(str)


class _RunWorker(QRunnable):
    def __init__(self, projects, behaviors, cfg) -> None:
        super().__init__()
        self.projects = projects
        self.behaviors = behaviors
        self.cfg = cfg
        self.signals = _Signals()

    @Slot()
    def run(self) -> None:
        try:
            out = run_validation(
                self.projects, self.behaviors, self.cfg,
                progress_cb=lambda m, f: self.signals.progress.emit(m, f),
            )
            self.signals.finished.emit(out)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class _PreflightWorker(QRunnable):
    """Phase 1 of the rare-behaviour workflow — the seconds-long rarity check."""

    def __init__(self, projects, behaviors, cfg) -> None:
        super().__init__()
        self.projects = projects
        self.behaviors = behaviors
        self.cfg = cfg
        self.signals = _Signals()   # finished carries list[ProjectPreflight]

    @Slot()
    def run(self) -> None:
        try:
            out = run_rarity_preflight(
                self.projects, self.behaviors, self.cfg,
                progress_cb=lambda m, f: self.signals.progress.emit(m, f),
            )
            self.signals.finished.emit(out)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class _DemoSignals(QObject):
    progress = Signal(str, float)
    finished = Signal(str)      # output mp4 path
    error = Signal(str)


class _DemoWorker(QRunnable):
    """Runs feature_demo.export_feature_demo off the UI thread."""

    def __init__(self, kwargs: dict) -> None:
        super().__init__()
        self._kwargs = kwargs
        self.signals = _DemoSignals()

    @Slot()
    def run(self) -> None:
        try:
            from abel.validation.feature_demo import export_feature_demo  # noqa: PLC0415
            out = export_feature_demo(
                progress_cb=lambda m, f: self.signals.progress.emit(m, f),
                **self._kwargs,
            )
            self.signals.finished.emit(str(out))
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class _BehaviorscapeSignals(QObject):
    progress = Signal(str, float)
    finished = Signal(object)   # dict: images, csv, summary
    error = Signal(str)


class _BehaviorscapeWorker(QRunnable):
    """Collects per-(project, behavior) feature importance and renders the four
    behaviorscape figures, off the UI thread."""

    def __init__(self, projects, behaviors, holdout_kwargs, build_kwargs, out_dir) -> None:
        super().__init__()
        self.projects = projects
        self.behaviors = behaviors
        self.holdout_kwargs = holdout_kwargs
        self.build_kwargs = build_kwargs
        self.out_dir = Path(out_dir)
        self.signals = _BehaviorscapeSignals()

    @Slot()
    def run(self) -> None:
        try:
            from abel.services.active_learning_trainer_service import (  # noqa: PLC0415
                ActiveLearningTrainerService,
            )
            from abel.validation import plots  # noqa: PLC0415
            from abel.validation.analyses import behaviorscape  # noqa: PLC0415

            trainer = ActiveLearningTrainerService()
            sources = behaviorscape.collect_feature_importance(
                trainer, self.projects, self.behaviors,
                progress_cb=lambda m, f: self.signals.progress.emit(m, f * 0.9),
                **self.holdout_kwargs,
            )
            self.signals.progress.emit("Pooling + rendering figures…", 0.92)
            data = behaviorscape.build_behaviorscape(sources, **self.build_kwargs)
            if data.is_empty():
                n_ok = sum(1 for s in sources if s.ok)
                errs = "; ".join(sorted({s.error for s in sources if s.error})[:3])
                raise RuntimeError(
                    f"No feature importance to plot ({n_ok} usable models). "
                    "Models may be untrained/degenerate or the threshold too high."
                    + (f" Errors: {errs}" if errs else "")
                )
            self.out_dir.mkdir(parents=True, exist_ok=True)
            images = plots.behaviorscape_figures(data, self.out_dir)

            # Significance test: do behaviors rely on different features?  Computed
            # up front so the distinctiveness export and the headline share one result.
            stats = None
            try:
                stats = behaviorscape.behavior_distinctiveness_stats(data)
            except Exception:
                stats = None

            # One tidy table per figure, so every plotted series can be exported / copied.
            tables: dict[str, Path] = {}

            def _dump(label: str, df, fname: str) -> None:
                if df is None or getattr(df, "empty", True):
                    return
                path = self.out_dir / fname
                index = label == "Behavior similarity matrix"  # square matrix keeps its labels
                df.to_csv(path, index=index)
                tables[label] = path

            _dump("Feature importance (long)", data.to_long_df(),
                  "behaviorscape_importance.csv")
            _dump("Modality shares per behavior", data.modality_fraction_long_df(),
                  "behaviorscape_modality_shares.csv")
            _dump("Behavior distinctiveness", behaviorscape.distinctiveness_df(stats),
                  "behaviorscape_distinctiveness.csv")
            _dump("Behavior similarity matrix", data.similarity_matrix_df(),
                  "behaviorscape_similarity_matrix.csv")

            stats_line = ""
            try:
                import json as _json  # noqa: PLC0415
                if stats is not None:
                    stats_payload = {
                        "permanova": stats.permanova,
                        "mean_distinctiveness": stats.mean_distinctiveness,
                        "distinctiveness": stats.distinctiveness,
                        "n_replicates": stats.n_replicates,
                    }
                    (self.out_dir / "behaviorscape_stats.json").write_text(
                        _json.dumps(stats_payload, indent=2), encoding="utf-8")
                    if stats.permanova:
                        pm = stats.permanova
                        p_txt = "p<0.001" if pm["p"] < 0.001 else f"p={pm['p']:.3f}"
                        stats_line = (
                            f"PERMANOVA — behavior explains {pm['R2'] * 100:.0f}% of "
                            f"importance variance (pseudo-F={pm['pseudo_F']:.1f}, {p_txt}, "
                            f"{pm['n_groups']} behaviors with ≥2 projects).\n"
                        )
                    else:
                        stats_line = ("Distinctiveness computed; PERMANOVA needs ≥2 behaviors "
                                      "with ≥2 projects each.\n")
            except Exception:
                stats_line = ""

            members_lines = [
                f"  • {beh}  ←  {', '.join(members)}"
                for beh, members in sorted(data.pooled_members.items())
            ]
            summary = (
                f"{len(data.behaviors)} behaviors · {data.n_features_kept} of "
                f"{data.n_features_total} features kept "
                f"(threshold {data.threshold:g}, {data.normalize}).\n"
                + stats_line
                + "Pooled behaviors:\n" + "\n".join(members_lines)
            )
            self.signals.progress.emit("Behaviorscape complete.", 1.0)
            self.signals.finished.emit({
                "images": images, "tables": tables, "summary": summary,
                "out_dir": self.out_dir,
            })
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class _JobSignals(QObject):
    progress = Signal(str, float)
    finished = Signal(object)   # dict: images, tables, summary
    error = Signal(str)


class _VideoValueWorker(QRunnable):
    """Paired with/without video-feature comparison, off the UI thread."""

    def __init__(self, projects, behaviors, holdout_kwargs, n_seeds, out_dir) -> None:
        super().__init__()
        self.projects = projects
        self.behaviors = behaviors
        self.holdout_kwargs = holdout_kwargs
        self.n_seeds = int(n_seeds)
        self.out_dir = Path(out_dir)
        self.signals = _JobSignals()

    @Slot()
    def run(self) -> None:
        try:
            from abel.services.active_learning_trainer_service import (  # noqa: PLC0415
                ActiveLearningTrainerService,
            )
            from abel.validation import holdout, video_value  # noqa: PLC0415

            trainer = ActiveLearningTrainerService()
            total = sum(len(self.behaviors.get(p.project_id, [])) for p in self.projects) or 1
            done = 0
            results: list = []
            for project in self.projects:
                bids = self.behaviors.get(project.project_id, [])
                if not bids:
                    continue
                hsplit = holdout.split(
                    project,
                    min_confidence=self.holdout_kwargs["min_confidence"],
                    test_size=self.holdout_kwargs["holdout_test_size"],
                    seed=self.holdout_kwargs["holdout_seed"])
                for bid in bids:
                    name = project.behavior_label(bid)
                    self.signals.progress.emit(
                        f"Video-feature comparison — {project.project_id}: {name}",
                        done / total)
                    results.append(video_value.run_video_value(
                        trainer, project, str(bid), hsplit, n_seeds=self.n_seeds,
                        progress_cb=lambda m: self.signals.progress.emit(m, done / total)))
                    done += 1

            self.out_dir.mkdir(parents=True, exist_ok=True)
            df = video_value.results_to_frame(results)
            csv_path = self.out_dir / "video_value.csv"
            df.to_csv(csv_path, index=False)
            png = video_value.plot_video_value(results, self.out_dir / "video_value.png")

            usable = [r for r in results if not r.error]
            wins = [r for r in usable if r.significant and r.gain > 0]
            lines = [f"  • {r.project_id} · {r.behavior_name}: "
                     f"F1 {r.f1_no_video:.3f} → {r.f1_with_video:.3f}  "
                     f"(Δ{r.gain:+.3f}{' *' if r.significant else ''})"
                     for r in usable]
            errs = [f"  • {r.project_id} · {r.behavior_name}: {r.error}"
                    for r in results if r.error]
            summary = (
                f"{len(usable)} comparison(s); {len(wins)} show a significant video-feature "
                f"gain (95% CI excludes 0).\n" + "\n".join(lines)
                + ("\nSkipped:\n" + "\n".join(errs) if errs else ""))
            self.signals.progress.emit("Video-feature comparison complete.", 1.0)
            self.signals.finished.emit({
                "images": [png], "tables": {"Video-feature value": csv_path},
                "summary": summary, "out_dir": self.out_dir})
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class _BenchmarkWorker(QRunnable):
    """Pipeline throughput benchmark (extract / train / infer), off the UI thread."""

    def __init__(self, projects, behaviors, stages, out_dir) -> None:
        super().__init__()
        self.projects = projects
        self.behaviors = behaviors
        self.stages = list(stages)
        self.out_dir = Path(out_dir)
        self.signals = _JobSignals()

    @Slot()
    def run(self) -> None:
        try:
            from abel.validation import benchmark  # noqa: PLC0415

            total = max(1, len(self.projects) * len(self.stages))
            done = {"n": 0}

            def _emit(msg: str) -> None:
                self.signals.progress.emit(msg, min(1.0, done["n"] / total))

            results: list = []
            for project in self.projects:
                names = [project.behavior_label(b)
                         for b in self.behaviors.get(project.project_id, [])] or None
                if benchmark.STAGE_EXTRACT in self.stages:
                    results.append(benchmark.time_extraction(project.root, log=_emit))
                    done["n"] += 1
                if benchmark.STAGE_TRAIN in self.stages:
                    results.extend(benchmark.time_training(project.root, names, log=_emit))
                    done["n"] += 1
                if benchmark.STAGE_INFER in self.stages:
                    results.append(benchmark.time_inference(project.root, log=_emit))
                    done["n"] += 1

            self.out_dir.mkdir(parents=True, exist_ok=True)
            df = benchmark.results_to_frame(results)
            csv_path = self.out_dir / "benchmark.csv"
            df.to_csv(csv_path, index=False)
            png = benchmark.plot_benchmark(results, self.out_dir / "benchmark.png")

            def _fmt(r) -> str:
                if r.error:
                    return f"  • {r.project_id} · {r.stage}{('/' + r.detail) if r.detail else ''}: {r.error}"
                if r.stage == benchmark.STAGE_TRAIN:
                    return f"  • {r.project_id} · train {r.detail}: {r.seconds:.1f}s ({r.units})"
                return (f"  • {r.project_id} · {r.stage}: {r.seconds:.1f}s "
                        f"for {r.video_seconds:.0f}s video → {r.faster_than_realtime:.1f}× real-time")
            summary = ("Pipeline throughput (higher × = faster than real-time):\n"
                       + "\n".join(_fmt(r) for r in results))
            self.signals.progress.emit("Benchmark complete.", 1.0)
            self.signals.finished.emit({
                "images": [png], "tables": {"Throughput": csv_path},
                "summary": summary, "out_dir": self.out_dir})
        except Exception:
            self.signals.error.emit(traceback.format_exc())


# ── Pop-out figure viewer ────────────────────────────────────────────────────

class _FigurePopout(QMainWindow):
    """Standalone resizable window showing one figure at full resolution.

    Opens when a thumbnail is clicked.  Supports fit-to-window / 100% zoom and
    saving a copy.  Kept alive in a class-level registry so it isn't garbage
    collected the moment the opener returns.
    """

    _open: list["_FigurePopout"] = []

    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = Path(path)
        self._pix = QPixmap(str(self._path))
        self._fit = True
        self.setWindowTitle(self._path.stem.replace("_", " "))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        bar = QHBoxLayout()
        fit_btn = QPushButton("Fit to window"); fit_btn.clicked.connect(self._fit_to_window)
        full_btn = QPushButton("100%"); full_btn.clicked.connect(self._actual_size)
        save_btn = QPushButton("Save copy…"); save_btn.clicked.connect(self._save)
        bar.addWidget(fit_btn); bar.addWidget(full_btn); bar.addStretch(); bar.addWidget(save_btn)
        root.addLayout(bar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img = QLabel()
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setWidget(self._img)
        root.addWidget(self._scroll, 1)
        self.setCentralWidget(central)

        # Size the window to ~75% of the available screen, capped at native size.
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(int(avail.width() * 0.75), self._pix.width() + 60)
            h = min(int(avail.height() * 0.8), self._pix.height() + 110)
            self.resize(max(480, w), max(360, h))

        _FigurePopout._open.append(self)
        self.destroyed.connect(lambda *_: _FigurePopout._open.remove(self)
                               if self in _FigurePopout._open else None)
        self._apply()

    def _apply(self) -> None:
        if self._pix.isNull():
            self._img.setText("Could not load figure.")
            return
        if self._fit:
            vp = self._scroll.viewport().size()
            scaled = self._pix.scaled(vp, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
            self._img.setPixmap(scaled)
            self._img.resize(scaled.size())
        else:
            self._img.setPixmap(self._pix)
            self._img.resize(self._pix.size())

    def _fit_to_window(self) -> None:
        self._fit = True
        self._apply()

    def _actual_size(self) -> None:
        self._fit = False
        self._apply()

    def _save(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(self, "Save figure", self._path.name,
                                              "PNG image (*.png)")
        if dest:
            shutil.copyfile(self._path, dest)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit:
            self._apply()


# ── Reusable image strip ────────────────────────────────────────────────────

class _ClickableThumbnail(QLabel):
    """A figure thumbnail that opens a full-size :class:`_FigurePopout` on click."""

    BORDER_PX = 2   # widest border state (hover); callers pad the widget for it

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = Path(path)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to open this figure full size")
        self.setStyleSheet(
            "QLabel { border: 1px solid #45475a; }"
            "QLabel:hover { border: 2px solid #89b4fa; }"
        )

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._path.exists():
            _FigurePopout(self._path, parent=self.window()).show()
        super().mouseReleaseEvent(event)


class _ImageStrip(QScrollArea):
    """Scrollable wrapping grid of figure thumbnails that fill the pane.

    Figures wrap into as many columns as fit at ``_MIN_THUMB_W``, then each
    thumbnail is *grown* to divide the available width evenly (capped at the
    figure's native resolution, so nothing is ever upscaled into blur).  A wide
    results pane therefore renders one or two figures big enough to actually
    read, instead of leaving a fixed-width thumbnail marooned in white space.
    Click any thumbnail to open it full size in a separate window; use Export
    Figure for the original file.
    """

    _MIN_THUMB_W = 380   # never wrap to a column narrower than this
    _MAX_THUMB_W = 1600  # sanity cap on very wide panes

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        # Thumbnails are sized to the pane, so a horizontal scrollbar would only
        # ever mean the sizing was wrong — never let one appear.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._host = QWidget()
        self._grid = QGridLayout(self._host)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._grid.setSpacing(10)
        self.setWidget(self._host)
        # Source pixmaps are kept so a resize can re-scale from the original
        # rather than repeatedly re-scaling an already-scaled copy.
        self._items: list[tuple[Path, QPixmap]] = []
        self._last_w = -1

    def show_images(self, paths: list[Path]) -> None:
        self._items = []
        for p in paths:
            # A plot helper that bailed out returns None; skip rather than raise.
            if p is None or not Path(p).exists():
                continue
            pix = QPixmap(str(p))
            if pix.isNull():
                continue
            self._items.append((Path(p), pix))
        self._last_w = -1
        self._rebuild()

    def _clear(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _rebuild(self) -> None:
        self._clear()
        if not self._items:
            empty = QLabel("No figures yet — run this analysis to populate the panel.")
            empty.setStyleSheet("color:#6c7086; padding: 24px;")
            self._grid.addWidget(empty, 0, 0)
            return
        sp = self._grid.spacing()
        # Reserve the vertical scrollbar unconditionally: laying out to the full
        # viewport makes the content tall enough to summon the scrollbar, which
        # then narrows the viewport and clips the right-hand column.
        margins = self._grid.contentsMargins()
        reserve = (self.verticalScrollBar().sizeHint().width() + 4
                   + margins.left() + margins.right())
        avail = max(240, self.viewport().width() - reserve)
        self._last_w = self.viewport().width()
        # The label's border is drawn *inside* the widget, so each thumbnail
        # occupies its pixmap plus the frame — budget for it when dividing the
        # width, or the last column overflows by exactly the border.
        pad = 2 * _ClickableThumbnail.BORDER_PX
        cols = max(1, (avail + sp) // (self._MIN_THUMB_W + pad + sp))
        cols = int(min(cols, len(self._items)))
        thumb_w = int((avail - sp * (cols - 1)) / cols) - pad
        thumb_w = max(200, min(self._MAX_THUMB_W, thumb_w))
        for i, (path, pix) in enumerate(self._items):
            w = min(thumb_w, pix.width())   # never upscale past native resolution
            scaled = pix.scaledToWidth(int(w), Qt.TransformationMode.SmoothTransformation)
            lab = _ClickableThumbnail(path)
            lab.setPixmap(scaled)
            lab.setFixedSize(scaled.width() + pad, scaled.height() + pad)
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid.addWidget(lab, i // cols, i % cols)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Re-scaling every pixmap on every pixel of a drag is wasteful; only
        # redo the layout once the width has moved appreciably.
        if abs(self.viewport().width() - self._last_w) > 12:
            self._rebuild()


def _open_in_file_manager(path: Path) -> None:
    """Open ``path`` in the OS file manager (or its parent, if it is a file)."""
    target = Path(path)
    if target.is_file():
        target = target.parent
    if sys.platform.startswith("win"):
        os.startfile(str(target))  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.run(["open", str(target)], check=False)
    else:
        subprocess.run(["xdg-open", str(target)], check=False)


# ── Result panel: figures + optional view dropdown + export buttons ──────────

class _ResultPanel(QWidget):
    """An image strip with Export-figure / Export-data buttons and an optional
    view dropdown (used by the learning-curve tab to switch metric views)."""

    def __init__(self, views: dict[str, str] | None = None,
                 title: str = "Results") -> None:
        super().__init__()
        self._view_labels = dict(views or {})   # view key -> display label
        self._images_by_view: dict[str | None, list[Path]] = {}
        self._csv_tables: dict[str, Path] = {}
        self._folders: dict[str, Path] = {}     # label -> on-disk folder for this run

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        head = QHBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("color:#89b4fa; font-weight:bold; font-size:14px;")
        hint = QLabel("Click a figure to open it full size")
        hint.setStyleSheet("color:#6c7086; font-size:11px;")
        head.addWidget(title_lbl)
        head.addStretch(1)
        head.addWidget(hint)
        lay.addLayout(head)

        # The view row is always built, even when no views are known up front:
        # analyses whose views depend on what the run produced (discrimination's
        # feature families, generalization's derived analyses) fill it in from
        # set_views().  Hidden while there is nothing to switch between.
        top = QHBoxLayout()
        self._view_lbl = QLabel("View:")
        self._view_combo = QComboBox()
        self._view_combo.currentIndexChanged.connect(lambda _i: self._show_current())
        top.addWidget(self._view_lbl)
        top.addWidget(self._view_combo, 1)
        lay.addLayout(top)
        self._view_row_widgets = (self._view_lbl, self._view_combo)
        self._set_view_items(list(self._view_labels))

        self._strip = _ImageStrip()
        lay.addWidget(self._strip, 1)

        # Data-table picker (only shown when a tab exposes more than one table,
        # e.g. Cross-Project's accuracy vs. speed or the behaviorscape tables).
        data_row = QHBoxLayout()
        self._data_lbl = QLabel("Data:")
        self._data_combo = QComboBox()
        self._data_combo.setToolTip("Choose which underlying data table to export or copy.")
        self._data_combo.currentIndexChanged.connect(lambda _i: self._sync_data_buttons())
        data_row.addWidget(self._data_lbl)
        data_row.addWidget(self._data_combo, 1)
        self._data_row_widgets = (self._data_lbl, self._data_combo)
        for _w in self._data_row_widgets:
            _w.setVisible(False)  # shown only once >1 table is available
        lay.addLayout(data_row)

        btn_row = QHBoxLayout()
        # Straight to the files this tab was built from: every figure, CSV and
        # intermediate the run wrote for THIS analysis lives in that folder, and
        # hunting for it under the timestamped run directory is a chore.
        self._folder_btn = QPushButton("Open Data Folder")
        self._folder_btn.setToolTip("Open the folder holding this tab's run output.")
        self._folder_btn.clicked.connect(self._open_folder)
        self._folder_btn.setEnabled(False)
        self._png_btn = QPushButton("Export Figure (PNG)…")
        self._png_btn.clicked.connect(self._export_figure)
        self._csv_btn = QPushButton("Export Data (CSV)…")
        self._csv_btn.clicked.connect(self._export_data)
        self._copy_btn = QPushButton("Copy Data (clipboard)")
        self._copy_btn.setToolTip(
            "Copy the selected data table to the clipboard as tab-separated values — "
            "paste straight into Excel, Prism, GraphPad, or Origin.")
        self._copy_btn.clicked.connect(self._copy_data)
        self._png_btn.setEnabled(False)
        self._csv_btn.setEnabled(False)
        self._copy_btn.setEnabled(False)
        btn_row.addWidget(self._folder_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._png_btn)
        btn_row.addWidget(self._copy_btn)
        btn_row.addWidget(self._csv_btn)
        lay.addLayout(btn_row)
        self._sync_data_buttons()

    # ── data wiring ──
    @staticmethod
    def _as_tables(csv_path: "Path | dict[str, Path] | None") -> dict[str, Path]:
        if csv_path is None:
            return {}
        if isinstance(csv_path, dict):
            return {str(k): Path(v) for k, v in csv_path.items()}
        return {"data": Path(csv_path)}

    def set_simple(self, images: list[Path],
                   csv_path: "Path | dict[str, Path] | None",
                   folder: "Path | dict[str, Path] | None" = None) -> None:
        self._images_by_view = {None: list(images)}
        self._set_view_items([])
        self._set_tables(csv_path)
        self.set_folder(folder)
        self._show_current()

    def set_views(self, images_by_view: dict[str, list[Path]],
                  csv_path: "Path | dict[str, Path] | None",
                  folder: "Path | dict[str, Path] | None" = None) -> None:
        self._images_by_view = dict(images_by_view)
        self._set_view_items(list(images_by_view))
        self._set_tables(csv_path)
        self.set_folder(folder)
        self._show_current()

    def set_folder(self, folder: "Path | dict[str, Path] | None") -> None:
        """Point "Open Data Folder" at this tab's run output.

        Passing ``None`` falls back to the folders the figures and tables
        themselves live in, so a panel populated straight from a worker still
        gets a working button without every caller repeating the path.
        """
        folders: dict[str, Path] = {}
        if isinstance(folder, dict):
            folders = {str(k): Path(v) for k, v in folder.items()}
        elif folder is not None:
            folders = {Path(folder).name: Path(folder)}
        else:
            for p in [*self._all_images(), *self._csv_tables.values()]:
                d = Path(p).parent
                folders.setdefault(d.name, d)
        self._folders = {k: v for k, v in folders.items() if v.is_dir()}
        self._folder_btn.setEnabled(bool(self._folders))
        self._folder_btn.setToolTip(
            "\n".join(str(v) for v in self._folders.values())
            if self._folders else "Run this analysis to produce output files.")

    def _all_images(self) -> list[Path]:
        return [p for imgs in self._images_by_view.values() for p in imgs if p]

    def _open_folder(self) -> None:
        """One folder opens directly; several offer a menu (e.g. Generalization
        writes its agreement, time-budget and calibration output side by side)."""
        if not self._folders:
            return
        if len(self._folders) == 1:
            _open_in_file_manager(next(iter(self._folders.values())))
            return
        menu = QMenu(self)
        for label, path in self._folders.items():
            act = menu.addAction(label)
            act.setToolTip(str(path))
            act.triggered.connect(lambda _checked=False, p=path: _open_in_file_manager(p))
        menu.exec(self._folder_btn.mapToGlobal(self._folder_btn.rect().bottomLeft()))

    def _set_view_items(self, keys: list[str]) -> None:
        """Rebuild the view dropdown, keeping the current selection when it survives."""
        prior = self._view_combo.currentData()
        self._view_combo.blockSignals(True)
        self._view_combo.clear()
        for key in keys:
            self._view_combo.addItem(self._view_labels.get(key, str(key)), userData=key)
        idx = self._view_combo.findData(prior) if prior is not None else -1
        if idx >= 0:
            self._view_combo.setCurrentIndex(idx)
        self._view_combo.blockSignals(False)
        for w in self._view_row_widgets:
            w.setVisible(len(keys) > 1)

    def _set_tables(self, csv_path: "Path | dict[str, Path] | None") -> None:
        # Keep only tables whose file actually exists, preserving insertion order.
        self._csv_tables = {k: v for k, v in self._as_tables(csv_path).items()
                            if Path(v).exists()}
        self._data_combo.blockSignals(True)
        self._data_combo.clear()
        for label, path in self._csv_tables.items():
            self._data_combo.addItem(label, userData=str(path))
        self._data_combo.blockSignals(False)
        multi = len(self._csv_tables) > 1
        for w in self._data_row_widgets:
            w.setVisible(multi)

    def _current_images(self) -> list[Path]:
        # With no view items the combo reports None, which is exactly the key
        # set_simple() files its single image list under.
        return self._images_by_view.get(self._view_combo.currentData(), [])

    def _current_table(self) -> "Path | None":
        if not self._csv_tables:
            return None
        data = self._data_combo.currentData()
        if data:
            return Path(data)
        return next(iter(self._csv_tables.values()))

    def _sync_data_buttons(self) -> None:
        tbl = self._current_table()
        ok = tbl is not None and Path(tbl).exists()
        self._csv_btn.setEnabled(ok)
        self._copy_btn.setEnabled(ok)

    def _show_current(self) -> None:
        imgs = self._current_images()
        self._strip.show_images(imgs)
        self._png_btn.setEnabled(bool(imgs))
        self._sync_data_buttons()

    # ── exports ──
    def _export_figure(self) -> None:
        imgs = [p for p in self._current_images() if Path(p).exists()]
        if not imgs:
            return
        if len(imgs) == 1:
            dest, _ = QFileDialog.getSaveFileName(
                self, "Export figure", imgs[0].name, "PNG image (*.png)")
            if dest:
                shutil.copyfile(imgs[0], dest)
                self._notify(f"Saved figure → {dest}")
        else:
            folder = QFileDialog.getExistingDirectory(self, "Export all figures to folder")
            if folder:
                for p in imgs:
                    shutil.copyfile(p, Path(folder) / Path(p).name)
                self._notify(f"Saved {len(imgs)} figures → {folder}")

    def _export_data(self) -> None:
        tbl = self._current_table()
        if not tbl or not Path(tbl).exists():
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export data", Path(tbl).name, "CSV file (*.csv)")
        if dest:
            shutil.copyfile(tbl, dest)
            self._notify(f"Saved data → {dest}")

    def _copy_data(self) -> None:
        """Copy the selected data table to the clipboard as tab-separated values."""
        tbl = self._current_table()
        if not tbl or not Path(tbl).exists():
            return
        try:
            with open(tbl, newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Could not read data: {exc}")
            return
        buf = io.StringIO()
        csv.writer(buf, delimiter="\t", lineterminator="\n").writerows(rows)
        QApplication.clipboard().setText(buf.getvalue())
        n = max(0, len(rows) - 1)
        self._notify(f"Copied {n} rows (tab-separated) to clipboard — paste into your graphing app.")

    def _notify(self, msg: str) -> None:
        win = self.window()
        if isinstance(win, QMainWindow):
            win.statusBar().showMessage(msg, 6000)


# ── Standard analysis-tab layout: settings left, figures right ───────────────

def _explain(text: str) -> QLabel:
    """A wrapped explanatory paragraph for the left settings column.

    Explanations were written for a full-width tab, so hard line breaks are
    stripped: in a ~380 px column they would wrap twice and waste vertical
    space.  Blank lines (paragraph breaks) and bullet indents are preserved.
    """
    out: list[str] = []
    for para in text.split("\n\n"):
        lines = [ln.rstrip() for ln in para.split("\n")]
        joined = ""
        for ln in lines:
            stripped = ln.strip()
            if not joined:
                joined = stripped
            elif ln.startswith(("  ", "\t", "•", "  •")) or stripped.startswith("•"):
                joined += "\n" + ln
            else:
                joined += " " + stripped
        out.append(joined)
    lab = QLabel("\n\n".join(out))
    lab.setWordWrap(True)
    lab.setStyleSheet("color:#a6adc8; font-size:12px;")
    return lab


def _relax_forms(obj) -> None:
    """Let every QFormLayout under ``obj`` survive a narrow column.

    The settings forms were written for a full-width tab, so labels like
    "Seed exemplars (define the behavior):" set a minimum width the ~390 px
    column cannot honour and the row gets clipped.  WrapLongRows drops the
    field onto its own line instead whenever the row will not fit — which is
    also the only display-scaling-safe behaviour here, since the label width
    grows with the user's DPI setting.
    """
    layout = obj.layout() if isinstance(obj, QWidget) else obj
    if layout is None:
        return
    if isinstance(layout, QFormLayout):
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is not None:
            _relax_forms(item.widget())
        elif item.layout() is not None:
            _relax_forms(item.layout())


def _split_tab(left_items: list, panel: QWidget, *, left_chars: int = 64) -> QWidget:
    """Build an analysis tab as *settings/explanation left, results right*.

    The left column is scrollable and stays narrow; the results panel takes
    every remaining pixel, which is the whole point — figures are the output,
    the settings are read once.  The splitter is user-draggable and the left
    side is collapsible, so the figures can be given the entire tab.

    ``left_chars`` sizes that column in characters rather than pixels, and the
    result is floored at whatever the widgets actually need, so the settings
    stay readable under Windows display scaling instead of clipping.

    ``left_items`` accepts widgets, layouts, or ``(item, stretch)`` tuples for
    anything that should grow with the column (tables, findings boxes).
    """
    host = QWidget()
    ll = QVBoxLayout(host)
    ll.setContentsMargins(2, 2, 8, 2)
    ll.setSpacing(8)
    stretched = False
    for entry in left_items:
        item, stretch = entry if isinstance(entry, tuple) else (entry, 0)
        stretched = stretched or bool(stretch)
        if isinstance(item, QWidget):
            ll.addWidget(item, stretch)
        else:
            ll.addLayout(item, stretch)
    if not stretched:
        ll.addStretch(1)
    _relax_forms(host)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(host)
    # AsNeeded, not AlwaysOff: a wrapped form still has a floor, and clipping a
    # setting the user cannot reach is worse than a scrollbar.
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setMinimumWidth(300)

    split = QSplitter(Qt.Orientation.Horizontal)
    split.addWidget(scroll)
    split.addWidget(panel)
    split.setStretchFactor(0, 0)
    split.setStretchFactor(1, 1)
    split.setCollapsible(0, True)
    split.setCollapsible(1, False)

    def _size_left() -> None:
        # Measured, not hard-coded: character width tracks the font, which
        # tracks the display-scaling factor, and the floor guarantees the
        # settings are never clipped.  Deferred to the event loop because the
        # window's stylesheet — which changes every widget's padding, and so
        # its minimum width — is applied after the tabs are built.
        fm = host.fontMetrics()
        sb = scroll.verticalScrollBar().sizeHint().width()
        wanted = max(fm.averageCharWidth() * left_chars,
                     host.minimumSizeHint().width() + sb + 12)
        split.setSizes([wanted, max(3 * wanted, split.width() - wanted)])

    _size_left()
    QTimer.singleShot(0, _size_left)

    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.addWidget(split)
    return w


# ── Main window ─────────────────────────────────────────────────────────────

class ValidationWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ABEL — Validation & Meta-Analysis Suite")
        # Sized for the split layout: ~390 px of settings plus a results pane
        # wide enough for two side-by-side figures.
        self.resize(1560, 900)
        self._projects: dict[str, ProjectRef] = {}
        self._selected: dict[str, set[str]] = {}     # project_id -> behavior_ids
        self._output_dir: Path | None = None
        # Named setup this window is working in (see abel.validation.workspace).
        # Its runs land in the session's own runs/ folder unless the user picks
        # an output folder by hand.
        self._session_store = SessionStore()
        self._session_name: str = ""
        self._run_session: SessionRecord | None = None   # setup snapshot for the running job
        # Projects a reload could not open (unmounted drive, typically). Carried
        # through every save so an offline drive never erases them from the record.
        self._session_offline: list = []
        self._busy = False
        self._last_run: RunOutputs | None = None
        self._worker: _RunWorker | None = None
        self._pending_pdf = False   # render the summary PDF when this run finishes
        # Feature-demo state
        self._demo_imports = None             # lazily-created ImportService
        self._demo_manifest = None            # manifest of the selected demo project
        self._demo_cancel: list[bool] = [False]
        self._demo_worker = None
        self._demo_preview_dialog = None      # live preview window (kept alive)
        self._demo_out_dir: Path | None = None  # folder of the last exported demo clip

        self._build_ui()
        self.setStyleSheet(_DARK_STYLE)

    # ── construction ──
    def _build_ui(self) -> None:
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_projects_tab(), "Projects")
        # Built before Run All: the suite tab reads the behaviorscape alias table.
        self._bscape_tab = self._build_behaviorscape_tab()
        self._tabs.addTab(self._build_run_all_tab(), "Run All")
        self._tabs.addTab(self._build_learning_curve_tab(), "Learning Curves")
        self._tabs.addTab(self._build_ablation_tab(), "Ablation")
        self._tabs.addTab(self._build_discrimination_tab(), "Discrimination")
        self._tabs.addTab(self._build_generalization_tab(), "Generalization")
        self._tabs.addTab(self._build_al_tab(), "Active Learning")
        self._tabs.addTab(self._build_rare_discovery_tab(), "Rare Discovery")
        self._tabs.addTab(self._build_cross_tab(), "Cross-Project")
        self._tabs.addTab(self._bscape_tab, "Behaviorscape")
        self._tabs.addTab(self._build_video_value_tab(), "Video Features")
        self._tabs.addTab(self._build_throughput_tab(), "Throughput")
        self._tabs.addTab(self._build_feature_demo_tab(), "Feature Demo")
        self._tabs.addTab(self._build_log_tab(), "Log")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        central = QWidget()
        root = QVBoxLayout(central)
        root.addWidget(self._tabs)

        bar = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress_lbl = QLabel("Idle.")
        bar.addWidget(self._progress, 3)
        bar.addWidget(self._progress_lbl, 5)
        root.addLayout(bar)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    def _build_projects_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        split = QSplitter(Qt.Orientation.Horizontal)
        # left: projects
        left = QWidget(); ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Projects"))
        self._proj_list = QListWidget()
        self._proj_list.currentItemChanged.connect(self._on_project_selected)
        self._proj_list.itemDoubleClicked.connect(lambda _it: self._rename_project())
        ll.addWidget(self._proj_list)
        btns = QHBoxLayout()
        add = QPushButton("Add Project(s)…"); add.clicked.connect(self._add_project)
        pren = QPushButton("Rename…")
        pren.setToolTip(
            "Give this project a different display name (or double-click it).\n\n"
            "The new name replaces the original everywhere downstream — every figure\n"
            "title, table, and export. Your project on disk is NOT modified.")
        pren.clicked.connect(self._rename_project)
        rem = QPushButton("Remove"); rem.clicked.connect(self._remove_project)
        btns.addWidget(add); btns.addWidget(pren); btns.addWidget(rem)
        ll.addLayout(btns)

        # Auto-add: scan a directory, add every project, pre-check only behaviors
        # whose existing trained model clears a strength threshold.
        auto_box = QGroupBox("Auto-add from a directory")
        af = QFormLayout(auto_box)
        self._autoadd_metric = QComboBox()
        self._autoadd_metric.addItem("F1", userData="f1")
        self._autoadd_metric.addItem("PR-AUC (avg precision)", userData="pr_auc")
        self._autoadd_metric.setToolTip(
            "Metric used to judge each behavior's existing trained model.\n"
            "F1: interpretable balance of precision/recall at the operating point.\n"
            "PR-AUC: threshold-independent — more robust for rare behaviors.")
        af.addRow("Strength metric:", self._autoadd_metric)
        self._autoadd_thresh = QDoubleSpinBox()
        self._autoadd_thresh.setRange(0.0, 1.0)
        self._autoadd_thresh.setSingleStep(0.05)
        self._autoadd_thresh.setValue(0.50)
        self._autoadd_thresh.setToolTip(
            "Behaviors whose model scores below this are added but left unchecked. "
            "Behaviors with no trained model are also left unchecked.")
        af.addRow("Min model strength:", self._autoadd_thresh)
        scan_btn = QPushButton("Scan directory && add projects…")
        scan_btn.setToolTip("Recursively find every ABEL project under a folder, add them all, "
                            "and check only behaviors whose model clears the threshold.")
        scan_btn.clicked.connect(self._auto_add_directory)
        af.addRow(scan_btn)
        apply_btn = QPushButton("Apply strength filter to loaded projects")
        apply_btn.setToolTip("Re-check behaviors across the already-loaded projects using the "
                             "metric + threshold above (overrides current checkbox selection).")
        apply_btn.clicked.connect(self._apply_strength_to_loaded)
        af.addRow(apply_btn)
        ll.addWidget(auto_box)

        split.addWidget(left)
        # right: behaviors
        right = QWidget(); rl = QVBoxLayout(right)
        rl.addWidget(QLabel("Behaviors (checked = included)"))
        self._beh_list = QListWidget()
        self._beh_list.itemChanged.connect(self._on_behavior_toggled)
        self._beh_list.itemDoubleClicked.connect(lambda _it: self._rename_behavior())
        rl.addWidget(self._beh_list)

        beh_btns = QHBoxLayout()
        ren = QPushButton("Rename…")
        ren.setToolTip(
            "Give this behavior a different display name (or double-click it).\n\n"
            "The new name is used everywhere downstream — every figure, table and export —\n"
            "as if it were the behavior's real name. Your project on disk is NOT modified.\n\n"
            "Rename behaviors to MATCH across projects (e.g. 'Grooming' → 'Groom') and the\n"
            "generalization figure pools them into a single bar.")
        ren.clicked.connect(self._rename_behavior)
        reset = QPushButton("Reset name")
        reset.setToolTip("Restore this behavior's original name from the project.")
        reset.clicked.connect(self._reset_behavior_name)
        beh_btns.addWidget(ren); beh_btns.addWidget(reset); beh_btns.addStretch(1)
        rl.addLayout(beh_btns)

        hint = QLabel("Renames are display-only and last for this session. Matching names "
                      "across projects merges them in the Generalization figure.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #78909C; font-size: 11px;")
        rl.addWidget(hint)
        split.addWidget(right)
        split.setSizes([400, 600])
        lay.addWidget(split, 1)

        # saved setup: projects loaded, behaviors checked, renames — and where runs go
        sess_box = QGroupBox("Validation session (saved setup)")
        sv = QVBoxLayout(sess_box)
        self._session_lbl = QLabel("No session — runs go to the unfiled folder.")
        self._session_lbl.setWordWrap(True)
        self._session_lbl.setToolTip(
            "A session records which projects are loaded, which behaviors are checked,\n"
            "and every rename you applied — and files each run inside its own folder.")
        sv.addWidget(self._session_lbl)
        srow = QHBoxLayout()
        for text, tip, slot in (
            ("Save", "Save this setup back to the current session.",
             lambda: self._save_session()),
            ("Save As…", "Save this setup under a new session name.",
             lambda: self._save_session(as_new=True)),
            ("Load…", "Reload a saved setup: projects, checked behaviors and renames.",
             self._load_session),
            ("Open Data Folder", "Open this session's folder (setup + all its runs).",
             self._open_session_folder),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            srow.addWidget(b)
        sv.addLayout(srow)
        lay.addWidget(sess_box)

        # shared holdout settings + output dir
        box = QGroupBox("Held-out evaluation (shared)")
        form = QFormLayout(box)
        self._min_conf = QDoubleSpinBox(); self._min_conf.setRange(0.0, 1.0)
        self._min_conf.setSingleStep(0.05); self._min_conf.setValue(1.0)
        form.addRow("Min reviewer confidence (held-out only):", self._min_conf)
        self._test_size = QDoubleSpinBox(); self._test_size.setRange(0.1, 0.5)
        self._test_size.setSingleStep(0.05); self._test_size.setValue(0.25)
        form.addRow("Held-out fraction of subjects/sessions:", self._test_size)
        self._holdout_seed = QSpinBox(); self._holdout_seed.setRange(0, 99999); self._holdout_seed.setValue(42)
        form.addRow("Holdout seed:", self._holdout_seed)
        out_row = QHBoxLayout()
        self._out_lbl = QLabel(str(workspace_root() / "unfiled_runs"))
        self._out_lbl.setToolTip(
            "Where run results are written. Save a session and runs go into that "
            "session's runs/ folder instead.")
        out_btn = QPushButton("Choose output folder…"); out_btn.clicked.connect(self._choose_output)
        out_row.addWidget(self._out_lbl, 1); out_row.addWidget(out_btn)
        form.addRow("Output folder:", self._wrap(out_row))
        lay.addWidget(box)
        return w

    # ── Run All: one button, publication settings, one consolidated report ──
    def _build_run_all_tab(self) -> QWidget:
        left: list = []

        intro = _explain(
            "Run the whole validation suite in one go, at fixed publication settings.\n"
            "Pick your projects and behaviors on the Projects tab, tick the analyses you\n"
            "want below, and click Run. Every analysis shares ONE held-out split, so the\n"
            "numbers are directly comparable across sections. When it finishes you get a\n"
            "consolidated report (findings in plain language + the headline figures), and\n"
            "Export Everything writes all figures and data CSVs to a folder of your choice.")
        left.append(intro)

        abox = QGroupBox("Analyses to include")
        av = QVBoxLayout(abox)
        self._suite_checks: dict[str, QCheckBox] = {}
        _tips = {
            ANALYSIS_THROUGHPUT:
                "Times feature extraction and training on one session per project. "
                "Extraction is a slow full rebuild. Dense inference is deliberately "
                "excluded here — it rewrites the project's traces; run it from the "
                "Throughput tab if you want it.",
            ANALYSIS_AL_CURVE:
                "The most expensive analysis: it retrains at every acquisition step, "
                "for two arms, for every seed.",
            ANALYSIS_BEHAVIORSCAPE:
                "Pools feature importance across every project, so it is most "
                "informative with 2+ projects loaded.",
        }
        for key in FULL_SUITE:
            cb = QCheckBox(ANALYSIS_LABELS.get(key, key))
            cb.setChecked(True)
            if key in _tips:
                cb.setToolTip(_tips[key])
            self._suite_checks[key] = cb
            av.addWidget(cb)
        row = QHBoxLayout()
        all_btn = QPushButton("Select all")
        all_btn.clicked.connect(
            lambda: [cb.setChecked(True) for cb in self._suite_checks.values()])
        none_btn = QPushButton("Select none")
        none_btn.clicked.connect(
            lambda: [cb.setChecked(False) for cb in self._suite_checks.values()])
        row.addWidget(all_btn); row.addWidget(none_btn); row.addStretch()
        av.addLayout(row)
        left.append(abox)

        pbox = QGroupBox("Publication settings (fixed)")
        pv = QVBoxLayout(pbox)
        preset_lbl = QLabel(preset_description(publication_config()))
        preset_lbl.setWordWrap(True)
        preset_lbl.setStyleSheet(
            "color:#a6adc8; font-family:'Cascadia Code','Consolas',monospace; "
            "font-size:11px;")
        pv.addWidget(preset_lbl)
        note = QLabel(
            "These are deliberately not editable — the point of this tab is that the run "
            "is reproducible and defensible without anyone tuning it. The shared held-out "
            "settings on the Projects tab are still honoured. Use the individual analysis "
            "tabs if you need to change a setting.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#6c7086; font-size:11px;")
        pv.addWidget(note)
        left.append(pbox)

        run = QPushButton("Run Full Validation Suite")
        run.setObjectName("runBtn")
        run.clicked.connect(self._run_full_suite)
        left.append(run)
        self._suite_run_btn = run

        self._suite_status = QLabel("")
        self._suite_status.setWordWrap(True)
        self._suite_status.setStyleSheet("color:#a6adc8;")
        left.append(self._suite_status)

        find_lbl = QLabel("Key findings")
        find_lbl.setStyleSheet("color:#89b4fa; font-weight:bold;")
        left.append(find_lbl)
        self._suite_findings = QTextEdit()
        self._suite_findings.setReadOnly(True)
        self._suite_findings.setMinimumHeight(180)
        self._suite_findings.setPlaceholderText(
            "Findings appear here once the run completes.")
        left.append((self._suite_findings, 1))

        btn_row = QHBoxLayout()
        self._suite_pdf_btn = QPushButton("Open Report (PDF)…")
        self._suite_pdf_btn.setToolTip(
            "The consolidated summary: findings in plain language, the key table per "
            "analysis, and the headline figures.")
        self._suite_pdf_btn.clicked.connect(self._open_summary_pdf)
        self._suite_export_btn = QPushButton("Export Everything to Folder…")
        self._suite_export_btn.setToolTip(
            "Write the report PDF, every figure (figures/) and every data CSV (data/) "
            "to a folder you choose, with an INDEX.csv describing each file.")
        self._suite_export_btn.clicked.connect(self._export_bundle)
        for b in (self._suite_pdf_btn, self._suite_export_btn):
            b.setEnabled(False)
        btn_row.addWidget(self._suite_pdf_btn)
        btn_row.addWidget(self._suite_export_btn)
        left.append(btn_row)

        self._suite_panel = _ResultPanel(title="Headline figures")
        return _split_tab(left, self._suite_panel, left_chars=70)

    def _selected_suite_analyses(self) -> list[str]:
        return [k for k in FULL_SUITE if self._suite_checks[k].isChecked()]

    def _run_full_suite(self) -> None:
        analyses = self._selected_suite_analyses()
        if not analyses:
            QMessageBox.warning(self, "Nothing to run",
                                "Tick at least one analysis to include.")
            return
        behaviors = self._collect_behaviors()
        if not behaviors:
            QMessageBox.warning(self, "Nothing selected",
                                "Add a project and check at least one behavior on the "
                                "Projects tab.")
            return
        n_beh = sum(len(v) for v in behaviors.values())
        if QMessageBox.question(
            self, "Run full validation suite",
            f"Run {len(analyses)} analyses over {len(behaviors)} project(s) and "
            f"{n_beh} behavior(s) at publication settings?\n\n"
            "This retrains ABEL's real classifier many times and can take hours.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        ) != QMessageBox.StandardButton.Yes:
            return

        cfg = publication_config(
            analyses=analyses,
            output_root=str(self._output_dir) if self._output_dir else "",
            # The Projects tab's held-out settings are shared across every tab, so a
            # deliberate change there must survive the preset.
            min_confidence=self._min_conf.value(),
            holdout_test_size=self._test_size.value(),
            holdout_seed=self._holdout_seed.value(),
            # Reuse any behavior-pooling aliases the user set up for behaviorscape.
            bscape_alias_map=self._bscape_alias_map(),
        )
        self._pending_pdf = True
        self._suite_findings.clear()
        self._suite_status.setText("Running the full suite… progress is at the bottom of "
                                   "the window; the Log tab has the detail.")
        for b in (self._suite_pdf_btn, self._suite_export_btn):
            b.setEnabled(False)
        self._run(analyses, cfg=cfg)

    def _findings_to_html(self, items: list) -> str:
        if not items:
            return "<p style='color:#6c7086;'>No findings were derived.</p>"
        colours = {KIND_CAVEAT: "#f9e2af", KIND_WARNING: "#f38ba8"}
        tags = {KIND_CAVEAT: "CAVEAT", KIND_WARNING: "WARNING"}
        out: list[str] = []
        current = None
        for f in items:
            if f.analysis != current:
                current = f.analysis
                out.append(f"<h3 style='color:#89b4fa;margin:10px 0 4px;'>"
                           f"{html.escape(current)}</h3>")
            colour = colours.get(f.kind, "#cdd6f4")
            tag = tags.get(f.kind)
            prefix = (f"<b style='color:{colour};'>{tag} — </b>" if tag else "")
            out.append(
                f"<p style='margin:0 0 8px;'>"
                f"<span style='color:{colour};'>{prefix}"
                f"<b>{html.escape(f.headline)}</b></span><br/>"
                f"<span style='color:#a6adc8;font-size:11px;'>"
                f"{html.escape(f.detail)}</span></p>")
        return "".join(out)

    def _populate_suite_tab(self, out: RunOutputs) -> None:
        """Findings + headline figures + the export buttons, after a run."""
        self._suite_findings.setHtml(self._findings_to_html(out.findings))
        headline = pdf_report.headline_figures(out.run_dir)
        # The whole run, plus the two folders people actually go looking for:
        # the Prism-ready pivots and the meta summary tables.
        self._suite_panel.set_simple(headline, out.run_dir / "findings.csv", folder={
            "Run folder": out.run_dir,
            "Prism tables": out.run_dir / "prism",
            "Meta summaries": out.run_dir / "summary",
        })
        self._suite_export_btn.setEnabled(True)
        self._suite_pdf_btn.setEnabled(out.summary_html is not None)
        self._report_figures("Full suite", headline)

    def _ensure_pdf(self) -> "Path | None":
        """Render the summary PDF if it isn't on disk yet. Must run on the GUI thread."""
        out = self._last_run
        if out is None or out.summary_html is None:
            return None
        if out.pdf_path and Path(out.pdf_path).exists():
            return out.pdf_path
        pdf_path = out.run_dir / "ABEL_validation_report.pdf"
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            pdf_report.render_pdf(out.summary_html, pdf_path)
            out.pdf_path = pdf_path
            self._log_msg(f"Summary PDF → {pdf_path}")
            return pdf_path
        except Exception as exc:  # noqa: BLE001 — HTML fallback is always there
            self._log_msg(f"PDF RENDER FAILED: {exc}")
            return None
        finally:
            QApplication.restoreOverrideCursor()

    def _open_summary_pdf(self) -> None:
        if not self._last_run:
            QMessageBox.information(self, "No report", "Run the suite first.")
            return
        pdf = self._ensure_pdf()
        target = pdf or (self._last_run.summary_html if self._last_run else None)
        if not target:
            QMessageBox.warning(self, "No report",
                                "The summary report was not produced — see the Log tab.")
            return
        if pdf is None:
            QMessageBox.information(
                self, "PDF unavailable",
                "The PDF could not be rendered, so the summary HTML will open instead — "
                "print it to PDF from your browser. See the Log tab for the reason.")
        self._open_path(Path(target))

    def _export_bundle(self) -> None:
        if not self._last_run:
            QMessageBox.information(self, "Nothing to export", "Run the suite first.")
            return
        dest = QFileDialog.getExistingDirectory(self, "Export everything to folder")
        if not dest:
            return
        # Render the PDF first so it lands in the bundle rather than being missing
        # from the one folder the user is going to hand to someone else.
        self._ensure_pdf()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            res = bundle.export_bundle(self._last_run.run_dir, dest)
        except Exception as exc:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            self._log_msg(f"EXPORT FAILED: {traceback.format_exc()}")
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self._log_msg(res.summary())
        self._suite_status.setText(res.summary())
        self.statusBar().showMessage(res.summary(), 10000)
        if QMessageBox.question(
            self, "Export complete",
            f"{res.summary()}\n\nOpen the folder?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        ) == QMessageBox.StandardButton.Yes:
            self._open_path(Path(dest))

    @staticmethod
    def _open_path(path: Path) -> None:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def _build_learning_curve_tab(self) -> QWidget:
        box = QGroupBox("Optimal-clips learning curve settings")
        form = QFormLayout(box)
        self._lc_sizes = QLineEdit("10, 25, 50, 100, 200, all")
        form.addRow("Clip-size schedule (per step):", self._lc_sizes)
        self._lc_seeds = QSpinBox(); self._lc_seeds.setRange(1, 20); self._lc_seeds.setValue(5)
        form.addRow("Seeds per point:", self._lc_seeds)
        self._lc_negpolicy = QComboBox(); self._lc_negpolicy.addItems(["all", "ratio"])
        form.addRow("Negatives policy:", self._lc_negpolicy)
        self._lc_negratio = QDoubleSpinBox(); self._lc_negratio.setRange(0.5, 20.0); self._lc_negratio.setValue(3.0)
        form.addRow("Negatives per positive (ratio mode):", self._lc_negratio)
        run = QPushButton("Run Learning Curves"); run.setObjectName("runBtn")
        run.clicked.connect(lambda: self._run([ANALYSIS_LEARNING_CURVE]))
        self._lc_panel = _ResultPanel(views=LEARNING_CURVE_VIEWS,
                                      title="Learning curves")
        self._lc_run_btn = run
        return _split_tab([box, run], self._lc_panel)

    def _build_ablation_tab(self) -> QWidget:
        box = QGroupBox("Ablation settings")
        form = QFormLayout(box)
        self._abl_seeds = QSpinBox(); self._abl_seeds.setRange(1, 20); self._abl_seeds.setValue(5)
        form.addRow("Seeds per config:", self._abl_seeds)
        self._abl_budgets = QLineEdit("50, all")
        self._abl_budgets.setToolTip(
            "Clip budget(s) to run the ablation at. A low budget (e.g. 50) trains every\n"
            "config on only that many labeled positives, then 'all' uses the full pool.\n"
            "Comparing them shows where each enhancement adds the most value — regularizers\n"
            "like adaptive complexity and augmentation pay off most in the low-data regime.")
        form.addRow("Clip budget(s):", self._abl_budgets)
        info = _explain(
            "How to read this: the comparison point is a pose-only Baseline with every\n"
            "enhancement off. Each bar adds ONE enhancement on its own — video features,\n"
            "probability calibration, adaptive model complexity, feature augmentation, and\n"
            "co-occurring labels (when the project uses them) — and the final bar enables ALL\n"
            "of them together. Bars show ΔF1 vs. the baseline; positive ⇒ that feature helps.\n"
            "Error bars are 95% CIs across seeds; FADED bars overlap 0 (not distinguishable\n"
            "from baseline — a small ± there is noise, not harm). One chart per clip budget.")
        form.addRow(info)
        run = QPushButton("Run Ablation"); run.setObjectName("runBtn")
        run.clicked.connect(lambda: self._run([ANALYSIS_ABLATION]))
        self._abl_panel = _ResultPanel(title="Ablation")
        self._abl_run_btn = run
        return _split_tab([box, run], self._abl_panel)

    def _build_discrimination_tab(self) -> QWidget:
        box = QGroupBox("Behavior discrimination settings")
        form = QFormLayout(box)
        self._disc_seeds = QSpinBox(); self._disc_seeds.setRange(1, 20); self._disc_seeds.setValue(3)
        form.addRow("Seeds per feature set:", self._disc_seeds)
        self._disc_max_pairs = QSpinBox(); self._disc_max_pairs.setRange(1, 100)
        self._disc_max_pairs.setValue(15)
        self._disc_max_pairs.setToolTip(
            "Cap on how many behavior pairs to test. Every pair of the behaviors you\n"
            "checked on the Projects tab is a separate A-vs-B model, so the count grows\n"
            "as n(n-1)/2. When the cap bites, the pairs whose feature-space centroids sit\n"
            "closest together (the ones most likely to be confused) are kept.")
        form.addRow("Max behavior pairs:", self._disc_max_pairs)
        info = _explain(
            "How this differs from Ablation: Ablation asks a DETECTION question — can we\n"
            "find behavior X against everything else? Because 'everything else' is mostly\n"
            "easy negatives, a feature family can look useless there while doing the job\n"
            "that matters: telling two SIMILAR behaviors apart (Freeze vs Groom; Sniff vs Eat).\n\n"
            "This tab asks that DISCRIMINATION question. For every behavior pair it trains a\n"
            "binary A-vs-B model on just those clips, once per feature family (pose only →\n"
            "+ video → + social), all sharing the same clips and seed, and scores separability\n"
            "with ROC-AUC.\n\n"
            "Read the LANDSCAPE figure first — it is the whole run, every pair of every\n"
            "assay, in two panels. LEFT: each point is one behavior pair, placed by how much\n"
            "error pose alone leaves (x, right = pose confuses it) against the share of that\n"
            "error the best feature family removes (y), coloured by WHICH family. Pairs in\n"
            "the shaded band are already solved by pose. Hollow points mean no family\n"
            "measurably helps. RIGHT: a volcano over every pair × family — how big the gain\n"
            "was (x) against how reproducible it was across seeds (y), so a large-but-noisy\n"
            "gain is visibly different from a small-but-rock-solid one.\n\n"
            "The per-project matrix views behind it are the per-assay detail: LEFT = pose-only\n"
            "separability (dark = the model confuses that pair). RIGHT = the share of the pose\n"
            "baseline's REMAINING error each family removes (red = it disambiguates the pair).\n"
            "Hatched cells were never trained — past the max-pairs cap, or too few clips.")
        form.addRow(info)
        run = QPushButton("Run Discrimination"); run.setObjectName("runBtn")
        run.clicked.connect(lambda: self._run([ANALYSIS_DISCRIMINATION]))
        self._disc_panel = _ResultPanel(title="Discrimination")
        self._disc_run_btn = run
        return _split_tab([box, run], self._disc_panel)

    def _build_generalization_tab(self) -> QWidget:
        box = QGroupBox("Generalization / human-agreement settings")
        form = QFormLayout(box)
        self._gen_seeds = QSpinBox(); self._gen_seeds.setRange(1, 20); self._gen_seeds.setValue(3)
        form.addRow("Seeds:", self._gen_seeds)
        info = _explain(
            "Trains on training-pool subjects, evaluates on held-out subjects.\n"
            "Reports F1 + Cohen's κ vs. held-out reviewed labels.\n\n"
            "This run also produces two further analyses from the SAME held-out predictions\n"
            "(so they cost no extra training) — pick them from the view dropdown below:\n"
            "  • Biological readout — does the model reproduce the measure a scorer would\n"
            "    report? Per-session time-in-behavior and bout counts, model vs. reviewed,\n"
            "    with Pearson r, Lin's CCC, R² and Bland-Altman bias / limits of agreement.\n"
            "  • Calibration — a reliability diagram with ECE and Brier score, i.e. whether\n"
            "    a predicted probability of 0.8 really means right 80% of the time.")
        form.addRow(info)
        run = QPushButton("Run Generalization"); run.setObjectName("runBtn")
        run.clicked.connect(lambda: self._run([ANALYSIS_GENERALIZATION]))
        self._gen_panel = _ResultPanel(title="Generalization")
        self._gen_run_btn = run
        return _split_tab([box, run], self._gen_panel)

    def _build_al_tab(self) -> QWidget:
        box = QGroupBox("Active learning vs. random selection settings")
        form = QFormLayout(box)
        self._al_seeds = QSpinBox(); self._al_seeds.setRange(1, 20); self._al_seeds.setValue(3)
        form.addRow("Seeds:", self._al_seeds)
        self._al_seed_pos = QSpinBox(); self._al_seed_pos.setRange(1, 50); self._al_seed_pos.setValue(5)
        form.addRow("Seed-example positives (warm start):", self._al_seed_pos)
        self._al_k0 = QSpinBox(); self._al_k0.setRange(2, 200); self._al_k0.setValue(20)
        form.addRow("Initial seed clips (k0):", self._al_k0)
        self._al_batch = QSpinBox(); self._al_batch.setRange(1, 200); self._al_batch.setValue(15)
        form.addRow("Acquisition batch size:", self._al_batch)
        self._al_max = QSpinBox(); self._al_max.setRange(20, 5000); self._al_max.setValue(200)
        form.addRow("Max clips reviewed (budget):", self._al_max)
        self._al_acq = QComboBox(); self._al_acq.addItems(["probability", "uncertainty"])
        form.addRow("AL acquisition rule:", self._al_acq)
        info = _explain(
            "Compares ABEL's candidate-ranked review (acquire likely positives) against\n"
            "random clip review, on the same held-out set. Both warm-start identically.\n"
            "Headline: positive clips discovered per labeling effort.")
        form.addRow(info)
        run = QPushButton("Run Active Learning vs. Random"); run.setObjectName("runBtn")
        run.clicked.connect(lambda: self._run([ANALYSIS_AL_CURVE]))
        self._al_panel = _ResultPanel(title="Active learning vs. random")
        self._al_run_btn = run
        return _split_tab([box, run], self._al_panel)

    def _build_rare_discovery_tab(self) -> QWidget:
        box = QGroupBox("Rare-behavior discovery (clip hunting) settings")
        form = QFormLayout(box)
        # Checkbox text cannot word-wrap, so a long label sets a minimum width the
        # settings column cannot honour — keep the labels terse and put the
        # explanation in the tooltip.
        self._rare_auto = QCheckBox("Auto-target the rarest behavior per project")
        self._rare_auto.setChecked(True)
        self._rare_auto.setToolTip(
            "Per project, rank behaviors by rarity first (cheap) and hunt only "
            "the rarest one.\n"
            "With several projects checked, the rarity pass runs first for each "
            "project — it reads the dense bout detections only, no model fitting — "
            "and the whole discovery/effort-to-quality budget then goes to that "
            "project's rarest behavior before moving to the next project.\n"
            "If the rarest behavior has too few confirmed positives to "
            "cross-validate, the next-rarest is used and the report says so.\n"
            "Uncheck to hunt every behavior you have checked, in every project.")
        form.addRow(self._rare_auto)
        self._rare_seeds = QSpinBox(); self._rare_seeds.setRange(1, 20); self._rare_seeds.setValue(5)
        form.addRow("Seeds (cross-validation folds):", self._rare_seeds)
        self._rare_seed_pos = QSpinBox(); self._rare_seed_pos.setRange(2, 100); self._rare_seed_pos.setValue(8)
        form.addRow("Seed exemplars (define the behavior):", self._rare_seed_pos)
        self._rare_budget = QSpinBox(); self._rare_budget.setRange(50, 5000); self._rare_budget.setValue(400)
        form.addRow("Review budget (clips):", self._rare_budget)
        self._rare_rarity = QCheckBox("Rarity-scaling figure")
        self._rare_rarity.setToolTip("Effort vs. prevalence — how the hunt scales as the "
                                     "target behavior gets rarer.")
        self._rare_rarity.setChecked(True)
        form.addRow(self._rare_rarity)
        self._rare_fullpool = QCheckBox("Full segment pool at deployment rarity")
        self._rare_fullpool.setToolTip(
            "Recommended — the reviewed pool is ~12× enriched for the target, so "
            "scoring against it flatters every method equally and understates how "
            "much work deployment rarity really takes.")
        self._rare_fullpool.setChecked(True)
        form.addRow(self._rare_fullpool)
        self._rare_quality = QCheckBox("Effort-to-quality figure")
        self._rare_quality.setToolTip(
            "Labeling effort → held-out target-class F1 / PR-AUC. This is the primary "
            "result; it trains a model at every step, so it is the expensive part of "
            "the run.")
        self._rare_quality.setChecked(True)
        form.addRow(self._rare_quality)
        self._rare_exclude = QLineEdit()
        self._rare_exclude.setPlaceholderText("e.g. Freeze, Groom  (comma-separated)")
        form.addRow("Exclude from rarity comparison:", self._rare_exclude)
        info = _explain(
            "Compares ABEL's clip-hunting tools (Essence Miner, Active Learning,\n"
            "UMAP selection) against random clips and whole-video scanning for finding\n"
            "a rare behavior. Cross-validated: the definition is built from the seed\n"
            "exemplars and scored only on held-out positives it never saw. Also reports\n"
            "how rare each behavior is (from dense bout detections). The optional\n"
            "effort-to-quality figure asks the paired question — how much labeling each\n"
            "tool needs before the trained model itself is good (held-out F1 / PR-AUC).\n"
            "Check several projects to also get the combined cross-project panels\n"
            "(mean discovery curve, fold-enrichment and effort saved per project).")
        form.addRow(info)

        # ── two-phase workflow ──
        # Phase 1 is seconds and tells you whether phase 2 (hours) is worth
        # starting: a behavior with eight confirmed examples cannot be
        # cross-validated at all, and finding that out after the hunt has been
        # running overnight is the expensive way to learn it.
        phase = QGroupBox("Two-phase run")
        pl = QVBoxLayout(phase)
        check = QPushButton("1 · Check rarity + examples (fast)")
        check.setToolTip(
            "Reads the dense bout detections and the training set's label column "
            "for every checked project — no model fitting. Reports which behavior "
            "is rarest in each project and whether it has enough confirmed "
            "examples to hunt.")
        check.clicked.connect(self._run_rare_preflight)
        run = QPushButton("2 · Run discovery on the rare behaviors")
        run.setObjectName("runBtn")
        run.clicked.connect(self._run_rare_discovery)
        # Stacked, not side-by-side: the settings column is ~390 px wide and
        # these labels are long enough to be clipped in half of it.
        pl.addWidget(check)
        pl.addWidget(run)
        self._rare_preflight_msg = QLabel(
            "Run the fast check first — it tells you if any project needs more "
            "labeled examples before the hunt is worth starting.")
        self._rare_preflight_msg.setWordWrap(True)
        self._rare_preflight_msg.setStyleSheet("color:#a6adc8;")
        pl.addWidget(self._rare_preflight_msg)
        self._rare_preflight_table = QTableWidget(0, 7)
        self._rare_preflight_table.setHorizontalHeaderLabels([
            "Project", "Behavior", "Rarity rank", "Prevalence",
            "Confirmed examples", "Left to discover", "Verdict"])
        self._rare_preflight_table.setAlternatingRowColors(True)
        self._rare_preflight_table.setMinimumHeight(160)
        self._rare_preflight_table.verticalHeader().setVisible(False)
        self._rare_preflight_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._rare_preflight_table.itemSelectionChanged.connect(
            self._sync_rare_swap_btn)
        pl.addWidget(self._rare_preflight_table)
        # The ranking is a heuristic; the user knows things the files do not (that a
        # behaviour is gated by the design, that the second-rarest is the
        # interesting one).  Let them say so rather than re-running with different
        # exclusions until the table agrees.
        swap_row = QHBoxLayout()
        self._rare_swap_btn = QPushButton("Hunt the selected behavior instead")
        self._rare_swap_btn.setEnabled(False)
        self._rare_swap_btn.setToolTip(
            "Override the automatic pick for that project. The hunt will target the "
            "behavior on the selected row.")
        self._rare_swap_btn.clicked.connect(self._swap_rare_target)
        swap_row.addWidget(self._rare_swap_btn)
        swap_row.addStretch(1)
        pl.addLayout(swap_row)

        self._rare_panel = _ResultPanel(title="Rare-behavior discovery")
        self._rare_run_btn = run
        self._rare_check_btn = check
        self._rare_preflight: list = []
        # The preflight table is the one left-column widget worth extra height.
        return _split_tab([box, (phase, 1)], self._rare_panel, left_chars=72)

    def _selected_preflight_row(self) -> "tuple | None":
        """(ProjectPreflight, BehaviorPreflight) behind the selected table row."""
        tbl = self._rare_preflight_table
        rows = {i.row() for i in tbl.selectedIndexes()}
        if len(rows) != 1 or not self._rare_preflight:
            return None
        row = rows.pop()
        # The table is the flattened preflight_rows() list, in the same order.
        flat = [(p, b) for p in self._rare_preflight for b in p.behaviors]
        return flat[row] if 0 <= row < len(flat) else None

    def _sync_rare_swap_btn(self) -> None:
        sel = self._selected_preflight_row()
        self._rare_swap_btn.setEnabled(
            sel is not None and sel[1].runnable() and not self._busy)

    def _swap_rare_target(self) -> None:
        from abel.validation.analyses import rare_discovery as rd  # noqa: PLC0415
        sel = self._selected_preflight_row()
        if sel is None:
            return
        proj, beh = sel
        if not beh.runnable():
            QMessageBox.warning(
                self, "Cannot hunt this behavior",
                f"{beh.behavior_name} has too few confirmed examples "
                f"({beh.n_labeled}) to cross-validate.\n\n{beh.note}")
            return
        if beh.status == rd.PREFLIGHT_WARN and QMessageBox.question(
            self, "Thin evidence",
            f"{beh.behavior_name} leaves only {beh.n_held_out} examples to "
            f"discover — the curves will be noisy.\n\nHunt it anyway?",
        ) != QMessageBox.StandardButton.Yes:
            return
        proj.target_override = beh.behavior_id
        self._log_msg(f"Target override — {proj.project_name}: hunting "
                      f"{beh.behavior_name}")
        self._on_preflight_done(self._rare_preflight)   # repaint "← will be hunted"

    # ── phase 1: the cheap rarity + evidence check ──
    def _run_rare_preflight(self) -> None:
        if self._busy:
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return
        behaviors = self._collect_behaviors()
        if not behaviors:
            QMessageBox.warning(self, "Nothing selected",
                                "Add a project and check at least one behavior.")
            return
        cfg = self._build_config([ANALYSIS_RARE_DISCOVERY])
        projects = [self._projects[pid] for pid in behaviors]
        self._set_busy(True)
        self._log_msg(f"Rarity check: {[p.name for p in projects]}")
        worker = _PreflightWorker(projects, behaviors, cfg)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_preflight_done)
        worker.signals.error.connect(self._on_error)
        self._worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_preflight_done(self, results: list) -> None:
        from abel.validation.analyses import rare_discovery as rd  # noqa: PLC0415
        self._rare_preflight = results
        self._set_busy(False)
        self._progress.setValue(100)
        rows = rd.preflight_rows(results)
        tbl = self._rare_preflight_table
        tbl.setRowCount(len(rows))
        colors = {rd.PREFLIGHT_OK: "#a6e3a1", rd.PREFLIGHT_WARN: "#f9e2af",
                  rd.PREFLIGHT_BLOCKED: "#f38ba8"}
        verdicts = {rd.PREFLIGHT_OK: "OK", rd.PREFLIGHT_WARN: "Thin",
                    rd.PREFLIGHT_BLOCKED: "Too few examples"}
        for i, r in enumerate(rows):
            prev = r.get("time_fraction", r.get("bout_rate", float("nan")))
            hunted = " ← will be hunted" if r["would_be_hunted"] else ""
            cells = [r["project"], r["behavior"] + hunted, str(r["rarity_rank"]),
                     "—" if prev != prev else f"{prev:.4g}",
                     str(r["confirmed_examples"]), str(r["left_to_discover"]),
                     verdicts.get(r["status"], r["status"])]
            for j, text in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if r["note"]:
                    it.setToolTip(r["note"])
                if j == 6:
                    it.setForeground(QColor(colors.get(r["status"], "#cdd6f4")))
                tbl.setItem(i, j, it)
        tbl.resizeColumnsToContents()

        # The banner is the actual point of phase 1: say plainly whether to go
        # label more examples, and in which project.
        blocked = [p for p in results if p.rarest and not p.rarest.runnable()]
        thin = [p for p in results
                if p.target and p.target.status == rd.PREFLIGHT_WARN]
        dead = [p for p in results if p.target is None]
        bits = []
        for p in results:
            if p.target is not None:
                bits.append(f"{p.project_name}: {p.target.behavior_name} "
                            f"({p.target.n_labeled} examples)")
        msg = "Will hunt — " + "; ".join(bits) if bits else "Nothing is runnable."
        if blocked:
            msg += ("\n⚠ Rarest behavior has too few confirmed examples in: "
                    + "; ".join(f"{p.project_name} ({p.rarest.behavior_name}, "
                                f"{p.rarest.n_labeled})" for p in blocked)
                    + ". Label more of it in the Review tab, or accept the "
                      "next-rarest behavior shown above.")
        if thin:
            msg += ("\n⚠ Thin evidence (curves will be noisy) in: "
                    + "; ".join(p.project_name for p in thin))
        if dead:
            msg += ("\n⚠ No huntable behavior at all in: "
                    + "; ".join(f"{p.project_name}"
                                + (f" — {p.error}" if p.error else "")
                                for p in dead))
        self._rare_preflight_msg.setText(msg)
        self._rare_preflight_msg.setStyleSheet(
            "color:#f38ba8;" if (blocked or dead) else
            "color:#f9e2af;" if thin else "color:#a6e3a1;")
        self._log_msg(msg.replace("\n", "  "))

    # ── phase 2: the heavy hunt, on the behaviors phase 1 picked ──
    def _run_rare_discovery(self) -> None:
        """Run the hunt — on the preflight's targets when a check has been run."""
        from abel.validation.analyses import rare_discovery as rd  # noqa: PLC0415
        override = None
        if self._rare_preflight:
            override = {p.project_id: [p.target.behavior_id]
                        for p in self._rare_preflight if p.target is not None}
            skipped = [p.project_name for p in self._rare_preflight
                       if p.target is None]
            if not override:
                QMessageBox.warning(
                    self, "Nothing to hunt",
                    "No project has a behavior with enough confirmed examples.\n\n"
                    "Label more examples of the rare behavior and re-run the check.")
                return
            thin = [p for p in self._rare_preflight
                    if p.target is not None and p.target.status != rd.PREFLIGHT_OK]
            warn = ""
            if skipped:
                warn += f"Skipping (no huntable behavior): {', '.join(skipped)}\n\n"
            if thin:
                warn += ("Thin evidence — the curves will be noisy:\n"
                         + "\n".join(f"  • {p.project_name}: "
                                     f"{p.target.behavior_name} "
                                     f"({p.target.n_labeled} examples)"
                                     for p in thin) + "\n\n")
            if warn and QMessageBox.question(
                    self, "Run anyway?",
                    warn + "This run takes hours. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
            ) != QMessageBox.StandardButton.Yes:
                return
        elif QMessageBox.question(
                self, "No rarity check yet",
                "You have not run the fast rarity check.\n\n"
                "It takes seconds and tells you whether any project needs more "
                "labeled examples before committing to a run that takes hours.\n\n"
                "Start the full run anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        self._run([ANALYSIS_RARE_DISCOVERY], behaviors_override=override)

    def _build_cross_tab(self) -> QWidget:
        intro = _explain(
            "Cross-project dashboard, assembled from the most recent run. Nothing to "
            "configure here — it pools whatever the last analysis produced across every "
            "project you had loaded.")
        openb = QPushButton("Open full HTML report…")
        openb.clicked.connect(self._open_report)
        self._cross_panel = _ResultPanel(title="Cross-project dashboard")
        return _split_tab([intro, openb], self._cross_panel, left_chars=52)

    def _on_tab_changed(self, index: int) -> None:
        # Auto-populate the behaviorscape alias table from the current selection,
        # but never clobber edits once the table is already populated.
        if (self._tabs.widget(index) is getattr(self, "_bscape_tab", None)
                and self._bscape_alias_table.rowCount() == 0):
            self._refresh_bscape_aliases()

    def _build_behaviorscape_tab(self) -> QWidget:
        intro = _explain(
            "The behaviorscape: which feature TYPES drive which behaviors, pooled across all\n"
            "selected projects. Trains one model per checked (project, behavior) on the shared\n"
            "held-out split, captures per-feature importance, classifies each feature into a data\n"
            "modality (pose geometry · kinematics · video flow/appearance · context ROI/target),\n"
            "and renders five publication figures: a clustered heatmap, per-behavior modality\n"
            "bars, a PERMANOVA distinctiveness test (do behaviors rely on different features?),\n"
            "a behavior-similarity matrix, and a feature↔behavior network. Behaviors are pooled\n"
            "by name — use the alias table to merge across slight naming differences.")

        box = QGroupBox("Settings")
        form = QFormLayout(box)
        self._bscape_threshold = QDoubleSpinBox()
        self._bscape_threshold.setRange(0.0, 1.0)
        self._bscape_threshold.setSingleStep(0.005)
        self._bscape_threshold.setDecimals(3)
        self._bscape_threshold.setValue(0.010)
        self._bscape_threshold.setToolTip(
            "Drop any feature whose pooled importance never reaches this value in ANY behavior\n"
            "(i.e. below threshold across all included projects/models). With 'fraction'\n"
            "normalization, importance is each model's share of total gain, so 0.010 ≈ 1%.")
        form.addRow("Importance threshold (drop below in all behaviors):", self._bscape_threshold)
        self._bscape_norm = QComboBox()
        self._bscape_norm.addItems(["fraction", "max"])
        self._bscape_norm.setToolTip(
            "fraction: each model's importances divided by their sum (comparable across\n"
            "behaviors regardless of scale). max: divided by the model's largest importance.")
        form.addRow("Per-model normalization:", self._bscape_norm)

        alias_box = QGroupBox("Behavior pooling (edit 'Pooled name' to merge behaviors)")
        av = QVBoxLayout(alias_box)
        refresh = QPushButton("Refresh from checked behaviors")
        refresh.setToolTip("Populate the table from the behaviors currently checked on the "
                           "Projects tab.")
        refresh.clicked.connect(self._refresh_bscape_aliases)
        av.addWidget(refresh)
        self._bscape_alias_table = QTableWidget(0, 3)
        self._bscape_alias_table.setHorizontalHeaderLabels(
            ["Project", "Behavior", "Pooled name"])
        self._bscape_alias_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._bscape_alias_table.setMinimumHeight(140)
        # Click a column header to sort — handy for spotting replicates / variants.
        self._bscape_alias_table.setSortingEnabled(True)
        self._bscape_alias_table.horizontalHeader().setSortIndicatorShown(True)
        self._bscape_alias_table.horizontalHeader().setToolTip(
            "Click a column header to sort (e.g. by Behavior to group replicates "
            "and near-duplicate names).")
        av.addWidget(self._bscape_alias_table)

        run = QPushButton("Build Behaviorscape"); run.setObjectName("runBtn")
        run.clicked.connect(self._run_behaviorscape)
        self._bscape_run_btn = run

        # Status (incl. the long pooled-behaviors list) sits at the foot of the
        # settings column; it scrolls with the column rather than being capped.
        self._bscape_status = QLabel("")
        self._bscape_status.setWordWrap(True)
        self._bscape_status.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._bscape_status.setStyleSheet("color:#a6adc8; font-size:11px;")

        self._bscape_panel = _ResultPanel(title="Behaviorscape")
        return _split_tab([intro, box, (alias_box, 1), run, self._bscape_status],
                          self._bscape_panel, left_chars=70)

    def _refresh_bscape_aliases(self) -> None:
        behaviors = self._collect_behaviors()
        # Preserve any pooled-name edits the user already made, keyed by raw name.
        prior: dict[str, str] = {}
        for r in range(self._bscape_alias_table.rowCount()):
            raw_item = self._bscape_alias_table.item(r, 1)
            pooled_item = self._bscape_alias_table.item(r, 2)
            if raw_item and pooled_item:
                prior[raw_item.text()] = pooled_item.text()

        rows: list[tuple[str, str]] = []
        for pid, bids in behaviors.items():
            proj = self._projects.get(pid)
            if not proj:
                continue
            for bid in bids:
                rows.append((proj.name, proj.behavior_label(bid)))
        # Disable sorting while filling, or rows reshuffle mid-insert and setItem
        # writes land in the wrong logical row.
        self._bscape_alias_table.setSortingEnabled(False)
        self._bscape_alias_table.setRowCount(len(rows))
        for r, (pname, bname) in enumerate(rows):
            it_p = QTableWidgetItem(pname); it_p.setFlags(it_p.flags() & ~Qt.ItemFlag.ItemIsEditable)
            it_b = QTableWidgetItem(bname); it_b.setFlags(it_b.flags() & ~Qt.ItemFlag.ItemIsEditable)
            it_pool = QTableWidgetItem(prior.get(bname, bname))
            self._bscape_alias_table.setItem(r, 0, it_p)
            self._bscape_alias_table.setItem(r, 1, it_b)
            self._bscape_alias_table.setItem(r, 2, it_pool)
        self._bscape_alias_table.setSortingEnabled(True)
        if not rows:
            self._bscape_status.setText(
                "No behaviors checked. Check behaviors on the Projects tab, then refresh.")

    def _bscape_alias_map(self) -> dict[str, str]:
        """Raw behavior name → pooled name, from the alias table."""
        amap: dict[str, str] = {}
        for r in range(self._bscape_alias_table.rowCount()):
            raw_item = self._bscape_alias_table.item(r, 1)
            pooled_item = self._bscape_alias_table.item(r, 2)
            if raw_item and pooled_item:
                pooled = pooled_item.text().strip()
                if pooled:
                    amap[raw_item.text()] = pooled
        return amap

    def _run_behaviorscape(self) -> None:
        if self._busy:
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return
        behaviors = self._collect_behaviors()
        if not behaviors:
            QMessageBox.warning(self, "Nothing selected",
                                "Add a project and check at least one behavior on the Projects tab.")
            return
        projects = [self._projects[pid] for pid in behaviors]
        holdout_kwargs = dict(
            min_confidence=self._min_conf.value(),
            holdout_test_size=self._test_size.value(),
            holdout_seed=self._holdout_seed.value(),
        )
        build_kwargs = dict(
            threshold=self._bscape_threshold.value(),
            alias_map=self._bscape_alias_map(),
            normalize=self._bscape_norm.currentText(),
        )
        base = self._output_dir or Path(tempfile.gettempdir()) / "abel_behaviorscape"
        out_dir = Path(base) / f"behaviorscape_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._set_busy(True)
        self._bscape_status.setText("Training models and collecting feature importance…")
        self._log_msg(f"Behaviorscape: projects={[p.name for p in projects]}, "
                      f"threshold={build_kwargs['threshold']}, out={out_dir}")
        worker = _BehaviorscapeWorker(projects, behaviors, holdout_kwargs, build_kwargs, out_dir)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_behaviorscape_finished)
        worker.signals.error.connect(self._on_behaviorscape_error)
        self._bscape_worker = worker  # keep alive
        QThreadPool.globalInstance().start(worker)

    def _on_behaviorscape_finished(self, result: object) -> None:
        self._set_busy(False)
        self._progress.setValue(100)
        if not isinstance(result, dict):
            return
        images = result.get("images") or []
        self._bscape_panel.set_simple(images, result.get("tables"),
                                      folder=result.get("out_dir"))
        self._bscape_status.setText(str(result.get("summary", "")))
        self._log_msg(f"Behaviorscape complete: {len(images)} figure(s).")
        self.statusBar().showMessage("Behaviorscape figures ready.", 8000)

    def _on_behaviorscape_error(self, tb: str) -> None:
        self._set_busy(False)
        self._bscape_status.setText("Behaviorscape failed — see Log tab.")
        self._log_msg("BEHAVIORSCAPE ERROR:\n" + tb)
        QMessageBox.critical(self, "Behaviorscape failed",
                             tb.splitlines()[-1] if tb else "Unknown error")

    # ── Video-feature value tab ──────────────────────────────────────────
    def _build_video_value_tab(self) -> QWidget:
        intro = _explain(
            "Video-motion-feature value: how much do the video features (optical flow,\n"
            "surface motion, R3D appearance) improve detection? For each checked (project,\n"
            "behavior) this trains ABEL's real classifier twice on the SAME held-out split and\n"
            "the SAME training subsample — once WITHOUT the video features and once WITH them —\n"
            "so the F1 difference is a clean paired estimate of what the video motion features\n"
            "add. Great for the Groom-vs-Freeze case: both are low-locomotion and confusable by\n"
            "pose alone, so the rhythmic-motion signal is where video features pay off.\n"
            "Check the behaviors to compare (e.g. Groom and Freeze) on the Projects tab.")

        box = QGroupBox("Settings"); form = QFormLayout(box)
        self._vv_seeds = QSpinBox(); self._vv_seeds.setRange(1, 20); self._vv_seeds.setValue(5)
        self._vv_seeds.setToolTip("Paired train/eval repeats per behavior; more seeds → tighter "
                                  "confidence interval on the gain.")
        form.addRow("Seeds per behavior:", self._vv_seeds)

        run = QPushButton("Run Video-Feature Comparison"); run.setObjectName("runBtn")
        run.clicked.connect(self._run_video_value)
        self._vv_run_btn = run

        self._vv_status = QLabel(""); self._vv_status.setWordWrap(True)
        self._vv_status.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._vv_status.setStyleSheet("color:#a6adc8; font-size:11px;")

        self._vv_panel = _ResultPanel(title="Video-feature value")
        return _split_tab([intro, box, run, self._vv_status], self._vv_panel)

    def _run_video_value(self) -> None:
        if self._busy:
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return
        behaviors = self._collect_behaviors()
        if not behaviors:
            QMessageBox.warning(self, "Nothing selected",
                                "Check the behaviors to compare (e.g. Groom and Freeze) on the "
                                "Projects tab first.")
            return
        projects = [self._projects[pid] for pid in behaviors]
        holdout_kwargs = dict(
            min_confidence=self._min_conf.value(),
            holdout_test_size=self._test_size.value(),
            holdout_seed=self._holdout_seed.value())
        base = self._output_dir or Path(tempfile.gettempdir()) / "abel_video_value"
        out_dir = Path(base) / f"video_value_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._set_busy(True)
        self._vv_status.setText("Training with vs. without video features…")
        self._log_msg(f"Video-feature comparison: projects={[p.name for p in projects]}, "
                      f"seeds={self._vv_seeds.value()}, out={out_dir}")
        worker = _VideoValueWorker(projects, behaviors, holdout_kwargs,
                                   self._vv_seeds.value(), out_dir)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_video_value_finished)
        worker.signals.error.connect(self._on_job_error)
        self._vv_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_video_value_finished(self, result: object) -> None:
        self._set_busy(False); self._progress.setValue(100)
        if not isinstance(result, dict):
            return
        self._vv_panel.set_simple(result.get("images") or [], result.get("tables"),
                                  folder=result.get("out_dir"))
        self._vv_status.setText(str(result.get("summary", "")))
        self._log_msg("Video-feature comparison complete.")
        self.statusBar().showMessage("Video-feature comparison ready.", 8000)

    # ── Throughput benchmark tab ─────────────────────────────────────────
    def _build_throughput_tab(self) -> QWidget:
        intro = _explain(
            "Pipeline throughput: how fast is ABEL's data processing? Times the three stages\n"
            "on one representative session per added project, normalized by the video's real\n"
            "duration (× real-time; higher = faster). Runs on ALL projects added on the Projects\n"
            "tab; training is timed for the checked behaviors (or all, if none are checked).\n"
            "• Feature extraction — full pose+video+representation rebuild (SLOW; GPU-bound).\n"
            "• Training — time to fit one classifier once features exist.\n"
            "• Dense inference — running the models over every window of the session. This\n"
            "  recomputes that one session's temporal-refinement traces on the real project.")

        box = QGroupBox("Stages to benchmark"); form = QVBoxLayout(box)
        self._bench_extract = QCheckBox("Feature extraction / session")
        self._bench_extract.setToolTip("Slow — a full pose+video+representation rebuild.")
        self._bench_extract.setChecked(True)
        self._bench_train = QCheckBox("Model training (given features)")
        self._bench_train.setChecked(True)
        self._bench_infer = QCheckBox("Dense inference / video")
        self._bench_infer.setToolTip(
            "Recomputes that session's temporal-refinement traces on the real project.")
        self._bench_infer.setChecked(False)
        for cb in (self._bench_extract, self._bench_train, self._bench_infer):
            form.addWidget(cb)

        run = QPushButton("Run Throughput Benchmark"); run.setObjectName("runBtn")
        run.clicked.connect(self._run_benchmark)
        self._bench_run_btn = run

        self._bench_status = QLabel(""); self._bench_status.setWordWrap(True)
        self._bench_status.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._bench_status.setStyleSheet("color:#a6adc8; font-size:11px;")

        self._bench_panel = _ResultPanel(title="Pipeline throughput")
        return _split_tab([intro, box, run, self._bench_status], self._bench_panel)

    def _run_benchmark(self) -> None:
        if self._busy:
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return
        projects = list(self._projects.values())
        if not projects:
            QMessageBox.warning(self, "No projects",
                                "Add at least one project on the Projects tab first.")
            return
        stages = []
        if self._bench_extract.isChecked():
            stages.append("extract")
        if self._bench_train.isChecked():
            stages.append("train")
        if self._bench_infer.isChecked():
            stages.append("infer")
        if not stages:
            QMessageBox.warning(self, "No stages", "Check at least one stage to benchmark.")
            return
        behaviors = self._collect_behaviors()
        base = self._output_dir or Path(tempfile.gettempdir()) / "abel_benchmark"
        out_dir = Path(base) / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._set_busy(True)
        self._bench_status.setText("Benchmarking pipeline stages…")
        self._log_msg(f"Throughput benchmark: projects={[p.name for p in projects]}, "
                      f"stages={stages}, out={out_dir}")
        worker = _BenchmarkWorker(projects, behaviors, stages, out_dir)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_benchmark_finished)
        worker.signals.error.connect(self._on_job_error)
        self._bench_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_benchmark_finished(self, result: object) -> None:
        self._set_busy(False); self._progress.setValue(100)
        if not isinstance(result, dict):
            return
        self._bench_panel.set_simple(result.get("images") or [], result.get("tables"),
                                     folder=result.get("out_dir"))
        self._bench_status.setText(str(result.get("summary", "")))
        self._log_msg("Throughput benchmark complete.")
        self.statusBar().showMessage("Throughput benchmark ready.", 8000)

    def _on_job_error(self, tb: str) -> None:
        self._set_busy(False)
        self._log_msg("JOB ERROR:\n" + tb)
        for status in ("_vv_status", "_bench_status"):
            if hasattr(self, status):
                getattr(self, status).setText("Run failed — see Log tab.")
        QMessageBox.critical(self, "Run failed",
                             tb.splitlines()[-1] if tb else "Unknown error")

    def _build_feature_demo_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        intro = QLabel(
            "Export a feature-demonstration clip (raw vs. smoothed DLC tracking with a\n"
            "live trace strip) to an MP4 — the same visual as the main GUI's video preview,\n"
            "for showing what a feature responds to. Uses a random window of the chosen session.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#a6adc8;")
        lay.addWidget(intro)

        box = QGroupBox("Source")
        form = QFormLayout(box)
        self._demo_proj_combo = QComboBox()
        self._demo_proj_combo.currentIndexChanged.connect(self._on_demo_project_changed)
        form.addRow("Project:", self._demo_proj_combo)
        self._demo_sess_combo = QComboBox()
        form.addRow("Session:", self._demo_sess_combo)
        lay.addWidget(box)

        sett = QGroupBox("Settings")
        sform = QFormLayout(sett)
        self._demo_smooth = QSpinBox(); self._demo_smooth.setRange(1, 31); self._demo_smooth.setValue(5)
        self._demo_smooth.setSuffix(" frames")
        sform.addRow("Smoothing window:", self._demo_smooth)
        self._demo_like = QDoubleSpinBox(); self._demo_like.setRange(0.0, 1.0)
        self._demo_like.setSingleStep(0.05); self._demo_like.setValue(0.2)
        sform.addRow("Min likelihood:", self._demo_like)
        self._demo_radius = QSpinBox(); self._demo_radius.setRange(8, 2048)
        self._demo_radius.setSingleStep(4); self._demo_radius.setValue(36); self._demo_radius.setSuffix(" px")
        sform.addRow("Local motion radius:", self._demo_radius)
        self._demo_mog2 = QSpinBox(); self._demo_mog2.setRange(4, 100); self._demo_mog2.setValue(16)
        sform.addRow("BG-subtract sensitivity (var):", self._demo_mog2)
        self._demo_dur = QSpinBox(); self._demo_dur.setRange(2, 60); self._demo_dur.setValue(10); self._demo_dur.setSuffix(" s")
        sform.addRow("Clip duration:", self._demo_dur)
        lay.addWidget(sett)

        tbox = QGroupBox("Traces (checked = drawn in the graph strip)")
        tlay = QVBoxLayout(tbox)
        self._demo_trace_list = QListWidget()
        self._demo_trace_list.setMaximumHeight(150)
        self._populate_demo_traces()
        tlay.addWidget(self._demo_trace_list)
        lay.addWidget(tbox)

        demo_btn_row = QHBoxLayout()
        self._demo_folder_btn = QPushButton("Open Data Folder")
        self._demo_folder_btn.setToolTip("Open the folder the last demo clip was written to.")
        self._demo_folder_btn.setEnabled(False)
        self._demo_folder_btn.clicked.connect(
            lambda: self._open_folder_or_warn(self._demo_out_dir, "demo clip"))
        demo_btn_row.addWidget(self._demo_folder_btn)
        self._demo_preview_btn = QPushButton("Preview…")
        self._demo_preview_btn.setToolTip(
            "Play a live ~10 s sample with the current settings (raw vs. smoothed tracking +\n"
            "trace strip) so you can see what the export will look like before writing it.")
        self._demo_preview_btn.clicked.connect(self._preview_demo)
        self._demo_btn = QPushButton("Export Demo Video…"); self._demo_btn.setObjectName("runBtn")
        self._demo_btn.clicked.connect(self._export_demo)
        demo_btn_row.addWidget(self._demo_preview_btn)
        demo_btn_row.addWidget(self._demo_btn, 1)
        lay.addLayout(demo_btn_row)
        self._demo_status = QLabel("")
        self._demo_status.setWordWrap(True); self._demo_status.setStyleSheet("color:#a6adc8;")
        lay.addWidget(self._demo_status)
        lay.addStretch()
        return w

    def _populate_demo_traces(self) -> None:
        try:
            from abel.ui.smoothing_preview_dialog import TRACE_CATALOG  # noqa: PLC0415
        except Exception:
            return
        self._demo_trace_list.clear()
        for tdef in TRACE_CATALOG:
            it = QListWidgetItem(f"{tdef.label}  ({tdef.category})")
            it.setData(Qt.ItemDataRole.UserRole, tdef.key)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if tdef.default_on else Qt.CheckState.Unchecked)
            self._demo_trace_list.addItem(it)

    def _selected_demo_traces(self) -> set[str]:
        out: set[str] = set()
        for i in range(self._demo_trace_list.count()):
            it = self._demo_trace_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.add(it.data(Qt.ItemDataRole.UserRole))
        return out

    def _refresh_demo_projects(self) -> None:
        if not hasattr(self, "_demo_proj_combo"):
            return
        cur = self._demo_proj_combo.currentData()
        self._demo_proj_combo.blockSignals(True)
        self._demo_proj_combo.clear()
        for pid, proj in self._projects.items():
            self._demo_proj_combo.addItem(proj.name, userData=pid)
        # restore selection if still present
        idx = self._demo_proj_combo.findData(cur) if cur else -1
        if idx >= 0:
            self._demo_proj_combo.setCurrentIndex(idx)
        self._demo_proj_combo.blockSignals(False)
        self._on_demo_project_changed()

    def _on_demo_project_changed(self, _idx: int = 0) -> None:
        self._demo_sess_combo.clear()
        self._demo_manifest = None
        pid = self._demo_proj_combo.currentData()
        proj = self._projects.get(pid) if pid else None
        if not proj:
            return
        try:
            from abel.services.import_service import ImportService  # noqa: PLC0415
            if self._demo_imports is None:
                self._demo_imports = ImportService()
            self._demo_manifest = self._demo_imports.load_manifest(proj.root)
        except Exception as exc:  # noqa: BLE001
            self._demo_status.setText(f"Could not load sessions for {proj.name}: {exc}")
            return
        if not self._demo_manifest or not self._demo_manifest.linked_sessions:
            self._demo_status.setText("No imported sessions found in this project.")
            return
        for s in self._demo_manifest.linked_sessions:
            self._demo_sess_combo.addItem(s.session_id, userData=s.session_id)
        self._demo_status.setText("")

    def _demo_smoothing(self):
        from abel.models.schemas import PoseSmoothingSettings  # noqa: PLC0415
        return PoseSmoothingSettings(
            likelihood_threshold=self._demo_like.value(),
            interpolate_dropouts=True,
            smoothing_window=self._demo_smooth.value(),
        )

    def _preview_demo(self) -> None:
        proj = self._projects.get(self._demo_proj_combo.currentData())
        if not proj or not self._demo_manifest or not self._demo_manifest.linked_sessions:
            QMessageBox.warning(self, "No session",
                                "Pick a project with imported sessions first.")
            return
        # Reuse the main GUI's preview dialog so the demo matches the feature tab exactly.
        from abel.ui.smoothing_preview_dialog import SmoothingPreviewDialog  # noqa: PLC0415
        if self._demo_imports is None:
            from abel.services.import_service import ImportService  # noqa: PLC0415
            self._demo_imports = ImportService()
        if self._demo_preview_dialog is not None and self._demo_preview_dialog.isVisible():
            self._demo_preview_dialog.raise_(); self._demo_preview_dialog.activateWindow()
            return
        dlg = SmoothingPreviewDialog(
            import_service=self._demo_imports,
            manifest=self._demo_manifest,
            get_smoothing_fn=self._demo_smoothing,
            get_local_radius_fn=lambda: self._demo_radius.value(),
            project_root=proj.root,
            parent=self,
        )
        # Reflect the demo tab's export choices in the preview (without writing to project.yaml).
        dlg._visible_traces = set(self._selected_demo_traces())
        dlg._mog2_thresh.blockSignals(True)
        dlg._mog2_thresh.setValue(self._demo_mog2.value())
        dlg._mog2_thresh.blockSignals(False)
        sid = self._demo_sess_combo.currentData()
        if sid is not None:
            i = dlg._session_combo.findData(sid)
            if i >= 0:
                dlg._session_combo.setCurrentIndex(i)
        self._demo_preview_dialog = dlg  # keep a reference alive
        dlg.show()

    def _export_demo(self) -> None:
        proj = self._projects.get(self._demo_proj_combo.currentData())
        sid = self._demo_sess_combo.currentData()
        if not proj or not sid or not self._demo_manifest:
            QMessageBox.warning(self, "Nothing selected", "Pick a project and session first.")
            return
        if self._demo_imports is None:
            from abel.services.import_service import ImportService  # noqa: PLC0415
            self._demo_imports = ImportService()
        video_path = self._demo_imports.video_path_for_session(self._demo_manifest, sid)
        pose_path = self._demo_imports.pose_path_for_session(self._demo_manifest, sid)
        if not video_path or not Path(video_path).exists():
            QMessageBox.warning(self, "Missing video", "Video file not found for this session.")
            return
        if not pose_path or not Path(pose_path).exists():
            QMessageBox.warning(self, "Missing pose", "Pose file not found for this session.")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export demo video", f"feature_demo_{sid}.mp4", "MP4 video (*.mp4)")
        if not dest:
            return

        kwargs = dict(
            video_path=Path(video_path),
            pose_path=Path(pose_path),
            smoothing=self._demo_smoothing(),
            out_path=Path(dest),
            local_radius_px=self._demo_radius.value(),
            mog2_var_threshold=self._demo_mog2.value(),
            duration_sec=float(self._demo_dur.value()),
            visible_traces=self._selected_demo_traces(),
            cancel_flag=self._demo_cancel,
        )
        self._demo_cancel[0] = False
        self._demo_btn.setEnabled(False)
        self._demo_status.setText("Rendering demo video…")
        worker = _DemoWorker(kwargs)
        worker.signals.progress.connect(self._on_demo_progress)
        worker.signals.finished.connect(self._on_demo_finished)
        worker.signals.error.connect(self._on_demo_error)
        self._demo_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_demo_progress(self, msg: str, frac: float) -> None:
        self._progress.setValue(int(frac * 100))
        self._progress_lbl.setText(msg)
        self._demo_status.setText(msg)

    def _on_demo_finished(self, out_path: str) -> None:
        self._demo_btn.setEnabled(True)
        self._demo_out_dir = Path(out_path).parent
        self._demo_folder_btn.setEnabled(True)
        self._progress.setValue(100)
        self._demo_status.setText(f"Saved → {out_path}")
        self._log_msg(f"Feature demo exported → {out_path}")
        self.statusBar().showMessage(f"Demo video: {out_path}", 10000)

    def _on_demo_error(self, tb: str) -> None:
        self._demo_btn.setEnabled(True)
        self._demo_status.setText("Export failed — see Log tab.")
        self._log_msg("DEMO ERROR:\n" + tb)
        QMessageBox.critical(self, "Demo export failed", tb.splitlines()[-1] if tb else "Unknown error")

    def _build_log_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        self._log = QTextEdit(); self._log.setReadOnly(True)
        lay.addWidget(self._log)
        row = QHBoxLayout()
        self._log_folder_btn = QPushButton("Open Data Folder")
        self._log_folder_btn.setToolTip(
            "Open the run folder this log describes (the output folder until a run finishes).")
        self._log_folder_btn.clicked.connect(
            lambda: self._open_folder_or_warn(
                self._last_run.run_dir if self._last_run else self._current_output_dir(),
                "run"))
        row.addWidget(self._log_folder_btn)
        row.addStretch(1)
        lay.addLayout(row)
        return w

    def _current_output_dir(self) -> "Path | None":
        """Where a run launched right now would write (label text is the source of
        truth: it tracks both the session's runs/ folder and a hand-picked one)."""
        if self._output_dir:
            return Path(self._output_dir)
        text = self._out_lbl.text().strip()
        return Path(text) if text else None

    def _open_folder_or_warn(self, path: "Path | None", what: str) -> None:
        if path is None or not Path(path).is_dir():
            QMessageBox.information(
                self, "No folder yet",
                f"There is no {what} folder on disk yet"
                + (f":\n\n{path}" if path else "."))
            return
        _open_in_file_manager(Path(path))

    @staticmethod
    def _wrap(layout) -> QWidget:
        c = QWidget(); c.setLayout(layout); return c

    # ── project/behavior management ──
    @staticmethod
    def _pick_project_dirs(parent) -> list[str]:
        """Folder picker that allows selecting MULTIPLE directories at once.

        The native OS dialog only permits a single folder, so we use Qt's own
        non-native dialog and switch its internal list/tree views to extended
        (multi) selection.
        """
        dlg = QFileDialog(parent, "Select one or more ABEL project folders")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        for view_type in (QListView, QTreeView):
            for view in dlg.findChildren(view_type):
                view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        if dlg.exec():
            return dlg.selectedFiles()
        return []

    def _add_project(self) -> None:
        paths = self._pick_project_dirs(self)
        if not paths:
            return
        added: list[tuple[ProjectRef, QListWidgetItem]] = []
        invalid: list[str] = []
        duplicate: list[str] = []
        for d in paths:
            proj = ProjectRef.load(d)
            if not proj.is_valid():
                invalid.append(Path(d).name)
                continue
            if proj.project_id in self._projects:
                duplicate.append(proj.name)
                continue
            self._projects[proj.project_id] = proj
            self._selected[proj.project_id] = {
                bid for bid in proj.behavior_names if bid != "no_behavior"
            }
            item = QListWidgetItem(self._project_item_text(proj))
            item.setData(Qt.ItemDataRole.UserRole, proj.project_id)
            self._proj_list.addItem(item)
            added.append((proj, item))

        if added:
            self._proj_list.setCurrentItem(added[-1][1])
            self._refresh_demo_projects()
            self._refresh_session_label()
            for proj, _ in added:
                self._log_msg(f"Added project: {proj.name} ({proj.root})")

        # Summarize skips (only worth a popup when something didn't go in).
        notes: list[str] = []
        if added:
            notes.append(f"Added {len(added)} project(s).")
        if duplicate:
            notes.append(f"Already in list: {', '.join(duplicate)}.")
        if invalid:
            notes.append("Skipped — no derived/training_sets/training_set.parquet: "
                         + ", ".join(invalid) + ".")
        if duplicate or invalid:
            QMessageBox.information(self, "Add Projects", "\n".join(notes))

    @staticmethod
    def _find_project_roots(base: Path) -> list[Path]:
        """Every ABEL project root under ``base`` (identified by its training set)."""
        roots: set[Path] = set()
        for ts in Path(base).rglob("training_set.parquet"):
            if ts.parent.name == "training_sets" and ts.parent.parent.name == "derived":
                roots.add(ts.parent.parent.parent)
        return sorted(roots)

    @staticmethod
    def _strength_filtered_selection(proj: ProjectRef, metric: str, thr: float):
        """Behaviors to check for ``proj`` given a strength metric + threshold.

        Returns ``(selected_ids, n_behaviors, n_included, n_weak, n_no_model)``.
        Reads existing on-disk model metrics only — no training.
        """
        from abel.validation.datamodel import read_behavior_model_metrics  # noqa: PLC0415

        metrics = read_behavior_model_metrics(proj)
        sel: set[str] = set()
        n_behaviors = n_weak = n_nomodel = 0
        for bid in proj.behavior_names:
            if bid == "no_behavior":
                continue
            n_behaviors += 1
            score = metrics.get(bid, {}).get(metric)
            if score is None:
                n_nomodel += 1
            elif score >= thr:
                sel.add(bid)
            else:
                n_weak += 1
        return sel, n_behaviors, len(sel), n_weak, n_nomodel

    def _project_item(self, pid: str) -> "QListWidgetItem | None":
        for i in range(self._proj_list.count()):
            it = self._proj_list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == pid:
                return it
        return None

    def _auto_add_directory(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select a directory to scan for ABEL projects")
        if not d:
            return
        metric = self._autoadd_metric.currentData()
        metric_label = self._autoadd_metric.currentText()
        thr = self._autoadd_thresh.value()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            roots = self._find_project_roots(Path(d))
        finally:
            QApplication.restoreOverrideCursor()
        if not roots:
            QMessageBox.information(
                self, "Auto-add",
                "No ABEL projects (with derived/training_sets/training_set.parquet) "
                "were found under that directory.")
            return

        added = dup = n_incl = n_weak = n_nomodel = 0
        last_item: QListWidgetItem | None = None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for root in roots:
                proj = ProjectRef.load(root)
                if not proj.is_valid():
                    continue
                if proj.project_id in self._projects:
                    dup += 1
                    continue
                self._projects[proj.project_id] = proj
                sel, n_behaviors, inc, weak, nomodel = \
                    self._strength_filtered_selection(proj, metric, thr)
                n_incl += inc; n_weak += weak; n_nomodel += nomodel
                self._selected[proj.project_id] = sel
                item = QListWidgetItem(f"{proj.name}  ({len(sel)}/{n_behaviors} behaviors ≥ thr)")
                item.setData(Qt.ItemDataRole.UserRole, proj.project_id)
                self._proj_list.addItem(item)
                last_item = item
                added += 1
                self._log_msg(
                    f"Auto-added {proj.name}: {len(sel)}/{n_behaviors} behavior(s) "
                    f"with {metric_label} ≥ {thr:g} ({proj.root})")
        finally:
            QApplication.restoreOverrideCursor()

        if last_item is not None:
            self._proj_list.setCurrentItem(last_item)
        self._refresh_demo_projects()
        self._refresh_session_label()
        QMessageBox.information(
            self, "Auto-add complete",
            f"Added {added} project(s)"
            + (f"  ({dup} already in the list, skipped)" if dup else "") + ".\n\n"
            f"Checked {n_incl} behavior(s) with {metric_label} ≥ {thr:g}.\n"
            f"Left unchecked: {n_weak} below threshold, {n_nomodel} with no trained model.\n\n"
            "Review the checked behaviors per project before running.")

    def _apply_strength_to_loaded(self) -> None:
        """Re-check behaviors across already-loaded projects by model strength."""
        if not self._projects:
            QMessageBox.information(self, "Apply strength filter", "No projects are loaded.")
            return
        metric = self._autoadd_metric.currentData()
        metric_label = self._autoadd_metric.currentText()
        thr = self._autoadd_thresh.value()
        if QMessageBox.question(
            self, "Apply strength filter",
            f"Re-check behaviors in all {len(self._projects)} loaded project(s) using "
            f"{metric_label} ≥ {thr:g}?\n\nThis overrides your current checkbox selection.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        ) != QMessageBox.StandardButton.Yes:
            return

        n_incl = n_weak = n_nomodel = 0
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for pid, proj in self._projects.items():
                sel, n_behaviors, inc, weak, nomodel = \
                    self._strength_filtered_selection(proj, metric, thr)
                n_incl += inc; n_weak += weak; n_nomodel += nomodel
                self._selected[pid] = sel
                item = self._project_item(pid)
                if item is not None:
                    item.setText(f"{proj.name}  ({len(sel)}/{n_behaviors} behaviors ≥ thr)")
                self._log_msg(
                    f"Strength filter on {proj.name}: {len(sel)}/{n_behaviors} behavior(s) "
                    f"with {metric_label} ≥ {thr:g}")
        finally:
            QApplication.restoreOverrideCursor()

        # Reflect the new selection in the behaviors list for the current project.
        self._on_project_selected(self._proj_list.currentItem())
        QMessageBox.information(
            self, "Strength filter applied",
            f"Re-checked {len(self._projects)} project(s) using {metric_label} ≥ {thr:g}.\n\n"
            f"Checked {n_incl} behavior(s); left unchecked {n_weak} below threshold "
            f"and {n_nomodel} with no trained model.")

    @staticmethod
    def _project_item_text(proj: ProjectRef) -> str:
        """List label — a renamed project shows what it was, so the mapping is auditable."""
        base = f"{proj.name}  ({len(proj.behavior_names)} behaviors)"
        return f"{base}   [was: {proj.original_name}]" if proj.is_renamed else base

    def _rename_project(self) -> None:
        """Rename a project for the whole suite (display-only; disk untouched)."""
        item = self._proj_list.currentItem()
        if not item:
            QMessageBox.information(self, "Rename project", "Select a project to rename.")
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        proj = self._projects.get(pid)
        if proj is None:
            return

        new_name, ok = QInputDialog.getText(
            self, "Rename project", f"Report '{proj.original_name}' as:", text=proj.name)
        if not ok:
            return
        new_name = new_name.strip() or proj.original_name
        if new_name == proj.project_id:
            return
        # project_id doubles as the key of _projects/_selected and as the label every
        # figure groups by, so a collision would silently merge two projects' results.
        if new_name in self._projects:
            QMessageBox.warning(self, "Rename project",
                                f"Another project is already named '{new_name}'. "
                                "Pick a different name.")
            return

        old_pid = proj.project_id
        proj.rename(new_name)
        # Re-key both maps: everything downstream looks the project up by project_id.
        self._projects = {
            (proj.project_id if k == old_pid else k): v for k, v in self._projects.items()
        }
        if old_pid in self._selected:
            self._selected[proj.project_id] = self._selected.pop(old_pid)
        item.setData(Qt.ItemDataRole.UserRole, proj.project_id)
        item.setText(self._project_item_text(proj))
        self._refresh_demo_projects()
        self._log_msg(f"Project '{proj.original_name}' now reported as '{proj.name}'")

    def _remove_project(self) -> None:
        item = self._proj_list.currentItem()
        if not item:
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        self._projects.pop(pid, None)
        self._selected.pop(pid, None)
        self._proj_list.takeItem(self._proj_list.row(item))
        self._beh_list.clear()
        self._refresh_demo_projects()
        self._refresh_session_label()

    def _on_project_selected(self, cur, _prev=None) -> None:
        self._beh_list.blockSignals(True)
        self._beh_list.clear()
        if cur:
            pid = cur.data(Qt.ItemDataRole.UserRole)
            proj = self._projects.get(pid)
            sel = self._selected.get(pid, set())
            if proj:
                for bid in proj.behavior_names:
                    if bid == "no_behavior":
                        continue
                    it = QListWidgetItem(self._behavior_item_text(proj, bid))
                    it.setData(Qt.ItemDataRole.UserRole, bid)
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(Qt.CheckState.Checked if bid in sel else Qt.CheckState.Unchecked)
                    self._beh_list.addItem(it)
        self._beh_list.blockSignals(False)

    @staticmethod
    def _behavior_item_text(proj: ProjectRef, bid: str) -> str:
        """List label — a renamed behavior shows what it was, so the mapping is auditable."""
        label = proj.behavior_label(bid)
        disk = proj.behavior_disk_name(bid)
        return label if label == disk else f"{label}   (was: {disk})"

    def _current_behavior(self) -> "tuple[ProjectRef, str, QListWidgetItem] | None":
        """The selected (project, behavior_id, list item), or None with a nudge."""
        pitem = self._proj_list.currentItem()
        bitem = self._beh_list.currentItem()
        if not pitem or not bitem:
            QMessageBox.information(self, "Rename behavior",
                                    "Select a project, then a behavior to rename.")
            return None
        proj = self._projects.get(pitem.data(Qt.ItemDataRole.UserRole))
        if proj is None:
            return None
        return proj, str(bitem.data(Qt.ItemDataRole.UserRole)), bitem

    def _set_behavior_name(self, proj: ProjectRef, bid: str, new_name: str,
                           item: QListWidgetItem) -> None:
        proj.set_behavior_alias(bid, new_name)
        # Rewriting the text re-emits itemChanged, which would re-read the check
        # state as a toggle — block it, the checkbox hasn't moved.
        self._beh_list.blockSignals(True)
        item.setText(self._behavior_item_text(proj, bid))
        self._beh_list.blockSignals(False)
        self._log_msg(f"{proj.name}: behavior '{proj.behavior_disk_name(bid)}' "
                      f"now reported as '{proj.behavior_label(bid)}'")

    def _rename_behavior(self) -> None:
        """Rename a behavior for the whole suite (display-only; disk untouched)."""
        picked = self._current_behavior()
        if picked is None:
            return
        proj, bid, item = picked
        new_name, ok = QInputDialog.getText(
            self, "Rename behavior",
            f"Report '{proj.behavior_disk_name(bid)}' as:",
            text=proj.behavior_label(bid))
        if not ok:
            return
        self._set_behavior_name(proj, bid, new_name, item)

    def _reset_behavior_name(self) -> None:
        picked = self._current_behavior()
        if picked is None:
            return
        proj, bid, item = picked
        self._set_behavior_name(proj, bid, "", item)

    def _on_behavior_toggled(self, item: QListWidgetItem) -> None:
        cur = self._proj_list.currentItem()
        if not cur:
            return
        pid = cur.data(Qt.ItemDataRole.UserRole)
        bid = item.data(Qt.ItemDataRole.UserRole)
        sel = self._selected.setdefault(pid, set())
        if item.checkState() == Qt.CheckState.Checked:
            sel.add(bid)
        else:
            sel.discard(bid)
        self._refresh_session_label()

    def _choose_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self._output_dir = Path(d)
            self._out_lbl.setText(d)

    # ── saved sessions (projects, checked behaviors, renames + their runs) ──
    def _holdout_settings(self) -> dict:
        return {
            "min_confidence": self._min_conf.value(),
            "holdout_test_size": self._test_size.value(),
            "holdout_seed": self._holdout_seed.value(),
        }

    def _capture_session(self, name: str, created_at: str = "") -> SessionRecord:
        return SessionRecord.capture(
            name, self._projects, self._selected,
            holdout=self._holdout_settings(), created_at=created_at,
            keep_entries=self._session_offline)

    def _refresh_session_label(self) -> None:
        if not self._session_name:
            self._session_lbl.setText("No session — runs go to the unfiled folder.")
            return
        runs = self._session_store.session_dir(self._session_name) / RUNS_DIRNAME
        n_beh = sum(len(b) for b in self._selected.values())
        self._session_lbl.setText(
            f"<b>{html.escape(self._session_name)}</b> — {len(self._projects)} project(s), "
            f"{n_beh} behavior(s) checked<br><span style='color:#78909C;'>"
            f"{html.escape(str(runs))}</span>")

    def _use_session_output(self) -> None:
        """Point run output at the active session's runs/ folder."""
        if not self._session_name:
            return
        self._output_dir = self._session_store.runs_dir(self._session_name)
        self._out_lbl.setText(str(self._output_dir))

    def _save_session(self, as_new: bool = False) -> None:
        name = self._session_name
        if as_new or not name:
            suggested = name or (next(iter(self._projects), "") or "validation") + " session"
            name, ok = QInputDialog.getText(self, "Save session", "Session name:", text=suggested)
            if not ok or not name.strip():
                return
            name = name.strip()
            if self._session_store.exists(name) and name != self._session_name:
                # Two names can share a folder slug ("EPM v1" / "EPM-v1"), so report
                # the session actually occupying it rather than the name just typed.
                try:
                    occupant = self._session_store.load(name).name
                except Exception:  # noqa: BLE001
                    occupant = name
                if QMessageBox.question(
                    self, "Save session",
                    f"The session '{occupant}' already uses this folder. Overwrite its setup?\n\n"
                    "(Its previous runs are kept.)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                ) != QMessageBox.StandardButton.Yes:
                    return
        created = ""
        if self._session_store.exists(name):
            try:
                created = self._session_store.load(name).created_at
            except Exception:  # noqa: BLE001 — a stale file must not block a save
                created = ""
        path = self._session_store.save(self._capture_session(name, created_at=created))
        self._session_name = name
        self._use_session_output()
        self._refresh_session_label()
        self._log_msg(f"Session '{name}' saved → {path}")
        self.statusBar().showMessage(f"Session saved: {path}", 6000)

    def _load_session(self) -> None:
        infos = self._session_store.list_sessions()
        if not infos:
            QMessageBox.information(
                self, "Load session",
                "No saved sessions yet.\n\nSet up projects and behaviors, then use "
                f"'Save As…'. Sessions live in:\n{self._session_store.root}")
            return
        labels = [
            f"{i.name}  —  {i.n_projects} project(s), {i.n_behaviors} behavior(s), "
            f"{i.n_runs} run(s)   [{i.updated_at[:16]}]"
            for i in infos
        ]
        choice, ok = QInputDialog.getItem(self, "Load session", "Session:", labels, 0, False)
        if not ok:
            return
        info = infos[labels.index(choice)]
        try:
            record = self._session_store.load(info.name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Load session", f"Could not read that session:\n{exc}")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            restored = record.restore()
        finally:
            QApplication.restoreOverrideCursor()

        self._projects = restored.projects
        self._selected = restored.selected
        self._session_offline = restored.unavailable
        self._proj_list.clear()
        self._beh_list.clear()
        for pid, proj in self._projects.items():
            item = QListWidgetItem(self._project_item_text(proj))
            item.setData(Qt.ItemDataRole.UserRole, pid)
            self._proj_list.addItem(item)
        if self._proj_list.count():
            self._proj_list.setCurrentRow(0)
        self._min_conf.setValue(float(record.holdout.get("min_confidence", self._min_conf.value())))
        self._test_size.setValue(float(record.holdout.get("holdout_test_size",
                                                          self._test_size.value())))
        self._holdout_seed.setValue(int(record.holdout.get("holdout_seed",
                                                           self._holdout_seed.value())))
        self._session_name = record.name
        self._use_session_output()
        self._refresh_session_label()
        self._refresh_demo_projects()
        self._log_msg(f"Loaded session '{record.name}': {len(self._projects)} project(s), "
                      f"{sum(len(v) for v in self._selected.values())} behavior(s) checked")

        # Never let a reloaded setup silently differ from the saved one.
        if restored.missing_projects or restored.missing_behaviors:
            msg = []
            if restored.missing_projects:
                msg.append("Projects that could not be loaded:\n  "
                           + "\n  ".join(restored.missing_projects)
                           + "\n\nThey are kept in the session, so saving now will not "
                             "erase them — reconnect the drive and load again to use them.")
            if restored.missing_behaviors:
                msg.append("Behaviors no longer in their project:\n  "
                           + "\n  ".join(restored.missing_behaviors))
            QMessageBox.warning(self, "Session loaded with gaps", "\n\n".join(msg))

    def _open_session_folder(self) -> None:
        target = (self._session_store.session_dir(self._session_name) if self._session_name
                  else workspace_root())
        target.mkdir(parents=True, exist_ok=True)
        self._open_path(target)

    # ── run orchestration ──
    @staticmethod
    def _parse_size_list(text: str, default: list[int]) -> list[int]:
        out: list[int] = []
        for tok in text.split(","):
            tok = tok.strip().lower()
            if not tok:
                continue
            if tok in ("all", "max"):
                out.append(subsample.ALL_CLIPS)
            else:
                try:
                    out.append(int(tok))
                except ValueError:
                    pass
        # De-dup while preserving order.
        seen: set[int] = set()
        out = [x for x in out if not (x in seen or seen.add(x))]
        return out or list(default)

    def _parse_sizes(self) -> list[int]:
        return self._parse_size_list(self._lc_sizes.text(), list(DEFAULT_SIZES))

    def _parse_ablation_budgets(self) -> list[int]:
        return self._parse_size_list(self._abl_budgets.text(), [subsample.ALL_CLIPS])

    def _collect_behaviors(self) -> dict[str, list[str]]:
        return {pid: sorted(bids) for pid, bids in self._selected.items() if bids}

    def _run(self, analyses: list[str], cfg: ValidationRunConfig | None = None,
             behaviors_override: dict[str, list[str]] | None = None) -> None:
        """Start a run. ``cfg`` overrides the per-tab settings (the full-suite
        button passes the publication preset); ``behaviors_override`` narrows the
        behaviors to run on (the rare tab's phase-2 button passes the targets the
        rarity check picked)."""
        if self._busy:
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return
        behaviors = behaviors_override or self._collect_behaviors()
        if not behaviors:
            QMessageBox.warning(self, "Nothing selected",
                                "Add a project and check at least one behavior.")
            return
        if cfg is None:
            cfg = self._build_config(analyses)
        if behaviors_override:
            # The targets are already chosen — re-ranking would only risk
            # disagreeing with the table the user just approved.
            cfg.rare_auto_target = False
        projects = [self._projects[pid] for pid in behaviors]
        # Gate before the run, not after: a suite run takes hours and an unreachable
        # pose drive silently disables whole arms (the Essence Miner in particular),
        # which is indistinguishable from a real negative result on the figure.
        if not confirm_run_with_missing_raw_data(
                self, [p.root for p in projects], what="this validation run"):
            self._log_msg("Run cancelled: raw data unavailable.")
            return
        # The setup as of this run: saved to the session now (so it can never drift
        # away from what launched the run) and frozen into the run dir when it lands.
        self._run_session = self._capture_session(self._session_name or "unfiled")
        if self._session_name:
            self._session_store.save(self._run_session)
        self._set_busy(True)
        self._log_msg(f"Starting run: analyses={analyses}, projects={[p.name for p in projects]}")
        worker = _RunWorker(projects, behaviors, cfg)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(lambda out: self._on_finished(out, analyses))
        worker.signals.error.connect(self._on_error)
        # Retain a reference: QThreadPool does not keep the Python wrapper alive,
        # and losing it would garbage-collect the signals before delivery.
        self._worker = worker
        QThreadPool.globalInstance().start(worker)

    def _build_config(self, analyses: list[str]) -> ValidationRunConfig:
        """The per-tab settings as a run config (shared by every run button)."""
        return ValidationRunConfig(
                analyses=analyses,
                sizes=self._parse_sizes(),
                n_seeds_lc=self._lc_seeds.value(),
                neg_policy=self._lc_negpolicy.currentText(),
                neg_per_pos=self._lc_negratio.value(),
                n_seeds_ablation=self._abl_seeds.value(),
                ablation_budgets=self._parse_ablation_budgets(),
                n_seeds_generalization=self._gen_seeds.value(),
                n_seeds_discrimination=self._disc_seeds.value(),
                discrimination_max_pairs=self._disc_max_pairs.value(),
                n_seeds_al=self._al_seeds.value(),
                al_k0=self._al_k0.value(),
                al_batch=self._al_batch.value(),
                al_max_budget=self._al_max.value(),
                al_seed_pos=self._al_seed_pos.value(),
                al_acquisition=self._al_acq.currentText(),
                rare_auto_target=self._rare_auto.isChecked(),
                n_seeds_rare=self._rare_seeds.value(),
                rare_n_seed_pos=self._rare_seed_pos.value(),
                rare_al_budget=self._rare_budget.value(),
                rare_include_rarity_scaling=self._rare_rarity.isChecked(),
                rare_include_fullpool=self._rare_fullpool.isChecked(),
                rare_include_quality=self._rare_quality.isChecked(),
                rare_exclude_behaviors=[s.strip() for s in
                                        self._rare_exclude.text().split(",") if s.strip()],
                min_confidence=self._min_conf.value(),
                holdout_test_size=self._test_size.value(),
                holdout_seed=self._holdout_seed.value(),
                output_root=str(self._output_dir) if self._output_dir else "",
            )

    def _on_progress(self, msg: str, frac: float) -> None:
        self._progress.setValue(int(frac * 100))
        self._progress_lbl.setText(msg)

    def _on_finished(self, out: RunOutputs, analyses: list[str]) -> None:
        self._last_run = out
        self._set_busy(False)
        self._progress.setValue(100)
        self._log_msg(f"Run complete → {out.run_dir}")
        if self._run_session is not None:
            try:
                self._session_store.attach_to_run(self._run_session, out.run_dir)
            except Exception as exc:  # noqa: BLE001 — bookkeeping must not sink a finished run
                self._log_msg(f"Could not write the run's setup snapshot: {exc}")
            self._refresh_session_label()
        # populate the relevant tab's panels (figures + export targets)
        if ANALYSIS_LEARNING_CURVE in analyses:
            lc_dir = out.run_dir / "learning_curves"
            images_by_view = {
                view: sorted(lc_dir.glob(f"*__{view}.png")) for view in LEARNING_CURVE_VIEWS
            }
            self._lc_panel.set_views(images_by_view, lc_dir / "learning_curve_points.csv",
                                     folder=lc_dir)
            self._report_figures("Learning curves", images_by_view)
        if ANALYSIS_ABLATION in analyses:
            abl_dir = out.run_dir / "ablation"
            abl_imgs = sorted(abl_dir.glob("*.png"))
            self._abl_panel.set_simple(abl_imgs, abl_dir / "ablation_results.csv",
                                       folder=abl_dir)
            self._report_figures("Ablation", abl_imgs)
        if ANALYSIS_DISCRIMINATION in analyses:
            disc_dir = out.run_dir / "discrimination"
            # A separability matrix is written per ADD-ON feature family
            # (…__separability_matrix__pose_video.png), so which views exist is only
            # known from what the run produced — a fixed view list silently showed
            # an empty panel.
            views: dict[str, list[Path]] = {}
            # First = the tab's default view. The pooled landscape answers the whole
            # analysis in one figure; the per-project matrices behind it are the
            # per-assay detail, and opening on one of those buried the result.
            landscape = disc_dir / "discrimination_landscape.png"
            if landscape.exists():
                views["Discrimination landscape (all projects)"] = [landscape]
            for png in sorted(disc_dir.glob("*__separability_matrix__*.png")):
                fs = png.stem.split("__separability_matrix__", 1)[-1]
                label = FEATURE_SET_LABELS.get(fs, fs).lstrip("+ ").strip()
                views.setdefault(f"Separability matrix — Δ from {label}", []).append(png)
            gain_imgs = sorted(disc_dir.glob("*__feature_gain_by_pair.png"))
            if gain_imgs:
                views["Feature gain per behavior pair"] = gain_imgs
            self._disc_panel.set_views(
                views,
                {
                    "Hardest pairs (ranked)": disc_dir / "confusable_pairs.csv",
                    "Per-pair × feature set": disc_dir / "discrimination_results.csv",
                    "Per-seed ROC-AUC (replicates)":
                        disc_dir / "discrimination_seed_scores.csv",
                },
                folder=disc_dir,
            )
            self._report_figures("Discrimination", views, hint=(
                "Discrimination needs at least 2 behaviors checked in the SAME project, "
                "each with enough held-out clips."))
        if ANALYSIS_GENERALIZATION in analyses:
            # The generalization run also emits the biological-readout and
            # calibration analyses (they reuse its held-out predictions, so they
            # cost no extra training) — surface them as views on the same panel.
            gen_dir = out.run_dir / "generalization"
            tb_dir = out.run_dir / "time_budget"
            cal_dir = out.run_dir / "calibration"
            gen_views = {
                "Model vs. human agreement (κ)": sorted(gen_dir.glob("*.png")),
                "Biological readout (time budget & bouts)": sorted(tb_dir.glob("*.png")),
                "Probability calibration (reliability)": sorted(cal_dir.glob("*.png")),
            }
            gen_views = {k: v for k, v in gen_views.items() if v}
            self._gen_panel.set_views(
                gen_views,
                {
                    "Agreement (F1, κ)": gen_dir / "agreement.csv",
                    "Time-budget agreement": tb_dir / "time_budget_agreement.csv",
                    "Time budget per session": tb_dir / "time_budget_points.csv",
                    "Calibration (ECE / Brier)": cal_dir / "calibration.csv",
                },
                folder={"Generalization": gen_dir, "Time budget": tb_dir,
                        "Calibration": cal_dir},
            )
            self._report_figures("Generalization", gen_views)
        if ANALYSIS_AL_CURVE in analyses:
            al_dir = out.run_dir / "active_learning"
            al_imgs = sorted(al_dir.glob("*.png"))
            self._al_panel.set_simple(al_imgs, {
                "Curves (F1 & positives vs clips)": al_dir / "al_vs_random_points.csv",
                "Summary (clips-to-target)": al_dir / "al_vs_random_summary.csv",
            }, folder=al_dir)
            self._report_figures("Active learning", al_imgs)
        if ANALYSIS_RARE_DISCOVERY in analyses:
            rd_dir = out.run_dir / "rare_discovery"
            prism_dir = out.run_dir / "prism"
            # Behaviour rarity first (0_ prefix), then discovery/effort/rarity figures.
            rd_imgs = sorted(rd_dir.glob("*.png"))
            # A multi-project run files each project's Prism tables under its own
            # stem (…__<project>.csv), so match by prefix rather than exact name
            # and label the per-project copies with the stem.
            rare_tables: dict[str, Path] = {}
            for label, base in (
                ("Combined enrichment", "prism_combined_enrichment"),
                ("Behavior rarity", "prism_behavior_rarity"),
                ("Discovery curve (reviewed)", "prism_discovery_reviewed"),
                ("Effort-to-N (reviewed)", "prism_effort_reviewed"),
                ("Discovery curve (full pool)", "prism_discovery_fullpool"),
                ("Rarity scaling", "prism_rarity_scaling"),
            ):
                for p in sorted(prism_dir.glob(f"{base}*.csv")):
                    stem = p.stem[len(base):].lstrip("_").replace("_", " ").strip()
                    rare_tables[f"{label} — {stem}" if stem else label] = p
            for label, p in (("Which behavior was hunted", rd_dir / "hunted_targets.csv"),
                             ("Combined across projects",
                              rd_dir / "combined_across_projects.csv")):
                if p.exists():
                    rare_tables[label] = p
            if not rare_tables:
                rare_tables = {
                    "Discovery & effort": rd_dir / "discovery.csv",
                    "Rarity scaling": rd_dir / "rarity_scaling.csv",
                }
            self._rare_panel.set_simple(
                rd_imgs, rare_tables,
                folder={"Rare discovery": rd_dir, "Prism tables": prism_dir})
            self._report_figures("Rare-behavior discovery", rd_imgs, hint=(
                "Needs a project with dense bout detections and enough confirmed "
                "positives of the target behavior to cross-validate."))
        cross_dir = out.run_dir / "cross_project"
        cross_imgs = sorted(cross_dir.glob("*.png"))
        self._cross_panel.set_simple(cross_imgs, {
            "Accuracy by project": cross_dir / "dashboard.csv",
            "Training speed by project": cross_dir / "training_speed.csv",
        }, folder=cross_dir)
        self._report_figures("Cross-project", cross_imgs)
        if ANALYSIS_BEHAVIORSCAPE in analyses:
            bs_dir = out.run_dir / "behaviorscape"
            bs_imgs = sorted(bs_dir.glob("*.png"))
            self._bscape_panel.set_simple(bs_imgs, {
                "Feature importance (long)": bs_dir / "behaviorscape_importance.csv",
                "Modality shares per behavior": bs_dir / "behaviorscape_modality_shares.csv",
                "Behavior distinctiveness": bs_dir / "behaviorscape_distinctiveness.csv",
                "Behavior similarity matrix": bs_dir / "behaviorscape_similarity_matrix.csv",
            }, folder=bs_dir)
            self._report_figures("Behaviorscape", bs_imgs)
        if ANALYSIS_VIDEO_VALUE in analyses:
            vv_dir = out.run_dir / "video_value"
            vv_imgs = sorted(vv_dir.glob("*.png"))
            self._vv_panel.set_simple(vv_imgs, {"Video-feature value": vv_dir / "video_value.csv"},
                                      folder=vv_dir)
            self._report_figures("Video features", vv_imgs)
        if ANALYSIS_THROUGHPUT in analyses:
            th_dir = out.run_dir / "throughput"
            th_imgs = sorted(th_dir.glob("*.png"))
            self._bench_panel.set_simple(th_imgs, {"Throughput": th_dir / "benchmark.csv"},
                                         folder=th_dir)
            self._report_figures("Throughput", th_imgs)

        # The consolidated summary. Findings are cheap and always shown; the PDF
        # render needs the GUI thread (QtWebEngine wants an event loop), which is
        # exactly where we are now — it cannot be done inside the worker.
        if out.summary_html is not None:
            self._populate_suite_tab(out)
            if self._pending_pdf:
                self._pending_pdf = False
                self._progress_lbl.setText("Rendering the summary PDF…")
                pdf = self._ensure_pdf()
                msg = (f"Done. Report: {pdf}" if pdf
                       else f"Done. Summary (HTML): {out.summary_html}")
                self._suite_status.setText(
                    msg + "\nUse Export Everything to write the figures and data CSVs "
                          "to a folder.")
        self.statusBar().showMessage(f"Results: {out.run_dir}", 10000)

    def _report_figures(
        self,
        tab: str,
        figures: "list[Path] | dict[str, list[Path]]",
        hint: str = "",
    ) -> None:
        """Log how many figures a tab actually received, and warn when it got none.

        A panel that silently renders "No figures produced yet." is how the
        discrimination tab stayed broken: the run succeeded, the figures were on
        disk, and only the GUI's lookup was wrong.  Every populate now says what it
        found, so a mismatch shows up in the Log tab instead of as a blank pane.
        """
        paths = (figures if isinstance(figures, list)
                 else [p for group in figures.values() for p in group])
        n = sum(1 for p in paths if Path(p).exists())
        if n:
            self._log_msg(f"{tab}: {n} figure(s) loaded into the tab.")
            return
        msg = f"{tab}: NO figures to display."
        if hint:
            msg += " " + hint
        self._log_msg(msg)
        self.statusBar().showMessage(msg, 12000)

    def _on_error(self, tb: str) -> None:
        self._set_busy(False)
        self._pending_pdf = False
        self._suite_status.setText("Run failed — see the Log tab.")
        self._log_msg("ERROR:\n" + tb)
        QMessageBox.critical(self, "Run failed", tb.splitlines()[-1] if tb else "Unknown error")

    def _open_report(self) -> None:
        if not self._last_run:
            QMessageBox.information(self, "No report", "Run an analysis first.")
            return
        try:
            self._open_path(Path(self._last_run.report_path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not open", str(exc))

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        buttons = [self._lc_run_btn, self._abl_run_btn, self._disc_run_btn,
                   self._gen_run_btn, self._al_run_btn, self._rare_run_btn]
        for attr in ("_bscape_run_btn", "_vv_run_btn", "_bench_run_btn",
                     "_suite_run_btn", "_rare_check_btn"):
            if hasattr(self, attr):
                buttons.append(getattr(self, attr))
        for b in buttons:
            b.setEnabled(not busy)
        if not busy:
            self._progress_lbl.setText("Idle.")

    def _log_msg(self, msg: str) -> None:
        self._log.append(msg)


def launch_validation_gui() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = ValidationWindow()
    win.show()
    return app.exec()
