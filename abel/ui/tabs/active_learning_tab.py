"""Active-learning behavior modeling tab.

Runs the closed-loop pipeline:
pose -> context -> representation -> model -> uncertainty -> candidates.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import os
import pickle
import re
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PySide6.QtCore import QThreadPool, Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.models.schemas import BehaviorModelConfig, CandidateWindow
from abel.services.active_learning_scheduler import ActiveLearningScheduler
from abel.services.active_learning_trainer_service import ActiveLearningTrainerService, TrainingConfig
from abel.services.behavior_adaptive_benchmark_service import BehaviorAdaptiveBenchmarkService
from abel.services.behavior_awareness_ablation_service import BehaviorAwarenessAblationService
from abel.services.behavior_service import BehaviorService
from abel.services.behavior_representation_service import BehaviorRepresentationService, RepresentationConfig
from abel.services.candidate_service import CandidateGenerationService, SegmentCandidateGenerationConfig
from abel.services.representation_reuse import reuse_or_build_representation
from abel.services.context_feature_service import ContextFeatureConfig, ContextFeatureService
from abel.services.evaluation_service import BoutMergeConfig, EvaluationService
from abel.services.fusion_inference_service import FusionConfig, FusionInferenceService
from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseProcessingService
from abel.services.roi_service import ROIService
from abel.services.seed_service import SeedService
from abel.services.uncertainty_service import UncertaintyScoringService, UncertaintyWeights
from abel.services.workflow_snapshot_service import WorkflowSnapshot, WorkflowSnapshotService
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml
from abel.utils.eta_estimator import StageEtaEstimator
from abel.workers.task_worker import TaskWorker

logger = logging.getLogger("abel")


NO_BEHAVIOR_ID = "no_behavior"


class PipelineCancelledError(RuntimeError):
    """Raised when the user requests cancellation of the active-learning run."""


@dataclass
class _RunSummary:
    n_sessions: int = 0
    n_frame_rows: int = 0
    n_segment_rows: int = 0
    n_train_rows: int = 0
    n_candidates: int = 0
    model_version: str = "behavior_model_v1"
    f1: float = float("nan")
    pr_auc: float = float("nan")
    model_device_used: str = "cpu"
    fusion_device_used: str = "cpu"
    fallback_reason: str = ""


class ActiveLearningTab(QWidget):
    """End-to-end active-learning interface for target behavior modeling."""

    _pipeline_progress_updated = Signal(int, int, str, str)  # (value, max, log_line, status)
    edge_case_candidates_requested = Signal(list, str)
    # Emitted after every pipeline run so clip extraction can be pre-populated
    # with uncertainty-ranked candidates.  The main window connects this WITHOUT
    # switching the active tab (unlike edge_case_candidates_requested).
    uncertainty_candidates_updated = Signal(list, str)
    # Like uncertainty_candidates_updated but ACCUMULATES into the clip
    # extraction queue instead of replacing the previous set.  Used by batch
    # runs (Retrain All / Pipeline All) so each behavior's review clips add up
    # rather than overwriting clips from the previous model's training.
    uncertainty_candidates_appended = Signal(list, str)

    def __init__(
        self,
        import_service: ImportService,
        seed_service: SeedService,
        behavior_service: BehaviorService,
        candidate_service: CandidateGenerationService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._imports = import_service
        self._seeds = seed_service
        self._behaviors = behavior_service
        self._candidates = candidate_service
        self._trainer = ActiveLearningTrainerService()
        self._context = ContextFeatureService()
        self._pose = PoseProcessingService()
        self._repr = BehaviorRepresentationService()
        self._uncertainty = UncertaintyScoringService()
        self._fusion = FusionInferenceService()
        self._rois = ROIService()
        self._evaluation = EvaluationService()
        self._scheduler = ActiveLearningScheduler()
        self._phase1 = BehaviorAdaptiveBenchmarkService()
        self._project_root: Path | None = None
        self._loading_ui_settings: bool = False
        self._applying_quick_profile: bool = False
        self._co_occurring_enabled: bool = False
        self._excluded_feature_cols: set[str] = set()
        self._selected_session_ids: set[str] = set()
        self._snapshot_svc = WorkflowSnapshotService()
        self._batch_cancel_flag: list[bool] = [False]
        self._pool = QThreadPool.globalInstance()
        self._cancel_flag: list[bool] = [False]
        self._pipeline_progress_updated.connect(self._apply_pipeline_progress)

        # -- graph size settings ------------------------------------------
        self._al_graph_settings: dict[str, Any] = {"max_w": 900, "max_h": 450}

        self._status = QLabel("Open a project to run active learning.")
        self._status.setWordWrap(True)
        self._pipeline_step_scale: int = 1  # set to 100 during pipeline runs for sub-step bar resolution

        self._mode = QComboBox()
        self._mode.addItem("Uncertainty", userData="uncertainty")
        self._mode.addItem("Random Low-Prob (absent)", userData="random_absent")

        self._target_behavior = QComboBox()
        self._target_behavior.addItem("Auto (seed/review dominant)", userData="")

        self._model_name = QLineEdit()
        self._model_name.setPlaceholderText("Optional model name (blank = timestamped)")

        self._saved_model_combo = QComboBox()
        self._refresh_saved_models_btn = QPushButton("Refresh Saved")
        self._refresh_saved_models_btn.clicked.connect(self._refresh_saved_model_options)
        self._load_saved_model_btn = QPushButton("Load Saved")
        self._load_saved_model_btn.clicked.connect(self._load_selected_saved_model)
        self._show_saved_model_settings_btn = QPushButton("Show Settings")
        self._show_saved_model_settings_btn.clicked.connect(self._show_selected_saved_model_settings)

        self._query_size = QSpinBox()
        self._query_size.setRange(0, 99999)
        self._query_size.setSpecialValueText("All")
        self._query_size.setValue(100)

        self._segment_window_value = QLabel("-")
        self._segment_stride_value = QLabel("-")
        self._segment_settings_hint = QLabel(
            "Window/stride are auto-synced from extracted feature settings for this project."
        )
        self._segment_settings_hint.setWordWrap(True)

        self._quick_test = QCheckBox("Enable quick test mode")
        self._quick_test.setToolTip("Use fewer sessions/segments and optional stage skips to validate pipeline wiring quickly.")

        self._balanced_sampling = QCheckBox("Balanced time sampling")
        self._balanced_sampling.setToolTip(
            "Sample a random time window from each session to reduce processing time\n"
            "while maintaining temporal coverage. Unlike quick mode, all pipeline stages\n"
            "(training, evaluation, fusion) still run at full quality.\n\n"
            "Use the 'Sampling minutes/session' and 'Sampling seed' controls below to\n"
            "configure sampling duration and reproducibility."
        )
        self._balanced_sampling_minutes = QSpinBox()
        self._balanced_sampling_minutes.setRange(1, 120)
        self._balanced_sampling_minutes.setValue(5)
        self._balanced_sampling_minutes.setSuffix(" min/session")
        self._balanced_sampling_minutes.setToolTip(
            "How many minutes of video to sample per session.\n"
            "Each session gets a contiguous random chunk of this duration."
        )
        self._balanced_sampling_seed = QSpinBox()
        self._balanced_sampling_seed.setRange(0, 999999)
        self._balanced_sampling_seed.setValue(42)
        self._balanced_sampling_seed.setToolTip(
            "Random seed for reproducible chunk selection. Change to get different time regions."
        )

        self._temperature_scale = QDoubleSpinBox()
        self._temperature_scale.setRange(0.5, 5.0)
        self._temperature_scale.setSingleStep(0.1)
        self._temperature_scale.setValue(1.5)
        self._temperature_scale.setDecimals(1)
        self._temperature_scale.setToolTip(
            "Temperature scaling for prediction probabilities.\n"
            "T=1.0: raw model probabilities (often overconfident).\n"
            "T=1.5: moderate softening (recommended starting point).\n"
            "T=2.0+: aggressive softening for highly overconfident models.\n"
            "Higher values push probabilities toward 0.5."
        )

        self._quick_profile = QComboBox()
        self._quick_profile.addItem("Custom", userData="custom")
        self._quick_profile.addItem("Very Quick (smoke)", userData="very_quick")
        self._quick_profile.addItem("Quick Identification (recommended)", userData="quick_identification")
        self._quick_profile.addItem("Balanced", userData="balanced")
        self._quick_profile.addItem("Large Dataset (80+ subjects)", userData="large_dataset")
        self._quick_profile.addItem("In-Depth", userData="in_depth")
        self._apply_quick_profile_btn = QPushButton("Apply Preset")
        self._apply_quick_profile_btn.clicked.connect(self._apply_selected_quick_profile)

        self._quick_mode_summary = QLabel("Quick mode OFF: full selected-session span.")
        self._quick_mode_summary.setWordWrap(True)

        self._session_scope_summary = QLabel("Session scope: all linked sessions")
        self._session_scope_summary.setWordWrap(True)
        self._select_sessions_btn = QPushButton("Choose Sessions...")
        self._select_sessions_btn.clicked.connect(self._open_session_selection_dialog)

        self._include_imported = QCheckBox("Include imported examples")
        self._include_imported.setChecked(True)
        self._include_imported.setToolTip(
            "Include cross-project examples added via Model Refinement when "
            "training. Uncheck to train on only this project's own labeled data "
            "(imported examples stay in the project; they're just not used here)."
        )

        self._examples_per_session = QSpinBox()
        self._examples_per_session.setRange(0, 10000)
        self._examples_per_session.setSpecialValueText("Unlimited")
        self._examples_per_session.setValue(0)
        self._examples_per_session.setToolTip(
            "Maximum candidate examples extracted per selected session. 0 = no per-session cap."
        )

        self._max_segments = QSpinBox()
        self._max_segments.setRange(0, 1000000)
        self._max_segments.setSpecialValueText("All")
        self._max_segments.setValue(0)

        self._max_train_samples_per_class = QSpinBox()
        self._max_train_samples_per_class.setRange(0, 100000)
        self._max_train_samples_per_class.setSpecialValueText("Unlimited")
        self._max_train_samples_per_class.setValue(0)
        self._max_train_samples_per_class.setToolTip(
            "Cap the number of labeled examples per class used for training.\n"
            "Recommended for datasets with many subjects (e.g. 80+): try 500–2000.\n"
            "0 = use all available labeled examples (default)."
        )

        self._no_behavior_sample_weight = QDoubleSpinBox()
        self._no_behavior_sample_weight.setRange(0.0, 5.0)
        self._no_behavior_sample_weight.setSingleStep(0.1)
        self._no_behavior_sample_weight.setDecimals(1)
        self._no_behavior_sample_weight.setValue(0.0)
        self._no_behavior_sample_weight.setSpecialValueText("Auto")
        self._no_behavior_sample_weight.setToolTip(
            "Weight multiplier for no_behavior samples during training.\n"
            "0 = Auto: adaptively computed from class imbalance ratio.\n"
            "Values >1.0 make the model less confident on positives by\n"
            "emphasizing explicitly labeled no-behavior negatives.\n"
            "Increase to reduce false positives / overconfidence."
        )

        self._quick_ident_minutes = QSpinBox()
        self._quick_ident_minutes.setRange(1, 120)
        self._quick_ident_minutes.setValue(5)
        self._quick_ident_minutes.setSuffix(" min/session")
        self._quick_ident_minutes.setToolTip(
            "In quick test mode, randomly sample this many minutes from each selected session."
        )

        self._quick_ident_seed = QSpinBox()
        self._quick_ident_seed.setRange(0, 999999)
        self._quick_ident_seed.setValue(42)
        self._quick_ident_seed.setToolTip(
            "Random seed for selecting quick-test time windows."
        )

        self._skip_fusion = QCheckBox("Skip video fusion (faster)")
        self._skip_fusion.setChecked(False)

        self._skip_evaluation = QCheckBox("Skip evaluation reports + UMAP (faster)")
        self._skip_evaluation.setChecked(False)
        self._skip_evaluation.setToolTip(
            "Skip evaluation metrics, UMAP separation plots, and PR curves.\n"
            "Uncheck this if you want UMAP plots after training."
        )

        self._enable_umap = QCheckBox("Enable UMAP generation")
        self._enable_umap.setChecked(True)
        self._enable_umap.setToolTip(
            "When enabled, unified UMAP plots are generated after pipeline run,\n"
            "pipeline all, retrain, and retrain all.\n"
            "Disable to skip UMAP generation for faster runs."
        )

        self._reuse_cached_features = QCheckBox("Reuse cached pose/context features when available")
        self._reuse_cached_features.setChecked(True)

        self._remap_reviewed_windows = QCheckBox("Map reviewed clip IDs to current window length")
        self._remap_reviewed_windows.setToolTip(
            "When enabled, reviewer labels from older clip/window sizes are remapped to current segment windows. "
            "Any trailing frames that do not fill a full window are dropped."
        )
        self._remap_reviewed_windows.setChecked(False)

        self._auto_generate_reviewed_windows = QCheckBox("Auto-generate features for all reviewed segments")
        self._auto_generate_reviewed_windows.setToolTip(
            "When enabled, segments from all review sources (bout-based, random, prior window sizes) "
            "that lack pre-computed features are automatically computed on-the-fly before training. "
            "This ensures every reviewed label is included in the training set."
        )
        self._auto_generate_reviewed_windows.setChecked(True)

        self._strict_gpu = QCheckBox("Require GPU (fail if fallback occurs)")
        self._strict_gpu.setChecked(False)

        self._flow_temporal_stride = QSpinBox()
        self._flow_temporal_stride.setRange(1, 10)
        self._flow_temporal_stride.setValue(10)
        self._flow_temporal_stride.setToolTip(
            "Compute optical flow every Nth frame and interpolate between.\n"
            "1 = every frame (slowest, highest fidelity).\n"
            "10 = every 10th frame (default, ~10× faster).\n"
            "Lower values increase fidelity at the cost of speed."
        )

        self._validation_pct = QSpinBox()
        self._validation_pct.setRange(5, 50)
        self._validation_pct.setValue(25)
        self._validation_pct.setSuffix(" %")

        self._split_strategy = QComboBox()
        self._split_strategy.addItem("Group by session", userData="group_shuffle_session")
        self._split_strategy.addItem("Group by subject", userData="group_shuffle_subject")

        self._guided_settings_btn = QPushButton("Suggest Settings From Data…")
        self._guided_settings_btn.setToolTip(
            "Ask a few questions, inspect current labels, and apply recommended active-learning settings."
        )
        self._guided_settings_btn.clicked.connect(self._run_guided_settings_helper)

        self._phase1_enable = QCheckBox("Enable adaptive benchmarks")
        self._phase1_enable.setChecked(False)
        self._phase1_enable.setToolTip(
            "Opt-in Phase 1 benchmarking and diagnostics. Baseline workflow remains unchanged when disabled."
        )

        self._phase1_modality = QCheckBox("Benchmark feature families")
        self._phase1_modality.setChecked(True)
        self._phase1_modality.setToolTip("Compare pose, visual, motion, context, and fused experts.")

        self._phase1_multiscale = QCheckBox("Benchmark time scales")
        self._phase1_multiscale.setChecked(False)
        self._phase1_multiscale.setToolTip("Evaluate AP across multiple temporal windows.")

        self._phase1_confound = QCheckBox("Run confound analysis")
        self._phase1_confound.setChecked(False)
        self._phase1_confound.setToolTip("Estimate top non-target confounds when labels are available.")

        self._phase1_diagnostics = QCheckBox("Generate diagnostics")
        self._phase1_diagnostics.setChecked(True)
        self._phase1_diagnostics.setToolTip(
            "Create baseline-vs-adaptive comparison charts only when enabled."
        )

        self._phase1_regenerate = QCheckBox("Regenerate cached diagnostics")
        self._phase1_regenerate.setChecked(False)

        self._phase1_export_hires = QCheckBox("Export publication quality (PNG+SVG)")
        self._phase1_export_hires.setChecked(True)

        self._phase1_scales = QLineEdit("0.1, 0.2, 0.25, 0.5, 1.0, 2.0")
        self._phase1_scales.setToolTip("Seconds. Example: 0.1,0.2,0.25,0.5,1.0,2.0")

        self._queue_weighted_enable = QCheckBox("Enable weighted queue scoring")
        self._queue_weighted_enable.setChecked(False)
        self._queue_weighted_enable.setToolTip(
            "Phase 2 opt-in: combine modular queue scores (candidate, uncertainty, disagreement, diversity, confound, hard-negative, exploration)."
        )

        self._queue_enable_disagreement = QCheckBox("Use disagreement component")
        self._queue_enable_disagreement.setChecked(True)

        self._queue_enable_diversity = QCheckBox("Use diversity component")
        self._queue_enable_diversity.setChecked(True)

        self._queue_enable_confound = QCheckBox("Use confound-boundary component")
        self._queue_enable_confound.setChecked(True)

        self._queue_enable_hardneg = QCheckBox("Use hard-negative component")
        self._queue_enable_hardneg.setChecked(True)

        self._queue_diversity_mode = QComboBox()
        self._queue_diversity_mode.addItem("Distance to reviewed", userData="distance_to_reviewed")
        self._queue_diversity_mode.addItem("Clustering quota", userData="clustering_quota")

        self._queue_exploration_fraction = QDoubleSpinBox()
        self._queue_exploration_fraction.setRange(0.0, 0.5)
        self._queue_exploration_fraction.setSingleStep(0.01)
        self._queue_exploration_fraction.setDecimals(2)
        self._queue_exploration_fraction.setValue(0.15)

        # High-level candidate focus slider: controls the balance between edge-case
        # candidates (hard negatives, confounds, disagreements, diversity) and
        # strong/confident candidates.  0% = all strong candidates, 100% = all edge cases.
        self._candidate_focus_pct = QSpinBox()
        self._candidate_focus_pct.setRange(0, 100)
        self._candidate_focus_pct.setSingleStep(10)
        self._candidate_focus_pct.setValue(50)
        self._candidate_focus_pct.setSuffix(" % edge cases")
        self._candidate_focus_pct.setToolTip(
            "Controls what fraction of candidates are edge cases (hard negatives, confound "
            "boundaries, disagreements) vs strong/confident predictions.\n"
            "0% = surface only strong candidates.\n"
            "50% = balanced mix (default).\n"
            "100% = surface only edge cases for targeted refinement."
        )
        self._focus_queue_weights: dict[str, float] | None = None
        self._candidate_focus_pct.valueChanged.connect(self._on_candidate_focus_changed)

        self._umap_pred_ratio = QDoubleSpinBox()
        self._umap_pred_ratio.setRange(1.0, 50.0)
        self._umap_pred_ratio.setSingleStep(0.5)
        self._umap_pred_ratio.setDecimals(1)
        self._umap_pred_ratio.setValue(5.0)
        self._umap_pred_ratio.setToolTip(
            "Maximum ratio of predicted (unlabeled) segments to real (reviewed) segments per class\n"
            "in the unified UMAP embedding. Higher values show more predicted context;\n"
            "lower values keep the plot focused on reviewed examples.\n"
            "Default: 5x as many predicted as reviewed."
        )

        self._all_behavior_aware = QCheckBox("Use all-behavior-aware candidate ranking")
        self._all_behavior_aware.setChecked(True)
        self._all_behavior_aware.setToolTip(
            "When enabled, candidate ranking uses competing behavior-model probabilities to prioritize mutually-exclusive boundary cases."
        )

        self._all_behavior_competition_margin = QDoubleSpinBox()
        self._all_behavior_competition_margin.setRange(0.0, 1.0)
        self._all_behavior_competition_margin.setSingleStep(0.01)
        self._all_behavior_competition_margin.setDecimals(2)
        self._all_behavior_competition_margin.setValue(0.05)
        self._all_behavior_competition_margin.setToolTip(
            "Minimum probability gap required for the target behavior to beat a competing behavior.\n"
            "< 1.0 (e.g. 0.05): candidates are included even when the target only marginally outscores competitors — "
            "good for catching subtle or ambiguous events.\n"
            "→ 1.0: only candidates where the target clearly dominates all competitors are kept — "
            "stricter filtering that reduces confound overlap but may miss boundary cases."
        )

        self._settings_btn = QPushButton("⚙ Settings")
        self._settings_btn.setToolTip("Open the full settings panel for all pipeline options.")
        self._settings_btn.clicked.connect(self._open_settings_dialog)

        # --- Compact top-level settings (always visible) ---
        compact_settings = QWidget()
        compact_form = QFormLayout(compact_settings)
        compact_form.setContentsMargins(8, 6, 8, 6)
        compact_form.setSpacing(6)
        compact_form.addRow("Target behavior:", self._target_behavior)
        compact_form.addRow("Selection mode:", self._mode)
        compact_form.addRow("Model name:", self._model_name)
        saved_model_row = QHBoxLayout()
        saved_model_row.addWidget(self._saved_model_combo, 1)
        saved_model_row.addWidget(self._refresh_saved_models_btn)
        saved_model_row.addWidget(self._load_saved_model_btn)
        compact_form.addRow("Saved models:", saved_model_row)

        # --- Preset row: Quick / Standard / Complete / Recommend ---
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        self._preset_quick_btn = QPushButton("⚡ Quick")
        self._preset_quick_btn.setToolTip(
            "Fast iteration (~2-5 min). Samples a subset, skips fusion & evaluation.\n"
            "Skips: fusion inference, evaluation reports, UMAP plots, Phase 1 benchmarks.\n"
            "Best for: early rounds, smoke-testing pipeline, first-pass identification."
        )
        self._preset_standard_btn = QPushButton("◉ Standard")
        self._preset_standard_btn.setToolTip(
            "Balanced speed & quality (~10-20 min). Moderate caps with evaluation + UMAP.\n"
            "Includes: evaluation reports, UMAP plots. Skips: fusion, Phase 1 benchmarks.\n"
            "Best for: mid-stage refinement when you have 20+ labeled reviews."
        )
        self._preset_complete_btn = QPushButton("✦ Complete")
        self._preset_complete_btn.setToolTip(
            "Full-depth run (20-60+ min). No segment caps, full evaluation + fusion + UMAP.\n"
            "Includes: everything — fusion, evaluation, UMAP, Phase 1 benchmarks.\n"
            "Best for: final runs, publication-quality analysis, all-sessions coverage."
        )
        self._preset_recommend_btn = QPushButton("★ Recommend")
        self._preset_recommend_btn.setToolTip(
            "Analyze your dataset (file count, sizes, label balance) and automatically\n"
            "select the best settings for your project."
        )
        for btn in (self._preset_quick_btn, self._preset_standard_btn, self._preset_complete_btn, self._preset_recommend_btn):
            btn.setMinimumHeight(32)
            btn.setStyleSheet(
                "QPushButton { background: #0D2240; color: #90CAF9; border: 1px solid #1565C0; "
                "border-radius: 5px; padding: 4px 14px; font-weight: 700; font-size: 12px; }"
                "QPushButton:hover { background: #1A3A60; border-color: #42A5F5; }"
                "QPushButton:pressed { background: #0A1929; }"
            )
        self._preset_recommend_btn.setStyleSheet(
            "QPushButton { background: #1B2E1B; color: #81C784; border: 1px solid #388E3C; "
            "border-radius: 5px; padding: 4px 14px; font-weight: 700; font-size: 12px; }"
            "QPushButton:hover { background: #2E4D2E; border-color: #66BB6A; }"
            "QPushButton:pressed { background: #0D1F0D; }"
        )
        self._preset_quick_btn.clicked.connect(lambda: self._apply_tiered_preset("quick"))
        self._preset_standard_btn.clicked.connect(lambda: self._apply_tiered_preset("standard"))
        self._preset_complete_btn.clicked.connect(lambda: self._apply_tiered_preset("complete"))
        self._preset_recommend_btn.clicked.connect(self._run_recommend_settings)
        preset_row.addWidget(self._preset_quick_btn)
        preset_row.addWidget(self._preset_standard_btn)
        preset_row.addWidget(self._preset_complete_btn)
        preset_row.addWidget(self._preset_recommend_btn)
        preset_row.addStretch()

        # --- Active settings summary (always visible) ---
        self._active_settings_summary = QLabel("")
        self._active_settings_summary.setWordWrap(True)
        self._active_settings_summary.setStyleSheet(
            "padding: 6px 10px; border: 1px solid #1565C0; border-radius: 4px; "
            "background: #0A1929; color: #B0BEC5; font-size: 11px; font-weight: 600;"
        )

        # --- Session scope ---
        session_scope_row = QHBoxLayout()
        session_scope_row.addWidget(self._session_scope_summary, 1)
        session_scope_row.addWidget(self._select_sessions_btn)
        compact_form.addRow("Candidate focus:", self._candidate_focus_pct)
        compact_form.addRow("Sessions:", session_scope_row)
        compact_form.addRow("Imported data:", self._include_imported)

        form_box = QGroupBox("Active Learning")
        form_layout = QVBoxLayout(form_box)
        form_layout.setSpacing(8)
        form_layout.addWidget(compact_settings)
        form_layout.addLayout(preset_row)
        form_layout.addWidget(self._active_settings_summary)

        # --- Primary action buttons (streamlined) ---
        self._run_btn = QPushButton("▶ Run Pipeline")
        self._run_btn.setMinimumHeight(34)
        self._run_btn.setStyleSheet(
            "QPushButton { background: #1565C0; color: #FFFFFF; border-radius: 5px; "
            "padding: 6px 18px; font-weight: 800; font-size: 13px; }"
            "QPushButton:hover { background: #1976D2; }"
            "QPushButton:disabled { background: #263238; color: #546E7A; }"
        )
        self._run_btn.clicked.connect(self._run_pipeline)

        self._retrain_btn = QPushButton("↻ Retrain")
        self._retrain_btn.setToolTip("Retrain using project review labels without re-running the full pipeline.")
        self._retrain_btn.clicked.connect(self._run_retrain)

        self._retrain_all_btn = QPushButton("↻ Retrain All")
        self._retrain_all_btn.setToolTip(
            "Retrain each behavior individually in sequence using the current settings."
        )
        self._retrain_all_btn.clicked.connect(self._run_retrain_all)

        self._run_existing_btn = QPushButton("⏵ Run Existing Model")
        self._run_existing_btn.setToolTip("Run inference using a previously saved model.")
        self._run_existing_btn.clicked.connect(self._run_existing_model)

        self._run_models_btn = QPushButton("⏵ Run Models…")
        self._run_models_btn.setToolTip(
            "Run inference for selected behaviors using their existing trained\n"
            "models on this project's data — like Pipeline All, but without\n"
            "retraining. Pick which behaviors to score."
        )
        self._run_models_btn.clicked.connect(self._run_models_selected_behaviors)

        self._run_pipeline_all_btn = QPushButton("▶ Pipeline All")
        self._run_pipeline_all_btn.setMinimumHeight(34)
        self._run_pipeline_all_btn.setStyleSheet(
            "QPushButton { background: #0D47A1; color: #FFFFFF; border-radius: 5px; "
            "padding: 6px 14px; font-weight: 800; font-size: 12px; }"
            "QPushButton:hover { background: #1565C0; }"
            "QPushButton:disabled { background: #263238; color: #546E7A; }"
        )
        self._run_pipeline_all_btn.setToolTip(
            "Run the full active-learning pipeline for every defined behavior in sequence.\n"
            "Features are extracted once and reused for subsequent behaviors."
        )
        self._run_pipeline_all_btn.clicked.connect(self._run_pipeline_all_behaviors)

        self._gen_clips_btn = QPushButton("🎬 Generate Review Clips")
        self._gen_clips_btn.setCheckable(True)
        self._gen_clips_btn.setChecked(True)
        self._gen_clips_btn.setToolTip(
            "When enabled, batch runs (Retrain All / Pipeline All) generate review\n"
            "candidate clips for each behavior and add them to the Clips tab.\n"
            "Clips accumulate across runs rather than replacing the previous run's.\n\n"
            "When disabled, batch runs train/score only and skip clip generation\n"
            "(faster).  On by default."
        )
        self._gen_clips_btn.setStyleSheet(
            "QPushButton { background: #1A2027; color: #78909C; border: 1px solid #37474F; "
            "border-radius: 4px; padding: 4px 12px; font-weight: 600; }"
            "QPushButton:checked { background: #0D2B0D; color: #81C784; border: 1px solid #388E3C; }"
        )
        self._gen_clips_btn.toggled.connect(self._on_gen_clips_toggled)

        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.clicked.connect(self._confirm_stop)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background: #37474F; color: #EF9A9A; border-radius: 4px; "
            "padding: 4px 12px; font-weight: 700; }"
            "QPushButton:hover { background: #B71C1C; color: white; }"
            "QPushButton:disabled { background: #1A2027; color: #37474F; }"
        )

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.hide()
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

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(72)
        self._log.setMaximumHeight(110)
        self._log.verticalScrollBar().rangeChanged.connect(
            lambda _min, _max: self._log.verticalScrollBar().setValue(_max)
        )

        # --- Visualization panel ---
        self._viz_title = QLabel("Behavior separation preview")
        self._viz_model_selector = QComboBox()
        self._viz_model_selector.addItem("Latest run", userData="__latest__")
        self._viz_model_selector.currentIndexChanged.connect(self._refresh_visualization_preview)
        self._viz_selector = QComboBox()
        self._viz_selector.addItem("Auto", userData="auto")
        self._viz_selector.addItem("Behavior Separation (UMAP)", userData="umap")
        self._viz_selector.addItem("Confusion Matrix", userData="confusion")
        self._viz_selector.addItem("PR Curve", userData="pr")
        self._viz_selector.addItem("Feature Family Comparison", userData="feature_family")
        self._viz_selector.addItem("Multi-Scale Performance", userData="multiscale")
        self._viz_selector.addItem("Target-vs-Confound Margin", userData="margin")
        self._viz_selector.addItem("Calibration", userData="calibration")
        self._viz_selector.addItem("Queue Composition", userData="queue")
        self._viz_selector.addItem("Pipeline Timing", userData="timing")
        self._viz_selector.addItem("Cross-Behaviour Confounds", userData="confound_cross")
        self._viz_selector.addItem("Unified UMAP (All Behaviours)", userData="unified_umap")
        self._viz_selector.addItem("Unsupervised UMAP (Clusters)", userData="unsupervised_umap")
        self._viz_selector.addItem("Expert Assignment (Per Model)", userData="expert_assignment")
        self._viz_selector.currentIndexChanged.connect(self._on_viz_selection_changed)
        self._viz_help = QLabel("(?)")
        self._update_viz_help_tooltip()
        self._viz_preview = QLabel("Run training/evaluation to generate a separation graph.")
        self._viz_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._viz_preview.setMinimumHeight(220)
        self._viz_preview.setMaximumSize(
            int(self._al_graph_settings.get("max_w", 900)),
            int(self._al_graph_settings.get("max_h", 450)),
        )
        self._viz_preview.setStyleSheet("border: 1px solid #1A2027; background: #0A1929; border-radius: 4px; color: #546E7A;")
        self._viz_pixmap_original: QPixmap | None = None
        self._viz_source_path: Path | None = None
        self._viz_save_btn = QPushButton("Save Visualization...")
        self._viz_save_btn.setToolTip(
            "Save the current visualization from its original file.\n"
            "For UMAP plots, SVG is offered when available for high-quality export."
        )
        self._viz_save_btn.clicked.connect(self._save_visualization)
        self._viz_save_btn.setEnabled(False)

        # Keep the table for internal pipeline use but don't show it in the layout
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Subject", "Frames", "Segment ID", "Model Prob", "Uncertainty"])
        self._table.hide()

        # Also keep edge-case/configure-features/phase1 buttons for settings dialog access
        self._edge_case_btn = QPushButton("Find Edge Cases…")
        self._edge_case_btn.setToolTip(
            "Find windows that are difficult to differentiate between two behaviors."
        )
        self._edge_case_btn.clicked.connect(self._open_edge_case_finder)

        self._configure_features_btn = QPushButton("Configure Features…")
        self._configure_features_btn.setToolTip(
            "Choose which feature columns are included in model training."
        )
        self._configure_features_btn.clicked.connect(self._show_feature_config_dialog)

        self._phase1_run_btn = QPushButton("Run Phase 1 Feature Test")
        self._phase1_run_btn.clicked.connect(self._run_phase1_benchmarks)

        self._confound_graph_btn = QPushButton("Confound Analysis")
        self._confound_graph_btn.setToolTip(
            "Generate a between-behaviour co-activation matrix showing which\n"
            "behaviours overlap and suggestions for improving labels."
        )
        self._confound_graph_btn.clicked.connect(self._generate_confound_graph)

        self._unified_umap_btn = QPushButton("Unified UMAP")
        self._unified_umap_btn.setToolTip(
            "Generate a single UMAP embedding combining predictions from\n"
            "all behaviour models, coloured by dominant behaviour."
        )
        self._unified_umap_btn.clicked.connect(self._generate_unified_umap)

        self._unsupervised_umap_btn = QPushButton("Unsupervised UMAP")
        self._unsupervised_umap_btn.setToolTip(
            "Generate a UMAP embedding directly from the raw segment features —\n"
            "no models or labels required. Points are colour-coded by\n"
            "automatically-discovered clusters (HDBSCAN)."
        )
        self._unsupervised_umap_btn.clicked.connect(self._generate_unsupervised_umap)

        self._umap_select_btn = QPushButton("Select from UMAP")
        self._umap_select_btn.setToolTip(
            "Open an interactive UMAP plot where you can lasso-select\n"
            "segments and send them to clip extraction."
        )
        self._umap_select_btn.clicked.connect(self._open_interactive_umap_selection)

        self._awareness_ablation_btn = QPushButton("Test Behavior Awareness")
        self._awareness_ablation_btn.setToolTip(
            "Run a lightweight ablation study comparing behavior-aware vs.\n"
            "behavior-unaware pipeline modes across candidate ranking,\n"
            "temporal refinement, and model feature quality."
        )
        self._awareness_ablation_btn.clicked.connect(self._run_awareness_ablation)

        # Consolidate the analysis/visualisation actions into a single compact
        # menu button so the toolbar row doesn't overflow and truncate labels.
        self._viz_menu_btn = QToolButton()
        self._viz_menu_btn.setText("Visualize  ▾")
        self._viz_menu_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._viz_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._viz_menu_btn.setToolTip(
            "Analysis & visualisation tools:\n"
            "confound analysis, UMAP embeddings, and behaviour-awareness tests."
        )
        _viz_menu = QMenu(self._viz_menu_btn)
        for _label, _btn in (
            ("Confound Analysis", self._confound_graph_btn),
            ("Unified UMAP (behaviours)", self._unified_umap_btn),
            ("Unsupervised UMAP (clusters)", self._unsupervised_umap_btn),
            ("Select from UMAP…", self._umap_select_btn),
            ("Test Behavior Awareness", self._awareness_ablation_btn),
        ):
            _act = _viz_menu.addAction(_label)
            _act.setToolTip(_btn.toolTip())
            _act.triggered.connect(_btn.click)
        self._viz_menu_btn.setMenu(_viz_menu)

        # --- Layout assembly ---
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._run_pipeline_all_btn)
        btn_row.addWidget(self._retrain_btn)
        btn_row.addWidget(self._retrain_all_btn)
        btn_row.addWidget(self._run_existing_btn)
        btn_row.addWidget(self._run_models_btn)
        btn_row.addWidget(self._gen_clips_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._viz_menu_btn)
        btn_row.addWidget(self._settings_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        root.addWidget(form_box)
        root.addLayout(btn_row)
        root.addWidget(self._status)
        root.addWidget(self._progress)
        root.addWidget(self._log)

        viz_head = QHBoxLayout()
        viz_head.addWidget(self._viz_title)
        viz_head.addStretch(1)
        viz_head.addWidget(QLabel("Model:"))
        viz_head.addWidget(self._viz_model_selector)
        viz_head.addWidget(QLabel("Graph:"))
        viz_head.addWidget(self._viz_selector)
        viz_head.addWidget(self._viz_save_btn)
        viz_head.addWidget(self._viz_help)
        _al_graph_size_btn = QPushButton("Graph Size\u2026")
        _al_graph_size_btn.setToolTip("Set maximum display width and height for the visualization panel.")
        _al_graph_size_btn.clicked.connect(self._open_al_graph_size_dialog)
        viz_head.addWidget(_al_graph_size_btn)

        root.addLayout(viz_head)
        root.addWidget(self._viz_preview, 1)

        self._bind_project_setting_persistence()
        self._bind_quick_mode_summary_updates()
        self._refresh_quick_mode_summary()

    def set_project(self, project_root: Path) -> None:
        self._reset_runtime_state_for_project_switch()
        self._project_root = project_root
        self._status.setText(f"Loading project\u2026")
        # Defer all service I/O and refresh calls so the tab switch paints immediately.
        from PySide6.QtCore import QTimer  # noqa: PLC0415
        QTimer.singleShot(0, lambda: self._deferred_project_init(project_root))

    def _open_al_graph_size_dialog(self) -> None:
        """Open a dialog to set max display width/height for the visualization preview."""
        from PySide6.QtWidgets import (  # noqa: PLC0415
            QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QVBoxLayout,
        )
        gs = self._al_graph_settings
        dlg = QDialog(self)
        dlg.setWindowTitle("Graph Size")
        dlg.resize(300, 130)
        form = QFormLayout()

        max_w_spin = QSpinBox(dlg)
        max_w_spin.setRange(200, 3840)
        max_w_spin.setSingleStep(50)
        max_w_spin.setSuffix(" px")
        max_w_spin.setToolTip("Maximum display width of the visualization panel in pixels.")
        max_w_spin.setValue(int(gs.get("max_w", 900)))

        max_h_spin = QSpinBox(dlg)
        max_h_spin.setRange(150, 2160)
        max_h_spin.setSingleStep(50)
        max_h_spin.setSuffix(" px")
        max_h_spin.setToolTip("Maximum display height of the visualization panel in pixels.")
        max_h_spin.setValue(int(gs.get("max_h", 450)))

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
        self._viz_preview.setMaximumSize(gs["max_w"], gs["max_h"])
        self._refresh_visualization_preview()

    def _is_umap_enabled(self) -> bool:
        """Check whether UMAP generation is enabled in the active-learning settings."""
        return bool(self._enable_umap.isChecked())

    def _deferred_project_init(self, project_root: Path) -> None:
        if self._project_root != project_root:
            return
        self._behaviors.set_project(project_root)
        self._seeds.set_project(project_root)
        self._candidates.set_project(project_root)
        self._scheduler.set_project(project_root)
        self._refresh_behavior_options()
        self._refresh_saved_model_options()
        self._refresh_viz_model_options()
        self._load_ui_settings_from_project()
        self._refresh_session_scope_summary()
        self._refresh_segment_settings_display()
        self._refresh_active_settings_summary()
        self._status.setText(f"Project ready: {project_root}")
        self._refresh_visualization_preview()

    def _reset_runtime_state_for_project_switch(self) -> None:
        self._cancel_flag[0] = False
        self._set_busy(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Idle")
        self._log.clear()

    def _reset_ui_settings_to_defaults(self) -> None:
        idx_mode = self._mode.findData("uncertainty")
        if idx_mode >= 0:
            self._mode.setCurrentIndex(idx_mode)
        self._target_behavior.setCurrentIndex(0)
        self._saved_model_combo.setCurrentIndex(0)
        self._model_name.clear()
        self._query_size.setValue(100)
        self._quick_test.setChecked(False)
        self._examples_per_session.setValue(0)
        self._max_segments.setValue(0)
        self._quick_ident_minutes.setValue(5)
        self._quick_ident_seed.setValue(42)
        idx_profile = self._quick_profile.findData("custom")
        if idx_profile >= 0:
            self._quick_profile.setCurrentIndex(idx_profile)
        self._include_imported.setChecked(True)
        self._skip_fusion.setChecked(False)
        self._skip_evaluation.setChecked(False)
        self._enable_umap.setChecked(True)
        self._reuse_cached_features.setChecked(True)
        self._remap_reviewed_windows.setChecked(True)
        self._auto_generate_reviewed_windows.setChecked(True)
        self._strict_gpu.setChecked(False)
        self._all_behavior_aware.setChecked(True)
        self._all_behavior_competition_margin.setValue(0.05)
        self._umap_pred_ratio.setValue(5.0)
        self._validation_pct.setValue(25)
        idx_split = self._split_strategy.findData("group_shuffle_session")
        if idx_split >= 0:
            self._split_strategy.setCurrentIndex(idx_split)
        self._phase1_enable.setChecked(False)
        self._phase1_modality.setChecked(True)
        self._phase1_multiscale.setChecked(False)
        self._phase1_confound.setChecked(False)
        self._phase1_diagnostics.setChecked(True)
        self._phase1_regenerate.setChecked(False)
        self._phase1_export_hires.setChecked(True)
        self._phase1_scales.setText("0.1, 0.2, 0.25, 0.5, 1.0, 2.0")
        self._candidate_focus_pct.setValue(50)
        self._queue_weighted_enable.setChecked(False)
        self._queue_enable_disagreement.setChecked(True)
        self._queue_enable_diversity.setChecked(True)
        self._queue_enable_confound.setChecked(True)
        self._queue_enable_hardneg.setChecked(True)
        self._selected_session_ids = set()
        idx_div = self._queue_diversity_mode.findData("distance_to_reviewed")
        if idx_div >= 0:
            self._queue_diversity_mode.setCurrentIndex(idx_div)
        self._queue_exploration_fraction.setValue(0.15)
        self._excluded_feature_cols = set()

    def _bind_project_setting_persistence(self) -> None:
        self._mode.currentIndexChanged.connect(lambda _i: self._persist_ui_settings_to_project())
        self._target_behavior.currentIndexChanged.connect(lambda _i: self._persist_ui_settings_to_project())
        self._saved_model_combo.currentIndexChanged.connect(lambda _i: self._persist_ui_settings_to_project())
        self._model_name.textChanged.connect(lambda _t: self._persist_ui_settings_to_project())
        self._query_size.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._quick_test.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._examples_per_session.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._max_segments.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._max_train_samples_per_class.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._quick_ident_minutes.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._quick_ident_seed.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._quick_profile.currentIndexChanged.connect(lambda _i: self._persist_ui_settings_to_project())
        self._include_imported.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._skip_fusion.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._skip_evaluation.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._enable_umap.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._reuse_cached_features.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._remap_reviewed_windows.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._auto_generate_reviewed_windows.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._strict_gpu.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._all_behavior_aware.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._all_behavior_competition_margin.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._validation_pct.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._split_strategy.currentIndexChanged.connect(lambda _i: self._persist_ui_settings_to_project())
        self._phase1_enable.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._phase1_modality.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._phase1_multiscale.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._phase1_confound.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._phase1_diagnostics.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._phase1_regenerate.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._phase1_export_hires.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._phase1_scales.textChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._queue_weighted_enable.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._queue_enable_disagreement.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._queue_enable_diversity.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._queue_enable_confound.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._queue_enable_hardneg.toggled.connect(lambda _v: self._persist_ui_settings_to_project())
        self._queue_diversity_mode.currentIndexChanged.connect(lambda _i: self._persist_ui_settings_to_project())
        self._queue_exploration_fraction.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())

    def _bind_quick_mode_summary_updates(self) -> None:
        self._quick_test.toggled.connect(lambda _v: self._refresh_quick_mode_summary())
        self._examples_per_session.valueChanged.connect(lambda _v: self._refresh_quick_mode_summary())
        self._max_segments.valueChanged.connect(lambda _v: self._refresh_quick_mode_summary())
        self._max_train_samples_per_class.valueChanged.connect(lambda _v: self._refresh_quick_mode_summary())
        self._quick_ident_minutes.valueChanged.connect(lambda _v: self._refresh_quick_mode_summary())
        self._quick_ident_seed.valueChanged.connect(lambda _v: self._refresh_quick_mode_summary())
        self._skip_fusion.toggled.connect(lambda _v: self._refresh_quick_mode_summary())
        self._skip_evaluation.toggled.connect(lambda _v: self._refresh_quick_mode_summary())
        self._enable_umap.toggled.connect(lambda _v: self._refresh_quick_mode_summary())
        self._quick_test.toggled.connect(lambda _v: self._mark_quick_profile_custom())
        self._examples_per_session.valueChanged.connect(lambda _v: self._mark_quick_profile_custom())
        self._max_segments.valueChanged.connect(lambda _v: self._mark_quick_profile_custom())
        self._quick_ident_minutes.valueChanged.connect(lambda _v: self._mark_quick_profile_custom())
        self._skip_fusion.toggled.connect(lambda _v: self._mark_quick_profile_custom())
        self._skip_evaluation.toggled.connect(lambda _v: self._mark_quick_profile_custom())
        self._enable_umap.toggled.connect(lambda _v: self._mark_quick_profile_custom())
        self._query_size.valueChanged.connect(lambda _v: self._mark_quick_profile_custom())
        self._validation_pct.valueChanged.connect(lambda _v: self._mark_quick_profile_custom())

    def _mark_quick_profile_custom(self) -> None:
        if self._loading_ui_settings or self._applying_quick_profile:
            return
        if str(self._quick_profile.currentData() or "custom") == "custom":
            return
        idx = self._quick_profile.findData("custom")
        if idx >= 0:
            self._quick_profile.setCurrentIndex(idx)

    @staticmethod
    def _quick_profile_definitions() -> dict[str, dict[str, Any]]:
        return {
            "very_quick": {
                "quick_test": True,
                "examples_per_session": 8,
                "quick_ident_minutes": 2,
                "max_segments": 4000,
                "skip_fusion": True,
                "skip_evaluation": True,
                "enable_umap": False,
                "query_size": 40,
                "validation_pct": 20,
            },
            "quick_identification": {
                "quick_test": True,
                "examples_per_session": 20,
                "quick_ident_minutes": 5,
                "max_segments": 12000,
                "skip_fusion": True,
                "skip_evaluation": False,
                "enable_umap": True,
                "query_size": 80,
                "validation_pct": 25,
            },
            "balanced": {
                "quick_test": True,
                "examples_per_session": 40,
                "quick_ident_minutes": 8,
                "max_segments": 25000,
                "skip_fusion": False,
                "skip_evaluation": False,
                "enable_umap": True,
                "query_size": 120,
                "validation_pct": 25,
            },
            "in_depth": {
                "quick_test": False,
                "examples_per_session": 0,
                "quick_ident_minutes": 15,
                "max_segments": 0,
                "skip_fusion": False,
                "skip_evaluation": False,
                "enable_umap": True,
                "query_size": 180,
                "validation_pct": 30,
            },
            "large_dataset": {
                "quick_test": False,
                "examples_per_session": 15,
                "quick_ident_minutes": 10,
                "max_segments": 50000,
                "skip_fusion": True,
                "skip_evaluation": False,
                "query_size": 100,
                "validation_pct": 25,
                "max_train_samples_per_class": 1000,
                "phase1_enable": False,
                "phase1_scales": "1.0",
            },
        }

    def _apply_selected_quick_profile(self) -> None:
        key = str(self._quick_profile.currentData() or "custom")
        presets = self._quick_profile_definitions()
        if key not in presets:
            self._status.setText("Quick profile is set to Custom.")
            self._refresh_quick_mode_summary()
            return

        cfg = presets[key]
        self._applying_quick_profile = True
        try:
            self._quick_test.setChecked(bool(cfg.get("quick_test", False)))
            self._examples_per_session.setValue(int(cfg.get("examples_per_session", self._examples_per_session.value())))
            self._quick_ident_minutes.setValue(int(cfg.get("quick_ident_minutes", self._quick_ident_minutes.value())))
            self._max_segments.setValue(int(cfg.get("max_segments", self._max_segments.value())))
            self._max_train_samples_per_class.setValue(int(cfg.get("max_train_samples_per_class", self._max_train_samples_per_class.value())))
            self._skip_fusion.setChecked(bool(cfg.get("skip_fusion", self._skip_fusion.isChecked())))
            self._skip_evaluation.setChecked(bool(cfg.get("skip_evaluation", self._skip_evaluation.isChecked())))
            self._enable_umap.setChecked(bool(cfg.get("enable_umap", self._enable_umap.isChecked())))
            self._query_size.setValue(int(cfg.get("query_size", self._query_size.value())))
            self._validation_pct.setValue(int(cfg.get("validation_pct", self._validation_pct.value())))
            if "phase1_enable" in cfg:
                self._phase1_enable.setChecked(bool(cfg["phase1_enable"]))
            if "phase1_scales" in cfg:
                self._phase1_scales.setText(str(cfg["phase1_scales"]))
        finally:
            self._applying_quick_profile = False

        self._persist_ui_settings_to_project()
        self._refresh_quick_mode_summary()
        self._status.setText(f"Applied quick profile: {self._quick_profile.currentText()}.")
        self._append_log(f"Applied quick profile: {self._quick_profile.currentText()}.")

    def _refresh_quick_mode_summary(self) -> None:
        _examples_cap = int(self._examples_per_session.value())
        _examples_text = "unlimited" if _examples_cap <= 0 else f"{_examples_cap}"
        if not self._quick_test.isChecked():
            _active_limits: list[str] = []
            _seg_cap = int(self._max_segments.value())
            _train_cap = int(self._max_train_samples_per_class.value())
            if _examples_cap > 0:
                _active_limits.append(f"examples/session: {_examples_cap}")
            if _seg_cap > 0:
                _active_limits.append(f"segment cap: {_seg_cap:,}")
            if _train_cap > 0:
                _active_limits.append(f"training cap: {_train_cap:,}/class")
            if _active_limits:
                self._quick_mode_summary.setText(
                    "Quick mode OFF. Active limits: " + ", ".join(_active_limits) + "."
                )
                self._quick_mode_summary.setStyleSheet(
                    "padding: 4px; border: 1px solid #d9b24c; background: #fff7e0;"
                )
            else:
                self._quick_mode_summary.setText(
                    "Quick mode OFF. Pipeline uses all sessions in current session scope, all segments, all labels."
                )
                self._quick_mode_summary.setStyleSheet(
                    "padding: 4px; border: 1px solid #c8c8c8; background: #f6f6f6;"
                )
            return

        seg_cap = int(self._max_segments.value())
        seg_cap_text = "All" if seg_cap <= 0 else str(seg_cap)
        skip_eval = self._skip_evaluation.isChecked()
        skip_fus = self._skip_fusion.isChecked()
        umap_on = self._enable_umap.isChecked()
        skips: list[str] = []
        if skip_fus:
            skips.append("fusion")
        if skip_eval:
            skips.append("evaluation/UMAP")
        if not umap_on:
            skips.append("UMAP generation")
        skips_text = ", ".join(skips) if skips else "none"
        includes_text = "evaluation + UMAP" if (not skip_eval and umap_on) else ""
        summary = (
            "Quick mode ON. "
            f"{_examples_text} example(s)/session, "
            f"{int(self._quick_ident_minutes.value())} min/session, "
            f"segment cap {seg_cap_text}. "
            f"Skipping: {skips_text}."
        )
        if includes_text:
            summary += f"  Includes: {includes_text}."
        self._quick_mode_summary.setText(summary)
        self._quick_mode_summary.setStyleSheet("padding: 4px; border: 1px solid #d9b24c; background: #fff7e0;")
        self._refresh_active_settings_summary()

    # ------------------------------------------------------------------
    # Tiered presets: Quick / Standard / Complete
    # ------------------------------------------------------------------

    def _apply_tiered_preset(self, tier: str) -> None:
        """Apply one of the three simplified presets."""
        presets = {
            "quick": {
                "quick_test": True,
                "examples_per_session": 10,
                "quick_ident_minutes": 3,
                "max_segments": 8000,
                "max_train_samples_per_class": 0,
                "skip_fusion": True,
                "skip_evaluation": True,
                "query_size": 60,
                "validation_pct": 20,
                "phase1_enable": False,
                "queue_weighted_enable": False,
            },
            "standard": {
                "quick_test": True,
                "examples_per_session": 30,
                "quick_ident_minutes": 8,
                "max_segments": 25000,
                "max_train_samples_per_class": 0,
                "skip_fusion": True,
                "skip_evaluation": False,
                "query_size": 120,
                "validation_pct": 25,
                "phase1_enable": False,
                "queue_weighted_enable": False,
            },
            "complete": {
                "quick_test": False,
                "examples_per_session": 0,
                "quick_ident_minutes": 15,
                "max_segments": 0,
                "max_train_samples_per_class": 0,
                "skip_fusion": False,
                "skip_evaluation": False,
                "query_size": 200,
                "validation_pct": 30,
                "phase1_enable": True,
                "phase1_diagnostics": True,
                "phase1_modality": True,
                "phase1_confound": True,
                "queue_weighted_enable": True,
                "queue_enable_disagreement": True,
                "queue_enable_diversity": True,
                "queue_enable_confound": True,
                "queue_enable_hardneg": True,
            },
        }
        cfg = presets.get(tier)
        if cfg is None:
            return

        self._applying_quick_profile = True
        try:
            self._quick_test.setChecked(bool(cfg.get("quick_test", False)))
            self._examples_per_session.setValue(int(cfg.get("examples_per_session", 0)))
            self._quick_ident_minutes.setValue(int(cfg.get("quick_ident_minutes", 5)))
            self._max_segments.setValue(int(cfg.get("max_segments", 0)))
            self._max_train_samples_per_class.setValue(int(cfg.get("max_train_samples_per_class", 0)))
            self._skip_fusion.setChecked(bool(cfg.get("skip_fusion", False)))
            self._skip_evaluation.setChecked(bool(cfg.get("skip_evaluation", False)))
            self._enable_umap.setChecked(bool(cfg.get("enable_umap", True)))
            self._query_size.setValue(int(cfg.get("query_size", 100)))
            self._validation_pct.setValue(int(cfg.get("validation_pct", 25)))
            if "phase1_enable" in cfg:
                self._phase1_enable.setChecked(bool(cfg["phase1_enable"]))
            if "phase1_diagnostics" in cfg:
                self._phase1_diagnostics.setChecked(bool(cfg["phase1_diagnostics"]))
            if "phase1_modality" in cfg:
                self._phase1_modality.setChecked(bool(cfg["phase1_modality"]))
            if "phase1_confound" in cfg:
                self._phase1_confound.setChecked(bool(cfg["phase1_confound"]))
            if "queue_weighted_enable" in cfg:
                self._queue_weighted_enable.setChecked(bool(cfg["queue_weighted_enable"]))
            if "queue_enable_disagreement" in cfg:
                self._queue_enable_disagreement.setChecked(bool(cfg["queue_enable_disagreement"]))
            if "queue_enable_diversity" in cfg:
                self._queue_enable_diversity.setChecked(bool(cfg["queue_enable_diversity"]))
            if "queue_enable_confound" in cfg:
                self._queue_enable_confound.setChecked(bool(cfg["queue_enable_confound"]))
            if "queue_enable_hardneg" in cfg:
                self._queue_enable_hardneg.setChecked(bool(cfg["queue_enable_hardneg"]))
            # Mark the old quick_profile as custom since we use the new tiered system
            idx = self._quick_profile.findData("custom")
            if idx >= 0:
                self._quick_profile.setCurrentIndex(idx)
        finally:
            self._applying_quick_profile = False

        self._persist_ui_settings_to_project()
        self._refresh_quick_mode_summary()
        tier_label = tier.capitalize()
        self._status.setText(f"Applied {tier_label} preset.")
        self._append_log(f"Applied {tier_label} preset.")

    # ------------------------------------------------------------------
    # Recommend settings from dataset analysis
    # ------------------------------------------------------------------

    def _run_recommend_settings(self) -> None:
        """Analyze the current project data to recommend optimal settings."""
        if self._project_root is None:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return

        # 1. Gather dataset metrics
        manifest = self._imports.load_manifest(self._project_root)
        n_sessions = len(manifest.linked_sessions) if manifest and manifest.linked_sessions else 0
        if n_sessions == 0:
            QMessageBox.information(
                self, "Recommend Settings",
                "No sessions imported yet. Import data first, then use Recommend."
            )
            return

        n_subjects = len({s.subject_id for s in manifest.linked_sessions if s.subject_id}) if manifest else 0
        n_seeds = len(self._seeds.seeds)
        target_behavior = self._selected_target_behavior_id()
        stats = self._collect_review_balance_stats(target_behavior)
        pos_labels = int(stats.get("positive", 0))
        neg_labels = int(stats.get("negative", 0))
        total_labels = pos_labels + neg_labels

        # Estimate total data size from video files
        total_video_mb = 0.0
        if manifest:
            for vid in manifest.videos:
                try:
                    p = Path(vid.source_path)
                    if p.exists():
                        total_video_mb += p.stat().st_size / (1024 * 1024)
                except Exception:
                    pass

        # 2. Determine recommendations
        reasons: list[str] = []

        # Dataset size classification
        if n_sessions <= 5:
            size_class = "small"
            reasons.append(f"{n_sessions} sessions — small dataset, Complete preset recommended")
        elif n_sessions <= 30:
            size_class = "medium"
            reasons.append(f"{n_sessions} sessions — medium dataset, Standard preset recommended")
        else:
            size_class = "large"
            reasons.append(f"{n_sessions} sessions — large dataset, Quick preset with caps")

        # Label maturity
        if total_labels == 0:
            label_stage = "initial"
            reasons.append("No reviewed labels yet — early exploration phase")
        elif total_labels < 30:
            label_stage = "early"
            reasons.append(f"{total_labels} labels — still in early annotation rounds")
        elif total_labels < 100:
            label_stage = "developing"
            reasons.append(f"{total_labels} labels — developing model, standard depth appropriate")
        else:
            label_stage = "mature"
            reasons.append(f"{total_labels} labels — mature dataset, full evaluation valuable")

        # Apply recommended tier
        if size_class == "large" or (size_class == "medium" and label_stage == "initial"):
            tier = "quick"
            extra: dict[str, Any] = {}
            if n_sessions > 60:
                extra["max_train_samples_per_class"] = 1500
                reasons.append("80+ sessions: training class cap set to 1500")
        elif label_stage == "mature" and size_class != "large":
            tier = "complete"
            extra = {}
            reasons.append("Mature labels + manageable size: enabling full evaluation")
        else:
            tier = "standard"
            extra = {}

        # Subject-aware split strategy
        split = "group_shuffle_subject" if n_subjects >= 3 else "group_shuffle_session"
        reasons.append(f"Split strategy: {split} ({n_subjects} unique subjects)")

        # Validation fraction based on label count
        if total_labels < 20:
            val_pct = 20
        elif total_labels < 60:
            val_pct = 25
        else:
            val_pct = 30

        # Apply the tier first
        self._apply_tiered_preset(tier)

        # Then override with data-specific values
        self._applying_quick_profile = True
        try:
            self._validation_pct.setValue(val_pct)
            idx_split = self._split_strategy.findData(split)
            if idx_split >= 0:
                self._split_strategy.setCurrentIndex(idx_split)
            if "max_train_samples_per_class" in extra:
                self._max_train_samples_per_class.setValue(int(extra["max_train_samples_per_class"]))
        finally:
            self._applying_quick_profile = False

        self._persist_ui_settings_to_project()
        self._refresh_quick_mode_summary()

        # Show summary to user
        summary_lines = [
            f"<b>Recommended preset: {tier.capitalize()}</b>",
            "",
            "<b>Dataset analysis:</b>",
        ]
        summary_lines.extend(f"  • {r}" for r in reasons)
        summary_lines.extend([
            "",
            f"<b>Settings applied:</b>",
            f"  • Validation: {val_pct}%",
            f"  • Split: {split}",
            f"  • Video data: {total_video_mb:,.0f} MB across {n_sessions} sessions",
        ])
        if n_seeds > 0:
            summary_lines.append(f"  • Seed examples: {n_seeds}")
        if total_labels > 0:
            summary_lines.append(f"  • Reviewed labels: {pos_labels} positive, {neg_labels} negative")

        QMessageBox.information(
            self, "Recommended Settings",
            "<br>".join(summary_lines),
        )
        self._status.setText(f"Applied recommended settings ({tier.capitalize()} preset).")
        self._append_log(f"Recommend: applied {tier} preset — " + "; ".join(reasons))

    # ------------------------------------------------------------------
    # Active settings summary (always-visible badge)
    # ------------------------------------------------------------------

    def _refresh_active_settings_summary(self) -> None:
        """Update the visible summary of currently enabled settings."""
        parts: list[str] = []

        mode = str(self._mode.currentData() or "uncertainty")
        parts.append(f"Mode: <b>{mode}</b>")

        qsize = int(self._query_size.value())
        parts.append(f"Query: <b>{qsize if qsize > 0 else 'All'}</b>")

        if self._quick_test.isChecked():
            parts.append("Quick: <b>ON</b>")
        seg_cap = int(self._max_segments.value())
        if seg_cap > 0:
            parts.append(f"Seg cap: <b>{seg_cap:,}</b>")
        train_cap = int(self._max_train_samples_per_class.value())
        if train_cap > 0:
            parts.append(f"Train cap: <b>{train_cap:,}/class</b>")

        val_pct = int(self._validation_pct.value())
        parts.append(f"Val: <b>{val_pct}%</b>")

        if self._skip_fusion.isChecked():
            parts.append("Fusion: <b>OFF</b>")
        if self._skip_evaluation.isChecked():
            parts.append("Eval: <b>OFF</b>")
        if self._all_behavior_aware.isChecked():
            parts.append("Multi-behavior: <b>ON</b>")
        if self._phase1_enable.isChecked():
            parts.append("Benchmarks: <b>ON</b>")
        if self._queue_weighted_enable.isChecked():
            parts.append("Weighted queue: <b>ON</b>")

        focus = int(self._candidate_focus_pct.value())
        if focus != 50:
            parts.append(f"Edge focus: <b>{focus}%</b>")

        self._active_settings_summary.setText("  ·  ".join(parts))

    # ------------------------------------------------------------------
    # High-level candidate focus → queue weight mapping
    # ------------------------------------------------------------------

    def _on_candidate_focus_changed(self, pct: int) -> None:
        """Map the 0-100 'edge case focus' percentage to queue component weights.

        0 %  → mostly strong/confident candidates (high candidate weight)
        50 % → balanced defaults
        100% → mostly edge cases (disagreement, confound, hard-neg heavy)
        """
        t = max(0.0, min(1.0, pct / 100.0))  # normalise to [0, 1]

        # Linearly interpolate between 'strong' and 'edge' weight profiles.
        # strong (t=0):  candidate=0.70, unc=0.15, dis=0.05, div=0.03, conf=0.03, hn=0.02, exp=0.02
        # balanced (t=0.5): candidate=0.35, unc=0.20, dis=0.15, div=0.10, conf=0.10, hn=0.07, exp=0.03
        # edge (t=1):    candidate=0.05, unc=0.25, dis=0.25, div=0.12, conf=0.15, hn=0.13, exp=0.05
        def _lerp(a: float, b: float) -> float:
            return round(a + (b - a) * t, 3)

        self._focus_queue_weights = {
            "candidate":     _lerp(0.70, 0.05),
            "uncertainty":   _lerp(0.15, 0.25),
            "disagreement":  _lerp(0.05, 0.25),
            "diversity":     _lerp(0.03, 0.12),
            "confound":      _lerp(0.03, 0.15),
            "hard_negative": _lerp(0.02, 0.13),
            "exploration":   _lerp(0.02, 0.05),
        }

        # Auto-enable weighted queue when the user moves off centre.
        if pct != 50 and not self._queue_weighted_enable.isChecked():
            self._queue_weighted_enable.setChecked(True)

        self._persist_ui_settings_to_project()
        self._refresh_active_settings_summary()

    def _ui_settings_payload(self) -> dict[str, Any]:
        return {
            "mode": str(self._mode.currentData() or "uncertainty"),
            "excluded_feature_cols": sorted(self._excluded_feature_cols),
            "target_behavior_id": str(self._target_behavior.currentData() or ""),
            "model_name": str(self._model_name.text() or ""),
            "saved_model": str(self._saved_model_combo.currentData() or ""),
            "query_size": int(self._query_size.value()),
            "quick_test": bool(self._quick_test.isChecked()),
            "examples_per_session": int(self._examples_per_session.value()),
            "selected_session_ids": sorted(str(s) for s in self._selected_session_ids),
            "max_segments": int(self._max_segments.value()),
            "max_train_samples_per_class": int(self._max_train_samples_per_class.value()),
            "no_behavior_sample_weight": float(self._no_behavior_sample_weight.value()),
            "include_imported": bool(self._include_imported.isChecked()),
            "quick_ident_minutes": int(self._quick_ident_minutes.value()),
            "quick_ident_seed": int(self._quick_ident_seed.value()),
            "quick_profile": str(self._quick_profile.currentData() or "custom"),
            "skip_fusion": bool(self._skip_fusion.isChecked()),
            "skip_evaluation": bool(self._skip_evaluation.isChecked()),
            "enable_umap": bool(self._enable_umap.isChecked()),
            "reuse_cached_features": bool(self._reuse_cached_features.isChecked()),
            "remap_reviewed_windows": bool(self._remap_reviewed_windows.isChecked()),
            "auto_generate_reviewed_windows": bool(self._auto_generate_reviewed_windows.isChecked()),
            "strict_gpu": bool(self._strict_gpu.isChecked()),
            "flow_temporal_stride": int(self._flow_temporal_stride.value()),
            "all_behavior_aware": bool(self._all_behavior_aware.isChecked()),
            "all_behavior_competition_margin": float(self._all_behavior_competition_margin.value()),
            "umap_pred_ratio": float(self._umap_pred_ratio.value()),
            "validation_pct": int(self._validation_pct.value()),
            "split_strategy": str(self._split_strategy.currentData() or "group_shuffle_session"),
            "phase1_enable": bool(self._phase1_enable.isChecked()),
            "phase1_modality": bool(self._phase1_modality.isChecked()),
            "phase1_multiscale": bool(self._phase1_multiscale.isChecked()),
            "phase1_confound": bool(self._phase1_confound.isChecked()),
            "phase1_diagnostics": bool(self._phase1_diagnostics.isChecked()),
            "phase1_regenerate": bool(self._phase1_regenerate.isChecked()),
            "phase1_export_hires": bool(self._phase1_export_hires.isChecked()),
            "phase1_scales": str(self._phase1_scales.text() or ""),
            "candidate_focus_pct": int(self._candidate_focus_pct.value()),
            "queue_weighted_enable": bool(self._queue_weighted_enable.isChecked()),
            "queue_enable_disagreement": bool(self._queue_enable_disagreement.isChecked()),
            "queue_enable_diversity": bool(self._queue_enable_diversity.isChecked()),
            "queue_enable_confound": bool(self._queue_enable_confound.isChecked()),
            "queue_enable_hardneg": bool(self._queue_enable_hardneg.isChecked()),
            "queue_diversity_mode": str(self._queue_diversity_mode.currentData() or "distance_to_reviewed"),
            "queue_exploration_fraction": float(self._queue_exploration_fraction.value()),
            "balanced_sampling": bool(self._balanced_sampling.isChecked()),
            "balanced_sampling_minutes": int(self._balanced_sampling_minutes.value()),
            "balanced_sampling_seed": int(self._balanced_sampling_seed.value()),
            "temperature_scale": float(self._temperature_scale.value()),
        }

    def _keypoint_aliases(self) -> dict[str, str]:
        """Project-level keypoint rename map ({data_name: canonical_name}).

        Written by the Data Import tab when imported pose files use keypoint
        names that differ from the project's canonical scheme.  Applied during
        pose/context feature extraction so derived feature columns stay
        consistent across the project.
        """
        if self._project_root is None:
            return {}
        data = read_json(self._project_root / "config" / "keypoint_aliases.json", {})
        return {str(k): str(v) for k, v in data.items() if str(k) and str(v)}

    def _persist_ui_settings_to_project(self) -> None:
        if self._project_root is None or self._loading_ui_settings:
            return
        path = self._project_root / "project.yaml"
        raw = read_yaml(path, {})

        payload = self._ui_settings_payload()
        raw["active_learning_ui"] = payload

        model = dict(raw.get("behavior_model") or {})
        model["active_learning_query_size"] = int(payload["query_size"])
        model["query_strategy"] = str(payload["mode"])
        model["evaluation_split_strategy"] = str(payload["split_strategy"])
        raw["behavior_model"] = model

        write_yaml(path, raw)
        self._persist_phase1_settings_to_config(payload)
        self._refresh_active_settings_summary()

    def _load_ui_settings_from_project(self) -> None:
        if self._project_root is None:
            return
        path = self._project_root / "project.yaml"
        raw = read_yaml(path, {})
        ui = dict(raw.get("active_learning_ui") or {})
        model = dict(raw.get("behavior_model") or {})

        self._loading_ui_settings = True
        try:
            self._reset_ui_settings_to_defaults()
            mode = str(ui.get("mode") or model.get("query_strategy") or "").strip()
            if mode:
                idx = self._mode.findData(mode)
                if idx >= 0:
                    self._mode.setCurrentIndex(idx)

            target = str(ui.get("target_behavior_id") or "").strip()
            idx_target = self._target_behavior.findData(target)
            if idx_target >= 0:
                self._target_behavior.setCurrentIndex(idx_target)

            self._model_name.setText(str(ui.get("model_name") or ""))

            saved = str(ui.get("saved_model") or "").strip()
            if saved:
                idx_saved = self._saved_model_combo.findData(saved)
                if idx_saved >= 0:
                    self._saved_model_combo.setCurrentIndex(idx_saved)

            self._query_size.setValue(int(ui.get("query_size", model.get("active_learning_query_size", 100))))
            self._quick_test.setChecked(bool(ui.get("quick_test", False)))
            self._examples_per_session.setValue(int(ui.get("examples_per_session", 0)))
            self._selected_session_ids = {
                str(v).strip()
                for v in list(ui.get("selected_session_ids") or [])
                if str(v).strip()
            }
            self._max_segments.setValue(int(ui.get("max_segments", 0)))
            self._max_train_samples_per_class.setValue(int(ui.get("max_train_samples_per_class", 0)))
            self._no_behavior_sample_weight.setValue(float(ui.get("no_behavior_sample_weight", 0.0)))
            self._quick_ident_minutes.setValue(int(ui.get("quick_ident_minutes", 5)))
            self._quick_ident_seed.setValue(int(ui.get("quick_ident_seed", 42)))
            profile_key = str(ui.get("quick_profile", "custom") or "custom")
            idx_profile = self._quick_profile.findData(profile_key)
            if idx_profile >= 0:
                self._quick_profile.setCurrentIndex(idx_profile)
            self._include_imported.setChecked(bool(ui.get("include_imported", True)))
            self._skip_fusion.setChecked(bool(ui.get("skip_fusion", False)))
            self._skip_evaluation.setChecked(bool(ui.get("skip_evaluation", False)))
            self._enable_umap.setChecked(bool(ui.get("enable_umap", True)))
            self._reuse_cached_features.setChecked(bool(ui.get("reuse_cached_features", True)))
            self._remap_reviewed_windows.setChecked(bool(ui.get("remap_reviewed_windows", True)))
            self._auto_generate_reviewed_windows.setChecked(bool(ui.get("auto_generate_reviewed_windows", True)))
            self._strict_gpu.setChecked(bool(ui.get("strict_gpu", False)))
            self._flow_temporal_stride.setValue(int(ui.get("flow_temporal_stride", 10)))
            self._all_behavior_aware.setChecked(bool(ui.get("all_behavior_aware", True)))
            self._all_behavior_competition_margin.setValue(float(ui.get("all_behavior_competition_margin", 0.05)))
            self._umap_pred_ratio.setValue(float(ui.get("umap_pred_ratio", 5.0)))
            self._co_occurring_enabled = bool(model.get("allow_co_occurring_behaviors", False))
            self._validation_pct.setValue(int(ui.get("validation_pct", 25)))

            split = str(ui.get("split_strategy") or model.get("evaluation_split_strategy") or "").strip()
            if split:
                idx_split = self._split_strategy.findData(split)
                if idx_split >= 0:
                    self._split_strategy.setCurrentIndex(idx_split)

            self._phase1_enable.setChecked(bool(ui.get("phase1_enable", False)))
            self._phase1_modality.setChecked(bool(ui.get("phase1_modality", True)))
            self._phase1_multiscale.setChecked(bool(ui.get("phase1_multiscale", True)))
            self._phase1_confound.setChecked(bool(ui.get("phase1_confound", True)))
            self._phase1_diagnostics.setChecked(bool(ui.get("phase1_diagnostics", True)))
            self._phase1_regenerate.setChecked(bool(ui.get("phase1_regenerate", False)))
            self._phase1_export_hires.setChecked(bool(ui.get("phase1_export_hires", True)))
            self._phase1_scales.setText(str(ui.get("phase1_scales", "0.1, 0.2, 0.25, 0.5, 1.0, 2.0")))
            self._candidate_focus_pct.setValue(int(ui.get("candidate_focus_pct", 50)))
            self._queue_weighted_enable.setChecked(bool(ui.get("queue_weighted_enable", False)))
            self._queue_enable_disagreement.setChecked(bool(ui.get("queue_enable_disagreement", True)))
            self._queue_enable_diversity.setChecked(bool(ui.get("queue_enable_diversity", True)))
            self._queue_enable_confound.setChecked(bool(ui.get("queue_enable_confound", True)))
            self._queue_enable_hardneg.setChecked(bool(ui.get("queue_enable_hardneg", True)))
            idx_div = self._queue_diversity_mode.findData(str(ui.get("queue_diversity_mode", "distance_to_reviewed")))
            if idx_div >= 0:
                self._queue_diversity_mode.setCurrentIndex(idx_div)
            self._queue_exploration_fraction.setValue(float(ui.get("queue_exploration_fraction", 0.15)))
            self._balanced_sampling.setChecked(bool(ui.get("balanced_sampling", False)))
            self._balanced_sampling_minutes.setValue(int(ui.get("balanced_sampling_minutes", 5)))
            self._balanced_sampling_seed.setValue(int(ui.get("balanced_sampling_seed", 42)))
            self._temperature_scale.setValue(float(ui.get("temperature_scale", 1.5)))
            self._excluded_feature_cols = set(
                list(ui.get("excluded_feature_cols") or [])
            )
        finally:
            self._loading_ui_settings = False
        self._load_phase1_settings_from_config()
        self._refresh_quick_mode_summary()
        self._refresh_session_scope_summary()

    def _session_options_from_manifest(self) -> list[tuple[str, str, str]]:
        if self._project_root is None:
            return []
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None or not manifest.linked_sessions:
            return []
        video_by_id = {v.asset_id: v for v in (manifest.videos or [])}
        rows: list[tuple[str, str, str]] = []
        for linked in manifest.linked_sessions:
            sid = str(linked.session_id)
            subject = str(linked.subject_id or "")
            # Derive session type from video filename (strip subject prefix)
            stype = ""
            video = video_by_id.get(linked.video_asset_id)
            if video:
                stem = Path(video.source_path).stem
                if subject and stem.startswith(subject):
                    stype = stem[len(subject):].lstrip("_- ")
                elif not subject:
                    stype = stem
            label = f"{sid}"
            if subject:
                label += f"  |  subject: {subject}"
            if stype:
                label += f"  |  type: {stype}"
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
            self._session_scope_summary.setText(f"Session scope: all linked sessions ({total})")
            return

        valid_selected = {sid for sid in self._selected_session_ids if sid in all_ids}
        if valid_selected != self._selected_session_ids:
            self._selected_session_ids = valid_selected
            self._persist_ui_settings_to_project()

        selected_n = len(self._selected_session_ids)
        if selected_n <= 0:
            self._session_scope_summary.setText(
                "Session scope: no selected sessions match current manifest"
            )
            return
        self._session_scope_summary.setText(f"Session scope: {selected_n}/{total} session(s) selected")

    def _open_session_selection_dialog(self) -> None:
        if self._project_root is None:
            QMessageBox.warning(self, "Active Learning", "Open a project first.")
            return

        options = self._session_options_from_manifest()
        if not options:
            QMessageBox.information(
                self,
                "Active Learning",
                "No linked sessions available. Import data first.",
            )
            return

        all_ids = {sid for sid, _subj, _label in options}
        current = set(self._selected_session_ids) if self._selected_session_ids else set(all_ids)

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Sessions For Active Learning")
        dlg.resize(560, 640)

        info = QLabel(
            "Only selected sessions will be used in Active Learning. "
            "Unselected sessions are excluded from preprocessing, representations, and candidate generation.",
            dlg,
        )
        info.setWordWrap(True)

        list_widget = QListWidget(dlg)
        for sid, _subject, label in options:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(Qt.ItemDataRole.UserRole, sid)
            item.setCheckState(Qt.CheckState.Checked if sid in current else Qt.CheckState.Unchecked)
            list_widget.addItem(item)

        select_all_btn = QPushButton("Select All", dlg)
        deselect_all_btn = QPushButton("Deselect All", dlg)

        def _set_all(state: Qt.CheckState) -> None:
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                item.setCheckState(state)

        select_all_btn.clicked.connect(lambda: _set_all(Qt.CheckState.Checked))
        deselect_all_btn.clicked.connect(lambda: _set_all(Qt.CheckState.Unchecked))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
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
                self,
                "Active Learning",
                "Select at least one session.",
            )
            return

        self._selected_session_ids = set(selected_ids)
        self._persist_ui_settings_to_project()
        self._refresh_session_scope_summary()
        self._status.setText(f"Active-learning session scope updated: {len(self._selected_session_ids)} selected.")

    def _resolve_linked_sessions_for_active_learning(self, manifest: Any) -> list[Any]:
        linked_sessions = list(manifest.linked_sessions)
        if not self._selected_session_ids:
            return linked_sessions

        selected = set(self._selected_session_ids)
        filtered = [s for s in linked_sessions if str(s.session_id) in selected]
        if not filtered:
            raise ValueError(
                "No selected sessions are available in the current manifest. "
                "Use 'Choose Sessions...' to update the active-learning session scope."
            )

        available_ids = {str(s.session_id) for s in linked_sessions}
        valid_selected = selected & available_ids
        if valid_selected != self._selected_session_ids:
            self._selected_session_ids = valid_selected
            self._persist_ui_settings_to_project()
            self._refresh_session_scope_summary()

        return filtered

    # ------------------------------------------------------------------
    # Feature column configuration
    # ------------------------------------------------------------------

    def _discover_available_feature_cols(self) -> list[str]:
        """Return the feature column list from the last representation manifest.

        Falls back to reading parquet headers if the manifest is absent.  Returns
        an empty list when no data has been extracted yet.
        """
        if self._project_root is None:
            return []
        manifest = self._project_root / "derived" / "representations" / "representations.manifest.json"
        if manifest.exists():
            try:
                data = read_json(manifest, {})
                cols = list(data.get("feature_columns") or [])
                if cols:
                    return cols
            except Exception:
                pass
        # Fallback: read schema from parquets without loading all row data.
        pose_path = self._project_root / "derived" / "pose_features" / "frame_pose.parquet"
        ctx_path = self._project_root / "derived" / "context_features" / "frame_context.parquet"
        if pose_path.exists():
            try:
                import pyarrow.parquet as pq  # pyarrow is a pandas/parquet dep
                ignored = {"frame", "animal_id", "session_id", "video_id"}
                pose_cols = [c for c in pq.read_schema(pose_path).names if c not in ignored]
                ctx_cols = [c for c in pq.read_schema(ctx_path).names if c not in ignored] if ctx_path.exists() else []
                return list(dict.fromkeys(pose_cols + ctx_cols))
            except Exception:
                pass
        return []

    def _show_feature_config_dialog(self) -> None:
        """Open the feature-selection dialog.

        Shows all discovered numeric feature columns grouped by source (pose
        kinematics, per-keypoint kinematics, context/environment).  Unchecked
        columns are added to ``_excluded_feature_cols`` and excluded from the
        next representation-building step.
        """
        cols = self._discover_available_feature_cols()
        if not cols:
            QMessageBox.information(
                self,
                "Configure Features",
                "No feature columns found.\n\n"
                "Run the full pipeline once so ABEL can discover which features\n"
                "are available for this project's tracking model.",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Configure Features")
        dlg.setMinimumWidth(520)
        dlg.setMinimumHeight(600)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(
            "Uncheck features to exclude them from model training.\n"
            "Changes take effect on the next pipeline or retrain run."
        ))

        scroll = QScrollArea(dlg)
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(2)

        checkboxes: dict[str, QCheckBox] = {}

        # Classify columns into display groups by naming convention.
        _pose_prefixes = (
            "forepaw_", "nose_velocity", "head_pitch", "body_orientation",
            "centroid_velocity", "forepaw_movement_frequency", "oscillation_energy",
            "forepaw_autocorr_peak",
        )
        _kp_suffixes = (
            "_velocity_x", "_velocity_y", "_speed", "_acceleration", "_jerk",
        )
        pose_cols = [c for c in cols if any(c.startswith(p) for p in _pose_prefixes)]
        kp_cols = [c for c in cols if c not in pose_cols and any(c.endswith(s) for s in _kp_suffixes)]
        ctx_cols = [c for c in cols if c not in pose_cols and c not in kp_cols]

        def _add_group(title: str, members: list[str]) -> None:
            if not members:
                return
            lbl = QLabel(f"<b>{title}</b>")
            content_layout.addWidget(lbl)
            for col in sorted(members):
                cb = QCheckBox(col, content)
                cb.setChecked(col not in self._excluded_feature_cols)
                content_layout.addWidget(cb)
                checkboxes[col] = cb
            content_layout.addSpacing(6)

        _add_group("Pose kinematics (body-level)", pose_cols)
        _add_group("Per-keypoint kinematics", kp_cols)
        _add_group("Context / environment features", ctx_cols)
        content_layout.addStretch()
        scroll.setWidget(content)

        sel_row = QHBoxLayout()
        sel_all_btn = QPushButton("Select All")
        clr_all_btn = QPushButton("Clear All")
        sel_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb in checkboxes.values()])
        clr_all_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in checkboxes.values()])
        n_lbl = QLabel()
        n_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def _update_count() -> None:
            n_on = sum(1 for cb in checkboxes.values() if cb.isChecked())
            n_lbl.setText(f"{n_on}/{len(checkboxes)} included")

        for cb in checkboxes.values():
            cb.toggled.connect(lambda _: _update_count())
        _update_count()

        sel_row.addWidget(sel_all_btn)
        sel_row.addWidget(clr_all_btn)
        sel_row.addStretch()
        sel_row.addWidget(n_lbl)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout.addWidget(scroll, 1)
        layout.addLayout(sel_row)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._excluded_feature_cols = {col for col, cb in checkboxes.items() if not cb.isChecked()}
        self._persist_ui_settings_to_project()
        n_excluded = len(self._excluded_feature_cols)
        n_total = len(cols)
        self._status.setText(
            f"Feature configuration saved: {n_total - n_excluded}/{n_total} features included"
            + (f" ({n_excluded} excluded)." if n_excluded else ".")
        )

    def _refresh_behavior_options(self) -> None:
        current = self._target_behavior.currentData()
        self._target_behavior.blockSignals(True)
        self._target_behavior.clear()
        self._target_behavior.addItem("Auto (seed/review dominant)", userData="")
        for b in self._behaviors.behaviors:
            self._target_behavior.addItem(b.name, userData=b.behavior_id)
        idx = self._target_behavior.findData(current)
        self._target_behavior.setCurrentIndex(max(0, idx))
        self._target_behavior.blockSignals(False)

    def _selected_target_behavior_id(self) -> str:
        # Thread-safe override from Pipeline-All.
        override = getattr(self, "_pipeline_all_target_override", None)
        if override:
            return str(override)

        chosen = str(self._target_behavior.currentData() or "").strip()
        if chosen:
            return chosen

        # Auto mode: choose most represented behavior from seeds and reviewed labels.
        counts: dict[str, int] = {}
        for s in self._seeds.seeds:
            if s.label_type == "positive" and s.behavior_id:
                counts[s.behavior_id] = counts.get(s.behavior_id, 0) + 1
        if self._project_root is not None:
            path = self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
            if path.exists():
                try:
                    lbl = pd.read_parquet(path)
                    for val in lbl.get("review_label", pd.Series(dtype=str)).astype(str):
                        if not val or val.startswith("not_") or val in {"ambiguous", "boundary_error", NO_BEHAVIOR_ID}:
                            continue
                        counts[val] = counts.get(val, 0) + 1
                except Exception:
                    pass
        if counts:
            return max(counts.items(), key=lambda kv: kv[1])[0]
        for behavior in self._behaviors.behaviors:
            bid = str(behavior.behavior_id).strip()
            if bid and bid != NO_BEHAVIOR_ID:
                return bid
        return "target_behavior"

    def _behavior_display_name(self, bid: str) -> str:
        """Return the human-readable name for a behavior ID, falling back to the ID itself."""
        for b in self._behaviors.behaviors:
            if str(b.behavior_id).strip() == str(bid).strip():
                return str(b.name)
        return bid

    def _resolved_query_size_limit(self) -> int:
        value = int(self._query_size.value())
        # 0 means uncapped selection in the candidate service.
        return max(0, value)

    def _collect_review_balance_stats(self, target_behavior: str) -> dict[str, int]:
        stats = {
            "positive": 0,
            "negative": 0,
            "ambiguous": 0,
            "other": 0,
            "sessions": 0,
            "subjects": 0,
        }
        if self._project_root is None:
            return stats

        review_path = self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if review_path.exists():
            try:
                lbl = pd.read_parquet(review_path)
                if not lbl.empty and "review_label" in lbl.columns:
                    for raw in lbl["review_label"].astype(str):
                        token = raw.strip()
                        if not token:
                            continue
                        if token in {"ambiguous", "boundary_error"}:
                            stats["ambiguous"] += 1
                        elif token == target_behavior:
                            stats["positive"] += 1
                        elif token.startswith("not_") or token == NO_BEHAVIOR_ID:
                            stats["negative"] += 1
                        else:
                            stats["other"] += 1
                    if "session_id" in lbl.columns:
                        stats["sessions"] = int(lbl["session_id"].astype(str).nunique())
            except Exception:
                pass

        seed_pos = sum(
            1
            for s in self._seeds.seeds
            if str(s.label_type).strip().lower() == "positive"
            and str(s.behavior_id).strip() == target_behavior
        )
        seed_neg = sum(
            1
            for s in self._seeds.seeds
            if str(s.label_type).strip().lower() != "positive"
            and str(s.behavior_id).strip() == target_behavior
        )
        stats["positive"] += int(seed_pos)
        stats["negative"] += int(seed_neg)

        seg_path = self._project_root / "derived" / "representations" / "segment_features.parquet"
        if seg_path.exists():
            try:
                seg = pd.read_parquet(seg_path, columns=["animal_id", "session_id"])
                if "animal_id" in seg.columns:
                    stats["subjects"] = int(seg["animal_id"].astype(str).nunique())
                if stats["sessions"] <= 0 and "session_id" in seg.columns:
                    stats["sessions"] = int(seg["session_id"].astype(str).nunique())
            except Exception:
                pass

        return stats

    def _set_behavior_model_fields(self, updates: dict[str, Any]) -> None:
        if self._project_root is None:
            return
        path = self._project_root / "project.yaml"
        raw = read_yaml(path, {})
        model = dict(raw.get("behavior_model") or {})
        model.update(updates)
        raw["behavior_model"] = model
        write_yaml(path, raw)

    def _run_guided_settings_helper(self) -> None:
        if self._project_root is None:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return

        target_behavior = self._selected_target_behavior_id()
        stats = self._collect_review_balance_stats(target_behavior)
        pos = int(stats.get("positive", 0))
        neg = int(stats.get("negative", 0))
        sessions = int(stats.get("sessions", 0))
        subjects = int(stats.get("subjects", 0))

        assumption_options = [
            "Rare positives (recommended default)",
            "Roughly balanced classes",
            "Unsure",
        ]
        assumption, ok = QInputDialog.getItem(
            self,
            "Guided Active-Learning Setup",
            (
                "Class-balance assumption:\n"
                "Choose the expected prevalence for your target behavior."
            ),
            assumption_options,
            0,
            False,
        )
        if not ok:
            return

        goal_options = [
            "Higher recall (catch more positives)",
            "Balanced precision/recall",
            "Higher precision (fewer false positives)",
        ]
        goal, ok = QInputDialog.getItem(
            self,
            "Guided Active-Learning Setup",
            "Primary objective for this run:",
            goal_options,
            0,
            False,
        )
        if not ok:
            return

        speed_options = ["Standard", "Fast turnaround", "Thorough"]
        speed, ok = QInputDialog.getItem(
            self,
            "Guided Active-Learning Setup",
            "Runtime preference:",
            speed_options,
            0,
            False,
        )
        if not ok:
            return

        observed_skew_negative = neg >= max(1, pos)
        assume_negative_skew = (
            assumption == "Rare positives (recommended default)"
            or assumption == "Unsure"
            or observed_skew_negative
        )

        # Default policy intentionally assumes more negatives than positives.
        hard_neg_ratio = 0.7 if assume_negative_skew else 0.5
        if goal == "Higher precision (fewer false positives)":
            hard_neg_ratio = max(hard_neg_ratio, 0.8)
        elif goal == "Higher recall (catch more positives)":
            hard_neg_ratio = min(hard_neg_ratio, 0.6)

        if pos <= 15:
            validation_pct = 20
        elif pos <= 40:
            validation_pct = 25
        else:
            validation_pct = 30
        if speed == "Fast turnaround":
            validation_pct = max(15, validation_pct - 5)

        if speed == "Fast turnaround":
            query_size = 60
            max_segments = 12000
            skip_fusion = True
            skip_eval = True
        elif speed == "Thorough":
            query_size = 180
            max_segments = 0
            skip_fusion = False
            skip_eval = False
        else:
            query_size = 100
            max_segments = 0
            skip_fusion = False
            skip_eval = False

        split_data = "group_shuffle_subject" if subjects >= 3 else "group_shuffle_session"
        mode_data = "uncertainty"

        idx_mode = self._mode.findData(mode_data)
        if idx_mode >= 0:
            self._mode.setCurrentIndex(idx_mode)
        idx_split = self._split_strategy.findData(split_data)
        if idx_split >= 0:
            self._split_strategy.setCurrentIndex(idx_split)

        self._query_size.setValue(int(query_size))
        self._validation_pct.setValue(int(validation_pct))
        self._max_segments.setValue(int(max_segments))
        self._skip_fusion.setChecked(bool(skip_fusion))
        self._skip_evaluation.setChecked(bool(skip_eval))
        self._reuse_cached_features.setChecked(True)
        self._remap_reviewed_windows.setChecked(True)
        self._auto_generate_reviewed_windows.setChecked(True)
        self._quick_test.setChecked(False)
        self._strict_gpu.setChecked(False)

        self._set_behavior_model_fields(
            {
                "hard_negative_sampling_ratio": float(hard_neg_ratio),
                "active_learning_query_size": int(query_size),
                "query_strategy": str(mode_data),
                "evaluation_split_strategy": str(split_data),
            }
        )
        self._persist_ui_settings_to_project()

        summary = (
            f"Applied guided settings for '{target_behavior}'.\n\n"
            f"Observed labels: positive={pos}, negative={neg}, ambiguous={int(stats.get('ambiguous', 0))}, "
            f"other={int(stats.get('other', 0))}.\n"
            f"Observed groups: sessions={sessions}, subjects={subjects}.\n\n"
            f"Applied: mode=uncertainty, split={split_data}, validation={validation_pct}%, "
            f"query_size={query_size}, hard_negative_ratio={hard_neg_ratio:.2f}, "
            f"skip_fusion={skip_fusion}, skip_evaluation={skip_eval}."
        )
        self._append_log(summary)
        self._status.setText("Guided active-learning settings applied.")
        QMessageBox.information(self, "Guided Settings Applied", summary)

    def _parse_phase1_scales(self) -> list[float]:
        raw = str(self._phase1_scales.text() or "").replace(";", ",")
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        values: list[float] = []
        for token in parts:
            try:
                val = float(token)
            except ValueError:
                continue
            if val > 0.0:
                values.append(val)
        if not values:
            values = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0]
        return sorted(set(values))

    def _persist_phase1_settings_to_config(self, payload: dict[str, Any]) -> None:
        if self._project_root is None:
            return
        settings = self._phase1.load_or_init_settings(self._project_root)
        phase1 = dict(settings.get("phase1") or {})
        phase1.update(
            {
                "enabled": bool(payload.get("phase1_enable", False)),
                "enable_modality_benchmarking": bool(payload.get("phase1_modality", True)),
                "enable_multiscale_benchmarking": bool(payload.get("phase1_multiscale", True)),
                "enable_confound_analysis": bool(payload.get("phase1_confound", True)),
                "diagnostics_enabled": bool(payload.get("phase1_diagnostics", True)),
                "regenerate_diagnostics": bool(payload.get("phase1_regenerate", False)),
                "export_high_resolution": bool(payload.get("phase1_export_hires", True)),
                # Keep Phase 1 subset/quick behavior consistent with the main
                # pipeline quick-test toggle visible in the UI.
                "quick_feature_test": bool(payload.get("quick_test", False)),
                "scales_sec": self._parse_phase1_scales(),
            }
        )
        settings["phase1"] = phase1
        self._phase1.save_settings(self._project_root, settings)

    def _load_phase1_settings_from_config(self) -> None:
        if self._project_root is None:
            return
        settings = self._phase1.load_or_init_settings(self._project_root)
        phase1 = dict(settings.get("phase1") or {})
        self._loading_ui_settings = True
        try:
            self._phase1_enable.setChecked(bool(phase1.get("enabled", self._phase1_enable.isChecked())))
            self._phase1_modality.setChecked(bool(phase1.get("enable_modality_benchmarking", self._phase1_modality.isChecked())))
            self._phase1_multiscale.setChecked(bool(phase1.get("enable_multiscale_benchmarking", self._phase1_multiscale.isChecked())))
            self._phase1_confound.setChecked(bool(phase1.get("enable_confound_analysis", self._phase1_confound.isChecked())))
            self._phase1_diagnostics.setChecked(bool(phase1.get("diagnostics_enabled", self._phase1_diagnostics.isChecked())))
            self._phase1_regenerate.setChecked(bool(phase1.get("regenerate_diagnostics", self._phase1_regenerate.isChecked())))
            self._phase1_export_hires.setChecked(bool(phase1.get("export_high_resolution", self._phase1_export_hires.isChecked())))
            scales = phase1.get("scales_sec", self._parse_phase1_scales())
            if isinstance(scales, list):
                self._phase1_scales.setText(", ".join(str(float(v)).rstrip("0").rstrip(".") for v in scales))
        finally:
            self._loading_ui_settings = False

    def _run_phase1_benchmarks(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return
        target_behavior = self._selected_target_behavior_id()
        target_behavior_label = self._behavior_display_name(target_behavior)
        self._persist_ui_settings_to_project()
        self._set_busy(True)
        self._progress.setRange(0, 0)
        self._progress.setFormat("Running…")
        self._status.setText(f"Running Phase 1 benchmarks for '{target_behavior_label}'…")
        self._append_log(f"Starting Phase 1 benchmarks for {target_behavior_label}.")
        worker = TaskWorker(self._run_phase1_benchmarks_task, target_behavior, self._pipeline_progress_updated.emit)
        worker.signals.finished.connect(self._on_phase1_finished)
        worker.signals.failed.connect(self._on_failed)
        self._pool.start(worker)

    def _run_phase1_benchmarks_task(
        self,
        target_behavior: str,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, Any]:
        assert self._project_root is not None

        def _phase1_progress(msg: str) -> None:
            if progress_cb is not None:
                progress_cb(0, 1, msg, "Running Phase 1 benchmarks…")

        result = self._phase1.run_phase1(
            project_root=self._project_root,
            target_behavior=target_behavior,
            progress_cb=_phase1_progress,
            force=True,
        )
        return result

    def _on_phase1_finished(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        if not payload or not bool(payload.get("enabled", False)):
            self._status.setText("Phase 1 benchmarks are disabled.")
            self._append_log("Phase 1 run skipped because the module is disabled.")
            return
        cards = list(payload.get("summary_cards") or [])
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setFormat("Complete")
        self._status.setText("Phase 1 benchmarking complete. Diagnostics and summaries were saved.")
        self._append_log(f"Phase 1 benchmark summary: {payload.get('benchmark_summary_path', '')}")
        for card in cards:
            self._append_log(f"- {card}")
        self._refresh_visualization_preview()

    def _open_settings_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Active Learning — Advanced Settings")
        dlg.resize(740, 800)

        form_host = QWidget(dlg)
        form = QFormLayout(form_host)
        form.setSpacing(6)

        # Helper to create a labeled section separator
        def _section(title: str) -> QLabel:
            lbl = QLabel(f"<b style='color:#42A5F5;'>{title}</b>")
            lbl.setStyleSheet("padding: 6px 0 2px 0;")
            return lbl

        # Helper to add info tooltip to a widget
        def _info(widget, when: str, workload: str) -> None:
            widget.setToolTip(f"When valuable: {when}\nWorkload impact: {workload}")

        # ── Core ──
        form.addRow(_section("Core Pipeline"))

        mode = QComboBox(dlg)
        for i in range(self._mode.count()):
            mode.addItem(self._mode.itemText(i), self._mode.itemData(i))
        mode.setCurrentIndex(max(0, mode.findData(self._mode.currentData())))
        _info(mode,
              "Uncertainty is best for most iterative AL workflows. Prototype for cluster-based exploration. "
              "Novelty for finding rare events. Low-prob to fill gaps.",
              "Negligible — affects only candidate ranking, not training cost.")
        form.addRow("Selection mode:", mode)

        query_size = QSpinBox(dlg)
        query_size.setRange(0, 99999)
        query_size.setSpecialValueText("All")
        query_size.setValue(int(self._query_size.value()))
        _info(query_size,
              "How many candidate clips are generated per run. Higher = more review work.",
              "Minimal — only affects output volume. Has no effect on training duration.")
        form.addRow("Query size:", query_size)

        validation_pct = QSpinBox(dlg)
        validation_pct.setRange(5, 50)
        validation_pct.setSuffix(" %")
        validation_pct.setValue(int(self._validation_pct.value()))
        _info(validation_pct,
              "Holdout % for evaluation. Higher = better metric estimates but fewer training samples. "
              "Use 20-25% for small datasets, 30% for mature ones.",
              "No impact on runtime.")
        form.addRow("Validation holdout:", validation_pct)

        split_strategy = QComboBox(dlg)
        for i in range(self._split_strategy.count()):
            split_strategy.addItem(self._split_strategy.itemText(i), self._split_strategy.itemData(i))
        split_strategy.setCurrentIndex(max(0, split_strategy.findData(self._split_strategy.currentData())))
        _info(split_strategy,
              "Group-by-subject prevents data leakage when ≥3 subjects exist. "
              "Group-by-session is appropriate for single-subject or when each session is independent.",
              "No impact on runtime.")
        form.addRow("Split strategy:", split_strategy)

        # ── Speed & Scope ──
        form.addRow(_section("Speed & Scope"))

        quick_test = QCheckBox("Enable quick test mode", dlg)
        quick_test.setChecked(bool(self._quick_test.isChecked()))
        _info(quick_test,
              "Samples a time window from each session instead of processing everything. "
              "Essential for early smoke-testing and rapid iteration.",
              "Major — reduces runtime from 20+ min to 2-5 min.")
        form.addRow(quick_test)

        quick_profile = QComboBox(dlg)
        for i in range(self._quick_profile.count()):
            quick_profile.addItem(self._quick_profile.itemText(i), self._quick_profile.itemData(i))
        quick_profile.setCurrentIndex(max(0, quick_profile.findData(self._quick_profile.currentData())))
        form.addRow("Legacy preset:", quick_profile)

        examples_per_session = QSpinBox(dlg)
        examples_per_session.setRange(0, 10000)
        examples_per_session.setSpecialValueText("Unlimited")
        examples_per_session.setValue(int(self._examples_per_session.value()))
        _info(examples_per_session,
              "Caps per-session candidates. Useful when sessions vary widely in duration.",
              "Reduces candidate generation time proportionally.")
        form.addRow("Examples per session:", examples_per_session)

        max_segments = QSpinBox(dlg)
        max_segments.setRange(0, 1000000)
        max_segments.setSpecialValueText("All")
        max_segments.setValue(int(self._max_segments.value()))
        _info(max_segments,
              "Caps total segment pool before training. Critical for large datasets (30+ sessions) "
              "to keep training fast.",
              "High — training time scales linearly with segment count.")
        form.addRow("Max segments:", max_segments)

        max_train_samples_per_class = QSpinBox(dlg)
        max_train_samples_per_class.setRange(0, 100000)
        max_train_samples_per_class.setSpecialValueText("Unlimited")
        max_train_samples_per_class.setValue(int(self._max_train_samples_per_class.value()))
        _info(max_train_samples_per_class,
              "Critical for datasets with 80+ subjects. Try 500-2000. "
              "Prevents memory issues and training slowdown from class imbalance.",
              "High — directly limits training data size.")
        form.addRow("Training cap per class:", max_train_samples_per_class)

        no_behavior_weight = QDoubleSpinBox(dlg)
        no_behavior_weight.setRange(0.0, 5.0)
        no_behavior_weight.setSingleStep(0.1)
        no_behavior_weight.setDecimals(1)
        no_behavior_weight.setValue(float(self._no_behavior_sample_weight.value()))
        no_behavior_weight.setSpecialValueText("Auto")
        _info(no_behavior_weight,
              "Upweights no_behavior samples during training. "
              "Auto = adaptively computed from class imbalance ratio. "
              "Manual values >1.0 penalize false positives more aggressively.",
              "Moderate — affects confidence calibration.")
        form.addRow("No-behavior sample weight:", no_behavior_weight)

        quick_ident_minutes = QSpinBox(dlg)
        quick_ident_minutes.setRange(1, 120)
        quick_ident_minutes.setSuffix(" min/session")
        quick_ident_minutes.setValue(int(self._quick_ident_minutes.value()))
        _info(quick_ident_minutes,
              "Only active when quick test mode is ON. Higher = more data per session.",
              "Moderate — directly proportional to preprocessing time.")
        form.addRow("Quick sample duration:", quick_ident_minutes)

        quick_ident_seed = QSpinBox(dlg)
        quick_ident_seed.setRange(0, 999999)
        quick_ident_seed.setValue(int(self._quick_ident_seed.value()))
        form.addRow("Quick random seed:", quick_ident_seed)

        # ── Balanced Time Sampling ──
        form.addRow(_section("Balanced Time Sampling"))

        balanced_sampling = QCheckBox("Enable balanced time sampling", dlg)
        balanced_sampling.setChecked(bool(self._balanced_sampling.isChecked()))
        _info(balanced_sampling,
              "Sample a random time window from each session without skipping pipeline stages. "
              "Good for reducing processing time while keeping full evaluation quality.",
              "Moderate — reduces preprocessing proportionally to sampling fraction.")
        form.addRow(balanced_sampling)

        balanced_minutes = QSpinBox(dlg)
        balanced_minutes.setRange(1, 120)
        balanced_minutes.setSuffix(" min/session")
        balanced_minutes.setValue(int(self._balanced_sampling_minutes.value()))
        _info(balanced_minutes,
              "Minutes of video to sample per session. Higher = more data, slower.",
              "Proportional to duration.")
        form.addRow("Sampling duration:", balanced_minutes)

        balanced_seed = QSpinBox(dlg)
        balanced_seed.setRange(0, 999999)
        balanced_seed.setValue(int(self._balanced_sampling_seed.value()))
        form.addRow("Sampling seed:", balanced_seed)

        # ── Calibration ──
        form.addRow(_section("Calibration"))

        temperature_scale = QDoubleSpinBox(dlg)
        temperature_scale.setRange(0.5, 5.0)
        temperature_scale.setSingleStep(0.1)
        temperature_scale.setDecimals(1)
        temperature_scale.setValue(float(self._temperature_scale.value()))
        _info(temperature_scale,
              "Temperature scaling softens overconfident model probabilities.\n"
              "T=1.0: raw probabilities (no change). T=1.5: moderate softening.\n"
              "T=2.0+: aggressive. Higher values push predictions toward 0.5.",
              "No performance cost. Applied after inference.")
        form.addRow("Temperature scale:", temperature_scale)

        # ── Pipeline Stages ──
        form.addRow(_section("Pipeline Stages"))

        skip_fusion = QCheckBox("Skip video fusion", dlg)
        skip_fusion.setChecked(bool(self._skip_fusion.isChecked()))
        _info(skip_fusion,
              "Fusion refines predictions using video-based re-scoring. Skip it for faster iteration "
              "during early rounds when model quality is low.",
              "Moderate — saves 2-10 min depending on dataset size.")
        form.addRow(skip_fusion)

        skip_evaluation = QCheckBox("Skip evaluation reports", dlg)
        skip_evaluation.setChecked(bool(self._skip_evaluation.isChecked()))
        _info(skip_evaluation,
              "Evaluation generates UMAP, confusion matrix, PR curves. Skip for pure speed. "
              "Enable when you need to assess model quality.",
              "Low-Moderate — adds 1-3 min for charts and metrics.")
        form.addRow(skip_evaluation)

        enable_umap = QCheckBox("Enable UMAP generation", dlg)
        enable_umap.setChecked(bool(self._enable_umap.isChecked()))
        _info(enable_umap,
              "Generate unified UMAP plots after pipeline run, pipeline all, retrain, and retrain all. "
              "Disable to skip UMAP for faster runs.",
              "Moderate — UMAP computation can take 1-5 min on large datasets.")
        form.addRow(enable_umap)

        reuse_cached_features = QCheckBox("Reuse cached features", dlg)
        reuse_cached_features.setChecked(bool(self._reuse_cached_features.isChecked()))
        _info(reuse_cached_features,
              "Always recommended unless you changed pose smoothing or feature settings.",
              "Major — skipping recompute saves 5-30 min.")
        form.addRow(reuse_cached_features)

        remap_reviewed_windows = QCheckBox("Remap reviewed windows", dlg)
        remap_reviewed_windows.setChecked(bool(self._remap_reviewed_windows.isChecked()))
        _info(remap_reviewed_windows,
              "Enable when you changed the segment window size between rounds. "
              "Maps old review labels onto the new window geometry.",
              "Negligible.")
        form.addRow(remap_reviewed_windows)

        auto_gen_windows = QCheckBox("Auto-generate features for all reviewed segments", dlg)
        auto_gen_windows.setChecked(bool(self._auto_generate_reviewed_windows.isChecked()))
        _info(auto_gen_windows,
              "Automatically computes features for reviewed segments (bout-based, random, prior windows) "
              "that are not in the current feature table so they can be included in training.",
              "Low-Moderate — adds a few seconds per session with missing segments.")
        form.addRow(auto_gen_windows)

        strict_gpu = QCheckBox("Require GPU (fail if fallback)", dlg)
        strict_gpu.setChecked(bool(self._strict_gpu.isChecked()))
        _info(strict_gpu,
              "Only enable if you need to guarantee GPU training (e.g., timing benchmarks).",
              "None — affects error handling only.")
        form.addRow(strict_gpu)

        # ── Optical Flow Speed ──
        form.addRow(_section("Optical Flow Speed"))

        flow_temporal_stride = QSpinBox(dlg)
        flow_temporal_stride.setRange(1, 10)
        flow_temporal_stride.setValue(int(self._flow_temporal_stride.value()))
        _info(flow_temporal_stride,
              "Compute optical flow every Nth frame and linearly interpolate between. "
              "Higher values dramatically reduce context feature computation time. "
              "1 = every frame (slowest). 3 = default (~3× faster). 5 = aggressive (~5× faster).",
              "Major — directly reduces GPU optical flow compute time proportionally.")
        form.addRow("Flow temporal stride:", flow_temporal_stride)

        # ── Multi-Behavior Ranking ──
        form.addRow(_section("Multi-Behavior Ranking"))

        all_behavior_aware = QCheckBox("All-behavior-aware ranking", dlg)
        all_behavior_aware.setChecked(bool(self._all_behavior_aware.isChecked()))
        _info(all_behavior_aware,
              "Enable when ≥2 behaviors are defined. Uses competing models to prioritize "
              "boundary cases between behaviors.",
              "Low — loads additional model files but does not retrain them.")
        form.addRow(all_behavior_aware)

        competition_margin = QDoubleSpinBox(dlg)
        competition_margin.setRange(0.0, 1.0)
        competition_margin.setSingleStep(0.01)
        competition_margin.setDecimals(2)
        competition_margin.setValue(float(self._all_behavior_competition_margin.value()))
        _info(competition_margin,
              "Minimum probability gap required for the target behavior to beat a competing behavior.\n"
              "< 1.0 (e.g. 0.05): candidates are included even when the target only marginally outscores competitors — "
              "good for catching subtle or ambiguous events.\n"
              "→ 1.0: only candidates where the target clearly dominates all competitors are kept — "
              "stricter filtering that reduces confound overlap but may miss boundary cases.",
              "Negligible — applied only during candidate ranking.")
        form.addRow("Competition margin:", competition_margin)

        # ── Phase 1 Benchmarks ──
        form.addRow(_section("Adaptive Benchmarks (Phase 1)"))

        phase1_enable = QCheckBox("Enable adaptive benchmarks", dlg)
        phase1_enable.setChecked(bool(self._phase1_enable.isChecked()))
        _info(phase1_enable,
              "Runs diagnostic tests comparing feature families (pose, motion, context) and "
              "temporal scales. Valuable for understanding which features drive your model. "
              "Most useful for publication or when model performance plateaus.",
              "High — adds 5-20 min. Run only when you need diagnostic insight.")
        form.addRow(phase1_enable)

        phase1_modality = QCheckBox("Benchmark feature families", dlg)
        phase1_modality.setChecked(bool(self._phase1_modality.isChecked()))
        _info(phase1_modality,
              "Compare pose, visual, motion, context, and fused expert accuracy.",
              "Moderate — trains multiple sub-models.")
        form.addRow(phase1_modality)

        phase1_multiscale = QCheckBox("Benchmark time scales", dlg)
        phase1_multiscale.setChecked(bool(self._phase1_multiscale.isChecked()))
        _info(phase1_multiscale,
              "Test which temporal window durations capture the behavior best.",
              "High — trains across multiple window sizes.")
        form.addRow(phase1_multiscale)

        phase1_confound = QCheckBox("Confound analysis", dlg)
        phase1_confound.setChecked(bool(self._phase1_confound.isChecked()))
        _info(phase1_confound,
              "Identifies the top non-target confounders. Requires existing review labels.",
              "Moderate — statistical analysis on existing predictions.")
        form.addRow(phase1_confound)

        phase1_diagnostics = QCheckBox("Generate diagnostics", dlg)
        phase1_diagnostics.setChecked(bool(self._phase1_diagnostics.isChecked()))
        form.addRow(phase1_diagnostics)

        phase1_regenerate = QCheckBox("Regenerate cached", dlg)
        phase1_regenerate.setChecked(bool(self._phase1_regenerate.isChecked()))
        form.addRow(phase1_regenerate)

        phase1_export_hires = QCheckBox("Export publication quality (PNG+SVG)", dlg)
        phase1_export_hires.setChecked(bool(self._phase1_export_hires.isChecked()))
        form.addRow(phase1_export_hires)

        phase1_scales = QLineEdit(dlg)
        phase1_scales.setText(str(self._phase1_scales.text() or ""))
        form.addRow("Scale set (sec):", phase1_scales)

        phase1_run_btn = QPushButton("Run Phase 1 Benchmarks Now", dlg)
        phase1_run_btn.clicked.connect(lambda: (dlg.accept(), self._run_phase1_benchmarks()))
        form.addRow("", phase1_run_btn)

        # ── Weighted Queue (Phase 2) ──
        form.addRow(_section("Weighted Queue Composition (Phase 2)"))

        candidate_focus_pct = QSpinBox(dlg)
        candidate_focus_pct.setRange(0, 100)
        candidate_focus_pct.setSingleStep(10)
        candidate_focus_pct.setSuffix(" % edge cases")
        candidate_focus_pct.setValue(int(self._candidate_focus_pct.value()))
        _info(candidate_focus_pct,
              "High-level control: 0% = mostly strong candidates, 50% = balanced, "
              "100% = mostly edge cases (hard negatives, confound boundaries, disagreements).",
              "None — only affects candidate composition.")
        form.addRow("Candidate focus:", candidate_focus_pct)

        queue_weighted_enable = QCheckBox("Enable weighted queue", dlg)
        queue_weighted_enable.setChecked(bool(self._queue_weighted_enable.isChecked()))
        _info(queue_weighted_enable,
              "Combines multiple ranking signals (disagreement, diversity, hard negatives) "
              "into a composite candidate score. Most useful after 50+ reviewed labels when "
              "pure uncertainty sampling starts to plateau.",
              "Low — affects candidate scoring, not training.")
        form.addRow(queue_weighted_enable)

        queue_enable_disagreement = QCheckBox("Disagreement", dlg)
        queue_enable_disagreement.setChecked(bool(self._queue_enable_disagreement.isChecked()))
        _info(queue_enable_disagreement,
              "Surfaces examples where multiple expert models disagree.",
              "Negligible.")
        form.addRow(queue_enable_disagreement)

        queue_enable_diversity = QCheckBox("Diversity", dlg)
        queue_enable_diversity.setChecked(bool(self._queue_enable_diversity.isChecked()))
        _info(queue_enable_diversity,
              "Ensures review queue covers the representation space broadly.",
              "Low.")
        form.addRow(queue_enable_diversity)

        queue_enable_confound = QCheckBox("Confound-boundary", dlg)
        queue_enable_confound.setChecked(bool(self._queue_enable_confound.isChecked()))
        _info(queue_enable_confound,
              "Targets decision boundaries near competing behaviors.",
              "Low.")
        form.addRow(queue_enable_confound)

        queue_enable_hardneg = QCheckBox("Hard-negative mining", dlg)
        queue_enable_hardneg.setChecked(bool(self._queue_enable_hardneg.isChecked()))
        _info(queue_enable_hardneg,
              "Surfaces the most confusing negative examples for targeted labeling.",
              "Low.")
        form.addRow(queue_enable_hardneg)

        queue_diversity_mode = QComboBox(dlg)
        for i in range(self._queue_diversity_mode.count()):
            queue_diversity_mode.addItem(self._queue_diversity_mode.itemText(i), self._queue_diversity_mode.itemData(i))
        queue_diversity_mode.setCurrentIndex(max(0, queue_diversity_mode.findData(self._queue_diversity_mode.currentData())))
        form.addRow("Diversity mode:", queue_diversity_mode)

        queue_exploration_fraction = QDoubleSpinBox(dlg)
        queue_exploration_fraction.setRange(0.0, 0.5)
        queue_exploration_fraction.setSingleStep(0.01)
        queue_exploration_fraction.setDecimals(2)
        queue_exploration_fraction.setValue(float(self._queue_exploration_fraction.value()))
        form.addRow("Exploration fraction:", queue_exploration_fraction)

        # ── Tools ──
        form.addRow(_section("Tools"))

        configure_features_btn = QPushButton("Configure Features…", dlg)
        configure_features_btn.setToolTip("Select which feature columns the model trains on.")
        configure_features_btn.clicked.connect(lambda: (dlg.accept(), self._show_feature_config_dialog()))
        form.addRow("", configure_features_btn)

        edge_case_btn = QPushButton("Find Edge Cases…", dlg)
        edge_case_btn.setToolTip("Find segments where two behaviors compete closely.")
        edge_case_btn.clicked.connect(lambda: (dlg.accept(), self._open_edge_case_finder()))
        form.addRow("", edge_case_btn)

        # ── Scroll + buttons ──
        scroll = QScrollArea(dlg)
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_host)

        apply_btn = QPushButton("Apply", dlg)
        apply_btn.setMinimumHeight(32)
        apply_btn.setStyleSheet(
            "QPushButton { background: #1565C0; color: #FFFFFF; border-radius: 4px; "
            "padding: 6px 24px; font-weight: 700; }"
            "QPushButton:hover { background: #1976D2; }"
        )
        close_btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dlg)
        close_btns.rejected.connect(dlg.reject)

        button_row = QHBoxLayout()
        button_row.addWidget(apply_btn)
        button_row.addStretch(1)

        def _apply() -> None:
            self._mode.setCurrentIndex(max(0, self._mode.findData(mode.currentData())))
            self._query_size.setValue(int(query_size.value()))
            self._quick_test.setChecked(bool(quick_test.isChecked()))
            self._quick_profile.setCurrentIndex(max(0, self._quick_profile.findData(quick_profile.currentData())))
            self._examples_per_session.setValue(int(examples_per_session.value()))
            self._max_segments.setValue(int(max_segments.value()))
            self._max_train_samples_per_class.setValue(int(max_train_samples_per_class.value()))
            self._no_behavior_sample_weight.setValue(float(no_behavior_weight.value()))
            self._quick_ident_minutes.setValue(int(quick_ident_minutes.value()))
            self._quick_ident_seed.setValue(int(quick_ident_seed.value()))
            self._balanced_sampling.setChecked(bool(balanced_sampling.isChecked()))
            self._balanced_sampling_minutes.setValue(int(balanced_minutes.value()))
            self._balanced_sampling_seed.setValue(int(balanced_seed.value()))
            self._temperature_scale.setValue(float(temperature_scale.value()))
            self._validation_pct.setValue(int(validation_pct.value()))
            self._split_strategy.setCurrentIndex(max(0, self._split_strategy.findData(split_strategy.currentData())))
            self._reuse_cached_features.setChecked(bool(reuse_cached_features.isChecked()))
            self._remap_reviewed_windows.setChecked(bool(remap_reviewed_windows.isChecked()))
            self._auto_generate_reviewed_windows.setChecked(bool(auto_gen_windows.isChecked()))
            self._skip_fusion.setChecked(bool(skip_fusion.isChecked()))
            self._skip_evaluation.setChecked(bool(skip_evaluation.isChecked()))
            self._enable_umap.setChecked(bool(enable_umap.isChecked()))
            self._strict_gpu.setChecked(bool(strict_gpu.isChecked()))
            self._flow_temporal_stride.setValue(int(flow_temporal_stride.value()))
            self._all_behavior_aware.setChecked(bool(all_behavior_aware.isChecked()))
            self._all_behavior_competition_margin.setValue(float(competition_margin.value()))
            self._phase1_enable.setChecked(bool(phase1_enable.isChecked()))
            self._phase1_modality.setChecked(bool(phase1_modality.isChecked()))
            self._phase1_multiscale.setChecked(bool(phase1_multiscale.isChecked()))
            self._phase1_confound.setChecked(bool(phase1_confound.isChecked()))
            self._phase1_diagnostics.setChecked(bool(phase1_diagnostics.isChecked()))
            self._phase1_regenerate.setChecked(bool(phase1_regenerate.isChecked()))
            self._phase1_export_hires.setChecked(bool(phase1_export_hires.isChecked()))
            self._phase1_scales.setText(str(phase1_scales.text() or ""))
            self._candidate_focus_pct.setValue(int(candidate_focus_pct.value()))
            self._queue_weighted_enable.setChecked(bool(queue_weighted_enable.isChecked()))
            self._queue_enable_disagreement.setChecked(bool(queue_enable_disagreement.isChecked()))
            self._queue_enable_diversity.setChecked(bool(queue_enable_diversity.isChecked()))
            self._queue_enable_confound.setChecked(bool(queue_enable_confound.isChecked()))
            self._queue_enable_hardneg.setChecked(bool(queue_enable_hardneg.isChecked()))
            self._queue_diversity_mode.setCurrentIndex(
                max(0, self._queue_diversity_mode.findData(queue_diversity_mode.currentData()))
            )
            self._queue_exploration_fraction.setValue(float(queue_exploration_fraction.value()))
            self._persist_ui_settings_to_project()
            self._refresh_quick_mode_summary()
            self._status.setText("Saved active-learning settings.")

        apply_btn.clicked.connect(_apply)

        layout = QVBoxLayout(dlg)
        layout.addWidget(scroll, 1)
        layout.addLayout(button_row)
        layout.addWidget(close_btns)
        dlg.exec()

    @staticmethod
    def _model_version_for_behavior(behavior_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in behavior_id.strip())
        safe = safe or "target_behavior"
        return f"behavior_model_{safe}_v1"

    @staticmethod
    def _safe_model_name(name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name).strip())
        return safe.strip("_")

    def _resolved_model_version(self, target_behavior: str) -> str:
        # Pipeline-All override: use per-behavior name so each behavior
        # gets its own model directory instead of sharing a single one.
        pipeline_all_name = getattr(self, "_pipeline_all_model_name_override", None)
        if pipeline_all_name:
            return f"behavior_model_{pipeline_all_name}"
        custom = self._safe_model_name(self._model_name.text())
        if custom:
            return f"behavior_model_{custom}"
        base = self._model_version_for_behavior(target_behavior)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{stamp}"

    @staticmethod
    def _display_model_name(model_version: str) -> str:
        mv = str(model_version or "").strip()
        if not mv:
            return "Unknown model"
        name = mv
        if name.startswith("behavior_model_"):
            name = name[len("behavior_model_") :]
        # Trim timestamp suffix for readability (e.g., _20260316_204512)
        name = re.sub(r"_\d{8}_\d{6}$", "", name)
        return name or mv

    def _refresh_saved_model_options(self) -> None:
        current = str(self._saved_model_combo.currentData() or "")
        self._saved_model_combo.blockSignals(True)
        self._saved_model_combo.clear()
        if self._project_root is not None:
            models_root = self._project_root / "derived" / "models"
            if models_root.exists():
                rows: list[tuple[str, str]] = []
                for p in models_root.iterdir():
                    if p.is_dir() and (p / "model_state.pkl").exists():
                        ts = datetime.utcfromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        rows.append((ts, p.name))
                rows.sort(reverse=True)
                for ts, name in rows:
                    disp = self._display_model_name(name)
                    self._saved_model_combo.addItem(f"{disp} | {ts}", userData=name)
        idx = self._saved_model_combo.findData(current)
        if idx >= 0:
            self._saved_model_combo.setCurrentIndex(idx)
        self._saved_model_combo.blockSignals(False)
        self._refresh_viz_model_options()

    def _refresh_viz_model_options(self) -> None:
        current = str(self._viz_model_selector.currentData() or "__latest__")
        self._viz_model_selector.blockSignals(True)
        self._viz_model_selector.clear()
        self._viz_model_selector.addItem("Latest run", userData="__latest__")
        if self._project_root is not None:
            models_root = self._project_root / "derived" / "models"
            model_rows: list[tuple[float, str]] = []
            if models_root.exists():
                for p in models_root.iterdir():
                    if p.is_dir() and (p / "model_state.pkl").exists():
                        model_rows.append((float(p.stat().st_mtime), p.name))
            model_rows.sort(reverse=True)
            for _mtime, version in model_rows:
                disp = self._display_model_name(version)
                self._viz_model_selector.addItem(disp, userData=version)
        idx = self._viz_model_selector.findData(current)
        self._viz_model_selector.setCurrentIndex(idx if idx >= 0 else 0)
        self._viz_model_selector.blockSignals(False)

    @staticmethod
    def _write_pipeline_timing_chart(
        step_timings: list[tuple[str, float]],
        eval_dir: Path,
        model_version: str = "",
    ) -> Path | None:
        """Generate a horizontal bar chart of pipeline step durations and save it as a PNG."""
        if not step_timings:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from abel.storage.file_store import write_json
        except Exception:
            return None

        # Preserve insertion order; deduplicate by keeping last entry for each label.
        seen: dict[str, float] = {}
        for label, seconds in step_timings:
            seen[label] = float(seconds)

        # Place longest items at top for a natural reading order.
        ordered = sorted(seen.items(), key=lambda x: x[1])
        labels = [item[0] for item in ordered]
        durations = [item[1] for item in ordered]

        def _fmt(s: float) -> str:
            if s >= 60:
                return f"{int(s // 60)}m {int(s % 60):02d}s"
            return f"{s:.1f}s"

        fig, ax = plt.subplots(figsize=(8, max(3.2, 0.55 * len(labels) + 1.4)))
        bar_colors = ["#2563eb" if d == max(durations) else "#93c5fd" for d in durations]
        bars = ax.barh(labels, durations, color=bar_colors, edgecolor="white", linewidth=0.5)
        # Annotate bars with human-readable duration
        for bar, seconds in zip(bars, durations):
            ax.text(
                bar.get_width() + max(durations) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                _fmt(seconds),
                va="center",
                ha="left",
                fontsize=9,
                color="#1e3a5f",
            )
        total = sum(durations)
        title = "Pipeline Step Timings"
        if model_version:
            short_ver = model_version[-20:] if len(model_version) > 20 else model_version
            title += f"  [{short_ver}]"
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel(f"Duration (seconds) — total {_fmt(total)}", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(0, max(durations) * 1.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        eval_dir.mkdir(parents=True, exist_ok=True)
        png_path = eval_dir / "pipeline_timing.png"
        try:
            fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)

        # Also save raw JSON for reproducibility
        try:
            write_json(
                eval_dir / "pipeline_timing.json",
                {"model_version": model_version, "steps": [{"label": l, "seconds": s} for l, s in seen.items()]},
            )
        except Exception:
            pass

        return png_path

    def _snapshot_evaluation_graphs_for_model(self, model_version: str, target_behavior: str | None = None) -> None:
        if self._project_root is None:
            return
        model = str(model_version or "").strip()
        if not model:
            return
        src_root = self._project_root / "derived" / "evaluation"
        dst_root = src_root / "by_model" / model
        dst_root.mkdir(parents=True, exist_ok=True)

        core_files = [
            "unified_behavior_umap.png",
            "confusion_matrix.png",
            "PR_curve.png",
            "pipeline_timing.png",
            "cross_behavior_confound_matrix.png",
        ]
        for name in core_files:
            src = src_root / name
            if src.exists():
                try:
                    dst = dst_root / name
                    dst.write_bytes(src.read_bytes())
                except Exception:
                    continue

        behavior_id = str(target_behavior or self._selected_target_behavior_id() or "").strip()
        safe_behavior = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in behavior_id) or "target_behavior"
        diag_root = self._project_root / "derived" / "analysis" / "diagnostics" / safe_behavior
        latest_diag = read_json(diag_root / "latest.json", {}) if (diag_root / "latest.json").exists() else {}
        latest_diag_dir = Path(str(latest_diag.get("diagnostic_dir", ""))) if latest_diag.get("diagnostic_dir") else None
        diag_src_root = latest_diag_dir if (latest_diag_dir is not None and latest_diag_dir.exists()) else diag_root
        for name in [
            "feature_family_comparison.png",
            "multiscale_performance.png",
            "target_confound_margin_histogram.png",
            "calibration_reliability_phase1.png",
        ]:
            src = diag_src_root / name
            if src.exists():
                try:
                    (dst_root / name).write_bytes(src.read_bytes())
                except Exception:
                    continue

        queue_diag_root = self._project_root / "derived" / "analysis" / "diagnostics" / "queue"
        latest_queue = read_json(queue_diag_root / "latest.json", {}) if (queue_diag_root / "latest.json").exists() else {}
        queue_run_path = Path(str(latest_queue.get("queue_composition", ""))) if latest_queue.get("queue_composition") else None
        queue_src = (queue_run_path.parent / "queue_composition.png") if queue_run_path is not None else (queue_diag_root / "queue_composition.png")
        if queue_src.exists():
            try:
                (dst_root / "queue_composition.png").write_bytes(queue_src.read_bytes())
            except Exception:
                pass

    def _load_selected_saved_model(self) -> None:
        chosen = str(self._saved_model_combo.currentData() or "").strip()
        if not chosen:
            QMessageBox.information(self, "Active Learning", "No saved model selected.")
            return
        prefix = "behavior_model_"
        custom = chosen[len(prefix):] if chosen.startswith(prefix) else chosen
        self._model_name.setText(custom)
        self._append_log(f"Loaded saved model: {self._display_model_name(chosen)}")
        self._status.setText(f"Loaded saved model: {self._display_model_name(chosen)}")

    def _show_selected_saved_model_settings(self) -> None:
        if self._project_root is None:
            QMessageBox.warning(self, "Active Learning", "Open a project first.")
            return
        chosen = str(self._saved_model_combo.currentData() or "").strip()
        if not chosen:
            QMessageBox.information(self, "Active Learning", "No saved model selected.")
            return

        model_dir = self._project_root / "derived" / "models" / chosen
        if not model_dir.exists():
            QMessageBox.warning(self, "Active Learning", f"Model folder not found: {chosen}")
            return

        run_settings = read_json(model_dir / "run_settings.json", {})
        model_card = read_yaml(model_dir / "model_card.yaml", {}) if (model_dir / "model_card.yaml").exists() else {}
        metrics = read_json(model_dir / "metrics.json", {}) if (model_dir / "metrics.json").exists() else {}

        lines: list[str] = []
        lines.append(f"Model: {chosen}")
        lines.append("")

        if run_settings:
            lines.append("Run settings (saved at model creation):")
            lines.append(f"- target_behavior: {run_settings.get('target_behavior', '')}")
            lines.append(f"- segment_window_frames: {run_settings.get('segment_window_frames', '')}")
            lines.append(f"- segment_stride_frames: {run_settings.get('segment_stride_frames', '')}")
            lines.append(f"- quick_test: {run_settings.get('quick_test', '')}")
            lines.append(f"- quick_ident_minutes: {run_settings.get('quick_ident_minutes', '')}")
            lines.append(f"- quick_ident_seed: {run_settings.get('quick_ident_seed', '')}")
            lines.append(f"- mode: {run_settings.get('mode', '')}")
            lines.append(f"- query_size: {run_settings.get('query_size', '')}")
            lines.append(f"- split_strategy: {run_settings.get('split_strategy', '')}")
            lines.append(f"- validation_pct: {run_settings.get('validation_pct', '')}")
            lines.append(f"- skip_fusion: {run_settings.get('skip_fusion', '')}")
            lines.append(f"- skip_evaluation: {run_settings.get('skip_evaluation', '')}")
            lines.append("")
        else:
            lines.append("Run settings were not saved for this older model.")
            lines.append("")

        training_cfg = ((model_card.get("provenance") or {}).get("config") or {}).get("training_config") or {}
        if training_cfg:
            lines.append("Training config from model card:")
            lines.append(f"- classifier_family: {training_cfg.get('classifier_family', '')}")
            lines.append(f"- calibration_method: {training_cfg.get('calibration_method', '')}")
            lines.append(f"- split_strategy: {training_cfg.get('split_strategy', '')}")
            lines.append(f"- test_size: {training_cfg.get('test_size', '')}")
            lines.append(f"- target_label: {training_cfg.get('target_label', '')}")
            lines.append("")

        if metrics:
            lines.append("Validation metrics:")
            lines.append(f"- f1: {metrics.get('f1', '')}")
            lines.append(f"- pr_auc: {metrics.get('pr_auc', '')}")
            lines.append(f"- n_train: {metrics.get('n_train', '')}")
            lines.append(f"- n_val: {metrics.get('n_val', '')}")
            lines.append(f"- model_device_used: {metrics.get('model_device_used', '')}")

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Model Settings - {self._display_model_name(chosen)}")
        dlg.resize(760, 560)
        layout = QVBoxLayout(dlg)
        text = QTextEdit(dlg)
        text.setReadOnly(True)
        text.setPlainText("\n".join(lines))
        layout.addWidget(text)

        action_row = QHBoxLayout()
        rehydrate_btn = QPushButton("Rehydrate Settings To UI", dlg)
        if not run_settings and not training_cfg:
            rehydrate_btn.setEnabled(False)
            rehydrate_btn.setToolTip("No saved settings are available for this model.")

        def _rehydrate() -> None:
            ok = self._rehydrate_ui_from_model_settings(
                model_version=chosen,
                run_settings=run_settings,
                training_cfg=training_cfg,
            )
            if ok:
                QMessageBox.information(
                    dlg,
                    "Rehydration Complete",
                    "Applied saved model settings to the current Active Learning UI.",
                )
                dlg.accept()
            else:
                QMessageBox.warning(
                    dlg,
                    "Rehydration Unavailable",
                    "No compatible settings were found for this model.",
                )

        rehydrate_btn.clicked.connect(_rehydrate)
        action_row.addWidget(rehydrate_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        close_btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dlg)
        close_btns.rejected.connect(dlg.reject)
        layout.addWidget(close_btns)
        dlg.exec()

    def _rehydrate_ui_from_model_settings(
        self,
        *,
        model_version: str,
        run_settings: dict[str, Any],
        training_cfg: dict[str, Any],
    ) -> bool:
        ui = dict(run_settings.get("ui_settings") or {})

        if not ui:
            # Fallback for older models that predate run_settings.ui_settings.
            if run_settings or training_cfg:
                ui = {
                    "mode": str(run_settings.get("mode") or "uncertainty"),
                    "target_behavior_id": str(run_settings.get("target_behavior") or training_cfg.get("target_label") or ""),
                    "query_size": int(run_settings.get("query_size", self._query_size.value())),
                    "quick_test": bool(run_settings.get("quick_test", False)),
                    "examples_per_session": int(run_settings.get("examples_per_session", self._examples_per_session.value())),
                    "max_segments": int(run_settings.get("max_segments", self._max_segments.value())),
                    "quick_ident_minutes": int(run_settings.get("quick_ident_minutes", self._quick_ident_minutes.value())),
                    "quick_ident_seed": int(run_settings.get("quick_ident_seed", self._quick_ident_seed.value())),
                    "skip_fusion": bool(run_settings.get("skip_fusion", self._skip_fusion.isChecked())),
                    "skip_evaluation": bool(run_settings.get("skip_evaluation", self._skip_evaluation.isChecked())),
                    "validation_pct": int(
                        run_settings.get(
                            "validation_pct",
                            int(round(float(training_cfg.get("test_size", 0.25)) * 100.0)),
                        )
                    ),
                    "split_strategy": str(
                        run_settings.get(
                            "split_strategy",
                            training_cfg.get("split_strategy", "group_shuffle_session"),
                        )
                    ),
                    "quick_profile": str(run_settings.get("quick_profile") or "custom"),
                }

        if not ui:
            return False

        self._loading_ui_settings = True
        try:
            mode = str(ui.get("mode") or "").strip()
            idx = self._mode.findData(mode)
            if idx >= 0:
                self._mode.setCurrentIndex(idx)

            target = str(ui.get("target_behavior_id") or "").strip()
            idx_target = self._target_behavior.findData(target)
            if idx_target >= 0:
                self._target_behavior.setCurrentIndex(idx_target)

            name = str(ui.get("model_name") or "").strip()
            if not name:
                prefix = "behavior_model_"
                name = model_version[len(prefix) :] if str(model_version).startswith(prefix) else str(model_version)
            self._model_name.setText(name)

            idx_saved = self._saved_model_combo.findData(str(model_version))
            if idx_saved >= 0:
                self._saved_model_combo.setCurrentIndex(idx_saved)

            self._query_size.setValue(int(ui.get("query_size", self._query_size.value())))
            self._quick_test.setChecked(bool(ui.get("quick_test", self._quick_test.isChecked())))
            self._examples_per_session.setValue(int(ui.get("examples_per_session", self._examples_per_session.value())))
            self._max_segments.setValue(int(ui.get("max_segments", self._max_segments.value())))
            self._quick_ident_minutes.setValue(int(ui.get("quick_ident_minutes", self._quick_ident_minutes.value())))
            self._quick_ident_seed.setValue(int(ui.get("quick_ident_seed", self._quick_ident_seed.value())))
            self._skip_fusion.setChecked(bool(ui.get("skip_fusion", self._skip_fusion.isChecked())))
            self._skip_evaluation.setChecked(bool(ui.get("skip_evaluation", self._skip_evaluation.isChecked())))
            self._validation_pct.setValue(int(ui.get("validation_pct", self._validation_pct.value())))

            split = str(ui.get("split_strategy") or "").strip()
            idx_split = self._split_strategy.findData(split)
            if idx_split >= 0:
                self._split_strategy.setCurrentIndex(idx_split)

            profile = str(ui.get("quick_profile") or "custom")
            idx_profile = self._quick_profile.findData(profile)
            if idx_profile >= 0:
                self._quick_profile.setCurrentIndex(idx_profile)
        finally:
            self._loading_ui_settings = False

        self._refresh_quick_mode_summary()
        self._persist_ui_settings_to_project()
        self._status.setText(f"Rehydrated settings from model: {self._display_model_name(model_version)}")
        self._append_log(f"Rehydrated UI settings from model {self._display_model_name(model_version)}.")
        return True

    def _save_model_run_settings(
        self,
        *,
        model_version: str,
        target_behavior: str,
        segment_window: int,
        segment_stride: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._project_root is None:
            return
        model_dir = self._project_root / "derived" / "models" / str(model_version)
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "model_version": str(model_version),
            "target_behavior": str(target_behavior),
            "segment_window_frames": int(segment_window),
            "segment_stride_frames": int(segment_stride),
            "mode": str(self._mode.currentData() or "uncertainty"),
            "query_size": int(self._query_size.value()),
            "split_strategy": str(self._split_strategy.currentData() or "group_shuffle_session"),
            "validation_pct": int(self._validation_pct.value()),
            "quick_test": bool(self._quick_test.isChecked()),
            "quick_ident_minutes": int(self._quick_ident_minutes.value()),
            "quick_ident_seed": int(self._quick_ident_seed.value()),
            "skip_fusion": bool(self._skip_fusion.isChecked()),
            "skip_evaluation": bool(self._skip_evaluation.isChecked()),
            "quick_profile": str(self._quick_profile.currentData() or "custom"),
            "ui_settings": self._ui_settings_payload(),
            "behavior_model": self._load_behavior_cfg().model_dump(mode="json"),
        }
        if extra:
            payload.update(extra)
        write_json(model_dir / "run_settings.json", payload)

    def _resolved_segment_settings(
        self,
        behavior_cfg: BehaviorModelConfig | None = None,
    ) -> tuple[int, int, str]:
        cfg = behavior_cfg or self._load_behavior_cfg()
        fallback_window = max(8, int(cfg.segment_window_frames))
        fallback_stride = max(1, int(cfg.segment_stride_frames))
        if self._project_root is None:
            return fallback_window, fallback_stride, "Project config"

        manifest_path = self._project_root / "derived" / "representations" / "representations.manifest.json"
        manifest = read_json(manifest_path, {}) if manifest_path.exists() else {}
        rep_cfg = (
            (manifest.get("provenance") or {})
            .get("config", {})
            .get("representation_config", {})
        )

        try:
            window = int(rep_cfg.get("window_size_frames"))
            stride = int(rep_cfg.get("window_stride_frames"))
            if window >= 8 and stride >= 1:
                return window, stride, "Representations manifest"
        except (TypeError, ValueError):
            pass

        return fallback_window, fallback_stride, "Project config"

    def _refresh_segment_settings_display(self) -> None:
        window, stride, source = self._resolved_segment_settings()
        self._segment_window_value.setText(str(window))
        self._segment_stride_value.setText(str(stride))
        self._segment_settings_hint.setText(f"Auto-synced source: {source}.")

    def _run_pipeline(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return

        target_behavior = self._selected_target_behavior_id()
        label_stats = self._collect_review_balance_stats(target_behavior)
        total_labels = int(label_stats.get("positive", 0)) + int(label_stats.get("negative", 0)) + int(
            label_stats.get("ambiguous", 0)
        ) + int(label_stats.get("other", 0))
        current_mode = str(self._mode.currentData() or "uncertainty")
        if total_labels <= 0 and current_mode != "random_absent":
            answer = QMessageBox.question(
                self,
                "No Seed Examples Found",
                "No seed examples provided. Generate random windows?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            idx_mode = self._mode.findData("random_absent")
            if idx_mode >= 0:
                self._mode.setCurrentIndex(idx_mode)
            self._append_log("No labels found; switched to random window bootstrap mode.")

        self._persist_ui_settings_to_project()
        self._refresh_segment_settings_display()
        self._set_busy(True)
        self._status.setText("Starting active-learning pipeline (detecting backend…)")
        self._pipeline_step_scale = 100
        self._progress.setRange(0, 0)  # indeterminate until worker reports
        self._progress.setFormat("Initializing…")
        self._cancel_flag[0] = False
        self._append_log("Starting full active-learning pipeline.")

        worker = TaskWorker(self._run_pipeline_task, self._pipeline_progress_updated.emit, self._cancel_flag)
        worker.signals.finished.connect(self._on_pipeline_finished)
        worker.signals.failed.connect(self._on_failed)
        self._pool.start(worker)

    def _run_retrain(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return
        self._persist_ui_settings_to_project()
        self._set_busy(True)
        self._cancel_flag[0] = False
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Initializing…")
        self._status.setText("Starting retrain from review labels…")
        self._append_log("Starting retrain from new review labels.")
        worker = TaskWorker(self._run_retrain_task, self._pipeline_progress_updated.emit)
        worker.signals.finished.connect(self._on_retrain_finished)
        worker.signals.failed.connect(self._on_failed)
        self._pool.start(worker)

    def _on_gen_clips_toggled(self, checked: bool) -> None:
        """Log the review-clip generation toggle state for batch runs."""
        self._append_log(
            "Review clip generation for batch runs: "
            + ("ENABLED (clips accumulate in the Clips tab)." if checked
               else "DISABLED (batch runs skip clip generation).")
        )

    def _batch_generate_clips_enabled(self) -> bool:
        """Read the 'Generate Review Clips' toggle (defaults on)."""
        return bool(getattr(self, "_gen_clips_btn", None) is None or self._gen_clips_btn.isChecked())

    # ------------------------------------------------------------------
    # Retrain All Behaviors (sequential)
    # ------------------------------------------------------------------

    def _run_retrain_all(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return
        behaviors = list(self._behaviors.behaviors)
        if not behaviors:
            QMessageBox.warning(self, "No behaviors", "No behaviors are defined in this project.")
            return
        # All active behaviors are trainable, including no_behavior which
        # trains a binary model using positives from other behaviors as
        # its negative class.
        trainable = [b for b in behaviors if b.is_active]
        if not trainable:
            QMessageBox.warning(self, "No trainable behaviors", "No active behaviors to retrain.")
            return
        # ----- behaviour-selection dialog -----
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Behaviors to Retrain")
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.addWidget(QLabel(
            f"{len(trainable)} trainable behavior(s) found.  "
            "Uncheck any you want to skip:"
        ))
        checks: list[tuple[QCheckBox, Any]] = []
        for b in trainable:
            cb = QCheckBox(b.name)
            cb.setChecked(True)
            dlg_layout.addWidget(cb)
            checks.append((cb, b))
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        dlg_layout.addWidget(btn_box)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = [b for cb, b in checks if cb.isChecked()]
        if not selected:
            QMessageBox.information(self, "Nothing selected", "No behaviors were selected.")
            return
        self._persist_ui_settings_to_project()
        self._set_busy(True)
        self._cancel_flag[0] = False
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Initializing…")
        self._status.setText("Starting retrain-all-behaviors…")
        self._retrain_all_selected = selected
        # Capture the clip-generation toggle on the GUI thread for the worker.
        self._batch_generate_clips = self._batch_generate_clips_enabled()
        self._append_log(f"Starting retrain-all for {len(selected)} behavior(s).")
        worker = TaskWorker(self._run_retrain_all_task, self._pipeline_progress_updated.emit)
        worker.signals.finished.connect(self._on_retrain_all_finished)
        worker.signals.failed.connect(self._on_failed)
        self._pool.start(worker)

    def _run_retrain_all_task(
        self,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, Any]:
        """Retrain every defined behavior in sequence, reusing _run_retrain_task."""
        results: list[dict[str, Any]] = []

        trainable = getattr(self, "_retrain_all_selected", None)
        if trainable is None:
            # Fallback: use all active behaviors (including no_behavior)
            trainable = [b for b in self._behaviors.behaviors if b.is_active]
        n_total = len(trainable)

        if n_total == 0:
            if progress_cb:
                progress_cb(1, 1, "No active behaviors found.", "Nothing to retrain.")
            return {"retrain_all": True, "results": [], "succeeded": 0, "total": 0}

        # Global timer and step scaling so ETA accounts for all behaviors.
        _all_started_at = time.monotonic()
        _run_evaluation = not bool(self._skip_evaluation.isChecked())
        _steps_per_behavior = 6 if _run_evaluation else 5
        _global_max = n_total * _steps_per_behavior
        # Stages within a behavior have very unequal wall-clock cost, so a plain
        # "fraction of stages done" ETA oscillates.  Learn each stage's typical
        # duration and sum the remaining stages' expected times instead.
        _eta = StageEtaEstimator(n_total, _steps_per_behavior)

        def _fmt_dur(seconds: float) -> str:
            if seconds < 1.0:
                return f"{seconds * 1000.0:.0f} ms"
            if seconds < 60.0:
                return f"{seconds:.1f} s"
            mins = int(seconds // 60)
            rem = int(seconds % 60)
            return f"{mins}m {rem:02d}s"

        def _outer_progress(value: int, maximum: int, log_line: str, status: str) -> None:
            if progress_cb is not None:
                # Map behavior-index space → global step space so the bar
                # advances continuously across all behaviors.
                progress_cb(value * _steps_per_behavior, _global_max, log_line, status)

        def _make_behavior_progress_cb(beh_idx: int) -> Callable[[int, int, str, str], None]:
            """Wrap progress_cb to report global ETA across all behaviors."""
            def _wrapped(value: int, maximum: int, log_line: str, status: str) -> None:
                if progress_cb is None:
                    return
                steps = max(1, maximum)
                global_value = beh_idx * steps + value
                g_max = n_total * steps
                elapsed = max(0.0, time.monotonic() - _all_started_at)
                # Per-stage-duration-weighted estimate (handles unequal stages).
                eta_seconds = _eta.update(beh_idx, value)
                eta_local = datetime.now() + timedelta(seconds=eta_seconds)
                # Strip per-behavior "| elapsed Xs" added by _run_retrain_task
                # so we can replace it with the global timing line.
                clean_log = log_line.split(" | elapsed ")[0] if " | elapsed " in log_line else log_line
                timing = (
                    f" | elapsed {_fmt_dur(elapsed)}"
                    f" | ETA {_fmt_dur(eta_seconds)}"
                    f" | finish ~ {eta_local.strftime('%H:%M:%S')}"
                )
                progress_cb(global_value, g_max, f"{clean_log}{timing}", status)
            return _wrapped

        # Ensure frame_features.parquet exists BEFORE any per-behavior
        # retrain runs.  The representation build (session_ids=None) writes
        # BOTH frame_features.parquet and segment_features.parquet.  If this
        # happens mid-loop (e.g. inside the first behavior's evaluation step)
        # it overwrites segment_features.parquet, causing later behaviors to
        # see more segments than the first behavior — a scope mismatch that
        # corrupts the unified UMAP via fillna(0.0).
        frame_path = self._project_root / "derived" / "representations" / "frame_features.parquet"
        if not frame_path.exists():
            _outer_progress(0, n_total, "Building full representation cache (one-time)…", "Preparing representations…")
            try:
                behavior_cfg = self._load_behavior_cfg()
                _pose = self._project_root / "derived" / "pose_features" / "frame_pose.parquet"
                _ctx = self._project_root / "derived" / "context_features" / "frame_context.parquet"
                if _pose.exists():
                    _use_vid = bool(behavior_cfg.use_video_features)
                    _ctx_arg = _ctx if (_use_vid and _ctx.exists()) else None
                    _seg_w, _seg_s, _ = self._resolved_segment_settings(behavior_cfg)
                    self._repr.build(
                        project_root=self._project_root,
                        frame_pose_path=_pose,
                        frame_context_path=_ctx_arg,
                        config=RepresentationConfig(
                            window_size_frames=_seg_w,
                            window_stride_frames=_seg_s,
                            excluded_feature_cols=frozenset(self._excluded_feature_cols),
                        ),
                        session_ids=None,
                        progress_cb=lambda msg: logger.info(msg),
                    )
            except Exception as exc:
                logger.warning("Retrain-all: failed to pre-build frame features: %s", exc)

        # Also clean up stale no_behavior model directories that may
        # have been created by earlier buggy runs.
        for nb_name in ("No_Behavior", "no_behavior", "No_behaviour", "no_behaviour"):
            stale_nb = self._project_root / "derived" / "models" / f"behavior_model_{nb_name}"
            if stale_nb.exists() and stale_nb.is_dir():
                import shutil
                try:
                    shutil.rmtree(stale_nb)
                    logger.info("Retrain-all: removed stale no_behavior model directory %s", stale_nb)
                except Exception:
                    pass

        try:
            for idx, behavior in enumerate(trainable):
                bid = str(behavior.behavior_id).strip()
                bname = str(behavior.name)

                # Override model name so each behavior gets its own directory
                # instead of all sharing the global custom model name.
                self._pipeline_all_model_name_override = self._safe_model_name(bname)

                _outer_progress(
                    idx,
                    n_total,
                    f"[{idx + 1}/{n_total}] Starting retrain for '{bname}'…",
                    f"Retraining {idx + 1}/{n_total}: {bname}…",
                )

                # Delete the old model directory so stale artifacts from a
                # previous run cannot leak into the new model.
                old_model_dir = self._project_root / "derived" / "models" / f"behavior_model_{self._safe_model_name(bname)}"
                if old_model_dir.exists() and old_model_dir.is_dir():
                    import shutil
                    try:
                        shutil.rmtree(old_model_dir)
                        logger.info("Retrain-all: removed old model directory %s", old_model_dir)
                    except Exception as rm_exc:
                        logger.warning("Retrain-all: could not remove old model dir %s: %s", old_model_dir, rm_exc)

                try:
                    result = self._run_retrain_task(
                        progress_cb=_make_behavior_progress_cb(idx),
                        target_behavior_override=bid,
                        skip_candidates=not getattr(self, "_batch_generate_clips", True),
                    )
                except Exception as exc:
                    logger.error("Retrain-all: failed on behavior '%s': %s", bname, exc)
                    result = {"retrained": False, "target_behavior": bid, "error": str(exc)}
                results.append(result)
        finally:
            # Clean up the override so subsequent single-behavior runs
            # are not affected.
            self._pipeline_all_model_name_override = None

        succeeded = sum(1 for r in results if r.get("retrained"))
        _outer_progress(
            n_total,
            n_total,
            f"Retrain-all complete: {succeeded}/{n_total} behaviors retrained successfully.",
            "Retrain all behaviors complete.",
        )
        return {"retrain_all": True, "results": results, "succeeded": succeeded, "total": n_total}

    def _on_retrain_all_finished(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        self._refresh_saved_model_options()
        results = payload.get("results", [])
        succeeded = int(payload.get("succeeded", 0))
        total = int(payload.get("total", 0))

        # Persist snapshots for each successfully retrained behavior.
        for r in results:
            if r.get("retrained"):
                model_version = str(r.get("model_version", "")).strip()
                target_behavior = str(r.get("target_behavior", ""))
                if model_version:
                    self._save_workflow_snapshot(
                        model_version=model_version,
                        target_behavior=target_behavior,
                    )
                    self._snapshot_evaluation_graphs_for_model(model_version, target_behavior=target_behavior)

        self._refresh_viz_model_options()
        self._status.setText(
            f"Retrain-all complete: {succeeded}/{total} behaviors retrained successfully."
        )
        self._progress.setValue(self._progress.maximum())
        self._progress.setFormat("Complete")

        # Log per-behavior summary.
        for r in results:
            bid = str(r.get("target_behavior", "?"))
            if r.get("retrained"):
                metrics = r.get("metrics", {})
                self._append_log(
                    f"  {bid}: F1={float(metrics.get('f1', 0.0)):.3f}, "
                    f"PR-AUC={float(metrics.get('pr_auc', 0.0)):.3f}, "
                    f"candidates={int(r.get('n_candidates', 0))}"
                )
            elif r.get("error"):
                self._append_log(f"  {bid}: FAILED — {r['error']}")
            else:
                self._append_log(f"  {bid}: skipped (no usable labels)")

        self._append_log(f"Retrain-all summary: {succeeded}/{total} succeeded.")

        # Emit all candidates to clip extraction.
        all_candidates: list = []
        for r in results:
            if r.get("retrained"):
                all_candidates.extend(r.get("candidates", []))
        if all_candidates:
            self._emit_uncertainty_candidates_for_clip_extraction(
                all_candidates,
                source_label="Active Learning \u2014 Retrain All Behaviors",
                append=True,
            )

        self._refresh_visualization_preview()

        if succeeded > 0 and self._is_umap_enabled():
            self._regenerate_unified_umap_inline()
        elif succeeded > 0:
            self._append_log("UMAP generation disabled in settings — skipping.")

    # ------------------------------------------------------------------
    # Run Pipeline All Behaviors (sequential full pipeline)
    # ------------------------------------------------------------------

    def _run_pipeline_all_behaviors(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return
        behaviors = [
            b for b in self._behaviors.behaviors
            if b.is_active
        ]
        if not behaviors:
            QMessageBox.warning(self, "No behaviors", "No active behaviors are defined in this project.")
            return
        # ----- behaviour-selection dialog -----
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Behaviors for Pipeline")
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.addWidget(QLabel(
            f"{len(behaviors)} active behavior(s) found.  "
            "Uncheck any you want to skip:\n\n"
            "Features will be extracted once and reused. Each behavior will be "
            "trained, scored, and have candidates generated independently."
        ))
        checks: list[tuple[QCheckBox, Any]] = []
        for b in behaviors:
            cb = QCheckBox(b.name)
            cb.setChecked(True)
            dlg_layout.addWidget(cb)
            checks.append((cb, b))
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        dlg_layout.addWidget(btn_box)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = [b for cb, b in checks if cb.isChecked()]
        if not selected:
            QMessageBox.information(self, "Nothing selected", "No behaviors were selected.")
            return
        self._persist_ui_settings_to_project()
        self._refresh_segment_settings_display()
        self._set_busy(True)
        self._cancel_flag[0] = False
        self._pipeline_step_scale = 100
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Initializing\u2026")
        self._pipeline_all_selected = selected
        # Capture the clip-generation toggle on the GUI thread for the worker.
        self._batch_generate_clips = self._batch_generate_clips_enabled()
        self._status.setText(f"Starting pipeline-all for {len(selected)} behavior(s)\u2026")
        self._append_log(f"Starting full pipeline for {len(selected)} behavior(s).")
        worker = TaskWorker(
            self._run_pipeline_all_behaviors_task,
            self._pipeline_progress_updated.emit,
            self._cancel_flag,
        )
        worker.signals.finished.connect(self._on_pipeline_all_behaviors_finished)
        worker.signals.failed.connect(self._on_failed)
        self._pool.start(worker)

    def _build_or_reuse_representation(
        self,
        *,
        frame_pose_path: Path,
        frame_ctx_path: Path | None,
        segment_window: int,
        segment_stride: int,
        selected_session_ids: set[str],
        progress_cb: Callable[[str], None] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build frame/segment representations, reusing across a Pipeline-All run.

        During Pipeline-All every behavior is trained over the *same* sessions
        with the same window/stride/exclusions, so the frame and segment tables
        are identical for all of them.  Re-deriving (and re-reading the multi-GB
        segment parquet) on each behavior is pure waste, so the first behavior's
        result is cached in memory and reused by the rest.  Outside Pipeline-All
        (``_pipeline_all_active`` is False) this is a plain pass-through to
        ``BehaviorRepresentationService.build``.
        """
        sig = (
            tuple(sorted(selected_session_ids)),
            int(segment_window),
            int(segment_stride),
            tuple(sorted(self._excluded_feature_cols)),
            str(frame_pose_path),
            str(frame_ctx_path),
        )

        def _build() -> tuple[pd.DataFrame, pd.DataFrame]:
            return self._repr.build(
                project_root=self._project_root,
                frame_pose_path=frame_pose_path,
                frame_context_path=frame_ctx_path,
                config=RepresentationConfig(
                    window_size_frames=segment_window,
                    window_stride_frames=segment_stride,
                    excluded_feature_cols=frozenset(self._excluded_feature_cols),
                ),
                session_ids=selected_session_ids,
                progress_cb=progress_cb,
            )

        def _on_reuse() -> None:
            if progress_cb is not None:
                progress_cb(
                    "Representation: reusing in-memory frame/segment tables from this Pipeline-All run."
                )

        frame_df, segment_df, new_cache = reuse_or_build_representation(
            active=bool(getattr(self, "_pipeline_all_active", False)),
            cache=getattr(self, "_pipeline_all_repr_cache", None),
            signature=sig,
            build_fn=_build,
            on_reuse=_on_reuse,
        )
        self._pipeline_all_repr_cache = new_cache
        return frame_df, segment_df

    def _run_pipeline_all_behaviors_task(
        self,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> dict[str, Any]:
        """Run _run_pipeline_task for every active behavior in sequence.

        The first run extracts features normally; subsequent runs hit the
        cached features path (reuse_cached_features is force-enabled).
        """
        behaviors = getattr(self, "_pipeline_all_selected", None)
        if behaviors is None:
            behaviors = [
                b for b in self._behaviors.behaviors
                if b.is_active
            ]
        n_total = len(behaviors)
        results: list[dict[str, Any]] = []

        def _outer_progress(value: int, maximum: int, log_line: str, status: str) -> None:
            if progress_cb is not None:
                progress_cb(value, maximum, log_line, status)

        # Force cache reuse on from the second behavior onward.
        # NOTE: We cannot safely mutate Qt widgets from a worker thread.
        # Instead, set a thread-safe flag that _run_pipeline_task reads.
        original_reuse = self._reuse_cached_features.isChecked()
        self._pipeline_all_force_reuse = False
        self._pipeline_all_skip_candidates = not getattr(self, "_batch_generate_clips", True)
        # Pipeline-All processes every behavior over the *same* sessions, so the
        # representation (frame/segment) tables are identical across behaviors.
        # Enable an in-memory reuse cache so behaviors 2..N skip re-reading the
        # multi-GB segment parquet and re-filtering it on every iteration.
        self._pipeline_all_active = True
        self._pipeline_all_repr_cache = None

        try:
            for idx, behavior in enumerate(behaviors):
                if cancel_flag and cancel_flag[0]:
                    break
                bid = str(behavior.behavior_id).strip()
                bname = str(behavior.name)

                # Override target behavior for this iteration.
                # Store as a thread-safe attribute instead of mutating the widget.
                self._pipeline_all_target_override = bid

                # Override model name so each behavior gets its own directory
                # instead of all sharing the global custom model name.
                self._pipeline_all_model_name_override = self._safe_model_name(bname)

                # After first behavior, force feature reuse via flag.
                if idx > 0:
                    self._pipeline_all_force_reuse = True

                _sep = "\u2501" * 18
                _outer_progress(
                    idx,
                    n_total,
                    f"{_sep} Behavior {idx + 1}/{n_total}: {bname} {_sep}",
                    f"Pipeline {idx + 1}/{n_total}: {bname}\u2026",
                )
                try:
                    result = self._run_pipeline_task(
                        progress_cb=progress_cb,
                        cancel_flag=cancel_flag,
                    )
                    result["target_behavior_name"] = bname
                except Exception as exc:
                    logger.error("Pipeline-all: failed on behavior '%s': %s", bname, exc)
                    result = {"target_behavior": bid, "target_behavior_name": bname, "error": str(exc)}
                results.append(result)
        finally:
            # Clean up thread-safe overrides.
            self._pipeline_all_force_reuse = False
            self._pipeline_all_target_override = None
            self._pipeline_all_model_name_override = None
            self._pipeline_all_skip_candidates = False
            self._pipeline_all_active = False
            self._pipeline_all_repr_cache = None

        succeeded = sum(1 for r in results if r.get("summary") is not None)
        _outer_progress(
            n_total,
            n_total,
            f"Pipeline-all complete: {succeeded}/{n_total} behaviors processed.",
            "Pipeline all behaviors complete.",
        )
        return {"pipeline_all": True, "results": results, "succeeded": succeeded, "total": n_total}

    def _on_pipeline_all_behaviors_finished(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        self._cancel_flag[0] = False
        self._pipeline_step_scale = 1
        self._refresh_saved_model_options()
        results = payload.get("results", [])
        succeeded = int(payload.get("succeeded", 0))
        total = int(payload.get("total", 0))

        all_candidates: list[Any] = []
        for r in results:
            summary = r.get("summary")
            target_behavior = str(r.get("target_behavior", ""))
            if summary is not None:
                model_version = str(getattr(summary, "model_version", ""))
                if model_version:
                    self._save_workflow_snapshot(
                        model_version=model_version,
                        target_behavior=target_behavior,
                    )
                    self._snapshot_evaluation_graphs_for_model(model_version, target_behavior=target_behavior)
            candidates = r.get("candidates", [])
            all_candidates.extend(candidates)

        self._refresh_viz_model_options()
        self._progress.setValue(self._progress.maximum())
        self._progress.setFormat("Complete")
        self._status.setText(
            f"Pipeline-all complete: {succeeded}/{total} behaviors processed."
        )

        for r in results:
            bname = str(r.get("target_behavior_name", r.get("target_behavior", "?")))
            summary = r.get("summary")
            if summary is not None:
                metrics = r.get("metrics", {})
                self._append_log(
                    f"  {bname}: F1={float(metrics.get('f1', 0.0)):.3f}, "
                    f"PR-AUC={float(metrics.get('pr_auc', 0.0)):.3f}, "
                    f"candidates={int(getattr(summary, 'n_candidates', 0))}"
                )
            elif r.get("error"):
                self._append_log(f"  {bname}: FAILED \u2014 {r['error']}")

        self._append_log(f"Pipeline-all summary: {succeeded}/{total} succeeded.")

        # Emit all candidates to clip extraction.
        if all_candidates:
            self._emit_uncertainty_candidates_for_clip_extraction(
                all_candidates,
                source_label="Active Learning \u2014 Pipeline All Behaviors",
                append=True,
            )
        self._generate_expert_assignment_chart()

        # Auto-generate cross-behaviour confound analysis and unified UMAP
        # now that all models have fresh predictions.
        if succeeded > 0 and self._project_root:
            self._append_log("Generating cross-behaviour confound analysis…")
            try:
                from abel.services.evaluation_service import EvaluationService as _ES
                svc = _ES()
                behavior_names = {
                    str(b.behavior_id).strip(): str(b.short_name or b.name)
                    for b in self._behaviors.behaviors
                    if b.behavior_id != NO_BEHAVIOR_ID and b.is_active
                }
                confound = svc.generate_cross_behavior_confound_report(
                    self._project_root,
                    behavior_names=behavior_names,
                    target_behavior_id=self._selected_target_behavior_id(),
                )
                if confound.get("error"):
                    self._append_log(f"Confound analysis: {confound['error']}")
                else:
                    n_behaviors = len(confound.get("behavior_ids", []))
                    self._append_log(f"Confound analysis complete: {n_behaviors} behaviours.")
            except Exception as exc:
                self._append_log(f"Confound analysis failed: {exc}")

            if self._is_umap_enabled():
                self._append_log("Generating unified behaviour UMAP…")
                try:
                    from abel.services.evaluation_service import EvaluationService as _ES2
                    svc2 = _ES2()
                    umap_result = svc2.generate_unified_umap(
                        self._project_root,
                        behavior_names=behavior_names,
                        predicted_to_labeled_ratio=float(self._umap_pred_ratio.value()),
                        target_behavior_label=next(
                            (str(b.name) for b in self._behaviors.behaviors
                             if str(b.behavior_id).strip() == str(self._selected_target_behavior_id()).strip()),
                            None,
                        ),
                    )
                    if umap_result.get("error"):
                        self._append_log(f"Unified UMAP: {umap_result['error']}")
                    else:
                        self._append_log(
                            f"Unified UMAP complete: {umap_result.get('n_segments', 0)} segments, "
                            f"{umap_result.get('method', 'PCA')}."
                        )
                except Exception as exc:
                    self._append_log(f"Unified UMAP failed: {exc}")
            else:
                self._append_log("UMAP generation disabled in settings — skipping.")

        self._refresh_visualization_preview()

    # ------------------------------------------------------------------
    # Run Models (inference-only, selected behaviors)
    # ------------------------------------------------------------------

    def _run_models_selected_behaviors(self) -> None:
        """Run inference for user-selected behaviors using their existing models.

        Like Pipeline All, but no training — each selected behavior is scored
        with its most recent trained model.
        """
        if not self._project_root:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return
        # Resolve each behaviour to its newest trained model directory.
        model_map = self._snapshot_svc._auto_resolve_behavior_models(self._project_root)
        if not model_map:
            QMessageBox.information(
                self, "Run Models",
                "No trained behavior models were found. Train at least one "
                "behavior first (Run Pipeline / Pipeline All).",
            )
            return
        name_by_id = {
            str(b.behavior_id).strip(): str(b.name)
            for b in self._behaviors.behaviors
        }
        # Only offer behaviours that actually have a model.
        entries = [
            (bid, name_by_id.get(bid, bid), mv)
            for bid, mv in model_map.items()
        ]
        entries.sort(key=lambda e: e[1].lower())

        dlg = QDialog(self)
        dlg.setWindowTitle("Run Models — Select Behaviors")
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.addWidget(QLabel(
            f"{len(entries)} behavior(s) have a trained model.  "
            "Select which to run inference for:\n\n"
            "Each selected behavior is scored with its existing model on this "
            "project's data. No retraining is performed."
        ))
        checks: list[tuple[QCheckBox, tuple[str, str, str]]] = []
        for bid, bname, mv in entries:
            cb = QCheckBox(f"{bname}   (model: {mv})")
            cb.setChecked(True)
            dlg_layout.addWidget(cb)
            checks.append((cb, (bid, bname, mv)))
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        dlg_layout.addWidget(btn_box)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = [e for cb, e in checks if cb.isChecked()]
        if not selected:
            QMessageBox.information(self, "Nothing selected", "No behaviors were selected.")
            return

        self._persist_ui_settings_to_project()
        self._refresh_segment_settings_display()
        self._set_busy(True)
        self._cancel_flag[0] = False
        self._pipeline_step_scale = 100
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Initializing…")
        self._run_models_selected = selected
        # Capture the clip-generation toggle on the GUI thread for the worker.
        self._batch_generate_clips = self._batch_generate_clips_enabled()
        self._status.setText(f"Running models for {len(selected)} behavior(s)…")
        self._append_log(f"Starting model inference for {len(selected)} behavior(s).")
        worker = TaskWorker(
            self._run_models_selected_task,
            self._pipeline_progress_updated.emit,
            self._cancel_flag,
        )
        worker.signals.finished.connect(self._on_run_models_selected_finished)
        worker.signals.failed.connect(self._on_failed)
        self._pool.start(worker)

    def _run_models_selected_task(
        self,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> dict[str, Any]:
        """Run _run_existing_model_task for each selected (behavior, model)."""
        selected = getattr(self, "_run_models_selected", []) or []
        n_total = len(selected)
        results: list[dict[str, Any]] = []
        all_candidates: list[Any] = []
        # Honor the "Generate Review Clips" toggle — _run_existing_model_task
        # reads this thread-safe flag to decide whether to generate candidates.
        self._pipeline_all_skip_candidates = not getattr(self, "_batch_generate_clips", True)
        try:
            for idx, (bid, bname, model_version) in enumerate(selected):
                if cancel_flag and cancel_flag[0]:
                    break
                _sep = "━" * 18
                if progress_cb is not None:
                    progress_cb(
                        idx, n_total,
                        f"{_sep} Model {idx + 1}/{n_total}: {bname} {_sep}",
                        f"Running {idx + 1}/{n_total}: {bname}…",
                    )
                try:
                    result = self._run_existing_model_task(
                        model_version,
                        progress_cb=progress_cb,
                        cancel_flag=cancel_flag,
                    )
                    result["target_behavior_name"] = bname
                    all_candidates.extend(result.get("candidates", []))
                except Exception as exc:
                    logger.error("Run-models: failed on behavior '%s': %s", bname, exc)
                    result = {"target_behavior": bid, "target_behavior_name": bname, "error": str(exc)}
                results.append(result)
        finally:
            self._pipeline_all_skip_candidates = False

        succeeded = sum(1 for r in results if not r.get("error"))
        if progress_cb is not None:
            progress_cb(
                n_total, n_total,
                f"Run-models complete: {succeeded}/{n_total} behaviors scored.",
                "Run models complete.",
            )
        return {
            "run_models": True, "results": results, "succeeded": succeeded,
            "total": n_total, "candidates": all_candidates,
        }

    def _on_run_models_selected_finished(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        self._cancel_flag[0] = False
        self._pipeline_step_scale = 1
        results = payload.get("results", [])
        succeeded = int(payload.get("succeeded", 0))
        total = int(payload.get("total", 0))
        for r in results:
            model_version = str(r.get("model_version", ""))
            target_behavior = str(r.get("target_behavior", ""))
            if model_version and not r.get("error"):
                self._snapshot_evaluation_graphs_for_model(
                    model_version, target_behavior=target_behavior
                )
        self._refresh_viz_model_options()
        self._progress.setValue(self._progress.maximum())
        self._progress.setFormat("Complete")
        self._status.setText(f"Run-models complete: {succeeded}/{total} behaviors scored.")
        self._append_log(f"Run-models summary: {succeeded}/{total} succeeded.")
        for r in results:
            bname = str(r.get("target_behavior_name", r.get("target_behavior", "?")))
            if r.get("error"):
                self._append_log(f"  {bname}: FAILED — {r['error']}")
            else:
                self._append_log(
                    f"  {bname}: segments={int(r.get('segment_rows', 0))}, "
                    f"candidates={int(r.get('n_candidates', 0))}"
                )
        candidates = list(payload.get("candidates", []))
        if candidates:
            self._populate_candidate_table(candidates)
            self._emit_uncertainty_candidates_for_clip_extraction(
                candidates, source_label="Active Learning — Run Models",
                append=True,
            )
        self._refresh_visualization_preview()

    # ------------------------------------------------------------------
    # Behavior-awareness ablation study
    # ------------------------------------------------------------------

    def _run_awareness_ablation(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return
        target = self._selected_target_behavior_id()
        if not target:
            QMessageBox.warning(
                self, "No target behavior",
                "Select a target behavior before running the ablation study.",
            )
            return
        self._set_busy(True)
        self._status.setText("Running behavior-awareness ablation study…")
        self._append_log("─── Behavior-Awareness Ablation Study ───")

        worker = TaskWorker(
            self._awareness_ablation_task,
            self._project_root,
            target,
        )
        worker.signals.finished.connect(self._on_awareness_ablation_finished)
        worker.signals.failed.connect(lambda msg: self._on_task_error("Behavior-awareness ablation", msg))
        QThreadPool.globalInstance().start(worker)

    def _awareness_ablation_task(
        self,
        project_root: Path,
        target_behavior: str,
    ) -> dict[str, Any]:
        svc = BehaviorAwarenessAblationService()
        log_lines: list[str] = []
        result = svc.run_ablation(
            project_root=project_root,
            target_behavior=target_behavior,
            classifier_family="lightgbm",
            n_folds=3,
            progress_cb=lambda msg: log_lines.append(msg),
        )
        return {
            "result": result,
            "log_lines": log_lines,
        }

    def _on_awareness_ablation_finished(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        result = payload.get("result")
        log_lines = payload.get("log_lines", [])
        for line in log_lines:
            self._append_log(line)
        if result is None:
            self._status.setText("Ablation study returned no result.")
            return
        self._status.setText(f"Ablation complete — verdict: {result.verdict}")
        if result.summary:
            self._append_log("")
            for line in result.summary.split("\n"):
                self._append_log(line)
        if result.warnings:
            for w in result.warnings:
                self._append_log(f"⚠ {w}")

    # ------------------------------------------------------------------
    # Cross-behaviour confound analysis
    # ------------------------------------------------------------------

    def _generate_confound_graph(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return
        self._set_busy(True)
        self._status.setText("Generating cross-behaviour confound analysis…")
        self._append_log("Generating confound analysis…")

        worker = TaskWorker(
            self._confound_graph_task,
            self._pipeline_progress_updated.emit,
            self._cancel_flag,
        )
        worker.signals.finished.connect(self._on_confound_graph_finished)
        worker.signals.failed.connect(lambda msg: self._on_task_error("Confound analysis", msg))
        QThreadPool.globalInstance().start(worker)

    def _confound_graph_task(
        self,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> dict[str, Any]:
        assert self._project_root is not None
        behavior_names = {
            str(b.behavior_id).strip(): str(b.short_name or b.name)
            for b in self._behaviors.behaviors
            if b.behavior_id != NO_BEHAVIOR_ID and b.is_active
        }
        target_bid = self._selected_target_behavior_id()
        from abel.services.evaluation_service import EvaluationService as _ES  # noqa: PLC0415

        svc = _ES()
        return svc.generate_cross_behavior_confound_report(
            self._project_root,
            behavior_names=behavior_names,
            target_behavior_id=target_bid,
        )

    def _on_confound_graph_finished(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        if payload.get("error"):
            self._status.setText(f"Confound analysis: {payload['error']}")
            self._append_log(f"Confound analysis failed: {payload['error']}")
            QMessageBox.warning(self, "Confound Analysis", str(payload["error"]))
            return
        out_path = payload.get("out_path")
        suggestions = payload.get("suggestions", [])
        self._status.setText("Confound analysis complete.")
        for s in suggestions:
            self._append_log(f"  \u2022 {s}")
        if out_path:
            self._append_log(f"Saved: {out_path}")
            self._show_image_in_viz_preview(Path(out_path))
        self._refresh_visualization_preview()

    # ------------------------------------------------------------------
    # Expert assignment chart — per-model metrics overview
    # ------------------------------------------------------------------

    def _generate_expert_assignment_chart(self) -> None:
        """Scan all behaviour model directories and render a grouped bar chart
        of F1 / PR-AUC per behaviour model.  The chart is saved to
        ``derived/evaluation/expert_assignment_per_model.png`` and displayed
        in the visualisation preview.
        """
        if self._project_root is None:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        models_root = self._project_root / "derived" / "models"
        if not models_root.exists():
            return

        # Build a lookup from behavior_id → short_name for readable labels.
        behavior_short_names: dict[str, str] = {}
        for b in self._behaviors.behaviors:
            bid = str(b.behavior_id).strip()
            if bid:
                behavior_short_names[bid] = str(b.short_name or b.name)

        # Collect metrics from every behaviour model directory
        rows: list[dict[str, Any]] = []
        for model_dir in sorted(models_root.iterdir()):
            if not model_dir.is_dir():
                continue
            metrics_path = model_dir / "metrics.json"
            settings_path = model_dir / "run_settings.json"
            if not metrics_path.exists():
                continue
            metrics = read_json(metrics_path, {})
            settings = read_json(settings_path, {}) if settings_path.exists() else {}
            raw_bid = str(settings.get("target_behavior", model_dir.name))
            label = behavior_short_names.get(raw_bid, raw_bid)
            rows.append({
                "label": str(label),
                "f1": float(metrics.get("f1", 0)),
                "pr_auc": float(metrics.get("pr_auc", 0)),
                "precision": float(metrics.get("precision", 0)),
                "recall": float(metrics.get("recall", 0)),
                "n_train": int(metrics.get("n_train", 0)),
                "n_val": int(metrics.get("n_val", 0)),
            })

        if not rows:
            return

        behaviours = [r["label"] for r in rows]
        f1s = [r["f1"] for r in rows]
        pr_aucs = [r["pr_auc"] for r in rows]
        precisions = [r["precision"] for r in rows]
        recalls = [r["recall"] for r in rows]

        x = np.arange(len(behaviours))
        width = 0.2
        fig, ax = plt.subplots(figsize=(max(6, len(behaviours) * 2.2), 5))
        ax.bar(x - 1.5 * width, f1s, width, label="F1", color="#2196F3")
        ax.bar(x - 0.5 * width, pr_aucs, width, label="PR-AUC", color="#4CAF50")
        ax.bar(x + 0.5 * width, precisions, width, label="Precision", color="#FF9800")
        ax.bar(x + 1.5 * width, recalls, width, label="Recall", color="#9C27B0")

        ax.set_ylabel("Score")
        ax.set_title("Per-Model Performance — Expert Assignment Overview")
        ax.set_xticks(x)
        ax.set_xticklabels(behaviours, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)

        # Annotate with training set size
        for i, r in enumerate(rows):
            ax.text(
                i, max(r["f1"], r["pr_auc"], r["precision"], r["recall"]) + 0.03,
                f"n={r['n_train']}",
                ha="center", va="bottom", fontsize=8, color="#555",
            )

        fig.tight_layout()
        out_dir = self._project_root / "derived" / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "expert_assignment_per_model.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        self._append_log(f"Expert assignment chart saved: {out_path}")

    # ------------------------------------------------------------------
    # Interactive UMAP selection → clip extraction
    # ------------------------------------------------------------------

    def _open_interactive_umap_selection(self) -> None:
        """Open a dialog with an interactive UMAP scatter plot.

        Users can draw a rectangle selection around points of interest.
        Selected segments are emitted to the clip extraction tab as
        CandidateWindow objects.
        """
        if self._project_root is None:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return

        eval_dir = self._project_root / "derived" / "evaluation"
        sources: list[tuple[str, Path]] = []
        unified_path = eval_dir / "unified_umap_coordinates.parquet"
        unsup_path = eval_dir / "unsupervised_umap_coordinates.parquet"
        if unified_path.exists():
            sources.append(("Unified UMAP (behaviours)", unified_path))
        if unsup_path.exists():
            sources.append(("Unsupervised UMAP (clusters)", unsup_path))

        if not sources:
            QMessageBox.information(
                self,
                "No UMAP data",
                "Run 'Unified UMAP' or 'Unsupervised UMAP' first to generate the coordinate data.",
            )
            return

        if len(sources) == 1:
            coord_path = sources[0][1]
        else:
            from PySide6.QtWidgets import QInputDialog  # noqa: PLC0415
            choice, ok = QInputDialog.getItem(
                self,
                "Select UMAP Source",
                "Which embedding do you want to select from?",
                [name for name, _ in sources],
                0,
                False,
            )
            if not ok:
                return
            coord_path = next(p for name, p in sources if name == choice)

        import pandas as pd
        import numpy as np
        coord_df = pd.read_parquet(coord_path)
        required = {"segment_id", "umap_x", "umap_y", "behavior_label"}
        if not required.issubset(coord_df.columns) or coord_df.empty:
            QMessageBox.warning(self, "Bad data", "UMAP coordinate data is empty or malformed.")
            return

        from PySide6.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QLabel as _QL  # noqa: PLC0415
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: PLC0415
        from matplotlib.figure import Figure  # noqa: PLC0415
        from matplotlib.widgets import RectangleSelector  # noqa: PLC0415

        dlg = QDialog(self)
        dlg.setWindowTitle("UMAP Interactive Selection")
        dlg.resize(900, 700)
        lay = QVBoxLayout(dlg)

        hint = _QL("Click and drag to draw a rectangle around segments. Press OK to send selection to clip extraction.")
        lay.addWidget(hint)

        fig = Figure(figsize=(9, 6.5))
        canvas = FigureCanvasQTAgg(fig)
        lay.addWidget(canvas, 1)

        ax = fig.add_subplot(111)
        xs = coord_df["umap_x"].to_numpy(dtype=float)
        ys = coord_df["umap_y"].to_numpy(dtype=float)
        labels = coord_df["behavior_label"].to_numpy()

        import matplotlib.pyplot as plt
        classes = sorted(set(labels))
        cmap = plt.get_cmap("tab10", max(1, len(classes)))
        color_map = {cls: cmap(i) for i, cls in enumerate(classes)}
        colors = [color_map.get(l, (0.5, 0.5, 0.5, 1.0)) for l in labels]

        ax.scatter(xs, ys, s=6, c=colors, alpha=0.5, rasterized=True)
        ax.set_title("Select a region to extract clips")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.tight_layout()

        # Track selected indices
        selected_indices: list[int] = []
        highlight_collection = [None]  # mutable ref

        def _on_select(eclick, erelease):
            x1, x2 = sorted([eclick.xdata, erelease.xdata])
            y1, y2 = sorted([eclick.ydata, erelease.ydata])
            mask = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
            selected_indices.clear()
            selected_indices.extend(np.where(mask)[0].tolist())
            # Highlight selected
            if highlight_collection[0] is not None:
                highlight_collection[0].remove()
            highlight_collection[0] = ax.scatter(
                xs[mask], ys[mask], s=20, facecolors="none",
                edgecolors="red", linewidths=1.2, zorder=10,
            )
            hint.setText(f"{len(selected_indices)} segments selected. Press OK to send to clip extraction.")
            canvas.draw_idle()

        _selector = RectangleSelector(  # noqa: F841
            ax, _on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        # Keep a reference to prevent garbage collection
        canvas._rect_selector = _selector  # type: ignore[attr-defined]

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted or not selected_indices:
            return

        # Build CandidateWindow objects from selected segments
        from abel.models.schemas import CandidateWindow  # noqa: PLC0415

        windows: list[CandidateWindow] = []
        for idx in selected_indices:
            row = coord_df.iloc[idx]
            sid = str(row.get("session_id", ""))
            seg_id = str(row["segment_id"])
            start = int(row["start_frame"]) if "start_frame" in row.index else 0
            end = int(row["end_frame"]) if "end_frame" in row.index else start + 1
            windows.append(CandidateWindow(
                window_id=seg_id,
                session_id=sid,
                start_frame=start,
                end_frame=end,
                behavior_id=str(row.get("behavior_label", "")),
                total_score=1.0,
                source="umap_interactive_selection",
                selection_reason="umap_selection",
            ))

        if windows:
            self.uncertainty_candidates_updated.emit(
                windows, f"UMAP Selection ({len(windows)} segments)"
            )
            self._append_log(f"Sent {len(windows)} UMAP-selected segments to clip extraction.")

    # ------------------------------------------------------------------
    # Unified UMAP across all behaviour models
    # ------------------------------------------------------------------

    def _regenerate_unified_umap_inline(self) -> None:
        """Regenerate unified UMAP synchronously (called after retrain)."""
        if not self._project_root:
            return
        self._append_log("Regenerating unified behaviour UMAP\u2026")
        try:
            from abel.services.evaluation_service import EvaluationService as _ES_u

            svc_u = _ES_u()
            behavior_names = {
                str(b.behavior_id).strip(): str(b.short_name or b.name)
                for b in self._behaviors.behaviors
                if b.is_active
            }
            umap_result = svc_u.generate_unified_umap(
                self._project_root,
                behavior_names=behavior_names,
                predicted_to_labeled_ratio=float(self._umap_pred_ratio.value()),
                target_behavior_label=next(
                    (
                        str(b.name)
                        for b in self._behaviors.behaviors
                        if str(b.behavior_id).strip()
                        == str(self._selected_target_behavior_id()).strip()
                    ),
                    None,
                ),
            )
            if umap_result.get("error"):
                self._append_log(f"Unified UMAP: {umap_result['error']}")
            else:
                self._append_log(
                    f"Unified UMAP complete: {umap_result.get('n_segments', 0)} segments, "
                    f"{umap_result.get('method', 'PCA')}."
                )
                out_path = umap_result.get("out_path")
                if out_path:
                    self._show_image_in_viz_preview(Path(out_path))
        except Exception as exc:
            self._append_log(f"Unified UMAP failed: {exc}")
        self._refresh_visualization_preview()

    def _generate_unified_umap(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return

        # --- Settings dialog ---------------------------------------------------
        from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QSpinBox as _QSpinBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Unified UMAP Settings")
        form = QFormLayout(dlg)

        nn_spin = _QSpinBox(dlg)
        nn_spin.setRange(3, 100)
        nn_spin.setValue(15)
        nn_spin.setToolTip("Controls local vs global structure. Lower = more local clusters, higher = broader patterns.")
        form.addRow("n_neighbors:", nn_spin)

        cap_spin = _QSpinBox(dlg)
        cap_spin.setRange(500, 50000)
        cap_spin.setValue(8000)
        cap_spin.setSingleStep(500)
        cap_spin.setToolTip("Max total points (labeled clips always included, unlabeled subsampled up to this).")
        form.addRow("Subsample cap:", cap_spin)

        from PySide6.QtWidgets import QDoubleSpinBox as _QDoubleSpinBox
        ratio_spin = _QDoubleSpinBox(dlg)
        ratio_spin.setRange(1.0, 50.0)
        ratio_spin.setSingleStep(0.5)
        ratio_spin.setDecimals(1)
        ratio_spin.setValue(float(self._umap_pred_ratio.value()))
        ratio_spin.setToolTip("Max ratio of predicted to reviewed segments per class.")
        form.addRow("Predicted / labeled ratio:", ratio_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._umap_n_neighbors = nn_spin.value()
        self._umap_cap = cap_spin.value()
        self._umap_pred_ratio.setValue(ratio_spin.value())

        self._set_busy(True)
        self._status.setText("Generating unified UMAP embedding…")
        self._append_log("Generating unified UMAP…")

        worker = TaskWorker(
            self._unified_umap_task,
            self._pipeline_progress_updated.emit,
            self._cancel_flag,
        )
        worker.signals.finished.connect(self._on_unified_umap_finished)
        worker.signals.failed.connect(lambda msg: self._on_task_error("Unified UMAP", msg))
        QThreadPool.globalInstance().start(worker)

    def _unified_umap_task(
        self,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> dict[str, Any]:
        assert self._project_root is not None
        behavior_names = {
            str(b.behavior_id).strip(): str(b.short_name or b.name)
            for b in self._behaviors.behaviors
            if b.behavior_id != NO_BEHAVIOR_ID and b.is_active
        }
        target_bid = self._selected_target_behavior_id()
        target_short = behavior_names.get(str(target_bid).strip(), "")
        from abel.services.evaluation_service import EvaluationService as _ES  # noqa: PLC0415

        svc = _ES()
        return svc.generate_unified_umap(
            self._project_root,
            behavior_names=behavior_names,
            cap=getattr(self, "_umap_cap", 8000),
            n_neighbors=getattr(self, "_umap_n_neighbors", 15),
            predicted_to_labeled_ratio=float(self._umap_pred_ratio.value()),
            target_behavior_label=target_short,
        )

    def _on_unified_umap_finished(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        if payload.get("error"):
            self._status.setText(f"Unified UMAP: {payload['error']}")
            self._append_log(f"Unified UMAP failed: {payload['error']}")
            QMessageBox.warning(self, "Unified UMAP", str(payload["error"]))
            return
        out_path = payload.get("out_path")
        method = payload.get("method", "UMAP")
        n_segments = payload.get("n_segments", 0)
        behaviors = payload.get("behaviors_used", [])
        self._status.setText(f"Unified UMAP complete ({method}, {n_segments} segments).")
        self._append_log(f"Unified {method}: {n_segments} segments, behaviours={behaviors}.")
        if out_path:
            self._append_log(f"Saved: {out_path}")
            self._show_image_in_viz_preview(Path(out_path))
        self._refresh_visualization_preview()

    # ------------------------------------------------------------------
    # Unsupervised UMAP (raw features, no models/labels)
    # ------------------------------------------------------------------

    def _generate_unsupervised_umap(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return

        from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QSpinBox as _QSpinBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Unsupervised UMAP Settings")
        form = QFormLayout(dlg)

        nn_spin = _QSpinBox(dlg)
        nn_spin.setRange(3, 100)
        nn_spin.setValue(int(getattr(self, "_unsup_umap_n_neighbors", 15)))
        nn_spin.setToolTip("Local vs global structure. Lower = more local clusters, higher = broader patterns.")
        form.addRow("n_neighbors:", nn_spin)

        cap_spin = _QSpinBox(dlg)
        cap_spin.setRange(500, 50000)
        cap_spin.setSingleStep(500)
        cap_spin.setValue(int(getattr(self, "_unsup_umap_cap", 8000)))
        cap_spin.setToolTip("Max segments to embed (stratified subsample by session).")
        form.addRow("Subsample cap:", cap_spin)

        mcs_spin = _QSpinBox(dlg)
        mcs_spin.setRange(5, 2000)
        mcs_spin.setSingleStep(5)
        mcs_spin.setValue(int(getattr(self, "_unsup_umap_min_cluster_size", 50)))
        mcs_spin.setToolTip(
            "HDBSCAN minimum cluster size. Larger = fewer, broader clusters;\n"
            "smaller = more, finer clusters. Points in no dense region become 'Noise'."
        )
        form.addRow("Min cluster size:", mcs_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._unsup_umap_n_neighbors = nn_spin.value()
        self._unsup_umap_cap = cap_spin.value()
        self._unsup_umap_min_cluster_size = mcs_spin.value()

        self._set_busy(True)
        self._progress.setRange(0, 5)
        self._progress.setValue(0)
        self._progress.setFormat("Starting…")
        self._status.setText("Generating unsupervised UMAP embedding…")
        self._append_log(
            "Generating unsupervised UMAP from raw features "
            "(first run compiles UMAP/numba — this can take a minute)…"
        )

        worker = TaskWorker(
            self._unsupervised_umap_task,
            self._pipeline_progress_updated.emit,
            self._cancel_flag,
        )
        worker.signals.finished.connect(self._on_unsupervised_umap_finished)
        worker.signals.failed.connect(lambda msg: self._on_task_error("Unsupervised UMAP", msg))
        QThreadPool.globalInstance().start(worker)

    def _unsupervised_umap_task(
        self,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> dict[str, Any]:
        assert self._project_root is not None
        from abel.services.evaluation_service import EvaluationService as _ES  # noqa: PLC0415

        svc = _ES()
        return svc.generate_unsupervised_umap(
            self._project_root,
            cap=int(getattr(self, "_unsup_umap_cap", 8000)),
            n_neighbors=int(getattr(self, "_unsup_umap_n_neighbors", 15)),
            min_cluster_size=int(getattr(self, "_unsup_umap_min_cluster_size", 50)),
            progress_cb=progress_cb,
        )

    def _on_unsupervised_umap_finished(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setFormat("Complete")
        if payload.get("error"):
            self._status.setText(f"Unsupervised UMAP: {payload['error']}")
            self._append_log(f"Unsupervised UMAP failed: {payload['error']}")
            QMessageBox.warning(self, "Unsupervised UMAP", str(payload["error"]))
            return
        out_path = payload.get("out_path")
        method = payload.get("method", "UMAP")
        n_segments = payload.get("n_segments", 0)
        n_clusters = payload.get("n_clusters", 0)
        cluster_method = payload.get("cluster_method", "")
        self._status.setText(
            f"Unsupervised UMAP complete ({method}, {n_segments} segments, "
            f"{n_clusters} clusters via {cluster_method})."
        )
        self._append_log(
            f"Unsupervised {method}: {n_segments} segments, {n_clusters} clusters ({cluster_method})."
        )
        if out_path:
            self._append_log(f"Saved: {out_path}")
            self._show_image_in_viz_preview(Path(out_path))
        self._refresh_visualization_preview()

    def _run_existing_model(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return
        self._persist_ui_settings_to_project()
        chosen = str(self._saved_model_combo.currentData() or "").strip()
        if not chosen:
            QMessageBox.information(self, "Active Learning", "Select a saved model to run.")
            return

        self._refresh_segment_settings_display()
        self._set_busy(True)
        self._cancel_flag[0] = False
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Initializing…")
        self._status.setText(f"Running existing model: {chosen}…")
        self._append_log(f"Starting existing-model run: {chosen}")

        worker = TaskWorker(self._run_existing_model_task, chosen, self._pipeline_progress_updated.emit, self._cancel_flag)
        worker.signals.finished.connect(self._on_existing_model_finished)
        worker.signals.failed.connect(self._on_failed)
        self._pool.start(worker)

    def _run_existing_model_task(
        self,
        model_version: str,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> dict[str, Any]:
        assert self._project_root is not None

        started_at = time.monotonic()

        def _fmt_duration(seconds: float) -> str:
            if seconds < 1.0:
                return f"{seconds * 1000.0:.0f} ms"
            if seconds < 60.0:
                return f"{seconds:.1f} s"
            mins = int(seconds // 60)
            rem = int(seconds % 60)
            return f"{mins}m {rem:02d}s"

        def _progress(
            value: int,
            maximum: int,
            log_line: str,
            status: str,
            step_seconds: float | None = None,
        ) -> None:
            if progress_cb is not None:
                elapsed = max(0.0, time.monotonic() - started_at)
                done_steps = max(1, int(value))
                avg_per_step = elapsed / done_steps
                remaining_steps = max(0, int(maximum) - int(value))
                eta_seconds = remaining_steps * avg_per_step
                eta_local = datetime.now() + timedelta(seconds=eta_seconds)
                step_text = f" | step took {_fmt_duration(step_seconds)}" if step_seconds is not None else ""
                timing_text = (
                    f"{step_text} | elapsed {_fmt_duration(elapsed)}"
                    f" | ETA {_fmt_duration(eta_seconds)}"
                    f" | finish ~ {eta_local.strftime('%H:%M:%S')}"
                )
                progress_cb(value, maximum, f"{log_line}{timing_text}", status)

        def _check_cancel() -> None:
            if cancel_flag and cancel_flag[0]:
                raise PipelineCancelledError("PIPELINE_CANCELLED_BY_USER")

        mode = str(self._mode.currentData() or "uncertainty")
        random_absent_mode = mode == "random_absent"

        run_evaluation = not bool(self._skip_evaluation.isChecked())
        total_steps = 2 if random_absent_mode else (5 if run_evaluation else 4)
        current_step = 0
        behavior_cfg = self._load_behavior_cfg()
        target_behavior = self._selected_target_behavior_id()
        segment_window, segment_stride, segment_source = self._resolved_segment_settings(behavior_cfg)

        model_dir = self._project_root / "derived" / "models" / str(model_version)
        model_state = model_dir / "model_state.pkl"
        if not random_absent_mode and not model_state.exists():
            raise FileNotFoundError(f"Saved model not found: {model_state}")

        # ── Load saved model settings to avoid re-assessments ─────────────
        # When running an existing model, use the settings it was trained with
        # (window/stride/excluded features/target behavior) rather than
        # forcing the user to match them manually.  This skips re-assessment
        # of feature family and other configuration decisions.
        saved_settings = read_json(model_dir / "run_settings.json", {})
        if saved_settings:
            saved_window = int(saved_settings.get("segment_window_frames", segment_window))
            saved_stride = int(saved_settings.get("segment_stride_frames", segment_stride))
            saved_target = str(saved_settings.get("target_behavior", target_behavior)).strip()
            if saved_window > 0:
                segment_window = saved_window
            if saved_stride > 0:
                segment_stride = saved_stride
            if saved_target:
                target_behavior = saved_target
            saved_ui = saved_settings.get("ui_settings", {})
            if isinstance(saved_ui, dict) and "excluded_feature_cols" in saved_ui:
                saved_excluded = frozenset(
                    str(c) for c in (saved_ui["excluded_feature_cols"] or [])
                )
            else:
                saved_excluded = frozenset(self._excluded_feature_cols)

        selected_session_ids: list[str] = []
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is not None and manifest.linked_sessions:
            scoped_sessions = self._resolve_linked_sessions_for_active_learning(manifest)
            selected_session_ids = sorted({str(s.session_id) for s in scoped_sessions})

        _progress(
            current_step,
            total_steps,
            (
                f"Running saved model '{model_version}' with segment settings "
                f"window={segment_window}, stride={segment_stride} ({segment_source})."
            ),
            "Preparing existing model run…",
            0.0,
        )

        if random_absent_mode:
            _check_cancel()
            step_started = time.monotonic()
            cfg = self._segment_candidate_config(
                mode=mode,
                target_behavior_id=None,
                model_version=str(model_version),
                hard_negative_ratio=float(behavior_cfg.hard_negative_sampling_ratio),
                sample_window_frames=segment_window,
                selected_session_ids=selected_session_ids,
            )
            cand = self._candidates.generate_random_absent_candidates(cfg)
            if not cand.success:
                raise ValueError("Random absent candidate generation failed: " + "; ".join(cand.warnings))
            self._candidates.save_segment_candidates(cand, cfg)

            current_step += 1
            _progress(
                current_step,
                total_steps,
                (
                    f"Generated {int(cand.n_segments_selected)} random low-probability/absent candidate segment(s) "
                    "without model inference."
                ),
                "Sampling random absent windows…",
                time.monotonic() - step_started,
            )

            current_step = total_steps
            _progress(
                current_step,
                total_steps,
                f"Existing-model run complete. Generated {int(cand.n_segments_selected)} candidate segment(s).",
                "Existing model run complete.",
                0.0,
            )

            return {
                "model_version": str(model_version),
                "target_behavior": target_behavior,
                "segment_rows": 0,
                "n_candidates": int(cand.n_segments_selected),
                "candidates": list(cand.candidates),
            }

        frame_pose_path = self._project_root / "derived" / "pose_features" / "frame_pose.parquet"
        _use_vid = bool(behavior_cfg.use_video_features)
        _ctx_candidate = self._project_root / "derived" / "context_features" / "frame_context.parquet"
        frame_ctx_path = _ctx_candidate if (_use_vid and _ctx_candidate.exists()) else None
        if not frame_pose_path.exists():
            raise FileNotFoundError(
                "Missing cached frame pose features. Run the full pipeline once to build them."
            )

        _check_cancel()
        step_started = time.monotonic()

        # Use the excluded features from the saved model settings, not the
        # current UI state, so we don't re-assess feature family selection.
        excluded_features = (
            saved_excluded if saved_settings else frozenset(self._excluded_feature_cols)
        )

        # Try to reuse cached segment features if config matches what was used
        # for the saved model.  This avoids the expensive representation rebuild.
        # Respects the reuse_cached_features checkbox: when unchecked, force a
        # full rebuild so that freshly re-extracted pose/context features are
        # reflected in the segment representations.
        reuse_cached_features = bool(self._reuse_cached_features.isChecked())
        cached_seg_path = self._project_root / "derived" / "representations" / "segment_features.parquet"
        cached_frame_path = self._project_root / "derived" / "representations" / "frame_features.parquet"
        repr_manifest_path = self._project_root / "derived" / "representations" / "representations.manifest.json"
        repr_manifest = read_json(repr_manifest_path, {}) if repr_manifest_path.exists() else {}
        cache_match = (
            reuse_cached_features
            and cached_seg_path.exists()
            and cached_frame_path.exists()
            and int(repr_manifest.get("window_size_frames", 0)) == segment_window
            and int(repr_manifest.get("window_stride_frames", 0)) == segment_stride
        )

        if cache_match:
            _progress(
                current_step,
                total_steps,
                "Reusing cached segment representations (config matches saved model).",
                "Loading cached representations…",
                None,
            )
            # Only the 3 identifier columns are needed downstream (evaluation
            # label assignment). Reading only these columns avoids decompressing
            # the full 3-4 GB parquet file into a 25+ GB in-memory DataFrame.
            frame_df = pd.read_parquet(cached_frame_path, columns=["frame", "animal_id", "session_id"])
            segment_df = pd.read_parquet(cached_seg_path)
            if selected_session_ids and "session_id" in segment_df.columns:
                sid_set = set(selected_session_ids)
                segment_df = segment_df[segment_df["session_id"].astype(str).isin(sid_set)].reset_index(drop=True)
                frame_df = frame_df[frame_df["session_id"].astype(str).isin(sid_set)].reset_index(drop=True)
        else:
            def _repr_progress(msg: str) -> None:
                _progress(
                    current_step,
                    total_steps,
                    msg,
                    "Building behavior representations…",
                    None,
                )

            frame_df, segment_df = self._repr.build(
                project_root=self._project_root,
                frame_pose_path=frame_pose_path,
                frame_context_path=frame_ctx_path,
                config=RepresentationConfig(
                    window_size_frames=segment_window,
                    window_stride_frames=segment_stride,
                    excluded_feature_cols=excluded_features,
                ),
                session_ids=set(selected_session_ids) if selected_session_ids else None,
                progress_cb=_repr_progress,
            )
        self._persist_segment_settings(segment_window, segment_stride)
        current_step += 1
        _progress(
            current_step,
            total_steps,
            f"Built representations for {len(segment_df)} segment(s).",
            "Building representations…",
            time.monotonic() - step_started,
        )

        _check_cancel()
        step_started = time.monotonic()
        train_df = self._build_training_set(segment_df, target_behavior)
        pred_df = self._infer_with_uncertainty(
            segment_df,
            train_df,
            model_dir,
            behavior_cfg,
            target_behavior,
            use_fusion=not bool(self._skip_fusion.isChecked()),
            strict_gpu=bool(self._strict_gpu.isChecked()),
            fusion_diagnostics={},
            progress_cb=lambda msg: _progress(current_step, total_steps, f"[Inference] {msg}", f"Inference: {msg}", None),
        )
        pred_df[["segment_id", "prediction_prob"]].to_parquet(model_dir / "segment_predictions.parquet", index=False)
        pred_df[["segment_id", "uncertainty_score", "uncertainty_entropy", "prediction_variance", "density_outlier_score"]].to_parquet(
            model_dir / "segment_uncertainty.parquet", index=False
        )
        current_step += 1
        _progress(
            current_step,
            total_steps,
            f"Scored segments with saved model '{model_version}'.",
            "Running model inference…",
            time.monotonic() - step_started,
        )

        _check_cancel()
        step_started = time.monotonic()
        _skip_candidates = getattr(self, "_pipeline_all_skip_candidates", False)
        n_candidates = 0
        candidates_out: list[Any] = []
        if _skip_candidates:
            current_step += 1
            _progress(
                current_step,
                total_steps,
                "Skipped candidate generation (batch pipeline mode).",
                "Skipped candidate generation.",
                0.0,
            )
        else:
            cfg = self._segment_candidate_config(
                mode=mode,
                target_behavior_id=target_behavior,
                model_version=str(model_version),
                hard_negative_ratio=float(behavior_cfg.hard_negative_sampling_ratio),
                sample_window_frames=segment_window,
                selected_session_ids=selected_session_ids,
            )
            cand = self._candidates.generate_segment_candidates(cfg, segment_df=segment_df)
            if not cand.success:
                raise ValueError("Segment candidate generation failed: " + "; ".join(cand.warnings))
            self._candidates.save_segment_candidates(cand, cfg)
            n_candidates = int(cand.n_segments_selected)
            candidates_out = list(cand.candidates)
            current_step += 1
            _progress(
                current_step,
                total_steps,
                f"Generated {n_candidates} candidate segment(s) from existing model run.",
                "Selecting review candidates…",
                time.monotonic() - step_started,
            )

        _check_cancel()
        step_started = time.monotonic()
        if run_evaluation:
            _progress(
                current_step,
                total_steps,
                "Running evaluation reports (this may take a moment)…",
                "Evaluating predictions and writing reports…",
                0.0,
            )
            self._evaluate_if_possible(
                pred_df,
                frame_df,
                behavior_cfg,
                target_behavior=target_behavior,
                model_version=str(model_version),
            )
            current_step += 1
            _progress(
                current_step,
                total_steps,
                "Updated evaluation artifacts for existing-model run.",
                "Evaluation complete.",
                time.monotonic() - step_started,
            )

        current_step = total_steps
        _progress(
            current_step,
            total_steps,
            f"Existing-model run complete. Generated {n_candidates} candidate segment(s).",
            "Existing model run complete.",
            0.0,
        )

        return {
            "model_version": str(model_version),
            "target_behavior": target_behavior,
            "segment_rows": int(len(segment_df)),
            "n_candidates": n_candidates,
            "candidates": candidates_out,
        }

    def _run_retrain_task(
        self,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
        target_behavior_override: str | None = None,
        skip_candidates: bool = False,
    ) -> dict[str, Any]:
        assert self._project_root is not None
        started_at = time.monotonic()

        def _progress(value: int, maximum: int, log_line: str, status: str) -> None:
            if progress_cb is not None:
                elapsed = max(0.0, time.monotonic() - started_at)
                progress_cb(
                    value,
                    maximum,
                    f"{log_line} | elapsed {elapsed:.1f}s",
                    status,
                )

        run_evaluation = not bool(self._skip_evaluation.isChecked())
        total_steps = 6 if run_evaluation else 5
        current_step = 0

        behavior_cfg = self._load_behavior_cfg()
        target_behavior = target_behavior_override or self._selected_target_behavior_id()
        target_behavior_label = self._behavior_display_name(target_behavior)
        _progress(
            current_step,
            total_steps,
            f"Retrain initialized for target behavior '{target_behavior_label}'.",
            f"Preparing retrain for '{target_behavior_label}'…",
        )

        retrain_cfg = self._training_config(behavior_cfg, target_behavior)
        segment_path = self._project_root / "derived" / "representations" / "segment_features.parquet"
        if not segment_path.exists():
            current_step = total_steps
            _progress(
                current_step,
                total_steps,
                "Segment features missing; retraining skipped.",
                "No segment features available.",
            )
            return {"retrained": False}

        try:
            segment_df = pd.read_parquet(segment_path)
        except Exception as _exc:
            # Corrupted or incompatible parquet — delete it so the next run
            # triggers a clean rebuild, then fail with a clear message.
            import logging as _logging  # noqa: PLC0415
            _logging.getLogger("abel").warning(
                "segment_features.parquet is corrupted (%s) — deleting for rebuild.", _exc
            )
            try:
                segment_path.unlink(missing_ok=True)
            except Exception:
                pass
            current_step = total_steps
            _progress(
                current_step,
                total_steps,
                f"Segment cache was corrupted and has been deleted. Please re-run 'Build Representations' then retry.",
                "Segment cache corrupted — deleted for rebuild.",
            )
            return {"retrained": False, "error": f"Segment cache corrupted: {_exc}"}
        scoped_ids: set[str] | None = None
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is not None and manifest.linked_sessions and "session_id" in segment_df.columns:
            scoped_sessions = self._resolve_linked_sessions_for_active_learning(manifest)
            scoped_ids = {str(s.session_id) for s in scoped_sessions}
            segment_df = segment_df[segment_df["session_id"].astype(str).isin(scoped_ids)].reset_index(drop=True)
            if segment_df.empty:
                # Cached segment file may be stale — try rebuilding through
                # the representation service which auto-detects this case.
                _progress(
                    current_step,
                    total_steps,
                    "Segment cache appears stale for selected sessions; attempting rebuild…",
                    "Rebuilding segment representations…",
                )
                frame_pose_path = self._project_root / "derived" / "pose_features" / "frame_pose.parquet"
                frame_ctx_path = self._project_root / "derived" / "context_features" / "frame_context.parquet"
                if frame_pose_path.exists():
                    _use_vid = bool(behavior_cfg.use_video_features)
                    _ctx_path = frame_ctx_path if (_use_vid and frame_ctx_path.exists()) else None
                    segment_window, segment_stride, _ = self._resolved_segment_settings(behavior_cfg)
                    _, segment_df = self._repr.build(
                        project_root=self._project_root,
                        frame_pose_path=frame_pose_path,
                        frame_context_path=_ctx_path,
                        config=RepresentationConfig(
                            window_size_frames=segment_window,
                            window_stride_frames=segment_stride,
                            excluded_feature_cols=frozenset(self._excluded_feature_cols),
                        ),
                        session_ids=scoped_ids,
                    )
                if segment_df.empty:
                    current_step = total_steps
                    _progress(
                        current_step,
                        total_steps,
                        "No segment rows remain after applying selected session scope.",
                        "Retraining skipped due to empty selected-session scope.",
                    )
                    return {"retrained": False}

        current_step += 1
        _progress(
            current_step,
            total_steps,
            "Loading persisted seed and review labels for retraining.",
            "Retraining model from seeds + reviewed labels…",
        )

        # Ensure the segment feature table covers all reviewed labels.
        # The cached segment_features.parquet may be a sparse subsample
        # (e.g. from a max_segments cap), causing most reviewed labels to
        # be silently dropped during the inner merge.
        segment_df = self._enrich_segment_df_for_reviewed_labels(
            segment_df,
            progress_cb=lambda msg: _progress(current_step, total_steps, msg, msg),
        )

        train_df = self._build_training_set(segment_df, target_behavior)
        if train_df.empty:
            result = None
        else:
            self._trainer.merge_and_snapshot_training_set(self._project_root, train_df)

            def _retrain_sub_progress(msg: str) -> None:
                _progress(current_step, total_steps, f"[Retrain] {msg}", f"Retraining: {msg}")

            result = self._trainer.train(
                self._project_root, retrain_cfg, session_ids=scoped_ids,
                progress_cb=_retrain_sub_progress,
            )
        if not result:
            current_step = total_steps
            _progress(
                current_step,
                total_steps,
                "No usable seed/review labels found; retraining skipped.",
                "No usable seed/review labels found.",
            )
            return {"retrained": False}

        segment_window, segment_stride, _ = self._resolved_segment_settings(behavior_cfg)
        model_dir = Path(str(result["model_dir"]))
        self._save_model_run_settings(
            model_version=model_dir.name,
            target_behavior=target_behavior,
            segment_window=segment_window,
            segment_stride=segment_stride,
            extra={"run_type": "retrain"},
        )

        metrics = result.get("metrics", {})
        current_step += 1
        _progress(
            current_step,
            total_steps,
            (
                "Retraining complete. "
                f"F1={float(metrics.get('f1', 0.0)):.3f}, "
                f"PR-AUC={float(metrics.get('pr_auc', 0.0)):.3f}."
            ),
            "Retraining complete; scoring candidates…",
        )

        train_df = self._build_training_set(segment_df, target_behavior)
        retrain_fusion_diag: dict[str, Any] = {}

        def _retrain_infer_sub_progress(msg: str) -> None:
            _progress(current_step, total_steps, f"[Inference] {msg}", f"Inference: {msg}")

        # Predict on the FULL enriched segment set (not just the scoped
        # training subset) so that every model's segment_predictions.parquet
        # covers all known segments.  Otherwise the unified UMAP fills
        # missing predictions with 0.0, corrupting the embedding.
        full_segment_df = segment_df
        if scoped_ids is not None and segment_path.exists():
            try:
                _full_seg = pd.read_parquet(segment_path)
                _full_seg = self._enrich_segment_df_for_reviewed_labels(
                    _full_seg,
                    progress_cb=lambda msg: _progress(current_step, total_steps, msg, msg),
                )
                if len(_full_seg) > len(segment_df):
                    full_segment_df = _full_seg
                    logger.info(
                        "Retrain inference: expanded from %d scoped to %d full segments.",
                        len(segment_df), len(full_segment_df),
                    )
            except Exception:
                logger.debug("Failed loading full segment set for inference; using scoped set.", exc_info=True)

        pred_df = self._infer_with_uncertainty(
            full_segment_df,
            train_df,
            model_dir,
            behavior_cfg,
            target_behavior,
            use_fusion=not bool(self._skip_fusion.isChecked()),
            strict_gpu=bool(self._strict_gpu.isChecked()),
            fusion_diagnostics=retrain_fusion_diag,
            progress_cb=_retrain_infer_sub_progress,
        )

        if bool(retrain_fusion_diag.get("fusion_used_cpu_fallback", False)):
            _progress(
                current_step,
                total_steps,
                (
                    "Fusion fallback during retrain: "
                    f"{str(retrain_fusion_diag.get('fusion_fallback_reason', '') or 'no additional details')}"
                ),
                "Continuing retrain with fallback fusion behavior…",
            )

        out_model_dir = self._project_root / "derived" / "models" / model_dir.name
        out_model_dir.mkdir(parents=True, exist_ok=True)
        pred_df[["segment_id", "prediction_prob"]].to_parquet(out_model_dir / "segment_predictions.parquet", index=False)
        pred_df[["segment_id", "uncertainty_score", "uncertainty_entropy", "prediction_variance", "density_outlier_score"]].to_parquet(
            out_model_dir / "segment_uncertainty.parquet", index=False
        )

        current_step += 1
        _progress(
            current_step,
            total_steps,
            "Updated segment predictions and uncertainty outputs.",
            "Selecting refreshed review candidates…",
        )

        n_candidates = 0
        candidates_out: list[Any] = []
        if skip_candidates:
            current_step += 1
            _progress(
                current_step,
                total_steps,
                "Skipped candidate generation (batch retrain mode).",
                "Skipped candidate generation.",
            )
        else:
            cfg = self._segment_candidate_config(
                mode=str(self._mode.currentData() or "uncertainty"),
                target_behavior_id=target_behavior,
                model_version=model_dir.name,
                hard_negative_ratio=float(behavior_cfg.hard_negative_sampling_ratio),
                selected_session_ids=(
                    sorted(segment_df["session_id"].astype(str).unique())
                    if "session_id" in segment_df.columns
                    else []
                ),
            )
            cand = self._candidates.generate_segment_candidates(cfg, segment_df=segment_df)
            if cand.success:
                self._candidates.save_segment_candidates(cand, cfg)
                candidates_out = list(cand.candidates)
            n_candidates = int(cand.n_segments_selected if cand.success else 0)

            current_step += 1
            _progress(
                current_step,
                total_steps,
                f"Generated {n_candidates} candidate segment(s).",
                "Selecting refreshed review candidates…",
            )

        if run_evaluation:
            frame_path = self._project_root / "derived" / "representations" / "frame_features.parquet"
            if not frame_path.exists():
                # frame_features.parquet was never persisted (all prior builds
                # were session-scoped).  Trigger a full (un-scoped) build so
                # the cache is created for this and future runs.
                _progress(
                    current_step,
                    total_steps,
                    "Building full frame features cache for evaluation…",
                    "Generating frame-level features (one-time)…",
                )
                try:
                    _pose = self._project_root / "derived" / "pose_features" / "frame_pose.parquet"
                    _ctx = self._project_root / "derived" / "context_features" / "frame_context.parquet"
                    if _pose.exists():
                        _use_vid = bool(behavior_cfg.use_video_features)
                        _ctx_arg = _ctx if (_use_vid and _ctx.exists()) else None
                        _seg_w, _seg_s, _ = self._resolved_segment_settings(behavior_cfg)
                        self._repr.build(
                            project_root=self._project_root,
                            frame_pose_path=_pose,
                            frame_context_path=_ctx_arg,
                            config=RepresentationConfig(
                                window_size_frames=_seg_w,
                                window_stride_frames=_seg_s,
                                excluded_feature_cols=frozenset(self._excluded_feature_cols),
                            ),
                            session_ids=None,
                            progress_cb=lambda msg: logger.info(msg),
                        )
                except Exception as exc:
                    logger.warning("Failed to build frame features cache: %s", exc)

            if frame_path.exists():
                try:
                    _progress(
                        current_step,
                        total_steps,
                        "Loading frame features and running evaluation (this may take a moment)…",
                        "Evaluating retrained model and writing reports…",
                    )
                    # Read only the 3 columns needed for evaluation — avoids
                    # loading the full 4+ GB frame_features.parquet into RAM.
                    frame_df = pd.read_parquet(frame_path, columns=["frame", "animal_id", "session_id"])
                    self._evaluate_if_possible(
                        pred_df,
                        frame_df,
                        behavior_cfg,
                        target_behavior=target_behavior,
                        model_version=model_dir.name,
                    )
                    del frame_df
                    current_step += 1
                    _progress(
                        current_step,
                        total_steps,
                        "Updated evaluation artifacts after retraining.",
                        "Evaluation complete.",
                    )
                except Exception as exc:
                    current_step += 1
                    _progress(
                        current_step,
                        total_steps,
                        f"Evaluation update skipped after retrain ({exc}).",
                        "Skipped evaluation artifact refresh.",
                    )
            else:
                current_step += 1
                _progress(
                    current_step,
                    total_steps,
                    "Evaluation update skipped after retrain (missing frame_features.parquet).",
                    "Skipped evaluation artifact refresh.",
                )

        current_step = total_steps
        _progress(
            current_step,
            total_steps,
            (
                "Retrain pipeline finished. "
                f"Generated {n_candidates} candidate segment(s)."
            ),
            "Retrain pipeline complete.",
        )

        return {
            "retrained": True,
            "target_behavior": target_behavior,
            "model_version": model_dir.name,
            "metrics": result.get("metrics", {}),
            "n_candidates": n_candidates,
            "candidates": candidates_out,
        }

    def _run_pipeline_task(
        self,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> dict[str, Any]:
        assert self._project_root is not None
        summary = _RunSummary()
        behavior_cfg = self._load_behavior_cfg()
        target_behavior = self._selected_target_behavior_id()
        target_behavior_label = self._behavior_display_name(target_behavior)
        segment_window, segment_stride, segment_source = self._resolved_segment_settings(behavior_cfg)
        started_at = time.monotonic()
        step_timings: list[tuple[str, float]] = []
        quick_test = bool(self._quick_test.isChecked())
        examples_per_session = max(0, int(self._examples_per_session.value()))
        max_segments = max(0, int(self._max_segments.value()))
        quick_ident_minutes = max(1, int(self._quick_ident_minutes.value()))
        quick_ident_seed = max(0, int(self._quick_ident_seed.value()))
        balanced_sampling = bool(self._balanced_sampling.isChecked()) and not quick_test
        balanced_sampling_minutes = max(1, int(self._balanced_sampling_minutes.value()))
        balanced_sampling_seed = max(0, int(self._balanced_sampling_seed.value()))
        reuse_cached_features = bool(self._reuse_cached_features.isChecked())
        # Thread-safe override for Pipeline-All: force feature reuse after first behavior.
        if getattr(self, "_pipeline_all_force_reuse", False):
            reuse_cached_features = True
        use_fusion = not bool(self._skip_fusion.isChecked())
        use_video_features = bool(behavior_cfg.use_video_features)
        # If video features are disabled, force-disable fusion too
        if not use_video_features:
            use_fusion = False
        run_evaluation = not bool(self._skip_evaluation.isChecked())
        run_phase1_after_eval = bool(self._phase1_enable.isChecked() and self._phase1_diagnostics.isChecked())
        strict_gpu = bool(self._strict_gpu.isChecked())

        def _fmt_duration(seconds: float) -> str:
            if seconds < 1.0:
                return f"{seconds * 1000.0:.0f} ms"
            if seconds < 60.0:
                return f"{seconds:.1f} s"
            mins = int(seconds // 60)
            rem = int(seconds % 60)
            return f"{mins}m {rem:02d}s"

        # Fragments used to identify major pipeline steps worth charting.
        _TIMED_STEP_FRAGMENTS: tuple[tuple[str, str], ...] = (
            ("Building behavior representations", "Build Representations"),
            ("Assembling training labels", "Assemble Training Set"),
            ("Training model", "Train Classifier"),
            ("Scoring segments with uncertainty model", "Inference & Uncertainty"),
            ("Selecting next review candidates", "Generate Candidates"),
            ("Evaluating predictions and writing reports", "Evaluation & Reports"),
            ("Completed Phase 1 diagnostics", "Phase 1 Diagnostics"),
        )

        # ── Seed ETA from prior run ──────────────────────────────────────
        # Load per-step timing from the most recent completed run so the
        # progress bar shows a calibrated ETA immediately instead of staying
        # indeterminate until several steps have been timed.
        _timing_seed_path = self._project_root / "derived" / "evaluation" / "pipeline_timing_seed.json"
        try:
            from abel.storage.file_store import read_json as _read_json
            _seed_data = _read_json(_timing_seed_path, {})
            _seed_secs = float(_seed_data.get("per_step_seconds", 0.0))
            if _seed_secs > 0.5:  # sanity-check: ignore implausibly small values
                _ema_step_seconds = _seed_secs
                logger.debug(
                    "ETA seeded from prior run: %.1f s/step (%d steps, %.0f s total)",
                    _seed_secs,
                    int(_seed_data.get("total_steps", 0)),
                    float(_seed_data.get("total_wall_seconds", 0.0)),
                )
        except Exception:
            pass  # no prior run or file missing — EMA will self-seed after step 1

        # Scale factor: all values emitted to progress_cb are multiplied by
        # _CHUNK_SCALE so the bar can represent fractional session progress
        # (each session = _CHUNK_SCALE bar units, each chunk = a fraction of that).
        # _apply_pipeline_progress divides by self._pipeline_step_scale (= _CHUNK_SCALE)
        # to recover the human-readable step count for the label.
        _CHUNK_SCALE: int = 100

        # Thread-safe chunk-fraction tracker: updated from parallel session workers
        # so the bar advances as individual chunks complete, not just whole sessions.
        _chunk_lock = threading.Lock()
        _chunk_fracs: dict[str, float] = {}  # session_id → fraction complete [0.0, 1.0]

        # ETA state — exponential moving average of per-step durations.
        # α=0.3 blends 30% new step, 70% history, so early outliers (GPU
        # init, MOG2 warmup) decay quickly once regular steps flow in.
        _ema_step_seconds: float | None = None
        _ema_alpha: float = 0.3
        _last_progress_time: list[float] = [started_at]  # mutable via closure

        def _progress(
            value: int,
            maximum: int,
            log_line: str,
            status: str | None = None,
            step_seconds: float | None = None,
        ) -> None:
            nonlocal _ema_step_seconds
            if step_seconds is not None and step_seconds > 0.0 and status:
                for fragment, short_label in _TIMED_STEP_FRAGMENTS:
                    if fragment in status:
                        step_timings.append((short_label, float(step_seconds)))
                        break
            if progress_cb is not None:
                elapsed = max(0.0, time.monotonic() - started_at)
                remaining_steps = max(0, int(maximum) - int(value))

                # Measure wall time since last _progress call as the step
                # duration when step_seconds is not explicitly provided.
                now = time.monotonic()
                wall_step = now - _last_progress_time[0]
                _last_progress_time[0] = now
                observed = step_seconds if (step_seconds is not None and step_seconds > 0.0) else wall_step

                # Update EMA only when the step took non-trivial time.
                if observed > 0.05:
                    if _ema_step_seconds is None:
                        _ema_step_seconds = observed
                    else:
                        _ema_step_seconds = (
                            _ema_alpha * observed
                            + (1.0 - _ema_alpha) * _ema_step_seconds
                        )

                # Fall back to simple mean when EMA not yet seeded.
                if _ema_step_seconds is not None and _ema_step_seconds > 0:
                    eta_seconds = remaining_steps * _ema_step_seconds
                elif int(value) > 0:
                    eta_seconds = (elapsed / int(value)) * remaining_steps
                else:
                    eta_seconds = 0.0

                eta_local = datetime.now() + timedelta(seconds=eta_seconds)
                step_text = f" | step took {_fmt_duration(step_seconds)}" if step_seconds is not None else ""
                timing_text = (
                    f"{step_text} | elapsed {_fmt_duration(elapsed)}"
                    f" | ETA {_fmt_duration(eta_seconds)}"
                    f" | finish ~ {eta_local.strftime('%H:%M:%S')}"
                )
                progress_cb(int(value) * _CHUNK_SCALE, int(maximum) * _CHUNK_SCALE, f"{log_line}{timing_text}", status or log_line)

        def _check_cancel() -> None:
            if cancel_flag and cancel_flag[0]:
                raise PipelineCancelledError("PIPELINE_CANCELLED_BY_USER")

        # Fast path: random bootstrap requires no pose/context preprocessing or model training.
        selected_mode = str(self._mode.currentData() or "uncertainty")
        if selected_mode == "random_absent":
            selected_random_session_ids: list[str] = []
            manifest = self._imports.load_manifest(self._project_root)
            if manifest is not None and manifest.linked_sessions:
                selected_random_session_ids = sorted(
                    str(s.session_id)
                    for s in self._resolve_linked_sessions_for_active_learning(manifest)
                )
            cfg = self._segment_candidate_config(
                mode="random_absent",
                target_behavior_id=None,
                model_version="bootstrap_random_absent",
                hard_negative_ratio=float(behavior_cfg.hard_negative_sampling_ratio),
                sample_window_frames=segment_window,
                examples_per_session=examples_per_session,
                selected_session_ids=selected_random_session_ids,
            )
            _progress(0, 2, "Generating random bootstrap windows (no model training required).", "Sampling random windows…", 0.0)
            rand = self._candidates.generate_random_absent_candidates(cfg)
            if not rand.success:
                raise ValueError(
                    "Random bootstrap failed: " + "; ".join(rand.warnings or ["unknown error"])
                )
            self._candidates.save_segment_candidates(rand, cfg)
            summary.model_version = str(cfg.model_version)
            summary.n_candidates = int(rand.n_segments_selected)
            _progress(
                2,
                2,
                f"Generated {summary.n_candidates} random bootstrap candidate window(s).",
                "Random bootstrap complete.",
                0.0,
            )
            return {
                "summary": summary,
                "target_behavior": target_behavior,
                "metrics": {"f1": float("nan"), "pr_auc": float("nan"), "n_train": 0, "n_val": 0},
                "fusion_diagnostics": {},
                "candidates": list(rand.candidates),
            }

        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None or not manifest.linked_sessions:
            raise ValueError("No linked sessions available. Import data first.")

        _progress(0, 1, "Detecting GPU/CPU backend availability…", "Detecting backend…", 0.0)
        backend = self._detect_backend_status()
        backend_summary = self._detect_backend_plan(status=backend)
        _progress(0, 1, f"Backend check: {backend_summary}", f"Backend: {backend_summary}", 0.0)
        if strict_gpu:
            if backend.get("modeling") != "GPU":
                raise RuntimeError(
                    "Strict GPU mode is enabled, but model training backend is not available on GPU. "
                    "Disable 'Require GPU (fail if fallback occurs)' to allow automatic CPU fallback."
                )
            if use_fusion and backend.get("fusion") != "GPU":
                raise RuntimeError(
                    "Strict GPU mode is enabled, but fusion backend is not available on GPU. "
                    "Disable 'Require GPU (fail if fallback occurs)' to allow automatic CPU fallback."
                )

        linked_sessions = self._resolve_linked_sessions_for_active_learning(manifest)
        summary.n_sessions = len(linked_sessions)

        # Early-exit: fail before expensive preprocessing if there are no labels.
        # Without seeds or reviewed clips there's nothing to train on; the user
        # should add seed examples first or switch to 'Random bootstrap' mode.
        has_seeds = bool(self._seeds.seeds)
        has_reviews = False
        if not has_seeds:
            review_path = self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
            if review_path.exists():
                try:
                    _probe = pd.read_parquet(review_path, columns=["segment_id"])
                    has_reviews = not _probe.empty
                except Exception:
                    pass
        if not has_seeds and not has_reviews:
            raise ValueError(
                "No labeled training data found (0 seeds, no reviewed clips). "
                "Add at least one seed example in the 'Seeds' panel before running the pipeline, "
                "or switch mode to 'Random bootstrap' to generate starter windows without labels."
            )

        cached_pose_sessions: set[str] = set()
        cached_ctx_sessions: set[str] = set()
        if reuse_cached_features:
            # Fast path: the preprocessing stage now writes one parquet per
            # session into sessions/ subdirectories.  A directory listing is
            # O(1) versus loading the entire monolithic parquet just to get
            # the session_id column.
            pose_sessions_dir = self._project_root / "derived" / "pose_features" / "sessions"
            ctx_sessions_dir  = self._project_root / "derived" / "context_features" / "sessions"
            if pose_sessions_dir.exists():
                cached_pose_sessions = {f.stem for f in pose_sessions_dir.glob("*.parquet")}
            else:
                # Legacy: monolithic file only (older projects).
                pose_cache_path = self._project_root / "derived" / "pose_features" / "frame_pose.parquet"
                if pose_cache_path.exists():
                    try:
                        pose_cache_df = pd.read_parquet(pose_cache_path, columns=["session_id"])
                        cached_pose_sessions = set(pose_cache_df["session_id"].astype(str).unique())
                    except Exception:
                        cached_pose_sessions = set()
            if ctx_sessions_dir.exists() and use_video_features:
                cached_ctx_sessions = {f.stem for f in ctx_sessions_dir.glob("*.parquet")}
            elif use_video_features:
                ctx_cache_path = self._project_root / "derived" / "context_features" / "frame_context.parquet"
                if ctx_cache_path.exists():
                    try:
                        ctx_cache_df = pd.read_parquet(ctx_cache_path, columns=["session_id"])
                        cached_ctx_sessions = set(ctx_cache_df["session_id"].astype(str).unique())
                    except Exception:
                        cached_ctx_sessions = set()

        total_steps = summary.n_sessions + 7 + (1 if run_phase1_after_eval else 0)
        current_step = 0
        _progress(
            current_step,
            total_steps,
            (
                f"Loaded project manifest with {len(manifest.linked_sessions)} session(s). "
                f"Using {summary.n_sessions} session(s){' (quick test mode)' if quick_test else ''}."
            ),
            f"Preparing pipeline for {summary.n_sessions} session(s)…",
            0.0,
        )
        if quick_test:
            _progress(
                current_step,
                total_steps,
                (
                    "Quick identification sampling enabled: "
                    f"random time windows (seed={quick_ident_seed}), "
                    f"{quick_ident_minutes} minute(s) per selected session."
                ),
                "Preparing quick identification sampling…",
                0.0,
            )
        _progress(
            current_step,
            total_steps,
            (
                "Using segment settings: "
                f"window={segment_window}, stride={segment_stride} ({segment_source})."
            ),
            "Preparing pipeline settings…",
            0.0,
        )

        fps_by_video_asset_id = {
            v.asset_id: float(v.fps)
            for v in manifest.videos
            if v.fps is not None
        }
        fps_by_session_id: dict[str, float] = {}
        session_jobs: list[tuple[int, str, str, Path, Path, float]] = []
        for i, linked in enumerate(linked_sessions, start=1):
            session_id = linked.session_id
            subject_id = linked.subject_id or session_id
            step_started = time.monotonic()
            video_path = self._imports.video_path_for_session(manifest, session_id)
            pose_path = self._imports.pose_path_for_session(manifest, session_id)
            if not video_path or not pose_path:
                current_step += 1
                _progress(
                    current_step,
                    total_steps,
                    f"[{i}/{summary.n_sessions}] Skipping {session_id}: missing video or pose path.",
                    f"Skipping {session_id}: missing required files.",
                    time.monotonic() - step_started,
                )
                continue

            cache_hit = (
                reuse_cached_features
                and session_id in cached_pose_sessions
                and (not use_video_features or session_id in cached_ctx_sessions)
            )
            if cache_hit:
                current_step += 1
                _progress(
                    current_step,
                    total_steps,
                    f"[{i}/{summary.n_sessions}] Reused cached session {session_id} (pose{' + context' if use_video_features else ''} features).",
                    f"Reused cached features for {session_id}",
                    time.monotonic() - step_started,
                )
                continue

            fps = fps_by_video_asset_id.get(linked.video_asset_id, 30.0)
            fps_by_session_id[str(session_id)] = float(fps)
            session_jobs.append((i, session_id, subject_id, video_path, pose_path, fps))

        _preproc_wall_start = time.monotonic()
        if session_jobs:
            # ── Probe & cache video metadata (width, height, fps, frame_count) ──
            # This is a fast one-time step (~1 ms/video, no frame decoding).
            # Populating the manifest avoids re-opening each video during
            # downsample factor resolution and makes dimensions visible in the UI.
            try:
                from abel.services.import_service import ImportService as _IS
                _manifest_path = self._project_root / "derived" / "review_tables" / "import_manifest.json"
                if _manifest_path.exists():
                    from abel.storage.file_store import read_json as _rj
                    from abel.models.schemas import ImportManifest as _IM
                    _raw = _rj(_manifest_path, {})
                    _manifest = _IM(**_raw)
                    _IS().probe_and_cache_video_metadata(_manifest, _manifest_path)
            except Exception:
                pass

            requested_workers = 0
            env_workers = os.environ.get("ABEL_PREPROCESS_WORKERS", "").strip()
            if env_workers:
                try:
                    requested_workers = max(1, int(env_workers))
                except Exception:
                    requested_workers = 0
            cpu_cap = max(1, (os.cpu_count() or 1) - 1)

            # ── GPU-aware session parallelism ──────────────────────────────
            # When GPU optical flow is active, multiple session workers all
            # serialize on a single GPU lock.  Spawning cpu_cap workers is
            # wasteful — most just block waiting for the lock.  We detect the
            # flow backend directly (not just VRAM) and scale parallelism to
            # what the hardware can actually sustain.
            _gpu_info: dict = {}
            _gpu_warnings: list[str] = []
            _gpu_warning_lock = threading.Lock()

            def _collect_gpu_warning(msg: str) -> None:
                with _gpu_warning_lock:
                    _gpu_warnings.append(msg)

            try:
                from abel.utils.gpu_optical_flow import gpu_summary
                _gpu_info = gpu_summary()
            except Exception as _gpu_exc:
                logger.warning("GPU summary probe failed: %s", _gpu_exc)

            gpu_total_mb = _gpu_info.get("total_mb", 0)
            gpu_name = _gpu_info.get("name", "(none)")
            gpu_batch = _gpu_info.get("batch_size", 0)
            gpu_backend = _gpu_info.get("backend", "cpu")

            # The flow backend is the authoritative signal: if it's GPU-based,
            # we MUST cap workers even when VRAM probing fails (total_mb == 0).
            _uses_gpu_flow = gpu_backend in ("torch", "cv2_cuda")

            if not requested_workers and _uses_gpu_flow:
                # Scale session workers to GPU capability.  All GPU flow work
                # serialises through a single lock, so extra session workers
                # mostly just block.  Keeping the count low avoids lock
                # contention, memory pressure, and external-drive I/O storms.
                if gpu_total_mb > 0:
                    if gpu_total_mb <= 2048:
                        gpu_session_cap = 1
                    elif gpu_total_mb <= 4096:
                        gpu_session_cap = 2
                    elif gpu_total_mb <= 8192:
                        gpu_session_cap = min(4, cpu_cap)
                    elif gpu_total_mb <= 12288:
                        gpu_session_cap = min(6, cpu_cap)
                    else:
                        # Even high-end GPUs bottleneck on one lock; cap at 8
                        # concurrent sessions so CPU-side work (frame decode,
                        # MOG2, crop ops) stays productive without flooding
                        # the lock queue.
                        gpu_session_cap = min(8, cpu_cap)
                    worker_source = (
                        f"GPU-adaptive ({gpu_name}, "
                        f"{gpu_total_mb:.0f} MB → {gpu_session_cap} session cap, "
                        f"batch={gpu_batch})"
                    )
                else:
                    # VRAM probing failed but GPU flow backend IS active.
                    # Conservative default: 2 session workers.
                    gpu_session_cap = min(2, cpu_cap)
                    worker_source = (
                        f"GPU-safe fallback ({gpu_name}, VRAM unknown → "
                        f"{gpu_session_cap} session cap)"
                    )
                    logger.warning(
                        "VRAM not available but GPU flow backend '%s' detected; "
                        "capping to %d session workers as safety measure.",
                        gpu_backend, gpu_session_cap,
                    )

                max_workers = min(len(session_jobs), gpu_session_cap)
            elif requested_workers:
                max_workers = min(len(session_jobs), requested_workers)
                worker_source = "environment override"
            else:
                # CPU-only flow — fill available cores.
                max_workers = min(len(session_jobs), cpu_cap)
                worker_source = f"auto (cpu_count-1={cpu_cap})"

            # Divide remaining cores across frame chunks within each session.
            # Ceiling division ensures no core is left idle due to rounding:
            #   e.g. 7 cores / 4 sessions → ceil(7/4)=2 chunks/session → 8 threads.
            intra_session_workers = max(1, -(-cpu_cap // max(1, max_workers)))

            _progress(
                current_step,
                total_steps,
                (
                    "Running session preprocessing in parallel "
                    f"with {max_workers} session worker(s) × {intra_session_workers} chunk worker(s)/session "
                    f"({max_workers * intra_session_workers} total threads across {cpu_cap} available cores). "
                    f"Session worker source: {worker_source}. "
                    "Set ABEL_PREPROCESS_WORKERS to override."
                ),
                f"Preprocessing {len(session_jobs)} session(s) with {max_workers} worker(s)…",
                0.0,
            )

            kp_aliases = self._keypoint_aliases()

            def _process_one_session(job: tuple[int, str, str, Path, Path, float]) -> tuple[int, str, float]:
                job_index, session_id, subject_id, video_path, pose_path, fps = job
                step_started = time.monotonic()
                PoseProcessingService().extract_and_save_frame_pose_features(
                    project_root=self._project_root,
                    pose_path=pose_path,
                    fps=fps,
                    animal_id=subject_id,
                    session_id=session_id,
                    video_id=session_id,
                    keypoint_aliases=kp_aliases,
                )

                def _chunk_progress(chunks_done: int, chunks_total: int, msg: str) -> None:
                    full_msg = f"[{job_index}/{summary.n_sessions}] {session_id} — {msg}"
                    # chunks_done=0 means "chunk starting" — don't update the fraction
                    # or it will reset any completed-chunk progress back to 0.
                    # Only update when chunks_done > 0 (i.e., a chunk just finished).
                    with _chunk_lock:
                        if chunks_done > 0 and chunks_total > 0:
                            _chunk_fracs[session_id] = chunks_done / chunks_total
                        frac_total = sum(_chunk_fracs.values())
                    if progress_cb is not None:
                        elapsed = max(0.0, time.monotonic() - started_at)
                        effective_val = current_step + frac_total
                        remaining_frac = max(0.0, total_steps - effective_val)
                        if _ema_step_seconds and _ema_step_seconds > 0:
                            eta_s = remaining_frac * _ema_step_seconds
                        elif elapsed > 0 and effective_val > 0:
                            eta_s = (elapsed / effective_val) * remaining_frac
                        else:
                            eta_s = 0.0
                        timing = f" | elapsed {_fmt_duration(elapsed)}"
                        if eta_s > 0:
                            eta_local = datetime.now() + timedelta(seconds=eta_s)
                            timing += f" | ETA {_fmt_duration(eta_s)} | finish ~ {eta_local.strftime('%H:%M:%S')}"
                        progress_cb(
                            int(effective_val * _CHUNK_SCALE),
                            int(total_steps * _CHUNK_SCALE),
                            f"{full_msg}{timing}",
                            f"{session_id}: {msg}",
                        )

                if use_video_features:
                    ContextFeatureService().compute_frame_context(
                        project_root=self._project_root,
                        video_path=video_path,
                        pose_path=pose_path,
                        animal_id=subject_id,
                        session_id=session_id,
                        config=ContextFeatureConfig(flow_temporal_stride=int(self._flow_temporal_stride.value())),
                        progress_cb=_chunk_progress,
                        intra_session_workers=intra_session_workers,
                        warning_cb=_collect_gpu_warning,
                        keypoint_aliases=kp_aliases,
                    )
                return job_index, session_id, time.monotonic() - step_started

            with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_job = {
                    executor.submit(_process_one_session, job): job
                    for job in session_jobs
                }
                done_count = 0
                for future in cf.as_completed(future_to_job):
                    _check_cancel()
                    job_index, session_id, duration = future.result()
                    done_count += 1
                    with _chunk_lock:
                        _chunk_fracs.pop(session_id, None)
                    current_step += 1
                    _feat_label = "pose + context features" if use_video_features else "pose features"
                    _progress(
                        current_step,
                        total_steps,
                        f"[{job_index}/{summary.n_sessions}] Processed session {session_id} ({_feat_label}).",
                        f"Processed session {done_count}/{len(session_jobs)}: {session_id}",
                        duration,
                    )
            _preproc_wall = time.monotonic() - _preproc_wall_start
            if _preproc_wall > 0.5:
                step_timings.append(("Session Preprocessing", _preproc_wall))

            # ── Surface GPU warnings to the user ──────────────────────────
            if _gpu_warnings:
                _n_oom = sum(1 for w in _gpu_warnings if "OOM" in w)
                _n_timeout = sum(1 for w in _gpu_warnings if "timed out" in w)
                _parts: list[str] = []
                if _n_oom:
                    _parts.append(
                        f"{_n_oom} GPU out-of-memory event(s) — "
                        "those sub-batches fell back to CPU"
                    )
                if _n_timeout:
                    _parts.append(
                        f"{_n_timeout} GPU lock timeout(s) — "
                        "those sub-batches fell back to CPU"
                    )
                _warn_summary = "; ".join(_parts) or f"{len(_gpu_warnings)} GPU issue(s)"
                _warn_detail = (
                    f"⚠ GPU issues during preprocessing: {_warn_summary}. "
                    "Results are still valid (CPU fallback was used), but "
                    "processing was slower than expected. "
                    "This typically means the GPU ran out of VRAM. "
                    "Close other GPU-intensive applications or reduce "
                    "the number of sessions processed in parallel "
                    "(ABEL_PREPROCESS_WORKERS=1)."
                )
                logger.warning("Active learning preprocessing: %s", _warn_detail)
                _progress(
                    current_step,
                    total_steps,
                    _warn_detail,
                    f"⚠ {_warn_summary}",
                    0.0,
                )

            # Consolidate the per-session parquet files written by the parallel
            # workers into the canonical monolithic files.  This single
            # sequential write replaces the old O(N²) lock-based
            # read-modify-write pattern (each of N parallel workers was
            # reading and rewriting an ever-growing shared file).
            _consol_start = time.monotonic()
            _progress(
                current_step,
                total_steps,
                "Consolidating per-session feature files into merged caches…",
                "Consolidating feature caches…",
                0.0,
            )
            PoseProcessingService.consolidate_session_files(self._project_root)
            if use_video_features:
                ContextFeatureService.consolidate_session_files(self._project_root)
            _consol_dur = time.monotonic() - _consol_start
            if _consol_dur > 0.3:
                step_timings.append(("Feature Cache Consolidation", _consol_dur))

        # Ensure consolidated monolithic files exist even when all sessions
        # hit the per-session cache (session_jobs was empty).  Consolidation
        # is idempotent — if per-session files are absent it returns None.
        _pose_mono = self._project_root / "derived" / "pose_features" / "frame_pose.parquet"
        if not _pose_mono.exists():
            PoseProcessingService.consolidate_session_files(self._project_root)
        if use_video_features:
            _ctx_mono = self._project_root / "derived" / "context_features" / "frame_context.parquet"
            if not _ctx_mono.exists():
                ContextFeatureService.consolidate_session_files(self._project_root)

        frame_pose_path = self._project_root / "derived" / "pose_features" / "frame_pose.parquet"
        frame_ctx_path = (
            self._project_root / "derived" / "context_features" / "frame_context.parquet"
            if use_video_features else None
        )
        _check_cancel()
        step_started = time.monotonic()
        selected_session_ids = {str(s.session_id) for s in linked_sessions}

        def _repr_progress(msg: str) -> None:
            _progress(
                current_step,
                total_steps,
                msg,
                "Building behavior representations…",
                None,
            )

        frame_df, segment_df = self._build_or_reuse_representation(
            frame_pose_path=frame_pose_path,
            frame_ctx_path=frame_ctx_path,
            segment_window=segment_window,
            segment_stride=segment_stride,
            selected_session_ids=selected_session_ids,
            progress_cb=_repr_progress,
        )
        self._persist_segment_settings(segment_window, segment_stride)
        full_segment_df_for_label_fallback = segment_df.copy()
        _apply_time_subset = quick_test or balanced_sampling
        _subset_minutes = quick_ident_minutes if quick_test else balanced_sampling_minutes
        _subset_seed = quick_ident_seed if quick_test else balanced_sampling_seed
        if _apply_time_subset:
            full_frame_df = frame_df
            frame_df, segment_df, quick_subset_info = self._quick_identification_subset(
                full_frame_df,
                full_segment_df_for_label_fallback,
                fps_by_session_id,
                minutes_per_session=_subset_minutes,
                random_seed=_subset_seed,
            )
            _subset_label = "Quick identification" if quick_test else "Balanced time sampling"
            if bool(quick_subset_info.get("applied", False)):
                _progress(
                    current_step,
                    total_steps,
                    (
                        f"{_subset_label} subset applied: "
                        f"frame_rows={len(frame_df)}, segment_rows={len(segment_df)}."
                    ),
                    f"Applying {_subset_label.lower()} subset…",
                    0.0,
                )
            else:
                _progress(
                    current_step,
                    total_steps,
                    (
                        f"{_subset_label} subset skipped: "
                        f"{str(quick_subset_info.get('reason', 'unknown'))}."
                    ),
                    "Using full selected-session range…",
                    0.0,
                )

        summary.n_frame_rows = int(len(frame_df))
        summary.n_segment_rows = int(len(segment_df))
        if segment_df.empty:
            raise ValueError("No segment features generated. Check session duration and window settings.")
        if max_segments > 0 and len(segment_df) > max_segments:
            # Stratified per-session sampling: each session contributes proportionally
            # so the segment cap does not implicitly discard all-but-first subjects.
            _rng_seg = np.random.RandomState(int(quick_ident_seed))
            _session_ids_in_seg = sorted(segment_df["session_id"].astype(str).unique())
            _n_sessions_in_seg = max(1, len(_session_ids_in_seg))
            _per_session = max(1, max_segments // _n_sessions_in_seg)
            _sampled_parts: list[pd.DataFrame] = []
            for _sid in _session_ids_in_seg:
                _sub = segment_df[segment_df["session_id"].astype(str) == _sid]
                if len(_sub) <= _per_session:
                    _sampled_parts.append(_sub)
                else:
                    _sampled_parts.append(
                        _sub.sample(n=_per_session, random_state=_rng_seg.randint(0, 2**31))
                    )
            segment_df = pd.concat(_sampled_parts, ignore_index=True)
            # Trim any over-sampling from flooring arithmetic
            if len(segment_df) > max_segments:
                segment_df = segment_df.sample(
                    n=max_segments, random_state=int(quick_ident_seed)
                ).reset_index(drop=True)
            summary.n_segment_rows = int(len(segment_df))
            _progress(
                current_step,
                total_steps,
                (
                    f"Segment cap applied: stratified sample of {summary.n_segment_rows} segment(s) "
                    f"across {_n_sessions_in_seg} session(s)."
                ),
                "Applying segment limit…",
                0.0,
            )
        current_step += 1
        _progress(
            current_step,
            total_steps,
            "Built frame/segment representations from pose and context features.",
            "Building behavior representations…",
            time.monotonic() - step_started,
        )

        _check_cancel()
        step_started = time.monotonic()
        # Build training set from the FULL (pre-cap) segment grid so that
        # reviewed labels from all frame ranges can be matched — not just the
        # sparse subset selected by the max_segments stratified sample.
        _training_segment_df = full_segment_df_for_label_fallback if not full_segment_df_for_label_fallback.empty else segment_df
        train_df = self._build_training_set(_training_segment_df, target_behavior)
        if train_df.empty and quick_test:
            train_df = self._build_training_set(full_segment_df_for_label_fallback, target_behavior)
            if not train_df.empty:
                _progress(
                    current_step,
                    total_steps,
                    "Quick subset had no labels; reused labels from selected-session full segment set.",
                    "Recovering labels for quick test…",
                    0.0,
                )
        summary.n_train_rows = int(len(train_df))
        if train_df.empty:
            selected_mode = str(self._mode.currentData() or "uncertainty")
            # If no labels are available (or remapping cannot align them), bootstrap with
            # random-absent windows so Active Learning can still proceed without crashing.
            cfg = self._segment_candidate_config(
                mode="random_absent",
                target_behavior_id=None,
                model_version=(
                    "bootstrap_random_absent"
                    if selected_mode == "random_absent"
                    else f"bootstrap_{selected_mode}_no_labels"
                ),
                hard_negative_ratio=float(behavior_cfg.hard_negative_sampling_ratio),
                sample_window_frames=segment_window,
                selected_session_ids=sorted(selected_session_ids),
            )
            rand = self._candidates.generate_random_absent_candidates(cfg)
            if not rand.success:
                raise ValueError(
                    "No labeled training rows found from seeds/reviews, and random bootstrap failed: "
                    + "; ".join(rand.warnings or ["unknown error"])
                )
            self._candidates.save_segment_candidates(rand, cfg)
            summary.model_version = str(cfg.model_version)
            summary.n_candidates = int(rand.n_segments_selected)
            current_step = total_steps
            fallback_note = (
                "No labeled training rows found. Generated random starter windows."
                if selected_mode == "random_absent"
                else (
                    "No labeled training rows found for model training; "
                    f"mode '{selected_mode}' fell back to random starter windows."
                )
            )
            _progress(
                current_step,
                total_steps,
                f"{fallback_note} Generated {summary.n_candidates} window(s).",
                "Generated random starter windows.",
                time.monotonic() - step_started,
            )
            return {
                "summary": summary,
                "target_behavior": target_behavior,
                "metrics": {"f1": float("nan"), "pr_auc": float("nan"), "n_train": 0, "n_val": 0},
                "fusion_diagnostics": {},
                "candidates": list(rand.candidates),
            }
        current_step += 1
        _progress(
            current_step,
            total_steps,
            "Built training set from seed and reviewed labels.",
            "Assembling training labels…",
            time.monotonic() - step_started,
        )

        _check_cancel()
        step_started = time.monotonic()
        self._trainer.merge_and_snapshot_training_set(self._project_root, train_df)
        train_cfg = self._training_config(behavior_cfg, target_behavior)
        gpu_pref_note = self._apply_gpu_training_preference(train_cfg, backend)
        if gpu_pref_note:
            _progress(
                current_step,
                total_steps,
                gpu_pref_note,
                "Preparing GPU-preferred training backend…",
                0.0,
            )
        train_cfg.require_gpu = strict_gpu

        def _training_sub_progress(msg: str) -> None:
            _progress(current_step, total_steps, f"[Training] {msg}", f"Training model: {msg}", None)

        train_result = self._trainer.train(
            self._project_root, train_cfg, session_ids=selected_session_ids,
            progress_cb=_training_sub_progress,
        )
        model_dir = Path(str(train_result["model_dir"]))
        summary.model_version = model_dir.name
        self._save_model_run_settings(
            model_version=summary.model_version,
            target_behavior=target_behavior,
            segment_window=segment_window,
            segment_stride=segment_stride,
            extra={
                "run_type": "full_pipeline",
                "examples_per_session": int(examples_per_session),
                "max_segments": int(max_segments),
            },
        )
        metrics = train_result.get("metrics", {})
        summary.f1 = float(metrics.get("f1", float("nan")))
        summary.pr_auc = float(metrics.get("pr_auc", float("nan")))
        summary.model_device_used = str(train_result.get("model_device_used", metrics.get("model_device_used", "cpu")))
        summary.fallback_reason = str(train_result.get("fallback_reason", metrics.get("fallback_reason", "")))
        current_step += 1
        _progress(
            current_step,
            total_steps,
            (
                f"Trained classifier for target behavior '{target_behavior_label}'. "
                f"F1={summary.f1:.3f}, PR-AUC={summary.pr_auc:.3f}, device={summary.model_device_used}."
            ),
            f"Training model for '{target_behavior_label}'…",
            time.monotonic() - step_started,
        )
        if bool(train_result.get("used_cpu_fallback", False)):
            warning = summary.fallback_reason or "XGBoost CUDA fallback occurred"
            _progress(
                current_step,
                total_steps,
                f"WARNING: Model training fell back to CPU. Reason: {warning}",
                "Model training fell back to CPU",
                0.0,
            )

        _check_cancel()
        step_started = time.monotonic()
        fusion_diag: dict[str, Any] = {}

        def _infer_sub_progress(msg: str) -> None:
            _progress(current_step, total_steps, f"[Inference] {msg}", f"Inference: {msg}", None)

        pred_df = self._infer_with_uncertainty(
            segment_df,
            train_df,
            model_dir,
            behavior_cfg,
            target_behavior,
            use_fusion=use_fusion,
            strict_gpu=strict_gpu,
            fusion_diagnostics=fusion_diag,
            progress_cb=_infer_sub_progress,
        )
        summary.fusion_device_used = str(fusion_diag.get("fusion_device_used", "cpu" if use_fusion else "skipped"))
        if use_fusion and bool(fusion_diag.get("fusion_used_cpu_fallback", False)):
            fuse_reason = str(fusion_diag.get("fusion_fallback_reason", ""))
            if not summary.fallback_reason:
                summary.fallback_reason = fuse_reason
            _progress(
                current_step,
                total_steps,
                f"WARNING: Fusion used CPU fallback ({fuse_reason or 'no additional details'}).",
                "Fusion fell back to CPU",
                0.0,
            )

        out_model_dir = self._project_root / "derived" / "models" / summary.model_version
        out_model_dir.mkdir(parents=True, exist_ok=True)
        pred_df[["segment_id", "prediction_prob"]].to_parquet(out_model_dir / "segment_predictions.parquet", index=False)
        pred_df[["segment_id", "uncertainty_score", "uncertainty_entropy", "prediction_variance", "density_outlier_score"]].to_parquet(
            out_model_dir / "segment_uncertainty.parquet", index=False
        )
        current_step += 1
        _progress(
            current_step,
            total_steps,
            "Completed segment inference and uncertainty scoring.",
            "Scoring segments with uncertainty model…",
            time.monotonic() - step_started,
        )

        _skip_candidates = getattr(self, "_pipeline_all_skip_candidates", False)
        candidates_list: list[Any] = []
        if _skip_candidates:
            summary.n_candidates = 0
            current_step += 1
            _progress(
                current_step,
                total_steps,
                "Skipped candidate generation (batch pipeline mode).",
                "Skipped candidate generation.",
                0.0,
            )
        else:
            cfg = self._segment_candidate_config(
                mode=str(self._mode.currentData() or "uncertainty"),
                target_behavior_id=target_behavior,
                model_version=summary.model_version,
                hard_negative_ratio=float(behavior_cfg.hard_negative_sampling_ratio),
                sample_window_frames=segment_window,
                examples_per_session=examples_per_session,
                selected_session_ids=sorted(selected_session_ids),
            )
            _check_cancel()
            step_started = time.monotonic()
            result = self._candidates.generate_segment_candidates(cfg, segment_df=segment_df)
            if not result.success:
                raise ValueError("Segment candidate generation failed: " + "; ".join(result.warnings))
            self._candidates.save_segment_candidates(result, cfg)
            summary.n_candidates = result.n_segments_selected
            candidates_list = list(result.candidates)
            current_step += 1
            _progress(
                current_step,
                total_steps,
                "Generated candidate segments from uncertainty scores.",
                "Selecting next review candidates…",
                time.monotonic() - step_started,
            )

        _check_cancel()
        step_started = time.monotonic()
        if run_evaluation:
            _progress(
                current_step,
                total_steps,
                "Running evaluation reports (this may take a moment)…",
                "Evaluating predictions and writing reports…",
                0.0,
            )
            # Pass only the 3 identifier columns — frame_df from _repr.build()
            # has all pose features (25+ GB in RAM) and we only need these 3.
            _frame_eval_cols = [c for c in ("frame", "animal_id", "session_id") if c in frame_df.columns]
            self._evaluate_if_possible(
                pred_df,
                frame_df[_frame_eval_cols],
                behavior_cfg,
                target_behavior=target_behavior,
                model_version=summary.model_version,
            )
        else:
            _progress(
                current_step,
                total_steps,
                "Skipped evaluation reports and UMAP plots (skip_evaluation is enabled).",
                "Skipping evaluation + UMAP…",
                0.0,
            )
        current_step += 1
        _progress(
            current_step,
            total_steps,
            "Finished evaluation and report writing.",
            "Evaluation complete.",
            time.monotonic() - step_started,
        )

        phase1_result: dict[str, Any] | None = None
        if run_phase1_after_eval:
            _check_cancel()
            step_started = time.monotonic()

            def _phase1_progress(msg: str) -> None:
                _progress(
                    current_step,
                    total_steps,
                    msg,
                    "Running Phase 1 diagnostics…",
                    None,
                )

            phase1_result = self._phase1.run_phase1(
                project_root=self._project_root,
                target_behavior=target_behavior,
                progress_cb=_phase1_progress,
                force=False,
                session_ids=sorted(selected_session_ids),
            )
            current_step += 1
            _progress(
                current_step,
                total_steps,
                "Completed Phase 1 behavior-adaptive benchmarks.",
                "Completed Phase 1 diagnostics.",
                time.monotonic() - step_started,
            )

        _check_cancel()
        step_started = time.monotonic()
        write_json(
            self._project_root / "derived" / "active_learning" / "last_run_summary.json",
            {
                "sessions": summary.n_sessions,
                "frame_rows": summary.n_frame_rows,
                "segment_rows": summary.n_segment_rows,
                "train_rows": summary.n_train_rows,
                "candidates": summary.n_candidates,
                "model_version": summary.model_version,
                "model_device_used": summary.model_device_used,
                "fusion_device_used": summary.fusion_device_used,
                "fallback_reason": summary.fallback_reason,
                "strict_gpu": bool(strict_gpu),
                "quick_test": bool(quick_test),
                "quick_ident_minutes": int(quick_ident_minutes),
                "quick_ident_seed": int(quick_ident_seed),
                "examples_per_session": int(examples_per_session),
            },
        )
        current_step += 1
        _progress(
            current_step,
            total_steps,
            f"Pipeline finished successfully. model={summary.model_version}, candidates={summary.n_candidates}.",
            "Pipeline complete.",
            time.monotonic() - step_started,
        )
        timing_chart_path = self._write_pipeline_timing_chart(
            step_timings,
            self._project_root / "derived" / "evaluation",
            model_version=summary.model_version,
        )
        # Persist lightweight per-step timing so the next run can seed its
        # ETA estimate immediately without waiting for several steps to elapse.
        _total_wall = max(0.0, time.monotonic() - started_at)
        if total_steps > 0 and _total_wall > 1.0:
            try:
                from abel.storage.file_store import write_json as _write_json
                (_timing_seed_path.parent).mkdir(parents=True, exist_ok=True)
                _write_json(
                    _timing_seed_path,
                    {
                        "total_steps": total_steps,
                        "total_wall_seconds": round(_total_wall, 1),
                        "per_step_seconds": round(_total_wall / total_steps, 2),
                    },
                )
            except Exception:
                pass
        return {
            "summary": summary,
            "target_behavior": target_behavior,
            "candidates": candidates_list,
            "metrics": train_result.get("metrics", {}),
            "model_device_used": summary.model_device_used,
            "fusion_device_used": summary.fusion_device_used,
            "fallback_reason": summary.fallback_reason,
            "fusion_diagnostics": fusion_diag,
            "phase1_result": phase1_result,
            "timing_chart_path": str(timing_chart_path) if timing_chart_path else "",
        }

    def _detect_backend_status(self) -> dict[str, str]:
        model_backend = "CPU"
        fusion_backend = "CPU"

        try:
            from xgboost import XGBClassifier

            x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
            y = np.array([0, 1, 0, 1], dtype=int)
            gpu_probe = XGBClassifier(
                n_estimators=1,
                max_depth=1,
                tree_method="hist",
                device="cuda",
                eval_metric="logloss",
            )
            gpu_probe.fit(x, y)
            model_backend = "GPU"
        except Exception as exc:
            logger.info("XGBoost CUDA probe failed (%s); modeling backend = CPU.", str(exc).splitlines()[0])
            model_backend = "CPU"

        try:
            import torch

            if torch.cuda.is_available():
                _ = torch.zeros(1).to("cuda")
                fusion_backend = "GPU"
            else:
                logger.info("torch.cuda.is_available() = False; fusion backend = CPU.")
                fusion_backend = "CPU"
        except Exception as exc:
            logger.info("Torch CUDA probe failed (%s); fusion backend = CPU.", str(exc).splitlines()[0])
            fusion_backend = "CPU"

        logger.info("Backend detection: modeling=%s, fusion=%s", model_backend, fusion_backend)
        return {"modeling": model_backend, "fusion": fusion_backend}

    def _detect_backend_plan(self, status: dict[str, str] | None = None) -> str:
        if status is None:
            status = self._detect_backend_status()
        cfg = self._load_behavior_cfg()
        train_cfg = self._training_config(cfg, self._selected_target_behavior_id())
        self._apply_gpu_training_preference(train_cfg, status)
        family = str(train_cfg.classifier_family or "").strip().lower() or "unknown"
        params = dict(train_cfg.classifier_params or {})
        device = str(params.get("device", "auto")).strip().lower() or "auto"
        return (
            f"modeling_available={status['modeling']}, fusion_available={status['fusion']}, "
            f"trainer={family}(device={device})"
        )

    @staticmethod
    def _apply_gpu_training_preference(train_cfg: TrainingConfig, backend_status: dict[str, str]) -> str:
        """Promote legacy CPU-leaning configs to a GPU-capable trainer at runtime."""
        if str(backend_status.get("modeling", "CPU")).upper() != "GPU":
            return ""

        family_before = str(train_cfg.classifier_family or "").strip().lower()
        params = dict(train_cfg.classifier_params or {})

        note = ""
        if family_before != "xgboost":
            train_cfg.classifier_family = "xgboost"
            params = {"tree_method": "hist", "device": "cuda"}
            note = (
                f"GPU preference: switched training backend from {family_before or 'default'} "
                "to xgboost(device=cuda)."
            )
        else:
            params.setdefault("tree_method", "hist")
            requested = str(params.get("device", "")).strip().lower()
            if requested in {"", "auto", "cpu"}:
                params["device"] = "cuda"
                note = "GPU preference: set xgboost device=cuda for this run."

        train_cfg.classifier_params = params
        return note

    # ------------------------------------------------------------------
    # On-the-fly feature computation for reviewed segments
    # ------------------------------------------------------------------

    def _enrich_segment_df_for_reviewed_labels(
        self,
        segment_df: pd.DataFrame,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        """Add feature rows for reviewed segments not covered by *segment_df*.

        The segment feature table may be a sparse subsample (due to the
        max_segments cap or a stale cache), so many reviewed labels cannot
        be matched during the inner merge in ``_aggregate_reviewer_labels``.
        This method detects those gaps and computes the missing segment
        features on-the-fly from the cached per-session frame-level pose
        data, using the same z-scoring and summary statistics as the
        representation builder.

        Enriched segments are persisted to
        ``derived/representations/enriched_segments.parquet`` so that
        subsequent runs can skip recomputation for previously enriched
        segment IDs.
        """
        if self._project_root is None:
            return segment_df
        review_path = self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if not review_path.exists():
            return segment_df
        labels_df = pd.read_parquet(review_path)
        if labels_df.empty or "segment_id" not in labels_df.columns:
            return segment_df

        # ── Load enrichment cache ─────────────────────────────────────
        cache_path = self._project_root / "derived" / "representations" / "enriched_segments.parquet"
        cached_enriched_df = pd.DataFrame()
        if cache_path.exists():
            # Invalidate the cache when frame_pose.parquet or frame_context.parquet
            # is newer than the cache — this happens when the user re-runs feature
            # extraction with new settings, ensuring temporal-review clips that were
            # previously only in the enrichment cache get recomputed from fresh data.
            _cache_mtime = cache_path.stat().st_mtime
            _feature_sources = [
                self._project_root / "derived" / "pose_features" / "frame_pose.parquet",
                self._project_root / "derived" / "context_features" / "frame_context.parquet",
            ]
            _cache_stale = any(
                p.exists() and p.stat().st_mtime > _cache_mtime
                for p in _feature_sources
            )
            if _cache_stale:
                logger.info(
                    "Enrichment cache: feature files are newer than cache — invalidating "
                    "so temporal-review clips get fresh features from recomputed data."
                )
            else:
                try:
                    cached_enriched_df = pd.read_parquet(cache_path)
                    logger.info(
                        "Enrichment cache: loaded %d previously enriched segment(s).",
                        len(cached_enriched_df),
                    )
                except Exception:
                    logger.warning("Enrichment cache: failed to read %s; will recompute.", cache_path)
                    cached_enriched_df = pd.DataFrame()

        # Build a set of existing segment_ids for fast exact-match lookup.
        existing_ids = set(segment_df["segment_id"].astype(str)) if not segment_df.empty else set()
        if not cached_enriched_df.empty and "segment_id" in cached_enriched_df.columns:
            existing_ids |= set(cached_enriched_df["segment_id"].astype(str))

        # Identify reviewed labels whose segment_id doesn't exist in the feature table.
        needs_features: dict[str, list[tuple[str, int, int]]] = {}  # session_id -> [(orig_seg_id, start, end)]
        n_already_matched = 0
        for seg_id in labels_df["segment_id"].unique():
            seg_id_str = str(seg_id)
            if seg_id_str in existing_ids:
                n_already_matched += 1
                continue
            parsed = self._parse_segment_id_interval(seg_id_str)
            if parsed is None:
                continue
            session_id, start, end = parsed
            start, end = int(min(start, end)), int(max(start, end))
            needs_features.setdefault(session_id, []).append((seg_id_str, start, end))

        n_needs = sum(len(v) for v in needs_features.values())
        logger.info(
            "Enrichment: %d label(s) already in segment_df/cache, %d need on-the-fly features across %d session(s).",
            n_already_matched, n_needs, len(needs_features),
        )

        if not needs_features:
            # Still merge any previously cached enriched rows into the result.
            if not cached_enriched_df.empty:
                enriched = pd.concat(
                    [segment_df, cached_enriched_df.reindex(columns=segment_df.columns, fill_value=0.0)],
                    ignore_index=True,
                )
                enriched = enriched.drop_duplicates(subset=["segment_id"], keep="first")
                return enriched
            return segment_df

        # Determine raw feature columns from the stat-suffixed segment columns.
        stat_suffixes = ("_mean", "_std", "_median", "_max", "_p10", "_p90", "_energy", "_periodicity")
        raw_feature_cols_set: set[str] = set()
        for c in segment_df.columns:
            for sfx in stat_suffixes:
                if c.endswith(sfx):
                    raw_feature_cols_set.add(c[: -len(sfx)])
                    break
        raw_feature_cols = sorted(raw_feature_cols_set)
        if not raw_feature_cols:
            return segment_df

        from abel.services.behavior_representation_service import BehaviorRepresentationService as _BRS

        pose_sessions_dir = self._project_root / "derived" / "pose_features" / "sessions"
        pose_monolith = self._project_root / "derived" / "pose_features" / "frame_pose.parquet"
        ctx_sessions_dir = self._project_root / "derived" / "context_features" / "sessions"
        ctx_monolith = self._project_root / "derived" / "context_features" / "frame_context.parquet"

        new_rows: list[dict] = []
        n_sessions = len(needs_features)
        for idx, (session_id, windows) in enumerate(sorted(needs_features.items()), 1):
            if progress_cb:
                progress_cb(
                    f"Computing features for {len(windows)} reviewed segment(s) "
                    f"in {session_id} ({idx}/{n_sessions})…"
                )

            # Load frame-level pose data for this session.
            session_file = pose_sessions_dir / f"{session_id}.parquet"
            if session_file.exists():
                frame_df = pd.read_parquet(session_file)
            elif pose_monolith.exists():
                frame_df = pd.read_parquet(pose_monolith)
                frame_df = frame_df[frame_df["session_id"].astype(str) == session_id]
            else:
                continue
            if frame_df.empty:
                continue

            # Merge context features (optical flow, etc.) if available, so
            # the enriched rows have the same feature set as the main build.
            ctx_file = ctx_sessions_dir / f"{session_id}.parquet"
            ctx_df: pd.DataFrame | None = None
            if ctx_file.exists():
                ctx_df = pd.read_parquet(ctx_file)
            elif ctx_monolith.exists():
                try:
                    ctx_df = pd.read_parquet(ctx_monolith)
                    ctx_df = ctx_df[ctx_df["session_id"].astype(str) == session_id]
                except Exception:
                    ctx_df = None
            if ctx_df is not None and not ctx_df.empty:
                join_cols = [c for c in ("frame", "animal_id", "session_id") if c in frame_df.columns and c in ctx_df.columns]
                if join_cols:
                    frame_df = frame_df.merge(ctx_df, on=join_cols, how="inner")

            available_features = [c for c in raw_feature_cols if c in frame_df.columns]
            if not available_features:
                continue

            # Z-score using the same per-(animal, session) strategy as the
            # representation builder so the summary statistics are comparable.
            frame_df = _BRS._zscore_by_group(frame_df, available_features)
            frame_df = frame_df.sort_values("frame").reset_index(drop=True)
            frame_arr = frame_df["frame"].to_numpy(dtype=int)
            animal_id = str(frame_df["animal_id"].iloc[0]) if "animal_id" in frame_df.columns else ""

            for orig_seg_id, start, end in windows:
                mask = (frame_arr >= start) & (frame_arr <= end)
                window_df = frame_df.loc[mask]
                if len(window_df) < 2:
                    continue
                summary = _BRS._segment_summary(window_df, available_features, orig_seg_id)
                summary["session_id"] = session_id
                summary["start_frame"] = start
                summary["end_frame"] = end
                if animal_id:
                    summary["animal_id"] = animal_id
                new_rows.append(summary)

        if not new_rows:
            # No new segments computed, but still merge cached enriched rows.
            if not cached_enriched_df.empty:
                enriched = pd.concat(
                    [segment_df, cached_enriched_df.reindex(columns=segment_df.columns, fill_value=0.0)],
                    ignore_index=True,
                )
                enriched = enriched.drop_duplicates(subset=["segment_id"], keep="first")
                return enriched
            return segment_df

        new_df = pd.DataFrame(new_rows)

        # ── Persist newly computed rows to the enrichment cache ───────
        try:
            if not cached_enriched_df.empty:
                updated_cache = pd.concat([cached_enriched_df, new_df], ignore_index=True)
                updated_cache = updated_cache.drop_duplicates(subset=["segment_id"], keep="last")
            else:
                updated_cache = new_df.copy()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            updated_cache.to_parquet(cache_path, index=False)
            logger.info(
                "Enrichment cache: wrote %d total row(s) to %s (%d newly computed).",
                len(updated_cache), cache_path, len(new_rows),
            )
        except Exception:
            logger.warning("Enrichment cache: failed to write %s; results are still usable in-memory.", cache_path, exc_info=True)

        # Align columns — fill any missing stat columns with 0.
        # Combine: original segment_df + all cached enriched rows + newly computed rows.
        all_enriched = pd.concat([cached_enriched_df, new_df], ignore_index=True) if not cached_enriched_df.empty else new_df
        for c in segment_df.columns:
            if c not in all_enriched.columns:
                all_enriched[c] = 0.0
        enriched = pd.concat(
            [segment_df, all_enriched.reindex(columns=segment_df.columns, fill_value=0.0)],
            ignore_index=True,
        )
        enriched = enriched.drop_duplicates(subset=["segment_id"], keep="first")
        logger.info("Enrichment: added %d on-the-fly feature row(s) for reviewed segments.", len(new_rows))
        if progress_cb:
            progress_cb(f"Added {len(new_rows)} on-the-fly feature row(s) ({len(all_enriched)} total enriched).")
        return enriched

    def _build_training_set(
        self, segment_df: pd.DataFrame, target_behavior: str,
    ) -> pd.DataFrame:
        if self._project_root is None:
            return pd.DataFrame()
        rows: list[pd.DataFrame] = []

        seeds = self._seeds.seeds
        if seeds:
            seed_rows = []
            for seed in seeds:
                mask = (
                    (segment_df["session_id"].astype(str) == str(seed.session_id))
                    & (segment_df["start_frame"] <= int(seed.end_frame))
                    & (segment_df["end_frame"] >= int(seed.start_frame))
                )
                hit = segment_df[mask].copy()
                if hit.empty:
                    continue
                if seed.label_type == "positive":
                    hit["label"] = str(seed.behavior_id or target_behavior)
                else:
                    seed_bid = str(seed.behavior_id or target_behavior)
                    hit["label"] = f"not_{seed_bid}"
                hit["label_source"] = "seed"
                hit["reviewer_confidence"] = 1.0
                seed_rows.append(hit)
            if seed_rows:
                rows.append(pd.concat(seed_rows, ignore_index=True))

        review_path = self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if review_path.exists():
            lbl = pd.read_parquet(review_path)

            # Auto-generate features for reviewed segments that are not in the
            # current segment feature table (e.g. bout-based or random-sampled
            # review clips with different segment ID prefixes).
            if bool(self._auto_generate_reviewed_windows.isChecked()):
                segment_df = self._enrich_segment_df_for_reviewed_labels(segment_df)

            remap_enabled = bool(self._remap_reviewed_windows.isChecked())
            work_lbl = self._remap_review_labels_to_current_windows(lbl, segment_df) if remap_enabled else lbl
            merged = self._aggregate_reviewer_labels(segment_df, work_lbl)
            if merged.empty and not remap_enabled:
                # Automatic fallback keeps reviewed clip labels usable when segment geometry has changed.
                remapped = self._remap_review_labels_to_current_windows(lbl, segment_df)
                merged = self._aggregate_reviewer_labels(segment_df, remapped)
            if not merged.empty:
                rows.append(merged)

        if not rows:
            return pd.DataFrame()

        train = pd.concat(rows, ignore_index=True)
        train = train.drop_duplicates(subset=["segment_id"], keep="last")

        return train

    @staticmethod
    def _parse_segment_id_interval(segment_id: str) -> tuple[str, int, int] | None:
        text = str(segment_id or "").strip()
        if not text:
            return None
        parts = text.split("_")
        if len(parts) < 4:
            return None
        try:
            end = int(parts[-1])
            start = int(parts[-2])
        except ValueError:
            return None

        sid_idx = -1
        for i, token in enumerate(parts):
            if token == "session" and i + 1 < len(parts):
                sid_idx = i
                break
        if sid_idx < 0:
            return None
        sid = "_".join(parts[sid_idx : sid_idx + 2])
        if not sid.startswith("session_"):
            return None
        return sid, int(start), int(end)

    def _remap_review_labels_to_current_windows(
        self,
        labels_df: pd.DataFrame,
        segment_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if labels_df.empty or segment_df.empty:
            return labels_df
        if "segment_id" not in labels_df.columns:
            return labels_df
        req = {"segment_id", "session_id", "start_frame", "end_frame"}
        if not req.issubset(set(segment_df.columns)):
            return labels_df

        seg_len = (segment_df["end_frame"].to_numpy(dtype=int) - segment_df["start_frame"].to_numpy(dtype=int) + 1)
        if len(seg_len) <= 0:
            return labels_df
        window_len = int(np.median(seg_len))
        if window_len <= 0:
            return labels_df

        lookup: dict[tuple[str, int, int], str] = {}
        session_segments: dict[str, list[tuple[int, int, str]]] = {}
        for row in segment_df[["segment_id", "session_id", "start_frame", "end_frame"]].itertuples(index=False):
            sid = str(row.session_id)
            start = int(row.start_frame)
            end = int(row.end_frame)
            seg_id = str(row.segment_id)
            lookup[(sid, start, end)] = seg_id
            session_segments.setdefault(sid, []).append((start, end, seg_id))

        def _best_overlap_segment_id(sid: str, start: int, end: int) -> str | None:
            candidates = session_segments.get(sid, [])
            if not candidates:
                return None
            best_seg: str | None = None
            best_overlap = 0
            best_center_gap: int | None = None
            target_center = (int(start) + int(end)) // 2
            for s0, e0, seg_id in candidates:
                overlap = min(int(end), e0) - max(int(start), s0) + 1
                if overlap <= 0:
                    continue
                center_gap = abs(((s0 + e0) // 2) - target_center)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_center_gap = center_gap
                    best_seg = seg_id
                    continue
                if overlap == best_overlap and best_center_gap is not None and center_gap < best_center_gap:
                    best_center_gap = center_gap
                    best_seg = seg_id
            return best_seg

        remapped_rows: list[dict[str, Any]] = []
        passthrough_rows: list[dict[str, Any]] = []
        mapped_count = 0

        # Labels whose segment_id already exists in the feature table can be
        # kept verbatim — no need to split into sub-windows and re-match.
        existing_seg_ids = set(segment_df["segment_id"].astype(str))

        for item in labels_df.to_dict(orient="records"):
            seg_id_str = str(item.get("segment_id", ""))
            if seg_id_str in existing_seg_ids:
                remapped_rows.append(dict(item))
                mapped_count += 1
                continue

            parsed = self._parse_segment_id_interval(seg_id_str)
            if parsed is None:
                passthrough_rows.append(dict(item))
                continue

            sid, clip_start, clip_end = parsed
            original_start = int(clip_start)
            original_end = int(clip_end)
            clip_start = int(min(original_start, original_end))
            clip_end = int(max(original_start, original_end))
            clip_len = int(clip_end - clip_start + 1)
            n_full = int(clip_len // window_len)
            if n_full <= 0:
                approx_seg = _best_overlap_segment_id(sid, clip_start, clip_end)
                if approx_seg:
                    new_row = dict(item)
                    new_row["segment_id"] = str(approx_seg)
                    remapped_rows.append(new_row)
                    mapped_count += 1
                else:
                    passthrough_rows.append(dict(item))
                continue

            local_mapped = 0
            for idx in range(n_full):
                start = int(clip_start + idx * window_len)
                end = int(start + window_len - 1)
                new_seg_id = lookup.get((sid, start, end))
                if not new_seg_id:
                    new_seg_id = _best_overlap_segment_id(sid, start, end)
                if not new_seg_id:
                    continue
                new_row = dict(item)
                new_row["segment_id"] = str(new_seg_id)
                remapped_rows.append(new_row)
                local_mapped += 1

            if local_mapped <= 0:
                passthrough_rows.append(dict(item))
            else:
                mapped_count += 1

        out_rows = remapped_rows + passthrough_rows
        if not out_rows:
            return labels_df

        if mapped_count > 0:
            logger.info(
                "Remapped reviewer labels to current window size: %d source row(s) expanded/adjusted.",
                mapped_count,
            )
        else:
            logger.info(
                "Remap enabled, but no reviewed clip IDs matched current window geometry; labels were kept as-is.",
            )
        return pd.DataFrame(out_rows)

    @staticmethod
    def _aggregate_reviewer_labels(segment_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
        if labels_df.empty:
            return pd.DataFrame()
        req = {"segment_id", "review_label", "reviewer_id", "confidence"}
        if not req.issubset(set(labels_df.columns)):
            return pd.DataFrame()

        grouped = labels_df.groupby("segment_id", as_index=False).agg(
            review_labels=("review_label", lambda s: [str(v) for v in s if str(v)]),
            reviewer_id=("reviewer_id", lambda s: ",".join(sorted(set(str(v) for v in s if str(v))))),
            confidence=("confidence", "mean"),
        )

        def _resolve(labels: list[str]) -> tuple[str, bool]:
            uniq = sorted(set(labels))
            if not uniq:
                return "ambiguous", False
            informative = [
                x
                for x in uniq
                if x not in {"ambiguous", "boundary_error", "no_behavior"} and not x.startswith("not_")
            ]
            if len(informative) == 1:
                return informative[0], False
            if len(informative) > 1:
                return "ambiguous", True
            neg = [x for x in uniq if x.startswith("not_") or x == "no_behavior"]
            if len(neg) == 1:
                return neg[0], False
            return "ambiguous", False

        resolved = grouped["review_labels"].apply(_resolve)
        grouped["label"] = [x[0] for x in resolved]
        grouped["overlap_allowed"] = [bool(x[1]) for x in resolved]
        grouped = grouped.rename(columns={"reviewer_id": "label_source", "confidence": "reviewer_confidence"})

        # Avoid merge suffix collisions if segment_df already carries label helper columns
        # from prior intermediate joins.
        base = segment_df.drop(
            columns=["label", "label_source", "reviewer_confidence", "overlap_allowed"],
            errors="ignore",
        )

        merged = base.merge(
            grouped[["segment_id", "label", "label_source", "reviewer_confidence", "overlap_allowed"]],
            on="segment_id",
            how="inner",
        )
        return merged

    @staticmethod
    def _build_binary_target_with_overlap_guard(train_df: pd.DataFrame, target_label: str) -> tuple[np.ndarray, np.ndarray]:
        if train_df.empty:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)

        labels = train_df["label"].astype(str)
        sessions = train_df["session_id"].astype(str)
        starts = train_df["start_frame"].to_numpy(dtype=int) if "start_frame" in train_df.columns else np.zeros(len(train_df), dtype=int)
        ends = train_df["end_frame"].to_numpy(dtype=int) if "end_frame" in train_df.columns else starts

        y = np.full(len(train_df), -1, dtype=int)
        pos_mask = labels == target_label
        y[pos_mask.to_numpy()] = 1

        ignorable = {"ambiguous", "boundary_error"}
        neg_mask = (~pos_mask) & (~labels.isin(ignorable))

        # Absence-of-others rule with overlap exception.
        for sid in sorted(set(sessions)):
            s_mask = sessions == sid
            s_idx = np.where(s_mask.to_numpy())[0]
            s_pos = s_idx[pos_mask.to_numpy()[s_idx]]
            if len(s_pos) == 0:
                y[s_idx[neg_mask.to_numpy()[s_idx]]] = 0
                continue

            pos_intervals = [(starts[i], ends[i]) for i in s_pos]
            for i in s_idx:
                if not bool(neg_mask.to_numpy()[i]):
                    continue
                s0, e0 = int(starts[i]), int(ends[i])
                overlaps_target = any((s0 <= pe and e0 >= ps) for ps, pe in pos_intervals)
                if not overlaps_target:
                    y[i] = 0

        keep = y >= 0
        return np.where(keep)[0], y[keep]

    def _infer_with_uncertainty(
        self,
        segment_df: pd.DataFrame,
        train_df: pd.DataFrame,
        model_dir: Path,
        behavior_cfg: BehaviorModelConfig,
        target_behavior: str,
        use_fusion: bool = True,
        strict_gpu: bool = False,
        fusion_diagnostics: dict[str, Any] | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        def _log(msg: str) -> None:
            if progress_cb is not None:
                progress_cb(msg)

        if self._project_root is None:
            return segment_df
        _log("Loading model state…")
        with open(model_dir / "model_state.pkl", "rb") as handle:
            payload = pickle.load(handle)

        clf = payload["model"]
        feature_cols: list[str] = payload["feature_cols"]
        label_map: dict[int, str] = payload["label_map"]

        missing_cols = [col for col in feature_cols if col not in segment_df.columns]
        if missing_cols:
            # A model feature absent from the current segment table is backfilled
            # with 0.0 so inference can proceed — but if a whole feature *family*
            # is missing, those zeros silently degrade the scores. The usual
            # cause is video/context features (optical flow, substrate motion,
            # TMT-zone distances) being absent because the sessions were extracted
            # with "Include video-derived features" disabled.
            _CONTEXT_HINTS = (
                "flow_mag", "flow_directionality", "flow_entropy", "flow_",
                "bedding_motion", "substrate_change", "to_tmt", "_tmt",
            )
            ctx_missing = [
                c for c in missing_cols
                if any(h in c.lower() for h in _CONTEXT_HINTS)
            ]
            logger.warning(
                "Model requested %d feature columns missing from current segment table "
                "(%d look context/video-derived). Backfilling with 0.0. Missing sample: %s",
                len(missing_cols), len(ctx_missing), ", ".join(missing_cols[:8]),
            )
            if ctx_missing and (
                len(ctx_missing) >= 10 or len(ctx_missing) >= 0.5 * len(missing_cols)
            ):
                _log(
                    f"⚠ WARNING: '{target_behavior}' was trained with video/context "
                    f"features (e.g. optical flow, TMT-zone distances) but {len(ctx_missing)} "
                    "such feature(s) are ABSENT for these sessions and were filled with 0. "
                    "Scores for context-dependent behaviours will be UNRELIABLE. Fix: enable "
                    "\"Include video-derived features (optical flow, motion)\" in the Features "
                    "tab and re-run the Active Learning pipeline so context features are "
                    "extracted, then re-run the models."
                )
            elif len(missing_cols) >= max(15, int(0.1 * len(feature_cols))):
                _log(
                    f"⚠ WARNING: {len(missing_cols)} of {len(feature_cols)} feature(s) this "
                    "model needs are missing from the current features and were filled with "
                    "0 — scores may be unreliable. The current features were likely extracted "
                    "with different settings than this model was trained with."
                )
            for col in missing_cols:
                segment_df[col] = 0.0

        _log(f"Running primary inference on {len(segment_df)} segments…")
        x = segment_df[feature_cols].to_numpy(dtype=float)
        probs = clf.predict_proba(x)
        idx_by_label = {label: idx for idx, label in label_map.items()}
        tgt_idx = idx_by_label.get(target_behavior, int(np.argmax(np.bincount(np.argmax(probs, axis=1)))))

        tgt_prob = probs[:, tgt_idx]

        # Temperature scaling: soften overconfident probabilities.
        # logit(p)/T → sigmoid → scaled probability.
        temp = float(self._temperature_scale.value())
        if temp != 1.0 and temp > 0:
            _log(f"Applying temperature scaling (T={temp:.1f})…")
            eps = 1e-7
            clamped = np.clip(tgt_prob, eps, 1.0 - eps)
            logits = np.log(clamped / (1.0 - clamped))
            tgt_prob = 1.0 / (1.0 + np.exp(-logits / temp))

        binary_probs = np.column_stack([1.0 - tgt_prob, tgt_prob])

        missing_train_cols = [col for col in feature_cols if col not in train_df.columns]
        if missing_train_cols:
            logger.warning(
                "Model requested %d feature columns missing from training rows. "
                "Backfilling missing columns with 0.0. Missing sample: %s",
                len(missing_train_cols),
                ", ".join(missing_train_cols[:8]),
            )
            for col in missing_train_cols:
                train_df[col] = 0.0

        y_train = train_df["label"].astype(str).to_numpy()
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        _log("Building ensemble for uncertainty estimation (3 models)…")
        ensemble_probs: list[np.ndarray] = []
        keep_idx, y_bin = self._build_binary_target_with_overlap_guard(train_df, target_label=target_behavior)
        if len(keep_idx) == 0:
            keep_idx = np.arange(len(train_df), dtype=int)
            y_bin = (y_train == target_behavior).astype(int)
        x_train_bin = x_train[keep_idx]

        # Small/quick-test runs can produce a single-class target; XGBoost cannot
        # train in that case, so emit deterministic probabilities instead.
        unique_classes = np.unique(y_bin)
        if unique_classes.size < 2:
            const_pos = float(unique_classes[0]) if unique_classes.size == 1 else 0.5
            const_pos = float(np.clip(const_pos, 0.0, 1.0))
            const_probs = np.full(len(x), const_pos, dtype=float)
            const_2c = np.column_stack([1.0 - const_probs, const_probs])
            ensemble_probs = [const_2c.copy() for _ in (11, 23, 37)]
        else:
            # Diversified ensemble: vary max_depth, subsample, and colsample
            # across members to produce meaningfully different predictions
            # (Lakshminarayanan et al. 2017).
            _ensemble_configs: list[tuple[int, dict[str, Any]]] = [
                (11, {"tree_method": "hist", "max_depth": 4, "subsample": 0.7, "colsample_bytree": 0.7, "learning_rate": 0.05}),
                (23, {"tree_method": "hist", "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.8, "learning_rate": 0.1}),
                (37, {"tree_method": "hist", "max_depth": 8, "subsample": 0.6, "colsample_bytree": 0.6, "learning_rate": 0.15}),
            ]
            for i_ens, (seed, ens_params) in enumerate(_ensemble_configs, 1):
                _log(f"Fitting ensemble model {i_ens}/3 (seed={seed}, depth={ens_params['max_depth']})…")
                est = ActiveLearningTrainerService._make_estimator("xgboost", ens_params, seed)
                try:
                    est.fit(x_train_bin, y_bin)
                except Exception:
                    ens_params_cpu = dict(ens_params)
                    ens_params_cpu["device"] = "cpu"
                    est = ActiveLearningTrainerService._make_estimator(
                        "xgboost",
                        ens_params_cpu,
                        seed,
                    )
                    est.fit(x_train_bin, y_bin)
                p = est.predict_proba(x)[:, 1]
                ensemble_probs.append(np.column_stack([1.0 - p, p]))

        _log("Scoring uncertainty (entropy, variance, density, margin)…")
        weighted = behavior_cfg.uncertainty_weights
        scored = self._uncertainty.score_segments(
            segment_df=segment_df,
            class_probs=binary_probs,
            ensemble_probs=ensemble_probs,
            feature_cols=feature_cols,
            weights=UncertaintyWeights(
                entropy=float(weighted.get("entropy", 0.4)),
                ensemble_variance=float(weighted.get("ensemble_variance", 0.4)),
                density_outlier=float(weighted.get("density_outlier", 0.2)),
                margin=float(weighted.get("margin", 0.0)),
            ),
        )
        scored["prediction_prob"] = tgt_prob

        manifest = self._imports.load_manifest(self._project_root)
        if manifest is not None and use_fusion:
            _log("Running fusion inference on uncertain segments…")
            video_lookup = {}
            subject_lookup = {}
            for linked in manifest.linked_sessions:
                path = self._imports.video_path_for_session(manifest, linked.session_id)
                if path:
                    video_lookup[linked.session_id] = path
                    subject_lookup[linked.session_id] = linked.subject_id or None
            roi_lookup = {
                sid: self._rois.resolve_subject_crop_roi(self._project_root, subject_lookup.get(sid))
                for sid in video_lookup
            }
            try:
                fused = self._fusion.fuse_uncertain_segments(
                    segments=scored,
                    video_lookup=video_lookup,
                    roi_lookup=roi_lookup,
                    config=FusionConfig(uncertainty_threshold=float(behavior_cfg.fusion_threshold)),
                    diagnostics=fusion_diagnostics,
                )
            except Exception as exc:
                if strict_gpu:
                    raise RuntimeError(
                        "Strict GPU mode is enabled, but fusion failed and fallback is disabled. "
                        f"Reason: {exc}. Disable 'Require GPU (fail if fallback occurs)' to allow CPU fallback."
                    ) from exc
                logger.warning("Fusion inference skipped due to runtime error: %s", exc)
                if fusion_diagnostics is not None:
                    fusion_diagnostics.setdefault("fusion_device_used", "skipped")
                    fusion_diagnostics["fusion_used_cpu_fallback"] = True
                    fusion_diagnostics["fusion_fallback_reason"] = str(exc)
                fused = scored
            if strict_gpu and fusion_diagnostics is not None and fusion_diagnostics.get("fusion_device_used") != "gpu":
                reason = str(fusion_diagnostics.get("fusion_fallback_reason", ""))
                raise RuntimeError(
                    "Strict GPU mode is enabled, but fusion did not execute on GPU. "
                    f"Reason: {reason or 'no additional details'}. Disable 'Require GPU (fail if fallback occurs)' to allow CPU fallback."
                )
            if "prediction_prob_fused" in fused.columns:
                fused["prediction_prob"] = fused["prediction_prob_fused"]
            scored = fused

        repr_path = self._project_root / "derived" / "representations" / "segment_features.parquet"
        scored.to_parquet(repr_path, index=False)
        return scored

    def _evaluate_if_possible(
        self,
        pred_df: pd.DataFrame,
        frame_df: pd.DataFrame,
        behavior_cfg: BehaviorModelConfig,
        target_behavior: str,
        model_version: str,
    ) -> None:
        if self._project_root is None:
            return
        label_path = self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if not label_path.exists():
            return

        labels = pd.read_parquet(label_path)
        if labels.empty:
            return

        merged = pred_df.merge(labels[["segment_id", "review_label"]], on="segment_id", how="inner")
        if merged.empty:
            return

        merged["label_true"] = (merged["review_label"].astype(str) == target_behavior).astype(int)
        merged["label_pred"] = (merged["prediction_prob"].astype(float) >= 0.5).astype(int)
        if bool(self._all_behavior_aware.isChecked()):
            raw_umap_labels = merged["review_label"].astype(str).str.strip()
            behavior_name_by_id = {
                str(b.behavior_id).strip(): str(b.name).strip()
                for b in self._behaviors.behaviors
                if str(b.behavior_id).strip()
            }
            merged["umap_label"] = raw_umap_labels.where(
                ~raw_umap_labels.isin({"", "ambiguous", "boundary_error"})
                & ~raw_umap_labels.str.startswith("not_"),
                "Other",
            )
            merged.loc[raw_umap_labels == NO_BEHAVIOR_ID, "umap_label"] = "Other"
            merged["umap_label"] = merged["umap_label"].map(lambda token: behavior_name_by_id.get(str(token), str(token)))
        else:
            target_name = next(
                (
                    str(b.name).strip()
                    for b in self._behaviors.behaviors
                    if str(b.behavior_id).strip() == str(target_behavior).strip()
                ),
                "Target",
            )
            merged["umap_label"] = np.where(merged["label_true"].astype(int) == 1, target_name, "Other")

        # Frame-level proxy: assign each frame the containing segment's labels.
        # Vectorized interval join using numpy searchsorted per session.
        # Replaces O(N_segments * N_frames) Python loop with
        # O(N_sessions * N_frames/session * log(N_segs/session)), which avoids
        # allocating one boolean mask per labeled segment and calling astype(str)
        # on the full session_id column every iteration.
        frame_labels = frame_df[["frame", "animal_id", "session_id"]].copy()
        frame_labels = frame_labels.reset_index(drop=True)
        _fl_session = frame_labels["session_id"].astype(str).to_numpy()
        _fl_frame = frame_labels["frame"].to_numpy(dtype=np.int64)
        _lt_arr = np.zeros(len(frame_labels), dtype=np.int8)
        _lp_arr = np.zeros(len(frame_labels), dtype=np.int8)
        _pp_arr = np.zeros(len(frame_labels), dtype=np.float32)
        if not merged.empty:
            _seg_session = merged["session_id"].astype(str).to_numpy()
            _seg_starts = merged["start_frame"].to_numpy(dtype=np.int64)
            _seg_ends = merged["end_frame"].to_numpy(dtype=np.int64)
            _seg_lt = merged["label_true"].to_numpy(dtype=np.int8)
            _seg_lp = merged["label_pred"].to_numpy(dtype=np.int8)
            _seg_pp = merged["prediction_prob"].to_numpy(dtype=np.float32)
            for _sid in np.unique(_seg_session):
                _seg_mask = _seg_session == _sid
                _frame_mask = _fl_session == _sid
                if not np.any(_frame_mask):
                    continue
                _f = _fl_frame[_frame_mask]
                _ss = _seg_starts[_seg_mask]
                _se = _seg_ends[_seg_mask]
                _slt = _seg_lt[_seg_mask]
                _slp = _seg_lp[_seg_mask]
                _spp = _seg_pp[_seg_mask]
                _order = np.argsort(_ss, kind="stable")
                _ss = _ss[_order]; _se = _se[_order]
                _slt = _slt[_order]; _slp = _slp[_order]; _spp = _spp[_order]
                _ci = np.searchsorted(_ss, _f, side="right") - 1
                _valid = _ci >= 0
                _ci_v = _ci[_valid]
                _within = _f[_valid] <= _se[_ci_v]
                _fi = np.where(_frame_mask)[0][_valid][_within]
                _si = _ci_v[_within]
                _lt_arr[_fi] = _slt[_si]
                _lp_arr[_fi] = _slp[_si]
                _pp_arr[_fi] = _spp[_si]
        frame_labels["label_true"] = _lt_arr.astype(int)
        frame_labels["label_pred"] = _lp_arr.astype(int)
        frame_labels["prediction_prob"] = _pp_arr.astype(float)
        del _fl_session, _fl_frame, _lt_arr, _lp_arr, _pp_arr

        dist_col = next(
            (
                c
                for c in [
                    "nose_to_TMT_dist_mean",
                    "forepaw_centroid_to_TMT_dist_mean",
                    "body_centroid_to_TMT_dist_mean",
                    "nose_to_target_dist_mean",
                    "forepaw_centroid_to_target_dist_mean",
                    "body_centroid_to_target_dist_mean",
                ]
                if c in merged.columns
            ),
            None,
        )
        pos_cols = ["animal_id", "session_id", "start_frame", "end_frame"]
        if dist_col:
            pos_cols.append(dist_col)
        pos = merged[merged["label_pred"] == 1][pos_cols].copy()
        if dist_col:
            pos = pos.rename(columns={dist_col: "distance_to_TMT_mean"})

        fps = 30.0
        merge_cfg = BoutMergeConfig(
            max_gap_frames=int(behavior_cfg.bout_merge_gap),
            min_bout_duration=int(behavior_cfg.min_bout_duration),
        )
        self._evaluation.evaluate_and_save(
            project_root=self._project_root,
            frame_labels=frame_labels,
            segment_labels=merged[["label_true", "label_pred", "prediction_prob", "umap_label"]],
            positive_segments=pos,
            fps=fps,
            merge_config=merge_cfg,
            behavior_id=target_behavior,
            model_version=model_version,
            feature_version="representation_v1",
            config={
                "bout_merge_gap": int(behavior_cfg.bout_merge_gap),
                "min_bout_duration": int(behavior_cfg.min_bout_duration),
                "distance_metric": dist_col or "none",
                "target_behavior": target_behavior,
            },
        )

    def _load_behavior_cfg(self) -> BehaviorModelConfig:
        if self._project_root is None:
            return BehaviorModelConfig()
        raw = read_yaml(self._project_root / "project.yaml", {})
        model = raw.get("behavior_model") or {}
        # The Features tab's "Include video-derived features" checkbox is the
        # single source of truth for context extraction; it is persisted under
        # the `feature_extraction` block. `behavior_model.use_video_features`
        # only syncs after a pose-extraction run, so honour the checkbox value
        # directly here — otherwise toggling it on and running the pipeline
        # would silently skip context features.
        fx = raw.get("feature_extraction") or {}
        if "use_video_features" in fx:
            model["use_video_features"] = bool(fx["use_video_features"])
        model["active_learning_query_size"] = int(self._query_size.value())
        model["query_strategy"] = str(self._mode.currentData() or model.get("query_strategy", "uncertainty"))
        model["evaluation_split_strategy"] = str(
            self._split_strategy.currentData() or model.get("evaluation_split_strategy", "group_shuffle_session")
        )
        return BehaviorModelConfig.model_validate(model)

    def _persist_segment_settings(self, window_frames: int, stride_frames: int) -> None:
        if self._project_root is None:
            return
        path = self._project_root / "project.yaml"
        raw = read_yaml(path, {})
        model = dict(raw.get("behavior_model") or {})
        model["segment_window_frames"] = int(max(8, window_frames))
        model["segment_stride_frames"] = int(max(1, stride_frames))
        raw["behavior_model"] = model
        write_yaml(path, raw)

    def _training_config(self, cfg: BehaviorModelConfig, target_behavior: str) -> TrainingConfig:
        return TrainingConfig(
            classifier_family=cfg.classifier_type,
            classifier_params=cfg.classifier_params,
            calibration_method=cfg.calibration_method,
            split_strategy=str(self._split_strategy.currentData() or cfg.evaluation_split_strategy),
            test_size=float(self._validation_pct.value()) / 100.0,
            target_label=target_behavior,
            model_version=self._resolved_model_version(target_behavior),
            feature_version="representation_v1",
            max_train_samples_per_class=max(0, int(self._max_train_samples_per_class.value())),
            no_behavior_sample_weight=float(self._no_behavior_sample_weight.value()),
            allow_co_occurring_behaviors=bool(cfg.allow_co_occurring_behaviors),
            include_imported=bool(self._include_imported.isChecked()),
        )

    def _segment_candidate_config(
        self,
        *,
        mode: str,
        target_behavior_id: str | None,
        model_version: str,
        hard_negative_ratio: float,
        sample_window_frames: int = 60,
        examples_per_session: int | None = None,
        selected_session_ids: list[str] | None = None,
    ) -> SegmentCandidateGenerationConfig:
        resolved_examples_per_session = int(self._examples_per_session.value()) if examples_per_session is None else int(examples_per_session)
        query_size = self._resolved_query_size_limit()
        # Honour query_size as the hard upper limit.  If the per-session
        # distribution (examples_per_session × n_sessions) exceeds it, reduce
        # examples_per_session so the total fits within query_size rather
        # than silently expanding to a much larger number.
        if resolved_examples_per_session > 0 and query_size > 0:
            n_sessions = max(1, len(selected_session_ids or []))
            distributed_total = resolved_examples_per_session * n_sessions
            if distributed_total > query_size:
                capped_eps = max(1, query_size // n_sessions)
                logger.info(
                    "Per-session distribution (%d × %d = %d) exceeds query_size (%d). "
                    "Capping examples_per_session to %d.",
                    resolved_examples_per_session, n_sessions, distributed_total,
                    query_size, capped_eps,
                )
                resolved_examples_per_session = capped_eps
            effective_top_k = query_size
        else:
            effective_top_k = query_size
        return SegmentCandidateGenerationConfig(
            top_k=effective_top_k,
            mode=str(mode),
            target_behavior_id=target_behavior_id,
            model_version=str(model_version),
            feature_version="representation_v1",
            hard_negative_ratio=float(hard_negative_ratio),
            query_size=query_size,
            sample_window_frames=int(sample_window_frames),
            examples_per_session=max(0, resolved_examples_per_session),
            selected_session_ids=sorted(str(s) for s in (selected_session_ids or []) if str(s).strip()),
            all_behavior_aware=bool(self._all_behavior_aware.isChecked()),
            all_behavior_competition_margin=float(self._all_behavior_competition_margin.value()),
            allow_co_occurring_behaviors=bool(getattr(self, '_co_occurring_enabled', False)),
            enable_weighted_queue_scoring=bool(self._queue_weighted_enable.isChecked()),
            enable_uncertainty_sampling=True,
            enable_expert_disagreement=bool(self._queue_enable_disagreement.isChecked()),
            enable_diversity_sampling=bool(self._queue_enable_diversity.isChecked()),
            diversity_mode=str(self._queue_diversity_mode.currentData() or "distance_to_reviewed"),
            enable_confound_sampling=bool(self._queue_enable_confound.isChecked()),
            enable_hard_negative_mining=bool(self._queue_enable_hardneg.isChecked()),
            exploration_fraction=float(self._queue_exploration_fraction.value()),
            **self._resolved_queue_weights(),
        )

    def _resolved_queue_weights(self) -> dict[str, float]:
        """Return queue_weight_* kwargs derived from the candidate focus slider."""
        w = getattr(self, "_focus_queue_weights", None)
        if not w:
            # No focus adjustment yet — use dataclass defaults.
            return {}
        return {
            "queue_weight_candidate":     w["candidate"],
            "queue_weight_uncertainty":   w["uncertainty"],
            "queue_weight_disagreement":  w["disagreement"],
            "queue_weight_diversity":     w["diversity"],
            "queue_weight_confound":      w["confound"],
            "queue_weight_hard_negative": w["hard_negative"],
            "queue_weight_exploration":   w["exploration"],
        }

    @staticmethod
    def _quick_identification_subset(
        frame_df: pd.DataFrame,
        segment_df: pd.DataFrame,
        session_fps: dict[str, float],
        minutes_per_session: int,
        random_seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        if frame_df.empty or segment_df.empty:
            return frame_df, segment_df, {"applied": False, "reason": "empty inputs"}

        work_frame = frame_df.copy()
        work_seg = segment_df.copy()
        work_frame["session_id"] = work_frame["session_id"].astype(str)
        work_seg["session_id"] = work_seg["session_id"].astype(str)

        rng = np.random.default_rng(int(random_seed))
        minutes = max(1, int(minutes_per_session))
        sampled_frames: list[pd.DataFrame] = []
        sampled_segments: list[pd.DataFrame] = []
        sampled_info: list[dict[str, Any]] = []

        for sid, grp in work_frame.groupby("session_id", sort=False):
            fps = max(1.0, float(session_fps.get(str(sid), 30.0)))
            target_len = max(1, int(round(minutes * 60.0 * fps)))
            min_frame = int(grp["frame"].min())
            max_frame = int(grp["frame"].max())
            available_len = max(0, max_frame - min_frame + 1)

            if available_len <= target_len:
                start_f = min_frame
                end_f = max_frame
            else:
                latest_start = max_frame - target_len + 1
                start_f = int(rng.integers(min_frame, latest_start + 1))
                end_f = int(start_f + target_len - 1)

            frame_mask = (grp["frame"] >= start_f) & (grp["frame"] <= end_f)
            grp_sample = grp.loc[frame_mask].copy()
            if grp_sample.empty:
                continue
            sampled_frames.append(grp_sample)

            seg_grp = work_seg[work_seg["session_id"] == str(sid)]
            seg_mask = (seg_grp["start_frame"] <= end_f) & (seg_grp["end_frame"] >= start_f)
            seg_sample = seg_grp.loc[seg_mask].copy()
            if not seg_sample.empty:
                sampled_segments.append(seg_sample)

            sampled_info.append(
                {
                    "session_id": str(sid),
                    "fps": float(fps),
                    "start_frame": int(start_f),
                    "end_frame": int(end_f),
                    "frame_rows": int(len(grp_sample)),
                    "segment_rows": int(len(seg_sample)),
                }
            )

        if not sampled_frames:
            return frame_df, segment_df, {"applied": False, "reason": "no sampled frame rows"}

        out_frame = pd.concat(sampled_frames, ignore_index=True)
        out_segment = (
            pd.concat(sampled_segments, ignore_index=True)
            if sampled_segments
            else segment_df.iloc[0:0].copy()
        )

        if out_segment.empty:
            return frame_df, segment_df, {"applied": False, "reason": "no sampled segment rows"}

        return out_frame, out_segment, {
            "applied": True,
            "minutes_per_session": int(minutes),
            "seed": int(random_seed),
            "session_windows": sampled_info,
        }

    @staticmethod
    def _quality_explanation(metrics: dict[str, Any]) -> str:
        f1 = float(metrics.get("f1", float("nan")))
        pr_auc = float(metrics.get("pr_auc", float("nan")))
        if np.isnan(f1) or np.isnan(pr_auc):
            return "Quality: unavailable (insufficient validation labels)."

        def _band(score: float, strong: float, good: float, fair: float) -> str:
            if score >= strong:
                return "strong"
            if score >= good:
                return "good"
            if score >= fair:
                return "fair"
            return "limited"

        rank_band = _band(pr_auc, strong=0.90, good=0.80, fair=0.70)
        threshold_band = _band(f1, strong=0.85, good=0.72, fair=0.60)

        diff_score = float("nan")
        diff_detail = "differentiation details are limited"
        cm_raw = metrics.get("confusion_matrix")
        try:
            cm = np.asarray(cm_raw, dtype=float)
            if cm.ndim == 2 and cm.shape[0] == cm.shape[1] and cm.sum() > 0:
                if cm.shape == (2, 2):
                    tn, fp = float(cm[0, 0]), float(cm[0, 1])
                    fn, tp = float(cm[1, 0]), float(cm[1, 1])
                    target_recall = tp / max(1.0, tp + fn)
                    non_target_specificity = tn / max(1.0, tn + fp)
                    diff_score = 0.5 * (target_recall + non_target_specificity)
                    diff_detail = (
                        "target differentiation is "
                        f"{_band(diff_score, strong=0.88, good=0.78, fair=0.68)} "
                        f"(target recall {target_recall:.2f}, non-target specificity {non_target_specificity:.2f})"
                    )
                else:
                    diag = float(np.trace(cm))
                    total = float(cm.sum())
                    diag_rate = diag / max(1.0, total)
                    diff_score = diag_rate
                    diff_detail = (
                        "class differentiation is "
                        f"{_band(diag_rate, strong=0.88, good=0.78, fair=0.68)} "
                        f"(diagonal agreement {diag_rate:.2f})"
                    )
        except Exception:
            pass

        if rank_band in {"strong", "good"} and (np.isnan(diff_score) or diff_score >= 0.75):
            quality_prefix = "Quality: promising"
        elif rank_band == "limited" and threshold_band == "limited":
            quality_prefix = "Quality: weak"
        else:
            quality_prefix = "Quality: mixed"

        n_val = int(metrics.get("n_val", 0) or 0)
        if threshold_band in {"limited", "fair"}:
            improvement = "Add more diverse reviewed clips, especially boundary and hard-negative windows, to stabilize threshold decisions."
        else:
            improvement = "A modest batch of additional diverse clips should further improve consistency across sessions."
        if n_val < 300:
            improvement = (
                "Validation coverage is still thin; add more diverse reviewed clips across sessions to improve confidence in the score."
            )

        return (
            f"{quality_prefix}. Ranking signal is {rank_band} (PR-AUC={pr_auc:.3f}) and operating-threshold performance is "
            f"{threshold_band} (F1={f1:.3f}); {diff_detail}. {improvement}"
        )

    def _set_busy(self, busy: bool) -> None:
        self._run_btn.setEnabled(not busy)
        self._run_pipeline_all_btn.setEnabled(not busy)
        self._retrain_btn.setEnabled(not busy)
        self._retrain_all_btn.setEnabled(not busy)
        self._run_existing_btn.setEnabled(not busy)
        self._run_models_btn.setEnabled(not busy)
        self._phase1_run_btn.setEnabled(not busy)
        self._confound_graph_btn.setEnabled(not busy)
        self._unified_umap_btn.setEnabled(not busy)
        self._unsupervised_umap_btn.setEnabled(not busy)
        self._umap_select_btn.setEnabled(not busy)
        self._awareness_ablation_btn.setEnabled(not busy)
        self._viz_menu_btn.setEnabled(not busy)
        self._stop_btn.setEnabled(busy)
        self._progress.setVisible(busy)

    def _confirm_stop(self) -> None:
        if not self._stop_btn.isEnabled():
            return
        answer = QMessageBox.question(
            self,
            "Stop Active Learning",
            "Stop the active-learning run after the current step finishes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._cancel_flag[0] = True
        self._status.setText("Stop requested. Waiting for current step to finish…")
        self._append_log("Stop requested by user. Cancelling after current step.")

    def _append_log(self, message: str) -> None:
        self._log.append(message)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Workflow snapshot
    # ------------------------------------------------------------------

    def _save_workflow_snapshot(self, *, model_version: str, target_behavior: str) -> None:
        """Write a workflow snapshot for the just-completed model run.

        Call this after every successful full pipeline or retrain so that
        ``Batch Run`` always reflects the current state without any manual step.
        """
        if self._project_root is None or not model_version:
            return
        behavior_cfg = self._load_behavior_cfg()
        segment_window, segment_stride, _ = self._resolved_segment_settings(behavior_cfg)
        # Capture context feature config params from the ContextFeatureConfig defaults.
        # (Advanced users who changed these via direct config edits will have them
        # persisted to project.yaml; for now we read the live dataclass defaults.)
        ctx_cfg = ContextFeatureConfig(flow_temporal_stride=int(self._flow_temporal_stride.value()))
        ctx_dict = {
            "farneback_pyr_scale": ctx_cfg.farneback_pyr_scale,
            "farneback_levels": ctx_cfg.farneback_levels,
            "farneback_winsize": ctx_cfg.farneback_winsize,
            "farneback_iterations": ctx_cfg.farneback_iterations,
            "farneback_poly_n": ctx_cfg.farneback_poly_n,
            "farneback_poly_sigma": ctx_cfg.farneback_poly_sigma,
            "downsample_factor": ctx_cfg.downsample_factor,
            "flow_temporal_stride": ctx_cfg.flow_temporal_stride,
        }
        # Resolve fps from project config.
        project_yaml = read_yaml(self._project_root / "project.yaml", {})
        fps = float(project_yaml.get("default_fps") or 30.0)
        # Resolve pose preset from pose_features config.
        pose_preset_id: str | None = None
        pose_cfg_path = self._project_root / "config" / "pose_features.yaml"
        if pose_cfg_path.exists():
            pose_raw = read_yaml(pose_cfg_path, {})
            pose_preset_id = str(pose_raw.get("active_preset_id") or "") or None

        # Capture temporal refinement settings so batch run can offer temporal precision.
        tr_settings_path = self._project_root / "config" / "temporal_refinement_settings.json"
        tr_settings: dict[str, Any] = {}
        if tr_settings_path.exists():
            try:
                tr_settings = dict(read_json(tr_settings_path, {}))
            except Exception:
                pass

        snapshot = WorkflowSnapshot(
            model_version=model_version,
            target_behavior=target_behavior,
            segment_window_frames=segment_window,
            segment_stride_frames=segment_stride,
            excluded_feature_cols=sorted(self._excluded_feature_cols),
            fps=fps,
            context_feature_config=ctx_dict,
            pose_preset_id=pose_preset_id,
            run_settings=self._ui_settings_payload(),
            temporal_refinement_settings=tr_settings,
        )
        try:
            self._snapshot_svc.save(self._project_root, snapshot)
        except Exception as exc:
            logger.warning("Failed to save workflow snapshot: %s", exc)

    # ------------------------------------------------------------------
    # Batch run dialog + task
    # ------------------------------------------------------------------

    def _show_batch_run_dialog(self) -> None:
        """Dialog to run the current workflow on a new set of video files.

        The user selects video + pose file pairs, chooses an output directory,
        and optionally overrides per-run settings.  Everything is pre-populated
        from the live workflow snapshot.
        """
        from PySide6.QtWidgets import QFileDialog, QListWidget, QListWidgetItem  # type: ignore[attr-defined]

        if self._project_root is None:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return

        snapshot = self._snapshot_svc.load(self._project_root)
        valid, reason = (True, "") if snapshot else (False, "No workflow snapshot found.")
        if snapshot:
            valid, reason = self._snapshot_svc.is_valid(self._project_root, snapshot)

        if not valid:
            # If there is no snapshot at all, offer to create one from current UI state.
            answer = QMessageBox.question(
                self,
                "No Workflow Saved",
                f"{reason}\n\n"
                "A workflow is saved automatically every time you complete a pipeline\n"
                "or retrain run.  Run the active-learning pipeline first, then use\n"
                "Batch Run to apply that model to new videos.\n\n"
                "Would you like to create a snapshot from the currently selected model\n"
                "right now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            chosen_model = str(self._saved_model_combo.currentData() or "").strip()
            if not chosen_model:
                QMessageBox.warning(
                    self,
                    "No Model Selected",
                    "Select a saved model in the 'Saved model' dropdown first.",
                )
                return
            target = self._selected_target_behavior_id()
            self._save_workflow_snapshot(model_version=chosen_model, target_behavior=target)
            snapshot = self._snapshot_svc.load(self._project_root)
            if not snapshot:
                QMessageBox.critical(self, "Error", "Failed to create snapshot.")
                return

        # ── Dialog layout ────────────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Batch Run — Apply Workflow to New Videos")
        dlg.setMinimumWidth(680)
        dlg.setMinimumHeight(560)
        root_layout = QVBoxLayout(dlg)

        # Snapshot summary banner
        banner = QLabel()
        banner.setWordWrap(True)
        banner.setStyleSheet("background:#e8f4e8; border:1px solid #88bb88; padding:6px; border-radius:3px;")
        banner.setText(
            f"<b>Current workflow:</b> model <code>{snapshot.model_version}</code> | "
            f"behavior <code>{snapshot.target_behavior}</code> | "
            f"window {snapshot.segment_window_frames} frames | "
            f"saved {str(snapshot.created_at or '—')[:19]}"
        )
        root_layout.addWidget(banner)

        # Session list
        root_layout.addWidget(QLabel("<b>Sessions to process</b> (each row = one video + one pose file):"))
        session_list = QListWidget()
        session_list.setMinimumHeight(160)
        session_list.setToolTip(
            "Each item represents one linked video + pose pair.\n"
            "Use the Add / Remove buttons to build the list."
        )
        root_layout.addWidget(session_list)

        # session_data: list of (video_path, pose_path, animal_id)
        session_data: list[dict[str, str]] = []

        def _refresh_list() -> None:
            session_list.clear()
            for entry in session_data:
                vid = Path(entry.get("video_path", "")).name
                pose = Path(entry.get("pose_path", "")).name
                animal = str(entry.get("animal_id", "")).strip() or "(auto)"
                session_list.addItem(f"{vid}  +  {pose}  [{animal}]")

        def _add_session() -> None:
            vid_path, _ = QFileDialog.getOpenFileName(
                dlg, "Select video file", str(self._project_root),
                "Video files (*.mp4 *.avi *.mkv *.mov *.wmv *.m4v);;All files (*)",
            )
            if not vid_path:
                return
            pose_path, _ = QFileDialog.getOpenFileName(
                dlg, "Select pose/tracking file for this video",
                str(Path(vid_path).parent),
                "Tracking files (*.csv *.h5 *.hdf5);;All files (*)",
            )
            if not pose_path:
                return
            # Derive animal_id from filename stem before common DLC suffixes.
            stem = Path(vid_path).stem
            for token in ("DLC", "dlc", "_tracked", "_pose"):
                stem = stem.split(token)[0]
            animal_id = stem.strip("_-") or "animal_01"
            session_data.append({"video_path": vid_path, "pose_path": pose_path, "animal_id": animal_id})
            _refresh_list()

        def _remove_session() -> None:
            rows = sorted({i.row() for i in session_list.selectedItems()}, reverse=True)
            for r in rows:
                session_data.pop(r)
            _refresh_list()

        list_btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Session…")
        add_btn.clicked.connect(_add_session)
        rem_btn = QPushButton("Remove Selected")
        rem_btn.clicked.connect(_remove_session)
        list_btn_row.addWidget(add_btn)
        list_btn_row.addWidget(rem_btn)
        list_btn_row.addStretch()
        root_layout.addLayout(list_btn_row)

        # Output directory
        out_row = QHBoxLayout()
        out_edit = QLineEdit()
        default_out = str(self._project_root / "derived" / "batch_results")
        out_edit.setText(default_out)
        out_btn = QPushButton("Browse…")

        def _pick_out() -> None:
            d = QFileDialog.getExistingDirectory(dlg, "Select output folder", out_edit.text())
            if d:
                out_edit.setText(d)

        out_btn.clicked.connect(_pick_out)
        out_row.addWidget(QLabel("Output folder:"))
        out_row.addWidget(out_edit, 1)
        out_row.addWidget(out_btn)
        root_layout.addLayout(out_row)

        # Per-run options
        opts = QGroupBox("Output options")
        opts_form = QFormLayout(opts)
        chk_csv = QCheckBox()
        chk_csv.setChecked(snapshot.export_csv)
        chk_xlsx = QCheckBox()
        chk_xlsx.setChecked(snapshot.export_xlsx)
        chk_video = QCheckBox()
        chk_video.setChecked(snapshot.export_labeled_video)
        chk_video.setToolTip("Write labeled overlay video (slow — requires OpenCV).")
        fps_spin = QDoubleSpinBox()
        fps_spin.setRange(1.0, 1000.0)
        fps_spin.setDecimals(1)
        fps_spin.setValue(snapshot.fps or 30.0)
        fps_spin.setToolTip("Override frames per second (only needed if project default is wrong for these files).")
        # Temporal precision is available if the model pkl exists for the snapshot's target.
        _temporal_model_pkl = (
            self._project_root / "derived" / "models" / snapshot.model_version / "model_state.pkl"
            if self._project_root else None
        )
        _temporal_available = bool(_temporal_model_pkl and _temporal_model_pkl.exists())
        chk_temporal = QCheckBox()
        chk_temporal.setChecked(False)
        chk_temporal.setEnabled(_temporal_available)
        chk_temporal.setToolTip(
            "Re-run the trained model at high-overlap stride over each frame, then apply smoothing\n"
            "and threshold to produce frame-precise bout boundaries (start/end frame per bout).\n"
            "Outputs an additional *_bouts.csv per session alongside the segment predictions."
            if _temporal_available else
            "Not available — model file not found for snapshot version."
        )
        opts_form.addRow("Export CSV results:", chk_csv)
        opts_form.addRow("Export XLSX results:", chk_xlsx)
        opts_form.addRow("Export labeled video:", chk_video)
        opts_form.addRow("FPS override:", fps_spin)
        opts_form.addRow("Temporal precision bouts:", chk_temporal)
        root_layout.addWidget(opts)

        # Progress area inside dialog
        batch_progress = QProgressBar()
        batch_progress.setRange(0, 1)
        batch_progress.setValue(0)
        batch_progress.setFormat("Idle")
        batch_log = QTextEdit()
        batch_log.setReadOnly(True)
        batch_log.setMaximumHeight(100)
        root_layout.addWidget(batch_progress)
        root_layout.addWidget(batch_log)

        # Dialog buttons
        ok_btn = QPushButton("Run Batch")
        ok_btn.setDefault(True)
        cancel_dlg_btn = QPushButton("Close")
        stop_batch_btn = QPushButton("Stop")
        stop_batch_btn.setEnabled(False)
        btn_row_dlg = QHBoxLayout()
        btn_row_dlg.addWidget(ok_btn)
        btn_row_dlg.addWidget(stop_batch_btn)
        btn_row_dlg.addStretch()
        btn_row_dlg.addWidget(cancel_dlg_btn)
        root_layout.addLayout(btn_row_dlg)
        cancel_dlg_btn.clicked.connect(dlg.reject)

        _batch_running = [False]

        def _run_batch() -> None:
            if not session_data:
                QMessageBox.warning(dlg, "No sessions", "Add at least one session before running.")
                return
            ok_btn.setEnabled(False)
            stop_batch_btn.setEnabled(True)
            self._batch_cancel_flag[0] = False
            _batch_running[0] = True
            out_dir = Path(out_edit.text().strip()) if out_edit.text().strip() else (
                self._project_root / "derived" / "batch_results"  # type: ignore[operator]
            )
            # Snapshot with any dialog overrides.
            run_snapshot = WorkflowSnapshot(
                model_version=snapshot.model_version,
                target_behavior=snapshot.target_behavior,
                segment_window_frames=snapshot.segment_window_frames,
                segment_stride_frames=snapshot.segment_stride_frames,
                excluded_feature_cols=snapshot.excluded_feature_cols,
                fps=float(fps_spin.value()),
                context_feature_config=snapshot.context_feature_config,
                pose_preset_id=snapshot.pose_preset_id,
                export_csv=bool(chk_csv.isChecked()),
                export_xlsx=bool(chk_xlsx.isChecked()),
                export_labeled_video=bool(chk_video.isChecked()),
                run_settings=snapshot.run_settings,
                created_at=snapshot.created_at,
            )

            _total = len(session_data)
            _done = [0]

            def _task(
                progress_cb: Callable[[int, int, str, str], None] | None = None,
            ) -> dict[str, Any]:
                return self._batch_run_task(
                    sessions=list(session_data),
                    snapshot=run_snapshot,
                    out_dir=out_dir,
                    cancel_flag=self._batch_cancel_flag,
                    apply_temporal=bool(chk_temporal.isChecked()),
                    progress_cb=progress_cb,
                )

            def _on_progress(value: int, maximum: int, log_line: str, status: str) -> None:
                batch_progress.setRange(0, max(1, maximum))
                batch_progress.setValue(value)
                batch_progress.setFormat(status[:60] if status else f"{value}/{maximum}")
                batch_log.append(log_line)
                batch_log.verticalScrollBar().setValue(batch_log.verticalScrollBar().maximum())

            def _on_done(result: dict[str, Any]) -> None:
                ok_btn.setEnabled(True)
                stop_batch_btn.setEnabled(False)
                _batch_running[0] = False
                n_ok = int(result.get("n_ok", 0))
                n_fail = int(result.get("n_fail", 0))
                out_path = str(result.get("output_dir", ""))
                batch_log.append(
                    f"Batch complete: {n_ok} session(s) succeeded, {n_fail} failed. "
                    f"Results written to: {out_path}"
                )
                batch_progress.setFormat("Complete")
                if n_fail == 0:
                    QMessageBox.information(
                        dlg,
                        "Batch Run Complete",
                        f"All {n_ok} session(s) processed successfully.\nResults: {out_path}",
                    )
                else:
                    QMessageBox.warning(
                        dlg,
                        "Batch Run Complete (with errors)",
                        f"{n_ok} session(s) succeeded, {n_fail} failed.\nResults: {out_path}",
                    )

            def _on_fail(tb: str) -> None:
                ok_btn.setEnabled(True)
                stop_batch_btn.setEnabled(False)
                _batch_running[0] = False
                if "BATCH_CANCELLED" in tb:
                    batch_log.append("Batch run cancelled.")
                    batch_progress.setFormat("Cancelled")
                else:
                    batch_log.append("ERROR: batch run failed — check logs.")
                    logger.error("Batch run failed:\n%s", tb)

            stop_batch_btn.clicked.connect(lambda: self._batch_cancel_flag.__setitem__(0, True))

            worker = TaskWorker(_task, _on_progress)
            worker.signals.finished.connect(_on_done)
            worker.signals.failed.connect(_on_fail)
            QThreadPool.globalInstance().start(worker)

        ok_btn.clicked.connect(_run_batch)
        dlg.exec()

    def _show_run_all_behaviors_dialog(self) -> None:
        """Run the batch inference pipeline for every behavior in the project.

        Each behavior's trained model is identified by its name converted to a
        filesystem-safe slug (spaces/special chars → underscores), matching the
        ``derived/models/<slug>/model_state.pkl`` path convention.  Behaviors
        whose model directory is not yet present are shown in grey but can still
        be checked — the run will skip them gracefully.

        Current project settings (window, stride, excluded features, FPS, pose
        preset, context config) are captured at dialog-open time so the run
        reflects the live configuration without requiring a saved snapshot.
        """
        from PySide6.QtWidgets import QFileDialog, QListWidget, QListWidgetItem  # type: ignore[attr-defined]

        if self._project_root is None:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return

        behaviors = [b for b in self._behaviors.behaviors if b.behavior_id != NO_BEHAVIOR_ID]
        if not behaviors:
            QMessageBox.warning(self, "No behaviors", "No behaviors are defined in this project.")
            return

        # ── Capture current project settings ──────────────────────────────────
        def _slug(name: str) -> str:
            return re.sub(r"[^\w.\-]", "_", name).strip("_") or "behavior"

        behavior_cfg = self._load_behavior_cfg()
        seg_window, seg_stride, _ = self._resolved_segment_settings(behavior_cfg)
        ctx_cfg_defaults = ContextFeatureConfig(flow_temporal_stride=int(self._flow_temporal_stride.value()))
        ctx_dict: dict[str, Any] = {
            "farneback_pyr_scale": ctx_cfg_defaults.farneback_pyr_scale,
            "farneback_levels": ctx_cfg_defaults.farneback_levels,
            "farneback_winsize": ctx_cfg_defaults.farneback_winsize,
            "farneback_iterations": ctx_cfg_defaults.farneback_iterations,
            "farneback_poly_n": ctx_cfg_defaults.farneback_poly_n,
            "farneback_poly_sigma": ctx_cfg_defaults.farneback_poly_sigma,
            "downsample_factor": ctx_cfg_defaults.downsample_factor,
            "flow_temporal_stride": ctx_cfg_defaults.flow_temporal_stride,
        }
        project_yaml = read_yaml(self._project_root / "project.yaml", {})
        default_fps = float(project_yaml.get("default_fps") or 30.0)
        pose_preset_id: str | None = None
        pose_cfg_path = self._project_root / "config" / "pose_features.yaml"
        if pose_cfg_path.exists():
            pose_raw = read_yaml(pose_cfg_path, {})
            pose_preset_id = str(pose_raw.get("active_preset_id") or "") or None
        tr_settings_path = self._project_root / "config" / "temporal_refinement_settings.json"
        tr_settings: dict[str, Any] = {}
        if tr_settings_path.exists():
            try:
                tr_settings = dict(read_json(tr_settings_path, {}))
            except Exception:
                pass

        def _model_exists(beh_name: str) -> bool:
            return (
                self._project_root / "derived" / "models" / _slug(beh_name) / "model_state.pkl"
            ).exists()

        # ── Dialog ────────────────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Run All Behaviors \u2014 Apply Each Model to New Videos")
        dlg.setMinimumWidth(720)
        dlg.setMinimumHeight(640)
        root_layout = QVBoxLayout(dlg)

        # Behavior list with model-availability indicators
        root_layout.addWidget(QLabel(
            "<b>Behaviors to run</b> \u2014 model resolved from "
            "<code>derived/models/&lt;behavior_name&gt;/</code> "
            "(grey = no trained model found yet):"
        ))
        beh_list = QListWidget()
        beh_list.setMaximumHeight(160)
        beh_checkboxes: dict[str, QCheckBox] = {}
        for b in behaviors:
            item = QListWidgetItem()
            beh_list.addItem(item)
            row_w = QWidget()
            row_layout = QHBoxLayout(row_w)
            row_layout.setContentsMargins(4, 0, 4, 0)
            exists = _model_exists(b.name)
            chk = QCheckBox(f"{b.name}  [model: {_slug(b.name)}]")
            chk.setChecked(exists)
            chk.setToolTip(
                f"Trained model found at derived/models/{_slug(b.name)}/"
                if exists else
                f"No model found at derived/models/{_slug(b.name)}/\n"
                "Train this behavior first, or verify the model folder name matches."
            )
            if not exists:
                chk.setStyleSheet("color: #888888;")
            row_layout.addWidget(chk)
            row_layout.addStretch()
            row_w.setLayout(row_layout)
            item.setSizeHint(row_w.sizeHint())
            beh_list.setItemWidget(item, row_w)
            beh_checkboxes[b.behavior_id] = chk
        root_layout.addWidget(beh_list)

        # Session list
        root_layout.addWidget(QLabel(
            "<b>Sessions to process</b> (each row = one video + one pose file):"
        ))
        session_list = QListWidget()
        session_list.setMinimumHeight(130)
        root_layout.addWidget(session_list)
        session_data: list[dict[str, str]] = []

        def _refresh_list() -> None:
            session_list.clear()
            for entry in session_data:
                vid = Path(entry.get("video_path", "")).name
                pose = Path(entry.get("pose_path", "")).name
                animal = str(entry.get("animal_id", "")).strip() or "(auto)"
                session_list.addItem(f"{vid}  +  {pose}  [{animal}]")

        def _add_session() -> None:
            vid_path, _ = QFileDialog.getOpenFileName(
                dlg, "Select video file", str(self._project_root),
                "Video files (*.mp4 *.avi *.mkv *.mov *.wmv *.m4v);;All files (*)",
            )
            if not vid_path:
                return
            pose_path, _ = QFileDialog.getOpenFileName(
                dlg, "Select pose/tracking file for this video",
                str(Path(vid_path).parent),
                "Tracking files (*.csv *.h5 *.hdf5);;All files (*)",
            )
            if not pose_path:
                return
            stem = Path(vid_path).stem
            for token in ("DLC", "dlc", "_tracked", "_pose"):
                stem = stem.split(token)[0]
            animal_id = stem.strip("_-") or "animal_01"
            session_data.append({"video_path": vid_path, "pose_path": pose_path, "animal_id": animal_id})
            _refresh_list()

        def _remove_session() -> None:
            rows = sorted({i.row() for i in session_list.selectedItems()}, reverse=True)
            for r in rows:
                session_data.pop(r)
            _refresh_list()

        list_btn_row = QHBoxLayout()
        add_sess_btn = QPushButton("Add Session\u2026")
        add_sess_btn.clicked.connect(_add_session)
        rem_sess_btn = QPushButton("Remove Selected")
        rem_sess_btn.clicked.connect(_remove_session)
        list_btn_row.addWidget(add_sess_btn)
        list_btn_row.addWidget(rem_sess_btn)
        list_btn_row.addStretch()
        root_layout.addLayout(list_btn_row)

        # Output directory
        out_row = QHBoxLayout()
        out_edit = QLineEdit()
        out_edit.setText(str(self._project_root / "derived" / "batch_results"))
        out_btn = QPushButton("Browse\u2026")

        def _pick_out() -> None:
            d = QFileDialog.getExistingDirectory(dlg, "Select output folder", out_edit.text())
            if d:
                out_edit.setText(d)

        out_btn.clicked.connect(_pick_out)
        out_row.addWidget(QLabel("Output folder:"))
        out_row.addWidget(out_edit, 1)
        out_row.addWidget(out_btn)
        root_layout.addLayout(out_row)

        # Output options
        opts = QGroupBox("Output options")
        opts_form = QFormLayout(opts)
        chk_csv = QCheckBox()
        chk_csv.setChecked(True)
        chk_xlsx = QCheckBox()
        chk_xlsx.setChecked(False)
        chk_video = QCheckBox()
        chk_video.setChecked(False)
        chk_video.setToolTip("Write labeled overlay video (slow \u2014 requires OpenCV).")
        fps_spin = QDoubleSpinBox()
        fps_spin.setRange(1.0, 1000.0)
        fps_spin.setDecimals(1)
        fps_spin.setValue(default_fps)
        fps_spin.setToolTip("Override frames per second (only needed if project default is wrong for these files).")
        opts_form.addRow("Export CSV results:", chk_csv)
        opts_form.addRow("Export XLSX results:", chk_xlsx)
        opts_form.addRow("Export labeled video:", chk_video)
        opts_form.addRow("FPS override:", fps_spin)
        root_layout.addWidget(opts)

        # Progress
        all_progress = QProgressBar()
        all_progress.setRange(0, 1)
        all_progress.setValue(0)
        all_progress.setFormat("Idle")
        all_log = QTextEdit()
        all_log.setReadOnly(True)
        all_log.setMaximumHeight(110)
        root_layout.addWidget(all_progress)
        root_layout.addWidget(all_log)

        # Dialog buttons
        run_all_btn = QPushButton("Run All")
        run_all_btn.setDefault(True)
        stop_all_btn = QPushButton("Stop")
        stop_all_btn.setEnabled(False)
        close_dlg_btn = QPushButton("Close")
        btns_row = QHBoxLayout()
        btns_row.addWidget(run_all_btn)
        btns_row.addWidget(stop_all_btn)
        btns_row.addStretch()
        btns_row.addWidget(close_dlg_btn)
        root_layout.addLayout(btns_row)
        close_dlg_btn.clicked.connect(dlg.reject)

        _running = [False]

        def _run_all() -> None:
            selected = [b for b in behaviors if beh_checkboxes[b.behavior_id].isChecked()]
            if not selected:
                QMessageBox.warning(dlg, "Nothing selected", "Select at least one behavior to run.")
                return
            if not session_data:
                QMessageBox.warning(dlg, "No sessions", "Add at least one session before running.")
                return
            run_all_btn.setEnabled(False)
            stop_all_btn.setEnabled(True)
            self._batch_cancel_flag[0] = False
            _running[0] = True
            out_dir = Path(out_edit.text().strip()) if out_edit.text().strip() else (
                self._project_root / "derived" / "batch_results"  # type: ignore[operator]
            )
            fps_val = float(fps_spin.value())
            run_settings = self._ui_settings_payload()

            # Build per-behavior snapshots — model_version = slugified behavior name.
            snapshots_to_run: list[tuple[str, WorkflowSnapshot]] = [
                (
                    b.name,
                    WorkflowSnapshot(
                        model_version=_slug(b.name),
                        target_behavior=b.behavior_id,
                        segment_window_frames=seg_window,
                        segment_stride_frames=seg_stride,
                        excluded_feature_cols=sorted(self._excluded_feature_cols),
                        fps=fps_val,
                        context_feature_config=ctx_dict,
                        pose_preset_id=pose_preset_id,
                        export_csv=bool(chk_csv.isChecked()),
                        export_xlsx=bool(chk_xlsx.isChecked()),
                        export_labeled_video=bool(chk_video.isChecked()),
                        run_settings=run_settings,
                        temporal_refinement_settings=tr_settings,
                    ),
                )
                for b in selected
            ]
            n_behaviors = len(snapshots_to_run)
            steps_per_session = 4  # pose, ctx, repr, infer
            steps_per_behavior = len(session_data) * steps_per_session
            total_steps = n_behaviors * steps_per_behavior

            def _task(
                progress_cb: Callable[[int, int, str, str], None] | None = None,
            ) -> dict[str, Any]:
                all_results: dict[str, Any] = {}
                step_offset = 0
                for beh_name, snap in snapshots_to_run:
                    if self._batch_cancel_flag[0]:
                        raise RuntimeError("BATCH_CANCELLED")
                    if progress_cb:
                        progress_cb(step_offset, total_steps, f"Starting: {beh_name}", f"{beh_name} \u2026")
                    beh_out_dir = out_dir / _slug(beh_name)
                    _offset_capture = step_offset

                    def _beh_progress(
                        v: int, m: int, line: str, status: str,
                        _bn: str = beh_name, _off: int = _offset_capture,
                    ) -> None:
                        if progress_cb:
                            progress_cb(_off + v, total_steps, f"[{_bn}] {line}", f"[{_bn}] {status}")

                    beh_result = self._batch_run_task(
                        sessions=list(session_data),
                        snapshot=snap,
                        out_dir=beh_out_dir,
                        cancel_flag=self._batch_cancel_flag,
                        apply_temporal=False,
                        progress_cb=_beh_progress,
                    )
                    all_results[beh_name] = beh_result
                    step_offset += steps_per_behavior
                return {"behaviors": all_results, "output_dir": str(out_dir)}

            def _on_progress(value: int, maximum: int, log_line: str, status: str) -> None:
                all_progress.setRange(0, max(1, maximum))
                all_progress.setValue(value)
                all_progress.setFormat(status[:70] if status else f"{value}/{maximum}")
                all_log.append(log_line)
                all_log.verticalScrollBar().setValue(all_log.verticalScrollBar().maximum())

            def _on_done(result: dict[str, Any]) -> None:
                run_all_btn.setEnabled(True)
                stop_all_btn.setEnabled(False)
                _running[0] = False
                all_progress.setFormat("Complete")
                beh_results = result.get("behaviors", {})
                n_ok_total = sum(int(r.get("n_ok", 0)) for r in beh_results.values())
                n_fail_total = sum(int(r.get("n_fail", 0)) for r in beh_results.values())
                out_path = str(result.get("output_dir", ""))
                all_log.append(
                    f"Complete: {len(beh_results)} behavior(s) \u2014 "
                    f"{n_ok_total} session(s) succeeded, {n_fail_total} failed. "
                    f"Results: {out_path}"
                )
                if n_fail_total == 0:
                    QMessageBox.information(
                        dlg,
                        "Run All Behaviors Complete",
                        f"All {len(beh_results)} behavior(s), {n_ok_total} session(s) processed.\nResults: {out_path}",
                    )
                else:
                    QMessageBox.warning(
                        dlg,
                        "Run All Behaviors Complete (with errors)",
                        f"{n_ok_total} session(s) succeeded, {n_fail_total} failed.\nResults: {out_path}",
                    )

            def _on_fail(tb: str) -> None:
                run_all_btn.setEnabled(True)
                stop_all_btn.setEnabled(False)
                _running[0] = False
                if "BATCH_CANCELLED" in tb:
                    all_log.append("Run cancelled.")
                    all_progress.setFormat("Cancelled")
                else:
                    all_log.append("ERROR: run failed \u2014 check logs.")
                    logger.error("Run all behaviors failed:\n%s", tb)

            stop_all_btn.clicked.connect(lambda: self._batch_cancel_flag.__setitem__(0, True))
            worker = TaskWorker(_task, _on_progress)
            worker.signals.finished.connect(_on_done)
            worker.signals.failed.connect(_on_fail)
            QThreadPool.globalInstance().start(worker)

        run_all_btn.clicked.connect(_run_all)
        dlg.exec()

    def _batch_run_task(
        self,
        sessions: list[dict[str, str]],
        snapshot: WorkflowSnapshot,
        out_dir: Path,
        cancel_flag: list[bool],
        apply_temporal: bool = False,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, Any]:
        """Background task: run inference on each session using the workflow snapshot.

        For each session:
        1. Extract pose features (frame-level kinematics)
        2. Extract context features (optical-flow, surface motion, nose features)
        3. Build segment representations (sliding windows, excluded_feature_cols applied)
        4. Infer with the snapshot's trained model
        5. (Optional) Dense temporal inference → frame-precise bout intervals
        6. Write per-session CSV (and optionally XLSX / labeled video)

        Returns: {n_ok, n_fail, output_dir, session_results: [{session_id, status, output_file}]}
        """
        import hashlib
        import json as _json
        import shutil

        assert self._project_root is not None

        def _log(value: int, total: int, msg: str, status: str = "") -> None:
            if progress_cb:
                progress_cb(value, total, msg, status or msg[:70])

        total_sessions = len(sessions)
        steps_per_session = 5 if apply_temporal else 4  # pose, ctx, repr, infer, (temporal)
        total_steps = total_sessions * steps_per_session
        step = 0
        n_ok = 0
        n_fail = 0
        session_results: list[dict[str, Any]] = []

        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self._project_root / "derived" / "models" / snapshot.model_version

        fps = float(snapshot.fps or 30.0)
        excluded = frozenset(snapshot.excluded_feature_cols or [])
        ctx_cfg_dict = dict(snapshot.context_feature_config or {})
        ctx_cfg = ContextFeatureConfig(
            farneback_pyr_scale=float(ctx_cfg_dict.get("farneback_pyr_scale", 0.5)),
            farneback_levels=int(ctx_cfg_dict.get("farneback_levels", 3)),
            farneback_winsize=int(ctx_cfg_dict.get("farneback_winsize", 15)),
            farneback_iterations=int(ctx_cfg_dict.get("farneback_iterations", 3)),
            farneback_poly_n=int(ctx_cfg_dict.get("farneback_poly_n", 5)),
            farneback_poly_sigma=float(ctx_cfg_dict.get("farneback_poly_sigma", 1.2)),
            downsample_factor=int(ctx_cfg_dict.get("downsample_factor", 0)),
            flow_temporal_stride=int(ctx_cfg_dict.get("flow_temporal_stride", 10)),
        )

        # Build the TemporalRefinementConfig once if we need it.
        tr_cfg = None
        if apply_temporal:
            from abel.temporal_refinement.temporal_refinement_service import TemporalRefinementConfig
            tr_raw = dict(snapshot.temporal_refinement_settings or {})
            # Prefer per-behavior settings over global if available.
            behavior_overrides = dict(
                (tr_raw.get("by_behavior") or {}).get(snapshot.target_behavior, {}) or {}
            )
            effective = {**dict(tr_raw.get("__all__") or {}), **behavior_overrides}
            tr_cfg = TemporalRefinementConfig(
                selected_behavior_models={snapshot.target_behavior: snapshot.model_version},
                competition_include_no_behavior=bool(effective.get("competition_include_no_behavior", True)),
                competition_temperature=float(effective.get("competition_temperature", 0.85)),
                competition_score_power=float(effective.get("competition_score_power", 1.4)),
                no_behavior_prior=float(effective.get("no_behavior_prior", 0.05)),
                no_behavior_complement_weight=float(effective.get("no_behavior_complement_weight", 0.7)),
                no_behavior_floor=float(effective.get("no_behavior_floor", 0.0)),
                competition_min_win_margin=float(effective.get("competition_min_win_margin", 0.0)),
                inference_step_seconds=float(effective.get("inference_step_seconds", 0.1)),
                inference_warmup_no_behavior_seconds=float(effective.get("inference_warmup_no_behavior_seconds", 1.5)),
                smoothing_method=str(effective.get("smoothing_method", "moving_average")),
                smoothing_window=int(effective.get("smoothing_window", 5)),
                onset_threshold=float(effective.get("onset_threshold", 0.5)),
                offset_threshold=(
                    float(effective["offset_threshold"])
                    if effective.get("offset_threshold") is not None else None
                ),
                min_bout_duration_frames=int(effective.get("min_bout_duration_frames", 6)),
                merge_gap_frames=int(effective.get("merge_gap_frames", 3)),
            )

        batch_kp_aliases = self._keypoint_aliases()

        for i, entry in enumerate(sessions):
            if cancel_flag[0]:
                raise RuntimeError("BATCH_CANCELLED")
            video_path = Path(entry.get("video_path", ""))
            pose_path = Path(entry.get("pose_path", ""))
            animal_id = str(entry.get("animal_id", "")).strip() or video_path.stem
            # Use a hash-based session_id that is unique per video and isolated
            # from the project's real sessions so batch data never pollutes the
            # project-level parquet files.
            session_id = "batch_" + hashlib.md5(str(video_path).encode()).hexdigest()[:8]

            # Each session gets its own temp scratch directory so feature files
            # stay isolated from the real project derived/ tree.
            temp_root = out_dir / "_batch_cache" / session_id
            temp_root.mkdir(parents=True, exist_ok=True)

            sess_label = f"[{i + 1}/{total_sessions}] {video_path.name}"

            try:
                # ── Step 1: Pose features ──────────────────────────────────────
                _log(step, total_steps, f"{sess_label}: extracting pose features…", "Extracting pose features")
                self._pose.extract_and_save_frame_pose_features(
                    project_root=temp_root,
                    pose_path=pose_path,
                    fps=fps,
                    animal_id=animal_id,
                    session_id=session_id,
                    video_id=session_id,
                    keypoint_aliases=batch_kp_aliases,
                )
                step += 1
                if cancel_flag[0]:
                    raise RuntimeError("BATCH_CANCELLED")

                # ── Step 2: Context features ───────────────────────────────────
                _log(step, total_steps, f"{sess_label}: extracting context features…", "Extracting context features")
                self._context.compute_frame_context(
                    project_root=temp_root,
                    video_path=video_path,
                    pose_path=pose_path,
                    animal_id=animal_id,
                    session_id=session_id,
                    config=ctx_cfg,
                    keypoint_aliases=batch_kp_aliases,
                )
                step += 1
                if cancel_flag[0]:
                    raise RuntimeError("BATCH_CANCELLED")

                # ── Step 3: Build representations ─────────────────────────────
                _log(step, total_steps, f"{sess_label}: building segment representations…", "Building representations")
                frame_pose_path = temp_root / "derived" / "pose_features" / "frame_pose.parquet"
                frame_ctx_path = temp_root / "derived" / "context_features" / "frame_context.parquet"
                _repr_svc = BehaviorRepresentationService()
                _, segment_df = _repr_svc.build(
                    project_root=temp_root,
                    frame_pose_path=frame_pose_path,
                    frame_context_path=frame_ctx_path,
                    config=RepresentationConfig(
                        window_size_frames=int(snapshot.segment_window_frames),
                        window_stride_frames=int(snapshot.segment_stride_frames),
                        excluded_feature_cols=excluded,
                    ),
                    session_ids={session_id},
                )
                step += 1
                if cancel_flag[0]:
                    raise RuntimeError("BATCH_CANCELLED")

                # ── Step 4: Segment-level inference ───────────────────────────
                _log(step, total_steps, f"{sess_label}: running model inference…", "Running inference")
                if segment_df.empty:
                    raise ValueError("No segments generated — recording may be too short for the current window settings.")
                pred_df = ActiveLearningTrainerService.predict_segments(model_dir, segment_df)
                step += 1

                # ── Step 5 (optional): Dense temporal inference ────────────────
                bouts_df: pd.DataFrame | None = None
                if apply_temporal and tr_cfg is not None:
                    _log(step, total_steps, f"{sess_label}: running temporal precision…", "Temporal inference")
                    bouts_df = self._run_temporal_inference_for_batch(
                        temp_root=temp_root,
                        session_id=session_id,
                        snapshot=snapshot,
                        tr_cfg=tr_cfg,
                        fps=fps,
                        cancel_flag=cancel_flag,
                        log_fn=lambda msg: _log(step, total_steps, f"{sess_label}: {msg}"),
                    )
                    step += 1

                # ── Write outputs ──────────────────────────────────────────────
                out_stem = f"{animal_id}_{video_path.stem}"
                out_csv = out_dir / f"{out_stem}_predictions.csv"
                pred_df.to_csv(out_csv, index=False)

                if snapshot.export_xlsx:
                    try:
                        out_xlsx = out_dir / f"{out_stem}_predictions.xlsx"
                        pred_df.to_excel(str(out_xlsx), index=False, engine="openpyxl")
                    except Exception as xlsx_exc:
                        _log(step, total_steps, f"{sess_label}: XLSX export skipped ({xlsx_exc}).")

                bout_csv: str | None = None
                if bouts_df is not None and not bouts_df.empty:
                    bout_path = out_dir / f"{out_stem}_bouts.csv"
                    bouts_df.to_csv(bout_path, index=False)
                    bout_csv = str(bout_path)

                session_results.append({
                    "session_id": session_id,
                    "animal_id": animal_id,
                    "video": str(video_path),
                    "status": "ok",
                    "n_segments": len(pred_df),
                    "n_bouts": len(bouts_df) if bouts_df is not None else None,
                    "output_csv": str(out_csv),
                    "output_bouts_csv": bout_csv,
                })
                n_ok += 1
                bout_note = f", {len(bouts_df)} bouts" if bouts_df is not None else ""
                _log(step, total_steps, f"{sess_label}: done. {len(pred_df)} segments scored{bout_note}.", "Done")

            except RuntimeError as re_exc:
                if "BATCH_CANCELLED" in str(re_exc):
                    raise
                n_fail += 1
                _log(step, total_steps, f"{sess_label}: FAILED — {re_exc}")
                session_results.append({
                    "session_id": session_id,
                    "animal_id": animal_id,
                    "video": str(video_path),
                    "status": "failed",
                    "error": str(re_exc),
                })
                step = (i + 1) * steps_per_session
            except Exception as exc:
                n_fail += 1
                _log(step, total_steps, f"{sess_label}: FAILED — {exc}")
                logger.exception("Batch run error for session %s", session_id)
                session_results.append({
                    "session_id": session_id,
                    "animal_id": animal_id,
                    "video": str(video_path),
                    "status": "failed",
                    "error": str(exc),
                })
                step = (i + 1) * steps_per_session
            finally:
                # Remove per-session scratch data regardless of outcome.
                shutil.rmtree(temp_root, ignore_errors=True)

        # Write a run manifest so the user can audit what was processed.
        write_json(
            out_dir / "batch_run_manifest.json",
            {
                "model_version": snapshot.model_version,
                "target_behavior": snapshot.target_behavior,
                "n_ok": n_ok,
                "n_fail": n_fail,
                "sessions": session_results,
            },
        )

        return {
            "n_ok": n_ok,
            "n_fail": n_fail,
            "output_dir": str(out_dir),
            "session_results": session_results,
        }

    def _run_temporal_inference_for_batch(
        self,
        *,
        temp_root: Path,
        session_id: str,
        snapshot: WorkflowSnapshot,
        tr_cfg: "TemporalRefinementConfig",  # type: ignore[name-defined]
        fps: float,
        cancel_flag: list[bool],
        log_fn: Callable[[str], None],
    ) -> pd.DataFrame | None:
        """Run dense temporal competition inference for one batch session.

        Uses the same trained AL model as the segment pipeline, but strides it
        across every ~3 frames (at 30 fps with 0.1 s step) rather than the
        normal 15-frame stride.  Multi-model softmax competition and threshold
        post-processing produce frame-precise bout intervals.

        The heavy lifting is delegated to ``TemporalRefinementService`` using
        a mirror of only the required files inside *temp_root* (which gets
        cleaned up by the caller's ``finally`` block).

        Returns a DataFrame with columns ``start_frame``, ``end_frame``,
        ``session_id``, ``animal_id`` or ``None`` on failure.
        """
        import shutil

        from abel.temporal_refinement.temporal_refinement_service import TemporalRefinementService

        log_fn("setting up temporal mirror…")

        # ── Mirror only what the service needs into temp_root ──────────────
        # 1. Behavior model pkl(s) — hard-link first, copy as fallback.
        required_models: dict[str, str] = dict(tr_cfg.selected_behavior_models or {})
        if not required_models:
            required_models = {snapshot.target_behavior: snapshot.model_version}

        for _bid, _mv in required_models.items():
            src_model_dir = self._project_root / "derived" / "models" / _mv  # type: ignore[operator]
            dst_model_dir = temp_root / "derived" / "models" / _mv
            dst_model_dir.mkdir(parents=True, exist_ok=True)
            for fname in ("model_state.pkl", "model_card.yaml"):
                src = src_model_dir / fname
                dst = dst_model_dir / fname
                if src.exists() and not dst.exists():
                    try:
                        os.link(src, dst)  # hard link — no data copy, works on same volume
                    except OSError:
                        shutil.copy2(src, dst)

        # 2. Segment features stub — the service reads this only to infer window size.
        #    We create a minimal two-row parquet with the correct start/end frame spread
        #    instead of copying the (potentially large) real project file.
        seg_stub_dir = temp_root / "derived" / "representations"
        seg_stub_dir.mkdir(parents=True, exist_ok=True)
        seg_stub_path = seg_stub_dir / "segment_features.parquet"
        if not seg_stub_path.exists():
            _win = int(snapshot.segment_window_frames)
            _stub = pd.DataFrame({"start_frame": [0], "end_frame": [_win - 1]})
            _stub.to_parquet(seg_stub_path, index=False)

        # 3. Minimal import manifest so FPS is resolved correctly for this session.
        manifest_dir = temp_root / "derived" / "review_tables"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "import_manifest.json"
        if not manifest_path.exists():
            import json as _json
            _video_id = f"video_{session_id}"
            _pose_id = f"pose_{session_id}"
            manifest_path.write_text(
                _json.dumps(
                    {
                        "videos": [{"asset_id": _video_id, "source_path": "", "fps": fps}],
                        "poses": [{"asset_id": _pose_id, "source_path": "", "format": "dlc", "has_likelihood": True}],
                        "linked_sessions": [
                            {
                                "session_id": session_id,
                                "video_asset_id": _video_id,
                                "pose_asset_id": _pose_id,
                            }
                        ],
                        "recordings": [],
                    }
                ),
                encoding="utf-8",
            )

        if cancel_flag[0]:
            raise RuntimeError("BATCH_CANCELLED")

        # ── Run dense inference ────────────────────────────────────────────
        log_fn("running dense competition inference…")
        temporal_svc = TemporalRefinementService()
        temporal_svc.set_project(temp_root)
        try:
            temporal_svc.run_temporal_refinement_inference(
                concept_id=snapshot.target_behavior,
                sessions=[session_id],
                config=tr_cfg,
                force=True,
                progress_cb=lambda msg: log_fn(f"  {msg}"),
            )
        except Exception as exc:
            log_fn(f"dense inference failed — {exc}")
            logger.warning("Batch temporal inference failed for session %s: %s", session_id, exc)
            return None

        if cancel_flag[0]:
            raise RuntimeError("BATCH_CANCELLED")

        # ── Postprocess → bout intervals ───────────────────────────────────
        log_fn("extracting bout intervals…")
        try:
            pp_result = temporal_svc.run_temporal_refinement_postprocess(
                concept_id=snapshot.target_behavior,
                sessions=[session_id],
                config=tr_cfg,
                force=True,
                progress_cb=lambda msg: log_fn(f"  {msg}"),
            )
        except Exception as exc:
            log_fn(f"postprocess failed — {exc}")
            logger.warning("Batch temporal postprocess failed for session %s: %s", session_id, exc)
            return None

        # ── Collect bout DataFrame from artifact ──────────────────────────
        bout_dir_raw = str((pp_result or {}).get("postprocess_dir", "")).strip()
        if not bout_dir_raw:
            return None
        bout_dir = Path(bout_dir_raw)
        bout_path = bout_dir / "bout_outputs" / f"{session_id}_bouts.parquet"
        if not bout_path.exists():
            return None
        try:
            return pd.read_parquet(bout_path)
        except Exception as exc:
            log_fn(f"could not read bout file — {exc}")
            return None

    def _on_pipeline_finished(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        self._cancel_flag[0] = False
        self._pipeline_step_scale = 1
        self._refresh_saved_model_options()
        summary: _RunSummary = payload["summary"]
        self._snapshot_evaluation_graphs_for_model(summary.model_version, target_behavior=str(payload.get("target_behavior", "")))
        self._refresh_viz_model_options()
        target_behavior = str(payload.get("target_behavior", "target_behavior"))
        # Persist workflow snapshot so batch-run can reproduce this exact pipeline.
        self._save_workflow_snapshot(
            model_version=summary.model_version,
            target_behavior=target_behavior,
        )
        model_disp = self._display_model_name(summary.model_version)
        candidates = payload.get("candidates", [])
        metrics = payload.get("metrics", {})
        fusion_diag = payload.get("fusion_diagnostics", {})
        subject_map = self._subject_map()
        self._progress.setValue(self._progress.maximum())
        self._progress.setFormat("Complete")
        self._status.setText(
            f"Pipeline complete ({target_behavior}). subjects={summary.n_sessions} frames={summary.n_frame_rows} "
            f"segments={summary.n_segment_rows} train_rows={summary.n_train_rows} candidates={summary.n_candidates} "
            f"F1={float(metrics.get('f1', 0.0)):.3f} PR-AUC={float(metrics.get('pr_auc', 0.0)):.3f} "
            f"model={model_disp} model_device={summary.model_device_used} fusion_device={summary.fusion_device_used} | "
            f"{self._quality_explanation(metrics)}"
        )
        self._append_log(
            "Pipeline complete. "
            f"model={model_disp}, "
            f"F1={float(metrics.get('f1', 0.0)):.3f}, PR-AUC={float(metrics.get('pr_auc', 0.0)):.3f}, "
            f"train={int(metrics.get('n_train', 0))}, val={int(metrics.get('n_val', 0))}, "
            f"model_device={summary.model_device_used}, fusion_device={summary.fusion_device_used}."
        )
        self._append_log(self._quality_explanation(metrics))

        phase1_result = payload.get("phase1_result") if isinstance(payload, dict) else None
        if isinstance(phase1_result, dict) and bool(phase1_result.get("enabled", False)):
            self._append_log("Phase 1 benchmarking complete.")
            for card in list(phase1_result.get("summary_cards") or []):
                self._append_log(f"- {card}")

        model_cpu_fallback = bool(metrics.get("used_cpu_fallback", False))
        fusion_cpu_fallback = bool(fusion_diag.get("fusion_used_cpu_fallback", False))
        if model_cpu_fallback or fusion_cpu_fallback:
            model_reason = str(metrics.get("fallback_reason", "")).strip()
            fusion_reason = str(fusion_diag.get("fusion_fallback_reason", "")).strip()
            details = []
            if model_cpu_fallback:
                details.append(f"Model fallback: {model_reason or 'no additional details'}")
            if fusion_cpu_fallback:
                details.append(f"Fusion fallback: {fusion_reason or 'no additional details'}")
            QMessageBox.warning(
                self,
                "GPU Fallback Notice",
                "One or more stages fell back to CPU. The run completed, but performance may be slower.\n\n"
                + "\n".join(details)
                + "\n\nThis is common on Windows when CUDA safeguards trigger or when GPU/CPU device mismatch occurs. "
                "Leave 'Require GPU (fail if fallback occurs)' disabled unless strict GPU execution is required.",
            )

        self._populate_candidate_table(candidates)
        self._emit_uncertainty_candidates_for_clip_extraction(
            candidates,
            source_label=f"Active Learning — Uncertainty Ranking ({target_behavior})",
        )
        self._refresh_visualization_preview()

    def _on_existing_model_finished(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        self._cancel_flag[0] = False
        model_version = str(payload.get("model_version", ""))
        self._snapshot_evaluation_graphs_for_model(model_version, target_behavior=str(payload.get("target_behavior", "")))
        self._refresh_viz_model_options()
        model_disp = self._display_model_name(model_version)
        target_behavior = str(payload.get("target_behavior", "target_behavior"))
        n_candidates = int(payload.get("n_candidates", 0))
        segment_rows = int(payload.get("segment_rows", 0))
        self._progress.setValue(self._progress.maximum())
        self._progress.setFormat("Complete")
        self._status.setText(
            f"Existing model run complete ({model_disp}). "
            f"target={target_behavior}, segments={segment_rows}, candidates={n_candidates}."
        )
        self._append_log(
            f"Existing model run complete: model={model_disp}, target={target_behavior}, "
            f"segments={segment_rows}, candidates={n_candidates}."
        )
        self._populate_candidate_table(list(payload.get("candidates", [])))
        self._emit_uncertainty_candidates_for_clip_extraction(
            list(payload.get("candidates", [])),
            source_label=f"Active Learning — Uncertainty Ranking ({target_behavior})",
        )
        self._refresh_visualization_preview()

    def _populate_candidate_table(self, candidates: list[Any]) -> None:
        subject_map = self._subject_map()
        self._table.setRowCount(0)
        for cand in candidates[:200]:
            row = self._table.rowCount()
            self._table.insertRow(row)
            session_id = str(getattr(cand, "session_id", ""))
            subject = subject_map.get(session_id, session_id)
            start_frame = int(getattr(cand, "start_frame", 0))
            end_frame = int(getattr(cand, "end_frame", 0))
            seg_id = str(getattr(cand, "segment_id", getattr(cand, "window_id", "")))
            pred_prob = float(getattr(cand, "prediction_prob", 0.0))
            unc_score = float(getattr(cand, "uncertainty_score", 0.0))
            self._table.setItem(row, 0, QTableWidgetItem(subject))
            self._table.setItem(row, 1, QTableWidgetItem(f"{start_frame}-{end_frame}"))
            self._table.setItem(row, 2, QTableWidgetItem(seg_id))
            self._table.setItem(row, 3, QTableWidgetItem(f"{pred_prob:.3f}"))
            self._table.setItem(row, 4, QTableWidgetItem(f"{unc_score:.3f}"))

    def _subject_map(self) -> dict[str, str]:
        if self._project_root is None:
            return {}
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return {}
        out: dict[str, str] = {}
        for linked in manifest.linked_sessions:
            label = (linked.subject_id or "").strip()
            out[str(linked.session_id)] = label or str(linked.session_id)
        return out

    def _emit_uncertainty_candidates_for_clip_extraction(
        self, candidates: list[Any], source_label: str = "Active Learning — Uncertainty Ranking",
        append: bool = False,
    ) -> None:
        """Convert pipeline CandidateSegment objects to CandidateWindow objects scored by
        uncertainty and emit them so the clip extraction tab can be pre-populated.
        Sorting by ``total_score`` in clip extraction then naturally produces
        top/median/bottom *by uncertainty* rather than by prediction score.

        When *append* is True the candidates ACCUMULATE in the Clips tab
        (used by batch runs so each behavior's clips add up); otherwise they
        replace the previous uncertainty set (single-run clean re-population)."""
        if not candidates:
            return
        windows: list[CandidateWindow] = []
        for cand in candidates:
            session_id = str(getattr(cand, "session_id", "")).strip()
            if not session_id:
                continue
            seg_id = str(getattr(cand, "segment_id", getattr(cand, "window_id", ""))).strip()
            start_frame = int(getattr(cand, "start_frame", 0))
            end_frame = int(getattr(cand, "end_frame", 0))
            uncertainty = float(getattr(cand, "uncertainty_score", 0.0))
            behavior_id = str(getattr(cand, "behavior_id", "") or "").strip() or None
            window_id = seg_id or f"al_unc_{session_id}_{start_frame}_{end_frame}"
            windows.append(
                CandidateWindow(
                    window_id=window_id,
                    session_id=session_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    behavior_id=behavior_id,
                    seed_similarity_score=0.0,
                    total_score=float(np.clip(uncertainty, 0.0, 1.0)),
                    source="active_learning_uncertainty",
                )
            )
        if windows:
            if append:
                self.uncertainty_candidates_appended.emit(windows, source_label)
            else:
                self.uncertainty_candidates_updated.emit(windows, source_label)

    # ------------------------------------------------------------------
    # Edge-case finder
    # ------------------------------------------------------------------

    def _open_edge_case_finder(self) -> None:
        if self._project_root is None:
            QMessageBox.warning(self, "Edge Case Finder", "Open a project first.")
            return

        models_root = self._project_root / "derived" / "models"
        if not models_root.exists():
            QMessageBox.information(self, "Edge Case Finder", "No trained models found. Run the active-learning pipeline first.")
            return

        # Collect all behavior models that have segment_predictions.parquet
        available: list[tuple[str, str, Path]] = []  # (model_key, display_name, pred_path)
        for model_dir in sorted(models_root.iterdir()):
            if not model_dir.is_dir() or not model_dir.name.startswith("behavior_model_"):
                continue
            pred_path = model_dir / "segment_predictions.parquet"
            if pred_path.exists():
                available.append((model_dir.name, self._display_model_name(model_dir.name), pred_path))

        if len(available) < 2:
            QMessageBox.information(
                self,
                "Edge Case Finder",
                "At least two trained behavior models with predictions are required.\n"
                "Run the pipeline on each behavior to generate predictions first.",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Edge Case Finder — Competing Behaviors")
        dlg.setMinimumWidth(700)

        form = QFormLayout()

        behavior_a = QComboBox()
        behavior_b = QComboBox()
        model_by_key: dict[str, tuple[str, Path]] = {}
        for model_key, display_name, pred_path in available:
            model_by_key[model_key] = (display_name, pred_path)
            behavior_a.addItem(display_name, userData=model_key)
            behavior_b.addItem(display_name, userData=model_key)

        ALL_OTHER_KEY = "__all_other_behaviors__"
        behavior_b.insertItem(0, "All other behaviors", userData=ALL_OTHER_KEY)
        if len(available) >= 2:
            behavior_b.setCurrentIndex(2)
        else:
            behavior_b.setCurrentIndex(0)

        min_score_target = QDoubleSpinBox()
        min_score_target.setRange(0.1, 0.99)
        min_score_target.setSingleStep(0.05)
        min_score_target.setDecimals(2)
        min_score_target.setValue(0.35)
        min_score_target.setToolTip("Minimum target behavior score (Behavior A).")

        min_score_other = QDoubleSpinBox()
        min_score_other.setRange(0.1, 0.99)
        min_score_other.setSingleStep(0.05)
        min_score_other.setDecimals(2)
        min_score_other.setValue(0.35)
        min_score_other.setToolTip("Minimum competing behavior score (Behavior B or all other behaviors).")

        top_n = QSpinBox()
        top_n.setRange(10, 2000)
        top_n.setValue(200)

        form.addRow("Behavior A:", behavior_a)
        form.addRow("Behavior B:", behavior_b)
        form.addRow("Min. score (target):", min_score_target)
        form.addRow("Min. score (other):", min_score_other)
        form.addRow("Max results:", top_n)

        result_table = QTableWidget(0, 6)
        result_table.setHorizontalHeaderLabels(["Subject", "Session", "Frames", "Score A", "Score B", "Margin"])
        result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        result_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        result_table.horizontalHeader().setStretchLastSection(True)
        result_table.setMinimumHeight(280)

        status_label = QLabel("Configure parameters above and click Find.")
        edge_rows: list[dict[str, Any]] = []

        def _behavior_id_from_model_key(model_key: str) -> str | None:
            name = str(model_key or "").strip()
            if name.startswith("behavior_model_"):
                name = name[len("behavior_model_") :]
            name = re.sub(r"_\d{8}_\d{6}$", "", name)
            name = re.sub(r"_v\d+$", "", name)
            name = name.strip("_")
            return name or None

        def _run_search() -> None:
            nonlocal edge_rows
            model_key_a = str(behavior_a.currentData() or "").strip()
            model_key_b = str(behavior_b.currentData() or "").strip()
            if not model_key_a or model_key_a not in model_by_key:
                QMessageBox.warning(dlg, "Edge Case Finder", "Select Behavior A.")
                return
            if model_key_b != ALL_OTHER_KEY and (not model_key_b or model_key_b not in model_by_key):
                QMessageBox.warning(dlg, "Edge Case Finder", "Select Behavior B.")
                return
            if model_key_a == model_key_b:
                QMessageBox.warning(dlg, "Edge Case Finder", "Select two different behaviors.")
                return

            name_a, path_a = model_by_key[model_key_a]
            behavior_a_id = _behavior_id_from_model_key(model_key_a)
            thresh_target = float(min_score_target.value())
            thresh_other = float(min_score_other.value())
            limit = int(top_n.value())

            try:
                df_a = pd.read_parquet(path_a, columns=["segment_id", "prediction_prob"]).rename(
                    columns={"prediction_prob": "score_a"}
                )
            except Exception as exc:
                QMessageBox.warning(dlg, "Edge Case Finder", f"Failed to load predictions:\n{exc}")
                return

            merged = pd.DataFrame()
            behavior_b_label = ""
            if model_key_b == ALL_OTHER_KEY:
                merged_frames: list[pd.DataFrame] = []
                for other_key, (other_name, other_path) in model_by_key.items():
                    if other_key == model_key_a:
                        continue
                    try:
                        df_b = pd.read_parquet(other_path, columns=["segment_id", "prediction_prob"]).rename(
                            columns={"prediction_prob": "score_b"}
                        )
                    except Exception:
                        continue
                    merged_pair = df_a.merge(df_b, on="segment_id", how="inner")
                    if merged_pair.empty:
                        continue
                    merged_pair["behavior_b_name"] = other_name
                    merged_frames.append(merged_pair)
                if not merged_frames:
                    status_label.setText("No overlapping segments found between Behavior A and other models.")
                    result_table.setRowCount(0)
                    edge_rows = []
                    send_btn.setEnabled(False)
                    return
                merged = pd.concat(merged_frames, ignore_index=True)
                behavior_b_label = "all other behaviors"
            else:
                name_b, path_b = model_by_key[model_key_b]
                try:
                    df_b = pd.read_parquet(path_b, columns=["segment_id", "prediction_prob"]).rename(
                        columns={"prediction_prob": "score_b"}
                    )
                except Exception as exc:
                    QMessageBox.warning(dlg, "Edge Case Finder", f"Failed to load predictions:\n{exc}")
                    return
                merged = df_a.merge(df_b, on="segment_id", how="inner")
                merged["behavior_b_name"] = name_b
                behavior_b_label = name_b

            if merged.empty:
                status_label.setText("No overlapping segments found between the two models.")
                result_table.setRowCount(0)
                edge_rows = []
                send_btn.setEnabled(False)
                return

            # Keep windows where both sides clear their own minimum score thresholds.
            score_a = merged["score_a"].to_numpy(dtype=float)
            score_b = merged["score_b"].to_numpy(dtype=float)
            both_above = (score_a >= thresh_target) & (score_b >= thresh_other)
            margin = np.abs(score_a - score_b)
            mask = both_above

            if not np.any(mask):
                status_label.setText(
                    f"No competing windows found with target ≥ {thresh_target:.2f} and other ≥ {thresh_other:.2f}."
                )
                result_table.setRowCount(0)
                edge_rows = []
                send_btn.setEnabled(False)
                return

            candidates_df = merged[mask].copy()
            candidates_df["margin"] = margin[mask]
            candidates_df["competition_score"] = np.minimum(score_a[mask], score_b[mask]) - candidates_df["margin"]
            candidates_df = candidates_df.sort_values(["margin", "competition_score"], ascending=[True, False])
            if "segment_id" in candidates_df.columns:
                candidates_df = candidates_df.drop_duplicates(subset=["segment_id"], keep="first")
            candidates_df = candidates_df.head(limit)

            # Enrich with spatial metadata (start/end frame, session, subject)
            seg_path = self._project_root / "derived" / "representations" / "segment_features.parquet"  # type: ignore[union-attr]
            meta_cols = ["segment_id", "start_frame", "end_frame", "session_id", "animal_id"]
            seg_meta = pd.DataFrame()
            if seg_path.exists():
                try:
                    seg_meta = pd.read_parquet(seg_path, columns=meta_cols)
                except Exception:
                    try:
                        full_meta = pd.read_parquet(seg_path)
                        load_cols = [c for c in meta_cols if c in full_meta.columns]
                        if load_cols:
                            seg_meta = full_meta[load_cols].copy()
                    except Exception:
                        pass

            if not seg_meta.empty:
                candidates_df = candidates_df.merge(seg_meta, on="segment_id", how="left")

            subject_map = self._subject_map()
            edge_rows = []

            result_table.setRowCount(0)
            for _, row in candidates_df.iterrows():
                r = result_table.rowCount()
                result_table.insertRow(r)
                session_id = str(row.get("session_id", ""))
                subject = subject_map.get(session_id, session_id)
                start_f = int(row.get("start_frame", 0)) if "start_frame" in row.index else 0
                end_f = int(row.get("end_frame", 0)) if "end_frame" in row.index else 0
                segment_id = str(row.get("segment_id", "")).strip()
                result_table.setItem(r, 0, QTableWidgetItem(subject))
                result_table.setItem(r, 1, QTableWidgetItem(session_id))
                result_table.setItem(r, 2, QTableWidgetItem(f"{start_f}–{end_f}"))
                result_table.setItem(r, 3, QTableWidgetItem(f"{row['score_a']:.3f}"))
                result_table.setItem(r, 4, QTableWidgetItem(f"{row['score_b']:.3f}"))
                result_table.setItem(r, 5, QTableWidgetItem(f"{row['margin']:.3f}"))
                edge_rows.append(
                    {
                        "segment_id": segment_id,
                        "session_id": session_id,
                        "start_frame": int(start_f),
                        "end_frame": int(end_f),
                        "score_a": float(row["score_a"]),
                        "score_b": float(row["score_b"]),
                        "margin": float(row["margin"]),
                        "competition_score": float(row["competition_score"]),
                        "behavior_id": behavior_a_id,
                        "behavior_b_name": str(row.get("behavior_b_name", "")),
                    }
                )

            send_btn.setEnabled(bool(edge_rows))

            status_label.setText(
                f"Found {len(candidates_df)} competing window(s) where '{name_a}' and '{behavior_b_label}' "
                f"met target ≥ {thresh_target:.2f} and other ≥ {thresh_other:.2f}. "
                "Sorted by ambiguity (lowest margin first)."
            )

        def _send_to_clip_extraction() -> None:
            if not edge_rows:
                QMessageBox.information(dlg, "Edge Case Finder", "No edge-case windows to send.")
                return

            selected_rows = sorted({idx.row() for idx in result_table.selectionModel().selectedRows()})
            use_rows = selected_rows if selected_rows else list(range(len(edge_rows)))
            selected = [edge_rows[i] for i in use_rows if 0 <= i < len(edge_rows)]
            if not selected:
                QMessageBox.information(dlg, "Edge Case Finder", "No valid rows selected.")
                return

            sent: list[CandidateWindow] = []
            for row in selected:
                segment_id = str(row.get("segment_id", "")).strip()
                session_id = str(row.get("session_id", "")).strip()
                start_f = int(row.get("start_frame", 0))
                end_f = int(row.get("end_frame", 0))
                if not session_id:
                    continue
                window_id = segment_id or f"edge_{session_id}_{start_f}_{end_f}"
                score = float(row.get("competition_score", min(float(row.get("score_a", 0.0)), float(row.get("score_b", 0.0)))))
                sent.append(
                    CandidateWindow(
                        window_id=window_id,
                        session_id=session_id,
                        start_frame=start_f,
                        end_frame=end_f,
                        behavior_id=(str(row.get("behavior_id") or "").strip() or None),
                        seed_similarity_score=0.0,
                        total_score=float(np.clip(score, 0.0, 1.0)),
                    )
                )

            if not sent:
                QMessageBox.warning(dlg, "Edge Case Finder", "Selected rows are missing session/frame data.")
                return

            self.edge_case_candidates_requested.emit(
                sent,
                "Edge Case Finder competing windows",
            )
            QMessageBox.information(
                dlg,
                "Edge Case Finder",
                f"Sent {len(sent)} window(s) to Clip Extraction."
            )

        find_btn = QPushButton("Find Competing Windows")
        find_btn.clicked.connect(_run_search)
        send_btn = QPushButton("Send to Clip Extraction")
        send_btn.setEnabled(False)
        send_btn.clicked.connect(_send_to_clip_extraction)
        close_btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_btn.rejected.connect(dlg.reject)

        action_row = QHBoxLayout()
        action_row.addWidget(find_btn)
        action_row.addWidget(send_btn)
        action_row.addStretch()

        layout = QVBoxLayout(dlg)
        layout.addLayout(form)
        layout.addLayout(action_row)
        layout.addWidget(status_label)
        layout.addWidget(result_table, 1)
        layout.addWidget(close_btn)

        dlg.exec()

    def _on_retrain_finished(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        self._refresh_saved_model_options()
        if not payload or not payload.get("retrained", False):
            self._status.setText("No usable project review labels found. Retraining skipped.")
            self._progress.setFormat("No updates")
            self._append_log("Retraining skipped: no usable project review labels were available.")
            return
        metrics = payload.get("metrics", {})
        target_behavior = str(payload.get("target_behavior", "target_behavior"))
        model_version = str(payload.get("model_version", "")).strip()
        # Persist workflow snapshot.
        self._save_workflow_snapshot(
            model_version=model_version,
            target_behavior=target_behavior,
        )
        self._snapshot_evaluation_graphs_for_model(model_version, target_behavior=target_behavior)
        self._refresh_viz_model_options()
        model_disp = self._display_model_name(model_version)
        candidates = payload.get("candidates", [])
        if not candidates:
            # Fallback to persisted candidates so UI stays in sync even if payload is minimal.
            try:
                candidates = self._candidates.load_segment_candidates()
            except Exception:
                candidates = []
        self._status.setText(
            f"Retrained model {model_disp} ({target_behavior}). F1={metrics.get('f1', 0.0):.3f} "
            f"PR-AUC={metrics.get('pr_auc', 0.0):.3f} | train={int(metrics.get('n_train', 0))} "
            f"val={int(metrics.get('n_val', 0))} | candidates={int(payload.get('n_candidates', 0))} | "
            f"{self._quality_explanation(metrics)}"
        )
        self._progress.setValue(self._progress.maximum())
        self._progress.setFormat("Complete")
        self._append_log(
            "Retraining complete. "
            f"model={model_disp}, "
            f"target={target_behavior}, "
            f"F1={float(metrics.get('f1', 0.0)):.3f}, "
            f"PR-AUC={float(metrics.get('pr_auc', 0.0)):.3f}, "
            f"train={int(metrics.get('n_train', 0))}, val={int(metrics.get('n_val', 0))}, "
            f"candidates={int(payload.get('n_candidates', 0))}."
        )
        self._append_log(self._quality_explanation(metrics))
        self._populate_candidate_table(candidates)
        self._emit_uncertainty_candidates_for_clip_extraction(
            candidates,
            source_label=f"Active Learning \u2014 Retrain ({target_behavior})",
        )
        self._refresh_visualization_preview()
        if self._is_umap_enabled():
            self._regenerate_unified_umap_inline()
        else:
            self._append_log("UMAP generation disabled in settings — skipping.")

    def _on_failed(self, traceback_text: str) -> None:
        self._set_busy(False)
        self._cancel_flag[0] = False
        if "PIPELINE_CANCELLED_BY_USER" in traceback_text:
            self._status.setText("Active-learning run stopped.")
            self._progress.setFormat("Stopped")
            self._append_log("Pipeline stopped by user.")
            return
        self._status.setText("Active-learning pipeline failed. Check logs for traceback details.")
        self._progress.setFormat("Failed")
        logger.error("Active learning pipeline failed:\n%s", traceback_text)
        self._append_log("ERROR: pipeline failed. See logs tab for traceback.")
        QMessageBox.critical(self, "Active Learning", "Active-learning run failed. Check logs for details.")

    def _on_task_error(self, task_name: str, traceback_text: str) -> None:
        self._set_busy(False)
        self._status.setText(f"{task_name} failed.")
        self._append_log(f"ERROR: {task_name} failed. {traceback_text[:200]}")
        logger.error("%s failed:\n%s", task_name, traceback_text)

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

    def _set_viz_source_path(self, path: Path | None) -> None:
        self._viz_source_path = path
        self._viz_save_btn.setEnabled(path is not None and path.exists())

    @staticmethod
    def _available_visualization_exports(path: Path | None) -> dict[str, Path]:
        if path is None or not path.exists():
            return {}
        available: dict[str, Path] = {}
        stem_path = path.with_suffix("")
        for candidate in [path, stem_path.with_suffix(".svg"), stem_path.with_suffix(".pdf")]:
            suffix = candidate.suffix.lower()
            if suffix and candidate.exists() and suffix not in available:
                available[suffix] = candidate
        return available

    def _save_visualization(self) -> None:
        available = self._available_visualization_exports(self._viz_source_path)
        if not available:
            QMessageBox.information(self, "Save Visualization", "No visualization is currently available to save.")
            return

        filters: list[str] = []
        if ".svg" in available:
            filters.append("SVG (*.svg)")
        if ".pdf" in available:
            filters.append("PDF (*.pdf)")
        if ".png" in available:
            filters.append("PNG (*.png)")
        filters.append("All Files (*)")

        preferred = available.get(".svg") or available.get(".pdf") or available.get(".png")
        default_name = preferred.name if preferred is not None else "visualization.png"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Visualization",
            default_name,
            ";;".join(filters),
        )
        if not path_str:
            return

        target = Path(path_str)
        chosen_suffix = target.suffix.lower()
        source = available.get(chosen_suffix)
        if source is None:
            source = preferred
            if source is None:
                QMessageBox.warning(self, "Save Visualization", "Could not determine which source file to save.")
                return
            if not target.suffix:
                target = target.with_suffix(source.suffix)

        try:
            shutil.copy2(source, target)
            self._status.setText(f"Saved visualization to {target}")
            self._append_log(f"Visualization saved: {target}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Visualization", str(exc))

    def _update_viz_help_tooltip(self) -> None:
        selected = str(self._viz_selector.currentData() or "auto")
        if selected == "umap":
            text = (
                "Behavior Separation plot:\n"
                "- Each point is a segment in reduced 2-D feature space.\n"
                "- Tight target cluster(s) with separation from others is favorable.\n"
                "- Strong overlap suggests feature ambiguity or label noise.\n"
                "- Orientation/axis values are not directly interpretable; relative structure is what matters."
            )
        elif selected == "confusion":
            text = (
                "Confusion Matrix:\n"
                "- Rows are true labels, columns are predicted labels.\n"
                "- Top-left: true Other, top-right: false Target (false positive).\n"
                "- Bottom-left: missed Target (false negative), bottom-right: true Target.\n"
                "- Better models concentrate counts on the diagonal."
            )
        elif selected == "pr":
            text = (
                "Precision-Recall curve:\n"
                "- Shows precision/recall tradeoff across score thresholds.\n"
                "- Curves closer to top-right are better.\n"
                "- PR-AUC summarizes ranking quality, especially for imbalanced labels.\n"
                "- Use this to choose a threshold for your tolerance of false positives vs misses."
            )
        elif selected == "feature_family":
            text = (
                "Feature-family comparison:\n"
                "- Compares AP/F1 across modality experts.\n"
                "- Helps verify whether behavior-adaptive modality weighting is useful."
            )
        elif selected == "multiscale":
            text = (
                "Multi-scale performance:\n"
                "- Plots validation AP versus temporal window size.\n"
                "- Identifies whether short or longer windows separate this behavior better."
            )
        elif selected == "margin":
            text = (
                "Target-vs-confound margin:\n"
                "- Distribution of target score minus top-confound score for TP/FP/FN groups.\n"
                "- Larger positive margins indicate better target/confound separation."
            )
        elif selected == "calibration":
            text = (
                "Calibration reliability:\n"
                "- Predicted probability versus observed correctness.\n"
                "- Includes ECE to quantify confidence quality."
            )
        elif selected == "queue":
            text = (
                "Queue composition:\n"
                "- Shows which component primarily selected clips in the latest queue.\n"
                "- Helps verify weighted queue scoring behavior."
            )
        elif selected == "timing":
            text = (
                "Pipeline Timing:\n"
                "- Horizontal bar chart of how long each pipeline step took.\n"
                "- The longest bar is highlighted in dark blue; others in light blue.\n"
                "- Total runtime is shown in the x-axis label.\n"
                "- Generated automatically at the end of each full pipeline run."
            )
        elif selected == "confound_cross":
            text = (
                "Cross-behaviour confound matrix:\n"
                "- NxN heatmap of co-activation rates between behaviour models.\n"
                "- High off-diagonal values indicate overlapping predictions.\n"
                "- Click 'Confound Analysis' to generate or refresh."
            )
        elif selected == "unified_umap":
            text = (
                "Unified UMAP (all behaviours):\n"
                "- Single 2-D embedding coloured by dominant behaviour.\n"
                "- Helps visualise how well behaviours separate in feature space.\n"
                "- Click 'Unified UMAP' to generate or refresh."
            )
        elif selected == "unsupervised_umap":
            text = (
                "Unsupervised UMAP (clusters):\n"
                "- 2-D embedding built directly from raw segment features.\n"
                "- No models or labels required; colours are auto-discovered clusters (HDBSCAN).\n"
                "- Use 'Select from UMAP' to lasso a cluster and send it to clip extraction.\n"
                "- Click 'Unsupervised UMAP' to generate or refresh."
            )
        else:
            text = (
                "Auto graph mode:\n"
                "- Shows the first available artifact in this order: Separation, Confusion, PR.\n"
                "- Select a specific graph to lock the view to that artifact type."
            )
        self._viz_help.setToolTip(text)

    def _on_viz_selection_changed(self, _index: int) -> None:
        self._update_viz_help_tooltip()
        selected = str(self._viz_selector.currentData() or "auto")
        if selected == "expert_assignment" and self._project_root is not None:
            dest = self._project_root / "derived" / "evaluation" / "expert_assignment_per_model.png"
            if not dest.exists():
                self._generate_expert_assignment_chart()
        self._refresh_visualization_preview()

    def _refresh_visualization_preview(self) -> None:
        if self._project_root is None:
            self._set_viz_source_path(None)
            self._viz_pixmap_original = None
            self._viz_preview.setPixmap(QPixmap())
            self._viz_preview.setText("Open a project to view behavior separation previews.")
            return

        base_eval_dir = self._project_root / "derived" / "evaluation"
        chosen_model = str(self._viz_model_selector.currentData() or "__latest__").strip()
        if chosen_model and chosen_model != "__latest__":
            model_eval_dir = base_eval_dir / "by_model" / chosen_model
            eval_dir = model_eval_dir if model_eval_dir.exists() else base_eval_dir
        else:
            eval_dir = base_eval_dir
        target_behavior = self._selected_target_behavior_id()
        safe_behavior = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in target_behavior) or "target_behavior"
        diag_root = self._project_root / "derived" / "analysis" / "diagnostics" / safe_behavior
        latest_diag = read_json(diag_root / "latest.json", {}) if (diag_root / "latest.json").exists() else {}
        latest_diag_dir = Path(str(latest_diag.get("diagnostic_dir", ""))) if latest_diag.get("diagnostic_dir") else None
        queue_diag_root = self._project_root / "derived" / "analysis" / "diagnostics" / "queue"
        latest_queue = read_json(queue_diag_root / "latest.json", {}) if (queue_diag_root / "latest.json").exists() else {}
        queue_run_path = Path(str(latest_queue.get("queue_composition", ""))) if latest_queue.get("queue_composition") else None

        def _diag_file(name: str) -> Path:
            if chosen_model and chosen_model != "__latest__":
                model_path = base_eval_dir / "by_model" / chosen_model / name
                if model_path.exists():
                    return model_path
            if latest_diag_dir is not None:
                return latest_diag_dir / name
            return diag_root / name

        def _queue_file(name: str) -> Path:
            if chosen_model and chosen_model != "__latest__":
                model_path = base_eval_dir / "by_model" / chosen_model / name
                if model_path.exists():
                    return model_path
            if queue_run_path is not None:
                return queue_run_path.parent / name
            return queue_diag_root / name

        selected = str(self._viz_selector.currentData() or "auto")
        path_map = {
            "umap": [eval_dir / "unified_behavior_umap.png"],
            "confusion": [eval_dir / "confusion_matrix.png"],
            "pr": [eval_dir / "PR_curve.png"],
            "feature_family": [_diag_file("feature_family_comparison.png")],
            "multiscale": [_diag_file("multiscale_performance.png")],
            "margin": [_diag_file("target_confound_margin_histogram.png")],
            "calibration": [_diag_file("calibration_reliability_phase1.png")],
            "queue": [_queue_file("queue_composition.png")],
            "timing": [eval_dir / "pipeline_timing.png"],
            "confound_cross": [eval_dir / "cross_behavior_confound_matrix.png"],
            "unified_umap": [eval_dir / "unified_behavior_umap.png"],
            "unsupervised_umap": [eval_dir / "unsupervised_umap.png"],
            "expert_assignment": [eval_dir / "expert_assignment_per_model.png"],
            "auto": [
                eval_dir / "unified_behavior_umap.png",
                eval_dir / "confusion_matrix.png",
                eval_dir / "PR_curve.png",
                _diag_file("feature_family_comparison.png"),
                _diag_file("multiscale_performance.png"),
                _diag_file("target_confound_margin_histogram.png"),
                _diag_file("calibration_reliability_phase1.png"),
                _queue_file("queue_composition.png"),
            ],
        }
        candidates = path_map.get(selected, path_map["auto"])
        image_path = next((p for p in candidates if p.exists()), None)
        if image_path is None:
            self._set_viz_source_path(None)
            self._viz_pixmap_original = None
            self._viz_preview.setPixmap(QPixmap())
            model_disp = "Latest run" if chosen_model == "__latest__" else self._display_model_name(chosen_model)
            if selected == "auto":
                self._viz_preview.setText(
                    f"No evaluation graph found yet for model '{model_disp}'. Run training/evaluation to populate this panel."
                )
            else:
                self._viz_preview.setText(
                    f"Selected graph is not available yet for model '{model_disp}'."
                )
            return

        pix = QPixmap(str(image_path))
        if pix.isNull():
            self._set_viz_source_path(None)
            self._viz_pixmap_original = None
            self._viz_preview.setPixmap(QPixmap())
            self._viz_preview.setText(f"Could not load graph image: {image_path.name}")
            return

        self._set_viz_source_path(image_path)
        self._viz_pixmap_original = pix
        self._viz_preview.setText("")
        self._render_visualization_pixmap()

    def _show_image_in_viz_preview(self, path: Path) -> None:
        """Display an arbitrary image file in the visualisation preview label."""
        pix = QPixmap(str(path))
        if pix.isNull():
            return
        self._set_viz_source_path(path)
        self._viz_pixmap_original = pix
        self._viz_preview.setText("")
        self._render_visualization_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_visualization_pixmap()

    @Slot(int, int, str, str)
    def _apply_pipeline_progress(self, value: int, maximum: int, log_line: str, status: str) -> None:
        maximum = max(1, int(maximum))
        value = max(0, min(int(value), maximum))
        self._progress.setRange(0, maximum)
        self._progress.setValue(value)
        # Extract finish time from log_line for the progress bar label.
        # log_line contains "... | finish ~ HH:MM:SS" when ETA is known.
        _scale = max(1, getattr(self, "_pipeline_step_scale", 1))
        _dv = round(value / _scale, 1)
        _dm = maximum // _scale
        _bar_label = f"{_dv}/{_dm}"
        if "finish ~" in log_line:
            try:
                _finish = log_line.split("finish ~", 1)[1].strip().split()[0]
                _bar_label = f"{_dv}/{_dm}  ·  done ~ {_finish}"
            except Exception:
                pass
        elif value == 0:
            _bar_label = "Running…"
        self._progress.setFormat(_bar_label)
        self._status.setText(status)
        if log_line:
            self._append_log(log_line)
