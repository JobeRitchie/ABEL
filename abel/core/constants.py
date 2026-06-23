"""Application-wide constants."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "ABEL"
APP_SCHEMA_VERSION = "0.2.0"
PROJECT_SCHEMA_VERSION = "0.2.0"

GLOBAL_CONFIG_DIR = Path.home() / ".abel"
GLOBAL_LOG_DIR = GLOBAL_CONFIG_DIR / "logs"
GLOBAL_SETTINGS_PATH = GLOBAL_CONFIG_DIR / "app_settings.yaml"
RECENT_PROJECTS_PATH = GLOBAL_CONFIG_DIR / "recent_projects.json"
DEPENDENCY_CACHE_PATH = GLOBAL_CONFIG_DIR / "dependency_state.json"

PROJECT_DIRS = [
    "logs",
    "config",
    "raw/videos",
    "raw/pose",
    "derived/pose_clean",
    "derived/pose_features",
    "derived/context_features",
    "derived/representations",
    "derived/models",
    "derived/training_sets",
    "derived/review_labels",
    "derived/evaluation",
    "derived/behavior_bouts",
    "derived/windows",
    "derived/clips",
    "derived/crops",
    "derived/stabilized",
    "derived/features",
    "derived/review_tables",
    "derived/thumbnails",
    "derived/caches",
    "derived/analysis/benchmarks",
    "derived/analysis/diagnostics",
    "derived/temporal_refinement",
    "exports/csv",
    "exports/parquet",
    "exports/reports",
    "temp",
    "backups",
]

PROJECT_CONFIG_FILES = {
    "app_settings": "config/app_settings.yaml",
    "behavior_definitions": "config/behavior_definitions.yaml",
    "preprocessing": "config/preprocessing.yaml",
    "model_backends": "config/model_backends.yaml",
    "export_settings": "config/export_settings.yaml",
    "experiment": "config/experiment.yaml",
    "environment_rois": "config/environment_rois.yaml",
    "behavior_adaptive_settings": "config/behavior_adaptive_settings.yaml",
}

DEFAULT_REQUIRED_MINIMAL_DEPENDENCIES = {
    "PySide6": ">=6.7",
    "pydantic": ">=2.7",
    "numpy": ">=1.26",
    "pandas": ">=2.2",
    "PyYAML": ">=6.0",
}
