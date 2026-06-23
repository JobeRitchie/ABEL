"""Centralized app/project logging setup."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from abel.core.constants import GLOBAL_LOG_DIR
from abel.utils.paths import ensure_global_dirs


class LoggingService:
    """Configures rotating logs for app and project scopes."""

    def __init__(self) -> None:
        self._configured = False

    def configure_app_logging(self) -> logging.Logger:
        ensure_global_dirs()
        logger = logging.getLogger("abel")
        logger.setLevel(logging.INFO)
        if self._configured:
            return logger

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        file_handler = RotatingFileHandler(
            GLOBAL_LOG_DIR / "abel_app.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        self._configured = True
        return logger

    def attach_project_handler(self, project_root: Path) -> None:
        logger = logging.getLogger("abel")
        project_log_dir = project_root / "logs"
        project_log_dir.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler = RotatingFileHandler(
            project_log_dir / "project.log",
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        handler.name = f"project_log_{project_root}"

        if not any(h.name == handler.name for h in logger.handlers if hasattr(h, "name")):
            logger.addHandler(handler)
