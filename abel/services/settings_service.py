"""Global app settings and recent-project management."""

from __future__ import annotations

from pathlib import Path

from abel.core.constants import GLOBAL_SETTINGS_PATH, RECENT_PROJECTS_PATH
from abel.models.schemas import AppSettings
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml
from abel.utils.paths import ensure_global_dirs


class SettingsService:
    """Handles global settings and recent project list persistence."""

    def __init__(self) -> None:
        ensure_global_dirs()

    def load_app_settings(self) -> AppSettings:
        raw = read_yaml(GLOBAL_SETTINGS_PATH, {})
        if not raw:
            settings = AppSettings()
            self.save_app_settings(settings)
            return settings
        return AppSettings.model_validate(raw)

    def save_app_settings(self, settings: AppSettings) -> None:
        write_yaml(GLOBAL_SETTINGS_PATH, settings.model_dump(mode="json"))

    def load_recent_projects(self) -> list[str]:
        raw = read_json(RECENT_PROJECTS_PATH, {"recent_projects": []})
        return list(raw.get("recent_projects", []))

    def add_recent_project(self, path: Path, max_items: int = 12) -> list[str]:
        existing = [p for p in self.load_recent_projects() if p != str(path)]
        merged = [str(path), *existing][:max_items]
        write_json(RECENT_PROJECTS_PATH, {"recent_projects": merged})
        return merged
