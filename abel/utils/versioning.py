"""Simple schema version helpers and migration placeholders."""

from __future__ import annotations

from abel.core.constants import APP_SCHEMA_VERSION, PROJECT_SCHEMA_VERSION


def ensure_app_schema(schema_version: str) -> bool:
    return schema_version == APP_SCHEMA_VERSION


def ensure_project_schema(schema_version: str) -> bool:
    return schema_version == PROJECT_SCHEMA_VERSION
