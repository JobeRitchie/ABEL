"""Helpers for reproducible artifact metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from abel.core.constants import APP_SCHEMA_VERSION
from abel.models.schemas import ArtifactProvenance


class ProvenanceService:
    """Build immutable provenance stamps for derived artifacts."""

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert common Python objects into deterministic JSON-safe values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, dict):
            return {str(k): ProvenanceService._json_safe(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [ProvenanceService._json_safe(v) for v in value]

        if isinstance(value, (set, frozenset)):
            items = [ProvenanceService._json_safe(v) for v in value]
            # Deterministic ordering independent of hash randomization.
            return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))

        if hasattr(value, "model_dump"):
            try:
                return ProvenanceService._json_safe(value.model_dump(mode="json"))
            except Exception:
                pass

        if hasattr(value, "item"):
            try:
                return ProvenanceService._json_safe(value.item())
            except Exception:
                pass

        return str(value)

    @staticmethod
    def config_hash(config: dict[str, Any]) -> str:
        payload = json.dumps(
            ProvenanceService._json_safe(config),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def git_commit_hash(project_root: Path) -> str:
        try:
            out = subprocess.check_output(
                ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return out.strip() or "unknown"
        except Exception:
            return "unknown"

    def make_provenance(
        self,
        project_root: Path,
        model_version: str,
        feature_version: str,
        config: dict[str, Any],
    ) -> ArtifactProvenance:
        return ArtifactProvenance(
            app_version=APP_SCHEMA_VERSION,
            git_commit_hash=self.git_commit_hash(project_root),
            model_version=model_version,
            feature_version=feature_version,
            config_hash=self.config_hash(config),
            timestamp=datetime.utcnow(),
        )
