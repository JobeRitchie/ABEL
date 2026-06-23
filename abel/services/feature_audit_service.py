"""Feature audit service — detect bodyparts, identify dead features, and recommend exclusions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abel.services.pose_processing_service import PoseProcessingService, normalize_bodypart_name
from abel.storage.file_store import write_json

logger = logging.getLogger("abel")


@dataclass
class BodypartReport:
    """Summary of a single detected bodypart across sessions."""

    name: str
    sessions_present: int
    sessions_total: int
    mean_likelihood: float
    low_likelihood_fraction: float  # fraction of frames < 0.2

    @property
    def coverage(self) -> float:
        return self.sessions_present / max(1, self.sessions_total)


@dataclass
class FeatureHealthReport:
    """Health summary for a single feature column."""

    name: str
    nonzero_fraction: float  # fraction of rows that are not zero
    nan_fraction: float
    std: float
    family: str  # pose / motion / context
    source_bodypart: str  # "" if not bodypart-specific

    @property
    def is_dead(self) -> bool:
        return self.std < 1e-12 or self.nan_fraction > 0.99

    @property
    def is_weak(self) -> bool:
        return not self.is_dead and self.nonzero_fraction < 0.05


@dataclass
class FeatureAuditResult:
    """Full audit result for a project."""

    bodyparts: list[BodypartReport] = field(default_factory=list)
    features: list[FeatureHealthReport] = field(default_factory=list)
    dead_feature_names: list[str] = field(default_factory=list)
    weak_feature_names: list[str] = field(default_factory=list)
    recommended_exclusions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bodyparts": [
                {
                    "name": b.name,
                    "sessions_present": b.sessions_present,
                    "sessions_total": b.sessions_total,
                    "mean_likelihood": round(b.mean_likelihood, 4),
                    "low_likelihood_fraction": round(b.low_likelihood_fraction, 4),
                    "coverage": round(b.coverage, 4),
                }
                for b in self.bodyparts
            ],
            "dead_features": self.dead_feature_names,
            "weak_features": self.weak_feature_names,
            "recommended_exclusions": self.recommended_exclusions,
            "n_total_features": len(self.features),
            "n_dead": len(self.dead_feature_names),
            "n_weak": len(self.weak_feature_names),
        }


class FeatureAuditService:
    """Scan a project's pose files and derived features to identify dead/weak columns."""

    def __init__(self) -> None:
        self._pose_service = PoseProcessingService()

    # ------------------------------------------------------------------
    # Bodypart detection
    # ------------------------------------------------------------------

    def _collect_pose_files(self, project_root: Path) -> list[Path]:
        """Gather all available pose files from local storage and the import manifest."""
        pose_files: list[Path] = []
        seen: set[str] = set()

        # 1. Pose files in the project's raw/pose/ directory
        pose_dir = project_root / "raw" / "pose"
        if pose_dir.exists():
            for pf in sorted(pose_dir.glob("*.csv")) + sorted(pose_dir.glob("*.h5")):
                resolved = str(pf.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    pose_files.append(pf)

        # 2. Pose files referenced in the import manifest (may be external)
        manifest_path = project_root / "derived" / "review_tables" / "import_manifest.json"
        if manifest_path.exists():
            try:
                from abel.storage.file_store import read_json
                manifest_data = read_json(manifest_path, {})
                for pose_entry in manifest_data.get("poses", []):
                    for key in ("local_path", "source_path"):
                        raw = pose_entry.get(key)
                        if not raw:
                            continue
                        p = Path(raw)
                        resolved = str(p.resolve())
                        if resolved not in seen and p.exists():
                            seen.add(resolved)
                            pose_files.append(p)
                            break  # prefer local_path over source_path
            except Exception as exc:
                logger.debug("Could not read import manifest for pose files: %s", exc)

        return pose_files

    def detect_bodyparts(
        self,
        project_root: Path,
        session_registry: dict[str, Any] | None = None,
    ) -> list[BodypartReport]:
        """Scan all pose files and report which bodyparts are present and their quality."""
        pose_files = self._collect_pose_files(project_root)
        if not pose_files:
            return []

        # Track bodypart presence and quality across sessions.
        bodypart_sessions: dict[str, int] = {}
        bodypart_likelihoods: dict[str, list[float]] = {}
        bodypart_low_frac: dict[str, list[float]] = {}
        n_sessions = len(pose_files)

        for pf in pose_files:
            try:
                pose = self._pose_service.load(pf)
            except Exception as exc:
                logger.warning("Feature audit: skipping %s: %s", pf.name, exc)
                continue

            for bp in pose.body_parts:
                bodypart_sessions[bp] = bodypart_sessions.get(bp, 0) + 1
                lk = np.asarray(pose.likelihood[bp], dtype=float)
                mean_lk = float(np.nanmean(lk)) if len(lk) > 0 else 0.0
                low_frac = float(np.mean(lk < 0.2)) if len(lk) > 0 else 1.0
                bodypart_likelihoods.setdefault(bp, []).append(mean_lk)
                bodypart_low_frac.setdefault(bp, []).append(low_frac)

        reports = []
        for bp in sorted(bodypart_sessions.keys()):
            reports.append(BodypartReport(
                name=bp,
                sessions_present=bodypart_sessions[bp],
                sessions_total=n_sessions,
                mean_likelihood=float(np.mean(bodypart_likelihoods.get(bp, [0.0]))),
                low_likelihood_fraction=float(np.mean(bodypart_low_frac.get(bp, [1.0]))),
            ))
        return reports

    # ------------------------------------------------------------------
    # Feature health audit
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_family(col: str) -> str:
        cl = col.lower()
        motion_keys = ("flow", "motion", "velocity", "acceleration", "jerk")
        context_keys = ("target", "dist", "zone", "roi", "occup", "context", "surface")
        if any(k in cl for k in motion_keys):
            return "motion"
        if any(k in cl for k in context_keys):
            return "context"
        return "pose"

    @staticmethod
    def _guess_bodypart(col: str) -> str:
        known = [
            "nose", "left_ear", "right_ear", "ear_left", "ear_right",
            "left_paw", "right_paw", "lateral_left", "lateral_right",
            "tail_base", "tail_tip", "spine", "head", "neck", "hip",
            "forepaw", "paw", "centroid", "body", "center", "centre",
            "snout", "shoulder", "knee", "ankle", "wrist", "elbow",
        ]
        cl = col.lower()
        for bp in known:
            if cl.startswith(bp) or f"_{bp}" in cl:
                return bp
        return ""

    def audit_features(
        self,
        project_root: Path,
        training_set_path: Path | None = None,
    ) -> FeatureAuditResult:
        """Audit feature health from the training set or segment features."""
        # Prefer training set if it exists; fall back to segment features.
        ts_path = training_set_path
        if ts_path is None:
            ts_path = project_root / "derived" / "training_sets" / "training_set.parquet"
        if not ts_path.exists():
            ts_path = project_root / "derived" / "representations" / "segment_features.parquet"
        if not ts_path.exists():
            return FeatureAuditResult()

        df = pd.read_parquet(ts_path)
        meta_cols = {"segment_id", "label", "label_source", "animal_id", "session_id",
                     "start_frame", "end_frame", "reviewer_confidence"}
        numeric_cols = [
            c for c in df.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        features: list[FeatureHealthReport] = []
        dead: list[str] = []
        weak: list[str] = []

        for col in numeric_cols:
            vals = df[col]
            n = len(vals)
            nan_frac = float(vals.isna().sum() / max(1, n))
            non_nan = vals.dropna()
            nonzero_frac = float((non_nan != 0).sum() / max(1, len(non_nan))) if len(non_nan) > 0 else 0.0
            col_std = float(non_nan.std()) if len(non_nan) > 1 else 0.0

            report = FeatureHealthReport(
                name=col,
                nonzero_fraction=nonzero_frac,
                nan_fraction=nan_frac,
                std=col_std,
                family=self._guess_family(col),
                source_bodypart=self._guess_bodypart(col),
            )
            features.append(report)
            if report.is_dead:
                dead.append(col)
            elif report.is_weak:
                weak.append(col)

        # Recommended exclusions: dead features always, weak features as advisory.
        recommended = sorted(set(dead))

        result = FeatureAuditResult(
            bodyparts=self.detect_bodyparts(project_root),
            features=features,
            dead_feature_names=sorted(dead),
            weak_feature_names=sorted(weak),
            recommended_exclusions=recommended,
        )
        return result

    def save_audit_report(self, project_root: Path, result: FeatureAuditResult) -> Path:
        """Persist the audit report as JSON."""
        out_dir = project_root / "derived" / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "feature_audit_report.json"
        write_json(out_path, result.to_dict())
        logger.info("Feature audit report saved to %s", out_path)
        return out_path

    def get_auto_exclusions(
        self,
        project_root: Path,
        training_set_path: Path | None = None,
    ) -> list[str]:
        """Return the list of features that should be auto-excluded (dead columns)."""
        result = self.audit_features(project_root, training_set_path)
        return result.recommended_exclusions

    @staticmethod
    def load_feature_importance(project_root: Path) -> dict[str, dict[str, float]]:
        """Load feature importance from all trained models.

        Returns a dict mapping model_version → {feature_name: importance_score}.
        Importance scores are aggregated across all models by averaging.
        """
        from abel.storage.file_store import read_json
        models_dir = project_root / "derived" / "models"
        if not models_dir.exists():
            return {}

        per_model: dict[str, dict[str, float]] = {}
        for model_dir in sorted(models_dir.iterdir()):
            imp_path = model_dir / "feature_importance.json"
            if imp_path.exists():
                try:
                    data = read_json(imp_path, {})
                    if isinstance(data, dict) and data:
                        per_model[model_dir.name] = {
                            str(k): float(v) for k, v in data.items()
                        }
                except Exception as exc:
                    logger.debug("Could not load feature importance from %s: %s", imp_path, exc)
        return per_model

    @staticmethod
    def aggregate_feature_importance(
        per_model: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Average feature importance across all models."""
        if not per_model:
            return {}
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for _model, imp in per_model.items():
            for feat, score in imp.items():
                totals[feat] = totals.get(feat, 0.0) + score
                counts[feat] = counts.get(feat, 0) + 1
        averaged = {
            feat: round(totals[feat] / counts[feat], 6)
            for feat in totals
        }
        return dict(sorted(averaged.items(), key=lambda kv: kv[1], reverse=True))
