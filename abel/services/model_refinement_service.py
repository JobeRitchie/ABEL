"""Model Refinement — import labeled examples from other ABEL projects.

Refining a model means giving it more labeled examples to learn from.  This
service pulls labeled segments (their *features* + the reviewer's *label*) out
of one or more source projects and merges them into the host project's
training set, after which the host's models can be retrained on the larger,
more diverse dataset.

What is and isn't imported
--------------------------
* Imported: the per-segment feature rows and their behaviour labels — the
  source project's *entire* labeled set, read straight from its assembled
  training set (``training_set.parquet``).  These are exactly what a model
  trains on.  Sources that were reviewed but never had a training set built
  fall back to the Review-tab label log (``reviewer_labels.parquet``).
* Also registered for review: each imported segment is surfaced in the Review
  tab as a *reviewed*, source-tagged entry, and its clip video is copied into
  this project so it can be played back here.  The model never sees the clip —
  this is purely so a human can see and audit what was imported and where it
  came from.

Hard constraint — feature-schema compatibility
----------------------------------------------
Two projects only produce comparable feature columns when they share the same
pose keypoint scheme (and ROI layout).  Different keypoints generate different
per-keypoint kinematics and O(n^2) pairwise-distance columns, so naively
merging incompatible projects would train a model on mostly-missing features.
``preview`` therefore reports column coverage and refuses the import when the
host's feature columns are not almost-entirely present in the source.

Keypoint-name remapping
-----------------------
Two projects can track the *same* physical keypoints under different names
(e.g. ``back_mid`` vs ``center_body``, ``ear_left`` vs ``left_ear``).  That
alone makes every derived feature column look mismatched.  Before the coverage
check, the source's feature columns are renamed onto the host's keypoint scheme
using the host's saved Direct Use map (``config/direct_use_keypoint_map.json``)
or, failing that, an auto-suggested keypoint mapping (see ``keypoint_mapping``).
Only the keypoint *tokens* inside column names are rewritten, so two
differently-named-but-identical schemes line up and import cleanly.

Behaviour identity
------------------
Behaviours are matched across projects by *name* (case-insensitive), mirroring
``ProjectMergeService``.  ``no_behavior`` maps to ``no_behavior``.  Source
behaviours with no host match are reported and skipped.

Name remapping
--------------
Different labs name the same behaviour differently — one project's "Dip" is
another's "Head Dip".  A per-host *alias table* (``config/behavior_aliases.json``)
maps a source behaviour *name* to the host behaviour *name* it should be treated
as, so otherwise-unmatched examples can still be imported.  The Model Refinement
tab edits this table through a helper dialog that auto-suggests likely matches.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abel.models.schemas import CandidateWindow, ReviewDecision, ReviewDecisionType
from abel.services import keypoint_mapping
from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.services.candidate_service import CandidateGenerationService
from abel.services.review_service import ReviewService
from abel.storage.file_store import read_json, write_json

logger = logging.getLogger("abel")

# Columns that are identifiers / bookkeeping rather than model features.
_NON_FEATURE_COLS = frozenset({
    "segment_id", "label", "label_source", "reviewer_confidence",
    "animal_id", "session_id", "start_frame", "end_frame",
    "overlap_allowed",
})

# Labels that are not behaviour UUIDs but are still valid training targets.
_PASSTHROUGH_LABELS = frozenset({"no_behavior"})

# Minimum fraction of host feature columns that must exist in the source
# project for the two to be considered compatible.
COMPAT_THRESHOLD = 0.95


@dataclass
class BehaviorMapping:
    """How one source behaviour maps onto the host project."""

    source_behavior_id: str
    source_name: str
    host_behavior_id: str  # "" when unmatched
    host_name: str         # "" when unmatched
    example_count: int
    remapped: bool = False  # matched via a manual name alias (not exact name)

    @property
    def matched(self) -> bool:
        return bool(self.host_behavior_id)


@dataclass
class CompatibilityDiagnostics:
    """Project-similarity signals beyond raw feature-column overlap.

    Schema compatibility (column names line up) does not guarantee the feature
    *values* mean the same thing across projects.  These metrics flag the
    upstream differences — calibration, pose model, extraction settings — and
    the net distribution shift they produce, so the user can judge whether
    merging is scientifically sound, not just mechanically allowed.
    """

    # Spatial calibration (pixels per mm); raw-pixel distances scale with this.
    host_px_per_mm: float | None = None
    source_px_per_mm: float | None = None
    px_per_mm_pct_diff: float | None = None  # |h-s| / mean, as a percentage

    # DLC network(s) that produced the pose; different nets can place the same
    # named keypoint slightly differently.
    host_pose_models: list[str] = field(default_factory=list)
    source_pose_models: list[str] = field(default_factory=list)
    pose_models_match: bool = True

    # Net feature-value shift over shared columns, in pooled-IQR units, with a
    # within-host baseline (≈0 == sampling noise) for reference.
    feature_shift_median: float | None = None
    feature_shift_p90: float | None = None
    feature_shift_frac_gt_half: float | None = None  # fraction of cols > 0.5 IQR
    within_host_shift_median: float | None = None

    # Feature-extraction settings that differ between the two project configs.
    config_mismatches: list[str] = field(default_factory=list)


@dataclass
class CoverageDiagnosis:
    """Human-readable explanation of *why* models can't be imported.

    Built when one or more source models fall below
    :data:`ModelRefinementService.MODEL_COMPAT_THRESHOLD`: it groups the feature
    columns the host is missing into recognizable families and pairs them with
    concrete fixes (usually: match the host's feature-extraction settings to the
    source's and re-extract).  ``None`` is used when every model is importable.
    """

    models_blocked: int = 0          # models below the coverage threshold
    models_total: int = 0
    worst_coverage: float = 1.0      # lowest per-model coverage (0..1)
    missing_total: int = 0           # distinct host-aligned columns missing
    # (group label, count) sorted by count desc — e.g. ("Video / optical-flow", 228)
    missing_groups: list[tuple[str, int]] = field(default_factory=list)
    sample_missing: list[str] = field(default_factory=list)  # a few example names
    causes: list[str] = field(default_factory=list)          # likely reasons
    fixes: list[str] = field(default_factory=list)           # ordered fix steps

    @property
    def has_blocked_models(self) -> bool:
        return self.models_blocked > 0


@dataclass
class ImportRecord:
    """A source whose examples have been imported into this project.

    Persisted so the Model Refinement tab can list prior imports across sessions
    and offer a clean removal (un-import).
    """

    tag: str
    source_root: str = ""
    imported_rows: int = 0
    review_registered: int = 0
    behaviors: dict[str, int] = field(default_factory=dict)  # host behaviour name -> count
    imported_at: str = ""


@dataclass
class RefinementPreview:
    """Everything the UI needs to decide whether / what to import."""

    source_root: Path
    tag: str
    compatible: bool = False
    reason: str = ""
    host_feature_count: int = 0
    source_feature_count: int = 0
    shared_feature_count: int = 0
    coverage: float = 0.0
    behavior_mappings: list[BehaviorMapping] = field(default_factory=list)
    total_labeled: int = 0
    importable_labeled: int = 0  # labels that map to a host behaviour
    keypoint_renames: dict[str, str] = field(default_factory=dict)  # source_kp -> host_kp
    diagnostics: CompatibilityDiagnostics | None = None

    @property
    def matched_behaviors(self) -> list[BehaviorMapping]:
        return [m for m in self.behavior_mappings if m.matched]

    @property
    def unmatched_behaviors(self) -> list[BehaviorMapping]:
        return [m for m in self.behavior_mappings if not m.matched and m.example_count]


# Sentinel host-behaviour decisions for model import (vs. an explicit host id).
AUTO_CREATE_BEHAVIOR = "__auto_create__"
SKIP_BEHAVIOR = "__skip__"


@dataclass
class SourceModel:
    """A trained behaviour model discovered in a source project."""

    model_dir: str           # directory name under derived/models
    behavior_id: str         # source behaviour id the model predicts
    behavior_name: str
    feature_columns: list[str] = field(default_factory=list)

    @property
    def feature_count(self) -> int:
        return len(self.feature_columns)


@dataclass
class ModelImportItem:
    """Per-model compatibility + behaviour-mapping result in a preview."""

    model: SourceModel
    coverage: float                 # fraction of the model's feature cols the host has
    missing_features: int
    host_behavior_id: str = ""      # "" when the behaviour is unmatched
    host_behavior_name: str = ""
    matched_by_alias: bool = False
    compatible: bool = False        # host covers (nearly) all required features
    # Model feature columns (host-aligned) the host doesn't have — the concrete
    # gap behind a sub-threshold coverage.  Used by the coverage diagnosis.
    missing_columns: list[str] = field(default_factory=list)

    @property
    def behavior_matched(self) -> bool:
        return bool(self.host_behavior_id)

    @property
    def has_model_gap(self) -> bool:
        """True when this model is below the import coverage threshold."""
        return not self.compatible


@dataclass
class ModelImportPreview:
    """Everything the UI needs to decide which models to import."""

    source_root: Path
    tag: str
    host_feature_count: int = 0
    keypoint_renames: dict[str, str] = field(default_factory=dict)
    items: list[ModelImportItem] = field(default_factory=list)
    diagnostics: CompatibilityDiagnostics | None = None

    @property
    def importable(self) -> list[ModelImportItem]:
        return [i for i in self.items if i.compatible]

    @property
    def unmatched_behaviors(self) -> list[ModelImportItem]:
        return [i for i in self.items if i.compatible and not i.behavior_matched]


@dataclass
class BaselineBehaviorRow:
    """One source behaviour in a baseline-import detection summary.

    Combines the example side (labeled clips/feature rows) and the model side
    (trained model) for a single source behaviour, plus how it maps onto the host.
    """

    source_behavior_id: str
    source_name: str
    example_count: int = 0          # importable labeled examples
    has_model: bool = False
    model_coverage: float = 0.0     # 0..1 (0 when no model)
    model_compatible: bool = False
    matched_host_id: str = ""       # "" when no existing host behaviour matches
    matched_host_name: str = ""

    @property
    def status(self) -> str:
        return "matched" if self.matched_host_id else "new"


@dataclass
class BaselinePreview:
    """Detection summary for importing another project as a baseline."""

    source_root: Path
    tag: str
    host_is_new: bool = False
    host_feature_count: int = 0
    coverage: float = 0.0           # example-feature schema coverage (0..1)
    schema_ok: bool = False
    reason: str = ""
    rows: list[BaselineBehaviorRow] = field(default_factory=list)
    keypoint_renames: dict[str, str] = field(default_factory=dict)
    diagnostics: CompatibilityDiagnostics | None = None
    model_count: int = 0
    # Why models (if any) can't be imported + how to fix it.  None when every
    # trained model is importable.
    coverage_diagnosis: "CoverageDiagnosis | None" = None

    @property
    def total_examples(self) -> int:
        return sum(r.example_count for r in self.rows)

    @property
    def matched_rows(self) -> list[BaselineBehaviorRow]:
        return [r for r in self.rows if r.matched_host_id]

    @property
    def new_rows(self) -> list[BaselineBehaviorRow]:
        return [r for r in self.rows if not r.matched_host_id]


class ModelRefinementService:
    """Read labeled examples from source projects and merge them into a host."""

    def __init__(self) -> None:
        self._trainer = ActiveLearningTrainerService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preview(
        self,
        host_root: Path,
        source_root: Path,
        tag: str = "",
        name_overrides: dict[str, str] | None = None,
        compute_diagnostics: bool = True,
    ) -> RefinementPreview:
        """Inspect a source project without modifying anything.

        ``name_overrides`` maps a source behaviour *name* (case-insensitive) to
        the host behaviour *name* it should be imported as, letting differently
        named-but-identical behaviours match.  When omitted, the host project's
        saved alias table is used.

        When ``compute_diagnostics`` is set, attaches a
        :class:`CompatibilityDiagnostics` (calibration, pose model, extraction
        settings, feature-value shift).  Imports turn it off to avoid the extra
        parquet read, since they re-validate via the cheaper compatibility path.
        """
        tag = tag or source_root.name
        if name_overrides is None:
            name_overrides = self.load_aliases(host_root)
        pv = RefinementPreview(source_root=source_root, tag=tag)

        host_features = self._host_feature_cols(host_root)
        if host_features is None:
            pv.reason = (
                "This project has no extracted features yet. Run feature "
                "extraction before importing examples or a baseline."
            )
            return pv

        source_features = self._source_feature_cols(source_root)
        if source_features is None:
            pv.reason = (
                f"Source project '{tag}' has no segment_features.parquet — "
                "it has not been processed/labeled."
            )
            return pv

        labels = self._load_labels(source_root)
        if labels is None or labels.empty:
            pv.reason = f"Source project '{tag}' has no reviewer labels to import."
            return pv

        # Rename the source's feature columns onto the host's keypoint scheme so
        # that identical keypoints under different names (back_mid/center_body,
        # ear_left/left_ear, ...) line up before the coverage check.
        kp_rename = self._keypoint_rename_map(host_root, host_features, source_features)
        if kp_rename:
            col_rename = self._rename_cols(source_features, kp_rename)
            source_features = {col_rename.get(c, c) for c in source_features}
            pv.keypoint_renames = kp_rename

        shared = host_features & source_features
        coverage = (len(shared) / len(host_features)) if host_features else 0.0
        pv.host_feature_count = len(host_features)
        pv.source_feature_count = len(source_features)
        pv.shared_feature_count = len(shared)
        pv.coverage = coverage

        if compute_diagnostics:
            try:
                pv.diagnostics = self._compute_diagnostics(
                    host_root, source_root, shared, kp_rename,
                )
            except Exception:  # diagnostics are advisory; never block on them
                logger.exception("Failed to compute refinement diagnostics")

        # Behaviour mapping + per-behaviour example counts.
        source_behaviors = self._read_behaviors(source_root)
        host_name_to_id = {
            name.lower(): bid for bid, name in self._read_behaviors(host_root).items()
        }
        label_counts = labels["review_label"].astype(str).value_counts().to_dict()

        mappings: list[BehaviorMapping] = []
        importable = 0
        for raw_label, count in label_counts.items():
            host_bid, host_name, via_alias = self._resolve_host_behavior(
                raw_label, source_behaviors, host_name_to_id, name_overrides,
            )
            mappings.append(BehaviorMapping(
                source_behavior_id=raw_label,
                source_name=source_behaviors.get(raw_label, raw_label),
                host_behavior_id=host_bid,
                host_name=host_name,
                example_count=int(count),
                remapped=via_alias,
            ))
            if host_bid:
                importable += int(count)
        mappings.sort(key=lambda m: (-m.example_count, m.source_name))

        pv.behavior_mappings = mappings
        pv.total_labeled = int(len(labels))
        pv.importable_labeled = importable

        if coverage < COMPAT_THRESHOLD:
            missing = len(host_features) - len(shared)
            pv.compatible = False
            pv.reason = (
                f"Incompatible feature schemas: only {coverage:.0%} of this "
                f"project's {len(host_features)} feature columns exist in "
                f"'{tag}' ({missing} missing). The projects likely use "
                "different pose keypoints or ROI layouts. Import is blocked to "
                "avoid training on missing features."
            )
            return pv

        if importable == 0:
            pv.compatible = False
            pv.reason = (
                f"None of '{tag}'s labeled behaviours match a behaviour in "
                "this project (matched by name)."
            )
            return pv

        pv.compatible = True
        return pv

    def import_examples(
        self,
        host_root: Path,
        source_root: Path,
        tag: str = "",
        name_overrides: dict[str, str] | None = None,
        behavior_decisions: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Merge a source project's labeled examples into the host training set.

        Re-validates compatibility via ``preview`` first.  ``name_overrides``
        (source name -> host name aliases) is applied to remap differently
        named behaviours; when omitted the host's saved alias table is used.

        ``behavior_decisions`` (source behaviour id -> host id /
        ``AUTO_CREATE_BEHAVIOR`` / ``SKIP_BEHAVIOR``) lets a baseline import seed a
        project that has no matching behaviours yet: behaviours chosen for
        auto-create are added from the source definition, and only the
        feature-schema half of compatibility is enforced (a brand-new host has no
        behaviours to match by name).  When omitted, behaviour identity is matched
        by name/alias exactly as before.

        Returns a result dict with ``status`` and, on success, the number of
        imported rows and the snapshot path.
        """
        if name_overrides is None:
            name_overrides = self.load_aliases(host_root)
        pv = self.preview(
            host_root, source_root, tag=tag, name_overrides=name_overrides,
            compute_diagnostics=False,
        )
        if behavior_decisions:
            # The decision set supplies behaviour identity, so only the feature
            # schema needs to line up here (the importable-by-name gate inside
            # ``preview`` would otherwise block a not-yet-labeled host).
            schema_ok = pv.host_feature_count > 0 and pv.coverage >= COMPAT_THRESHOLD
            if not schema_ok:
                return {"status": "error", "error": pv.reason or
                        "Incompatible feature schema.", "preview": pv}
        elif not pv.compatible:
            return {"status": "error", "error": pv.reason, "preview": pv}

        tag = pv.tag

        # Auto-create any host behaviours the baseline import chose to add as new,
        # so the training set's labels resolve to defined behaviours downstream.
        if behavior_decisions:
            for src_bid, decision in behavior_decisions.items():
                if decision == AUTO_CREATE_BEHAVIOR:
                    self._auto_create_behavior(host_root, source_root, src_bid)

        # Host column schema: the assembled training set when it exists, else the
        # host's extracted segment features (so a features-extracted-but-untrained
        # project can receive a baseline; merge_and_snapshot creates the file).
        host_cols = self._host_training_columns(host_root)

        merged, label_to_host = self._merged_labeled(
            host_root, source_root, pv, name_overrides, behavior_decisions,
        )
        if merged.empty:
            return {
                "status": "error",
                "error": "No labeled segments could be joined to features.",
                "preview": pv,
            }

        # Build the host-shaped training rows.
        out = pd.DataFrame(index=merged.index)
        for col in host_cols:
            if col in merged.columns:
                out[col] = merged[col]
            else:
                out[col] = pd.NA  # missing host feature (within tolerance)

        out["label"] = merged["review_label"].map(label_to_host)
        out["label_source"] = f"imported:{tag}"
        if "reviewer_confidence" in host_cols:
            out["reviewer_confidence"] = pd.to_numeric(
                merged.get("confidence", 1.0), errors="coerce",
            ).fillna(1.0)

        # Namespace identifiers so imported rows never collide with host rows.
        out["segment_id"] = merged["segment_id"].astype(str).map(lambda s: f"{tag}__{s}")
        if "session_id" in merged.columns:
            out["session_id"] = merged["session_id"].astype(str).map(
                lambda s: f"{tag}__{s}"
            )

        out = out[host_cols]  # exact column order
        snap_path = self._trainer.merge_and_snapshot_training_set(host_root, out)

        logger.info(
            "Imported %d labeled example(s) from '%s' into %s training set.",
            len(out), tag, host_root.name,
        )

        # Surface the imported examples in the Review tab as reviewed, source-
        # tagged entries (copying their clips so they're viewable here).  Advisory
        # — a failure here must not undo the training-set merge above.
        review_registered = 0
        try:
            review_registered = self._register_review_examples(
                host_root, source_root, tag, merged, label_to_host,
            )
        except Exception:
            logger.exception("Failed to register imported examples for review")

        # Persist a manifest record so this import is listed (and removable)
        # across sessions in the Model Refinement tab.  With a decision set the
        # name/alias-based ``matched_behaviors`` doesn't capture auto-created /
        # remapped imports, so count the rows actually written per host behaviour.
        if behavior_decisions:
            host_id_to_name = {bid: name for bid, name in self._read_behaviors(host_root).items()}
            counts = (
                out["label"].astype(str).value_counts().to_dict() if "label" in out else {}
            )
            behaviors = {
                host_id_to_name.get(str(bid), str(bid)): int(c)
                for bid, c in counts.items()
            }
        else:
            behaviors = {m.host_name: m.example_count for m in pv.matched_behaviors}
        self._record_import(
            host_root, tag, source_root, int(len(out)),
            int(review_registered), behaviors,
        )

        return {
            "status": "success",
            "imported_rows": int(len(out)),
            "review_registered": int(review_registered),
            "tag": tag,
            "source_root": str(source_root),
            "snapshot_path": str(snap_path),
            "preview": pv,
        }

    def register_imported_for_review(
        self,
        host_root: Path,
        source_root: Path,
        tag: str = "",
        name_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Register an *already-imported* source's labeled examples in the Review
        tab without re-merging training rows.

        Use this to backfill review visibility for a source whose examples were
        imported before review registration existed.  Safe to re-run: it clears
        any prior registration for this source first.
        """
        if name_overrides is None:
            name_overrides = self.load_aliases(host_root)
        pv = self.preview(
            host_root, source_root, tag=tag, name_overrides=name_overrides,
            compute_diagnostics=False,
        )
        if not pv.compatible:
            return {"status": "error", "error": pv.reason, "preview": pv}
        merged, label_to_host = self._merged_labeled(
            host_root, source_root, pv, name_overrides,
        )
        if merged.empty:
            return {"status": "error", "error": "No labeled segments to register."}
        n = self._register_review_examples(
            host_root, source_root, pv.tag, merged, label_to_host,
        )
        # Record (or refresh) the manifest entry so this source is listed and
        # removable even when its training rows were merged by an older import.
        imported_rows = self._count_imported_rows(host_root, pv.tag)
        behaviors = {m.host_name: m.example_count for m in pv.matched_behaviors}
        self._record_import(
            host_root, pv.tag, source_root, imported_rows, int(n), behaviors,
        )
        return {"status": "success", "review_registered": int(n), "tag": pv.tag}

    # ------------------------------------------------------------------
    # Import manifest (list + cleanly remove prior imports)
    # ------------------------------------------------------------------

    @staticmethod
    def _imports_manifest_path(host_root: Path) -> Path:
        return host_root / "derived" / "training_sets" / "refinement_imports.json"

    def list_imports(self, host_root: Path) -> list[ImportRecord]:
        """Return the sources imported into ``host_root``, newest first."""
        raw = read_json(self._imports_manifest_path(host_root), {"imports": []})
        out: list[ImportRecord] = []
        for item in (raw.get("imports", []) if isinstance(raw, dict) else []):
            try:
                out.append(ImportRecord(
                    tag=str(item["tag"]),
                    source_root=str(item.get("source_root", "")),
                    imported_rows=int(item.get("imported_rows", 0)),
                    review_registered=int(item.get("review_registered", 0)),
                    behaviors={str(k): int(v) for k, v in (item.get("behaviors") or {}).items()},
                    imported_at=str(item.get("imported_at", "")),
                ))
            except Exception:
                continue
        out.sort(key=lambda r: r.imported_at, reverse=True)
        return out

    def _record_import(
        self, host_root: Path, tag: str, source_root: Path | str,
        imported_rows: int, review_registered: int, behaviors: dict[str, int],
    ) -> None:
        """Add or update the manifest entry for ``tag`` (keyed by tag)."""
        records = {r.tag: r for r in self.list_imports(host_root)}
        records[tag] = ImportRecord(
            tag=tag,
            source_root=str(source_root),
            imported_rows=int(imported_rows),
            review_registered=int(review_registered),
            behaviors=dict(behaviors),
            imported_at=datetime.utcnow().isoformat(timespec="seconds"),
        )
        write_json(
            self._imports_manifest_path(host_root),
            {"imports": [r.__dict__ for r in records.values()]},
        )

    def _remove_import_record(self, host_root: Path, tag: str) -> None:
        records = [r for r in self.list_imports(host_root) if r.tag != tag]
        write_json(
            self._imports_manifest_path(host_root),
            {"imports": [r.__dict__ for r in records]},
        )

    @staticmethod
    def _count_imported_rows(host_root: Path, tag: str) -> int:
        ts_path = host_root / "derived" / "training_sets" / "training_set.parquet"
        if not ts_path.exists():
            return 0
        try:
            df = pd.read_parquet(ts_path, columns=["label_source"])
        except Exception:
            return 0
        if "label_source" not in df.columns:
            return 0
        return int((df["label_source"].astype(str) == f"imported:{tag}").sum())

    def remove_import(self, host_root: Path, tag: str) -> dict[str, Any]:
        """Cleanly un-import a source: drop its training rows, review entries,
        and copied clips, then forget it.

        Reverses everything :meth:`import_examples` added for ``tag``.  Other
        sources' data is untouched.  Safe to call even if some pieces are
        already gone (counts what it actually removed).
        """
        result: dict[str, Any] = {"status": "success", "tag": tag}

        # 1. Training rows (label_source == imported:<tag>), with a fresh snapshot.
        removed_rows = 0
        ts_path = host_root / "derived" / "training_sets" / "training_set.parquet"
        if ts_path.exists():
            df = pd.read_parquet(ts_path)
            if "label_source" in df.columns:
                mask = df["label_source"].astype(str) == f"imported:{tag}"
                removed_rows = int(mask.sum())
                if removed_rows:
                    df = df[~mask].reset_index(drop=True)
                    df.to_parquet(ts_path, index=False)
                    snap_dir = ts_path.parent / "snapshots"
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    df.to_parquet(snap_dir / f"training_set_{stamp}.parquet", index=False)
        result["removed_rows"] = removed_rows

        # 2. External review candidates tagged with this source.
        cand = CandidateGenerationService()
        cand.set_project(host_root)
        result["removed_candidates"] = int(cand.remove_external_candidates_by_source(tag))

        # 3. Review decisions for this source's namespaced clip ids.
        review = ReviewService()
        review.set_project(host_root)
        decisions = review.load_decisions()
        prefix = f"{tag}__"
        kept = [d for d in decisions if not str(d.clip_id).startswith(prefix)]
        removed_decisions = len(decisions) - len(kept)
        if removed_decisions:
            review.save_decisions(kept)
        result["removed_decisions"] = int(removed_decisions)

        # 4. Copied clip folders (derived/clips/<tag>__<session>/).
        removed_clips = 0
        clips_root = host_root / "derived" / "clips"
        if clips_root.exists():
            for d in clips_root.glob(f"{tag}__*"):
                if d.is_dir():
                    removed_clips += sum(1 for _ in d.rglob("*.mp4"))
                    shutil.rmtree(d, ignore_errors=True)
        result["removed_clips"] = int(removed_clips)

        # 5. Forget the manifest entry.
        self._remove_import_record(host_root, tag)

        logger.info(
            "Removed import '%s': %d training rows, %d candidates, %d decisions, %d clips.",
            tag, removed_rows, result["removed_candidates"], removed_decisions, removed_clips,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _feature_cols(cls, columns: list[str]) -> set[str]:
        return {c for c in columns if c not in _NON_FEATURE_COLS}

    # ------------------------------------------------------------------
    # Shared labeled-segment join (used by import + review registration)
    # ------------------------------------------------------------------

    def _merged_labeled(
        self, host_root: Path, source_root: Path, pv: RefinementPreview,
        name_overrides: dict[str, str] | None,
        behavior_decisions: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Return ``(merged, label_to_host)`` for the source's importable labels.

        ``merged`` is the source's per-segment features (renamed onto the host
        keypoint scheme) carrying each labeled segment's behaviour label, keeping
        only labels that map to a host behaviour.  ``label_to_host`` maps each
        source label to the host behaviour id it imports as.

        Features and labels come from the source's assembled training set when it
        exists: that one file already pairs every labeled segment with its
        features, so the import covers the project's whole labeled set rather than
        only what is in the Review-tab log.  Sources without a training set fall
        back to joining the reviewer-label log to the raw segment-features store.
        """
        labels = self._load_labels(source_root)
        if labels is None or labels.empty:
            return pd.DataFrame(), {}

        source_behaviors = self._read_behaviors(source_root)
        host_name_to_id = {
            name.lower(): bid for bid, name in self._read_behaviors(host_root).items()
        }
        label_to_host: dict[str, str] = {}
        for raw_label in labels["review_label"].astype(str).unique():
            host_bid = self._decisioned_host_bid(
                raw_label, source_behaviors, host_name_to_id, name_overrides,
                behavior_decisions,
            )
            if host_bid:
                label_to_host[raw_label] = host_bid

        feat_path = self._source_feature_path(source_root)
        source_features = pd.read_parquet(feat_path)
        if pv.keypoint_renames:
            col_rename = self._rename_cols(source_features.columns, pv.keypoint_renames)
            if col_rename:
                source_features = source_features.rename(columns=col_rename)

        if feat_path == self._source_training_path(source_root) and "label" in source_features.columns:
            # Training-set rows already pair features with their label; take the
            # label/confidence off each row directly (no separate-table join, so
            # every labeled segment is covered — not just the review log).
            merged = source_features.copy()
            merged["review_label"] = merged["label"].astype(str)
            if "reviewer_confidence" in merged.columns:
                merged["confidence"] = pd.to_numeric(
                    merged["reviewer_confidence"], errors="coerce",
                ).fillna(1.0)
            else:
                merged["confidence"] = 1.0
            merged = merged[merged["review_label"].isin(label_to_host)].reset_index(drop=True)
        else:
            labels = labels.copy()
            labels["review_label"] = labels["review_label"].astype(str)
            labels = labels[labels["review_label"].isin(label_to_host)]
            merged = source_features.merge(
                labels[["segment_id", "review_label", "confidence"]],
                on="segment_id", how="inner",
            )
        return merged, label_to_host

    # ------------------------------------------------------------------
    # Review-tab registration (show imported examples as reviewed clips)
    # ------------------------------------------------------------------

    def _register_review_examples(
        self, host_root: Path, source_root: Path, tag: str,
        merged: pd.DataFrame, label_to_host: dict[str, str],
    ) -> int:
        """Register imported segments as reviewed, source-tagged Review entries.

        For each importable segment: copy its clip from the source project into
        this project, add an external candidate (carrying ``source=tag`` and the
        copied clip path), and record an accept decision so it shows as reviewed.
        Idempotent per source — prior registrations for ``tag`` are cleared first.
        """
        clip_index = self._index_source_clips(source_root)
        host_clips_root = host_root / "derived" / "clips"

        # Slim to the identity columns only: merged can carry 1000+ feature
        # columns, and itertuples drops field names past 255 columns.
        meta_cols = [
            c for c in ("segment_id", "review_label", "session_id",
                        "start_frame", "end_frame")
            if c in merged.columns
        ]
        slim = merged[meta_cols]
        has_session = "session_id" in slim.columns
        has_start = "start_frame" in slim.columns
        has_end = "end_frame" in slim.columns

        candidates: list[CandidateWindow] = []
        new_decisions: dict[str, ReviewDecision] = {}
        copied = 0
        for row in slim.itertuples(index=False):
            host_bid = label_to_host.get(str(getattr(row, "review_label", "")))
            if not host_bid:
                continue
            seg = str(getattr(row, "segment_id"))
            session = str(getattr(row, "session_id")) if has_session else ""
            window_id = f"{tag}__{seg}"
            ns_session = f"{tag}__{session}" if session else tag
            start = int(getattr(row, "start_frame")) if has_start else 0
            end = int(getattr(row, "end_frame")) if has_end else 0

            clip_path: str | None = None
            src_clip = clip_index.get(seg)
            if src_clip is not None:
                dest_dir = host_clips_root / ns_session
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / src_clip.name
                try:
                    if not dest.exists():
                        shutil.copy2(src_clip, dest)
                    clip_path = str(dest)
                    copied += 1
                except Exception:
                    logger.exception("Failed to copy imported clip %s", src_clip)

            candidates.append(CandidateWindow(
                window_id=window_id,
                session_id=ns_session,
                start_frame=start,
                end_frame=end,
                behavior_id=host_bid,
                clip_path=clip_path,
                source=tag,
                selection_reason=f"imported from {tag}",
            ))
            new_decisions[window_id] = ReviewDecision(
                decision_id=uuid.uuid4().hex[:12],
                clip_id=window_id,
                reviewer=f"imported:{tag}",
                old_status="unscored",
                new_status="reviewed",
                decision=ReviewDecisionType.ACCEPT,
                behavior_label=host_bid,
                notes=f"Imported from {tag}",
                confidence_override=1.0,
                adjusted_start_frame=start,
                adjusted_end_frame=end,
            )

        cand_svc = CandidateGenerationService()
        cand_svc.set_project(host_root)
        cand_svc.remove_external_candidates_by_source(tag)
        cand_svc.upsert_external_window_candidates(candidates)

        review = ReviewService()
        review.set_project(host_root)
        kept = [d for d in review.load_decisions() if d.clip_id not in new_decisions]
        review.save_decisions(kept + list(new_decisions.values()))

        logger.info(
            "Registered %d imported example(s) from '%s' for review (%d clips copied).",
            len(candidates), tag, copied,
        )
        return len(candidates)

    @staticmethod
    def _index_source_clips(source_root: Path) -> dict[str, Path]:
        """Map ``segment_id -> clip file`` for a source project.

        Clip files are named ``{segment_id}_{hash}.mp4`` under ``derived/clips``;
        stripping the trailing hash recovers the segment id.
        """
        clips_root = source_root / "derived" / "clips"
        out: dict[str, Path] = {}
        if not clips_root.exists():
            return out
        for p in clips_root.rglob("*.mp4"):
            stem = re.sub(r"_[0-9a-f]{6,}$", "", p.stem)
            out.setdefault(stem, p)
        return out

    # ------------------------------------------------------------------
    # Keypoint-name remapping (align differently-named pose schemes)
    # ------------------------------------------------------------------

    @staticmethod
    def _keypoints_from_cols(columns) -> set[str]:
        """Recover keypoint names from pairwise-distance feature columns.

        Distance columns are ``dist_<kpA>_to_<kpB>[_norm]``; both endpoints are
        keypoint names, so they enumerate the scheme without a separate config.
        """
        kps: set[str] = set()
        for c in columns:
            m = re.match(r"dist_(.+)_to_(.+?)(?:_norm)?$", c)
            if m:
                kps.add(m.group(1))
                kps.add(m.group(2))
        return kps

    @staticmethod
    def _rename_cols(columns, kp_rename: dict[str, str]) -> dict[str, str]:
        """Return ``{old_col: new_col}`` for columns whose keypoint tokens change.

        ``kp_rename`` maps source keypoint -> host keypoint.  Only whole
        ``_``-delimited keypoint tokens are rewritten (so ``ear_left`` in
        ``dist_nose_to_ear_left`` is replaced, but no partial/substring match),
        in a single pass to avoid one rename cascading into another.
        """
        if not kp_rename:
            return {}
        alts = "|".join(
            re.escape(k) for k in sorted(kp_rename, key=len, reverse=True)
        )
        pattern = re.compile(rf"(?<![a-z])(?:{alts})(?![a-z])")
        out: dict[str, str] = {}
        for c in columns:
            nc = pattern.sub(lambda m: kp_rename[m.group(0)], c)
            if nc != c:
                out[c] = nc
        return out

    def _keypoint_rename_map(
        self, host_root: Path, host_feat_cols, source_feat_cols
    ) -> dict[str, str]:
        """Best ``{source_kp: host_kp}`` map to align the source onto the host.

        Considers the host's saved Direct Use map and an auto-suggested mapping,
        and picks whichever maximises feature-column overlap with the host.  The
        empty (no-op) map is always a candidate, so remapping can never *reduce*
        overlap below what the raw column names already give.
        """
        host_set = set(host_feat_cols)
        candidates: list[dict[str, str]] = [{}]

        saved = keypoint_mapping.load_saved(host_root)  # {host_kp: source_kp}
        if saved:
            candidates.append(keypoint_mapping.to_rename_map(saved))

        host_kps = self._keypoints_from_cols(host_feat_cols)
        source_kps = self._keypoints_from_cols(source_feat_cols)
        if host_kps and source_kps:
            suggested = keypoint_mapping.suggest_mapping(
                sorted(host_kps), sorted(source_kps)
            )
            candidates.append(keypoint_mapping.to_rename_map(suggested))

        best, best_overlap = {}, -1
        for cand in candidates:
            col_rename = self._rename_cols(source_feat_cols, cand)
            renamed = {col_rename.get(c, c) for c in source_feat_cols}
            overlap = len(host_set & renamed)
            if overlap > best_overlap:
                best, best_overlap = cand, overlap
        return best

    # ------------------------------------------------------------------
    # Compatibility diagnostics (value-level similarity, not just schema)
    # ------------------------------------------------------------------

    # behavior_model keys whose value affects what a feature column *means*, so
    # a mismatch makes otherwise-aligned columns not directly comparable.
    _DIAG_CFG_KEYS = (
        "segment_window_frames", "segment_stride_frames",
        "use_video_features",
    )
    _DIAG_INVARIANT_KEYS = (
        "enable_egocentric_kinematics", "enable_body_length_normalization",
        "enable_relative_geometry", "enable_head_direction",
        "enable_joint_angles", "enable_spine_curvature",
        "enable_clipwise_deltas",
    )

    def _compute_diagnostics(
        self, host_root: Path, source_root: Path, shared, kp_rename: dict[str, str],
    ) -> CompatibilityDiagnostics:
        diag = CompatibilityDiagnostics()

        # Spatial calibration (median pixels/mm across each project's sessions).
        h_ppm, h_models = self._session_meta(host_root)
        s_ppm, s_models = self._session_meta(source_root)
        diag.host_px_per_mm = h_ppm
        diag.source_px_per_mm = s_ppm
        if h_ppm and s_ppm:
            diag.px_per_mm_pct_diff = abs(h_ppm - s_ppm) / ((h_ppm + s_ppm) / 2) * 100

        # Pose model (DLC network) signatures.
        diag.host_pose_models = sorted(h_models)
        diag.source_pose_models = sorted(s_models)
        diag.pose_models_match = bool(h_models) and h_models == s_models

        # Feature-extraction config differences.
        diag.config_mismatches = self._config_mismatches(
            self._behavior_model_cfg(host_root),
            self._behavior_model_cfg(source_root),
        )

        # Net feature-value distribution shift over shared columns.
        if len(shared) >= 20:
            self._fill_feature_shift(diag, host_root, source_root, shared, kp_rename)
        return diag

    def _fill_feature_shift(
        self, diag: CompatibilityDiagnostics, host_root: Path, source_root: Path,
        shared, kp_rename: dict[str, str], sample_cap: int = 8000,
    ) -> None:
        """Median/p90 per-column shift (pooled-IQR units) host vs source, with a
        within-host baseline that shows the sampling-noise floor for reference."""
        shared = sorted(shared)
        host_path = host_root / "derived" / "training_sets" / "training_set.parquet"
        src_path = self._source_feature_path(source_root)

        # Source columns are stored under source-keypoint names; map host-named
        # shared columns back to the on-disk source names so we read the right ones.
        col_rename = self._rename_cols(
            self._parquet_columns(src_path) or [], kp_rename,
        )
        new_to_orig = {new: old for old, new in col_rename.items()}
        src_on_disk = [new_to_orig.get(c, c) for c in shared]

        host_df = pd.read_parquet(host_path, columns=shared)
        src_df = pd.read_parquet(src_path, columns=src_on_disk)
        src_df.columns = shared  # align to host names (same order)

        host_df = self._sample_rows(host_df, sample_cap)
        src_df = self._sample_rows(src_df, sample_cap)
        hn = host_df.apply(pd.to_numeric, errors="coerce")
        sn = src_df.apply(pd.to_numeric, errors="coerce")

        cross = self._column_shifts(hn, sn, shared)
        # Within-host baseline: random halves -> noise floor (~0 when stable).
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(hn))
        h1, h2 = hn.iloc[idx[: len(idx) // 2]], hn.iloc[idx[len(idx) // 2:]]
        within = self._column_shifts(h1, h2, shared)

        if cross.size:
            diag.feature_shift_median = float(np.median(cross))
            diag.feature_shift_p90 = float(np.percentile(cross, 90))
            diag.feature_shift_frac_gt_half = float(np.mean(cross > 0.5))
        if within.size:
            diag.within_host_shift_median = float(np.median(within))

    @staticmethod
    def _column_shifts(a: pd.DataFrame, b: pd.DataFrame, cols) -> np.ndarray:
        """|median(a) - median(b)| per column, scaled by the pooled IQR."""
        out = []
        for c in cols:
            x, y = a[c].dropna().to_numpy(), b[c].dropna().to_numpy()
            if len(x) < 30 or len(y) < 30:
                continue
            pooled = np.concatenate([x, y])
            iqr = float(np.subtract(*np.percentile(pooled, [75, 25]))) or 1e-9
            out.append(abs(float(np.median(x)) - float(np.median(y))) / iqr)
        return np.asarray(out, dtype=float)

    @staticmethod
    def _sample_rows(df: pd.DataFrame, cap: int) -> pd.DataFrame:
        return df.sample(cap, random_state=0) if len(df) > cap else df

    @staticmethod
    def _session_meta(root: Path) -> tuple[float | None, set[str]]:
        """Return (median pixels/mm, {DLC network signatures}) for a project."""
        path = root / "config" / "session_registry.json"
        if not path.exists():
            return None, set()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None, set()
        entries = data.get("entries", data)
        rows = list(entries.values()) if isinstance(entries, dict) else (entries or [])
        ppm = [
            float(r["pixels_per_mm"]) for r in rows
            if isinstance(r, dict) and r.get("pixels_per_mm")
        ]
        models: set[str] = set()
        for r in rows:
            if not isinstance(r, dict):
                continue
            m = re.search(r"(DLC_[A-Za-z0-9_]+?shuffle\d+)", str(r.get("pose_filename", "")))
            if m:
                models.add(m.group(1))
        median_ppm = float(np.median(ppm)) if ppm else None
        return median_ppm, models

    @staticmethod
    def _behavior_model_cfg(root: Path) -> dict[str, Any]:
        """Feature-extraction settings that define what a feature column means.

        Reads ``project.yaml`` — the live config the extraction pipeline runs
        from — preferentially over the secondary ``config/experiment.yaml``,
        which can hold a stale ``behavior_model`` copy that the Features-tab
        checkboxes don't keep in sync (e.g. ``use_video_features`` /
        ``segment_stride_frames``).  ``use_video_features`` is taken from the
        ``feature_extraction`` block, the single source of truth for the
        video-features toggle, mirroring ``ActiveLearningTab._load_behavior_cfg``
        so the diagnostic reports what was actually extracted.
        """
        import yaml  # noqa: PLC0415

        def _load(p: Path) -> dict:
            if not p.exists():
                return {}
            try:
                return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                return {}

        project = _load(root / "project.yaml")
        experiment = _load(root / "config" / "experiment.yaml")
        bm = project.get("behavior_model") or experiment.get("behavior_model") or {}
        cfg = dict(bm) if isinstance(bm, dict) else {}
        # The Features-tab "Include video-derived features" checkbox persists to
        # project.yaml's feature_extraction block; honour it over any stale
        # behavior_model copy so the comparison reflects the real extraction.
        fx = project.get("feature_extraction") or {}
        if isinstance(fx, dict) and "use_video_features" in fx:
            cfg["use_video_features"] = bool(fx["use_video_features"])
        return cfg

    @classmethod
    def _config_mismatches(cls, host_cfg: dict, source_cfg: dict) -> list[str]:
        """Human-readable list of extraction settings that differ.

        Empty == the two projects extract features the same way.  Skipped when
        either config is unavailable (can't tell, so don't cry wolf).
        """
        if not host_cfg or not source_cfg:
            return []
        out: list[str] = []
        for key in cls._DIAG_CFG_KEYS:
            h, s = host_cfg.get(key), source_cfg.get(key)
            if h != s:
                out.append(f"{key}: {h} vs {s}")
        h_inv = host_cfg.get("invariant_features", {}) or {}
        s_inv = source_cfg.get("invariant_features", {}) or {}
        for key in cls._DIAG_INVARIANT_KEYS:
            h, s = h_inv.get(key), s_inv.get(key)
            if h != s:
                out.append(f"{key}: {h} vs {s}")
        return out

    @staticmethod
    def _parquet_columns(path: Path) -> list[str] | None:
        """Read only the column names from a parquet file (no row data).

        Falls back to a full read if the schema-only path is unavailable.
        """
        if not path.exists():
            return None
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415
            return list(pq.read_schema(path).names)
        except Exception:
            try:
                return pd.read_parquet(path).columns.tolist()
            except Exception:
                return None

    @classmethod
    def _host_feature_path(cls, host_root: Path) -> Path:
        """On-disk parquet the host's feature columns come from.

        Prefers the assembled training set, but falls back to the raw
        segment-features store when the host has *extracted features but not yet
        run active learning* (no training set).  Without the fallback, importing a
        baseline (examples + models) into a freshly-prepared project would be
        blocked because the host's feature schema looked empty.  Mirrors
        :meth:`_source_feature_path`.
        """
        return cls._source_feature_path(host_root)

    def _host_feature_cols(self, host_root: Path) -> set[str] | None:
        cols = self._parquet_columns(self._host_feature_path(host_root))
        return self._feature_cols(cols) if cols is not None else None

    @staticmethod
    def _source_training_path(source_root: Path) -> Path:
        return source_root / "derived" / "training_sets" / "training_set.parquet"

    @staticmethod
    def _source_segment_features_path(source_root: Path) -> Path:
        return source_root / "derived" / "representations" / "segment_features.parquet"

    @classmethod
    def _source_feature_path(cls, source_root: Path) -> Path:
        """On-disk parquet the source's feature columns/values come from.

        Prefers the assembled training set (features + labels for the project's
        whole labeled set in one file); falls back to the raw segment-features
        store for sources processed but never assembled into a training set.
        """
        ts = cls._source_training_path(source_root)
        return ts if ts.exists() else cls._source_segment_features_path(source_root)

    def _source_feature_cols(self, source_root: Path) -> set[str] | None:
        cols = self._parquet_columns(self._source_feature_path(source_root))
        return self._feature_cols(cols) if cols is not None else None

    def _load_labels(self, source_root: Path) -> pd.DataFrame | None:
        """One labeled row per segment as ``[segment_id, review_label, confidence]``.

        Prefers the source's assembled training set
        (``derived/training_sets/training_set.parquet``) — the project's full
        labeled set, i.e. every reviewer/seed/feedback example its own models
        train on.  Falls back to the Review-tab decision log
        (``derived/review_labels/reviewer_labels.parquet``) for sources reviewed
        but never assembled into a training set.
        """
        ts_path = self._source_training_path(source_root)
        if ts_path.exists():
            df = self._labels_from_training_set(ts_path)
            if df is not None and not df.empty:
                return df
        return self._labels_from_review_log(source_root)

    @classmethod
    def _labels_from_training_set(cls, ts_path: Path) -> pd.DataFrame | None:
        """Per-segment labels taken from an assembled training set's own rows."""
        cols = cls._parquet_columns(ts_path)
        if not cols or "segment_id" not in cols or "label" not in cols:
            return None
        want = [c for c in ("segment_id", "label", "reviewer_confidence") if c in cols]
        try:
            df = pd.read_parquet(ts_path, columns=want)
        except Exception:
            return None
        df = df.rename(columns={"label": "review_label"})
        df["review_label"] = df["review_label"].astype(str)
        if "reviewer_confidence" in df.columns:
            df["confidence"] = pd.to_numeric(
                df["reviewer_confidence"], errors="coerce",
            ).fillna(1.0)
        else:
            df["confidence"] = 1.0
        # Training sets carry one row per segment, but de-dupe defensively.
        df = df.drop_duplicates("segment_id", keep="last")
        return df[["segment_id", "review_label", "confidence"]]

    @staticmethod
    def _labels_from_review_log(source_root: Path) -> pd.DataFrame | None:
        p = source_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception:
            return None
        if "segment_id" not in df.columns or "review_label" not in df.columns:
            return None
        if "confidence" not in df.columns:
            df["confidence"] = 1.0
        # Keep the most recent label per segment when duplicates exist.
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").drop_duplicates("segment_id", keep="last")
        else:
            df = df.drop_duplicates("segment_id", keep="last")
        return df

    @staticmethod
    def _read_behaviors(root: Path) -> dict[str, str]:
        """Return {behavior_id: name} from a project's behavior_definitions.yaml."""
        import yaml  # noqa: PLC0415
        p = root / "config" / "behavior_definitions.yaml"
        if not p.exists():
            return {}
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        out: dict[str, str] = {}
        for b in data.get("behaviors", []) or []:
            bid = str(b.get("behavior_id") or b.get("id") or "").strip()
            name = str(b.get("name") or "").strip()
            if bid:
                out[bid] = name or bid
        return out

    @staticmethod
    def _resolve_host_behavior(
        raw_label: str,
        source_behaviors: dict[str, str],
        host_name_to_id: dict[str, str],
        name_overrides: dict[str, str] | None = None,
    ) -> tuple[str, str, bool]:
        """Map a source review_label to (host_behavior_id, host_name, via_alias).

        Matching order: exact name (case-insensitive), then a manual name
        alias from ``name_overrides`` (source name -> host name).  Returns
        ("", "", False) when there is no host behaviour the label maps to.
        """
        name_overrides = name_overrides or {}
        label = str(raw_label).strip()
        if label in _PASSTHROUGH_LABELS:
            # no_behavior is a universal id shared by every project.
            return label, "No Behavior", False
        # Behaviour UUID → name → host id by name.
        source_name = source_behaviors.get(label, "")
        if not source_name:
            return "", "", False
        host_bid = host_name_to_id.get(source_name.lower(), "")
        if host_bid:
            return host_bid, source_name, False
        # Manual remap: source behaviour name → host behaviour name.
        override_name = name_overrides.get(source_name.lower(), "")
        if override_name:
            host_bid = host_name_to_id.get(override_name.lower(), "")
            if host_bid:
                return host_bid, override_name, True
        return "", "", False

    def _decisioned_host_bid(
        self,
        raw_label: str,
        source_behaviors: dict[str, str],
        host_name_to_id: dict[str, str],
        name_overrides: dict[str, str] | None,
        behavior_decisions: dict[str, str] | None,
    ) -> str:
        """Resolve a source label to the host behaviour id it imports as ("" = drop).

        Pure (no side effects): an explicit per-behaviour decision wins, with
        ``AUTO_CREATE_BEHAVIOR`` resolving to the *source* behaviour id (the id the
        definition is created under by :meth:`_auto_create_behavior`) and
        ``SKIP_BEHAVIOR`` dropping the label.  With no decision it falls back to
        the existing name/alias match, so the examples-only flow is unchanged.
        """
        decision = (behavior_decisions or {}).get(str(raw_label))
        if decision == SKIP_BEHAVIOR:
            return ""
        if decision == AUTO_CREATE_BEHAVIOR:
            return str(raw_label)
        if decision:
            return str(decision)
        host_bid, _, _ = self._resolve_host_behavior(
            raw_label, source_behaviors, host_name_to_id, name_overrides,
        )
        return host_bid

    def _host_training_columns(self, host_root: Path) -> list[str]:
        """Column schema for host training rows.

        Uses the assembled training set's columns when it exists; otherwise
        derives them from the host's extracted segment features plus the standard
        training-set bookkeeping columns, so a features-extracted-but-untrained
        project gets a well-formed ``training_set.parquet`` on first baseline
        import (``merge_and_snapshot_training_set`` creates the file).
        """
        ts_path = self._source_training_path(host_root)
        ts_cols = self._parquet_columns(ts_path)
        if ts_cols:
            return list(ts_cols)
        seg_cols = self._parquet_columns(self._source_segment_features_path(host_root)) or []
        host_cols = list(seg_cols)
        for c in ("segment_id", "label", "label_source", "reviewer_confidence", "session_id"):
            if c not in host_cols:
                host_cols.append(c)
        return host_cols

    # ------------------------------------------------------------------
    # Behaviour name remapping
    # ------------------------------------------------------------------

    def list_host_behaviors(self, host_root: Path) -> list[tuple[str, str]]:
        """Return [(behavior_id, name)] defined in the host, sorted by name.

        Excludes the universal ``no_behavior`` id (it never needs remapping).
        """
        behaviors = self._read_behaviors(host_root)
        out = [
            (bid, name)
            for bid, name in behaviors.items()
            if bid not in _PASSTHROUGH_LABELS
        ]
        out.sort(key=lambda x: x[1].lower())
        return out

    @staticmethod
    def suggest_host_match(source_name: str, host_names: list[str]) -> str:
        """Best-guess host behaviour name for ``source_name`` (or "").

        Uses substring containment ("Dip" ⊂ "Head Dip"), shared word tokens,
        and overall string similarity.  Returns "" when nothing is close.
        """
        s = source_name.lower().strip()
        if not s:
            return ""
        s_tokens = {t for t in s.split() if t}
        best_name, best_score = "", 0.0
        for hn in host_names:
            h = hn.lower().strip()
            if not h:
                continue
            if h == s:
                return hn
            score = difflib.SequenceMatcher(None, s, h).ratio()
            if h in s or s in h:
                score = max(score, 0.85)
            shared = s_tokens & {t for t in h.split() if t}
            if shared:
                score = max(score, 0.6 + 0.1 * len(shared))
            if score > best_score:
                best_score, best_name = score, hn
        return best_name if best_score >= 0.6 else ""

    @staticmethod
    def _aliases_path(host_root: Path) -> Path:
        return host_root / "config" / "behavior_aliases.json"

    @classmethod
    def load_aliases(cls, host_root: Path) -> dict[str, str]:
        """Load the host's {source_name_lower: host_name} remap table."""
        p = cls._aliases_path(host_root)
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.warning("Could not read behavior_aliases.json in %s", host_root)
            return {}
        return {
            str(k).strip().lower(): str(v).strip()
            for k, v in data.items()
            if str(v).strip()
        }

    @classmethod
    def save_aliases(cls, host_root: Path, aliases: dict[str, str]) -> Path:
        """Persist the {source_name_lower: host_name} remap table."""
        p = cls._aliases_path(host_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        clean = {
            str(k).strip().lower(): str(v).strip()
            for k, v in aliases.items()
            if str(v).strip()
        }
        p.write_text(
            json.dumps(clean, indent=2, sort_keys=True), encoding="utf-8"
        )
        return p

    # ------------------------------------------------------------------
    # Model import — apply another project's trained models in this project
    # ------------------------------------------------------------------
    #
    # Unlike example import (which adds labeled rows to the training set), this
    # copies a source project's trained behaviour models into the host so they
    # can score the host's *already-extracted* features — no re-extraction, no
    # new project.  The same feature-schema / keypoint-rename compatibility used
    # for examples gates it; the model's feature column names are realigned onto
    # the host scheme so it runs natively here.

    # Models need near-complete feature coverage (a few missing columns are
    # backfilled with 0.0 at score time, but a real schema gap is unsafe).
    MODEL_COMPAT_THRESHOLD = 0.98

    @staticmethod
    def _models_dir(root: Path) -> Path:
        return root / "derived" / "models"

    @staticmethod
    def _model_feature_columns(model_dir: Path) -> list[str]:
        """Feature columns a model expects.

        Read from ``model_card.yaml`` so we don't have to unpickle the
        classifier (which can require the exact training sklearn version);
        falls back to ``model_state.pkl`` only if the card is missing.
        """
        import yaml  # noqa: PLC0415

        card = model_dir / "model_card.yaml"
        if card.exists():
            try:
                data = yaml.safe_load(card.read_text(encoding="utf-8")) or {}
                cols = data.get("feature_columns")
                if isinstance(cols, list) and cols:
                    return [str(c) for c in cols]
            except Exception:
                logger.debug("Failed reading model_card.yaml in %s", model_dir, exc_info=True)
        try:
            import pickle  # noqa: PLC0415
            with open(model_dir / "model_state.pkl", "rb") as f:
                payload = pickle.load(f)
            return [str(c) for c in payload.get("feature_cols", [])]
        except Exception:
            logger.debug("Failed reading feature_cols from %s", model_dir, exc_info=True)
            return []

    def list_source_models(self, source_root: Path) -> list[SourceModel]:
        """Trained behaviour models in a source project (``no_behavior`` excluded)."""
        out: list[SourceModel] = []
        behaviors = self._read_behaviors(source_root)
        mdir = self._models_dir(source_root)
        if not mdir.exists():
            return out
        for p in sorted(mdir.iterdir()):
            if not (p.is_dir() and p.name.startswith("behavior_model_")
                    and (p / "model_state.pkl").exists()):
                continue
            settings = read_json(p / "run_settings.json", {})
            bid = str(settings.get("target_behavior", "")).strip()
            if not bid or bid in _PASSTHROUGH_LABELS or "no_behavior" in bid.lower():
                continue
            name = behaviors.get(bid, p.name.removeprefix("behavior_model_"))
            out.append(SourceModel(
                model_dir=p.name, behavior_id=bid, behavior_name=name,
                feature_columns=self._model_feature_columns(p),
            ))
        return out

    def preview_model_import(
        self,
        host_root: Path,
        source_root: Path,
        model_dirs: list[str] | None = None,
        name_overrides: dict[str, str] | None = None,
        compute_diagnostics: bool = True,
    ) -> ModelImportPreview:
        """Inspect which source models can run in the host, without copying.

        Reports per-model feature coverage (after aligning the model's keypoint
        scheme onto the host's) and how each model's behaviour maps onto a host
        behaviour.  ``model_dirs`` limits the preview to specific source model
        directories; when omitted, every trained model is considered.
        """
        tag = source_root.name
        if name_overrides is None:
            name_overrides = self.load_aliases(host_root)
        pv = ModelImportPreview(source_root=source_root, tag=tag)

        host_features = self._host_feature_cols(host_root) or set()
        pv.host_feature_count = len(host_features)

        source_behaviors = self._read_behaviors(source_root)
        host_name_to_id = {
            name.lower(): bid for bid, name in self._read_behaviors(host_root).items()
        }

        models = self.list_source_models(source_root)
        if model_dirs is not None:
            want = set(model_dirs)
            models = [m for m in models if m.model_dir in want]

        all_model_cols: set[str] = set()
        for m in models:
            all_model_cols.update(m.feature_columns)
        kp_rename = (
            self._keypoint_rename_map(host_root, host_features, all_model_cols)
            if host_features and all_model_cols else {}
        )
        pv.keypoint_renames = kp_rename

        for m in models:
            col_rename = self._rename_cols(m.feature_columns, kp_rename) if kp_rename else {}
            renamed = [col_rename.get(c, c) for c in m.feature_columns]
            total = max(len(renamed), 1)
            missing_cols = [c for c in renamed if c not in host_features]
            present = total - len(missing_cols)
            host_bid, host_name, via_alias = self._resolve_host_behavior(
                m.behavior_id, source_behaviors, host_name_to_id, name_overrides,
            )
            pv.items.append(ModelImportItem(
                model=m,
                coverage=present / total,
                missing_features=len(missing_cols),
                host_behavior_id=host_bid,
                host_behavior_name=host_name,
                matched_by_alias=via_alias,
                compatible=(present / total) >= self.MODEL_COMPAT_THRESHOLD,
                missing_columns=missing_cols,
            ))

        if compute_diagnostics and host_features and all_model_cols:
            try:
                col_rename_all = self._rename_cols(all_model_cols, kp_rename) if kp_rename else {}
                renamed_all = {col_rename_all.get(c, c) for c in all_model_cols}
                shared = host_features & renamed_all
                pv.diagnostics = self._compute_diagnostics(
                    host_root, source_root, shared, kp_rename,
                )
            except Exception:
                logger.debug("Failed computing model-import diagnostics", exc_info=True)

        return pv

    # Ordered (label, token-substrings) — a missing column is attributed to the
    # first family whose token it contains, so put the most specific first.
    _MISSING_FEATURE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Video / optical-flow context", (
            "flow_", "surface_motion_energy", "motion_energy",
            "_near_target", "_near_nose", "_to_target", "_tmt", "optic",
        )),
        ("ROI / zone context", ("roi", "zone", "in_arena", "in_roi", "_target_dist")),
        ("Oscillation / frequency", (
            "oscillation", "autocorr", "movement_frequency", "frequency", "_power",
        )),
        ("Velocity / speed / acceleration", (
            "velocity", "_speed", "accel", "jerk",
        )),
        ("Orientation / angles / posture", (
            "orientation", "pitch", "angle", "curvature", "head_direction", "spine",
        )),
        ("Inter-keypoint distances", ("dist_", "_to_")),
    )

    @classmethod
    def _classify_missing_columns(cls, cols: "set[str]") -> list[tuple[str, int]]:
        """Group missing column names into recognizable feature families."""
        counts: dict[str, int] = {}
        for col in cols:
            lc = str(col).lower()
            label = "Other features"
            for grp_label, tokens in cls._MISSING_FEATURE_GROUPS:
                if any(tok in lc for tok in tokens):
                    label = grp_label
                    break
            counts[label] = counts.get(label, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    @classmethod
    def build_coverage_diagnosis(
        cls,
        items: list[ModelImportItem],
        diagnostics: "CompatibilityDiagnostics | None",
    ) -> "CoverageDiagnosis | None":
        """Explain why models are blocked and how to fix it (``None`` if none are).

        Aggregates the host-aligned feature columns missing across all
        below-threshold models, groups them into feature families, and pairs the
        families (plus any extraction-config mismatches) with concrete fix steps.
        """
        blocked = [it for it in items if it.has_model_gap]
        if not blocked:
            return None

        missing: set[str] = set()
        for it in blocked:
            missing.update(it.missing_columns)
        groups = cls._classify_missing_columns(missing)
        group_labels = {label for label, _ in groups}

        diag = CoverageDiagnosis(
            models_blocked=len(blocked),
            models_total=len(items),
            worst_coverage=min((it.coverage for it in blocked), default=1.0),
            missing_total=len(missing),
            missing_groups=groups,
            sample_missing=sorted(missing)[:12],
        )

        cfg_mismatches = list(diagnostics.config_mismatches) if diagnostics else []
        video_like = {"Video / optical-flow context", "ROI / zone context"}
        cfg_text = "; ".join(cfg_mismatches)
        video_implicated = bool(group_labels & video_like) or (
            "use_video_features" in cfg_text
        )

        # ── Likely causes ─────────────────────────────────────────────
        if video_implicated:
            diag.causes.append(
                "This project is missing video/context features (optical flow, "
                "surface motion energy, target/ROI distances). The source models "
                "were trained with “Include video features” on, but this project "
                "extracted pose-only features (or its sessions have no linked video)."
            )
        if cfg_mismatches:
            diag.causes.append(
                "Feature-extraction settings differ between the projects: "
                + cfg_text + "."
            )
        only_geom = group_labels and group_labels <= {
            "Inter-keypoint distances", "Orientation / angles / posture", "Other features",
        }
        if only_geom and not cfg_mismatches:
            diag.causes.append(
                "The missing columns are geometric (distances/angles). The projects "
                "likely use different pose keypoints, or different invariant-feature "
                "toggles, so some columns can't be produced here."
            )
        if not diag.causes:
            diag.causes.append(
                "This project's extracted feature set doesn't include all the "
                "columns the source models were trained on."
            )

        # ── Ordered fixes ─────────────────────────────────────────────
        if video_implicated:
            diag.fixes.append(
                "Open the Features tab, enable “Include video features”, and confirm "
                "every session has a linked video. Then re-run feature extraction."
            )
        if cfg_mismatches:
            diag.fixes.append(
                "Match these Features-tab settings to the source project, then "
                "re-extract: " + cfg_text + "."
            )
        diag.fixes.append(
            "After re-extracting, re-open this Import Baseline dialog — model "
            "coverage should reach ~100% and the models will import."
        )
        if only_geom:
            diag.fixes.append(
                "If coverage is still low, the projects use different keypoints — "
                "check the keypoint mapping (Direct Use / keypoint map) so the "
                "source columns realign onto this project's scheme."
            )
        diag.fixes.append(
            "Alternatively, you can still import the labeled examples now (they "
            "are unaffected) and train fresh models in this project."
        )
        return diag

    def import_models(
        self,
        host_root: Path,
        source_root: Path,
        model_dirs: list[str],
        behavior_decisions: dict[str, str] | None = None,
        name_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Copy selected source models into the host project.

        ``behavior_decisions`` maps each source behaviour id to the host
        behaviour id it should predict as, or one of ``AUTO_CREATE_BEHAVIOR``
        (add the source behaviour definition to this project) /
        ``SKIP_BEHAVIOR``.  When a behaviour isn't listed, it falls back to the
        name/alias match, and to auto-create if still unmatched.

        Each copied model is namespaced (so it never clobbers the host's own
        models), its feature columns realigned onto the host keypoint scheme,
        and its stale source-scored predictions dropped so it re-scores on host
        features when next run.  Returns the imported / skipped models.
        """
        if name_overrides is None:
            name_overrides = self.load_aliases(host_root)
        pv = self.preview_model_import(
            host_root, source_root, model_dirs, name_overrides, compute_diagnostics=False,
        )
        decisions = dict(behavior_decisions or {})

        imported: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for item in pv.items:
            m = item.model
            if not item.compatible:
                skipped.append({"model_dir": m.model_dir, "reason": "incompatible features"})
                continue
            decision = decisions.get(m.behavior_id) or item.host_behavior_id or AUTO_CREATE_BEHAVIOR
            if decision == SKIP_BEHAVIOR:
                skipped.append({"model_dir": m.model_dir, "reason": "behaviour skipped"})
                continue
            if decision == AUTO_CREATE_BEHAVIOR:
                host_bid = self._auto_create_behavior(host_root, source_root, m.behavior_id)
            else:
                host_bid = decision
            if not host_bid:
                skipped.append({"model_dir": m.model_dir, "reason": "no behaviour mapping"})
                continue
            new_dir = self._copy_and_rewrite_model(
                host_root, source_root, m, host_bid, pv.tag, pv.keypoint_renames,
            )
            imported.append({
                "model_dir": new_dir,
                "source_model_dir": m.model_dir,
                "behavior_id": host_bid,
                "behavior_name": self._read_behaviors(host_root).get(host_bid, m.behavior_name),
            })

        self._record_model_import(host_root, pv.tag, source_root, imported)
        logger.info(
            "Imported %d model(s) from '%s' into %s (%d skipped).",
            len(imported), pv.tag, host_root.name, len(skipped),
        )
        return {
            "status": "success" if imported else "error",
            "imported": imported,
            "skipped": skipped,
            "tag": pv.tag,
            "error": "" if imported else "No compatible models could be imported.",
            "preview": pv,
        }

    @staticmethod
    def _namespaced_model_dir(behavior_name: str, tag: str) -> str:
        safe_beh = re.sub(r"[^0-9A-Za-z]+", "_", str(behavior_name)).strip("_") or "model"
        safe_tag = re.sub(r"[^0-9A-Za-z]+", "_", str(tag)).strip("_") or "import"
        return f"behavior_model_{safe_beh}__{safe_tag}"

    def _auto_create_behavior(
        self, host_root: Path, source_root: Path, source_behavior_id: str,
    ) -> str:
        """Copy the source project's behaviour definition into the host verbatim.

        Preserves the source behaviour id so the imported model's target lines
        up, and uses the source naming scheme for the new definition.  No-op
        (returns the id) if the host already defines that behaviour id.
        """
        import yaml  # noqa: PLC0415

        src_path = source_root / "config" / "behavior_definitions.yaml"
        host_path = host_root / "config" / "behavior_definitions.yaml"
        try:
            src_data = yaml.safe_load(src_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return ""
        src_item = None
        for b in src_data.get("behaviors", []) or []:
            if str(b.get("behavior_id") or b.get("id") or "").strip() == source_behavior_id:
                src_item = dict(b)
                break
        if src_item is None:
            return ""

        host_data: dict[str, Any] = {}
        if host_path.exists():
            try:
                host_data = yaml.safe_load(host_path.read_text(encoding="utf-8")) or {}
            except Exception:
                host_data = {}
        behaviors = host_data.get("behaviors") or []
        existing = {
            str(b.get("behavior_id") or b.get("id") or "").strip() for b in behaviors
        }
        if source_behavior_id not in existing:
            behaviors.append(src_item)
            host_data["behaviors"] = behaviors
            host_path.parent.mkdir(parents=True, exist_ok=True)
            host_path.write_text(yaml.safe_dump(host_data, sort_keys=False), encoding="utf-8")
        return source_behavior_id

    def _copy_and_rewrite_model(
        self, host_root: Path, source_root: Path, model: SourceModel,
        host_behavior_id: str, tag: str, kp_rename: dict[str, str],
    ) -> str:
        """Copy one model dir into the host and realign it to this project."""
        import pickle  # noqa: PLC0415
        import yaml  # noqa: PLC0415

        src = self._models_dir(source_root) / model.model_dir
        new_name = self._namespaced_model_dir(model.behavior_name, tag)
        dst = self._models_dir(host_root) / new_name
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

        col_rename = self._rename_cols(model.feature_columns, kp_rename) if kp_rename else {}

        rs = read_json(dst / "run_settings.json", {})
        if isinstance(rs, dict):
            rs["target_behavior"] = host_behavior_id
            rs["model_version"] = new_name
            rs["imported_from"] = str(source_root)
            write_json(dst / "run_settings.json", rs)

        card_path = dst / "model_card.yaml"
        if card_path.exists():
            try:
                card = yaml.safe_load(card_path.read_text(encoding="utf-8")) or {}
                card["model_version"] = new_name
                if isinstance(card.get("labels"), list):
                    card["labels"] = [
                        host_behavior_id if str(lbl) == model.behavior_id else lbl
                        for lbl in card["labels"]
                    ]
                if col_rename and isinstance(card.get("feature_columns"), list):
                    card["feature_columns"] = [
                        col_rename.get(str(c), str(c)) for c in card["feature_columns"]
                    ]
                card["imported_from"] = str(source_root)
                card_path.write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")
            except Exception:
                logger.debug("Failed rewriting model_card for %s", new_name, exc_info=True)

        # Only the keypoint-differing case needs the pickle rewritten (which
        # requires unpickling); identical schemes copy verbatim and run as-is.
        if col_rename:
            try:
                with open(dst / "model_state.pkl", "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict) and "feature_cols" in payload:
                    payload["feature_cols"] = [
                        col_rename.get(str(c), str(c)) for c in payload["feature_cols"]
                    ]
                    with open(dst / "model_state.pkl", "wb") as f:
                        pickle.dump(payload, f)
            except Exception:
                logger.exception(
                    "Could not realign feature names for imported model %s; it may "
                    "not score correctly against this project.", new_name,
                )

        # Drop predictions scored on the *source* project's segments — they are
        # meaningless here and would otherwise masquerade as host results until
        # the model is re-run.
        for stale in (
            "segment_predictions.parquet", "segment_uncertainty.parquet",
            "validation_predictions.parquet",
        ):
            try:
                (dst / stale).unlink()
            except FileNotFoundError:
                pass
            except Exception:
                logger.debug("Could not remove stale %s in %s", stale, new_name, exc_info=True)

        return new_name

    @staticmethod
    def _model_imports_manifest_path(host_root: Path) -> Path:
        return host_root / "derived" / "models" / "model_imports.json"

    def list_model_imports(self, host_root: Path) -> list[dict[str, Any]]:
        """Sources whose models have been imported into ``host_root``."""
        raw = read_json(self._model_imports_manifest_path(host_root), {"imports": []})
        items = raw.get("imports", []) if isinstance(raw, dict) else []
        return [r for r in items if isinstance(r, dict)]

    def _record_model_import(
        self, host_root: Path, tag: str, source_root: Path, imported: list[dict[str, Any]],
    ) -> None:
        records = {r.get("tag"): r for r in self.list_model_imports(host_root)}
        existing = records.get(tag) or {"tag": tag, "models": []}
        by_dir = {m["model_dir"]: m for m in existing.get("models", []) if "model_dir" in m}
        for m in imported:
            by_dir[m["model_dir"]] = m
        existing["models"] = list(by_dir.values())
        existing["source_root"] = str(source_root)
        existing["imported_at"] = datetime.utcnow().isoformat(timespec="seconds")
        records[tag] = existing
        write_json(
            self._model_imports_manifest_path(host_root),
            {"imports": [r for r in records.values() if r]},
        )

    def remove_model_import(self, host_root: Path, tag: str) -> dict[str, Any]:
        """Delete a source's imported model directories and forget them.

        Auto-created behaviour definitions are left in place (other things may
        now reference them); only the copied model directories are removed.
        """
        keep: list[dict[str, Any]] = []
        removed = 0
        for r in self.list_model_imports(host_root):
            if r.get("tag") != tag:
                keep.append(r)
                continue
            for m in r.get("models", []):
                d = self._models_dir(host_root) / str(m.get("model_dir", ""))
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
        write_json(
            self._model_imports_manifest_path(host_root), {"imports": keep},
        )
        logger.info("Removed %d imported model(s) for '%s' from %s.", removed, tag, host_root.name)
        return {"status": "success", "tag": tag, "removed_models": removed}

    # ------------------------------------------------------------------
    # Baseline import — seed a new project from another project's whole basis
    # ------------------------------------------------------------------
    #
    # Unlike example import (training rows only) or model import (models only),
    # this brings over a source project's *clips + labeled feature rows + trained
    # models* together, governed by one per-behaviour decision set, so a project
    # that has extracted features but not yet run active learning can either run
    # the imported models immediately or fold the imported clips/features into its
    # own training pool and train/refine.

    def _host_is_new(self, host_root: Path) -> bool:
        """True when the host has no training set, no behaviours, and no models.

        Such a project has (at most) extracted features — the baseline-import
        target — so the summary can tell the user they're seeding a fresh project
        rather than refining an existing one.
        """
        if self._source_training_path(host_root).exists():
            return False
        if any(b not in _PASSTHROUGH_LABELS for b in self._read_behaviors(host_root)):
            return False
        mdir = self._models_dir(host_root)
        if mdir.exists():
            for p in mdir.iterdir():
                if (p.is_dir() and p.name.startswith("behavior_model_")
                        and (p / "model_state.pkl").exists()):
                    return False
        return True

    def preview_baseline(
        self,
        host_root: Path,
        source_root: Path,
        name_overrides: dict[str, str] | None = None,
    ) -> BaselinePreview:
        """Detection summary for importing ``source_root`` as a baseline.

        Composes the example preview (feature-schema coverage, per-behaviour
        labeled counts, diagnostics) and the model preview (per-model coverage),
        and reports whether the host is new vs already has matching behaviours so
        the UI can warn and require an explicit Accept before importing.
        """
        if name_overrides is None:
            name_overrides = self.load_aliases(host_root)
        tag = source_root.name
        pv = BaselinePreview(source_root=source_root, tag=tag)

        ex = self.preview(
            host_root, source_root, tag=tag, name_overrides=name_overrides,
            compute_diagnostics=True,
        )
        mpv = self.preview_model_import(
            host_root, source_root, name_overrides=name_overrides,
            compute_diagnostics=False,
        )

        pv.host_feature_count = ex.host_feature_count
        pv.coverage = ex.coverage
        pv.keypoint_renames = dict(ex.keypoint_renames)
        pv.diagnostics = ex.diagnostics
        pv.model_count = len(mpv.items)
        pv.host_is_new = self._host_is_new(host_root)

        pv.schema_ok = ex.host_feature_count > 0 and ex.coverage >= COMPAT_THRESHOLD
        if ex.host_feature_count == 0 and mpv.host_feature_count > 0:
            # Source has no labeled examples to gauge coverage; fall back to
            # whether any trained model's features the host covers.
            pv.host_feature_count = mpv.host_feature_count
            pv.schema_ok = any(it.compatible for it in mpv.items)
        if not pv.schema_ok:
            pv.reason = ex.reason or "Incompatible feature schema."

        rows: dict[str, BaselineBehaviorRow] = {}
        for m in ex.behavior_mappings:
            bid = m.source_behavior_id
            if bid in _PASSTHROUGH_LABELS:
                continue
            row = rows.setdefault(
                bid, BaselineBehaviorRow(source_behavior_id=bid, source_name=m.source_name)
            )
            row.example_count = m.example_count
            if m.host_behavior_id:
                row.matched_host_id = m.host_behavior_id
                row.matched_host_name = m.host_name
        for it in mpv.items:
            bid = it.model.behavior_id
            if bid in _PASSTHROUGH_LABELS:
                continue
            row = rows.setdefault(
                bid, BaselineBehaviorRow(source_behavior_id=bid, source_name=it.model.behavior_name)
            )
            row.has_model = True
            row.model_coverage = it.coverage
            row.model_compatible = it.compatible
            if it.host_behavior_id and not row.matched_host_id:
                row.matched_host_id = it.host_behavior_id
                row.matched_host_name = it.host_behavior_name

        pv.rows = sorted(
            rows.values(), key=lambda r: (-r.example_count, r.source_name.lower())
        )
        # Explain any blocked models (which columns are missing + how to fix).
        # ``pv.diagnostics`` carries the config-mismatch list from the example
        # preview (the model preview skips diagnostics for speed).
        pv.coverage_diagnosis = self.build_coverage_diagnosis(mpv.items, pv.diagnostics)
        return pv

    def import_baseline(
        self,
        host_root: Path,
        source_root: Path,
        behavior_decisions: dict[str, str] | None = None,
        name_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Import a source project's clips + labeled rows + models as a baseline.

        ``behavior_decisions`` (source behaviour id -> host id /
        ``AUTO_CREATE_BEHAVIOR`` / ``SKIP_BEHAVIOR``) governs both halves at once.
        Best-effort: a failure on one half does not abort the other.  Returns a
        combined summary.
        """
        if name_overrides is None:
            name_overrides = self.load_aliases(host_root)
        decisions = dict(behavior_decisions or {})
        tag = source_root.name

        # 1. Clips + labeled feature rows into the training pool.
        ex_res = self.import_examples(
            host_root, source_root, tag=tag, name_overrides=name_overrides,
            behavior_decisions=decisions,
        )

        # 2. Trained models for the same non-skipped, compatible behaviours.
        mpv = self.preview_model_import(
            host_root, source_root, name_overrides=name_overrides,
            compute_diagnostics=False,
        )
        model_dirs = [
            it.model.model_dir for it in mpv.items
            if it.compatible and decisions.get(it.model.behavior_id) != SKIP_BEHAVIOR
        ]
        if model_dirs:
            model_res = self.import_models(
                host_root, source_root, model_dirs,
                behavior_decisions=decisions, name_overrides=name_overrides,
            )
        else:
            model_res = {"status": "skipped", "imported": [], "skipped": []}

        imported_rows = (
            int(ex_res.get("imported_rows", 0)) if ex_res.get("status") == "success" else 0
        )
        imported_models = list(model_res.get("imported", []))
        ok = bool(imported_rows or imported_models)
        logger.info(
            "Baseline import from '%s' into %s: %d row(s), %d model(s).",
            tag, host_root.name, imported_rows, len(imported_models),
        )
        return {
            "status": "success" if ok else "error",
            "tag": tag,
            "imported_rows": imported_rows,
            "review_registered": int(ex_res.get("review_registered", 0)),
            "imported_models": imported_models,
            "skipped_models": list(model_res.get("skipped", [])),
            "examples_result": ex_res,
            "models_result": model_res,
            "error": "" if ok else (
                ex_res.get("error") or model_res.get("error") or "Nothing imported."
            ),
        }
