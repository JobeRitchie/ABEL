"""Behavior definition CRUD with project YAML persistence."""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from abel.models.schemas import BehaviorDefinition
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml


logger = logging.getLogger(__name__)

NO_BEHAVIOR_ID = "no_behavior"

# ---------------------------------------------------------------------------
# Built-in behavior templates
# ---------------------------------------------------------------------------
_TEMPLATES: dict[str, list[dict]] = {
    "Rodent Basic (4 behaviors)": [
        {
            "name": "Grooming", "short_name": "groom",
            "description": "Self-directed grooming behavior.",
            "operational_definition": (
                "Repetitive forepaw-directed stroking movements aimed at the face, head, or body, "
                "often preceded by licking of forepaws."
            ),
            "inclusion_criteria": "Forepaw contact with fur; rhythmic repetitive motion; face/body directed.",
            "exclusion_criteria": "Locomotor scratching; single paw shake; incidental brief touch.",
            "min_duration_sec": 0.5, "color": "#4A90E2", "keyboard_shortcut": "g", "review_priority": 1,
        },
        {
            "name": "Rearing", "short_name": "rear",
            "description": "Vertical exploratory posture.",
            "operational_definition": "Animal stands on hindlimbs with forelimbs raised fully off the substrate.",
            "inclusion_criteria": "Forelimbs fully off ground; vertical body posture.",
            "exclusion_criteria": "Brief stumbles; supported rearing only when wall contact is incidental.",
            "min_duration_sec": 0.2, "color": "#7ED321", "keyboard_shortcut": "r", "review_priority": 2,
        },
        {
            "name": "Freezing", "short_name": "freeze",
            "description": "Complete immobility — fear response.",
            "operational_definition": (
                "All tracked body parts remain stationary; centroid speed near zero "
                "for a sustained minimum duration."
            ),
            "inclusion_criteria": "Zero locomotion; all keypoints below movement threshold for >= 1 s.",
            "exclusion_criteria": "Brief pauses during locomotion; sleep postures.",
            "min_duration_sec": 1.0, "color": "#9013FE", "keyboard_shortcut": "z", "review_priority": 1,
        },
        {
            "name": "Locomotion", "short_name": "loco",
            "description": "Directed movement across the arena.",
            "operational_definition": "Centroid speed exceeds threshold continuously for the minimum duration.",
            "inclusion_criteria": "Centroid speed > 5 cm/s; duration > 0.5 s.",
            "exclusion_criteria": "In-place pivoting; head movement only; rearing.",
            "min_duration_sec": 0.5, "color": "#F5A623", "keyboard_shortcut": "l", "review_priority": 3,
        },
    ],
    "Open Field Extended (6 behaviors)": [
        {
            "name": "Grooming", "short_name": "groom",
            "description": "Self-directed grooming behavior.",
            "operational_definition": "Repetitive forepaw-directed stroking aimed at face/body.",
            "inclusion_criteria": "Forepaw contact with fur; rhythmic motion.",
            "exclusion_criteria": "Scratching; single paw touches.",
            "min_duration_sec": 0.5, "color": "#4A90E2", "keyboard_shortcut": "g", "review_priority": 1,
        },
        {
            "name": "Rearing", "short_name": "rear",
            "description": "Vertical exploratory posture.",
            "operational_definition": "Animal stands on hindlimbs with forelimbs raised.",
            "inclusion_criteria": "Forelimbs fully off ground.",
            "exclusion_criteria": "Brief stumbles.",
            "min_duration_sec": 0.2, "color": "#7ED321", "keyboard_shortcut": "r", "review_priority": 2,
        },
        {
            "name": "Freezing", "short_name": "freeze",
            "description": "Complete immobility.",
            "operational_definition": "All keypoints near-stationary for >= 1 s.",
            "inclusion_criteria": "Zero locomotion; all keypoints stationary.",
            "exclusion_criteria": "Brief pauses.",
            "min_duration_sec": 1.0, "color": "#9013FE", "keyboard_shortcut": "z", "review_priority": 1,
        },
        {
            "name": "Locomotion", "short_name": "loco",
            "description": "Directed movement.", "operational_definition": "Centroid speed > 5 cm/s for > 0.5 s.",
            "inclusion_criteria": "Centroid speed threshold met.", "exclusion_criteria": "Pivoting; rearing.",
            "min_duration_sec": 0.5, "color": "#F5A623", "keyboard_shortcut": "l", "review_priority": 3,
        },
        {
            "name": "Investigation", "short_name": "invest",
            "description": "Directed sniffing / nose-contact investigation.",
            "operational_definition": "Nose directed toward object/wall at close proximity with sustained contact.",
            "inclusion_criteria": "Nose within 2 cm of object/wall; directed head movement.",
            "exclusion_criteria": "Passing locomotion; grooming near object.",
            "min_duration_sec": 0.3, "color": "#50E3C2", "keyboard_shortcut": "i", "review_priority": 2,
        },
        {
            "name": "Eating", "short_name": "eat",
            "description": "Food consumption.",
            "operational_definition": "Sustained jaw movements and forepaw manipulation of food pellet at feeder.",
            "inclusion_criteria": "Mouse at food source; jaw/paw movements visible.",
            "exclusion_criteria": "Sniffing only; grooming near food area.",
            "min_duration_sec": 0.5, "color": "#D0021B", "keyboard_shortcut": "e", "review_priority": 2,
        },
    ],
    "EPM (3 behaviors)": [
        {
            "name": "Open Arm Entry", "short_name": "oae",
            "description": "Complete entry into open arm of EPM.",
            "operational_definition": "All four paws fully within an open arm zone.",
            "inclusion_criteria": "All four paws in open arm.", "exclusion_criteria": "Partial entries.",
            "min_duration_sec": 0.0, "color": "#F5A623", "keyboard_shortcut": "o", "review_priority": 1,
        },
        {
            "name": "Closed Arm Time", "short_name": "cat",
            "description": "Time spent within a closed arm of EPM.",
            "operational_definition": "Animal centroid located within closed arm zone.",
            "inclusion_criteria": "Centroid in closed arm.", "exclusion_criteria": "Transition regions.",
            "min_duration_sec": 0.0, "color": "#4A90E2", "keyboard_shortcut": "c", "review_priority": 2,
        },
        {
            "name": "Head Dip", "short_name": "hdip",
            "description": "Head dipping from open arm over platform edge.",
            "operational_definition": "Head/nose directed downward below platform level over open arm edge.",
            "inclusion_criteria": "Head below platform level.", "exclusion_criteria": "Brief postural adjustments.",
            "min_duration_sec": 0.2, "color": "#7ED321", "keyboard_shortcut": "h", "review_priority": 1,
        },
    ],
}


class BehaviorService:
    """CRUD service for behavior definitions, persisted to project config YAML."""

    def __init__(self) -> None:
        self._behaviors: list[BehaviorDefinition] = []
        self._project_root: Path | None = None

    # ------------------------------------------------------------------
    # Project binding
    # ------------------------------------------------------------------

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._load()

    @property
    def project_root(self) -> Path | None:
        return self._project_root

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def behaviors(self) -> list[BehaviorDefinition]:
        return list(self._behaviors)

    @property
    def template_names(self) -> list[str]:
        return list(_TEMPLATES.keys())

    # ------------------------------------------------------------------
    # Multi-animal structured labels
    # ------------------------------------------------------------------

    def get(self, behavior_id: str) -> "BehaviorDefinition | None":
        """Return the behavior definition for ``behavior_id``, or ``None``."""
        bid = str(behavior_id or "")
        return next((b for b in self._behaviors if str(b.behavior_id) == bid), None)

    def is_social(self, behavior_id: str) -> bool:
        b = self.get(behavior_id)
        return bool(b and b.is_social)

    def label_animal_fields(
        self,
        behavior_id: str,
        focal_animal_id: str | None,
        partner_animal_id: str | None = None,
    ) -> dict:
        """Derive the structured-label animal/role fields for a behavior.

        Returns a dict ready to splat into ``ReviewerLabelRecord`` / ``SeedExample``:

        * **solo** behavior -> ``social_role='none'``, no partner (matches legacy
          single-animal labels, so single-animal projects are unaffected).
        * **directed** social -> ``focal`` is the ``actor``; ``partner`` the recipient.
        * **mutual** social -> ``social_role='mutual'`` for the unordered pair.

        The *training label* stays the behavior id itself (identity-agnostic), so
        instances pool across animals ("a mouse is a mouse"); the animal fields
        only tell downstream feature extraction *which* animal(s) to use.
        """
        b = self.get(behavior_id)
        if b is None or not b.is_social:
            return {
                "focal_animal_id": focal_animal_id,
                "partner_animal_id": None,
                "social_role": "none",
            }
        role = "actor" if b.directionality == "directed" else "mutual"
        return {
            "focal_animal_id": focal_animal_id,
            "partner_animal_id": partner_animal_id,
            "social_role": role,
        }

    def aggregate_clip_labels(
        self,
        labels: "list[dict]",
        session_id: str,
        start_frame: int,
        end_frame: int,
    ) -> list[dict]:
        """Fan out per-clip structured labels to per-animal-segment records.

        ``labels`` is a list of ``{behavior_id, focal_animal_id,
        partner_animal_id}`` (as emitted by the soundboard). Each label is keyed
        to its focal animal's own segment (``seg_{animal}_{session}_{start}_{end}``)
        so instances pool by identity-agnostic behavior id at training time
        ("a mouse is a mouse"). Semantics:

        * **solo** -> labels the focal animal's segment.
        * **directed social** -> labels only the *actor*'s segment.
        * **mutual social** -> labels *both* animals' segments (both exhibit it).

        Multiple behaviors landing on the same animal-segment are merged into one
        pipe-joined ``review_label`` (the co-occurring convention the trainer
        expands), so they aren't collapsed to ``ambiguous`` and dropped.

        Returns a list of ``{segment_id, review_label, fields}`` dicts, where
        ``fields`` carries the structured animal/role columns for a single-behavior
        segment (empty for a merged, multi-behavior segment).
        """
        by_segment: dict[str, dict] = {}

        def _apply(bid: str, focal: str, partner: "str | None") -> None:
            if not bid or not focal:
                return
            fields = self.label_animal_fields(bid, focal, partner)
            seg_id = f"seg_{focal}_{session_id}_{int(start_frame)}_{int(end_frame)}"
            entry = by_segment.setdefault(seg_id, {"bids": [], "fields": []})
            if bid not in entry["bids"]:
                entry["bids"].append(bid)
                entry["fields"].append(fields)

        for lab in labels:
            bid = str(lab.get("behavior_id") or "")
            focal = lab.get("focal_animal_id")
            partner = lab.get("partner_animal_id")
            _apply(bid, focal, partner)
            b = self.get(bid)
            if partner and b and b.is_social and str(b.directionality) == "mutual":
                _apply(bid, partner, focal)

        out: list[dict] = []
        for seg_id, entry in by_segment.items():
            bids = sorted(set(entry["bids"]))
            out.append({
                "segment_id": seg_id,
                "review_label": "|".join(bids),
                "fields": entry["fields"][0] if len(bids) == 1 else {},
            })
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _config_path(self) -> Path:
        assert self._project_root is not None
        return self._project_root / "config" / "behavior_definitions.yaml"

    def _load(self) -> None:
        raw = read_yaml(self._config_path(), {})
        self._behaviors = []
        for item in raw.get("behaviors", []):
            try:
                self._behaviors.append(BehaviorDefinition.model_validate(item))
            except Exception:
                pass
        if self._ensure_system_behaviors() and self._project_root:
            self.save()

    def _ensure_system_behaviors(self) -> bool:
        """Ensure required built-in behavior labels exist in every project."""
        if any(str(b.behavior_id).strip() == NO_BEHAVIOR_ID for b in self._behaviors):
            return False
        self._behaviors.append(
            BehaviorDefinition(
                behavior_id=NO_BEHAVIOR_ID,
                name="No Behavior",
                short_name="none",
                description="Universal negative label indicating none of the defined behaviors.",
                operational_definition=(
                    "Use when the clip does not contain any behavior currently defined in this project."
                ),
                inclusion_criteria="No defined target behavior is present.",
                exclusion_criteria="Any clip where a defined behavior is clearly present.",
                min_duration_sec=0.0,
                review_priority=999,
                color="#90A4AE",
                keyboard_shortcut="n",
            )
        )
        return True

    def save(self) -> None:
        if not self._project_root:
            return
        write_yaml(
            self._config_path(),
            {"behaviors": [b.model_dump(mode="json") for b in self._behaviors]},
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, behavior: BehaviorDefinition) -> BehaviorDefinition:
        if not behavior.behavior_id:
            behavior = behavior.model_copy(update={"behavior_id": str(uuid.uuid4())})
        self._behaviors.append(behavior)
        self.save()
        return behavior

    def update(self, behavior_id: str, updated: BehaviorDefinition) -> bool:
        for i, b in enumerate(self._behaviors):
            if b.behavior_id == behavior_id:
                history = list(b.version_history) + [
                    {"timestamp": datetime.utcnow().isoformat(), "snapshot": b.model_dump(mode="json")}
                ]
                self._behaviors[i] = updated.model_copy(
                    update={"behavior_id": behavior_id, "version_history": history}
                )
                self.save()
                return True
        return False

    def delete(self, behavior_id: str) -> bool:
        if str(behavior_id).strip() == NO_BEHAVIOR_ID:
            return False
        before = len(self._behaviors)
        self._behaviors = [b for b in self._behaviors if b.behavior_id != behavior_id]
        if len(self._behaviors) < before:
            self.save()
            self._purge_trained_models(behavior_id)
            self._purge_behavior_references(behavior_id)
            return True
        return False

    def _purge_trained_models(self, behavior_id: str) -> list[str]:
        """Remove trained-model directories bound to a deleted behavior.

        Downstream tools (unified UMAP, behavior analytics, apply-models)
        discover behaviours by scanning ``derived/models`` for
        ``behavior_model_*`` directories. A leftover directory keeps a removed
        behaviour visible everywhere, so we delete every model whose
        ``run_settings.json`` target behaviour matches the id being removed.
        Directory names are unreliable (custom names vs. behaviour ids), so we
        match on the recorded target behaviour id rather than the folder name.

        Returns the list of removed directory names.
        """
        removed: list[str] = []
        if self._project_root is None:
            return removed
        models_root = self._project_root / "derived" / "models"
        if not models_root.exists():
            return removed
        bid = str(behavior_id).strip()
        if not bid:
            return removed
        for p in sorted(models_root.iterdir()):
            if not (p.is_dir() and p.name.startswith("behavior_model_")):
                continue
            settings = read_json(p / "run_settings.json", {})
            tb = str(
                settings.get("target_behavior")
                or settings.get("target_behavior_id")
                or ""
            ).strip()
            if tb and tb == bid:
                try:
                    shutil.rmtree(p)
                    removed.append(p.name)
                    logger.info("Removed orphaned model directory %s for deleted behaviour %s", p.name, bid)
                except OSError:
                    logger.warning("Failed to remove model directory %s", p, exc_info=True)
                    continue
                # Also drop the matching per-model evaluation output, which is
                # keyed by the model directory name and otherwise lingers in
                # analytics/evaluation views for the removed behaviour.
                eval_dir = self._project_root / "derived" / "evaluation" / "by_model" / p.name
                if eval_dir.exists():
                    try:
                        shutil.rmtree(eval_dir)
                    except OSError:
                        logger.warning("Failed to remove evaluation directory %s", eval_dir, exc_info=True)
        return removed

    @staticmethod
    def _strip_label(raw_label: str, dead_id: str) -> str | None:
        """Remove *dead_id* from a (possibly pipe-joined) behavior label.

        Returns the remaining label, or ``None`` if nothing survives (the row
        referenced only the deleted behaviour and should be dropped).
        """
        parts = [p.strip() for p in str(raw_label).split("|") if p.strip()]
        kept = [p for p in parts if p != dead_id]
        if not kept:
            return None
        return "|".join(kept)

    def _purge_behavior_references(self, behavior_id: str) -> dict[str, int]:
        """Cleanse a deleted behaviour from all review/candidate/label stores.

        A deleted behaviour otherwise lingers in the review-tab filter dropdown
        and in training data because candidate windows, review decisions, and
        reviewer labels still carry its id.  This mirrors
        :meth:`_purge_trained_models`: once a behaviour is gone from the
        definitions it must be gone everywhere.  Pipe-joined multi-labels have
        only the deleted constituent stripped; rows that referenced the deleted
        behaviour exclusively are removed.
        """
        counts = {"candidates": 0, "decisions": 0, "labels": 0}
        if self._project_root is None:
            return counts
        dead = str(behavior_id).strip()
        if not dead:
            return counts
        root = self._project_root

        # 1. Candidate queues + clip manifest (behavior_id field).
        for rel, key in (
            ("derived/review_tables/external_window_candidates.json", "candidates"),
            ("derived/review_tables/candidate_segments.json", "candidates"),
            ("derived/review_tables/candidate_windows.json", "candidates"),
            ("derived/review_tables/clip_manifest.json", "clips"),
        ):
            path = root / rel
            if not path.exists():
                continue
            raw = read_json(path, {})
            rows = raw.get(key)
            if not isinstance(rows, list):
                continue
            kept = []
            for row in rows:
                if not isinstance(row, dict):
                    kept.append(row)
                    continue
                new_label = self._strip_label(row.get("behavior_id", ""), dead)
                if new_label is None:
                    counts["candidates"] += 1
                    continue
                if new_label != str(row.get("behavior_id", "")):
                    row = {**row, "behavior_id": new_label}
                kept.append(row)
            if len(kept) != len(rows):
                write_json(path, {**raw, key: kept})

        # 2. Review decisions (behavior_label field).
        dec_path = root / "derived" / "review_tables" / "review_decisions.json"
        if dec_path.exists():
            raw = read_json(dec_path, {})
            rows = raw.get("decisions")
            if isinstance(rows, list):
                kept = []
                for row in rows:
                    if not isinstance(row, dict):
                        kept.append(row)
                        continue
                    new_label = self._strip_label(row.get("behavior_label", ""), dead)
                    if new_label is None:
                        counts["decisions"] += 1
                        continue
                    if new_label != str(row.get("behavior_label", "")):
                        row = {**row, "behavior_label": new_label}
                    kept.append(row)
                if len(kept) != len(rows):
                    write_json(dec_path, {**raw, "decisions": kept})

        # 3. Reviewer labels parquet (review_label column — training source).
        lbl_path = root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if lbl_path.exists():
            try:
                import pandas as pd

                df = pd.read_parquet(lbl_path)
                if "review_label" in df.columns and not df.empty:
                    original = df["review_label"]
                    stripped = original.map(
                        lambda v: self._strip_label(v, dead) if v is not None else v
                    )
                    drop_mask = stripped.isna() & original.notna()
                    counts["labels"] = int(drop_mask.sum())
                    changed = counts["labels"] > 0 or bool(
                        (stripped.notna() & (stripped != original)).any()
                    )
                    if changed:
                        df = df.assign(review_label=stripped)[~drop_mask].copy()
                        df.to_parquet(lbl_path, index=False)
            except Exception:
                logger.warning("Failed to purge reviewer labels for %s", dead, exc_info=True)

        if any(counts.values()):
            logger.info(
                "Purged deleted behaviour %s: %d candidate windows, %d decisions, %d reviewer labels",
                dead, counts["candidates"], counts["decisions"], counts["labels"],
            )
        return counts

    def get(self, behavior_id: str) -> BehaviorDefinition | None:
        return next((b for b in self._behaviors if b.behavior_id == behavior_id), None)

    def reorder(self, from_idx: int, to_idx: int) -> None:
        if 0 <= from_idx < len(self._behaviors) and 0 <= to_idx < len(self._behaviors):
            b = self._behaviors.pop(from_idx)
            self._behaviors.insert(to_idx, b)
            self.save()

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def apply_template(self, template_name: str, skip_existing: bool = True) -> int:
        """Append template behaviors (skip names already present). Returns count added."""
        items = _TEMPLATES.get(template_name, [])
        existing_names = {b.name.lower() for b in self._behaviors}
        added = 0
        for raw in items:
            if skip_existing and raw["name"].lower() in existing_names:
                continue
            self._behaviors.append(
                BehaviorDefinition.model_validate({**raw, "behavior_id": str(uuid.uuid4())})
            )
            added += 1
        if added:
            self.save()
        return added

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_definitions(self, path: Path) -> None:
        write_yaml(path, {"behaviors": [b.model_dump(mode="json") for b in self._behaviors]})

    def import_definitions(self, path: Path) -> int:
        raw = read_yaml(path, {})
        existing_ids = {b.behavior_id for b in self._behaviors}
        added = 0
        for item in raw.get("behaviors", []):
            try:
                b = BehaviorDefinition.model_validate(item)
                if b.behavior_id in existing_ids:
                    b = b.model_copy(update={"behavior_id": str(uuid.uuid4())})
                self._behaviors.append(b)
                added += 1
            except Exception:
                pass
        if added:
            self.save()
        return added
