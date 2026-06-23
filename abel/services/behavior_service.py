"""Behavior definition CRUD with project YAML persistence."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from abel.models.schemas import BehaviorDefinition
from abel.storage.file_store import read_yaml, write_yaml


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
            return True
        return False

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
