"""Behavior definition CRUD with project YAML persistence."""

from __future__ import annotations

import logging
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from abel.core.constants import GLOBAL_CONFIG_DIR
from abel.models.schemas import BehaviorDefinition
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml


logger = logging.getLogger(__name__)

NO_BEHAVIOR_ID = "no_behavior"

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def _looks_like_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(str(value).strip()))


def behavior_label(behavior_id: str | None, name_map: dict[str, str] | None = None) -> str:
    """Resolve a behavior ID to text safe to show the user.

    Behavior IDs are UUIDs and must never surface in the interface. Callers that
    already hold a ``{behavior_id: name}`` map pass it as *name_map*; anything the
    map cannot resolve degrades to a labelled short prefix instead of the full
    opaque token, so distinct unknowns stay distinguishable in a plot or table.
    """
    bid = str(behavior_id or "").strip()
    if not bid:
        return "—"
    if name_map:
        name = str(name_map.get(bid, "") or "").strip()
        if name:
            return name
    if _looks_like_uuid(bid):
        return f"Unknown behavior ({bid[:8]})"
    return bid

# ---------------------------------------------------------------------------
# Behavior presets
# ---------------------------------------------------------------------------
# Presets are named starting sets of behavior definitions. The built-in ones are
# composed from a shared library so that a behavior common to several assays
# (rear/groom/walk) carries the same operational definition everywhere — labels
# stay comparable when a lab runs more than one assay. User-defined presets are
# saved to GLOBAL_CONFIG_DIR so they are available in every project.

USER_PRESETS_PATH = GLOBAL_CONFIG_DIR / "behavior_presets.yaml"

_BEHAVIOR_LIBRARY: dict[str, dict] = {
    "rear": {
        "name": "Rear", "short_name": "rear",
        "description": "Vertical exploratory posture.",
        "operational_definition": (
            "Animal raises its forebody so both forelimbs leave the substrate and the "
            "trunk approaches vertical, pivoting over the hindlimbs."
        ),
        "inclusion_criteria": "Both forelimbs off the floor; trunk elevated toward vertical; wall-supported rears included.",
        "exclusion_criteria": "Brief stumbles or head-lifts with forepaws still down; grooming in an upright crouch.",
        "min_duration_sec": 0.2, "color": "#7ED321", "keyboard_shortcut": "r", "review_priority": 2,
    },
    "groom": {
        "name": "Groom", "short_name": "groom",
        "description": "Self-directed grooming.",
        "operational_definition": (
            "Repetitive forepaw strokes directed at the face, head, or body, or licking/nibbling "
            "of the flank, tail, or genitals, typically from a stationary crouched posture."
        ),
        "inclusion_criteria": "Rhythmic forepaw-to-face strokes or sustained licking of own fur; body largely stationary.",
        "exclusion_criteria": "Single paw shake or wipe; hindlimb scratching; wet-dog shakes; grooming-like motion while walking.",
        "min_duration_sec": 0.5, "color": "#4A90E2", "keyboard_shortcut": "g", "review_priority": 1,
    },
    "walk": {
        "name": "Walk", "short_name": "walk",
        "description": "Ambulation across the arena.",
        "operational_definition": (
            "Coordinated stepping that translates the body forward, with the centroid moving "
            "continuously in a consistent heading."
        ),
        "inclusion_criteria": "Sustained forward translation of the centroid with stepping gait.",
        "exclusion_criteria": "In-place pivoting or head scanning; postural shifts; rearing; being displaced without stepping.",
        "min_duration_sec": 0.5, "color": "#F5A623", "keyboard_shortcut": "w", "review_priority": 3,
    },
    "head_dip": {
        "name": "Head Dip", "short_name": "hdip",
        "description": "Head dip over an arm edge (elevated plus maze).",
        "operational_definition": (
            "From an arm of the maze, the animal extends its head over the platform edge and "
            "lowers the nose below platform level, scanning the space beneath."
        ),
        "inclusion_criteria": "Nose crosses the platform edge and drops below platform level; shoulders remain on the arm.",
        "exclusion_criteria": "Nose past the edge without dropping below platform level; head lowered on the platform surface; slips or falls.",
        "min_duration_sec": 0.2, "color": "#50E3C2", "keyboard_shortcut": "h", "review_priority": 1,
    },
    "protected_head_dip": {
        "name": "Protected Head Dip", "short_name": "phdip",
        "description": "Head dip performed from the centre or a closed arm.",
        "operational_definition": (
            "A head dip (nose over the edge and below platform level) performed while the body "
            "remains in the centre square or a closed arm, i.e. from cover."
        ),
        "inclusion_criteria": "Head-dip criteria met with the trunk in the centre platform or a closed arm.",
        "exclusion_criteria": "Dips with the trunk on an open arm (those are unprotected head dips).",
        "min_duration_sec": 0.2, "color": "#00838F", "keyboard_shortcut": "p", "review_priority": 1,
    },
    "stretch_attend": {
        "name": "Stretch Attend", "short_name": "sap",
        "description": "Stretch-attend posture — risk assessment.",
        "operational_definition": (
            "The animal elongates its body forward with the hindpaws planted, keeping the trunk "
            "low, then holds or withdraws without ambulating forward."
        ),
        "inclusion_criteria": "Body markedly elongated and flattened; forward head extension with stationary hindquarters.",
        "exclusion_criteria": "Stretching that continues into ambulation; ordinary sniffing without body elongation; grooming stretches.",
        "min_duration_sec": 0.3, "color": "#BD10E0", "keyboard_shortcut": "s", "review_priority": 1,
    },
    "freeze": {
        "name": "Freeze", "short_name": "freeze",
        "description": "Fear-related immobility.",
        "operational_definition": (
            "Complete absence of movement apart from respiration, held in a tense posture; "
            "all tracked keypoints stay below the movement threshold."
        ),
        "inclusion_criteria": "No locomotion, head, or limb movement for at least the minimum duration; respiration only.",
        "exclusion_criteria": "Brief pauses between bouts of locomotion; immobility while grooming, eating, or sleeping.",
        "min_duration_sec": 1.0, "color": "#9013FE", "keyboard_shortcut": "z", "review_priority": 1,
    },
    "dart": {
        "name": "Dart", "short_name": "dart",
        "description": "Darting — brief high-velocity forward burst.",
        "operational_definition": (
            "A short, abrupt forward acceleration well above the animal's ordinary locomotor "
            "speed, beginning and ending within roughly a second."
        ),
        "inclusion_criteria": "Sharp velocity spike from a slow or immobile state; brief, ballistic forward run.",
        "exclusion_criteria": "Steady ambulation at normal speed; jumps and escape climbing; startle without translation.",
        "min_duration_sec": 0.1, "color": "#D0021B", "keyboard_shortcut": "d", "review_priority": 1,
    },
    "approach": {
        "name": "Approach", "short_name": "appr",
        "description": "Directed approach to the food source.",
        "operational_definition": (
            "Oriented locomotion that reduces the distance to the food/novel stimulus, ending "
            "with the nose near it, without consumption beginning."
        ),
        "inclusion_criteria": "Head oriented toward the food source while distance decreases; terminates in proximity or investigation.",
        "exclusion_criteria": "Passing by without orientation; contact that begins feeding (score as Eat); retreat from the source.",
        "min_duration_sec": 0.3, "color": "#F8B500", "keyboard_shortcut": "a", "review_priority": 1,
    },
    "eat": {
        "name": "Eat", "short_name": "eat",
        "description": "Feeding on the pellet.",
        "operational_definition": (
            "The animal holds or contacts the food with mouth and/or forepaws and makes sustained "
            "biting/chewing movements."
        ),
        "inclusion_criteria": "Mouth on food with visible chewing or gnawing; pellet held in forepaws.",
        "exclusion_criteria": "Sniffing or touching the pellet without biting; carrying without chewing; grooming beside the food.",
        "min_duration_sec": 0.5, "color": "#C0392B", "keyboard_shortcut": "e", "review_priority": 1,
    },
}

_BUILTIN_PRESETS: dict[str, list[str]] = {
    "Standard Rodent": ["rear", "groom", "walk"],
    "Elevated Plus Maze": [
        "rear", "groom", "walk", "head_dip", "protected_head_dip", "stretch_attend",
    ],
    "Fear Conditioning": ["rear", "groom", "walk", "freeze", "dart"],
    "Novelty Suppressed Feeding": ["rear", "groom", "walk", "approach", "eat"],
}


def builtin_preset_definitions(preset_name: str) -> list[dict]:
    """Return the behavior definition dicts for a built-in preset."""
    keys = _BUILTIN_PRESETS.get(preset_name, [])
    return [dict(_BEHAVIOR_LIBRARY[k]) for k in keys]


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

    def display_name(self, behavior_id: str | None, *, short: bool = False) -> str:
        """Return the human-readable name for ``behavior_id``.

        Behavior IDs are UUIDs, which must never reach the user. Anything the
        user reads — status text, run logs, dialogs, plot labels — goes through
        here. Unresolvable UUIDs degrade to a labelled short prefix rather than
        the full opaque token, so two unknowns stay distinguishable.
        """
        bid = str(behavior_id or "").strip()
        b = self.get(bid) if bid else None
        if b is not None:
            return str(b.short_name or b.name) if short else str(b.name)
        return behavior_label(bid)

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
    # Presets
    # ------------------------------------------------------------------

    @staticmethod
    def is_builtin_preset(preset_name: str) -> bool:
        return str(preset_name) in _BUILTIN_PRESETS

    @staticmethod
    def _load_user_presets() -> dict[str, list[dict]]:
        raw = read_yaml(USER_PRESETS_PATH, {})
        presets = raw.get("presets") or {}
        if not isinstance(presets, dict):
            return {}
        out: dict[str, list[dict]] = {}
        for name, items in presets.items():
            if isinstance(items, list):
                out[str(name)] = [i for i in items if isinstance(i, dict)]
        return out

    @staticmethod
    def _write_user_presets(presets: dict[str, list[dict]]) -> None:
        write_yaml(USER_PRESETS_PATH, {"presets": presets})

    def preset_names(self) -> list[str]:
        """Built-in preset names first, then the user's saved presets."""
        return list(_BUILTIN_PRESETS.keys()) + sorted(
            self._load_user_presets().keys(), key=str.lower
        )

    def preset_definitions(self, preset_name: str) -> list[dict]:
        """Behavior definition dicts for a built-in or user preset."""
        if self.is_builtin_preset(preset_name):
            return builtin_preset_definitions(preset_name)
        return [dict(item) for item in self._load_user_presets().get(str(preset_name), [])]

    def apply_preset(self, preset_name: str, skip_existing: bool = True) -> int:
        """Append a preset's behaviors to the project. Returns the count added.

        Behaviors whose name already exists are skipped, so applying a second
        preset that shares rear/groom/walk does not duplicate them. A keyboard
        shortcut already claimed by an existing behavior is dropped rather than
        duplicated — two behaviors on one key make the soundboard ambiguous.
        """
        items = self.preset_definitions(preset_name)
        existing_names = {b.name.strip().lower() for b in self._behaviors}
        taken_keys = {
            str(b.keyboard_shortcut).lower() for b in self._behaviors if b.keyboard_shortcut
        }
        added = 0
        for raw in items:
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            if skip_existing and name.lower() in existing_names:
                continue
            fields = {k: v for k, v in raw.items() if k not in ("behavior_id", "version_history")}
            key = str(fields.get("keyboard_shortcut") or "").lower()
            if key and key in taken_keys:
                fields["keyboard_shortcut"] = None
            try:
                behavior = BehaviorDefinition.model_validate(
                    {**fields, "behavior_id": str(uuid.uuid4())}
                )
            except Exception:
                logger.warning("Skipped invalid behavior in preset %s: %s", preset_name, name)
                continue
            self._behaviors.append(behavior)
            existing_names.add(name.lower())
            if behavior.keyboard_shortcut:
                taken_keys.add(str(behavior.keyboard_shortcut).lower())
            added += 1
        if added:
            self.save()
        return added

    def save_preset(self, preset_name: str, overwrite: bool = False) -> int:
        """Save the project's current behaviors as a named user preset.

        The system ``No Behavior`` label is excluded (every project gets its own),
        as are ids and version history, so the preset carries only the reusable
        definition. Returns the number of behaviors stored.
        """
        name = str(preset_name).strip()
        if not name:
            raise ValueError("Preset name is required.")
        if self.is_builtin_preset(name):
            raise ValueError(f"'{name}' is a built-in preset and cannot be overwritten.")
        presets = self._load_user_presets()
        if name in presets and not overwrite:
            raise ValueError(f"A preset named '{name}' already exists.")

        items: list[dict] = []
        for b in self._behaviors:
            if str(b.behavior_id).strip() == NO_BEHAVIOR_ID:
                continue
            data = b.model_dump(mode="json")
            data.pop("behavior_id", None)
            data.pop("version_history", None)
            items.append(data)
        if not items:
            raise ValueError("There are no behaviors to save as a preset.")

        presets[name] = items
        self._write_user_presets(presets)
        logger.info("Saved behavior preset '%s' with %d behavior(s)", name, len(items))
        return len(items)

    def delete_preset(self, preset_name: str) -> bool:
        """Delete a user preset. Built-in presets cannot be deleted."""
        name = str(preset_name).strip()
        if self.is_builtin_preset(name):
            return False
        presets = self._load_user_presets()
        if name not in presets:
            return False
        presets.pop(name)
        self._write_user_presets(presets)
        logger.info("Deleted behavior preset '%s'", name)
        return True

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
