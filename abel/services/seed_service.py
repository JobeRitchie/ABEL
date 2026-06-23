"""Seed example CRUD with project JSON persistence."""

from __future__ import annotations

import uuid
from pathlib import Path

from abel.models.schemas import SeedExample
from abel.storage.file_store import read_json, write_json


class SeedService:
    """Stores and retrieves seed behavior examples for a project."""

    def __init__(self) -> None:
        self._seeds: list[SeedExample] = []
        self._assume_negatives: dict[str, bool] = {}
        self._project_root: Path | None = None

    # ------------------------------------------------------------------
    # Project binding
    # ------------------------------------------------------------------

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _path(self) -> Path:
        assert self._project_root is not None
        return self._project_root / "config" / "seeds.json"

    def _load(self) -> None:
        raw = read_json(self._path(), {"seeds": []})
        self._seeds = []
        for item in raw.get("seeds", []):
            try:
                self._seeds.append(SeedExample.model_validate(item))
            except Exception:
                pass
        self._assume_negatives = raw.get("assume_negatives", {})

    def save(self) -> None:
        if not self._project_root:
            return
        write_json(self._path(), {
            "seeds": [s.model_dump(mode="json") for s in self._seeds],
            "assume_negatives": self._assume_negatives,
        })

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @property
    def seeds(self) -> list[SeedExample]:
        return list(self._seeds)

    def seeds_for_behavior(self, behavior_id: str) -> list[SeedExample]:
        return [s for s in self._seeds if s.behavior_id == behavior_id]

    def count_by_behavior(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self._seeds:
            counts[s.behavior_id] = counts.get(s.behavior_id, 0) + 1
        return counts

    def add(self, seed: SeedExample) -> SeedExample:
        if not seed.seed_id:
            seed = seed.model_copy(update={"seed_id": str(uuid.uuid4())})
        self._seeds.append(seed)
        self.save()
        return seed

    def update(self, seed_id: str, updated: SeedExample) -> bool:
        for i, s in enumerate(self._seeds):
            if s.seed_id == seed_id:
                self._seeds[i] = updated.model_copy(update={"seed_id": seed_id})
                self.save()
                return True
        return False

    def delete(self, seed_id: str) -> bool:
        before = len(self._seeds)
        self._seeds = [s for s in self._seeds if s.seed_id != seed_id]
        if len(self._seeds) < before:
            self.save()
            return True
        return False

    def get(self, seed_id: str) -> SeedExample | None:
        return next((s for s in self._seeds if s.seed_id == seed_id), None)

    def copy_to_sessions(
        self, seed_id: str, target_session_ids: list[str]
    ) -> list[SeedExample]:
        """Duplicate a seed to one or more other sessions/subjects.

        Each target receives a fresh ``seed_id`` with identical behavior, frame
        range, label, quality flag, and notes.  Useful when a behavior occurs at
        a known frame range across multiple subjects' videos.

        Targets equal to the source session, or that already contain an
        identical seed (same behavior + frame range + label), are skipped so
        repeated copies don't create duplicates.  Returns the newly created seeds.
        """
        source = self.get(seed_id)
        if source is None:
            return []
        created: list[SeedExample] = []
        for sid in dict.fromkeys(str(t) for t in target_session_ids):  # dedupe, keep order
            if sid == source.session_id:
                continue
            already = any(
                s.session_id == sid
                and s.behavior_id == source.behavior_id
                and s.start_frame == source.start_frame
                and s.end_frame == source.end_frame
                and s.label_type == source.label_type
                for s in self._seeds
            )
            if already:
                continue
            new_seed = source.model_copy(update={
                "seed_id": str(uuid.uuid4()),
                "session_id": sid,
            })
            self._seeds.append(new_seed)
            created.append(new_seed)
        if created:
            self.save()
        return created

    # ------------------------------------------------------------------
    # Assume-negatives flag
    # ------------------------------------------------------------------

    def get_assume_negative(self, behavior_id: str) -> bool:
        return self._assume_negatives.get(behavior_id, False)

    def set_assume_negative(self, behavior_id: str, value: bool) -> None:
        self._assume_negatives[behavior_id] = value
        self.save()
