"""Detect and repair a repurposed ``No Behavior`` label.

``no_behavior`` is not an ordinary behavior id: it is the universal negative
class.  Training collapses every alternate label onto it, dense refinement skips
it in the behavior competition, and exports drop it.  Renaming the built-in
``No Behavior`` definition therefore does not create a new behavior, it aliases
a real behavior (say *Freezing*) onto the negative class, so the same token
means "this is freezing" in the review store and "this is nothing" in the
trainer.  Adding a second behavior *named* ``No Behavior`` compounds it, because
several code paths match the negative class on the normalized **name** as well
as the id.

:class:`~abel.services.behavior_service.BehaviorService` now refuses both edits,
but projects made before that guard already carry the conflict.  This module
detects it and repairs it by swapping the two identities:

* the repurposed behavior (holding ``no_behavior``) is moved onto a fresh UUID,
  taking all of its labels, decisions, candidates and settings with it;
* the user's own ``No Behavior`` behavior is promoted onto the reserved
  ``no_behavior`` id, so the negatives they recorded become the real negative
  class (when they never made one, a fresh built-in definition is created).

Hard negatives the application itself wrote (temporal-review false-positive
rejections) are recognized by their provenance and stay on ``no_behavior``
instead of being carried over to the repurposed behavior.  Rows that are
genuinely ambiguous, a click on the review tab's built-in "No Behavior" button
is stored exactly like a positive label for the repurposed behavior, are
counted and reported rather than guessed at silently.

Every model trained while the conflict existed was fitted against a contaminated
negative class, so the repair retires those artifacts (into the backup) and the
behaviors must be retrained.  Nothing is deleted outright: every file the repair
touches is copied into ``derived/backups/no_behavior_repair_<timestamp>/`` first.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from abel.models.schemas import BehaviorDefinition
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml


logger = logging.getLogger(__name__)

# Written by the temporal-review tab when the reviewer rejects a stretch of a
# behavior's trace; the parquet row carries the negative sentinel and the
# decision carries the concept that was wrong.  This pairing is what lets the
# repair tell a system-written hard negative from a user's positive label.
_TEMPORAL_FEEDBACK_REVIEWER = "temporal_feedback"

# Project stores that key rows by behavior id.  ``key`` is the list the rows
# live under; ``fields`` are the columns holding (possibly pipe-joined) ids.
_JSON_ROW_STORES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("derived/review_tables/external_window_candidates.json", "candidates", ("behavior_id",)),
    ("derived/review_tables/candidate_segments.json", "candidates", ("behavior_id",)),
    ("derived/review_tables/candidate_windows.json", "candidates", ("behavior_id",)),
    ("derived/review_tables/clip_manifest.json", "clips", ("behavior_id",)),
    ("config/seeds.json", "seeds", ("behavior_id",)),
)

# Settings files whose *keys* (and nested keys) are behavior ids.
_JSON_KEYED_SETTINGS: tuple[str, ...] = (
    "config/temporal_review_settings.json",
    "config/temporal_refinement_settings.json",
)


def safe_id_token(behavior_id: str) -> str:
    """Filesystem-safe form of a behavior id (matches the artifact writers)."""
    return "".join(
        ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(behavior_id).strip()
    )


@dataclass
class NoBehaviorConflict:
    """A project where the reserved negative identity is held by the wrong rows.

    Two independent halves, either of which is enough to corrupt training:
    ``repurposed_name``, the built-in negative was renamed into a real behavior,
    so the ``no_behavior`` **id** now means something positive; and
    ``replacement_id``, a second behavior is *named* "No Behavior", which the
    name-based negative checks read as the negative class.
    """

    repurposed_name: str | None = None
    """Name the built-in negative was renamed to (e.g. ``"Freezing"``)."""

    replacement_id: str | None = None
    """Id of the user-created behavior named "No Behavior", if they made one."""

    replacement_name: str | None = None

    def describe(self) -> str:
        lines: list[str] = []
        if self.repurposed_name:
            lines.append(
                f"The built-in “No Behavior” label was renamed to “{self.repurposed_name}”. "
                "“no_behavior” is the universal negative class every model is trained "
                f"against, so “{self.repurposed_name}” is currently being read as "
                "“nothing happened” by training, temporal refinement and export."
            )
            if self.replacement_id:
                lines.append(
                    f"A second behavior named “{self.replacement_name}” was added and holds "
                    "the labels that were meant to be the negatives."
                )
        elif self.replacement_id:
            lines.append(
                f"A second behavior named “{self.replacement_name}” duplicates the built-in "
                "negative label. Several code paths match the negative class by name, so the "
                "two are read as one and its labels land in the wrong class."
            )
        return " ".join(lines)


def detect_no_behavior_conflict(
    behaviors: Sequence[BehaviorDefinition],
) -> NoBehaviorConflict | None:
    """Return the conflict in *behaviors*, or ``None`` when the project is healthy."""
    from abel.services.behavior_service import NO_BEHAVIOR_ID, is_no_behavior_name  # noqa: PLC0415

    sentinel = next(
        (b for b in behaviors if str(b.behavior_id).strip() == NO_BEHAVIOR_ID), None
    )
    repurposed = sentinel is not None and not is_no_behavior_name(sentinel.name)

    replacement = next(
        (
            b
            for b in behaviors
            if str(b.behavior_id).strip() != NO_BEHAVIOR_ID and is_no_behavior_name(b.name)
        ),
        None,
    )
    if not repurposed and replacement is None:
        return None
    return NoBehaviorConflict(
        repurposed_name=str(sentinel.name) if repurposed else None,
        replacement_id=str(replacement.behavior_id) if replacement else None,
        replacement_name=str(replacement.name) if replacement else None,
    )


@dataclass
class RepairReport:
    """What the repair did (``dry_run=True``) or would do (``dry_run=False``)."""

    dry_run: bool
    repurposed_name: str | None = None
    new_behavior_id: str | None = None
    promoted_id: str | None = None
    promoted_name: str | None = None
    merged_duplicate: bool = False
    counts: dict[str, int] = field(default_factory=dict)
    ambiguous_label_rows: int = 0
    retired_models: list[str] = field(default_factory=list)
    retired_artifacts: list[str] = field(default_factory=list)
    backup_dir: str | None = None

    def summary_lines(self) -> list[str]:
        verb = "Would move" if self.dry_run else "Moved"
        lines: list[str] = []
        if self.repurposed_name and self.new_behavior_id:
            lines.append(
                f"{verb} “{self.repurposed_name}” onto its own behavior id "
                f"({self.new_behavior_id[:8]}…), freeing “no_behavior” for the negative class."
            )
        if self.merged_duplicate:
            lines.append(
                f"“{self.promoted_name}” is merged into the built-in negative label; "
                "its labels keep counting as negatives."
            )
        elif self.promoted_id:
            lines.append(
                f"“{self.promoted_name}” becomes the project's built-in negative label."
            )
        elif self.repurposed_name:
            lines.append("A fresh built-in “No Behavior” label is created.")
        for label, key in (
            ("reviewer label rows", "labels"),
            ("review decisions", "decisions"),
            ("candidate/clip rows", "candidates"),
            ("settings entries", "settings"),
            ("seed examples", "seeds"),
            ("soundboard label rows", "soundboard"),
        ):
            n = int(self.counts.get(key, 0))
            if n:
                lines.append(f"  • {n} {label} re-pointed")
        if self.ambiguous_label_rows:
            lines.append(
                f"  • {self.ambiguous_label_rows} label row(s) recorded through the "
                f"review tab's built-in “No Behavior” button are indistinguishable from "
                f"“{self.repurposed_name}” labels and are treated as "
                f"“{self.repurposed_name}”. Spot-check these clips after the repair."
            )
        if self.retired_models:
            lines.append(
                f"  • {len(self.retired_models)} trained model(s) retired to the backup, "
                "they were fitted against the mixed-up negative class and must be retrained."
            )
        if self.retired_artifacts:
            lines.append(
                f"  • {len(self.retired_artifacts)} derived artifact folder(s) retired "
                "(rebuilt on the next run)."
            )
        return lines


class NoBehaviorRepair:
    """Swap a repurposed ``No Behavior`` back into a well-formed identity."""

    def __init__(self, project_root: Path, behaviors: Sequence[BehaviorDefinition]) -> None:
        self._root = Path(project_root)
        self._behaviors = list(behaviors)
        self._conflict = detect_no_behavior_conflict(self._behaviors)

    @property
    def conflict(self) -> NoBehaviorConflict | None:
        return self._conflict

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def plan(self) -> RepairReport:
        """Count what a repair would change without writing anything."""
        return self._run(dry_run=True)

    def apply(self) -> RepairReport:
        """Perform the repair, backing up every file it touches first."""
        return self._run(dry_run=False)

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _run(self, dry_run: bool) -> RepairReport:
        from abel.services.behavior_service import NO_BEHAVIOR_ID  # noqa: PLC0415

        conflict = self._conflict
        if conflict is None:
            raise ValueError("This project's “No Behavior” label is already well-formed.")

        # Applied as one simultaneous swap so neither id clobbers the other. When
        # only a duplicate exists the built-in stays put and the duplicate merges
        # into it: its rows already meant "negative".
        new_id = str(uuid.uuid4()) if conflict.repurposed_name else None
        remap: dict[str, str] = {}
        if new_id:
            remap[NO_BEHAVIOR_ID] = new_id
        if conflict.replacement_id:
            remap[conflict.replacement_id] = NO_BEHAVIOR_ID

        backup_dir: Path | None = None
        if not dry_run:
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_dir = self._root / "derived" / "backups" / f"no_behavior_repair_{stamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)

        report = RepairReport(
            dry_run=dry_run,
            repurposed_name=conflict.repurposed_name,
            new_behavior_id=new_id,
            promoted_id=conflict.replacement_id,
            promoted_name=conflict.replacement_name,
            merged_duplicate=bool(conflict.replacement_id and not conflict.repurposed_name),
            backup_dir=str(backup_dir) if backup_dir else None,
        )

        keep_negative = self._system_negative_segment_ids()
        counts = report.counts

        counts["labels"], report.ambiguous_label_rows = self._remap_reviewer_labels(
            remap, keep_negative, dry_run, backup_dir
        )
        counts["decisions"] = self._remap_review_decisions(remap, dry_run, backup_dir)
        counts["candidates"], counts["seeds"] = self._remap_json_row_stores(
            remap, dry_run, backup_dir
        )
        counts["settings"] = self._remap_keyed_settings(remap, dry_run, backup_dir)
        counts["soundboard"] = self._remap_soundboard(remap, dry_run, backup_dir)

        report.retired_models = self._retire_models(remap, dry_run, backup_dir)
        report.retired_artifacts = self._retire_artifacts(remap, dry_run, backup_dir)

        if not dry_run:
            self._rewrite_definitions(remap, backup_dir)
            logger.info(
                "Repaired repurposed No Behavior label: %s → %s (%s)",
                conflict.repurposed_name, new_id, counts,
            )
        return report

    # --- provenance -----------------------------------------------------

    def _system_negative_segment_ids(self) -> set[str]:
        """Segments the app itself labeled as hard negatives.

        The temporal-review tab writes a rejection as ``reviewer='temporal_feedback'``
        with ``decision='reject'``; the matching parquet row carries the negative
        sentinel.  Those rows mean "nothing here", not the repurposed behavior, so
        they must stay on ``no_behavior`` through the swap.
        """
        path = self._root / "derived" / "review_tables" / "review_decisions.json"
        rows = read_json(path, {}).get("decisions")
        if not isinstance(rows, list):
            return set()
        out: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("reviewer") or "") != _TEMPORAL_FEEDBACK_REVIEWER:
                continue
            if str(row.get("decision") or "").lower() != "reject":
                continue
            clip_id = str(row.get("clip_id") or "").strip()
            if clip_id:
                out.add(clip_id)
        return out

    # --- generic helpers ------------------------------------------------

    @staticmethod
    def _remap_token(value: object, remap: dict[str, str]) -> str:
        """Rewrite a (possibly pipe-joined) behavior label through *remap*."""
        parts = [p.strip() for p in str(value or "").split("|")]
        return "|".join(remap.get(p, p) for p in parts)

    def _backup(self, path: Path, backup_dir: Path | None) -> None:
        if backup_dir is None or not path.exists():
            return
        rel = path.relative_to(self._root)
        dest = backup_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    # --- individual stores ----------------------------------------------

    def _remap_reviewer_labels(
        self,
        remap: dict[str, str],
        keep_negative: set[str],
        dry_run: bool,
        backup_dir: Path | None,
    ) -> tuple[int, int]:
        from abel.services.behavior_service import NO_BEHAVIOR_ID  # noqa: PLC0415

        path = self._root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if not path.exists():
            return 0, 0
        import pandas as pd  # noqa: PLC0415

        try:
            df = pd.read_parquet(path)
        except Exception:
            logger.warning("Could not read %s for No Behavior repair", path, exc_info=True)
            return 0, 0
        if df.empty or "review_label" not in df.columns:
            return 0, 0

        seg_ids = (
            df["segment_id"].astype(str)
            if "segment_id" in df.columns
            else pd.Series([""] * len(df), index=df.index)
        )
        reviewers = (
            df["reviewer_id"].astype(str)
            if "reviewer_id" in df.columns
            else pd.Series([""] * len(df), index=df.index)
        )
        original = df["review_label"].astype(str)

        # Rows the app wrote as hard negatives keep the sentinel.
        pinned = seg_ids.isin(keep_negative) & (reviewers == _TEMPORAL_FEEDBACK_REVIEWER)
        rewritten = original.map(lambda v: self._remap_token(v, remap))
        rewritten = rewritten.where(~pinned, original)

        changed = int((rewritten != original).sum())
        # Sentinel rows written by a human reviewer cannot be told apart from a
        # genuine label for the repurposed behavior; they follow the behavior.
        # Only meaningful when the sentinel id is actually being moved.
        ambiguous = (
            int(
                (
                    (original == NO_BEHAVIOR_ID)
                    & ~pinned
                    & (reviewers != _TEMPORAL_FEEDBACK_REVIEWER)
                ).sum()
            )
            if NO_BEHAVIOR_ID in remap
            else 0
        )

        if changed and not dry_run:
            self._backup(path, backup_dir)
            df = df.assign(review_label=rewritten)
            df.to_parquet(path, index=False)
        return changed, ambiguous

    def _remap_review_decisions(
        self, remap: dict[str, str], dry_run: bool, backup_dir: Path | None
    ) -> int:
        path = self._root / "derived" / "review_tables" / "review_decisions.json"
        if not path.exists():
            return 0
        raw = read_json(path, {})
        rows = raw.get("decisions")
        if not isinstance(rows, list):
            return 0
        changed = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            # A temporal-review rejection is a hard negative, not a label for the
            # repurposed behavior: leave it on the sentinel.
            if (
                str(row.get("reviewer") or "") == _TEMPORAL_FEEDBACK_REVIEWER
                and str(row.get("decision") or "").lower() == "reject"
            ):
                continue
            value = row.get("behavior_label")
            if not isinstance(value, str) or not value:
                continue
            new_value = self._remap_token(value, remap)
            if new_value != value:
                row["behavior_label"] = new_value
                changed += 1
        if changed and not dry_run:
            self._backup(path, backup_dir)
            write_json(path, {**raw, "decisions": rows})
        return changed

    def _remap_json_row_stores(
        self, remap: dict[str, str], dry_run: bool, backup_dir: Path | None
    ) -> tuple[int, int]:
        """Returns (candidate/clip rows changed, seed rows changed)."""
        totals = {"candidates": 0, "seeds": 0}
        for rel, key, fields in _JSON_ROW_STORES:
            path = self._root / rel
            if not path.exists():
                continue
            raw = read_json(path, {})
            rows = raw.get(key)
            if not isinstance(rows, list):
                continue
            changed = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for name in fields:
                    value = row.get(name)
                    if not isinstance(value, str) or not value:
                        continue
                    new_value = self._remap_token(value, remap)
                    if new_value != value:
                        row[name] = new_value
                        changed += 1
            if changed:
                totals["seeds" if key == "seeds" else "candidates"] += changed
                if not dry_run:
                    self._backup(path, backup_dir)
                    write_json(path, {**raw, key: rows})
        return totals["candidates"], totals["seeds"]

    def _remap_keyed_settings(
        self, remap: dict[str, str], dry_run: bool, backup_dir: Path | None
    ) -> int:
        total = 0
        for rel in _JSON_KEYED_SETTINGS:
            path = self._root / rel
            if not path.exists():
                continue
            raw = read_json(path, {})
            if not isinstance(raw, dict):
                continue
            new_raw, changed = self._remap_keys_deep(raw, remap)
            if changed:
                total += changed
                if not dry_run:
                    self._backup(path, backup_dir)
                    write_json(path, new_raw)
        return total

    @classmethod
    def _remap_keys_deep(cls, value: Any, remap: dict[str, str]) -> tuple[Any, int]:
        """Rewrite dict keys that are behavior ids, at any nesting depth.

        Per-behavior settings are stored as ``{behavior_id: {...}}`` and the
        suppression matrix nests a second level of ids, so the rewrite has to walk
        the whole structure rather than only the top level.
        """
        if isinstance(value, dict):
            out: dict[Any, Any] = {}
            changed = 0
            for key, sub in value.items():
                new_key = remap.get(key, key) if isinstance(key, str) else key
                if new_key != key:
                    changed += 1
                new_sub, sub_changed = cls._remap_keys_deep(sub, remap)
                changed += sub_changed
                out[new_key] = new_sub
            return out, changed
        if isinstance(value, list):
            out_list = []
            changed = 0
            for item in value:
                new_item, item_changed = cls._remap_keys_deep(item, remap)
                changed += item_changed
                out_list.append(new_item)
            return out_list, changed
        return value, 0

    def _remap_soundboard(
        self, remap: dict[str, str], dry_run: bool, backup_dir: Path | None
    ) -> int:
        path = self._root / "derived" / "review_labels" / "soundboard_labels.json"
        if not path.exists():
            return 0
        raw = read_json(path, {"windows": {}})
        windows = raw.get("windows")
        if not isinstance(windows, dict):
            return 0
        changed = 0
        for _wid, payload in windows.items():
            if not isinstance(payload, list):
                continue
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                value = entry.get("behavior_id")
                if not isinstance(value, str) or not value:
                    continue
                new_value = self._remap_token(value, remap)
                if new_value != value:
                    entry["behavior_id"] = new_value
                    changed += 1
        if changed and not dry_run:
            self._backup(path, backup_dir)
            write_json(path, {**raw, "windows": windows})
        return changed

    # --- derived artifacts ----------------------------------------------

    def _retire_models(
        self, remap: dict[str, str], dry_run: bool, backup_dir: Path | None
    ) -> list[str]:
        """Move models trained on either affected id into the backup.

        A model whose target was the repurposed behavior was trained down the
        ``no_behavior`` branch of the trainer, its positives are the negatives of
        every other behavior.  Remapping its id would keep a wrong model wearing a
        right name, so it is retired and must be retrained.
        """
        models_root = self._root / "derived" / "models"
        if not models_root.exists():
            return []
        retired: list[str] = []
        for path in sorted(models_root.iterdir()):
            if not (path.is_dir() and path.name.startswith("behavior_model_")):
                continue
            settings = read_json(path / "run_settings.json", {})
            target = str(
                settings.get("target_behavior") or settings.get("target_behavior_id") or ""
            ).strip()
            if target not in remap:
                continue
            retired.append(path.name)
            if dry_run:
                continue
            self._move_to_backup(path, backup_dir)
            eval_dir = self._root / "derived" / "evaluation" / "by_model" / path.name
            if eval_dir.exists():
                self._move_to_backup(eval_dir, backup_dir)
        return retired

    def _retire_artifacts(
        self, remap: dict[str, str], dry_run: bool, backup_dir: Path | None
    ) -> list[str]:
        """Retire per-behavior derived folders keyed by the old ids."""
        retired: list[str] = []
        tokens = {safe_id_token(bid) for bid in remap}
        for parent in (
            self._root / "derived" / "temporal_refinement",
            self._root / "derived" / "behavior_bouts",
        ):
            if not parent.exists():
                continue
            for path in sorted(parent.iterdir()):
                if not path.is_dir() or path.name not in tokens:
                    continue
                retired.append(str(path.relative_to(self._root)))
                if not dry_run:
                    self._move_to_backup(path, backup_dir)
        return retired

    def _move_to_backup(self, path: Path, backup_dir: Path | None) -> None:
        if backup_dir is None:
            return
        dest = backup_dir / path.relative_to(self._root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(path), str(dest))
        except OSError:
            logger.warning("Could not retire %s during No Behavior repair", path, exc_info=True)

    # --- definitions -----------------------------------------------------

    def _rewrite_definitions(self, remap: dict[str, str], backup_dir: Path | None) -> None:
        from abel.services.behavior_service import (  # noqa: PLC0415
            NO_BEHAVIOR_ID,
            NO_BEHAVIOR_NAME,
            NO_BEHAVIOR_SHORT_NAME,
            BehaviorService,
        )

        path = self._root / "config" / "behavior_definitions.yaml"
        self._backup(path, backup_dir)
        raw = read_yaml(path, {})
        items = list(raw.get("behaviors", []) or [])

        out: list[dict] = []
        seen_ids: set[str] = set()
        promoted = False
        for item in items:
            if not isinstance(item, dict):
                continue
            bid = str(item.get("behavior_id") or "").strip()
            new_item = dict(item)
            if bid in remap:
                new_item["behavior_id"] = remap[bid]
            new_id = str(new_item.get("behavior_id") or "").strip()
            if new_id == NO_BEHAVIOR_ID:
                if promoted:
                    # A duplicate "No Behavior" merged onto the built-in id: its
                    # rows already moved across, so drop the redundant definition.
                    continue
                # Whatever now holds the reserved id is the built-in negative:
                # pin its name so name-based negative detection stays exact.
                new_item["name"] = NO_BEHAVIOR_NAME
                new_item["short_name"] = NO_BEHAVIOR_SHORT_NAME
                promoted = True
            elif new_id in seen_ids:
                continue
            seen_ids.add(new_id)
            out.append(new_item)

        if not promoted:
            out.append(BehaviorService.system_no_behavior_definition().model_dump(mode="json"))
        write_yaml(path, {**raw, "behaviors": out})


def repair_project(project_root: Path, behaviors: Iterable[BehaviorDefinition]) -> RepairReport:
    """Convenience wrapper: repair *project_root* in place."""
    return NoBehaviorRepair(project_root, list(behaviors)).apply()
