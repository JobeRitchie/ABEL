"""Detect and repair false-positive rejections stored as universal negatives.

A false positive means "this clip is not behavior X".  Earlier versions saved
every rejection (Review tab *Reject*, Temporal Review false-positive intervals)
as ``no_behavior`` in ``reviewer_labels.parquet``, the table training reads.
``no_behavior`` is the universal negative, so each rejected clip became a
negative for **every** behavior model.  When the rejected clip was really a
different behavior (e.g. a protected stretch attend rejected from the
unprotected queue), that behavior's model was taught its own positives are
"nothing".

Rejections are now stored as ``not_<behavior_id>``: a negative for that
behavior's model only, ignored by the others.  This module finds the old rows
and relabels them in the reviewer labels and the accumulated training set.
False negatives were always stored as positives of the named behavior, so they
need no repair.

Provenance comes from the stores themselves:

* Review-tab rejections: the latest decision in ``review_decisions.json`` is
  ``reject`` with a real behavior id in ``behavior_label``.
* Temporal-review tiles injected without a decision carry
  ``notes='auto:no_behavior:<behavior_id>'``.

A rejection made with *No Behavior* selected stays a universal negative.
Every file touched is copied into ``derived/backups/fp_label_repair_<timestamp>/``
first.  Models trained before the repair still carry the old negatives and
should be retrained.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from abel.services.review_service import NO_BEHAVIOR_ID, reject_review_label
from abel.storage.file_store import atomic_write_parquet, read_json

logger = logging.getLogger(__name__)

_AUTO_FP_NOTE_PREFIX = f"auto:{NO_BEHAVIOR_ID}:"


@dataclass
class FalsePositiveRepairReport:
    """What the repair found (``plan``) or changed (``apply``)."""

    # segment_id -> corrected label (``not_<behavior_id>``)
    relabel: dict[str, str] = field(default_factory=dict)
    reviewer_label_rows: int = 0
    training_set_rows: int = 0
    per_behavior: dict[str, int] = field(default_factory=dict)
    backup_dir: Path | None = None
    applied: bool = False

    @property
    def needed(self) -> bool:
        return self.reviewer_label_rows > 0 or self.training_set_rows > 0


class FalsePositiveLabelRepair:
    """Relabel false-positive rejections from ``no_behavior`` to ``not_<behavior>``."""

    def __init__(self, project_root: Path, behavior_ids: Iterable[str]) -> None:
        self._root = Path(project_root)
        self._known = {str(b).strip() for b in behavior_ids if str(b).strip() and str(b) != NO_BEHAVIOR_ID}

    @property
    def _decisions_path(self) -> Path:
        return self._root / "derived" / "review_tables" / "review_decisions.json"

    @property
    def _labels_path(self) -> Path:
        return self._root / "derived" / "review_labels" / "reviewer_labels.parquet"

    @property
    def _training_path(self) -> Path:
        return self._root / "derived" / "training_sets" / "training_set.parquet"

    def plan(self) -> FalsePositiveRepairReport:
        return self._run(apply=False)

    def apply(self) -> FalsePositiveRepairReport:
        return self._run(apply=True)

    def _corrections(self, labels: pd.DataFrame | None) -> dict[str, str]:
        out: dict[str, str] = {}
        rows = read_json(self._decisions_path, {}).get("decisions")
        if isinstance(rows, list):
            latest: dict[str, dict] = {}
            for row in rows:
                if isinstance(row, dict) and row.get("clip_id"):
                    latest[str(row["clip_id"])] = row
            for clip_id, row in latest.items():
                if str(row.get("decision") or "").lower() != "reject":
                    continue
                bid = str(row.get("behavior_label") or "").strip()
                if bid in self._known:
                    out[clip_id] = reject_review_label(bid)
        if labels is not None and "notes" in labels.columns:
            notes = labels["notes"].astype(str)
            auto = notes.str.startswith(_AUTO_FP_NOTE_PREFIX) & (
                labels["review_label"].astype(str) == NO_BEHAVIOR_ID
            )
            for sid, note in zip(labels.loc[auto, "segment_id"].astype(str), notes[auto]):
                bid = note[len(_AUTO_FP_NOTE_PREFIX):].strip()
                if bid in self._known:
                    out.setdefault(sid, reject_review_label(bid))
        return out

    def _run(self, apply: bool) -> FalsePositiveRepairReport:
        rep = FalsePositiveRepairReport()
        labels = pd.read_parquet(self._labels_path) if self._labels_path.exists() else None
        fixes = self._corrections(labels)
        if not fixes:
            return rep

        lbl_mask = None
        if labels is not None and not labels.empty:
            seg = labels["segment_id"].astype(str)
            lbl_mask = seg.isin(fixes) & (labels["review_label"].astype(str) == NO_BEHAVIOR_ID)
            rep.reviewer_label_rows = int(lbl_mask.sum())

        ts = None
        ts_mask = None
        if self._training_path.exists():
            ts = pd.read_parquet(self._training_path)
            if {"segment_id", "label"} <= set(ts.columns):
                ts_mask = ts["segment_id"].astype(str).isin(fixes) & (ts["label"].astype(str) == NO_BEHAVIOR_ID)
                rep.training_set_rows = int(ts_mask.sum())

        hit = set()
        if lbl_mask is not None:
            hit |= set(labels.loc[lbl_mask, "segment_id"].astype(str))
        if ts_mask is not None:
            hit |= set(ts.loc[ts_mask, "segment_id"].astype(str))
        rep.relabel = {s: fixes[s] for s in hit}
        for new in rep.relabel.values():
            bid = new[len("not_"):]
            rep.per_behavior[bid] = rep.per_behavior.get(bid, 0) + 1

        if not apply or not rep.needed:
            return rep

        rep.backup_dir = self._root / "derived" / "backups" / f"fp_label_repair_{datetime.now():%Y%m%d_%H%M%S}"
        rep.backup_dir.mkdir(parents=True, exist_ok=True)
        if lbl_mask is not None and rep.reviewer_label_rows:
            shutil.copy2(self._labels_path, rep.backup_dir / self._labels_path.name)
            labels.loc[lbl_mask, "review_label"] = labels.loc[lbl_mask, "segment_id"].astype(str).map(fixes)
            atomic_write_parquet(labels, self._labels_path, index=False)
        if ts_mask is not None and rep.training_set_rows:
            shutil.copy2(self._training_path, rep.backup_dir / self._training_path.name)
            ts.loc[ts_mask, "label"] = ts.loc[ts_mask, "segment_id"].astype(str).map(fixes)
            atomic_write_parquet(ts, self._training_path, index=False)
        rep.applied = True
        logger.info(
            "False-positive label repair: %d reviewer label(s), %d training row(s) relabelled; backup in %s",
            rep.reviewer_label_rows, rep.training_set_rows, rep.backup_dir,
        )
        return rep
