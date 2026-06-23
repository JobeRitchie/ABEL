"""Review and triage persistence service."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from abel.models.schemas import ReviewDecision, ReviewDecisionType, ReviewerLabelRecord
from abel.storage.file_store import read_json, write_json


@dataclass
class ReviewSummary:
    total: int = 0
    accepted: int = 0
    rejected: int = 0
    ambiguous: int = 0
    skipped: int = 0


@dataclass
class ReviewStore:
    decisions: list[ReviewDecision] = field(default_factory=list)


class ReviewService:
    """Persists reviewer decisions to project storage."""

    def __init__(self) -> None:
        self._project_root: Path | None = None

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root

    def _path(self) -> Path:
        assert self._project_root is not None
        return self._project_root / "derived" / "review_tables" / "review_decisions.json"

    def _segment_label_path(self) -> Path:
        assert self._project_root is not None
        return self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"

    def load_decisions(self) -> list[ReviewDecision]:
        if not self._project_root:
            return []
        raw = read_json(self._path(), {"decisions": []})
        rows: list[ReviewDecision] = []
        for item in raw.get("decisions", []):
            try:
                rows.append(ReviewDecision.model_validate(item))
            except Exception:
                pass
        return rows

    def save_decisions(self, decisions: list[ReviewDecision]) -> None:
        if not self._project_root:
            return
        write_json(self._path(), {"decisions": [d.model_dump(mode="json") for d in decisions]})

    def upsert_decision(
        self,
        clip_id: str,
        reviewer: str,
        decision: ReviewDecisionType,
        behavior_label: str | None = None,
        notes: str = "",
        confidence_override: float | None = None,
        adjusted_start_frame: int | None = None,
        adjusted_end_frame: int | None = None,
    ) -> ReviewDecision:
        rows = self.load_decisions()
        existing = next((r for r in rows if r.clip_id == clip_id), None)
        old_status = existing.new_status if existing else "unscored"

        new_status = "reviewed"
        if decision == ReviewDecisionType.REJECT:
            new_status = "rejected"
        elif decision == ReviewDecisionType.AMBIGUOUS:
            new_status = "ambiguous"
        elif decision == ReviewDecisionType.SKIP:
            new_status = "skipped"
        elif decision == ReviewDecisionType.BOOKMARK:
            new_status = "bookmarked"

        rec = ReviewDecision(
            decision_id=existing.decision_id if existing else uuid.uuid4().hex[:12],
            clip_id=clip_id,
            reviewer=reviewer,
            old_status=old_status,
            new_status=new_status,
            decision=decision,
            behavior_label=behavior_label,
            notes=notes,
            confidence_override=confidence_override,
            adjusted_start_frame=adjusted_start_frame,
            adjusted_end_frame=adjusted_end_frame,
        )

        rows = [r for r in rows if r.clip_id != clip_id]
        rows.append(rec)
        self.save_decisions(rows)
        return rec

    @staticmethod
    def summary(decisions: list[ReviewDecision]) -> ReviewSummary:
        out = ReviewSummary(total=len(decisions))
        for d in decisions:
            if d.decision == ReviewDecisionType.ACCEPT:
                out.accepted += 1
            elif d.decision == ReviewDecisionType.REJECT:
                out.rejected += 1
            elif d.decision == ReviewDecisionType.AMBIGUOUS:
                out.ambiguous += 1
            elif d.decision == ReviewDecisionType.SKIP:
                out.skipped += 1
        return out

    def load_segment_labels(self) -> list[ReviewerLabelRecord]:
        if not self._project_root:
            return []
        path = self._segment_label_path()
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        rows: list[ReviewerLabelRecord] = []
        for rec in df.to_dict(orient="records"):
            try:
                rows.append(ReviewerLabelRecord.model_validate(rec))
            except Exception:
                continue
        return rows

    def delete_decisions(self, clip_ids: list[str]) -> int:
        """Remove decisions and reviewer labels for the given clip IDs.

        Returns the number of decision records deleted.
        """
        if not self._project_root:
            return 0
        id_set = set(clip_ids)
        rows = self.load_decisions()
        kept = [r for r in rows if r.clip_id not in id_set]
        removed = len(rows) - len(kept)
        if removed:
            self.save_decisions(kept)
        # Also purge matching rows from the reviewer labels parquet.
        lbl_path = self._segment_label_path()
        if lbl_path.exists():
            try:
                df = pd.read_parquet(lbl_path)
                df = df[~df["segment_id"].isin(id_set)]
                df.to_parquet(lbl_path, index=False)
            except Exception:
                pass
        return removed

    def append_segment_label(self, record: ReviewerLabelRecord) -> None:
        if not self._project_root:
            return
        path = self._segment_label_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([record.model_dump(mode="json")])
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, row], ignore_index=True)
            except Exception:
                # File is corrupted or empty — start fresh
                merged = row
        else:
            merged = row
        merged.to_parquet(path, index=False)
