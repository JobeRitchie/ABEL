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

    def _structured_label_path(self) -> Path:
        assert self._project_root is not None
        return self._project_root / "derived" / "review_labels" / "soundboard_labels.json"

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

    def upsert_decisions_bulk(self, specs: list[dict]) -> list[ReviewDecision]:
        """Upsert many decisions in a single load/save.

        Each spec is a dict with the same keys as :meth:`upsert_decision`
        (``clip_id``, ``reviewer``, ``decision``, ``behavior_label``, ``notes``,
        ``confidence_override``, ``adjusted_start_frame``, ``adjusted_end_frame``).
        Used by bulk operations so assigning a behavior to hundreds of clips is
        one JSON write instead of one per clip.  Returns the upserted records.
        """
        if not self._project_root or not specs:
            return []

        status_by_decision = {
            ReviewDecisionType.REJECT: "rejected",
            ReviewDecisionType.AMBIGUOUS: "ambiguous",
            ReviewDecisionType.SKIP: "skipped",
            ReviewDecisionType.BOOKMARK: "bookmarked",
        }

        by_clip: dict[str, ReviewDecision] = {r.clip_id: r for r in self.load_decisions()}
        out: list[ReviewDecision] = []
        for spec in specs:
            clip_id = str(spec["clip_id"])
            decision = spec["decision"]
            existing = by_clip.get(clip_id)
            rec = ReviewDecision(
                decision_id=existing.decision_id if existing else uuid.uuid4().hex[:12],
                clip_id=clip_id,
                reviewer=str(spec.get("reviewer", "reviewer")),
                old_status=existing.new_status if existing else "unscored",
                new_status=status_by_decision.get(decision, "reviewed"),
                decision=decision,
                behavior_label=spec.get("behavior_label"),
                notes=str(spec.get("notes", "")),
                confidence_override=spec.get("confidence_override"),
                adjusted_start_frame=spec.get("adjusted_start_frame"),
                adjusted_end_frame=spec.get("adjusted_end_frame"),
            )
            by_clip[clip_id] = rec
            out.append(rec)

        self.save_decisions(list(by_clip.values()))
        return out

    def replace_segment_labels(self, records: list[ReviewerLabelRecord]) -> None:
        """Append reviewer labels, replacing any existing rows for the same
        ``segment_id`` — one parquet read/write for the whole batch.

        This keeps re-labelling idempotent (no duplicate/stale rows accumulate)
        and is far faster than calling :meth:`append_segment_label` per record.
        """
        if not self._project_root or not records:
            return
        path = self._segment_label_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        seg_ids = {str(r.segment_id) for r in records}
        new_df = pd.DataFrame([r.model_dump(mode="json") for r in records])
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                existing = existing[~existing["segment_id"].astype(str).isin(seg_ids)]
                merged = pd.concat([existing, new_df], ignore_index=True)
            except Exception:
                merged = new_df
        else:
            merged = new_df
        merged.to_parquet(path, index=False)

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
        # Drop any round-trip soundboard payloads for these windows.
        store = self._load_structured_store()
        if any(cid in store for cid in id_set):
            for cid in id_set:
                store.pop(cid, None)
            write_json(self._structured_label_path(), {"windows": store})
        return removed

    def remove_segment_labels(self, segment_ids: list[str]) -> int:
        """Delete reviewer-label rows for the given segment IDs (labels only).

        Unlike :meth:`delete_decisions`, this leaves the review decisions
        untouched — used when re-committing a clip's soundboard labels so the
        fresh set replaces the prior rows instead of duplicating them.
        Returns the number of label rows removed.
        """
        if not self._project_root:
            return 0
        path = self._segment_label_path()
        if not path.exists() or not segment_ids:
            return 0
        try:
            df = pd.read_parquet(path)
        except Exception:
            return 0
        id_set = set(segment_ids)
        before = len(df)
        df = df[~df["segment_id"].isin(id_set)]
        removed = before - len(df)
        if removed:
            df.to_parquet(path, index=False)
        return removed

    # --- Soundboard structured-label round-trip store ---------------------
    # The training-side reviewer_labels are lossy (pipe-joined, mutual doubled,
    # structured fields dropped on merge), so the *exact* soundboard payload is
    # kept separately keyed by window id, letting a revisited clip repopulate
    # its labels for editing.

    def _load_structured_store(self) -> dict:
        if not self._project_root:
            return {}
        raw = read_json(self._structured_label_path(), {"windows": {}})
        return dict(raw.get("windows", {}) or {})

    def get_structured_labels(self, window_id: str) -> list[dict]:
        """Return the exact soundboard label payload committed for ``window_id``."""
        if not window_id:
            return []
        return list(self._load_structured_store().get(str(window_id), []) or [])

    def save_structured_labels(self, window_id: str, labels: list[dict]) -> None:
        """Persist (or clear, if empty) the soundboard payload for ``window_id``."""
        if not self._project_root or not window_id:
            return
        store = self._load_structured_store()
        wid = str(window_id)
        if labels:
            store[wid] = [
                {
                    "behavior_id": str(lab.get("behavior_id") or ""),
                    "focal_animal_id": lab.get("focal_animal_id"),
                    "partner_animal_id": lab.get("partner_animal_id"),
                }
                for lab in labels
            ]
        else:
            store.pop(wid, None)
        write_json(self._structured_label_path(), {"windows": store})

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
