"""Export reviewed labels for downstream analysis and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
from typing import Any, Callable

import numpy as np
import pandas as pd

from abel.models.schemas import CandidateWindow, ReviewDecision, ReviewDecisionType
from abel.services.behavior_service import BehaviorService
from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseProcessingService
from abel.temporal_refinement.bout_postprocess import smooth_probabilities


@dataclass
class _SessionRenderTask:
    """All data needed by a worker thread to render one session's labeled video."""

    session_id: str
    video_path: Path
    output_path: Path
    fps: float
    width: int
    height: int
    n_frames: int
    behavior_intervals: dict[str, list[tuple[int, int]]]
    part_names: list[str]
    x_vals: np.ndarray
    y_vals: np.ndarray
    lk_vals: np.ndarray
    overlay_mode: str
    overlay_settings: dict | None
    context_frames: int
    adv_behavior_info: dict
    display_name_cache: dict
    # When True, encode the entire session continuously (frame 0..n_frames-1)
    # instead of only the bout windows.  Avoids the animal "teleporting" between
    # distant bouts in the output.
    whole_video: bool = False
    segment_workers: int = 1
    # Optional progress callback: (current_seg, total_segs, message) -> None
    # Set after construction once the full task list size is known.
    progress_fn: Any = None


@dataclass
class ExportResult:
    output_path: Path | None = None
    n_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    success: bool = False
    output_paths: list[Path] = field(default_factory=list)


class ExportService:
    """Creates flat review/export tables from pipeline artifacts."""

    def __init__(self) -> None:
        self._project_root: Path | None = None
        self._imports = ImportService()
        self._pose = PoseProcessingService()
        self._behaviors: BehaviorService | None = None
        self._max_docx_frame_rows = 20000
        self._subject_order: list[str] = []

    def set_behavior_service(self, svc: BehaviorService) -> None:
        self._behaviors = svc

    def set_subject_order(self, order: list[str]) -> None:
        """Set a custom subject ordering for exports.

        Subjects in *order* appear first; any remaining subjects are
        appended alphabetically.
        """
        self._subject_order = list(order)

    def _ordered_subjects(self, subjects: dict | set | list) -> list[str]:
        """Return subjects in user-defined order, falling back to sorted."""
        all_subj = set(subjects) if not isinstance(subjects, set) else subjects
        if not self._subject_order:
            return sorted(all_subj)
        ordered = [s for s in self._subject_order if s in all_subj]
        remaining = sorted(all_subj - set(self._subject_order))
        return ordered + remaining

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
        return safe or "unknown_behavior"

    @staticmethod
    def _make_unique_sheet_name(subject: str, used_lower: set[str]) -> str:
        """Return a valid, unique Excel sheet title (<=31 chars, no invalid chars)."""
        raw = str(subject or "").strip()
        # Excel forbids these characters in sheet titles: : \\ / ? * [ ]
        cleaned = re.sub(r"[:\\/\?\*\[\]]", "_", raw).strip()
        base = cleaned[:31] or "subject"
        candidate = base
        counter = 2
        while candidate.lower() in used_lower:
            suffix = f"_{counter}"
            candidate = f"{base[: max(1, 31 - len(suffix))]}{suffix}"
            counter += 1
        used_lower.add(candidate.lower())
        return candidate

    def _behavior_display_name(self, name: str) -> str:
        """Return the short_name for display purposes, falling back to name."""
        if not self._behaviors:
            return name
        raw_norm = name.strip().lower()
        for candidate in self._behaviors.behaviors:
            candidate_name = str(candidate.name or "").strip()
            if candidate_name.lower() == raw_norm:
                short = str(candidate.short_name or "").strip()
                return short if short else candidate_name
        return name

    def _behavior_name(self, behavior_id: str | None) -> str:
        """Resolve behavior IDs/tokens/names to a canonical display name."""
        raw = (behavior_id or "").strip()
        if not raw:
            return "unknown_behavior"
        if self._behaviors:
            defn = self._behaviors.get(raw)
            if defn:
                return defn.name
            raw_norm = raw.lower()
            raw_safe = self._safe_name(raw).lower()
            for candidate in self._behaviors.behaviors:
                candidate_id = str(candidate.behavior_id or "").strip()
                candidate_name = str(candidate.name or "").strip()
                aliases = {
                    candidate_id,
                    candidate_name,
                    self._safe_name(candidate_id),
                    self._safe_name(candidate_name),
                }
                aliases_norm = {alias.lower() for alias in aliases if alias}
                aliases_safe = {self._safe_name(alias).lower() for alias in aliases if alias}
                if raw_norm in aliases_norm or raw_safe in aliases_safe:
                    return candidate_name or candidate_id
        return raw

    def _canonicalize_behavior_intervals(
        self,
        intervals_by_session: dict[str, dict[str, list[tuple[int, int]]]],
    ) -> dict[str, dict[str, list[tuple[int, int]]]]:
        """Merge behavior aliases (id/name/safe token) to canonical behavior names."""
        out: dict[str, dict[str, list[tuple[int, int]]]] = {}
        for session_id, by_behavior in intervals_by_session.items():
            merged: dict[str, set[tuple[int, int]]] = {}
            for behavior_raw, intervals in by_behavior.items():
                behavior_name = self._behavior_name(str(behavior_raw))
                bucket = merged.setdefault(behavior_name, set())
                for start, end in intervals:
                    s = int(start)
                    e = int(end)
                    if e < s:
                        e = s
                    bucket.add((s, e))
            out[session_id] = {k: sorted(v, key=lambda x: (x[0], x[1])) for k, v in merged.items()}
        return out

    @staticmethod
    def _normalize_behavior_token(value: str) -> str:
        raw = str(value or "").strip().lower()
        return "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")

    @classmethod
    def _is_no_behavior_token(cls, value: str) -> bool:
        norm = cls._normalize_behavior_token(value)
        return norm in {"no_behavior", "no_behaviour", "nobehavior", "nobehaviour"}

    def _behavior_alias_tokens(self) -> dict[str, set[str]]:
        """Map normalized aliases to the full alias token set for that behavior."""
        alias_map: dict[str, set[str]] = {}
        if not self._behaviors:
            return alias_map
        for behavior in self._behaviors.behaviors:
            behavior_id = str(behavior.behavior_id or "").strip()
            behavior_name = str(behavior.name or "").strip()
            aliases = {
                behavior_id,
                behavior_name,
                self._safe_name(behavior_id),
                self._safe_name(behavior_name),
            }
            normalized = {
                self._normalize_behavior_token(alias)
                for alias in aliases
                if str(alias).strip()
            }
            for token in normalized:
                alias_map[token] = set(normalized)
        return alias_map

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root

    def export_review_csv(
        self,
        candidates: list[CandidateWindow],
        decisions: list[ReviewDecision],
        filename: str = "review_export.csv",
    ) -> ExportResult:
        out = ExportResult()
        if not self._project_root:
            out.warnings.append("No project loaded.")
            return out
        intervals_by_session = self._confirmed_intervals_by_session(candidates, decisions)
        if not intervals_by_session:
            out.warnings.append("No confirmed bouts to export.")
            return out

        subject_by_session = self._subject_by_session()

        rows: list[dict] = []
        for session_id in sorted(intervals_by_session.keys()):
            by_behavior = intervals_by_session.get(session_id, {})
            subject_id = subject_by_session.get(session_id, session_id)
            for behavior in sorted(by_behavior.keys()):
                for start_frame, end_frame in by_behavior.get(behavior, []):
                    rows.append(
                        {
                            "candidate_id": None,
                            "session_id": session_id,
                            "subject_id": subject_id,
                            "start_frame": int(start_frame),
                            "end_frame": int(end_frame),
                            "adjusted_start_frame": int(start_frame),
                            "adjusted_end_frame": int(end_frame),
                            "behavior": behavior,
                            "seed_similarity_score": None,
                            "total_score": None,
                            "review_decision": "confirmed_bout",
                            "reviewer": None,
                            "review_notes": "Exported from confirmed bout intervals.",
                        }
                    )

        df = pd.DataFrame(rows)
        out_dir = self._project_root / "exports" / "csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / filename
        df.to_csv(output, index=False, encoding="utf-8-sig")

        out.output_path = output
        out.n_rows = len(rows)
        out.success = True
        return out

    def export_behavior_presence_docx(
        self,
        candidates: list[CandidateWindow],
        decisions: list[ReviewDecision],
        filename: str = "behavior_presence.docx",
    ) -> ExportResult:
        """Export per-behavior, per-frame 0/1 presence tables into a Word document.

        Each page corresponds to a behavior. Table columns are subjects (session_id),
        and each row is a frame index.
        """
        out = ExportResult()
        if not self._project_root:
            out.warnings.append("No project loaded.")
            return out

        document_factory = None
        try:
            docx_module = importlib.import_module("docx")
            document_factory = getattr(docx_module, "Document")
        except Exception:
            out.warnings.append("python-docx is not installed. Install dependencies and retry export.")
            return out

        intervals_by_session = self._confirmed_intervals_by_session(candidates, decisions)
        if not intervals_by_session:
            out.warnings.append("No confirmed behavior bouts to export.")
            return out

        by_behavior: dict[str, dict[str, list[tuple[int, int]]]] = {}
        for session_id, behavior_map in intervals_by_session.items():
            for behavior, intervals in behavior_map.items():
                by_behavior.setdefault(behavior, {})[session_id] = list(intervals)

        doc = document_factory()
        first_behavior = True
        total_rows = 0
        for behavior in sorted(by_behavior):
            session_intervals = by_behavior[behavior]
            by_subject: dict[str, list[tuple[int, int]]] = {}
            subject_by_session = self._subject_by_session()
            for session_id, intervals in session_intervals.items():
                subject = subject_by_session.get(session_id, session_id)
                by_subject.setdefault(subject, []).extend((int(s), int(e)) for s, e in intervals)

            subjects = self._ordered_subjects(by_subject)
            max_end = max(end for bouts in by_subject.values() for _, end in bouts)
            frame_count = max_end + 1
            row_step = max(1, int(np.ceil(frame_count / max(1, self._max_docx_frame_rows))))
            sampled_frames = list(range(0, frame_count, row_step))

            if row_step > 1:
                out.warnings.append(
                    f"Behavior '{behavior}': frame matrix downsampled every {row_step} frame(s) "
                    f"to keep output size manageable ({len(sampled_frames)} rows from {frame_count})."
                )

            if not first_behavior:
                doc.add_page_break()
            first_behavior = False

            doc.add_heading(f"Behavior: {behavior}", level=1)
            doc.add_paragraph(
                f"Subjects: {len(subjects)} | Frames: 0-{max_end} | Confirmed bouts: {sum(len(v) for v in by_subject.values())}"
            )
            if row_step > 1:
                doc.add_paragraph(f"Note: rows are sampled every {row_step} frame(s).")

            table = doc.add_table(rows=len(sampled_frames) + 1, cols=len(subjects) + 1)
            table.style = "Table Grid"
            table.cell(0, 0).text = "frame"
            for col, subject in enumerate(subjects, start=1):
                table.cell(0, col).text = subject

            sampled_arr = np.asarray(sampled_frames, dtype=np.int32)
            presence_by_subject: dict[str, np.ndarray] = {}
            for subject in subjects:
                mask = np.zeros(len(sampled_frames), dtype=np.uint8)
                for start, end in by_subject[subject]:
                    s = max(0, min(start, frame_count - 1))
                    e = max(s, min(end, frame_count - 1))
                    li = int(np.searchsorted(sampled_arr, s, side="left"))
                    ri = int(np.searchsorted(sampled_arr, e, side="right")) - 1
                    if li < len(mask) and ri >= li:
                        mask[li : ri + 1] = 1
                presence_by_subject[subject] = mask

            for row_idx, frame in enumerate(sampled_frames):
                table.cell(row_idx + 1, 0).text = str(frame)
                for col, subject in enumerate(subjects, start=1):
                    table.cell(row_idx + 1, col).text = str(int(presence_by_subject[subject][row_idx]))
            total_rows += len(sampled_frames)

        out_dir = self._project_root / "exports" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / filename
        doc.save(output)

        out.output_path = output
        out.n_rows = total_rows
        out.success = True
        return out

    def export_boutframes_xlsx(
        self,
        candidates: list[CandidateWindow],
        decisions: list[ReviewDecision],
        filename: str = "boutframes.xlsx",
        include_end_frames: bool = False,
        behavior_filter: list[str] | None = None,
        binary_mode: bool = False,
    ) -> ExportResult:
        """Export bout start frames by subject and behavior into an Excel workbook.

        Workbook format (standard mode):
        - one sheet per subject (session_id)
        - one column per behavior (start frames), or two columns per behavior when include_end_frames=True
        - rows list bout starts (and ends when requested)

        Binary mode (binary_mode=True):
        - one sheet per subject
        - one row per frame (0 to max_end_frame inclusive)
        - column "frame" plus one column per behavior, values 0 or 1
        """
        out = ExportResult()
        if not self._project_root:
            out.warnings.append("No project loaded.")
            return out

        intervals_by_session = self._confirmed_intervals_by_session(candidates, decisions)
        if not intervals_by_session:
            out.warnings.append("No confirmed behavior bouts to export.")
            return out

        # Apply behavior filter
        all_behaviors = sorted({
            behavior
            for by_behavior in intervals_by_session.values()
            for behavior in by_behavior.keys()
        })
        if behavior_filter is not None:
            allowed = {b.strip() for b in behavior_filter}
            behaviors = [b for b in all_behaviors if b in allowed] or all_behaviors
        else:
            behaviors = all_behaviors

        subject_by_session = self._subject_by_session()

        # Detect whether any subject has more than one session in the data.
        # If so, we produce one workbook per distinct session type instead of
        # merging all sessions for a subject into a single file.
        subject_session_ids: dict[str, list[str]] = {}
        for sid in intervals_by_session:
            subj = subject_by_session.get(sid, sid)
            subject_session_ids.setdefault(subj, []).append(sid)
        multi_session = any(len(ids) > 1 for ids in subject_session_ids.values())

        out_dir = self._project_root / "exports" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        p = Path(filename)
        stem = p.stem
        suffix = p.suffix or ".xlsx"

        # Build (session_type, [session_ids]) groups to export.
        if not multi_session:
            session_groups: list[tuple[str, list[str]]] = [("", list(intervals_by_session.keys()))]
        else:
            session_type_by_sid = self._session_type_by_session()
            type_groups: dict[str, list[str]] = {}
            for sid in intervals_by_session:
                stype = session_type_by_sid.get(sid, "") or ""
                type_groups.setdefault(stype, []).append(sid)
            session_groups = sorted(type_groups.items(), key=lambda x: x[0])

        total_rows = 0
        output_paths: list[Path] = []

        for session_type, session_ids in session_groups:
            by_subject: dict[str, dict[str, list[tuple[int, int]]]] = {}
            for session_id in session_ids:
                by_behavior_data = intervals_by_session[session_id]
                subject = subject_by_session.get(session_id, session_id)
                subject_block = by_subject.setdefault(subject, {})
                for behavior, intervals in by_behavior_data.items():
                    if behavior not in behaviors:
                        continue
                    rows = subject_block.setdefault(behavior, [])
                    for start, end in intervals:
                        rows.append((int(start), int(end)))

            if not by_subject:
                continue

            out_filename = f"{stem}_{session_type}{suffix}" if session_type else filename
            output = out_dir / out_filename

            used_sheet_names_lower: set[str] = set()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                for subject in self._ordered_subjects(by_subject):
                    intervals_by_behavior: dict[str, list[tuple[int, int]]] = {
                        b: sorted(by_subject[subject].get(b, []), key=lambda x: (x[0], x[1]))
                        for b in behaviors
                    }
                    sheet_name = self._make_unique_sheet_name(subject, used_sheet_names_lower)

                    if binary_mode:
                        # Determine total frame count from all interval end frames
                        max_end = max(
                            (e for vals in intervals_by_behavior.values() for _s, e in vals),
                            default=0,
                        )
                        n_frames = max_end + 1
                        import numpy as _np  # noqa: PLC0415
                        frame_col = list(range(n_frames))
                        binary_data: dict[str, Any] = {"frame": frame_col}
                        for b in behaviors:
                            arr = _np.zeros(n_frames, dtype=_np.int8)
                            for s, e in intervals_by_behavior.get(b, []):
                                s = max(0, s)
                                e = min(n_frames - 1, e)
                                arr[s : e + 1] = 1
                            binary_data[b] = arr.tolist()
                        df = pd.DataFrame(binary_data)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        total_rows += len(df)
                    else:
                        max_len = max((len(vals) for vals in intervals_by_behavior.values()), default=0)
                        data: dict[str, list[int | None]] = {}
                        for b in behaviors:
                            vals = intervals_by_behavior.get(b, [])
                            starts = [int(s) for s, _e in vals]
                            padded_starts: list[int | None] = starts + [None] * (max_len - len(starts))
                            if include_end_frames:
                                ends = [int(e) for _s, e in vals]
                                padded_ends: list[int | None] = ends + [None] * (max_len - len(ends))
                                data[f"{b}__start"] = padded_starts
                                data[f"{b}__end"] = padded_ends
                            else:
                                data[b] = padded_starts
                        df = pd.DataFrame(data)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        total_rows += len(df)

                # Summary sheet for bout counts by subject and behavior.
                count_rows: list[dict[str, Any]] = []
                for subject in self._ordered_subjects(by_subject):
                    row: dict[str, Any] = {"subject": subject}
                    total_bouts = 0
                    for b in behaviors:
                        n = int(len(by_subject[subject].get(b, [])))
                        row[b] = n
                        total_bouts += n
                    row["total_bouts"] = total_bouts
                    count_rows.append(row)
                if count_rows:
                    counts_df = pd.DataFrame(count_rows)
                    counts_df.to_excel(writer, sheet_name="_bout_counts", index=False)

            output_paths.append(output)

        out.output_path = output_paths[0] if output_paths else None
        out.output_paths = output_paths
        out.n_rows = total_rows
        out.success = True
        return out

    def build_behaviogram(
        self,
        candidates: list[CandidateWindow],
        decisions: list[ReviewDecision],
        behavior_filter: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Build per-subject accepted behavior intervals for behaviogram rendering."""
        intervals_by_session = self._confirmed_intervals_by_session(candidates, decisions)
        if behavior_filter is not None:
            selected = {self._normalize_behavior_token(v) for v in behavior_filter if str(v).strip()}
            alias_map = self._behavior_alias_tokens()
            allowed: set[str] = set()
            for token in selected:
                allowed.add(token)
                allowed.update(alias_map.get(token, set()))
            intervals_by_session = {
                sid: {
                    b: ivs
                    for b, ivs in by_b.items()
                    if self._normalize_behavior_token(b) in allowed
                }
                for sid, by_b in intervals_by_session.items()
            }
        if not intervals_by_session:
            return {}

        subject_by_session = self._subject_by_session()
        by_subject: dict[str, dict[str, Any]] = {}

        for session_id, behavior_map in intervals_by_session.items():
            subject_id = subject_by_session.get(session_id, session_id)
            for behavior, intervals in behavior_map.items():
                for start, end in intervals:
                    start_i = int(start)
                    end_i = int(end)
                    if end_i < start_i:
                        end_i = start_i

                    subject_block = by_subject.setdefault(
                        subject_id,
                        {
                            "max_end_frame": -1,
                            "behaviors": {},
                        },
                    )
                    behaviors: dict[str, list[tuple[int, int]]] = subject_block["behaviors"]
                    behaviors.setdefault(behavior, []).append((start_i, end_i))
                    subject_block["max_end_frame"] = max(int(subject_block["max_end_frame"]), end_i)

        for subject_block in by_subject.values():
            behaviors = subject_block["behaviors"]
            for behavior, intervals in behaviors.items():
                intervals.sort(key=lambda x: (x[0], x[1]))

        return by_subject

    def export_labeled_tracking_videos(
        self,
        candidates: list[CandidateWindow],
        decisions: list[ReviewDecision],
        progress_callback: Callable[[int, int, str], None] | None = None,
        behavior_filter: list[str] | None = None,
        subject_filter: list[str] | None = None,
        session_filter: list[str] | None = None,
        overlay_mode: str = "basic",
        context_frames: int = 150,
        n_workers: int = 0,
        whole_video: bool = True,
    ) -> ExportResult:
        """Export annotated tracking videos for sessions that have confirmed bouts.

        When *whole_video* is True (the default) each session is re-encoded in full
        — every frame from 0 to the last frame — with overlays drawn on the frames
        that fall inside a behavior bout.  This produces one continuous video per
        session with no gaps, so the animal never "teleports" between bouts.

        When *whole_video* is False, only the frames within behavior intervals
        (±*context_frames*) are decoded and encoded and then concatenated.  For
        sparse behavior data this is much faster, but distant bouts are spliced
        together so the animal appears to jump between them.

        *overlay_mode* controls the on-screen annotation style:
        - ``"basic"``: current active behavior label in the top-right corner.
        - ``"advanced"``: live cumulative durations for every behavior plus a
          top-centre prediction panel showing the most likely behavior and its
          colour-coded probability.
        """
        out = ExportResult()
        if not self._project_root:
            out.warnings.append("No project loaded.")
            return out

        try:
            import cv2  # noqa: PLC0415
        except Exception:
            out.warnings.append("OpenCV is not installed. Install preprocessing dependencies and retry.")
            return out

        from abel.ui.overlay_settings_dialog import load_overlay_settings  # noqa: PLC0415
        _overlay_settings = load_overlay_settings(self._project_root)

        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None or not manifest.linked_sessions:
            out.warnings.append("No import manifest found or no linked sessions available.")
            return out

        intervals_by_session = self._confirmed_intervals_by_session(candidates, decisions)
        if behavior_filter is not None:
            selected = {
                self._normalize_behavior_token(v)
                for v in behavior_filter
                if str(v).strip()
            }
            alias_map = self._behavior_alias_tokens()
            allowed: set[str] = set()
            for token in selected:
                allowed.add(token)
                allowed.update(alias_map.get(token, set()))
            intervals_by_session = {
                sid: {
                    b: ivs
                    for b, ivs in by_b.items()
                    if self._normalize_behavior_token(b) in allowed
                }
                for sid, by_b in intervals_by_session.items()
            }
        if session_filter is not None:
            allowed_sessions = set(session_filter)
            intervals_by_session = {
                sid: by_b
                for sid, by_b in intervals_by_session.items()
                if sid in allowed_sessions
            }
        elif subject_filter is not None:
            allowed_subjects = {str(v).strip() for v in subject_filter if str(v).strip()}
            subject_by_session = self._subject_by_session()
            intervals_by_session = {
                sid: by_b
                for sid, by_b in intervals_by_session.items()
                if str(subject_by_session.get(sid, sid)).strip() in allowed_subjects
            }
        if not intervals_by_session:
            out.warnings.append("No confirmed behavior bouts found to annotate.")
            return out

        total_behaviors = sum(len(by_b) for by_b in intervals_by_session.values())
        total_bouts = sum(len(intervals) for by_b in intervals_by_session.values() for intervals in by_b.values())
        debug_filters = (
            f"behavior_filter={list(behavior_filter or [])}, "
            f"session_filter={list(session_filter or [])}, "
            f"subject_filter={list(subject_filter or [])}"
        )
        debug_overall = (
            "[export-debug] Prepared labeled video export: "
            f"sessions={len(intervals_by_session)}, behaviors={total_behaviors}, bouts={total_bouts}, {debug_filters}"
        )
        print(debug_overall)
        _progress = progress_callback or (lambda _done, _total, _msg: None)
        _progress(0, 1, debug_overall)

        subject_by_session = self._subject_by_session()
        out_dir = self._project_root / "exports" / "labeled_videos"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute global advanced-overlay behavior info once (read-only, safe across threads)
        _adv_behavior_info_global: dict[str, dict] = (
            self._get_behavior_overlay_info() if overlay_mode == "advanced" else {}
        )

        # ── Task preparation (sequential, fast) ──────────────────────────────
        # Load pose arrays and video metadata here; workers do only cv2 I/O.
        tasks: list[_SessionRenderTask] = []
        total_sessions = max(1, sum(
            1 for s in manifest.linked_sessions if intervals_by_session.get(s.session_id, {})
        ))
        _progress(0, total_sessions, "Preparing session tasks...")

        for session in manifest.linked_sessions:
            session_id = session.session_id
            behavior_intervals = intervals_by_session.get(session_id, {})
            if not behavior_intervals:
                continue

            video_path = self._imports.video_path_for_session(manifest, session_id)
            pose_path = self._imports.pose_path_for_session(manifest, session_id)
            if not video_path or not video_path.exists():
                out.warnings.append(f"[{session_id}] video not found.")
                continue
            if not pose_path or not pose_path.exists():
                out.warnings.append(f"[{session_id}] pose file not found.")
                continue

            try:
                pose = self._pose.load_and_clean(pose_path, manifest.smoothing_settings)
            except Exception as exc:
                out.warnings.append(f"[{session_id}] failed to load pose: {exc}")
                continue

            # Open briefly to read video metadata, then release; worker will reopen.
            _probe = cv2.VideoCapture(str(video_path))
            if not _probe.isOpened():
                out.warnings.append(f"[{session_id}] failed to open video.")
                continue

            _fps = _probe.get(cv2.CAP_PROP_FPS) or 30.0
            _width = int(_probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            _height = int(_probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            _n_frames_probe = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            _probe.release()
            if _width <= 0 or _height <= 0:
                out.warnings.append(f"[{session_id}] invalid video dimensions.")
                continue

            subject = (subject_by_session.get(session_id, session_id) or session_id).strip()
            session_bout_count = sum(len(intervals) for intervals in behavior_intervals.values())
            behavior_breakdown = ", ".join(
                f"{name}={len(intervals)}"
                for name, intervals in sorted(behavior_intervals.items(), key=lambda x: str(x[0]).lower())
            )
            debug_session = (
                f"[export-debug] Session {session_id} (subject={subject}): "
                f"behaviors={len(behavior_intervals)}, bouts={session_bout_count}"
            )
            if behavior_breakdown:
                debug_session = f"{debug_session} -> {behavior_breakdown}"
            print(debug_session)

            safe_subject = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in subject)
            safe_session = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)
            output_path = out_dir / f"{safe_subject}_{safe_session}_labeled.mp4"

            _n_frames = min(_n_frames_probe, int(pose.n_frames))
            if _n_frames <= 0:
                out.warnings.append(f"[{session_id}] no frames available.")
                continue

            part_names = list(pose.body_parts)
            try:
                _x_vals = pose.x.to_numpy(dtype=float)
                _y_vals = pose.y.to_numpy(dtype=float)
                _lk_vals = pose.likelihood.to_numpy(dtype=float)
            except Exception as exc:
                out.warnings.append(f"[{session_id}] failed to parse pose matrix: {exc}")
                continue

            _display_name_cache = {b: self._behavior_display_name(b) for b in behavior_intervals}
            tasks.append(_SessionRenderTask(
                session_id=session_id,
                video_path=video_path,
                output_path=output_path,
                fps=_fps,
                width=_width,
                height=_height,
                n_frames=_n_frames,
                behavior_intervals=behavior_intervals,
                part_names=part_names,
                x_vals=_x_vals,
                y_vals=_y_vals,
                lk_vals=_lk_vals,
                overlay_mode=overlay_mode,
                overlay_settings=_overlay_settings,
                context_frames=context_frames,
                adv_behavior_info=_adv_behavior_info_global,
                display_name_cache=_display_name_cache,
                whole_video=whole_video,
            ))

        if not tasks:
            out.warnings.append("No sessions with valid data found for export.")
            return out

        # ── Session-sequential, segment-parallel rendering ───────────────────
        _cpu_workers = max(1, os.cpu_count() or 4)
        _requested_workers = int(n_workers) if int(n_workers) > 0 else min(16, _cpu_workers)
        _segment_workers = max(1, min(_requested_workers, _cpu_workers))
        _total_tasks = len(tasks)
        print(
            f"[export] Rendering {_total_tasks} sessions sequentially "
            f"(up to {_segment_workers} parallel segment workers per session)..."
        )
        _progress(
            0,
            _total_tasks,
            f"Rendering {_total_tasks} sessions (1 at a time, up to {_segment_workers} segment workers)...",
        )

        # Attach a progress callback to every task so the worker can report
        # per-segment status without needing access to shared state.
        _completed_count = 0
        _lock = threading.Lock()

        def _make_progress_fn(sid: str) -> Any:
            def _cb(
                seg_idx: int,
                total_segs: int,
                total_frames: int,
                segment_frame_done: int | None = None,
                segment_frame_total: int | None = None,
            ) -> None:
                if segment_frame_done is not None and segment_frame_total is not None:
                    msg = (
                        f"[{sid}] segment {seg_idx}/{total_segs} "
                        f"frame {segment_frame_done:,}/{segment_frame_total:,} "
                        f"({total_frames:,} frames total)"
                    )
                else:
                    msg = (
                        f"[{sid}] segment {seg_idx}/{total_segs} "
                        f"({total_frames:,} frames total)"
                    )
                print(f"[export] {msg}")
                with _lock:
                    _progress(_completed_count, _total_tasks, msg)
            return _cb

        for task in tasks:
            task.segment_workers = _segment_workers
            task.progress_fn = _make_progress_fn(task.session_id)

        exported = 0

        for task in tasks:
            sid = task.session_id
            try:
                task_warnings, frames_written = ExportService._render_session_worker(task)
            except Exception as exc:
                task_warnings = [f"[{sid}] worker crashed: {exc}"]
                frames_written = 0
            with _lock:
                out.warnings.extend(task_warnings)
                if frames_written > 0:
                    exported += 1
                    out.n_rows += frames_written
                    out.output_path = out_dir
                _completed_count += 1
                _progress(
                    _completed_count,
                    len(tasks),
                    f"Completed {_completed_count}/{len(tasks)} sessions (last: {sid})",
                )

        if exported <= 0:
            out.warnings.append("No labeled videos were exported.")
            return out

        out.success = True
        _progress(len(tasks), len(tasks), "Labeled video export complete")
        return out

    def list_available_subjects(self) -> list[str]:
        mapping = self._subject_by_session()
        if not mapping:
            return []
        subjects = {str(v).strip() for v in mapping.values() if str(v).strip()}
        return self._ordered_subjects(subjects)

    def list_available_sessions(self) -> list[tuple[str, str]]:
        """Return ``(display_label, session_id)`` pairs for every linked session.

        When a subject has only one session, the label is just the subject
        name.  When a subject has multiple sessions, the label includes the
        session type derived from the video filename (e.g. "Mouse1 – Conditioning").
        """
        if not self._project_root:
            return []
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return []
        video_by_id = {v.asset_id: v for v in manifest.videos}
        subject_by_sid: dict[str, str] = {}
        session_type_by_sid: dict[str, str] = {}
        for session in manifest.linked_sessions:
            sid = session.session_id
            subject = (session.subject_id or "").strip()
            video = video_by_id.get(session.video_asset_id)
            if not subject:
                subject = (video.subject_id or "").strip() if video else ""
            if not subject and video:
                subject = Path(video.source_path).stem.strip()
            subject = subject or sid
            subject_by_sid[sid] = subject
            # Derive session type from video filename
            stype = ""
            if video:
                stem = Path(video.source_path).stem
                if subject and stem.startswith(subject):
                    stype = stem[len(subject):].lstrip("_- ")
            session_type_by_sid[sid] = stype
        # Determine which subjects have multiple sessions
        subject_count: dict[str, int] = {}
        for subj in subject_by_sid.values():
            subject_count[subj] = subject_count.get(subj, 0) + 1
        # Build display labels
        result: list[tuple[str, str]] = []
        for session in manifest.linked_sessions:
            sid = session.session_id
            subj = subject_by_sid.get(sid, sid)
            stype = session_type_by_sid.get(sid, "")
            if subject_count.get(subj, 1) > 1 and stype:
                label = f"{subj} \u2013 {stype}"
            elif subject_count.get(subj, 1) > 1:
                label = f"{subj} \u2013 {sid[:8]}"
            else:
                label = subj
            result.append((label, sid))
        return result

    def _subject_by_session(self) -> dict[str, str]:
        if not self._project_root:
            return {}
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return {}
        mapping: dict[str, str] = {}
        video_by_id = {v.asset_id: v for v in manifest.videos}
        for session in manifest.linked_sessions:
            subject = (session.subject_id or "").strip()
            video = video_by_id.get(session.video_asset_id)
            if not subject:
                subject = (video.subject_id or "").strip() if video else ""
            if not subject and video:
                subject = Path(video.source_path).stem.strip()
            mapping[session.session_id] = subject or session.session_id
        return mapping

    def _session_type_by_session(self) -> dict[str, str]:
        """Derive a session-type label for every session from its video filename.

        The label is produced by stripping the subject prefix (and any leading
        ``_-`` separators) from the video stem.  Returns an empty string for
        sessions where no such suffix can be derived.
        """
        if not self._project_root:
            return {}
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return {}
        subject_by_sid = self._subject_by_session()
        video_by_id = {v.asset_id: v for v in manifest.videos}
        result: dict[str, str] = {}
        for session in manifest.linked_sessions:
            sid = session.session_id
            subject = subject_by_sid.get(sid, "")
            video = video_by_id.get(session.video_asset_id)
            stype = ""
            if video:
                stem = Path(video.source_path).stem
                if subject and stem.startswith(subject):
                    stype = stem[len(subject):].lstrip("_- ")
            result[sid] = stype
        return result

    def _accepted_behavior_intervals_by_session(
        self,
        candidates: list[CandidateWindow],
        decisions: list[ReviewDecision],
    ) -> dict[str, dict[str, list[tuple[int, int]]]]:
        accepted = self._confirmed_candidates(candidates, decisions)
        dec_by_id = {d.clip_id: d for d in decisions}
        out: dict[str, dict[str, list[tuple[int, int]]]] = {}
        for cand in accepted:
            dec = dec_by_id.get(cand.window_id)
            behavior_token = cand.behavior_id
            if dec and str(dec.behavior_label or "").strip():
                behavior_token = str(dec.behavior_label).strip()
            behavior = self._behavior_name(behavior_token)
            start = int(dec.adjusted_start_frame) if dec and dec.adjusted_start_frame is not None else int(cand.start_frame)
            end = int(dec.adjusted_end_frame) if dec and dec.adjusted_end_frame is not None else int(cand.end_frame)
            if end < start:
                end = start
            by_behavior = out.setdefault(cand.session_id, {})
            by_behavior.setdefault(behavior, []).append((start, end))

        for by_behavior in out.values():
            for behavior, intervals in by_behavior.items():
                intervals.sort(key=lambda x: (x[0], x[1]))
        return out

    def _confirmed_intervals_by_session(
        self,
        candidates: list[CandidateWindow],
        decisions: list[ReviewDecision],
    ) -> dict[str, dict[str, list[tuple[int, int]]]]:
        temporal = self._canonicalize_behavior_intervals(self._temporal_confirmed_intervals_by_session())
        bouts_store = self._canonicalize_behavior_intervals(self._behavior_bouts_intervals_by_session())
        fallback = self._canonicalize_behavior_intervals(
            self._accepted_behavior_intervals_by_session(candidates, decisions)
        )
        if not temporal and not bouts_store:
            return fallback

        merged: dict[str, dict[str, list[tuple[int, int]]]] = {}
        all_sessions = set(temporal.keys()) | set(bouts_store.keys()) | set(fallback.keys())
        for session_id in sorted(all_sessions):
            merged[session_id] = {}
            fallback_behaviors = fallback.get(session_id, {})
            bouts_behaviors = bouts_store.get(session_id, {})
            temporal_behaviors = temporal.get(session_id, {})
            all_behaviors = (
                set(fallback_behaviors.keys())
                | set(bouts_behaviors.keys())
                | set(temporal_behaviors.keys())
            )
            for behavior in sorted(all_behaviors):
                if behavior in temporal_behaviors:
                    merged[session_id][behavior] = list(temporal_behaviors[behavior])
                elif behavior in bouts_behaviors:
                    merged[session_id][behavior] = list(bouts_behaviors[behavior])
                else:
                    merged[session_id][behavior] = list(fallback_behaviors.get(behavior, []))
        return merged

    def _behavior_bouts_intervals_by_session(self) -> dict[str, dict[str, list[tuple[int, int]]]]:
        """Read pre-computed bout parquets from derived/behavior_bouts/."""
        if self._project_root is None:
            return {}

        bouts_dir = self._project_root / "derived" / "behavior_bouts"
        if not bouts_dir.exists():
            return {}

        out: dict[str, dict[str, list[tuple[int, int]]]] = {}
        for path in sorted(bouts_dir.glob("*_bouts.parquet")):
            behavior_id = path.stem.removesuffix("_bouts")
            if not behavior_id:
                continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if df.empty or not {"session_id", "start_frame", "end_frame"}.issubset(df.columns):
                continue

            behavior_name = self._behavior_name(behavior_id)
            for session_id, group in df.groupby("session_id"):
                sid = str(session_id)
                by_behavior = out.setdefault(sid, {})
                by_behavior.setdefault(behavior_name, []).extend(
                    (int(row.start_frame), int(row.end_frame))
                    for row in group[["start_frame", "end_frame"]].itertuples(index=False)
                )

        for by_behavior in out.values():
            for behavior, intervals in by_behavior.items():
                intervals.sort(key=lambda x: (x[0], x[1]))
        return out

    def _temporal_confirmed_intervals_by_session(self) -> dict[str, dict[str, list[tuple[int, int]]]]:
        if self._project_root is None:
            return {}

        root = self._project_root / "derived" / "temporal_refinement"
        if not root.exists():
            return {}

        # Primary source: recompute bouts from probability traces using the
        # current temporal-review threshold settings + smoothing, exactly as
        # the Temporal Review UI does.  This guarantees exports match what
        # the user sees after adjusting thresholds.
        out = self._competition_trace_intervals_by_session()

        # Fallback: saved bout parquets from the last postprocess run.  Only
        # used for session/behavior combos where traces are unavailable.
        for concept_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            latest_path = concept_dir / "latest.json"
            if not latest_path.exists():
                continue
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            concept_id = str(latest.get("concept_id", concept_dir.name)).strip() or concept_dir.name
            post_dir_raw = str(latest.get("postprocess_dir", "")).strip()
            if not post_dir_raw:
                continue

            post_manifest_path = Path(post_dir_raw) / "postprocess_manifest.json"
            if not post_manifest_path.exists():
                continue
            try:
                post_manifest = json.loads(post_manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            behavior_name = self._behavior_name(concept_id)
            if behavior_name in {"target_behavior", "unknown_behavior"}:
                behavior_name = self._behavior_name(concept_dir.name)
            if behavior_name in {"target_behavior", "unknown_behavior"}:
                continue
            bout_paths = {str(k): str(v) for k, v in (post_manifest.get("bout_paths", {}) or {}).items()}
            for session_id, path_raw in bout_paths.items():
                # Only fill in if trace recomputation didn't produce results
                target = out.setdefault(str(session_id), {})
                if target.get(behavior_name):
                    continue

                path = Path(str(path_raw).strip())
                if not path.exists():
                    continue
                try:
                    df = pd.read_parquet(path)
                except Exception:
                    continue
                if df.empty or not {"start_frame", "end_frame"}.issubset(df.columns):
                    continue

                target.setdefault(behavior_name, [])
                target[behavior_name].extend(
                    (int(row.start_frame), int(row.end_frame))
                    for row in df[["start_frame", "end_frame"]].itertuples(index=False)
                )

        for by_behavior in out.values():
            for behavior, intervals in by_behavior.items():
                intervals.sort(key=lambda x: (x[0], x[1]))
        return out

    def _competition_trace_intervals_by_session(self) -> dict[str, dict[str, list[tuple[int, int]]]]:
        """Recompute bout intervals from probability traces using current
        temporal-review settings + smoothing, matching the Temporal Review UI."""
        if self._project_root is None:
            return {}

        root = self._project_root / "derived" / "temporal_refinement"
        if not root.exists():
            return {}

        # Collect trace paths from ALL concept directories, not just target_behavior.
        all_trace_paths: dict[str, str] = {}
        for concept_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            latest_path = concept_dir / "latest.json"
            if not latest_path.exists():
                continue
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            inference_dir = Path(str(latest.get("inference_dir", "")).strip())
            manifest_path = inference_dir / "inference_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            for k, v in dict(manifest.get("trace_paths", {}) or {}).items():
                session_id = str(k)
                # Keep the first (or latest) trace path per session
                if session_id not in all_trace_paths:
                    all_trace_paths[session_id] = str(v)

        out: dict[str, dict[str, list[tuple[int, int]]]] = {}
        for session_id, path_raw in all_trace_paths.items():
            path = Path(str(path_raw).strip())
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if df.empty or "frame" not in df.columns:
                continue

            # Support both multi-behavior (prob_*) and single-behavior (probability) columns
            prob_cols = [c for c in df.columns if str(c).startswith("prob_")]
            single_prob = "probability" in df.columns and not prob_cols
            if single_prob:
                prob_cols = ["probability"]
            if not prob_cols:
                continue

            frames = pd.to_numeric(df["frame"], errors="coerce").fillna(0).to_numpy(dtype=int)
            by_behavior = out.setdefault(str(session_id), {})
            for col in prob_cols:
                if single_prob:
                    # Single-column traces don't encode a behavior token;
                    # skip — they are handled by the bout-parquet fallback.
                    continue
                token = str(col).removeprefix("prob_").strip()
                if not token or self._is_no_behavior_token(token):
                    continue

                behavior_name = self._behavior_name(token)
                if self._is_no_behavior_token(behavior_name):
                    continue

                cfg = self._temporal_review_settings_for_token(token)
                onset = float(cfg.get("onset_threshold", 0.65))
                min_bout = int(cfg.get("min_bout_duration_frames", 8))
                merge_gap = int(cfg.get("merge_gap_frames", 4))

                probs = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                # Apply smoothing BEFORE thresholding, matching Temporal Review UI
                probs = smooth_probabilities(probs, method="moving_average", window=5)
                binary = self._threshold_probabilities(probs, onset)
                binary = self._merge_close_bouts(binary, merge_gap)
                binary = self._remove_short_bouts(binary, min_bout)
                intervals = self._binary_trace_to_intervals(binary)
                if not intervals:
                    continue

                mapped: list[tuple[int, int]] = []
                for s, e in intervals:
                    if s < 0 or s >= len(frames):
                        continue
                    e_idx = min(max(s, e), len(frames) - 1)
                    start_frame = int(frames[s])
                    end_frame = int(frames[e_idx])
                    if end_frame < start_frame:
                        end_frame = start_frame
                    mapped.append((start_frame, end_frame))

                if mapped:
                    by_behavior.setdefault(behavior_name, []).extend(mapped)

        for by_behavior in out.values():
            for behavior, intervals in by_behavior.items():
                intervals.sort(key=lambda x: (x[0], x[1]))
        return out

    def _temporal_review_settings_for_token(self, token: str) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "onset_threshold": 0.65,
            "min_bout_duration_frames": 8,
            "merge_gap_frames": 4,
        }
        if self._project_root is None:
            return defaults

        path = self._project_root / "config" / "temporal_review_settings.json"
        if not path.exists():
            return defaults

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return defaults

        global_cfg = dict(raw.get("__all__", {}) or {})
        cfg = {**defaults, **global_cfg}
        by_behavior = dict(raw.get("by_behavior", {}) or {})

        candidate_keys: set[str] = {
            str(token).strip(),
            self._safe_name(token),
            self._behavior_name(token),
            self._safe_name(self._behavior_name(token)),
        }
        if self._behaviors:
            token_norm = self._normalize_behavior_token(token)
            for behavior in self._behaviors.behaviors:
                bid = str(behavior.behavior_id or "").strip()
                bname = str(behavior.name or "").strip()
                aliases = {
                    bid,
                    bname,
                    self._safe_name(bid),
                    self._safe_name(bname),
                }
                alias_tokens = {
                    self._normalize_behavior_token(v)
                    for v in aliases
                    if str(v).strip()
                }
                if token_norm in alias_tokens:
                    candidate_keys.update({bid, bname, self._safe_name(bid), self._safe_name(bname)})

        for key in candidate_keys:
            k = str(key or "").strip()
            if not k:
                continue
            values = by_behavior.get(k)
            if isinstance(values, dict):
                cfg.update(values)

        return cfg

    @staticmethod
    def _threshold_probabilities(prob_trace: np.ndarray, onset_thresh: float) -> np.ndarray:
        x = np.asarray(prob_trace, dtype=np.float32)
        return (x >= float(onset_thresh)).astype(np.uint8)

    @staticmethod
    def _remove_short_bouts(binary_trace: np.ndarray, min_duration_frames: int) -> np.ndarray:
        x = np.asarray(binary_trace, dtype=np.uint8).copy()
        minimum = max(1, int(min_duration_frames))
        i = 0
        while i < len(x):
            if x[i] == 0:
                i += 1
                continue
            j = i
            while j < len(x) and x[j] == 1:
                j += 1
            if (j - i) < minimum:
                x[i:j] = 0
            i = j
        return x

    @staticmethod
    def _merge_close_bouts(binary_trace: np.ndarray, max_gap_frames: int) -> np.ndarray:
        x = np.asarray(binary_trace, dtype=np.uint8).copy()
        gap = max(0, int(max_gap_frames))
        if gap <= 0 or len(x) == 0:
            return x
        i = 0
        while i < len(x):
            while i < len(x) and x[i] == 0:
                i += 1
            if i >= len(x):
                break
            j = i
            while j < len(x) and x[j] == 1:
                j += 1
            k = j
            while k < len(x) and x[k] == 0:
                k += 1
            if k < len(x) and (k - j) <= gap:
                x[j:k] = 1
                i = k
            else:
                i = k
        return x

    @staticmethod
    def _binary_trace_to_intervals(binary_trace: np.ndarray) -> list[tuple[int, int]]:
        x = np.asarray(binary_trace, dtype=np.uint8)
        intervals: list[tuple[int, int]] = []
        i = 0
        while i < len(x):
            if x[i] == 0:
                i += 1
                continue
            j = i
            while j < len(x) and x[j] == 1:
                j += 1
            intervals.append((int(i), int(j - 1)))
            i = j
        return intervals

    @staticmethod
    def _render_session_worker(task: _SessionRenderTask) -> tuple[list[str], int]:
        """Render one session's labeled video. Thread-safe: all state is local."""
        import cv2 as _cv2  # noqa: PLC0415
        from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

        warnings_out: list[str] = []
        session_id = task.session_id
        n_frames = task.n_frames

        # Build per-behavior boolean arrays for O(1) active-set lookup per frame.
        _behavior_names_sorted = sorted(task.behavior_intervals.keys())
        _behavior_active_arr: dict[str, np.ndarray] = {}
        for _bname, _bintervals in task.behavior_intervals.items():
            _a = np.zeros(n_frames, dtype=bool)
            for _s, _e in _bintervals:
                _a[max(0, _s) : min(n_frames, _e + 1)] = True
            _behavior_active_arr[_bname] = _a

        # Build seek-based segment list.  In whole-video mode the entire session
        # is encoded continuously so the output never jumps between distant bouts;
        # the full range is split into contiguous chunks only so the segment
        # workers can still render in parallel (the chunks are merged back in
        # order, yielding one seamless video).  Otherwise only bout windows are
        # encoded.
        if task.whole_video:
            export_segments = ExportService._split_contiguous_range(
                n_frames, max(1, int(task.segment_workers))
            )
        else:
            export_segments = ExportService._build_export_segments(
                task.behavior_intervals, n_frames, task.context_frames
            )
        if not export_segments:
            warnings_out.append(f"[{session_id}] no export segments after interval merge.")
            return warnings_out, 0

        total_export_frames = sum(e - s + 1 for s, e in export_segments)
        print(
            f"[export-debug] {session_id}: {len(export_segments)} segment(s), "
            f"{total_export_frames} frames to encode (of {n_frames} total)"
        )

        # Advanced overlay: cumulative sums for fast-forward after seek.
        _adv_sorted_names: list[str] = []
        _adv_cumulative: dict[str, int] = {}
        _adv_cumsum: dict[str, np.ndarray] = {}
        if task.overlay_mode == "advanced":
            _adv_sorted_names = sorted(task.behavior_intervals.keys(), key=str.lower)
            _adv_cumulative = {name: 0 for name in _adv_sorted_names}
            _adv_cumsum = {
                name: np.cumsum(_behavior_active_arr[name]) for name in _adv_sorted_names
            }

        # Per-session loop-invariant constants.
        _ov_s = task.overlay_settings or {}
        _kp_r_offset = int(_ov_s.get("keypoint_radius_offset", 0))
        _kp_outline_on = bool(_ov_s.get("keypoint_outline_enabled", True))
        _kp_conf_thresh = float(_ov_s.get("keypoint_confidence_threshold", 0.20))
        _circle_r = max(2, int(round(task.height / 250)) + _kp_r_offset)
        _circle_t = max(1, _circle_r // 2)
        x_vals = task.x_vals
        y_vals = task.y_vals
        lk_vals = task.lk_vals
        _n_parts = min(len(task.part_names), x_vals.shape[1], y_vals.shape[1], lk_vals.shape[1]) if x_vals.ndim == 2 and y_vals.ndim == 2 and lk_vals.ndim == 2 else 0
        _part_color_cache = [ExportService._part_color(i) for i in range(_n_parts)]

        # Precompute integer coordinates and confidence mask once to reduce per-frame Python overhead.
        _x_int = np.rint(x_vals[:n_frames, :_n_parts]).astype(np.int32, copy=False) if _n_parts > 0 else np.empty((n_frames, 0), dtype=np.int32)
        _y_int = np.rint(y_vals[:n_frames, :_n_parts]).astype(np.int32, copy=False) if _n_parts > 0 else np.empty((n_frames, 0), dtype=np.int32)
        _conf_mask = (lk_vals[:n_frames, :_n_parts] >= _kp_conf_thresh) if _n_parts > 0 else np.empty((n_frames, 0), dtype=bool)
        total_segs = len(export_segments)
        heartbeat_every = max(120, int(round(task.fps * 5.0)))

        def _draw_segment(
            cap: Any,
            writer: Any,
            seg_idx: int,
            seg_start: int,
            seg_end: int,
        ) -> tuple[int, list[str]]:
            local_warnings: list[str] = []
            local_frames_written = 0
            _label_render_cache: dict[str, tuple] = {}
            _adv_buf: dict = {}
            _adv_cumulative_local: dict[str, int] = {}

            if task.overlay_mode == "advanced" and _adv_cumsum:
                for name in _adv_sorted_names:
                    if seg_start <= 0:
                        _adv_cumulative_local[name] = 0
                    else:
                        _adv_cumulative_local[name] = int(_adv_cumsum[name][min(seg_start - 1, n_frames - 1)])

            if task.progress_fn is not None:
                task.progress_fn(seg_idx + 1, total_segs, total_export_frames)

            segment_frame_total = max(1, seg_end - seg_start + 1)
            # Guard against seek imprecision: compressed video (e.g. H.264/MP4)
            # can only seek to the nearest keyframe, which may be *before*
            # seg_start.  Discard frames without decoding until the capture is
            # positioned exactly at seg_start so we never write frames that
            # belong to the gap between segments.
            actual_pos = int(round(cap.get(_cv2.CAP_PROP_POS_FRAMES)))
            while actual_pos < seg_start:
                if not cap.grab():
                    break
                actual_pos += 1
            for frame_idx in range(seg_start, seg_end + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    local_warnings.append(f"[{session_id}] segment {seg_idx + 1}/{total_segs} decode failed at frame {frame_idx}.")
                    break

                if _n_parts > 0 and frame_idx < _conf_mask.shape[0]:
                    visible_parts = np.flatnonzero(_conf_mask[frame_idx])
                    for part_idx in visible_parts.tolist():
                        px = int(_x_int[frame_idx, part_idx])
                        py = int(_y_int[frame_idx, part_idx])
                        color = _part_color_cache[part_idx]
                        _cv2.circle(frame, (px, py), _circle_r, color, -1, lineType=_cv2.LINE_AA)
                        if _kp_outline_on:
                            _cv2.circle(frame, (px, py), _circle_r, (0, 0, 0), _circle_t, lineType=_cv2.LINE_AA)

                active = [
                    name for name in _behavior_names_sorted
                    if _behavior_active_arr[name][frame_idx]
                ]

                if task.overlay_mode == "advanced":
                    for name in active:
                        _adv_cumulative_local[name] = _adv_cumulative_local.get(name, 0) + 1
                    ExportService._draw_advanced_overlay(
                        frame,
                        task.fps,
                        _adv_sorted_names,
                        active,
                        _adv_cumulative_local,
                        task.adv_behavior_info,
                        overlay_settings=task.overlay_settings,
                        _buf=_adv_buf,
                    )
                else:
                    display_names = [task.display_name_cache.get(b, b) for b in active]
                    text = "Behavior: " + (", ".join(display_names) if display_names else "none")
                    ExportService._draw_top_right_label(
                        frame,
                        text,
                        overlay_settings=task.overlay_settings,
                        _cache=_label_render_cache,
                    )

                writer.write(frame)
                local_frames_written += 1

                if task.progress_fn is not None:
                    segment_frame_done = frame_idx - seg_start + 1
                    if (
                        segment_frame_done == segment_frame_total
                        or (segment_frame_done % heartbeat_every) == 0
                    ):
                        task.progress_fn(
                            seg_idx + 1,
                            total_segs,
                            total_export_frames,
                            segment_frame_done,
                            segment_frame_total,
                        )

            return local_frames_written, local_warnings

        # Sequential fallback: one worker or one segment.
        if task.segment_workers <= 1 or total_segs <= 1:
            cap = _cv2.VideoCapture(str(task.video_path))
            if not cap.isOpened():
                warnings_out.append(f"[{session_id}] failed to open video.")
                return warnings_out, 0

            writer = _cv2.VideoWriter(
                str(task.output_path),
                _cv2.VideoWriter_fourcc(*"mp4v"),
                task.fps,
                (task.width, task.height),
            )
            if not writer.isOpened():
                cap.release()
                warnings_out.append(f"[{session_id}] failed to create output video.")
                return warnings_out, 0

            frames_written = 0
            try:
                for seg_idx, (seg_start, seg_end) in enumerate(export_segments):
                    cap.set(_cv2.CAP_PROP_POS_FRAMES, seg_start)
                    seg_frames, seg_warnings = _draw_segment(cap, writer, seg_idx, seg_start, seg_end)
                    frames_written += seg_frames
                    warnings_out.extend(seg_warnings)
            finally:
                cap.release()
                writer.release()

            return warnings_out, frames_written

        # Segment-parallel path: render each segment to a temp file, then merge in order.
        segment_workers = max(1, min(int(task.segment_workers), total_segs, os.cpu_count() or 4))
        tmp_dir = task.output_path.parent / f".{task.output_path.stem}_segments_{os.getpid()}_{threading.get_ident()}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        def _render_one_segment(seg_idx: int, seg_start: int, seg_end: int) -> tuple[int, Path, int, list[str]]:
            seg_path = tmp_dir / f"segment_{seg_idx + 1:05d}.mp4"
            cap = _cv2.VideoCapture(str(task.video_path))
            if not cap.isOpened():
                return seg_idx, seg_path, 0, [f"[{session_id}] segment {seg_idx + 1}/{total_segs} failed to open video."]

            cap.set(_cv2.CAP_PROP_POS_FRAMES, seg_start)
            writer = _cv2.VideoWriter(
                str(seg_path),
                _cv2.VideoWriter_fourcc(*"mp4v"),
                task.fps,
                (task.width, task.height),
            )
            if not writer.isOpened():
                cap.release()
                return seg_idx, seg_path, 0, [f"[{session_id}] segment {seg_idx + 1}/{total_segs} failed to create temp writer."]

            try:
                seg_frames, seg_warnings = _draw_segment(cap, writer, seg_idx, seg_start, seg_end)
            finally:
                cap.release()
                writer.release()

            return seg_idx, seg_path, seg_frames, seg_warnings

        segment_paths: dict[int, Path] = {}
        segment_frame_counts: dict[int, int] = {}

        try:
            with ThreadPoolExecutor(max_workers=segment_workers) as pool:
                futures = [
                    pool.submit(_render_one_segment, idx, seg_start, seg_end)
                    for idx, (seg_start, seg_end) in enumerate(export_segments)
                ]
                for future in as_completed(futures):
                    try:
                        seg_idx, seg_path, seg_frames, seg_warnings = future.result()
                    except Exception as exc:
                        warnings_out.append(f"[{session_id}] segment worker crashed: {exc}")
                        continue
                    warnings_out.extend(seg_warnings)
                    if seg_frames > 0 and seg_path.exists():
                        segment_paths[seg_idx] = seg_path
                        segment_frame_counts[seg_idx] = seg_frames

            if not segment_paths:
                warnings_out.append(f"[{session_id}] no segment outputs were rendered.")
                return warnings_out, 0

            writer = _cv2.VideoWriter(
                str(task.output_path),
                _cv2.VideoWriter_fourcc(*"mp4v"),
                task.fps,
                (task.width, task.height),
            )
            if not writer.isOpened():
                warnings_out.append(f"[{session_id}] failed to create output video.")
                return warnings_out, 0

            merged_frames = 0
            try:
                for seg_idx in range(total_segs):
                    seg_path = segment_paths.get(seg_idx)
                    if seg_path is None or not seg_path.exists():
                        warnings_out.append(f"[{session_id}] missing rendered segment {seg_idx + 1}/{total_segs}.")
                        continue

                    seg_cap = _cv2.VideoCapture(str(seg_path))
                    if not seg_cap.isOpened():
                        warnings_out.append(f"[{session_id}] failed to open temp segment {seg_idx + 1}/{total_segs}.")
                        continue
                    try:
                        while True:
                            ok, frame = seg_cap.read()
                            if not ok or frame is None:
                                break
                            writer.write(frame)
                            merged_frames += 1
                    finally:
                        seg_cap.release()
            finally:
                writer.release()

            return warnings_out, merged_frames
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _split_contiguous_range(n_frames: int, n_chunks: int) -> list[tuple[int, int]]:
        """Split ``[0, n_frames-1]`` into up to *n_chunks* contiguous, gap-free ranges.

        The chunks tile the whole range with no overlaps and no gaps, so encoding
        each one and concatenating in order reproduces the full session exactly.
        """
        if n_frames <= 0:
            return []
        n_chunks = max(1, min(int(n_chunks), n_frames))
        base = n_frames // n_chunks
        rem = n_frames % n_chunks
        segments: list[tuple[int, int]] = []
        start = 0
        for i in range(n_chunks):
            length = base + (1 if i < rem else 0)
            if length <= 0:
                continue
            end = start + length - 1
            segments.append((start, end))
            start = end + 1
        return segments

    @staticmethod
    def _build_export_segments(
        behavior_intervals: dict[str, list[tuple[int, int]]],
        n_frames: int,
        context_frames: int,
    ) -> list[tuple[int, int]]:
        """Return a sorted, merged list of (start, end) frame ranges to encode.

        Each behavior interval is expanded by *context_frames* on both sides.
        Overlapping expanded intervals are merged into a single range so we
        never seek backwards.  When *context_frames* == 0 and the interval list
        covers the entire session this degenerates to [(0, n_frames-1)].
        """
        raw: list[tuple[int, int]] = []
        for intervals in behavior_intervals.values():
            for s, e in intervals:
                raw.append((max(0, s - context_frames), min(n_frames - 1, e + context_frames)))
        if not raw:
            return []
        raw.sort()
        merged: list[list[int]] = [list(raw[0])]
        for s, e in raw[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    @staticmethod
    def _active_behaviors_for_frame(
        behavior_intervals: dict[str, list[tuple[int, int]]],
        frame_idx: int,
    ) -> list[str]:
        active: list[str] = []
        for behavior, intervals in behavior_intervals.items():
            for start, end in intervals:
                if start <= frame_idx <= end:
                    active.append(behavior)
                    break
        active.sort()
        return active

    @staticmethod
    def _part_color(part_idx: int) -> tuple[int, int, int]:
        # Distinct, deterministic BGR colors for body-part points.
        palette = [
            (60, 180, 255),
            (80, 220, 80),
            (255, 200, 70),
            (200, 120, 255),
            (255, 110, 110),
            (220, 220, 220),
            (255, 170, 0),
            (180, 255, 255),
        ]
        return palette[part_idx % len(palette)]

    @staticmethod
    def _overlay_supersample_scale(frame_height: int, overlay_settings: dict | None = None) -> int:
        """Choose a higher internal render scale for overlays on small videos."""
        ov = overlay_settings or {}
        explicit = int(ov.get("overlay_supersample_scale", 0) or 0)
        if explicit > 0:
            return max(1, min(4, explicit))
        if frame_height <= 480:
            return 3
        if frame_height <= 720:
            return 2
        return 1

    @staticmethod
    def _blend_masked_color(
        overlay: np.ndarray,
        alpha: np.ndarray,
        mask: np.ndarray,
        color_bgr: tuple[int, int, int],
        opacity: float = 1.0,
    ) -> None:
        if mask.size == 0 or not mask.any():
            return
        mask_f = mask.astype(np.float32)
        mask_f *= float(max(0.0, min(1.0, opacity))) / 255.0
        if float(mask_f.max(initial=0.0)) <= 0.0:
            return
        mask_3 = mask_f[..., np.newaxis]  # view, no copy
        color_arr = np.asarray(color_bgr, dtype=np.float32).reshape(1, 1, 3)
        # In-place: overlay += (color - overlay) * mask_3  (one large temp instead of two)
        tmp = color_arr - overlay
        tmp *= mask_3
        overlay += tmp
        # In-place alpha: alpha = alpha + mask_f * (1 - alpha)
        one_minus_alpha = 1.0 - alpha
        one_minus_alpha *= mask_f
        alpha += one_minus_alpha

    @staticmethod
    def _composite_supersampled_roi(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        supersample_scale: int,
        draw_callback: Callable[[np.ndarray, np.ndarray, int], None],
        _buf: dict | None = None,
    ) -> None:
        import cv2  # noqa: PLC0415

        fh, fw = frame.shape[:2]
        x1 = max(0, min(fw, int(x1)))
        y1 = max(0, min(fh, int(y1)))
        x2 = max(0, min(fw, int(x2)))
        y2 = max(0, min(fh, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return

        roi_w = x2 - x1
        roi_h = y2 - y1
        ss = max(1, int(supersample_scale))
        ss_h, ss_w = roi_h * ss, roi_w * ss

        if _buf is not None:
            if _buf.get('_ss_shape') != (ss_h, ss_w):
                _buf['_overlay'] = np.empty((ss_h, ss_w, 3), dtype=np.float32)
                _buf['_alpha'] = np.empty((ss_h, ss_w), dtype=np.float32)
                _buf['_ss_shape'] = (ss_h, ss_w)
            overlay = _buf['_overlay']
            alpha = _buf['_alpha']
            overlay.fill(0.0)
            alpha.fill(0.0)
        else:
            overlay = np.zeros((ss_h, ss_w, 3), dtype=np.float32)
            alpha = np.zeros((ss_h, ss_w), dtype=np.float32)

        draw_callback(overlay, alpha, ss)

        if ss > 1:
            overlay_small = cv2.resize(overlay, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
            alpha_small = cv2.resize(alpha, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
        else:
            overlay_small = overlay
            alpha_small = alpha

        alpha_small = np.clip(alpha_small, 0.0, 1.0)
        if float(alpha_small.max(initial=0.0)) <= 0.0:
            return

        roi = frame[y1:y2, x1:x2].astype(np.float32)
        roi[:] = overlay_small * alpha_small[..., None] + roi * (1.0 - alpha_small[..., None])
        frame[y1:y2, x1:x2] = np.clip(roi, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _draw_top_right_label(
        frame, text: str, overlay_settings: dict | None = None, _cache: dict | None = None
    ) -> None:
        import cv2  # noqa: PLC0415

        ov = overlay_settings or {}
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Scale text relative to video height — larger for HD+ content
        scale = max(0.35, min(0.8, h / 1200.0)) * float(ov.get("basic_label_scale_factor", 1.0))
        thickness = max(1, int(round(h / 600)) + int(ov.get("text_thickness_offset", 0)))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        margin = max(6, int(h * 0.005))
        x2 = w - margin
        x1 = max(0, x2 - tw - margin * 2)
        y1 = margin
        y2 = min(h - 1, y1 + th + baseline + margin * 2)

        x1c, y1c = max(0, min(w, x1)), max(0, min(h, y1))
        x2c, y2c = max(0, min(w, x2)), max(0, min(h, y2))
        if x2c <= x1c or y2c <= y1c:
            return
        roi_w, roi_h = x2c - x1c, y2c - y1c

        supersample = ExportService._overlay_supersample_scale(h, ov)
        ss = max(1, supersample)

        # The label overlay (dark box + text) depends only on the text string, not on
        # the underlying video pixels.  Cache it so the expensive supersampled render
        # only runs once per unique text value (i.e. once per active-behavior transition).
        if _cache is not None and text in _cache:
            overlay_small, alpha_small = _cache[text]
        else:
            ss_h, ss_w = roi_h * ss, roi_w * ss
            overlay_ss = np.zeros((ss_h, ss_w, 3), dtype=np.float32)
            alpha_ss = np.zeros((ss_h, ss_w), dtype=np.float32)
            overlay_ss[:] = (20, 20, 20)
            alpha_ss[:] = 1.0

            border_mask = np.zeros((ss_h, ss_w), dtype=np.uint8)
            cv2.rectangle(border_mask, (0, 0), (ss_w - 1, ss_h - 1), 255, max(1, ss), cv2.LINE_AA)
            ExportService._blend_masked_color(overlay_ss, alpha_ss, border_mask, (230, 230, 230), 1.0)

            text_mask = np.zeros((ss_h, ss_w), dtype=np.uint8)
            cv2.putText(
                text_mask,
                text,
                (margin * ss, ss_h - margin * ss),
                font,
                scale * ss,
                255,
                max(1, int(round(thickness * ss))),
                cv2.LINE_AA,
            )
            ExportService._blend_masked_color(overlay_ss, alpha_ss, text_mask, (240, 240, 240), 1.0)

            if ss > 1:
                overlay_small = cv2.resize(overlay_ss, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
                alpha_small = cv2.resize(alpha_ss, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
            else:
                overlay_small = overlay_ss
                alpha_small = alpha_ss

            if _cache is not None:
                _cache[text] = (overlay_small, alpha_small)

        alpha_clip = np.clip(alpha_small, 0.0, 1.0)
        if float(alpha_clip.max(initial=0.0)) <= 0.0:
            return
        roi = frame[y1c:y2c, x1c:x2c].astype(np.float32)
        roi[:] = overlay_small * alpha_clip[..., None] + roi * (1.0 - alpha_clip[..., None])
        frame[y1c:y2c, x1c:x2c] = np.clip(roi, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Advanced overlay helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        if len(h) < 6:
            h = h.ljust(6, "0")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    def _get_behavior_overlay_info(self) -> dict[str, dict[str, Any]]:
        """Return ``{name: {color_bgr, threshold, short_name}}`` for active behaviours."""
        info: dict[str, dict[str, Any]] = {}
        if not self._behaviors:
            return info
        for behavior in self._behaviors.behaviors:
            if not behavior.is_active:
                continue
            name = str(behavior.name or "").strip()
            if not name or self._is_no_behavior_token(name):
                continue
            bid = str(behavior.behavior_id or "").strip()
            color_hex = str(behavior.color or "#4A90E2").strip()
            short = str(behavior.short_name or name).strip()
            cfg = self._temporal_review_settings_for_token(bid or name)
            threshold = float(cfg.get("onset_threshold", 0.65))
            info[name] = {
                "color_bgr": self._hex_to_bgr(color_hex),
                "threshold": threshold,
                "short_name": short,
            }
        return info

    @staticmethod
    def _draw_advanced_overlay(
        frame,
        fps: float,
        behavior_names: list[str],
        active_behaviors: list[str],
        cumulative_frames: dict[str, int],
        behavior_info: dict[str, dict],
        overlay_settings: dict | None = None,
        _buf: dict | None = None,
    ) -> None:
        """Draw the advanced overlay: cumulative-duration panel (bottom-left)
        with prominent active-behavior indicators."""
        import cv2  # noqa: PLC0415

        if not behavior_names:
            return

        ov = overlay_settings or {}

        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_factor = float(ov.get("font_scale_factor", 1.0))
        row_factor = float(ov.get("row_spacing_factor", 1.0))
        dot_offset = int(ov.get("dot_radius_offset", 0))
        panel_opacity = float(ov.get("panel_opacity", 0.72))
        border_gray = int(ov.get("panel_border_gray", 60))
        hl_intensity = float(ov.get("highlight_intensity", 0.30))
        glow_px = int(ov.get("active_glow_ring_px", 4))
        accent_bar_on = bool(ov.get("accent_bar_enabled", True))
        accent_bar_hf = float(ov.get("accent_bar_height_factor", 0.18))
        active_txt = tuple(ov.get("active_text_color", [255, 255, 255]))
        inactive_txt = tuple(ov.get("inactive_text_color", [160, 160, 160]))

        thick_offset = int(ov.get("text_thickness_offset", 0))

        base_scale = max(0.30, min(0.65, h / 1400.0)) * font_factor
        thick = max(1, int(round(h / 700)) + thick_offset)
        margin = max(6, int(h * 0.005))

        row_scale = base_scale * 0.82
        active_scale = base_scale * 0.88
        sample_text = "Ag"
        (_, sample_h), _ = cv2.getTextSize(sample_text, font, row_scale, thick)
        row_h = int(sample_h * 2.4 * row_factor)
        dot_r = max(3, int(sample_h * 0.45) + dot_offset)
        bar_h = max(3, int(row_h * accent_bar_hf))

        # Column widths are cached in _buf so the O(n_behaviors) getTextSize calls
        # only happen once per session rather than once per frame.
        if _buf is not None and '_adv_col_widths' in _buf:
            max_name_w, max_dur_w = _buf['_adv_col_widths']
        else:
            max_name_w = 0
            for name in behavior_names:
                info = behavior_info.get(name, {})
                short = str(info.get("short_name", name))
                # Use active_scale (wider) for a conservative panel width that stays
                # stable across active/inactive transitions.
                (nw, _), _ = cv2.getTextSize(short, font, active_scale, thick)
                max_name_w = max(max_name_w, nw)
            # Use a fixed-width reference string so panel_w stays constant as
            # cumulative durations grow (avoids repeated resizing of preallocated buffers).
            (max_dur_w, _), _ = cv2.getTextSize("9999.9s", font, active_scale, thick)
            if _buf is not None:
                _buf['_adv_col_widths'] = (max_name_w, max_dur_w)

        col_gap = max(6, int(margin * 1.5))
        panel_w = margin + dot_r * 2 + col_gap + max_name_w + col_gap + max_dur_w + margin
        panel_h = len(behavior_names) * row_h + margin * 2

        bx1 = margin
        by1 = h - margin - panel_h
        bx2 = bx1 + panel_w
        by2 = h - margin

        # Semi-transparent dark background
        border_bgr = (border_gray, border_gray, border_gray)
        bx1c, by1c = max(0, bx1), max(0, by1)
        bx2c, by2c = min(w, bx2), min(h, by2)
        if bx2c <= bx1c or by2c <= by1c:
            return

        active_set = set(active_behaviors)
        supersample = ExportService._overlay_supersample_scale(h, ov)

        # Pre-allocate a pool of uint8 mask arrays sized to the supersampled panel
        # dimensions.  The pool grows lazily and is reused across every frame call,
        # eliminating the ~15-25 np.zeros allocations that previously occurred per frame.
        _mask_ss_h = (by2c - by1c) * max(1, supersample)
        _mask_ss_w = (bx2c - bx1c) * max(1, supersample)
        _mask_pool: list[np.ndarray] = _buf.get('_adv_masks', []) if _buf is not None else []
        _mask_pool_idx = [0]

        def _alloc_mask() -> np.ndarray:
            idx = _mask_pool_idx[0]
            _mask_pool_idx[0] += 1
            if idx >= len(_mask_pool):
                _mask_pool.append(np.zeros((_mask_ss_h, _mask_ss_w), dtype=np.uint8))
            m = _mask_pool[idx]
            m.fill(0)
            return m

        if _buf is not None:
            _buf['_adv_masks'] = _mask_pool

        def _draw(overlay: np.ndarray, alpha: np.ndarray, ss: int) -> None:
            roi_h, roi_w = overlay.shape[:2]
            overlay[:] = (18, 18, 18)
            alpha[:] = max(0.0, min(1.0, panel_opacity))

            border_mask = _alloc_mask()
            cv2.rectangle(border_mask, (0, 0), (roi_w - 1, roi_h - 1), 255, max(1, ss), cv2.LINE_AA)
            ExportService._blend_masked_color(overlay, alpha, border_mask, border_bgr, 1.0)

            for idx, name in enumerate(behavior_names):
                info = behavior_info.get(name, {})
                short = str(info.get("short_name", name))
                color_bgr = info.get("color_bgr", (200, 200, 200))
                is_active = name in active_set

                row_top = margin + idx * row_h
                row_cy = row_top + row_h // 2

                if is_active:
                    tint_mask = _alloc_mask()
                    cv2.rectangle(
                        tint_mask,
                        (max(0, 2 * ss), max(0, row_top * ss)),
                        (max(0, roi_w - 2 * ss - 1), min(roi_h - 1, (row_top + row_h) * ss - 1)),
                        255,
                        -1,
                        cv2.LINE_AA,
                    )
                    ExportService._blend_masked_color(overlay, alpha, tint_mask, color_bgr, hl_intensity)

                dot_x = margin + dot_r
                dot_cx = int(round(dot_x * ss))
                dot_cy = int(round(row_cy * ss))
                if is_active:
                    glow_thick = max(2, glow_px // 2)
                    step = max(1, glow_px // 3)
                    for g_off in range(glow_px, 0, -step):
                        glow_mask = _alloc_mask()
                        cv2.circle(
                            glow_mask,
                            (dot_cx, dot_cy),
                            max(1, int(round((dot_r + g_off) * ss))),
                            255,
                            max(1, int(round(glow_thick * ss))),
                            cv2.LINE_AA,
                        )
                        glow_alpha = 0.25 + 0.35 * (1.0 - g_off / max(1, glow_px))
                        ExportService._blend_masked_color(overlay, alpha, glow_mask, color_bgr, glow_alpha)

                    dot_mask = _alloc_mask()
                    cv2.circle(
                        dot_mask,
                        (dot_cx, dot_cy),
                        max(1, int(round((dot_r + 1) * ss))),
                        255,
                        -1,
                        cv2.LINE_AA,
                    )
                    ExportService._blend_masked_color(overlay, alpha, dot_mask, color_bgr, 1.0)

                    outline_mask = _alloc_mask()
                    cv2.circle(
                        outline_mask,
                        (dot_cx, dot_cy),
                        max(1, int(round((dot_r + 1) * ss))),
                        255,
                        max(1, ss),
                        cv2.LINE_AA,
                    )
                    ExportService._blend_masked_color(overlay, alpha, outline_mask, (255, 255, 255), 1.0)
                else:
                    dot_mask = _alloc_mask()
                    cv2.circle(
                        dot_mask,
                        (dot_cx, dot_cy),
                        max(1, int(round(dot_r * ss))),
                        255,
                        -1,
                        cv2.LINE_AA,
                    )
                    ExportService._blend_masked_color(overlay, alpha, dot_mask, color_bgr, 1.0)

                cur_scale = active_scale if is_active else row_scale
                cur_thick = thick + 1 if is_active else thick
                text_y = row_cy + sample_h // 2
                text_color = active_txt if is_active else inactive_txt
                name_x = dot_x + dot_r + col_gap

                name_mask = _alloc_mask()
                cv2.putText(
                    name_mask,
                    short,
                    (int(round(name_x * ss)), int(round(text_y * ss))),
                    font,
                    cur_scale * ss,
                    255,
                    max(1, int(round(cur_thick * ss))),
                    cv2.LINE_AA,
                )
                ExportService._blend_masked_color(overlay, alpha, name_mask, text_color, 1.0)

                dur_sec = cumulative_frames.get(name, 0) / max(1.0, fps)
                dur_text = f"{dur_sec:.1f}s"
                dur_x = panel_w - margin - max_dur_w
                dur_mask = _alloc_mask()
                cv2.putText(
                    dur_mask,
                    dur_text,
                    (int(round(dur_x * ss)), int(round(text_y * ss))),
                    font,
                    cur_scale * ss,
                    255,
                    max(1, int(round(cur_thick * ss))),
                    cv2.LINE_AA,
                )
                ExportService._blend_masked_color(overlay, alpha, dur_mask, text_color, 1.0)

                if is_active and accent_bar_on:
                    bar_y = row_top + row_h - bar_h - 1
                    bar_mask = _alloc_mask()
                    cv2.rectangle(
                        bar_mask,
                        (int(round(margin * ss)), int(round(bar_y * ss))),
                        (int(round((panel_w - margin) * ss)) - 1, int(round((bar_y + bar_h) * ss)) - 1),
                        255,
                        -1,
                        cv2.LINE_AA,
                    )
                    ExportService._blend_masked_color(overlay, alpha, bar_mask, color_bgr, 1.0)

        ExportService._composite_supersampled_roi(frame, bx1c, by1c, bx2c, by2c, supersample, _draw, _buf=_buf)

    @staticmethod
    def _confirmed_candidates(
        candidates: list[CandidateWindow],
        decisions: list[ReviewDecision],
    ) -> list[CandidateWindow]:
        accepted_ids = {
            d.clip_id
            for d in decisions
            if d.decision == ReviewDecisionType.ACCEPT
        }
        return [c for c in candidates if c.window_id in accepted_ids]
