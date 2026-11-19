"""Per-subject section boundaries read from an event-time file.

Session Sections normally cuts every session into the same fixed-length
sections.  When stimuli were given by hand the timing drifts between
subjects, so each subject needs its own boundaries.  An event file holds
them: one row per subject and a ``<Section> Start`` / ``<Section> End``
column pair per section, in video frames counted from the first frame of
the video.

Blank cells borrow from their neighbour: a blank start is the previous
section's end (0 for the first section) and a blank end is the next
section's start (the end of the session for the last section).  A section
that still has no start or end is missing for that subject and is left out
of its numbers, never counted as zero.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SECTION_EVENTS_SHEET = "Section Events"

_BOUNDARY_RE = re.compile(r"^\s*(.+?)\s+(start|end)\s*$", re.IGNORECASE)
_SUBJECT_HEADERS = ("subject", "subject id", "subject_id", "animal", "mouse", "session")

Frame = float | None


def _to_frame(value: object) -> Frame:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


@dataclass
class SectionEventTable:
    """Section names and each subject's (start_frame, end_frame) per section."""

    section_names: list[str]
    subjects: dict[str, list[tuple[Frame, Frame]]] = field(default_factory=dict)
    source: str = ""

    # ── Persistence ───────────────────────────────────────────────────
    def to_state(self) -> dict:
        return {
            "source": self.source,
            "section_names": list(self.section_names),
            "subjects": {
                subj: [[s, e] for s, e in rows] for subj, rows in self.subjects.items()
            },
        }

    @classmethod
    def from_state(cls, state: dict | None) -> "SectionEventTable | None":
        if not isinstance(state, dict):
            return None
        names = [str(n) for n in (state.get("section_names") or [])]
        if not names:
            return None
        subjects: dict[str, list[tuple[Frame, Frame]]] = {}
        for subj, rows in (state.get("subjects") or {}).items():
            pairs: list[tuple[Frame, Frame]] = []
            for pair in list(rows or [])[: len(names)]:
                s, e = (list(pair) + [None, None])[:2]
                pairs.append((_to_frame(s), _to_frame(e)))
            pairs += [(None, None)] * (len(names) - len(pairs))
            subjects[str(subj)] = pairs
        return cls(names, subjects, str(state.get("source") or ""))

    # ── Lookup ────────────────────────────────────────────────────────
    def match_subject(self, *candidates: object) -> str | None:
        """Return the file's subject key for the first candidate name that matches.

        Exact match first, then case- and whitespace-insensitive.
        """
        for cand in candidates:
            if cand is not None and str(cand) in self.subjects:
                return str(cand)
        folded = {k.strip().casefold(): k for k in self.subjects}
        for cand in candidates:
            if cand is None:
                continue
            key = folded.get(str(cand).strip().casefold())
            if key is not None:
                return key
        return None

    def bounds_seconds(
        self,
        subject: str,
        fps: float,
        *,
        prechop_frames: float = 0.0,
        session_end_s: float | None = None,
    ) -> list[tuple[float, float] | None]:
        """Section bounds in seconds on the analysis clock for one subject.

        The analysis clock starts at the subject's prechop, the same clock
        Analytics bouts are rebased to.  Returns one entry per section, None
        where the section cannot be placed for this subject.
        """
        rows = self.subjects.get(subject)
        if rows is None or fps <= 0:
            return [None] * len(self.section_names)
        n = len(rows)
        out: list[tuple[float, float] | None] = []
        for i, (start, end) in enumerate(rows):
            if start is None:
                start = rows[i - 1][1] if i > 0 else 0.0
            if end is None:
                if i + 1 < n:
                    end = rows[i + 1][0]
                elif session_end_s is not None:
                    end = session_end_s * fps + prechop_frames
            if start is None or end is None:
                out.append(None)
                continue
            s0 = (float(start) - prechop_frames) / fps
            s1 = (float(end) - prechop_frames) / fps
            s0 = max(0.0, s0)
            if session_end_s is not None:
                s1 = min(float(session_end_s), s1)
            out.append((s0, s1) if s1 > s0 else None)
        return out

    def problems(self) -> list[str]:
        """Plain-language list of rows whose times run backwards."""
        issues: list[str] = []
        for subj, rows in self.subjects.items():
            prev_end: Frame = None
            for name, (s, e) in zip(self.section_names, rows):
                if s is not None and e is not None and e < s:
                    issues.append(f"{subj}: {name} ends before it starts")
                if prev_end is not None and s is not None and s < prev_end:
                    issues.append(f"{subj}: {name} starts before the previous section ends")
                prev_end = e if e is not None else prev_end
        return issues


def parse_section_events(df: pd.DataFrame, source: str = "") -> SectionEventTable:
    """Build a table from a DataFrame laid out as described in the module doc."""
    if df is None or df.empty:
        raise ValueError("The event file has no rows.")
    cols = [str(c) for c in df.columns]
    subj_col = next((c for c in cols if c.strip().casefold() in _SUBJECT_HEADERS), cols[0])

    names: list[str] = []
    start_col: dict[str, str] = {}
    end_col: dict[str, str] = {}
    for c in cols:
        if c == subj_col:
            continue
        m = _BOUNDARY_RE.match(c)
        if not m:
            continue
        name, which = m.group(1).strip(), m.group(2).casefold()
        if name not in names:
            names.append(name)
        (start_col if which == "start" else end_col)[name] = c
    if not names:
        raise ValueError(
            'No section columns found. Name them "<Section> Start" and "<Section> End", '
            'for example "Home Cage Start" and "Home Cage End".'
        )

    df = df.copy()
    df.columns = cols
    subjects: dict[str, list[tuple[Frame, Frame]]] = {}
    for _, row in df.iterrows():
        subj_raw = row[subj_col]
        if subj_raw is None or (isinstance(subj_raw, float) and math.isnan(subj_raw)):
            continue
        subj = str(subj_raw).strip()
        if not subj:
            continue
        subjects[subj] = [
            (
                _to_frame(row[start_col[n]]) if n in start_col else None,
                _to_frame(row[end_col[n]]) if n in end_col else None,
            )
            for n in names
        ]
    if not subjects:
        raise ValueError("The event file has no subject rows.")
    return SectionEventTable(names, subjects, source)


def load_section_events(path: str | Path) -> SectionEventTable:
    """Read an event file (.xlsx/.xls/.csv/.tsv).

    In a workbook the "Section Events" sheet is used when present, otherwise
    the first sheet.
    """
    p = Path(path)
    suffix = p.suffix.casefold()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        with pd.ExcelFile(p) as xls:
            sheet = (
                SECTION_EVENTS_SHEET
                if SECTION_EVENTS_SHEET in xls.sheet_names
                else xls.sheet_names[0]
            )
            df = pd.read_excel(xls, sheet)
    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(p, sep="\t")
    else:
        df = pd.read_csv(p)
    return parse_section_events(df, source=str(p))


TEMPLATE_INSTRUCTIONS = [
    "How to fill in this event file",
    "",
    "One row per subject. The Subject cell must match the subject (or session) name in ABEL.",
    'Each section has two columns, "<Section> Start" and "<Section> End".',
    "Times are video frames counted from the first frame of the video, not seconds.",
    "Rename, add or remove section column pairs as needed. Sections run in column order.",
    "",
    "Blank cells borrow from their neighbour:",
    "  a blank Start uses the previous section's End (the first section starts at frame 0);",
    "  a blank End uses the next section's Start (the last section runs to the end of the video).",
    "A subject whose section still has no start or end is left out of that section, never counted as zero.",
    "",
    "Columns that are not a Start/End pair (notes, cohort, and so on) are ignored.",
    'ABEL reads the "Section Events" sheet. This sheet is ignored.',
]


def write_section_events_template(
    path: str | Path,
    section_names: Iterable[str],
    subjects: Iterable[str] = (),
) -> Path:
    """Write a blank event file with one row per subject and Start/End columns.

    With no subjects a single example row is written.  A .csv path gets the
    table only; any other path is written as a workbook with an
    instructions sheet.
    """
    p = Path(path)
    names = [str(n).strip() for n in section_names if str(n).strip()]
    if not names:
        names = ["Baseline", "Stimulus", "Post"]
    subj = [str(s) for s in subjects if str(s).strip()]
    cols = ["Subject"] + [f"{n} {w}" for n in names for w in ("Start", "End")] + ["Notes"]
    if subj:
        df = pd.DataFrame({"Subject": subj}).reindex(columns=cols)
    else:
        row: dict[str, object] = {"Subject": "Example01", "Notes": "Example row, replace it"}
        for i, n in enumerate(names):
            row[f"{n} Start"] = 0 if i == 0 else None
            row[f"{n} End"] = (i + 1) * 9000 if i + 1 < len(names) else None
        df = pd.DataFrame([row]).reindex(columns=cols)
    if p.suffix.casefold() == ".csv":
        df.to_csv(p, index=False)
        return p
    if p.suffix.casefold() not in {".xlsx", ".xlsm"}:
        p = p.with_suffix(".xlsx")
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=SECTION_EVENTS_SHEET, index=False)
        pd.DataFrame({"Instructions": TEMPLATE_INSTRUCTIONS}).to_excel(
            w, sheet_name="Instructions", index=False,
        )
        ws = w.sheets[SECTION_EVENTS_SHEET]
        for i, c in enumerate(cols, start=1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = max(12, len(c) + 2)
        w.sheets["Instructions"].column_dimensions["A"].width = 100
    return p


def section_metrics(
    bout_start_s: np.ndarray,
    bout_end_s: np.ndarray,
    s0: float,
    s1: float,
) -> tuple[int, float, float, float]:
    """(bout count, seconds of behavior, % of section, bouts per minute) in [s0, s1).

    A bout is counted in every section it overlaps and contributes only its
    overlapping seconds, the same rule the fixed-duration sections use.
    """
    overlap = np.maximum(
        0.0, np.minimum(bout_end_s, s1) - np.maximum(bout_start_s, s0)
    )
    n = int((overlap > 0.0).sum())
    dur = float(overlap.sum())
    length = s1 - s0
    pct = dur / length * 100.0 if length > 0 else 0.0
    rate = n / (length / 60.0) if length > 0 else 0.0
    return n, dur, pct, rate


def mean_section_durations(
    table: SectionEventTable,
    subjects: Iterable[str],
    fps_by_subject: dict[str, float],
    session_end_s_by_subject: dict[str, float] | None = None,
) -> list[float]:
    """Mean length in seconds of each section across the given subjects.

    Without a session end an open-ended last section has no length and is
    skipped for that subject.
    """
    ends = session_end_s_by_subject or {}
    sums = [0.0] * len(table.section_names)
    counts = [0] * len(table.section_names)
    for subj in subjects:
        fps = float(fps_by_subject.get(subj, 0.0) or 0.0)
        if fps <= 0:
            continue
        for i, b in enumerate(table.bounds_seconds(subj, fps, session_end_s=ends.get(subj))):
            if b is not None:
                sums[i] += b[1] - b[0]
                counts[i] += 1
    return [s / c if c else 0.0 for s, c in zip(sums, counts)]
