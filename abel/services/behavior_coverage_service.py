"""Per-subject label coverage: which animals still lack examples of a behavior.

Leave-one-subject-out CV can only score a mouse that has positives of the
behavior, and a model trained on a handful of animals learns those animals.
Targeted Clip Mining uses this service to aim a hunt at the subjects that are
missing examples, instead of letting the animals that already score well fill
every batch.

Counting follows the LOSO "mice" column: the reviewer's own labels
(``reviewer_labels.parquet``, which never holds temporal-feedback or imported
refinement rows), grouped by the session's current subject so the two tracks of
a dyad count as that session's subject. A co-occurring label (``a|b``) is a
positive for each part, as the trainer reads it.

Some subjects never do the behavior (the subordinate animal of a dyad never
dominates), so mining for them would go on forever. Every window loaded by a
gap hunt is logged per behavior; once enough of a subject's gap-mined windows
have been reviewed without a single hit, that subject is reported as likely
absent and drops out of the gap set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from abel.storage.file_store import read_json, write_json

logger = logging.getLogger(__name__)

DEFAULT_MIN_POSITIVES = 3
DEFAULT_ABSENT_AFTER = 30


@dataclass
class SubjectCoverage:
    """One subject's standing for one behavior."""

    subject: str
    positives: int = 0
    screened: int = 0  # gap-mined windows for this behavior that have been reviewed
    hits: int = 0      # of those, how many were labeled this behavior


@dataclass
class CoverageReport:
    """Coverage of one behavior across every subject in the project."""

    behavior_id: str
    min_positives: int
    absent_after: int
    subjects: dict[str, SubjectCoverage] = field(default_factory=dict)
    sessions_by_subject: dict[str, set[str]] = field(default_factory=dict)

    def is_likely_absent(self, subject: str) -> bool:
        """No positive anywhere, and ``absent_after`` gap-mined windows reviewed with no hit.

        A single positive (on either track of a dyad) means the subject does the
        behavior, so it stays in the gap set however many misses it has.
        """
        cov = self.subjects.get(subject)
        return bool(
            cov is not None
            and self.absent_after > 0
            and cov.positives == 0
            and cov.screened >= self.absent_after
            and cov.hits == 0
        )

    def likely_absent(self) -> set[str]:
        return {s for s in self.subjects if self.is_likely_absent(s)}

    def gap_subjects(self) -> set[str]:
        """Subjects below the positive threshold that are still worth mining."""
        return {
            s for s, cov in self.subjects.items()
            if cov.positives < self.min_positives and not self.is_likely_absent(s)
        }

    def gap_sessions(self) -> set[str]:
        out: set[str] = set()
        for s in self.gap_subjects():
            out |= self.sessions_by_subject.get(s, set())
        return out

    def n_covered(self) -> int:
        return sum(1 for c in self.subjects.values() if c.positives >= self.min_positives)


class BehaviorCoverageService:
    """Reads label coverage per subject and logs gap-mined windows."""

    def __init__(self, project_root: Path) -> None:
        self._root = Path(project_root)
        self._session_by_segment: dict[str, str] | None = None
        self._subject_by_session: dict[str, str] | None = None

    # -- paths ---------------------------------------------------------------

    def _labels_path(self) -> Path:
        return self._root / "derived" / "review_labels" / "reviewer_labels.parquet"

    def _decisions_path(self) -> Path:
        return self._root / "derived" / "review_tables" / "review_decisions.json"

    def _log_path(self) -> Path:
        return self._root / "derived" / "review_tables" / "coverage_gap_log.json"

    # -- identity maps -------------------------------------------------------

    def subject_by_session(self) -> dict[str, str]:
        """Each manifest session's current subject, as LOSO groups them."""
        if self._subject_by_session is None:
            from abel.services.import_service import ImportService  # noqa: PLC0415

            out: dict[str, str] = {}
            try:
                manifest = ImportService().load_manifest(self._root)
            except Exception:
                logger.debug("Coverage: manifest unreadable", exc_info=True)
                manifest = None
            if manifest is not None:
                for s in manifest.linked_sessions:
                    subj = str(s.subject_id or s.subject_key or s.session_id)
                    out[str(s.session_id)] = subj
            self._subject_by_session = out
        return self._subject_by_session

    def _segment_sessions(self) -> dict[str, str]:
        """segment_id -> session_id over every table a reviewed window can live in.

        Reviewer labels carry no session column. The grid pool covers most of
        them; reviewed off-grid windows are only in the training set.
        """
        if self._session_by_segment is None:
            out: dict[str, str] = {}
            for path in (
                self._root / "derived" / "training_sets" / "training_set.parquet",
                self._root / "derived" / "representations" / "segment_features.parquet",
            ):
                if not path.exists():
                    continue
                try:
                    df = pd.read_parquet(path, columns=["segment_id", "session_id"])
                except Exception:
                    logger.debug("Coverage: could not read %s", path, exc_info=True)
                    continue
                out.update(zip(df["segment_id"].astype(str), df["session_id"].astype(str)))
            self._session_by_segment = out
        return self._session_by_segment

    def register_windows(self, session_by_window: dict[str, str]) -> None:
        """Add window -> session pairs the caller already knows (e.g. the mined pool)."""
        self._segment_sessions().update(
            {str(k): str(v) for k, v in session_by_window.items()}
        )

    # -- reads ---------------------------------------------------------------

    def _labels(self) -> pd.DataFrame:
        path = self._labels_path()
        if not path.exists():
            return pd.DataFrame(columns=["segment_id", "review_label"])
        try:
            return pd.read_parquet(path, columns=["segment_id", "review_label"])
        except Exception:
            logger.warning("Coverage: could not read reviewer labels", exc_info=True)
            return pd.DataFrame(columns=["segment_id", "review_label"])

    def _decided_ids(self) -> set[str]:
        raw = read_json(self._decisions_path(), {"decisions": []})
        return {str(d.get("clip_id")) for d in raw.get("decisions", []) if d.get("clip_id")}

    def _load_log(self) -> dict[str, dict[str, str]]:
        """behavior_id -> {window_id: session_id} for every gap-mined window."""
        raw = read_json(self._log_path(), {"behaviors": {}})
        beh = raw.get("behaviors", {}) if isinstance(raw, dict) else {}
        return beh if isinstance(beh, dict) else {}

    @staticmethod
    def _label_sets(labels: pd.DataFrame) -> dict[str, set[str]]:
        """segment_id -> the behavior ids its reviewer labels name."""
        out: dict[str, set[str]] = {}
        for sid, lab in zip(labels["segment_id"].astype(str), labels["review_label"].astype(str)):
            parts = {p.strip() for p in lab.split("|") if p.strip()}
            out.setdefault(sid, set()).update(parts)
        return out

    def dominant_behavior(self, window_ids: list[str]) -> str | None:
        """The behavior most of these windows are labeled with (for a default pick)."""
        want = {str(w) for w in window_ids}
        if not want:
            return None
        counts: dict[str, int] = {}
        for sid, parts in self._label_sets(self._labels()).items():
            if sid in want:
                for p in parts:
                    if p != "no_behavior":
                        counts[p] = counts.get(p, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def report(
        self,
        behavior_id: str,
        min_positives: int = DEFAULT_MIN_POSITIVES,
        absent_after: int = DEFAULT_ABSENT_AFTER,
    ) -> CoverageReport:
        target = str(behavior_id)
        subj_of = self.subject_by_session()
        seg_sess = self._segment_sessions()
        rep = CoverageReport(target, int(min_positives), int(absent_after))
        for sess, subj in subj_of.items():
            rep.subjects.setdefault(subj, SubjectCoverage(subj))
            rep.sessions_by_subject.setdefault(subj, set()).add(sess)

        def _subject_of_window(wid: str) -> str | None:
            sess = seg_sess.get(wid)
            return subj_of.get(sess) if sess is not None else None

        label_sets = self._label_sets(self._labels())
        unmapped = 0
        for sid, parts in label_sets.items():
            if target not in parts:
                continue
            subj = _subject_of_window(sid)
            if subj is None:
                unmapped += 1
                continue
            rep.subjects[subj].positives += 1
        if unmapped:
            logger.debug("Coverage: %d positive(s) of %s not mapped to a subject", unmapped, target)

        reviewed = set(label_sets) | self._decided_ids()
        for wid, sess in self._load_log().get(target, {}).items():
            subj = subj_of.get(str(sess))
            if wid not in reviewed or subj is None:
                continue
            rep.subjects[subj].screened += 1
            if target in label_sets.get(wid, set()):
                rep.subjects[subj].hits += 1
        return rep

    # -- writes --------------------------------------------------------------

    def record_mined(self, behavior_id: str, window_ids: list[str]) -> int:
        """Log windows a gap hunt loaded, so later reviews count as screening.

        Stored per window as its session, not its subject, so a subject rename
        keeps the history.
        """
        seg_sess = self._segment_sessions()
        log = self._load_log()
        entry = log.setdefault(str(behavior_id), {})
        n = 0
        for wid in window_ids:
            sess = seg_sess.get(str(wid))
            if sess is None or str(wid) in entry:
                continue
            entry[str(wid)] = sess
            n += 1
        if n:
            write_json(self._log_path(), {"behaviors": log})
        return n
