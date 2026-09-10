"""Keep project state attached to its sessions when subjects are renamed.

Subject names are display labels the Data Import tab lets the user change at any
time — by re-applying the filename regex or by typing over a cell.  Some saved
state is keyed by those names rather than by session id:

* ``derived/analytics_groups.json`` — factor assignments and the session order
  are keyed by the Analytics session label (``m1`` or ``m1 – cond1``), the
  per-subject prechop by subject name;
* ``config/environment_rois.yaml`` — per-subject ROIs are keyed by subject name
  or ``subject::session_id``.

:func:`propagate_subject_renames` re-keys both whenever a manifest save changes
a subject.  The Analytics tab also holds its state in memory and may save it
later, so that state carries *anchors* — the session ids behind every key — and
:func:`remap_group_state` re-keys it on load and refresh too.

Segment and clip ids are protected separately: extraction writes the frozen
``LinkedSession.subject_key`` as ``animal_id``, so derived tables never see a
rename.  Anything that groups rows *by subject* (subject-level CV splits, LOSO)
must therefore resolve the current name through :func:`add_subject_groups`
rather than read ``animal_id`` directly.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger("abel")

ANCHORS_KEY = "session_anchors"
GROUP_STATE_FILE = Path("derived") / "analytics_groups.json"
SUBJECT_GROUP_COL = "subject_group"


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _invert(mapping: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for sid, key in mapping.items():
        out.setdefault(key, []).append(sid)
    return out


@dataclass(frozen=True)
class SessionLabels:
    """How the Analytics tab names each session, derived from the manifest."""

    subject_by_session: dict[str, str] = field(default_factory=dict)
    session_type_by_session: dict[str, str] = field(default_factory=dict)
    label_by_session: dict[str, str] = field(default_factory=dict)

    def sessions_by_label(self) -> dict[str, list[str]]:
        return _invert(self.label_by_session)

    def sessions_by_subject(self) -> dict[str, list[str]]:
        return _invert(self.subject_by_session)


def session_labels(manifest: Any) -> SessionLabels:
    """Subject, session type and display label of every linked session.

    A subject with one session is labelled by its name alone; a subject with
    several is labelled ``"{subject} – {session type}"`` per session.
    """
    if manifest is None:
        return SessionLabels()
    video_by_id = {v.asset_id: v for v in manifest.videos}
    subjects: dict[str, str] = {}
    types: dict[str, str] = {}
    for session in manifest.linked_sessions:
        sid = str(session.session_id)
        video = video_by_id.get(session.video_asset_id)
        subject = (session.subject_id or "").strip()
        if not subject and video is not None:
            subject = (video.subject_id or "").strip()
        subject = subject or sid
        subjects[sid] = subject
        # Session type from the video filename first — the stem after the
        # subject prefix, minus any DLC suffix ("m10_cond1" with subject "m10"
        # → "cond1") — then from a "{subject}_{session type}" subject label.
        stype = ""
        if video is not None:
            stem = Path(video.source_path).stem
            if stem.startswith(subject):
                remainder = stem[len(subject):].lstrip("_- ")
                if remainder and not remainder.upper().startswith("DLC"):
                    stype = remainder
        if not stype and "_" in subject:
            stype = subject.split("_", 1)[1]
        types[sid] = stype
    per_subject = Counter(subjects.values())
    labels = {
        sid: f"{subj} – {types[sid]}" if per_subject[subj] > 1 and types[sid] else subj
        for sid, subj in subjects.items()
    }
    return SessionLabels(subjects, types, labels)


# ----------------------------------------------------------------------
# Grouping derived rows by subject
# ----------------------------------------------------------------------


def current_subjects(df: pd.DataFrame, manifest: Any) -> pd.Series:
    """Each row's current subject name.

    A row whose ``(animal_id, session_id)`` is a manifest session's
    ``(subject_key, session_id)`` gets that session's ``subject_id``.  Every
    other row — multi-animal individuals, rows imported from other projects,
    sessions no longer in the manifest — keeps its ``animal_id``.
    """
    animal = df["animal_id"].astype(str)
    if manifest is None or "session_id" not in df.columns:
        return animal
    name_of = {
        (str(s.subject_key), str(s.session_id)): str(s.subject_id or s.subject_key)
        for s in manifest.linked_sessions
    }
    return pd.Series(
        [name_of.get(k, k[0]) for k in zip(animal, df["session_id"].astype(str))],
        index=df.index,
        dtype=object,
    )


def add_subject_groups(df: pd.DataFrame, project_root: Path | None) -> pd.DataFrame:
    """Set ``df[SUBJECT_GROUP_COL]`` to each row's current subject; returns *df*.

    Recomputed on every call, so a value left over from an earlier rename is
    never trusted.  A string column, so the trainer never takes it for a feature.
    """
    if "animal_id" not in df.columns:
        return df
    manifest = None
    if project_root is not None:
        from abel.services.import_service import ImportService  # noqa: PLC0415

        manifest = ImportService().load_manifest(Path(project_root))
    df[SUBJECT_GROUP_COL] = current_subjects(df, manifest)
    return df


# ----------------------------------------------------------------------
# Analytics group state
# ----------------------------------------------------------------------


def _anchor_keys(
    keys: Iterable[str], current: dict[str, list[str]], previous: dict[str, Any]
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in keys:
        if key in current:
            out[key] = sorted(current[key])
        elif previous.get(key):
            out[key] = [str(s) for s in previous[key]]
    return out


def anchor_group_state(state: dict, labels: SessionLabels) -> dict:
    """Record, for every label- and subject-keyed entry, the sessions it covers.

    A key *labels* does not know keeps the anchor it already had, so an entry
    left behind by an earlier rename can still be followed later.
    """
    previous = state.get(ANCHORS_KEY) or {}
    label_keys = _unique(
        [*(state.get("session_factors") or {}), *(state.get("subject_order") or [])]
    )
    state[ANCHORS_KEY] = {
        "labels": _anchor_keys(
            label_keys, labels.sessions_by_label(), previous.get("labels") or {}
        ),
        "subjects": _anchor_keys(
            state.get("subject_prechop_frames") or {},
            labels.sessions_by_subject(),
            previous.get("subjects") or {},
        ),
    }
    return state


def remap_group_state(state: dict, labels: SessionLabels) -> list[str]:
    """Move label- and subject-keyed entries of *state* onto current keys.

    A key that is no longer current follows its anchored sessions to the
    label (or subject) they have now — to several keys when a subject was
    split.  Entries already under a current key win over entries moved onto
    it.  A key without an anchor, or whose sessions are all gone, stays as it
    is.  Mutates *state*, re-anchors it against *labels*, and returns a note
    for every value that could not be carried over.
    """
    anchors = state.get(ANCHORS_KEY) or {}
    notes: list[str] = []

    def targets(key: str, anchor: dict, key_of: dict[str, str], current: set[str]) -> list[str]:
        if key in current:
            return [key]
        moved = _unique(key_of[s] for s in (anchor.get(key) or []) if s in key_of)
        return moved or [key]

    label_of = labels.label_by_session
    current_labels = set(label_of.values())
    label_anchor = anchors.get("labels") or {}

    factors = state.get("session_factors") or {}
    if factors:
        merged: dict[str, dict[str, str]] = {}
        # Current keys first, so their assignments win any collision.
        for key in sorted(factors, key=lambda k: k not in current_labels):
            for target in targets(key, label_anchor, label_of, current_labels):
                slot = merged.setdefault(target, {})
                for factor, level in (factors[key] or {}).items():
                    if not level:
                        continue
                    have = slot.get(factor)
                    if not have:
                        slot[factor] = level
                    elif have != level:
                        notes.append(
                            f"{factor} for '{target}' kept '{have}'; "
                            f"'{level}' from '{key}' was dropped."
                        )
        state["session_factors"] = merged

    order = state.get("subject_order") or []
    if order:
        state["subject_order"] = _unique(
            t for key in order for t in targets(key, label_anchor, label_of, current_labels)
        )

    subject_of = labels.subject_by_session
    current_subjects = set(subject_of.values())
    subject_anchor = anchors.get("subjects") or {}
    prechop = state.get("subject_prechop_frames") or {}
    if prechop:
        kept = {k: v for k, v in prechop.items() if k in current_subjects}
        proposed: dict[str, dict[str, Any]] = {}
        for key, value in prechop.items():
            if key in kept:
                continue
            for target in targets(key, subject_anchor, subject_of, current_subjects):
                proposed.setdefault(target, {})[key] = value
        for target, by_old in proposed.items():
            values = set(by_old.values())
            if target in kept:
                if values - {kept[target]}:
                    notes.append(
                        f"Prechop for '{target}' kept {kept[target]} frames; "
                        f"{', '.join(f'{k}={v}' for k, v in by_old.items())} dropped."
                    )
            elif len(values) == 1:
                kept[target] = values.pop()
            else:
                notes.append(
                    f"Prechop for '{target}' not carried over — its sessions had "
                    f"different values ({', '.join(f'{k}={v}' for k, v in by_old.items())}). "
                    "Set it again in Analytics."
                )
        state["subject_prechop_frames"] = kept

    anchor_group_state(state, labels)
    return notes


# ----------------------------------------------------------------------
# Per-subject ROIs
# ----------------------------------------------------------------------


def remap_subject_rois(
    subject_rois: dict[str, Any],
    old_subjects: dict[str, str],
    new_subjects: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    """Re-key per-subject ROI blocks from old to new subject names.

    ``subject::session_id`` keys follow their session.  A plain subject key
    moves to the new name when every session under that name came from the old
    subject (a rename or a split); when the new name gathers sessions from
    other subjects too, each of the old subject's sessions gets its own
    ``new::session_id`` entry, so every recording keeps exactly the ROI it had.
    """
    old_sessions = _invert(old_subjects)
    new_sessions = _invert(new_subjects)
    out: dict[str, Any] = {}
    notes: list[str] = []

    def put(key: str, block: Any, source: str) -> None:
        if key in out and out[key] != block:
            notes.append(f"ROI '{key}' kept; the one from '{source}' was dropped.")
            return
        out[key] = block

    composite: list[tuple[str, Any]] = []
    for key, block in subject_rois.items():
        if "::" in key:
            composite.append((key, block))
            continue
        sessions = old_sessions.get(key)
        if not sessions:
            put(key, block, key)
            continue
        for new_name in _unique(new_subjects[s] for s in sessions if s in new_subjects):
            if set(new_sessions.get(new_name, [])) <= set(sessions):
                put(new_name, block, key)
            else:
                for sid in sessions:
                    if new_subjects.get(sid) == new_name:
                        put(f"{new_name}::{sid}", block, key)
    # Explicit per-session ROIs are the most specific; they override anything
    # a plain key fanned out onto the same session.
    for key, block in composite:
        subject, _, sid = key.partition("::")
        if sid in new_subjects and old_subjects.get(sid) == subject:
            out[f"{new_subjects[sid]}::{sid}"] = block
        else:
            out.setdefault(key, block)
    return out, notes


# ----------------------------------------------------------------------
# Manifest-save hook
# ----------------------------------------------------------------------


def _labels_changed(old: SessionLabels, new: SessionLabels) -> bool:
    common = old.label_by_session.keys() & new.label_by_session.keys()
    return any(
        old.label_by_session[s] != new.label_by_session[s]
        or old.subject_by_session[s] != new.subject_by_session[s]
        for s in common
    )


def _backup(path: Path) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.stem}.backup-{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    logger.info("Kept pre-rename copy of %s at %s", path.name, backup)


def propagate_subject_renames(project_root: Path, old_manifest: Any, new_manifest: Any) -> list[str]:
    """Re-key subject-keyed project state after a manifest changes subjects.

    Rewrites ``analytics_groups.json`` and ``environment_rois.yaml`` in place
    (each backed up first) so factor assignments, session order, prechop and
    per-subject ROIs stay with their sessions.  A no-op when no existing
    session changed subject or Analytics label.  Returns human-readable notes
    about what moved and about anything that could not be carried over.
    """
    old = session_labels(old_manifest)
    new = session_labels(new_manifest)
    if not _labels_changed(old, new):
        return []
    notes: list[str] = []

    groups_path = project_root / GROUP_STATE_FILE
    if groups_path.exists():
        try:
            state = json.loads(groups_path.read_text(encoding="utf-8"))
        except Exception:
            state = None
            notes.append(f"Could not read {groups_path.name}; Analytics groups were not updated.")
        if isinstance(state, dict):
            before = json.dumps(state, sort_keys=True)
            anchor_group_state(state, old)  # legacy files carry no anchors yet
            notes.extend(remap_group_state(state, new))
            if json.dumps(state, sort_keys=True) != before:
                _backup(groups_path)
                groups_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                notes.append("Analytics groups, session order and prechop moved to the new names.")

    from abel.services.roi_service import ROIService  # noqa: PLC0415

    rois = ROIService()
    roi_path = project_root / rois.ROI_FILE
    if roi_path.exists():
        cfg = rois.load(project_root)
        subject_rois = cfg.get("subject_rois") or {}
        if subject_rois:
            moved, roi_notes = remap_subject_rois(
                subject_rois, old.subject_by_session, new.subject_by_session
            )
            notes.extend(roi_notes)
            if moved != subject_rois:
                _backup(roi_path)
                cfg["subject_rois"] = moved
                rois.save(project_root, cfg)
                notes.append("Per-subject ROIs moved to the new names.")

    for note in notes:
        logger.info("Subject rename: %s", note)
    return notes
