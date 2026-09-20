"""Keep existing labels attached to the right animal when identities change.

Reviewer labels are keyed by ``seg_{animal_id}_{session}_{start}_{end}`` and the
per-animal feature rows are keyed the same way, so *both* sides move together
only as long as ``individual_subject_map`` and ``identity_corrections`` stay
put.  The moment either changes, a label committed earlier points at whichever
animal now occupies that id:

* **renaming** an individual (``track_0`` -> ``green``) changes every segment id
  for that animal, orphaning its labels;
* **adding a swap correction** at frame *t* exchanges the two animals' tracks
  from *t* onward, so a label committed on ``track_0`` after *t* now describes
  the other mouse.

This module computes, for each existing label, which animal id it should carry
under the new identities, and rewrites the label stores.  Labels whose window
*straddles* a correction frame are reported separately: the window contains both
animals' data, so no single id is right for it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def resolved_animal_ids(
    individuals: "list[str]",
    individual_subject_map: "dict[str, str] | None",
    subject_key: str,
) -> "dict[str, str]":
    """``{individual: animal_id}``, mirrors ``FeaturePrepService._process_one``."""
    imap = individual_subject_map or {}
    return {ind: (imap.get(ind) or f"{subject_key}:{ind}") for ind in individuals}


def permutation_at(
    individuals: "list[str]",
    corrections: "list[dict] | None",
    frame: int,
) -> "dict[str, str]":
    """``{identity: source_track}`` after every correction at or before ``frame``.

    Same composition rule as
    :meth:`PoseProcessingService.apply_identity_corrections`, so the ids here
    describe exactly what the extracted features hold.
    """
    perm = {o: o for o in individuals}
    for c in sorted(corrections or [], key=lambda c: int(c.get("frame", 0))):
        try:
            f = int(c.get("frame", 0))
            a, b = str(c.get("a")), str(c.get("b"))
        except Exception:
            continue
        if f <= int(frame) and a in perm and b in perm:
            perm[a], perm[b] = perm[b], perm[a]
    return perm


def parse_segment_id(segment_id: str, session_id: str) -> "tuple[str, int, int] | None":
    """``(animal_id, start, end)`` from ``seg_{animal}_{session}_{start}_{end}``.

    Animal ids contain underscores (``track_0``) and may contain a colon
    (``COA313:track_0``), so the id is peeled from the *ends* rather than split.
    """
    sid = str(segment_id or "")
    if not sid.startswith("seg_"):
        return None
    head, sep, tail = sid.rpartition(f"_{session_id}_")
    if not sep:
        return None
    bits = tail.split("_")
    if len(bits) != 2:
        return None
    try:
        start, end = int(bits[0]), int(bits[1])
    except ValueError:
        return None
    animal = head[len("seg_"):]
    return (animal, start, end) if animal else None


@dataclass
class IdentityRemapPlan:
    """What changing a session's identities would do to its existing labels."""

    session_id: str
    segment_renames: "dict[str, str]" = field(default_factory=dict)
    """old segment id -> new segment id, only where the id actually changes.

    A label can need updating without its id moving: renaming the *other*
    animal leaves this segment's own id alone but changes the partner id
    recorded inside its payload.  Those segments appear in
    :attr:`animal_renames` only.
    """
    animal_renames: "dict[str, dict[str, str]]" = field(default_factory=dict)
    """old segment id -> {old animal id: new animal id} for that window.

    Every segment affected at all, whether or not its own id moves.
    """
    n_labels: int = 0
    straddling: "list[str]" = field(default_factory=list)
    """Segment ids whose window contains a correction frame (both animals inside)."""

    @property
    def n_changed(self) -> int:
        """Labels this remap would touch: not just the ones whose id moves."""
        return len(self.animal_renames)

    def __bool__(self) -> bool:
        return bool(self.animal_renames)


def plan_identity_remap(
    *,
    session_id: str,
    individuals: "list[str]",
    subject_key: str,
    old_map: "dict[str, str] | None",
    new_map: "dict[str, str] | None",
    old_corrections: "list[dict] | None",
    new_corrections: "list[dict] | None",
    segment_ids: "list[str]",
) -> IdentityRemapPlan:
    """Work out the new animal id for every label segment of one session."""
    plan = IdentityRemapPlan(session_id=session_id)
    old_ids = resolved_animal_ids(individuals, old_map, subject_key)
    new_ids = resolved_animal_ids(individuals, new_map, subject_key)
    by_old_animal = {aid: ind for ind, aid in old_ids.items()}
    corr_frames = {
        int(c.get("frame", 0))
        for c in list(old_corrections or []) + list(new_corrections or [])
        if str(c.get("a")) in set(individuals) and str(c.get("b")) in set(individuals)
    }

    for seg in segment_ids:
        parsed = parse_segment_id(seg, session_id)
        if parsed is None:
            continue
        animal, start, end = parsed
        ind = by_old_animal.get(animal)
        if ind is None:  # a label from some other identity scheme: leave it alone
            continue
        plan.n_labels += 1

        # The physical track each identity was committed against, then the
        # identity that carries that same track under the new corrections.
        old_perm = permutation_at(individuals, old_corrections, start)
        new_perm = permutation_at(individuals, new_corrections, start)
        by_source = {src: o for o, src in new_perm.items()}
        mapping = {
            old_ids[o]: new_ids[by_source.get(old_perm.get(o, o), o)]
            for o in individuals
        }
        new_animal = mapping.get(animal, animal)
        if any(k != v for k, v in mapping.items()):
            plan.animal_renames[seg] = mapping
            # Only a *moved* id is a rename.  Recording a segment whose own
            # animal is unchanged made the confirmation dialog claim labels
            # were being moved when the rewrite was a no-op.
            if new_animal != animal:
                plan.segment_renames[seg] = f"seg_{new_animal}_{session_id}_{start}_{end}"
        if any(start < f <= end for f in corr_frames):
            plan.straddling.append(seg)
    return plan


def apply_identity_remap(project_root: Path, plan: IdentityRemapPlan) -> dict:
    """Rewrite the label stores in place. Returns ``{"labels", "soundboard"}`` counts."""
    out = {"labels": 0, "soundboard": 0}
    if not plan:
        return out
    out["labels"] = _remap_reviewer_labels(project_root, plan)
    out["soundboard"] = _remap_soundboard_labels(project_root, plan)
    return out


def _remap_reviewer_labels(project_root: Path, plan: IdentityRemapPlan) -> int:
    import pandas as pd  # noqa: PLC0415

    from abel.storage.file_store import atomic_write_parquet  # noqa: PLC0415

    path = project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
    if not path.exists():
        return 0
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - unreadable store
        logger.warning("Identity remap: could not read %s (%s)", path, exc)
        return 0
    if df.empty or "segment_id" not in df.columns:
        return 0

    changed = 0
    for idx, seg in df["segment_id"].items():
        animal_map = plan.animal_renames.get(str(seg))
        if not animal_map:
            continue
        new_seg = plan.segment_renames.get(str(seg))
        if new_seg:
            df.at[idx, "segment_id"] = new_seg
        for col in ("focal_animal_id", "partner_animal_id"):
            if col in df.columns:
                cur = df.at[idx, col]
                if isinstance(cur, str) and cur in animal_map:
                    df.at[idx, col] = animal_map[cur]
        changed += 1

    if changed:
        # Atomic: a half-written parquet here costs every reviewer label in the
        # project, and the store has no other copy.
        atomic_write_parquet(df, path, index=False)
    return changed


def _remap_soundboard_labels(project_root: Path, plan: IdentityRemapPlan) -> int:
    from abel.storage.file_store import read_json, write_json  # noqa: PLC0415

    path = project_root / "derived" / "review_labels" / "soundboard_labels.json"
    if not path.exists():
        return 0
    store = dict((read_json(path, {"windows": {}}) or {}).get("windows", {}) or {})
    if not store:
        return 0

    # The soundboard payload is keyed by window id (itself a segment id), and its
    # rows name animals directly, so both need the window's own mapping.
    changed = 0
    new_store: dict = {}
    for window_id, labels in store.items():
        parsed = parse_segment_id(str(window_id), plan.session_id)
        animal_map: dict = {}
        if parsed is not None:
            # Any segment of this window shares its frames, so reuse whichever
            # plan entry covers the same window.
            _animal, start, end = parsed
            for seg, mapping in plan.animal_renames.items():
                p = parse_segment_id(seg, plan.session_id)
                if p and p[1] == start and p[2] == end:
                    animal_map = mapping
                    break
        if animal_map:
            remapped = []
            for lab in (labels or []):
                row = dict(lab)
                # Only keys the payload already carries: adding
                # ``partner_animal_id: None`` to a solo label changed its shape.
                for key in ("focal_animal_id", "partner_animal_id"):
                    if key in row and row[key] is not None:
                        row[key] = animal_map.get(str(row[key]), row[key])
                remapped.append(row)
            labels = remapped
            changed += 1
        # The soundboard store is keyed by window id, which *is* the segment id,
        # so a renamed segment must take its payload with it, otherwise the
        # reviewer labels move and the per-subject payload is orphaned on the
        # old key.
        new_store[plan.segment_renames.get(str(window_id), window_id)] = labels

    if changed:
        write_json(path, {"windows": new_store})
    return changed


def label_segment_ids_for_session(project_root: Path, session_id: str) -> "list[str]":
    """Existing reviewer-label segment ids belonging to one session."""
    import pandas as pd  # noqa: PLC0415

    path = project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
    if not path.exists():
        return []
    try:
        df = pd.read_parquet(path, columns=["segment_id"])
    except Exception:
        return []
    if df.empty:
        return []
    needle = f"_{session_id}_"
    return [str(s) for s in df["segment_id"] if needle in str(s)]
