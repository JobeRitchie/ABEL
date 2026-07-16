"""Video and DLC pose import plus auto-linking logic."""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from abel.models.schemas import ImportManifest, ImportNameSettings, LinkedSession, PoseAsset, VideoAsset
from abel.storage.file_store import read_json, write_json
from abel.utils.sleap_converter import (
    SLEAP_POSE_EXTENSIONS,
    convert_slp_to_dlc,
    default_converted_path,
    is_sleap_pose_file,
)

logger = logging.getLogger("abel")


class ImportService:
    """Handles import bookkeeping and video/pose filename matching."""

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
    POSE_EXTENSIONS = {".csv", ".h5", ".hdf5"}
    # Segment ids, clip ids and review keys embed their session as a ``session_<hex>``
    # token (``bout_<uuid>_session_0952e047_44761_44775``), so re-pointing a recording
    # at a new session id is a token substitution wherever those ids are persisted.
    _SESSION_TOKEN_RE = re.compile(r"session_[0-9a-fA-F]+")
    # SLEAP predictions aren't read directly — they're converted to a DLC ``.h5``
    # (see :meth:`convert_sleap_poses`) that then flows through the normal path.
    SLEAP_EXTENSIONS = set(SLEAP_POSE_EXTENSIONS)

    def convert_sleap_poses(
        self,
        slp_paths: list[Path],
        *,
        reuse_existing: bool = True,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> tuple[list[Path], list[tuple[Path, str]]]:
        """Convert SLEAP ``.slp`` files to DeepLabCut ``.h5`` files.

        Returns ``(converted_paths, failures)`` where ``failures`` is a list of
        ``(source_path, error_message)``.  Already-converted files are reused
        when ``reuse_existing`` is set (keyed on the sibling ``*.sleap.h5``).
        """
        converted: list[Path] = []
        failures: list[tuple[Path, str]] = []
        total = len(slp_paths)
        for i, slp in enumerate(slp_paths):
            slp = Path(slp)
            if progress_cb:
                progress_cb(i, total, slp.name)
            try:
                target = default_converted_path(slp)
                if reuse_existing and target.exists() and target.stat().st_mtime >= slp.stat().st_mtime:
                    converted.append(target)
                    continue
                converted.append(convert_slp_to_dlc(slp, target))
            except Exception as exc:  # noqa: BLE001 - surface per-file, keep going
                logger.warning("SLEAP conversion failed for %s: %s", slp.name, exc)
                failures.append((slp, str(exc)))
        if progress_cb:
            progress_cb(total, total, "done")
        return converted, failures

    def build_manifest(
        self,
        video_paths: list[Path],
        pose_paths: list[Path],
        subject_name_settings: ImportNameSettings | None = None,
    ) -> ImportManifest:
        settings = subject_name_settings or ImportNameSettings()
        videos = [
            self._video_asset(path, settings)
            for path in video_paths
            if path.suffix.lower() in self.VIDEO_EXTENSIONS
        ]
        poses = [
            self._pose_asset(path, settings)
            for path in pose_paths
            if path.suffix.lower() in self.POSE_EXTENSIONS
        ]
        linked = self.auto_match(videos, poses)
        return ImportManifest(
            subject_name_settings=settings,
            videos=videos,
            poses=poses,
            linked_sessions=linked,
        )

    def auto_match(self, videos: list[VideoAsset], poses: list[PoseAsset]) -> list[LinkedSession]:
        pose_by_key = {self._match_key(Path(p.source_path)): p for p in poses}
        linked: list[LinkedSession] = []
        for video in videos:
            key = self._match_key(Path(video.source_path))
            pose = pose_by_key.get(key)
            if pose is None:
                continue
            individuals = list(pose.individuals or [])
            linked.append(
                LinkedSession(
                    session_id=f"session_{uuid4().hex[:8]}",
                    video_asset_id=video.asset_id,
                    pose_asset_id=pose.asset_id,
                    subject_id=video.subject_id,
                    pixels_per_mm=video.pixels_per_mm,
                    pairing_score=1.0,
                    pairing_notes="Auto-matched by filename stem",
                    individuals=individuals,
                    # Default each detected individual to itself; the import UI
                    # lets the user remap to real subject identities (green/black).
                    individual_subject_map={ind: ind for ind in individuals},
                )
            )
        return linked

    def update_session_individual_map(
        self,
        manifest: ImportManifest,
        session_id: str,
        individual_subject_map: dict[str, str],
    ) -> ImportManifest:
        """Set the individual→subject identity mapping for a multi-animal session."""
        session = next((s for s in manifest.linked_sessions if s.session_id == session_id), None)
        if session is None:
            return manifest
        cleaned = {
            str(ind): (str(name).strip() or str(ind))
            for ind, name in (individual_subject_map or {}).items()
            if str(ind) in set(session.individuals)
        }
        session.individual_subject_map = cleaned
        return manifest

    def update_session_identity_corrections(
        self,
        manifest: ImportManifest,
        session_id: str,
        corrections: list[dict],
    ) -> ImportManifest:
        """Set the identity-swap corrections for a multi-animal session."""
        session = next((s for s in manifest.linked_sessions if s.session_id == session_id), None)
        if session is None:
            return manifest
        valid_inds = set(session.individuals)
        cleaned: list[dict] = []
        for c in corrections or []:
            try:
                frame = int(c.get("frame"))
                a, b = str(c.get("a")), str(c.get("b"))
            except Exception:
                continue
            if a in valid_inds and b in valid_inds and a != b and frame > 0:
                cleaned.append({"frame": frame, "a": a, "b": b})
        session.identity_corrections = cleaned
        return manifest

    def update_session_subject(self, manifest: ImportManifest, session_id: str, subject_id: str) -> ImportManifest:
        """Update one linked session's subject and mirror to linked assets."""
        clean_subject = subject_id.strip() or None
        session = next((s for s in manifest.linked_sessions if s.session_id == session_id), None)
        if session is None:
            return manifest

        session.subject_id = clean_subject
        session.subject_locked = clean_subject is not None  # manual edits are protected from regex reapply
        video = next((v for v in manifest.videos if v.asset_id == session.video_asset_id), None)
        pose = next((p for p in manifest.poses if p.asset_id == session.pose_asset_id), None)
        if video is not None:
            video.subject_id = clean_subject
        if pose is not None:
            pose.subject_id = clean_subject
        return manifest

    def update_session_type(
        self, manifest: ImportManifest, session_id: str, session_type: str
    ) -> ImportManifest:
        """Set one linked session's explicit session-type override.

        An empty string clears the override so the effective type falls back to
        the regex-derived value.  Manual overrides survive a regex reapply, just
        like hand-set subjects.
        """
        clean = session_type.strip() or None
        session = next(
            (s for s in manifest.linked_sessions if s.session_id == session_id), None
        )
        if session is None:
            return manifest
        session.session_type = clean
        return manifest

    def effective_session_type(
        self, manifest: ImportManifest, session: LinkedSession
    ) -> str:
        """Resolve the display session type for a linked session.

        Priority: explicit override → regex-derived type stored on the video
        asset (``VideoAsset.session_id``) → filename stem with the subject prefix
        stripped.  Returns an empty string when nothing can be derived.
        """
        if session.session_type:
            return session.session_type
        video = next(
            (v for v in manifest.videos if v.asset_id == session.video_asset_id), None
        )
        if video is None:
            return ""
        if video.session_id:
            return str(video.session_id)
        # Fallback: strip the subject prefix off the filename stem.
        subject = str(session.subject_id or video.subject_id or "")
        stem = Path(video.source_path).stem
        if subject and stem.startswith(subject):
            return stem[len(subject):].lstrip("_- ")
        if not subject:
            return stem
        return ""

    def update_session_pixels_per_mm(
        self,
        manifest: ImportManifest,
        session_id: str,
        pixels_per_mm: float | None,
    ) -> ImportManifest:
        """Update one linked session's pixel scale and mirror to linked video asset."""
        clean_scale: float | None
        if pixels_per_mm is None:
            clean_scale = None
        else:
            clean_scale = float(pixels_per_mm)
            if clean_scale <= 0:
                clean_scale = None

        session = next((s for s in manifest.linked_sessions if s.session_id == session_id), None)
        if session is None:
            return manifest

        session.pixels_per_mm = clean_scale
        video = next((v for v in manifest.videos if v.asset_id == session.video_asset_id), None)
        if video is not None:
            video.pixels_per_mm = clean_scale
        return manifest

    def pixels_per_mm_for_session(self, manifest: ImportManifest, session_id: str) -> float | None:
        """Resolve a positive pixels/mm value for a session, if available."""
        session = next((s for s in manifest.linked_sessions if s.session_id == session_id), None)
        if not session:
            return None

        if session.pixels_per_mm is not None:
            try:
                val = float(session.pixels_per_mm)
            except Exception:
                val = 0.0
            if val > 0:
                return val

        video = next((v for v in manifest.videos if v.asset_id == session.video_asset_id), None)
        if video and video.pixels_per_mm is not None:
            try:
                val = float(video.pixels_per_mm)
            except Exception:
                val = 0.0
            if val > 0:
                return val
        return None

    def apply_subject_name_settings(
        self,
        manifest: ImportManifest,
        settings: ImportNameSettings,
    ) -> ImportManifest:
        """Recompute subject IDs from filenames using current parsing settings.

        Sessions whose subject was manually set (``subject_locked=True``) are
        left untouched so that hand-written corrections survive a regex reapply.
        """
        manifest.subject_name_settings = settings
        locked_video_ids = {
            s.video_asset_id for s in manifest.linked_sessions if s.subject_locked
        }
        locked_pose_ids = {
            s.pose_asset_id for s in manifest.linked_sessions if s.subject_locked
        }
        for video in manifest.videos:
            if video.asset_id not in locked_video_ids:
                video.subject_id = self.extract_subject_name(Path(video.source_path), settings)
                video.session_id = self.extract_session_type(Path(video.source_path), settings)
        for pose in manifest.poses:
            if pose.asset_id not in locked_pose_ids:
                pose.subject_id = self.extract_subject_name(Path(pose.source_path), settings)
                pose.session_id = self.extract_session_type(Path(pose.source_path), settings)

        for session in manifest.linked_sessions:
            if session.subject_locked:
                continue  # preserve the user's override
            video = next((v for v in manifest.videos if v.asset_id == session.video_asset_id), None)
            session.subject_id = video.subject_id if video else None
        return manifest

    def merge_new_files(
        self,
        manifest: ImportManifest,
        video_paths: list[Path],
        pose_paths: list[Path],
        settings: ImportNameSettings | None = None,
    ) -> ImportManifest:
        """Add newly selected files to *manifest* without disturbing existing sessions.

        Only paths not already recorded in the manifest are processed.
        Auto-matching attempts to pair:
        - new videos against new poses, and
        - new videos against previously-unlinked poses (and vice-versa),
        so that files imported in separate batches can still be paired.

        Existing sessions (including hand-edited subject names) are never modified.

        A file is recognised by *filename*, not by its absolute path: re-adding a
        recording that already exists under a different folder (after the project
        or the media folder moved) re-points the existing asset instead of adding
        a second copy of it, so the session — and every label and derived artifact
        hanging off its ``session_id`` — stays attached.
        """
        settings = settings or manifest.subject_name_settings or ImportNameSettings()
        manifest.subject_name_settings = settings

        fresh_videos = self._collect_fresh(
            video_paths, self.VIDEO_EXTENSIONS, manifest.videos,
            lambda path: self._video_asset(path, settings),
        )
        fresh_poses = self._collect_fresh(
            pose_paths, self.POSE_EXTENSIONS, manifest.poses,
            lambda path: self._pose_asset(path, settings),
        )

        manifest.videos.extend(fresh_videos)
        manifest.poses.extend(fresh_poses)

        if fresh_videos or fresh_poses:
            # All assets not yet part of an existing session — includes fresh ones and
            # any old unlinked assets, with no duplicates.
            linked_video_ids = {s.video_asset_id for s in manifest.linked_sessions}
            linked_pose_ids = {s.pose_asset_id for s in manifest.linked_sessions}
            matchable_videos = [v for v in manifest.videos if v.asset_id not in linked_video_ids]
            matchable_poses = [p for p in manifest.poses if p.asset_id not in linked_pose_ids]
            new_sessions = self.auto_match(matchable_videos, matchable_poses)
            # auto_match may suggest pairs already linked; keep only genuinely new ones.
            existing_pairs = {
                (s.video_asset_id, s.pose_asset_id) for s in manifest.linked_sessions
            }
            # A recording may only be linked once. Assets are deduplicated above, so
            # this only bites when a manifest already carries duplicates; it stops a
            # merge from compounding them.
            videos_by_id = {v.asset_id: v for v in manifest.videos}
            linked_keys = {
                self._match_key(Path(videos_by_id[vid].source_path))
                for vid in linked_video_ids
                if vid in videos_by_id
            }
            new_sessions = [
                s for s in new_sessions
                if (s.video_asset_id, s.pose_asset_id) not in existing_pairs
                and self._match_key(Path(videos_by_id[s.video_asset_id].source_path))
                not in linked_keys
            ]
            manifest.linked_sessions.extend(new_sessions)

        return manifest

    def _collect_fresh(
        self,
        paths: list[Path],
        extensions: set[str],
        existing: list,
        make_asset: Callable[[Path], object],
    ) -> list:
        """Return assets for the genuinely new files among *paths*.

        Files already imported are skipped. When such a file now lives somewhere
        else and the recorded path has gone stale, the existing asset is re-pointed
        at the new location rather than duplicated.
        """
        by_name = {self._asset_key(a.source_path): a for a in existing}
        fresh: list = []
        for path in paths:
            if path.suffix.lower() not in extensions:
                continue
            known = by_name.get(self._asset_key(path))
            if known is not None:
                if not Path(known.source_path).exists() and path.exists():
                    logger.info(
                        "Re-pointing %s: %s -> %s", path.name, known.source_path, path
                    )
                    known.source_path = str(path)
                continue
            asset = make_asset(path)
            by_name[self._asset_key(path)] = asset
            fresh.append(asset)
        return fresh

    @staticmethod
    def _asset_key(path: Path | str) -> str:
        """Identity of an imported file: its filename, case-folded.

        Filenames are already the project-wide key for a recording — poses are
        paired to videos by filename stem and the session registry is keyed by
        ``video_filename`` — so two assets sharing a filename are the same
        recording, whatever folder they were picked from.
        """
        return Path(path).name.casefold()

    def find_duplicate_sessions(self, manifest: ImportManifest) -> dict[str, list[str]]:
        """Group sessions that describe the same recording.

        Returns ``{canonical_session_id: [duplicate_session_ids...]}`` for every
        recording linked more than once, keyed by video filename. The first-linked
        session wins as canonical — it is the one the project's labels and derived
        data were built against.

        Projects imported before duplicate detection existed can carry these when a
        recording was re-added from a different folder (see :meth:`merge_new_files`).
        """
        videos_by_id = {v.asset_id: v for v in manifest.videos}
        by_key: dict[str, list[str]] = {}
        for session in manifest.linked_sessions:
            video = videos_by_id.get(session.video_asset_id)
            if video is None:
                continue
            key = self._asset_key(video.source_path)
            by_key.setdefault(key, []).append(session.session_id)
        return {ids[0]: ids[1:] for ids in by_key.values() if len(ids) > 1}

    def duplicate_session_remap(
        self, manifest: ImportManifest, removed_ids: set[str],
    ) -> dict[str, str]:
        """Map each removed session onto the kept session for the same recording.

        Only sessions whose video stays linked under another id are included:
        dropping one of two entries for one recording is a de-duplication, and the
        review work recorded against the dropped id describes the same frames of the
        same video, so it belongs to the surviving session.
        """
        videos_by_id = {v.asset_id: v for v in manifest.videos}

        def video_key(session: LinkedSession) -> str | None:
            video = videos_by_id.get(session.video_asset_id)
            return self._asset_key(video.source_path) if video else None

        kept_by_key: dict[str, str] = {}
        for session in manifest.linked_sessions:
            if session.session_id in removed_ids:
                continue
            key = video_key(session)
            if key and key not in kept_by_key:
                kept_by_key[key] = session.session_id

        remap: dict[str, str] = {}
        for session in manifest.linked_sessions:
            if session.session_id not in removed_ids:
                continue
            key = video_key(session)
            canonical = kept_by_key.get(key) if key else None
            if canonical:
                remap[session.session_id] = canonical
        return remap

    def stale_session_remap(self, project_root: Path, manifest: ImportManifest) -> dict[str, str]:
        """Map registry sessions the manifest no longer knows onto their current id.

        A recording keeps its filename across re-imports but is minted a fresh
        ``session_id`` each time, so labels recorded before a re-import or a
        de-duplication address a session that no longer exists. The registry — a log
        of every session ever imported, keyed by video filename — is what lets those
        labels be traced back to the session that now owns the same recording.
        """
        current_ids = {s.session_id for s in manifest.linked_sessions}
        videos_by_id = {v.asset_id: v for v in manifest.videos}
        current_by_key: dict[str, list[str]] = {}
        for session in manifest.linked_sessions:
            video = videos_by_id.get(session.video_asset_id)
            if video:
                key = self._asset_key(video.source_path)
                current_by_key.setdefault(key, []).append(session.session_id)

        remap: dict[str, str] = {}
        for old_id, entry in self.load_registry(project_root).items():
            if old_id in current_ids or not isinstance(entry, dict):
                continue
            filename = entry.get("video_filename")
            if not filename:
                continue
            matches = current_by_key.get(self._asset_key(filename), [])
            if len(matches) == 1:
                remap[old_id] = matches[0]
        return remap

    def remap_session_references(self, project_root: Path, remap: dict[str, str]) -> dict[str, int]:
        """Re-point persisted review work from old session ids onto current ones.

        Labels, review decisions and seeds are the user's own work, and they are keyed
        by ids that embed the owning session (``segment_id``, ``clip_id``). When a
        recording's session id changes, those keys must follow it — otherwise the
        labels match no feature row and are silently dropped from training. Derived
        caches are deliberately not touched here: they are rebuilt from the manifest.

        Returns the number of rows rewritten per artifact.
        """
        counts = {"labels": 0, "decisions": 0, "seeds": 0}
        if not remap:
            return counts

        labels_path = project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if labels_path.exists():
            try:
                import pandas as pd

                labels = pd.read_parquet(labels_path)
                if not labels.empty and "segment_id" in labels.columns:
                    before = labels["segment_id"].astype(str)
                    after = before.map(lambda text: self._rewrite_session_tokens(text, remap))
                    changed = int((after != before).sum())
                    if changed:
                        labels["segment_id"] = after
                        labels.to_parquet(labels_path, index=False)
                        counts["labels"] = changed
            except Exception:
                logger.warning("Could not remap session ids in %s", labels_path, exc_info=True)

        counts["decisions"] = self._remap_json_rows(
            project_root / "derived" / "review_tables" / "review_decisions.json",
            key="decisions",
            fields=("clip_id", "segment_id", "session_id"),
            remap=remap,
        )
        counts["seeds"] = self._remap_json_rows(
            project_root / "config" / "seeds.json",
            key="seeds",
            fields=("session_id", "segment_id"),
            remap=remap,
        )

        if any(counts.values()):
            logger.info(
                "Re-pointed review work from %d old session id(s): %d label(s), "
                "%d decision(s), %d seed(s).",
                len(remap), counts["labels"], counts["decisions"], counts["seeds"],
            )
        return counts

    @classmethod
    def _rewrite_session_tokens(cls, text: object, remap: dict[str, str]) -> str:
        return cls._SESSION_TOKEN_RE.sub(
            lambda match: remap.get(match.group(0), match.group(0)), str(text)
        )

    @classmethod
    def _remap_json_rows(
        cls, path: Path, key: str, fields: tuple[str, ...], remap: dict[str, str],
    ) -> int:
        if not path.exists():
            return 0
        try:
            raw = read_json(path, {})
        except Exception:
            return 0
        rows = raw.get(key, [])
        if not isinstance(rows, list):
            return 0

        changed = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            touched = False
            for field in fields:
                value = row.get(field)
                if not isinstance(value, str):
                    continue
                rewritten = cls._rewrite_session_tokens(value, remap)
                if rewritten != value:
                    row[field] = rewritten
                    touched = True
            changed += 1 if touched else 0

        if changed:
            write_json(path, {**raw, key: rows})
        return changed

    @classmethod
    def _prune_review_work_by_session(
        cls, project_root: Path, removed_ids: set[str],
    ) -> int:
        """Drop reviewer decisions/labels that address a removed session.

        Review decisions (``review_decisions.json``), reviewer labels
        (``reviewer_labels.parquet``) and the soundboard round-trip store are the
        user's own work, keyed by ids that embed the owning session
        (``clip_id``/``segment_id``/window id). Unlike the derived caches they are
        *not* rebuilt from the manifest, so a removed session's reviews linger in
        the queue forever — showing a raw ``session_<hex>`` code with no subject
        and no clip. Duplicate removals are re-pointed onto the surviving session
        before this runs (see :meth:`remap_session_references`), so anything still
        naming a removed id here belongs to a recording that has genuinely left
        the project. The generic ``session_id``-column parquet pruner can't touch
        reviewer_labels — its session lives inside ``segment_id`` — so it is
        handled explicitly here. Returns rows removed across all three stores.
        """
        if not removed_ids:
            return 0

        def targets_removed(text: object) -> bool:
            return any(
                tok in removed_ids for tok in cls._SESSION_TOKEN_RE.findall(str(text))
            )

        removed = cls._filter_json_list(
            project_root / "derived" / "review_tables" / "review_decisions.json",
            key="decisions",
            keep_fn=lambda row: not any(
                targets_removed(row.get(f))
                for f in ("clip_id", "segment_id", "session_id")
            ),
        )

        # Soundboard structured-label store is keyed by window id (embeds session).
        struct_path = project_root / "derived" / "review_labels" / "soundboard_labels.json"
        if struct_path.exists():
            try:
                raw = read_json(struct_path, {"windows": {}})
                windows = raw.get("windows", {}) or {}
                kept = {wid: v for wid, v in windows.items() if not targets_removed(wid)}
                if len(kept) != len(windows):
                    removed += len(windows) - len(kept)
                    write_json(struct_path, {"windows": kept})
            except Exception:
                logger.warning(
                    "Could not prune soundboard labels for removed sessions", exc_info=True
                )

        labels_path = project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if labels_path.exists():
            try:
                import pandas as pd

                labels = pd.read_parquet(labels_path)
                if not labels.empty and "segment_id" in labels.columns:
                    keep = ~labels["segment_id"].astype(str).map(targets_removed)
                    dropped = int((~keep).sum())
                    if dropped:
                        labels[keep].reset_index(drop=True).to_parquet(labels_path, index=False)
                        removed += dropped
            except Exception:
                logger.warning(
                    "Could not prune reviewer labels for removed sessions", exc_info=True
                )

        return removed

    def remove_sessions(
        self,
        project_root: Path,
        manifest: ImportManifest,
        session_ids: list[str],
    ) -> dict[str, int]:
        """Remove sessions from manifest and delete session-associated project data."""
        target_ids = {sid for sid in session_ids if sid}
        if not target_ids:
            return {"sessions": 0, "files": 0, "rows": 0, "remapped": 0}

        kept_sessions = [s for s in manifest.linked_sessions if s.session_id not in target_ids]
        removed_sessions = len(manifest.linked_sessions) - len(kept_sessions)
        if removed_sessions <= 0:
            return {"sessions": 0, "files": 0, "rows": 0, "remapped": 0}

        # Must run before the manifest is rewritten below — it needs the sessions and
        # videos that are about to be dropped. Removing a duplicate of a recording that
        # stays in the project re-points its review work at the surviving session; the
        # pruning further down would otherwise orphan every label recorded against it.
        remapped = self.remap_session_references(
            project_root, self.duplicate_session_remap(manifest, target_ids)
        )

        manifest.linked_sessions = kept_sessions

        used_video_ids = {s.video_asset_id for s in manifest.linked_sessions}
        used_pose_ids = {s.pose_asset_id for s in manifest.linked_sessions}
        manifest.videos = [v for v in manifest.videos if v.asset_id in used_video_ids]
        manifest.poses = [p for p in manifest.poses if p.asset_id in used_pose_ids]

        files_removed = 0
        rows_removed = 0

        for sid in sorted(target_ids):
            for path in [
                project_root / "derived" / "pose_features" / f"{sid}.npz",
                project_root / "derived" / "pose_cache" / f"{sid}_frame_features.npz",
                project_root / "derived" / "syllables" / f"{sid}_syllables.npz",
            ]:
                if path.exists():
                    path.unlink(missing_ok=True)
                    files_removed += 1

            clips_dir = project_root / "derived" / "clips" / sid
            files_removed += self._delete_tree(clips_dir)

        # Per-session caches are scattered across derived/ under names that embed
        # the session id — pose_features/sessions/<sid>.parquet, context_features/
        # sessions/<sid>.parquet, temporal_refinement/**/<sid>_bouts.parquet,
        # analytics_cache/<sid>_*.json. None are rebuilt on removal, so a session
        # deleted here otherwise keeps feeding inference, analytics and the UMAP.
        # Session ids are unique 8-hex tokens, so a name match is an exact hit.
        files_removed += self._delete_session_named_artifacts(
            project_root / "derived", target_ids
        )

        rows_removed += self._filter_json_list(
            project_root / "config" / "seeds.json",
            key="seeds",
            keep_fn=lambda row: row.get("session_id") not in target_ids,
        )

        rows_removed += self._filter_json_list(
            project_root / "derived" / "pose_features" / "summaries.json",
            key="summaries",
            keep_fn=lambda row: row.get("session_id") not in target_ids,
        )

        rows_removed += self._filter_json_list(
            project_root / "derived" / "review_tables" / "candidate_segments.json",
            key="candidates",
            keep_fn=lambda row: row.get("session_id") not in target_ids,
        )

        rows_removed += self._filter_json_list(
            project_root / "derived" / "review_tables" / "clip_manifest.json",
            key="clips",
            keep_fn=lambda row: row.get("session_id") not in target_ids,
            mutate_raw_fn=lambda raw: {
                **raw,
                "session_ids": [sid for sid in raw.get("session_ids", []) if sid not in target_ids],
            },
        )

        rows_removed += self._filter_json_list(
            project_root / "derived" / "candidates" / "candidates.json",
            key="candidates",
            keep_fn=lambda row: row.get("session_id") not in target_ids,
        )

        # External window candidates (active-learning / temporal-bout review queues)
        # persist across tabs and are NOT rebuilt from the manifest, so removed
        # sessions leak clips/windows here unless explicitly pruned.
        rows_removed += self._filter_json_list(
            project_root / "derived" / "review_tables" / "external_window_candidates.json",
            key="candidates",
            keep_fn=lambda row: row.get("session_id") not in target_ids,
        )

        # Reviewer decisions/labels are the user's own work and are NOT rebuilt.
        # Without this, a removed session's reviews linger in the queue as a raw
        # session_<hex> code with no subject and no extracted clip.
        rows_removed += self._prune_review_work_by_session(project_root, target_ids)

        # Aggregate feature/representation parquet stores are keyed by session and
        # are large content-caches that are NOT rebuilt on session removal. If not
        # pruned, removed sessions keep flowing into dense inference, the unified
        # UMAP, analytics, etc. ("Loaded ... 71 sessions" long after 45 were
        # dropped). Filter each in place so only retained sessions survive.
        kept_ids = {s.session_id for s in manifest.linked_sessions}
        for rel in (
            "derived/representations/frame_features.parquet",
            "derived/representations/segment_features.parquet",
            "derived/representations/enriched_segments.parquet",
            "derived/pose_features/frame_pose.parquet",
            "derived/context_features/frame_context.parquet",
            "derived/training_sets/training_set.parquet",
        ):
            rows_removed += self._prune_parquet_by_session(project_root / rel, kept_ids)

        # Bout tables are written one-per-behavior-model, so glob rather than name them.
        for path in sorted((project_root / "derived" / "behavior_bouts").glob("*.parquet")):
            rows_removed += self._prune_parquet_by_session(path, kept_ids)

        return {
            "sessions": removed_sessions,
            "files": files_removed,
            "rows": rows_removed,
            "remapped": sum(remapped.values()),
        }

    @staticmethod
    def _delete_session_named_artifacts(root: Path, session_ids: set[str]) -> int:
        """Delete every file/dir under *root* whose name embeds a removed session id.

        Returns the number of files deleted. Failures are logged, never raised — a
        session removal must not abort because one cache file was locked.
        """
        if not root.exists() or not session_ids:
            return 0
        targets = [
            path
            for path in root.rglob("*")
            if any(sid in path.name for sid in session_ids)
        ]
        removed = 0
        for path in sorted(targets, key=lambda p: len(p.parts), reverse=True):
            try:
                if path.is_dir():
                    removed += ImportService._delete_tree(path)
                elif path.exists():
                    path.unlink()
                    removed += 1
            except Exception:  # noqa: BLE001 - best-effort cache cleanup
                logger.warning("Could not delete %s for removed session", path, exc_info=True)
        return removed

    @staticmethod
    def _prune_parquet_by_session(path: Path, kept_session_ids: set[str]) -> int:
        """Drop rows for removed sessions from a ``session_id``-keyed parquet.

        Rewrites *path* in place keeping only rows whose ``session_id`` is in
        *kept_session_ids*. Returns the number of rows removed (0 when the file
        is absent, has no ``session_id`` column, or nothing matched). Failures
        are swallowed and logged — a session removal must not abort because one
        derived cache could not be rewritten.
        """
        if not path.exists():
            return 0
        try:
            import pandas as pd

            df = pd.read_parquet(path)
        except Exception:
            logger.warning("Could not read %s while pruning removed sessions", path, exc_info=True)
            return 0
        if "session_id" not in df.columns or df.empty:
            return 0
        keep_mask = df["session_id"].astype(str).isin({str(s) for s in kept_session_ids})
        removed = int((~keep_mask).sum())
        if removed <= 0:
            return 0
        try:
            df[keep_mask].reset_index(drop=True).to_parquet(path, index=False)
            logger.info("Pruned %d rows for removed sessions from %s", removed, path.name)
        except Exception:
            logger.warning("Could not rewrite %s while pruning removed sessions", path, exc_info=True)
            return 0
        return removed

    @staticmethod
    def extract_subject_name(path: Path, settings: ImportNameSettings | None = None) -> str | None:
        settings = settings or ImportNameSettings()
        stem = path.stem
        try:
            match = re.search(settings.subject_regex, stem)
        except re.error:
            return None
        if not match:
            return None
        try:
            value = match.group(settings.subject_group_index)
        except IndexError:
            return None
        clean = (value or "").strip()
        return clean or None

    @staticmethod
    def extract_session_type(path: Path, settings: ImportNameSettings | None = None) -> str | None:
        settings = settings or ImportNameSettings()
        if not settings.session_regex:
            return None
        stem = path.stem
        try:
            match = re.search(settings.session_regex, stem)
        except re.error:
            return None
        if not match:
            return None
        try:
            value = match.group(settings.session_group_index)
        except IndexError:
            return None
        clean = (value or "").strip()
        return clean or None

    def save_manifest(self, project_root: Path, manifest: ImportManifest) -> None:
        path = project_root / "derived" / "review_tables" / "import_manifest.json"
        write_json(path, manifest.model_dump(mode="json"))
        self.update_registry(project_root, manifest)

    def load_manifest(self, project_root: Path) -> ImportManifest | None:
        path = project_root / "derived" / "review_tables" / "import_manifest.json"
        if not path.exists():
            return None
        try:
            return ImportManifest.model_validate(read_json(path, {}))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Session registry — persistent log of every session ever imported
    # ------------------------------------------------------------------

    @staticmethod
    def _registry_path(project_root: Path) -> Path:
        return project_root / "config" / "session_registry.json"

    def load_registry(self, project_root: Path) -> dict[str, dict]:
        """Return the registry dict keyed by session_id.

        Each entry:
            video_filename  – bare filename (stable across re-imports)
            video_path      – full source path at import time
            subject_id      – subject label or None
            pose_filename   – bare pose filename
            first_seen      – ISO timestamp of first import
            last_seen       – ISO timestamp of most recent import
        """
        raw = read_json(self._registry_path(project_root), {"entries": {}})
        return raw.get("entries", {})

    def update_registry(self, project_root: Path, manifest: ImportManifest) -> None:
        """Append/update all sessions currently in *manifest* to the registry.

        Called automatically by save_manifest so no code outside this class
        needs to think about it.
        """
        registry = self.load_registry(project_root)
        now = datetime.utcnow().isoformat(timespec="seconds")

        for session in manifest.linked_sessions:
            video = next((v for v in manifest.videos if v.asset_id == session.video_asset_id), None)
            pose = next((p for p in manifest.poses if p.asset_id == session.pose_asset_id), None)

            video_filename = Path(video.source_path).name if video else None
            video_path = video.source_path if video else None
            pose_filename = Path(pose.source_path).name if pose else None

            existing = registry.get(session.session_id)
            registry[session.session_id] = {
                "video_filename": video_filename,
                "video_path": video_path,
                "subject_id": session.subject_id,
                "pixels_per_mm": session.pixels_per_mm,
                "pose_filename": pose_filename,
                "first_seen": existing["first_seen"] if existing else now,
                "last_seen": now,
            }

        path = self._registry_path(project_root)
        write_json(path, {"schema_version": "0.2.0", "entries": registry})

    def find_new_session_for_video(
        self,
        project_root: Path,
        old_session_id: str,
        manifest: ImportManifest,
    ) -> str | None:
        """Given an *old* session_id, use the registry to find which *current*
        manifest session has the same video filename.

        Returns the new session_id string, or None if no unique match is found.
        """
        registry = self.load_registry(project_root)
        entry = registry.get(old_session_id)
        if not entry or not entry.get("video_filename"):
            return None
        target_filename = entry["video_filename"]

        # Build a map from video_filename → list of current session_ids
        current: dict[str, list[str]] = {}
        for session in manifest.linked_sessions:
            video = next((v for v in manifest.videos if v.asset_id == session.video_asset_id), None)
            if video:
                fn = Path(video.source_path).name
                current.setdefault(fn, []).append(session.session_id)

        matches = current.get(target_filename, [])
        return matches[0] if len(matches) == 1 else None

    def resolve_session_id(self, project_root: Path, session_id: str, manifest: ImportManifest) -> str:
        """Return the canonical (current) session ID for *session_id*.

        If *session_id* is already present in the manifest it is returned as-is.
        Otherwise the registry is consulted via ``find_new_session_for_video`` to
        locate which current session owns the same source video.  Falls back to
        the original *session_id* when no match is found (the caller's usual
        "missing source video" path will then handle it gracefully).
        """
        if any(s.session_id == session_id for s in manifest.linked_sessions):
            return session_id
        resolved = self.find_new_session_for_video(project_root, session_id, manifest)
        return resolved if resolved else session_id

    def video_path_for_session(self, manifest: ImportManifest, session_id: str) -> Path | None:
        session = next((s for s in manifest.linked_sessions if s.session_id == session_id), None)
        if not session:
            return None
        video = next((v for v in manifest.videos if v.asset_id == session.video_asset_id), None)
        if not video:
            return None

        # Prefer existing local project copies, then existing source paths.
        candidates = [video.local_path, video.source_path]
        for raw in candidates:
            if not raw:
                continue
            path = Path(raw)
            if path.exists():
                return path

        # Fall back to source path for legacy behavior; caller will existence-check.
        return Path(video.source_path)

    def probe_and_cache_video_metadata(
        self,
        manifest: ImportManifest,
        manifest_path: Path,
    ) -> None:
        """Probe width, height, fps, and frame_count for any video asset that has
        null metadata and write the updated manifest back to disk.

        This is a fast one-time operation (~1 ms per video — no frame decoding).
        Callers can invoke it before any processing step to ensure metadata is
        available for resolution-dependent decisions (e.g. downsample factor) and
        to avoid repeated probing across multiple runs.
        """
        try:
            import cv2
        except ImportError:
            return

        updated = False
        for video in manifest.videos:
            if video.width and video.height and video.fps:
                continue  # already probed
            # Resolve the actual file path.
            candidates = [video.local_path, video.source_path]
            vid_path: Path | None = None
            for raw in candidates:
                if not raw:
                    continue
                p = Path(raw)
                if p.exists():
                    vid_path = p
                    break
            if vid_path is None:
                continue
            try:
                cap = cv2.VideoCapture(str(vid_path))
                if not cap.isOpened():
                    cap.release()
                    continue
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if w > 0 and h > 0:
                    video.width = w
                    video.height = h
                if fps > 0:
                    video.fps = fps
                if fc > 0:
                    video.frame_count = fc
                    video.duration_sec = fc / fps if fps > 0 else None
                updated = True
            except Exception:
                continue

        if updated:
            try:
                write_json(manifest_path, manifest.model_dump(mode="json"))
            except Exception:
                pass

    def pose_path_for_session(self, manifest: ImportManifest, session_id: str) -> Path | None:
        session = next((s for s in manifest.linked_sessions if s.session_id == session_id), None)
        if not session:
            return None
        pose = next((p for p in manifest.poses if p.asset_id == session.pose_asset_id), None)
        if not pose:
            return None

        # Prefer existing local project copies, then existing source paths.
        candidates = [pose.local_path, pose.source_path]
        for raw in candidates:
            if not raw:
                continue
            path = Path(raw)
            if path.exists():
                return path

        # Fall back to source path for legacy behavior; caller will existence-check.
        return Path(pose.source_path)

    @staticmethod
    def _match_key(path: Path) -> str:
        stem = path.stem.lower()
        # SLEAP prediction exports embed the source video *filename* (with its
        # extension), e.g. "myvideo.mp4_SLEAP...predictions.sleap" -> "myvideo".
        # Only strip when the extension sits at a real boundary (followed by a
        # separator or the end); otherwise a DLC name like "cage.moving_dlc"
        # would match ".mov" mid-word and be truncated to "cage".
        for vext in (".mp4", ".avi", ".mov", ".mkv"):
            vidx = stem.find(vext)
            if vidx > 0:
                after = stem[vidx + len(vext): vidx + len(vext) + 1]
                if after == "" or not after.isalnum():
                    return stem[:vidx].rstrip("_- .")
        # DLC standard naming: {video_name}DLC_{scorer}_{model}shuffleN_snapshot_N
        # The "DLC_" is appended directly (no leading underscore) to the video stem.
        dlc_idx = stem.find("dlc_")
        if dlc_idx > 0:
            return stem[:dlc_idx].rstrip("_- .")
        # Other common suffixes used by various trackers
        for suffix in ["_dlc", "_pose", "_tracking", "_tracked", "_labeled"]:
            if stem.endswith(suffix):
                return stem[: -len(suffix)].rstrip("_- .")
        return stem

    @staticmethod
    def _video_asset(path: Path, settings: ImportNameSettings) -> VideoAsset:
        return VideoAsset(
            asset_id=f"vid_{uuid4().hex[:10]}",
            source_path=str(path),
            local_path=None,
            subject_id=ImportService.extract_subject_name(path, settings),
            session_id=ImportService.extract_session_type(path, settings),
            pixels_per_mm=None,
        )

    @staticmethod
    def _pose_asset(path: Path, settings: ImportNameSettings) -> PoseAsset:
        # Probe body parts and frame count from the pose file header.
        body_parts: list[str] = []
        individuals: list[str] = []
        frame_count: int | None = None
        try:
            from abel.services.pose_processing_service import PoseProcessingService
            meta = PoseProcessingService.probe_metadata(path)
            body_parts = meta.get("body_parts", [])
            individuals = meta.get("individuals", []) or []
            n = meta.get("n_frames", 0)
            frame_count = n if n > 0 else None
        except Exception:
            pass
        return PoseAsset(
            asset_id=f"pose_{uuid4().hex[:10]}",
            source_path=str(path),
            local_path=None,
            format=path.suffix.lower().lstrip("."),
            frame_count=frame_count,
            body_parts=body_parts,
            individuals=individuals,
            subject_id=ImportService.extract_subject_name(path, settings),
            session_id=ImportService.extract_session_type(path, settings),
        )

    @staticmethod
    def _delete_tree(root: Path) -> int:
        if not root.exists():
            return 0
        removed = 0
        files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: len(p.parts), reverse=True)
        for file_path in files:
            try:
                file_path.unlink()
                removed += 1
            except Exception:
                pass
        dirs = sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True)
        for dir_path in dirs:
            try:
                dir_path.rmdir()
            except Exception:
                pass
        try:
            root.rmdir()
        except Exception:
            pass
        return removed

    # ------------------------------------------------------------------
    # Copy assets to project local storage
    # ------------------------------------------------------------------

    def copy_assets_to_project(
        self,
        project_root: Path,
        manifest: ImportManifest,
        *,
        copy_videos: bool = True,
        copy_poses: bool = True,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, int]:
        """Copy referenced video/pose files into the project ``raw/`` folders.

        Groups files by source directory and runs one ``robocopy`` call per
        group on Windows (falls back to ``shutil.copy2`` elsewhere).  This
        avoids per-file subprocess overhead and lets the OS optimise the
        transfer for the underlying device.

        Returns a dict with keys ``videos_copied`` and ``poses_copied``.
        """
        import subprocess
        import sys
        from collections import defaultdict

        videos_dir = project_root / "raw" / "videos"
        poses_dir = project_root / "raw" / "pose"
        videos_dir.mkdir(parents=True, exist_ok=True)
        poses_dir.mkdir(parents=True, exist_ok=True)

        # ---------- build work list ----------
        # Each item: (asset, src_path, dst_dir, kind)
        work: list[tuple] = []
        if copy_videos:
            for video in manifest.videos:
                if video.local_path and Path(video.local_path).exists():
                    continue
                src = Path(video.source_path)
                if not src.exists():
                    logger.warning("Copy skipped — source missing: %s", src)
                    continue
                dst = videos_dir / src.name
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    video.local_path = str(dst)
                    continue
                work.append((video, src, videos_dir, "video"))

        if copy_poses:
            for pose in manifest.poses:
                if pose.local_path and Path(pose.local_path).exists():
                    continue
                src = Path(pose.source_path)
                if not src.exists():
                    logger.warning("Copy skipped — source missing: %s", src)
                    continue
                dst = poses_dir / src.name
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    pose.local_path = str(dst)
                    continue
                work.append((pose, src, poses_dir, "pose"))

        total = len(work)
        if total == 0:
            if progress_cb:
                progress_cb(0, 0, "Nothing to copy — all files already local.")
            self.save_manifest(project_root, manifest)
            return {"videos_copied": 0, "poses_copied": 0}

        # ---------- group by (source_dir, dest_dir) ----------
        groups: dict[tuple[str, str], list[tuple]] = defaultdict(list)
        for item in work:
            _asset, src, dst_dir, _kind = item
            groups[(str(src.parent), str(dst_dir))].append(item)

        videos_copied = 0
        poses_copied = 0
        done = 0
        use_robocopy = sys.platform == "win32"

        for (src_dir, dst_dir), items in groups.items():
            filenames = [item[1].name for item in items]
            if progress_cb:
                progress_cb(done, total,
                            f"Copying {len(filenames)} files from {Path(src_dir).name}/ …")

            if use_robocopy:
                # One robocopy call copies every file in this group at once.
                # /MT:8  = 8 threads inside robocopy (its own internal parallelism)
                # /J     = unbuffered I/O for large files
                # /R:1 /W:1 = minimal retry/wait on transient errors
                cmd = ["robocopy", src_dir, dst_dir] + filenames + [
                    "/J", "/MT:8", "/R:1", "/W:1",
                    "/NP", "/NFL", "/NDL", "/NJH", "/NJS",
                ]
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=3600)
                    if result.returncode > 7:
                        err = result.stderr.decode(errors="replace").strip()
                        logger.error("robocopy failed (exit %d) for %s: %s",
                                     result.returncode, src_dir, err)
                except subprocess.TimeoutExpired:
                    logger.error("robocopy timed out for %s", src_dir)
                except Exception as exc:
                    logger.error("robocopy error for %s: %s", src_dir, exc)

            # Verify each file individually and update local_path.
            for asset, src, dst_dir_path, kind in items:
                done += 1
                dst = Path(dst_dir) / src.name
                if not dst.exists():
                    # Fallback: copy individually with Python.
                    try:
                        shutil.copy2(str(src), str(dst))
                    except Exception as exc:
                        logger.error("Failed to copy %s: %s", src, exc)
                        if progress_cb:
                            progress_cb(done, total, f"FAILED: {src.name} — {exc}")
                        continue

                asset.local_path = str(dst)
                if kind == "video":
                    videos_copied += 1
                else:
                    poses_copied += 1
                if progress_cb:
                    progress_cb(done, total, f"Copied: {src.name}")

        # Persist updated local_path values.
        self.save_manifest(project_root, manifest)

        return {"videos_copied": videos_copied, "poses_copied": poses_copied}

    @staticmethod
    def _filter_json_list(
        path: Path,
        key: str,
        keep_fn,
        mutate_raw_fn=None,
    ) -> int:
        if not path.exists():
            return 0
        try:
            raw = read_json(path, {})
        except Exception:
            return 0

        rows = raw.get(key, [])
        if not isinstance(rows, list):
            return 0
        kept = [row for row in rows if isinstance(row, dict) and keep_fn(row)]
        removed = len(rows) - len(kept)
        if removed <= 0 and mutate_raw_fn is None:
            return 0

        updated = {**raw, key: kept}
        if mutate_raw_fn is not None:
            updated = mutate_raw_fn(updated)
        write_json(path, updated)
        return max(0, removed)
