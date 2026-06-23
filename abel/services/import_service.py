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

logger = logging.getLogger("abel")


class ImportService:
    """Handles import bookkeeping and video/pose filename matching."""

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
    POSE_EXTENSIONS = {".csv", ".h5", ".hdf5"}

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
            linked.append(
                LinkedSession(
                    session_id=f"session_{uuid4().hex[:8]}",
                    video_asset_id=video.asset_id,
                    pose_asset_id=pose.asset_id,
                    subject_id=video.subject_id,
                    pixels_per_mm=video.pixels_per_mm,
                    pairing_score=1.0,
                    pairing_notes="Auto-matched by filename stem",
                )
            )
        return linked

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
        """
        settings = settings or manifest.subject_name_settings or ImportNameSettings()
        manifest.subject_name_settings = settings

        existing_video_paths = {v.source_path for v in manifest.videos}
        existing_pose_paths = {p.source_path for p in manifest.poses}

        fresh_videos = [
            self._video_asset(path, settings)
            for path in video_paths
            if path.suffix.lower() in self.VIDEO_EXTENSIONS
            and str(path) not in existing_video_paths
        ]
        fresh_poses = [
            self._pose_asset(path, settings)
            for path in pose_paths
            if path.suffix.lower() in self.POSE_EXTENSIONS
            and str(path) not in existing_pose_paths
        ]

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
            new_sessions = [
                s for s in new_sessions
                if (s.video_asset_id, s.pose_asset_id) not in existing_pairs
            ]
            manifest.linked_sessions.extend(new_sessions)

        return manifest

    def remove_sessions(
        self,
        project_root: Path,
        manifest: ImportManifest,
        session_ids: list[str],
    ) -> dict[str, int]:
        """Remove sessions from manifest and delete session-associated project data."""
        target_ids = {sid for sid in session_ids if sid}
        if not target_ids:
            return {"sessions": 0, "files": 0, "rows": 0}

        kept_sessions = [s for s in manifest.linked_sessions if s.session_id not in target_ids]
        removed_sessions = len(manifest.linked_sessions) - len(kept_sessions)
        if removed_sessions <= 0:
            return {"sessions": 0, "files": 0, "rows": 0}

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
            ]:
                if path.exists():
                    path.unlink(missing_ok=True)
                    files_removed += 1

            clips_dir = project_root / "derived" / "clips" / sid
            files_removed += self._delete_tree(clips_dir)

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

        return {
            "sessions": removed_sessions,
            "files": files_removed,
            "rows": rows_removed,
        }

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
        frame_count: int | None = None
        try:
            from abel.services.pose_processing_service import PoseProcessingService
            meta = PoseProcessingService.probe_metadata(path)
            body_parts = meta.get("body_parts", [])
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
