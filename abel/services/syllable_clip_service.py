"""Representative-clip extraction for each discovered syllable.

For each syllable the service scans every session with a sliding window of
``clip_frames`` width, scores each window by the fraction of frames labelled
with that syllable (enrichment), picks the top-N non-overlapping windows
globally across all sessions, and extracts the clips.

Each extracted clip has the syllable label burned into the top-left corner so
it is immediately identifiable during review.

Clips are written to:
    <project_root>/results/syllable_clips/<model_name>/syllable_<NNN>/

A per-syllable JSON manifest is saved alongside the clips.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import numpy as np

from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseProcessingService
from abel.services.preprocessing_service import (
    ClipExtractionConfig,
    ClipExtractionService,
    DEFAULT_PRESETS,
)
from abel.models.schemas import CandidateWindow, PreprocessingPreset

if TYPE_CHECKING:
    from abel.services.keypoint_moseq_service import KeypointMoSeqResult

logger = logging.getLogger("abel")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SyllableClipConfig:
    """Parameters for representative-clip extraction."""

    n_clips_per_syllable: int = 3
    """How many clips to extract per syllable, chosen by highest enrichment."""

    clip_frames: int = 90
    """Width of the sliding window in source-rate frames (e.g. 90 ≈ 3 s at 30 fps)."""

    min_syllable_frames: int = 10
    """Minimum number of frames of the target syllable required inside a window
    for that window to be considered a candidate clip."""

    model_name: str = "moseq_model_v1"
    """Used to name the output sub-directory."""

    preset: PreprocessingPreset = field(default_factory=lambda: DEFAULT_PRESETS[0])
    """Video encoding preset (resolution, FPS, crop margin, etc.)."""


@dataclass
class SyllableClipResult:
    """Outcome of a representative-clip extraction run."""

    output_dir: Path | None = None
    """Root folder that contains per-syllable sub-directories."""

    per_syllable_clips: dict[int, list[Path]] = field(default_factory=dict)
    """Mapping syllable_id → list of extracted clip paths."""

    per_syllable_candidates: dict[int, int] = field(default_factory=dict)
    """Total candidate windows found (before capping) per syllable."""

    warnings: list[str] = field(default_factory=list)
    total_clips: int = 0
    success: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Internal record for a candidate enrichment window
@dataclass
class _WindowRecord:
    syllable_id: int
    session_id: str
    start_frame: int
    end_frame: int
    enrichment: float  # fraction of window frames labelled as syllable_id


def _find_enriched_windows(
    assignments: np.ndarray,
    syllable_id: int,
    window_frames: int,
    min_syllable_frames: int = 5,
) -> list[_WindowRecord]:
    """Slide a fixed-length window across *assignments* and return enrichment-scored candidates.

    Uses a cumulative-sum trick for O(n) scoring. Windows with fewer than
    *min_syllable_frames* frames of *syllable_id* are discarded.
    """
    n = len(assignments)
    if n < window_frames or window_frames <= 0:
        return []

    step = max(1, window_frames // 4)  # 75% overlap between successive windows

    # Cumsum of the binary mask for O(1) range queries
    mask = (assignments == syllable_id).astype(np.int32)
    cs = np.zeros(n + 1, dtype=np.int32)
    cs[1:] = np.cumsum(mask)

    candidates: list[_WindowRecord] = []
    for start in range(0, n - window_frames + 1, step):
        end = start + window_frames - 1
        count = int(cs[end + 1] - cs[start])
        if count < min_syllable_frames:
            continue
        candidates.append(
            _WindowRecord(
                syllable_id=syllable_id,
                session_id="",  # filled by caller
                start_frame=start,
                end_frame=end,
                enrichment=count / window_frames,
            )
        )
    return candidates


def _select_nonoverlapping(
    candidates: list[_WindowRecord],
    n: int,
) -> list[_WindowRecord]:
    """Greedy non-overlapping selection of the top-n windows (highest enrichment first)."""
    # Assume candidates are already sorted by enrichment descending
    selected: list[_WindowRecord] = []
    for win in candidates:
        if len(selected) >= n:
            break
        overlap = any(
            not (win.end_frame < sel.start_frame or win.start_frame > sel.end_frame)
            for sel in selected
            if sel.session_id == win.session_id
        )
        if not overlap:
            selected.append(win)
    return selected


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SyllableClipService:
    """Extracts representative clips for each discovered syllable."""

    def __init__(self) -> None:
        self._project_root: Path | None = None
        self._imports = ImportService()
        self._pose = PoseProcessingService()
        self._clip_svc = ClipExtractionService()

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._clip_svc.set_project(project_root)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        discovery_result: "KeypointMoSeqResult",
        config: SyllableClipConfig,
        progress_callback: Callable[[str], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> SyllableClipResult:
        """Extract top-N representative clips per syllable.

        Parameters
        ----------
        discovery_result:
            The result of a syllable discovery run; contains per-session
            assignment file paths.
        config:
            Clip extraction settings.
        progress_callback:
            Optional callable that receives a plain string status message.
        cancel_flag:
            ``cancel_flag[0] = True`` to request cancellation mid-run.

        Returns
        -------
        SyllableClipResult
        """
        _prog = progress_callback or (lambda _msg: None)
        result = SyllableClipResult()

        if not self._project_root:
            result.warnings.append("No project loaded.")
            return result

        if not discovery_result or not discovery_result.success:
            result.warnings.append("Syllable discovery result is not available.")
            return result

        assignments_by_session = discovery_result.syllable_assignments  # sid → Path
        if not assignments_by_session:
            result.warnings.append("No assignment files found in the discovery result.")
            return result

        # ── Output directory ─────────────────────────────────────────────
        out_root = (
            self._project_root
            / "results"
            / "syllable_clips"
            / config.model_name
        )
        out_root.mkdir(parents=True, exist_ok=True)
        result.output_dir = out_root

        # ── Load import manifest once ────────────────────────────────────
        manifest = self._imports.load_manifest(self._project_root)
        if not manifest:
            result.warnings.append(
                "Import manifest not found. Ensure sessions are imported before extracting clips."
            )
            return result

        # ── Step 1: Load assignment arrays and scan for enriched windows ──
        _prog(
            f"Loading syllable assignments for {len(assignments_by_session)} session(s)..."
        )

        # all_windows[syllable_id] = list of _WindowRecord across all sessions
        all_windows: dict[int, list[_WindowRecord]] = {}

        for sid, assign_path in assignments_by_session.items():
            if cancel_flag and cancel_flag[0]:
                result.warnings.append("Cancelled.")
                return result

            _prog(f"Scanning enrichment windows: {sid}...")
            try:
                arr = np.load(str(assign_path))
                if isinstance(arr, np.lib.npyio.NpzFile):
                    keys = list(arr.files)
                    arr = arr[keys[0]]
                arr = np.asarray(arr, dtype=np.int32).ravel()
            except Exception as exc:
                logger.warning("Cannot load assignment for %s (%s): %s", sid, assign_path, exc)
                result.warnings.append(f"Could not load assignments for {sid}: {exc}")
                continue

            n_syllables_local = int(arr.max()) + 1 if arr.size > 0 else 0

            for syl_id in range(n_syllables_local):
                windows = _find_enriched_windows(
                    arr,
                    syl_id,
                    window_frames=config.clip_frames,
                    min_syllable_frames=config.min_syllable_frames,
                )
                for w in windows:
                    w.session_id = sid
                if syl_id not in all_windows:
                    all_windows[syl_id] = []
                all_windows[syl_id].extend(windows)

        if not all_windows:
            result.warnings.append("No enriched windows found across all sessions.")
            return result

        # ── Step 2: Sort by enrichment and select top-N non-overlapping ──
        _prog("Selecting highest-enrichment windows per syllable...")

        selected: dict[int, list[_WindowRecord]] = {}
        for syl_id, wins in sorted(all_windows.items()):
            result.per_syllable_candidates[syl_id] = len(wins)
            sorted_wins = sorted(wins, key=lambda w: -w.enrichment)
            selected[syl_id] = _select_nonoverlapping(sorted_wins, config.n_clips_per_syllable)

        # ── Step 3: Group selected windows by session for extraction ──────
        wins_by_session: dict[str, list[tuple[int, _WindowRecord]]] = {}
        for syl_id, wins in selected.items():
            for win in wins:
                wins_by_session.setdefault(win.session_id, []).append((syl_id, win))

        all_syllable_ids = sorted(selected.keys())
        _prog(
            f"Extracting clips: {len(all_syllable_ids)} syllable(s) across "
            f"{len(wins_by_session)} session(s)..."
        )

        per_syllable_clips: dict[int, list[Path]] = {sid: [] for sid in all_syllable_ids}

        # ── Step 4: Process each session ─────────────────────────────────
        for session_idx, (sess_id, win_list) in enumerate(sorted(wins_by_session.items())):
            if cancel_flag and cancel_flag[0]:
                result.warnings.append("Cancelled.")
                break

            _prog(
                f"Session {session_idx + 1}/{len(wins_by_session)}: "
                f"resolving video for {sess_id}..."
            )

            video_path = self._imports.video_path_for_session(manifest, sess_id)
            if not video_path or not video_path.exists():
                result.warnings.append(
                    f"Video not found for session {sess_id} — skipping "
                    f"{len(win_list)} clip(s)."
                )
                continue

            # Load pose centroid for crop centering (optional; fall back gracefully)
            pose_cx: np.ndarray | None = None
            pose_cy: np.ndarray | None = None
            pose_path = self._imports.pose_path_for_session(manifest, sess_id)
            if pose_path and pose_path.exists():
                try:
                    pose = self._pose.load_and_clean(pose_path, manifest.smoothing_settings)
                    pose_cx = pose.centroid_x
                    pose_cy = pose.centroid_y
                except Exception as exc:
                    logger.warning("Could not load pose centroid for %s: %s", sess_id, exc)

            # Build CandidateWindow list and per-syllable output dirs
            candidate_wins: list[CandidateWindow] = []
            win_to_syl: dict[str, int] = {}

            for syl_id, win in win_list:
                win_id = (
                    f"syl{syl_id:03d}_{sess_id}_{win.start_frame}_{win.end_frame}"
                    f"_{uuid.uuid4().hex[:6]}"
                )
                candidate_wins.append(
                    CandidateWindow(
                        window_id=win_id,
                        session_id=sess_id,
                        start_frame=win.start_frame,
                        end_frame=win.end_frame,
                        behavior_id=f"syllable_{syl_id:03d}",
                    )
                )
                win_to_syl[win_id] = syl_id

            syl_out_dirs: dict[int, Path] = {}
            for syl_id, _ in win_list:
                d = out_root / f"syllable_{syl_id:03d}"
                d.mkdir(parents=True, exist_ok=True)
                syl_out_dirs[syl_id] = d

            for cwin in candidate_wins:
                if cancel_flag and cancel_flag[0]:
                    break

                syl_id = win_to_syl[cwin.window_id]
                clip_out_dir = syl_out_dirs[syl_id]

                clip_cfg = ClipExtractionConfig(
                    video_path=video_path,
                    session_id=sess_id,
                    preset=config.preset,
                    output_dir=clip_out_dir,
                    pose_centroid_x=pose_cx,
                    pose_centroid_y=pose_cy,
                    pixels_per_mm=self._imports.pixels_per_mm_for_session(manifest, sess_id),
                    overlay_text=f"Syllable {syl_id:03d}",
                )

                _prog(
                    f"  Extracting syl {syl_id:03d} | "
                    f"{cwin.start_frame}\u2013{cwin.end_frame} from {sess_id}..."
                )

                ext_result = self._clip_svc.extract_selected_clips(
                    windows=[cwin],
                    config=clip_cfg,
                )

                if ext_result.warnings:
                    for w in ext_result.warnings:
                        # Skip the informational "no OpenCV" warning from bloating the log
                        if "OpenCV not installed" not in w:
                            result.warnings.append(f"  [{sess_id} syl{syl_id}] {w}")

                for clip in ext_result.clips:
                    if clip.processed_clip_path:
                        clip_path = Path(clip.processed_clip_path)
                        per_syllable_clips[syl_id].append(clip_path)
                    elif clip.start_frame is not None:
                        # No-OpenCV manifest-only mode: record placeholder
                        per_syllable_clips[syl_id].append(
                            clip_out_dir
                            / f"syl{syl_id:03d}_{clip.start_frame}_{clip.end_frame}.mp4"
                        )

        if cancel_flag and cancel_flag[0]:
            result.warnings.append("Clip extraction was cancelled before completion.")
            result.per_syllable_clips = per_syllable_clips
            result.total_clips = sum(len(v) for v in per_syllable_clips.values())
            result.success = False
            return result

        # ── Step 5: Write per-syllable manifest ───────────────────────────
        _prog("Writing manifest...")
        self._write_manifest(
            out_root=out_root,
            per_syllable_clips=per_syllable_clips,
            per_syllable_candidates=result.per_syllable_candidates,
            config=config,
        )

        result.per_syllable_clips = per_syllable_clips
        result.total_clips = sum(len(v) for v in per_syllable_clips.values())
        result.success = True

        logger.info(
            "Syllable clip extraction complete: %d clip(s) across %d syllable(s) → %s",
            result.total_clips,
            len(per_syllable_clips),
            out_root,
        )
        _prog(
            f"Done — {result.total_clips} clip(s) extracted "
            f"({len(per_syllable_clips)} syllable(s))."
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_manifest(
        out_root: Path,
        per_syllable_clips: dict[int, list[Path]],
        per_syllable_candidates: dict[int, int],
        config: SyllableClipConfig,
    ) -> None:
        """Write a JSON summary manifest to out_root/clip_manifest.json."""
        manifest: dict = {
            "model_name": config.model_name,
            "n_clips_per_syllable": config.n_clips_per_syllable,
            "clip_frames": config.clip_frames,
            "min_syllable_frames": config.min_syllable_frames,
            "preset": config.preset.preset_id,
            "syllables": {},
        }
        for syl_id in sorted(per_syllable_clips.keys()):
            clips = per_syllable_clips[syl_id]
            manifest["syllables"][str(syl_id)] = {
                "total_candidate_windows": per_syllable_candidates.get(syl_id, 0),
                "clips_extracted": len(clips),
                "clip_paths": [str(p) for p in clips],
            }

        out_path = out_root / "clip_manifest.json"
        try:
            out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not write clip manifest: %s", exc)
