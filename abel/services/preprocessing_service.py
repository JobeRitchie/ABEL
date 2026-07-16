"""Clip extraction service.

Extracts video clips for a *specific* list of candidate windows selected by
Candidate Generation.  Video is decoded only for the windows
that scored above threshold — not for every possible window in the recording.

Pipeline position:
    Pose Features → Behavior Representations → Candidate Generation
    → **Clip Extraction** ← here
    → Review
"""



from __future__ import annotations



import logging

import math

import re

import uuid

from hashlib import sha1

from dataclasses import dataclass, field

from pathlib import Path

from typing import TYPE_CHECKING, Callable



if TYPE_CHECKING:

    import numpy as np



from abel.models.schemas import CandidateWindow, ClipAsset, ClipManifest, PreprocessingPreset

from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml



logger = logging.getLogger("abel")



# ---------------------------------------------------------------------------

# Default clip output presets

# ---------------------------------------------------------------------------



DEFAULT_PRESETS: list[PreprocessingPreset] = [

    PreprocessingPreset(

        preset_id="standard_crop",

        name="Standard Crop (224×224)",

        output_fps=15.0,

        resize_width=224,

        resize_height=224,

        crop_margin_px=80,

        grayscale=False,

    ),

    PreprocessingPreset(

        preset_id="large_crop",

        name="Large Crop (256×256)",

        output_fps=15.0,

        resize_width=256,

        resize_height=256,

        crop_margin_px=120,

        grayscale=False,

    ),

    PreprocessingPreset(

        preset_id="grayscale_small",

        name="Grayscale (128×128, 10 fps)",

        output_fps=10.0,

        resize_width=128,

        resize_height=128,

        crop_margin_px=60,

        grayscale=True,

    ),

]





@dataclass

class ClipExtractionConfig:

    """Parameters for one clip-extraction batch."""

    video_path: Path

    session_id: str

    preset: PreprocessingPreset

    output_dir: Path

    pose_centroid_x: "np.ndarray | None" = None  # optional centering

    pose_centroid_y: "np.ndarray | None" = None

    pixels_per_mm: float | None = None

    overlay_text: str | None = None  # optional text drawn in top-left of every frame

    static_center: bool = True  # crop on the clip-mean centroid (stable); False = per-frame follow (can jitter)

    individual_overlays: "list | None" = None  # [{name, color(BGR), cx[], cy[]}] -> colored dots + legend per animal





@dataclass

class ClipExtractionResult:

    """Outcome of extracting clips for one session."""

    session_id: str

    clips: list[ClipAsset] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)

    success: bool = False





def _has_cv2() -> bool:

    try:

        import cv2  # noqa: F401

        return True

    except ImportError:

        return False





class ClipExtractionService:

    """Extracts video clips for specific candidate windows only.



    Accepts a list of CandidateWindow objects (already scored and ranked by

    Candidate Generation) and decodes video only for those windows.

    """



    def __init__(self) -> None:

        self._project_root: Path | None = None



    def set_project(self, project_root: Path) -> None:

        self._project_root = project_root



    @staticmethod

    def can_decode_video() -> bool:

        """Return True when OpenCV video decoding is available in this environment."""

        return _has_cv2()



    @staticmethod

    def _scaled_crop_margin(

        base_margin_px: int,

        frame_width: int,

        frame_height: int,

        crop_area_scale: float = 1.25,

    ) -> int:

        """Scale crop margin with source resolution so higher-res files are less zoomed."""

        base = max(8, int(base_margin_px))

        short_edge = max(1, min(int(frame_width), int(frame_height)))

        reference_short_edge = 480

        scale = max(1.0, float(short_edge) / float(reference_short_edge))

        area_scale = max(0.1, float(crop_area_scale))

        linear_scale = float(math.sqrt(area_scale))

        scaled = int(round(base * scale * linear_scale))

        max_margin = max(8, short_edge // 2 - 1)

        return max(8, min(scaled, max_margin))



    @staticmethod
    def build_individual_overlays(pose_svc, pose_path, settings=None, individual_subject_map=None):
        """Build per-animal overlay dicts for a pose file: ``{name, color(BGR), cx, cy}``.

        Colors come from the shared palette (same as the identity dialog). ``name``
        uses the session's identity map when available, else the track id. Returns
        ``None`` for single-animal files (nothing to disambiguate) or on any error.
        """
        try:
            multi = pose_svc.load_and_clean_multi(pose_path, settings)
        except Exception:
            return None
        per = getattr(multi, "per_individual", {}) or {}
        if len(per) <= 1:
            return None
        from abel.utils.individual_colors import color_for_bgr
        imap = individual_subject_map or {}
        overlays = []
        for idx, (ind_id, pdata) in enumerate(per.items()):
            overlays.append({
                "name": str(imap.get(ind_id, ind_id)),
                "color": color_for_bgr(idx),
                "cx": pdata.centroid_x,
                "cy": pdata.centroid_y,
            })
        return overlays

    @staticmethod

    def clip_filename_for_id(clip_id: str) -> str:

        """Build a deterministic, Windows-safe filename stem for a clip id."""

        raw = (clip_id or "").strip()

        if not raw:

            raw = f"clip_{uuid.uuid4().hex[:10]}"



        # Keep filenames portable across platforms and append a short digest to avoid collisions.

        ascii_raw = raw.encode("ascii", "ignore").decode("ascii")

        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_raw).strip(" ._-")

        if not safe:

            safe = "clip"



        digest = sha1(raw.encode("utf-8")).hexdigest()[:8]

        stem = f"{safe}_{digest}"

        return stem[:180]



    # ------------------------------------------------------------------

    # Preset management

    # ------------------------------------------------------------------



    @property

    def default_presets(self) -> list[PreprocessingPreset]:

        return list(DEFAULT_PRESETS)



    def load_project_presets(self) -> list[PreprocessingPreset]:

        if not self._project_root:

            return list(DEFAULT_PRESETS)

        path = self._project_root / "config" / "preprocessing.yaml"

        raw = read_yaml(path, {})

        custom: list[PreprocessingPreset] = []

        for item in raw.get("presets", []):

            try:

                custom.append(PreprocessingPreset.model_validate(item))

            except Exception:

                pass

        custom_ids = {p.preset_id for p in custom}

        merged = list(custom)

        for p in DEFAULT_PRESETS:

            if p.preset_id not in custom_ids:

                merged.append(p)

        return merged



    def save_project_preset(self, preset: PreprocessingPreset) -> None:

        if not self._project_root:

            return

        path = self._project_root / "config" / "preprocessing.yaml"

        raw = read_yaml(path, {})

        presets = [p for p in raw.get("presets", []) if p.get("preset_id") != preset.preset_id]

        presets.append(preset.model_dump(mode="json"))

        write_yaml(path, {**raw, "presets": presets})



    # ------------------------------------------------------------------

    # Selective clip extraction

    # ------------------------------------------------------------------



    def extract_selected_clips(

        self,

        windows: list[CandidateWindow],

        config: ClipExtractionConfig,

        progress_callback: Callable[[int, int], None] | None = None,

        cancel_flag: list[bool] | None = None,

    ) -> ClipExtractionResult:

        """Decode video only for the provided candidate windows."""

        result = ClipExtractionResult(session_id=config.session_id)



        if not windows:

            result.warnings.append("No candidate windows provided.")

            return result



        config.output_dir.mkdir(parents=True, exist_ok=True)



        if not _has_cv2():

            # Manifest-only: record frame ranges without decoding video

            for i, win in enumerate(windows):

                if cancel_flag and cancel_flag[0]:

                    break

                result.clips.append(ClipAsset(

                    clip_id=win.window_id or f"clip_{uuid.uuid4().hex[:10]}",

                    session_id=config.session_id,

                    start_frame=win.start_frame,

                    end_frame=win.end_frame,

                ))

                if progress_callback:

                    progress_callback(i + 1, len(windows))

            result.warnings.append(

                "OpenCV not installed â€” frame range manifest written but no video files extracted."

            )

            result.success = True

            return result



        import cv2  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415



        cap = cv2.VideoCapture(str(config.video_path))

        if not cap.isOpened():

            result.warnings.append(f"Cannot open video: {config.video_path}")

            return result



        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        sorted_windows = sorted(windows, key=lambda w: (w.start_frame, w.end_frame))



        # Cache frame geometry once to avoid probing each candidate window.

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)



        try:

            for i, win in enumerate(sorted_windows):

                if cancel_flag and cancel_flag[0]:

                    result.warnings.append("Cancelled by user.")

                    break



                try:

                    start_frame = int(win.start_frame)

                    end_frame = int(win.end_frame)

                except Exception:

                    result.warnings.append(

                        f"Window has invalid frame values ({win.start_frame!r}, {win.end_frame!r}); skipped."

                    )

                    if progress_callback:

                        progress_callback(i + 1, len(sorted_windows))

                    continue



                if start_frame < 0:

                    start_frame = 0

                if end_frame < start_frame:

                    end_frame = start_frame

                if frame_count > 0:

                    start_frame = min(start_frame, frame_count - 1)

                    end_frame = min(max(start_frame, end_frame), frame_count - 1)



                clip_id = win.window_id or f"clip_{uuid.uuid4().hex[:10]}"

                out_path = config.output_dir / f"{self.clip_filename_for_id(clip_id)}.mp4"



                # Crop center: use pose centroid if available, else frame center

                if config.pose_centroid_x is not None and config.pose_centroid_y is not None:

                    n_pose = min(len(config.pose_centroid_x), len(config.pose_centroid_y))

                    if n_pose <= 0:

                        result.warnings.append(f"Window {start_frame}-{end_frame}: pose centroid arrays are empty; using frame center.")

                        if frame_w <= 0 or frame_h <= 0:

                            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                            ret, probe = cap.read()

                            if not ret:

                                result.warnings.append(f"Cannot read frame {start_frame}")

                                if progress_callback:

                                    progress_callback(i + 1, len(sorted_windows))

                                continue

                            frame_h, frame_w = probe.shape[:2]

                        cx, cy = frame_w / 2.0, frame_h / 2.0

                    else:

                        sf = max(0, min(start_frame, n_pose - 1))

                        ef = max(sf + 1, min(end_frame + 1, n_pose))

                        cx = float(np.nanmean(config.pose_centroid_x[sf:ef]))

                        cy = float(np.nanmean(config.pose_centroid_y[sf:ef]))

                        if not np.isfinite(cx) or not np.isfinite(cy):

                            if frame_w <= 0 or frame_h <= 0:

                                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                                ret, probe = cap.read()

                                if not ret:

                                    result.warnings.append(f"Cannot read frame {start_frame}")

                                    if progress_callback:

                                        progress_callback(i + 1, len(sorted_windows))

                                    continue

                                frame_h, frame_w = probe.shape[:2]

                            cx, cy = frame_w / 2.0, frame_h / 2.0

                else:

                    if frame_w <= 0 or frame_h <= 0:

                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                        ret, probe = cap.read()

                        if not ret:

                            result.warnings.append(f"Cannot read frame {start_frame}")

                            if progress_callback:

                                progress_callback(i + 1, len(sorted_windows))

                            continue

                        frame_h, frame_w = probe.shape[:2]

                    cx, cy = frame_w / 2.0, frame_h / 2.0



                error = self._write_clip(

                    cap,

                    start_frame,

                    end_frame,

                    out_path,

                    cx,

                    cy,

                    config.preset,

                    pose_centroid_x=config.pose_centroid_x,

                    pose_centroid_y=config.pose_centroid_y,

                    overlay_text=config.overlay_text,

                    static_center=config.static_center,

                    individual_overlays=config.individual_overlays,

                )

                if error:

                    result.warnings.append(f"Window {start_frame}-{end_frame}: {error}")

                    if progress_callback:

                        progress_callback(i + 1, len(sorted_windows))

                    continue



                if (not out_path.exists()) or out_path.stat().st_size <= 0:

                    result.warnings.append(

                        f"Window {start_frame}-{end_frame}: output clip is empty or missing"

                    )

                    if progress_callback:

                        progress_callback(i + 1, len(sorted_windows))

                    continue



                clip = ClipAsset(

                    clip_id=clip_id,

                    session_id=config.session_id,

                    start_frame=start_frame,

                    end_frame=end_frame,

                    processed_clip_path=str(out_path),

                )

                result.clips.append(clip)



                if progress_callback:

                    progress_callback(i + 1, len(sorted_windows))

        finally:

            cap.release()



        if not result.clips:

            result.warnings.append("No playable clips were extracted.")

            result.success = False

            return result



        result.success = True

        return result



    # ------------------------------------------------------------------

    # Low-level clip writer

    # ------------------------------------------------------------------



    @staticmethod

    def _write_clip(

        cap,

        start_frame: int,

        end_frame: int,

        out_path: Path,

        cx: float,

        cy: float,

        preset: PreprocessingPreset,

        pose_centroid_x=None,

        pose_centroid_y=None,

        overlay_text: str | None = None,

        static_center: bool = True,

        individual_overlays=None,

    ) -> str | None:

        """Write a single video clip cropped around (cx, cy). Returns error string or None.

        With ``static_center`` (default) the crop stays fixed on ``(cx, cy)`` — which
        callers pass as the clip-mean centroid — so the view does not jitter when the
        per-frame centroid is noisy. ``individual_overlays`` draws a colored dot per
        animal plus a legend so reviewers can tell the animals apart.
        """

        import cv2  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415



        try:

            start_frame = int(start_frame)

            end_frame = int(end_frame)

        except Exception:

            return "Invalid start/end frame values"

        if start_frame < 0:

            start_frame = 0

        if end_frame < start_frame:

            end_frame = start_frame



        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        ret, probe = cap.read()

        if not ret:

            return f"Cannot read frame {start_frame}"



        vid_h, vid_w = probe.shape[:2]

        m = ClipExtractionService._scaled_crop_margin(

            preset.crop_margin_px,

            vid_w,

            vid_h,

            crop_area_scale=float(getattr(preset, "crop_area_scale", 1.25) or 1.25),

        )

        out_w = preset.resize_width if preset.resize_width > 0 else (m * 2)

        out_h = preset.resize_height if preset.resize_height > 0 else (m * 2)

        out_fps = preset.output_fps if preset.output_fps > 0 else 30.0



        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]

        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))

        if not writer.isOpened():

            return f"Cannot create output file {out_path}"



        use_dynamic_center = (
            not static_center
            and pose_centroid_x is not None
            and pose_centroid_y is not None
        )



        if not np.isfinite(cx) or not np.isfinite(cy):

            cx, cy = vid_w / 2.0, vid_h / 2.0

        last_cx = float(cx)

        last_cy = float(cy)

        smoothing_alpha = 0.2

        max_center_jump = max(4.0, float(m) * 0.35)



        x1 = max(0, int(cx - m))

        y1 = max(0, int(cy - m))

        x2 = min(vid_w, int(cx + m))

        y2 = min(vid_h, int(cy + m))



        try:

            frame = probe

            n_frames_to_write = max(1, end_frame - start_frame + 1)

            for idx in range(n_frames_to_write):

                if idx > 0:

                    ret, frame = cap.read()

                    if not ret:

                        break



                if use_dynamic_center:

                    assert pose_centroid_x is not None and pose_centroid_y is not None

                    n_pose = min(len(pose_centroid_x), len(pose_centroid_y))

                    if n_pose <= 0:

                        dyn_cx, dyn_cy = last_cx, last_cy

                    else:

                        fidx = start_frame + idx

                        if fidx < 0:

                            fidx = 0

                        if fidx >= n_pose:

                            fidx = n_pose - 1

                        dyn_cx = float(pose_centroid_x[fidx])

                        dyn_cy = float(pose_centroid_y[fidx])

                    if not np.isfinite(dyn_cx) or not np.isfinite(dyn_cy):

                        dyn_cx, dyn_cy = last_cx, last_cy

                    else:

                        dyn_cx = float(smoothing_alpha * dyn_cx + (1.0 - smoothing_alpha) * last_cx)

                        dyn_cy = float(smoothing_alpha * dyn_cy + (1.0 - smoothing_alpha) * last_cy)

                        dx = dyn_cx - last_cx

                        dy = dyn_cy - last_cy

                        jump = float(np.hypot(dx, dy))

                        if jump > max_center_jump:

                            scale = max_center_jump / max(jump, 1e-9)

                            dyn_cx = last_cx + dx * scale

                            dyn_cy = last_cy + dy * scale

                        last_cx, last_cy = dyn_cx, dyn_cy

                    x1 = max(0, int(dyn_cx - m))

                    y1 = max(0, int(dyn_cy - m))

                    x2 = min(vid_w, int(dyn_cx + m))

                    y2 = min(vid_h, int(dyn_cy + m))



                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:

                    crop = frame

                crop = cv2.resize(crop, (out_w, out_h))

                if preset.grayscale:

                    crop = cv2.cvtColor(

                        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR

                    )

                # Per-animal colored dots + legend so reviewers can tell animals apart.
                if individual_overlays:
                    cw = max(1, x2 - x1)
                    ch = max(1, y2 - y1)
                    sx = out_w / float(cw)
                    sy = out_h / float(ch)
                    fidx = start_frame + idx
                    dot_r = max(3, out_w // 70)
                    for ov in individual_overlays:
                        ocx = ov.get("cx")
                        ocy = ov.get("cy")
                        if ocx is None or ocy is None:
                            continue
                        n_ov = min(len(ocx), len(ocy))
                        if n_ov <= 0:
                            continue
                        fi = 0 if fidx < 0 else (n_ov - 1 if fidx >= n_ov else fidx)
                        ax = float(ocx[fi])
                        ay = float(ocy[fi])
                        if not (np.isfinite(ax) and np.isfinite(ay)):
                            continue
                        px = int(round((ax - x1) * sx))
                        py = int(round((ay - y1) * sy))
                        if 0 <= px < out_w and 0 <= py < out_h:
                            color = ov.get("color", (0, 0, 255))
                            cv2.circle(crop, (px, py), dot_r + 1, (0, 0, 0), -1, cv2.LINE_AA)
                            cv2.circle(crop, (px, py), dot_r, color, -1, cv2.LINE_AA)

                    lfont = cv2.FONT_HERSHEY_SIMPLEX
                    lscale = max(0.35, min(0.55, out_w / 420.0))
                    row_h = max(14, int(18 * lscale / 0.4))
                    sw = max(8, row_h - 6)
                    lpad = 4
                    for i, ov in enumerate(individual_overlays):
                        name = str(ov.get("name", ""))
                        color = ov.get("color", (0, 0, 255))
                        (tw, _th), _bl = cv2.getTextSize(name, lfont, lscale, 1)
                        total_w = sw + 4 + tw
                        x0 = max(0, out_w - total_w - lpad)
                        y0 = lpad + i * row_h
                        cv2.rectangle(crop, (x0 - 2, y0 - 1), (out_w - lpad + 1, y0 + row_h - 3), (0, 0, 0), -1)
                        cv2.rectangle(crop, (x0, y0 + 1), (x0 + sw, y0 + sw + 1), color, -1)
                        cv2.putText(crop, name, (x0 + sw + 4, y0 + sw), lfont, lscale, (255, 255, 255), 1, cv2.LINE_AA)

                if overlay_text:

                    font = cv2.FONT_HERSHEY_SIMPLEX

                    font_scale = max(0.35, min(0.6, out_w / 400.0))

                    thickness = 1

                    (tw, th), baseline = cv2.getTextSize(

                        overlay_text, font, font_scale, thickness

                    )

                    pad = 3

                    cv2.rectangle(

                        crop, (0, 0), (tw + pad * 2, th + baseline + pad * 2), (0, 0, 0), -1

                    )

                    cv2.putText(

                        crop, overlay_text, (pad, th + pad),

                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,

                    )

                writer.write(crop)

        finally:

            writer.release()



        return None



    # ------------------------------------------------------------------

    # Manifest persistence

    # ------------------------------------------------------------------



    def save_manifest(self, manifest: ClipManifest) -> None:

        if not self._project_root:

            return

        path = self._project_root / "derived" / "review_tables" / "clip_manifest.json"

        write_json(path, manifest.model_dump(mode="json"))

        logger.info("Clip manifest saved: %d clips â†’ %s", len(manifest.clips), path)



    def load_manifest(self) -> ClipManifest | None:

        if not self._project_root:

            return None

        path = self._project_root / "derived" / "review_tables" / "clip_manifest.json"

        if not path.exists():

            return None

        try:

            return ClipManifest.model_validate(read_json(path, {}))

        except Exception:

            return None



    def clear_extracted_clips(self, session_id: str | None = None) -> int:

        """Delete extracted clip files for one session (or all sessions when None)."""

        if not self._project_root:

            return 0



        clips_root = self._project_root / "derived" / "clips"

        removed = 0



        targets: list[Path]

        if session_id:

            targets = [clips_root / session_id]

        else:

            targets = [clips_root]



        for target in targets:

            if not target.exists():

                continue



            # Remove files first so we can log Windows file-lock failures clearly.

            files = sorted((p for p in target.rglob("*") if p.is_file()), key=lambda p: len(p.parts), reverse=True)

            for file_path in files:

                try:

                    file_path.unlink()

                    removed += 1

                except Exception as exc:

                    logger.warning("Could not delete clip file %s: %s", file_path, exc)



            # Then remove now-empty directories bottom-up.

            dirs = sorted((p for p in target.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True)

            for dir_path in dirs:

                try:

                    dir_path.rmdir()

                except Exception:

                    pass

            try:

                target.rmdir()

            except Exception:

                pass



        # Remove stale clip manifest after deletion.

        manifest_path = self._project_root / "derived" / "review_tables" / "clip_manifest.json"

        if manifest_path.exists():

            manifest_path.unlink(missing_ok=True)



        logger.info(

            "Cleared %d extracted clip file(s)%s",

            removed,

            f" for session {session_id}" if session_id else "",

        )

        return removed





# ---------------------------------------------------------------------------
# Clip regeneration for candidates whose extracted files went missing
# ---------------------------------------------------------------------------


def regenerate_clips_for_windows(
    project_root: Path,
    windows: "list[CandidateWindow]",
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_flag: list[bool] | None = None,
) -> dict:
    """Re-extract clip files for an arbitrary set of candidate windows.

    Used by the Review tab to rebuild clips for candidates that no longer have
    a clip file on disk (for example after "Clear Unreviewed Clips", a session
    removal cascade, or the clip queue growing past what was ever extracted).

    The extraction settings (preset, crop area, before/after padding) are read
    from ``project.yaml`` so regenerated clips match whatever the user last used
    in the Clip Extraction tab.  Newly written clips are merged into the
    existing ``clip_manifest.json`` rather than replacing it.

    Returns a summary ``dict`` with ``extracted``, ``requested``,
    ``session_ids`` and ``warnings`` keys.
    """
    from abel.services.import_service import ImportService
    from abel.services.pose_processing_service import PoseProcessingService

    summary: dict = {"extracted": 0, "requested": len(windows), "session_ids": [], "warnings": []}
    if not windows:
        summary["warnings"].append("No candidate windows provided.")
        return summary

    imports = ImportService()
    pose_processing = PoseProcessingService()
    clip_svc = ClipExtractionService()
    clip_svc.set_project(project_root)

    manifest = imports.load_manifest(project_root)
    if manifest is None:
        summary["warnings"].append("Import manifest not found — run Data Import first.")
        return summary

    # Extraction settings mirror the Clip Extraction tab so regenerated clips
    # match the crop/padding of the originals.
    project_cfg = read_yaml(project_root / "project.yaml", {})
    ui = dict(project_cfg.get("clip_extraction_ui") or {})
    try:
        source_fps = float(project_cfg.get("default_fps", 30.0))
    except Exception:
        source_fps = 30.0
    before_sec = float(ui.get("before_sec", 0.0) or 0.0)
    after_sec = float(ui.get("after_sec", 0.0) or 0.0)
    before_frames = int(round(max(0.0, before_sec) * max(source_fps, 1.0)))
    after_frames = int(round(max(0.0, after_sec) * max(source_fps, 1.0)))

    presets = clip_svc.load_project_presets()
    preset_id = str(ui.get("preset_id") or "").strip()
    preset = next((p for p in presets if p.preset_id == preset_id), None)
    if preset is None:
        preset = presets[0]
    crop_area_percent = float(ui.get("crop_area_percent", 0.0) or 0.0)
    if crop_area_percent > 0:
        crop_area_scale = max(0.5, min(10.0, crop_area_percent / 100.0))
        preset = PreprocessingPreset.model_validate(
            preset.model_dump(mode="python") | {"crop_area_scale": crop_area_scale}
        )

    # Group requested windows by canonical session id, applying before/after
    # padding so regenerated clips carry the same context as the originals.
    # Windows may arrive as CandidateWindow (pydantic) or plain row objects, so
    # rebuild a fresh CandidateWindow from duck-typed attributes rather than
    # relying on ``.model_copy``.
    current_ids = {s.session_id for s in manifest.linked_sessions}
    session_plan: dict[str, list[CandidateWindow]] = {}
    for win in windows:
        sid = str(getattr(win, "session_id", "")).strip()
        wid = str(getattr(win, "window_id", "")).strip()
        if not sid or not wid:
            continue
        if sid not in current_ids:
            sid = imports.resolve_session_id(project_root, sid, manifest)
        try:
            raw_start = int(getattr(win, "start_frame", 0))
        except Exception:
            raw_start = 0
        try:
            raw_end = int(getattr(win, "end_frame", raw_start))
        except Exception:
            raw_end = raw_start
        start = max(0, raw_start - before_frames)
        end = max(start, raw_end + after_frames)
        behavior_id = getattr(win, "behavior_id", None)
        padded = CandidateWindow(
            window_id=wid,
            session_id=sid,
            start_frame=int(start),
            end_frame=int(end),
            behavior_id=str(behavior_id) if behavior_id else None,
        )
        session_plan.setdefault(sid, []).append(padded)

    total = sum(len(v) for v in session_plan.values())
    if total <= 0:
        summary["warnings"].append("No windows resolved to a known session.")
        return summary

    done = 0

    def _emit(delta: int) -> None:
        nonlocal done
        if delta > 0:
            done += delta
        if progress_callback:
            progress_callback(min(done, total), total)

    all_clips: list[ClipAsset] = []
    extracted_sessions: list[str] = []

    for sid, sess_windows in session_plan.items():
        if cancel_flag and cancel_flag[0]:
            summary["warnings"].append("Cancelled by user.")
            break

        video_path = imports.video_path_for_session(manifest, sid)
        if not video_path or not video_path.exists():
            summary["warnings"].append(f"{sid}: missing source video, skipped {len(sess_windows)} clip(s).")
            _emit(len(sess_windows))
            continue

        pose_cx = None
        pose_cy = None
        pose_path = imports.pose_path_for_session(manifest, sid)
        if pose_path and pose_path.exists():
            try:
                pose = pose_processing.load(pose_path)
                pose = pose_processing.clean_pose(
                    pose,
                    likelihood_threshold=0.2,
                    interpolate=True,
                    smoothing_window=5,
                )
                pose_cx = pose.centroid_x
                pose_cy = pose.centroid_y
            except Exception as exc:
                summary["warnings"].append(f"{sid}: could not load pose centroids ({exc}); using static center crop.")

        individual_overlays = None
        if pose_path and pose_path.exists():
            _sess = next((s for s in manifest.linked_sessions if s.session_id == sid), None)
            _imap = dict(getattr(_sess, "individual_subject_map", {}) or {}) if _sess else {}
            individual_overlays = ClipExtractionService.build_individual_overlays(
                pose_processing, pose_path,
                getattr(manifest, "smoothing_settings", None), _imap,
            )

        cfg = ClipExtractionConfig(
            video_path=video_path,
            session_id=sid,
            preset=preset,
            output_dir=project_root / "derived" / "clips" / sid,
            pose_centroid_x=pose_cx,
            pose_centroid_y=pose_cy,
            pixels_per_mm=imports.pixels_per_mm_for_session(manifest, sid),
            individual_overlays=individual_overlays,
        )

        local_done = 0

        def _local_progress(d: int, _t: int) -> None:
            nonlocal local_done
            delta = int(d) - int(local_done)
            if delta > 0:
                local_done = int(d)
                _emit(delta)

        result = clip_svc.extract_selected_clips(
            sess_windows, cfg, progress_callback=_local_progress, cancel_flag=cancel_flag,
        )
        if local_done < len(sess_windows):
            _emit(len(sess_windows) - local_done)

        if result.clips:
            extracted_sessions.append(sid)
            all_clips.extend(result.clips)
        for warning in result.warnings:
            summary["warnings"].append(f"{sid}: {warning}")

    # Merge newly written clips into the existing manifest (never overwrite).
    if all_clips:
        existing = clip_svc.load_manifest()
        by_id: dict[str, ClipAsset] = {}
        session_ids: list[str] = []
        if existing is not None:
            for c in existing.clips:
                by_id[str(c.clip_id)] = c
            session_ids = list(existing.session_ids)
        for c in all_clips:
            by_id[str(c.clip_id)] = c
        session_ids = list(dict.fromkeys(session_ids + extracted_sessions))
        merged = ClipManifest(
            session_ids=session_ids,
            preset_name=(existing.preset_name if existing is not None else preset.name),
            clips=list(by_id.values()),
            total_windows=len(by_id),
            opencv_available=_has_cv2(),
            warnings=list(summary["warnings"]),
        )
        clip_svc.save_manifest(merged)

    summary["extracted"] = len(all_clips)
    summary["session_ids"] = extracted_sessions
    return summary


# ---------------------------------------------------------------------------

# Backwards-compatibility alias â€” remove once Phase 3 tabs are wired

# ---------------------------------------------------------------------------

PreprocessingService = ClipExtractionService

