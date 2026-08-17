"""Per-segment R3D-18 appearance features — a *video* feature family.

These are pixel-derived features in exactly the sense
:mod:`abel.validation.features` means it: they exist only because a camera saw
the animal, and they die with the video.  They are computed **before training**
and merged into ``segment_features.parquet`` as ordinary numeric columns
(``r3d_000``…``r3d_511``) so the classifier learns what to do with them, the
ablation machinery can switch them off, and the validation suite scores them
alongside optical flow and surface motion.

Design notes, each of which was measured rather than assumed:

* **The crop follows the animal.**  Each sampled frame is cropped around that
  frame's pose centroid, with the box sized from the animal's own body extent
  (``_CROP_BODY_MULT`` × median distance from centroid to the furthest tracked
  keypoint).  A fixed frame-relative rectangle embeds whatever happens to sit
  at those coordinates — usually empty arena.  In a multi-animal session each
  individual gets its own centre track and its own box size, so a segment
  belonging to ``Mouse2`` is not embedded from a crop around ``Mouse1``.
* **The full trunk runs.**  Pooling after ``layer4`` gives the 512-d semantic
  embedding.  Stopping earlier yields texture/motion primitives that carry no
  behaviour information.
* **The embedding stays a vector.**  Collapsing 512 dimensions to one scalar by
  a fixed formula discards the signal: on a labelled TMT digging set the 512-d
  vector alone reaches PR-AUC 0.79 against a 0.32 base rate, while a hand-rolled
  scalar over the same network scored at chance.

Embeddings are cached per session under ``derived/r3d_features/`` keyed by
segment id plus the window geometry, so re-extraction with unchanged settings is
a parquet read.

Dense inference (:meth:`R3DFeatureService.attach_dense`) asks for a window every
``inference_step_seconds`` — five times as many windows as training builds.  It
gets them from an *anchor grid* rather than a clip per window, for reasons that
were measured on the manuscript projects rather than assumed:

* **Anchors sit on the training stride.**  The embedding summarises a 15-frame
  window and does not move meaningfully in 3 frames: interpolating onto the
  dense grid from anchors one training-stride apart costs 5x fewer forwards for
  a probability MAE of 0.005, against 0.064 for the zero fill that dense
  inference used before this path existed.  Anchors land exactly on segments
  already in the cache above, so a session that was part of training needs no
  forward pass at all.
* **Each frame is cropped once.**  Dispatching a decoded frame to every window
  that samples it cropped and resized it 5.3x over and materialised a 3.7 GB
  clip buffer per session.  A per-frame crop stack is a fifth of the size.
* **The clip convention is preserved exactly.**  Anchors are ordinary 16-frame
  clips, so this is a drop-in for models already trained on ``r3d_*``.  Running
  the trunk fully convolutionally over a long sequence would be cheaper still,
  but it is a *different feature*: a 16-frame clip yields only two ``layer4``
  temporal positions and both are dominated by that clip's own zero padding
  (giving it real context instead moves the value by rel L2 0.37).  Adopting it
  would mean re-extracting and retraining everything.
"""

from __future__ import annotations

import logging
import threading
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseProcessingService

logger = logging.getLogger("abel")

R3D_PREFIX = "r3d_"
R3D_DIMS = 512
N_SAMPLE_FRAMES = 16
# fp16 + channels_last_3d holds a 64-clip batch in the same memory fp32 needed
# for 32, and is 3.0x faster on an RTX 4070 for rel L2 4e-4.
_BATCH = 64
_CROP_BODY_MULT = 2.5
_CROP_MIN_PX = 96
_CACHE_VERSION = "r3d_v1"
# Frames held as 112x112 crops at once during the dense pass (~310 MB).
_STACK_CHUNK = 8192

_IMAGENET_MEAN = (0.43216, 0.394666, 0.37645)
_IMAGENET_STD = (0.22803, 0.22145, 0.216989)

# Serialise GPU forwards so concurrent worker threads can't collide on CUDA.
_gpu_lock = threading.Lock()
_model_lock = threading.Lock()
_MODELS: dict[str, Any] = {}


def r3d_columns() -> list[str]:
    """Canonical column names, in order."""
    return [f"{R3D_PREFIX}{i:03d}" for i in range(R3D_DIMS)]


def is_r3d_column(name: str) -> bool:
    return str(name).startswith(R3D_PREFIX)


def _get_model(device_name: str):
    """Return a cached, eval-mode R3D-18 for ``device_name`` (loads once)."""
    with _model_lock:
        model = _MODELS.get(device_name)
        if model is None:
            import torch
            from torchvision.models.video import R3D_18_Weights, r3d_18

            model = r3d_18(weights=R3D_18_Weights.DEFAULT).to(torch.device(device_name)).eval()
            _MODELS[device_name] = model
        return model


class R3DUnavailable(RuntimeError):
    """Raised when torch/torchvision or the pretrained weights aren't usable."""


def _forward_batch(batch: np.ndarray, device_name: str) -> np.ndarray:
    """``(n, 16, 112, 112, 3)`` uint8 → ``(n, 512)`` float32 on one device.

    CUDA runs fp16 under autocast in ``channels_last_3d``; the CPU fallback stays
    fp32 in contiguous layout, where neither helps.
    """
    import torch

    dev = torch.device(device_name)
    model = _get_model(device_name)
    half = device_name == "cuda"
    mean = torch.tensor(_IMAGENET_MEAN, device=dev).view(1, 3, 1, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=dev).view(1, 3, 1, 1, 1)

    if not batch.flags["C_CONTIGUOUS"]:
        batch = np.ascontiguousarray(batch)
    x = torch.from_numpy(batch).to(dev).permute(0, 4, 1, 2, 3).float().div_(255.0)
    x = (x - mean) / std
    if half:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=half):
        f = model.stem(x)
        f = model.layer1(f)
        f = model.layer2(f)
        f = model.layer3(f)
        f = model.layer4(f)
        pooled = f.float().mean(dim=(2, 3, 4))
    return pooled.detach().cpu().numpy().astype(np.float32)


class R3DFeatureService:
    """Compute and cache per-segment R3D-18 appearance embeddings."""

    def __init__(self) -> None:
        self._imports = ImportService()
        self._pose = PoseProcessingService()

    # ── Public API ───────────────────────────────────────────────────────
    def attach(
        self,
        project_root: Path,
        segment_df: pd.DataFrame,
        *,
        progress_cb: Callable[[str], None] | None = None,
        strict: bool = False,
    ) -> pd.DataFrame:
        """Return ``segment_df`` with ``r3d_*`` columns merged on ``segment_id``.

        Sessions whose video or pose can't be reached are skipped and their rows
        left as NaN — the trainer's dead-feature pass handles an all-NaN column,
        and a partially covered project still benefits from the sessions that
        did resolve.  With ``strict`` the first failure raises instead.
        """

        def _log(msg: str) -> None:
            if progress_cb is not None:
                progress_cb(msg)

        required = {"segment_id", "session_id", "start_frame", "end_frame"}
        missing = required - set(segment_df.columns)
        if missing:
            raise ValueError(f"segment_df is missing {sorted(missing)}")
        if segment_df.empty:
            return segment_df

        manifest = self._imports.load_manifest(project_root)
        if manifest is None:
            _log("R3D features: no import manifest; skipping.")
            return segment_df

        frames: list[pd.DataFrame] = []
        sessions = [str(s) for s in segment_df["session_id"].astype(str).unique()]
        n_ok = 0
        for i, session_id in enumerate(sessions, 1):
            grp = segment_df.loc[segment_df["session_id"].astype(str) == session_id]
            _log(f"R3D features: session {i}/{len(sessions)} ({len(grp)} segments)…")
            try:
                frames.append(self._session_embeddings(project_root, manifest, session_id, grp))
                n_ok += 1
            except Exception as exc:
                if strict:
                    raise
                logger.warning("R3D features skipped for %s: %s", session_id, exc)
                _log(f"R3D features: skipped {session_id} ({exc}).")

        if not frames:
            _log("R3D features: no session produced embeddings.")
            return segment_df
        emb = pd.concat(frames, ignore_index=True)
        out = segment_df.merge(emb, on="segment_id", how="left")
        _log(f"R3D features: {len(emb)} segments embedded across {n_ok}/{len(sessions)} sessions.")
        return out

    # ── Per-session extraction (cached) ──────────────────────────────────
    def _cache_path(self, project_root: Path, session_id: str) -> Path:
        return project_root / "derived" / "r3d_features" / f"{session_id}.parquet"

    def _session_embeddings(
        self, project_root: Path, manifest: Any, session_id: str, grp: pd.DataFrame
    ) -> pd.DataFrame:
        cols = r3d_columns()
        seg_ids = grp["segment_id"].astype(str).tolist()
        starts = grp["start_frame"].to_numpy(dtype=int)
        ends = grp["end_frame"].to_numpy(dtype=int)

        cache_path = self._cache_path(project_root, session_id)
        cached = pd.DataFrame()
        if cache_path.exists():
            try:
                cached = pd.read_parquet(cache_path)
                if str(cached.attrs.get("cache_version", _CACHE_VERSION)) != _CACHE_VERSION:
                    cached = pd.DataFrame()
            except Exception:
                cached = pd.DataFrame()
        have = set(cached["segment_id"].astype(str)) if len(cached) else set()
        todo = [i for i, sid in enumerate(seg_ids) if sid not in have]
        if not todo:
            return cached[cached["segment_id"].astype(str).isin(seg_ids)][["segment_id", *cols]]

        video_path = self._imports.video_path_for_session(manifest, session_id)
        if not video_path or not Path(video_path).exists():
            raise FileNotFoundError(f"video not reachable for {session_id}")
        pose_path = self._imports.pose_path_for_session(manifest, session_id)
        if not pose_path or not Path(pose_path).exists():
            raise FileNotFoundError(f"pose not reachable for {session_id}")

        tracks, animal_row = self._session_geometry(manifest, session_id, Path(pose_path), grp)
        clips, kept = self._decode_clips(
            Path(video_path), starts[todo], ends[todo], tracks, animal_row[todo]
        )
        if not kept:
            raise RuntimeError("no clip could be decoded")
        emb = self._embed(clips)

        fresh = pd.DataFrame(emb, columns=cols)
        fresh.insert(0, "segment_id", [seg_ids[todo[k]] for k in kept])
        merged = pd.concat([cached, fresh], ignore_index=True) if len(cached) else fresh
        merged = merged.drop_duplicates(subset="segment_id", keep="last")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            merged.attrs["cache_version"] = _CACHE_VERSION
            merged.to_parquet(cache_path, index=False)
        except Exception as exc:  # cache is an optimisation, never a hard failure
            logger.debug("Could not write R3D cache for %s: %s", session_id, exc)
        return merged[merged["segment_id"].astype(str).isin(seg_ids)][["segment_id", *cols]]

    # ── Crop geometry from pose ──────────────────────────────────────────
    def _session_geometry(
        self, manifest: Any, session_id: str, pose_path: Path, grp: pd.DataFrame
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], np.ndarray]:
        """Per-animal crop tracks plus the track each segment in ``grp`` uses.

        A single-animal session yields one track and every segment points at it.
        A multi-animal session yields one track per individual, keyed by the same
        ``animal_id`` the pose/context feature tables use, so each segment crops
        around the animal it actually describes.  Pose is loaded with the
        project's own smoothing settings and identity corrections — the same
        tracks the rest of the feature pipeline sees.
        """
        session = next(
            (s for s in getattr(manifest, "linked_sessions", []) if str(s.session_id) == str(session_id)),
            None,
        )
        smoothing = getattr(manifest, "smoothing_settings", None)
        individuals = list(getattr(session, "individuals", None) or []) if session else []

        if not individuals:
            pose = self._pose.load_and_clean(pose_path, settings=smoothing)
            return [self._crop_geometry(pose)], np.zeros(len(grp), dtype=int)

        subject_map = dict(getattr(session, "individual_subject_map", None) or {})
        fallback_subject = getattr(session, "subject_id", None) or session_id
        multi = self._pose.load_and_clean_multi(
            pose_path,
            settings=smoothing,
            identity_corrections=list(getattr(session, "identity_corrections", None) or []),
        )
        tracks: list[tuple[np.ndarray, np.ndarray, int]] = []
        row_of_animal: dict[str, int] = {}
        for ind, pose in multi.per_individual.items():
            animal_id = subject_map.get(ind) or f"{fallback_subject}:{ind}"
            row_of_animal[str(animal_id)] = len(tracks)
            tracks.append(self._crop_geometry(pose))

        # Segments carry the animal_id the representation was built with.  An id
        # the pose file doesn't know (renamed subject, legacy cache) falls back
        # to the first track rather than dropping the segment.
        if "animal_id" in grp.columns and row_of_animal:
            animal_row = np.array(
                [row_of_animal.get(str(a), 0) for a in grp["animal_id"].astype(str)],
                dtype=int,
            )
            unknown = {
                str(a) for a in grp["animal_id"].astype(str).unique()
                if str(a) not in row_of_animal
            }
            if unknown:
                logger.warning(
                    "R3D features: %s has segments for unmapped animal(s) %s; "
                    "cropping them around %s.",
                    session_id, sorted(unknown), next(iter(row_of_animal)),
                )
        else:
            animal_row = np.zeros(len(grp), dtype=int)
        return tracks, animal_row

    @staticmethod
    def _crop_geometry(pose: Any) -> tuple[np.ndarray, np.ndarray, int]:
        """Per-frame crop centre plus a single session-wide crop side in px."""
        cx = np.asarray(pose.centroid_x, dtype=float)
        cy = np.asarray(pose.centroid_y, dtype=float)
        xs = pose.x.to_numpy(dtype=float)
        ys = pose.y.to_numpy(dtype=float)
        # Frames where tracking dropped out are all-NaN; that is expected and the
        # median below ignores them, so don't let numpy narrate it per session.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            extent = np.nanmax(
                np.sqrt((xs - cx[:, None]) ** 2 + (ys - cy[:, None]) ** 2), axis=1
            )
        med = float(np.nanmedian(extent)) if np.isfinite(extent).any() else 0.0
        side = max(_CROP_MIN_PX, int(round(_CROP_BODY_MULT * med))) if med > 0 else _CROP_MIN_PX
        # Carry the centre across untracked frames so every sampled frame crops
        # somewhere sensible rather than at (0, 0).
        cx = pd.Series(cx).ffill().bfill().to_numpy()
        cy = pd.Series(cy).ffill().bfill().to_numpy()
        if not np.isfinite(cx).any():
            cx = np.zeros_like(cx)
            cy = np.zeros_like(cy)
        return cx, cy, side

    # ── Decode (one sequential pass per video) ───────────────────────────
    @staticmethod
    def _decode_clips(
        video_path: Path,
        starts: np.ndarray,
        ends: np.ndarray,
        tracks: list[tuple[np.ndarray, np.ndarray, int]],
        animal_row: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        """Decode ``(n, 16, 112, 112, 3)`` uint8 crops in one forward pass.

        Frames are read in ascending order and dispatched to every segment that
        wants them, so a video is decoded once no matter how much its segments
        overlap — including when the segments belong to different animals in the
        same video, each with its own crop track (``tracks[animal_row[i]]``).
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video {video_path}")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_side = max(1, min(width, height))
            sides = [int(np.clip(s, _CROP_MIN_PX, max_side)) for _, _, s in tracks]
            n_pose = min(
                (int(min(len(tcx), len(tcy))) for tcx, tcy, _ in tracks), default=0
            )
            limit = max(0, (min(n_video, n_pose) if n_pose else n_video) - 1)

            n_seg = len(starts)
            idx = np.empty((n_seg, N_SAMPLE_FRAMES), dtype=np.int64)
            for i in range(n_seg):
                n_total = max(1, int(ends[i]) - int(starts[i]) + 1)
                idx[i] = int(starts[i]) + np.linspace(
                    0, n_total - 1, num=N_SAMPLE_FRAMES, dtype=int
                )
            idx = np.clip(idx, 0, limit)

            wanted: dict[int, list[tuple[int, int]]] = {}
            for i in range(n_seg):
                for s in range(N_SAMPLE_FRAMES):
                    wanted.setdefault(int(idx[i, s]), []).append((i, s))

            clips = np.zeros((n_seg, N_SAMPLE_FRAMES, 112, 112, 3), dtype=np.uint8)
            filled = np.zeros(n_seg, dtype=int)
            lo, hi = int(idx.min()), int(idx.max())
            cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
            frame_no = lo
            while frame_no <= hi:
                ok, frame = cap.read()
                if not ok:
                    break
                for i, s in wanted.get(frame_no, ()):
                    f = int(idx[i, s])
                    row = int(animal_row[i])
                    cx, cy, _ = tracks[row]
                    side = sides[row]
                    x0 = int(np.clip(cx[f] - side / 2.0, 0, max(0, width - side)))
                    y0 = int(np.clip(cy[f] - side / 2.0, 0, max(0, height - side)))
                    crop = frame[y0 : y0 + side, x0 : x0 + side]
                    if crop.size == 0:
                        continue
                    clips[i, s] = cv2.resize(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                        (112, 112),
                        interpolation=cv2.INTER_AREA,
                    )
                    filled[i] += 1
                frame_no += 1
        finally:
            cap.release()

        # A segment needs most of its frames to be a fair embedding.
        kept = [i for i in range(len(starts)) if filled[i] >= N_SAMPLE_FRAMES // 2]
        return clips[kept], kept

    # ── Dense inference: anchor grid + interpolation ─────────────────────
    def attach_dense(
        self,
        project_root: Path,
        dense_df: pd.DataFrame,
        session_id: str,
        *,
        manifest: Any = None,
        window_frames: int = 15,
        anchor_stride: int | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        """Return ``dense_df`` with ``r3d_*`` columns for every dense window.

        Embeddings are computed on anchors ``anchor_stride`` frames apart (the
        training stride by default, so anchors coincide with rows already in
        ``derived/r3d_features/``) and linearly interpolated onto each window's
        centre.  A session whose video or pose can't be reached is returned
        unchanged, leaving the caller's existing zero fill in place — a degraded
        trace for one session, not a failed run.
        """
        if dense_df.empty or "start_frame" not in dense_df.columns:
            return dense_df
        stride = int(anchor_stride or window_frames)
        if stride < 1:
            stride = 1

        def _log(msg: str) -> None:
            if progress_cb is not None:
                progress_cb(msg)

        try:
            if manifest is None:
                manifest = self._imports.load_manifest(project_root)
            if manifest is None:
                raise FileNotFoundError("no import manifest")
            video_path = self._imports.video_path_for_session(manifest, session_id)
            if not video_path or not Path(video_path).exists():
                raise FileNotFoundError(f"video not reachable for {session_id}")
            pose_path = self._imports.pose_path_for_session(manifest, session_id)
            if not pose_path or not Path(pose_path).exists():
                raise FileNotFoundError(f"pose not reachable for {session_id}")

            tracks, animal_row = self._session_geometry(
                manifest, session_id, Path(pose_path), dense_df
            )
            # Dense windows for a session are built from a single animal's frame
            # table, so one crop track covers them all.
            cx, cy, side = tracks[int(animal_row[0]) if len(animal_row) else 0]

            starts = dense_df["start_frame"].to_numpy(dtype=int)
            n_pose = int(min(len(cx), len(cy)))
            anchors = self._anchor_frames(starts, window_frames, stride, n_pose)
            if not len(anchors):
                raise RuntimeError("no anchor fits the session")

            cached = self._cached_by_start(project_root, session_id, window_frames)
            emb = self._anchor_embeddings(
                Path(video_path), anchors, cx, cy, side, window_frames, cached, _log,
                session_id,
            )
        except Exception as exc:
            logger.warning("R3D dense features skipped for %s: %s", session_id, exc)
            _log(f"R3D dense features: skipped {session_id} ({exc}).")
            return dense_df

        cols = r3d_columns()
        centre = starts + (window_frames - 1) / 2.0
        anchor_centre = anchors + (window_frames - 1) / 2.0
        values = np.empty((len(starts), R3D_DIMS), dtype=np.float32)
        for d in range(R3D_DIMS):
            values[:, d] = np.interp(centre, anchor_centre, emb[:, d])
        return pd.concat(
            [dense_df.reset_index(drop=True),
             pd.DataFrame(values, columns=cols)],
            axis=1,
        )

    @staticmethod
    def _anchor_frames(
        starts: np.ndarray, window_frames: int, stride: int, n_pose: int
    ) -> np.ndarray:
        """Anchor start frames spanning the dense grid, clamped to tracked pose.

        The first and last dense windows are always anchors so no window is
        extrapolated beyond the computed range.
        """
        lo = int(starts.min())
        hi = int(starts.max())
        if n_pose:
            hi = min(hi, max(lo, n_pose - window_frames))
        anchors = np.arange(lo, hi + 1, stride, dtype=int)
        if not len(anchors):
            anchors = np.array([lo], dtype=int)
        if anchors[-1] != hi:
            anchors = np.append(anchors, hi)
        return anchors

    def _cached_by_start(
        self, project_root: Path, session_id: str, window_frames: int
    ) -> dict[int, np.ndarray]:
        """Cached training embeddings keyed by their window's start frame.

        Only segments whose window is exactly ``window_frames`` long qualify: a
        training set can hold more than one segment length, and a 30-frame
        embedding is not the same quantity as a 15-frame one.
        """
        cache_path = self._cache_path(project_root, session_id)
        seg_path = project_root / "derived" / "representations" / "segment_features.parquet"
        if not cache_path.exists() or not seg_path.exists():
            return {}
        try:
            seg = pd.read_parquet(
                seg_path, columns=["segment_id", "session_id", "start_frame", "end_frame"]
            )
            seg = seg[seg["session_id"].astype(str) == str(session_id)]
            lengths = seg["end_frame"].to_numpy(dtype=int) - seg["start_frame"].to_numpy(dtype=int) + 1
            seg = seg[lengths == int(window_frames)]
            if seg.empty:
                return {}
            start_of = dict(zip(seg["segment_id"].astype(str), seg["start_frame"].astype(int)))

            df = pd.read_parquet(cache_path)
            if str(df.attrs.get("cache_version", _CACHE_VERSION)) != _CACHE_VERSION:
                return {}
            cols = r3d_columns()
            if any(c not in df.columns for c in cols):
                return {}
            rows = df[cols].to_numpy(dtype=np.float32)
            out: dict[int, np.ndarray] = {}
            for sid, row in zip(df["segment_id"].astype(str), rows):
                start = start_of.get(sid)
                if start is not None:
                    out[int(start)] = row
            return out
        except Exception as exc:
            logger.debug("Could not read R3D cache for %s: %s", session_id, exc)
            return {}

    def _anchor_embeddings(
        self,
        video_path: Path,
        anchors: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
        side: int,
        window_frames: int,
        cached: dict[int, np.ndarray],
        log: Callable[[str], None],
        session_id: str,
    ) -> np.ndarray:
        """Embed every anchor, reusing cached rows and decoding the rest once."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video {video_path}")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        side = int(np.clip(side, _CROP_MIN_PX, max(1, min(width, height))))

        emb = np.zeros((len(anchors), R3D_DIMS), dtype=np.float32)
        filled = np.zeros(len(anchors), dtype=bool)
        for i, a in enumerate(anchors):
            row = cached.get(int(a))
            if row is not None:
                emb[i] = row
                filled[i] = True
        todo = anchors[~filled]
        if not len(todo):
            log(f"R3D dense features: {session_id} fully covered by the training cache.")
            return emb

        log(
            f"R3D dense features: {session_id} embedding {len(todo)} anchors "
            f"({len(anchors) - len(todo)} reused)…"
        )
        offsets = np.linspace(0, window_frames - 1, num=N_SAMPLE_FRAMES, dtype=int)
        limit = max(0, (n_video if n_video > 0 else int(todo.max()) + window_frames) - 1)
        lo = int(todo.min())
        hi = int(min(limit, int(todo.max()) + window_frames - 1))
        embedded = 0
        for chunk_start in range(lo, hi + 1, _STACK_CHUNK):
            chunk_end = min(hi, chunk_start + _STACK_CHUNK - 1)
            sel = todo[(todo >= chunk_start) & (todo + window_frames - 1 <= chunk_end)]
            if not len(sel):
                continue
            stack = self._crop_stack(
                video_path, chunk_start, chunk_end, cx, cy, side, width, height
            )
            idx = np.clip(
                sel[:, None] + offsets[None, :] - chunk_start, 0, chunk_end - chunk_start
            )
            rows = np.searchsorted(anchors, sel)
            emb[rows] = self._embed(stack[idx])
            filled[rows] = True
            embedded += len(sel)
        if embedded == 0:
            raise RuntimeError("no anchor could be decoded")

        # A video shorter than its pose table leaves trailing anchors undecoded.
        # Carrying the nearest real embedding into them keeps the interpolation
        # honest; leaving them as zero vectors would drag every window near the
        # end of the session toward the very zero fill this path exists to undo.
        if not filled.all():
            good = np.flatnonzero(filled)
            nearest = good[np.argmin(np.abs(good[None, :] - np.flatnonzero(~filled)[:, None]), axis=1)]
            emb[~filled] = emb[nearest]
            logger.debug(
                "R3D dense features: %s carried %d undecodable anchors from neighbours.",
                session_id, int((~filled).sum()),
            )
        return emb

    @staticmethod
    def _crop_stack(
        video_path: Path,
        lo: int,
        hi: int,
        cx: np.ndarray,
        cy: np.ndarray,
        side: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Decode frames ``[lo, hi]`` once, cropping and resizing each exactly once."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video {video_path}")
        stack = np.zeros((hi - lo + 1, 112, 112, 3), dtype=np.uint8)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
            n_track = int(min(len(cx), len(cy)))
            for k in range(hi - lo + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                f = min(lo + k, n_track - 1) if n_track else 0
                x0 = int(np.clip(cx[f] - side / 2.0, 0, max(0, width - side)))
                y0 = int(np.clip(cy[f] - side / 2.0, 0, max(0, height - side)))
                crop = frame[y0 : y0 + side, x0 : x0 + side]
                if crop.size == 0:
                    continue
                stack[k] = cv2.resize(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (112, 112),
                    interpolation=cv2.INTER_AREA,
                )
        finally:
            cap.release()
        return stack

    # ── Batched forward ──────────────────────────────────────────────────
    @staticmethod
    def _embed(clips: np.ndarray) -> np.ndarray:
        """``(n, 16, 112, 112, 3)`` uint8 → ``(n, 512)`` float32."""
        try:
            import torch
        except Exception as exc:
            raise R3DUnavailable(f"torch unavailable: {exc}") from exc

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        out: list[np.ndarray] = []
        for start in range(0, len(clips), _BATCH):
            batch = clips[start : start + _BATCH]
            acquired = False
            try:
                if device_name == "cuda":
                    acquired = _gpu_lock.acquire(timeout=120.0)
                    if not acquired:
                        device_name = "cpu"
                out.append(_forward_batch(batch, device_name))
            except Exception as exc:
                if device_name == "cuda":
                    device_name = "cpu"
                    logger.warning("R3D GPU forward failed (%s); falling back to CPU.", exc)
                    out.append(_forward_batch(batch, "cpu"))
                else:
                    raise R3DUnavailable(str(exc)) from exc
            finally:
                if acquired:
                    _gpu_lock.release()
        return np.concatenate(out, axis=0) if out else np.zeros((0, R3D_DIMS), dtype=np.float32)
