"""Convert SLEAP prediction files (``.slp``) into DeepLabCut-format pose files.

ABEL's pose pipeline reads DeepLabCut layouts only (a pandas ``DataFrame`` with
``scorer/individuals/bodyparts/coords`` columns, stored under the HDF5 key
``df_with_missing``).  SLEAP's native ``.slp`` — and even its ``analysis.h5``
export — use a different structure, so this module bridges the two: it reads a
``.slp`` with :mod:`sleap_io` and writes a DLC ``.h5`` that flows through the
existing importer unchanged (probe, keypoint mapping, multi-animal identity,
features, analytics — everything DLC files get).

The output matches DeepLabCut's multi-animal HDF5 exactly:

* one row per frame from 0 (missing frames / points are ``NaN``),
* coords named ``x`` / ``y`` / ``likelihood`` (SLEAP point score → likelihood),
* ``format="table"`` under key ``df_with_missing`` (what ABEL's reader and the
  ``stop=0`` header probe expect).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("abel")

SLEAP_POSE_EXTENSIONS = {".slp"}


def is_sleap_pose_file(path: str | Path) -> bool:
    """True when ``path`` is a SLEAP predictions/labels file (``.slp``)."""
    return Path(path).suffix.lower() in SLEAP_POSE_EXTENSIONS


def default_converted_path(slp_path: str | Path) -> Path:
    """DLC ``.h5`` path for a converted ``.slp`` (sibling ``*.sleap.h5``).

    Keeps the leading part of the SLEAP filename (which embeds the original
    video name) so ABEL's filename auto-matching can still pair it to a video.
    """
    slp_path = Path(slp_path)
    stem = slp_path.name
    if stem.lower().endswith(".slp"):
        stem = stem[: -len(".slp")]
    return slp_path.with_name(f"{stem}.sleap.h5")


def convert_slp_to_dlc(
    slp_path: str | Path,
    out_path: str | Path | None = None,
    *,
    scorer: str = "SLEAP",
    video_index: int = 0,
    max_individuals: int = 12,
) -> Path:
    """Convert a SLEAP ``.slp`` file to a DeepLabCut-format ``.h5``.

    Args:
        slp_path: Source ``.slp`` predictions file.
        out_path: Destination ``.h5``.  Defaults to :func:`default_converted_path`.
        scorer: Value for the DLC ``scorer`` column level (cosmetic; ABEL drops it).
        video_index: Which video in the ``.slp`` to export (per-video prediction
            files contain a single video; combined files are filtered to this one).
        max_individuals: Guard against files tracked *without* a max-tracks limit,
            which fragment one animal into hundreds of short tracks.  Such files
            can't be mapped to fixed identities and would overflow the HDF5
            format, so conversion is refused above this many distinct tracks.

    Returns:
        The path to the written DLC ``.h5`` file.

    Raises:
        ValueError: If the file has no skeleton/nodes, no frames for the video,
            or more than ``max_individuals`` distinct tracks.
    """
    import sleap_io as sio  # local import: only needed when converting

    slp_path = Path(slp_path)
    labels = sio.load_slp(str(slp_path))

    if not labels.skeletons or not labels.skeletons[0].node_names:
        raise ValueError(f"No skeleton/nodes found in {slp_path.name}")
    nodes = list(labels.skeletons[0].node_names)
    n_nodes = len(nodes)

    # Restrict to a single video. Per-video prediction files have exactly one;
    # a combined file is filtered to the requested index.
    videos = list(labels.videos)
    target_video = None
    if videos:
        idx = video_index if 0 <= video_index < len(videos) else 0
        target_video = videos[idx]
    frames = [
        lf
        for lf in labels.labeled_frames
        if target_video is None or lf.video is target_video
    ]
    if not frames:
        raise ValueError(f"No predicted frames for video {video_index} in {slp_path.name}")

    n_frames = int(max(lf.frame_idx for lf in frames)) + 1

    # Individuals come from track identities.  Seed the ordering from the
    # skeleton's declared tracks so column order is stable across files.
    per_ind: dict[str, np.ndarray] = {}

    def _blank() -> np.ndarray:
        return np.full((n_frames, n_nodes, 3), np.nan, dtype=float)

    for t in labels.tracks:
        per_ind[t.name] = _blank()

    untracked_seen = False
    for lf in frames:
        fi = int(lf.frame_idx)
        if not (0 <= fi < n_frames):
            continue
        for slot, inst in enumerate(lf.instances):
            if inst.track is not None:
                ind = inst.track.name
            else:
                # Untracked instance: fall back to positional slot so a
                # tracker-less file still yields per-animal columns.
                ind = f"track_{slot}"
                untracked_seen = True
            arr = per_ind.get(ind)
            if arr is None:
                arr = _blank()
                per_ind[ind] = arr
            pts = inst.numpy(scores=True)  # (n_nodes, 3): x, y, score
            if pts.shape[0] == n_nodes:
                arr[fi] = pts

    if not per_ind:
        raise ValueError(f"No instances found to convert in {slp_path.name}")
    if untracked_seen and labels.tracks:
        logger.warning(
            "%s has frames with untracked instances; identities may be inconsistent.",
            slp_path.name,
        )

    individuals = list(per_ind.keys())
    if len(individuals) > max_individuals:
        raise ValueError(
            f"{slp_path.name} has {len(individuals)} tracks - it looks like SLEAP "
            "tracking ran without a max-tracks limit, fragmenting animals into many "
            "short tracks. Re-run tracking with a max-tracks / target-instance-count "
            f"set (e.g. the number of animals), then convert. (limit: {max_individuals})"
        )
    # Emit the multi-animal (4-level) layout whenever tracks exist or more than
    # one individual was found; otherwise a single-animal (3-level) DLC file.
    multi = len(individuals) > 1 or bool(labels.tracks)
    coords = ["x", "y", "likelihood"]

    col_tuples: list[tuple] = []
    columns_data: list[np.ndarray] = []
    for ind in individuals:
        arr = per_ind[ind]
        for ni, node in enumerate(nodes):
            for ci, coord in enumerate(coords):
                col_tuples.append((scorer, ind, node, coord) if multi else (scorer, node, coord))
                columns_data.append(arr[:, ni, ci])

    names = ["scorer", "individuals", "bodyparts", "coords"] if multi else ["scorer", "bodyparts", "coords"]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=names)
    data = np.column_stack(columns_data) if columns_data else np.empty((n_frames, 0))
    df = pd.DataFrame(data, index=pd.RangeIndex(n_frames), columns=columns)

    if out_path is None:
        out_path = default_converted_path(slp_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically (temp + replace) so a failed write never leaves a partial
    # file that a later import could reuse. Table format under 'df_with_missing'
    # keeps ABEL's reader and its stop=0 header probe both working.
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        df.to_hdf(str(tmp_path), key="df_with_missing", format="table", mode="w")
        os.replace(str(tmp_path), str(out_path))
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    logger.info(
        "Converted SLEAP %s -> DLC %s (%d frames, %d nodes, %d individual(s))",
        slp_path.name, out_path.name, n_frames, n_nodes, len(individuals),
    )
    return out_path
