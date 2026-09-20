"""Tell tracks apart by how the animals *look*, not by where they jump.

The position-based swap detector (``PoseProcessingService.detect_identity_swaps``)
can only see a swap as a discontinuity, so it misses exchanges that happen while
the animals are touching and it fires on every occlusion.  When the animals
differ in coat color, a black and a white mouse, the common social-assay pair,
the video answers the question directly: sample the pixels at each track's
centroid and the darker track *is* the black mouse, frame after frame.

The measurement is a signed brightness difference between the two tracks,
``diff = mean_grey(track_a) - mean_grey(track_b)``.  Its sign is the identity
assignment; a sustained sign change is a swap.  Sessions whose animals are not
visually distinct show a ``separability`` near zero, and this module says so
rather than inventing corrections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Gray levels (0-255) by which two tracks must differ to count as distinguishable.
CONFIDENT_DIFF = 25.0


@dataclass
class AppearanceIdentityResult:
    """What the pixels say about a session's two tracks."""

    individuals: "list[str]" = field(default_factory=list)
    frames: "list[int]" = field(default_factory=list)
    diff: "list[float]" = field(default_factory=list)
    median_abs_diff: float = 0.0
    separability: float = 0.0
    """Fraction of sampled frames where the two tracks differ confidently."""
    corrections: "list[dict]" = field(default_factory=list)
    """Swap corrections implied by the appearance track, ready to apply."""
    consistency_before: float = 0.0
    consistency_after: float = 0.0
    """Share of sampled frames on one consistent assignment, before/after them."""
    message: str = ""

    @property
    def usable(self) -> bool:
        """True when the animals are distinct enough to trust this."""
        return self.separability >= 0.6 and self.median_abs_diff >= CONFIDENT_DIFF


def analyze_appearance_identity(
    video_path: "Path | str",
    multi,
    *,
    sample_every: int = 15,
    patch_radius: int = 4,
    min_run_samples: int = 5,
    max_samples: int = 4000,
    progress_cb=None,
) -> AppearanceIdentityResult:
    """Sample coat brightness at each track's centroid and derive swap corrections.

    ``sample_every`` frames are decoded (the rest are skipped without decoding),
    so the cost is roughly a fast-forward pass over the video.  ``min_run_samples``
    is the shortest run of a consistent assignment that counts as real, which is
    what keeps a one-second occlusion from being reported as two swaps.
    """
    import cv2  # noqa: PLC0415

    res = AppearanceIdentityResult()
    inds = list(getattr(multi, "individuals", []) or [])[:2]
    if len(inds) < 2:
        res.message = "Appearance check needs exactly two tracked animals."
        return res
    res.individuals = inds

    cx = {i: np.asarray(multi.per_individual[i].centroid_x, dtype=float) for i in inds}
    cy = {i: np.asarray(multi.per_individual[i].centroid_y, dtype=float) for i in inds}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        res.message = "Could not open the session video for the appearance check."
        return res
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n = min(int(multi.n_frames), n_video) if n_video > 0 else int(multi.n_frames)
    step = max(1, int(sample_every))
    if n // step > max_samples:
        step = max(step, n // max_samples)

    frames: list[int] = []
    values: dict[str, list[float]] = {i: [] for i in inds}
    try:
        idx = 0
        while idx < n:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, img = cap.retrieve()
                if ok:
                    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    h, w = grey.shape
                    frames.append(idx)
                    for i in inds:
                        values[i].append(_patch_mean(grey, cx[i], cy[i], idx, w, h, patch_radius))
                    if progress_cb is not None and len(frames) % 50 == 0:
                        progress_cb(idx, n)
            idx += 1
    finally:
        cap.release()

    if len(frames) < 4 * min_run_samples:
        res.message = "Too few frames could be sampled from the video."
        return res

    f = np.asarray(frames, dtype=int)
    diff = np.asarray(values[inds[0]], dtype=float) - np.asarray(values[inds[1]], dtype=float)
    res.frames = f.tolist()
    res.diff = [float(v) for v in diff]

    confident = np.isfinite(diff) & (np.abs(diff) > CONFIDENT_DIFF)
    res.median_abs_diff = float(np.nanmedian(np.abs(diff))) if np.isfinite(diff).any() else 0.0
    res.separability = float(confident.mean())
    if not res.usable:
        res.message = (
            f"The two animals look too similar for this check "
            f"(only {res.separability:.0%} of frames tell them apart). "
            "Review the swaps by eye instead."
        )
        return res

    f_ok, sign = f[confident], np.sign(diff[confident])
    flips = _sustained_sign_changes(f_ok, sign, min_run_samples)
    res.corrections = [
        {"frame": int(fr), "a": inds[0], "b": inds[1]} for fr in flips
    ]
    res.consistency_before = _consistency(sign)
    parity = np.array([(-1.0) ** sum(1 for c in flips if c <= t) for t in f_ok])
    res.consistency_after = _consistency(sign * parity)
    res.message = (
        f"{len(flips)} swap(s) found. The animals are distinguishable in "
        f"{res.separability:.0%} of frames; applying these puts "
        f"{res.consistency_after:.0%} of the session on one consistent identity "
        f"(now {res.consistency_before:.0%})."
    )
    return res


def _patch_mean(grey, cx, cy, idx: int, w: int, h: int, radius: int) -> float:
    if idx >= len(cx) or idx >= len(cy):
        return float("nan")
    x, y = cx[idx], cy[idx]
    if not (np.isfinite(x) and np.isfinite(y)):
        return float("nan")
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    patch = grey[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size else float("nan")


def _sustained_sign_changes(frames: np.ndarray, sign: np.ndarray, min_run: int) -> "list[int]":
    """Frames where the assignment flips and *stays* flipped.

    Runs shorter than ``min_run`` samples are dropped rather than split around:
    a brief wrong-looking stretch is an animal climbing over the other, not two
    swaps a second apart.
    """
    runs: list[tuple[int, int, float]] = []
    start = 0
    for i in range(1, len(sign) + 1):
        if i == len(sign) or sign[i] != sign[start]:
            runs.append((start, i - 1, float(sign[start])))
            start = i
    keep = [r for r in runs if (r[1] - r[0] + 1) >= min_run]
    out: list[int] = []
    for prev, cur in zip(keep, keep[1:]):
        if cur[2] != prev[2]:
            out.append(int(frames[cur[0]]))
    return out


def _consistency(sign: np.ndarray) -> float:
    if len(sign) == 0:
        return 0.0
    majority = 1.0 if (sign > 0).mean() >= 0.5 else -1.0
    return float((sign == majority).mean())


def scan_sessions(
    manifest,
    import_service,
    pose_service,
    session_ids: "list[str] | None" = None,
    *,
    progress_cb=None,
    cancel_flag: "list[bool] | None" = None,
    **analyze_kwargs,
) -> "list[dict]":
    """Run the appearance check over every multi-animal session.

    Returns one row per session with what the pixels found and how it compares to
    the corrections already saved, so a project can be checked in one pass
    instead of opening each session by hand.  Sessions whose animals look alike
    come back with ``usable`` False and are left for manual review.
    """
    rows: list[dict] = []
    sessions = [
        s for s in getattr(manifest, "linked_sessions", [])
        if getattr(s, "individuals", None)
        and (session_ids is None or str(s.session_id) in set(session_ids))
    ]
    for i, session in enumerate(sessions):
        if cancel_flag and cancel_flag[0]:
            break
        label = getattr(session, "subject_id", None) or session.session_id
        if progress_cb is not None:
            progress_cb(i, len(sessions), str(label))
        saved = sorted(
            int(c.get("frame", 0)) for c in (getattr(session, "identity_corrections", None) or [])
        )
        row = {
            "session_id": str(session.session_id),
            "subject": str(label),
            "saved": saved,
            "detected": [],
            "usable": False,
            "separability": 0.0,
            "consistency_before": 0.0,
            "consistency_after": 0.0,
            "agrees": True,
            "message": "",
            "corrections": [],
        }
        rows.append(row)

        video_path = import_service.video_path_for_session(manifest, session.session_id)
        pose_path = import_service.pose_path_for_session(manifest, session.session_id)
        if not video_path or not pose_path:
            row["message"] = "Video or pose file could not be resolved."
            continue
        try:
            multi = pose_service.load_and_clean_multi(
                pose_path, getattr(manifest, "smoothing_settings", None)
            )
            result = analyze_appearance_identity(video_path, multi, **analyze_kwargs)
        except Exception as exc:  # pragma: no cover - unreadable inputs
            row["message"] = f"Could not check this session: {exc}"
            logger.warning("Appearance scan failed for %s: %s", session.session_id, exc)
            continue

        row.update(
            usable=result.usable,
            separability=result.separability,
            consistency_before=result.consistency_before,
            consistency_after=result.consistency_after,
            message=result.message,
            detected=[int(c["frame"]) for c in result.corrections],
            corrections=[dict(c) for c in result.corrections],
        )
        # Compare on *parity*, not on the exact frames: two corrections a few
        # frames apart cancel out, so a different list can still be the same
        # assignment.  Only a real disagreement is worth the user's attention.
        row["agrees"] = (not result.usable) or _same_assignment(
            saved, row["detected"], tolerance=45
        )
    if progress_cb is not None:
        progress_cb(len(sessions), len(sessions), "")
    return rows


def _same_assignment(a: "list[int]", b: "list[int]", tolerance: int = 45) -> bool:
    """True when two correction sets leave the session on the same identity.

    Compares the running parity of each set on a shared frame grid rather than
    the frames themselves, allowing ``tolerance`` frames of slack around each
    flip (the appearance check only samples every N frames).
    """
    if not a and not b:
        return True
    edges = sorted(set(a) | set(b))
    probes = [e - tolerance - 1 for e in edges] + [e + tolerance + 1 for e in edges]
    for t in probes:
        if t < 0:
            continue
        pa = sum(1 for f in a if f <= t) % 2
        pb = sum(1 for f in b if f <= t) % 2
        if pa != pb:
            return False
    return True


def probe_project_appearance(
    manifest,
    import_service,
    pose_service,
    *,
    max_sessions: int = 5,
    sample_every: int = 120,
    cancel_flag: "list[bool] | None" = None,
) -> dict:
    """Quick check: can appearance tell this project's animals apart at all?

    A full scan decodes every session; this samples a handful of sessions spread
    through the project at a much coarser step, so the answer arrives in seconds.
    It is the question worth asking first, with same-coat animals the whole
    approach is off the table and the swaps have to be judged by eye.

    Returns ``{"sessions": [...], "usable_fraction", "median_contrast",
    "verdict", "message"}``.
    """
    sessions = [
        s for s in getattr(manifest, "linked_sessions", [])
        if getattr(s, "individuals", None) and len(s.individuals) >= 2
    ]
    out: dict = {
        "sessions": [], "usable_fraction": 0.0, "median_contrast": 0.0,
        "verdict": "none", "message": "",
    }
    if not sessions:
        out["message"] = "No multi-animal sessions to check."
        return out

    # Spread the sample across the project rather than taking the first few, so
    # one odd recording session does not decide the answer.
    step = max(1, len(sessions) // max(1, max_sessions))
    picked = sessions[::step][:max_sessions]

    contrasts: list[float] = []
    for session in picked:
        if cancel_flag and cancel_flag[0]:
            break
        label = str(getattr(session, "subject_id", None) or session.session_id)
        row = {"subject": label, "usable": False, "separability": 0.0,
               "contrast": 0.0, "measured": False}
        out["sessions"].append(row)
        video_path = import_service.video_path_for_session(manifest, session.session_id)
        pose_path = import_service.pose_path_for_session(manifest, session.session_id)
        if not video_path or not pose_path:
            continue
        try:
            multi = pose_service.load_and_clean_multi(
                pose_path, getattr(manifest, "smoothing_settings", None)
            )
            res = analyze_appearance_identity(
                video_path, multi, sample_every=sample_every, min_run_samples=2,
            )
        except Exception as exc:  # pragma: no cover - unreadable inputs
            logger.warning("Appearance probe failed for %s: %s", session.session_id, exc)
            continue
        # ``measured`` separates "we looked and they are identical" from "we
        # could not read this session": both score zero contrast otherwise.
        row.update(
            usable=res.usable,
            separability=res.separability,
            contrast=res.median_abs_diff,
            measured=bool(res.frames),
        )
        if res.frames:
            contrasts.append(res.median_abs_diff)

    checked = [r for r in out["sessions"] if r["measured"]]
    if not checked:
        out["message"] = "None of the sampled sessions could be read."
        return out

    out["usable_fraction"] = sum(1 for r in checked if r["usable"]) / len(checked)
    out["median_contrast"] = float(np.median(contrasts)) if contrasts else 0.0
    if out["usable_fraction"] >= 0.8:
        out["verdict"] = "usable"
        out["message"] = (
            f"Appearance can tell the animals apart in {len(checked)} sampled "
            f"session(s), median contrast {out['median_contrast']:.0f} of 255 gray "
            "levels. A full scan will give reliable swap corrections."
        )
    elif out["usable_fraction"] > 0.0:
        out["verdict"] = "mixed"
        out["message"] = (
            f"Appearance works in only {out['usable_fraction']:.0%} of the sampled "
            f"sessions (median contrast {out['median_contrast']:.0f} of 255). A scan "
            "will judge the sessions it can and leave the rest for review by eye."
        )
    else:
        out["verdict"] = "unusable"
        out["message"] = (
            f"The animals look alike on camera (median contrast only "
            f"{out['median_contrast']:.0f} of 255 gray levels), so appearance cannot "
            "identify them. Swaps in this project have to be judged by eye."
        )
    return out
