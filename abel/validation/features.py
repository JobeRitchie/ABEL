"""Feature-family taxonomy — the single source of truth for every ablation.

Four families are separated so each can be ablated on its own:

* **pose** — the animal's own body: pose geometry and kinematics, derived purely
  from the tracked skeleton. This is the ablation *baseline*.
* **context** — the *environment*, and strictly its **geometry**: ROI/zone occupancy,
  distances and angles to objects/targets, ROI presence flags, arena walls and corners
  (``roi_*_present``, ``*_to_roi_*``, ``*_to_target_dist``, ``*_angle_to_*``,
  ``zone``/``arena``/``wall``/``corner``…).  These survive with pose alone; no camera
  is needed to compute them once the ROI is known.
* **video** — pixel-derived signal from ``context_feature_service`` (optical flow,
  surface/substrate motion, R3D embeddings) — *including* the ROI-anchored flow
  columns (``flow_mag_near_target``, ``flow_mag_near_roi_N``).  Those are optical
  flow: they exist only because a camera saw pixels move, and they die with the video.
  The ROI only says *where* the flow was sampled.  Bucketing them as context would let
  a "video off" ablation quietly keep reading pixels.
* **social** — inter-animal interaction features (``social_*``): distance to the
  nearest animal, approach velocity, heading alignment, contact state.

**Why context MUST be its own family (learned the hard way).** It used to be folded
into "pose", so the ablation baseline labelled "Pose only" was silently handed the
environment. On a novel-object project that is catastrophic: *Sniff Novel* vs
*Sniff Familiar* are the same motor act differing only in which object the animal
is at, and a single ``body_centroid_to_roi_2_dist`` column separates them at
AUC 0.999 on its own. The "pose-only" baseline scored **1.000** on that pair —
while true pose (ROI removed) scores **0.66**. Every gain the environment was
producing was being credited to pose, and the pairs that most needed interrogating
were reported as trivially solved.

The classifier is shared with :mod:`abel.validation.analyses.behaviorscape` (which
re-exports it) so the ablation bars and the behaviorscape modality bands can never
disagree about what a feature *is* — they did, and that divergence is what hid the
bug: behaviorscape already had a ``context`` modality that the ablation did not.
"""

from __future__ import annotations

import re

import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService

# ── Modalities ─────────────────────────────────────────────────────────────

MODALITY_POSE = "pose"
MODALITY_KINEMATICS = "kinematics"
MODALITY_VIDEO = "video"
MODALITY_CONTEXT = "context"
MODALITY_SOCIAL = "social"

MODALITY_ORDER = [
    MODALITY_POSE, MODALITY_KINEMATICS, MODALITY_VIDEO,
    MODALITY_CONTEXT, MODALITY_SOCIAL,
]

# Token-matched (not naive substring) so aggregation suffixes (_mean/_std/_energy)
# and embedded words don't cause misclassification.
_VIDEO_KEYS = (
    "flow", "surface", "r3d", "optical", "video", "pixel", "texture",
    # "local" — in this codebase a ``local`` column is always a pixel-neighbourhood
    # statistic from ContextFeatureService's frame pipeline, never a pose quantity:
    # local_surface_*, flow_entropy_local, and (the reason this key exists)
    # nose_local_change_rate / nose_local_variance, which are frame-differencing on a
    # tight crop around the nose tip.  Without it those two matched no key at all and
    # fell through to the POSE default — putting pixel-derived signal inside the
    # "pose only" ablation baseline, the exact bug the module docstring describes for
    # context.
    "local",
)
_KINEMATIC_KEYS = (
    "velocity", "speed", "accel", "jerk", "displacement", "distance",
    "angularvel", "momentum",
)
_CONTEXT_TOKEN_KEYS = ("roi", "zone", "arena", "wall", "corner", "env")
_CONTEXT_SUBSTR_KEYS = ("target", "occup")


def classify_modality(feature: str) -> str:
    """Map a feature column name to one of the five data modalities."""
    name = str(feature).lower()
    # Social carries an unambiguous prefix and must be caught first —
    # ``social_approach_velocity_*`` would otherwise read as kinematics.
    if name.startswith("social_"):
        return MODALITY_SOCIAL
    tokens = [t for t in re.split(r"[^a-z0-9]+", name) if t]

    def _has(keys: tuple[str, ...]) -> bool:
        return any(k in tok for tok in tokens for k in keys)

    # Video BEFORE context: ``flow_mag_near_roi_2`` is optical flow — it exists only
    # because a camera saw pixels move, and it dies with the video the way every other
    # flow column does.  The ROI merely says *where* the flow was sampled; it does not
    # make the measurement an environment descriptor.  So the "is video worth it?"
    # ablation must own it, and dropping video must drop it.  Only the *geometric* ROI
    # columns — distances, angles, presence — describe the environment, and none of
    # them carries a video token, so this ordering moves the flow columns and nothing
    # else.
    if _has(_VIDEO_KEYS):
        return MODALITY_VIDEO
    # Context BEFORE kinematics: ``nose_to_roi_2_dist`` is an object distance, not a
    # kinematic one.
    if _has(_CONTEXT_SUBSTR_KEYS) or any(
        tok == k or (tok.startswith(k) and tok[len(k):].isdigit())
        for tok in tokens for k in _CONTEXT_TOKEN_KEYS
    ):
        return MODALITY_CONTEXT
    if _has(_KINEMATIC_KEYS):
        return MODALITY_KINEMATICS
    return MODALITY_POSE


# ── Family predicates ──────────────────────────────────────────────────────


def is_video_feature(col: str) -> bool:
    return classify_modality(col) == MODALITY_VIDEO


def is_social_feature(col: str) -> bool:
    """True for inter-animal interaction features (``social_*`` prefix)."""
    return classify_modality(col) == MODALITY_SOCIAL


def is_context_feature(col: str) -> bool:
    """True for environment/ROI/object features (see the module docstring)."""
    return classify_modality(col) == MODALITY_CONTEXT


def is_pose_feature(col: str) -> bool:
    """True for the animal's own body: pose geometry + kinematics."""
    return classify_modality(col) in (MODALITY_POSE, MODALITY_KINEMATICS)


# ── Column selectors ───────────────────────────────────────────────────────


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """All numeric feature columns the trainer would consider (reuses its rule)."""
    return ActiveLearningTrainerService._numeric_feature_cols(df)


def pose_only_cols(df: pd.DataFrame) -> list[str]:
    """Baseline feature set: the animal's own body ONLY.

    Excludes video, social AND context/ROI. Folding context in here is what made
    the "pose-only" baseline able to tell two objects apart (see module docstring).
    """
    return [c for c in numeric_feature_cols(df) if is_pose_feature(c)]


def video_only_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in numeric_feature_cols(df) if is_video_feature(c)]


def social_only_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in numeric_feature_cols(df) if is_social_feature(c)]


def context_only_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in numeric_feature_cols(df) if is_context_feature(c)]


def select_feature_cols(
    df: pd.DataFrame,
    *,
    include_video: bool = False,
    include_social: bool = False,
    include_context: bool = False,
) -> list[str]:
    """Pose baseline plus the requested add-on families, in a stable order."""
    cols = pose_only_cols(df)
    if include_context:
        cols = cols + context_only_cols(df)
    if include_video:
        cols = cols + video_only_cols(df)
    if include_social:
        cols = cols + social_only_cols(df)
    return cols


def family_counts(df: pd.DataFrame) -> dict[str, int]:
    """How many columns each family contributes — for run manifests / sanity checks."""
    counts = {m: 0 for m in MODALITY_ORDER}
    for c in numeric_feature_cols(df):
        counts[classify_modality(c)] = counts.get(classify_modality(c), 0) + 1
    return counts
