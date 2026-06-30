"""Typed data models used throughout ABEL."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class SourceMode(str, Enum):
    COPY = "copy"
    REFERENCE = "reference"


class ReviewStatus(str, Enum):
    UNSCORED = "unscored"
    SCORED = "scored"
    REVIEWED = "reviewed"
    EXPORTED = "exported"


class ReviewDecisionType(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    AMBIGUOUS = "ambiguous"
    RELABEL = "relabel"
    SKIP = "skip"
    BOOKMARK = "bookmark"


class InvariantFeatureConfig(BaseModel):
    """Controls which robustness/invariance features are computed during pose feature extraction.

    All options default to ``True`` (enabled) so new projects get the full
    feature set automatically.  Existing projects that have already extracted
    features retain the old column set until they re-run extraction; toggling
    these flags and re-running extraction will rebuild the feature cache with
    the new columns.
    """

    enable_egocentric_kinematics: bool = True
    """Replace per-keypoint absolute velocity_x/velocity_y with forward/lateral velocity
    in the body-centred reference frame (tail-base origin, nose→tail forward axis).
    Makes velocity direction features invariant to camera orientation and animal heading.
    Speed and acceleration magnitudes are unaffected."""

    enable_body_length_normalization: bool = True
    """Normalize inter-keypoint distances by the estimated body length (nose-to-tail distance).
    Makes spatial features invariant to animal size and camera zoom differences."""

    enable_relative_geometry: bool = True
    """Compute all pairwise inter-keypoint distances, normalized by body length.
    Provides translation- and scale-invariant shape descriptors that capture
    posture without relying on absolute coordinates."""

    enable_head_direction: bool = True
    """Estimate head direction from ear and nose keypoints (falls back to nose+body axis).
    Adds head angle, head angular velocity, and head-relative forward/lateral movement
    speed.  Useful for distinguishing orienting, freezing, and locomotion."""

    enable_joint_angles: bool = True
    """Compute joint angles from keypoint triplets (nose–body-center–tail-base,
    and any detected limb triplets).  Provides rotation-invariant posture
    descriptors that capture flexion/extension and rearing posture."""

    enable_spine_curvature: bool = False
    """Compute spine curvature from midline keypoints (requires spine1/spine2 etc.).
    Useful for rearing, grooming, and escape behaviors.  Disabled by default because
    it requires at least three midline keypoints; the feature is all-zero when those
    keypoints are absent."""

    enable_social_features: bool = False
    """Compute inter-animal (social/interaction) features in multi-animal projects.

    For each focal animal, distances/orientation/contact to every *other* animal
    in the same session are computed and reduced over conspecifics into a fixed
    column set (``social_*_min`` = nearest other, ``social_*_mean`` = averaged
    over others) so the schema is independent of the number of animals.  Has no
    effect on single-animal projects (no other animals to compare against).
    Disabled by default."""

    enable_clipwise_deltas: bool = False
    """Add clip-wise posture-change features at the window-aggregation stage.

    For every per-frame angle column (joint angles, head direction, body
    orientation, head pitch, spine curvature) and every inter-keypoint proximity
    column (pairwise distances, including body-length-normalized variants) two extra
    statistics are emitted per window:

      * ``*_delta`` : last-frame minus first-frame value — the signed net change
        in that angle/proximity across the clip.
      * ``*_trend`` : slope of the least-squares linear fit (units per frame), a
        more noise-robust measure of the same directional change.

    These capture how posture *evolves* across a clip (e.g. an animal extending
    from a crouch, or two body parts drawing together) — information that
    mean/std aggregates discard.  Unlike the other robustness options these are
    computed during segment windowing, not per-frame, so they require the
    relevant base columns (relative geometry / joint angles / head direction) to
    be present.  Disabled by default."""

    @classmethod
    def load_from_project(cls, project_root: "Path") -> "InvariantFeatureConfig":
        """Load the robustness-feature toggles from a project's experiment.yaml.

        Reads ``behavior_model.invariant_features`` from
        ``<project_root>/config/experiment.yaml``.  Missing file or keys fall
        back to field defaults.  Never raises — returns defaults on any error.
        """
        from abel.storage.file_store import read_yaml  # noqa: PLC0415

        try:
            data = read_yaml(project_root / "config" / "experiment.yaml", {})
            inv = (data.get("behavior_model", {}) or {}).get("invariant_features", {}) or {}
            return cls(**{k: v for k, v in inv.items() if k in cls.model_fields})
        except Exception:
            return cls()


class AppSettings(BaseModel):
    schema_version: str = "0.2.0"
    theme: str = "system"
    autosave_seconds: int = 30
    max_recent_projects: int = 12
    default_output_format: str = "csv"
    check_updates_on_startup: bool = False


class ImportNameSettings(BaseModel):
    """Rules for deriving subject and session labels from imported filenames."""

    subject_regex: str = r"^([A-Za-z0-9]+)(?=_|\.|DLC|$)"
    subject_group_index: int = 1
    session_regex: str = r"^[A-Za-z0-9]+_([A-Za-z0-9]+?)(?=DLC|_|\.|$)"
    session_group_index: int = 1


class PoseSmoothingSettings(BaseModel):
    """Temporal smoothing applied to DLC tracking data when pose files are loaded."""

    likelihood_threshold: float = 0.2
    """Drop detections below this confidence before interpolating."""

    interpolate_dropouts: bool = True
    """Fill short gaps in tracking with linear interpolation."""

    interpolate_max_gap: int = 10
    """Maximum consecutive missing frames to interpolate across."""

    smoothing_window: int = 5
    """Odd number of frames for centred rolling-average smoothing (1 = no smoothing)."""


class BehaviorModelConfig(BaseModel):
    """Experiment controls for behavior modeling and active learning."""

    target_behavior_id: str = ""
    allow_co_occurring_behaviors: bool = False
    """When True, multiple behaviors can be assigned to the same frame window."""
    use_video_features: bool = True
    """When True, video-derived features (optical flow, motion) and fusion are included.
    When False, only pose-derived features are used — faster, but blind to local
    micro-motion (e.g. grooming paw movement over the face), which makes spatially
    stationary behaviors like freezing and grooming hard to tell apart."""
    segment_window_frames: int = 60
    segment_stride_frames: int = 15
    hard_negative_sampling_ratio: float = 0.3
    query_strategy: Literal[
        "prototype",
        "uncertainty",
        "novelty",
        "low_probability",
        "random_absent",
    ] = "uncertainty"
    invariant_features: InvariantFeatureConfig = Field(default_factory=InvariantFeatureConfig)
    """Controls which robustness/invariance features are computed during pose extraction."""
    enable_feature_augmentation: bool = True
    """Augment positive training examples with Gaussian jitter and feature dropout at training time.
    Creates additional synthetic copies of labeled positive examples to improve robustness
    to tracking noise and reduce overfitting to small labeled sets."""
    augmentation_jitter_sigma: float = 0.05
    """Gaussian noise added to augmented examples as a fraction of each feature's std.
    A value of 0.05 adds noise at 5% of each feature's standard deviation."""
    augmentation_dropout_prob: float = 0.10
    """Fraction of features randomly zeroed per augmented example.
    Simulates missing/occluded keypoints and reduces over-reliance on single features."""
    augmentation_copies: int = 3
    """Number of augmented copies created per positive example.
    Higher values improve robustness but increase training time linearly."""
    classifier_type: str = "xgboost"
    classifier_params: dict[str, Any] = Field(default_factory=lambda: {"tree_method": "hist"})
    calibration_method: Literal["none", "sigmoid", "isotonic"] = "sigmoid"
    uncertainty_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "entropy": 0.4,
            "ensemble_variance": 0.4,
            "density_outlier": 0.2,
            "margin": 0.0,
        }
    )
    fusion_threshold: float = 0.35
    fusion_thresholds: dict[str, float] = Field(default_factory=dict)
    active_learning_query_size: int = 50
    bout_merge_gap: int = 10
    min_bout_duration: int = 15
    evaluation_split_strategy: Literal[
        "group_shuffle_subject",
        "group_shuffle_session",
        "leave_one_subject_out",
    ] = "group_shuffle_session"


class ProjectConfig(BaseModel):
    schema_version: str = "0.2.0"
    project_name: str
    assay_name: str = "generic_assay"
    species: str = "mouse"
    single_animal: bool = True
    num_animals: int = 1
    """Number of animals tracked per session.  ``1`` (default) is the legacy
    single-animal path; ``> 1`` enables multi-animal ingestion, per-individual
    feature extraction, and (when ``enable_social_features`` is set) interaction
    features.  Kept consistent with ``single_animal`` (single_animal == num_animals <= 1)."""
    expected_pose_formats: list[str] = Field(default_factory=lambda: ["csv", "h5"])
    default_fps: float = 30.0
    default_clip_duration_sec: float = 2.0
    default_crop_margin_px: int = 40
    default_downsample_preset: str = "fast_preview"
    behavior_model: BehaviorModelConfig = Field(default_factory=BehaviorModelConfig)
    video_source_mode: SourceMode = SourceMode.REFERENCE
    pose_source_mode: SourceMode = SourceMode.REFERENCE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectState(BaseModel):
    schema_version: str = "0.2.0"
    selected_tab: str = "Home"
    selected_session_id: str | None = None
    pending_jobs: list[dict[str, Any]] = Field(default_factory=list)
    review_progress: dict[str, Any] = Field(default_factory=dict)
    last_opened_at: datetime = Field(default_factory=datetime.utcnow)


class DependencySpec(BaseModel):
    package: str
    purpose: str
    required_version: str
    tier: str
    installed_version: str | None = None
    status: str = "unknown"


class VideoAsset(BaseModel):
    asset_id: str
    source_path: str
    local_path: str | None = None
    fps: float | None = None
    frame_count: int | None = None
    width: int | None = None
    height: int | None = None
    duration_sec: float | None = None
    subject_id: str | None = None
    session_id: str | None = None
    pixels_per_mm: float | None = None


class PoseAsset(BaseModel):
    asset_id: str
    source_path: str
    local_path: str | None = None
    format: str
    frame_count: int | None = None
    body_parts: list[str] = Field(default_factory=list)
    individuals: list[str] = Field(default_factory=list)
    """Detected individuals for a multi-animal pose file (e.g. ["Mouse1","Mouse2"]).
    Empty for single-animal / 3-level files."""
    has_likelihood: bool = True
    subject_id: str | None = None
    session_id: str | None = None


class Recording(BaseModel):
    recording_id: str
    subject_id: str | None = None
    session_id: str | None = None
    video_asset_id: str | None = None
    pose_asset_id: str | None = None


class LinkedSession(BaseModel):
    session_id: str
    video_asset_id: str
    pose_asset_id: str
    subject_id: str | None = None
    pixels_per_mm: float | None = None
    subject_locked: bool = False  # True when subject was set by hand; protected from regex reapply
    pairing_score: float = 0.0
    pairing_notes: str = ""
    individuals: list[str] = Field(default_factory=list)
    """Generic individual IDs detected in a multi-animal pose file (e.g.
    ``["Mouse1", "Mouse2"]``).  Empty for single-animal / 3-level pose files."""
    individual_subject_map: dict[str, str] = Field(default_factory=dict)
    """Maps each detected individual to a real project subject identity (e.g.
    ``{"Mouse1": "green", "Mouse2": "black"}``).  Unmapped individuals fall back
    to ``{subject_id}:{individual}`` as their ``animal_id``."""
    identity_corrections: list[dict[str, Any]] = Field(default_factory=list)
    """User-confirmed identity-swap corrections, each ``{"frame": t, "a": A,
    "b": B}`` meaning individuals A and B exchange tracks from frame ``t`` onward.
    Applied on pose load so features see identity-consistent tracks."""


class BehaviorDefinition(BaseModel):
    behavior_id: str
    name: str
    short_name: str
    description: str = ""
    operational_definition: str = ""
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    min_duration_sec: float = 0.0
    parent_category: str | None = None
    review_priority: int = 1
    color: str = "#4A90E2"
    keyboard_shortcut: str | None = None
    is_active: bool = True
    is_social: bool = False
    """True for social/interaction behaviors that depend on inter-animal
    (``social_*``) features.  Requires a multi-animal project (num_animals > 1).
    Solo behaviors (default) work unchanged on single- and multi-animal projects."""
    directionality: Literal["none", "directed", "mutual"] = "none"
    """For social behaviors: ``directed`` labels the focal *actor* (e.g. the
    animal that displaces another); ``mutual`` labels both interacting animals'
    overlapping segments positive.  ``none`` for solo behaviors."""
    notes: str = ""
    version_history: list[dict[str, Any]] = Field(default_factory=list)
    prompt_template: dict[str, str] = Field(
        default_factory=lambda: {
            "primary_question": "Does this clip contain the target behavior?",
            "secondary_discriminators": "",
            "exclusion_cues": "",
            "ambiguity_instructions": "Use ambiguous if uncertain.",
        }
    )


class SeedExample(BaseModel):
    seed_id: str
    behavior_id: str
    session_id: str
    start_frame: int
    end_frame: int
    animal_id: str | None = None
    """Which focal animal in a multi-animal session this seed applies to.  None
    (default) means the sole animal in a single-animal session, or all animals."""
    label_type: str = "positive"
    quality_flag: str = "clean"
    notes: str = ""


class PoseFeaturePreset(BaseModel):
    """Parameters for pose cleaning and kinematic feature window extraction."""
    preset_id: str
    name: str
    window_duration_sec: float = 2.0
    stride_sec: float = 1.0
    source_fps: float = 30.0
    likelihood_threshold: float = 0.2
    interpolate_dropouts: bool = True
    smoothing_window: int = 5


class SessionFeatureSummary(BaseModel):
    """Metadata written after pose feature extraction for one session."""
    session_id: str
    n_frames: int = 0
    n_windows: int = 0
    body_parts: list[str] = Field(default_factory=list)
    fps: float = 30.0
    feature_path: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    warnings: list[str] = Field(default_factory=list)


class CandidateWindow(BaseModel):
    """A ranked pose window selected for video clip extraction."""
    window_id: str
    session_id: str
    start_frame: int
    end_frame: int
    behavior_id: str | None = None
    motif_score: float = 0.0
    seed_similarity_score: float = 0.0
    total_score: float = 0.0
    clip_path: str | None = None
    source: str = ""
    selection_reason: str = ""


class ArtifactProvenance(BaseModel):
    app_version: str
    git_commit_hash: str
    model_version: str
    feature_version: str
    config_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SegmentPrediction(BaseModel):
    segment_id: str
    prediction_prob: float
    uncertainty_score: float
    model_version: str
    feature_version: str


class CandidateSegment(BaseModel):
    segment_id: str
    start_frame: int
    end_frame: int
    video_id: str
    animal_id: str
    session_id: str
    prediction_prob: float
    uncertainty_score: float
    behavior_id: str | None = None
    pose_features: dict[str, float] = Field(default_factory=dict)
    context_features: dict[str, float] = Field(default_factory=dict)
    score_components: dict[str, float] = Field(default_factory=dict)
    final_priority_score: float = 0.0
    selection_reason: str = ""
    model_version: str
    feature_version: str
    provenance: ArtifactProvenance


class ReviewerLabelRecord(BaseModel):
    segment_id: str
    review_label: str
    reviewer_id: str
    confidence: float = 1.0
    notes: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainingSetRecord(BaseModel):
    segment_id: str
    label: str
    label_source: str
    reviewer_confidence: float = 1.0
    animal_id: str
    session_id: str
    features_vector: list[float]


class ModelCard(BaseModel):
    model_version: str
    classifier_family: str
    calibration_method: str
    training_split_strategy: str
    labels: list[str] = Field(default_factory=list)
    feature_columns: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    provenance: ArtifactProvenance


class PreprocessingPreset(BaseModel):
    """Video clip output parameters used during selective clip extraction."""
    preset_id: str
    name: str
    clip_duration_sec: float = 2.0
    stride_sec: float = 1.0
    output_fps: float = 15.0
    resize_width: int = 256
    resize_height: int = 256
    crop_margin_px: int = 80
    crop_area_scale: float = 1.25
    adaptive_crop: bool = True
    likelihood_threshold: float = 0.2
    interpolate_dropouts: bool = True
    smoothing_window: int = 5
    stabilize: bool = False
    body_axis_rotation: bool = False
    grayscale: bool = False


class ClipAsset(BaseModel):
    clip_id: str
    session_id: str
    start_frame: int
    end_frame: int
    processed_clip_path: str | None = None
    original_clip_path: str | None = None
    thumbnail_path: str | None = None


class ClipManifestEntry(BaseModel):
    clip_id: str
    session_id: str
    behavior_target: str | None = None
    score_components: dict[str, float] = Field(default_factory=dict)
    total_score: float = 0.0
    status: ReviewStatus = ReviewStatus.UNSCORED
    preprocessing_settings: dict[str, Any] = Field(default_factory=dict)


class CandidateScore(BaseModel):
    candidate_id: str
    behavior_target: str
    seed_similarity_score: float = 0.0
    pose_rule_score: float = 0.0
    temporal_score: float = 0.0
    heuristic_score: float = 0.0

    @property
    def combined(self) -> float:
        return (
            self.seed_similarity_score
            + self.pose_rule_score
            + self.temporal_score
            + self.heuristic_score
        )


class ReviewDecision(BaseModel):
    decision_id: str
    clip_id: str
    reviewer: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    old_status: str
    new_status: str
    decision: ReviewDecisionType
    behavior_label: str | None = None
    notes: str = ""
    confidence_override: float | None = None
    adjusted_start_frame: int | None = None
    adjusted_end_frame: int | None = None


class ExportJob(BaseModel):
    export_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    format: str = "csv"
    destination: str
    clip_count: int = 0
    frame_count: int = 0
    status: str = "pending"


class AuditEvent(BaseModel):
    event_id: str
    event_type: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    actor: str = "system"
    details: dict[str, Any] = Field(default_factory=dict)


class ImportManifest(BaseModel):
    subject_name_settings: ImportNameSettings = Field(default_factory=ImportNameSettings)
    smoothing_settings: PoseSmoothingSettings = Field(default_factory=PoseSmoothingSettings)
    videos: list[VideoAsset] = Field(default_factory=list)
    poses: list[PoseAsset] = Field(default_factory=list)
    linked_sessions: list[LinkedSession] = Field(default_factory=list)


class ClipManifest(BaseModel):
    """Collection of clips produced by a preprocessing run."""

    run_id: str = Field(default_factory=lambda: uuid4().hex[:10])
    session_ids: list[str] = Field(default_factory=list)
    preset_name: str = ""
    clips: list[ClipAsset] = Field(default_factory=list)
    total_windows: int = 0
    opencv_available: bool = False
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectContext(BaseModel):
    project_root: Path
    config: ProjectConfig
    state: ProjectState


# ---------------------------------------------------------------------------
# Validation (model-quality overview + interactive reviewer quiz)
# ---------------------------------------------------------------------------

ValidationCategory = Literal[
    "prior_accepted",
    "unreviewed_positive",
    "negative",
    "fringe",
]


class ValidationSettings(BaseModel):
    """User-tunable parameters for assembling a validation quiz.

    Proportions are relative weights; they are normalised when the test is
    assembled so they do not need to sum to exactly 1.0.
    """

    n_total_clips: int = 60
    prop_prior_accepted: float = 0.30
    prop_unreviewed_positive: float = 0.35
    prop_negative: float = 0.25
    prop_fringe: float = 0.10
    # Lower edges of the probability bands used to stratify positive samples.
    # Positives are drawn roughly evenly across [edge, next_edge) buckets so the
    # quiz spans clearly-positive through borderline detections.
    prob_band_edges: list[float] = Field(default_factory=lambda: [0.5, 0.6, 0.8])
    # Half-width around each behavior's onset threshold that defines "fringe".
    fringe_half_width: float = 0.10
    balance_across_behaviors: bool = True
    loop_default: bool = True
    autoadvance_default: bool = True
    clip_seconds: float = 2.0
    # Behavior Grid panel: crop half-width multiplier (>1 zooms out). Persisted
    # so the reviewer's preferred framing survives project reloads / new grids.
    behavior_grid_crop_scale: float = 1.0
    # Behavior Grid panel: keypoint-dot size multiplier (1.0 = the default size
    # that scales with frame height). Persisted alongside the crop scale.
    behavior_grid_keypoint_scale: float = 1.0


class ValidationClipRecord(BaseModel):
    """One clip presented during a validation quiz.

    ``machine_label`` is what the model/threshold asserts (a behavior_id for
    positive/fringe categories, ``no_behavior`` for negatives).  ``reference_label``
    is the best-known human truth where available (e.g. a prior accepted label),
    otherwise ``None``.
    """

    clip_id: str
    category: ValidationCategory
    behavior_id: str
    machine_label: str
    reference_label: str | None = None
    session_id: str
    start_frame: int
    end_frame: int
    probability: float = 0.0
    is_fringe: bool = False
    clip_path: str | None = None
    # Behaviors the model flagged simultaneously at this clip (incl. machine_label).
    # When two or more are present the clip is ambiguous and is excluded from
    # user-vs-machine scoring so it does not count against the reviewer.
    coactive_labels: list[str] = Field(default_factory=list)


class ValidationAnswerRecord(BaseModel):
    """A single reviewer's verdict for one quiz clip."""

    clip_id: str
    reviewer_id: str
    label: str  # behavior_id or "no_behavior"
    is_unsure: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationRun(BaseModel):
    """A fixed, assembled set of quiz clips that one or more reviewers complete."""

    run_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    config: dict[str, Any] = Field(default_factory=dict)
    clips: list[ValidationClipRecord] = Field(default_factory=list)
