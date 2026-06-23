"""Ablation toggle and suite configuration for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AblationToggle:
    """A single feature toggle that can be ON or OFF in an ablation run."""

    key: str
    label: str
    description: str
    default_on: bool = True

    # Where and how to toggle this feature programmatically.
    # Each entry is (config_attr, on_value, off_value).
    overrides: list[tuple[str, Any, Any]] = field(default_factory=list)


# ── Canonical toggle definitions ──────────────────────────────────────────

TOGGLE_VIDEO_FUSION = AblationToggle(
    key="video_fusion",
    label="Video-Derived Features",
    description=(
        "Include video-derived features (optical flow, surface motion, "
        "and R3D18 embeddings when available) in the training feature set."
    ),
    overrides=[
        ("use_video_features", True, False),
    ],
)

TOGGLE_MULTI_BEHAVIOR = AblationToggle(
    key="multi_behavior_aware",
    label="Co-Occurring Behavior Labels",
    description=(
        "Expand pipe-separated multi-behavior labels (e.g. 'groom|rear') "
        "into separate training rows so the model learns from segments "
        "where multiple behaviors co-occur."
    ),
    overrides=[
        ("allow_co_occurring_behaviors", True, False),
    ],
)

TOGGLE_CALIBRATION = AblationToggle(
    key="calibration",
    label="Probability Calibration",
    description=(
        "Post-hoc probability recalibration (sigmoid/isotonic) so predicted "
        "probabilities match observed class frequencies."
    ),
    overrides=[
        ("calibration_method", "sigmoid", "none"),
    ],
)

TOGGLE_ADAPTIVE_COMPLEXITY = AblationToggle(
    key="adaptive_complexity",
    label="Adaptive Model Complexity",
    description=(
        "Auto-tune tree depth and estimator count from the positive-to-feature "
        "ratio to prevent overfitting on small label sets."
    ),
    overrides=[
        ("adaptive_complexity", True, False),
    ],
)

TOGGLE_TEMPORAL_REFINEMENT = AblationToggle(
    key="temporal_refinement",
    label="Temporal Refinement (Bout Post-Processing)",
    description=(
        "Full bout extraction pipeline: probability smoothing, hysteresis "
        "thresholding (per-behavior onset/offset), gap merging, and minimum "
        "bout duration filtering.  Uses the project's established temporal "
        "review settings."
    ),
    overrides=[
        ("temporal_refinement_enabled", True, False),
        ("smooth_window", 5, 0),
    ],
)

ALL_TOGGLES: list[AblationToggle] = [
    TOGGLE_VIDEO_FUSION,
    TOGGLE_MULTI_BEHAVIOR,
    TOGGLE_CALIBRATION,
    TOGGLE_ADAPTIVE_COMPLEXITY,
    TOGGLE_TEMPORAL_REFINEMENT,
]


@dataclass
class AblationSuite:
    """Full configuration for an ablation benchmark run."""

    project_root: str = ""
    target_behaviors: list[str] = field(default_factory=list)
    toggles: list[AblationToggle] = field(default_factory=lambda: list(ALL_TOGGLES))
    classifier_family: str = ""  # auto-detect from project experiment.yaml
    test_size: float = 0.25
    random_state: int = 42
    n_cv_folds: int = 5
    parallel: bool = False
    max_workers: int = 4
    output_dir: str = ""  # defaults to <project>/derived/benchmark

    # Legacy single-behavior accessor
    @property
    def target_behavior(self) -> str:
        return self.target_behaviors[0] if self.target_behaviors else ""

    def enabled_toggles(self) -> list[AblationToggle]:
        return [t for t in self.toggles if t.default_on]

    def run_configs(self) -> list[dict[str, Any]]:
        """Generate the list of (name, overrides) pairs for every ablation run.

        Returns one baseline (all ON), then one per toggle with that single
        feature turned OFF.
        """
        configs: list[dict[str, Any]] = []

        # Baseline: all features ON
        baseline: dict[str, Any] = {"_run_name": "baseline_all_on"}
        for t in self.toggles:
            for attr, on_val, _off_val in t.overrides:
                baseline[attr] = on_val
        configs.append(baseline)

        # Per-toggle ablation: feature OFF, rest ON
        for t in self.toggles:
            run: dict[str, Any] = dict(baseline)
            run["_run_name"] = f"without_{t.key}"
            for attr, _on_val, off_val in t.overrides:
                run[attr] = off_val
            configs.append(run)

        # Video-only standalone: trains with ONLY video-derived features
        # to show their independent predictive power.
        if any(t.key == "video_fusion" for t in self.toggles):
            vid_only: dict[str, Any] = dict(baseline)
            vid_only["_run_name"] = "video_only"
            vid_only["video_features_only"] = True
            configs.append(vid_only)

        # All OFF
        all_off: dict[str, Any] = {"_run_name": "baseline_all_off"}
        for t in self.toggles:
            for attr, _on_val, off_val in t.overrides:
                all_off[attr] = off_val
        configs.append(all_off)

        return configs
