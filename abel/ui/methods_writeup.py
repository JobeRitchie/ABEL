"""Methods write-up drafting: turn a project's real settings into draft prose.

Powers the **Methods → Write-up Helper** subtab.  The user ticks the facts they
want covered; this module reads those facts out of the open project and renders
them as plain-text paragraphs that can be pasted into a manuscript draft.

Two rules shape everything here:

* **Never invent a number.**  Every value in the output is read from the project
  (``project.yaml``, ``behavior_definitions.yaml``, the model directories, the
  motif settings, the import manifest).  Anything ABEL cannot know, the pose
  tracker and its version, the animals, the apparatus, is emitted as an explicit
  ``[FILL IN: …]`` placeholder rather than a plausible guess.
* **The output is a draft, not manuscript text.**  :data:`WARNING_TEXT` says so,
  the UI repeats it, and it is prepended to the generated text by default.  A
  methods section is a claim about what was done; only the author can confirm it.

The module is deliberately Qt-free and side-effect-free so it can be tested
headlessly.  The two expensive inputs, per-behavior model metrics and reviewer
agreement, are *injected* by the caller (see :func:`gather_facts`), because both
come from :class:`~abel.services.validation_service.ValidationService` and belong
on a worker thread, not in a renderer.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from abel import __version__
from abel.changelog import VERSION_DATE

REPO_URL = "https://github.com/JobeRitchie/ABEL"

WARNING_TEXT = (
    "=" * 78 + "\n"
    "DRAFT ONLY: THIS IS NOT PUBLICATION TEXT\n"
    + "=" * 78 + "\n"
    "This draft was assembled from the settings and results stored in this ABEL\n"
    "project. It is a starting point and a checklist, NOT a methods section.\n"
    "Before any of it reaches a manuscript:\n"
    "  * Read every sentence and rewrite it in your own words and your journal's\n"
    "    style. Do not paste this text verbatim.\n"
    "  * Check every number against your own records. Settings can change between\n"
    "    a run and today; this text reports the CURRENT project state.\n"
    "  * Replace every [FILL IN: ...] placeholder. ABEL cannot know your animals,\n"
    "    apparatus, tracking software, or experimental design.\n"
    "  * Delete anything you did not actually do or do not intend to report.\n"
    "You are responsible for the accuracy of your methods section.\n"
    + "=" * 78
)


@dataclass(frozen=True)
class SectionSpec:
    """One tickable fact-group offered in the write-up helper."""

    key: str
    label: str
    group: str
    hint: str
    default: bool


SECTIONS: tuple[SectionSpec, ...] = (
    # ---- Core statements ----
    SectionSpec(
        "software", "Software and version", "Core statements",
        "Names ABEL, its version and release date, and the project/assay.", True,
    ),
    SectionSpec(
        "ml_approach", "Machine-learning approach (one sentence)", "Core statements",
        "One sentence naming the classifier family, the window-level feature input, "
        "the one-vs-rest design and the probability calibration.", True,
    ),
    SectionSpec(
        "behaviors", "Behaviors classified", "Core statements",
        "The list of behaviors models were trained for, excluding the No Behavior "
        "negative class.", True,
    ),
    SectionSpec(
        "behavior_definitions", "Operational definitions of each behavior", "Core statements",
        "Each behavior's operational definition, inclusion and exclusion criteria as "
        "written in the Behaviors tab.", False,
    ),
    # ---- Data and processing ----
    SectionSpec(
        "dataset", "Dataset (sessions, subjects, video)", "Data and processing",
        "Session and subject counts, frame rate, resolution and total recording time.", True,
    ),
    SectionSpec(
        "pose", "Pose tracking and cleaning", "Data and processing",
        "Keypoint set, pose file format, and the likelihood / interpolation / "
        "smoothing settings ABEL applied.", True,
    ),
    SectionSpec(
        "features", "Feature extraction", "Data and processing",
        "Window and stride, which feature families were enabled, and how many "
        "features the trained models used.", True,
    ),
    SectionSpec(
        "roi", "Regions of interest", "Data and processing",
        "How many target zones were defined and what ROI-relative features they "
        "contributed.", False,
    ),
    # ---- Training and performance ----
    SectionSpec(
        "training", "Model training and evaluation design", "Training and performance",
        "Train/held-out split strategy, calibration, augmentation, and the deploy "
        "refit.", True,
    ),
    SectionSpec(
        "performance", "Model performance metrics", "Training and performance",
        "Per-behavior precision, recall, F1 and PR-AUC on the held-out split.", True,
    ),
    SectionSpec(
        "temporal", "Bout definition / temporal refinement", "Training and performance",
        "Per-behavior onset threshold, minimum bout duration and merge gap used to "
        "turn probabilities into bouts.", True,
    ),
    SectionSpec(
        "active_learning", "Active-learning labeling procedure", "Training and performance",
        "Query strategy, clips proposed per round, and how many labels each behavior "
        "ended up with.", True,
    ),
    SectionSpec(
        "reliability", "Rater agreement (Cohen's / Fleiss' kappa)", "Training and performance",
        "Inter- and intra-rater agreement from the Validation tab's blind quizzes, "
        "plus reviewer-vs-model agreement.", False,
    ),
    # ---- Downstream analyses ----
    SectionSpec(
        "hmm", "HMM state analysis (one sentence)", "Downstream analyses",
        "One sentence giving the HMM's state count, fitting and occupancy settings.", False,
    ),
    SectionSpec(
        "motifs", "Transition and motif analysis", "Downstream analyses",
        "Transition matrices, n-gram motifs, permutation testing and FDR correction.", False,
    ),
    SectionSpec(
        "stats", "Group statistics", "Downstream analyses",
        "The tests the Analytics tab runs on exported per-session metrics.", False,
    ),
    # ---- Reproducibility ----
    SectionSpec(
        "software_stack", "Library versions", "Reproducibility",
        "Installed versions of the libraries that do the actual computation.", False,
    ),
    SectionSpec(
        "reproducibility", "Model provenance and data availability", "Reproducibility",
        "Model version names, training timestamps, config hashes and seeds.", False,
    ),
    SectionSpec(
        "citations", "Reference list for the sections above", "Reproducibility",
        "Full citations for the methods cited by whichever sections you ticked.", False,
    ),
)

SECTION_GROUPS: tuple[str, ...] = (
    "Core statements",
    "Data and processing",
    "Training and performance",
    "Downstream analyses",
    "Reproducibility",
)

DEFAULT_KEYS: tuple[str, ...] = tuple(s.key for s in SECTIONS if s.default)

# Citations per section, keyed to abel.ui.methods_content.REFERENCES. Kept here
# rather than on SectionSpec because a few depend on what the project enabled
# (R3D appearance features are only cited when the project actually used them).
_SECTION_REFS: dict[str, tuple[str, ...]] = {
    "ml_approach": ("chen2016", "platt1999", "zadrozny2002", "nilsson2020"),
    "pose": ("mathis2018", "pereira2022"),
    "features": (),
    "training": ("pedregosa2011", "saeb2017", "varoquaux2018"),
    "performance": ("saito2015", "chicco2020"),
    "active_learning": ("settles2009", "shannon1948", "mcinnes2018"),
    "reliability": ("cohen1960", "fleiss1971", "landis1977"),
    "hmm": ("rabiner1989",),
    "motifs": ("good2005", "benjamini1995"),
}

_NO_BEHAVIOR_TOKENS = {"no_behavior", "no behavior", "none"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _is_no_behavior(token: str) -> bool:
    normalized = str(token or "").strip().lower().replace("-", "_")
    return normalized in {t.replace(" ", "_") for t in _NO_BEHAVIOR_TOKENS}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _num(value: Any, digits: int = 3) -> str:
    """Format a metric, or a dash when it is missing or not finite."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(f):
        return "-"
    return f"{f:.{digits}f}"


def _count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or singular + "s")


def _join(items: list[str]) -> str:
    """Oxford-comma join, for prose lists."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _sec(frames: Any, fps: float) -> str:
    try:
        f = float(frames)
    except (TypeError, ValueError):
        return "-"
    if fps <= 0:
        return "-"
    return f"{f / fps:.2f}"


def _fill(what: str) -> str:
    return f"[FILL IN: {what}]"


# ---------------------------------------------------------------------------
# Fact gathering
# ---------------------------------------------------------------------------
def gather_facts(
    project_root: Path,
    *,
    model_rows: list[dict[str, Any]] | None = None,
    reliability: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read every cheap fact about *project_root* into a dict for the renderer.

    ``model_rows`` is :meth:`ValidationService.model_overview` output and
    ``reliability`` a list of per-run agreement summaries (see
    :func:`summarize_validation_runs`).  Both are injected rather than computed
    here: they read model artifacts and probability traces, which is worker-thread
    work, while everything below is a handful of small config files.
    """
    root = Path(project_root)
    project = _read_yaml(root / "project.yaml")
    behavior_model = project.get("behavior_model") or {}
    fps = float(project.get("default_fps") or 30.0)

    return {
        "project_root": root,
        "app_version": __version__,
        "version_date": VERSION_DATE,
        "repo_url": REPO_URL,
        "generated_on": datetime.now().strftime("%Y-%m-%d"),
        "project": project,
        "behavior_model": behavior_model,
        "fps": fps,
        "behaviors": _gather_behaviors(root),
        "recordings": _gather_recordings(root),
        "roi": _gather_roi(root),
        "motif": _gather_motif(root),
        "temporal_default": (
            _read_json(root / "config" / "temporal_review_settings.json").get("__all__") or {}
        ),
        "model_rows": list(model_rows or []),
        "reliability": list(reliability or []),
        "feature_modalities": _gather_feature_modalities(root),
        "libraries": _gather_library_versions(),
    }


def _gather_behaviors(root: Path) -> list[dict[str, Any]]:
    """Active behavior definitions, No Behavior excluded, in review-priority order."""
    raw = _read_yaml(root / "config" / "behavior_definitions.yaml")
    out: list[dict[str, Any]] = []
    for b in raw.get("behaviors", []) or []:
        if not isinstance(b, dict):
            continue
        if not b.get("is_active", b.get("active", True)):
            continue
        if _is_no_behavior(b.get("behavior_id", "")) or _is_no_behavior(b.get("name", "")):
            continue
        out.append(b)
    out.sort(key=lambda b: (int(b.get("review_priority") or 0), str(b.get("name") or "")))
    return out


def _gather_recordings(root: Path) -> dict[str, Any]:
    """Session / subject / video summary from the import manifest."""
    raw = _read_json(root / "derived" / "review_tables" / "import_manifest.json")
    sessions = raw.get("linked_sessions", []) or []
    videos = raw.get("videos", []) or []
    poses = raw.get("poses", []) or []

    subjects = sorted({str(s.get("subject_id") or "") for s in sessions} - {""})
    fps_values = sorted({round(float(v["fps"]), 2) for v in videos if v.get("fps")})
    resolutions = sorted(
        {(int(v["width"]), int(v["height"])) for v in videos if v.get("width") and v.get("height")}
    )
    durations = [float(v["duration_sec"]) for v in videos if v.get("duration_sec")]
    keypoints: list[str] = []
    individuals: list[str] = []
    for p in poses:
        if p.get("body_parts") and not keypoints:
            keypoints = [str(k) for k in p["body_parts"]]
        if p.get("individuals") and not individuals:
            individuals = [str(i) for i in p["individuals"]]

    return {
        "n_sessions": len(sessions),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "n_videos": len(videos),
        "fps_values": fps_values,
        "resolutions": resolutions,
        "total_minutes": (sum(durations) / 60.0) if durations else None,
        "mean_minutes": (sum(durations) / len(durations) / 60.0) if durations else None,
        "pose_formats": sorted({str(p.get("format") or "") for p in poses} - {""}),
        "keypoints": keypoints,
        "individuals": individuals,
        "smoothing": raw.get("smoothing_settings", {}) or {},
    }


def _gather_roi(root: Path) -> dict[str, Any]:
    """Target-zone count and shapes, read through the ROI service's normalizer."""
    try:
        from abel.services.roi_service import ROIService  # noqa: PLC0415

        svc = ROIService()
        cfg = svc.load(root, mutable=False)
        shapes: Counter[str] = Counter()
        blocks = [cfg.get("project_rois") or {}]
        blocks.extend((cfg.get("subject_rois") or {}).values())
        for block in blocks:
            for zone in (block or {}).get("target_zones", []) or []:
                if not isinstance(zone, dict):
                    continue
                if int(zone.get("w") or 0) <= 0 and int(zone.get("h") or 0) <= 0:
                    continue  # unset placeholder slot
                shapes[str(zone.get("shape") or "rect")] += 1
        return {
            "count": svc.get_roi_count(root),
            # roi_count is 1 even for a project that never drew a zone, so the
            # shape tally (built only from slots with real dimensions) is what
            # tells us whether any ROI exists.
            "has_zones": bool(shapes),
            "shapes": dict(shapes),
            "motion_radius_px": svc.local_motion_radius(root),
            "bg_var_threshold": svc.bg_var_threshold(root),
            "excluded_days": svc.get_roi_excluded_days(root),
        }
    except Exception:
        return {}


def _gather_motif(root: Path) -> dict[str, Any]:
    try:
        from abel.services.behavioral_motif_service import load_motif_settings  # noqa: PLC0415

        return load_motif_settings(root).to_dict()
    except Exception:
        return _read_json(root / "config" / "motif_settings.json")


def _gather_feature_modalities(root: Path) -> dict[str, Any]:
    """Count the most recently trained model's feature columns by modality.

    Read from the model card rather than the feature cache: the card lists the
    columns the shipped model actually consumed, which is what a methods section
    should report.
    """
    models_dir = root / "derived" / "models"
    if not models_dir.is_dir():
        return {}
    cards = sorted(models_dir.glob("*/model_card.yaml"), key=lambda p: p.stat().st_mtime)
    if not cards:
        return {}
    card = _read_yaml(cards[-1])
    cols = [str(c) for c in (card.get("feature_columns") or [])]
    if not cols:
        return {}
    try:
        from abel.validation.features import classify_modality  # noqa: PLC0415
    except Exception:
        return {"model_version": card.get("model_version"), "n_features": len(cols)}
    return {
        "model_version": card.get("model_version"),
        "n_features": len(cols),
        "by_modality": dict(Counter(classify_modality(c) for c in cols)),
    }


def _gather_library_versions() -> dict[str, str]:
    """Installed versions of the libraries that do the computation."""
    from importlib.metadata import version  # noqa: PLC0415

    out: dict[str, str] = {}
    for pkg in (
        "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm",
        "opencv-python", "torch", "torchvision", "hmmlearn", "umap-learn",
        "scipy", "statsmodels",
    ):
        try:
            out[pkg] = version(pkg)
        except Exception:
            continue
    return out


def summarize_validation_runs(service: Any) -> list[dict[str, Any]]:
    """Reduce every saved validation quiz to the agreement numbers worth reporting.

    Takes a project-scoped :class:`ValidationService`.  Kept here (rather than in
    the tab) so the shape of a reliability record is defined next to the renderer
    that consumes it.
    """
    out: list[dict[str, Any]] = []
    try:
        runs = service.list_runs()
    except Exception:
        return out
    for run in runs:
        try:
            answers = service.load_all_answers(run.run_id)
            if not answers:
                continue
            metrics = service.compute_metrics(run, answers)
        except Exception:
            continue
        per_reviewer = metrics.get("per_reviewer") or {}
        out.append({
            "run_id": run.run_id,
            "n_clips": metrics.get("n_clips"),
            "n_reviewers": len(per_reviewer),
            "inter_rater": metrics.get("inter_rater") or {},
            "intra_rater": metrics.get("intra_rater") or {},
            "reviewer_agreement": {
                rid: (v or {}).get("agreement") for rid, v in per_reviewer.items()
            },
        })
    return out


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
def _r_software(f: dict[str, Any]) -> str:
    p = f["project"]
    name = str(p.get("project_name") or _fill("project name"))
    assay = str(p.get("assay_name") or "").strip()
    species = str(p.get("species") or "").strip()
    n_animals = int(p.get("num_animals") or (1 if p.get("single_animal", True) else 2))

    detail = []
    if assay and assay != "generic_assay":
        detail.append(f"assay: {assay}")
    if species:
        detail.append(f"species: {species}")
    detail.append(f"{n_animals} {_plural(n_animals, 'animal')} tracked per session")
    return (
        f"Behavior was scored automatically with ABEL v{f['app_version']} "
        f"(Active-learning Behavior Estimation and Labeling; released {f['version_date']}), "
        f"an open-source desktop application for supervised behavior classification from "
        f"pose-tracked video ({f['repo_url']}). "
        f"Analyses reported here come from the ABEL project \"{name}\" "
        f"({'; '.join(detail)})."
    )


def _r_ml_approach(f: dict[str, Any]) -> str:
    bm = f["behavior_model"]
    family = str(bm.get("classifier_type") or "xgboost")
    family_label = {
        "xgboost": "gradient-boosted decision trees (XGBoost)",
        "lightgbm": "gradient-boosted decision trees (LightGBM)",
        "random_forest": "a random forest",
        "logistic_regression": "L2-regularized logistic regression",
    }.get(family, f"a {family} classifier")
    calib = str(bm.get("calibration_method") or "sigmoid")
    calib_label = {
        "sigmoid": "Platt scaling (sigmoid)",
        "isotonic": "isotonic regression",
        "none": "no post-hoc calibration",
    }.get(calib, calib)
    inputs = "pose-derived" if not bm.get("use_video_features", True) else "pose- and video-derived"

    text = (
        f"ABEL classifies behavior with {family_label} trained on fixed-length time windows "
        f"of {inputs} features, fitting one binary one-vs-rest model per behavior and "
        f"calibrating its output probabilities by {calib_label}; training examples are chosen "
        f"by active learning, so a human reviewer labels the clips the current model is least "
        f"certain about rather than a random sample of the video."
    )
    if bm.get("allow_co_occurring_behaviors"):
        text += (
            " Behaviors were allowed to co-occur, so a window can carry more than one "
            "behavior label and each model was scored independently."
        )
    else:
        text += (
            " Behaviors were treated as mutually exclusive within a window, so the "
            "highest-probability behavior wins where models disagree."
        )
    return text


def _r_behaviors(f: dict[str, Any]) -> str:
    behaviors = f["behaviors"]
    if not behaviors:
        return "No active behaviors are defined in this project. " + _fill("the behaviors you scored")

    names = [str(b.get("name") or b.get("short_name") or "?") for b in behaviors]
    trained = {
        str(r.get("behavior_id")): r
        for r in f["model_rows"]
        if not _is_no_behavior(str(r.get("behavior_id", "")))
        and str(r.get("model_version") or "-") != "-"
    }
    lines = [
        f"Separate classifiers were trained for {len(names)} "
        f"{_plural(len(names), 'behavior')}: {_join(names)}. "
        f"A universal negative class (\"No Behavior\") was used during training to define "
        f"each behavior's negatives and is not reported as a behavior."
    ]
    untrained = [
        str(b.get("name") or "?") for b in behaviors if str(b.get("behavior_id")) not in trained
    ]
    if untrained:
        lines.append(
            f"Note: {len(untrained)} of the {len(names)} defined "
            f"{_plural(len(names), 'behavior')} "
            f"{'has' if len(untrained) == 1 else 'have'} no trained model in this project "
            f"({_join(untrained)}). Decide whether to report or remove "
            f"{'it' if len(untrained) == 1 else 'them'}."
        )
    return "\n".join(lines)


def _r_behavior_definitions(f: dict[str, Any]) -> str:
    behaviors = f["behaviors"]
    if not behaviors:
        return ""
    out = ["Behaviors were scored against the following operational definitions."]
    for b in behaviors:
        name = str(b.get("name") or "?")
        pieces = []
        for label, key in (
            ("Definition", "operational_definition"),
            ("Include", "inclusion_criteria"),
            ("Exclude", "exclusion_criteria"),
        ):
            val = str(b.get(key) or "").strip()
            if val:
                pieces.append(f"{label}: {val}")
        if not pieces:
            desc = str(b.get("description") or "").strip()
            pieces.append(desc or _fill(f"operational definition of \"{name}\""))
        try:
            if float(b.get("min_duration_sec")) > 0:
                pieces.append(f"Minimum duration: {float(b['min_duration_sec']):.2f} s")
        except (TypeError, ValueError):
            pass
        out.append(f"  {name}, " + " ".join(pieces))
    return "\n".join(out)


def _r_dataset(f: dict[str, Any]) -> str:
    rec = f["recordings"]
    if not rec.get("n_sessions"):
        return (
            "No import manifest was found for this project, so session and subject counts "
            "could not be read. " + _fill("number of sessions, subjects and groups")
        )
    n_s, n_sub = rec["n_sessions"], rec["n_subjects"]
    parts = [
        f"The analysis covers {n_s} recording {_plural(n_s, 'session')} from "
        f"{n_sub} {_plural(n_sub, 'subject')} ({_fill('group / treatment assignment')})."
    ]
    vid = []
    if rec["fps_values"]:
        vid.append(
            f"{rec['fps_values'][0]:g} fps"
            if len(rec["fps_values"]) == 1
            else f"{min(rec['fps_values']):g}–{max(rec['fps_values']):g} fps"
        )
    if rec["resolutions"]:
        res = rec["resolutions"]
        vid.append(
            f"{res[0][0]}×{res[0][1]} px"
            if len(res) == 1
            else f"{len(res)} different frame sizes ({_fill('report the relevant one')})"
        )
    if vid:
        parts.append(f"Videos were recorded at {_join(vid)}.")
    if rec.get("total_minutes"):
        parts.append(
            f"Total recording time was {rec['total_minutes']:.1f} min "
            f"(mean {rec['mean_minutes']:.1f} min per session)."
        )
    if rec.get("individuals"):
        parts.append(
            f"Pose files carry {len(rec['individuals'])} tracked individuals per video; "
            f"features and labels are computed per individual."
        )
    parts.append(_fill("apparatus, housing, and any pre-trial handling"))
    return " ".join(parts)


def _r_pose(f: dict[str, Any]) -> str:
    rec = f["recordings"]
    kp = rec.get("keypoints") or []
    fmt = ", ".join(f.upper() for f in (rec.get("pose_formats") or [])) or "-"
    tracker = _fill(
        "pose-estimation software and version, e.g. DeepLabCut 2.3.x or SLEAP 1.3.x"
    )
    parts = [
        f"Body-part positions were tracked with {tracker}, and the resulting tracks were "
        f"imported into ABEL as {fmt} files."
    ]
    if kp:
        parts.append(
            f"{len(kp)} {_plural(len(kp), 'keypoint')} were tracked per animal: {_join(kp)}."
        )
    else:
        parts.append(_fill("the tracked keypoints"))

    sm = rec.get("smoothing") or {}
    bits = []
    if sm.get("likelihood_threshold") is not None:
        bits.append(
            f"points below a tracking likelihood of "
            f"{float(sm['likelihood_threshold']):g} were discarded"
        )
    if sm.get("interpolate_dropouts"):
        bits.append(
            f"gaps of up to {int(sm.get('interpolate_max_gap') or 0)} frames were "
            f"linearly interpolated"
        )
    if sm.get("smoothing_window"):
        bits.append(
            f"and coordinates were smoothed with a {int(sm['smoothing_window'])}-frame "
            f"rolling window"
        )
    if bits:
        parts.append("Before feature extraction, " + ", ".join(bits) + ".")
    _absence = int(sm.get("absence_max_fill_frames") or 0)
    if _absence > 0:
        parts.append(
            f"Animals undetected for more than {_absence} consecutive frames were "
            "treated as absent rather than held at their last known pose, and "
            "analysis windows containing any frame in which the animal was absent "
            "were excluded."
        )
    return " ".join(parts)


def _r_features(f: dict[str, Any]) -> str:
    bm = f["behavior_model"]
    fps = f["fps"]
    win = int(bm.get("segment_window_frames") or 0)
    stride = int(bm.get("segment_stride_frames") or 0)
    if win > 0 and stride > 0:
        window_txt = (
            f"windows of {win} frames ({_sec(win, fps)} s at {fps:g} fps) with a stride of "
            f"{stride} frames ({_sec(stride, fps)} s)"
        )
    else:
        window_txt = "windows of " + _fill("window and stride length")
    parts = [
        f"Pose tracks were divided into overlapping {window_txt}, and each window was "
        f"summarized by a fixed-length feature vector."
    ]

    families = ["per-keypoint kinematics (speed, acceleration, oscillation power)"]
    inv = bm.get("invariant_features") or {}
    inv_bits = []
    if inv.get("enable_egocentric_kinematics", True):
        inv_bits.append("egocentric (heading-relative) velocity")
    if inv.get("enable_body_length_normalization", True):
        inv_bits.append("body-length normalization of all distances")
    if inv.get("enable_relative_geometry", True):
        inv_bits.append("all pairwise inter-keypoint distances")
    if inv.get("enable_head_direction", True):
        inv_bits.append("head direction")
    if inv.get("enable_joint_angles", True):
        inv_bits.append("joint angles")
    if inv.get("enable_spine_curvature", True):
        inv_bits.append("spine curvature")
    if inv_bits:
        families.append(
            "size- and orientation-invariant posture descriptors (" + _join(inv_bits) + ")"
        )
    if bm.get("advanced_roi_features", True) and (f.get("roi") or {}).get("has_zones"):
        families.append(
            "zone-relative context features (inside/outside, signed distance to the zone "
            "boundary, nearest-corner distance, position within the zone)"
        )
    if bm.get("use_video_features", True):
        families.append("dense optical flow and local pixel-motion statistics (Farnebäck)")
        if bm.get("use_r3d_features", True):
            families.append(
                "512-dimensional appearance embeddings from an R3D-18 spatiotemporal CNN "
                "pretrained on Kinetics-400"
            )
    if int(f["project"].get("num_animals") or 1) > 1:
        families.append("inter-animal social features (distance, approach, relative heading)")
    parts.append("Features fell into the following families: " + _join(families) + ".")

    mods = f.get("feature_modalities") or {}
    if mods.get("n_features"):
        label = {
            "pose": "pose geometry", "kinematics": "kinematics", "video": "video",
            "context": "environment/ROI", "social": "social",
        }
        by = mods.get("by_modality") or {}
        if by:
            breakdown = _join([f"{v:,} {label.get(k, k)}" for k, v in sorted(by.items())])
            parts.append(
                f"Trained models used {mods['n_features']:,} features per window ({breakdown})."
            )
        else:
            parts.append(f"Trained models used {mods['n_features']:,} features per window.")
    if not bm.get("use_video_features", True):
        parts.append(
            "Video-derived (pixel) features were disabled for this project; classification "
            "used pose-derived features only."
        )
    return " ".join(parts)


def _r_roi(f: dict[str, Any]) -> str:
    roi = f.get("roi") or {}
    if not roi.get("has_zones"):
        return ""
    n = int(roi["count"])
    # Zones are stored once per subject, so the raw counts are copies of the same
    # slot; only the distinct shape kinds are worth reporting.
    names = {"rect": "rectangular", "circle": "circular", "polygon": "hand-drawn polygonal"}
    kinds = sorted({names.get(k, k) for k in (roi.get("shapes") or {})})
    shape_txt = f" ({_join(kinds)})" if kinds else ""
    parts = [
        f"{n} target {_plural(n, 'zone')} of interest{shape_txt} "
        f"{'was' if n == 1 else 'were'} drawn on a reference frame for each subject "
        f"({_fill('what the zone represents, e.g. open arms, novel-object zone')}), and every "
        f"window carried that zone's occupancy and distance features."
    ]
    if roi.get("motion_radius_px"):
        parts.append(
            f"Local pixel-motion features were computed within "
            f"{2 * int(roi['motion_radius_px'])} × {2 * int(roi['motion_radius_px'])} px "
            f"windows centered on the body centroid and nose, with MOG2 background "
            f"subtraction (history 200 frames, variance threshold "
            f"{int(roi.get('bg_var_threshold', 16))}) applied to that moving window "
            f"rather than the full frame "
            f"(windows kept a constant size at the frame edge by replicating border "
            f"pixels, and held at the last tracked position through keypoint dropouts)."
        )
    if roi.get("excluded_days"):
        parts.append(
            f"Sessions labeled {_join([str(d) for d in roi['excluded_days']])} were excluded "
            f"from zone-based analyses."
        )
    return " ".join(parts)


_SPLIT_LABELS = {
    "group_shuffle_session": (
        "a grouped random split in which whole sessions were assigned to either the "
        "training or the held-out set, so no session contributed windows to both"
    ),
    "group_shuffle_subject": (
        "a grouped random split in which whole subjects were assigned to either the "
        "training or the held-out set, so no animal contributed windows to both"
    ),
    "leave_one_subject_out": (
        "leave-one-subject-out cross-validation, in which each animal in turn was held "
        "out and scored by a model trained on the remaining animals"
    ),
}


def _r_training(f: dict[str, Any]) -> str:
    bm = f["behavior_model"]
    split = str(bm.get("evaluation_split_strategy") or "group_shuffle_session")
    parts = [
        "Each behavior model was evaluated with "
        + _SPLIT_LABELS.get(split, f"the '{split}' split strategy")
        + ". Windows from the same animal are strongly correlated, so a random "
        "window-level split would overstate accuracy."
    ]
    rows = [r for r in f["model_rows"] if r.get("n_train")]
    if rows:
        parts.append(
            f"Models were fitted on {_count(rows[0].get('n_train'))} labeled windows and "
            f"scored on {_count(rows[0].get('n_val'))} held-out windows."
        )
    calib = str(bm.get("calibration_method") or "sigmoid")
    if calib != "none":
        parts.append(
            f"Predicted probabilities were calibrated on the held-out split "
            f"({'Platt scaling' if calib == 'sigmoid' else 'isotonic regression'})."
        )
    if bm.get("enable_feature_augmentation", True):
        parts.append(
            f"Positive training examples were augmented with "
            f"{int(bm.get('augmentation_copies') or 3)} synthetic copies each, adding Gaussian "
            f"noise at {float(bm.get('augmentation_jitter_sigma') or 0.05):.0%} of every "
            f"feature's standard deviation and randomly zeroing "
            f"{float(bm.get('augmentation_dropout_prob') or 0.10):.0%} of features per copy, "
            f"to reduce overfitting to tracking noise."
        )
    if bm.get("hard_negative_sampling_ratio"):
        parts.append(
            f"Negatives were enriched with hard negatives (windows the model scored highly "
            f"but a reviewer rejected) at a ratio of "
            f"{float(bm['hard_negative_sampling_ratio']):g}."
        )
    if any(str(r.get("model_version") or "-") != "-" for r in f["model_rows"]):
        parts.append(
            "All reported metrics come from the held-out split; the model actually applied "
            "to the full dataset was then refit on all labeled windows, so every reviewer "
            "correction reaches inference."
        )
    return " ".join(parts)


def _r_performance(f: dict[str, Any]) -> str:
    rows = [
        r for r in f["model_rows"]
        if not _is_no_behavior(str(r.get("behavior_id", "")))
        and str(r.get("model_version") or "-") != "-"
    ]
    if not rows:
        return (
            "No trained behavior models were found in this project, so no performance "
            "metrics could be read. " + _fill("model performance")
        )
    rows.sort(key=lambda r: str(r.get("behavior_name") or ""))
    width = max(max(len(str(r.get("behavior_name") or "")) for r in rows), len("Behavior"))

    header = f"  {'Behavior'.ljust(width)}  Precision  Recall      F1    PR-AUC   Labels   Basis"
    lines = [header, "  " + "-" * (len(header) - 2)]
    any_macro = False
    for r in rows:
        basis = str(r.get("metrics_basis") or "macro")
        any_macro = any_macro or basis != "target"
        lines.append(
            f"  {str(r.get('behavior_name') or '?').ljust(width)}  "
            f"{_num(r.get('frame_precision')):>9}  "
            f"{_num(r.get('frame_recall')):>6}  "
            f"{_num(r.get('frame_f1')):>6}  "
            f"{_num(r.get('pr_auc')):>8}  "
            f"{_count(r.get('n_positive_labels')):>7}   {basis}"
        )

    intro = (
        "Classifier performance was measured on the held-out split at the window level. "
        "Precision, recall and F1 are reported for the target class, and PR-AUC is the area "
        "under the precision–recall curve (average precision), which is the informative "
        "summary for the rare-positive problem behavior scoring poses. \"Labels\" is the "
        "number of accepted positive windows behind each model."
    )
    notes = []
    if any_macro:
        notes.append(
            "Rows marked \"macro\" average the target and not-target classes; because the "
            "held-out set is overwhelmingly not-target, those precision/recall/F1 values sit "
            "well above the target-class values and are not comparable to the rows marked "
            "\"target\". Retrain those behaviors to obtain target-class scores before "
            "reporting them."
        )
    refined = [r for r in rows if _num(r.get("refined_f1")) != "-"]
    if refined:
        notes.append(
            "Window-level scores are before temporal refinement. After the bout "
            "post-processing described below, target-class F1 was "
            + _join([f"{r['behavior_name']} {_num(r.get('refined_f1'))}" for r in refined])
            + "."
        )
    notes.append(_fill("state explicitly which of these numbers you report, and at what threshold"))
    return intro + "\n\n" + "\n".join(lines) + "\n\n" + "\n".join(notes)


def _r_temporal(f: dict[str, Any]) -> str:
    fps = f["fps"]
    rows = [
        r for r in f["model_rows"]
        if not _is_no_behavior(str(r.get("behavior_id", ""))) and r.get("refined_settings")
    ]
    default = f.get("temporal_default") or {}
    if not rows and not default:
        return ""
    parts = [
        "Per-window probabilities were converted into behavior bouts by thresholding the "
        "probability trace, merging bouts separated by less than a merge gap, and discarding "
        "bouts shorter than a minimum duration. Thresholds were tuned per behavior against "
        "held-out reviewer labels."
    ]
    if rows:
        rows.sort(key=lambda r: str(r.get("behavior_name") or ""))
        width = max(max(len(str(r.get("behavior_name") or "")) for r in rows), len("Behavior"))
        header = (
            f"  {'Behavior'.ljust(width)}   Onset          Min bout         Merge gap"
        )
        lines = [header, "  " + "-" * (len(header) - 2)]
        for r in rows:
            cfg = r.get("refined_settings") or {}
            mb = int(cfg.get("min_bout_duration_frames") or 0)
            mg = int(cfg.get("merge_gap_frames") or 0)
            lines.append(
                f"  {str(r.get('behavior_name') or '?').ljust(width)}   "
                f"{_num(cfg.get('onset_threshold'), 2):>5}   "
                f"{mb:>3} fr ({_sec(mb, fps)} s)   {mg:>3} fr ({_sec(mg, fps)} s)"
            )
        parts.append("\n".join(lines))
    else:
        parts.append(
            f"  Onset threshold {_num(default.get('onset_threshold'), 2)}; minimum bout "
            f"{int(default.get('min_bout_duration_frames') or 0)} frames; merge gap "
            f"{int(default.get('merge_gap_frames') or 0)} frames (applied to all behaviors)."
        )
    return "\n\n".join(parts)


def _r_active_learning(f: dict[str, Any]) -> str:
    bm = f["behavior_model"]
    strategy = str(bm.get("query_strategy") or "uncertainty")
    strategy_label = {
        "uncertainty": (
            "uncertainty sampling (a weighted combination of predictive entropy, ensemble "
            "disagreement and feature-space density)"
        ),
        "prototype": "prototype sampling (clips closest to the current positive centroid)",
        "novelty": "novelty sampling (clips furthest from anything already labeled)",
        "low_probability": "low-probability sampling (clips the model is likely missing)",
        "random_absent": "random sampling from windows with no current prediction",
    }.get(strategy, f"{strategy} sampling")
    parts = [
        f"Training labels were collected by active learning. Starting from a small set of "
        f"manually seeded examples, each round ranked every unlabeled window by "
        f"{strategy_label}, extracted the top-ranked windows as short video clips, and "
        f"presented them to a human reviewer who accepted, rejected or relabeled each one; "
        f"the model was then retrained on the enlarged label set and the cycle repeated."
    ]
    n_query = int(bm.get("active_learning_query_size") or 0)
    if n_query:
        parts.append(f"Each round proposed up to {n_query:,} clips for review.")
    rows = [
        r for r in f["model_rows"]
        if not _is_no_behavior(str(r.get("behavior_id", ""))) and r.get("n_positive_labels")
    ]
    if rows:
        rows.sort(key=lambda r: str(r.get("behavior_name") or ""))
        counts = _join([f"{r['behavior_name']} {_count(r.get('n_positive_labels'))}" for r in rows])
        total = sum(int(r.get("n_positive_labels") or 0) for r in rows)
        parts.append(
            f"Reviewing ended with {_count(total)} accepted positive window labels "
            f"({counts}), plus the negative class."
        )
    parts.append(_fill("who reviewed the clips, and whether reviewers were blind to condition"))
    return " ".join(parts)


def _r_reliability(f: dict[str, Any]) -> str:
    runs = f.get("reliability") or []
    if not runs:
        return (
            "No completed validation quizzes were found in this project, so rater agreement "
            "was not measured. Run one from the Validation tab, or delete this paragraph."
        )
    parts = [
        "Rater agreement was measured with ABEL's blind validation quiz: clips were drawn "
        "from the model's own predictions and from previously accepted labels, stripped of "
        "their model label, and re-scored by human reviewers."
    ]
    for run in runs:
        inter = run.get("inter_rater") or {}
        intra = run.get("intra_rater") or {}
        bits = [f"Quiz {run.get('run_id')}: {_count(run.get('n_clips'))} clips"]
        if inter.get("kappa") is not None and int(inter.get("n_reviewers") or 0) >= 2:
            n_rev = int(inter["n_reviewers"])
            stat = "Cohen's κ" if n_rev == 2 else "Fleiss' κ"
            bits.append(
                f"inter-rater agreement across {n_rev} reviewers {stat} = "
                f"{_num(inter.get('kappa'), 2)} ({_num(inter.get('agreement'), 2)} raw "
                f"agreement on {_count(inter.get('shared_clips'))} shared clips)"
            )
        for rid, v in (intra or {}).items():
            if (v or {}).get("kappa") is None:
                continue
            bits.append(
                f"intra-rater (test–retest) agreement for reviewer '{rid}' κ = "
                f"{_num(v.get('kappa'), 2)} on {_count(v.get('n'))} repeated clips"
            )
        for rid, agree in (run.get("reviewer_agreement") or {}).items():
            if agree is None:
                continue
            bits.append(f"reviewer '{rid}' agreed with the model on {_num(agree, 2)} of clips")
        parts.append("; ".join(bits) + ".")
    parts.append(
        "Interpret κ on the conventional scale (Landis & Koch 1977: 0.61–0.80 substantial, "
        ">0.80 almost perfect). "
        + _fill("say how many reviewers scored, and how they were trained")
    )
    return "\n".join(parts)


def _r_hmm(f: dict[str, Any]) -> str:
    m = f.get("motif") or {}
    if not m:
        return ""
    if str(m.get("hmm_n_states_mode") or "auto") == "manual":
        states = f"a fixed {int(m.get('hmm_n_states') or 4)} states"
    else:
        crit = str(m.get("hmm_criterion") or "bic").upper()
        crit_label = {
            "ICL": "the integrated completed likelihood (ICL)",
            "BIC": "the Bayesian information criterion (BIC)",
            "AIC": "the Akaike information criterion (AIC)",
            "AICC": "the small-sample-corrected AIC (AICc)",
            "CV": "leave-one-session-out cross-validated held-out log-likelihood",
        }.get(crit, crit)
        states = (
            f"a state count selected over {int(m.get('hmm_n_states_min') or 2)}–"
            f"{int(m.get('hmm_n_states_max') or 8)} by {crit_label}"
        )
    occ_label = (
        "forward–backward posterior probabilities (expected fractional occupancy)"
        if str(m.get("hmm_occupancy_method") or "viterbi") == "posterior"
        else "the Viterbi maximum-likelihood state path"
    )
    return (
        f"Sequential structure in the behavior stream was modeled with a categorical hidden "
        f"Markov model over each subject's ordered bout sequence (hmmlearn), fitted by "
        f"Baum–Welch expectation–maximization with {states}, "
        f"{int(m.get('hmm_n_restarts') or 5)} random restarts of up to "
        f"{int(m.get('hmm_n_iter') or 200):,} iterations each (seed "
        f"{int(m.get('hmm_random_seed') or 0)}; the restart with the highest log-likelihood "
        f"was kept), with per-session fractional state occupancy taken from {occ_label} and "
        f"discrete state bouts taken from the Viterbi path."
    )


def _r_motifs(f: dict[str, Any]) -> str:
    m = f.get("motif") or {}
    if not m:
        return ""
    parts = []
    method = str(m.get("motif_method") or "both")
    if method in ("ngram", "both"):
        lo = int(m.get("ngram_min_n") or 2)
        hi = int(m.get("ngram_max_n") or 4)
        span = f"{lo}" if lo == hi else f"{lo}–{hi}"
        parts.append(
            f"Recurring behavior sequences were identified as n-grams of length {span} over "
            f"the ordered bout sequence, keeping the {int(m.get('ngram_top_k') or 15)} most "
            f"frequent motifs occurring at least {int(m.get('min_ngram_count') or 2)} times."
        )
    if method in ("sequence_clustering", "both"):
        parts.append(
            f"Sessions were additionally embedded from their "
            f"{int(m.get('cluster_ngram_n') or 3)}-gram profiles with UMAP "
            f"({int(m.get('umap_n_components') or 10)} components, "
            f"{int(m.get('umap_n_neighbors') or 10)} neighbors, min_dist "
            f"{float(m.get('umap_min_dist') or 0.1):g}) and clustered with HDBSCAN "
            f"(minimum cluster size {int(m.get('hdbscan_min_cluster_size') or 3)})."
        )
    corr_label = (
        "Benjamini–Hochberg false-discovery-rate correction"
        if str(m.get("transition_pval_correction") or "fdr_bh") == "fdr_bh"
        else "no multiple-comparison correction"
    )
    parts.append(
        f"Behavior-to-behavior transition matrices were tested against a null built by "
        f"{int(m.get('n_permutations') or 1000):,} random permutations of the bout sequence "
        f"(seed {int(m.get('permutation_seed') or 42)}); p-values are (b+1)/(m+1) so they are "
        f"never reported as exactly zero, and cell-wise p-values were adjusted with "
        f"{corr_label}."
    )
    return " ".join(parts)


def _r_stats(f: dict[str, Any]) -> str:
    return (
        "Per-session behavior metrics (bout count, total and mean bout duration, latency, "
        "percentage of session time) were exported from ABEL and compared between groups "
        "with an independent two-sample t-test for two groups, or one-way ANOVA followed by "
        "Šidák-corrected pairwise t-tests for more than two. The unit of analysis is the "
        "session. "
        + _fill(
            "your actual test, the software you ran it in, the alpha level, and any "
            "repeated-measures or covariate structure: ABEL's built-in tests are a "
            "screening tool, not a substitute for a designed analysis"
        )
    )


def _r_software_stack(f: dict[str, Any]) -> str:
    libs = f.get("libraries") or {}
    if not libs:
        return ""
    listed = ", ".join(f"{name} {ver}" for name, ver in sorted(libs.items()))
    return (
        f"Computation was performed in Python with {listed}. "
        f"ABEL v{f['app_version']} itself is available at {f['repo_url']}."
    )


def _r_reproducibility(f: dict[str, Any]) -> str:
    parts = [
        "Every model, feature table and export in this project carries a provenance record "
        "(model version, feature version, configuration hash and timestamp) written by ABEL "
        "at training time and stored under derived/models/ in the project folder."
    ]
    rows = [
        r for r in f["model_rows"]
        if str(r.get("model_version") or "-") != "-"
        and not _is_no_behavior(str(r.get("behavior_id", "")))
    ]
    if rows:
        rows.sort(key=lambda r: str(r.get("behavior_name") or ""))
        listed = _join([
            f"{r['behavior_name']} ({r.get('model_version')}, trained "
            f"{str(r.get('last_trained') or '?')[:10]})"
            for r in rows
        ])
        parts.append(f"The models reported here are {listed}.")
    m = f.get("motif") or {}
    seeds = []
    if m.get("hmm_random_seed") is not None:
        seeds.append(f"HMM fitting seed {int(m['hmm_random_seed'])}")
    if m.get("permutation_seed") is not None:
        seeds.append(f"permutation seed {int(m['permutation_seed'])}")
    if seeds:
        parts.append(f"Random seeds were fixed ({_join(seeds)}), so analyses rerun identically.")
    parts.append(
        _fill(
            "where the raw videos, pose files and the ABEL project will be deposited, and "
            "under what accession or DOI"
        )
    )
    return " ".join(parts)


_SECTION_RENDERERS = {
    "software": _r_software,
    "ml_approach": _r_ml_approach,
    "behaviors": _r_behaviors,
    "behavior_definitions": _r_behavior_definitions,
    "dataset": _r_dataset,
    "pose": _r_pose,
    "features": _r_features,
    "roi": _r_roi,
    "training": _r_training,
    "performance": _r_performance,
    "temporal": _r_temporal,
    "active_learning": _r_active_learning,
    "reliability": _r_reliability,
    "hmm": _r_hmm,
    "motifs": _r_motifs,
    "stats": _r_stats,
    "software_stack": _r_software_stack,
    "reproducibility": _r_reproducibility,
}


def _refs_for(keys: list[str], facts: dict[str, Any]) -> list[str]:
    """Reference keys cited by the selected sections."""
    wanted: set[str] = set()
    for key in keys:
        wanted.update(_SECTION_REFS.get(key, ()))
    bm = facts.get("behavior_model") or {}
    if "features" in keys and bm.get("use_video_features", True) and bm.get("use_r3d_features", True):
        wanted.update({"tran2018", "kay2017"})
    if "ml_approach" in keys and str(bm.get("classifier_type") or "") == "lightgbm":
        wanted.discard("chen2016")
        wanted.add("ke2017")
    return sorted(wanted)


def _render_citations(keys: list[str], facts: dict[str, Any]) -> str:
    wanted = _refs_for(keys, facts)
    if not wanted:
        return ""
    from abel.ui.methods_content import REFERENCES  # noqa: PLC0415

    by_key = {r.key: r for r in REFERENCES}
    lines = [
        "Cited above (check each against your journal's reference style; the full reference "
        "list with links is in the Methods → References subtab):"
    ]
    for key in wanted:
        r = by_key.get(key)
        if r is not None:
            lines.append(f"  {r.authors} ({r.year}). {r.title}. {r.venue}. {r.url}")
    return "\n".join(lines)


def render_methods_text(
    facts: dict[str, Any],
    keys: list[str] | tuple[str, ...],
    *,
    include_warning: bool = True,
) -> str:
    """Render the selected sections of *facts* as a draft methods write-up.

    Sections appear in :data:`SECTIONS` order regardless of tick order, and a
    section with nothing to say (no ROIs defined, no validation quiz run) is
    dropped rather than emitted empty.
    """
    wanted = set(keys)
    selected = [s for s in SECTIONS if s.key in wanted]
    blocks: list[str] = []
    if include_warning:
        blocks.append(WARNING_TEXT)

    for spec in selected:
        renderer = _SECTION_RENDERERS.get(spec.key)
        if renderer is None:  # "citations" is rendered last, from the others
            continue
        try:
            body = (renderer(facts) or "").strip()
        except Exception as exc:  # pragma: no cover - one bad section must not lose the rest
            body = f"[Could not build this section: {exc}]"
        if body:
            blocks.append(f"[{spec.label}]\n{body}")

    if "citations" in wanted:
        cites = _render_citations([s.key for s in selected], facts)
        if cites:
            blocks.append("[References]\n" + cites)

    if len(blocks) <= (1 if include_warning else 0):
        blocks.append("Select at least one section above, then click Generate draft.")

    project_name = str((facts.get("project") or {}).get("project_name") or "")
    blocks.append(
        f"Drafted by ABEL v{facts.get('app_version', '?')} on {facts.get('generated_on', '')}"
        + (f" from project \"{project_name}\"." if project_name else ".")
        + " Re-read the warning at the top before using any of this."
    )
    return "\n\n".join(blocks) + "\n"
