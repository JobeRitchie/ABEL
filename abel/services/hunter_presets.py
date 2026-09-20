"""Ready-made hunt definitions for the Auto Hunter tab of Targeted Clip Mining.

The Essence Extractor learns a behavior from clips the user has *already*
found.  A preset is the answer to the other half of the problem, the project
where nothing has been reviewed yet and there is nothing to point at.  Each
preset is a hand-written essence: a small set of extracted per-window features
with a direction and a weight, describing a *family* of behavior rather than
one label ("something social is happening here", not "this is Sniff Anogenital").

Two things make a preset portable to a project it was never tuned on:

* **Only schema-fixed features.**  Presets name columns that Feature Extraction
  always produces when its feature family is enabled (the ``social_*`` block is
  the same 284 columns for any multi-animal project), never anything
  project-specific like an ROI or a behavior id.
* **Nothing absolute.**  Weights are applied to a robust z-score taken over the
  *target project's own* pool, and the descriptive ranges are quantiles of that
  same pool.  Nothing carries a millimeter, a pixel scale or an arena size
  across projects, so a preset built against a 55 mm mouse in one rig still
  means "closer together than this animal usually is" in the next.

A preset delivers a ranker plus a descriptive box, and the dialog hunts with the
ranker (see ``ClipMetricsService.mine``'s ``rank_only``): a conjunction of ranges
is a cliff that throws away most of a behavior's real instances, while ranking
merely orders a poor fit late.  The box is shown so the definition stays
inspectable and editable, and it still filters if the user picks All/Any.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from abel.services.clip_metrics_service import (
    ClipMetricsService,
    Criterion,
    rich_column,
    rich_metric_id,
)

# How wide the descriptive box is drawn, as a quantile of the project's pool for
# each bounded feature.  Wider = more of the pool sits inside the ranges.
BREADTH_LEVELS: tuple[tuple[str, float], ...] = (
    ("Balanced", 0.45),
    ("Broad (more recall)", 0.65),
    ("Tight (more precise)", 0.25),
)


@dataclass(frozen=True)
class PresetFeature:
    """One feature in a preset: which column, which way, how much it counts.

    ``columns`` lists the accepted spellings of the same feature, best first,
    the aggregator's naming has changed across schema versions (the
    body-length-normalized distances are written ``..._norm_norm_median`` today
    and were ``..._norm_median`` before), and a preset that only knew one
    spelling would silently drop the feature on the other kind of project.

    ``weight`` is signed: negative means *lower is more behavior-like*.
    ``box`` marks the feature as one of the ones drawn as an editable range;
    ``"low"`` bounds it from above (small values wanted), ``"high"`` from below.
    """

    columns: tuple[str, ...]
    weight: float
    box: str | None = None  # None | "low" | "high"
    note: str = ""


@dataclass(frozen=True)
class HunterPreset:
    """A named, ready-to-run hunt over a family of behaviors."""

    id: str
    name: str
    tagline: str
    description: str
    requires: str
    features: tuple[PresetFeature, ...]
    # Below this many of its features present in the project, the preset is not
    # offered: a proximity hunt missing its proximity columns is not a hunt.
    min_features: int = 3


@dataclass
class PresetScorer:
    """Weighted robust-z ranker over a preset's features.

    Duck-types :class:`~abel.services.clip_metrics_service.EssenceScorer`, the
    dialog only ever asks a ranker for ``score(df)`` and ``active_feature_ids``
   , so a preset hunt rides the exact same ranking, capping and
    subject-spreading path an extracted essence does.

    The center and scale are the pool's own median and MAD, so the score reads
    as "how far this window sits toward the behavior, in units of this
    project's own spread".
    """

    metric_ids: list[str]
    center: dict[str, float]
    scale: dict[str, float]
    weight: dict[str, float]  # signed, normalized to sum |w| = 1

    @property
    def active_feature_ids(self) -> list[str]:
        return list(self.metric_ids)

    def score(self, df: pd.DataFrame) -> pd.Series:
        cols = [m for m in self.metric_ids if m in df.columns]
        if not cols:
            return pd.Series(0.0, index=df.index)
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        total = pd.Series(0.0, index=df.index)
        for m in cols:
            scale = self.scale.get(m) or 1.0
            # Clipped so one wild outlier feature can't dominate the ranking.
            z = ((X[m] - self.center.get(m, 0.0)) / scale).clip(-4, 4)
            total = total.add(z.fillna(0.0) * self.weight.get(m, 0.0), fill_value=0.0)
        return total


@dataclass
class PresetFit:
    """A preset resolved against one project: ranges, ranker and what was found."""

    preset: HunterPreset
    criteria: list[Criterion] = field(default_factory=list)
    scorer: PresetScorer | None = None
    used: list[str] = field(default_factory=list)     # metric ids that resolved
    missing: list[str] = field(default_factory=list)  # preset columns with no match
    note: str = ""

    def usable(self) -> bool:
        return self.scorer is not None


# ---------------------------------------------------------------------------
# The presets
# ---------------------------------------------------------------------------

# Weights below were set against the reviewed clips of a resident-intruder
# project (COACIECNO_social, 439 reviewed windows over 11 social behaviors):
# social-labeled windows separate from everything else at AUC 0.88, and the
# top 50 of the reviewed set run ~3.5x enriched for them over the base rate.
# They are deliberately spread over several proximity signals rather than
# leaning on the single best one, because which one carries a given social
# behavior differs (contact for allogrooming and attack, nose-to-body distance
# for sniffing, closing velocity for approach and chase).
SOCIAL_PRESET = HunterPreset(
    id="social",
    name="Social interaction",
    tagline="Two animals close, touching or oriented at each other",
    description=(
        "Hunts windows where the focal animal is near a conspecific, the "
        "proximity, contact and orientation signals shared by sniffing, "
        "allogrooming, approach, chase, attack and huddling.\n\n"
        "It is a *family* hunt, not a single behavior: expect a mix of social "
        "behaviors back, plus the non-social things animals do while they "
        "happen to be next to each other. Use it to build a first review set on "
        "a project with nothing labeled yet, then switch to Extract essence "
        "once you have exemplars of the one behavior you actually want.\n\n"
        "Everything it measures is relative to this project's own spread, so no "
        "arena size, pixel scale or body size is carried over from the project "
        "it was tuned on."
    ),
    requires="Social features (a multi-animal project with social features enabled)",
    features=(
        PresetFeature(
            columns=(
                "social_min_keypoint_dist_nearest_norm_norm_median",
                "social_min_keypoint_dist_nearest_norm_median",
            ),
            weight=-1.0,
            box="low",
            note="closest body-point gap to the nearest other animal",
        ),
        PresetFeature(
            columns=(
                "social_dist_centroid_to_centroid_nearest_norm_norm_median",
                "social_dist_centroid_to_centroid_nearest_norm_median",
            ),
            weight=-0.6,
            box="low",
            note="body-to-body separation",
        ),
        PresetFeature(
            columns=(
                "social_dist_nose_to_nose_nearest_norm_norm_median",
                "social_dist_nose_to_nose_nearest_norm_median",
            ),
            weight=-0.5,
            note="head-to-head distance (nose contact, facial investigation)",
        ),
        PresetFeature(
            columns=(
                "social_dist_nose_to_tail_base_nearest_norm_norm_median",
                "social_dist_nose_to_tail_base_nearest_norm_median",
            ),
            weight=-0.5,
            note="nose to the other's rear (anogenital and flank investigation)",
        ),
        PresetFeature(
            columns=("social_in_contact_mean",),
            weight=0.8,
            box="high",
            note="fraction of the window spent in body contact",
        ),
        PresetFeature(
            columns=("social_bbox_overlap_nearest_mean",),
            weight=0.5,
            note="how much the two animals' outlines overlap",
        ),
        PresetFeature(
            columns=("social_facing_angle_nearest_median",),
            weight=-0.4,
            note="how squarely the focal animal points at the other",
        ),
    ),
)

PRESETS: tuple[HunterPreset, ...] = (SOCIAL_PRESET,)


def preset_by_id(preset_id: str) -> HunterPreset | None:
    return next((p for p in PRESETS if p.id == preset_id), None)


# ---------------------------------------------------------------------------
# Resolving a preset against one project
# ---------------------------------------------------------------------------


def _resolve_columns(
    preset: HunterPreset, available: set[str]
) -> tuple[list[tuple[PresetFeature, str]], list[str]]:
    """``(feature, metric_id)`` for every preset feature the project has."""
    found: list[tuple[PresetFeature, str]] = []
    missing: list[str] = []
    for feat in preset.features:
        mid = next(
            (rich_metric_id(c) for c in feat.columns if rich_metric_id(c) in available),
            None,
        )
        if mid is None:
            missing.append(feat.columns[0])
        else:
            found.append((feat, mid))
    return found, missing


def fit_preset(
    metrics: ClipMetricsService,
    preset: HunterPreset,
    breadth: float = 0.45,
) -> PresetFit:
    """Resolve *preset* against the project behind *metrics*.

    Reads only the preset's own columns for the whole feature table (a handful
    of columns, so a fraction of a second even on a 200k-window project), then
    derives the ranker's center/scale and the descriptive box's bounds from that
    pool.  A project missing too many of the preset's features gets an
    unusable fit whose ``note`` says which ones were absent.
    """
    fit = PresetFit(preset=preset)
    available = set(metrics.rich_feature_ids())
    if not available:
        fit.note = (
            "This project has no extracted feature table yet, run Feature "
            "Extraction before using a preset hunt."
        )
        return fit
    found, missing = _resolve_columns(preset, available)
    fit.missing = missing
    if len(found) < preset.min_features:
        fit.note = (
            f"Only {len(found)} of this preset's {len(preset.features)} features "
            f"exist in this project ({preset.requires.lower()}), so it can't run "
            "here. Missing: " + ", ".join(rich_column(m) for m in missing[:4])
        )
        return fit

    metric_ids = [mid for _f, mid in found]
    pool = metrics.load_rich_features(metric_ids=metric_ids)
    if pool.empty:
        fit.note = "Could not read the extracted feature table for this project."
        return fit

    X = pool[[m for m in metric_ids if m in pool.columns]].apply(
        pd.to_numeric, errors="coerce"
    )
    center = X.median()
    # Robust spread (MAD), falling back to the plain std for a column the MAD
    # calls zero: a near-constant feature would otherwise divide the whole
    # ranker by nothing.
    scale = (X - center).abs().median() * 1.4826
    scale = scale.replace(0, np.nan).fillna(X.std()).replace(0, 1.0).fillna(1.0)

    total = sum(abs(f.weight) for f, _m in found) or 1.0
    fit.scorer = PresetScorer(
        metric_ids=list(X.columns),
        center={m: float(center[m]) for m in X.columns},
        scale={m: float(scale[m]) for m in X.columns},
        weight={m: float(f.weight / total) for f, m in found if m in X.columns},
    )
    fit.used = list(X.columns)

    # Descriptive ranges: a quantile of this project's own pool, on the side the
    # behavior sits.  Only the boxed features get one, a range on every
    # feature reads as a definition far tighter than the preset really is.
    lo_q = float(min(max(breadth, 0.02), 0.98))
    for feat, mid in found:
        if feat.box is None or mid not in X.columns:
            continue
        col = X[mid].dropna()
        if col.empty:
            continue
        if feat.box == "low":
            fit.criteria.append(Criterion(mid, None, float(col.quantile(lo_q))))
        else:
            fit.criteria.append(Criterion(mid, float(col.quantile(1.0 - lo_q)), None))

    bits = [
        f"Armed “{preset.name}” on {len(fit.used)} feature(s) over "
        f"{len(pool):,} scored window(s)."
    ]
    if missing:
        bits.append(
            f"{len(missing)} of the preset's features aren't in this project and "
            "were left out."
        )
    fit.note = " ".join(bits)
    return fit
