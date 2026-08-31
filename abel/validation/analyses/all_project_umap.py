"""All-project UMAP — one embedding holding every labeled clip from every project.

The other analyses ask "how well is behavior *X* in project *P* detected?".  This
one asks the map question instead: **laid side by side in one feature space, where
do all the assays' behaviors sit relative to each other?**  Every labeled clip in
every selected project becomes one point; the reducer places it; the figure colours
it by behavior (or by project) and labels the clusters.

Three things make that honest rather than decorative, and each is a setting rather
than a hidden default:

**One feature space, chosen explicitly.**  Projects do not share a column set —
across the eight manuscript projects the union is 5172 columns and the intersection
1282 (all 512 R3D dims included).  ``feature_space="shared"`` embeds on the
intersection, so every point is described by the same measurements and a cluster
cannot be an artifact of a column only one project has.  ``"union"`` keeps every
column and fills the gaps, which is faster to set up and much easier to misread —
a project-shaped blob is then guaranteed.  Shared is the default for that reason.

**Project identity is a confound, and it is measured.**  A map where the clusters
are the projects has told you about cameras and arenas, not behavior.  Every run
therefore reports behavior *and* project structure side by side (silhouette + kNN
purity, in the feature space and in the embedding), so the figure is never read
without the number that says whether it is a behavior map or a project map.
``per_project_zscore`` is the knob that trades one for the other: it centres each
project's features on that project's own mean, which removes rig offsets and also
removes any real between-assay difference.

**Behaviors are assay-scoped by default.**  An EPM ``Rear`` and an open-field
``Rear`` are different measurements of arguably different acts, and the rest of this
suite never pools them (see :mod:`abel.validation.analyses.behaviorscape`).  Neither
does this: groups are keyed ``"<project> · <behavior>"`` unless
``pool_behaviors_by_name`` is switched on deliberately.

Rendering lives in :mod:`abel.validation.umap_plot` and reads the saved
``embedding.parquet``, so restyling a figure — label distance, colours, hulls —
never re-runs the reducer.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from abel.validation import holdout as vholdout
from abel.validation.datamodel import ProjectRef
from abel.validation.features import (
    MODALITY_CONTEXT,
    MODALITY_KINEMATICS,
    MODALITY_POSE,
    MODALITY_SOCIAL,
    classify_modality,
    is_r3d_column,
    is_video_feature,
)

logger = logging.getLogger("abel")

#: Columns that are metadata, not features (mirrors the trainer's ignore/forbidden
#: sets so a name-only prefilter can run off the parquet schema, before any read).
META_COLS = {
    "segment_id", "label", "label_source", "reviewer_confidence",
    "animal_id", "session_id", "start_frame", "end_frame",
}
_FORBIDDEN = {
    "prediction_prob", "prediction_prob_fused", "prediction_variance",
    "density_outlier_score", "overlap_allowed", "overlap_allowed_x",
    "overlap_allowed_y", "label_true", "label_pred",
}

NO_BEHAVIOR = "no_behavior"


# ── Settings ────────────────────────────────────────────────────────────────


@dataclass
class EmbeddingSettings:
    """Everything that changes the *coordinates*.

    Split from :class:`abel.validation.umap_plot.PlotSettings` on purpose: changing
    anything here means recomputing the embedding (slow), changing anything there
    is a re-render of the saved coordinates (instant).
    """

    # ── Which rows enter the map ──
    #: Include the ``no_behavior`` background class.  Off by default: it is the
    #: largest and least interesting group and it swamps the colour legend, but
    #: turning it on is the only way to see whether a behavior is actually
    #: separable from "nothing in particular" rather than just from other behaviors.
    include_no_behavior: bool = False
    #: Cap on rows contributed per (project, behavior).  Without it the map is
    #: dominated by whichever assay labelled the most, and UMAP's local structure
    #: follows density.  0 = no cap.
    max_rows_per_behavior: int = 1500
    #: Global row cap applied after the per-behavior cap (0 = none).  UMAP is
    #: roughly O(n log n); ~50k rows on ~1300 features takes a few minutes.
    max_rows_total: int = 50000
    #: Drop any (project, behavior) with fewer than this many rows — a 3-point
    #: cluster gets a label and a hull and means nothing.
    min_rows_per_behavior: int = 20
    #: Keep only rows at or above this reviewer confidence (0 = keep all).  The
    #: held-out analyses use 1.0; a map usually wants everything that was labelled.
    min_confidence: float = 0.0
    #: Drop ``imported:*`` rows — clips copied in from another project.  Leaving
    #: them in double-counts the same clip under two project names.
    exclude_imported: bool = True
    #: Drop ``temporal_feedback`` rows (reviewer FP/FN corrections).  These are
    #: deliberately atypical examples, so they sit at cluster edges.
    exclude_refine_only: bool = False
    #: Seed for the row subsampling.
    sample_seed: int = 42

    # ── Which columns describe a point ──
    use_pose: bool = True
    use_kinematics: bool = True
    use_context: bool = False
    use_video: bool = True
    #: OFF by default, unlike everywhere else in the suite. R3D is 512 columns of
    #: raw-pixel appearance, so it encodes the rig — lighting, camera angle, arena,
    #: bedding — and after z-scoring it outvotes the ~430 pose/kinematic columns.
    #: Measured across the eight manuscript projects it takes kNN project purity
    #: from 0.58 to 0.97 (chance 0.13) and collapses cross-assay neighbour mixing
    #: from ~50% of neighbours to 1.4%: the map becomes eight islands, one per
    #: camera. Switch it on only to ask a question about appearance itself.
    use_r3d: bool = False
    use_social: bool = False
    #: ``shared`` = intersect columns across all projects (comparable, the default).
    #: ``union`` = keep every column and fill what a project lacks (see nan_policy).
    feature_space: str = "shared"
    #: Drop columns that never vary across the pooled rows — they cost distance
    #: computation and contribute nothing.
    drop_zero_variance: bool = True
    #: Drop rows whose entire R3D block is zero.  Two known paths write all-zero
    #: R3D (reviewed non-grid segments; dense inference before the zero-fill fix),
    #: and 512 shared zeros pull those rows into one dense artificial cluster.
    #: Only applies when ``use_r3d`` is on.
    drop_zero_r3d_rows: bool = True
    #: ``median`` / ``zero`` fill missing values, ``drop_columns`` removes any
    #: column with a missing value anywhere (safe, but under ``union`` that
    #: collapses back to roughly the shared set).
    nan_policy: str = "median"
    #: Keep only the N highest-variance columns after scaling (0 = keep all).
    #: A cheap way to stop the 512 R3D dims outvoting ~700 pose columns.
    max_features: int = 0

    # ── Putting the columns on a common footing ──
    #: ``zscore`` (mean/sd) · ``robust`` (median/IQR, better with outliers) ·
    #: ``minmax`` · ``none``.  Never leave this at ``none`` with mixed units.
    scaler: str = "zscore"
    #: Standardize *within each project* before pooling.  Removes per-rig offsets
    #: (lighting, camera height, arena scale) — and removes real between-assay
    #: differences with them.  Check the project-structure QC numbers both ways.
    per_project_zscore: bool = False
    #: PCA pre-reduction before the reducer (0 = off).  50 is the conventional
    #: default: it denoises, makes UMAP several times faster, and barely changes
    #: the layout.  Raise it if the QC says variance explained is low.
    pca_components: int = 50
    pca_whiten: bool = False

    # ── The reducer ──
    #: ``umap`` (needs ``umap-learn``) · ``tsne`` · ``pca``.  t-SNE and PCA are
    #: fallbacks so the tab still produces a map on a machine without umap-learn;
    #: neither preserves global structure the way UMAP claims to.
    reducer: str = "umap"
    #: Size of the local neighbourhood. Low (5-15) = fine local detail, fragmented
    #: clusters. High (50-200) = global structure, blobs merge. This is the single
    #: most consequential UMAP setting. 60 measured best here: it maximizes the
    #: share of neighbours that both cross projects and share a behavior (0.137,
    #: vs 0.118 at 15 and 0.132 at 120), which is the quantity an all-project map
    #: exists to show.
    n_neighbors: int = 60
    #: How tightly points may pack within a cluster (0.0-0.99). Low = dense knots
    #: (good for counting clusters), high = evenly spread (good for reading labels).
    #: 0.20 pairs with the n_neighbors default: enough separation to see lobes,
    #: enough packing that a dense region still reads as dense.
    min_dist: float = 0.20
    #: Scale the embedding is spread over; with min_dist it sets cluster separation.
    spread: float = 1.0
    #: 2 for a figure. 3 only if you plan to export coordinates for something else.
    n_components: int = 2
    #: ``euclidean`` after z-scoring is standard. ``cosine`` ignores overall
    #: magnitude — often better when R3D embeddings dominate the column count.
    #: ``correlation`` and ``manhattan`` are also accepted.
    metric: str = "euclidean"
    #: Fuzzy-union vs fuzzy-intersection of the local simplicial sets (1.0 = pure
    #: union). Below 1.0 breaks weakly-connected regions apart.
    set_op_mix_ratio: float = 1.0
    #: How many neighbours are assumed fully connected. Raise (2-4) if a
    #: high-dimensional feature space shatters into speckle.
    local_connectivity: float = 1.0
    #: Weight on negative samples during layout. Higher = more empty space between
    #: clusters (and more distortion of within-cluster distances).
    repulsion_strength: float = 1.0
    #: Negative samples per positive edge. Higher = cleaner gaps, slower.
    negative_sample_rate: int = 5
    #: Optimization epochs (0 = UMAP's own default: 500 small / 200 large).
    #: Raise to 500-1000 if the layout still looks stringy.
    n_epochs: int = 0
    #: ``spectral`` (deterministic, better global structure) or ``random``.
    init: str = "spectral"
    #: densMAP: preserve local density so a tight cluster *looks* tight. Off by
    #: default because it changes what area on the page means.
    densmap: bool = False
    dens_lambda: float = 2.0
    #: Fixes the layout across runs. UMAP disables its parallelism when set, so
    #: clearing it (-1) is faster but no longer reproducible.
    random_state: int = 42
    #: t-SNE only: effective neighbourhood size (5-50).
    tsne_perplexity: float = 30.0

    # ── Grouping ──
    #: Off (default): groups are ``"<project> · <behavior>"``, so an EPM Rear and
    #: an open-field Rear stay separate — the convention the rest of the suite
    #: enforces. On: same-named behaviors merge into one group across projects.
    pool_behaviors_by_name: bool = False
    #: Raw behavior name → pooled name, applied before the assay prefix. Lets
    #: "Rearing" and "Rear" merge without renaming anything on disk.
    alias_map: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingSettings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


# ── Result ──────────────────────────────────────────────────────────────────


@dataclass
class EmbeddingResult:
    """Coordinates + provenance + the QC that says how to read the picture."""

    frame: pd.DataFrame            # one row per point: x, y[, z] + metadata
    feature_cols: list[str]
    settings: EmbeddingSettings
    qc: dict[str, float] = field(default_factory=dict)
    counts: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)
    reducer_used: str = ""

    @property
    def n_points(self) -> int:
        return int(len(self.frame))


# ── Column selection ────────────────────────────────────────────────────────


def _wanted_modality(col: str, s: EmbeddingSettings) -> bool:
    """Does this column belong to a family the user switched on?

    R3D is tested before the generic video test because it is a sub-family with its
    own toggle: 512 columns that can outnumber every handcrafted feature combined,
    and its own known all-zero failure mode.
    """
    if is_r3d_column(col):
        return s.use_r3d
    if is_video_feature(col):
        return s.use_video
    mod = classify_modality(col)
    if mod == MODALITY_SOCIAL:
        return s.use_social
    if mod == MODALITY_CONTEXT:
        return s.use_context
    if mod == MODALITY_KINEMATICS:
        return s.use_kinematics
    if mod == MODALITY_POSE:
        return s.use_pose
    return False


def _schema_columns(path: Path) -> set[str]:
    """Column names from a parquet footer, without reading a single row group."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    return set(pq.ParquetFile(str(path)).schema.names)


def resolve_feature_columns(
    projects: list[ProjectRef], s: EmbeddingSettings
) -> tuple[list[str], dict[str, int]]:
    """The column set to embed on, plus per-project counts for the QC report.

    Reads only parquet footers, so choosing a feature space is instant even when
    the training sets are hundreds of megabytes.
    """
    per_project: dict[str, set[str]] = {}
    for p in projects:
        if not p.training_set_path.exists():
            continue
        cols = _schema_columns(p.training_set_path)
        keep = {
            c for c in cols
            if c not in META_COLS
            and c not in _FORBIDDEN
            and not str(c).startswith("uncertainty_")
            and _wanted_modality(c, s)
        }
        per_project[p.project_id] = keep

    if not per_project:
        return [], {}
    sets = list(per_project.values())
    chosen = set.intersection(*sets) if s.feature_space == "shared" else set.union(*sets)
    ordered = sorted(chosen)
    return ordered, {k: len(v) for k, v in per_project.items()}


# ── Row assembly ────────────────────────────────────────────────────────────


def canonical_name_map(names: "Iterable[str]") -> dict[str, str]:
    """Raw behavior name -> one canonical spelling, matched case-insensitively.

    Projects are labelled by different people at different times, so the same act
    reaches this module as ``"Wet Dog Shake"`` in one project and ``"Wet dog shake"``
    in another.  Treated as distinct they are two behaviors: they get two colours in
    every figure, two legend entries, and — worse — the cross-assay conservation
    analysis drops them entirely, because that requires one name to appear in two or
    more assays.  A rare behavior scored in exactly two projects is precisely the
    case a spelling difference silently deletes.

    Whitespace is normalized and case is folded.  The winning spelling is the most
    frequent one (ties broken alphabetically) rather than a lower- or title-cased
    invention, so figures keep a spelling the user actually wrote.
    """
    from collections import Counter  # noqa: PLC0415

    counts: dict[str, Counter] = {}
    originals: dict[str, set] = {}
    for n in names:
        original = str(n)
        squeezed = " ".join(original.split())
        if not squeezed:
            continue
        key = squeezed.casefold()
        counts.setdefault(key, Counter())[squeezed] += 1
        originals.setdefault(key, set()).add(original)
    out: dict[str, str] = {}
    for key, c in counts.items():
        winner = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        # Key by the *original* strings as well as the squeezed ones: a name with a
        # double space must still resolve, and it is the original that callers hold.
        for raw in originals[key] | set(c):
            out[raw] = winner
    return out


def _group_key(project_label: str, behavior_label: str, s: EmbeddingSettings) -> str:
    name = s.alias_map.get(behavior_label, behavior_label)
    return name if s.pool_behaviors_by_name else f"{project_label} · {name}"


def collect_rows(
    projects: list[ProjectRef],
    behaviors: dict[str, list[str]],
    feature_cols: list[str],
    s: EmbeddingSettings,
    progress_cb: Callable[[str, float], None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Pool every selected project's labeled rows into one frame.

    Returns ``(frame, warnings)``.  The frame carries the feature columns plus the
    metadata every downstream step needs (group key, project, behavior, session,
    animal, segment).  Columns a project lacks arrive as NaN and are handled later
    by ``nan_policy`` — that is the whole cost of ``feature_space="union"``.
    """
    warnings: list[str] = []
    rng = np.random.default_rng(int(s.sample_seed))
    parts: list[pd.DataFrame] = []
    total = len(projects) or 1

    for i, project in enumerate(projects):
        if progress_cb:
            progress_cb(f"Reading {project.project_id}…", i / total)
        bids = [str(b) for b in behaviors.get(project.project_id, [])]
        if not bids:
            continue
        if not project.training_set_path.exists():
            warnings.append(f"{project.project_id}: no training_set.parquet — skipped.")
            continue

        available = _schema_columns(project.training_set_path)
        read_cols = [c for c in feature_cols if c in available]
        read_cols += [c for c in META_COLS if c in available]
        try:
            df = pd.read_parquet(project.training_set_path, columns=read_cols)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{project.project_id}: unreadable ({exc}) — skipped.")
            continue
        if df.empty:
            continue

        # Row filters, in the same order and with the same meaning as holdout.split.
        if s.exclude_imported:
            df = df.loc[~vholdout.is_imported(df)]
        if s.exclude_refine_only:
            df = df.loc[~vholdout.is_refine_only(df)]
        if s.min_confidence > 0 and "reviewer_confidence" in df.columns:
            conf = pd.to_numeric(df["reviewer_confidence"], errors="coerce").fillna(0.0)
            df = df.loc[conf >= float(s.min_confidence)]
        if df.empty:
            warnings.append(f"{project.project_id}: no rows survive the row filters.")
            continue

        wanted = set(bids)
        if s.include_no_behavior:
            wanted.add(NO_BEHAVIOR)
        lab = df["label"].astype(str).str.strip()
        df = df.loc[lab.isin(wanted)].copy()
        if df.empty:
            warnings.append(
                f"{project.project_id}: none of the checked behaviors have labeled rows.")
            continue
        df["behavior_id"] = lab.loc[df.index]

        # Per-(project, behavior) cap and floor.
        keep_idx: list[np.ndarray] = []
        for bid, sub in df.groupby("behavior_id", sort=False):
            idx = sub.index.to_numpy()
            if len(idx) < int(s.min_rows_per_behavior):
                name = (NO_BEHAVIOR if bid == NO_BEHAVIOR
                        else project.behavior_label(str(bid)))
                warnings.append(
                    f"{project.project_id} · {name}: {len(idx)} rows "
                    f"(< min {s.min_rows_per_behavior}) — dropped.")
                continue
            cap = int(s.max_rows_per_behavior)
            if cap and len(idx) > cap:
                idx = rng.choice(idx, size=cap, replace=False)
            keep_idx.append(idx)
        if not keep_idx:
            continue
        df = df.loc[np.concatenate(keep_idx)].copy()

        df["project_id"] = project.project_id
        df["behavior_name"] = [
            NO_BEHAVIOR if b == NO_BEHAVIOR else project.behavior_label(str(b))
            for b in df["behavior_id"]
        ]
        df["group"] = [
            _group_key(project.project_id, str(n), s) for n in df["behavior_name"]
        ]
        parts.append(df)

    if not parts:
        return pd.DataFrame(), warnings

    pooled = pd.concat(parts, ignore_index=True, sort=False)

    # Fold case/whitespace variants of the same behavior together before any group
    # key is read downstream. Done here, after every project has contributed, so the
    # canonical spelling is chosen from the whole pool rather than per project.
    canon = canonical_name_map(pooled["behavior_name"])
    merged = {k: v for k, v in canon.items() if k != v}
    if merged:
        pooled["behavior_name"] = [canon.get(str(n), str(n)) for n in pooled["behavior_name"]]
        pooled["group"] = [
            _group_key(str(p), str(n), s)
            for p, n in zip(pooled["project_id"], pooled["behavior_name"])
        ]
        pairs = sorted({f"'{k}' -> '{v}'" for k, v in merged.items()})
        warnings.append(
            "Merged case/spacing variants of the same behavior name: "
            + "; ".join(pairs) + ".")

    cap = int(s.max_rows_total)
    if cap and len(pooled) > cap:
        # Proportional, not uniform: a uniform draw would delete the small clusters
        # the map exists to show. Every group keeps at least its floor.
        frac = cap / len(pooled)
        keep: list[np.ndarray] = []
        for _, sub in pooled.groupby("group", sort=False):
            idx = sub.index.to_numpy()
            n = max(int(s.min_rows_per_behavior), int(round(len(idx) * frac)))
            n = min(n, len(idx))
            keep.append(rng.choice(idx, size=n, replace=False) if n < len(idx) else idx)
        pooled = pooled.loc[np.concatenate(keep)].reset_index(drop=True)
        warnings.append(
            f"Subsampled to {len(pooled):,} rows (cap {cap:,}), proportionally by group.")

    return pooled.reset_index(drop=True), warnings


# ── Matrix preparation ──────────────────────────────────────────────────────


def _fill_missing(X: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    n_missing = int(X.isna().any().sum())
    if not n_missing:
        return X, notes
    if policy == "drop_columns":
        X = X.loc[:, ~X.isna().any()]
        notes.append(f"Dropped {n_missing} column(s) with missing values.")
    elif policy == "zero":
        X = X.fillna(0.0)
        notes.append(f"Zero-filled missing values in {n_missing} column(s).")
    else:
        X = X.fillna(X.median(numeric_only=True))
        # A column that is entirely NaN has no median; it would stay NaN.
        still = [c for c in X.columns if X[c].isna().any()]
        if still:
            X[still] = X[still].fillna(0.0)
        notes.append(f"Median-filled missing values in {n_missing} column(s).")
    return X, notes


def _scale(X: np.ndarray, method: str) -> np.ndarray:
    from sklearn.preprocessing import (  # noqa: PLC0415
        MinMaxScaler, RobustScaler, StandardScaler,
    )

    if method == "none":
        return X
    scaler = {"robust": RobustScaler, "minmax": MinMaxScaler}.get(
        method, StandardScaler)()
    return np.asarray(scaler.fit_transform(X), dtype=np.float32)


def prepare_matrix(
    pooled: pd.DataFrame, feature_cols: list[str], s: EmbeddingSettings
) -> tuple[np.ndarray, list[str], list[str]]:
    """Pooled frame → the numeric matrix the reducer sees.

    Returns ``(X, kept_columns, notes)``.  Every reshaping decision that could
    change the map is recorded in ``notes`` and reaches the run's summary — the
    silent version of this function is how a figure ends up meaning something
    other than what its caption says.

    Mutates ``pooled`` in place when all-zero-R3D rows are dropped, so the caller's
    metadata frame stays row-aligned with the returned matrix.
    """
    notes: list[str] = []
    cols = [c for c in feature_cols if c in pooled.columns]
    X = pooled[cols].apply(pd.to_numeric, errors="coerce")

    if s.use_r3d and s.drop_zero_r3d_rows:
        r3d = [c for c in cols if is_r3d_column(c)]
        if r3d:
            block = X[r3d].fillna(0.0).to_numpy(dtype=np.float32)
            dead = ~np.any(block != 0.0, axis=1)
            if dead.any():
                notes.append(
                    f"Dropped {int(dead.sum()):,} row(s) whose entire R3D block was "
                    f"zero (no appearance features were ever written for them).")
                X = X.loc[~dead].reset_index(drop=True)
                pooled.drop(index=pooled.index[dead], inplace=True)
                pooled.reset_index(drop=True, inplace=True)

    # Measured BEFORE the fill, because after it every cell looks like data.
    # Under ``union`` a column a project lacks is NaN for *every one of that
    # project's rows*, so the fill writes one constant there — and the pattern of
    # which columns are constant is a near-perfect project barcode. Measured on
    # eight manuscript projects this alone put kNN project purity at 0.75-0.99
    # against a 0.13 chance floor, i.e. it manufactures exactly the project
    # clustering the map exists to test for. It has to be visible in the QC, not
    # inferable from a column count.
    frac_imputed = float(X.isna().to_numpy().mean()) if len(X) else 0.0

    X, fill_notes = _fill_missing(X, s.nan_policy)
    notes += fill_notes
    cols = list(X.columns)
    if frac_imputed > 0.02:
        notes.append(
            f"{100 * frac_imputed:.1f}% OF THE MATRIX WAS IMPUTED, not measured — "
            f"a project that lacks a column gets one constant value in it, so the "
            f"pattern of imputed columns identifies the project. Expect the map to "
            f"cluster by project for that reason alone."
            + ("  Set the feature space to 'shared' to remove it entirely."
               if s.feature_space != "shared" else
               "  You are already on 'shared', so this is sporadic per-row "
               "missingness (an untracked frame), not a whole column a project "
               "lacks — far less dangerous, but check it is not concentrated in "
               "one project."))
    if s.per_project_zscore and frac_imputed > 0.02:
        notes.append(
            "Per-project standardization on an imputed matrix makes that worse, not "
            "better: an imputed column is constant within its project, so it "
            "standardizes to exactly zero there while the projects that have it "
            "spread — sharpening the barcode. Measured: project purity 0.75 → 0.99.")

    if s.drop_zero_variance:
        var = X.var(axis=0, numeric_only=True)
        keep = [c for c in cols if float(var.get(c, 0.0) or 0.0) > 0.0]
        if len(keep) < len(cols):
            notes.append(f"Dropped {len(cols) - len(keep)} zero-variance column(s).")
        X, cols = X[keep], keep

    if not cols:
        raise ValueError(
            "No feature columns survive. Switch on more feature families, or set "
            "the feature space to 'union'.")

    arr = X.to_numpy(dtype=np.float32)

    if s.per_project_zscore:
        pid = pooled["project_id"].to_numpy()
        for p in np.unique(pid):
            m = pid == p
            block = arr[m]
            sd = block.std(axis=0)
            sd[sd == 0] = 1.0
            arr[m] = (block - block.mean(axis=0)) / sd
        notes.append(
            "Standardized within each project before pooling — rig offsets removed, "
            "and any genuine between-assay difference with them.")

    arr = _scale(arr, s.scaler)

    if s.max_features and arr.shape[1] > int(s.max_features):
        order = np.argsort(arr.var(axis=0))[::-1][: int(s.max_features)]
        order = np.sort(order)
        arr, cols = arr[:, order], [cols[i] for i in order]
        notes.append(f"Kept the {len(cols)} highest-variance columns.")

    if s.pca_components and arr.shape[1] > int(s.pca_components):
        from sklearn.decomposition import PCA  # noqa: PLC0415

        n = min(int(s.pca_components), arr.shape[0] - 1, arr.shape[1])
        pca = PCA(n_components=n, whiten=bool(s.pca_whiten),
                  random_state=int(s.random_state) if s.random_state >= 0 else None)
        arr = np.asarray(pca.fit_transform(arr), dtype=np.float32)
        notes.append(
            f"PCA to {n} components before the reducer "
            f"({100 * float(pca.explained_variance_ratio_.sum()):.1f}% of variance kept).")

    return arr, cols, notes, frac_imputed


# ── The reducer ─────────────────────────────────────────────────────────────


def reduce_matrix(
    X: np.ndarray, s: EmbeddingSettings,
    progress_cb: Callable[[str, float], None] | None = None,
) -> tuple[np.ndarray, str, list[str]]:
    """Run the chosen reducer.  Returns ``(coords, reducer_used, warnings)``."""
    warnings: list[str] = []
    want = str(s.reducer).lower()
    seed = int(s.random_state) if int(s.random_state) >= 0 else None

    if want == "umap":
        try:
            import umap  # noqa: PLC0415
        except ImportError:
            warnings.append(
                "umap-learn is not installed — fell back to PCA, which shows only "
                "linear structure. Install it with:  pip install umap-learn")
            want = "pca"
        else:
            if progress_cb:
                progress_cb(
                    f"UMAP: {X.shape[0]:,} points × {X.shape[1]} dims "
                    f"(n_neighbors={s.n_neighbors}, min_dist={s.min_dist})…", 0.55)
            reducer = umap.UMAP(
                n_neighbors=int(s.n_neighbors),
                min_dist=float(s.min_dist),
                spread=float(s.spread),
                n_components=int(s.n_components),
                metric=str(s.metric),
                set_op_mix_ratio=float(s.set_op_mix_ratio),
                local_connectivity=float(s.local_connectivity),
                repulsion_strength=float(s.repulsion_strength),
                negative_sample_rate=int(s.negative_sample_rate),
                n_epochs=int(s.n_epochs) or None,
                init=str(s.init),
                densmap=bool(s.densmap),
                dens_lambda=float(s.dens_lambda),
                random_state=seed,
                verbose=False,
            )
            return np.asarray(reducer.fit_transform(X), dtype=np.float32), "umap", warnings

    if want == "tsne":
        from sklearn.manifold import TSNE  # noqa: PLC0415

        if progress_cb:
            progress_cb(f"t-SNE: {X.shape[0]:,} points…", 0.55)
        # Perplexity must stay below n/3 or sklearn refuses outright.
        perp = float(min(float(s.tsne_perplexity), max(5.0, (X.shape[0] - 1) / 3.0)))
        if perp != float(s.tsne_perplexity):
            warnings.append(f"Perplexity lowered to {perp:.0f} for {X.shape[0]:,} points.")
        tsne = TSNE(n_components=min(int(s.n_components), 3), perplexity=perp,
                    init="pca", random_state=seed if seed is not None else 0)
        return np.asarray(tsne.fit_transform(X), dtype=np.float32), "tsne", warnings

    from sklearn.decomposition import PCA  # noqa: PLC0415

    if progress_cb:
        progress_cb("PCA projection…", 0.55)
    pca = PCA(n_components=int(s.n_components), random_state=seed)
    return np.asarray(pca.fit_transform(X), dtype=np.float32), "pca", warnings


# ── QC: is this a behavior map or a project map? ────────────────────────────


def structure_metrics(
    X: np.ndarray, coords: np.ndarray, labels: pd.DataFrame, *, k: int = 15,
    seed: int = 42,
) -> dict[str, float]:
    """How much of the layout is behavior, and how much is project identity?

    Four numbers, reported for both the feature space and the embedding:

    * ``silhouette_behavior`` / ``silhouette_project`` — separation of each
      labelling, in [-1, 1].  A project silhouette approaching the behavior one
      means the clusters are assays, not acts.
    * ``knn_purity_behavior`` / ``knn_purity_project`` — of a point's *k* nearest
      neighbours, the fraction sharing its label.  Project purity has a floor at
      ``knn_purity_project_chance`` (the probability two random points share a
      project); well-mixed projects sit near that floor, an unmixed map near 1.0.

    Both are computed in the feature space *and* in the embedding, because a
    silhouette read off a UMAP layout partly measures UMAP.  The feature-space
    number is the one to quote.
    """
    from sklearn.metrics import silhouette_score  # noqa: PLC0415
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

    out: dict[str, float] = {}
    n = len(labels)
    if n < 10:
        return out

    rng = np.random.default_rng(int(seed))
    # Silhouette is O(n^2); subsample above 8k rather than refusing to report.
    idx = rng.choice(n, size=8000, replace=False) if n > 8000 else np.arange(n)

    beh = labels["group"].to_numpy()
    proj = labels["project_id"].to_numpy()

    for space, M in (("feat", X), ("embed", coords)):
        for name, lab in (("behavior", beh), ("project", proj)):
            try:
                if len(np.unique(lab[idx])) > 1:
                    out[f"silhouette_{name}_{space}"] = float(
                        silhouette_score(M[idx], lab[idx]))
            except Exception:  # noqa: BLE001 — a QC number must never sink a run
                pass

        try:
            kk = int(min(k, len(idx) - 1))
            if kk >= 2:
                nn = NearestNeighbors(n_neighbors=kk + 1).fit(M[idx])
                _, ind = nn.kneighbors(M[idx])
                for name, lab in (("behavior", beh), ("project", proj)):
                    same = lab[idx][ind[:, 1:]] == lab[idx][:, None]
                    out[f"knn_purity_{name}_{space}"] = float(same.mean())
        except Exception:  # noqa: BLE001
            pass

    # The floor kNN project purity would hit if projects were perfectly mixed.
    shares = pd.Series(proj).value_counts(normalize=True).to_numpy()
    out["knn_purity_project_chance"] = float((shares ** 2).sum())
    return out


def group_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows per group, with the project and behavior that produced them."""
    g = (frame.groupby(["group", "project_id", "behavior_name"], sort=False)
         .size().reset_index(name="n_points"))
    return g.sort_values(["project_id", "n_points"], ascending=[True, False])


# ── Top-level entry point ───────────────────────────────────────────────────


def build_all_project_umap(
    projects: list[ProjectRef],
    behaviors: dict[str, list[str]],
    settings: EmbeddingSettings,
    *,
    progress_cb: Callable[[str, float], None] | None = None,
) -> EmbeddingResult:
    """Assemble, scale, reduce and QC — everything up to (not including) drawing."""
    if progress_cb:
        progress_cb("Resolving the shared feature space…", 0.02)
    feature_cols, per_project = resolve_feature_columns(projects, settings)
    if not feature_cols:
        raise ValueError(
            "No feature columns in common. Switch on more feature families, or set "
            "the feature space to 'union'.")

    pooled, warnings = collect_rows(
        projects, behaviors, feature_cols, settings, progress_cb=progress_cb)
    if pooled.empty:
        raise ValueError(
            "No rows to embed. Check that behaviors are ticked on the Projects tab "
            "and that the row filters are not excluding everything.")

    if progress_cb:
        progress_cb(f"Preparing {len(pooled):,} × {len(feature_cols)} matrix…", 0.35)
    X, kept_cols, notes, frac_imputed = prepare_matrix(pooled, feature_cols, settings)
    warnings += notes

    coords, reducer_used, red_warn = reduce_matrix(X, settings, progress_cb=progress_cb)
    warnings += red_warn

    if progress_cb:
        progress_cb("Measuring behavior vs. project structure…", 0.90)
    meta_cols = [c for c in ("segment_id", "session_id", "animal_id", "label_source",
                             "start_frame", "end_frame") if c in pooled.columns]
    frame = pooled[["group", "project_id", "behavior_id", "behavior_name"] + meta_cols].copy()
    for i, axis in enumerate(("x", "y", "z")[: coords.shape[1]]):
        frame[axis] = coords[:, i]

    qc = structure_metrics(X, coords, frame, seed=int(settings.sample_seed))
    qc["frac_imputed"] = float(frac_imputed)
    qc["n_points"] = float(len(frame))
    qc["n_features"] = float(len(kept_cols))
    qc["n_groups"] = float(frame["group"].nunique())
    qc["n_projects"] = float(frame["project_id"].nunique())
    for pid, n in per_project.items():
        qc[f"n_candidate_features::{pid}"] = float(n)

    return EmbeddingResult(
        frame=frame, feature_cols=kept_cols, settings=settings, qc=qc,
        counts=group_counts(frame), warnings=warnings, reducer_used=reducer_used,
    )


# ── Persistence — the embed-once / restyle-many contract ────────────────────


def save_embedding(result: EmbeddingResult, out_dir: Path) -> dict[str, Path]:
    """Write the coordinates, their settings and their QC to ``out_dir``.

    ``embedding.parquet`` is the only file the renderer needs, so a saved run can
    be restyled months later — or in another session — without the projects being
    mounted.  The settings and feature list travel with it because a map whose
    provenance is lost is not evidence of anything.
    """
    import json  # noqa: PLC0415

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    emb = out_dir / "embedding.parquet"
    result.frame.to_parquet(emb, index=False)
    paths["embedding"] = emb

    coords = out_dir / "embedding.csv"
    result.frame.to_csv(coords, index=False)
    paths["Embedding coordinates"] = coords

    if result.counts is not None and len(result.counts):
        cpath = out_dir / "group_counts.csv"
        result.counts.to_csv(cpath, index=False)
        paths["Points per group"] = cpath

    qpath = out_dir / "qc.json"
    qpath.write_text(json.dumps({
        "reducer_used": result.reducer_used,
        "n_points": result.n_points,
        "feature_cols": result.feature_cols,
        "qc": result.qc,
        "warnings": result.warnings,
        "settings": result.settings.to_dict(),
    }, indent=2), encoding="utf-8")
    paths["QC / settings (json)"] = qpath

    qcsv = out_dir / "qc_metrics.csv"
    pd.DataFrame(sorted(result.qc.items()), columns=["metric", "value"]).to_csv(
        qcsv, index=False)
    paths["Structure metrics"] = qcsv
    return paths


def load_embedding(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read back a saved embedding.  Accepts the run folder or the parquet itself.

    Returns ``(frame, meta)`` where ``meta`` is the contents of ``qc.json`` (empty
    if it is missing — an embedding is still renderable without its provenance,
    just less trustworthy).
    """
    import json  # noqa: PLC0415

    path = Path(path)
    parquet = path if path.suffix == ".parquet" else path / "embedding.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"No embedding.parquet at {parquet}")
    frame = pd.read_parquet(parquet)
    meta_path = parquet.parent / "qc.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}
    return frame, meta


def qc_summary(result: EmbeddingResult) -> str:
    """The paragraph that tells the reader how to read the map."""
    q = result.qc
    lines = [
        f"{result.n_points:,} points · {len(result.feature_cols):,} features · "
        f"{int(q.get('n_groups', 0))} groups from {int(q.get('n_projects', 0))} "
        f"project(s) · reducer: {result.reducer_used}",
    ]
    imp = q.get("frac_imputed") or 0.0
    if imp > 0.02:
        lines.append(
            f"WARNING: {100 * imp:.1f}% of the feature matrix was imputed, which "
            f"identifies the project on its own — switch the feature space to "
            f"'shared' before reading anything into the clustering below.")
    sb = q.get("silhouette_behavior_feat")
    sp = q.get("silhouette_project_feat")
    if sb is not None and sp is not None:
        verdict = ("behavior structure dominates" if sb > sp else
                   "PROJECT identity dominates — the clusters are largely assays, "
                   "not acts")
        lines.append(
            f"Feature-space silhouette: behavior {sb:+.3f} vs project {sp:+.3f} "
            f"→ {verdict}.")
    kb = q.get("knn_purity_behavior_feat")
    kp = q.get("knn_purity_project_feat")
    kc = q.get("knn_purity_project_chance")
    if kb is not None and kp is not None:
        chance = f" (chance {kc:.2f})" if kc is not None else ""
        lines.append(
            f"kNN purity (k=15, feature space): behavior {kb:.2f}, "
            f"project {kp:.2f}{chance}.")
    if result.warnings:
        lines.append("Notes:")
        lines += [f"  • {w}" for w in result.warnings]
    return "\n".join(lines)
