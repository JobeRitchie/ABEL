"""Data model for the validation platform.

A *ValidationRun* spans ``N projects × M behaviors × K analyses``.  The atomic
record is a :class:`CellResult` — one ``(project, behavior, analysis, config,
n_clips, seed)`` evaluation.  All cells flatten into a single tidy
``cells.parquet`` table (see :mod:`abel.validation.aggregate`) that every
dashboard and plot groups over.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ── Project reference ──────────────────────────────────────────────────────


@dataclass
class ProjectRef:
    """Lightweight handle to an ABEL project on disk + its training config."""

    project_id: str
    name: str
    root: Path
    # The project's own name (project.yaml / folder) at load time.  Never changes, so
    # a rename can always be shown as "was: X" and reset.  NOT a disk locator — `root`
    # is the only path — which is what makes renaming project_id safe (see rename()).
    source_name: str = ""
    classifier_type: str = "xgboost"
    calibration_method: str = "sigmoid"
    split_strategy: str = "group_shuffle_session"
    use_video_features: bool = True
    allow_co_occurring_behaviors: bool = False
    # Frame rate, only so a clip count can be reported in seconds. The clip
    # *length* is deliberately not read from config: ``segment_window_frames``
    # defaults to 60 but real projects set it from their own clip duration
    # (most use ~0.5 s), so the honest number is measured from the labeled rows
    # themselves — see :func:`abel.validation.holdout.median_clip_frames`.
    fps: float = 30.0
    behavior_names: dict[str, str] = field(default_factory=dict)  # behavior_id -> name on disk
    # behavior_id -> user rename set on the Projects tab.  Display-only: it never
    # touches the project on disk, and disk lookups must not follow it (see
    # behavior_disk_name).  Harmonizing names across projects is what lets the
    # generalization figure pool a behavior that two projects spell differently.
    behavior_aliases: dict[str, str] = field(default_factory=dict)

    @property
    def training_set_path(self) -> Path:
        return self.root / "derived" / "training_sets" / "training_set.parquet"

    @classmethod
    def load(cls, root: str | Path) -> "ProjectRef":
        """Read project.yaml + behavior_definitions.yaml into a ProjectRef."""
        root = Path(root)
        cfg: dict[str, Any] = {}
        proj_yaml = root / "project.yaml"
        if proj_yaml.exists():
            cfg = yaml.safe_load(proj_yaml.read_text(encoding="utf-8")) or {}
        bm = cfg.get("behavior_model", {}) or {}

        names: dict[str, str] = {}
        bd_path = root / "config" / "behavior_definitions.yaml"
        if bd_path.exists():
            bd = yaml.safe_load(bd_path.read_text(encoding="utf-8")) or {}
            for b in bd.get("behaviors", []) or []:
                bid = str(b.get("behavior_id", "")).strip()
                if bid:
                    names[bid] = str(b.get("name", bid))

        name = str(cfg.get("project_name") or root.name)
        return cls(
            project_id=name,
            name=name,
            source_name=name,
            root=root,
            classifier_type=str(bm.get("classifier_type", "xgboost")),
            calibration_method=str(bm.get("calibration_method", "sigmoid")),
            split_strategy=str(bm.get("evaluation_split_strategy", "group_shuffle_session")),
            use_video_features=bool(bm.get("use_video_features", True)),
            allow_co_occurring_behaviors=bool(bm.get("allow_co_occurring_behaviors", False)),
            fps=float(cfg.get("default_fps") or 30.0),
            behavior_names=names,
        )

    @property
    def original_name(self) -> str:
        """The project's own name, whatever it is currently displayed as."""
        return self.source_name or self.name

    @property
    def is_renamed(self) -> bool:
        return bool(self.source_name) and self.project_id != self.source_name

    def rename(self, new_name: str) -> None:
        """Rename the project for the whole suite; blank restores its own name.

        This rewrites ``project_id`` as well as ``name``, deliberately.  ``project_id``
        is a human-readable label, never a disk locator (``root`` is the only path it
        has), and it is what every figure titles itself with and what output filenames
        are stemmed from.  Rewriting it is therefore what makes the new name appear in
        the figures, tables and exports without threading a display name through ~20
        plot call sites — and the project on disk is untouched either way.

        Callers holding the project in a dict keyed by ``project_id`` must re-key.
        """
        name = str(new_name).strip() or self.original_name
        self.project_id = name
        self.name = name

    def behavior_label(self, behavior_id: str) -> str:
        """The name every analysis, plot and export shows for this behavior.

        The user's rename wins over the project's own name.  This is the single
        choke point the whole suite labels through, so a rename set on the
        Projects tab travels everywhere without any analysis knowing about it.
        """
        bid = str(behavior_id)
        alias = str(self.behavior_aliases.get(bid, "")).strip()
        return alias or self.behavior_names.get(bid, bid)

    def behavior_disk_name(self, behavior_id: str) -> str:
        """The name as written in ``behavior_definitions.yaml`` — never the rename.

        Trained-model directories and per-behavior refinement settings are keyed
        by this on disk, so resolving them must NOT follow a rename or renaming a
        behavior would silently orphan its model.
        """
        bid = str(behavior_id)
        return self.behavior_names.get(bid, bid)

    def set_behavior_alias(self, behavior_id: str, new_name: str) -> None:
        """Rename a behavior for display; blank or unchanged clears the rename."""
        bid = str(behavior_id)
        name = str(new_name).strip()
        if not name or name == self.behavior_disk_name(bid):
            self.behavior_aliases.pop(bid, None)
        else:
            self.behavior_aliases[bid] = name

    def behavior_ids_matching(self, names: "Iterable[str]") -> list[str]:
        """Behavior ids matching any of *names* by display name OR on-disk name.

        Both are accepted so a name typed into an analysis (e.g. the video-value
        tab's "Groom, Freeze") still resolves whether the user typed the rename
        they set on the Projects tab or the project's original name.
        """
        want = {str(n).strip().lower() for n in names if str(n).strip()}
        return [
            bid for bid in self.behavior_names
            if str(bid) != "no_behavior"
            and (self.behavior_label(bid).strip().lower() in want
                 or self.behavior_disk_name(bid).strip().lower() in want)
        ]

    def is_valid(self) -> bool:
        return self.training_set_path.exists()


def _safe_model_name(value: str) -> str:
    """Mirror the trainer's model-dir sanitization (see validation_service)."""
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_"
                   for ch in str(value).strip())
    return safe or "target_behavior"


def read_behavior_model_metrics(project: "ProjectRef") -> dict[str, dict[str, float]]:
    """Read each behavior's trained-model metrics from disk (no training).

    Returns ``{behavior_id: {"f1": .., "pr_auc": .., "precision": .., "recall": ..}}``.
    A behavior with no trained model maps to an empty dict.  Model directories are
    resolved the same way the product does: the workflow snapshot's recorded
    version first, then the conventional ``behavior_model_<SafeName>`` folder.
    """
    import json  # noqa: PLC0415

    root = project.root
    models_root = root / "derived" / "models"

    snapshot: dict[str, str] = {}
    snap_path = root / "derived" / "workflow_snapshot.json"
    if snap_path.exists():
        try:
            raw = json.loads(snap_path.read_text(encoding="utf-8"))
            sbm = raw.get("selected_behavior_models") or {}
            snapshot = {str(k): str(v) for k, v in sbm.items() if str(v).strip()}
        except Exception:
            snapshot = {}

    def _resolve(bid: str, name: str) -> "Path | None":
        candidates: list[str] = []
        if snapshot.get(bid):
            candidates.append(snapshot[bid])
        for token in (name, bid):
            safe = _safe_model_name(token)
            candidates.append(f"behavior_model_{safe}")
            candidates.append(safe)
        seen: set[str] = set()
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            mdir = models_root / cand
            if (mdir / "metrics.json").exists():
                return mdir
        return None

    out: dict[str, dict[str, float]] = {}
    for bid in project.behavior_names:
        if str(bid) == "no_behavior":
            continue
        metrics: dict[str, float] = {}
        # Disk name, not behavior_label(): a user rename is display-only and the
        # model folder on disk still carries the original name.
        disk_name = project.behavior_disk_name(bid)
        mdir = _resolve(str(bid), disk_name)
        if mdir is not None:
            try:
                raw = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
                for key in ("f1", "pr_auc", "precision", "recall"):
                    try:
                        val = float(raw.get(key))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(val):
                        metrics[key] = val
            except Exception:
                pass
            # Post-temporal-refinement held-out metrics (present only for models
            # trained after held-out probabilities were persisted). Reuses the
            # exact same refinement engine as the Validation tab so the external
            # suite reports the same refined numbers.
            try:
                from abel.temporal_refinement.refined_eval import (  # noqa: PLC0415
                    refined_holdout_metrics,
                )

                refined = refined_holdout_metrics(
                    mdir, root, disk_name, target_behavior_id=str(bid)
                )
                if refined is not None:
                    for src, dst in (
                        ("refined_f1", "refined_f1"),
                        ("refined_precision", "refined_precision"),
                        ("refined_recall", "refined_recall"),
                    ):
                        v = refined.get(src)
                        if v is not None and np.isfinite(float(v)):
                            metrics[dst] = float(v)
            except Exception:
                pass
        out[str(bid)] = metrics
    return out


# ── Per-config evaluation result (engine output) ───────────────────────────


@dataclass
class ConfigEvalResult:
    """Result of a single train+evaluate run produced by :mod:`engine`.

    Holds the headline metrics plus the binary (target-vs-rest) arrays needed
    for PR curves / confusion matrices.  ``fitted_estimator`` and ``val_meta``
    are retained only transiently (for the later video-overlay phase) and are
    NOT serialized into cells.
    """

    project_id: str
    behavior_id: str
    n_pos_train: int
    n_neg_train: int
    n_features: int

    # Headline metrics (ABEL's own macro precision/recall/f1; target PR-AUC).
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    pr_auc: float = float("nan")
    cohen_kappa: float = float("nan")

    # Imbalance-robust classifier summaries (target-vs-rest held-out).
    mcc: float = float("nan")
    balanced_accuracy: float = float("nan")
    specificity: float = float("nan")
    roc_auc: float = float("nan")

    # Target-vs-rest confusion counts on the held-out set.
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    elapsed_sec_fit: float = 0.0
    elapsed_sec_total: float = 0.0
    degenerate: bool = False
    error: str = ""

    # Binary target-vs-rest arrays (for plots); not persisted in the tidy table.
    y_true: np.ndarray | None = None
    y_score: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    confusion_matrix: list[list[int]] | None = None

    # Transient handles for the later feature-overlay phase (never serialized).
    fitted_estimator: Any | None = None
    val_meta: Any | None = None

    # Transient per-feature importance (XGBoost gain), retained only when the
    # caller asks for it (behaviorscape analysis).  Never serialized into cells.
    feature_importance: dict[str, float] | None = None


# ── Tidy cell record (the meta-analysis substrate) ─────────────────────────


@dataclass
class CellResult:
    """One atom of a ValidationRun — serialized into ``cells.parquet``."""

    project_id: str
    project_name: str
    behavior_id: str
    behavior_name: str
    analysis: str            # "learning_curve" | "ablation" | "generalization"
    config_name: str         # e.g. "baseline_none", "add_calibration", "all_features", "n=50"
    n_clips: int             # positive training clips (−1 = "all"/not-applicable)
    seed: int

    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    pr_auc: float = float("nan")
    cohen_kappa: float = float("nan")

    # Imbalance-robust classifier summaries (persisted in cells.parquet).
    mcc: float = float("nan")
    balanced_accuracy: float = float("nan")
    specificity: float = float("nan")
    roc_auc: float = float("nan")

    # Target-vs-rest confusion counts on the held-out set (persisted in cells.parquet).
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    n_pos_train: int = 0
    n_neg_train: int = 0
    n_features: int = 0
    elapsed_sec_fit: float = 0.0
    elapsed_sec_total: float = 0.0
    degenerate: bool = False
    error: str = ""

    # hash key into arrays/<cell_hash>.parquet (PR curve / confusion sidecar)
    arrays_ref: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


# ── Run manifest ───────────────────────────────────────────────────────────


@dataclass
class RunManifest:
    """Top-level description of a validation run (one results directory)."""

    run_id: str
    created_at: str
    analyses: list[str]
    projects: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
