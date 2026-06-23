"""Evaluation and bout-level assay metric reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abel.services.provenance_service import ProvenanceService
from abel.storage.file_store import read_json, write_json


logger = logging.getLogger("abel")


@dataclass
class BoutMergeConfig:
    max_gap_frames: int = 10
    min_bout_duration: int = 15


class EvaluationService:
    """Compute segment/frame/bout metrics and persist reports."""

    def __init__(self) -> None:
        self._provenance = ProvenanceService()

    @staticmethod
    def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
        exclude = {
            "label_true",
            "label_pred",
            "prediction_prob",
            "review_label",
            "segment_id",
            "session_id",
            "animal_id",
            "subject_id",
            "label",
            "label_source",
            "start_frame",
            "end_frame",
        }
        cols: list[str] = []
        for col in df.columns:
            if col in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
        return cols

    def _write_behavior_separation_plot(self, out_dir: Path, segment_labels: pd.DataFrame) -> None:
        if segment_labels.empty or "label_true" not in segment_labels.columns:
            return

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        work = segment_labels.copy()
        if "prediction_prob" in work.columns:
            work["prediction_prob"] = pd.to_numeric(work["prediction_prob"], errors="coerce").fillna(0.0)
        else:
            work["prediction_prob"] = 0.0

        feature_cols = self._numeric_feature_columns(work)
        if not feature_cols:
            # Ensure at least 2 dimensions for plotting.
            work["_sep_aux"] = pd.to_numeric(work.get("label_pred", 0), errors="coerce").fillna(0.0)
            feature_cols = ["prediction_prob", "_sep_aux"]
        elif len(feature_cols) == 1:
            feature_cols = [feature_cols[0], "prediction_prob"]

        x_all = work[feature_cols].to_numpy(dtype=float)
        if x_all.size == 0:
            return
        y_true_all = work["label_true"].to_numpy(dtype=int)
        if "umap_label" in work.columns:
            plot_labels_all = work["umap_label"].astype(str).to_numpy()
        else:
            plot_labels_all = np.where(y_true_all == 1, "Target", "Other")

        # Keep figure generation responsive on very large runs.
        cap = 6000
        if len(x_all) > cap:
            idx = np.random.default_rng(42).choice(len(x_all), size=cap, replace=False)
            x = x_all[idx]
            plot_labels = plot_labels_all[idx]
        else:
            x = x_all
            plot_labels = plot_labels_all

        coords = None
        method = "PCA"
        try:
            import umap  # type: ignore[import]

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=max(5, min(25, int(np.sqrt(max(10, len(x)))))),
                min_dist=0.15,
                metric="euclidean",
                random_state=42,
            )
            coords = reducer.fit_transform(x)
            method = "UMAP"
        except Exception as exc:
            logger.info("Behavior separation plot using PCA fallback (UMAP unavailable or failed): %s", exc)
            try:
                from sklearn.decomposition import PCA

                coords = PCA(n_components=2, random_state=42).fit_transform(x)
            except Exception:
                return

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        classes = pd.unique(pd.Series(plot_labels).astype(str))
        cmap = plt.get_cmap("tab20", max(1, len(classes)))
        for i, label in enumerate(classes):
            mask = np.asarray(plot_labels, dtype=str) == str(label)
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=8,
                alpha=0.65,
                color=cmap(i),
                label=str(label),
            )
        ax.set_title(f"Behavior Separation ({method})")
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        ax.legend(loc="best", frameon=False, fontsize=8)
        fig.tight_layout()
        # Keep a pipeline-specific filename while preserving the legacy path.
        fig.savefig(out_dir / "behavior_separation_active_learning.png", dpi=200, bbox_inches="tight")
        fig.savefig(out_dir / "behavior_separation_active_learning.svg", bbox_inches="tight")
        fig.savefig(out_dir / "behavior_separation_umap.png", dpi=200, bbox_inches="tight")
        fig.savefig(out_dir / "behavior_separation_umap.svg", bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def merge_bouts(segments: pd.DataFrame, config: BoutMergeConfig | None = None) -> pd.DataFrame:
        cfg = config or BoutMergeConfig()
        if segments.empty:
            return pd.DataFrame(columns=["animal_id", "session_id", "start_frame", "end_frame", "duration_frames"])

        rows = segments.sort_values(["animal_id", "session_id", "start_frame"]).to_dict(orient="records")
        merged: list[dict[str, int | str | float]] = []

        cur = rows[0].copy()
        for row in rows[1:]:
            same = row["animal_id"] == cur["animal_id"] and row["session_id"] == cur["session_id"]
            gap = int(row["start_frame"]) - int(cur["end_frame"])
            if same and gap <= cfg.max_gap_frames:
                cur["end_frame"] = max(int(cur["end_frame"]), int(row["end_frame"]))
            else:
                dur = int(cur["end_frame"]) - int(cur["start_frame"]) + 1
                if dur >= cfg.min_bout_duration:
                    merged.append(
                        {
                            "animal_id": cur["animal_id"],
                            "session_id": cur["session_id"],
                            "start_frame": int(cur["start_frame"]),
                            "end_frame": int(cur["end_frame"]),
                            "duration_frames": dur,
                        }
                    )
                cur = row.copy()

        dur = int(cur["end_frame"]) - int(cur["start_frame"]) + 1
        if dur >= cfg.min_bout_duration:
            merged.append(
                {
                    "animal_id": cur["animal_id"],
                    "session_id": cur["session_id"],
                    "start_frame": int(cur["start_frame"]),
                    "end_frame": int(cur["end_frame"]),
                    "duration_frames": dur,
                }
            )

        return pd.DataFrame(merged)

    @staticmethod
    def segment_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None = None,
    ) -> dict[str, float]:
        from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

        if y_score is None:
            y_score = y_pred.astype(float)
        return {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
        }

    @staticmethod
    def frame_metrics(frame_labels: pd.DataFrame) -> dict[str, float]:
        score = frame_labels["prediction_prob"].to_numpy(dtype=float) if "prediction_prob" in frame_labels.columns else None
        return EvaluationService.segment_metrics(
            frame_labels["label_true"].to_numpy(dtype=int),
            frame_labels["label_pred"].to_numpy(dtype=int),
            y_score=score,
        )

    @staticmethod
    def bout_metrics(bouts: pd.DataFrame, fps: float) -> dict[str, float]:
        if bouts.empty:
            return {
                "latency_to_first_behavior": float("nan"),
                "total_behavior_time": 0.0,
                "bout_count": 0.0,
                "mean_bout_duration": 0.0,
                "behavior_rate": 0.0,
                "distance_to_context_during_behavior": float("nan"),
            }
        durations = bouts["duration_frames"].to_numpy(dtype=float) / fps
        dist = float("nan")
        dist_col = None
        for candidate in (
            "nose_to_TMT_dist_mean",
            "forepaw_centroid_to_TMT_dist_mean",
            "body_centroid_to_TMT_dist_mean",
            "nose_to_target_dist_mean",
            "body_centroid_to_target_dist_mean",
            "distance_to_TMT_mean",
        ):
            if candidate in bouts.columns:
                dist_col = candidate
                break
        if dist_col is not None:
            vals = bouts[dist_col].to_numpy(dtype=float)
            durs = bouts["duration_frames"].to_numpy(dtype=float)
            if len(vals) > 0 and np.sum(durs) > 0:
                dist = float(np.average(vals, weights=durs))
        return {
            "latency_to_first_behavior": float(bouts["start_frame"].min() / fps),
            "total_behavior_time": float(np.sum(durations)),
            "bout_count": float(len(bouts)),
            "mean_bout_duration": float(np.mean(durations)),
            "behavior_rate": float(len(bouts) / max(1e-9, (bouts["end_frame"].max() / fps))),
            "distance_to_context_during_behavior": dist,
        }

    def evaluate_and_save(
        self,
        project_root: Path,
        frame_labels: pd.DataFrame,
        segment_labels: pd.DataFrame,
        positive_segments: pd.DataFrame,
        fps: float,
        merge_config: BoutMergeConfig | None = None,
        behavior_id: str = "target_behavior",
        model_version: str = "behavior_model_v1",
        feature_version: str = "representation_v1",
        config: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, float]]:
        out_dir = project_root / "derived" / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_report = self.frame_metrics(frame_labels)
        del frame_labels  # free large frame-level DataFrame now that metrics are extracted
        seg_score = segment_labels["prediction_prob"].to_numpy(dtype=float) if "prediction_prob" in segment_labels.columns else None
        seg_report = self.segment_metrics(
            segment_labels["label_true"].to_numpy(dtype=int),
            segment_labels["label_pred"].to_numpy(dtype=int),
            y_score=seg_score,
        )

        bouts = self.merge_bouts(positive_segments, merge_config)
        del positive_segments  # no longer needed
        bout_report = self.bout_metrics(bouts, fps=fps)

        metrics = {
            "frame_level": frame_report,
            "segment_level": seg_report,
            "bout_level": bout_report,
        }
        write_json(out_dir / "model_metrics.json", metrics)
        safe_behavior = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in behavior_id) or "target_behavior"
        bouts.to_parquet(project_root / "derived" / "behavior_bouts" / f"{safe_behavior}_bouts.parquet", index=False)

        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix

            y_true = segment_labels["label_true"].to_numpy(dtype=int)
            y_pred = segment_labels["label_pred"].to_numpy(dtype=int)

            y_prob = segment_labels["prediction_prob"].to_numpy(dtype=float) if "prediction_prob" in segment_labels.columns else y_pred.astype(float)
            disp = PrecisionRecallDisplay.from_predictions(y_true, y_prob)
            disp.figure_.savefig(out_dir / "PR_curve.png", dpi=200, bbox_inches="tight")
            plt.close(disp.figure_)

            cm = confusion_matrix(y_true, y_pred)
            # Persist confusion matrix in metrics for downstream quality summaries.
            metrics["confusion_matrix"] = cm.tolist()

            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            tick_labels = ["Other", "Target"]
            if cm.shape[0] == 2 and cm.shape[1] == 2:
                ax.set_xticks([0, 1], tick_labels)
                ax.set_yticks([0, 1], tick_labels)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            fig.colorbar(im, ax=ax)
            fig.text(
                0.5,
                0.01,
                "Rows=true labels, columns=predicted labels. Diagonal cells are correct predictions.",
                ha="center",
                fontsize=8,
            )
            fig.savefig(out_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            logger.exception("Failed generating confusion matrix / PR curve")

        # NOTE: Per-behaviour separation plots removed in favour of the unified
        # UMAP which aggregates all behaviour models in a single embedding.

        # Lightweight report artifacts for downstream plotting.
        pd.DataFrame(frame_report, index=[0]).to_csv(out_dir / "frame_metrics.csv", index=False)
        pd.DataFrame(seg_report, index=[0]).to_csv(out_dir / "segment_metrics.csv", index=False)
        pd.DataFrame(bout_report, index=[0]).to_csv(out_dir / "bout_metrics.csv", index=False)

        prov = self._provenance.make_provenance(
            project_root=project_root,
            model_version=model_version,
            feature_version=feature_version,
            config=config or {},
        )
        write_json(
            out_dir / "evaluation.manifest.json",
            {
                "provenance": prov.model_dump(mode="json"),
                "artifacts": [
                    "model_metrics.json",
                    "unified_behavior_umap.png",
                    "PR_curve.png",
                    "confusion_matrix.png",
                    "cross_behavior_confound_matrix.png",
                    "frame_metrics.csv",
                    "segment_metrics.csv",
                    "bout_metrics.csv",
                ],
            },
        )

        return metrics

    # ------------------------------------------------------------------
    # Cross-behavior confound analysis
    # ------------------------------------------------------------------

    def generate_cross_behavior_confound_report(
        self,
        project_root: Path,
        behavior_names: dict[str, str] | None = None,
        threshold: float = 0.3,
        target_behavior_id: str | None = None,
    ) -> dict[str, Any]:
        """Build an NxN co-activation matrix across all trained behaviour models.

        For every segment, each model's prediction probability is collected.
        A pair of behaviours (A, B) is co-activated when both predict above
        *threshold*.  The result is a symmetric matrix of co-activation rates
        plus ranked suggestions for the user.

        Additionally computes margin-based confound detection: segments where
        the top two behaviour predictions are within 0.2 of each other are
        flagged as potential confounds even when one is below *threshold*.

        Parameters
        ----------
        project_root:
            Project root directory.
        behavior_names:
            Optional mapping ``{behavior_id: display_name}``.
        threshold:
            Probability cutoff for considering a behaviour "active".

        Returns
        -------
        dict with keys: behavior_ids, behavior_labels, coactivation_matrix,
        suggestions, out_path.
        """
        models_root = project_root / "derived" / "models"
        if not models_root.exists():
            return {"error": "No models directory found."}

        # Discover models and their target behaviours.
        model_dirs: list[Path] = []
        for p in sorted(models_root.iterdir()):
            if p.is_dir() and (p / "model_state.pkl").exists() and p.name.startswith("behavior_model_"):
                model_dirs.append(p)

        # Keep latest model per behaviour.
        # Use directory name as the canonical key rather than run_settings
        # target_behavior, because saved target_behavior can be wrong
        # (e.g. No_Behavior model may have been saved with another
        # behaviour's ID).  Directory names are always unique and user-controlled.
        latest_by_behavior: dict[str, Path] = {}
        for md in model_dirs:
            # Derive behaviour key from directory name: "behavior_model_Freeze" -> "Freeze"
            dir_key = md.name.removeprefix("behavior_model_").strip()
            if not dir_key:
                settings = read_json(md / "run_settings.json", {})
                dir_key = str(settings.get("target_behavior") or settings.get("target_behavior_id") or "").strip()
            if not dir_key:
                continue
            latest_by_behavior[dir_key] = md  # dirs are sorted; last wins (most recent)

        if len(latest_by_behavior) < 2:
            return {"error": "Need at least two behaviour models for confound analysis."}

        # Load predictions.
        pred_frames: dict[str, pd.DataFrame] = {}
        for bid, md in latest_by_behavior.items():
            pred_path = md / "segment_predictions.parquet"
            if not pred_path.exists():
                continue
            df = pd.read_parquet(pred_path)
            if "segment_id" in df.columns and "prediction_prob" in df.columns:
                pred_frames[bid] = df[["segment_id", "prediction_prob"]].copy()

        if len(pred_frames) < 2:
            return {"error": "Need prediction files from at least two behaviours."}

        # Merge predictions on segment_id.
        merged: pd.DataFrame | None = None
        bid_list: list[str] = []
        for bid, df in pred_frames.items():
            col = f"prob_{bid}"
            df = df.rename(columns={"prediction_prob": col})
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on="segment_id", how="outer")
            bid_list.append(bid)

        assert merged is not None
        prob_cols = [f"prob_{b}" for b in bid_list]
        merged[prob_cols] = merged[prob_cols].fillna(0.0)

        # Build co-activation matrix (symmetric).
        n = len(bid_list)
        coact = np.zeros((n, n), dtype=float)
        total = len(merged)
        for i in range(n):
            active_i = merged[prob_cols[i]].to_numpy() >= threshold
            for j in range(i, n):
                active_j = merged[prob_cols[j]].to_numpy() >= threshold
                count = int(np.sum(active_i & active_j))
                rate = count / max(1, total)
                coact[i, j] = rate
                coact[j, i] = rate

        # Labels for display.
        bname = behavior_names or {}
        labels = [bname.get(b, b) for b in bid_list]

        # Rank off-diagonal pairs by co-activation rate.
        pairs: list[dict[str, Any]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append({
                    "behavior_a": labels[i],
                    "behavior_b": labels[j],
                    "coactivation_rate": round(float(coact[i, j]), 4),
                })
        pairs.sort(key=lambda p: p["coactivation_rate"], reverse=True)

        suggestions: list[str] = []
        for p in pairs:
            rate = p["coactivation_rate"]
            a, b = p["behavior_a"], p["behavior_b"]
            if rate > 0.10:
                suggestions.append(
                    f"High co-activation ({rate:.1%}) between '{a}' and '{b}': "
                    f"review labels near transitions to ensure clear boundaries."
                )
            elif rate > 0.03:
                suggestions.append(
                    f"Moderate co-activation ({rate:.1%}) between '{a}' and '{b}': "
                    f"consider adding more labeled examples in overlapping time periods."
                )

        # ── Margin-based confound detection ───────────────────────────────
        # Find segments where the top two behaviour predictions are close
        # (within 0.2), indicating the feature space is ambiguous between
        # those behaviours even when one doesn't exceed the threshold.
        if n >= 2:
            prob_matrix = merged[prob_cols].to_numpy(dtype=float)
            sorted_probs = np.sort(prob_matrix, axis=1)[:, ::-1]
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            narrow_margin_mask = margins < 0.2
            n_narrow = int(np.sum(narrow_margin_mask))
            narrow_rate = n_narrow / max(1, total)
            if narrow_rate > 0.01:
                # Identify which pair is most confused
                top2_idx = np.argsort(prob_matrix, axis=1)[:, -2:]
                narrow_segments = np.where(narrow_margin_mask)[0]
                pair_counts: dict[tuple[int, int], int] = {}
                for seg_idx in narrow_segments:
                    i_pair = tuple(sorted(top2_idx[seg_idx]))
                    pair_counts[i_pair] = pair_counts.get(i_pair, 0) + 1
                if pair_counts:
                    top_pair = max(pair_counts.items(), key=lambda kv: kv[1])
                    pi, pj = top_pair[0]
                    suggestions.append(
                        f"Margin-based confound: {n_narrow} segment(s) ({narrow_rate:.1%}) have "
                        f"<0.2 probability margin between top two predictions. "
                        f"Most confused pair: '{labels[pi]}' vs '{labels[pj]}' "
                        f"({top_pair[1]} segment(s))."
                    )

        if not suggestions:
            # Still report the max co-activation so the user knows the analysis ran
            max_coact = max((p["coactivation_rate"] for p in pairs), default=0.0)
            suggestions.append(
                f"No significant between-behaviour confounds detected "
                f"(max co-activation rate: {max_coact:.1%}, threshold: {threshold:.0%}). "
                f"If the UMAP shows overlap, try reviewing clips in the overlapping region."
            )

        # Generate heatmap.
        # Reorder so target behaviour comes first (if specified).
        target_idx: int | None = None
        if target_behavior_id:
            tbid = str(target_behavior_id).strip()
            # bid_list now uses directory-derived keys (e.g. "Freeze").
            # The caller may pass a UUID or a display name — check both.
            for idx, bid in enumerate(bid_list):
                if bid == tbid:
                    target_idx = idx
                    break
            if target_idx is None:
                # Try matching via behavior_names: UUID → display name → dir_key
                target_display = (behavior_names or {}).get(tbid, "")
                for idx, bid in enumerate(bid_list):
                    if bid == target_display or labels[idx] == target_display:
                        target_idx = idx
                        break

        if target_idx is not None and target_idx != 0:
            order = [target_idx] + [i for i in range(n) if i != target_idx]
            bid_list = [bid_list[i] for i in order]
            labels = [labels[i] for i in order]
            coact = coact[np.ix_(order, order)]

        out_dir = project_root / "derived" / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cross_behavior_confound_matrix.png"
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(max(4, 1.5 * n), max(3.5, 1.3 * n)))
            im = ax.imshow(coact, cmap="YlOrRd", vmin=0)

            # Bold-face the target behaviour label.
            x_labels = []
            y_labels = []
            for i, lbl in enumerate(labels):
                if target_idx is not None and i == 0:
                    x_labels.append(f"▸ {lbl}")
                    y_labels.append(f"▸ {lbl}")
                else:
                    x_labels.append(lbl)
                    y_labels.append(lbl)

            ax.set_xticks(range(n), x_labels, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(n), y_labels, fontsize=9)
            # Bold the target-behaviour tick labels
            if target_idx is not None:
                for tick_label in ax.get_xticklabels():
                    if tick_label.get_text().startswith("▸"):
                        tick_label.set_fontweight("bold")
                for tick_label in ax.get_yticklabels():
                    if tick_label.get_text().startswith("▸"):
                        tick_label.set_fontweight("bold")

            title = "Between-Behaviour Co-activation"
            if target_idx is not None:
                title = f"Confound Analysis — {labels[0]} vs All Others"
            ax.set_title(title, fontsize=11)

            for i in range(n):
                for j in range(n):
                    val = coact[i, j]
                    ax.text(j, i, f"{val:.2%}", ha="center", va="center", fontsize=8,
                            color="white" if val > 0.3 else "black")
            fig.colorbar(im, ax=ax, label="Co-activation rate", shrink=0.8)
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            logger.exception("Failed generating confound heatmap")
            out_path = None

        report = {
            "behavior_ids": bid_list,
            "behavior_labels": labels,
            "coactivation_matrix": coact.tolist(),
            "pairwise_rankings": pairs,
            "suggestions": suggestions,
            "out_path": str(out_path) if out_path else None,
        }
        write_json(out_dir / "cross_behavior_confound_report.json", report)
        return report

    # ------------------------------------------------------------------
    # Unified UMAP across all behaviour models
    # ------------------------------------------------------------------

    @staticmethod
    def _load_imported_training_rows(project_root: Path) -> pd.DataFrame:
        """Labeled rows imported from other projects (label_source ``imported:*``).

        These live only in the training set — they have no entry in this
        project's segment_features.parquet or reviewer_labels.parquet — so the
        unified UMAP must pull them from here to represent them at all.
        """
        ts = project_root / "derived" / "training_sets" / "training_set.parquet"
        if not ts.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(ts)
        except Exception:
            logger.debug("Failed to read training_set.parquet for imported rows", exc_info=True)
            return pd.DataFrame()
        if "label_source" not in df.columns or "label" not in df.columns:
            return pd.DataFrame()
        mask = df["label_source"].astype(str).str.startswith("imported:")
        return df[mask].reset_index(drop=True) if bool(mask.any()) else pd.DataFrame()

    @staticmethod
    def _embed_imported_segments(
        imported_df: pd.DataFrame,
        bid_list: list[str],
        latest_by_behavior: dict[str, Path],
        meta_cols: list[str],
        prob_cols: list[str],
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Score imported training rows through every behaviour model.

        Imported rows carry full features (in the model's own representation —
        it is what the model trained on) but no stored predictions, so they are
        absent from the prob-space the unified UMAP embeds in.  Running each
        binary behaviour model's ``predict_proba(...)[:, 1]`` (P of that
        behaviour) reconstructs the same ``prob_<bid>`` columns, placing imported
        examples in the identical coordinate space as this project's segments.

        Returns ``(rows, label_map)``: ``rows`` has ``meta_cols + prob_cols`` and
        ``label_map`` maps each imported segment_id to its imported label.
        """
        import pickle  # noqa: PLC0415

        n = len(imported_df)
        out = pd.DataFrame()
        for c in meta_cols:
            if c in imported_df.columns:
                out[c] = imported_df[c].values
            elif c in ("start_frame", "end_frame"):
                out[c] = np.zeros(n, dtype=int)
            else:
                out[c] = ""
        out["segment_id"] = imported_df["segment_id"].astype(str).values

        for bid, col in zip(bid_list, prob_cols):
            probs = np.zeros(n, dtype=float)
            md = latest_by_behavior.get(bid)
            if md is not None:
                try:
                    with open(md / "model_state.pkl", "rb") as f:
                        payload = pickle.load(f)
                    clf = payload["model"]
                    fcols = list(payload["feature_cols"])
                    feats = imported_df.reindex(columns=fcols, fill_value=0.0)
                    x = np.nan_to_num(
                        feats.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
                        nan=0.0, posinf=0.0, neginf=0.0,
                    )
                    proba = clf.predict_proba(x)
                    probs = (
                        proba[:, 1] if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2
                        else np.ravel(proba)
                    )
                except Exception:
                    logger.debug("Failed scoring imported rows through %s", bid, exc_info=True)
            out[col] = np.asarray(probs, dtype=float)

        label_map = dict(zip(
            out["segment_id"].tolist(),
            imported_df["label"].astype(str).tolist(),
        ))
        return out, label_map

    def generate_unified_umap(
        self,
        project_root: Path,
        behavior_names: dict[str, str] | None = None,
        cap: int = 8000,
        n_neighbors: int = 15,
        predicted_to_labeled_ratio: float = 5.0,
        target_behavior_label: str | None = None,
    ) -> dict[str, Any]:
        """Generate a single UMAP embedding coloured by dominant behaviour.

        Loads the shared segment representations and prediction probabilities
        from every behaviour model, assigns each segment the behaviour with
        the highest probability, and produces one combined UMAP plot.

        Labelled clips (from reviewer_labels.parquet) and examples imported from
        other projects (training-set rows tagged ``imported:*``, scored through
        each model into the same prob-space) are always included in full; the
        remaining unlabelled segments are representatively subsampled (stratified
        by session) up to *cap*.

        Returns
        -------
        dict with keys: out_path, n_segments, behaviors_used, method.
        """
        models_root = project_root / "derived" / "models"
        rep_dir = project_root / "derived" / "representations"

        # --- Load segment features -----------------------------------------------
        seg_path = rep_dir / "segment_features.parquet"
        if not seg_path.exists():
            return {"error": "segment_features.parquet not found."}

        # Load ONLY metadata columns — segment_features.parquet is 1-2 GB on
        # disk and 10+ GB decompressed. We only need identifier columns here;
        # prediction probabilities are loaded separately per-model below.
        # Skipping raw feature columns also causes _numeric_feature_columns()
        # to return [] which naturally bypasses the expensive pre-subsample PCA.
        _seg_meta_cols = ["segment_id", "session_id", "start_frame", "end_frame", "animal_id"]
        try:
            import pyarrow.parquet as _pq
            _available = set(_pq.read_schema(str(seg_path)).names)
            _load_cols = [c for c in _seg_meta_cols if c in _available]
        except Exception:
            _load_cols = _seg_meta_cols
        seg_df = pd.read_parquet(seg_path, columns=_load_cols if _load_cols else None)
        if "segment_id" not in seg_df.columns or seg_df.empty:
            return {"error": "Segment features are empty or malformed."}

        # Include enriched segments (computed on-the-fly for reviewed labels)
        # so that reviewed-label segments are represented in the embedding.
        enriched_path = rep_dir / "enriched_segments.parquet"
        if enriched_path.exists():
            try:
                enriched_df = pd.read_parquet(enriched_path, columns=_load_cols if _load_cols else None)
                if not enriched_df.empty and "segment_id" in enriched_df.columns:
                    seg_df = pd.concat(
                        [seg_df, enriched_df.reindex(columns=seg_df.columns, fill_value=0.0)],
                        ignore_index=True,
                    )
                    seg_df = seg_df.drop_duplicates(subset=["segment_id"], keep="first")
            except Exception:
                logger.debug("Failed loading enriched_segments.parquet", exc_info=True)

        # --- Discover model directories ------------------------------------------
        latest_by_behavior: dict[str, Path] = {}
        _nb_keys = {"no_behavior", "no_behaviour", "nobehavior", "nobehaviour"}
        if models_root.exists():
            for p in sorted(models_root.iterdir()):
                if not (p.is_dir() and (p / "model_state.pkl").exists() and p.name.startswith("behavior_model_")):
                    continue
                # Use directory name as canonical key — run_settings target_behavior
                # can be wrong (e.g. No_Behavior saved with another behaviour's ID).
                dir_key = p.name.removeprefix("behavior_model_").strip()
                if not dir_key:
                    settings = read_json(p / "run_settings.json", {})
                    dir_key = str(settings.get("target_behavior") or settings.get("target_behavior_id") or "").strip()
                if dir_key:
                    # Skip no_behavior models — they are trained with 0
                    # positives and produce degenerate predictions that
                    # corrupt the embedding via fillna(0.0).
                    if dir_key.lower().replace("_", "").replace(" ", "") in _nb_keys:
                        continue
                    latest_by_behavior[dir_key] = p

        if not latest_by_behavior:
            return {"error": "No behaviour models found."}

        bname = behavior_names or {}

        # --- Merge predictions from each model -----------------------------------
        # Keep metadata columns (session_id, start/end frame, animal_id) so they
        # propagate through to the coordinate parquet used for interactive selection.
        _meta_cols = ["segment_id"] + [
            c for c in ("session_id", "start_frame", "end_frame", "animal_id")
            if c in seg_df.columns
        ]
        merged = seg_df[_meta_cols].copy()
        prob_cols: list[str] = []
        bid_list: list[str] = []
        for bid, md in latest_by_behavior.items():
            pred_path = md / "segment_predictions.parquet"
            if not pred_path.exists():
                continue
            pred = pd.read_parquet(pred_path)
            if "segment_id" not in pred.columns or "prediction_prob" not in pred.columns:
                continue
            col = f"prob_{bid}"
            pred = pred[["segment_id", "prediction_prob"]].rename(columns={"prediction_prob": col})
            merged = merged.merge(pred, on="segment_id", how="left")
            prob_cols.append(col)
            bid_list.append(bid)

        if not prob_cols:
            return {"error": "No valid prediction files found."}

        merged[prob_cols] = merged[prob_cols].fillna(0.0)

        # --- Load reviewer labels for ground-truth colouring ---------------------
        label_path = project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        reviewed_labels: dict[str, str] = {}
        if label_path.exists():
            try:
                lbl_df = pd.read_parquet(label_path)
                if "segment_id" in lbl_df.columns and "review_label" in lbl_df.columns:
                    # Keep last review per segment — direct ID matching first.
                    _rl_vals = lbl_df["review_label"].astype(str).str.strip()
                    _rl_valid = _rl_vals.ne("") & ~_rl_vals.isin({"ambiguous", "boundary_error"})
                    reviewed_labels = dict(zip(
                        lbl_df.loc[_rl_valid, "segment_id"].astype(str),
                        _rl_vals[_rl_valid],
                    ))

                    # Fuzzy-match reviewed segments that don't have a direct ID
                    # match — reviewer labels often use different segment ID
                    # prefixes (e.g. "rand_session_…") than the representation
                    # features (e.g. "seg_m1_session_…"), so we fall back to
                    # matching by (session_id, closest frame range).
                    import re as _re

                    def _parse_review_segment(raw_id: str):
                        """Extract (session_id, start_frame, end_frame) from a reviewed segment ID."""
                        parts = raw_id.rsplit("_", 2)
                        if len(parts) >= 3:
                            try:
                                s = int(parts[-2])
                                e = int(parts[-1])
                                m = _re.search(r"(session_[0-9a-f]+)", "_".join(parts[:-2]))
                                if m:
                                    return m.group(1), s, e
                            except (ValueError, TypeError):
                                pass
                        return None, None, None

                    # Build spatial index for fast lookup:
                    # session_id → sorted list of (mid_frame, segment_id)
                    _feat_by_session: dict[str, list[tuple[float, str]]] = {}
                    # Prefer explicit columns which are always present in seg_df.
                    if "session_id" in seg_df.columns and "start_frame" in seg_df.columns:
                        _fidx = seg_df[["segment_id", "session_id", "start_frame", "end_frame"]].copy()
                        _fidx["_mid"] = (_fidx["start_frame"].astype(float) + _fidx["end_frame"].astype(float)) / 2.0
                        for _s, _grp in _fidx.groupby("session_id", sort=False):
                            _feat_by_session[str(_s)] = sorted(
                                zip(_grp["_mid"].tolist(), _grp["segment_id"].astype(str).tolist())
                            )
                        del _fidx
                    else:
                        # Fallback: parse from segment_id strings.
                        for fsid in merged["segment_id"].astype(str):
                            _s, _sf, _ef = _parse_review_segment(fsid)
                            if _s is not None:
                                _feat_by_session.setdefault(_s, []).append(((_sf + _ef) / 2.0, fsid))
                        for _s in _feat_by_session:
                            _feat_by_session[_s].sort()

                    # Precompute set of segment IDs already in merged for fast lookup.
                    _merged_seg_ids = set(merged["segment_id"].astype(str))

                    # Resolve all unmatched reviewed segment IDs to the nearest
                    # feature segment in the same session (no distance cap — the
                    # feature segment grid is sparse relative to the reviewed
                    # windows, so even distant matches are the best available).
                    unmatched_reviews: dict[str, str] = {}
                    for rev_sid, rev_label in reviewed_labels.items():
                        if rev_sid in _merged_seg_ids:
                            continue  # Already matched by exact ID.
                        rsess, rstart, rend = _parse_review_segment(rev_sid)
                        if rsess is None or rsess not in _feat_by_session:
                            continue
                        # Binary search for nearest segment midpoint.
                        r_mid = (rstart + rend) / 2.0
                        entries = _feat_by_session[rsess]
                        import bisect
                        pos = bisect.bisect_left(entries, (r_mid,))
                        best_seg: str | None = None
                        best_dist = float("inf")
                        for idx in (pos - 1, pos):
                            if 0 <= idx < len(entries):
                                f_mid, fsid = entries[idx]
                                dist = abs(r_mid - f_mid)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_seg = fsid
                        if best_seg is not None:
                            # If multiple reviewed segments map to the same
                            # feature segment, keep the first assignment.
                            if best_seg not in unmatched_reviews:
                                unmatched_reviews[best_seg] = rev_label

                    # Merge fuzzy-matched labels (only if not already labeled).
                    for fsid, rlabel in unmatched_reviews.items():
                        if fsid not in reviewed_labels:
                            reviewed_labels[fsid] = rlabel
            except Exception:
                logger.debug("Failed loading reviewer labels", exc_info=True)

        # Fold in examples imported from other projects.  They live only in the
        # training set (no segment_features / reviewer_labels entry), so without
        # this they would be invisible in the embedding even though the model
        # trains on them.  Scoring them through each model places them in the
        # same prob-space; their imported label is treated as ground truth so
        # they render as reviewed points of their behaviour.
        try:
            imported_rows = self._load_imported_training_rows(project_root)
            if not imported_rows.empty and bid_list:
                imp_merged, imp_label_map = self._embed_imported_segments(
                    imported_rows, bid_list, latest_by_behavior, _meta_cols, prob_cols,
                )
                if not imp_merged.empty:
                    imp_merged = imp_merged.reindex(columns=merged.columns, fill_value=0.0)
                    merged = pd.concat([merged, imp_merged], ignore_index=True)
                    merged[prob_cols] = merged[prob_cols].fillna(0.0)
                    for _sid, _lab in imp_label_map.items():
                        _lab = str(_lab).strip()
                        if _lab and _lab not in {"ambiguous", "boundary_error"}:
                            reviewed_labels[str(_sid)] = _lab
                    logger.info(
                        "Unified UMAP: embedded %d imported example(s) from other projects.",
                        len(imp_merged),
                    )
        except Exception:
            logger.debug("Failed to embed imported examples in unified UMAP", exc_info=True)

        # Assign dominant behaviour label using all models.
        prob_arr = merged[prob_cols].to_numpy(dtype=float)
        dominant_idx = np.argmax(prob_arr, axis=1)
        dominant_prob = np.max(prob_arr, axis=1)

        # Build case-insensitive canonical label map so that directory-derived
        # names (e.g. "No_Behavior") and reviewer-label values (e.g.
        # "no_behavior") collapse to one consistent label.
        _label_canon: dict[str, str] = {}
        for bid in bid_list:
            _label_canon[bid.lower()] = bid
        for _k, _v in bname.items():
            _label_canon[_k.lower()] = _v
            _label_canon[_v.lower()] = _v

        def _canon(label: str) -> str:
            return _label_canon.get(label.lower(), label)

        def _resolve_label(raw: str) -> str:
            """Resolve a raw behavior label (including pipe-separated multi-labels) to a display name."""
            raw = raw.strip()
            if "|" in raw:
                parts = sorted(p.strip() for p in raw.split("|") if p.strip())
                return " + ".join(bname.get(p, p) for p in parts)
            return _canon(bname.get(raw, raw))

        # Build label for each segment: prefer reviewer label, else use model prediction.
        # Vectorized to avoid O(N) pandas iloc lookups.
        _seg_ids_arr = merged["segment_id"].astype(str).to_numpy()
        _rev_series = pd.Series(reviewed_labels, dtype=str) if reviewed_labels else pd.Series(dtype=str)
        _rev_mapped = pd.Series(_seg_ids_arr).map(_rev_series).to_numpy(dtype=object)
        _has_review = ~pd.isnull(_rev_mapped)
        _pred_bids = np.array([bid_list[i] for i in dominant_idx])
        _pred_labels = np.array([_canon(bname.get(b, b)) for b in _pred_bids])
        _review_labels_canon = np.where(
            _has_review,
            np.array([_resolve_label(str(v)) if v is not None else "" for v in _rev_mapped]),
            "",
        )
        # Threshold of 0.5 (majority-vote confidence) keeps only segments where
        # one behaviour clearly dominates.  Lower-confidence predictions have
        # ambiguous probability vectors that land near the scaled-feature mean
        # (undefined cosine direction) and scatter randomly in the UMAP.
        _unclassified_mask = (~_has_review) & (dominant_prob < 0.5)
        merged["dominant_behavior"] = np.where(
            _has_review, _review_labels_canon,
            np.where(_unclassified_mask, "Unclassified", _pred_labels)
        )
        merged["is_labeled"] = _has_review
        del _seg_ids_arr, _rev_series, _rev_mapped, _has_review, _pred_bids, _pred_labels, _review_labels_canon, _unclassified_mask

        # --- Prepare feature matrix for UMAP ------------------------------------
        # Use model prediction probabilities as the primary feature space.
        # These are compact, supervised outputs that capture behavior-discriminative
        # patterns far better than raw, behavior-agnostic segment statistics
        # (which can have hundreds of noisy dimensions that dilute separation).
        # Optionally augment with a small number of PCA-derived raw features.
        feature_cols = prob_cols[:]  # prediction probs from each behaviour model

        # Add a few PCA components of the raw segment features to capture
        # additional structure not reflected in model predictions.
        # NOTE: seg_df was loaded with metadata-only columns, so
        # _numeric_feature_columns returns [] and this block is skipped,
        # which is intentional — prob_cols are already the best features.
        raw_feature_cols = self._numeric_feature_columns(seg_df)
        if raw_feature_cols:
            work_raw = merged.merge(seg_df[["segment_id"] + raw_feature_cols], on="segment_id", how="left", suffixes=("", "_raw"))
            # Resolve column names after merge (avoid duplicates with prob cols).
            resolved_raw = [c for c in raw_feature_cols if c in work_raw.columns and c not in prob_cols]
            if len(resolved_raw) > 10:
                try:
                    from sklearn.decomposition import PCA as _PCA
                    from sklearn.preprocessing import StandardScaler as _Scaler

                    raw_vals = work_raw[resolved_raw].to_numpy(dtype=float)
                    del work_raw
                    raw_vals = np.nan_to_num(raw_vals, nan=0.0, posinf=0.0, neginf=0.0)
                    raw_vals = _Scaler().fit_transform(raw_vals)
                    n_pca = min(10, len(resolved_raw), raw_vals.shape[0])
                    pca_coords = _PCA(n_components=n_pca, random_state=42).fit_transform(raw_vals)
                    del raw_vals
                    for i in range(n_pca):
                        col_name = f"_pca_raw_{i}"
                        merged[col_name] = pca_coords[:, i]
                        feature_cols.append(col_name)
                    del pca_coords
                except Exception:
                    logger.debug("PCA augmentation failed, using probabilities only", exc_info=True)
            elif resolved_raw:
                feature_cols.extend(resolved_raw)
        del seg_df  # no longer needed — free metadata DataFrame

        if not feature_cols:
            return {"error": "No numeric feature columns for UMAP."}

        # seg_df features were already merged into `merged` (prob columns);
        # any additional columns are added above. Avoid a redundant full copy.
        work = merged
        if work.empty:
            return {"error": "No overlapping segments between features and predictions."}

        # --- Representative subsampling: all labeled + stratified unlabeled ------
        labeled_mask = work["is_labeled"].to_numpy(dtype=bool)
        labeled_df = work[labeled_mask]
        unlabeled_df = work[~labeled_mask]

        rng = np.random.default_rng(42)
        unlabeled_budget = max(0, cap - len(labeled_df))
        if len(unlabeled_df) > unlabeled_budget and unlabeled_budget > 0:
            # Stratified subsample by session to keep representation even
            if "session_id" in unlabeled_df.columns:
                sessions = unlabeled_df["session_id"].astype(str).to_numpy()
                unique_sessions = sorted(set(sessions))
                per_session = max(1, unlabeled_budget // len(unique_sessions))
                sampled_idx: list[int] = []
                for sess in unique_sessions:
                    sess_idx = np.where(sessions == sess)[0]
                    n_take = min(len(sess_idx), per_session)
                    sampled_idx.extend(rng.choice(sess_idx, size=n_take, replace=False).tolist())
                # Fill remaining budget from any session
                if len(sampled_idx) < unlabeled_budget:
                    remaining = sorted(set(range(len(unlabeled_df))) - set(sampled_idx))
                    extra = min(unlabeled_budget - len(sampled_idx), len(remaining))
                    sampled_idx.extend(rng.choice(remaining, size=extra, replace=False).tolist())
                unlabeled_df = unlabeled_df.iloc[sorted(sampled_idx)]
            else:
                idx = rng.choice(len(unlabeled_df), size=unlabeled_budget, replace=False)
                unlabeled_df = unlabeled_df.iloc[sorted(idx)]
        elif unlabeled_budget <= 0:
            unlabeled_df = unlabeled_df.iloc[:0]

        work = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
        del merged, labeled_df, unlabeled_df  # free full-set DataFrame before UMAP

        # --- Class-balanced downsampling of PREDICTED segments --------------------
        # Without balancing, the dominant predicted class (often No_Behavior /
        # Unclassified) can have 5-10x more points than real behaviours,
        # drowning out useful structure in the plot.  Cap each predicted class
        # to at most `predicted_to_labeled_ratio` × the largest *reviewed*
        # class so that the UMAP reflects behaviour diversity rather than
        # background prevalence.
        _labeled_in_work = work["is_labeled"].to_numpy(dtype=bool)
        _labels_in_work = work["dominant_behavior"].to_numpy(dtype=str)
        _label_classes = sorted(set(_labels_in_work))
        # Largest reviewed class count (or fallback of cap/len_classes).
        _max_reviewed = 0
        for _cls in _label_classes:
            _cnt = int(np.sum((_labels_in_work == _cls) & _labeled_in_work))
            if _cnt > _max_reviewed:
                _max_reviewed = _cnt
        _ratio = max(1.0, float(predicted_to_labeled_ratio))
        _pred_cap_per_class = max(200, int(_max_reviewed * _ratio)) if _max_reviewed > 0 else max(200, cap // max(1, len(_label_classes)))
        _keep_mask = np.ones(len(work), dtype=bool)
        for _cls in _label_classes:
            _cls_pred_mask = (_labels_in_work == _cls) & (~_labeled_in_work)
            _cls_pred_idx = np.where(_cls_pred_mask)[0]
            if len(_cls_pred_idx) > _pred_cap_per_class:
                _drop_idx = rng.choice(_cls_pred_idx, size=len(_cls_pred_idx) - _pred_cap_per_class, replace=False)
                _keep_mask[_drop_idx] = False
        work = work[_keep_mask].reset_index(drop=True)

        # Drop "Unclassified" *predicted* (non-reviewed) segments before embedding.
        # These are segments where every model's confidence is < 0.3 — their
        # probability vectors are nearly uniform and low-magnitude.  After
        # StandardScaler the cosine direction of these vectors is undefined, so
        # UMAP scatters them randomly across the plot rather than placing them
        # in a meaningful cluster.  Since "Unclassified" is a model-uncertainty
        # bucket (not a real behaviour), removing them from the embedding produces
        # a cleaner plot that accurately reflects behaviour structure.  Reviewed
        # segments labelled "Unclassified" are always preserved.
        _unclass_pred_mask = (
            (work["dominant_behavior"].to_numpy(dtype=str) == "Unclassified")
            & (~work["is_labeled"].to_numpy(dtype=bool))
        )
        if np.any(_unclass_pred_mask):
            n_unclass_dropped = int(np.sum(_unclass_pred_mask))
            work = work[~_unclass_pred_mask].reset_index(drop=True)
            logger.debug(
                "Dropped %d Unclassified predicted segments from UMAP embedding "
                "(low-confidence, max_prob < 0.3; scatter would be random under cosine metric)",
                n_unclass_dropped,
            )

        # Drop segments with all-zero probability vectors — these are segments that
        # were never scored by any model (they ended up in seg_df but not in any
        # segment_predictions.parquet).  Under cosine distance, a zero vector has
        # undefined similarity to everything else, so UMAP places these points
        # randomly far from the real clusters, producing the scattered outlier dots
        # visible in the plot.  Labeled segments are always preserved regardless.
        _prob_cols_in_work = [c for c in prob_cols if c in work.columns]
        if _prob_cols_in_work:
            _has_any_pred = work[_prob_cols_in_work].abs().sum(axis=1) > 1e-9
            _keep_nonzero = _has_any_pred | work["is_labeled"]
            work = work[_keep_nonzero].reset_index(drop=True)

        x = work[feature_cols].to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        labels_arr = work["dominant_behavior"].to_numpy()
        is_labeled_arr = work["is_labeled"].to_numpy(dtype=bool)

        # Impute labeled segments that have all-zero feature vectors using the
        # class-mean of non-zero segments with the same behavior label.
        # Zero vectors under cosine distance have undefined similarity, which
        # causes UMAP to place them at random positions — producing the
        # scattered outlier dots visible when reviewed segments lack predictions
        # (i.e. they were labeled before any model was trained or come from
        # enriched_segments that were never scored by segment_predictions.parquet).
        _row_norms = np.linalg.norm(x, axis=1)
        _zero_labeled = is_labeled_arr & (_row_norms < 1e-10)
        if np.any(_zero_labeled):
            # Pre-build map: canonical class label → index in feature_cols for the
            # corresponding probability column.  Used by the fallback branch so it
            # can construct a class-specific identity vector instead of the global
            # mean.  The global mean maps to ~zero after StandardScaler (centering
            # subtracts it), which produces an undefined cosine direction and causes
            # random UMAP placement — the root cause of the scattered outlier dots.
            _cls_to_prob_idx: dict[str, int] = {}
            for _fi, _fc in enumerate(feature_cols):
                if _fc.startswith("prob_"):
                    _raw_bid = _fc[len("prob_"):]
                    _cls_to_prob_idx[_raw_bid] = _fi
                    _cls_to_prob_idx[_canon(bname.get(_raw_bid, _raw_bid))] = _fi

            for _cls in set(labels_arr[_zero_labeled]):
                _cls_nonzero_mask = (~_zero_labeled) & (labels_arr == _cls)
                if np.any(_cls_nonzero_mask):
                    _impute_vec = x[_cls_nonzero_mask].mean(axis=0)
                else:
                    # No same-class non-zero rows exist (e.g. all labeled instances
                    # were reviewed before any model was trained, or no predicted
                    # segments of this class survived downsampling).
                    # Using the global mean here would map to ~zero after
                    # StandardScaler, giving an undefined cosine direction.
                    # Instead, construct a class-specific probability identity vector.
                    _impute_vec = np.zeros(x.shape[1])
                    _cls_norm = _cls.strip()
                    if _cls_norm in _cls_to_prob_idx:
                        _impute_vec[_cls_to_prob_idx[_cls_norm]] = 1.0
                    elif " + " in _cls_norm:
                        # Multi-label class (e.g. "Dip + Freeze"): distribute
                        # probability evenly across the component columns.
                        _parts = [p.strip() for p in _cls_norm.split(" + ")]
                        _matched_idxs = [_cls_to_prob_idx[p] for p in _parts if p in _cls_to_prob_idx]
                        for _fi in _matched_idxs:
                            _impute_vec[_fi] = 1.0 / max(1, len(_matched_idxs))
                    if np.linalg.norm(_impute_vec) < 1e-10:
                        # Unknown class with no matching probability column: use a
                        # tiny uniform value so the vector is non-zero after scaling
                        # (the point will land near the embedding centre, but cosine
                        # similarity will be defined and deterministic).
                        _impute_vec[:] = 1e-4
                x[_zero_labeled & (labels_arr == _cls)] = _impute_vec
            logger.debug(
                "Imputed %d labeled zero-vector segments with class-mean features",
                int(np.sum(_zero_labeled)),
            )

        # --- Standardise features before UMAP so probability and PCA columns
        # are on comparable scales. -------------------------------------------
        try:
            from sklearn.preprocessing import StandardScaler
            x = StandardScaler().fit_transform(x)
        except ImportError:
            col_means = np.nanmean(x, axis=0)
            col_stds = np.nanstd(x, axis=0)
            col_stds[col_stds < 1e-10] = 1.0
            x = (x - col_means) / col_stds

        # Safety net: after scaling, any still-zero-norm rows produce undefined
        # cosine similarity and UMAP would place them randomly at the periphery.
        # For predicted (non-reviewed) segments: drop them entirely — random
        # placement is worse than absence.
        # For labeled segments: must keep, so use a tiny deterministic
        # perturbation so they at least land near the embedding centre.
        _post_norms = np.linalg.norm(x, axis=1)
        _still_zero = _post_norms < 1e-10
        if np.any(_still_zero):
            _still_zero_pred = _still_zero & ~is_labeled_arr
            if np.any(_still_zero_pred):
                _keep_sz = ~_still_zero_pred
                x = x[_keep_sz]
                labels_arr = labels_arr[_keep_sz]
                is_labeled_arr = is_labeled_arr[_keep_sz]
                work = work[_keep_sz].reset_index(drop=True)
                logger.debug(
                    "Dropped %d post-scale zero-norm predicted segments (would scatter randomly)",
                    int(np.sum(_still_zero_pred)),
                )
            # Labeled zero-norm rows: tiny perturbation so cosine is defined.
            _post_norms2 = np.linalg.norm(x, axis=1)
            _still_zero_lbl = (_post_norms2 < 1e-10) & is_labeled_arr
            if np.any(_still_zero_lbl):
                _n_lbl_zero = int(np.sum(_still_zero_lbl))
                x[_still_zero_lbl] = np.random.default_rng(42).standard_normal((_n_lbl_zero, x.shape[1])) * 1e-4
                logger.debug("Post-scale zero-norm safety: perturbed %d labeled rows", _n_lbl_zero)

        # --- UMAP / PCA ---------------------------------------------------------
        # Split into labeled (reviewed) and predicted subsets for coordinate
        # tracking; UMAP is fit on ALL points so the global topology is preserved
        # (fitting on labeled-only produces serpentine chains when the labeled
        # probability vectors don't span the full simplex).
        x_labeled = x[is_labeled_arr]
        x_predicted = x[~is_labeled_arr]

        coords = None
        method = "PCA"
        try:
            import umap as umap_lib  # type: ignore[import]

            # Use cosine metric — works well for probability-based features
            # where direction matters more than magnitude.
            n_samples = x.shape[0]
            effective_neighbors = max(5, min(50, int(n_neighbors), n_samples // 4))
            reducer = umap_lib.UMAP(
                n_components=2,
                n_neighbors=effective_neighbors,
                min_dist=0.15,
                metric="cosine",
                random_state=42,
            )
            # Spectral initialisation can fail with a small eigengap when
            # probability features cluster tightly along a low-dimensional
            # simplex.  A tiny jitter (1e-3 σ) breaks the degeneracy and
            # lets spectral init succeed, producing the stable round-blob
            # topology rather than the random-fallback serpentine one.
            _x_jitter = x + np.random.default_rng(0).standard_normal(x.shape) * 1e-3
            coords = reducer.fit_transform(_x_jitter)
            del _x_jitter
            method = "UMAP"
        except Exception:
            try:
                from sklearn.decomposition import PCA

                coords = PCA(n_components=2, random_state=42).fit_transform(x)
            except Exception:
                return {"error": "Neither UMAP nor PCA available."}

        # --- Plot ----------------------------------------------------------------
        out_dir = project_root / "derived" / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "unified_behavior_umap.png"
        out_svg_path = out_dir / "unified_behavior_umap.svg"
        try:
            import matplotlib.pyplot as plt

            classes = sorted(set(labels_arr))
            n_classes = len(classes)

            # Dynamic figure width: extra space on the right for the legend
            legend_col_width = 2.2  # inches per legend column
            legend_ncol = max(1, min(3, n_classes // 8 + 1))
            fig_w = max(9.0, 7.0 + legend_ncol * legend_col_width)
            fig_h = max(5.5, 4.5 + n_classes * 0.05)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            fig.subplots_adjust(right=1.0 - legend_ncol * legend_col_width / fig_w)

            cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20", max(1, n_classes))
            color_map = {cls: cmap(i) for i, cls in enumerate(classes)}

            # Map each raw class value → friendly display name.
            # Multi-label classes (pipe-separated) are already resolved in dominant_behavior,
            # but handle any residual raw IDs here as a safety net.
            def _disp(raw_cls: str) -> str:
                if "|" in raw_cls:
                    parts = sorted(p.strip() for p in raw_cls.split("|") if p.strip())
                    return " + ".join(bname.get(p, p) for p in parts)
                return bname.get(raw_cls, raw_cls)
            display_name: dict[str, str] = {cls: _disp(cls) for cls in classes}

            # Determine rendering order: target behaviour last so it draws on top.
            target_lbl = str(target_behavior_label or "").strip()
            non_target = [c for c in classes if c != target_lbl and c != "Unclassified"]
            render_order = ["Unclassified"] + non_target
            if target_lbl and target_lbl in classes:
                render_order.append(target_lbl)
            else:
                render_order = ["Unclassified"] + [c for c in classes if c != "Unclassified"]
            render_order = [c for c in render_order if c in set(classes)]

            # Per-behaviour reviewed counts for the subtitle.
            reviewed_counts: dict[str, int] = {}
            for cls in classes:
                reviewed_counts[cls] = int(np.sum(
                    (np.array(labels_arr) == cls) & is_labeled_arr
                ))

            z_base = 1
            # Plot unlabeled points first (smaller, more transparent)
            for z_idx, cls in enumerate(render_order):
                mask = (np.array(labels_arr) == cls) & (~is_labeled_arr)
                if not np.any(mask):
                    continue
                is_target = (cls == target_lbl and target_lbl)
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=6 if is_target else 4,
                    alpha=0.45 if is_target else 0.3,
                    color=color_map[cls], label=f"{display_name[cls]} (predicted)",
                    rasterized=True,
                    zorder=z_base + z_idx,
                )
            z_labeled_base = z_base + len(render_order)
            # Plot labeled points on top (larger, opaque, edged)
            for z_idx, cls in enumerate(render_order):
                mask = (np.array(labels_arr) == cls) & is_labeled_arr
                if not np.any(mask):
                    continue
                is_target = (cls == target_lbl and target_lbl)
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=36 if is_target else 28,
                    alpha=0.95 if is_target else 0.85,
                    color=color_map[cls],
                    edgecolors="black", linewidths=0.6 if is_target else 0.4,
                    label=f"{display_name[cls]} ({reviewed_counts.get(cls, 0)} reviewed)",
                    zorder=z_labeled_base + z_idx,
                )

            # Annotate cluster labels at the densest region of each class,
            # not the centroid, so labels point to where most points actually are.
            x_range = float(coords[:, 0].max() - coords[:, 0].min()) or 1.0
            y_range = float(coords[:, 1].max() - coords[:, 1].min()) or 1.0
            offset_dist_x = x_range * 0.08
            offset_dist_y = y_range * 0.10
            # Find density peak for each class using 2D histogram.
            centroids: dict[str, tuple[float, float]] = {}
            for cls in classes:
                if cls == "Unclassified":
                    continue
                mask = np.array(labels_arr) == cls
                if not np.any(mask):
                    continue
                cx_arr = coords[mask, 0]
                cy_arr = coords[mask, 1]
                if len(cx_arr) < 3:
                    # Too few points — fall back to median.
                    centroids[cls] = (float(np.median(cx_arr)), float(np.median(cy_arr)))
                    continue
                try:
                    # Use 2D histogram to find the densest bin.
                    n_bins = max(10, min(50, int(np.sqrt(len(cx_arr)))))
                    hist, xedges, yedges = np.histogram2d(cx_arr, cy_arr, bins=n_bins)
                    peak_idx = np.unravel_index(np.argmax(hist), hist.shape)
                    peak_x = float((xedges[peak_idx[0]] + xedges[peak_idx[0] + 1]) / 2)
                    peak_y = float((yedges[peak_idx[1]] + yedges[peak_idx[1] + 1]) / 2)
                    centroids[cls] = (peak_x, peak_y)
                except Exception:
                    centroids[cls] = (float(np.median(cx_arr)), float(np.median(cy_arr)))
            # Offset directions: cycle through angles to avoid collisions.
            _angles = [45, -45, 135, -135, 0, 90, 180, -90]
            import math
            # Smaller annotation font when many classes to reduce overlap.
            annot_fs = max(5, 9 - n_classes // 5)
            for i, cls in enumerate(sorted(centroids.keys())):
                cx, cy = centroids[cls]
                angle = math.radians(_angles[i % len(_angles)])
                ox = cx + offset_dist_x * math.cos(angle)
                oy = cy + offset_dist_y * math.sin(angle)
                ax.annotate(
                    display_name[cls], xy=(cx, cy), xytext=(ox, oy),
                    fontsize=annot_fs, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_map[cls], alpha=0.85),
                    arrowprops=dict(
                        arrowstyle="-",
                        color=color_map[cls],
                        lw=1.2,
                        shrinkA=0,
                        shrinkB=4,
                    ),
                    zorder=z_labeled_base + len(render_order) + 1,
                )

            # Build title with per-behaviour reviewed counts.
            n_labeled = int(np.sum(is_labeled_arr))
            n_unlabeled = int(len(is_labeled_arr)) - n_labeled
            review_parts = [
                f"{display_name.get(cls, cls)}: {reviewed_counts[cls]}"
                for cls in classes if cls != "Unclassified" and reviewed_counts.get(cls, 0) > 0
            ]
            # Wrap review summary if it would be very long (many behaviors reviewed)
            if len(review_parts) > 6:
                review_summary = f"{n_labeled} labeled across {len(review_parts)} behaviors"
            else:
                review_summary = ", ".join(review_parts) if review_parts else f"{n_labeled} total"
            ax.set_title(
                f"Unified Behaviour Embedding ({method}) — "
                f"{n_classes} classes  |  reviewed: {review_summary}  |  "
                f"{n_unlabeled} predicted",
                fontsize=9,
                wrap=True,
            )
            ax.set_xlabel(f"{method} 1")
            ax.set_ylabel(f"{method} 2")
            # Place legend outside the axes on the right; dynamic font + ncol
            legend_fs = max(5, 9 - n_classes // 8)
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
                frameon=True,
                framealpha=0.7,
                fontsize=legend_fs,
                markerscale=2,
                ncol=legend_ncol,
            )
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            fig.savefig(out_svg_path, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            logger.exception("Failed generating unified behaviour UMAP")
            out_path = None

        result = {
            "out_path": str(out_path) if out_path else None,
            "n_segments": int(len(x)),
            "behaviors_used": [bname.get(b, b) for b in bid_list],
            "method": method,
        }
        write_json(out_dir / "unified_umap_report.json", result)

        # Persist coordinate data for interactive selection downstream
        if coords is not None:
            try:
                coord_df = work[["segment_id"]].copy()
                for _mc in ("session_id", "animal_id"):
                    if _mc in work.columns:
                        coord_df[_mc] = work[_mc].astype(str).values
                    else:
                        coord_df[_mc] = ""
                for _mc in ("start_frame", "end_frame"):
                    if _mc in work.columns:
                        coord_df[_mc] = work[_mc].astype(int).values
                coord_df["umap_x"] = coords[:, 0]
                coord_df["umap_y"] = coords[:, 1]
                coord_df["behavior_label"] = labels_arr
                coord_df["is_labeled"] = is_labeled_arr
                coord_df.to_parquet(out_dir / "unified_umap_coordinates.parquet", index=False)
            except Exception:
                logger.debug("Failed to save UMAP coordinate data", exc_info=True)

        return result

    def generate_unsupervised_umap(
        self,
        project_root: Path,
        cap: int = 8000,
        n_neighbors: int = 15,
        min_cluster_size: int = 50,
        pca_components: int = 50,
        progress_cb: Any = None,
    ) -> dict[str, Any]:
        """Generate an *unsupervised* UMAP embedding from raw segment features.

        Unlike :meth:`generate_unified_umap` (which embeds per-behaviour model
        prediction probabilities and therefore needs trained models + reviewer
        labels), this method embeds the raw numeric feature columns of
        ``segment_features.parquet`` directly — no models, no labels required.
        Points are colour-coded by density-based cluster (HDBSCAN, falling back
        to KMeans), so latent structure in the data surfaces without any
        supervision.

        Writes
        ------
        ``derived/evaluation/unsupervised_umap.png`` (+ ``.svg``)
        ``derived/evaluation/unsupervised_umap_coordinates.parquet`` — the same
        schema the interactive "Select from UMAP" dialog consumes
        (``segment_id, session_id, animal_id, start_frame, end_frame, umap_x,
        umap_y, behavior_label, is_labeled``), with ``behavior_label`` holding
        the cluster name.

        Returns
        -------
        dict with keys: out_path, coord_path, n_segments, n_clusters, method,
        cluster_method.
        """
        import math

        def _emit(step: int, msg: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(int(step), 5, msg, msg)
                except Exception:
                    pass

        rep_dir = project_root / "derived" / "representations"
        seg_path = rep_dir / "segment_features.parquet"
        if not seg_path.exists():
            return {"error": "segment_features.parquet not found."}

        # --- Determine metadata + numeric feature columns from the schema -------
        # segment_features.parquet can be 1-2 GB on disk / 10+ GB decompressed,
        # so we read only a representative subset of row groups rather than the
        # whole file (see the row-group sampling below).
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception:
            return {"error": "pyarrow is required to read segment features."}

        exclude = {
            "label_true", "label_pred", "prediction_prob", "uncertainty_score",
            "review_label", "segment_id", "session_id", "animal_id",
            "subject_id", "label", "label_source", "start_frame", "end_frame",
        }
        meta_wanted = ["segment_id", "session_id", "animal_id", "start_frame", "end_frame"]
        try:
            pf = pq.ParquetFile(str(seg_path))
            schema = pf.schema_arrow
        except Exception as exc:
            return {"error": f"Could not read segment features: {exc}"}

        all_names = list(schema.names)
        meta_cols = [c for c in meta_wanted if c in all_names]
        feature_cols = [
            field.name
            for field in schema
            if field.name not in exclude
            and field.name not in meta_wanted
            and (
                pa.types.is_floating(field.type)
                or pa.types.is_integer(field.type)
                or pa.types.is_boolean(field.type)
            )
        ]
        if "segment_id" not in meta_cols:
            return {"error": "segment_features.parquet has no segment_id column."}
        if not feature_cols:
            return {"error": "No numeric feature columns found for unsupervised UMAP."}

        # --- Load a representative subset of row groups ------------------------
        # Reading evenly-spaced row groups (rather than the whole file) keeps I/O
        # bounded while still covering the full recording timeline.
        _emit(1, "Loading segment features…")
        cap = max(100, int(cap))
        rg_count = pf.num_row_groups
        rg_rows = [pf.metadata.row_group(i).num_rows for i in range(rg_count)]
        n_total = int(sum(rg_rows))
        if n_total == 0:
            return {"error": "Segment features are empty."}

        # Oversample ~3x so the per-session subsample below has choices.
        target_read = min(n_total, max(cap * 3, cap + 1))
        if n_total > target_read and rg_count > 1:
            avg_rows = max(1, n_total // rg_count)
            n_groups = max(1, min(rg_count, math.ceil(target_read / avg_rows)))
            picked = sorted(set(np.linspace(0, rg_count - 1, num=n_groups).astype(int).tolist()))
        else:
            picked = list(range(rg_count))

        try:
            table = pf.read_row_groups(picked, columns=meta_cols + feature_cols)
            work = table.to_pandas()
            del table
        except Exception as exc:
            return {"error": f"Failed reading segment features: {exc}"}
        if work.empty:
            return {"error": "No segment rows available for embedding."}

        # Stratified-by-session subsample down to the cap.
        rng = np.random.default_rng(42)
        if len(work) > cap:
            if "session_id" in work.columns:
                sessions = work["session_id"].astype(str).to_numpy()
                unique_sessions = sorted(set(sessions))
                per_session = max(1, cap // len(unique_sessions))
                keep_pos: list[int] = []
                for sess in unique_sessions:
                    sess_idx = np.where(sessions == sess)[0]
                    n_take = min(len(sess_idx), per_session)
                    keep_pos.extend(rng.choice(sess_idx, size=n_take, replace=False).tolist())
                if len(keep_pos) < cap:
                    remaining = sorted(set(range(len(work))) - set(keep_pos))
                    extra = min(cap - len(keep_pos), len(remaining))
                    if extra > 0:
                        keep_pos.extend(rng.choice(remaining, size=extra, replace=False).tolist())
                work = work.iloc[sorted(keep_pos)].reset_index(drop=True)
            else:
                idx = sorted(rng.choice(len(work), size=cap, replace=False).tolist())
                work = work.iloc[idx].reset_index(drop=True)

        # --- Build + scale feature matrix --------------------------------------
        _emit(2, "Scaling & reducing features…")
        x = work[feature_cols].to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        n_samples = x.shape[0]
        try:
            from sklearn.preprocessing import StandardScaler
            x = StandardScaler().fit_transform(x)
        except ImportError:
            col_means = np.nanmean(x, axis=0)
            col_stds = np.nanstd(x, axis=0)
            col_stds[col_stds < 1e-10] = 1.0
            x = (x - col_means) / col_stds
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # PCA pre-reduction with whitening: denoises high-dimensional raw
        # features, speeds up the UMAP neighbour search, and — crucially —
        # equalises component variance so the embedding isn't dominated by a
        # handful of high-variance features (which collapses everything into one
        # undifferentiated blob).
        pca_components = int(pca_components)
        if pca_components > 0 and x.shape[1] > pca_components and n_samples > pca_components:
            try:
                from sklearn.decomposition import PCA
                x = PCA(n_components=pca_components, whiten=True, random_state=42).fit_transform(x)
            except Exception:
                logger.debug("PCA pre-reduction failed; using scaled features", exc_info=True)

        # --- UMAP --------------------------------------------------------------
        # min_dist=0.0 packs each neighbourhood tightly, which separates dense
        # regions into distinct islands instead of one smooth cloud — this is
        # what lets the downstream clustering find structure.
        _emit(3, "Computing UMAP embedding…")
        coords = None
        method = "PCA"
        try:
            import umap as umap_lib  # type: ignore[import]

            effective_neighbors = max(5, min(50, int(n_neighbors), max(2, n_samples // 4)))
            reducer = umap_lib.UMAP(
                n_components=2,
                n_neighbors=effective_neighbors,
                min_dist=0.0,
                metric="euclidean",
                random_state=42,
            )
            coords = reducer.fit_transform(x)
            method = "UMAP"
        except Exception:
            try:
                from sklearn.decomposition import PCA
                coords = PCA(n_components=2, random_state=42).fit_transform(x)
            except Exception:
                return {"error": "Neither UMAP nor PCA available."}

        # --- Cluster on the 2-D embedding so colours match visible blobs -------
        _emit(4, "Clustering…")
        cluster_method = "none"
        cluster_ids = np.zeros(n_samples, dtype=int)
        try:
            import hdbscan  # type: ignore[import-untyped]

            mcs = max(5, int(min_cluster_size))
            # min_samples below min_cluster_size makes HDBSCAN less eager to
            # label points as noise, so genuine sub-structure is recovered
            # rather than swallowed into one giant cluster.
            min_samples = max(1, min(10, mcs // 5))
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=min_samples,
                cluster_selection_method="eom",
            )
            cluster_ids = np.asarray(clusterer.fit_predict(coords), dtype=int)
            cluster_method = "HDBSCAN"
            n_found = len({int(c) for c in cluster_ids if c >= 0})
            if n_found == 0:
                raise ValueError("HDBSCAN assigned every point to noise")
        except Exception:
            try:
                from sklearn.cluster import KMeans

                k = max(2, min(12, n_samples // 50)) if n_samples >= 100 else 2
                cluster_ids = np.asarray(
                    KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(coords),
                    dtype=int,
                )
                cluster_method = "KMeans"
            except Exception:
                logger.debug("Clustering failed; all points in one cluster", exc_info=True)

        def _clabel(c: int) -> str:
            return "Noise" if c < 0 else f"Cluster {c + 1}"

        labels_arr = np.array([_clabel(int(c)) for c in cluster_ids])
        n_clusters = len({int(c) for c in cluster_ids if c >= 0})

        # --- Plot --------------------------------------------------------------
        _emit(5, "Rendering plot…")
        out_dir = project_root / "derived" / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path: Path | None = out_dir / "unsupervised_umap.png"
        out_svg_path = out_dir / "unsupervised_umap.svg"
        try:
            import matplotlib.pyplot as plt

            classes = sorted(
                {str(c) for c in labels_arr},
                key=lambda s: (s == "Noise", int(s.split()[-1]) if s.startswith("Cluster") else 0),
            )
            n_classes = len(classes)
            legend_col_width = 2.2
            legend_ncol = max(1, min(3, n_classes // 8 + 1))
            fig_w = max(9.0, 7.0 + legend_ncol * legend_col_width)
            fig_h = max(5.5, 4.5 + n_classes * 0.05)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            fig.subplots_adjust(right=1.0 - legend_ncol * legend_col_width / fig_w)

            cluster_classes = [c for c in classes if c != "Noise"]
            cmap = plt.get_cmap("tab10" if len(cluster_classes) <= 10 else "tab20", max(1, len(cluster_classes)))
            color_map: dict[str, Any] = {c: cmap(i) for i, c in enumerate(cluster_classes)}
            color_map["Noise"] = (0.6, 0.6, 0.6, 1.0)

            # Draw noise first (underneath), then clusters on top.
            for cls in ["Noise"] + cluster_classes:
                if cls not in classes:
                    continue
                mask = labels_arr == cls
                if not np.any(mask):
                    continue
                is_noise = cls == "Noise"
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=4 if is_noise else 7,
                    alpha=0.25 if is_noise else 0.6,
                    color=color_map[cls],
                    label=f"{cls} ({int(np.sum(mask))})",
                    rasterized=True,
                )

            # Annotate each cluster at its density peak.
            import math
            _angles = [45, -45, 135, -135, 0, 90, 180, -90]
            annot_fs = max(5, 9 - n_classes // 5)
            x_range = float(coords[:, 0].max() - coords[:, 0].min()) or 1.0
            y_range = float(coords[:, 1].max() - coords[:, 1].min()) or 1.0
            for i, cls in enumerate(cluster_classes):
                mask = labels_arr == cls
                if not np.any(mask):
                    continue
                cx = float(np.median(coords[mask, 0]))
                cy = float(np.median(coords[mask, 1]))
                angle = math.radians(_angles[i % len(_angles)])
                ax.annotate(
                    cls, xy=(cx, cy),
                    xytext=(cx + x_range * 0.06 * math.cos(angle), cy + y_range * 0.08 * math.sin(angle)),
                    fontsize=annot_fs, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_map[cls], alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=color_map[cls], lw=1.0, shrinkA=0, shrinkB=4),
                )

            ax.set_title(
                f"Unsupervised Embedding ({method}) — "
                f"{n_clusters} clusters ({cluster_method})  |  {n_samples} segments",
                fontsize=9, wrap=True,
            )
            ax.set_xlabel(f"{method} 1")
            ax.set_ylabel(f"{method} 2")
            legend_fs = max(5, 9 - n_classes // 8)
            ax.legend(
                loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                frameon=True, framealpha=0.7, fontsize=legend_fs, markerscale=2,
                ncol=legend_ncol,
            )
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            fig.savefig(out_svg_path, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            logger.exception("Failed generating unsupervised UMAP plot")
            out_path = None

        # --- Persist coordinates for interactive 'Select from UMAP' ------------
        coord_path = out_dir / "unsupervised_umap_coordinates.parquet"
        try:
            coord_df = work[["segment_id"]].copy()
            for _mc in ("session_id", "animal_id"):
                coord_df[_mc] = work[_mc].astype(str).values if _mc in work.columns else ""
            for _mc in ("start_frame", "end_frame"):
                coord_df[_mc] = work[_mc].astype(int).values if _mc in work.columns else 0
            coord_df["umap_x"] = coords[:, 0]
            coord_df["umap_y"] = coords[:, 1]
            coord_df["behavior_label"] = labels_arr
            coord_df["is_labeled"] = False
            coord_df.to_parquet(coord_path, index=False)
        except Exception:
            logger.debug("Failed to save unsupervised UMAP coordinates", exc_info=True)
            coord_path = None

        result = {
            "out_path": str(out_path) if out_path else None,
            "coord_path": str(coord_path) if coord_path else None,
            "n_segments": int(n_samples),
            "n_clusters": int(n_clusters),
            "method": method,
            "cluster_method": cluster_method,
        }
        write_json(out_dir / "unsupervised_umap_report.json", result)
        return result
