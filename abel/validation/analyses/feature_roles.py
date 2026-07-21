"""Feature-role clustering: do ABEL's extracted features play distinct roles?

The behaviorscape importance profile says, per behavior, *which* feature modality the
model leans on — pose geometry, kinematics, environment/ROI context, or video motion.
If features were interchangeable every behavior would lean the same way. They do not:
Groom is pose-driven, Explore is kinematics-driven, Climb/Sniff are context-driven,
Approach is video-driven. This module makes that quantitative:

1. Cluster behaviors by their modality-reliance profile (hierarchical, Ward linkage,
   k=4 by default — one group per dominant modality) → a dendrogram.
2. For each cluster, name its dominant modality and measure how much that reliance
   *buys*: the F1 improvement over a pose-only baseline the corresponding features
   provide (ablation ΔF1). Context and video are added features and can lift F1;
   pose and kinematics are already IN the baseline, so their lift is 0 by construction
   — which is the point (those behaviors need no extra features).
3. A one-sample t-test asks whether each cluster's lift is real; a Kruskal-Wallis
   across clusters shows the lift is not uniform — the feature roles are distinct.

The bar table (:func:`dominant_modality_improvement_bars`) is Prism-ready: one row per
cluster, y = mean improvement over pose-only, with a 95% CI and p.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Which ablation enhancement realises each modality's over-pose lift. Pose geometry and
# kinematics are part of the pose-only baseline, so they have no over-pose enhancement.
MODALITY_TO_ENHANCEMENT: dict[str, str] = {
    "Context (ROI / target)": "+ Environment / ROI context",
    "Video (flow / appearance)": "+ Video features",
}
_BASELINE_LABEL = "Baseline (pose only)"
DEFAULT_K = 4  # one cluster per dominant modality (pose / kinematics / context / video)
# Stable display order for the modality groups (added-feature modalities first).
MODALITY_ORDER = ("Context (ROI / target)", "Video (flow / appearance)",
                  "Kinematics", "Pose geometry")


def modality_groups(matrix: pd.DataFrame) -> np.ndarray:
    """Assign each behavior to its dominant modality → one group per modality present.

    This is the k=4-by-modality grouping the manuscript bar uses: exactly one bar per
    feature type. (The Ward dendrogram from :func:`cluster_behaviors` is the relatedness
    visual alongside it.)
    """
    if matrix.empty:
        return np.array([], dtype=int)
    dom = matrix.idxmax(axis=1)
    present = ([m for m in MODALITY_ORDER if m in set(dom)]
               + [m for m in dict.fromkeys(dom) if m not in MODALITY_ORDER])
    code = {m: i + 1 for i, m in enumerate(present)}
    return np.array([code[dom[b]] for b in matrix.index], dtype=int)


def modality_reliance_matrix(shares_df: pd.DataFrame) -> pd.DataFrame:
    """Behaviors × modality reliance shares (rows sum to 1), from the behaviorscape
    ``modality_shares`` long table (columns: behavior, modality_label, percent)."""
    if shares_df is None or shares_df.empty:
        return pd.DataFrame()
    W = shares_df.pivot_table(index="behavior", columns="modality_label",
                              values="percent", aggfunc="first").fillna(0.0)
    W.columns.name = None
    row = W.sum(axis=1).replace(0, np.nan)
    return W.div(row, axis=0).fillna(0.0)


def ablation_gain_by_behavior(abl_df: pd.DataFrame, *, budget: str = "all",
                              by: str = "assay") -> pd.DataFrame:
    """Enhancement ΔF1 per behavior for one clip budget.

    ``by="assay"`` indexes rows as ``"project · behavior"`` (the production, assay-scoped
    key); ``by="behavior"`` collapses across assays to the bare behavior name (mean),
    for joining against a pooled importance table.
    """
    df = abl_df[(abl_df["clip_budget"].astype(str) == budget)
                & (abl_df["label"] != _BASELINE_LABEL)].copy()
    if df.empty:
        return pd.DataFrame()
    if by == "assay":
        df["__row"] = df["project"].astype(str) + " · " + df["behavior"].astype(str)
        idx, agg = "__row", "first"
    else:
        df["__row"], idx, agg = df["behavior"].astype(str), "__row", "mean"
    piv = df.pivot_table(index=idx, columns="label", values="gain_over_baseline",
                         aggfunc=agg)
    piv.columns.name = None
    return piv


def cluster_behaviors(matrix: pd.DataFrame, *, k: int = DEFAULT_K, seed: int = 0) -> dict:
    """Ward-cluster the reliance profiles into ``k`` groups. Returns linkage, leaf
    order (for a dendrogram / ordered heatmap), integer labels and ``k``."""
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage  # noqa: PLC0415
    from scipy.spatial.distance import pdist  # noqa: PLC0415

    X = matrix.to_numpy(dtype=float)
    n = X.shape[0]
    if n < 3:
        raise ValueError(f"Need ≥3 behaviors to cluster, got {n}")
    k = int(max(2, min(k, n - 1)))
    Z = linkage(pdist(X, metric="euclidean"), method="ward")
    labels = fcluster(Z, k, criterion="maxclust")
    return {"linkage": Z, "order": list(leaves_list(Z)), "labels": labels,
            "k": k, "index": list(matrix.index)}


def _ci95(vals: np.ndarray) -> float:
    v = vals[np.isfinite(vals)]
    if v.size < 2:
        return 0.0
    return float(1.96 * np.std(v, ddof=1) / np.sqrt(v.size))


def _behavior_improvement(matrix: pd.DataFrame, gain_df: pd.DataFrame,
                          dominant: dict[str, str]) -> pd.Series:
    """Per behavior: the over-pose ΔF1 of the enhancement realising its cluster's
    dominant modality (0 when that modality is pose/kinematics — already in baseline)."""
    imp = {}
    for beh in matrix.index:
        enh = MODALITY_TO_ENHANCEMENT.get(dominant.get(beh, ""))
        if enh is None:
            imp[beh] = 0.0                                   # pose / kinematics baseline
        elif beh in gain_df.index and enh in gain_df.columns:
            imp[beh] = float(gain_df.loc[beh, enh])
        else:
            imp[beh] = np.nan
    return pd.Series(imp)


def dominant_modality_improvement_bars(matrix: pd.DataFrame, labels: np.ndarray,
                                       gain_df: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster: dominant modality + mean improvement over pose-only.

    Improvement is the ablation ΔF1 of the enhancement that realises the cluster's
    dominant modality (0 for pose/kinematics-dominant clusters). Carries a 95% CI
    across the cluster's behaviors and a one-sample t-test p vs 0. Ordered by
    descending improvement so the most feature-dependent cluster reads first.
    """
    from scipy import stats  # noqa: PLC0415

    lab = np.asarray(labels)
    # Each cluster's dominant modality = the modality with the largest mean share.
    dom_by_beh: dict[str, str] = {}
    cluster_dom: dict[int, str] = {}
    for cl in sorted(set(lab)):
        sub = matrix.iloc[lab == cl]
        dom = str(sub.mean(axis=0).idxmax())
        cluster_dom[cl] = dom
        for beh in sub.index:
            dom_by_beh[beh] = dom

    imp = _behavior_improvement(matrix, gain_df, dom_by_beh)
    rows = []
    for cl in sorted(set(lab)):
        behs = matrix.index[lab == cl]
        vals = imp[behs].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2 and np.ptp(vals) > 0:
            t, p = stats.ttest_1samp(vals, 0.0)
        else:
            t, p = float("nan"), float("nan")
        rows.append({
            "cluster": int(cl),
            "dominant_modality": cluster_dom[cl],
            "n_behaviors": int(len(behs)),
            "mean_improvement_over_pose": float(np.mean(vals)) if vals.size else float("nan"),
            "ci95": _ci95(vals),
            "t_stat": float(t),
            "p_value": float(p),
        })
    out = pd.DataFrame(rows).sort_values(
        "mean_improvement_over_pose", ascending=False, ignore_index=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def cluster_membership(matrix: pd.DataFrame, labels: np.ndarray,
                       gain_df: pd.DataFrame) -> pd.DataFrame:
    """Long table: each behavior's cluster, its own dominant modality, the modality
    reliance shares and the over-pose improvement attributed to it."""
    lab = np.asarray(labels)
    own_dom = {b: str(matrix.loc[b].idxmax()) for b in matrix.index}
    cluster_dom = {int(cl): str(matrix.iloc[lab == cl].mean(axis=0).idxmax())
                   for cl in sorted(set(lab))}
    imp = _behavior_improvement(matrix, gain_df,
                                {b: cluster_dom[int(c)] for b, c in zip(matrix.index, lab)})
    out = matrix.copy()
    out.insert(0, "cluster", lab)
    out.insert(1, "own_dominant_modality", [own_dom[b] for b in matrix.index])
    out.insert(2, "improvement_over_pose", [imp[b] for b in matrix.index])
    out = out.reset_index().rename(columns={"index": "behavior", "behavior": "behavior"})
    return out.sort_values(["cluster", "behavior"], ignore_index=True)


def kruskal_across_clusters(matrix: pd.DataFrame, labels: np.ndarray,
                            gain_df: pd.DataFrame) -> dict:
    """Kruskal-Wallis: does the over-pose improvement differ across the clusters?"""
    from scipy import stats  # noqa: PLC0415

    lab = np.asarray(labels)
    cluster_dom = {int(cl): str(matrix.iloc[lab == cl].mean(axis=0).idxmax())
                   for cl in sorted(set(lab))}
    imp = _behavior_improvement(matrix, gain_df,
                                {b: cluster_dom[int(c)] for b, c in zip(matrix.index, lab)})
    groups = [imp[matrix.index[lab == cl]].to_numpy(dtype=float) for cl in sorted(set(lab))]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if g.size >= 1]
    if len(groups) < 2:
        return {}
    try:
        H, p = stats.kruskal(*groups)
        return {"H": float(H), "p_value": float(p), "n_groups": len(groups)}
    except Exception:  # noqa: BLE001 — degenerate (all-equal) groups
        return {}


def plot_dendrogram(cl: dict, save_path: Path) -> Path | None:
    """Draw the behavior dendrogram (Prism can't; this is a matplotlib figure)."""
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from scipy.cluster.hierarchy import dendrogram  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    Z, index, k = cl["linkage"], cl["index"], cl["k"]
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.24 * len(index) + 1.5)))
    dendrogram(Z, labels=[str(b)[:34] for b in index], orientation="right", ax=ax,
               color_threshold=Z[-(k - 1), 2] if k > 1 else 0.0, leaf_font_size=7.5)
    ax.set_title(f"Behaviors clustered by feature-modality reliance (Ward, k={k})",
                 fontsize=11, loc="left")
    ax.set_xlabel("Ward linkage distance", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def run_feature_roles(shares_df: pd.DataFrame, abl_df: pd.DataFrame, out_dir: str | Path,
                      *, k: int = DEFAULT_K, scope: str = "assay") -> list[Path]:
    """Cluster behaviors by modality reliance, then write the dendrogram + Prism-ready
    bar/membership CSVs into ``out_dir``. ``scope`` matches the importance table's key:
    ``"assay"`` (production, ``project · behavior``) or ``"behavior"`` (pooled names)."""
    out_dir = Path(out_dir)
    matrix = modality_reliance_matrix(shares_df)
    gain = ablation_gain_by_behavior(abl_df, by=scope)
    if matrix.empty or gain.empty or matrix.shape[0] < 3:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    # Bars/stats group by dominant modality (one bar per feature type); the Ward
    # linkage drives only the dendrogram, the relatedness visual beside the bars.
    groups = modality_groups(matrix)
    cl = cluster_behaviors(matrix, k=k)
    bars = dominant_modality_improvement_bars(matrix, groups, gain)
    memb = cluster_membership(matrix, groups, gain)
    kw = kruskal_across_clusters(matrix, groups, gain)
    if kw:
        bars["kruskal_H_across_clusters"] = kw["H"]
        bars["kruskal_p_across_clusters"] = kw["p_value"]

    written: list[Path] = []
    bars.round(6).to_csv(out_dir / "feature_role_cluster_bars.csv",
                         index=False, encoding="utf-8")
    memb.round(6).to_csv(out_dir / "feature_role_clusters.csv",
                         index=False, encoding="utf-8")
    written += [out_dir / "feature_role_cluster_bars.csv",
                out_dir / "feature_role_clusters.csv"]
    fig = plot_dendrogram(cl, out_dir / "feature_role_dendrogram.png")
    if fig is not None:
        written.append(fig)
    return written
