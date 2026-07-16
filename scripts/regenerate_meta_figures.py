"""Re-render the meta-analysis figures + write Prism tables from an export folder.

The validation run that produced the manuscript exports takes hours and needs the
original ABEL projects.  Every figure here, though, is a pure function of a CSV that
the run already wrote — so this rebuilds them from the exported CSVs alone, using the
fixed plot functions in ``abel.validation.{plots,video_value,benchmark}``.

    python scripts/regenerate_meta_figures.py "J:/.../Meta analysis metric exports"

Outputs land in a ``regenerated/`` subfolder (figures + ``prism/`` tables); the
original exports are left untouched.

One thing cannot be rebuilt: the **per-seed** F1 values.  The old exporters wrote
only the mean + CI half-width, so the paired tests behind the asterisks are not
reproducible from these CSVs.  The source fix (per-seed columns) applies to the next
validation run; the tables written here fall back to means and say so.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abel.validation import benchmark, plots, prism, video_value  # noqa: E402
from abel.validation.analyses.behaviorscape import (  # noqa: E402
    MODALITY_ORDER,
    BehaviorscapeData,
    DistinctivenessStats,
)


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _no_error(row) -> bool:
    """True when the row's ``error`` cell is blank.

    An empty CSV cell reads back as NaN, and ``str(nan)`` is the *non-empty* string
    ``"nan"`` — so a naive truthiness check silently discards every good row.
    """
    val = row.get("error", "")
    return pd.isna(val) or not str(val).strip()


# ── Shims: rebuild the objects the plot functions expect, from the CSVs ──────


def _gen_results(df: pd.DataFrame) -> list:
    return [
        SimpleNamespace(
            project_id=str(r["project"]), behavior_name=str(r["behavior"]),
            kappa_mean=float(r["cohen_kappa"]), f1_mean=float(r["f1"]),
            human_ceiling_kappa=float(r.get("human_ceiling_kappa", np.nan)),
        )
        for _, r in df.iterrows()
    ]


def _ablation_results(df: pd.DataFrame) -> dict[str, list]:
    """One list of AblationResult-alikes per clip budget."""
    out: dict[str, list] = {}
    for budget, bgrp in df.groupby("clip_budget", sort=False):
        results = []
        for (proj, beh), grp in bgrp.groupby(["project", "behavior"], sort=False):
            order, gain, gain_ci, labels, f1_means, sig = [], {}, {}, {}, {}, {}
            for _, r in grp.iterrows():
                cfg = str(r["config"])
                labels[cfg] = str(r["label"])
                f1_means[cfg] = float(r["f1_mean"])
                if cfg == "baseline_none":
                    continue
                order.append(cfg)
                gain[cfg] = float(r["gain_over_baseline"])
                gain_ci[cfg] = float(r["gain_ci95"])
                # The CSV already carries the verdict; trust it rather than
                # re-deriving it without the seeds.
                sig[cfg] = str(r.get("significant", "")).strip().lower() == "true"
            results.append(SimpleNamespace(
                project_id=str(proj), behavior_name=str(beh), order=order,
                gain=gain, gain_ci=gain_ci, labels=labels, f1_means=f1_means,
                is_significant=lambda n, _s=sig: bool(_s.get(n, False)),
            ))
        out[str(budget)] = results
    return out


def _video_results(df: pd.DataFrame) -> list:
    return [
        SimpleNamespace(
            project_id=str(r["project_id"]), behavior_name=str(r["behavior_name"]),
            f1_no_video=float(r["f1_no_video"]), f1_with_video=float(r["f1_with_video"]),
            gain=float(r["gain"]), gain_ci95=float(r["gain_ci95"]),
            significant=str(r["significant"]).strip().lower() == "true",
            f1_no_video_seeds=[], f1_with_video_seeds=[], error="",
        )
        for _, r in df.iterrows()
        if _no_error(r)
    ]


def _bench_results(df: pd.DataFrame) -> list:
    return [
        SimpleNamespace(
            project_id=str(r["project_id"]), stage=str(r["stage"]),
            detail=str(r.get("detail", "") or ""), seconds=float(r["seconds"]),
            faster_than_realtime=float(r.get("faster_than_realtime", np.nan)),
            error="",
        )
        for _, r in df.iterrows()
        if _no_error(r)
    ]


def _behaviorscape(imp: pd.DataFrame, dist: pd.DataFrame | None):
    """Rebuild BehaviorscapeData (+ stats) from the exported long importance CSV."""
    matrix = imp.pivot_table(index="feature", columns="behavior", values="importance",
                             aggfunc="mean").fillna(0.0)
    modality = dict(zip(imp["feature"].astype(str), imp["modality"].astype(str)))
    data = BehaviorscapeData(
        matrix=matrix, modality=modality, sources=[], pooled_members={},
        threshold=0.01, normalize="fraction",
        n_features_total=3844, n_features_kept=int(len(matrix.index)),
    )
    stats = None
    if dist is not None and not dist.empty:
        behaviors = dist["behavior"].astype(str).tolist()
        stats = DistinctivenessStats(
            behaviors=behaviors,
            distinctiveness=dict(zip(behaviors, dist["distinctiveness_cosine"].astype(float))),
            err=dict(zip(behaviors, dist["se"].astype(float))),
            n_replicates=dict(zip(behaviors, dist["n_replicates"].astype(int))),
            dominant_modality=dict(zip(behaviors, dist["dominant_modality"].astype(str))),
            mean_distinctiveness=float(dist["distinctiveness_cosine"].astype(float).mean()),
            # Carried over from the run that produced the export (the PERMANOVA needs
            # the per-project source vectors, which the pooled CSV does not retain).
            permanova={"pseudo_F": 1.6, "R2": 0.23, "p": 0.001, "n_groups": 4,
                       "n_samples": len(behaviors), "n_perm": 999},
        )
    return data, stats


# ── Main ────────────────────────────────────────────────────────────────────


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    root = Path(argv[1])
    bscape = root / "behaviorscaped"
    out = root / "regenerated"
    out.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    gen = _read(root / "agreement_generalization.csv")
    if gen is not None:
        plots.human_ceiling_plot(_gen_results(gen), out / "model_vs_human_kappa.png")
        made.append("model_vs_human_kappa.png")

    abl = _read(root / "ablation_results.csv")
    if abl is not None:
        order = ["n50", "n100", "n250", "all"]
        groups = _ablation_results(abl)
        for i, budget in enumerate([b for b in order if b in groups]):
            title = "full data" if budget == "all" else f"{budget[1:]} clips"
            name = f"feature_impact__{i}_{budget}.png"
            plots.ablation_impact_plot(groups[budget], out / name, budget_title=title)
            made.append(name)

    vv = _read(bscape / "video_value.csv")
    if vv is not None:
        video_value.plot_video_value(_video_results(vv), out / "video_value.png")
        made.append("video_value.png")

    bench = _read(bscape / "throughput benchmark.csv")
    if bench is not None:
        benchmark.plot_benchmark(_bench_results(bench), out / "throughput_benchmark.png")
        made.append("throughput_benchmark.png")

    imp = _read(bscape / "behaviorscape_importance.csv")
    if imp is not None:
        data, stats = _behaviorscape(imp, _read(bscape / "behaviorscape_distinctiveness.csv"))
        for name, fn in (
            ("behaviorscape_heatmap.png", plots.behaviorscape_heatmap),
            ("behaviorscape_modality_bars.png", plots.behaviorscape_modality_bars),
            ("behaviorscape_clusters.png", plots.behaviorscape_clusters),
            ("behaviorscape_network.png", plots.behaviorscape_network),
        ):
            fn(data, save_path=out / name)
            made.append(name)
        plots.behaviorscape_distinctiveness(
            data, save_path=out / "behaviorscape_distinctiveness.png", stats=stats)
        made.append("behaviorscape_distinctiveness.png")

    plots.close_all()

    written = prism.write_all(out, gen_df=gen, video_df=vv, ablation_df=abl,
                              bench_df=bench)
    print(f"Figures  -> {out}")
    for m in made:
        print(f"    {m}")
    print(f"Prism    -> {out / 'prism'}")
    for p in written:
        print(f"    {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
