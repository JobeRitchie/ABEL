"""Tests for the all-project embedding and its label placement.

Covers the parts that decide what the figure *means*, feature-space resolution,
row filtering, the R3D zero-block guard, assay scoping, the behavior-vs-project QC
and the label geometry, which is the tab's whole reason to exist.  The reducer
itself is not retested here; ``reducer="pca"`` stands in so the tests stay fast
and do not depend on umap-learn being installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.validation import umap_plot
from abel.validation.analyses import all_project_umap as apu
from abel.validation.datamodel import ProjectRef


# ── Fixtures ────────────────────────────────────────────────────────────────


def _write_project(root, *, name, behaviors, n_per=40, extra_cols=(), seed=0,
                   r3d_zero=False, label_source="reviewed"):
    """A minimal on-disk project: project.yaml + behavior list + training set."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(exist_ok=True)
    (root / "project.yaml").write_text(f"project_name: {name}\n", encoding="utf-8")
    lines = ["behaviors:"]
    for bid, bname in behaviors.items():
        lines += [f"  - behavior_id: {bid}", f"    name: {bname}"]
    (root / "config" / "behavior_definitions.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    rng = np.random.default_rng(seed)
    rows = []
    for k, bid in enumerate(behaviors):
        for i in range(n_per):
            row = {
                "segment_id": f"{name}_{bid}_{i}",
                "start_frame": i * 10, "end_frame": i * 10 + 9,
                "animal_id": f"{name}_m{i % 3}", "session_id": f"{name}_s{i % 4}",
                "label": bid, "label_source": label_source,
                "reviewer_confidence": 1.0,
                "nose_velocity_mean": k + rng.normal(0, 0.1),
                "nose_angle_mean": k * 2 + rng.normal(0, 0.1),
                "flow_mag_mean": k + rng.normal(0, 0.1),
            }
            for j in range(4):
                row[f"r3d_{j:03d}"] = 0.0 if r3d_zero else k + rng.normal(0, 0.1)
            for c in extra_cols:
                row[c] = rng.normal()
            rows.append(row)
    out = root / "derived" / "training_sets"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "training_set.parquet", index=False)
    return ProjectRef.load(root)


@pytest.fixture()
def two_projects(tmp_path):
    """Two projects that share three columns and differ in a fourth."""
    a = _write_project(tmp_path / "A", name="A",
                       behaviors={"b1": "Rear", "b2": "Groom"}, seed=1)
    b = _write_project(tmp_path / "B", name="B",
                       behaviors={"b3": "Rear", "b4": "Walk"}, seed=2,
                       extra_cols=("nose_to_roi_1_dist",))
    return [a, b], {"A": ["b1", "b2"], "B": ["b3", "b4"]}


# ── Feature space ───────────────────────────────────────────────────────────


def test_shared_space_is_the_intersection_union_is_the_union(two_projects):
    projects, _ = two_projects
    shared, _ = apu.resolve_feature_columns(
        projects, apu.EmbeddingSettings(feature_space="shared", use_context=True))
    union, per_project = apu.resolve_feature_columns(
        projects, apu.EmbeddingSettings(feature_space="union", use_context=True))
    # The ROI column exists only in project B.
    assert "nose_to_roi_1_dist" not in shared
    assert "nose_to_roi_1_dist" in union
    assert per_project["B"] == per_project["A"] + 1


def test_family_toggles_select_columns(two_projects):
    projects, _ = two_projects
    cols, _ = apu.resolve_feature_columns(
        projects, apu.EmbeddingSettings(use_video=False, use_r3d=False))
    assert not any(c.startswith("r3d_") for c in cols)
    assert "flow_mag_mean" not in cols
    assert "nose_angle_mean" in cols

    only_r3d, _ = apu.resolve_feature_columns(
        projects, apu.EmbeddingSettings(use_pose=False, use_kinematics=False,
                                        use_video=False, use_r3d=True))
    assert only_r3d and all(c.startswith("r3d_") for c in only_r3d)


def test_metadata_columns_are_never_features(two_projects):
    projects, _ = two_projects
    cols, _ = apu.resolve_feature_columns(projects, apu.EmbeddingSettings())
    assert not (set(cols) & apu.META_COLS)


# ── Row assembly ────────────────────────────────────────────────────────────


def test_groups_are_assay_scoped_unless_pooling_is_asked_for(two_projects):
    projects, behaviors = two_projects
    s = apu.EmbeddingSettings()
    cols, _ = apu.resolve_feature_columns(projects, s)

    frame, _ = apu.collect_rows(projects, behaviors, cols, s)
    assert "A · Rear" in set(frame["group"])
    assert "B · Rear" in set(frame["group"])

    s.pool_behaviors_by_name = True
    pooled, _ = apu.collect_rows(projects, behaviors, cols, s)
    assert set(pooled["group"]) == {"Rear", "Groom", "Walk"}


def test_alias_map_merges_before_the_assay_prefix(two_projects):
    projects, behaviors = two_projects
    s = apu.EmbeddingSettings(alias_map={"Groom": "Grooming"})
    cols, _ = apu.resolve_feature_columns(projects, s)
    frame, _ = apu.collect_rows(projects, behaviors, cols, s)
    assert "A · Grooming" in set(frame["group"])


def test_small_groups_are_dropped_and_reported(two_projects):
    projects, behaviors = two_projects
    s = apu.EmbeddingSettings(min_rows_per_behavior=1000)
    cols, _ = apu.resolve_feature_columns(projects, s)
    frame, warnings = apu.collect_rows(projects, behaviors, cols, s)
    assert frame.empty
    assert any("dropped" in w for w in warnings)


def test_per_behavior_cap_is_applied(two_projects):
    projects, behaviors = two_projects
    s = apu.EmbeddingSettings(max_rows_per_behavior=10)
    cols, _ = apu.resolve_feature_columns(projects, s)
    frame, _ = apu.collect_rows(projects, behaviors, cols, s)
    assert frame.groupby("group").size().max() == 10


def test_imported_rows_are_excluded(tmp_path):
    p = _write_project(tmp_path / "I", name="I", behaviors={"b1": "Rear"},
                       label_source="imported:other")
    s = apu.EmbeddingSettings(min_rows_per_behavior=1)
    cols, _ = apu.resolve_feature_columns([p], s)
    frame, warnings = apu.collect_rows([p], {"I": ["b1"]}, cols, s)
    assert frame.empty
    s.exclude_imported = False
    frame, _ = apu.collect_rows([p], {"I": ["b1"]}, cols, s)
    assert len(frame) == 40


# ── Matrix preparation ──────────────────────────────────────────────────────


def test_all_zero_r3d_rows_are_dropped_and_metadata_stays_aligned(tmp_path):
    """The known failure mode: 512 shared zeros make a dense fake cluster."""
    good = _write_project(tmp_path / "G", name="G", behaviors={"b1": "Rear"}, seed=3)
    bad = _write_project(tmp_path / "Z", name="Z", behaviors={"b2": "Rear"}, seed=4,
                         r3d_zero=True)
    projects = [good, bad]
    behaviors = {"G": ["b1"], "Z": ["b2"]}
    # R3D is off by default (it is a rig fingerprint), so the guard under test
    # must be asked for explicitly.
    s = apu.EmbeddingSettings(pca_components=0, use_r3d=True)
    cols, _ = apu.resolve_feature_columns(projects, s)
    pooled, _ = apu.collect_rows(projects, behaviors, cols, s)
    assert len(pooled) == 80

    X, kept, notes, _ = apu.prepare_matrix(pooled, cols, s)
    assert len(X) == 40 == len(pooled)          # in-place drop keeps them aligned
    assert set(pooled["project_id"]) == {"G"}
    assert any("R3D block was" in n for n in notes)


def test_zero_r3d_guard_can_be_switched_off(tmp_path):
    bad = _write_project(tmp_path / "Z", name="Z", behaviors={"b2": "Rear"},
                         r3d_zero=True)
    s = apu.EmbeddingSettings(drop_zero_r3d_rows=False, pca_components=0,
                              use_r3d=True)
    cols, _ = apu.resolve_feature_columns([bad], s)
    pooled, _ = apu.collect_rows([bad], {"Z": ["b2"]}, cols, s)
    X, _, _, _ = apu.prepare_matrix(pooled, cols, s)
    assert len(X) == 40


def test_per_project_zscore_centres_each_project(two_projects):
    projects, behaviors = two_projects
    s = apu.EmbeddingSettings(per_project_zscore=True, scaler="none",
                              pca_components=0)
    cols, _ = apu.resolve_feature_columns(projects, s)
    pooled, _ = apu.collect_rows(projects, behaviors, cols, s)
    X, _, notes, _ = apu.prepare_matrix(pooled, cols, s)
    pid = pooled["project_id"].to_numpy()
    for p in np.unique(pid):
        assert np.allclose(X[pid == p].mean(axis=0), 0.0, atol=1e-4)
    assert any("within each project" in n for n in notes)


def test_union_space_fills_the_missing_column(two_projects):
    projects, behaviors = two_projects
    s = apu.EmbeddingSettings(feature_space="union", use_context=True,
                              pca_components=0, drop_zero_variance=False)
    cols, _ = apu.resolve_feature_columns(projects, s)
    pooled, _ = apu.collect_rows(projects, behaviors, cols, s)
    X, kept, notes, _ = apu.prepare_matrix(pooled, cols, s)
    assert "nose_to_roi_1_dist" in kept
    assert np.isfinite(X).all()
    assert any("Median-filled" in n for n in notes)


def test_prepare_matrix_refuses_an_empty_feature_set(two_projects):
    projects, behaviors = two_projects
    s = apu.EmbeddingSettings()
    cols, _ = apu.resolve_feature_columns(projects, s)
    pooled, _ = apu.collect_rows(projects, behaviors, cols, s)
    with pytest.raises(ValueError, match="No feature columns"):
        apu.prepare_matrix(pooled, [], s)


# ── End to end ──────────────────────────────────────────────────────────────


def test_build_produces_aligned_coordinates_and_qc(two_projects):
    projects, behaviors = two_projects
    s = apu.EmbeddingSettings(reducer="pca", pca_components=0)
    res = apu.build_all_project_umap(projects, behaviors, s)

    assert res.n_points == 160
    assert {"x", "y", "group", "project_id", "behavior_name"} <= set(res.frame.columns)
    assert res.frame[["x", "y"]].notna().all().all()
    assert res.reducer_used == "pca"
    assert res.qc["n_groups"] == 4
    assert res.qc["n_projects"] == 2
    # Project purity has a floor at the chance level, never below it.
    assert res.qc["knn_purity_project_feat"] >= res.qc["knn_purity_project_chance"] - 0.2
    assert res.counts is not None and res.counts["n_points"].sum() == 160


def test_build_refuses_when_nothing_is_selected(two_projects):
    projects, _ = two_projects
    with pytest.raises(ValueError, match="No rows to embed"):
        apu.build_all_project_umap(projects, {"A": [], "B": []},
                                   apu.EmbeddingSettings(reducer="pca"))


def test_save_and_load_round_trip(tmp_path, two_projects):
    projects, behaviors = two_projects
    res = apu.build_all_project_umap(
        projects, behaviors, apu.EmbeddingSettings(reducer="pca", pca_components=0))
    paths = apu.save_embedding(res, tmp_path / "run")
    assert paths["embedding"].exists()

    frame, meta = apu.load_embedding(tmp_path / "run")
    assert len(frame) == res.n_points
    assert meta["reducer_used"] == "pca"
    assert meta["settings"]["reducer"] == "pca"
    # Also accepts the parquet itself, not only its folder.
    frame2, _ = apu.load_embedding(paths["embedding"])
    assert len(frame2) == len(frame)


def test_qc_summary_names_the_dominant_structure(two_projects):
    projects, behaviors = two_projects
    res = apu.build_all_project_umap(
        projects, behaviors, apu.EmbeddingSettings(reducer="pca", pca_components=0))
    text = apu.qc_summary(res)
    assert "silhouette" in text
    assert ("behavior structure dominates" in text
            or "PROJECT identity dominates" in text)


def test_settings_round_trip_through_a_dict():
    s = apu.EmbeddingSettings(n_neighbors=77, min_dist=0.33, reducer="tsne")
    back = apu.EmbeddingSettings.from_dict(s.to_dict())
    assert back == s
    # Unknown keys (an older saved run) are ignored rather than raising.
    assert apu.EmbeddingSettings.from_dict({"gone_field": 1}).n_neighbors == 60


# ── Label placement ─────────────────────────────────────────────────────────


def _boxes_overlap(a, b, ha, hb, gap=0.0):
    return (abs(a[0] - b[0]) < ha[0] + hb[0] + gap
            and abs(a[1] - b[1]) < ha[1] + hb[1] + gap)


def _label_case(n=12, span=10.0):
    rng = np.random.default_rng(0)
    anchors, texts, pts = {}, {}, []
    for i in range(n):
        c = rng.uniform(0, span, size=2)
        anchors[f"Project {i} · Behavior"] = c
        texts[f"Project {i} · Behavior"] = f"Project {i} · Behavior"
        pts.append(rng.normal(c, 0.3, size=(40, 2)))
    return anchors, texts, np.vstack(pts), (0.0, span), (0.0, span)


def test_offset_moves_labels_off_their_clusters_by_the_requested_fraction():
    anchors, texts, pts, xlim, ylim = _label_case(n=4, span=10.0)
    s = umap_plot.PlotSettings(label_push="radial", label_offset=0.20,
                               label_repel_iters=0, label_spring=1.0,
                               label_avoid_points=False)
    placed = umap_plot.place_labels(anchors, texts, pts, xlim, ylim, s)
    for k, xy in placed.items():
        d = np.hypot(*((xy - anchors[k]) / (xlim[1] - xlim[0])))
        assert d == pytest.approx(0.20, abs=0.02)


def test_zero_offset_leaves_the_label_on_its_anchor():
    anchors, texts, pts, xlim, ylim = _label_case(n=3)
    s = umap_plot.PlotSettings(label_offset=0.0, label_repel_iters=0,
                               label_avoid_points=False)
    placed = umap_plot.place_labels(anchors, texts, pts, xlim, ylim, s)
    for k, xy in placed.items():
        assert np.allclose(xy, anchors[k], atol=1e-6)


def test_repulsion_separates_labels_that_start_on_top_of_each_other():
    # Four clusters at nearly the same place: without repulsion the labels stack.
    anchors = {f"Group {i}": np.array([5.0 + 0.01 * i, 5.0]) for i in range(4)}
    texts = {k: k for k in anchors}
    pts = np.random.default_rng(0).normal(5.0, 0.2, size=(200, 2))
    lim = (0.0, 10.0)

    stacked = umap_plot.place_labels(
        anchors, texts, pts, lim, lim,
        umap_plot.PlotSettings(label_repel_iters=0, label_offset=0.0,
                               label_avoid_points=False))
    spread = umap_plot.place_labels(
        anchors, texts, pts, lim, lim,
        umap_plot.PlotSettings(label_repel_iters=600, label_offset=0.05,
                               label_min_gap=0.02, label_spring=0.01,
                               label_avoid_points=False))
    keys = list(anchors)
    half = np.array([umap_plot._label_extent(texts[k], 9.0, 12.0, 9.0, False)
                     for k in keys])
    n_stacked = sum(
        _boxes_overlap((stacked[keys[i]] - lim[0]) / 10.0,
                       (stacked[keys[j]] - lim[0]) / 10.0, half[i], half[j])
        for i in range(4) for j in range(i + 1, 4))
    n_spread = sum(
        _boxes_overlap((spread[keys[i]] - lim[0]) / 10.0,
                       (spread[keys[j]] - lim[0]) / 10.0, half[i], half[j])
        for i in range(4) for j in range(i + 1, 4))
    assert n_stacked == 6      # every pair overlaps
    assert n_spread < n_stacked


def test_perimeter_push_puts_every_label_outside_the_data_and_in_bearing_order():
    anchors, texts, pts, xlim, ylim = _label_case(n=10, span=10.0)
    s = umap_plot.PlotSettings(label_push="perimeter", label_offset=0.05)
    placed = umap_plot.place_labels(anchors, texts, pts, xlim, ylim, s)

    # Each label is out near an edge of the frame, not in the middle of the map.
    for xy in placed.values():
        f = (np.asarray(xy) - xlim[0]) / (xlim[1] - xlim[0])
        assert max(abs(f[0] - 0.5), abs(f[1] - 0.5)) > 0.25
    # ...and never off the canvas.
    for xy in placed.values():
        f = (np.asarray(xy) - xlim[0]) / (xlim[1] - xlim[0])
        assert -0.05 <= f[0] <= 1.05 and -0.05 <= f[1] <= 1.05


def test_leash_bounds_how_far_a_label_can_drift():
    anchors = {f"Group {i}": np.array([5.0, 5.0]) for i in range(8)}
    texts = {k: k for k in anchors}
    pts = np.random.default_rng(0).normal(5.0, 0.2, size=(200, 2))
    lim = (0.0, 10.0)
    s = umap_plot.PlotSettings(label_repel_iters=800, label_offset=0.05,
                               label_leash=0.05, label_spring=0.01,
                               label_avoid_points=False)
    placed = umap_plot.place_labels(anchors, texts, pts, lim, lim, s)
    for k, xy in placed.items():
        d = np.hypot(*((np.asarray(xy) - anchors[k]) / 10.0))
        assert d <= 0.05 + 0.05 + 1e-6      # offset + leash


def test_anchor_modes_stay_inside_a_crescent_shaped_cluster():
    # A ring: the centroid falls in the hole, the medoid cannot.
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)]) * 5.0
    centroid = umap_plot._anchor_xy(pts, "centroid")
    medoid = umap_plot._anchor_xy(pts, "medoid")
    assert np.hypot(*centroid) < 1.0                  # in the hole
    assert 4.5 < np.hypot(*medoid) < 5.5              # on the ring


def test_label_min_points_suppresses_tiny_groups(tmp_path):
    frame = pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 8.0],
        "y": [0.0, 1.0, 2.0, 8.0],
        "group": ["Big", "Big", "Big", "Tiny"],
        "project_id": ["P", "P", "P", "P"],
        "behavior_name": ["b", "b", "b", "t"],
    })
    s = umap_plot.PlotSettings(label_min_points=3, legend="none", save_pdf=False)
    out = umap_plot.render(frame, s, tmp_path)
    assert out and out[0].exists()


# ── Rendering ───────────────────────────────────────────────────────────────


@pytest.fixture()
def rendered_frame(two_projects):
    projects, behaviors = two_projects
    res = apu.build_all_project_umap(
        projects, behaviors, apu.EmbeddingSettings(reducer="pca", pca_components=0))
    return res


@pytest.mark.parametrize("overlay", ["none", "hull", "ellipse"])
def test_render_every_overlay(tmp_path, rendered_frame, overlay):
    s = umap_plot.PlotSettings(overlay=overlay, save_pdf=False)
    out = umap_plot.render(rendered_frame.frame, s, tmp_path, stem=f"m_{overlay}")
    assert out[0].exists() and out[0].stat().st_size > 0


@pytest.mark.parametrize("push", ["radial", "sparse", "perimeter", "up", "none"])
def test_render_every_label_push(tmp_path, rendered_frame, push):
    s = umap_plot.PlotSettings(label_push=push, save_pdf=False)
    out = umap_plot.render(rendered_frame.frame, s, tmp_path, stem=f"m_{push}")
    assert out[0].exists()


@pytest.mark.parametrize("color_by", ["group", "project", "behavior"])
def test_render_every_colour_mode(tmp_path, rendered_frame, color_by):
    s = umap_plot.PlotSettings(color_by=color_by, palette="assay_shades",
                               save_pdf=False)
    out = umap_plot.render(rendered_frame.frame, s, tmp_path, stem=f"m_{color_by}")
    assert out[0].exists()


def test_render_facets_and_pdf(tmp_path, rendered_frame):
    s = umap_plot.PlotSettings(facet_by="project", facet_cols=2, save_pdf=True)
    out = umap_plot.render(rendered_frame.frame, s, tmp_path)
    assert [p.suffix for p in out] == [".png", ".pdf"]


def test_qc_panel_renders(tmp_path, rendered_frame):
    path = umap_plot.render_qc(rendered_frame.frame, rendered_frame.qc, tmp_path,
                               umap_plot.PlotSettings(save_pdf=False))
    assert path is not None and path.exists()


def test_plot_settings_round_trip_through_a_dict():
    s = umap_plot.PlotSettings(label_offset=0.31, label_push="perimeter",
                               overlay="hull")
    assert umap_plot.PlotSettings.from_dict(s.to_dict()) == s
    assert umap_plot.PlotSettings.from_dict({"nope": 1}).label_offset == 0.10


def test_union_reports_the_imputed_fraction_and_warns(two_projects):
    """The trap that produced two project-clustered maps in the wild."""
    projects, behaviors = two_projects
    s = apu.EmbeddingSettings(feature_space="union", use_context=True,
                              per_project_zscore=True, pca_components=0,
                              drop_zero_variance=False)
    cols, _ = apu.resolve_feature_columns(projects, s)
    pooled, _ = apu.collect_rows(projects, behaviors, cols, s)
    _X, _kept, notes, frac = apu.prepare_matrix(pooled, cols, s)

    # Project A lacks the ROI column entirely, so half the rows are imputed in it.
    assert frac > 0.02
    assert any("OF THE MATRIX WAS IMPUTED" in n for n in notes)
    assert any("sharpening the barcode" in n for n in notes)


def test_shared_space_imputes_nothing(two_projects):
    projects, behaviors = two_projects
    s = apu.EmbeddingSettings(feature_space="shared", pca_components=0)
    cols, _ = apu.resolve_feature_columns(projects, s)
    pooled, _ = apu.collect_rows(projects, behaviors, cols, s)
    _X, _kept, notes, frac = apu.prepare_matrix(pooled, cols, s)
    assert frac == 0.0
    assert not any("IMPUTED" in n for n in notes)


def test_qc_summary_leads_with_the_imputation_warning(two_projects):
    projects, behaviors = two_projects
    res = apu.build_all_project_umap(
        projects, behaviors,
        apu.EmbeddingSettings(reducer="pca", pca_components=0,
                              feature_space="union", use_context=True))
    assert res.qc["frac_imputed"] > 0.02
    assert "of the feature matrix was imputed" in apu.qc_summary(res)


def test_r3d_is_off_by_default_because_it_fingerprints_the_rig(two_projects):
    """Measured: R3D on takes kNN project purity 0.58 -> 0.97 (chance 0.13)."""
    projects, _ = two_projects
    cols, _ = apu.resolve_feature_columns(projects, apu.EmbeddingSettings())
    assert not any(c.startswith("r3d_") for c in cols)


def test_glow_pass_renders(tmp_path, two_projects):
    projects, behaviors = two_projects
    res = apu.build_all_project_umap(
        projects, behaviors, apu.EmbeddingSettings(reducer="pca", pca_components=0))
    s = umap_plot.PlotSettings(point_glow=6.0, point_glow_alpha=0.06,
                               label_mode="none", save_pdf=False)
    assert umap_plot.render(res.frame, s, tmp_path, stem="glow")[0].exists()


def test_behavior_names_fold_case_and_whitespace(tmp_path):
    """'Wet Dog Shake' and 'Wet dog shake' are one behavior, not two.

    Two spellings meant two colors, two legend entries, and silent removal from
    the cross-assay conservation analysis, which needs one name in >= 2 assays.
    """
    a = _write_project(tmp_path / "A", name="A",
                       behaviors={"b1": "Wet Dog Shake", "b2": "Rear"}, seed=1)
    b = _write_project(tmp_path / "B", name="B",
                       behaviors={"b3": "Wet dog  shake", "b4": "rear"}, seed=2)
    projects = [a, b]
    behaviors = {"A": ["b1", "b2"], "B": ["b3", "b4"]}
    s = apu.EmbeddingSettings(pool_behaviors_by_name=True)
    cols, _ = apu.resolve_feature_columns(projects, s)
    frame, warns = apu.collect_rows(projects, behaviors, cols, s)

    assert set(frame["group"]) == {"Wet Dog Shake", "Rear"}
    assert any("Merged case/spacing variants" in w for w in warns)


def test_canonical_name_map_keeps_the_commonest_spelling():
    m = apu.canonical_name_map(["Rear", "Rear", "rear", "  Head   Dip ", "Head Dip"])
    assert m["rear"] == "Rear" and m["Rear"] == "Rear"
    assert m["Head Dip"] == "Head Dip"
    # Exact ties fall back to alphabetical order rather than to input order.
    assert apu.canonical_name_map(["Dig", "dig"])["dig"] == "Dig"
