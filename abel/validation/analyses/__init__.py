"""Validation analyses — each reuses the shared engine.run_one_config primitive.

- learning_curve : optimal-clips data-efficiency curves (headline)
- ablation       : per-feature/pipeline impact
- generalization : held-out subject/session agreement + human ceiling
- cross_project  : pure aggregation over cells.parquet (no new training)
- all_project_umap : one embedding of every labeled clip from every project
                     (no training at all — a map of the shared feature space,
                     rendered by abel.validation.umap_plot)
"""
