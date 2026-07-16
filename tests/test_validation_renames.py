"""Project and behavior renames in the validation suite.

Renames are display-only. The load-bearing invariant is that they must NOT follow
through to disk: trained-model directories and temporal-refinement settings are keyed
by the project's *own* behavior name, so a rename that leaked into a disk lookup would
silently orphan a behavior's trained model and the suite would report it as untrained.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abel.validation.analyses.generalization import GeneralizationResult
from abel.validation.datamodel import CellResult, ProjectRef


def _project() -> ProjectRef:
    return ProjectRef(
        project_id="CAB_NOP_2023", name="CAB_NOP_2023", source_name="CAB_NOP_2023",
        root=Path("/projects/cab_nop"),
        behavior_names={"b1": "Grooming", "b2": "Freeze", "no_behavior": "None"},
    )


# ── Behavior renames ───────────────────────────────────────────────────────

def test_behavior_rename_changes_the_label_but_never_the_disk_name() -> None:
    p = _project()
    assert p.behavior_label("b1") == "Grooming"

    p.set_behavior_alias("b1", "Groom")
    assert p.behavior_label("b1") == "Groom"
    # The model folder on disk is still behavior_model_Grooming.
    assert p.behavior_disk_name("b1") == "Grooming"


def test_blank_or_unchanged_behavior_rename_clears_the_alias() -> None:
    p = _project()
    p.set_behavior_alias("b1", "Groom")
    p.set_behavior_alias("b1", "")
    assert p.behavior_label("b1") == "Grooming"
    assert not p.behavior_aliases

    p.set_behavior_alias("b1", "Grooming")  # same as disk — not worth storing
    assert not p.behavior_aliases


def test_behavior_matches_by_either_old_or_new_name() -> None:
    """Analyses that take typed behavior names must accept whichever the user knows."""
    p = _project()
    p.set_behavior_alias("b1", "Groom")
    assert p.behavior_ids_matching(["Groom"]) == ["b1"]
    assert p.behavior_ids_matching(["Grooming"]) == ["b1"]
    assert p.behavior_ids_matching(["groom"]) == ["b1"]  # case-insensitive
    assert p.behavior_ids_matching(["Rear"]) == []


# ── Project renames ────────────────────────────────────────────────────────

def test_project_rename_relabels_but_leaves_root_alone() -> None:
    p = _project()
    assert not p.is_renamed

    p.rename("Novel Object")
    assert p.project_id == "Novel Object"   # what every figure titles itself with
    assert p.name == "Novel Object"
    assert p.original_name == "CAB_NOP_2023"
    assert p.is_renamed
    # root is the only disk locator the project has — a rename must not move it.
    assert p.root == Path("/projects/cab_nop")


def test_blank_project_rename_restores_the_original_name() -> None:
    p = _project()
    p.rename("Novel Object")
    p.rename("   ")
    assert p.project_id == "CAB_NOP_2023"
    assert not p.is_renamed


# ── The point of it all: renames make two projects poolable ────────────────

def test_renames_let_two_projects_pool_into_one_generalization_bar() -> None:
    from abel.validation.plots import pool_generalization_by_behavior

    a = _project()
    a.rename("Novel Object")
    a.set_behavior_alias("b1", "Groom")          # "Grooming" -> "Groom"

    b = ProjectRef(project_id="EPM_v2", name="EPM_v2", source_name="EPM_v2",
                   root=Path("/projects/epm"), behavior_names={"g": "Groom"})

    def result(proj: ProjectRef, bid: str, kappa: float) -> GeneralizationResult:
        name = proj.behavior_label(bid)
        return GeneralizationResult(
            project_id=proj.project_id, behavior_id=bid, behavior_name=name,
            kappa_mean=kappa, f1_mean=kappa,
            cells=[
                CellResult(
                    project_id=proj.project_id, project_name=proj.name, behavior_id=bid,
                    behavior_name=name, analysis="generalization",
                    config_name="held_out_subjects", n_clips=50, seed=s,
                    cohen_kappa=kappa, f1=kappa,
                )
                for s in range(3)
            ],
        )

    df = pool_generalization_by_behavior([result(a, "b1", 0.9), result(b, "g", 0.7)])

    # One bar, not two: the rename is what asserted these are the same behavior.
    assert list(df["behavior"]) == ["Groom"]
    assert int(df.loc[0, "n_projects"]) == 2
    assert int(df.loc[0, "n_cells"]) == 6      # pooled over project x seed, not mean-of-means
    assert df.loc[0, "kappa"] == pytest.approx(0.8)
