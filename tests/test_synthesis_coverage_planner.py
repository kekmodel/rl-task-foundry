from pathlib import Path

from rl_task_foundry.config import load_config
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.coverage_planner import SynthesisCoveragePlanner
from rl_task_foundry.synthesis.environment_registry import (
    DifficultyBand,
    EnvironmentRegistryCoverageEntry,
)
from rl_task_foundry.synthesis.orchestrator import SynthesisDbRegistryEntry


def test_synthesis_coverage_planner_builds_pair_and_cell_deficits() -> None:
    planner = SynthesisCoveragePlanner(target_count_per_band=2)
    plan = planner.build_plan(
        [
            SynthesisDbRegistryEntry(
                db_id="sakila",
                categories=[
                    CategoryTaxonomy.ASSIGNMENT,
                    CategoryTaxonomy.ITINERARY,
                ],
            ),
            SynthesisDbRegistryEntry(
                db_id="northwind",
                categories=[CategoryTaxonomy.ASSIGNMENT],
            ),
        ],
        [
            EnvironmentRegistryCoverageEntry(
                db_id="sakila",
                category=CategoryTaxonomy.ASSIGNMENT,
                difficulty_band=DifficultyBand.LOW,
                count=2,
            ),
            EnvironmentRegistryCoverageEntry(
                db_id="sakila",
                category=CategoryTaxonomy.ASSIGNMENT,
                difficulty_band=DifficultyBand.MEDIUM,
                count=1,
            ),
            EnvironmentRegistryCoverageEntry(
                db_id="northwind",
                category=CategoryTaxonomy.ASSIGNMENT,
                difficulty_band=DifficultyBand.HIGH,
                count=2,
            ),
        ],
    )

    assert plan.tracked_bands == (
        DifficultyBand.LOW,
        DifficultyBand.MEDIUM,
        DifficultyBand.HIGH,
    )
    assert plan.total_pairs == 3
    assert plan.total_cells == 9
    assert plan.satisfied_cells == 2
    assert plan.deficit_cells == 7
    assert plan.deficit_pairs == 3
    assert plan.total_deficit == 13
    assert plan.pair_plans[0].db_id == "sakila"
    assert plan.pair_plans[0].category == CategoryTaxonomy.ITINERARY
    assert plan.pair_plans[0].total_deficit == 6
    assert plan.pair_plans[0].missing_bands == (
        DifficultyBand.LOW,
        DifficultyBand.MEDIUM,
        DifficultyBand.HIGH,
    )
    assert any(
        cell.db_id == "sakila"
        and cell.category == CategoryTaxonomy.ASSIGNMENT
        and cell.difficulty_band == DifficultyBand.MEDIUM
        and cell.deficit == 1
        for cell in plan.cell_plans
    )


def test_synthesis_coverage_planner_can_track_unset_band() -> None:
    planner = SynthesisCoveragePlanner(
        target_count_per_band=1,
        include_unset_band=True,
    )
    plan = planner.build_plan(
        [
            SynthesisDbRegistryEntry(
                db_id="sakila",
                categories=[CategoryTaxonomy.ASSIGNMENT],
            )
        ],
        [
            EnvironmentRegistryCoverageEntry(
                db_id="sakila",
                category=CategoryTaxonomy.ASSIGNMENT,
                difficulty_band=DifficultyBand.UNSET,
                count=1,
            )
        ],
    )

    assert plan.tracked_bands == (
        DifficultyBand.LOW,
        DifficultyBand.MEDIUM,
        DifficultyBand.HIGH,
        DifficultyBand.UNSET,
    )
    assert plan.total_cells == 4
    assert plan.satisfied_cells == 1
    assert plan.deficit_cells == 3
    assert plan.pair_plans[0].missing_bands == (
        DifficultyBand.LOW,
        DifficultyBand.MEDIUM,
        DifficultyBand.HIGH,
    )


def test_synthesis_coverage_planner_reads_config_defaults() -> None:
    planner = SynthesisCoveragePlanner.for_config(load_config(Path("rl_task_foundry.yaml")))

    assert planner.target_count_per_band == 3
    assert planner.include_unset_band is False
    assert planner.tracked_bands == (
        DifficultyBand.LOW,
        DifficultyBand.MEDIUM,
        DifficultyBand.HIGH,
    )
