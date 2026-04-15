from pathlib import Path

from rl_task_foundry.config import load_config
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.coverage_planner import SynthesisCoveragePlanner
from rl_task_foundry.synthesis.orchestrator import SynthesisDbRegistryEntry
from rl_task_foundry.synthesis.task_registry import (
    TaskRegistryCoverageEntry,
)


def test_synthesis_coverage_planner_builds_db_deficits() -> None:
    planner = SynthesisCoveragePlanner(target_count_per_pair=2)
    plan = planner.build_plan(
        [
            SynthesisDbRegistryEntry(db_id="sakila"),
            SynthesisDbRegistryEntry(db_id="northwind"),
        ],
        [
            TaskRegistryCoverageEntry(
                db_id="sakila",
                category=CategoryTaxonomy.ASSIGNMENT,
                count=2,
            ),
            TaskRegistryCoverageEntry(
                db_id="northwind",
                category=CategoryTaxonomy.ASSIGNMENT,
                count=1,
            ),
        ],
    )

    assert plan.total_pairs == 2
    assert plan.total_cells == 2
    # sakila has 2 total coverage entries, meets target of 2
    assert plan.satisfied_cells == 1
    assert plan.deficit_cells == 1
    assert plan.deficit_pairs == 1
    assert plan.total_deficit == 1
    # northwind has 1 count → deficit=1, sorted first by deficit desc
    assert plan.pair_plans[0].db_id == "northwind"
    assert plan.pair_plans[0].total_deficit == 1
    assert plan.pair_plans[1].db_id == "sakila"
    assert plan.pair_plans[1].total_deficit == 0


def test_synthesis_coverage_planner_reads_config_defaults() -> None:
    planner = SynthesisCoveragePlanner.for_config(load_config(Path("rl_task_foundry.yaml")))

    assert planner.target_count_per_pair == 3
