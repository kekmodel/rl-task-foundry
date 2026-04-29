"""Coverage target planning for the durable synthesis task registry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.orchestrator import SynthesisDbRegistryEntry
from rl_task_foundry.synthesis.task_registry import (
    TaskRegistryCoverageEntry,
)


@dataclass(slots=True)
class SynthesisCoverageCellPlan:
    db_id: str
    topic: str
    current_count: int
    target_count: int
    deficit: int

    @property
    def satisfied(self) -> bool:
        return self.deficit == 0


@dataclass(slots=True)
class SynthesisCoveragePairPlan:
    db_id: str
    topic: str
    cells: tuple[SynthesisCoverageCellPlan, ...]
    total_current_count: int
    total_target_count: int
    total_deficit: int


@dataclass(slots=True)
class SynthesisCoveragePlan:
    target_count_per_pair: int
    pair_plans: list[SynthesisCoveragePairPlan]
    cell_plans: list[SynthesisCoverageCellPlan]

    @property
    def total_pairs(self) -> int:
        return len(self.pair_plans)

    @property
    def total_cells(self) -> int:
        return len(self.cell_plans)

    @property
    def satisfied_cells(self) -> int:
        return sum(1 for cell in self.cell_plans if cell.satisfied)

    @property
    def deficit_cells(self) -> int:
        return sum(1 for cell in self.cell_plans if cell.deficit > 0)

    @property
    def deficit_pairs(self) -> int:
        return sum(1 for pair in self.pair_plans if pair.total_deficit > 0)

    @property
    def total_deficit(self) -> int:
        return sum(pair.total_deficit for pair in self.pair_plans)


@dataclass(slots=True)
class SynthesisCoveragePlanner:
    """Compute per-db coverage deficits against the registry inventory."""

    target_count_per_pair: int = 3

    @classmethod
    def for_config(cls, config: AppConfig) -> "SynthesisCoveragePlanner":
        planner_config = config.synthesis.coverage_planner
        return cls(
            target_count_per_pair=planner_config.target_count_per_band,
        )

    def build_plan(
        self,
        registry: Sequence[SynthesisDbRegistryEntry],
        coverage_entries: Sequence[TaskRegistryCoverageEntry],
    ) -> SynthesisCoveragePlan:
        # Build a lookup of existing coverage by db_id (sum across all topics)
        db_counts: dict[str, int] = {}
        for cov_entry in coverage_entries:
            db_counts[cov_entry.db_id] = db_counts.get(cov_entry.db_id, 0) + cov_entry.count

        pair_plans: list[SynthesisCoveragePairPlan] = []
        cell_plans: list[SynthesisCoverageCellPlan] = []
        for entry in registry:
            current_count = db_counts.get(entry.db_id, 0)
            deficit = max(0, self.target_count_per_pair - current_count)
            cell = SynthesisCoverageCellPlan(
                db_id=entry.db_id,
                topic=entry.db_id,
                current_count=current_count,
                target_count=self.target_count_per_pair,
                deficit=deficit,
            )
            cell_plans.append(cell)
            pair_plans.append(
                SynthesisCoveragePairPlan(
                    db_id=entry.db_id,
                    topic=entry.db_id,
                    cells=(cell,),
                    total_current_count=cell.current_count,
                    total_target_count=cell.target_count,
                    total_deficit=cell.deficit,
                )
            )
        pair_plans.sort(
            key=lambda pair: (
                -pair.total_deficit,
                pair.db_id,
            )
        )
        cell_plans.sort(
            key=lambda cell: (
                -cell.deficit,
                cell.db_id,
            )
        )
        return SynthesisCoveragePlan(
            target_count_per_pair=self.target_count_per_pair,
            pair_plans=pair_plans,
            cell_plans=cell_plans,
        )
