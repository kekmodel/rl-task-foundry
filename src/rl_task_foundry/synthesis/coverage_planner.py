"""Coverage target planning for the durable synthesis task registry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.orchestrator import SynthesisDbRegistryEntry
from rl_task_foundry.synthesis.task_registry import (
    DifficultyBand,
    TaskRegistryCoverageEntry,
)


@dataclass(slots=True)
class SynthesisCoverageCellPlan:
    db_id: str
    topic: str
    difficulty_band: DifficultyBand
    current_count: int
    target_count: int
    deficit: int

    @property
    def satisfied(self) -> bool:
        return self.deficit == 0

    @property
    def category(self) -> str:
        return self.topic


@dataclass(slots=True)
class SynthesisCoveragePairPlan:
    db_id: str
    topic: str
    cells: tuple[SynthesisCoverageCellPlan, ...]
    total_current_count: int
    total_target_count: int
    total_deficit: int
    missing_bands: tuple[DifficultyBand, ...]
    max_band_deficit: int

    @property
    def category(self) -> str:
        return self.topic


@dataclass(slots=True)
class SynthesisCoveragePlan:
    target_count_per_band: int
    tracked_bands: tuple[DifficultyBand, ...]
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
    """Compute per-db/topic coverage deficits against the registry inventory."""

    target_count_per_band: int = 3
    include_unset_band: bool = False

    @classmethod
    def for_config(cls, config: AppConfig) -> "SynthesisCoveragePlanner":
        planner_config = config.synthesis.coverage_planner
        return cls(
            target_count_per_band=planner_config.target_count_per_band,
            include_unset_band=planner_config.include_unset_band,
        )

    @property
    def tracked_bands(self) -> tuple[DifficultyBand, ...]:
        bands = (
            DifficultyBand.LOW,
            DifficultyBand.MEDIUM,
            DifficultyBand.HIGH,
        )
        if self.include_unset_band:
            return bands + (DifficultyBand.UNSET,)
        return bands

    def build_plan(
        self,
        registry: Sequence[SynthesisDbRegistryEntry],
        coverage_entries: Sequence[TaskRegistryCoverageEntry],
    ) -> SynthesisCoveragePlan:
        counts = {
            (entry.db_id, entry.category, entry.difficulty_band): entry.count
            for entry in coverage_entries
        }
        pair_plans: list[SynthesisCoveragePairPlan] = []
        cell_plans: list[SynthesisCoverageCellPlan] = []
        for entry in registry:
            for topic in entry.topics:
                pair_cells: list[SynthesisCoverageCellPlan] = []
                for difficulty_band in self.tracked_bands:
                    current_count = counts.get((entry.db_id, topic, difficulty_band), 0)
                    deficit = max(0, self.target_count_per_band - current_count)
                    cell = SynthesisCoverageCellPlan(
                        db_id=entry.db_id,
                        topic=topic,
                        difficulty_band=difficulty_band,
                        current_count=current_count,
                        target_count=self.target_count_per_band,
                        deficit=deficit,
                    )
                    pair_cells.append(cell)
                    cell_plans.append(cell)
                pair_plans.append(
                    SynthesisCoveragePairPlan(
                        db_id=entry.db_id,
                        topic=topic,
                        cells=tuple(pair_cells),
                        total_current_count=sum(cell.current_count for cell in pair_cells),
                        total_target_count=sum(cell.target_count for cell in pair_cells),
                        total_deficit=sum(cell.deficit for cell in pair_cells),
                        missing_bands=tuple(
                            cell.difficulty_band for cell in pair_cells if cell.deficit > 0
                        ),
                        max_band_deficit=max((cell.deficit for cell in pair_cells), default=0),
                    )
                )
        pair_plans.sort(
            key=lambda pair: (
                -pair.total_deficit,
                pair.db_id,
                pair.topic,
            )
        )
        cell_plans.sort(
            key=lambda cell: (
                -cell.deficit,
                cell.db_id,
                cell.topic,
                _difficulty_band_order(cell.difficulty_band),
            )
        )
        return SynthesisCoveragePlan(
            target_count_per_band=self.target_count_per_band,
            tracked_bands=self.tracked_bands,
            pair_plans=pair_plans,
            cell_plans=cell_plans,
        )


def _difficulty_band_order(difficulty_band: DifficultyBand) -> int:
    order = {
        DifficultyBand.LOW: 0,
        DifficultyBand.MEDIUM: 1,
        DifficultyBand.HIGH: 2,
        DifficultyBand.UNSET: 3,
    }
    return order[difficulty_band]
