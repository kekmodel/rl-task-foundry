"""Multi-db orchestration helpers for synthesis runtime scheduling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rl_task_foundry.schema.graph import SchemaGraph
from rl_task_foundry.config.models import DatabaseConfig, DomainConfig
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.runtime import (
    SynthesisCategoryStatus,
    SynthesisEnvironmentDraft,
)
from rl_task_foundry.synthesis.scheduler import (
    SynthesisDbSnapshot,
    SynthesisDomainScheduler,
    SynthesisSchedulerDecision,
    SynthesisSelectionStatus,
)


class SynthesisRuntimeHandle(Protocol):
    async def category_status(
        self,
        *,
        db_id: str | None = None,
    ) -> dict[CategoryTaxonomy, SynthesisCategoryStatus]: ...

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_category: CategoryTaxonomy,
        graph: SchemaGraph | None = None,
    ) -> SynthesisEnvironmentDraft: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class SynthesisDbRegistryEntry:
    db_id: str
    categories: list[CategoryTaxonomy]
    database: DatabaseConfig | None = None
    domain: DomainConfig | None = None
    graph: SchemaGraph | None = None


@dataclass(slots=True)
class SynthesisOrchestrationStep:
    decision: SynthesisSchedulerDecision
    snapshots: list[SynthesisDbSnapshot] = field(default_factory=list)
    draft: SynthesisEnvironmentDraft | None = None


RuntimeFactory = Callable[[SynthesisDbRegistryEntry], SynthesisRuntimeHandle]


@dataclass(slots=True)
class SynthesisOrchestrator:
    """Thin multi-db orchestrator that delegates per-db work to single-db runtimes.

    Not async-safe. Callers should drive one orchestrator instance from a single
    coordinator task.
    """

    runtime_factory: RuntimeFactory
    scheduler: SynthesisDomainScheduler = field(default_factory=SynthesisDomainScheduler)
    _runtimes: dict[str, SynthesisRuntimeHandle] = field(default_factory=dict, init=False, repr=False)

    async def build_snapshots(
        self,
        registry: Sequence[SynthesisDbRegistryEntry],
    ) -> list[SynthesisDbSnapshot]:
        """Build scheduler snapshots from cached runtime state.

        A db that has not been selected yet has no instantiated runtime and
        therefore contributes an empty category-status snapshot until its first
        draft run.
        """
        snapshots: list[SynthesisDbSnapshot] = []
        for entry in registry:
            runtime = self._runtimes.get(entry.db_id)
            category_status = (
                await runtime.category_status(db_id=entry.db_id) if runtime is not None else {}
            )
            snapshots.append(
                SynthesisDbSnapshot(
                    db_id=entry.db_id,
                    categories=list(entry.categories),
                    category_status=category_status,
                )
            )
        return snapshots

    async def choose_next(
        self,
        registry: Sequence[SynthesisDbRegistryEntry],
    ) -> SynthesisOrchestrationStep:
        snapshots = await self.build_snapshots(registry)
        decision = self.scheduler.choose_next(snapshots)
        return SynthesisOrchestrationStep(
            decision=decision,
            snapshots=snapshots,
        )

    async def run_next(
        self,
        registry: Sequence[SynthesisDbRegistryEntry],
    ) -> SynthesisOrchestrationStep:
        step = await self.choose_next(registry)
        if step.decision.status != SynthesisSelectionStatus.READY:
            return step
        assert step.decision.db_id is not None
        assert step.decision.category is not None
        entry = next(item for item in registry if item.db_id == step.decision.db_id)
        runtime = self._runtime_for(entry)
        draft = await runtime.synthesize_environment_draft(
            db_id=entry.db_id,
            requested_category=step.decision.category,
            graph=entry.graph,
        )
        return SynthesisOrchestrationStep(
            decision=step.decision,
            snapshots=step.snapshots,
            draft=draft,
        )

    async def close(self) -> None:
        for runtime in self._runtimes.values():
            await runtime.close()
        self._runtimes.clear()

    def _runtime_for(self, entry: SynthesisDbRegistryEntry) -> SynthesisRuntimeHandle:
        runtime = self._runtimes.get(entry.db_id)
        if runtime is None:
            runtime = self.runtime_factory(entry)
            self._runtimes[entry.db_id] = runtime
        return runtime
