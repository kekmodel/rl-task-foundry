from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.orchestrator import (
    SynthesisDbRegistryEntry,
    SynthesisOrchestrator,
)
from rl_task_foundry.synthesis.runtime import SynthesisCategoryStatus
from rl_task_foundry.synthesis.scheduler import SynthesisSelectionStatus


@dataclass(slots=True)
class _FakeRuntime:
    category_status_payload: dict[CategoryTaxonomy, SynthesisCategoryStatus] = field(
        default_factory=dict
    )
    draft_result: object | None = None
    category_status_calls: list[str | None] = field(default_factory=list)
    synthesize_calls: list[tuple[str, CategoryTaxonomy, object | None]] = field(
        default_factory=list
    )
    closed: bool = False

    async def category_status(
        self,
        *,
        db_id: str | None = None,
    ) -> dict[CategoryTaxonomy, SynthesisCategoryStatus]:
        self.category_status_calls.append(db_id)
        return self.category_status_payload

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_category: CategoryTaxonomy,
        graph: object | None = None,
    ) -> object:
        self.synthesize_calls.append((db_id, requested_category, graph))
        return self.draft_result if self.draft_result is not None else SimpleNamespace(db_id=db_id)

    async def close(self) -> None:
        self.closed = True


def _backed_off_status(
    *,
    db_id: str,
    category: CategoryTaxonomy,
    remaining_s: float = 30.0,
) -> SynthesisCategoryStatus:
    now = datetime.now(timezone.utc)
    return SynthesisCategoryStatus(
        db_id=db_id,
        category=category,
        consecutive_discards=3,
        backed_off=True,
        backoff_until=now + timedelta(seconds=remaining_s),
        backoff_remaining_s=remaining_s,
        last_updated_at=now,
    )


@pytest.mark.asyncio
async def test_synthesis_orchestrator_choose_next_builds_snapshots_without_runtime() -> None:
    orchestrator = SynthesisOrchestrator(runtime_factory=lambda _entry: _FakeRuntime())

    step = await orchestrator.choose_next(
        [
            SynthesisDbRegistryEntry(
                db_id="sakila",
                categories=[CategoryTaxonomy.ASSIGNMENT],
            ),
            SynthesisDbRegistryEntry(
                db_id="northwind",
                categories=[CategoryTaxonomy.ITINERARY],
            ),
        ]
    )

    assert step.decision.status == SynthesisSelectionStatus.READY
    assert (step.decision.db_id, step.decision.category) == (
        "sakila",
        CategoryTaxonomy.ASSIGNMENT,
    )
    assert len(step.snapshots) == 2
    assert step.snapshots[0].category_status == {}
    assert step.snapshots[1].category_status == {}
    assert step.draft is None


@pytest.mark.asyncio
async def test_synthesis_orchestrator_build_snapshots_uses_cached_runtime_status() -> None:
    runtime = _FakeRuntime(
        category_status_payload={
            CategoryTaxonomy.ASSIGNMENT: _backed_off_status(
                db_id="sakila",
                category=CategoryTaxonomy.ASSIGNMENT,
            )
        }
    )
    orchestrator = SynthesisOrchestrator(runtime_factory=lambda _entry: runtime)
    orchestrator._runtimes["sakila"] = runtime

    snapshots = await orchestrator.build_snapshots(
        [
            SynthesisDbRegistryEntry(
                db_id="sakila",
                categories=[CategoryTaxonomy.ASSIGNMENT],
            ),
            SynthesisDbRegistryEntry(
                db_id="northwind",
                categories=[CategoryTaxonomy.ITINERARY],
            ),
        ]
    )

    assert runtime.category_status_calls == ["sakila"]
    assert snapshots[0].category_status[CategoryTaxonomy.ASSIGNMENT].backed_off is True
    assert snapshots[1].category_status == {}


@pytest.mark.asyncio
async def test_synthesis_orchestrator_run_next_uses_cached_runtime_per_db() -> None:
    created: dict[str, _FakeRuntime] = {}

    def _factory(entry: SynthesisDbRegistryEntry) -> _FakeRuntime:
        db_id = entry.db_id
        runtime = _FakeRuntime(draft_result=SimpleNamespace(env_id=f"env-{db_id}"))
        created[db_id] = runtime
        return runtime

    orchestrator = SynthesisOrchestrator(runtime_factory=_factory)
    registry = [
        SynthesisDbRegistryEntry(
            db_id="sakila",
            categories=[CategoryTaxonomy.ASSIGNMENT],
            graph="graph-a",
        ),
    ]

    first = await orchestrator.run_next(registry)
    second = await orchestrator.run_next(registry)

    assert first.decision.status == SynthesisSelectionStatus.READY
    assert second.decision.status == SynthesisSelectionStatus.READY
    assert len(created) == 1
    assert created["sakila"].category_status_calls == ["sakila"]
    assert created["sakila"].synthesize_calls == [
        ("sakila", CategoryTaxonomy.ASSIGNMENT, "graph-a"),
        ("sakila", CategoryTaxonomy.ASSIGNMENT, "graph-a"),
    ]
    assert getattr(first.draft, "env_id") == "env-sakila"
    assert getattr(second.draft, "env_id") == "env-sakila"


@pytest.mark.asyncio
async def test_synthesis_orchestrator_run_next_rotates_across_dbs() -> None:
    created: dict[str, _FakeRuntime] = {}

    def _factory(entry: SynthesisDbRegistryEntry) -> _FakeRuntime:
        db_id = entry.db_id
        runtime = _FakeRuntime(draft_result=SimpleNamespace(env_id=f"env-{db_id}"))
        created[db_id] = runtime
        return runtime

    orchestrator = SynthesisOrchestrator(runtime_factory=_factory)
    registry = [
        SynthesisDbRegistryEntry(
            db_id="sakila",
            categories=[CategoryTaxonomy.ASSIGNMENT],
        ),
        SynthesisDbRegistryEntry(
            db_id="northwind",
            categories=[CategoryTaxonomy.ITINERARY],
        ),
    ]

    first = await orchestrator.run_next(registry)
    second = await orchestrator.run_next(registry)

    assert (first.decision.db_id, first.decision.category) == (
        "sakila",
        CategoryTaxonomy.ASSIGNMENT,
    )
    assert (second.decision.db_id, second.decision.category) == (
        "northwind",
        CategoryTaxonomy.ITINERARY,
    )
    assert getattr(first.draft, "env_id") == "env-sakila"
    assert getattr(second.draft, "env_id") == "env-northwind"


@pytest.mark.asyncio
async def test_synthesis_orchestrator_run_next_returns_backoff_without_synthesizing() -> None:
    runtime = _FakeRuntime(
        category_status_payload={
            CategoryTaxonomy.ASSIGNMENT: _backed_off_status(
                db_id="sakila",
                category=CategoryTaxonomy.ASSIGNMENT,
            )
        }
    )
    orchestrator = SynthesisOrchestrator(runtime_factory=lambda _entry: runtime)
    orchestrator._runtimes["sakila"] = runtime

    step = await orchestrator.run_next(
        [
            SynthesisDbRegistryEntry(
                db_id="sakila",
                categories=[CategoryTaxonomy.ASSIGNMENT],
            )
        ]
    )

    assert step.decision.status == SynthesisSelectionStatus.BACKOFF
    assert step.draft is None
    assert runtime.synthesize_calls == []


@pytest.mark.asyncio
async def test_synthesis_orchestrator_close_closes_cached_runtimes() -> None:
    created: dict[str, _FakeRuntime] = {}

    def _factory(entry: SynthesisDbRegistryEntry) -> _FakeRuntime:
        db_id = entry.db_id
        runtime = _FakeRuntime()
        created[db_id] = runtime
        return runtime

    orchestrator = SynthesisOrchestrator(runtime_factory=_factory)
    await orchestrator.run_next(
        [
            SynthesisDbRegistryEntry(
                db_id="sakila",
                categories=[CategoryTaxonomy.ASSIGNMENT],
            ),
            SynthesisDbRegistryEntry(
                db_id="northwind",
                categories=[CategoryTaxonomy.ITINERARY],
            ),
        ]
    )
    await orchestrator.run_next(
        [
            SynthesisDbRegistryEntry(
                db_id="northwind",
                categories=[CategoryTaxonomy.ITINERARY],
            )
        ]
    )

    await orchestrator.close()

    assert created["sakila"].closed is True
    assert created["northwind"].closed is True
