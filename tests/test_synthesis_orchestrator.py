from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from rl_task_foundry.synthesis.orchestrator import (
    SynthesisDbRegistryEntry,
    SynthesisOrchestrator,
)
from rl_task_foundry.synthesis.scheduler import SynthesisSelectionStatus


@dataclass(slots=True)
class _FakeRuntime:
    draft_result: object | None = None
    category_status_calls: list[str | None] = field(default_factory=list)
    synthesize_calls: list[tuple[str, object, object | None]] = field(default_factory=list)
    closed: bool = False

    async def category_status(
        self,
        *,
        db_id: str | None = None,
    ) -> dict[str, object]:
        self.category_status_calls.append(db_id)
        return {}

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_topic: object = None,
        graph: object | None = None,
    ) -> object:
        self.synthesize_calls.append((db_id, requested_topic, graph))
        return self.draft_result if self.draft_result is not None else SimpleNamespace(db_id=db_id)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_synthesis_orchestrator_choose_next_builds_snapshots_without_runtime() -> None:
    orchestrator = SynthesisOrchestrator(runtime_factory=lambda _entry: _FakeRuntime())

    step = await orchestrator.choose_next(
        [
            SynthesisDbRegistryEntry(db_id="sakila"),
            SynthesisDbRegistryEntry(db_id="northwind"),
        ]
    )

    assert step.decision.status == SynthesisSelectionStatus.READY
    assert step.decision.db_id == "sakila"
    assert len(step.snapshots) == 2
    assert step.draft is None


@pytest.mark.asyncio
async def test_synthesis_orchestrator_run_next_uses_cached_runtime_per_db() -> None:
    created: dict[str, _FakeRuntime] = {}

    def _factory(entry: SynthesisDbRegistryEntry) -> _FakeRuntime:
        db_id = entry.db_id
        runtime = _FakeRuntime(draft_result=SimpleNamespace(task_id=f"task-{db_id}"))
        created[db_id] = runtime
        return runtime

    orchestrator = SynthesisOrchestrator(runtime_factory=_factory)
    registry = [
        SynthesisDbRegistryEntry(db_id="sakila", graph="graph-a"),
    ]

    first = await orchestrator.run_next(registry)
    second = await orchestrator.run_next(registry)

    assert first.decision.status == SynthesisSelectionStatus.READY
    assert second.decision.status == SynthesisSelectionStatus.READY
    assert len(created) == 1
    assert created["sakila"].synthesize_calls == [
        ("sakila", None, "graph-a"),
        ("sakila", None, "graph-a"),
    ]
    assert getattr(first.draft, "task_id") == "task-sakila"
    assert getattr(second.draft, "task_id") == "task-sakila"


@pytest.mark.asyncio
async def test_synthesis_orchestrator_run_next_rotates_across_dbs() -> None:
    created: dict[str, _FakeRuntime] = {}

    def _factory(entry: SynthesisDbRegistryEntry) -> _FakeRuntime:
        db_id = entry.db_id
        runtime = _FakeRuntime(draft_result=SimpleNamespace(task_id=f"task-{db_id}"))
        created[db_id] = runtime
        return runtime

    orchestrator = SynthesisOrchestrator(runtime_factory=_factory)
    registry = [
        SynthesisDbRegistryEntry(db_id="sakila"),
        SynthesisDbRegistryEntry(db_id="northwind"),
    ]

    first = await orchestrator.run_next(registry)
    second = await orchestrator.run_next(registry)

    assert first.decision.db_id == "sakila"
    assert second.decision.db_id == "northwind"
    assert getattr(first.draft, "task_id") == "task-sakila"
    assert getattr(second.draft, "task_id") == "task-northwind"


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
            SynthesisDbRegistryEntry(db_id="sakila"),
            SynthesisDbRegistryEntry(db_id="northwind"),
        ]
    )
    await orchestrator.run_next(
        [SynthesisDbRegistryEntry(db_id="northwind")]
    )

    await orchestrator.close()

    assert created["sakila"].closed is True
    assert created["northwind"].closed is True
