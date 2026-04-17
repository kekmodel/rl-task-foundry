from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import DatabaseConfig, DomainConfig
from rl_task_foundry.synthesis.contracts import TaskBundleStatus
from rl_task_foundry.synthesis.orchestrator import SynthesisDbRegistryEntry
from rl_task_foundry.synthesis.runner import (
    SynthesisRegistryRunner,
    SynthesisRegistryRunOutcome,
    load_synthesis_registry,
)
from rl_task_foundry.synthesis.scheduler import SynthesisSelectionStatus
from rl_task_foundry.synthesis.task_registry import (
    TaskRegistryCommitResult,
    TaskRegistryCommitStatus,
)
from tests.test_synthesis_task_registry import _sample_draft


@dataclass(slots=True)
class _FakeRuntime:
    category_status_payload: dict[str, object] = field(default_factory=dict)
    category_status_calls: list[str | None] = field(default_factory=list)
    synthesize_calls: list[tuple[str, object, object | None]] = field(default_factory=list)
    closed: bool = False

    async def category_status(
        self,
        *,
        db_id: str | None = None,
    ) -> dict[str, object]:
        self.category_status_calls.append(db_id)
        return self.category_status_payload

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_topic: object = None,
        graph: object | None = None,
    ) -> object:
        self.synthesize_calls.append((db_id, requested_topic, graph))
        draft = _sample_draft(
            tmp_task_id=f"task_{db_id}_{len(self.synthesize_calls)}",
            db_id=db_id,
            topic="agent_generated",
            created_at=datetime.now(timezone.utc),
        )
        return draft.model_copy(
            update={
                "task_bundle": draft.task_bundle.model_copy(
                    update={
                        "status": TaskBundleStatus.ACCEPTED,
                        "quality_metrics": draft.task_bundle.quality_metrics.model_copy(
                            update={
                                "solver_pass_rate": 0.5,
                                "solver_ci_low": 0.1,
                                "solver_ci_high": 0.9,
                            }
                        ),
                    }
                )
            }
        )

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _FakeRegistry:
    root_dir: Path
    index_db_path: Path
    commit_results: list[TaskRegistryCommitResult] = field(default_factory=list)
    committed_drafts: list[object] = field(default_factory=list)
    closed: bool = False

    def commit_draft(self, draft: object) -> TaskRegistryCommitResult:
        self.committed_drafts.append(draft)
        if self.commit_results:
            return self.commit_results.pop(0)
        task_id = draft.task_bundle.task_id
        return TaskRegistryCommitResult(
            status=TaskRegistryCommitStatus.COMMITTED,
            task_id=task_id,
            exact_signature=f"sha256:{task_id}",
            filesystem_path=self.root_dir / task_id,
        )

    def close(self) -> None:
        self.closed = True


def _config_with_run_db(tmp_path: Path):
    config = load_config(Path("rl_task_foundry.yaml"))
    output = config.output.model_copy(
        update={
            "run_db_path": tmp_path / "run.db",
            "traces_dir": tmp_path / "traces",
        },
        deep=True,
    )
    return config.model_copy(update={"output": output}, deep=True)


@pytest.mark.asyncio
async def test_synthesis_registry_runner_marks_entries_and_resumes_from_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config_with_run_db(tmp_path)
    created: dict[str, _FakeRuntime] = {}

    def _factory(entry: SynthesisDbRegistryEntry) -> _FakeRuntime:
        runtime = _FakeRuntime()
        created[entry.db_id] = runtime
        return runtime

    registry = [
        SynthesisDbRegistryEntry(db_id="sakila"),
    ]

    fake_registry = _FakeRegistry(
        root_dir=tmp_path / "tasks",
        index_db_path=tmp_path / "task_registry.db",
    )
    runner = SynthesisRegistryRunner(
        config,
        runtime_factory=_factory,
        task_registry=fake_registry,
    )
    try:
        summary = await runner.run_steps(
            registry,
            max_steps=1,
            checkpoint_namespace="synthesis_test",
        )
    finally:
        await runner.close()

    assert summary.initially_processed_entries == 0
    assert summary.outcome == SynthesisRegistryRunOutcome.COMPLETED_ALL
    assert summary.processed_entries_after_run == 1
    assert summary.generated_drafts == 1
    assert summary.quality_accepted_tasks == 1
    assert summary.quality_rejected_tasks == 0
    assert summary.registry_committed_tasks == 1
    assert summary.remaining_entries == 0
    assert len(summary.generated_task_ids) == 1
    assert summary.committed_task_ids == summary.generated_task_ids
    assert summary.phase_monitor_log_path == tmp_path / "phase_monitors.jsonl"
    phase_monitor_lines = [
        json.loads(line)
        for line in summary.phase_monitor_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(line["phase"] == "quality_gate" for line in phase_monitor_lines)
    assert any(line["phase"] == "registry_commit" for line in phase_monitor_lines)
    assert len(created["sakila"].synthesize_calls) == 1
    assert created["sakila"].synthesize_calls[0] == ("sakila", None, None)
    assert fake_registry.closed is True

    runner2 = SynthesisRegistryRunner(
        _config_with_run_db(tmp_path),
        runtime_factory=lambda _entry: pytest.fail("runtime should not be created on resume"),
        task_registry=_FakeRegistry(
            root_dir=tmp_path / "tasks",
            index_db_path=tmp_path / "task_registry.db",
        ),
    )
    try:
        resumed = await runner2.run_steps(
            registry,
            max_steps=1,
            checkpoint_namespace="synthesis_test",
        )
    finally:
        await runner2.close()

    assert resumed.initially_processed_entries == 1
    assert resumed.outcome == SynthesisRegistryRunOutcome.COMPLETED_ALL
    assert resumed.generated_drafts == 0


@pytest.mark.asyncio
async def test_synthesis_registry_runner_cold_start_produces_draft(
    tmp_path: Path,
) -> None:
    config = _config_with_run_db(tmp_path)
    runtime = _FakeRuntime()
    fake_registry = _FakeRegistry(
        root_dir=tmp_path / "tasks",
        index_db_path=tmp_path / "task_registry.db",
    )
    runner = SynthesisRegistryRunner(
        config,
        runtime_factory=lambda _entry: runtime,
        task_registry=fake_registry,
    )
    registry = [
        SynthesisDbRegistryEntry(db_id="sakila"),
    ]

    try:
        summary = await runner.run_steps(registry, max_steps=1)
    finally:
        await runner.close()

    assert summary.outcome == SynthesisRegistryRunOutcome.COMPLETED_ALL
    assert summary.executed_steps == 1
    assert summary.generated_drafts == 1


def test_load_synthesis_registry_supports_plain_db_ids_and_legacy_topics(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            [
                {
                    "db_id": "sakila",
                    "topics": ["assignment"],
                    "database": {
                        "dsn": "postgresql://reader@localhost/sakila",
                        "readonly_role": "rlvr_reader",
                    },
                },
                {
                    "db_id": "northwind",
                    "categories": ["itinerary"],
                    "database": {
                        "dsn": "postgresql://reader@localhost/northwind",
                        "readonly_role": "rlvr_reader",
                    },
                    "domain": {
                        "name": "support_ops",
                    },
                },
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    entries = load_synthesis_registry(registry_path)

    assert [entry.db_id for entry in entries] == ["sakila", "northwind"]
    assert isinstance(entries[0].database, DatabaseConfig)
    assert isinstance(entries[1].domain, DomainConfig)


def test_synthesis_registry_run_outcome_exposes_ready_status_name() -> None:
    assert SynthesisSelectionStatus.READY.value == "ready"
