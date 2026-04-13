from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import DatabaseConfig, DomainConfig
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy, EnvironmentStatus
from rl_task_foundry.synthesis.environment_registry import (
    DifficultyBand,
    EnvironmentRegistryCommitResult,
    EnvironmentRegistryCommitStatus,
)
from rl_task_foundry.synthesis.orchestrator import SynthesisDbRegistryEntry
from rl_task_foundry.synthesis.runner import (
    SynthesisRegistryRunOutcome,
    SynthesisRegistryRunner,
    load_synthesis_registry,
)
from rl_task_foundry.synthesis.runtime import SynthesisCategoryStatus
from rl_task_foundry.synthesis.scheduler import SynthesisSelectionStatus
from tests.test_synthesis_environment_registry import _sample_draft


@dataclass(slots=True)
class _FakeRuntime:
    category_status_payload: dict[str, SynthesisCategoryStatus] = field(default_factory=dict)
    category_status_calls: list[str | None] = field(default_factory=list)
    synthesize_calls: list[tuple[str, str, object | None]] = field(default_factory=list)
    closed: bool = False

    async def category_status(
        self,
        *,
        db_id: str | None = None,
    ) -> dict[str, SynthesisCategoryStatus]:
        self.category_status_calls.append(db_id)
        return self.category_status_payload

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_topic: str,
        graph: object | None = None,
    ) -> object:
        self.synthesize_calls.append((db_id, requested_topic, graph))
        draft = _sample_draft(
            tmp_env_id=f"env_{db_id}_{requested_topic}_{len(self.synthesize_calls)}",
            db_id=db_id,
            topic=requested_topic,
            created_at=datetime.now(timezone.utc),
        )
        return draft.model_copy(
            update={
                "environment": draft.environment.model_copy(
                    update={
                        "status": EnvironmentStatus.ACCEPTED,
                        "quality_metrics": draft.environment.quality_metrics.model_copy(
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
    commit_results: list[EnvironmentRegistryCommitResult] = field(default_factory=list)
    committed_drafts: list[object] = field(default_factory=list)
    closed: bool = False

    def commit_draft(self, draft: object) -> EnvironmentRegistryCommitResult:
        self.committed_drafts.append(draft)
        if self.commit_results:
            return self.commit_results.pop(0)
        env_id = draft.environment.env_id
        return EnvironmentRegistryCommitResult(
            status=EnvironmentRegistryCommitStatus.COMMITTED,
            env_id=env_id,
            exact_signature=f"sha256:{env_id}",
            difficulty_band=DifficultyBand.UNSET,
            filesystem_path=self.root_dir / env_id,
        )

    def close(self) -> None:
        self.closed = True


def _config_with_run_db(tmp_path: Path):
    config = load_config(Path("rl_task_foundry.yaml"))
    output = config.output.model_copy(
        update={
            "run_db_path": tmp_path / "run.db",
            "traces_dir": tmp_path / "traces",
            "events_jsonl_path": tmp_path / "events.jsonl",
        },
        deep=True,
    )
    return config.model_copy(update={"output": output}, deep=True)


def _backed_off_status(*, db_id: str, category: CategoryTaxonomy) -> SynthesisCategoryStatus:
    now = datetime.now(timezone.utc)
    return SynthesisCategoryStatus(
        db_id=db_id,
        category=category,
        consecutive_discards=3,
        backed_off=True,
        backoff_until=now + timedelta(seconds=30),
        backoff_remaining_s=30.0,
        last_updated_at=now,
    )


@pytest.mark.asyncio
async def test_synthesis_registry_runner_marks_pairs_and_resumes_from_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config_with_run_db(tmp_path)
    created: dict[str, _FakeRuntime] = {}

    def _factory(entry: SynthesisDbRegistryEntry) -> _FakeRuntime:
        runtime = _FakeRuntime()
        created[entry.db_id] = runtime
        return runtime

    registry = [
        SynthesisDbRegistryEntry(
            db_id="sakila",
            categories=[CategoryTaxonomy.ASSIGNMENT, CategoryTaxonomy.ITINERARY],
        )
    ]

    fake_registry = _FakeRegistry(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    runner = SynthesisRegistryRunner(
        config,
        runtime_factory=_factory,
        environment_registry=fake_registry,
    )
    try:
        summary = await runner.run_steps(
            registry,
            max_steps=2,
            checkpoint_namespace="synthesis_test",
        )
    finally:
        await runner.close()

    assert summary.initially_processed_pairs == 0
    assert summary.outcome == SynthesisRegistryRunOutcome.COMPLETED_ALL
    assert summary.processed_pairs_after_run == 2
    assert summary.generated_drafts == 2
    assert summary.quality_accepted_envs == 2
    assert summary.quality_rejected_envs == 0
    assert summary.registry_committed_envs == 2
    assert summary.remaining_pairs == 0
    assert len(summary.generated_env_ids) == 2
    assert summary.committed_env_ids == summary.generated_env_ids
    assert summary.phase_monitor_log_path == tmp_path / "phase_monitors.jsonl"
    phase_monitor_lines = [
        json.loads(line)
        for line in summary.phase_monitor_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(line["phase"] == "quality_gate" for line in phase_monitor_lines)
    assert any(line["phase"] == "registry_commit" for line in phase_monitor_lines)
    assert created["sakila"].synthesize_calls == [
        ("sakila", CategoryTaxonomy.ASSIGNMENT, None),
        ("sakila", CategoryTaxonomy.ITINERARY, None),
    ]
    assert fake_registry.closed is True

    runner2 = SynthesisRegistryRunner(
        _config_with_run_db(tmp_path),
        runtime_factory=lambda _entry: pytest.fail("runtime should not be created on resume"),
        environment_registry=_FakeRegistry(
            root_dir=tmp_path / "environments",
            index_db_path=tmp_path / "environment_registry.db",
        ),
    )
    try:
        resumed = await runner2.run_steps(
            registry,
            max_steps=2,
            checkpoint_namespace="synthesis_test",
        )
    finally:
        await runner2.close()

    assert resumed.initially_processed_pairs == 2
    assert resumed.outcome == SynthesisRegistryRunOutcome.COMPLETED_ALL
    assert resumed.generated_drafts == 0


@pytest.mark.asyncio
async def test_synthesis_registry_runner_cold_start_ignores_backoff_until_runtime_is_cached(
    tmp_path: Path,
) -> None:
    config = _config_with_run_db(tmp_path)
    runtime = _FakeRuntime(
        category_status_payload={
            CategoryTaxonomy.ASSIGNMENT: _backed_off_status(
                db_id="sakila",
                category=CategoryTaxonomy.ASSIGNMENT,
            )
        }
    )
    fake_registry = _FakeRegistry(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    runner = SynthesisRegistryRunner(
        config,
        runtime_factory=lambda _entry: runtime,
        environment_registry=fake_registry,
    )
    registry = [
        SynthesisDbRegistryEntry(
            db_id="sakila",
            categories=[CategoryTaxonomy.ASSIGNMENT],
        )
    ]

    try:
        summary = await runner.run_steps(registry, max_steps=1)
    finally:
        await runner.close()

    assert summary.outcome == SynthesisRegistryRunOutcome.COMPLETED_ALL
    assert summary.executed_steps == 1
    assert summary.generated_drafts == 1


def test_load_synthesis_registry_supports_topics_and_legacy_categories(tmp_path: Path) -> None:
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

    assert [(entry.db_id, entry.topics) for entry in entries] == [
        ("sakila", ["assignment"]),
        ("northwind", ["itinerary"]),
    ]
    assert isinstance(entries[0].database, DatabaseConfig)
    assert isinstance(entries[1].domain, DomainConfig)


def test_synthesis_registry_entry_requires_topics() -> None:
    with pytest.raises(ValueError):
        SynthesisDbRegistryEntry(db_id="sakila", topics=[])
