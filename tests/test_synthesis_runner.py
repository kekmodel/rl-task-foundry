from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import DatabaseConfig, DomainConfig
from rl_task_foundry.pipeline.environment_orchestrator import EnvironmentRolloutSummary
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.contracts import EnvironmentStatus
from rl_task_foundry.synthesis.environment_registry import (
    DifficultyBand,
    EnvironmentRegistryCommitResult,
    EnvironmentRegistryCommitStatus,
)
from rl_task_foundry.synthesis.orchestrator import SynthesisDbRegistryEntry
from rl_task_foundry.synthesis.runner import (
    SynthesisRegistryRunOutcome,
    SynthesisRegistryRunSummary,
    SynthesisRegistryRunner,
    load_synthesis_registry,
)
from rl_task_foundry.synthesis.runtime import SynthesisCategoryStatus
from rl_task_foundry.synthesis.scheduler import SynthesisSelectionStatus
from tests.test_synthesis_environment_registry import _sample_draft


@dataclass(slots=True)
class _FakeRuntime:
    category_status_payload: dict[CategoryTaxonomy, SynthesisCategoryStatus] = field(
        default_factory=dict
    )
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
        return _sample_draft(
            tmp_env_id=f"env_{db_id}_{requested_category.value}_{len(self.synthesize_calls)}",
            db_id=db_id,
            category=requested_category,
            created_at=datetime.now(timezone.utc),
        )

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _FakeEnvironmentOrchestrator:
    summaries: list[EnvironmentRolloutSummary] = field(default_factory=list)
    run_calls: list[object] = field(default_factory=list)
    closed: bool = False

    async def run_draft(self, draft: object) -> EnvironmentRolloutSummary:
        self.run_calls.append(draft)
        if self.summaries:
            return self.summaries.pop(0)
        env_id = draft.environment.env_id
        return _rollout_summary(env_id=env_id, matched=1, total=2)

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _FakeRegistry:
    root_dir: Path
    index_db_path: Path
    commit_results: list[EnvironmentRegistryCommitResult] = field(default_factory=list)
    committed_drafts: list[object] = field(default_factory=list)

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


def _rollout_summary(*, env_id: str, matched: int, total: int) -> EnvironmentRolloutSummary:
    runs = tuple(
        SimpleNamespace(
            reward_result=SimpleNamespace(
                status="matched" if index < matched else "em_mismatch"
            )
        )
        for index in range(total)
    )
    return EnvironmentRolloutSummary(
        env_id=env_id,
        db_id="sakila",
        planned_solver_runs=total,
        total_instances=1,
        total_solver_runs=total,
        matched_solver_runs=matched,
        runs=runs,
    )


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
        environment_orchestrator=_FakeEnvironmentOrchestrator(),
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
    assert summary.registry_committed_envs == 2
    assert summary.registry_duplicate_envs == 0
    assert summary.remaining_pairs == 0
    assert len(summary.generated_env_ids) == 2
    assert summary.committed_env_ids == summary.generated_env_ids
    assert summary.registry_root_dir == fake_registry.root_dir
    assert summary.registry_index_db_path == fake_registry.index_db_path
    assert summary.flow_id is not None
    assert summary.event_log_path == tmp_path / "events.jsonl"
    assert summary.steps[0].draft_env_id is not None
    assert summary.steps[0].registry_status == EnvironmentRegistryCommitStatus.COMMITTED
    assert not hasattr(summary.steps[0], "draft")
    event_lines = [
        json.loads(line)
        for line in summary.event_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert event_lines[0]["stage"] == "registry_run"
    assert event_lines[0]["status"] == "started"
    assert event_lines[-1]["stage"] == "registry_run"
    assert event_lines[-1]["status"] == "completed"
    assert any(event["stage"] == "quality_gate" for event in event_lines)
    assert any(event["stage"] == "registry_commit" for event in event_lines)
    assert any(event["stage"] == "checkpoint" for event in event_lines)
    assert all(event["flow_id"] == summary.flow_id for event in event_lines)
    assert created["sakila"].synthesize_calls == [
        ("sakila", CategoryTaxonomy.ASSIGNMENT, None),
        ("sakila", CategoryTaxonomy.ITINERARY, None),
    ]

    runner2 = SynthesisRegistryRunner(
        _config_with_run_db(tmp_path),
        runtime_factory=lambda _entry: pytest.fail("runtime should not be created on resume"),
        environment_registry=_FakeRegistry(
            root_dir=tmp_path / "environments",
            index_db_path=tmp_path / "environment_registry.db",
        ),
        environment_orchestrator=_FakeEnvironmentOrchestrator(),
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
    assert resumed.processed_pairs_after_run == 2
    assert resumed.generated_drafts == 0
    assert resumed.registry_committed_envs == 0
    assert resumed.registry_duplicate_envs == 0
    assert resumed.remaining_pairs == 0
    assert resumed.last_decision is None


@pytest.mark.asyncio
async def test_synthesis_registry_runner_stops_on_backoff_decision(tmp_path: Path) -> None:
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
        environment_orchestrator=_FakeEnvironmentOrchestrator(),
    )
    runner.orchestrator._runtimes["sakila"] = runtime

    try:
        summary = await runner.run_steps(
            [
                SynthesisDbRegistryEntry(
                    db_id="sakila",
                    categories=[CategoryTaxonomy.ASSIGNMENT],
                )
            ],
            max_steps=3,
            checkpoint_namespace="synthesis_backoff",
        )
    finally:
        await runner.close()

    assert summary.outcome == SynthesisRegistryRunOutcome.ALL_BACKED_OFF
    assert summary.generated_drafts == 0
    assert summary.registry_committed_envs == 0
    assert summary.remaining_pairs == 1
    assert summary.last_decision is not None
    assert summary.last_decision.status == SynthesisSelectionStatus.BACKOFF
    assert runtime.synthesize_calls == []


@pytest.mark.asyncio
async def test_synthesis_registry_runner_reports_max_steps_reached(tmp_path: Path) -> None:
    config = _config_with_run_db(tmp_path)
    fake_registry = _FakeRegistry(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    runner = SynthesisRegistryRunner(
        config,
        runtime_factory=lambda _entry: _FakeRuntime(),
        environment_registry=fake_registry,
        environment_orchestrator=_FakeEnvironmentOrchestrator(),
    )

    try:
        summary = await runner.run_steps(
            [
                SynthesisDbRegistryEntry(
                    db_id="sakila",
                    categories=[CategoryTaxonomy.ASSIGNMENT, CategoryTaxonomy.ITINERARY],
                )
            ],
            max_steps=1,
            checkpoint_namespace="synthesis_budget",
        )
    finally:
        await runner.close()

    assert summary.outcome == SynthesisRegistryRunOutcome.MAX_STEPS_REACHED
    assert summary.generated_drafts == 1
    assert summary.registry_committed_envs == 1
    assert summary.remaining_pairs == 1
    assert summary.last_decision is not None
    assert summary.last_decision.status == SynthesisSelectionStatus.READY


@pytest.mark.asyncio
async def test_synthesis_registry_runner_tracks_registry_duplicates(tmp_path: Path) -> None:
    config = _config_with_run_db(tmp_path)
    fake_registry = _FakeRegistry(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
        commit_results=[
            EnvironmentRegistryCommitResult(
                status=EnvironmentRegistryCommitStatus.DUPLICATE,
                env_id="env_existing",
                exact_signature="sha256:existing",
                difficulty_band=DifficultyBand.UNSET,
                filesystem_path=tmp_path / "environments" / "env_existing",
                duplicate_of_env_id="env_existing",
            )
        ],
    )
    runner = SynthesisRegistryRunner(
        config,
        runtime_factory=lambda _entry: _FakeRuntime(),
        environment_registry=fake_registry,
        environment_orchestrator=_FakeEnvironmentOrchestrator(),
    )

    try:
        summary = await runner.run_steps(
            [
                SynthesisDbRegistryEntry(
                    db_id="sakila",
                    categories=[CategoryTaxonomy.ASSIGNMENT],
                )
            ],
            max_steps=1,
            checkpoint_namespace="synthesis_duplicate_commit",
        )
    finally:
        await runner.close()

    assert summary.generated_drafts == 1
    assert summary.registry_committed_envs == 0
    assert summary.registry_duplicate_envs == 1
    assert summary.duplicate_env_ids == ["env_existing"]
    assert summary.steps[0].registry_status == EnvironmentRegistryCommitStatus.DUPLICATE
    assert summary.steps[0].registry_env_id == "env_existing"


@pytest.mark.asyncio
async def test_synthesis_registry_runner_rejects_quality_gate_before_registry_commit(
    tmp_path: Path,
) -> None:
    config = _config_with_run_db(tmp_path)
    fake_registry = _FakeRegistry(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    quality_orchestrator = _FakeEnvironmentOrchestrator(
        summaries=[_rollout_summary(env_id="env_sakila_assignment_1", matched=0, total=2)]
    )
    runner = SynthesisRegistryRunner(
        config,
        runtime_factory=lambda _entry: _FakeRuntime(),
        environment_registry=fake_registry,
        environment_orchestrator=quality_orchestrator,
    )

    try:
        summary = await runner.run_steps(
            [
                SynthesisDbRegistryEntry(
                    db_id="sakila",
                    categories=[CategoryTaxonomy.ASSIGNMENT],
                )
            ],
            max_steps=1,
            checkpoint_namespace="synthesis_quality_reject",
        )
    finally:
        await runner.close()

    assert summary.outcome == SynthesisRegistryRunOutcome.MAX_STEPS_REACHED
    assert summary.generated_drafts == 1
    assert summary.quality_accepted_envs == 0
    assert summary.quality_rejected_envs == 1
    assert summary.registry_committed_envs == 0
    assert summary.registry_duplicate_envs == 0
    assert summary.processed_pairs_after_run == 0
    assert summary.remaining_pairs == 1
    assert summary.quality_rejected_env_ids == ["env_sakila_assignment_1"]
    assert summary.steps[0].quality_gate_status == "reject_too_hard"
    assert summary.steps[0].registry_status is None
    assert fake_registry.committed_drafts == []


@pytest.mark.asyncio
async def test_synthesis_registry_runner_records_quality_metrics_on_accepted_draft(
    tmp_path: Path,
) -> None:
    config = _config_with_run_db(tmp_path)
    fake_registry = _FakeRegistry(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    quality_orchestrator = _FakeEnvironmentOrchestrator(
        summaries=[_rollout_summary(env_id="env_sakila_assignment_1", matched=1, total=2)]
    )
    runner = SynthesisRegistryRunner(
        config,
        runtime_factory=lambda _entry: _FakeRuntime(),
        environment_registry=fake_registry,
        environment_orchestrator=quality_orchestrator,
    )

    try:
        summary = await runner.run_steps(
            [
                SynthesisDbRegistryEntry(
                    db_id="sakila",
                    categories=[CategoryTaxonomy.ASSIGNMENT],
                )
            ],
            max_steps=1,
            checkpoint_namespace="synthesis_quality_accept",
        )
    finally:
        await runner.close()

    assert summary.quality_accepted_envs == 1
    assert summary.quality_rejected_envs == 0
    assert summary.registry_committed_envs == 1
    committed = fake_registry.committed_drafts[0]
    assert committed.environment.status == EnvironmentStatus.ACCEPTED
    assert committed.environment.quality_metrics.shadow_disagreement_rate == 0.0
    assert committed.environment.quality_metrics.solver_pass_rate == 0.5
    assert committed.environment.quality_metrics.solver_ci_low is not None
    assert committed.environment.quality_metrics.solver_ci_high is not None


@pytest.mark.asyncio
async def test_synthesis_registry_runner_rejects_cross_instance_inconsistency_before_rollout(
    tmp_path: Path,
) -> None:
    config = _config_with_run_db(tmp_path)
    fake_registry = _FakeRegistry(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    quality_orchestrator = _FakeEnvironmentOrchestrator()

    @dataclass(slots=True)
    class _BadCrossInstanceRuntime:
        async def category_status(
            self,
            *,
            db_id: str | None = None,
        ) -> dict[CategoryTaxonomy, SynthesisCategoryStatus]:
            return {}

        async def synthesize_environment_draft(
            self,
            *,
            db_id: str,
            requested_category: CategoryTaxonomy,
            graph: object | None = None,
        ) -> object:
            draft = _sample_draft(
                tmp_env_id=f"env_{db_id}_{requested_category.value}_bad_cross_instance",
                db_id=db_id,
                category=requested_category,
                created_at=datetime.now(timezone.utc),
            )
            environment = draft.environment.model_copy(
                update={
                    "cross_instance_set": draft.environment.cross_instance_set.model_copy(
                        update={"minimum_required": 2}
                    )
                }
            )
            return draft.model_copy(update={"environment": environment})

        async def close(self) -> None:
            return None

    runner = SynthesisRegistryRunner(
        config,
        runtime_factory=lambda _entry: _BadCrossInstanceRuntime(),
        environment_registry=fake_registry,
        environment_orchestrator=quality_orchestrator,
    )

    try:
        summary = await runner.run_steps(
            [
                SynthesisDbRegistryEntry(
                    db_id="sakila",
                    categories=[CategoryTaxonomy.ASSIGNMENT],
                )
            ],
            max_steps=1,
            checkpoint_namespace="synthesis_cross_instance_reject",
        )
    finally:
        await runner.close()

    assert summary.outcome == SynthesisRegistryRunOutcome.MAX_STEPS_REACHED
    assert summary.generated_drafts == 1
    assert summary.quality_rejected_envs == 1
    assert summary.registry_committed_envs == 0
    assert summary.remaining_pairs == 1
    assert summary.steps[0].quality_gate_status == "reject_cross_instance"
    assert "insufficient_instances" in summary.steps[0].cross_instance_error_codes
    assert fake_registry.committed_drafts == []
    assert quality_orchestrator.run_calls == []


@pytest.mark.asyncio
async def test_synthesis_registry_runner_preserves_category_order_across_checkpoint_shrinks(
    tmp_path: Path,
) -> None:
    config = _config_with_run_db(tmp_path)
    created: dict[str, _FakeRuntime] = {}

    def _factory(entry: SynthesisDbRegistryEntry) -> _FakeRuntime:
        runtime = _FakeRuntime()
        created[entry.db_id] = runtime
        return runtime

    fake_registry = _FakeRegistry(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    runner = SynthesisRegistryRunner(
        config,
        runtime_factory=_factory,
        environment_registry=fake_registry,
        environment_orchestrator=_FakeEnvironmentOrchestrator(),
    )
    try:
        summary = await runner.run_steps(
            [
                SynthesisDbRegistryEntry(
                    db_id="sakila",
                    categories=[
                        CategoryTaxonomy.ASSIGNMENT,
                        CategoryTaxonomy.ITINERARY,
                        CategoryTaxonomy.BUNDLE_SELECTION,
                    ],
                )
            ],
            max_steps=3,
            checkpoint_namespace="synthesis_rr_order",
        )
    finally:
        await runner.close()

    assert summary.outcome == SynthesisRegistryRunOutcome.COMPLETED_ALL
    assert created["sakila"].synthesize_calls == [
        ("sakila", CategoryTaxonomy.ASSIGNMENT, None),
        ("sakila", CategoryTaxonomy.ITINERARY, None),
        ("sakila", CategoryTaxonomy.BUNDLE_SELECTION, None),
    ]


def test_load_synthesis_registry_accepts_database_and_domain_overrides(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            [
                {
                    "db_id": "northwind",
                    "categories": ["assignment"],
                    "database": {
                        "dsn": "postgresql://northwind:northwind@127.0.0.1:5434/northwind",
                        "schema_allowlist": ["public"],
                        "readonly_role": "rlvr_reader",
                    },
                    "domain": {
                        "name": "sales_ops",
                        "language": "ko",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    [entry] = load_synthesis_registry(registry_path)

    assert entry.db_id == "northwind"
    assert entry.categories == [CategoryTaxonomy.ASSIGNMENT]
    assert entry.database == DatabaseConfig(
        dsn="postgresql://northwind:northwind@127.0.0.1:5434/northwind",
        schema_allowlist=["public"],
        readonly_role="rlvr_reader",
    )
    assert entry.domain == DomainConfig(name="sales_ops", language="ko")


def test_load_synthesis_registry_rejects_duplicate_db_ids(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            [
                {"db_id": "sakila", "categories": ["assignment"]},
                {"db_id": "sakila", "categories": ["itinerary"]},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate db_id"):
        load_synthesis_registry(registry_path)


def test_synthesis_registry_runner_default_runtime_factory_applies_entry_overrides(
    tmp_path: Path,
) -> None:
    config = _config_with_run_db(tmp_path)
    runner = SynthesisRegistryRunner(config)

    runtime = runner._build_runtime(
        SynthesisDbRegistryEntry(
            db_id="northwind",
            categories=[CategoryTaxonomy.ASSIGNMENT],
            database=DatabaseConfig(
                dsn="postgresql://northwind:northwind@127.0.0.1:5434/northwind",
                schema_allowlist=["public"],
                readonly_role="rlvr_reader",
            ),
            domain=DomainConfig(name="sales_ops", language="ko"),
        )
    )

    assert runtime.config.database.dsn == "postgresql://northwind:northwind@127.0.0.1:5434/northwind"
    assert runtime.config.domain.name == "sales_ops"


@pytest.mark.asyncio
async def test_synthesis_registry_runner_rejects_duplicate_db_ids(tmp_path: Path) -> None:
    config = _config_with_run_db(tmp_path)
    runner = SynthesisRegistryRunner(
        config,
        runtime_factory=lambda _entry: _FakeRuntime(),
        environment_registry=_FakeRegistry(
            root_dir=tmp_path / "environments",
            index_db_path=tmp_path / "environment_registry.db",
        ),
        environment_orchestrator=_FakeEnvironmentOrchestrator(),
    )

    try:
        with pytest.raises(ValueError, match="duplicate db_id"):
            await runner.run_steps(
                [
                    SynthesisDbRegistryEntry(
                        db_id="sakila",
                        categories=[CategoryTaxonomy.ASSIGNMENT],
                    ),
                    SynthesisDbRegistryEntry(
                        db_id="sakila",
                        categories=[CategoryTaxonomy.ITINERARY],
                    ),
                ],
                max_steps=1,
                checkpoint_namespace="synthesis_duplicate",
            )
    finally:
        await runner.close()
