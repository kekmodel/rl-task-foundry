from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import DatabaseConfig, DomainConfig
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.orchestrator import SynthesisDbRegistryEntry
from rl_task_foundry.synthesis.runner import (
    SynthesisRegistryRunOutcome,
    SynthesisRegistryRunSummary,
    SynthesisRegistryRunner,
    load_synthesis_registry,
)
from rl_task_foundry.synthesis.runtime import SynthesisCategoryStatus
from rl_task_foundry.synthesis.scheduler import SynthesisSelectionStatus


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
        return SimpleNamespace(
            environment=SimpleNamespace(
                env_id=f"env_{db_id}_{requested_category.value}_{len(self.synthesize_calls)}"
            ),
            created_at=datetime.now(timezone.utc),
        )

    async def close(self) -> None:
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

    runner = SynthesisRegistryRunner(config, runtime_factory=_factory)
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
    assert summary.remaining_pairs == 0
    assert len(summary.generated_env_ids) == 2
    assert summary.steps[0].draft_env_id is not None
    assert not hasattr(summary.steps[0], "draft")
    assert created["sakila"].synthesize_calls == [
        ("sakila", CategoryTaxonomy.ASSIGNMENT, None),
        ("sakila", CategoryTaxonomy.ITINERARY, None),
    ]

    runner2 = SynthesisRegistryRunner(
        _config_with_run_db(tmp_path),
        runtime_factory=lambda _entry: pytest.fail("runtime should not be created on resume"),
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
    runner = SynthesisRegistryRunner(config, runtime_factory=lambda _entry: runtime)
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
    assert summary.remaining_pairs == 1
    assert summary.last_decision is not None
    assert summary.last_decision.status == SynthesisSelectionStatus.BACKOFF
    assert runtime.synthesize_calls == []


@pytest.mark.asyncio
async def test_synthesis_registry_runner_reports_max_steps_reached(tmp_path: Path) -> None:
    config = _config_with_run_db(tmp_path)
    runner = SynthesisRegistryRunner(config, runtime_factory=lambda _entry: _FakeRuntime())

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
    assert summary.remaining_pairs == 1
    assert summary.last_decision is not None
    assert summary.last_decision.status == SynthesisSelectionStatus.READY


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

    runner = SynthesisRegistryRunner(config, runtime_factory=_factory)
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
    runner = SynthesisRegistryRunner(config, runtime_factory=lambda _entry: _FakeRuntime())

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
