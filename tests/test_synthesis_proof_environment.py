from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig
from rl_task_foundry.pipeline.environment_orchestrator import EnvironmentOrchestrator
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.synthesis.bundle_exporter import EnvironmentBundleExporter
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy, OutputFieldType
from rl_task_foundry.synthesis.environment_registry import (
    EnvironmentRegistryCommitStatus,
    EnvironmentRegistryWriter,
)
from rl_task_foundry.synthesis.proof_environment import (
    PROOF_DB_ID,
    PROOF_ENV_ID,
    ProofEnvironmentRunner,
    build_proof_environment_draft,
    write_proof_fixture_sql,
)


def _config(tmp_path: Path):
    config = load_config("rl_task_foundry.yaml")
    output = OutputConfig(
        run_db_path=tmp_path / "run.db",
        accepted_jsonl_path=tmp_path / "accepted.jsonl",
        rejected_jsonl_path=tmp_path / "rejected.jsonl",
        events_jsonl_path=tmp_path / "events.jsonl",
        traces_dir=tmp_path / "traces",
    )
    return config.model_copy(update={"output": output}, deep=True)


def test_build_proof_environment_draft_is_compositional() -> None:
    draft = build_proof_environment_draft(
        created_at=datetime(2026, 4, 12, tzinfo=timezone.utc)
    )

    root = draft.environment.task.output_schema.root
    assert draft.environment.db_id == PROOF_DB_ID
    assert draft.environment.env_id == PROOF_ENV_ID
    assert draft.environment.category is CategoryTaxonomy.ITINERARY
    assert draft.environment.atomic_tool_set_ref == f"db://{PROOF_DB_ID}"
    assert root.type is OutputFieldType.LIST
    assert root.sort_key == ("day",)
    assert root.unique_elements is True
    assert len(draft.environment.task.constraint_summary) == 4
    assert len(draft.canonical_answers) == 1
    assert len(draft.canonical_answers[0].canonical_answer) == 3
    assert "Submit Result Format:" in draft.instances[0].rendered_user_prompt
    assert "연속된 day의 city는 인접한 지역이어야 합니다." in draft.instances[0].rendered_user_prompt


def test_write_proof_fixture_sql_writes_schema_and_seed(tmp_path: Path) -> None:
    files = write_proof_fixture_sql(tmp_path / "fixture_db")

    assert files.schema_path.exists()
    assert files.seed_path.exists()
    assert "CREATE TABLE IF NOT EXISTS proof_cities" in files.schema_path.read_text(
        encoding="utf-8"
    )
    assert "INSERT INTO proof_activities" in files.seed_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_proof_environment_runner_commits_and_exports_bundle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "registry" / "environments",
        index_db_path=tmp_path / "registry" / "environment_registry.db",
    )
    exporter = EnvironmentBundleExporter(
        registry=writer,
        materializer=writer.atomic_tool_materializer,
    )

    class _FakeRuntime:
        async def run(self, episode, *, replica_index: int):
            matched = replica_index % 2 == 0
            payload = (
                [
                    {
                        "day": 1,
                        "city": "Seoul",
                        "lodging": "Seoul Station Stay",
                        "activity": "Han River Night Walk",
                        "total_cost": 180,
                    },
                    {
                        "day": 2,
                        "city": "Suwon",
                        "lodging": "Suwon Fortress Hotel",
                        "activity": "Fortress Loop Tour",
                        "total_cost": 160,
                    },
                    {
                        "day": 3,
                        "city": "Incheon",
                        "lodging": "Incheon Harbor Inn",
                        "activity": "Harbor Sunset Ferry",
                        "total_cost": 170,
                    },
                ]
                if matched
                else [
                    {
                        "day": 1,
                        "city": "Gangneung",
                        "lodging": "Seoul Station Stay",
                        "activity": "Han River Night Walk",
                        "total_cost": 180,
                    }
                ]
            )
            return SolverResult(
                task_id=episode.task_id,
                solver_id="solver_a",
                provider="codex_oauth",
                model="gpt-5.4-mini",
                replica_index=replica_index,
                transcript_ref="memory://transcript",
                tool_trace_ref="memory://tools",
                raw_output_text=json.dumps(payload, ensure_ascii=False),
                structured_output=None,
                status="completed",
                termination_reason="submitted",
            )

    orchestrator = EnvironmentOrchestrator(
        config,
        runtime_factory=lambda *_args: _FakeRuntime(),
        tool_executor_factory=lambda _bundle: {},
    )
    runner = ProofEnvironmentRunner(
        config,
        environment_orchestrator=orchestrator,
        registry=writer,
        exporter=exporter,
    )
    try:
        summary = await runner.run(tmp_path / "proof_output")
    finally:
        await runner.close()

    assert summary.quality_gate_status == "accept"
    assert summary.flow_id is not None
    assert summary.event_log_path == tmp_path / "proof_output" / "debug" / "pipeline_events.jsonl"
    assert summary.registry_status is EnvironmentRegistryCommitStatus.COMMITTED
    assert summary.bundle_root is not None
    assert (summary.fixture_sql_root / "schema.sql").exists()
    assert (summary.bundle_root / "databases" / PROOF_DB_ID / "atomic_tools.py").exists()
    assert (summary.bundle_root / "environments" / PROOF_ENV_ID / "environment.yaml").exists()
    assert writer.environment_count(db_id=PROOF_DB_ID) == 1
    event_lines = [
        json.loads(line)
        for line in summary.event_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["stage"] for event in event_lines] == [
        "proof_run",
        "fixture_sql",
        "draft",
        "cross_instance",
        "cross_instance",
        "rollout",
        "rollout",
        "quality_gate",
        "registry_commit",
        "registry_commit",
        "bundle_export",
        "bundle_export",
        "proof_run",
    ]


@pytest.mark.asyncio
async def test_proof_environment_runner_skips_commit_when_quality_gate_rejects(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "registry" / "environments",
        index_db_path=tmp_path / "registry" / "environment_registry.db",
    )

    class _FakeRuntime:
        async def run(self, episode, *, replica_index: int):
            return SolverResult(
                task_id=episode.task_id,
                solver_id="solver_a",
                provider="codex_oauth",
                model="gpt-5.4-mini",
                replica_index=replica_index,
                transcript_ref="memory://transcript",
                tool_trace_ref="memory://tools",
                raw_output_text='[{"day":1,"city":"Gangneung","lodging":"X","activity":"Y","total_cost":500}]',
                structured_output=None,
                status="completed",
                termination_reason="submitted",
            )

    orchestrator = EnvironmentOrchestrator(
        config,
        runtime_factory=lambda *_args: _FakeRuntime(),
        tool_executor_factory=lambda _bundle: {},
    )
    runner = ProofEnvironmentRunner(
        config,
        environment_orchestrator=orchestrator,
        registry=writer,
        exporter=EnvironmentBundleExporter(
            registry=writer,
            materializer=writer.atomic_tool_materializer,
        ),
    )
    try:
        summary = await runner.run(tmp_path / "proof_output")
    finally:
        await runner.close()

    assert summary.quality_gate_status == "reject_too_hard"
    assert summary.event_log_path is not None
    assert summary.registry_status is None
    assert summary.bundle_root is None
    assert writer.environment_count(db_id=PROOF_DB_ID) == 0


@pytest.mark.asyncio
async def test_proof_environment_runner_rejects_cross_instance_mismatch_before_rollout(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "registry" / "environments",
        index_db_path=tmp_path / "registry" / "environment_registry.db",
    )
    orchestrator = EnvironmentOrchestrator(
        config,
        runtime_factory=lambda *_args: pytest.fail("rollout should be skipped"),
        tool_executor_factory=lambda _bundle: {},
    )
    runner = ProofEnvironmentRunner(
        config,
        environment_orchestrator=orchestrator,
        registry=writer,
        exporter=EnvironmentBundleExporter(
            registry=writer,
            materializer=writer.atomic_tool_materializer,
        ),
    )

    def _bad_draft():
        draft = build_proof_environment_draft()
        environment = draft.environment.model_copy(
            update={
                "cross_instance_set": draft.environment.cross_instance_set.model_copy(
                    update={"minimum_required": 2}
                )
            }
        )
        return draft.model_copy(update={"environment": environment})

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "rl_task_foundry.synthesis.proof_environment.build_proof_environment_draft",
        _bad_draft,
    )
    try:
        summary = await runner.run(tmp_path / "proof_output")
    finally:
        monkeypatch.undo()
        await runner.close()

    assert summary.quality_gate_status == "reject_cross_instance"
    assert "insufficient_instances" in summary.cross_instance_error_codes
    assert summary.solver_pass_rate is None
    assert summary.event_log_path is not None
    assert summary.registry_status is None
    assert summary.bundle_root is None
    assert writer.environment_count(db_id=PROOF_DB_ID) == 0


def test_synthesis_proof_environment_module_has_zero_legacy_imports() -> None:
    from rl_task_foundry.synthesis import proof_environment as proof_module

    module_source = Path(proof_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(module_source)
    imported_modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert all(not name.startswith("rl_task_foundry.tools") for name in imported_modules)
    assert all(not name.startswith("rl_task_foundry.tasks") for name in imported_modules)
    assert all(not name.startswith("rl_task_foundry.truth") for name in imported_modules)
