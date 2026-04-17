from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig
from rl_task_foundry.pipeline.solver_orchestrator import SolverOrchestrator
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.synthesis.bundle_exporter import TaskBundleExporter
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy, OutputFieldType
from rl_task_foundry.synthesis.proof_environment import (
    PROOF_DB_ID,
    PROOF_TASK_ID,
    ProofTaskRunner,
    build_proof_task_draft,
    write_proof_fixture_sql,
)
from rl_task_foundry.synthesis.task_registry import (
    TaskRegistryCommitStatus,
    TaskRegistryWriter,
)


def _config(tmp_path: Path):
    config = load_config("rl_task_foundry.yaml")
    output = OutputConfig(
        run_db_path=tmp_path / "run.db",
        traces_dir=tmp_path / "traces",
    )
    return config.model_copy(update={"output": output}, deep=True)


def test_build_proof_task_draft_is_compositional() -> None:
    config = load_config("rl_task_foundry.yaml")
    draft = build_proof_task_draft(
        config=config, created_at=datetime(2026, 4, 12, tzinfo=timezone.utc)
    )

    root = draft.task_bundle.task.output_schema.root
    assert draft.task_bundle.db_id == PROOF_DB_ID
    assert draft.task_bundle.task_id == PROOF_TASK_ID
    assert draft.task_bundle.topic == CategoryTaxonomy.ITINERARY
    assert draft.task_bundle.atomic_tool_set_ref == f"db://{PROOF_DB_ID}"
    assert root.type is OutputFieldType.LIST
    assert root.sort_key == ("day",)
    assert root.unique_elements is True
    assert len(draft.task_bundle.task.constraint_summary) == 4
    assert len(draft.canonical_answer) == 3
    assert draft.task_bundle.rollout_constraints.max_turns == config.solver_runtime.max_turns
    assert (
        draft.task_bundle.rollout_constraints.max_episode_duration_ms
        == config.database.statement_timeout_ms * config.solver_runtime.max_turns
    )
    assert (
        draft.task_bundle.rollout_constraints.max_tool_rows
        == config.atomic_tools.bounded_result_limit
    )
    assert "# Submit Result Format" in draft.rendered_user_prompt
    assert "연속된 day의 city는 인접한 지역이어야 합니다." in draft.rendered_user_prompt


def test_write_proof_fixture_sql_writes_schema_and_seed(tmp_path: Path) -> None:
    files = write_proof_fixture_sql(tmp_path / "fixture_db")

    assert files.schema_path.exists()
    assert files.seed_path.exists()
    assert "CREATE TABLE IF NOT EXISTS proof_cities" in files.schema_path.read_text(
        encoding="utf-8"
    )
    assert "INSERT INTO proof_activities" in files.seed_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_proof_task_runner_commits_and_exports_bundle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    writer = TaskRegistryWriter(
        root_dir=tmp_path / "registry" / "tasks",
        index_db_path=tmp_path / "registry" / "task_registry.db",
    )
    exporter = TaskBundleExporter(
        registry=writer,
        materializer=writer.atomic_tool_materializer,
    )

    class _FakeRuntime:
        call_count = 0

        async def run(self, episode):
            matched = self.__class__.call_count % 2 == 0
            self.__class__.call_count += 1
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
                transcript_ref="memory://transcript",
                tool_trace_ref="memory://tools",
                raw_output_text=json.dumps(payload, ensure_ascii=False),
                structured_output=None,
                status="completed",
                termination_reason="submitted",
            )

    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda *_args: _FakeRuntime(),
        tool_executor_factory=lambda _bundle: {},
    )
    runner = ProofTaskRunner(
        config,
        solver_orchestrator=orchestrator,
        registry=writer,
        exporter=exporter,
    )
    try:
        summary = await runner.run(tmp_path / "proof_output")
    finally:
        await runner.close()

    assert summary.quality_gate_status == "accept"
    assert summary.flow_id is not None
    assert (
        summary.phase_monitor_log_path
        == tmp_path / "proof_output" / "debug" / "phase_monitors.jsonl"
    )
    assert summary.registry_status is TaskRegistryCommitStatus.COMMITTED
    assert summary.bundle_root is not None
    assert (summary.fixture_sql_root / "schema.sql").exists()
    assert (summary.bundle_root / "databases" / PROOF_DB_ID / "atomic_tools.py").exists()
    assert (summary.bundle_root / "tasks" / PROOF_TASK_ID / "task.yaml").exists()
    assert writer.task_count(db_id=PROOF_DB_ID) == 1
    phase_monitor_lines = [
        json.loads(line)
        for line in summary.phase_monitor_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["phase"] for line in phase_monitor_lines] == [
        "draft_build",
        "rollout",
        "quality_gate",
        "registry_commit",
        "bundle_export",
    ]


@pytest.mark.asyncio
async def test_proof_task_runner_skips_commit_when_quality_gate_rejects(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    writer = TaskRegistryWriter(
        root_dir=tmp_path / "registry" / "tasks",
        index_db_path=tmp_path / "registry" / "task_registry.db",
    )

    class _FakeRuntime:
        async def run(self, episode):
            return SolverResult(
                task_id=episode.task_id,
                solver_id="solver_a",
                provider="codex_oauth",
                model="gpt-5.4-mini",
                transcript_ref="memory://transcript",
                tool_trace_ref="memory://tools",
                raw_output_text='[{"day":1,"city":"Gangneung","lodging":"X","activity":"Y","total_cost":500}]',
                structured_output=None,
                status="completed",
                termination_reason="submitted",
            )

    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda *_args: _FakeRuntime(),
        tool_executor_factory=lambda _bundle: {},
    )
    runner = ProofTaskRunner(
        config,
        solver_orchestrator=orchestrator,
        registry=writer,
        exporter=TaskBundleExporter(
            registry=writer,
            materializer=writer.atomic_tool_materializer,
        ),
    )
    try:
        summary = await runner.run(tmp_path / "proof_output")
    finally:
        await runner.close()

    assert summary.quality_gate_status == "reject_too_hard"
    assert summary.phase_monitor_log_path is not None
    assert summary.registry_status is None
    assert summary.bundle_root is None
    assert writer.task_count(db_id=PROOF_DB_ID) == 0


def test_proof_environment_module_has_no_legacy_imports() -> None:
    module_path = Path("src/rl_task_foundry/synthesis/proof_environment.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "task_registry" in node.module or "registry" not in node.module
            assert "solver_orchestrator" in node.module or "orchestrator" not in node.module
