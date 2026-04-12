from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig, SolverModelConfig
from rl_task_foundry.pipeline.environment_orchestrator import (
    EnvironmentOrchestrator,
    EnvironmentRolloutBundle,
)
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.synthesis.atomic_tools import (
    AtomicToolBundle,
    AtomicToolDefinition,
    AtomicToolFamily,
    AtomicToolResultMode,
)
from rl_task_foundry.synthesis.contracts import (
    CategoryTaxonomy,
    CrossInstanceSet,
    EnvironmentContract,
    EnvironmentQualityMetrics,
    EnvironmentStatus,
    InstanceSpaceContract,
    MaterializedFactsSchema,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    RolloutConstraintsContract,
    ShadowVerifierContract,
    SolutionContract,
    TaskContract,
    VerifierContract,
    AnchorQueryContract,
)
from rl_task_foundry.synthesis.registration_policy import ArtifactKind
from rl_task_foundry.synthesis.registration_runner import (
    ArtifactRegistrationResult,
    GeneratedArtifactBundle,
    RegistrationArtifactName,
    RegistrationBundleDiagnostics,
    RegistrationBundleReport,
    RegistrationBundleStatus,
)
from rl_task_foundry.synthesis.runtime import (
    MaterializedCanonicalAnswerRecord,
    MaterializedInstanceRecord,
    SynthesisEnvironmentDraft,
    SynthesisSelfConsistencyDiagnostics,
)


def _config(tmp_path):
    config = load_config("rl_task_foundry.yaml")
    output = OutputConfig(
        run_db_path=tmp_path / "run.db",
        accepted_jsonl_path=tmp_path / "accepted.jsonl",
        rejected_jsonl_path=tmp_path / "rejected.jsonl",
        events_jsonl_path=tmp_path / "events.jsonl",
        traces_dir=tmp_path / "traces",
    )
    models = config.models.model_copy(
        update={
            "solvers": [
                SolverModelConfig(
                    solver_id="solver_a",
                    provider="codex_oauth",
                    model="gpt-5.4-mini",
                    replicas=1,
                )
            ]
        }
    )
    return config.model_copy(update={"output": output, "models": models}, deep=True)


def _sample_atomic_tool_bundle(db_id: str = "sakila") -> AtomicToolBundle:
    tool = AtomicToolDefinition(
        name="count_customer",
        family=AtomicToolFamily.T2_BOUNDED_ENUMERATION,
        description="Count customer rows.",
        params_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        returns_schema={"type": "integer"},
        sql="SELECT 7",
        result_mode=AtomicToolResultMode.SCALAR,
        semantic_key="public.customer:count",
    )
    source = (
        '"""Atomic tools."""\n'
        "from __future__ import annotations\n\n"
        f"DB_ID = {db_id!r}\n"
        "MAX_BATCH_VALUES = 128\n\n"
        "async def count_customer(conn):\n"
        "    return await conn.fetchval('SELECT 7')\n"
    )
    return AtomicToolBundle(db_id=db_id, tools=[tool], source=source)


def _sample_environment() -> EnvironmentContract:
    output_schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[OutputFieldContract(name="city", type=OutputFieldType.STRING)],
        ),
        primary_output_format="json_object",
    )
    task = TaskContract(
        question="도시를 알려 주세요.",
        category=CategoryTaxonomy.ASSIGNMENT,
        output_schema=output_schema,
    )
    return EnvironmentContract(
        env_id="env_assignment_city",
        db_id="sakila",
        domain="customer_support",
        category=CategoryTaxonomy.ASSIGNMENT,
        atomic_tool_set_ref="db://sakila",
        difficulty_vector={},
        created_at=datetime(2026, 4, 12, tzinfo=timezone.utc),
        generator_version="test-version",
        tool_signature="sha256:tool",
        task_signature="sha256:task",
        verifier_signature="sha256:verifier",
        status=EnvironmentStatus.ACCEPTED,
        quality_metrics=EnvironmentQualityMetrics(self_consistency_pass=True),
        rollout_constraints=RolloutConstraintsContract(
            max_turns=3,
            max_episode_duration_ms=10000,
            max_tool_rows=100,
        ),
        task=task,
        solution=SolutionContract(),
        verifier=VerifierContract(facts_schema=MaterializedFactsSchema()),
        shadow_verifier=ShadowVerifierContract(facts_schema=MaterializedFactsSchema()),
        instance_space=InstanceSpaceContract(
            anchor_query=AnchorQueryContract(
                sql="SELECT customer_id FROM customer ORDER BY customer_id",
                outputs=["customer_id"],
            )
        ),
        cross_instance_set=CrossInstanceSet(minimum_required=1),
    )


def _sample_bundle() -> EnvironmentRolloutBundle:
    return EnvironmentRolloutBundle(
        environment=_sample_environment(),
        atomic_tool_bundle=_sample_atomic_tool_bundle(),
        instances=(
            MaterializedInstanceRecord(
                instance_id="instance_0001",
                rendered_user_prompt=(
                    "도시를 알려 주세요.\n\n"
                    "Submit Result Format:\n"
                    '{"type":"object","properties":{"city":{"type":"string"}}}\n'
                ),
                params={},
                anchor_values={},
            ),
        ),
        canonical_answers=(
            MaterializedCanonicalAnswerRecord(
                instance_id="instance_0001",
                canonical_answer={"city": "Seoul"},
                canonical_answer_json='{"city":"Seoul"}',
                solution_fingerprint="sha256:answer",
            ),
        ),
    )


def _sample_draft() -> SynthesisEnvironmentDraft:
    bundle = _sample_bundle()
    return SynthesisEnvironmentDraft.model_construct(
        created_at=datetime(2026, 4, 12, tzinfo=timezone.utc),
        db_id=bundle.environment.db_id,
        requested_category=bundle.environment.category,
        schema_summary={},
        selected_category=bundle.environment.category,
        environment=bundle.environment,
        atomic_tool_bundle=bundle.atomic_tool_bundle,
        artifacts=GeneratedArtifactBundle(
            solution_source="def solve(tools):\n    return {'city': 'Seoul'}\n",
            verifier_source="def compute_canonical_answer(tools):\n    return {'city': 'Seoul'}\n",
            shadow_verifier_source=(
                "def compute_canonical_answer(tools):\n    return {'city': 'Seoul'}\n"
            ),
        ),
            registration_report=RegistrationBundleReport(
                status=RegistrationBundleStatus.PASSED,
                tool=ArtifactRegistrationResult(
                    artifact_name=RegistrationArtifactName.TOOL,
                    artifact_kind=ArtifactKind.TOOL_MODULE,
                ),
                tool_self_test=ArtifactRegistrationResult(
                    artifact_name=RegistrationArtifactName.TOOL_SELF_TEST,
                    artifact_kind=ArtifactKind.TOOL_SELF_TEST_MODULE,
                ),
                solution=ArtifactRegistrationResult(
                    artifact_name=RegistrationArtifactName.SOLUTION,
                    artifact_kind=ArtifactKind.SOLUTION_MODULE,
                ),
                verifier=ArtifactRegistrationResult(
                    artifact_name=RegistrationArtifactName.VERIFIER,
                    artifact_kind=ArtifactKind.VERIFIER_MODULE,
                ),
                shadow_verifier=ArtifactRegistrationResult(
                    artifact_name=RegistrationArtifactName.SHADOW_VERIFIER,
                    artifact_kind=ArtifactKind.SHADOW_VERIFIER_MODULE,
                ),
            ),
        registration_diagnostics=RegistrationBundleDiagnostics.model_construct(
            status=RegistrationBundleStatus.PASSED,
            failing_artifacts=[],
            error_codes=[],
            weak_signal_codes=[],
            tool=None,
            tool_self_test=None,
            solution=None,
            verifier=None,
            shadow_verifier=None,
        ),
        self_consistency_diagnostics=SynthesisSelfConsistencyDiagnostics(passed=True),
        instances=list(bundle.instances),
        canonical_answers=list(bundle.canonical_answers),
        self_consistency_attempts=[],
        stage_results=[],
        memory=[],
        tool_traces=[],
        provider_status={},
    )


@pytest.mark.asyncio
async def test_environment_orchestrator_runs_draft_and_scores_reward(tmp_path):
    config = _config(tmp_path)
    draft = _sample_draft()
    runtime_calls: list[tuple[object, int]] = []
    runtime_inputs: dict[str, object] = {}

    class _FakeRuntime:
        async def run(self, episode, *, replica_index: int):
            runtime_calls.append((episode, replica_index))
            return SolverResult(
                task_id=episode.task_id,
                solver_id="solver_a",
                provider="codex_oauth",
                model="gpt-5.4-mini",
                replica_index=replica_index,
                transcript_ref="memory://transcript",
                tool_trace_ref="memory://tools",
                raw_output_text='{"city":"Seoul"}',
                structured_output={"city": "Seoul"},
                status="completed",
                termination_reason="submitted",
            )

    def _runtime_factory(solver_config, provider_config, environment, tool_definitions, tool_executors):
        runtime_inputs["environment"] = environment
        runtime_inputs["tool_definitions"] = tool_definitions
        runtime_inputs["tool_executor_names"] = sorted(tool_executors)
        return _FakeRuntime()

    orchestrator = EnvironmentOrchestrator(
        config,
        runtime_factory=_runtime_factory,
        tool_executor_factory=lambda _bundle: {"count_customer": lambda _kwargs: 7},
    )

    summary = await orchestrator.run_draft(draft)

    assert summary.env_id == draft.environment.env_id
    assert summary.total_instances == 1
    assert summary.total_solver_runs == 1
    assert summary.matched_solver_runs == 1
    assert summary.pass_rate == 1.0
    assert runtime_calls[0][0].rendered_user_prompt == draft.instances[0].rendered_user_prompt
    assert runtime_calls[0][0].environment == draft.environment
    assert runtime_inputs["environment"] == draft.environment
    assert runtime_inputs["tool_definitions"] == draft.atomic_tool_bundle.actor_tool_definitions()
    assert runtime_inputs["tool_executor_names"] == ["count_customer"]


@pytest.mark.asyncio
async def test_environment_orchestrator_rejects_missing_canonical_answer(tmp_path):
    config = _config(tmp_path)
    bundle = EnvironmentRolloutBundle(
        environment=_sample_environment(),
        atomic_tool_bundle=_sample_atomic_tool_bundle(),
        instances=(
            MaterializedInstanceRecord(
                instance_id="instance_0001",
                rendered_user_prompt="도시를 알려 주세요.",
                params={},
                anchor_values={},
            ),
        ),
        canonical_answers=(),
    )

    orchestrator = EnvironmentOrchestrator(
        config,
        runtime_factory=lambda *_args: SimpleNamespace(run=None),
        tool_executor_factory=lambda _bundle: {"count_customer": lambda _kwargs: 7},
    )

    with pytest.raises(ValueError, match="missing canonical answer"):
        await orchestrator.run_bundle(bundle)


@pytest.mark.asyncio
async def test_environment_orchestrator_uses_environment_max_turns_in_default_backend(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    bundle = _sample_bundle()
    captured: dict[str, object] = {}

    class _FakeBackend:
        def __init__(self, **kwargs):
            captured["runtime_config"] = kwargs["runtime_config"]
            captured["tool_definitions"] = kwargs["tool_definitions"]
            captured["tool_executor_names"] = sorted(kwargs["tool_executors"])

        async def run(self, episode, *, replica_index: int):
            return SolverResult(
                task_id=episode.task_id,
                solver_id="solver_a",
                provider="codex_oauth",
                model="gpt-5.4-mini",
                replica_index=replica_index,
                transcript_ref="memory://transcript",
                tool_trace_ref="memory://tools",
                raw_output_text='{"city":"Seoul"}',
                structured_output={"city": "Seoul"},
                status="completed",
                termination_reason="submitted",
            )

    monkeypatch.setattr(
        "rl_task_foundry.pipeline.environment_orchestrator.OpenAIAgentsSolverBackend",
        _FakeBackend,
    )

    orchestrator = EnvironmentOrchestrator(
        config,
        tool_executor_factory=lambda _bundle: {"count_customer": lambda _kwargs: 7},
    )
    summary = await orchestrator.run_bundle(bundle)

    assert summary.matched_solver_runs == 1
    assert captured["runtime_config"].max_turns == bundle.environment.rollout_constraints.max_turns
    assert captured["tool_definitions"] == bundle.atomic_tool_bundle.actor_tool_definitions()
    assert captured["tool_executor_names"] == ["count_customer"]


@pytest.mark.asyncio
async def test_environment_orchestrator_materializes_atomic_bundle_and_builds_db_executor(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    bundle = _sample_bundle()
    captured: dict[str, object] = {}

    class _FakeConnection:
        async def fetchval(self, sql, *args):
            captured["sql"] = sql
            captured["args"] = args
            return 7

    class _FakeSolverConnectionContext:
        async def __aenter__(self):
            return _FakeConnection()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class _FakePools:
        def solver_connection(self):
            return _FakeSolverConnectionContext()

        async def close(self):
            return None

    async def _fake_create(_database_config):
        return _FakePools()

    class _FakeRuntime:
        async def run(self, episode, *, replica_index: int):
            payload = await captured["tool_executors"]["count_customer"]({})
            captured["tool_result"] = payload
            return SolverResult(
                task_id=episode.task_id,
                solver_id="solver_a",
                provider="codex_oauth",
                model="gpt-5.4-mini",
                replica_index=replica_index,
                transcript_ref="memory://transcript",
                tool_trace_ref="memory://tools",
                raw_output_text='{"city":"Seoul"}',
                structured_output={"city": "Seoul"},
                status="completed",
                termination_reason="submitted",
            )

    def _runtime_factory(_solver_config, _provider_config, _environment, _tool_definitions, tool_executors):
        captured["tool_executors"] = tool_executors
        return _FakeRuntime()

    monkeypatch.setattr(
        "rl_task_foundry.pipeline.environment_orchestrator.DatabasePools.create",
        _fake_create,
    )

    orchestrator = EnvironmentOrchestrator(config, runtime_factory=_runtime_factory)
    summary = await orchestrator.run_bundle(bundle)

    assert summary.matched_solver_runs == 1
    assert captured["tool_result"] == 7
    assert captured["sql"] == "SELECT 7"
    assert captured["args"] == ()
    assert (config.output.traces_dir.parent / "databases" / "sakila" / "atomic_tools.py").exists()
    await orchestrator.close()


def test_environment_rollout_bundle_from_draft_preserves_environment_contract():
    draft = _sample_draft()
    bundle = EnvironmentRolloutBundle.from_draft(draft)

    assert bundle.environment == draft.environment
    assert bundle.atomic_tool_bundle == draft.atomic_tool_bundle
    assert bundle.instances == tuple(draft.instances)
    assert bundle.canonical_answers == tuple(draft.canonical_answers)


def test_environment_pipeline_orchestrator_module_has_zero_legacy_imports() -> None:
    from rl_task_foundry.pipeline import environment_orchestrator as orchestrator_module

    module_source = Path(orchestrator_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(module_source)
    imported_modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert all(not name.startswith("rl_task_foundry.tasks") for name in imported_modules)
    assert all(not name.startswith("rl_task_foundry.tools") for name in imported_modules)
    assert all(not name.startswith("rl_task_foundry.truth") for name in imported_modules)
