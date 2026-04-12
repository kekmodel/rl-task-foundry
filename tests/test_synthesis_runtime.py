from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig
from rl_task_foundry.schema.graph import ColumnProfile, SchemaGraph, TableProfile
from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle
from rl_task_foundry.synthesis.contracts import (
    AnchorQueryContract,
    CategoryTaxonomy,
    ConstraintKind,
    ConstraintSummaryItem,
    DifficultyVectorContract,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    SolutionContract,
    TaskContract,
    build_difficulty_vector,
)
from rl_task_foundry.synthesis.runtime import (
    ArtifactGenerationOutput,
    CategoryInferenceOutput,
    GeneratedArtifactBundle,
    LabelConstructionOutput,
    ProposedEnvironmentDraft,
    SchemaExplorationOutput,
    SynthesisAgentRuntime,
    SynthesisArtifactGenerationError,
    SynthesisGenerationOutcome,
    SynthesisMemoryEntry,
    SynthesisPhase,
    SynthesisPhaseExecutionError,
    SynthesisStageResult,
    SynthesisToolTraceEntry,
    TaskSynthesisOutput,
)


def _sample_graph() -> SchemaGraph:
    customer_id = ColumnProfile(
        schema_name="public",
        table_name="customer",
        column_name="customer_id",
        data_type="integer",
        ordinal_position=1,
        is_nullable=False,
        visibility="user_visible",
        is_primary_key=True,
    )
    customer_name = ColumnProfile(
        schema_name="public",
        table_name="customer",
        column_name="customer_name",
        data_type="text",
        ordinal_position=2,
        is_nullable=False,
        visibility="user_visible",
    )
    return SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="customer",
                columns=[customer_id, customer_name],
                primary_key=("customer_id",),
                row_estimate=5,
            )
        ],
        edges=[],
    )


def _config_with_synthesis_output(tmp_path: Path):
    config = load_config("rl_task_foundry.yaml")
    output = OutputConfig(
        run_db_path=tmp_path / "run.db",
        accepted_jsonl_path=tmp_path / "accepted.jsonl",
        rejected_jsonl_path=tmp_path / "rejected.jsonl",
        events_jsonl_path=tmp_path / "events.jsonl",
        traces_dir=tmp_path / "traces",
    )
    return config.model_copy(update={"output": output}, deep=True)


def _sample_atomic_tool_bundle(db_id: str = "sakila") -> AtomicToolBundle:
    return AtomicToolBundle(
        db_id=db_id,
        tools=[],
        source="async def lookup_customer(conn, customer_id):\n    return {'customer_name': 'A'}\n",
    )


def _output_schema() -> OutputSchemaContract:
    return OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[
                OutputFieldContract(name="customer_name", type=OutputFieldType.STRING),
            ],
        ),
        primary_output_format="json_object",
    )


def _authoritative_task() -> TaskContract:
    return TaskContract(
        question="고객 이름을 반환해 주세요.",
        category=CategoryTaxonomy.ASSIGNMENT,
        output_schema=_output_schema(),
        constraint_summary=[
            ConstraintSummaryItem(
                key="single_customer",
                kind=ConstraintKind.CARDINALITY,
                summary="Return exactly one customer.",
            )
        ],
        difficulty_vector=build_difficulty_vector(search_cost=1.0),
        instance_parameters={"customer_id": 1},
    )


def _stage_result(
    *,
    phase: SynthesisPhase,
    payload: object,
    tool_traces: list[SynthesisToolTraceEntry] | None = None,
) -> SynthesisStageResult:
    return SynthesisStageResult(
        phase=phase,
        provider="codex_oauth",
        model="gpt-5.4-mini",
        payload=payload,
        memory_entry=SynthesisMemoryEntry(
            phase=phase,
            provider="codex_oauth",
            model="gpt-5.4-mini",
            summary=f"{phase.value} completed",
        ),
        tool_traces=tool_traces or [],
    )


def _schema_result(*, grounded: bool = True) -> SynthesisStageResult:
    tool_traces = (
        [
            SynthesisToolTraceEntry(
                phase=SynthesisPhase.SCHEMA_EXPLORATION,
                provider="codex_oauth",
                model="gpt-5.4-mini",
                tool_name="list_customer_ids",
                semantic_key="customer:list_ids",
            )
        ]
        if grounded
        else []
    )
    return _stage_result(
        phase=SynthesisPhase.SCHEMA_EXPLORATION,
        payload=SchemaExplorationOutput(
            domain_hypothesis="customer_support",
            candidate_categories=[CategoryTaxonomy.ASSIGNMENT],
            sample_observations=["Observed customer_id=1, customer_name='Alice'"],
            memory_summary="schema exploration completed",
        ),
        tool_traces=tool_traces,
    )


def _category_result() -> SynthesisStageResult:
    return _stage_result(
        phase=SynthesisPhase.CATEGORY_INFERENCE,
        payload=CategoryInferenceOutput(
            selected_category=CategoryTaxonomy.ASSIGNMENT,
            rationale="Matches assignment semantics.",
            memory_summary="category inference completed",
        ),
    )


def _label_result(
    *,
    canonical_answer_json: str = '{"customer_name":"Alice"}',
    difficulty_vector: DifficultyVectorContract | None = None,
) -> SynthesisStageResult:
    return _stage_result(
        phase=SynthesisPhase.LABEL_CONSTRUCTION,
        payload=LabelConstructionOutput(
            canonical_answer_json=canonical_answer_json,
            output_schema=_output_schema(),
            difficulty_vector=difficulty_vector or build_difficulty_vector(search_cost=1.0),
            instance_parameters={"customer_id": 1},
            label_summary="Alice is the only valid customer answer.",
            memory_summary="label construction completed",
        ),
    )


def _task_result() -> SynthesisStageResult:
    return _stage_result(
        phase=SynthesisPhase.TASK_SYNTHESIS,
        payload=TaskSynthesisOutput(
            question="고객 이름을 반환해 주세요.",
            constraint_summary=[
                ConstraintSummaryItem(
                    key="single_customer",
                    kind=ConstraintKind.CARDINALITY,
                    summary="Return exactly one customer.",
                )
            ],
            instance_space={
                "anchor_query": {
                    "sql": "SELECT customer_id FROM customer ORDER BY customer_id",
                    "outputs": ["customer_id"],
                }
            },
            memory_summary="task synthesis completed",
        ),
    )


def _artifact_result(
    *,
    solution_source: str = "def solve(tools):\n    return {'customer_name': 'Alice'}\n",
    task: TaskContract | None = None,
) -> SynthesisStageResult:
    proposed_task = task or _authoritative_task().model_copy(update={"question": "WRONG"})
    return _stage_result(
        phase=SynthesisPhase.ARTIFACT_GENERATION,
        payload=ArtifactGenerationOutput(
            proposed_environment=ProposedEnvironmentDraft(
                task=proposed_task,
                solution=SolutionContract(),
                instance_space={
                    "anchor_query": {
                        "sql": "SELECT customer_id FROM customer ORDER BY customer_id",
                        "outputs": ["customer_id"],
                    }
                },
            ),
            artifacts=GeneratedArtifactBundle(solution_source=solution_source),
            memory_summary="artifact generation completed",
        ),
    )


def _install_runtime_stubs(
    monkeypatch: pytest.MonkeyPatch,
    runtime: SynthesisAgentRuntime,
    *,
    results: dict[SynthesisPhase, SynthesisStageResult],
) -> None:
    async def _fake_run_phase(self, request):
        return results[request.phase]

    async def _fake_prime(self, *, db_id: str, bundle: AtomicToolBundle) -> None:
        del self, db_id, bundle

    async def _fake_reset(self, db_id: str, category: CategoryTaxonomy) -> None:
        del self, db_id, category

    async def _fake_ensure_bundle(self, *, db_id: str, graph: SchemaGraph) -> AtomicToolBundle:
        del self, graph
        return _sample_atomic_tool_bundle(db_id)

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_phase", _fake_run_phase)
    monkeypatch.setattr(SynthesisAgentRuntime, "_prime_phase_backends_with_atomic_tools", _fake_prime)
    monkeypatch.setattr(SynthesisAgentRuntime, "_reset_category_failure_state", _fake_reset)
    monkeypatch.setattr(SynthesisAgentRuntime, "_ensure_atomic_tool_bundle", _fake_ensure_bundle)


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_materializes_solution_only_label_first_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SynthesisAgentRuntime(_config_with_synthesis_output(tmp_path), phase_backends={})
    _install_runtime_stubs(
        monkeypatch,
        runtime,
        results={
            SynthesisPhase.SCHEMA_EXPLORATION: _schema_result(),
            SynthesisPhase.CATEGORY_INFERENCE: _category_result(),
            SynthesisPhase.LABEL_CONSTRUCTION: _label_result(),
            SynthesisPhase.TASK_SYNTHESIS: _task_result(),
            SynthesisPhase.ARTIFACT_GENERATION: _artifact_result(),
        },
    )

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        graph=_sample_graph(),
    )

    assert draft.environment.task.question == "고객 이름을 반환해 주세요."
    assert draft.artifacts.solution_source.startswith("def solve")
    assert len(draft.instances) == 1
    assert len(draft.canonical_answers) == 1
    assert draft.canonical_answers[0].canonical_answer_json == '{"customer_name":"Alice"}'
    assert [result.phase for result in draft.stage_results] == [
        SynthesisPhase.SCHEMA_EXPLORATION,
        SynthesisPhase.CATEGORY_INFERENCE,
        SynthesisPhase.LABEL_CONSTRUCTION,
        SynthesisPhase.TASK_SYNTHESIS,
        SynthesisPhase.ARTIFACT_GENERATION,
    ]
    assert [attempt.outcome for attempt in draft.generation_attempts] == [
        SynthesisGenerationOutcome.PASSED
    ]
    assert draft.generation_attempts[0].artifact_diagnostics is not None
    assert (
        "artifact_task_overridden_from_task_synthesis"
        in draft.generation_attempts[0].artifact_diagnostics.payload_repair_codes
    )
    assert "verifier" not in draft.environment.model_dump(mode="json")
    assert "shadow_verifier" not in draft.environment.model_dump(mode="json")


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_requires_grounded_schema_exploration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SynthesisAgentRuntime(_config_with_synthesis_output(tmp_path), phase_backends={})
    _install_runtime_stubs(
        monkeypatch,
        runtime,
        results={
            SynthesisPhase.SCHEMA_EXPLORATION: _schema_result(grounded=False),
            SynthesisPhase.CATEGORY_INFERENCE: _category_result(),
            SynthesisPhase.LABEL_CONSTRUCTION: _label_result(),
            SynthesisPhase.TASK_SYNTHESIS: _task_result(),
            SynthesisPhase.ARTIFACT_GENERATION: _artifact_result(),
        },
    )

    with pytest.raises(SynthesisPhaseExecutionError):
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_raises_when_solution_source_is_blank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_with_synthesis_output(tmp_path)
    config.synthesis.runtime.max_generation_attempts = 1
    runtime = SynthesisAgentRuntime(config, phase_backends={})
    _install_runtime_stubs(
        monkeypatch,
        runtime,
        results={
            SynthesisPhase.SCHEMA_EXPLORATION: _schema_result(),
            SynthesisPhase.CATEGORY_INFERENCE: _category_result(),
            SynthesisPhase.LABEL_CONSTRUCTION: _label_result(),
            SynthesisPhase.TASK_SYNTHESIS: _task_result(),
            SynthesisPhase.ARTIFACT_GENERATION: _artifact_result(solution_source="   "),
        },
    )

    with pytest.raises(SynthesisArtifactGenerationError) as exc_info:
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )

    assert exc_info.value.attempts[-1].outcome is SynthesisGenerationOutcome.ARTIFACT_INVALID
    assert exc_info.value.last_artifact_diagnostics is not None
    assert exc_info.value.last_artifact_diagnostics.error_codes == ["solution_source_missing"]


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_raises_when_label_cannot_be_canonicalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_with_synthesis_output(tmp_path)
    config.synthesis.runtime.max_generation_attempts = 1
    runtime = SynthesisAgentRuntime(config, phase_backends={})
    _install_runtime_stubs(
        monkeypatch,
        runtime,
        results={
            SynthesisPhase.SCHEMA_EXPLORATION: _schema_result(),
            SynthesisPhase.CATEGORY_INFERENCE: _category_result(),
            SynthesisPhase.LABEL_CONSTRUCTION: _label_result(canonical_answer_json='"wrong-shape"'),
            SynthesisPhase.TASK_SYNTHESIS: _task_result(),
            SynthesisPhase.ARTIFACT_GENERATION: _artifact_result(task=_authoritative_task()),
        },
    )

    with pytest.raises(SynthesisArtifactGenerationError) as exc_info:
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )

    assert exc_info.value.attempts[-1].outcome is SynthesisGenerationOutcome.ARTIFACT_INVALID
    assert exc_info.value.last_artifact_diagnostics is not None
    assert exc_info.value.last_artifact_diagnostics.error_codes == [
        "canonical_answer_schema_mismatch"
    ]


@pytest.mark.asyncio
async def test_synthesis_agent_runtime_records_category_backoff_from_generation_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_with_synthesis_output(tmp_path)
    config.synthesis.runtime.max_generation_attempts = 1
    config.synthesis.runtime.max_consecutive_category_discards = 1
    runtime = SynthesisAgentRuntime(config, phase_backends={})
    _install_runtime_stubs(
        monkeypatch,
        runtime,
        results={
            SynthesisPhase.SCHEMA_EXPLORATION: _schema_result(),
            SynthesisPhase.CATEGORY_INFERENCE: _category_result(),
            SynthesisPhase.LABEL_CONSTRUCTION: _label_result(),
            SynthesisPhase.TASK_SYNTHESIS: _task_result(),
            SynthesisPhase.ARTIFACT_GENERATION: _artifact_result(solution_source=""),
        },
    )

    with pytest.raises(SynthesisArtifactGenerationError):
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_category=CategoryTaxonomy.ASSIGNMENT,
            graph=_sample_graph(),
        )

    status = await runtime.category_status(db_id="sakila")
    assignment_status = status[CategoryTaxonomy.ASSIGNMENT]
    assert assignment_status.backed_off is True
    assert assignment_status.last_outcome is SynthesisGenerationOutcome.ARTIFACT_INVALID
    assert assignment_status.last_error_codes == ["solution_source_missing"]


def test_synthesis_runtime_keeps_zero_legacy_import_boundary() -> None:
    runtime_path = Path("src/rl_task_foundry/synthesis/runtime.py")
    module = ast.parse(runtime_path.read_text(encoding="utf-8"), filename=str(runtime_path))
    forbidden_roots = {"tools", "tasks", "truth", "verification"}

    for node in ast.walk(module):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if not node.module.startswith("rl_task_foundry."):
            continue
        root = node.module.split(".", 2)[1]
        assert root not in forbidden_roots, node.module
