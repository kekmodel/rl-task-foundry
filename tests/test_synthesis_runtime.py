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
    ConstraintKind,
    ConstraintSummaryItem,
    DifficultyVectorContract,
    InstanceSpaceContract,
    build_difficulty_vector,
)
from rl_task_foundry.synthesis.runtime import (
    CategoryInferenceOutput,
    LabelConstructionOutput,
    SchemaExplorationOutput,
    SynthesisAgentRuntime,
    SynthesisMemoryEntry,
    SynthesisPhase,
    SynthesisPhaseExecutionError,
    SynthesisStageRequest,
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
            candidate_topics=["assignment"],
            sample_observations=["Observed customer_id=1, customer_name='Alice'"],
            memory_summary="schema exploration completed",
        ),
        tool_traces=tool_traces,
    )


def _category_result() -> SynthesisStageResult:
    return _stage_result(
        phase=SynthesisPhase.CATEGORY_INFERENCE,
        payload=CategoryInferenceOutput(
            selected_topic="assignment",
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
            anchor_entity={"customer_id": 1},
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
            instance_space=InstanceSpaceContract.model_validate(
                {
                    "anchor_query": {
                        "sql": "SELECT customer_id FROM customer ORDER BY customer_id",
                        "outputs": ["customer_id"],
                    }
                }
            ),
            memory_summary="task synthesis completed",
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

    async def _fake_bind_db_id(self, db_id: str):
        self._bound_db_id = db_id

    async def _fake_ensure_category_available(self, db_id: str, topic: str):
        return None

    async def _fake_introspect_graph(self):
        return _sample_graph()

    async def _fake_ensure_atomic_tool_bundle(self, *, db_id: str, graph: SchemaGraph):
        return _sample_atomic_tool_bundle(db_id)

    async def _fake_prime_phase_backends_with_atomic_tools(self, *, db_id: str, bundle):
        return None

    async def _fake_reset(self, db_id: str, topic: str):
        return None

    async def _fake_record_discard(self, db_id: str, topic: str, outcome=None, error_codes=None):
        return None

    monkeypatch.setattr(SynthesisAgentRuntime, "_run_phase", _fake_run_phase)
    monkeypatch.setattr(SynthesisAgentRuntime, "_bind_db_id", _fake_bind_db_id)
    monkeypatch.setattr(
        SynthesisAgentRuntime,
        "_ensure_category_available",
        _fake_ensure_category_available,
    )
    monkeypatch.setattr(SynthesisAgentRuntime, "_introspect_graph", _fake_introspect_graph)
    monkeypatch.setattr(
        SynthesisAgentRuntime,
        "_ensure_atomic_tool_bundle",
        _fake_ensure_atomic_tool_bundle,
    )
    monkeypatch.setattr(
        SynthesisAgentRuntime,
        "_prime_phase_backends_with_atomic_tools",
        _fake_prime_phase_backends_with_atomic_tools,
    )
    monkeypatch.setattr(SynthesisAgentRuntime, "_reset_category_failure_state", _fake_reset)
    monkeypatch.setattr(SynthesisAgentRuntime, "_record_category_discard", _fake_record_discard)


@pytest.mark.asyncio
async def test_synthesize_environment_draft_materializes_label_first_prompt_and_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SynthesisAgentRuntime(_config_with_synthesis_output(tmp_path))
    _install_runtime_stubs(
        monkeypatch,
        runtime,
        results={
            SynthesisPhase.SCHEMA_EXPLORATION: _schema_result(),
            SynthesisPhase.CATEGORY_INFERENCE: _category_result(),
            SynthesisPhase.LABEL_CONSTRUCTION: _label_result(),
            SynthesisPhase.TASK_SYNTHESIS: _task_result(),
        },
    )

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_topic="assignment",
    )

    assert draft.requested_topic == "assignment"
    assert draft.selected_topic == "assignment"
    assert not hasattr(draft, "artifacts")
    assert draft.environment.task.output_schema.root.type.value == "object"
    assert draft.environment.task.output_schema.root.fields[0].name == "customer_name"
    assert draft.instances[0].rendered_user_prompt.startswith("<entity>\n")
    assert '"customer_id": 1' in draft.instances[0].rendered_user_prompt
    assert draft.canonical_answers[0].label_signature.startswith("sha256:")
    assert [result.phase for result in draft.stage_results] == [
        SynthesisPhase.SCHEMA_EXPLORATION,
        SynthesisPhase.CATEGORY_INFERENCE,
        SynthesisPhase.LABEL_CONSTRUCTION,
        SynthesisPhase.TASK_SYNTHESIS,
    ]


def test_runtime_rejects_ungrounded_schema_exploration(tmp_path: Path) -> None:
    runtime = SynthesisAgentRuntime(_config_with_synthesis_output(tmp_path))
    request = SynthesisStageRequest(
        phase=SynthesisPhase.SCHEMA_EXPLORATION,
        db_id="sakila",
        requested_topic="assignment",
        domain_name="customer_support",
        scenario_description="help requests",
    )
    with pytest.raises(SynthesisPhaseExecutionError):
        runtime._ensure_grounded_schema_exploration(  # type: ignore[attr-defined]
            request,
            _schema_result(grounded=False),
        )


def test_runtime_module_has_no_legacy_imports() -> None:
    module_path = Path("src/rl_task_foundry/synthesis/runtime.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    banned_prefixes = (
        "rl_task_foundry.tools",
        "rl_task_foundry.tasks",
        "rl_task_foundry.truth",
        "rl_task_foundry.verification",
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(banned_prefixes)
