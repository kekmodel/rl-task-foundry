from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from pydantic import ValidationError

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig
from rl_task_foundry.schema.graph import ColumnProfile, SchemaGraph, TableProfile
from rl_task_foundry.synthesis.atomic_tools import (
    AtomicToolBundle,
    AtomicToolDefinition,
    AtomicToolFamily,
    AtomicToolResultMode,
)
from rl_task_foundry.synthesis.contracts import DifficultyAxis
from rl_task_foundry.synthesis.runtime import (
    SynthesisAgentRuntime,
    SynthesisArtifactGenerationError,
)
from rl_task_foundry.synthesis.submit_draft_tool import (
    SubmitDraftController,
    SubmitDraftPayload,
    _label_summary_matches_selected_topic,
    _next_difficulty_crank_axis,
)


def _wrap_user_prompt(anchor_entity: dict[str, object], body: str) -> str:
    return (
        "<entity>\n"
        f"{json.dumps(anchor_entity, ensure_ascii=False, sort_keys=True)}\n"
        "</entity>\n\n"
        f"{body}"
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
    store_id = ColumnProfile(
        schema_name="public",
        table_name="customer",
        column_name="store_id",
        data_type="integer",
        ordinal_position=2,
        is_nullable=False,
        visibility="user_visible",
    )
    return SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="customer",
                columns=[customer_id, store_id],
                primary_key=("customer_id",),
                row_estimate=5,
            )
        ],
        edges=[],
    )


def _sample_atomic_tool_bundle(db_id: str = "sakila") -> AtomicToolBundle:
    return AtomicToolBundle(
        db_id=db_id,
        tools=[
            AtomicToolDefinition(
                name="get_customer_by_id",
                family=AtomicToolFamily.T1_POINT_LOOKUP,
                description="Look up a single customer by its primary key. Returns one row or nothing.",
                params_schema={"type": "object", "properties": {"customer_id": {"type": "integer"}}},
                returns_schema={
                    "type": "object",
                    "properties": {
                        "customer_name": {"type": "string"},
                    },
                },
                sql="SELECT 1",
                result_mode=AtomicToolResultMode.OBJECT_OR_NULL,
                semantic_key="customer:get_by_id",
            )
        ],
        source="async def get_customer_by_id(conn, customer_id):\n    return {'store_id': 1}\n",
    )


@dataclass(slots=True)
class _FakeSolverOrchestrator:
    matched_solver_runs: int
    total_solver_runs: int

    async def run_draft(self, draft):
        return type(
            "Summary",
            (),
            {
                "task_id": draft.task_bundle.task_id,
                "db_id": draft.task_bundle.db_id,
                "planned_solver_runs": self.total_solver_runs,
                "total_solver_runs": self.total_solver_runs,
                "matched_solver_runs": self.matched_solver_runs,
                "early_stop_decision": None,
                "runs": (),
                "pass_rate": self.matched_solver_runs / self.total_solver_runs,
            },
        )()


@dataclass(slots=True)
class _FakeBackend:
    accept_payload: SubmitDraftPayload | None = None
    reject_payload: SubmitDraftPayload | None = None
    provider_name: str = "codex_oauth"
    model_name: str = "gpt-5.4-mini"
    bound_controller: object | None = None
    bound_tool_definitions: list[dict[str, object]] | None = None
    bound_tool_executors: dict[str, object] | None = None
    seen_max_turns: list[int] = field(default_factory=list)

    def bind_atomic_tools(self, *, tool_definitions, tool_executors) -> None:
        self.bound_tool_definitions = tool_definitions
        self.bound_tool_executors = tool_executors

    def bind_submit_draft_controller(self, controller) -> None:
        self.bound_controller = controller

    async def run_synthesis(
        self,
        *,
        db_id: str,
        requested_topic: str,
        domain_name: str,
        task_language: str,
        scenario_description: str,
        schema_summary: dict[str, object],
        tool_surface_summary: dict[str, object],
        max_turns: int,
    ):
        del db_id, requested_topic, domain_name, task_language, scenario_description, schema_summary, tool_surface_summary
        self.seen_max_turns.append(max_turns)
        assert self.bound_controller is not None
        self.bound_controller.record_atomic_tool_call(
            tool_name="get_customer_by_id",
            params={"customer_id": 1},
            result={"store_id": 1},
        )
        self.bound_controller.record_atomic_tool_call(
            tool_name="count_customer",
            params={},
            result=5,
        )
        if self.reject_payload is not None:
            await self.bound_controller.submit(self.reject_payload)
        if self.accept_payload is not None:
            await self.bound_controller.submit(self.accept_payload)
        return type(
            "ConversationResult",
            (),
            {
                "provider": self.provider_name,
                "model": self.model_name,
                "final_output_text": "done",
                "turn_count": 6,
                "token_usage": {"requests": 1},
                "transcript_ref": "memory://transcript",
                "tool_trace_ref": "memory://tool-trace",
                "tool_calls": ("get_customer_by_id", "submit_draft"),
            },
        )()


def _accepted_payload() -> SubmitDraftPayload:
    anchor_entity = {"customer_id": 1}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "canonical_answer_json": '{"store_id": 1, "customer_count": 5}',
            "anchor_entity": anchor_entity,
            "difficulty_vector": {
                "search_cost": 2.0,
                "solution_space": 1.0,
                "constraint_density": 1.0,
            },
            "question": _wrap_user_prompt(
                anchor_entity,
                "내가 배정된 매장과 전체 고객 수를 알려 주세요.",
            ),
            "constraint_summary": [
                {
                    "key": "single_store",
                    "kind": "uniqueness",
                    "summary": "배정 매장은 하나여야 한다.",
                }
            ],
            "anchor_query": {
                "sql": "SELECT customer_id FROM customer ORDER BY customer_id",
                "outputs": ["customer_id"],
            },
            "label_summary": "The assignment label is grounded in customer lookup and count evidence.",
        }
    )


def _feedback_payload() -> SubmitDraftPayload:
    payload = _accepted_payload().model_dump(mode="json")
    payload["question"] = "내가 배정된 매장을 알려 주세요."
    return SubmitDraftPayload.model_validate(payload)


def _count_without_count_evidence_payload() -> SubmitDraftPayload:
    anchor_entity = {"customer_id": 1}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "canonical_answer_json": '{"customer_count": 1, "customer_name": "Alice"}',
            "anchor_entity": anchor_entity,
            "difficulty_vector": {
                "search_cost": 2.0,
                "solution_space": 1.0,
                "constraint_density": 1.0,
            },
            "question": _wrap_user_prompt(
                anchor_entity,
                "제 기록을 기준으로 고객 수와 제 이름을 알려 주세요.",
            ),
            "constraint_summary": [
                {
                    "key": "anchor_self",
                    "kind": "membership",
                    "summary": "Answer must be about the anchored customer.",
                }
            ],
            "anchor_query": {
                "sql": "SELECT customer_id FROM customer ORDER BY customer_id",
                "outputs": ["customer_id"],
            },
            "label_summary": "The assignment label is grounded in anchored customer evidence.",
        }
    )


def test_submit_draft_payload_caches_parsed_canonical_answer() -> None:
    payload = SubmitDraftPayload.model_validate(_accepted_payload().model_dump(mode="json"))
    cached_answer = payload.canonical_answer

    with patch("rl_task_foundry.synthesis.submit_draft_tool.json.loads") as mocked_loads:
        assert payload.canonical_answer == cached_answer
        mocked_loads.assert_not_called()


def test_selected_topic_matching_normalizes_word_separators() -> None:
    assert _label_summary_matches_selected_topic(
        selected_topic="bundle_selection",
        label_summary="This bundle selection label is grounded in observed rows.",
        min_token_length=3,
    )
    assert _label_summary_matches_selected_topic(
        selected_topic="bundle_selection",
        label_summary="This bundle_selection label is grounded in observed rows.",
        min_token_length=3,
    )


def test_next_difficulty_crank_axis_rotates_after_two_attempts() -> None:
    assert _next_difficulty_crank_axis([]) is DifficultyAxis.SEARCH_COST
    assert _next_difficulty_crank_axis([DifficultyAxis.SEARCH_COST]) is DifficultyAxis.SEARCH_COST
    assert _next_difficulty_crank_axis(
        [DifficultyAxis.SEARCH_COST, DifficultyAxis.SEARCH_COST]
    ) is DifficultyAxis.SOLUTION_SPACE


@pytest.mark.asyncio
async def test_submit_draft_feedback_consumes_total_submit_budget(tmp_path: Path) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        build_draft=lambda payload: payload,
        max_submissions=2,
    )

    first = await controller.submit(_feedback_payload())
    second = await controller.submit(_feedback_payload())
    third = await controller.submit(_feedback_payload())

    assert "1 attempts left." in first
    assert "Budget exhausted. No more attempts." in second
    assert third == "Budget exhausted. No more attempts."
    assert controller.submissions_left() == 0


@pytest.mark.asyncio
async def test_submit_draft_rejects_mixed_count_label_without_count_evidence(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        build_draft=lambda payload: payload,
        max_submissions=3,
    )
    controller.record_atomic_tool_call(
        tool_name="get_customer_by_id",
        params={"customer_id": 1},
        result={"customer_name": "Alice"},
    )

    message = await controller.submit(_count_without_count_evidence_payload())

    assert "count-like label needs explicit count evidence" in message.lower()


@pytest.mark.asyncio
async def test_synthesis_runtime_returns_accepted_task_draft(tmp_path: Path) -> None:
    backend = _FakeBackend(accept_payload=_accepted_payload())
    runtime = SynthesisAgentRuntime(
        _config_with_synthesis_output(tmp_path),
        synthesis_backends=[backend],
    )
    runtime._solver_orchestrator = _FakeSolverOrchestrator(
        matched_solver_runs=1,
        total_solver_runs=2,
    )
    runtime._graph_cache = _sample_graph()
    runtime._atomic_tool_bundles["sakila"] = _sample_atomic_tool_bundle()

    async def _get_customer_by_id(_kwargs):
        return {"customer_name": "Alice"}

    runtime._tool_executor_cache["sakila"] = {
        "get_customer_by_id": _get_customer_by_id,
    }

    try:
        draft = await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_topic="assignment",
            graph=_sample_graph(),
        )
    finally:
        await runtime.close()

    assert draft.task_bundle.topic == "assignment"
    assert draft.task_bundle.status.value == "accepted"
    assert draft.task_bundle.quality_metrics.solver_pass_rate == 0.5
    assert draft.rendered_user_prompt.startswith("<entity>")
    assert backend.seen_max_turns == [50]


@pytest.mark.asyncio
async def test_synthesis_runtime_raises_after_invalid_only_submission(tmp_path: Path) -> None:
    backend = _FakeBackend(reject_payload=_feedback_payload())
    runtime = SynthesisAgentRuntime(
        _config_with_synthesis_output(tmp_path),
        synthesis_backends=[backend],
    )
    runtime._solver_orchestrator = _FakeSolverOrchestrator(
        matched_solver_runs=1,
        total_solver_runs=2,
    )
    runtime._graph_cache = _sample_graph()
    runtime._atomic_tool_bundles["sakila"] = _sample_atomic_tool_bundle()

    async def _get_customer_by_id(_kwargs):
        return {"customer_name": "Alice"}

    runtime._tool_executor_cache["sakila"] = {
        "get_customer_by_id": _get_customer_by_id,
    }

    try:
        with pytest.raises(SynthesisArtifactGenerationError):
            await runtime.synthesize_environment_draft(
                db_id="sakila",
                requested_topic="assignment",
                graph=_sample_graph(),
            )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_synthesis_runtime_close_clears_cached_tool_executors(tmp_path: Path) -> None:
    runtime = SynthesisAgentRuntime(
        _config_with_synthesis_output(tmp_path),
        synthesis_backends=[_FakeBackend()],
    )
    runtime._tool_executor_cache["sakila"] = {"noop": lambda _kwargs: {}}

    await runtime.close()

    assert runtime._tool_executor_cache == {}


def test_submit_draft_payload_rejects_blank_text() -> None:
    payload = _accepted_payload().model_dump(mode="json")
    payload["label_summary"] = "   "

    with pytest.raises(ValidationError):
        SubmitDraftPayload.model_validate(payload)
