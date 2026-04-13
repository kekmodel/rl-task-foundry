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
    _count_semantics_present,
    _mentions_global_scope,
    _next_difficulty_crank_axis,
    _uses_unanchored_global_ranking,
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


def _seed_min_initial_exploration(
    controller: SubmitDraftController,
    *,
    customer_id: int = 1,
) -> None:
    controller.record_atomic_tool_call(
        tool_name="find_customer_by_store_id",
        params={"op": "eq", "value": 2, "sort_by": None, "direction": "asc", "limit": 5},
        result=[{"customer_id": customer_id}, {"customer_id": customer_id + 1}],
    )
    controller.record_atomic_tool_call(
        tool_name="get_customer",
        params={"id": customer_id},
        result={"customer_id": customer_id, "store_id": 2},
    )
    controller.record_atomic_tool_call(
        tool_name="find_payment_by_customer_id",
        params={"op": "eq", "value": customer_id, "sort_by": None, "direction": "asc", "limit": 3},
        result=[{"payment_id": 11}, {"payment_id": 12}],
    )
    controller.record_atomic_tool_call(
        tool_name="find_rental_by_customer_id",
        params={"op": "eq", "value": customer_id, "sort_by": None, "direction": "asc", "limit": 3},
        result=[{"rental_id": 101}, {"rental_id": 102}],
    )
    controller.record_atomic_tool_call(
        tool_name="get_payment",
        params={"id": 11},
        result={"payment_id": 11},
    )
    controller.record_atomic_tool_call(
        tool_name="calc_payment",
        params={"fn": "count", "metric": None, "by": "customer_id", "op": "eq", "value": customer_id},
        result=2,
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
                name="get_customer",
                family=AtomicToolFamily.GET,
                description="Retrieve one customer by ID. Returns all fields or nothing.",
                params_schema={
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
                returns_schema={
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "customer_name": {"type": "string"},
                            },
                            "required": ["customer_name"],
                            "additionalProperties": False,
                        },
                        {"type": "null"},
                    ],
                },
                sql="SELECT 1",
                result_mode=AtomicToolResultMode.OBJECT_OR_NULL,
                semantic_key="customer:get",
            )
        ],
        source="async def get_customer(conn, id):\n    return {'store_id': 1}\n",
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
            tool_name="find_customer_by_store_id",
            params={"op": "eq", "value": 1, "sort_by": None, "direction": "asc", "limit": 5},
            result=[{"customer_id": 1}, {"customer_id": 2}],
        )
        self.bound_controller.record_atomic_tool_call(
            tool_name="get_customer",
            params={"id": 1},
            result={"store_id": 1},
        )
        self.bound_controller.record_atomic_tool_call(
            tool_name="find_payment_by_customer_id",
            params={"op": "eq", "value": 1, "sort_by": None, "direction": "asc", "limit": 3},
            result=[{"payment_id": 11}, {"payment_id": 12}],
        )
        self.bound_controller.record_atomic_tool_call(
            tool_name="find_rental_by_customer_id",
            params={"op": "eq", "value": 1, "sort_by": None, "direction": "asc", "limit": 3},
            result=[{"rental_id": 201}, {"rental_id": 202}],
        )
        self.bound_controller.record_atomic_tool_call(
            tool_name="calc_customer",
            params={"fn": "count", "metric": None, "by": None, "op": None, "value": None},
            result=5,
        )
        self.bound_controller.record_atomic_tool_call(
            tool_name="get_payment",
            params={"id": 11},
            result={"payment_id": 11},
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
                "tool_calls": ("get_customer", "submit_draft"),
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
        }
    )


def _ungrounded_text_payload(*, customer_id: int = 1) -> SubmitDraftPayload:
    anchor_entity = {"customer_id": customer_id}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "record_history",
            "canonical_answer_json": '{"film_title": "Airplane Sierra"}',
            "anchor_entity": anchor_entity,
            "difficulty_vector": {
                "search_cost": 2.0,
                "solution_space": 1.0,
                "constraint_density": 1.0,
            },
            "question": _wrap_user_prompt(
                anchor_entity,
                "내 기록과 관련된 영화 제목을 알려 주세요.",
            ),
            "constraint_summary": [
                {
                    "key": "anchored_record",
                    "kind": "membership",
                    "summary": "Answer must stay within the anchored customer's own records.",
                }
            ],
        }
    )


def _global_count_payload() -> SubmitDraftPayload:
    anchor_entity = {"customer_id": 1}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "canonical_answer_json": '{"customer_name": "Alice", "customer_count": 5}',
            "anchor_entity": anchor_entity,
            "difficulty_vector": {
                "search_cost": 2.0,
                "solution_space": 1.0,
                "constraint_density": 1.0,
            },
            "question": _wrap_user_prompt(
                anchor_entity,
                "제 기록을 기준으로 제 이름과 관련 고객 수를 알려 주세요.",
            ),
            "constraint_summary": [
                {
                    "key": "anchor_self",
                    "kind": "membership",
                    "summary": "Answer must stay within the anchored customer.",
                }
            ],
        }
    )


def _id_chain_payload() -> SubmitDraftPayload:
    anchor_entity = {"customer_id": 1}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "canonical_answer_json": '{"rental_id": 777, "payment_id": 9710}',
            "anchor_entity": anchor_entity,
            "difficulty_vector": {
                "search_cost": 2.0,
                "solution_space": 1.0,
                "constraint_density": 1.0,
            },
            "question": _wrap_user_prompt(
                anchor_entity,
                "제 계정과 연결된 대여와 결제 한 건을 알려 주세요.",
            ),
            "constraint_summary": [
                {
                    "key": "anchor_self",
                    "kind": "membership",
                    "summary": "Answer must stay within the anchored customer.",
                }
            ],
        }
    )


def _opaque_identifier_payload() -> SubmitDraftPayload:
    anchor_entity = {"customer_id": 1}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "record_history",
            "canonical_answer_json": '{"tracking_token": "550e8400-e29b-41d4-a716-446655440000"}',
            "anchor_entity": anchor_entity,
            "difficulty_vector": {
                "search_cost": 2.0,
                "solution_space": 1.0,
                "constraint_density": 1.0,
            },
            "question": _wrap_user_prompt(
                anchor_entity,
                "제 기록과 연결된 추적 정보를 알려 주세요.",
            ),
            "constraint_summary": [
                {
                    "key": "anchor_self",
                    "kind": "membership",
                    "summary": "Answer must stay within the anchored customer.",
                }
            ],
        }
    )


def test_submit_draft_payload_caches_parsed_canonical_answer() -> None:
    payload = SubmitDraftPayload.model_validate(_accepted_payload().model_dump(mode="json"))
    cached_answer = payload.canonical_answer

    with patch("rl_task_foundry.synthesis.submit_draft_tool.json.loads") as mocked_loads:
        assert payload.canonical_answer == cached_answer
        mocked_loads.assert_not_called()


def test_submit_draft_payload_rejects_legacy_anchor_query_field() -> None:
    payload = _accepted_payload().model_dump(mode="json")
    payload["anchor_query"] = {
        "sql": "SELECT customer_id FROM customer ORDER BY customer_id",
        "outputs": ["customer_id"],
    }

    with pytest.raises(ValidationError):
        SubmitDraftPayload.model_validate(payload)


def test_next_difficulty_crank_axis_rotates_after_two_attempts() -> None:
    assert _next_difficulty_crank_axis([]) is DifficultyAxis.SEARCH_COST
    assert _next_difficulty_crank_axis([DifficultyAxis.SEARCH_COST]) is DifficultyAxis.SEARCH_COST
    assert _next_difficulty_crank_axis(
        [DifficultyAxis.SEARCH_COST, DifficultyAxis.SEARCH_COST]
    ) is DifficultyAxis.SOLUTION_SPACE


def test_count_semantics_present_does_not_match_account_substrings() -> None:
    assert not _count_semantics_present(
        {"customer_id": 360},
        [
            {
                "key": "anchor_self",
                "kind": "identity",
                "summary": "Use the anchored customer's own account.",
            }
        ],
    )


def test_mentions_global_scope_does_not_treat_local_total_as_global() -> None:
    assert not _mentions_global_scope(
        "제 계정의 전체 결제 건수를 알려주세요.",
        [
            {
                "key": "payment_total",
                "kind": "count",
                "summary": "Return the total number of payments for the anchored customer.",
            }
        ],
    )


def test_unanchored_global_ranking_matches_rank_family() -> None:
    assert _uses_unanchored_global_ranking(
        [
            {
                "tool_name": "rank_payment_by_store_id",
                "params": {
                    "fn": "count",
                    "metric": None,
                    "direction": "desc",
                    "limit": 5,
                    "by": None,
                    "op": None,
                    "value": None,
                },
                "result": [{"group_key": 1, "value": 5}],
            }
        ],
        anchor_entity={"customer_id": 1},
    )


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
async def test_submit_draft_requires_more_first_submit_exploration(
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
        tool_name="get_customer",
        params={"id": 1},
        result={"customer_name": "Alice"},
    )
    controller.record_atomic_tool_call(
        tool_name="find_payment_by_customer_id",
        params={"op": "eq", "value": 1, "sort_by": None, "direction": "asc", "limit": 3},
        result=[{"payment_id": 11}, {"payment_id": 12}],
    )

    message = await controller.submit(_accepted_payload())

    assert "Do not submit yet" in message
    assert "Go back to research mode first." in message
    assert "map the database relationships around the anchored user" in message
    assert "until you fully understand the nearby evidence paths" in message
    assert "at least 6 atomic observations" in message
    assert "at least 4 distinct tool names" in message
    assert "at least 3 anchor-scoped observations" in message
    assert "classify nearby relationships and paths as readable, id-only, local-only, countable, aggregate-capable, or dead ends" in message
    assert "why every answer slot is grounded and needed" in message


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
        tool_name="get_customer",
        params={"id": 1},
        result={"customer_name": "Alice"},
    )

    message = await controller.submit(_count_without_count_evidence_payload())

    assert "count-like label needs explicit count evidence" in message.lower()


@pytest.mark.asyncio
async def test_submit_draft_keeps_locked_self_anchor_across_feedback_retries(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="record_history",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        build_draft=lambda payload: payload,
        max_submissions=3,
        self_anchor_surface_names=("get_customer",),
    )
    _seed_min_initial_exploration(controller)

    first = await controller.submit(_ungrounded_text_payload(customer_id=1))
    second = await controller.submit(_ungrounded_text_payload(customer_id=2))

    assert "readable text fields" in first
    assert "same anchored user entity across retries" in second
    assert controller._locked_anchor_entity == {"customer_id": 1}


@pytest.mark.asyncio
async def test_submit_draft_calls_out_id_only_anchor_path_for_ungrounded_strings(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="record_history",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        build_draft=lambda payload: payload,
        max_submissions=3,
        self_anchor_surface_names=("get_customer",),
    )
    _seed_min_initial_exploration(controller)

    message = await controller.submit(_ungrounded_text_payload())

    assert "does not expose readable text fields" in message
    assert "Stop retrying names, titles" in message


@pytest.mark.asyncio
async def test_submit_draft_pushes_self_scoped_count_back_to_anchor_evidence(
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
        tool_name="get_customer",
        params={"id": 1},
        result={"customer_name": "Alice"},
    )
    controller.record_atomic_tool_call(
        tool_name="calc_customer",
        params={"fn": "count", "metric": None, "by": None, "op": None, "value": None},
        result=5,
    )

    message = await controller.submit(_global_count_payload())

    assert "only keep a count field if you observed a count or aggregate tool call" in message
    assert "drop the count field" in message


@pytest.mark.asyncio
async def test_submit_draft_calls_out_id_only_identifier_chain_path(
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
    _seed_min_initial_exploration(controller)
    controller.record_atomic_tool_call(
        tool_name="find_rental_by_customer_id",
        params={"op": "eq", "value": 1, "sort_by": None, "direction": "asc", "limit": 3},
        result=[{"rental_id": 777}],
    )
    controller.record_atomic_tool_call(
        tool_name="find_payment_by_customer_id",
        params={"op": "eq", "value": 1, "sort_by": None, "direction": "asc", "limit": 3},
        result=[{"payment_id": 9710}],
    )

    message = await controller.submit(_id_chain_payload())

    assert "current anchored evidence path is still id-only" in message
    assert "Do not submit another answer made only of *_id fields" in message


@pytest.mark.asyncio
async def test_submit_draft_rejects_opaque_identifier_values(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="record_history",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        build_draft=lambda payload: payload,
        max_submissions=3,
    )
    controller.record_atomic_tool_call(
        tool_name="get_customer",
        params={"id": 1},
        result={"customer_id": 1, "tracking_token": "550e8400-e29b-41d4-a716-446655440000"},
    )

    message = await controller.submit(_opaque_identifier_payload())

    assert "opaque identifier values" in message
    assert "UUIDs, hashes, encrypted tokens" in message


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

    async def _get_customer(_kwargs):
        return {"customer_name": "Alice"}

    runtime._tool_executor_cache["sakila"] = {
        "get_customer": _get_customer,
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

    async def _get_customer(_kwargs):
        return {"customer_name": "Alice"}

    runtime._tool_executor_cache["sakila"] = {
        "get_customer": _get_customer,
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
    payload["question"] = "   "

    with pytest.raises(ValidationError):
        SubmitDraftPayload.model_validate(payload)
