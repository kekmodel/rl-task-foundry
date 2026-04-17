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
from rl_task_foundry.schema.profiler import DataProfile
from rl_task_foundry.synthesis.atomic_tools import (
    AtomicToolBundle,
    AtomicToolDefinition,
    AtomicToolFamily,
    AtomicToolResultMode,
)
from rl_task_foundry.synthesis.runtime import (
    SynthesisAgentRuntime,
    SynthesisArtifactGenerationError,
)
from rl_task_foundry.synthesis.synthesis_db import SynthesisDb
from rl_task_foundry.synthesis.submit_draft_tool import (
    SubmitDraftController,
    SubmitDraftPayload,
    _count_semantics_present,
    _ungrounded_answer_strings,
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
        params={
            "fn": "count",
            "metric": None,
            "by": "customer_id",
            "op": "eq",
            "value": customer_id,
        },
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
        anchor_hint: dict[str, object] | None = None,
        data_profile: object | None = None,
    ):
        del (
            db_id,
            requested_topic,
            domain_name,
            task_language,
            scenario_description,
            schema_summary,
            tool_surface_summary,
        )
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
            "label": '{"store_id": 1, "customer_count": 5}',
            "entity": anchor_entity,
            "question": _wrap_user_prompt(
                anchor_entity,
                "내가 배정된 매장과 전체 고객 수를 알려 주세요.",
            ),
        }
    )

def _feedback_payload() -> SubmitDraftPayload:
    payload = _accepted_payload().model_dump(mode="json")
    payload["question"] = "내가 배정된 매장을 알려 주세요."
    return SubmitDraftPayload.model_validate(payload)

def _too_easy_readable_payload() -> SubmitDraftPayload:
    anchor_entity = {"customer_id": 1}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "assignment_summary",
            "label": json.dumps(
                {
                    "staff_name": "Mike Hillyer",
                    "staff_email": "Mike.Hillyer@sakilastaff.com",
                    "latest_rental_count": 32,
                },
                ensure_ascii=False,
            ),
            "entity": anchor_entity,
            "question": _wrap_user_prompt(
                anchor_entity,
                "내 계정 기준으로 담당 직원 이름과 이메일을 알려주고,"
                " 제가 지금까지 빌린 건수도 함께 알려주세요.",
            ),
        }
    )

def _count_without_count_evidence_payload() -> SubmitDraftPayload:
    anchor_entity = {"customer_id": 1}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "label": '{"customer_count": 1, "customer_name": "Alice"}',
            "entity": anchor_entity,
            "question": _wrap_user_prompt(
                anchor_entity,
                "제 기록을 기준으로 고객 수와 제 이름을 알려 주세요.",
            ),
        }
    )

def _ungrounded_text_payload(*, customer_id: int = 1) -> SubmitDraftPayload:
    anchor_entity = {"customer_id": customer_id}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "record_history",
            "label": '{"film_title": "Airplane Sierra"}',
            "entity": anchor_entity,
            "question": _wrap_user_prompt(
                anchor_entity,
                "내 기록과 관련된 영화 제목을 알려 주세요.",
            ),
        }
    )

def _partially_rewritten_string_payload() -> SubmitDraftPayload:
    anchor_entity = {"customer_id": 1}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "latest_rental_assignment",
            "label": json.dumps(
                {
                    "rental_staff_name": "Bob",
                    "latest_rental_date": "2005-08-22T20:03:46",
                },
                ensure_ascii=False,
            ),
            "entity": anchor_entity,
            "question": _wrap_user_prompt(
                anchor_entity,
                "제가 최근에 대여한 기록의 처리 직원 이름과 대여 시각을 알려주세요.",
            ),
        }
    )

def _global_count_payload() -> SubmitDraftPayload:
    anchor_entity = {"customer_id": 1}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "label": '{"customer_name": "Alice", "customer_count": 5}',
            "entity": anchor_entity,
            "question": _wrap_user_prompt(
                anchor_entity,
                "제 기록을 기준으로 제 이름과 관련 고객 수를 알려 주세요.",
            ),
        }
    )

def _id_chain_payload() -> SubmitDraftPayload:
    anchor_entity = {"customer_id": 1}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "label": '{"rental_id": 777, "payment_id": 9710}',
            "entity": anchor_entity,
            "question": _wrap_user_prompt(
                anchor_entity,
                "제 계정과 연결된 대여와 결제 한 건을 알려 주세요.",
            ),
        }
    )

def test_ungrounded_answer_strings_accepts_datetime_observations() -> None:
    ungrounded = _ungrounded_answer_strings(
        {
            "latest_rental_date": "2005-08-22 20:03:46",
            "latest_rental_return_date": "2005-08-30 01:51:46",
        },
        observed_strings={
            "2005-08-22 20:03:46",
            "2005-08-30 01:51:46",
        },
    )

    assert ungrounded == []

def test_submit_draft_controller_caches_observed_tool_response_values(tmp_path: Path) -> None:
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
        tool_name="get_staff",
        params={"id": 1},
        result={
            "first_name": "Mike",
            "last_name": "Hillyer",
            "email": "Mike.Hillyer@sakilastaff.com",
        },
    )

    assert "mike" in controller._observed_response_strings
    assert "hillyer" in controller._observed_response_strings
    assert "mike.hillyer@sakilastaff.com" in controller._observed_response_strings

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

def test_submit_draft_payload_schema_does_not_require_constraint_summary() -> None:
    required_fields = set(SubmitDraftPayload.model_json_schema().get("required", []))

    assert "constraint_summary" not in required_fields

def test_count_semantics_present_does_not_match_account_substrings() -> None:
    assert not _count_semantics_present({"customer_id": 360}, "제 계정 정보를 알려주세요.")

def test_count_semantics_present_does_not_match_country_key() -> None:
    assert not _count_semantics_present({"country": "USA"}, "이 나라를 알려주세요.")
    assert not _count_semantics_present({"discount": 0.1}, "할인율을 알려주세요.")
    assert not _count_semantics_present({"account": "abc"}, "계정을 알려주세요.")
    assert _count_semantics_present({"rental_count": 5}, "how many rentals?")
    assert _count_semantics_present({"inventory_count": 8}, "count the inventory")

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

    assert "Attempts left: 1." in first
    assert "BudgetExhaustedError: No more attempts." in second
    assert third == "BudgetExhaustedError: No more attempts."
    assert controller.submissions_left() == 0

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
    )
    _seed_min_initial_exploration(controller)

    message = await controller.submit(_ungrounded_text_payload())

    assert "does not expose readable text fields" in message
    assert "Stop retrying names, titles" in message

@pytest.mark.asyncio
async def test_submit_draft_requires_exact_observed_string_values(
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
        params={"op": "eq", "value": 1, "sort_by": "rental_date", "direction": "desc", "limit": 1},
        result=[
            {
                "customer_id": 1,
                "rental_id": 15315,
                "rental_date": "2005-08-22 20:03:46",
                "return_date": "2005-08-30 01:51:46",
                "staff_id": 2,
            }
        ],
    )
    controller.record_atomic_tool_call(
        tool_name="get_staff",
        params={"id": 2},
        result={
            "staff_id": 2,
            "first_name": "Jon",
            "last_name": "Stephens",
            "email": "Jon.Stephens@sakilastaff.com",
        },
    )

    message = await controller.submit(_partially_rewritten_string_payload())

    assert "copy them exactly as they appeared there" in message
    assert "Do not shorten names" in message
    assert "exact raw value from the chosen tool response row" in message
    assert "Ungrounded values included" in message

@pytest.mark.asyncio
async def test_submit_draft_too_easy_feedback_preserves_readable_path(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=5,
            total_solver_runs=6,
        ),
        build_draft=lambda payload: type(
            "Draft",
            (),
            {"task_bundle": type("TaskBundle", (), {"task_id": "task-1", "db_id": "sakila"})()},
        )(),
        max_submissions=3,
    )
    controller.record_atomic_tool_call(
        tool_name="get_customer",
        params={"id": 1},
        result={
            "customer_id": 1,
            "first_name": "MARY",
            "last_name": "SMITH",
            "email": "MARY.SMITH@sakilacustomer.org",
            "store_id": 1,
        },
    )
    controller.record_atomic_tool_call(
        tool_name="get_staff",
        params={"id": 1},
        result={
            "staff_id": 1,
            "staff_name": "Mike Hillyer",
            "first_name": "Mike",
            "last_name": "Hillyer",
            "email": "Mike.Hillyer@sakilastaff.com",
        },
    )
    controller.record_atomic_tool_call(
        tool_name="find_rental_by_customer_id",
        params={"op": "eq", "value": 1, "sort_by": "rental_date", "direction": "desc", "limit": 3},
        result=[
            {
                "rental_id": 13486,
                "customer_id": 1,
                "rental_date": "2006-02-14 15:16:03",
                "return_date": None,
                "staff_id": 1,
            }
        ],
    )
    controller.record_atomic_tool_call(
        tool_name="calc_rental",
        params={"fn": "count", "metric": None, "by": "customer_id", "op": "eq", "value": 1},
        result=32,
    )
    controller.record_atomic_tool_call(
        tool_name="find_payment_by_customer_id",
        params={"op": "eq", "value": 1, "sort_by": "payment_date", "direction": "desc", "limit": 3},
        result=[{"payment_id": 10, "customer_id": 1, "payment_date": "2007-02-14 15:16:03"}],
    )
    controller.record_atomic_tool_call(
        tool_name="find_customer_by_store_id",
        params={"op": "eq", "value": 1, "sort_by": "customer_id", "direction": "asc", "limit": 3},
        result=[{"customer_id": 1}, {"customer_id": 2}],
    )

    message = await controller.submit(_too_easy_readable_payload())

    assert "Too easy" in message
    assert "Pick ONE structural change" in message
    assert "follow one more FK hop" in message
    assert "filter condition" in message
    assert "Do not just add or remove a single field" in message
    # no DB-specific field names in feedback
    assert "staff_name" not in message

@pytest.mark.asyncio
async def test_synthesis_runtime_returns_accepted_task_draft(tmp_path: Path) -> None:
    backend = _FakeBackend(accept_payload=_accepted_payload())
    config = _config_with_synthesis_output(tmp_path)
    synthesis_db = SynthesisDb(db_id="sakila", config=config)
    synthesis_db._graph_cache = _sample_graph()
    synthesis_db._data_profile_cache = DataProfile()
    synthesis_db._atomic_tool_bundle = _sample_atomic_tool_bundle()

    async def _get_customer(_kwargs):
        return {"customer_name": "Alice"}

    synthesis_db._tool_executors = {"get_customer": _get_customer}
    runtime = SynthesisAgentRuntime(
        config,
        synthesis_backends=[backend],
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        synthesis_db=synthesis_db,
    )

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
    assert backend.seen_max_turns == [20]

@pytest.mark.asyncio
async def test_synthesis_runtime_raises_after_invalid_only_submission(tmp_path: Path) -> None:
    backend = _FakeBackend(reject_payload=_feedback_payload())
    config = _config_with_synthesis_output(tmp_path)
    synthesis_db = SynthesisDb(db_id="sakila", config=config)
    synthesis_db._graph_cache = _sample_graph()
    synthesis_db._data_profile_cache = DataProfile()
    synthesis_db._atomic_tool_bundle = _sample_atomic_tool_bundle()

    async def _get_customer(_kwargs):
        return {"customer_name": "Alice"}

    synthesis_db._tool_executors = {"get_customer": _get_customer}
    runtime = SynthesisAgentRuntime(
        config,
        synthesis_backends=[backend],
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        synthesis_db=synthesis_db,
    )

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
async def test_synthesis_runtime_close_clears_owned_synthesis_db(tmp_path: Path) -> None:
    config = _config_with_synthesis_output(tmp_path)
    synthesis_db = SynthesisDb(db_id="sakila", config=config)
    synthesis_db._tool_executors = {"noop": lambda _kwargs: {}}
    runtime = SynthesisAgentRuntime(
        config,
        synthesis_backends=[_FakeBackend()],
        synthesis_db=synthesis_db,
    )

    await runtime.close()

    # injected synthesis_db is preserved (caller owns lifecycle)
    assert synthesis_db._tool_executors == {"noop": synthesis_db._tool_executors["noop"]}
    assert runtime._synthesis_db is None


@pytest.mark.asyncio
async def test_synthesis_runtime_close_disposes_owned_synthesis_db(tmp_path: Path) -> None:
    config = _config_with_synthesis_output(tmp_path)
    runtime = SynthesisAgentRuntime(
        config,
        synthesis_backends=[_FakeBackend()],
    )
    synthesis_db = runtime._ensure_synthesis_db("sakila")
    synthesis_db._tool_executors = {"noop": lambda _kwargs: {}}

    await runtime.close()

    assert runtime._synthesis_db is None
    assert synthesis_db._tool_executors is None

def test_submit_draft_payload_rejects_blank_text() -> None:
    payload = _accepted_payload().model_dump(mode="json")
    payload["question"] = "   "

    with pytest.raises(ValidationError):
        SubmitDraftPayload.model_validate(payload)

@pytest.mark.asyncio
async def test_submit_draft_rejects_values_from_disconnected_tool_chain(
    tmp_path: Path,
) -> None:
    """Values observed only in global/unanchored calls must not appear in the label."""
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="itinerary",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        build_draft=lambda payload: payload,
        max_submissions=3,
    )
    _seed_min_initial_exploration(controller)
    # anchor-connected call: customer_id=1 → rental with inventory_id=3021
    controller.record_atomic_tool_call(
        tool_name="find_rental_by_customer_id",
        params={"op": "eq", "value": 1, "sort_by": "rental_date", "direction": "asc", "limit": 1},
        result=[
            {"rental_id": 76, "inventory_id": 3021, "rental_date": "2005-05-25T11:30:37"}
        ],
    )
    # disconnected global scan — not connected to anchor entity
    controller.record_atomic_tool_call(
        tool_name="find_film_by_language_id",
        params={"op": "any", "value": None, "sort_by": "film_id", "direction": "asc", "limit": 3},
        result=[
            {"film_id": 1, "title": "ACADEMY DINOSAUR"},
            {"film_id": 2, "title": "ACE GOLDFINGER"},
        ],
    )
    # submit label using value from disconnected call
    payload = SubmitDraftPayload(
        topic="itinerary",
        entity={"customer_id": 1},
        label='{"film_title":"ACADEMY DINOSAUR","rental_date":"2005-05-25T11:30:37"}',
        question=(
            "<entity>\n{\"customer_id\":1}\n</entity>\n\n"
            "이 고객의 첫 대여 영화 제목과 대여 시각을 알려주세요."
        ),
    )
    # Disconnected check is diagnostic-only (not blocking) due to
    # integer ID collision false positives. Verify the detection works
    # by checking _rebuild_anchor_connected_strings directly.
    from rl_task_foundry.synthesis.submit_draft_tool import (
        _rebuild_anchor_connected_strings,
        _disconnected_answer_strings,
    )

    anchor_strings = _rebuild_anchor_connected_strings(
        controller._raw_atomic_tool_calls,
        anchor_entity=payload.parsed_entity,
    )
    disconnected = _disconnected_answer_strings(
        {"film_title": "ACADEMY DINOSAUR", "rental_date": "2005-05-25T11:30:37"},
        observed_strings=controller._observed_response_strings,
        anchor_connected_strings=anchor_strings,
    )
    assert "academy dinosaur" in disconnected

def test_temporal_ordering_combines_question_and_sort_type() -> None:
    """Temporal check requires BOTH a temporal claim in the question
    AND a non-temporal sort_by in tool calls."""
    from rl_task_foundry.synthesis.submit_draft_tool import (
        _has_non_temporal_sort_on_temporal_result,
        _question_claims_temporal_ordering,
    )

    calls_bad_sort = [
        {
            "tool_name": "find_rental_by_inventory_id",
            "params": {"sort_by": "rental_id", "direction": "asc"},
            "result": [
                {"rental_id": 4863, "rental_date": "2005-07-08T19:03:15"}
            ],
        }
    ]
    calls_good_sort = [
        {
            "tool_name": "find_rental_by_inventory_id",
            "params": {"sort_by": "rental_date", "direction": "asc"},
            "result": [
                {"rental_id": 4863, "rental_date": "2005-07-08T19:03:15"}
            ],
        }
    ]
    calls_no_sort = [
        {
            "tool_name": "get_rental",
            "params": {"id": 4863},
            "result": {"rental_id": 4863, "rental_date": "2005-07-08T19:03:15"},
        }
    ]

    # Korean temporal claim + bad sort → flag
    assert _question_claims_temporal_ordering("가장 이른 대여 기록을 알려주세요")
    assert _has_non_temporal_sort_on_temporal_result(calls_bad_sort)

    # Korean temporal claim + good sort → pass
    assert not _has_non_temporal_sort_on_temporal_result(calls_good_sort)

    # English temporal claim
    assert _question_claims_temporal_ordering("Show the earliest rental")

    # No temporal claim → pass regardless of sort
    assert not _question_claims_temporal_ordering("대여 목록을 알려주세요")
    assert not _question_claims_temporal_ordering("Show rentals for this customer")

    # No sort_by → pass
    assert not _has_non_temporal_sort_on_temporal_result(calls_no_sort)
