from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig
from rl_task_foundry.schema.graph import ColumnProfile, SchemaGraph, TableProfile
from rl_task_foundry.schema.profiler import DataProfile
from rl_task_foundry.synthesis.contracts import (
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    RolloutConstraintsContract,
    TaskBundleContract,
    TaskContract,
)
from rl_task_foundry.synthesis.runtime import (
    SynthesisAgentRuntime,
    SynthesisArtifactGenerationError,
    SynthesisTaskDraft,
)
from rl_task_foundry.synthesis.submit_draft_tool import (
    SubmitDraftController,
    SubmitDraftErrorCode,
    SubmitDraftPayload,
    build_submit_draft_sdk_tool,
)
from rl_task_foundry.synthesis.submit_draft_validators import _ungrounded_answer_strings
from rl_task_foundry.synthesis.synthesis_db import SynthesisDb


def _wrap_user_prompt(anchor_entity: dict[str, object], body: str) -> str:
    return (
        "<entity>\n"
        f"{json.dumps(anchor_entity, ensure_ascii=False, sort_keys=True)}\n"
        "</entity>\n\n"
        f"{body}"
    )

def _scalar_answer_contract(
    *,
    phrase: str,
    constraint_phrases: list[str] | None = None,
    **legacy: object,
) -> dict[str, object]:
    phrases = _legacy_constraint_phrases(constraint_phrases, legacy)
    return {
        "kind": "scalar",
        "answer_phrase": phrase,
        "constraint_phrases": phrases,
        "limit_phrase": None,
    }

def _list_answer_contract(
    *,
    phrase: str,
    constraint_phrases: list[str] | None = None,
    limit_phrase: str | None = "3건",
    **legacy: object,
) -> dict[str, object]:
    phrases = _legacy_constraint_phrases(constraint_phrases, legacy)
    return {
        "kind": "list",
        "answer_phrase": phrase,
        "constraint_phrases": phrases,
        "limit_phrase": limit_phrase,
    }


def _legacy_constraint_phrases(
    phrases: list[str] | None,
    legacy: dict[str, object],
) -> list[str]:
    normalized = list(phrases or [])
    for key in ("predicates", "order_by"):
        raw_entries = legacy.get(key)
        if not isinstance(raw_entries, list):
            continue
        for entry in raw_entries:
            if isinstance(entry, dict) and isinstance(entry.get("phrase"), str):
                normalized.append(entry["phrase"])
    return normalized

def _record_query_evidence(
    controller: SubmitDraftController,
    label: object,
    *,
    answer_contract: object | None = None,
    column_sources: list[dict[str, object]] | None = None,
    referenced_columns: list[dict[str, object]] | None = None,
    query_params: dict[str, object] | None = None,
) -> None:
    del answer_contract
    rows = [label] if isinstance(label, dict) else label
    columns: list[str] = []
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        columns = list(rows[0].keys())
    if column_sources is None:
        column_sources = [
            {
                "output": column,
                "kind": "select",
                "visibility": "user_visible",
                "is_handle": False,
                "value_exposes_source": True,
            }
            for column in columns
        ]
    controller.record_atomic_tool_call(
        tool_name="query",
        params=query_params or {"spec": {"test_evidence": True, "where": [{"value": 1}]}},
        result={
            "columns": columns,
            "column_sources": column_sources,
            "referenced_columns": referenced_columns or [],
            "rows": rows,
            "row_count": len(rows) if isinstance(rows, list) else 0,
        },
    )


def _draft_with_task_bundle(payload: SubmitDraftPayload):
    task = TaskContract(
        question=payload.user_request,
        topic=payload.topic,
        output_schema=OutputSchemaContract(
            root=OutputFieldContract(
                name="answer",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(
                        name="value",
                        type=OutputFieldType.STRING,
                    )
                ],
            )
        ),
    )
    task_bundle = TaskBundleContract(
        task_id="task-1",
        db_id="pagila",
        domain="test",
        topic=payload.topic,
        atomic_tool_set_ref="test",
        created_at=datetime.now(UTC),
        generator_version="test",
        tool_signature="test",
        task_signature="test",
        rollout_constraints=RolloutConstraintsContract(
            max_turns=16,
            max_episode_duration_ms=1000,
        ),
        task=task,
    )
    return SynthesisTaskDraft(
        created_at=datetime.now(UTC),
        db_id="pagila",
        requested_topic=payload.topic,
        selected_topic=payload.topic,
        task_bundle=task_bundle,
        rendered_user_prompt=payload.user_request,
        anchor_entity=payload.entity,
        canonical_answer_json=json.dumps(payload.label, ensure_ascii=False, sort_keys=True),
        label_signature="test-label",
    )


def test_submit_draft_feedback_examples_are_database_agnostic(tmp_path: Path) -> None:
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

    message = controller._invalid_submission_message(
        [
            SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED,
            SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED,
        ],
        feedback_only=True,
    )

    assert '{"<pk_column>": 123}' in message
    assert '{"<pk_part_1>": 7, "<pk_part_2>": 1}' in message
    for db_specific in (
        "customer_id",
        "order_id",
        "line_no",
        "staff member",
        "order 17",
    ):
        assert db_specific not in message

def _config_with_synthesis_output(tmp_path: Path):
    config = load_config("rl_task_foundry.yaml")
    output = OutputConfig(
        run_db_path=tmp_path / "run.db",
        traces_dir=tmp_path / "traces",
    )
    runtime_config = config.synthesis.runtime.model_copy(
        update={"anchor_candidates_enabled": False}
    )
    synthesis_config = config.synthesis.model_copy(update={"runtime": runtime_config})
    return config.model_copy(
        update={"output": output, "synthesis": synthesis_config},
        deep=True,
    )

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
class _FakeDatabasePools:
    control_acquires: int = 0
    solver_acquires: int = 0
    control_connection_value: object = field(default_factory=object)

    @asynccontextmanager
    async def control_connection(self):
        self.control_acquires += 1
        yield self.control_connection_value

    @asynccontextmanager
    async def solver_connection(self):
        self.solver_acquires += 1
        yield object()


@dataclass(slots=True)
class _FakeAnchorConnection:
    async def fetchrow(self, sql: str, *params: object):
        del sql, params
        return {"customer_id": 284}

    async def fetchval(self, sql: str, *params: object):
        del sql, params
        return 0


@dataclass(slots=True)
class _FakeBackend:
    accept_payload: SubmitDraftPayload | None = None
    reject_payload: SubmitDraftPayload | None = None
    provider_name: str = "codex_oauth"
    model_name: str = "gpt-5.4-mini"
    seen_conversations: list[object] = field(default_factory=list)
    seen_max_turns: list[int] = field(default_factory=list)
    seen_anchor_hints: list[dict[str, object] | None] = field(default_factory=list)

    async def run_synthesis(
        self,
        *,
        conversation,
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
        examples_pack: object | None = None,
        affordance_map: dict[str, object] | None = None,
    ):
        del (
            db_id,
            requested_topic,
            domain_name,
            task_language,
            scenario_description,
            schema_summary,
            tool_surface_summary,
            examples_pack,
            affordance_map,
        )
        self.seen_max_turns.append(max_turns)
        self.seen_anchor_hints.append(anchor_hint)
        self.seen_conversations.append(conversation)
        controller = conversation.controller
        controller.record_atomic_tool_call(
            tool_name="find_customer_by_store_id",
            params={"op": "eq", "value": 1, "sort_by": None, "direction": "asc", "limit": 5},
            result=[{"customer_id": 1}, {"customer_id": 2}],
        )
        controller.record_atomic_tool_call(
            tool_name="get_customer",
            params={"id": 1},
            result={"store_id": 1},
        )
        controller.record_atomic_tool_call(
            tool_name="find_payment_by_customer_id",
            params={"op": "eq", "value": 1, "sort_by": None, "direction": "asc", "limit": 3},
            result=[{"payment_id": 11}, {"payment_id": 12}],
        )
        controller.record_atomic_tool_call(
            tool_name="find_rental_by_customer_id",
            params={"op": "eq", "value": 1, "sort_by": None, "direction": "asc", "limit": 3},
            result=[{"rental_id": 201}, {"rental_id": 202}],
        )
        controller.record_atomic_tool_call(
            tool_name="calc_customer",
            params={"fn": "count", "metric": None, "by": None, "op": None, "value": None},
            result=5,
        )
        controller.record_atomic_tool_call(
            tool_name="get_payment",
            params={"id": 11},
            result={"payment_id": 11},
        )
        if self.reject_payload is not None:
            _record_query_evidence(controller, self.reject_payload.label)
            await controller.submit(self.reject_payload)
        if self.accept_payload is not None:
            _record_query_evidence(controller, self.accept_payload.label)
            await controller.submit(self.accept_payload)
        return type(
            "ConversationResult",
            (),
            {
                "provider": self.provider_name,
                "model": self.model_name,
                "final_output_text": "done",
                "turn_count": 6,
                "token_usage": {"requests": 1},
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
            "answer_contract": _scalar_answer_contract(phrase="전체 고객 수"),
        }
    )

def _feedback_payload() -> SubmitDraftPayload:
    payload = _accepted_payload().model_dump(mode="json")
    payload["user_request"] = "이 <entity> 기준으로 내가 배정된 매장을 알려 주세요."
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
            "answer_contract": _scalar_answer_contract(
                phrase="빌린 건수",
                table="rental",
                column="rental_id",
            ),
        }
    )

def _too_easy_list_payload() -> SubmitDraftPayload:
    anchor_entity = {"customer_id": 1}
    return SubmitDraftPayload.model_validate(
        {
            "topic": "recent_payments",
            "label": [
                {"amount": "7.99", "payment_date": "2007-02-14 15:16:03"},
                {"amount": "5.99", "payment_date": "2007-02-13 10:11:12"},
                {"amount": "4.99", "payment_date": "2007-02-12 09:08:07"},
            ],
            "entity": anchor_entity,
            "user_request": "제 결제 중 5달러 이상인 최근 3건을 결제일 내림차순으로 알려 주세요.",
            "answer_contract": _list_answer_contract(
                phrase="알려 주세요",
                predicates=[
                    {
                        "table": "payment",
                        "column": "amount",
                        "op": "gte",
                        "value": 5,
                        "phrase": "5달러 이상",
                    }
                ],
                order_by=[
                    {
                        "table": "payment",
                        "column": "payment_date",
                        "direction": "desc",
                        "phrase": "결제일 내림차순",
                    }
                ],
                limit=3,
                limit_phrase="3건",
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
            "answer_contract": _scalar_answer_contract(phrase="고객 수"),
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
            "answer_contract": _scalar_answer_contract(
                phrase="영화 제목",
                table="film",
                fn="max",
                column="title",
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
            "answer_contract": _scalar_answer_contract(
                phrase="대여 시각",
                table="rental",
                fn="max",
                column="rental_date",
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
            "answer_contract": _scalar_answer_contract(phrase="고객 수"),
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
            "answer_contract": _scalar_answer_contract(
                phrase="한 건",
                table="rental",
                column="rental_id",
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
    schema = SubmitDraftPayload.model_json_schema()
    required_fields = set(schema.get("required", []))
    contract_required = set(schema["$defs"]["AnswerContract"]["required"])
    answer_contract_schema = schema["properties"]["answer_contract"]

    assert "constraint_summary" not in required_fields
    assert "user_request" in required_fields
    assert "answer_contract" in required_fields
    assert "question" not in required_fields
    assert {
        "kind",
        "answer_phrase",
        "constraint_phrases",
        "limit_phrase",
    } == contract_required
    assert answer_contract_schema["$ref"] == "#/$defs/AnswerContract"
    assert "Do not restate tables" in answer_contract_schema["description"]
    assert schema["$defs"]["AnswerContract"]["properties"]["kind"]["enum"] == [
        "scalar",
        "list",
    ]

def test_submit_draft_payload_schema_uses_strict_json_string_fields() -> None:
    schema = SubmitDraftPayload.model_json_schema()
    entity_schema = schema["properties"]["entity_json"]
    label_schema = schema["properties"]["label_json"]

    assert entity_schema["type"] == "string"
    assert label_schema["type"] == "string"
    assert "JSON string" in entity_schema["description"]
    assert "JSON string" in label_schema["description"]
    assert "scoped to that entity" in label_schema["description"]
    assert "Use 'my'/'own' wording only" in schema["properties"]["user_request"]["description"]
    assert "hidden context naturally represents the requester" in (
        schema["properties"]["user_request"]["description"]
    )
    assert "entity scope" in schema["properties"]["answer_contract"]["description"]
    assert "Do not restate tables" in schema["properties"]["answer_contract"]["description"]
    assert SubmitDraftPayload.model_validate(
        {
            **_accepted_payload().model_dump(mode="json"),
            "entity": '{"customer_id": 1}',
        }
    ).parsed_entity == {"customer_id": 1}
    legacy_label_payload = _accepted_payload().model_dump(mode="json")
    legacy_label_payload.pop("label_json")
    legacy_label_payload["label"] = '{"customer_count": 1}'
    assert (
        SubmitDraftPayload.model_validate(legacy_label_payload).canonical_answer
        == {"customer_count": 1}
    )
    assert _loose_json_schema_paths(schema) == []


def _loose_json_schema_paths(value: object, *, path: str = "$") -> list[str]:
    loose: list[str] = []
    if isinstance(value, dict):
        if value.get("items") == {}:
            loose.append(f"{path}.items")
        if value.get("additionalProperties") is True:
            loose.append(f"{path}.additionalProperties")
        for key in ("properties", "$defs"):
            child_map = value.get(key)
            if isinstance(child_map, dict):
                for name, child in child_map.items():
                    loose.extend(_loose_json_schema_paths(child, path=f"{path}.{name}"))
        if "items" in value:
            loose.extend(_loose_json_schema_paths(value["items"], path=f"{path}[]"))
        for combiner in ("anyOf", "oneOf", "allOf"):
            branches = value.get(combiner)
            if isinstance(branches, list):
                for index, child in enumerate(branches):
                    loose.extend(
                        _loose_json_schema_paths(
                            child,
                            path=f"{path}.{combiner}[{index}]",
                        )
                    )
    return loose

def test_submit_draft_tool_schema_descriptions_are_prompt_aligned(tmp_path: Path) -> None:
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
    tool = build_submit_draft_sdk_tool(controller)
    schema_surface = json.dumps(tool.params_json_schema, sort_keys=True)

    assert "Submit one grounded task draft" in tool.description
    assert "successful query produced the exact label values" in tool.description
    assert "Answer shape copied from the latest query" in schema_surface
    assert "the query rows array" in schema_surface
    assert "Do not restate tables, columns, operators, or SQL" in schema_surface
    assert "JSON string for the hidden current-context grounding handle" in schema_surface
    assert "JSON string for the canonical submit_result payload" in schema_surface
    assert "decorative anchor" in schema_surface
    assert "observed values derived from it" in schema_surface
    assert "global answer that can be produced without the hidden entity" in schema_surface
    assert "Include only answer fields the user_request asks to receive" in schema_surface
    assert "rerun query with only the fields intended for submit_result" in schema_surface
    assert "Do not include profile/scope fields" in schema_surface
    assert "do not make a raw handle the main selected answer" in schema_surface
    assert "latest successful query supplies structural evidence" in schema_surface
    assert "those values should stay hidden from user_request" in schema_surface
    assert "Bad: '<entity type> 38'" not in schema_surface
    assert "Good: 'my account'" not in schema_surface
    assert "hidden structural handles" in schema_surface
    assert "visible value only when it appeared in tool evidence" in schema_surface
    assert "copied exactly from the latest successful query result" in schema_surface
    for leaked in ("solver", "actor", "RLVR", "pass_rate", "training"):
        assert leaked not in schema_surface

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
    assert "exact raw value from the chosen tool response row" not in message
    assert "Ungrounded values included" in message

@pytest.mark.asyncio
async def test_submit_draft_too_easy_feedback_preserves_readable_path(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=30,
            total_solver_runs=30,
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
    _record_query_evidence(controller, _too_easy_readable_payload().label)

    message = await controller.submit(_too_easy_readable_payload())

    assert "needs more specificity" in message
    assert "choose a feasible structural strengthening" in message
    assert "Replacing a field on the same path is not an escalation" in message
    assert "new grounded filter" in message
    assert "feasible visible dimension" in message
    assert "do not switch to a list, Cardinality, or Cross-item rule" in message
    assert "return a list of N records" not in message
    assert "solver" not in message.lower()
    assert "pass rate" not in message.lower()
    assert "Width" not in message
    # no DB-specific field names in feedback
    assert "staff_name" not in message

@pytest.mark.asyncio
async def test_submit_draft_too_easy_feedback_is_list_aware(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=30,
            total_solver_runs=30,
        ),
        build_draft=lambda payload: type(
            "Draft",
            (),
            {"task_bundle": type("TaskBundle", (), {"task_id": "task-1", "db_id": "sakila"})()},
        )(),
        max_submissions=3,
    )
    _seed_min_initial_exploration(controller)
    payload = _too_easy_list_payload()
    _record_query_evidence(
        controller,
        payload.label,
        answer_contract=payload.answer_contract,
    )

    message = await controller.submit(payload)

    assert "needs more specificity" in message
    assert "This is a list answer" in message
    assert "selected-row query target" in message
    assert "supported by the current DB evidence" in message
    for axis in ("Filter", "Composite", "Cardinality", "Item-complexity", "Order"):
        assert axis in message
    assert "passive display-only fields" in message
    assert "Width" not in message

@pytest.mark.asyncio
async def test_submit_draft_rejects_answer_contract_phrase_absent_from_request(
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
    payload = _accepted_payload()
    payload = SubmitDraftPayload.model_validate(
        {
            **payload.model_dump(mode="json"),
            "answer_contract": _scalar_answer_contract(phrase="대여 횟수"),
        }
    )
    _record_query_evidence(
        controller,
        payload.label,
        answer_contract=payload.answer_contract,
    )

    message = await controller.submit(payload)

    assert "Every answer_contract phrase" in message
    assert controller.accepted_draft is None


def test_submit_draft_reports_malformed_answer_contract_as_feedback(
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
    payload = _accepted_payload().model_dump(mode="json")
    payload["answer_contract"] = (
        '{"kind": "list", "operation": {"fn": "select"}, '
        '"order_by": [{"table": "actor", "column": "actor_id"}'
    )
    with pytest.raises(ValidationError) as exc_info:
        SubmitDraftPayload.model_validate(payload)

    message = controller.reject_invalid_payload(
        parsed=payload,
        error=exc_info.value,
    )

    assert "answer_contract must be a valid JSON object" in message
    assert "malformed JSON string" in message
    assert controller.attempts == []
    assert controller.submissions_left() == 2


def test_submit_draft_schema_feedback_reports_entity_and_evidence_fixes(
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
    payload = _accepted_payload().model_dump(mode="json")
    payload.pop("entity_json")
    payload["answer_contract"]["evidence"] = '{"rows": [{"count": 1}]}'
    with pytest.raises(ValidationError) as exc_info:
        SubmitDraftPayload.model_validate(payload)

    message = controller.reject_invalid_payload(
        parsed=payload,
        error=exc_info.value,
    )

    assert "entity must contain at least one primary-key value" in message
    assert "Do not paste query result JSON or SQL structure" in message
    assert controller.attempts == []
    assert controller.submissions_left() == 2


@pytest.mark.asyncio
async def test_submit_draft_rejects_label_that_does_not_match_latest_query(
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
    payload = _accepted_payload()
    _record_query_evidence(
        controller,
        {"store_id": 1, "customer_count": 6},
    )

    message = await controller.submit(payload)

    assert "label must exactly match the latest successful query result" in message
    assert "For kind='list', copy the query rows array" in message
    assert "rerun query with only the intended label fields" in message
    assert controller.accepted_draft is None


@pytest.mark.asyncio
async def test_submit_draft_does_not_require_contract_to_restate_query_predicates(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        build_draft=_draft_with_task_bundle,
        max_submissions=3,
    )
    _seed_min_initial_exploration(controller)
    payload = SubmitDraftPayload.model_validate(
        {
            "topic": "family_films",
            "label": {"family_high_rental_count": 2},
            "entity": {"actor_id": 112},
            "user_request": (
                "2015년 이후에 출시된 가족(Family) 영화면서 대여료가 "
                "2.99달러보다 높은 영화는 몇 개인가요?"
            ),
            "answer_contract": _scalar_answer_contract(
                phrase="몇 개인가요",
                table="film",
                column="film_id",
                predicates=[
                    {
                        "table": "film",
                        "column": "rental_rate",
                        "op": "gt",
                        "value": 2.99,
                        "phrase": "2.99달러보다 높은",
                    },
                    {
                        "table": "film",
                        "column": "release_year",
                        "op": "gte",
                        "value": 2015,
                        "phrase": "2015년 이후에 출시된",
                    },
                    {
                        "table": "category",
                        "column": "name",
                        "op": "eq",
                        "value": "Family",
                        "phrase": "가족(Family) 영화",
                    },
                ],
            ),
        }
    )
    _record_query_evidence(
        controller,
        payload.label,
        referenced_columns=[
            {
                "usage": "where",
                "table": "film",
                "column": "rental_rate",
                "visibility": "user_visible",
                "is_handle": False,
                "op": "gt",
                "value": 2.99,
            },
            {
                "usage": "where",
                "table": "film",
                "column": "release_year",
                "visibility": "user_visible",
                "is_handle": False,
                "op": "gte",
                "value": 2015,
            },
        ],
    )

    message = await controller.submit(payload)

    assert "Draft accepted" in message
    assert "answer_contract predicates/order_by must be present" not in message
    assert controller.accepted_draft is not None


@pytest.mark.asyncio
async def test_submit_draft_treats_list_limit_one_as_rows_array(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=30,
            total_solver_runs=30,
        ),
        build_draft=lambda payload: type(
            "Draft",
            (),
            {"task_bundle": type("TaskBundle", (), {"task_id": "task-1", "db_id": "pagila"})()},
        )(),
        max_submissions=3,
    )
    _seed_min_initial_exploration(controller)
    payload = SubmitDraftPayload.model_validate(
        {
            "topic": "recent_rental",
            "label": [
                {
                    "대여일": "2022-08-22 23:56:01+00:00",
                    "반납예정일": "2022-08-24 00:00:01+00:00",
                }
            ],
            "entity": {"film_id": 245},
            "user_request": "가장 최근 1건의 대여 일자와 반납 예정일을 알려 주세요.",
            "answer_contract": _list_answer_contract(
                phrase="대여 일자와 반납 예정일",
                table="rental",
                order_by=[
                    {
                        "table": "rental",
                        "column": "rental_date",
                        "direction": "desc",
                        "phrase": "가장 최근",
                    }
                ],
                limit=1,
                limit_phrase="가장 최근 1건",
            ),
        }
    )
    _record_query_evidence(
        controller,
        payload.label,
        answer_contract=payload.answer_contract,
        query_params={"spec": {"test_evidence": True, "where": [{"value": 245}]}},
    )

    message = await controller.submit(payload)

    assert "needs more specificity" in message
    assert "label must exactly match the latest successful query result" not in message
    assert controller.accepted_draft is None


@pytest.mark.asyncio
async def test_submit_draft_rejects_label_from_non_user_visible_query_source(
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
    payload = SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "label": {"staff_email": "agent@example.test"},
            "entity": {"customer_id": 1},
            "user_request": "내 담당 직원의 이메일을 알려 주세요.",
            "answer_contract": _scalar_answer_contract(
                phrase="이메일",
                table="staff",
                fn="max",
                column="email",
            ),
        }
    )
    _record_query_evidence(
        controller,
        payload.label,
        column_sources=[
            {
                "output": "staff_email",
                "kind": "select",
                "table": "staff",
                "column": "email",
                "visibility": "internal",
                "is_handle": False,
                "value_exposes_source": True,
            }
        ],
    )

    message = await controller.submit(payload)

    assert "explicitly marked internal or blocked" in message
    assert "is_handle: true" not in message
    assert controller.accepted_draft is None


@pytest.mark.asyncio
async def test_submit_draft_allows_handle_label_when_visibility_policy_allows_it(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        build_draft=_draft_with_task_bundle,
        max_submissions=3,
    )
    _seed_min_initial_exploration(controller)
    payload = SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "label": {"assigned_location_reference": 117},
            "entity": {"account_reference": 1},
            "user_request": "연결된 지점 참조 값을 알려 주세요.",
            "answer_contract": _scalar_answer_contract(
                phrase="지점 참조 값",
                table="location",
                fn="max",
                column="location_id",
            ),
        }
    )
    _record_query_evidence(
        controller,
        payload.label,
        column_sources=[
            {
                "output": "assigned_location_reference",
                "kind": "select",
                "table": "location",
                "column": "location_id",
                "visibility": "user_visible",
                "is_handle": True,
                "value_exposes_source": True,
            }
        ],
    )

    message = await controller.submit(payload)

    assert "Draft accepted" in message
    assert controller.accepted_draft is not None


@pytest.mark.asyncio
async def test_submit_draft_rejects_query_without_visibility_metadata(
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
    payload = _accepted_payload()
    rows = [payload.label]
    controller.record_atomic_tool_call(
        tool_name="query",
        params={"spec": {"legacy_result": True}},
        result={
            "columns": list(payload.label.keys()),
            "rows": rows,
            "row_count": len(rows),
        },
    )

    message = await controller.submit(payload)

    assert "field visibility evidence" in message
    assert controller.accepted_draft is None


@pytest.mark.asyncio
async def test_submit_draft_allows_non_user_visible_query_predicate(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        build_draft=_draft_with_task_bundle,
        max_submissions=3,
    )
    _seed_min_initial_exploration(controller)
    payload = SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "label": {"matching_customer_count": 3},
            "entity": {"customer_id": 1},
            "user_request": "이메일이 있는 관련 고객 수를 알려 주세요.",
            "answer_contract": _scalar_answer_contract(
                phrase="고객 수",
                table="customer",
                column="customer_id",
                predicates=[
                    {
                        "table": "customer",
                        "column": "email",
                        "op": "is_not_null",
                        "value": None,
                        "phrase": "이메일이 있는",
                    }
                ],
            ),
        }
    )
    _record_query_evidence(
        controller,
        payload.label,
        column_sources=[
            {
                "output": "matching_customer_count",
                "kind": "aggregate",
                "fn": "count",
                "visibility": "derived",
                "value_exposes_source": False,
            }
        ],
        referenced_columns=[
            {
                "usage": "where",
                "table": "customer",
                "column": "email",
                "visibility": "internal",
                "is_handle": False,
                "op": "is_not_null",
                "value": None,
            }
        ],
    )

    message = await controller.submit(payload)

    assert "Draft accepted" in message
    assert controller.accepted_draft is not None


@pytest.mark.asyncio
async def test_submit_draft_allows_handle_order_by_when_label_is_visible(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=1,
            total_solver_runs=2,
        ),
        build_draft=_draft_with_task_bundle,
        max_submissions=3,
    )
    _seed_min_initial_exploration(controller)
    payload = SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "label": [{"first_name": "JULIA"}],
            "entity": {"film_reference": 830},
            "user_request": "배우 이름을 첫 1명만 알려 주세요.",
            "answer_contract": _list_answer_contract(
                phrase="배우 이름",
                table="actor",
                column=None,
                order_by=[
                    {
                        "table": "actor",
                        "column": "actor_id",
                        "direction": "asc",
                        "phrase": "첫 1명",
                    }
                ],
                limit=1,
                limit_phrase="첫 1명",
            ),
        }
    )
    _record_query_evidence(
        controller,
        payload.label,
        column_sources=[
            {
                "output": "first_name",
                "kind": "select",
                "table": "actor",
                "column": "first_name",
                "visibility": "user_visible",
                "is_handle": False,
                "value_exposes_source": True,
            }
        ],
        referenced_columns=[
            {
                "usage": "order_by",
                "table": "actor",
                "column": "actor_id",
                "visibility": "user_visible",
                "is_handle": True,
                "direction": "asc",
            }
        ],
        query_params={"spec": {"test_evidence": True, "where": [{"value": 830}]}},
    )

    message = await controller.submit(payload)

    assert "Draft accepted" in message
    assert controller.accepted_draft is not None


@pytest.mark.asyncio
async def test_submit_draft_too_easy_requires_incremental_answer_contract(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=30,
            total_solver_runs=30,
        ),
        build_draft=lambda payload: type(
            "Draft",
            (),
            {"task_bundle": type("TaskBundle", (), {"task_id": "task-1", "db_id": "sakila"})()},
        )(),
        max_submissions=3,
    )
    _seed_min_initial_exploration(controller)
    first_payload = _accepted_payload()
    _record_query_evidence(controller, first_payload.label)

    first_message = await controller.submit(first_payload)

    assert "needs more specificity" in first_message
    second_payload = SubmitDraftPayload.model_validate(
        {
            "topic": "assignment",
            "label": {"store_id": 1, "payment_count": 7},
            "entity": {"customer_id": 1},
            "user_request": "내가 배정된 매장과 전체 결제 수를 알려 주세요.",
            "answer_contract": _scalar_answer_contract(
                phrase="전체 결제 수",
                table="payment",
                column="payment_id",
            ),
        }
    )
    _record_query_evidence(controller, second_payload.label)

    second_message = await controller.submit(second_payload)

    assert "keep the same answer kind and query output target" in second_message
    assert controller.accepted_draft is None


@pytest.mark.asyncio
async def test_submit_draft_too_easy_rejects_renamed_same_scalar_value(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="rental_status",
        solver_orchestrator=_FakeSolverOrchestrator(
            matched_solver_runs=30,
            total_solver_runs=30,
        ),
        build_draft=lambda payload: type(
            "Draft",
            (),
            {"task_bundle": type("TaskBundle", (), {"task_id": "task-1", "db_id": "sakila"})()},
        )(),
        max_submissions=3,
    )
    _seed_min_initial_exploration(controller)
    first_payload = SubmitDraftPayload.model_validate(
        {
            "topic": "rental_status",
            "label": {"unreturned_count": 2},
            "entity": {"customer_id": 1},
            "user_request": "이 고객의 미반납 대여 건수를 알려 주세요.",
            "answer_contract": _scalar_answer_contract(
                phrase="미반납 대여 건수",
                table="rental",
                column="rental_id",
                predicates=[
                    {
                        "table": "rental",
                        "column": "return_date",
                        "op": "is_null",
                        "phrase": "미반납",
                    }
                ],
            ),
        }
    )
    _record_query_evidence(
        controller,
        first_payload.label,
        answer_contract=first_payload.answer_contract,
    )

    first_message = await controller.submit(first_payload)

    assert "needs more specificity" in first_message
    second_payload = SubmitDraftPayload.model_validate(
        {
            "topic": "rental_status",
            "label": {"unreturned_since_february": 2},
            "entity": {"customer_id": 1},
            "user_request": "이 고객의 2022년 2월 이후 미반납 대여 건수를 알려 주세요.",
            "answer_contract": _scalar_answer_contract(
                phrase="미반납 대여 건수",
                table="rental",
                column="rental_id",
                predicates=[
                    {
                        "table": "rental",
                        "column": "return_date",
                        "op": "is_null",
                        "phrase": "미반납",
                    },
                    {
                        "table": "rental",
                        "column": "rental_date",
                        "op": "gte",
                        "value": "2022-02-01",
                        "phrase": "2022년 2월 이후",
                    },
                ],
            ),
        }
    )
    _record_query_evidence(
        controller,
        second_payload.label,
        answer_contract=second_payload.answer_contract,
    )

    second_message = await controller.submit(second_payload)

    assert "same single-field answer value" in second_message
    assert "submitted value changes" in second_message
    assert controller.accepted_draft is None


@pytest.mark.asyncio
async def test_synthesis_runtime_returns_accepted_task_draft(tmp_path: Path) -> None:
    backend = _FakeBackend(accept_payload=_accepted_payload())
    config = _config_with_synthesis_output(tmp_path)
    database_pools = _FakeDatabasePools()
    synthesis_db = SynthesisDb(
        db_id="sakila",
        config=config,
        database_pools=database_pools,  # type: ignore[arg-type]
    )
    synthesis_db._graph_cache = _sample_graph()
    synthesis_db._data_profile_cache = DataProfile()
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
    assert database_pools.control_acquires == 1
    assert database_pools.solver_acquires == 0
    assert draft.task_bundle.status.value == "accepted"
    assert draft.task_bundle.quality_metrics.solver_pass_rate == 0.5
    assert draft.rendered_user_prompt.startswith("<entity>")
    assert backend.seen_max_turns == [20]
    assert backend.seen_anchor_hints == [None]


@pytest.mark.asyncio
async def test_synthesis_runtime_passes_candidate_anchor_hint_when_enabled(
    tmp_path: Path,
) -> None:
    backend = _FakeBackend(accept_payload=_accepted_payload())
    config = _config_with_synthesis_output(tmp_path)
    runtime_config = config.synthesis.runtime.model_copy(
        update={
            "anchor_candidates_enabled": True,
            "anchor_candidate_limit": 1,
        }
    )
    synthesis_config = config.synthesis.model_copy(
        update={"runtime": runtime_config}
    )
    config = config.model_copy(update={"synthesis": synthesis_config})
    database_pools = _FakeDatabasePools(
        control_connection_value=_FakeAnchorConnection()
    )
    synthesis_db = SynthesisDb(
        db_id="sakila",
        config=config,
        database_pools=database_pools,  # type: ignore[arg-type]
    )
    synthesis_db._graph_cache = _sample_graph()
    synthesis_db._data_profile_cache = DataProfile()
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
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_topic="assignment",
            graph=_sample_graph(),
        )
    finally:
        await runtime.close()

    assert database_pools.control_acquires == 2
    anchor_hint = backend.seen_anchor_hints[0]
    assert anchor_hint is not None
    candidates = anchor_hint["candidate_entities"]
    assert isinstance(candidates, list)
    assert candidates[0]["row_id"] == 284
    assert candidates[0]["entity"] == {"customer_id": 284}

@pytest.mark.asyncio
async def test_synthesis_runtime_raises_after_invalid_only_submission(tmp_path: Path) -> None:
    backend = _FakeBackend(reject_payload=_feedback_payload())
    config = _config_with_synthesis_output(tmp_path)
    synthesis_db = SynthesisDb(db_id="sakila", config=config)
    synthesis_db._graph_cache = _sample_graph()
    synthesis_db._data_profile_cache = DataProfile()
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
    synthesis_db._graph_cache = _sample_graph()
    runtime = SynthesisAgentRuntime(
        config,
        synthesis_backends=[_FakeBackend()],
        synthesis_db=synthesis_db,
    )

    await runtime.close()

    # injected synthesis_db is preserved (caller owns lifecycle)
    assert synthesis_db._graph_cache is not None
    assert runtime._synthesis_db is None


@pytest.mark.asyncio
async def test_synthesis_runtime_close_disposes_owned_synthesis_db(tmp_path: Path) -> None:
    config = _config_with_synthesis_output(tmp_path)
    runtime = SynthesisAgentRuntime(
        config,
        synthesis_backends=[_FakeBackend()],
    )
    synthesis_db = runtime._ensure_synthesis_db("sakila")

    await runtime.close()

    assert runtime._synthesis_db is None
    assert synthesis_db._database_pools is None

def test_submit_draft_payload_rejects_blank_text() -> None:
    payload = _accepted_payload().model_dump(mode="json")
    payload["user_request"] = "   "

    with pytest.raises(ValidationError):
        SubmitDraftPayload.model_validate(payload)

def test_submit_draft_diagnoses_values_from_disconnected_tool_chain(
    tmp_path: Path,
) -> None:
    """Disconnected label evidence is useful diagnostic context, not a hard reject."""
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
    from rl_task_foundry.synthesis.submit_draft_validators import (
        _disconnected_answer_strings,
        _rebuild_anchor_connected_strings,
    )

    anchor_strings = _rebuild_anchor_connected_strings(
        controller._raw_atomic_tool_calls,
        anchor_entity={"customer_id": 1},
    )
    disconnected = _disconnected_answer_strings(
        {"film_title": "ACADEMY DINOSAUR", "rental_date": "2005-05-25T11:30:37"},
        observed_strings=controller._observed_response_strings,
        anchor_connected_strings=anchor_strings,
    )

    assert disconnected == ["academy dinosaur"]


def test_anchor_connected_replay_follows_nested_query_params() -> None:
    from rl_task_foundry.synthesis.submit_draft_validators import (
        _rebuild_anchor_connected_strings,
    )

    anchor_strings = _rebuild_anchor_connected_strings(
        [
            {
                "tool_name": "query",
                "params": {
                    "spec": {
                        "from": {"table": "rental"},
                        "where": [{"ref": {"column": "customer_id"}, "op": "eq", "value": 545}],
                    }
                },
                "result": {"rows": [{"total_amount": "38.91"}]},
            }
        ],
        anchor_entity={"customer_id": 545},
    )

    assert "38.91" in anchor_strings
