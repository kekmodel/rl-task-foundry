from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig
from pydantic import ValidationError
from rl_task_foundry.schema.graph import ColumnProfile, SchemaGraph, TableProfile
from rl_task_foundry.synthesis.atomic_tools import (
    AtomicToolBundle,
    AtomicToolDefinition,
    AtomicToolFamily,
    AtomicToolResultMode,
)
from rl_task_foundry.synthesis.contracts import DifficultyAxis, DifficultyVectorContract
from rl_task_foundry.synthesis.phase_monitor import PipelinePhaseMonitorLogger
from rl_task_foundry.synthesis.runtime import (
    SynthesisAgentRuntime,
    SynthesisArtifactGenerationError,
)

from rl_task_foundry.synthesis.submit_draft_tool import (
    SubmitDraftController,
    SubmitDraftPayload,
    _label_summary_matches_selected_topic,
    _next_difficulty_crank_axis,
    build_submit_draft_sdk_tool,
)


def _wrap_user_prompt(anchor_entity: dict[str, object], body: str) -> str:
    return (
        "<entity>\n"
        f"{json.dumps(anchor_entity, ensure_ascii=False, sort_keys=True)}\n"
        "</entity>\n\n"
        f"{body}"
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
    )
    assert _label_summary_matches_selected_topic(
        selected_topic="bundle_selection",
        label_summary="This bundle_selection label is grounded in observed rows.",
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
class _FakeEnvironmentOrchestrator:
    matched_solver_runs: int
    total_solver_runs: int

    async def run_draft(self, draft):
        return type(
            "Summary",
            (),
            {
                "env_id": draft.environment.env_id,
                "db_id": draft.environment.db_id,
                "planned_solver_runs": self.total_solver_runs,
                "total_instances": 1,
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
        del db_id, requested_topic, domain_name, task_language, scenario_description, schema_summary, tool_surface_summary, max_turns
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
                "내가 배정된 매장의 ID와 그 매장의 전체 고객 수를 알려 주세요.",
            ),
            "constraint_summary": [
                {
                    "key": "store_and_count",
                    "kind": "cardinality",
                    "summary": "Return exactly one store and one total customer count.",
                }
            ],
            "instance_space": {
                "anchor_query": {
                    "sql": "SELECT customer_id FROM customer ORDER BY customer_id",
                    "outputs": ["customer_id"],
                }
            },
            "label_summary": "This assignment answer is grounded because the store_id comes from the customer lookup and the total customer count comes from a separate aggregate.",
        }
    )


def _minimal_draft():
    return type(
        "Draft",
        (),
        {
            "environment": type(
                "Env",
                (),
                {
                    "env_id": "env_test",
                    "db_id": "sakila",
                },
            )(),
        },
    )()


@pytest.mark.asyncio
async def test_prime_synthesis_backends_injects_one_shuffle_seed_per_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SynthesisAgentRuntime(_config_with_synthesis_output(tmp_path))
    backend = _FakeBackend()
    runtime.synthesis_backends = [backend]
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: payload,
    )
    seen_payloads: list[dict[str, object]] = []

    async def _recording_executor(kwargs: dict[str, object]) -> dict[str, object]:
        seen_payloads.append(dict(kwargs))
        return {"store_id": 1}

    async def _fake_tool_executors(self, *, db_id: str, bundle: AtomicToolBundle):
        del db_id, bundle
        return {"get_customer_by_id": _recording_executor}

    monkeypatch.setattr(
        SynthesisAgentRuntime,
        "_tool_executors_for_bundle",
        _fake_tool_executors,
    )

    await runtime._prime_synthesis_backends_with_context(
        db_id="sakila",
        bundle=_sample_atomic_tool_bundle(),
        controller=controller,
        shuffle_seed="seed-run-123",
    )

    assert backend.bound_tool_executors is not None
    executor = backend.bound_tool_executors["get_customer_by_id"]
    await executor({"customer_id": 1})
    await executor({"customer_id": 2})

    assert [payload["_shuffle_seed"] for payload in seen_payloads] == [
        "seed-run-123",
        "seed-run-123",
    ]


def test_submit_draft_payload_schema_requires_nonempty_anchor_and_constraints() -> None:
    schema = SubmitDraftPayload.model_json_schema()

    assert "constraint_summary" in schema["required"]
    assert schema["properties"]["anchor_entity"]["minProperties"] == 1
    assert schema["properties"]["constraint_summary"]["minItems"] == 1


def test_submit_draft_payload_rejects_invalid_canonical_answer_json() -> None:
    payload = _accepted_payload().model_dump(mode="json")
    payload["canonical_answer_json"] = "not json"

    with pytest.raises(ValidationError, match="valid JSON string"):
        SubmitDraftPayload.model_validate(payload)


def test_reject_invalid_payload_reports_constraint_summary_requirement(tmp_path: Path) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=0,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: payload,
        max_submissions=1,
    )
    payload = _accepted_payload().model_dump(mode="json")
    payload.pop("constraint_summary")

    with pytest.raises(ValidationError) as exc_info:
        SubmitDraftPayload.model_validate(payload)

    message = controller.reject_invalid_payload(parsed=payload, error=exc_info.value)

    assert "constraint_summary must include at least one grounded constraint" in message
    assert "Budget exhausted. No more attempts." in message


@pytest.mark.asyncio
async def test_synthesize_environment_draft_runs_single_agent_and_returns_accepted_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SynthesisAgentRuntime(_config_with_synthesis_output(tmp_path))
    runtime.synthesis_backends = [
        _FakeBackend(
            accept_payload=_accepted_payload().model_copy(
                update={
                    "topic": "store_assignment",
                    "label_summary": (
                        "This store assignment answer is grounded because the store_id comes "
                        "from the customer lookup and the total customer count comes from a "
                        "separate aggregate."
                    ),
                }
            )
        )
    ]
    runtime._environment_orchestrator = _FakeEnvironmentOrchestrator(
        matched_solver_runs=2,
        total_solver_runs=4,
    )

    async def _fake_bind_db_id(self, db_id: str):
        self._bound_db_id = db_id

    async def _fake_ensure_category_available(self, db_id: str, topic: str):
        return None

    async def _fake_introspect_graph(self):
        return _sample_graph()

    async def _fake_ensure_atomic_tool_bundle(self, *, db_id: str, graph: SchemaGraph):
        del graph
        return _sample_atomic_tool_bundle(db_id)

    async def _fake_tool_executors(self, *, db_id: str, bundle: AtomicToolBundle):
        del db_id, bundle
        return {}

    async def _fake_reset(self, db_id: str, topic: str):
        return None

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
        "_tool_executors_for_bundle",
        _fake_tool_executors,
    )
    monkeypatch.setattr(SynthesisAgentRuntime, "_reset_category_failure_state", _fake_reset)

    draft = await runtime.synthesize_environment_draft(
        db_id="sakila",
        requested_topic="payment_history",
    )

    assert draft.requested_topic == "payment_history"
    assert draft.selected_topic == "store_assignment"
    assert draft.environment.topic == "store_assignment"
    assert draft.environment.status.value == "accepted"
    assert draft.environment.quality_metrics.solver_pass_rate == 0.5
    assert draft.instances[0].rendered_user_prompt.startswith("<entity>\n")
    assert '"customer_id": 1' in draft.instances[0].rendered_user_prompt
    assert draft.canonical_answers[0].label_signature.startswith("sha256:")
    assert draft.generation_attempts[-1].outcome.value == "passed"


@pytest.mark.asyncio
async def test_synthesize_environment_draft_raises_when_submit_budget_exhausts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_with_synthesis_output(tmp_path)
    config.synthesis.runtime.max_generation_attempts = 1
    runtime = SynthesisAgentRuntime(config)
    runtime.synthesis_backends = [_FakeBackend(reject_payload=_accepted_payload())]
    runtime._environment_orchestrator = _FakeEnvironmentOrchestrator(
        matched_solver_runs=4,
        total_solver_runs=4,
    )

    async def _fake_bind_db_id(self, db_id: str):
        self._bound_db_id = db_id

    async def _fake_ensure_category_available(self, db_id: str, topic: str):
        return None

    async def _fake_introspect_graph(self):
        return _sample_graph()

    async def _fake_ensure_atomic_tool_bundle(self, *, db_id: str, graph: SchemaGraph):
        del graph
        return _sample_atomic_tool_bundle(db_id)

    async def _fake_tool_executors(self, *, db_id: str, bundle: AtomicToolBundle):
        del db_id, bundle
        return {}

    async def _fake_record_discard(self, db_id: str, topic: str, outcome=None, error_codes=None):
        return None

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
        "_tool_executors_for_bundle",
        _fake_tool_executors,
    )
    monkeypatch.setattr(SynthesisAgentRuntime, "_record_category_discard", _fake_record_discard)

    with pytest.raises(SynthesisArtifactGenerationError) as exc_info:
        await runtime.synthesize_environment_draft(
            db_id="sakila",
            requested_topic="assignment",
        )

    assert exc_info.value.attempts


@pytest.mark.asyncio
async def test_submit_draft_rejects_questions_that_leak_internal_schema_terms(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
        forbidden_question_tokens=frozenset({"film_category", " join "}),
    )
    controller.record_atomic_tool_call(
        tool_name="get_film_category_by_id",
        params={"film_id": 62, "category_id": 6},
        result={"film_id": 62, "category_id": 6},
    )
    payload = _accepted_payload().model_copy(
        update={
            "anchor_entity": {"actor_id": 107},
            "question": _wrap_user_prompt(
                {"actor_id": 107},
                "film_category 를 통해 category_id를 확인하세요.",
            ),
        }
    )

    message = await controller.submit(payload)

    assert "raw table names" in message
    assert "question_internal_schema_leak" in controller.attempts[-1].error_codes


@pytest.mark.asyncio
async def test_submit_draft_rejects_questions_that_repeat_raw_identifier_fields(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_customer_by_id",
        params={"customer_id": 1},
        result={"store_id": 1},
    )
    controller.record_atomic_tool_call(
        tool_name="count_customer",
        params={},
        result=5,
    )
    payload = _accepted_payload().model_copy(
        update={
            "question": _wrap_user_prompt(
                {"customer_id": 1},
                "customer_id 1의 store_id와 customer_count를 알려 주세요.",
            )
        }
    )

    message = await controller.submit(payload)

    assert "raw identifier field names" in message
    assert controller.attempts[-1].error_codes[0] == "question_raw_identifier_leak"


@pytest.mark.asyncio
async def test_submit_draft_requires_entity_wrapped_question_shape(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_customer_by_id",
        params={"customer_id": 1},
        result={"store_id": 1},
    )
    controller.record_atomic_tool_call(
        tool_name="count_customer",
        params={},
        result=5,
    )
    payload = _accepted_payload().model_copy(
        update={"question": "내가 배정된 매장의 ID와 그 매장의 전체 고객 수를 알려 주세요."}
    )

    message = await controller.submit(payload)

    assert "Feedback. question must already be the full user prompt in this exact shape" in message
    assert "Submission budget unchanged." in message
    assert controller.attempts == []
    assert controller.submissions_left() == controller.max_submissions


@pytest.mark.asyncio
async def test_submit_draft_tool_backfills_anchor_entity_from_entity_block(
    tmp_path: Path,
) -> None:
    config = _config_with_synthesis_output(tmp_path)
    runtime = SynthesisAgentRuntime(config)
    controller = SubmitDraftController(
        config=config,
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: runtime._build_draft_from_submission(
            db_id="sakila",
            requested_topic="assignment",
            atomic_tool_bundle=_sample_atomic_tool_bundle(),
            submission=payload,
            schema_summary={"tables": []},
        ),
    )
    controller.record_atomic_tool_call(
        tool_name="get_customer_by_id",
        params={"customer_id": 1},
        result={"store_id": 1},
    )
    controller.record_atomic_tool_call(
        tool_name="count_customer",
        params={},
        result=5,
    )
    sdk_tool = build_submit_draft_sdk_tool(controller)
    payload = _accepted_payload().model_dump(mode="json")
    payload.pop("anchor_entity")

    message = await sdk_tool.on_invoke_tool(None, json.dumps(payload))

    assert message.startswith("Accepted. solver pass rate 2/4.")
    assert controller.accepted_draft is not None
    assert controller.accepted_draft.instances[0].anchor_values == {"customer_id": 1}


@pytest.mark.asyncio
async def test_submit_draft_requires_entity_block_to_match_anchor_entity(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_customer_by_id",
        params={"customer_id": 1},
        result={"store_id": 1},
    )
    controller.record_atomic_tool_call(
        tool_name="count_customer",
        params={},
        result=5,
    )
    payload = _accepted_payload().model_copy(
        update={
            "question": _wrap_user_prompt(
                {"customer_id": 999},
                "내가 배정된 매장의 ID와 그 매장의 전체 고객 수를 알려 주세요.",
            )
        }
    )

    message = await controller.submit(payload)

    assert "Feedback. The JSON inside the <entity> block must exactly match anchor_entity." in message
    assert "Submission budget unchanged." in message
    assert controller.attempts == []
    assert controller.submissions_left() == controller.max_submissions


@pytest.mark.asyncio
async def test_submit_draft_requires_label_strengthening_after_too_easy(
    tmp_path: Path,
) -> None:
    phase_monitor = PipelinePhaseMonitorLogger(
        phase_monitor_log_path=tmp_path / "phase_monitors.jsonl",
        flow_kind="test",
        flow_id="flow:test",
    )
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=4,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: _minimal_draft(),
        phase_monitor=phase_monitor,
    )
    controller.record_atomic_tool_call(
        tool_name="get_customer_by_id",
        params={"customer_id": 1},
        result={"store_id": 1},
    )
    controller.record_atomic_tool_call(
        tool_name="count_customer",
        params={},
        result=5,
    )

    first_message = await controller.submit(_accepted_payload())

    assert "Crank search_cost" in first_message
    assert controller.required_axis is DifficultyAxis.SEARCH_COST

    controller.record_atomic_tool_call(
        tool_name="list_customer_ids",
        params={"limit": 5},
        result=[1, 2, 3],
    )
    second_payload = _accepted_payload().model_copy(
        update={
            "difficulty_vector": DifficultyVectorContract(
                search_cost=3.0,
                solution_space=1.0,
                constraint_density=1.0,
            )
        }
    )

    second_message = await controller.submit(second_payload)

    assert "do not resubmit the same label" in second_message
    assert "label_not_strengthened" in controller.attempts[-1].error_codes

    records = [
        json.loads(line)
        for line in (tmp_path / "phase_monitors.jsonl").read_text().splitlines()
    ]
    assert records[0]["actual_data"]["label_change"]["label_changed"] is None
    assert records[0]["actual_data"]["label_axis_proxies"]["search_cost_observations"] == 2
    assert records[-1]["actual_data"]["canonical_answer_preview"] == {
        "customer_count": 5,
        "store_id": 1,
    }
    assert records[-1]["actual_data"]["label_axis_proxies"]["search_cost_observations"] == 1
    assert records[-1]["actual_data"]["label_axis_proxies"]["solution_space_slots"] == 2
    assert records[-1]["actual_data"]["label_summary"].startswith("This assignment answer")
    assert records[-1]["actual_data"]["label_change"]["label_changed"] is False
    assert records[-1]["actual_data"]["label_change"]["previous_canonical_answer_preview"] == {
        "customer_count": 5,
        "store_id": 1,
    }
    assert records[-1]["actual_data"]["label_change"]["slot_count_delta"] == 0


@pytest.mark.asyncio
async def test_submit_draft_rejects_raw_identifier_fields_with_korean_particles(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="payment_history",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_payment_by_id",
        params={"payment_id": 800},
        result={"payment_id": 800, "amount": "2.99"},
    )
    controller.record_atomic_tool_call(
        tool_name="traverse_payment_to_customer_by_customer_id",
        params={"payment_id": 800},
        result={"customer_id": 29},
    )
    payload = _accepted_payload().model_copy(
        update={
            "question": _wrap_user_prompt(
                {"customer_id": 1},
                "이 payment_id를 기준으로 같은 고객의 결제 내역을 알려 주세요.",
            ),
            "canonical_answer_json": '{"amount":"2.99","customer_count":5}',
            "constraint_summary": [
                {
                    "key": "ordered",
                    "kind": "ordering",
                    "summary": "Return one grounded payment detail.",
                }
            ],
        }
    )

    message = await controller.submit(payload)

    assert "raw identifier field names" in message
    assert controller.attempts[-1].error_codes[0] == "question_raw_identifier_leak"


@pytest.mark.asyncio
async def test_submit_draft_rejects_questions_that_repeat_anchor_entity_literals(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_staff_by_id",
        params={"staff_id": 1},
        result={"first_name": "Mike", "last_name": "Hillyer"},
    )
    controller.record_atomic_tool_call(
        tool_name="get_store_by_id",
        params={"store_id": 1},
        result={"manager_staff_id": 1},
    )
    payload = _accepted_payload().model_copy(
        update={
            "anchor_entity": {"staff_id": 1},
            "question": _wrap_user_prompt(
                {"staff_id": 1},
                "1번 staff가 배정된 매장의 정보를 알려 주세요.",
            ),
        }
    )

    message = await controller.submit(payload)

    assert "raw anchor entity id" in message
    assert controller.attempts[-1].error_codes[0] == "question_anchor_entity_leak"


@pytest.mark.asyncio
async def test_submit_draft_rejects_literal_entity_placeholder_in_question(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_film_by_id",
        params={"film_id": 147},
        result={"film_id": 147},
    )
    controller.record_atomic_tool_call(
        tool_name="traverse_film_to_film_actor_by_film_id",
        params={"film_id": 147, "limit": 5},
        result=[{"actor_id": 9, "film_id": 147}],
    )
    payload = _accepted_payload().model_copy(
        update={
            "anchor_entity": {"film_id": 147},
            "question": _wrap_user_prompt(
                {"film_id": 147},
                "영화 <entity>에 배정된 배우 수를 알려 주세요.",
            ),
            "canonical_answer_json": '{"assigned_cast_count":1}',
            "label_summary": "This assignment answer is grounded in the observed assigned cast count for the requested topic.",
        }
    )

    message = await controller.submit(payload)

    assert "literal <entity> token" in message
    assert controller.attempts[-1].error_codes[0] == "question_entity_placeholder_forbidden"


@pytest.mark.asyncio
async def test_submit_draft_rejects_labels_derivable_from_single_tool_call(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="traverse_store_to_customer_by_store_id",
        params={"store_id": 1, "limit": 3},
        result=[
            {"customer_id": 1},
            {"customer_id": 2},
            {"customer_id": 3},
        ],
    )
    payload = _accepted_payload().model_copy(
        update={
            "anchor_entity": {"store_id": 1},
            "canonical_answer_json": '[{"customer_id":1},{"customer_id":2},{"customer_id":3}]',
            "question": _wrap_user_prompt(
                {"store_id": 1},
                "store_id 1 매장의 customer_id 목록을 알려주세요.",
            ),
        }
    )

    message = await controller.submit(payload)

    assert "single atomic tool call" in message
    assert controller.attempts[-1].error_codes[0] == "label_single_tool_derivable"


@pytest.mark.asyncio
async def test_submit_draft_explains_when_single_tool_shortcut_is_global(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_customer_by_id",
        params={"customer_id": 450},
        result={"customer_id": 450},
    )
    controller.record_atomic_tool_call(
        tool_name="top_staff_id_by_count_payment_desc",
        params={"limit": 3},
        result=[
            {"staff_id": 1, "value": 8057},
            {"staff_id": 2, "value": 7992},
        ],
    )
    payload = _accepted_payload().model_copy(
        update={
            "anchor_entity": {"customer_id": 450},
            "canonical_answer_json": '{"staff_id":1}',
            "question": _wrap_user_prompt(
                {"customer_id": 450},
                "이 고객과 관련된 담당 직원을 알려 주세요.",
            ),
            "constraint_summary": [
                {
                    "key": "top_staff",
                    "kind": "ordering",
                    "summary": "Return one staff member.",
                }
            ],
            "label_summary": "This payment assignment by staff label is grounded because staff 1 appears to be the top staff member.",
        }
    )

    message = await controller.submit(payload)

    assert "single global tool call that does not depend on the anchor entity" in message
    assert controller.attempts[-1].error_codes[0] == "label_single_tool_derivable"


@pytest.mark.asyncio
async def test_submit_draft_rejects_identifier_only_labels_even_when_multi_observation(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_customer_by_id",
        params={"customer_id": 1},
        result={"store_id": 1},
    )
    controller.record_atomic_tool_call(
        tool_name="traverse_store_to_address_by_address_id",
        params={"store_id": 1},
        result={"address_id": 1},
    )
    payload = _accepted_payload().model_copy(
        update={
            "canonical_answer_json": '{"store_id": 1, "address_id": 1}',
            "question": _wrap_user_prompt(
                {"customer_id": 1},
                "내 매장의 기본 정보를 알려 주세요.",
            ),
            "label_summary": "The answer combines the customer's store and the store's address.",
        }
    )

    message = await controller.submit(payload)

    assert "only a chain of internal identifier fields" in message
    assert controller.attempts[-1].error_codes[0] == "label_identifier_chain_forbidden"


@pytest.mark.asyncio
async def test_submit_draft_rejects_plural_identifier_outputs(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="payment_history",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="traverse_customer_to_payment_by_customer_id",
        params={"customer_id": 29, "limit": 5},
        result=[
            {"payment_id": 783},
            {"payment_id": 806},
        ],
    )
    controller.record_atomic_tool_call(
        tool_name="get_payment_by_id",
        params={"payment_id": 783},
        result={"payment_id": 783, "amount": "2.99"},
    )
    payload = _accepted_payload().model_copy(
        update={
            "anchor_entity": {"payment_id": 800},
            "question": _wrap_user_prompt(
                {"payment_id": 800},
                "같은 고객의 결제 두 건을 시간 순서대로 알려 주세요.",
            ),
            "canonical_answer_json": '{"payment_ids":[783,806]}',
            "constraint_summary": [
                {
                    "key": "ordered",
                    "kind": "ordering",
                    "summary": "Return exactly two payments in order.",
                }
            ],
            "label_summary": "The answer returns two payment ids for the same customer.",
        }
    )

    message = await controller.submit(payload)

    assert "only a chain of internal identifier fields" in message
    assert controller.attempts[-1].error_codes[0] == "label_identifier_chain_forbidden"


@pytest.mark.asyncio
async def test_submit_draft_rejects_answers_that_repeat_anchor_entity(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_film_by_id",
        params={"film_id": 508},
        result={"film_id": 508},
    )
    controller.record_atomic_tool_call(
        tool_name="count_inventory_by_film_id_eq",
        params={"value": 508},
        result=3,
    )
    payload = _accepted_payload().model_copy(
        update={
            "anchor_entity": {"film_id": 508},
            "canonical_answer_json": '{"film_id":508,"inventory_count":3}',
            "question": _wrap_user_prompt(
                {"film_id": 508},
                "이 영화와 연결된 재고 수를 알려 주세요.",
            ),
            "constraint_summary": [
                {
                    "key": "inventory_count",
                    "kind": "cardinality",
                    "summary": "Return the total inventory count linked to the anchored film.",
                }
            ],
            "label_summary": "This inventory assignment label is grounded in the anchored film and its observed inventory count.",
        }
    )

    message = await controller.submit(payload)

    assert "Do not repeat anchor_entity fields inside the canonical answer" in message
    assert controller.attempts[-1].error_codes[0] == "label_repeats_anchor_entity"


@pytest.mark.asyncio
async def test_submit_draft_rejects_label_summaries_that_do_not_name_selected_topic(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="payment_history",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_payment_by_id",
        params={"payment_id": 10},
        result={"payment_id": 10, "amount": "4.99"},
    )
    controller.record_atomic_tool_call(
        tool_name="traverse_payment_to_rental_by_rental_id",
        params={"payment_id": 10},
        result={"rental_id": 20},
    )
    payload = _accepted_payload().model_copy(
        update={
            "anchor_entity": {"payment_id": 10},
            "canonical_answer_json": '{"payment_amount": "4.99", "rental_id": 20}',
            "question": _wrap_user_prompt(
                {"payment_id": 10},
                "결제 하나를 따라가서 연결된 대여 정보를 알려 주세요.",
            ),
            "label_summary": "The answer is grounded in observed payment and rental rows.",
        }
    )

    message = await controller.submit(payload)

    assert "label_summary must explicitly name the selected topic" in message
    assert controller.attempts[-1].error_codes == ("selected_topic_misaligned",)


@pytest.mark.asyncio
async def test_submit_draft_rejects_string_values_not_grounded_in_tool_results(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_store_by_id",
        params={"store_id": 1},
        result={"store_id": 1},
    )
    controller.record_atomic_tool_call(
        tool_name="traverse_store_to_staff_by_manager_staff_id",
        params={"store_id": 1},
        result={"staff_id": 1},
    )
    payload = _accepted_payload().model_copy(
        update={
            "anchor_entity": {"store_id": 1},
            "canonical_answer_json": '{"staff_name":"Staff #1","store_id":1}',
            "question": _wrap_user_prompt(
                {"store_id": 1},
                "이 매장의 담당자와 매장 정보를 알려 주세요.",
            ),
        }
    )

    message = await controller.submit(payload)

    assert "not directly grounded in the observed tool results" in message
    assert "label_values_not_grounded" in controller.attempts[-1].error_codes


@pytest.mark.asyncio
async def test_submit_draft_rejects_blank_string_values_in_canonical_answer(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=2,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: (_ for _ in ()).throw(AssertionError("should not build draft")),
    )
    controller.record_atomic_tool_call(
        tool_name="get_film_by_id",
        params={"film_id": 292},
        result={"film_title": "EXAMPLE FILM"},
    )
    controller.record_atomic_tool_call(
        tool_name="count_film_actor_by_film_id_eq",
        params={"film_id": 292},
        result=8,
    )
    payload = _accepted_payload().model_copy(
        update={
            "anchor_entity": {"film_id": 292},
            "canonical_answer_json": '{"film_title":"","assigned_cast_count":8}',
            "question": _wrap_user_prompt(
                {"film_id": 292},
                "기준 영화에 배정된 배우는 모두 몇 명인가요?",
            ),
            "label_summary": "This assignment answer is grounded in the observed film and assigned cast count.",
        }
    )

    message = await controller.submit(payload)

    assert "contains blank string fields" in message
    assert "switch to counts, dates, amounts, statuses, ordering" in message
    assert controller.attempts[-1].error_codes[0] == "label_blank_string_forbidden"


def test_next_difficulty_crank_axis_wraps_back_to_first_axis() -> None:
    assert _next_difficulty_crank_axis(
        [
            DifficultyAxis.SEARCH_COST,
            DifficultyAxis.SEARCH_COST,
            DifficultyAxis.SOLUTION_SPACE,
            DifficultyAxis.SOLUTION_SPACE,
            DifficultyAxis.CONSTRAINT_DENSITY,
            DifficultyAxis.CONSTRAINT_DENSITY,
        ]
    ) is DifficultyAxis.SEARCH_COST


@pytest.mark.asyncio
async def test_submit_draft_too_hard_is_terminal_and_preserves_required_axis(
    tmp_path: Path,
) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=0,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: SynthesisAgentRuntime(
            _config_with_synthesis_output(tmp_path)
        )._build_draft_from_submission(
            db_id="sakila",
            requested_topic="assignment",
            atomic_tool_bundle=_sample_atomic_tool_bundle(),
            submission=payload,
            schema_summary={"tables": []},
        ),
    )
    controller.required_axis = DifficultyAxis.SEARCH_COST
    controller.record_atomic_tool_call(
        tool_name="get_customer_by_id",
        params={"customer_id": 1},
        result={"store_id": 1},
    )
    controller.record_atomic_tool_call(
        tool_name="count_customer",
        params={},
        result=5,
    )

    message = await controller.submit(_accepted_payload())

    assert controller.required_axis is DifficultyAxis.SEARCH_COST
    assert controller.strongest_difficulty_vector.search_cost == 2.0
    assert "too hard for the configured band" in message
    assert "Budget exhausted. No more attempts." in message
    assert controller.submissions_left() == 0


def test_reject_invalid_payload_preserves_detail_when_budget_exhausts(tmp_path: Path) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=0,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: payload,
        max_submissions=1,
    )

    with pytest.raises(ValidationError) as exc_info:
        SubmitDraftPayload.model_validate({})

    message = controller.reject_invalid_payload(parsed={}, error=exc_info.value)

    assert "canonical_answer_json is required" in message
    assert "Budget exhausted. No more attempts." in message


def test_reject_invalid_payload_explains_flat_anchor_entity_requirement(tmp_path: Path) -> None:
    controller = SubmitDraftController(
        config=_config_with_synthesis_output(tmp_path),
        requested_topic="assignment",
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            matched_solver_runs=0,
            total_solver_runs=4,
        ),
        build_draft=lambda payload: payload,
    )

    with pytest.raises(ValidationError) as exc_info:
        SubmitDraftPayload.model_validate(
            {
                **_accepted_payload().model_dump(mode="json"),
                "anchor_entity": {"entity_type": "film", "primary_key": {"film_id": 1}},
            }
        )

    message = controller.reject_invalid_payload(
        parsed={"anchor_entity": {"entity_type": "film", "primary_key": {"film_id": 1}}},
        error=exc_info.value,
    )

    assert "Feedback. anchor_entity must be a flat JSON object mapping one or more primary-key field names to scalar values" in message
    assert "Submission budget unchanged." in message
    assert controller.attempts == []
