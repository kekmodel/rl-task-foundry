from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig
from pydantic import ValidationError
from rl_task_foundry.schema.graph import ColumnProfile, SchemaGraph, TableProfile
from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle
from rl_task_foundry.synthesis.contracts import DifficultyAxis
from rl_task_foundry.synthesis.runtime import (
    SynthesisAgentRuntime,
    SynthesisArtifactGenerationError,
)

from rl_task_foundry.synthesis.submit_draft_tool import (
    SubmitDraftController,
    SubmitDraftPayload,
    _next_difficulty_crank_axis,
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
        tools=[],
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

    def bind_atomic_tools(self, *, tool_definitions, tool_executors) -> None:
        del tool_executors
        self.bound_tool_definitions = tool_definitions

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
        max_turns: int,
    ):
        del db_id, requested_topic, domain_name, task_language, scenario_description, schema_summary, max_turns
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
    return SubmitDraftPayload.model_validate(
        {
            "canonical_answer_json": '{"store_id": 1, "customer_count": 5}',
            "anchor_entity": {"customer_id": 1},
            "difficulty_vector": {
                "search_cost": 2.0,
                "solution_space": 1.0,
                "constraint_density": 1.0,
            },
            "question": "내가 속한 매장의 ID와 전체 고객 수를 알려 주세요.",
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
            "label_summary": "The store_id comes from the customer lookup and the total customer count comes from a separate aggregate.",
        }
    )


@pytest.mark.asyncio
async def test_synthesize_environment_draft_runs_single_agent_and_returns_accepted_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SynthesisAgentRuntime(_config_with_synthesis_output(tmp_path))
    runtime.synthesis_backends = [_FakeBackend(accept_payload=_accepted_payload())]
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
        requested_topic="assignment",
    )

    assert draft.requested_topic == "assignment"
    assert draft.selected_topic == "assignment"
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
            "question": "film_category 를 통해 category_id를 확인하세요.",
        }
    )

    message = await controller.submit(payload)

    assert "raw table names" in message
    assert controller.attempts[-1].error_codes == ("question_internal_schema_leak",)


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
        update={"question": "customer_id 1의 store_id와 customer_count를 알려 주세요."}
    )

    message = await controller.submit(payload)

    assert "raw identifier field names" in message
    assert controller.attempts[-1].error_codes == ("question_raw_identifier_leak",)


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
        tool_name="traverse_store_to_customer_via_store_id",
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
            "question": "store_id 1 매장의 customer_id 목록을 알려주세요.",
        }
    )

    message = await controller.submit(payload)

    assert "single atomic tool call" in message
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
        tool_name="traverse_store_to_address_via_address_id",
        params={"store_id": 1},
        result={"address_id": 1},
    )
    payload = _accepted_payload().model_copy(
        update={
            "canonical_answer_json": '{"store_id": 1, "address_id": 1}',
            "question": "내 매장의 기본 정보를 알려 주세요.",
            "label_summary": "The answer combines the customer's store and the store's address.",
        }
    )

    message = await controller.submit(payload)

    assert "only a chain of internal identifier fields" in message
    assert controller.attempts[-1].error_codes == ("label_identifier_chain_forbidden",)


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
