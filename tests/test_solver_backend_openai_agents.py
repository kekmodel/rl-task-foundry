from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from rl_task_foundry.config.models import ProviderConfig, SolverModelConfig, SolverRuntimeConfig
from rl_task_foundry.solver import backend_openai_agents as backend_module
from rl_task_foundry.solver.backend_openai_agents import OpenAIAgentsSolverBackend
from rl_task_foundry.solver.prompts import build_solver_prompt
from rl_task_foundry.solver.runtime import SolverEpisodeInput
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


def _sample_episode() -> SolverEpisodeInput:
    environment = EnvironmentContract(
        env_id="env_assignment_solver",
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
            max_turns=8,
            max_episode_duration_ms=40000,
            max_tool_rows=100,
        ),
        task=TaskContract(
            question="INTERNAL QUESTION SHOULD NOT BE USED",
            category=CategoryTaxonomy.ASSIGNMENT,
            output_schema=OutputSchemaContract(
                root=OutputFieldContract(
                    name="answer",
                    type=OutputFieldType.OBJECT,
                    fields=[
                        OutputFieldContract(
                            name="delivery_status",
                            type=OutputFieldType.STRING,
                        )
                    ],
                ),
                primary_output_format="json_object",
            ),
        ),
        solution=SolutionContract(),
        verifier=VerifierContract(facts_schema=MaterializedFactsSchema()),
        shadow_verifier=ShadowVerifierContract(facts_schema=MaterializedFactsSchema()),
        instance_space=InstanceSpaceContract(
            anchor_query=AnchorQueryContract(
                sql="SELECT order_id FROM orders ORDER BY order_id",
                outputs=["order_id"],
            )
        ),
        cross_instance_set=CrossInstanceSet(minimum_required=1),
    )
    return SolverEpisodeInput(
        environment=environment,
        instance_id="instance_0007",
        rendered_user_prompt=(
            "현재 배송 상태는 무엇인가요?\n\n"
            "Submit Result Format:\n"
            '{"type":"object","properties":{"delivery_status":{"type":"string"}}}\n\n'
            "Call submit_result with a JSON string."
        ),
    )


def _sample_tool_definitions() -> list[dict[str, object]]:
    return [
        {
            "name": "delivery_lookup",
            "description": "Look up delivery status for an order id.",
            "params_schema": {
                "type": "object",
                "properties": {
                    "anchor_order_id": {
                        "type": "integer",
                        "description": "Order id",
                    }
                },
                "required": ["anchor_order_id"],
                "additionalProperties": False,
            },
            "returns_schema": {
                "type": "object",
                "properties": {
                    "delivery_status": {
                        "type": "string",
                    }
                },
                "required": ["delivery_status"],
                "additionalProperties": False,
            },
            "semantic_key": "orders.shipments:lookup",
        }
    ]


@pytest.mark.asyncio
async def test_openai_agents_solver_backend_returns_solver_result(tmp_path, monkeypatch):
    tracing_disabled: list[bool] = []
    episode = _sample_episode()

    class FakeAsyncOpenAI:
        calls: list[dict[str, object]] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.calls.append(kwargs)

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.last_instance = self

    class FakeSQLiteSession:
        last_instance = None

        def __init__(self, session_id: str, db_path: str):
            self.session_id = session_id
            self.db_path = db_path
            self.__class__.last_instance = self

    class FakeRunner:
        calls: list[dict[str, object]] = []

        @staticmethod
        async def run(agent, input, max_turns, session=None):
            FakeRunner.calls.append(
                {
                    "agent": agent,
                    "input": input,
                    "max_turns": max_turns,
                    "session": session,
                }
            )
            return SimpleNamespace(
                final_output={
                    "submitted": True,
                    "answer_text": '{"delivery_status":"IN_TRANSIT"}',
                },
                _current_turn=3,
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(
                        requests=1,
                        input_tokens=11,
                        output_tokens=7,
                        total_tokens=18,
                    )
                ),
                to_input_list=lambda mode="preserve_all": [{"role": "user", "content": input}],
                new_items=["tool-call(delivery_lookup)", "tool-call(submit_result)"],
            )

    monkeypatch.setattr(
        backend_module,
        "_load_sdk_components",
        lambda: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            SQLiteSession=FakeSQLiteSession,
            set_tracing_disabled=lambda *, disabled: tracing_disabled.append(disabled),
        ),
    )

    backend = OpenAIAgentsSolverBackend(
        solver_config=SolverModelConfig(
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
            replicas=1,
            memory_mode="session_only",
            summarization_mode="off",
        ),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SolverRuntimeConfig(
            max_turns=8,
            structured_output_required=True,
            tracing=True,
            sdk_sessions_enabled=True,
            canonical_state_store="run_db",
        ),
        tool_definitions=_sample_tool_definitions(),
        tool_executors={"delivery_lookup": lambda _kwargs: {"delivery_status": "IN_TRANSIT"}},
        session_db_path=tmp_path / "sessions.sqlite",
        traces_dir=tmp_path / "traces",
    )

    result = await backend.run(episode, replica_index=2)

    assert result.task_id == "env_assignment_solver__instance_0007"
    assert result.solver_id == "solver_a"
    assert result.raw_output_text == '{"delivery_status":"IN_TRANSIT"}'
    assert result.structured_output == {"delivery_status": "IN_TRANSIT"}
    assert result.token_usage == {
        "requests": 1,
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert result.turn_count == 3
    assert result.status == "completed"
    assert result.termination_reason == "submitted"
    assert result.termination_metadata == {}
    assert tracing_disabled == [True]

    assert FakeAsyncOpenAI.calls[0]["base_url"] == "http://127.0.0.1:10531/v1"
    assert FakeAsyncOpenAI.calls[0]["api_key"] == "dummy"
    assert FakeRunner.calls[0]["input"] == episode.rendered_user_prompt
    assert FakeRunner.calls[0]["max_turns"] == 8
    assert FakeSQLiteSession.last_instance.session_id == "env_assignment_solver__instance_0007:solver_a:2"

    transcript_path = (
        tmp_path
        / "traces"
        / "transcripts"
        / "env_assignment_solver__instance_0007__solver_a__replica_2.json"
    )
    tool_trace_path = (
        tmp_path
        / "traces"
        / "tool_traces"
        / "env_assignment_solver__instance_0007__solver_a__replica_2.json"
    )
    assert result.transcript_ref == str(transcript_path)
    assert result.tool_trace_ref == str(tool_trace_path)

    transcript_payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript_payload["final_output"] == '{"delivery_status":"IN_TRANSIT"}'
    assert transcript_payload["input_items"][0]["content"] == episode.rendered_user_prompt

    tool_trace_payload = json.loads(tool_trace_path.read_text(encoding="utf-8"))
    assert tool_trace_payload["run_items"] == [
        "'tool-call(delivery_lookup)'",
        "'tool-call(submit_result)'",
    ]
    assert tool_trace_payload["tool_calls"] == [
        {
            "name": "delivery_lookup",
            "repr": "'tool-call(delivery_lookup)'",
            "semantic_key": "orders.shipments:lookup",
        },
        {"name": "submit_result", "repr": "'tool-call(submit_result)'"},
    ]

    assert FakeAgent.last_instance.kwargs["instructions"] == build_solver_prompt()
    assert "delivery_status" not in FakeAgent.last_instance.kwargs["instructions"]
    assert "INTERNAL QUESTION SHOULD NOT BE USED" not in FakeAgent.last_instance.kwargs["instructions"]
    assert FakeAgent.last_instance.kwargs["output_type"] is None
    assert callable(FakeAgent.last_instance.kwargs["tool_use_behavior"])
    assert any(
        getattr(tool, "name", None) == "submit_result"
        for tool in FakeAgent.last_instance.kwargs["tools"]
    )
    assert FakeAgent.last_instance.kwargs["model_settings"].kwargs["parallel_tool_calls"] is False


def test_openai_agents_solver_backend_rejects_unsupported_provider():
    backend = OpenAIAgentsSolverBackend(
        solver_config=SolverModelConfig(
            solver_id="solver_a",
            provider="anthropic_main",
            model="claude-opus",
            replicas=1,
        ),
        provider_config=ProviderConfig(
            type="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            max_concurrency=4,
            timeout_s=30,
        ),
        runtime_config=SolverRuntimeConfig(),
    )

    with pytest.raises(NotImplementedError):
        backend._build_model(SimpleNamespace())


def test_tool_use_behavior_stops_on_failed_submit():
    class FakeToolsToFinalOutputResult:
        def __init__(self, *, is_final_output: bool, final_output):
            self.is_final_output = is_final_output
            self.final_output = final_output

    behavior = OpenAIAgentsSolverBackend._build_tool_use_behavior(
        SimpleNamespace(ToolsToFinalOutputResult=FakeToolsToFinalOutputResult)
    )

    result = behavior(
        None,
        [
            SimpleNamespace(
                tool=SimpleNamespace(name="submit_result"),
                output={
                    "submitted": False,
                    "error": "submit_result payload failed schema validation",
                    "details": [{"loc": ["answer_text"], "msg": "Field required"}],
                },
            )
        ],
    )

    assert result.is_final_output is True
    assert result.final_output["submitted"] is False


def test_extract_submission_output_classifies_invalid_submit():
    submitted_answer_text, structured_output, status, termination_reason, termination_metadata = (
        backend_module._extract_submission_output(
            {
                "submitted": False,
                "error": "submit_result payload failed schema validation",
                "details": [{"loc": ["answer_text"], "msg": "Field required"}],
            }
        )
    )

    assert submitted_answer_text is None
    assert structured_output is None
    assert status == "invalid_submit"
    assert termination_reason == "invalid_submit_schema"
    assert termination_metadata["error"] == "submit_result payload failed schema validation"


def test_extract_submission_output_keeps_non_object_json_as_raw_text_only():
    submitted_answer_text, structured_output, status, termination_reason, termination_metadata = (
        backend_module._extract_submission_output(
            {
                "submitted": True,
                "answer_text": '["Seoul","Busan"]',
            }
        )
    )

    assert submitted_answer_text == '["Seoul","Busan"]'
    assert structured_output is None
    assert status == "completed"
    assert termination_reason == "submitted"
    assert termination_metadata == {}
