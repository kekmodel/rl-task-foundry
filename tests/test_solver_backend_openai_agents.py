from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_task_foundry.config.models import ProviderConfig, SolverModelConfig, SolverRuntimeConfig
from rl_task_foundry.solver import backend_openai_agents as backend_module
from rl_task_foundry.solver.backend_openai_agents import OpenAIAgentsSolverBackend
from rl_task_foundry.solver.runtime import SolverEpisodeInput
from rl_task_foundry.synthesis.contracts import (
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    RolloutConstraintsContract,
    TaskBundleContract,
    TaskBundleStatus,
    TaskContract,
    TaskQualityMetrics,
)


def _sample_episode() -> SolverEpisodeInput:
    task_bundle = TaskBundleContract(
        task_id="task_assignment_solver",
        db_id="sakila",
        domain="customer_support",
        topic="assignment",
        atomic_tool_set_ref="db://sakila",
        created_at=datetime(2026, 4, 12, tzinfo=timezone.utc),
        generator_version="test-version",
        tool_signature="sha256:tool",
        task_signature="sha256:task",
        status=TaskBundleStatus.ACCEPTED,
        quality_metrics=TaskQualityMetrics(),
        rollout_constraints=RolloutConstraintsContract(
            max_turns=8,
            max_episode_duration_ms=40000,
            max_tool_rows=100,
        ),
        task=TaskContract(
            question="INTERNAL QUESTION SHOULD NOT BE USED",
            topic="assignment",
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
    )
    return SolverEpisodeInput(
        task_bundle=task_bundle,
        rendered_user_prompt=(
            "<entity>\n"
            '{"order_id": 7}\n'
            "</entity>\n\n"
            "현재 배송 상태는 무엇인가요?"
        ),
    )


class _RecordingReasoningLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.filenames: list[str] = []
        self.records: list[dict[str, object]] = []

    def write_sidecar_jsonl(
        self,
        filename: str,
        records: list[dict[str, object]],
    ) -> Path:
        self.filenames.append(filename)
        self.records.extend(records)
        return self.path


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
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI, **_kwargs):
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
                    "answer": {"delivery_status": "IN_TRANSIT"},
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
            ToolsToFinalOutputResult=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )

    backend = OpenAIAgentsSolverBackend(
        solver_config=SolverModelConfig(
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
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
            tracing=True,
            sdk_sessions_enabled=True,
        ),
        sdk_tools=[],
        session_db_path=tmp_path / "sessions.sqlite",
    )

    result = await backend.run(episode)

    assert result.task_id == "task_assignment_solver"
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
    # termination_metadata now carries run_items for unified logging;
    # assert the shape rather than emptiness.
    assert "run_items" in result.termination_metadata
    assert tracing_disabled == [True]

    assert FakeAsyncOpenAI.calls[0]["base_url"] == "http://127.0.0.1:10531/v1"
    assert FakeAsyncOpenAI.calls[0]["api_key"] == "dummy"
    assert FakeRunner.calls[0]["input"] == episode.rendered_user_prompt
    assert FakeRunner.calls[0]["max_turns"] == 8
    assert FakeSQLiteSession.last_instance.session_id == "task_assignment_solver:solver_a"

    # Per-file trace writes were removed in the unified-logger
    # cleanup; run_items now travel back via termination_metadata so
    # the orchestrator can emit them through TrialEventLogger.
    assert result.termination_metadata["run_items"] == [
        {"type": "str", "text_preview": "tool-call(delivery_lookup)"},
        {"type": "str", "text_preview": "tool-call(submit_result)"},
    ]

    assert FakeAgent.last_instance.kwargs["instructions"] is None
    assert FakeAgent.last_instance.kwargs["output_type"] is None
    assert FakeAgent.last_instance.kwargs["reset_tool_choice"] is False
    assert callable(FakeAgent.last_instance.kwargs["tool_use_behavior"])
    assert any(
        getattr(tool, "name", None) == "submit_result"
        for tool in FakeAgent.last_instance.kwargs["tools"]
    )
    assert FakeAgent.last_instance.kwargs["model_settings"].kwargs["parallel_tool_calls"] is False
    assert FakeAgent.last_instance.kwargs["model_settings"].kwargs["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_openai_agents_solver_backend_enforces_episode_duration(monkeypatch):
    episode = _sample_episode()
    episode = SolverEpisodeInput(
        task_bundle=episode.task_bundle.model_copy(
            update={
                "rollout_constraints": episode.task_bundle.rollout_constraints.model_copy(
                    update={"max_episode_duration_ms": 1}
                )
            }
        ),
        rendered_user_prompt=episode.rendered_user_prompt,
    )

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, **_kwargs):
            pass

    class FakeAgent:
        def __init__(self, **_kwargs):
            pass

    class FakeRunner:
        @staticmethod
        async def run(*_args, **_kwargs):
            await asyncio.sleep(1)
            return SimpleNamespace(final_output=None, new_items=[])

    monkeypatch.setattr(
        backend_module,
        "_load_sdk_components",
        lambda: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            SQLiteSession=lambda **_kwargs: None,
            set_tracing_disabled=lambda *, disabled: None,
            ToolsToFinalOutputResult=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )

    backend = OpenAIAgentsSolverBackend(
        solver_config=SolverModelConfig(
            solver_id="solver_timeout",
            provider="openrouter",
            model="moonshotai/kimi-k2.5",
        ),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="MISSING_OPENROUTER_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SolverRuntimeConfig(max_turns=8),
    )

    result = await backend.run(episode)

    assert result.status == "failed"
    assert result.termination_reason == "TimeoutError"
    assert "max_episode_duration_ms=1" in result.termination_metadata["detail"]


@pytest.mark.asyncio
async def test_openai_agents_solver_backend_persists_raw_reasoning_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode = _sample_episode()
    reasoning_item = SimpleNamespace(
        raw_item=SimpleNamespace(
            type="reasoning",
            summary=[
                SimpleNamespace(
                    type="summary_text",
                    text="provider-visible solver reasoning",
                )
            ],
        )
    )

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI, **_kwargs):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            del agent, input, max_turns, session
            return SimpleNamespace(
                final_output={
                    "submitted": True,
                    "answer": {"delivery_status": "IN_TRANSIT"},
                },
                _current_turn=1,
                context_wrapper=SimpleNamespace(usage=SimpleNamespace(requests=1)),
                new_items=[reasoning_item, "tool-call(submit_result)"],
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
            SQLiteSession=lambda **_kwargs: None,
            set_tracing_disabled=lambda **_kwargs: None,
            ToolsToFinalOutputResult=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    event_logger = _RecordingReasoningLogger(tmp_path / "reasoning_content.jsonl")

    backend = OpenAIAgentsSolverBackend(
        solver_config=SolverModelConfig(
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        ),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SolverRuntimeConfig(max_turns=8),
        sdk_tools=[],
        event_logger=event_logger,
    )

    result = await backend.run(episode)

    assert event_logger.filenames == ["reasoning_content.jsonl"]
    assert len(event_logger.records) == 1
    assert event_logger.records[0]["actor"] == "solver"
    assert event_logger.records[0]["actor_id"] == "solver_a"
    assert event_logger.records[0]["provider"] == "codex_oauth"
    assert event_logger.records[0]["model"] == "gpt-5.4-mini"
    assert event_logger.records[0]["raw_item"] == {
        "type": "reasoning",
        "summary": [
            {
                "type": "summary_text",
                "text": "provider-visible solver reasoning",
            }
        ],
    }
    assert result.termination_metadata["reasoning_content_path"] == str(
        tmp_path / "reasoning_content.jsonl"
    )
    assert result.termination_metadata["reasoning_content_items"] == 1


@pytest.mark.asyncio
async def test_openai_agents_solver_backend_returns_failed_result_on_runner_exception(
    tmp_path, monkeypatch
):
    episode = _sample_episode()

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI, **_kwargs):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSQLiteSession:
        def __init__(self, session_id: str, db_path: str):
            self.session_id = session_id
            self.db_path = db_path

    class MaxTurnsExceeded(RuntimeError):
        pass

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            del agent, input, max_turns, session
            raise MaxTurnsExceeded("Max turns (8) exceeded")

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
            set_tracing_disabled=lambda *, disabled: None,
            ToolsToFinalOutputResult=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )

    backend = OpenAIAgentsSolverBackend(
        solver_config=SolverModelConfig(
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
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
            tracing=True,
            sdk_sessions_enabled=True,
        ),
        sdk_tools=[],
        session_db_path=tmp_path / "sessions.sqlite",
    )

    result = await backend.run(episode)

    assert result.status == "failed"
    assert result.termination_reason == "MaxTurnsExceeded"
    assert result.raw_output_text == ""
    assert result.structured_output is None
    # Error-path termination_metadata carries detail + recovered run_items.
    assert result.termination_metadata["detail"] == "Max turns (8) exceeded"
    assert "run_items" in result.termination_metadata


@pytest.mark.asyncio
async def test_openai_agents_solver_backend_reuses_cached_sdk_model_across_backends(
    tmp_path, monkeypatch
):
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
        calls: list[tuple[str, object]] = []

        def __init__(self, model: str, openai_client: FakeAsyncOpenAI, **_kwargs):
            self.model = model
            self.openai_client = openai_client
            self.__class__.calls.append((model, openai_client))

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            del agent, input, max_turns, session
            return SimpleNamespace(
                final_output={
                    "submitted": True,
                    "answer": {"delivery_status": "IN_TRANSIT"},
                },
                _current_turn=1,
                context_wrapper=SimpleNamespace(usage=SimpleNamespace(requests=1)),
                to_input_list=lambda mode="preserve_all": [],
                new_items=["tool-call(submit_result)"],
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
            SQLiteSession=lambda **_kwargs: None,
            set_tracing_disabled=lambda *, disabled: None,
            ToolsToFinalOutputResult=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    OpenAIAgentsSolverBackend.clear_model_cache()

    backend_a = OpenAIAgentsSolverBackend(
        solver_config=SolverModelConfig(
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
            memory_mode="none",
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
            tracing=True,
            sdk_sessions_enabled=False,
        ),
        sdk_tools=[],
    )
    backend_b = OpenAIAgentsSolverBackend(
        solver_config=SolverModelConfig(
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
            memory_mode="none",
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
            tracing=True,
            sdk_sessions_enabled=False,
        ),
        sdk_tools=[],
    )

    await backend_a.run(episode)
    await backend_b.run(episode)

    assert len(FakeAsyncOpenAI.calls) == 1
    assert len(FakeChatModel.calls) == 1


def test_submit_result_tool_uses_task_specific_object_schema() -> None:
    tool = backend_module._make_submit_result_tool(
        _sample_episode().task_bundle.task.output_schema
    )

    schema = tool.params_json_schema

    assert schema["type"] == "object"
    assert schema["required"] == ["delivery_status"]
    assert schema["properties"]["delivery_status"]["type"] == "string"
    assert (
        "Exact value from tool responses"
        in schema["properties"]["delivery_status"]["description"]
    )
    assert "Preserve capitalization" in schema["properties"]["delivery_status"]["description"]
    assert schema["additionalProperties"] is False
    assert "Submit the final structured result" in tool.description
    assert "Plain text final answers are invalid" in tool.description


def test_submit_result_tool_wraps_non_object_roots() -> None:
    output_schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.LIST,
            ordered=True,
            length=2,
            items=OutputFieldContract(
                name="item",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="day", type=OutputFieldType.INT),
                    OutputFieldContract(name="city", type=OutputFieldType.STRING),
                ],
            ),
        ),
        primary_output_format="json_array",
    )

    tool = backend_module._make_submit_result_tool(output_schema)
    schema = tool.params_json_schema

    assert schema["type"] == "object"
    assert schema["required"] == ["answer"]
    assert schema["properties"]["answer"]["type"] == "array"
    assert schema["properties"]["answer"]["minItems"] == 2
    assert schema["properties"]["answer"]["maxItems"] == 2
    assert schema["properties"]["answer"]["description"] == (
        "Final structured result items. Preserve the required item order."
    )
    assert schema["properties"]["answer"]["items"]["required"] == ["day", "city"]
    assert (
        "Preserve capitalization"
        in schema["properties"]["answer"]["items"]["properties"]["city"]["description"]
    )


@pytest.mark.asyncio
async def test_submit_result_tool_accepts_nullable_fields() -> None:
    output_schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.LIST,
            ordered=True,
            length=2,
            items=OutputFieldContract(
                name="item",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="medication", type=OutputFieldType.STRING),
                    OutputFieldContract(
                        name="rate",
                        type=OutputFieldType.FLOAT,
                        nullable=True,
                    ),
                ],
            ),
        ),
        primary_output_format="json_array",
    )
    tool = backend_module._make_submit_result_tool(output_schema)

    rate_schema = tool.params_json_schema["properties"]["answer"]["items"][
        "properties"
    ]["rate"]
    assert {"type": "null"} in rate_schema["anyOf"]
    assert any(
        branch.get("type") == "number"
        for branch in rate_schema["anyOf"]
        if isinstance(branch, dict)
    )

    result = await tool.on_invoke_tool(  # pyright: ignore[reportAttributeAccessIssue]
        None,
        json.dumps(
            {
                "answer": [
                    {"medication": "A", "rate": 4.5},
                    {"medication": "B", "rate": None},
                ]
            }
        ),
    )

    assert result == {
        "submitted": True,
        "answer": [
            {"medication": "A", "rate": 4.5},
            {"medication": "B", "rate": None},
        ],
    }


@pytest.mark.asyncio
async def test_submit_result_tool_rejects_wrong_list_length() -> None:
    output_schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.LIST,
            ordered=True,
            length=2,
            items=OutputFieldContract(name="item", type=OutputFieldType.STRING),
        ),
        primary_output_format="json_array",
    )
    tool = backend_module._make_submit_result_tool(output_schema)

    result = await tool.on_invoke_tool(  # pyright: ignore[reportAttributeAccessIssue]
        None,
        json.dumps({"answer": ["only-one"]}),
    )

    assert result["submitted"] is False
    assert "exactly 2 items" in result["details"][0]["msg"]


@pytest.mark.asyncio
async def test_submit_result_tool_rejects_schema_object_as_answer() -> None:
    tool = backend_module._make_submit_result_tool(
        _sample_episode().task_bundle.task.output_schema
    )

    result = await tool.on_invoke_tool(  # pyright: ignore[reportAttributeAccessIssue]
        None,
        json.dumps(
            {
                "properties": {
                    "delivery_status": {
                        "title": "Delivery Status",
                        "type": "string",
                    }
                },
                "required": ["delivery_status"],
                "title": "AnswerSchema",
                "type": "object",
            }
        ),
    )

    assert result["submitted"] is False
    assert result["error"] == "submit_result payload failed schema validation"
    assert "unexpected object keys" in result["details"][0]["msg"]


def test_extract_turn_count_preserves_explicit_zero() -> None:
    run_result = SimpleNamespace(_current_turn=0, raw_responses=["a", "b"])

    assert backend_module._extract_turn_count(run_result) == 0


def test_openai_agents_solver_backend_rejects_unsupported_provider():
    backend = OpenAIAgentsSolverBackend(
        solver_config=SolverModelConfig(
            solver_id="solver_a",
            provider="anthropic_main",
            model="claude-opus",
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
                    "details": [{"loc": ["submit_result"], "msg": "expected object"}],
                },
            )
        ],
    )

    assert result.is_final_output is True
    assert json.loads(result.final_output)["submitted"] is False


def test_tool_use_behavior_serializes_successful_submit_as_json_string():
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
                    "submitted": True,
                    "answer": {"store_id": 1},
                },
            )
        ],
    )

    assert result.is_final_output is True
    assert result.final_output == '{"answer": {"store_id": 1}, "submitted": true}'


def test_extract_submission_output_accepts_json_stringified_submit_payload():
    submitted_answer_text, structured_output, status, termination_reason, termination_metadata = (
        backend_module._extract_submission_output(
            '{"submitted": true, "answer": {"store_id": 1}}'
        )
    )

    assert submitted_answer_text == '{"store_id":1}'
    assert structured_output == {"store_id": 1}
    assert status == "completed"
    assert termination_reason == "submitted"
    assert termination_metadata == {}


def test_extract_submission_output_rejects_python_repr_success_payload():
    submitted_answer_text, structured_output, status, termination_reason, termination_metadata = (
        backend_module._extract_submission_output(
            "{'submitted': True, 'answer_text': '{\"store_id\":1}'}"
        )
    )

    assert submitted_answer_text is None
    assert structured_output is None
    assert status == "completed"
    assert termination_reason is None
    assert termination_metadata == {}


@pytest.mark.asyncio
async def test_openai_agents_solver_backend_continues_after_missing_submit_result(
    monkeypatch,
):
    episode = _sample_episode()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI, **_kwargs):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRunner:
        calls: list[dict[str, object]] = []

        @classmethod
        async def run(cls, agent, input, max_turns, session=None):
            del agent, max_turns, session
            cls.calls.append({"input": input})
            if len(cls.calls) == 1:
                return SimpleNamespace(
                    final_output='{"delivery_status":"IN_TRANSIT"}',
                    _current_turn=1,
                    context_wrapper=SimpleNamespace(
                        usage=SimpleNamespace(
                            requests=1,
                            input_tokens=5,
                            output_tokens=3,
                            total_tokens=8,
                        )
                    ),
                    to_input_list=lambda mode="preserve_all": [
                        {"role": "user", "content": input},
                        {
                            "role": "assistant",
                            "content": '{"delivery_status":"IN_TRANSIT"}',
                        },
                    ],
                    new_items=["tool-call(delivery_lookup)"],
                )
            assert isinstance(input, list)
            assert input[-1]["role"] == "user"
            assert "plain text final answers are invalid" in input[-1]["content"]
            assert "submit_result" in input[-1]["content"]
            return SimpleNamespace(
                final_output={
                    "submitted": True,
                    "answer": {"delivery_status": "IN_TRANSIT"},
                },
                _current_turn=1,
                context_wrapper=SimpleNamespace(
                    usage=SimpleNamespace(
                        requests=1,
                        input_tokens=5,
                        output_tokens=3,
                        total_tokens=8,
                    )
                ),
                to_input_list=lambda mode="preserve_all": input,
                new_items=["tool-call(submit_result)"],
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
            SQLiteSession=lambda *args, **kwargs: None,
            set_tracing_disabled=lambda **kwargs: None,
            ToolsToFinalOutputResult=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )

    backend = OpenAIAgentsSolverBackend(
        solver_config=SolverModelConfig(
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        ),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SolverRuntimeConfig(max_turns=8),
        sdk_tools=[],
    )

    result = await backend.run(episode)

    assert len(FakeRunner.calls) == 2
    assert result.status == "completed"
    assert result.termination_reason == "submitted"
    assert result.raw_output_text == '{"delivery_status":"IN_TRANSIT"}'
    assert result.structured_output == {"delivery_status": "IN_TRANSIT"}
    assert result.turn_count == 2
    assert result.token_usage == {
        "requests": 2,
        "input_tokens": 10,
        "output_tokens": 6,
        "total_tokens": 16,
    }
    assert result.termination_metadata["protocol_feedback_events"] == 1
    assert result.termination_metadata["run_items"] == [
        {"type": "str", "text_preview": "tool-call(delivery_lookup)"},
        {"type": "str", "text_preview": "tool-call(submit_result)"},
    ]
