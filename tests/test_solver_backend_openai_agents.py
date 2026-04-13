from __future__ import annotations

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
    build_difficulty_vector,
)


def _sample_episode() -> SolverEpisodeInput:
    task_bundle = TaskBundleContract(
        task_id="task_assignment_solver",
        db_id="sakila",
        domain="customer_support",
        topic="assignment",
        atomic_tool_set_ref="db://sakila",
        difficulty_vector=build_difficulty_vector(),
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
            "현재 배송 상태는 무엇인가요?\n\n"
            "# Submit Result Format\n"
            '{"type":"object","properties":{"delivery_status":{"type":"string"}}}\n'
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
        tool_definitions=_sample_tool_definitions(),
        tool_executors={"delivery_lookup": lambda _kwargs: {"delivery_status": "IN_TRANSIT"}},
        session_db_path=tmp_path / "sessions.sqlite",
        traces_dir=tmp_path / "traces",
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
    assert result.termination_metadata == {}
    assert tracing_disabled == [True]

    assert FakeAsyncOpenAI.calls[0]["base_url"] == "http://127.0.0.1:10531/v1"
    assert FakeAsyncOpenAI.calls[0]["api_key"] == "dummy"
    assert FakeRunner.calls[0]["input"] == episode.rendered_user_prompt
    assert FakeRunner.calls[0]["max_turns"] == 8
    assert FakeSQLiteSession.last_instance.session_id == "task_assignment_solver:solver_a"

    transcript_path = (
        tmp_path
        / "traces"
        / "transcripts"
        / "task_assignment_solver__solver_a.json"
    )
    tool_trace_path = (
        tmp_path
        / "traces"
        / "tool_traces"
        / "task_assignment_solver__solver_a.json"
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
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
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
        tool_definitions=_sample_tool_definitions(),
        tool_executors={"delivery_lookup": lambda _kwargs: {"delivery_status": "IN_TRANSIT"}},
        session_db_path=tmp_path / "sessions.sqlite",
        traces_dir=tmp_path / "traces",
    )

    result = await backend.run(episode)

    assert result.status == "failed"
    assert result.termination_reason == "MaxTurnsExceeded"
    assert result.raw_output_text == ""
    assert result.structured_output is None
    assert result.termination_metadata == {"detail": "Max turns (8) exceeded"}

    transcript_payload = json.loads(Path(result.transcript_ref).read_text(encoding="utf-8"))
    assert transcript_payload["error"]["type"] == "MaxTurnsExceeded"
    assert transcript_payload["error"]["detail"] == "Max turns (8) exceeded"


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

        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
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
                    "answer_text": '{"delivery_status":"IN_TRANSIT"}',
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
        tool_definitions=_sample_tool_definitions(),
        tool_executors={"delivery_lookup": lambda _kwargs: {"delivery_status": "IN_TRANSIT"}},
        traces_dir=tmp_path / "traces",
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
        tool_definitions=_sample_tool_definitions(),
        tool_executors={"delivery_lookup": lambda _kwargs: {"delivery_status": "IN_TRANSIT"}},
        traces_dir=tmp_path / "traces",
    )

    await backend_a.run(episode)
    await backend_b.run(episode)

    assert len(FakeAsyncOpenAI.calls) == 1
    assert len(FakeChatModel.calls) == 1


def test_submit_result_tool_is_cached_singleton() -> None:
    first = backend_module._make_submit_result_tool()
    second = backend_module._make_submit_result_tool()

    assert first is second


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
                    "details": [{"loc": ["answer_text"], "msg": "Field required"}],
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
                    "answer_text": '{"store_id":1}',
                },
            )
        ],
    )

    assert result.is_final_output is True
    assert result.final_output == '{"answer_text": "{\\"store_id\\":1}", "submitted": true}'


def test_extract_submission_output_accepts_json_stringified_submit_payload():
    submitted_answer_text, structured_output, status, termination_reason, termination_metadata = (
        backend_module._extract_submission_output(
            '{"submitted": true, "answer_text": "{\\"store_id\\":1}"}'
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


def test_solver_backend_write_artifact_creates_dir_once_per_kind(
    tmp_path: Path,
    monkeypatch,
):
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
        traces_dir=tmp_path / "traces",
    )

    mkdir_calls: list[Path] = []
    original_mkdir = Path.mkdir

    def _recording_mkdir(self: Path, *args, **kwargs):
        mkdir_calls.append(self)
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _recording_mkdir)

    backend._write_artifact("transcripts", "task_a", {"a": 1})
    first_transcript_dir_calls = mkdir_calls.count(tmp_path / "traces" / "transcripts")
    backend._write_artifact("transcripts", "task_a", {"a": 2})

    transcript_dir = tmp_path / "traces" / "transcripts"
    assert mkdir_calls.count(transcript_dir) == first_transcript_dir_calls


@pytest.mark.asyncio
async def test_openai_agents_solver_backend_writes_transcript_before_missing_submit_error(
    tmp_path, monkeypatch
):
    episode = _sample_episode()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeChatModel:
        def __init__(self, model: str, openai_client: FakeAsyncOpenAI):
            self.model = model
            self.openai_client = openai_client

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            del agent, max_turns, session
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
                to_input_list=lambda mode="preserve_all": [{"role": "user", "content": input}],
                new_items=["tool-call(delivery_lookup)"],
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
        tool_definitions=_sample_tool_definitions(),
        tool_executors={"delivery_lookup": lambda _kwargs: {"delivery_status": "IN_TRANSIT"}},
        traces_dir=tmp_path / "traces",
    )

    with pytest.raises(RuntimeError, match="did not submit an answer"):
        await backend.run(episode)

    transcript_path = (
        tmp_path
        / "traces"
        / "transcripts"
        / "task_assignment_solver__solver_a.json"
    )
    assert transcript_path.exists()
