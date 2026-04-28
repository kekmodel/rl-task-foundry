from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_task_foundry.config.models import ModelRef, ProviderConfig, SynthesisRuntimeConfig
from rl_task_foundry.synthesis import backend_openai_agents as backend_module
from rl_task_foundry.synthesis.backend_openai_agents import OpenAIAgentsSynthesisBackend
from rl_task_foundry.synthesis.conversation import SynthesisConversation


def _conversation_with_controller(controller: object) -> SynthesisConversation:
    return SynthesisConversation(
        controller=controller,  # type: ignore[arg-type]
        sdk_tools=[],
        shuffle_seed="seed",
    )


class _RecordingEventLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.sidecar_filenames: list[str] = []
        self.sidecar_records: list[dict[str, object]] = []

    def log_sync(self, **kwargs: object) -> None:
        self.events.append(dict(kwargs))

    def write_sidecar_jsonl(
        self,
        filename: str,
        records: list[dict[str, object]],
    ) -> Path:
        self.sidecar_filenames.append(filename)
        self.sidecar_records.extend(records)
        return Path("/tmp/reasoning_content.jsonl")


def test_synthesis_backend_reuses_shared_model_client(monkeypatch) -> None:
    class FakeAsyncOpenAI:
        calls: list[dict[str, object]] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.calls.append(kwargs)

    class FakeChatModel:
        calls: list[tuple[str, FakeAsyncOpenAI]] = []

        def __init__(self, model: str, openai_client: FakeAsyncOpenAI, **_kwargs):
            self.__class__.calls.append((model, openai_client))
            self.model = model
            self.openai_client = openai_client

    monkeypatch.setattr(
        backend_module,
        "_shared_load_sdk_components",
        lambda **_kwargs: SimpleNamespace(
            Agent=object,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=object,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=object,
            set_tracing_disabled=lambda **_kwargs: None,
        ),
    )
    OpenAIAgentsSynthesisBackend.clear_model_cache()

    backend_a = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SynthesisRuntimeConfig(),
    )
    backend_b = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SynthesisRuntimeConfig(),
    )

    model_a = backend_a._build_model(backend_a._sdk_components())
    model_b = backend_b._build_model(backend_b._sdk_components())

    assert model_a is model_b
    assert len(FakeAsyncOpenAI.calls) == 1
    assert len(FakeChatModel.calls) == 1


@pytest.mark.asyncio
async def test_synthesis_backend_continues_after_final_output_without_submit(
    monkeypatch,
) -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeChatModel:
        def __init__(self, model: str, openai_client, **_kwargs):
            self.model = model
            self.openai_client = openai_client

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRunResult:
        def __init__(
            self,
            *,
            input_payload,
            final_output: str,
            new_items: list[object],
            turn_count: int,
        ) -> None:
            self.input = input_payload
            self.final_output = final_output
            self.new_items = new_items
            self.context_wrapper = SimpleNamespace(
                usage=SimpleNamespace(
                    requests=turn_count,
                    input_tokens=10 * turn_count,
                    output_tokens=2 * turn_count,
                    total_tokens=12 * turn_count,
                )
            )
            self._current_turn = turn_count

        def to_input_list(self, *, mode="preserve_all"):
            assert mode == "preserve_all"
            if isinstance(self.input, list):
                items = list(self.input)
            else:
                items = [{"role": "user", "content": self.input}]
            items.append({"role": "assistant", "content": self.final_output})
            return items

    class FakeRunner:
        calls: list[dict[str, object]] = []

        @classmethod
        async def run(cls, agent, input, max_turns, session=None):
            del agent, session
            cls.calls.append({"input": input, "max_turns": max_turns})
            if len(cls.calls) == 1:
                return FakeRunResult(
                    input_payload=input,
                    final_output="I can answer this now.",
                    new_items=["tool-call(query)"],
                    turn_count=1,
                )
            assert isinstance(input, list)
            assert input[-1]["role"] == "user"
            assert "Plain final output is invalid" in input[-1]["content"]
            assert "call submit_draft" in input[-1]["content"]
            return FakeRunResult(
                input_payload=input,
                final_output="Accepted: Draft accepted.",
                new_items=[
                    SimpleNamespace(
                        raw_item=SimpleNamespace(
                            type="reasoning",
                            summary=[
                                SimpleNamespace(
                                    type="summary_text",
                                    text="provider-visible composer reasoning",
                                )
                            ],
                        )
                    ),
                    "tool-call(submit_draft)",
                ],
                turn_count=1,
            )

    class FakeController:
        accepted_draft = None
        _terminated_too_hard = False
        event_logger = _RecordingEventLogger()

        def __init__(self) -> None:
            self.feedback_calls: list[dict[str, object]] = []
            self._submissions_left = 3

        def submissions_left(self) -> int:
            return self._submissions_left

        def record_missing_submit_feedback(
            self,
            *,
            final_output_text: str,
            tool_calls: tuple[str, ...],
        ) -> str:
            self.feedback_calls.append(
                {
                    "final_output_text": final_output_text,
                    "tool_calls": tool_calls,
                }
            )
            self._submissions_left -= 1
            return (
                "FeedbackError: Plain final output is invalid for this role. "
                "Next step: Continue with data tools if more evidence is needed; "
                "when the task draft is valid, call submit_draft. Do not end the "
                "run with text only. Attempts left: 2."
            )

    monkeypatch.setattr(
        backend_module,
        "_shared_load_sdk_components",
        lambda **_kwargs: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            set_tracing_disabled=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(backend_module, "build_submit_draft_sdk_tool", lambda _controller: object())

    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="openrouter", model="moonshotai/kimi-k2.5"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SynthesisRuntimeConfig(max_turns=5),
    )
    controller = FakeController()
    conversation = _conversation_with_controller(controller)

    result = await backend.run_synthesis(
        conversation=conversation,
        db_id="sakila",
        requested_topic="assignment",
        domain_name="customer support",
        task_language="en",
        scenario_description="end user support",
        schema_summary={},
        tool_surface_summary={},
        max_turns=5,
    )

    assert len(FakeRunner.calls) == 2
    assert FakeRunner.calls[1]["max_turns"] == 4
    assert controller.feedback_calls == [
        {
            "final_output_text": "I can answer this now.",
            "tool_calls": ("query",),
        }
    ]
    assert result.final_output_text == "Accepted: Draft accepted."
    assert result.turn_count == 2
    assert result.token_usage == {
        "requests": 2,
        "input_tokens": 20,
        "output_tokens": 4,
        "total_tokens": 24,
    }
    assert result.tool_calls == ("query", "submit_draft")
    assert controller.event_logger.sidecar_filenames == ["reasoning_content.jsonl"]
    assert controller.event_logger.sidecar_records[0]["actor"] == "composer"
    assert controller.event_logger.sidecar_records[0]["segment_index"] == 2
    assert controller.event_logger.sidecar_records[0]["raw_item"] == {
        "type": "reasoning",
        "summary": [
            {
                "type": "summary_text",
                "text": "provider-visible composer reasoning",
            }
        ],
    }
    completion_events = [
        event
        for event in controller.event_logger.events
        if event["event_type"] == "synthesis_completed"
    ]
    assert len(completion_events) == 1
    assert completion_events[0]["payload"]["protocol_feedback_events"] == 1
    assert completion_events[0]["payload"]["reasoning_content_items"] == 1
    assert completion_events[0]["payload"]["reasoning_content_path"] == (
        "/tmp/reasoning_content.jsonl"
    )


@pytest.mark.asyncio
async def test_synthesis_backend_continues_after_feedback_without_resubmit(
    monkeypatch,
) -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeChatModel:
        def __init__(self, model: str, openai_client, **_kwargs):
            self.model = model
            self.openai_client = openai_client

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRunResult:
        def __init__(
            self,
            *,
            input_payload,
            final_output: str,
            new_items: list[object],
            turn_count: int,
        ) -> None:
            self.input = input_payload
            self.final_output = final_output
            self.new_items = new_items
            self.context_wrapper = SimpleNamespace(
                usage=SimpleNamespace(requests=turn_count)
            )
            self._current_turn = turn_count

        def to_input_list(self, *, mode="preserve_all"):
            assert mode == "preserve_all"
            if isinstance(self.input, list):
                items = list(self.input)
            else:
                items = [{"role": "user", "content": self.input}]
            items.append({"role": "assistant", "content": self.final_output})
            return items

    class FakeRunner:
        calls: list[dict[str, object]] = []

        @classmethod
        async def run(cls, agent, input, max_turns, session=None):
            del agent, session
            cls.calls.append({"input": input, "max_turns": max_turns})
            if len(cls.calls) == 1:
                return FakeRunResult(
                    input_payload=input,
                    final_output="I found a tie-breaker and will fix the query.",
                    new_items=[
                        "tool-call(query)",
                        "tool-call(submit_draft)",
                        "tool-call(sample)",
                        "tool-call(query)",
                    ],
                    turn_count=4,
                )
            assert isinstance(input, list)
            assert input[-1]["role"] == "user"
            assert "Plain final output is invalid" in input[-1]["content"]
            assert "call submit_draft" in input[-1]["content"]
            return FakeRunResult(
                input_payload=input,
                final_output="Accepted: Draft accepted.",
                new_items=["tool-call(submit_draft)"],
                turn_count=1,
            )

    class FakeController:
        accepted_draft = None
        _terminated_too_hard = False

        def __init__(self) -> None:
            self.feedback_calls: list[dict[str, object]] = []
            self._submissions_left = 3

        def submissions_left(self) -> int:
            return self._submissions_left

        def record_missing_submit_feedback(
            self,
            *,
            final_output_text: str,
            tool_calls: tuple[str, ...],
        ) -> str:
            self.feedback_calls.append(
                {
                    "final_output_text": final_output_text,
                    "tool_calls": tool_calls,
                }
            )
            self._submissions_left -= 1
            return (
                "FeedbackError: Plain final output is invalid for this role. "
                "Next step: call submit_draft. Do not end the run with text only. "
                "Attempts left: 2."
            )

    monkeypatch.setattr(
        backend_module,
        "_shared_load_sdk_components",
        lambda **_kwargs: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            set_tracing_disabled=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(backend_module, "build_submit_draft_sdk_tool", lambda _controller: object())

    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="openrouter", model="moonshotai/kimi-k2.5"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SynthesisRuntimeConfig(max_turns=5),
    )
    controller = FakeController()

    result = await backend.run_synthesis(
        conversation=_conversation_with_controller(controller),
        db_id="sakila",
        requested_topic="assignment",
        domain_name="customer support",
        task_language="en",
        scenario_description="end user support",
        schema_summary={},
        tool_surface_summary={},
        max_turns=5,
    )

    assert len(FakeRunner.calls) == 2
    assert FakeRunner.calls[1]["max_turns"] == 1
    assert controller.feedback_calls == [
        {
            "final_output_text": "I found a tie-breaker and will fix the query.",
            "tool_calls": ("query", "submit_draft", "sample", "query"),
        }
    ]
    assert result.tool_calls == (
        "query",
        "submit_draft",
        "sample",
        "query",
        "submit_draft",
    )


@pytest.mark.asyncio
async def test_synthesis_backend_logs_unified_event_before_reraising_runner_error(
    monkeypatch,
) -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeChatModel:
        def __init__(self, model: str, openai_client, **_kwargs):
            self.model = model
            self.openai_client = openai_client

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class MaxTurnsExceeded(RuntimeError):
        pass

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            del agent, input, max_turns, session
            raise MaxTurnsExceeded("Max turns (50) exceeded")

    monkeypatch.setattr(
        backend_module,
        "_shared_load_sdk_components",
        lambda **_kwargs: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            set_tracing_disabled=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(backend_module, "build_submit_draft_sdk_tool", lambda _controller: object())

    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SynthesisRuntimeConfig(max_turns=50),
    )
    event_logger = _RecordingEventLogger()
    conversation = _conversation_with_controller(
        SimpleNamespace(
            _atomic_tool_calls=[],
            record_atomic_tool_call=lambda **_kwargs: None,
            _terminated_too_hard=False,
            event_logger=event_logger,
        )
    )

    with pytest.raises(MaxTurnsExceeded):
        await backend.run_synthesis(
            conversation=conversation,
            db_id="sakila",
            requested_topic="assignment",
            domain_name="customer support",
            task_language="en",
            scenario_description="end user support",
            schema_summary={},
            tool_surface_summary={},
            max_turns=50,
        )

    assert len(event_logger.events) == 1
    event = event_logger.events[0]
    assert event["actor"] == "composer"
    assert event["event_type"] == "synthesis_failed"
    payload = event["payload"]
    assert isinstance(payload, dict)
    assert payload["error_type"] == "MaxTurnsExceeded"
    assert payload["error_detail"] == "Max turns (50) exceeded"
    assert payload["max_turns"] == 50
    assert payload["run_items"] == []
    assert isinstance(payload["latency_ms"], int)


@pytest.mark.asyncio
async def test_synthesis_backend_requires_tool_use_and_finalizes_on_submit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeChatModel:
        def __init__(self, model: str, openai_client, **_kwargs):
            self.model = model
            self.openai_client = openai_client

    class FakeModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            FakeAgent.last_instance = self

    class FakeToolsToFinalOutputResult:
        def __init__(self, *, is_final_output: bool, final_output: str | None):
            self.is_final_output = is_final_output
            self.final_output = final_output

    class FakeRunner:
        @staticmethod
        async def run(agent, input, max_turns, session=None):
            del input, max_turns, session
            finalize = agent.kwargs["tool_use_behavior"](
                None,
                [
                    SimpleNamespace(
                        tool=SimpleNamespace(name="submit_draft"),
                        output="Accepted: Draft accepted.",
                    )
                ],
            )
            assert finalize.is_final_output is True
            assert finalize.final_output == "Accepted: Draft accepted."
            return SimpleNamespace(
                final_output="Accepted: Draft accepted.",
                new_items=[],
                context_wrapper=SimpleNamespace(usage=SimpleNamespace(requests=1)),
                _current_turn=1,
            )

    monkeypatch.setattr(
        backend_module,
        "_shared_load_sdk_components",
        lambda include_tools_to_final_output=False: SimpleNamespace(
            Agent=FakeAgent,
            AsyncOpenAI=FakeAsyncOpenAI,
            ModelSettings=FakeModelSettings,
            OpenAIChatCompletionsModel=FakeChatModel,
            Runner=FakeRunner,
            ToolsToFinalOutputResult=FakeToolsToFinalOutputResult,
            set_tracing_disabled=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(backend_module, "build_submit_draft_sdk_tool", lambda _controller: object())

    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SynthesisRuntimeConfig(max_turns=50),
    )
    conversation = _conversation_with_controller(
        SimpleNamespace(
            _atomic_tool_calls=[],
            record_atomic_tool_call=lambda **_kwargs: None,
            _terminated_too_hard=False,
        )
    )

    result = await backend.run_synthesis(
        conversation=conversation,
        db_id="sakila",
        requested_topic="assignment",
        domain_name="customer support",
        task_language="en",
        scenario_description="end user support",
        schema_summary={},
        tool_surface_summary={},
        max_turns=50,
    )

    assert result.final_output_text == "Accepted: Draft accepted."
    assert FakeAgent.last_instance.kwargs["reset_tool_choice"] is False
    assert FakeAgent.last_instance.kwargs["model_settings"].kwargs["tool_choice"] == "required"


def _make_tool_use_behavior_backend() -> OpenAIAgentsSynthesisBackend:
    return OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SynthesisRuntimeConfig(max_turns=50),
    )


def test_synthesis_tool_use_behavior_keeps_feedback_as_tool_response() -> None:
    class FakeToolsToFinalOutputResult:
        def __init__(self, *, is_final_output: bool, final_output: str | None):
            self.is_final_output = is_final_output
            self.final_output = final_output

    sdk = SimpleNamespace(ToolsToFinalOutputResult=FakeToolsToFinalOutputResult)
    controller = SimpleNamespace(_terminated_too_hard=False)
    backend = _make_tool_use_behavior_backend()
    conversation = _conversation_with_controller(controller)

    finalize = backend._build_tool_use_behavior(sdk, conversation)(
        None,
        [
            SimpleNamespace(
                tool=SimpleNamespace(name="submit_draft"),
                output=(
                    "FeedbackError: Fix the identifier chain and resubmit. "
                    "Next step: Make another data-tool call if needed,"
                    " then call submit_draft again. "
                    "Do not stop with plain text. Attempts left: 2."
                ),
            )
        ],
    )

    assert finalize.is_final_output is False
    assert finalize.final_output is None


def test_synthesis_tool_use_behavior_finalizes_budget_exhausted_feedback() -> None:
    class FakeToolsToFinalOutputResult:
        def __init__(self, *, is_final_output: bool, final_output: str | None):
            self.is_final_output = is_final_output
            self.final_output = final_output

    sdk = SimpleNamespace(ToolsToFinalOutputResult=FakeToolsToFinalOutputResult)
    controller = SimpleNamespace(_terminated_too_hard=False)
    backend = _make_tool_use_behavior_backend()
    conversation = _conversation_with_controller(controller)

    finalize = backend._build_tool_use_behavior(sdk, conversation)(
        None,
        [
            SimpleNamespace(
                tool=SimpleNamespace(name="submit_draft"),
                output=(
                    "FeedbackError: Fix the identifier chain and resubmit. "
                    "Next step: Make another data-tool call if needed,"
                    " then call submit_draft again. "
                    "Do not stop with plain text. Attempts left: 0. "
                    "BudgetExhaustedError: No more attempts."
                ),
            )
        ],
    )

    assert finalize.is_final_output is True
    assert finalize.final_output is not None
    assert "BudgetExhaustedError: No more attempts." in finalize.final_output


def test_synthesis_tool_use_behavior_finalizes_on_too_hard_termination() -> None:
    class FakeToolsToFinalOutputResult:
        def __init__(self, *, is_final_output: bool, final_output: str | None):
            self.is_final_output = is_final_output
            self.final_output = final_output

    sdk = SimpleNamespace(ToolsToFinalOutputResult=FakeToolsToFinalOutputResult)
    controller = SimpleNamespace(_terminated_too_hard=True)
    backend = _make_tool_use_behavior_backend()
    conversation = _conversation_with_controller(controller)

    finalize = backend._build_tool_use_behavior(sdk, conversation)(
        None,
        [
            SimpleNamespace(
                tool=SimpleNamespace(name="submit_draft"),
                output=(
                    "RejectedError: Draft is overconstrained. Primary issue: "
                    "The current draft is not reachable enough. This conversation is "
                    "terminated. Attempts left: 0."
                ),
            )
        ],
    )

    assert finalize.is_final_output is True
    assert finalize.final_output is not None
