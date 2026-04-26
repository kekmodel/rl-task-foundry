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

    def log_sync(self, **kwargs: object) -> None:
        self.events.append(dict(kwargs))


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
