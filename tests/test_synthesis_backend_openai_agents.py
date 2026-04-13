from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_task_foundry.config.models import ModelRef, ProviderConfig, SynthesisRuntimeConfig
from rl_task_foundry.synthesis import backend_openai_agents as backend_module
from rl_task_foundry.synthesis.backend_openai_agents import OpenAIAgentsSynthesisBackend


def test_synthesis_backend_write_artifact_creates_dir_once_per_kind(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = OpenAIAgentsSynthesisBackend(
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        provider_config=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="MISSING_OPENAI_KEY",
            max_concurrency=8,
            timeout_s=30,
        ),
        runtime_config=SynthesisRuntimeConfig(),
        traces_dir=tmp_path / "traces",
    )

    mkdir_calls: list[Path] = []
    original_mkdir = Path.mkdir

    def _recording_mkdir(self: Path, *args, **kwargs):
        mkdir_calls.append(self)
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _recording_mkdir)

    backend._write_artifact(
        kind="transcripts",
        db_id="sakila",
        requested_topic="assignment",
        payload={"a": 1},
    )
    first_transcript_dir_calls = mkdir_calls.count(tmp_path / "traces" / "transcripts")
    backend._write_artifact(
        kind="transcripts",
        db_id="sakila",
        requested_topic="assignment",
        payload={"a": 2},
    )

    transcript_dir = tmp_path / "traces" / "transcripts"
    assert mkdir_calls.count(transcript_dir) == first_transcript_dir_calls


@pytest.mark.asyncio
async def test_synthesis_backend_writes_artifacts_before_reraising_runner_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            pass

    class FakeChatModel:
        def __init__(self, model: str, openai_client):
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
        lambda: SimpleNamespace(
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
        traces_dir=tmp_path / "traces",
    )
    backend.bind_submit_draft_controller(
        SimpleNamespace(
            _atomic_tool_calls=[],
            record_atomic_tool_call=lambda **_kwargs: None,
        )
    )

    with pytest.raises(MaxTurnsExceeded):
        await backend.run_synthesis(
            db_id="sakila",
            requested_topic="assignment",
            domain_name="customer support",
            task_language="en",
            scenario_description="end user support",
            schema_summary={},
            tool_surface_summary={},
            max_turns=50,
        )

    transcript_path = (
        tmp_path
        / "traces"
        / "transcripts"
        / "sakila__assignment__synthesis__codex_oauth__gpt-5.4-mini.json"
    )
    tool_trace_path = (
        tmp_path
        / "traces"
        / "tool_traces"
        / "sakila__assignment__synthesis__codex_oauth__gpt-5.4-mini.json"
    )
    assert transcript_path.exists()
    assert tool_trace_path.exists()
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    tool_trace = json.loads(tool_trace_path.read_text(encoding="utf-8"))
    assert transcript["error"]["type"] == "MaxTurnsExceeded"
    assert transcript["max_turns"] == 50
    assert tool_trace["error"]["type"] == "MaxTurnsExceeded"
