"""OpenAI Agents SDK-backed backend for the single synthesis agent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from rl_task_foundry.config.models import ModelRef, ProviderConfig, SynthesisRuntimeConfig
from rl_task_foundry.infra.sdk_helpers import (
    ToolExecutor,
    extract_token_usage as _extract_token_usage,
    extract_tool_call_name as _extract_tool_call_name,
    extract_turn_count as _extract_turn_count,
    load_sdk_components as _shared_load_sdk_components,
    make_sdk_tool as _shared_make_sdk_tool,
    normalize_tool_definition as _normalize_tool_definition,
)
from rl_task_foundry.synthesis.prompts import (
    build_synthesis_agent_instructions,
    build_synthesis_input,
)
from rl_task_foundry.synthesis.submit_draft_tool import (
    SubmitDraftController,
    build_submit_draft_sdk_tool,
)


@dataclass(frozen=True, slots=True)
class SynthesisConversationResult:
    provider: str
    model: str
    final_output_text: str
    turn_count: int
    token_usage: dict[str, int]
    transcript_ref: str
    tool_trace_ref: str
    tool_calls: tuple[str, ...] = ()


def _trace_stub(kind: str, db_id: str, provider: str, model: str) -> str:
    return f"memory://{kind}/{db_id}/synthesis/{provider}/{model}"


def _build_agent(
    sdk: SimpleNamespace,
    *,
    model: Any,
    tools: list[object],
) -> Any:
    return sdk.Agent(
        name="synthesis",
        instructions=build_synthesis_agent_instructions(),
        model=model,
        tools=tools,
        output_type=None,
        model_settings=sdk.ModelSettings(parallel_tool_calls=False),
    )


def _load_sdk_components() -> SimpleNamespace:
    return _shared_load_sdk_components()


@dataclass(slots=True)
class OpenAIAgentsSynthesisBackend:
    model_ref: ModelRef
    provider_config: ProviderConfig
    runtime_config: SynthesisRuntimeConfig
    traces_dir: Path | None = None
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    tool_executors: dict[str, ToolExecutor] = field(default_factory=dict)
    submit_draft_controller: SubmitDraftController | None = None

    @property
    def provider_name(self) -> str:
        return self.model_ref.provider

    @property
    def model_name(self) -> str:
        return self.model_ref.model

    def _resolve_api_key(self) -> str:
        env_name = self.provider_config.api_key_env
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value
        if self.provider_config.type == "openai_compatible":
            return "dummy"
        raise RuntimeError(f"Required API key env var is missing: {env_name}")

    def _build_model(self, sdk: SimpleNamespace) -> Any:
        if self.provider_config.type not in {"openai", "openai_compatible"}:
            raise NotImplementedError(
                "OpenAI Agents synthesis backend only supports openai/openai_compatible providers"
            )
        client = sdk.AsyncOpenAI(
            api_key=self._resolve_api_key(),
            base_url=self.provider_config.base_url,
            timeout=float(self.provider_config.timeout_s),
        )
        return sdk.OpenAIChatCompletionsModel(
            model=self.model_ref.model,
            openai_client=client,
        )

    def bind_atomic_tools(
        self,
        *,
        tool_definitions: list[dict[str, Any]],
        tool_executors: dict[str, ToolExecutor],
    ) -> None:
        self.tool_definitions = [dict(definition) for definition in tool_definitions]
        self.tool_executors = dict(tool_executors)

    def bind_submit_draft_controller(self, controller: SubmitDraftController) -> None:
        self.submit_draft_controller = controller

    def _normalized_tool_definitions(self) -> list[dict[str, Any]]:
        return [_normalize_tool_definition(definition) for definition in self.tool_definitions]

    def _build_tools(self) -> list[object]:
        if self.submit_draft_controller is None:
            raise RuntimeError("submit_draft controller must be bound before run_synthesis")
        sdk_tools: list[object] = []
        for definition in self._normalized_tool_definitions():
            tool_name = str(definition["name"])
            executor = self.tool_executors.get(tool_name)
            if executor is None:
                continue
            sdk_tools.append(
                _shared_make_sdk_tool(
                    definition,
                    executor,
                    after_invoke=lambda name, payload, result, controller=self.submit_draft_controller: controller.record_atomic_tool_call(
                        tool_name=name,
                        params=payload,
                        result=result,
                    ),
                )
            )
        sdk_tools.append(build_submit_draft_sdk_tool(self.submit_draft_controller))
        return sdk_tools

    def _write_artifact(
        self,
        *,
        kind: str,
        db_id: str,
        requested_topic: str,
        payload: dict[str, Any],
    ) -> str:
        if self.traces_dir is None:
            return _trace_stub(kind, db_id, self.provider_name, self.model_name)
        target_dir = self.traces_dir / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / (
            f"{db_id}__{requested_topic}__synthesis__{self.provider_name}__{self.model_name}.json"
        )
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(target_path)

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
        max_turns: int = 50,
    ) -> SynthesisConversationResult:
        sdk = _load_sdk_components()
        if hasattr(sdk, "set_tracing_disabled"):
            sdk.set_tracing_disabled(
                disabled=(not self.runtime_config.tracing) or self.provider_config.type != "openai"
            )
        model = self._build_model(sdk)
        tools = self._build_tools()
        agent = _build_agent(sdk, model=model, tools=tools)
        request_input = build_synthesis_input(
            domain_name=domain_name,
            scenario_description=scenario_description,
            requested_topic=requested_topic,
            task_language=task_language,
            schema_summary=schema_summary,
            tool_surface_summary=tool_surface_summary,
        )
        started_at = perf_counter()
        run_result = await sdk.Runner.run(
            agent,
            request_input,
            max_turns=max_turns,
            session=None,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
        tool_calls = tuple(
            tool_name
            for item in getattr(run_result, "new_items", []) or []
            if (tool_name := _extract_tool_call_name(item)) is not None
        )
        final_output = run_result.final_output
        final_output_text = final_output if isinstance(final_output, str) else str(final_output or "")
        transcript_ref = self._write_artifact(
            kind="transcripts",
            db_id=db_id,
            requested_topic=requested_topic,
            payload={
                "input": request_input,
                "final_output_text": final_output_text,
                "latency_ms": latency_ms,
                "turn_count": _extract_turn_count(run_result),
                "token_usage": _extract_token_usage(run_result),
                "tool_calls": list(tool_calls),
            },
        )
        tool_trace_ref = self._write_artifact(
            kind="tool_traces",
            db_id=db_id,
            requested_topic=requested_topic,
            payload={
                "tool_calls": list(tool_calls),
                "run_items": [repr(item) for item in getattr(run_result, "new_items", []) or []],
                "recent_atomic_tool_calls": (
                    self.submit_draft_controller._atomic_tool_calls[-20:]
                    if self.submit_draft_controller is not None
                    else []
                ),
            },
        )
        return SynthesisConversationResult(
            provider=self.provider_name,
            model=self.model_name,
            final_output_text=final_output_text,
            turn_count=_extract_turn_count(run_result),
            token_usage=_extract_token_usage(run_result),
            transcript_ref=transcript_ref,
            tool_trace_ref=tool_trace_ref,
            tool_calls=tool_calls,
        )
