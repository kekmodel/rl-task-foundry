"""OpenAI Agents SDK-backed synthesis runtime backend."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from rl_task_foundry.config.models import ModelRef, ProviderConfig, SynthesisRuntimeConfig
from rl_task_foundry.synthesis.prompts import (
    build_synthesis_phase_input,
    build_synthesis_phase_instructions,
)
from rl_task_foundry.synthesis.runtime import (
    ArtifactGenerationOutput,
    CategoryInferenceOutput,
    SchemaExplorationOutput,
    SynthesisMemoryEntry,
    SynthesisPhase,
    SynthesisStageRequest,
    SynthesisStageResult,
    SynthesisToolTraceEntry,
)


def _load_sdk_components() -> SimpleNamespace:
    from agents import (
        Agent,
        ModelSettings,
        OpenAIChatCompletionsModel,
        Runner,
        SQLiteSession,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    return SimpleNamespace(
        Agent=Agent,
        AsyncOpenAI=AsyncOpenAI,
        ModelSettings=ModelSettings,
        OpenAIChatCompletionsModel=OpenAIChatCompletionsModel,
        Runner=Runner,
        SQLiteSession=SQLiteSession,
        set_tracing_disabled=set_tracing_disabled,
    )


def _trace_stub(kind: str, request: SynthesisStageRequest, provider: str, model: str) -> str:
    return f"memory://{kind}/{request.db_id}/{request.phase.value}/{provider}/{model}"


def _extract_token_usage(run_result: Any) -> dict[str, int]:
    usage = getattr(getattr(run_result, "context_wrapper", None), "usage", None)
    if usage is None:
        return {}
    return {
        "requests": int(getattr(usage, "requests", 0) or 0),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _extract_turn_count(run_result: Any) -> int:
    explicit_turn_count = getattr(run_result, "_current_turn", None)
    if explicit_turn_count is None:
        explicit_turn_count = getattr(run_result, "current_turn", None)
    if explicit_turn_count:
        return int(explicit_turn_count)
    raw_responses = getattr(run_result, "raw_responses", None)
    if isinstance(raw_responses, list) and raw_responses:
        return len(raw_responses)
    usage = getattr(getattr(run_result, "context_wrapper", None), "usage", None)
    requests = getattr(usage, "requests", 0) if usage is not None else 0
    return int(requests or 0)


def _extract_tool_call_name(item: Any) -> str | None:
    for attr in ("name", "tool_name"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value:
            return value
    raw_item = getattr(item, "raw_item", None)
    if raw_item is not None:
        for attr in ("name", "tool_name"):
            value = getattr(raw_item, attr, None)
            if isinstance(value, str) and value:
                return value
    if isinstance(item, str) and item.startswith("tool-call(") and item.endswith(")"):
        return item[len("tool-call(") : -1]
    return None


def _extract_tool_traces(
    run_result: Any,
    *,
    request: SynthesisStageRequest,
    provider: str,
    model: str,
) -> list[SynthesisToolTraceEntry]:
    traces: list[SynthesisToolTraceEntry] = []
    for item in getattr(run_result, "new_items", []) or []:
        tool_name = _extract_tool_call_name(item)
        if tool_name is None:
            continue
        traces.append(
            SynthesisToolTraceEntry(
                phase=request.phase,
                provider=provider,
                model=model,
                tool_name=tool_name,
                raw_repr=repr(item),
            )
        )
    return traces


def _extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _phase_output_type(phase: SynthesisPhase) -> type[BaseModel]:
    if phase == SynthesisPhase.SCHEMA_EXPLORATION:
        return SchemaExplorationOutput
    if phase == SynthesisPhase.CATEGORY_INFERENCE:
        return CategoryInferenceOutput
    return ArtifactGenerationOutput


def _normalize_phase_payload(phase: SynthesisPhase, final_output: Any) -> BaseModel:
    output_type = _phase_output_type(phase)
    if isinstance(final_output, output_type):
        return final_output
    if isinstance(final_output, BaseModel):
        return output_type.model_validate(final_output.model_dump(mode="python"))
    if isinstance(final_output, dict):
        return output_type.model_validate(final_output)
    if isinstance(final_output, str):
        payload_text = _extract_json_object_text(final_output)
        parsed = json.loads(payload_text)
        if not isinstance(parsed, dict):
            raise RuntimeError("synthesis backend expected a JSON object payload")
        return output_type.model_validate(parsed)
    raise RuntimeError("synthesis backend returned an unsupported final_output type")


@dataclass(slots=True)
class OpenAIAgentsSynthesisBackend:
    model_ref: ModelRef
    provider_config: ProviderConfig
    runtime_config: SynthesisRuntimeConfig
    session_db_path: Path | None = None
    traces_dir: Path | None = None

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

    def _build_session(self, sdk: SimpleNamespace, request: SynthesisStageRequest) -> Any | None:
        if not self.runtime_config.sdk_sessions_enabled or self.session_db_path is None:
            return None
        self.session_db_path.parent.mkdir(parents=True, exist_ok=True)
        return sdk.SQLiteSession(
            session_id=f"{request.db_id}:{request.phase.value}:{self.provider_name}",
            db_path=str(self.session_db_path),
        )

    def _write_artifact(
        self,
        *,
        kind: str,
        request: SynthesisStageRequest,
        payload: dict[str, Any],
    ) -> str:
        if self.traces_dir is None:
            return _trace_stub(kind, request, self.provider_name, self.model_name)
        target_dir = self.traces_dir / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / (
            f"{request.db_id}__{request.phase.value}__{self.provider_name}__{self.model_name}.json"
        )
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(target_path)

    async def run_stage(self, request: SynthesisStageRequest) -> SynthesisStageResult:
        sdk = _load_sdk_components()
        if hasattr(sdk, "set_tracing_disabled"):
            sdk.set_tracing_disabled(
                disabled=(not self.runtime_config.tracing) or self.provider_config.type != "openai"
            )
        agent = sdk.Agent(
            name=f"synthesis-{request.phase.value}",
            instructions=build_synthesis_phase_instructions(request.phase),
            model=self._build_model(sdk),
            tools=[],
            output_type=_phase_output_type(request.phase),
            model_settings=sdk.ModelSettings(parallel_tool_calls=False),
        )
        session = self._build_session(sdk, request)

        request_input = build_synthesis_phase_input(request)
        started_at = perf_counter()
        run_result = await sdk.Runner.run(
            agent,
            request_input,
            max_turns=self.runtime_config.max_turns,
            session=session,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)

        payload_model = _normalize_phase_payload(request.phase, run_result.final_output)
        tool_traces = _extract_tool_traces(
            run_result,
            request=request,
            provider=self.provider_name,
            model=self.model_name,
        )
        transcript_ref = self._write_artifact(
            kind="transcripts",
            request=request,
            payload={
                "phase": request.phase.value,
                "input": request_input,
                "final_output": payload_model.model_dump(mode="json"),
                "latency_ms": latency_ms,
                "turn_count": _extract_turn_count(run_result),
                "token_usage": _extract_token_usage(run_result),
            },
        )
        tool_trace_ref = self._write_artifact(
            kind="tool_traces",
            request=request,
            payload={
                "phase": request.phase.value,
                "tool_calls": [trace.model_dump(mode="json") for trace in tool_traces],
                "run_items": [repr(item) for item in getattr(run_result, "new_items", []) or []],
            },
        )
        summary = getattr(payload_model, "memory_summary", f"{request.phase.value} completed")
        memory_entry = SynthesisMemoryEntry(
            phase=request.phase,
            provider=self.provider_name,
            model=self.model_name,
            summary=summary if isinstance(summary, str) else f"{request.phase.value} completed",
            turn_count=_extract_turn_count(run_result),
            token_usage=_extract_token_usage(run_result),
            transcript_ref=transcript_ref,
            tool_trace_ref=tool_trace_ref,
        )
        return SynthesisStageResult(
            phase=request.phase,
            provider=self.provider_name,
            model=self.model_name,
            payload=payload_model,
            memory_entry=memory_entry,
            tool_traces=tool_traces,
        )
