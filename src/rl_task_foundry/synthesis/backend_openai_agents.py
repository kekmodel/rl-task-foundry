"""OpenAI Agents SDK-backed synthesis runtime backend."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from rl_task_foundry.config.models import ModelRef, ProviderConfig, SynthesisRuntimeConfig
from rl_task_foundry.synthesis.contracts import normalize_topic
from rl_task_foundry.synthesis.prompts import (
    build_synthesis_phase_input,
    build_synthesis_phase_instructions,
)
from rl_task_foundry.synthesis.runtime import (
    CategoryInferenceOutput,
    LabelConstructionOutput,
    SchemaExplorationOutput,
    SynthesisMemoryEntry,
    SynthesisPhase,
    SynthesisStageRequest,
    SynthesisStageResult,
    SynthesisToolTraceEntry,
    TaskSynthesisOutput,
)
from rl_task_foundry.synthesis.tool_runtime import ToolExecutor


def _load_sdk_components() -> SimpleNamespace:
    from agents import (
        Agent,
        AgentOutputSchema,
        ModelSettings,
        OpenAIChatCompletionsModel,
        Runner,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    return SimpleNamespace(
        Agent=Agent,
        AgentOutputSchema=AgentOutputSchema,
        AsyncOpenAI=AsyncOpenAI,
        ModelSettings=ModelSettings,
        OpenAIChatCompletionsModel=OpenAIChatCompletionsModel,
        Runner=Runner,
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
    return int(getattr(usage, "requests", 0) or 0)


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
    if phase == SynthesisPhase.LABEL_CONSTRUCTION:
        return LabelConstructionOutput
    return TaskSynthesisOutput


def _normalize_tool_definition(definition: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(definition, dict):
        raise TypeError("tool definitions must be dict-like payloads")
    return {
        "name": definition["name"],
        "description": definition["description"],
        "params_schema": dict(definition.get("params_schema", {})),
    }


def _make_sdk_tool(definition: dict[str, Any], executor: ToolExecutor) -> object:
    from agents import FunctionTool
    from agents.strict_schema import ensure_strict_json_schema

    params_json_schema = ensure_strict_json_schema(dict(definition["params_schema"]))

    async def _invoke_tool(_tool_context: Any, input_json: str) -> Any:
        payload = json.loads(input_json) if input_json else {}
        if not isinstance(payload, dict):
            raise ValueError("Tool input must be a JSON object")
        result = executor(payload)
        if hasattr(result, "__await__"):
            return await result
        return result

    return FunctionTool(
        name=str(definition["name"]),
        description=str(definition["description"]),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=True,
    )


def _build_agent(
    sdk: SimpleNamespace,
    *,
    request: SynthesisStageRequest,
    model: Any,
    structured_output: bool,
    tools: list[object] | None = None,
) -> Any:
    output_type: Any | None = None
    if structured_output:
        output_type = _phase_output_type(request.phase)
        output_schema_factory = getattr(sdk, "AgentOutputSchema", None)
        if output_schema_factory is not None:
            output_type = output_schema_factory(output_type, strict_json_schema=False)
    model_settings_kwargs: dict[str, Any] = {"parallel_tool_calls": False}
    if request.phase == SynthesisPhase.SCHEMA_EXPLORATION and tools:
        model_settings_kwargs["tool_choice"] = "required"
    return sdk.Agent(
        name=f"synthesis-{request.phase.value}",
        instructions=build_synthesis_phase_instructions(request.phase),
        model=model,
        tools=tools or [],
        output_type=output_type,
        model_settings=sdk.ModelSettings(**model_settings_kwargs),
    )


def _phase_max_turns(
    request: SynthesisStageRequest,
    *,
    runtime_max_turns: int,
    tools: list[object],
) -> int:
    if request.phase == SynthesisPhase.SCHEMA_EXPLORATION and tools:
        return max(len(tools) * 2, runtime_max_turns)
    return runtime_max_turns


def _normalize_phase_payload(
    request: SynthesisStageRequest,
    final_output: Any,
) -> tuple[BaseModel, list[str]]:
    output_type = _phase_output_type(request.phase)
    if isinstance(final_output, output_type):
        return final_output, []
    if isinstance(final_output, BaseModel):
        return output_type.model_validate(final_output.model_dump(mode="python")), []
    if isinstance(final_output, str):
        parsed = json.loads(_extract_json_object_text(final_output))
    elif isinstance(final_output, dict):
        parsed = final_output
    else:
        raise RuntimeError("synthesis backend returned an unsupported final_output type")
    if not isinstance(parsed, dict):
        raise RuntimeError("synthesis backend expected a JSON object payload")
    normalized_payload, repair_codes = _coerce_phase_payload_dict(request, parsed)
    return output_type.model_validate(normalized_payload), repair_codes


def _coerce_phase_payload_dict(
    request: SynthesisStageRequest,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    repair_codes: list[str] = []
    if request.phase == SynthesisPhase.SCHEMA_EXPLORATION:
        if {"domain_hypothesis", "candidate_topics", "sample_observations"} <= set(payload):
            return payload, []
        candidate_topics: list[str] = []
        for key in ("candidate_topics", "candidate_categories", "categories"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        topic = normalize_topic(item)
                        if topic not in candidate_topics:
                            candidate_topics.append(topic)
        explicit_topic = payload.get("topic") or payload.get("category")
        if isinstance(explicit_topic, str) and explicit_topic.strip():
            topic = normalize_topic(explicit_topic)
            if topic not in candidate_topics:
                candidate_topics.append(topic)
        sample_observations = payload.get("sample_observations") or payload.get("observations") or []
        if not isinstance(sample_observations, list):
            sample_observations = []
        normalized = {
            "domain_hypothesis": payload.get("domain_hypothesis")
            or payload.get("domain_fit")
            or "schema exploration completed",
            "candidate_topics": candidate_topics,
            "sample_observations": [
                item.strip() for item in sample_observations if isinstance(item, str) and item.strip()
            ],
            "memory_summary": payload.get("memory_summary", "schema exploration completed"),
        }
        repair_codes.append("schema_exploration_remapped")
        return normalized, repair_codes

    if request.phase == SynthesisPhase.CATEGORY_INFERENCE:
        if {"selected_topic", "rationale"} <= set(payload):
            return payload, []
        selected_topic = payload.get("selected_topic") or payload.get("selected_category") or payload.get("topic") or payload.get("category")
        normalized = {
            "selected_topic": normalize_topic(selected_topic),
            "rationale": payload.get("rationale") or "topic inference completed",
            "memory_summary": payload.get("memory_summary", "topic inference completed"),
        }
        repair_codes.append("category_inference_remapped")
        return normalized, repair_codes

    if request.phase == SynthesisPhase.LABEL_CONSTRUCTION:
        if {
            "canonical_answer_json",
            "anchor_entity",
            "difficulty_vector",
            "label_summary",
        } <= set(payload):
            return payload, []
        canonical_answer_json: str | None = None
        for key in ("canonical_answer_json", "answer_json"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                canonical_answer_json = value.strip()
                break
        if canonical_answer_json is None and "canonical_answer" in payload:
            canonical_answer_json = json.dumps(
                payload["canonical_answer"],
                ensure_ascii=False,
                sort_keys=True,
            )
            repair_codes.append("label_construction_answer_jsonized")
        normalized = {
            "canonical_answer_json": canonical_answer_json,
            "anchor_entity": payload.get("anchor_entity") or payload.get("anchor") or {},
            "difficulty_vector": payload.get("difficulty_vector") or payload.get("difficulty") or {},
            "instance_parameters": payload.get("instance_parameters") or payload.get("params") or {},
            "label_summary": payload.get("label_summary")
            or payload.get("rationale")
            or "label construction completed",
            "memory_summary": payload.get("memory_summary", "label construction completed"),
        }
        repair_codes.append("label_construction_remapped")
        return normalized, repair_codes

    if {"question", "constraint_summary", "instance_space"} <= set(payload):
        return payload, []
    instance_space = payload.get("instance_space")
    if instance_space is None and "anchor_query" in payload:
        instance_space = {
            "anchor_query": payload.get("anchor_query"),
            "parameters": payload.get("parameters") or {},
            "sampling": payload.get("sampling") or {"strategy": "deterministic_hash", "seed": 0},
            "instance_count": payload.get("instance_count"),
        }
        repair_codes.append("task_synthesis_instance_space_nested")
    normalized = {
        "question": payload.get("question") or payload.get("user_request") or "",
        "constraint_summary": payload.get("constraint_summary") or payload.get("constraints") or [],
        "instance_space": instance_space,
        "memory_summary": payload.get("memory_summary", "task synthesis completed"),
    }
    repair_codes.append("task_synthesis_remapped")
    return normalized, repair_codes


@dataclass(slots=True)
class OpenAIAgentsSynthesisBackend:
    model_ref: ModelRef
    provider_config: ProviderConfig
    runtime_config: SynthesisRuntimeConfig
    session_db_path: Path | None = None
    traces_dir: Path | None = None
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    tool_executors: dict[str, ToolExecutor] = field(default_factory=dict)

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

    def _normalized_tool_definitions(self) -> list[dict[str, Any]]:
        return [_normalize_tool_definition(definition) for definition in self.tool_definitions]

    def _build_tools(self) -> list[object]:
        sdk_tools: list[object] = []
        for definition in self._normalized_tool_definitions():
            tool_name = str(definition["name"])
            executor = self.tool_executors.get(tool_name)
            if executor is None:
                continue
            sdk_tools.append(_make_sdk_tool(definition, executor))
        return sdk_tools

    def _build_session(self, sdk: SimpleNamespace, request: SynthesisStageRequest) -> Any | None:
        del sdk, request
        return None

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
        model = self._build_model(sdk)
        tools = self._build_tools() if request.phase == SynthesisPhase.SCHEMA_EXPLORATION else []
        agent = _build_agent(
            sdk,
            request=request,
            model=model,
            structured_output=True,
            tools=tools,
        )
        session = self._build_session(sdk, request)
        request_input = build_synthesis_phase_input(request)
        max_turns = _phase_max_turns(
            request,
            runtime_max_turns=self.runtime_config.max_turns,
            tools=tools,
        )
        started_at = perf_counter()
        try:
            run_result = await sdk.Runner.run(
                agent,
                request_input,
                max_turns=max_turns,
                session=session,
            )
        except Exception as exc:
            if type(exc).__name__ != "ModelBehaviorError":
                raise
            fallback_agent = _build_agent(
                sdk,
                request=request,
                model=model,
                structured_output=False,
                tools=tools,
            )
            run_result = await sdk.Runner.run(
                fallback_agent,
                request_input,
                max_turns=max_turns,
                session=None,
            )
        latency_ms = int((perf_counter() - started_at) * 1000)

        tool_traces = _extract_tool_traces(
            run_result,
            request=request,
            provider=self.provider_name,
            model=self.model_name,
        )
        try:
            payload_model, payload_repair_codes = _normalize_phase_payload(
                request, run_result.final_output
            )
        except Exception as exc:
            self._write_artifact(
                kind="normalize_failures",
                request=request,
                payload={
                    "phase": request.phase.value,
                    "input": request_input,
                    "raw_final_output": run_result.final_output,
                    "latency_ms": latency_ms,
                    "turn_count": _extract_turn_count(run_result),
                    "token_usage": _extract_token_usage(run_result),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise
        transcript_ref = self._write_artifact(
            kind="transcripts",
            request=request,
            payload={
                "phase": request.phase.value,
                "input": request_input,
                "final_output": payload_model.model_dump(mode="json"),
                "payload_repair_codes": payload_repair_codes,
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
            payload_repair_codes=payload_repair_codes,
            memory_entry=memory_entry,
            tool_traces=tool_traces,
        )
