"""OpenAI Agents SDK-backed solver runtime."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from rl_task_foundry.config.models import ProviderConfig, SolverModelConfig, SolverRuntimeConfig
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.solver.runtime import SolverEpisodeInput

ArtifactWriter = Callable[[str, dict[str, Any]], str]
ToolExecutor = Callable[[dict[str, Any]], Any]


def _load_sdk_components() -> SimpleNamespace:
    from agents import (
        Agent,
        ModelSettings,
        OpenAIChatCompletionsModel,
        Runner,
        SQLiteSession,
        ToolsToFinalOutputResult,
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
        ToolsToFinalOutputResult=ToolsToFinalOutputResult,
        set_tracing_disabled=set_tracing_disabled,
    )


def _raw_output_text(final_output: Any, submitted_answer_text: str | None) -> str:
    if submitted_answer_text is not None:
        return submitted_answer_text
    if isinstance(final_output, str):
        return final_output
    if isinstance(final_output, dict):
        return json.dumps(final_output, ensure_ascii=False, sort_keys=True)
    return str(final_output)


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
    match = re.search(r"tool-call\(([^)]+)\)", repr(item))
    if match:
        return match.group(1)
    return None


def _extract_tool_calls(
    run_result: Any,
    *,
    semantic_keys_by_name: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for item in getattr(run_result, "new_items", []) or []:
        tool_name = _extract_tool_call_name(item)
        if tool_name is None:
            continue
        payload = {"name": tool_name, "repr": repr(item)}
        if semantic_keys_by_name is not None and tool_name in semantic_keys_by_name:
            payload["semantic_key"] = semantic_keys_by_name[tool_name]
        tool_calls.append(payload)
    return tool_calls


def _trace_stub(kind: str, task_id: str, solver_id: str, replica_index: int) -> str:
    return f"memory://{kind}/{task_id}/{solver_id}/replica-{replica_index}"


def _is_successful_submission(output: Any) -> bool:
    return (
        isinstance(output, dict)
        and output.get("submitted") is True
        and isinstance(output.get("answer_text"), str)
    )


def _is_failed_submission(output: Any) -> bool:
    return isinstance(output, dict) and output.get("submitted") is False


def _extract_submission_output(
    final_output: Any,
    ) -> tuple[str | None, dict[str, object] | None, str, str | None, dict[str, object]]:
    if _is_successful_submission(final_output):
        answer_text = final_output["answer_text"]
        structured_output: dict[str, object] | None = None
        try:
            parsed = json.loads(answer_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            structured_output = dict(parsed)
        return answer_text, structured_output, "completed", "submitted", {}
    if _is_failed_submission(final_output):
        metadata = {
            key: value
            for key, value in final_output.items()
            if key in {"error", "details"}
        }
        return None, None, "invalid_submit", "invalid_submit_schema", metadata
    return None, None, "completed", None, {}


def _json_schema_type(json_type: str) -> str:
    normalized = json_type.strip().lower()
    if normalized in {"string", "integer", "number", "boolean", "object", "array"}:
        return normalized
    if normalized == "int":
        return "integer"
    if normalized == "float":
        return "number"
    if normalized == "bool":
        return "boolean"
    return "string"


def _legacy_params_schema(parameters: list[object]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter in parameters:
        name = getattr(parameter, "name", None)
        json_type = getattr(parameter, "json_type", None)
        if not isinstance(name, str) or not isinstance(json_type, str):
            continue
        properties[name] = {
            "type": _json_schema_type(json_type),
            "description": str(getattr(parameter, "description", "") or name),
        }
        if bool(getattr(parameter, "required", True)):
            required.append(name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _normalize_tool_definition(definition: object) -> dict[str, Any]:
    if isinstance(definition, dict):
        return {
            "name": definition["name"],
            "description": definition["description"],
            "params_schema": dict(definition.get("params_schema", {})),
            "semantic_key": definition.get("semantic_key"),
        }
    name = getattr(definition, "name", None)
    description = getattr(definition, "description", None)
    if not isinstance(name, str) or not isinstance(description, str):
        raise TypeError("tool definitions must be dict-like payloads or legacy ToolSpec-like objects")
    return {
        "name": name,
        "description": description,
        "params_schema": _legacy_params_schema(list(getattr(definition, "parameters", []))),
        "semantic_key": getattr(definition, "semantic_key", None),
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


def _make_submit_result_tool() -> object:
    from agents import FunctionTool
    from agents.strict_schema import ensure_strict_json_schema

    params_json_schema = ensure_strict_json_schema(
        {
            "type": "object",
            "properties": {
                "answer_text": {
                    "type": "string",
                    "description": "Final answer as a JSON string matching the rendered prompt format.",
                }
            },
            "required": ["answer_text"],
            "additionalProperties": False,
        }
    )

    async def _invoke_tool(_tool_context: Any, input_json: str) -> Any:
        try:
            payload = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError as exc:
            return {
                "submitted": False,
                "error": "submit_result payload failed schema validation",
                "details": [{"loc": ["answer_text"], "msg": f"invalid JSON: {exc}"}],
            }
        if not isinstance(payload, dict) or not isinstance(payload.get("answer_text"), str):
            return {
                "submitted": False,
                "error": "submit_result payload failed schema validation",
                "details": [{"loc": ["answer_text"], "msg": "Field required"}],
            }
        return {"submitted": True, "answer_text": payload["answer_text"]}

    return FunctionTool(
        name="submit_result",
        description=(
            "Submit the final answer as a JSON string once you have enough evidence. "
            "Call this exactly once when you are ready to finish."
        ),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=True,
    )


@dataclass(slots=True)
class OpenAIAgentsSolverBackend:
    """Thin adapter over the OpenAI Agents SDK."""

    solver_config: SolverModelConfig
    provider_config: ProviderConfig
    runtime_config: SolverRuntimeConfig
    tool_specs: list[object] = field(default_factory=list)
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    tool_executors: dict[str, ToolExecutor] = field(default_factory=dict)
    session_db_path: Path | None = None
    traces_dir: Path | None = None
    artifact_writer: ArtifactWriter | None = None

    def _resolve_api_key(self) -> str:
        env_name = self.provider_config.api_key_env
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value
        if self.provider_config.type == "openai_compatible":
            # Most local OpenAI-compatible servers accept an arbitrary non-empty key.
            return "dummy"
        raise RuntimeError(f"Required API key env var is missing: {env_name}")

    def _build_model(self, sdk: SimpleNamespace) -> Any:
        if self.provider_config.type not in {"openai", "openai_compatible"}:
            raise NotImplementedError(
                f"OpenAI Agents backend does not yet support provider type: {self.provider_config.type}"
            )
        client = sdk.AsyncOpenAI(
            api_key=self._resolve_api_key(),
            base_url=self.provider_config.base_url,
            timeout=float(self.provider_config.timeout_s),
        )
        return sdk.OpenAIChatCompletionsModel(
            model=self.solver_config.model,
            openai_client=client,
        )

    def _normalized_tool_definitions(self) -> list[dict[str, Any]]:
        raw_definitions: list[object] = (
            list(self.tool_definitions) if self.tool_definitions else list(self.tool_specs)
        )
        return [_normalize_tool_definition(definition) for definition in raw_definitions]

    def _build_tools(self) -> list[object]:
        sdk_tools: list[object] = []
        for definition in self._normalized_tool_definitions():
            tool_name = str(definition["name"])
            executor = self.tool_executors.get(tool_name)
            if executor is None:
                raise RuntimeError(f"Missing tool executor for tool: {tool_name}")
            sdk_tools.append(_make_sdk_tool(definition, executor))
        return sdk_tools

    @staticmethod
    def _build_tool_use_behavior(sdk: SimpleNamespace) -> Callable[[Any, list[Any]], Any]:
        def _finalize_on_submit(_context_wrapper: Any, tool_results: list[Any]) -> Any:
            for tool_result in tool_results:
                if getattr(getattr(tool_result, "tool", None), "name", None) != "submit_result":
                    continue
                output = getattr(tool_result, "output", None)
                if _is_successful_submission(output):
                    return sdk.ToolsToFinalOutputResult(
                        is_final_output=True,
                        final_output=output,
                    )
                if _is_failed_submission(output):
                    return sdk.ToolsToFinalOutputResult(
                        is_final_output=True,
                        final_output=output,
                    )
            return sdk.ToolsToFinalOutputResult(is_final_output=False, final_output=None)

        return _finalize_on_submit

    def _build_session(self, sdk: SimpleNamespace, task_id: str, replica_index: int) -> Any | None:
        if not self.runtime_config.sdk_sessions_enabled:
            return None
        if self.solver_config.memory_mode == "none":
            return None

        session_id = f"{task_id}:{self.solver_config.solver_id}:{replica_index}"
        db_path: str | Path = ":memory:"
        if self.session_db_path is not None:
            self.session_db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path = self.session_db_path
        return sdk.SQLiteSession(session_id=session_id, db_path=db_path)

    def _write_artifact(
        self,
        kind: str,
        task_id: str,
        replica_index: int,
        payload: dict[str, Any],
    ) -> str:
        if self.artifact_writer is not None:
            return self.artifact_writer(kind, payload)
        if self.traces_dir is None:
            return _trace_stub(kind, task_id, self.solver_config.solver_id, replica_index)

        target_dir = self.traces_dir / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / (
            f"{task_id}__{self.solver_config.solver_id}__replica_{replica_index}.json"
        )
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(target_path)

    @staticmethod
    def _coerce_run_input(input_item: object) -> tuple[str, str]:
        if not isinstance(input_item, SolverEpisodeInput):
            raise TypeError("solver runtime requires SolverEpisodeInput")
        return input_item.task_id, input_item.rendered_user_prompt

    async def run(self, episode: object, *, replica_index: int) -> SolverResult:
        sdk = _load_sdk_components()
        # For non-OpenAI providers we default tracing off to avoid accidental export attempts.
        sdk.set_tracing_disabled(
            disabled=(not self.runtime_config.tracing) or self.provider_config.type != "openai"
        )

        task_id, rendered_user_prompt = self._coerce_run_input(episode)
        tools = self._build_tools()
        tools.append(_make_submit_result_tool())
        agent = sdk.Agent(
            name=self.solver_config.solver_id,
            instructions=None,
            model=self._build_model(sdk),
            tools=tools,
            output_type=None,
            tool_use_behavior=self._build_tool_use_behavior(sdk),
            model_settings=sdk.ModelSettings(parallel_tool_calls=False),
        )
        session = self._build_session(sdk, task_id, replica_index)

        started_at = perf_counter()
        run_result = await sdk.Runner.run(
            agent,
            rendered_user_prompt,
            max_turns=self.runtime_config.max_turns,
            session=session,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)

        (
            submitted_answer_text,
            structured_output,
            status,
            termination_reason,
            termination_metadata,
        ) = _extract_submission_output(
            run_result.final_output
        )
        if submitted_answer_text is None and status == "completed":
            raise RuntimeError("Solver run did not submit an answer via submit_result()")
        raw_output_text = _raw_output_text(run_result.final_output, submitted_answer_text)
        transcript_ref = self._write_artifact(
            "transcripts",
            task_id,
            replica_index,
            {
                "task_id": task_id,
                "solver_id": self.solver_config.solver_id,
                "replica_index": replica_index,
                "input_items": run_result.to_input_list(mode="preserve_all"),
                "final_output": raw_output_text,
            },
        )
        tool_trace_ref = self._write_artifact(
            "tool_traces",
            task_id,
            replica_index,
            {
                "task_id": task_id,
                "solver_id": self.solver_config.solver_id,
                "replica_index": replica_index,
                "run_items": [repr(item) for item in getattr(run_result, "new_items", [])],
                "tool_calls": _extract_tool_calls(
                    run_result,
                    semantic_keys_by_name={
                        str(definition["name"]): str(definition["semantic_key"])
                        for definition in self._normalized_tool_definitions()
                        if definition.get("semantic_key")
                    },
                ),
            },
        )

        return SolverResult(
            task_id=task_id,
            solver_id=self.solver_config.solver_id,
            provider=self.solver_config.provider,
            model=self.solver_config.model,
            replica_index=replica_index,
            transcript_ref=transcript_ref,
            tool_trace_ref=tool_trace_ref,
            raw_output_text=raw_output_text,
            structured_output=structured_output,
            explicit_memory_events=[],
            token_usage=_extract_token_usage(run_result),
            latency_ms=latency_ms,
            turn_count=_extract_turn_count(run_result),
            status=status,
            termination_reason=termination_reason,
            termination_metadata=termination_metadata,
        )
