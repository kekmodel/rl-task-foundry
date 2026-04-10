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

from pydantic import BaseModel, ConfigDict, Field, create_model

from rl_task_foundry.config.models import ProviderConfig, SolverModelConfig, SolverRuntimeConfig
from rl_task_foundry.solver.prompts import build_solver_prompt
from rl_task_foundry.tasks.models import SolverResult, TaskSpec
from rl_task_foundry.tools.models import ToolSpec
from rl_task_foundry.tools.openai_agents_adapter import (
    ToolExecutor,
    make_sdk_tool,
    make_submit_result_tool,
)
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema

ArtifactWriter = Callable[[str, dict[str, Any]], str]


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


def _python_type_for_field(field: AnswerField) -> Any:
    base_type_map: dict[str, Any] = {
        "string": str,
        "int": int,
        "float": float,
        "bool": bool,
        "date": str,
        "datetime": str,
        "enum": str,
        "list[string]": list[str],
        "list[int]": list[int],
    }
    python_type = base_type_map[field.type]
    if field.nullable:
        return python_type | None
    return python_type


def build_output_model(answer_schema: AnswerSchema) -> type[BaseModel]:
    """Build a strict Pydantic model that matches the task answer schema."""

    model_fields: dict[str, tuple[Any, Any]] = {}
    for answer_field in answer_schema.fields:
        default = None if answer_field.nullable else ...
        model_fields[answer_field.name] = (
            _python_type_for_field(answer_field),
            Field(default=default, description=answer_field.description or answer_field.name),
        )

    return create_model(
        f"TaskAnswer_{abs(hash(answer_schema.model_dump_json()))}",
        __config__=ConfigDict(extra="forbid"),
        **model_fields,
    )


def _normalize_final_output(final_output: Any, task: TaskSpec) -> dict[str, object] | None:
    if isinstance(final_output, BaseModel):
        return final_output.model_dump(mode="python")
    if isinstance(final_output, dict):
        return dict(final_output)
    if isinstance(final_output, str) and len(task.answer_schema.fields) == 1:
        return {task.answer_schema.fields[0].name: final_output}
    return None


def _raw_output_text(final_output: Any, structured_output: dict[str, object] | None) -> str:
    if isinstance(final_output, str):
        return final_output
    if isinstance(final_output, dict):
        return json.dumps(final_output, ensure_ascii=False, sort_keys=True)
    if structured_output is not None:
        return json.dumps(structured_output, ensure_ascii=False, sort_keys=True)
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


def _extract_tool_calls(run_result: Any) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for item in getattr(run_result, "new_items", []) or []:
        tool_name = _extract_tool_call_name(item)
        if tool_name is None:
            continue
        tool_calls.append(
            {
                "name": tool_name,
                "repr": repr(item),
            }
        )
    return tool_calls


def _trace_stub(kind: str, task: TaskSpec, solver_id: str, replica_index: int) -> str:
    return f"memory://{kind}/{task.task_id}/{solver_id}/replica-{replica_index}"


def _is_successful_submission(output: Any) -> bool:
    return (
        isinstance(output, dict)
        and output.get("submitted") is True
        and isinstance(output.get("answer"), dict)
    )


def _is_failed_submission(output: Any) -> bool:
    return isinstance(output, dict) and output.get("submitted") is False


def _extract_submission_output(
    final_output: Any,
    task: TaskSpec,
) -> tuple[dict[str, object] | None, str, str | None, dict[str, object]]:
    if _is_successful_submission(final_output):
        answer = final_output["answer"]
        return dict(answer), "completed", "submitted", {}
    if _is_failed_submission(final_output):
        metadata = {
            key: value
            for key, value in final_output.items()
            if key in {"error", "details"}
        }
        return None, "invalid_submit", "invalid_submit_schema", metadata
    return _normalize_final_output(final_output, task), "completed", None, {}


@dataclass(slots=True)
class OpenAIAgentsSolverBackend:
    """Thin adapter over the OpenAI Agents SDK."""

    solver_config: SolverModelConfig
    provider_config: ProviderConfig
    runtime_config: SolverRuntimeConfig
    tool_specs: list[ToolSpec] = field(default_factory=list)
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

    def _build_tools(self) -> list[object]:
        sdk_tools: list[object] = []
        for spec in self.tool_specs:
            executor = self.tool_executors.get(spec.name)
            if executor is None:
                raise RuntimeError(f"Missing tool executor for tool: {spec.name}")
            sdk_tools.append(make_sdk_tool(spec, executor))
        return sdk_tools

    @staticmethod
    def _submit_result_payload(answer: dict[str, Any]) -> dict[str, Any]:
        return {
            "submitted": True,
            "answer": dict(answer),
        }

    def _build_submit_result_tool(self, answer_model: type[BaseModel]) -> object:
        return make_submit_result_tool(
            answer_model=answer_model,
            on_submit=self._submit_result_payload,
        )

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

    def _build_session(self, sdk: SimpleNamespace, task: TaskSpec, replica_index: int) -> Any | None:
        if not self.runtime_config.sdk_sessions_enabled:
            return None
        if self.solver_config.memory_mode == "none":
            return None

        session_id = f"{task.task_id}:{self.solver_config.solver_id}:{replica_index}"
        db_path: str | Path = ":memory:"
        if self.session_db_path is not None:
            self.session_db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path = self.session_db_path
        return sdk.SQLiteSession(session_id=session_id, db_path=db_path)

    def _write_artifact(
        self,
        kind: str,
        task: TaskSpec,
        replica_index: int,
        payload: dict[str, Any],
    ) -> str:
        if self.artifact_writer is not None:
            return self.artifact_writer(kind, payload)
        if self.traces_dir is None:
            return _trace_stub(kind, task, self.solver_config.solver_id, replica_index)

        target_dir = self.traces_dir / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / (
            f"{task.task_id}__{self.solver_config.solver_id}__replica_{replica_index}.json"
        )
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(target_path)

    async def run(self, task: TaskSpec, *, replica_index: int) -> SolverResult:
        sdk = _load_sdk_components()
        # For non-OpenAI providers we default tracing off to avoid accidental export attempts.
        sdk.set_tracing_disabled(
            disabled=(not self.runtime_config.tracing) or self.provider_config.type != "openai"
        )

        output_model = build_output_model(task.answer_schema)
        tools = self._build_tools()
        tools.append(self._build_submit_result_tool(output_model))
        agent = sdk.Agent(
            name=self.solver_config.solver_id,
            instructions=build_solver_prompt(task),
            model=self._build_model(sdk),
            tools=tools,
            output_type=None,
            tool_use_behavior=self._build_tool_use_behavior(sdk),
            model_settings=sdk.ModelSettings(parallel_tool_calls=False),
        )
        session = self._build_session(sdk, task, replica_index)

        started_at = perf_counter()
        run_result = await sdk.Runner.run(
            agent,
            task.question,
            max_turns=self.runtime_config.max_turns,
            session=session,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)

        structured_output, status, termination_reason, termination_metadata = _extract_submission_output(
            run_result.final_output, task
        )
        if structured_output is None and status == "completed":
            raise RuntimeError("Solver run did not submit a structured answer via submit_result()")
        raw_output_text = _raw_output_text(run_result.final_output, structured_output)
        transcript_ref = self._write_artifact(
            "transcripts",
            task,
            replica_index,
            {
                "task_id": task.task_id,
                "solver_id": self.solver_config.solver_id,
                "replica_index": replica_index,
                "input_items": run_result.to_input_list(mode="preserve_all"),
                "final_output": structured_output if structured_output is not None else raw_output_text,
            },
        )
        tool_trace_ref = self._write_artifact(
            "tool_traces",
            task,
            replica_index,
            {
                "task_id": task.task_id,
                "solver_id": self.solver_config.solver_id,
                "replica_index": replica_index,
                "run_items": [repr(item) for item in getattr(run_result, "new_items", [])],
                "tool_calls": _extract_tool_calls(run_result),
            },
        )

        return SolverResult(
            task_id=task.task_id,
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
