"""OpenAI Agents SDK-backed solver runtime."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, ClassVar

from rl_task_foundry.config.models import ProviderConfig, SolverModelConfig, SolverRuntimeConfig
from rl_task_foundry.infra.sdk_helpers import (
    build_reasoning_preserving_chat_completions_model,
    build_reasoning_replay_hook,
    tool_choice_for_model,
)
from rl_task_foundry.infra.sdk_helpers import (
    extract_raw_reasoning_records as _extract_raw_reasoning_records,
)
from rl_task_foundry.infra.sdk_helpers import (
    extract_run_error_items as _extract_run_error_items,
)
from rl_task_foundry.infra.sdk_helpers import (
    extract_token_usage as _extract_token_usage,
)
from rl_task_foundry.infra.sdk_helpers import (
    extract_tool_call_name as _extract_tool_call_name,
)
from rl_task_foundry.infra.sdk_helpers import (
    extract_turn_count as _extract_turn_count,
)
from rl_task_foundry.infra.sdk_helpers import (
    load_sdk_components as _shared_load_sdk_components,
)
from rl_task_foundry.infra.sdk_helpers import (
    resolve_provider_api_key as _resolve_provider_api_key,
)
from rl_task_foundry.infra.sdk_helpers import (
    summarize_run_item as _summarize_run_item,
)
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.solver.runtime import SolverEpisodeInput
from rl_task_foundry.synthesis.canonicalize import (
    CanonicalizationError,
    canonical_json,
    canonicalize_output,
)
from rl_task_foundry.synthesis.contracts import (
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
)


def _load_sdk_components() -> SimpleNamespace:
    return _shared_load_sdk_components(
        include_sqlite_session=True,
        include_tools_to_final_output=True,
    )


def _raw_output_text(final_output: Any, submitted_answer_text: str | None) -> str:
    if submitted_answer_text is not None:
        return submitted_answer_text
    if isinstance(final_output, str):
        return final_output
    if isinstance(final_output, dict):
        return json.dumps(final_output, ensure_ascii=False, sort_keys=True)
    return str(final_output)


def _failure_raw_output_text(error: Exception) -> str:
    del error
    return ""


def _final_output_preview(final_output: Any, *, limit: int = 2000) -> str:
    if isinstance(final_output, str):
        text = final_output
    elif isinstance(final_output, dict):
        text = json.dumps(final_output, ensure_ascii=False, sort_keys=True)
    else:
        text = str(final_output)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _extract_tool_calls(run_result: Any) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for item in getattr(run_result, "new_items", []) or []:
        tool_name = _extract_tool_call_name(item)
        if tool_name is None:
            continue
        tool_calls.append({"name": tool_name, "repr": repr(item)})
    return tool_calls


def _add_token_usage(
    total: dict[str, int],
    usage: dict[str, int],
) -> None:
    for key, value in usage.items():
        total[key] = total.get(key, 0) + int(value)


def _append_missing_submit_feedback_input(run_result: Any) -> list[object] | None:
    to_input_list = getattr(run_result, "to_input_list", None)
    if not callable(to_input_list):
        return None
    continuation = list(to_input_list(mode="preserve_all"))
    continuation.append(
        {
            "role": "user",
            "content": (
                "Tool schema reminder: plain text final answers are invalid. "
                "Call submit_result with the final structured answer matching "
                "its schema. Do not end with text only."
            ),
        }
    )
    return continuation


def _persist_reasoning_records(
    *,
    event_logger: object | None,
    records: list[dict[str, Any]],
    provider: str,
    model: str,
    solver_id: str,
) -> tuple[str | None, int]:
    if not records:
        return None, 0
    writer = getattr(event_logger, "write_sidecar_jsonl", None)
    if not callable(writer):
        return None, len(records)
    enriched = [
        {
            "actor": "solver",
            "actor_id": solver_id,
            "provider": provider,
            "model": model,
            **record,
        }
        for record in records
    ]
    path = writer("reasoning_content.jsonl", enriched)
    return (str(path) if path is not None else None), len(records)


def _is_successful_submission(output: Any) -> bool:
    return (
        isinstance(output, dict)
        and output.get("submitted") is True
        and ("answer" in output or isinstance(output.get("answer_text"), str))
    )


def _is_failed_submission(output: Any) -> bool:
    return isinstance(output, dict) and output.get("submitted") is False


def _extract_submission_output(
    final_output: Any,
) -> tuple[str | None, dict[str, object] | None, str, str | None, dict[str, object]]:
    if isinstance(final_output, str):
        normalized_output = _parse_submission_output_string(final_output)
        if normalized_output is not None:
            final_output = normalized_output
    if _is_successful_submission(final_output):
        assert isinstance(final_output, dict)
        if "answer" in final_output:
            answer = final_output["answer"]
            answer_text = canonical_json(answer, default=str)
        else:
            answer_text = final_output["answer_text"]
            try:
                answer = json.loads(answer_text)
            except json.JSONDecodeError:
                answer = None
        structured_output: dict[str, object] | None = None
        if isinstance(answer, dict):
            structured_output = dict(answer)
        return answer_text, structured_output, "completed", "submitted", {}
    if _is_failed_submission(final_output):
        assert isinstance(final_output, dict)
        metadata = {
            key: value for key, value in final_output.items() if key in {"error", "details"}
        }
        return None, None, "invalid_submit", "invalid_submit_schema", metadata
    return None, None, "completed", None, {}


def _parse_submission_output_string(final_output: str) -> dict[str, Any] | None:
    text = final_output.strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


_SUBMIT_RESULT_EXACT_VALUE_CONTRACT = (
    "Use exact values copied from tool responses. Do not change string "
    "capitalization, spelling, punctuation, whitespace, numeric precision, or "
    "date/time format unless the user explicitly asks for that transformation."
)


def _make_submit_result_tool(output_schema: OutputSchemaContract) -> object:
    from agents import FunctionTool
    from agents.strict_schema import ensure_strict_json_schema

    params_json_schema = ensure_strict_json_schema(
        _submit_result_params_schema(output_schema)
    )

    async def _invoke_tool(_tool_context: Any, input_json: str) -> Any:
        try:
            payload = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError as exc:
            return {
                "submitted": False,
                "error": "submit_result payload failed schema validation",
                "details": [{"loc": ["submit_result"], "msg": f"invalid JSON: {exc}"}],
            }
        if not isinstance(payload, dict):
            return {
                "submitted": False,
                "error": "submit_result payload failed schema validation",
                "details": [{"loc": ["submit_result"], "msg": "expected object"}],
            }
        if output_schema.root.type is OutputFieldType.OBJECT:
            submitted_answer = payload
        else:
            submitted_answer = payload.get("answer")
        try:
            canonical_answer = canonicalize_output(output_schema, submitted_answer)
        except CanonicalizationError as exc:
            return {
                "submitted": False,
                "error": "submit_result payload failed schema validation",
                "details": [{"loc": ["submit_result"], "msg": str(exc)}],
            }
        return {"submitted": True, "answer": canonical_answer}

    return FunctionTool(
        name="submit_result",
        description=(
            "Submit the final structured result. Plain text final answers are "
            "invalid; call this tool once the answer is ready."
        ),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=True,
    )


def _submit_result_params_schema(output_schema: OutputSchemaContract) -> dict[str, object]:
    if output_schema.root.type is OutputFieldType.OBJECT:
        return _output_field_json_schema(output_schema.root)
    return {
        "type": "object",
        "description": "Final structured result payload.",
        "properties": {
            "answer": _output_field_json_schema(output_schema.root),
        },
        "required": ["answer"],
        "additionalProperties": False,
    }


def _output_field_json_schema(field: OutputFieldContract) -> dict[str, object]:
    if field.type is OutputFieldType.OBJECT:
        schema: dict[str, object] = {
            "type": "object",
            "description": _output_field_description(field),
            "properties": {
                child.name: _output_field_json_schema(child)
                for child in field.fields
            },
            "required": [child.name for child in field.fields],
            "additionalProperties": False,
        }
    elif field.type is OutputFieldType.LIST:
        schema = {
            "type": "array",
            "description": _output_field_description(field),
            "items": (
                _output_field_json_schema(field.items)
                if field.items is not None
                else {}
            ),
        }
        if field.length is not None:
            schema["minItems"] = field.length
            schema["maxItems"] = field.length
    elif field.type is OutputFieldType.INT:
        schema = {"type": "integer", "description": _output_field_description(field)}
    elif field.type is OutputFieldType.FLOAT:
        schema = {"type": "number", "description": _output_field_description(field)}
    elif field.type is OutputFieldType.BOOL:
        schema = {"type": "boolean", "description": _output_field_description(field)}
    elif field.type in {
        OutputFieldType.STRING,
        OutputFieldType.DATE,
        OutputFieldType.DATETIME,
    }:
        schema = {"type": "string", "description": _output_field_description(field)}
    elif field.type is OutputFieldType.ENUM:
        schema = {
            "type": "string",
            "description": _output_field_description(field),
            "enum": list(field.enum_values),
        }
    else:  # pragma: no cover
        raise ValueError(f"unsupported output field type: {field.type}")

    if field.nullable:
        return {
            "anyOf": [schema, {"type": "null"}],
            "description": _output_field_description(field),
        }
    return schema


def _output_field_description(field: OutputFieldContract) -> str:
    prefix = f"{field.description.strip()} " if field.description.strip() else ""
    if field.type is OutputFieldType.LIST:
        order = " Preserve the required item order." if field.ordered else ""
        return f"{prefix}Final structured result items.{order}"
    if field.type is OutputFieldType.OBJECT:
        return f"{prefix}Final structured result object."
    if field.type is OutputFieldType.STRING:
        return (
            f"{prefix}Exact value from tool responses. Preserve capitalization, "
            "spelling, punctuation, and whitespace."
        )
    if field.type is OutputFieldType.DATE:
        return f"{prefix}Exact date value from tool responses. Preserve date format."
    if field.type is OutputFieldType.DATETIME:
        return (
            f"{prefix}Exact date/time value from tool responses. Preserve "
            "date/time format and timezone."
        )
    if field.type is OutputFieldType.ENUM:
        return (
            f"{prefix}Use one allowed value exactly as shown. Preserve "
            "capitalization, spelling, punctuation, and whitespace."
        )
    if field.type is OutputFieldType.FLOAT:
        return (
            f"{prefix}Exact numeric value from tool responses. Do "
            "not round or reformat it."
        )
    if field.type is OutputFieldType.INT:
        return f"{prefix}Exact integer value from tool responses."
    if field.type is OutputFieldType.BOOL:
        return f"{prefix}Exact boolean value from tool responses."
    return f"{prefix}{_SUBMIT_RESULT_EXACT_VALUE_CONTRACT}"


@dataclass(slots=True)
class OpenAIAgentsSolverBackend:
    """Thin adapter over the OpenAI Agents SDK."""

    solver_config: SolverModelConfig
    provider_config: ProviderConfig
    runtime_config: SolverRuntimeConfig
    sdk_tools: list[object] = field(default_factory=list)
    session_db_path: Path | None = None
    event_logger: object | None = None
    _sdk: SimpleNamespace | None = field(default=None, init=False, repr=False)
    _model: Any | None = field(default=None, init=False, repr=False)
    _shared_models: ClassVar[dict[tuple[int, int, str, str | None, str, float, str], Any]] = {}

    @classmethod
    def clear_model_cache(cls) -> None:
        cls._shared_models.clear()

    def _sdk_components(self) -> SimpleNamespace:
        if self._sdk is None:
            self._sdk = _load_sdk_components()
        return self._sdk

    def _build_model(self, sdk: SimpleNamespace) -> Any:
        if self._model is not None:
            return self._model
        if self.provider_config.type not in {"openai", "openai_compatible"}:
            raise NotImplementedError(
                "OpenAI Agents backend does not yet support"
                f" provider type: {self.provider_config.type}"
            )
        api_key = _resolve_provider_api_key(self.provider_config)
        cache_key = (
            id(sdk.AsyncOpenAI),
            id(sdk.OpenAIChatCompletionsModel),
            self.provider_config.type,
            self.provider_config.base_url,
            api_key,
            float(self.provider_config.timeout_s),
            self.solver_config.model,
        )
        cached = self._shared_models.get(cache_key)
        if cached is not None:
            self._model = cached
            return self._model
        client = sdk.AsyncOpenAI(
            api_key=api_key,
            base_url=self.provider_config.base_url,
            timeout=float(self.provider_config.timeout_s),
            max_retries=self.provider_config.max_retries,
        )
        self._model = build_reasoning_preserving_chat_completions_model(
            sdk,
            model=self.solver_config.model,
            openai_client=client,
            should_replay_reasoning_content=build_reasoning_replay_hook(),
        )
        self._shared_models[cache_key] = self._model
        return self._model

    def _build_tools(self) -> list[object]:
        return list(self.sdk_tools)

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
                        final_output=json.dumps(output, ensure_ascii=False, sort_keys=True),
                    )
                if _is_failed_submission(output):
                    return sdk.ToolsToFinalOutputResult(
                        is_final_output=True,
                        final_output=json.dumps(output, ensure_ascii=False, sort_keys=True),
                    )
            return sdk.ToolsToFinalOutputResult(is_final_output=False, final_output=None)

        return _finalize_on_submit

    def _build_session(self, sdk: SimpleNamespace, task_id: str) -> Any | None:
        if not self.runtime_config.sdk_sessions_enabled:
            return None
        if self.solver_config.memory_mode == "none":
            return None

        session_id = f"{task_id}:{self.solver_config.solver_id}"
        db_path: str | Path = ":memory:"
        if self.session_db_path is not None:
            self.session_db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path = self.session_db_path
        return sdk.SQLiteSession(session_id=session_id, db_path=db_path)

    @staticmethod
    def _coerce_run_input(input_item: object) -> tuple[str, str]:
        if not isinstance(input_item, SolverEpisodeInput):
            raise TypeError("solver runtime requires SolverEpisodeInput")
        return input_item.task_id, input_item.rendered_user_prompt

    async def run(self, episode: SolverEpisodeInput) -> SolverResult:
        sdk = self._sdk_components()
        # For non-OpenAI providers we default tracing off to avoid accidental export attempts.
        sdk.set_tracing_disabled(
            disabled=(not self.runtime_config.tracing) or self.provider_config.type != "openai"
        )

        task_id, rendered_user_prompt = self._coerce_run_input(episode)
        tools = self._build_tools()
        tools.append(_make_submit_result_tool(episode.task_bundle.task.output_schema))
        agent = sdk.Agent(
            name=self.solver_config.solver_id,
            instructions=None,
            model=self._build_model(sdk),
            tools=tools,
            output_type=None,
            tool_use_behavior=self._build_tool_use_behavior(sdk),
            model_settings=sdk.ModelSettings(
                parallel_tool_calls=False,
                tool_choice=tool_choice_for_model(self.solver_config.model),
            ),
            reset_tool_choice=False,
        )
        session = self._build_session(sdk, task_id)

        started_at = perf_counter()
        run_input: str | list[object] = rendered_user_prompt
        run_result: Any | None = None
        total_turn_count = 0
        token_usage: dict[str, int] = {}
        run_items: list[Any] = []
        protocol_feedback_events = 0
        try:
            while True:
                turns_left = max(1, self.runtime_config.max_turns - total_turn_count)
                run_result = await sdk.Runner.run(
                    agent,
                    run_input,
                    max_turns=turns_left,
                    session=session,
                )
                total_turn_count += max(1, _extract_turn_count(run_result))
                _add_token_usage(token_usage, _extract_token_usage(run_result))
                run_items.extend(getattr(run_result, "new_items", []) or [])

                submitted_answer_text, _, status, _, _ = _extract_submission_output(
                    run_result.final_output
                )
                missing_submit = submitted_answer_text is None and status == "completed"
                if not missing_submit:
                    break
                if protocol_feedback_events >= 1:
                    break
                if total_turn_count >= self.runtime_config.max_turns:
                    break
                continuation_input = _append_missing_submit_feedback_input(run_result)
                if continuation_input is None:
                    break
                protocol_feedback_events += 1
                run_input = continuation_input
        except Exception as exc:
            latency_ms = int((perf_counter() - started_at) * 1000)
            raw_output_text = _failure_raw_output_text(exc)
            recovered_items = _extract_run_error_items(exc)
            reasoning_path, reasoning_items = _persist_reasoning_records(
                event_logger=self.event_logger,
                records=_extract_raw_reasoning_records(
                    getattr(getattr(exc, "run_data", None), "new_items", []) or []
                ),
                provider=self.solver_config.provider,
                model=self.solver_config.model,
                solver_id=self.solver_config.solver_id,
            )
            termination_metadata: dict[str, object] = {
                "detail": str(exc),
                "run_items": recovered_items,
            }
            if reasoning_path is not None:
                termination_metadata["reasoning_content_path"] = reasoning_path
            if reasoning_items:
                termination_metadata["reasoning_content_items"] = reasoning_items
            return SolverResult(
                task_id=task_id,
                solver_id=self.solver_config.solver_id,
                provider=self.solver_config.provider,
                model=self.solver_config.model,
                raw_output_text=raw_output_text,
                structured_output=None,
                explicit_memory_events=[],
                token_usage={},
                latency_ms=latency_ms,
                turn_count=0,
                status="failed",
                termination_reason=exc.__class__.__name__,
                termination_metadata=termination_metadata,
            )
        assert run_result is not None
        latency_ms = int((perf_counter() - started_at) * 1000)

        (
            submitted_answer_text,
            structured_output,
            status,
            termination_reason,
            termination_metadata,
        ) = _extract_submission_output(run_result.final_output)

        # run_items summary travels back on termination_metadata so the
        # orchestrator can include it in the solver_run_completed event.
        metadata = dict(termination_metadata)
        metadata["run_items"] = [
            _summarize_run_item(item) for item in run_items
        ]
        if protocol_feedback_events:
            metadata["protocol_feedback_events"] = protocol_feedback_events
        reasoning_path, reasoning_items = _persist_reasoning_records(
            event_logger=self.event_logger,
            records=_extract_raw_reasoning_records(
                run_items
            ),
            provider=self.solver_config.provider,
            model=self.solver_config.model,
            solver_id=self.solver_config.solver_id,
        )
        if reasoning_path is not None:
            metadata["reasoning_content_path"] = reasoning_path
        if reasoning_items:
            metadata["reasoning_content_items"] = reasoning_items
        if submitted_answer_text is None and status == "completed":
            metadata["final_output_preview"] = _final_output_preview(
                run_result.final_output
            )
            return SolverResult(
                task_id=task_id,
                solver_id=self.solver_config.solver_id,
                provider=self.solver_config.provider,
                model=self.solver_config.model,
                raw_output_text="",
                structured_output=None,
                explicit_memory_events=[],
                token_usage=token_usage,
                latency_ms=latency_ms,
                turn_count=total_turn_count,
                status="invalid_submit",
                termination_reason="missing_submit_result",
                termination_metadata=metadata,
            )

        raw_output_text = _raw_output_text(run_result.final_output, submitted_answer_text)

        return SolverResult(
            task_id=task_id,
            solver_id=self.solver_config.solver_id,
            provider=self.solver_config.provider,
            model=self.solver_config.model,
            raw_output_text=raw_output_text,
            structured_output=structured_output,
            explicit_memory_events=[],
            token_usage=token_usage,
            latency_ms=latency_ms,
            turn_count=total_turn_count,
            status=status,
            termination_reason=termination_reason,
            termination_metadata=metadata,
        )
