"""Shared OpenAI Agents SDK helper utilities."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

ToolExecutor = Callable[[dict[str, Any]], Any | Awaitable[Any]]
ToolInvokeCallback = Callable[[str, dict[str, Any], Any], Any | Awaitable[Any]]


def load_sdk_components(
    *,
    include_sqlite_session: bool = False,
    include_tools_to_final_output: bool = False,
) -> SimpleNamespace:
    from agents import (
        Agent,
        ModelSettings,
        OpenAIChatCompletionsModel,
        Runner,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    payload: dict[str, Any] = {
        "Agent": Agent,
        "AsyncOpenAI": AsyncOpenAI,
        "ModelSettings": ModelSettings,
        "OpenAIChatCompletionsModel": OpenAIChatCompletionsModel,
        "Runner": Runner,
        "set_tracing_disabled": set_tracing_disabled,
    }
    if include_sqlite_session:
        from agents import SQLiteSession

        payload["SQLiteSession"] = SQLiteSession
    if include_tools_to_final_output:
        from agents import ToolsToFinalOutputResult

        payload["ToolsToFinalOutputResult"] = ToolsToFinalOutputResult
    return SimpleNamespace(**payload)


def extract_token_usage(run_result: Any) -> dict[str, int]:
    usage = getattr(getattr(run_result, "context_wrapper", None), "usage", None)
    if usage is None:
        return {}
    return {
        "requests": int(getattr(usage, "requests", 0) or 0),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def extract_turn_count(run_result: Any) -> int:
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


def extract_tool_call_name(item: Any) -> str | None:
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
    match = re.search(r"tool-call\(([^)]+)\)", repr(item))
    if match:
        return match.group(1)
    return None


def normalize_tool_definition(
    definition: dict[str, Any],
    *,
    include_semantic_key: bool = False,
) -> dict[str, Any]:
    if not isinstance(definition, dict):
        raise TypeError("tool definitions must be dict-like payloads")
    normalized = {
        "name": definition["name"],
        "description": definition["description"],
        "params_schema": dict(definition.get("params_schema", {})),
    }
    if include_semantic_key:
        normalized["semantic_key"] = definition.get("semantic_key")
    return normalized


def preview_payload(value: object) -> object:
    if isinstance(value, str):
        return value[:400]
    if isinstance(value, list):
        return [preview_payload(item) for item in value[:3]]
    if isinstance(value, dict):
        preview: dict[str, object] = {}
        for key, item in list(value.items())[:6]:
            preview[str(key)] = preview_payload(item)
        return preview
    return value


def make_sdk_tool(
    definition: dict[str, Any],
    executor: ToolExecutor,
    *,
    after_invoke: ToolInvokeCallback | None = None,
) -> object:
    from agents import FunctionTool
    from agents.strict_schema import ensure_strict_json_schema

    params_json_schema = ensure_strict_json_schema(dict(definition["params_schema"]))

    async def _invoke_tool(_tool_context: Any, input_json: str) -> Any:
        payload = json.loads(input_json) if input_json else {}
        if not isinstance(payload, dict):
            raise ValueError("Tool input must be a JSON object")
        result = executor(payload)
        if hasattr(result, "__await__"):
            result = await result
        if after_invoke is not None:
            callback_result = after_invoke(str(definition["name"]), payload, result)
            if hasattr(callback_result, "__await__"):
                await callback_result
        return result

    return FunctionTool(
        name=str(definition["name"]),
        description=str(definition["description"]),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=True,
    )
