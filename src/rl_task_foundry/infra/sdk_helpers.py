"""Shared OpenAI Agents SDK helper utilities."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable, Callable
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rl_task_foundry.config.models import ProviderConfig

ToolExecutor = Callable[[dict[str, Any]], Any | Awaitable[Any]]
ToolInvokeCallback = Callable[[str, dict[str, Any], Any], Any | Awaitable[Any]]


def normalize_tool_result(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date | time):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, str):
        return value.rstrip()
    if isinstance(value, dict):
        return {str(key): normalize_tool_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_tool_result(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_tool_result(item) for item in value]
    return value


_THINKING_MODE_MODEL_MARKERS: tuple[str, ...] = (
    "qwen",
    "deepseek-r",
    "reasoning",
)


def tool_choice_for_model(model: str) -> str:
    """Return the correct `tool_choice` value for the target model.

    Alibaba's qwen thinking-mode gateway rejects `tool_choice="required"` and
    tool-object forms. Other reasoning-first providers (DeepSeek R-series) have
    the same constraint. Relax to `"auto"` for those; keep `"required"` for
    non-thinking models where we want the SDK to enforce tool-use each turn.
    """

    lowered = model.lower()
    for marker in _THINKING_MODE_MODEL_MARKERS:
        if marker in lowered:
            return "auto"
    return "required"


def build_reasoning_replay_hook() -> Callable[[Any], bool]:
    """Return a hook that tells openai-agents when to replay `reasoning_content`.

    Qwen3.5's canonical chat template expects reasoning for in-flight assistant
    turns (between the last user message and the current one) to be carried
    forward via `reasoning_content`. The openai-agents default only replays
    for DeepSeek, so Qwen loses that continuity without this hook.
    """

    from agents.models.reasoning_content_replay import (
        default_should_replay_reasoning_content,
    )

    def _should_replay(context: Any) -> bool:
        target_model = (getattr(context, "model", "") or "").lower()
        if "qwen" in target_model:
            return True
        return default_should_replay_reasoning_content(context)

    return _should_replay


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
    if explicit_turn_count is not None:
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


def preview_payload(
    value: object,
    *,
    max_string_length: int,
    max_list_items: int,
    max_dict_items: int,
) -> object:
    if isinstance(value, str):
        return value[:max_string_length]
    if isinstance(value, list):
        return [
            preview_payload(
                item,
                max_string_length=max_string_length,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
            for item in value[:max_list_items]
        ]
    if isinstance(value, dict):
        preview: dict[str, object] = {}
        for key, item in list(value.items())[:max_dict_items]:
            preview[str(key)] = preview_payload(
                item,
                max_string_length=max_string_length,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
        return preview
    return value


def resolve_provider_api_key(
    provider: ProviderConfig,
    *,
    missing_error_factory: Callable[[str], Exception] | None = None,
) -> str:
    env_value = os.environ.get(provider.api_key_env)
    if env_value:
        return env_value
    if provider.type == "openai_compatible":
        return "dummy"
    message = f"Required API key env var is missing: {provider.api_key_env}"
    if missing_error_factory is None:
        raise RuntimeError(message)
    raise missing_error_factory(message)


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
        try:
            result = executor(payload)
            if hasattr(result, "__await__"):
                result = await result
            result = normalize_tool_result(result)
        except Exception as exc:
            result = f"ToolError: {type(exc).__name__}: {exc}. Fix the tool arguments and continue."
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


def write_json_artifact(
    *,
    traces_dir: Path,
    created_dirs: set[Path],
    kind: str,
    filename: str,
    payload: dict[str, Any],
) -> str:
    target_dir = traces_dir / kind
    if target_dir not in created_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        created_dirs.add(target_dir)
    target_path = target_dir / filename
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return str(target_path)
