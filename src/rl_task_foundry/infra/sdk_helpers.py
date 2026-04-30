"""Shared OpenAI Agents SDK helper utilities."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable, Callable
from datetime import date, datetime, time
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from rl_task_foundry.config.models import ProviderConfig

ToolExecutor = Callable[[dict[str, Any]], Any | Awaitable[Any]]
ToolInvokeCallback = Callable[[str, dict[str, Any], Any], Any | Awaitable[Any]]
TextToolCallParser = Callable[
    [str],
    list[dict[str, Any]],
]


_KIMI_TOOL_ID_RE = re.compile(
    r"(?:functions\.)?(?P<name>[A-Za-z_][A-Za-z0-9_.-]*):(?P<index>[0-9]+)"
)
_TOOL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"


def normalize_chat_completion_reasoning_for_agents(
    response: Any,
    *,
    model: str | None = None,
    allowed_tool_names: set[str] | frozenset[str] | None = None,
) -> None:
    """Preserve reasoning and run model-specific tool-call normalization.

    The Agents SDK chat-completions converter promotes `reasoning_content` into
    a `ReasoningItem`, but OpenRouter normalizes thinking output as `reasoning`
    on the chat message. Copying it into `reasoning_content` keeps the raw
    provider-visible reasoning available to the analysis log.

    Provider/model families expose failed tool-call transport differently. Keep
    each repair as a model-specific adapter instead of one global parser. The
    first adapter is Kimi: OpenRouter may put Kimi's native tool-call template in
    `reasoning` while leaving `message.tool_calls` empty.
    """

    text_tool_call_parsers = _tool_call_text_parsers_for_model(model)
    choices = getattr(response, "choices", None) or []
    for choice in choices:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        if not getattr(message, "reasoning_content", None):
            reasoning = getattr(message, "reasoning", None)
            if reasoning:
                setattr(message, "reasoning_content", reasoning)
        promoted = _extract_provider_text_tool_calls(
            message,
            allowed_tool_names=allowed_tool_names,
            parsers=text_tool_call_parsers,
        )
        existing_tool_calls = getattr(message, "tool_calls", None)
        if existing_tool_calls:
            repaired = _replace_invalid_tool_calls_from_provider_text(
                existing_tool_calls,
                promoted,
                allowed_tool_names=allowed_tool_names,
            )
            if repaired is existing_tool_calls:
                continue
            setattr(message, "tool_calls", repaired)
            try:
                setattr(choice, "finish_reason", "tool_calls")
            except Exception:
                pass
            continue
        if not promoted:
            continue
        setattr(message, "tool_calls", promoted)
        if _content_is_only_tool_call_text(getattr(message, "content", None)):
            setattr(message, "content", None)
        try:
            setattr(choice, "finish_reason", "tool_calls")
        except Exception:
            pass


def _extract_provider_text_tool_calls(
    message: Any,
    *,
    allowed_tool_names: set[str] | frozenset[str] | None,
    parsers: tuple[TextToolCallParser, ...],
) -> list[Any]:
    if not allowed_tool_names or not parsers:
        return []
    texts = _candidate_tool_call_texts(message)
    calls: list[Any] = []
    seen_ids: set[str] = set()
    for text in texts:
        for parser in parsers:
            for parsed in parser(text):
                if not _tool_name_allowed(parsed["name"], allowed_tool_names):
                    continue
                call_id = str(parsed["id"])
                if call_id in seen_ids:
                    continue
                seen_ids.add(call_id)
                calls.append(_make_chat_completion_tool_call(parsed))
    return calls


def _tool_call_text_parsers_for_model(model: str | None) -> tuple[TextToolCallParser, ...]:
    if model is None:
        return ()
    lowered = model.lower()
    if "kimi" in lowered:
        return (_parse_kimi_template_tool_calls,)
    return ()


def _candidate_tool_call_texts(message: Any) -> list[str]:
    texts: list[str] = []
    for attr in ("reasoning", "reasoning_content", "content"):
        value = getattr(message, attr, None)
        texts.extend(_text_values(value))
    return [text for text in texts if text.strip()]


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        texts: list[str] = []
        for item in value:
            if isinstance(item, str):
                texts.append(item)
                continue
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text" and isinstance(item.get("text"), str):
                    texts.append(item["text"])
                elif item_type in {"tool_use", "function_call"}:
                    texts.append(json.dumps(item, ensure_ascii=False))
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        return texts
    return []


def _parse_kimi_template_tool_calls(
    text: str,
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for match in _KIMI_TOOL_ID_RE.finditer(text):
        name = match.group("name")
        args_start = _find_json_object_start(text, match.end())
        if args_start is None:
            continue
        arguments = _decode_provider_tool_arguments(text[args_start:])
        if arguments is None:
            continue
        if not isinstance(arguments, dict):
            continue
        parsed.append(
            {
                "id": f"functions.{name}:{match.group('index')}",
                "name": name,
                "arguments": arguments,
            }
        )
    return parsed


def _decode_provider_tool_arguments(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    try:
        arguments, _end = decoder.raw_decode(text)
        return arguments
    except json.JSONDecodeError:
        repaired = _escape_unescaped_control_chars_in_json_strings(text)
        if repaired == text:
            return None
    try:
        arguments, _end = decoder.raw_decode(repaired)
    except json.JSONDecodeError:
        return None
    return arguments


def _escape_unescaped_control_chars_in_json_strings(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    changed = False
    for char in text:
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            result.append(char)
            escaped = True
            continue
        if char == '"':
            result.append(char)
            in_string = not in_string
            continue
        if in_string and ord(char) < 0x20:
            result.append(_json_control_char_escape(char))
            changed = True
            continue
        result.append(char)
    if not changed:
        return text
    return "".join(result)


def _json_control_char_escape(char: str) -> str:
    escapes = {
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }
    return escapes.get(char, f"\\u{ord(char):04x}")


def _replace_invalid_tool_calls_from_provider_text(
    existing_tool_calls: Any,
    promoted_tool_calls: list[Any],
    *,
    allowed_tool_names: set[str] | frozenset[str] | None,
) -> Any:
    if not promoted_tool_calls or not isinstance(existing_tool_calls, list | tuple):
        return existing_tool_calls
    promoted_by_name: dict[str, list[Any]] = {}
    for call in promoted_tool_calls:
        name = _tool_call_function_name(call)
        if not name:
            continue
        promoted_by_name.setdefault(name, []).append(call)

    changed = False
    repaired: list[Any] = []
    for call in existing_tool_calls:
        name = _tool_call_function_name(call)
        if (
            name
            and _tool_name_allowed(name, allowed_tool_names)
            and not _tool_call_has_valid_object_arguments(call)
            and promoted_by_name.get(name)
        ):
            repaired.append(promoted_by_name[name].pop(0))
            changed = True
            continue
        repaired.append(call)
    if not changed:
        return existing_tool_calls
    if isinstance(existing_tool_calls, tuple):
        return tuple(repaired)
    return repaired


def _tool_call_function_name(call: Any) -> str | None:
    function = _tool_call_function(call)
    name = _dual_attr(function, "name")
    return name if isinstance(name, str) and name else None


def _tool_call_has_valid_object_arguments(call: Any) -> bool:
    function = _tool_call_function(call)
    arguments = _dual_attr(function, "arguments")
    if isinstance(arguments, dict):
        return True
    if not isinstance(arguments, str):
        return False
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)


def _tool_call_function(call: Any) -> Any:
    return _dual_attr(call, "function")


def _find_json_object_start(text: str, start: int) -> int | None:
    candidate = text.find("{", start)
    if candidate < 0:
        return None
    argument_marker = text.find(_TOOL_ARGUMENT_BEGIN, start)
    if argument_marker >= 0 and argument_marker < candidate:
        marker_end = argument_marker + len(_TOOL_ARGUMENT_BEGIN)
        marked_candidate = text.find("{", marker_end)
        if marked_candidate >= 0:
            return marked_candidate
    return candidate


def _tool_name_allowed(
    name: str,
    allowed_tool_names: set[str] | frozenset[str] | None,
) -> bool:
    return bool(allowed_tool_names) and name in allowed_tool_names


def _make_chat_completion_tool_call(parsed: dict[str, Any]) -> Any:
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )

    return ChatCompletionMessageFunctionToolCall(
        id=str(parsed["id"]),
        type="function",
        function=Function(
            name=str(parsed["name"]),
            arguments=json.dumps(parsed["arguments"], ensure_ascii=False),
        ),
    )


def _content_is_only_tool_call_text(value: Any) -> bool:
    texts = _text_values(value)
    if not texts:
        return False
    joined = "\n".join(texts).strip()
    if not joined:
        return False
    if "<|tool_call_begin|>" in joined:
        return True
    if joined.startswith(("{", "[")) and any(
        marker in joined
        for marker in (
            "tool_calls",
            "toolCalls",
            "function_calls",
            "functionCalls",
            "tool_name",
            "toolName",
        )
    ):
        return True
    return False


def tool_names_from_sdk_tools(tools: list[Any] | tuple[Any, ...]) -> frozenset[str]:
    names: set[str] = set()
    for tool in tools:
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name:
            names.add(name)
            continue
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.add(function["name"])
            elif isinstance(tool.get("name"), str):
                names.add(tool["name"])
    return frozenset(names)


def _tools_from_fetch_response_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[Any] | tuple[Any, ...]:
    tools = kwargs.get("tools")
    if isinstance(tools, list | tuple):
        return tools
    if len(args) >= 4 and isinstance(args[3], list | tuple):
        return args[3]
    return ()


def build_reasoning_preserving_chat_completions_model(
    sdk: SimpleNamespace,
    *,
    model: str,
    openai_client: Any,
    should_replay_reasoning_content: Callable[[Any], bool] | None = None,
) -> Any:
    """Build an Agents chat-completions model that preserves OpenRouter reasoning."""

    class ReasoningPreservingChatCompletionsModel(sdk.OpenAIChatCompletionsModel):
        async def _fetch_response(self, *args: Any, **kwargs: Any) -> Any:
            result = await super()._fetch_response(*args, **kwargs)
            if not isinstance(result, tuple):
                normalize_chat_completion_reasoning_for_agents(
                    result,
                    model=self.model,
                    allowed_tool_names=tool_names_from_sdk_tools(
                        _tools_from_fetch_response_args(args, kwargs)
                    ),
                )
            return result

    return ReasoningPreservingChatCompletionsModel(
        model=model,
        openai_client=openai_client,
        should_replay_reasoning_content=should_replay_reasoning_content,
    )


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


_AUTO_TOOL_CHOICE_MODEL_MARKERS: tuple[str, ...] = (
    "qwen",
    "deepseek-r",
    "reasoning",
    "minimax",
    "kimi",
)


def tool_choice_for_model(model: str) -> str:
    """Return the correct `tool_choice` value for the target model.

    Some OpenAI-compatible gateways reject or mishandle
    `tool_choice="required"` and tool-object forms. Relax to `"auto"` for
    those; keep `"required"` for models where the endpoint supports
    SDK-enforced tool-use each turn.
    """

    lowered = model.lower()
    for marker in _AUTO_TOOL_CHOICE_MODEL_MARKERS:
        if marker in lowered:
            return "auto"
    return "required"


def build_reasoning_replay_hook() -> Callable[[Any], bool]:
    """Return a hook that tells openai-agents when to replay `reasoning_content`.

    Qwen and Kimi-style thinking templates expect reasoning for in-flight
    assistant turns (between the last user message and the current one) to be
    carried forward via `reasoning_content`. The openai-agents default only
    replays for DeepSeek, so these models lose that continuity without this
    hook.
    """

    from agents.models.reasoning_content_replay import (
        default_should_replay_reasoning_content,
    )

    def _should_replay(context: Any) -> bool:
        target_model = (getattr(context, "model", "") or "").lower()
        if "qwen" in target_model or "kimi" in target_model:
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
    text = repr(item)
    marker = "tool-call("
    start = text.find(marker)
    if start >= 0:
        name_start = start + len(marker)
        name_end = text.find(")", name_start)
        if name_end > name_start:
            return text[name_start:name_end]
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
        if len(value) <= max_string_length:
            return value
        return (
            f"{value[:max_string_length]}... "
            f"[truncated; total_chars={len(value)}]"
        )
    if isinstance(value, list):
        preview = [
            preview_payload(
                item,
                max_string_length=max_string_length,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            preview.append(
                {
                    "__preview_truncated__": {
                        "kind": "list",
                        "shown_items": max_list_items,
                        "total_items": len(value),
                        "last_item_preview": preview_payload(
                            value[-1],
                            max_string_length=max_string_length,
                            max_list_items=max_list_items,
                            max_dict_items=max_dict_items,
                        ),
                    }
                }
            )
        return preview
    if isinstance(value, dict):
        preview: dict[str, object] = {}
        items = list(value.items())
        for key, item in items[:max_dict_items]:
            preview[str(key)] = preview_payload(
                item,
                max_string_length=max_string_length,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
        if len(items) > max_dict_items:
            preview["__preview_truncated__"] = {
                "kind": "dict",
                "shown_keys": max_dict_items,
                "total_keys": len(items),
                "omitted_keys": [str(key) for key, _ in items[max_dict_items:]],
            }
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


_RUN_ITEM_PREVIEW_LIMIT = 800


def _preview_text(value: Any, limit: int = _RUN_ITEM_PREVIEW_LIMIT) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            value = str(value)
    if len(value) > limit:
        return value[:limit] + f"… [truncated; total {len(value)} chars]"
    return value


def summarize_run_item(item: Any) -> dict[str, Any]:
    """Extract the meaningful fields from an agents-SDK RunItem into a
    compact dict. Using this instead of ``repr(item)`` avoids the ~25KB
    per-item blow-up where RunItem.repr() embeds the full Agent object
    (including every FunctionTool's params_json_schema) on every item.
    """

    if isinstance(item, str):
        return {"type": "str", "text_preview": _preview_text(item)}

    item_type = type(item).__name__
    summary: dict[str, Any] = {"type": item_type}
    raw = getattr(item, "raw_item", None)

    if item_type == "ToolCallItem":
        tool_name = _dual_attr(raw, "name")
        tool_args = _dual_attr(raw, "arguments")
        call_id = _dual_attr(raw, "call_id")
        summary.update(
            {
                "tool_name": tool_name,
                "arguments_preview": _preview_text(tool_args),
                "call_id": call_id,
            }
        )
        return summary

    if item_type == "ToolCallOutputItem":
        output = getattr(item, "output", None)
        call_id = _dual_attr(raw, "call_id")
        summary.update(
            {
                "call_id": call_id,
                "output_preview": _preview_text(output),
            }
        )
        return summary

    if item_type == "MessageOutputItem":
        summary["text_preview"] = _preview_text(_collect_text_parts(raw))
        return summary

    if item_type == "ReasoningItem":
        reasoning_text = _collect_reasoning_text(raw)
        summary["reasoning_preview"] = _preview_text(reasoning_text)
        return summary

    if item_type in {"HandoffCallItem", "HandoffOutputItem"}:
        summary["handoff_preview"] = _preview_text(str(raw))
        return summary

    summary["raw_preview"] = _preview_text(str(raw))
    return summary


def extract_raw_reasoning_records(items: list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
    """Return raw reasoning payloads exposed by the Agents SDK.

    OpenAI Responses reasoning and OpenAI-compatible chat-completion
    `reasoning_content` are surfaced by the SDK as `ReasoningItem` objects when
    the provider returns them. Callers persist these records as dedicated
    analysis events in the unified timeline.
    """

    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        raw = getattr(item, "raw_item", None)
        if not _is_reasoning_item(item, raw):
            continue
        raw_payload = _jsonable_sdk_payload(raw)
        if raw_payload is None:
            raw_payload = _jsonable_sdk_payload(item)
        records.append(
            {
                "run_item_index": index,
                "run_item_type": type(item).__name__,
                "raw_item": raw_payload,
            }
        )
    return records


def _is_reasoning_item(item: Any, raw: Any) -> bool:
    if type(item).__name__ == "ReasoningItem":
        return True
    raw_type = _dual_attr(raw, "type")
    if raw_type == "reasoning":
        return True
    for attr in (
        "reasoning_content",
        "reasoning",
        "reasoning_details",
        "thinking_blocks",
        "thinking",
    ):
        if getattr(raw, attr, None):
            return True
        if isinstance(raw, dict) and raw.get(attr):
            return True
    return False


def _jsonable_sdk_payload(value: Any) -> Any:
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable_sdk_payload(model_dump(mode="json"))
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _jsonable_sdk_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_sdk_payload(item) for item in value]
    payload = getattr(value, "__dict__", None)
    if isinstance(payload, dict):
        return {
            str(key): _jsonable_sdk_payload(item)
            for key, item in payload.items()
            if not key.startswith("_")
        }
    return str(value)


def _dual_attr(container: Any, key: str) -> Any:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


def _collect_text_parts(raw: Any) -> str:
    content = _dual_attr(raw, "content") or []
    parts: list[str] = []
    for part in content:
        text = _dual_attr(part, "text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _collect_reasoning_text(raw: Any) -> str:
    for key in ("summary", "content"):
        entries = _dual_attr(raw, key) or []
        parts: list[str] = []
        for entry in entries:
            text = _dual_attr(entry, "text")
            if isinstance(text, str) and text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return ""


def extract_run_error_items(exc: BaseException) -> list[dict[str, Any]]:
    """Return compact RunItem summaries from an AgentsException's
    ``run_data.new_items`` if present, else ``[]``. Agents SDK attaches
    ``run_data`` to every ``AgentsException`` (MaxTurnsExceeded,
    ModelBehaviorError, …) so the partial turns can be recovered on the
    error path — where ``run_result`` is unreachable.
    """

    run_data = getattr(exc, "run_data", None)
    if run_data is None:
        return []
    new_items = getattr(run_data, "new_items", None) or []
    return [summarize_run_item(item) for item in new_items]
