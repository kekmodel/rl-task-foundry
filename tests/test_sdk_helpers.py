import json
from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from rl_task_foundry.infra.sdk_helpers import (
    extract_raw_reasoning_records,
    make_sdk_tool,
    normalize_chat_completion_reasoning_for_agents,
    preview_payload,
)


@pytest.mark.asyncio
async def test_make_sdk_tool_returns_tool_error_string_instead_of_raising() -> None:
    def _boom(_payload: dict[str, object]) -> object:
        raise ValueError("bad params")

    tool = make_sdk_tool(
        {
            "name": "broken_tool",
            "description": "Broken tool.",
            "params_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        _boom,
    )

    result = await tool.on_invoke_tool(None, json.dumps({}))

    assert result == ("ToolError: ValueError: bad params. Fix the tool arguments and continue.")


@pytest.mark.asyncio
async def test_make_sdk_tool_normalizes_datetime_and_decimal_results() -> None:
    def _result(_payload: dict[str, object]) -> object:
        return {
            "created_at": datetime(2025, 8, 22, 20, 3, 46),
            "created_on": date(2025, 8, 22),
            "amount": Decimal("5.99"),
            "items": [datetime(2025, 8, 23, 1, 2, 3), Decimal("1.50")],
        }

    tool = make_sdk_tool(
        {
            "name": "normalized_tool",
            "description": "Normalized tool.",
            "params_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        _result,
    )

    result = await tool.on_invoke_tool(None, json.dumps({}))

    assert result == {
        "created_at": "2025-08-22T20:03:46",
        "created_on": "2025-08-22",
        "amount": "5.99",
        "items": ["2025-08-23T01:02:03", "1.50"],
    }


@pytest.mark.asyncio
async def test_make_sdk_tool_strips_trailing_whitespace_from_char_columns() -> None:
    """PostgreSQL char(N) columns return fixed-width strings with trailing spaces."""

    def _result(_payload: dict[str, object]) -> object:
        return {
            "name": "English             ",
            "code": "EN  ",
            "normal": "Hello",
        }

    tool = make_sdk_tool(
        {
            "name": "char_tool",
            "description": "Tool with char(N) columns.",
            "params_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        _result,
    )

    result = await tool.on_invoke_tool(None, json.dumps({}))

    assert result == {
        "name": "English",
        "code": "EN",
        "normal": "Hello",
    }


def test_preview_payload_marks_truncated_strings_lists_and_dicts() -> None:
    payload = {
        "join": [
            {"as": "a"},
            {"as": "b"},
            {"as": "c"},
            {"as": "d"},
        ],
        "long_text": "abcdefghijklmnopqrstuvwxy",
        "where": [
            {"column": "a"},
            {"column": "b"},
            {"column": "c"},
            {"column": "d"},
        ],
        "extra": "hidden",
    }

    preview = preview_payload(
        payload,
        max_string_length=20,
        max_list_items=3,
        max_dict_items=3,
    )

    assert preview == {
        "join": [
            {"as": "a"},
            {"as": "b"},
            {"as": "c"},
            {
                "__preview_truncated__": {
                    "kind": "list",
                    "shown_items": 3,
                    "total_items": 4,
                    "last_item_preview": {"as": "d"},
                }
            },
        ],
        "long_text": "abcdefghijklmnopqrst... [truncated; total_chars=25]",
        "where": [
            {"column": "a"},
            {"column": "b"},
            {"column": "c"},
            {
                "__preview_truncated__": {
                    "kind": "list",
                    "shown_items": 3,
                    "total_items": 4,
                    "last_item_preview": {"column": "d"},
                }
            },
        ],
        "__preview_truncated__": {
            "kind": "dict",
            "shown_keys": 3,
            "total_keys": 4,
            "omitted_keys": ["extra"],
        },
    }


def test_extract_raw_reasoning_records_serializes_sdk_reasoning_items() -> None:
    raw_reasoning = SimpleNamespace(
        type="reasoning",
        summary=[
            SimpleNamespace(
                type="summary_text",
                text="provider-visible reasoning",
            )
        ],
        encrypted_content="opaque",
    )

    records = extract_raw_reasoning_records(
        [
            SimpleNamespace(raw_item=SimpleNamespace(type="message", content="done")),
            SimpleNamespace(raw_item=raw_reasoning),
        ]
    )

    assert records == [
        {
            "run_item_index": 1,
            "run_item_type": "SimpleNamespace",
            "raw_item": {
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "provider-visible reasoning",
                    }
                ],
                "encrypted_content": "opaque",
            },
        }
    ]


def test_extract_raw_reasoning_records_preserves_model_dump_strings() -> None:
    class _FakeReasoningModel:
        type = "reasoning"

        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "keep trailing spaces  ",
                    }
                ],
            }

    records = extract_raw_reasoning_records(
        [SimpleNamespace(raw_item=_FakeReasoningModel())]
    )

    assert records[0]["raw_item"] == {
        "type": "reasoning",
        "summary": [
            {
                "type": "summary_text",
                "text": "keep trailing spaces  ",
            }
        ],
    }


def test_normalize_chat_completion_reasoning_for_agents_copies_openrouter_field() -> None:
    message = SimpleNamespace(reasoning="openrouter reasoning", content="done")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    normalize_chat_completion_reasoning_for_agents(response)

    assert message.reasoning_content == "openrouter reasoning"


def test_normalize_kimi_reasoning_tool_call_promotes_standard_tool_call() -> None:
    reasoning = (
        "<|tool_call_begin|><|tool_call_end|>"
        'functions.lookup:0{"id": 7}<|tool_call_argument_begin|>'
    )
    message = SimpleNamespace(reasoning=reasoning, content=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    response = SimpleNamespace(choices=[choice])

    normalize_chat_completion_reasoning_for_agents(
        response,
        model="moonshotai/kimi-k2.5",
        allowed_tool_names=frozenset({"lookup"}),
    )

    assert message.reasoning_content == reasoning
    assert choice.finish_reason == "tool_calls"
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call.id == "functions.lookup:0"
    assert tool_call.function.name == "lookup"
    assert json.loads(tool_call.function.arguments) == {"id": 7}


def test_normalize_kimi_official_template_tool_call_promotes_standard_tool_call() -> None:
    content = (
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        'functions.query:1<|tool_call_argument_begin|>{"sql": "select 1"}'
        "<|tool_call_end|><|tool_calls_section_end|>"
    )
    message = SimpleNamespace(content=content)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    normalize_chat_completion_reasoning_for_agents(
        response,
        model="kimi-k2.5",
        allowed_tool_names=frozenset({"query"}),
    )

    assert message.content is None
    assert message.tool_calls[0].id == "functions.query:1"
    assert message.tool_calls[0].function.name == "query"
    assert json.loads(message.tool_calls[0].function.arguments) == {"sql": "select 1"}


def test_normalize_kimi_tool_call_requires_allowed_tool_name() -> None:
    message = SimpleNamespace(
        reasoning='functions.not_registered:0{"id": 7}',
        content=None,
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    normalize_chat_completion_reasoning_for_agents(
        response,
        model="moonshotai/kimi-k2.5",
        allowed_tool_names=frozenset({"lookup"}),
    )

    assert getattr(message, "tool_calls", None) is None


def test_normalize_kimi_replaces_invalid_standard_tool_call_from_template() -> None:
    reasoning = (
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        'functions.submit_draft:0<|tool_call_argument_begin|>{"topic": "assignment"}'
        "<|tool_call_end|><|tool_calls_section_end|>"
    )
    invalid_tool_call = SimpleNamespace(
        id="bad-call",
        type="function",
        function=SimpleNamespace(
            name="submit_draft",
            arguments='{"topic": "assignment"} trailing',
        ),
    )
    message = SimpleNamespace(reasoning=reasoning, tool_calls=[invalid_tool_call])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    response = SimpleNamespace(choices=[choice])

    normalize_chat_completion_reasoning_for_agents(
        response,
        model="moonshotai/kimi-k2.5",
        allowed_tool_names=frozenset({"submit_draft"}),
    )

    assert choice.finish_reason == "tool_calls"
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].id == "functions.submit_draft:0"
    assert message.tool_calls[0].function.name == "submit_draft"
    assert json.loads(message.tool_calls[0].function.arguments) == {
        "topic": "assignment"
    }


def test_normalize_kimi_keeps_valid_standard_tool_call() -> None:
    reasoning = (
        "<|tool_call_begin|><|tool_call_end|>"
        'functions.lookup:0{"id": 99}<|tool_call_argument_begin|>'
    )
    valid_tool_call = SimpleNamespace(
        id="standard-call",
        type="function",
        function=SimpleNamespace(
            name="lookup",
            arguments='{"id": 7}',
        ),
    )
    message = SimpleNamespace(reasoning=reasoning, tool_calls=[valid_tool_call])
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
    )

    normalize_chat_completion_reasoning_for_agents(
        response,
        model="moonshotai/kimi-k2.5",
        allowed_tool_names=frozenset({"lookup"}),
    )

    assert message.tool_calls == [valid_tool_call]
    assert json.loads(message.tool_calls[0].function.arguments) == {"id": 7}


def test_normalize_kimi_template_escapes_unescaped_control_characters() -> None:
    reasoning = (
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        'functions.submit_draft:0<|tool_call_argument_begin|>{"user_request": "first line\n'
        'second line"}<|tool_call_end|><|tool_calls_section_end|>'
    )
    message = SimpleNamespace(reasoning=reasoning)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    normalize_chat_completion_reasoning_for_agents(
        response,
        model="moonshotai/kimi-k2.5",
        allowed_tool_names=frozenset({"submit_draft"}),
    )

    assert json.loads(message.tool_calls[0].function.arguments) == {
        "user_request": "first line\nsecond line"
    }


def test_normalize_kimi_tool_call_does_not_run_for_other_models() -> None:
    message = SimpleNamespace(
        reasoning='functions.lookup:0{"id": 7}',
        content=None,
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    normalize_chat_completion_reasoning_for_agents(
        response,
        model="deepseek/deepseek-chat-v3.1",
        allowed_tool_names=frozenset({"lookup"}),
    )

    assert getattr(message, "tool_calls", None) is None


def test_extract_raw_reasoning_records_accepts_openrouter_reasoning_field() -> None:
    raw_message = SimpleNamespace(
        type="message",
        reasoning="provider-visible reasoning",
        content="done",
    )

    records = extract_raw_reasoning_records([SimpleNamespace(raw_item=raw_message)])

    assert records == [
        {
            "run_item_index": 0,
            "run_item_type": "SimpleNamespace",
            "raw_item": {
                "type": "message",
                "reasoning": "provider-visible reasoning",
                "content": "done",
            },
        }
    ]
