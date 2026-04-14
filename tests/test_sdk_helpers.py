import json
from datetime import date, datetime
from decimal import Decimal

import pytest

from rl_task_foundry.infra.sdk_helpers import make_sdk_tool


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
