import json

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

    assert result == (
        "ToolError: ValueError: bad params. "
        "Fix the tool arguments and continue."
    )
