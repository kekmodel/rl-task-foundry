"""Shared runtime glue for agents-SDK tool handlers.

Both the composer and atomic tool factories wrap their handlers with
identical JSON-parse + exception-to-error-response logic. This module
holds the single canonical implementation so the two factories stay in
lockstep.

Exceptions caught intentionally include `NotImplementedError`:
- composer tools raise `NotImplementedError` for inputs the current DSL
  can't encode (multi-column PK, etc.) and expect that to surface as a
  structured tool error instead of a crash;
- atomic tools previously didn't list `NotImplementedError`, which
  meant any gap would bypass the safety net. Unifying catches the
  superset and closes that silent drift.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Awaitable, Callable

import asyncpg

from rl_task_foundry.tooling.common.payload import JsonObject

Handler = Callable[[JsonObject], Awaitable[JsonObject]]
Invoker = Callable[[object, str], Awaitable[str]]


def json_dumps_tool(payload: object) -> str:
    """Serialize a tool result / error envelope for the agents SDK."""
    return json.dumps(payload, default=_json_default, ensure_ascii=False)


def _json_default(value: object) -> str:
    if isinstance(value, _dt.datetime | _dt.date | _dt.time):
        return value.isoformat()
    return str(value)


_CAUGHT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    KeyError,
    ValueError,
    TypeError,
    LookupError,
    RuntimeError,
    NotImplementedError,
    asyncpg.exceptions.PostgresError,
)


def wrap_tool_handler(handler: Handler) -> Invoker:
    """Parse the agents-SDK input JSON, call handler, convert to envelope."""

    async def invoke(_tool_context: object, input_json: str) -> str:
        try:
            parsed_raw: object = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError as exc:
            return json_dumps_tool(
                {
                    "error": f"invalid JSON input: {exc}",
                    "error_type": "JSONDecodeError",
                }
            )
        if not isinstance(parsed_raw, dict):
            return json_dumps_tool(
                {
                    "error": "tool input must be a JSON object",
                    "error_type": "TypeError",
                }
            )
        parsed: JsonObject = {
            str(key): value for key, value in parsed_raw.items()
        }
        try:
            result = await handler(parsed)
        except _CAUGHT_EXCEPTIONS as exc:
            return json_dumps_tool(
                {"error": str(exc), "error_type": type(exc).__name__}
            )
        return json_dumps_tool(result)

    return invoke
