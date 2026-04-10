"""Thin abstraction over OpenHarness. Fallback to raw API if OH is unavailable."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Awaitable

log = logging.getLogger(__name__)


@dataclass
class ToolDef:
    """Tool definition — portable across OpenHarness and raw API."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_api_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """Result from a tool execution."""

    output: str
    is_error: bool = False


@dataclass
class AgentTurn:
    """One turn of agent output."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False


ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]


class AgentRunner:
    """Runs an LLM agent loop with tool use.

    Tries OpenHarness QueryEngine first. Falls back to raw OpenAI-compatible
    API with manual tool loop if OH is not available.
    """

    def __init__(
        self,
        *,
        provider_type: str,
        base_url: str | None,
        api_key: str,
        model: str,
        system_prompt: str,
        tools: list[ToolDef],
        tool_executor: ToolExecutor,
        max_turns: int = 10,
    ) -> None:
        self._provider_type = provider_type
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._system_prompt = system_prompt
        self._tools = tools
        self._tool_executor = tool_executor
        self._max_turns = max_turns
        self._backend: str = "unknown"

    async def run(self, user_message: str) -> AgentTurn:
        """Run agent loop. Returns final turn with accumulated text."""
        try:
            return await self._run_openharness(user_message)
        except ImportError:
            log.info("OpenHarness not available, using raw API fallback")
            return await self._run_raw_api(user_message)

    async def _run_openharness(self, user_message: str) -> AgentTurn:
        """Run via OpenHarness QueryEngine."""
        from openharness.engine import QueryEngine
        from openharness.tools.base import BaseTool, ToolRegistry, ToolExecutionContext
        from openharness.api.client import AnthropicApiClient

        # This path uses OpenHarness internals — implementation details
        # will be filled when integrating with actual OH in Plan 2+
        raise ImportError("OpenHarness integration deferred to Plan 2")

    async def _run_raw_api(self, user_message: str) -> AgentTurn:
        """Fallback: raw OpenAI-compatible API with manual tool loop."""
        import httpx

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]
        tool_schemas = [
            {"type": "function", "function": t.to_api_schema()} for t in self._tools
        ]

        accumulated_text = ""
        all_tool_calls: list[dict[str, Any]] = []

        base = self._base_url or "https://api.openai.com/v1"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        async with httpx.AsyncClient(base_url=base, headers=headers, timeout=120) as client:
            for turn in range(self._max_turns):
                payload: dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                }
                if tool_schemas:
                    payload["tools"] = tool_schemas

                resp = await client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()

                choice = data["choices"][0]
                msg = choice["message"]

                if msg.get("content"):
                    accumulated_text += msg["content"]

                if not msg.get("tool_calls"):
                    return AgentTurn(
                        text=accumulated_text,
                        tool_calls=all_tool_calls,
                        finished=True,
                    )

                # Execute tool calls
                messages.append(msg)
                for tc in msg["tool_calls"]:
                    import json
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    all_tool_calls.append({"name": fn_name, "arguments": fn_args})

                    result = await self._tool_executor(fn_name, fn_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result.output,
                    })

        return AgentTurn(text=accumulated_text, tool_calls=all_tool_calls, finished=True)
