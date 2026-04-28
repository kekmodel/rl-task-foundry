"""Synthesis-side wiring for the authoring toolset.

Bridges `tooling.composer.build_composer_tools` into the synthesis
conversation:

- `instrument_composer_tool` wraps a FunctionTool's `on_invoke_tool` so
  each call is mirrored onto `SubmitDraftController.record_atomic_tool_call`.
  The controller's grounded-observation guard and trace logging already
  depend on this telemetry; the composer tools must feed it the same way
  the old atomic tools did.

- `build_instrumented_composer_tools` composes the two pieces.

- `summarize_composer_tool_surface` produces a small JSON summary that
  the synthesis prompt can consume without hidden evaluator/runtime
  inventory.

The `agents` package is imported lazily so this module stays importable
without the SDK.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rl_task_foundry.synthesis.submit_draft_tool import SubmitDraftController

if TYPE_CHECKING:
    from agents import FunctionTool


def _safe_parse_json_object(raw: str) -> dict[str, object]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    if not isinstance(parsed, dict):
        return {"_value": parsed}
    return {str(key): value for key, value in parsed.items()}


def _safe_parse_result(raw: str) -> object:
    if not raw:
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def instrument_composer_tool(
    tool: "FunctionTool",
    controller: SubmitDraftController,
) -> "FunctionTool":
    """Return a new FunctionTool whose handler mirrors the call onto
    `controller.record_atomic_tool_call` after the inner handler runs.
    """
    from agents import FunctionTool

    inner = tool.on_invoke_tool
    name = tool.name

    async def wrapped(ctx: object, input_json: str) -> str:
        params = _safe_parse_json_object(input_json)
        budget_feedback = controller.data_tool_budget_feedback(tool_name=name)
        if budget_feedback is not None:
            controller.record_atomic_tool_call(
                tool_name=name,
                params=params,
                result=budget_feedback,
            )
            return json.dumps(budget_feedback, ensure_ascii=False)
        output = await inner(ctx, input_json)  # pyright: ignore[reportArgumentType]
        result = _safe_parse_result(output)
        controller.record_atomic_tool_call(
            tool_name=name,
            params=params,
            result=result,
        )
        return output

    return FunctionTool(
        name=tool.name,
        description=tool.description,
        params_json_schema=tool.params_json_schema,
        on_invoke_tool=wrapped,
        strict_json_schema=tool.strict_json_schema,
    )


def build_instrumented_composer_tools(
    raw_tools: list["FunctionTool"],
    controller: SubmitDraftController,
) -> list["FunctionTool"]:
    return [instrument_composer_tool(tool, controller) for tool in raw_tools]


def summarize_composer_tool_surface(
    tools: list["FunctionTool"],
) -> dict[str, object]:
    """JSON summary of the callable composer toolset.

    Keep evaluator/runtime internals out of the composer prompt. The composer
    needs to know which tools it may call, not which hidden tools will later
    judge or reproduce the answer.
    """
    entries: list[dict[str, object]] = []
    for tool in tools:
        entries.append(
            {"name": tool.name, "description": tool.description}
        )
    return {
        "tool_count": len(tools),
        "tools": entries,
    }


__all__ = [
    "build_instrumented_composer_tools",
    "instrument_composer_tool",
    "summarize_composer_tool_surface",
]
