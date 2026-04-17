"""Synthesis-side wiring for the composer toolset.

Bridges `tooling.composer.build_composer_tools` into the synthesis
conversation:

- `instrument_composer_tool` wraps a FunctionTool's `on_invoke_tool` so
  each call is mirrored onto `SubmitDraftController.record_atomic_tool_call`.
  The controller's grounded-observation guard and trace logging already
  depend on this telemetry; the composer tools must feed it the same way
  the old atomic tools did.

- `build_instrumented_composer_tools` composes the two pieces.

- `summarize_composer_tool_surface` produces a small JSON summary that
  the synthesis prompt can consume in place of the atomic-bundle
  surface summary. Prompt rewrites (checklist step 6) will dress this
  up; for now we just ship name+description.

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
        output = await inner(ctx, input_json)  # pyright: ignore[reportArgumentType]
        params = _safe_parse_json_object(input_json)
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


_SOLVER_ATOMIC_PRIMITIVES: dict[str, tuple[str, ...]] = {
    "set_producing": ("rows_where", "rows_via", "intersect"),
    "set_annotating": ("order_by",),
    "set_materializing": ("take", "count", "aggregate", "group_top"),
    "row_reading": ("read",),
}


def summarize_composer_tool_surface(
    tools: list["FunctionTool"],
) -> dict[str, object]:
    """JSON summary of the composer toolset plus the solver primitive
    inventory, both rendered by `synthesis/prompts.build_synthesis_input`.

    The composer entries (name + description) come straight from the
    FunctionTool instances built for this conversation. The solver
    primitive groups are fixed — the prompt surfaces them so the
    composer can anticipate how a solver will re-derive the canonical
    answer via an atomic calculus chain.
    """
    entries: list[dict[str, object]] = []
    for tool in tools:
        entries.append(
            {"name": tool.name, "description": tool.description}
        )
    return {
        "tool_count": len(tools),
        "tools": entries,
        "solver_primitives": {
            group: list(names)
            for group, names in _SOLVER_ATOMIC_PRIMITIVES.items()
        },
    }


__all__ = [
    "build_instrumented_composer_tools",
    "instrument_composer_tool",
    "summarize_composer_tool_surface",
]
