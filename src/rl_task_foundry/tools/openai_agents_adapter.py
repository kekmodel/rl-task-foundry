"""Lazy adapters from local tool specs to OpenAI Agents SDK tools."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from rl_task_foundry.tools.models import ToolSpec


ToolExecutor = Callable[[dict[str, Any]], Any]


def make_sdk_tool(spec: ToolSpec, executor: ToolExecutor) -> object:
    """Create an OpenAI Agents SDK tool lazily.

    Import is deferred so the rest of the package can be imported without the
    SDK present in the active environment.
    """

    from agents import function_tool

    @function_tool(name_override=spec.name, description_override=spec.description)
    async def _tool(**kwargs: Any) -> Any:
        result = executor(kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    return _tool


def make_submit_result_tool(
    *,
    answer_model: type[BaseModel],
    on_submit: ToolExecutor | None = None,
) -> object:
    """Create an explicit terminal tool for answer submission.

    The tool validates its JSON arguments against the generated answer model and
    returns a marker payload that the caller can turn into the final result.
    Validation failures are surfaced as tool output so the model can retry.
    """

    from agents import FunctionTool
    from agents.strict_schema import ensure_strict_json_schema

    params_json_schema = ensure_strict_json_schema(answer_model.model_json_schema())

    async def _invoke_tool(_tool_context: Any, input_json: str) -> Any:
        try:
            payload_model = answer_model.model_validate_json(input_json)
        except ValidationError as exc:
            return {
                "submitted": False,
                "error": "submit_result payload failed schema validation",
                "details": json.loads(exc.json()),
            }

        payload = payload_model.model_dump(mode="python")
        if on_submit is not None:
            result = on_submit(payload)
            if inspect.isawaitable(result):
                return await result
            return result

        return {"submitted": True, "answer": payload}

    return FunctionTool(
        name="submit_result",
        description=(
            "Submit the final structured answer once you have enough evidence. "
            "Call this exactly once when you are ready to finish."
        ),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=True,
    )
