"""Helpers for binding materialized atomic tools to live database executors."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from rl_task_foundry.infra.db import DatabasePools

ToolExecutor = Callable[[dict[str, Any]], Any]


def load_atomic_tool_module(source_path: Path, *, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load atomic tool module: {source_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_atomic_tool_executor(
    *,
    module: ModuleType,
    tool_name: str,
    pools: DatabasePools,
) -> ToolExecutor:
    function = getattr(module, tool_name)

    async def _execute(kwargs: dict[str, Any]) -> Any:
        async with pools.solver_connection() as conn:
            result = function(conn, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

    return _execute
