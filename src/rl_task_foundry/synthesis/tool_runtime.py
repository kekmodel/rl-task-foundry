"""Helpers for binding materialized atomic tools to live database executors."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from rl_task_foundry.infra.db import DatabasePools

ToolExecutor = Callable[[dict[str, Any]], Awaitable[Any]]


def build_shuffle_seed(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


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
    accepts_shuffle_seed = "_shuffle_seed" in inspect.signature(function).parameters

    async def _execute(kwargs: dict[str, Any]) -> Any:
        payload = dict(kwargs)
        if not accepts_shuffle_seed:
            payload.pop("_shuffle_seed", None)
        async with pools.solver_connection() as conn:
            result = function(conn, **payload)
            if inspect.isawaitable(result):
                return await result
            return result

    return _execute


def with_tool_shuffle_seed(
    executor: ToolExecutor,
    *,
    shuffle_seed: str | None,
) -> ToolExecutor:
    async def _execute(kwargs: dict[str, Any]) -> Any:
        payload = dict(kwargs)
        if shuffle_seed is not None:
            payload["_shuffle_seed"] = shuffle_seed
        result = executor(payload)
        if inspect.isawaitable(result):
            return await result
        return result

    return _execute
