"""Persistent subprocess worker for registration-lane preflight tasks."""

from __future__ import annotations

import asyncio
import builtins
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import CodeType
from typing import Any
from types import SimpleNamespace

from pydantic import ValidationError

from rl_task_foundry.config.models import RegistrationPolicyConfig
from rl_task_foundry.synthesis.registration_policy import (
    ArtifactKind,
    RegistrationError,
    validate_generated_module,
)

try:
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None


_SAFE_BUILTIN_NAMES = (
    "__import__",
    "abs",
    "all",
    "any",
    "AssertionError",
    "bool",
    "dict",
    "enumerate",
    "Exception",
    "filter",
    "float",
    "int",
    "isinstance",
    "KeyError",
    "len",
    "list",
    "map",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "RuntimeError",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "TypeError",
    "ValueError",
    "zip",
)

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in _SAFE_BUILTIN_NAMES
}


def _set_memory_limit(memory_limit_mb: int | None) -> None:
    if resource is None or not memory_limit_mb:
        return
    limit_bytes = memory_limit_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (OSError, ValueError):  # pragma: no cover - OS dependent
        return


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _error_payload(
    *,
    request_id: str | None,
    code: str,
    detail: str,
) -> dict[str, Any]:
    error = RegistrationError(code=code, detail=detail)
    return {
        "request_id": request_id,
        "worker_pid": os.getpid(),
        "errors": [error.model_dump(mode="json")],
    }


def _execution_error_payload(
    *,
    request_id: str | None,
    code: str,
    detail: str,
    call_count: int | None = None,
) -> dict[str, Any]:
    error = RegistrationError(code=code, detail=detail)
    return {
        "request_id": request_id,
        "worker_pid": os.getpid(),
        "call_count": call_count,
        "return_value": None,
        "errors": [error.model_dump(mode="json")],
    }


def _handle_validate(payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("request_id")
    try:
        policy = RegistrationPolicyConfig.model_validate(payload["policy"])
        kind = ArtifactKind(payload["artifact_kind"])
        source = payload["source"]
        memory_limit_mb = payload.get("memory_limit_mb")
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        return _error_payload(
            request_id=request_id,
            code="invalid_worker_request",
            detail=f"Worker request validation failed: {exc}",
        )

    _set_memory_limit(memory_limit_mb)
    errors = validate_generated_module(
        source,
        kind=kind,
        policy=policy,
    )
    return {
        "request_id": request_id,
        "worker_pid": os.getpid(),
        "errors": [error.model_dump(mode="json") for error in errors],
    }


@dataclass(slots=True)
class _CallCounter:
    limit: int
    count: int = 0

    def profiler(self, _frame: Any, event: str, _arg: Any) -> Callable[..., Any] | None:
        # This profiler is process-global, so worker requests must remain serialized.
        if event == "call":
            self.count += 1
            if self.count > self.limit:
                raise RuntimeError(f"call_count_limit_exceeded:{self.limit}")
        return self.profiler


def _build_module_globals() -> dict[str, Any]:
    return {
        "__name__": "__generated_artifact__",
        "__builtins__": _SAFE_BUILTINS,
    }


def _load_entrypoint(
    *,
    source: str,
    artifact_kind: ArtifactKind,
    entrypoint: str,
) -> Callable[..., Any]:
    compiled: CodeType = compile(source, f"<{artifact_kind.value}>", "exec")
    namespace = _build_module_globals()
    exec(compiled, namespace, namespace)
    function = namespace.get(entrypoint)
    if not callable(function):
        raise RuntimeError(f"missing_entrypoint:{entrypoint}")
    return function


def _load_namespace(
    *,
    source: str,
    artifact_kind: ArtifactKind,
) -> dict[str, Any]:
    compiled: CodeType = compile(source, f"<{artifact_kind.value}>", "exec")
    namespace = _build_module_globals()
    exec(compiled, namespace, namespace)
    return namespace


def _ensure_json_serializable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
    except TypeError as exc:
        raise RuntimeError("non_json_serializable_result") from exc
    return value


async def _invoke_entrypoint(
    *,
    function: Callable[..., Any],
    args: list[Any],
    kwargs: dict[str, Any],
    call_count_limit: int,
) -> tuple[Any, int]:
    counter = _CallCounter(limit=call_count_limit)
    previous_profiler = sys.getprofile()
    sys.setprofile(counter.profiler)
    try:
        result = function(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return _ensure_json_serializable(result), counter.count
    finally:
        sys.setprofile(previous_profiler)


def _build_tool_facade(tool_source: str) -> SimpleNamespace:
    namespace = _load_namespace(
        source=tool_source,
        artifact_kind=ArtifactKind.TOOL_MODULE,
    )
    public_functions = {
        name: value
        for name, value in namespace.items()
        if callable(value) and not name.startswith("_")
    }

    facade = SimpleNamespace()
    for name, function in public_functions.items():
        setattr(facade, name, _bind_tool_function(function))
    return facade


def _bind_tool_function(function: Callable[..., Any]) -> Callable[..., Any]:
    async def _bound(*args: Any, **kwargs: Any) -> Any:
        # Milestone 2 self-tests validate tool logic against a lightweight facade only.
        # DB-backed tool execution is wired in later milestones.
        result = function(None, *args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return _bound


def _handle_execute(payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("request_id")
    try:
        policy = RegistrationPolicyConfig.model_validate(payload["policy"])
        kind = ArtifactKind(payload["artifact_kind"])
        source = payload["source"]
        entrypoint = payload["entrypoint"]
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        call_count_limit = int(payload["call_count_limit"])
        memory_limit_mb = payload.get("memory_limit_mb")
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        return _execution_error_payload(
            request_id=request_id,
            code="invalid_worker_request",
            detail=f"Worker request validation failed: {exc}",
        )

    _set_memory_limit(memory_limit_mb)
    errors = validate_generated_module(
        source,
        kind=kind,
        policy=policy,
    )
    if errors:
        return {
            "request_id": request_id,
            "worker_pid": os.getpid(),
            "call_count": None,
            "return_value": None,
            "errors": [error.model_dump(mode="json") for error in errors],
        }

    try:
        function = _load_entrypoint(
            source=source,
            artifact_kind=kind,
            entrypoint=entrypoint,
        )
        return_value, call_count = asyncio.run(
            _invoke_entrypoint(
                function=function,
                args=list(args),
                kwargs=dict(kwargs),
                call_count_limit=call_count_limit,
            )
        )
    except RuntimeError as exc:
        detail = str(exc)
        code = "execution_error"
        if detail.startswith("call_count_limit_exceeded:"):
            code = "call_count_limit_exceeded"
        elif detail.startswith("missing_entrypoint:"):
            code = "missing_entrypoint"
        elif detail == "non_json_serializable_result":
            code = "non_json_serializable_result"
        return _execution_error_payload(
            request_id=request_id,
            code=code,
            detail=detail,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return _execution_error_payload(
            request_id=request_id,
            code="execution_error",
            detail=f"{type(exc).__name__}: {exc}",
        )

    return {
        "request_id": request_id,
        "worker_pid": os.getpid(),
        "call_count": call_count,
        "return_value": return_value,
        "errors": [],
    }


def _handle_tool_self_test(payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("request_id")
    try:
        policy = RegistrationPolicyConfig.model_validate(payload["policy"])
        tool_source = payload["tool_source"]
        self_test_source = payload["self_test_source"]
        call_count_limit = int(payload["call_count_limit"])
        memory_limit_mb = payload.get("memory_limit_mb")
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        return _execution_error_payload(
            request_id=request_id,
            code="invalid_worker_request",
            detail=f"Worker request validation failed: {exc}",
        )

    _set_memory_limit(memory_limit_mb)
    errors = [
        *validate_generated_module(
            tool_source,
            kind=ArtifactKind.TOOL_MODULE,
            policy=policy,
        ),
        *validate_generated_module(
            self_test_source,
            kind=ArtifactKind.TOOL_SELF_TEST_MODULE,
            policy=policy,
        ),
    ]
    if errors:
        return {
            "request_id": request_id,
            "worker_pid": os.getpid(),
            "call_count": None,
            "return_value": None,
            "errors": [error.model_dump(mode="json") for error in errors],
        }

    try:
        tools = _build_tool_facade(tool_source)
        function = _load_entrypoint(
            source=self_test_source,
            artifact_kind=ArtifactKind.TOOL_SELF_TEST_MODULE,
            entrypoint="run_self_test",
        )
        return_value, call_count = asyncio.run(
            _invoke_entrypoint(
                function=function,
                args=[tools],
                kwargs={},
                call_count_limit=call_count_limit,
            )
        )
    except RuntimeError as exc:
        detail = str(exc)
        code = "execution_error"
        if detail.startswith("call_count_limit_exceeded:"):
            code = "call_count_limit_exceeded"
        elif detail.startswith("missing_entrypoint:"):
            code = "missing_entrypoint"
        elif detail == "non_json_serializable_result":
            code = "non_json_serializable_result"
        return _execution_error_payload(
            request_id=request_id,
            code=code,
            detail=detail,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return _execution_error_payload(
            request_id=request_id,
            code="execution_error",
            detail=f"{type(exc).__name__}: {exc}",
        )

    return {
        "request_id": request_id,
        "worker_pid": os.getpid(),
        "call_count": call_count,
        "return_value": return_value,
        "errors": [],
    }


def _handle_request(line: str) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        return _error_payload(
            request_id=None,
            code="invalid_worker_json",
            detail=f"Worker request is not valid JSON: {exc}",
        )

    request_type = payload.get("type")
    request_id = payload.get("request_id")
    if request_type == "shutdown":
        return {
            "request_id": request_id,
            "worker_pid": os.getpid(),
            "shutdown": True,
            "errors": [],
        }
    if request_type == "run_tool_self_test":
        return _handle_tool_self_test(payload)
    if request_type == "execute_module_entrypoint":
        return _handle_execute(payload)
    if request_type != "validate_module":
        return _error_payload(
            request_id=request_id,
            code="unsupported_worker_request",
            detail=f"Unsupported worker request type: {request_type!r}",
        )
    return _handle_validate(payload)


def main() -> int:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        response = _handle_request(line)
        sys.stdout.write(_json_response(response))
        sys.stdout.write("\n")
        sys.stdout.flush()
        if response.get("shutdown"):
            return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
