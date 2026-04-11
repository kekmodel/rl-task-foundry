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


def _verifier_probe_error_payload(
    *,
    request_id: str | None,
    code: str,
    detail: str,
    expected_fact_keys: list[str] | None = None,
    fetch_facts_return_keys: list[str] | None = None,
    missing_fact_keys: list[str] | None = None,
    extra_fact_keys: list[str] | None = None,
    fetch_facts_tool_calls: int | None = None,
    verify_tool_calls: int | None = None,
    facts_match_result: bool | None = None,
    check_constraints_result: bool | None = None,
    verify_result: bool | None = None,
) -> dict[str, Any]:
    error = RegistrationError(code=code, detail=detail)
    return {
        "request_id": request_id,
        "worker_pid": os.getpid(),
        "fetch_facts_return_keys": fetch_facts_return_keys or [],
        "expected_fact_keys": expected_fact_keys or [],
        "missing_fact_keys": missing_fact_keys or [],
        "extra_fact_keys": extra_fact_keys or [],
        "fetch_facts_tool_calls": fetch_facts_tool_calls,
        "verify_tool_calls": verify_tool_calls,
        "facts_match_result": facts_match_result,
        "check_constraints_result": check_constraints_result,
        "verify_result": verify_result,
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


def _public_tool_names(tool_source: str) -> set[str]:
    namespace = _load_namespace(
        source=tool_source,
        artifact_kind=ArtifactKind.TOOL_MODULE,
    )
    return {
        name
        for name, value in namespace.items()
        if callable(value) and not name.startswith("_")
    }


def _bind_tool_function(function: Callable[..., Any]) -> Callable[..., Any]:
    async def _bound(*args: Any, **kwargs: Any) -> Any:
        # Milestone 2 self-tests validate tool logic against a lightweight facade only.
        # DB-backed tool execution is wired in later milestones.
        result = function(None, *args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return _bound


class _VerifierProbeTools:
    def __init__(self, tool_names: set[str]) -> None:
        self._tool_names = tool_names
        self.call_count = 0

    def __getattr__(self, name: str) -> Callable[..., Any]:
        def _bound(*args: Any, **kwargs: Any) -> Any:
            self.call_count += 1
            return _synthetic_tool_value(name, args=args, kwargs=kwargs)

        return _bound


def _synthetic_tool_value(
    name: str,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    scalar = _first_scalar(args, kwargs)
    row = {
        "tool": name,
        "id": 1,
        "name": str(scalar) if scalar is not None else f"{name}_name",
        "value": scalar if scalar is not None else 1,
        "city": "sample_city",
        "country": "sample_country",
        "status": "active",
        "price": 100.0,
        "rating": 4.0,
        "count": 1,
        "allowed": True,
        "date": "2026-01-01",
        "datetime": "2026-01-01T00:00:00Z",
    }
    if _looks_plural(name):
        return [row]
    return row


def _looks_plural(name: str) -> bool:
    return name.endswith("s") or name.startswith(("list_", "get_all_", "lookup_all_"))


def _first_scalar(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, (str, int, float, bool)):
            return value
    return None


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


def _handle_probe_verifier(payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("request_id")
    try:
        policy = RegistrationPolicyConfig.model_validate(payload["policy"])
        tool_source = payload["tool_source"]
        verifier_source = payload["verifier_source"]
        kind = ArtifactKind(payload["artifact_kind"])
        answer_sample = payload["answer_sample"]
        expected_fact_keys = list(payload.get("expected_fact_keys", []))
        call_count_limit = int(payload["call_count_limit"])
        memory_limit_mb = payload.get("memory_limit_mb")
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        return _verifier_probe_error_payload(
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
            verifier_source,
            kind=kind,
            policy=policy,
        ),
    ]
    if errors:
        return {
            "request_id": request_id,
            "worker_pid": os.getpid(),
            "fetch_facts_return_keys": [],
            "expected_fact_keys": expected_fact_keys,
            "missing_fact_keys": [],
            "extra_fact_keys": [],
            "fetch_facts_tool_calls": None,
            "verify_tool_calls": None,
            "facts_match_result": None,
            "check_constraints_result": None,
            "verify_result": None,
            "errors": [error.model_dump(mode="json") for error in errors],
        }

    try:
        namespace = _load_namespace(
            source=verifier_source,
            artifact_kind=kind,
        )
        fetch_facts = namespace["fetch_facts"]
        facts_match = namespace["facts_match_answer_claims"]
        check_constraints = namespace["check_constraints"]
        verify = namespace["verify"]
        tool_names = _public_tool_names(tool_source)

        fetch_tools = _VerifierProbeTools(tool_names)
        facts, _ = asyncio.run(
            _invoke_entrypoint(
                function=fetch_facts,
                args=[answer_sample, fetch_tools],
                kwargs={},
                call_count_limit=call_count_limit,
            )
        )
        if not isinstance(facts, dict):
            return _verifier_probe_error_payload(
                request_id=request_id,
                code="fetch_facts_result_not_object",
                detail="fetch_facts() must return a dict of materialized facts.",
                expected_fact_keys=expected_fact_keys,
                fetch_facts_tool_calls=fetch_tools.call_count,
            )

        actual_fact_keys = sorted(str(key) for key in facts)
        expected_keys = sorted(str(key) for key in expected_fact_keys)
        missing_keys = [key for key in expected_keys if key not in actual_fact_keys]
        extra_keys = [key for key in actual_fact_keys if key not in expected_keys]

        facts_match_result, _ = asyncio.run(
            _invoke_entrypoint(
                function=facts_match,
                args=[answer_sample, facts],
                kwargs={},
                call_count_limit=call_count_limit,
            )
        )
        check_constraints_result, _ = asyncio.run(
            _invoke_entrypoint(
                function=check_constraints,
                args=[answer_sample, facts],
                kwargs={},
                call_count_limit=call_count_limit,
            )
        )
        verify_tools = _VerifierProbeTools(tool_names)
        verify_result, _ = asyncio.run(
            _invoke_entrypoint(
                function=verify,
                args=[answer_sample, verify_tools],
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
        return _verifier_probe_error_payload(
            request_id=request_id,
            code=code,
            detail=detail,
            expected_fact_keys=expected_fact_keys,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return _verifier_probe_error_payload(
            request_id=request_id,
            code="execution_error",
            detail=f"{type(exc).__name__}: {exc}",
            expected_fact_keys=expected_fact_keys,
        )

    if not isinstance(facts_match_result, bool):
        return _verifier_probe_error_payload(
            request_id=request_id,
            code="facts_match_result_not_bool",
            detail="facts_match_answer_claims() must return a bool.",
            expected_fact_keys=expected_keys,
            fetch_facts_return_keys=actual_fact_keys,
            missing_fact_keys=missing_keys,
            extra_fact_keys=extra_keys,
            fetch_facts_tool_calls=fetch_tools.call_count,
            verify_tool_calls=verify_tools.call_count,
            check_constraints_result=check_constraints_result
            if isinstance(check_constraints_result, bool)
            else None,
            verify_result=verify_result if isinstance(verify_result, bool) else None,
        )
    if not isinstance(check_constraints_result, bool):
        return _verifier_probe_error_payload(
            request_id=request_id,
            code="check_constraints_result_not_bool",
            detail="check_constraints() must return a bool.",
            expected_fact_keys=expected_keys,
            fetch_facts_return_keys=actual_fact_keys,
            missing_fact_keys=missing_keys,
            extra_fact_keys=extra_keys,
            fetch_facts_tool_calls=fetch_tools.call_count,
            verify_tool_calls=verify_tools.call_count,
            facts_match_result=facts_match_result,
            verify_result=verify_result if isinstance(verify_result, bool) else None,
        )
    if not isinstance(verify_result, bool):
        return _verifier_probe_error_payload(
            request_id=request_id,
            code="verify_result_not_bool",
            detail="verify() must return a bool.",
            expected_fact_keys=expected_keys,
            fetch_facts_return_keys=actual_fact_keys,
            missing_fact_keys=missing_keys,
            extra_fact_keys=extra_keys,
            fetch_facts_tool_calls=fetch_tools.call_count,
            verify_tool_calls=verify_tools.call_count,
            facts_match_result=facts_match_result,
            check_constraints_result=check_constraints_result,
        )
    if missing_keys or extra_keys:
        return _verifier_probe_error_payload(
            request_id=request_id,
            code="facts_schema_keys_mismatch",
            detail="fetch_facts() returned fact keys that do not match the declared facts schema.",
            expected_fact_keys=expected_keys,
            fetch_facts_return_keys=actual_fact_keys,
            missing_fact_keys=missing_keys,
            extra_fact_keys=extra_keys,
            fetch_facts_tool_calls=fetch_tools.call_count,
            verify_tool_calls=verify_tools.call_count,
            facts_match_result=facts_match_result,
            check_constraints_result=check_constraints_result,
            verify_result=verify_result,
        )
    expected_verify_result = check_constraints_result if facts_match_result else False
    if verify_result != expected_verify_result:
        return _verifier_probe_error_payload(
            request_id=request_id,
            code="verify_stage_outcome_mismatch",
            detail="verify() must reflect the staged pipeline outcome of facts_match_answer_claims() and check_constraints().",
            expected_fact_keys=expected_keys,
            fetch_facts_return_keys=actual_fact_keys,
            missing_fact_keys=missing_keys,
            extra_fact_keys=extra_keys,
            fetch_facts_tool_calls=fetch_tools.call_count,
            verify_tool_calls=verify_tools.call_count,
            facts_match_result=facts_match_result,
            check_constraints_result=check_constraints_result,
            verify_result=verify_result,
        )

    return {
        "request_id": request_id,
        "worker_pid": os.getpid(),
        "fetch_facts_return_keys": actual_fact_keys,
        "expected_fact_keys": expected_keys,
        "missing_fact_keys": [],
        "extra_fact_keys": [],
        "fetch_facts_tool_calls": fetch_tools.call_count,
        "verify_tool_calls": verify_tools.call_count,
        "facts_match_result": facts_match_result,
        "check_constraints_result": check_constraints_result,
        "verify_result": verify_result,
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
    if request_type == "probe_verifier_module":
        return _handle_probe_verifier(payload)
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
