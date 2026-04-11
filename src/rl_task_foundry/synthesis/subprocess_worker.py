"""Persistent subprocess worker for registration-lane preflight tasks."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

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
