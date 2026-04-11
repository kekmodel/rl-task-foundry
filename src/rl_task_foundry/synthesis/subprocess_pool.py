"""Persistent subprocess worker pool for registration-lane execution."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.registration_policy import (
    ArtifactKind,
    RegistrationError,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegistrationSubprocessError(RuntimeError):
    """Raised when a registration worker subprocess fails unexpectedly."""


class RegistrationSubprocessResult(StrictModel):
    request_id: str
    worker_pid: int
    errors: list[RegistrationError]


class RegistrationExecutionResult(StrictModel):
    request_id: str
    worker_pid: int
    errors: list[RegistrationError]
    call_count: int | None = None
    return_value: object | None = None


@dataclass(slots=True)
class RegistrationWorkerHandle:
    config: AppConfig
    process: asyncio.subprocess.Process
    worker_index: int
    _request_counter: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    async def start(
        cls,
        *,
        config: AppConfig,
        worker_index: int,
    ) -> "RegistrationWorkerHandle":
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "rl_task_foundry.synthesis.subprocess_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return cls(config=config, process=process, worker_index=worker_index)

    @property
    def pid(self) -> int | None:
        return self.process.pid

    async def validate_module(
        self,
        *,
        source: str,
        artifact_kind: ArtifactKind,
    ) -> RegistrationSubprocessResult:
        request_id = f"worker-{self.worker_index}-req-{self._request_counter}"
        self._request_counter += 1
        payload = {
            "type": "validate_module",
            "request_id": request_id,
            "artifact_kind": artifact_kind.value,
            "source": source,
            "policy": self.config.synthesis.registration_policy.model_dump(mode="json"),
            "memory_limit_mb": self.config.synthesis.registration_workers.memory_limit_mb,
        }
        response = await self._request(payload, expected_request_id=request_id)
        return RegistrationSubprocessResult.model_validate(response)

    async def execute_module_entrypoint(
        self,
        *,
        source: str,
        artifact_kind: ArtifactKind,
        entrypoint: str,
        args: list[object] | None = None,
        kwargs: dict[str, object] | None = None,
    ) -> RegistrationExecutionResult:
        request_id = f"worker-{self.worker_index}-req-{self._request_counter}"
        self._request_counter += 1
        payload = {
            "type": "execute_module_entrypoint",
            "request_id": request_id,
            "artifact_kind": artifact_kind.value,
            "source": source,
            "entrypoint": entrypoint,
            "args": args or [],
            "kwargs": kwargs or {},
            "policy": self.config.synthesis.registration_policy.model_dump(mode="json"),
            "memory_limit_mb": self.config.synthesis.registration_workers.memory_limit_mb,
            "call_count_limit": self.config.synthesis.registration_workers.call_count_limit,
        }
        response = await self._request(payload, expected_request_id=request_id)
        return RegistrationExecutionResult.model_validate(response)

    async def run_tool_self_test(
        self,
        *,
        tool_source: str,
        self_test_source: str,
    ) -> RegistrationExecutionResult:
        request_id = f"worker-{self.worker_index}-req-{self._request_counter}"
        self._request_counter += 1
        payload = {
            "type": "run_tool_self_test",
            "request_id": request_id,
            "tool_source": tool_source,
            "self_test_source": self_test_source,
            "policy": self.config.synthesis.registration_policy.model_dump(mode="json"),
            "memory_limit_mb": self.config.synthesis.registration_workers.memory_limit_mb,
            "call_count_limit": self.config.synthesis.registration_workers.call_count_limit,
        }
        response = await self._request(payload, expected_request_id=request_id)
        return RegistrationExecutionResult.model_validate(response)

    async def close(self) -> None:
        if self.process.returncode is not None:
            return
        try:
            await self._request({"type": "shutdown", "request_id": f"worker-{self.worker_index}-shutdown"})
        except RegistrationSubprocessError:
            self.process.terminate()
        await self.process.wait()

    async def _request(
        self,
        payload: dict[str, object],
        *,
        expected_request_id: str | None = None,
    ) -> dict[str, object]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RegistrationSubprocessError("registration worker subprocess pipes are unavailable")
        if self.process.returncode is not None:
            stderr = await self._read_stderr()
            raise RegistrationSubprocessError(
                f"registration worker exited before request: returncode={self.process.returncode}, stderr={stderr}"
            )

        async with self._lock:
            self.process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            await self.process.stdin.drain()
            try:
                raw = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=self.config.synthesis.registration_workers.task_timeout_s,
                )
            except TimeoutError as exc:
                self.process.kill()
                await self.process.wait()
                raise RegistrationSubprocessError("registration worker request timed out") from exc
            if not raw:
                stderr = await self._read_stderr()
                raise RegistrationSubprocessError(
                    f"registration worker closed stdout unexpectedly: stderr={stderr}"
                )
            try:
                response = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                stderr = await self._read_stderr()
                raise RegistrationSubprocessError(
                    f"registration worker emitted invalid JSON: stderr={stderr}"
                ) from exc

        request_id = response.get("request_id")
        if expected_request_id is not None and request_id != expected_request_id:
            raise RegistrationSubprocessError(
                f"registration worker returned mismatched request id: expected={expected_request_id}, actual={request_id}"
            )

        response.pop("shutdown", None)
        return response

    async def _read_stderr(self) -> str:
        if self.process.stderr is None:
            return ""
        raw = await self.process.stderr.read()
        return raw.decode("utf-8", errors="replace").strip()


@dataclass(slots=True)
class RegistrationSubprocessPool:
    config: AppConfig
    workers: list[RegistrationWorkerHandle]
    _next_worker_index: int = 0

    @classmethod
    async def start(cls, config: AppConfig) -> "RegistrationSubprocessPool":
        workers = [
            await RegistrationWorkerHandle.start(config=config, worker_index=index)
            for index in range(config.synthesis.registration_workers.worker_count)
        ]
        return cls(config=config, workers=workers)

    async def validate_module(
        self,
        *,
        source: str,
        artifact_kind: ArtifactKind,
    ) -> RegistrationSubprocessResult:
        if not self.workers:
            raise RegistrationSubprocessError("registration subprocess pool has no workers")
        worker = self.workers[self._next_worker_index]
        self._next_worker_index = (self._next_worker_index + 1) % len(self.workers)
        return await worker.validate_module(source=source, artifact_kind=artifact_kind)

    async def execute_module_entrypoint(
        self,
        *,
        source: str,
        artifact_kind: ArtifactKind,
        entrypoint: str,
        args: list[object] | None = None,
        kwargs: dict[str, object] | None = None,
    ) -> RegistrationExecutionResult:
        if not self.workers:
            raise RegistrationSubprocessError("registration subprocess pool has no workers")
        worker = self.workers[self._next_worker_index]
        self._next_worker_index = (self._next_worker_index + 1) % len(self.workers)
        return await worker.execute_module_entrypoint(
            source=source,
            artifact_kind=artifact_kind,
            entrypoint=entrypoint,
            args=args,
            kwargs=kwargs,
        )

    async def run_tool_self_test(
        self,
        *,
        tool_source: str,
        self_test_source: str,
    ) -> RegistrationExecutionResult:
        if not self.workers:
            raise RegistrationSubprocessError("registration subprocess pool has no workers")
        worker = self.workers[self._next_worker_index]
        self._next_worker_index = (self._next_worker_index + 1) % len(self.workers)
        return await worker.run_tool_self_test(
            tool_source=tool_source,
            self_test_source=self_test_source,
        )

    async def close(self) -> None:
        await asyncio.gather(*(worker.close() for worker in self.workers))

    async def __aenter__(self) -> "RegistrationSubprocessPool":
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        await self.close()
