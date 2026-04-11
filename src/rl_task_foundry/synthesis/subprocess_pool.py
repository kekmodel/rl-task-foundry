"""Persistent subprocess worker pool for registration-lane execution."""

from __future__ import annotations

import asyncio
import contextlib
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
    _stderr_chunks: list[str] = field(default_factory=list)
    _stderr_task: asyncio.Task[None] | None = None

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
        handle = cls(config=config, process=process, worker_index=worker_index)
        handle._stderr_task = asyncio.create_task(handle._drain_stderr())
        return handle

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def is_alive(self) -> bool:
        return self.process.returncode is None

    async def validate_module(
        self,
        *,
        source: str,
        artifact_kind: ArtifactKind,
    ) -> RegistrationSubprocessResult:
        response = await self._perform_request(
            request_type="validate_module",
            payload={
                "artifact_kind": artifact_kind.value,
                "source": source,
                "policy": self.config.synthesis.registration_policy.model_dump(mode="json"),
                "memory_limit_mb": self.config.synthesis.registration_workers.memory_limit_mb,
            },
        )
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
        response = await self._perform_request(
            request_type="execute_module_entrypoint",
            payload={
                "artifact_kind": artifact_kind.value,
                "source": source,
                "entrypoint": entrypoint,
                "args": args or [],
                "kwargs": kwargs or {},
                "policy": self.config.synthesis.registration_policy.model_dump(mode="json"),
                "memory_limit_mb": self.config.synthesis.registration_workers.memory_limit_mb,
                "call_count_limit": self.config.synthesis.registration_workers.call_count_limit,
            },
        )
        return RegistrationExecutionResult.model_validate(response)

    async def run_tool_self_test(
        self,
        *,
        tool_source: str,
        self_test_source: str,
    ) -> RegistrationExecutionResult:
        response = await self._perform_request(
            request_type="run_tool_self_test",
            payload={
                "tool_source": tool_source,
                "self_test_source": self_test_source,
                "policy": self.config.synthesis.registration_policy.model_dump(mode="json"),
                "memory_limit_mb": self.config.synthesis.registration_workers.memory_limit_mb,
                "call_count_limit": self.config.synthesis.registration_workers.call_count_limit,
            },
        )
        return RegistrationExecutionResult.model_validate(response)

    async def close(self) -> None:
        async with self._lock:
            if self.process.returncode is None:
                try:
                    request_id = f"worker-{self.worker_index}-shutdown-{self._request_counter}"
                    self._request_counter += 1
                    await self._request_locked(
                        {"type": "shutdown", "request_id": request_id},
                        expected_request_id=request_id,
                    )
                except RegistrationSubprocessError:
                    self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task

    async def _perform_request(
        self,
        *,
        request_type: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        async with self._lock:
            request_id = f"worker-{self.worker_index}-req-{self._request_counter}"
            self._request_counter += 1
            full_payload = {
                "type": request_type,
                "request_id": request_id,
                **payload,
            }
            return await self._request_locked(full_payload, expected_request_id=request_id)

    async def _request_locked(
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
        return "".join(self._stderr_chunks).strip()

    async def _drain_stderr(self) -> None:
        if self.process.stderr is None:
            return
        while True:
            chunk = await self.process.stderr.read(4096)
            if not chunk:
                return
            self._stderr_chunks.append(chunk.decode("utf-8", errors="replace"))
            joined = "".join(self._stderr_chunks)
            if len(joined) > 65536:
                self._stderr_chunks = [joined[-65536:]]


@dataclass(slots=True)
class RegistrationSubprocessPool:
    config: AppConfig
    workers: list[RegistrationWorkerHandle]
    _next_worker_index: int = 0
    _selection_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

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
        return await self._dispatch(
            "validate_module",
            source=source,
            artifact_kind=artifact_kind,
        )

    async def execute_module_entrypoint(
        self,
        *,
        source: str,
        artifact_kind: ArtifactKind,
        entrypoint: str,
        args: list[object] | None = None,
        kwargs: dict[str, object] | None = None,
    ) -> RegistrationExecutionResult:
        return await self._dispatch(
            "execute_module_entrypoint",
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
        return await self._dispatch(
            "run_tool_self_test",
            tool_source=tool_source,
            self_test_source=self_test_source,
        )

    async def close(self) -> None:
        await asyncio.gather(*(worker.close() for worker in self.workers))

    async def __aenter__(self) -> "RegistrationSubprocessPool":
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        await self.close()

    async def _dispatch(self, method_name: str, **kwargs: object):
        if not self.workers:
            raise RegistrationSubprocessError("registration subprocess pool has no workers")
        for _attempt in range(2):
            index, worker = await self._choose_worker()
            try:
                method = getattr(worker, method_name)
                return await method(**kwargs)
            except RegistrationSubprocessError:
                await self._replace_dead_worker(index)
        raise RegistrationSubprocessError(
            f"registration subprocess request failed after worker restart: method={method_name}"
        )

    async def _choose_worker(self) -> tuple[int, RegistrationWorkerHandle]:
        async with self._selection_lock:
            if not self.workers:
                raise RegistrationSubprocessError("registration subprocess pool has no workers")
            index = self._next_worker_index
            worker = self.workers[index]
            self._next_worker_index = (self._next_worker_index + 1) % len(self.workers)
        if not worker.is_alive:
            worker = await self._replace_dead_worker(index)
        return index, worker

    async def _replace_dead_worker(self, index: int) -> RegistrationWorkerHandle:
        async with self._selection_lock:
            worker = self.workers[index]
            if worker.is_alive:
                return worker
            with contextlib.suppress(Exception):
                await worker.close()
            replacement = await RegistrationWorkerHandle.start(
                config=self.config,
                worker_index=worker.worker_index,
            )
            self.workers[index] = replacement
            return replacement
