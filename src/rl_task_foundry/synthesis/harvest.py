"""Harvest target-bound synthesis runner.

Drives many independent single-shot trials in parallel until either a target
number of registry-committed tasks is reached, or no new commit lands within a
stall timeout. Each trial is a fresh anchor + fresh conversation; failures
(too_hard, validation, provider) just count as discarded attempts and another
trial follows.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.infra.db import DatabasePools, ensure_database_pools
from rl_task_foundry.pipeline.solver_orchestrator import SolverOrchestrator
from rl_task_foundry.synthesis.backend_openai_agents import OpenAIAgentsSynthesisBackend
from rl_task_foundry.synthesis.bundle_exporter import TaskBundleExporter
from rl_task_foundry.synthesis.phase_monitor import PipelinePhaseMonitorLogger
from rl_task_foundry.synthesis.pipeline_events import build_flow_id
from rl_task_foundry.synthesis.real_db_trial import (
    RealDbTrialRunner,
    RealDbTrialStatus,
    RealDbTrialSummary,
)
from rl_task_foundry.synthesis.task_registry import TaskRegistryWriter

TrialRunnerFactory = Callable[[], RealDbTrialRunner]


class HarvestOutcome(StrEnum):
    TARGET_REACHED = "target_reached"
    STALLED = "stalled"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class HarvestSummary:
    outcome: HarvestOutcome
    db_id: str
    target_committed: int
    committed: int
    attempted: int
    accepted_task_ids: tuple[str, ...]
    flow_id: str
    phase_monitor_log_path: Path
    output_root: Path
    elapsed_seconds: float
    trials: tuple[RealDbTrialSummary, ...]


@dataclass(slots=True)
class HarvestRunner:
    """Run trials until a target commit count or a stall timeout."""

    config: AppConfig
    registry: TaskRegistryWriter | None = None
    exporter: TaskBundleExporter | None = None
    trial_runner_factory: TrialRunnerFactory | None = None
    _shared_pools: DatabasePools | None = field(default=None, init=False, repr=False)
    _shared_solver_orchestrator: SolverOrchestrator | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = TaskRegistryWriter.for_config(self.config)
        if self.exporter is None:
            assert self.registry.atomic_tool_materializer is not None
            self.exporter = TaskBundleExporter(
                registry=self.registry,
                materializer=self.registry.atomic_tool_materializer,
            )

    def _build_trial_runner(self) -> RealDbTrialRunner:
        if self.trial_runner_factory is not None:
            return self.trial_runner_factory()
        return RealDbTrialRunner(
            self.config,
            registry=self.registry,
            exporter=self.exporter,
            database_pools=self._shared_pools,
            solver_orchestrator=self._shared_solver_orchestrator,
        )

    async def run(
        self,
        output_root: Path,
        *,
        db_id: str,
        target_committed: int,
        stall_timeout_seconds: float,
        parallel_workers: int = 1,
    ) -> HarvestSummary:
        if target_committed < 1:
            raise ValueError("target_committed must be >= 1")
        if stall_timeout_seconds <= 0:
            raise ValueError("stall_timeout_seconds must be > 0")
        if parallel_workers < 1:
            raise ValueError("parallel_workers must be >= 1")

        output_root.mkdir(parents=True, exist_ok=True)
        if self._shared_pools is None and self.trial_runner_factory is None:
            self._shared_pools = await ensure_database_pools(
                None, self.config.database
            )
        if (
            self._shared_solver_orchestrator is None
            and self.trial_runner_factory is None
        ):
            self._shared_solver_orchestrator = SolverOrchestrator(
                self.config,
                database_pools=self._shared_pools,
            )
        harvest_monitor_path = output_root / "phase_monitors.jsonl"
        flow_id = build_flow_id("harvest")
        harvest_monitor = PipelinePhaseMonitorLogger(
            phase_monitor_log_path=harvest_monitor_path,
            flow_kind="harvest",
            flow_id=flow_id,
        )

        started_at = time.monotonic()
        committed = 0
        attempted = 0
        last_commit_at = started_at
        accepted_task_ids: list[str] = []
        results: list[RealDbTrialSummary] = []
        results_lock = asyncio.Lock()
        counter_lock = asyncio.Lock()
        commit_state_lock = asyncio.Lock()
        done = asyncio.Event()

        harvest_monitor.emit(
            phase="harvest",
            status="started",
            expected_contract={
                "db_id": db_id,
                "target_committed": target_committed,
                "stall_timeout_seconds": stall_timeout_seconds,
                "parallel_workers": parallel_workers,
            },
            actual_data={"output_root": str(output_root)},
            checks={},
            diagnostics={},
        )

        async def worker(worker_id: int) -> None:
            nonlocal committed, attempted, last_commit_at
            while not done.is_set():
                async with counter_lock:
                    attempted += 1
                    trial_idx = attempted
                trial_root = output_root / "trials" / f"trial_{trial_idx:04d}"
                trial_runner = self._build_trial_runner()
                try:
                    summary = await trial_runner.run(
                        trial_root,
                        db_id=db_id,
                        mirror_monitor_path=harvest_monitor_path,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    harvest_monitor.emit(
                        phase="trial",
                        status="errored",
                        actual_data={"trial_index": trial_idx, "worker_id": worker_id},
                        diagnostics={
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    continue
                finally:
                    await trial_runner.close()

                async with results_lock:
                    results.append(summary)

                if summary.trial_status == RealDbTrialStatus.ACCEPTED:
                    async with commit_state_lock:
                        committed += 1
                        last_commit_at = time.monotonic()
                        if summary.registry_task_id is not None:
                            accepted_task_ids.append(summary.registry_task_id)
                        target_hit = committed >= target_committed
                    if target_hit:
                        done.set()
                        return

        async def watchdog() -> None:
            poll = max(1.0, min(stall_timeout_seconds / 4, 30.0))
            while not done.is_set():
                try:
                    await asyncio.wait_for(done.wait(), timeout=poll)
                except asyncio.TimeoutError:
                    pass
                if done.is_set():
                    return
                async with commit_state_lock:
                    idle = time.monotonic() - last_commit_at
                if idle >= stall_timeout_seconds:
                    done.set()
                    return

        worker_tasks = [
            asyncio.create_task(worker(i)) for i in range(parallel_workers)
        ]
        watchdog_task = asyncio.create_task(watchdog())
        try:
            await done.wait()
        finally:
            for task in worker_tasks:
                task.cancel()
            watchdog_task.cancel()
            await asyncio.gather(*worker_tasks, watchdog_task, return_exceptions=True)

        elapsed = time.monotonic() - started_at
        if committed >= target_committed:
            outcome = HarvestOutcome.TARGET_REACHED
        elif attempted == 0:
            outcome = HarvestOutcome.EMPTY
        else:
            outcome = HarvestOutcome.STALLED

        harvest_monitor.emit(
            phase="harvest",
            status=outcome.value,
            actual_data={
                "committed": committed,
                "attempted": attempted,
                "elapsed_seconds": round(elapsed, 1),
                "accepted_task_ids": list(accepted_task_ids),
            },
            checks={"target_reached": committed >= target_committed},
            diagnostics={},
        )
        harvest_monitor.close()

        return HarvestSummary(
            outcome=outcome,
            db_id=db_id,
            target_committed=target_committed,
            committed=committed,
            attempted=attempted,
            accepted_task_ids=tuple(accepted_task_ids),
            flow_id=flow_id,
            phase_monitor_log_path=harvest_monitor_path,
            output_root=output_root,
            elapsed_seconds=elapsed,
            trials=tuple(results),
        )

    async def close(self) -> None:
        if self._shared_solver_orchestrator is not None:
            await self._shared_solver_orchestrator.close()
            self._shared_solver_orchestrator = None
        OpenAIAgentsSynthesisBackend.clear_model_cache()
        close_registry = getattr(self.registry, "close", None)
        if callable(close_registry):
            close_registry()
        if self._shared_pools is not None:
            await self._shared_pools.close()
            self._shared_pools = None
