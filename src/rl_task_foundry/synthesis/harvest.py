"""Harvest target-bound synthesis runner.

Drives many independent single-shot trials in parallel until either a target
number of registry-committed tasks is reached, or no new commit lands within a
stall timeout. Each trial is a fresh anchor + fresh conversation; failures
(too_hard, validation, provider) just count as discarded attempts and another
trial follows. Production timing excludes one-time DB/pool/schema warm-up and
separates provider-issue trials from productive loop cost.
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
from rl_task_foundry.synthesis.phase_monitor import PipelinePhaseMonitorLogger
from rl_task_foundry.synthesis.pipeline_events import build_flow_id
from rl_task_foundry.synthesis.real_db_trial import (
    RealDbTrialRunner,
    RealDbTrialStatus,
    RealDbTrialSummary,
)
from rl_task_foundry.synthesis.synthesis_db import SynthesisDb
from rl_task_foundry.synthesis.task_registry import TaskRegistryWriter

TrialRunnerFactory = Callable[[], RealDbTrialRunner]


class HarvestOutcome(StrEnum):
    TARGET_REACHED = "target_reached"
    PRODUCTIVE_BUDGET_EXCEEDED = "productive_budget_exceeded"
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
    productive_elapsed_seconds: float
    productive_budget_seconds: float | None
    provider_issue_elapsed_seconds: float
    provider_issue_trials: int
    productive_seconds_per_accepted: float | None
    production_viability_passed: bool | None
    trials: tuple[RealDbTrialSummary, ...]


@dataclass(slots=True)
class HarvestRunner:
    """Run trials until a target commit count or a stall timeout."""

    config: AppConfig
    registry: TaskRegistryWriter | None = None
    trial_runner_factory: TrialRunnerFactory | None = None
    _shared_pools: DatabasePools | None = field(default=None, init=False, repr=False)
    _shared_solver_orchestrator: SolverOrchestrator | None = field(
        default=None, init=False, repr=False
    )
    _synthesis_dbs: dict[str, SynthesisDb] = field(
        default_factory=dict, init=False, repr=False
    )
    _synthesis_db_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = TaskRegistryWriter.for_config(self.config)

    def _build_trial_runner(self, *, db_id: str) -> RealDbTrialRunner:
        if self.trial_runner_factory is not None:
            return self.trial_runner_factory()
        return RealDbTrialRunner(
            self.config,
            registry=self.registry,
            database_pools=self._shared_pools,
            solver_orchestrator=self._shared_solver_orchestrator,
            synthesis_db=self._synthesis_dbs.get(db_id),
        )

    async def _ensure_synthesis_db(self, db_id: str) -> SynthesisDb:
        existing = self._synthesis_dbs.get(db_id)
        if existing is not None:
            return existing
        async with self._synthesis_db_lock:
            existing = self._synthesis_dbs.get(db_id)
            if existing is not None:
                return existing
            synthesis_db = SynthesisDb(
                db_id=db_id,
                config=self.config,
                database_pools=self._shared_pools,
            )
            self._synthesis_dbs[db_id] = synthesis_db
            return synthesis_db

    async def _warm_synthesis_db(self, db_id: str) -> None:
        synthesis_db = await self._ensure_synthesis_db(db_id)
        await synthesis_db.schema_graph()
        await synthesis_db.data_profile()
        await synthesis_db.schema_snapshot()

    async def run(
        self,
        output_root: Path,
        *,
        db_id: str,
        target_committed: int,
        stall_timeout_seconds: float,
        parallel_workers: int = 1,
        productive_budget_seconds: float | None = None,
    ) -> HarvestSummary:
        if target_committed < 1:
            raise ValueError("target_committed must be >= 1")
        if stall_timeout_seconds <= 0:
            raise ValueError("stall_timeout_seconds must be > 0")
        if parallel_workers < 1:
            raise ValueError("parallel_workers must be >= 1")
        if productive_budget_seconds is not None and productive_budget_seconds <= 0:
            raise ValueError("productive_budget_seconds must be > 0")

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
        if self.trial_runner_factory is None:
            await self._warm_synthesis_db(db_id)
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
                "productive_budget_seconds": productive_budget_seconds,
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
                trial_runner = self._build_trial_runner(db_id=db_id)
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
                    productive_so_far = _productive_elapsed_seconds(results)

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
                if (
                    productive_budget_seconds is not None
                    and productive_so_far >= productive_budget_seconds
                ):
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
        provider_issue_trials = tuple(
            trial for trial in results if _is_provider_issue_trial(trial)
        )
        provider_issue_elapsed = _provider_issue_elapsed_seconds(results)
        productive_elapsed = _productive_elapsed_seconds(results)
        productive_seconds_per_accepted = (
            productive_elapsed / committed if committed else None
        )
        production_viability_passed = (
            committed >= target_committed
            and productive_elapsed <= productive_budget_seconds
            if productive_budget_seconds is not None
            else None
        )
        productive_budget_exceeded = (
            productive_budget_seconds is not None
            and productive_elapsed >= productive_budget_seconds
            and production_viability_passed is not True
        )
        if productive_budget_exceeded:
            outcome = HarvestOutcome.PRODUCTIVE_BUDGET_EXCEEDED
        elif committed >= target_committed:
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
                "productive_elapsed_seconds": round(productive_elapsed, 1),
                "productive_budget_seconds": productive_budget_seconds,
                "productive_seconds_per_accepted": (
                    round(productive_seconds_per_accepted, 1)
                    if productive_seconds_per_accepted is not None
                    else None
                ),
                "provider_issue_trials": len(provider_issue_trials),
                "provider_issue_elapsed_seconds": round(provider_issue_elapsed, 1),
                "production_viability_passed": production_viability_passed,
                "productive_budget_exceeded": productive_budget_exceeded,
                "accepted_task_ids": list(accepted_task_ids),
            },
            checks={
                "target_reached": committed >= target_committed,
                "accepted_3_within_productive_15_min": (
                    production_viability_passed
                    if productive_budget_seconds is not None
                    and target_committed == 3
                    else None
                ),
                "productive_seconds_per_accepted_within_300s": (
                    productive_seconds_per_accepted <= 300.0
                    if productive_seconds_per_accepted is not None
                    else None
                ),
            },
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
            productive_elapsed_seconds=productive_elapsed,
            productive_budget_seconds=productive_budget_seconds,
            provider_issue_elapsed_seconds=provider_issue_elapsed,
            provider_issue_trials=len(provider_issue_trials),
            productive_seconds_per_accepted=productive_seconds_per_accepted,
            production_viability_passed=production_viability_passed,
            trials=tuple(results),
        )

    async def close(self) -> None:
        if self._shared_solver_orchestrator is not None:
            await self._shared_solver_orchestrator.close()
            self._shared_solver_orchestrator = None
        for synthesis_db in self._synthesis_dbs.values():
            await synthesis_db.close()
        self._synthesis_dbs.clear()
        OpenAIAgentsSynthesisBackend.clear_model_cache()
        close_registry = getattr(self.registry, "close", None)
        if callable(close_registry):
            close_registry()
        if self._shared_pools is not None:
            await self._shared_pools.close()
            self._shared_pools = None


def _is_provider_issue_trial(summary: RealDbTrialSummary) -> bool:
    if summary.synthesis_error_type == "SynthesisProviderUnavailableError":
        return True
    return (
        summary.synthesis_error_type == "SynthesisPhaseExecutionError"
        and bool(summary.backend_failures)
    )


def _provider_issue_elapsed_seconds(summaries: list[RealDbTrialSummary]) -> float:
    return sum(
        summary.elapsed_seconds or 0.0
        for summary in summaries
        if _is_provider_issue_trial(summary)
    )


def _productive_elapsed_seconds(summaries: list[RealDbTrialSummary]) -> float:
    return sum(
        summary.elapsed_seconds or 0.0
        for summary in summaries
        if not _is_provider_issue_trial(summary)
    )
