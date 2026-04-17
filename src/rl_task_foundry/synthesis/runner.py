"""Checkpoint-aware multi-db synthesis registry runner."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from rl_task_foundry.config.models import AppConfig, DatabaseConfig, DomainConfig
from rl_task_foundry.infra.checkpoint import CheckpointStore, ensure_checkpoint
from rl_task_foundry.synthesis.contracts import (
    StrictModel,
    TaskBundleStatus,
)
from rl_task_foundry.synthesis.backend_openai_agents import OpenAIAgentsSynthesisBackend
from rl_task_foundry.synthesis.orchestrator import (
    SynthesisDbRegistryEntry,
    SynthesisOrchestrator,
    SynthesisRuntimeHandle,
)
from rl_task_foundry.synthesis.phase_monitor import (
    PipelinePhaseMonitorLogger,
    default_phase_monitor_log_path,
)
from rl_task_foundry.synthesis.pipeline_events import build_flow_id
from rl_task_foundry.synthesis.runtime import SynthesisAgentRuntime
from rl_task_foundry.synthesis.scheduler import (
    SynthesisSchedulerDecision,
    SynthesisSelectionStatus,
)
from rl_task_foundry.synthesis.task_registry import (
    TaskRegistryCommitStatus,
    TaskRegistryWriter,
)


class SynthesisRegistryFileEntry(StrictModel):
    db_id: str
    database: DatabaseConfig | None = None
    domain: DomainConfig | None = None

    def to_registry_entry(self) -> SynthesisDbRegistryEntry:
        return SynthesisDbRegistryEntry(
            db_id=self.db_id,
            database=self.database,
            domain=self.domain,
        )


class SynthesisRegistryRunOutcome(StrEnum):
    COMPLETED_ALL = "completed_all"
    MAX_STEPS_REACHED = "max_steps_reached"
    ALL_BACKED_OFF = "all_backed_off"
    EMPTY_REGISTRY = "empty_registry"


@dataclass(slots=True)
class SynthesisRegistryStepSummary:
    decision: SynthesisSchedulerDecision
    draft_task_id: str | None = None
    draft_created_at: datetime | None = None
    quality_gate_status: str | None = None
    quality_gate_pass_rate: float | None = None
    quality_gate_ci_low: float | None = None
    quality_gate_ci_high: float | None = None
    registry_status: TaskRegistryCommitStatus | None = None
    registry_task_id: str | None = None


@dataclass(slots=True)
class SynthesisRegistryRunSummary:
    outcome: SynthesisRegistryRunOutcome
    checkpoint_namespace: str
    requested_steps: int
    executed_steps: int
    total_entries: int
    initially_processed_entries: int
    processed_entries_after_run: int
    generated_drafts: int
    registry_committed_tasks: int
    registry_duplicate_tasks: int
    remaining_entries: int
    flow_id: str | None = None
    phase_monitor_log_path: Path | None = None
    quality_accepted_tasks: int = 0
    quality_rejected_tasks: int = 0
    generated_task_ids: list[str] = field(default_factory=list)
    committed_task_ids: list[str] = field(default_factory=list)
    duplicate_task_ids: list[str] = field(default_factory=list)
    quality_rejected_task_ids: list[str] = field(default_factory=list)
    registry_root_dir: Path | None = None
    registry_index_db_path: Path | None = None
    steps: list[SynthesisRegistryStepSummary] = field(default_factory=list)

    # Backward-compatible aliases
    @property
    def total_pairs(self) -> int:
        return self.total_entries

    @property
    def initially_processed_pairs(self) -> int:
        return self.initially_processed_entries

    @property
    def processed_pairs_after_run(self) -> int:
        return self.processed_entries_after_run

    @property
    def remaining_pairs(self) -> int:
        return self.remaining_entries

    @property
    def last_decision(self) -> SynthesisSchedulerDecision | None:
        if not self.steps:
            return None
        return self.steps[-1].decision


@dataclass(slots=True)
class SynthesisRegistryRunner:
    """Drive the synthesis orchestrator over a registry with checkpoint resume.

    The registry must contain unique `db_id` values. Only entries whose generated
    draft passes the solver pass-rate quality gate are flushed to checkpoint,
    which keeps quality-rejected entries eligible for regeneration on the
    next bounded run.
    """

    base_config: AppConfig
    runtime_factory: Callable[[SynthesisDbRegistryEntry], SynthesisRuntimeHandle] | None = None
    task_registry: TaskRegistryWriter | None = None
    checkpoint: CheckpointStore | None = None
    orchestrator: SynthesisOrchestrator | None = None

    def __post_init__(self) -> None:
        if self.checkpoint is None:
            self.checkpoint = ensure_checkpoint(self.base_config.output.run_db_path)
        if self.task_registry is None:
            self.task_registry = TaskRegistryWriter.for_config(self.base_config)
        if self.orchestrator is None:
            self.orchestrator = SynthesisOrchestrator(
                runtime_factory=self.runtime_factory or self._build_runtime
            )

    async def run_steps(
        self,
        registry: list[SynthesisDbRegistryEntry],
        *,
        max_steps: int,
        checkpoint_namespace: str = "synthesis_registry",
    ) -> SynthesisRegistryRunSummary:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        _validate_unique_db_ids(registry)
        flow_id = build_flow_id("synthesis_registry")
        phase_monitor_log_path = default_phase_monitor_log_path(self.base_config.output.traces_dir)
        phase_monitor = PipelinePhaseMonitorLogger(
            phase_monitor_log_path=phase_monitor_log_path,
            flow_kind="synthesis_registry",
            flow_id=flow_id,
        )
        try:
            assert self.task_registry is not None
            if not registry:
                return SynthesisRegistryRunSummary(
                    outcome=SynthesisRegistryRunOutcome.EMPTY_REGISTRY,
                    checkpoint_namespace=checkpoint_namespace,
                    flow_id=flow_id,
                    requested_steps=max_steps,
                    executed_steps=0,
                    total_entries=0,
                    initially_processed_entries=0,
                    processed_entries_after_run=0,
                    generated_drafts=0,
                    registry_committed_tasks=0,
                    registry_duplicate_tasks=0,
                    remaining_entries=0,
                    phase_monitor_log_path=phase_monitor_log_path,
                    quality_accepted_tasks=0,
                    quality_rejected_tasks=0,
                    registry_root_dir=self.task_registry.root_dir,
                    registry_index_db_path=self.task_registry.index_db_path,
                )

            total_entries = len(registry)
            pending_registry, initially_processed_entries = self._pending_registry(
                registry,
                checkpoint_namespace=checkpoint_namespace,
            )
            steps: list[SynthesisRegistryStepSummary] = []
            generated_task_ids: list[str] = []
            committed_task_ids: list[str] = []
            duplicate_task_ids: list[str] = []
            quality_rejected_task_ids: list[str] = []
            executed_steps = 0
            generated_drafts = 0
            registry_committed_tasks = 0
            registry_duplicate_tasks = 0
            quality_accepted_tasks = 0
            quality_rejected_tasks = 0
            processed_entries_after_run = initially_processed_entries
            assert self.orchestrator is not None
            assert self.checkpoint is not None
            assert self.task_registry is not None
            orchestrator = self.orchestrator
            checkpoint = self.checkpoint
            task_registry = self.task_registry
            outcome: SynthesisRegistryRunOutcome | None = None

            parallel_workers = self.base_config.synthesis.parallel_workers
            results_lock = asyncio.Lock()
            step_counter = 0
            steps_remaining = max_steps
            entry_queue: asyncio.Queue[SynthesisDbRegistryEntry] = asyncio.Queue()
            for entry in pending_registry:
                entry_queue.put_nowait(entry)

            async def _run_worker(worker_id: int) -> None:
                nonlocal step_counter, steps_remaining
                nonlocal executed_steps, generated_drafts, quality_accepted_tasks
                nonlocal quality_rejected_tasks, registry_committed_tasks
                nonlocal registry_duplicate_tasks, processed_entries_after_run
                nonlocal pending_registry, outcome

                worker_orchestrator = SynthesisOrchestrator(
                    runtime_factory=self.runtime_factory or self._build_runtime
                )
                log = logging.getLogger(f"synthesis.worker.{worker_id}")
                try:
                    while True:
                        async with results_lock:
                            if steps_remaining <= 0:
                                return
                            steps_remaining -= 1
                        if entry_queue.empty():
                            return
                        try:
                            queue_entry = entry_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return

                        try:
                            step = await worker_orchestrator.run_next([queue_entry])
                        except Exception as exc:
                            log.warning("worker %d step failed: %s", worker_id, exc)
                            async with results_lock:
                                executed_steps += 1
                            continue

                        async with results_lock:
                            step_counter += 1
                            executed_steps += 1

                            if step.decision.status != SynthesisSelectionStatus.READY:
                                steps.append(SynthesisRegistryStepSummary(decision=step.decision))
                                return
                            if step.draft is None:
                                return

                            assert step.decision.db_id is not None
                            phase_monitor.emit(
                                phase="quality_gate",
                                status="accept",
                                expected_contract={
                                    "step_index": step_counter,
                                    "internal_submit_draft_acceptance": True,
                                },
                                actual_data={
                                    "task_id": step.draft.task_bundle.task_id,
                                    "pass_rate": step.draft.task_bundle.quality_metrics.solver_pass_rate,
                                    "ci_low": step.draft.task_bundle.quality_metrics.solver_ci_low,
                                    "ci_high": step.draft.task_bundle.quality_metrics.solver_ci_high,
                                },
                                checks={
                                    "accepted": step.draft.task_bundle.status is TaskBundleStatus.ACCEPTED,
                                },
                                diagnostics={"step_index": step_counter},
                            )
                            generated_drafts += 1
                            generated_task_ids.append(step.draft.task_bundle.task_id)
                            quality_accepted_tasks += 1
                            commit_result = task_registry.commit_draft(step.draft)
                            phase_monitor.emit(
                                phase="registry_commit",
                                status=commit_result.status.value,
                                expected_contract={"step_index": step_counter},
                                actual_data={
                                    "task_id": step.draft.task_bundle.task_id,
                                    "registry_task_id": commit_result.task_id,
                                    "status": commit_result.status.value,
                                },
                                checks={},
                                diagnostics={"step_index": step_counter},
                            )
                            steps.append(
                                SynthesisRegistryStepSummary(
                                    decision=step.decision,
                                    draft_task_id=step.draft.task_bundle.task_id,
                                    draft_created_at=step.draft.created_at,
                                    quality_gate_status="accept",
                                    quality_gate_pass_rate=step.draft.task_bundle.quality_metrics.solver_pass_rate,
                                    quality_gate_ci_low=step.draft.task_bundle.quality_metrics.solver_ci_low,
                                    quality_gate_ci_high=step.draft.task_bundle.quality_metrics.solver_ci_high,
                                    registry_status=commit_result.status,
                                    registry_task_id=commit_result.task_id,
                                )
                            )
                            checkpoint.mark_processed(
                                self._checkpoint_key(step.decision.db_id),
                                namespace=checkpoint_namespace,
                                payload={
                                    "db_id": step.decision.db_id,
                                    "task_id": commit_result.task_id,
                                    "created_at": step.draft.created_at.isoformat(),
                                    "registry_status": commit_result.status.value,
                                    "quality_gate_status": "accept",
                                    "solver_pass_rate": step.draft.task_bundle.quality_metrics.solver_pass_rate,
                                    "solver_ci_low": step.draft.task_bundle.quality_metrics.solver_ci_low,
                                    "solver_ci_high": step.draft.task_bundle.quality_metrics.solver_ci_high,
                                },
                            )
                            checkpoint.flush()
                            processed_entries_after_run += 1
                            if commit_result.status == TaskRegistryCommitStatus.COMMITTED:
                                registry_committed_tasks += 1
                                committed_task_ids.append(commit_result.task_id)
                            else:
                                registry_duplicate_tasks += 1
                                duplicate_task_ids.append(commit_result.task_id)
                            pending_registry = [
                                e for e in pending_registry
                                if e.db_id != step.decision.db_id
                            ]
                finally:
                    await worker_orchestrator.close()

            worker_count = min(parallel_workers, max_steps, len(pending_registry) or 1)
            results = await asyncio.gather(
                *(_run_worker(i) for i in range(worker_count)),
                return_exceptions=True,
            )
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    logging.getLogger("synthesis.runner").error(
                        "worker %d crashed: %s: %s", i, type(result).__name__, result
                    )

            remaining_entries = total_entries - processed_entries_after_run
            if outcome is None:
                outcome = (
                    SynthesisRegistryRunOutcome.COMPLETED_ALL
                    if remaining_entries == 0
                    else SynthesisRegistryRunOutcome.MAX_STEPS_REACHED
                )
            assert remaining_entries == total_entries - processed_entries_after_run

            return SynthesisRegistryRunSummary(
                outcome=outcome,
                checkpoint_namespace=checkpoint_namespace,
                flow_id=flow_id,
                requested_steps=max_steps,
                executed_steps=executed_steps,
                total_entries=total_entries,
                initially_processed_entries=initially_processed_entries,
                processed_entries_after_run=processed_entries_after_run,
                generated_drafts=generated_drafts,
                registry_committed_tasks=registry_committed_tasks,
                registry_duplicate_tasks=registry_duplicate_tasks,
                remaining_entries=remaining_entries,
                phase_monitor_log_path=phase_monitor_log_path,
                quality_accepted_tasks=quality_accepted_tasks,
                quality_rejected_tasks=quality_rejected_tasks,
                generated_task_ids=generated_task_ids,
                committed_task_ids=committed_task_ids,
                duplicate_task_ids=duplicate_task_ids,
                quality_rejected_task_ids=quality_rejected_task_ids,
                registry_root_dir=task_registry.root_dir,
                registry_index_db_path=task_registry.index_db_path,
                steps=steps,
            )
        finally:
            phase_monitor.close()

    async def close(self) -> None:
        assert self.orchestrator is not None
        await self.orchestrator.close()
        OpenAIAgentsSynthesisBackend.clear_model_cache()
        close_registry = getattr(self.task_registry, "close", None)
        if callable(close_registry):
            close_registry()

    def _build_runtime(self, entry: SynthesisDbRegistryEntry) -> SynthesisRuntimeHandle:
        config = self.base_config
        if entry.database is not None or entry.domain is not None:
            config = self.base_config.model_copy(
                update={
                    "database": entry.database or self.base_config.database,
                    "domain": entry.domain or self.base_config.domain,
                },
                deep=True,
            )
        return SynthesisAgentRuntime(config)

    def _pending_registry(
        self,
        registry: list[SynthesisDbRegistryEntry],
        *,
        checkpoint_namespace: str,
    ) -> tuple[list[SynthesisDbRegistryEntry], int]:
        assert self.checkpoint is not None
        pending_entries: list[SynthesisDbRegistryEntry] = []
        already_processed = 0
        for entry in registry:
            if self.checkpoint.is_processed(
                self._checkpoint_key(entry.db_id),
                namespace=checkpoint_namespace,
            ):
                already_processed += 1
            else:
                pending_entries.append(entry)
        return pending_entries, already_processed

    @staticmethod
    def _checkpoint_key(db_id: str) -> str:
        return db_id


def load_synthesis_registry(path: Path) -> list[SynthesisDbRegistryEntry]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        payload = json.loads(raw)
        entries = [_parse_registry_item(item) for item in payload]
    else:
        entries = [
            _parse_registry_item(json.loads(line)) for line in raw.splitlines() if line.strip()
        ]
    _validate_unique_db_ids(entries)
    return entries


def _parse_registry_item(item: dict[str, Any]) -> SynthesisDbRegistryEntry:
    # Accept and ignore legacy topics/categories keys
    cleaned = {k: v for k, v in item.items() if k not in ("topics", "categories")}
    payload = SynthesisRegistryFileEntry.model_validate(cleaned)
    return payload.to_registry_entry()


def _validate_unique_db_ids(registry: list[SynthesisDbRegistryEntry]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in registry:
        if entry.db_id in seen and entry.db_id not in duplicates:
            duplicates.append(entry.db_id)
        seen.add(entry.db_id)
    if duplicates:
        raise ValueError(
            "synthesis registry must not contain duplicate db_id values: "
            + ", ".join(sorted(duplicates))
        )
