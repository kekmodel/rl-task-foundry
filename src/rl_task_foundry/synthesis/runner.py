"""Checkpoint-aware multi-db synthesis registry runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from enum import StrEnum

from pydantic import Field

from rl_task_foundry.config.models import AppConfig, DatabaseConfig, DomainConfig
from rl_task_foundry.infra.checkpoint import CheckpointStore, ensure_checkpoint
from rl_task_foundry.synthesis.contracts import (
    CategoryTaxonomy,
    EnvironmentContract,
    EnvironmentStatus,
    StrictModel,
)
from rl_task_foundry.synthesis.cross_instance import evaluate_cross_instance_draft
from rl_task_foundry.synthesis.environment_registry import (
    EnvironmentRegistryCommitResult,
    EnvironmentRegistryCommitStatus,
    EnvironmentRegistryWriter,
)
from rl_task_foundry.synthesis.quality_gate import accepted_draft_with_quality_metrics
from rl_task_foundry.synthesis.orchestrator import (
    SynthesisDbRegistryEntry,
    SynthesisOrchestrator,
    SynthesisRuntimeHandle,
)
from rl_task_foundry.synthesis.runtime import SynthesisAgentRuntime, SynthesisEnvironmentDraft
from rl_task_foundry.synthesis.scheduler import (
    SynthesisSchedulerDecision,
    SynthesisSelectionStatus,
)


class SynthesisRegistryFileEntry(StrictModel):
    db_id: str
    categories: list[CategoryTaxonomy] = Field(min_length=1)
    database: DatabaseConfig | None = None
    domain: DomainConfig | None = None

    def to_registry_entry(self) -> SynthesisDbRegistryEntry:
        return SynthesisDbRegistryEntry(
            db_id=self.db_id,
            categories=list(self.categories),
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
    draft_env_id: str | None = None
    draft_created_at: datetime | None = None
    quality_gate_status: str | None = None
    cross_instance_error_codes: list[str] = field(default_factory=list)
    quality_gate_pass_rate: float | None = None
    quality_gate_ci_low: float | None = None
    quality_gate_ci_high: float | None = None
    registry_status: EnvironmentRegistryCommitStatus | None = None
    registry_env_id: str | None = None


@dataclass(slots=True)
class SynthesisRegistryRunSummary:
    outcome: SynthesisRegistryRunOutcome
    checkpoint_namespace: str
    requested_steps: int
    executed_steps: int
    total_pairs: int
    initially_processed_pairs: int
    processed_pairs_after_run: int
    generated_drafts: int
    registry_committed_envs: int
    registry_duplicate_envs: int
    remaining_pairs: int
    quality_accepted_envs: int = 0
    quality_rejected_envs: int = 0
    generated_env_ids: list[str] = field(default_factory=list)
    committed_env_ids: list[str] = field(default_factory=list)
    duplicate_env_ids: list[str] = field(default_factory=list)
    quality_rejected_env_ids: list[str] = field(default_factory=list)
    registry_root_dir: Path | None = None
    registry_index_db_path: Path | None = None
    steps: list[SynthesisRegistryStepSummary] = field(default_factory=list)

    @property
    def last_decision(self) -> SynthesisSchedulerDecision | None:
        if not self.steps:
            return None
        return self.steps[-1].decision


@dataclass(slots=True)
class SynthesisRegistryRunner:
    """Drive the synthesis orchestrator over a registry with checkpoint resume.

    The registry must contain unique `db_id` values. Only pairs whose generated
    draft passes the solver pass-rate quality gate are flushed to checkpoint,
    which keeps quality-rejected categories eligible for regeneration on the
    next bounded run.
    """

    base_config: AppConfig
    runtime_factory: Callable[[SynthesisDbRegistryEntry], SynthesisRuntimeHandle] | None = None
    environment_registry: EnvironmentRegistryWriter | None = None
    checkpoint: CheckpointStore | None = None
    orchestrator: SynthesisOrchestrator | None = None
    environment_orchestrator: Any | None = None

    def __post_init__(self) -> None:
        if self.checkpoint is None:
            self.checkpoint = ensure_checkpoint(self.base_config.output.run_db_path)
        if self.environment_registry is None:
            self.environment_registry = EnvironmentRegistryWriter.for_config(self.base_config)
        if self.orchestrator is None:
            self.orchestrator = SynthesisOrchestrator(
                runtime_factory=self.runtime_factory or self._build_runtime
            )
        if self.environment_orchestrator is None:
            from rl_task_foundry.pipeline.environment_orchestrator import EnvironmentOrchestrator

            self.environment_orchestrator = EnvironmentOrchestrator(self.base_config)

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
        if not registry:
            return SynthesisRegistryRunSummary(
                outcome=SynthesisRegistryRunOutcome.EMPTY_REGISTRY,
                checkpoint_namespace=checkpoint_namespace,
                requested_steps=max_steps,
                executed_steps=0,
                total_pairs=0,
                initially_processed_pairs=0,
                processed_pairs_after_run=0,
                generated_drafts=0,
                registry_committed_envs=0,
                registry_duplicate_envs=0,
                remaining_pairs=0,
                quality_accepted_envs=0,
                quality_rejected_envs=0,
                registry_root_dir=self.environment_registry.root_dir,
                registry_index_db_path=self.environment_registry.index_db_path,
            )

        total_pairs = sum(len(entry.categories) for entry in registry)
        pending_registry, initially_processed_pairs = self._pending_registry(
            registry,
            checkpoint_namespace=checkpoint_namespace,
        )
        steps: list[SynthesisRegistryStepSummary] = []
        generated_env_ids: list[str] = []
        committed_env_ids: list[str] = []
        duplicate_env_ids: list[str] = []
        quality_rejected_env_ids: list[str] = []
        executed_steps = 0
        generated_drafts = 0
        registry_committed_envs = 0
        registry_duplicate_envs = 0
        quality_accepted_envs = 0
        quality_rejected_envs = 0
        cross_instance_rejected_envs = 0
        processed_pairs_after_run = initially_processed_pairs
        orchestrator = self.orchestrator
        environment_orchestrator = self.environment_orchestrator
        checkpoint = self.checkpoint
        environment_registry = self.environment_registry
        outcome: SynthesisRegistryRunOutcome | None = None

        for _ in range(max_steps):
            if not pending_registry:
                outcome = SynthesisRegistryRunOutcome.COMPLETED_ALL
                break

            step = await orchestrator.run_next(pending_registry)
            executed_steps += 1

            if step.decision.status != SynthesisSelectionStatus.READY:
                steps.append(SynthesisRegistryStepSummary(decision=step.decision))
                outcome = (
                    SynthesisRegistryRunOutcome.ALL_BACKED_OFF
                    if step.decision.status == SynthesisSelectionStatus.BACKOFF
                    else SynthesisRegistryRunOutcome.COMPLETED_ALL
                )
                break
            if step.draft is None:
                raise RuntimeError("synthesis orchestrator returned READY without a draft")

            assert step.decision.db_id is not None
            assert step.decision.category is not None
            cross_instance_summary = evaluate_cross_instance_draft(step.draft)
            if not cross_instance_summary.passed:
                generated_drafts += 1
                quality_rejected_envs += 1
                cross_instance_rejected_envs += 1
                quality_rejected_env_ids.append(step.draft.environment.env_id)
                generated_env_ids.append(step.draft.environment.env_id)
                steps.append(
                    SynthesisRegistryStepSummary(
                        decision=step.decision,
                        draft_env_id=step.draft.environment.env_id,
                        draft_created_at=step.draft.created_at,
                        quality_gate_status="reject_cross_instance",
                        cross_instance_error_codes=list(cross_instance_summary.error_codes),
                    )
                )
                continue
            from rl_task_foundry.pipeline.environment_orchestrator import (
                evaluate_rollout_summary,
            )

            rollout_summary = await environment_orchestrator.run_draft(step.draft)
            quality_gate_summary = evaluate_rollout_summary(self.base_config, rollout_summary)
            generated_drafts += 1
            generated_env_ids.append(step.draft.environment.env_id)
            if quality_gate_summary.status.value != "accept":
                quality_rejected_envs += 1
                quality_rejected_env_ids.append(step.draft.environment.env_id)
                steps.append(
                    SynthesisRegistryStepSummary(
                        decision=step.decision,
                        draft_env_id=step.draft.environment.env_id,
                        draft_created_at=step.draft.created_at,
                        quality_gate_status=quality_gate_summary.status.value,
                        quality_gate_pass_rate=quality_gate_summary.pass_rate,
                        quality_gate_ci_low=quality_gate_summary.ci_lower,
                        quality_gate_ci_high=quality_gate_summary.ci_upper,
                    )
                )
                continue

            quality_accepted_envs += 1
            accepted_draft = accepted_draft_with_quality_metrics(
                step.draft,
                quality_gate_summary=quality_gate_summary,
            )
            commit_result = environment_registry.commit_draft(accepted_draft)
            steps.append(
                SynthesisRegistryStepSummary(
                    decision=step.decision,
                    draft_env_id=accepted_draft.environment.env_id,
                    draft_created_at=accepted_draft.created_at,
                    quality_gate_status=quality_gate_summary.status.value,
                    quality_gate_pass_rate=quality_gate_summary.pass_rate,
                    quality_gate_ci_low=quality_gate_summary.ci_lower,
                    quality_gate_ci_high=quality_gate_summary.ci_upper,
                    registry_status=commit_result.status,
                    registry_env_id=commit_result.env_id,
                )
            )
            checkpoint.mark_processed(
                self._checkpoint_key(step.decision.db_id, step.decision.category),
                namespace=checkpoint_namespace,
                payload={
                    "db_id": step.decision.db_id,
                    "category": step.decision.category.value,
                    "env_id": commit_result.env_id,
                    "created_at": accepted_draft.created_at.isoformat(),
                    "registry_status": commit_result.status.value,
                    "quality_gate_status": quality_gate_summary.status.value,
                    "solver_pass_rate": quality_gate_summary.pass_rate,
                    "solver_ci_low": quality_gate_summary.ci_lower,
                    "solver_ci_high": quality_gate_summary.ci_upper,
                },
            )
            checkpoint.flush()
            processed_pairs_after_run += 1
            if commit_result.status == EnvironmentRegistryCommitStatus.COMMITTED:
                registry_committed_envs += 1
                committed_env_ids.append(commit_result.env_id)
            else:
                registry_duplicate_envs += 1
                duplicate_env_ids.append(commit_result.env_id)
            pending_registry = self._strip_processed_pair(
                pending_registry,
                db_id=step.decision.db_id,
                category=step.decision.category,
            )

        remaining_pairs = total_pairs - processed_pairs_after_run
        if outcome is None:
            outcome = (
                SynthesisRegistryRunOutcome.COMPLETED_ALL
                if remaining_pairs == 0
                else SynthesisRegistryRunOutcome.MAX_STEPS_REACHED
            )
        assert generated_drafts == quality_accepted_envs + quality_rejected_envs
        assert processed_pairs_after_run == initially_processed_pairs + quality_accepted_envs
        assert remaining_pairs == total_pairs - processed_pairs_after_run

        return SynthesisRegistryRunSummary(
            outcome=outcome,
            checkpoint_namespace=checkpoint_namespace,
            requested_steps=max_steps,
            executed_steps=executed_steps,
            total_pairs=total_pairs,
            initially_processed_pairs=initially_processed_pairs,
            processed_pairs_after_run=processed_pairs_after_run,
            generated_drafts=generated_drafts,
            registry_committed_envs=registry_committed_envs,
            registry_duplicate_envs=registry_duplicate_envs,
            remaining_pairs=remaining_pairs,
            quality_accepted_envs=quality_accepted_envs,
            quality_rejected_envs=quality_rejected_envs,
            generated_env_ids=generated_env_ids,
            committed_env_ids=committed_env_ids,
            duplicate_env_ids=duplicate_env_ids,
            quality_rejected_env_ids=quality_rejected_env_ids,
            registry_root_dir=environment_registry.root_dir,
            registry_index_db_path=environment_registry.index_db_path,
            steps=steps,
        )

    async def close(self) -> None:
        await self.orchestrator.close()
        await self.environment_orchestrator.close()

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
        pending_entries: list[SynthesisDbRegistryEntry] = []
        already_processed = 0
        for entry in registry:
            pending_categories: list[CategoryTaxonomy] = []
            for category in entry.categories:
                if self.checkpoint.is_processed(
                    self._checkpoint_key(entry.db_id, category),
                    namespace=checkpoint_namespace,
                ):
                    already_processed += 1
                else:
                    pending_categories.append(category)
            if pending_categories:
                pending_entries.append(replace(entry, categories=pending_categories))
        return pending_entries, already_processed

    def _strip_processed_pair(
        self,
        registry: list[SynthesisDbRegistryEntry],
        *,
        db_id: str,
        category: CategoryTaxonomy,
    ) -> list[SynthesisDbRegistryEntry]:
        stripped: list[SynthesisDbRegistryEntry] = []
        for entry in registry:
            if entry.db_id != db_id:
                stripped.append(entry)
                continue
            remaining_categories = [item for item in entry.categories if item != category]
            if remaining_categories:
                stripped.append(replace(entry, categories=remaining_categories))
        return stripped

    @staticmethod
    def _checkpoint_key(db_id: str, category: CategoryTaxonomy) -> str:
        return f"{db_id}:{category.value}"


def load_synthesis_registry(path: Path) -> list[SynthesisDbRegistryEntry]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        payload = json.loads(raw)
        entries = [_parse_registry_item(item) for item in payload]
    else:
        entries = [_parse_registry_item(json.loads(line)) for line in raw.splitlines() if line.strip()]
    _validate_unique_db_ids(entries)
    return entries


def _parse_registry_item(item: dict[str, Any]) -> SynthesisDbRegistryEntry:
    payload = SynthesisRegistryFileEntry.model_validate(item)
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
