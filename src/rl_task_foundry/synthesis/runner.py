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
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy, StrictModel
from rl_task_foundry.synthesis.orchestrator import (
    SynthesisDbRegistryEntry,
    SynthesisOrchestrator,
    SynthesisRuntimeHandle,
)
from rl_task_foundry.synthesis.runtime import SynthesisAgentRuntime
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
    remaining_pairs: int
    generated_env_ids: list[str] = field(default_factory=list)
    steps: list[SynthesisRegistryStepSummary] = field(default_factory=list)

    @property
    def last_decision(self) -> SynthesisSchedulerDecision | None:
        if not self.steps:
            return None
        return self.steps[-1].decision


@dataclass(slots=True)
class SynthesisRegistryRunner:
    """Drive the synthesis orchestrator over a registry with checkpoint resume.

    The registry must contain unique `db_id` values. Successful `(db_id, category)`
    pairs are flushed to checkpoint immediately for crash-safe resume semantics.
    """

    base_config: AppConfig
    runtime_factory: Callable[[SynthesisDbRegistryEntry], SynthesisRuntimeHandle] | None = None
    checkpoint: CheckpointStore | None = None
    orchestrator: SynthesisOrchestrator | None = None

    def __post_init__(self) -> None:
        if self.checkpoint is None:
            self.checkpoint = ensure_checkpoint(self.base_config.output.run_db_path)
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
                remaining_pairs=0,
            )

        total_pairs = sum(len(entry.categories) for entry in registry)
        pending_registry, initially_processed_pairs = self._pending_registry(
            registry,
            checkpoint_namespace=checkpoint_namespace,
        )
        steps: list[SynthesisRegistryStepSummary] = []
        generated_env_ids: list[str] = []
        executed_steps = 0
        generated_drafts = 0
        processed_pairs_after_run = initially_processed_pairs
        orchestrator = self.orchestrator
        checkpoint = self.checkpoint
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
            steps.append(
                SynthesisRegistryStepSummary(
                    decision=step.decision,
                    draft_env_id=step.draft.environment.env_id,
                    draft_created_at=step.draft.created_at,
                )
            )
            checkpoint.mark_processed(
                self._checkpoint_key(step.decision.db_id, step.decision.category),
                namespace=checkpoint_namespace,
                payload={
                    "db_id": step.decision.db_id,
                    "category": step.decision.category.value,
                    "env_id": step.draft.environment.env_id,
                    "created_at": step.draft.created_at.isoformat(),
                },
            )
            checkpoint.flush()
            generated_drafts += 1
            processed_pairs_after_run += 1
            generated_env_ids.append(step.draft.environment.env_id)
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
        assert processed_pairs_after_run == initially_processed_pairs + generated_drafts
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
            remaining_pairs=remaining_pairs,
            generated_env_ids=generated_env_ids,
            steps=steps,
        )

    async def close(self) -> None:
        await self.orchestrator.close()

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
