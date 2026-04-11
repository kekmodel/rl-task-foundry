"""Checkpoint-aware multi-db synthesis registry runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import Field

from rl_task_foundry.config.models import AppConfig, DatabaseConfig, DomainConfig
from rl_task_foundry.infra.checkpoint import CheckpointStore, ensure_checkpoint
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy, StrictModel
from rl_task_foundry.synthesis.orchestrator import (
    SynthesisDbRegistryEntry,
    SynthesisOrchestrationStep,
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


@dataclass(slots=True)
class SynthesisRegistryRunSummary:
    checkpoint_namespace: str
    requested_steps: int
    executed_steps: int
    total_pairs: int
    initially_processed_pairs: int
    processed_pairs_after_run: int
    generated_drafts: int
    remaining_pairs: int
    generated_env_ids: list[str] = field(default_factory=list)
    steps: list[SynthesisOrchestrationStep] = field(default_factory=list)

    @property
    def last_decision(self) -> SynthesisSchedulerDecision | None:
        if not self.steps:
            return None
        return self.steps[-1].decision


@dataclass(slots=True)
class SynthesisRegistryRunner:
    """Drive the synthesis orchestrator over a registry with checkpoint resume."""

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

        total_pairs = sum(len(entry.categories) for entry in registry)
        _, initially_processed_pairs = self._pending_registry(
            registry,
            checkpoint_namespace=checkpoint_namespace,
        )
        steps: list[SynthesisOrchestrationStep] = []
        generated_env_ids: list[str] = []
        executed_steps = 0
        generated_drafts = 0
        orchestrator = self.orchestrator
        checkpoint = self.checkpoint

        for _ in range(max_steps):
            pending_registry, _ = self._pending_registry(
                registry,
                checkpoint_namespace=checkpoint_namespace,
            )
            if not pending_registry:
                decision = orchestrator.scheduler.choose_next([])
                steps.append(SynthesisOrchestrationStep(decision=decision, snapshots=[]))
                executed_steps += 1
                break

            step = await orchestrator.run_next(pending_registry)
            steps.append(step)
            executed_steps += 1

            if step.decision.status != SynthesisSelectionStatus.READY:
                break
            if step.draft is None:
                break

            assert step.decision.db_id is not None
            assert step.decision.category is not None
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
            generated_env_ids.append(step.draft.environment.env_id)

        _, processed_pairs_after_run = self._pending_registry(
            registry,
            checkpoint_namespace=checkpoint_namespace,
        )
        remaining_pairs = total_pairs - processed_pairs_after_run

        return SynthesisRegistryRunSummary(
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

    @staticmethod
    def _checkpoint_key(db_id: str, category: CategoryTaxonomy) -> str:
        return f"{db_id}:{category.value}"


def load_synthesis_registry(path: Path) -> list[SynthesisDbRegistryEntry]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        payload = json.loads(raw)
        return [_parse_registry_item(item) for item in payload]
    return [_parse_registry_item(json.loads(line)) for line in raw.splitlines() if line.strip()]


def _parse_registry_item(item: dict[str, Any]) -> SynthesisDbRegistryEntry:
    payload = SynthesisRegistryFileEntry.model_validate(item)
    return payload.to_registry_entry()
