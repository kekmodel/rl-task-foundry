"""Scheduler helpers for synthesis domain/category selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import Field, model_validator

from rl_task_foundry.synthesis.contracts import CategoryTaxonomy, StrictModel
from rl_task_foundry.synthesis.runtime import SynthesisCategoryStatus


class SynthesisSelectionStatus(StrEnum):
    READY = "ready"
    BACKOFF = "backoff"
    EMPTY = "empty"


class SynthesisDbSnapshot(StrictModel):
    db_id: str
    categories: list[CategoryTaxonomy] = Field(min_length=1)
    category_status: dict[CategoryTaxonomy, SynthesisCategoryStatus] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_category_status(self) -> SynthesisDbSnapshot:
        category_set = set(self.categories)
        for category, status in self.category_status.items():
            if category not in category_set:
                raise ValueError("category_status keys must exist in categories")
            if status.category != category:
                raise ValueError("category_status key must match status.category")
            if status.db_id != self.db_id:
                raise ValueError("category_status entries must match snapshot db_id")
        return self


class SynthesisSchedulerDecision(StrictModel):
    status: SynthesisSelectionStatus
    db_id: str | None = None
    category: CategoryTaxonomy | None = None
    reason: str = ""
    wait_until: datetime | None = None
    wait_seconds: float = 0.0


@dataclass(slots=True)
class SynthesisDomainScheduler:
    """DB-major round-robin scheduler with backoff awareness.

    Not thread-safe. Use from a single orchestrator task/thread.
    Each DB is visited in turn, and one category is selected per DB visit.
    DBs with fewer categories will therefore revisit the same category more often
    than DBs with many categories.
    """

    _last_selected_db_id: str | None = None
    _db_order: list[str] = field(default_factory=list)
    _category_cursors: dict[str, int] = field(default_factory=dict)

    def choose_next(
        self,
        snapshots: list[SynthesisDbSnapshot],
        *,
        now: datetime | None = None,
    ) -> SynthesisSchedulerDecision:
        if not snapshots:
            return SynthesisSchedulerDecision(
                status=SynthesisSelectionStatus.EMPTY,
                reason="no database snapshots available",
            )
        observed_at = now or datetime.now(timezone.utc)
        snapshot_by_db_id = {snapshot.db_id: snapshot for snapshot in snapshots}
        current_db_ids = [snapshot.db_id for snapshot in snapshots]
        self._category_cursors = {
            db_id: cursor
            for db_id, cursor in self._category_cursors.items()
            if db_id in snapshot_by_db_id
        }
        ordered_db_ids = self._ordered_db_ids(current_db_ids)
        earliest_wait_until: datetime | None = None

        for db_id in ordered_db_ids:
            snapshot = snapshot_by_db_id[db_id]
            category_count = len(snapshot.categories)
            category_start = self._category_cursors.get(snapshot.db_id, 0) % category_count
            for category_offset in range(category_count):
                category_index = (category_start + category_offset) % category_count
                category = snapshot.categories[category_index]
                status = snapshot.category_status.get(category)
                if status is None or not status.backed_off:
                    self._last_selected_db_id = snapshot.db_id
                    self._db_order = list(current_db_ids)
                    self._category_cursors[snapshot.db_id] = (category_index + 1) % category_count
                    return SynthesisSchedulerDecision(
                        status=SynthesisSelectionStatus.READY,
                        db_id=snapshot.db_id,
                        category=category,
                        reason="selected next available db/category pair",
                    )
                if status.backoff_until is not None and (
                    earliest_wait_until is None or status.backoff_until < earliest_wait_until
                ):
                    earliest_wait_until = status.backoff_until

        if earliest_wait_until is None:
            return SynthesisSchedulerDecision(
                status=SynthesisSelectionStatus.EMPTY,
                reason="no selectable categories were provided",
            )
        return SynthesisSchedulerDecision(
            status=SynthesisSelectionStatus.BACKOFF,
            reason="all candidate categories are currently backed off",
            wait_until=earliest_wait_until,
            wait_seconds=max(0.0, (earliest_wait_until - observed_at).total_seconds()),
        )

    def _ordered_db_ids(self, current_db_ids: list[str]) -> list[str]:
        if not current_db_ids:
            return []
        if self._last_selected_db_id is None or not self._db_order:
            self._db_order = list(current_db_ids)
            return list(current_db_ids)

        current_db_set = set(current_db_ids)
        ordered: list[str] = []
        seen: set[str] = set()
        previous_order = list(self._db_order)
        if self._last_selected_db_id in previous_order:
            start = (previous_order.index(self._last_selected_db_id) + 1) % len(previous_order)
            for offset in range(len(previous_order)):
                db_id = previous_order[(start + offset) % len(previous_order)]
                if db_id in current_db_set and db_id not in seen:
                    ordered.append(db_id)
                    seen.add(db_id)
        for db_id in current_db_ids:
            if db_id not in seen:
                ordered.append(db_id)
                seen.add(db_id)
        self._db_order = list(current_db_ids)
        return ordered
