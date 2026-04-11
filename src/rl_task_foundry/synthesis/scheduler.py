"""Scheduler helpers for synthesis domain/category selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import Field

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


class SynthesisSchedulerDecision(StrictModel):
    status: SynthesisSelectionStatus
    db_id: str | None = None
    category: CategoryTaxonomy | None = None
    reason: str = ""
    wait_until: datetime | None = None
    wait_seconds: float = 0.0


@dataclass(slots=True)
class SynthesisDomainScheduler:
    """Round-robin scheduler over db/category pairs with backoff awareness."""

    _db_cursor: int = 0
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
        db_count = len(snapshots)
        earliest_wait_until: datetime | None = None

        for db_offset in range(db_count):
            db_index = (self._db_cursor + db_offset) % db_count
            snapshot = snapshots[db_index]
            category_count = len(snapshot.categories)
            if category_count == 0:
                continue
            category_start = self._category_cursors.get(snapshot.db_id, 0) % category_count
            for category_offset in range(category_count):
                category_index = (category_start + category_offset) % category_count
                category = snapshot.categories[category_index]
                status = snapshot.category_status.get(category)
                if status is None or not status.backed_off:
                    self._db_cursor = (db_index + 1) % db_count
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
