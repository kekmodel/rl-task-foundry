"""Scheduler helpers for synthesis domain/category selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from pydantic import model_validator

from rl_task_foundry.synthesis.contracts import StrictModel


class SynthesisSelectionStatus(StrEnum):
    READY = "ready"
    BACKOFF = "backoff"
    EMPTY = "empty"


class SynthesisDbSnapshot(StrictModel):
    db_id: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_keys(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        # Drop legacy fields silently
        payload.pop("topics", None)
        payload.pop("categories", None)
        payload.pop("topic_status", None)
        payload.pop("category_status", None)
        return payload


class SynthesisSchedulerDecision(StrictModel):
    status: SynthesisSelectionStatus
    db_id: str | None = None
    reason: str = ""
    wait_until: datetime | None = None
    wait_seconds: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_category_key(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        # Drop legacy topic/category fields silently
        payload.pop("topic", None)
        payload.pop("category", None)
        return payload


@dataclass(slots=True)
class SynthesisDomainScheduler:
    """DB-major round-robin scheduler.

    Not thread-safe. Use from a single orchestrator task/thread.
    Each DB is visited in turn via round-robin.
    """

    _last_selected_db_id: str | None = None
    _db_order: list[str] = field(default_factory=list)

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
        current_db_ids = [snapshot.db_id for snapshot in snapshots]
        ordered_db_ids = self._ordered_db_ids(current_db_ids)

        if ordered_db_ids:
            db_id = ordered_db_ids[0]
            self._last_selected_db_id = db_id
            self._db_order = list(current_db_ids)
            return SynthesisSchedulerDecision(
                status=SynthesisSelectionStatus.READY,
                db_id=db_id,
                reason="selected next available db",
            )

        return SynthesisSchedulerDecision(
            status=SynthesisSelectionStatus.EMPTY,
            reason="no selectable databases were provided",
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
