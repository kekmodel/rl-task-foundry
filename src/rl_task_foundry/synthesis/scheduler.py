"""Scheduler helpers for synthesis domain/category selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import Field, model_validator

from rl_task_foundry.synthesis.contracts import StrictModel, normalize_topic
from rl_task_foundry.synthesis.runtime import SynthesisCategoryStatus


class SynthesisSelectionStatus(StrEnum):
    READY = "ready"
    BACKOFF = "backoff"
    EMPTY = "empty"


class SynthesisDbSnapshot(StrictModel):
    db_id: str
    topics: list[str] = Field(min_length=1)
    topic_status: dict[str, SynthesisCategoryStatus] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_keys(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "topics" not in payload and "categories" in payload:
            payload["topics"] = payload.pop("categories")
        if "topic_status" not in payload and "category_status" in payload:
            payload["topic_status"] = payload.pop("category_status")
        return payload

    @model_validator(mode="after")
    def _validate_category_status(self) -> SynthesisDbSnapshot:
        self.topics = [normalize_topic(topic) for topic in self.topics]
        topic_set = set(self.topics)
        for topic, status in self.topic_status.items():
            if topic not in topic_set:
                raise ValueError("topic_status keys must exist in topics")
            if status.topic != topic:
                raise ValueError("topic_status key must match status.topic")
            if status.db_id != self.db_id:
                raise ValueError("topic_status entries must match snapshot db_id")
        return self

    @property
    def categories(self) -> list[str]:
        return list(self.topics)

    @property
    def category_status(self) -> dict[str, SynthesisCategoryStatus]:
        return dict(self.topic_status)


class SynthesisSchedulerDecision(StrictModel):
    status: SynthesisSelectionStatus
    db_id: str | None = None
    topic: str | None = None
    reason: str = ""
    wait_until: datetime | None = None
    wait_seconds: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_category_key(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "topic" not in payload and "category" in payload:
            payload["topic"] = payload.pop("category")
        return payload

    @property
    def category(self) -> str | None:
        return self.topic


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
    _topic_orders: dict[str, list[str]] = field(default_factory=dict)
    _last_selected_topics: dict[str, str] = field(default_factory=dict)

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
        self._topic_orders = {
            db_id: order
            for db_id, order in self._topic_orders.items()
            if db_id in snapshot_by_db_id
        }
        self._last_selected_topics = {
            db_id: topic
            for db_id, topic in self._last_selected_topics.items()
            if db_id in snapshot_by_db_id
        }
        ordered_db_ids = self._ordered_db_ids(current_db_ids)
        earliest_wait_until: datetime | None = None

        for db_id in ordered_db_ids:
            snapshot = snapshot_by_db_id[db_id]
            ordered_topics = self._ordered_topics(snapshot.db_id, snapshot.topics)
            topic_count = len(ordered_topics)
            last_selected_topic = self._last_selected_topics.get(snapshot.db_id)
            topic_start = 0
            if last_selected_topic in ordered_topics:
                topic_start = (ordered_topics.index(last_selected_topic) + 1) % topic_count
            current_topic_set = set(snapshot.topics)
            for topic_offset in range(topic_count):
                topic = ordered_topics[(topic_start + topic_offset) % topic_count]
                if topic not in current_topic_set:
                    continue
                status = snapshot.topic_status.get(topic)
                if status is None or not status.backed_off:
                    self._last_selected_db_id = snapshot.db_id
                    self._db_order = list(current_db_ids)
                    self._last_selected_topics[snapshot.db_id] = topic
                    return SynthesisSchedulerDecision(
                        status=SynthesisSelectionStatus.READY,
                        db_id=snapshot.db_id,
                        topic=topic,
                        reason="selected next available db/topic pair",
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
            reason="all candidate topics are currently backed off",
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

    def _ordered_topics(
        self,
        db_id: str,
        current_topics: list[str],
    ) -> list[str]:
        previous_order = self._topic_orders.get(db_id, [])
        ordered = list(previous_order)
        for topic in current_topics:
            if topic not in ordered:
                ordered.append(topic)
        self._topic_orders[db_id] = ordered
        return ordered
