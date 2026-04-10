"""Event bus — non-authoritative telemetry. Bounded queue, drop on lag."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable

from pydantic import BaseModel

log = logging.getLogger(__name__)


class Event(BaseModel):
    """Base event. Subclass for each event type."""
    event_type: str = "base"


# ── Concrete event types ─────────────────────────


class PhaseChanged(Event):
    event_type: str = "phase_changed"
    phase: str
    iteration: int = 0


class TaskCreated(Event):
    event_type: str = "task_created"
    task_id: str
    pk: Any
    depth: int
    conditions: int
    question_preview: str = ""


class SolverSpawned(Event):
    event_type: str = "solver_spawned"
    solver_id: str
    model: str


class SolverToolCall(Event):
    event_type: str = "solver_tool_call"
    solver_id: str
    tool_name: str
    args_summary: str = ""
    result_summary: str = ""
    duration: float = 0.0


class SolverCompleted(Event):
    event_type: str = "solver_completed"
    solver_id: str
    pass_fail: bool | None = None
    duration: float = 0.0
    num_calls: int = 0


class CalibrationResult(Event):
    event_type: str = "calibration_result"
    pass_rate: float
    ci_low: float = 0.0
    ci_high: float = 0.0
    action: str = ""
    difficulty_change: str = ""


class TaskAccepted(Event):
    event_type: str = "task_accepted"
    task_id: str
    pass_rate: float
    iterations: int


class TaskRejected(Event):
    event_type: str = "task_rejected"
    task_id: str
    reason: str


class BudgetUpdate(Event):
    event_type: str = "budget_update"
    api_cost: float = 0.0
    gpu_hours: float = 0.0


class ProviderHealth(Event):
    event_type: str = "provider_health"
    provider_id: str
    status: str
    active: int = 0
    max_slots: int = 0
    error_rate: float = 0.0


class PoolUtilization(Event):
    event_type: str = "pool_utilization"
    solver_active: int = 0
    solver_max: int = 0
    control_active: int = 0
    control_max: int = 0


# ── Bus ──────────────────────────────────────────

Listener = Callable[[Event], Awaitable[None]]


class EventBus:
    """Non-authoritative telemetry bus. Bounded queue, drop on lag."""

    def __init__(self, max_queue_size: int = 1000) -> None:
        self._subscribers: dict[int, tuple[Listener, asyncio.Queue[Event]]] = {}
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._next_id = 0
        self._max_queue_size = max_queue_size

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Subscribe a listener. Returns an unsubscribe callable."""
        sub_id = self._next_id
        self._next_id += 1
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers[sub_id] = (listener, queue)

        async def _drain() -> None:
            while True:
                event = await queue.get()
                try:
                    await listener(event)
                except Exception:
                    log.exception("Subscriber %d error", sub_id)

        task = asyncio.ensure_future(_drain())
        self._tasks[sub_id] = task

        def unsubscribe() -> None:
            self._subscribers.pop(sub_id, None)
            t = self._tasks.pop(sub_id, None)
            if t:
                t.cancel()

        return unsubscribe

    async def publish(self, event: Event) -> None:
        """Publish event. Non-blocking: drops if subscriber queue is full."""
        for sub_id, (_, queue) in list(self._subscribers.items()):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                log.debug("Subscriber %d queue full, dropping event %s", sub_id, event.event_type)

    async def shutdown(self) -> None:
        """Cancel all subscriber drain tasks."""
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        self._subscribers.clear()
