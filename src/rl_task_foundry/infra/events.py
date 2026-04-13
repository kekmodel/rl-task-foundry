"""Typed event bus."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable

EventListener = Callable[["Event"], Awaitable[None]]


@dataclass(slots=True)
class Event:
    event_type: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _Subscriber:
    listener: EventListener
    queue: asyncio.Queue[Event]
    worker_task: asyncio.Task[None] | None = None
    dropped_count: int = 0


class EventBus:
    """Async event bus with bounded per-subscriber queues and drop-on-lag semantics."""

    def __init__(self, *, max_queue_size: int) -> None:
        self.max_queue_size = max_queue_size
        self._listeners: dict[str, list[_Subscriber]] = defaultdict(list)

    def subscribe(self, event_type: str, listener: EventListener) -> None:
        self._listeners[event_type].append(
            _Subscriber(listener=listener, queue=asyncio.Queue(maxsize=self.max_queue_size))
        )

    async def _worker(self, subscriber: _Subscriber) -> None:
        while True:
            event = await subscriber.queue.get()
            try:
                await subscriber.listener(event)
            finally:
                subscriber.queue.task_done()

    def _ensure_worker(self, subscriber: _Subscriber) -> None:
        if subscriber.worker_task is None or subscriber.worker_task.done():
            subscriber.worker_task = asyncio.create_task(self._worker(subscriber))

    async def emit(self, event: Event) -> None:
        for subscriber in self._listeners.get(event.event_type, []):
            self._ensure_worker(subscriber)
            try:
                subscriber.queue.put_nowait(event)
            except asyncio.QueueFull:
                subscriber.dropped_count += 1

    def dropped_count(self, event_type: str) -> int:
        return sum(subscriber.dropped_count for subscriber in self._listeners.get(event_type, []))

    async def aclose(self) -> None:
        tasks = [
            subscriber.worker_task
            for subscribers in self._listeners.values()
            for subscriber in subscribers
            if subscriber.worker_task is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
