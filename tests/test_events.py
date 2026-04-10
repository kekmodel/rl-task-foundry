from __future__ import annotations

import asyncio

import pytest

from rl_task_foundry.infra.events import Event, EventBus


@pytest.mark.asyncio
async def test_event_bus_drops_on_lag_without_blocking_emit():
    bus = EventBus(max_queue_size=1)
    received: list[dict[str, object]] = []
    release = asyncio.Event()

    async def slow_listener(event: Event) -> None:
        received.append(event.payload)
        await release.wait()

    bus.subscribe("task.accepted", slow_listener)

    await bus.emit(Event("task.accepted", {"seq": 1}))
    await bus.emit(Event("task.accepted", {"seq": 2}))
    await bus.emit(Event("task.accepted", {"seq": 3}))

    assert bus.dropped_count("task.accepted") >= 1

    release.set()
    await asyncio.sleep(0)
    await bus.aclose()


@pytest.mark.asyncio
async def test_event_bus_delivers_when_listener_keeps_up():
    bus = EventBus(max_queue_size=4)
    received: list[int] = []

    async def listener(event: Event) -> None:
        received.append(int(event.payload["seq"]))

    bus.subscribe("task.started", listener)

    await bus.emit(Event("task.started", {"seq": 1}))
    await bus.emit(Event("task.started", {"seq": 2}))
    await asyncio.sleep(0)

    assert received == [1, 2]
    await bus.aclose()
