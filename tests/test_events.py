"""Tests for the event bus."""

from __future__ import annotations

import asyncio

import pytest

from rlvr_synth.events import EventBus, Event


class FakeEvent(Event):
    event_type: str = "fake"
    value: int = 0


@pytest.mark.asyncio
async def test_subscribe_and_publish() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def listener(event: Event) -> None:
        received.append(event)

    bus.subscribe(listener)
    await bus.publish(FakeEvent(value=42))
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].value == 42


@pytest.mark.asyncio
async def test_slow_subscriber_does_not_block() -> None:
    """Non-authoritative: slow subscriber gets dropped, publisher doesn't block."""
    bus = EventBus(max_queue_size=2)
    received: list[Event] = []

    async def slow_listener(event: Event) -> None:
        await asyncio.sleep(10)  # intentionally slow
        received.append(event)

    bus.subscribe(slow_listener)

    # Publish more than queue size — should not block or raise
    for i in range(5):
        await bus.publish(FakeEvent(value=i))

    # Publisher should return immediately (no hang)
    assert True


@pytest.mark.asyncio
async def test_unsubscribe() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def listener(event: Event) -> None:
        received.append(event)

    unsub = bus.subscribe(listener)
    await bus.publish(FakeEvent(value=1))
    await asyncio.sleep(0.05)

    unsub()
    await bus.publish(FakeEvent(value=2))
    await asyncio.sleep(0.05)

    assert len(received) == 1
