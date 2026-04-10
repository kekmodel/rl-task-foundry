"""Integration test: all core infrastructure components work together."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rlvr_synth.config import load_config
from rlvr_synth.events import EventBus, TaskAccepted, BudgetUpdate
from rlvr_synth.checkpoint import RunCheckpoint
from rlvr_synth.budget import BudgetTracker
from rlvr_synth.privacy import PiiDetector, redact_dict
from rlvr_synth.abstraction import ToolDef, ToolResult


@pytest.mark.asyncio
async def test_core_components_integrate(tmp_path: Path) -> None:
    """All core components can be instantiated and interact."""
    # 1. Event bus
    bus = EventBus()
    received: list = []
    bus.subscribe(lambda e: received.append(e))  # type: ignore

    # 2. Checkpoint
    cp = RunCheckpoint(output_dir=tmp_path, run_id="integration_test")
    assert not cp.is_processed(1)

    # 3. Budget
    bt = BudgetTracker(mode="hybrid", max_api_usd=10.0, max_gpu_hours=0.5)
    reservation = bt.reserve(api_usd=2.0, gpu_hours=0.1)

    # 4. Privacy
    detector = PiiDetector(patterns=["email", "phone"])
    pii = detector.detect(["id", "name", "email", "phone_number"])
    assert pii == {"email", "phone_number"}

    data = {"id": 1, "name": "Kim", "email": "kim@test.com", "phone_number": "010-1234"}
    redacted = redact_dict(data, pii)
    assert redacted["name"] == "Kim"
    assert redacted["email"] == "***REDACTED***"

    # 5. Tool definition
    tool = ToolDef(name="test_tool", description="test", parameters={"x": {"type": "int"}})
    assert tool.to_api_schema()["name"] == "test_tool"

    # 6. Simulate pipeline flow
    await bus.publish(BudgetUpdate(api_cost=1.5, gpu_hours=0.05))
    bt.settle(reservation, actual_api_usd=1.5, actual_gpu_hours=0.05)
    bt.record_processed(accepted=True)
    await cp.mark_processed(1)
    await bus.publish(TaskAccepted(task_id="t1", pass_rate=0.6, iterations=2))

    await asyncio.sleep(0.1)

    assert cp.is_processed(1)
    assert not bt.exceeded()
    assert bt.spent_api_usd == 1.5
    assert len(received) == 2

    await bus.shutdown()
