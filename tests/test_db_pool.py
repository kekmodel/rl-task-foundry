"""Tests for the 2-lane DB pool manager."""

from __future__ import annotations

import pytest

from rlvr_synth.db_pool import DbPoolManager


@pytest.mark.asyncio
async def test_pool_manager_creation() -> None:
    """DbPoolManager can be created with config-like params."""
    manager = DbPoolManager(
        connection="postgresql://u:p@localhost:5432/test",
        solver_pool_min=1,
        solver_pool_max=5,
        solver_inflight_max=3,
        control_pool_size=2,
        statement_timeout_ms=3000,
        acquire_timeout_ms=3000,
        read_only=True,
    )
    assert manager.solver_inflight_max == 3
    assert manager.control_pool_size == 2
    assert manager._solver_pool is None  # not connected yet
    assert manager._control_pool is None


@pytest.mark.asyncio
async def test_solver_semaphore_bounds() -> None:
    """Solver semaphore should have correct capacity."""
    manager = DbPoolManager(
        connection="postgresql://u:p@localhost:5432/test",
        solver_pool_min=1,
        solver_pool_max=10,
        solver_inflight_max=5,
        control_pool_size=2,
        statement_timeout_ms=3000,
        acquire_timeout_ms=3000,
        read_only=True,
    )
    # Semaphore is created eagerly
    assert manager._solver_semaphore._value == 5
