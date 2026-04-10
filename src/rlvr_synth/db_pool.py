"""2-lane asyncpg pool: solver (high-throughput) + control-plane (low-latency)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import asyncpg

log = logging.getLogger(__name__)


class DbPoolManager:
    """Manages two separate asyncpg pools with semaphore-based backpressure."""

    def __init__(
        self,
        *,
        connection: str,
        solver_pool_min: int,
        solver_pool_max: int,
        solver_inflight_max: int,
        control_pool_size: int,
        statement_timeout_ms: int,
        acquire_timeout_ms: int,
        read_only: bool,
    ) -> None:
        self._dsn = connection
        self._solver_pool_min = solver_pool_min
        self._solver_pool_max = solver_pool_max
        self.solver_inflight_max = solver_inflight_max
        self.control_pool_size = control_pool_size
        self._statement_timeout_ms = statement_timeout_ms
        self._acquire_timeout_s = acquire_timeout_ms / 1000.0
        self._read_only = read_only

        self._solver_pool: asyncpg.Pool | None = None
        self._control_pool: asyncpg.Pool | None = None
        self._solver_semaphore = asyncio.Semaphore(solver_inflight_max)

    async def connect(self) -> None:
        """Create both pools."""
        self._solver_pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._solver_pool_min,
            max_size=self._solver_pool_max,
        )
        self._control_pool = await asyncpg.create_pool(
            self._dsn,
            min_size=1,
            max_size=self.control_pool_size,
        )
        log.info(
            "DB pools connected: solver=%d-%d (sem=%d), control=%d",
            self._solver_pool_min,
            self._solver_pool_max,
            self.solver_inflight_max,
            self.control_pool_size,
        )

    async def close(self) -> None:
        """Close both pools."""
        if self._solver_pool:
            await self._solver_pool.close()
        if self._control_pool:
            await self._control_pool.close()

    async def solver_query(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        """Execute a read-only query on the solver lane with semaphore."""
        assert self._solver_pool is not None, "Call connect() first"
        async with self._solver_semaphore:
            try:
                async with self._solver_pool.acquire(timeout=self._acquire_timeout_s) as conn:
                    await conn.execute(
                        f"SET LOCAL statement_timeout = '{self._statement_timeout_ms}'"
                    )
                    if self._read_only:
                        return await conn.fetch(sql, *args)
                    return await conn.fetch(sql, *args)
            except asyncio.CancelledError:
                log.warning("Solver query cancelled, connection will be terminated")
                raise

    async def control_query(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        """Execute a read-only query on the control-plane lane (no semaphore)."""
        assert self._control_pool is not None, "Call connect() first"
        async with self._control_pool.acquire(timeout=self._acquire_timeout_s) as conn:
            await conn.execute(
                f"SET LOCAL statement_timeout = '{self._statement_timeout_ms}'"
            )
            return await conn.fetch(sql, *args)

    @property
    def solver_pool_utilization(self) -> tuple[int, int]:
        """(active, max) for solver pool."""
        if not self._solver_pool:
            return (0, self._solver_pool_max)
        used = self._solver_pool.get_size() - self._solver_pool.get_idle_size()
        return (used, self._solver_pool_max)

    @property
    def control_pool_utilization(self) -> tuple[int, int]:
        """(active, max) for control pool."""
        if not self._control_pool:
            return (0, self.control_pool_size)
        used = self._control_pool.get_size() - self._control_pool.get_idle_size()
        return (used, self.control_pool_size)
