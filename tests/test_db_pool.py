from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from rl_task_foundry.config.models import DatabaseConfig
from rl_task_foundry.infra import db as db_module
from rl_task_foundry.infra.db import DatabasePools


class _FakeConn:
    def __init__(self) -> None:
        self.statements: list[str] = []

    async def execute(self, statement: str) -> None:
        self.statements.append(statement)


class _FakePool:
    def __init__(self) -> None:
        self.conn = _FakeConn()
        self.closed = False

    @asynccontextmanager
    async def acquire(self):
        yield self.conn

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_database_pools_create_solver_and_control_lanes(monkeypatch):
    created: list[dict[str, object]] = []
    pools: list[_FakePool] = []

    async def fake_create_pool(*, dsn, min_size, max_size, init):
        pool = _FakePool()
        pools.append(pool)
        created.append(
            {
                "dsn": dsn,
                "min_size": min_size,
                "max_size": max_size,
            }
        )
        await init(pool.conn)
        return pool

    monkeypatch.setattr(db_module.asyncpg, "create_pool", fake_create_pool)

    config = DatabaseConfig(
        dsn="postgresql://sakila:sakila@127.0.0.1:5433/sakila",
        readonly_role="rlvr_reader",
        solver_pool_size=12,
        control_pool_size=3,
    )

    pools_obj = await DatabasePools.create(config)

    assert len(created) == 2
    assert created[0]["max_size"] == 12
    assert created[1]["max_size"] == 3
    assert "SET default_transaction_read_only = on" in pools[0].conn.statements
    assert "SET ROLE rlvr_reader" in pools[0].conn.statements
    assert "SET default_transaction_read_only = on" not in pools[1].conn.statements

    async with pools_obj.solver_connection() as solver_conn:
        assert solver_conn is pools[0].conn
    async with pools_obj.control_connection() as control_conn:
        assert control_conn is pools[1].conn

    await pools_obj.close()
    assert pools[0].closed is True
    assert pools[1].closed is True
