"""Integration test for the atomic calculus vertical slice.

Runs a four-call chain against the live sakila database:
  rows_where → order_by → take → read

Proves the primitives compose correctly end-to-end before the remaining
calculus operators (rows_via, intersect, count, aggregate, group_top)
land next session.
"""

from __future__ import annotations

from pathlib import Path

import asyncpg
import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.infra.db import (
    _apply_session_settings,
    solver_session_settings,
)
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.tooling.atomic import (
    AtomicSession,
    CursorStore,
    order_by,
    read,
    rows_where,
    take,
)
from rl_task_foundry.tooling.common import snapshot_from_graph


async def _build_session():
    config = load_config(Path("rl_task_foundry.yaml"))
    introspector = PostgresSchemaIntrospector(
        database=config.database,
        default_visibility=config.privacy.default_visibility,
        visibility_overrides=config.privacy.visibility_overrides,
    )
    graph = await introspector.introspect()
    snapshot = snapshot_from_graph(graph)

    conn = await asyncpg.connect(config.database.dsn)
    await _apply_session_settings(conn, solver_session_settings(config.database))
    session = AtomicSession(
        snapshot=snapshot,
        connection=conn,
        store=CursorStore(),
    )
    return session, conn


@pytest.mark.asyncio
async def test_vertical_slice_over_sakila():
    session, conn = await _build_session()
    try:
        rentals_of_customer_45 = rows_where(
            session,
            table="rental",
            column="customer_id",
            op="eq",
            value=45,
        )

        ordered = order_by(
            session.store,
            rentals_of_customer_45,
            column="rental_date",
            direction="asc",
        )

        ids = await take(session, cursor=ordered, n=3)
        assert len(ids) == 3
        assert all(isinstance(rid, int) for rid in ids)

        first = await read(
            session,
            table="rental",
            row_id=ids[0],
            columns=["rental_id", "customer_id", "rental_date"],
        )
        assert first["rental_id"] == ids[0]
        assert first["customer_id"] == 45
        assert "rental_date" in first
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_take_rejects_n_below_two_and_above_five():
    session, conn = await _build_session()
    try:
        cursor = rows_where(
            session, table="rental", column="customer_id", op="eq", value=45
        )
        with pytest.raises(ValueError):
            await take(session, cursor=cursor, n=1)
        with pytest.raises(ValueError):
            await take(session, cursor=cursor, n=6)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_rows_where_rejects_unknown_column():
    session, conn = await _build_session()
    try:
        with pytest.raises(KeyError):
            rows_where(
                session,
                table="rental",
                column="nonexistent",
                op="eq",
                value=1,
            )
    finally:
        await conn.close()
