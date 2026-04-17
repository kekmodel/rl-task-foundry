"""Tests for tooling.composer.sample.

Unit tests use a recording stub connection so we can assert on SQL text
and parameter bindings. Integration tests run against live sakila.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import asyncpg
import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.infra.db import (
    _apply_session_settings,
    solver_session_settings,
)
from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.tooling.common import snapshot_from_graph
from rl_task_foundry.tooling.composer import ComposerSession, sample
from rl_task_foundry.tooling.composer._session import _Row


class _RecordingConnection:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows: list[dict[str, object]] = rows or []
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, sql: str, *args: object) -> Sequence[_Row]:
        self.calls.append((sql, args))
        return cast(Sequence[_Row], list(self.rows))

    async def fetchrow(self, sql: str, *args: object):  # pragma: no cover
        raise AssertionError("fetchrow must not be used by sample")

    async def fetchval(self, sql: str, *args: object):  # pragma: no cover
        raise AssertionError("fetchval must not be used by sample")


def _column(
    table: str,
    name: str,
    data_type: str = "integer",
    *,
    is_primary_key: bool = False,
    is_foreign_key: bool = False,
) -> ColumnProfile:
    return ColumnProfile(
        schema_name="public",
        table_name=table,
        column_name=name,
        data_type=data_type,
        ordinal_position=1,
        is_nullable=False,
        visibility="user_visible",
        is_primary_key=is_primary_key,
        is_foreign_key=is_foreign_key,
    )


def _snapshot():
    customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[
            _column("customer", "customer_id", is_primary_key=True),
            _column("customer", "store_id"),
            _column("customer", "active"),
            _column("customer", "first_name", data_type="text"),
        ],
        primary_key=("customer_id",),
    )
    rental = TableProfile(
        schema_name="public",
        table_name="rental",
        columns=[
            _column("rental", "rental_id", is_primary_key=True),
            _column("rental", "customer_id", is_foreign_key=True),
            _column("rental", "rental_date", data_type="timestamp"),
        ],
        primary_key=("rental_id",),
    )
    edge = ForeignKeyEdge(
        constraint_name="rental_customer_fk",
        source_schema="public",
        source_table="rental",
        source_columns=("customer_id",),
        target_schema="public",
        target_table="customer",
        target_columns=("customer_id",),
    )
    return snapshot_from_graph(
        SchemaGraph(tables=[customer, rental], edges=[edge])
    )


def _stub_session(
    rows: list[dict[str, object]] | None = None,
) -> tuple[ComposerSession, _RecordingConnection]:
    conn = _RecordingConnection(rows=rows)
    session = ComposerSession(snapshot=_snapshot(), connection=conn)
    return session, conn


# ---------- SQL shape ----------


@pytest.mark.asyncio
async def test_sample_without_seed_orders_by_primary_key_ascending():
    session, conn = _stub_session()
    await sample(session, table="customer", n=3)
    sql, params = conn.calls[0]
    assert "SELECT t.\"customer_id\"" in sql
    assert "FROM \"public\".\"customer\"" in sql
    assert "ORDER BY t.\"customer_id\" ASC" in sql
    assert sql.endswith("LIMIT 3")
    assert params == ()


@pytest.mark.asyncio
async def test_sample_with_seed_orders_by_md5_of_pk_plus_seed():
    session, conn = _stub_session()
    await sample(session, table="customer", n=4, seed=42)
    sql, params = conn.calls[0]
    assert "md5((t.\"customer_id\")::text || $1)" in sql
    assert params == ("42",)
    assert sql.endswith("LIMIT 4")


@pytest.mark.asyncio
async def test_sample_with_predicate_binds_params_before_seed():
    session, conn = _stub_session()
    await sample(
        session,
        table="customer",
        n=5,
        seed=7,
        predicate=[
            {"column": "store_id", "op": "eq", "value": 1},
            {"column": "active", "op": "eq", "value": 1},
        ],
    )
    sql, params = conn.calls[0]
    assert "WHERE t.\"store_id\" = $1 AND t.\"active\" = $2" in sql
    assert "md5((t.\"customer_id\")::text || $3)" in sql
    assert params == (1, 1, "7")


@pytest.mark.asyncio
async def test_sample_predicate_with_in_uses_array_cast():
    session, conn = _stub_session()
    await sample(
        session,
        table="customer",
        n=2,
        predicate=[
            {"column": "store_id", "op": "in", "value": [1, 2]},
        ],
    )
    sql, params = conn.calls[0]
    assert "t.\"store_id\" = ANY($1::int4[])" in sql
    assert params == ([1, 2],)


@pytest.mark.asyncio
async def test_sample_predicate_like_preserves_pattern():
    session, conn = _stub_session()
    await sample(
        session,
        table="customer",
        n=2,
        predicate=[
            {"column": "first_name", "op": "like", "value": "A%"},
        ],
    )
    sql, params = conn.calls[0]
    assert "t.\"first_name\" ILIKE $1" in sql
    assert params == ("A%",)


# ---------- validation ----------


@pytest.mark.asyncio
async def test_sample_rejects_non_positive_n():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await sample(session, table="customer", n=0)
    with pytest.raises(ValueError):
        await sample(session, table="customer", n=-1)


@pytest.mark.asyncio
async def test_sample_rejects_unknown_table():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await sample(session, table="unknown", n=3)


@pytest.mark.asyncio
async def test_sample_rejects_unknown_predicate_column():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await sample(
            session,
            table="customer",
            n=3,
            predicate=[{"column": "nope", "op": "eq", "value": 1}],
        )


@pytest.mark.asyncio
async def test_sample_rejects_unsupported_op():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await sample(
            session,
            table="customer",
            n=3,
            predicate=[{"column": "store_id", "op": "regex", "value": "."}],
        )


@pytest.mark.asyncio
async def test_sample_rejects_non_list_predicate():
    session, _ = _stub_session()
    with pytest.raises(TypeError):
        await sample(
            session,
            table="customer",
            n=3,
            predicate={"column": "store_id", "op": "eq", "value": 1},  # type: ignore[arg-type]
        )


# ---------- integration against sakila ----------


async def _live_session() -> tuple[ComposerSession, asyncpg.Connection]:
    config = load_config(Path("rl_task_foundry.yaml"))
    introspector = PostgresSchemaIntrospector(
        database=config.database,
        default_visibility=config.privacy.default_visibility,
        visibility_overrides=config.privacy.visibility_overrides,
    )
    graph = await introspector.introspect()
    snap = snapshot_from_graph(graph)
    conn = await asyncpg.connect(config.database.dsn)
    await _apply_session_settings(conn, solver_session_settings(config.database))
    return ComposerSession(snapshot=snap, connection=conn), conn


@pytest.mark.asyncio
async def test_sample_runs_against_live_sakila_customer_table():
    session, conn = await _live_session()
    try:
        rows = await sample(session, table="customer", n=3)
        assert len(rows) == 3
        assert all("customer_id" in row for row in rows)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_sample_seed_is_reproducible_across_calls():
    session, conn = await _live_session()
    try:
        first = await sample(session, table="customer", n=4, seed=123)
        second = await sample(session, table="customer", n=4, seed=123)
        assert [row["customer_id"] for row in first] == [
            row["customer_id"] for row in second
        ]
        different = await sample(session, table="customer", n=4, seed=124)
        assert [row["customer_id"] for row in first] != [
            row["customer_id"] for row in different
        ]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_sample_honors_predicate_against_sakila():
    session, conn = await _live_session()
    try:
        rows = await sample(
            session,
            table="customer",
            n=3,
            predicate=[{"column": "store_id", "op": "eq", "value": 1}],
        )
        assert len(rows) == 3
        assert all(row["store_id"] == 1 for row in rows)
    finally:
        await conn.close()
