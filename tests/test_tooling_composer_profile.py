"""Tests for tooling.composer.profile."""

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
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.tooling.common import snapshot_from_graph
from rl_task_foundry.tooling.composer import ComposerSession, profile
from rl_task_foundry.tooling.composer._session import _Row


class _RecordingConnection:
    def __init__(
        self,
        fetchrow_payload: dict[str, object] | None = None,
        fetch_payload: list[dict[str, object]] | None = None,
    ) -> None:
        self.fetchrow_payload = fetchrow_payload
        self.fetch_payload: list[dict[str, object]] = fetch_payload or []
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, sql: str, *args: object) -> Sequence[_Row]:
        self.calls.append((sql, args))
        return cast(Sequence[_Row], list(self.fetch_payload))

    async def fetchrow(self, sql: str, *args: object):
        self.calls.append((sql, args))
        if self.fetchrow_payload is None:
            return None
        return cast(_Row, self.fetchrow_payload)

    async def fetchval(self, sql: str, *args: object):  # pragma: no cover
        raise AssertionError("fetchval must not be used by profile")


def _column(
    table: str,
    name: str,
    data_type: str = "integer",
    *,
    is_primary_key: bool = False,
    visibility: str = "user_visible",
) -> ColumnProfile:
    return ColumnProfile(
        schema_name="public",
        table_name=table,
        column_name=name,
        data_type=data_type,
        ordinal_position=1,
        is_nullable=False,
        visibility=visibility,  # type: ignore[arg-type]
        is_primary_key=is_primary_key,
    )


def _snapshot():
    customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[
            _column("customer", "customer_id", is_primary_key=True),
            _column("customer", "store_id"),
            _column("customer", "first_name", data_type="text"),
            _column("customer", "api_token", data_type="text", visibility="blocked"),
        ],
        primary_key=("customer_id",),
    )
    return snapshot_from_graph(SchemaGraph(tables=[customer], edges=[]))


def _stub_session(
    fetchrow_payload: dict[str, object] | None = None,
    fetch_payload: list[dict[str, object]] | None = None,
) -> tuple[ComposerSession, _RecordingConnection]:
    conn = _RecordingConnection(
        fetchrow_payload=fetchrow_payload, fetch_payload=fetch_payload
    )
    session = ComposerSession(snapshot=_snapshot(), connection=conn)
    return session, conn


# ---------- table-level shape ----------


@pytest.mark.asyncio
async def test_profile_table_emits_per_column_distinct_and_null_counts():
    session, conn = _stub_session(
        fetchrow_payload={
            "row_count": 599,
            "distinct_customer_id": 599,
            "null_customer_id": 0,
            "distinct_store_id": 2,
            "null_store_id": 0,
            "distinct_first_name": 591,
            "null_first_name": 0,
        }
    )
    payload = await profile(session, table="customer")
    assert payload["table"] == "customer"
    assert payload["row_count"] == 599
    columns = payload["columns"]
    assert isinstance(columns, list)
    names = [col["name"] for col in columns if isinstance(col, dict)]
    assert names == ["customer_id", "store_id", "first_name"]
    sql, _ = conn.calls[0]
    assert "COUNT(*) AS row_count" in sql
    assert "api_token" not in sql
    assert "COUNT(DISTINCT t.\"store_id\") AS \"distinct_store_id\"" in sql


@pytest.mark.asyncio
async def test_profile_table_honors_predicate():
    session, conn = _stub_session(
        fetchrow_payload={
            "row_count": 273,
            "distinct_customer_id": 273,
            "null_customer_id": 0,
            "distinct_store_id": 1,
            "null_store_id": 0,
            "distinct_first_name": 271,
            "null_first_name": 0,
        }
    )
    payload = await profile(
        session,
        table="customer",
        predicate=[{"column": "store_id", "op": "eq", "value": 1}],
    )
    sql, params = conn.calls[0]
    assert "WHERE t.\"store_id\" = $1" in sql
    assert params == (1,)
    assert payload["row_count"] == 273


# ---------- column-level shape ----------


@pytest.mark.asyncio
async def test_profile_column_returns_min_max_and_top_k():
    session, conn = _stub_session(
        fetchrow_payload={
            "row_count": 599,
            "distinct_count": 2,
            "null_count": 0,
            "min_value": 1,
            "max_value": 2,
        },
        fetch_payload=[
            {"value": 2, "frequency": 326},
            {"value": 1, "frequency": 273},
        ],
    )
    payload = await profile(session, table="customer", column="store_id")
    assert payload["column"] == "store_id"
    assert payload["distinct_count"] == 2
    assert payload["min"] == 1
    assert payload["max"] == 2
    top_k = payload["top_k"]
    assert isinstance(top_k, list)
    assert [entry["value"] for entry in top_k if isinstance(entry, dict)] == [2, 1]
    aggregate_sql, _ = conn.calls[0]
    top_k_sql, _ = conn.calls[1]
    assert "MIN(t.\"store_id\")" in aggregate_sql
    assert "MAX(t.\"store_id\")" in aggregate_sql
    assert "GROUP BY t.\"store_id\"" in top_k_sql
    assert "ORDER BY frequency DESC, t.\"store_id\" ASC" in top_k_sql
    assert top_k_sql.rstrip().endswith("LIMIT 5")


@pytest.mark.asyncio
async def test_profile_column_omits_min_max_for_unsupported_types():
    # Use a text column (supports MIN/MAX) vs hypothetical binary (not). The
    # snapshot's only non-supported type here would be something like bytea —
    # emulate by injecting a column with that type.
    customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[
            _column("customer", "customer_id", is_primary_key=True),
            _column("customer", "avatar", data_type="bytea"),
        ],
        primary_key=("customer_id",),
    )
    snap = snapshot_from_graph(SchemaGraph(tables=[customer], edges=[]))
    conn = _RecordingConnection(
        fetchrow_payload={
            "row_count": 599,
            "distinct_count": 400,
            "null_count": 10,
        },
        fetch_payload=[],
    )
    session = ComposerSession(snapshot=snap, connection=conn)
    payload = await profile(session, table="customer", column="avatar")
    assert "min" not in payload
    assert "max" not in payload
    aggregate_sql, _ = conn.calls[0]
    assert "MIN(" not in aggregate_sql
    assert "MAX(" not in aggregate_sql


# ---------- validation ----------


@pytest.mark.asyncio
async def test_profile_rejects_unknown_table():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await profile(session, table="unknown")


@pytest.mark.asyncio
async def test_profile_rejects_unknown_column():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await profile(session, table="customer", column="nope")


@pytest.mark.asyncio
async def test_profile_rejects_blocked_non_handle_column():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await profile(session, table="customer", column="api_token")


@pytest.mark.asyncio
async def test_profile_rejects_non_positive_top_k():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await profile(
            session, table="customer", column="store_id", top_k=0
        )


@pytest.mark.asyncio
async def test_profile_column_with_predicate_anchors_top_k_to_same_filter():
    session, conn = _stub_session(
        fetchrow_payload={
            "row_count": 273,
            "distinct_count": 1,
            "null_count": 0,
            "min_value": 1,
            "max_value": 1,
        },
        fetch_payload=[{"value": 1, "frequency": 273}],
    )
    await profile(
        session,
        table="customer",
        column="store_id",
        predicate=[{"column": "store_id", "op": "eq", "value": 1}],
    )
    _, agg_params = conn.calls[0]
    top_k_sql, top_k_params = conn.calls[1]
    assert agg_params == (1,)
    assert top_k_params == (1,)
    assert (
        "WHERE t.\"store_id\" = $1 AND t.\"store_id\" IS NOT NULL"
        in top_k_sql
    )


# ---------- integration ----------


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
async def test_profile_table_against_pagila_customer():
    session, conn = await _live_session()
    try:
        payload = await profile(session, table="customer")
        assert isinstance(payload["row_count"], int)
        assert payload["row_count"] > 0
        columns = payload["columns"]
        assert isinstance(columns, list)
        column_names = {
            col["name"] for col in columns if isinstance(col, dict)
        }
        assert "customer_id" in column_names
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_profile_column_store_id_against_pagila():
    session, conn = await _live_session()
    try:
        payload = await profile(
            session, table="customer", column="store_id"
        )
        assert payload["distinct_count"] == 2
        top_k = payload["top_k"]
        assert isinstance(top_k, list)
        assert len(top_k) == 2
        # should be ordered by frequency desc
        freqs = [
            entry["frequency"]
            for entry in top_k
            if isinstance(entry, dict)
        ]
        assert freqs == sorted(cast(list[int], freqs), reverse=True)
    finally:
        await conn.close()
