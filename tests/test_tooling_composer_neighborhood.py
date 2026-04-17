"""Tests for tooling.composer.neighborhood."""

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
from rl_task_foundry.tooling.composer import ComposerSession, neighborhood
from rl_task_foundry.tooling.composer._session import _Row


class _ScriptedConnection:
    """Mock connection that serves canned results in script order.

    Each call pops its next payload from a FIFO queue. fetch() expects
    a list payload; fetchrow() expects a dict payload; fetchval()
    expects a scalar. This keeps the test intent close to the call
    sequence neighborhood() performs.
    """

    def __init__(self) -> None:
        self.fetch_results: list[list[dict[str, object]]] = []
        self.fetchrow_results: list[dict[str, object] | None] = []
        self.fetchval_results: list[object] = []
        self.calls: list[tuple[str, str, tuple[object, ...]]] = []

    async def fetch(self, sql: str, *args: object) -> Sequence[_Row]:
        self.calls.append(("fetch", sql, args))
        payload = self.fetch_results.pop(0)
        return cast(Sequence[_Row], list(payload))

    async def fetchrow(self, sql: str, *args: object):
        self.calls.append(("fetchrow", sql, args))
        payload = self.fetchrow_results.pop(0)
        if payload is None:
            return None
        return cast(_Row, payload)

    async def fetchval(self, sql: str, *args: object):
        self.calls.append(("fetchval", sql, args))
        return self.fetchval_results.pop(0)


def _column(
    table: str,
    name: str,
    data_type: str = "integer",
    *,
    is_primary_key: bool = False,
    is_foreign_key: bool = False,
    is_nullable: bool = False,
) -> ColumnProfile:
    return ColumnProfile(
        schema_name="public",
        table_name=table,
        column_name=name,
        data_type=data_type,
        ordinal_position=1,
        is_nullable=is_nullable,
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
            _column("customer", "store_id", is_foreign_key=True),
            _column("customer", "first_name", data_type="text"),
        ],
        primary_key=("customer_id",),
    )
    store = TableProfile(
        schema_name="public",
        table_name="store",
        columns=[_column("store", "store_id", is_primary_key=True)],
        primary_key=("store_id",),
    )
    rental = TableProfile(
        schema_name="public",
        table_name="rental",
        columns=[
            _column("rental", "rental_id", is_primary_key=True),
            _column("rental", "customer_id", is_foreign_key=True),
        ],
        primary_key=("rental_id",),
    )
    edges = [
        ForeignKeyEdge(
            constraint_name="customer_store",
            source_schema="public",
            source_table="customer",
            source_columns=("store_id",),
            target_schema="public",
            target_table="store",
            target_columns=("store_id",),
        ),
        ForeignKeyEdge(
            constraint_name="rental_customer",
            source_schema="public",
            source_table="rental",
            source_columns=("customer_id",),
            target_schema="public",
            target_table="customer",
            target_columns=("customer_id",),
        ),
    ]
    return snapshot_from_graph(
        SchemaGraph(tables=[customer, store, rental], edges=edges)
    )


def _stub_session() -> tuple[ComposerSession, _ScriptedConnection]:
    conn = _ScriptedConnection()
    session = ComposerSession(snapshot=_snapshot(), connection=conn)
    return session, conn


# ---------- forward edge (many-to-one) ----------


@pytest.mark.asyncio
async def test_neighborhood_forward_edge_with_pk_target_skips_lookup():
    # customer.store_id -> store (target_column IS store's PK).
    # Forward edge should shortcut to sample_ids=[store_id], total_count=1
    # with no extra SQL beyond the anchor fetch and the reverse-edge pass.
    session, conn = _stub_session()
    conn.fetchrow_results = [
        {"customer_id": 45, "store_id": 1, "first_name": "ALICE"}
    ]
    # reverse edge rental<-customer still needs sample + count
    conn.fetch_results = [[{"id": 10}, {"id": 11}]]
    conn.fetchval_results = [32]
    payload = await neighborhood(
        session, table="customer", row_id=45
    )
    anchor = payload["anchor"]
    assert isinstance(anchor, dict)
    assert anchor["table"] == "customer"
    assert anchor["row_id"] == 45
    edges = payload["edges"]
    assert isinstance(edges, list)
    forward = next(
        edge for edge in edges if isinstance(edge, dict)
        and edge.get("direction") == "forward"
    )
    assert forward["destination_table"] == "store"
    assert forward["total_count"] == 1
    assert forward["sample_ids"] == [1]


@pytest.mark.asyncio
async def test_neighborhood_reverse_edge_runs_sample_and_count():
    session, conn = _stub_session()
    conn.fetchrow_results = [
        {"customer_id": 45, "store_id": 1, "first_name": "ALICE"}
    ]
    conn.fetch_results = [
        [{"id": 10}, {"id": 11}, {"id": 12}]
    ]
    conn.fetchval_results = [32]
    payload = await neighborhood(
        session, table="customer", row_id=45, max_per_edge=3
    )
    edges = payload["edges"]
    assert isinstance(edges, list)
    reverse = next(
        edge for edge in edges if isinstance(edge, dict)
        and edge.get("direction") == "reverse"
    )
    assert reverse["destination_table"] == "rental"
    assert reverse["sample_ids"] == [10, 11, 12]
    assert reverse["total_count"] == 32
    # Verify the sample SQL actually used LIMIT 3 and ordered by rental_id.
    sample_call = conn.calls[1]  # anchor fetchrow first, then reverse sample
    assert sample_call[0] == "fetch"
    assert "ORDER BY dst.\"rental_id\" ASC" in sample_call[1]
    assert sample_call[1].rstrip().endswith("LIMIT 3")


@pytest.mark.asyncio
async def test_neighborhood_null_fk_reports_zero_count_for_forward_edge():
    # If customer.store_id is NULL, the forward edge should report count 0
    # without executing a target lookup.
    session, conn = _stub_session()
    conn.fetchrow_results = [
        {"customer_id": 45, "store_id": None, "first_name": "ALICE"}
    ]
    conn.fetch_results = [[]]
    conn.fetchval_results = [0]
    payload = await neighborhood(session, table="customer", row_id=45)
    edges = payload["edges"]
    assert isinstance(edges, list)
    forward = next(
        edge for edge in edges if isinstance(edge, dict)
        and edge.get("direction") == "forward"
    )
    assert forward["total_count"] == 0
    assert forward["sample_ids"] == []


# ---------- validation ----------


@pytest.mark.asyncio
async def test_neighborhood_raises_for_unknown_row_id():
    session, conn = _stub_session()
    conn.fetchrow_results = [None]
    with pytest.raises(LookupError):
        await neighborhood(session, table="customer", row_id=9999)


@pytest.mark.asyncio
async def test_neighborhood_rejects_unknown_table():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await neighborhood(session, table="unknown", row_id=1)


@pytest.mark.asyncio
async def test_neighborhood_rejects_depth_other_than_one():
    session, _ = _stub_session()
    with pytest.raises(NotImplementedError):
        await neighborhood(
            session, table="customer", row_id=45, depth=2
        )


@pytest.mark.asyncio
async def test_neighborhood_rejects_non_positive_max_per_edge():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await neighborhood(
            session, table="customer", row_id=45, max_per_edge=0
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
async def test_neighborhood_against_sakila_customer_45():
    session, conn = await _live_session()
    try:
        payload = await neighborhood(
            session, table="customer", row_id=45, max_per_edge=3
        )
        anchor = payload["anchor"]
        assert isinstance(anchor, dict)
        attrs = anchor["attributes"]
        assert isinstance(attrs, dict)
        assert attrs["customer_id"] == 45
        edges = payload["edges"]
        assert isinstance(edges, list)
        rental_edge = next(
            edge
            for edge in edges
            if isinstance(edge, dict)
            and edge.get("destination_table") == "rental"
        )
        assert isinstance(rental_edge["total_count"], int)
        assert rental_edge["total_count"] > 0
        sample_ids = rental_edge["sample_ids"]
        assert isinstance(sample_ids, list)
        assert len(sample_ids) <= 3
    finally:
        await conn.close()
