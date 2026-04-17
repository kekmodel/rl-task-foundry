"""Tests for tooling.composer.query.

Unit tests exercise parsing, validation, and SQL shape with a recording
stub connection. Integration tests hit live sakila to confirm the DSL
actually authors canonical answers (the core composer use case).
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
from rl_task_foundry.tooling.composer import ComposerSession, query
from rl_task_foundry.tooling.composer._session import _Row


class _RecordingConnection:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows: list[dict[str, object]] = rows or []
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, sql: str, *args: object) -> Sequence[_Row]:
        self.calls.append((sql, args))
        return cast(Sequence[_Row], list(self.rows))

    async def fetchrow(self, sql: str, *args: object):  # pragma: no cover
        raise AssertionError("fetchrow must not be used by query")

    async def fetchval(self, sql: str, *args: object):  # pragma: no cover
        raise AssertionError("fetchval must not be used by query")


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
            _column("rental", "inventory_id", is_foreign_key=True),
            _column("rental", "rental_date", data_type="timestamp"),
        ],
        primary_key=("rental_id",),
    )
    inventory = TableProfile(
        schema_name="public",
        table_name="inventory",
        columns=[
            _column("inventory", "inventory_id", is_primary_key=True),
            _column("inventory", "film_id", is_foreign_key=True),
        ],
        primary_key=("inventory_id",),
    )
    film = TableProfile(
        schema_name="public",
        table_name="film",
        columns=[
            _column("film", "film_id", is_primary_key=True),
            _column("film", "title", data_type="text"),
        ],
        primary_key=("film_id",),
    )
    edges = [
        ForeignKeyEdge(
            constraint_name="rental_customer",
            source_schema="public",
            source_table="rental",
            source_columns=("customer_id",),
            target_schema="public",
            target_table="customer",
            target_columns=("customer_id",),
        ),
        ForeignKeyEdge(
            constraint_name="rental_inventory",
            source_schema="public",
            source_table="rental",
            source_columns=("inventory_id",),
            target_schema="public",
            target_table="inventory",
            target_columns=("inventory_id",),
        ),
        ForeignKeyEdge(
            constraint_name="inventory_film",
            source_schema="public",
            source_table="inventory",
            source_columns=("film_id",),
            target_schema="public",
            target_table="film",
            target_columns=("film_id",),
        ),
    ]
    return snapshot_from_graph(
        SchemaGraph(
            tables=[customer, rental, inventory, film],
            edges=edges,
        )
    )


def _stub_session(
    rows: list[dict[str, object]] | None = None,
) -> tuple[ComposerSession, _RecordingConnection]:
    conn = _RecordingConnection(rows=rows)
    session = ComposerSession(snapshot=_snapshot(), connection=conn)
    return session, conn


# ---------- basic SELECT shapes ----------


@pytest.mark.asyncio
async def test_query_from_only_selects_all_snapshot_columns():
    session, conn = _stub_session()
    await query(session, spec={"from": "customer"})
    sql, params = conn.calls[0]
    assert "SELECT t0.\"customer_id\" AS \"customer_id\"" in sql
    assert "t0.\"first_name\" AS \"first_name\"" in sql
    assert "FROM \"public\".\"customer\" AS t0" in sql
    assert params == ()


@pytest.mark.asyncio
async def test_query_select_restricts_columns_to_list():
    session, conn = _stub_session()
    await query(
        session,
        spec={"from": "customer", "select": ["customer_id", "first_name"]},
    )
    sql, _ = conn.calls[0]
    assert "SELECT t0.\"customer_id\" AS \"customer_id\"" in sql
    assert "t0.\"first_name\" AS \"first_name\"" in sql
    assert "store_id" not in sql


@pytest.mark.asyncio
async def test_query_filter_binds_params_and_builds_where():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": "customer",
            "filter": [
                {"column": "store_id", "op": "eq", "value": 1},
                {"column": "customer_id", "op": "in", "value": [1, 2, 3]},
            ],
            "select": ["customer_id"],
        },
    )
    sql, params = conn.calls[0]
    assert "WHERE t0.\"store_id\" = $1 AND t0.\"customer_id\" = ANY($2::int4[])" in sql
    assert params == (1, [1, 2, 3])


@pytest.mark.asyncio
async def test_query_limit_and_sort_emitted_in_order():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": "customer",
            "select": ["customer_id"],
            "sort": [{"column": "customer_id", "direction": "desc"}],
            "limit": 3,
        },
    )
    sql, _ = conn.calls[0]
    assert "ORDER BY t0.\"customer_id\" DESC" in sql
    assert sql.rstrip().endswith("LIMIT 3")


# ---------- join chain ----------


@pytest.mark.asyncio
async def test_query_single_forward_join_moves_target_to_destination():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": "rental",
            "join": [{"via_edge": "rental.customer_id->customer"}],
            "select": ["first_name"],
        },
    )
    sql, _ = conn.calls[0]
    assert (
        "JOIN \"public\".\"customer\" AS t1 "
        "ON t1.\"customer_id\" = t0.\"customer_id\""
    ) in sql
    assert "t1.\"first_name\" AS \"first_name\"" in sql


@pytest.mark.asyncio
async def test_query_multi_join_walks_rental_to_film():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": "rental",
            "join": [
                {"via_edge": "rental.inventory_id->inventory"},
                {"via_edge": "inventory.film_id->film"},
            ],
            "select": ["title"],
        },
    )
    sql, _ = conn.calls[0]
    assert (
        "JOIN \"public\".\"inventory\" AS t1 "
        "ON t1.\"inventory_id\" = t0.\"inventory_id\""
    ) in sql
    assert (
        "JOIN \"public\".\"film\" AS t2 "
        "ON t2.\"film_id\" = t1.\"film_id\""
    ) in sql
    assert "t2.\"title\" AS \"title\"" in sql


@pytest.mark.asyncio
async def test_query_sort_resolves_from_table_column_after_join():
    # Iter13 regression: composer wrote `from=rental, join=->customer,
    # sort=[rental_date]` and the DSL threw KeyError by resolving sort
    # only against the join destination (customer). rental_date lives
    # on rental (t0), so chain resolution must pick t0 here.
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": "rental",
            "join": [{"via_edge": "rental.customer_id->customer"}],
            "select": ["first_name"],
            "sort": [{"column": "rental_date", "direction": "asc"}],
            "limit": 3,
        },
    )
    sql, _ = conn.calls[0]
    assert "ORDER BY t0.\"rental_date\" ASC" in sql
    assert "t1.\"first_name\" AS \"first_name\"" in sql


@pytest.mark.asyncio
async def test_query_select_spans_from_and_joined_tables():
    # Iter13 regression variant: composer's multi-hop select asked for
    # rental_date plus a customer attribute after chaining inventory ->
    # rental -> customer. Under old target-only resolution, rental_date
    # failed with KeyError on customer. Chain resolution routes each
    # column to its owning table.
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": "inventory",
            "join": [
                {"via_edge": "inventory<-rental.inventory_id"},
                {"via_edge": "rental.customer_id->customer"},
            ],
            "select": ["rental_date", "first_name"],
        },
    )
    sql, _ = conn.calls[0]
    assert "t1.\"rental_date\" AS \"rental_date\"" in sql
    assert "t2.\"first_name\" AS \"first_name\"" in sql


@pytest.mark.asyncio
async def test_query_reverse_join_follows_edge_backwards():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": "customer",
            "join": [{"via_edge": "customer<-rental.customer_id"}],
            "select": ["rental_date"],
        },
    )
    sql, _ = conn.calls[0]
    assert (
        "JOIN \"public\".\"rental\" AS t1 "
        "ON t1.\"customer_id\" = t0.\"customer_id\""
    ) in sql


# ---------- aggregate + group_by ----------


@pytest.mark.asyncio
async def test_query_aggregate_count_without_group_by():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": "rental",
            "aggregate": [{"fn": "count", "alias": "total"}],
        },
    )
    sql, _ = conn.calls[0]
    assert "COUNT(*) AS \"total\"" in sql
    assert "GROUP BY" not in sql


@pytest.mark.asyncio
async def test_query_aggregate_with_group_by_emits_group_and_sort():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": "rental",
            "group_by": ["customer_id"],
            "aggregate": [
                {"fn": "count", "alias": "rentals"},
                {"fn": "max", "column": "rental_date", "alias": "last"},
            ],
            "sort": [{"column": "rentals", "direction": "desc"}],
            "limit": 3,
        },
    )
    sql, _ = conn.calls[0]
    assert (
        "SELECT t0.\"customer_id\" AS \"customer_id\", "
        "COUNT(*) AS \"rentals\", "
        "MAX(t0.\"rental_date\") AS \"last\""
    ) in sql
    assert "GROUP BY t0.\"customer_id\"" in sql
    assert "ORDER BY \"rentals\" DESC" in sql
    assert sql.rstrip().endswith("LIMIT 3")


@pytest.mark.asyncio
async def test_query_aggregate_on_joined_table_resolves_column_in_chain():
    session, conn = _stub_session()
    await query(
        session,
        spec={
            "from": "rental",
            "join": [
                {"via_edge": "rental.inventory_id->inventory"},
                {"via_edge": "inventory.film_id->film"},
            ],
            "group_by": ["film_id"],
            "aggregate": [{"fn": "count", "alias": "rentals"}],
            "sort": [{"column": "rentals", "direction": "desc"}],
            "limit": 5,
        },
    )
    sql, _ = conn.calls[0]
    # Chain-order resolution picks the earliest table owning film_id
    # (inventory at t1). The JOIN equality makes t1.film_id / t2.film_id
    # produce identical groupings on inner joins, so the rebind is
    # semantically a no-op; the assertion documents the new rule.
    assert "GROUP BY t1.\"film_id\"" in sql
    assert "COUNT(*) AS \"rentals\"" in sql


# ---------- validation ----------


@pytest.mark.asyncio
async def test_query_rejects_missing_from_table():
    session, _ = _stub_session()
    with pytest.raises(TypeError):
        await query(session, spec={})


@pytest.mark.asyncio
async def test_query_rejects_unknown_table():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await query(session, spec={"from": "nope"})


@pytest.mark.asyncio
async def test_query_rejects_select_and_aggregate_together():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(
            session,
            spec={
                "from": "rental",
                "select": ["rental_id"],
                "aggregate": [{"fn": "count", "alias": "n"}],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_group_by_without_aggregate():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(
            session,
            spec={"from": "rental", "group_by": ["customer_id"]},
        )


@pytest.mark.asyncio
async def test_query_rejects_unknown_select_column():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await query(
            session,
            spec={"from": "customer", "select": ["not_a_column"]},
        )


@pytest.mark.asyncio
async def test_query_rejects_join_edge_not_originating_at_current_table():
    session, _ = _stub_session()
    with pytest.raises(KeyError):
        await query(
            session,
            spec={
                "from": "customer",
                "join": [{"via_edge": "rental.inventory_id->inventory"}],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_unsupported_aggregate_fn():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(
            session,
            spec={
                "from": "rental",
                "aggregate": [{"fn": "median", "column": "rental_id", "alias": "m"}],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_duplicate_aggregate_aliases():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(
            session,
            spec={
                "from": "rental",
                "aggregate": [
                    {"fn": "count", "alias": "n"},
                    {"fn": "max", "column": "rental_date", "alias": "n"},
                ],
            },
        )


@pytest.mark.asyncio
async def test_query_rejects_non_positive_limit():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(session, spec={"from": "customer", "limit": 0})


@pytest.mark.asyncio
async def test_query_rejects_invalid_sort_direction():
    session, _ = _stub_session()
    with pytest.raises(ValueError):
        await query(
            session,
            spec={
                "from": "customer",
                "sort": [{"column": "customer_id", "direction": "up"}],
            },
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
async def test_query_top_films_by_rental_count_against_sakila():
    session, conn = await _live_session()
    try:
        result = await query(
            session,
            spec={
                "from": "rental",
                "join": [
                    {"via_edge": "rental.inventory_id->inventory"},
                    {"via_edge": "inventory.film_id->film"},
                ],
                "group_by": ["title"],
                "aggregate": [{"fn": "count", "alias": "rentals"}],
                "sort": [{"column": "rentals", "direction": "desc"}],
                "limit": 3,
            },
        )
        rows = result["rows"]
        assert isinstance(rows, list)
        assert len(rows) == 3
        counts = [row["rentals"] for row in rows if isinstance(row, dict)]
        assert counts == sorted(counts, reverse=True)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_query_filter_and_select_against_sakila():
    session, conn = await _live_session()
    try:
        result = await query(
            session,
            spec={
                "from": "customer",
                "filter": [
                    {"column": "store_id", "op": "eq", "value": 1},
                ],
                "select": ["customer_id", "first_name"],
                "sort": [{"column": "customer_id", "direction": "asc"}],
                "limit": 5,
            },
        )
        rows = result["rows"]
        assert isinstance(rows, list)
        assert len(rows) == 5
        ids = [row["customer_id"] for row in rows if isinstance(row, dict)]
        assert ids == sorted(cast(list[int], ids))
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_query_aggregate_max_rental_date_against_sakila():
    session, conn = await _live_session()
    try:
        result = await query(
            session,
            spec={
                "from": "rental",
                "aggregate": [
                    {"fn": "count", "alias": "total"},
                    {"fn": "max", "column": "rental_date", "alias": "last"},
                ],
            },
        )
        rows = result["rows"]
        assert isinstance(rows, list)
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, dict)
        assert isinstance(row["total"], int)
        assert row["total"] > 0
    finally:
        await conn.close()
