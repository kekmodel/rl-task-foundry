"""Tests for tooling.atomic.tool_factory.

Unit tests exercise schema baking and the on_invoke_tool error path with
a stub connection. Integration tests run a full rows_where → order_by →
take → read chain through the FunctionTool handlers against sakila.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
from rl_task_foundry.tooling.atomic import (
    AtomicSession,
    CursorStore,
    build_aggregate_tool,
    build_atomic_tools,
    build_count_tool,
    build_group_top_tool,
    build_intersect_tool,
    build_order_by_tool,
    build_read_tool,
    build_rows_via_tool,
    build_rows_where_tool,
    build_take_tool,
)
from rl_task_foundry.tooling.common import snapshot_from_graph


class _StubConnection:
    async def fetch(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("connection must not be used in plan-only tests")

    async def fetchrow(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("connection must not be used in plan-only tests")


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


def _stub_session() -> AtomicSession:
    return AtomicSession(
        snapshot=_snapshot(),
        connection=_StubConnection(),
        store=CursorStore(),
    )


async def _invoke(tool: Any, payload: dict[str, Any]) -> dict[str, Any]:
    result = await tool.on_invoke_tool(None, json.dumps(payload))
    return json.loads(result)


# ---------- schema baking ----------


def test_rows_where_schema_enumerates_tables_columns_and_ops():
    tool = build_rows_where_tool(_stub_session())
    schema = tool.params_json_schema
    assert tool.name == "rows_where"
    assert set(schema["properties"]["table"]["enum"]) == {"customer", "rental"}
    assert "customer_id" in schema["properties"]["column"]["enum"]
    assert "rental_date" in schema["properties"]["column"]["enum"]
    assert set(schema["properties"]["op"]["enum"]) == {
        "eq",
        "in",
        "gt",
        "gte",
        "lt",
        "lte",
        "like",
    }
    assert schema["additionalProperties"] is False


def test_rows_via_schema_enumerates_forward_and_reverse_edges():
    tool = build_rows_via_tool(_stub_session())
    labels = set(tool.params_json_schema["properties"]["edge_label"]["enum"])
    assert "rental.customer_id->customer" in labels
    assert "customer<-rental.customer_id" in labels


def test_order_by_schema_restricts_direction_to_asc_desc():
    tool = build_order_by_tool(_stub_session())
    schema = tool.params_json_schema
    assert schema["properties"]["direction"]["enum"] == ["asc", "desc"]


def test_take_schema_enforces_n_between_two_and_five():
    tool = build_take_tool(_stub_session())
    schema = tool.params_json_schema
    assert schema["properties"]["n"]["minimum"] == 2
    assert schema["properties"]["n"]["maximum"] == 5


def test_group_top_schema_lists_five_fns_including_count():
    tool = build_group_top_tool(_stub_session())
    fns = set(tool.params_json_schema["properties"]["fn"]["enum"])
    assert fns == {"count", "sum", "avg", "min", "max"}


def test_aggregate_schema_lists_four_fns():
    tool = build_aggregate_tool(_stub_session())
    fns = set(tool.params_json_schema["properties"]["fn"]["enum"])
    assert fns == {"sum", "avg", "min", "max"}


def test_build_atomic_tools_returns_nine_tools_in_calculus_order():
    tools = build_atomic_tools(_stub_session())
    assert [tool.name for tool in tools] == [
        "rows_where",
        "rows_via",
        "intersect",
        "order_by",
        "take",
        "count",
        "aggregate",
        "group_top",
        "read",
    ]


# ---------- invoke handlers (plan-only) ----------


@pytest.mark.asyncio
async def test_rows_where_invoke_returns_cursor_payload():
    session = _stub_session()
    tool = build_rows_where_tool(session)
    response = await _invoke(
        tool,
        {"table": "customer", "column": "store_id", "op": "eq", "value": 1},
    )
    assert response["target_table"] == "customer"
    assert response["cursor_id"].startswith("c_")


@pytest.mark.asyncio
async def test_rows_where_invoke_surfaces_unknown_column_as_error():
    tool = build_rows_where_tool(_stub_session())
    response = await _invoke(
        tool,
        {
            "table": "rental",
            "column": "does_not_exist",
            "op": "eq",
            "value": 1,
        },
    )
    assert response["error_type"] == "KeyError"


@pytest.mark.asyncio
async def test_rows_via_invoke_threads_cursor_through_edge():
    session = _stub_session()
    where_tool = build_rows_where_tool(session)
    via_tool = build_rows_via_tool(session)
    origin = await _invoke(
        where_tool,
        {
            "table": "rental",
            "column": "customer_id",
            "op": "eq",
            "value": 45,
        },
    )
    projected = await _invoke(
        via_tool,
        {
            "cursor": origin["cursor_id"],
            "edge_label": "rental.customer_id->customer",
        },
    )
    assert projected["target_table"] == "customer"


@pytest.mark.asyncio
async def test_intersect_invoke_rejects_mismatched_tables():
    session = _stub_session()
    where_tool = build_rows_where_tool(session)
    intersect_tool = build_intersect_tool(session)
    customers = await _invoke(
        where_tool,
        {"table": "customer", "column": "store_id", "op": "eq", "value": 1},
    )
    rentals = await _invoke(
        where_tool,
        {
            "table": "rental",
            "column": "customer_id",
            "op": "eq",
            "value": 45,
        },
    )
    response = await _invoke(
        intersect_tool,
        {"left": customers["cursor_id"], "right": rentals["cursor_id"]},
    )
    assert response["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_invalid_json_input_is_surfaced_as_error():
    tool = build_rows_where_tool(_stub_session())
    result = await tool.on_invoke_tool(None, "{not-json")
    parsed = json.loads(result)
    assert parsed["error_type"] == "JSONDecodeError"


# ---------- integration against live sakila ----------


async def _live_session():
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
        snapshot=snapshot, connection=conn, store=CursorStore()
    )
    return session, conn


@pytest.mark.asyncio
async def test_end_to_end_tool_chain_against_sakila():
    session, conn = await _live_session()
    try:
        rows_where = build_rows_where_tool(session)
        order_by = build_order_by_tool(session)
        take = build_take_tool(session)
        read = build_read_tool(session)

        cursor = await _invoke(
            rows_where,
            {
                "table": "rental",
                "column": "customer_id",
                "op": "eq",
                "value": 45,
            },
        )
        ordered = await _invoke(
            order_by,
            {
                "cursor": cursor["cursor_id"],
                "column": "rental_date",
                "direction": "asc",
            },
        )
        taken = await _invoke(
            take, {"cursor": ordered["cursor_id"], "n": 3}
        )
        assert len(taken["row_ids"]) == 3

        first = await _invoke(
            read,
            {
                "table": "rental",
                "row_id": taken["row_ids"][0],
                "columns": ["rental_id", "customer_id"],
            },
        )
        assert first["row"]["customer_id"] == 45
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_group_top_count_tool_returns_descending_counts():
    from datetime import datetime

    session, conn = await _live_session()
    try:
        rows_where = build_rows_where_tool(session)
        group_top = build_group_top_tool(session)
        cursor = await _invoke(
            rows_where,
            {
                "table": "rental",
                "column": "rental_date",
                "op": "gt",
                "value": datetime(2005, 1, 1).isoformat(),
            },
        )
        tops = await _invoke(
            group_top,
            {
                "cursor": cursor["cursor_id"],
                "group_column": "customer_id",
                "fn": "count",
                "n": 3,
            },
        )
        counts = [entry["agg_value"] for entry in tops["tops"]]
        assert len(counts) == 3
        assert counts == sorted(counts, reverse=True)
    finally:
        await conn.close()
