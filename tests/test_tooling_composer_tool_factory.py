"""Tests for tooling.composer.tool_factory.

Unit tests exercise JSON schema enum baking and the on_invoke_tool
error path with a stub connection. Integration tests run against live
sakila — end-to-end schema_map, sample, and query through the SDK
handler.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
from rl_task_foundry.tooling.composer import (
    ComposerSession,
    build_composer_tools,
    build_neighborhood_tool,
    build_profile_tool,
    build_query_tool,
    build_sample_tool,
    build_schema_map_tool,
)
from rl_task_foundry.tooling.composer._session import _Row

if TYPE_CHECKING:
    from agents import FunctionTool


class _StubConnection:
    async def fetch(self, sql: str, *args: object) -> Sequence[_Row]:
        return cast(Sequence[_Row], [])

    async def fetchrow(self, sql: str, *args: object):
        return None

    async def fetchval(self, sql: str, *args: object):
        return 0


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
        constraint_name="rental_customer",
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


def _stub_session() -> ComposerSession:
    return ComposerSession(snapshot=_snapshot(), connection=_StubConnection())


async def _invoke(
    tool: "FunctionTool", payload: dict[str, object]
) -> dict[str, object]:
    raw = await tool.on_invoke_tool(None, json.dumps(payload))  # pyright: ignore[reportArgumentType]
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return {str(key): value for key, value in parsed.items()}


# ---------- schema baking ----------


def test_schema_map_tool_bakes_table_enum():
    tool = build_schema_map_tool(_stub_session())
    enum = tool.params_json_schema["properties"]["root_table"]["anyOf"][0]["enum"]
    assert set(enum) == {"customer", "rental"}


def test_sample_tool_bakes_filter_predicate_schema():
    tool = build_sample_tool(_stub_session())
    columns = tool.params_json_schema["properties"]["predicate"]["items"]["properties"]["column"]["enum"]
    assert "customer_id" in columns
    assert "rental_date" in columns


def test_query_tool_bakes_edge_enum_and_aggregate_fns():
    tool = build_query_tool(_stub_session())
    spec = tool.params_json_schema["properties"]["spec"]
    join_edge_enum = spec["properties"]["join"]["items"]["properties"]["via_edge"]["enum"]
    assert "rental.customer_id->customer" in join_edge_enum
    agg_fns = spec["properties"]["aggregate"]["items"]["properties"]["fn"]["enum"]
    assert set(agg_fns) == {"avg", "count", "max", "min", "sum"}


def test_build_composer_tools_returns_five_tools_in_fixed_order():
    tools = build_composer_tools(_stub_session())
    assert [tool.name for tool in tools] == [
        "schema_map",
        "profile",
        "sample",
        "neighborhood",
        "query",
    ]


# ---------- invoke handlers ----------


@pytest.mark.asyncio
async def test_schema_map_tool_runs_with_default_depth():
    tool = build_schema_map_tool(_stub_session())
    response = await _invoke(tool, {})
    assert response["root_table"] is None
    tables = response["tables"]
    assert isinstance(tables, list)


@pytest.mark.asyncio
async def test_schema_map_tool_surfaces_unknown_root_as_error():
    tool = build_schema_map_tool(_stub_session())
    response = await _invoke(
        tool, {"root_table": "nope", "depth": 1}
    )
    assert response["error_type"] == "KeyError"


@pytest.mark.asyncio
async def test_sample_tool_surfaces_validation_errors():
    tool = build_sample_tool(_stub_session())
    response = await _invoke(
        tool, {"table": "customer", "n": 0}
    )
    assert response["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_neighborhood_tool_surfaces_not_implemented_for_depth_two():
    tool = build_neighborhood_tool(_stub_session())
    response = await _invoke(
        tool, {"table": "customer", "row_id": 45, "depth": 2}
    )
    assert response["error_type"] == "NotImplementedError"


@pytest.mark.asyncio
async def test_query_tool_requires_spec_object():
    tool = build_query_tool(_stub_session())
    response = await _invoke(tool, {})
    assert response["error_type"] == "TypeError"


@pytest.mark.asyncio
async def test_profile_tool_surfaces_unknown_column_as_error():
    tool = build_profile_tool(_stub_session())
    response = await _invoke(
        tool, {"table": "customer", "column": "nope"}
    )
    assert response["error_type"] == "KeyError"


# ---------- integration against live sakila ----------


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
async def test_sample_tool_against_sakila_returns_rows():
    session, conn = await _live_session()
    try:
        tool = build_sample_tool(session)
        response = await _invoke(tool, {"table": "customer", "n": 3})
        assert response["row_count"] == 3
        rows = response["rows"]
        assert isinstance(rows, list) and len(rows) == 3
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_query_tool_coerces_iso_timestamp_against_sakila():
    session, conn = await _live_session()
    try:
        tool = build_query_tool(session)
        response = await _invoke(
            tool,
            {
                "spec": {
                    "from": "rental",
                    "filter": [
                        {
                            "column": "rental_date",
                            "op": "gt",
                            "value": "2005-01-01T00:00:00",
                        }
                    ],
                    "aggregate": [
                        {"fn": "count", "alias": "total"},
                    ],
                }
            },
        )
        rows = response["rows"]
        assert isinstance(rows, list)
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, dict)
        assert isinstance(row["total"], int)
        assert row["total"] > 0
    finally:
        await conn.close()
