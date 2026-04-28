"""Tests for tooling.composer.tool_factory.

Unit tests exercise JSON schema enum baking and the on_invoke_tool
error path with a stub connection. Integration tests run against live
pagila — end-to-end schema_map, sample, and query through the SDK
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
    variants = tool.params_json_schema["properties"]["target"]["oneOf"]
    by_table = {
        variant["properties"]["table"]["enum"][0]: variant
        for variant in variants
    }
    customer_columns = by_table["customer"]["properties"]["predicate"][
        "items"
    ]["properties"]["column"]["enum"]
    rental_columns = by_table["rental"]["properties"]["predicate"]["items"][
        "properties"
    ]["column"]["enum"]
    assert "customer_id" in customer_columns
    assert "rental_date" not in customer_columns
    assert "rental_date" in rental_columns
    assert "store_id" not in rental_columns


def test_query_tool_bakes_edge_enum_and_aggregate_fns():
    tool = build_query_tool(_stub_session())
    assert "atomic" not in tool.description.lower()
    spec = tool.params_json_schema["properties"]["spec"]
    assert "filter" not in spec["properties"]
    assert "sort" not in spec["properties"]
    from_schema = spec["properties"]["from"]
    assert from_schema["required"] == ["table", "as"]
    join_item = spec["properties"]["join"]["items"]
    assert join_item["required"] == ["from", "via_edge", "as"]
    assert "previously declared alias" in spec["properties"]["join"]["description"]
    join_edge_enum = join_item["properties"]["via_edge"]["enum"]
    assert "rental.customer_id->customer" in join_edge_enum
    where_ref = spec["properties"]["where"]["items"]["properties"]["ref"]
    assert where_ref["required"] == ["as", "column"]
    assert spec["properties"]["where"]["items"]["required"] == ["ref", "op"]
    where_ops = spec["properties"]["where"]["items"]["properties"]["op"]["enum"]
    assert {"neq", "is_null", "is_not_null"}.issubset(set(where_ops))
    agg_fns = spec["properties"]["aggregate"]["items"]["properties"]["fn"]["enum"]
    assert set(agg_fns) == {"avg", "count", "max", "min", "sum"}
    assert spec["properties"]["aggregate"]["items"]["required"] == ["fn", "as"]


def test_build_composer_tools_returns_five_tools_in_fixed_order():
    tools = build_composer_tools(_stub_session())
    assert [tool.name for tool in tools] == [
        "schema_map",
        "profile",
        "sample",
        "neighborhood",
        "query",
    ]


def test_neighborhood_tool_row_id_schema_disallows_null():
    tool = build_neighborhood_tool(_stub_session())
    row_id_schema = tool.params_json_schema["properties"]["row_id"]
    assert {"type": "null"} not in row_id_schema["anyOf"]
    assert {"type": "string"} in row_id_schema["anyOf"]
    assert {"type": "integer"} in row_id_schema["anyOf"]
    assert any(
        branch.get("type") == "array" for branch in row_id_schema["anyOf"]
    )


def test_composer_tool_surface_is_db_native_without_internal_framing():
    tools = build_composer_tools(_stub_session())
    surface = json.dumps(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.params_json_schema,
            }
            for tool in tools
        ],
        sort_keys=True,
    ).lower()
    for leaked in ("atomic", "solver", "rlvr", "record_set", "cursor"):
        assert leaked not in surface
    assert "row_id" in surface
    assert "row_count" in surface


def _description_map(schema: object, *, path: str = "$") -> dict[str, str]:
    descriptions: dict[str, str] = {}
    if isinstance(schema, dict):
        description = schema.get("description")
        if isinstance(description, str):
            descriptions[path] = description
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for name, child in properties.items():
                descriptions.update(
                    _description_map(child, path=f"{path}.{name}")
                )
        items = schema.get("items")
        if items is not None:
            descriptions.update(_description_map(items, path=f"{path}[]"))
        for combiner in ("anyOf", "oneOf", "allOf"):
            branches = schema.get(combiner)
            if isinstance(branches, list):
                for index, child in enumerate(branches):
                    descriptions.update(
                        _description_map(child, path=f"{path}.{combiner}[{index}]")
                    )
    return descriptions


def _loose_schema_paths(schema: object, *, path: str = "$") -> list[str]:
    loose: list[str] = []
    if isinstance(schema, dict):
        if schema.get("items") == {}:
            loose.append(f"{path}.items")
        if schema.get("additionalProperties") is True:
            loose.append(f"{path}.additionalProperties")
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for name, child in properties.items():
                loose.extend(_loose_schema_paths(child, path=f"{path}.{name}"))
        items = schema.get("items")
        if items is not None:
            loose.extend(_loose_schema_paths(items, path=f"{path}[]"))
        for combiner in ("anyOf", "oneOf", "allOf"):
            branches = schema.get(combiner)
            if isinstance(branches, list):
                for index, child in enumerate(branches):
                    loose.extend(
                        _loose_schema_paths(child, path=f"{path}.{combiner}[{index}]")
                    )
    return loose


def test_composer_tool_schema_descriptions_are_prompt_aligned():
    tools = {tool.name: tool for tool in build_composer_tools(_stub_session())}
    descriptions = {
        name: _description_map(tool.params_json_schema)
        for name, tool in tools.items()
    }

    for tool in tools.values():
        assert tool.description
        tool_surface = json.dumps(
            {
                "description": tool.description,
                "schema": tool.params_json_schema,
            },
            sort_keys=True,
        ).lower()
        for leaked in ("solver", "actor", "rlvr", "pass_rate", "training"):
            assert leaked not in tool_surface

    assert "live rows still provide label evidence" in tools["schema_map"].description
    assert "the draft is grounded" in tools["sample"].description
    assert "nontrivial filters" in tools["profile"].description
    assert "reachable task paths" in tools["neighborhood"].description
    assert "exact rows that will be copied into the label" in tools["query"].description

    assert any(
        "Each column is scoped to the selected table" in description
        for description in descriptions["sample"].values()
    )
    assert any(
        "Each column is scoped to the selected table" in description
        for description in descriptions["profile"].values()
    )
    assert "same event/record" in descriptions["query"]["$.spec.join"]
    assert "independent sibling joins" in descriptions["query"]["$.spec.join"]
    assert "Filters define row membership" in descriptions["query"]["$.spec.where"]
    assert "customer-visible constraint" in descriptions["query"]["$.spec.where"]
    assert "Do not add helper filters" in descriptions["query"]["$.spec.where"]
    assert "Match the requested scope granularity" in descriptions["query"][
        "$.spec.where"
    ]
    assert "do not narrow a whole-context/history/list request" in descriptions[
        "query"
    ]["$.spec.where"]
    assert "Blocked or internal handle values" in descriptions["query"][
        "$.spec.where[].value"
    ]
    assert "Every selected field becomes a canonical label field" in descriptions[
        "query"
    ]["$.spec.select"]
    assert "select only values the user_request asks to receive" in descriptions[
        "query"
    ]["$.spec.select"]
    assert "Do not select profile/scope fields" in descriptions["query"]["$.spec.select"]
    assert "Do not select constraint, filter, scope" in descriptions[
        "query"
    ]["$.spec.select"]
    assert "During specificity retries" in descriptions["query"]["$.spec.select"]
    assert "append new requested fields instead of replacing them" in descriptions[
        "query"
    ]["$.spec.select"]
    assert "preserves the selected source column meaning" in descriptions[
        "query"
    ]["$.spec.select[].as"]
    assert "note/comment/description text" in descriptions[
        "query"
    ]["$.spec.select[].as"]
    assert "Prefer user-visible non-handle" in descriptions["query"]["$.spec.select"]
    assert "evidence marks them user-visible" in descriptions["query"]["$.spec.select"]
    assert "use an answer-visible/reproducible tie-breaker only if" in descriptions[
        "query"
    ]["$.spec.order_by"]
    assert "query.order_by.direction match that wording exactly" in descriptions[
        "query"
    ]["$.spec.order_by"]
    assert "selecting the field as output is not enough" in descriptions["query"][
        "$.spec.order_by"
    ]
    assert "returning the tied rows is safer" in descriptions["query"][
        "$.spec.order_by"
    ]
    assert "Match the direction or ranking stated in user_request" in descriptions[
        "query"
    ]["$.spec.order_by[].direction"]
    assert "same N in user_request" in descriptions["query"]["$.spec.limit"]
    assert "answer_contract.limit_phrase" in descriptions["query"]["$.spec.limit"]
    assert "cuts through rows with the same requested order key" in descriptions[
        "query"
    ]["$.spec.limit"]
    assert "answer-visible/reproducible" in descriptions["query"]["$.spec.order_by"]
    assert "unique visible ordering" in descriptions["query"]["$.spec.order_by"]
    assert "without group_by so it returns one row" in descriptions["query"][
        "$.spec.aggregate"
    ]
    assert {
        tool.name: _loose_schema_paths(tool.params_json_schema)
        for tool in tools.values()
    } == {tool.name: [] for tool in tools.values()}


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
        tool, {"target": {"table": "customer", "n": 0}}
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
async def test_neighborhood_tool_rejects_null_row_id_before_querying():
    tool = build_neighborhood_tool(_stub_session())
    response = await _invoke(tool, {"table": "customer", "row_id": None})
    assert response["error_type"] == "TypeError"
    assert "cannot be null" in response["error"]


@pytest.mark.asyncio
async def test_query_tool_requires_spec_object():
    tool = build_query_tool(_stub_session())
    response = await _invoke(tool, {})
    assert response["error_type"] == "TypeError"


@pytest.mark.asyncio
async def test_profile_tool_surfaces_unknown_column_as_error():
    tool = build_profile_tool(_stub_session())
    response = await _invoke(
        tool, {"target": {"table": "customer", "column": "nope"}}
    )
    assert response["error_type"] == "KeyError"


# ---------- integration against live pagila ----------


async def _live_session() -> tuple[ComposerSession, asyncpg.Connection]:
    config = load_config(Path("rl_task_foundry.yaml"))
    introspector = PostgresSchemaIntrospector(
        database=config.database,
        default_visibility=config.visibility.default_visibility,
        visibility_overrides=config.visibility.visibility_overrides,
    )
    graph = await introspector.introspect()
    snap = snapshot_from_graph(graph)
    conn = await asyncpg.connect(config.database.dsn)
    await _apply_session_settings(conn, solver_session_settings(config.database))
    return ComposerSession(snapshot=snap, connection=conn), conn


@pytest.mark.asyncio
async def test_sample_tool_against_pagila_returns_rows():
    session, conn = await _live_session()
    try:
        tool = build_sample_tool(session)
        response = await _invoke(
            tool,
            {"target": {"table": "customer", "n": 3}},
        )
        assert response["row_count"] == 3
        rows = response["rows"]
        assert isinstance(rows, list) and len(rows) == 3
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_neighborhood_tool_accepts_string_row_id_for_integer_pk_against_pagila():
    # Regression for a smoke-trial failure where the composer LLM passed
    # ``row_id="5"`` as a JSON string against ``customer.customer_id`` (integer).
    # asyncpg's binary protocol rejected the bound str, surfacing as UserError.
    session, conn = await _live_session()
    try:
        tool = build_neighborhood_tool(session)
        response = await _invoke(tool, {"table": "customer", "row_id": "5"})
        anchor = response["anchor"]
        assert isinstance(anchor, dict)
        assert anchor["table"] == "customer"
        attributes = anchor["attributes"]
        assert isinstance(attributes, dict)
        assert attributes["customer_id"] == 5
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_query_tool_coerces_iso_timestamp_against_pagila():
    session, conn = await _live_session()
    try:
        tool = build_query_tool(session)
        response = await _invoke(
            tool,
                {
                    "spec": {
                        "from": {"table": "rental", "as": "r"},
                        "where": [
                            {
                                "ref": {"as": "r", "column": "rental_date"},
                                "op": "gt",
                                "value": "2005-01-01T00:00:00",
                            }
                        ],
                        "aggregate": [
                            {"fn": "count", "as": "total"},
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
