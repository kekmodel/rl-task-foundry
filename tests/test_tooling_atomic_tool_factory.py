"""Tests for tooling.atomic.tool_factory.

Unit tests exercise v2 schema baking and API-envelope behavior with a
stub connection. Integration tests cover the solver-facing v2 resource
tools against pagila.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import asyncpg
import pytest

if TYPE_CHECKING:
    from agents import FunctionTool

import rl_task_foundry.tooling.atomic as atomic_public
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
    build_atomic_tools,
)
from rl_task_foundry.tooling.common import snapshot_from_graph


class _StubConnection:
    async def fetch(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("connection must not be used in plan-only tests")

    async def fetchrow(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("connection must not be used in plan-only tests")


class _MaterializingStubConnection:
    async def fetch(self, *args, **kwargs):
        return [{"id": 123}]

    async def fetchrow(self, *args, **kwargs):
        return {"cnt": 9, "agg": 42, "store_id": 1}


class _ConcurrentGuardConnection:
    def __init__(self) -> None:
        self.active = False
        self.calls = 0

    async def fetch(self, *args, **kwargs):
        raise AssertionError("fetch is not used by this test")

    async def fetchrow(self, *args, **kwargs):
        if self.active:
            raise RuntimeError("concurrent connection use")
        self.active = True
        self.calls += 1
        try:
            await asyncio.sleep(0)
            return {"store_id": self.calls}
        finally:
            self.active = False


def _column(
    table: str,
    name: str,
    data_type: str = "integer",
    *,
    schema: str = "public",
    is_primary_key: bool = False,
    is_foreign_key: bool = False,
    visibility: str = "user_visible",
) -> ColumnProfile:
    return ColumnProfile(
        schema_name=schema,
        table_name=table,
        column_name=name,
        data_type=data_type,
        ordinal_position=1,
        is_nullable=False,
        visibility=visibility,  # type: ignore[arg-type]
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
            _column("customer", "api_token", data_type="text", visibility="blocked"),
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
            _column("rental", "api_token", data_type="text", visibility="blocked"),
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


def _duplicate_name_snapshot():
    public_customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[
            _column("customer", "id", schema="public", is_primary_key=True),
        ],
        primary_key=("id",),
    )
    crm_customer = TableProfile(
        schema_name="crm",
        table_name="customer",
        columns=[
            _column("customer", "id", schema="crm", is_primary_key=True),
            _column(
                "customer",
                "public_customer_id",
                schema="crm",
                is_foreign_key=True,
            ),
        ],
        primary_key=("id",),
    )
    booking = TableProfile(
        schema_name="public",
        table_name="booking",
        columns=[
            _column("booking", "id", is_primary_key=True),
            _column("booking", "customer_id", is_foreign_key=True),
        ],
        primary_key=("id",),
    )
    return snapshot_from_graph(
        SchemaGraph(
            tables=[public_customer, crm_customer, booking],
            edges=[
                ForeignKeyEdge(
                    constraint_name="crm_customer_public_customer_fk",
                    source_schema="crm",
                    source_table="customer",
                    source_columns=("public_customer_id",),
                    target_schema="public",
                    target_table="customer",
                    target_columns=("id",),
                ),
                ForeignKeyEdge(
                    constraint_name="booking_crm_customer_fk",
                    source_schema="public",
                    source_table="booking",
                    source_columns=("customer_id",),
                    target_schema="crm",
                    target_table="customer",
                    target_columns=("id",),
                ),
            ],
        )
    )


def _stub_session() -> AtomicSession:
    return AtomicSession(
        snapshot=_snapshot(),
        connection=_StubConnection(),
        store=CursorStore(),
    )


async def _invoke(
    tool: "FunctionTool", payload: dict[str, object]
) -> dict[str, object]:
    result = await tool.on_invoke_tool(None, json.dumps(payload))  # pyright: ignore[reportArgumentType]
    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    return {str(key): value for key, value in parsed.items()}


# ---------- v2 schema baking ----------


def test_build_atomic_tools_returns_tools_in_calculus_order():
    tools = build_atomic_tools(_stub_session())
    assert [tool.name for tool in tools] == [
        "create_record_set",
        "filter_record_set",
        "filter_record_set_by_values",
        "filter_record_set_by_pattern",
        "filter_record_set_by_null",
        "filter_record_set_by_related",
        "follow_relation",
        "intersect_record_sets",
        "sort_record_set",
        "list_record_refs",
        "list_records",
        "count_records",
        "aggregate_records",
        "get_record",
    ]


def test_package_public_surface_is_v2_only():
    assert hasattr(atomic_public, "build_atomic_tools")
    for endpoint_builder in (
        "build_filter_record_set_tool",
        "build_filter_record_set_by_values_tool",
        "build_filter_record_set_by_pattern_tool",
        "build_filter_record_set_by_null_tool",
        "build_filter_record_set_by_related_tool",
        "build_sort_record_set_tool",
        "build_list_record_refs_tool",
        "build_list_records_tool",
        "build_get_record_tool",
    ):
        assert hasattr(atomic_public, endpoint_builder)
    for legacy_name in (
        "build_rows_where_tool",
        "build_rows_via_tool",
        "build_take_tool",
        "build_group_top_tool",
        "build_filter_rows_tool",
        "build_sort_rows_tool",
        "build_fetch_rows_tool",
        "build_read_row_tool",
        "rows_where",
        "take",
        "group_top",
    ):
        assert not hasattr(atomic_public, legacy_name)


def test_v2_create_and_filter_schemas_are_resource_oriented():
    tools = {tool.name: tool for tool in build_atomic_tools(_stub_session())}
    create_schema = tools["create_record_set"].params_json_schema
    assert create_schema["required"] == ["table"]
    assert set(create_schema["properties"]["table"]["enum"]) == {
        "customer",
        "rental",
    }

    filter_schema = tools["filter_record_set"].params_json_schema
    assert filter_schema["required"] == [
        "record_set_id",
        "column",
        "op",
        "value",
    ]
    assert set(filter_schema["properties"]["op"]["enum"]) == {
        "eq",
        "gt",
        "gte",
        "lt",
        "lte",
        "neq",
    }
    assert "is_null" not in filter_schema["properties"]["op"]["enum"]
    assert "in" not in filter_schema["properties"]["op"]["enum"]
    assert "like" not in filter_schema["properties"]["op"]["enum"]
    assert {"type": "null"} not in filter_schema["properties"]["value"]["anyOf"]
    assert all(
        branch.get("type") != "array"
        for branch in filter_schema["properties"]["value"]["anyOf"]
    )
    assert "cursor" not in filter_schema["properties"]
    assert "record_set_id" in filter_schema["properties"]
    assert "enum" not in filter_schema["properties"]["column"]

    values_filter_schema = tools["filter_record_set_by_values"].params_json_schema
    assert values_filter_schema["required"] == [
        "record_set_id",
        "column",
        "values",
    ]
    assert values_filter_schema["properties"]["values"]["type"] == "array"
    assert values_filter_schema["properties"]["values"]["minItems"] == 1

    pattern_filter_schema = tools["filter_record_set_by_pattern"].params_json_schema
    assert pattern_filter_schema["required"] == [
        "record_set_id",
        "column",
        "pattern",
    ]
    assert pattern_filter_schema["properties"]["pattern"]["type"] == "string"

    null_filter_schema = tools["filter_record_set_by_null"].params_json_schema
    assert null_filter_schema["required"] == ["record_set_id", "column", "op"]
    assert set(null_filter_schema["properties"]["op"]["enum"]) == {
        "is_null",
        "is_not_null",
    }

    follow_schema = tools["follow_relation"].params_json_schema
    assert "enum" not in follow_schema["properties"]["edge_label"]

    sort_schema = tools["sort_record_set"].params_json_schema
    assert "enum" not in sort_schema["properties"]["column"]

    aggregate_schema = tools["aggregate_records"].params_json_schema
    assert "enum" not in aggregate_schema["properties"]["column"]

    read_schema = tools["get_record"].params_json_schema
    assert "enum" not in read_schema["properties"]["columns"]["items"]
    assert {"type": "null"} not in read_schema["properties"]["record_id"]["anyOf"]


def test_v2_tools_are_strict_schemas():
    tools = build_atomic_tools(_stub_session())

    assert all(tool.strict_json_schema is True for tool in tools)


@pytest.mark.asyncio
async def test_v2_tool_surface_uses_schema_handles_for_duplicate_table_names():
    session = AtomicSession(
        snapshot=_duplicate_name_snapshot(),
        connection=_StubConnection(),
        store=CursorStore(),
    )
    tools = {tool.name: tool for tool in build_atomic_tools(session)}
    create_schema = tools["create_record_set"].params_json_schema
    assert set(create_schema["properties"]["table"]["enum"]) == {
        "booking",
        "crm.customer",
        "public.customer",
    }

    created = await _invoke(
        tools["create_record_set"], {"table": "crm.customer"}
    )
    resource = created["resource"]
    assert isinstance(resource, dict)
    assert resource["table"] == "crm.customer"
    assert resource["relations"] == [
        "crm%2Ecustomer.public_customer_id->public%2Ecustomer",
        "crm%2Ecustomer<-booking.customer_id",
    ]


def _schema_descriptions(value: object) -> list[str]:
    descriptions: list[str] = []
    if isinstance(value, dict):
        description = value.get("description")
        if isinstance(description, str):
            descriptions.append(description)
        for child in value.values():
            descriptions.extend(_schema_descriptions(child))
    elif isinstance(value, list):
        for child in value:
            descriptions.extend(_schema_descriptions(child))
    return descriptions


def _loose_schema_paths(value: object, *, path: str = "$") -> list[str]:
    loose: list[str] = []
    if isinstance(value, dict):
        if value.get("items") == {}:
            loose.append(f"{path}.items")
        if value.get("additionalProperties") is True:
            loose.append(f"{path}.additionalProperties")
        for key in ("properties", "$defs"):
            child_map = value.get(key)
            if isinstance(child_map, dict):
                for name, child in child_map.items():
                    loose.extend(_loose_schema_paths(child, path=f"{path}.{name}"))
        if "items" in value:
            loose.extend(_loose_schema_paths(value["items"], path=f"{path}[]"))
        for combiner in ("anyOf", "oneOf", "allOf"):
            branches = value.get(combiner)
            if isinstance(branches, list):
                for index, child in enumerate(branches):
                    loose.extend(
                        _loose_schema_paths(child, path=f"{path}.{combiner}[{index}]")
                    )
    return loose


def test_v2_tool_descriptions_are_endpoint_facing():
    tools = build_atomic_tools(_stub_session())
    banned_fragments = (
        "filter_rows",
        "sort_rows",
        "fetch_rows",
        "read_row",
        "atomic",
        "cursor",
        "SQL",
        "trace",
        "solver",
        "composer",
        "canonical",
        "row_set",
        "row_ref",
        "get_row",
        "list_row_refs",
        " row ",
        " rows ",
    )
    for tool in tools:
        text = " ".join(
            [
                tool.description,
                *_schema_descriptions(tool.params_json_schema),
            ]
        )
        for banned in banned_fragments:
            assert banned not in text


def test_v2_tool_schemas_do_not_use_empty_or_unbounded_subschemas():
    tools = build_atomic_tools(_stub_session())

    assert {
        tool.name: _loose_schema_paths(tool.params_json_schema)
        for tool in tools
    } == {tool.name: [] for tool in tools}


def test_v2_list_records_schema_distinguishes_paths_from_columns():
    tools = {tool.name: tool for tool in build_atomic_tools(_stub_session())}
    descriptions = _schema_descriptions(tools["list_records"].params_json_schema)
    text = " ".join(descriptions)

    assert "Put only relation labels in path" in text
    assert "put the final field name in column" in text
    assert "Do not include column names here" in text


def test_v2_list_record_refs_allows_limit_one_and_uses_api_cap():
    session = _stub_session()
    session.max_fetch_limit = 17
    tools = {tool.name: tool for tool in build_atomic_tools(session)}
    limit_schema = tools["list_record_refs"].params_json_schema["properties"]["limit"]
    assert limit_schema["minimum"] == 1
    assert limit_schema["maximum"] == 17


# ---------- invoke handlers (plan-only) ----------


@pytest.mark.asyncio
async def test_v2_create_filter_and_follow_relation_return_resource_payloads():
    session = _stub_session()
    tools = {tool.name: tool for tool in build_atomic_tools(session)}
    created = await _invoke(tools["create_record_set"], {"table": "rental"})
    assert created["ok"] is True
    resource = created["resource"]
    assert isinstance(resource, dict)
    assert resource["id"] == "record_set_1"
    assert resource["type"] == "record_set"
    assert resource["table"] == "rental"
    assert resource["columns"] == ["rental_id", "customer_id", "rental_date"]
    assert resource["column_types"]["rental_date"] == "timestamp"
    assert resource["column_visibility"]["rental_date"] == "user_visible"
    assert "api_token" not in resource["columns"]
    assert resource["primary_key"] == ["rental_id"]
    assert resource["relations"] == ["rental.customer_id->customer"]

    filtered = await _invoke(
        tools["filter_record_set"],
        {
            "record_set_id": resource["id"],
            "column": "customer_id",
            "op": "eq",
            "value": 45,
        },
    )
    filtered_resource = filtered["resource"]
    assert isinstance(filtered_resource, dict)
    assert filtered_resource["id"] == "record_set_2"
    assert filtered_resource["table"] == "rental"

    followed = await _invoke(
        tools["follow_relation"],
        {
            "source_record_set_id": filtered_resource["id"],
            "edge_label": "rental.customer_id->customer",
        },
    )
    followed_resource = followed["resource"]
    assert isinstance(followed_resource, dict)
    assert followed_resource["id"] == "record_set_3"
    assert followed_resource["table"] == "customer"

    events = session.trace_events
    assert [event["operation"] for event in events] == [
        "create_record_set",
        "filter_record_set",
        "follow_relation",
    ]
    trace_resource = {
        "id": "record_set_1",
        "type": "record_set",
        "table": "rental",
    }
    filtered_trace_resource = {
        "id": "record_set_2",
        "type": "record_set",
        "table": "rental",
    }
    followed_trace_resource = {
        "id": "record_set_3",
        "type": "record_set",
        "table": "customer",
    }
    assert events[0]["action"] == "create_resource"
    assert events[0]["output_resource"] == trace_resource
    assert "columns" not in events[0]["output_resource"]
    assert "relations" not in events[0]["output_resource"]
    assert events[1]["input_resource"] == trace_resource
    assert events[1]["predicate"] == {
        "column": "customer_id",
        "op": "eq",
        "value": 45,
    }
    assert events[1]["output_resource"] == filtered_trace_resource
    assert events[2]["relation"] == {
        "edge_label": "rental.customer_id->customer"
    }
    assert events[2]["output_resource"] == followed_trace_resource


@pytest.mark.asyncio
async def test_v2_materializing_tools_record_hidden_trace_events():
    session = AtomicSession(
        snapshot=_snapshot(),
        connection=_MaterializingStubConnection(),
        store=CursorStore(),
    )
    tools = {tool.name: tool for tool in build_atomic_tools(session)}

    base = await _invoke(tools["create_record_set"], {"table": "customer"})
    base_resource = base["resource"]
    assert isinstance(base_resource, dict)

    fetched = await _invoke(
        tools["list_record_refs"],
        {"record_set_id": base_resource["id"], "limit": 1},
    )
    assert fetched["ok"] is True
    listed = await _invoke(
        tools["list_records"],
        {
            "record_set_id": base_resource["id"],
            "limit": 1,
            "offset": 0,
            "fields": [{"name": "store_id", "column": "store_id", "path": []}],
        },
    )
    assert listed["ok"] is True
    assert listed["data"] == {
        "items": [{"store_id": 1}],
        "limit": 1,
        "offset": 0,
        "returned": 1,
    }
    counted = await _invoke(
        tools["count_records"],
        {"record_set_id": base_resource["id"]},
    )
    assert counted["data"] == {"count": 9}
    aggregated = await _invoke(
        tools["aggregate_records"],
        {"record_set_id": base_resource["id"], "fn": "sum", "column": "store_id"},
    )
    assert aggregated["data"] == {"value": 42}
    detail = await _invoke(
        tools["get_record"],
        {"table": "customer", "record_id": 123, "columns": ["store_id"]},
    )
    assert detail["data"] == {"record": {"store_id": 1}}

    events = session.trace_events
    assert [event["operation"] for event in events] == [
        "create_record_set",
        "list_record_refs",
        "list_records",
        "count_records",
        "aggregate_records",
        "get_record",
    ]
    base_trace_resource = {
        "id": base_resource["id"],
        "type": "record_set",
        "table": "customer",
    }
    assert events[0]["output_resource"] == base_trace_resource
    assert "columns" not in events[0]["output_resource"]
    assert "relations" not in events[0]["output_resource"]
    assert events[1]["input_resource"] == base_trace_resource
    assert events[1]["pagination"] == {"limit": 1, "offset": 0}
    assert events[1]["result_shape"] == {
        "kind": "record_ref_list",
        "table": "customer",
        "returned": 1,
    }
    assert events[2]["result_shape"] == {
        "kind": "record_list",
        "table": "customer",
        "returned": 1,
    }
    assert events[2]["fields"] == [
        {"name": "store_id", "path": [], "column": "store_id"}
    ]
    assert events[3]["result_shape"] == {"kind": "scalar", "field": "count"}
    assert events[4]["aggregate"] == {"fn": "sum", "column": "store_id"}
    assert events[5]["record_ref"] == {
        "type": "record_ref",
        "table": "customer",
        "id": 123,
    }


@pytest.mark.asyncio
async def test_v2_errors_use_api_envelope_without_repair_hints():
    session = _stub_session()
    tools = {tool.name: tool for tool in build_atomic_tools(session)}
    response = await _invoke(
        tools["filter_record_set"],
        {
            "record_set_id": "record_set_missing",
            "column": "customer_id",
            "op": "eq",
            "value": 45,
        },
    )
    assert response == {
        "ok": False,
        "error": {"type": "action_error", "code": "not_found"},
    }
    assert session.trace_events == [
        {
            "action": "tool_error",
            "operation": "filter_record_set",
            "visible_ok": False,
            "error": {"type": "action_error", "code": "not_found"},
            "request": {
                "record_set_id": "record_set_missing",
                "column": "customer_id",
                "op": "eq",
                "value": 45,
            },
        }
    ]


@pytest.mark.asyncio
async def test_v2_get_record_rejects_null_record_id_as_request_error():
    session = _stub_session()
    tools = {tool.name: tool for tool in build_atomic_tools(session)}
    response = await _invoke(
        tools["get_record"],
        {"table": "customer", "record_id": None, "columns": ["store_id"]},
    )
    assert response["ok"] is False
    assert response["error"]["type"] == "request_error"
    assert response["error"]["code"] == "invalid_request"
    assert "record_id" in response["error"]["message"]


@pytest.mark.asyncio
async def test_v2_tools_serialize_parallel_connection_use():
    conn = _ConcurrentGuardConnection()
    session = AtomicSession(
        snapshot=_snapshot(),
        connection=conn,
        store=CursorStore(),
    )
    tools = {tool.name: tool for tool in build_atomic_tools(session)}

    first, second = await asyncio.gather(
        _invoke(
            tools["get_record"],
            {"table": "customer", "record_id": 123, "columns": ["store_id"]},
        ),
        _invoke(
            tools["get_record"],
            {"table": "customer", "record_id": 124, "columns": ["store_id"]},
        ),
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert conn.calls == 2


@pytest.mark.asyncio
async def test_v2_tools_reject_blocked_non_handle_columns():
    session = _stub_session()
    tools = {tool.name: tool for tool in build_atomic_tools(session)}
    created = await _invoke(tools["create_record_set"], {"table": "customer"})
    resource = created["resource"]
    assert isinstance(resource, dict)
    assert "api_token" not in resource["columns"]

    filtered = await _invoke(
        tools["filter_record_set"],
        {
            "record_set_id": resource["id"],
            "column": "api_token",
            "op": "eq",
            "value": "secret",
        },
    )
    assert filtered["ok"] is False
    assert filtered["error"] == {"type": "action_error", "code": "not_found"}

    detail = await _invoke(
        tools["get_record"],
        {"table": "customer", "record_id": 123, "columns": ["api_token"]},
    )
    assert detail["ok"] is False
    assert detail["error"] == {"type": "action_error", "code": "not_found"}


@pytest.mark.asyncio
async def test_v2_filter_rejects_null_value_for_binary_ops_as_request_error():
    session = _stub_session()
    tools = {tool.name: tool for tool in build_atomic_tools(session)}
    created = await _invoke(tools["create_record_set"], {"table": "customer"})
    resource = created["resource"]
    assert isinstance(resource, dict)

    response = await _invoke(
        tools["filter_record_set"],
        {
            "record_set_id": resource["id"],
            "column": "store_id",
            "op": "eq",
            "value": None,
        },
    )

    assert response["ok"] is False
    assert response["error"]["type"] == "request_error"
    assert response["error"]["code"] == "invalid_request"
    assert "cannot be null" in response["error"]["message"]


@pytest.mark.asyncio
async def test_v2_filter_allows_nullary_null_predicates_without_value():
    session = _stub_session()
    tools = {tool.name: tool for tool in build_atomic_tools(session)}
    created = await _invoke(tools["create_record_set"], {"table": "customer"})
    resource = created["resource"]
    assert isinstance(resource, dict)

    filtered = await _invoke(
        tools["filter_record_set_by_null"],
        {
            "record_set_id": resource["id"],
            "column": "store_id",
            "op": "is_not_null",
        },
    )

    assert filtered["ok"] is True
    assert filtered["resource"]["id"] == "record_set_2"
    assert session.trace_events[-1]["predicate"] == {
        "column": "store_id",
        "op": "is_not_null",
    }


@pytest.mark.asyncio
async def test_v2_filter_shape_specific_endpoints_return_resources():
    session = _stub_session()
    tools = {tool.name: tool for tool in build_atomic_tools(session)}
    created = await _invoke(tools["create_record_set"], {"table": "customer"})
    resource = created["resource"]
    assert isinstance(resource, dict)

    by_values = await _invoke(
        tools["filter_record_set_by_values"],
        {
            "record_set_id": resource["id"],
            "column": "store_id",
            "values": [1, 2],
        },
    )
    assert by_values["ok"] is True
    assert by_values["resource"]["id"] == "record_set_2"
    assert session.trace_events[-1]["operation"] == "filter_record_set_by_values"
    assert session.trace_events[-1]["predicate"] == {
        "column": "store_id",
        "op": "in",
        "value": [1, 2],
    }

    by_pattern = await _invoke(
        tools["filter_record_set_by_pattern"],
        {
            "record_set_id": resource["id"],
            "column": "first_name",
            "pattern": "%ma%",
        },
    )
    assert by_pattern["ok"] is True
    assert by_pattern["resource"]["id"] == "record_set_3"
    assert session.trace_events[-1]["operation"] == "filter_record_set_by_pattern"
    assert session.trace_events[-1]["predicate"] == {
        "column": "first_name",
        "op": "like",
        "value": "%ma%",
    }


@pytest.mark.asyncio
async def test_invalid_json_input_is_surfaced_as_api_error():
    session = _stub_session()
    tools = {tool.name: tool for tool in build_atomic_tools(session)}
    tool = tools["create_record_set"]
    result = await tool.on_invoke_tool(None, "{not-json")  # pyright: ignore[reportArgumentType]
    parsed = json.loads(result)
    assert parsed == {
        "ok": False,
        "error": {"type": "request_error", "code": "invalid_request"},
    }


# ---------- integration against live pagila ----------


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
async def test_v2_resource_tool_chain_against_pagila():
    session, conn = await _live_session()
    try:
        tools = {tool.name: tool for tool in build_atomic_tools(session)}

        base = await _invoke(tools["create_record_set"], {"table": "rental"})
        base_resource = base["resource"]
        assert isinstance(base_resource, dict)

        filtered = await _invoke(
            tools["filter_record_set"],
            {
                "record_set_id": base_resource["id"],
                "column": "customer_id",
                "op": "eq",
                "value": 45,
            },
        )
        filtered_resource = filtered["resource"]
        assert isinstance(filtered_resource, dict)

        sorted_rows = await _invoke(
            tools["sort_record_set"],
            {
                "record_set_id": filtered_resource["id"],
                "column": "rental_date",
                "direction": "asc",
            },
        )
        sorted_resource = sorted_rows["resource"]
        assert isinstance(sorted_resource, dict)

        fetched = await _invoke(
            tools["list_record_refs"],
            {"record_set_id": sorted_resource["id"], "limit": 1},
        )
        assert fetched["ok"] is True
        data = fetched["data"]
        assert isinstance(data, dict)
        assert data["returned"] == 1
        items = data["items"]
        assert isinstance(items, list)
        record_ref = items[0]
        assert isinstance(record_ref, dict)
        assert record_ref["type"] == "record_ref"
        assert record_ref["table"] == "rental"

        detail = await _invoke(
            tools["get_record"],
            {
                "table": "rental",
                "record_id": record_ref["id"],
                "columns": ["rental_id", "customer_id"],
            },
        )
        detail_data = detail["data"]
        assert isinstance(detail_data, dict)
        record = detail_data["record"]
        assert isinstance(record, dict)
        assert record["customer_id"] == 45
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_v2_list_records_preserves_source_alignment_across_fk_path():
    session, conn = await _live_session()
    try:
        tools = {tool.name: tool for tool in build_atomic_tools(session)}

        base = await _invoke(tools["create_record_set"], {"table": "payment"})
        base_resource = base["resource"]
        assert isinstance(base_resource, dict)

        customer_filtered = await _invoke(
            tools["filter_record_set"],
            {
                "record_set_id": base_resource["id"],
                "column": "customer_id",
                "op": "eq",
                "value": 487,
            },
        )
        customer_resource = customer_filtered["resource"]
        assert isinstance(customer_resource, dict)

        amount_filtered = await _invoke(
            tools["filter_record_set"],
            {
                "record_set_id": customer_resource["id"],
                "column": "amount",
                "op": "gt",
                "value": 2,
            },
        )
        amount_resource = amount_filtered["resource"]
        assert isinstance(amount_resource, dict)

        sorted_payments = await _invoke(
            tools["sort_record_set"],
            {
                "record_set_id": amount_resource["id"],
                "column": "payment_date",
                "direction": "desc",
            },
        )
        sorted_resource = sorted_payments["resource"]
        assert isinstance(sorted_resource, dict)

        listed = await _invoke(
            tools["list_records"],
            {
                "record_set_id": sorted_resource["id"],
                "limit": 5,
                "offset": 0,
                "fields": [
                    {"name": "amount", "column": "amount", "path": []},
                    {
                        "name": "payment_date",
                        "column": "payment_date",
                        "path": [],
                    },
                    {
                        "name": "film_title",
                        "column": "title",
                        "path": [
                            "payment.rental_id->rental",
                            "rental.inventory_id->inventory",
                            "inventory.film_id->film",
                        ],
                    },
                ],
            },
        )
        assert listed["ok"] is True
        data = listed["data"]
        assert isinstance(data, dict)
        assert data["returned"] == 5
        assert data["items"] == [
            {
                "amount": "3.99",
                "payment_date": "2022-07-13T04:08:47.090772+00:00",
                "film_title": "ENTRAPMENT SATISFACTION",
            },
            {
                "amount": "2.99",
                "payment_date": "2022-07-12T08:18:13.441249+00:00",
                "film_title": "SIERRA DIVIDE",
            },
            {
                "amount": "2.99",
                "payment_date": "2022-07-05T12:02:16.908779+00:00",
                "film_title": "SAMURAI LION",
            },
            {
                "amount": "7.99",
                "payment_date": "2022-07-03T20:50:57.546298+00:00",
                "film_title": "COMMAND DARLING",
            },
            {
                "amount": "3.99",
                "payment_date": "2022-06-24T09:54:08.027069+00:00",
                "film_title": "BILL OTHERS",
            },
        ]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_v2_filter_record_set_by_related_filters_source_records():
    session, conn = await _live_session()
    try:
        tools = {tool.name: tool for tool in build_atomic_tools(session)}

        base = await _invoke(tools["create_record_set"], {"table": "rental"})
        base_resource = base["resource"]
        assert isinstance(base_resource, dict)

        customer_filtered = await _invoke(
            tools["filter_record_set"],
            {
                "record_set_id": base_resource["id"],
                "column": "customer_id",
                "op": "eq",
                "value": 546,
            },
        )
        customer_resource = customer_filtered["resource"]
        assert isinstance(customer_resource, dict)

        rating_filtered = await _invoke(
            tools["filter_record_set_by_related"],
            {
                "record_set_id": customer_resource["id"],
                "path": [
                    "rental.inventory_id->inventory",
                    "inventory.film_id->film",
                ],
                "column": "rating",
                "op": "eq",
                "value": "PG-13",
            },
        )
        rating_resource = rating_filtered["resource"]
        assert isinstance(rating_resource, dict)
        assert rating_resource["table"] == "rental"

        sorted_rentals = await _invoke(
            tools["sort_record_set"],
            {
                "record_set_id": rating_resource["id"],
                "column": "rental_date",
                "direction": "desc",
            },
        )
        sorted_resource = sorted_rentals["resource"]
        assert isinstance(sorted_resource, dict)

        listed = await _invoke(
            tools["list_records"],
            {
                "record_set_id": sorted_resource["id"],
                "limit": 5,
                "offset": 0,
                "fields": [
                    {
                        "name": "film_title",
                        "column": "title",
                        "path": [
                            "rental.inventory_id->inventory",
                            "inventory.film_id->film",
                        ],
                    },
                    {"name": "rented_on", "column": "rental_date", "path": []},
                    {"name": "returned_on", "column": "return_date", "path": []},
                    {
                        "name": "rating",
                        "column": "rating",
                        "path": [
                            "rental.inventory_id->inventory",
                            "inventory.film_id->film",
                        ],
                    },
                ],
            },
        )
        assert listed["ok"] is True
        data = listed["data"]
        assert isinstance(data, dict)
        assert data["items"] == [
            {
                "film_title": "DRIFTER COMMANDMENTS",
                "rented_on": "2022-08-17T10:17:21+00:00",
                "returned_on": "2022-08-18T08:14:21+00:00",
                "rating": "PG-13",
            },
            {
                "film_title": "ENGLISH BULWORTH",
                "rented_on": "2022-07-12T21:11:21+00:00",
                "returned_on": "2022-07-21T01:35:21+00:00",
                "rating": "PG-13",
            },
            {
                "film_title": "HOBBIT ALIEN",
                "rented_on": "2022-07-08T12:12:12+00:00",
                "returned_on": "2022-07-10T08:01:12+00:00",
                "rating": "PG-13",
            },
            {
                "film_title": "EVOLUTION ALTER",
                "rented_on": "2022-07-08T09:01:28+00:00",
                "returned_on": "2022-07-12T09:37:28+00:00",
                "rating": "PG-13",
            },
            {
                "film_title": "SEATTLE EXPECATIONS",
                "rented_on": "2022-06-18T12:19:05+00:00",
                "returned_on": "2022-06-23T06:59:05+00:00",
                "rating": "PG-13",
            },
        ]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_v2_composite_pk_chain_against_pagila_film_actor():
    # Regression for iter21's observed failure: actor -> film_actor -> film
    # must keep working when an intermediate resource has a composite PK.
    session, conn = await _live_session()
    try:
        tools = {tool.name: tool for tool in build_atomic_tools(session)}

        base = await _invoke(tools["create_record_set"], {"table": "film_actor"})
        base_resource = base["resource"]
        assert isinstance(base_resource, dict)

        filtered = await _invoke(
            tools["filter_record_set"],
            {
                "record_set_id": base_resource["id"],
                "column": "actor_id",
                "op": "eq",
                "value": 87,
            },
        )
        filtered_resource = filtered["resource"]
        assert isinstance(filtered_resource, dict)

        composite_refs = await _invoke(
            tools["list_record_refs"],
            {"record_set_id": filtered_resource["id"], "limit": 3},
        )
        composite_data = composite_refs["data"]
        assert isinstance(composite_data, dict)
        composite_items = composite_data["items"]
        assert isinstance(composite_items, list)
        assert len(composite_items) == 3
        for record_ref in composite_items:
            assert isinstance(record_ref, dict)
            row_id = record_ref["id"]
            assert isinstance(row_id, list)
            assert len(row_id) == 2
            assert row_id[0] == 87

        aligned_films = await _invoke(
            tools["list_records"],
            {
                "record_set_id": filtered_resource["id"],
                "limit": 3,
                "offset": 0,
                "fields": [
                    {"name": "last_update", "column": "last_update", "path": []},
                    {
                        "name": "film_title",
                        "column": "title",
                        "path": ["film_actor.film_id->film"],
                    },
                ],
            },
        )
        aligned_data = aligned_films["data"]
        assert isinstance(aligned_data, dict)
        aligned_items = aligned_data["items"]
        assert isinstance(aligned_items, list)
        assert len(aligned_items) == 3
        for item in aligned_items:
            assert isinstance(item, dict)
            assert "last_update" in item
            assert isinstance(item["film_title"], str)

        film_set = await _invoke(
            tools["follow_relation"],
            {
                "source_record_set_id": filtered_resource["id"],
                "edge_label": "film_actor.film_id->film",
            },
        )
        film_resource = film_set["resource"]
        assert isinstance(film_resource, dict)

        sorted_films = await _invoke(
            tools["sort_record_set"],
            {
                "record_set_id": film_resource["id"],
                "column": "title",
                "direction": "asc",
            },
        )
        sorted_resource = sorted_films["resource"]
        assert isinstance(sorted_resource, dict)

        film_refs = await _invoke(
            tools["list_record_refs"],
            {"record_set_id": sorted_resource["id"], "limit": 3},
        )
        film_data = film_refs["data"]
        assert isinstance(film_data, dict)
        film_items = film_data["items"]
        assert isinstance(film_items, list)
        assert len(film_items) == 3
        for record_ref in film_items:
            assert isinstance(record_ref, dict)
            assert isinstance(record_ref["id"], int)

        first_composite_ref = composite_items[0]
        assert isinstance(first_composite_ref, dict)
        details = await _invoke(
            tools["get_record"],
            {
                "table": "film_actor",
                "record_id": first_composite_ref["id"],
                "columns": ["last_update"],
            },
        )
        detail_data = details["data"]
        assert isinstance(detail_data, dict)
        row = detail_data["record"]
        assert isinstance(row, dict)
        assert "last_update" in row
    finally:
        await conn.close()
