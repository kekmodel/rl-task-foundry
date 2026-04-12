from __future__ import annotations

import ast
import json
from pathlib import Path

import sqlglot

from rl_task_foundry.config.load import load_config
from rl_task_foundry.config.models import AtomicToolConfig
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.synthesis import atomic_tools as atomic_tools_module
from rl_task_foundry.synthesis.atomic_tools import (
    AtomicToolFamily,
    AtomicToolGenerator,
)


def _sample_graph() -> SchemaGraph:
    return SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="customers",
                primary_key=("customer_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="customer_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="tier",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="signup_date",
                        data_type="date",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="score",
                        data_type="numeric",
                        ordinal_position=4,
                        is_nullable=True,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="email",
                        data_type="text",
                        ordinal_position=5,
                        is_nullable=True,
                        visibility="internal",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="api_key",
                        data_type="text",
                        ordinal_position=6,
                        is_nullable=True,
                        visibility="blocked",
                    ),
                ],
                row_estimate=1000,
            ),
            TableProfile(
                schema_name="public",
                table_name="orders",
                primary_key=("order_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="order_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="customer_id",
                        data_type="int4",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="status",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="total_amount",
                        data_type="numeric",
                        ordinal_position=4,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="ordered_at",
                        data_type="timestamp without time zone",
                        ordinal_position=5,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
                row_estimate=5000,
            ),
            TableProfile(
                schema_name="public",
                table_name="line_items",
                primary_key=("line_item_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="line_items",
                        column_name="line_item_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="line_items",
                        column_name="order_id",
                        data_type="int4",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="line_items",
                        column_name="sku",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="line_items",
                        column_name="quantity",
                        data_type="int4",
                        ordinal_position=4,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="line_items",
                        column_name="unit_price",
                        data_type="numeric",
                        ordinal_position=5,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
                row_estimate=18000,
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_customer_id_fkey",
                source_schema="public",
                source_table="orders",
                source_columns=("customer_id",),
                target_schema="public",
                target_table="customers",
                target_columns=("customer_id",),
                source_is_unique=False,
                fanout_estimate=5.0,
            ),
            ForeignKeyEdge(
                constraint_name="line_items_order_id_fkey",
                source_schema="public",
                source_table="line_items",
                source_columns=("order_id",),
                target_schema="public",
                target_table="orders",
                target_columns=("order_id",),
                source_is_unique=False,
                fanout_estimate=3.6,
            ),
        ],
    )


def test_atomic_tool_generator_is_deterministic_and_covers_tool_families() -> None:
    config = load_config("rl_task_foundry.yaml")
    generator = AtomicToolGenerator(config.atomic_tools)
    graph = _sample_graph()

    first = generator.generate_bundle(graph, db_id="sakila")
    second = generator.generate_bundle(graph, db_id="sakila")

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.actor_tool_definitions_json() == second.actor_tool_definitions_json()
    assert {tool.family for tool in first.tools} == {
        AtomicToolFamily.T1_POINT_LOOKUP,
        AtomicToolFamily.T2_BOUNDED_ENUMERATION,
        AtomicToolFamily.T3_SINGLE_COLUMN_FILTER,
        AtomicToolFamily.T4_FK_TRAVERSAL,
        AtomicToolFamily.T5_DISTINCT_VALUES,
        AtomicToolFamily.T6_FILTERED_AGGREGATE,
        AtomicToolFamily.T7_SORTED_TOP_K,
        AtomicToolFamily.T8_GROUPED_AGGREGATE_TOP_K,
    }

    names = {tool.name for tool in first.tools}
    tool_by_name = {tool.name: tool for tool in first.tools}
    assert "get_customer_by_id" in names
    assert "get_customer_by_ids_batch" in names
    assert "count_customer" in names
    assert "list_customer_ids" in names
    assert "filter_customer_by_tier_like" in names
    assert "filter_order_by_total_amount_range" in names
    assert "distinct_order_status" in names
    assert "traverse_order_to_customer_via_customer_id" in names
    assert "traverse_customer_to_order_via_customer_id" in names
    assert "count_order_by_status_eq" in names
    assert "sum_order_total_amount_by_status_eq" in names
    assert "top_k_order_by_total_amount_asc" in names
    assert "top_k_order_by_total_amount_asc_where_status_eq" in names
    assert "top_k_order_grouped_by_customer_id_sum_total_amount_desc" in names
    assert "top_k_order_grouped_by_customer_id_sum_total_amount_desc_where_status_eq" in names
    assert tool_by_name["get_customer_by_id"].description == "Get customer for given primary key."
    assert (
        tool_by_name["top_k_order_by_total_amount_asc"].description
        == "Get top order rows ordered by total amount ascending."
    )
    assert (
        tool_by_name["top_k_order_by_total_amount_asc_where_status_eq"].description
        == "Get top order rows for given status, ordered by total amount ascending."
    )
    assert (
        tool_by_name["top_k_order_grouped_by_customer_id_sum_total_amount_desc"].description
        == "Get top customer id groups for order, ordered by sum total amount descending."
    )


def test_atomic_tool_generator_excludes_internal_and_blocked_columns() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(_sample_graph(), db_id="sakila")

    assert all("email" not in tool.name for tool in bundle.tools)
    assert all("api_key" not in tool.name for tool in bundle.tools)
    assert all("email" not in tool.sql for tool in bundle.tools)
    assert all("api_key" not in tool.sql for tool in bundle.tools)


def test_atomic_tool_generator_applies_deterministic_compression_priority() -> None:
    bundle = AtomicToolGenerator(
        AtomicToolConfig(
            max_tool_count=26,
            bounded_result_limit=100,
            max_batch_values=128,
        )
    ).generate_bundle(_sample_graph(), db_id="sakila")

    names = {tool.name for tool in bundle.tools}
    assert len(bundle.tools) <= 26
    assert "get_customer_by_id" in names
    assert "filter_customer_by_tier_eq" in names
    assert "traverse_order_to_customer_via_customer_id" in names
    assert all(tool.family != AtomicToolFamily.T6_FILTERED_AGGREGATE for tool in bundle.tools)
    assert all(tool.family != AtomicToolFamily.T8_GROUPED_AGGREGATE_TOP_K for tool in bundle.tools)
    assert all(tool.family != AtomicToolFamily.T7_SORTED_TOP_K for tool in bundle.tools)
    assert all(not tool.name.endswith("_like") for tool in bundle.tools)
    assert all(not tool.name.startswith("distinct_") for tool in bundle.tools)
    assert all(not tool.name.endswith("_range") for tool in bundle.tools)


def test_atomic_tool_generator_renders_actor_payload_and_source() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(_sample_graph(), db_id="sakila")
    tool_definitions = json.loads(bundle.actor_tool_definitions_json())

    assert tool_definitions
    assert set(tool_definitions[0]) == {
        "description",
        "name",
        "params_schema",
        "returns_schema",
    }
    assert "async def get_customer_by_id(conn, customer_id):" in bundle.source
    assert "async def list_customer_ids(conn, limit):" in bundle.source
    assert "MAX_BATCH_VALUES = 128" in bundle.source
    assert "MAX_BOUNDED_RESULT_LIMIT = 100" in bundle.source
    assert "limit = _bounded_limit(limit)" in bundle.source
    assert "await conn.fetchrow" in bundle.source
    assert "await conn.fetch(" in bundle.source
    assert "await conn.fetchval" in bundle.source
    assert "INSERT" not in bundle.source
    assert "UPDATE" not in bundle.source
    assert "DELETE" not in bundle.source


def test_atomic_multi_row_tools_require_limit_param_with_runtime_cap() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(_sample_graph(), db_id="sakila")
    tool_by_name = {tool.name: tool for tool in bundle.tools}

    assert tool_by_name["list_customer_ids"].params_schema["required"] == ["limit"]
    assert "limit" in tool_by_name["filter_customer_by_tier_eq"].params_schema["required"]
    assert "limit" in tool_by_name["filter_order_by_total_amount_range"].params_schema["required"]
    assert "limit" in tool_by_name["distinct_order_status"].params_schema["required"]
    assert "limit" in tool_by_name["traverse_customer_to_order_via_customer_id"].params_schema["required"]
    assert tool_by_name["top_k_order_by_total_amount_asc"].params_schema["required"] == ["limit"]
    assert "limit" in tool_by_name["top_k_order_by_total_amount_asc_where_status_eq"].params_schema["required"]
    assert tool_by_name["top_k_order_grouped_by_customer_id_sum_total_amount_desc"].params_schema["required"] == [
        "limit"
    ]
    assert "limit" in tool_by_name[
        "top_k_order_grouped_by_customer_id_sum_total_amount_desc_where_status_eq"
    ].params_schema["required"]
    assert tool_by_name["count_customer"].params_schema["required"] == []
    assert tool_by_name["count_order_by_status_eq"].params_schema["required"] == ["value"]
    assert "maximum" not in tool_by_name["list_customer_ids"].params_schema["properties"]["limit"]


def test_filtered_aggregate_rounds_avg_and_float_sum_with_configured_precision() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig(float_precision=2)).generate_bundle(
        _sample_graph(),
        db_id="sakila",
    )
    tool_by_name = {tool.name: tool for tool in bundle.tools}

    avg_sql = tool_by_name["avg_customer_score_by_tier_eq"].sql
    float_sum_sql = tool_by_name["sum_order_total_amount_by_status_eq"].sql
    int_sum_sql = tool_by_name["sum_line_item_quantity_by_sku_eq"].sql

    assert "ROUND(AVG(t.\"score\")::numeric, 2) AS value" in avg_sql
    assert "ROUND(SUM(t.\"total_amount\")::numeric, 2) AS value" in float_sum_sql
    assert "ROUND(" not in int_sum_sql
    assert tool_by_name["avg_line_item_quantity_by_sku_eq"].returns_schema == {
        "type": ["number", "null"]
    }
    assert tool_by_name["sum_line_item_quantity_by_sku_eq"].returns_schema == {
        "type": ["integer", "null"]
    }


def test_grouped_aggregate_rounds_avg_and_float_sum_with_configured_precision() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig(float_precision=2)).generate_bundle(
        _sample_graph(),
        db_id="sakila",
    )
    tool_by_name = {tool.name: tool for tool in bundle.tools}

    avg_sql = tool_by_name["top_k_customer_grouped_by_tier_avg_score_desc"].sql
    float_sum_sql = tool_by_name["top_k_order_grouped_by_customer_id_sum_total_amount_desc"].sql
    count_sql = tool_by_name["top_k_order_grouped_by_customer_id_count_order_id_desc"].sql

    assert "ROUND(AVG(t.\"score\")::numeric, 2) AS value" in avg_sql
    assert "ROUND(SUM(t.\"total_amount\")::numeric, 2) AS value" in float_sum_sql
    assert "COUNT(t.\"order_id\")::bigint AS value" in count_sql
    assert tool_by_name["top_k_customer_grouped_by_tier_avg_score_desc"].returns_schema == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "tier": {"type": "string"},
                "value": {"type": "number"},
            },
            "required": ["tier", "value"],
            "additionalProperties": False,
        },
        "maxItems": 100,
    }
    assert tool_by_name["top_k_order_grouped_by_customer_id_count_order_id_desc"].returns_schema == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "value": {"type": "integer"},
            },
            "required": ["customer_id", "value"],
            "additionalProperties": False,
        },
        "maxItems": 100,
    }


def test_generated_atomic_tool_source_is_ast_valid_and_async_conn_first() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(_sample_graph(), db_id="sakila")

    module = ast.parse(bundle.source)
    async_defs = [node for node in module.body if isinstance(node, ast.AsyncFunctionDef)]

    assert len(async_defs) == len(bundle.tools)
    assert all(node.args.args for node in async_defs)
    assert all(node.args.args[0].arg == "conn" for node in async_defs)


def test_generated_atomic_tool_sql_parses_under_postgres_dialect() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(_sample_graph(), db_id="sakila")

    for tool in bundle.tools:
        parsed = sqlglot.parse_one(tool.sql, dialect="postgres")
        assert parsed is not None


def test_atomic_tool_names_do_not_leak_task_intent_keywords() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(_sample_graph(), db_id="sakila")
    forbidden_keywords = {
        "assignment",
        "itinerary",
        "recommendation",
        "roster",
        "schedule",
        "trip",
        "budget",
    }

    for tool in bundle.tools:
        assert all(keyword not in tool.name for keyword in forbidden_keywords)


def test_atomic_tools_module_has_zero_legacy_tool_imports() -> None:
    module_source = Path(atomic_tools_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(module_source)
    imported_modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert all(not name.startswith("rl_task_foundry.tools") for name in imported_modules)
