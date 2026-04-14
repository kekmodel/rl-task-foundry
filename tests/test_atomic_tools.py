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
    AtomicToolDefinition,
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
                        n_distinct=-1.0,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="tier",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                        n_distinct=3,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="signup_date",
                        data_type="date",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                        n_distinct=365,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="score",
                        data_type="numeric",
                        ordinal_position=4,
                        is_nullable=True,
                        visibility="user_visible",
                        n_distinct=120,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="email",
                        data_type="text",
                        ordinal_position=5,
                        is_nullable=True,
                        visibility="internal",
                        n_distinct=-0.95,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="segment",
                        data_type="text",
                        ordinal_position=6,
                        is_nullable=True,
                        visibility="user_visible",
                        n_distinct=None,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="api_key",
                        data_type="text",
                        ordinal_position=7,
                        is_nullable=True,
                        visibility="blocked",
                        n_distinct=-1.0,
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
                        n_distinct=-1.0,
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
                        n_distinct=100,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="status",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                        n_distinct=4,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="total_amount",
                        data_type="numeric",
                        ordinal_position=4,
                        is_nullable=False,
                        visibility="user_visible",
                        n_distinct=300,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="ordered_at",
                        data_type="timestamp without time zone",
                        ordinal_position=5,
                        is_nullable=False,
                        visibility="user_visible",
                        n_distinct=1000,
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
                        n_distinct=-1.0,
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
                        n_distinct=500,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="line_items",
                        column_name="sku",
                        data_type="text",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                        n_distinct=500,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="line_items",
                        column_name="quantity",
                        data_type="int4",
                        ordinal_position=4,
                        is_nullable=False,
                        visibility="user_visible",
                        n_distinct=8,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="line_items",
                        column_name="unit_price",
                        data_type="numeric",
                        ordinal_position=5,
                        is_nullable=False,
                        visibility="user_visible",
                        n_distinct=200,
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


def test_atomic_tool_generator_is_deterministic_and_covers_new_tool_families() -> None:
    config = load_config("rl_task_foundry.yaml")
    generator = AtomicToolGenerator(config.atomic_tools)
    graph = _sample_graph()

    first = generator.generate_bundle(graph, db_id="sakila")
    second = generator.generate_bundle(graph, db_id="sakila")

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.actor_tool_definitions_json() == second.actor_tool_definitions_json()
    assert {tool.family for tool in first.tools} == {
        AtomicToolFamily.GET,
        AtomicToolFamily.FIND,
        AtomicToolFamily.CALC,
        AtomicToolFamily.RANK,
    }

    names = {tool.name for tool in first.tools}
    tool_by_name = {tool.name: tool for tool in first.tools}
    assert "get_customer" in names
    assert "find_customer_by_tier" in names
    assert "find_customer_by_segment" in names
    assert "find_customer_by_signup_date" in names
    assert "find_customer_by_score" in names
    assert "find_customer_by_email" not in names
    assert "find_customer_by_api_key" not in names
    assert "find_order_by_customer_id" in names
    assert "calc_order" in names
    assert "rank_customer_by_tier" in names
    assert "rank_customer_by_segment" in names
    assert "rank_customer_by_api_key" not in names
    assert "rank_customer_by_score" not in names
    assert "rank_order_by_customer_id" in names
    assert (
        tool_by_name["get_customer"].description
        == "Retrieve one customer by ID. Returns all fields or nothing."
    )
    assert (
        tool_by_name["find_order_by_customer_id"].description
        == "Find order for a given customer. Returns a list."
    )
    assert (
        tool_by_name["find_order_by_status"].description
        == "Find order entries where status matches a condition. Returns a list."
    )
    assert (
        tool_by_name["calc_order"].description
        == "Compute a statistic over order entries. Returns one number."
    )
    assert (
        tool_by_name["rank_order_by_customer_id"].description
        == "Rank customer groups across order. Returns a sorted list."
    )


def test_atomic_tool_generator_returns_all_table_fields() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(_sample_graph(), db_id="sakila")
    tool_by_name = {tool.name: tool for tool in bundle.tools}

    customer_schema = tool_by_name["get_customer"].returns_schema["anyOf"][0]
    assert customer_schema["properties"]["email"] == {"type": ["string", "null"]}
    assert customer_schema["properties"]["api_key"] == {"type": ["string", "null"]}
    assert "customer_id" in customer_schema["properties"]


def test_atomic_tool_generator_compresses_by_removing_whole_tables() -> None:
    bundle = AtomicToolGenerator(
        AtomicToolConfig(
            max_tools=18,
            bounded_result_limit=100,
            max_batch_values=128,
        )
    ).generate_bundle(_sample_graph(), db_id="sakila")

    assert len(bundle.tools) <= 18
    remaining_tables = {tool.runtime_metadata["qualified_name"] for tool in bundle.tools}
    assert remaining_tables == {"public.orders", "public.line_items"}


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
    assert "async def get_customer(conn, id):" in bundle.source
    assert (
        "async def find_customer_by_tier(conn, op, value, sort_by, direction, limit, _shuffle_seed=None):"
        in bundle.source
    )
    assert "async def calc_customer(conn, fn, metric=None, by=None, op=None, value=None):" in bundle.source
    assert (
        "async def rank_customer_by_tier(conn, fn, metric=None, direction='desc', limit=1, by=None, op=None, value=None, _shuffle_seed=None):"
        in bundle.source
    )
    assert 'raise ValueError("metric must be omitted when fn=count")' in bundle.source
    assert "MAX_BATCH_VALUES = 128" in bundle.source
    assert "MAX_BOUNDED_RESULT_LIMIT = 100" in bundle.source
    assert "FLOAT_PRECISION = 2" in bundle.source
    assert "INSERT" not in bundle.source
    assert "UPDATE" not in bundle.source
    assert "DELETE" not in bundle.source


def test_atomic_tool_generator_applies_seeded_row_shuffle_to_find_and_rank_only() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(_sample_graph(), db_id="sakila")
    tool_by_name = {tool.name: tool for tool in bundle.tools}

    assert "md5(" not in tool_by_name["get_customer"].sql
    assert "md5(" in tool_by_name["find_customer_by_tier"].sql
    assert "md5(" not in tool_by_name["calc_customer"].sql
    assert "md5(" in tool_by_name["rank_customer_by_tier"].sql


def test_atomic_tool_params_follow_family_patterns() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(_sample_graph(), db_id="sakila")
    tool_by_name = {tool.name: tool for tool in bundle.tools}

    get_schema = tool_by_name["get_customer"].params_schema
    assert get_schema["required"] == ["id"]

    text_find_schema = tool_by_name["find_customer_by_tier"].params_schema
    assert text_find_schema["required"] == ["op", "value", "sort_by", "direction", "limit"]
    assert text_find_schema["properties"]["op"]["enum"] == ["any", "eq", "in", "like"]
    assert text_find_schema["properties"]["limit"]["maximum"] == 100
    assert text_find_schema["properties"]["sort_by"]["anyOf"][0]["enum"] == [
        "customer_id",
        "tier",
        "signup_date",
        "score",
        "email",
        "segment",
        "api_key",
    ]

    numeric_find_schema = tool_by_name["find_order_by_total_amount"].params_schema
    assert numeric_find_schema["properties"]["op"]["enum"] == [
        "any",
        "eq",
        "in",
        "lt",
        "gt",
        "lte",
        "gte",
    ]

    calc_schema = tool_by_name["calc_order"].params_schema
    assert calc_schema["type"] == "object"
    assert calc_schema["required"] == ["fn", "by", "op", "value"]
    assert calc_schema["properties"]["fn"]["enum"] == ["count", "sum", "avg", "min", "max"]
    assert calc_schema["if"] == {"properties": {"fn": {"const": "count"}}}
    assert calc_schema["then"] == {"properties": {"metric": {"type": "null"}}}
    assert calc_schema["else"]["required"] == ["metric"]
    assert calc_schema["else"]["properties"]["metric"]["enum"] == ["order_id", "customer_id", "total_amount"]

    rank_schema = tool_by_name["rank_order_by_customer_id"].params_schema
    assert rank_schema["type"] == "object"
    assert rank_schema["required"] == ["fn", "direction", "limit", "by", "op", "value"]
    assert rank_schema["properties"]["fn"]["enum"] == ["count", "sum", "avg", "min", "max"]
    assert rank_schema["properties"]["direction"]["enum"] == ["asc", "desc"]
    assert rank_schema["properties"]["limit"]["maximum"] == 100
    assert rank_schema["if"] == {"properties": {"fn": {"const": "count"}}}
    assert rank_schema["then"] == {"properties": {"metric": {"type": "null"}}}
    assert rank_schema["else"]["required"] == ["metric"]
    assert rank_schema["else"]["properties"]["metric"]["enum"] == ["order_id", "customer_id", "total_amount"]


def test_atomic_tool_definition_sql_is_documented_as_display_only() -> None:
    assert AtomicToolDefinition.model_fields["sql"].description == (
        "Display/audit only SQL template. Not executed directly."
    )


def test_calc_and_rank_filter_fields_follow_graph_foreign_keys_not_column_flags() -> None:
    parent_id = ColumnProfile(
        schema_name="public",
        table_name="orders",
        column_name="parent_id",
        data_type="int4",
        ordinal_position=1,
        is_nullable=False,
        visibility="blocked",
        n_distinct=10,
    )
    order_id = ColumnProfile(
        schema_name="public",
        table_name="orders",
        column_name="order_id",
        data_type="int4",
        ordinal_position=2,
        is_nullable=False,
        visibility="blocked",
        is_primary_key=True,
        n_distinct=-1.0,
    )
    status = ColumnProfile(
        schema_name="public",
        table_name="orders",
        column_name="status",
        data_type="text",
        ordinal_position=3,
        is_nullable=False,
        visibility="user_visible",
        n_distinct=4,
    )
    graph = SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="parents",
                primary_key=("parent_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="parents",
                        column_name="parent_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        n_distinct=-1.0,
                    )
                ],
                row_estimate=10,
            ),
            TableProfile(
                schema_name="public",
                table_name="orders",
                primary_key=("order_id",),
                columns=[parent_id, order_id, status],
                row_estimate=100,
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_parent_id_fkey",
                source_schema="public",
                source_table="orders",
                source_columns=("parent_id",),
                target_schema="public",
                target_table="parents",
                target_columns=("parent_id",),
                source_is_unique=False,
                fanout_estimate=10.0,
            )
        ],
    )

    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(graph, db_id="sakila")
    tool_by_name = {tool.name: tool for tool in bundle.tools}

    calc_schema = tool_by_name["calc_order"].params_schema
    rank_schema = tool_by_name["rank_order_by_parent_id"].params_schema
    calc_by_enum = calc_schema["properties"]["by"]["anyOf"][0]["enum"]
    rank_by_enum = rank_schema["properties"]["by"]["anyOf"][0]["enum"]
    calc_value_variants = calc_schema["properties"]["value"]["anyOf"]
    rank_value_variants = rank_schema["properties"]["value"]["anyOf"]

    assert "parent_id" in calc_by_enum
    assert "parent_id" in rank_by_enum
    assert {"type": "integer"} in calc_value_variants
    assert {"type": "integer"} in rank_value_variants


def test_generated_atomic_tool_source_is_ast_valid_and_tool_functions_are_async() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(_sample_graph(), db_id="sakila")

    module = ast.parse(bundle.source)
    async_defs = {
        node.name: node for node in module.body if isinstance(node, ast.AsyncFunctionDef)
    }

    for tool in bundle.tools:
        assert tool.name in async_defs
        assert async_defs[tool.name].args.args
        assert async_defs[tool.name].args.args[0].arg == "conn"


def test_generated_atomic_tool_source_trims_fixed_width_char_padding_only() -> None:
    bundle = AtomicToolGenerator(AtomicToolConfig()).generate_bundle(_sample_graph(), db_id="sakila")
    namespace: dict[str, object] = {}
    exec(bundle.source, namespace)

    row_to_dict = namespace["_row_to_dict"]
    assert callable(row_to_dict)

    normalized = row_to_dict(
        {"code": "AB   ", "title": "Seoul   ", "note": "  keep-leading"},
        {
            "column_types": {
                "code": "bpchar",
                "title": "varchar",
                "note": "character varying",
            }
        },
    )

    assert normalized == {
        "code": "AB",
        "title": "Seoul   ",
        "note": "  keep-leading",
    }


def test_generated_atomic_tool_sql_templates_parse_under_postgres_dialect() -> None:
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
