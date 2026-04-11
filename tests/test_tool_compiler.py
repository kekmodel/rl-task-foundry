from __future__ import annotations

import pytest

from rl_task_foundry.config.load import load_config
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.schema.path_catalog import build_path_catalog
from rl_task_foundry.tools.compiler import compile_all_tool_levels, compile_path_tools


def _synthetic_graph() -> tuple[SchemaGraph, str]:
    graph = SchemaGraph(
        tables=[
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
                        column_name="shipment_id",
                        data_type="int4",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                ],
                row_estimate=1000,
            ),
            TableProfile(
                schema_name="public",
                table_name="shipments",
                primary_key=("shipment_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="shipment_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="status",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="amount",
                        data_type="numeric",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="phone",
                        data_type="text",
                        ordinal_position=4,
                        is_nullable=True,
                        visibility="internal",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="shipped_at",
                        data_type="timestamp without time zone",
                        ordinal_position=5,
                        is_nullable=True,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="api_key",
                        data_type="text",
                        ordinal_position=6,
                        is_nullable=True,
                        visibility="blocked",
                    ),
                ],
                row_estimate=300,
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_shipments_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("shipment_id",),
                target_schema="public",
                target_table="shipments",
                target_columns=("shipment_id",),
                source_is_unique=False,
                fanout_estimate=3.0,
            )
        ],
    )
    return graph, "orders.shipments"


def test_compile_path_tools_enforces_stable_semantics_across_levels():
    graph, path_id = _synthetic_graph()
    path = build_path_catalog(graph, max_hops=2).get(path_id)

    l1 = compile_path_tools(graph, path, tool_level=1, label_tier="B", max_list_cardinality=7)
    l2 = compile_path_tools(graph, path, tool_level=2, label_tier="B", max_list_cardinality=7)

    assert l1.path_id == path_id
    assert l2.path_id == path_id
    assert l1.tool_level == 1
    assert l2.tool_level == 2
    assert {tool.name_source for tool in l1.tools} == {"rule_based"}
    assert {tool.name_source for tool in l2.tools} == {"fallback_alias"}

    l1_by_semantic = {tool.semantic_key: tool for tool in l1.tools}
    l2_by_semantic = {tool.semantic_key: tool for tool in l2.tools}

    assert {
        tool.kind for tool in l1.tools
    } == {"lookup", "list_related", "count", "exists", "aggregate", "timeline"}
    assert l1_by_semantic[f"{path_id}:lookup"].output_fields == ["status", "amount", "phone", "shipped_at"]
    assert l1_by_semantic[f"{path_id}:list_related"].output_fields == ["status", "amount", "phone", "shipped_at"]
    assert "LIMIT 7" in l1_by_semantic[f"{path_id}:list_related"].sql_template
    assert "ORDER BY" in l1_by_semantic[f"{path_id}:list_related"].sql_template
    assert "api_key" not in l1_by_semantic[f"{path_id}:lookup"].output_fields

    stable_semantics = [
        f"{path_id}:lookup",
        f"{path_id}:list_related",
        f"{path_id}:count",
        f"{path_id}:count_related:orders:orders_shipments_fk",
        f"{path_id}:exists",
        f"{path_id}:aggregate:sum:amount",
        f"{path_id}:aggregate:avg:amount",
        f"{path_id}:aggregate:min:amount",
        f"{path_id}:aggregate:max:amount",
        f"{path_id}:timeline:shipped_at",
    ]
    for semantic_key in stable_semantics:
        assert l1_by_semantic[semantic_key].semantic_key == l2_by_semantic[semantic_key].semantic_key
        assert l1_by_semantic[semantic_key].sql_template == l2_by_semantic[semantic_key].sql_template
        assert [parameter.name for parameter in l1_by_semantic[semantic_key].parameters] == ["anchor_order_id"]
        assert l1_by_semantic[semantic_key].output_fields == l2_by_semantic[semantic_key].output_fields

    assert l1_by_semantic[f"{path_id}:lookup"].name != l2_by_semantic[f"{path_id}:lookup"].name
    assert 'JOIN "public"."orders" AS r2' in l1_by_semantic[
        f"{path_id}:count_related:orders:orders_shipments_fk"
    ].sql_template
    assert l1_by_semantic[f"{path_id}:aggregate:sum:amount"].output_fields == ["sum_amount"]
    assert 'ROUND((SUM(t1."amount"))::numeric, 6) AS "sum_amount"' in l1_by_semantic[
        f"{path_id}:aggregate:sum:amount"
    ].sql_template
    assert 'ROUND((AVG(t1."amount"))::numeric, 6) AS "avg_amount"' in l1_by_semantic[
        f"{path_id}:aggregate:avg:amount"
    ].sql_template
    assert 'AND t1."amount" IS NOT NULL' in l1_by_semantic[f"{path_id}:aggregate:sum:amount"].sql_template
    assert 'AND t1."amount" IS NOT NULL' in l1_by_semantic[f"{path_id}:aggregate:min:amount"].sql_template
    assert 'DISTINCT ON (t1."status", t1."amount", t1."phone", t1."shipped_at")' in l1_by_semantic[
        f"{path_id}:timeline:shipped_at"
    ].sql_template
    assert 't1."shipped_at" DESC' in l1_by_semantic[f"{path_id}:timeline:shipped_at"].sql_template
    assert l2_by_semantic[f"{path_id}:lookup"].name.startswith("inspect_")
    assert l2_by_semantic[f"{path_id}:aggregate:sum:amount"].name.endswith("_summary")
    assert l2_by_semantic[f"{path_id}:timeline:shipped_at"].name.startswith("review_")
    assert path_id not in l1_by_semantic[f"{path_id}:lookup"].description
    assert f"{path_id}:aggregate:sum:shipment_id" not in l1_by_semantic
    assert f"{path_id}:aggregate:sum:shipment_id" not in l2_by_semantic


def test_compile_path_tools_respects_aggregate_and_timeline_flags():
    graph, path_id = _synthetic_graph()
    path = build_path_catalog(graph, max_hops=2).get(path_id)

    bundle = compile_path_tools(
        graph,
        path,
        tool_level=1,
        label_tier="A",
        max_list_cardinality=5,
        allow_aggregates=False,
        allow_timelines=False,
    )

    assert {tool.kind for tool in bundle.tools} == {"lookup", "list_related", "count", "exists"}


@pytest.mark.asyncio
async def test_compile_all_tool_levels_for_real_sakila_path():
    config = load_config("rl_task_foundry.yaml")
    introspector = PostgresSchemaIntrospector(
        database=config.database,
        default_visibility=config.privacy.default_visibility,
        visibility_overrides=config.privacy.visibility_overrides,
    )
    graph = await introspector.introspect()
    catalog = build_path_catalog(graph, max_hops=config.tool_compiler.max_hops)

    levels = compile_all_tool_levels(
        graph,
        catalog,
        label_tier=config.task_composer.label_tier,
        max_list_cardinality=config.tool_compiler.max_list_cardinality,
        allow_aggregates=config.tool_compiler.allow_aggregates,
        allow_timelines=config.tool_compiler.allow_timelines,
    )
    assert set(levels) == {1, 2}

    l1_customer_address_city = next(bundle for bundle in levels[1] if bundle.path_id == "customer.address.city")
    l2_customer_address_city = next(bundle for bundle in levels[2] if bundle.path_id == "customer.address.city")

    l1_lookup = next(tool for tool in l1_customer_address_city.tools if tool.kind == "lookup")
    l1_list_related = next(tool for tool in l1_customer_address_city.tools if tool.kind == "list_related")
    l2_lookup = next(tool for tool in l2_customer_address_city.tools if tool.kind == "lookup")

    assert l1_lookup.sql_template == l2_lookup.sql_template
    assert f"LIMIT {config.tool_compiler.max_list_cardinality}" in l1_list_related.sql_template
    assert all(field != "api_key" for field in l1_lookup.output_fields)
    assert l1_lookup.name.startswith("get_")
    assert l1_list_related.name.startswith("list_")


def test_compile_path_tools_tier_a_excludes_aggregate_and_timeline_capabilities():
    graph, path_id = _synthetic_graph()
    path = build_path_catalog(graph, max_hops=2).get(path_id)

    bundle = compile_path_tools(
        graph,
        path,
        tool_level=1,
        label_tier="A",
        max_list_cardinality=7,
        allow_aggregates=True,
        allow_timelines=True,
    )

    assert {tool.kind for tool in bundle.tools} == {"lookup", "list_related", "count", "exists"}


def test_compile_path_tools_tier_b_enables_aggregate_and_timeline_capabilities():
    graph, path_id = _synthetic_graph()
    path = build_path_catalog(graph, max_hops=2).get(path_id)

    bundle = compile_path_tools(
        graph,
        path,
        tool_level=1,
        label_tier="B",
        max_list_cardinality=7,
        allow_aggregates=True,
        allow_timelines=True,
    )

    assert "aggregate" in {tool.kind for tool in bundle.tools}
    assert "timeline" in {tool.kind for tool in bundle.tools}


def test_compile_path_tools_uses_configured_float_precision_for_aggregate_sql():
    graph, path_id = _synthetic_graph()
    path = build_path_catalog(graph, max_hops=2).get(path_id)

    bundle = compile_path_tools(
        graph,
        path,
        tool_level=1,
        label_tier="B",
        max_list_cardinality=7,
        allow_aggregates=True,
        allow_timelines=False,
        float_precision=3,
    )

    by_semantic = {tool.semantic_key: tool for tool in bundle.tools}
    assert 'ROUND((SUM(t1."amount"))::numeric, 3) AS "sum_amount"' in by_semantic[
        f"{path_id}:aggregate:sum:amount"
    ].sql_template
    assert 'ROUND((AVG(t1."amount"))::numeric, 3) AS "avg_amount"' in by_semantic[
        f"{path_id}:aggregate:avg:amount"
    ].sql_template
