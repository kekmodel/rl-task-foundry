from __future__ import annotations

from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.schema.path_catalog import build_path_catalog
from rl_task_foundry.tools.compiler import compile_path_tools
from rl_task_foundry.tools.naming_eval import evaluate_tool_bundle_naming


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
                        column_name="amount",
                        data_type="numeric",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="shipped_at",
                        data_type="timestamp without time zone",
                        ordinal_position=3,
                        is_nullable=True,
                        visibility="user_visible",
                    ),
                ],
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
            )
        ],
    )
    return graph, "orders.shipments"


def test_evaluate_tool_bundle_naming_distinguishes_l1_and_l2():
    graph, path_id = _synthetic_graph()
    path = build_path_catalog(graph, max_hops=2).get(path_id)

    l1 = compile_path_tools(graph, path, tool_level=1, label_tier="A", max_list_cardinality=5)
    l2 = compile_path_tools(graph, path, tool_level=2, label_tier="A", max_list_cardinality=5)

    l1_eval = evaluate_tool_bundle_naming(graph, path, l1)
    l2_eval = evaluate_tool_bundle_naming(graph, path, l2)

    assert l1_eval.passes_hard_constraints is True
    assert l2_eval.passes_hard_constraints is True
    assert l1_eval.tools_with_raw_table_hits > 0
    assert l2_eval.schema_opacity_score > l1_eval.schema_opacity_score
    assert not l1_eval.policy_violations
    assert l2_eval.tools_with_raw_table_hits <= l1_eval.tools_with_raw_table_hits
