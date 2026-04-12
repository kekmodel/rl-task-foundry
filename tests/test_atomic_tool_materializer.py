from __future__ import annotations

import json
from pathlib import Path

from rl_task_foundry.config import load_config
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.synthesis.atomic_tool_materializer import AtomicToolMaterializer
from rl_task_foundry.synthesis.atomic_tools import AtomicToolGenerator


def _sample_graph() -> SchemaGraph:
    customer_id = ColumnProfile(
        schema_name="public",
        table_name="customer",
        column_name="customer_id",
        data_type="integer",
        ordinal_position=1,
        is_nullable=False,
        visibility="user_visible",
        is_primary_key=True,
    )
    customer_name = ColumnProfile(
        schema_name="public",
        table_name="customer",
        column_name="name",
        data_type="text",
        ordinal_position=2,
        is_nullable=False,
        visibility="user_visible",
    )
    order_id = ColumnProfile(
        schema_name="public",
        table_name="orders",
        column_name="order_id",
        data_type="integer",
        ordinal_position=1,
        is_nullable=False,
        visibility="internal",
        is_primary_key=True,
    )
    order_customer_id = ColumnProfile(
        schema_name="public",
        table_name="orders",
        column_name="customer_id",
        data_type="integer",
        ordinal_position=2,
        is_nullable=False,
        visibility="internal",
        is_foreign_key=True,
    )
    return SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="customer",
                columns=[customer_id, customer_name],
                primary_key=("customer_id",),
                row_estimate=5,
            ),
            TableProfile(
                schema_name="public",
                table_name="orders",
                columns=[order_id, order_customer_id],
                primary_key=("order_id",),
                row_estimate=20,
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_customer_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("customer_id",),
                target_schema="public",
                target_table="customer",
                target_columns=("customer_id",),
                source_is_unique=False,
                fanout_estimate=4.0,
            )
        ],
    )


def test_atomic_tool_materializer_writes_bundle_files(tmp_path: Path) -> None:
    config = load_config("rl_task_foundry.yaml")
    bundle = AtomicToolGenerator(config.atomic_tools).generate_bundle(_sample_graph(), db_id="sakila")
    materializer = AtomicToolMaterializer(root_dir=tmp_path / "databases")

    materialization = materializer.materialize_bundle(bundle)

    assert materialization.bundle_dir == tmp_path / "databases" / "sakila"
    assert materialization.source_path.exists()
    assert materialization.definitions_path.exists()
    assert materialization.source_path.read_text(encoding="utf-8") == bundle.source
    assert json.loads(materialization.definitions_path.read_text(encoding="utf-8")) == (
        bundle.actor_tool_definitions()
    )


def test_atomic_tool_materializer_reads_actor_tool_definitions(tmp_path: Path) -> None:
    config = load_config("rl_task_foundry.yaml")
    bundle = AtomicToolGenerator(config.atomic_tools).generate_bundle(_sample_graph(), db_id="sakila")
    materializer = AtomicToolMaterializer(root_dir=tmp_path / "databases")
    materializer.materialize_bundle(bundle)

    payload = materializer.read_actor_tool_definitions(db_id="sakila")

    assert payload == bundle.actor_tool_definitions()
