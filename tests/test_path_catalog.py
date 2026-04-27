from __future__ import annotations

import pytest

from rl_task_foundry.config.load import load_config
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.schema.path_catalog import build_path_catalog


def _table(
    name: str,
    *,
    schema_name: str = "public",
    row_estimate: int | None = None,
    foreign_key_columns: tuple[str, ...] = (),
    nullable_fk_columns: set[str] | None = None,
) -> TableProfile:
    nullable_fk_columns = nullable_fk_columns or set()
    columns = [
        ColumnProfile(
            schema_name=schema_name,
            table_name=name,
            column_name=f"{name}_id",
            data_type="int4",
            ordinal_position=1,
            is_nullable=False,
            visibility="blocked",
            is_primary_key=True,
            is_unique=True,
        )
    ]
    for index, foreign_key_column in enumerate(foreign_key_columns, start=2):
        columns.append(
            ColumnProfile(
                schema_name=schema_name,
                table_name=name,
                column_name=foreign_key_column,
                data_type="int4",
                ordinal_position=index,
                is_nullable=foreign_key_column in nullable_fk_columns,
                visibility="blocked",
                is_foreign_key=True,
            )
        )
    return TableProfile(
        schema_name=schema_name,
        table_name=name,
        row_estimate=row_estimate,
        primary_key=(f"{name}_id",),
        columns=columns,
    )


def test_build_path_catalog_detects_shortcuts_and_features():
    graph = SchemaGraph(
        tables=[
            _table(
                "orders",
                row_estimate=1000,
                foreign_key_columns=("shipments_fk", "carriers_fk"),
            ),
            _table(
                "shipments",
                row_estimate=300,
                foreign_key_columns=("carriers_fk",),
                nullable_fk_columns={"carriers_fk"},
            ),
            _table("carriers", row_estimate=20),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_shipments_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("shipments_fk",),
                target_schema="public",
                target_table="shipments",
                target_columns=("shipments_id",),
                source_is_unique=False,
                fanout_estimate=3.0,
            ),
            ForeignKeyEdge(
                constraint_name="shipments_carriers_fk",
                source_schema="public",
                source_table="shipments",
                source_columns=("carriers_fk",),
                target_schema="public",
                target_table="carriers",
                target_columns=("carriers_id",),
                source_is_unique=False,
                fanout_estimate=2.0,
            ),
            ForeignKeyEdge(
                constraint_name="orders_carriers_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("carriers_fk",),
                target_schema="public",
                target_table="carriers",
                target_columns=("carriers_id",),
                source_is_unique=False,
                fanout_estimate=1.0,
            ),
        ],
    )

    catalog = build_path_catalog(graph, max_hops=3)

    direct = catalog.get("orders.carriers")
    indirect = catalog.get("orders.shipments.carriers")

    assert direct.hop_count == 1
    assert indirect.hop_count == 2
    assert indirect.shortcut_candidates == ["orders.carriers"]
    assert indirect.difficulty_features["shortcut_count"] == 1
    assert indirect.difficulty_features["fanout_max"] == 3.0
    assert indirect.difficulty_features["fanout_product"] == 6.0
    assert indirect.difficulty_features["has_nullable_hop"] is True


async def _load_pagila_catalog():
    config = load_config("rl_task_foundry.yaml")
    introspector = PostgresSchemaIntrospector(
        database=config.database,
        default_visibility=config.visibility.default_visibility,
        visibility_overrides=config.visibility.visibility_overrides,
    )
    graph = await introspector.introspect()
    return build_path_catalog(graph, max_hops=4)


@pytest.mark.asyncio
async def test_build_path_catalog_reads_real_pagila_paths():
    catalog = await _load_pagila_catalog()
    customer_paths = catalog.for_root("customer")
    path_ids = {path.path_id for path in customer_paths}
    assert "customer.address.city" in path_ids
    assert "customer.store" in path_ids

    address_city = catalog.get("customer.address.city")
    assert address_city.hop_count == 2
    assert address_city.difficulty_features["required_hops"] == 2
    assert "fanout_product" in address_city.difficulty_features
