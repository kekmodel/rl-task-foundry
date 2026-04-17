"""Unit tests for tooling.composer.schema_map.

Pure-snapshot primitive — no DB required. Exercises BFS selection,
hub/bridge classification, and the JSON payload shape.
"""

from __future__ import annotations

import pytest

from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.tooling.common import snapshot_from_graph
from rl_task_foundry.tooling.composer import schema_map


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


def _fk(
    name: str,
    src_table: str,
    src_col: str,
    tgt_table: str,
    tgt_col: str,
) -> ForeignKeyEdge:
    return ForeignKeyEdge(
        constraint_name=name,
        source_schema="public",
        source_table=src_table,
        source_columns=(src_col,),
        target_schema="public",
        target_table=tgt_table,
        target_columns=(tgt_col,),
    )


def _graph():
    customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[
            _column("customer", "customer_id", is_primary_key=True),
            _column("customer", "store_id", is_foreign_key=True),
        ],
        primary_key=("customer_id",),
    )
    store = TableProfile(
        schema_name="public",
        table_name="store",
        columns=[_column("store", "store_id", is_primary_key=True)],
        primary_key=("store_id",),
    )
    rental = TableProfile(
        schema_name="public",
        table_name="rental",
        columns=[
            _column("rental", "rental_id", is_primary_key=True),
            _column("rental", "customer_id", is_foreign_key=True),
            _column("rental", "inventory_id", is_foreign_key=True),
        ],
        primary_key=("rental_id",),
    )
    inventory = TableProfile(
        schema_name="public",
        table_name="inventory",
        columns=[
            _column("inventory", "inventory_id", is_primary_key=True),
            _column("inventory", "film_id", is_foreign_key=True),
        ],
        primary_key=("inventory_id",),
    )
    film = TableProfile(
        schema_name="public",
        table_name="film",
        columns=[_column("film", "film_id", is_primary_key=True)],
        primary_key=("film_id",),
    )
    payment = TableProfile(
        schema_name="public",
        table_name="payment",
        columns=[
            _column("payment", "payment_id", is_primary_key=True),
            _column("payment", "customer_id", is_foreign_key=True),
            _column("payment", "rental_id", is_foreign_key=True),
        ],
        primary_key=("payment_id",),
    )
    return SchemaGraph(
        tables=[customer, store, rental, inventory, film, payment],
        edges=[
            _fk("customer_store", "customer", "store_id", "store", "store_id"),
            _fk("rental_customer", "rental", "customer_id", "customer", "customer_id"),
            _fk("rental_inventory", "rental", "inventory_id", "inventory", "inventory_id"),
            _fk("inventory_film", "inventory", "film_id", "film", "film_id"),
            _fk("payment_customer", "payment", "customer_id", "customer", "customer_id"),
            _fk("payment_rental", "payment", "rental_id", "rental", "rental_id"),
        ],
    )


def test_schema_map_returns_whole_schema_when_root_table_omitted():
    snapshot = snapshot_from_graph(_graph())
    payload = schema_map(snapshot)
    assert payload["root_table"] is None
    tables = payload["tables"]
    assert isinstance(tables, list)
    names = [table["name"] for table in tables if isinstance(table, dict)]
    assert set(names) == {
        "customer",
        "store",
        "rental",
        "inventory",
        "film",
        "payment",
    }


def test_schema_map_bfs_depth_one_picks_direct_neighbors_only():
    snapshot = snapshot_from_graph(_graph())
    payload = schema_map(snapshot, root_table="rental", depth=1)
    tables = payload["tables"]
    assert isinstance(tables, list)
    names = {table["name"] for table in tables if isinstance(table, dict)}
    # rental reaches customer and inventory forward; payment reaches rental
    # reverse. film requires 2 hops and should NOT appear at depth=1.
    assert "rental" in names
    assert "customer" in names
    assert "inventory" in names
    assert "payment" in names
    assert "film" not in names


def test_schema_map_bfs_depth_two_reaches_film_and_store():
    snapshot = snapshot_from_graph(_graph())
    payload = schema_map(snapshot, root_table="rental", depth=2)
    tables = payload["tables"]
    assert isinstance(tables, list)
    names = {table["name"] for table in tables if isinstance(table, dict)}
    assert "film" in names  # rental → inventory → film
    assert "store" in names  # rental → customer → store


def test_schema_map_depth_zero_returns_only_root():
    snapshot = snapshot_from_graph(_graph())
    payload = schema_map(snapshot, root_table="rental", depth=0)
    tables = payload["tables"]
    assert isinstance(tables, list)
    names = {table["name"] for table in tables if isinstance(table, dict)}
    assert names == {"rental"}


def test_schema_map_rejects_negative_depth():
    snapshot = snapshot_from_graph(_graph())
    with pytest.raises(ValueError):
        schema_map(snapshot, root_table="rental", depth=-1)


def test_schema_map_rejects_unknown_root_table():
    snapshot = snapshot_from_graph(_graph())
    with pytest.raises(KeyError):
        schema_map(snapshot, root_table="nonexistent", depth=1)


def test_schema_map_hub_tables_include_customer_and_rental():
    snapshot = snapshot_from_graph(_graph())
    payload = schema_map(snapshot)
    hubs = payload["hub_tables"]
    assert isinstance(hubs, list)
    # customer is referenced by rental + payment and points at store (degree 3).
    # rental is referenced by payment and points at customer + inventory
    # (degree 3). Both should beat leaf tables like film (degree 1).
    assert "customer" in hubs
    assert "rental" in hubs


def test_schema_map_bridge_tables_lists_junctions():
    snapshot = snapshot_from_graph(_graph())
    payload = schema_map(snapshot)
    bridges = payload["bridge_tables"]
    assert isinstance(bridges, list)
    # rental has 2 outgoing FKs (customer, inventory); payment has 2
    # (customer, rental); they are the m:n-style junctions here.
    assert "rental" in bridges
    assert "payment" in bridges
    # film / store are leaves and must not appear.
    assert "film" not in bridges
    assert "store" not in bridges


def test_schema_map_edges_scoped_to_included_tables():
    snapshot = snapshot_from_graph(_graph())
    payload = schema_map(snapshot, root_table="rental", depth=1)
    edges = payload["edges"]
    assert isinstance(edges, list)
    for edge in edges:
        assert isinstance(edge, dict)
        # rental neighborhood at depth=1: rental, customer, inventory, payment.
        # store / film edges must be excluded.
        assert edge["source_table"] != "inventory" or edge["target_table"] != "film"


def test_schema_map_typed_edges_cover_forward_and_reverse():
    snapshot = snapshot_from_graph(_graph())
    payload = schema_map(snapshot, root_table="rental", depth=1)
    typed = payload["typed_edges"]
    assert isinstance(typed, dict)
    rental_edges = typed["rental"]
    assert isinstance(rental_edges, list)
    directions = {
        entry["direction"]
        for entry in rental_edges
        if isinstance(entry, dict)
    }
    assert "forward" in directions  # rental.customer_id->customer
    assert "reverse" in directions  # payment<-rental
