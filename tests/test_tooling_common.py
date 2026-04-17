"""Unit tests for tooling/common: schema snapshot, SQL helpers, edges."""

from __future__ import annotations

import pytest

from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.tooling.common import (
    EdgeDirection,
    SchemaSnapshot,
    available_edges,
    coerce_param,
    quote_ident,
    quote_table,
    resolve_edge,
    snapshot_from_graph,
)


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


def _toy_graph() -> SchemaGraph:
    customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[_column("customer", "customer_id", is_primary_key=True)],
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
        constraint_name="rental_customer_fk",
        source_schema="public",
        source_table="rental",
        source_columns=("customer_id",),
        target_schema="public",
        target_table="customer",
        target_columns=("customer_id",),
    )
    return SchemaGraph(tables=[customer, rental], edges=[edge])


def test_snapshot_from_graph_preserves_columns_and_edges():
    snapshot = snapshot_from_graph(_toy_graph())
    assert isinstance(snapshot, SchemaSnapshot)
    assert snapshot.table_names() == ("customer", "rental")
    rental = snapshot.table("rental")
    assert rental.primary_key == ("rental_id",)
    assert rental.column("customer_id").is_foreign_key is True
    assert snapshot.edges_from("rental")[0].target_table == "customer"
    assert snapshot.edges_to("customer")[0].source_table == "rental"


def test_available_edges_lists_both_directions():
    snapshot = snapshot_from_graph(_toy_graph())
    rental_edges = available_edges(snapshot, "rental")
    customer_edges = available_edges(snapshot, "customer")
    assert any(edge.direction is EdgeDirection.FORWARD for edge in rental_edges)
    assert any(
        edge.direction is EdgeDirection.REVERSE for edge in customer_edges
    )
    forward = next(
        edge
        for edge in rental_edges
        if edge.direction is EdgeDirection.FORWARD
    )
    assert forward.origin_table == "rental"
    assert forward.destination_table == "customer"
    reverse = next(
        edge
        for edge in customer_edges
        if edge.direction is EdgeDirection.REVERSE
    )
    assert reverse.origin_table == "customer"
    assert reverse.destination_table == "rental"


def test_resolve_edge_by_label_round_trips():
    snapshot = snapshot_from_graph(_toy_graph())
    edges = available_edges(snapshot, "rental")
    forward = next(
        edge for edge in edges if edge.direction is EdgeDirection.FORWARD
    )
    resolved = resolve_edge(snapshot, "rental", forward.label)
    assert resolved == forward
    with pytest.raises(KeyError):
        resolve_edge(snapshot, "rental", "nonexistent")


def test_sql_identifier_quoting_handles_embedded_quotes():
    assert quote_ident("simple") == '"simple"'
    assert quote_ident('quo"te') == '"quo""te"'
    assert quote_table("public", "rental") == '"public"."rental"'


def test_coerce_param_allows_scalars_and_lists_and_rejects_others():
    assert coerce_param(None) is None
    assert coerce_param(42) == 42
    assert coerce_param("text") == "text"
    assert coerce_param([1, 2, 3]) == [1, 2, 3]
    with pytest.raises(TypeError):
        coerce_param({"bad": 1})
    with pytest.raises(TypeError):
        coerce_param((1, 2))
