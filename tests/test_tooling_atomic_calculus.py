"""Unit tests for calculus primitives that don't need a live DB.

Integration coverage for the data paths (take/count/aggregate/group_top
execution) lives in tests/test_tooling_atomic_integration.py which runs
against sakila.
"""

from __future__ import annotations

import pytest

from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.tooling.atomic import (
    AtomicSession,
    CursorStore,
    IntersectNode,
    ViaNode,
    intersect,
    rows_via,
    rows_where,
)
from rl_task_foundry.tooling.common import snapshot_from_graph


class _StubConnection:
    async def fetch(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("connection must not be used for plan-only tests")

    async def fetchrow(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("connection must not be used for plan-only tests")


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
            _column("customer", "active"),
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


def _build_session() -> AtomicSession:
    return AtomicSession(
        snapshot=_snapshot(),
        connection=_StubConnection(),
        store=CursorStore(),
    )


def test_rows_via_resolves_label_and_wraps_in_via_node():
    session = _build_session()
    cursor = rows_where(
        session,
        table="rental",
        column="customer_id",
        op="eq",
        value=45,
    )
    projected = rows_via(
        session,
        cursor=cursor,
        edge_label="rental.customer_id->customer",
    )
    plan = session.store.resolve(projected)
    assert isinstance(plan, ViaNode)
    assert plan.target_table == "customer"


def test_rows_via_rejects_label_that_does_not_originate_at_cursor_table():
    session = _build_session()
    cursor = rows_where(
        session,
        table="customer",
        column="store_id",
        op="eq",
        value=1,
    )
    with pytest.raises(KeyError):
        rows_via(
            session,
            cursor=cursor,
            edge_label="rental.customer_id->customer",
        )


def test_intersect_requires_matching_target_tables():
    session = _build_session()
    customers = rows_where(
        session, table="customer", column="store_id", op="eq", value=1
    )
    rentals = rows_where(
        session,
        table="rental",
        column="customer_id",
        op="eq",
        value=45,
    )
    with pytest.raises(ValueError, match="same table"):
        intersect(session, left=customers, right=rentals)


def test_intersect_wraps_plans_in_intersect_node():
    session = _build_session()
    a = rows_where(
        session, table="customer", column="store_id", op="eq", value=1
    )
    b = rows_where(
        session, table="customer", column="active", op="eq", value=1
    )
    combined = intersect(session, left=a, right=b)
    plan = session.store.resolve(combined)
    assert isinstance(plan, IntersectNode)
    assert plan.target_table == "customer"
