"""Unit tests for sql_compile: CursorPlan → SQL translation."""

from __future__ import annotations

import pytest

from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.tooling.atomic.cursor import (
    IntersectNode,
    OrderNode,
    ViaNode,
    WhereNode,
)
from rl_task_foundry.tooling.atomic.sql_compile import (
    compile_aggregate,
    compile_count,
    compile_group_top,
    compile_read,
    compile_take,
)
from rl_task_foundry.tooling.common import (
    EdgeDirection,
    TypedEdge,
    resolve_edge,
    snapshot_from_graph,
)


def _column(
    table: str,
    name: str,
    data_type: str = "int4",
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


def _rental_snapshot():
    rental = TableProfile(
        schema_name="public",
        table_name="rental",
        columns=[
            _column("rental", "rental_id", is_primary_key=True),
            _column("rental", "customer_id"),
            _column("rental", "rental_date", data_type="timestamp"),
        ],
        primary_key=("rental_id",),
    )
    return snapshot_from_graph(SchemaGraph(tables=[rental], edges=[]))


def _customer_rental_snapshot():
    customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[
            _column("customer", "customer_id", is_primary_key=True),
            _column("customer", "store_id"),
            _column("customer", "first_name", data_type="text"),
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
            _column("rental", "inventory_id"),
            _column("rental", "rental_date", data_type="timestamp"),
        ],
        primary_key=("rental_id",),
    )
    payment = TableProfile(
        schema_name="public",
        table_name="payment",
        columns=[
            _column("payment", "payment_id", is_primary_key=True),
            _column("payment", "customer_id", is_foreign_key=True),
            _column("payment", "amount", data_type="numeric"),
        ],
        primary_key=("payment_id",),
    )
    rental_fk = ForeignKeyEdge(
        constraint_name="rental_customer_fk",
        source_schema="public",
        source_table="rental",
        source_columns=("customer_id",),
        target_schema="public",
        target_table="customer",
        target_columns=("customer_id",),
    )
    payment_fk = ForeignKeyEdge(
        constraint_name="payment_customer_fk",
        source_schema="public",
        source_table="payment",
        source_columns=("customer_id",),
        target_schema="public",
        target_table="customer",
        target_columns=("customer_id",),
    )
    return snapshot_from_graph(
        SchemaGraph(
            tables=[customer, rental, payment],
            edges=[rental_fk, payment_fk],
        )
    )


def _composite_order_snapshot():
    order = TableProfile(
        schema_name="public",
        table_name="order",
        columns=[
            _column("order", "tenant_id", is_primary_key=True),
            _column("order", "order_id", is_primary_key=True),
            _column("order", "status", data_type="text"),
        ],
        primary_key=("tenant_id", "order_id"),
    )
    line_item = TableProfile(
        schema_name="public",
        table_name="line_item",
        columns=[
            _column("line_item", "tenant_id", is_foreign_key=True),
            _column("line_item", "order_id", is_foreign_key=True),
            _column("line_item", "line_no", is_primary_key=True),
            _column("line_item", "sku", data_type="text"),
        ],
        primary_key=("tenant_id", "order_id", "line_no"),
    )
    return snapshot_from_graph(
        SchemaGraph(
            tables=[order, line_item],
            edges=[
                ForeignKeyEdge(
                    constraint_name="line_item_order_fk",
                    source_schema="public",
                    source_table="line_item",
                    source_columns=("tenant_id", "order_id"),
                    target_schema="public",
                    target_table="order",
                    target_columns=("tenant_id", "order_id"),
                )
            ],
        )
    )


def test_compile_take_emits_parameterized_sql_with_pk_tiebreak():
    snapshot = _rental_snapshot()
    plan = WhereNode(
        table="rental", column="customer_id", op="eq", value=45
    )
    compiled = compile_take(snapshot, plan, limit=3)
    assert "SELECT t.\"rental_id\" AS id" in compiled.sql
    assert "FROM \"public\".\"rental\"" in compiled.sql
    assert "WHERE t.\"customer_id\" = $1" in compiled.sql
    assert "ORDER BY t.\"rental_id\" ASC" in compiled.sql
    assert compiled.sql.endswith("LIMIT 3")
    assert compiled.params == (45,)


def test_compile_take_honors_order_annotation_before_tiebreak():
    snapshot = _rental_snapshot()
    base = WhereNode(
        table="rental", column="customer_id", op="eq", value=45
    )
    ordered = OrderNode(source=base, column="rental_date", direction="desc")
    compiled = compile_take(snapshot, ordered, limit=3)
    clauses = compiled.sql.split("ORDER BY ")[1].split("LIMIT")[0].strip()
    assert clauses.startswith("t.\"rental_date\" DESC")
    assert "t.\"rental_id\" ASC" in clauses


def test_compile_take_orders_by_forward_related_key():
    snapshot = _customer_rental_snapshot()
    base = WhereNode(
        table="rental", column="customer_id", op="eq", value=45
    )
    related_tie_break = OrderNode(
        source=base,
        path=("rental.customer_id->customer",),
        column="first_name",
        direction="asc",
    )
    ordered = OrderNode(
        source=related_tie_break,
        column="rental_date",
        direction="asc",
    )

    compiled = compile_take(snapshot, ordered, limit=5)

    assert "GROUP BY base.id" in compiled.sql
    assert (
        "LEFT JOIN \"public\".\"customer\" AS ord_1_0 "
        "ON ord_1_0.\"customer_id\" = tgt.\"customer_id\""
    ) in compiled.sql
    clauses = compiled.sql.split("ORDER BY ")[1].split("LIMIT")[0].strip()
    assert clauses.startswith(
        "MIN(tgt.\"rental_date\") ASC, MIN(ord_1_0.\"first_name\") ASC"
    )
    assert "base.id ASC" in clauses
    assert compiled.params == (45,)


def test_compile_take_rejects_reverse_related_order_path():
    snapshot = _customer_rental_snapshot()
    base = WhereNode(table="customer", column="store_id", op="eq", value=1)
    ordered = OrderNode(
        source=base,
        path=("customer<-rental.customer_id",),
        column="rental_date",
        direction="asc",
    )

    with pytest.raises(ValueError, match="forward relation"):
        compile_take(snapshot, ordered, limit=5)


def test_compile_take_supports_in_op_with_array_cast():
    snapshot = _rental_snapshot()
    plan = WhereNode(
        table="rental",
        column="customer_id",
        op="in",
        value=[45, 46, 47],
    )
    compiled = compile_take(snapshot, plan, limit=5)
    assert "= ANY($1::int4[])" in compiled.sql
    assert compiled.params == ([45, 46, 47],)


def test_compile_take_supports_neq_and_nullary_ops():
    snapshot = _rental_snapshot()
    neq = compile_take(
        snapshot,
        WhereNode(table="rental", column="customer_id", op="neq", value=45),
        limit=5,
    )
    assert "WHERE t.\"customer_id\" <> $1" in neq.sql
    assert neq.params == (45,)

    is_null = compile_take(
        snapshot,
        WhereNode(table="rental", column="customer_id", op="is_null", value=None),
        limit=5,
    )
    assert "WHERE t.\"customer_id\" IS NULL" in is_null.sql
    assert is_null.params == ()


def test_compile_take_rejects_null_for_binary_ops():
    snapshot = _rental_snapshot()
    plan = WhereNode(
        table="rental", column="customer_id", op="eq", value=None
    )
    with pytest.raises(TypeError, match="non-null"):
        compile_take(snapshot, plan, limit=5)


def test_compile_read_selects_named_columns():
    snapshot = _rental_snapshot()
    compiled = compile_read(
        snapshot,
        "rental",
        row_id=100,
        columns=("customer_id", "rental_date"),
    )
    assert (
        "SELECT t.\"customer_id\", t.\"rental_date\""
        in compiled.sql
    )
    assert "WHERE t.\"rental_id\" = $1" in compiled.sql
    assert compiled.params == (100,)


def test_compile_read_rejects_unknown_column():
    snapshot = _rental_snapshot()
    with pytest.raises(KeyError):
        compile_read(snapshot, "rental", 1, ("nonexistent",))


# ---------- ViaNode ----------


def test_compile_take_forward_via_joins_through_fk():
    snapshot = _customer_rental_snapshot()
    edge = resolve_edge(
        snapshot, "rental", "rental.customer_id->customer"
    )
    base = WhereNode(
        table="rental", column="rental_date", op="gt", value="2005-01-01"
    )
    via = ViaNode(source=base, edge=edge)
    compiled = compile_take(snapshot, via, limit=3)
    assert "SELECT DISTINCT dst.\"customer_id\" AS id" in compiled.sql
    assert (
        "JOIN \"public\".\"rental\" AS origin "
        "ON origin.\"rental_id\" = inner_stream.id"
    ) in compiled.sql
    assert (
        "JOIN \"public\".\"customer\" AS dst "
        "ON dst.\"customer_id\" = origin.\"customer_id\""
    ) in compiled.sql
    assert "GROUP BY id ORDER BY id ASC LIMIT 3" in compiled.sql
    assert compiled.params == ("2005-01-01",)


def test_compile_take_forward_via_supports_composite_fk():
    snapshot = _composite_order_snapshot()
    edge = resolve_edge(
        snapshot,
        "line_item",
        "line_item.(tenant_id,order_id)->order.(tenant_id,order_id)",
    )
    base = WhereNode(table="line_item", column="sku", op="eq", value="A-1")
    via = ViaNode(source=base, edge=edge)

    compiled = compile_take(snapshot, via, limit=3)

    assert (
        "SELECT DISTINCT (dst.\"tenant_id\", dst.\"order_id\") AS id"
        in compiled.sql
    )
    assert (
        "ON dst.\"tenant_id\" = origin.\"tenant_id\" "
        "AND dst.\"order_id\" = origin.\"order_id\""
    ) in compiled.sql
    assert compiled.params == ("A-1",)


def test_compile_take_reverse_via_joins_from_parent_to_child():
    snapshot = _customer_rental_snapshot()
    edges = [
        edge
        for edge in snapshot.edges_to("customer")
        if edge.source_table == "rental"
    ]
    assert edges, "expected rental→customer FK in snapshot"
    typed = TypedEdge(spec=edges[0], direction=EdgeDirection.REVERSE)
    base = WhereNode(
        table="customer", column="store_id", op="eq", value=1
    )
    via = ViaNode(source=base, edge=typed)
    compiled = compile_take(snapshot, via, limit=3)
    assert "SELECT DISTINCT dst.\"rental_id\" AS id" in compiled.sql
    assert (
        "JOIN \"public\".\"customer\" AS origin "
        "ON origin.\"customer_id\" = inner_stream.id"
    ) in compiled.sql
    assert (
        "JOIN \"public\".\"rental\" AS dst "
        "ON dst.\"customer_id\" = origin.\"customer_id\""
    ) in compiled.sql


def test_compile_take_via_with_order_annotation_uses_group_by_dedup():
    snapshot = _customer_rental_snapshot()
    edge = resolve_edge(
        snapshot, "rental", "rental.customer_id->customer"
    )
    base = WhereNode(
        table="rental", column="rental_date", op="gt", value="2005-01-01"
    )
    via = ViaNode(source=base, edge=edge)
    ordered = OrderNode(source=via, column="first_name", direction="asc")
    compiled = compile_take(snapshot, ordered, limit=3)
    assert "GROUP BY base.id" in compiled.sql
    assert "MIN(tgt.\"first_name\") ASC" in compiled.sql
    assert "base.id ASC" in compiled.sql
    assert (
        "JOIN \"public\".\"customer\" AS tgt "
        "ON tgt.\"customer_id\" = base.id"
    ) in compiled.sql


# ---------- IntersectNode ----------


def test_compile_take_intersect_combines_left_and_right_streams():
    snapshot = _customer_rental_snapshot()
    left = WhereNode(
        table="customer", column="store_id", op="eq", value=1
    )
    right = WhereNode(
        table="customer", column="active", op="eq", value=1
    )
    plan = IntersectNode(left=left, right=right)
    compiled = compile_take(snapshot, plan, limit=3)
    assert "INTERSECT" in compiled.sql
    assert "WHERE t.\"store_id\" = $1" in compiled.sql
    assert "WHERE t.\"active\" = $2" in compiled.sql
    assert compiled.params == (1, 1)
    assert "GROUP BY id ORDER BY id ASC LIMIT 3" in compiled.sql


def test_compile_take_rejects_intersect_with_mismatched_targets():
    snapshot = _customer_rental_snapshot()
    left = WhereNode(
        table="customer", column="store_id", op="eq", value=1
    )
    right = WhereNode(
        table="rental", column="customer_id", op="eq", value=45
    )
    plan = IntersectNode(left=left, right=right)
    with pytest.raises(ValueError, match="same table"):
        compile_take(snapshot, plan, 3)


# ---------- count ----------


def test_compile_count_deduplicates_destination_records_for_via_chains():
    snapshot = _customer_rental_snapshot()
    edge = resolve_edge(
        snapshot, "rental", "rental.customer_id->customer"
    )
    base = WhereNode(
        table="rental", column="customer_id", op="eq", value=45
    )
    via = ViaNode(source=base, edge=edge)
    compiled = compile_count(snapshot, via)
    assert compiled.sql.startswith("SELECT COUNT(*) AS cnt FROM (")
    assert "SELECT DISTINCT dst.\"customer_id\" AS id" in compiled.sql
    assert compiled.params == (45,)


def test_compile_count_on_where_is_trivial():
    snapshot = _rental_snapshot()
    plan = WhereNode(
        table="rental", column="customer_id", op="eq", value=45
    )
    compiled = compile_count(snapshot, plan)
    assert "COUNT(*) AS cnt" in compiled.sql
    assert "WHERE t.\"customer_id\" = $1" in compiled.sql
    assert compiled.params == (45,)


# ---------- aggregate ----------


def test_compile_aggregate_joins_target_table_for_column():
    snapshot = _customer_rental_snapshot()
    base = WhereNode(
        table="payment", column="customer_id", op="eq", value=45
    )
    compiled = compile_aggregate(snapshot, base, fn="sum", column="amount")
    assert "SELECT SUM(tgt.\"amount\") AS agg" in compiled.sql
    assert (
        "JOIN \"public\".\"payment\" AS tgt "
        "ON tgt.\"payment_id\" = base.id"
    ) in compiled.sql
    assert compiled.params == (45,)


def test_compile_aggregate_rejects_unsupported_fn():
    snapshot = _rental_snapshot()
    plan = WhereNode(
        table="rental", column="customer_id", op="eq", value=45
    )
    with pytest.raises(ValueError):
        compile_aggregate(snapshot, plan, fn="stddev", column="rental_id")  # type: ignore[arg-type]


def test_compile_aggregate_rejects_unknown_column():
    snapshot = _rental_snapshot()
    plan = WhereNode(
        table="rental", column="customer_id", op="eq", value=45
    )
    with pytest.raises(KeyError):
        compile_aggregate(
            snapshot, plan, fn="max", column="nonexistent"
        )


# ---------- group_top ----------


def test_compile_group_top_count_orders_desc_with_group_tiebreak():
    snapshot = _customer_rental_snapshot()
    base = WhereNode(
        table="rental", column="rental_date", op="gt", value="2005-01-01"
    )
    compiled = compile_group_top(
        snapshot,
        base,
        group_column="customer_id",
        fn="count",
        agg_column=None,
        limit=3,
    )
    assert (
        "SELECT tgt.\"customer_id\" AS group_value, "
        "COUNT(*) AS agg_value"
    ) in compiled.sql
    assert "GROUP BY tgt.\"customer_id\"" in compiled.sql
    assert "ORDER BY agg_value DESC, group_value ASC" in compiled.sql
    assert compiled.sql.endswith("LIMIT 3")


def test_compile_group_top_sum_requires_agg_column():
    snapshot = _customer_rental_snapshot()
    base = WhereNode(
        table="payment", column="customer_id", op="in", value=[1, 2, 3]
    )
    compiled = compile_group_top(
        snapshot,
        base,
        group_column="customer_id",
        fn="sum",
        agg_column="amount",
        limit=5,
    )
    assert "SUM(tgt.\"amount\") AS agg_value" in compiled.sql
    with pytest.raises(ValueError):
        compile_group_top(
            snapshot,
            base,
            group_column="customer_id",
            fn="sum",
            agg_column=None,
            limit=5,
        )


def test_compile_group_top_count_rejects_agg_column():
    snapshot = _customer_rental_snapshot()
    base = WhereNode(
        table="rental", column="rental_date", op="gt", value="2005-01-01"
    )
    with pytest.raises(ValueError):
        compile_group_top(
            snapshot,
            base,
            group_column="customer_id",
            fn="count",
            agg_column="rental_id",
            limit=3,
        )
