"""Unit tests for sql_compile: CursorPlan → SQL translation."""

from __future__ import annotations

import pytest

from rl_task_foundry.schema.graph import (
    ColumnProfile,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.tooling.atomic.cursor import OrderNode, WhereNode
from rl_task_foundry.tooling.atomic.sql_compile import (
    compile_read,
    compile_take,
)
from rl_task_foundry.tooling.common import snapshot_from_graph


def _column(
    table: str,
    name: str,
    data_type: str = "integer",
    *,
    is_primary_key: bool = False,
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


def test_compile_take_rejects_via_nodes_for_now():
    snapshot = _rental_snapshot()
    plan = WhereNode(
        table="rental", column="customer_id", op="eq", value=1
    )
    with pytest.raises(NotImplementedError):
        # intersect/via not in the vertical slice
        from rl_task_foundry.tooling.atomic.cursor import IntersectNode

        compile_take(snapshot, IntersectNode(left=plan, right=plan), 3)


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
