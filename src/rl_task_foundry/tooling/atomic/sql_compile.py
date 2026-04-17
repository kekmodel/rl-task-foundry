"""Compile CursorPlan trees into parameterized SQL.

The vertical slice needs only:
- compile_take: plan + LIMIT → SQL returning primary-key values
- compile_read: table + row_id + columns → SQL returning one row's fields

rows_via, intersect, count, aggregate, group_top compilation lands next
session alongside the remaining calculus primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_task_foundry.tooling.atomic.cursor import (
    CursorPlan,
    OrderNode,
    WhereNode,
    _FILTER_OPS,
)
from rl_task_foundry.tooling.common.schema import SchemaSnapshot, TableSpec
from rl_task_foundry.tooling.common.sql import (
    quote_ident,
    quote_table,
    readonly_select,
)


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    sql: str
    params: tuple[object, ...]


def _pk_expression(table: TableSpec, alias: str) -> str:
    if len(table.primary_key) == 1:
        return f"{alias}.{quote_ident(table.primary_key[0])}"
    parts = ", ".join(
        f"{alias}.{quote_ident(column)}" for column in table.primary_key
    )
    return f"({parts})"


def _deterministic_tiebreak(table: TableSpec, alias: str) -> str:
    return ", ".join(
        f"{alias}.{quote_ident(column)} ASC" for column in table.primary_key
    )


def _compile_where_predicate(
    snapshot: SchemaSnapshot,
    plan: WhereNode,
    alias: str,
    param_start: int,
) -> tuple[str, tuple[object, ...], int]:
    table = snapshot.table(plan.table)
    column = table.column(plan.column)
    qualified = f"{alias}.{quote_ident(column.name)}"
    op = plan.op
    if op not in _FILTER_OPS:
        raise ValueError(f"unsupported op: {op!r}")
    if op == "in":
        if not isinstance(plan.value, list) or len(plan.value) == 0:
            raise ValueError("op='in' requires a non-empty list value")
        predicate = f"{qualified} = ANY(${param_start}::{_array_cast(column.data_type)})"
        return predicate, (list(plan.value),), param_start + 1
    if op == "like":
        if not isinstance(plan.value, str):
            raise TypeError("op='like' requires a string pattern")
        predicate = f"{qualified} ILIKE ${param_start}"
        return predicate, (plan.value,), param_start + 1
    sql_op = {
        "eq": "=",
        "lt": "<",
        "gt": ">",
        "lte": "<=",
        "gte": ">=",
    }[op]
    predicate = f"{qualified} {sql_op} ${param_start}"
    return predicate, (plan.value,), param_start + 1


def _array_cast(data_type: str) -> str:
    # Minimal set; extend as new DBs appear. asyncpg handles most scalar
    # arrays via ANY when the base element type is inferred.
    mapping = {
        "integer": "int4[]",
        "bigint": "int8[]",
        "smallint": "int2[]",
        "text": "text[]",
        "character varying": "text[]",
        "varchar": "text[]",
        "uuid": "uuid[]",
    }
    return mapping.get(data_type, "text[]")


def _collect_order(plan: CursorPlan) -> tuple[CursorPlan, list[tuple[str, str]]]:
    """Strip OrderNode wrappers, returning the inner plan and the
    (column, direction) stack from outermost (highest priority) inward."""
    orderings: list[tuple[str, str]] = []
    current = plan
    while isinstance(current, OrderNode):
        orderings.append((current.column, current.direction))
        current = current.source
    return current, orderings


def compile_take(
    snapshot: SchemaSnapshot,
    plan: CursorPlan,
    limit: int,
) -> CompiledQuery:
    """Compile a plan into a SELECT returning primary-key values.

    Applies declared OrderNode orderings followed by a deterministic
    primary-key tiebreak. Returns `limit` rows.
    """
    inner, orderings = _collect_order(plan)
    if not isinstance(inner, WhereNode):
        raise NotImplementedError(
            "compile_take currently supports rows_where + order_by chains "
            "only. rows_via / intersect land next session."
        )
    table = snapshot.table(inner.table)
    alias = "t"
    predicate, params, _ = _compile_where_predicate(
        snapshot=snapshot, plan=inner, alias=alias, param_start=1
    )
    order_clauses: list[str] = []
    for column_name, direction in orderings:
        # validate column exists
        table.column(column_name)
        order_clauses.append(
            f"{alias}.{quote_ident(column_name)} {direction.upper()}"
        )
    order_clauses.append(_deterministic_tiebreak(table, alias))
    pk_expr = _pk_expression(table, alias)
    # rows_where over a base table yields distinct rows by construction
    # (each row is itself). DISTINCT becomes relevant only once rows_via
    # / intersect introduce join multiplicity; handled in a later slice.
    sql = readonly_select(
        f"SELECT {pk_expr} AS id "
        f"FROM {quote_table(table.schema, table.name)} AS {alias} "
        f"WHERE {predicate} "
        f"ORDER BY {', '.join(order_clauses)} "
        f"LIMIT {int(limit)}"
    )
    return CompiledQuery(sql=sql, params=params)


def compile_read(
    snapshot: SchemaSnapshot,
    table_name: str,
    row_id: object,
    columns: tuple[str, ...],
) -> CompiledQuery:
    """Compile a single-row read of the named columns."""
    table = snapshot.table(table_name)
    if len(table.primary_key) != 1:
        raise NotImplementedError(
            "compile_read supports single-column primary keys only for now"
        )
    pk = table.primary_key[0]
    table.column(pk)
    if not columns:
        raise ValueError("columns must be non-empty")
    selected = []
    for column_name in columns:
        table.column(column_name)
        selected.append(f"t.{quote_ident(column_name)}")
    sql = readonly_select(
        f"SELECT {', '.join(selected)} "
        f"FROM {quote_table(table.schema, table.name)} AS t "
        f"WHERE t.{quote_ident(pk)} = $1 "
        f"LIMIT 1"
    )
    return CompiledQuery(sql=sql, params=(row_id,))


__all__ = [
    "CompiledQuery",
    "compile_read",
    "compile_take",
]
