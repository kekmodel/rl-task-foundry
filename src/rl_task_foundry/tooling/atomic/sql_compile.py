"""Compile CursorPlan trees into parameterized SQL.

Set-producing plans (WhereNode / ViaNode / IntersectNode) are first lowered
into an "id stream" subquery — a SELECT that yields a single column `id`
holding primary-key values of the plan's target table, preserving
multiplicity for rows_via chains.

Materializers wrap the id stream:

- compile_take: DISTINCT + ORDER BY + LIMIT (dedup for ViaNode chains)
- compile_count: COUNT(*) over the bag
- compile_aggregate: SUM/AVG/MIN/MAX(column) joined back to target table
- compile_group_top: GROUP BY group_column + aggregate + LIMIT

compile_read reads named columns of a single row by primary key; it does
not participate in cursor plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rl_task_foundry.tooling.atomic.cursor import (
    CursorPlan,
    IntersectNode,
    OrderNode,
    ViaNode,
    WhereNode,
    _FILTER_OPS,
)
from rl_task_foundry.tooling.common.edges import EdgeDirection
from rl_task_foundry.tooling.common.schema import SchemaSnapshot, TableSpec
from rl_task_foundry.tooling.common.sql import (
    quote_ident,
    quote_table,
    readonly_select,
)


AggregateFn = Literal["sum", "avg", "min", "max"]
GroupAggregateFn = Literal["count", "sum", "avg", "min", "max"]

_AGGREGATE_FNS: frozenset[AggregateFn] = frozenset(
    ("sum", "avg", "min", "max")
)
_GROUP_AGGREGATE_FNS: frozenset[GroupAggregateFn] = frozenset(
    ("count", "sum", "avg", "min", "max")
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


def _single_column_pk(table: TableSpec) -> str:
    if len(table.primary_key) != 1:
        raise NotImplementedError(
            f"table {table.qualified_name!r} has a composite primary key; "
            "atomic calculus supports single-column PKs only for now"
        )
    return table.primary_key[0]


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


def _compile_where_id_stream(
    snapshot: SchemaSnapshot,
    plan: WhereNode,
    param_start: int,
) -> tuple[str, tuple[object, ...], int]:
    table = snapshot.table(plan.table)
    _ = _single_column_pk(table)
    alias = "t"
    predicate, params, next_idx = _compile_where_predicate(
        snapshot=snapshot, plan=plan, alias=alias, param_start=param_start
    )
    pk_expr = _pk_expression(table, alias)
    sql = (
        f"SELECT {pk_expr} AS id "
        f"FROM {quote_table(table.schema, table.name)} AS {alias} "
        f"WHERE {predicate}"
    )
    return sql, params, next_idx


def _compile_via_id_stream(
    snapshot: SchemaSnapshot,
    plan: ViaNode,
    param_start: int,
) -> tuple[str, tuple[object, ...], int]:
    inner_sql, params, next_idx = _compile_id_stream(
        snapshot, plan.source, param_start
    )
    edge = plan.edge
    origin_spec = snapshot.table(edge.origin_table)
    dest_spec = snapshot.table(edge.destination_table)
    origin_pk = _single_column_pk(origin_spec)
    dest_pk = _single_column_pk(dest_spec)
    if edge.direction is EdgeDirection.FORWARD:
        # Join origin.<source_column> = dest.<target_column>; inner emits
        # origin PKs so we self-join origin to get source_column values.
        origin_match_col = edge.spec.source_column
        dest_match_col = edge.spec.target_column
    else:
        origin_match_col = edge.spec.target_column
        dest_match_col = edge.spec.source_column
    sql = (
        f"SELECT dst.{quote_ident(dest_pk)} AS id "
        f"FROM ({inner_sql}) AS inner_stream "
        f"JOIN {quote_table(origin_spec.schema, origin_spec.name)} AS origin "
        f"ON origin.{quote_ident(origin_pk)} = inner_stream.id "
        f"JOIN {quote_table(dest_spec.schema, dest_spec.name)} AS dst "
        f"ON dst.{quote_ident(dest_match_col)} "
        f"= origin.{quote_ident(origin_match_col)}"
    )
    return sql, params, next_idx


def _compile_intersect_id_stream(
    snapshot: SchemaSnapshot,
    plan: IntersectNode,
    param_start: int,
) -> tuple[str, tuple[object, ...], int]:
    if plan.left.target_table != plan.right.target_table:
        raise ValueError(
            "intersect operands must target the same table; got "
            f"{plan.left.target_table!r} and {plan.right.target_table!r}"
        )
    left_sql, left_params, mid_idx = _compile_id_stream(
        snapshot, plan.left, param_start
    )
    right_sql, right_params, next_idx = _compile_id_stream(
        snapshot, plan.right, mid_idx
    )
    sql = (
        f"SELECT id FROM ({left_sql}) AS left_stream "
        f"INTERSECT "
        f"SELECT id FROM ({right_sql}) AS right_stream"
    )
    return sql, left_params + right_params, next_idx


def _compile_id_stream(
    snapshot: SchemaSnapshot,
    plan: CursorPlan,
    param_start: int,
) -> tuple[str, tuple[object, ...], int]:
    if isinstance(plan, OrderNode):
        # Ordering is not set-producing; strip for any caller that passes
        # a plan still wearing its OrderNode wrappers.
        return _compile_id_stream(snapshot, plan.source, param_start)
    if isinstance(plan, WhereNode):
        return _compile_where_id_stream(snapshot, plan, param_start)
    if isinstance(plan, ViaNode):
        return _compile_via_id_stream(snapshot, plan, param_start)
    if isinstance(plan, IntersectNode):
        return _compile_intersect_id_stream(snapshot, plan, param_start)
    raise TypeError(f"unknown plan node: {type(plan).__name__}")


def compile_take(
    snapshot: SchemaSnapshot,
    plan: CursorPlan,
    limit: int,
) -> CompiledQuery:
    """Compile a plan into a SELECT returning primary-key values.

    WhereNode plans emit a direct SELECT with ORDER BY + PK tiebreak. Plans
    that may introduce join multiplicity (ViaNode) or set deduplication
    (IntersectNode) go through a GROUP BY dedup path — the outermost query
    groups by `id` and orders by MIN/MAX(order_column) per group, matching
    the declared ASC/DESC direction.
    """
    inner, orderings = _collect_order(plan)
    if isinstance(inner, WhereNode):
        return _compile_take_where(snapshot, inner, orderings, limit)
    if not isinstance(inner, (ViaNode, IntersectNode)):
        raise TypeError(
            f"compile_take expected Where/Via/Intersect base; got "
            f"{type(inner).__name__}"
        )
    target = snapshot.table(inner.target_table)
    target_pk = _single_column_pk(target)
    base_sql, params, _ = _compile_id_stream(snapshot, inner, 1)
    if not orderings:
        sql = readonly_select(
            f"SELECT id FROM ({base_sql}) AS base "
            f"GROUP BY id ORDER BY id ASC "
            f"LIMIT {int(limit)}"
        )
        return CompiledQuery(sql=sql, params=params)
    order_clauses: list[str] = []
    for column_name, direction in orderings:
        target.column(column_name)  # validate
        agg = "MIN" if direction == "asc" else "MAX"
        order_clauses.append(
            f"{agg}(tgt.{quote_ident(column_name)}) {direction.upper()}"
        )
    order_clauses.append("base.id ASC")
    sql = readonly_select(
        f"SELECT base.id "
        f"FROM ({base_sql}) AS base "
        f"JOIN {quote_table(target.schema, target.name)} AS tgt "
        f"ON tgt.{quote_ident(target_pk)} = base.id "
        f"GROUP BY base.id "
        f"ORDER BY {', '.join(order_clauses)} "
        f"LIMIT {int(limit)}"
    )
    return CompiledQuery(sql=sql, params=params)


def _compile_take_where(
    snapshot: SchemaSnapshot,
    inner: WhereNode,
    orderings: list[tuple[str, str]],
    limit: int,
) -> CompiledQuery:
    table = snapshot.table(inner.table)
    alias = "t"
    predicate, params, _ = _compile_where_predicate(
        snapshot=snapshot, plan=inner, alias=alias, param_start=1
    )
    order_clauses: list[str] = []
    for column_name, direction in orderings:
        table.column(column_name)
        order_clauses.append(
            f"{alias}.{quote_ident(column_name)} {direction.upper()}"
        )
    order_clauses.append(_deterministic_tiebreak(table, alias))
    pk_expr = _pk_expression(table, alias)
    sql = readonly_select(
        f"SELECT {pk_expr} AS id "
        f"FROM {quote_table(table.schema, table.name)} AS {alias} "
        f"WHERE {predicate} "
        f"ORDER BY {', '.join(order_clauses)} "
        f"LIMIT {int(limit)}"
    )
    return CompiledQuery(sql=sql, params=params)


def compile_count(
    snapshot: SchemaSnapshot,
    plan: CursorPlan,
) -> CompiledQuery:
    """Compile COUNT(*) over the plan's row bag.

    Multiplicity is preserved — rows_via chains contribute one count entry
    per joined source row, matching the spec's bag semantics.
    """
    inner, _orderings = _collect_order(plan)
    if not isinstance(inner, (WhereNode, ViaNode, IntersectNode)):
        raise TypeError(
            f"compile_count expected Where/Via/Intersect base; got "
            f"{type(inner).__name__}"
        )
    base_sql, params, _ = _compile_id_stream(snapshot, inner, 1)
    sql = readonly_select(
        f"SELECT COUNT(*) AS cnt FROM ({base_sql}) AS base"
    )
    return CompiledQuery(sql=sql, params=params)


def compile_aggregate(
    snapshot: SchemaSnapshot,
    plan: CursorPlan,
    fn: AggregateFn,
    column: str,
) -> CompiledQuery:
    """Compile a scalar aggregate (SUM/AVG/MIN/MAX) of `column` over the
    plan's row bag. The column is resolved against the plan's target table
    and the join to that table preserves multiplicity.
    """
    if fn not in _AGGREGATE_FNS:
        raise ValueError(
            f"fn must be one of {sorted(_AGGREGATE_FNS)}; got {fn!r}"
        )
    inner, _orderings = _collect_order(plan)
    if not isinstance(inner, (WhereNode, ViaNode, IntersectNode)):
        raise TypeError(
            f"compile_aggregate expected Where/Via/Intersect base; got "
            f"{type(inner).__name__}"
        )
    target = snapshot.table(inner.target_table)
    target_pk = _single_column_pk(target)
    target.column(column)
    base_sql, params, _ = _compile_id_stream(snapshot, inner, 1)
    sql = readonly_select(
        f"SELECT {fn.upper()}(tgt.{quote_ident(column)}) AS agg "
        f"FROM ({base_sql}) AS base "
        f"JOIN {quote_table(target.schema, target.name)} AS tgt "
        f"ON tgt.{quote_ident(target_pk)} = base.id"
    )
    return CompiledQuery(sql=sql, params=params)


def compile_group_top(
    snapshot: SchemaSnapshot,
    plan: CursorPlan,
    group_column: str,
    fn: GroupAggregateFn,
    agg_column: str | None,
    limit: int,
) -> CompiledQuery:
    """Compile a GROUP BY group_column + <fn>(agg_column) + ORDER BY
    descending aggregate + LIMIT. Deterministic tiebreak is group_column
    ascending. `fn="count"` ignores agg_column and uses COUNT(*).
    """
    if fn not in _GROUP_AGGREGATE_FNS:
        raise ValueError(
            f"fn must be one of {sorted(_GROUP_AGGREGATE_FNS)}; got {fn!r}"
        )
    if fn == "count":
        if agg_column is not None:
            raise ValueError("agg_column must be None when fn='count'")
    else:
        if agg_column is None:
            raise ValueError(f"fn={fn!r} requires agg_column")
    inner, _orderings = _collect_order(plan)
    if not isinstance(inner, (WhereNode, ViaNode, IntersectNode)):
        raise TypeError(
            f"compile_group_top expected Where/Via/Intersect base; got "
            f"{type(inner).__name__}"
        )
    target = snapshot.table(inner.target_table)
    target_pk = _single_column_pk(target)
    target.column(group_column)
    if agg_column is not None:
        target.column(agg_column)
    base_sql, params, _ = _compile_id_stream(snapshot, inner, 1)
    if fn == "count":
        agg_expr = "COUNT(*)"
    else:
        agg_expr = f"{fn.upper()}(tgt.{quote_ident(agg_column)})"  # type: ignore[arg-type]
    sql = readonly_select(
        f"SELECT tgt.{quote_ident(group_column)} AS group_value, "
        f"{agg_expr} AS agg_value "
        f"FROM ({base_sql}) AS base "
        f"JOIN {quote_table(target.schema, target.name)} AS tgt "
        f"ON tgt.{quote_ident(target_pk)} = base.id "
        f"GROUP BY tgt.{quote_ident(group_column)} "
        f"ORDER BY agg_value DESC, group_value ASC "
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
    "AggregateFn",
    "CompiledQuery",
    "GroupAggregateFn",
    "compile_aggregate",
    "compile_count",
    "compile_group_top",
    "compile_read",
    "compile_take",
]
