"""Compile CursorPlan trees into parameterized SQL.

Set-producing plans (WhereNode / ViaNode / IntersectNode) are first lowered
into an "id stream" subquery — a SELECT that yields a single column `id`
holding primary-key values of the plan's target table. A record_set represents
unique target records; relation traversal deduplicates by destination primary
key.

Materializers wrap the id stream:

- compile_take: ORDER BY + LIMIT over unique record ids
- compile_count: COUNT(*) over unique record ids
- compile_aggregate: SUM/AVG/MIN/MAX(column) over unique target records
- compile_group_top: GROUP BY group_column + aggregate + LIMIT

compile_read reads named columns of a single row by primary key; it does
not participate in cursor plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rl_task_foundry.tooling.atomic.cursor import (
    _FILTER_OPS,
    CursorPlan,
    FilterNode,
    IntersectNode,
    OrderNode,
    TableNode,
    ViaNode,
    WhereNode,
)
from rl_task_foundry.tooling.common.edges import EdgeDirection, resolve_edge
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
    """Emit a PostgreSQL ROW expression for the table's primary key.

    Single-column PKs render as ``alias.col`` (scalar). Composite PKs
    render as ``(alias.c1, alias.c2, ...)`` which PostgreSQL treats as
    an anonymous composite row — row equality, set ops, GROUP BY, and
    ORDER BY all work on these, so the rest of the calculus can treat
    ``id`` as a single column regardless of PK arity.
    """

    if not table.primary_key:
        raise ValueError(
            f"table {table.qualified_name!r} has no primary key; row-set "
            "materialization requires a primary key"
        )
    if len(table.primary_key) == 1:
        return f"{alias}.{quote_ident(table.primary_key[0])}"
    parts = ", ".join(
        f"{alias}.{quote_ident(column)}" for column in table.primary_key
    )
    return f"({parts})"


def _deterministic_tiebreak(table: TableSpec, alias: str) -> str:
    if not table.primary_key:
        raise ValueError(
            f"table {table.qualified_name!r} has no primary key; deterministic "
            "row ordering requires a primary key"
        )
    return ", ".join(
        f"{alias}.{quote_ident(column)} ASC" for column in table.primary_key
    )


def _limit_offset_clause(limit: int, offset: int) -> str:
    clause = f"LIMIT {int(limit)}"
    if offset:
        clause += f" OFFSET {int(offset)}"
    return clause


def _join_columns_predicate(
    *,
    left_alias: str,
    left_columns: tuple[str, ...],
    right_alias: str,
    right_columns: tuple[str, ...],
) -> str:
    if len(left_columns) != len(right_columns):
        raise ValueError("join column lists must have the same length")
    return " AND ".join(
        f"{left_alias}.{quote_ident(left)} = {right_alias}.{quote_ident(right)}"
        for left, right in zip(left_columns, right_columns, strict=True)
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
    if op == "is_null":
        return f"{qualified} IS NULL", (), param_start
    if op == "is_not_null":
        return f"{qualified} IS NOT NULL", (), param_start
    if op == "in":
        if not isinstance(plan.value, list) or len(plan.value) == 0:
            raise ValueError("op='in' requires a non-empty list value")
        if any(item is None for item in plan.value):
            raise TypeError("op='in' does not accept null list items")
        predicate = f"{qualified} = ANY(${param_start}::{_array_cast(column.data_type)})"
        return predicate, (list(plan.value),), param_start + 1
    if op == "like":
        if plan.value is None:
            raise TypeError("op='like' requires a non-null string pattern")
        if not isinstance(plan.value, str):
            raise TypeError("op='like' requires a string pattern")
        predicate = f"{qualified} ILIKE ${param_start}"
        return predicate, (plan.value,), param_start + 1
    if plan.value is None:
        raise TypeError(f"op={op!r} requires a non-null value")
    sql_op = {
        "eq": "=",
        "neq": "<>",
        "lt": "<",
        "gt": ">",
        "lte": "<=",
        "gte": ">=",
    }[op]
    predicate = f"{qualified} {sql_op} ${param_start}"
    return predicate, (plan.value,), param_start + 1


def _array_cast(data_type: str) -> str:
    # Schema introspection stores PostgreSQL udt_name values such as int4,
    # bool, and enum type names. Cast arrays to the column's actual element
    # type so `column = ANY($1::type[])` works across arbitrary DBs.
    mapping = {
        "int2": "int2[]",
        "int4": "int4[]",
        "int8": "int8[]",
        "integer": "int4[]",
        "bigint": "int8[]",
        "smallint": "int2[]",
        "serial": "int4[]",
        "bigserial": "int8[]",
        "smallserial": "int2[]",
        "float4": "float4[]",
        "float8": "float8[]",
        "real": "float4[]",
        "double precision": "float8[]",
        "numeric": "numeric[]",
        "decimal": "numeric[]",
        "money": "money[]",
        "bool": "bool[]",
        "boolean": "bool[]",
        "text": "text[]",
        "bpchar": "bpchar[]",
        "char": "bpchar[]",
        "character": "bpchar[]",
        "character varying": "text[]",
        "varchar": "text[]",
        "uuid": "uuid[]",
        "date": "date[]",
        "time": "time[]",
        "timetz": "timetz[]",
        "timestamp": "timestamp[]",
        "timestamptz": "timestamptz[]",
        "timestamp without time zone": "timestamp[]",
        "timestamp with time zone": "timestamptz[]",
        "bytea": "bytea[]",
    }
    normalized = data_type.strip().lower()
    return mapping.get(normalized, f"{quote_ident(data_type)}[]")


OrderTerm = tuple[tuple[str, ...], str, str]


def _collect_order(plan: CursorPlan) -> tuple[CursorPlan, list[OrderTerm]]:
    """Strip OrderNode wrappers, returning the inner plan and the
    (path, column, direction) stack from outermost (highest priority) inward."""
    orderings: list[OrderTerm] = []
    current = plan
    while isinstance(current, OrderNode):
        orderings.append((current.path, current.column, current.direction))
        current = current.source
    return current, orderings


def _has_related_order(orderings: list[OrderTerm]) -> bool:
    return any(path for path, _column, _direction in orderings)


def _compile_order_path_joins(
    snapshot: SchemaSnapshot,
    *,
    source_table: str,
    source_alias: str,
    path: tuple[str, ...],
    key_index: int,
) -> tuple[list[str], str, TableSpec]:
    current_table = source_table
    current_alias = source_alias
    joins: list[str] = []
    for step_index, edge_label in enumerate(path):
        edge = resolve_edge(snapshot, current_table, edge_label)
        if edge.direction is not EdgeDirection.FORWARD:
            raise ValueError(
                "sort_record_set related paths must follow forward relation labels"
            )
        destination = snapshot.table(edge.destination_table)
        destination_alias = f"ord_{key_index}_{step_index}"
        predicate = _join_columns_predicate(
            left_alias=destination_alias,
            left_columns=edge.destination_columns,
            right_alias=current_alias,
            right_columns=edge.origin_columns,
        )
        joins.append(
            f"LEFT JOIN {quote_table(destination.schema, destination.name)} "
            f"AS {destination_alias} ON {predicate}"
        )
        current_table = edge.destination_table
        current_alias = destination_alias
    return joins, current_alias, snapshot.table(current_table)


def _compile_grouped_take(
    snapshot: SchemaSnapshot,
    *,
    inner: CursorPlan,
    orderings: list[OrderTerm],
    limit: int,
    offset: int,
) -> CompiledQuery:
    target = snapshot.table(inner.target_table)
    target_pk_expr = _pk_expression(target, "tgt")
    base_sql, params, _ = _compile_id_stream(snapshot, inner, 1)
    if not orderings:
        sql = readonly_select(
            f"SELECT id FROM ({base_sql}) AS base "
            f"GROUP BY id ORDER BY id ASC "
            f"{_limit_offset_clause(limit, offset)}"
        )
        return CompiledQuery(sql=sql, params=params)

    joins: list[str] = []
    order_clauses: list[str] = []
    for key_index, (path, column_name, direction) in enumerate(orderings):
        if path:
            path_joins, order_alias, order_table = _compile_order_path_joins(
                snapshot,
                source_table=target.handle,
                source_alias="tgt",
                path=path,
                key_index=key_index,
            )
            joins.extend(path_joins)
        else:
            order_alias = "tgt"
            order_table = target
        order_table.column(column_name)
        agg = "MIN" if direction == "asc" else "MAX"
        order_clauses.append(
            f"{agg}({order_alias}.{quote_ident(column_name)}) {direction.upper()}"
        )
    order_clauses.append("base.id ASC")
    join_sql = " ".join(joins)
    if join_sql:
        join_sql += " "
    sql = readonly_select(
        f"SELECT base.id "
        f"FROM ({base_sql}) AS base "
        f"JOIN {quote_table(target.schema, target.name)} AS tgt "
        f"ON {target_pk_expr} = base.id "
        f"{join_sql}"
        f"GROUP BY base.id "
        f"ORDER BY {', '.join(order_clauses)} "
        f"{_limit_offset_clause(limit, offset)}"
    )
    return CompiledQuery(sql=sql, params=params)


def _compile_where_id_stream(
    snapshot: SchemaSnapshot,
    plan: WhereNode,
    param_start: int,
) -> tuple[str, tuple[object, ...], int]:
    table = snapshot.table(plan.table)
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


def _compile_table_id_stream(
    snapshot: SchemaSnapshot,
    plan: TableNode,
    param_start: int,
) -> tuple[str, tuple[object, ...], int]:
    table = snapshot.table(plan.table)
    alias = "t"
    pk_expr = _pk_expression(table, alias)
    sql = (
        f"SELECT {pk_expr} AS id "
        f"FROM {quote_table(table.schema, table.name)} AS {alias}"
    )
    return sql, (), param_start


def _compile_filter_id_stream(
    snapshot: SchemaSnapshot,
    plan: FilterNode,
    param_start: int,
) -> tuple[str, tuple[object, ...], int]:
    inner_sql, params, next_idx = _compile_id_stream(
        snapshot, plan.source, param_start
    )
    target = snapshot.table(plan.target_table)
    target_pk_expr = _pk_expression(target, "tgt")
    where_plan = WhereNode(
        table=plan.target_table,
        column=plan.column,
        op=plan.op,
        value=plan.value,
    )
    predicate, filter_params, final_idx = _compile_where_predicate(
        snapshot=snapshot,
        plan=where_plan,
        alias="tgt",
        param_start=next_idx,
    )
    sql = (
        f"SELECT base.id AS id "
        f"FROM ({inner_sql}) AS base "
        f"JOIN {quote_table(target.schema, target.name)} AS tgt "
        f"ON {target_pk_expr} = base.id "
        f"WHERE {predicate}"
    )
    return sql, params + filter_params, final_idx


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
    origin_pk_expr = _pk_expression(origin_spec, "origin")
    dest_pk_expr = _pk_expression(dest_spec, "dst")
    join_predicate = _join_columns_predicate(
        left_alias="dst",
        left_columns=edge.destination_columns,
        right_alias="origin",
        right_columns=edge.origin_columns,
    )
    sql = (
        f"SELECT DISTINCT {dest_pk_expr} AS id "
        f"FROM ({inner_sql}) AS inner_stream "
        f"JOIN {quote_table(origin_spec.schema, origin_spec.name)} AS origin "
        f"ON {origin_pk_expr} = inner_stream.id "
        f"JOIN {quote_table(dest_spec.schema, dest_spec.name)} AS dst "
        f"ON {join_predicate}"
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
    if isinstance(plan, TableNode):
        return _compile_table_id_stream(snapshot, plan, param_start)
    if isinstance(plan, WhereNode):
        return _compile_where_id_stream(snapshot, plan, param_start)
    if isinstance(plan, FilterNode):
        return _compile_filter_id_stream(snapshot, plan, param_start)
    if isinstance(plan, ViaNode):
        return _compile_via_id_stream(snapshot, plan, param_start)
    if isinstance(plan, IntersectNode):
        return _compile_intersect_id_stream(snapshot, plan, param_start)
    raise TypeError(f"unknown plan node: {type(plan).__name__}")


def compile_take(
    snapshot: SchemaSnapshot,
    plan: CursorPlan,
    limit: int,
    offset: int = 0,
) -> CompiledQuery:
    """Compile a plan into a SELECT returning primary-key values.

    WhereNode plans emit a direct SELECT with ORDER BY + PK tiebreak. Wrapped
    plans go through the same GROUP BY path used for deterministic ordering of
    unique record ids.
    """
    inner, orderings = _collect_order(plan)
    if isinstance(inner, TableNode):
        if _has_related_order(orderings):
            return _compile_grouped_take(
                snapshot,
                inner=inner,
                orderings=orderings,
                limit=limit,
                offset=offset,
            )
        return _compile_take_table(snapshot, inner, orderings, limit, offset)
    if isinstance(inner, WhereNode):
        if _has_related_order(orderings):
            return _compile_grouped_take(
                snapshot,
                inner=inner,
                orderings=orderings,
                limit=limit,
                offset=offset,
            )
        return _compile_take_where(snapshot, inner, orderings, limit, offset)
    if not isinstance(inner, (FilterNode, ViaNode, IntersectNode)):
        raise TypeError(
            f"compile_take expected Table/Where/Filter/Via/Intersect base; got "
            f"{type(inner).__name__}"
        )
    return _compile_grouped_take(
        snapshot,
        inner=inner,
        orderings=orderings,
        limit=limit,
        offset=offset,
    )


def _compile_take_table(
    snapshot: SchemaSnapshot,
    inner: TableNode,
    orderings: list[OrderTerm],
    limit: int,
    offset: int,
) -> CompiledQuery:
    table = snapshot.table(inner.table)
    alias = "t"
    order_clauses: list[str] = []
    for path, column_name, direction in orderings:
        if path:
            raise ValueError("related order paths require grouped materialization")
        table.column(column_name)
        order_clauses.append(
            f"{alias}.{quote_ident(column_name)} {direction.upper()}"
        )
    order_clauses.append(_deterministic_tiebreak(table, alias))
    pk_expr = _pk_expression(table, alias)
    sql = readonly_select(
        f"SELECT {pk_expr} AS id "
        f"FROM {quote_table(table.schema, table.name)} AS {alias} "
        f"ORDER BY {', '.join(order_clauses)} "
        f"{_limit_offset_clause(limit, offset)}"
    )
    return CompiledQuery(sql=sql, params=())


def _compile_take_where(
    snapshot: SchemaSnapshot,
    inner: WhereNode,
    orderings: list[OrderTerm],
    limit: int,
    offset: int,
) -> CompiledQuery:
    table = snapshot.table(inner.table)
    alias = "t"
    predicate, params, _ = _compile_where_predicate(
        snapshot=snapshot, plan=inner, alias=alias, param_start=1
    )
    order_clauses: list[str] = []
    for path, column_name, direction in orderings:
        if path:
            raise ValueError("related order paths require grouped materialization")
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
        f"{_limit_offset_clause(limit, offset)}"
    )
    return CompiledQuery(sql=sql, params=params)


def compile_count(
    snapshot: SchemaSnapshot,
    plan: CursorPlan,
) -> CompiledQuery:
    """Compile COUNT(*) over the plan's unique target records."""
    inner, _orderings = _collect_order(plan)
    if not isinstance(inner, (TableNode, WhereNode, FilterNode, ViaNode, IntersectNode)):
        raise TypeError(
            f"compile_count expected Table/Where/Filter/Via/Intersect base; got "
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
    plan's unique target records.
    """
    if fn not in _AGGREGATE_FNS:
        raise ValueError(
            f"fn must be one of {sorted(_AGGREGATE_FNS)}; got {fn!r}"
        )
    inner, _orderings = _collect_order(plan)
    if not isinstance(inner, (TableNode, WhereNode, FilterNode, ViaNode, IntersectNode)):
        raise TypeError(
            f"compile_aggregate expected Table/Where/Filter/Via/Intersect base; got "
            f"{type(inner).__name__}"
        )
    target = snapshot.table(inner.target_table)
    target_pk_expr = _pk_expression(target, "tgt")
    target.column(column)
    base_sql, params, _ = _compile_id_stream(snapshot, inner, 1)
    sql = readonly_select(
        f"SELECT {fn.upper()}(tgt.{quote_ident(column)}) AS agg "
        f"FROM ({base_sql}) AS base "
        f"JOIN {quote_table(target.schema, target.name)} AS tgt "
        f"ON {target_pk_expr} = base.id"
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
    if not isinstance(inner, (TableNode, WhereNode, FilterNode, ViaNode, IntersectNode)):
        raise TypeError(
            f"compile_group_top expected Table/Where/Filter/Via/Intersect base; got "
            f"{type(inner).__name__}"
        )
    target = snapshot.table(inner.target_table)
    target_pk_expr = _pk_expression(target, "tgt")
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
        f"ON {target_pk_expr} = base.id "
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
    """Compile a single-row read of the named columns.

    ``row_id`` is a scalar for single-column PKs and a sequence
    (list/tuple) of the same length as ``table.primary_key`` for
    composite PKs. The generated WHERE clause binds one parameter
    per PK column so asyncpg can type-check each component.
    """

    table = snapshot.table(table_name)
    if not columns:
        raise ValueError("columns must be non-empty")
    selected = []
    for column_name in columns:
        table.column(column_name)
        selected.append(f"t.{quote_ident(column_name)}")
    pk_cols = table.primary_key
    if not pk_cols:
        raise ValueError(
            f"table {table.qualified_name!r} has no primary key; get_row "
            "requires a primary key"
        )
    if len(pk_cols) == 1:
        pk = pk_cols[0]
        table.column(pk)
        where_clause = f"t.{quote_ident(pk)} = $1"
        params: tuple[object, ...] = (row_id,)
    else:
        if not isinstance(row_id, (list, tuple)):
            raise TypeError(
                f"table {table.qualified_name!r} has a composite primary "
                f"key {list(pk_cols)}; row_id must be a list/tuple of "
                f"length {len(pk_cols)}, got {type(row_id).__name__}"
            )
        if len(row_id) != len(pk_cols):
            raise ValueError(
                f"composite primary key for {table.qualified_name!r} has "
                f"{len(pk_cols)} columns; row_id has {len(row_id)}"
            )
        equalities = []
        for index, pk_col in enumerate(pk_cols):
            table.column(pk_col)
            equalities.append(f"t.{quote_ident(pk_col)} = ${index + 1}")
        where_clause = " AND ".join(equalities)
        params = tuple(row_id)
    sql = readonly_select(
        f"SELECT {', '.join(selected)} "
        f"FROM {quote_table(table.schema, table.name)} AS t "
        f"WHERE {where_clause} "
        f"LIMIT 1"
    )
    return CompiledQuery(sql=sql, params=params)


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
