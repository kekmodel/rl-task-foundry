"""Shared SQL helpers for composer primitives.

Both `sample` and `query` need to compile a list of
`{column, op, value}` filter clauses over a single aliased table. Rather
than duplicate the op vocabulary and array-cast table, we keep one copy
here. Kept private to the composer package; not shared with atomic.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_task_foundry.tooling.common.schema import TableSpec
from rl_task_foundry.tooling.common.sql import (
    coerce_param,
    coerce_scalar,
    quote_ident,
)

FILTER_OPS: frozenset[str] = frozenset(
    ("eq", "neq", "in", "lt", "gt", "lte", "gte", "like", "is_null", "is_not_null")
)

_SQL_BINARY_OPS: dict[str, str] = {
    "eq": "=",
    "neq": "<>",
    "lt": "<",
    "gt": ">",
    "lte": "<=",
    "gte": ">=",
}

_ARRAY_CAST: dict[str, str] = {
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


def array_cast_for(data_type: str) -> str:
    normalized = data_type.strip().lower()
    return _ARRAY_CAST.get(normalized, f"{quote_ident(data_type)}[]")


@dataclass(frozen=True, slots=True)
class FilterClause:
    column: str
    op: str
    value: object


def parse_filter_clauses(
    raw: list[dict[str, object]] | None,
) -> list[FilterClause]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("filter must be a list of {column, op, value}")
    clauses: list[FilterClause] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise TypeError(
                f"filter[{index}] must be a mapping; got "
                f"{type(entry).__name__}"
            )
        column = entry.get("column")
        op = entry.get("op")
        if not isinstance(column, str):
            raise TypeError(
                f"filter[{index}].column must be a string"
            )
        if not isinstance(op, str):
            raise TypeError(
                f"filter[{index}].op must be a string"
            )
        clauses.append(
            FilterClause(column=column, op=op, value=entry.get("value"))
        )
    return clauses


def compile_filter_clauses(
    *,
    table_spec: TableSpec,
    alias: str,
    clauses: list[FilterClause],
    start_param: int,
) -> tuple[str, tuple[object, ...], int]:
    """Compile an AND of filter clauses anchored at the given alias.

    Returns `(where_clause_without_WHERE_prefix, params, next_param_index)`.
    When there are no clauses the clause string is empty — callers decide
    whether to prepend WHERE or drop the fragment entirely.
    """
    if not clauses:
        return "", (), start_param
    parts: list[str] = []
    params: list[object] = []
    next_index = start_param
    for clause in clauses:
        if clause.op not in FILTER_OPS:
            raise ValueError(
                f"unsupported op {clause.op!r}; expected one of "
                f"{sorted(FILTER_OPS)}"
            )
        column_spec = table_spec.exposed_column(clause.column)
        qualified = f"{alias}.{quote_ident(clause.column)}"
        value = coerce_scalar(clause.value, column_spec.data_type)
        value = coerce_param(value)
        if clause.op == "is_null":
            parts.append(f"{qualified} IS NULL")
            continue
        if clause.op == "is_not_null":
            parts.append(f"{qualified} IS NOT NULL")
            continue
        if clause.op == "in":
            if not isinstance(value, list) or len(value) == 0:
                raise ValueError(
                    f"predicate on {clause.column!r} with op='in' "
                    "requires a non-empty list value"
                )
            if any(item is None for item in value):
                raise TypeError(
                    f"predicate on {clause.column!r} with op='in' "
                    "does not accept null list items"
                )
            parts.append(
                f"{qualified} = ANY(${next_index}::"
                f"{array_cast_for(column_spec.data_type)})"
            )
            params.append(list(value))
            next_index += 1
            continue
        if clause.op == "like":
            if value is None:
                raise TypeError(
                    f"predicate on {clause.column!r} with op='like' "
                    "requires a non-null string pattern"
                )
            if not isinstance(value, str):
                raise TypeError(
                    f"predicate on {clause.column!r} with op='like' "
                    "requires a string pattern"
                )
            parts.append(f"{qualified} ILIKE ${next_index}")
            params.append(value)
            next_index += 1
            continue
        if value is None:
            raise TypeError(
                f"predicate on {clause.column!r} with op={clause.op!r} "
                "requires a non-null value"
            )
        sql_op = _SQL_BINARY_OPS[clause.op]
        parts.append(f"{qualified} {sql_op} ${next_index}")
        params.append(value)
        next_index += 1
    return " AND ".join(parts), tuple(params), next_index


def require_single_column_pk(table_spec: TableSpec, *, tool_name: str) -> str:
    """Return the single PK column name; raise NotImplementedError on composite.

    Some composer primitives need scalar anchor resolution. This helper
    centralizes that assertion with a caller-specific error message.
    """
    if len(table_spec.primary_key) != 1:
        raise NotImplementedError(
            f"{tool_name} supports single-column primary keys only; "
            f"{table_spec.qualified_name!r} has a composite PK"
        )
    return table_spec.primary_key[0]


def coerce_asyncpg_int(value: object) -> int:
    """Coerce an asyncpg-returned scalar to int.

    asyncpg returns Decimal for numeric aggregates like SUM(CASE ...)
    and COUNT DISTINCT; calling int() on Decimal works but is verbose.
    This helper also accepts already-int values unchanged and rejects
    bool (which is an int subclass) as schema-violating.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return int(str(value))
    return value


__all__ = [
    "FILTER_OPS",
    "FilterClause",
    "array_cast_for",
    "coerce_asyncpg_int",
    "compile_filter_clauses",
    "parse_filter_clauses",
    "require_single_column_pk",
]
