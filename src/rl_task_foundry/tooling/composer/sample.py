"""sample primitive — representative rows from a table.

Single call returning up to `n` rows of a table, optionally filtered by
an AND of `{column, op, value}` clauses. Ordering is deterministic: when
`seed` is None the result is sorted by primary key ascending; when seed
is supplied rows are ordered by md5 of the PK concatenated with the
seed, so two runs with the same seed see the same sample.

Composer-side primitive. Solvers never see it.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_task_foundry.tooling.common.schema import SchemaSnapshot, TableSpec
from rl_task_foundry.tooling.common.sql import (
    coerce_param,
    quote_ident,
    quote_table,
    readonly_select,
)
from rl_task_foundry.tooling.composer._session import ComposerSession


_FILTER_OPS = frozenset(
    ("eq", "in", "lt", "gt", "lte", "gte", "like")
)

_SQL_BINARY_OPS = {
    "eq": "=",
    "lt": "<",
    "gt": ">",
    "lte": "<=",
    "gte": ">=",
}

_ARRAY_CAST = {
    "integer": "int4[]",
    "bigint": "int8[]",
    "smallint": "int2[]",
    "text": "text[]",
    "character varying": "text[]",
    "varchar": "text[]",
    "uuid": "uuid[]",
}


@dataclass(frozen=True, slots=True)
class _Clause:
    column: str
    op: str
    value: object


def _parse_clauses(
    raw: list[dict[str, object]] | None,
) -> list[_Clause]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("predicate must be a list of {column, op, value}")
    clauses: list[_Clause] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise TypeError(
                f"predicate[{index}] must be a mapping; got "
                f"{type(entry).__name__}"
            )
        column = entry.get("column")
        op = entry.get("op")
        if not isinstance(column, str):
            raise TypeError(
                f"predicate[{index}].column must be a string"
            )
        if not isinstance(op, str):
            raise TypeError(
                f"predicate[{index}].op must be a string"
            )
        clauses.append(
            _Clause(column=column, op=op, value=entry.get("value"))
        )
    return clauses


def _compile_clauses(
    table_spec: TableSpec,
    clauses: list[_Clause],
    start_param: int,
) -> tuple[str, tuple[object, ...], int]:
    if not clauses:
        return "", (), start_param
    parts: list[str] = []
    params: list[object] = []
    next_index = start_param
    for clause in clauses:
        if clause.op not in _FILTER_OPS:
            raise ValueError(
                f"unsupported op {clause.op!r}; expected one of "
                f"{sorted(_FILTER_OPS)}"
            )
        column_spec = table_spec.column(clause.column)
        qualified = f"t.{quote_ident(clause.column)}"
        value = coerce_param(clause.value)
        if clause.op == "in":
            if not isinstance(value, list) or len(value) == 0:
                raise ValueError(
                    f"predicate on {clause.column!r} with op='in' "
                    "requires a non-empty list value"
                )
            cast_name = _ARRAY_CAST.get(column_spec.data_type, "text[]")
            parts.append(
                f"{qualified} = ANY(${next_index}::{cast_name})"
            )
            params.append(list(value))
            next_index += 1
            continue
        if clause.op == "like":
            if not isinstance(value, str):
                raise TypeError(
                    f"predicate on {clause.column!r} with op='like' "
                    "requires a string pattern"
                )
            parts.append(f"{qualified} ILIKE ${next_index}")
            params.append(value)
            next_index += 1
            continue
        sql_op = _SQL_BINARY_OPS[clause.op]
        parts.append(f"{qualified} {sql_op} ${next_index}")
        params.append(value)
        next_index += 1
    return f"WHERE {' AND '.join(parts)}", tuple(params), next_index


def _single_column_pk(table_spec: TableSpec) -> str:
    if len(table_spec.primary_key) != 1:
        raise NotImplementedError(
            f"sample supports single-column primary keys only; "
            f"{table_spec.qualified_name!r} has a composite PK"
        )
    return table_spec.primary_key[0]


async def sample(
    session: ComposerSession,
    *,
    table: str,
    n: int = 5,
    seed: int | None = None,
    predicate: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Return up to `n` rows from `table` as dicts of column → value.

    `seed` drives deterministic md5-based ordering; when None, rows come
    back in primary-key ascending order. `predicate` is an optional AND
    of `{column, op, value}` mappings using the same op vocabulary as
    the atomic calculus (eq, in, lt, gt, lte, gte, like).
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    snapshot: SchemaSnapshot = session.snapshot
    table_spec = snapshot.table(table)
    pk = _single_column_pk(table_spec)
    column_names = [column.name for column in table_spec.columns]
    where_sql, params, next_index = _compile_clauses(
        table_spec=table_spec,
        clauses=_parse_clauses(predicate),
        start_param=1,
    )
    order_params: tuple[object, ...]
    if seed is None:
        order_sql = f"ORDER BY t.{quote_ident(pk)} ASC"
        order_params = ()
    else:
        order_sql = (
            f"ORDER BY md5((t.{quote_ident(pk)})::text || ${next_index})"
        )
        order_params = (str(seed),)
    select_parts = ", ".join(
        f"t.{quote_ident(column)}" for column in column_names
    )
    sql = readonly_select(
        f"SELECT {select_parts} "
        f"FROM {quote_table(table_spec.schema, table_spec.name)} AS t "
        f"{where_sql} "
        f"{order_sql} "
        f"LIMIT {int(n)}"
    )
    rows = await session.connection.fetch(sql, *params, *order_params)
    return [{name: row[name] for name in column_names} for row in rows]


__all__ = ["sample"]
