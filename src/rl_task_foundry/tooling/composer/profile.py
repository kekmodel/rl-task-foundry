"""profile primitive — column distribution snapshot.

Two shapes:

- `column=None` — table overview: total row count plus per-column
  distinct/null counts. One SQL round-trip assembled as a single
  aggregate SELECT so every column shows up in one result object.
- `column=<name>` — single-column detail: distinct count, null count,
  min, max, plus a top-k frequency list. Two SQL round-trips
  (aggregates + grouped top-k).

Both shapes honor an optional AND of `{column, op, value}` filter
clauses, letting the composer profile a subset (e.g. `store_id=1`).
"""

from __future__ import annotations

from rl_task_foundry.tooling.common.schema import ColumnSpec, TableSpec
from rl_task_foundry.tooling.common.sql import (
    quote_ident,
    quote_table,
    readonly_select,
)
from rl_task_foundry.tooling.composer._session import ComposerSession
from rl_task_foundry.tooling.composer._sql import (
    coerce_asyncpg_int,
    compile_filter_clauses,
    parse_filter_clauses,
)

_MIN_MAX_TYPES = {
    "integer",
    "bigint",
    "smallint",
    "numeric",
    "decimal",
    "real",
    "double precision",
    "float4",
    "float8",
    "date",
    "timestamp",
    "timestamp without time zone",
    "timestamp with time zone",
    "timestamptz",
    "time",
    "text",
    "character varying",
    "varchar",
    "character",
    "bpchar",
}


def _supports_min_max(column: ColumnSpec) -> bool:
    return column.data_type in _MIN_MAX_TYPES


def _build_where(
    table_spec: TableSpec,
    predicate: list[dict[str, object]] | None,
    start_param: int,
) -> tuple[str, tuple[object, ...], int]:
    clause_sql, params, next_index = compile_filter_clauses(
        table_spec=table_spec,
        alias="t",
        clauses=parse_filter_clauses(predicate),
        start_param=start_param,
    )
    where_sql = f"WHERE {clause_sql}" if clause_sql else ""
    return where_sql, params, next_index


async def _profile_table(
    session: ComposerSession,
    table_spec: TableSpec,
    predicate: list[dict[str, object]] | None,
) -> dict[str, object]:
    where_sql, params, _ = _build_where(table_spec, predicate, 1)
    select_parts: list[str] = ["COUNT(*) AS row_count"]
    columns = table_spec.exposed_columns
    for column in columns:
        ident = quote_ident(column.name)
        select_parts.append(
            f"COUNT(DISTINCT t.{ident}) AS \"distinct_{column.name}\""
        )
        select_parts.append(
            f"SUM(CASE WHEN t.{ident} IS NULL THEN 1 ELSE 0 END) "
            f"AS \"null_{column.name}\""
        )
    sql = readonly_select(
        f"SELECT {', '.join(select_parts)} "
        f"FROM {quote_table(table_spec.schema, table_spec.name)} AS t "
        f"{where_sql}"
    )
    row = await session.connection.fetchrow(sql, *params)
    if row is None:
        row_count = 0
        columns_payload: list[dict[str, object]] = [
            {
                "name": column.name,
                "data_type": column.data_type,
                "visibility": column.visibility,
                "distinct_count": 0,
                "null_count": 0,
            }
            for column in columns
        ]
    else:
        row_count = int(coerce_asyncpg_int(row["row_count"]))
        columns_payload = []
        for column in columns:
            columns_payload.append(
                {
                    "name": column.name,
                    "data_type": column.data_type,
                    "visibility": column.visibility,
                    "distinct_count": int(
                        coerce_asyncpg_int(row[f"distinct_{column.name}"])
                    ),
                    "null_count": int(
                        coerce_asyncpg_int(row[f"null_{column.name}"])
                    ),
                }
            )
    return {
        "table": table_spec.handle,
        "row_count": row_count,
        "columns": columns_payload,
    }


async def _profile_column(
    session: ComposerSession,
    table_spec: TableSpec,
    column_spec: ColumnSpec,
    predicate: list[dict[str, object]] | None,
    top_k: int,
) -> dict[str, object]:
    where_sql, params, _ = _build_where(table_spec, predicate, 1)
    ident = quote_ident(column_spec.name)
    selects: list[str] = [
        "COUNT(*) AS row_count",
        f"COUNT(DISTINCT t.{ident}) AS distinct_count",
        f"SUM(CASE WHEN t.{ident} IS NULL THEN 1 ELSE 0 END) AS null_count",
    ]
    if _supports_min_max(column_spec):
        selects.append(f"MIN(t.{ident}) AS min_value")
        selects.append(f"MAX(t.{ident}) AS max_value")
    aggregate_sql = readonly_select(
        f"SELECT {', '.join(selects)} "
        f"FROM {quote_table(table_spec.schema, table_spec.name)} AS t "
        f"{where_sql}"
    )
    aggregate_row = await session.connection.fetchrow(aggregate_sql, *params)
    row_count = 0
    distinct_count = 0
    null_count = 0
    min_value: object | None = None
    max_value: object | None = None
    if aggregate_row is not None:
        row_count = int(coerce_asyncpg_int(aggregate_row["row_count"]))
        distinct_count = int(coerce_asyncpg_int(aggregate_row["distinct_count"]))
        null_count = int(coerce_asyncpg_int(aggregate_row["null_count"]))
        if _supports_min_max(column_spec):
            min_value = aggregate_row["min_value"]
            max_value = aggregate_row["max_value"]

    top_k_where = (
        f"{where_sql} AND t.{ident} IS NOT NULL"
        if where_sql
        else f"WHERE t.{ident} IS NOT NULL"
    )
    top_k_sql = readonly_select(
        f"SELECT t.{ident} AS value, COUNT(*) AS frequency "
        f"FROM {quote_table(table_spec.schema, table_spec.name)} AS t "
        f"{top_k_where} "
        f"GROUP BY t.{ident} "
        f"ORDER BY frequency DESC, t.{ident} ASC "
        f"LIMIT {int(top_k)}"
    )
    top_k_rows = await session.connection.fetch(top_k_sql, *params)
    top_k_payload = [
        {"value": row["value"], "frequency": int(coerce_asyncpg_int(row["frequency"]))}
        for row in top_k_rows
    ]

    payload: dict[str, object] = {
        "table": table_spec.handle,
        "column": column_spec.name,
        "data_type": column_spec.data_type,
        "row_count": row_count,
        "distinct_count": distinct_count,
        "null_count": null_count,
        "top_k": top_k_payload,
    }
    if _supports_min_max(column_spec):
        payload["min"] = min_value
        payload["max"] = max_value
    return payload


async def profile(
    session: ComposerSession,
    *,
    table: str,
    column: str | None = None,
    predicate: list[dict[str, object]] | None = None,
    top_k: int = 5,
) -> dict[str, object]:
    """Return a distribution profile for a table or a single column.

    Without `column`: row count plus per-column distinct/null counts.
    With `column`: adds min/max (for comparable types) and a top-k
    frequency list excluding NULL.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    table_spec = session.snapshot.table(table)
    if column is None:
        return await _profile_table(session, table_spec, predicate)
    column_spec = table_spec.exposed_column(column)
    return await _profile_column(
        session, table_spec, column_spec, predicate, top_k
    )


__all__ = ["profile"]
