"""sample primitive — representative rows from a table.

Single call returning up to `n` rows of a table, optionally filtered by
an AND of `{column, op, value}` clauses. Ordering is deterministic: when
`seed` is None the result is sorted by primary-key columns ascending; when seed
is supplied rows are ordered by md5 of the key columns concatenated with the
seed, so two runs with the same seed see the same sample. Composite primary
keys are supported.

Authoring-side primitive. Solvers never see it.
"""

from __future__ import annotations

from rl_task_foundry.tooling.common.schema import SchemaSnapshot
from rl_task_foundry.tooling.common.sql import (
    quote_ident,
    quote_table,
    readonly_select,
)
from rl_task_foundry.tooling.composer._session import ComposerSession
from rl_task_foundry.tooling.composer._sql import (
    compile_filter_clauses,
    parse_filter_clauses,
)


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
    back in primary-key ascending order. Composite primary keys are ordered
    column-by-column. `predicate` is an optional AND
    of `{column, op, value}` mappings using the same op vocabulary as
    the authoring query DSL (eq, in, lt, gt, lte, gte, like).
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    snapshot: SchemaSnapshot = session.snapshot
    table_spec = snapshot.table(table)
    column_names = [column.name for column in table_spec.exposed_columns]
    ordering_columns = list(table_spec.primary_key or tuple(column_names))
    if not ordering_columns:
        raise NotImplementedError(
            f"sample requires at least one sortable column for "
            f"{table_spec.qualified_name!r}"
        )
    clause_sql, params, next_index = compile_filter_clauses(
        table_spec=table_spec,
        alias="t",
        clauses=parse_filter_clauses(predicate),
        start_param=1,
    )
    where_sql = f"WHERE {clause_sql}" if clause_sql else ""
    order_params: tuple[object, ...]
    if seed is None:
        order_sql = "ORDER BY " + ", ".join(
            f"t.{quote_ident(column)} ASC" for column in ordering_columns
        )
        order_params = ()
    else:
        key_sql = " || '|' || ".join(
            f"COALESCE((t.{quote_ident(column)})::text, '')"
            for column in ordering_columns
        )
        tie_breaker_sql = ", ".join(
            f"t.{quote_ident(column)} ASC" for column in ordering_columns
        )
        order_sql = (
            f"ORDER BY md5(({key_sql}) || ${next_index}), {tie_breaker_sql}"
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
