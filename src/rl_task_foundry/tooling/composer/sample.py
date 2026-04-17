"""sample primitive — representative rows from a table.

Single call returning up to `n` rows of a table, optionally filtered by
an AND of `{column, op, value}` clauses. Ordering is deterministic: when
`seed` is None the result is sorted by primary key ascending; when seed
is supplied rows are ordered by md5 of the PK concatenated with the
seed, so two runs with the same seed see the same sample.

Composer-side primitive. Solvers never see it.
"""

from __future__ import annotations

from rl_task_foundry.tooling.common.schema import SchemaSnapshot, TableSpec
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
    clause_sql, params, next_index = compile_filter_clauses(
        table_spec=table_spec,
        alias="t",
        clauses=parse_filter_clauses(predicate),
        start_param=1,
    )
    where_sql = f"WHERE {clause_sql}" if clause_sql else ""
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
