"""Solver atomic calculus — async primitives bound to a DB session.

Vertical slice this session: `rows_where`, `order_by` (re-exported from
cursor module), `take`, `read`. `rows_via`, `intersect`, `count`,
`aggregate`, `group_top` land next session.

Each primitive validates its inputs against the `SchemaSnapshot` before
building SQL. Execution goes through asyncpg with read-only session
settings enforced by the caller (see `infra.db.solver_session_settings`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from rl_task_foundry.tooling.atomic.cursor import (
    CursorId,
    CursorStore,
    FilterOp,
    WhereNode,
    _FILTER_OPS,
)
from rl_task_foundry.tooling.atomic.sql_compile import (
    compile_read,
    compile_take,
)
from rl_task_foundry.tooling.common.schema import SchemaSnapshot
from rl_task_foundry.tooling.common.sql import coerce_param


class _ConnectionLike(Protocol):
    async def fetch(self, sql: str, *args: Any) -> list[Any]: ...
    async def fetchrow(self, sql: str, *args: Any) -> Any: ...


@dataclass(slots=True)
class AtomicSession:
    """One solver conversation's calculus state.

    Holds the schema snapshot (read-only), the asyncpg connection, and
    the cursor store. Created once per task attempt and discarded at
    conversation end.
    """

    snapshot: SchemaSnapshot
    connection: _ConnectionLike
    store: CursorStore

    def __post_init__(self) -> None:
        if not isinstance(self.store, CursorStore):
            raise TypeError("store must be a CursorStore")


# ---------- set-producing primitives ----------


def rows_where(
    session: AtomicSession,
    *,
    table: str,
    column: str,
    op: FilterOp,
    value: Any,
) -> CursorId:
    """Build a cursor over `table` filtered by `column op value`.

    No SQL runs here; the plan is deferred until `take` / `count` /
    aggregation materializes it.
    """
    table_spec = session.snapshot.table(table)
    table_spec.column(column)  # validates existence
    if op not in _FILTER_OPS:
        raise ValueError(
            f"op must be one of {sorted(_FILTER_OPS)}; got {op!r}"
        )
    coerced = coerce_param(value)
    plan = WhereNode(table=table, column=column, op=op, value=coerced)
    return session.store.intern(plan)


# ---------- set-materializing primitives ----------


async def take(
    session: AtomicSession,
    *,
    cursor: CursorId,
    n: int,
) -> list[Any]:
    """Materialize up to `n` primary-key values from the cursor,
    honouring any `order_by` annotations and adding a primary-key
    tiebreak for determinism. `n` must be in [2, 5] to prevent
    sort+limit=1 shortcuts.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 2 or n > 5:
        raise ValueError("n must be in [2, 5]")
    plan = session.store.resolve(cursor)
    compiled = compile_take(session.snapshot, plan, n)
    rows = await session.connection.fetch(compiled.sql, *compiled.params)
    return [row["id"] for row in rows]


async def read(
    session: AtomicSession,
    *,
    table: str,
    row_id: Any,
    columns: list[str],
) -> dict[str, Any]:
    """Return the named columns of a single row identified by its
    primary key value.
    """
    column_tuple = tuple(columns)
    compiled = compile_read(session.snapshot, table, row_id, column_tuple)
    row = await session.connection.fetchrow(compiled.sql, *compiled.params)
    if row is None:
        raise LookupError(
            f"no row with primary key {row_id!r} in table {table!r}"
        )
    return {column: row[column] for column in column_tuple}


__all__ = [
    "AtomicSession",
    "read",
    "rows_where",
    "take",
]
