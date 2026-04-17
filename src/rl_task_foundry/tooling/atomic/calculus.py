"""Solver atomic calculus — async primitives bound to a DB session.

Nine primitives (plus `order_by` re-exported from cursor module):

Set-producing:   rows_where, rows_via, intersect
Set-annotating:  order_by
Set-materializing: take, count, aggregate, group_top
Row-reading:     read

Each primitive validates its inputs against the `SchemaSnapshot` before
building SQL. Execution goes through asyncpg with read-only session
settings enforced by the caller (see `infra.db.solver_session_settings`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from rl_task_foundry.tooling.atomic.cursor import (
    CursorId,
    CursorStore,
    FilterOp,
    IntersectNode,
    ViaNode,
    WhereNode,
    _FILTER_OPS,
)
from rl_task_foundry.tooling.atomic.sql_compile import (
    AggregateFn,
    GroupAggregateFn,
    _AGGREGATE_FNS,
    _GROUP_AGGREGATE_FNS,
    compile_aggregate,
    compile_count,
    compile_group_top,
    compile_read,
    compile_take,
)
from rl_task_foundry.tooling.common.edges import resolve_edge
from rl_task_foundry.tooling.common.schema import SchemaSnapshot
from rl_task_foundry.tooling.common.sql import coerce_param


class _Row(Protocol):
    def __getitem__(self, key: str) -> object: ...


class _ConnectionLike(Protocol):
    async def fetch(self, sql: str, *args: object) -> list[_Row]: ...
    async def fetchrow(self, sql: str, *args: object) -> _Row | None: ...


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
    value: object,
) -> CursorId:
    """Build a cursor over `table` filtered by `column op value`.

    No SQL runs here; the plan is deferred until `take` / `count` /
    aggregation materializes it.
    """
    table_spec = session.snapshot.table(table)
    table_spec.column(column)
    if op not in _FILTER_OPS:
        raise ValueError(
            f"op must be one of {sorted(_FILTER_OPS)}; got {op!r}"
        )
    coerced = coerce_param(value)
    plan = WhereNode(table=table, column=column, op=op, value=coerced)
    return session.store.intern(plan)


def rows_via(
    session: AtomicSession,
    *,
    cursor: CursorId,
    edge_label: str,
) -> CursorId:
    """Project a cursor through a typed FK edge.

    `edge_label` is resolved against the source cursor's target table, so
    the agent never crosses an edge that doesn't originate where the
    cursor currently points. Multiplicity is preserved (bag semantics).
    """
    source_plan = session.store.resolve(cursor)
    origin_table = source_plan.target_table
    edge = resolve_edge(session.snapshot, origin_table, edge_label)
    plan = ViaNode(source=source_plan, edge=edge)
    return session.store.intern(plan)


def intersect(
    session: AtomicSession,
    *,
    left: CursorId,
    right: CursorId,
) -> CursorId:
    """Set intersection of two cursors. Both must point at the same table.

    The SQL layer enforces distinctness (`INTERSECT` dedupes). Use when
    composing "rows matching condition A AND condition B" across two
    independent filter chains.
    """
    left_plan = session.store.resolve(left)
    right_plan = session.store.resolve(right)
    if left_plan.target_table != right_plan.target_table:
        raise ValueError(
            "intersect requires cursors over the same table; got "
            f"{left_plan.target_table!r} and {right_plan.target_table!r}"
        )
    plan = IntersectNode(left=left_plan, right=right_plan)
    return session.store.intern(plan)


# ---------- set-materializing primitives ----------


async def take(
    session: AtomicSession,
    *,
    cursor: CursorId,
    n: int,
) -> list[object]:
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


async def count(
    session: AtomicSession,
    *,
    cursor: CursorId,
) -> int:
    """Return the bag count of the cursor (multiplicity preserved)."""
    plan = session.store.resolve(cursor)
    compiled = compile_count(session.snapshot, plan)
    row = await session.connection.fetchrow(compiled.sql, *compiled.params)
    if row is None:
        return 0
    return cast(int, row["cnt"])


async def aggregate(
    session: AtomicSession,
    *,
    cursor: CursorId,
    fn: AggregateFn,
    column: str,
) -> object:
    """Scalar aggregate over a column of the cursor's target table.

    fn ∈ {sum, avg, min, max}. The column is looked up on the target
    table of the plan; multiplicity is preserved.
    """
    if fn not in _AGGREGATE_FNS:
        raise ValueError(
            f"fn must be one of {sorted(_AGGREGATE_FNS)}; got {fn!r}"
        )
    plan = session.store.resolve(cursor)
    compiled = compile_aggregate(session.snapshot, plan, fn, column)
    row = await session.connection.fetchrow(compiled.sql, *compiled.params)
    if row is None:
        return None
    return row["agg"]


async def group_top(
    session: AtomicSession,
    *,
    cursor: CursorId,
    group_column: str,
    fn: GroupAggregateFn,
    n: int,
    agg_column: str | None = None,
) -> list[tuple[object, object]]:
    """Top-n (group_value, aggregate_value) tuples for the cursor.

    `n ∈ [2, 5]` mirrors `take`'s no-shortcut constraint. `fn='count'`
    counts rows per group and ignores `agg_column`; other aggregates
    require `agg_column`. Tiebreak: group_value ascending.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 2 or n > 5:
        raise ValueError("n must be in [2, 5]")
    if fn not in _GROUP_AGGREGATE_FNS:
        raise ValueError(
            f"fn must be one of {sorted(_GROUP_AGGREGATE_FNS)}; got {fn!r}"
        )
    plan = session.store.resolve(cursor)
    compiled = compile_group_top(
        session.snapshot,
        plan,
        group_column=group_column,
        fn=fn,
        agg_column=agg_column,
        limit=n,
    )
    rows = await session.connection.fetch(compiled.sql, *compiled.params)
    return [(row["group_value"], row["agg_value"]) for row in rows]


# ---------- row reading ----------


async def read(
    session: AtomicSession,
    *,
    table: str,
    row_id: object,
    columns: list[str],
) -> dict[str, object]:
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
    "aggregate",
    "count",
    "group_top",
    "intersect",
    "read",
    "rows_via",
    "rows_where",
    "take",
]
