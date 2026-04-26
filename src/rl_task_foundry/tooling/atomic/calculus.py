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

from dataclasses import dataclass, field
from typing import Protocol, cast

from rl_task_foundry.tooling.atomic.cursor import (
    _FILTER_OPS,
    CursorId,
    CursorStore,
    FilterNode,
    FilterOp,
    IntersectNode,
    TableNode,
    ViaNode,
    WhereNode,
)
from rl_task_foundry.tooling.atomic.sql_compile import (
    _AGGREGATE_FNS,
    _GROUP_AGGREGATE_FNS,
    AggregateFn,
    GroupAggregateFn,
    compile_aggregate,
    compile_count,
    compile_group_top,
    compile_read,
    compile_take,
)
from rl_task_foundry.tooling.common.edges import resolve_edge
from rl_task_foundry.tooling.common.schema import SchemaSnapshot
from rl_task_foundry.tooling.common.sql import coerce_param, coerce_scalar


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
    max_fetch_limit: int = 100
    trace_events: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.store, CursorStore):
            raise TypeError("store must be a CursorStore")
        if not isinstance(self.max_fetch_limit, int) or self.max_fetch_limit < 1:
            raise ValueError("max_fetch_limit must be a positive integer")


# ---------- set-producing primitives ----------


def create_row_set(
    session: AtomicSession,
    *,
    table: str,
) -> CursorId:
    """Build an unfiltered cursor over `table`.

    No SQL runs here; the plan is deferred until materialization.
    """
    table_spec = session.snapshot.table(table)
    plan = TableNode(table=table_spec.handle)
    return session.store.intern(plan)


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
    column_spec = table_spec.column(column)
    if op not in _FILTER_OPS:
        raise ValueError(
            f"op must be one of {sorted(_FILTER_OPS)}; got {op!r}"
        )
    coerced = coerce_param(coerce_scalar(value, column_spec.data_type))
    plan = WhereNode(
        table=table_spec.handle,
        column=column,
        op=op,
        value=coerced,
    )
    return session.store.intern(plan)


def filter_rows(
    session: AtomicSession,
    *,
    cursor: CursorId,
    column: str,
    op: FilterOp,
    value: object,
) -> CursorId:
    """Apply one predicate to an existing cursor's target table."""
    source_plan = session.store.resolve(cursor)
    table_spec = session.snapshot.table(source_plan.target_table)
    column_spec = table_spec.column(column)
    if op not in _FILTER_OPS:
        raise ValueError(
            f"op must be one of {sorted(_FILTER_OPS)}; got {op!r}"
        )
    coerced = coerce_param(coerce_scalar(value, column_spec.data_type))
    plan = FilterNode(source=source_plan, column=column, op=op, value=coerced)
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
    cursor currently points. Destination records are unique by primary key.
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
    offset: int = 0,
) -> list[object]:
    """Materialize up to `n` primary-key values from the cursor,
    honouring any `order_by` annotations and adding a primary-key
    tiebreak for determinism.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise TypeError("offset must be an integer")
    if n < 1:
        raise ValueError("n must be at least 1")
    if n > session.max_fetch_limit:
        raise ValueError(
            f"n must be <= max_fetch_limit ({session.max_fetch_limit})"
        )
    if offset < 0:
        raise ValueError("offset must be non-negative")
    plan = session.store.resolve(cursor)
    compiled = compile_take(session.snapshot, plan, n, offset)
    rows = await session.connection.fetch(compiled.sql, *compiled.params)
    return [row["id"] for row in rows]


async def count(
    session: AtomicSession,
    *,
    cursor: CursorId,
) -> int:
    """Return the count of unique records in the cursor."""
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
    table of the plan; each target record contributes once.
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
    "create_row_set",
    "filter_rows",
    "group_top",
    "intersect",
    "read",
    "rows_via",
    "rows_where",
    "take",
]
