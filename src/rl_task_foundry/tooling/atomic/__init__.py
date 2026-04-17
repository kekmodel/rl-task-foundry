"""Solver atomic calculus — composition primitives for the RL target.

See docs/spec/tooling-redesign.md. This subpackage exposes 9 primitives
that, by construction, force the solver to chain operations to reach any
value in the database. There are no bulk-list shortcuts. The surface is
schema-parameterized: one set of functions handles every DB.

- Set-producing:      rows_where, rows_via, intersect
- Set-annotating:     order_by
- Set-materializing:  take, count, aggregate, group_top
- Row-reading:        read
"""

from rl_task_foundry.tooling.atomic.calculus import (
    AtomicSession,
    aggregate,
    count,
    group_top,
    intersect,
    read,
    rows_via,
    rows_where,
    take,
)
from rl_task_foundry.tooling.atomic.cursor import (
    CursorId,
    CursorPlan,
    CursorStore,
    IntersectNode,
    OrderNode,
    ViaNode,
    WhereNode,
    order_by,
)

__all__ = [
    "AtomicSession",
    "CursorId",
    "CursorPlan",
    "CursorStore",
    "IntersectNode",
    "OrderNode",
    "ViaNode",
    "WhereNode",
    "aggregate",
    "count",
    "group_top",
    "intersect",
    "order_by",
    "read",
    "rows_via",
    "rows_where",
    "take",
]
