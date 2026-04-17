"""Solver atomic calculus — composition primitives for the RL target.

See docs/spec/tooling-redesign.md. This subpackage exposes ~10 primitives
that, by construction, force the solver to chain operations to reach any
value in the database. There are no bulk-list shortcuts. The surface is
schema-parameterized: one set of functions handles every DB.

Vertical slice delivered this session: `rows_where → order_by → take → read`.
Remaining primitives (rows_via, intersect, count, aggregate, group_top)
land next session along with the agents-SDK tool factory.
"""

from rl_task_foundry.tooling.atomic.calculus import (
    AtomicSession,
    read,
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
    "order_by",
    "read",
    "rows_where",
    "take",
]
