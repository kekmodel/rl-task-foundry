"""CursorPlan — immutable query plan composed by the atomic calculus.

A cursor is an opaque `CursorId` backed by a `CursorPlan`. Plans are
built by `rows_where`, extended by `rows_via` / `intersect` / `order_by`,
and materialized by `take` / `count` / `aggregate` / `group_top`.

Content-hashed IDs give two properties:
- Structurally identical plans produce the same ID (natural deduplication).
- IDs are short opaque strings in agent traces; full plans are server-side.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, NewType, Union

from rl_task_foundry.tooling.common.edges import TypedEdge

CursorId = NewType("CursorId", str)

Direction = Literal["asc", "desc"]

FilterOp = Literal["eq", "in", "lt", "gt", "lte", "gte", "like"]

_FILTER_OPS: frozenset[FilterOp] = frozenset(
    ("eq", "in", "lt", "gt", "lte", "gte", "like")
)


@dataclass(frozen=True, slots=True)
class WhereNode:
    table: str
    column: str
    op: FilterOp
    value: object

    @property
    def target_table(self) -> str:
        return self.table


@dataclass(frozen=True, slots=True)
class ViaNode:
    source: "CursorPlan"
    edge: TypedEdge

    @property
    def target_table(self) -> str:
        return self.edge.destination_table


@dataclass(frozen=True, slots=True)
class IntersectNode:
    left: "CursorPlan"
    right: "CursorPlan"

    @property
    def target_table(self) -> str:
        return self.left.target_table


@dataclass(frozen=True, slots=True)
class OrderNode:
    source: "CursorPlan"
    column: str
    direction: Direction

    @property
    def target_table(self) -> str:
        return self.source.target_table


CursorPlan = Union[WhereNode, ViaNode, IntersectNode, OrderNode]


def plan_target_table(plan: CursorPlan) -> str:
    return plan.target_table


def plan_to_dict(plan: CursorPlan) -> dict[str, object]:
    """Canonical JSON-compatible representation. Used for hashing and
    trace emission; stable keys let two runtimes reproduce the same ID.
    """
    if isinstance(plan, WhereNode):
        return {
            "kind": "where",
            "table": plan.table,
            "column": plan.column,
            "op": plan.op,
            "value": plan.value,
        }
    if isinstance(plan, ViaNode):
        return {
            "kind": "via",
            "source": plan_to_dict(plan.source),
            "edge": {
                "label": plan.edge.label,
                "direction": plan.edge.direction.value,
                "source_table": plan.edge.spec.source_table,
                "source_column": plan.edge.spec.source_column,
                "target_table": plan.edge.spec.target_table,
                "target_column": plan.edge.spec.target_column,
            },
        }
    if isinstance(plan, IntersectNode):
        return {
            "kind": "intersect",
            "left": plan_to_dict(plan.left),
            "right": plan_to_dict(plan.right),
        }
    if isinstance(plan, OrderNode):
        return {
            "kind": "order",
            "source": plan_to_dict(plan.source),
            "column": plan.column,
            "direction": plan.direction,
        }
    raise TypeError(f"unknown plan node: {type(plan).__name__}")


def hash_plan(plan: CursorPlan) -> CursorId:
    payload = json.dumps(
        plan_to_dict(plan), sort_keys=True, separators=(",", ":"), default=str
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return CursorId(f"c_{digest}")


class CursorStore:
    """Per-session map from CursorId to CursorPlan.

    Scope matches one synthesis or solver conversation. Cap prevents
    unbounded growth if the agent keeps composing new cursors.
    """

    def __init__(self, *, max_entries: int = 256) -> None:
        self._plans: dict[CursorId, CursorPlan] = {}
        self._max_entries = max_entries

    def intern(self, plan: CursorPlan) -> CursorId:
        cursor_id = hash_plan(plan)
        if cursor_id not in self._plans:
            if len(self._plans) >= self._max_entries:
                raise RuntimeError(
                    f"CursorStore capacity exceeded ({self._max_entries}); "
                    "reset the conversation or reuse existing cursors"
                )
            self._plans[cursor_id] = plan
        return cursor_id

    def resolve(self, cursor_id: CursorId) -> CursorPlan:
        try:
            return self._plans[cursor_id]
        except KeyError as exc:
            raise KeyError(
                f"cursor {cursor_id!r} not found in this session"
            ) from exc

    def __len__(self) -> int:
        return len(self._plans)


def order_by(
    store: CursorStore,
    cursor_id: CursorId,
    column: str,
    direction: Direction,
) -> CursorId:
    """Annotate a cursor with an ordering.

    order_by does not execute SQL. The annotation is consumed by
    `take` at materialization time.
    """
    if direction not in ("asc", "desc"):
        raise ValueError("direction must be 'asc' or 'desc'")
    source = store.resolve(cursor_id)
    plan = OrderNode(source=source, column=column, direction=direction)
    return store.intern(plan)


__all__ = [
    "CursorId",
    "CursorPlan",
    "CursorStore",
    "Direction",
    "FilterOp",
    "IntersectNode",
    "OrderNode",
    "ViaNode",
    "WhereNode",
    "hash_plan",
    "order_by",
    "plan_target_table",
    "plan_to_dict",
    "_FILTER_OPS",
]
