"""CursorPlan — immutable query plan composed by the atomic calculus.

Internally a cursor is an opaque `CursorId` backed by a `CursorPlan`. The
actor-facing v2 tool surface exposes session-local record-set resource IDs instead
of these hashes, but the execution engine keeps content-hashed cursors so plans
remain immutable and naturally deduplicated.

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

FilterOp = Literal[
    "eq",
    "neq",
    "in",
    "lt",
    "gt",
    "lte",
    "gte",
    "like",
    "is_null",
    "is_not_null",
]

_FILTER_OPS: frozenset[FilterOp] = frozenset(
    ("eq", "neq", "in", "lt", "gt", "lte", "gte", "like", "is_null", "is_not_null")
)


@dataclass(frozen=True, slots=True)
class TableNode:
    table: str

    @property
    def target_table(self) -> str:
        return self.table


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
class FilterNode:
    source: "CursorPlan"
    column: str
    op: FilterOp
    value: object

    @property
    def target_table(self) -> str:
        return self.source.target_table


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
    path: tuple[str, ...] = ()

    @property
    def target_table(self) -> str:
        return self.source.target_table


CursorPlan = Union[TableNode, WhereNode, FilterNode, ViaNode, IntersectNode, OrderNode]


def plan_target_table(plan: CursorPlan) -> str:
    return plan.target_table


def plan_to_dict(plan: CursorPlan) -> dict[str, object]:
    """Canonical JSON-compatible representation. Used for hashing and
    trace emission; stable keys let two runtimes reproduce the same ID.
    """
    if isinstance(plan, TableNode):
        return {
            "kind": "table",
            "table": plan.table,
        }
    if isinstance(plan, WhereNode):
        return {
            "kind": "where",
            "table": plan.table,
            "column": plan.column,
            "op": plan.op,
            "value": plan.value,
        }
    if isinstance(plan, FilterNode):
        return {
            "kind": "filter",
            "source": plan_to_dict(plan.source),
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
                "source_columns": list(plan.edge.spec.source_columns),
                "target_table": plan.edge.spec.target_table,
                "target_columns": list(plan.edge.spec.target_columns),
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
            "path": list(plan.path),
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
        self._public_by_cursor: dict[CursorId, str] = {}
        self._cursor_by_public: dict[str, CursorId] = {}
        self._next_public_index = 1

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

    def expose(self, cursor_id: CursorId) -> str:
        """Return a session-local actor-facing record-set resource id."""
        self.resolve(cursor_id)
        existing = self._public_by_cursor.get(cursor_id)
        if existing is not None:
            return existing
        public_id = f"record_set_{self._next_public_index}"
        self._next_public_index += 1
        self._public_by_cursor[cursor_id] = public_id
        self._cursor_by_public[public_id] = cursor_id
        return public_id

    def resolve_public(self, record_set_id: str) -> CursorId:
        try:
            return self._cursor_by_public[record_set_id]
        except KeyError as exc:
            raise KeyError(
                f"record set {record_set_id!r} not found in this session"
            ) from exc

    def __len__(self) -> int:
        return len(self._plans)


def order_by(
    store: CursorStore,
    cursor_id: CursorId,
    column: str,
    direction: Direction,
    path: tuple[str, ...] = (),
) -> CursorId:
    """Annotate a cursor with an ordering.

    order_by does not execute SQL. The annotation is consumed by
    `take` at materialization time.
    """
    if direction not in ("asc", "desc"):
        raise ValueError("direction must be 'asc' or 'desc'")
    if any(not isinstance(label, str) or not label for label in path):
        raise ValueError("path must contain only non-empty relation labels")
    source = store.resolve(cursor_id)
    plan = OrderNode(source=source, column=column, direction=direction, path=path)
    return store.intern(plan)


__all__ = [
    "CursorId",
    "CursorPlan",
    "CursorStore",
    "Direction",
    "FilterOp",
    "FilterNode",
    "IntersectNode",
    "OrderNode",
    "TableNode",
    "ViaNode",
    "WhereNode",
    "hash_plan",
    "order_by",
    "plan_target_table",
    "plan_to_dict",
    "_FILTER_OPS",
]
