"""Build agents-SDK FunctionTool instances for the atomic calculus.

Each builder closes over an `AtomicSession`, so the tools share one
conversation's snapshot, cursor store, and asyncpg connection. Tools are
schema-parameterized: table/column/edge/op/fn/direction arguments are
baked into the JSON schema as enums derived from the snapshot. Dependent
constraints (e.g. the column must belong to the chosen table) are
validated at runtime by the underlying calculus functions.

The `agents` package is imported lazily so this module is importable
without the SDK present (mirrors `synthesis.submit_draft_tool`).
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Callable

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
from rl_task_foundry.tooling.atomic.cursor import CursorId, order_by
from rl_task_foundry.tooling.atomic.sql_compile import (
    _AGGREGATE_FNS,
    _GROUP_AGGREGATE_FNS,
)
from rl_task_foundry.tooling.atomic.cursor import _FILTER_OPS
from rl_task_foundry.tooling.common.edges import available_edges
from rl_task_foundry.tooling.common.schema import SchemaSnapshot


def _all_table_names(snapshot: SchemaSnapshot) -> list[str]:
    return sorted(snapshot.table_names())


def _all_column_names(snapshot: SchemaSnapshot) -> list[str]:
    names: set[str] = set()
    for table in snapshot.tables:
        for column in table.columns:
            names.add(column.name)
    return sorted(names)


def _all_edge_labels(snapshot: SchemaSnapshot) -> list[str]:
    labels: set[str] = set()
    for table in snapshot.tables:
        for edge in available_edges(snapshot, table.name):
            labels.add(edge.label)
    return sorted(labels)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False)


def _cursor_payload(session: AtomicSession, cursor_id: CursorId) -> dict[str, Any]:
    plan = session.store.resolve(cursor_id)
    return {"cursor_id": str(cursor_id), "target_table": plan.target_table}


def _with_error_handling(
    handler: Callable[[dict[str, Any]], Any],
) -> Callable[[Any, str], Any]:
    async def invoke(_tool_context: Any, input_json: str) -> str:
        try:
            parsed = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError as exc:
            return _json_dumps(
                {"error": f"invalid JSON input: {exc}", "error_type": "JSONDecodeError"}
            )
        try:
            result = await handler(parsed)
        except (KeyError, ValueError, TypeError, LookupError, RuntimeError) as exc:
            return _json_dumps(
                {"error": str(exc), "error_type": type(exc).__name__}
            )
        return _json_dumps(result)

    return invoke


def _filter_op_enum() -> list[str]:
    return sorted(_FILTER_OPS)


def _aggregate_fn_enum() -> list[str]:
    return sorted(_AGGREGATE_FNS)


def _group_aggregate_fn_enum() -> list[str]:
    return sorted(_GROUP_AGGREGATE_FNS)


_VALUE_ANY_OF = [
    {"type": "string"},
    {"type": "number"},
    {"type": "integer"},
    {"type": "boolean"},
    {"type": "null"},
    {"type": "array", "items": {}},
]


_TIMESTAMP_TYPES = {
    "timestamp",
    "timestamp without time zone",
    "timestamp with time zone",
    "timestamptz",
}
_DATE_TYPES = {"date"}
_TIME_TYPES = {"time", "time without time zone", "time with time zone"}


def _coerce_temporal(value: Any, data_type: str) -> Any:
    """Promote ISO-formatted strings arriving via JSON to the temporal
    type expected by asyncpg for the column. Leaves non-strings alone so
    callers that already pass datetime/date/time objects still work.
    """
    if isinstance(value, list):
        return [_coerce_temporal(item, data_type) for item in value]
    if not isinstance(value, str):
        return value
    if data_type in _TIMESTAMP_TYPES:
        return _dt.datetime.fromisoformat(value)
    if data_type in _DATE_TYPES:
        return _dt.date.fromisoformat(value)
    if data_type in _TIME_TYPES:
        return _dt.time.fromisoformat(value)
    return value


# ---------- individual tool builders ----------


def build_rows_where_tool(session: AtomicSession) -> Any:
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    columns = _all_column_names(session.snapshot)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["table", "column", "op", "value"],
        "properties": {
            "table": {
                "type": "string",
                "enum": tables,
                "description": "Table to filter rows of.",
            },
            "column": {
                "type": "string",
                "enum": columns,
                "description": (
                    "Column on the chosen table to compare. Must belong to "
                    "the given table."
                ),
            },
            "op": {
                "type": "string",
                "enum": _filter_op_enum(),
                "description": "Comparison operator.",
            },
            "value": {
                "description": (
                    "Comparison value. For op='in' pass a non-empty list; "
                    "for op='like' pass an ILIKE pattern."
                ),
                "anyOf": _VALUE_ANY_OF,
            },
        },
    }

    async def handler(payload: dict[str, Any]) -> dict[str, Any]:
        table_spec = session.snapshot.table(payload["table"])
        column_spec = table_spec.column(payload["column"])
        value = _coerce_temporal(payload.get("value"), column_spec.data_type)
        cursor_id = rows_where(
            session,
            table=payload["table"],
            column=payload["column"],
            op=payload["op"],
            value=value,
        )
        return _cursor_payload(session, cursor_id)

    return FunctionTool(
        name="rows_where",
        description=(
            "Build a cursor over rows of a single table matching "
            "`column op value`. Returns an opaque cursor_id for "
            "subsequent rows_via / intersect / order_by / take / count / "
            "aggregate / group_top calls."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_rows_via_tool(session: AtomicSession) -> Any:
    from agents import FunctionTool

    edges = _all_edge_labels(session.snapshot)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["cursor", "edge_label"],
        "properties": {
            "cursor": {
                "type": "string",
                "description": "Cursor ID produced by a prior set-producing call.",
            },
            "edge_label": {
                "type": "string",
                "enum": edges,
                "description": (
                    "Directed FK edge label. Forward form: "
                    "'<src>.<col>-><tgt>'. Reverse form: "
                    "'<tgt><-<src>.<col>'. Must originate at the cursor's "
                    "current table."
                ),
            },
        },
    }

    async def handler(payload: dict[str, Any]) -> dict[str, Any]:
        cursor_id = rows_via(
            session,
            cursor=CursorId(payload["cursor"]),
            edge_label=payload["edge_label"],
        )
        return _cursor_payload(session, cursor_id)

    return FunctionTool(
        name="rows_via",
        description=(
            "Project a cursor through a typed foreign-key edge to produce "
            "a new cursor over the destination table. Multiplicity is "
            "preserved (bag semantics); dedup happens inside `take`."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_intersect_tool(session: AtomicSession) -> Any:
    from agents import FunctionTool

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["left", "right"],
        "properties": {
            "left": {"type": "string", "description": "Left cursor ID."},
            "right": {"type": "string", "description": "Right cursor ID."},
        },
    }

    async def handler(payload: dict[str, Any]) -> dict[str, Any]:
        cursor_id = intersect(
            session,
            left=CursorId(payload["left"]),
            right=CursorId(payload["right"]),
        )
        return _cursor_payload(session, cursor_id)

    return FunctionTool(
        name="intersect",
        description=(
            "Set-intersect two cursors over the same target table. The "
            "result is a cursor whose rows appear in both inputs (dedup "
            "applied via SQL INTERSECT)."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_order_by_tool(session: AtomicSession) -> Any:
    from agents import FunctionTool

    columns = _all_column_names(session.snapshot)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["cursor", "column", "direction"],
        "properties": {
            "cursor": {"type": "string"},
            "column": {
                "type": "string",
                "enum": columns,
                "description": "Column on the cursor's target table to sort by.",
            },
            "direction": {
                "type": "string",
                "enum": ["asc", "desc"],
            },
        },
    }

    async def handler(payload: dict[str, Any]) -> dict[str, Any]:
        cursor_id = order_by(
            session.store,
            CursorId(payload["cursor"]),
            payload["column"],
            payload["direction"],
        )
        return _cursor_payload(session, cursor_id)

    return FunctionTool(
        name="order_by",
        description=(
            "Annotate a cursor with an ordering. No SQL runs here; the "
            "annotation is consumed by `take` at materialization time."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_take_tool(session: AtomicSession) -> Any:
    from agents import FunctionTool

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["cursor", "n"],
        "properties": {
            "cursor": {"type": "string"},
            "n": {
                "type": "integer",
                "minimum": 2,
                "maximum": 5,
                "description": (
                    "Number of primary-key values to return. Restricted "
                    "to [2, 5] to prevent sort+limit=1 shortcuts."
                ),
            },
        },
    }

    async def handler(payload: dict[str, Any]) -> dict[str, Any]:
        ids = await take(
            session,
            cursor=CursorId(payload["cursor"]),
            n=int(payload["n"]),
        )
        return {"row_ids": ids}

    return FunctionTool(
        name="take",
        description=(
            "Materialize up to n primary-key values from the cursor, "
            "honouring any `order_by` annotations and a deterministic "
            "primary-key tiebreak."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_count_tool(session: AtomicSession) -> Any:
    from agents import FunctionTool

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["cursor"],
        "properties": {"cursor": {"type": "string"}},
    }

    async def handler(payload: dict[str, Any]) -> dict[str, Any]:
        value = await count(session, cursor=CursorId(payload["cursor"]))
        return {"count": value}

    return FunctionTool(
        name="count",
        description=(
            "Return the bag count of the cursor (multiplicity preserved "
            "through rows_via chains)."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_aggregate_tool(session: AtomicSession) -> Any:
    from agents import FunctionTool

    columns = _all_column_names(session.snapshot)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["cursor", "fn", "column"],
        "properties": {
            "cursor": {"type": "string"},
            "fn": {
                "type": "string",
                "enum": _aggregate_fn_enum(),
            },
            "column": {
                "type": "string",
                "enum": columns,
                "description": "Column on the cursor's target table to aggregate.",
            },
        },
    }

    async def handler(payload: dict[str, Any]) -> dict[str, Any]:
        value = await aggregate(
            session,
            cursor=CursorId(payload["cursor"]),
            fn=payload["fn"],
            column=payload["column"],
        )
        return {"value": value}

    return FunctionTool(
        name="aggregate",
        description=(
            "Scalar aggregate (sum/avg/min/max) of a column over the "
            "cursor's row bag. Multiplicity is preserved."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_group_top_tool(session: AtomicSession) -> Any:
    from agents import FunctionTool

    columns = _all_column_names(session.snapshot)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["cursor", "group_column", "fn", "n"],
        "properties": {
            "cursor": {"type": "string"},
            "group_column": {
                "type": "string",
                "enum": columns,
                "description": "Column to group by (on cursor's target table).",
            },
            "fn": {
                "type": "string",
                "enum": _group_aggregate_fn_enum(),
                "description": (
                    "Aggregate function. 'count' ignores agg_column; "
                    "sum/avg/min/max require it."
                ),
            },
            "agg_column": {
                "description": "Column to aggregate. Required unless fn='count'.",
                "anyOf": [
                    {"type": "string", "enum": columns},
                    {"type": "null"},
                ],
            },
            "n": {
                "type": "integer",
                "minimum": 2,
                "maximum": 5,
            },
        },
    }

    async def handler(payload: dict[str, Any]) -> dict[str, Any]:
        tops = await group_top(
            session,
            cursor=CursorId(payload["cursor"]),
            group_column=payload["group_column"],
            fn=payload["fn"],
            n=int(payload["n"]),
            agg_column=payload.get("agg_column"),
        )
        return {
            "tops": [
                {"group_value": group, "agg_value": value}
                for group, value in tops
            ]
        }

    return FunctionTool(
        name="group_top",
        description=(
            "Return the top n (group_value, agg_value) tuples for the "
            "cursor, ordered by aggregate descending with group_value "
            "ascending as tiebreak. n is restricted to [2, 5]."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_read_tool(session: AtomicSession) -> Any:
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    columns = _all_column_names(session.snapshot)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["table", "row_id", "columns"],
        "properties": {
            "table": {"type": "string", "enum": tables},
            "row_id": {
                "description": "Primary-key value of the row.",
                "anyOf": _VALUE_ANY_OF,
            },
            "columns": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": columns},
            },
        },
    }

    async def handler(payload: dict[str, Any]) -> dict[str, Any]:
        row = await read(
            session,
            table=payload["table"],
            row_id=payload["row_id"],
            columns=list(payload["columns"]),
        )
        return {"row": row}

    return FunctionTool(
        name="read",
        description=(
            "Reveal specific columns of one row identified by its "
            "primary-key value. Separated from rows_where so `find by "
            "condition` and `read attributes` remain distinct primitives."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


# ---------- aggregate entrypoint ----------


def build_atomic_tools(session: AtomicSession) -> list[Any]:
    """Build all nine atomic calculus FunctionTool instances for a session.

    Tool list order matches the prompt-facing calculus grouping:
    set-producing → set-annotating → set-materializing → row-reading.
    """
    return [
        build_rows_where_tool(session),
        build_rows_via_tool(session),
        build_intersect_tool(session),
        build_order_by_tool(session),
        build_take_tool(session),
        build_count_tool(session),
        build_aggregate_tool(session),
        build_group_top_tool(session),
        build_read_tool(session),
    ]


__all__ = [
    "build_aggregate_tool",
    "build_atomic_tools",
    "build_count_tool",
    "build_group_top_tool",
    "build_intersect_tool",
    "build_order_by_tool",
    "build_read_tool",
    "build_rows_via_tool",
    "build_rows_where_tool",
    "build_take_tool",
]
