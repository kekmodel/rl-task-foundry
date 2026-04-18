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

import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, cast

import asyncpg

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
    Direction,
    FilterOp,
    _FILTER_OPS,
    order_by,
)
from rl_task_foundry.tooling.atomic.sql_compile import (
    AggregateFn,
    GroupAggregateFn,
    _AGGREGATE_FNS,
    _GROUP_AGGREGATE_FNS,
)
from rl_task_foundry.tooling.common.edges import available_edges
from rl_task_foundry.tooling.common.schema import SchemaSnapshot
from rl_task_foundry.tooling.common.sql import coerce_scalar

if TYPE_CHECKING:
    from agents import FunctionTool


JsonObject = dict[str, object]
Handler = Callable[[JsonObject], Awaitable[JsonObject]]
Invoker = Callable[[object, str], Awaitable[str]]


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


def _filter_op_enum() -> list[str]:
    return sorted(_FILTER_OPS)


def _aggregate_fn_enum() -> list[str]:
    return sorted(_AGGREGATE_FNS)


def _group_aggregate_fn_enum() -> list[str]:
    return sorted(_GROUP_AGGREGATE_FNS)


_VALUE_ANY_OF: list[JsonObject] = [
    {"type": "string"},
    {"type": "number"},
    {"type": "integer"},
    {"type": "boolean"},
    {"type": "null"},
    {"type": "array", "items": {}},
]


def _require_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise TypeError(
            f"{key!r} must be a string; got {type(value).__name__}"
        )
    return value


def _require_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{key!r} must be an integer; got {type(value).__name__}"
        )
    return value


def _require_str_list(payload: JsonObject, key: str) -> list[str]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        raise TypeError(
            f"{key!r} must be a list; got {type(raw).__name__}"
        )
    items: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise TypeError(
                f"{key}[{index}] must be a string; got {type(item).__name__}"
            )
        items.append(item)
    return items


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False)


def _cursor_payload(session: AtomicSession, cursor_id: CursorId) -> JsonObject:
    plan = session.store.resolve(cursor_id)
    return {"cursor_id": str(cursor_id), "target_table": plan.target_table}


def _with_error_handling(handler: Handler) -> Invoker:
    async def invoke(_tool_context: object, input_json: str) -> str:
        try:
            parsed_raw: object = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError as exc:
            return _json_dumps(
                {
                    "error": f"invalid JSON input: {exc}",
                    "error_type": "JSONDecodeError",
                }
            )
        if not isinstance(parsed_raw, dict):
            return _json_dumps(
                {
                    "error": "tool input must be a JSON object",
                    "error_type": "TypeError",
                }
            )
        parsed: JsonObject = {
            str(key): value for key, value in parsed_raw.items()
        }
        try:
            result = await handler(parsed)
        except (
            KeyError,
            ValueError,
            TypeError,
            LookupError,
            RuntimeError,
            asyncpg.exceptions.PostgresError,
        ) as exc:
            return _json_dumps(
                {"error": str(exc), "error_type": type(exc).__name__}
            )
        return _json_dumps(result)

    return invoke


# ---------- individual tool builders ----------


def build_rows_where_tool(session: AtomicSession) -> "FunctionTool":
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    columns = _all_column_names(session.snapshot)
    schema: JsonObject = {
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

    async def handler(payload: JsonObject) -> JsonObject:
        table_name = _require_str(payload, "table")
        column_name = _require_str(payload, "column")
        op_name = _require_str(payload, "op")
        table_spec = session.snapshot.table(table_name)
        column_spec = table_spec.column(column_name)
        value = coerce_scalar(payload.get("value"), column_spec.data_type)
        cursor_id = rows_where(
            session,
            table=table_name,
            column=column_name,
            op=cast(FilterOp, op_name),
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


def build_rows_via_tool(session: AtomicSession) -> "FunctionTool":
    from agents import FunctionTool

    edges = _all_edge_labels(session.snapshot)
    schema: JsonObject = {
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

    async def handler(payload: JsonObject) -> JsonObject:
        cursor = _require_str(payload, "cursor")
        edge_label = _require_str(payload, "edge_label")
        cursor_id = rows_via(
            session,
            cursor=CursorId(cursor),
            edge_label=edge_label,
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


def build_intersect_tool(session: AtomicSession) -> "FunctionTool":
    from agents import FunctionTool

    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["left", "right"],
        "properties": {
            "left": {"type": "string", "description": "Left cursor ID."},
            "right": {"type": "string", "description": "Right cursor ID."},
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        left = _require_str(payload, "left")
        right = _require_str(payload, "right")
        cursor_id = intersect(
            session,
            left=CursorId(left),
            right=CursorId(right),
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


def build_order_by_tool(session: AtomicSession) -> "FunctionTool":
    from agents import FunctionTool

    columns = _all_column_names(session.snapshot)
    schema: JsonObject = {
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

    async def handler(payload: JsonObject) -> JsonObject:
        cursor = _require_str(payload, "cursor")
        column = _require_str(payload, "column")
        direction = _require_str(payload, "direction")
        if direction not in ("asc", "desc"):
            raise ValueError("'direction' must be 'asc' or 'desc'")
        cursor_id = order_by(
            session.store,
            CursorId(cursor),
            column,
            cast(Direction, direction),
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


def build_take_tool(session: AtomicSession) -> "FunctionTool":
    from agents import FunctionTool

    schema: JsonObject = {
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

    async def handler(payload: JsonObject) -> JsonObject:
        cursor = _require_str(payload, "cursor")
        n = _require_int(payload, "n")
        ids = await take(session, cursor=CursorId(cursor), n=n)
        # Composite-PK ids come back as asyncpg Records (tuple-like but
        # not list). Normalise to plain list so downstream JSON
        # serialization doesn't leak Record objects.
        normalised: list[object] = []
        for item in ids:
            if isinstance(item, (tuple, list)):
                normalised.append(list(item))
            elif hasattr(item, "__iter__") and not isinstance(item, (str, bytes, bytearray)):
                normalised.append(list(item))
            else:
                normalised.append(item)
        return {"row_ids": normalised}

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


def build_count_tool(session: AtomicSession) -> "FunctionTool":
    from agents import FunctionTool

    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["cursor"],
        "properties": {"cursor": {"type": "string"}},
    }

    async def handler(payload: JsonObject) -> JsonObject:
        cursor = _require_str(payload, "cursor")
        value = await count(session, cursor=CursorId(cursor))
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


def build_aggregate_tool(session: AtomicSession) -> "FunctionTool":
    from agents import FunctionTool

    columns = _all_column_names(session.snapshot)
    schema: JsonObject = {
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

    async def handler(payload: JsonObject) -> JsonObject:
        cursor = _require_str(payload, "cursor")
        fn_name = _require_str(payload, "fn")
        column = _require_str(payload, "column")
        value = await aggregate(
            session,
            cursor=CursorId(cursor),
            fn=cast(AggregateFn, fn_name),
            column=column,
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


def build_group_top_tool(session: AtomicSession) -> "FunctionTool":
    from agents import FunctionTool

    columns = _all_column_names(session.snapshot)
    schema: JsonObject = {
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

    async def handler(payload: JsonObject) -> JsonObject:
        cursor = _require_str(payload, "cursor")
        group_column = _require_str(payload, "group_column")
        fn_name = _require_str(payload, "fn")
        n = _require_int(payload, "n")
        raw_agg_column = payload.get("agg_column")
        agg_column: str | None
        if raw_agg_column is None:
            agg_column = None
        elif isinstance(raw_agg_column, str):
            agg_column = raw_agg_column
        else:
            raise TypeError(
                "'agg_column' must be a string or null; got "
                f"{type(raw_agg_column).__name__}"
            )
        tops = await group_top(
            session,
            cursor=CursorId(cursor),
            group_column=group_column,
            fn=cast(GroupAggregateFn, fn_name),
            n=n,
            agg_column=agg_column,
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


def build_read_tool(session: AtomicSession) -> "FunctionTool":
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    columns = _all_column_names(session.snapshot)
    row_id_any_of: list[JsonObject] = list(_VALUE_ANY_OF)
    row_id_any_of.append(
        {
            "type": "array",
            "description": (
                "For composite primary keys — one entry per PK column, "
                "in the order declared by the table. take() also emits "
                "composite PKs as arrays."
            ),
            "items": {"anyOf": _VALUE_ANY_OF},
            "minItems": 1,
        }
    )
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["table", "row_id", "columns"],
        "properties": {
            "table": {"type": "string", "enum": tables},
            "row_id": {
                "description": (
                    "Primary-key value of the row. Scalar for single-"
                    "column PKs; array [v1, v2, ...] in PK-column order "
                    "for composite PKs."
                ),
                "anyOf": row_id_any_of,
            },
            "columns": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": columns},
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        table_name = _require_str(payload, "table")
        table_spec = session.snapshot.table(table_name)
        raw_row_id = payload.get("row_id")
        pk_cols = table_spec.primary_key
        if len(pk_cols) == 1:
            pk_column_spec = table_spec.column(pk_cols[0])
            row_id = coerce_scalar(raw_row_id, pk_column_spec.data_type)
        else:
            if not isinstance(raw_row_id, list):
                raise ValueError(
                    f"table {table_name!r} has a composite primary key "
                    f"{list(pk_cols)}; row_id must be an array of length "
                    f"{len(pk_cols)}"
                )
            coerced: list[object] = []
            for component, pk_col in zip(raw_row_id, pk_cols, strict=True):
                col_spec = table_spec.column(pk_col)
                coerced.append(coerce_scalar(component, col_spec.data_type))
            row_id = coerced
        column_list = _require_str_list(payload, "columns")
        row = await read(
            session,
            table=table_name,
            row_id=row_id,
            columns=column_list,
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


def build_atomic_tools(session: AtomicSession) -> list["FunctionTool"]:
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
