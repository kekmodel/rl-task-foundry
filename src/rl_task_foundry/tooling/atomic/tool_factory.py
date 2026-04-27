"""Build agents-SDK FunctionTool instances for the atomic resource API.

Each builder closes over an `AtomicSession`, so the tools share one
conversation's snapshot, cursor store, and asyncpg connection. Table
names and small closed vocabularies are baked into JSON schemas, while
resource-local columns and relations are validated at runtime from the
record_set resource metadata.

The `agents` package is imported lazily so this module is importable
without the SDK present (mirrors `synthesis.submit_draft_tool`).
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import asyncpg

from rl_task_foundry.tooling.atomic.calculus import (
    AtomicSession,
    aggregate,
    count,
    filter_rows,
    intersect,
    read,
    rows_via,
    take,
)
from rl_task_foundry.tooling.atomic.calculus import (
    create_row_set as create_record_set,
)
from rl_task_foundry.tooling.atomic.cursor import (
    CursorId,
    Direction,
    FilterOp,
    order_by,
)
from rl_task_foundry.tooling.atomic.sql_compile import (
    _AGGREGATE_FNS,
    AggregateFn,
)
from rl_task_foundry.tooling.common.edges import (
    EdgeDirection,
    TypedEdge,
    available_edges,
    resolve_edge,
)
from rl_task_foundry.tooling.common.payload import (
    JsonObject,
)
from rl_task_foundry.tooling.common.payload import (
    optional_int as _optional_int,
)
from rl_task_foundry.tooling.common.payload import (
    require_int as _require_int,
)
from rl_task_foundry.tooling.common.payload import (
    require_str as _require_str,
)
from rl_task_foundry.tooling.common.payload import (
    require_str_list as _require_str_list,
)
from rl_task_foundry.tooling.common.schema import SchemaSnapshot
from rl_task_foundry.tooling.common.sql import (
    coerce_scalar,
    quote_ident,
    quote_table,
    readonly_select,
)
from rl_task_foundry.tooling.common.tool_runtime import (
    Handler,
    Invoker,
)
from rl_task_foundry.tooling.common.tool_runtime import (
    json_dumps_tool as _json_dumps,
)

if TYPE_CHECKING:
    from agents import FunctionTool


@dataclass(frozen=True, slots=True)
class _ProjectionField:
    name: str
    path: tuple[str, ...]
    column: str
    final_table: str


def _all_table_names(snapshot: SchemaSnapshot) -> list[str]:
    return sorted(snapshot.table_names())


_FILTER_VALUE_SCALAR_ANY_OF: list[JsonObject] = [
    {"type": "string"},
    {"type": "number"},
    {"type": "integer"},
    {"type": "boolean"},
]

_SCALAR_FILTER_OPS = frozenset(("eq", "neq", "lt", "gt", "lte", "gte"))
_NULL_FILTER_OPS = frozenset(("is_null", "is_not_null"))

_ROW_ID_SCALAR_ANY_OF: list[JsonObject] = [
    {"type": "string"},
    {"type": "number"},
    {"type": "integer"},
    {"type": "boolean"},
]


def _row_id_any_of() -> list[JsonObject]:
    return [
        *_ROW_ID_SCALAR_ANY_OF,
        {
            "type": "array",
            "description": (
                "For composite primary keys — one entry per PK column, "
                "in the order declared by the table."
            ),
            "items": {"anyOf": _ROW_ID_SCALAR_ANY_OF},
            "minItems": 1,
        },
    ]


def _table_capabilities(
    snapshot: SchemaSnapshot,
    table_name: str,
) -> JsonObject:
    table = snapshot.table(table_name)
    columns = table.exposed_columns
    return {
        "columns": [column.name for column in columns],
        "column_types": {
            column.name: column.data_type for column in columns
        },
        "column_visibility": {
            column.name: column.visibility for column in columns
        },
        "primary_key": list(table.primary_key),
        "relations": sorted(
            edge.label for edge in available_edges(snapshot, table.handle)
        ),
    }


def _record_set_resource(session: AtomicSession, cursor_id: CursorId) -> JsonObject:
    plan = session.store.resolve(cursor_id)
    return {
        "id": session.store.expose(cursor_id),
        "type": "record_set",
        "table": plan.target_table,
        **_table_capabilities(session.snapshot, plan.target_table),
    }


def _record_set_trace_resource(
    session: AtomicSession,
    cursor_id: CursorId,
) -> JsonObject:
    plan = session.store.resolve(cursor_id)
    return {
        "id": session.store.expose(cursor_id),
        "type": "record_set",
        "table": plan.target_table,
    }


def _record_set_resource_payload(
    session: AtomicSession,
    cursor_id: CursorId,
) -> JsonObject:
    return {
        "ok": True,
        "resource": _record_set_resource(session, cursor_id),
    }


def _resolve_record_set(session: AtomicSession, record_set_id: str) -> CursorId:
    return session.store.resolve_public(record_set_id)


def _normalise_row_id(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return [_normalise_row_id(item) for item in value]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray, dict)):
        return [_normalise_row_id(item) for item in value]
    return value


def _data_payload(data: JsonObject) -> JsonObject:
    return {"ok": True, "data": data}


def _parse_projection_fields(
    session: AtomicSession,
    *,
    source_table: str,
    raw_fields: object,
) -> list[_ProjectionField]:
    if not isinstance(raw_fields, list) or not raw_fields:
        raise TypeError("'fields' must be a non-empty array")
    seen_names: set[str] = set()
    parsed: list[_ProjectionField] = []
    for raw_field in raw_fields:
        if not isinstance(raw_field, dict):
            raise TypeError("each field must be an object")
        name = raw_field.get("name")
        column = raw_field.get("column")
        if not isinstance(name, str) or not name:
            raise TypeError("field.name must be a non-empty string")
        if name in seen_names:
            raise ValueError("field names must be unique")
        seen_names.add(name)
        if not isinstance(column, str) or not column:
            raise TypeError("field.column must be a non-empty string")
        raw_path = raw_field.get("path", [])
        if raw_path is None:
            raw_path = []
        if not isinstance(raw_path, list) or any(
            not isinstance(label, str) or not label for label in raw_path
        ):
            raise TypeError("field.path must be an array of relation labels")
        path = tuple(cast(list[str], raw_path))
        current_table = source_table
        for edge_label in path:
            edge = resolve_edge(session.snapshot, current_table, edge_label)
            current_table = edge.destination_table
        session.snapshot.table(current_table).exposed_column(column)
        parsed.append(
            _ProjectionField(
                name=name,
                path=path,
                column=column,
                final_table=current_table,
            )
        )
    return parsed


def _resolve_relation_path(
    session: AtomicSession,
    *,
    source_table: str,
    path: list[str],
) -> list[TypedEdge]:
    if not path:
        raise ValueError("'path' must contain at least one relation label")
    current_table = source_table
    edges: list[TypedEdge] = []
    for edge_label in path:
        edge = resolve_edge(session.snapshot, current_table, edge_label)
        edges.append(edge)
        current_table = edge.destination_table
    return edges


def _inverse_edge_label(edge: TypedEdge) -> str:
    if edge.direction is EdgeDirection.FORWARD:
        return edge.spec.reverse_label
    return edge.spec.forward_label


def _row_id_cache_key(table: str, row_id: object) -> tuple[str, str]:
    return (
        table,
        json.dumps(
            _normalise_row_id(row_id),
            default=str,
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def _pk_from_record(table_primary_key: tuple[str, ...], row: object) -> object:
    values = [row[f"pk_{index}"] for index, _ in enumerate(table_primary_key)]  # type: ignore[index]
    if len(values) == 1:
        return values[0]
    return values


async def _find_single_record_by_columns(
    session: AtomicSession,
    *,
    table_name: str,
    columns: tuple[str, ...],
    values: tuple[object, ...],
) -> object | None:
    table = session.snapshot.table(table_name)
    if len(columns) != len(values):
        raise ValueError("columns and values must have the same length")
    if not table.primary_key:
        raise ValueError(
            f"table {table.qualified_name!r} has no primary key; related "
            "field materialization requires primary keys"
        )
    selected = ", ".join(
        f"dst.{quote_ident(pk_column)} AS pk_{index}"
        for index, pk_column in enumerate(table.primary_key)
    )
    predicates = []
    coerced_values: list[object] = []
    for index, (column_name, value) in enumerate(zip(columns, values, strict=True)):
        column_spec = table.column(column_name)
        predicates.append(f"dst.{quote_ident(column_name)} = ${index + 1}")
        coerced_values.append(coerce_scalar(value, column_spec.data_type))
    order_by_pk = ", ".join(
        f"dst.{quote_ident(pk_column)} ASC" for pk_column in table.primary_key
    )
    sql = readonly_select(
        f"SELECT {selected} "
        f"FROM {quote_table(table.schema, table.name)} AS dst "
        f"WHERE {' AND '.join(predicates)} "
        f"ORDER BY {order_by_pk} "
        "LIMIT 2"
    )
    rows = await session.connection.fetch(sql, *coerced_values)
    if not rows:
        return None
    if len(rows) > 1:
        raise ValueError(
            "list_records related fields require each path step to return "
            "at most one destination record per source record"
        )
    return _pk_from_record(table.primary_key, rows[0])


async def _resolve_related_record_id(
    session: AtomicSession,
    *,
    source_table: str,
    source_row_id: object,
    path: tuple[str, ...],
) -> tuple[str, object | None]:
    current_table = source_table
    current_row_id: object | None = source_row_id
    for edge_label in path:
        edge = resolve_edge(session.snapshot, current_table, edge_label)
        if current_row_id is None:
            current_table = edge.destination_table
            continue
        origin_values = await read(
            session,
            table=current_table,
            row_id=current_row_id,
            columns=list(edge.origin_columns),
        )
        values = tuple(origin_values[column] for column in edge.origin_columns)
        current_table = edge.destination_table
        if any(value is None for value in values):
            current_row_id = None
            continue
        current_row_id = await _find_single_record_by_columns(
            session,
            table_name=current_table,
            columns=edge.destination_columns,
            values=values,
        )
    return current_table, current_row_id


def _record_success_trace(
    session: AtomicSession,
    *,
    action: str,
    operation: str,
    **fields: object,
) -> None:
    event: JsonObject = {
        "action": action,
        "operation": operation,
        "visible_ok": True,
    }
    event.update(fields)
    session.trace_events.append(event)


def _error_object(error_type: str, code: str) -> JsonObject:
    return {"type": error_type, "code": code}


def _error_payload(
    error_type: str,
    code: str,
    *,
    message: str | None = None,
) -> JsonObject:
    error = _error_object(error_type, code)
    if message:
        error["message"] = message
    return {"ok": False, "error": error}


def _classify_atomic_exception(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, (json.JSONDecodeError, TypeError)):
        return "request_error", "invalid_request"
    if isinstance(exc, ValueError):
        text = str(exc)
        if any(
            marker in text
            for marker in (
                "must be",
                "requires",
                "non-empty",
                "max_fetch_limit",
                "non-negative",
                "at least",
            )
        ):
            return "request_error", "invalid_request"
        return "action_error", "invalid_action"
    if isinstance(exc, KeyError):
        return "action_error", "not_found"
    if isinstance(exc, LookupError):
        return "action_error", "not_found"
    if isinstance(exc, asyncpg.exceptions.PostgresError):
        return "runtime_error", "database_error"
    if isinstance(exc, RuntimeError):
        return "runtime_error", "runtime_error"
    return "runtime_error", "internal_error"


def _record_error_trace(
    session: AtomicSession,
    *,
    operation: str,
    error_type: str,
    code: str,
    request: JsonObject | None = None,
    request_json_type: str | None = None,
) -> None:
    event: JsonObject = {
        "action": "tool_error",
        "operation": operation,
        "visible_ok": False,
        "error": _error_object(error_type, code),
    }
    if request is not None:
        event["request"] = request
    if request_json_type is not None:
        event["request_json_type"] = request_json_type
    session.trace_events.append(event)


def _with_atomic_error_handling(
    session: AtomicSession,
    tool_name: str,
    handler: Handler,
) -> Invoker:
    async def invoke(_tool_context: object, input_json: str) -> str:
        try:
            parsed_raw: object = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError as exc:
            error_type, code = _classify_atomic_exception(exc)
            _record_error_trace(
                session,
                operation=tool_name,
                error_type=error_type,
                code=code,
            )
            return _json_dumps(_error_payload(error_type, code))
        if not isinstance(parsed_raw, dict):
            _record_error_trace(
                session,
                operation=tool_name,
                error_type="request_error",
                code="invalid_request",
                request_json_type=type(parsed_raw).__name__,
            )
            return _json_dumps(
                _error_payload("request_error", "invalid_request")
            )
        parsed: JsonObject = {
            str(key): value for key, value in parsed_raw.items()
        }
        try:
            async with session.operation_lock:
                result = await handler(parsed)
        except (
            KeyError,
            ValueError,
            TypeError,
            LookupError,
            RuntimeError,
            NotImplementedError,
            asyncpg.exceptions.PostgresError,
        ) as exc:
            error_type, code = _classify_atomic_exception(exc)
            _record_error_trace(
                session,
                operation=tool_name,
                error_type=error_type,
                code=code,
                request=parsed,
            )
            message = (
                str(exc)
                if error_type == "request_error"
                and not isinstance(exc, json.JSONDecodeError)
                else None
            )
            return _json_dumps(_error_payload(error_type, code, message=message))
        return _json_dumps(result)

    return invoke


def _make_tool(
    session: AtomicSession,
    *,
    name: str,
    description: str,
    schema: JsonObject,
    handler: Handler,
) -> "FunctionTool":
    from agents import FunctionTool
    from agents.strict_schema import ensure_strict_json_schema

    return FunctionTool(
        name=name,
        description=description,
        params_json_schema=ensure_strict_json_schema(deepcopy(schema)),
        on_invoke_tool=_with_atomic_error_handling(session, name, handler),
        strict_json_schema=True,
    )


# ---------- resource-oriented v2 builders ----------


def build_create_record_set_tool(session: AtomicSession) -> "FunctionTool":
    tables = _all_table_names(session.snapshot)
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["table"],
        "properties": {
            "table": {
                "type": "string",
                "enum": tables,
                "description": "Table to create an unfiltered record_set resource from.",
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        table_name = _require_str(payload, "table")
        cursor_id = create_record_set(session, table=table_name)
        _record_success_trace(
            session,
            action="create_resource",
            operation="create_record_set",
            output_resource=_record_set_trace_resource(session, cursor_id),
        )
        return _record_set_resource_payload(session, cursor_id)

    return _make_tool(
        session,
        name="create_record_set",
        description=(
            "Create an unfiltered record_set resource for one table. Returns "
            "{ok, resource} with the record_set id, table, columns, primary key, "
            "and available relations."
        ),
        schema=schema,
        handler=handler,
    )


def build_filter_record_set_tool(session: AtomicSession) -> "FunctionTool":
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record_set_id", "column", "op", "value"],
        "properties": {
            "record_set_id": {
                "type": "string",
                "description": "ID of an existing record_set resource.",
            },
            "column": {
                "type": "string",
                "description": (
                    "Column on the record_set resource's table to compare. "
                    "Must be one of the record_set resource's columns."
                ),
            },
            "op": {
                "type": "string",
                "enum": sorted(_SCALAR_FILTER_OPS),
                "description": "Scalar comparison operator that requires a non-null value.",
            },
            "value": {
                "description": (
                    "Single comparison value. Must be a scalar and must not be null."
                ),
                "anyOf": _FILTER_VALUE_SCALAR_ANY_OF,
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        record_set_id = _require_str(payload, "record_set_id")
        column_name = _require_str(payload, "column")
        op_name = _require_str(payload, "op")
        cursor_id = _resolve_record_set(session, record_set_id)
        source_resource = _record_set_trace_resource(session, cursor_id)
        source_plan = session.store.resolve(cursor_id)
        table_spec = session.snapshot.table(source_plan.target_table)
        column_spec = table_spec.exposed_column(column_name)
        if op_name not in _SCALAR_FILTER_OPS:
            raise ValueError(
                f"'op' must be one of {sorted(_SCALAR_FILTER_OPS)}"
            )
        if "value" not in payload or payload.get("value") is None:
            raise TypeError(
                "'value' is required and cannot be null for this operator"
            )
        if isinstance(payload.get("value"), list | dict):
            raise TypeError("'value' must be a scalar for this endpoint")
        value = coerce_scalar(payload.get("value"), column_spec.data_type)
        filtered = filter_rows(
            session,
            cursor=cursor_id,
            column=column_name,
            op=cast(FilterOp, op_name),
            value=value,
        )
        _record_success_trace(
            session,
            action="transform_resource",
            operation="filter_record_set",
            input_resource=source_resource,
            predicate={
                "column": column_name,
                "op": op_name,
                "value": value,
            },
            output_resource=_record_set_trace_resource(session, filtered),
        )
        return _record_set_resource_payload(session, filtered)

    return _make_tool(
        session,
        name="filter_record_set",
        description=(
            "Create a new record_set resource by applying one scalar comparison "
            "predicate to an existing record_set. The predicate column must "
            "belong to the record_set resource's table."
        ),
        schema=schema,
        handler=handler,
    )


def build_filter_record_set_by_values_tool(session: AtomicSession) -> "FunctionTool":
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record_set_id", "column", "values"],
        "properties": {
            "record_set_id": {
                "type": "string",
                "description": "ID of an existing record_set resource.",
            },
            "column": {
                "type": "string",
                "description": (
                    "Column on the record_set resource's table to compare. "
                    "Must be one of the record_set resource's columns."
                ),
            },
            "values": {
                "type": "array",
                "minItems": 1,
                "description": (
                    "Allowed scalar values for this column. Keeps records whose "
                    "column value equals any listed value."
                ),
                "items": {"anyOf": _FILTER_VALUE_SCALAR_ANY_OF},
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        record_set_id = _require_str(payload, "record_set_id")
        column_name = _require_str(payload, "column")
        cursor_id = _resolve_record_set(session, record_set_id)
        source_resource = _record_set_trace_resource(session, cursor_id)
        source_plan = session.store.resolve(cursor_id)
        table_spec = session.snapshot.table(source_plan.target_table)
        column_spec = table_spec.exposed_column(column_name)
        raw_values = payload.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise TypeError("'values' must be a non-empty array")
        if any(value is None or isinstance(value, list | dict) for value in raw_values):
            raise TypeError("'values' must contain only non-null scalar values")
        values = coerce_scalar(raw_values, column_spec.data_type)
        filtered = filter_rows(
            session,
            cursor=cursor_id,
            column=column_name,
            op="in",
            value=values,
        )
        _record_success_trace(
            session,
            action="transform_resource",
            operation="filter_record_set_by_values",
            input_resource=source_resource,
            predicate={
                "column": column_name,
                "op": "in",
                "value": values,
            },
            output_resource=_record_set_trace_resource(session, filtered),
        )
        return _record_set_resource_payload(session, filtered)

    return _make_tool(
        session,
        name="filter_record_set_by_values",
        description=(
            "Create a new record_set resource by keeping records whose column "
            "value equals one of the provided scalar values."
        ),
        schema=schema,
        handler=handler,
    )


def build_filter_record_set_by_pattern_tool(session: AtomicSession) -> "FunctionTool":
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record_set_id", "column", "pattern"],
        "properties": {
            "record_set_id": {
                "type": "string",
                "description": "ID of an existing record_set resource.",
            },
            "column": {
                "type": "string",
                "description": (
                    "Text-like column on the record_set resource's table to "
                    "match. Must be one of the record_set resource's columns."
                ),
            },
            "pattern": {
                "type": "string",
                "description": (
                    "Case-insensitive text pattern. Use % as a wildcard when "
                    "matching a substring."
                ),
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        record_set_id = _require_str(payload, "record_set_id")
        column_name = _require_str(payload, "column")
        pattern = _require_str(payload, "pattern")
        cursor_id = _resolve_record_set(session, record_set_id)
        source_resource = _record_set_trace_resource(session, cursor_id)
        source_plan = session.store.resolve(cursor_id)
        table_spec = session.snapshot.table(source_plan.target_table)
        table_spec.exposed_column(column_name)
        filtered = filter_rows(
            session,
            cursor=cursor_id,
            column=column_name,
            op="like",
            value=pattern,
        )
        _record_success_trace(
            session,
            action="transform_resource",
            operation="filter_record_set_by_pattern",
            input_resource=source_resource,
            predicate={
                "column": column_name,
                "op": "like",
                "value": pattern,
            },
            output_resource=_record_set_trace_resource(session, filtered),
        )
        return _record_set_resource_payload(session, filtered)

    return _make_tool(
        session,
        name="filter_record_set_by_pattern",
        description=(
            "Create a new record_set resource by applying one case-insensitive "
            "text pattern predicate to an existing record_set."
        ),
        schema=schema,
        handler=handler,
    )


def build_filter_record_set_by_null_tool(session: AtomicSession) -> "FunctionTool":
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record_set_id", "column", "op"],
        "properties": {
            "record_set_id": {
                "type": "string",
                "description": "ID of an existing record_set resource.",
            },
            "column": {
                "type": "string",
                "description": (
                    "Column on the record_set resource's table to test for "
                    "missing or present values."
                ),
            },
            "op": {
                "type": "string",
                "enum": sorted(_NULL_FILTER_OPS),
                "description": "Null-test operator.",
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        record_set_id = _require_str(payload, "record_set_id")
        column_name = _require_str(payload, "column")
        op_name = _require_str(payload, "op")
        if op_name not in _NULL_FILTER_OPS:
            raise ValueError(
                f"'op' must be one of {sorted(_NULL_FILTER_OPS)}"
            )
        cursor_id = _resolve_record_set(session, record_set_id)
        source_resource = _record_set_trace_resource(session, cursor_id)
        source_plan = session.store.resolve(cursor_id)
        table_spec = session.snapshot.table(source_plan.target_table)
        table_spec.exposed_column(column_name)
        filtered = filter_rows(
            session,
            cursor=cursor_id,
            column=column_name,
            op=cast(FilterOp, op_name),
            value=None,
        )
        _record_success_trace(
            session,
            action="transform_resource",
            operation="filter_record_set_by_null",
            input_resource=source_resource,
            predicate={
                "column": column_name,
                "op": op_name,
            },
            output_resource=_record_set_trace_resource(session, filtered),
        )
        return _record_set_resource_payload(session, filtered)

    return _make_tool(
        session,
        name="filter_record_set_by_null",
        description=(
            "Create a new record_set resource by keeping records where one "
            "column is null or not null."
        ),
        schema=schema,
        handler=handler,
    )


def build_filter_record_set_by_related_tool(session: AtomicSession) -> "FunctionTool":
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record_set_id", "path", "column", "op", "value"],
        "properties": {
            "record_set_id": {
                "type": "string",
                "description": "ID of an existing source record_set resource.",
            },
            "path": {
                "type": "array",
                "minItems": 1,
                "description": (
                    "Relation labels to follow from the source record_set table "
                    "to the related table that owns column."
                ),
                "items": {
                    "type": "string",
                    "description": (
                        "Relation label copied from the current record_set "
                        "resource's relations list, then from each returned "
                        "table's relations list."
                    ),
                },
            },
            "column": {
                "type": "string",
                "description": (
                    "Column on the related table reached after path. The related "
                    "column must be exposed by that table."
                ),
            },
            "op": {
                "type": "string",
                "enum": sorted(_SCALAR_FILTER_OPS),
                "description": "Scalar comparison operator that requires a non-null value.",
            },
            "value": {
                "description": (
                    "Single comparison value for the related column. Must be a "
                    "scalar and must not be null."
                ),
                "anyOf": _FILTER_VALUE_SCALAR_ANY_OF,
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        record_set_id = _require_str(payload, "record_set_id")
        raw_path = _require_str_list(payload, "path")
        column_name = _require_str(payload, "column")
        op_name = _require_str(payload, "op")
        if op_name not in _SCALAR_FILTER_OPS:
            raise ValueError(
                f"'op' must be one of {sorted(_SCALAR_FILTER_OPS)}"
            )
        if "value" not in payload or payload.get("value") is None:
            raise TypeError(
                "'value' is required and cannot be null for this operator"
            )
        if isinstance(payload.get("value"), list | dict):
            raise TypeError("'value' must be a scalar for this endpoint")

        source_cursor = _resolve_record_set(session, record_set_id)
        source_resource = _record_set_trace_resource(session, source_cursor)
        source_table = session.store.resolve(source_cursor).target_table
        edges = _resolve_relation_path(
            session,
            source_table=source_table,
            path=raw_path,
        )
        related_table = edges[-1].destination_table
        related_spec = session.snapshot.table(related_table)
        column_spec = related_spec.exposed_column(column_name)
        value = coerce_scalar(payload.get("value"), column_spec.data_type)

        related_cursor = create_record_set(session, table=related_table)
        matching_related = filter_rows(
            session,
            cursor=related_cursor,
            column=column_name,
            op=cast(FilterOp, op_name),
            value=value,
        )
        back_projected = matching_related
        for edge in reversed(edges):
            back_projected = rows_via(
                session,
                cursor=back_projected,
                edge_label=_inverse_edge_label(edge),
            )
        filtered = intersect(
            session,
            left=source_cursor,
            right=back_projected,
        )
        _record_success_trace(
            session,
            action="transform_resource",
            operation="filter_record_set_by_related",
            input_resource=source_resource,
            related_predicate={
                "path": raw_path,
                "column": column_name,
                "op": op_name,
                "value": value,
            },
            output_resource=_record_set_trace_resource(session, filtered),
        )
        return _record_set_resource_payload(session, filtered)

    return _make_tool(
        session,
        name="filter_record_set_by_related",
        description=(
            "Create a new source record_set by keeping source records that have "
            "at least one related record, reached through path, whose column "
            "matches one scalar comparison."
        ),
        schema=schema,
        handler=handler,
    )


def build_follow_relation_tool(session: AtomicSession) -> "FunctionTool":
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_record_set_id", "edge_label"],
        "properties": {
            "source_record_set_id": {
                "type": "string",
                "description": "ID of the record_set resource to traverse from.",
            },
            "edge_label": {
                "type": "string",
                "description": (
                    "Directed relation label. Forward form: '<src>.<col>-><tgt>'. "
                    "Reverse form: '<tgt><-<src>.<col>'. Must be one of the "
                    "source record_set resource's relations. Copy the exact "
                    "string from resource.relations; do not infer, translate, "
                    "paraphrase, or combine table/column names."
                ),
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        record_set_id = _require_str(payload, "source_record_set_id")
        edge_label = _require_str(payload, "edge_label")
        cursor_id = _resolve_record_set(session, record_set_id)
        source_resource = _record_set_trace_resource(session, cursor_id)
        projected = rows_via(
            session,
            cursor=cursor_id,
            edge_label=edge_label,
        )
        _record_success_trace(
            session,
            action="transform_resource",
            operation="follow_relation",
            input_resource=source_resource,
            relation={"edge_label": edge_label},
            output_resource=_record_set_trace_resource(session, projected),
        )
        return _record_set_resource_payload(session, projected)

    return _make_tool(
        session,
        name="follow_relation",
        description=(
            "Create a new record_set resource of unique destination records by "
            "following exactly one directed relation from the source record_set "
            "resource."
        ),
        schema=schema,
        handler=handler,
    )


def build_intersect_record_sets_tool(session: AtomicSession) -> "FunctionTool":
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["left_record_set_id", "right_record_set_id"],
        "properties": {
            "left_record_set_id": {
                "type": "string",
                "description": "ID of the first record_set resource.",
            },
            "right_record_set_id": {
                "type": "string",
                "description": "ID of the second record_set resource.",
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        left_record_set_id = _require_str(payload, "left_record_set_id")
        right_record_set_id = _require_str(payload, "right_record_set_id")
        left = _resolve_record_set(session, left_record_set_id)
        right = _resolve_record_set(session, right_record_set_id)
        left_resource = _record_set_trace_resource(session, left)
        right_resource = _record_set_trace_resource(session, right)
        combined = intersect(session, left=left, right=right)
        _record_success_trace(
            session,
            action="transform_resource",
            operation="intersect_record_sets",
            input_resources=[left_resource, right_resource],
            output_resource=_record_set_trace_resource(session, combined),
        )
        return _record_set_resource_payload(session, combined)

    return _make_tool(
        session,
        name="intersect_record_sets",
        description=(
            "Create a new record_set resource containing records present in "
            "both input record_set resources. Both inputs must target the same "
            "table."
        ),
        schema=schema,
        handler=handler,
    )


def build_sort_record_set_tool(session: AtomicSession) -> "FunctionTool":
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record_set_id", "column", "direction"],
        "properties": {
            "record_set_id": {
                "type": "string",
                "description": "ID of the record_set resource to order.",
            },
            "column": {
                "type": "string",
                "description": (
                    "Column on the record_set resource's table to sort by. "
                    "Must be one of the record_set resource's columns."
                ),
            },
            "direction": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "Sort direction.",
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        record_set_id = _require_str(payload, "record_set_id")
        column = _require_str(payload, "column")
        direction = _require_str(payload, "direction")
        if direction not in ("asc", "desc"):
            raise ValueError("'direction' must be 'asc' or 'desc'")
        cursor_id = _resolve_record_set(session, record_set_id)
        source_resource = _record_set_trace_resource(session, cursor_id)
        plan = session.store.resolve(cursor_id)
        session.snapshot.table(plan.target_table).exposed_column(column)
        sorted_cursor = order_by(
            session.store,
            cursor_id,
            column,
            cast(Direction, direction),
        )
        _record_success_trace(
            session,
            action="transform_resource",
            operation="sort_record_set",
            input_resource=source_resource,
            sort={"column": column, "direction": direction},
            output_resource=_record_set_trace_resource(session, sorted_cursor),
        )
        return _record_set_resource_payload(session, sorted_cursor)

    return _make_tool(
        session,
        name="sort_record_set",
        description=(
            "Create a new record_set resource with a stable ordering. Listing "
            "record references from the new record_set uses this order."
        ),
        schema=schema,
        handler=handler,
    )


def build_list_record_refs_tool(session: AtomicSession) -> "FunctionTool":
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record_set_id", "limit"],
        "properties": {
            "record_set_id": {
                "type": "string",
                "description": "ID of the record_set resource to list from.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": session.max_fetch_limit,
                "description": "Maximum number of record_ref items to return.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "Number of record_ref items to skip.",
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        record_set_id = _require_str(payload, "record_set_id")
        limit = _require_int(payload, "limit")
        offset = _optional_int(payload, "offset")
        cursor_id = _resolve_record_set(session, record_set_id)
        source_resource = _record_set_trace_resource(session, cursor_id)
        table = session.store.resolve(cursor_id).target_table
        offset_value = offset if offset is not None else 0
        ids = await take(
            session,
            cursor=cursor_id,
            n=limit,
            offset=offset_value,
        )
        items = [
            {"type": "record_ref", "table": table, "id": _normalise_row_id(row_id)}
            for row_id in ids
        ]
        _record_success_trace(
            session,
            action="materialize_resource",
            operation="list_record_refs",
            input_resource=source_resource,
            pagination={"limit": limit, "offset": offset_value},
            result_shape={
                "kind": "record_ref_list",
                "table": table,
                "returned": len(items),
            },
        )
        return _data_payload(
            {
                "items": items,
                "limit": limit,
                "offset": offset_value,
                "returned": len(items),
            }
        )

    return _make_tool(
        session,
        name="list_record_refs",
        description=(
            "Return a paginated list of record_ref objects from a record_set "
            "resource. Record data is not included in this response."
        ),
        schema=schema,
        handler=handler,
    )


def build_list_records_tool(session: AtomicSession) -> "FunctionTool":
    field_schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "column", "path"],
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Output key to use for this field in each returned item."
                ),
            },
            "column": {
                "type": "string",
                "description": (
                    "Column to read from the source table, or from the table "
                    "reached after path."
                ),
            },
            "path": {
                "type": "array",
                "default": [],
                "description": (
                    "Relation labels to follow from each source record before "
                    "reading column. Use [] for a source-table field. Each "
                    "step must produce at most one related record for each "
                    "source record. Put only relation labels from the "
                    "record_set relations list in path, not foreign-key "
                    "column names; put the final field name in column."
                ),
                "items": {
                    "type": "string",
                    "description": (
                        "Relation label copied exactly from a record_set "
                        "resource's relations list. Do not include column "
                        "names here."
                    ),
                },
            },
        },
    }
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record_set_id", "limit", "offset", "fields"],
        "properties": {
            "record_set_id": {
                "type": "string",
                "description": "ID of the record_set resource to list from.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": session.max_fetch_limit,
                "description": "Maximum number of records to return.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "Number of records to skip. Use 0 for the first page.",
            },
            "fields": {
                "type": "array",
                "minItems": 1,
                "description": (
                    "Fields to return for each source record. Direct fields "
                    "read source-table columns; related fields preserve source "
                    "record order while following single-record relation paths."
                ),
                "items": field_schema,
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        record_set_id = _require_str(payload, "record_set_id")
        limit = _require_int(payload, "limit")
        offset = _optional_int(payload, "offset")
        cursor_id = _resolve_record_set(session, record_set_id)
        source_resource = _record_set_trace_resource(session, cursor_id)
        source_table = session.store.resolve(cursor_id).target_table
        fields = _parse_projection_fields(
            session,
            source_table=source_table,
            raw_fields=payload.get("fields"),
        )
        offset_value = offset if offset is not None else 0
        row_ids = await take(
            session,
            cursor=cursor_id,
            n=limit,
            offset=offset_value,
        )
        direct_columns = list(
            dict.fromkeys(field.column for field in fields if not field.path)
        )
        related_id_cache: dict[tuple[str, str, tuple[str, ...]], object | None] = {}
        items: list[JsonObject] = []
        for row_id in row_ids:
            direct_values = (
                await read(
                    session,
                    table=source_table,
                    row_id=row_id,
                    columns=direct_columns,
                )
                if direct_columns
                else {}
            )
            item: JsonObject = {}
            for field in fields:
                if not field.path:
                    item[field.name] = direct_values[field.column]
                    continue
                cache_key = (
                    *_row_id_cache_key(source_table, row_id),
                    field.path,
                )
                related_id = related_id_cache.get(cache_key)
                if cache_key not in related_id_cache:
                    resolved_table, related_id = await _resolve_related_record_id(
                        session,
                        source_table=source_table,
                        source_row_id=row_id,
                        path=field.path,
                    )
                    if resolved_table != field.final_table:
                        raise RuntimeError("resolved relation path changed tables")
                    related_id_cache[cache_key] = related_id
                if related_id is None:
                    item[field.name] = None
                    continue
                related_values = await read(
                    session,
                    table=field.final_table,
                    row_id=related_id,
                    columns=[field.column],
                )
                item[field.name] = related_values[field.column]
            items.append(item)
        _record_success_trace(
            session,
            action="materialize_resource",
            operation="list_records",
            input_resource=source_resource,
            fields=[
                {
                    "name": field.name,
                    "path": list(field.path),
                    "column": field.column,
                }
                for field in fields
            ],
            pagination={"limit": limit, "offset": offset_value},
            result_shape={
                "kind": "record_list",
                "table": source_table,
                "returned": len(items),
            },
        )
        return _data_payload(
            {
                "items": items,
                "limit": limit,
                "offset": offset_value,
                "returned": len(items),
            }
        )

    return _make_tool(
        session,
        name="list_records",
        description=(
            "Return selected field values for records in a record_set, preserving "
            "the record_set order. Fields may read source columns or columns on "
            "single-record relation paths."
        ),
        schema=schema,
        handler=handler,
    )


def build_count_records_tool(session: AtomicSession) -> "FunctionTool":
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record_set_id"],
        "properties": {
            "record_set_id": {
                "type": "string",
                "description": "ID of the record_set resource to count.",
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        cursor_id = _resolve_record_set(session, _require_str(payload, "record_set_id"))
        source_resource = _record_set_trace_resource(session, cursor_id)
        value = await count(session, cursor=cursor_id)
        _record_success_trace(
            session,
            action="materialize_resource",
            operation="count_records",
            input_resource=source_resource,
            result_shape={"kind": "scalar", "field": "count"},
        )
        return _data_payload({"count": value})

    return _make_tool(
        session,
        name="count_records",
        description=(
            "Return the number of unique records in a record_set resource."
        ),
        schema=schema,
        handler=handler,
    )


def build_aggregate_records_tool(session: AtomicSession) -> "FunctionTool":
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record_set_id", "fn", "column"],
        "properties": {
            "record_set_id": {
                "type": "string",
                "description": "ID of the record_set resource to aggregate.",
            },
            "fn": {
                "type": "string",
                "enum": sorted(_AGGREGATE_FNS),
                "description": "Aggregate function.",
            },
            "column": {
                "type": "string",
                "description": (
                    "Column on the record_set resource's table to aggregate. "
                    "Must be one of the record_set resource's columns. Use "
                    "sum/avg only with numeric column_types."
                ),
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        cursor_id = _resolve_record_set(session, _require_str(payload, "record_set_id"))
        source_resource = _record_set_trace_resource(session, cursor_id)
        fn_name = _require_str(payload, "fn")
        column = _require_str(payload, "column")
        plan = session.store.resolve(cursor_id)
        session.snapshot.table(plan.target_table).exposed_column(column)
        value = await aggregate(
            session,
            cursor=cursor_id,
            fn=cast(AggregateFn, fn_name),
            column=column,
        )
        _record_success_trace(
            session,
            action="materialize_resource",
            operation="aggregate_records",
            input_resource=source_resource,
            aggregate={"fn": fn_name, "column": column},
            result_shape={"kind": "scalar", "field": "value"},
        )
        return _data_payload({"value": value})

    return _make_tool(
        session,
        name="aggregate_records",
        description=(
            "Return sum, avg, min, or max for one column over a record_set "
            "resource. Each record contributes once; use sum/avg only with "
            "numeric column_types."
        ),
        schema=schema,
        handler=handler,
    )


def build_get_record_tool(session: AtomicSession) -> "FunctionTool":
    tables = _all_table_names(session.snapshot)
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["table", "record_id", "columns"],
        "properties": {
            "table": {
                "type": "string",
                "enum": tables,
                "description": (
                    "Table handle containing the record. When using a "
                    "record_ref, copy record_ref.table."
                ),
            },
            "record_id": {
                "description": (
                    "Primary-key value of the record. Scalar for single-column "
                    "PKs; array for composite PKs. When using a record_ref, "
                    "copy record_ref.id."
                ),
                "anyOf": _row_id_any_of(),
            },
            "columns": {
                "type": "array",
                "minItems": 1,
                "description": (
                    "Columns to return from the requested table. Each column "
                    "must be present in that table's resource columns."
                ),
                "items": {
                    "type": "string",
                    "description": (
                        "Column to return from the requested table."
                    ),
                },
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        table_name = _require_str(payload, "table")
        table_spec = session.snapshot.table(table_name)
        table_handle = table_spec.handle
        raw_record_id = payload.get("record_id")
        if raw_record_id is None:
            raise TypeError("'record_id' is required and cannot be null")
        pk_cols = table_spec.primary_key
        if len(pk_cols) == 1:
            pk_column_spec = table_spec.column(pk_cols[0])
            row_id = coerce_scalar(raw_record_id, pk_column_spec.data_type)
        else:
            if not isinstance(raw_record_id, list):
                raise ValueError(
                    f"table {table_name!r} has a composite primary key; "
                    "record_id must be an array"
                )
            if len(raw_record_id) != len(pk_cols):
                raise ValueError(
                    f"table {table_name!r} has a composite primary key; "
                    f"record_id must be an array of length {len(pk_cols)}"
                )
            coerced: list[object] = []
            for component, pk_col in zip(raw_record_id, pk_cols, strict=True):
                col_spec = table_spec.column(pk_col)
                coerced.append(coerce_scalar(component, col_spec.data_type))
            row_id = coerced
        column_list = _require_str_list(payload, "columns")
        for column_name in column_list:
            table_spec.exposed_column(column_name)
        row = await read(
            session,
            table=table_handle,
            row_id=row_id,
            columns=column_list,
        )
        _record_success_trace(
            session,
            action="read_resource",
            operation="get_record",
            record_ref={
                "type": "record_ref",
                "table": table_handle,
                "id": _normalise_row_id(row_id),
            },
            columns=column_list,
            result_shape={"kind": "object", "field": "record"},
        )
        return _data_payload({"record": row})

    return _make_tool(
        session,
        name="get_record",
        description=(
            "Return selected fields for one table record identified by primary-key "
            "value. Use table and id values copied from a record_ref item."
        ),
        schema=schema,
        handler=handler,
    )


# ---------- aggregate entrypoint ----------


def build_atomic_tools(session: AtomicSession) -> list["FunctionTool"]:
    """Build the solver-facing atomic resource API tools."""
    return [
        build_create_record_set_tool(session),
        build_filter_record_set_tool(session),
        build_filter_record_set_by_values_tool(session),
        build_filter_record_set_by_pattern_tool(session),
        build_filter_record_set_by_null_tool(session),
        build_filter_record_set_by_related_tool(session),
        build_follow_relation_tool(session),
        build_intersect_record_sets_tool(session),
        build_sort_record_set_tool(session),
        build_list_record_refs_tool(session),
        build_list_records_tool(session),
        build_count_records_tool(session),
        build_aggregate_records_tool(session),
        build_get_record_tool(session),
    ]


__all__ = [
    "build_aggregate_records_tool",
    "build_atomic_tools",
    "build_count_records_tool",
    "build_create_record_set_tool",
    "build_filter_record_set_by_pattern_tool",
    "build_filter_record_set_by_related_tool",
    "build_filter_record_set_by_null_tool",
    "build_filter_record_set_by_values_tool",
    "build_filter_record_set_tool",
    "build_follow_relation_tool",
    "build_get_record_tool",
    "build_intersect_record_sets_tool",
    "build_list_record_refs_tool",
    "build_list_records_tool",
    "build_sort_record_set_tool",
]
