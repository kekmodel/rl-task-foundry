"""Build agents-SDK FunctionTool instances for the composer toolset.

Each builder closes over a `ComposerSession` so the tools share one
synthesis conversation's snapshot and connection. JSON-schema enums for
table / column / edge / direction are baked from the snapshot; the
on_invoke_tool wrapper handles JSON parsing, temporal coercion at the
boundary (ISO strings → datetime/date/time for the matching column
types), and structured `{error, error_type}` payloads instead of
raises.

The `agents` package is imported lazily so this module is importable
without the SDK present.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import asyncpg

from rl_task_foundry.tooling.common.edges import available_edges
from rl_task_foundry.tooling.common.payload import (
    JsonObject,
    optional_int as _optional_int,
    optional_str as _optional_str,
    require_int as _require_int,
    require_str as _require_str,
)
from rl_task_foundry.tooling.common.schema import SchemaSnapshot
from rl_task_foundry.tooling.common.sql import coerce_scalar
from rl_task_foundry.tooling.composer._session import ComposerSession
from rl_task_foundry.tooling.composer._sql import FILTER_OPS
from rl_task_foundry.tooling.composer.neighborhood import neighborhood
from rl_task_foundry.tooling.composer.profile import profile
from rl_task_foundry.tooling.composer.query import query
from rl_task_foundry.tooling.composer.sample import sample
from rl_task_foundry.tooling.composer.schema_map import schema_map

if TYPE_CHECKING:
    from agents import FunctionTool


Handler = Callable[[JsonObject], Awaitable[JsonObject]]
Invoker = Callable[[object, str], Awaitable[str]]


_AGGREGATE_FNS = ("avg", "count", "max", "min", "sum")


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
    return sorted(FILTER_OPS)


def _coerce_predicate(
    snapshot: SchemaSnapshot,
    table: str,
    predicate: object,
) -> list[dict[str, object]] | None:
    if predicate is None:
        return None
    if not isinstance(predicate, list):
        raise TypeError(
            "predicate must be a list of {column, op, value}"
        )
    table_spec = snapshot.table(table)
    out: list[dict[str, object]] = []
    for index, entry in enumerate(predicate):
        if not isinstance(entry, dict):
            raise TypeError(
                f"predicate[{index}] must be a mapping; got "
                f"{type(entry).__name__}"
            )
        column = entry.get("column")
        op = entry.get("op")
        if not isinstance(column, str):
            raise TypeError(
                f"predicate[{index}].column must be a string"
            )
        if not isinstance(op, str):
            raise TypeError(
                f"predicate[{index}].op must be a string"
            )
        column_spec = table_spec.column(column)
        value = coerce_scalar(entry.get("value"), column_spec.data_type)
        out.append({"column": column, "op": op, "value": value})
    return out


def _coerce_query_spec(
    snapshot: SchemaSnapshot,
    spec: JsonObject,
) -> JsonObject:
    """Coerce the query spec's filter values to each column's data type.

    LLM-supplied filter values arrive as JSON scalars, so integer PKs can be
    strings and temporals arrive as ISO text. query.py re-parses the spec so
    we only need to normalize ``value`` entries against ``table_spec.column(…)``.
    """
    from_table = spec.get("from")
    if not isinstance(from_table, str):
        return spec
    if from_table not in snapshot.table_names():
        return spec
    table_spec = snapshot.table(from_table)
    raw_filter = spec.get("filter")
    if raw_filter is None or not isinstance(raw_filter, list):
        return spec
    rebuilt: list[object] = []
    for entry in raw_filter:
        if not isinstance(entry, dict):
            rebuilt.append(entry)
            continue
        column = entry.get("column")
        if not isinstance(column, str):
            rebuilt.append(entry)
            continue
        try:
            column_spec = table_spec.column(column)
        except KeyError:
            rebuilt.append(entry)
            continue
        rewritten = dict(entry)
        rewritten["value"] = coerce_scalar(
            entry.get("value"), column_spec.data_type
        )
        rebuilt.append(rewritten)
    return {**spec, "filter": rebuilt}


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False)


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
            NotImplementedError,
            asyncpg.exceptions.PostgresError,
        ) as exc:
            return _json_dumps(
                {"error": str(exc), "error_type": type(exc).__name__}
            )
        return _json_dumps(result)

    return invoke


_VALUE_ANY_OF = [
    {"type": "string"},
    {"type": "number"},
    {"type": "integer"},
    {"type": "boolean"},
    {"type": "null"},
    {"type": "array", "items": {}},
]


def _predicate_schema(columns: list[str]) -> JsonObject:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["column", "op"],
            "additionalProperties": False,
            "properties": {
                "column": {"type": "string", "enum": columns},
                "op": {"type": "string", "enum": _filter_op_enum()},
                "value": {"anyOf": _VALUE_ANY_OF},
            },
        },
    }


# ---------- individual tool builders ----------


def build_schema_map_tool(session: ComposerSession) -> "FunctionTool":
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "root_table": {
                "anyOf": [
                    {"type": "string", "enum": tables},
                    {"type": "null"},
                ],
                "description": (
                    "Anchor for a depth-limited BFS slice. Omit / null "
                    "returns the whole schema."
                ),
            },
            "depth": {
                "type": "integer",
                "minimum": 0,
                "maximum": 4,
                "default": 2,
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        root_table = _optional_str(payload, "root_table")
        depth = _optional_int(payload, "depth")
        result = schema_map(
            session.snapshot,
            root_table=root_table,
            depth=depth if depth is not None else 2,
        )
        return result

    return FunctionTool(
        name="schema_map",
        description=(
            "Return a JSON-ready slice of the schema graph for the "
            "composer to orient inside the database."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_sample_tool(session: ComposerSession) -> "FunctionTool":
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    columns = _all_column_names(session.snapshot)
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["table"],
        "properties": {
            "table": {"type": "string", "enum": tables},
            "n": {"type": "integer", "minimum": 1, "default": 5},
            "seed": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "description": (
                    "Integer seed for deterministic md5-based ordering; "
                    "null sorts by primary key ASC."
                ),
            },
            "predicate": _predicate_schema(columns),
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        table = _require_str(payload, "table")
        n = _optional_int(payload, "n")
        seed = _optional_int(payload, "seed")
        predicate = _coerce_predicate(
            session.snapshot, table, payload.get("predicate")
        )
        rows = await sample(
            session,
            table=table,
            n=n if n is not None else 5,
            seed=seed,
            predicate=predicate,
        )
        return {"table": table, "rows": rows, "row_count": len(rows)}

    return FunctionTool(
        name="sample",
        description=(
            "Return up to n representative rows from a table with "
            "optional filter + deterministic seed."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_profile_tool(session: ComposerSession) -> "FunctionTool":
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    columns = _all_column_names(session.snapshot)
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["table"],
        "properties": {
            "table": {"type": "string", "enum": tables},
            "column": {
                "anyOf": [
                    {"type": "string", "enum": columns},
                    {"type": "null"},
                ],
                "description": (
                    "Single column for a detail profile; null returns "
                    "per-column distinct/null counts for the whole table."
                ),
            },
            "predicate": _predicate_schema(columns),
            "top_k": {"type": "integer", "minimum": 1, "default": 5},
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        table = _require_str(payload, "table")
        column = _optional_str(payload, "column")
        top_k = _optional_int(payload, "top_k")
        predicate = _coerce_predicate(
            session.snapshot, table, payload.get("predicate")
        )
        result = await profile(
            session,
            table=table,
            column=column,
            predicate=predicate,
            top_k=top_k if top_k is not None else 5,
        )
        return result

    return FunctionTool(
        name="profile",
        description=(
            "Return a distribution snapshot: row_count plus per-column "
            "distinct/null counts, or (with column set) min/max + top-k "
            "frequency for one column."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_neighborhood_tool(session: ComposerSession) -> "FunctionTool":
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["table", "row_id"],
        "properties": {
            "table": {"type": "string", "enum": tables},
            "row_id": {
                "description": "Primary-key value of the anchor row.",
                "anyOf": _VALUE_ANY_OF,
            },
            "depth": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1,
                "default": 1,
                "description": "depth=1 only for now.",
            },
            "max_per_edge": {
                "type": "integer",
                "minimum": 1,
                "default": 5,
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        table = _require_str(payload, "table")
        table_spec = session.snapshot.table(table)
        if len(table_spec.primary_key) == 1:
            pk_column_spec = table_spec.column(table_spec.primary_key[0])
            row_id = coerce_scalar(payload.get("row_id"), pk_column_spec.data_type)
        else:
            row_id = payload.get("row_id")
        depth = _optional_int(payload, "depth")
        max_per_edge = _optional_int(payload, "max_per_edge")
        result = await neighborhood(
            session,
            table=table,
            row_id=row_id,
            depth=depth if depth is not None else 1,
            max_per_edge=max_per_edge if max_per_edge is not None else 5,
        )
        return result

    return FunctionTool(
        name="neighborhood",
        description=(
            "Return the anchor row's attributes plus, for each outbound "
            "FK edge, a bounded sample of connected row IDs and the "
            "total edge count."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


def build_query_tool(session: ComposerSession) -> "FunctionTool":
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    columns = _all_column_names(session.snapshot)
    edges = _all_edge_labels(session.snapshot)
    spec_schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["from"],
        "properties": {
            "from": {"type": "string", "enum": tables},
            "filter": _predicate_schema(columns),
            "join": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["via_edge"],
                    "additionalProperties": False,
                    "properties": {
                        "via_edge": {"type": "string", "enum": edges},
                    },
                },
            },
            "select": {
                "type": "array",
                "items": {"type": "string", "enum": columns},
            },
            "sort": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["column"],
                    "additionalProperties": False,
                    "properties": {
                        "column": {"type": "string"},
                        "direction": {
                            "type": "string",
                            "enum": ["asc", "desc"],
                        },
                    },
                },
            },
            "limit": {"type": "integer", "minimum": 1},
            "group_by": {
                "type": "array",
                "items": {"type": "string", "enum": columns},
            },
            "aggregate": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["fn", "alias"],
                    "additionalProperties": False,
                    "properties": {
                        "fn": {"type": "string", "enum": list(_AGGREGATE_FNS)},
                        "column": {"type": "string", "enum": columns},
                        "alias": {"type": "string"},
                    },
                },
            },
        },
    }
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["spec"],
        "properties": {"spec": spec_schema},
    }

    async def handler(payload: JsonObject) -> JsonObject:
        raw_spec = payload.get("spec")
        if not isinstance(raw_spec, dict):
            raise TypeError("'spec' must be an object")
        spec_dict: JsonObject = {
            str(key): value for key, value in raw_spec.items()
        }
        coerced = _coerce_query_spec(session.snapshot, spec_dict)
        result = await query(session, spec=coerced)
        return result

    return FunctionTool(
        name="query",
        description=(
            "Execute the composer query DSL: filter + FK join chain + "
            "select or group_by+aggregate, plus sort and limit. Returns "
            "{columns, rows, row_count}. Single call authors canonical "
            "answers without scripting an atomic chain."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler),
        strict_json_schema=False,
    )


# ---------- aggregate entrypoint ----------


def build_composer_tools(session: ComposerSession) -> list["FunctionTool"]:
    """Build all five composer FunctionTool instances for a session."""
    return [
        build_schema_map_tool(session),
        build_profile_tool(session),
        build_sample_tool(session),
        build_neighborhood_tool(session),
        build_query_tool(session),
    ]


__all__ = [
    "build_composer_tools",
    "build_neighborhood_tool",
    "build_profile_tool",
    "build_query_tool",
    "build_sample_tool",
    "build_schema_map_tool",
]
