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

from typing import TYPE_CHECKING

from rl_task_foundry.tooling.common.edges import available_edges
from rl_task_foundry.tooling.common.payload import (
    JsonObject,
)
from rl_task_foundry.tooling.common.payload import (
    optional_int as _optional_int,
)
from rl_task_foundry.tooling.common.payload import (
    optional_str as _optional_str,
)
from rl_task_foundry.tooling.common.payload import (
    require_str as _require_str,
)
from rl_task_foundry.tooling.common.schema import SchemaSnapshot
from rl_task_foundry.tooling.common.sql import coerce_scalar
from rl_task_foundry.tooling.common.tool_runtime import wrap_tool_handler as _with_error_handling
from rl_task_foundry.tooling.composer._session import ComposerSession
from rl_task_foundry.tooling.composer._sql import FILTER_OPS
from rl_task_foundry.tooling.composer.neighborhood import neighborhood
from rl_task_foundry.tooling.composer.profile import profile
from rl_task_foundry.tooling.composer.query import query
from rl_task_foundry.tooling.composer.sample import sample
from rl_task_foundry.tooling.composer.schema_map import schema_map

if TYPE_CHECKING:
    from agents import FunctionTool


_AGGREGATE_FNS = ("avg", "count", "max", "min", "sum")


def _all_table_names(snapshot: SchemaSnapshot) -> list[str]:
    return sorted(snapshot.table_names())


def _table_column_names(snapshot: SchemaSnapshot, table_name: str) -> list[str]:
    table = snapshot.table(table_name)
    return sorted(column.name for column in table.exposed_columns)


def _all_edge_labels(snapshot: SchemaSnapshot) -> list[str]:
    labels: set[str] = set()
    for table in snapshot.tables:
        for edge in available_edges(snapshot, table.handle):
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
        if op not in FILTER_OPS:
            raise ValueError(
                f"predicate[{index}].op must be one of {sorted(FILTER_OPS)}"
            )
        column_spec = table_spec.exposed_column(column)
        if op in {"is_null", "is_not_null"}:
            if "value" in entry and entry.get("value") is not None:
                raise TypeError(
                    f"predicate[{index}].value must be omitted or null "
                    "for is_null/is_not_null"
                )
            value = None
        else:
            if "value" not in entry or entry.get("value") is None:
                raise TypeError(
                    f"predicate[{index}].value is required and cannot "
                    "be null for this operator"
                )
            value = coerce_scalar(entry.get("value"), column_spec.data_type)
        out.append({"column": column, "op": op, "value": value})
    return out


_VALUE_SCALAR_ANY_OF = [
    {"type": "string"},
    {"type": "number"},
    {"type": "integer"},
    {"type": "boolean"},
]

_VALUE_ANY_OF = [
    *_VALUE_SCALAR_ANY_OF,
    {"type": "null"},
    {
        "type": "array",
        "items": {"anyOf": _VALUE_SCALAR_ANY_OF},
        "minItems": 1,
    },
]

_ROW_ID_ANY_OF = [
    {"type": "string"},
    {"type": "integer"},
    {
        "type": "array",
        "description": (
            "For composite primary keys — one entry per PK column, "
            "in the order declared by the table."
        ),
        "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        "minItems": 1,
    },
]


def _predicate_schema(columns: list[str]) -> JsonObject:
    return {
        "type": "array",
        "description": (
            "Optional filters on this table. Each column is scoped to the "
            "selected table; use observed values from sample/profile/query."
        ),
        "items": {
            "type": "object",
            "required": ["column", "op"],
            "additionalProperties": False,
            "properties": {
                "column": {
                    "type": "string",
                    "enum": columns,
                    "description": "Filter column on the selected table.",
                },
                "op": {
                    "type": "string",
                    "enum": _filter_op_enum(),
                    "description": "Filter operator.",
                },
                "value": {
                    "anyOf": _VALUE_ANY_OF,
                    "description": (
                        "Filter value copied from observed data; omit/null "
                        "only for is_null or is_not_null."
                    ),
                },
            },
        },
    }


def _sample_table_variant(table: str, columns: list[str]) -> JsonObject:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["table"],
        "properties": {
            "table": {
                "type": "string",
                "enum": [table],
                "description": "Table handle to sample.",
            },
            "n": {
                "type": "integer",
                "minimum": 1,
                "default": 5,
                "description": "Maximum number of real rows to return.",
            },
            "seed": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "description": (
                    "Integer seed for deterministic sampling. Null sorts by "
                    "primary key ascending."
                ),
            },
            "predicate": _predicate_schema(columns),
        },
    }


def _profile_table_variant(table: str, columns: list[str]) -> JsonObject:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["table"],
        "properties": {
            "table": {
                "type": "string",
                "enum": [table],
                "description": "Table handle to profile.",
            },
            "column": {
                "anyOf": [
                    {"type": "string", "enum": columns},
                    {"type": "null"},
                ],
                "description": (
                    "Optional column on this table. Null returns per-column "
                    "distinct/null counts for the table."
                ),
            },
            "predicate": _predicate_schema(columns),
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "default": 5,
                "description": "Number of frequent values to return for one column.",
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
                    "Optional table handle to center a depth-limited schema "
                    "map. Null returns the whole schema."
                ),
            },
            "depth": {
                "type": "integer",
                "minimum": 0,
                "maximum": 4,
                "default": 2,
                "description": (
                    "Relationship depth from root_table. Larger values show "
                    "more paths but are only a map, not label evidence."
                ),
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
            "Inspect the current DB schema: table handles, columns, "
            "relationship labels, and hub/bridge hints. Use first to choose "
            "grounded paths; live rows still provide label evidence."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler, lock=session.operation_lock),
        strict_json_schema=False,
    )


def build_sample_tool(session: ComposerSession) -> "FunctionTool":
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["target"],
        "description": "Sample one selected table with table-scoped predicates.",
        "properties": {
            "target": {
                "description": (
                    "Table-scoped sample request. Choose the branch whose table "
                    "matches the table being sampled."
                ),
                "oneOf": [
                    _sample_table_variant(
                        table,
                        _table_column_names(session.snapshot, table),
                    )
                    for table in tables
                ],
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        target = payload.get("target")
        if not isinstance(target, dict):
            raise TypeError("'target' must be an object")
        target_payload: JsonObject = {str(key): value for key, value in target.items()}
        table = _require_str(target_payload, "table")
        table_spec = session.snapshot.table(table)
        table_handle = table_spec.handle
        n = _optional_int(target_payload, "n")
        seed = _optional_int(target_payload, "seed")
        predicate = _coerce_predicate(
            session.snapshot, table_handle, target_payload.get("predicate")
        )
        rows = await sample(
            session,
            table=table_handle,
            n=n if n is not None else 5,
            seed=seed,
            predicate=predicate,
        )
        return {
            "table": table_handle,
            "rows": rows,
            "row_count": len(rows),
        }

    return FunctionTool(
        name="sample",
        description=(
            "Return real rows from one table. Use before choosing entities, "
            "visible wording, or filter values so the draft is grounded. "
            "Returns rows and row_count."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler, lock=session.operation_lock),
        strict_json_schema=False,
    )


def build_profile_tool(session: ComposerSession) -> "FunctionTool":
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["target"],
        "description": "Profile one selected table with table-scoped columns.",
        "properties": {
            "target": {
                "description": (
                    "Table-scoped profile request. Choose the branch whose table "
                    "matches the table being profiled."
                ),
                "oneOf": [
                    _profile_table_variant(
                        table,
                        _table_column_names(session.snapshot, table),
                    )
                    for table in tables
                ],
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        target = payload.get("target")
        if not isinstance(target, dict):
            raise TypeError("'target' must be an object")
        target_payload: JsonObject = {str(key): value for key, value in target.items()}
        table = _require_str(target_payload, "table")
        table_spec = session.snapshot.table(table)
        table_handle = table_spec.handle
        column = _optional_str(target_payload, "column")
        top_k = _optional_int(target_payload, "top_k")
        predicate = _coerce_predicate(
            session.snapshot, table_handle, target_payload.get("predicate")
        )
        result = await profile(
            session,
            table=table_handle,
            column=column,
            predicate=predicate,
            top_k=top_k if top_k is not None else 5,
        )
        return result

    return FunctionTool(
        name="profile",
        description=(
            "Summarize table or column distributions. Use to choose "
            "nontrivial filters, thresholds, and date ranges before querying. "
            "Returns row_count and distribution statistics."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler, lock=session.operation_lock),
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
            "table": {
                "type": "string",
                "enum": tables,
                "description": "Table handle for the observed row.",
            },
            "row_id": {
                "description": (
                    "Primary-key value of the observed row to inspect. Never "
                    "null; copy a value returned by sample/query or another "
                    "data tool."
                ),
                "anyOf": _ROW_ID_ANY_OF,
            },
            "depth": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1,
                "default": 1,
                "description": "Immediate relationships only.",
            },
            "max_per_edge": {
                "type": "integer",
                "minimum": 1,
                "default": 5,
                "description": (
                    "Maximum connected row ids to return per relationship edge."
                ),
            },
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        table = _require_str(payload, "table")
        raw_row_id = payload.get("row_id")
        if raw_row_id is None:
            raise TypeError(
                "'row_id' is required and cannot be null; use a primary-key "
                "value returned by sample/query or another data tool"
            )
        if isinstance(raw_row_id, bool | float | dict):
            raise TypeError(
                "'row_id' must be a primary-key string/integer scalar or array"
            )
        table_spec = session.snapshot.table(table)
        table_handle = table_spec.handle
        if len(table_spec.primary_key) == 1:
            if isinstance(raw_row_id, list):
                raise TypeError(
                    "'row_id' must be a scalar for a single-column primary key"
                )
            pk_column_spec = table_spec.column(table_spec.primary_key[0])
            row_id = coerce_scalar(raw_row_id, pk_column_spec.data_type)
        else:
            if not isinstance(raw_row_id, list):
                raise TypeError(
                    "'row_id' must be an array for a composite primary key"
                )
            if len(raw_row_id) != len(table_spec.primary_key):
                raise ValueError(
                    f"'row_id' must have {len(table_spec.primary_key)} values"
                )
            row_id = [
                coerce_scalar(value, table_spec.column(pk_column).data_type)
                for value, pk_column in zip(
                    raw_row_id,
                    table_spec.primary_key,
                    strict=True,
                )
            ]
        depth = _optional_int(payload, "depth")
        max_per_edge = _optional_int(payload, "max_per_edge")
        result = await neighborhood(
            session,
            table=table_handle,
            row_id=row_id,
            depth=depth if depth is not None else 1,
            max_per_edge=max_per_edge if max_per_edge is not None else 5,
        )
        return result

    return FunctionTool(
        name="neighborhood",
        description=(
            "Inspect one observed row and its immediate relationship counts "
            "and sample ids. Use after choosing an entity to find reachable "
            "task paths."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler, lock=session.operation_lock),
        strict_json_schema=False,
    )


def build_query_tool(session: ComposerSession) -> "FunctionTool":
    from agents import FunctionTool

    tables = _all_table_names(session.snapshot)
    edges = _all_edge_labels(session.snapshot)
    ref_schema: JsonObject = {
        "type": "object",
        "description": "Alias-qualified column reference.",
        "required": ["as", "column"],
        "additionalProperties": False,
        "properties": {
            "as": {
                "type": "string",
                "description": "Alias declared in spec.from or spec.join.",
            },
            "column": {
                "type": "string",
                "description": "Column on that alias's table.",
            },
        },
    }
    spec_schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["from"],
        "properties": {
            "from": {
                "type": "object",
                "description": "Root table and alias for the query path.",
                "required": ["table", "as"],
                "additionalProperties": False,
                "properties": {
                    "table": {
                        "type": "string",
                        "enum": tables,
                        "description": (
                            "Root table handle and selected source surface. "
                            "If ordinary user wording points to another "
                            "reachable source, query that source or make this "
                            "source/lifecycle role explicit in user_request/topic/"
                            "answer_contract. Do not relabel one lifecycle "
                            "surface as another, such as order/request/event/"
                            "fulfillment/log, just because fields overlap."
                        ),
                    },
                    "as": {
                        "type": "string",
                        "description": "Short query-local alias for the root table.",
                    },
                },
            },
            "join": {
                "type": "array",
                "description": (
                    "FK relationship steps from a previously declared alias. "
                    "Each step declares the source alias, relationship label, "
                    "and a new alias for the destination table. When one "
                    "answer item combines facts from the same event/record, "
                    "continue joining from that event/record alias; avoid "
                    "independent sibling joins that merely share the root."
                ),
                "items": {
                    "type": "object",
                    "required": ["from", "via_edge", "as"],
                    "additionalProperties": False,
                    "properties": {
                        "from": {
                            "type": "string",
                            "description": (
                                "Alias declared in spec.from or an earlier "
                                "join. The relationship must originate from "
                                "this alias's table."
                            ),
                        },
                        "via_edge": {
                            "type": "string",
                            "enum": edges,
                            "description": "Relationship label from schema_map.",
                        },
                        "as": {
                            "type": "string",
                            "description": (
                                "Query-local alias for the destination table."
                            ),
                        },
                    },
                },
            },
            "where": {
                "type": "array",
                "description": (
                    "Filters define row membership over from/join aliases. "
                    "A filter is valid when it implements hidden entity scope "
                    "or a customer-visible constraint stated in user_request "
                    "and submit_draft.answer_contract. Do not use filters as "
                    "hidden helper row-set controls; keep scope aligned with "
                    "the Source Surface Policy. If wording implies status, "
                    "type, or category membership, implement it here; "
                    "otherwise request records plus that status/type/category "
                    "field."
                ),
                "items": {
                    "type": "object",
                    "required": ["ref", "op"],
                    "additionalProperties": False,
                    "properties": {
                        "ref": ref_schema,
                        "op": {
                            "type": "string",
                            "enum": _filter_op_enum(),
                            "description": "Filter operator.",
                        },
                        "value": {
                            "anyOf": _VALUE_ANY_OF,
                            "description": (
                                "Filter value copied from observed data; "
                                "omit/null only for null checks. Blocked or "
                                "internal handle values are valid only when "
                                "they are the submitted hidden entity scope."
                            ),
                        },
                    },
                },
            },
            "select": {
                "type": "array",
                "description": (
                    "Selected row fields. Every selected field becomes a "
                    "canonical label field, so select only values the "
                    "user_request asks to receive. Use where/order_by for "
                    "helper context without selecting those helper fields. "
                    "One requested output slot should become one selected "
                    "field; do not split a single customer phrase across "
                    "multiple selected fields. If a timestamp already carries "
                    "date and time, select that timestamp alone unless the "
                    "request separately asks for date and time. Split value/"
                    "unit only when both have distinct natural request "
                    "phrases. "
                    "Do not select generic measurement/value fields unless "
                    "the request can name the measured amount or source "
                    "value role in ordinary language; otherwise omit them. "
                    "When selecting multiple date/time-like values, each must "
                    "have a naturally requestable distinct source role; "
                    "generic time/value aliases or phrases are not enough. "
                    "Process/status/type/category outputs are not passive "
                    "width; include them only when user_request names the "
                    "exact source/lifecycle surface, and choose the ordinary "
                    "source if that wording would sound awkward. "
                    "Do not substitute a different user-visible field for a "
                    "blocked/internal source; if the request would naturally "
                    "ask for the blocked source role, choose another label "
                    "or source. "
                    "For aggregate queries, do not use select; use group_by "
                    "for copied group keys and aggregate for metric fields. "
                    "Preserve output source meanings under the Source Surface "
                    "Policy and Difficulty-Up Policy. Prefer user-visible "
                    "non-handle values; expose handle-like values only when "
                    "evidence marks them user-visible and the request asks for "
                    "that reference. Do not select source sequence/reference/"
                    "order numbers to repair list determinism or binding "
                    "unless that sequence is the natural domain answer. "
                    "Do not expose row values from a table "
                    "without a primary key; choose a primary-key-backed path."
                    " After query returns, treat "
                    "label_source_diagnostics.submit_blocked as blocking; "
                    "rerun with visible outputs or choose another label."
                ),
                "items": {
                    "type": "object",
                    "required": ["ref", "as"],
                    "additionalProperties": False,
                    "properties": {
                        "ref": ref_schema,
                        "as": {
                            "type": "string",
                            "description": (
                                "Stable output field name that preserves the "
                                "selected source column meaning. Output aliases "
                                "do not disambiguate competing reachable "
                                "sources; user_request and answer_contract "
                                "phrases must name the selected source role. "
                                "Aliases cannot make a visible substitute "
                                "mean a different blocked source role. "
                                "For temporal values, preserve the chosen "
                                "role such as event, value, scheduled, or "
                                "stored/entered time; do not hide competing "
                                "time surfaces behind generic names. "
                                "Do not make note/comment/description text "
                                "look like a result/status/value field unless "
                                "the request names that text surface. Do not "
                                "make source status/type/category text look "
                                "like broad object state/type/category or "
                                "current/derived state wording."
                            ),
                        },
                    },
                },
            },
            "order_by": {
                "type": "array",
                "maxItems": 2,
                "description": (
                    "Deterministic ordering. Direction must match user_request "
                    "wording exactly. Follow the List Determinism Policy for "
                    "ties and limited lists; order_by must not introduce "
                    "unrequested or hidden row-order controls. Returned rows "
                    "are the canonical label order; do not plan to reverse "
                    "them through request or answer_contract wording. Use no more "
                    "than two order keys total; if more are needed, choose "
                    "another label or return tied rows. A selected output "
                    "field is not an order request unless the request also "
                    "states its direction, recency, rank, or tie-break role. "
                    "If two order keys still leave duplicate_order_key or "
                    "limit_boundary_tie diagnostics, abandon that ordered "
                    "limited list or return tied rows. "
                    "When a full timestamp is available, prefer it as the "
                    "time order key instead of spending separate date and "
                    "time order keys; split date/time only for display. "
                    "If several temporal fields are reachable or selected, "
                    "the order wording must identify the chosen time role; "
                    "generic latest/recent time wording is not enough. "
                    "When both start and end times are selected or reachable, "
                    "plain recent/latest is ambiguous: either request start-time "
                    "or end-time recency explicitly, or order by the boundary "
                    "the request already names. "
                    "If diagnostics list handle_order_by_columns, that key is "
                    "a handle; do not keep it as a silent tie-break. Each "
                    "visible sequence/reference/order key is still technical "
                    "unless the request naturally asks for that source "
                    "sequence; do not add it during feedback repair just to "
                    "make the list unique. Each "
                    "order item must contain exactly one of ref or output, "
                    "never both."
                ),
                "items": {
                    "type": "object",
                    "required": ["direction"],
                    "additionalProperties": False,
                    "properties": {
                        "ref": ref_schema,
                        "output": {
                            "type": "string",
                            "description": (
                                "Name declared by select.as, group_by.as, "
                                "or aggregate.as. Mutually exclusive with ref."
                            ),
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["asc", "desc"],
                            "description": (
                                "Sort direction. Match the direction or ranking "
                                "stated in user_request before using the rows as "
                                "the canonical label; do not use an opposite "
                                "display-order phrase in submit_draft."
                            ),
                        },
                    },
                },
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Fixed row cap. If the task asks for N items, use the "
                    "same N in user_request and "
                    "submit_draft.answer_contract.limit_phrase. Follow the "
                    "Task Shapes policy before the final label query: choose "
                    "a 3-5 row boundary and include this limit unless prior "
                    "evidence proves all matching rows are already 3-5. "
                    "Do not submit an unbounded row-list query when the full "
                    "matching set may exceed 5 rows. "
                    "Follow the "
                    "List Determinism Policy when a limit shapes membership; "
                    "a final list query with limit must have the same fixed "
                    "size requested and bound before submit_draft. Do not "
                    "request one order to select limited membership and a "
                    "different order to display the returned rows. Do not "
                    "describe a limited query as all/every matching records."
                ),
            },
            "group_by": {
                "type": "array",
                "description": (
                    "Group keys for aggregate lists. Prefer user-visible "
                    "non-handle values for copied label keys; expose "
                    "handle-like values only when evidence marks them "
                    "user-visible and the request asks for that reference. "
                    "When aggregate is present, group_by replaces select."
                ),
                "items": {
                    "type": "object",
                    "required": ["ref", "as"],
                    "additionalProperties": False,
                    "properties": {
                        "ref": ref_schema,
                        "as": {
                            "type": "string",
                            "description": "Stable output field name for this group key.",
                        },
                    },
                },
            },
            "aggregate": {
                "type": "array",
                "description": (
                    "Aggregate outputs. For scalar submit_draft labels, use "
                    "an aggregate query without group_by so it returns one row. "
                    "Aggregate source rows must come from primary-key-backed "
                    "tables; do not aggregate over a table without a primary "
                    "key because the answering tool surface cannot reproduce it. "
                    "Do not combine aggregate with select; grouped aggregate "
                    "lists use group_by plus aggregate only."
                ),
                "items": {
                    "type": "object",
                    "required": ["fn", "as"],
                    "additionalProperties": False,
                    "properties": {
                        "fn": {
                            "type": "string",
                            "enum": list(_AGGREGATE_FNS),
                            "description": "Aggregate function.",
                        },
                        "ref": ref_schema,
                        "as": {
                            "type": "string",
                            "description": "Stable output field name for the aggregate.",
                        },
                    },
                },
            },
        },
    }
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["spec"],
        "properties": {
            "spec": {
                **spec_schema,
                "description": (
                    "Structured read-only query spec. Use immediately before "
                    "submit_draft; returned rows are canonical label evidence. "
                    "After a successful final label query, submit_draft instead "
                    "of calling schema/profile/sample/neighborhood tools; only "
                    "rerun query when returned diagnostics block submission. "
                    "Before submit_draft, ensure user_request/topic name the "
                    "selected source role when multiple reachable sources could "
                    "answer the broad noun; label/output field names do not "
                    "disambiguate source surface. Status/type/category/"
                    "classification/result outputs must name the selected "
                    "source role when related surfaces can differ. If naming "
                    "the chosen source "
                    "role would sound awkward, choose another label/source "
                    "instead of submitting broad wording. "
                    "Copy returned row values exactly into the label, including "
                    "spacing; do not trim, normalize, or rewrite strings. "
                    "Inspect returned diagnostics before submit_draft: ordering "
                    "diagnostics flag unstable list order, and projection "
                    "diagnostics flag answer rows that are indistinguishable "
                    "after the selected output fields. If ordering diagnostics "
                    "include duplicate_order_key_in_returned_rows or "
                    "unrepresented_order_by_tie_breakers for a list, do not "
                    "submit that result as final label evidence; revise the "
                    "request/order/output fields or choose another label. "
                    "selected_visible_tie_breaker_candidates are only already "
                    "selected visible outputs that split the observed tie; use "
                    "one only when it is a natural requested tie-break. "
                    "After adding a tie-break order key, rewrite user_request "
                    "and answer_contract to name that order role exactly; "
                    "selected output wording alone is not order wording, and "
                    "fixed limits still need limit phrases. "
                    "If projection diagnostics report duplicate projected "
                    "answer rows, the result is blocking even when the "
                    "underlying source rows are distinct; do not add source "
                    "sequence/reference/order fields solely to make rows unique; "
                    "use one natural visible domain field or aggregate, and if "
                    "rows still duplicate choose another label. "
                    "If handle_order_by_columns appears, use the handle only "
                    "when the request explicitly asks for that record sequence "
                    "or reference role; otherwise pick a natural visible key."
                ),
            }
        },
    }

    async def handler(payload: JsonObject) -> JsonObject:
        raw_spec = payload.get("spec")
        if not isinstance(raw_spec, dict):
            raise TypeError("'spec' must be an object")
        spec_dict: JsonObject = {
            str(key): value for key, value in raw_spec.items()
        }
        result = await query(session, spec=spec_dict)
        return result

    return FunctionTool(
        name="query",
        description=(
            "Run a structured read-only query over aliases and FK joins. Use "
            "to produce the exact rows that will be copied into the label. "
            "The query path is the selected source surface; if broad request "
            "wording could fit another reachable source, make this source role "
            "explicit or choose the ordinary matching source. "
            "Returns columns, rows, row_count, and diagnostics for list order "
            "or duplicate projected answer rows when present; blocking "
            "diagnostics must be fixed before submit_draft, and duplicate "
            "projected answer rows are blocking even when source rows differ."
        ),
        params_json_schema=schema,
        on_invoke_tool=_with_error_handling(handler, lock=session.operation_lock),
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
