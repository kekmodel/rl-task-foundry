"""Deterministic schema-derived atomic tools for one database."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import Field

from rl_task_foundry.config.models import AtomicToolConfig, StrictModel
from rl_task_foundry.schema.graph import ColumnProfile, SchemaGraph, TableProfile

_TEXT_TYPES = {
    "bpchar",
    "char",
    "character",
    "character varying",
    "citext",
    "name",
    "text",
    "uuid",
    "varchar",
}
_INTEGER_TYPES = {
    "bigint",
    "int2",
    "int4",
    "int8",
    "integer",
    "smallint",
}
_FLOAT_TYPES = {
    "decimal",
    "double precision",
    "float4",
    "float8",
    "numeric",
    "real",
}
_BOOLEAN_TYPES = {
    "bool",
    "boolean",
}
_DATE_TYPES = {"date"}
_DATETIME_TYPES = {
    "timestamp",
    "timestamp with time zone",
    "timestamp without time zone",
    "timestamptz",
}
_SCALAR_TYPES = (
    _TEXT_TYPES | _INTEGER_TYPES | _FLOAT_TYPES | _BOOLEAN_TYPES | _DATE_TYPES | _DATETIME_TYPES
)
_FIND_COMPARABLE_TYPES = _INTEGER_TYPES | _FLOAT_TYPES | _DATE_TYPES | _DATETIME_TYPES
_ALL_FIND_OPS = ("any", "eq", "in", "lt", "gt", "lte", "gte", "like")
_CALC_FILTER_OPS = ("eq", "in", "lt", "gt", "lte", "gte", "like")

logger = logging.getLogger(__name__)


class AtomicToolFamily(StrEnum):
    GET = "get"
    FIND = "find"
    CALC = "calc"
    RANK = "rank"


class AtomicToolResultMode(StrEnum):
    OBJECT_OR_NULL = "object_or_null"
    ROW_LIST = "row_list"
    SCALAR = "scalar"


class AtomicToolDefinition(StrictModel):
    name: str
    family: AtomicToolFamily
    description: str
    params_schema: dict[str, Any]
    returns_schema: dict[str, Any]
    sql: str = Field(description="Display/audit only SQL template. Not executed directly.")
    result_mode: AtomicToolResultMode
    semantic_key: str
    runtime_metadata: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def actor_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "params_schema": self.params_schema,
            "returns_schema": self.returns_schema,
        }


class AtomicToolBundle(StrictModel):
    db_id: str
    tools: list[AtomicToolDefinition] = Field(default_factory=list)
    source: str

    def actor_tool_definitions(self) -> list[dict[str, Any]]:
        return [tool.actor_payload() for tool in self.tools]

    def actor_tool_definitions_json(self) -> str:
        return json.dumps(
            self.actor_tool_definitions(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


@dataclass(slots=True)
class AtomicToolGenerator:
    config: AtomicToolConfig

    def generate_bundle(self, graph: SchemaGraph, *, db_id: str) -> AtomicToolBundle:
        table_slugs = _table_slug_map(graph)
        compiled: list[AtomicToolDefinition] = []
        for table in sorted(graph.tables, key=lambda item: (item.schema_name, item.table_name)):
            compiled.extend(self._build_get_tools(table, table_slugs))
            compiled.extend(self._build_find_tools(graph, table, table_slugs))
            compiled.extend(self._build_calc_tools(graph, table, table_slugs))
            compiled.extend(self._build_rank_tools(graph, table, table_slugs))

        compiled = _compress_tools(
            sorted(compiled, key=lambda tool: (_family_rank(tool.family), tool.name)),
            graph=graph,
            max_tools=self.config.max_tools,
        )
        return AtomicToolBundle(
            db_id=db_id,
            tools=compiled,
            source=_render_atomic_tool_source(
                db_id=db_id,
                tools=compiled,
                max_batch_values=self.config.max_batch_values,
                max_bounded_result_limit=self.config.bounded_result_limit,
                float_precision=self.config.float_precision,
            ),
        )

    def _build_get_tools(
        self,
        table: TableProfile,
        table_slugs: dict[tuple[str, str], str],
    ) -> list[AtomicToolDefinition]:
        if not table.primary_key:
            return []
        table_slug = table_slugs[(table.schema_name, table.table_name)]
        return [
            AtomicToolDefinition(
                name=f"get_{table_slug}",
                family=AtomicToolFamily.GET,
                description=(
                    f"Retrieve one {humanize_identifier(table_slug)} by ID. "
                    "Returns all fields or nothing."
                ),
                params_schema=_get_params_schema(table),
                returns_schema=_nullable_object_schema(tuple(table.columns)),
                sql=_get_sql_template(table),
                result_mode=AtomicToolResultMode.OBJECT_OR_NULL,
                semantic_key=f"{table.qualified_name}:get",
                runtime_metadata=_table_runtime_metadata(None, table),
            )
        ]

    def _build_find_tools(
        self,
        graph: SchemaGraph,
        table: TableProfile,
        table_slugs: dict[tuple[str, str], str],
    ) -> list[AtomicToolDefinition]:
        table_slug = table_slugs[(table.schema_name, table.table_name)]
        tools: list[AtomicToolDefinition] = []
        for column in _find_columns(graph, table):
            allowed_ops = _find_ops_for_column(column)
            target_label = _foreign_key_target_label(graph, table, column, table_slugs)
            if target_label is not None:
                description = (
                    f"Find {humanize_identifier(table_slug)} for a given {target_label}. "
                    "Returns a list."
                )
            else:
                description = (
                    f"Find {humanize_identifier(table_slug)} entries where "
                    f"{humanize_identifier(column.column_name)} matches a condition. Returns a list."
                )
            tools.append(
                AtomicToolDefinition(
                    name=f"find_{table_slug}_by_{column.column_name}",
                    family=AtomicToolFamily.FIND,
                    description=description,
                    params_schema=_find_params_schema(
                        table,
                        column=column,
                        allowed_ops=allowed_ops,
                        max_batch_values=self.config.max_batch_values,
                        max_items=self.config.bounded_result_limit,
                    ),
                    returns_schema=_row_list_schema(
                        tuple(table.columns),
                        max_items=self.config.bounded_result_limit,
                    ),
                    sql=_find_sql_template(table, column),
                    result_mode=AtomicToolResultMode.ROW_LIST,
                    semantic_key=f"{table.qualified_name}:find:{column.column_name}",
                    runtime_metadata={
                        **_table_runtime_metadata(graph, table),
                        "filter_column": column.column_name,
                        "allowed_ops": list(allowed_ops),
                    },
                )
            )
        return tools

    def _build_calc_tools(
        self,
        graph: SchemaGraph,
        table: TableProfile,
        table_slugs: dict[tuple[str, str], str],
    ) -> list[AtomicToolDefinition]:
        table_slug = table_slugs[(table.schema_name, table.table_name)]
        numeric_columns = _numeric_columns(table)
        allowed_fns = ("count",) if not numeric_columns else ("count", "sum", "avg", "min", "max")
        return [
            AtomicToolDefinition(
                name=f"calc_{table_slug}",
                family=AtomicToolFamily.CALC,
                description=(
                    f"Compute a statistic over {humanize_identifier(table_slug)} entries. "
                    "Returns one number."
                ),
                params_schema=_calc_params_schema(
                    graph,
                    table,
                    allowed_fns=allowed_fns,
                    max_batch_values=self.config.max_batch_values,
                ),
                returns_schema=_calc_returns_schema(),
                sql=_calc_sql_template(table),
                result_mode=AtomicToolResultMode.SCALAR,
                semantic_key=f"{table.qualified_name}:calc",
                runtime_metadata={
                    **_table_runtime_metadata(graph, table),
                    "allowed_fns": list(allowed_fns),
                },
            )
        ]

    def _build_rank_tools(
        self,
        graph: SchemaGraph,
        table: TableProfile,
        table_slugs: dict[tuple[str, str], str],
    ) -> list[AtomicToolDefinition]:
        table_slug = table_slugs[(table.schema_name, table.table_name)]
        numeric_columns = _numeric_columns(table)
        allowed_fns = ("count",) if not numeric_columns else ("count", "sum", "avg", "min", "max")
        tools: list[AtomicToolDefinition] = []
        for column in _group_columns(graph, table):
            target_label = _foreign_key_target_label(graph, table, column, table_slugs)
            if target_label is not None:
                description = (
                    f"Rank {target_label} groups across {humanize_identifier(table_slug)}. "
                    "Returns a sorted list."
                )
            else:
                description = (
                    f"Rank {humanize_identifier(column.column_name)} groups by a statistic over "
                    f"{humanize_identifier(table_slug)}. Returns a sorted list."
                )
            tools.append(
                AtomicToolDefinition(
                    name=f"rank_{table_slug}_by_{column.column_name}",
                    family=AtomicToolFamily.RANK,
                    description=description,
                    params_schema=_rank_params_schema(
                        graph,
                        table,
                        allowed_fns=allowed_fns,
                        max_batch_values=self.config.max_batch_values,
                        max_items=self.config.bounded_result_limit,
                    ),
                    returns_schema=_rank_returns_schema(
                        group_column=column,
                        max_items=self.config.bounded_result_limit,
                    ),
                    sql=_rank_sql_template(table, column),
                    result_mode=AtomicToolResultMode.ROW_LIST,
                    semantic_key=f"{table.qualified_name}:rank:{column.column_name}",
                    runtime_metadata={
                        **_table_runtime_metadata(graph, table),
                        "group_column": column.column_name,
                        "allowed_fns": list(allowed_fns),
                    },
                )
            )
        return tools


def _family_rank(family: AtomicToolFamily) -> int:
    order = {
        AtomicToolFamily.GET: 0,
        AtomicToolFamily.FIND: 1,
        AtomicToolFamily.CALC: 2,
        AtomicToolFamily.RANK: 3,
    }
    return order[family]


def _compress_tools(
    tools: list[AtomicToolDefinition],
    *,
    graph: SchemaGraph,
    max_tools: int,
) -> list[AtomicToolDefinition]:
    if len(tools) <= max_tools:
        return tools
    return _compress_by_table_removal(tools, graph=graph, max_tools=max_tools)


def _compress_by_table_removal(
    tools: list[AtomicToolDefinition],
    *,
    graph: SchemaGraph,
    max_tools: int,
) -> list[AtomicToolDefinition]:
    if len(tools) <= max_tools:
        return tools

    edge_counts = {table.qualified_name: 0 for table in graph.tables}
    for edge in graph.edges:
        edge_counts[edge.source_qualified_name] = edge_counts.get(edge.source_qualified_name, 0) + 1
        edge_counts[edge.target_qualified_name] = edge_counts.get(edge.target_qualified_name, 0) + 1

    removal_order = sorted(
        graph.tables,
        key=lambda table: (
            edge_counts.get(table.qualified_name, 0),
            table.qualified_name,
        ),
    )

    removed_tables: set[str] = set()
    remaining = list(tools)
    for table in removal_order:
        if len(remaining) <= max_tools:
            break
        table_name = table.qualified_name
        remaining = [
            tool
            for tool in remaining
            if str(tool.runtime_metadata.get("qualified_name")) != table_name
        ]
        removed_tables.add(table_name)

    if removed_tables:
        logger.warning(
            "schema exceeds max_tools=%d, removed %d tables: %s",
            max_tools,
            len(removed_tables),
            sorted(removed_tables),
        )
    return remaining


def _table_slug_map(graph: SchemaGraph) -> dict[tuple[str, str], str]:
    singularized: dict[tuple[str, str], str] = {}
    slug_counts: dict[str, int] = {}
    for table in graph.tables:
        slug = _singularize_identifier(table.table_name)
        singularized[(table.schema_name, table.table_name)] = slug
        slug_counts[slug] = slug_counts.get(slug, 0) + 1

    resolved: dict[tuple[str, str], str] = {}
    for table in graph.tables:
        key = (table.schema_name, table.table_name)
        slug = singularized[key]
        if slug_counts[slug] > 1:
            resolved[key] = f"{_singularize_identifier(table.schema_name)}_{slug}"
        else:
            resolved[key] = slug
    return resolved


def _singularize_identifier(identifier: str) -> str:
    parts = [part for part in identifier.lower().split("_") if part]
    if not parts:
        return identifier.lower()
    parts[-1] = singularize_token(parts[-1])
    return "_".join(parts)


def _normalized_data_type(data_type: str) -> str:
    return " ".join(data_type.strip().lower().split())


def _is_scalar(column: ColumnProfile) -> bool:
    return _normalized_data_type(column.data_type) in _SCALAR_TYPES


def _is_numeric(column: ColumnProfile) -> bool:
    return _normalized_data_type(column.data_type) in (_INTEGER_TYPES | _FLOAT_TYPES)


def _is_float_like(column: ColumnProfile) -> bool:
    return _normalized_data_type(column.data_type) in _FLOAT_TYPES


def _is_numeric_or_date(column: ColumnProfile) -> bool:
    return _normalized_data_type(column.data_type) in _FIND_COMPARABLE_TYPES


def _is_useful_for_find(column: ColumnProfile, table: TableProfile) -> bool:
    if column.n_distinct is None:
        if _normalized_data_type(column.data_type) in _TEXT_TYPES:
            logger.debug(
                "Including %s.%s in find/rank selection because n_distinct is unavailable.",
                table.qualified_name,
                column.column_name,
            )
            return True
        return False
    if column.n_distinct < 0:
        return abs(column.n_distinct) < 0.9
    if table.row_estimate is None or table.row_estimate <= 0:
        return False
    return column.n_distinct < (table.row_estimate * 0.9)


def _foreign_key_column_names(graph: SchemaGraph | None, table: TableProfile) -> set[str]:
    if graph is None:
        return {
            column.column_name
            for column in table.columns
            if column.is_foreign_key
        }
    names: set[str] = set()
    for edge in graph.edges_from(table.table_name, schema_name=table.schema_name):
        names.update(edge.source_columns)
    return names


def _foreign_key_target_label(
    graph: SchemaGraph | None,
    table: TableProfile,
    column: ColumnProfile,
    table_slugs: dict[tuple[str, str], str],
) -> str | None:
    if graph is None:
        return None
    for edge in graph.edges_from(table.table_name, schema_name=table.schema_name):
        if edge.source_columns != (column.column_name,):
            continue
        target_slug = table_slugs.get((edge.target_schema, edge.target_table))
        if target_slug is not None:
            return humanize_identifier(target_slug)
        return humanize_identifier(edge.target_table)
    return None


def _find_columns(graph: SchemaGraph | None, table: TableProfile) -> list[ColumnProfile]:
    fk_column_names = _foreign_key_column_names(graph, table)
    columns: list[ColumnProfile] = []
    for column in table.columns:
        if not _is_scalar(column):
            continue
        if column.column_name in table.primary_key:
            continue
        if column.column_name in fk_column_names:
            columns.append(column)
            continue
        if _is_useful_for_find(column, table):
            columns.append(column)
            continue
        if _is_numeric_or_date(column):
            columns.append(column)
    return columns


def _numeric_columns(table: TableProfile) -> list[ColumnProfile]:
    return [column for column in table.columns if _is_numeric(column)]


def _group_columns(graph: SchemaGraph | None, table: TableProfile) -> list[ColumnProfile]:
    fk_column_names = _foreign_key_column_names(graph, table)
    columns: list[ColumnProfile] = []
    for column in table.columns:
        if not _is_scalar(column):
            continue
        if column.column_name in table.primary_key:
            continue
        if column.column_name in fk_column_names:
            columns.append(column)
            continue
        if _is_numeric_or_date(column):
            continue
        if _is_useful_for_find(column, table):
            columns.append(column)
    return columns


def _sortable_columns(table: TableProfile) -> list[ColumnProfile]:
    return [column for column in table.columns if _is_scalar(column)]


def _find_ops_for_column(column: ColumnProfile) -> tuple[str, ...]:
    normalized = _normalized_data_type(column.data_type)
    ops = ["any", "eq", "in"]
    if normalized in _FIND_COMPARABLE_TYPES:
        ops.extend(["lt", "gt", "lte", "gte"])
    if normalized in _TEXT_TYPES:
        ops.append("like")
    return tuple(ops)


def _table_runtime_metadata(graph: SchemaGraph | None, table: TableProfile) -> dict[str, Any]:
    return {
        "schema_name": table.schema_name,
        "table_name": table.table_name,
        "qualified_name": table.qualified_name,
        "all_columns": [column.column_name for column in table.columns],
        "primary_key": list(table.primary_key),
        "order_columns": list(table.primary_key) or [column.column_name for column in table.columns],
        "column_types": {
            column.column_name: _normalized_data_type(column.data_type) for column in table.columns
        },
        "array_casts": {column.column_name: _postgres_array_cast(column) for column in table.columns},
        "numeric_columns": [column.column_name for column in _numeric_columns(table)],
        "filter_columns": [column.column_name for column in _find_columns(graph, table)],
        "sortable_columns": [column.column_name for column in _sortable_columns(table)],
    }


def _get_params_schema(table: TableProfile) -> dict[str, Any]:
    if len(table.primary_key) == 1:
        pk_column = table.get_column(table.primary_key[0])
        return _object_schema(
            {
                "id": _described_schema(
                    _non_null_json_schema_for_column(pk_column),
                    "Identifier of the entry to retrieve.",
                )
            },
            required=("id",),
        )

    return _object_schema(
        {
            "id": _described_schema(
                _object_schema(
                    {
                        column_name: _non_null_json_schema_for_column(table.get_column(column_name))
                        for column_name in table.primary_key
                    },
                    required=table.primary_key,
                ),
                "Identifier of the entry to retrieve. Use an object keyed by ID field name when the entry uses multiple ID fields.",
            )
        },
        required=("id",),
    )


def _find_params_schema(
    table: TableProfile,
    *,
    column: ColumnProfile,
    allowed_ops: tuple[str, ...],
    max_batch_values: int,
    max_items: int,
) -> dict[str, Any]:
    sort_fields = [candidate.column_name for candidate in _sortable_columns(table)]
    return _object_schema(
        {
            "op": _described_schema(
                {"type": "string", "enum": list(allowed_ops)},
                _op_description(allow_any="any" in allowed_ops),
            ),
            "value": _described_schema(
                _find_value_schema(column, max_batch_values=max_batch_values),
                "Value to match against. Use a list for op=in. Use null for op=any.",
            ),
            "sort_by": _described_schema(
                _nullable_enum_schema(sort_fields),
                "Field to order results by. Use null for seeded deterministic order.",
            ),
            "direction": _described_schema(
                {"type": "string", "enum": ["asc", "desc"]},
                "Sort order: asc (smallest first) or desc (largest first).",
            ),
            "limit": _limit_param_schema(max_items=max_items),
        },
        required=("op", "value", "sort_by", "direction", "limit"),
    )


def _calc_params_schema(
    graph: SchemaGraph | None,
    table: TableProfile,
    *,
    allowed_fns: tuple[str, ...],
    max_batch_values: int,
) -> dict[str, Any]:
    metric_columns = [column.column_name for column in _numeric_columns(table)]
    filter_columns = [column.column_name for column in _find_columns(graph, table)]
    return _object_schema(
        {
            "fn": _described_schema(
                {"type": "string", "enum": list(allowed_fns)},
                "Statistic to compute: count, sum, avg, min, or max.",
            ),
            "metric": _described_schema(
                _nullable_enum_schema(metric_columns),
                "Field to compute the statistic on. Use null when fn=count.",
            ),
            "by": _described_schema(
                _nullable_enum_schema(filter_columns),
                "Field to filter on. Use null for no filter.",
            ),
            "op": _described_schema(
                _nullable_enum_schema(_CALC_FILTER_OPS),
                "Condition type: eq (exact), in (any of list), lt, gt, lte, gte (comparison), like (pattern). Use null for no filter.",
            ),
            "value": _described_schema(
                _generic_filter_value_schema(graph, table, max_batch_values=max_batch_values),
                "Value to match against. Single value or list (for op=in). Use null for no filter.",
            ),
        },
        required=("fn", "metric", "by", "op", "value"),
    )


def _rank_params_schema(
    graph: SchemaGraph | None,
    table: TableProfile,
    *,
    allowed_fns: tuple[str, ...],
    max_batch_values: int,
    max_items: int,
) -> dict[str, Any]:
    metric_columns = [column.column_name for column in _numeric_columns(table)]
    filter_columns = [column.column_name for column in _find_columns(graph, table)]
    return _object_schema(
        {
            "fn": _described_schema(
                {"type": "string", "enum": list(allowed_fns)},
                "Statistic to compute: count, sum, avg, min, or max.",
            ),
            "metric": _described_schema(
                _nullable_enum_schema(metric_columns),
                "Field to compute the statistic on. Use null when fn=count.",
            ),
            "direction": _described_schema(
                {"type": "string", "enum": ["asc", "desc"]},
                "Sort order: asc (smallest first) or desc (largest first).",
            ),
            "limit": _limit_param_schema(max_items=max_items),
            "by": _described_schema(
                _nullable_enum_schema(filter_columns),
                "Field to filter on. Use null for no filter.",
            ),
            "op": _described_schema(
                _nullable_enum_schema(_CALC_FILTER_OPS),
                "Condition type: eq (exact), in (any of list), lt, gt, lte, gte (comparison), like (pattern). Use null for no filter.",
            ),
            "value": _described_schema(
                _generic_filter_value_schema(graph, table, max_batch_values=max_batch_values),
                "Value to match against. Single value or list (for op=in). Use null for no filter.",
            ),
        },
        required=("fn", "metric", "direction", "limit", "by", "op", "value"),
    )


def _object_schema(
    properties: dict[str, dict[str, Any]],
    *,
    required: tuple[str, ...] | tuple[()] = (),
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _described_schema(schema: dict[str, Any], description: str) -> dict[str, Any]:
    payload = dict(schema)
    payload["description"] = description
    return payload


def _nullable_enum_schema(values: tuple[str, ...] | list[str]) -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "string", "enum": list(values)},
            {"type": "null"},
        ]
    }


def _find_value_schema(column: ColumnProfile, *, max_batch_values: int) -> dict[str, Any]:
    scalar_schema = _non_null_json_schema_for_column(column)
    return {
        "anyOf": [
            scalar_schema,
            {
                "type": "array",
                "items": scalar_schema,
                "minItems": 1,
                "maxItems": max_batch_values,
            },
            {"type": "null"},
        ]
    }


def _generic_filter_value_schema(
    graph: SchemaGraph | None,
    table: TableProfile,
    *,
    max_batch_values: int,
) -> dict[str, Any]:
    scalar_variants = _dedupe_schemas(
        [_non_null_json_schema_for_column(column) for column in _find_columns(graph, table)]
    ) or [{"type": "string"}]
    array_variants = [
        {
            "type": "array",
            "items": schema,
            "minItems": 1,
            "maxItems": max_batch_values,
        }
        for schema in scalar_variants
    ]
    return {
        "anyOf": [
            *scalar_variants,
            *array_variants,
            {"type": "null"},
        ]
    }


def _dedupe_schemas(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for schema in schemas:
        key = json.dumps(schema, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(schema)
    return deduped


def _row_object_schema(columns: tuple[ColumnProfile, ...]) -> dict[str, Any]:
    return _object_schema(
        {column.column_name: _json_schema_for_column(column) for column in columns},
        required=tuple(column.column_name for column in columns if not column.is_nullable),
    )


def _nullable_object_schema(columns: tuple[ColumnProfile, ...]) -> dict[str, Any]:
    return {
        "anyOf": [
            _row_object_schema(columns),
            {"type": "null"},
        ]
    }


def _row_list_schema(
    columns: tuple[ColumnProfile, ...],
    *,
    max_items: int,
) -> dict[str, Any]:
    return {
        "type": "array",
        "items": _row_object_schema(columns),
        "maxItems": max_items,
    }


def _calc_returns_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "integer"},
            {"type": "number"},
            {"type": "null"},
        ]
    }


def _rank_returns_schema(
    *,
    group_column: ColumnProfile,
    max_items: int,
) -> dict[str, Any]:
    return {
        "type": "array",
        "items": _object_schema(
            {
                "group_key": _json_schema_for_column(group_column),
                "value": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "number"},
                    ]
                },
            },
            required=("group_key", "value"),
        ),
        "maxItems": max_items,
    }


def _json_schema_for_column(column: ColumnProfile) -> dict[str, Any]:
    base = _non_null_json_schema_for_column(column)
    if column.is_nullable:
        if isinstance(base.get("type"), str):
            base["type"] = [base["type"], "null"]
    return base


def _non_null_json_schema_for_column(column: ColumnProfile) -> dict[str, Any]:
    normalized = _normalized_data_type(column.data_type)
    if normalized in _INTEGER_TYPES:
        return {"type": "integer"}
    if normalized in _FLOAT_TYPES:
        return {"type": "number"}
    if normalized in _BOOLEAN_TYPES:
        return {"type": "boolean"}
    if normalized in _DATE_TYPES:
        return {"type": "string", "format": "date"}
    if normalized in _DATETIME_TYPES:
        return {"type": "string", "format": "date-time"}
    return {"type": "string"}


def _limit_param_schema(*, max_items: int) -> dict[str, Any]:
    return {
        "type": "integer",
        "minimum": 1,
        "maximum": max_items,
        "description": "Maximum number of results.",
    }


def _get_sql_template(table: TableProfile) -> str:
    alias = "t"
    return readonly_query(
        f"""
        SELECT *
        FROM {quote_table(table.schema_name, table.table_name)} AS {alias}
        WHERE {_pk_predicate_sql(alias, table, start_index=1)}
        LIMIT 1
        """
    )


def _find_sql_template(table: TableProfile, column: ColumnProfile) -> str:
    alias = "t"
    seed_index = 3
    return readonly_query(
        f"""
        SELECT *
        FROM {quote_table(table.schema_name, table.table_name)} AS {alias}
        WHERE /* dynamic condition for {quote_ident(column.column_name)} */ TRUE
        ORDER BY {_seeded_order_hash_sql(alias, table, seed_param_index=seed_index)}{", " if _order_by_sql(alias, table) else ""}{_order_by_sql(alias, table)}
        LIMIT $1
        """
    )


def _calc_sql_template(table: TableProfile) -> str:
    return readonly_query(
        f"""
        SELECT /* dynamic statistic */ COUNT(*)::bigint AS value
        FROM {quote_table(table.schema_name, table.table_name)} AS t
        WHERE /* optional dynamic condition */ TRUE
        """
    )


def _rank_sql_template(table: TableProfile, group_column: ColumnProfile) -> str:
    seed_index = 2
    return readonly_query(
        f"""
        SELECT t.{quote_ident(group_column.column_name)} AS group_key,
               COUNT(*)::bigint AS value
        FROM {quote_table(table.schema_name, table.table_name)} AS t
        WHERE /* optional dynamic condition */ TRUE
        GROUP BY t.{quote_ident(group_column.column_name)}
        ORDER BY value DESC,
                 md5(concat_ws('|', COALESCE(t.{quote_ident(group_column.column_name)}::text, ''), COALESCE(${seed_index}::text, ''))) ASC,
                 t.{quote_ident(group_column.column_name)} ASC
        LIMIT $1
        """
    )


def _pk_predicate_sql(alias: str, table: TableProfile, *, start_index: int) -> str:
    return " AND ".join(
        f"{alias}.{quote_ident(column_name)} = ${index}"
        for index, column_name in enumerate(table.primary_key, start=start_index)
    )


def _seeded_order_hash_sql(alias: str, table: TableProfile, *, seed_param_index: int) -> str:
    column_names = list(table.primary_key) or [column.column_name for column in table.columns]
    parts = [
        *(f"COALESCE({alias}.{quote_ident(column_name)}::text, '')" for column_name in column_names),
        f"COALESCE(${seed_param_index}::text, '')",
    ]
    return f"md5(concat_ws('|', {', '.join(parts)})) ASC"


def _order_by_sql(alias: str, table: TableProfile) -> str:
    column_names = list(table.primary_key) or [column.column_name for column in table.columns]
    return ", ".join(f"{alias}.{quote_ident(column_name)} ASC" for column_name in column_names)


def _postgres_array_cast(column: ColumnProfile) -> str:
    normalized = _normalized_data_type(column.data_type)
    if normalized in _INTEGER_TYPES:
        if normalized == "smallint":
            return "int2"
        if normalized in {"integer", "int4"}:
            return "int4"
        return "int8"
    if normalized in _FLOAT_TYPES:
        if normalized == "real":
            return "float4"
        if normalized == "double precision":
            return "float8"
        return "numeric"
    if normalized in _BOOLEAN_TYPES:
        return "bool"
    if normalized in _DATE_TYPES:
        return "date"
    if normalized in _DATETIME_TYPES:
        return "timestamptz" if normalized == "timestamptz" else "timestamp"
    if normalized == "uuid":
        return "uuid"
    return "text"


def _render_atomic_tool_source(
    *,
    db_id: str,
    tools: list[AtomicToolDefinition],
    max_batch_values: int,
    max_bounded_result_limit: int,
    float_precision: int,
) -> str:
    lines = [
        '"""Schema-derived atomic tools for one database."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        f"DB_ID = {db_id!r}",
        f"MAX_BATCH_VALUES = {max_batch_values}",
        f"MAX_BOUNDED_RESULT_LIMIT = {max_bounded_result_limit}",
        f"FLOAT_PRECISION = {float_precision}",
        "",
        "def _row_to_dict(row: Any) -> dict[str, Any] | None:",
        "    if row is None:",
        "        return None",
        "    return dict(row)",
        "",
        "def _rows_to_dicts(rows: list[Any]) -> list[dict[str, Any]]:",
        "    return [dict(row) for row in rows]",
        "",
        "def _quote_ident(identifier: str) -> str:",
        "    return '\"' + identifier.replace('\"', '\"\"') + '\"'",
        "",
        "def _quote_table(schema_name: str, table_name: str) -> str:",
        "    return f\"{_quote_ident(schema_name)}.{_quote_ident(table_name)}\"",
        "",
        "def _readonly_query(sql: str) -> str:",
        "    return ' '.join(sql.split())",
        "",
        "def _bounded_limit(limit: int) -> int:",
        "    if isinstance(limit, bool) or not isinstance(limit, int):",
        '        raise TypeError("limit must be an integer")',
        "    if limit < 1:",
        '        raise ValueError("limit must be >= 1")',
        "    return min(limit, MAX_BOUNDED_RESULT_LIMIT)",
        "",
        "def _ensure_bounded_sequence(values: list[Any], *, name: str) -> None:",
        "    if len(values) > MAX_BATCH_VALUES:",
        '        raise ValueError(f"{name} exceeds MAX_BATCH_VALUES")',
        "",
        "def _normalize_direction(direction: str) -> str:",
        "    if direction not in {'asc', 'desc'}:",
        '        raise ValueError("direction must be asc or desc")',
        "    return direction.upper()",
        "",
        "def _seeded_order_hash_sql(alias: str, column_names: list[str], *, seed_param_index: int) -> str:",
        "    parts = [*(f\"COALESCE({alias}.{_quote_ident(column_name)}::text, '')\" for column_name in column_names), f\"COALESCE(${seed_param_index}::text, '')\"]",
        "    return f\"md5(concat_ws('|', {', '.join(parts)})) ASC\"",
        "",
        "def _stable_order_sql(alias: str, column_names: list[str]) -> str:",
        "    if not column_names:",
        "        return ''",
        "    return ', '.join(f\"{alias}.{_quote_ident(column_name)} ASC\" for column_name in column_names)",
        "",
        "def _coerce_id_params(meta: dict[str, Any], id_value: Any) -> list[Any]:",
        "    primary_key = list(meta['primary_key'])",
        "    if len(primary_key) == 1:",
        "        if isinstance(id_value, dict):",
        "            expected_key = primary_key[0]",
        "            if set(id_value) != {expected_key}:",
        '                raise ValueError("id object must match the single ID field exactly")',
        "            return [id_value[expected_key]]",
        "        return [id_value]",
        "    if not isinstance(id_value, dict):",
        '        raise TypeError("id must be an object when the entry uses multiple ID fields")',
        "    if set(id_value) != set(primary_key):",
        '        raise ValueError("id object must include exactly the ID field names for the entry")',
        "    return [id_value[column_name] for column_name in primary_key]",
        "",
        "def _validate_choice(value: Any, allowed: list[str], *, allow_null: bool = False, field_name: str) -> Any:",
        "    if value is None:",
        "        if allow_null:",
        "            return None",
        '        raise ValueError(f"{field_name} cannot be null")',
        "    if value not in allowed:",
        '        raise ValueError(f\"{field_name} must be one of {sorted(allowed)}\")',
        "    return value",
        "",
        "def _validate_find_value(op: str, value: Any) -> None:",
        "    if op == 'any':",
        "        if value is not None:",
        '            raise ValueError("value must be null when op=any")',
        "        return",
        "    if op == 'in':",
        "        if not isinstance(value, list) or not value:",
        '            raise ValueError("value must be a non-empty list when op=in")',
        "        _ensure_bounded_sequence(value, name='value')",
        "        return",
        "    if value is None:",
        '        raise ValueError("value cannot be null for this op")',
        "    if op == 'like' and not isinstance(value, str):",
        '        raise ValueError("value must be a string when op=like")',
        "",
        "def _validate_filter_context(meta: dict[str, Any], by: Any, op: Any, value: Any) -> tuple[str | None, str | None, Any]:",
        "    filter_columns = list(meta['filter_columns'])",
        "    if by is None:",
        "        if op is not None or value is not None:",
        '            raise ValueError("op and value must be null when by is null")',
        "        return None, None, None",
        "    by = _validate_choice(by, filter_columns, field_name='by')",
        "    if op is None:",
        '        raise ValueError("op is required when by is provided")',
        "    op = _validate_choice(op, list(meta.get('filter_ops', ['eq', 'in', 'lt', 'gt', 'lte', 'gte', 'like'])), field_name='op')",
        "    _validate_dynamic_op(meta, by, op)",
        "    _validate_find_value(op, value)",
        "    return by, op, value",
        "",
        "def _validate_dynamic_op(meta: dict[str, Any], column_name: str, op: str) -> None:",
        "    column_type = meta['column_types'][column_name]",
        "    allowed = {'eq', 'in'}",
        "    if column_type in {'bigint', 'int2', 'int4', 'int8', 'integer', 'smallint', 'decimal', 'double precision', 'float4', 'float8', 'numeric', 'real', 'date', 'timestamp', 'timestamp with time zone', 'timestamp without time zone', 'timestamptz'}:",
        "        allowed.update({'lt', 'gt', 'lte', 'gte'})",
        "    if column_type in {'bpchar', 'char', 'character', 'character varying', 'citext', 'name', 'text', 'uuid', 'varchar'}:",
        "        allowed.add('like')",
        "    if op not in allowed:",
        '        raise ValueError(f"op={op} is not supported for the selected field")',
        "",
        "def _predicate_sql(*, alias: str, column_name: str, column_cast: str, op: str, start_index: int) -> tuple[str, int]:",
        "    column_sql = f\"{alias}.{_quote_ident(column_name)}\"",
        "    if op == 'eq':",
        "        return f\"{column_sql} = ${start_index}\", 1",
        "    if op == 'in':",
        "        return f\"{column_sql} = ANY(${start_index}::{column_cast}[])\", 1",
        "    if op == 'lt':",
        "        return f\"{column_sql} < ${start_index}\", 1",
        "    if op == 'gt':",
        "        return f\"{column_sql} > ${start_index}\", 1",
        "    if op == 'lte':",
        "        return f\"{column_sql} <= ${start_index}\", 1",
        "    if op == 'gte':",
        "        return f\"{column_sql} >= ${start_index}\", 1",
        "    if op == 'like':",
        "        return f\"{column_sql} ILIKE ${start_index}\", 1",
        "    if op == 'any':",
        "        return 'TRUE', 0",
        '    raise ValueError(f"unsupported op: {op}")',
        "",
        "def _aggregate_sql(fn: str, metric: str, *, column_type: str) -> str:",
        "    column_sql = f\"t.{_quote_ident(metric)}\"",
        "    if fn == 'count':",
        "        return 'COUNT(*)::bigint'",
        "    base = f\"{fn.upper()}({column_sql})\"",
        "    if fn == 'avg':",
        "        return f\"ROUND({base}::numeric, {FLOAT_PRECISION})\"",
        "    if fn == 'sum' and column_type in {'decimal', 'double precision', 'float4', 'float8', 'numeric', 'real'}:",
        "        return f\"ROUND({base}::numeric, {FLOAT_PRECISION})\"",
        "    return base",
        "",
        "async def _run_get(conn, meta: dict[str, Any], id: Any) -> dict[str, Any] | None:",
        "    params = _coerce_id_params(meta, id)",
        "    predicates = ' AND '.join(",
        "        f\"t.{_quote_ident(column_name)} = ${index}\"",
        "        for index, column_name in enumerate(meta['primary_key'], start=1)",
        "    )",
        "    sql = _readonly_query(",
        "        f\"SELECT * FROM {_quote_table(meta['schema_name'], meta['table_name'])} AS t WHERE {predicates} LIMIT 1\"",
        "    )",
        "    row = await conn.fetchrow(sql, *params)",
        "    return _row_to_dict(row)",
        "",
        "async def _run_find(conn, meta: dict[str, Any], op: str, value: Any, sort_by: Any, direction: str, limit: int, shuffle_seed: Any) -> list[dict[str, Any]]:",
        "    op = _validate_choice(op, list(meta['allowed_ops']), field_name='op')",
        "    _validate_find_value(op, value)",
        "    limit = _bounded_limit(limit)",
        "    if sort_by is not None:",
        "        sort_by = _validate_choice(sort_by, list(meta['sortable_columns']), field_name='sort_by')",
        "    direction_sql = _normalize_direction(direction)",
        "    params: list[Any] = []",
        "    if op == 'any':",
        "        where_sql = 'TRUE'",
        "    else:",
        "        where_sql, _ = _predicate_sql(",
        "            alias='t',",
        "            column_name=str(meta['filter_column']),",
        "            column_cast=str(meta['array_casts'][meta['filter_column']]),",
        "            op=op,",
        "            start_index=1,",
        "        )",
        "        params.append(value)",
        "    params.append(limit)",
        "    seed_param_index = len(params) + 1",
        "    seed_sql = _seeded_order_hash_sql('t', list(meta['order_columns']), seed_param_index=seed_param_index)",
        "    stable_sql = _stable_order_sql('t', list(meta['order_columns']))",
        "    if sort_by is None:",
        "        order_parts = [seed_sql]",
        "    else:",
        "        order_parts = [f\"t.{_quote_ident(sort_by)} {direction_sql}\", seed_sql]",
        "    if stable_sql:",
        "        order_parts.append(stable_sql)",
        "    sql = _readonly_query(",
        "        f\"SELECT * FROM {_quote_table(meta['schema_name'], meta['table_name'])} AS t WHERE {where_sql} ORDER BY {', '.join(order_parts)} LIMIT ${len(params)}\"",
        "    )",
        "    params.append(shuffle_seed)",
        "    rows = await conn.fetch(sql, *params)",
        "    return _rows_to_dicts(rows)",
        "",
        "async def _run_calc(conn, meta: dict[str, Any], fn: str, metric: Any, by: Any, op: Any, value: Any) -> Any:",
        "    fn = _validate_choice(fn, list(meta['allowed_fns']), field_name='fn')",
        "    if fn == 'count':",
        "        if metric is not None:",
        '            raise ValueError("metric must be null when fn=count")',
        "        metric_name = None",
        "    else:",
        "        metric_name = _validate_choice(metric, list(meta['numeric_columns']), field_name='metric')",
        "    by, op, value = _validate_filter_context(meta, by, op, value)",
        "    params: list[Any] = []",
        "    predicates: list[str] = []",
        "    if by is not None and op is not None:",
        "        predicate_sql, _ = _predicate_sql(",
        "            alias='t',",
        "            column_name=str(by),",
        "            column_cast=str(meta['array_casts'][by]),",
        "            op=op,",
        "            start_index=1,",
        "        )",
        "        predicates.append(predicate_sql)",
        "        params.append(value)",
        "    if metric_name is not None:",
        "        predicates.append(f\"t.{_quote_ident(metric_name)} IS NOT NULL\")",
        "        column_type = str(meta['column_types'][metric_name])",
        "        aggregate_sql = _aggregate_sql(fn, metric_name, column_type=column_type)",
        "    else:",
        "        aggregate_sql = _aggregate_sql('count', '__unused__', column_type='integer')",
        "    where_sql = 'TRUE' if not predicates else ' AND '.join(predicates)",
        "    sql = _readonly_query(",
        "        f\"SELECT {aggregate_sql} AS value FROM {_quote_table(meta['schema_name'], meta['table_name'])} AS t WHERE {where_sql}\"",
        "    )",
        "    return await conn.fetchval(sql, *params)",
        "",
        "async def _run_rank(conn, meta: dict[str, Any], fn: str, metric: Any, direction: str, limit: int, by: Any, op: Any, value: Any, shuffle_seed: Any) -> list[dict[str, Any]]:",
        "    fn = _validate_choice(fn, list(meta['allowed_fns']), field_name='fn')",
        "    if fn == 'count':",
        "        if metric is not None:",
        '            raise ValueError("metric must be null when fn=count")',
        "        metric_name = None",
        "    else:",
        "        metric_name = _validate_choice(metric, list(meta['numeric_columns']), field_name='metric')",
        "    by, op, value = _validate_filter_context(meta, by, op, value)",
        "    limit = _bounded_limit(limit)",
        "    direction_sql = _normalize_direction(direction)",
        "    params: list[Any] = []",
        "    predicates: list[str] = []",
        "    if by is not None and op is not None:",
        "        predicate_sql, _ = _predicate_sql(",
        "            alias='t',",
        "            column_name=str(by),",
        "            column_cast=str(meta['array_casts'][by]),",
        "            op=op,",
        "            start_index=1,",
        "        )",
        "        predicates.append(predicate_sql)",
        "        params.append(value)",
        "    if metric_name is not None:",
        "        predicates.append(f\"t.{_quote_ident(metric_name)} IS NOT NULL\")",
        "        column_type = str(meta['column_types'][metric_name])",
        "        aggregate_sql = _aggregate_sql(fn, metric_name, column_type=column_type)",
        "    else:",
        "        aggregate_sql = _aggregate_sql('count', '__unused__', column_type='integer')",
        "    params.append(limit)",
        "    seed_param_index = len(params) + 1",
        "    group_column = str(meta['group_column'])",
        "    tiebreak_sql = _seeded_order_hash_sql('t', [group_column], seed_param_index=seed_param_index)",
        "    where_sql = 'TRUE' if not predicates else ' AND '.join(predicates)",
        "    sql = _readonly_query(",
        "        f\"SELECT t.{_quote_ident(group_column)} AS group_key, {aggregate_sql} AS value FROM {_quote_table(meta['schema_name'], meta['table_name'])} AS t WHERE {where_sql} GROUP BY t.{_quote_ident(group_column)} ORDER BY value {direction_sql}, {tiebreak_sql}, t.{_quote_ident(group_column)} ASC LIMIT ${len(params)}\"",
        "    )",
        "    params.append(shuffle_seed)",
        "    rows = await conn.fetch(sql, *params)",
        "    return _rows_to_dicts(rows)",
        "",
    ]
    for tool in tools:
        lines.extend(_render_tool_function(tool))
    return "\n".join(lines).rstrip() + "\n"


def _render_tool_function(tool: AtomicToolDefinition) -> list[str]:
    meta_literal = repr(tool.runtime_metadata)
    if tool.family is AtomicToolFamily.GET:
        return [
            f"async def {tool.name}(conn, id):",
            f"    meta = {meta_literal}",
            "    return await _run_get(conn, meta, id)",
            "",
        ]
    if tool.family is AtomicToolFamily.FIND:
        return [
            f"async def {tool.name}(conn, op, value, sort_by, direction, limit, _shuffle_seed=None):",
            f"    meta = {meta_literal}",
            "    return await _run_find(conn, meta, op, value, sort_by, direction, limit, _shuffle_seed)",
            "",
        ]
    if tool.family is AtomicToolFamily.CALC:
        return [
            f"async def {tool.name}(conn, fn, metric, by, op, value):",
            f"    meta = {meta_literal}",
            "    return await _run_calc(conn, meta, fn, metric, by, op, value)",
            "",
        ]
    return [
        f"async def {tool.name}(conn, fn, metric, direction, limit, by, op, value, _shuffle_seed=None):",
        f"    meta = {meta_literal}",
        "    return await _run_rank(conn, meta, fn, metric, direction, limit, by, op, value, _shuffle_seed)",
        "",
    ]


def _tool_requires_shuffle_seed(tool: AtomicToolDefinition) -> bool:
    return tool.family in {AtomicToolFamily.FIND, AtomicToolFamily.RANK}


def readonly_query(sql: str) -> str:
    return " ".join(sql.split())


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def quote_table(schema_name: str, table_name: str) -> str:
    return f"{quote_ident(schema_name)}.{quote_ident(table_name)}"


def singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 1:
        return token[:-1]
    return token


def humanize_identifier(identifier: str) -> str:
    parts = [part for part in identifier.replace("-", "_").split("_") if part]
    if not parts:
        return identifier.strip()
    return " ".join(parts)


def _op_description(*, allow_any: bool) -> str:
    if allow_any:
        return (
            "Condition type: any (no condition), eq (exact), in (any of list), "
            "lt, gt, lte, gte (comparison), like (pattern)."
        )
    return (
        "Condition type: eq (exact), in (any of list), lt, gt, lte, gte "
        "(comparison), like (pattern)."
    )
