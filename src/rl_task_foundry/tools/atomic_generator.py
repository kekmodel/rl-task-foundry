"""Deterministic db-level atomic tool generation from a schema graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from rl_task_foundry.config.models import AtomicToolConfig, StrictModel
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.tools.sql_templates import quote_ident, quote_table, readonly_query
from rl_task_foundry.tools.text_utils import humanize_identifier, singularize_token

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
_SCALAR_TYPES = _TEXT_TYPES | _INTEGER_TYPES | _FLOAT_TYPES | _BOOLEAN_TYPES | _DATE_TYPES | _DATETIME_TYPES
_COMPRESSION_DROP_ORDER = ("aggregate", "like", "distinct", "range", "in")


class AtomicToolFamily(StrEnum):
    T1_POINT_LOOKUP = "t1_point_lookup"
    T2_BOUNDED_ENUMERATION = "t2_bounded_enumeration"
    T3_SINGLE_COLUMN_FILTER = "t3_single_column_filter"
    T4_FK_TRAVERSAL = "t4_fk_traversal"
    T5_DISTINCT_VALUES = "t5_distinct_values"
    T6_FILTERED_AGGREGATE = "t6_filtered_aggregate"


class AtomicToolResultMode(StrEnum):
    OBJECT_OR_NULL = "object_or_null"
    ROW_LIST = "row_list"
    SCALAR = "scalar"
    SCALAR_LIST = "scalar_list"


class AtomicToolDefinition(StrictModel):
    name: str
    family: AtomicToolFamily
    description: str
    params_schema: dict[str, Any]
    returns_schema: dict[str, Any]
    sql: str
    result_mode: AtomicToolResultMode
    semantic_key: str
    compression_bucket: Literal["core", "aggregate", "like", "distinct", "range", "in"] = "core"
    scalar_list_column: str | None = None

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
            compiled.extend(self._generate_table_tools(table, table_slugs))
        for edge in sorted(
            graph.edges,
            key=lambda item: (
                item.source_schema,
                item.source_table,
                item.target_schema,
                item.target_table,
                item.constraint_name,
            ),
        ):
            compiled.extend(self._generate_edge_tools(graph, edge, table_slugs))

        compiled = _compress_tools(
            sorted(compiled, key=lambda tool: (_family_rank(tool.family), tool.name)),
            max_tool_count=self.config.max_tool_count,
        )
        return AtomicToolBundle(
            db_id=db_id,
            tools=compiled,
            source=_render_atomic_tool_source(
                db_id=db_id,
                tools=compiled,
                max_batch_values=self.config.max_batch_values,
            ),
        )

    def _generate_table_tools(
        self,
        table: TableProfile,
        table_slugs: dict[tuple[str, str], str],
    ) -> list[AtomicToolDefinition]:
        table_slug = table_slugs[(table.schema_name, table.table_name)]
        projection_columns = _projection_columns(table)
        tools: list[AtomicToolDefinition] = []

        if table.primary_key:
            tools.append(self._build_get_by_id_tool(table, table_slug, projection_columns))
            if len(table.primary_key) == 1:
                tools.append(self._build_get_by_ids_batch_tool(table, table_slug, projection_columns))
            tools.append(self._build_count_tool(table, table_slug))
            tools.append(self._build_list_ids_tool(table, table_slug))
        else:
            tools.append(self._build_count_tool(table, table_slug))

        filter_columns = _filter_columns(table)
        for column in filter_columns:
            tools.append(self._build_eq_filter_tool(table, table_slug, projection_columns, column))
            if _supports_in_filter(column):
                tools.append(self._build_in_filter_tool(table, table_slug, projection_columns, column))
            if _supports_range_filter(column):
                tools.append(self._build_range_filter_tool(table, table_slug, projection_columns, column))
            if _supports_like_filter(column):
                tools.append(self._build_like_filter_tool(table, table_slug, projection_columns, column))
            if _supports_distinct(column):
                tools.append(self._build_distinct_tool(table, table_slug, column))

        metric_columns = [column for column in filter_columns if _is_numeric(column)]
        aggregate_filter_columns = [column for column in filter_columns if _supports_eq_filter(column)]
        for metric in metric_columns:
            for filter_column in aggregate_filter_columns:
                if filter_column.column_name == metric.column_name:
                    continue
                tools.extend(
                    self._build_aggregate_tools(
                        table,
                        table_slug,
                        metric=metric,
                        filter_column=filter_column,
                    )
                )
        return tools

    def _generate_edge_tools(
        self,
        graph: SchemaGraph,
        edge: ForeignKeyEdge,
        table_slugs: dict[tuple[str, str], str],
    ) -> list[AtomicToolDefinition]:
        source_table = graph.get_table(edge.source_table, schema_name=edge.source_schema)
        target_table = graph.get_table(edge.target_table, schema_name=edge.target_schema)
        source_slug = table_slugs[(source_table.schema_name, source_table.table_name)]
        target_slug = table_slugs[(target_table.schema_name, target_table.table_name)]
        fk_slug = "_".join(edge.source_columns)
        source_projection = _projection_columns(source_table)
        target_projection = _projection_columns(target_table)
        if not source_table.primary_key or not target_table.primary_key:
            return []

        forward_sql = readonly_query(
            f"""
            SELECT {_select_projection_sql("target", target_projection)}
            FROM {quote_table(source_table.schema_name, source_table.table_name)} AS source
            JOIN {quote_table(target_table.schema_name, target_table.table_name)} AS target
              ON {_edge_join_sql("source", "target", edge)}
            WHERE {_pk_predicate_sql("source", source_table, start_index=1)}
            LIMIT 1
            """
        )
        reverse_sql = readonly_query(
            f"""
            SELECT {_select_projection_sql("source", source_projection)}
            FROM {quote_table(target_table.schema_name, target_table.table_name)} AS target
            JOIN {quote_table(source_table.schema_name, source_table.table_name)} AS source
              ON {_edge_join_sql("source", "target", edge)}
            WHERE {_pk_predicate_sql("target", target_table, start_index=1)}
            ORDER BY {_order_by_sql("source", source_table, source_projection)}
            LIMIT {self.config.bounded_result_limit}
            """
        )
        return [
            AtomicToolDefinition(
                name=f"traverse_{source_slug}_to_{target_slug}_via_{fk_slug}",
                family=AtomicToolFamily.T4_FK_TRAVERSAL,
                description=(
                    f"One-hop FK traversal from {humanize_identifier(source_slug)} to "
                    f"{humanize_identifier(target_slug)} via {humanize_identifier(fk_slug)}."
                ),
                params_schema=_pk_params_schema(source_table),
                returns_schema=_nullable_object_schema(target_projection),
                sql=forward_sql,
                result_mode=AtomicToolResultMode.OBJECT_OR_NULL,
                semantic_key=f"{source_table.qualified_name}->{edge.constraint_name}->{target_table.qualified_name}:forward",
            ),
            AtomicToolDefinition(
                name=f"traverse_{target_slug}_to_{source_slug}_via_{fk_slug}",
                family=AtomicToolFamily.T4_FK_TRAVERSAL,
                description=(
                    f"One-hop reverse FK traversal from {humanize_identifier(target_slug)} to "
                    f"{humanize_identifier(source_slug)} via {humanize_identifier(fk_slug)}."
                ),
                params_schema=_pk_params_schema(target_table),
                returns_schema=_row_list_schema(
                    source_projection,
                    max_items=self.config.bounded_result_limit,
                ),
                sql=reverse_sql,
                result_mode=AtomicToolResultMode.ROW_LIST,
                semantic_key=f"{target_table.qualified_name}->{edge.constraint_name}->{source_table.qualified_name}:reverse",
            ),
        ]

    def _build_get_by_id_tool(
        self,
        table: TableProfile,
        table_slug: str,
        projection_columns: tuple[ColumnProfile, ...],
    ) -> AtomicToolDefinition:
        sql = readonly_query(
            f"""
            SELECT {_select_projection_sql("t", projection_columns)}
            FROM {quote_table(table.schema_name, table.table_name)} AS t
            WHERE {_pk_predicate_sql("t", table, start_index=1)}
            LIMIT 1
            """
        )
        return AtomicToolDefinition(
            name=f"get_{table_slug}_by_id",
            family=AtomicToolFamily.T1_POINT_LOOKUP,
            description=f"Point lookup for one {humanize_identifier(table_slug)} row by primary key.",
            params_schema=_pk_params_schema(table),
            returns_schema=_nullable_object_schema(projection_columns),
            sql=sql,
            result_mode=AtomicToolResultMode.OBJECT_OR_NULL,
            semantic_key=f"{table.qualified_name}:get_by_id",
        )

    def _build_get_by_ids_batch_tool(
        self,
        table: TableProfile,
        table_slug: str,
        projection_columns: tuple[ColumnProfile, ...],
    ) -> AtomicToolDefinition:
        pk_column = table.get_column(table.primary_key[0])
        sql = readonly_query(
            f"""
            SELECT {_select_projection_sql("t", projection_columns)}
            FROM {quote_table(table.schema_name, table.table_name)} AS t
            WHERE t.{quote_ident(pk_column.column_name)} = ANY($1::{_postgres_array_cast(pk_column)}[])
            ORDER BY {_order_by_sql("t", table, projection_columns)}
            """
        )
        return AtomicToolDefinition(
            name=f"get_{table_slug}_by_ids_batch",
            family=AtomicToolFamily.T1_POINT_LOOKUP,
            description=(
                f"Batch point lookup for {humanize_identifier(table_slug)} rows by primary key values."
            ),
            params_schema=_array_params_schema(
                "ids",
                items_schema=_non_null_json_schema_for_column(pk_column),
                max_items=self.config.max_batch_values,
                description=f"Primary key values for {humanize_identifier(table_slug)}.",
            ),
            returns_schema=_row_list_schema(
                projection_columns,
                max_items=self.config.max_batch_values,
            ),
            sql=sql,
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key=f"{table.qualified_name}:get_by_ids_batch",
        )

    def _build_count_tool(self, table: TableProfile, table_slug: str) -> AtomicToolDefinition:
        sql = readonly_query(
            f"""
            SELECT COUNT(*)::bigint AS value
            FROM {quote_table(table.schema_name, table.table_name)} AS t
            """
        )
        return AtomicToolDefinition(
            name=f"count_{table_slug}",
            family=AtomicToolFamily.T2_BOUNDED_ENUMERATION,
            description=f"Count rows in {humanize_identifier(table_slug)}.",
            params_schema=_empty_params_schema(),
            returns_schema={"type": "integer"},
            sql=sql,
            result_mode=AtomicToolResultMode.SCALAR,
            semantic_key=f"{table.qualified_name}:count",
        )

    def _build_list_ids_tool(self, table: TableProfile, table_slug: str) -> AtomicToolDefinition:
        id_columns = tuple(table.get_column(column_name) for column_name in table.primary_key)
        sql = readonly_query(
            f"""
            SELECT {_select_projection_sql("t", id_columns)}
            FROM {quote_table(table.schema_name, table.table_name)} AS t
            ORDER BY {_order_by_sql("t", table, id_columns)}
            LIMIT {self.config.bounded_result_limit}
            """
        )
        if len(id_columns) == 1:
            returns_schema = _scalar_list_schema(
                items_schema=_non_null_json_schema_for_column(id_columns[0]),
                max_items=self.config.bounded_result_limit,
            )
            result_mode = AtomicToolResultMode.SCALAR_LIST
        else:
            returns_schema = _row_list_schema(
                id_columns,
                max_items=self.config.bounded_result_limit,
            )
            result_mode = AtomicToolResultMode.ROW_LIST
        return AtomicToolDefinition(
            name=f"list_{table_slug}_ids",
            family=AtomicToolFamily.T2_BOUNDED_ENUMERATION,
            description=f"List bounded primary-key identifiers for {humanize_identifier(table_slug)}.",
            params_schema=_empty_params_schema(),
            returns_schema=returns_schema,
            sql=sql,
            result_mode=result_mode,
            semantic_key=f"{table.qualified_name}:list_ids",
            scalar_list_column=id_columns[0].column_name if len(id_columns) == 1 else None,
        )

    def _build_eq_filter_tool(
        self,
        table: TableProfile,
        table_slug: str,
        projection_columns: tuple[ColumnProfile, ...],
        column: ColumnProfile,
    ) -> AtomicToolDefinition:
        sql = readonly_query(
            f"""
            SELECT {_select_projection_sql("t", projection_columns)}
            FROM {quote_table(table.schema_name, table.table_name)} AS t
            WHERE t.{quote_ident(column.column_name)} = $1
            ORDER BY {_order_by_sql("t", table, projection_columns)}
            LIMIT {self.config.bounded_result_limit}
            """
        )
        return AtomicToolDefinition(
            name=f"filter_{table_slug}_by_{column.column_name}_eq",
            family=AtomicToolFamily.T3_SINGLE_COLUMN_FILTER,
            description=(
                f"Filter {humanize_identifier(table_slug)} rows where "
                f"{humanize_identifier(column.column_name)} exactly matches the supplied value."
            ),
            params_schema=_single_value_params_schema(
                "value",
                _non_null_json_schema_for_column(column),
                description=f"Exact match value for {humanize_identifier(column.column_name)}.",
            ),
            returns_schema=_row_list_schema(
                projection_columns,
                max_items=self.config.bounded_result_limit,
            ),
            sql=sql,
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key=f"{table.qualified_name}:filter:{column.column_name}:eq",
        )

    def _build_in_filter_tool(
        self,
        table: TableProfile,
        table_slug: str,
        projection_columns: tuple[ColumnProfile, ...],
        column: ColumnProfile,
    ) -> AtomicToolDefinition:
        sql = readonly_query(
            f"""
            SELECT {_select_projection_sql("t", projection_columns)}
            FROM {quote_table(table.schema_name, table.table_name)} AS t
            WHERE t.{quote_ident(column.column_name)} = ANY($1::{_postgres_array_cast(column)}[])
            ORDER BY {_order_by_sql("t", table, projection_columns)}
            LIMIT {self.config.bounded_result_limit}
            """
        )
        return AtomicToolDefinition(
            name=f"filter_{table_slug}_by_{column.column_name}_in",
            family=AtomicToolFamily.T3_SINGLE_COLUMN_FILTER,
            description=(
                f"Filter {humanize_identifier(table_slug)} rows where "
                f"{humanize_identifier(column.column_name)} is in the supplied value set."
            ),
            params_schema=_array_params_schema(
                "values",
                items_schema=_non_null_json_schema_for_column(column),
                max_items=self.config.max_batch_values,
                description=f"Accepted values for {humanize_identifier(column.column_name)}.",
            ),
            returns_schema=_row_list_schema(
                projection_columns,
                max_items=self.config.bounded_result_limit,
            ),
            sql=sql,
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key=f"{table.qualified_name}:filter:{column.column_name}:in",
            compression_bucket="in",
        )

    def _build_range_filter_tool(
        self,
        table: TableProfile,
        table_slug: str,
        projection_columns: tuple[ColumnProfile, ...],
        column: ColumnProfile,
    ) -> AtomicToolDefinition:
        boundary_schema = _nullable_json_schema_for_column(column)
        sql = readonly_query(
            f"""
            SELECT {_select_projection_sql("t", projection_columns)}
            FROM {quote_table(table.schema_name, table.table_name)} AS t
            WHERE ($1 IS NULL OR t.{quote_ident(column.column_name)} >= $1)
              AND ($2 IS NULL OR t.{quote_ident(column.column_name)} <= $2)
            ORDER BY {_order_by_sql("t", table, projection_columns)}
            LIMIT {self.config.bounded_result_limit}
            """
        )
        return AtomicToolDefinition(
            name=f"filter_{table_slug}_by_{column.column_name}_range",
            family=AtomicToolFamily.T3_SINGLE_COLUMN_FILTER,
            description=(
                f"Filter {humanize_identifier(table_slug)} rows whose "
                f"{humanize_identifier(column.column_name)} falls within the supplied range."
            ),
            params_schema=_object_schema(
                {
                    "minimum": boundary_schema,
                    "maximum": boundary_schema,
                },
                required=(),
            ),
            returns_schema=_row_list_schema(
                projection_columns,
                max_items=self.config.bounded_result_limit,
            ),
            sql=sql,
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key=f"{table.qualified_name}:filter:{column.column_name}:range",
            compression_bucket="range",
        )

    def _build_like_filter_tool(
        self,
        table: TableProfile,
        table_slug: str,
        projection_columns: tuple[ColumnProfile, ...],
        column: ColumnProfile,
    ) -> AtomicToolDefinition:
        sql = readonly_query(
            f"""
            SELECT {_select_projection_sql("t", projection_columns)}
            FROM {quote_table(table.schema_name, table.table_name)} AS t
            WHERE t.{quote_ident(column.column_name)} ILIKE $1
            ORDER BY {_order_by_sql("t", table, projection_columns)}
            LIMIT {self.config.bounded_result_limit}
            """
        )
        return AtomicToolDefinition(
            name=f"filter_{table_slug}_by_{column.column_name}_like",
            family=AtomicToolFamily.T3_SINGLE_COLUMN_FILTER,
            description=(
                f"Filter {humanize_identifier(table_slug)} rows where "
                f"{humanize_identifier(column.column_name)} matches the supplied LIKE pattern."
            ),
            params_schema=_single_value_params_schema(
                "pattern",
                {"type": "string"},
                description=f"Case-insensitive LIKE pattern for {humanize_identifier(column.column_name)}.",
            ),
            returns_schema=_row_list_schema(
                projection_columns,
                max_items=self.config.bounded_result_limit,
            ),
            sql=sql,
            result_mode=AtomicToolResultMode.ROW_LIST,
            semantic_key=f"{table.qualified_name}:filter:{column.column_name}:like",
            compression_bucket="like",
        )

    def _build_distinct_tool(
        self,
        table: TableProfile,
        table_slug: str,
        column: ColumnProfile,
    ) -> AtomicToolDefinition:
        sql = readonly_query(
            f"""
            SELECT DISTINCT t.{quote_ident(column.column_name)} AS value
            FROM {quote_table(table.schema_name, table.table_name)} AS t
            WHERE t.{quote_ident(column.column_name)} IS NOT NULL
            ORDER BY t.{quote_ident(column.column_name)} ASC
            LIMIT {self.config.bounded_result_limit}
            """
        )
        return AtomicToolDefinition(
            name=f"distinct_{table_slug}_{column.column_name}",
            family=AtomicToolFamily.T5_DISTINCT_VALUES,
            description=(
                f"List bounded distinct values for {humanize_identifier(column.column_name)} "
                f"on {humanize_identifier(table_slug)}."
            ),
            params_schema=_empty_params_schema(),
            returns_schema=_scalar_list_schema(
                items_schema=_non_null_json_schema_for_column(column),
                max_items=self.config.bounded_result_limit,
            ),
            sql=sql,
            result_mode=AtomicToolResultMode.SCALAR_LIST,
            semantic_key=f"{table.qualified_name}:distinct:{column.column_name}",
            compression_bucket="distinct",
            scalar_list_column="value",
        )

    def _build_aggregate_tools(
        self,
        table: TableProfile,
        table_slug: str,
        *,
        metric: ColumnProfile,
        filter_column: ColumnProfile,
    ) -> list[AtomicToolDefinition]:
        tools: list[AtomicToolDefinition] = []
        aggregate_output_schema = _nullable_json_schema_for_metric(metric)
        for aggregate_name in ("sum", "avg", "min", "max"):
            sql = readonly_query(
                f"""
                SELECT {aggregate_name.upper()}(t.{quote_ident(metric.column_name)}) AS value
                FROM {quote_table(table.schema_name, table.table_name)} AS t
                WHERE t.{quote_ident(filter_column.column_name)} = $1
                  AND t.{quote_ident(metric.column_name)} IS NOT NULL
                """
            )
            tools.append(
                AtomicToolDefinition(
                    name=(
                        f"{aggregate_name}_{table_slug}_{metric.column_name}"
                        f"_by_{filter_column.column_name}_eq"
                    ),
                    family=AtomicToolFamily.T6_FILTERED_AGGREGATE,
                    description=(
                        f"Compute {aggregate_name} over {humanize_identifier(metric.column_name)} "
                        f"for {humanize_identifier(table_slug)} rows filtered by "
                        f"{humanize_identifier(filter_column.column_name)}."
                    ),
                    params_schema=_single_value_params_schema(
                        "value",
                        _non_null_json_schema_for_column(filter_column),
                        description=f"Filter value for {humanize_identifier(filter_column.column_name)}.",
                    ),
                    returns_schema=aggregate_output_schema,
                    sql=sql,
                    result_mode=AtomicToolResultMode.SCALAR,
                    semantic_key=(
                        f"{table.qualified_name}:aggregate:{aggregate_name}:"
                        f"{metric.column_name}:by:{filter_column.column_name}"
                    ),
                    compression_bucket="aggregate",
                )
            )
        return tools


def _family_rank(family: AtomicToolFamily) -> int:
    order = {
        AtomicToolFamily.T1_POINT_LOOKUP: 0,
        AtomicToolFamily.T2_BOUNDED_ENUMERATION: 1,
        AtomicToolFamily.T3_SINGLE_COLUMN_FILTER: 2,
        AtomicToolFamily.T4_FK_TRAVERSAL: 3,
        AtomicToolFamily.T5_DISTINCT_VALUES: 4,
        AtomicToolFamily.T6_FILTERED_AGGREGATE: 5,
    }
    return order[family]


def _compress_tools(
    tools: list[AtomicToolDefinition],
    *,
    max_tool_count: int,
) -> list[AtomicToolDefinition]:
    if len(tools) <= max_tool_count:
        return tools

    kept = list(tools)
    for bucket in _COMPRESSION_DROP_ORDER:
        if len(kept) <= max_tool_count:
            break
        kept = [tool for tool in kept if tool.compression_bucket != bucket]

    if len(kept) <= max_tool_count:
        return kept
    return kept[:max_tool_count]


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


def _visible_columns(table: TableProfile) -> tuple[ColumnProfile, ...]:
    return tuple(column for column in table.columns if column.visibility == "user_visible")


def _projection_columns(table: TableProfile) -> tuple[ColumnProfile, ...]:
    seen: set[str] = set()
    projection: list[ColumnProfile] = []
    for column_name in table.primary_key:
        column = table.get_column(column_name)
        projection.append(column)
        seen.add(column.column_name)
    for column in _visible_columns(table):
        if column.column_name in seen:
            continue
        projection.append(column)
        seen.add(column.column_name)
    return tuple(projection)


def _filter_columns(table: TableProfile) -> list[ColumnProfile]:
    return [
        column
        for column in _visible_columns(table)
        if column.column_name not in set(table.primary_key) and _is_scalar(column)
    ]


def _is_scalar(column: ColumnProfile) -> bool:
    return _normalized_data_type(column.data_type) in _SCALAR_TYPES


def _is_numeric(column: ColumnProfile) -> bool:
    return _normalized_data_type(column.data_type) in (_INTEGER_TYPES | _FLOAT_TYPES)


def _supports_eq_filter(column: ColumnProfile) -> bool:
    return _is_scalar(column)


def _supports_in_filter(column: ColumnProfile) -> bool:
    return _supports_eq_filter(column)


def _supports_range_filter(column: ColumnProfile) -> bool:
    return _normalized_data_type(column.data_type) in (
        _INTEGER_TYPES | _FLOAT_TYPES | _DATE_TYPES | _DATETIME_TYPES
    )


def _supports_like_filter(column: ColumnProfile) -> bool:
    return _normalized_data_type(column.data_type) in _TEXT_TYPES


def _supports_distinct(column: ColumnProfile) -> bool:
    return _supports_eq_filter(column)


def _normalized_data_type(data_type: str) -> str:
    return " ".join(data_type.strip().lower().split())


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


def _pk_params_schema(table: TableProfile) -> dict[str, Any]:
    return _object_schema(
        {
            column_name: _non_null_json_schema_for_column(table.get_column(column_name))
            for column_name in table.primary_key
        },
        required=table.primary_key,
    )


def _empty_params_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def _single_value_params_schema(
    name: str,
    schema: dict[str, Any],
    *,
    description: str,
) -> dict[str, Any]:
    payload = dict(schema)
    payload["description"] = description
    return _object_schema({name: payload}, required=(name,))


def _array_params_schema(
    name: str,
    *,
    items_schema: dict[str, Any],
    max_items: int,
    description: str,
) -> dict[str, Any]:
    return _object_schema(
        {
            name: {
                "type": "array",
                "description": description,
                "items": items_schema,
                "minItems": 1,
                "maxItems": max_items,
            }
        },
        required=(name,),
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


def _scalar_list_schema(
    *,
    items_schema: dict[str, Any],
    max_items: int,
) -> dict[str, Any]:
    return {
        "type": "array",
        "items": items_schema,
        "maxItems": max_items,
    }


def _nullable_json_schema_for_metric(column: ColumnProfile) -> dict[str, Any]:
    schema = _non_null_json_schema_for_column(column)
    schema["type"] = [schema["type"], "null"] if isinstance(schema["type"], str) else [*schema["type"], "null"]
    return schema


def _nullable_json_schema_for_column(column: ColumnProfile) -> dict[str, Any]:
    schema = _non_null_json_schema_for_column(column)
    if isinstance(schema.get("type"), str):
        schema["type"] = [schema["type"], "null"]
    return schema


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


def _select_projection_sql(alias: str, columns: tuple[ColumnProfile, ...]) -> str:
    return ", ".join(
        f"{alias}.{quote_ident(column.column_name)} AS {quote_ident(column.column_name)}"
        for column in columns
    )


def _order_by_sql(alias: str, table: TableProfile, projection_columns: tuple[ColumnProfile, ...]) -> str:
    column_names = list(table.primary_key) or [column.column_name for column in projection_columns]
    return ", ".join(f"{alias}.{quote_ident(column_name)} ASC" for column_name in column_names)


def _pk_predicate_sql(alias: str, table: TableProfile, *, start_index: int) -> str:
    return " AND ".join(
        f"{alias}.{quote_ident(column_name)} = ${index}"
        for index, column_name in enumerate(table.primary_key, start=start_index)
    )


def _edge_join_sql(source_alias: str, target_alias: str, edge: ForeignKeyEdge) -> str:
    return " AND ".join(
        (
            f"{source_alias}.{quote_ident(source_column)} = "
            f"{target_alias}.{quote_ident(target_column)}"
        )
        for source_column, target_column in zip(
            edge.source_columns,
            edge.target_columns,
            strict=True,
        )
    )


def _render_atomic_tool_source(
    *,
    db_id: str,
    tools: list[AtomicToolDefinition],
    max_batch_values: int,
) -> str:
    lines = [
        '"""Schema-derived atomic tools for one database."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        f'DB_ID = {db_id!r}',
        f"MAX_BATCH_VALUES = {max_batch_values}",
        "",
        "def _row_to_dict(row: Any) -> dict[str, Any] | None:",
        "    if row is None:",
        "        return None",
        "    return dict(row)",
        "",
        "def _rows_to_dicts(rows: list[Any]) -> list[dict[str, Any]]:",
        "    return [dict(row) for row in rows]",
        "",
        "def _scalar_list(rows: list[Any], column_name: str) -> list[Any]:",
        "    return [row[column_name] for row in rows]",
        "",
        "def _ensure_bounded_sequence(values: list[Any], *, name: str) -> None:",
        "    if len(values) > MAX_BATCH_VALUES:",
        '        raise ValueError(f"{name} exceeds MAX_BATCH_VALUES")',
        "",
    ]
    for tool in tools:
        lines.extend(_render_tool_function(tool))
    return "\n".join(lines).rstrip() + "\n"


def _render_tool_function(tool: AtomicToolDefinition) -> list[str]:
    param_names = _ordered_param_names(tool.params_schema)
    signature = ", ".join(["conn", *param_names])
    lines = [
        f"async def {tool.name}({signature}):",
        f"    sql = {tool.sql!r}",
    ]
    if any(param_schema.get("type") == "array" for param_schema in tool.params_schema.get("properties", {}).values()):
        for name, schema in tool.params_schema.get("properties", {}).items():
            if schema.get("type") == "array":
                lines.append(f"    _ensure_bounded_sequence({name}, name={name!r})")
    call_args = ", ".join(param_names)
    if tool.result_mode == AtomicToolResultMode.OBJECT_OR_NULL:
        call = f"await conn.fetchrow(sql{', ' if call_args else ''}{call_args})"
        lines.extend(
            [
                f"    row = {call}",
                "    return _row_to_dict(row)",
            ]
        )
    elif tool.result_mode == AtomicToolResultMode.ROW_LIST:
        call = f"await conn.fetch(sql{', ' if call_args else ''}{call_args})"
        lines.extend(
            [
                f"    rows = {call}",
                "    return _rows_to_dicts(rows)",
            ]
        )
    elif tool.result_mode == AtomicToolResultMode.SCALAR_LIST:
        call = f"await conn.fetch(sql{', ' if call_args else ''}{call_args})"
        lines.extend(
            [
                f"    rows = {call}",
                f"    return _scalar_list(rows, {tool.scalar_list_column!r})",
            ]
        )
    else:
        call = f"await conn.fetchval(sql{', ' if call_args else ''}{call_args})"
        lines.append(f"    return {call}")
    lines.append("")
    return lines


def _ordered_param_names(params_schema: dict[str, Any]) -> list[str]:
    properties = params_schema.get("properties", {})
    required = params_schema.get("required", [])
    ordered = [name for name in required if name in properties]
    ordered.extend(name for name in properties if name not in ordered)
    return ordered
