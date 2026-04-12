"""Deterministic db-level atomic tool generation from a schema graph.

This module intentionally duplicates a few tiny helpers that also exist under the
legacy ``rl_task_foundry.tools`` directory. The duplication is temporary and
keeps the atomic-tool transition free of imports from the path-centric stack that
will be deleted in C11.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from rl_task_foundry.config.models import AtomicToolConfig, StrictModel
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile

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
_COMPRESSION_DROP_ORDER = (
    "aggregate",
    "grouped_aggregate",
    "sorted_top_k",
    "like",
    "distinct",
    "range",
    "in",
)


class AtomicToolFamily(StrEnum):
    T1_POINT_LOOKUP = "t1_point_lookup"
    T2_BOUNDED_ENUMERATION = "t2_bounded_enumeration"
    T3_SINGLE_COLUMN_FILTER = "t3_single_column_filter"
    T4_FK_TRAVERSAL = "t4_fk_traversal"
    T5_DISTINCT_VALUES = "t5_distinct_values"
    T6_FILTERED_AGGREGATE = "t6_filtered_aggregate"
    T7_SORTED_TOP_K = "t7_sorted_top_k"
    T8_GROUPED_AGGREGATE_TOP_K = "t8_grouped_aggregate_top_k"


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
    compression_bucket: Literal[
        "core",
        "aggregate",
        "grouped_aggregate",
        "sorted_top_k",
        "like",
        "distinct",
        "range",
        "in",
    ] = "core"
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
                max_bounded_result_limit=self.config.bounded_result_limit,
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
        for filter_column in aggregate_filter_columns:
            tools.append(self._build_filtered_count_tool(table, table_slug, filter_column=filter_column))
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
        if table.primary_key:
            sortable_columns = [column for column in filter_columns if _supports_sorted_top_k(column)]
            for sort_column in sortable_columns:
                tools.extend(
                    self._build_sorted_top_k_tools(
                        table,
                        table_slug,
                        projection_columns,
                        sort_column=sort_column,
                    )
                )
                for filter_column in aggregate_filter_columns:
                    if filter_column.column_name == sort_column.column_name:
                        continue
                    tools.extend(
                        self._build_sorted_top_k_tools(
                            table,
                            table_slug,
                            projection_columns,
                            sort_column=sort_column,
                            filter_column=filter_column,
                        )
                    )
        group_columns = _grouped_aggregate_group_columns(table)
        count_metric_columns = _grouped_aggregate_count_metric_columns(table)
        grouped_aggregate_filter_columns = _grouped_aggregate_filter_columns(table)
        for group_column in group_columns:
            for metric in count_metric_columns:
                tools.extend(
                    self._build_grouped_aggregate_top_k_tools(
                        table,
                        table_slug,
                        group_column=group_column,
                        aggregate_name="count",
                        metric=metric,
                    )
                )
                for filter_column in grouped_aggregate_filter_columns:
                    if filter_column.column_name in {group_column.column_name, metric.column_name}:
                        continue
                    tools.extend(
                        self._build_grouped_aggregate_top_k_tools(
                            table,
                            table_slug,
                            group_column=group_column,
                            aggregate_name="count",
                            metric=metric,
                            filter_column=filter_column,
                        )
                    )
        for group_column in group_columns:
            for metric in metric_columns:
                if metric.column_name == group_column.column_name:
                    continue
                for aggregate_name in ("sum", "avg", "min", "max"):
                    tools.extend(
                        self._build_grouped_aggregate_top_k_tools(
                            table,
                            table_slug,
                            group_column=group_column,
                            aggregate_name=aggregate_name,
                            metric=metric,
                        )
                    )
                    for filter_column in grouped_aggregate_filter_columns:
                        if filter_column.column_name in {group_column.column_name, metric.column_name}:
                            continue
                        tools.extend(
                            self._build_grouped_aggregate_top_k_tools(
                                table,
                                table_slug,
                                group_column=group_column,
                                aggregate_name=aggregate_name,
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
            LIMIT ${len(target_table.primary_key) + 1}
            """
        )
        return [
            AtomicToolDefinition(
                name=f"traverse_{source_slug}_to_{target_slug}_via_{fk_slug}",
                family=AtomicToolFamily.T4_FK_TRAVERSAL,
                description=(
                    f"Get {humanize_identifier(target_slug)} for given "
                    f"{humanize_identifier(source_slug)}."
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
                    f"Get {humanize_identifier(source_slug)} rows for given "
                    f"{humanize_identifier(target_slug)}."
                ),
                params_schema=_with_limit_param(
                    _pk_params_schema(target_table),
                    max_items=self.config.bounded_result_limit,
                ),
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
            description=f"Get {humanize_identifier(table_slug)} for given primary key.",
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
            description=f"Get {humanize_identifier(table_slug)} rows for given primary key values.",
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
            description=f"Get row count for {humanize_identifier(table_slug)}.",
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
            LIMIT $1
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
            description=f"Get primary key ids for {humanize_identifier(table_slug)}.",
            params_schema=_limit_only_params_schema(max_items=self.config.bounded_result_limit),
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
            LIMIT $2
            """
        )
        return AtomicToolDefinition(
            name=f"filter_{table_slug}_by_{column.column_name}_eq",
            family=AtomicToolFamily.T3_SINGLE_COLUMN_FILTER,
            description=(
                f"Get {humanize_identifier(table_slug)} rows for given "
                f"{humanize_identifier(column.column_name)}."
            ),
            params_schema=_with_limit_param(
                _single_value_params_schema(
                    "value",
                    _non_null_json_schema_for_column(column),
                    description=f"Exact match value for {humanize_identifier(column.column_name)}.",
                ),
                max_items=self.config.bounded_result_limit,
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
            LIMIT $2
            """
        )
        return AtomicToolDefinition(
            name=f"filter_{table_slug}_by_{column.column_name}_in",
            family=AtomicToolFamily.T3_SINGLE_COLUMN_FILTER,
            description=(
                f"Get {humanize_identifier(table_slug)} rows for given "
                f"{humanize_identifier(column.column_name)} values."
            ),
            params_schema=_with_limit_param(
                _array_params_schema(
                    "values",
                    items_schema=_non_null_json_schema_for_column(column),
                    max_items=self.config.max_batch_values,
                    description=f"Accepted values for {humanize_identifier(column.column_name)}.",
                ),
                max_items=self.config.bounded_result_limit,
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
            LIMIT $3
            """
        )
        return AtomicToolDefinition(
            name=f"filter_{table_slug}_by_{column.column_name}_range",
            family=AtomicToolFamily.T3_SINGLE_COLUMN_FILTER,
            description=(
                f"Get {humanize_identifier(table_slug)} rows for given "
                f"{humanize_identifier(column.column_name)} range."
            ),
            params_schema=_with_limit_param(
                _object_schema(
                    {
                        "minimum": boundary_schema,
                        "maximum": boundary_schema,
                    },
                    required=(),
                ),
                max_items=self.config.bounded_result_limit,
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
            LIMIT $2
            """
        )
        return AtomicToolDefinition(
            name=f"filter_{table_slug}_by_{column.column_name}_like",
            family=AtomicToolFamily.T3_SINGLE_COLUMN_FILTER,
            description=(
                f"Get {humanize_identifier(table_slug)} rows for given "
                f"{humanize_identifier(column.column_name)} pattern."
            ),
            params_schema=_with_limit_param(
                _single_value_params_schema(
                    "pattern",
                    {"type": "string"},
                    description=(
                        f"Case-insensitive LIKE pattern for "
                        f"{humanize_identifier(column.column_name)}."
                    ),
                ),
                max_items=self.config.bounded_result_limit,
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
            LIMIT $1
            """
        )
        return AtomicToolDefinition(
            name=f"distinct_{table_slug}_{column.column_name}",
            family=AtomicToolFamily.T5_DISTINCT_VALUES,
            description=(
                f"Get distinct {humanize_identifier(column.column_name)} values for "
                f"{humanize_identifier(table_slug)}."
            ),
            params_schema=_limit_only_params_schema(max_items=self.config.bounded_result_limit),
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

    def _build_filtered_count_tool(
        self,
        table: TableProfile,
        table_slug: str,
        *,
        filter_column: ColumnProfile,
    ) -> AtomicToolDefinition:
        sql = readonly_query(
            f"""
            SELECT COUNT(*)::bigint AS value
            FROM {quote_table(table.schema_name, table.table_name)} AS t
            WHERE t.{quote_ident(filter_column.column_name)} = $1
            """
        )
        return AtomicToolDefinition(
            name=f"count_{table_slug}_by_{filter_column.column_name}_eq",
            family=AtomicToolFamily.T6_FILTERED_AGGREGATE,
            description=(
                f"Get count of {humanize_identifier(table_slug)} rows for given "
                f"{humanize_identifier(filter_column.column_name)}."
            ),
            params_schema=_single_value_params_schema(
                "value",
                _non_null_json_schema_for_column(filter_column),
                description=f"Filter value for {humanize_identifier(filter_column.column_name)}.",
            ),
            returns_schema={"type": "integer"},
            sql=sql,
            result_mode=AtomicToolResultMode.SCALAR,
            semantic_key=f"{table.qualified_name}:aggregate:count:by:{filter_column.column_name}",
            compression_bucket="aggregate",
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
        for aggregate_name in ("sum", "avg", "min", "max"):
            aggregate_expression = _aggregate_sql_expression(
                metric,
                aggregate_name=aggregate_name,
                float_precision=self.config.float_precision,
            )
            sql = readonly_query(
                f"""
                SELECT {aggregate_expression} AS value
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
                        f"Get {aggregate_name} {humanize_identifier(metric.column_name)} for "
                        f"{humanize_identifier(table_slug)} rows with given "
                        f"{humanize_identifier(filter_column.column_name)}."
                    ),
                    params_schema=_single_value_params_schema(
                        "value",
                        _non_null_json_schema_for_column(filter_column),
                        description=f"Filter value for {humanize_identifier(filter_column.column_name)}.",
                    ),
                    returns_schema=_aggregate_returns_schema(metric, aggregate_name=aggregate_name),
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

    def _build_sorted_top_k_tools(
        self,
        table: TableProfile,
        table_slug: str,
        projection_columns: tuple[ColumnProfile, ...],
        *,
        sort_column: ColumnProfile,
        filter_column: ColumnProfile | None = None,
    ) -> list[AtomicToolDefinition]:
        tools: list[AtomicToolDefinition] = []
        for direction_slug, direction_sql in (("asc", "ASC"), ("desc", "DESC")):
            predicate_lines = []
            if filter_column is not None:
                predicate_lines.append(
                    f"t.{quote_ident(filter_column.column_name)} = $1"
                )
            predicate_lines.append(f"t.{quote_ident(sort_column.column_name)} IS NOT NULL")
            where_sql = ""
            if predicate_lines:
                where_sql = "WHERE " + "\n                  AND ".join(predicate_lines)
            limit_index = 2 if filter_column is not None else 1
            sql = readonly_query(
                f"""
                SELECT {_select_projection_sql("t", projection_columns)}
                FROM {quote_table(table.schema_name, table.table_name)} AS t
                {where_sql}
                ORDER BY t.{quote_ident(sort_column.column_name)} {direction_sql},
                         {_order_by_sql("t", table, projection_columns)}
                LIMIT ${limit_index}
                """
            )
            if filter_column is None:
                name = f"top_k_{table_slug}_by_{sort_column.column_name}_{direction_slug}"
                description = (
                    f"Get top {humanize_identifier(table_slug)} rows ordered by "
                    f"{humanize_identifier(sort_column.column_name)} "
                    f"{'ascending' if direction_slug == 'asc' else 'descending'}."
                )
                params_schema = _limit_only_params_schema(max_items=self.config.bounded_result_limit)
                semantic_key = (
                    f"{table.qualified_name}:sorted_top_k:{sort_column.column_name}:{direction_slug}:all"
                )
            else:
                name = (
                    f"top_k_{table_slug}_by_{sort_column.column_name}_{direction_slug}"
                    f"_where_{filter_column.column_name}_eq"
                )
                description = (
                    f"Get top {humanize_identifier(table_slug)} rows for given "
                    f"{humanize_identifier(filter_column.column_name)}, ordered by "
                    f"{humanize_identifier(sort_column.column_name)} "
                    f"{'ascending' if direction_slug == 'asc' else 'descending'}."
                )
                params_schema = _with_limit_param(
                    _single_value_params_schema(
                        "value",
                        _non_null_json_schema_for_column(filter_column),
                        description=(
                            f"Filter value for {humanize_identifier(filter_column.column_name)}."
                        ),
                    ),
                    max_items=self.config.bounded_result_limit,
                )
                semantic_key = (
                    f"{table.qualified_name}:sorted_top_k:{sort_column.column_name}:"
                    f"{direction_slug}:by:{filter_column.column_name}"
                )
            tools.append(
                AtomicToolDefinition(
                    name=name,
                    family=AtomicToolFamily.T7_SORTED_TOP_K,
                    description=description,
                    params_schema=params_schema,
                    returns_schema=_row_list_schema(
                        projection_columns,
                        max_items=self.config.bounded_result_limit,
                    ),
                    sql=sql,
                    result_mode=AtomicToolResultMode.ROW_LIST,
                    semantic_key=semantic_key,
                    compression_bucket="sorted_top_k",
                )
            )
        return tools

    def _build_grouped_aggregate_top_k_tools(
        self,
        table: TableProfile,
        table_slug: str,
        *,
        group_column: ColumnProfile,
        aggregate_name: str,
        metric: ColumnProfile,
        filter_column: ColumnProfile | None = None,
    ) -> list[AtomicToolDefinition]:
        tools: list[AtomicToolDefinition] = []
        for direction_slug, direction_sql in (("asc", "ASC"), ("desc", "DESC")):
            aggregate_expression = _grouped_aggregate_sql_expression(
                metric,
                aggregate_name=aggregate_name,
                float_precision=self.config.float_precision,
            )
            predicates = [f"t.{quote_ident(metric.column_name)} IS NOT NULL"]
            if filter_column is not None:
                predicates.insert(0, f"t.{quote_ident(filter_column.column_name)} = $1")
            limit_index = 2 if filter_column is not None else 1
            sql = readonly_query(
                f"""
                SELECT t.{quote_ident(group_column.column_name)} AS {quote_ident(group_column.column_name)},
                       {aggregate_expression} AS value
                FROM {quote_table(table.schema_name, table.table_name)} AS t
                WHERE {" AND ".join(predicates)}
                GROUP BY t.{quote_ident(group_column.column_name)}
                ORDER BY value {direction_sql},
                         t.{quote_ident(group_column.column_name)} ASC
                LIMIT ${limit_index}
                """
            )
            if filter_column is None:
                name = (
                    f"top_k_{table_slug}_grouped_by_{group_column.column_name}_"
                    f"{aggregate_name}_{metric.column_name}_{direction_slug}"
                )
                description = (
                    f"Get top {humanize_identifier(group_column.column_name)} groups for "
                    f"{humanize_identifier(table_slug)}, ordered by "
                    f"{aggregate_name} {humanize_identifier(metric.column_name)} "
                    f"{'ascending' if direction_slug == 'asc' else 'descending'}."
                )
                params_schema = _limit_only_params_schema(max_items=self.config.bounded_result_limit)
                semantic_key = (
                    f"{table.qualified_name}:grouped_aggregate:{aggregate_name}:{metric.column_name}:"
                    f"group_by:{group_column.column_name}:{direction_slug}:all"
                )
            else:
                name = (
                    f"top_k_{table_slug}_grouped_by_{group_column.column_name}_"
                    f"{aggregate_name}_{metric.column_name}_{direction_slug}"
                    f"_where_{filter_column.column_name}_eq"
                )
                description = (
                    f"Get top {humanize_identifier(group_column.column_name)} groups for "
                    f"{humanize_identifier(table_slug)} rows with given "
                    f"{humanize_identifier(filter_column.column_name)}, ordered by "
                    f"{aggregate_name} {humanize_identifier(metric.column_name)} "
                    f"{'ascending' if direction_slug == 'asc' else 'descending'}."
                )
                params_schema = _with_limit_param(
                    _single_value_params_schema(
                        "value",
                        _non_null_json_schema_for_column(filter_column),
                        description=(
                            f"Filter value for {humanize_identifier(filter_column.column_name)}."
                        ),
                    ),
                    max_items=self.config.bounded_result_limit,
                )
                semantic_key = (
                    f"{table.qualified_name}:grouped_aggregate:{aggregate_name}:{metric.column_name}:"
                    f"group_by:{group_column.column_name}:{direction_slug}:by:{filter_column.column_name}"
                )
            tools.append(
                AtomicToolDefinition(
                    name=name,
                    family=AtomicToolFamily.T8_GROUPED_AGGREGATE_TOP_K,
                    description=description,
                    params_schema=params_schema,
                    returns_schema=_grouped_aggregate_row_list_schema(
                        group_column,
                        aggregate_name=aggregate_name,
                        metric=metric,
                        max_items=self.config.bounded_result_limit,
                    ),
                    sql=sql,
                    result_mode=AtomicToolResultMode.ROW_LIST,
                    semantic_key=semantic_key,
                    compression_bucket="grouped_aggregate",
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
        AtomicToolFamily.T7_SORTED_TOP_K: 6,
        AtomicToolFamily.T8_GROUPED_AGGREGATE_TOP_K: 7,
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


def _grouped_aggregate_group_columns(table: TableProfile) -> list[ColumnProfile]:
    columns: list[ColumnProfile] = []
    for column in table.columns:
        if not _supports_grouped_aggregate_group(column):
            continue
        if column.is_foreign_key or column.visibility == "user_visible":
            columns.append(column)
    return columns


def _grouped_aggregate_count_metric_columns(table: TableProfile) -> list[ColumnProfile]:
    if table.primary_key:
        primary_key_column = table.get_column(table.primary_key[0])
        if _is_scalar(primary_key_column):
            return [primary_key_column]
    return _grouped_aggregate_group_columns(table)[:1]


def _grouped_aggregate_filter_columns(table: TableProfile) -> list[ColumnProfile]:
    return [
        column
        for column in _filter_columns(table)
        if _supports_grouped_aggregate_filter(column)
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


def _supports_sorted_top_k(column: ColumnProfile) -> bool:
    return _normalized_data_type(column.data_type) in (
        _INTEGER_TYPES | _FLOAT_TYPES | _DATE_TYPES | _DATETIME_TYPES
    )


def _supports_grouped_aggregate_group(column: ColumnProfile) -> bool:
    normalized = _normalized_data_type(column.data_type)
    if column.is_foreign_key:
        return _is_scalar(column)
    if column.visibility != "user_visible":
        return False
    return normalized in (_TEXT_TYPES | _BOOLEAN_TYPES)


def _supports_grouped_aggregate_filter(column: ColumnProfile) -> bool:
    return _normalized_data_type(column.data_type) in (
        _TEXT_TYPES | _BOOLEAN_TYPES | _DATE_TYPES | _DATETIME_TYPES
    )


def _is_non_integer_numeric(column: ColumnProfile) -> bool:
    return _normalized_data_type(column.data_type) in _FLOAT_TYPES


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


def _limit_only_params_schema(*, max_items: int) -> dict[str, Any]:
    return _object_schema({"limit": _limit_param_schema(max_items=max_items)}, required=("limit",))


def _with_limit_param(params_schema: dict[str, Any], *, max_items: int) -> dict[str, Any]:
    properties = dict(params_schema.get("properties", {}))
    properties["limit"] = _limit_param_schema(max_items=max_items)
    required = [*params_schema.get("required", [])]
    if "limit" not in required:
        required.append("limit")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": params_schema.get("additionalProperties", False),
    }


def _limit_param_schema(*, max_items: int) -> dict[str, Any]:
    return {
        "type": "integer",
        "minimum": 1,
        "description": (
            "Maximum rows to return. Values above "
            f"{max_items} are capped by the runtime."
        ),
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


def _grouped_aggregate_row_list_schema(
    group_column: ColumnProfile,
    *,
    aggregate_name: str,
    metric: ColumnProfile,
    max_items: int,
) -> dict[str, Any]:
    required = ["value"]
    if not group_column.is_nullable:
        required.insert(0, group_column.column_name)
    return {
        "type": "array",
        "items": _object_schema(
            {
                group_column.column_name: _json_schema_for_column(group_column),
                "value": _grouped_aggregate_value_schema(
                    metric,
                    aggregate_name=aggregate_name,
                ),
            },
            required=tuple(required),
        ),
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


def _aggregate_returns_schema(
    column: ColumnProfile,
    *,
    aggregate_name: str,
) -> dict[str, Any]:
    if aggregate_name == "avg":
        return {"type": ["number", "null"]}
    if aggregate_name == "sum" and _is_non_integer_numeric(column):
        return {"type": ["number", "null"]}
    return _nullable_json_schema_for_metric(column)


def _grouped_aggregate_value_schema(
    column: ColumnProfile,
    *,
    aggregate_name: str,
) -> dict[str, Any]:
    if aggregate_name == "count":
        return {"type": "integer"}
    if aggregate_name == "avg":
        return {"type": "number"}
    if aggregate_name == "sum" and _is_non_integer_numeric(column):
        return {"type": "number"}
    return _non_null_json_schema_for_column(column)


def _aggregate_sql_expression(
    column: ColumnProfile,
    *,
    aggregate_name: str,
    float_precision: int,
) -> str:
    column_sql = f"t.{quote_ident(column.column_name)}"
    base = f"{aggregate_name.upper()}({column_sql})"
    if aggregate_name == "avg":
        return f"ROUND({base}::numeric, {float_precision})"
    if aggregate_name == "sum" and _is_non_integer_numeric(column):
        return f"ROUND({base}::numeric, {float_precision})"
    return base


def _grouped_aggregate_sql_expression(
    column: ColumnProfile,
    *,
    aggregate_name: str,
    float_precision: int,
) -> str:
    column_sql = f"t.{quote_ident(column.column_name)}"
    if aggregate_name == "count":
        return f"COUNT({column_sql})::bigint"
    return _aggregate_sql_expression(
        column,
        aggregate_name=aggregate_name,
        float_precision=float_precision,
    )


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
    max_bounded_result_limit: int,
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
        f"MAX_BOUNDED_RESULT_LIMIT = {max_bounded_result_limit}",
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
        "def _bounded_limit(limit: int) -> int:",
        "    if isinstance(limit, bool) or not isinstance(limit, int):",
        '        raise TypeError("limit must be an integer")',
        "    if limit < 1:",
        '        raise ValueError("limit must be >= 1")',
        "    return min(limit, MAX_BOUNDED_RESULT_LIMIT)",
        "",
    ]
    for tool in tools:
        lines.extend(_render_tool_function(tool))
    return "\n".join(lines).rstrip() + "\n"


def _render_tool_function(tool: AtomicToolDefinition) -> list[str]:
    signature_parts = _signature_param_parts(tool.params_schema)
    sql_param_names = _sql_param_names(tool.params_schema)
    signature = ", ".join(["conn", *signature_parts])
    lines = [
        f"async def {tool.name}({signature}):",
        f"    sql = {tool.sql!r}",
    ]
    if any(param_schema.get("type") == "array" for param_schema in tool.params_schema.get("properties", {}).values()):
        for name, schema in tool.params_schema.get("properties", {}).items():
            if schema.get("type") == "array":
                lines.append(f"    _ensure_bounded_sequence({name}, name={name!r})")
    if "limit" in tool.params_schema.get("properties", {}):
        lines.append("    limit = _bounded_limit(limit)")
    call_args = ", ".join(sql_param_names)
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
    return _sql_param_names(params_schema)


def _sql_param_names(params_schema: dict[str, Any]) -> list[str]:
    return list(params_schema.get("properties", {}).keys())


def _signature_param_parts(params_schema: dict[str, Any]) -> list[str]:
    properties = params_schema.get("properties", {})
    required = set(params_schema.get("required", []))
    signature_parts = [name for name in properties if name in required]
    signature_parts.extend(f"{name}=None" for name in properties if name not in required)
    return signature_parts


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
