"""Rule-based DB affordance summaries for composer context.

The affordance map is not a task oracle. It compresses schema/profile
observations into small cards so the composer can decide what to inspect next.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.schema.profiler import DataProfile

_NUMERIC_TYPES = frozenset({
    "integer",
    "bigint",
    "smallint",
    "int2",
    "int4",
    "int8",
    "numeric",
    "decimal",
    "real",
    "double precision",
    "float4",
    "float8",
    "money",
})

_TIME_TYPES = frozenset({
    "date",
    "time",
    "time without time zone",
    "time with time zone",
    "timetz",
    "timestamp",
    "timestamp without time zone",
    "timestamp with time zone",
    "timestamptz",
})


def build_db_affordance_map(
    graph: SchemaGraph,
    *,
    data_profile: DataProfile | None = None,
) -> dict[str, object]:
    """Build a complete DB-adaptive context artifact for composer.

    The map must not hide or prioritize tables or relationships. Composer uses
    it as a navigation map, so order follows the schema graph order.
    """

    numeric_profile = _profile_columns(data_profile.numeric if data_profile else ())
    categorical_profile = _profile_columns(
        data_profile.categorical if data_profile else ()
    )
    table_cards = [
        _table_card(
            graph,
            table,
            numeric_profile=numeric_profile,
            categorical_profile=categorical_profile,
        )
        for table in graph.tables
    ]

    path_cards = [
        _path_card(
            graph,
            edge,
            numeric_profile=numeric_profile,
            categorical_profile=categorical_profile,
        )
        for edge in graph.edges
    ]

    return {
        "table_count": len(table_cards),
        "path_count": len(path_cards),
        "table_affordances": table_cards,
        "path_affordances": path_cards,
    }


def _profile_columns(stats: Iterable[object]) -> dict[str, list[str]]:
    by_table: dict[str, list[str]] = defaultdict(list)
    for stat in stats:
        table = getattr(stat, "table", None)
        column = getattr(stat, "column", None)
        if isinstance(table, str) and isinstance(column, str):
            by_table[table].append(column)
    return {table: sorted(set(columns)) for table, columns in by_table.items()}


def _visible_non_key_columns(table: TableProfile) -> list[ColumnProfile]:
    return [
        column
        for column in table.columns
        if column.visibility == "user_visible"
        and not column.is_primary_key
        and not column.is_foreign_key
    ]


def _column_names(columns: list[ColumnProfile], *, limit: int = 5) -> list[str]:
    return [column.column_name for column in columns[:limit]]


def _is_time_column(column: ColumnProfile) -> bool:
    return column.data_type.strip().lower() in _TIME_TYPES


def _is_numeric_column(column: ColumnProfile) -> bool:
    return column.data_type.lower() in _NUMERIC_TYPES


def _is_measure_column(column: ColumnProfile) -> bool:
    return _is_numeric_column(column)


def _table_structure(
    graph: SchemaGraph,
    table: TableProfile,
    *,
    readable: list[ColumnProfile],
) -> str:
    inbound = graph.edges_to(table.table_name, schema_name=table.schema_name)
    outbound = graph.edges_from(table.table_name, schema_name=table.schema_name)
    fk_count = sum(1 for column in table.columns if column.is_foreign_key)
    if not readable and fk_count >= 2:
        return "bridge"
    if len(inbound) >= 2:
        return "hub"
    if inbound and outbound:
        return "connected"
    if inbound:
        return "referenced"
    if outbound:
        return "referrer"
    if readable:
        return "surface"
    return "structural"


def _table_card(
    graph: SchemaGraph,
    table: TableProfile,
    *,
    numeric_profile: dict[str, list[str]],
    categorical_profile: dict[str, list[str]],
) -> dict[str, object]:
    qualified = table.qualified_name
    readable = _visible_non_key_columns(table)
    readable_names = {column.column_name for column in readable}
    time_columns = [column for column in readable if _is_time_column(column)]
    categorical_profile_names = set(categorical_profile.get(qualified, [])) & readable_names
    numeric_columns = sorted(
        (
            {
                column.column_name
                for column in readable
                if _is_measure_column(column)
            }
            | {
                column.column_name
                for column in readable
                if column.column_name in numeric_profile.get(qualified, [])
                and _is_measure_column(column)
            }
        )
        - categorical_profile_names
    )
    categorical_columns = sorted(
        {
            column.column_name
            for column in readable
            if not _is_measure_column(column) and not _is_time_column(column)
        }
        | categorical_profile_names
    )
    inbound = graph.edges_to(table.table_name, schema_name=table.schema_name)
    outbound = graph.edges_from(table.table_name, schema_name=table.schema_name)
    structure = _table_structure(
        graph,
        table,
        readable=readable,
    )
    affordances: list[str] = []
    if readable:
        affordances.append("label_surface")
    if categorical_columns or time_columns:
        affordances.append("filter_surface")
    if numeric_columns or time_columns:
        affordances.append("aggregate_surface")
    if inbound:
        affordances.append("anchor_candidate")
    return {
        "table": qualified,
        "structure": structure,
        "row_estimate": table.row_estimate,
        "readable": _column_names(readable),
        "categorical_filters": categorical_columns[:5],
        "numeric_metrics": numeric_columns[:5],
        "time_columns": _column_names(time_columns),
        "incoming_paths": len(inbound),
        "outgoing_refs": len(outbound),
        "affordances": affordances,
    }


def _edge_qualified_path(edge: ForeignKeyEdge) -> str:
    return (
        f"{edge.target_qualified_name} -> {edge.source_qualified_name} "
        f"via {_edge_relation_label(edge)}"
    )


def _edge_relation_label(edge: ForeignKeyEdge) -> str:
    source = ", ".join(edge.source_columns)
    target = ", ".join(edge.target_columns)
    return f"{edge.source_table}.{source} = {edge.target_table}.{target}"


def _path_card(
    graph: SchemaGraph,
    edge: ForeignKeyEdge,
    *,
    numeric_profile: dict[str, list[str]],
    categorical_profile: dict[str, list[str]],
) -> dict[str, object]:
    target = graph.get_table(edge.target_table, schema_name=edge.target_schema)
    source = graph.get_table(edge.source_table, schema_name=edge.source_schema)
    source_readable = _visible_non_key_columns(source)
    source_readable_names = {column.column_name for column in source_readable}
    time_columns = [column for column in source_readable if _is_time_column(column)]
    categorical_profile_names = (
        set(categorical_profile.get(source.qualified_name, []))
        & source_readable_names
    )
    numeric_columns = sorted(
        (
            {
                column.column_name
                for column in source_readable
                if _is_measure_column(column)
            }
            | {
                column.column_name
                for column in source_readable
                if column.column_name in numeric_profile.get(source.qualified_name, [])
                and _is_measure_column(column)
            }
        )
        - categorical_profile_names
    )
    categorical_columns = sorted(
        {
            column.column_name
            for column in source_readable
            if not _is_measure_column(column) and not _is_time_column(column)
        }
        | categorical_profile_names
    )
    supports: list[str] = []
    if source_readable:
        supports.append("ordered_list")
    if edge.fanout_estimate is not None and edge.fanout_estimate > 1.2:
        supports.append("count_or_cardinality")
    if numeric_columns:
        supports.append("numeric_aggregate")
    if time_columns:
        supports.append("timeline_or_time_filter")
    if categorical_columns:
        supports.append("categorical_filter")
    return {
        "path": _edge_qualified_path(edge),
        "relation": _edge_relation_label(edge),
        "anchor_table": target.qualified_name,
        "target_table": source.qualified_name,
        "fanout": round(edge.fanout_estimate, 2) if edge.fanout_estimate else None,
        "readable": _column_names(source_readable),
        "filters": (categorical_columns + _column_names(time_columns))[:5],
        "metrics": numeric_columns[:5],
        "supports": supports,
    }


__all__ = ["build_db_affordance_map"]
