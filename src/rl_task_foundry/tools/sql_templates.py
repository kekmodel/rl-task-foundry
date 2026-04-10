"""Read-only SQL compilation helpers."""

from __future__ import annotations

from rl_task_foundry.schema.graph import SchemaGraph
from rl_task_foundry.schema.path_catalog import PathSpec

DEFAULT_FLOAT_PRECISION = 6
_ROUNDABLE_AGGREGATE_DATA_TYPES = {
    "numeric",
    "decimal",
    "float4",
    "float8",
    "real",
    "double precision",
}


def readonly_query(sql: str) -> str:
    """Normalize a compiled SQL string."""

    return " ".join(sql.split())


def compile_lookup_sql(
    graph: SchemaGraph,
    path: PathSpec,
    *,
    output_fields: list[str],
) -> str:
    """Compile a deterministic read-only lookup query for a path."""

    if not output_fields:
        raise ValueError("lookup tool requires at least one output field")
    root_table = _root_table_for_path(graph, path)
    target_table = graph.get_table(path.edges[-1].target_table, schema_name=path.edges[-1].target_schema)
    target_alias = _alias_for_index(path.hop_count)
    select_expressions = [
        (f'{target_alias}.{quote_ident(field)}', field)
        for field in output_fields
    ]
    return _compile_distinct_projection_query(
        select_expressions=select_expressions,
        from_and_joins_sql=_render_from_and_joins(path),
        predicate_sql=_render_anchor_predicate(root_table),
        order_specs=_target_order_specs(target_alias, target_table, output_fields),
    )


def compile_count_sql(graph: SchemaGraph, path: PathSpec) -> str:
    """Compile a read-only count query for a path."""

    root_table = _root_table_for_path(graph, path)
    target_alias = _alias_for_index(path.hop_count)
    target_table = graph.get_table(path.edges[-1].target_table, schema_name=path.edges[-1].target_schema)
    distinct_target_sql = _distinct_target_value_expression(target_alias, target_table)
    return readonly_query(
        f"""
        SELECT COUNT(DISTINCT {distinct_target_sql})::bigint AS count
        {_render_from_and_joins(path)}
        WHERE {_render_anchor_predicate(root_table)}
          AND {target_alias}.{quote_ident(path.edges[-1].target_columns[0])} IS NOT NULL
        """
    )


def compile_list_related_sql(
    graph: SchemaGraph,
    path: PathSpec,
    *,
    output_fields: list[str],
    limit: int,
) -> str:
    """Compile a deterministic bounded list query for a 1:N relation path."""

    if not output_fields:
        raise ValueError("list_related tool requires at least one output field")
    if limit <= 0:
        raise ValueError("list_related tool requires a positive limit")

    root_table = _root_table_for_path(graph, path)
    target_table = graph.get_table(path.edges[-1].target_table, schema_name=path.edges[-1].target_schema)
    target_alias = _alias_for_index(path.hop_count)
    select_expressions = [
        (f'{target_alias}.{quote_ident(field)}', field)
        for field in output_fields
    ]
    return _compile_distinct_projection_query(
        select_expressions=select_expressions,
        from_and_joins_sql=_render_from_and_joins(path),
        predicate_sql=_render_anchor_predicate(root_table),
        order_specs=_target_order_specs(target_alias, target_table, output_fields),
        limit=limit,
    )


def compile_aggregate_sql(
    graph: SchemaGraph,
    path: PathSpec,
    *,
    aggregate_function: str,
    aggregate_column: str,
    output_alias: str,
    float_precision: int = DEFAULT_FLOAT_PRECISION,
) -> str:
    """Compile a deterministic aggregate query for a path."""

    if aggregate_function not in {"sum", "avg", "min", "max"}:
        raise ValueError(f"unsupported aggregate function: {aggregate_function}")
    root_table = _root_table_for_path(graph, path)
    target_table = graph.get_table(path.edges[-1].target_table, schema_name=path.edges[-1].target_schema)
    target_alias = _alias_for_index(path.hop_count)
    target_column = target_table.get_column(aggregate_column)
    column_sql = f"{target_alias}.{quote_ident(aggregate_column)}"
    aggregate_sql = f"{aggregate_function.upper()}({column_sql})"
    projection_sql = aggregate_sql
    if aggregate_function in {"sum", "avg"} and _should_round_aggregate(target_column.data_type):
        projection_sql = f"ROUND(({aggregate_sql})::numeric, {int(float_precision)})"
    return readonly_query(
        f"""
        SELECT {projection_sql} AS {quote_ident(output_alias)}
        {_render_from_and_joins(path)}
        WHERE {_render_anchor_predicate(root_table)}
          AND {column_sql} IS NOT NULL
        """
    )


def compile_timeline_sql(
    graph: SchemaGraph,
    path: PathSpec,
    *,
    output_fields: list[str],
    order_column: str,
    limit: int,
) -> str:
    """Compile a deterministic bounded timeline query for a path."""

    if not output_fields:
        raise ValueError("timeline tool requires at least one output field")
    if limit <= 0:
        raise ValueError("timeline tool requires a positive limit")

    root_table = _root_table_for_path(graph, path)
    target_table = graph.get_table(path.edges[-1].target_table, schema_name=path.edges[-1].target_schema)
    target_alias = _alias_for_index(path.hop_count)
    select_expressions = [
        (f'{target_alias}.{quote_ident(field)}', field)
        for field in output_fields
    ]
    return _compile_distinct_projection_query(
        select_expressions=select_expressions,
        from_and_joins_sql=_render_from_and_joins(path),
        predicate_sql=(
            f"{_render_anchor_predicate(root_table)} "
            f'AND {target_alias}.{quote_ident(order_column)} IS NOT NULL'
        ),
        order_specs=_timeline_order_specs(target_alias, target_table, order_column),
        limit=limit,
    )


def compile_exists_sql(graph: SchemaGraph, path: PathSpec) -> str:
    """Compile a read-only existence query for a path."""

    root_table = _root_table_for_path(graph, path)
    target_alias = _alias_for_index(path.hop_count)
    return readonly_query(
        f"""
        SELECT EXISTS(
          SELECT 1
          {_render_from_and_joins(path)}
          WHERE {_render_anchor_predicate(root_table)}
            AND {target_alias}.{quote_ident(path.edges[-1].target_columns[0])} IS NOT NULL
        ) AS exists
        """
    )


def compile_anchor_parameters(graph: SchemaGraph, path: PathSpec) -> tuple[str, ...]:
    """Return root PK-derived parameter names for a path."""

    root_table = _root_table_for_path(graph, path)
    if not root_table.primary_key:
        raise ValueError(f"root table has no primary key: {root_table.qualified_name}")
    return tuple(f"anchor_{column_name}" for column_name in root_table.primary_key)


def _root_table_for_path(graph: SchemaGraph, path: PathSpec):
    source_schema = path.edges[0].source_schema
    return graph.get_table(path.root_table, schema_name=source_schema)


def _render_from_and_joins(path: PathSpec) -> str:
    root_schema = path.edges[0].source_schema
    root_alias = _alias_for_index(0)
    chunks = [f'FROM {quote_table(root_schema, path.root_table)} AS {root_alias}']
    for index, edge in enumerate(path.edges, start=1):
        source_alias = _alias_for_index(index - 1)
        target_alias = _alias_for_index(index)
        predicates = " AND ".join(
            f"{source_alias}.{quote_ident(source_column)} = {target_alias}.{quote_ident(target_column)}"
            for source_column, target_column in zip(edge.source_columns, edge.target_columns, strict=True)
        )
        chunks.append(
            f'JOIN {quote_table(edge.target_schema, edge.target_table)} AS {target_alias} ON {predicates}'
        )
    return " ".join(chunks)


def _render_anchor_predicate(root_table) -> str:
    root_alias = _alias_for_index(0)
    return " AND ".join(
        f"{root_alias}.{quote_ident(column_name)} = :anchor_{column_name}"
        for column_name in root_table.primary_key
    )


def _alias_for_index(index: int) -> str:
    return f"t{index}"


def _render_target_order_clause(target_alias: str, target_table, output_fields: list[str]) -> str:
    specs = _target_order_specs(target_alias, target_table, output_fields)
    rendered = ", ".join(f"{expression} {direction}" for expression, direction in specs)
    return f"ORDER BY {rendered}"


def _render_timeline_order_clause(target_alias: str, target_table, order_column: str) -> str:
    specs = _timeline_order_specs(target_alias, target_table, order_column)
    return "ORDER BY " + ", ".join(f"{expression} {direction}" for expression, direction in specs)


def _compile_distinct_projection_query(
    *,
    select_expressions: list[tuple[str, str]],
    from_and_joins_sql: str,
    predicate_sql: str,
    order_specs: list[tuple[str, str]],
    limit: int | None = None,
) -> str:
    if not select_expressions:
        raise ValueError("distinct projection query requires at least one selected field")
    if not order_specs:
        raise ValueError("distinct projection query requires at least one ordering expression")

    distinct_on_clause = ", ".join(expression for expression, _alias in select_expressions)
    visible_select_clause = ", ".join(
        f"{expression} AS {quote_ident(alias)}"
        for expression, alias in select_expressions
    )
    order_select_clause = ", ".join(
        f"{expression} AS {quote_ident(f'__ord_{index}')}"
        for index, (expression, _direction) in enumerate(order_specs)
    )
    inner_select_clause = ", ".join(
        [visible_select_clause, order_select_clause] if order_select_clause else [visible_select_clause]
    )
    inner_order_clause = ", ".join(
        [distinct_on_clause, *[f"{expression} {direction}" for expression, direction in order_specs]]
    )
    outer_select_clause = ", ".join(f'base.{quote_ident(alias)}' for _expression, alias in select_expressions)
    outer_order_clause = ", ".join(
        f'base.{quote_ident(f"__ord_{index}")} {direction}'
        for index, (_expression, direction) in enumerate(order_specs)
    )
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
    return readonly_query(
        f"""
        SELECT {outer_select_clause}
        FROM (
          SELECT DISTINCT ON ({distinct_on_clause}) {inner_select_clause}
          {from_and_joins_sql}
          WHERE {predicate_sql}
          ORDER BY {inner_order_clause}
        ) AS base
        ORDER BY {outer_order_clause}
        {limit_clause}
        """
    )


def _target_order_specs(target_alias: str, target_table, output_fields: list[str]) -> list[tuple[str, str]]:
    order_columns = list(target_table.primary_key) if target_table.primary_key else list(output_fields)
    if not order_columns:
        raise ValueError("cannot build deterministic ordering without primary key or output fields")
    return [
        (f"{target_alias}.{quote_ident(column_name)}", "ASC")
        for column_name in order_columns
    ]


def _timeline_order_specs(target_alias: str, target_table, order_column: str) -> list[tuple[str, str]]:
    tiebreakers = [column_name for column_name in target_table.primary_key if column_name != order_column]
    return [
        (f"{target_alias}.{quote_ident(order_column)}", "DESC"),
        *[
            (f"{target_alias}.{quote_ident(column_name)}", "ASC")
            for column_name in tiebreakers
        ],
    ]


def _should_round_aggregate(data_type: str) -> bool:
    return data_type.lower() in _ROUNDABLE_AGGREGATE_DATA_TYPES


def _distinct_target_value_expression(target_alias: str, target_table) -> str:
    if target_table.primary_key:
        if len(target_table.primary_key) == 1:
            return f"{target_alias}.{quote_ident(target_table.primary_key[0])}"
        columns = ", ".join(
            f"{target_alias}.{quote_ident(column_name)}"
            for column_name in target_table.primary_key
        )
        return f"({columns})"
    first_column = target_table.columns[0].column_name
    return f"{target_alias}.{quote_ident(first_column)}"


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def quote_table(schema_name: str, table_name: str) -> str:
    return f"{quote_ident(schema_name)}.{quote_ident(table_name)}"
