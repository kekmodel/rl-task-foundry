"""Deterministic ground truth generator interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Literal

import asyncpg

from rl_task_foundry.config.models import DatabaseConfig
from rl_task_foundry.infra.db import control_session_settings
from rl_task_foundry.schema.graph import ColumnProfile, SchemaGraph
from rl_task_foundry.schema.path_catalog import PathCatalog, PathSpec
from rl_task_foundry.tasks.models import TaskSpec
from rl_task_foundry.tools.sql_templates import (
    _alias_for_index,
    _compile_distinct_projection_query,
    _render_anchor_predicate,
    _render_from_and_joins,
    _target_order_specs,
    _timeline_order_specs,
    compile_count_sql,
    compile_exists_sql,
    compile_reverse_count_sql,
    quote_ident,
)
from rl_task_foundry.truth.canonicalize import canonicalize_answer
from rl_task_foundry.truth.schemas import AnswerField
from rl_task_foundry.truth.schemas import GroundTruth


class GroundTruthGenerator:
    """Compile deterministic SQL and canonical answers for a task."""

    async def generate(self, task: TaskSpec) -> GroundTruth:
        raise NotImplementedError


Strategy = Literal["direct", "list", "count", "exists", "latest"]

_NAMED_PARAM_PATTERN = re.compile(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)")
_INT_DATA_TYPES = {
    "int2",
    "int4",
    "int8",
    "smallint",
    "integer",
    "bigint",
    "serial",
    "serial2",
    "serial4",
    "serial8",
    "bigserial",
}
_FLOAT_DATA_TYPES = {
    "numeric",
    "decimal",
    "float4",
    "float8",
    "real",
    "double precision",
}
_BOOL_DATA_TYPES = {"bool", "boolean"}
_DATE_DATA_TYPES = {"date"}
_DATETIME_DATA_TYPES = {
    "timestamp",
    "timestamp without time zone",
    "timestamp with time zone",
    "timestamptz",
}


@dataclass(slots=True)
class Projection:
    answer_field: AnswerField
    source_column: ColumnProfile
    expression_sql: str


@dataclass(slots=True)
class QueryPlan:
    strategy: Strategy
    verification_sql: str
    projections: list[Projection]


@dataclass(slots=True)
class TierAGroundTruthGenerator(GroundTruthGenerator):
    database: DatabaseConfig
    graph: SchemaGraph
    catalog: PathCatalog
    float_precision: int = 6

    async def generate(self, task: TaskSpec) -> GroundTruth:
        if task.label_tier != "A":
            raise NotImplementedError("Tier A ground truth generator only supports label_tier='A'")
        if task.outcome_type not in {"answer", "no_result"}:
            raise NotImplementedError(
                "Tier A ground truth generator currently supports only answer/no_result branches"
            )
        if not task.answer_schema.fields:
            raise ValueError("answer_schema must define at least one field")

        path = self.catalog.get(task.selected_path_id)
        plan = self._build_plan(task, path)
        sql_params = self._sql_params(task, path)
        rows = await self._fetch_rows(plan.verification_sql, sql_params)
        canonical_answer = self._canonical_answer(task, plan, rows)
        return GroundTruth(
            task_id=task.task_id,
            verification_sql=plan.verification_sql,
            sql_params=sql_params,
            canonical_answer=canonical_answer,
            row_context=rows,
            answer_schema_version=task.answer_schema.version,
            provenance_path=list(path.tables),
        )

    async def _fetch_rows(
        self,
        verification_sql: str,
        sql_params: dict[str, object],
    ) -> list[dict[str, object]]:
        prepared_sql, prepared_args = _prepare_asyncpg_query(verification_sql, sql_params)
        conn = await asyncpg.connect(dsn=self.database.dsn)
        try:
            settings = control_session_settings(self.database)
            for statement in settings.timeout_sql:
                await conn.execute(statement)
            rows = await conn.fetch(prepared_sql, *prepared_args)
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    def _build_plan(self, task: TaskSpec, path: PathSpec) -> QueryPlan:
        fields = task.answer_schema.fields
        field_markers = [_field_marker(field) for field in fields]
        if any(marker is not None for marker in field_markers) and not all(
            marker in {"meta:count", "meta:exists"} for marker in field_markers
        ):
            raise ValueError("meta answer fields cannot be mixed with direct projection fields")
        if any(marker == "meta:count" for marker in field_markers):
            return self._build_count_plan(task, path)
        if any(marker == "meta:exists" for marker in field_markers):
            return self._build_exists_plan(task, path)
        if any(field.type.startswith("list[") for field in fields) and not all(
            field.type.startswith("list[") for field in fields
        ):
            raise ValueError("Tier A list answers cannot mix list and scalar fields")
        if all(field.type.startswith("list[") for field in fields):
            projections = [self._build_projection(field, path) for field in fields]
            return QueryPlan(
                strategy="list",
                verification_sql=_compile_projection_sql(
                    self.graph,
                    path,
                    projections=projections,
                    distinct=True,
                    limit=None,
                ),
                projections=projections,
            )
        projections = [self._build_projection(field, path) for field in fields]
        if task.question_family == "timeline_resolution":
            return self._build_latest_plan(task, path, projections)
        return QueryPlan(
            strategy="direct",
            verification_sql=_compile_projection_sql(
                self.graph,
                path,
                projections=projections,
                distinct=True,
                limit=None,
            ),
            projections=projections,
        )

    def _build_count_plan(self, task: TaskSpec, path: PathSpec) -> QueryPlan:
        if len(task.answer_schema.fields) != 1:
            raise ValueError("count ground truth requires exactly one answer field")
        field = task.answer_schema.fields[0]
        if field.type != "int":
            raise ValueError("count ground truth field must have type 'int'")
        if task.contract_metadata.get("count_mode") == "reverse_relation":
            relation_edge = self._resolve_reverse_count_relation(task, path)
            return QueryPlan(
                strategy="count",
                verification_sql=compile_reverse_count_sql(
                    self.graph,
                    path,
                    relation_edge=relation_edge,
                ),
                projections=[],
            )
        return QueryPlan(
            strategy="count",
            verification_sql=compile_count_sql(self.graph, path),
            projections=[],
        )

    def _build_exists_plan(self, task: TaskSpec, path: PathSpec) -> QueryPlan:
        if len(task.answer_schema.fields) != 1:
            raise ValueError("exists ground truth requires exactly one answer field")
        field = task.answer_schema.fields[0]
        if field.type != "bool":
            raise ValueError("exists ground truth field must have type 'bool'")
        return QueryPlan(
            strategy="exists",
            verification_sql=compile_exists_sql(self.graph, path),
            projections=[],
        )

    def _build_latest_plan(
        self,
        task: TaskSpec,
        path: PathSpec,
        projections: list[Projection],
    ) -> QueryPlan:
        visible_fields = _visible_answer_fields(task)
        if len(visible_fields) != 1:
            raise ValueError("timeline ground truth currently requires exactly one visible field")
        field = visible_fields[0]
        if field.type not in {"date", "datetime"}:
            raise ValueError("timeline ground truth requires a date/datetime visible field")
        if len(projections) != 1:
            raise ValueError("timeline ground truth currently requires exactly one projection")
        target_table = self.graph.get_table(
            path.edges[-1].target_table,
            schema_name=path.edges[-1].target_schema,
        )
        target_alias = _alias_for_index(path.hop_count)
        order_column = projections[0].source_column.column_name
        return QueryPlan(
            strategy="latest",
            verification_sql=_compile_projection_sql(
                self.graph,
                path,
                projections=projections,
                distinct=False,
                limit=1,
                order_specs=_timeline_order_specs(
                    target_alias,
                    target_table,
                    order_column,
                ),
            ),
            projections=projections,
        )

    def _build_projection(self, field: AnswerField, path: PathSpec) -> Projection:
        marker = _field_marker(field)
        if marker in {"meta:count", "meta:exists"}:
            raise ValueError("meta count/exists fields must use dedicated plans")
        column = _resolve_source_column(self.graph, path, field)
        alias = _alias_for_table(path, column.table_name, column.schema_name)
        return Projection(
            answer_field=field,
            source_column=column,
            expression_sql=_projection_expression(
                field,
                alias=alias,
                column=column,
                float_precision=self.float_precision,
            ),
        )

    def _canonical_answer(
        self,
        task: TaskSpec,
        plan: QueryPlan,
        rows: list[dict[str, object]],
    ) -> dict[str, object]:
        visible_fields = _visible_answer_fields(task)
        if plan.strategy == "count":
            visible_field = _require_single_visible_field(visible_fields, strategy="count")
            value = rows[0]["count"] if rows else 0
            raw_answer = {visible_field.name: value}
            return canonicalize_answer(
                task.answer_schema,
                raw_answer,
                float_precision=self.float_precision,
            )
        if plan.strategy == "exists":
            visible_field = _require_single_visible_field(visible_fields, strategy="exists")
            value = rows[0]["exists"] if rows else False
            raw_answer = {visible_field.name: value}
            return canonicalize_answer(
                task.answer_schema,
                raw_answer,
                float_precision=self.float_precision,
            )
        if plan.strategy == "list":
            raw_answer = {
                field.name: [row.get(field.name) for row in rows]
                for field in visible_fields
            }
            return canonicalize_answer(
                task.answer_schema,
                raw_answer,
                float_precision=self.float_precision,
            )
        if not rows:
            if task.outcome_type == "no_result":
                raw_answer = {field.name: None for field in visible_fields}
                return canonicalize_answer(
                    task.answer_schema,
                    raw_answer,
                    float_precision=self.float_precision,
                )
            raise ValueError("answer branch ground truth produced no rows")
        if len(rows) > 1:
            raise ValueError("scalar Tier A ground truth must be uniquely determined")
        raw_answer = {field.name: rows[0].get(field.name) for field in visible_fields}
        return canonicalize_answer(
            task.answer_schema,
            raw_answer,
            float_precision=self.float_precision,
        )

    def _sql_params(self, task: TaskSpec, path: PathSpec) -> dict[str, object]:
        root_table = self.graph.get_table(path.root_table, schema_name=path.edges[0].source_schema)
        if not root_table.primary_key:
            raise ValueError(f"root table has no primary key: {root_table.qualified_name}")
        if len(root_table.primary_key) != 1:
            raise NotImplementedError("Tier A ground truth generator does not yet support composite anchors")
        if task.anchor_pk_column != root_table.primary_key[0]:
            raise ValueError(
                f"task anchor column {task.anchor_pk_column!r} does not match root primary key "
                f"{root_table.primary_key[0]!r}"
            )
        anchor_column = root_table.get_column(task.anchor_pk_column)
        return {
            f"anchor_{task.anchor_pk_column}": _coerce_anchor_value(anchor_column, task.anchor_pk_value)
        }

    def _resolve_reverse_count_relation(self, task: TaskSpec, path: PathSpec):
        metadata = task.contract_metadata
        expected_constraint = str(metadata.get("count_relation_constraint", ""))
        expected_schema = str(metadata.get("count_relation_source_schema", ""))
        expected_table = str(metadata.get("count_relation_source_table", ""))
        for edge in self.graph.edges_to(
            path.edges[-1].target_table,
            schema_name=path.edges[-1].target_schema,
        ):
            if (
                edge.constraint_name == expected_constraint
                and edge.source_schema == expected_schema
                and edge.source_table == expected_table
            ):
                return edge
        raise KeyError(
            f"reverse count relation not found for "
            f"{expected_schema}.{expected_table} via {expected_constraint}"
        )


def _field_marker(field: AnswerField) -> str | None:
    if not field.source_columns:
        return None
    marker = field.source_columns[0].strip().lower()
    if marker in {"meta:count", "meta:exists"}:
        return marker
    return None


def _visible_answer_fields(task: TaskSpec) -> list[AnswerField]:
    visible_fields = [
        field for field in task.answer_schema.fields if field.visibility == "user_visible"
    ]
    if not visible_fields:
        raise ValueError("Tier A ground truth requires at least one user_visible answer field")
    return visible_fields


def _require_single_visible_field(
    fields: list[AnswerField],
    *,
    strategy: Strategy,
) -> AnswerField:
    if len(fields) != 1:
        raise ValueError(f"{strategy} ground truth requires exactly one user_visible answer field")
    return fields[0]


def _resolve_source_column(
    graph: SchemaGraph,
    path: PathSpec,
    field: AnswerField,
) -> ColumnProfile:
    ref = field.source_columns[0] if field.source_columns else f"{path.tables[-1]}.{field.name}"
    parts = ref.split(".")
    if len(parts) == 2:
        schema_name = path.edges[-1].target_schema
        table_name, column_name = parts
    elif len(parts) == 3:
        schema_name, table_name, column_name = parts
    else:
        raise ValueError(f"unsupported source column reference: {ref}")
    if table_name not in path.tables:
        raise ValueError(f"source column table is not present on selected path: {ref}")
    return graph.get_table(table_name, schema_name=schema_name).get_column(column_name)


def _alias_for_table(path: PathSpec, table_name: str, schema_name: str) -> str:
    if path.root_table == table_name and path.edges[0].source_schema == schema_name:
        return _alias_for_index(0)
    for index, edge in enumerate(path.edges, start=1):
        if edge.target_table == table_name and edge.target_schema == schema_name:
            return _alias_for_index(index)
    raise KeyError(f"table {schema_name}.{table_name} is not addressable on path {path.path_id}")


def _projection_expression(
    field: AnswerField,
    *,
    alias: str,
    column: ColumnProfile,
    float_precision: int = 6,
) -> str:
    column_sql = f"{alias}.{quote_ident(column.column_name)}"
    if field.type == "date":
        return f"{column_sql}::date"
    if field.type == "float":
        return f"ROUND(({column_sql})::numeric, {int(float_precision)})"
    return column_sql


def _compile_projection_sql(
    graph: SchemaGraph,
    path: PathSpec,
    *,
    projections: list[Projection],
    distinct: bool,
    limit: int | None,
    order_specs: list[tuple[str, str]] | None = None,
) -> str:
    root_table = graph.get_table(path.root_table, schema_name=path.edges[0].source_schema)
    target_table = graph.get_table(path.edges[-1].target_table, schema_name=path.edges[-1].target_schema)
    resolved_order_specs = order_specs or _target_order_specs(
        _alias_for_index(path.hop_count),
        target_table,
        [projection.answer_field.name for projection in projections],
    )
    select_expressions = [
        (projection.expression_sql, projection.answer_field.name)
        for projection in projections
    ]
    if distinct:
        return _compile_distinct_projection_query(
            select_expressions=select_expressions,
            from_and_joins_sql=_render_from_and_joins(path),
            predicate_sql=_render_anchor_predicate(root_table),
            order_specs=resolved_order_specs,
            limit=limit,
        )

    select_clause = ", ".join(
        f"{expression} AS {quote_ident(alias)}" for expression, alias in select_expressions
    )
    order_clause = ", ".join(
        f"{expression} {direction}"
        for expression, direction in resolved_order_specs
    )
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
    return " ".join(
        f"""
        SELECT {select_clause}
        {_render_from_and_joins(path)}
        WHERE {_render_anchor_predicate(root_table)}
        ORDER BY {order_clause}
        {limit_clause}
        """.split()
    )


def _prepare_asyncpg_query(
    verification_sql: str,
    sql_params: dict[str, object],
) -> tuple[str, list[object]]:
    ordered_names: list[str] = []
    index_by_name: dict[str, int] = {}

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in sql_params:
            raise KeyError(f"missing SQL parameter: {name}")
        if name not in index_by_name:
            index_by_name[name] = len(ordered_names) + 1
            ordered_names.append(name)
        return f"${index_by_name[name]}"

    prepared_sql = _NAMED_PARAM_PATTERN.sub(replace, verification_sql)
    args = [sql_params[name] for name in ordered_names]
    return prepared_sql, args


def _coerce_anchor_value(column: ColumnProfile, raw_value: object) -> object:
    if raw_value is None:
        return None

    data_type = column.data_type.lower()
    if data_type in _INT_DATA_TYPES:
        return int(raw_value)
    if data_type in _FLOAT_DATA_TYPES:
        return float(raw_value)
    if data_type in _BOOL_DATA_TYPES:
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"true", "t", "1", "yes", "y"}:
                return True
            if normalized in {"false", "f", "0", "no", "n"}:
                return False
        return bool(raw_value)
    if data_type in _DATE_DATA_TYPES:
        if isinstance(raw_value, date):
            return raw_value
        return date.fromisoformat(str(raw_value).strip())
    if data_type in _DATETIME_DATA_TYPES:
        if isinstance(raw_value, datetime):
            return raw_value
        return datetime.fromisoformat(str(raw_value).strip().replace("Z", "+00:00"))
    return raw_value
