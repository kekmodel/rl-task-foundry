"""Automatic Tier A task-spec generation from anchors and path catalog."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Literal

import asyncpg

from rl_task_foundry.config.models import (
    DatabaseConfig,
    DomainConfig,
    TaskComposerConfig,
    ToolCompilerConfig,
    VerificationConfig,
)
from rl_task_foundry.infra.db import control_session_settings
from rl_task_foundry.schema.graph import ColumnProfile, SchemaGraph
from rl_task_foundry.schema.path_catalog import PathCatalog, PathSpec
from rl_task_foundry.tasks.models import TaskSpec
from rl_task_foundry.tasks.validator import TaskValidator
from rl_task_foundry.tools.compiler import compile_canonical_tool_bundle
from rl_task_foundry.tools.sql_templates import (
    _alias_for_index,
    _distinct_target_value_expression,
    _render_from_and_joins,
    quote_ident,
    quote_table,
    readonly_query,
)
from rl_task_foundry.tools.text_utils import humanize_identifier, singularize_token
from rl_task_foundry.truth.generator import TierAGroundTruthGenerator
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema


_INT_TYPES = {
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
_FLOAT_TYPES = {
    "numeric",
    "decimal",
    "float4",
    "float8",
    "real",
    "double precision",
}
_BOOL_TYPES = {"bool", "boolean"}
_DATE_TYPES = {"date"}
_DATETIME_TYPES = {
    "timestamp",
    "timestamp without time zone",
    "timestamp with time zone",
    "timestamptz",
}
_DISALLOWED_ANSWER_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|_)(ssn|social_security|passport|tax_id|cvv|pin|password|secret|token|api_key|private_key)($|_)",
        r"(^|_)(credit_card|card_number|bank_account|iban|routing_number)($|_)",
        r"(^|_)(email|e_mail|phone|mobile|telephone|fax)($|_)",
        r"(^|_)(street|address|zipcode|postal_code|postcode)($|_)",
        r"(^|_)(birth|birthdate|birthday|dob|first_name|last_name|full_name)($|_)",
    )
)
_PROMOTABLE_INTERNAL_FIELD_NAMES = {
    "city",
    "state",
    "province",
    "country",
    "region",
    "language",
    "status",
    "category",
    "type",
    "kind",
    "title",
    "label",
    "name",
    "code",
}
_IDENTITY_TABLE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|_)(customer|user|member|profile|account|person|staff|employee|contact|address)($|_)",
    )
)


@dataclass(slots=True)
class TaskContractDraft:
    question_family: str
    outcome_type: Literal["answer", "no_result", "clarify", "deny"]
    answer_schema: AnswerSchema
    field_label: str | None = None
    target_label: str | None = None


@dataclass(slots=True)
class TaskSource:
    path: PathSpec
    contract: TaskContractDraft
    anchor_values: list[object]


@dataclass(slots=True)
class TierATaskFactory:
    database: DatabaseConfig
    domain: DomainConfig
    task_config: TaskComposerConfig
    tool_compiler: ToolCompilerConfig
    verification: VerificationConfig

    async def generate(
        self,
        graph: SchemaGraph,
        catalog: PathCatalog,
        *,
        limit: int,
        path_ids: list[str] | None = None,
    ) -> list[TaskSpec]:
        """Generate validated Tier A task specs from anchor samples."""

        if limit <= 0:
            return []

        selected_paths = self._selected_paths(catalog, path_ids=path_ids)
        generator = TierAGroundTruthGenerator(
            database=self.database,
            graph=graph,
            catalog=catalog,
            float_precision=self.verification.float_precision,
        )
        validator = TaskValidator()
        tasks: list[TaskSpec] = []
        negative_budget = round(limit * self.task_config.negative_outcome_ratio)
        sources = await self._build_sources(
            graph,
            selected_paths,
            limit=limit,
            negative_budget=negative_budget,
        )

        while sources and len(tasks) < limit:
            next_sources: list[TaskSource] = []
            progress_made = False
            for source in sources:
                while source.anchor_values and len(tasks) < limit:
                    anchor_value = source.anchor_values.pop(0)
                    task = self._build_task_spec(
                        graph=graph,
                        path=source.path,
                        contract=source.contract,
                        anchor_pk_value=str(anchor_value),
                    )
                    validation_issues = validator.validate(task)
                    if validation_issues:
                        continue
                    if not await self._validate_ground_truth(generator, task):
                        continue
                    tasks.append(task)
                    progress_made = True
                    break
                if source.anchor_values:
                    next_sources.append(source)
            if not progress_made:
                break
            sources = next_sources
        return tasks

    def candidate_tasks_for_anchor(
        self,
        graph: SchemaGraph,
        catalog: PathCatalog,
        *,
        anchor_table: str,
        anchor_pk_column: str,
        anchor_pk_value: str,
        path_ids: list[str] | None = None,
        question_families: list[str] | None = None,
        outcome_types: list[Literal["answer", "no_result", "clarify", "deny"]] | None = None,
    ) -> list[TaskSpec]:
        selected_paths = self._selected_paths(catalog, path_ids=path_ids)
        validator = TaskValidator()
        tasks: list[TaskSpec] = []
        allowed_families = set(question_families) if question_families is not None else None
        allowed_outcomes = set(outcome_types) if outcome_types is not None else None

        for path in selected_paths:
            if path.root_table != anchor_table:
                continue
            if not self._supports_anchor_sampling(graph, path):
                continue
            root_table = graph.get_table(path.root_table, schema_name=path.edges[0].source_schema)
            if root_table.primary_key[0] != anchor_pk_column:
                continue
            for contract in self._contract_drafts_for_path(graph, path):
                if allowed_families is not None and contract.question_family not in allowed_families:
                    continue
                if allowed_outcomes is not None and contract.outcome_type not in allowed_outcomes:
                    continue
                task = self._build_task_spec(
                    graph=graph,
                    path=path,
                    contract=contract,
                    anchor_pk_value=anchor_pk_value,
                )
                validation_issues = validator.validate(task)
                if validation_issues:
                    continue
                tasks.append(task)
        return tasks

    async def _build_sources(
        self,
        graph: SchemaGraph,
        selected_paths: list[PathSpec],
        *,
        limit: int,
        negative_budget: int,
    ) -> list[TaskSource]:
        sources_by_family: dict[str, list[TaskSource]] = {
            family: [] for family in self.task_config.question_families
        }
        scheduled_negative = 0
        for path in selected_paths:
            if not self._supports_anchor_sampling(graph, path):
                continue
            for contract in self._contract_drafts_for_path(graph, path):
                if contract.outcome_type != "answer":
                    if scheduled_negative >= negative_budget:
                        continue
                    scheduled_negative += 1
                anchor_values = await self._sample_anchor_values(
                    graph,
                    path,
                    contract,
                    limit=max(1, min(self.task_config.anchor_samples_per_source, limit)),
                )
                if not anchor_values:
                    continue
                sources_by_family.setdefault(contract.question_family, []).append(
                    TaskSource(path=path, contract=contract, anchor_values=list(anchor_values))
                )
        for family, sources in sources_by_family.items():
            sources.sort(
                key=lambda source: _source_priority_key(
                    family,
                    source,
                    aggregate_discouraged_target_patterns=self.task_config.aggregate_discouraged_target_patterns,
                )
            )
        return _interleave_sources_by_family(
            sources_by_family,
            family_order=self.task_config.question_families,
        )

    async def _validate_ground_truth(
        self,
        generator: TierAGroundTruthGenerator,
        task: TaskSpec,
    ) -> bool:
        try:
            await generator.generate(task)
        except Exception:
            return False
        return True

    def _selected_paths(self, catalog: PathCatalog, *, path_ids: list[str] | None) -> list[PathSpec]:
        if not path_ids:
            return sorted(catalog.paths, key=lambda path: (path.hop_count, path.path_id))
        return [catalog.get(path_id) for path_id in path_ids]

    def _supports_anchor_sampling(self, graph: SchemaGraph, path: PathSpec) -> bool:
        root_table = graph.get_table(path.root_table, schema_name=path.edges[0].source_schema)
        return len(root_table.primary_key) == 1

    def _contract_drafts_for_path(
        self,
        graph: SchemaGraph,
        path: PathSpec,
    ) -> list[TaskContractDraft]:
        drafts: list[TaskContractDraft] = []
        target_table = graph.get_table(path.edges[-1].target_table, schema_name=path.edges[-1].target_schema)
        scalar_field = self._select_scalar_field(target_table)
        list_field = self._select_list_field(target_table)
        temporal_field = self._select_temporal_field(target_table)

        for family in self.task_config.question_families:
            if family == "status_lookup" and scalar_field is not None:
                drafts.append(
                    self._scalar_contract_draft(
                        path,
                        family=family,
                        column=scalar_field,
                        outcome_type="answer",
                    )
                )
                drafts.append(
                    self._scalar_contract_draft(
                        path,
                        family=family,
                        column=scalar_field,
                        outcome_type="no_result",
                    )
                )
            elif family == "causal_chain" and path.hop_count >= 2 and scalar_field is not None:
                if list_field is not None:
                    drafts.append(
                        self._list_contract_draft(
                            path,
                            family=family,
                            column=list_field,
                            outcome_type="answer",
                        )
                    )
                drafts.append(
                    self._scalar_contract_draft(
                        path,
                        family=family,
                        column=scalar_field,
                        outcome_type="answer",
                    )
                )
                drafts.append(
                    self._scalar_contract_draft(
                        path,
                        family=family,
                        column=scalar_field,
                        outcome_type="no_result",
                    )
                )
            elif family == "timeline_resolution" and temporal_field is not None:
                drafts.append(
                    self._scalar_contract_draft(
                        path,
                        family=family,
                        column=temporal_field,
                        outcome_type="answer",
                    )
                )
                drafts.append(
                    self._scalar_contract_draft(
                        path,
                        family=family,
                        column=temporal_field,
                        outcome_type="no_result",
                    )
                )
            elif family == "aggregate_verification":
                drafts.append(self._count_contract_draft(path))
        return drafts

    def _list_contract_draft(
        self,
        path: PathSpec,
        *,
        family: str,
        column: ColumnProfile,
        outcome_type: Literal["answer", "no_result"],
    ) -> TaskContractDraft:
        base_field_name = self._answer_field_name(path, column)
        field_name = f"{base_field_name}_list"
        field_label = humanize_identifier(base_field_name)
        target_table_label = humanize_identifier(path.tables[-1])
        answer_type = _column_to_list_answer_type(column)
        answer_schema = AnswerSchema(
            fields=[
                AnswerField(
                    name=field_name,
                    type=answer_type,
                    ordered=False,
                    canonicalizer=_default_canonicalizer(column),
                    description=f"User-facing list of related {field_label} values.",
                    visibility=self._answer_field_visibility(column),
                    source_columns=[f"{column.table_name}.{column.column_name}"],
                )
            ]
        )
        return TaskContractDraft(
            question_family=family,
            outcome_type=outcome_type,
            answer_schema=answer_schema,
            field_label=field_label,
            target_label=target_table_label,
        )

    def _scalar_contract_draft(
        self,
        path: PathSpec,
        *,
        family: str,
        column: ColumnProfile,
        outcome_type: Literal["answer", "no_result"],
    ) -> TaskContractDraft:
        field_name = self._answer_field_name(path, column)
        field_label = humanize_identifier(field_name)
        target_table_label = humanize_identifier(path.tables[-1])
        answer_schema = AnswerSchema(
            fields=[
                AnswerField(
                    name=field_name,
                    type=_column_to_answer_type(column),
                    canonicalizer=_default_canonicalizer(column),
                    description=f"User-facing {field_label} value.",
                    visibility=self._answer_field_visibility(column),
                    source_columns=[f"{column.table_name}.{column.column_name}"],
                )
            ]
        )
        return TaskContractDraft(
            question_family=family,
            outcome_type=outcome_type,
            answer_schema=answer_schema,
            field_label=field_label,
            target_label=target_table_label,
        )

    def _count_contract_draft(self, path: PathSpec) -> TaskContractDraft:
        target_label = singularize_token(path.tables[-1])
        target_human = humanize_identifier(target_label)
        answer_schema = AnswerSchema(
            fields=[
                AnswerField(
                    name="related_count",
                    type="int",
                    canonicalizer="int_cast",
                    description=f"Count of related {target_human} items.",
                    source_columns=["meta:count"],
                )
            ]
        )
        return TaskContractDraft(
            question_family="aggregate_verification",
            outcome_type="answer",
            answer_schema=answer_schema,
            target_label=target_human,
        )

    async def _sample_anchor_values(
        self,
        graph: SchemaGraph,
        path: PathSpec,
        contract: TaskContractDraft,
        *,
        limit: int,
    ) -> list[object]:
        root_table = graph.get_table(path.root_table, schema_name=path.edges[0].source_schema)
        root_pk = root_table.primary_key[0]
        conn = await asyncpg.connect(dsn=self.database.dsn)
        try:
            settings = control_session_settings(self.database)
            for statement in settings.timeout_sql:
                await conn.execute(statement)
            sql = self._compile_anchor_sampling_sql(graph, path, contract, limit=limit)
            rows = await conn.fetch(sql)
            if not rows and contract.question_family == "aggregate_verification":
                fallback_sql = self._compile_anchor_sampling_sql(
                    graph,
                    path,
                    contract,
                    limit=limit,
                    min_related_count=1,
                )
                rows = await conn.fetch(fallback_sql)
        finally:
            await conn.close()
        return [row["anchor_pk"] for row in rows]

    def _compile_anchor_sampling_sql(
        self,
        graph: SchemaGraph,
        path: PathSpec,
        contract: TaskContractDraft,
        *,
        limit: int,
        min_related_count: int = 2,
    ) -> str:
        root_table = graph.get_table(path.root_table, schema_name=path.edges[0].source_schema)
        root_alias = _alias_for_index(0)
        root_pk = root_table.primary_key[0]
        root_pk_sql = f"{root_alias}.{quote_ident(root_pk)}"
        if contract.question_family == "aggregate_verification":
            target_alias = _alias_for_index(path.hop_count)
            target_table = graph.get_table(
                path.edges[-1].target_table,
                schema_name=path.edges[-1].target_schema,
            )
            target_pk = target_table.primary_key[0]
            distinct_target_sql = _distinct_target_value_expression(target_alias, target_table)
            having_clause = (
                f"HAVING COUNT(DISTINCT {distinct_target_sql}) > 1"
                if min_related_count <= 2
                else f"HAVING COUNT(DISTINCT {distinct_target_sql}) >= {int(min_related_count)}"
            )
            if min_related_count == 1:
                having_clause = f"HAVING COUNT(DISTINCT {distinct_target_sql}) >= 1"
            return readonly_query(
                f"""
                SELECT {root_pk_sql} AS anchor_pk
                {_render_from_and_joins(path)}
                WHERE {target_alias}.{quote_ident(target_pk)} IS NOT NULL
                GROUP BY {root_pk_sql}
                {having_clause}
                ORDER BY {root_pk_sql} ASC
                LIMIT {int(limit)}
                """
            )

        field = contract.answer_schema.fields[0]
        target_alias = _alias_for_index(path.hop_count)
        source_ref = field.source_columns[0]
        column_name = source_ref.split(".")[-1]
        value_predicate = f"{target_alias}.{quote_ident(column_name)} IS NOT NULL"
        if field.type.startswith("list[") and contract.outcome_type == "answer":
            return readonly_query(
                f"""
                SELECT {root_pk_sql} AS anchor_pk
                {_render_from_and_joins(path)}
                WHERE {value_predicate}
                GROUP BY {root_pk_sql}
                HAVING COUNT(DISTINCT {target_alias}.{quote_ident(column_name)}) > 1
                ORDER BY {root_pk_sql} ASC
                LIMIT {int(limit)}
                """
            )
        if contract.outcome_type == "answer":
            return readonly_query(
                f"""
                SELECT DISTINCT {root_pk_sql} AS anchor_pk
                {_render_from_and_joins(path)}
                WHERE {value_predicate}
                ORDER BY anchor_pk ASC
                LIMIT {int(limit)}
                """
            )
        outer_root_alias = "a0"
        outer_root_pk_sql = f"{outer_root_alias}.{quote_ident(root_pk)}"
        return readonly_query(
            f"""
            SELECT {outer_root_pk_sql} AS anchor_pk
            FROM {quote_table(root_table.schema_name, root_table.table_name)} AS {outer_root_alias}
            WHERE NOT EXISTS (
              SELECT 1
              {_render_from_and_joins(path)}
              WHERE {root_pk_sql} = {outer_root_pk_sql}
                AND {value_predicate}
            )
            ORDER BY anchor_pk ASC
            LIMIT {int(limit)}
            """
        )

    def _build_task_spec(
        self,
        *,
        graph: SchemaGraph,
        path: PathSpec,
        contract: TaskContractDraft,
        anchor_pk_value: str,
    ) -> TaskSpec:
        root_table = graph.get_table(path.root_table, schema_name=path.edges[0].source_schema)
        canonical_bundle = compile_canonical_tool_bundle(
            graph,
            path,
            label_tier=self.task_config.label_tier,
            max_list_cardinality=self.tool_compiler.max_list_cardinality,
            allow_aggregates=self.tool_compiler.allow_aggregates,
            allow_timelines=self.tool_compiler.allow_timelines,
            float_precision=self.verification.float_precision,
            business_alias_overrides=self.tool_compiler.business_alias_overrides,
        )
        task_id = _task_id(
            path_id=path.path_id,
            question_family=contract.question_family,
            outcome_type=contract.outcome_type,
            anchor_pk_value=anchor_pk_value,
            field_names=[field.name for field in contract.answer_schema.fields],
        )
        return TaskSpec(
            task_id=task_id,
            anchor_table=path.root_table,
            anchor_pk_column=root_table.primary_key[0],
            anchor_pk_value=anchor_pk_value,
            domain=self.domain.name,
            language=self.domain.language,
            label_tier=self.task_config.label_tier,
            question_family=contract.question_family,
            question=self._render_question(contract),
            outcome_type=contract.outcome_type,
            answer_schema=contract.answer_schema,
            selected_path_id=path.path_id,
            required_hops=path.hop_count,
            tool_level=self.task_config.selected_tool_level,
            tool_bundle_id=canonical_bundle.bundle_id,
            provenance_requirements=[path.path_id],
            difficulty_features={
                **dict(path.difficulty_features),
                "answer_shape": _answer_schema_shape(
                    contract.answer_schema,
                    question_family=contract.question_family,
                ),
            },
            sensitivity_policy="default",
        )

    def _select_scalar_field(self, table) -> ColumnProfile | None:
        for column in table.columns:
            if not self._is_candidate_answer_column(column):
                continue
            answer_type = _column_to_answer_type(column)
            if answer_type in {"string", "int", "float", "bool", "date", "datetime"}:
                return column
        return None

    def _select_temporal_field(self, table) -> ColumnProfile | None:
        for column in table.columns:
            if not self._is_candidate_answer_column(column):
                continue
            if _column_to_answer_type(column) in {"date", "datetime"}:
                return column
        return None

    def _select_list_field(self, table) -> ColumnProfile | None:
        for column in table.columns:
            if not self._is_candidate_answer_column(column):
                continue
            if _column_to_list_answer_type(column) in {"list[string]", "list[int]"}:
                return column
        return None

    def _render_question(self, contract: TaskContractDraft) -> str:
        if self.domain.language == "ko":
            return self._render_ko_question(contract)
        return self._render_en_question(contract)

    def _render_ko_question(self, contract: TaskContractDraft) -> str:
        field_label = contract.field_label or contract.target_label or "정보"
        target_label = contract.target_label or field_label
        if contract.question_family == "status_lookup":
            return f"연결된 {field_label} 값을 확인해줘."
        if contract.question_family == "causal_chain":
            if _answer_schema_shape(contract.answer_schema, question_family=contract.question_family) == "list":
                return f"연결 경로를 따라 어떤 {field_label} 항목들이 있는지 확인해줘."
            return f"연결 경로를 따라 최종 {field_label} 값을 확인해줘."
        if contract.question_family == "timeline_resolution":
            return f"연결된 기록 중 가장 최근 {field_label} 시점을 확인해줘."
        return f"연결된 {target_label} 항목 수를 확인해줘."

    def _render_en_question(self, contract: TaskContractDraft) -> str:
        field_label = contract.field_label or contract.target_label or "information"
        target_label = contract.target_label or field_label
        if contract.question_family == "status_lookup":
            return f"Check the related {field_label} value."
        if contract.question_family == "causal_chain":
            if _answer_schema_shape(contract.answer_schema, question_family=contract.question_family) == "list":
                return f"Check which related {field_label} items exist."
            return f"Follow the relation path and find the final {field_label} value."
        if contract.question_family == "timeline_resolution":
            return f"Find the most recent related {field_label} timestamp."
        return f"Check how many related {target_label} records exist."

    def _is_candidate_answer_column(self, column: ColumnProfile) -> bool:
        if self._answer_field_visibility(column) != "user_visible":
            return False
        if column.is_primary_key or column.is_foreign_key:
            return False
        normalized = column.column_name.strip().lower()
        for pattern in _DISALLOWED_ANSWER_PATTERNS:
            if pattern.search(normalized):
                return False
        for raw_pattern in self.task_config.exclude_answer_column_patterns:
            try:
                pattern = re.compile(raw_pattern, re.IGNORECASE)
            except re.error:
                continue
            if pattern.search(normalized):
                return False
        return True

    def _answer_field_visibility(
        self,
        column: ColumnProfile,
    ) -> Literal["blocked", "internal", "user_visible"]:
        if column.visibility == "user_visible":
            return "user_visible"
        if column.visibility == "internal" and self._can_promote_internal_answer_column(column):
            return "user_visible"
        return column.visibility

    def _can_promote_internal_answer_column(self, column: ColumnProfile) -> bool:
        if column.visibility != "internal":
            return False
        normalized = column.column_name.strip().lower()
        if normalized not in _PROMOTABLE_INTERNAL_FIELD_NAMES:
            return False
        table_name = singularize_token(column.table_name.strip().lower())
        if any(pattern.search(table_name) for pattern in _IDENTITY_TABLE_PATTERNS):
            return False
        return True

    @staticmethod
    def _answer_field_name(path: PathSpec, column: ColumnProfile) -> str:
        table_token = singularize_token(path.tables[-1].strip().lower())
        column_token = column.column_name.strip().lower()
        if column_token == table_token:
            return column_token
        if column_token in {"name", "title", "status", "type", "kind", "label", "code"}:
            return f"{table_token}_{column_token}"
        return column_token


def _task_id(
    *,
    path_id: str,
    question_family: str,
    outcome_type: str,
    anchor_pk_value: str,
    field_names: list[str],
) -> str:
    payload = "|".join([path_id, question_family, outcome_type, anchor_pk_value, *field_names])
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=6).hexdigest()
    return f"task::{question_family}::{digest}"


def _column_to_answer_type(column: ColumnProfile) -> Literal[
    "string",
    "int",
    "float",
    "bool",
    "date",
    "datetime",
]:
    data_type = column.data_type.lower()
    if data_type in _INT_TYPES:
        return "int"
    if data_type in _FLOAT_TYPES:
        return "float"
    if data_type in _BOOL_TYPES:
        return "bool"
    if data_type in _DATE_TYPES:
        return "date"
    if data_type in _DATETIME_TYPES:
        return "datetime"
    return "string"


def _column_to_list_answer_type(column: ColumnProfile) -> Literal["list[string]", "list[int]"] | None:
    answer_type = _column_to_answer_type(column)
    if answer_type == "string":
        return "list[string]"
    if answer_type == "int":
        return "list[int]"
    return None


def _default_canonicalizer(column: ColumnProfile) -> str:
    answer_type = _column_to_answer_type(column)
    if answer_type in {"date", "datetime"}:
        return answer_type
    if answer_type == "float":
        return "round_custom"
    if answer_type == "int":
        return "int_cast"
    if answer_type == "bool":
        return "bool_cast"
    return "lower_trim"


def _answer_schema_shape(
    schema: AnswerSchema,
    *,
    question_family: str,
) -> str:
    if any(field.source_columns and field.source_columns[0] == "meta:count" for field in schema.fields):
        return "count"
    if any(field.source_columns and field.source_columns[0] == "meta:exists" for field in schema.fields):
        return "exists"
    if schema.fields and all(field.type.startswith("list[") for field in schema.fields):
        return "list"
    if question_family == "timeline_resolution":
        return "latest_scalar"
    return "scalar"


def _interleave_sources_by_family(
    sources_by_family: dict[str, list[TaskSource]],
    *,
    family_order: list[str],
) -> list[TaskSource]:
    queues = {
        family: list(sources_by_family.get(family, []))
        for family in family_order
    }
    for family, sources in sources_by_family.items():
        if family not in queues:
            queues[family] = list(sources)
    interleaved: list[TaskSource] = []
    while any(queues.values()):
        for family in family_order:
            queue = queues.get(family)
            if queue:
                interleaved.append(queue.pop(0))
        for family, queue in queues.items():
            if family in family_order:
                continue
            if queue:
                interleaved.append(queue.pop(0))
    return interleaved


def _source_priority_key(
    family: str,
    source: TaskSource,
    *,
    aggregate_discouraged_target_patterns: list[str],
) -> tuple[float, ...] | tuple[int, ...] | tuple[object, ...]:
    fanout = float(source.path.difficulty_features.get("fanout_product", 1.0))
    hop_count = int(source.path.hop_count)
    answer_shape = _answer_schema_shape(
        source.contract.answer_schema,
        question_family=source.contract.question_family,
    )
    if family == "aggregate_verification":
        discouraged_rank = int(
            _matches_any_pattern(
                source.path.tables[-1],
                aggregate_discouraged_target_patterns,
            )
        )
        return (discouraged_rank, -fanout, -hop_count, source.path.path_id)
    if family == "causal_chain":
        list_rank = 0 if answer_shape == "list" else 1
        return (list_rank, -hop_count, -fanout, source.path.path_id)
    if family == "timeline_resolution":
        return (-hop_count, -fanout, source.path.path_id)
    return (hop_count, source.path.path_id)


def _matches_any_pattern(value: str, patterns: list[str]) -> bool:
    normalized = value.strip().lower()
    for raw_pattern in patterns:
        try:
            pattern = re.compile(raw_pattern, re.IGNORECASE)
        except re.error:
            continue
        if pattern.search(normalized):
            return True
    return False
