"""Compiled DB tool generation."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from rl_task_foundry.schema.graph import SchemaGraph
from rl_task_foundry.schema.path_catalog import PathCatalog, PathSpec
from rl_task_foundry.tools.models import ToolBundle, ToolParameter, ToolSpec
from rl_task_foundry.tools.text_utils import singularize_token
from rl_task_foundry.tools.sql_templates import (
    compile_anchor_parameters,
    compile_aggregate_sql,
    compile_count_sql,
    compile_exists_sql,
    compile_list_related_sql,
    compile_lookup_sql,
    compile_timeline_sql,
)

ToolLevel = Literal[1, 2]
LabelTier = Literal["A", "B"]

_DEFAULT_BUSINESS_ALIAS_MAP: dict[str, str] = {
    "account": "account_profile",
    "address": "location_profile",
    "addresses": "location_profile",
    "amount": "amount_value",
    "category": "category_group",
    "categories": "category_group",
    "city": "locality_profile",
    "cities": "locality_profile",
    "contact": "contact_profile",
    "country": "region_profile",
    "countries": "region_profile",
    "customer": "account_profile",
    "customers": "account_profile",
    "date": "event_date",
    "email": "contact_profile",
    "location": "location_profile",
    "member": "member_profile",
    "order": "request_case",
    "orders": "request_case",
    "payment": "payment_record",
    "payments": "payment_record",
    "phone": "contact_profile",
    "price": "amount_value",
    "staff": "service_agent",
    "status": "status_signal",
    "store": "service_location",
    "stores": "service_location",
    "time": "event_time",
    "timeline": "event_timeline",
    "user": "account_profile",
    "users": "account_profile",
}
_FALLBACK_ALIAS_PREFIXES = [
    "portfolio",
    "program",
    "service",
    "operations",
    "commerce",
    "support",
    "settlement",
    "fulfillment",
]
_FALLBACK_ALIAS_SUFFIXES = [
    "hub",
    "record",
    "profile",
    "ledger",
    "signal",
    "desk",
    "channel",
    "registry",
]


def compile_path_tools(
    graph: SchemaGraph,
    path: PathSpec,
    *,
    tool_level: ToolLevel,
    label_tier: LabelTier = "A",
    max_list_cardinality: int = 20,
    allow_aggregates: bool = True,
    allow_timelines: bool = True,
    float_precision: int = 6,
    business_alias_overrides: dict[str, str] | None = None,
) -> ToolBundle:
    """Compile one path into a level-specific tool bundle.

    `tool_level=1` is the canonical semantic bundle used by the task factory.
    `tool_level=2` is a preview/debug fallback bundle; production L2 presentation
    should be generated with task context by the task composer.
    `label_tier="A"` compiles only the low-noise core capability set.
    `label_tier="B"` may additionally enable aggregate/timeline capabilities.
    """

    output_fields = _stable_output_fields(graph, path)
    parameters = _anchor_parameters(graph, path)
    tools: list[ToolSpec] = []
    aggregate_enabled, timeline_enabled = _resolve_capability_flags(
        label_tier=label_tier,
        allow_aggregates=allow_aggregates,
        allow_timelines=allow_timelines,
    )

    if output_fields:
        tools.append(
            ToolSpec(
                name=_tool_name(
                    "lookup",
                    path,
                    tool_level,
                    business_alias_overrides=business_alias_overrides,
                ),
                description=_tool_description("lookup", tool_level),
                sql_template=compile_lookup_sql(graph, path, output_fields=output_fields),
                parameters=parameters,
                output_fields=output_fields,
                path_id=path.path_id,
                kind="lookup",
                tool_level=tool_level,
                semantic_key=f"{path.path_id}:lookup",
                name_source=_name_source_for_level(tool_level),
            )
        )
        tools.append(
            ToolSpec(
                name=_tool_name(
                    "list_related",
                    path,
                    tool_level,
                    business_alias_overrides=business_alias_overrides,
                ),
                description=_tool_description("list_related", tool_level, max_list_cardinality),
                sql_template=compile_list_related_sql(
                    graph,
                    path,
                    output_fields=output_fields,
                    limit=max_list_cardinality,
                ),
                parameters=parameters,
                output_fields=output_fields,
                path_id=path.path_id,
                kind="list_related",
                tool_level=tool_level,
                semantic_key=f"{path.path_id}:list_related",
                name_source=_name_source_for_level(tool_level),
            )
        )
        if aggregate_enabled:
            for aggregate_function, aggregate_column in _aggregate_capabilities(graph, path):
                output_alias = f"{aggregate_function}_{aggregate_column}"
                tools.append(
                    ToolSpec(
                        name=_aggregate_tool_name(
                            path,
                            tool_level,
                            aggregate_function,
                            aggregate_column,
                            business_alias_overrides=business_alias_overrides,
                        ),
                        description=_aggregate_tool_description(
                            tool_level,
                            aggregate_function,
                            aggregate_column,
                        ),
                        sql_template=compile_aggregate_sql(
                            graph,
                            path,
                            aggregate_function=aggregate_function,
                            aggregate_column=aggregate_column,
                            output_alias=output_alias,
                            float_precision=float_precision,
                        ),
                        parameters=parameters,
                        output_fields=[output_alias],
                        path_id=path.path_id,
                        kind="aggregate",
                        tool_level=tool_level,
                        semantic_key=f"{path.path_id}:aggregate:{aggregate_function}:{aggregate_column}",
                        name_source=_name_source_for_level(tool_level),
                    )
                )
        if timeline_enabled:
            for order_column in _timeline_capabilities(graph, path):
                tools.append(
                    ToolSpec(
                        name=_timeline_tool_name(
                            path,
                            tool_level,
                            order_column,
                            business_alias_overrides=business_alias_overrides,
                        ),
                        description=_timeline_tool_description(
                            tool_level,
                            order_column,
                            max_list_cardinality,
                        ),
                        sql_template=compile_timeline_sql(
                            graph,
                            path,
                            output_fields=output_fields,
                            order_column=order_column,
                            limit=max_list_cardinality,
                        ),
                        parameters=parameters,
                        output_fields=output_fields,
                        path_id=path.path_id,
                        kind="timeline",
                        tool_level=tool_level,
                        semantic_key=f"{path.path_id}:timeline:{order_column}",
                        name_source=_name_source_for_level(tool_level),
                    )
                )

    tools.extend(
        [
            ToolSpec(
                name=_tool_name(
                    "count",
                    path,
                    tool_level,
                    business_alias_overrides=business_alias_overrides,
                ),
                description=_tool_description("count", tool_level),
                sql_template=compile_count_sql(graph, path),
                parameters=parameters,
                output_fields=["count"],
                path_id=path.path_id,
                kind="count",
                tool_level=tool_level,
                semantic_key=f"{path.path_id}:count",
                name_source=_name_source_for_level(tool_level),
            ),
            ToolSpec(
                name=_tool_name(
                    "exists",
                    path,
                    tool_level,
                    business_alias_overrides=business_alias_overrides,
                ),
                description=_tool_description("exists", tool_level),
                sql_template=compile_exists_sql(graph, path),
                parameters=parameters,
                output_fields=["exists"],
                path_id=path.path_id,
                kind="exists",
                tool_level=tool_level,
                semantic_key=f"{path.path_id}:exists",
                name_source=_name_source_for_level(tool_level),
            ),
        ]
    )
    return ToolBundle(
        bundle_id=f"{path.path_id}.L{tool_level}",
        path_id=path.path_id,
        tool_level=tool_level,
        tools=tools,
    )


def compile_catalog_tools(
    graph: SchemaGraph,
    catalog: PathCatalog,
    *,
    tool_level: ToolLevel,
    label_tier: LabelTier = "A",
    max_list_cardinality: int = 20,
    allow_aggregates: bool = True,
    allow_timelines: bool = True,
    float_precision: int = 6,
    business_alias_overrides: dict[str, str] | None = None,
) -> list[ToolBundle]:
    """Compile one bundle per path for the requested tool level."""

    bundles: list[ToolBundle] = []
    for path in catalog.paths:
        try:
            bundles.append(
                compile_path_tools(
                    graph,
                    path,
                    tool_level=tool_level,
                    label_tier=label_tier,
                    max_list_cardinality=max_list_cardinality,
                    allow_aggregates=allow_aggregates,
                    allow_timelines=allow_timelines,
                    float_precision=float_precision,
                    business_alias_overrides=business_alias_overrides,
                )
            )
        except ValueError:
            # Non-anchorable paths are excluded from compiled tool sets.
            continue
    return bundles


def compile_all_tool_levels(
    graph: SchemaGraph,
    catalog: PathCatalog,
    *,
    label_tier: LabelTier = "A",
    max_list_cardinality: int = 20,
    allow_aggregates: bool = True,
    allow_timelines: bool = True,
    float_precision: int = 6,
    business_alias_overrides: dict[str, str] | None = None,
) -> dict[int, list[ToolBundle]]:
    """Compile all supported tool levels for preview/debug workflows."""

    return {
        1: compile_catalog_tools(
            graph,
            catalog,
            tool_level=1,
            label_tier=label_tier,
            max_list_cardinality=max_list_cardinality,
            allow_aggregates=allow_aggregates,
            allow_timelines=allow_timelines,
            float_precision=float_precision,
            business_alias_overrides=business_alias_overrides,
        ),
        2: compile_catalog_tools(
            graph,
            catalog,
            tool_level=2,
            label_tier=label_tier,
            max_list_cardinality=max_list_cardinality,
            allow_aggregates=allow_aggregates,
            allow_timelines=allow_timelines,
            float_precision=float_precision,
            business_alias_overrides=business_alias_overrides,
        ),
    }


def compile_canonical_tool_bundle(
    graph: SchemaGraph,
    path: PathSpec,
    *,
    label_tier: LabelTier = "A",
    max_list_cardinality: int = 20,
    allow_aggregates: bool = True,
    allow_timelines: bool = True,
    float_precision: int = 6,
    business_alias_overrides: dict[str, str] | None = None,
) -> ToolBundle:
    """Compile the canonical L1 semantic bundle for one path."""

    return compile_path_tools(
        graph,
        path,
        tool_level=1,
        label_tier=label_tier,
        max_list_cardinality=max_list_cardinality,
        allow_aggregates=allow_aggregates,
        allow_timelines=allow_timelines,
        float_precision=float_precision,
        business_alias_overrides=business_alias_overrides,
    )


def _resolve_capability_flags(
    *,
    label_tier: LabelTier,
    allow_aggregates: bool,
    allow_timelines: bool,
) -> tuple[bool, bool]:
    if label_tier == "A":
        return False, False
    return allow_aggregates, allow_timelines


def _anchor_parameters(graph: SchemaGraph, path: PathSpec) -> list[ToolParameter]:
    root_table = graph.get_table(path.root_table, schema_name=path.edges[0].source_schema)
    parameter_names = compile_anchor_parameters(graph, path)
    return [
        ToolParameter(
            name=parameter_name,
            json_type=_json_type_for_column(root_table.get_column(column_name).data_type),
            description=f"Anchor primary key value for {column_name}",
        )
        for parameter_name, column_name in zip(parameter_names, root_table.primary_key, strict=True)
    ]


def _stable_output_fields(graph: SchemaGraph, path: PathSpec) -> list[str]:
    target_edge = path.edges[-1]
    target_table = graph.get_table(target_edge.target_table, schema_name=target_edge.target_schema)
    return [
        column.column_name
        for column in target_table.columns
        if column.visibility != "blocked"
    ]


def _aggregate_capabilities(graph: SchemaGraph, path: PathSpec) -> list[tuple[str, str]]:
    target_edge = path.edges[-1]
    target_table = graph.get_table(target_edge.target_table, schema_name=target_edge.target_schema)
    numeric_columns = [
        column.column_name
        for column in target_table.columns
        if (
            column.visibility != "blocked"
            and _is_numeric_type(column.data_type)
            and not column.is_primary_key
            and not column.is_foreign_key
            and not _looks_like_identifier_metric(column.column_name)
        )
    ]
    capabilities: list[tuple[str, str]] = []
    for column_name in numeric_columns:
        for aggregate_function in ("sum", "avg", "min", "max"):
            capabilities.append((aggregate_function, column_name))
    return capabilities


def _looks_like_identifier_metric(column_name: str) -> bool:
    normalized = column_name.lower()
    return normalized == "id" or normalized.endswith("_id")


def _timeline_capabilities(graph: SchemaGraph, path: PathSpec) -> list[str]:
    target_edge = path.edges[-1]
    target_table = graph.get_table(target_edge.target_table, schema_name=target_edge.target_schema)
    return [
        column.column_name
        for column in target_table.columns
        if column.visibility != "blocked" and _is_temporal_type(column.data_type)
    ]


def _tool_name(
    kind: Literal["lookup", "list_related", "count", "exists"],
    path: PathSpec,
    tool_level: ToolLevel,
    *,
    business_alias_overrides: dict[str, str] | None = None,
) -> str:
    root = _slug(path.tables[0])
    target = _slug(path.tables[-1])
    via = "_".join(_slug(table_name) for table_name in path.tables[1:-1])
    via_suffix = f"_via_{via}" if via else ""

    if tool_level == 1:
        name_map = {
            "lookup": f"get_{target}_for_{root}{via_suffix}",
            "list_related": f"list_{target}_for_{root}{via_suffix}",
            "count": f"count_{target}_for_{root}{via_suffix}",
            "exists": f"has_{target}_for_{root}{via_suffix}",
        }
        return name_map[kind]
    if tool_level == 2:
        target_alias = _business_alias(path.tables[-1], overrides=business_alias_overrides)
        name_map = {
            "lookup": f"inspect_{target_alias}",
            "list_related": f"list_{target_alias}_options",
            "count": f"count_{target_alias}_matches",
            "exists": f"has_{target_alias}_match",
        }
        return name_map[kind]
    raise AssertionError(f"Unsupported tool level: {tool_level}")


def _aggregate_tool_name(
    path: PathSpec,
    tool_level: ToolLevel,
    aggregate_function: str,
    aggregate_column: str,
    *,
    business_alias_overrides: dict[str, str] | None = None,
) -> str:
    root = _slug(path.tables[0])
    target = _slug(path.tables[-1])
    via = "_".join(_slug(table_name) for table_name in path.tables[1:-1])
    via_suffix = f"_via_{via}" if via else ""
    column_slug = _slug(aggregate_column)

    if tool_level == 1:
        return f"{aggregate_function}_{column_slug}_for_{target}_for_{root}{via_suffix}"
    if tool_level == 2:
        column_alias = _business_alias(aggregate_column, overrides=business_alias_overrides)
        return f"{aggregate_function}_{column_alias}_summary"
    raise AssertionError(f"Unsupported tool level: {tool_level}")


def _timeline_tool_name(
    path: PathSpec,
    tool_level: ToolLevel,
    order_column: str,
    *,
    business_alias_overrides: dict[str, str] | None = None,
) -> str:
    root = _slug(path.tables[0])
    target = _slug(path.tables[-1])
    via = "_".join(_slug(table_name) for table_name in path.tables[1:-1])
    via_suffix = f"_via_{via}" if via else ""
    column_slug = _slug(order_column)

    if tool_level == 1:
        return f"timeline_{target}_for_{root}_by_{column_slug}{via_suffix}"
    if tool_level == 2:
        target_alias = _business_alias(path.tables[-1], overrides=business_alias_overrides)
        return f"review_{target_alias}_timeline"
    raise AssertionError(f"Unsupported tool level: {tool_level}")


def _tool_description(
    kind: Literal["lookup", "list_related", "count", "exists"],
    tool_level: ToolLevel,
    max_list_cardinality: int | None = None,
) -> str:
    descriptions = {
        "lookup": "Fetch the matching related details.",
        "list_related": (
            "Browse a deterministic bounded list of matching related details."
        ),
        "count": "Count how many matching related results exist.",
        "exists": "Check whether any matching related result exists.",
    }
    suffix = ""
    if kind == "list_related" and max_list_cardinality is not None:
        suffix = f" Maximum {max_list_cardinality} rows."
    return f"L{tool_level} {kind} tool. {descriptions[kind]}{suffix}"


def _aggregate_tool_description(
    tool_level: ToolLevel,
    aggregate_function: str,
    aggregate_column: str,
) -> str:
    return (
        f"L{tool_level} aggregate tool. "
        f"Return {aggregate_function.upper()} over a related numeric field."
    )


def _timeline_tool_description(
    tool_level: ToolLevel,
    order_column: str,
    max_list_cardinality: int,
) -> str:
    return (
        f"L{tool_level} timeline tool. "
        f"Return up to {max_list_cardinality} related results in deterministic most-recent-first order."
    )


def _json_type_for_column(data_type: str) -> str:
    normalized = data_type.lower()
    if normalized in {"int2", "int4", "int8", "integer", "smallint", "bigint"}:
        return "integer"
    if normalized in {"numeric", "decimal", "float4", "float8", "double precision", "real"}:
        return "number"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    return "string"


def _name_source_for_level(tool_level: ToolLevel) -> Literal["rule_based", "model_generated", "fallback_alias"]:
    if tool_level == 1:
        return "rule_based"
    return "fallback_alias"


def _is_numeric_type(data_type: str) -> bool:
    normalized = data_type.lower()
    return normalized in {
        "int2",
        "int4",
        "int8",
        "integer",
        "smallint",
        "bigint",
        "numeric",
        "decimal",
        "float4",
        "float8",
        "double precision",
        "real",
        "money",
    }


def _is_temporal_type(data_type: str) -> bool:
    normalized = data_type.lower()
    return normalized in {
        "date",
        "datetime",
        "timestamp",
        "timestamp without time zone",
        "timestamp with time zone",
        "timestamptz",
        "time",
        "time without time zone",
        "time with time zone",
    }


def _slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _business_route_alias(
    parts: list[str],
    *,
    overrides: dict[str, str] | None = None,
) -> str:
    return "_".join(_business_alias(part, overrides=overrides) for part in parts)


def _business_via_suffix(
    parts: list[str],
    *,
    overrides: dict[str, str] | None = None,
) -> str:
    if not parts:
        return ""
    return "_via_" + _business_route_alias(parts, overrides=overrides)


def _business_alias(value: str, *, overrides: dict[str, str] | None = None) -> str:
    alias_map = dict(_DEFAULT_BUSINESS_ALIAS_MAP)
    if overrides:
        alias_map.update({key.strip().lower(): val for key, val in overrides.items()})
    normalized = _slug(value)
    direct_match = alias_map.get(normalized)
    if direct_match is not None:
        return direct_match

    tokens = [token for token in re.split(r"[_\\W]+", normalized) if token]
    if not tokens:
        return _opaque_business_alias(normalized)

    aliased_tokens: list[str] = []
    for token in tokens:
        mapped = alias_map.get(token)
        if mapped is None:
            singular = singularize_token(token)
            mapped = alias_map.get(singular)
        if mapped is None:
            mapped = _opaque_business_alias(token)
        aliased_tokens.extend(mapped.split("_"))

    collapsed: list[str] = []
    for token in aliased_tokens:
        if not collapsed or collapsed[-1] != token:
            collapsed.append(token)
    return "_".join(collapsed)
def _opaque_business_alias(token: str) -> str:
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=4).hexdigest()
    digest_int = int(digest, 16)
    prefix = _FALLBACK_ALIAS_PREFIXES[digest_int % len(_FALLBACK_ALIAS_PREFIXES)]
    suffix = _FALLBACK_ALIAS_SUFFIXES[(digest_int // len(_FALLBACK_ALIAS_PREFIXES)) % len(_FALLBACK_ALIAS_SUFFIXES)]
    code = digest[:2]
    return f"{prefix}_{suffix}_{code}"
