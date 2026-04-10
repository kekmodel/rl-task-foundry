"""Heuristic evaluation for compiled tool naming quality."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rl_task_foundry.schema.graph import SchemaGraph
from rl_task_foundry.schema.path_catalog import PathSpec
from rl_task_foundry.tools.models import ToolBundle
from rl_task_foundry.tools.text_utils import singularize_token

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,100}$")
_GENERIC_TOKENS = {
    "get",
    "list",
    "count",
    "has",
    "inspect",
    "enumerate",
    "measure",
    "check",
    "lookup",
    "browse",
    "read",
    "confirm",
    "trace",
    "find",
    "peek",
    "rows",
    "row",
    "link",
    "links",
    "by",
    "for",
    "via",
    "avg",
    "sum",
    "min",
    "max",
}


@dataclass(slots=True)
class ToolNameCheck:
    semantic_key: str
    name: str
    invalid_format: bool
    raw_table_hits: list[str] = field(default_factory=list)
    raw_column_hits: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ToolNamingEvaluation:
    tool_level: int
    total_tools: int
    name_sources: list[str]
    duplicate_name_count: int
    invalid_name_count: int
    raw_identifier_hit_count: int
    raw_identifier_overlap_ratio: float
    schema_opacity_score: float
    tools_with_raw_table_hits: int
    tools_with_raw_column_hits: int
    policy_violations: list[str] = field(default_factory=list)
    per_tool: list[ToolNameCheck] = field(default_factory=list)

    @property
    def passes_hard_constraints(self) -> bool:
        return self.duplicate_name_count == 0 and self.invalid_name_count == 0


def evaluate_tool_bundle_naming(
    graph: SchemaGraph,
    path: PathSpec,
    bundle: ToolBundle,
) -> ToolNamingEvaluation:
    """Evaluate naming quality and level-policy heuristics for one tool bundle."""

    raw_table_tokens = _table_tokens(path)
    raw_column_tokens = _column_tokens(graph, path, bundle)

    seen_names: set[str] = set()
    duplicate_name_count = 0
    invalid_name_count = 0
    total_name_tokens = 0
    raw_identifier_hit_count = 0
    tools_with_raw_table_hits = 0
    tools_with_raw_column_hits = 0
    per_tool: list[ToolNameCheck] = []

    for tool in bundle.tools:
        invalid_format = _NAME_PATTERN.fullmatch(tool.name) is None
        if invalid_format:
            invalid_name_count += 1
        if tool.name in seen_names:
            duplicate_name_count += 1
        seen_names.add(tool.name)

        name_tokens = _meaningful_tokens(tool.name)
        total_name_tokens += len(name_tokens)
        table_hits = sorted(raw_table_tokens.intersection(name_tokens))
        column_hits = sorted(raw_column_tokens.intersection(name_tokens))
        raw_hits = set(table_hits).union(column_hits)
        if table_hits:
            tools_with_raw_table_hits += 1
        if column_hits:
            tools_with_raw_column_hits += 1
        raw_identifier_hit_count += len(raw_hits)
        per_tool.append(
            ToolNameCheck(
                semantic_key=tool.semantic_key,
                name=tool.name,
                invalid_format=invalid_format,
                raw_table_hits=table_hits,
                raw_column_hits=column_hits,
            )
        )

    overlap_ratio = (
        raw_identifier_hit_count / total_name_tokens if total_name_tokens else 0.0
    )
    opacity_score = max(0.0, 1.0 - overlap_ratio)
    evaluation = ToolNamingEvaluation(
        tool_level=bundle.tool_level,
        total_tools=len(bundle.tools),
        name_sources=sorted({tool.name_source for tool in bundle.tools}),
        duplicate_name_count=duplicate_name_count,
        invalid_name_count=invalid_name_count,
        raw_identifier_hit_count=raw_identifier_hit_count,
        raw_identifier_overlap_ratio=overlap_ratio,
        schema_opacity_score=opacity_score,
        tools_with_raw_table_hits=tools_with_raw_table_hits,
        tools_with_raw_column_hits=tools_with_raw_column_hits,
        policy_violations=_policy_violations(
            bundle.tool_level,
            overlap_ratio=overlap_ratio,
            tools_with_raw_table_hits=tools_with_raw_table_hits,
            tools_with_raw_column_hits=tools_with_raw_column_hits,
        ),
        per_tool=per_tool,
    )
    return evaluation


def _policy_violations(
    tool_level: int,
    *,
    overlap_ratio: float,
    tools_with_raw_table_hits: int,
    tools_with_raw_column_hits: int,
) -> list[str]:
    violations: list[str] = []
    if tool_level == 1:
        if tools_with_raw_table_hits == 0:
            violations.append("L1 naming is too opaque; expected direct table cues")
    elif tool_level == 2:
        if overlap_ratio == 0.0:
            violations.append("L2 naming is too opaque; expected some discoverable schema cues")
        if overlap_ratio > 0.75:
            violations.append("L2 naming is too literal; expected semi-indirect naming")
    return violations


def _table_tokens(path: PathSpec) -> set[str]:
    tokens: set[str] = set()
    for table_name in path.tables:
        tokens.update(_identifier_tokens(table_name))
    return tokens


def _column_tokens(graph: SchemaGraph, path: PathSpec, bundle: ToolBundle) -> set[str]:
    tokens: set[str] = set()
    root_table = graph.get_table(path.root_table, schema_name=path.edges[0].source_schema)
    for column_name in root_table.primary_key:
        tokens.update(_identifier_tokens(column_name))
    for tool in bundle.tools:
        for parameter in tool.parameters:
            tokens.update(_identifier_tokens(parameter.name.removeprefix("anchor_")))
        for field_name in tool.output_fields:
            tokens.update(_identifier_tokens(field_name))
    return tokens


def _meaningful_tokens(value: str) -> set[str]:
    return {token for token in _identifier_tokens(value) if token not in _GENERIC_TOKENS}


def _identifier_tokens(value: str) -> set[str]:
    raw_tokens = [token for token in _TOKEN_PATTERN.findall(value.lower()) if token]
    tokens: set[str] = set()
    for token in raw_tokens:
        tokens.add(token)
        singular = singularize_token(token)
        tokens.add(singular)
    return tokens
