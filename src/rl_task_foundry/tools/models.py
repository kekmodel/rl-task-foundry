"""Tool specification models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class ToolParameter:
    name: str
    json_type: str
    description: str
    required: bool = True


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    sql_template: str
    parameters: list[ToolParameter]
    output_fields: list[str]
    path_id: str
    kind: Literal["lookup", "list_related", "count", "exists", "aggregate", "timeline"]
    tool_level: Literal[1, 2]
    semantic_key: str
    name_source: Literal["rule_based", "model_generated", "fallback_alias"]


@dataclass(slots=True)
class ToolBundle:
    bundle_id: str
    path_id: str
    tool_level: Literal[1, 2]
    tools: list[ToolSpec] = field(default_factory=list)
