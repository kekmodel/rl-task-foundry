"""Canonical answer schema and ground truth contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnswerField(StrictModel):
    name: str
    type: Literal[
        "string",
        "int",
        "float",
        "bool",
        "date",
        "datetime",
        "enum",
        "list[string]",
        "list[int]",
    ]
    nullable: bool = False
    ordered: bool = False
    canonicalizer: str
    description: str = ""
    visibility: Literal["user_visible", "internal", "blocked"] = "user_visible"
    source_columns: list[str] = Field(default_factory=list)


class AnswerSchema(StrictModel):
    version: str = "v1"
    fields: list[AnswerField]
    primary_output_format: Literal["json_object"] = "json_object"


class GroundTruth(StrictModel):
    task_id: str
    verification_sql: str
    sql_params: dict[str, object] = Field(default_factory=dict)
    canonical_answer: dict[str, object]
    row_context: list[dict[str, object]] = Field(default_factory=list)
    answer_schema_version: str
    provenance_path: list[str] = Field(default_factory=list)
