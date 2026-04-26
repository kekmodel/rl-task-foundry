"""Core contracts for the synthesis-agent rewrite."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from rl_task_foundry.config.models import StrictModel


class TopicName(str):
    """Compatibility helper for older `.value` call sites while topics are plain strings."""

    @property
    def value(self) -> str:
        return str(self)


class CategoryTaxonomy:
    """Compatibility constants only. Topics are plain strings, not enums."""

    ITINERARY = TopicName("itinerary")
    ASSIGNMENT = TopicName("assignment")
    BUNDLE_SELECTION = TopicName("bundle_selection")
    ELIGIBILITY_FILTER = TopicName("eligibility_filter")
    NO_REPEAT_RECOMMENDATION = TopicName("no_repeat_recommendation")
    THRESHOLD_ROUTING = TopicName("threshold_routing")
    TEMPORAL_PLANNING = TopicName("temporal_planning")
    OTHER = TopicName("other")


def normalize_topic(value: object) -> str:
    if isinstance(value, TopicName):
        return str(value)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    raise ValueError("topic must be a non-empty string")


def normalize_words(value: str, *, lowercase: bool = False) -> str:
    normalized = value.strip()
    if lowercase:
        return normalized.lower()
    return normalized


def topic_phrase(value: object, *, lowercase: bool = False) -> str:
    return normalize_words(normalize_topic(value), lowercase=lowercase)


class TaskBundleStatus(StrEnum):
    DRAFT = "draft"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ARCHIVED = "archived"
    LEGACY = "legacy"


class ConstraintKind(StrEnum):
    UNIQUENESS = "uniqueness"
    MEMBERSHIP = "membership"
    RANGE = "range"
    CONDITIONAL = "conditional"
    CARDINALITY = "cardinality"
    TEMPORAL = "temporal"
    OTHER = "other"


class OutputFieldType(StrEnum):
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"
    ENUM = "enum"
    OBJECT = "object"
    LIST = "list"


ScalarRuntimeValue = str | int | float | bool | date | datetime | None
RuntimeValue = ScalarRuntimeValue | list[ScalarRuntimeValue]


class OutputFieldContract(StrictModel):
    name: str
    type: OutputFieldType
    description: str = ""
    nullable: bool = False
    ordered: bool = False
    sort_key: tuple[str, ...] | None = None
    unique_elements: bool = False
    length: int | None = None
    enum_values: list[str] = Field(default_factory=list)
    fields: list[OutputFieldContract] = Field(default_factory=list)
    items: OutputFieldContract | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> OutputFieldContract:
        if self.type == OutputFieldType.ENUM and not self.enum_values:
            raise ValueError("enum output fields must declare enum_values")
        if self.type != OutputFieldType.ENUM and self.enum_values:
            raise ValueError("enum_values are only allowed for enum output fields")
        if self.type == OutputFieldType.OBJECT:
            if not self.fields:
                raise ValueError("object output fields must declare nested fields")
            if self.items is not None:
                raise ValueError("object output fields cannot declare list items")
        elif self.fields:
            raise ValueError("only object output fields may declare nested fields")
        if self.type == OutputFieldType.LIST:
            if self.items is None:
                raise ValueError("list output fields must declare items")
            if self.length is not None and self.length < 0:
                raise ValueError("list output field length must be non-negative")
        elif self.items is not None:
            raise ValueError("only list output fields may declare items")
        if self.type != OutputFieldType.LIST and self.length is not None:
            raise ValueError("length is only allowed for list output fields")
        if self.sort_key is not None:
            if self.type != OutputFieldType.LIST:
                raise ValueError("sort_key is only allowed for list output fields")
            if self.ordered:
                raise ValueError("ordered list output fields cannot declare sort_key")
            if self.items is None or self.items.type != OutputFieldType.OBJECT:
                raise ValueError("sort_key requires list items to be objects")
            item_fields = {field.name: field for field in self.items.fields}
            primitive_types = {
                OutputFieldType.STRING,
                OutputFieldType.INT,
                OutputFieldType.FLOAT,
                OutputFieldType.BOOL,
                OutputFieldType.DATE,
                OutputFieldType.DATETIME,
                OutputFieldType.ENUM,
            }
            for key_part in self.sort_key:
                if key_part not in item_fields:
                    raise ValueError(f"sort_key references unknown field: {key_part}")
                key_field = item_fields[key_part]
                if key_field.type not in primitive_types:
                    raise ValueError("sort_key components must reference primitive item fields")
                if key_field.nullable:
                    raise ValueError("sort_key components must reference non-nullable item fields")
        if self.unique_elements:
            if self.type != OutputFieldType.LIST:
                raise ValueError("unique_elements is only allowed for list output fields")
            if self.ordered:
                raise ValueError("ordered list output fields cannot declare unique_elements")
        if (
            self.type == OutputFieldType.LIST
            and not self.ordered
            and self.items is not None
            and self.items.type == OutputFieldType.OBJECT
            and self.sort_key is None
        ):
            raise ValueError("unordered list of objects must declare sort_key")
        return self


OutputFieldContract.model_rebuild()


class OutputSchemaContract(StrictModel):
    version: Literal["v2"] = "v2"
    root: OutputFieldContract
    primary_output_format: Literal["json_object", "json_array"] = "json_object"

    @model_validator(mode="after")
    def _validate_output_format(self) -> OutputSchemaContract:
        if self.root.type == OutputFieldType.LIST and self.primary_output_format != "json_array":
            raise ValueError("list roots must use json_array output format")
        if self.root.type != OutputFieldType.LIST and self.primary_output_format != "json_object":
            raise ValueError("non-list roots must use json_object output format")
        return self


class ConstraintSummaryItem(StrictModel):
    key: str
    kind: ConstraintKind
    summary: str
    hard: bool = True
    representation_role: Literal["review_summary"] = "review_summary"


class TaskContract(StrictModel):
    question: str
    topic: str
    output_schema: OutputSchemaContract
    constraint_summary: list[ConstraintSummaryItem] = Field(default_factory=list)
    instance_parameters: dict[str, RuntimeValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_topic(self) -> TaskContract:
        self.topic = normalize_topic(self.topic)
        return self

    @property
    def category(self) -> TopicName:
        return TopicName(self.topic)


class RolloutConstraintsContract(StrictModel):
    max_turns: int = Field(ge=1)
    max_episode_duration_ms: int = Field(ge=1)
    max_tool_rows: int = Field(default=1000, ge=1)


class TaskQualityMetrics(StrictModel):
    solver_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    solver_ci_low: float | None = Field(default=None, ge=0.0, le=1.0)
    solver_ci_high: float | None = Field(default=None, ge=0.0, le=1.0)
    solver_planned_runs: int | None = Field(default=None, ge=0)
    solver_completed_runs: int | None = Field(default=None, ge=0)
    solver_evaluable_runs: int | None = Field(default=None, ge=0)
    solver_failed_runs: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_interval(self) -> TaskQualityMetrics:
        if (self.solver_ci_low is None) != (self.solver_ci_high is None):
            raise ValueError("solver CI bounds must both be set or both be omitted")
        if (
            self.solver_ci_low is not None
            and self.solver_ci_high is not None
            and self.solver_ci_low > self.solver_ci_high
        ):
            raise ValueError("solver_ci_low must be <= solver_ci_high")
        return self


class TaskBundleContract(StrictModel):
    task_id: str
    db_id: str
    domain: str
    topic: str
    atomic_tool_set_ref: str
    created_at: datetime
    generator_version: str
    tool_signature: str
    task_signature: str
    status: TaskBundleStatus = TaskBundleStatus.DRAFT
    quality_metrics: TaskQualityMetrics = Field(default_factory=TaskQualityMetrics)
    rollout_constraints: RolloutConstraintsContract
    task: TaskContract

    @model_validator(mode="after")
    def _validate_contract_consistency(self) -> TaskBundleContract:
        self.topic = normalize_topic(self.topic)
        if self.task.topic != self.topic:
            raise ValueError("task bundle topic must match task topic")
        return self

    @property
    def category(self) -> TopicName:
        return TopicName(self.topic)
