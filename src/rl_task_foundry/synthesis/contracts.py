"""Core contracts for the synthesis-agent rewrite."""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal

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


_WORD_SEPARATOR_RE = re.compile(r"[_\-\s]+")
_PERSON_LIKE_IDENTIFIER_ALIASES: tuple[str, ...] = (
    "customer",
    "user",
    "member",
    "patient",
    "guest",
    "client",
    "subscriber",
    "rider",
    "driver",
    "student",
    "teacher",
    "employee",
    "staff",
    "agent",
    "buyer",
    "seller",
    "owner",
    "passenger",
    "traveler",
    "account holder",
    "account_holder",
)


def normalize_topic(value: object) -> str:
    if isinstance(value, TopicName):
        return str(value)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    raise ValueError("topic must be a non-empty string")


def normalize_words(value: str, *, lowercase: bool = False) -> str:
    normalized = _WORD_SEPARATOR_RE.sub(" ", value).strip()
    if lowercase:
        return normalized.lower()
    return normalized


def topic_phrase(value: object, *, lowercase: bool = False) -> str:
    return normalize_words(normalize_topic(value), lowercase=lowercase)


def topic_tokens(value: object) -> tuple[str, ...]:
    normalized = topic_phrase(value, lowercase=True)
    return tuple(token for token in re.findall(r"[a-z0-9]+", normalized) if len(token) >= 3)


def is_person_like_identifier(value: str) -> bool:
    normalized = normalize_words(value, lowercase=True)
    if not normalized:
        return False
    tokens = tuple(token for token in normalized.split() if token)
    if not tokens:
        return False
    for alias in _PERSON_LIKE_IDENTIFIER_ALIASES:
        alias_tokens = tuple(token for token in normalize_words(alias, lowercase=True).split() if token)
        if alias_tokens and all(token in tokens for token in alias_tokens):
            return True
    return False


def entity_slug_from_get_tool_name(tool_name: str) -> str | None:
    normalized = tool_name.strip().lower()
    if normalized.startswith("get_") and normalized.endswith("_by_id"):
        return normalized[4:-6]
    if normalized.startswith("get_") and normalized.endswith("_by_ids_batch"):
        return normalized[4:-13]
    return None


class DifficultyAxis(StrEnum):
    SEARCH_COST = "search_cost"
    SOLUTION_SPACE = "solution_space"
    CONSTRAINT_DENSITY = "constraint_density"


class EnvironmentStatus(StrEnum):
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


class InstanceSamplingStrategy(StrEnum):
    DETERMINISTIC_HASH = "deterministic_hash"
    GRID = "grid"
    STRATIFIED_HASH = "stratified_hash"


class DifficultyVectorContract(StrictModel):
    search_cost: float = Field(default=0.0, ge=0.0)
    solution_space: float = Field(default=0.0, ge=0.0)
    constraint_density: float = Field(default=0.0, ge=0.0)

    def flatten(self) -> dict[DifficultyAxis, float]:
        return {
            DifficultyAxis.SEARCH_COST: float(self.search_cost),
            DifficultyAxis.SOLUTION_SPACE: float(self.solution_space),
            DifficultyAxis.CONSTRAINT_DENSITY: float(self.constraint_density),
        }

    def total_score(self) -> float:
        return sum(self.flatten().values())

    def nonzero_axes(self) -> dict[DifficultyAxis, float]:
        return {axis: value for axis, value in self.flatten().items() if value > 0.0}


DIFFICULTY_CRANK_ORDER: tuple[DifficultyAxis, ...] = (
    DifficultyAxis.SEARCH_COST,
    DifficultyAxis.SOLUTION_SPACE,
    DifficultyAxis.CONSTRAINT_DENSITY,
)


def flatten_difficulty_vector(
    difficulty_vector: DifficultyVectorContract,
) -> dict[DifficultyAxis, float]:
    return difficulty_vector.flatten()


def difficulty_vector_json(
    difficulty_vector: DifficultyVectorContract,
) -> dict[str, object]:
    return difficulty_vector.model_dump(mode="json")


def build_difficulty_vector(
    *,
    search_cost: float = 0.0,
    solution_space: float = 0.0,
    constraint_density: float = 0.0,
) -> DifficultyVectorContract:
    return DifficultyVectorContract(
        search_cost=search_cost,
        solution_space=solution_space,
        constraint_density=constraint_density,
    )


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
        elif self.items is not None:
            raise ValueError("only list output fields may declare items")
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
    difficulty_vector: DifficultyVectorContract = Field(default_factory=DifficultyVectorContract)
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


class AnchorQueryContract(StrictModel):
    sql: str = Field(min_length=1)
    outputs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_outputs(self) -> AnchorQueryContract:
        if not self.outputs:
            raise ValueError("anchor_query must declare at least one output column")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("anchor_query outputs must be unique")
        return self


ScalarParameterValue = str | int | float | bool


class EnumParameterSpace(StrictModel):
    kind: Literal["enum"] = "enum"
    values: list[ScalarParameterValue] = Field(min_length=1)


class IntRangeParameterSpace(StrictModel):
    kind: Literal["int_range"] = "int_range"
    min: int
    max: int
    step: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_bounds(self) -> IntRangeParameterSpace:
        if self.min > self.max:
            raise ValueError("int_range min must be <= max")
        return self


class FloatRangeParameterSpace(StrictModel):
    kind: Literal["float_range"] = "float_range"
    min: float
    max: float
    step: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> FloatRangeParameterSpace:
        if self.min > self.max:
            raise ValueError("float_range min must be <= max")
        return self


class DateRangeParameterSpace(StrictModel):
    kind: Literal["date_range"] = "date_range"
    start: date
    end: date
    step_days: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_bounds(self) -> DateRangeParameterSpace:
        if self.start > self.end:
            raise ValueError("date_range start must be <= end")
        return self


class DerivedBucketParameterSpace(StrictModel):
    kind: Literal["derived_bucket"] = "derived_bucket"
    source_key: str
    buckets: list[str] = Field(min_length=1)


ParameterSpace = Annotated[
    EnumParameterSpace
    | IntRangeParameterSpace
    | FloatRangeParameterSpace
    | DateRangeParameterSpace
    | DerivedBucketParameterSpace,
    Field(discriminator="kind"),
]


class InstanceSamplingContract(StrictModel):
    strategy: InstanceSamplingStrategy = InstanceSamplingStrategy.DETERMINISTIC_HASH
    seed: int = 0


class InstanceSpaceContract(StrictModel):
    anchor_query: AnchorQueryContract
    parameters: dict[str, ParameterSpace] = Field(default_factory=dict)
    sampling: InstanceSamplingContract = Field(default_factory=InstanceSamplingContract)
    instance_count: int | None = Field(default=None, ge=1)


class InstanceContract(StrictModel):
    instance_id: str
    anchor_values: dict[str, RuntimeValue] = Field(default_factory=dict)
    parameter_values: dict[str, RuntimeValue] = Field(default_factory=dict)
    expected_label_signature: str | None = None

    @property
    def expected_solution_fingerprint(self) -> str | None:
        return self.expected_label_signature


class CrossInstanceSet(StrictModel):
    minimum_required: int = Field(default=3, ge=1)
    require_distinct_label_signatures: bool = True
    instances: list[InstanceContract] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_instance_ids(self) -> CrossInstanceSet:
        instance_ids = [instance.instance_id for instance in self.instances]
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("cross-instance sets must not reuse instance_id values")
        if self.require_distinct_label_signatures:
            fingerprints = [
                instance.expected_label_signature
                for instance in self.instances
                if instance.expected_label_signature is not None
            ]
            if len(set(fingerprints)) != len(fingerprints):
                raise ValueError(
                    "cross-instance solution fingerprints must be distinct when required"
                )
        return self

    @property
    def require_distinct_solution_fingerprints(self) -> bool:
        return self.require_distinct_label_signatures


class EnvironmentQualityMetrics(StrictModel):
    solver_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    solver_ci_low: float | None = Field(default=None, ge=0.0, le=1.0)
    solver_ci_high: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_interval(self) -> EnvironmentQualityMetrics:
        if (self.solver_ci_low is None) != (self.solver_ci_high is None):
            raise ValueError("solver CI bounds must both be set or both be omitted")
        if (
            self.solver_ci_low is not None
            and self.solver_ci_high is not None
            and self.solver_ci_low > self.solver_ci_high
        ):
            raise ValueError("solver_ci_low must be <= solver_ci_high")
        return self


class EnvironmentContract(StrictModel):
    env_id: str
    db_id: str
    domain: str
    topic: str
    atomic_tool_set_ref: str
    difficulty_vector: DifficultyVectorContract
    created_at: datetime
    generator_version: str
    tool_signature: str
    task_signature: str
    status: EnvironmentStatus = EnvironmentStatus.DRAFT
    quality_metrics: EnvironmentQualityMetrics = Field(
        default_factory=EnvironmentQualityMetrics
    )
    rollout_constraints: RolloutConstraintsContract
    task: TaskContract
    instance_space: InstanceSpaceContract
    cross_instance_set: CrossInstanceSet = Field(default_factory=CrossInstanceSet)

    @model_validator(mode="after")
    def _validate_contract_consistency(self) -> EnvironmentContract:
        self.topic = normalize_topic(self.topic)
        if self.task.topic != self.topic:
            raise ValueError("environment topic must match task topic")
        if self.task.difficulty_vector != self.difficulty_vector:
            raise ValueError("environment difficulty_vector must match task difficulty_vector")
        return self

    @property
    def category(self) -> TopicName:
        return TopicName(self.topic)
