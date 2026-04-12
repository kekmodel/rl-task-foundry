"""Core contracts for the synthesis-agent rewrite."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model with strict extra-field rejection."""

    model_config = ConfigDict(extra="forbid")


class CategoryTaxonomy(StrEnum):
    ITINERARY = "itinerary"
    ASSIGNMENT = "assignment"
    BUNDLE_SELECTION = "bundle_selection"
    ELIGIBILITY_FILTER = "eligibility_filter"
    NO_REPEAT_RECOMMENDATION = "no_repeat_recommendation"
    THRESHOLD_ROUTING = "threshold_routing"
    TEMPORAL_PLANNING = "temporal_planning"
    OTHER = "other"


class DifficultyAxis(StrEnum):
    SEARCH_COST = "search_cost"
    SOLUTION_SPACE = "solution_space"
    CONSTRAINT_DENSITY = "constraint_density"


class DifficultyAxisUnit(StrEnum):
    SCORE = "score"


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


class ToolParameterType(StrEnum):
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"
    ENUM = "enum"
    LIST_STRING = "list[string]"
    LIST_INT = "list[int]"
    LIST_FLOAT = "list[float]"
    OBJECT = "object"


class ToolEmptyResultBehavior(StrEnum):
    RETURN_EMPTY = "return_empty"
    RETURN_NULL = "return_null"
    RAISE_NOT_FOUND = "raise_not_found"


class ToolTimeoutBehavior(StrEnum):
    RAISE_TIMEOUT = "raise_timeout"
    RETURN_ERROR = "return_error"


class ToolSelfTestCheck(StrEnum):
    HAPPY_PATH = "happy_path"
    EMPTY_RESULT_BEHAVIOR = "empty_result_behavior"
    TIMEOUT_BEHAVIOR = "timeout_behavior"
    DETERMINISTIC_ORDERING = "deterministic_ordering"


class FactCardinality(StrEnum):
    ONE = "one"
    MANY = "many"


class FactValueType(StrEnum):
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"
    LIST_STR = "list[str]"
    LIST_INT = "list[int]"
    LIST_FLOAT = "list[float]"


class ShadowPromptStrategy(StrEnum):
    TOP_DOWN = "top_down"
    BOTTOM_UP = "bottom_up"


class ShadowIndependenceRequirement(StrEnum):
    SEPARATE_SESSION = "separate_session"
    SEPARATE_PROMPT_TEMPLATE = "separate_prompt_template"
    SEPARATE_TEMPERATURE = "separate_temperature"
    SEPARATE_MODEL_FAMILY_PREFERRED = "separate_model_family_preferred"


class InstanceSamplingStrategy(StrEnum):
    DETERMINISTIC_HASH = "deterministic_hash"
    GRID = "grid"
    STRATIFIED_HASH = "stratified_hash"


class DifficultyAxisSpec(StrictModel):
    axis: DifficultyAxis
    unit: DifficultyAxisUnit
    crank_rank: int
    description: str
    minimum: float = 0.0
    maximum: float | None = None


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


DIFFICULTY_AXIS_SPECS: dict[DifficultyAxis, DifficultyAxisSpec] = {
    DifficultyAxis.SEARCH_COST: DifficultyAxisSpec(
        axis=DifficultyAxis.SEARCH_COST,
        unit=DifficultyAxisUnit.SCORE,
        crank_rank=1,
        description="Tool-call search cost required to gather the DB facts needed for the answer.",
        minimum=0.0,
    ),
    DifficultyAxis.SOLUTION_SPACE: DifficultyAxisSpec(
        axis=DifficultyAxis.SOLUTION_SPACE,
        unit=DifficultyAxisUnit.SCORE,
        crank_rank=2,
        description="Size of the candidate solution space that must be explored or composed.",
        minimum=0.0,
    ),
    DifficultyAxis.CONSTRAINT_DENSITY: DifficultyAxisSpec(
        axis=DifficultyAxis.CONSTRAINT_DENSITY,
        unit=DifficultyAxisUnit.SCORE,
        crank_rank=3,
        description="Inverse density of valid solutions after applying hard constraints.",
        minimum=0.0,
    ),
}

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


class ToolParameterContract(StrictModel):
    name: str
    type: ToolParameterType
    description: str = ""
    required: bool = True
    enum_values: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_enum_values(self) -> ToolParameterContract:
        if self.type == ToolParameterType.ENUM and not self.enum_values:
            raise ValueError("enum tool parameters must declare enum_values")
        if self.type != ToolParameterType.ENUM and self.enum_values:
            raise ValueError("enum_values are only allowed for enum tool parameters")
        return self


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


class ToolContract(StrictModel):
    name: str
    description: str = ""
    parameters: list[ToolParameterContract] = Field(default_factory=list)
    return_schema: OutputFieldContract
    async_callable: Literal[True] = True
    connection_parameter: Literal["conn"] = "conn"
    read_only: Literal[True] = True
    empty_result_behavior: ToolEmptyResultBehavior = ToolEmptyResultBehavior.RETURN_EMPTY
    timeout_behavior: ToolTimeoutBehavior = ToolTimeoutBehavior.RAISE_TIMEOUT


class TaskContract(StrictModel):
    question: str
    category: CategoryTaxonomy
    output_schema: OutputSchemaContract
    constraint_summary: list[ConstraintSummaryItem] = Field(default_factory=list)
    difficulty_vector: DifficultyVectorContract = Field(default_factory=DifficultyVectorContract)
    instance_parameters: dict[str, RuntimeValue] = Field(default_factory=dict)


class SolutionContract(StrictModel):
    entrypoint: Literal["solve"] = "solve"
    role: Literal["oracle_reference"] = "oracle_reference"
    visible_to_solver: Literal[False] = False


class ToolSelfTestContract(StrictModel):
    entrypoint: Literal["run_self_test"] = "run_self_test"
    role: Literal["registration_gate"] = "registration_gate"
    async_entrypoint: Literal[True] = True
    visible_to_solver: Literal[False] = False
    required_checks: list[ToolSelfTestCheck] = Field(
        default_factory=lambda: [
            ToolSelfTestCheck.HAPPY_PATH,
            ToolSelfTestCheck.EMPTY_RESULT_BEHAVIOR,
            ToolSelfTestCheck.TIMEOUT_BEHAVIOR,
            ToolSelfTestCheck.DETERMINISTIC_ORDERING,
        ]
    )


class FactSpec(StrictModel):
    key: str
    entity_ref: str
    attribute: str
    value_type: FactValueType
    nullable: bool = False
    cardinality: FactCardinality = FactCardinality.ONE


class MaterializedFactsSchema(StrictModel):
    facts: list[FactSpec] = Field(default_factory=list)


class VerifierContract(StrictModel):
    entrypoint: Literal["verify"] = "verify"
    fetch_facts_function: Literal["fetch_facts"] = "fetch_facts"
    facts_match_function: Literal["facts_match_answer_claims"] = "facts_match_answer_claims"
    check_constraints_function: Literal["check_constraints"] = "check_constraints"
    facts_schema: MaterializedFactsSchema
    official_judgment: Literal[True] = True


class ShadowVerifierContract(VerifierContract):
    official_judgment: Literal[False] = False
    prompt_strategy: ShadowPromptStrategy = ShadowPromptStrategy.BOTTOM_UP
    independence_requirements: list[ShadowIndependenceRequirement] = Field(
        default_factory=lambda: [
            ShadowIndependenceRequirement.SEPARATE_SESSION,
            ShadowIndependenceRequirement.SEPARATE_PROMPT_TEMPLATE,
            ShadowIndependenceRequirement.SEPARATE_TEMPERATURE,
            ShadowIndependenceRequirement.SEPARATE_MODEL_FAMILY_PREFERRED,
        ]
    )


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
    expected_solution_fingerprint: str | None = None


class CrossInstanceSet(StrictModel):
    minimum_required: int = Field(default=3, ge=1)
    require_distinct_solution_fingerprints: bool = True
    instances: list[InstanceContract] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_instance_ids(self) -> CrossInstanceSet:
        instance_ids = [instance.instance_id for instance in self.instances]
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("cross-instance sets must not reuse instance_id values")
        if self.require_distinct_solution_fingerprints:
            fingerprints = [
                instance.expected_solution_fingerprint
                for instance in self.instances
                if instance.expected_solution_fingerprint is not None
            ]
            if len(set(fingerprints)) != len(fingerprints):
                raise ValueError(
                    "cross-instance solution fingerprints must be distinct when required"
                )
        return self


class EnvironmentQualityMetrics(StrictModel):
    self_consistency_pass: bool = False
    shadow_disagreement_rate: float | None = Field(default=None, ge=0.0, le=1.0)
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
    category: CategoryTaxonomy
    atomic_tool_set_ref: str
    difficulty_vector: DifficultyVectorContract
    created_at: datetime
    generator_version: str
    tool_signature: str
    task_signature: str
    verifier_signature: str
    status: EnvironmentStatus = EnvironmentStatus.DRAFT
    quality_metrics: EnvironmentQualityMetrics = Field(
        default_factory=EnvironmentQualityMetrics
    )
    rollout_constraints: RolloutConstraintsContract
    task: TaskContract
    solution: SolutionContract
    verifier: VerifierContract
    shadow_verifier: ShadowVerifierContract
    instance_space: InstanceSpaceContract
    cross_instance_set: CrossInstanceSet = Field(default_factory=CrossInstanceSet)

    @model_validator(mode="after")
    def _validate_contract_consistency(self) -> EnvironmentContract:
        if self.task.category != self.category:
            raise ValueError("environment category must match task category")
        if self.task.difficulty_vector != self.difficulty_vector:
            raise ValueError("environment difficulty_vector must match task difficulty_vector")
        return self
