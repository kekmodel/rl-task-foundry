"""Core task and solver result models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from rl_task_foundry.truth.schemas import AnswerSchema, GroundTruth


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskSpec(StrictModel):
    task_id: str
    anchor_table: str
    anchor_pk_column: str
    anchor_pk_value: str
    domain: str
    language: str
    label_tier: str
    question_family: str = "status_lookup"
    question: str
    question_source: Literal["seed_rule_based", "model_generated", "seed_fallback"] = "seed_rule_based"
    question_generation_metadata: dict[str, object] = Field(default_factory=dict)
    outcome_type: Literal["answer", "no_result", "clarify", "deny"] = "answer"
    answer_schema: AnswerSchema
    selected_path_id: str
    required_hops: int
    tool_level: Literal[1, 2] = 1
    tool_bundle_id: str
    presented_tool_bundle_id: str | None = None
    provenance_requirements: list[str] = Field(default_factory=list)
    difficulty_features: dict[str, int | float | str | bool] = Field(default_factory=dict)
    sensitivity_policy: str


class PresentedToolSpec(StrictModel):
    name: str
    description: str
    semantic_key: str
    kind: Literal["lookup", "list_related", "count", "exists", "aggregate", "timeline"]
    parameter_names: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    name_source: Literal["rule_based", "model_generated", "fallback_alias"]
    presentation_role: Literal["core", "distractor"] = "core"


class PresentedToolBundle(StrictModel):
    bundle_id: str
    canonical_bundle_id: str
    path_id: str
    tool_level: Literal[1, 2]
    question_family: str
    outcome_type: Literal["answer", "no_result", "clarify", "deny"]
    tools: list[PresentedToolSpec] = Field(default_factory=list)
    generation_metadata: dict[str, object] = Field(default_factory=dict)


class TaskPackage(StrictModel):
    task: TaskSpec
    presented_tool_bundle: PresentedToolBundle
    presentation_options: list[PresentedToolBundle] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def available_tool_levels(self) -> list[int]:
        levels = sorted({bundle.tool_level for bundle in self.presentation_options})
        if levels:
            return levels
        return [self.presented_tool_bundle.tool_level]


class SolverResult(StrictModel):
    task_id: str
    solver_id: str
    provider: str
    model: str
    replica_index: int
    transcript_ref: str
    tool_trace_ref: str
    raw_output_text: str
    structured_output: dict[str, object] | None = None
    explicit_memory_events: list[dict[str, object]] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
    turn_count: int = 0
    status: str
    termination_reason: str | None = None
    termination_metadata: dict[str, object] = Field(default_factory=dict)


class VerifyResult(StrictModel):
    task_id: str
    solver_id: str
    pass_exact: bool
    field_scores: dict[str, bool] = Field(default_factory=dict)
    provenance_pass: bool = False
    canonical_prediction: dict[str, object] | None = None
    shadow_verifier_status: Literal["not_run", "match", "disagree"] = "not_run"
    shadow_pass_exact: bool | None = None
    shadow_failure_reason: str | None = None
    failure_reason: str | None = None


class AcceptedExample(StrictModel):
    task: TaskSpec
    ground_truth: GroundTruth
    solver_results: list[SolverResult]
    verification_results: list[VerifyResult]
    pass_rate: float
    calibration_band: tuple[float, float]
    export_payload: dict[str, object]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_correct_solver_turns_rounded(self) -> int | None:
        turn_counts = [
            solver_result.turn_count
            for solver_result, verify_result in zip(
                self.solver_results,
                self.verification_results,
                strict=False,
            )
            if verify_result.pass_exact and solver_result.turn_count > 0
        ]
        if not turn_counts:
            return None
        return round(sum(turn_counts) / len(turn_counts))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def training_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {}
        if self.mean_correct_solver_turns_rounded is not None:
            metadata["mean_correct_solver_turns_rounded"] = self.mean_correct_solver_turns_rounded
        return metadata
