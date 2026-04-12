"""submit_draft tool for the single-agent synthesis loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import Field, ValidationError

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.contracts import (
    DIFFICULTY_CRANK_ORDER,
    DifficultyAxis,
    DifficultyVectorContract,
    InstanceSpaceContract,
    StrictModel,
    flatten_difficulty_vector,
)
from rl_task_foundry.synthesis.phase_monitor import PipelinePhaseMonitorLogger

if TYPE_CHECKING:
    from rl_task_foundry.pipeline.environment_orchestrator import EnvironmentOrchestrator
    from rl_task_foundry.synthesis.runtime import SynthesisEnvironmentDraft


_FORBIDDEN_PLACEHOLDER_TOKENS = (
    "__REAL_",
    "anchor_id",
    "anchor_table",
    "example_field",
    "replace_with_real_",
)


class SubmitDraftPayload(StrictModel):
    canonical_answer_json: str = Field(
        min_length=1,
        description="Canonical answer as a JSON string. This is the exact label used for EM scoring.",
    )
    anchor_entity: dict[str, object] = Field(
        description=(
            "Mandatory anchor entity with at least one real primary-key value from the current database. "
            'Example: {"customer_id": 148} or {"store_id": 1}. Never omit this field.'
        )
    )
    difficulty_vector: DifficultyVectorContract = Field(
        description="Declared difficulty levels for search_cost, solution_space, and constraint_density."
    )
    question: str = Field(
        min_length=1,
        description="Natural user-facing request in the configured task language. Do not mention internal tool paths.",
    )
    constraint_summary: list["SubmitConstraintSummaryItem"] = Field(
        default_factory=list,
        description="List of grounded hard constraints or tie-break rules expressed in plain language.",
    )
    instance_space: InstanceSpaceContract = Field(
        description="Anchor query and sampling plan used to materialize runtime instances."
    )
    label_summary: str = Field(
        min_length=1,
        description="Short English summary of why the canonical answer is grounded and unique.",
    )


class SubmitConstraintSummaryItem(StrictModel):
    key: str = Field(description="Stable identifier for this constraint.")
    kind: str = Field(description="Constraint kind such as ordering, uniqueness, membership, or other.")
    summary: str = Field(description="Natural-language summary of the constraint.")
    hard: bool = Field(default=True, description="Whether this constraint is mandatory.")


SubmitDraftPayload.model_rebuild()


@dataclass(frozen=True, slots=True)
class SubmitDraftAttemptRecord:
    index: int
    outcome: str
    message: str
    error_codes: tuple[str, ...] = ()
    pass_rate: float | None = None
    matched_solver_runs: int | None = None
    total_solver_runs: int | None = None


def _next_difficulty_crank_axis(history: list[DifficultyAxis]) -> DifficultyAxis:
    if not history:
        return DifficultyAxis.SEARCH_COST
    last_axis = history[-1]
    repeat_count = history.count(last_axis)
    if repeat_count < 2:
        return last_axis
    last_index = DIFFICULTY_CRANK_ORDER.index(last_axis)
    if last_index + 1 < len(DIFFICULTY_CRANK_ORDER):
        return DIFFICULTY_CRANK_ORDER[last_index + 1]
    return last_axis


def _merge_strongest_difficulty_vector(
    previous: DifficultyVectorContract,
    current: DifficultyVectorContract,
) -> DifficultyVectorContract:
    return DifficultyVectorContract(
        search_cost=max(previous.search_cost, current.search_cost),
        solution_space=max(previous.solution_space, current.solution_space),
        constraint_density=max(previous.constraint_density, current.constraint_density),
    )


def _weakened_difficulty_axes(
    *,
    previous: DifficultyVectorContract,
    current: DifficultyVectorContract,
) -> list[str]:
    weakened: list[str] = []
    previous_flat = flatten_difficulty_vector(previous)
    current_flat = flatten_difficulty_vector(current)
    for axis, previous_value in previous_flat.items():
        if current_flat.get(axis, 0.0) < previous_value:
            weakened.append(axis.value)
    return weakened


def _difficulty_axis_hint(axis: DifficultyAxis) -> str:
    if axis is DifficultyAxis.SEARCH_COST:
        return (
            "Increase search_cost by requiring a longer evidence path, a deeper join chain, "
            "or more grounded exploration before the final answer is obvious."
        )
    if axis is DifficultyAxis.SOLUTION_SPACE:
        return (
            "Increase solution_space by asking for a larger or ordered answer set, more answer fields, "
            "or a choice among multiple grounded candidates."
        )
    return (
        "Increase constraint_density by adding more hard conditions, stronger tie-breakers, "
        "or tighter grounded constraints that reduce the valid answer set."
    )


def _placeholder_tokens(payload: object) -> list[str]:
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    return sorted({token for token in _FORBIDDEN_PLACEHOLDER_TOKENS if token in serialized})


def _preview_payload(value: object) -> object:
    if isinstance(value, str):
        return value[:400]
    if isinstance(value, list):
        return [_preview_payload(item) for item in value[:3]]
    if isinstance(value, dict):
        preview: dict[str, object] = {}
        for key, item in list(value.items())[:6]:
            preview[str(key)] = _preview_payload(item)
        return preview
    return value


@dataclass(slots=True)
class SubmitDraftController:
    config: AppConfig
    requested_topic: str
    environment_orchestrator: EnvironmentOrchestrator
    build_draft: Any
    phase_monitor: PipelinePhaseMonitorLogger | None = None
    max_submissions: int = 5
    accepted_draft: SynthesisEnvironmentDraft | None = None
    attempts: list[SubmitDraftAttemptRecord] = field(default_factory=list)
    strongest_difficulty_vector: DifficultyVectorContract = field(
        default_factory=DifficultyVectorContract
    )
    difficulty_crank_history: list[DifficultyAxis] = field(default_factory=list)
    required_axis: DifficultyAxis | None = None
    last_quality_gate_status: str | None = None
    last_quality_gate_pass_rate: float | None = None
    _atomic_tool_calls: list[dict[str, object]] = field(default_factory=list, init=False)
    _tool_call_count_at_last_submission: int = field(default=0, init=False)

    def submissions_left(self) -> int:
        return max(0, self.max_submissions - len(self.attempts))

    def record_atomic_tool_call(
        self,
        *,
        tool_name: str,
        params: dict[str, object],
        result: object,
    ) -> None:
        self._atomic_tool_calls.append(
            {
                "tool_name": tool_name,
                "params": _preview_payload(params),
                "result": _preview_payload(result),
            }
        )

    def reject_invalid_payload(self, *, parsed: dict[str, object], error: ValidationError) -> str:
        if self.accepted_draft is not None:
            return "Accepted. Draft already stored."
        if self.submissions_left() <= 0:
            return "Budget exhausted. No more attempts."

        error_codes: list[str] = []
        for error_item in error.errors():
            location = tuple(str(part) for part in error_item.get("loc", ()))
            if location == ("anchor_entity",):
                error_codes.append("anchor_entity_required")
            else:
                error_codes.append("submit_payload_invalid")
        deduped_error_codes = list(dict.fromkeys(error_codes)) or ["submit_payload_invalid"]
        return self._record_rejection(
            submission_index=len(self.attempts) + 1,
            message=self._invalid_submission_message(deduped_error_codes),
            error_codes=deduped_error_codes,
            payload=parsed,
            diagnostics={
                "validation_errors": [
                    {
                        "loc": [str(part) for part in error_item.get("loc", ())],
                        "type": str(error_item.get("type", "")),
                    }
                    for error_item in error.errors()
                ]
            },
        )

    async def submit(self, payload: SubmitDraftPayload) -> str:
        if self.accepted_draft is not None:
            return "Accepted. Draft already stored."
        if self.submissions_left() <= 0:
            return "Budget exhausted. No more attempts."

        submission_index = len(self.attempts) + 1
        error_codes: list[str] = []

        if len(self._atomic_tool_calls) <= self._tool_call_count_at_last_submission:
            error_codes.append("no_new_grounded_observation")
        if not payload.anchor_entity:
            error_codes.append("anchor_entity_required")
        placeholder_tokens = _placeholder_tokens(
            {
                "canonical_answer_json": payload.canonical_answer_json,
                "anchor_entity": payload.anchor_entity,
                "question": payload.question,
                "label_summary": payload.label_summary,
                "constraint_summary": [item.model_dump(mode="json") for item in payload.constraint_summary],
                "instance_space": payload.instance_space.model_dump(mode="json"),
            }
        )
        if placeholder_tokens:
            error_codes.append("placeholder_tokens_not_allowed")

        weakened_axes = _weakened_difficulty_axes(
            previous=self.strongest_difficulty_vector,
            current=payload.difficulty_vector,
        )
        if weakened_axes:
            error_codes.append("difficulty_weakened")
        if self.required_axis is not None:
            current_axis_value = getattr(payload.difficulty_vector, self.required_axis.value)
            strongest_axis_value = getattr(
                self.strongest_difficulty_vector, self.required_axis.value
            )
            if current_axis_value <= strongest_axis_value:
                error_codes.append("required_difficulty_axis_not_strengthened")

        if error_codes:
            return self._record_rejection(
                submission_index=submission_index,
                message=self._invalid_submission_message(error_codes),
                error_codes=error_codes,
                payload=payload,
            )

        try:
            draft = self.build_draft(payload)
        except Exception as exc:
            return self._record_rejection(
                submission_index=submission_index,
                message=f"Rejected. Invalid draft: {type(exc).__name__}: {exc}",
                error_codes=["draft_validation_failed"],
                payload=payload,
            )

        rollout_summary = await self.environment_orchestrator.run_draft(draft)
        from rl_task_foundry.pipeline.environment_orchestrator import (
            EnvironmentQualityGateStatus,
            evaluate_rollout_summary,
        )
        from rl_task_foundry.synthesis.quality_gate import accepted_draft_with_quality_metrics

        quality_gate_summary = evaluate_rollout_summary(self.config, rollout_summary)
        self.last_quality_gate_status = quality_gate_summary.status.value
        self.last_quality_gate_pass_rate = quality_gate_summary.pass_rate
        self._tool_call_count_at_last_submission = len(self._atomic_tool_calls)

        if quality_gate_summary.status is EnvironmentQualityGateStatus.ACCEPT:
            accepted_draft = accepted_draft_with_quality_metrics(
                draft,
                quality_gate_summary=quality_gate_summary,
            )
            self.accepted_draft = accepted_draft
            self.attempts.append(
                SubmitDraftAttemptRecord(
                    index=submission_index,
                    outcome="accepted",
                    message="accepted",
                    pass_rate=quality_gate_summary.pass_rate,
                    matched_solver_runs=quality_gate_summary.matched_solver_runs,
                    total_solver_runs=quality_gate_summary.total_solver_runs,
                )
            )
            self._emit_monitor(
                status="accepted",
                payload=payload,
                pass_rate=quality_gate_summary.pass_rate,
                matched_solver_runs=quality_gate_summary.matched_solver_runs,
                total_solver_runs=quality_gate_summary.total_solver_runs,
                diagnostics={},
            )
            return (
                "Accepted. solver pass rate "
                f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}."
            )

        attempts_left_after = self.submissions_left() - 1
        if quality_gate_summary.status is EnvironmentQualityGateStatus.REJECT_TOO_EASY:
            requested_axis = _next_difficulty_crank_axis(self.difficulty_crank_history)
            self.strongest_difficulty_vector = _merge_strongest_difficulty_vector(
                self.strongest_difficulty_vector,
                payload.difficulty_vector,
            )
            self.required_axis = requested_axis
            self.difficulty_crank_history.append(requested_axis)
            return self._record_rejection(
                submission_index=submission_index,
                message=(
                    "Rejected. solver pass rate "
                    f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}. "
                    f"Crank {requested_axis.value}. {_difficulty_axis_hint(requested_axis)} "
                    f"Make at least one new atomic tool call, gather new grounded evidence, and strengthen that axis before resubmitting. "
                    f"{max(0, attempts_left_after)} attempts left."
                ),
                error_codes=["reject_too_easy"],
                pass_rate=quality_gate_summary.pass_rate,
                matched_solver_runs=quality_gate_summary.matched_solver_runs,
                total_solver_runs=quality_gate_summary.total_solver_runs,
                diagnostics={"requested_axis": requested_axis.value},
                payload=payload,
            )

        self.required_axis = None
        return self._record_rejection(
            submission_index=submission_index,
            message=(
                "Rejected. solver pass rate "
                f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}. "
                f"Choose a different grounded anchor and resubmit. {max(0, attempts_left_after)} attempts left."
            ),
            error_codes=["reject_too_hard"],
            pass_rate=quality_gate_summary.pass_rate,
            matched_solver_runs=quality_gate_summary.matched_solver_runs,
            total_solver_runs=quality_gate_summary.total_solver_runs,
            diagnostics={},
            payload=payload,
        )

    def _invalid_submission_message(self, error_codes: list[str]) -> str:
        message_map = {
            "no_new_grounded_observation": (
                "Rejected. Observe more real database facts with atomic tools before resubmitting."
            ),
            "anchor_entity_required": "Rejected. anchor_entity must contain at least one primary-key value.",
            "placeholder_tokens_not_allowed": (
                "Rejected. Replace every placeholder token with grounded names and values from the current database."
            ),
            "difficulty_weakened": (
                "Rejected. Do not weaken the declared difficulty vector relative to the strongest prior attempt."
            ),
            "required_difficulty_axis_not_strengthened": (
                "Rejected. The requested difficulty axis was not strengthened."
            ),
            "submit_payload_invalid": (
                "Rejected. submit_draft arguments did not match the required schema."
            ),
            "draft_validation_failed": "Rejected. The submitted draft could not be validated.",
        }
        primary = message_map.get(error_codes[0], "Rejected. Fix the draft and resubmit.")
        attempts_left_after = self.submissions_left() - 1
        return f"{primary} {max(0, attempts_left_after)} attempts left."

    def _record_rejection(
        self,
        *,
        submission_index: int,
        message: str,
        error_codes: list[str],
        payload: SubmitDraftPayload | None = None,
        pass_rate: float | None = None,
        matched_solver_runs: int | None = None,
        total_solver_runs: int | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> str:
        attempts_left_after = self.submissions_left() - 1
        self.attempts.append(
            SubmitDraftAttemptRecord(
                index=submission_index,
                outcome=error_codes[0] if error_codes else "rejected",
                message=message,
                error_codes=tuple(error_codes),
                pass_rate=pass_rate,
                matched_solver_runs=matched_solver_runs,
                total_solver_runs=total_solver_runs,
            )
        )
        status = "budget_exhausted" if attempts_left_after <= 0 else "rejected"
        self._emit_monitor(
            status=status,
            payload=payload,
            pass_rate=pass_rate,
            matched_solver_runs=matched_solver_runs,
            total_solver_runs=total_solver_runs,
            diagnostics={"error_codes": error_codes, **(diagnostics or {})},
        )
        if attempts_left_after <= 0:
            return "Budget exhausted. No more attempts."
        return message

    def _emit_monitor(
        self,
        *,
        status: str,
        payload: SubmitDraftPayload | dict[str, object] | None,
        pass_rate: float | None,
        matched_solver_runs: int | None,
        total_solver_runs: int | None,
        diagnostics: dict[str, object],
    ) -> None:
        if self.phase_monitor is None:
            return
        self.phase_monitor.emit(
            phase="submit_draft",
            status=status,
            expected_contract={
                "requested_topic": self.requested_topic,
                "max_submissions": self.max_submissions,
            },
            actual_data={
                "submission_index": len(self.attempts) + (0 if status == "accepted" else 0),
                "question": (
                    payload.question
                    if isinstance(payload, SubmitDraftPayload)
                    else payload.get("question")
                    if isinstance(payload, dict)
                    else None
                ),
                "anchor_entity": (
                    payload.anchor_entity
                    if isinstance(payload, SubmitDraftPayload)
                    else payload.get("anchor_entity")
                    if isinstance(payload, dict)
                    else None
                ),
                "difficulty_vector": (
                    payload.difficulty_vector.model_dump(mode="json")
                    if isinstance(payload, SubmitDraftPayload)
                    else payload.get("difficulty_vector")
                    if isinstance(payload, dict)
                    else None
                ),
                "pass_rate": pass_rate,
                "matched_solver_runs": matched_solver_runs,
                "total_solver_runs": total_solver_runs,
            },
            checks={
                "accepted": status == "accepted",
                "atomic_tool_calls_seen": len(self._atomic_tool_calls),
            },
            diagnostics={
                "required_axis": self.required_axis.value if self.required_axis is not None else None,
                "difficulty_crank_history": [axis.value for axis in self.difficulty_crank_history],
                "recent_tool_calls": self._atomic_tool_calls[-5:],
                **diagnostics,
            },
        )


def build_submit_draft_sdk_tool(controller: SubmitDraftController) -> object:
    from agents import FunctionTool

    params_json_schema = SubmitDraftPayload.model_json_schema()

    async def _invoke_tool(_tool_context: Any, input_json: str) -> str:
        parsed = json.loads(input_json) if input_json else {}
        try:
            payload = SubmitDraftPayload.model_validate(parsed)
        except ValidationError as exc:
            return controller.reject_invalid_payload(parsed=parsed, error=exc)
        return await controller.submit(payload)

    return FunctionTool(
        name="submit_draft",
        description=(
            "Submit a grounded RLVR task draft after inspecting real database rows. "
            "Include the canonical answer JSON, anchor entity, declared difficulty vector, "
            "natural user-facing question, constraint summary, instance space, and label summary. "
            "anchor_entity is mandatory and must include at least one real primary-key value, "
            "for example {\"customer_id\": 148} or {\"store_id\": 1}. "
            "After any rejection, make at least one new atomic tool call before calling submit_draft again."
        ),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=False,
    )
