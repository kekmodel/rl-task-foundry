"""submit_draft tool for the single-agent synthesis loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import Field, ValidationError, field_validator

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
        min_length=1,
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
        min_length=1,
        description="List of grounded hard constraints or tie-break rules expressed in plain language.",
    )
    instance_space: InstanceSpaceContract = Field(
        description="Anchor query and sampling plan used to materialize runtime instances."
    )
    label_summary: str = Field(
        min_length=1,
        description="Short English summary of why the canonical answer is grounded and unique.",
    )

    @field_validator("canonical_answer_json")
    @classmethod
    def _validate_canonical_answer_json(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("canonical_answer_json must not be blank")
        try:
            json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise ValueError("canonical_answer_json must be a valid JSON string") from exc
        return normalized

    @field_validator("question", "label_summary")
    @classmethod
    def _validate_non_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text fields must not be blank")
        return normalized

    @field_validator("anchor_entity")
    @classmethod
    def _validate_anchor_entity(cls, value: dict[str, object]) -> dict[str, object]:
        if not value:
            raise ValueError("anchor_entity must contain at least one primary-key value")
        normalized: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ValueError("anchor_entity keys must be non-empty strings")
            if raw_value is None or isinstance(raw_value, (dict, list)):
                raise ValueError("anchor_entity values must be scalar JSON values")
            normalized[raw_key.strip()] = raw_value
        return normalized


class SubmitConstraintSummaryItem(StrictModel):
    key: str = Field(min_length=1, description="Stable identifier for this constraint.")
    kind: str = Field(
        min_length=1,
        description="Constraint kind such as ordering, uniqueness, membership, or other.",
    )
    summary: str = Field(min_length=1, description="Natural-language summary of the constraint.")
    hard: bool = Field(default=True, description="Whether this constraint is mandatory.")

    @field_validator("key", "kind", "summary")
    @classmethod
    def _validate_non_blank_constraint_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("constraint fields must not be blank")
        return normalized


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
    repeat_count = 1
    for axis in reversed(history[:-1]):
        if axis != last_axis:
            break
        repeat_count += 1
    if repeat_count < 2:
        return last_axis
    last_index = DIFFICULTY_CRANK_ORDER.index(last_axis)
    return DIFFICULTY_CRANK_ORDER[(last_index + 1) % len(DIFFICULTY_CRANK_ORDER)]


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


def _constraint_summary_payload(
    items: list[object],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            payload.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            payload.append({str(key): value for key, value in item.items()})
    return payload


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)


def _is_single_tool_derivable(answer: object, result: object) -> bool:
    if _stable_json(answer) == _stable_json(result):
        return True

    if isinstance(answer, dict):
        if isinstance(result, dict):
            return all(key in result and result[key] == value for key, value in answer.items())
        if isinstance(result, list):
            return any(_is_single_tool_derivable(answer, item) for item in result)
        return False

    if isinstance(answer, list):
        if not isinstance(result, list) or len(result) < len(answer):
            return False
        prefix = result[: len(answer)]
        if _stable_json(answer) == _stable_json(prefix):
            return True
        if answer and all(isinstance(item, dict) for item in answer) and all(
            isinstance(item, dict) for item in prefix
        ):
            if all(
                all(key in result_item and result_item[key] == value for key, value in answer_item.items())
                for answer_item, result_item in zip(answer, prefix)
            ):
                return True
        if answer and all(not isinstance(item, (dict, list)) for item in answer) and all(
            isinstance(item, dict) for item in prefix
        ):
            shared_keys = set(prefix[0].keys())
            for item in prefix[1:]:
                shared_keys &= set(item.keys())
            for key in shared_keys:
                if [item.get(key) for item in prefix] == answer:
                    return True
        return False

    if isinstance(result, dict):
        return answer in result.values()
    if isinstance(result, list):
        if answer in result:
            return True
        return any(_is_single_tool_derivable(answer, item) for item in result)
    return False


def _single_tool_derivation_source(
    answer: object,
    tool_calls: list[dict[str, object]],
) -> str | None:
    for record in tool_calls:
        if _is_single_tool_derivable(answer, record.get("result")):
            return str(record.get("tool_name", "unknown_tool"))
    return None


def _answer_uses_only_identifier_fields(answer: object) -> bool:
    def _keys_are_identifier_only(mapping: dict[str, object]) -> bool:
        if not mapping:
            return False
        return all(isinstance(key, str) and _is_identifier_field_name(key) for key in mapping)

    if isinstance(answer, dict):
        return _keys_are_identifier_only(answer)
    if isinstance(answer, list) and answer and all(isinstance(item, dict) for item in answer):
        return all(_keys_are_identifier_only(item) for item in answer)
    return False


def _collect_observed_string_tokens(value: object, *, strings: set[str], tokens: set[str]) -> None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            strings.add(normalized)
            tokens.update(re.findall(r"[a-z0-9가-힣]+", normalized))
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_observed_string_tokens(item, strings=strings, tokens=tokens)
        return
    if isinstance(value, list):
        for item in value:
            _collect_observed_string_tokens(item, strings=strings, tokens=tokens)


def _collect_answer_strings(value: object, *, sink: list[str]) -> None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            sink.append(normalized)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_answer_strings(item, sink=sink)
        return
    if isinstance(value, list):
        for item in value:
            _collect_answer_strings(item, sink=sink)


def _ungrounded_answer_strings(
    answer: object,
    tool_calls: list[dict[str, object]],
) -> list[str]:
    observed_strings: set[str] = set()
    observed_tokens: set[str] = set()
    for record in tool_calls:
        _collect_observed_string_tokens(record.get("result"), strings=observed_strings, tokens=observed_tokens)

    answer_strings: list[str] = []
    _collect_answer_strings(answer, sink=answer_strings)
    ungrounded: list[str] = []
    for value in answer_strings:
        if value in observed_strings:
            continue
        tokens = re.findall(r"[a-z0-9가-힣]+", value)
        if tokens and all(token in observed_tokens for token in tokens):
            continue
        ungrounded.append(value)
    return sorted(dict.fromkeys(ungrounded))


_ASSIGNMENT_TOPIC_MARKERS = (
    "assign",
    "assigned",
    "assignment",
    "responsible",
    "responsibility",
    "owner",
    "owned",
    "managed",
    "manager",
    "allocated",
    "allocation",
    "배정",
    "담당",
    "소속",
    "맡",
    "할당",
    "책임",
)

_IDENTIFIER_FIELD_TOKEN_RE = re.compile(
    r"(?<![a-z0-9_])[a-z][a-z0-9_]*_ids?(?![a-z0-9_])",
    re.IGNORECASE,
)


def _is_identifier_field_name(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized.endswith("_id") or normalized.endswith("_ids")


def _contains_raw_identifier_token(text: str) -> bool:
    return _IDENTIFIER_FIELD_TOKEN_RE.search(text.lower()) is not None


def _question_matches_requested_topic(*, requested_topic: str, question: str) -> bool:
    normalized_topic = requested_topic.strip().lower()
    if normalized_topic != "assignment":
        return True
    lowered = question.lower()
    return any(marker in lowered for marker in _ASSIGNMENT_TOPIC_MARKERS)


def _question_repeats_anchor_entity(question: str) -> bool:
    lowered = question.lower()
    patterns = (
        r"\b(?:staff|customer|store|address|payment|rental|inventory|film|city|country)\s*\d+(?:\b|[가-힣])",
        r"\b\d+\s*(?:번)?\s*(?:staff|customer|store|address|payment|rental|inventory|film|city|country)(?:\b|[가-힣])",
        r"(?:매장|고객|직원|영화|주소|도시|국가)\s*\d+",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


@dataclass(slots=True)
class SubmitDraftController:
    config: AppConfig
    requested_topic: str
    environment_orchestrator: EnvironmentOrchestrator
    build_draft: Any
    phase_monitor: PipelinePhaseMonitorLogger | None = None
    max_submissions: int = 5
    forbidden_question_tokens: frozenset[str] = field(default_factory=frozenset)
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
    _raw_atomic_tool_calls: list[dict[str, object]] = field(default_factory=list, init=False)
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
        self._raw_atomic_tool_calls.append(
            {
                "tool_name": tool_name,
                "params": params,
                "result": result,
            }
        )
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
            error_type = str(error_item.get("type", ""))
            if error_type == "missing" and len(location) == 1:
                error_codes.append(f"{location[0]}_required")
            elif location == ("canonical_answer_json",):
                error_codes.append("canonical_answer_json_invalid")
            elif location == ("constraint_summary",):
                error_codes.append("constraint_summary_required")
            elif location == ("anchor_entity",):
                error_codes.append("anchor_entity_required")
            else:
                error_codes.append("submit_payload_invalid")
        raw_question = parsed.get("question")
        if isinstance(raw_question, str) and _contains_raw_identifier_token(raw_question):
            error_codes.append("question_raw_identifier_leak")
        raw_canonical = parsed.get("canonical_answer_json")
        if isinstance(raw_canonical, str):
            try:
                raw_answer = json.loads(raw_canonical)
            except json.JSONDecodeError:
                raw_answer = None
            if raw_answer is not None and _answer_uses_only_identifier_fields(raw_answer):
                error_codes.append("label_identifier_chain_forbidden")
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
        invalid_diagnostics: dict[str, object] = {}

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
                "constraint_summary": _constraint_summary_payload(payload.constraint_summary),
                "instance_space": payload.instance_space.model_dump(mode="json"),
            }
        )
        if placeholder_tokens:
            error_codes.append("placeholder_tokens_not_allowed")
        try:
            canonical_answer = json.loads(payload.canonical_answer_json)
        except json.JSONDecodeError:
            canonical_answer = None
        if canonical_answer is not None:
            derivation_tool = _single_tool_derivation_source(
                canonical_answer,
                self._raw_atomic_tool_calls,
            )
            if derivation_tool is not None:
                error_codes.append("label_single_tool_derivable")
                invalid_diagnostics["single_tool_name"] = derivation_tool
            ungrounded_strings = _ungrounded_answer_strings(
                canonical_answer,
                self._raw_atomic_tool_calls,
            )
            if ungrounded_strings:
                error_codes.append("label_values_not_grounded")
                invalid_diagnostics["ungrounded_strings"] = ungrounded_strings[:5]
        question_lower = payload.question.lower()
        if _contains_raw_identifier_token(question_lower):
            error_codes.append("question_raw_identifier_leak")
        if _question_repeats_anchor_entity(payload.question):
            error_codes.append("question_anchor_entity_leak")
        if any(token in question_lower for token in self.forbidden_question_tokens):
            error_codes.append("question_internal_schema_leak")
        if canonical_answer is not None and _answer_uses_only_identifier_fields(canonical_answer):
            error_codes.append("label_identifier_chain_forbidden")
        if not _question_matches_requested_topic(
            requested_topic=self.requested_topic,
            question=payload.question,
        ):
            error_codes.append("requested_topic_misaligned")

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
                diagnostics=invalid_diagnostics or None,
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
        self.strongest_difficulty_vector = _merge_strongest_difficulty_vector(
            self.strongest_difficulty_vector,
            payload.difficulty_vector,
        )

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
            self.required_axis = requested_axis
            self.difficulty_crank_history.append(requested_axis)
            strongest_axis_value = getattr(self.strongest_difficulty_vector, requested_axis.value)
            return self._record_rejection(
                submission_index=submission_index,
                message=(
                    "Rejected. solver pass rate "
                    f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}. "
                    f"Crank {requested_axis.value}. {_difficulty_axis_hint(requested_axis)} "
                    f"Make at least one new atomic tool call, gather new grounded evidence, and strengthen only that axis above {strongest_axis_value:.1f} with the smallest grounded step you can justify before resubmitting. "
                    f"{max(0, attempts_left_after)} attempts left."
                ),
                error_codes=["reject_too_easy"],
                pass_rate=quality_gate_summary.pass_rate,
                matched_solver_runs=quality_gate_summary.matched_solver_runs,
                total_solver_runs=quality_gate_summary.total_solver_runs,
                diagnostics={"requested_axis": requested_axis.value},
                payload=payload,
            )

        self.max_submissions = len(self.attempts) + 1
        return self._record_rejection(
            submission_index=submission_index,
            message=(
                "Rejected. solver pass rate "
                f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}. "
                "This draft is too hard for the configured band."
            ),
            error_codes=["reject_too_hard"],
            pass_rate=quality_gate_summary.pass_rate,
            matched_solver_runs=quality_gate_summary.matched_solver_runs,
            total_solver_runs=quality_gate_summary.total_solver_runs,
            diagnostics={"terminal_rejection": True},
            payload=payload,
        )

    def _invalid_submission_message(self, error_codes: list[str]) -> str:
        message_map = {
            "no_new_grounded_observation": (
                "Rejected. Observe more real database facts with atomic tools before resubmitting."
            ),
            "anchor_entity_required": "Rejected. anchor_entity must contain at least one primary-key value.",
            "canonical_answer_json_required": "Rejected. canonical_answer_json is required.",
            "difficulty_vector_required": "Rejected. difficulty_vector is required.",
            "question_required": "Rejected. question is required.",
            "constraint_summary_required": (
                "Rejected. constraint_summary must include at least one grounded constraint or tie-break rule."
            ),
            "instance_space_required": "Rejected. instance_space is required.",
            "label_summary_required": "Rejected. label_summary is required.",
            "canonical_answer_json_invalid": (
                "Rejected. canonical_answer_json must be a valid JSON string."
            ),
            "placeholder_tokens_not_allowed": (
                "Rejected. Replace every placeholder token with grounded names and values from the current database."
            ),
            "question_internal_schema_leak": (
                "Rejected. Rewrite the user-facing question without raw table names, join-table names, or SQL keywords."
            ),
            "question_raw_identifier_leak": (
                "Rejected. Rewrite the user-facing question without raw identifier field names such as customer_id or store_id. Keep identifiers only inside anchor_entity."
            ),
            "question_anchor_entity_leak": (
                "Rejected. Rewrite the user-facing question without repeating the raw anchor entity id. Keep the raw anchor only inside the entity block and refer to it naturally in the question."
            ),
            "label_single_tool_derivable": (
                "Rejected. The canonical answer can be recovered from a single atomic tool call. Redesign the task so the label requires combining multiple observations."
            ),
            "label_identifier_chain_forbidden": (
                "Rejected. The canonical answer is only a chain of internal identifier fields. Return user-relevant business values such as names, titles, dates, amounts, counts, or statuses instead."
            ),
            "label_values_not_grounded": (
                "Rejected. Some label values were not directly grounded in the observed tool results. Only use business strings you actually observed."
            ),
            "requested_topic_misaligned": (
                "Rejected. The user-facing question drifted away from the requested topic. For assignment, ask about who or what is assigned, handled, owned, responsible, attached, or linked to the anchor."
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
        additional_messages: list[str] = []
        for error_code in error_codes[1:3]:
            extra = message_map.get(error_code)
            if extra is None:
                continue
            additional_messages.append(extra.removeprefix("Rejected. ").strip())
        attempts_left_after = self.submissions_left() - 1
        if additional_messages:
            return (
                f"{primary} Also fix: {' '.join(additional_messages)} "
                f"{max(0, attempts_left_after)} attempts left."
            )
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
            return f"{message} Budget exhausted. No more attempts."
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
                "submission_index": len(self.attempts),
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
            "Do not call submit_draft until anchor_entity is present and final for that draft. "
            "Do not submit labels that can be read from a single atomic tool call or a direct projection of a single tool result. "
            "Do not submit questions or labels that are only chains of internal *_id fields. Prefer business-facing values. "
            "For assignment tasks, make the assignee relation explicit and use human-readable assignee attributes when possible. "
            "If the candidate assignee surface only exposes opaque ids in tool results, choose a different assignment relation instead of centering the task on that entity. "
            "After any rejection, make at least one new atomic tool call before calling submit_draft again. "
            "A rejection means keep going in the same conversation, not stop. "
            "All submit_draft arguments are schema-validated; missing required fields and invalid canonical_answer_json values are rejected automatically."
        ),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=False,
    )
