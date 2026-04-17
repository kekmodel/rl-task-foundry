"""submit_draft tool for the single-agent synthesis loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import Field, ValidationError, field_validator

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.infra.sdk_helpers import preview_payload
from rl_task_foundry.synthesis.canonicalize import canonical_json
from rl_task_foundry.synthesis.contracts import StrictModel
from rl_task_foundry.synthesis.phase_monitor import PipelinePhaseMonitorLogger
from rl_task_foundry.synthesis.submit_draft_messages import (
    _format_ungrounded_value_guidance,
    _render_structured_message,
    _too_easy_retry_guidance,
)
from rl_task_foundry.synthesis.submit_draft_validators import (
    _blank_string_paths,
    _collect_observed_strings,
    _contains_entity_placeholder_token,
    _contains_raw_identifier_token,
    _disconnected_answer_strings,
    _observed_anchor_readable_string_surface,
    _rebuild_anchor_connected_strings,
    _ungrounded_answer_strings,
)

if TYPE_CHECKING:
    from rl_task_foundry.pipeline.solver_orchestrator import SolverOrchestrator
    from rl_task_foundry.synthesis.runtime import SynthesisTaskDraft


class SubmitDraftErrorCode(StrEnum):
    NO_NEW_GROUNDED_OBSERVATION = "no_new_grounded_observation"
    TOPIC_REQUIRED = "topic_required"
    ANCHOR_ENTITY_REQUIRED = "anchor_entity_required"
    ANCHOR_ENTITY_SCALAR_MAP_REQUIRED = "anchor_entity_scalar_map_required"
    CANONICAL_ANSWER_JSON_INVALID = "canonical_answer_json_invalid"
    QUESTION_REQUIRED = "question_required"
    QUESTION_ENTITY_BLOCK_REQUIRED = "question_entity_block_required"
    QUESTION_ENTITY_BLOCK_INVALID_JSON = "question_entity_block_invalid_json"
    QUESTION_ENTITY_BLOCK_MISMATCH = "question_entity_block_mismatch"
    QUESTION_BODY_REQUIRED = "question_body_required"
    QUESTION_RAW_IDENTIFIER_LEAK = "question_raw_identifier_leak"
    QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN = "question_entity_placeholder_forbidden"
    LABEL_BLANK_STRING_FORBIDDEN = "label_blank_string_forbidden"
    LABEL_VALUES_NOT_GROUNDED = "label_values_not_grounded"
    LABEL_NOT_STRENGTHENED = "label_not_strengthened"
    SUBMIT_PAYLOAD_INVALID = "submit_payload_invalid"
    DRAFT_VALIDATION_FAILED = "draft_validation_failed"
    REJECT_TOO_EASY = "reject_too_easy"
    REJECT_TOO_HARD = "reject_too_hard"


_FEEDBACK_ONLY_ERROR_CODES = frozenset(
    {
        SubmitDraftErrorCode.NO_NEW_GROUNDED_OBSERVATION,
        SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED,
        SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED,
        SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_REQUIRED,
        SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_INVALID_JSON,
        SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_MISMATCH,
        SubmitDraftErrorCode.QUESTION_BODY_REQUIRED,
        SubmitDraftErrorCode.QUESTION_RAW_IDENTIFIER_LEAK,
        SubmitDraftErrorCode.QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN,
        SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN,
        SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED,
        SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED,
    }
)


def _error_code_values(
    codes: list[SubmitDraftErrorCode] | tuple[SubmitDraftErrorCode, ...],
) -> list[str]:
    return [code.value for code in codes]


class SubmitDraftPayload(StrictModel):
    topic: str = Field(
        min_length=1,
        description="Selected topic string derived from the grounded label and evidence.",
    )
    label: dict[str, object] | list[object] = Field(
        description="Ground-truth answer. A flat object or a list of uniform objects.",
    )
    entity: str = Field(
        min_length=1,
        description='Anchor entity as a JSON string, e.g. \'{"<pk_name>": 123}\'.',
    )
    question: str = Field(
        min_length=1,
        description=(
            "Full user-facing prompt in the configured task language, starting with "
            "<entity> ... </entity> on its own lines, followed by a blank line and the natural user request body. "  # noqa: E501
            "The entity block must exactly match entity."
        ),
    )

    @field_validator("label", mode="before")
    @classmethod
    def _validate_label(cls, value: object) -> dict[str, object] | list[object]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("label must be valid JSON") from exc
        if not isinstance(value, (dict, list)):
            raise ValueError("label must be an object or array")
        if isinstance(value, dict) and not value:
            raise ValueError("label must not be empty")
        if isinstance(value, list) and not value:
            raise ValueError("label must not be empty")
        return value

    @field_validator("topic", "question")
    @classmethod
    def _validate_non_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text fields must not be blank")
        return normalized

    @field_validator("entity", mode="before")
    @classmethod
    def _validate_entity(cls, value: object) -> str:
        if isinstance(value, dict):
            _normalize_anchor_entity_map(value)
            return json.dumps(value, ensure_ascii=False)
        raw = str(value).strip() if not isinstance(value, str) else value.strip()
        if not raw:
            raise ValueError("entity must not be blank")
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raise ValueError("entity must be a valid JSON string")
        if not isinstance(parsed, dict):
            raise ValueError("entity must be a JSON object")
        _normalize_anchor_entity_map(parsed)
        return raw

    @property
    def parsed_entity(self) -> dict[str, object]:
        return _normalize_anchor_entity_map(json.loads(self.entity))

    @property
    def canonical_answer(self) -> object:
        return self.label


SubmitDraftPayload.model_rebuild()


def _normalize_anchor_entity_map(value: dict[str, object]) -> dict[str, object]:
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


@dataclass(frozen=True, slots=True)
class SubmitDraftAttemptRecord:
    index: int
    outcome: str
    message: str
    error_codes: tuple[str, ...] = ()
    pass_rate: float | None = None
    matched_solver_runs: int | None = None
    total_solver_runs: int | None = None


def _preview_runtime_payload(value: object, *, config: AppConfig) -> object:
    return preview_payload(
        value,
        max_string_length=config.synthesis.runtime.payload_preview_max_string_length,
        max_list_items=config.synthesis.runtime.payload_preview_max_list_items,
        max_dict_items=config.synthesis.runtime.payload_preview_max_dict_items,
    )


def _monitor_answer_snapshot(
    value: object,
    *,
    config: AppConfig,
) -> dict[str, object]:
    if isinstance(value, dict):
        return {
            "root_type": "object",
            "slot_count": len(value),
            "field_names": list(value.keys())[: config.synthesis.runtime.label_preview_field_limit],
            "preview": _preview_runtime_payload(value, config=config),
        }
    if isinstance(value, list):
        field_names: list[str] = []
        all_keys: set[str] = set()
        for item in value:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        field_names = sorted(all_keys)[: config.synthesis.runtime.label_preview_field_limit]
        return {
            "root_type": "array",
            "slot_count": len(all_keys),
            "item_count": len(value),
            "field_names": field_names,
            "preview": _preview_runtime_payload(value, config=config),
        }
    return {
        "root_type": type(value).__name__,
        "slot_count": 1,
        "field_names": [],
        "preview": _preview_runtime_payload(value, config=config),
    }


def _answer_slot_count(value: object) -> int:
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, list):
        all_keys: set[str] = set()
        for item in value:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        return len(all_keys) if all_keys else len(value)
    return 1


def _monitor_label_data(
    payload: SubmitDraftPayload | dict[str, object] | None,
    *,
    config: AppConfig,
) -> dict[str, object]:
    if payload is None:
        return {
            "canonical_answer_preview": None,
            "canonical_answer_root_type": None,
            "canonical_answer_slot_count": None,
            "canonical_answer_field_names": [],
        }

    if isinstance(payload, SubmitDraftPayload):
        canonical_answer = payload.canonical_answer
    else:
        canonical_answer = None
        raw_canonical = payload.get("label")
        if isinstance(raw_canonical, (dict, list)):
            canonical_answer = raw_canonical
        elif isinstance(raw_canonical, str):
            try:
                canonical_answer = json.loads(raw_canonical)
            except json.JSONDecodeError:
                canonical_answer = raw_canonical

    snapshot = (
        _monitor_answer_snapshot(canonical_answer, config=config)
        if canonical_answer is not None
        else {
            "root_type": None,
            "slot_count": None,
            "field_names": [],
            "preview": None,
        }
    )
    return {
        "canonical_answer_preview": snapshot["preview"],
        "canonical_answer_root_type": snapshot["root_type"],
        "canonical_answer_slot_count": snapshot["slot_count"],
        "canonical_answer_field_names": snapshot["field_names"],
        "canonical_answer_signature": (
            canonical_json(canonical_answer, default=str) if canonical_answer is not None else None
        ),
    }


def _label_change_summary(
    *,
    previous: dict[str, object] | None,
    current: dict[str, object],
) -> dict[str, object]:
    previous_field_names: list[object] = (
        list(previous.get("canonical_answer_field_names", []))  # type: ignore[call-overload]
        if previous
        else []
    )
    current_field_names: list[object] = list(
        current.get("canonical_answer_field_names", [])  # type: ignore[call-overload]
    )
    previous_slot_count = previous.get("canonical_answer_slot_count") if previous else None
    current_slot_count = current.get("canonical_answer_slot_count")
    return {
        "label_changed": (
            previous.get("canonical_answer_signature") != current.get("canonical_answer_signature")
            if previous is not None
            else None
        ),
        "previous_canonical_answer_preview": (
            previous.get("canonical_answer_preview") if previous is not None else None
        ),
        "previous_canonical_answer_slot_count": previous_slot_count,
        "added_field_names": [
            field_name
            for field_name in current_field_names
            if field_name not in previous_field_names
        ],
        "removed_field_names": [
            field_name
            for field_name in previous_field_names
            if field_name not in current_field_names
        ],
        "slot_count_delta": (
            current_slot_count - previous_slot_count
            if isinstance(current_slot_count, int) and isinstance(previous_slot_count, int)
            else None
        ),
    }




_ENTITY_PROMPT_RE = re.compile(
    r"\A<entity>\n(?P<entity_json>.+?)\n</entity>\n\n(?P<body>.+)\Z",
    re.DOTALL,
)


def _split_entity_wrapped_prompt(
    value: str,
) -> tuple[dict[str, object] | None, str | None, SubmitDraftErrorCode | None]:
    normalized = value.strip()
    match = _ENTITY_PROMPT_RE.match(normalized)
    if match is None:
        return None, None, SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_REQUIRED
    entity_json = match.group("entity_json").strip()
    try:
        parsed_entity = json.loads(entity_json)
    except json.JSONDecodeError:
        return None, None, SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_INVALID_JSON
    if not isinstance(parsed_entity, dict):
        return None, None, SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_INVALID_JSON
    try:
        anchor_entity = _normalize_anchor_entity_map(parsed_entity)
    except ValueError:
        return None, None, SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_INVALID_JSON
    body = match.group("body").strip()
    if not body:
        return anchor_entity, None, SubmitDraftErrorCode.QUESTION_BODY_REQUIRED
    return anchor_entity, body, None


@dataclass(slots=True)
class SubmitDraftController:
    config: AppConfig
    requested_topic: str | None
    solver_orchestrator: SolverOrchestrator
    build_draft: Any
    max_submissions: int
    phase_monitor: PipelinePhaseMonitorLogger | None = None
    accepted_draft: SynthesisTaskDraft | None = None
    attempts: list[SubmitDraftAttemptRecord] = field(default_factory=list)
    _needs_label_change: bool = field(default=False, init=False)
    last_quality_gate_status: str | None = None
    last_quality_gate_pass_rate: float | None = None
    _atomic_tool_calls: list[dict[str, object]] = field(default_factory=list, init=False)
    _raw_atomic_tool_calls: list[dict[str, object]] = field(default_factory=list, init=False)
    _tool_call_count_at_last_submission: int = field(default=0, init=False)
    _last_label_signature: str | None = field(default=None, init=False)
    _last_label_slot_count: int | None = field(default=None, init=False)
    _last_monitored_label_data: dict[str, object] | None = field(default=None, init=False)
    _feedback_events: int = field(default=0, init=False)
    _locked_anchor_entity: dict[str, object] | None = field(default=None, init=False)
    _observed_response_strings: set[str] = field(default_factory=set, init=False)
    _terminated_too_hard: bool = field(default=False, init=False)

    def submissions_left(self) -> int:
        if self._terminated_too_hard:
            return 0
        return max(0, self.max_submissions - (len(self.attempts) + self._feedback_events))

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
                "params": _preview_runtime_payload(params, config=self.config),
                "result": _preview_runtime_payload(result, config=self.config),
            }
        )
        _collect_observed_strings(result, strings=self._observed_response_strings)

    def reject_invalid_payload(self, *, parsed: dict[str, object], error: ValidationError) -> str:
        if self.accepted_draft is not None:
            return _render_structured_message(
                kind="Accepted",
                result="Draft already stored.",
            )
        if self.submissions_left() <= 0:
            return "BudgetExhaustedError: No more attempts."

        error_codes: list[SubmitDraftErrorCode] = []
        for error_item in error.errors():
            location = tuple(str(part) for part in error_item.get("loc", ()))
            error_type = str(error_item.get("type", ""))
            if error_type == "missing" and len(location) == 1:
                required_code = getattr(
                    SubmitDraftErrorCode,
                    f"{location[0].upper()}_REQUIRED",
                    SubmitDraftErrorCode.SUBMIT_PAYLOAD_INVALID,
                )
                error_codes.append(required_code)
            elif location == ("label",):
                error_codes.append(SubmitDraftErrorCode.CANONICAL_ANSWER_JSON_INVALID)
            elif location == ("entity",):
                if error_type in ("value_error", "dict_type"):
                    error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED)
                else:
                    error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED)
            else:
                error_codes.append(SubmitDraftErrorCode.SUBMIT_PAYLOAD_INVALID)
        raw_question = parsed.get("question")
        if isinstance(raw_question, str):
            _, question_body, prompt_error = _split_entity_wrapped_prompt(raw_question)
            if prompt_error is not None:
                error_codes.append(prompt_error)
            elif question_body is not None and _contains_raw_identifier_token(question_body):
                error_codes.append(SubmitDraftErrorCode.QUESTION_RAW_IDENTIFIER_LEAK)
        deduped_error_codes = list(dict.fromkeys(error_codes)) or [
            SubmitDraftErrorCode.SUBMIT_PAYLOAD_INVALID
        ]
        search_cost_observations = (
            len(self._raw_atomic_tool_calls) - self._tool_call_count_at_last_submission
        )
        if all(error_code in _FEEDBACK_ONLY_ERROR_CODES for error_code in deduped_error_codes):
            return self._record_feedback(
                message=self._invalid_submission_message(
                    deduped_error_codes,
                    feedback_only=True,
                ),
                error_codes=deduped_error_codes,
                payload=parsed,
                search_cost_observations=search_cost_observations,
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
        return self._record_rejection(
            submission_index=len(self.attempts) + 1,
            message=self._invalid_submission_message(deduped_error_codes),
            error_codes=deduped_error_codes,
            payload=parsed,
            search_cost_observations=search_cost_observations,
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
            return _render_structured_message(
                kind="Accepted",
                result="Draft already stored.",
            )
        if self._terminated_too_hard:
            return (
                "TerminatedError: This conversation was discarded after a "
                "too-hard rejection. Stop calling submit_draft."
            )
        if self.submissions_left() <= 0:
            return "BudgetExhaustedError: No more attempts."

        submission_index = len(self.attempts) + 1
        error_codes: list[SubmitDraftErrorCode] = []
        invalid_diagnostics: dict[str, object] = {}
        search_cost_observations = (
            len(self._raw_atomic_tool_calls) - self._tool_call_count_at_last_submission
        )
        prompt_anchor_entity, question_body, prompt_error = _split_entity_wrapped_prompt(
            payload.question
        )

        if len(self._raw_atomic_tool_calls) <= self._tool_call_count_at_last_submission:
            error_codes.append(SubmitDraftErrorCode.NO_NEW_GROUNDED_OBSERVATION)
        parsed_anchor = payload.parsed_entity
        if not parsed_anchor:
            error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED)
        else:
            self._locked_anchor_entity = dict(parsed_anchor)
        canonical_answer = payload.canonical_answer
        if prompt_error is not None:
            error_codes.append(prompt_error)
        elif prompt_anchor_entity != parsed_anchor:
            error_codes.append(SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_MISMATCH)
        label_signature = canonical_json(canonical_answer, default=str)
        label_slot_count = _answer_slot_count(canonical_answer)
        blank_paths = _blank_string_paths(canonical_answer)
        if blank_paths:
            error_codes.append(SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN)
            invalid_diagnostics["blank_string_paths"] = blank_paths[
                : self.config.synthesis.runtime.diagnostic_item_limit
            ]
        ungrounded_strings = _ungrounded_answer_strings(
            canonical_answer,
            observed_strings=self._observed_response_strings,
        )
        if ungrounded_strings:
            error_codes.append(SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED)
            invalid_diagnostics["ungrounded_strings"] = ungrounded_strings[
                : self.config.synthesis.runtime.diagnostic_item_limit
            ]
            invalid_diagnostics["anchor_path_has_readable_strings"] = (
                _observed_anchor_readable_string_surface(
                    self._raw_atomic_tool_calls,
                    anchor_entity=parsed_anchor,
                )
            )
        if not ungrounded_strings and self._locked_anchor_entity:
            anchor_strings = _rebuild_anchor_connected_strings(
                self._raw_atomic_tool_calls,
                anchor_entity=self._locked_anchor_entity,
            )
            disconnected = _disconnected_answer_strings(
                canonical_answer,
                observed_strings=self._observed_response_strings,
                anchor_connected_strings=anchor_strings,
            )
            if disconnected:
                # Diagnostic only — integer ID collision causes
                # false positives until param-name-aware tracking
                # is implemented.
                invalid_diagnostics["disconnected_strings"] = disconnected[
                    : self.config.synthesis.runtime.diagnostic_item_limit
                ]
        if question_body is not None:
            question_lower = question_body.lower()
            if _contains_entity_placeholder_token(question_lower):
                error_codes.append(SubmitDraftErrorCode.QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN)

        # After a too-easy rejection, check that the label
        # actually changed — no difficulty vector math.
        if (
            self._needs_label_change
            and self._last_label_signature is not None
            and label_signature == self._last_label_signature
        ):
            error_codes.append(
                SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED
            )

        if error_codes:
            deduped_error_codes = list(dict.fromkeys(error_codes))
            if all(error_code in _FEEDBACK_ONLY_ERROR_CODES for error_code in deduped_error_codes):
                return self._record_feedback(
                    message=self._invalid_submission_message(
                        deduped_error_codes,
                        feedback_only=True,
                        diagnostics=invalid_diagnostics or None,
                    ),
                    error_codes=deduped_error_codes,
                    payload=payload,
                    search_cost_observations=search_cost_observations,
                    diagnostics=invalid_diagnostics or None,
                )
            return self._record_rejection(
                submission_index=submission_index,
                message=self._invalid_submission_message(
                    deduped_error_codes,
                    diagnostics=invalid_diagnostics or None,
                ),
                error_codes=deduped_error_codes,
                payload=payload,
                search_cost_observations=search_cost_observations,
                diagnostics=invalid_diagnostics or None,
            )

        try:
            draft = self.build_draft(payload)
        except Exception as exc:
            return self._record_rejection(
                submission_index=submission_index,
                message=f"RejectedError: Invalid draft: {type(exc).__name__}: {exc}",
                error_codes=[SubmitDraftErrorCode.DRAFT_VALIDATION_FAILED],
                payload=payload,
                search_cost_observations=search_cost_observations,
            )

        rollout_summary = await self.solver_orchestrator.run_draft(draft)
        from rl_task_foundry.pipeline.solver_orchestrator import (
            TaskQualityGateStatus,
            evaluate_rollout_summary,
        )
        from rl_task_foundry.synthesis.quality_gate import accepted_draft_with_quality_metrics

        quality_gate_summary = evaluate_rollout_summary(self.config, rollout_summary)
        self.last_quality_gate_status = quality_gate_summary.status.value
        self.last_quality_gate_pass_rate = quality_gate_summary.pass_rate
        self._tool_call_count_at_last_submission = len(self._raw_atomic_tool_calls)
        self._last_label_signature = label_signature
        self._last_label_slot_count = label_slot_count
        if quality_gate_summary.status is TaskQualityGateStatus.ACCEPT:
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
                search_cost_observations=search_cost_observations,
                diagnostics={},
            )
            return _render_structured_message(
                kind="Accepted",
                result=(
                    "solver pass rate "
                    f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}."
                ),
            )

        attempts_left_after = self.submissions_left() - 1
        if quality_gate_summary.status is TaskQualityGateStatus.REJECT_TOO_EASY:
            self._needs_label_change = True
            strengthening_guidance = _too_easy_retry_guidance()
            return self._record_rejection(
                submission_index=submission_index,
                message=_render_structured_message(
                    kind="RejectedError",
                    result=(
                        "solver pass rate "
                        f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}."
                    ),
                    primary=(
                        "Too easy — all solvers passed. "
                        "Make the task harder by changing "
                        "the label, not just the wording."
                    ),
                    important=strengthening_guidance.strip(),
                    next_step=(
                        "Make at least one new tool call "
                        "for the new evidence, then "
                        "resubmit."
                    ),
                    attempts_left=max(0, attempts_left_after),
                ),
                error_codes=[SubmitDraftErrorCode.REJECT_TOO_EASY],
                pass_rate=quality_gate_summary.pass_rate,
                matched_solver_runs=quality_gate_summary.matched_solver_runs,
                total_solver_runs=quality_gate_summary.total_solver_runs,
                diagnostics={},
                payload=payload,
                search_cost_observations=search_cost_observations,
            )

        # Too hard is terminal — discard and let the outer
        # loop start fresh rather than oscillating.
        self._terminated_too_hard = True
        return self._record_rejection(
            submission_index=submission_index,
            message=_render_structured_message(
                kind="RejectedError",
                result=(
                    "solver pass rate "
                    f"{quality_gate_summary.matched_solver_runs}"
                    f"/{quality_gate_summary.total_solver_runs}."
                ),
                primary=(
                    "Too hard — no solver passed. "
                    "This conversation is terminated. "
                    "Do not call submit_draft again."
                ),
                attempts_left=0,
            ),
            error_codes=[SubmitDraftErrorCode.REJECT_TOO_HARD],
            pass_rate=quality_gate_summary.pass_rate,
            matched_solver_runs=quality_gate_summary.matched_solver_runs,
            total_solver_runs=quality_gate_summary.total_solver_runs,
            diagnostics={
                "terminal_rejection": True,
            },
            payload=payload,
            search_cost_observations=search_cost_observations,
        )

    def _invalid_submission_message(
        self,
        error_codes: list[SubmitDraftErrorCode],
        *,
        feedback_only: bool = False,
        diagnostics: dict[str, object] | None = None,
    ) -> str:
        message_map = {
            SubmitDraftErrorCode.NO_NEW_GROUNDED_OBSERVATION: (
                "Rejected. Observe more real database facts with atomic tools before resubmitting."
            ),
            SubmitDraftErrorCode.TOPIC_REQUIRED: "Rejected. topic is required.",
            SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED: "Rejected. entity must contain at least one primary-key value.",  # noqa: E501
            SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED: (
                'Rejected. entity must be a flat JSON object mapping one or more primary-key field names to scalar values, for example {"customer_id": 123} or {"order_id": 7, "line_no": 2}. Do not nest it under entity_type, primary_key, or primary_keys.'  # noqa: E501
            ),
            SubmitDraftErrorCode.QUESTION_REQUIRED: "Rejected. question is required.",
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_REQUIRED: (
                "Rejected. question must already be the full user prompt in this exact shape: <entity> newline JSON newline </entity> blank line user request."  # noqa: E501
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_INVALID_JSON: (
                "Rejected. The <entity> block must contain a valid flat JSON object with one or more primary-key fields."  # noqa: E501
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_MISMATCH: (
                "Rejected. The JSON inside the <entity> block must exactly match entity."
            ),
            SubmitDraftErrorCode.QUESTION_BODY_REQUIRED: (
                "Rejected. After the <entity> block, include a natural user request body."
            ),
            SubmitDraftErrorCode.CANONICAL_ANSWER_JSON_INVALID: (
                "Rejected. label must be a valid JSON string."
            ),
            SubmitDraftErrorCode.QUESTION_RAW_IDENTIFIER_LEAK: (
                "Rejected. Rewrite the user-facing question for a user who does not know internal field names. Remove raw identifier names such as <entity>_id and keep identifiers only inside entity."  # noqa: E501
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN: (
                "Rejected. Do not repeat the literal <entity> token inside the user-request body. Use it only once as the required XML entity block at the top."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN: (
                "Rejected. The canonical answer contains blank string fields. Every answer field must contain a grounded, non-empty value. Schema orientation alone is not enough; only fields you actually observed in tool results are grounded. If the chosen surface is id-only, keep the same anchored user and switch to grounded counts, dates, amounts, statuses, ordering, or make new anchored tool calls until you observe readable fields."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED: (
                "Rejected. Some label values were not directly grounded in the observed tool results. Schema orientation alone is not enough; only use business strings, dates, and other readable values that you actually observed in real tool outputs, and copy them exactly as they appeared there. Do not shorten names, paraphrase labels, normalize timestamp formatting, or manufacture readable labels by wrapping an id in generic words such as 'staff member 2' or 'order 17'. If the chosen surface is id-only, keep the same anchored user and switch to counts, dates, amounts, statuses, ordering, make new anchored tool calls until you observe readable fields, or choose a better grounded topic for the same anchored user need."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED: (
                "Rejected. After a too-easy result, do not resubmit the same label. Strengthen the canonical answer itself with a new grounded step."  # noqa: E501
            ),
            SubmitDraftErrorCode.SUBMIT_PAYLOAD_INVALID: (
                "Rejected. submit_draft arguments did not match the required schema."
            ),
            SubmitDraftErrorCode.DRAFT_VALIDATION_FAILED: "Rejected. The submitted draft could not be validated.",  # noqa: E501
        }
        primary = message_map.get(error_codes[0], "Rejected. Fix the draft and resubmit.")
        if error_codes and error_codes[0] is SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED:
            if (
                diagnostics is not None
                and diagnostics.get("anchor_path_has_readable_strings") is False
            ):
                primary = (
                    "Rejected. The current anchored evidence path does not expose readable text fields in real tool outputs. "  # noqa: E501
                    "Stop retrying names, titles, or other readable strings on this same path. Keep the same anchored user and either answer with grounded counts, dates, amounts, statuses, or ordering, or make new anchored tool calls until you actually observe readable fields."  # noqa: E501
                )
            else:
                primary += _format_ungrounded_value_guidance(diagnostics)
        if error_codes and error_codes[0] is SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED:
            primary += _too_easy_retry_guidance()
        preserve_guidance = ""
        if self._last_monitored_label_data is not None:
            preserve_guidance = (
                "Keep the same anchored user need and fix only the failing part when possible. "
                "Do not reset to a different topic, a different anchor, or a simpler global count just to satisfy this feedback."  # noqa: E501
            )
        additional_messages: list[str] = []
        for error_code in error_codes[1:3]:
            extra = message_map.get(error_code)
            if extra is None:
                continue
            if error_code is SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED:
                extra += _format_ungrounded_value_guidance(diagnostics)
            additional_messages.append(extra)
        if feedback_only:
            attempts_left_after = self.submissions_left() - 1
            return _render_structured_message(
                kind="FeedbackError",
                primary=primary,
                important=preserve_guidance or None,
                also_fix=additional_messages,
                next_step=(
                    "Make another atomic tool call if needed, then call submit_draft again. "
                    "Do not stop with plain text."
                ),
                attempts_left=max(0, attempts_left_after),
            )
        attempts_left_after = self.submissions_left() - 1
        return _render_structured_message(
            kind="RejectedError",
            primary=primary,
            important=preserve_guidance or None,
            also_fix=additional_messages,
            attempts_left=max(0, attempts_left_after),
        )

    def _record_feedback(
        self,
        *,
        message: str,
        error_codes: list[SubmitDraftErrorCode],
        payload: SubmitDraftPayload | dict[str, object] | None = None,
        search_cost_observations: int | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> str:
        self._feedback_events += 1
        attempts_left_after = self.submissions_left()
        self._emit_monitor(
            status="budget_exhausted" if attempts_left_after <= 0 else "feedback",
            payload=payload,
            pass_rate=None,
            matched_solver_runs=None,
            total_solver_runs=None,
            search_cost_observations=search_cost_observations,
            diagnostics={"error_codes": _error_code_values(error_codes), **(diagnostics or {})},
        )
        if attempts_left_after <= 0 and "BudgetExhaustedError: No more attempts." not in message:
            return f"{message} BudgetExhaustedError: No more attempts."
        return message

    def _record_rejection(
        self,
        *,
        submission_index: int,
        message: str,
        error_codes: list[SubmitDraftErrorCode],
        payload: SubmitDraftPayload | dict[str, object] | None = None,
        pass_rate: float | None = None,
        matched_solver_runs: int | None = None,
        total_solver_runs: int | None = None,
        search_cost_observations: int | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> str:
        attempts_left_after = self.submissions_left() - 1
        self.attempts.append(
            SubmitDraftAttemptRecord(
                index=submission_index,
                outcome=error_codes[0].value if error_codes else "rejected",
                message=message,
                error_codes=tuple(_error_code_values(error_codes)),
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
            search_cost_observations=search_cost_observations,
            diagnostics={"error_codes": _error_code_values(error_codes), **(diagnostics or {})},
        )
        if attempts_left_after <= 0:
            return f"{message} BudgetExhaustedError: No more attempts."
        return message

    def _emit_monitor(
        self,
        *,
        status: str,
        payload: SubmitDraftPayload | dict[str, object] | None,
        pass_rate: float | None,
        matched_solver_runs: int | None,
        total_solver_runs: int | None,
        search_cost_observations: int | None,
        diagnostics: dict[str, object],
    ) -> None:
        if self.phase_monitor is None:
            return
        label_data = _monitor_label_data(payload, config=self.config)
        label_change = _label_change_summary(
            previous=self._last_monitored_label_data,
            current=label_data,
        )
        self.phase_monitor.emit(
            phase="submit_draft",
            status=status,
            expected_contract={
                "requested_topic_hint": self.requested_topic,
                "max_submissions": self.max_submissions,
            },
            actual_data={
                "submission_index": len(self.attempts) + self._feedback_events,
                "selected_topic": (
                    payload.topic
                    if isinstance(payload, SubmitDraftPayload)
                    else payload.get("topic")
                    if isinstance(payload, dict)
                    else None
                ),
                "question": (
                    payload.question
                    if isinstance(payload, SubmitDraftPayload)
                    else payload.get("question")
                    if isinstance(payload, dict)
                    else None
                ),
                "anchor_entity": (
                    payload.parsed_entity
                    if isinstance(payload, SubmitDraftPayload)
                    else payload.get("entity")
                    if isinstance(payload, dict)
                    else None
                ),
                **label_data,
                "label_axis_proxies": {
                    "search_cost_observations": search_cost_observations
                    if search_cost_observations is not None
                    else len(self._raw_atomic_tool_calls)
                    - self._tool_call_count_at_last_submission,
                    "solution_space_slots": label_data["canonical_answer_slot_count"],
                    "constraint_density_constraints": 0,
                },
                "label_change": label_change,
                "pass_rate": pass_rate,
                "matched_solver_runs": matched_solver_runs,
                "total_solver_runs": total_solver_runs,
            },
            checks={
                "accepted": status == "accepted",
                "atomic_tool_calls_seen": len(self._atomic_tool_calls),
            },
            diagnostics={
                "needs_label_change": self._needs_label_change,
                "locked_anchor_entity": self._locked_anchor_entity,
                "recent_tool_calls": self._atomic_tool_calls[
                    -self.config.synthesis.runtime.diagnostic_item_limit :
                ],
                **diagnostics,
            },
        )
        self._last_monitored_label_data = label_data


def build_submit_draft_sdk_tool(controller: SubmitDraftController) -> object:
    from agents import FunctionTool

    params_json_schema = SubmitDraftPayload.model_json_schema()

    async def _invoke_tool(_tool_context: Any, input_json: str) -> str:
        parsed = json.loads(input_json) if input_json else {}
        if "entity" not in parsed:
            raw_question = parsed.get("question")
            if isinstance(raw_question, str):
                parsed_anchor_entity, _, prompt_error = (
                    _split_entity_wrapped_prompt(raw_question)
                )
                if (
                    prompt_error is None
                    and parsed_anchor_entity is not None
                ):
                    parsed["entity"] = json.dumps(parsed_anchor_entity, ensure_ascii=False)
        try:
            payload = SubmitDraftPayload.model_validate(parsed)
        except ValidationError as exc:
            return controller.reject_invalid_payload(
                parsed=parsed, error=exc
            )
        return await controller.submit(payload)

    return FunctionTool(
        name="submit_draft",
        description=(
            "Submit a task draft. Include topic, "
            "label, entity, "
            "and question. "
            "Explore paths with tools first — do not "
            "call this until you understand the anchor, "
            "the evidence path, and every answer slot."
        ),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=False,
    )
