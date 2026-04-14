"""submit_draft tool for the single-agent synthesis loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from pydantic import Field, Json, ValidationError, field_validator

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.infra.sdk_helpers import preview_payload
from rl_task_foundry.synthesis.canonicalize import canonical_json
from rl_task_foundry.synthesis.contracts import (
    DifficultyAxis,
    DifficultyVectorContract,
    StrictModel,
    flatten_difficulty_vector,
    normalize_words,
)
from rl_task_foundry.synthesis.phase_monitor import PipelinePhaseMonitorLogger

if TYPE_CHECKING:
    from rl_task_foundry.pipeline.solver_orchestrator import SolverOrchestrator
    from rl_task_foundry.synthesis.runtime import SynthesisTaskDraft


_FORBIDDEN_PLACEHOLDER_TOKENS = (
    "__REAL_",
    "anchor_id",
    "anchor_table",
    "example_field",
    "replace_with_real_",
)


def _strip_message_status_prefix(text: str) -> str:
    for prefix in (
        "Rejected. ",
        "Feedback. ",
        "Accepted. ",
        "STATUS: REJECTED. PRIMARY ISSUE: ",
        "STATUS: FEEDBACK. PRIMARY ISSUE: ",
        "STATUS: ACCEPTED. RESULT: ",
        "RejectedError: ",
        "FeedbackError: ",
        "Accepted: ",
        "ToolError: ",
        "BudgetExhaustedError: ",
    ):
        if text.startswith(prefix):
            return text.removeprefix(prefix).strip()
    return text.strip()


def _render_structured_message(
    *,
    kind: str,
    primary: str | None = None,
    result: str | None = None,
    important: str | None = None,
    also_fix: list[str] | None = None,
    next_step: str | None = None,
    attempts_left: int | None = None,
) -> str:
    headline = _strip_message_status_prefix(result or primary or "")
    parts = [f"{kind}: {headline}"]
    if result:
        if primary:
            parts.append(f"Primary issue: {_strip_message_status_prefix(primary)}")
    elif primary:
        # headline already used the primary body
        pass
    if important:
        parts.append(f"Important: {important.strip()}")
    if also_fix:
        cleaned = [_strip_message_status_prefix(message) for message in also_fix if message.strip()]
        if cleaned:
            parts.append(f"Also fix: {' '.join(cleaned)}")
    if next_step:
        parts.append(f"Next step: {next_step.strip()}")
    if attempts_left is not None:
        parts.append(f"Attempts left: {attempts_left}.")
    return " ".join(parts)

class SubmitDraftErrorCode(StrEnum):
    NO_NEW_GROUNDED_OBSERVATION = "no_new_grounded_observation"
    TOPIC_REQUIRED = "topic_required"
    ANCHOR_ENTITY_REQUIRED = "anchor_entity_required"
    ANCHOR_ENTITY_SCALAR_MAP_REQUIRED = "anchor_entity_scalar_map_required"
    CANONICAL_ANSWER_JSON_REQUIRED = "canonical_answer_json_required"
    CANONICAL_ANSWER_JSON_INVALID = "canonical_answer_json_invalid"
    DIFFICULTY_VECTOR_REQUIRED = "difficulty_vector_required"
    QUESTION_REQUIRED = "question_required"
    QUESTION_ENTITY_BLOCK_REQUIRED = "question_entity_block_required"
    QUESTION_ENTITY_BLOCK_INVALID_JSON = "question_entity_block_invalid_json"
    QUESTION_ENTITY_BLOCK_MISMATCH = "question_entity_block_mismatch"
    QUESTION_BODY_REQUIRED = "question_body_required"
    PLACEHOLDER_TOKENS_NOT_ALLOWED = "placeholder_tokens_not_allowed"
    QUESTION_INTERNAL_SCHEMA_LEAK = "question_internal_schema_leak"
    QUESTION_RAW_IDENTIFIER_LEAK = "question_raw_identifier_leak"
    QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN = "question_entity_placeholder_forbidden"
    QUESTION_ANCHOR_ENTITY_LEAK = "question_anchor_entity_leak"
    ANCHOR_ENTITY_CHANGED = "anchor_entity_changed"
    TEMPORAL_ORDERING_NOT_GROUNDED = "temporal_ordering_not_grounded"
    GLOBAL_RANKING_OUTSIDE_ANCHOR_SCOPE = "global_ranking_outside_anchor_scope"
    COUNT_LABEL_REQUIRES_COUNT_EVIDENCE = "count_label_requires_count_evidence"
    COUNT_LABEL_OUTSIDE_ANCHOR_SCOPE = "count_label_outside_anchor_scope"
    INITIAL_LABEL_TOO_BROAD = "initial_label_too_broad"
    LABEL_SINGLE_TOOL_DERIVABLE = "label_single_tool_derivable"
    LABEL_REPEATS_ANCHOR_ENTITY = "label_repeats_anchor_entity"
    LABEL_BLANK_STRING_FORBIDDEN = "label_blank_string_forbidden"
    LABEL_IDENTIFIER_CHAIN_FORBIDDEN = "label_identifier_chain_forbidden"
    LABEL_VALUES_NOT_GROUNDED = "label_values_not_grounded"
    DIFFICULTY_WEAKENED = "difficulty_weakened"
    REQUIRED_DIFFICULTY_AXIS_NOT_STRENGTHENED = "required_difficulty_axis_not_strengthened"
    REQUIRED_DIFFICULTY_AXIS_NOT_RELAXED = "required_difficulty_axis_not_relaxed"
    SEARCH_COST_IDENTIFIER_SHORTCUT_FORBIDDEN = "search_cost_identifier_shortcut_forbidden"
    REQUIRED_LABEL_AXIS_NOT_STRENGTHENED = "required_label_axis_not_strengthened"
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
        SubmitDraftErrorCode.QUESTION_INTERNAL_SCHEMA_LEAK,
        SubmitDraftErrorCode.QUESTION_RAW_IDENTIFIER_LEAK,
        SubmitDraftErrorCode.QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN,
        SubmitDraftErrorCode.QUESTION_ANCHOR_ENTITY_LEAK,
        SubmitDraftErrorCode.ANCHOR_ENTITY_CHANGED,
        SubmitDraftErrorCode.TEMPORAL_ORDERING_NOT_GROUNDED,
        SubmitDraftErrorCode.GLOBAL_RANKING_OUTSIDE_ANCHOR_SCOPE,
        SubmitDraftErrorCode.COUNT_LABEL_REQUIRES_COUNT_EVIDENCE,
        SubmitDraftErrorCode.COUNT_LABEL_OUTSIDE_ANCHOR_SCOPE,
        SubmitDraftErrorCode.INITIAL_LABEL_TOO_BROAD,
        SubmitDraftErrorCode.LABEL_SINGLE_TOOL_DERIVABLE,
        SubmitDraftErrorCode.LABEL_REPEATS_ANCHOR_ENTITY,
        SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN,
        SubmitDraftErrorCode.LABEL_IDENTIFIER_CHAIN_FORBIDDEN,
        SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED,
        SubmitDraftErrorCode.DIFFICULTY_WEAKENED,
        SubmitDraftErrorCode.REQUIRED_DIFFICULTY_AXIS_NOT_STRENGTHENED,
        SubmitDraftErrorCode.REQUIRED_DIFFICULTY_AXIS_NOT_RELAXED,
        SubmitDraftErrorCode.SEARCH_COST_IDENTIFIER_SHORTCUT_FORBIDDEN,
        SubmitDraftErrorCode.REQUIRED_LABEL_AXIS_NOT_STRENGTHENED,
        SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED,
    }
)


class DifficultyAdjustmentMode(StrEnum):
    STRENGTHEN = "strengthen"
    RELAX = "relax"


class _ParsedCanonicalAnswerJson(str):
    """Validated JSON string with an attached parsed-object cache.

    The `parsed` attribute is an optimization only. If model round-tripping turns
    this subclass back into a plain `str`, `SubmitDraftPayload.canonical_answer`
    falls back to `json.loads` and remains correct.
    """

    __slots__ = ("parsed",)

    def __new__(cls, value: str, *, parsed: object) -> _ParsedCanonicalAnswerJson:
        instance = str.__new__(cls, value)
        instance.parsed = parsed
        return instance


def _error_code_values(
    codes: list[SubmitDraftErrorCode] | tuple[SubmitDraftErrorCode, ...],
) -> list[str]:
    return [code.value for code in codes]


class SubmitDraftPayload(StrictModel):
    topic: str = Field(
        min_length=1,
        description="Selected topic string derived from the grounded label and evidence.",
    )
    canonical_answer_json: str = Field(
        min_length=1,
        description="Canonical answer as a JSON string. This is the exact label used for EM scoring.",
    )
    anchor_entity: dict[str, object] = Field(
        min_length=1,
        description=(
            "Mandatory anchor entity as a flat JSON object from primary-key field name to scalar value, "
            'for example {"<pk_name>": 123}. Never omit this field and do not nest it under keys such as '
            "entity_type, primary_key, primary_keys, or metadata."
        )
    )
    difficulty_vector: DifficultyVectorContract = Field(
        description="Declared difficulty levels for search_cost, solution_space, and constraint_density."
    )
    question: str = Field(
        min_length=1,
        description=(
            "Full user-facing prompt in the configured task language, starting with "
            "the exact literal tags <entity> ... </entity> on its own lines, followed by a blank line and the natural user request body. "
            "Never replace <entity> with an entity-specific tag such as <customer> or <film>. "
            "The entity block must exactly match anchor_entity."
        ),
    )

    @field_validator("canonical_answer_json")
    @classmethod
    def _validate_canonical_answer_json(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("canonical_answer_json must not be blank")
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise ValueError("canonical_answer_json must be a valid JSON string") from exc
        return _ParsedCanonicalAnswerJson(normalized, parsed=parsed)

    @field_validator("topic", "question")
    @classmethod
    def _validate_non_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text fields must not be blank")
        return normalized

    @field_validator("anchor_entity")
    @classmethod
    def _validate_anchor_entity(cls, value: dict[str, object]) -> dict[str, object]:
        return _normalize_anchor_entity_map(value)

    @property
    def canonical_answer(self) -> object:
        value = self.canonical_answer_json
        if isinstance(value, _ParsedCanonicalAnswerJson):
            return value.parsed
        return json.loads(value)


SubmitDraftPayload.model_rebuild()


type AnchorEntityScalar = str | int | float | bool | None


class SubmitDraftToolPayload(StrictModel):
    topic: str = Field(
        min_length=1,
        description="Selected topic string derived from the grounded label and evidence.",
    )
    canonical_answer_json: str = Field(
        min_length=1,
        description="Canonical answer as a JSON string. This is the exact label used for EM scoring.",
    )
    anchor_entity: Json[dict[str, AnchorEntityScalar]] = Field(
        description=(
            "Mandatory anchor entity encoded as a compact JSON object string from primary-key field name "
            'to scalar value, for example "{\\"<pk_name>\\": 123}". Never omit this field and do not '
            "nest it under keys such as entity_type, primary_key, primary_keys, or metadata."
        )
    )
    difficulty_vector: DifficultyVectorContract = Field(
        description="Declared difficulty levels for search_cost, solution_space, and constraint_density."
    )
    question: str = Field(
        min_length=1,
        description=(
            "Full user-facing prompt in the configured task language, starting with "
            "the exact literal tags <entity> ... </entity> on its own lines, followed by a blank line and the natural user request body. "
            "Never replace <entity> with an entity-specific tag such as <customer> or <film>. "
            "The entity block must exactly match anchor_entity."
        ),
    )

    @field_validator("anchor_entity")
    @classmethod
    def _validate_anchor_entity(cls, value: dict[str, AnchorEntityScalar]) -> dict[str, AnchorEntityScalar]:
        return _normalize_anchor_entity_map(dict(value))

    def to_submit_payload(self) -> SubmitDraftPayload:
        return SubmitDraftPayload.model_validate(
            {
                "topic": self.topic,
                "canonical_answer_json": self.canonical_answer_json,
                "anchor_entity": dict(self.anchor_entity),
                "difficulty_vector": self.difficulty_vector.model_dump(mode="json"),
                "question": self.question,
            }
        )


SubmitDraftToolPayload.model_rebuild()


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


def _difficulty_axis_deltas(
    *,
    previous: DifficultyVectorContract,
    current: DifficultyVectorContract,
) -> tuple[list[DifficultyAxis], list[DifficultyAxis]]:
    previous_flat = flatten_difficulty_vector(previous)
    current_flat = flatten_difficulty_vector(current)
    increased: list[DifficultyAxis] = []
    decreased: list[DifficultyAxis] = []
    for axis, previous_value in previous_flat.items():
        current_value = current_flat.get(axis, previous_value)
        if current_value > previous_value:
            increased.append(axis)
        elif current_value < previous_value:
            decreased.append(axis)
    return increased, decreased


def _placeholder_tokens(payload: object) -> list[str]:
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    return sorted({token for token in _FORBIDDEN_PLACEHOLDER_TOKENS if token in serialized})


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
        if value and isinstance(value[0], dict):
            field_names = [
                str(key)
                for key in list(value[0].keys())[
                    : config.synthesis.runtime.label_preview_field_limit
                ]
            ]
        return {
            "root_type": "array",
            "slot_count": len(value),
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
        if not value:
            return 0
        return sum(_answer_slot_count(item) for item in value)
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
        raw_canonical = payload.get("canonical_answer_json")
        if isinstance(raw_canonical, str):
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
            canonical_json(canonical_answer, default=str)
            if canonical_answer is not None
            else None
        ),
    }


def _label_change_summary(
    *,
    previous: dict[str, object] | None,
    current: dict[str, object],
) -> dict[str, object]:
    previous_field_names = list(previous.get("canonical_answer_field_names", [])) if previous else []
    current_field_names = list(current.get("canonical_answer_field_names", []))
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


def _is_single_tool_derivable(answer: object, result: object) -> bool:
    if canonical_json(answer, default=str) == canonical_json(result, default=str):
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
        if canonical_json(answer, default=str) == canonical_json(prefix, default=str):
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


def _single_tool_derivation_record(
    answer: object,
    tool_calls: list[dict[str, object]],
) -> dict[str, object] | None:
    for record in tool_calls:
        if _is_single_tool_derivable(answer, record.get("result")):
            return record
    return None


def _value_tree_contains_scalar(value: object, target: object) -> bool:
    if value == target:
        return True
    if isinstance(value, dict):
        return any(_value_tree_contains_scalar(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_value_tree_contains_scalar(item, target) for item in value)
    return False


def _tool_call_depends_on_anchor_entity(
    record: dict[str, object],
    *,
    anchor_entity: dict[str, object],
) -> bool:
    params = record.get("params")
    if not isinstance(params, dict):
        return False
    return any(_value_tree_contains_scalar(params, value) for value in anchor_entity.values())


def _answer_uses_only_identifier_fields(answer: object) -> bool:
    def _keys_are_identifier_only(mapping: dict[str, object]) -> bool:
        if not mapping:
            return False
        return all(isinstance(key, str) and _is_identifier_field_name(key) for key in mapping)

    def _is_identifier_only_tree(value: object) -> bool:
        if isinstance(value, dict):
            if _keys_are_identifier_only(value):
                return True
            child_values = list(value.values())
            return bool(child_values) and all(_is_identifier_only_tree(item) for item in child_values)
        if isinstance(value, list):
            return bool(value) and all(_is_identifier_only_tree(item) for item in value)
        return False

    if isinstance(answer, dict):
        return _is_identifier_only_tree(answer)
    if isinstance(answer, list) and answer and all(isinstance(item, dict) for item in answer):
        return _is_identifier_only_tree(answer)
    return False


def _answer_repeats_anchor_entity(
    answer: object,
    *,
    anchor_entity: dict[str, object],
) -> bool:
    if not anchor_entity:
        return False

    def _contains_anchor(mapping: dict[str, object]) -> bool:
        return all(mapping.get(key) == value for key, value in anchor_entity.items())

    if isinstance(answer, dict):
        return _contains_anchor(answer)
    if isinstance(answer, list) and answer and all(isinstance(item, dict) for item in answer):
        return all(_contains_anchor(item) for item in answer)
    return False


def _collect_observed_strings(value: object, *, strings: set[str]) -> None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            strings.add(normalized)
        return
    if isinstance(value, datetime | date):
        normalized = value.isoformat().strip().lower()
        if normalized:
            strings.add(normalized)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_observed_strings(item, strings=strings)
        return
    if isinstance(value, list):
        for item in value:
            _collect_observed_strings(item, strings=strings)


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


def _blank_string_paths(value: object, *, path: str = "$") -> list[str]:
    blank_paths: list[str] = []
    if isinstance(value, str):
        return [path] if not value.strip() else []
    if isinstance(value, dict):
        for key, item in value.items():
            blank_paths.extend(_blank_string_paths(item, path=f"{path}.{key}"))
        return blank_paths
    if isinstance(value, list):
        for index, item in enumerate(value):
            blank_paths.extend(_blank_string_paths(item, path=f"{path}[{index}]"))
        return blank_paths
    return []


def _ungrounded_answer_strings(
    answer: object,
    *,
    observed_strings: set[str],
) -> list[str]:
    answer_strings: list[str] = []
    _collect_answer_strings(answer, sink=answer_strings)
    ungrounded: list[str] = []
    for value in answer_strings:
        if value in observed_strings:
            continue
        ungrounded.append(value)
    return sorted(dict.fromkeys(ungrounded))


_DATETIME_LITERAL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2}$")


def _format_ungrounded_value_guidance(diagnostics: dict[str, object] | None) -> str:
    if diagnostics is None:
        return ""
    raw_values = diagnostics.get("ungrounded_strings")
    if not isinstance(raw_values, list):
        return ""
    values = [str(value) for value in raw_values if isinstance(value, str)]
    if not values:
        return ""
    preview = ", ".join(repr(value) for value in values[:3])
    message = f" Ungrounded values included: {preview}."
    name_like_values = [
        value
        for value in values
        if " " in value
        and any(character.isalpha() for character in value)
        and not any(character.isdigit() for character in value)
    ]
    datetime_like_values = [value for value in values if _DATETIME_LITERAL_RE.fullmatch(value)]
    if name_like_values:
        message += (
            " If the tool response exposed first_name and last_name separately, keep them as separate answer fields instead of merging them into one full-name string."
        )
    if datetime_like_values:
        message += (
            " If you use a date or timestamp field, copy the exact raw value from the chosen tool response row without changing its formatting."
        )
    return message


_IDENTIFIER_FIELD_TOKEN_RE = re.compile(
    r"(?<![a-z0-9_])[a-z][a-z0-9_]*_ids?(?![a-z0-9_])",
    re.IGNORECASE,
)
_TEMPORAL_FIELD_TOKEN_RE = re.compile(
    r"(?:^|_)(?:date|time|timestamp|created|updated|start|end|due|expires?|expiry|scheduled|last_update)(?:$|_)",
    re.IGNORECASE,
)
_TEMPORAL_CONSTRAINT_TOKENS = (
    "recent",
    "latest",
    "earliest",
    "oldest",
    "newest",
    "first",
    "last",
)
_GLOBAL_SCOPE_TOKENS = (
    "global",
    "overall",
    "across all",
    "among all",
    "entire database",
    "전역",
    "전체 데이터베이스",
    "전체 고객",
    "전체 사용자",
    "모든 고객",
    "모든 사용자",
)
_COUNT_SEMANTIC_TOKENS = frozenset({"count", "counting", "cardinality"})


def _is_identifier_field_name(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized.endswith("_id") or normalized.endswith("_ids")


def _contains_raw_identifier_token(text: str) -> bool:
    return _IDENTIFIER_FIELD_TOKEN_RE.search(text.lower()) is not None


def _contains_entity_placeholder_token(text: str) -> bool:
    lowered = text.lower()
    return "<entity>" in lowered or "&lt;entity&gt;" in lowered


def _uses_temporal_ordering_language(text: str | None) -> bool:
    if not text:
        return False
    combined = normalize_words(text, lowercase=True)
    return any(token in combined for token in _TEMPORAL_CONSTRAINT_TOKENS)


def _observed_temporal_surface(tool_calls: list[dict[str, object]]) -> bool:
    def _walk(value: object) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                if _TEMPORAL_FIELD_TOKEN_RE.search(str(key)):
                    return True
                if _walk(item):
                    return True
            return False
        if isinstance(value, list):
            return any(_walk(item) for item in value)
        return False

    return any(_walk(record.get("result")) for record in tool_calls)


def _uses_unanchored_global_ranking(
    tool_calls: list[dict[str, object]],
    *,
    anchor_entity: dict[str, object],
) -> bool:
    for record in tool_calls:
        tool_name = str(record.get("tool_name", ""))
        if not tool_name.startswith(("rank_", "top_")):
            continue
        if _tool_call_depends_on_anchor_entity(record, anchor_entity=anchor_entity):
            continue
        return True
    return False


def _mentions_global_scope(question_body: str | None) -> bool:
    combined = normalize_words(question_body or "", lowercase=True)
    return any(token in combined for token in _GLOBAL_SCOPE_TOKENS)


def _count_semantics_present(
    answer: object,
    question_body: str | None,
) -> bool:
    if isinstance(answer, dict):
        count_keys = [key for key in answer if isinstance(key, str) and "count" in key.lower()]
        if count_keys:
            return True
        return any(_count_semantics_present(item, question_body) for item in answer.values())
    if isinstance(answer, list):
        return any(_count_semantics_present(item, question_body) for item in answer)
    if not isinstance(answer, (int, float)):
        return False
    combined_tokens: set[str] = set()
    combined_tokens.update(normalize_words(question_body or "", lowercase=True).split())
    return not combined_tokens.isdisjoint(_COUNT_SEMANTIC_TOKENS)


def _observed_count_evidence(tool_calls: list[dict[str, object]]) -> bool:
    for record in tool_calls:
        if _tool_call_is_count_evidence(record):
            return True
    return False


def _observed_anchor_scoped_count_evidence(
    tool_calls: list[dict[str, object]],
    *,
    anchor_entity: dict[str, object],
) -> bool:
    for record in tool_calls:
        if not _tool_call_is_count_evidence(record):
            continue
        if _tool_call_depends_on_anchor_entity(record, anchor_entity=anchor_entity):
            return True
    return False


def _tool_call_is_count_evidence(record: dict[str, object]) -> bool:
    tool_name = str(record.get("tool_name", "")).strip()
    if tool_name.startswith("count_"):
        return True
    if not tool_name.startswith("calc_"):
        return False
    params = record.get("params")
    if not isinstance(params, dict):
        return False
    return str(params.get("fn", "")).strip().lower() == "count"


def _distinct_tool_name_count(tool_calls: list[dict[str, object]]) -> int:
    return len(
        {
            str(record.get("tool_name", "")).strip()
            for record in tool_calls
            if str(record.get("tool_name", "")).strip()
        }
    )


def _anchor_scoped_tool_call_count(
    tool_calls: list[dict[str, object]],
    *,
    anchor_entity: dict[str, object],
) -> int:
    return sum(
        1
        for record in tool_calls
        if _tool_call_depends_on_anchor_entity(record, anchor_entity=anchor_entity)
    )


def _observed_anchor_readable_string_surface(
    tool_calls: list[dict[str, object]],
    *,
    anchor_entity: dict[str, object],
) -> bool:
    observed_strings: set[str] = set()
    for record in tool_calls:
        if not _tool_call_depends_on_anchor_entity(record, anchor_entity=anchor_entity):
            continue
        _collect_observed_strings(record.get("result"), strings=observed_strings)
    return bool(observed_strings)


def _answer_has_multi_item_collection(value: object) -> bool:
    if isinstance(value, list):
        if len(value) > 1:
            return True
        return any(_answer_has_multi_item_collection(item) for item in value)
    if isinstance(value, dict):
        return any(_answer_has_multi_item_collection(item) for item in value.values())
    return False


def _human_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _field_preservation_guidance(label_data: dict[str, object] | None) -> str:
    if not isinstance(label_data, dict):
        return ""
    raw_field_names = label_data.get("canonical_answer_field_names")
    if not isinstance(raw_field_names, list):
        return ""
    readable_field_names = [
        str(field_name)
        for field_name in raw_field_names
        if isinstance(field_name, str) and not _is_identifier_field_name(field_name)
    ]
    if not readable_field_names:
        return ""
    preview = readable_field_names[:3]
    return (
        " Preserve grounded readable answer slots such as "
        f"{_human_join(preview)} if they still fit the same anchored user need."
    )


def _too_easy_retry_guidance(
    *,
    label_data: dict[str, object] | None,
) -> str:
    preserve_guidance = _field_preservation_guidance(label_data)
    return (
        " Stay inside the same connected anchored neighborhood and keep the same anchored user need."
        f"{preserve_guidance} Choose exactly one axis from the observed data and current label. "
        "Use search_cost if the path is too shallow and needs one more connected grounded hop or fact. "
        "Use solution_space if the answer needs one more grounded slot or one more connected ordered item. "
        "Use constraint_density if the same path needs one more grounded rule, tie-breaker, or filter. "
        "Do not replace the current readable path with a disconnected lookup, an id-only fallback, or a simpler global count."
    )


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

def _question_repeats_anchor_entity(
    question: str,
    *,
    anchor_entity: dict[str, object],
) -> bool:
    lowered = question.lower()
    anchor_items = tuple(sorted((str(key), str(value)) for key, value in anchor_entity.items()))
    return any(pattern.search(lowered) for pattern in _anchor_entity_patterns(anchor_items))


@lru_cache(maxsize=256)
def _anchor_entity_patterns(
    anchor_items: tuple[tuple[str, str], ...],
) -> tuple[re.Pattern[str], ...]:
    compiled_patterns: list[re.Pattern[str]] = []
    for raw_key, raw_value in anchor_items:
        key = raw_key.strip().lower()
        if not key:
            continue
        base_key = key
        if base_key.endswith("_ids"):
            base_key = base_key[:-4]
        elif base_key.endswith("_id"):
            base_key = base_key[:-3]
        if not base_key:
            continue
        key_pattern = re.escape(normalize_words(base_key, lowercase=True))
        value = raw_value.strip().lower()
        if not value:
            continue
        value_pattern = re.escape(value)
        compiled_patterns.extend(
            (
                re.compile(
                    rf"(?<![a-z0-9_]){key_pattern}\s*[:#-]?\s*{value_pattern}(?![a-z0-9_])"
                ),
                re.compile(
                    rf"(?<![a-z0-9_]){value_pattern}(?:번)?\s*[:#-]?\s*{key_pattern}(?![a-z0-9_])"
                ),
            )
        )
    return tuple(compiled_patterns)


@dataclass(slots=True)
class SubmitDraftController:
    config: AppConfig
    requested_topic: str
    solver_orchestrator: SolverOrchestrator
    build_draft: Any
    max_submissions: int
    phase_monitor: PipelinePhaseMonitorLogger | None = None
    forbidden_question_tokens: frozenset[str] = field(default_factory=frozenset)
    accepted_draft: SynthesisTaskDraft | None = None
    attempts: list[SubmitDraftAttemptRecord] = field(default_factory=list)
    strongest_difficulty_vector: DifficultyVectorContract = field(
        default_factory=DifficultyVectorContract
    )
    required_axis: DifficultyAxis | None = None
    required_axis_mode: DifficultyAdjustmentMode | None = None
    required_axis_reference_vector: DifficultyVectorContract | None = None
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

    def submissions_left(self) -> int:
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
            elif location == ("canonical_answer_json",):
                error_codes.append(SubmitDraftErrorCode.CANONICAL_ANSWER_JSON_INVALID)
            elif location and location[0] == "anchor_entity":
                if error_type == "value_error":
                    error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED)
                else:
                    anchor_code = (
                        SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED
                        if error_type == "missing" and len(location) == 1
                        else SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED
                    )
                    error_codes.append(anchor_code)
            else:
                error_codes.append(SubmitDraftErrorCode.SUBMIT_PAYLOAD_INVALID)
        raw_question = parsed.get("question")
        if isinstance(raw_question, str):
            _, question_body, prompt_error = _split_entity_wrapped_prompt(raw_question)
            if prompt_error is not None:
                error_codes.append(prompt_error)
            elif question_body is not None and _contains_raw_identifier_token(question_body):
                error_codes.append(SubmitDraftErrorCode.QUESTION_RAW_IDENTIFIER_LEAK)
        raw_canonical = parsed.get("canonical_answer_json")
        if isinstance(raw_canonical, str):
            try:
                raw_answer = json.loads(raw_canonical)
            except json.JSONDecodeError:
                raw_answer = None
            if raw_answer is not None and _answer_uses_only_identifier_fields(raw_answer):
                error_codes.append(SubmitDraftErrorCode.LABEL_IDENTIFIER_CHAIN_FORBIDDEN)
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
        if not payload.anchor_entity:
            error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED)
        elif self._locked_anchor_entity is not None and payload.anchor_entity != self._locked_anchor_entity:
            error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_CHANGED)
            invalid_diagnostics["locked_anchor_entity"] = self._locked_anchor_entity
        elif payload.anchor_entity:
            self._locked_anchor_entity = dict(payload.anchor_entity)
        if _uses_temporal_ordering_language(question_body) and not _observed_temporal_surface(
            self._raw_atomic_tool_calls
        ):
            error_codes.append(SubmitDraftErrorCode.TEMPORAL_ORDERING_NOT_GROUNDED)
        placeholder_tokens = _placeholder_tokens(
            {
                "topic": payload.topic,
                "canonical_answer_json": payload.canonical_answer_json,
                "anchor_entity": payload.anchor_entity,
                "question": payload.question,
            }
        )
        if placeholder_tokens:
            error_codes.append(SubmitDraftErrorCode.PLACEHOLDER_TOKENS_NOT_ALLOWED)
        canonical_answer = payload.canonical_answer
        if prompt_error is not None:
            error_codes.append(prompt_error)
        elif prompt_anchor_entity != payload.anchor_entity:
            error_codes.append(SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_MISMATCH)
        if (
            _uses_unanchored_global_ranking(
                self._raw_atomic_tool_calls,
                anchor_entity=payload.anchor_entity,
            )
            and not _mentions_global_scope(question_body)
        ):
            error_codes.append(SubmitDraftErrorCode.GLOBAL_RANKING_OUTSIDE_ANCHOR_SCOPE)
        constraint_count = 0
        label_signature = canonical_json(canonical_answer, default=str)
        label_slot_count = _answer_slot_count(canonical_answer)
        blank_paths = _blank_string_paths(canonical_answer)
        if blank_paths:
            error_codes.append(SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN)
            invalid_diagnostics["blank_string_paths"] = blank_paths[
                : self.config.synthesis.runtime.diagnostic_item_limit
            ]
        derivation_record = _single_tool_derivation_record(
            canonical_answer,
            self._raw_atomic_tool_calls,
        )
        if derivation_record is not None:
            error_codes.append(SubmitDraftErrorCode.LABEL_SINGLE_TOOL_DERIVABLE)
            invalid_diagnostics["single_tool_name"] = str(
                derivation_record.get("tool_name", "unknown_tool")
            )
            invalid_diagnostics["single_tool_scope"] = (
                "anchor_scoped"
                if _tool_call_depends_on_anchor_entity(
                    derivation_record,
                    anchor_entity=payload.anchor_entity,
                )
                else "global"
            )
        if _answer_repeats_anchor_entity(canonical_answer, anchor_entity=payload.anchor_entity):
            error_codes.append(SubmitDraftErrorCode.LABEL_REPEATS_ANCHOR_ENTITY)
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
                    anchor_entity=payload.anchor_entity,
                )
            )
        if question_body is not None:
            question_lower = question_body.lower()
            if _contains_entity_placeholder_token(question_lower):
                error_codes.append(SubmitDraftErrorCode.QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN)
            has_raw_identifier_leak = False
            has_anchor_entity_leak = False
            if _contains_raw_identifier_token(question_lower):
                has_raw_identifier_leak = True
                error_codes.append(SubmitDraftErrorCode.QUESTION_RAW_IDENTIFIER_LEAK)
            if _question_repeats_anchor_entity(
                question_body,
                anchor_entity=payload.anchor_entity,
            ):
                has_anchor_entity_leak = True
                error_codes.append(SubmitDraftErrorCode.QUESTION_ANCHOR_ENTITY_LEAK)
            if any(token in question_lower for token in self.forbidden_question_tokens):
                error_codes.append(SubmitDraftErrorCode.QUESTION_INTERNAL_SCHEMA_LEAK)
        if _answer_uses_only_identifier_fields(canonical_answer):
            error_codes.append(SubmitDraftErrorCode.LABEL_IDENTIFIER_CHAIN_FORBIDDEN)
            invalid_diagnostics["anchor_path_has_readable_strings"] = (
                _observed_anchor_readable_string_surface(
                    self._raw_atomic_tool_calls,
                    anchor_entity=payload.anchor_entity,
                )
            )
        if _count_semantics_present(canonical_answer, question_body) and not _observed_count_evidence(
            self._raw_atomic_tool_calls
        ):
            error_codes.append(SubmitDraftErrorCode.COUNT_LABEL_REQUIRES_COUNT_EVIDENCE)
        if (
            question_body is not None
            and _count_semantics_present(canonical_answer, question_body)
            and _observed_count_evidence(self._raw_atomic_tool_calls)
            and not _mentions_global_scope(question_body)
            and not _observed_anchor_scoped_count_evidence(
                self._raw_atomic_tool_calls,
                anchor_entity=payload.anchor_entity,
            )
        ):
            error_codes.append(SubmitDraftErrorCode.COUNT_LABEL_OUTSIDE_ANCHOR_SCOPE)
        if (
            not self.attempts
            and self.required_axis is None
            and _answer_has_multi_item_collection(canonical_answer)
        ):
            error_codes.append(SubmitDraftErrorCode.INITIAL_LABEL_TOO_BROAD)

        reference_vector = self.required_axis_reference_vector or self.strongest_difficulty_vector
        increased_axes, decreased_axes = _difficulty_axis_deltas(
            previous=reference_vector,
            current=payload.difficulty_vector,
        )
        self.required_axis = None
        if self.required_axis_mode is None:
            if decreased_axes:
                error_codes.append(SubmitDraftErrorCode.DIFFICULTY_WEAKENED)
        elif self.required_axis_mode is DifficultyAdjustmentMode.STRENGTHEN:
            if decreased_axes or len(increased_axes) != 1:
                error_codes.append(SubmitDraftErrorCode.REQUIRED_DIFFICULTY_AXIS_NOT_STRENGTHENED)
            else:
                self.required_axis = increased_axes[0]
                if self._last_label_signature is not None and label_signature == self._last_label_signature:
                    error_codes.append(SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED)
        elif self.required_axis_mode is DifficultyAdjustmentMode.RELAX:
            if increased_axes or len(decreased_axes) != 1:
                error_codes.append(SubmitDraftErrorCode.REQUIRED_DIFFICULTY_AXIS_NOT_RELAXED)
            else:
                self.required_axis = decreased_axes[0]

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
        self.strongest_difficulty_vector = _merge_strongest_difficulty_vector(
            self.strongest_difficulty_vector,
            payload.difficulty_vector,
        )

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
            self.required_axis = None
            self.required_axis_mode = DifficultyAdjustmentMode.STRENGTHEN
            self.required_axis_reference_vector = self.strongest_difficulty_vector
            strengthening_guidance = _too_easy_retry_guidance(
                label_data=_monitor_label_data(payload, config=self.config),
            )
            return self._record_rejection(
                submission_index=submission_index,
                message=_render_structured_message(
                    kind="RejectedError",
                    result=(
                        "solver pass rate "
                        f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}."
                    ),
                    primary=(
                        "Choose exactly one difficulty axis yourself from the observed data and current label. "
                        "Increase exactly one of search_cost, solution_space, or constraint_density above the previous level, and leave the other two unchanged."
                    ),
                    important=(
                        "Keep the same anchored user need and preserve the other two axes at least as strong as before. "
                        f"{strengthening_guidance.strip()}"
                    ),
                    next_step=(
                        "Make at least one new atomic tool call, gather new grounded evidence, and strengthen only that one axis before resubmitting."
                    ),
                    attempts_left=max(0, attempts_left_after),
                ),
                error_codes=[SubmitDraftErrorCode.REJECT_TOO_EASY],
                pass_rate=quality_gate_summary.pass_rate,
                matched_solver_runs=quality_gate_summary.matched_solver_runs,
                total_solver_runs=quality_gate_summary.total_solver_runs,
                diagnostics={"requested_axis": None},
                payload=payload,
                search_cost_observations=search_cost_observations,
            )

        self.required_axis = None
        self.required_axis_mode = DifficultyAdjustmentMode.RELAX
        self.required_axis_reference_vector = payload.difficulty_vector
        return self._record_rejection(
            submission_index=submission_index,
            message=_render_structured_message(
                kind="RejectedError",
                result=(
                    "solver pass rate "
                    f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}."
                ),
                primary=(
                    "This draft is too hard for the configured band. Choose exactly one difficulty axis yourself from the current label and observed data. "
                    "Reduce exactly one of search_cost, solution_space, or constraint_density by one grounded step, and leave the other two unchanged."
                ),
                important=(
                    "Keep the same anchored user need while simplifying only that one axis before changing topic or anchor."
                ),
                attempts_left=max(0, attempts_left_after),
            ),
            error_codes=[SubmitDraftErrorCode.REJECT_TOO_HARD],
            pass_rate=quality_gate_summary.pass_rate,
            matched_solver_runs=quality_gate_summary.matched_solver_runs,
            total_solver_runs=quality_gate_summary.total_solver_runs,
            diagnostics={
                "terminal_rejection": False,
                "requested_axis": None,
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
            SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED: "Rejected. anchor_entity must contain at least one primary-key value.",
            SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED: (
                "Rejected. anchor_entity must be a flat JSON object mapping one or more primary-key field names to scalar values, for example {\"customer_id\": 123} or {\"order_id\": 7, \"line_no\": 2}. Do not nest it under entity_type, primary_key, or primary_keys."
            ),
            SubmitDraftErrorCode.CANONICAL_ANSWER_JSON_REQUIRED: "Rejected. canonical_answer_json is required.",
            SubmitDraftErrorCode.DIFFICULTY_VECTOR_REQUIRED: "Rejected. difficulty_vector is required.",
            SubmitDraftErrorCode.QUESTION_REQUIRED: "Rejected. question is required.",
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_REQUIRED: (
                "Rejected. question must already be the full user prompt in this exact shape: <entity> newline JSON newline </entity> blank line user request."
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_INVALID_JSON: (
                "Rejected. The <entity> block must contain a valid flat JSON object with one or more primary-key fields."
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_MISMATCH: (
                "Rejected. The JSON inside the <entity> block must exactly match anchor_entity."
            ),
            SubmitDraftErrorCode.QUESTION_BODY_REQUIRED: (
                "Rejected. After the <entity> block, include a natural user request body."
            ),
            SubmitDraftErrorCode.CANONICAL_ANSWER_JSON_INVALID: (
                "Rejected. canonical_answer_json must be a valid JSON string."
            ),
            SubmitDraftErrorCode.PLACEHOLDER_TOKENS_NOT_ALLOWED: (
                "Rejected. Replace every placeholder token with grounded names and values from the current database."
            ),
            SubmitDraftErrorCode.QUESTION_INTERNAL_SCHEMA_LEAK: (
                "Rejected. Rewrite the user-facing question for a non-technical user who knows nothing about the database schema. Remove raw table names, bridge-table names, SQL keywords, and other internal data-model language."
            ),
            SubmitDraftErrorCode.QUESTION_RAW_IDENTIFIER_LEAK: (
                "Rejected. Rewrite the user-facing question for a user who does not know internal field names. Remove raw identifier names such as <entity>_id and keep identifiers only inside anchor_entity."
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN: (
                "Rejected. Do not repeat the literal <entity> token inside the user-request body. Use it only once as the required XML entity block at the top."
            ),
            SubmitDraftErrorCode.QUESTION_ANCHOR_ENTITY_LEAK: (
                "Rejected. Rewrite the user-request body without repeating the raw anchor entity id. The user only sees the entity block, so refer to that anchor naturally instead of surfacing its internal identifier again."
            ),
            SubmitDraftErrorCode.ANCHOR_ENTITY_CHANGED: (
                "Rejected. Keep the same anchored user entity across retries. The requesting user has not changed, so do not switch anchor_entity to a different person or role while repairing this draft."
            ),
            SubmitDraftErrorCode.TEMPORAL_ORDERING_NOT_GROUNDED: (
                "Rejected. Do not use recent, latest, earliest, first, or similar ordering language unless you directly observed a temporal or sequence field that grounds that ordering. The current evidence path only shows ids or unordered rows, so sampled rows are not a valid notion of recency or priority."
            ),
            SubmitDraftErrorCode.GLOBAL_RANKING_OUTSIDE_ANCHOR_SCOPE: (
                "Rejected. The draft jumps from the anchored user's own records to a global ranking without saying so. Keep rankings local to the anchored scope, or explicitly ask for a global benchmark in the user-facing request and constraints."
            ),
            SubmitDraftErrorCode.COUNT_LABEL_REQUIRES_COUNT_EVIDENCE: (
                "Rejected. A count-like label needs explicit count evidence. Do not infer a total from the first sampled rows you happened to inspect; use a grounded count or aggregate observation for that anchored scope."
            ),
            SubmitDraftErrorCode.COUNT_LABEL_OUTSIDE_ANCHOR_SCOPE: (
                "Rejected. The count evidence is not scoped to the anchored user. For a self-scoped request, only keep a count field if you observed a count or aggregate tool call whose parameters depend on anchor_entity. Otherwise drop the count field or gather anchored count evidence on the same user path instead of using a global total."
            ),
            SubmitDraftErrorCode.INITIAL_LABEL_TOO_BROAD: (
                "Rejected. Start the first judged draft with a smaller anchored label. Use one grounded record, one small object, or one anchored summary that still needs multiple observations. Do not start with a multi-item set, top-few list, or paired bundle before the loop proves a smaller label is too easy."
            ),
            SubmitDraftErrorCode.LABEL_SINGLE_TOOL_DERIVABLE: (
                "Rejected. The canonical answer can be recovered from a single atomic tool call. Redesign the task so the label requires combining multiple observations. A one-hop foreign-key lookup that only returns identifiers is still too weak."
            ),
            SubmitDraftErrorCode.LABEL_REPEATS_ANCHOR_ENTITY: (
                "Rejected. Do not repeat anchor_entity fields inside the canonical answer. The entity block already provides that grounding, so use the answer slots for new grounded information."
            ),
            SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN: (
                "Rejected. The canonical answer contains blank string fields. Every answer field must contain a grounded, non-empty value. Schema orientation alone is not enough; only fields you actually observed in tool results are grounded. If the chosen surface is id-only, keep the same anchored user and switch to grounded counts, dates, amounts, statuses, ordering, or make new anchored tool calls until you observe readable fields."
            ),
            SubmitDraftErrorCode.LABEL_IDENTIFIER_CHAIN_FORBIDDEN: (
                "Rejected. The canonical answer is only a chain of internal identifier fields. A relation made only of ids is still an internal identifier chain. Return user-relevant business values such as names, titles, dates, amounts, counts, or statuses instead. If the current hinted topic keeps forcing id-only answers, choose a better grounded topic for the same anchored user need before resubmitting."
            ),
            SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED: (
                "Rejected. Some label values were not directly grounded in the observed tool results. Schema orientation alone is not enough; only use business strings, dates, and other readable values that you actually observed in real tool outputs, and copy them exactly as they appeared there. Do not shorten names, paraphrase labels, normalize timestamp formatting, or manufacture readable labels by wrapping an id in generic words such as 'staff member 2' or 'order 17'. If the chosen surface is id-only, keep the same anchored user and switch to counts, dates, amounts, statuses, ordering, make new anchored tool calls until you observe readable fields, or choose a better grounded topic for the same anchored user need."
            ),
            SubmitDraftErrorCode.DIFFICULTY_WEAKENED: (
                "Rejected. Do not weaken the declared difficulty vector relative to the strongest prior attempt."
            ),
            SubmitDraftErrorCode.REQUIRED_DIFFICULTY_AXIS_NOT_STRENGTHENED: (
                "Rejected. The requested difficulty axis was not strengthened."
            ),
            SubmitDraftErrorCode.REQUIRED_DIFFICULTY_AXIS_NOT_RELAXED: (
                "Rejected. The requested difficulty axis was not reduced. Keep the same anchored user need and simplify only that axis by one grounded step."
            ),
            SubmitDraftErrorCode.REQUIRED_LABEL_AXIS_NOT_STRENGTHENED: (
                "Rejected. Strengthen the label itself along the requested axis by one grounded step before resubmitting."
            ),
            SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED: (
                "Rejected. After a too-easy result, do not resubmit the same label. Strengthen the canonical answer itself with a new grounded step."
            ),
            SubmitDraftErrorCode.SUBMIT_PAYLOAD_INVALID: (
                "Rejected. submit_draft arguments did not match the required schema."
            ),
            SubmitDraftErrorCode.DRAFT_VALIDATION_FAILED: "Rejected. The submitted draft could not be validated.",
        }
        primary = message_map.get(error_codes[0], "Rejected. Fix the draft and resubmit.")
        if error_codes and error_codes[0] is SubmitDraftErrorCode.LABEL_SINGLE_TOOL_DERIVABLE:
            if (
                diagnostics is not None
                and diagnostics.get("single_tool_scope") == "global"
            ):
                primary = (
                    "Rejected. The canonical answer can be recovered from a single global tool call that does not depend on the anchor entity. "
                    "Keep the label anchored to the selected entity and combine multiple anchored observations."
                )
        if error_codes and error_codes[0] is SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED:
            if diagnostics is not None and diagnostics.get("anchor_path_has_readable_strings") is False:
                primary = (
                    "Rejected. The current anchored evidence path does not expose readable text fields in real tool outputs. "
                    "Stop retrying names, titles, or other readable strings on this same path. Keep the same anchored user and either answer with grounded counts, dates, amounts, statuses, or ordering, or make new anchored tool calls until you actually observe readable fields."
                )
            else:
                primary += _format_ungrounded_value_guidance(diagnostics)
        if error_codes and error_codes[0] is SubmitDraftErrorCode.LABEL_IDENTIFIER_CHAIN_FORBIDDEN:
            if diagnostics is not None and diagnostics.get("anchor_path_has_readable_strings") is False:
                primary = (
                    "Rejected. The current anchored evidence path is still id-only. Do not submit another answer made only of *_id fields on this same path. "
                    "Keep the same anchored user and either answer with grounded counts, dates, amounts, statuses, or ordering, or pivot to a better grounded topic for that same user need."
                )
        if error_codes and error_codes[0] in (
            SubmitDraftErrorCode.REQUIRED_LABEL_AXIS_NOT_STRENGTHENED,
            SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED,
        ):
            primary += _too_easy_retry_guidance(
                label_data=self._last_monitored_label_data,
            )
        preserve_guidance = ""
        if self._last_monitored_label_data is not None:
            preserve_guidance = (
                "Keep the same anchored user need and fix only the failing part when possible. "
                "Do not reset to a different topic, a different anchor, or a simpler global count just to satisfy this feedback."
            )
        additional_messages: list[str] = []
        for error_code in error_codes[1:3]:
            extra = message_map.get(error_code)
            if extra is None:
                continue
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
                "required_axis": self.required_axis.value if self.required_axis is not None else None,
                "required_axis_mode": (
                    self.required_axis_mode.value if self.required_axis_mode is not None else None
                ),
                "required_axis_reference_vector": (
                    self.required_axis_reference_vector.model_dump(mode="json")
                    if self.required_axis_reference_vector is not None
                    else None
                ),
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
    from agents.strict_schema import ensure_strict_json_schema

    params_json_schema = ensure_strict_json_schema(SubmitDraftToolPayload.model_json_schema())

    async def _invoke_tool(_tool_context: Any, input_json: str) -> str:
        parsed = json.loads(input_json) if input_json else {}
        if isinstance(parsed.get("anchor_entity"), dict):
            parsed["anchor_entity"] = json.dumps(
                parsed["anchor_entity"],
                ensure_ascii=False,
                sort_keys=True,
            )
        if "anchor_entity" not in parsed:
            raw_question = parsed.get("question")
            if isinstance(raw_question, str):
                parsed_anchor_entity, _, prompt_error = _split_entity_wrapped_prompt(raw_question)
                if prompt_error is None and parsed_anchor_entity is not None:
                    parsed["anchor_entity"] = json.dumps(
                        parsed_anchor_entity,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
        try:
            tool_payload = SubmitDraftToolPayload.model_validate(parsed)
            payload = tool_payload.to_submit_payload()
        except ValidationError as exc:
            return controller.reject_invalid_payload(parsed=parsed, error=exc)
        return await controller.submit(payload)

    return FunctionTool(
        name="submit_draft",
        description=(
            "Submit a grounded RLVR task draft. "
            "Include topic, canonical_answer_json, anchor_entity, difficulty_vector, and question. "
            "anchor_entity must be a compact JSON object string. "
            "Use only tool-observed evidence; do not invent hidden joins, hidden values, or SQL. "
            "Call this only after you have one verified evidence chain for the label. "
            "If a row gives only references, resolve that chain before using downstream readable fields. "
            "Keep anchor_entity fixed across retries. "
            "question must start with the exact literal tags <entity> and </entity>, then a blank line, then the request body; never substitute <customer>, <film>, or any entity-specific tag. The entity block must match anchor_entity. "
            "Do not submit blank strings, opaque identifiers, identifier-only labels, or labels derivable from one atomic tool call. "
            "After any rejection, make at least one new atomic tool call before resubmitting. "
            "Continue until accepted or budget exhausted."
        ),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=True,
    )
