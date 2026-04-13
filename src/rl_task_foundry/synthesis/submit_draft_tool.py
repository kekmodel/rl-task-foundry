"""submit_draft tool for the single-agent synthesis loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from pydantic import Field, ValidationError, field_validator

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.infra.sdk_helpers import preview_payload
from rl_task_foundry.synthesis.canonicalize import canonical_json
from rl_task_foundry.synthesis.contracts import (
    DIFFICULTY_CRANK_ORDER,
    DifficultyAxis,
    DifficultyVectorContract,
    StrictModel,
    flatten_difficulty_vector,
    is_person_like_identifier,
    normalize_words,
    topic_tokens,
)
from rl_task_foundry.synthesis.phase_monitor import PipelinePhaseMonitorLogger
from rl_task_foundry.synthesis.prompts import difficulty_axis_feedback

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
    CONSTRAINT_SUMMARY_REQUIRED = "constraint_summary_required"
    LABEL_SUMMARY_REQUIRED = "label_summary_required"
    PLACEHOLDER_TOKENS_NOT_ALLOWED = "placeholder_tokens_not_allowed"
    QUESTION_INTERNAL_SCHEMA_LEAK = "question_internal_schema_leak"
    QUESTION_RAW_IDENTIFIER_LEAK = "question_raw_identifier_leak"
    QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN = "question_entity_placeholder_forbidden"
    QUESTION_ANCHOR_ENTITY_LEAK = "question_anchor_entity_leak"
    QUESTION_SELF_PERSPECTIVE_REQUIRED = "question_self_perspective_required"
    SELF_ENTITY_ANCHOR_REQUIRED = "self_entity_anchor_required"
    ANCHOR_ENTITY_CHANGED = "anchor_entity_changed"
    TEMPORAL_ORDERING_NOT_GROUNDED = "temporal_ordering_not_grounded"
    GLOBAL_RANKING_OUTSIDE_ANCHOR_SCOPE = "global_ranking_outside_anchor_scope"
    COUNT_LABEL_REQUIRES_COUNT_EVIDENCE = "count_label_requires_count_evidence"
    COUNT_LABEL_OUTSIDE_ANCHOR_SCOPE = "count_label_outside_anchor_scope"
    INITIAL_EXPLORATION_INSUFFICIENT = "initial_exploration_insufficient"
    INITIAL_LABEL_TOO_BROAD = "initial_label_too_broad"
    LABEL_SINGLE_TOOL_DERIVABLE = "label_single_tool_derivable"
    LABEL_REPEATS_ANCHOR_ENTITY = "label_repeats_anchor_entity"
    LABEL_BLANK_STRING_FORBIDDEN = "label_blank_string_forbidden"
    LABEL_IDENTIFIER_CHAIN_FORBIDDEN = "label_identifier_chain_forbidden"
    LABEL_OPAQUE_IDENTIFIER_FORBIDDEN = "label_opaque_identifier_forbidden"
    LABEL_VALUES_NOT_GROUNDED = "label_values_not_grounded"
    SELECTED_TOPIC_MISALIGNED = "selected_topic_misaligned"
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
        SubmitDraftErrorCode.QUESTION_SELF_PERSPECTIVE_REQUIRED,
        SubmitDraftErrorCode.SELF_ENTITY_ANCHOR_REQUIRED,
        SubmitDraftErrorCode.ANCHOR_ENTITY_CHANGED,
        SubmitDraftErrorCode.TEMPORAL_ORDERING_NOT_GROUNDED,
        SubmitDraftErrorCode.GLOBAL_RANKING_OUTSIDE_ANCHOR_SCOPE,
        SubmitDraftErrorCode.COUNT_LABEL_REQUIRES_COUNT_EVIDENCE,
        SubmitDraftErrorCode.COUNT_LABEL_OUTSIDE_ANCHOR_SCOPE,
        SubmitDraftErrorCode.INITIAL_EXPLORATION_INSUFFICIENT,
        SubmitDraftErrorCode.INITIAL_LABEL_TOO_BROAD,
        SubmitDraftErrorCode.LABEL_SINGLE_TOOL_DERIVABLE,
        SubmitDraftErrorCode.LABEL_REPEATS_ANCHOR_ENTITY,
        SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN,
        SubmitDraftErrorCode.LABEL_IDENTIFIER_CHAIN_FORBIDDEN,
        SubmitDraftErrorCode.LABEL_OPAQUE_IDENTIFIER_FORBIDDEN,
        SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED,
        SubmitDraftErrorCode.SELECTED_TOPIC_MISALIGNED,
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
            "<entity> ... </entity> on its own lines, followed by a blank line and the natural user request body. "
            "The entity block must exactly match anchor_entity."
        ),
    )
    constraint_summary: list["SubmitConstraintSummaryItem"] = Field(
        min_length=1,
        description="List of grounded hard constraints or tie-break rules expressed in plain language.",
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
            parsed = json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise ValueError("canonical_answer_json must be a valid JSON string") from exc
        return _ParsedCanonicalAnswerJson(normalized, parsed=parsed)

    @field_validator("topic", "question", "label_summary")
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


def _placeholder_tokens(payload: object) -> list[str]:
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    return sorted({token for token in _FORBIDDEN_PLACEHOLDER_TOKENS if token in serialized})


def _constraint_summary_payload(
    items: list[SubmitConstraintSummaryItem | dict[str, object]],
) -> list[dict[str, object]]:
    return [SubmitConstraintSummaryItem.model_validate(item).model_dump(mode="json") for item in items]


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
            "label_summary": None,
            "constraint_summary": [],
        }

    if isinstance(payload, SubmitDraftPayload):
        canonical_answer = payload.canonical_answer
        label_summary = payload.label_summary
        constraint_summary = _constraint_summary_payload(payload.constraint_summary)
    else:
        canonical_answer = None
        raw_canonical = payload.get("canonical_answer_json")
        label_summary = payload.get("label_summary")
        raw_constraints = payload.get("constraint_summary")
        constraint_summary = (
            _preview_runtime_payload(raw_constraints, config=config)
            if isinstance(raw_constraints, list)
            else []
        )
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
        "label_summary": label_summary,
        "constraint_summary": constraint_summary,
    }


def _label_change_summary(
    *,
    previous: dict[str, object] | None,
    current: dict[str, object],
) -> dict[str, object]:
    previous_field_names = list(previous.get("canonical_answer_field_names", [])) if previous else []
    current_field_names = list(current.get("canonical_answer_field_names", []))
    previous_constraint_summary = (
        list(previous.get("constraint_summary", [])) if previous else []
    )
    current_constraint_summary = list(current.get("constraint_summary", []))
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
        "previous_constraint_count": (
            len(previous_constraint_summary) if previous is not None else None
        ),
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
        "constraint_count_delta": (
            len(current_constraint_summary) - len(previous_constraint_summary)
            if previous is not None
            else None
        ),
        "label_summary_changed": (
            previous.get("label_summary") != current.get("label_summary")
            if previous is not None
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


_UUID_VALUE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HEX_VALUE_RE = re.compile(r"^[0-9a-f]+$", re.IGNORECASE)
_OPAQUE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _looks_like_opaque_identifier_string(
    value: str,
    *,
    config: AppConfig,
) -> bool:
    normalized = value.strip()
    if not normalized or any(character.isspace() for character in normalized):
        return False
    lowered = normalized.lower()
    if _UUID_VALUE_RE.fullmatch(lowered):
        return True
    if (
        len(lowered) >= config.synthesis.runtime.opaque_identifier_hex_min_length
        and _HEX_VALUE_RE.fullmatch(lowered)
    ):
        return True
    if (
        len(normalized) >= config.synthesis.runtime.opaque_identifier_token_min_length
        and _OPAQUE_TOKEN_RE.fullmatch(normalized)
        and any(character.isalpha() for character in normalized)
        and any(character.isdigit() for character in normalized)
    ):
        return True
    return False


def _opaque_identifier_value_paths(
    value: object,
    *,
    config: AppConfig,
    path: str = "$",
) -> list[str]:
    opaque_paths: list[str] = []
    if isinstance(value, str):
        return [path] if _looks_like_opaque_identifier_string(value, config=config) else []
    if isinstance(value, dict):
        for key, item in value.items():
            opaque_paths.extend(_opaque_identifier_value_paths(item, config=config, path=f"{path}.{key}"))
        return opaque_paths
    if isinstance(value, list):
        for index, item in enumerate(value):
            opaque_paths.extend(_opaque_identifier_value_paths(item, config=config, path=f"{path}[{index}]"))
        return opaque_paths
    return opaque_paths


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


def _uses_temporal_ordering_language(constraint_summary: list[SubmitConstraintSummaryItem]) -> bool:
    normalized_parts: list[str] = []
    for item in constraint_summary:
        if isinstance(item, SubmitConstraintSummaryItem):
            key = item.key
            kind = item.kind
            summary = item.summary
        elif isinstance(item, dict):
            key = str(item.get("key", ""))
            kind = str(item.get("kind", ""))
            summary = str(item.get("summary", ""))
        else:
            continue
        normalized_parts.extend(
            [
                normalize_words(key, lowercase=True),
                normalize_words(kind, lowercase=True),
                normalize_words(summary, lowercase=True),
            ]
        )
    combined = " ".join(part for part in normalized_parts if part)
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


def _mentions_global_scope(
    question_body: str | None,
    constraint_summary: list[SubmitConstraintSummaryItem],
) -> bool:
    text_parts: list[str] = []
    if question_body:
        text_parts.append(normalize_words(question_body, lowercase=True))
    for item in constraint_summary:
        if isinstance(item, SubmitConstraintSummaryItem):
            text_parts.extend(
                [
                    normalize_words(item.key, lowercase=True),
                    normalize_words(item.kind, lowercase=True),
                    normalize_words(item.summary, lowercase=True),
                ]
            )
        elif isinstance(item, dict):
            text_parts.extend(
                [
                    normalize_words(str(item.get("key", "")), lowercase=True),
                    normalize_words(str(item.get("kind", "")), lowercase=True),
                    normalize_words(str(item.get("summary", "")), lowercase=True),
                ]
            )
    combined = " ".join(part for part in text_parts if part)
    return any(token in combined for token in _GLOBAL_SCOPE_TOKENS)


def _count_semantics_present(
    answer: object,
    constraint_summary: list[SubmitConstraintSummaryItem],
) -> bool:
    if isinstance(answer, dict):
        count_keys = [key for key in answer if isinstance(key, str) and "count" in key.lower()]
        if count_keys:
            return True
        return any(_count_semantics_present(item, constraint_summary) for item in answer.values())
    if isinstance(answer, list):
        return any(_count_semantics_present(item, constraint_summary) for item in answer)
    if not isinstance(answer, (int, float)):
        return False
    text_parts: list[str] = []
    for item in constraint_summary:
        if isinstance(item, SubmitConstraintSummaryItem):
            text_parts.extend([item.key, item.kind, item.summary])
        elif isinstance(item, dict):
            text_parts.extend(
                [
                    str(item.get("key", "")),
                    str(item.get("kind", "")),
                    str(item.get("summary", "")),
                ]
            )
    combined_tokens: set[str] = set()
    for part in text_parts:
        if not part:
            continue
        combined_tokens.update(normalize_words(part, lowercase=True).split())
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
    observed_tokens: set[str] = set()
    for record in tool_calls:
        if not _tool_call_depends_on_anchor_entity(record, anchor_entity=anchor_entity):
            continue
        _collect_observed_string_tokens(
            record.get("result"),
            strings=observed_strings,
            tokens=observed_tokens,
        )
    return bool(observed_strings)


def _answer_has_multi_item_collection(value: object) -> bool:
    if isinstance(value, list):
        if len(value) > 1:
            return True
        return any(_answer_has_multi_item_collection(item) for item in value)
    if isinstance(value, dict):
        return any(_answer_has_multi_item_collection(item) for item in value.values())
    return False


def _axis_to_relax_for_too_hard(
    *,
    answer: object,
    constraint_count: int,
    difficulty_vector: DifficultyVectorContract,
    constraint_density_relax_threshold: int,
) -> DifficultyAxis:
    if _answer_has_multi_item_collection(answer):
        return DifficultyAxis.SOLUTION_SPACE
    if constraint_count > constraint_density_relax_threshold:
        return DifficultyAxis.CONSTRAINT_DENSITY
    ranked_axes = sorted(
        flatten_difficulty_vector(difficulty_vector).items(),
        key=lambda item: (
            item[1],
            2 if item[0] is DifficultyAxis.SEARCH_COST else 1 if item[0] is DifficultyAxis.SOLUTION_SPACE else 0,
        ),
        reverse=True,
    )
    return ranked_axes[0][0]


_ENTITY_PROMPT_RE = re.compile(
    r"\A<entity>\n(?P<entity_json>.+?)\n</entity>\n\n(?P<body>.+)\Z",
    re.DOTALL,
)

_PERSON_LIKE_ANCHOR_ALIASES: dict[str, tuple[str, ...]] = {
    "customer": ("customer", "고객"),
    "user": ("user", "사용자", "유저"),
    "member": ("member", "회원"),
    "patient": ("patient", "환자"),
    "guest": ("guest", "게스트", "투숙객"),
    "client": ("client", "고객", "클라이언트"),
    "subscriber": ("subscriber", "구독자", "가입자"),
    "rider": ("rider", "라이더", "탑승객"),
    "driver": ("driver", "기사", "운전자", "드라이버"),
    "student": ("student", "학생"),
    "teacher": ("teacher", "교사", "선생님"),
    "employee": ("employee", "직원"),
    "staff": ("staff", "직원", "스태프"),
    "agent": ("agent", "상담원", "에이전트"),
    "buyer": ("buyer", "구매자"),
    "seller": ("seller", "판매자"),
    "owner": ("owner", "소유자"),
    "passenger": ("passenger", "승객"),
    "traveler": ("traveler", "여행자"),
    "account_holder": ("account holder", "account_holder", "계정 소유자"),
}


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


def _label_summary_matches_selected_topic(
    *,
    selected_topic: str,
    label_summary: str,
    min_token_length: int,
) -> bool:
    selected_topic_tokens = topic_tokens(
        selected_topic,
        min_token_length=min_token_length,
    )
    if not selected_topic_tokens:
        return True
    normalized_summary = normalize_words(label_summary, lowercase=True)
    return all(token in normalized_summary for token in selected_topic_tokens)


def _person_like_anchor_aliases(anchor_entity: dict[str, object]) -> tuple[str, ...]:
    aliases: list[str] = []
    for raw_key in anchor_entity:
        normalized_key = normalize_words(str(raw_key), lowercase=True)
        key_tokens = tuple(token for token in normalized_key.split() if token)
        for token, token_aliases in _PERSON_LIKE_ANCHOR_ALIASES.items():
            if token in key_tokens:
                aliases.extend(token_aliases)
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _anchor_entity_is_person_like(anchor_entity: dict[str, object]) -> bool:
    return any(is_person_like_identifier(str(raw_key)) for raw_key in anchor_entity)


def _question_uses_non_self_perspective(
    question: str,
    *,
    anchor_entity: dict[str, object],
) -> bool:
    aliases = _person_like_anchor_aliases(anchor_entity)
    if not aliases:
        return False
    lowered = question.lower()
    for alias in aliases:
        alias_pattern = re.escape(alias.lower())
        english_patterns = (
            rf"\b(?:this|that|the)\s+{alias_pattern}\b",
            rf"\b{alias_pattern}'s\b",
            rf"\bfor\s+(?:this|that|the)\s+{alias_pattern}\b",
        )
        korean_patterns = (
            rf"(?:이|그|저|해당)\s*{alias_pattern}",
            rf"{alias_pattern}(?:의|에게|한테|을|를|은|는|이|가)\b",
        )
        if any(re.search(pattern, lowered) for pattern in (*english_patterns, *korean_patterns)):
            return True
    return False


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
    self_anchor_surface_names: tuple[str, ...] = ()
    accepted_draft: SynthesisTaskDraft | None = None
    attempts: list[SubmitDraftAttemptRecord] = field(default_factory=list)
    strongest_difficulty_vector: DifficultyVectorContract = field(
        default_factory=DifficultyVectorContract
    )
    difficulty_crank_history: list[DifficultyAxis] = field(default_factory=list)
    required_axis: DifficultyAxis | None = None
    required_axis_mode: DifficultyAdjustmentMode | None = None
    required_axis_reference_value: float | None = None
    last_quality_gate_status: str | None = None
    last_quality_gate_pass_rate: float | None = None
    _atomic_tool_calls: list[dict[str, object]] = field(default_factory=list, init=False)
    _raw_atomic_tool_calls: list[dict[str, object]] = field(default_factory=list, init=False)
    _tool_call_count_at_last_submission: int = field(default=0, init=False)
    _last_label_signature: str | None = field(default=None, init=False)
    _last_label_slot_count: int | None = field(default=None, init=False)
    _last_constraint_count: int | None = field(default=None, init=False)
    _last_monitored_label_data: dict[str, object] | None = field(default=None, init=False)
    _feedback_events: int = field(default=0, init=False)
    _last_primary_error_code: SubmitDraftErrorCode | None = field(default=None, init=False)
    _consecutive_primary_error_count: int = field(default=0, init=False)
    _locked_anchor_entity: dict[str, object] | None = field(default=None, init=False)

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

    def reject_invalid_payload(self, *, parsed: dict[str, object], error: ValidationError) -> str:
        if self.accepted_draft is not None:
            return "Accepted. Draft already stored."
        if self.submissions_left() <= 0:
            return "Budget exhausted. No more attempts."

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
            elif location == ("constraint_summary",):
                error_codes.append(SubmitDraftErrorCode.CONSTRAINT_SUMMARY_REQUIRED)
            elif location == ("anchor_entity",):
                if error_type == "value_error":
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
            return "Accepted. Draft already stored."
        if self.submissions_left() <= 0:
            return "Budget exhausted. No more attempts."

        submission_index = len(self.attempts) + 1
        error_codes: list[SubmitDraftErrorCode] = []
        invalid_diagnostics: dict[str, object] = {}
        search_cost_observations = (
            len(self._raw_atomic_tool_calls) - self._tool_call_count_at_last_submission
        )

        if len(self._raw_atomic_tool_calls) <= self._tool_call_count_at_last_submission:
            error_codes.append(SubmitDraftErrorCode.NO_NEW_GROUNDED_OBSERVATION)
        if not payload.anchor_entity:
            error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED)
        elif self.self_anchor_surface_names and not _anchor_entity_is_person_like(payload.anchor_entity):
            error_codes.append(SubmitDraftErrorCode.SELF_ENTITY_ANCHOR_REQUIRED)
            invalid_diagnostics["available_self_anchor_surfaces"] = list(
                self.self_anchor_surface_names[
                    : self.config.synthesis.runtime.diagnostic_item_limit
                ]
            )
        elif self._locked_anchor_entity is not None and payload.anchor_entity != self._locked_anchor_entity:
            error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_CHANGED)
            invalid_diagnostics["locked_anchor_entity"] = self._locked_anchor_entity
        elif payload.anchor_entity and (
            not self.self_anchor_surface_names or _anchor_entity_is_person_like(payload.anchor_entity)
        ):
            self._locked_anchor_entity = dict(payload.anchor_entity)
        if (
            _uses_temporal_ordering_language(payload.constraint_summary)
            and not _observed_temporal_surface(self._raw_atomic_tool_calls)
        ):
            error_codes.append(SubmitDraftErrorCode.TEMPORAL_ORDERING_NOT_GROUNDED)
        if not self.attempts:
            atomic_observations = len(self._raw_atomic_tool_calls)
            distinct_tool_count = _distinct_tool_name_count(self._raw_atomic_tool_calls)
            anchor_scoped_observations = _anchor_scoped_tool_call_count(
                self._raw_atomic_tool_calls,
                anchor_entity=payload.anchor_entity,
            )
            if (
                atomic_observations
                < self.config.synthesis.runtime.initial_submit_min_atomic_observations
                or distinct_tool_count
                < self.config.synthesis.runtime.initial_submit_min_distinct_tools
                or anchor_scoped_observations
                < self.config.synthesis.runtime.initial_submit_min_anchor_scoped_observations
            ):
                error_codes.append(SubmitDraftErrorCode.INITIAL_EXPLORATION_INSUFFICIENT)
                invalid_diagnostics["atomic_observations"] = atomic_observations
                invalid_diagnostics["distinct_tool_count"] = distinct_tool_count
                invalid_diagnostics["anchor_scoped_observations"] = anchor_scoped_observations
        placeholder_tokens = _placeholder_tokens(
            {
                "topic": payload.topic,
                "canonical_answer_json": payload.canonical_answer_json,
                "anchor_entity": payload.anchor_entity,
                "question": payload.question,
                "label_summary": payload.label_summary,
                "constraint_summary": _constraint_summary_payload(payload.constraint_summary),
            }
        )
        if placeholder_tokens:
            error_codes.append(SubmitDraftErrorCode.PLACEHOLDER_TOKENS_NOT_ALLOWED)
        canonical_answer = payload.canonical_answer
        prompt_anchor_entity, question_body, prompt_error = _split_entity_wrapped_prompt(
            payload.question
        )
        if prompt_error is not None:
            error_codes.append(prompt_error)
        elif prompt_anchor_entity != payload.anchor_entity:
            error_codes.append(SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_MISMATCH)
        if (
            _anchor_entity_is_person_like(payload.anchor_entity)
            and _uses_unanchored_global_ranking(
                self._raw_atomic_tool_calls,
                anchor_entity=payload.anchor_entity,
            )
            and not _mentions_global_scope(question_body, payload.constraint_summary)
        ):
            error_codes.append(SubmitDraftErrorCode.GLOBAL_RANKING_OUTSIDE_ANCHOR_SCOPE)
        constraint_count = len(payload.constraint_summary)
        label_signature = canonical_json(canonical_answer, default=str)
        label_slot_count = _answer_slot_count(canonical_answer)
        blank_paths = _blank_string_paths(canonical_answer)
        if blank_paths:
            error_codes.append(SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN)
            invalid_diagnostics["blank_string_paths"] = blank_paths[
                : self.config.synthesis.runtime.diagnostic_item_limit
            ]
        opaque_identifier_paths = _opaque_identifier_value_paths(
            canonical_answer,
            config=self.config,
        )
        if opaque_identifier_paths:
            error_codes.append(SubmitDraftErrorCode.LABEL_OPAQUE_IDENTIFIER_FORBIDDEN)
            invalid_diagnostics["opaque_identifier_paths"] = opaque_identifier_paths[
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
            self._raw_atomic_tool_calls,
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
            if (
                not has_raw_identifier_leak
                and not has_anchor_entity_leak
                and _question_uses_non_self_perspective(
                    question_body,
                    anchor_entity=payload.anchor_entity,
                )
            ):
                error_codes.append(SubmitDraftErrorCode.QUESTION_SELF_PERSPECTIVE_REQUIRED)
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
        if (
            _count_semantics_present(canonical_answer, payload.constraint_summary)
            and not _observed_count_evidence(self._raw_atomic_tool_calls)
        ):
            error_codes.append(SubmitDraftErrorCode.COUNT_LABEL_REQUIRES_COUNT_EVIDENCE)
        if (
            question_body is not None
            and _count_semantics_present(canonical_answer, payload.constraint_summary)
            and _observed_count_evidence(self._raw_atomic_tool_calls)
            and not _mentions_global_scope(question_body, payload.constraint_summary)
            and not _observed_anchor_scoped_count_evidence(
                self._raw_atomic_tool_calls,
                anchor_entity=payload.anchor_entity,
            )
        ):
            error_codes.append(SubmitDraftErrorCode.COUNT_LABEL_OUTSIDE_ANCHOR_SCOPE)
        if not _label_summary_matches_selected_topic(
            selected_topic=payload.topic,
            label_summary=payload.label_summary,
            min_token_length=self.config.synthesis.runtime.selected_topic_min_token_length,
        ):
            error_codes.append(SubmitDraftErrorCode.SELECTED_TOPIC_MISALIGNED)

        if (
            not self.attempts
            and self.required_axis is None
            and _answer_has_multi_item_collection(canonical_answer)
        ):
            error_codes.append(SubmitDraftErrorCode.INITIAL_LABEL_TOO_BROAD)

        weakened_axes = _weakened_difficulty_axes(
            previous=self.strongest_difficulty_vector,
            current=payload.difficulty_vector,
        )
        allowed_weaken_axis = (
            self.required_axis.value
            if self.required_axis is not None
            and self.required_axis_mode is DifficultyAdjustmentMode.RELAX
            else None
        )
        disallowed_weakened_axes = [
            axis for axis in weakened_axes if axis != allowed_weaken_axis
        ]
        if disallowed_weakened_axes:
            error_codes.append(SubmitDraftErrorCode.DIFFICULTY_WEAKENED)
        if (
            self.required_axis is not None
            and self.required_axis_mode is DifficultyAdjustmentMode.STRENGTHEN
        ):
            current_axis_value = getattr(payload.difficulty_vector, self.required_axis.value)
            reference_axis_value = (
                self.required_axis_reference_value
                if self.required_axis_reference_value is not None
                else getattr(self.strongest_difficulty_vector, self.required_axis.value)
            )
            if current_axis_value <= reference_axis_value:
                error_codes.append(SubmitDraftErrorCode.REQUIRED_DIFFICULTY_AXIS_NOT_STRENGTHENED)
        if (
            self.required_axis is not None
            and self.required_axis_mode is DifficultyAdjustmentMode.RELAX
        ):
            current_axis_value = getattr(payload.difficulty_vector, self.required_axis.value)
            reference_axis_value = (
                self.required_axis_reference_value
                if self.required_axis_reference_value is not None
                else getattr(self.strongest_difficulty_vector, self.required_axis.value)
            )
            if current_axis_value >= reference_axis_value:
                error_codes.append(SubmitDraftErrorCode.REQUIRED_DIFFICULTY_AXIS_NOT_RELAXED)
        if (
            self.required_axis is not None
            and self.required_axis_mode is DifficultyAdjustmentMode.STRENGTHEN
            and self._last_label_signature is not None
            and label_signature == self._last_label_signature
        ):
            error_codes.append(SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED)
        if (
            self.required_axis is DifficultyAxis.SOLUTION_SPACE
            and self.required_axis_mode is DifficultyAdjustmentMode.STRENGTHEN
            and self._last_label_slot_count is not None
            and label_slot_count <= self._last_label_slot_count
        ):
            error_codes.append(SubmitDraftErrorCode.REQUIRED_LABEL_AXIS_NOT_STRENGTHENED)
        if (
            self.required_axis is DifficultyAxis.CONSTRAINT_DENSITY
            and self.required_axis_mode is DifficultyAdjustmentMode.STRENGTHEN
            and self._last_constraint_count is not None
            and constraint_count <= self._last_constraint_count
        ):
            error_codes.append(SubmitDraftErrorCode.REQUIRED_LABEL_AXIS_NOT_STRENGTHENED)

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
                message=f"Rejected. Invalid draft: {type(exc).__name__}: {exc}",
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
        self._last_constraint_count = constraint_count
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
            return (
                "Accepted. solver pass rate "
                f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}."
            )

        attempts_left_after = self.submissions_left() - 1
        if quality_gate_summary.status is TaskQualityGateStatus.REJECT_TOO_EASY:
            requested_axis = _next_difficulty_crank_axis(self.difficulty_crank_history)
            self.required_axis = requested_axis
            self.required_axis_mode = DifficultyAdjustmentMode.STRENGTHEN
            self.difficulty_crank_history.append(requested_axis)
            strongest_axis_value = getattr(self.strongest_difficulty_vector, requested_axis.value)
            self.required_axis_reference_value = strongest_axis_value
            return self._record_rejection(
                submission_index=submission_index,
                message=(
                    "Rejected. solver pass rate "
                    f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}. "
                    f"Crank {requested_axis.value}. {difficulty_axis_feedback(requested_axis)} "
                    "Keep the same anchored user need and preserve the other two axes at least as strong as before. "
                    f"Make at least one new atomic tool call, gather new grounded evidence, and strengthen only that axis above {strongest_axis_value:.1f} with the smallest grounded step you can justify before resubmitting. "
                    f"{max(0, attempts_left_after)} attempts left."
                ),
                error_codes=[SubmitDraftErrorCode.REJECT_TOO_EASY],
                pass_rate=quality_gate_summary.pass_rate,
                matched_solver_runs=quality_gate_summary.matched_solver_runs,
                total_solver_runs=quality_gate_summary.total_solver_runs,
                diagnostics={"requested_axis": requested_axis.value},
                payload=payload,
                search_cost_observations=search_cost_observations,
            )

        requested_axis = _axis_to_relax_for_too_hard(
            answer=canonical_answer,
            constraint_count=constraint_count,
            difficulty_vector=payload.difficulty_vector,
            constraint_density_relax_threshold=(
                self.config.synthesis.runtime.constraint_density_relax_threshold
            ),
        )
        self.required_axis = requested_axis
        self.required_axis_mode = DifficultyAdjustmentMode.RELAX
        self.required_axis_reference_value = getattr(payload.difficulty_vector, requested_axis.value)
        return self._record_rejection(
            submission_index=submission_index,
            message=(
                "Rejected. solver pass rate "
                f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}. "
                "This draft is too hard for the configured band. "
                f"Reduce only {requested_axis.value} by one grounded step while keeping the same anchored user need and preserving the other two axes. "
                "Prefer the smallest simplification that keeps the task useful, such as shrinking a multi-item set to one item, removing one tie-breaker, or shortening one evidence hop before changing topic or anchor. "
                f"{max(0, attempts_left_after)} attempts left."
            ),
            error_codes=[SubmitDraftErrorCode.REJECT_TOO_HARD],
            pass_rate=quality_gate_summary.pass_rate,
            matched_solver_runs=quality_gate_summary.matched_solver_runs,
            total_solver_runs=quality_gate_summary.total_solver_runs,
            diagnostics={
                "terminal_rejection": False,
                "requested_axis": requested_axis.value,
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
            SubmitDraftErrorCode.CONSTRAINT_SUMMARY_REQUIRED: (
                "Rejected. constraint_summary must include at least one grounded constraint or tie-break rule."
            ),
            SubmitDraftErrorCode.LABEL_SUMMARY_REQUIRED: "Rejected. label_summary is required.",
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
            SubmitDraftErrorCode.QUESTION_SELF_PERSPECTIVE_REQUIRED: (
                "Rejected. Treat the anchor entity as the requesting user. Rewrite the request from the user's own perspective, for example 'my recent payments' rather than 'this customer's payments'."
            ),
            SubmitDraftErrorCode.SELF_ENTITY_ANCHOR_REQUIRED: (
                "Rejected. A person-like self entity surface is available in the observed tool set. Anchor the task on the requesting user's own record instead of anchoring it on a content object, and keep the content object as related evidence or answer content."
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
            SubmitDraftErrorCode.INITIAL_EXPLORATION_INSUFFICIENT: (
                "Rejected. You tried to submit the first judged draft before you had fully understood the anchored user and the evidence path. Keep researching the same anchored user before drafting: gather more atomic observations, use more distinct tools, and inspect more anchor-scoped evidence paths before the first submit_draft call."
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
            SubmitDraftErrorCode.LABEL_OPAQUE_IDENTIFIER_FORBIDDEN: (
                "Rejected. The canonical answer contains opaque identifier values such as UUIDs, hashes, encrypted tokens, or other random-looking reference strings. Even if those values were observed, they are not user-facing business labels. Return readable business values, dates, amounts, counts, statuses, or ordered records instead."
            ),
            SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED: (
                "Rejected. Some label values were not directly grounded in the observed tool results. Schema orientation alone is not enough; only use business strings you actually observed in real tool outputs. Do not manufacture readable labels by wrapping an id in generic words such as 'staff member 2' or 'order 17'. If the chosen surface is id-only, keep the same anchored user and switch to counts, dates, amounts, statuses, ordering, make new anchored tool calls until you observe readable fields, or choose a better grounded topic for the same anchored user need."
            ),
            SubmitDraftErrorCode.SELECTED_TOPIC_MISALIGNED: (
                "Rejected. label_summary must explicitly name the selected topic and keep the draft semantically centered on that topic."
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
        if error_codes and error_codes[0] is SubmitDraftErrorCode.LABEL_IDENTIFIER_CHAIN_FORBIDDEN:
            if diagnostics is not None and diagnostics.get("anchor_path_has_readable_strings") is False:
                primary = (
                    "Rejected. The current anchored evidence path is still id-only. Do not submit another answer made only of *_id fields on this same path. "
                    "Keep the same anchored user and either answer with grounded counts, dates, amounts, statuses, or ordering, or pivot to a better grounded topic for that same user need."
                )
        if feedback_only and primary.startswith("Rejected. "):
            primary = primary.replace("Rejected. ", "Feedback. ", 1)
        if error_codes and error_codes[0] is SubmitDraftErrorCode.INITIAL_EXPLORATION_INSUFFICIENT:
            primary = (
                "Feedback. Do not submit yet. Go back to research mode first. Before the first judged draft, map the database relationships around the anchored user and keep exploring until you fully understand the nearby evidence paths and have enough grounded context: "
                f"at least {self.config.synthesis.runtime.initial_submit_min_atomic_observations} atomic observations, "
                f"at least {self.config.synthesis.runtime.initial_submit_min_distinct_tools} distinct tool names, and "
                f"at least {self.config.synthesis.runtime.initial_submit_min_anchor_scoped_observations} anchor-scoped observations tied to anchor_entity. "
                "Use that research phase to classify nearby relationships and paths as readable, id-only, local-only, countable, aggregate-capable, or dead ends, and submit only after you understand why every answer slot is grounded and needed."
            )
        preserve_guidance = ""
        if self._last_monitored_label_data is not None:
            preserve_guidance = (
                " Keep the same anchored user need and fix only the failing part when possible. "
                "Do not reset to a different topic, a different anchor, or a simpler global count just to satisfy this feedback."
            )
        escalation_guidance = ""
        if (
            error_codes
            and error_codes[0] is self._last_primary_error_code
            and self._consecutive_primary_error_count
            >= self.config.synthesis.runtime.repeated_error_escalation_threshold
        ):
            escalation_guidance = (
                " You have repeated the same failure multiple times. Abandon this label family and pivot: "
                "make new atomic tool calls on a different anchored evidence path, or choose a different grounded topic for the same anchored user need. "
                "Do not resubmit another small variant of the same id-only, single-call, or ungrounded answer."
            )
        additional_messages: list[str] = []
        for error_code in error_codes[1:3]:
            extra = message_map.get(error_code)
            if extra is None:
                continue
            extra_text = extra.removeprefix("Rejected. ").strip()
            if extra.startswith("Feedback. "):
                extra_text = extra.removeprefix("Feedback. ").strip()
            additional_messages.append(extra_text)
        if feedback_only:
            attempts_left_after = self.submissions_left() - 1
            if additional_messages:
                return (
                    f"{primary}{preserve_guidance}{escalation_guidance} Also fix: {' '.join(additional_messages)} "
                    "Make another atomic tool call if needed, then call submit_draft again. Do not stop with plain text. "
                    f"{max(0, attempts_left_after)} attempts left."
                )
            return (
                f"{primary}{preserve_guidance}{escalation_guidance} Make another atomic tool call if needed, then call submit_draft again. "
                f"Do not stop with plain text. {max(0, attempts_left_after)} attempts left."
            )
        attempts_left_after = self.submissions_left() - 1
        if additional_messages:
            return (
                f"{primary}{preserve_guidance}{escalation_guidance} Also fix: {' '.join(additional_messages)} "
                f"{max(0, attempts_left_after)} attempts left."
            )
        return f"{primary}{preserve_guidance}{escalation_guidance} {max(0, attempts_left_after)} attempts left."

    def _record_feedback(
        self,
        *,
        message: str,
        error_codes: list[SubmitDraftErrorCode],
        payload: SubmitDraftPayload | dict[str, object] | None = None,
        search_cost_observations: int | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> str:
        self._update_primary_error_state(error_codes)
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
        if attempts_left_after <= 0 and "Budget exhausted. No more attempts." not in message:
            return f"{message} Budget exhausted. No more attempts."
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
        self._update_primary_error_state(error_codes)
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
            return f"{message} Budget exhausted. No more attempts."
        return message

    def _update_primary_error_state(
        self,
        error_codes: list[SubmitDraftErrorCode],
    ) -> None:
        if not error_codes:
            self._last_primary_error_code = None
            self._consecutive_primary_error_count = 0
            return
        primary = error_codes[0]
        if primary is self._last_primary_error_code:
            self._consecutive_primary_error_count += 1
        else:
            self._last_primary_error_code = primary
            self._consecutive_primary_error_count = 1

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
                    "constraint_density_constraints": (
                        len(label_data["constraint_summary"])
                        if isinstance(label_data["constraint_summary"], list)
                        else None
                    ),
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
                "required_axis_reference_value": self.required_axis_reference_value,
                "locked_anchor_entity": self._locked_anchor_entity,
                "difficulty_crank_history": [axis.value for axis in self.difficulty_crank_history],
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
        if "anchor_entity" not in parsed:
            raw_question = parsed.get("question")
            if isinstance(raw_question, str):
                parsed_anchor_entity, _, prompt_error = _split_entity_wrapped_prompt(raw_question)
                if prompt_error is None and parsed_anchor_entity is not None:
                    parsed["anchor_entity"] = parsed_anchor_entity
        try:
            payload = SubmitDraftPayload.model_validate(parsed)
        except ValidationError as exc:
            return controller.reject_invalid_payload(parsed=parsed, error=exc)
        return await controller.submit(payload)

    return FunctionTool(
        name="submit_draft",
        description=(
            "Submit a grounded RLVR task draft after inspecting real database rows with tools. "
            "Include the selected topic string, canonical answer JSON, anchor entity, declared difficulty vector, "
            "natural user-facing question, constraint summary, and label summary. "
            "Use only tool-observed evidence. Do not write SQL, do not invent hidden joins, and do not include SQL queries in the submission. "
            "Trace many relationships and interesting grounded paths with tools first, then choose one path and build a unique, verifiable label from that path. "
            "Do research and analysis first; do not call submit_draft while you are still figuring out the anchored user, the evidence path, or the label. "
            "Call submit_draft only when you fully understand the anchored user, the relevant evidence path, which observed fields are readable, which paths are id-only dead ends, and why every answer slot is needed. "
            "Choose topic from the grounded label and observed evidence, not by copying a planning hint. "
            "anchor_entity is mandatory and must be a flat JSON object mapping one or more primary-key field names to scalar values, for example {\"customer_id\": 123} or {\"order_id\": 7, \"line_no\": 2}. "
            "Do not call submit_draft until anchor_entity is present and final for that draft. After the first valid self anchor is established, keep that same anchor_entity across retries. "
            "question must already be the full user-facing prompt in this exact shape: <entity> newline JSON newline </entity> blank line user request. "
            "The JSON inside the <entity> block must exactly match anchor_entity. "
            "label_summary must be English, must explicitly include the selected topic phrase, and must explain why the label is grounded and unique. "
            "Do not submit blank or placeholder string fields in the canonical answer; every answer field must contain a grounded, non-empty value. "
            "Do not submit opaque identifier values such as UUIDs, hashes, encrypted tokens, or random-looking reference strings as answer labels, even if they were observed. "
            "Do not submit labels that can be read from a single atomic tool call or a direct projection of a single tool result. "
            "Do not submit questions or labels that are only chains of internal *_id fields. Prefer business-facing values. "
            "After any rejection, make at least one new atomic tool call before calling submit_draft again. "
            "A rejection means keep going in the same conversation, not stop. "
            "All submit_draft arguments are schema-validated; missing required fields and invalid canonical_answer_json values are rejected automatically."
        ),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=False,
    )
