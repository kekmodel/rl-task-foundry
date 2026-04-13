"""submit_draft tool for the single-agent synthesis loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
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
    normalize_words,
    topic_tokens,
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
    INSTANCE_SPACE_REQUIRED = "instance_space_required"
    LABEL_SUMMARY_REQUIRED = "label_summary_required"
    PLACEHOLDER_TOKENS_NOT_ALLOWED = "placeholder_tokens_not_allowed"
    QUESTION_INTERNAL_SCHEMA_LEAK = "question_internal_schema_leak"
    QUESTION_RAW_IDENTIFIER_LEAK = "question_raw_identifier_leak"
    QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN = "question_entity_placeholder_forbidden"
    QUESTION_ANCHOR_ENTITY_LEAK = "question_anchor_entity_leak"
    LABEL_SINGLE_TOOL_DERIVABLE = "label_single_tool_derivable"
    LABEL_REPEATS_ANCHOR_ENTITY = "label_repeats_anchor_entity"
    LABEL_BLANK_STRING_FORBIDDEN = "label_blank_string_forbidden"
    LABEL_IDENTIFIER_CHAIN_FORBIDDEN = "label_identifier_chain_forbidden"
    LABEL_VALUES_NOT_GROUNDED = "label_values_not_grounded"
    SELECTED_TOPIC_MISALIGNED = "selected_topic_misaligned"
    DIFFICULTY_WEAKENED = "difficulty_weakened"
    REQUIRED_DIFFICULTY_AXIS_NOT_STRENGTHENED = "required_difficulty_axis_not_strengthened"
    REQUIRED_LABEL_AXIS_NOT_STRENGTHENED = "required_label_axis_not_strengthened"
    LABEL_NOT_STRENGTHENED = "label_not_strengthened"
    SUBMIT_PAYLOAD_INVALID = "submit_payload_invalid"
    DRAFT_VALIDATION_FAILED = "draft_validation_failed"
    REJECT_TOO_EASY = "reject_too_easy"
    REJECT_TOO_HARD = "reject_too_hard"


_FEEDBACK_ONLY_ERROR_CODES = frozenset(
    {
        SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED,
        SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED,
        SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_REQUIRED,
        SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_INVALID_JSON,
        SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_MISMATCH,
        SubmitDraftErrorCode.QUESTION_BODY_REQUIRED,
    }
)


class _ParsedCanonicalAnswerJson(str):
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


def _difficulty_axis_hint(axis: DifficultyAxis) -> str:
    if axis is DifficultyAxis.SEARCH_COST:
        return (
            "Strengthen the label through search_cost. Change the label so it depends on a longer grounded evidence path. "
            "A good next step is to add one more linked entity, require one more lookup before the label is fixed, "
            "or combine facts from a deeper chain instead of reading one obvious record. "
            "Do not spend the extra hop on echoing the anchor id or on whichever related row happened to appear first in exploration results. "
            "If you need one related row among many, define a grounded ordering or tie-breaker that you can explain naturally to the user. "
            "Prefer a local ordering inside the anchored scope before jumping to a global ranking over the whole database."
        )
    if axis is DifficultyAxis.SOLUTION_SPACE:
        return (
            "Strengthen the label through solution_space. Change the label so it is larger or less immediately determined. "
            "A good next step is to return more answer fields, return an ordered set instead of one scalar, "
            "or choose among several grounded candidates with an explicit tie-breaker."
        )
    return (
        "Strengthen the label through constraint_density. Change the label by adding one more hard grounded rule. "
        "A good next step is to add a uniqueness rule, a stricter ordering rule, a tighter filter, "
        "or another grounded condition that removes valid answers."
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
    items: list[SubmitConstraintSummaryItem | dict[str, object]],
) -> list[dict[str, object]]:
    return [
        SubmitConstraintSummaryItem.model_validate(item).model_dump(mode="json")
        for item in items
    ]


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)


def _monitor_answer_snapshot(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {
            "root_type": "object",
            "slot_count": len(value),
            "field_names": list(value.keys())[:8],
            "preview": _preview_payload(value),
        }
    if isinstance(value, list):
        field_names: list[str] = []
        if value and isinstance(value[0], dict):
            field_names = [str(key) for key in list(value[0].keys())[:8]]
        return {
            "root_type": "array",
            "slot_count": len(value),
            "field_names": field_names,
            "preview": _preview_payload(value),
        }
    return {
        "root_type": type(value).__name__,
        "slot_count": 1,
        "field_names": [],
        "preview": _preview_payload(value),
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
            _preview_payload(raw_constraints)
            if isinstance(raw_constraints, list)
            else []
        )
        if isinstance(raw_canonical, str):
            try:
                canonical_answer = json.loads(raw_canonical)
            except json.JSONDecodeError:
                canonical_answer = raw_canonical

    snapshot = (
        _monitor_answer_snapshot(canonical_answer)
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
            _stable_json(canonical_answer) if canonical_answer is not None else None
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

    if isinstance(answer, dict):
        return _keys_are_identifier_only(answer)
    if isinstance(answer, list) and answer and all(isinstance(item, dict) for item in answer):
        return all(_keys_are_identifier_only(item) for item in answer)
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


def _is_identifier_field_name(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized.endswith("_id") or normalized.endswith("_ids")


def _contains_raw_identifier_token(text: str) -> bool:
    return _IDENTIFIER_FIELD_TOKEN_RE.search(text.lower()) is not None


def _contains_entity_placeholder_token(text: str) -> bool:
    lowered = text.lower()
    return "<entity>" in lowered or "&lt;entity&gt;" in lowered


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


def _label_summary_matches_selected_topic(*, selected_topic: str, label_summary: str) -> bool:
    selected_topic_tokens = topic_tokens(selected_topic)
    if not selected_topic_tokens:
        return True
    normalized_summary = normalize_words(label_summary, lowercase=True)
    return all(token in normalized_summary for token in selected_topic_tokens)


def _question_repeats_anchor_entity(
    question: str,
    *,
    anchor_entity: dict[str, object],
) -> bool:
    lowered = question.lower()
    for raw_key, raw_value in anchor_entity.items():
        key = str(raw_key).strip().lower()
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
        value = str(raw_value).strip().lower()
        if not value:
            continue
        value_pattern = re.escape(value)
        patterns = (
            rf"(?<![a-z0-9_]){key_pattern}\s*[:#-]?\s*{value_pattern}(?![a-z0-9_])",
            rf"(?<![a-z0-9_]){value_pattern}(?:번)?\s*[:#-]?\s*{key_pattern}(?![a-z0-9_])",
        )
        if any(re.search(pattern, lowered) for pattern in patterns):
            return True
    return False


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
    _last_label_signature: str | None = field(default=None, init=False)
    _last_label_slot_count: int | None = field(default=None, init=False)
    _last_constraint_count: int | None = field(default=None, init=False)
    _last_monitored_label_data: dict[str, object] | None = field(default=None, init=False)
    _feedback_events: int = field(default=0, init=False)

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

        if len(self._atomic_tool_calls) <= self._tool_call_count_at_last_submission:
            error_codes.append(SubmitDraftErrorCode.NO_NEW_GROUNDED_OBSERVATION)
        if not payload.anchor_entity:
            error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED)
        placeholder_tokens = _placeholder_tokens(
            {
                "topic": payload.topic,
                "canonical_answer_json": payload.canonical_answer_json,
                "anchor_entity": payload.anchor_entity,
                "question": payload.question,
                "label_summary": payload.label_summary,
                "constraint_summary": _constraint_summary_payload(payload.constraint_summary),
                "instance_space": payload.instance_space.model_dump(mode="json"),
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
        label_signature: str | None = None
        label_slot_count: int | None = None
        constraint_count = len(payload.constraint_summary)
        label_signature = _stable_json(canonical_answer)
        label_slot_count = _answer_slot_count(canonical_answer)
        blank_paths = _blank_string_paths(canonical_answer)
        if blank_paths:
            error_codes.append(SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN)
            invalid_diagnostics["blank_string_paths"] = blank_paths[:5]
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
            invalid_diagnostics["ungrounded_strings"] = ungrounded_strings[:5]
        if question_body is not None:
            question_lower = question_body.lower()
            if _contains_entity_placeholder_token(question_lower):
                error_codes.append(SubmitDraftErrorCode.QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN)
            if _contains_raw_identifier_token(question_lower):
                error_codes.append(SubmitDraftErrorCode.QUESTION_RAW_IDENTIFIER_LEAK)
            if _question_repeats_anchor_entity(
                question_body,
                anchor_entity=payload.anchor_entity,
            ):
                error_codes.append(SubmitDraftErrorCode.QUESTION_ANCHOR_ENTITY_LEAK)
            if any(token in question_lower for token in self.forbidden_question_tokens):
                error_codes.append(SubmitDraftErrorCode.QUESTION_INTERNAL_SCHEMA_LEAK)
        if _answer_uses_only_identifier_fields(canonical_answer):
            error_codes.append(SubmitDraftErrorCode.LABEL_IDENTIFIER_CHAIN_FORBIDDEN)
        if not _label_summary_matches_selected_topic(
            selected_topic=payload.topic,
            label_summary=payload.label_summary,
        ):
            error_codes.append(SubmitDraftErrorCode.SELECTED_TOPIC_MISALIGNED)

        weakened_axes = _weakened_difficulty_axes(
            previous=self.strongest_difficulty_vector,
            current=payload.difficulty_vector,
        )
        if weakened_axes:
            error_codes.append(SubmitDraftErrorCode.DIFFICULTY_WEAKENED)
        if self.required_axis is not None:
            current_axis_value = getattr(payload.difficulty_vector, self.required_axis.value)
            strongest_axis_value = getattr(
                self.strongest_difficulty_vector, self.required_axis.value
            )
            if current_axis_value <= strongest_axis_value:
                error_codes.append(SubmitDraftErrorCode.REQUIRED_DIFFICULTY_AXIS_NOT_STRENGTHENED)
        if (
            self.required_axis is not None
            and label_signature is not None
            and self._last_label_signature is not None
            and label_signature == self._last_label_signature
        ):
            error_codes.append(SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED)
        if (
            self.required_axis is DifficultyAxis.SOLUTION_SPACE
            and label_slot_count is not None
            and self._last_label_slot_count is not None
            and label_slot_count <= self._last_label_slot_count
        ):
            error_codes.append(SubmitDraftErrorCode.REQUIRED_LABEL_AXIS_NOT_STRENGTHENED)
        if (
            self.required_axis is DifficultyAxis.CONSTRAINT_DENSITY
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
        self._last_label_signature = label_signature
        self._last_label_slot_count = label_slot_count
        self._last_constraint_count = constraint_count
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
                search_cost_observations=search_cost_observations,
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
                error_codes=[SubmitDraftErrorCode.REJECT_TOO_EASY],
                pass_rate=quality_gate_summary.pass_rate,
                matched_solver_runs=quality_gate_summary.matched_solver_runs,
                total_solver_runs=quality_gate_summary.total_solver_runs,
                diagnostics={"requested_axis": requested_axis.value},
                payload=payload,
                search_cost_observations=search_cost_observations,
            )

        self.max_submissions = len(self.attempts) + 1
        return self._record_rejection(
            submission_index=submission_index,
            message=(
                "Rejected. solver pass rate "
                f"{quality_gate_summary.matched_solver_runs}/{quality_gate_summary.total_solver_runs}. "
                "This draft is too hard for the configured band."
            ),
            error_codes=[SubmitDraftErrorCode.REJECT_TOO_HARD],
            pass_rate=quality_gate_summary.pass_rate,
            matched_solver_runs=quality_gate_summary.matched_solver_runs,
            total_solver_runs=quality_gate_summary.total_solver_runs,
            diagnostics={"terminal_rejection": True},
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
            SubmitDraftErrorCode.INSTANCE_SPACE_REQUIRED: "Rejected. instance_space is required.",
            SubmitDraftErrorCode.LABEL_SUMMARY_REQUIRED: "Rejected. label_summary is required.",
            SubmitDraftErrorCode.CANONICAL_ANSWER_JSON_INVALID: (
                "Rejected. canonical_answer_json must be a valid JSON string."
            ),
            SubmitDraftErrorCode.PLACEHOLDER_TOKENS_NOT_ALLOWED: (
                "Rejected. Replace every placeholder token with grounded names and values from the current database."
            ),
            SubmitDraftErrorCode.QUESTION_INTERNAL_SCHEMA_LEAK: (
                "Rejected. Rewrite the user-facing question without raw table names, join-table names, or SQL keywords."
            ),
            SubmitDraftErrorCode.QUESTION_RAW_IDENTIFIER_LEAK: (
                "Rejected. Rewrite the user-facing question without raw identifier field names such as <entity>_id. Keep identifiers only inside anchor_entity."
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_PLACEHOLDER_FORBIDDEN: (
                "Rejected. Do not repeat the literal <entity> token inside the user-request body. Use it only once as the required XML entity block at the top."
            ),
            SubmitDraftErrorCode.QUESTION_ANCHOR_ENTITY_LEAK: (
                "Rejected. Rewrite the user-request body without repeating the raw anchor entity id. Keep the raw anchor only inside the entity block and refer to it naturally in the request."
            ),
            SubmitDraftErrorCode.LABEL_SINGLE_TOOL_DERIVABLE: (
                "Rejected. The canonical answer can be recovered from a single atomic tool call. Redesign the task so the label requires combining multiple observations. A one-hop foreign-key lookup that only returns identifiers is still too weak."
            ),
            SubmitDraftErrorCode.LABEL_REPEATS_ANCHOR_ENTITY: (
                "Rejected. Do not repeat anchor_entity fields inside the canonical answer. The entity block already provides that grounding, so use the answer slots for new grounded information."
            ),
            SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN: (
                "Rejected. The canonical answer contains blank string fields. Every answer field must contain a grounded, non-empty value. Schema orientation alone is not enough; only fields you actually observed in tool results are grounded. If the chosen surface is id-only, switch to counts, dates, amounts, statuses, ordering, or choose a different anchor with readable fields."
            ),
            SubmitDraftErrorCode.LABEL_IDENTIFIER_CHAIN_FORBIDDEN: (
                "Rejected. The canonical answer is only a chain of internal identifier fields. A relation made only of ids is still an internal identifier chain. Return user-relevant business values such as names, titles, dates, amounts, counts, or statuses instead."
            ),
            SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED: (
                "Rejected. Some label values were not directly grounded in the observed tool results. Schema orientation alone is not enough; only use business strings you actually observed in real tool outputs. If the chosen surface is id-only, switch to counts, dates, amounts, statuses, ordering, or choose a different anchor with readable fields."
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
        if feedback_only and primary.startswith("Rejected. "):
            primary = primary.replace("Rejected. ", "Feedback. ", 1)
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
            if additional_messages:
                return f"{primary} Also fix: {' '.join(additional_messages)} Submission budget unchanged."
            return f"{primary} Submission budget unchanged."
        attempts_left_after = self.submissions_left() - 1
        if additional_messages:
            return (
                f"{primary} Also fix: {' '.join(additional_messages)} "
                f"{max(0, attempts_left_after)} attempts left."
            )
        return f"{primary} {max(0, attempts_left_after)} attempts left."

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
        self._emit_monitor(
            status="feedback",
            payload=payload,
            pass_rate=None,
            matched_solver_runs=None,
            total_solver_runs=None,
            search_cost_observations=search_cost_observations,
            diagnostics={"error_codes": _error_code_values(error_codes), **(diagnostics or {})},
        )
        return message

    def _record_rejection(
        self,
        *,
        submission_index: int,
        message: str,
        error_codes: list[SubmitDraftErrorCode],
        payload: SubmitDraftPayload | None = None,
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
        search_cost_observations: int | None,
        diagnostics: dict[str, object],
    ) -> None:
        if self.phase_monitor is None:
            return
        label_data = _monitor_label_data(payload)
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
                "difficulty_crank_history": [axis.value for axis in self.difficulty_crank_history],
                "recent_tool_calls": self._atomic_tool_calls[-5:],
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
            "Submit a grounded RLVR task draft after inspecting real database rows. "
            "Include the selected topic string, canonical answer JSON, anchor entity, declared difficulty vector, "
            "natural user-facing question, constraint summary, instance space, and label summary. "
            "Choose topic from the grounded label and observed evidence, not by copying a planning hint. "
            "anchor_entity is mandatory and must be a flat JSON object mapping one or more primary-key field names to scalar values, for example {\"customer_id\": 123} or {\"order_id\": 7, \"line_no\": 2}. "
            "Do not call submit_draft until anchor_entity is present and final for that draft. "
            "question must already be the full user-facing prompt in this exact shape: <entity> newline JSON newline </entity> blank line user request. "
            "The JSON inside the <entity> block must exactly match anchor_entity. "
            "label_summary must be English, must explicitly include the selected topic phrase, and must explain why the label is grounded and unique. "
            "Do not submit blank or placeholder string fields in the canonical answer; every answer field must contain a grounded, non-empty value. "
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
