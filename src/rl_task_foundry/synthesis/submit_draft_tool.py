"""submit_draft tool for the single-agent synthesis loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import (
    AliasChoices,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.infra.sdk_helpers import preview_payload
from rl_task_foundry.infra.visibility import (
    blocks_direct_label_exposure,
    is_blocked_visibility,
)
from rl_task_foundry.synthesis.canonicalize import canonical_json
from rl_task_foundry.synthesis.contracts import StrictModel
from rl_task_foundry.synthesis.phase_monitor import PipelinePhaseMonitorLogger
from rl_task_foundry.synthesis.submit_draft_messages import (
    _format_missing_request_phrase_guidance,
    _format_ungrounded_value_guidance,
    _render_structured_message,
    _too_easy_retry_guidance,
)
from rl_task_foundry.synthesis.submit_draft_validators import (
    _blank_string_paths,
    _collect_observed_strings,
    _disconnected_answer_strings,
    _observed_anchor_readable_string_surface,
    _rebuild_anchor_connected_strings,
    _ungrounded_answer_strings,
)
from rl_task_foundry.synthesis.turn_budget import (
    FEEDBACK_REPAIR_MAX_DATA_TOOLS,
    FIRST_SUBMIT_MAX_DATA_TOOLS,
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
    LABEL_BLANK_STRING_FORBIDDEN = "label_blank_string_forbidden"
    LABEL_NULL_VALUE_FORBIDDEN = "label_null_value_forbidden"
    LABEL_VALUES_NOT_GROUNDED = "label_values_not_grounded"
    LABEL_NOT_STRENGTHENED = "label_not_strengthened"
    LABEL_CHANGED_DURING_REPAIR = "label_changed_during_repair"
    ANSWER_CONTRACT_REQUIRED = "answer_contract_required"
    ANSWER_CONTRACT_JSON_INVALID = "answer_contract_json_invalid"
    ANSWER_CONTRACT_PHRASE_MISSING = "answer_contract_phrase_missing"
    ANSWER_CONTRACT_EVIDENCE_MISSING = "answer_contract_evidence_missing"
    ANSWER_CONTRACT_EVIDENCE_MISMATCH = "answer_contract_evidence_mismatch"
    ANSWER_CONTRACT_QUERY_MISMATCH = "answer_contract_query_mismatch"
    ANSWER_CONTRACT_ORDER_AMBIGUOUS = "answer_contract_order_ambiguous"
    ANSWER_CONTRACT_ORDER_TOO_COMPLEX = "answer_contract_order_too_complex"
    ANSWER_CONTRACT_DUPLICATE_ANSWER_ROWS = (
        "answer_contract_duplicate_answer_rows"
    )
    ANSWER_CONTRACT_LIST_SIZE_INVALID = "answer_contract_list_size_invalid"
    ANSWER_CONTRACT_LIST_LIMIT_TOO_WIDE = "answer_contract_list_limit_too_wide"
    ANSWER_CONTRACT_FILTER_UNBOUND = "answer_contract_filter_unbound"
    ANSWER_CONTRACT_HIDDEN_FILTER_UNANCHORED = (
        "answer_contract_hidden_filter_unanchored"
    )
    ANSWER_CONTRACT_VISIBILITY_EVIDENCE_MISSING = (
        "answer_contract_visibility_evidence_missing"
    )
    ANSWER_CONTRACT_BINDING_MISSING = "answer_contract_binding_missing"
    LABEL_NON_USER_VISIBLE_SOURCE = "label_non_user_visible_source"
    LABEL_NO_PRIMARY_KEY_SOURCE = "label_no_primary_key_source"
    ANSWER_CONTRACT_NOT_INCREMENTAL = "answer_contract_not_incremental"
    SUBMIT_PAYLOAD_INVALID = "submit_payload_invalid"
    DRAFT_VALIDATION_FAILED = "draft_validation_failed"
    REJECT_TOO_EASY = "reject_too_easy"
    REJECT_TOO_HARD = "reject_too_hard"
    CALIBRATION_INCONCLUSIVE = "calibration_inconclusive"


_LABEL_LOCKING_REPAIR_ERROR_VALUES = frozenset(
    {
        SubmitDraftErrorCode.ANSWER_CONTRACT_BINDING_MISSING.value,
        SubmitDraftErrorCode.ANSWER_CONTRACT_PHRASE_MISSING.value,
        SubmitDraftErrorCode.ANSWER_CONTRACT_QUERY_MISMATCH.value,
    }
)


_FEEDBACK_ONLY_ERROR_CODES = frozenset(
    {
        SubmitDraftErrorCode.NO_NEW_GROUNDED_OBSERVATION,
        SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED,
        SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED,
        SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_REQUIRED,
        SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_INVALID_JSON,
        SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_MISMATCH,
        SubmitDraftErrorCode.QUESTION_BODY_REQUIRED,
        SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN,
        SubmitDraftErrorCode.LABEL_NULL_VALUE_FORBIDDEN,
        SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED,
        SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED,
        SubmitDraftErrorCode.LABEL_CHANGED_DURING_REPAIR,
        SubmitDraftErrorCode.ANSWER_CONTRACT_PHRASE_MISSING,
        SubmitDraftErrorCode.ANSWER_CONTRACT_JSON_INVALID,
        SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISSING,
        SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISMATCH,
        SubmitDraftErrorCode.ANSWER_CONTRACT_QUERY_MISMATCH,
        SubmitDraftErrorCode.ANSWER_CONTRACT_ORDER_AMBIGUOUS,
        SubmitDraftErrorCode.ANSWER_CONTRACT_ORDER_TOO_COMPLEX,
        SubmitDraftErrorCode.ANSWER_CONTRACT_DUPLICATE_ANSWER_ROWS,
        SubmitDraftErrorCode.ANSWER_CONTRACT_LIST_SIZE_INVALID,
        SubmitDraftErrorCode.ANSWER_CONTRACT_LIST_LIMIT_TOO_WIDE,
        SubmitDraftErrorCode.ANSWER_CONTRACT_FILTER_UNBOUND,
        SubmitDraftErrorCode.ANSWER_CONTRACT_HIDDEN_FILTER_UNANCHORED,
        SubmitDraftErrorCode.ANSWER_CONTRACT_VISIBILITY_EVIDENCE_MISSING,
        SubmitDraftErrorCode.ANSWER_CONTRACT_BINDING_MISSING,
        SubmitDraftErrorCode.LABEL_NON_USER_VISIBLE_SOURCE,
        SubmitDraftErrorCode.LABEL_NO_PRIMARY_KEY_SOURCE,
        SubmitDraftErrorCode.ANSWER_CONTRACT_NOT_INCREMENTAL,
    }
)

JsonScalar = str | int | float | bool


def _strip_required_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _strip_optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _error_code_values(
    codes: list[SubmitDraftErrorCode | str] | tuple[SubmitDraftErrorCode | str, ...],
) -> list[str]:
    return [
        code.value if isinstance(code, SubmitDraftErrorCode) else str(code)
        for code in codes
    ]


def _validation_error_diagnostics(error: ValidationError) -> list[dict[str, object]]:
    return [
        {
            "loc": [str(part) for part in error_item.get("loc", ())],
            "type": str(error_item.get("type", "")),
            "message": str(error_item.get("msg", "")),
        }
        for error_item in error.errors()
    ]


class AnswerOutputBinding(StrictModel):
    label_field: str = Field(
        min_length=1,
        description=(
            "Exact top-level field name from label_json that is meant to be "
            "returned. This names the submitted label field, not a source "
            "table or SQL column."
        ),
    )
    requested_by_phrase: str = Field(
        min_length=1,
        description=(
            "Exact contiguous substring from user_request that asks for this "
            "label field. Use wording that names this field's distinct role; "
            "do not reuse one vague phrase for different returned concepts. "
            "For result/status/type/category/sequence-like fields, preserve the "
            "source representation; do not turn source status text into "
            "boolean completion wording, or source record sequence into "
            "generated display rank. Do not add parenthetical normalized "
            "choices for source type/category/status fields. When two "
            "reachable sources could satisfy the same broad phrase, the "
            "phrase must name the exact source role, such as the current "
            "record's category versus a referenced item's category."
        ),
    )

    @field_validator("label_field", "requested_by_phrase")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        return _strip_required_text(value, field_name="binding text")


class AnswerOrderBinding(StrictModel):
    requested_by_phrase: str = Field(
        min_length=1,
        description=(
            "Exact contiguous substring from user_request that asks for the "
            "row order, recency, ranking, or natural tie-break. Each tie-break "
            "phrase must name that specific order key and its ordering role; "
            "include direction/recency/tie-break wording, not only the bare "
            "output noun; display-only output wording is not enough; do not "
            "reuse one broad order phrase for multiple different keys."
        ),
    )
    direction: Literal["asc", "desc"] | None = Field(
        default=None,
        description=(
            "Requested direction when the wording fixes it, or null when the "
            "direction is not explicit."
        ),
    )
    label_field: str | None = Field(
        default=None,
        description=(
            "Submitted label_json field that also represents the order key, "
            "or null when ordering is requested without returning that key. "
            "Do not put source table or SQL column names here."
        ),
    )

    @field_validator("requested_by_phrase")
    @classmethod
    def _validate_requested_by_phrase(cls, value: str) -> str:
        return _strip_required_text(value, field_name="requested_by_phrase")

    @field_validator("label_field")
    @classmethod
    def _validate_label_field(cls, value: str | None) -> str | None:
        return _strip_optional_text(value, field_name="label_field")


class AnswerContract(StrictModel):
    kind: Literal["scalar", "list"] = Field(
        description=(
            "Answer shape copied from the latest query: scalar means one "
            "aggregate row object; list means the query rows array, even when "
            "one row is returned."
        ),
    )
    answer_phrase: str = Field(
        min_length=1,
        description=(
            "Exact contiguous substring from user_request that states what the "
            "user wants returned. Do not restate tables, columns, or SQL."
        ),
    )
    constraint_phrases: list[str] = Field(
        description=(
            "Exact contiguous user_request substrings for meaningful filters, "
            "entity scope, ordering, or tie-breaks. Use [] only when the "
            "request truly has no additional constraint beyond the answer "
            "target. If query.order_by uses tie-break fields, user_request "
            "must visibly ask for that secondary order here; merely selecting "
            "the field as output is not enough. Sequence/rank wording must "
            "separate source record sequence from generated display rank. "
            "Non-null filters and date/time granularity must be explicit row-set "
            "or representation constraints, not inferred from output fields. "
            "Source type/category/status filters must use source-role wording, "
            "not broad synonyms. "
            "Structural evidence is derived from the latest query."
        ),
    )
    limit_phrase: str | None = Field(
        description=(
            "Exact user_request substring for a fixed requested list size, "
            "such as '3 items', or null when there is no fixed size phrase. "
            "For ordered limited lists, the limit phrase and order bindings "
            "must communicate the same row-selection boundary as query.order_by; "
            "do not imply a different boundary from the query order."
        ),
    )
    output_bindings: list[AnswerOutputBinding] | None = Field(
        default=None,
        description=(
            "Request-to-label bindings for fields returned in label_json. "
            "For list labels, provide one binding for every returned label "
            "field. For scalar labels, omit or use null when the answer phrase "
            "already binds the result. Broad output words are not enough when "
            "multiple reachable source surfaces could answer them."
        ),
    )
    order_bindings: list[AnswerOrderBinding] | None = Field(
        default=None,
        description=(
            "For ordered lists, provide one request-to-order binding for each "
            "query.order_by entry, in the same order. Use label_field when the "
            "order key is returned in label_json; otherwise use null. Omit or "
            "use null only when the query has no ordering. For limited lists, "
            "this ordering also selects row membership; avoid drafts requiring "
            "one hidden selection order and another display order. A bare field "
            "noun that only asks to display the field is not an order binding."
        ),
    )

    @field_validator("answer_phrase")
    @classmethod
    def _validate_answer_phrase(cls, value: str) -> str:
        return _strip_required_text(value, field_name="answer_phrase")

    @field_validator("constraint_phrases")
    @classmethod
    def _validate_constraint_phrases(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for phrase in value:
            stripped = phrase.strip()
            if not stripped:
                raise ValueError("constraint_phrases must not contain blank strings")
            normalized.append(stripped)
        return normalized


class SubmitDraftPayload(StrictModel):
    _cached_canonical_answer: object | None = PrivateAttr(default=None)
    _cached_parsed_entity: dict[str, JsonScalar] | None = PrivateAttr(default=None)

    topic: str = Field(
        min_length=1,
        description="Selected topic string derived from the grounded label and evidence.",
    )
    label_json: str = Field(
        min_length=1,
        description=(
            "JSON string for the canonical submit_result payload, copied "
            "exactly from the latest successful query result. For scalar, the "
            "JSON must encode one object with the aggregate field. For list, "
            "it must encode an array of row objects. Do not expose hidden "
            "PK/FK handle values as answer values, and do not make a raw "
            "handle the main selected answer merely because it is easy to "
            "query. The label must answer the exact scope of user_request; if "
            "the request is about the hidden entity's own records, the latest "
            "query must be scoped to that entity before you copy the result. "
            "Do not submit a global answer that can be produced without the "
            "hidden entity. Include only answer fields the user_request asks "
            "to receive; if the latest query selected helper/context fields, "
            "rerun query with only the fields intended for submit_result. Do "
            "not include profile/scope fields merely to identify the current "
            "entity unless the request asks for them. Constraint, filter, "
            "scope, ordering, and tie-break values belong in user_request and "
            "answer_contract phrases, not in label_json, unless the user also "
            "asks to receive those values."
        ),
    )
    entity_json: str = Field(
        min_length=1,
        description=(
            "JSON string for the hidden current-context grounding handle, e.g. "
            '{"<pk_name>": 123}. It may contain observed primary-key values; '
            "those values should stay hidden from user_request. This is not a "
            "decorative anchor: the canonical label must be scoped to this "
            "context, either directly or through observed values derived from it. "
            "If the answer rows come from a parent/list/history scope, put that "
            "parent or current-subject key in entity instead of only a child "
            "record key."
        ),
    )
    user_request: str = Field(
        min_length=1,
        validation_alias=AliasChoices("user_request", "question"),
        description=(
            "Natural user-facing request body in the configured task language. "
            "Do not include the hidden <entity> block; provide only the request "
            "body. The user does not know DB tables, rows, primary keys, "
            "foreign keys, or hidden structural handles. Use a visible value "
            "only when it appeared in tool evidence; copy visible source "
            "values exactly instead of translating or transliterating them. "
            "When multiple reachable source surfaces could satisfy broad "
            "wording, name the chosen source role in ordinary language. "
            "Use 'my'/'own' wording "
            "only when the hidden context naturally represents the requester "
            "or their records and the latest query is scoped to that context. "
            "Target-language wording must be fluent and must not contain "
            "mixed-script artifacts. "
            "Scope wording must match latest query evidence: direct hidden "
            "current-record handle lookups ask for that current record's own "
            "facts; parent period, list, or history wording requires query "
            "evidence at that broader parent/list/history scope."
        ),
    )
    answer_contract: AnswerContract = Field(
        description=(
            "Minimal request-binding contract. Provide the answer shape and "
            "exact user_request phrases for the answer target, entity scope, "
            "filters, ordering, tie-breaks, or fixed list size. Optional "
            "binding fields may map returned label fields and order wording "
            "back to exact request phrases. Do not restate tables, columns, "
            "operators, or SQL; the latest successful query supplies "
            "structural evidence."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_dynamic_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        copied = dict(value)
        if "label" in copied:
            legacy_label = copied.pop("label")
            if "label_json" not in copied:
                copied["label_json"] = legacy_label
        if "entity" in copied:
            legacy_entity = copied.pop("entity")
            if "entity_json" not in copied:
                copied["entity_json"] = legacy_entity
        return copied

    @field_validator("label_json", mode="before")
    @classmethod
    def _validate_label_json(cls, value: object) -> str:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("label must be valid JSON") from exc
        else:
            parsed = value
            value = json.dumps(parsed, ensure_ascii=False)
        if not isinstance(parsed, (dict, list)):
            raise ValueError("label must encode an object or array")
        if isinstance(parsed, dict) and not parsed:
            raise ValueError("label must not be empty")
        if isinstance(parsed, list) and not parsed:
            raise ValueError("label must not be empty")
        if isinstance(parsed, list) and not all(isinstance(item, dict) for item in parsed):
            raise ValueError("label array items must be objects")
        return str(value).strip()

    @field_validator("entity_json", mode="before")
    @classmethod
    def _validate_entity_json(cls, value: object) -> str:
        if not isinstance(value, (dict, list)):
            raw = str(value).strip() if not isinstance(value, str) else value.strip()
            if not raw:
                raise ValueError("entity must not be blank")
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError("entity must be a valid JSON object") from exc
        else:
            parsed = value
            raw = json.dumps(parsed, ensure_ascii=False)
        if not isinstance(parsed, dict):
            raise ValueError("entity must encode a JSON object")
        _normalize_anchor_entity_map(parsed)
        return raw

    @field_validator("topic")
    @classmethod
    def _validate_non_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text fields must not be blank")
        return normalized

    @field_validator("user_request", mode="before")
    @classmethod
    def _validate_user_request(cls, value: object) -> str:
        raw = str(value).strip() if not isinstance(value, str) else value.strip()
        if not raw:
            raise ValueError("text fields must not be blank")
        if raw.startswith("<entity>"):
            _, body, prompt_error = _split_entity_wrapped_prompt(raw)
            if prompt_error is None and body is not None:
                return body
        return raw

    @field_validator("answer_contract", mode="before")
    @classmethod
    def _validate_answer_contract(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("answer_contract must be a JSON object") from exc
            if not isinstance(parsed, dict):
                raise ValueError("answer_contract must be a JSON object")
            return parsed
        return value

    @property
    def parsed_entity(self) -> dict[str, JsonScalar]:
        if self._cached_parsed_entity is not None:
            return self._cached_parsed_entity
        parsed = json.loads(self.entity_json)
        if not isinstance(parsed, dict):
            raise ValueError("entity must encode a JSON object")
        self._cached_parsed_entity = _normalize_anchor_entity_map(parsed)
        return self._cached_parsed_entity

    @property
    def canonical_answer(self) -> object:
        if self._cached_canonical_answer is None:
            self._cached_canonical_answer = json.loads(self.label_json)
        return self._cached_canonical_answer

    @property
    def entity(self) -> dict[str, JsonScalar]:
        return self.parsed_entity

    @property
    def label(self) -> object:
        return self.canonical_answer

    @property
    def question(self) -> str:
        return self.user_request


SubmitDraftPayload.model_rebuild()


def _normalize_anchor_entity_map(value: dict[str, object]) -> dict[str, JsonScalar]:
    if not value:
        raise ValueError("anchor_entity must contain at least one primary-key value")
    normalized: dict[str, JsonScalar] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError("anchor_entity keys must be non-empty strings")
        if not isinstance(raw_value, (str, int, float, bool)):
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
    ci_lower: float | None = None
    ci_upper: float | None = None
    matched_solver_runs: int | None = None
    planned_solver_runs: int | None = None
    total_solver_runs: int | None = None
    evaluable_solver_runs: int | None = None
    failed_solver_runs: int | None = None


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


def _single_field_scalar_value_signature(value: object) -> str | None:
    if not isinstance(value, dict) or len(value) != 1:
        return None
    scalar_value = next(iter(value.values()))
    if isinstance(scalar_value, (dict, list)):
        return None
    return canonical_json(scalar_value, default=str)


def _null_answer_fields(value: object) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key, field_value in value.items() if field_value is None)
    if not isinstance(value, list):
        return []
    field_names = sorted(
        {
            str(key)
            for item in value
            if isinstance(item, dict)
            for key in item
        }
    )
    null_fields: list[str] = []
    for field_name in field_names:
        if any(
            item.get(field_name) is None
            for item in value
            if isinstance(item, dict) and field_name in item
        ):
            null_fields.append(field_name)
    return null_fields


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
            "canonical_answer_scalar_value_signature": None,
        }

    if isinstance(payload, SubmitDraftPayload):
        canonical_answer = payload.canonical_answer
    else:
        canonical_answer = None
        raw_canonical = payload.get("label_json", payload.get("label"))
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
        "canonical_answer_scalar_value_signature": (
            _single_field_scalar_value_signature(canonical_answer)
            if canonical_answer is not None
            else None
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
    previous_scalar_signature = (
        previous.get("canonical_answer_scalar_value_signature") if previous else None
    )
    current_scalar_signature = current.get("canonical_answer_scalar_value_signature")
    return {
        "label_changed": (
            previous.get("canonical_answer_signature") != current.get("canonical_answer_signature")
            if previous is not None
            else None
        ),
        "scalar_value_changed": (
            previous_scalar_signature != current_scalar_signature
            if previous_scalar_signature is not None and current_scalar_signature is not None
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


def _contract_phrase_surface(value: str) -> str:
    return " ".join(value.casefold().split())


def _phrase_is_in_request(*, phrase: str | None, user_request: str) -> bool:
    if phrase is None:
        return True
    normalized_phrase = _contract_phrase_surface(phrase)
    if not normalized_phrase:
        return False
    return normalized_phrase in _contract_phrase_surface(user_request)


def _contract_component_phrase_errors(
    contract: AnswerContract,
    *,
    user_request: str,
) -> list[str]:
    missing_phrases: list[str] = []
    components: list[tuple[str, str | None]] = [
        ("answer_phrase", contract.answer_phrase),
        ("limit", contract.limit_phrase),
    ]
    components.extend(
        (f"constraint_phrases[{index}]", phrase)
        for index, phrase in enumerate(contract.constraint_phrases)
    )
    components.extend(
        (
            f"output_bindings[{index}].requested_by_phrase",
            binding.requested_by_phrase,
        )
        for index, binding in enumerate(contract.output_bindings or [])
    )
    components.extend(
        (
            f"order_bindings[{index}].requested_by_phrase",
            binding.requested_by_phrase,
        )
        for index, binding in enumerate(contract.order_bindings or [])
    )
    for path, phrase in components:
        if phrase is not None and not _phrase_is_in_request(
            phrase=phrase,
            user_request=user_request,
        ):
            missing_phrases.append(path)
    return missing_phrases


def _answer_field_names(value: object) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value)
    if isinstance(value, list):
        field_names: set[str] = set()
        for item in value:
            if isinstance(item, dict):
                field_names.update(str(key) for key in item)
        return sorted(field_names)
    return []


def _ordered_unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _duplicate_output_binding_phrases(
    output_bindings: list[AnswerOutputBinding],
) -> list[dict[str, object]]:
    fields_by_phrase: dict[str, list[str]] = {}
    for binding in output_bindings:
        fields_by_phrase.setdefault(binding.requested_by_phrase, []).append(
            binding.label_field
        )
    duplicates: list[dict[str, object]] = []
    for phrase, fields in fields_by_phrase.items():
        unique_fields = _ordered_unique_texts(fields)
        if len(unique_fields) > 1:
            duplicates.append(
                {
                    "requested_by_phrase": phrase,
                    "label_fields": unique_fields,
                }
            )
    return duplicates


def _duplicate_order_binding_phrases(
    order_bindings: list[AnswerOrderBinding],
) -> list[dict[str, object]]:
    fields_by_phrase: dict[str, list[str]] = {}
    for index, binding in enumerate(order_bindings):
        field = (
            binding.label_field
            if binding.label_field is not None
            else f"order_bindings[{index}]"
        )
        fields_by_phrase.setdefault(binding.requested_by_phrase, []).append(field)
    duplicates: list[dict[str, object]] = []
    for phrase, fields in fields_by_phrase.items():
        unique_fields = _ordered_unique_texts(fields)
        if len(unique_fields) > 1:
            duplicates.append(
                {
                    "requested_by_phrase": phrase,
                    "label_fields": unique_fields,
                }
            )
    return duplicates


def _order_binding_reused_output_phrases(
    *,
    order_bindings: list[AnswerOrderBinding],
    output_bindings: list[AnswerOutputBinding],
) -> list[dict[str, object]]:
    output_fields_by_phrase: dict[str, list[str]] = {}
    for binding in output_bindings:
        output_fields_by_phrase.setdefault(binding.requested_by_phrase, []).append(
            binding.label_field
        )

    reused: list[dict[str, object]] = []
    for index, binding in enumerate(order_bindings):
        output_fields = output_fields_by_phrase.get(binding.requested_by_phrase)
        if not output_fields:
            continue
        reused.append(
            {
                "requested_by_phrase": binding.requested_by_phrase,
                "order_binding": (
                    binding.label_field
                    if binding.label_field is not None
                    else f"order_bindings[{index}]"
                ),
                "output_fields": _ordered_unique_texts(output_fields),
            }
        )
    return reused


def _query_order_output_names(query_result: dict[str, object]) -> list[str]:
    diagnostics = query_result.get("ordering_diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    raw_outputs = diagnostics.get("order_by_outputs")
    if not isinstance(raw_outputs, list):
        return []
    return _ordered_unique_texts(
        [output for output in raw_outputs if isinstance(output, str)]
    )


def _query_order_reference_count(query_result: dict[str, object]) -> int:
    referenced_columns = _as_object_list(query_result.get("referenced_columns")) or []
    return sum(1 for ref in referenced_columns if ref.get("usage") == "order_by")


def _query_order_is_too_complex(query_result: dict[str, object]) -> bool:
    return _query_order_reference_count(query_result) > 2


def _answer_contract_binding_diagnostics(
    contract: AnswerContract,
    *,
    user_request: str,
    canonical_answer: object,
    query_result: dict[str, object],
    query_limit: int | None,
    item_limit: int,
) -> dict[str, object]:
    label_fields = _answer_field_names(canonical_answer)
    label_field_set = set(label_fields)
    output_bindings = contract.output_bindings or []
    order_bindings = contract.order_bindings or []
    order_reference_count = _query_order_reference_count(query_result)
    answer_row_count = len(canonical_answer) if isinstance(canonical_answer, list) else None
    required_order_reference_count = order_reference_count
    if answer_row_count is not None and answer_row_count <= 1 and query_limit is None:
        required_order_reference_count = 0
    bound_output_fields = _ordered_unique_texts(
        [binding.label_field for binding in output_bindings]
    )
    duplicate_output_binding_phrases = _duplicate_output_binding_phrases(
        output_bindings
    )
    duplicate_order_binding_phrases = _duplicate_order_binding_phrases(order_bindings)
    order_binding_reused_output_phrases = _order_binding_reused_output_phrases(
        order_bindings=order_bindings,
        output_bindings=output_bindings,
    )
    bound_order_label_fields = _ordered_unique_texts(
        [binding.label_field for binding in order_bindings if binding.label_field is not None]
    )
    order_output_fields = [
        field_name
        for field_name in _query_order_output_names(query_result)
        if field_name in label_field_set
    ]
    missing_requested_by_phrases: list[str] = []
    missing_requested_by_phrase_bindings: list[dict[str, object]] = []
    for index, binding in enumerate(output_bindings):
        path = f"output_bindings[{index}].requested_by_phrase"
        if not _phrase_is_in_request(
            phrase=binding.requested_by_phrase,
            user_request=user_request,
        ):
            missing_requested_by_phrases.append(path)
            missing_requested_by_phrase_bindings.append(
                {
                    "path": path,
                    "label_field": binding.label_field,
                    "requested_by_phrase": binding.requested_by_phrase,
                }
            )
    for index, binding in enumerate(order_bindings):
        path = f"order_bindings[{index}].requested_by_phrase"
        if not _phrase_is_in_request(
            phrase=binding.requested_by_phrase,
            user_request=user_request,
        ):
            missing_requested_by_phrases.append(path)
            missing_requested_by_phrase_bindings.append(
                {
                    "path": path,
                    "label_field": binding.label_field,
                    "requested_by_phrase": binding.requested_by_phrase,
                }
            )

    return {
        "label_fields": label_fields[:item_limit],
        "bound_output_fields": bound_output_fields[:item_limit],
        "missing_output_bindings": sorted(
            label_field_set - set(bound_output_fields)
        )[:item_limit],
        "extra_output_bindings": sorted(
            set(bound_output_fields) - label_field_set
        )[:item_limit],
        "order_reference_count": order_reference_count,
        "required_order_reference_count": required_order_reference_count,
        "order_binding_count": len(order_bindings),
        "missing_order_binding_count": max(
            0,
            required_order_reference_count - len(order_bindings),
        ),
        "order_output_fields": order_output_fields[:item_limit],
        "bound_order_label_fields": bound_order_label_fields[:item_limit],
        "missing_order_label_bindings": (
            sorted(set(order_output_fields) - set(bound_order_label_fields))[:item_limit]
            if required_order_reference_count > 0
            else []
        ),
        "extra_order_label_fields": sorted(
            set(bound_order_label_fields) - label_field_set
        )[:item_limit],
        "missing_requested_by_phrases": missing_requested_by_phrases[:item_limit],
        "missing_requested_by_phrase_bindings": (
            missing_requested_by_phrase_bindings[:item_limit]
        ),
        "duplicate_output_binding_phrases": duplicate_output_binding_phrases[
            :item_limit
        ],
        "duplicate_order_binding_phrases": duplicate_order_binding_phrases[
            :item_limit
        ],
        "order_binding_reused_output_phrases": order_binding_reused_output_phrases[
            :item_limit
        ],
    }


def _answer_contract_binding_errors(
    diagnostics: dict[str, object],
    *,
    require_output_bindings: bool,
) -> list[str]:
    errors: list[str] = []
    if require_output_bindings:
        missing_outputs = diagnostics.get("missing_output_bindings")
        if isinstance(missing_outputs, list) and missing_outputs:
            errors.append("missing_output_bindings")
        extra_outputs = diagnostics.get("extra_output_bindings")
        if isinstance(extra_outputs, list) and extra_outputs:
            errors.append("extra_output_bindings")
        duplicate_output_phrases = diagnostics.get("duplicate_output_binding_phrases")
        if isinstance(duplicate_output_phrases, list) and duplicate_output_phrases:
            errors.append("duplicate_output_binding_phrases")
    missing_by_count = diagnostics.get("missing_order_binding_count")
    if isinstance(missing_by_count, int) and missing_by_count > 0:
        errors.append("missing_order_bindings")
    missing_label_bindings = diagnostics.get("missing_order_label_bindings")
    if isinstance(missing_label_bindings, list) and missing_label_bindings:
        errors.append("missing_order_label_bindings")
    duplicate_order_phrases = diagnostics.get("duplicate_order_binding_phrases")
    if isinstance(duplicate_order_phrases, list) and duplicate_order_phrases:
        errors.append("duplicate_order_binding_phrases")
    reused_output_phrases = diagnostics.get("order_binding_reused_output_phrases")
    if isinstance(reused_output_phrases, list) and reused_output_phrases:
        errors.append("order_binding_reused_output_phrases")
    return errors


_ORDER_BINDING_ERROR_NAMES = frozenset(
    {
        "missing_order_bindings",
        "missing_order_label_bindings",
        "duplicate_order_binding_phrases",
        "order_binding_reused_output_phrases",
    }
)


def _referenced_predicate_signature(ref: dict[str, object]) -> str | None:
    if ref.get("usage") != "where":
        return None
    table = ref.get("table")
    column = ref.get("column")
    op = ref.get("op")
    if not isinstance(table, str) or not isinstance(column, str) or not isinstance(op, str):
        return None
    return canonical_json(
        {
            "table": table,
            "column": column,
            "op": op,
            "value": ref.get("value"),
        },
        default=str,
    )


def _referenced_order_signature(ref: dict[str, object]) -> str | None:
    if ref.get("usage") != "order_by":
        return None
    table = ref.get("table")
    column = ref.get("column")
    direction = ref.get("direction")
    if (
        not isinstance(table, str)
        or not isinstance(column, str)
        or direction not in {"asc", "desc"}
    ):
        return None
    return canonical_json(
        {
            "table": table,
            "column": column,
            "direction": direction,
        },
        default=str,
    )


def _query_output_signature(source: dict[str, object]) -> str | None:
    kind = source.get("kind")
    if not isinstance(kind, str):
        return None
    payload: dict[str, object] = {"kind": kind}
    for key in ("fn", "table", "column", "value_exposes_source"):
        value = source.get(key)
        if value is not None:
            payload[key] = value
    return canonical_json(payload, default=str)


@dataclass(frozen=True, slots=True)
class QueryEvidenceSignature:
    kind: str
    output_sources: tuple[str, ...]
    output_kinds: tuple[str, ...]
    aggregate_fns: tuple[str, ...]
    predicates: tuple[str, ...]
    order_by: tuple[str, ...]
    item_count: int | None
    query_tables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LimitRepairScope:
    evidence: QueryEvidenceSignature
    max_item_count: int


def _query_evidence_signature(
    query_result: dict[str, object],
    *,
    answer_kind: str,
    query_params: dict[str, object] | None = None,
) -> QueryEvidenceSignature:
    column_sources = _as_object_list(query_result.get("column_sources")) or []
    referenced_columns = _as_object_list(query_result.get("referenced_columns")) or []
    rows = query_result.get("rows")
    item_count = len(rows) if answer_kind == "list" and isinstance(rows, list) else None
    return QueryEvidenceSignature(
        kind=answer_kind,
        output_sources=tuple(
            signature
            for source in column_sources
            if (signature := _query_output_signature(source)) is not None
        ),
        output_kinds=tuple(
            sorted({
                kind
                for source in column_sources
                if isinstance(kind := source.get("kind"), str)
            })
        ),
        aggregate_fns=tuple(
            sorted({
                fn
                for source in column_sources
                if source.get("kind") == "aggregate"
                and isinstance(fn := source.get("fn"), str)
            })
        ),
        predicates=tuple(
            sorted(
                signature
                for ref in referenced_columns
                if (signature := _referenced_predicate_signature(ref)) is not None
            )
        ),
        order_by=tuple(
            sorted(
                signature
                for ref in referenced_columns
                if (signature := _referenced_order_signature(ref)) is not None
            )
        ),
        item_count=item_count,
        query_tables=_query_spec_tables(query_params),
    )


def _query_spec_tables(query_params: dict[str, object] | None) -> tuple[str, ...]:
    if not isinstance(query_params, dict):
        return ()
    spec = query_params.get("spec")
    if not isinstance(spec, dict):
        return ()
    tables: set[str] = set()
    from_spec = spec.get("from")
    if isinstance(from_spec, dict) and isinstance(from_spec.get("table"), str):
        tables.add(from_spec["table"])
    for join in _as_object_list(spec.get("join")) or []:
        edge = _parse_query_edge(join.get("via_edge"))
        if edge is None:
            continue
        for key in ("source_table", "destination_table"):
            table = edge.get(key)
            if isinstance(table, str):
                tables.add(table)
    return tuple(sorted(tables))


def _allows_scalar_to_grouped_aggregate(
    previous: QueryEvidenceSignature,
    current: QueryEvidenceSignature,
) -> bool:
    previous_tables = set(previous.query_tables)
    current_tables = set(current.query_tables)
    return (
        previous.kind == "scalar"
        and current.kind == "list"
        and "aggregate" in previous.output_kinds
        and "aggregate" in current.output_kinds
        and "group_by" in current.output_kinds
        and bool(set(previous.aggregate_fns) & set(current.aggregate_fns))
        and (not previous_tables or previous_tables.issubset(current_tables))
    )


def _query_evidence_incremental_errors(
    *,
    previous: QueryEvidenceSignature,
    current: QueryEvidenceSignature,
) -> list[str]:
    errors: list[str] = []
    scalar_to_grouped = _allows_scalar_to_grouped_aggregate(previous, current)
    if current.kind != previous.kind and not scalar_to_grouped:
        errors.append("kind_changed")
    previous_outputs = set(previous.output_sources)
    current_outputs = set(current.output_sources)
    if not scalar_to_grouped and not previous_outputs.issubset(current_outputs):
        errors.append("operation_changed")

    previous_predicates = set(previous.predicates)
    current_predicates = set(current.predicates)
    missing_predicates = sorted(previous_predicates - current_predicates)
    if missing_predicates:
        errors.append("predicate_removed")

    previous_order = set(previous.order_by)
    current_order = set(current.order_by)
    missing_order = sorted(previous_order - current_order)
    if missing_order:
        errors.append("order_removed")

    added_predicate = bool(current_predicates - previous_predicates)
    added_order = bool(current_order - previous_order)
    added_output_source = bool(current_outputs - previous_outputs)
    strengthened_cardinality = False
    if current.kind == "list":
        if previous.item_count is None and current.item_count is not None:
            strengthened_cardinality = True
        elif (
            previous.item_count is not None
            and current.item_count is not None
            and current.item_count > previous.item_count
        ):
            strengthened_cardinality = True
        elif (
            previous.item_count is not None
            and current.item_count is not None
            and current.item_count < previous.item_count
        ):
            errors.append("cardinality_weakened")

    output_only_list_retry = (
        current.kind == "list"
        and added_output_source
        and not (added_predicate or added_order or strengthened_cardinality)
    )
    if output_only_list_retry:
        errors.append("list_output_only")

    output_source_strengthens = added_output_source and current.kind != "list"
    if not (
        added_predicate
        or added_order
        or output_source_strengthens
        or strengthened_cardinality
    ):
        errors.append("no_new_structural_constraint")
    return list(dict.fromkeys(errors))


def _query_evidence_limit_repair_errors(
    *,
    previous: LimitRepairScope,
    current: QueryEvidenceSignature,
) -> list[str]:
    errors: list[str] = []
    previous_evidence = previous.evidence
    if current.kind != previous_evidence.kind:
        errors.append("kind_changed")
    if current.output_sources != previous_evidence.output_sources:
        errors.append("operation_changed")
    if current.predicates != previous_evidence.predicates:
        errors.append("predicate_changed")
    if current.order_by != previous_evidence.order_by:
        errors.append("order_changed")
    if current.item_count is None:
        errors.append("list_count_missing")
    elif current.item_count < 3 or current.item_count > 5:
        errors.append("list_limit_not_3_to_5")
    elif current.item_count > previous.max_item_count:
        errors.append("list_limit_expanded")
    return list(dict.fromkeys(errors))


def _canonical_from_latest_query_result(
    result: dict[str, object],
    *,
    answer_kind: str | None = None,
) -> object | None:
    rows = result.get("rows")
    if not isinstance(rows, list):
        return None
    if answer_kind == "list":
        return rows
    if answer_kind == "scalar":
        if len(rows) == 1 and isinstance(rows[0], dict):
            return rows[0]
        return None
    if len(rows) == 1 and isinstance(rows[0], dict):
        return rows[0]
    return rows


def _canonical_label_matches_query_result(
    *,
    label: object,
    query_result: dict[str, object],
    answer_kind: str | None = None,
) -> bool:
    expected = _canonical_from_latest_query_result(
        query_result,
        answer_kind=answer_kind,
    )
    if expected is None:
        return False
    return canonical_json(label, default=str) == canonical_json(expected, default=str)


def _query_limit_from_params(params: object) -> int | None:
    if not isinstance(params, dict):
        return None
    spec = params.get("spec")
    if not isinstance(spec, dict):
        return None
    limit = spec.get("limit")
    if isinstance(limit, int) and limit > 0:
        return limit
    return None


def _query_limit_shapes_membership(
    query_result: dict[str, object],
    *,
    query_limit: int | None,
) -> bool:
    if query_limit is None:
        return False
    rows = query_result.get("rows")
    if not isinstance(rows, list) or len(rows) != query_limit:
        return False
    if query_limit != 1:
        return True
    return not _query_sources_are_primary_key_constrained(query_result)


def _query_sources_are_primary_key_constrained(query_result: dict[str, object]) -> bool:
    referenced_columns = _as_object_list(query_result.get("referenced_columns")) or []
    constrained_pk_columns_by_table: dict[str, set[str]] = {}
    primary_key_by_table: dict[str, tuple[str, ...]] = {}
    for ref in referenced_columns:
        if (
            ref.get("usage") != "where"
            or ref.get("op") != "eq"
            or ref.get("is_primary_key") is not True
        ):
            continue
        table = ref.get("table")
        column = ref.get("column")
        primary_key = ref.get("table_primary_key")
        if not isinstance(table, str) or not isinstance(column, str):
            continue
        if not isinstance(primary_key, list) or not all(
            isinstance(item, str) for item in primary_key
        ):
            continue
        constrained_pk_columns_by_table.setdefault(table, set()).add(column)
        primary_key_by_table[table] = tuple(primary_key)

    label_sources = _as_object_list(query_result.get("column_sources")) or []
    label_tables: set[str] = set()
    for source in label_sources:
        if source.get("value_exposes_source") is False:
            continue
        table = source.get("table")
        if isinstance(table, str):
            label_tables.add(table)
    if not label_tables:
        return False
    for table in label_tables:
        primary_key = primary_key_by_table.get(table)
        if not primary_key:
            return False
        if not set(primary_key).issubset(constrained_pk_columns_by_table.get(table, set())):
            return False
    return True


def _query_ordering_is_ambiguous(query_result: dict[str, object]) -> bool:
    rows = query_result.get("rows")
    if isinstance(rows, list) and len(rows) > 1:
        referenced_columns = _as_object_list(query_result.get("referenced_columns")) or []
        has_order_by = any(ref.get("usage") == "order_by" for ref in referenced_columns)
        if not has_order_by:
            return True
    diagnostics = query_result.get("ordering_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    return bool(
        diagnostics.get("missing_order_by_for_limit")
        or diagnostics.get("duplicate_order_key_in_returned_rows")
        or diagnostics.get("limit_boundary_tie")
        or diagnostics.get("unrepresented_order_by_tie_breakers")
        or diagnostics.get("unrepresented_handle_tie_breakers")
    )


def _query_projection_has_duplicate_answer_rows(
    query_result: dict[str, object],
) -> bool:
    diagnostics = query_result.get("projection_diagnostics")
    return (
        isinstance(diagnostics, dict)
        and diagnostics.get("duplicate_answer_rows") is True
    )


def _query_result_has_no_rows(query_result: dict[str, object]) -> bool:
    rows = query_result.get("rows")
    if isinstance(rows, list):
        return len(rows) == 0
    return query_result.get("row_count") == 0


def _dedicated_constraint_phrase_surfaces(contract: AnswerContract) -> set[str]:
    non_constraint_surfaces = {
        _contract_phrase_surface(contract.answer_phrase),
    }
    if contract.limit_phrase is not None:
        non_constraint_surfaces.add(_contract_phrase_surface(contract.limit_phrase))
    non_constraint_surfaces.update(
        _contract_phrase_surface(binding.requested_by_phrase)
        for binding in contract.output_bindings or []
    )
    non_constraint_surfaces.update(
        _contract_phrase_surface(binding.requested_by_phrase)
        for binding in contract.order_bindings or []
    )
    return {
        surface
        for phrase in contract.constraint_phrases
        if (surface := _contract_phrase_surface(phrase))
        and surface not in non_constraint_surfaces
    }


def _unbound_user_visible_filters(
    query_result: dict[str, object],
    *,
    contract: AnswerContract,
) -> list[dict[str, object]]:
    referenced_columns = _as_object_list(query_result.get("referenced_columns")) or []
    if _dedicated_constraint_phrase_surfaces(contract):
        return []
    unbound: list[dict[str, object]] = []
    for ref in referenced_columns:
        if ref.get("usage") != "where":
            continue
        if (
            is_blocked_visibility(ref.get("visibility"))
            or ref.get("is_handle") is True
        ):
            continue
        unbound.append(
            {
                "table": ref.get("table"),
                "column": ref.get("column"),
                "op": ref.get("op"),
                "value": ref.get("value") if "value" in ref else None,
            }
        )
    return unbound


def _unanchored_hidden_filter_sources(
    query_result: dict[str, object],
    *,
    anchor_entity: dict[str, object],
) -> list[dict[str, object]]:
    referenced_columns = _as_object_list(query_result.get("referenced_columns")) or []
    anchor_values = set(anchor_entity.values())
    unanchored: list[dict[str, object]] = []
    for ref in referenced_columns:
        if ref.get("usage") != "where":
            continue
        if (
            not is_blocked_visibility(ref.get("visibility"))
            or ref.get("is_handle") is not True
        ):
            continue
        value = ref.get("value")
        if "value" not in ref:
            continue
        if isinstance(value, (str, int, float, bool)) and value in anchor_values:
            continue
        unanchored.append(
            {
                "table": ref.get("table"),
                "column": ref.get("column"),
                "op": ref.get("op"),
            }
        )
    return unanchored


def _edge_column_names(edge_side: str) -> tuple[str, ...]:
    if ".(" in edge_side and edge_side.endswith(")"):
        _, columns = edge_side.split(".(", 1)
        return tuple(column.strip() for column in columns[:-1].split(",") if column.strip())
    if "." not in edge_side:
        return ()
    column = edge_side.rsplit(".", 1)[1].strip()
    return (column,) if column else ()


def _parse_query_edge(via_edge: object) -> dict[str, object] | None:
    if not isinstance(via_edge, str):
        return None
    if "->" in via_edge:
        left, right = via_edge.split("->", 1)
        source_table = left.split(".", 1)[0]
        destination_table = right.split(".", 1)[0]
        destination_columns = _edge_column_names(right) or _edge_column_names(left)
        return {
            "direction": "forward",
            "source_table": source_table,
            "destination_table": destination_table,
            "destination_columns": destination_columns,
        }
    if "<-" in via_edge:
        left, right = via_edge.split("<-", 1)
        return {
            "direction": "reverse",
            "source_table": left.split(".", 1)[0],
            "destination_table": right.split(".", 1)[0],
            "destination_columns": _edge_column_names(right),
        }
    return None


def _hidden_scope_relay_sources(
    query_params: dict[str, object] | None,
    query_result: dict[str, object],
    *,
    anchor_entity: dict[str, object],
) -> list[dict[str, object]]:
    if not isinstance(query_params, dict):
        return []
    spec = query_params.get("spec")
    if not isinstance(spec, dict):
        return []
    rows = query_result.get("rows")
    if not isinstance(rows, list) or len(rows) <= 1:
        return []
    from_spec = spec.get("from")
    if not isinstance(from_spec, dict):
        return []
    root_table = from_spec.get("table")
    if not isinstance(root_table, str):
        return []
    root_alias = from_spec.get("as") if isinstance(from_spec.get("as"), str) else root_table
    alias_tables: dict[str, str] = {root_alias: root_table}
    joins = _as_object_list(spec.get("join")) or []
    join_edges: list[dict[str, object]] = []
    for join in joins:
        alias = join.get("as")
        from_alias = join.get("from")
        edge = _parse_query_edge(join.get("via_edge"))
        if not isinstance(alias, str) or not isinstance(from_alias, str) or edge is None:
            continue
        destination_table = edge.get("destination_table")
        if not isinstance(destination_table, str):
            continue
        alias_tables[alias] = destination_table
        join_edges.append({"as": alias, "from": from_alias, **edge})

    anchor_values = set(anchor_entity.values())
    hidden_where_sources: set[tuple[str, str]] = set()
    for ref in _as_object_list(query_result.get("referenced_columns")) or []:
        if (
            ref.get("usage") == "where"
            and ref.get("is_handle") is True
            and is_blocked_visibility(ref.get("visibility"))
            and ref.get("value") in anchor_values
            and isinstance(ref.get("table"), str)
            and isinstance(ref.get("column"), str)
        ):
            hidden_where_sources.add((ref["table"], ref["column"]))
    hidden_filter_aliases: set[str] = set()
    for predicate in _as_object_list(spec.get("where")) or []:
        ref = predicate.get("ref")
        if not isinstance(ref, dict):
            continue
        alias = ref.get("as")
        column = ref.get("column")
        if not isinstance(alias, str) or not isinstance(column, str):
            continue
        table = alias_tables.get(alias)
        if table is not None and (table, column) in hidden_where_sources:
            hidden_filter_aliases.add(alias)

    if not hidden_filter_aliases:
        return []
    label_tables = {
        source.get("table")
        for source in _as_object_list(query_result.get("column_sources")) or []
        if source.get("value_exposes_source") is True and isinstance(source.get("table"), str)
    }
    relays: list[dict[str, object]] = []
    anchor_keys = set(anchor_entity)
    for first_edge in join_edges:
        if (
            first_edge.get("direction") != "forward"
            or first_edge.get("from") not in hidden_filter_aliases
        ):
            continue
        parent_alias = first_edge.get("as")
        parent_columns = first_edge.get("destination_columns")
        if not isinstance(parent_alias, str) or not isinstance(parent_columns, tuple):
            continue
        if parent_columns and set(parent_columns).issubset(anchor_keys):
            continue
        for second_edge in join_edges:
            if (
                second_edge.get("direction") == "reverse"
                and second_edge.get("from") == parent_alias
                and second_edge.get("destination_table") in label_tables
            ):
                relays.append(
                    {
                        "hidden_anchor_alias": first_edge.get("from"),
                        "parent_alias": parent_alias,
                        "answer_alias": second_edge.get("as"),
                        "parent_columns": list(parent_columns),
                    }
                )
    return relays


def _as_object_list(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None
    entries: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        entries.append({str(key): value for key, value in item.items()})
    return entries


def _query_visibility_errors(
    query_result: dict[str, object],
) -> tuple[list[SubmitDraftErrorCode], dict[str, object]]:
    column_sources = _as_object_list(query_result.get("column_sources"))
    if column_sources is None:
        return (
            [SubmitDraftErrorCode.ANSWER_CONTRACT_VISIBILITY_EVIDENCE_MISSING],
            {"visibility_metadata_missing": True},
        )

    label_sources: list[dict[str, object]] = []
    unstable_record_sources: list[dict[str, object]] = []
    for source in column_sources:
        if source.get("value_exposes_source") is not True:
            continue
        source_payload = {
            key: source.get(key)
            for key in (
                "output",
                "kind",
                "table",
                "column",
                "visibility",
                "table_has_primary_key",
            )
            if key in source
        }
        if blocks_direct_label_exposure(source.get("visibility")):
            label_sources.append(source_payload)
        if source.get("table_has_primary_key") is False:
            unstable_record_sources.append(source_payload)

    codes: list[SubmitDraftErrorCode] = []
    diagnostics: dict[str, object] = {}
    if label_sources:
        codes.append(SubmitDraftErrorCode.LABEL_NON_USER_VISIBLE_SOURCE)
        diagnostics["non_user_visible_label_sources"] = label_sources
    if unstable_record_sources:
        codes.append(SubmitDraftErrorCode.LABEL_NO_PRIMARY_KEY_SOURCE)
        diagnostics["no_primary_key_label_sources"] = unstable_record_sources
    return codes, diagnostics

_ENTITY_BLOCK_PREFIX = "<entity>\n"
_ENTITY_BLOCK_SEPARATOR = "\n</entity>\n\n"


def _split_entity_wrapped_prompt(
    value: str,
) -> tuple[dict[str, object] | None, str | None, SubmitDraftErrorCode | None]:
    normalized = value.strip()
    if not normalized.startswith(_ENTITY_BLOCK_PREFIX):
        return None, None, SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_REQUIRED
    rest = normalized[len(_ENTITY_BLOCK_PREFIX):]
    entity_json, separator, body = rest.partition(_ENTITY_BLOCK_SEPARATOR)
    if separator == "":
        return None, None, SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_REQUIRED
    entity_json = entity_json.strip()
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
    body = body.strip()
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
    _tool_call_count_at_last_protocol_boundary: int = field(default=0, init=False)
    _last_label_signature: str | None = field(default=None, init=False)
    _last_label_scalar_value_signature: str | None = field(default=None, init=False)
    _last_label_slot_count: int | None = field(default=None, init=False)
    _last_answer_contract: AnswerContract | None = field(default=None, init=False)
    _last_query_evidence_signature: QueryEvidenceSignature | None = field(
        default=None,
        init=False,
    )
    _last_monitored_label_data: dict[str, object] | None = field(default=None, init=False)
    _last_evaluated_label_data: dict[str, object] | None = field(default=None, init=False)
    _repair_locked_label_signature: str | None = field(default=None, init=False)
    _limit_repair_scope: LimitRepairScope | None = field(default=None, init=False)
    _feedback_events: int = field(default=0, init=False)
    _last_feedback_error_codes: tuple[str, ...] = field(default=(), init=False)
    _locked_anchor_entity: dict[str, object] | None = field(default=None, init=False)
    _observed_response_strings: set[str] = field(default_factory=set, init=False)
    _terminated_too_hard: bool = field(default=False, init=False)
    event_logger: Any = None

    def submissions_left(self) -> int:
        if self._terminated_too_hard:
            return 0
        return max(0, self.max_submissions - (len(self.attempts) + self._feedback_events))

    @property
    def feedback_events(self) -> int:
        return self._feedback_events

    @property
    def last_feedback_error_codes(self) -> tuple[str, ...]:
        return self._last_feedback_error_codes

    @staticmethod
    def _feedback_codes_lock_label(
        error_codes: list[SubmitDraftErrorCode | str] | tuple[str, ...],
    ) -> bool:
        values = tuple(_error_code_values(error_codes))
        return bool(values) and all(
            value in _LABEL_LOCKING_REPAIR_ERROR_VALUES for value in values
        )

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
        preview_entry = {
            "tool_name": tool_name,
            "params": _preview_runtime_payload(params, config=self.config),
            "result": _preview_runtime_payload(result, config=self.config),
        }
        self._atomic_tool_calls.append(preview_entry)
        _collect_observed_strings(result, strings=self._observed_response_strings)
        if self.event_logger is not None:
            self.event_logger.log_sync(
                actor="composer",
                event_type="atomic_tool_call",
                payload={
                    "tool_name": tool_name,
                    "params_preview": preview_entry["params"],
                    "result_preview": preview_entry["result"],
                    "call_index": len(self._atomic_tool_calls),
                },
            )

    def data_tool_budget_feedback(self, *, tool_name: str) -> dict[str, object] | None:
        if self.accepted_draft is not None or self._terminated_too_hard:
            return None
        calls_since_boundary = (
            len(self._raw_atomic_tool_calls)
            - self._tool_call_count_at_last_protocol_boundary
        )
        if self._last_feedback_error_codes == (
            SubmitDraftErrorCode.ANSWER_CONTRACT_BINDING_MISSING.value,
        ):
            return {
                "error": "submit_draft_required",
                "message": (
                    "ToolBudgetFeedback: Binding repair reminder: "
                    "answer_contract_binding_missing is contract-only. This is "
                    "not a data result; do not call data tools after this "
                    "message. Preserve the current label/query values and "
                    "resubmit with repaired user_request/answer_contract."
                ),
                "calls_since_boundary": calls_since_boundary,
                "limit": 0,
            }
        latest_query_call = self._latest_successful_query_call_since_protocol_boundary()
        latest_query_result = (
            latest_query_call.get("result") if latest_query_call is not None else None
        )
        query_needs_repair = isinstance(latest_query_result, dict) and (
            _query_result_has_no_rows(latest_query_result)
            or _query_ordering_is_ambiguous(latest_query_result)
            or _query_projection_has_duplicate_answer_rows(latest_query_result)
        )
        if tool_name == "query" and (
            latest_query_call is None or query_needs_repair
        ):
            return None
        if not self.attempts and self._feedback_events == 0:
            if calls_since_boundary < FIRST_SUBMIT_MAX_DATA_TOOLS:
                return None
            return {
                "error": "submit_draft_required",
                "message": (
                    "ToolBudgetFeedback: Draft Submission Budget reminder: "
                    "call submit_draft after at most "
                    f"{FIRST_SUBMIT_MAX_DATA_TOOLS} data tools before the first "
                    "submit_draft. This is not a data result; do not call more "
                    "data tools after this message. Use the best grounded label "
                    "query and submit_draft next."
                ),
                "calls_since_boundary": calls_since_boundary,
                "limit": FIRST_SUBMIT_MAX_DATA_TOOLS,
            }
        if calls_since_boundary < FEEDBACK_REPAIR_MAX_DATA_TOOLS:
            return None
        return {
            "error": "submit_draft_required",
            "message": (
                "ToolBudgetFeedback: Draft Submission Budget reminder: after "
                "feedback, call submit_draft after at most "
                f"{FEEDBACK_REPAIR_MAX_DATA_TOOLS} data tools. If the repair "
                "query has returned label values, submit them now. This is not "
                "a data result; do not call more data tools after this message."
            ),
            "calls_since_boundary": calls_since_boundary,
            "limit": FEEDBACK_REPAIR_MAX_DATA_TOOLS,
        }

    def _latest_successful_query_result_since_last_submission(self) -> dict[str, object] | None:
        call = self._latest_successful_query_call_since_last_submission()
        result = call.get("result") if call is not None else None
        return result if isinstance(result, dict) else None

    def _latest_successful_query_call_since_last_submission(self) -> dict[str, object] | None:
        recent_calls = self._raw_atomic_tool_calls[self._tool_call_count_at_last_submission :]
        return self._latest_successful_query_call(recent_calls)

    def _latest_successful_query_call_since_protocol_boundary(
        self,
    ) -> dict[str, object] | None:
        recent_calls = self._raw_atomic_tool_calls[
            self._tool_call_count_at_last_protocol_boundary :
        ]
        return self._latest_successful_query_call(recent_calls)

    @staticmethod
    def _latest_successful_query_call(
        recent_calls: list[dict[str, object]],
    ) -> dict[str, object] | None:
        for call in reversed(recent_calls):
            if call.get("tool_name") != "query":
                continue
            result = call.get("result")
            if (
                isinstance(result, dict)
                and "error" not in result
                and isinstance(result.get("rows"), list)
            ):
                return call
        return None

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
                if location[0] == "user_request":
                    required_code = SubmitDraftErrorCode.QUESTION_REQUIRED
                elif location[0] == "answer_contract":
                    required_code = SubmitDraftErrorCode.ANSWER_CONTRACT_REQUIRED
                elif location[0] in {"entity", "entity_json"}:
                    required_code = SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED
                elif location[0] in {"label", "label_json"}:
                    required_code = SubmitDraftErrorCode.CANONICAL_ANSWER_JSON_INVALID
                else:
                    required_code = getattr(
                        SubmitDraftErrorCode,
                        f"{location[0].upper()}_REQUIRED",
                        SubmitDraftErrorCode.SUBMIT_PAYLOAD_INVALID,
                    )
                error_codes.append(required_code)
            elif location in {("label",), ("label_json",)}:
                error_codes.append(SubmitDraftErrorCode.CANONICAL_ANSWER_JSON_INVALID)
            elif location in {("entity",), ("entity_json",)}:
                if error_type in ("value_error", "dict_type"):
                    error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED)
                else:
                    error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED)
            elif location and location[0] == "answer_contract":
                error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_JSON_INVALID)
            else:
                error_codes.append(SubmitDraftErrorCode.SUBMIT_PAYLOAD_INVALID)
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
                    diagnostics={"validation_errors": _validation_error_diagnostics(error)},
                ),
                error_codes=deduped_error_codes,
                payload=parsed,
                search_cost_observations=search_cost_observations,
                diagnostics={"validation_errors": _validation_error_diagnostics(error)},
            )
        return self._record_rejection(
            submission_index=len(self.attempts) + 1,
            message=self._invalid_submission_message(
                deduped_error_codes,
                diagnostics={"validation_errors": _validation_error_diagnostics(error)},
            ),
            error_codes=deduped_error_codes,
            payload=parsed,
            search_cost_observations=search_cost_observations,
            diagnostics={"validation_errors": _validation_error_diagnostics(error)},
        )

    def reject_malformed_tool_input(
        self,
        *,
        parsed: dict[str, object],
        validation_errors: list[dict[str, object]],
    ) -> str:
        if self.accepted_draft is not None:
            return _render_structured_message(
                kind="Accepted",
                result="Draft already stored.",
            )
        if self.submissions_left() <= 0:
            return "BudgetExhaustedError: No more attempts."

        search_cost_observations = (
            len(self._raw_atomic_tool_calls) - self._tool_call_count_at_last_submission
        )
        diagnostics = {"validation_errors": validation_errors}
        return self._record_rejection(
            submission_index=len(self.attempts) + 1,
            message=self._invalid_submission_message(
                [SubmitDraftErrorCode.SUBMIT_PAYLOAD_INVALID],
                diagnostics=diagnostics,
            ),
            error_codes=[SubmitDraftErrorCode.SUBMIT_PAYLOAD_INVALID],
            payload=parsed,
            search_cost_observations=search_cost_observations,
            diagnostics=diagnostics,
        )

    def record_missing_submit_feedback(
        self,
        *,
        final_output_text: str,
        tool_calls: tuple[str, ...],
    ) -> str:
        if self.accepted_draft is not None:
            return _render_structured_message(
                kind="Accepted",
                result="Draft already stored.",
            )
        if self.submissions_left() <= 0:
            return "BudgetExhaustedError: No more attempts."

        attempts_left_after = self.submissions_left() - 1
        primary = (
            "Protocol reminder: Plain final output is invalid for this role; "
            "the synthesis composer Workflow requires submit_draft."
        )
        message = _render_structured_message(
            kind="FeedbackError",
            primary=primary,
            next_step=(
                "Use data tools if more evidence is missing, then call "
                "submit_draft. Do not end the run with text only."
            ),
            attempts_left=max(0, attempts_left_after),
        )
        diagnostics: dict[str, object] = {
            "final_output_without_submit": True,
            "tool_calls": list(tool_calls),
        }
        final_preview = final_output_text.strip()
        if final_preview:
            diagnostics["final_output_preview"] = preview_payload(
                final_preview,
                max_string_length=(
                    self.config.synthesis.runtime.payload_preview_max_string_length
                ),
                max_list_items=self.config.synthesis.runtime.payload_preview_max_list_items,
                max_dict_items=self.config.synthesis.runtime.payload_preview_max_dict_items,
            )
        return self._record_feedback(
            message=message,
            error_codes=["composer_submit_draft_missing"],
            payload=None,
            search_cost_observations=(
                len(self._raw_atomic_tool_calls) - self._tool_call_count_at_last_submission
            ),
            diagnostics=diagnostics,
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
                "terminal rejection. Stop calling submit_draft."
            )
        if self.submissions_left() <= 0:
            return "BudgetExhaustedError: No more attempts."

        submission_index = len(self.attempts) + 1
        error_codes: list[SubmitDraftErrorCode] = []
        submission_diagnostics: dict[str, object] = {}
        invalid_diagnostics: dict[str, object] = {}
        search_cost_observations = (
            len(self._raw_atomic_tool_calls) - self._tool_call_count_at_last_submission
        )
        if len(self._raw_atomic_tool_calls) <= self._tool_call_count_at_last_submission:
            error_codes.append(SubmitDraftErrorCode.NO_NEW_GROUNDED_OBSERVATION)
        parsed_anchor = payload.parsed_entity
        if not parsed_anchor:
            error_codes.append(SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED)
        else:
            self._locked_anchor_entity = dict(parsed_anchor)
        canonical_answer = payload.canonical_answer
        label_signature = canonical_json(canonical_answer, default=str)
        label_scalar_value_signature = _single_field_scalar_value_signature(canonical_answer)
        label_slot_count = _answer_slot_count(canonical_answer)
        if (
            payload.answer_contract.kind == "list"
            and isinstance(canonical_answer, list)
            and len(canonical_answer) < 3
        ):
            error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_LIST_SIZE_INVALID)
            invalid_diagnostics["submitted_list_row_count"] = len(canonical_answer)
        elif (
            payload.answer_contract.kind == "list"
            and isinstance(canonical_answer, list)
            and len(canonical_answer) > 5
        ):
            error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_LIST_LIMIT_TOO_WIDE)
            invalid_diagnostics["submitted_list_row_count"] = len(canonical_answer)
        blank_paths = _blank_string_paths(canonical_answer)
        if blank_paths:
            error_codes.append(SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN)
            invalid_diagnostics["blank_string_paths"] = blank_paths[
                : self.config.synthesis.runtime.diagnostic_item_limit
            ]
        null_fields = _null_answer_fields(canonical_answer)
        if null_fields:
            error_codes.append(SubmitDraftErrorCode.LABEL_NULL_VALUE_FORBIDDEN)
            invalid_diagnostics["null_answer_fields"] = null_fields[
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
                # Diagnostic only: anchor connectivity is inferred from
                # value flow and is not precise enough for hard validation.
                invalid_diagnostics["disconnected_strings"] = disconnected[
                    : self.config.synthesis.runtime.diagnostic_item_limit
                ]
        contract_phrase_errors = _contract_component_phrase_errors(
            payload.answer_contract,
            user_request=payload.user_request,
        )
        if contract_phrase_errors:
            error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_PHRASE_MISSING)
            invalid_diagnostics["answer_contract_missing_phrases"] = (
                contract_phrase_errors[
                    : self.config.synthesis.runtime.diagnostic_item_limit
                ]
            )

        latest_query_call = self._latest_successful_query_call_since_last_submission()
        latest_query_result = (
            latest_query_call.get("result") if latest_query_call is not None else None
        )
        latest_query_params = (
            latest_query_call.get("params") if latest_query_call is not None else None
        )
        query_limit = _query_limit_from_params(latest_query_params)
        current_query_evidence_signature: QueryEvidenceSignature | None = None
        if not isinstance(latest_query_result, dict):
            error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISSING)
        else:
            submission_diagnostics["answer_contract_binding_diagnostics"] = (
                _answer_contract_binding_diagnostics(
                    payload.answer_contract,
                    user_request=payload.user_request,
                    canonical_answer=canonical_answer,
                    query_result=latest_query_result,
                    query_limit=query_limit,
                    item_limit=self.config.synthesis.runtime.diagnostic_item_limit,
                )
            )
            current_query_evidence_signature = _query_evidence_signature(
                latest_query_result,
                answer_kind=payload.answer_contract.kind,
                query_params=(
                    latest_query_params if isinstance(latest_query_params, dict) else None
                ),
            )
            if not _canonical_label_matches_query_result(
                label=canonical_answer,
                query_result=latest_query_result,
                answer_kind=payload.answer_contract.kind,
            ):
                error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISMATCH)
                invalid_diagnostics["latest_query_answer_preview"] = _preview_runtime_payload(
                    _canonical_from_latest_query_result(
                        latest_query_result,
                        answer_kind=payload.answer_contract.kind,
                    ),
                    config=self.config,
                )
                invalid_diagnostics["label_preview"] = _preview_runtime_payload(
                    canonical_answer,
                    config=self.config,
                )
            visibility_error_codes, visibility_diagnostics = _query_visibility_errors(
                latest_query_result
            )
            error_codes.extend(visibility_error_codes)
            invalid_diagnostics.update(visibility_diagnostics)
            if parsed_anchor:
                unanchored_hidden_filters = _unanchored_hidden_filter_sources(
                    latest_query_result,
                    anchor_entity=parsed_anchor,
                )
                unanchored_hidden_relays = _hidden_scope_relay_sources(
                    latest_query_params if isinstance(latest_query_params, dict) else None,
                    latest_query_result,
                    anchor_entity=parsed_anchor,
                )
                unanchored_hidden_filters.extend(unanchored_hidden_relays)
                if unanchored_hidden_filters:
                    error_codes.append(
                        SubmitDraftErrorCode.ANSWER_CONTRACT_HIDDEN_FILTER_UNANCHORED
                    )
                    invalid_diagnostics["unanchored_hidden_filters"] = (
                        unanchored_hidden_filters[
                            : self.config.synthesis.runtime.diagnostic_item_limit
                        ]
                    )
            unbound_user_visible_filters = _unbound_user_visible_filters(
                latest_query_result,
                contract=payload.answer_contract,
            )
            if unbound_user_visible_filters:
                error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_FILTER_UNBOUND)
                invalid_diagnostics["unbound_user_visible_filters"] = (
                    unbound_user_visible_filters[
                        : self.config.synthesis.runtime.diagnostic_item_limit
                    ]
                )
            if (
                payload.answer_contract.kind == "list"
                and payload.answer_contract.limit_phrase is None
                and _query_limit_shapes_membership(
                    latest_query_result,
                    query_limit=query_limit,
                )
            ):
                error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_QUERY_MISMATCH)
                invalid_diagnostics["missing_limit_phrase_for_query_limit"] = query_limit
            if (
                payload.answer_contract.kind == "list"
                and _query_ordering_is_ambiguous(latest_query_result)
            ):
                error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_ORDER_AMBIGUOUS)
                invalid_diagnostics["ordering_diagnostics"] = latest_query_result.get(
                    "ordering_diagnostics"
                )
            if (
                payload.answer_contract.kind == "list"
                and _query_order_is_too_complex(latest_query_result)
            ):
                error_codes.append(
                    SubmitDraftErrorCode.ANSWER_CONTRACT_ORDER_TOO_COMPLEX
                )
                invalid_diagnostics["order_reference_count"] = (
                    _query_order_reference_count(latest_query_result)
                )
                invalid_diagnostics["max_order_reference_count"] = 2
            if (
                payload.answer_contract.kind == "list"
                and _query_projection_has_duplicate_answer_rows(latest_query_result)
            ):
                error_codes.append(
                    SubmitDraftErrorCode.ANSWER_CONTRACT_DUPLICATE_ANSWER_ROWS
                )
                invalid_diagnostics["projection_diagnostics"] = latest_query_result.get(
                    "projection_diagnostics"
                )

        # After a too-easy rejection, require an answer change that is
        # visible to exact label verification.
        if (
            self._needs_label_change
            and self._last_label_signature is not None
            and label_signature == self._last_label_signature
        ):
            error_codes.append(SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED)
        elif (
            self._needs_label_change
            and self._last_label_scalar_value_signature is not None
            and label_scalar_value_signature is not None
            and label_scalar_value_signature == self._last_label_scalar_value_signature
            and self._last_query_evidence_signature is not None
            and current_query_evidence_signature is not None
            and current_query_evidence_signature.kind
            == self._last_query_evidence_signature.kind
            and current_query_evidence_signature.output_sources
            == self._last_query_evidence_signature.output_sources
        ):
            error_codes.append(SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED)
            invalid_diagnostics["unchanged_scalar_label_value"] = True
        if (
            self._needs_label_change
            and self._last_query_evidence_signature is not None
            and current_query_evidence_signature is not None
        ):
            incremental_errors = _query_evidence_incremental_errors(
                previous=self._last_query_evidence_signature,
                current=current_query_evidence_signature,
            )
            if incremental_errors:
                error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_NOT_INCREMENTAL)
                invalid_diagnostics["answer_contract_incremental_errors"] = (
                    incremental_errors[
                        : self.config.synthesis.runtime.diagnostic_item_limit
                    ]
                )
        if (
            not self._needs_label_change
            and self._repair_locked_label_signature is not None
            and label_signature != self._repair_locked_label_signature
        ):
            error_codes.append(SubmitDraftErrorCode.LABEL_CHANGED_DURING_REPAIR)
            invalid_diagnostics["repair_locked_label_changed"] = True
        if (
            not self._needs_label_change
            and self._limit_repair_scope is not None
            and current_query_evidence_signature is not None
        ):
            limit_repair_errors = _query_evidence_limit_repair_errors(
                previous=self._limit_repair_scope,
                current=current_query_evidence_signature,
            )
            if limit_repair_errors:
                error_codes.append(SubmitDraftErrorCode.LABEL_CHANGED_DURING_REPAIR)
                invalid_diagnostics["limit_repair_scope_errors"] = (
                    limit_repair_errors[
                        : self.config.synthesis.runtime.diagnostic_item_limit
                    ]
                )

        if submission_diagnostics:
            binding_diagnostics = submission_diagnostics.get(
                "answer_contract_binding_diagnostics"
            )
            if isinstance(binding_diagnostics, dict):
                binding_errors = _answer_contract_binding_errors(
                    binding_diagnostics,
                    require_output_bindings=payload.answer_contract.kind == "list",
                )
                binding_can_join_existing_feedback = any(
                    error_code
                    in {
                        SubmitDraftErrorCode.ANSWER_CONTRACT_PHRASE_MISSING,
                        SubmitDraftErrorCode.ANSWER_CONTRACT_ORDER_AMBIGUOUS,
                    }
                    for error_code in error_codes
                )
                if binding_errors and (
                    not error_codes
                    or (
                        binding_can_join_existing_feedback
                        and any(
                            error_name in _ORDER_BINDING_ERROR_NAMES
                            for error_name in binding_errors
                        )
                    )
                ):
                    if (
                        SubmitDraftErrorCode.ANSWER_CONTRACT_BINDING_MISSING
                        not in error_codes
                    ):
                        error_codes.append(
                            SubmitDraftErrorCode.ANSWER_CONTRACT_BINDING_MISSING
                        )
                    invalid_diagnostics["answer_contract_binding_errors"] = (
                        binding_errors[
                            : self.config.synthesis.runtime.diagnostic_item_limit
                        ]
                    )

        if submission_diagnostics:
            invalid_diagnostics = {
                **submission_diagnostics,
                **invalid_diagnostics,
            }

        if error_codes:
            deduped_error_codes = list(dict.fromkeys(error_codes))
            if all(error_code in _FEEDBACK_ONLY_ERROR_CODES for error_code in deduped_error_codes):
                if (
                    deduped_error_codes
                    == [SubmitDraftErrorCode.ANSWER_CONTRACT_LIST_LIMIT_TOO_WIDE]
                    and current_query_evidence_signature is not None
                    and isinstance(canonical_answer, list)
                ):
                    self._limit_repair_scope = LimitRepairScope(
                        evidence=current_query_evidence_signature,
                        max_item_count=len(canonical_answer),
                    )
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
                diagnostics=submission_diagnostics,
            )

        self._limit_repair_scope = None
        self._repair_locked_label_signature = None
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
        self._last_label_scalar_value_signature = label_scalar_value_signature
        self._last_label_slot_count = label_slot_count
        self._last_query_evidence_signature = current_query_evidence_signature
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
                    ci_lower=quality_gate_summary.ci_lower,
                    ci_upper=quality_gate_summary.ci_upper,
                    matched_solver_runs=quality_gate_summary.matched_solver_runs,
                    planned_solver_runs=quality_gate_summary.planned_solver_runs,
                    total_solver_runs=quality_gate_summary.total_solver_runs,
                    evaluable_solver_runs=quality_gate_summary.evaluable_solver_runs,
                    failed_solver_runs=quality_gate_summary.failed_solver_runs,
                )
            )
            self._emit_monitor(
                status="accepted",
                payload=payload,
                pass_rate=quality_gate_summary.pass_rate,
                matched_solver_runs=quality_gate_summary.matched_solver_runs,
                planned_solver_runs=quality_gate_summary.planned_solver_runs,
                total_solver_runs=quality_gate_summary.total_solver_runs,
                evaluable_solver_runs=quality_gate_summary.evaluable_solver_runs,
                failed_solver_runs=quality_gate_summary.failed_solver_runs,
                search_cost_observations=search_cost_observations,
                diagnostics=submission_diagnostics,
            )
            return _render_structured_message(
                kind="Accepted",
                result="Draft accepted.",
            )

        attempts_left_after = self.submissions_left() - 1
        if quality_gate_summary.status is TaskQualityGateStatus.REJECT_TOO_EASY:
            self._needs_label_change = True
            self._last_answer_contract = payload.answer_contract
            strengthening_guidance = _too_easy_retry_guidance(
                answer_kind=payload.answer_contract.kind
            )
            return self._record_rejection(
                submission_index=submission_index,
                message=_render_structured_message(
                    kind="RejectedError",
                    result="Draft needs more specificity.",
                    primary=(
                        "The current draft is too direct. Make it more "
                        "specific by changing the canonical label and matching "
                        "user_request/answer_contract, not just the wording."
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
                ci_lower=quality_gate_summary.ci_lower,
                ci_upper=quality_gate_summary.ci_upper,
                matched_solver_runs=quality_gate_summary.matched_solver_runs,
                planned_solver_runs=quality_gate_summary.planned_solver_runs,
                total_solver_runs=quality_gate_summary.total_solver_runs,
                evaluable_solver_runs=quality_gate_summary.evaluable_solver_runs,
                failed_solver_runs=quality_gate_summary.failed_solver_runs,
                diagnostics=submission_diagnostics,
                payload=payload,
                search_cost_observations=search_cost_observations,
            )

        if quality_gate_summary.status is TaskQualityGateStatus.CALIBRATION_INCONCLUSIVE:
            if quality_gate_summary.pass_rate > quality_gate_summary.band_upper:
                self._needs_label_change = True
                self._last_answer_contract = payload.answer_contract
                strengthening_guidance = _too_easy_retry_guidance(
                    answer_kind=payload.answer_contract.kind
                )
                return self._record_rejection(
                    submission_index=submission_index,
                    message=_render_structured_message(
                        kind="RejectedError",
                        result="Draft needs more specificity.",
                        primary=(
                            "The current draft is still too direct to accept "
                            "confidently. Make one grounded specificity change to the "
                            "canonical label and matching user_request/answer_contract, "
                            "not just the wording."
                        ),
                        important=strengthening_guidance.strip(),
                        next_step=(
                            "Make at least one new tool call "
                            "for the new evidence, then "
                            "resubmit."
                        ),
                        attempts_left=max(0, attempts_left_after),
                    ),
                    error_codes=[SubmitDraftErrorCode.CALIBRATION_INCONCLUSIVE],
                    pass_rate=quality_gate_summary.pass_rate,
                    ci_lower=quality_gate_summary.ci_lower,
                    ci_upper=quality_gate_summary.ci_upper,
                    matched_solver_runs=quality_gate_summary.matched_solver_runs,
                    planned_solver_runs=quality_gate_summary.planned_solver_runs,
                    total_solver_runs=quality_gate_summary.total_solver_runs,
                    evaluable_solver_runs=quality_gate_summary.evaluable_solver_runs,
                    failed_solver_runs=quality_gate_summary.failed_solver_runs,
                    diagnostics=submission_diagnostics,
                    payload=payload,
                    search_cost_observations=search_cost_observations,
                )

            self._terminated_too_hard = True
            return self._record_rejection(
                submission_index=submission_index,
                message=_render_structured_message(
                    kind="RejectedError",
                    result="Draft is not clearly reachable enough.",
                    primary=(
                        "The current draft is not clearly reachable enough. "
                        "This conversation is terminated. "
                        "Do not call submit_draft again."
                    ),
                    attempts_left=0,
                ),
                error_codes=[SubmitDraftErrorCode.CALIBRATION_INCONCLUSIVE],
                pass_rate=quality_gate_summary.pass_rate,
                ci_lower=quality_gate_summary.ci_lower,
                ci_upper=quality_gate_summary.ci_upper,
                matched_solver_runs=quality_gate_summary.matched_solver_runs,
                planned_solver_runs=quality_gate_summary.planned_solver_runs,
                total_solver_runs=quality_gate_summary.total_solver_runs,
                evaluable_solver_runs=quality_gate_summary.evaluable_solver_runs,
                failed_solver_runs=quality_gate_summary.failed_solver_runs,
                diagnostics={
                    **submission_diagnostics,
                    "terminal_rejection": True,
                },
                payload=payload,
                search_cost_observations=search_cost_observations,
            )

        # Overconstrained drafts are terminal — discard and let the outer
        # loop start fresh rather than oscillating.
        self._terminated_too_hard = True
        return self._record_rejection(
            submission_index=submission_index,
                message=_render_structured_message(
                    kind="RejectedError",
                    result="Draft is overconstrained.",
                    primary=(
                        "The current draft is not reachable enough. "
                        "This conversation is terminated. "
                        "Do not call submit_draft again."
                    ),
                attempts_left=0,
            ),
            error_codes=[SubmitDraftErrorCode.REJECT_TOO_HARD],
            pass_rate=quality_gate_summary.pass_rate,
            ci_lower=quality_gate_summary.ci_lower,
            ci_upper=quality_gate_summary.ci_upper,
            matched_solver_runs=quality_gate_summary.matched_solver_runs,
            planned_solver_runs=quality_gate_summary.planned_solver_runs,
            total_solver_runs=quality_gate_summary.total_solver_runs,
            evaluable_solver_runs=quality_gate_summary.evaluable_solver_runs,
            failed_solver_runs=quality_gate_summary.failed_solver_runs,
            diagnostics={
                **submission_diagnostics,
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
                "Rejected. Policy reminder: Label Grounding Policy requires "
                "real database facts observed through data tools before "
                "resubmission."
            ),
            SubmitDraftErrorCode.TOPIC_REQUIRED: (
                "Rejected. Tool schema reminder: topic is required."
            ),
            SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED: (
                "Rejected. Tool schema reminder: entity must contain at least "
                "one primary-key value."
            ),
            SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED: (
                'Rejected. Tool schema reminder: entity is a flat JSON object mapping observed primary-key field names to scalar values, for example {"<pk_column>": 123} or {"<pk_part_1>": 7, "<pk_part_2>": 1}; it is not nested under entity_type, primary_key, or primary_keys.'  # noqa: E501
            ),
            SubmitDraftErrorCode.QUESTION_REQUIRED: (
                "Rejected. Tool schema reminder: user_request is required."
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_REQUIRED: (
                "Rejected. Request Contract reminder: user_request is only the natural request body; hidden entity context belongs in entity."  # noqa: E501
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_INVALID_JSON: (
                "Rejected. Tool schema reminder: the hidden <entity> block requires a valid flat JSON object with one or more primary-key fields."  # noqa: E501
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_MISMATCH: (
                "Rejected. Tool schema reminder: the JSON inside the <entity> "
                "block must exactly match entity."
            ),
            SubmitDraftErrorCode.QUESTION_BODY_REQUIRED: (
                "Rejected. Request Contract reminder: user_request needs a natural request body."
            ),
            SubmitDraftErrorCode.CANONICAL_ANSWER_JSON_INVALID: (
                "Rejected. Tool schema reminder: label must be a valid JSON string."
            ),
            SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN: (
                "Rejected. Label Grounding Policy reminder: the canonical answer contains blank string fields; answer fields require grounded, non-empty values observed in tool results."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_NULL_VALUE_FORBIDDEN: (
                "Rejected. Label Grounding Policy reminder: answer fields require grounded non-null evidence. Do not submit null answer values; remove that nullable output field from label/request, rerun the label query with informative non-null fields, or choose another scoped label."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED: (
                "Rejected. Label Grounding Policy reminder: some label values were not directly grounded in observed tool results, or were reformatted from observed values."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED: (
                "Rejected. Difficulty-Up Policy reminder: after specificity feedback, the canonical answer itself must change through a grounded strengthening step. Use the last evaluated too-easy label as the baseline; keep fields already added and add one new grounded field, relationship, or coherent constraint."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_CHANGED_DURING_REPAIR: (
                "Rejected. Feedback Handling Policy reminder: this feedback only requires contract repair, so restore the repair-locked canonical label/query target and change only the failing request/answer_contract wording or rerun that same label query. Do not keep the last failed modified label."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_REQUIRED: (
                "Rejected. Tool schema reminder: answer_contract is required."
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_JSON_INVALID: (
                "Rejected. Tool schema reminder: answer_contract is a valid JSON object with kind, answer_phrase, constraint_phrases, and limit_phrase; it is not query result JSON, SQL structure, or a malformed JSON string."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_PHRASE_MISSING: (
                "Rejected. Label Contract reminder: every answer_contract phrase must be an exact contiguous substring copied from user_request. Feedback Handling Policy reminder: preserve the natural user_request wording and prior label_json fields/values; rewrite the full sentence cleanly when adding missing natural phrases to user_request/answer_contract instead of deleting, renaming, or splicing label fields."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISSING: (
                "Rejected. Label Contract reminder: final query evidence must immediately precede submit_draft, and the canonical label is copied from the latest successful query result."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISMATCH: (
                "Rejected. Label Contract reminder: label must exactly match the latest successful query result; scalar uses the one aggregate row object, list uses the query rows array, and helper/context fields are not label fields unless requested. Do not run helper/profile/count queries after final label evidence; if that happened, rerun the exact label query immediately before submit_draft."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_QUERY_MISMATCH: (
                "Rejected. Label Contract reminder: the latest successful query must contain structural evidence for this answer; if a list query limit fixes membership, that fixed size must be bound in user_request and answer_contract.limit_phrase."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_ORDER_AMBIGUOUS: (
                "Rejected. List Determinism Policy reminder: the latest list query does not uniquely determine submitted order or limited row membership for exact verification. For feedback retries, preserve the current anchor and target; repair ordering with a natural visible tie-break before query.order_by, choose unique ordering, or return tied rows. If the tie-break is sequence/rank-like, request wording must name source record sequence instead of a generated display rank. Do not repair this with hidden handles or artificial id wording. Request Contract reminder: preserve fluent request wording; use ordinary target-language words, not malformed terms. If a repair needs long/mechanical field lists, choose another label instead of stacking tie-break fields."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_ORDER_TOO_COMPLEX: (
                "Rejected. List Determinism Policy reminder: list order may use at most one natural visible tie-break, so query.order_by must have no more than two keys total. If more order keys are needed, choose another label or return tied rows instead of building a long mechanical sort contract."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_DUPLICATE_ANSWER_ROWS: (
                "Rejected. List Determinism Policy reminder: the latest list query returns duplicate projected answer rows, so returned rows are not distinguishable through requested output fields. Preserve the list size; add one natural visible distinguishing field or aggregate, then rerun the label query and submit_draft. Request Contract reminder: if rows are still duplicate or the repair needs long/mechanical field lists, choose another label instead of stacking fields."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_LIST_SIZE_INVALID: (
                "Rejected. Task Shapes reminder: list labels must return 3-5 rows. A 1-2 row list is too direct; do not keep probing the same target after it still returns fewer than 3 rows. Choose another scoped list with 3-5 distinguishable rows, or use a scalar aggregate when the request asks for a summary."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_LIST_LIMIT_TOO_WIDE: (
                "Rejected. Task Shapes reminder: fixed list labels must stay at 3-5 rows. Do not use 6+ rows to add difficulty; keep the same target/query scope and resubmit a smaller natural 3-5 row limit. List Determinism Policy reminder: when adding the smaller limit, the limit phrase must select the same row boundary as query.order_by; rerun the query with matching order if needed, and do not mix one hidden selection order with another display order."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_FILTER_UNBOUND: (
                "Rejected. Request Contract reminder: user-visible row-set filters need a dedicated constraint phrase in user_request/answer_contract; output field wording is not enough. If that filter is not intended, rerun the label query without it."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_HIDDEN_FILTER_UNANCHORED: (
                "Rejected. Request Contract reminder: hidden row-scope handles used by the latest query must be anchored in entity, not only hidden inside query filters. If the latest query relays from a child/current record through a parent to sibling answer rows, rewording as child-related is not enough: either put the parent/current-subject handle in entity and rerun the label query from that scope, or choose a label directly scoped to the existing entity."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_VISIBILITY_EVIDENCE_MISSING: (
                "Rejected. Label Contract reminder: latest query evidence must include field visibility evidence before submit_draft."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_BINDING_MISSING: (
                "Rejected. Label Contract reminder: for list labels, answer_contract.output_bindings cover every returned label field, and answer_contract.order_bindings cover each query.order_by entry in order using phrases copied from user_request. If an order key is only a tie-break, user_request still needs natural visible tie-break wording before that key can be bound; otherwise rerun query without that order key or return tied rows."  # noqa: E501
                " Each returned output field also needs its own natural role phrase; do not reuse one broad output phrase for multiple returned concepts. Order binding phrases need direction/recency/tie-break wording, not only the bare output noun; Display-only output wording is not enough. Do not reuse one broad order phrase for multiple different order keys."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_NON_USER_VISIBLE_SOURCE: (
                "Rejected. Label Contract reminder: the submitted label directly exposes a field marked internal or blocked in latest query metadata."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_NO_PRIMARY_KEY_SOURCE: (
                "Rejected. Source Surface Policy reminder: the latest query exposes label values from a table without a primary key, so those rows cannot be revisited as stable records. Choose a primary-key-backed path for row values, or use a derived aggregate over the no-primary-key table; do not resubmit the same row-value label from that surface."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_NOT_INCREMENTAL: (
                "Rejected. Difficulty-Up Policy reminder: this retry changed the prior answer kind, query shape, row set, or output source meanings instead of preserving the evaluated task and adding one grounded strengthening."  # noqa: E501
                " The baseline is the last solver-evaluated draft, not later failed detours. Restore that evaluated task's target scope, predicates, row set, and output sources, then append one grounded dimension that changes lookup, comparison, order, or row reasoning. Scalar aggregate retries may become grouped aggregate lists only when they keep the same target and predicates. For list retries, keep every prior output field/source and prior order binding, including fields already added by earlier too-easy retries. Output-only list field additions and same-row passive display/derived fields are still too direct."  # noqa: E501
            ),
            SubmitDraftErrorCode.SUBMIT_PAYLOAD_INVALID: (
                "Rejected. Tool schema reminder: submit_draft arguments did "
                "not match the required schema."
            ),
            SubmitDraftErrorCode.DRAFT_VALIDATION_FAILED: (
                "Rejected. Tool schema reminder: the submitted draft could not "
                "be validated."
            ),
        }
        primary = message_map.get(error_codes[0], "Rejected. Fix the draft and resubmit.")
        if error_codes and error_codes[0] is SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED:
            if (
                diagnostics is not None
                and diagnostics.get("anchor_path_has_readable_strings") is False
            ):
                primary = (
                    "Rejected. Label Grounding Policy reminder: the current anchored evidence path does not expose readable text fields in real tool outputs."  # noqa: E501
                )
            else:
                primary += _format_ungrounded_value_guidance(diagnostics)
        if error_codes and error_codes[0] is SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED:
            primary += _too_easy_retry_guidance(
                answer_kind=(
                    self._last_answer_contract.kind
                    if self._last_answer_contract is not None
                    else None
                )
            )
        if (
            error_codes
            and error_codes[0] is SubmitDraftErrorCode.ANSWER_CONTRACT_PHRASE_MISSING
        ):
            primary += _format_missing_request_phrase_guidance(diagnostics)
        preserve_guidance = ""
        if self._last_monitored_label_data is not None:
            preserve_guidance = (
                "Policy reminder: Feedback Handling Policy preserves anchor/language and changes the smallest failing part; preserve target only when the named policy does not require another scoped label."  # noqa: E501
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
                    "Use the referenced policy/tool contract as the repair source; "
                    "call data tools if evidence is missing, then call submit_draft again. "
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
        error_codes: list[SubmitDraftErrorCode | str],
        payload: SubmitDraftPayload | dict[str, object] | None = None,
        search_cost_observations: int | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> str:
        self._feedback_events += 1
        error_code_values = tuple(_error_code_values(error_codes))
        self._last_feedback_error_codes = error_code_values
        if self._feedback_codes_lock_label(error_codes) and isinstance(
            payload,
            SubmitDraftPayload,
        ):
            self._repair_locked_label_signature = canonical_json(
                payload.canonical_answer,
                default=str,
            )
        elif (
            SubmitDraftErrorCode.LABEL_CHANGED_DURING_REPAIR.value
            not in error_code_values
        ):
            self._repair_locked_label_signature = None
        if error_code_values != (
            SubmitDraftErrorCode.ANSWER_CONTRACT_LIST_LIMIT_TOO_WIDE.value,
        ):
            self._limit_repair_scope = None
        self._tool_call_count_at_last_protocol_boundary = len(self._raw_atomic_tool_calls)
        attempts_left_after = self.submissions_left()
        self._emit_monitor(
            status="budget_exhausted" if attempts_left_after <= 0 else "feedback",
            payload=payload,
            pass_rate=None,
            matched_solver_runs=None,
            planned_solver_runs=None,
            total_solver_runs=None,
            evaluable_solver_runs=None,
            failed_solver_runs=None,
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
        ci_lower: float | None = None,
        ci_upper: float | None = None,
        matched_solver_runs: int | None = None,
        planned_solver_runs: int | None = None,
        total_solver_runs: int | None = None,
        evaluable_solver_runs: int | None = None,
        failed_solver_runs: int | None = None,
        search_cost_observations: int | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> str:
        attempts_left_after = self.submissions_left() - 1
        self._tool_call_count_at_last_protocol_boundary = len(self._raw_atomic_tool_calls)
        self.attempts.append(
            SubmitDraftAttemptRecord(
                index=submission_index,
                outcome=error_codes[0].value if error_codes else "rejected",
                message=message,
                error_codes=tuple(_error_code_values(error_codes)),
                pass_rate=pass_rate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                matched_solver_runs=matched_solver_runs,
                planned_solver_runs=planned_solver_runs,
                total_solver_runs=total_solver_runs,
                evaluable_solver_runs=evaluable_solver_runs,
                failed_solver_runs=failed_solver_runs,
            )
        )
        status = "budget_exhausted" if attempts_left_after <= 0 else "rejected"
        self._emit_monitor(
            status=status,
            payload=payload,
            pass_rate=pass_rate,
            matched_solver_runs=matched_solver_runs,
            planned_solver_runs=planned_solver_runs,
            total_solver_runs=total_solver_runs,
            evaluable_solver_runs=evaluable_solver_runs,
            failed_solver_runs=failed_solver_runs,
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
        planned_solver_runs: int | None,
        total_solver_runs: int | None,
        evaluable_solver_runs: int | None,
        failed_solver_runs: int | None,
        search_cost_observations: int | None,
        diagnostics: dict[str, object],
    ) -> None:
        if self.phase_monitor is None:
            return
        label_data = _monitor_label_data(payload, config=self.config)
        previous_label_data = self._last_monitored_label_data
        if self._needs_label_change and self._last_evaluated_label_data is not None:
            previous_label_data = self._last_evaluated_label_data
        label_change = _label_change_summary(
            previous=previous_label_data,
            current=label_data,
        )
        self.phase_monitor.emit(
            phase="submit_draft",
            status=status,
            expected_contract={
                "topic_experiment_hint": self.requested_topic,
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
                "user_request": (
                    payload.user_request
                    if isinstance(payload, SubmitDraftPayload)
                    else payload.get("user_request", payload.get("question"))
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
                "answer_contract": (
                    payload.answer_contract.model_dump(mode="json")
                    if isinstance(payload, SubmitDraftPayload)
                    else payload.get("answer_contract")
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
                "planned_solver_runs": planned_solver_runs,
                "total_solver_runs": total_solver_runs,
                "completed_solver_runs": total_solver_runs,
                "evaluable_solver_runs": evaluable_solver_runs,
                "failed_solver_runs": failed_solver_runs,
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
        if pass_rate is not None:
            self._last_evaluated_label_data = label_data
        self._last_monitored_label_data = label_data


def build_submit_draft_sdk_tool(controller: SubmitDraftController) -> object:
    from agents import FunctionTool
    from agents.strict_schema import ensure_strict_json_schema

    params_json_schema = ensure_strict_json_schema(
        SubmitDraftPayload.model_json_schema()
    )

    async def _invoke_tool(_tool_context: Any, input_json: str) -> str:
        try:
            parsed_raw: object = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError as exc:
            return controller.reject_malformed_tool_input(
                parsed={
                    "tool_input_preview": _preview_runtime_payload(
                        input_json,
                        config=controller.config,
                    )
                },
                validation_errors=[
                    {
                        "loc": ["tool_input"],
                        "type": "json_decode_error",
                        "message": str(exc),
                    }
                ],
            )
        if not isinstance(parsed_raw, dict):
            return controller.reject_malformed_tool_input(
                parsed={
                    "tool_input_type": type(parsed_raw).__name__,
                    "tool_input_preview": _preview_runtime_payload(
                        parsed_raw,
                        config=controller.config,
                    ),
                },
                validation_errors=[
                    {
                        "loc": ["tool_input"],
                        "type": "object_type",
                        "message": "submit_draft input must be a JSON object",
                    }
                ],
            )
        parsed = {str(key): value for key, value in parsed_raw.items()}
        raw_question = parsed.pop("question", None)
        if isinstance(raw_question, str) and "user_request" not in parsed:
            parsed_anchor_entity, question_body, prompt_error = (
                _split_entity_wrapped_prompt(raw_question)
            )
            if prompt_error is None and question_body is not None:
                parsed["user_request"] = question_body
                if "entity" not in parsed and parsed_anchor_entity is not None:
                    parsed["entity"] = parsed_anchor_entity
            else:
                parsed["user_request"] = raw_question
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
            "Submit one grounded task draft for validation. Call only after "
            "a successful query produced the exact label values."
        ),
        params_json_schema=params_json_schema,
        on_invoke_tool=_invoke_tool,
        strict_json_schema=True,
    )
