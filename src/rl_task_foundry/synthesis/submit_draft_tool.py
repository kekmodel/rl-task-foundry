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
    LABEL_VALUES_NOT_GROUNDED = "label_values_not_grounded"
    LABEL_NOT_STRENGTHENED = "label_not_strengthened"
    ANSWER_CONTRACT_REQUIRED = "answer_contract_required"
    ANSWER_CONTRACT_JSON_INVALID = "answer_contract_json_invalid"
    ANSWER_CONTRACT_PHRASE_MISSING = "answer_contract_phrase_missing"
    ANSWER_CONTRACT_EVIDENCE_MISSING = "answer_contract_evidence_missing"
    ANSWER_CONTRACT_EVIDENCE_MISMATCH = "answer_contract_evidence_mismatch"
    ANSWER_CONTRACT_QUERY_MISMATCH = "answer_contract_query_mismatch"
    ANSWER_CONTRACT_ORDER_AMBIGUOUS = "answer_contract_order_ambiguous"
    ANSWER_CONTRACT_HIDDEN_FILTER_UNANCHORED = (
        "answer_contract_hidden_filter_unanchored"
    )
    ANSWER_CONTRACT_VISIBILITY_EVIDENCE_MISSING = (
        "answer_contract_visibility_evidence_missing"
    )
    LABEL_NON_USER_VISIBLE_SOURCE = "label_non_user_visible_source"
    ANSWER_CONTRACT_NOT_INCREMENTAL = "answer_contract_not_incremental"
    SUBMIT_PAYLOAD_INVALID = "submit_payload_invalid"
    DRAFT_VALIDATION_FAILED = "draft_validation_failed"
    REJECT_TOO_EASY = "reject_too_easy"
    REJECT_TOO_HARD = "reject_too_hard"
    CALIBRATION_INCONCLUSIVE = "calibration_inconclusive"


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
        SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED,
        SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED,
        SubmitDraftErrorCode.ANSWER_CONTRACT_PHRASE_MISSING,
        SubmitDraftErrorCode.ANSWER_CONTRACT_JSON_INVALID,
        SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISSING,
        SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISMATCH,
        SubmitDraftErrorCode.ANSWER_CONTRACT_QUERY_MISMATCH,
        SubmitDraftErrorCode.ANSWER_CONTRACT_ORDER_AMBIGUOUS,
        SubmitDraftErrorCode.ANSWER_CONTRACT_HIDDEN_FILTER_UNANCHORED,
        SubmitDraftErrorCode.ANSWER_CONTRACT_VISIBILITY_EVIDENCE_MISSING,
        SubmitDraftErrorCode.LABEL_NON_USER_VISIBLE_SOURCE,
        SubmitDraftErrorCode.ANSWER_CONTRACT_NOT_INCREMENTAL,
    }
)

JsonScalar = str | int | float | bool


def _error_code_values(
    codes: list[SubmitDraftErrorCode] | tuple[SubmitDraftErrorCode, ...],
) -> list[str]:
    return [code.value for code in codes]


def _validation_error_diagnostics(error: ValidationError) -> list[dict[str, object]]:
    return [
        {
            "loc": [str(part) for part in error_item.get("loc", ())],
            "type": str(error_item.get("type", "")),
            "message": str(error_item.get("msg", "")),
        }
        for error_item in error.errors()
    ]


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
            "target. Structural evidence is derived from the latest query."
        ),
    )
    limit_phrase: str | None = Field(
        description=(
            "Exact user_request substring for a fixed requested list size, "
            "such as '3 items', or null when there is no fixed size phrase."
        ),
    )

    @field_validator("answer_phrase")
    @classmethod
    def _validate_answer_phrase(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("answer_phrase must not be blank")
        return normalized

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
            "context, either directly or through observed values derived from it."
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
            "only when it appeared in tool evidence. Use 'my'/'own' wording "
            "only when the hidden context naturally represents the requester "
            "or their records and the latest query is scoped to that context."
        ),
    )
    answer_contract: AnswerContract = Field(
        description=(
            "Minimal request-binding contract. Provide the answer shape and "
            "exact user_request phrases for the answer target, entity scope, "
            "filters, ordering, tie-breaks, or fixed list size. Do not restate "
            "tables, columns, operators, or SQL; the latest successful query "
            "supplies structural evidence."
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
    for path, phrase in components:
        if phrase is not None and not _phrase_is_in_request(
            phrase=phrase,
            user_request=user_request,
        ):
            missing_phrases.append(path)
    return missing_phrases


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
    predicates: tuple[str, ...]
    order_by: tuple[str, ...]
    item_count: int | None


def _query_evidence_signature(
    query_result: dict[str, object],
    *,
    answer_kind: str,
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
    )


def _query_evidence_incremental_errors(
    *,
    previous: QueryEvidenceSignature,
    current: QueryEvidenceSignature,
) -> list[str]:
    errors: list[str] = []
    if current.kind != previous.kind:
        errors.append("kind_changed")
    previous_outputs = set(previous.output_sources)
    current_outputs = set(current.output_sources)
    if not previous_outputs.issubset(current_outputs):
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

    if not (
        added_predicate
        or added_order
        or added_output_source
        or strengthened_cardinality
    ):
        errors.append("no_new_structural_constraint")
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


def _query_ordering_is_ambiguous(query_result: dict[str, object]) -> bool:
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
    for source in column_sources:
        if source.get("value_exposes_source") is not True:
            continue
        if blocks_direct_label_exposure(source.get("visibility")):
            label_sources.append(
                {
                    key: source.get(key)
                    for key in (
                        "output",
                        "kind",
                        "table",
                        "column",
                        "visibility",
                    )
                    if key in source
                }
            )

    codes: list[SubmitDraftErrorCode] = []
    diagnostics: dict[str, object] = {}
    if label_sources:
        codes.append(SubmitDraftErrorCode.LABEL_NON_USER_VISIBLE_SOURCE)
        diagnostics["non_user_visible_label_sources"] = label_sources
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

    def _latest_successful_query_result_since_last_submission(self) -> dict[str, object] | None:
        call = self._latest_successful_query_call_since_last_submission()
        result = call.get("result") if call is not None else None
        return result if isinstance(result, dict) else None

    def _latest_successful_query_call_since_last_submission(self) -> dict[str, object] | None:
        recent_calls = self._raw_atomic_tool_calls[self._tool_call_count_at_last_submission :]
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
        current_query_evidence_signature: QueryEvidenceSignature | None = None
        if not isinstance(latest_query_result, dict):
            error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISSING)
        else:
            current_query_evidence_signature = _query_evidence_signature(
                latest_query_result,
                answer_kind=payload.answer_contract.kind,
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
                if unanchored_hidden_filters:
                    error_codes.append(
                        SubmitDraftErrorCode.ANSWER_CONTRACT_HIDDEN_FILTER_UNANCHORED
                    )
                    invalid_diagnostics["unanchored_hidden_filters"] = (
                        unanchored_hidden_filters[
                            : self.config.synthesis.runtime.diagnostic_item_limit
                        ]
                    )
            query_limit = _query_limit_from_params(
                latest_query_call.get("params") if latest_query_call is not None else None
            )
            latest_rows = latest_query_result.get("rows")
            if (
                payload.answer_contract.kind == "list"
                and payload.answer_contract.limit_phrase is None
                and query_limit is not None
                and isinstance(latest_rows, list)
                and len(latest_rows) == query_limit
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
                diagnostics={},
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
                diagnostics={},
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
                    diagnostics={},
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
                "Rejected. Observe more real database facts with data tools before resubmitting."
            ),
            SubmitDraftErrorCode.TOPIC_REQUIRED: "Rejected. topic is required.",
            SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED: "Rejected. entity must contain at least one primary-key value.",  # noqa: E501
            SubmitDraftErrorCode.ANCHOR_ENTITY_SCALAR_MAP_REQUIRED: (
                'Rejected. entity must be a flat JSON object mapping observed primary-key field names to scalar values, for example {"<pk_column>": 123} or {"<pk_part_1>": 7, "<pk_part_2>": 1}. Use field names from the observed table metadata. Do not nest it under entity_type, primary_key, or primary_keys.'  # noqa: E501
            ),
            SubmitDraftErrorCode.QUESTION_REQUIRED: "Rejected. user_request is required.",
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_REQUIRED: (
                "Rejected. user_request should contain only the natural user request body. Do not include the hidden entity block."  # noqa: E501
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_INVALID_JSON: (
                "Rejected. The <entity> block must contain a valid flat JSON object with one or more primary-key fields."  # noqa: E501
            ),
            SubmitDraftErrorCode.QUESTION_ENTITY_BLOCK_MISMATCH: (
                "Rejected. The JSON inside the <entity> block must exactly match entity."
            ),
            SubmitDraftErrorCode.QUESTION_BODY_REQUIRED: (
                "Rejected. Include a natural user request body in user_request."
            ),
            SubmitDraftErrorCode.CANONICAL_ANSWER_JSON_INVALID: (
                "Rejected. label must be a valid JSON string."
            ),
            SubmitDraftErrorCode.LABEL_BLANK_STRING_FORBIDDEN: (
                "Rejected. The canonical answer contains blank string fields. Every answer field must contain a grounded, non-empty value. Schema orientation alone is not enough; only fields you actually observed in tool results are grounded. If the chosen surface is id-only, keep the same anchored user and switch to grounded counts, dates, amounts, statuses, ordering, or make new anchored tool calls until you observe readable fields."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_VALUES_NOT_GROUNDED: (
                "Rejected. Some label values were not directly grounded in the observed tool results. Schema orientation alone is not enough; only use business strings, dates, and other readable values that you actually observed in real tool outputs, and copy them exactly as they appeared there. Do not shorten names, paraphrase labels, normalize timestamp formatting, or manufacture readable labels by wrapping an id in generic words such as 'record 17' or 'item 2'. If the chosen surface is id-only, keep the same anchored user and switch to counts, dates, amounts, statuses, ordering, make new anchored tool calls until you observe readable fields, or choose a better grounded topic for the same anchored user need."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED: (
                "Rejected. After a specificity rejection, do not resubmit the same label or the same single-field answer value under a new field name. Strengthen the canonical answer itself with a new grounded step whose submitted value changes."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_REQUIRED: (
                "Rejected. answer_contract is required."
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_JSON_INVALID: (
                "Rejected. answer_contract must be a valid JSON object with kind, answer_phrase, constraint_phrases, and limit_phrase. Do not paste query result JSON or SQL structure into answer_contract. Do not pass a malformed JSON string; pass a complete object whose brackets and list items are closed."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_PHRASE_MISSING: (
                "Rejected. Every answer_contract phrase must be an exact contiguous substring copied from user_request. Write the user_request wording first, then paste the exact same words into answer_phrase, constraint_phrases, and limit_phrase."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISSING: (
                "Rejected. Call query immediately before submit_draft; the canonical label must be copied from the latest successful query result."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISMATCH: (
                "Rejected. label must exactly match the latest successful query result. For kind='scalar', copy the one aggregate row as the label object. For kind='list', copy the query rows array as the label list, even when the query returned one row. If the latest query selected helper/context fields that the user did not ask to receive, rerun query with only the intended label fields instead of adding extras to label."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_QUERY_MISMATCH: (
                "Rejected. The latest successful query does not contain the required structural evidence for this answer. If a list query limit fixes the returned rows, user_request and answer_contract.limit_phrase must include that exact fixed size."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_ORDER_AMBIGUOUS: (
                "Rejected. The latest list query has ambiguous ordering for exact verification. State deterministic answer-visible query.order_by tie-breakers in user_request and answer_contract, choose a row set with unique visible ordering, or return the tied rows as the list before submit_draft. Do not rely on unseen order keys to break ties."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_HIDDEN_FILTER_UNANCHORED: (
                "Rejected. The latest query filters on a blocked handle value that is not present in entity. Put required hidden scope handles in entity or rerun the query using the submitted entity's handle."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_VISIBILITY_EVIDENCE_MISSING: (
                "Rejected. Call query again before submit_draft; the latest query result must include field visibility evidence."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_NON_USER_VISIBLE_SOURCE: (
                "Rejected. The label directly exposes a field that is explicitly marked internal or blocked in the latest query metadata. Keep internal/blocked source values out of the submitted label; use a user-visible output value or a derived aggregate that does not expose the source value."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_NOT_INCREMENTAL: (
                "Rejected. After a specificity rejection, keep the same answer kind, preserve prior filters/order and existing query output fields, then add a grounded filter, order, cardinality, or output field that the current database evidence supports."  # noqa: E501
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
            primary += _too_easy_retry_guidance(
                answer_kind=(
                    self._last_answer_contract.kind
                    if self._last_answer_contract is not None
                    else None
                )
            )
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
                    "Make another data-tool call if needed, then call submit_draft again. "
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
        self._last_feedback_error_codes = tuple(_error_code_values(error_codes))
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
