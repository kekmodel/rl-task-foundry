"""submit_draft tool for the single-agent synthesis loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import AliasChoices, Field, ValidationError, field_validator, model_validator

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
        SubmitDraftErrorCode.ANSWER_CONTRACT_VISIBILITY_EVIDENCE_MISSING,
        SubmitDraftErrorCode.LABEL_NON_USER_VISIBLE_SOURCE,
        SubmitDraftErrorCode.ANSWER_CONTRACT_NOT_INCREMENTAL,
    }
)

JsonScalar = str | int | float | bool
JsonLabelValue = str | int | float | bool | None


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


class AnswerOperationBase(StrictModel):
    table: str = Field(
        min_length=1,
        description="Table that supplies the answer rows or aggregate target.",
    )
    column: str | None = Field(
        default=None,
        description="Target column, or null only for count(*) or row-list selection.",
    )
    phrase: str = Field(
        min_length=1,
        description=(
            "Exact user_request substring that states the answer being asked for."
        ),
    )


class ScalarAnswerOperation(AnswerOperationBase):
    fn: Literal["count", "sum", "avg", "min", "max"] = Field(
        description=(
            "Aggregate function for kind=scalar. Why: scalar answers must be "
            "one aggregate value from the latest query."
        ),
    )


class ListAnswerOperation(AnswerOperationBase):
    fn: Literal["select"] = Field(
        description=(
            "Select operation for kind=list. Why: selected rows or lookup "
            "fields are list-shaped answers, even when one row is returned."
        ),
    )


AnswerOperation = ScalarAnswerOperation | ListAnswerOperation


class AnswerPredicate(StrictModel):
    table: str = Field(min_length=1, description="Table containing the filter column.")
    column: str = Field(min_length=1, description="Filter column used by latest query.")
    op: Literal[
        "eq",
        "neq",
        "in",
        "lt",
        "gt",
        "lte",
        "gte",
        "like",
        "is_null",
        "is_not_null",
    ] = Field(description="Predicate operator.")
    value: object | None = Field(
        default=None,
        description=(
            "Filter value copied from tool/query evidence. Omit or null only "
            "for null-check operators. Hidden PK/FK values may be used here, "
            "but raw hidden values must not appear in user_request."
        ),
    )
    phrase: str = Field(
        min_length=1,
        description=(
            "Exact user_request substring that states this filter in customer "
            "language. For hidden id filters, use wording like my account/my "
            "records or a visible name, not the raw id value."
        ),
    )

    @model_validator(mode="after")
    def _validate_value_shape(self) -> AnswerPredicate:
        if self.op in {"is_null", "is_not_null"}:
            return self
        if self.value is None:
            raise ValueError(f"value is required for predicate op={self.op!r}")
        return self


class AnswerOrderBy(StrictModel):
    table: str = Field(min_length=1, description="Table containing the sort column.")
    column: str = Field(min_length=1, description="Sort column.")
    direction: Literal["asc", "desc"] = Field(description="Sort direction.")
    phrase: str = Field(
        min_length=1,
        description=(
            "Exact user_request substring that states deterministic ordering."
        ),
    )


class AnswerContractBase(StrictModel):
    predicates: list[AnswerPredicate] = Field(
        description=(
            "All filters that define the answer subset and appear in the "
            "latest query evidence."
        ),
    )
    order_by: list[AnswerOrderBy] = Field(
        description="Ordering clauses from the latest query that make lists deterministic.",
    )
    evidence: Literal["latest_query"] = Field(
        description="Canonical label must be copied from the latest successful query call.",
    )


class ScalarAnswerContract(AnswerContractBase):
    kind: Literal["scalar"] = Field(
        description=(
            "Scalar answer shape: exactly one aggregate value "
            "(count/sum/avg/min/max)."
        ),
    )
    operation: ScalarAnswerOperation = Field(
        description=(
            "Aggregate answer target that matches the latest query result. "
            "Keep it fixed after a specificity rejection; strengthen with "
            "filters instead."
        )
    )
    limit: None = Field(
        description="Must be null for scalar answers.",
    )
    limit_phrase: None = Field(
        description="Must be null for scalar answers.",
    )


class ListAnswerContract(AnswerContractBase):
    kind: Literal["list"] = Field(
        description=(
            "List answer shape: selected rows or lookup fields copied from "
            "the latest query, even when one row is returned."
        ),
    )
    operation: ListAnswerOperation = Field(
        description=(
            "Select answer target that matches the latest query result. Keep "
            "it fixed after a specificity rejection; strengthen with filters, "
            "ordering, or limits instead."
        )
    )
    limit: int | None = Field(
        ge=1,
        description="Fixed list length, or null for all-matching list answers.",
    )
    limit_phrase: str | None = Field(
        description="Exact user_request substring that states the fixed list length.",
    )

    @model_validator(mode="after")
    def _validate_limit_phrase(self) -> ListAnswerContract:
        if self.limit is not None and not self.limit_phrase:
            raise ValueError("limit_phrase is required when limit is set")
        return self


AnswerContract = Annotated[
    ScalarAnswerContract | ListAnswerContract,
    Field(
        discriminator="kind",
        description=(
            "Machine-checkable answer contract. The kind field selects the "
            "valid schema: scalar requires aggregate fn; list requires select."
        ),
    ),
]


class SubmitDraftPayload(StrictModel):
    topic: str = Field(
        min_length=1,
        description="Selected topic string derived from the grounded label and evidence.",
    )
    label: dict[str, JsonLabelValue] | list[dict[str, JsonLabelValue]] = Field(
        description=(
            "Canonical submit_result payload copied exactly from the latest "
            "successful query result. For scalar, submit one object with the "
            "aggregate field. For list, submit an array of row objects. Do not "
            "expose hidden PK/FK handle values as answer values. The label "
            "must answer the exact scope of user_request; if the request is "
            "about the hidden entity's own records, the latest query must be "
            "scoped to that entity before you copy the result."
        ),
    )
    entity: dict[str, JsonScalar] = Field(
        description=(
            "Hidden grounding handle as an object, e.g. "
            '{"<pk_name>": 123}. It may contain observed primary-key values; '
            "those values should stay hidden from user_request."
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
            "only when label and the latest query are actually scoped to entity."
        ),
    )
    answer_contract: AnswerContract = Field(
        description=(
            "Machine-checkable meaning of the draft: answer target, every filter, "
            "ordering/cardinality, and the query evidence source. Include the "
            "entity scope when user_request depends on it, using a phrase that "
            "appears in user_request such as 'my' or 'this record'."
        ),
    )

    @field_validator("label", mode="before")
    @classmethod
    def _validate_label(
        cls,
        value: object,
    ) -> dict[str, JsonLabelValue] | list[dict[str, JsonLabelValue]]:
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
        if isinstance(value, list) and not all(isinstance(item, dict) for item in value):
            raise ValueError("label array items must be objects")
        return value

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

    @field_validator("entity", mode="before")
    @classmethod
    def _validate_entity(cls, value: object) -> dict[str, JsonScalar]:
        if isinstance(value, dict):
            return _normalize_anchor_entity_map(value)
        raw = str(value).strip() if not isinstance(value, str) else value.strip()
        if not raw:
            raise ValueError("entity must not be blank")
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raise ValueError("entity must be a valid JSON object")
        if not isinstance(parsed, dict):
            raise ValueError("entity must be a JSON object")
        return _normalize_anchor_entity_map(parsed)

    @property
    def parsed_entity(self) -> dict[str, JsonScalar]:
        return _normalize_anchor_entity_map(self.entity)

    @property
    def canonical_answer(self) -> object:
        return self.label

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
        ("operation", contract.operation.phrase),
        ("limit", contract.limit_phrase),
    ]
    components.extend(
        (f"predicate[{index}]", predicate.phrase)
        for index, predicate in enumerate(contract.predicates)
    )
    components.extend(
        (f"order_by[{index}]", order.phrase)
        for index, order in enumerate(contract.order_by)
    )
    for path, phrase in components:
        if phrase is not None and not _phrase_is_in_request(
            phrase=phrase,
            user_request=user_request,
        ):
            missing_phrases.append(path)
    return missing_phrases


def _operation_signature(operation: AnswerOperation) -> tuple[str, str, str | None]:
    return (operation.fn, operation.table, operation.column)


def _predicate_signature(predicate: AnswerPredicate) -> str:
    return canonical_json(
        {
            "table": predicate.table,
            "column": predicate.column,
            "op": predicate.op,
            "value": predicate.value,
        },
        default=str,
    )


def _order_signature(order: AnswerOrderBy) -> str:
    return canonical_json(
        {
            "table": order.table,
            "column": order.column,
            "direction": order.direction,
        },
        default=str,
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


def _query_contract_evidence_errors(
    *,
    contract: AnswerContract,
    query_result: dict[str, object],
) -> tuple[list[SubmitDraftErrorCode], dict[str, object]]:
    referenced_columns = _as_object_list(query_result.get("referenced_columns"))
    if referenced_columns is None:
        return [], {}

    query_predicates = {
        signature
        for ref in referenced_columns
        if (signature := _referenced_predicate_signature(ref)) is not None
    }
    query_orders = {
        signature
        for ref in referenced_columns
        if (signature := _referenced_order_signature(ref)) is not None
    }
    missing_predicates = [
        {
            "table": predicate.table,
            "column": predicate.column,
            "op": predicate.op,
            "value": predicate.value,
        }
        for predicate in contract.predicates
        if _predicate_signature(predicate) not in query_predicates
    ]
    missing_order_by = [
        {
            "table": order.table,
            "column": order.column,
            "direction": order.direction,
        }
        for order in contract.order_by
        if _order_signature(order) not in query_orders
    ]
    if not missing_predicates and not missing_order_by:
        return [], {}
    return (
        [SubmitDraftErrorCode.ANSWER_CONTRACT_QUERY_MISMATCH],
        {
            "missing_query_predicates": missing_predicates,
            "missing_query_order_by": missing_order_by,
        },
    )


def _answer_contract_incremental_errors(
    *,
    previous: AnswerContract,
    current: AnswerContract,
) -> list[str]:
    errors: list[str] = []
    if current.kind != previous.kind:
        errors.append("kind_changed")
    if _operation_signature(current.operation) != _operation_signature(previous.operation):
        errors.append("operation_changed")

    previous_predicates = {
        _predicate_signature(predicate) for predicate in previous.predicates
    }
    current_predicates = {
        _predicate_signature(predicate) for predicate in current.predicates
    }
    missing_predicates = sorted(previous_predicates - current_predicates)
    if missing_predicates:
        errors.append("predicate_removed")

    previous_order = {_order_signature(order) for order in previous.order_by}
    current_order = {_order_signature(order) for order in current.order_by}
    missing_order = sorted(previous_order - current_order)
    if missing_order:
        errors.append("order_removed")

    added_predicate = bool(current_predicates - previous_predicates)
    added_order = bool(current_order - previous_order)
    strengthened_limit = False
    if current.kind == "list":
        if previous.limit is None and current.limit is not None:
            strengthened_limit = True
        elif (
            previous.limit is not None
            and current.limit is not None
            and current.limit > previous.limit
        ):
            strengthened_limit = True
        elif (
            previous.limit is not None
            and current.limit is not None
            and current.limit < previous.limit
        ):
            errors.append("limit_weakened")

    if not (added_predicate or added_order or strengthened_limit):
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
        visibility = source.get("visibility")
        if visibility in {"blocked", "internal"}:
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
    _last_monitored_label_data: dict[str, object] | None = field(default=None, init=False)
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
                return result
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
                elif location[0] == "entity":
                    required_code = SubmitDraftErrorCode.ANCHOR_ENTITY_REQUIRED
                else:
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

        latest_query_result = self._latest_successful_query_result_since_last_submission()
        if latest_query_result is None:
            error_codes.append(SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISSING)
        else:
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
            contract_error_codes, contract_diagnostics = _query_contract_evidence_errors(
                contract=payload.answer_contract,
                query_result=latest_query_result,
            )
            error_codes.extend(contract_error_codes)
            invalid_diagnostics.update(contract_diagnostics)
            visibility_error_codes, visibility_diagnostics = _query_visibility_errors(
                latest_query_result
            )
            error_codes.extend(visibility_error_codes)
            invalid_diagnostics.update(visibility_diagnostics)

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
            and self._last_answer_contract is not None
            and payload.answer_contract.kind == self._last_answer_contract.kind
            and _operation_signature(payload.answer_contract.operation)
            == _operation_signature(
                self._last_answer_contract.operation
            )
        ):
            error_codes.append(SubmitDraftErrorCode.LABEL_NOT_STRENGTHENED)
            invalid_diagnostics["unchanged_scalar_label_value"] = True
        if self._needs_label_change and self._last_answer_contract is not None:
            incremental_errors = _answer_contract_incremental_errors(
                previous=self._last_answer_contract,
                current=payload.answer_contract,
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
                        "specific by changing the canonical label, not just "
                        "the wording."
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
                            "canonical label, not just the wording."
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
                "Rejected. answer_contract must be a valid JSON object with kind, operation, predicates, order_by, limit, limit_phrase, and evidence. Set evidence exactly to the string 'latest_query'; do not paste query result JSON into evidence. Do not pass a malformed JSON string; pass a complete object whose brackets and list items are closed."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_PHRASE_MISSING: (
                "Rejected. Every answer_contract phrase must be an exact contiguous substring copied from user_request. Write the user_request wording first, then paste the exact same words into operation, predicate, order_by, and limit phrase fields."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISSING: (
                "Rejected. Call query immediately before submit_draft; the canonical label must be copied from the latest successful query result."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_EVIDENCE_MISMATCH: (
                "Rejected. label must exactly match the latest successful query result. For kind='scalar', copy the one aggregate row as the label object. For kind='list', copy the query rows array as the label list, even when the query returned one row."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_QUERY_MISMATCH: (
                "Rejected. answer_contract predicates/order_by must be present in the latest successful query evidence with the same table, column, operator, value, and direction."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_VISIBILITY_EVIDENCE_MISSING: (
                "Rejected. Call query again before submit_draft; the latest query result must include field visibility evidence."  # noqa: E501
            ),
            SubmitDraftErrorCode.LABEL_NON_USER_VISIBLE_SOURCE: (
                "Rejected. The label directly exposes a field that is explicitly marked internal or blocked in the latest query metadata. Keep internal/blocked source values out of the submitted label; use a user-visible output value or a derived aggregate that does not expose the source value."  # noqa: E501
            ),
            SubmitDraftErrorCode.ANSWER_CONTRACT_NOT_INCREMENTAL: (
                "Rejected. After a specificity rejection, keep the same answer kind and target operation, preserve prior predicates/order, and add a new grounded structural constraint that the current database evidence supports."  # noqa: E501
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
        self._last_monitored_label_data = label_data


def build_submit_draft_sdk_tool(controller: SubmitDraftController) -> object:
    from agents import FunctionTool

    params_json_schema = SubmitDraftPayload.model_json_schema()

    async def _invoke_tool(_tool_context: Any, input_json: str) -> str:
        parsed = json.loads(input_json) if input_json else {}
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
        strict_json_schema=False,
    )
