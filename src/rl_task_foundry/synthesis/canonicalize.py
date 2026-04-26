"""Schema-driven canonicalization for synthesis output contracts.

This module is intentionally self-contained within the synthesis stack and does
not import from the legacy truth/tasks/tools directories scheduled for removal.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import StrEnum

from rl_task_foundry.synthesis.contracts import (
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
)


@dataclass(slots=True)
class CanonicalizationError(ValueError):
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class RewardStatus(StrEnum):
    MATCHED = "matched"
    JSON_DECODE_FAILED = "json_decode_failed"
    SCHEMA_MISMATCH = "schema_mismatch"
    EM_MISMATCH = "em_mismatch"


@dataclass(frozen=True, slots=True)
class RewardResult:
    reward: float
    status: RewardStatus
    detail: str | None = None


def canonicalize_output(
    schema: OutputSchemaContract,
    payload: object,
) -> object:
    """Canonicalize a parsed answer payload against the synthesis output schema."""

    return canonicalize_field(schema.root, payload, path="$")


def canonicalize_field(
    field: OutputFieldContract,
    value: object,
    *,
    path: str = "$",
) -> object:
    """Canonicalize one value according to a synthesis output field contract."""

    if value is None:
        if field.nullable:
            return None
        raise CanonicalizationError(path, "null is not allowed for this field")

    if field.type is OutputFieldType.STRING:
        return _canonicalize_string(value, path=path)
    if field.type is OutputFieldType.ENUM:
        return _canonicalize_enum(field, value, path=path)
    if field.type is OutputFieldType.INT:
        return _canonicalize_int(value, path=path)
    if field.type is OutputFieldType.FLOAT:
        return _canonicalize_float(value, path=path)
    if field.type is OutputFieldType.BOOL:
        return _canonicalize_bool(value, path=path)
    if field.type is OutputFieldType.DATE:
        return _canonicalize_date(value, path=path)
    if field.type is OutputFieldType.DATETIME:
        return _canonicalize_datetime(value, path=path)
    if field.type is OutputFieldType.OBJECT:
        return _canonicalize_object(field, value, path=path)
    if field.type is OutputFieldType.LIST:
        return _canonicalize_list(field, value, path=path)
    raise CanonicalizationError(path, f"unsupported field type: {field.type}")


def canonical_json(
    value: object,
    *,
    default: Callable[[object], object] | None = None,
) -> str:
    """Return a stable JSON representation for exact-match comparisons."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=default,
    )


def _canonicalize_string(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise CanonicalizationError(path, "expected string")
    return value


def _canonicalize_enum(field: OutputFieldContract, value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise CanonicalizationError(path, "expected enum string")
    if field.enum_values and value not in field.enum_values:
        raise CanonicalizationError(path, f"unexpected enum value: {value!r}")
    return value


def _canonicalize_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanonicalizationError(path, "expected int")
    return value


def _canonicalize_float(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanonicalizationError(path, "expected float")
    return float(value)


def _canonicalize_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise CanonicalizationError(path, "expected bool")
    return value


def _canonicalize_date(value: object, *, path: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise CanonicalizationError(path, "expected ISO date string")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return date.fromisoformat(normalized).isoformat()
    except ValueError:
        try:
            return datetime.fromisoformat(normalized).date().isoformat()
        except ValueError as exc:
            raise CanonicalizationError(path, f"invalid ISO date value: {value!r}") from exc


def _canonicalize_datetime(value: object, *, path: str) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return datetime.combine(value, time.min).isoformat(timespec="seconds")
    if not isinstance(value, str):
        raise CanonicalizationError(path, "expected ISO datetime string")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).isoformat(timespec="seconds")
    except ValueError as exc:
        raise CanonicalizationError(path, f"invalid ISO datetime value: {value!r}") from exc


def _canonicalize_object(
    field: OutputFieldContract,
    value: object,
    *,
    path: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CanonicalizationError(path, "expected object")

    allowed_names = {child.name for child in field.fields}
    unexpected = sorted(set(value) - allowed_names)
    if unexpected:
        raise CanonicalizationError(path, f"unexpected object keys: {unexpected}")

    canonical: dict[str, object] = {}
    for child in field.fields:
        child_path = _field_path(path, child.name)
        if child.name not in value:
            if child.nullable:
                canonical[child.name] = None
                continue
            raise CanonicalizationError(child_path, "missing required field")
        canonical[child.name] = canonicalize_field(
            child,
            value[child.name],
            path=child_path,
        )
    return canonical


def _canonicalize_list(
    field: OutputFieldContract,
    value: object,
    *,
    path: str,
) -> list[object]:
    if not isinstance(value, list):
        raise CanonicalizationError(path, "expected list")
    if field.items is None:
        raise CanonicalizationError(path, "list field is missing item schema")
    if field.length is not None and len(value) != field.length:
        raise CanonicalizationError(
            path,
            f"expected list with exactly {field.length} items",
        )

    items = [
        canonicalize_field(field.items, item, path=f"{path}[{index}]")
        for index, item in enumerate(value)
    ]
    if field.ordered:
        result = items
    elif field.sort_key is not None:
        assert field.items is not None
        sk = field.sort_key
        item_field = field.items
        result = sorted(
            items,
            key=lambda element: _object_sort_key(element, sk, item_field),
        )
    else:
        result = sorted(items, key=_unordered_sort_key)

    if field.unique_elements:
        result = _dedupe_preserving_order(result)

    return result


def _object_sort_key(
    element: object,
    sort_key_path: tuple[str, ...],
    item_field: OutputFieldContract,
) -> tuple[int, tuple[object, ...] | str, str]:
    if item_field.type is not OutputFieldType.OBJECT or not isinstance(element, dict):
        return (0, canonical_json(element), canonical_json(element))

    primary_key_parts: list[object] = []
    for field_name in sort_key_path:
        primary_key_parts.append(element.get(field_name))

    return (1, tuple(primary_key_parts), canonical_json(element))


def _dedupe_preserving_order(items: list[object]) -> list[object]:
    seen: set[str] = set()
    deduped: list[object] = []
    for item in items:
        key = canonical_json(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _unordered_sort_key(value: object) -> tuple[int, str]:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return (0, canonical_json(value))
    return (1, canonical_json(value))


def compute_reward(
    *,
    submitted_answer_text: str,
    canonical_answer: object,
    output_schema: OutputSchemaContract,
) -> RewardResult:
    """Pure-function reward computation for RL runtime and environment serving."""

    try:
        parsed = json.loads(submitted_answer_text)
    except json.JSONDecodeError as exc:
        return RewardResult(
            reward=0.0,
            status=RewardStatus.JSON_DECODE_FAILED,
            detail=str(exc),
        )

    try:
        canonical_submitted = canonicalize_output(output_schema, parsed)
    except CanonicalizationError as exc:
        return RewardResult(
            reward=0.0,
            status=RewardStatus.SCHEMA_MISMATCH,
            detail=str(exc),
        )

    try:
        canonical_expected = canonicalize_output(output_schema, canonical_answer)
    except CanonicalizationError as exc:
        return RewardResult(
            reward=0.0,
            status=RewardStatus.SCHEMA_MISMATCH,
            detail=f"canonical answer failed validation: {exc}",
        )

    if canonical_submitted == canonical_expected:
        return RewardResult(reward=1.0, status=RewardStatus.MATCHED)

    return RewardResult(reward=0.0, status=RewardStatus.EM_MISMATCH)


def _field_path(base: str, child_name: str) -> str:
    if base == "$":
        return f"$.{child_name}"
    return f"{base}.{child_name}"
