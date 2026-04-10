"""Typed answer canonicalization."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema

DEFAULT_FLOAT_PRECISION = 6


def canonicalize_field(
    field: AnswerField,
    value: Any,
    *,
    float_precision: int = DEFAULT_FLOAT_PRECISION,
) -> Any:
    """Canonicalize one answer field."""

    if value is None:
        return None
    if field.type in {"string", "enum"}:
        return str(value).strip().lower()
    if field.type == "int":
        return int(value)
    if field.type == "float":
        return round(float(value), float_precision)
    if field.type == "bool":
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "t", "1", "yes", "y"}:
                return True
            if normalized in {"false", "f", "0", "no", "n"}:
                return False
        return bool(value)
    if field.type == "date":
        return _canonicalize_date(value)
    if field.type == "datetime":
        return _canonicalize_datetime(value)
    if field.type == "list[string]":
        items = [str(item).strip().lower() for item in value]
        return items if field.ordered else sorted(items)
    if field.type == "list[int]":
        items = [int(item) for item in value]
        return items if field.ordered else sorted(items)
    return str(value).strip()


def canonicalize_answer(
    schema: AnswerSchema,
    answer: dict[str, Any],
    *,
    float_precision: int = DEFAULT_FLOAT_PRECISION,
) -> dict[str, Any]:
    """Canonicalize a structured answer according to schema."""

    return {
        field.name: canonicalize_field(
            field,
            answer.get(field.name),
            float_precision=float_precision,
        )
        for field in schema.fields
    }


def _canonicalize_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        normalized = value.strip()
        try:
            return datetime.fromisoformat(normalized.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return date.fromisoformat(normalized).isoformat()
    raise TypeError(f"unsupported date value: {value!r}")


def _canonicalize_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).isoformat(timespec="seconds")
    if isinstance(value, str):
        normalized = value.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).isoformat(timespec="seconds")
    raise TypeError(f"unsupported datetime value: {value!r}")
