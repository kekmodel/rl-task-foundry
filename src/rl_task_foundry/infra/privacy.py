"""Visibility policy helpers."""

from __future__ import annotations

import re
from typing import Literal

Visibility = Literal["blocked", "internal", "user_visible"]

_BLOCKED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|_)(ssn|social_security|passport|tax_id|cvv|pin|password|secret|token|api_key|private_key)($|_)",
        r"(^|_)(credit_card|card_number|bank_account|iban|routing_number)($|_)",
    )
)
_INTERNAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|_)(email|e_mail|phone|mobile|telephone|fax)($|_)",
        r"(^|_)(address|street|zipcode|postal_code|postcode|city|state|province|country)($|_)",
        r"(^|_)(birth|birthdate|birthday|dob|first_name|last_name|full_name)($|_)",
    )
)


def infer_visibility(column_name: str) -> Visibility | None:
    normalized = column_name.strip().lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(normalized):
            return "blocked"
    for pattern in _INTERNAL_PATTERNS:
        if pattern.search(normalized):
            return "internal"
    return None


def resolve_visibility(
    column_name: str,
    *,
    default_visibility: Visibility,
    overrides: dict[str, Visibility],
) -> Visibility:
    """Resolve a column's visibility with override precedence."""

    if column_name in overrides:
        return overrides[column_name]
    return infer_visibility(column_name) or default_visibility


def redact_value(value: object, visibility: Visibility) -> object:
    if value is None or visibility == "user_visible":
        return value
    if visibility == "internal":
        return "[INTERNAL]"
    return "[REDACTED]"


def redact_dict(payload: dict[str, object], visibilities: dict[str, Visibility]) -> dict[str, object]:
    return {
        key: redact_value(value, visibilities.get(key, "blocked"))
        for key, value in payload.items()
    }
