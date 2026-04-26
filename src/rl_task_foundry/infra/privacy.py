"""Visibility policy helpers."""

from __future__ import annotations

from typing import Literal

Visibility = Literal["blocked", "internal", "user_visible"]


def infer_visibility(column_name: str) -> Visibility | None:
    """Deprecated compatibility hook.

    Visibility must come from explicit configuration or database metadata, not
    from token rules over column names.
    """

    del column_name
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
    return default_visibility


def redact_value(value: object, visibility: Visibility) -> object:
    if value is None or visibility == "user_visible":
        return value
    if visibility == "internal":
        return "[INTERNAL]"
    return "[REDACTED]"


def redact_dict(
    payload: dict[str, object], visibilities: dict[str, Visibility]
) -> dict[str, object]:
    return {
        key: redact_value(value, visibilities.get(key, "blocked")) for key, value in payload.items()
    }
