"""Visibility policy helpers."""

from __future__ import annotations

from typing import Literal, TypeGuard

Visibility = Literal["blocked", "internal", "user_visible"]
VISIBILITY_BLOCKED: Visibility = "blocked"
VISIBILITY_INTERNAL: Visibility = "internal"
VISIBILITY_USER_VISIBLE: Visibility = "user_visible"
VISIBILITY_VALUES: frozenset[str] = frozenset(
    (
        VISIBILITY_BLOCKED,
        VISIBILITY_INTERNAL,
        VISIBILITY_USER_VISIBLE,
    )
)
DIRECT_LABEL_BLOCKING_VISIBILITIES: frozenset[str] = frozenset(
    (
        VISIBILITY_BLOCKED,
        VISIBILITY_INTERNAL,
    )
)


def is_visibility(value: object) -> TypeGuard[Visibility]:
    return isinstance(value, str) and value in VISIBILITY_VALUES


def is_blocked_visibility(value: object) -> bool:
    return value == VISIBILITY_BLOCKED


def is_internal_visibility(value: object) -> bool:
    return value == VISIBILITY_INTERNAL


def is_user_visible_visibility(value: object) -> bool:
    return value == VISIBILITY_USER_VISIBLE


def blocks_direct_label_exposure(value: object) -> bool:
    return value in DIRECT_LABEL_BLOCKING_VISIBILITIES


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
    if value is None or is_user_visible_visibility(visibility):
        return value
    if is_internal_visibility(visibility):
        return "[INTERNAL]"
    return "[REDACTED]"


def redact_dict(
    payload: dict[str, object], visibilities: dict[str, Visibility]
) -> dict[str, object]:
    return {
        key: redact_value(value, visibilities.get(key, VISIBILITY_BLOCKED))
        for key, value in payload.items()
    }
