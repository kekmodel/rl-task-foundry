"""Shared validation helpers for JSON payloads passed between tool handlers.

Consolidates four near-identical copies (atomic tool_factory, composer
tool_factory, composer query, common schema) into one home. Two shapes are
supported:

- `require_*(payload, key)` / `optional_*(payload, key)` — look up a key in a
  dict-shaped payload and validate the value's type. Raise `TypeError` on
  mismatch so downstream handlers can convert to structured tool errors.
- `ensure_*(value, field)` — validate an already-extracted value; used by
  callers that unpack a nested payload manually (e.g. `query.py` destructures
  a spec before validating each piece).
"""

from __future__ import annotations

JsonObject = dict[str, object]


def require_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise TypeError(
            f"{key!r} must be a string; got {type(value).__name__}"
        )
    return value


def require_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{key!r} must be an integer; got {type(value).__name__}"
        )
    return value


def require_str_list(payload: JsonObject, key: str) -> list[str]:
    raw = payload.get(key)
    return ensure_str_list(raw, key)


def optional_int(payload: JsonObject, key: str) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    return require_int(payload, key)


def optional_str(payload: JsonObject, key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    return require_str(payload, key)


def ensure_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{field!r} must be a string; got {type(value).__name__}"
        )
    return value


def ensure_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field!r} must be an integer; got {type(value).__name__}"
        )
    return value


def ensure_str_list(raw: object, field: str) -> list[str]:
    if not isinstance(raw, list):
        raise TypeError(
            f"{field!r} must be a list of strings; got {type(raw).__name__}"
        )
    items: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise TypeError(
                f"{field}[{index}] must be a string; got "
                f"{type(item).__name__}"
            )
        items.append(item)
    return items
