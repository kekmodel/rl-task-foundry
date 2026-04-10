"""Small text helpers shared across tool compilation and evaluation."""

from __future__ import annotations

_PEOPLE_LIKE_TOKENS = {
    "staff",
    "employee",
    "agent",
    "user",
    "customer",
    "member",
    "person",
    "patient",
    "student",
    "teacher",
    "driver",
    "courier",
}
_CASE_LIKE_TOKENS = {
    "order",
    "payment",
    "rental",
    "booking",
    "shipment",
    "request",
    "ticket",
    "case",
    "invoice",
    "transaction",
}
_PLACE_LIKE_TOKENS = {
    "store",
    "branch",
    "site",
    "location",
    "outlet",
    "shop",
    "warehouse",
    "office",
    "station",
}


def singularize_token(token: str) -> str:
    """Return a light-weight English singularization for identifier tokens."""

    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 1:
        return token[:-1]
    return token


def humanize_identifier(identifier: str) -> str:
    """Convert a snake_case-ish identifier into a space-delimited label."""

    parts = [part for part in identifier.replace("-", "_").split("_") if part]
    if not parts:
        return identifier.strip()
    return " ".join(parts)


def count_unit_hint_for_identifier(identifier: str) -> str:
    """Infer a coarse count unit for an entity identifier."""

    token = singularize_token(identifier.strip().lower())
    if token in _PEOPLE_LIKE_TOKENS:
        return "people"
    if token in _CASE_LIKE_TOKENS:
        return "cases"
    if token in _PLACE_LIKE_TOKENS:
        return "places"
    return "items"


def default_count_target_label(identifier: str, *, language: str) -> str:
    """Return a user-facing target label for count questions."""

    unit = count_unit_hint_for_identifier(identifier)
    if language == "ko":
        if unit == "people":
            return "사람"
        if unit == "cases":
            return "건"
        if unit == "places":
            return "장소"
    else:
        if unit == "people":
            return "people"
        if unit == "cases":
            return "cases"
        if unit == "places":
            return "locations"
    return humanize_identifier(singularize_token(identifier))


def count_phrase_reference(identifier: str) -> str:
    """Return a neutral concept phrase for aggregate question composition."""

    unit = count_unit_hint_for_identifier(identifier)
    if unit == "people":
        return "people involved in this"
    if unit == "cases":
        return "relevant cases"
    if unit == "places":
        return "locations involved in this"
    return humanize_identifier(singularize_token(identifier))
