"""Small text helpers shared across tool compilation and evaluation."""

from __future__ import annotations


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
