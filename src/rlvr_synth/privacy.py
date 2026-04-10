"""PII detection and redaction."""

from __future__ import annotations

from typing import Any

_REDACTED = "***REDACTED***"


class PiiDetector:
    """Detects PII columns by pattern matching against column names."""

    def __init__(self, patterns: list[str]) -> None:
        self._patterns = [p.lower() for p in patterns]

    def detect(self, column_names: list[str]) -> set[str]:
        """Return column names that match any PII pattern."""
        pii_columns: set[str] = set()
        for col in column_names:
            col_lower = col.lower()
            for pattern in self._patterns:
                if pattern in col_lower:
                    pii_columns.add(col)
                    break
        return pii_columns


def redact_value(value: Any) -> Any:
    """Replace a value with redacted placeholder. None stays None."""
    if value is None:
        return None
    return _REDACTED


def redact_dict(data: dict[str, Any], pii_columns: set[str]) -> dict[str, Any]:
    """Redact PII fields in a dict."""
    return {
        k: redact_value(v) if k in pii_columns else v
        for k, v in data.items()
    }
