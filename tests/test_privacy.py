"""Tests for PII detection and redaction."""

from __future__ import annotations

from rlvr_synth.privacy import PiiDetector, redact_value


def test_detect_pii_columns() -> None:
    detector = PiiDetector(patterns=["email", "phone", "address"])
    columns = ["id", "name", "email_address", "phone_number", "city", "home_address", "tier"]

    pii = detector.detect(columns)
    assert "email_address" in pii
    assert "phone_number" in pii
    assert "home_address" in pii
    assert "id" not in pii
    assert "name" not in pii
    assert "city" not in pii


def test_redact_value() -> None:
    assert redact_value("kim@example.com") == "***REDACTED***"
    assert redact_value(12345) == "***REDACTED***"
    assert redact_value(None) is None


def test_detect_empty_patterns() -> None:
    detector = PiiDetector(patterns=[])
    columns = ["email", "phone"]
    assert detector.detect(columns) == set()
