"""Verification policy models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VerificationPolicy:
    require_provenance: bool = True
    fail_on_internal_field_leak: bool = True
    float_precision: int = 6
    shadow_sample_rate: float = 0.1
