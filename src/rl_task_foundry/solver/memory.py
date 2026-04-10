"""Explicit memory and summary events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MemoryMode = Literal["none", "explicit_summary", "session_only"]


@dataclass(slots=True)
class SummaryEvent:
    source_range: tuple[int, int]
    summary_text: str
    generated_by: str
