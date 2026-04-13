"""Shared JSONL file sink utilities for synthesis logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from rl_task_foundry.synthesis.canonicalize import canonical_json


@dataclass(slots=True)
class JsonlFileSink:
    """Append JSONL records while reusing a single file handle."""

    path: Path
    _handle: TextIO | None = field(default=None, init=False, repr=False)

    def write_record(self, payload: dict[str, object]) -> None:
        handle = self._ensure_handle()
        handle.write(canonical_json(payload, default=str))
        handle.write("\n")
        handle.flush()

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None

    def _ensure_handle(self) -> TextIO:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        return self._handle
