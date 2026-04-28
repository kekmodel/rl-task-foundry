"""Unified per-trial event log.

A ``TrialEventLogger`` owns a single append-only JSONL file at
``<trial_debug_dir>/trial_events.jsonl`` and records every
composer/solver/runner event as one line. The design premise:

- Scattered traces (``synthesis/tool_traces``, per-solver files,
  ``phase_monitors.jsonl``) force an analyst to glue 5+ files by
  ``task_id`` / ``solver_id`` and guess the chronology.
- A single JSONL lets ``tail -f`` stream every actor interleaved in
  real time, and jq-style filters pull the per-actor view when needed.

Each line is a small dict. We avoid pretty-printing and keep heavyweight raw
payloads out of the primary timeline; SDK run items and tool traces should be
summarized into structured fields, with optional artifact paths used only when
a large debug payload genuinely needs a sidecar file.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any


class TrialEventLogger:
    """Append-only JSONL writer shared by composer + solver backends.

    Thread/coroutine safety: the single asyncio.Lock ensures a line
    write is not interleaved with another write. Individual text mode
    ``write``s under the GIL are already atomic for short strings, but
    long payloads may exceed the pipe-buf boundary, hence the lock.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def log(
        self,
        *,
        actor: str,
        event_type: str,
        payload: dict[str, Any],
        actor_id: str | None = None,
    ) -> None:
        line = json.dumps(
            {
                "ts": time.time(),
                "actor": actor,
                "actor_id": actor_id,
                "event_type": event_type,
                "payload": payload,
            },
            ensure_ascii=False,
            default=str,
        )
        async with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def log_sync(
        self,
        *,
        actor: str,
        event_type: str,
        payload: dict[str, Any],
        actor_id: str | None = None,
    ) -> None:
        """Fire-and-forget sync variant for non-async call sites. Skips
        the asyncio lock. Safe when only one writer touches the file or
        when the caller is responsible for serialization.
        """

        line = json.dumps(
            {
                "ts": time.time(),
                "actor": actor,
                "actor_id": actor_id,
                "event_type": event_type,
                "payload": payload,
            },
            ensure_ascii=False,
            default=str,
        )
        self._fh.write(line + "\n")
        self._fh.flush()

    def write_sidecar_jsonl(
        self,
        filename: str,
        records: list[dict[str, Any]],
    ) -> Path | None:
        if not records:
            return None
        if "/" in filename or filename in {"", ".", ".."}:
            raise ValueError("sidecar filename must be a simple relative filename")
        path = self._path.parent / filename
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str))
                handle.write("\n")
            handle.flush()
        return path

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except (OSError, ValueError):
            pass


__all__ = ["TrialEventLogger"]
