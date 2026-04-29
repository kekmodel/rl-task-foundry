"""Unified analysis event log.

A ``TrialEventLogger`` owns one append-only JSONL timeline, normally
``analysis.jsonl``. Phase transitions, runner events, composer events, solver
events, and provider-visible reasoning records all land in the same file so an
analysis pass can read one ordered stream instead of joining several files.
Large SDK/tool traces still belong under ``traces/``; this log is the compact
analysis index over those artifacts.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, TextIO


class TrialEventLogger:
    """Append-only JSONL writer shared by composer + solver backends.

    Thread/coroutine safety: the single asyncio.Lock ensures a line
    write is not interleaved with another write. Individual text mode
    ``write``s under the GIL are already atomic for short strings, but
    long payloads may exceed the pipe-buf boundary, hence the lock.
    """

    def __init__(
        self,
        path: Path,
        *,
        flow_kind: str | None = None,
        flow_id: str | None = None,
        mirror_path: Path | None = None,
    ) -> None:
        self._path = path
        self._flow_kind = flow_kind
        self._flow_id = flow_id
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._mirror_fh: TextIO | None = None
        if mirror_path is not None and mirror_path != path:
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
            self._mirror_fh = mirror_path.open("a", encoding="utf-8")
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
        line = self._encode_line(
            actor=actor,
            actor_id=actor_id,
            event_type=event_type,
            payload=payload,
        )
        async with self._lock:
            self._write_line(line)

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

        line = self._encode_line(
            actor=actor,
            actor_id=actor_id,
            event_type=event_type,
            payload=payload,
        )
        self._write_line(line)

    def write_analysis_records(
        self,
        source_name: str,
        records: list[dict[str, Any]],
    ) -> Path | None:
        """Append structured analysis records to the unified timeline."""

        if not records:
            return None
        if "/" in source_name or source_name in {"", ".", ".."}:
            raise ValueError("analysis source name must be a simple relative name")
        event_type = Path(source_name).stem
        for record in records:
            actor = str(record.get("actor") or "analysis")
            actor_id_value = record.get("actor_id")
            actor_id = str(actor_id_value) if actor_id_value is not None else None
            payload = {
                "source_name": source_name,
                **{
                    key: value
                    for key, value in record.items()
                    if key not in {"actor", "actor_id"}
                },
            }
            self.log_sync(
                actor=actor,
                actor_id=actor_id,
                event_type=event_type,
                payload=payload,
            )
        return self._path

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
            if self._mirror_fh is not None:
                self._mirror_fh.flush()
                self._mirror_fh.close()
        except (OSError, ValueError):
            pass

    def _encode_line(
        self,
        *,
        actor: str,
        actor_id: str | None,
        event_type: str,
        payload: dict[str, Any],
    ) -> str:
        record: dict[str, Any] = {
            "ts": time.time(),
            "flow_kind": self._flow_kind,
            "flow_id": self._flow_id,
            "actor": actor,
            "actor_id": actor_id,
            "event_type": event_type,
            "payload": payload,
        }
        return json.dumps(record, ensure_ascii=False, default=str)

    def _write_line(self, line: str) -> None:
        self._fh.write(line + "\n")
        self._fh.flush()
        if self._mirror_fh is not None:
            self._mirror_fh.write(line + "\n")
            self._mirror_fh.flush()


__all__ = ["TrialEventLogger"]
