"""Run checkpoint with WAL for crash-safe emit ordering."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_CHECKPOINT_FILE = "checkpoint.json"


class RunCheckpoint:
    """Tracks processed PKs with set-based O(1) lookup and file persistence."""

    def __init__(
        self,
        *,
        output_dir: Path,
        run_id: str = "",
        rng_seed: int = 0,
        schema_hash: str = "",
        tool_set_hash: str = "",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self._output_dir / _CHECKPOINT_FILE
        self._lock = asyncio.Lock()

        self._run_id = run_id or f"run_{int(time.time())}"
        self._rng_seed = rng_seed
        self._schema_hash = schema_hash
        self._tool_set_hash = tool_set_hash
        self._started_at = time.time()

        # Load existing checkpoint if present
        self._processed_pks: set[Any] = set()
        self._accepted_count = 0
        self._rejected_count = 0
        self._total_tokens = 0
        self._estimated_cost_usd = 0.0
        self._load()

    def _load(self) -> None:
        if self._checkpoint_path.exists():
            data = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
            self._processed_pks = set(data.get("processed_pks", []))
            self._accepted_count = data.get("accepted_count", 0)
            self._rejected_count = data.get("rejected_count", 0)
            self._total_tokens = data.get("total_tokens", 0)
            self._estimated_cost_usd = data.get("estimated_cost_usd", 0.0)
            log.info("Checkpoint loaded: %d PKs processed", len(self._processed_pks))

    def is_processed(self, pk: Any) -> bool:
        return pk in self._processed_pks

    async def mark_processed(self, pk: Any) -> None:
        """Mark PK as processed and persist to disk (under lock)."""
        async with self._lock:
            self._processed_pks.add(pk)
            await self._flush_locked()

    async def increment_accepted(self) -> None:
        async with self._lock:
            self._accepted_count += 1

    async def increment_rejected(self) -> None:
        async with self._lock:
            self._rejected_count += 1

    async def update_cost(self, tokens: int = 0, cost_usd: float = 0.0) -> None:
        async with self._lock:
            self._total_tokens += tokens
            self._estimated_cost_usd += cost_usd

    async def _flush_locked(self) -> None:
        """Write checkpoint to disk with fsync."""
        tmp_path = self._checkpoint_path.with_suffix(".tmp")
        data = json.dumps(self._to_dict(), indent=2)
        tmp_path.write_text(data, encoding="utf-8")
        # fsync for durability
        fd = os.open(str(tmp_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp_path.rename(self._checkpoint_path)

    async def flush(self) -> None:
        """Public flush (acquires lock)."""
        async with self._lock:
            await self._flush_locked()

    def _to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self._run_id,
            "rng_seed": self._rng_seed,
            "schema_hash": self._schema_hash,
            "tool_set_hash": self._tool_set_hash,
            "started_at": self._started_at,
            "processed_pks": sorted(self._processed_pks, key=str),
            "accepted_count": self._accepted_count,
            "rejected_count": self._rejected_count,
            "total_tokens": self._total_tokens,
            "estimated_cost_usd": self._estimated_cost_usd,
        }

    def get_manifest(self) -> dict[str, Any]:
        return self._to_dict()
