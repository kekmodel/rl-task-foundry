"""Checkpoint helpers built on top of run.db."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rl_task_foundry.infra.storage import bootstrap_run_db, connect_run_db
from rl_task_foundry.synthesis.canonicalize import canonical_json


@dataclass(slots=True)
class CheckpointStore:
    """SQLite-backed checkpoint with O(1) in-memory membership checks."""

    run_db_path: Path
    _processed: set[tuple[str, str]]
    _pending: dict[tuple[str, str], str | None]

    @classmethod
    def open(cls, path: str | Path) -> "CheckpointStore":
        run_db_path = bootstrap_run_db(path)
        with connect_run_db(run_db_path) as conn:
            rows = conn.execute(
                "SELECT namespace, item_key FROM processed_keys"
            ).fetchall()
        return cls(
            run_db_path=run_db_path,
            _processed={(row["namespace"], row["item_key"]) for row in rows},
            _pending={},
        )

    def is_processed(self, item_key: str, *, namespace: str = "anchors") -> bool:
        return (namespace, item_key) in self._processed

    def mark_processed(
        self,
        item_key: str,
        *,
        namespace: str = "anchors",
        payload: dict[str, object] | None = None,
    ) -> None:
        key = (namespace, item_key)
        self._processed.add(key)
        self._pending[key] = canonical_json(payload, default=str) if payload else None

    def flush(self) -> None:
        if not self._pending:
            return
        with connect_run_db(self.run_db_path) as conn:
            self.flush_to_connection(conn)

    def flush_to_connection(self, conn: sqlite3.Connection) -> None:
        if not self._pending:
            return
        rows = [
            (namespace, item_key, payload_json)
            for (namespace, item_key), payload_json in self._pending.items()
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO processed_keys (namespace, item_key, payload_json)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        self._pending.clear()


def ensure_checkpoint(path: str | Path) -> CheckpointStore:
    """Ensure durable run state exists and return its projection."""

    return CheckpointStore.open(path)
