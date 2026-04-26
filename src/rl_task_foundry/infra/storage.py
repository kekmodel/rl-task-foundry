"""SQLite-backed run state storage."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rl_task_foundry.synthesis.canonicalize import canonical_json

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        config_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS accepted_examples (
        task_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS verification_results (
        task_id TEXT NOT NULL,
        run_id TEXT NOT NULL DEFAULT '',
        solver_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (task_id, solver_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS budget_ledger (
        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
        phase TEXT NOT NULL,
        amount_usd REAL NOT NULL,
        kind TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS coverage_counters (
        key TEXT PRIMARY KEY,
        value INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_log (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL DEFAULT '',
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS processed_keys (
        namespace TEXT NOT NULL,
        item_key TEXT NOT NULL,
        payload_json TEXT,
        processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (namespace, item_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS budget_reservations (
        reservation_id TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
)

INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks (run_id)",
    "CREATE INDEX IF NOT EXISTS idx_accepted_examples_run_id ON accepted_examples (run_id)",
    "CREATE INDEX IF NOT EXISTS idx_verification_results_run_id ON verification_results (run_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_log_run_id ON event_log (run_id)",
)

MIGRATION_STATEMENTS = (
    "ALTER TABLE tasks ADD COLUMN run_id TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE accepted_examples ADD COLUMN run_id TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE verification_results ADD COLUMN run_id TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE event_log ADD COLUMN run_id TEXT NOT NULL DEFAULT ''",
)


@dataclass(slots=True)
class RunDbSummary:
    run_id: str
    total_tasks: int
    accepted_tasks: int
    rejected_tasks: int
    skipped_tasks: int
    verification_results: int
    event_count: int


def connect_run_db(path: str | Path) -> sqlite3.Connection:
    """Open the durable run database with strict safety pragmas."""

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def bootstrap_run_db(path: str | Path) -> Path:
    """Create the run database and required tables."""

    db_path = Path(path)
    with connect_run_db(db_path) as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        _apply_migrations(conn)
        for statement in INDEX_STATEMENTS:
            conn.execute(statement)
        conn.commit()
    return db_path


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for statement in MIGRATION_STATEMENTS:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc).lower():
                continue
            raise


def _json_payload(payload: dict[str, object]) -> str:
    return canonical_json(payload, default=str)


def record_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    config_hash: str,
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (run_id, config_hash, created_at)
        VALUES (?, ?, ?)
        """,
        (run_id, config_hash, created_at),
    )


def record_task(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    task_id: str,
    status: str,
    payload: dict[str, object],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO tasks (task_id, run_id, status, payload_json)
        VALUES (?, ?, ?, ?)
        """,
        (task_id, run_id, status, _json_payload(payload)),
    )


def record_verification_result(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    task_id: str,
    solver_id: str,
    payload: dict[str, object],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO verification_results (task_id, run_id, solver_id, payload_json)
        VALUES (?, ?, ?, ?)
        """,
        (task_id, run_id, solver_id, _json_payload(payload)),
    )


def record_accepted_example(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    task_id: str,
    payload: dict[str, object],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO accepted_examples (task_id, run_id, payload_json)
        VALUES (?, ?, ?)
        """,
        (task_id, run_id, _json_payload(payload)),
    )


def record_event(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    event_type: str,
    payload: dict[str, object],
) -> None:
    conn.execute(
        """
        INSERT INTO event_log (run_id, event_type, payload_json)
        VALUES (?, ?, ?)
        """,
        (run_id, event_type, _json_payload(payload)),
    )


def record_budget_reservation(
    conn: sqlite3.Connection,
    *,
    reservation_id: str,
    payload: dict[str, object],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO budget_reservations (reservation_id, payload_json)
        VALUES (?, ?)
        """,
        (reservation_id, _json_payload(payload)),
    )


def clear_budget_reservation(conn: sqlite3.Connection, *, reservation_id: str) -> None:
    conn.execute(
        "DELETE FROM budget_reservations WHERE reservation_id = ?",
        (reservation_id,),
    )


def append_budget_ledger_entry(
    conn: sqlite3.Connection,
    *,
    phase: str,
    amount_usd: float,
    kind: str,
) -> None:
    conn.execute(
        """
        INSERT INTO budget_ledger (phase, amount_usd, kind)
        VALUES (?, ?, ?)
        """,
        (phase, amount_usd, kind),
    )


def summarize_run(path: str | Path, *, run_id: str) -> RunDbSummary:
    db_path = bootstrap_run_db(path)
    with connect_run_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_tasks,
                SUM(CASE WHEN status = 'accepted' THEN 1 ELSE 0 END) AS accepted_tasks,
                SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) AS rejected_tasks,
                SUM(CASE WHEN status NOT IN ('accepted', 'rejected')
                    THEN 1 ELSE 0 END) AS skipped_tasks,
                (
                    SELECT COUNT(*)
                    FROM verification_results
                    WHERE run_id = ?
                ) AS verification_results,
                (
                    SELECT COUNT(*)
                    FROM event_log
                    WHERE run_id = ?
                ) AS event_count
            FROM tasks
            WHERE run_id = ?
            """,
            (run_id, run_id, run_id),
        ).fetchone()
        assert row is not None
    return RunDbSummary(
        run_id=run_id,
        total_tasks=int(row["total_tasks"] or 0),
        accepted_tasks=int(row["accepted_tasks"] or 0),
        rejected_tasks=int(row["rejected_tasks"] or 0),
        skipped_tasks=int(row["skipped_tasks"] or 0),
        verification_results=int(row["verification_results"] or 0),
        event_count=int(row["event_count"] or 0),
    )
