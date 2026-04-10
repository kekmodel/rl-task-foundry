import sqlite3

from rl_task_foundry.infra.storage import bootstrap_run_db


def test_bootstrap_run_db_creates_core_tables(tmp_path):
    db_path = bootstrap_run_db(tmp_path / "artifacts" / "run.db")
    with sqlite3.connect(db_path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert {
        "runs",
        "tasks",
        "accepted_examples",
        "verification_results",
        "event_log",
        "processed_keys",
        "budget_reservations",
    } <= names
    assert journal_mode.lower() == "wal"
