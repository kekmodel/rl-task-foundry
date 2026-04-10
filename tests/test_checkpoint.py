import sqlite3

from rl_task_foundry.infra.checkpoint import ensure_checkpoint


def test_checkpoint_store_tracks_processed_keys_and_flushes(tmp_path):
    store = ensure_checkpoint(tmp_path / "artifacts" / "run.db")

    assert store.is_processed("anchor:1") is False

    store.mark_processed("anchor:1", payload={"task_id": "task_1"})
    assert store.is_processed("anchor:1") is True
    store.flush()

    reopened = ensure_checkpoint(store.run_db_path)
    assert reopened.is_processed("anchor:1") is True

    with sqlite3.connect(store.run_db_path) as conn:
        row = conn.execute(
            "SELECT namespace, item_key FROM processed_keys WHERE item_key = ?",
            ("anchor:1",),
        ).fetchone()
    assert row == ("anchors", "anchor:1")
