"""Task dedup helpers."""

from __future__ import annotations

import hashlib

from rl_task_foundry.tasks.models import TaskSpec


def exact_signature(task: TaskSpec) -> str:
    """Return a stable exact-dedup signature for a task."""

    material = f"{task.question}|{task.selected_path_id}|{task.answer_schema.model_dump_json()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
