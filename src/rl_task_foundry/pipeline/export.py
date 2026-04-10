"""Dataset export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from rl_task_foundry.tasks.models import AcceptedExample


def export_accepted_jsonl(path: str | Path, examples: list[AcceptedExample]) -> Path:
    """Write accepted examples as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.model_dump_json())
            handle.write("\n")
    return output_path
