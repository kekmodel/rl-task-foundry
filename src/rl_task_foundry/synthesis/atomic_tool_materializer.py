"""Durable db-level atomic tool bundle materialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle


@dataclass(frozen=True, slots=True)
class AtomicToolMaterialization:
    db_id: str
    bundle_dir: Path
    source_path: Path
    definitions_path: Path


@dataclass(slots=True)
class AtomicToolMaterializer:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_config(cls, config: AppConfig) -> "AtomicToolMaterializer":
        return cls(root_dir=config.output.traces_dir.parent / "databases")

    def materialize_bundle(self, bundle: AtomicToolBundle) -> AtomicToolMaterialization:
        bundle_dir = self.root_dir / bundle.db_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        source_path = bundle_dir / "atomic_tools.py"
        definitions_path = bundle_dir / "atomic_tool_definitions.json"
        source_path.write_text(bundle.source, encoding="utf-8")
        definitions_path.write_text(
            bundle.actor_tool_definitions_json(),
            encoding="utf-8",
        )
        return AtomicToolMaterialization(
            db_id=bundle.db_id,
            bundle_dir=bundle_dir,
            source_path=source_path,
            definitions_path=definitions_path,
        )

    def read_actor_tool_definitions(self, *, db_id: str) -> list[dict[str, object]]:
        definitions_path = self.root_dir / db_id / "atomic_tool_definitions.json"
        if not definitions_path.exists():
            return []
        payload = json.loads(definitions_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("atomic_tool_definitions.json must contain a list payload")
        return [dict(item) for item in payload]
