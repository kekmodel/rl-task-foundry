"""Persist per-database schema snapshots + tooling version for bundle export.

The exported bundle ships:

- ``schema_snapshot.json`` — the immutable schema snapshot an env server
  can load to rebuild the atomic calculus (and, when desired, the
  composer DSL) via `tooling.common.snapshot_from_dict`.
- ``tooling_version.json`` — a small manifest listing the tooling
  version identifier plus the known composer tools and atomic
  primitives. Lets the env server reject mismatched artifacts without
  trying to import them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.tooling.common import SchemaSnapshot, snapshot_to_dict


TOOLING_VERSION = "atomic-calculus-v1"

_COMPOSER_TOOLS: tuple[str, ...] = (
    "schema_map",
    "profile",
    "sample",
    "neighborhood",
    "query",
)
_ATOMIC_PRIMITIVES: tuple[str, ...] = (
    "rows_where",
    "rows_via",
    "intersect",
    "order_by",
    "take",
    "count",
    "aggregate",
    "group_top",
    "read",
)


@dataclass(frozen=True, slots=True)
class SchemaSnapshotArtifact:
    db_id: str
    bundle_dir: Path
    snapshot_path: Path
    version_path: Path


@dataclass(slots=True)
class SchemaSnapshotMaterializer:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_config(cls, config: AppConfig) -> "SchemaSnapshotMaterializer":
        return cls(root_dir=config.output.traces_dir.parent / "databases")

    def materialize(
        self,
        *,
        db_id: str,
        snapshot: SchemaSnapshot,
    ) -> SchemaSnapshotArtifact:
        bundle_dir = self.root_dir / db_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = bundle_dir / "schema_snapshot.json"
        version_path = bundle_dir / "tooling_version.json"
        snapshot_path.write_text(
            json.dumps(
                snapshot_to_dict(snapshot),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        version_payload = {
            "tooling_version": TOOLING_VERSION,
            "composer_tools": list(_COMPOSER_TOOLS),
            "atomic_primitives": list(_ATOMIC_PRIMITIVES),
        }
        version_path.write_text(
            json.dumps(
                version_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return SchemaSnapshotArtifact(
            db_id=db_id,
            bundle_dir=bundle_dir,
            snapshot_path=snapshot_path,
            version_path=version_path,
        )


__all__ = [
    "SchemaSnapshotArtifact",
    "SchemaSnapshotMaterializer",
    "TOOLING_VERSION",
]
