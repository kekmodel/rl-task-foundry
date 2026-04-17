"""Export registered task bundles into bundle layout.

Per-database artifacts shipped in an exported bundle:

- ``schema_snapshot.json`` — the immutable schema snapshot; env servers
  rebuild the atomic calculus / composer DSL from it.
- ``tooling_version.json`` — tooling version identifier + inventory of
  the primitives the snapshot is expected to drive.

Per-task artifacts are unchanged (task.yaml, task.json, instance.json,
canonical_answer.json).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.snapshot_materializer import (
    SchemaSnapshotMaterializer,
)
from rl_task_foundry.synthesis.task_registry import (
    TaskRegistryRecord,
    TaskRegistryWriter,
)


@dataclass(frozen=True, slots=True)
class TaskBundleExportSummary:
    bundle_root: Path
    database_count: int
    task_count: int
    db_ids: tuple[str, ...]
    task_ids: tuple[str, ...]


@dataclass(slots=True)
class TaskBundleExporter:
    registry: TaskRegistryWriter
    snapshot_materializer: SchemaSnapshotMaterializer

    @classmethod
    def for_config(cls, config: AppConfig) -> "TaskBundleExporter":
        return cls(
            registry=TaskRegistryWriter.for_config(config),
            snapshot_materializer=SchemaSnapshotMaterializer.for_config(config),
        )

    def export_bundle(
        self,
        bundle_root: Path,
        *,
        db_id: str | None = None,
        topic: str | None = None,
        task_id: str | None = None,
    ) -> TaskBundleExportSummary:
        records = self._task_records(db_id=db_id, topic=topic, task_id=task_id)
        self._prepare_bundle_root(bundle_root)

        databases_dir = bundle_root / "databases"
        tasks_dir = bundle_root / "tasks"
        databases_dir.mkdir(parents=True, exist_ok=True)
        tasks_dir.mkdir(parents=True, exist_ok=True)

        exported_db_ids: list[str] = []
        for resolved_db_id in sorted({record.db_id for record in records}):
            self._export_database_bundle(bundle_root, resolved_db_id)
            exported_db_ids.append(resolved_db_id)

        exported_task_ids: list[str] = []
        for record in sorted(records, key=lambda item: (item.db_id, item.task_id)):
            self._export_task_bundle(bundle_root, record)
            exported_task_ids.append(record.task_id)

        return TaskBundleExportSummary(
            bundle_root=bundle_root,
            database_count=len(exported_db_ids),
            task_count=len(exported_task_ids),
            db_ids=tuple(exported_db_ids),
            task_ids=tuple(exported_task_ids),
        )

    def _task_records(
        self,
        *,
        db_id: str | None,
        topic: str | None,
        task_id: str | None,
    ) -> list[TaskRegistryRecord]:
        count = self.registry.task_count(db_id=db_id, topic=topic)
        records = self.registry.list_tasks(
            limit=count,
            db_id=db_id,
            topic=topic,
        )
        if task_id is not None:
            records = [record for record in records if record.task_id == task_id]
        return records

    @staticmethod
    def _prepare_bundle_root(bundle_root: Path) -> None:
        if bundle_root.exists():
            if any(bundle_root.iterdir()):
                raise FileExistsError(f"bundle output directory must be empty: {bundle_root}")
        else:
            bundle_root.mkdir(parents=True, exist_ok=True)

    def _export_database_bundle(self, bundle_root: Path, db_id: str) -> None:
        source_dir = self.snapshot_materializer.root_dir / db_id
        source_snapshot = source_dir / "schema_snapshot.json"
        source_version = source_dir / "tooling_version.json"
        if not source_snapshot.exists():
            raise FileNotFoundError(
                "missing materialized schema snapshot for "
                f"db_id={db_id!r}: {source_snapshot}. "
                "Resolve SynthesisDb.schema_snapshot() (or call "
                "SchemaSnapshotMaterializer.materialize) before exporting."
            )
        if not source_version.exists():
            raise FileNotFoundError(
                "missing tooling_version.json for "
                f"db_id={db_id!r}: {source_version}"
            )
        target_dir = bundle_root / "databases" / db_id
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_snapshot, target_dir / "schema_snapshot.json")
        shutil.copy2(source_version, target_dir / "tooling_version.json")

    def _export_task_bundle(
        self,
        bundle_root: Path,
        record: TaskRegistryRecord,
    ) -> None:
        source_dir = record.filesystem_path
        target_dir = bundle_root / "tasks" / record.task_id
        target_dir.mkdir(parents=True, exist_ok=True)

        self._copy_required_file(source_dir / "task.yaml", target_dir / "task.yaml")
        self._copy_required_file(source_dir / "task.json", target_dir / "task.json")
        self._copy_required_file(source_dir / "instance.json", target_dir / "instance.json")
        self._copy_required_file(
            source_dir / "canonical_answer.json",
            target_dir / "canonical_answer.json",
        )

    @staticmethod
    def _copy_required_file(source: Path, target: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(f"missing required export source file: {source}")
        shutil.copy2(source, target)
