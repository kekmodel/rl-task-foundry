"""Export registered synthesis environments into API-server bundle layout."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.atomic_tool_materializer import AtomicToolMaterializer
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.environment_registry import (
    EnvironmentRegistryRecord,
    EnvironmentRegistryWriter,
)


@dataclass(frozen=True, slots=True)
class EnvironmentBundleExportSummary:
    bundle_root: Path
    database_count: int
    environment_count: int
    db_ids: tuple[str, ...]
    env_ids: tuple[str, ...]


@dataclass(slots=True)
class EnvironmentBundleExporter:
    registry: EnvironmentRegistryWriter
    materializer: AtomicToolMaterializer

    @classmethod
    def for_config(cls, config: AppConfig) -> "EnvironmentBundleExporter":
        return cls(
            registry=EnvironmentRegistryWriter.for_config(config),
            materializer=AtomicToolMaterializer.for_config(config),
        )

    def export_bundle(
        self,
        bundle_root: Path,
        *,
        db_id: str | None = None,
        category: CategoryTaxonomy | None = None,
        env_id: str | None = None,
    ) -> EnvironmentBundleExportSummary:
        records = self._environment_records(db_id=db_id, category=category, env_id=env_id)
        self._prepare_bundle_root(bundle_root)

        databases_dir = bundle_root / "databases"
        environments_dir = bundle_root / "environments"
        databases_dir.mkdir(parents=True, exist_ok=True)
        environments_dir.mkdir(parents=True, exist_ok=True)

        exported_db_ids: list[str] = []
        for resolved_db_id in sorted({record.db_id for record in records}):
            self._export_database_bundle(bundle_root, resolved_db_id)
            exported_db_ids.append(resolved_db_id)

        exported_env_ids: list[str] = []
        for record in sorted(records, key=lambda item: (item.db_id, item.env_id)):
            self._export_environment_bundle(bundle_root, record)
            exported_env_ids.append(record.env_id)

        return EnvironmentBundleExportSummary(
            bundle_root=bundle_root,
            database_count=len(exported_db_ids),
            environment_count=len(exported_env_ids),
            db_ids=tuple(exported_db_ids),
            env_ids=tuple(exported_env_ids),
        )

    def _environment_records(
        self,
        *,
        db_id: str | None,
        category: CategoryTaxonomy | None,
        env_id: str | None,
    ) -> list[EnvironmentRegistryRecord]:
        count = self.registry.environment_count(db_id=db_id, category=category)
        records = self.registry.list_environments(
            limit=count,
            db_id=db_id,
            category=category,
        )
        if env_id is not None:
            records = [record for record in records if record.env_id == env_id]
        return records

    @staticmethod
    def _prepare_bundle_root(bundle_root: Path) -> None:
        if bundle_root.exists():
            if any(bundle_root.iterdir()):
                raise FileExistsError(f"bundle output directory must be empty: {bundle_root}")
        else:
            bundle_root.mkdir(parents=True, exist_ok=True)

    def _export_database_bundle(self, bundle_root: Path, db_id: str) -> None:
        source_dir = self.materializer.root_dir / db_id
        source_atomic_tools = source_dir / "atomic_tools.py"
        source_definitions = source_dir / "atomic_tool_definitions.json"
        if not source_atomic_tools.exists():
            raise FileNotFoundError(
                f"missing materialized atomic tool bundle for db_id={db_id!r}: {source_atomic_tools}"
            )
        if not source_definitions.exists():
            raise FileNotFoundError(
                "missing materialized atomic tool definitions for "
                f"db_id={db_id!r}: {source_definitions}"
            )
        target_dir = bundle_root / "databases" / db_id
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_atomic_tools, target_dir / "atomic_tools.py")
        shutil.copy2(source_definitions, target_dir / "atomic_tool_definitions.json")

    def _export_environment_bundle(
        self,
        bundle_root: Path,
        record: EnvironmentRegistryRecord,
    ) -> None:
        source_dir = record.filesystem_path
        target_dir = bundle_root / "environments" / record.env_id
        audit_dir = target_dir / "audit"
        target_dir.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir(parents=True, exist_ok=True)

        self._copy_required_file(source_dir / "environment.yaml", target_dir / "environment.yaml")
        self._copy_required_file(source_dir / "instances.jsonl", target_dir / "instances.jsonl")
        self._copy_required_file(
            source_dir / "canonical_answers.jsonl",
            target_dir / "canonical_answers.jsonl",
        )
        self._copy_required_file(source_dir / "solution.py", audit_dir / "solution.py")

    @staticmethod
    def _copy_required_file(source: Path, target: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(f"missing required export source file: {source}")
        shutil.copy2(source, target)
