"""Durable filesystem + sqlite registry for accepted synthesis environments."""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

import yaml

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.runtime import SynthesisEnvironmentDraft

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS environments (
        env_id TEXT PRIMARY KEY,
        db_id TEXT NOT NULL,
        domain TEXT NOT NULL,
        category TEXT NOT NULL,
        difficulty_band TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,
        generator_version TEXT NOT NULL,
        tool_signature TEXT NOT NULL,
        task_signature TEXT NOT NULL,
        verifier_signature TEXT NOT NULL,
        exact_signature TEXT NOT NULL UNIQUE,
        filesystem_path TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS coverage_counters (
        key TEXT PRIMARY KEY,
        value INTEGER NOT NULL
    )
    """,
)

INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_env_registry_db_category ON environments (db_id, category)",
    "CREATE INDEX IF NOT EXISTS idx_env_registry_exact_signature ON environments (exact_signature)",
)


class EnvironmentRegistryCommitStatus(StrEnum):
    COMMITTED = "committed"
    DUPLICATE = "duplicate"


class DifficultyBand(StrEnum):
    UNSET = "unset"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True)
class EnvironmentRegistryCommitResult:
    status: EnvironmentRegistryCommitStatus
    env_id: str
    exact_signature: str
    difficulty_band: DifficultyBand
    filesystem_path: Path
    duplicate_of_env_id: str | None = None


@dataclass(slots=True)
class EnvironmentRegistryWriter:
    root_dir: Path
    index_db_path: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._bootstrap()

    @classmethod
    def for_config(cls, config: AppConfig) -> "EnvironmentRegistryWriter":
        base_dir = config.output.traces_dir.parent
        return cls(
            root_dir=base_dir / "environments",
            index_db_path=base_dir / "environment_registry.db",
        )

    def commit_draft(
        self,
        draft: SynthesisEnvironmentDraft,
    ) -> EnvironmentRegistryCommitResult:
        env = draft.environment
        exact_signature = self._exact_signature(draft)
        difficulty_band = bucketize_difficulty_vector(env.difficulty_vector)
        existing = self._lookup_duplicate(exact_signature)
        if existing is not None:
            return EnvironmentRegistryCommitResult(
                status=EnvironmentRegistryCommitStatus.DUPLICATE,
                env_id=existing["env_id"],
                exact_signature=exact_signature,
                difficulty_band=DifficultyBand(existing["difficulty_band"]),
                filesystem_path=Path(existing["filesystem_path"]),
                duplicate_of_env_id=existing["env_id"],
            )

        temp_dir = self.root_dir / f".tmp-{env.env_id}"
        final_dir = self.root_dir / env.env_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if final_dir.exists():
            raise FileExistsError(f"environment directory already exists: {final_dir}")

        self._write_environment_bundle(temp_dir, draft, exact_signature, difficulty_band)

        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """
                    SELECT env_id, difficulty_band, filesystem_path
                    FROM environments
                    WHERE exact_signature = ?
                    """,
                    (exact_signature,),
                ).fetchone()
                if row is not None:
                    conn.rollback()
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return EnvironmentRegistryCommitResult(
                        status=EnvironmentRegistryCommitStatus.DUPLICATE,
                        env_id=row["env_id"],
                        exact_signature=exact_signature,
                        difficulty_band=DifficultyBand(row["difficulty_band"]),
                        filesystem_path=Path(row["filesystem_path"]),
                        duplicate_of_env_id=row["env_id"],
                    )

                temp_dir.rename(final_dir)
                conn.execute(
                    """
                    INSERT INTO environments (
                        env_id,
                        db_id,
                        domain,
                        category,
                        difficulty_band,
                        created_at,
                        status,
                        generator_version,
                        tool_signature,
                        task_signature,
                        verifier_signature,
                        exact_signature,
                        filesystem_path,
                        payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        env.env_id,
                        env.db_id,
                        env.domain,
                        env.category.value,
                        difficulty_band.value,
                        env.created_at.isoformat(),
                        env.status.value,
                        env.generator_version,
                        env.tool_signature,
                        env.task_signature,
                        env.verifier_signature,
                        exact_signature,
                        str(final_dir),
                        json.dumps(
                            {
                                "env_id": env.env_id,
                                "db_id": env.db_id,
                                "domain": env.domain,
                                "category": env.category.value,
                                "difficulty_band": difficulty_band.value,
                                "created_at": env.created_at.isoformat(),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
                self._increment_coverage_counter(
                    conn,
                    f"db={env.db_id}|category={env.category.value}|difficulty_band={difficulty_band.value}",
                )
                conn.commit()
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            if final_dir.exists() and not self._has_env_row(env.env_id):
                shutil.rmtree(final_dir, ignore_errors=True)
            raise

        return EnvironmentRegistryCommitResult(
            status=EnvironmentRegistryCommitStatus.COMMITTED,
            env_id=env.env_id,
            exact_signature=exact_signature,
            difficulty_band=difficulty_band,
            filesystem_path=final_dir,
        )

    def environment_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM environments").fetchone()
        return int(row["count"])

    def coverage_snapshot(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM coverage_counters").fetchall()
        return {str(row["key"]): int(row["value"]) for row in rows}

    def _lookup_duplicate(self, exact_signature: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT env_id, difficulty_band, filesystem_path
                FROM environments
                WHERE exact_signature = ?
                """,
                (exact_signature,),
            ).fetchone()

    def _has_env_row(self, env_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM environments WHERE env_id = ?",
                (env_id,),
            ).fetchone()
        return row is not None

    def _write_environment_bundle(
        self,
        directory: Path,
        draft: SynthesisEnvironmentDraft,
        exact_signature: str,
        difficulty_band: DifficultyBand,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        environment_payload = draft.environment.model_dump(mode="json")
        environment_payload["exact_signature"] = exact_signature
        environment_payload["difficulty_band"] = difficulty_band.value
        (directory / "environment.yaml").write_text(
            yaml.safe_dump(environment_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (directory / "task.json").write_text(
            draft.environment.task.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (directory / "instance_space.json").write_text(
            draft.environment.instance_space.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (directory / "cross_instance_set.json").write_text(
            draft.environment.cross_instance_set.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (directory / "tools.py").write_text(draft.artifacts.tool_source, encoding="utf-8")
        (directory / "tool_self_test.py").write_text(
            draft.artifacts.tool_self_test_source,
            encoding="utf-8",
        )
        (directory / "solution.py").write_text(
            draft.artifacts.solution_source,
            encoding="utf-8",
        )
        (directory / "verifier.py").write_text(
            draft.artifacts.verifier_source,
            encoding="utf-8",
        )
        (directory / "shadow_verifier.py").write_text(
            draft.artifacts.shadow_verifier_source,
            encoding="utf-8",
        )
        (directory / "registry_metadata.json").write_text(
            json.dumps(
                {
                    "exact_signature": exact_signature,
                    "difficulty_band": difficulty_band.value,
                    "registration_status": draft.registration_report.status.value,
                    "registration_error_codes": draft.registration_diagnostics.error_codes,
                    "self_consistency_passed": draft.self_consistency_diagnostics.passed,
                    "self_consistency_error_codes": draft.self_consistency_diagnostics.error_codes,
                    "provider_status": {
                        name: status.model_dump(mode="json")
                        for name, status in draft.provider_status.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _exact_signature(draft: SynthesisEnvironmentDraft) -> str:
        env = draft.environment
        material = "|".join(
            [
                env.db_id,
                env.category.value,
                env.tool_signature,
                env.task_signature,
                env.verifier_signature,
            ]
        )
        return f"sha256:{sha256(material.encode('utf-8')).hexdigest()}"

    def _bootstrap(self) -> None:
        with self._connect() as conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            for statement in INDEX_STATEMENTS:
                conn.execute(statement)
            conn.commit()

    def _increment_coverage_counter(self, conn: sqlite3.Connection, key: str) -> None:
        conn.execute(
            """
            INSERT INTO coverage_counters (key, value)
            VALUES (?, 1)
            ON CONFLICT(key) DO UPDATE SET value = value + 1
            """,
            (key,),
        )

    def _connect(self) -> sqlite3.Connection:
        self.index_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.index_db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn


def bucketize_difficulty_vector(difficulty_vector: dict[object, float]) -> DifficultyBand:
    if not difficulty_vector:
        return DifficultyBand.UNSET
    total = sum(float(value) for value in difficulty_vector.values())
    if total <= 3.0:
        return DifficultyBand.LOW
    if total <= 8.0:
        return DifficultyBand.MEDIUM
    return DifficultyBand.HIGH
