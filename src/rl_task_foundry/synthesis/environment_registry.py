"""Durable filesystem + sqlite registry for accepted synthesis environments."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

from datasketch import MinHash
import yaml

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.contracts import (
    CategoryTaxonomy,
    EnvironmentContract,
    EnvironmentStatus,
    OutputFieldContract,
    OutputFieldType,
)
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
        semantic_dedup_text TEXT NOT NULL DEFAULT '',
        semantic_dedup_text_version INTEGER NOT NULL DEFAULT 1,
        semantic_minhash_signature TEXT NOT NULL DEFAULT '[]',
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

MINHASH_NUM_PERM = 128
SEMANTIC_DEDUP_TEXT_VERSION = 1


class EnvironmentRegistryCommitStatus(StrEnum):
    COMMITTED = "committed"
    DUPLICATE = "duplicate"


class EnvironmentRegistryDuplicateReason(StrEnum):
    EXACT = "exact"
    MINHASH = "minhash"


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
    duplicate_reason: EnvironmentRegistryDuplicateReason | None = None
    semantic_similarity: float | None = None


@dataclass(slots=True)
class EnvironmentRegistryRecord:
    env_id: str
    db_id: str
    domain: str
    category: CategoryTaxonomy
    difficulty_band: DifficultyBand
    created_at: datetime
    status: EnvironmentStatus
    generator_version: str
    exact_signature: str
    filesystem_path: Path
    question: str | None = None


@dataclass(slots=True)
class EnvironmentRegistryCoverageEntry:
    db_id: str
    category: CategoryTaxonomy
    difficulty_band: DifficultyBand
    count: int


@dataclass(slots=True)
class SemanticDedupCandidate:
    env_id: str
    db_id: str
    domain: str
    category: CategoryTaxonomy
    difficulty_band: DifficultyBand
    question: str | None
    constraint_summaries: tuple[str, ...]
    semantic_text: str
    filesystem_path: Path


@dataclass(slots=True)
class EnvironmentRegistrySnapshot:
    environment_count: int
    coverage: list[EnvironmentRegistryCoverageEntry]
    recent_environments: list[EnvironmentRegistryRecord]


@dataclass(slots=True)
class EnvironmentRegistryWriter:
    """Durable accepted-environment registry.

    This writer is not safe for concurrent multi-writer use. The intended v1
    deployment model is a single registry writer lane that serializes commits.
    """

    root_dir: Path
    index_db_path: Path
    exact_dedup_enabled: bool = True
    near_dup_enabled: bool = True
    minhash_threshold: float = 0.9
    minhash_num_perm: int = MINHASH_NUM_PERM

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._bootstrap()

    @classmethod
    def for_config(cls, config: AppConfig) -> "EnvironmentRegistryWriter":
        base_dir = config.output.traces_dir.parent
        return cls(
            root_dir=base_dir / "environments",
            index_db_path=base_dir / "environment_registry.db",
            exact_dedup_enabled=config.dedup.exact_enabled,
            near_dup_enabled=config.dedup.near_dup_enabled,
            minhash_threshold=config.dedup.minhash_threshold,
        )

    def commit_draft(
        self,
        draft: SynthesisEnvironmentDraft,
    ) -> EnvironmentRegistryCommitResult:
        env = draft.environment
        exact_signature = self._exact_signature(draft)
        difficulty_band = bucketize_difficulty_vector(env.difficulty_vector)
        semantic_text = build_semantic_dedup_text(env)
        semantic_signature = _encode_minhash_signature(
            _build_minhash(semantic_text, num_perm=self.minhash_num_perm)
        )
        if self.exact_dedup_enabled:
            existing = self._lookup_duplicate(exact_signature)
            if existing is not None:
                return EnvironmentRegistryCommitResult(
                    status=EnvironmentRegistryCommitStatus.DUPLICATE,
                    env_id=existing["env_id"],
                    exact_signature=exact_signature,
                    difficulty_band=DifficultyBand(existing["difficulty_band"]),
                    filesystem_path=Path(existing["filesystem_path"]),
                    duplicate_of_env_id=existing["env_id"],
                    duplicate_reason=EnvironmentRegistryDuplicateReason.EXACT,
                )
        if self.near_dup_enabled:
            semantic_duplicate = self._lookup_semantic_duplicate(
                db_id=env.db_id,
                category=env.category,
                semantic_text=semantic_text,
                semantic_signature=semantic_signature,
            )
            if semantic_duplicate is not None:
                return EnvironmentRegistryCommitResult(
                    status=EnvironmentRegistryCommitStatus.DUPLICATE,
                    env_id=semantic_duplicate["env_id"],
                    exact_signature=exact_signature,
                    difficulty_band=DifficultyBand(semantic_duplicate["difficulty_band"]),
                    filesystem_path=Path(semantic_duplicate["filesystem_path"]),
                    duplicate_of_env_id=semantic_duplicate["env_id"],
                    duplicate_reason=EnvironmentRegistryDuplicateReason.MINHASH,
                    semantic_similarity=float(semantic_duplicate["semantic_similarity"]),
                )

        temp_dir = self.root_dir / f".tmp-{env.env_id}"
        final_dir = self.root_dir / env.env_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if final_dir.exists():
            raise FileExistsError(f"environment directory already exists: {final_dir}")

        self._write_environment_bundle(
            temp_dir,
            draft,
            exact_signature,
            difficulty_band,
            semantic_text,
        )

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
                        duplicate_reason=EnvironmentRegistryDuplicateReason.EXACT,
                    )
                if self.near_dup_enabled:
                    semantic_duplicate = self._lookup_semantic_duplicate(
                        db_id=env.db_id,
                        category=env.category,
                        semantic_text=semantic_text,
                        semantic_signature=semantic_signature,
                        conn=conn,
                    )
                    if semantic_duplicate is not None:
                        conn.rollback()
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return EnvironmentRegistryCommitResult(
                            status=EnvironmentRegistryCommitStatus.DUPLICATE,
                            env_id=semantic_duplicate["env_id"],
                            exact_signature=exact_signature,
                            difficulty_band=DifficultyBand(
                                semantic_duplicate["difficulty_band"]
                            ),
                            filesystem_path=Path(semantic_duplicate["filesystem_path"]),
                            duplicate_of_env_id=semantic_duplicate["env_id"],
                            duplicate_reason=EnvironmentRegistryDuplicateReason.MINHASH,
                            semantic_similarity=float(
                                semantic_duplicate["semantic_similarity"]
                            ),
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
                        semantic_dedup_text,
                        semantic_dedup_text_version,
                        semantic_minhash_signature,
                        filesystem_path,
                        payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        semantic_text,
                        SEMANTIC_DEDUP_TEXT_VERSION,
                        semantic_signature,
                        str(final_dir),
                        json.dumps(
                            self._build_registry_payload(
                                draft=draft,
                                exact_signature=exact_signature,
                                difficulty_band=difficulty_band,
                                semantic_text=semantic_text,
                            ),
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

    def environment_count(
        self,
        *,
        db_id: str | None = None,
        category: CategoryTaxonomy | None = None,
    ) -> int:
        return self._count_environments(db_id=db_id, category=category)

    def coverage_snapshot(self) -> dict[str, int]:
        return {
            (
                f"db={entry.db_id}|category={entry.category.value}"
                f"|difficulty_band={entry.difficulty_band.value}"
            ): entry.count
            for entry in self.coverage_entries()
        }

    def coverage_entries(
        self,
        *,
        db_id: str | None = None,
        category: CategoryTaxonomy | None = None,
    ) -> list[EnvironmentRegistryCoverageEntry]:
        where_sql, params = self._build_filters(db_id=db_id, category=category)
        query = (
            """
            SELECT db_id, category, difficulty_band, COUNT(*) AS count
            FROM environments
            """
            + where_sql
            + """
            GROUP BY db_id, category, difficulty_band
            ORDER BY db_id, category, difficulty_band
            """
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            EnvironmentRegistryCoverageEntry(
                db_id=str(row["db_id"]),
                category=CategoryTaxonomy(str(row["category"])),
                difficulty_band=DifficultyBand(str(row["difficulty_band"])),
                count=int(row["count"]),
            )
            for row in rows
        ]

    def list_environments(
        self,
        *,
        limit: int = 20,
        db_id: str | None = None,
        category: CategoryTaxonomy | None = None,
    ) -> list[EnvironmentRegistryRecord]:
        if limit <= 0:
            return []
        where_sql, params = self._build_filters(db_id=db_id, category=category)
        query = (
            """
            SELECT
                env_id,
                db_id,
                domain,
                category,
                difficulty_band,
                created_at,
                status,
                generator_version,
                exact_signature,
                filesystem_path,
                payload_json
            FROM environments
            """
            + where_sql
            + """
            ORDER BY created_at DESC, env_id DESC
            LIMIT ?
            """
        )
        with self._connect() as conn:
            rows = conn.execute(query, (*params, limit)).fetchall()
        records: list[EnvironmentRegistryRecord] = []
        for row in rows:
            payload = self._parse_payload(row["payload_json"])
            records.append(
                EnvironmentRegistryRecord(
                    env_id=str(row["env_id"]),
                    db_id=str(row["db_id"]),
                    domain=str(row["domain"]),
                    category=CategoryTaxonomy(str(row["category"])),
                    difficulty_band=DifficultyBand(str(row["difficulty_band"])),
                    created_at=datetime.fromisoformat(str(row["created_at"])),
                    status=EnvironmentStatus(str(row["status"])),
                    generator_version=str(row["generator_version"]),
                    exact_signature=str(row["exact_signature"]),
                    filesystem_path=Path(str(row["filesystem_path"])),
                    question=_optional_string(payload.get("question")),
                )
            )
        return records

    def semantic_dedup_candidates(
        self,
        *,
        limit: int = 20,
        db_id: str | None = None,
        category: CategoryTaxonomy | None = None,
    ) -> list[SemanticDedupCandidate]:
        if limit <= 0:
            return []
        where_sql, params = self._build_filters(db_id=db_id, category=category)
        query = (
            """
            SELECT
                env_id,
                db_id,
                domain,
                category,
                difficulty_band,
                filesystem_path,
                payload_json
            FROM environments
            """
            + where_sql
            + """
            ORDER BY created_at DESC, env_id DESC
            LIMIT ?
            """
        )
        with self._connect() as conn:
            rows = conn.execute(query, (*params, limit)).fetchall()
        candidates: list[SemanticDedupCandidate] = []
        for row in rows:
            payload = self._parse_payload(row["payload_json"])
            question = _optional_string(payload.get("question"))
            constraint_summaries = _constraint_summaries_from_payload(payload)
            semantic_text = _optional_string(payload.get("semantic_dedup_text"))
            if not semantic_text:
                semantic_text = _fallback_semantic_text(
                    category=CategoryTaxonomy(str(row["category"])),
                    question=question,
                    constraint_summaries=constraint_summaries,
                )
            candidates.append(
                SemanticDedupCandidate(
                    env_id=str(row["env_id"]),
                    db_id=str(row["db_id"]),
                    domain=str(row["domain"]),
                    category=CategoryTaxonomy(str(row["category"])),
                    difficulty_band=DifficultyBand(str(row["difficulty_band"])),
                    question=question,
                    constraint_summaries=constraint_summaries,
                    semantic_text=semantic_text,
                    filesystem_path=Path(str(row["filesystem_path"])),
                )
            )
        return candidates

    def snapshot(
        self,
        *,
        limit: int = 20,
        db_id: str | None = None,
        category: CategoryTaxonomy | None = None,
    ) -> EnvironmentRegistrySnapshot:
        return EnvironmentRegistrySnapshot(
            environment_count=self._count_environments(db_id=db_id, category=category),
            coverage=self.coverage_entries(db_id=db_id, category=category),
            recent_environments=self.list_environments(
                limit=limit,
                db_id=db_id,
                category=category,
            ),
        )

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

    def _lookup_semantic_duplicate(
        self,
        *,
        db_id: str,
        category: CategoryTaxonomy,
        semantic_text: str,
        semantic_signature: str,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, object] | None:
        best_match: dict[str, object] | None = None
        manages_connection = conn is None
        connection = conn or self._connect()
        try:
            rows = connection.execute(
                """
                SELECT
                    env_id,
                    difficulty_band,
                    filesystem_path,
                    semantic_dedup_text,
                    semantic_dedup_text_version,
                    semantic_minhash_signature,
                    payload_json
                FROM environments
                WHERE db_id = ? AND category = ?
                """,
                (db_id, category.value),
            ).fetchall()
            for row in rows:
                similarity = self._semantic_similarity_for_row(
                    row=row,
                    category=category,
                    semantic_text=semantic_text,
                    semantic_signature=semantic_signature,
                )
                if similarity < self.minhash_threshold:
                    continue
                if best_match is None or similarity > float(best_match["semantic_similarity"]):
                    best_match = {
                        "env_id": str(row["env_id"]),
                        "difficulty_band": str(row["difficulty_band"]),
                        "filesystem_path": str(row["filesystem_path"]),
                        "semantic_similarity": similarity,
                    }
            return best_match
        finally:
            if manages_connection:
                connection.close()

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
        semantic_text: str,
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
                    "semantic_dedup_text": semantic_text,
                    "semantic_dedup_text_version": SEMANTIC_DEDUP_TEXT_VERSION,
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

    def _build_registry_payload(
        self,
        *,
        draft: SynthesisEnvironmentDraft,
        exact_signature: str,
        difficulty_band: DifficultyBand,
        semantic_text: str,
    ) -> dict[str, Any]:
        env = draft.environment
        return {
            "env_id": env.env_id,
            "db_id": env.db_id,
            "domain": env.domain,
            "category": env.category.value,
            "difficulty_band": difficulty_band.value,
            "created_at": env.created_at.isoformat(),
            "status": env.status.value,
            "generator_version": env.generator_version,
            "exact_signature": exact_signature,
            "question": env.task.question,
            "difficulty_vector": {
                axis.value: float(value) for axis, value in env.difficulty_vector.items()
            },
            "constraint_summary": [
                item.model_dump(mode="json") for item in env.task.constraint_summary
            ],
            "semantic_dedup_text": semantic_text,
            "semantic_dedup_text_version": SEMANTIC_DEDUP_TEXT_VERSION,
        }

    def _semantic_similarity_for_row(
        self,
        *,
        row: sqlite3.Row,
        category: CategoryTaxonomy,
        semantic_text: str,
        semantic_signature: str,
    ) -> float:
        signature = _optional_string(row["semantic_minhash_signature"])
        version = row["semantic_dedup_text_version"]
        if (
            signature
            and int(version) == SEMANTIC_DEDUP_TEXT_VERSION
            and _signature_length(signature) == self.minhash_num_perm
        ):
            return _estimate_signature_similarity(semantic_signature, signature)
        payload = self._parse_payload(row["payload_json"])
        existing_text = _optional_string(row["semantic_dedup_text"]) or _optional_string(
            payload.get("semantic_dedup_text")
        )
        if not existing_text:
            existing_text = _fallback_semantic_text(
                category=category,
                question=_optional_string(payload.get("question")),
                constraint_summaries=_constraint_summaries_from_payload(payload),
            )
        return estimate_semantic_similarity(
            semantic_text,
            existing_text,
            num_perm=self.minhash_num_perm,
        )

    def _count_environments(
        self,
        *,
        db_id: str | None,
        category: CategoryTaxonomy | None,
    ) -> int:
        where_sql, params = self._build_filters(db_id=db_id, category=category)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM environments " + where_sql,
                params,
            ).fetchone()
        return int(row["count"])

    @staticmethod
    def _build_filters(
        *,
        db_id: str | None,
        category: CategoryTaxonomy | None,
    ) -> tuple[str, tuple[object, ...]]:
        clauses: list[str] = []
        params: list[object] = []
        if db_id is not None:
            clauses.append("db_id = ?")
            params.append(db_id)
        if category is not None:
            clauses.append("category = ?")
            params.append(category.value)
        if not clauses:
            return "", ()
        return "WHERE " + " AND ".join(clauses), tuple(params)

    @staticmethod
    def _parse_payload(raw: object) -> dict[str, Any]:
        if not isinstance(raw, str):
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    def _bootstrap(self) -> None:
        with self._connect() as conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            self._ensure_environment_column(
                conn,
                "semantic_dedup_text",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_environment_column(
                conn,
                "semantic_dedup_text_version",
                f"INTEGER NOT NULL DEFAULT {SEMANTIC_DEDUP_TEXT_VERSION}",
            )
            self._ensure_environment_column(
                conn,
                "semantic_minhash_signature",
                "TEXT NOT NULL DEFAULT '[]'",
            )
            for statement in INDEX_STATEMENTS:
                conn.execute(statement)
            conn.commit()

    @staticmethod
    def _ensure_environment_column(
        conn: sqlite3.Connection,
        column_name: str,
        definition_sql: str,
    ) -> None:
        columns = conn.execute("PRAGMA table_info(environments)").fetchall()
        if any(str(row["name"]) == column_name for row in columns):
            return
        conn.execute(
            f"ALTER TABLE environments ADD COLUMN {column_name} {definition_sql}"
        )

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


def build_semantic_dedup_text(environment: EnvironmentContract) -> str:
    task = environment.task
    output_shape = ", ".join(_flatten_output_schema(task.output_schema.root))
    constraints = " | ".join(
        f"{item.kind.value}:{item.key}:{item.summary}" for item in task.constraint_summary
    )
    difficulty = ", ".join(
        f"{axis.value}={float(value):g}"
        for axis, value in sorted(
            task.difficulty_vector.items(),
            key=lambda item: item[0].value,
        )
    )
    lines = [
        f"db_id:{environment.db_id}",
        f"domain:{environment.domain}",
        f"category:{task.category.value}",
        f"question:{task.question}",
        f"output_schema:{output_shape or '<empty>'}",
        f"constraints:{constraints or '<none>'}",
        f"difficulty:{difficulty or '<unset>'}",
    ]
    return "\n".join(lines)


def estimate_semantic_similarity(
    left_text: str,
    right_text: str,
    *,
    num_perm: int = MINHASH_NUM_PERM,
) -> float:
    return _estimate_signature_similarity(
        _encode_minhash_signature(_build_minhash(left_text, num_perm=num_perm)),
        _encode_minhash_signature(_build_minhash(right_text, num_perm=num_perm)),
    )


def _flatten_output_schema(field: OutputFieldContract, *, prefix: str = "") -> list[str]:
    path = f"{prefix}.{field.name}" if prefix else field.name
    if field.type == OutputFieldType.OBJECT:
        if not field.fields:
            return [f"{path}:object"]
        values: list[str] = []
        for child in field.fields:
            values.extend(_flatten_output_schema(child, prefix=path))
        return values
    if field.type == OutputFieldType.LIST:
        if field.items is None:
            return [f"{path}[]:list"]
        return _flatten_output_schema(field.items, prefix=f"{path}[]")
    return [f"{path}:{field.type.value}"]


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _constraint_summaries_from_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    raw = payload.get("constraint_summary")
    if not isinstance(raw, list):
        return ()
    summaries: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary")
        if isinstance(summary, str) and summary:
            summaries.append(summary)
    return tuple(summaries)


def _fallback_semantic_text(
    *,
    category: CategoryTaxonomy,
    question: str | None,
    constraint_summaries: tuple[str, ...],
) -> str:
    parts = [
        f"category:{category.value}",
        f"question:{question or '<missing>'}",
        "constraints:" + (" | ".join(constraint_summaries) or "<none>"),
    ]
    return "\n".join(parts)


def _build_minhash(text: str, *, num_perm: int) -> MinHash:
    minhash = MinHash(num_perm=num_perm)
    for token in _semantic_shingles(text):
        minhash.update(token.encode("utf-8"))
    return minhash


def _encode_minhash_signature(minhash: MinHash) -> str:
    return json.dumps(minhash.hashvalues.tolist(), separators=(",", ":"))


def _decode_minhash_signature(signature: str) -> tuple[int, ...]:
    try:
        values = json.loads(signature)
    except json.JSONDecodeError:
        return ()
    if not isinstance(values, list):
        return ()
    decoded: list[int] = []
    for value in values:
        if not isinstance(value, int):
            return ()
        decoded.append(value)
    return tuple(decoded)


def _estimate_signature_similarity(left_signature: str, right_signature: str) -> float:
    left = _decode_minhash_signature(left_signature)
    right = _decode_minhash_signature(right_signature)
    if not left or len(left) != len(right):
        return 0.0
    matches = sum(1 for left_value, right_value in zip(left, right, strict=False) if left_value == right_value)
    return matches / len(left)


def _signature_length(signature: str) -> int:
    return len(_decode_minhash_signature(signature))


def _semantic_shingles(text: str) -> tuple[str, ...]:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return ("<empty>",)
    tokens = re.findall(r"\w+", normalized)
    if not tokens:
        return (normalized,)
    if len(tokens) < 3:
        return tuple(tokens)
    return tuple(" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2))
