"""Durable filesystem + sqlite registry for accepted task bundles."""

from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasketch import MinHash, MinHashLSH

from rl_task_foundry.config.models import AppConfig, TaskRegistryConfig
from rl_task_foundry.synthesis.atomic_tool_materializer import AtomicToolMaterializer
from rl_task_foundry.synthesis.contracts import (
    DifficultyVectorContract,
    OutputFieldContract,
    OutputFieldType,
    TaskBundleContract,
    TaskBundleStatus,
    TopicName,
    difficulty_vector_json,
    flatten_difficulty_vector,
    normalize_topic,
)
from rl_task_foundry.synthesis.runtime import SynthesisTaskDraft

logger = logging.getLogger(__name__)

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        db_id TEXT NOT NULL,
        domain TEXT NOT NULL,
        topic TEXT NOT NULL,
        difficulty_band TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,
        generator_version TEXT NOT NULL,
        tool_signature TEXT NOT NULL,
        task_signature TEXT NOT NULL,
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
    "CREATE INDEX IF NOT EXISTS idx_task_registry_db_topic ON tasks (db_id, topic)",
    "CREATE INDEX IF NOT EXISTS idx_task_registry_exact_signature ON tasks (exact_signature)",
)

SEMANTIC_DEDUP_TEXT_VERSION = 1


class TaskRegistryCommitStatus(StrEnum):
    COMMITTED = "committed"
    DUPLICATE = "duplicate"


class TaskRegistryDuplicateReason(StrEnum):
    EXACT = "exact"
    MINHASH = "minhash"


class DifficultyBand(StrEnum):
    UNSET = "unset"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True)
class TaskRegistryCommitResult:
    status: TaskRegistryCommitStatus
    task_id: str
    exact_signature: str
    difficulty_band: DifficultyBand
    filesystem_path: Path
    duplicate_of_task_id: str | None = None
    duplicate_reason: TaskRegistryDuplicateReason | None = None
    semantic_similarity: float | None = None


@dataclass(init=False, slots=True)
class TaskRegistryRecord:
    task_id: str
    db_id: str
    domain: str
    topic: str
    difficulty_band: DifficultyBand
    created_at: datetime
    status: TaskBundleStatus
    generator_version: str
    exact_signature: str
    filesystem_path: Path
    question: str | None = None

    def __init__(
        self,
        *,
        task_id: str,
        db_id: str,
        domain: str,
        topic: str | None = None,
        category: object | None = None,
        difficulty_band: DifficultyBand,
        created_at: datetime,
        status: TaskBundleStatus,
        generator_version: str,
        exact_signature: str,
        filesystem_path: Path,
        question: str | None = None,
    ) -> None:
        resolved_topic = topic if topic is not None else category
        self.task_id = task_id
        self.db_id = db_id
        self.domain = domain
        self.topic = normalize_topic(resolved_topic)
        self.difficulty_band = difficulty_band
        self.created_at = created_at
        self.status = status
        self.generator_version = generator_version
        self.exact_signature = exact_signature
        self.filesystem_path = filesystem_path
        self.question = question

    @property
    def category(self) -> TopicName:
        return TopicName(self.topic)


@dataclass(init=False, slots=True)
class TaskRegistryCoverageEntry:
    db_id: str
    topic: str
    difficulty_band: DifficultyBand
    count: int

    def __init__(
        self,
        *,
        db_id: str,
        topic: str | None = None,
        category: object | None = None,
        difficulty_band: DifficultyBand,
        count: int,
    ) -> None:
        resolved_topic = topic if topic is not None else category
        self.db_id = db_id
        self.topic = normalize_topic(resolved_topic)
        self.difficulty_band = difficulty_band
        self.count = count

    @property
    def category(self) -> TopicName:
        return TopicName(self.topic)


@dataclass(init=False, slots=True)
class SemanticDedupCandidate:
    task_id: str
    db_id: str
    domain: str
    topic: str
    difficulty_band: DifficultyBand
    question: str | None
    constraint_summaries: tuple[str, ...]
    semantic_text: str
    filesystem_path: Path

    def __init__(
        self,
        *,
        task_id: str,
        db_id: str,
        domain: str,
        topic: str | None = None,
        category: object | None = None,
        difficulty_band: DifficultyBand,
        question: str | None,
        constraint_summaries: tuple[str, ...],
        semantic_text: str,
        filesystem_path: Path,
    ) -> None:
        resolved_topic = topic if topic is not None else category
        self.task_id = task_id
        self.db_id = db_id
        self.domain = domain
        self.topic = normalize_topic(resolved_topic)
        self.difficulty_band = difficulty_band
        self.question = question
        self.constraint_summaries = constraint_summaries
        self.semantic_text = semantic_text
        self.filesystem_path = filesystem_path

    @property
    def category(self) -> TopicName:
        return TopicName(self.topic)


@dataclass(slots=True)
class TaskRegistrySnapshot:
    task_count: int
    coverage: list[TaskRegistryCoverageEntry]
    recent_tasks: list[TaskRegistryRecord]


@dataclass(slots=True)
class _SemanticScopeIndex:
    lsh: MinHashLSH
    metadata_by_task_id: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass(slots=True)
class TaskRegistryWriter:
    """Durable accepted-task registry.

    This writer is not safe for concurrent multi-writer use. The intended v1
    deployment model is a single registry writer lane that serializes commits.
    """

    root_dir: Path
    index_db_path: Path
    exact_dedup_enabled: bool = True
    near_dup_enabled: bool = True
    minhash_threshold: float = 0.9
    registry_config: TaskRegistryConfig = field(default_factory=TaskRegistryConfig)
    _semantic_scope_indexes: dict[tuple[str, str], _SemanticScopeIndex] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    atomic_tool_materializer: AtomicToolMaterializer | None = field(
        default=None,
        repr=False,
    )
    _connection: sqlite3.Connection | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if self.atomic_tool_materializer is None:
            self.atomic_tool_materializer = AtomicToolMaterializer(
                root_dir=self.root_dir.parent / "databases"
            )
        self._bootstrap()

    @classmethod
    def for_config(cls, config: AppConfig) -> "TaskRegistryWriter":
        base_dir = config.output.traces_dir.parent
        return cls(
            root_dir=base_dir / "tasks",
            index_db_path=base_dir / "task_registry.db",
            exact_dedup_enabled=config.dedup.exact_enabled,
            near_dup_enabled=config.dedup.near_dup_enabled,
            minhash_threshold=config.dedup.minhash_threshold,
            registry_config=config.task_registry,
        )

    def commit_draft(
        self,
        draft: SynthesisTaskDraft,
    ) -> TaskRegistryCommitResult:
        task_bundle = draft.task_bundle
        exact_signature = self._exact_signature(draft)
        difficulty_band = bucketize_difficulty_vector(
            task_bundle.difficulty_vector,
            registry_config=self.registry_config,
        )
        semantic_text = build_semantic_dedup_text(task_bundle)
        semantic_signature = _encode_minhash_signature(
            _build_minhash(
                semantic_text,
                num_perm=self.registry_config.minhash_num_perm,
                shingle_size=self.registry_config.semantic_shingle_size,
            )
        )

        temp_dir = self.root_dir / f".tmp-{task_bundle.task_id}"
        final_dir = self.root_dir / task_bundle.task_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if final_dir.exists():
            raise FileExistsError(f"task directory already exists: {final_dir}")

        self._write_task_bundle(
            temp_dir,
            draft,
            exact_signature,
            difficulty_band,
            semantic_text,
        )
        assert self.atomic_tool_materializer is not None
        self.atomic_tool_materializer.materialize_bundle(draft.atomic_tool_bundle)

        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """
                    SELECT task_id, difficulty_band, filesystem_path
                    FROM tasks
                    WHERE exact_signature = ?
                    """,
                    (exact_signature,),
                ).fetchone()
                if row is not None:
                    conn.rollback()
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return TaskRegistryCommitResult(
                        status=TaskRegistryCommitStatus.DUPLICATE,
                        task_id=row["task_id"],
                        exact_signature=exact_signature,
                        difficulty_band=DifficultyBand(row["difficulty_band"]),
                        filesystem_path=Path(row["filesystem_path"]),
                        duplicate_of_task_id=row["task_id"],
                        duplicate_reason=TaskRegistryDuplicateReason.EXACT,
                    )
                if self.near_dup_enabled:
                    semantic_duplicate = self._lookup_semantic_duplicate(
                        db_id=task_bundle.db_id,
                        topic=task_bundle.topic,
                        semantic_text=semantic_text,
                        semantic_signature=semantic_signature,
                        conn=conn,
                    )
                    if semantic_duplicate is not None:
                        conn.rollback()
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return TaskRegistryCommitResult(
                            status=TaskRegistryCommitStatus.DUPLICATE,
                            task_id=str(semantic_duplicate["task_id"]),
                            exact_signature=exact_signature,
                            difficulty_band=DifficultyBand(str(semantic_duplicate["difficulty_band"])),
                            filesystem_path=Path(str(semantic_duplicate["filesystem_path"])),
                            duplicate_of_task_id=str(semantic_duplicate["task_id"]),
                            duplicate_reason=TaskRegistryDuplicateReason.MINHASH,
                            semantic_similarity=float(str(semantic_duplicate["semantic_similarity"])),
                        )

                temp_dir.rename(final_dir)
                conn.execute(
                    """
                    INSERT INTO tasks (
                        task_id,
                        db_id,
                        domain,
                        topic,
                        difficulty_band,
                        created_at,
                        status,
                        generator_version,
                        tool_signature,
                        task_signature,
                        exact_signature,
                        semantic_dedup_text,
                        semantic_dedup_text_version,
                        semantic_minhash_signature,
                        filesystem_path,
                        payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_bundle.task_id,
                        task_bundle.db_id,
                        task_bundle.domain,
                        task_bundle.topic,
                        difficulty_band.value,
                        task_bundle.created_at.isoformat(),
                        task_bundle.status.value,
                        task_bundle.generator_version,
                        task_bundle.tool_signature,
                        task_bundle.task_signature,
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
                    f"db={task_bundle.db_id}|topic={task_bundle.topic}|difficulty_band={difficulty_band.value}",
                )
                conn.commit()
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            if final_dir.exists():
                try:
                    has_task_row = self._has_task_row(task_bundle.task_id)
                except Exception:
                    logger.warning(
                        "Task registry cleanup could not verify task row existence for %s",
                        task_bundle.task_id,
                        exc_info=True,
                    )
                    has_task_row = True
                if not has_task_row:
                    shutil.rmtree(final_dir, ignore_errors=True)
            raise

        if self.near_dup_enabled:
            self._register_semantic_scope_entry(
                db_id=task_bundle.db_id,
                topic=task_bundle.topic,
                task_id=task_bundle.task_id,
                difficulty_band=difficulty_band,
                filesystem_path=final_dir,
                semantic_signature=semantic_signature,
            )
        return TaskRegistryCommitResult(
            status=TaskRegistryCommitStatus.COMMITTED,
            task_id=task_bundle.task_id,
            exact_signature=exact_signature,
            difficulty_band=difficulty_band,
            filesystem_path=final_dir,
        )

    def task_count(
        self,
        *,
        db_id: str | None = None,
        topic: str | None = None,
    ) -> int:
        return self._count_tasks(db_id=db_id, topic=topic)

    def coverage_snapshot(self) -> dict[str, int]:
        return {
            (
                f"db={entry.db_id}|topic={entry.topic}"
                f"|difficulty_band={entry.difficulty_band.value}"
            ): entry.count
            for entry in self.coverage_entries()
        }

    def coverage_entries(
        self,
        *,
        db_id: str | None = None,
        topic: str | None = None,
    ) -> list[TaskRegistryCoverageEntry]:
        where_sql, params = self._build_filters(db_id=db_id, topic=topic)
        query = (
            """
            SELECT db_id, topic, difficulty_band, COUNT(*) AS count
            FROM tasks
            """
            + where_sql
            + """
            GROUP BY db_id, topic, difficulty_band
            ORDER BY db_id, topic, difficulty_band
            """
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            TaskRegistryCoverageEntry(
                db_id=str(row["db_id"]),
                topic=normalize_topic(str(row["topic"])),
                difficulty_band=DifficultyBand(str(row["difficulty_band"])),
                count=int(row["count"]),
            )
            for row in rows
        ]

    def list_tasks(
        self,
        *,
        limit: int | None = None,
        db_id: str | None = None,
        topic: str | None = None,
    ) -> list[TaskRegistryRecord]:
        resolved_limit = self.registry_config.default_query_limit if limit is None else limit
        if resolved_limit <= 0:
            return []
        where_sql, params = self._build_filters(db_id=db_id, topic=topic)
        query = (
            """
            SELECT
                task_id,
                db_id,
                domain,
                topic,
                difficulty_band,
                created_at,
                status,
                generator_version,
                exact_signature,
                filesystem_path,
                payload_json
            FROM tasks
            """
            + where_sql
            + """
            ORDER BY created_at DESC, task_id DESC
            LIMIT ?
            """
        )
        with self._connect() as conn:
            rows = conn.execute(query, (*params, resolved_limit)).fetchall()
        records: list[TaskRegistryRecord] = []
        for row in rows:
            payload = self._parse_payload(row["payload_json"])
            records.append(
                TaskRegistryRecord(
                    task_id=str(row["task_id"]),
                    db_id=str(row["db_id"]),
                    domain=str(row["domain"]),
                    topic=normalize_topic(str(row["topic"])),
                    difficulty_band=DifficultyBand(str(row["difficulty_band"])),
                    created_at=datetime.fromisoformat(str(row["created_at"])),
                    status=TaskBundleStatus(str(row["status"])),
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
        limit: int | None = None,
        db_id: str | None = None,
        topic: str | None = None,
    ) -> list[SemanticDedupCandidate]:
        resolved_limit = self.registry_config.default_query_limit if limit is None else limit
        if resolved_limit <= 0:
            return []
        where_sql, params = self._build_filters(db_id=db_id, topic=topic)
        query = (
            """
            SELECT
                task_id,
                db_id,
                domain,
                topic,
                difficulty_band,
                filesystem_path,
                payload_json
            FROM tasks
            """
            + where_sql
            + """
            ORDER BY created_at DESC, task_id DESC
            LIMIT ?
            """
        )
        with self._connect() as conn:
            rows = conn.execute(query, (*params, resolved_limit)).fetchall()
        candidates: list[SemanticDedupCandidate] = []
        for row in rows:
            payload = self._parse_payload(row["payload_json"])
            question = _optional_string(payload.get("question"))
            constraint_summaries = _constraint_summaries_from_payload(payload)
            semantic_text = _optional_string(payload.get("semantic_dedup_text"))
            if not semantic_text:
                semantic_text = _fallback_semantic_text(
                    topic=normalize_topic(str(row["topic"])),
                    question=question,
                    constraint_summaries=constraint_summaries,
                )
            candidates.append(
                SemanticDedupCandidate(
                    task_id=str(row["task_id"]),
                    db_id=str(row["db_id"]),
                    domain=str(row["domain"]),
                    topic=normalize_topic(str(row["category"])),
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
        limit: int | None = None,
        db_id: str | None = None,
        topic: str | None = None,
    ) -> TaskRegistrySnapshot:
        return TaskRegistrySnapshot(
            task_count=self._count_tasks(db_id=db_id, topic=topic),
            coverage=self.coverage_entries(db_id=db_id, topic=topic),
            recent_tasks=self.list_tasks(
                limit=limit,
                db_id=db_id,
                topic=topic,
            ),
        )

    def _lookup_semantic_duplicate(
        self,
        *,
        db_id: str,
        topic: str,
        semantic_text: str,
        semantic_signature: str,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, object] | None:
        target = _decode_minhash_to_minhash(
            semantic_signature,
            num_perm=self.registry_config.minhash_num_perm,
        )
        if target is None:
            return None
        scope = self._semantic_scope_index(
            db_id=db_id,
            topic=topic,
            conn=conn,
        )
        candidate_ids = scope.lsh.query(target)
        best_match: dict[str, object] | None = None
        for task_id in candidate_ids:
            metadata = scope.metadata_by_task_id.get(str(task_id))
            if metadata is None:
                continue
            similarity = _estimate_signature_similarity(
                semantic_signature,
                str(metadata["semantic_signature"]),
            )
            if similarity < self.minhash_threshold:
                continue
            if best_match is None or similarity > float(str(best_match["semantic_similarity"])):
                best_match = {
                    "task_id": str(task_id),
                    "difficulty_band": str(metadata["difficulty_band"]),
                    "filesystem_path": str(metadata["filesystem_path"]),
                    "semantic_similarity": similarity,
                }
        return best_match

    def _has_task_row(self, task_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return row is not None

    def _write_task_bundle(
        self,
        directory: Path,
        draft: SynthesisTaskDraft,
        exact_signature: str,
        difficulty_band: DifficultyBand,
        semantic_text: str,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        task_bundle_payload = draft.task_bundle.model_dump(mode="json")
        task_bundle_payload["exact_signature"] = exact_signature
        task_bundle_payload["difficulty_band"] = difficulty_band.value
        (directory / "task.yaml").write_text(
            yaml.safe_dump(task_bundle_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (directory / "task.json").write_text(
            draft.task_bundle.task.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (directory / "instance.json").write_text(
            json.dumps(
                {
                    "rendered_user_prompt": draft.rendered_user_prompt,
                    "anchor_entity": draft.anchor_entity,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (directory / "canonical_answer.json").write_text(
            json.dumps(
                {
                    "canonical_answer_json": draft.canonical_answer_json,
                    "label_signature": draft.label_signature,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (directory / "tools.py").write_text(
            draft.atomic_tool_bundle.source,
            encoding="utf-8",
        )
        (directory / "registry_metadata.json").write_text(
            json.dumps(
                {
                    "exact_signature": exact_signature,
                    "difficulty_band": difficulty_band.value,
                    "semantic_dedup_text": semantic_text,
                    "semantic_dedup_text_version": SEMANTIC_DEDUP_TEXT_VERSION,
                    "generation_attempts": [
                        attempt.model_dump(mode="json") for attempt in draft.generation_attempts
                    ],
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
    def _exact_signature(draft: SynthesisTaskDraft) -> str:
        task_bundle = draft.task_bundle
        material = "|".join(
            [
                task_bundle.db_id,
                task_bundle.topic,
                task_bundle.tool_signature,
                task_bundle.task_signature,
            ]
        )
        return f"sha256:{sha256(material.encode('utf-8')).hexdigest()}"

    def _build_registry_payload(
        self,
        *,
        draft: SynthesisTaskDraft,
        exact_signature: str,
        difficulty_band: DifficultyBand,
        semantic_text: str,
    ) -> dict[str, Any]:
        task_bundle = draft.task_bundle
        return {
            "task_id": task_bundle.task_id,
            "db_id": task_bundle.db_id,
            "domain": task_bundle.domain,
            "topic": task_bundle.topic,
            "difficulty_band": difficulty_band.value,
            "created_at": task_bundle.created_at.isoformat(),
            "status": task_bundle.status.value,
            "generator_version": task_bundle.generator_version,
            "exact_signature": exact_signature,
            "question": task_bundle.task.question,
            "difficulty_vector": difficulty_vector_json(task_bundle.difficulty_vector),
            "constraint_summary": [
                item.model_dump(mode="json") for item in task_bundle.task.constraint_summary
            ],
            "semantic_dedup_text": semantic_text,
            "semantic_dedup_text_version": SEMANTIC_DEDUP_TEXT_VERSION,
        }

    def _semantic_scope_index(
        self,
        *,
        db_id: str,
        topic: str,
        conn: sqlite3.Connection | None = None,
    ) -> _SemanticScopeIndex:
        normalized_topic = normalize_topic(topic)
        scope_key = (db_id, normalized_topic)
        existing = self._semantic_scope_indexes.get(scope_key)
        if existing is not None:
            return existing
        lsh = MinHashLSH(
            threshold=self.minhash_threshold,
            num_perm=self.registry_config.minhash_num_perm,
        )
        metadata_by_task_id: dict[str, dict[str, object]] = {}
        connection = conn or self._connect()
        rows = connection.execute(
            """
            SELECT
                task_id,
                difficulty_band,
                filesystem_path,
                semantic_dedup_text,
                semantic_dedup_text_version,
                semantic_minhash_signature,
                payload_json
            FROM tasks
            WHERE db_id = ? AND topic = ?
            """,
            (db_id, normalized_topic),
        ).fetchall()
        for row in rows:
            task_id = str(row["task_id"])
            signature = self._semantic_signature_for_row(row=row, topic=normalized_topic)
            minhash = _decode_minhash_to_minhash(
                signature,
                num_perm=self.registry_config.minhash_num_perm,
            )
            if minhash is None:
                continue
            lsh.insert(task_id, minhash, check_duplication=False)
            metadata_by_task_id[task_id] = {
                "difficulty_band": str(row["difficulty_band"]),
                "filesystem_path": str(row["filesystem_path"]),
                "semantic_signature": signature,
            }
        scope = _SemanticScopeIndex(lsh=lsh, metadata_by_task_id=metadata_by_task_id)
        self._semantic_scope_indexes[scope_key] = scope
        return scope

    def _register_semantic_scope_entry(
        self,
        *,
        db_id: str,
        topic: str,
        task_id: str,
        difficulty_band: DifficultyBand,
        filesystem_path: Path,
        semantic_signature: str,
    ) -> None:
        scope_key = (db_id, normalize_topic(topic))
        scope = self._semantic_scope_indexes.get(scope_key)
        if scope is None:
            return
        minhash = _decode_minhash_to_minhash(
            semantic_signature,
            num_perm=self.registry_config.minhash_num_perm,
        )
        if minhash is None:
            return
        scope.lsh.insert(task_id, minhash, check_duplication=False)
        scope.metadata_by_task_id[task_id] = {
            "difficulty_band": difficulty_band.value,
            "filesystem_path": str(filesystem_path),
            "semantic_signature": semantic_signature,
        }

    def _semantic_signature_for_row(
        self,
        *,
        row: sqlite3.Row,
        topic: str,
    ) -> str:
        signature = _optional_string(row["semantic_minhash_signature"])
        version = row["semantic_dedup_text_version"]
        if (
            signature
            and int(version) == SEMANTIC_DEDUP_TEXT_VERSION
            and _signature_length(signature) == self.registry_config.minhash_num_perm
        ):
            return signature
        payload = self._parse_payload(row["payload_json"])
        existing_text = _optional_string(row["semantic_dedup_text"]) or _optional_string(
            payload.get("semantic_dedup_text")
        )
        if not existing_text:
            existing_text = _fallback_semantic_text(
                topic=topic,
                question=_optional_string(payload.get("question")),
                constraint_summaries=_constraint_summaries_from_payload(payload),
            )
        return _encode_minhash_signature(
            _build_minhash(
                existing_text,
                num_perm=self.registry_config.minhash_num_perm,
                shingle_size=self.registry_config.semantic_shingle_size,
            )
        )

    def _count_tasks(
        self,
        *,
        db_id: str | None,
        topic: str | None,
    ) -> int:
        where_sql, params = self._build_filters(db_id=db_id, topic=topic)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM tasks " + where_sql,
                params,
            ).fetchone()
        return int(row["count"])

    @staticmethod
    def _build_filters(
        *,
        db_id: str | None,
        topic: str | None,
    ) -> tuple[str, tuple[object, ...]]:
        clauses: list[str] = []
        params: list[object] = []
        if db_id is not None:
            clauses.append("db_id = ?")
            params.append(db_id)
        if topic is not None:
            clauses.append("topic = ?")
            params.append(normalize_topic(topic))
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
            self._migrate_legacy_task_table(conn)
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            self._ensure_task_column(
                conn,
                "semantic_dedup_text",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_task_column(
                conn,
                "semantic_dedup_text_version",
                f"INTEGER NOT NULL DEFAULT {SEMANTIC_DEDUP_TEXT_VERSION}",
            )
            self._ensure_task_column(
                conn,
                "semantic_minhash_signature",
                "TEXT NOT NULL DEFAULT '[]'",
            )
            for statement in INDEX_STATEMENTS:
                conn.execute(statement)
            conn.commit()

    @staticmethod
    def _migrate_legacy_task_table(conn: sqlite3.Connection) -> None:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('tasks', 'environments')"  # noqa: E501
        ).fetchall()
        if not tables:
            return
        if any(str(row["name"]) == "tasks" for row in tables):
            columns = conn.execute("PRAGMA table_info(tasks)").fetchall()
            column_names = {str(row["name"]) for row in columns}
            if "topic" in column_names and "task_id" in column_names:
                return
            return
        columns = conn.execute("PRAGMA table_info(environments)").fetchall()
        column_names = {str(row["name"]) for row in columns}
        conn.execute("DROP INDEX IF EXISTS idx_env_registry_db_category")
        conn.execute("DROP INDEX IF EXISTS idx_env_registry_exact_signature")
        conn.execute(
            """
            CREATE TABLE tasks_v2 (
                task_id TEXT PRIMARY KEY,
                db_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                topic TEXT NOT NULL,
                difficulty_band TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                generator_version TEXT NOT NULL,
                tool_signature TEXT NOT NULL,
                task_signature TEXT NOT NULL,
                exact_signature TEXT NOT NULL UNIQUE,
                semantic_dedup_text TEXT NOT NULL DEFAULT '',
                semantic_dedup_text_version INTEGER NOT NULL DEFAULT 1,
                semantic_minhash_signature TEXT NOT NULL DEFAULT '[]',
                filesystem_path TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        legacy_task_id_column = "env" + "_id"
        conn.execute(
            f"""
            INSERT INTO tasks_v2 (
                task_id,
                db_id,
                domain,
                topic,
                difficulty_band,
                created_at,
                status,
                generator_version,
                tool_signature,
                task_signature,
                exact_signature,
                semantic_dedup_text,
                semantic_dedup_text_version,
                semantic_minhash_signature,
                filesystem_path,
                payload_json
            )
            SELECT
                {legacy_task_id_column},
                db_id,
                domain,
                category,
                difficulty_band,
                created_at,
                status,
                generator_version,
                tool_signature,
                task_signature,
                exact_signature,
                semantic_dedup_text,
                semantic_dedup_text_version,
                semantic_minhash_signature,
                filesystem_path,
                payload_json
            FROM environments
            """
        )
        conn.execute("DROP TABLE environments")
        conn.execute("ALTER TABLE tasks_v2 RENAME TO tasks")

    @staticmethod
    def _ensure_task_column(
        conn: sqlite3.Connection,
        column_name: str,
        definition_sql: str,
    ) -> None:
        columns = conn.execute("PRAGMA table_info(tasks)").fetchall()
        if any(str(row["name"]) == column_name for row in columns):
            return
        conn.execute(f"ALTER TABLE tasks ADD COLUMN {column_name} {definition_sql}")

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
        if self._connection is None:
            self.index_db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.index_db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            self._connection = conn
        return self._connection

    def close(self) -> None:
        self._semantic_scope_indexes.clear()
        if self._connection is None:
            return
        self._connection.close()
        self._connection = None


def bucketize_difficulty_vector(
    difficulty_vector: object,
    *,
    registry_config: TaskRegistryConfig,
) -> DifficultyBand:
    if difficulty_vector is None:
        return DifficultyBand.UNSET
    if isinstance(difficulty_vector, DifficultyVectorContract):
        total = sum(float(value) for value in flatten_difficulty_vector(difficulty_vector).values())
    elif isinstance(difficulty_vector, dict):
        total = sum(float(value) for value in difficulty_vector.values())
    else:
        return DifficultyBand.UNSET
    if total == 0.0:
        return DifficultyBand.UNSET
    if total <= registry_config.difficulty_band_low_max_total:
        return DifficultyBand.LOW
    if total <= registry_config.difficulty_band_medium_max_total:
        return DifficultyBand.MEDIUM
    return DifficultyBand.HIGH


def build_semantic_dedup_text(task_bundle: TaskBundleContract) -> str:
    task = task_bundle.task
    output_shape = ", ".join(_flatten_output_schema(task.output_schema.root))
    constraints = " | ".join(
        f"{item.kind.value}:{item.key}:{item.summary}" for item in task.constraint_summary
    )
    difficulty = ", ".join(
        f"{metric.value}={float(value):g}"
        for metric, value in flatten_difficulty_vector(task.difficulty_vector).items()
        if value > 0.0
    )
    lines = [
        f"db_id:{task_bundle.db_id}",
        f"domain:{task_bundle.domain}",
        f"topic:{task.topic}",
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
    num_perm: int,
    shingle_size: int,
) -> float:
    return _build_minhash(
        left_text,
        num_perm=num_perm,
        shingle_size=shingle_size,
    ).jaccard(
        _build_minhash(
            right_text,
            num_perm=num_perm,
            shingle_size=shingle_size,
        )
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
    topic: str,
    question: str | None,
    constraint_summaries: tuple[str, ...],
) -> str:
    parts = [
        f"topic:{normalize_topic(topic)}",
        f"question:{question or '<missing>'}",
        "constraints:" + (" | ".join(constraint_summaries) or "<none>"),
    ]
    return "\n".join(parts)


def _build_minhash(text: str, *, num_perm: int, shingle_size: int) -> MinHash:
    minhash = MinHash(num_perm=num_perm)
    for token in _semantic_shingles(text, shingle_size=shingle_size):
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
    matches = sum(
        1 for left_value, right_value in zip(left, right, strict=True) if left_value == right_value
    )
    return matches / len(left)


def _signature_length(signature: str) -> int:
    return len(_decode_minhash_signature(signature))


def _decode_minhash_to_minhash(signature: str, *, num_perm: int) -> MinHash | None:
    values = _decode_minhash_signature(signature)
    if len(values) != num_perm:
        return None
    minhash = MinHash(num_perm=num_perm)
    minhash.hashvalues = np.array(values, dtype=np.uint64)
    return minhash


def _semantic_shingles(text: str, *, shingle_size: int) -> tuple[str, ...]:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return ("<empty>",)
    tokens = re.findall(r"\w+", normalized)
    if not tokens:
        return (normalized,)
    if len(tokens) < shingle_size:
        return tuple(tokens)
    return tuple(
        " ".join(tokens[index : index + shingle_size])
        for index in range(len(tokens) - (shingle_size - 1))
    )
