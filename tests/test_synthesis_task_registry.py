from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rl_task_foundry.config.models import TaskRegistryConfig
from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle
from rl_task_foundry.synthesis.canonicalize import canonical_json
from rl_task_foundry.synthesis.contracts import (
    ConstraintKind,
    ConstraintSummaryItem,
    DifficultyVectorContract,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    RolloutConstraintsContract,
    TaskBundleContract,
    TaskBundleStatus,
    TaskContract,
    TaskQualityMetrics,
    build_difficulty_vector,
)
from rl_task_foundry.synthesis.rendered_prompt_builder import build_rendered_user_prompt
from rl_task_foundry.synthesis.runtime import SynthesisTaskDraft
from rl_task_foundry.synthesis.task_registry import (
    DifficultyBand,
    TaskRegistryCommitStatus,
    TaskRegistryDuplicateReason,
    TaskRegistryWriter,
    _SemanticScopeIndex,
    build_semantic_dedup_text,
    bucketize_difficulty_vector,
)


def _sample_draft(
    tmp_task_id: str = "task_assignment_registrytest",
    *,
    db_id: str = "sakila",
    topic: str = "assignment",
    question: str = "내 배정 계획을 알려 주세요.",
    created_at: datetime | None = None,
    difficulty_vector: DifficultyVectorContract | None = None,
    tool_signature: str = "sha256:tool",
    task_signature: str = "sha256:task",
    canonical_answer: dict[str, object] | None = None,
) -> SynthesisTaskDraft:
    created_at = created_at or datetime(2026, 4, 12, tzinfo=timezone.utc)
    canonical_answer = canonical_answer or {"customer": "Alice", "day": "2026-04-12"}
    canonical_answer_json = canonical_json(canonical_answer)
    label_signature = "sha256:answer"
    output_schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[
                OutputFieldContract(name="customer", type=OutputFieldType.STRING),
                OutputFieldContract(name="day", type=OutputFieldType.DATE),
            ],
        )
    )
    task = TaskContract(
        question=question,
        topic=topic,
        output_schema=output_schema,
        constraint_summary=[
            ConstraintSummaryItem(
                key="unique_customer",
                kind=ConstraintKind.UNIQUENESS,
                summary="같은 고객을 중복 배정하지 않는다.",
            )
        ],
        difficulty_vector=difficulty_vector
        or build_difficulty_vector(solution_space=2.0, constraint_density=2.0),
        instance_parameters={"customer_id": 1},
    )
    task_bundle = TaskBundleContract(
        task_id=tmp_task_id,
        db_id=db_id,
        domain="service_operations",
        topic=topic,
        atomic_tool_set_ref=f"db://{db_id}",
        difficulty_vector=task.difficulty_vector,
        created_at=created_at,
        generator_version="test-version",
        tool_signature=tool_signature,
        task_signature=task_signature,
        status=TaskBundleStatus.DRAFT,
        quality_metrics=TaskQualityMetrics(),
        rollout_constraints=RolloutConstraintsContract(
            max_turns=16,
            max_episode_duration_ms=80000,
            max_tool_rows=100,
        ),
        task=task,
    )
    anchor_entity = {"customer_id": 1}
    return SynthesisTaskDraft(
        created_at=created_at,
        db_id=db_id,
        requested_topic=topic,
        schema_summary={"included_table_count": 2},
        selected_topic=topic,
        task_bundle=task_bundle,
        atomic_tool_bundle=AtomicToolBundle(
            db_id=db_id,
            tools=[],
            source="async def get_assignments(conn, customer_id, limit=10):\n    return []\n",
        ),
        rendered_user_prompt=build_rendered_user_prompt(
            task,
            anchor_entity=anchor_entity,
            canonical_answer=canonical_answer,
        ),
        anchor_entity=anchor_entity,
        canonical_answer_json=canonical_answer_json,
        label_signature=label_signature,
        generation_attempts=[],
        provider_status={},
    )


def test_bucketize_difficulty_vector() -> None:
    registry_config = TaskRegistryConfig()
    assert (
        bucketize_difficulty_vector(
            build_difficulty_vector(),
            registry_config=registry_config,
        )
        == DifficultyBand.UNSET
    )
    assert (
        bucketize_difficulty_vector(
            build_difficulty_vector(solution_space=2.0),
            registry_config=registry_config,
        )
        == DifficultyBand.LOW
    )
    assert (
        bucketize_difficulty_vector(
            build_difficulty_vector(solution_space=4.0),
            registry_config=registry_config,
        )
        == DifficultyBand.MEDIUM
    )
    assert (
        bucketize_difficulty_vector(
            build_difficulty_vector(solution_space=9.0),
            registry_config=registry_config,
        )
        == DifficultyBand.HIGH
    )


def test_task_registry_writer_commits_single_task_layout(tmp_path: Path) -> None:
    writer = TaskRegistryWriter(
        root_dir=tmp_path / "tasks",
        index_db_path=tmp_path / "task_registry.db",
    )
    draft = _sample_draft()

    result = writer.commit_draft(draft)

    assert result.status is TaskRegistryCommitStatus.COMMITTED
    task_dir = writer.root_dir / draft.task_bundle.task_id
    assert (task_dir / "task.yaml").exists()
    assert (task_dir / "task.json").exists()
    assert (task_dir / "instance.json").exists()
    assert (task_dir / "canonical_answer.json").exists()
    assert (task_dir / "tools.py").read_text(encoding="utf-8") == draft.atomic_tool_bundle.source

    metadata = json.loads((task_dir / "registry_metadata.json").read_text(encoding="utf-8"))
    assert metadata["generation_attempts"] == []
    instance_payload = json.loads((task_dir / "instance.json").read_text(encoding="utf-8"))
    assert instance_payload["anchor_entity"] == {"customer_id": 1}
    answer_payload = json.loads(
        (task_dir / "canonical_answer.json").read_text(encoding="utf-8")
    )
    assert answer_payload["label_signature"] == draft.label_signature


def test_task_registry_writer_reuses_cached_connection(tmp_path: Path) -> None:
    writer = TaskRegistryWriter(
        root_dir=tmp_path / "tasks",
        index_db_path=tmp_path / "task_registry.db",
    )

    first = writer._connect()
    second = writer._connect()
    writer.close()
    third = writer._connect()

    assert first is second
    assert first is not third


def test_task_registry_semantic_dedup_candidates_reads_topic_column(tmp_path: Path) -> None:
    writer = TaskRegistryWriter(
        root_dir=tmp_path / "tasks",
        index_db_path=tmp_path / "task_registry.db",
    )
    draft = _sample_draft()

    writer.commit_draft(draft)
    candidates = writer.semantic_dedup_candidates()

    assert len(candidates) == 1
    assert candidates[0].topic == "assignment"
    assert candidates[0].question == "내 배정 계획을 알려 주세요."


def test_task_registry_close_clears_semantic_scope_indexes(tmp_path: Path) -> None:
    writer = TaskRegistryWriter(
        root_dir=tmp_path / "tasks",
        index_db_path=tmp_path / "task_registry.db",
    )
    writer._semantic_scope_indexes[("sakila", "assignment")] = _SemanticScopeIndex(  # type: ignore[arg-type]
        lsh=object(),
    )

    writer.close()

    assert writer._semantic_scope_indexes == {}


def test_task_registry_commit_preserves_primary_error_when_cleanup_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = TaskRegistryWriter(
        root_dir=tmp_path / "tasks",
        index_db_path=tmp_path / "task_registry.db",
    )
    draft = _sample_draft()

    def _raise_primary(conn: sqlite3.Connection, key: str) -> None:
        del conn, key
        raise RuntimeError("primary failure")

    def _raise_secondary(task_id: str) -> bool:
        del task_id
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(
        TaskRegistryWriter,
        "_increment_coverage_counter",
        lambda self, conn, key: _raise_primary(conn, key),
    )
    monkeypatch.setattr(
        TaskRegistryWriter,
        "_has_task_row",
        lambda self, task_id: _raise_secondary(task_id),
    )

    with pytest.raises(RuntimeError, match="primary failure"):
        writer.commit_draft(draft)


def test_task_registry_exact_signature_ignores_label_signature(tmp_path: Path) -> None:
    writer = TaskRegistryWriter(
        root_dir=tmp_path / "tasks",
        index_db_path=tmp_path / "task_registry.db",
    )
    first = _sample_draft(tmp_task_id="task_a")
    second = _sample_draft(
        tmp_task_id="task_b",
        canonical_answer={"customer": "Bob", "day": "2026-04-13"},
    )

    first_result = writer.commit_draft(first)
    second_result = writer.commit_draft(second)

    assert first_result.status is TaskRegistryCommitStatus.COMMITTED
    assert second_result.status is TaskRegistryCommitStatus.DUPLICATE
    assert second_result.duplicate_reason is TaskRegistryDuplicateReason.EXACT
    assert second_result.duplicate_of_task_id == "task_a"


def test_task_registry_bootstrap_migrates_legacy_schema(tmp_path: Path) -> None:
    index_db_path = tmp_path / "task_registry.db"
    with sqlite3.connect(index_db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE environments (
                {'env' + '_id'} TEXT PRIMARY KEY,
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
            """
        )
        conn.commit()

    writer = TaskRegistryWriter(
        root_dir=tmp_path / "tasks",
        index_db_path=index_db_path,
    )
    result = writer.commit_draft(_sample_draft())

    assert result.status is TaskRegistryCommitStatus.COMMITTED
    with sqlite3.connect(index_db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
        }
    assert "tasks" in tables
    assert "environments" not in tables
    assert "verifier_signature" not in columns


def test_build_semantic_dedup_text_uses_task_surface() -> None:
    draft = _sample_draft()

    semantic_text = build_semantic_dedup_text(draft.task_bundle)

    assert "내 배정 계획" in semantic_text
    assert "unique_customer" in semantic_text
    assert "topic:assignment" in semantic_text
