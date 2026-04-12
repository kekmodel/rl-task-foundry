from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle
from rl_task_foundry.synthesis.canonicalize import canonical_json
from rl_task_foundry.synthesis.contracts import (
    AnchorQueryContract,
    ConstraintKind,
    ConstraintSummaryItem,
    CrossInstanceSet,
    DifficultyVectorContract,
    EnvironmentContract,
    EnvironmentQualityMetrics,
    EnvironmentStatus,
    InstanceContract,
    InstanceSpaceContract,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    RolloutConstraintsContract,
    TaskContract,
    build_difficulty_vector,
)
from rl_task_foundry.synthesis.environment_registry import (
    DifficultyBand,
    EnvironmentRegistryCommitStatus,
    EnvironmentRegistryDuplicateReason,
    EnvironmentRegistryWriter,
    build_semantic_dedup_text,
    bucketize_difficulty_vector,
)
from rl_task_foundry.synthesis.rendered_prompt_builder import build_rendered_user_prompt
from rl_task_foundry.synthesis.runtime import (
    MaterializedCanonicalAnswerRecord,
    MaterializedInstanceRecord,
    SynthesisEnvironmentDraft,
)


def _sample_draft(
    tmp_env_id: str = "env_assignment_registrytest",
    *,
    db_id: str = "sakila",
    topic: str = "assignment",
    question: str = "고객 배정 계획을 만들어 주세요.",
    created_at: datetime | None = None,
    difficulty_vector: DifficultyVectorContract | None = None,
    tool_signature: str = "sha256:tool",
    task_signature: str = "sha256:task",
    canonical_answer: dict[str, object] | None = None,
) -> SynthesisEnvironmentDraft:
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
    environment = EnvironmentContract(
        env_id=tmp_env_id,
        db_id=db_id,
        domain="service_operations",
        topic=topic,
        atomic_tool_set_ref=f"db://{db_id}",
        difficulty_vector=task.difficulty_vector,
        created_at=created_at,
        generator_version="test-version",
        tool_signature=tool_signature,
        task_signature=task_signature,
        status=EnvironmentStatus.DRAFT,
        quality_metrics=EnvironmentQualityMetrics(),
        rollout_constraints=RolloutConstraintsContract(
            max_turns=16,
            max_episode_duration_ms=80000,
            max_tool_rows=100,
        ),
        task=task,
        instance_space=InstanceSpaceContract(
            anchor_query=AnchorQueryContract(
                sql="SELECT customer_id FROM customer ORDER BY customer_id",
                outputs=["customer_id"],
            )
        ),
        cross_instance_set=CrossInstanceSet(
            minimum_required=1,
            instances=[
                InstanceContract(
                    instance_id="instance_0001",
                    parameter_values={"customer_id": 1},
                    anchor_values={"customer_id": 1},
                    expected_label_signature=label_signature,
                )
            ],
        ),
    )
    return SynthesisEnvironmentDraft(
        created_at=created_at,
        db_id=db_id,
        requested_topic=topic,
        schema_summary={"included_table_count": 2},
        selected_topic=topic,
        environment=environment,
        atomic_tool_bundle=AtomicToolBundle(
            db_id=db_id,
            tools=[],
            source="async def get_assignments(conn, customer_id, limit=10):\n    return []\n",
        ),
        instances=[
            MaterializedInstanceRecord(
                instance_id="instance_0001",
                rendered_user_prompt=build_rendered_user_prompt(
                    task,
                    anchor_entity={"customer_id": 1},
                    canonical_answer=canonical_answer,
                ),
                params={"customer_id": 1},
                anchor_values={"customer_id": 1},
            )
        ],
        canonical_answers=[
            MaterializedCanonicalAnswerRecord(
                instance_id="instance_0001",
                canonical_answer=canonical_answer,
                canonical_answer_json=canonical_answer_json,
                label_signature=label_signature,
            )
        ],
        generation_attempts=[],
        provider_status={},
    )


def test_bucketize_difficulty_vector() -> None:
    assert bucketize_difficulty_vector(build_difficulty_vector()) == DifficultyBand.UNSET
    assert (
        bucketize_difficulty_vector(build_difficulty_vector(solution_space=2.0))
        == DifficultyBand.LOW
    )
    assert (
        bucketize_difficulty_vector(build_difficulty_vector(solution_space=4.0))
        == DifficultyBand.MEDIUM
    )
    assert (
        bucketize_difficulty_vector(build_difficulty_vector(solution_space=9.0))
        == DifficultyBand.HIGH
    )


def test_environment_registry_writer_commits_label_only_bundle(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    draft = _sample_draft()

    result = writer.commit_draft(draft)

    assert result.status is EnvironmentRegistryCommitStatus.COMMITTED
    env_dir = writer.root_dir / draft.environment.env_id
    assert (env_dir / "environment.yaml").exists()
    assert (env_dir / "instances.jsonl").exists()
    assert (env_dir / "canonical_answers.jsonl").exists()
    assert (env_dir / "tools.py").read_text(encoding="utf-8") == draft.atomic_tool_bundle.source
    assert not (env_dir / "solution.py").exists()
    assert not (env_dir / "audit").exists()

    metadata = json.loads((env_dir / "registry_metadata.json").read_text(encoding="utf-8"))
    assert metadata["instance_count"] == 1
    assert metadata["canonical_answer_count"] == 1
    assert metadata["generation_attempts"] == []


def test_environment_registry_exact_signature_ignores_label_signature(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    first = _sample_draft(tmp_env_id="env_a")
    second = _sample_draft(
        tmp_env_id="env_b",
        canonical_answer={"customer": "Bob", "day": "2026-04-13"},
    )

    first_result = writer.commit_draft(first)
    second_result = writer.commit_draft(second)

    assert first_result.status is EnvironmentRegistryCommitStatus.COMMITTED
    assert second_result.status is EnvironmentRegistryCommitStatus.DUPLICATE
    assert second_result.duplicate_reason is EnvironmentRegistryDuplicateReason.EXACT
    assert second_result.duplicate_of_env_id == "env_a"


def test_environment_registry_bootstrap_migrates_legacy_verifier_signature_schema(
    tmp_path: Path,
) -> None:
    index_db_path = tmp_path / "environment_registry.db"
    with sqlite3.connect(index_db_path) as conn:
        conn.execute(
            """
            CREATE TABLE environments (
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
            """
        )
        conn.commit()

    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "environments",
        index_db_path=index_db_path,
    )
    draft = _sample_draft()

    result = writer.commit_draft(draft)

    assert result.status is EnvironmentRegistryCommitStatus.COMMITTED
    with sqlite3.connect(index_db_path) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(environments)").fetchall()
        }
    assert "verifier_signature" not in columns


def test_build_semantic_dedup_text_uses_task_surface() -> None:
    draft = _sample_draft()

    semantic_text = build_semantic_dedup_text(draft.environment)

    assert "고객 배정 계획" in semantic_text
    assert "unique_customer" in semantic_text
    assert "topic:assignment" in semantic_text
