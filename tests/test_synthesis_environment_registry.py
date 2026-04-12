from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rl_task_foundry.synthesis.contracts import (
    CategoryTaxonomy,
    ConstraintKind,
    ConstraintSummaryItem,
    CrossInstanceSet,
    DifficultyAxis,
    EnvironmentContract,
    EnvironmentQualityMetrics,
    EnvironmentStatus,
    InstanceSpaceContract,
    MaterializedFactsSchema,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    ShadowVerifierContract,
    SolutionContract,
    TaskContract,
    ToolContract,
    ToolSelfTestContract,
    VerifierContract,
    AnchorQueryContract,
)
from rl_task_foundry.synthesis.atomic_tools import AtomicToolBundle
from rl_task_foundry.synthesis.environment_registry import (
    DifficultyBand,
    EnvironmentRegistryCoverageEntry,
    EnvironmentRegistryCommitStatus,
    EnvironmentRegistryDuplicateReason,
    EnvironmentRegistrySnapshot,
    EnvironmentRegistryWriter,
    SemanticDedupCandidate,
    build_semantic_dedup_text,
    bucketize_difficulty_vector,
    estimate_semantic_similarity,
)
from rl_task_foundry.synthesis.registration_policy import ArtifactKind
from rl_task_foundry.synthesis.registration_runner import (
    ArtifactRegistrationResult,
    GeneratedArtifactBundle,
    RegistrationArtifactName,
    RegistrationBundleReport,
    RegistrationBundleStatus,
    build_registration_diagnostics,
)
from rl_task_foundry.synthesis.runtime import (
    SynthesisEnvironmentDraft,
    SynthesisSelfConsistencyDiagnostics,
)


def _sample_draft(
    tmp_env_id: str = "env_assignment_registrytest",
    *,
    db_id: str = "sakila",
    category: CategoryTaxonomy = CategoryTaxonomy.ASSIGNMENT,
    question: str = "고객 배정 계획을 만들어 주세요.",
    created_at: datetime | None = None,
    difficulty_vector: dict[DifficultyAxis, float] | None = None,
    tool_signature: str = "sha256:tool",
    task_signature: str = "sha256:task",
    verifier_signature: str = "sha256:verifier",
) -> SynthesisEnvironmentDraft:
    created_at = created_at or datetime(2026, 4, 12, tzinfo=timezone.utc)
    output_schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="assignment",
            type=OutputFieldType.OBJECT,
            fields=[
                OutputFieldContract(name="customer", type=OutputFieldType.STRING),
                OutputFieldContract(name="day", type=OutputFieldType.DATE),
            ],
        )
    )
    task = TaskContract(
        question=question,
        category=category,
        output_schema=output_schema,
        constraint_summary=[
            ConstraintSummaryItem(
                key="unique_customer",
                kind=ConstraintKind.UNIQUENESS,
                summary="같은 고객을 중복 배정하지 않는다.",
            )
        ],
        difficulty_vector=difficulty_vector
        or {
            DifficultyAxis.SLOT_COUNT: 2.0,
            DifficultyAxis.CONSTRAINT_COUNT: 2.0,
        },
    )
    environment = EnvironmentContract(
        env_id=tmp_env_id,
        db_id=db_id,
        domain="service_operations",
        category=category,
        difficulty_vector=task.difficulty_vector,
        created_at=created_at,
        generator_version="test-version",
        tool_signature=tool_signature,
        task_signature=task_signature,
        verifier_signature=verifier_signature,
        status=EnvironmentStatus.DRAFT,
        quality_metrics=EnvironmentQualityMetrics(self_consistency_pass=True),
        tools=[
            ToolContract(
                name="get_assignments",
                description="Return candidate assignments.",
                return_schema=OutputFieldContract(
                    name="rows",
                    type=OutputFieldType.LIST,
                    ordered=True,
                    items=OutputFieldContract(
                        name="row",
                        type=OutputFieldType.OBJECT,
                        fields=[
                            OutputFieldContract(name="customer", type=OutputFieldType.STRING)
                        ],
                    ),
                ),
            )
        ],
        task=task,
        solution=SolutionContract(),
        tool_self_test=ToolSelfTestContract(),
        verifier=VerifierContract(facts_schema=MaterializedFactsSchema()),
        shadow_verifier=ShadowVerifierContract(facts_schema=MaterializedFactsSchema()),
        instance_space=InstanceSpaceContract(
            anchor_query=AnchorQueryContract(
                sql="SELECT customer_id FROM customer ORDER BY customer_id",
                outputs=["customer_id"],
            )
        ),
        cross_instance_set=CrossInstanceSet(),
    )
    report = RegistrationBundleReport(
        status=RegistrationBundleStatus.PASSED,
        tool=ArtifactRegistrationResult(
            artifact_name=RegistrationArtifactName.TOOL,
            artifact_kind=ArtifactKind.TOOL_MODULE,
        ),
        tool_self_test=ArtifactRegistrationResult(
            artifact_name=RegistrationArtifactName.TOOL_SELF_TEST,
            artifact_kind=ArtifactKind.TOOL_SELF_TEST_MODULE,
            execution_required=True,
            executed=True,
            execution_call_count=1,
            execution_return_value={"ok": True},
        ),
        solution=ArtifactRegistrationResult(
            artifact_name=RegistrationArtifactName.SOLUTION,
            artifact_kind=ArtifactKind.SOLUTION_MODULE,
        ),
        verifier=ArtifactRegistrationResult(
            artifact_name=RegistrationArtifactName.VERIFIER,
            artifact_kind=ArtifactKind.VERIFIER_MODULE,
        ),
        shadow_verifier=ArtifactRegistrationResult(
            artifact_name=RegistrationArtifactName.SHADOW_VERIFIER,
            artifact_kind=ArtifactKind.SHADOW_VERIFIER_MODULE,
        ),
    )
    return SynthesisEnvironmentDraft(
        created_at=created_at,
        db_id=db_id,
        requested_category=category,
        schema_summary={"included_table_count": 2},
        selected_category=category,
        environment=environment,
        atomic_tool_bundle=AtomicToolBundle(
            db_id=db_id,
            tools=[],
            source="async def get_assignments(conn, customer_id):\n    return []\n",
        ),
        artifacts=GeneratedArtifactBundle(
            solution_source="def solve(tools):\n    return {'assignment': {}}\n",
            verifier_source=(
                "async def fetch_facts(answer, tools):\n    return {}\n\n"
                "def facts_match_answer_claims(answer, facts):\n    return True\n\n"
                "def check_constraints(answer, facts):\n    return True\n\n"
                "def verify(answer, tools):\n    return True\n"
            ),
            shadow_verifier_source=(
                "async def fetch_facts(answer, tools):\n    return {}\n\n"
                "def facts_match_answer_claims(answer, facts):\n    return True\n\n"
                "def check_constraints(answer, facts):\n    return True\n\n"
                "def verify(answer, tools):\n    return True\n"
            ),
        ),
        registration_report=report,
        registration_diagnostics=build_registration_diagnostics(report),
        self_consistency_diagnostics=SynthesisSelfConsistencyDiagnostics(passed=True),
        provider_status={},
    )


def test_bucketize_difficulty_vector() -> None:
    assert bucketize_difficulty_vector({}) == DifficultyBand.UNSET
    assert bucketize_difficulty_vector({DifficultyAxis.SLOT_COUNT: 2.0}) == DifficultyBand.LOW
    assert bucketize_difficulty_vector({DifficultyAxis.SLOT_COUNT: 4.0}) == DifficultyBand.MEDIUM
    assert bucketize_difficulty_vector({DifficultyAxis.SLOT_COUNT: 9.0}) == DifficultyBand.HIGH


def test_environment_registry_writer_commits_bundle_and_updates_index(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    draft = _sample_draft()

    result = writer.commit_draft(draft)

    assert result.status == EnvironmentRegistryCommitStatus.COMMITTED
    env_dir = writer.root_dir / draft.environment.env_id
    assert result.filesystem_path == env_dir
    assert (env_dir / "environment.yaml").exists()
    assert (env_dir / "task.json").exists()
    assert (env_dir / "instance_space.json").exists()
    assert (env_dir / "cross_instance_set.json").exists()
    assert (env_dir / "tools.py").exists()
    assert (env_dir / "tools.py").read_text(encoding="utf-8") == draft.atomic_tool_bundle.source
    assert not (env_dir / "tool_self_test.py").exists()
    assert (env_dir / "solution.py").exists()
    assert (env_dir / "verifier.py").exists()
    assert (env_dir / "shadow_verifier.py").exists()
    assert (env_dir / "registry_metadata.json").exists()
    assert (tmp_path / "databases" / "sakila" / "atomic_tools.py").exists()
    assert (tmp_path / "databases" / "sakila" / "atomic_tool_definitions.json").exists()
    assert writer.environment_count() == 1
    assert writer.coverage_snapshot() == {
        "db=sakila|category=assignment|difficulty_band=medium": 1
    }


def test_environment_registry_writer_deduplicates_exact_signature(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    draft = _sample_draft()

    first = writer.commit_draft(draft)
    second = writer.commit_draft(draft)

    assert first.status == EnvironmentRegistryCommitStatus.COMMITTED
    assert second.status == EnvironmentRegistryCommitStatus.DUPLICATE
    assert second.duplicate_of_env_id == draft.environment.env_id
    assert second.duplicate_reason == EnvironmentRegistryDuplicateReason.EXACT
    assert writer.environment_count() == 1
    assert writer.coverage_snapshot() == {
        "db=sakila|category=assignment|difficulty_band=medium": 1
    }


def test_environment_registry_writer_deduplicates_semantic_minhash(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
        minhash_threshold=0.75,
    )
    first = _sample_draft(
        "env_assignment_registrytest_a",
        tool_signature="sha256:tool_a",
        task_signature="sha256:task_a",
        verifier_signature="sha256:verifier_a",
    )
    second = _sample_draft(
        "env_assignment_registrytest_b",
        question="고객 배정 계획을 세워 주세요.",
        tool_signature="sha256:tool_b",
        task_signature="sha256:task_b",
        verifier_signature="sha256:verifier_b",
    )

    committed = writer.commit_draft(first)
    duplicate = writer.commit_draft(second)

    assert committed.status == EnvironmentRegistryCommitStatus.COMMITTED
    assert duplicate.status == EnvironmentRegistryCommitStatus.DUPLICATE
    assert duplicate.duplicate_reason == EnvironmentRegistryDuplicateReason.MINHASH
    assert duplicate.duplicate_of_env_id == "env_assignment_registrytest_a"
    assert duplicate.semantic_similarity is not None
    assert 0.75 <= duplicate.semantic_similarity < 1.0
    assert writer.environment_count() == 1


def test_environment_registry_writer_respects_near_dup_disabled(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
        near_dup_enabled=False,
        minhash_threshold=0.75,
    )
    first = _sample_draft(
        "env_assignment_registrytest_a",
        tool_signature="sha256:tool_a",
        task_signature="sha256:task_a",
        verifier_signature="sha256:verifier_a",
    )
    second = _sample_draft(
        "env_assignment_registrytest_b",
        question="고객 배정 계획을 세워 주세요.",
        tool_signature="sha256:tool_b",
        task_signature="sha256:task_b",
        verifier_signature="sha256:verifier_b",
    )

    committed_a = writer.commit_draft(first)
    committed_b = writer.commit_draft(second)

    assert committed_a.status == EnvironmentRegistryCommitStatus.COMMITTED
    assert committed_b.status == EnvironmentRegistryCommitStatus.COMMITTED
    assert writer.environment_count() == 2


def test_environment_registry_writer_persists_semantic_minhash_signature(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    writer.commit_draft(_sample_draft())

    conn = sqlite3.connect(writer.index_db_path)
    try:
        row = conn.execute(
            """
            SELECT semantic_dedup_text, semantic_dedup_text_version, semantic_minhash_signature
            FROM environments
            """
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert "question:고객 배정 계획을 만들어 주세요." in row[0]
    assert row[1] == 1
    assert len(row[2]) > 10


def test_environment_registry_snapshot_and_semantic_candidates(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    first = _sample_draft(
        "env_assignment_registrytest",
        created_at=datetime(2026, 4, 12, 10, tzinfo=timezone.utc),
    )
    second = _sample_draft(
        "env_itinerary_registrytest",
        category=CategoryTaxonomy.ITINERARY,
        question="3일 여정을 짜 주세요.",
        created_at=datetime(2026, 4, 12, 11, tzinfo=timezone.utc),
        difficulty_vector={DifficultyAxis.SLOT_COUNT: 9.0},
    )

    writer.commit_draft(first)
    writer.commit_draft(second)

    snapshot = writer.snapshot(limit=5)
    candidates = writer.semantic_dedup_candidates(limit=5)

    assert isinstance(snapshot, EnvironmentRegistrySnapshot)
    assert snapshot.environment_count == 2
    assert snapshot.coverage == [
        EnvironmentRegistryCoverageEntry(
            db_id="sakila",
            category=CategoryTaxonomy.ASSIGNMENT,
            difficulty_band=DifficultyBand.MEDIUM,
            count=1,
        ),
        EnvironmentRegistryCoverageEntry(
            db_id="sakila",
            category=CategoryTaxonomy.ITINERARY,
            difficulty_band=DifficultyBand.HIGH,
            count=1,
        ),
    ]
    assert [record.env_id for record in snapshot.recent_environments] == [
        "env_itinerary_registrytest",
        "env_assignment_registrytest",
    ]
    assert [candidate.env_id for candidate in candidates] == [
        "env_itinerary_registrytest",
        "env_assignment_registrytest",
    ]
    assert isinstance(candidates[0], SemanticDedupCandidate)
    assert "question:3일 여정을 짜 주세요." in candidates[0].semantic_text
    assert candidates[0].constraint_summaries == ("같은 고객을 중복 배정하지 않는다.",)


def test_environment_registry_snapshot_filters_by_db_and_category(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
    )
    writer.commit_draft(_sample_draft("env_sakila_assignment", db_id="sakila"))
    writer.commit_draft(
        _sample_draft(
            "env_northwind_assignment",
            db_id="northwind",
            created_at=datetime(2026, 4, 12, 11, tzinfo=timezone.utc),
        )
    )
    writer.commit_draft(
        _sample_draft(
            "env_northwind_itinerary",
            db_id="northwind",
            category=CategoryTaxonomy.ITINERARY,
            created_at=datetime(2026, 4, 12, 12, tzinfo=timezone.utc),
        )
    )

    snapshot = writer.snapshot(
        limit=5,
        db_id="northwind",
        category=CategoryTaxonomy.ASSIGNMENT,
    )
    candidates = writer.semantic_dedup_candidates(
        limit=5,
        db_id="northwind",
        category=CategoryTaxonomy.ASSIGNMENT,
    )

    assert snapshot.environment_count == 1
    assert [entry.db_id for entry in snapshot.coverage] == ["northwind"]
    assert [entry.category for entry in snapshot.coverage] == [CategoryTaxonomy.ASSIGNMENT]
    assert [record.env_id for record in snapshot.recent_environments] == [
        "env_northwind_assignment"
    ]
    assert [candidate.env_id for candidate in candidates] == ["env_northwind_assignment"]


def test_build_semantic_dedup_text_includes_question_and_schema() -> None:
    draft = _sample_draft()

    text = build_semantic_dedup_text(draft.environment)

    assert "question:고객 배정 계획을 만들어 주세요." in text
    assert "constraints:uniqueness:unique_customer:같은 고객을 중복 배정하지 않는다." in text
    assert "output_schema:assignment.customer:string" in text


def test_estimate_semantic_similarity_is_high_for_same_semantics() -> None:
    first = build_semantic_dedup_text(
        _sample_draft(
            "env_assignment_similarity_a",
            tool_signature="sha256:tool_a",
            task_signature="sha256:task_a",
            verifier_signature="sha256:verifier_a",
        ).environment
    )
    second = build_semantic_dedup_text(
        _sample_draft(
            "env_assignment_similarity_b",
            tool_signature="sha256:tool_b",
            task_signature="sha256:task_b",
            verifier_signature="sha256:verifier_b",
        ).environment
    )

    assert estimate_semantic_similarity(first, second) >= 0.9


def test_estimate_semantic_similarity_tracks_non_identical_near_duplicates() -> None:
    first = build_semantic_dedup_text(
        _sample_draft(
            "env_assignment_similarity_a",
            question="고객 배정 계획을 만들어 주세요.",
        ).environment
    )
    second = build_semantic_dedup_text(
        _sample_draft(
            "env_assignment_similarity_b",
            question="고객 배정 계획을 세워 주세요.",
            tool_signature="sha256:tool_b",
            task_signature="sha256:task_b",
            verifier_signature="sha256:verifier_b",
        ).environment
    )
    different = build_semantic_dedup_text(
        _sample_draft(
            "env_assignment_similarity_c",
            category=CategoryTaxonomy.ITINERARY,
            question="3일 여행 일정을 짜 주세요.",
            tool_signature="sha256:tool_c",
            task_signature="sha256:task_c",
            verifier_signature="sha256:verifier_c",
        ).environment
    )

    similarity = estimate_semantic_similarity(first, second)
    different_similarity = estimate_semantic_similarity(first, different)

    assert similarity < 1.0
    assert similarity > different_similarity


def test_environment_registry_writer_uses_in_transaction_semantic_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "environments",
        index_db_path=tmp_path / "environment_registry.db",
        minhash_threshold=0.75,
    )
    first = _sample_draft(
        "env_assignment_registrytest_a",
        tool_signature="sha256:tool_a",
        task_signature="sha256:task_a",
        verifier_signature="sha256:verifier_a",
    )
    second = _sample_draft(
        "env_assignment_registrytest_b",
        question="고객 배정 계획을 세워 주세요.",
        tool_signature="sha256:tool_b",
        task_signature="sha256:task_b",
        verifier_signature="sha256:verifier_b",
    )
    writer.commit_draft(first)

    original = EnvironmentRegistryWriter._lookup_semantic_duplicate
    call_counter = {"count": 0}

    def _wrapped(self, **kwargs):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return None
        return original(self, **kwargs)

    monkeypatch.setattr(EnvironmentRegistryWriter, "_lookup_semantic_duplicate", _wrapped)

    duplicate = writer.commit_draft(second)

    assert call_counter["count"] >= 2
    assert duplicate.status == EnvironmentRegistryCommitStatus.DUPLICATE
    assert duplicate.duplicate_reason == EnvironmentRegistryDuplicateReason.MINHASH
