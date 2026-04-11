from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
from rl_task_foundry.synthesis.environment_registry import (
    DifficultyBand,
    EnvironmentRegistryCommitStatus,
    EnvironmentRegistryWriter,
    bucketize_difficulty_vector,
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


def _sample_draft(tmp_env_id: str = "env_assignment_registrytest") -> SynthesisEnvironmentDraft:
    created_at = datetime(2026, 4, 12, tzinfo=timezone.utc)
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
        question="고객 배정 계획을 만들어 주세요.",
        category=CategoryTaxonomy.ASSIGNMENT,
        output_schema=output_schema,
        constraint_summary=[
            ConstraintSummaryItem(
                key="unique_customer",
                kind=ConstraintKind.UNIQUENESS,
                summary="같은 고객을 중복 배정하지 않는다.",
            )
        ],
        difficulty_vector={
            DifficultyAxis.SLOT_COUNT: 2.0,
            DifficultyAxis.CONSTRAINT_COUNT: 2.0,
        },
    )
    environment = EnvironmentContract(
        env_id=tmp_env_id,
        db_id="sakila",
        domain="service_operations",
        category=CategoryTaxonomy.ASSIGNMENT,
        difficulty_vector=task.difficulty_vector,
        created_at=created_at,
        generator_version="test-version",
        tool_signature="sha256:tool",
        task_signature="sha256:task",
        verifier_signature="sha256:verifier",
        status=EnvironmentStatus.DRAFT,
        quality_metrics=EnvironmentQualityMetrics(self_consistency_pass=True),
        tools=[
            ToolContract(
                name="get_assignments",
                description="Return candidate assignments.",
                return_schema=OutputFieldContract(
                    name="rows",
                    type=OutputFieldType.LIST,
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
        db_id="sakila",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        schema_summary={"included_table_count": 2},
        selected_category=CategoryTaxonomy.ASSIGNMENT,
        environment=environment,
        artifacts=GeneratedArtifactBundle(
            tool_source="async def get_assignments(conn, customer_id):\n    return []\n",
            tool_self_test_source="async def run_self_test(tools):\n    return {'ok': True}\n",
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
    assert (env_dir / "tool_self_test.py").exists()
    assert (env_dir / "solution.py").exists()
    assert (env_dir / "verifier.py").exists()
    assert (env_dir / "shadow_verifier.py").exists()
    assert (env_dir / "registry_metadata.json").exists()
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
    assert writer.environment_count() == 1
    assert writer.coverage_snapshot() == {
        "db=sakila|category=assignment|difficulty_band=medium": 1
    }
