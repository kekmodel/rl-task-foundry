from __future__ import annotations

from rl_task_foundry.synthesis.registration_runner import (
    RegistrationArtifactDiagnostics,
    RegistrationArtifactName,
    RegistrationBundleDiagnostics,
    RegistrationBundleStatus,
)
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy, DifficultyAxis
from rl_task_foundry.synthesis.contracts import build_difficulty_vector
from rl_task_foundry.synthesis.prompts import (
    build_synthesis_phase_input,
    build_synthesis_phase_instructions,
)
from rl_task_foundry.synthesis.runtime import (
    CategoryInferenceOutput,
    LabelConstructionOutput,
    SchemaExplorationOutput,
    SynthesisPhase,
    SynthesisQualityGateFeedback,
    SynthesisStageRequest,
    TaskSynthesisOutput,
)


def _artifact_diagnostics(
    artifact_name: RegistrationArtifactName,
) -> RegistrationArtifactDiagnostics:
    return RegistrationArtifactDiagnostics(
        artifact_name=artifact_name,
        passed=True,
        static_passed=True,
        runtime_passed=True,
        probe_passed=True,
    )


def _registration_diagnostics(*, error_codes: list[str]) -> RegistrationBundleDiagnostics:
    return RegistrationBundleDiagnostics(
        status=RegistrationBundleStatus.FAILED,
        failing_artifacts=[RegistrationArtifactName.VERIFIER],
        error_codes=error_codes,
        weak_signal_codes=[],
        tool=_artifact_diagnostics(RegistrationArtifactName.TOOL),
        tool_self_test=_artifact_diagnostics(RegistrationArtifactName.TOOL_SELF_TEST),
        solution=_artifact_diagnostics(RegistrationArtifactName.SOLUTION),
        verifier=_artifact_diagnostics(RegistrationArtifactName.VERIFIER),
        shadow_verifier=_artifact_diagnostics(RegistrationArtifactName.SHADOW_VERIFIER),
    )


def test_schema_exploration_prompt_keeps_tools_out_of_user_message() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.SCHEMA_EXPLORATION,
        db_id="sakila",
        atomic_tool_set_ref="db://sakila",
        available_atomic_tools=[
            {
                "name": "get_customer_by_id",
                "description": "Get customer for given primary key.",
                "params_schema": {"type": "object"},
                "returns_schema": {"type": "object"},
            }
        ],
        domain_name="customer_support",
        task_language="ko",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        schema_summary={"table_count": 2, "edge_count": 1, "tables": [{"qualified_name": "public.customer", "primary_key": ["customer_id"], "column_names": ["customer_id", "name"]}]},
    )

    prompt = build_synthesis_phase_input(request)

    assert "# Domain" in prompt
    assert "# Schema Orientation" in prompt
    assert "# Exploration Goal" in prompt
    assert "# Required Output Contract" in prompt
    assert "public.customer" in prompt
    assert "User role:" not in prompt
    assert "Assistant role:" not in prompt
    assert "available_atomic_tools" not in prompt
    assert "atomic_tool_set_ref" not in prompt
    assert "difficulty_crank_index" not in prompt


def test_artifact_generation_prompt_includes_relevant_atomic_tool_shortlist() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.ARTIFACT_GENERATION,
        db_id="sakila",
        atomic_tool_set_ref="db://sakila",
        available_atomic_tools=[
            {
                "name": "get_customer_by_id",
                "description": "Get customer for given primary key.",
                "params_schema": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "integer"}},
                },
                "returns_schema": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"customer_id": {"type": "integer"}},
                        },
                        {"type": "null"},
                    ]
                },
            },
            {
                "name": "traverse_customer_to_store_via_store_id",
                "description": "Get store for given customer reference.",
                "params_schema": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                },
                "returns_schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"store_id": {"type": "integer"}},
                    },
                },
            },
        ],
        domain_name="customer_support",
        task_language="ko",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        previous_outputs={
            SynthesisPhase.SCHEMA_EXPLORATION: SchemaExplorationOutput(
                domain_hypothesis="customer support",
                candidate_categories=[CategoryTaxonomy.ASSIGNMENT],
                sample_observations=["get_customer_by_id(148) returned customer_id 148"],
                memory_summary="schema done",
            ),
            SynthesisPhase.LABEL_CONSTRUCTION: LabelConstructionOutput(
                canonical_answer_json='{"store_id": 1}',
                output_schema={
                    "version": "v2",
                    "root": {
                        "name": "answer",
                        "type": "object",
                        "fields": [{"name": "store_id", "type": "int"}],
                    },
                    "primary_output_format": "json_object",
                },
                difficulty_vector=build_difficulty_vector(search_cost=2.0),
                instance_parameters={},
                label_summary="grounded answer",
                memory_summary="label complete",
            ),
            SynthesisPhase.TASK_SYNTHESIS: TaskSynthesisOutput(
                question="질문",
                constraint_summary=[],
                instance_space={
                    "anchor_query": {"sql": "SELECT customer_id FROM customer", "outputs": ["customer_id"]},
                    "parameters": {},
                    "sampling": {"strategy": "deterministic_hash", "seed": 0},
                    "instance_count": 1,
                },
                memory_summary="task complete",
            ),
        },
    )

    prompt = build_synthesis_phase_input(request)

    assert "# Relevant Atomic Tools" in prompt
    assert "get_customer_by_id(customer_id)" in prompt
    assert "traverse_customer_to_store_via_store_id(customer_id, limit)" in prompt
    assert "keys [store_id]" in prompt


def test_task_synthesis_prompt_includes_language_policy() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.TASK_SYNTHESIS,
        db_id="sakila",
        domain_name="customer_support",
        task_language="ko",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        previous_outputs={
            SynthesisPhase.LABEL_CONSTRUCTION: LabelConstructionOutput(
                canonical_answer_json='{"store_id": 1}',
                output_schema={
                    "version": "v2",
                    "root": {
                        "name": "answer",
                        "type": "object",
                        "fields": [{"name": "store_id", "type": "int"}],
                    },
                    "primary_output_format": "json_object",
                },
                difficulty_vector=build_difficulty_vector(search_cost=2.0),
                instance_parameters={},
                label_summary="grounded answer",
                memory_summary="label complete",
            )
        },
    )

    prompt = build_synthesis_phase_input(request)

    assert "# User-Facing Language" in prompt
    assert "Generate the user-facing question" in prompt
    assert "Korean" in prompt
    assert "All code (solution.py, verifier.py, shadow_verifier.py), field names, and tool calls must remain in English." in prompt


def test_artifact_generation_prompt_uses_natural_language_retry_feedback() -> None:
    diagnostics = _registration_diagnostics(
        error_codes=["facts_schema_keys_mismatch", "shadow_verifier_not_independent"]
    ).model_copy(
        update={
            "verifier": _artifact_diagnostics(RegistrationArtifactName.VERIFIER).model_copy(
                update={
                    "probe_error_codes": ["facts_schema_keys_mismatch"],
                    "probe_expected_fact_keys": ["store_id", "country_id"],
                    "probe_fetch_facts_return_keys": ["store_id", "debug_value"],
                }
            )
        }
    )
    request = SynthesisStageRequest(
        phase=SynthesisPhase.ARTIFACT_GENERATION,
        db_id="sakila",
        domain_name="customer_support",
        task_language="ko",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        latest_registration_diagnostics=diagnostics,
        previous_outputs={
            SynthesisPhase.LABEL_CONSTRUCTION: LabelConstructionOutput(
                canonical_answer_json='{"store_id": 1}',
                output_schema={
                    "version": "v2",
                    "root": {
                        "name": "answer",
                        "type": "object",
                        "fields": [{"name": "store_id", "type": "int"}],
                    },
                    "primary_output_format": "json_object",
                },
                difficulty_vector=build_difficulty_vector(search_cost=2.0),
                instance_parameters={},
                label_summary="grounded answer",
                memory_summary="label complete",
            ),
            SynthesisPhase.TASK_SYNTHESIS: TaskSynthesisOutput(
                question="질문",
                constraint_summary=[],
                instance_space={
                    "anchor_query": {"sql": "SELECT 1", "outputs": ["x"]},
                    "parameters": {},
                    "sampling": {"strategy": "deterministic_hash", "seed": 0},
                    "instance_count": 1,
                },
                memory_summary="task complete",
            ),
        },
    )

    prompt = build_synthesis_phase_input(request)

    assert "# Retry Fixes" in prompt
    assert "fetch_facts() must return exactly the expected fact keys" in prompt
    assert "expected fact keys: store_id, country_id" in prompt
    assert "shadow_verifier must not be structurally identical" in prompt
    assert "registration_fact_key_mismatches" not in prompt
    assert "latest_registration_diagnostics" not in prompt


def test_artifact_generation_prompt_includes_authoritative_task_contract_and_full_required_fields() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.ARTIFACT_GENERATION,
        db_id="sakila",
        domain_name="customer_support",
        task_language="ko",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        previous_outputs={
            SynthesisPhase.LABEL_CONSTRUCTION: LabelConstructionOutput(
                canonical_answer_json='{"store_id": 1}',
                output_schema={
                    "version": "v2",
                    "root": {
                        "name": "answer",
                        "type": "object",
                        "fields": [{"name": "store_id", "type": "int"}],
                    },
                    "primary_output_format": "json_object",
                },
                difficulty_vector=build_difficulty_vector(search_cost=2.0),
                instance_parameters={"customer_id": 1},
                label_summary="grounded answer",
                memory_summary="label complete",
            ),
            SynthesisPhase.TASK_SYNTHESIS: TaskSynthesisOutput(
                question="질문",
                constraint_summary=[],
                instance_space={
                    "anchor_query": {"sql": "SELECT 1", "outputs": ["x"]},
                    "parameters": {},
                    "sampling": {"strategy": "deterministic_hash", "seed": 0},
                    "instance_count": 1,
                },
                memory_summary="task complete",
            ),
            SynthesisPhase.SCHEMA_EXPLORATION: SchemaExplorationOutput(
                domain_hypothesis="customer support",
                candidate_categories=[CategoryTaxonomy.ASSIGNMENT],
                sample_observations=[
                    "get_customer_by_id(customer_id=1) returned store_id 1",
                    "traverse_customer_to_address_via_address_id(customer_id=1, limit=1) returned address 42",
                ],
                memory_summary="schema done",
            ),
        },
    )

    prompt = build_synthesis_phase_input(request)

    assert "# Authoritative Task Contract" in prompt
    assert "# Grounded Tool Evidence" in prompt
    assert '"output_schema"' in prompt
    assert '"difficulty_vector"' in prompt
    assert '"instance_parameters"' in prompt
    assert '"facts_schema"' in prompt
    assert '"fetch_facts_function"' in prompt
    assert '"key": "store_id"' in prompt
    assert "Never use raw SQL" in prompt
    assert "tools.query(...)" in prompt
    assert "get_customer_by_id(customer_id=1)" in prompt


def test_artifact_generation_prompt_uses_natural_language_difficulty_feedback() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.ARTIFACT_GENERATION,
        db_id="sakila",
        domain_name="customer_support",
        task_language="ko",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
        requested_category=CategoryTaxonomy.ASSIGNMENT,
        next_crank_axis=DifficultyAxis.SEARCH_COST,
        crank_hint="Increase search_cost by adding a new table join or traversal step that forces a longer evidence chain.",
        latest_quality_gate_feedback=SynthesisQualityGateFeedback(
            status="reject_too_easy",
            pass_rate=1.0,
            ci_lower=0.6,
            ci_upper=1.0,
            matched_solver_runs=6,
            total_solver_runs=6,
            previous_env_id="env_prev",
            previous_question="가장 이른 고객의 store를 반환하세요.",
            previous_rendered_user_prompt="질문 본문",
            previous_semantic_dedup_text="question:가장 이른 고객의 store를 반환하세요.",
        ),
    )

    prompt = build_synthesis_phase_input(request)

    assert "# Difficulty Guidance" in prompt
    assert "Status: reject_too_easy" in prompt
    assert "Increase search_cost by adding a new table join" in prompt
    assert "Do not repeat this previous question" in prompt
    assert "next_crank_axis" not in prompt
    assert "strongest_difficulty_vector" not in prompt
    assert "current_diversity" not in prompt
    assert "difficulty_crank_history" not in prompt


def test_task_and_artifact_prompts_include_one_shot_example() -> None:
    task_request = SynthesisStageRequest(
        phase=SynthesisPhase.TASK_SYNTHESIS,
        db_id="sakila",
        domain_name="customer_support",
        task_language="ko",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
    )
    artifact_request = task_request.model_copy(update={"phase": SynthesisPhase.ARTIFACT_GENERATION})

    task_prompt = build_synthesis_phase_input(task_request)
    artifact_prompt = build_synthesis_phase_input(artifact_request)

    assert "# One-Shot Example" in task_prompt
    assert '"category": "itinerary"' in task_prompt
    assert "# One-Shot Example" in artifact_prompt
    assert '"solution_source"' in artifact_prompt
    assert '"proposed_environment"' in artifact_prompt
    assert '"facts_schema"' in artifact_prompt


def test_schema_exploration_instructions_require_grounded_tool_use() -> None:
    instructions = build_synthesis_phase_instructions(SynthesisPhase.SCHEMA_EXPLORATION)

    assert "grounded data-exploration phase" in instructions
    assert "Use the provided tools" in instructions
    assert "small number of high-signal tool calls" in instructions
    assert "sample_observations" in instructions


def test_artifact_generation_instructions_are_english_and_label_first() -> None:
    instructions = build_synthesis_phase_instructions(SynthesisPhase.ARTIFACT_GENERATION)

    assert "The label and task outputs from prior phases are authoritative." in instructions
    assert "use only English for code and field names" in instructions
    assert "Do not invent new tools." in instructions
    assert "must fully restate the authoritative question" in instructions
    assert "must declare the exact keys returned by fetch_facts()" in instructions
    assert "Never use raw SQL" in instructions
    assert "tools.query(...)" in instructions


def test_phase_prompt_only_threads_relevant_previous_outputs() -> None:
    request = SynthesisStageRequest(
        phase=SynthesisPhase.ARTIFACT_GENERATION,
        db_id="sakila",
        domain_name="customer_support",
        task_language="ko",
        user_role="end user",
        agent_role="organization AI assistant",
        scenario_description="help requests",
        previous_outputs={
            SynthesisPhase.SCHEMA_EXPLORATION: SchemaExplorationOutput(
                domain_hypothesis="customer support",
                candidate_categories=[CategoryTaxonomy.ASSIGNMENT],
                sample_observations=["count_customer() returned 599"],
                memory_summary="schema done",
            ),
            SynthesisPhase.CATEGORY_INFERENCE: CategoryInferenceOutput(
                selected_category=CategoryTaxonomy.ASSIGNMENT,
                rationale="best fit",
                memory_summary="category done",
            ),
            SynthesisPhase.LABEL_CONSTRUCTION: LabelConstructionOutput(
                canonical_answer_json='{"store_id": 1}',
                output_schema={
                    "version": "v2",
                    "root": {
                        "name": "answer",
                        "type": "object",
                        "fields": [{"name": "store_id", "type": "int"}],
                    },
                    "primary_output_format": "json_object",
                },
                difficulty_vector=build_difficulty_vector(search_cost=2.0),
                instance_parameters={},
                label_summary="grounded answer",
                memory_summary="label done",
            ),
            SynthesisPhase.TASK_SYNTHESIS: TaskSynthesisOutput(
                question="질문",
                constraint_summary=[],
                instance_space={
                    "anchor_query": {"sql": "SELECT 1", "outputs": ["x"]},
                    "parameters": {},
                    "sampling": {"strategy": "deterministic_hash", "seed": 0},
                    "instance_count": 1,
                },
                memory_summary="task done",
            ),
        },
    )

    prompt = build_synthesis_phase_input(request)

    assert "label_construction" in prompt
    assert "task_synthesis" in prompt
    assert "schema_exploration" not in prompt
    assert "category_inference" not in prompt
