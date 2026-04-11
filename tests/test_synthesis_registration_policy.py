from __future__ import annotations

import asyncio

from rl_task_foundry.config import load_config
from rl_task_foundry.synthesis.contracts import (
    FactSpec,
    FactValueType,
    MaterializedFactsSchema,
)
from rl_task_foundry.synthesis.registration_policy import (
    ArtifactKind,
    validate_generated_module,
)
from rl_task_foundry.synthesis.registration_runner import (
    GeneratedArtifactBundle,
    RegistrationArtifactName,
    RegistrationBundleStatus,
    VerifierProbeSpec,
    run_registration_bundle,
)
from rl_task_foundry.synthesis.runtime_policy import build_runtime_isolation_plan
from rl_task_foundry.synthesis.subprocess_pool import RegistrationSubprocessPool


def _valid_verifier_source() -> str:
    return """
def verify(answer, tools):
    facts = fetch_facts(answer, tools)
    if not facts_match_answer_claims(answer, facts):
        return False
    return check_constraints(answer, facts)

def fetch_facts(answer, tools):
    city = answer.get("city")
    return {"city": tools.lookup_city(city)}

def facts_match_answer_claims(answer, facts):
    return answer.get("city") == facts.get("city")

def check_constraints(answer, facts):
    return bool(facts.get("city"))
"""


def _city_facts_schema() -> MaterializedFactsSchema:
    return MaterializedFactsSchema(
        facts=[
            FactSpec(
                key="city",
                entity_ref="assignment.city",
                attribute="name",
                value_type=FactValueType.STR,
            )
        ]
    )


def test_registration_policy_accepts_valid_tool_module() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
import math

async def get_budget_hotels(conn, city):
    return [{"city": city, "score": math.floor(4.2)}]
""",
        kind=ArtifactKind.TOOL_MODULE,
        policy=policy,
    )

    assert errors == []


def test_registration_policy_rejects_forbidden_import() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
import os

async def get_budget_hotels(conn, city):
    return []
""",
        kind=ArtifactKind.TOOL_MODULE,
        policy=policy,
    )

    assert any(error.code == "forbidden_import" for error in errors)


def test_registration_policy_rejects_forbidden_symbol_call() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
async def get_budget_hotels(conn, city):
    open("bad.txt")
    return []
""",
        kind=ArtifactKind.TOOL_MODULE,
        policy=policy,
    )

    assert any(error.code in {"forbidden_symbol", "forbidden_call"} for error in errors)


def test_registration_policy_rejects_dunder_access() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def solve(tools):
    return {"x": tools.__class__}
""",
        kind=ArtifactKind.SOLUTION_MODULE,
        policy=policy,
    )

    assert any(error.code == "dunder_access_forbidden" for error in errors)


def test_registration_policy_rejects_dunder_subscript_access() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def solve(tools):
    data = {"__class__": "secret"}
    return {"x": data["__class__"]}
""",
        kind=ArtifactKind.SOLUTION_MODULE,
        policy=policy,
    )

    assert any(error.code == "dunder_subscript_forbidden" for error in errors)


def test_registration_policy_rejects_forbidden_subscript_call() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def solve(tools):
    def helper():
        return 1
    handlers = {"open": helper}
    return handlers["open"]()
""",
        kind=ArtifactKind.SOLUTION_MODULE,
        policy=policy,
    )

    assert any(error.code == "forbidden_subscript_call" for error in errors)


def test_registration_policy_allows_type_annotation_names() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def solve(tools: type) -> dict[str, int]:
    return {"x": 1}
""",
        kind=ArtifactKind.SOLUTION_MODULE,
        policy=policy,
    )

    assert all(error.code != "forbidden_symbol" for error in errors)
    assert errors == []


def test_registration_policy_rejects_top_level_executable_statement() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
_cache = [i * i for i in range(10)]

def solve(tools):
    return {"x": len(_cache)}
""",
        kind=ArtifactKind.SOLUTION_MODULE,
        policy=policy,
    )

    assert any(error.code == "top_level_statement_forbidden" for error in errors)


def test_registration_policy_visits_nested_class_body() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
class Bad:
    value = holder.__class__

def solve(tools):
    return {"x": 1}
""",
        kind=ArtifactKind.SOLUTION_MODULE,
        policy=policy,
    )

    assert any(error.code == "class_definition_forbidden" for error in errors)
    assert any(error.code == "dunder_access_forbidden" for error in errors)


def test_registration_policy_rejects_wrong_solution_signature() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def solve(answer, tools):
    return {"x": 1}
""",
        kind=ArtifactKind.SOLUTION_MODULE,
        policy=policy,
    )

    assert any(error.code == "solve_signature_invalid" for error in errors)


def test_registration_policy_requires_verifier_stage_functions() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def verify(answer, tools):
    return True
""",
        kind=ArtifactKind.VERIFIER_MODULE,
        policy=policy,
    )

    assert any(error.code == "missing_fetch_facts_function" for error in errors)
    assert any(error.code == "missing_facts_match_function" for error in errors)
    assert any(error.code == "missing_check_constraints_function" for error in errors)


def test_registration_policy_rejects_verifier_without_tool_grounded_fetch_facts() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def verify(answer, tools):
    facts = fetch_facts(answer, tools)
    if not facts_match_answer_claims(answer, facts):
        return False
    return check_constraints(answer, facts)

def fetch_facts(answer, tools):
    return {"city": answer.get("city")}

def facts_match_answer_claims(answer, facts):
    return answer.get("city") == facts.get("city")

def check_constraints(answer, facts):
    return True
""",
        kind=ArtifactKind.VERIFIER_MODULE,
        policy=policy,
    )

    assert any(error.code == "fetch_facts_tools_call_required" for error in errors)


def test_registration_policy_rejects_fetch_facts_without_answer_reference() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def verify(answer, tools):
    facts = fetch_facts(answer, tools)
    if not facts_match_answer_claims(answer, facts):
        return False
    return check_constraints(answer, facts)

def fetch_facts(answer, tools):
    return {"city": tools.lookup_default_city()}

def facts_match_answer_claims(answer, facts):
    return answer.get("city") == facts.get("city")

def check_constraints(answer, facts):
    return facts.get("city") is not None
""",
        kind=ArtifactKind.VERIFIER_MODULE,
        policy=policy,
    )

    assert any(error.code == "fetch_facts_answer_reference_required" for error in errors)


def test_registration_policy_rejects_verifier_without_stage_pipeline() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def verify(answer, tools):
    return True

def fetch_facts(answer, tools):
    return {"city": tools.lookup_city(answer.get("city"))}

def facts_match_answer_claims(answer, facts):
    return answer.get("city") == facts.get("city")

def check_constraints(answer, facts):
    return True
""",
        kind=ArtifactKind.VERIFIER_MODULE,
        policy=policy,
    )

    assert any(error.code == "verify_missing_stage_call" for error in errors)


def test_registration_policy_rejects_tool_calls_in_pure_verifier_stages() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def verify(answer, tools):
    facts = fetch_facts(answer, tools)
    if not facts_match_answer_claims(answer, facts):
        return False
    return check_constraints(answer, facts)

def fetch_facts(answer, tools):
    return {"city": tools.lookup_city(answer.get("city"))}

def facts_match_answer_claims(answer, facts):
    expected = tools.lookup_city(answer.get("city"))
    return expected == facts.get("city")

def check_constraints(answer, facts):
    return tools.is_city_allowed(facts.get("city"))
""",
        kind=ArtifactKind.VERIFIER_MODULE,
        policy=policy,
    )

    assert any(error.code == "facts_match_answer_claims_tools_call_forbidden" for error in errors)
    assert any(error.code == "check_constraints_tools_call_forbidden" for error in errors)


def test_registration_policy_rejects_trivial_pure_verifier_stages() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def verify(answer, tools):
    facts = fetch_facts(answer, tools)
    if not facts_match_answer_claims(answer, facts):
        return False
    return check_constraints(answer, facts)

def fetch_facts(answer, tools):
    return {"city": tools.lookup_city(answer.get("city"))}

def facts_match_answer_claims(answer, facts):
    return True

def check_constraints(answer, facts):
    return False
""",
        kind=ArtifactKind.VERIFIER_MODULE,
        policy=policy,
    )

    assert any(
        error.code == "facts_match_answer_claims_constant_boolean_return_forbidden"
        for error in errors
    )
    assert any(
        error.code == "check_constraints_constant_boolean_return_forbidden"
        for error in errors
    )
    assert any(
        error.code == "facts_match_answer_claims_answer_reference_required" for error in errors
    )
    assert any(
        error.code == "facts_match_answer_claims_facts_reference_required" for error in errors
    )
    assert any(error.code == "check_constraints_facts_reference_required" for error in errors)


def test_registration_policy_rejects_preaggregated_fetch_facts_metrics() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def verify(answer, tools):
    facts = fetch_facts(answer, tools)
    if not facts_match_answer_claims(answer, facts):
        return False
    return check_constraints(answer, facts)

def fetch_facts(answer, tools):
    rows = tools.lookup_cities(answer.get("cities", []))
    return {"city_count": len(rows)}

def facts_match_answer_claims(answer, facts):
    return True

def check_constraints(answer, facts):
    return facts.get("city_count", 0) > 0
""",
        kind=ArtifactKind.VERIFIER_MODULE,
        policy=policy,
    )

    assert any(error.code == "fetch_facts_aggregate_call_forbidden" for error in errors)


def test_registration_policy_accepts_valid_tool_self_test_module() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
async def run_self_test(tools):
    return {"ok": True}
""",
        kind=ArtifactKind.TOOL_SELF_TEST_MODULE,
        policy=policy,
    )

    assert errors == []


def test_registration_policy_rejects_wrong_tool_self_test_signature() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def run_self_test():
    return {"ok": True}
""",
        kind=ArtifactKind.TOOL_SELF_TEST_MODULE,
        policy=policy,
    )

    assert any(error.code == "run_self_test_signature_invalid" for error in errors)


def test_registration_policy_rejects_duplicate_required_function() -> None:
    policy = load_config("rl_task_foundry.yaml").synthesis.registration_policy
    errors = validate_generated_module(
        """
def solve(tools):
    return {"x": 1}

def solve(tools):
    return {"x": 2}
""",
        kind=ArtifactKind.SOLUTION_MODULE,
        policy=policy,
    )

    assert any(error.code == "duplicate_public_function" for error in errors)
    assert any(error.code == "duplicate_solve_function" for error in errors)


def test_runtime_isolation_plan_uses_lane_a_and_b_split() -> None:
    config = load_config("rl_task_foundry.yaml")

    plan = build_runtime_isolation_plan(config)

    assert plan.registration_lane.subprocess_required is True
    assert plan.registration_lane.worker_mode == "persistent_subprocess_pool"
    assert plan.registration_lane.db_access_strategy == "worker_owned_pool"
    assert plan.registration_lane.max_db_connections == (
        config.synthesis.registration_workers.worker_count
        * config.synthesis.registration_workers.connections_per_worker
    )
    assert plan.production_solver_lane.main_process_execution is True
    assert plan.production_solver_lane.per_tool_subprocess_roundtrip is False
    assert str(plan.registration_lane.adr_path) == "docs/adr/0001-custom-ast-preflight.md"


def test_registration_subprocess_pool_reuses_persistent_worker() -> None:
    async def _run() -> tuple[int, int]:
        config = load_config("rl_task_foundry.yaml")
        config.synthesis.registration_workers.worker_count = 1
        pool = await RegistrationSubprocessPool.start(config)
        try:
            first = await pool.validate_module(
                source="""
async def get_city(conn, customer_id):
    return {"customer_id": customer_id}
""",
                artifact_kind=ArtifactKind.TOOL_MODULE,
            )
            second = await pool.validate_module(
                source="""
def solve(answer):
    return {"x": 1}
""",
                artifact_kind=ArtifactKind.SOLUTION_MODULE,
            )
            assert any(error.code == "solve_signature_invalid" for error in second.errors)
            return first.worker_pid, second.worker_pid
        finally:
            await pool.close()

    first_pid, second_pid = asyncio.run(_run())
    assert first_pid == second_pid


def test_registration_subprocess_pool_round_robins_across_workers() -> None:
    async def _run() -> tuple[int, int]:
        config = load_config("rl_task_foundry.yaml")
        config.synthesis.registration_workers.worker_count = 2
        pool = await RegistrationSubprocessPool.start(config)
        try:
            first = await pool.validate_module(
                source="""
async def get_city(conn, customer_id):
    return {"customer_id": customer_id}
""",
                artifact_kind=ArtifactKind.TOOL_MODULE,
            )
            second = await pool.validate_module(
                source="""
async def get_country(conn, customer_id):
    return {"customer_id": customer_id}
""",
                artifact_kind=ArtifactKind.TOOL_MODULE,
            )
            return first.worker_pid, second.worker_pid
        finally:
            await pool.close()

    first_pid, second_pid = asyncio.run(_run())
    assert first_pid != second_pid


def test_registration_subprocess_pool_restarts_dead_worker() -> None:
    async def _run() -> tuple[int, int]:
        config = load_config("rl_task_foundry.yaml")
        config.synthesis.registration_workers.worker_count = 1
        pool = await RegistrationSubprocessPool.start(config)
        try:
            first = await pool.validate_module(
                source="""
async def get_city(conn, customer_id):
    return {"customer_id": customer_id}
""",
                artifact_kind=ArtifactKind.TOOL_MODULE,
            )
            worker = pool.workers[0]
            worker.process.kill()
            await worker.process.wait()
            second = await pool.validate_module(
                source="""
async def get_country(conn, customer_id):
    return {"customer_id": customer_id}
""",
                artifact_kind=ArtifactKind.TOOL_MODULE,
            )
            return first.worker_pid, second.worker_pid
        finally:
            await pool.close()

    first_pid, second_pid = asyncio.run(_run())
    assert first_pid != second_pid


def test_registration_subprocess_pool_executes_async_tool_entrypoint() -> None:
    async def _run() -> tuple[object | None, int | None]:
        config = load_config("rl_task_foundry.yaml")
        config.synthesis.registration_workers.worker_count = 1
        pool = await RegistrationSubprocessPool.start(config)
        try:
            result = await pool.execute_module_entrypoint(
                source="""
async def get_city(conn, customer_id):
    return {"customer_id": customer_id, "city": "sasebo"}
""",
                artifact_kind=ArtifactKind.TOOL_MODULE,
                entrypoint="get_city",
                args=[None, 7],
            )
            assert result.errors == []
            return result.return_value, result.call_count
        finally:
            await pool.close()

    return_value, call_count = asyncio.run(_run())
    assert return_value == {"customer_id": 7, "city": "sasebo"}
    assert call_count is not None
    assert call_count >= 1


def test_registration_subprocess_pool_reports_execution_error() -> None:
    async def _run() -> str:
        config = load_config("rl_task_foundry.yaml")
        pool = await RegistrationSubprocessPool.start(config)
        try:
            result = await pool.execute_module_entrypoint(
                source="""
def solve(tools):
    raise ValueError("boom")
""",
                artifact_kind=ArtifactKind.SOLUTION_MODULE,
                entrypoint="solve",
                args=[{}],
            )
            assert result.errors
            return result.errors[0].code
        finally:
            await pool.close()

    error_code = asyncio.run(_run())
    assert error_code == "execution_error"


def test_registration_subprocess_pool_enforces_call_count_limit() -> None:
    async def _run() -> str:
        config = load_config("rl_task_foundry.yaml")
        config.synthesis.registration_workers.call_count_limit = 8
        pool = await RegistrationSubprocessPool.start(config)
        try:
            result = await pool.execute_module_entrypoint(
                source="""
def helper(n):
    if n <= 0:
        return 0
    return helper(n - 1) + 1

def solve(tools):
    return helper(32)
""",
                artifact_kind=ArtifactKind.SOLUTION_MODULE,
                entrypoint="solve",
                args=[{}],
            )
            assert result.errors
            return result.errors[0].code
        finally:
            await pool.close()

    error_code = asyncio.run(_run())
    assert error_code == "call_count_limit_exceeded"


def test_registration_subprocess_pool_runs_tool_self_test() -> None:
    async def _run() -> tuple[object | None, int | None]:
        config = load_config("rl_task_foundry.yaml")
        pool = await RegistrationSubprocessPool.start(config)
        try:
            result = await pool.run_tool_self_test(
                tool_source="""
async def get_city(conn, customer_id):
    return {"customer_id": customer_id, "city": "sasebo"}
""",
                self_test_source="""
async def run_self_test(tools):
    row = await tools.get_city(11)
    assert row["city"] == "sasebo"
    return {"ok": True, "customer_id": row["customer_id"]}
""",
            )
            assert result.errors == []
            return result.return_value, result.call_count
        finally:
            await pool.close()

    return_value, call_count = asyncio.run(_run())
    assert return_value == {"ok": True, "customer_id": 11}
    assert call_count is not None
    assert call_count >= 1


def test_registration_subprocess_pool_reports_tool_self_test_failure() -> None:
    async def _run() -> str:
        config = load_config("rl_task_foundry.yaml")
        pool = await RegistrationSubprocessPool.start(config)
        try:
            result = await pool.run_tool_self_test(
                tool_source="""
async def get_city(conn, customer_id):
    return {"customer_id": customer_id, "city": "sasebo"}
""",
                self_test_source="""
async def run_self_test(tools):
    row = await tools.get_city(11)
    assert row["city"] == "busan"
    return {"ok": True}
""",
            )
            assert result.errors
            return result.errors[0].code
        finally:
            await pool.close()

    error_code = asyncio.run(_run())
    assert error_code == "execution_error"


def test_registration_subprocess_pool_runs_solution_verifier_self_consistency_check() -> None:
    async def _run(
        ) -> tuple[
            bool | None,
            bool | None,
            bool | None,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
        ]:
        config = load_config("rl_task_foundry.yaml")
        pool = await RegistrationSubprocessPool.start(config)
        try:
            result = await pool.run_self_consistency_check(
                tool_source="""
async def lookup_city(conn, customer_id):
    return {"city": "sasebo"}
""",
                solution_source="""
def solve(tools):
    return {"city": "sasebo"}
""",
                verifier_source="""
def verify(answer, tools):
    facts = fetch_facts(answer, tools)
    if not facts_match_answer_claims(answer, facts):
        return False
    return check_constraints(answer, facts)

def fetch_facts(answer, tools):
    return {"city": tools.lookup_city(answer.get("city"))["city"]}

def facts_match_answer_claims(answer, facts):
    return answer.get("city") == facts.get("city")

def check_constraints(answer, facts):
    return bool(facts.get("city"))
""",
                expected_fact_keys=["city"],
            )
            assert result.errors == []
            return (
                result.facts_match_result,
                result.check_constraints_result,
                result.verify_result,
                result.solution_tool_calls,
                result.verifier_tool_calls,
                result.fetch_facts_answer_reads,
                result.facts_match_answer_reads,
                result.facts_match_facts_reads,
                result.check_constraints_facts_reads,
            )
        finally:
            await pool.close()

    (
        facts_match_result,
        check_constraints_result,
        verify_result,
        solution_tool_calls,
        verifier_tool_calls,
        fetch_facts_answer_reads,
        facts_match_answer_reads,
        facts_match_facts_reads,
        check_constraints_facts_reads,
    ) = asyncio.run(_run())
    assert facts_match_result is True
    assert check_constraints_result is True
    assert verify_result is True
    assert solution_tool_calls is not None
    assert verifier_tool_calls is not None
    assert fetch_facts_answer_reads is not None
    assert fetch_facts_answer_reads >= 1
    assert facts_match_answer_reads is not None
    assert facts_match_answer_reads >= 1
    assert facts_match_facts_reads is not None
    assert facts_match_facts_reads >= 1
    assert check_constraints_facts_reads is not None
    assert check_constraints_facts_reads >= 1


def test_registration_subprocess_pool_reports_failed_self_consistency_check() -> None:
    async def _run() -> tuple[bool | None, bool | None, bool | None, list[str]]:
        config = load_config("rl_task_foundry.yaml")
        pool = await RegistrationSubprocessPool.start(config)
        try:
            result = await pool.run_self_consistency_check(
                tool_source="""
async def lookup_city(conn, customer_id):
    return {"city": "sasebo"}
""",
                solution_source="""
def solve(tools):
    return {"city": "busan"}
""",
                verifier_source="""
def verify(answer, tools):
    facts = fetch_facts(answer, tools)
    if not facts_match_answer_claims(answer, facts):
        return False
    return check_constraints(answer, facts)

def fetch_facts(answer, tools):
    return {"city": tools.lookup_city(answer.get("city"))["city"]}

def facts_match_answer_claims(answer, facts):
    return answer.get("city") == facts.get("city")

def check_constraints(answer, facts):
    return bool(facts.get("city"))
""",
                expected_fact_keys=["city"],
            )
            return (
                result.facts_match_result,
                result.check_constraints_result,
                result.verify_result,
                [error.code for error in result.errors],
            )
        finally:
            await pool.close()

    facts_match_result, check_constraints_result, verify_result, error_codes = asyncio.run(_run())
    assert facts_match_result is False
    assert check_constraints_result is True
    assert verify_result is False
    assert error_codes == []


def test_registration_runner_passes_valid_bundle() -> None:
    async def _run() -> tuple[str, bool, bool, int, int, int, int, int, int, int, list[str]]:
        config = load_config("rl_task_foundry.yaml")
        bundle = GeneratedArtifactBundle(
            tool_source="""
async def get_city(conn, customer_id):
    return {"customer_id": customer_id, "city": "sasebo"}
""",
            tool_self_test_source="""
async def run_self_test(tools):
    row = await tools.get_city(9)
    assert row["city"] == "sasebo"
    return {"ok": True}
""",
            solution_source="""
def solve(tools):
    return {"city": "sasebo"}
""",
            verifier_source=_valid_verifier_source(),
            shadow_verifier_source=_valid_verifier_source(),
        )
        report = await run_registration_bundle(
            config=config,
            bundle=bundle,
            verifier_probe_specs={
                RegistrationArtifactName.VERIFIER: VerifierProbeSpec(
                    answer_sample={"city": "sample_city"},
                    facts_schema=_city_facts_schema(),
                ),
                RegistrationArtifactName.SHADOW_VERIFIER: VerifierProbeSpec(
                    answer_sample={"city": "sample_city"},
                    facts_schema=_city_facts_schema(),
                ),
            },
        )
        return (
            report.status,
            report.tool_self_test.executed,
            report.tool_self_test.passed,
            report.verifier.verifier_hybrid_analysis.fetch_facts_tool_calls,
            report.verifier.verifier_hybrid_analysis.facts_match_answer_references,
            report.verifier.verifier_hybrid_analysis.facts_match_facts_references,
            report.verifier.verifier_hybrid_analysis.check_constraints_facts_references,
            report.verifier.verifier_execution_probe.fetch_facts_answer_reads,
            report.verifier.verifier_execution_probe.facts_match_answer_reads,
            report.verifier.verifier_execution_probe.facts_match_facts_reads,
            report.verifier.verifier_execution_probe.check_constraints_facts_reads,
            report.verifier.verifier_execution_probe.fetch_facts_return_keys,
        )

    (
        status,
        executed,
        self_test_passed,
        fetch_facts_tool_calls,
        facts_match_answer_references,
        facts_match_facts_references,
        check_constraints_facts_references,
        probe_fetch_facts_answer_reads,
        probe_facts_match_answer_reads,
        probe_facts_match_facts_reads,
        probe_check_constraints_facts_reads,
        fetch_facts_return_keys,
    ) = asyncio.run(_run())
    assert status == RegistrationBundleStatus.PASSED
    assert executed is True
    assert self_test_passed is True
    assert fetch_facts_tool_calls == 1
    assert facts_match_answer_references >= 1
    assert facts_match_facts_references >= 1
    assert check_constraints_facts_references >= 1
    assert probe_fetch_facts_answer_reads >= 1
    assert probe_facts_match_answer_reads >= 1
    assert probe_facts_match_facts_reads >= 1
    assert probe_check_constraints_facts_reads >= 1
    assert fetch_facts_return_keys == ["city"]


def test_registration_runner_fails_when_verifier_probe_detects_fact_schema_mismatch() -> None:
    async def _run() -> tuple[str, str]:
        config = load_config("rl_task_foundry.yaml")
        bundle = GeneratedArtifactBundle(
            tool_source="""
async def get_city(conn, customer_id):
    return {"customer_id": customer_id, "city": "sasebo"}
""",
            tool_self_test_source="""
async def run_self_test(tools):
    return {"ok": True}
""",
            solution_source="""
def solve(tools):
    return {"city": "sasebo"}
""",
            verifier_source="""
def verify(answer, tools):
    facts = fetch_facts(answer, tools)
    if not facts_match_answer_claims(answer, facts):
        return False
    return check_constraints(answer, facts)

def fetch_facts(answer, tools):
    return {"country": tools.lookup_city(answer.get("city"))}

def facts_match_answer_claims(answer, facts):
    return answer.get("city") == facts.get("country")

def check_constraints(answer, facts):
    return bool(facts.get("country"))
""",
            shadow_verifier_source=_valid_verifier_source(),
        )
        report = await run_registration_bundle(
            config=config,
            bundle=bundle,
            verifier_probe_specs={
                RegistrationArtifactName.VERIFIER: VerifierProbeSpec(
                    answer_sample={"city": "sample_city"},
                    facts_schema=_city_facts_schema(),
                )
            },
        )
        return report.status, report.verifier.probe_errors[0].code

    status, error_code = asyncio.run(_run())
    assert status == RegistrationBundleStatus.FAILED
    assert error_code == "facts_schema_keys_mismatch"


def test_registration_runner_fails_when_static_validation_fails() -> None:
    async def _run() -> tuple[str, bool, str]:
        config = load_config("rl_task_foundry.yaml")
        bundle = GeneratedArtifactBundle(
            tool_source="""
async def get_city(conn, customer_id):
    return {"customer_id": customer_id, "city": "sasebo"}
""",
            tool_self_test_source="""
async def run_self_test(tools):
    return {"ok": True}
""",
            solution_source="""
def solve(answer, tools):
    return {"city": "sasebo"}
""",
            verifier_source=_valid_verifier_source(),
            shadow_verifier_source=_valid_verifier_source(),
        )
        report = await run_registration_bundle(config=config, bundle=bundle)
        return report.status, report.tool_self_test.executed, report.solution.static_errors[0].code

    status, executed, error_code = asyncio.run(_run())
    assert status == RegistrationBundleStatus.FAILED
    assert executed is True
    assert error_code == "solve_signature_invalid"


def test_registration_runner_surfaces_tool_self_test_execution_failure() -> None:
    async def _run() -> tuple[str, str]:
        config = load_config("rl_task_foundry.yaml")
        bundle = GeneratedArtifactBundle(
            tool_source="""
async def get_city(conn, customer_id):
    return {"customer_id": customer_id, "city": "sasebo"}
""",
            tool_self_test_source="""
async def run_self_test(tools):
    row = await tools.get_city(9)
    assert row["city"] == "busan"
    return {"ok": True}
""",
            solution_source="""
def solve(tools):
    return {"city": "sasebo"}
""",
            verifier_source=_valid_verifier_source(),
            shadow_verifier_source=_valid_verifier_source(),
        )
        report = await run_registration_bundle(config=config, bundle=bundle)
        return report.status, report.tool_self_test.execution_errors[0].code

    status, error_code = asyncio.run(_run())
    assert status == RegistrationBundleStatus.FAILED
    assert error_code == "execution_error"
