from __future__ import annotations

import asyncio

from rl_task_foundry.config import load_config
from rl_task_foundry.synthesis.registration_policy import (
    ArtifactKind,
    validate_generated_module,
)
from rl_task_foundry.synthesis.runtime_policy import build_runtime_isolation_plan
from rl_task_foundry.synthesis.subprocess_pool import RegistrationSubprocessPool


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
