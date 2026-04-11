from __future__ import annotations

from rl_task_foundry.config import load_config
from rl_task_foundry.synthesis.registration_policy import (
    ArtifactKind,
    validate_generated_module,
)
from rl_task_foundry.synthesis.runtime_policy import build_runtime_isolation_plan


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


def test_runtime_isolation_plan_uses_lane_a_and_b_split() -> None:
    config = load_config("rl_task_foundry.yaml")

    plan = build_runtime_isolation_plan(config)

    assert plan.registration_lane.subprocess_required is True
    assert plan.registration_lane.max_db_connections == (
        config.synthesis.registration_workers.worker_count
        * config.synthesis.registration_workers.connections_per_worker
    )
    assert plan.production_solver_lane.main_process_execution is True
    assert plan.production_solver_lane.per_tool_subprocess_roundtrip is False
