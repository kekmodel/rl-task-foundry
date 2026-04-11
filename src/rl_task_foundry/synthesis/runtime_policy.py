"""Runtime lane planning for synthesis-generated code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rl_task_foundry.config.models import AppConfig


class ExecutionLane(StrEnum):
    REGISTRATION_VERIFICATION = "registration_verification"
    PRODUCTION_SOLVER = "production_solver"


@dataclass(slots=True)
class RegistrationLanePlan:
    lane: ExecutionLane
    worker_count: int
    connections_per_worker: int
    max_db_connections: int
    task_timeout_s: int
    memory_limit_mb: int
    call_count_limit: int
    subprocess_required: bool


@dataclass(slots=True)
class ProductionSolverLanePlan:
    lane: ExecutionLane
    main_process_execution: bool
    uses_database_pools: bool
    per_tool_subprocess_roundtrip: bool


@dataclass(slots=True)
class RuntimeIsolationPlan:
    registration_lane: RegistrationLanePlan
    production_solver_lane: ProductionSolverLanePlan
    estimated_total_db_connections: int


def build_runtime_isolation_plan(config: AppConfig) -> RuntimeIsolationPlan:
    workers = config.synthesis.registration_workers
    return RuntimeIsolationPlan(
        registration_lane=RegistrationLanePlan(
            lane=ExecutionLane.REGISTRATION_VERIFICATION,
            worker_count=workers.worker_count,
            connections_per_worker=workers.connections_per_worker,
            max_db_connections=workers.max_db_connections,
            task_timeout_s=workers.task_timeout_s,
            memory_limit_mb=workers.memory_limit_mb,
            call_count_limit=workers.call_count_limit,
            subprocess_required=config.synthesis.registration_policy.require_subprocess_lane,
        ),
        production_solver_lane=ProductionSolverLanePlan(
            lane=ExecutionLane.PRODUCTION_SOLVER,
            main_process_execution=True,
            uses_database_pools=True,
            per_tool_subprocess_roundtrip=False,
        ),
        estimated_total_db_connections=config.estimated_total_db_connections,
    )
