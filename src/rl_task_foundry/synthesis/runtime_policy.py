"""Runtime lane planning for synthesis-generated code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from rl_task_foundry.config.models import AppConfig


class ExecutionLane(StrEnum):
    REGISTRATION_VERIFICATION = "registration_verification"
    PRODUCTION_SOLVER = "production_solver"


class RegistrationWorkerMode(StrEnum):
    PERSISTENT_SUBPROCESS_POOL = "persistent_subprocess_pool"


class RegistrationDbAccessStrategy(StrEnum):
    WORKER_OWNED_POOL = "worker_owned_pool"


@dataclass(slots=True)
class RegistrationLanePlan:
    lane: ExecutionLane
    worker_mode: RegistrationWorkerMode
    db_access_strategy: RegistrationDbAccessStrategy
    worker_count: int
    connections_per_worker: int
    max_db_connections: int
    task_timeout_s: int
    memory_limit_mb: int
    call_count_limit: int
    subprocess_required: bool
    adr_path: Path


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
            worker_mode=RegistrationWorkerMode.PERSISTENT_SUBPROCESS_POOL,
            db_access_strategy=RegistrationDbAccessStrategy.WORKER_OWNED_POOL,
            worker_count=workers.worker_count,
            connections_per_worker=workers.connections_per_worker,
            max_db_connections=workers.max_db_connections,
            task_timeout_s=workers.task_timeout_s,
            memory_limit_mb=workers.memory_limit_mb,
            call_count_limit=workers.call_count_limit,
            subprocess_required=config.synthesis.registration_policy.require_subprocess_lane,
            adr_path=Path("docs/adr/0001-custom-ast-preflight.md"),
        ),
        production_solver_lane=ProductionSolverLanePlan(
            lane=ExecutionLane.PRODUCTION_SOLVER,
            main_process_execution=True,
            uses_database_pools=True,
            per_tool_subprocess_roundtrip=False,
        ),
        estimated_total_db_connections=config.estimated_total_db_connections,
    )
