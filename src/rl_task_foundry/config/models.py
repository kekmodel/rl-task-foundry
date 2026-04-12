"""Application configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class StrictModel(BaseModel):
    """Base model with strict extra-field rejection."""

    model_config = ConfigDict(extra="forbid")


class DatabaseConfig(StrictModel):
    dsn: str
    schema_allowlist: list[str] = Field(default_factory=lambda: ["public"])
    readonly_role: str
    statement_timeout_ms: int = 5000
    lock_timeout_ms: int = 1000
    idle_tx_timeout_ms: int = 5000
    solver_pool_size: int = 32
    control_pool_size: int = 8
    max_total_connections: int | None = Field(default=None, ge=1)


class DomainConfig(StrictModel):
    name: str
    language: str = Field(default="ko", min_length=2)
    scenario_description: str = (
        "an end user asking the organization that owns the database for help or information"
    )


class ProviderConfig(StrictModel):
    type: Literal["openai_compatible", "anthropic", "openai", "google"]
    base_url: str | None = None
    api_key_env: str
    max_concurrency: int = 8
    timeout_s: int = 120


class ModelRef(StrictModel):
    provider: str
    model: str


class SolverModelConfig(StrictModel):
    solver_id: str
    backend: Literal["openai_agents"] = "openai_agents"
    provider: str
    model: str
    replicas: int = Field(default=1, ge=1)
    memory_mode: Literal["none", "explicit_summary", "session_only"] = "none"
    summarization_mode: Literal["off", "explicit"] = "off"


class ModelsConfig(StrictModel):
    composer: ModelRef
    solver_backbone: ModelRef
    solvers: list[SolverModelConfig]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_solver_replicas(self) -> int:
        return sum(solver.replicas for solver in self.solvers)


class AtomicToolConfig(StrictModel):
    max_tool_count: int = Field(default=256, ge=8)
    bounded_result_limit: int = Field(default=100, ge=1)
    max_batch_values: int = Field(default=128, ge=1)
    float_precision: int = Field(default=2, ge=0)


class SynthesisRuntimeConfig(StrictModel):
    max_turns: int = Field(default=8, ge=1)
    tracing: bool = True
    sdk_sessions_enabled: bool = True
    explicit_memory_window: int = Field(default=8, ge=1)
    max_generation_attempts: int = Field(default=5, ge=1)
    max_difficulty_cranks: int = Field(default=6, ge=1)
    max_consecutive_category_discards: int = Field(default=3, ge=1)
    category_backoff_duration_s: int = Field(default=3600, ge=1)


class SynthesisCoveragePlannerConfig(StrictModel):
    target_count_per_band: int = Field(default=3, ge=1)
    include_unset_band: bool = False


class SynthesisConfig(StrictModel):
    runtime: SynthesisRuntimeConfig = Field(default_factory=SynthesisRuntimeConfig)
    coverage_planner: SynthesisCoveragePlannerConfig = Field(
        default_factory=SynthesisCoveragePlannerConfig
    )


class SolverRuntimeConfig(StrictModel):
    max_turns: int = 16
    structured_output_required: bool = True
    tracing: bool = True
    sdk_sessions_enabled: bool = True
    canonical_state_store: Literal["run_db"] = "run_db"


class CalibrationConfig(StrictModel):
    lower_pass_rate: float = Field(ge=0.0, le=1.0)
    upper_pass_rate: float = Field(ge=0.0, le=1.0)
    ci_alpha: float = Field(default=0.1, ge=0.0, le=1.0)
    canary_replica_count: int = Field(default=3, ge=1)
    post_canary_batch_size: int = Field(default=3, ge=1)
    full_replica_limit: int = Field(default=30, ge=1)
    safe_early_termination: bool = True
    difficulty_weight_hops: float = 100.0
    difficulty_weight_family: float = 10.0
    difficulty_weight_outcome: float = 3.0
    difficulty_weight_answer_width: float = 1.0
    difficulty_weight_fanout: float = 0.1
    difficulty_weight_shortcuts: float = 0.5
    difficulty_fanout_cap: float = Field(default=99.0, ge=1.0)
    difficulty_family_order: dict[str, int] = Field(
        default_factory=lambda: {
            "status_lookup": 0,
            "timeline_resolution": 1,
            "causal_chain": 2,
            "aggregate_verification": 3,
        }
    )
    difficulty_outcome_order: dict[str, int] = Field(
        default_factory=lambda: {
            "answer": 0,
            "no_result": 1,
            "clarify": 2,
            "deny": 2,
        }
    )


class ProviderResilienceConfig(StrictModel):
    circuit_breaker_window_s: int = Field(default=60, ge=1)
    circuit_breaker_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    probe_interval_s: int = Field(default=30, ge=1)
    release_semaphore_on_backoff: bool = True


class DedupConfig(StrictModel):
    exact_enabled: bool = True
    near_dup_enabled: bool = True
    minhash_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class BudgetConfig(StrictModel):
    max_run_usd: float = Field(ge=0.0)
    max_gpu_hours: float | None = Field(default=None, ge=0.0)
    compose_phase_usd: float = Field(ge=0.0)
    solve_phase_usd: float = Field(ge=0.0)
    reserve_strategy: Literal["phase_specific"] = "phase_specific"


class PrivacyConfig(StrictModel):
    default_visibility: Literal["blocked", "internal", "user_visible"] = "blocked"
    visibility_overrides: dict[str, Literal["blocked", "internal", "user_visible"]] = (
        Field(default_factory=dict)
    )


class OutputConfig(StrictModel):
    run_db_path: Path
    accepted_jsonl_path: Path
    rejected_jsonl_path: Path
    events_jsonl_path: Path
    traces_dir: Path


class AppConfig(StrictModel):
    database: DatabaseConfig
    domain: DomainConfig
    providers: dict[str, ProviderConfig]
    models: ModelsConfig
    provider_resilience: ProviderResilienceConfig = Field(
        default_factory=ProviderResilienceConfig
    )
    atomic_tools: AtomicToolConfig = Field(default_factory=AtomicToolConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    solver_runtime: SolverRuntimeConfig
    calibration: CalibrationConfig
    dedup: DedupConfig
    budget: BudgetConfig
    privacy: PrivacyConfig
    output: OutputConfig

    @computed_field  # type: ignore[prop-decorator]
    @property
    def estimated_total_db_connections(self) -> int:
        return self.database.solver_pool_size + self.database.control_pool_size

    @model_validator(mode="after")
    def _validate_connection_budget(self) -> AppConfig:
        if (
            self.database.max_total_connections is not None
            and self.estimated_total_db_connections > self.database.max_total_connections
        ):
            raise ValueError(
                "estimated_total_db_connections exceeds database.max_total_connections"
            )
        return self
