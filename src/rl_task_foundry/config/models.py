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
    memory_mode: Literal["none", "explicit_summary", "session_only"] = "none"
    summarization_mode: Literal["off", "explicit"] = "off"


def derive_solver_id(model: str, index: int) -> str:
    """Derive a stable solver_id from the model name and a per-model index.

    The schema-level source of truth is ``(provider, model, index)``; keeping
    the model in the id prevents two runs at different models from colliding
    in ``verification_results (task_id, solver_id)``.
    """
    slug = model.replace("/", "-")
    return f"{slug}_{index:02d}"


class ModelsConfig(StrictModel):
    composer: ModelRef
    solvers: list[SolverModelConfig]

    @model_validator(mode="before")
    @classmethod
    def _fill_missing_solver_ids(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        solvers = data.get("solvers")
        if not isinstance(solvers, list):
            return data
        counters: dict[str, int] = {}
        filled: list[object] = []
        for entry in solvers:
            if not isinstance(entry, dict):
                filled.append(entry)
                continue
            if entry.get("solver_id"):
                filled.append(entry)
                continue
            model = entry.get("model")
            if not isinstance(model, str):
                filled.append(entry)
                continue
            index = counters.get(model, 0)
            counters[model] = index + 1
            filled.append({**entry, "solver_id": derive_solver_id(model, index)})
        return {**data, "solvers": filled}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_solver_runs(self) -> int:
        return len(self.solvers)


class AtomicToolConfig(StrictModel):
    max_tools: int = Field(default=300, ge=8)
    bounded_result_limit: int = Field(default=100, ge=1)
    max_batch_values: int = Field(default=128, ge=1)
    float_precision: int = Field(default=2, ge=0)


class SynthesisRuntimeConfig(StrictModel):
    max_turns: int = Field(default=50, ge=1)
    tracing: bool = True
    sdk_sessions_enabled: bool = False
    max_generation_attempts: int = Field(default=5, ge=1)
    max_consecutive_category_discards: int = Field(default=3, ge=1)
    category_backoff_duration_s: int = Field(default=3600, ge=1)
    schema_summary_max_tables: int = Field(default=32, ge=1)
    tool_surface_summary_max_entries: int = Field(default=24, ge=1)
    prompt_schema_orientation_max_tables: int = Field(default=8, ge=1)
    prompt_schema_orientation_max_columns: int = Field(default=8, ge=1)
    prompt_tool_surface_hint_limit: int = Field(default=16, ge=1)
    label_preview_field_limit: int = Field(default=8, ge=1)
    diagnostic_item_limit: int = Field(default=5, ge=1)
    recent_tool_call_limit: int = Field(default=20, ge=1)
    payload_preview_max_string_length: int = Field(default=400, ge=1)
    payload_preview_max_list_items: int = Field(default=3, ge=1)
    payload_preview_max_dict_items: int = Field(default=6, ge=1)


class SynthesisCoveragePlannerConfig(StrictModel):
    target_count_per_band: int = Field(default=3, ge=1)


class SynthesisConfig(StrictModel):
    runtime: SynthesisRuntimeConfig = Field(default_factory=SynthesisRuntimeConfig)
    coverage_planner: SynthesisCoveragePlannerConfig = Field(
        default_factory=SynthesisCoveragePlannerConfig
    )
    parallel_workers: int = Field(default=1, ge=1)


class SolverRuntimeConfig(StrictModel):
    max_turns: int = 16
    tracing: bool = True
    sdk_sessions_enabled: bool = True


class CalibrationConfig(StrictModel):
    lower_pass_rate: float = Field(ge=0.0, le=1.0)
    upper_pass_rate: float = Field(ge=0.0, le=1.0)
    ci_alpha: float = Field(default=0.1, ge=0.0, le=1.0)
    solver_batch_size: int = Field(default=3, ge=1)
    max_solver_runs: int = Field(default=30, ge=1)
    safe_early_termination: bool = True
    max_divergence_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class ProviderResilienceConfig(StrictModel):
    circuit_breaker_window_s: int = Field(default=60, ge=1)
    circuit_breaker_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    probe_interval_s: int = Field(default=30, ge=1)
    minimum_request_count: int = Field(default=2, ge=1)
    release_semaphore_on_backoff: bool = True


class DedupConfig(StrictModel):
    exact_enabled: bool = True
    near_dup_enabled: bool = True
    minhash_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class TaskRegistryConfig(StrictModel):
    minhash_num_perm: int = Field(default=128, ge=1)
    default_query_limit: int = Field(default=20, ge=1)
    semantic_shingle_size: int = Field(default=3, ge=1)


class BudgetConfig(StrictModel):
    max_run_usd: float = Field(ge=0.0)
    max_gpu_hours: float | None = Field(default=None, ge=0.0)
    compose_phase_usd: float = Field(ge=0.0)
    solve_phase_usd: float = Field(ge=0.0)
    min_accept_rate_attempts: int = Field(default=10, ge=1)
    reserve_strategy: Literal["phase_specific"] = "phase_specific"


class PrivacyConfig(StrictModel):
    default_visibility: Literal["blocked", "internal", "user_visible"] = "blocked"
    visibility_overrides: dict[str, Literal["blocked", "internal", "user_visible"]] = Field(
        default_factory=dict
    )


class OutputConfig(StrictModel):
    run_db_path: Path
    traces_dir: Path


class AppConfig(StrictModel):
    database: DatabaseConfig
    domain: DomainConfig
    providers: dict[str, ProviderConfig]
    models: ModelsConfig
    provider_resilience: ProviderResilienceConfig = Field(default_factory=ProviderResilienceConfig)
    atomic_tools: AtomicToolConfig = Field(default_factory=AtomicToolConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    solver_runtime: SolverRuntimeConfig
    calibration: CalibrationConfig
    dedup: DedupConfig
    task_registry: TaskRegistryConfig = Field(default_factory=TaskRegistryConfig)
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
