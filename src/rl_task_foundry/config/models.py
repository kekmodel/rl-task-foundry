"""Application configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


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


class DomainConfig(StrictModel):
    name: str
    language: str = Field(default="ko", min_length=2)
    user_role: str = "end user"
    agent_role: str = "organization AI assistant"
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


class ToolCompilerConfig(StrictModel):
    max_hops: int = 4
    allow_aggregates: bool = True
    allow_timelines: bool = True
    max_list_cardinality: int = 20
    naming_temperature_l2: float = Field(default=1.0, ge=0.0, le=2.0)
    business_alias_overrides: dict[str, str] = Field(default_factory=dict)


class TaskComposerConfig(StrictModel):
    label_tier: Literal["A", "B"] = "A"
    question_families: list[str]
    selected_tool_level: Literal[1, 2] = 1
    negative_outcome_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    max_attempts_per_anchor: int = 6
    anchor_samples_per_source: int = Field(default=3, ge=1)
    question_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    question_validation_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    family_min_required_hops: dict[str, int] = Field(default_factory=dict)
    max_status_lookup_answer_fields: int = Field(default=2, ge=1, le=4)
    enable_exists_status_lookup: bool = True
    exclude_answer_column_patterns: list[str] = Field(default_factory=list)
    exclude_anchor_table_patterns: list[str] = Field(
        default_factory=lambda: [
            "(^|_)(city|country|state|province|region|language|category|taxonomy|dimension|lookup|mapping|xref|bridge|association|relation)($|_)"
        ]
    )
    aggregate_discouraged_target_patterns: list[str] = Field(default_factory=list)
    causal_discouraged_target_patterns: list[str] = Field(
        default_factory=lambda: [
            "(^|_)(address|city|country|state|province|region|zipcode|postal|postcode|currency|timezone)($|_)"
        ]
    )
    causal_preferred_answer_patterns: list[str] = Field(
        default_factory=lambda: [
            "(^|_)(status|category|type|kind|title|label|code|language|method|channel|plan|tier|level|provider|carrier|reason|option|mode|format|service|policy|destination)($|_)"
        ]
    )
    causal_min_scalar_hops: int = Field(default=3, ge=2)


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
    canary_replica_count: int = Field(default=2, ge=1)
    post_canary_batch_size: int = Field(default=2, ge=1)
    full_replica_limit: int = Field(default=6, ge=1)
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


class VerificationConfig(StrictModel):
    require_provenance: bool = True
    fail_on_internal_field_leak: bool = True
    float_precision: int = Field(default=6, ge=0, le=12)
    shadow_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)


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
    tool_compiler: ToolCompilerConfig
    task_composer: TaskComposerConfig
    solver_runtime: SolverRuntimeConfig
    calibration: CalibrationConfig
    verification: VerificationConfig
    dedup: DedupConfig
    budget: BudgetConfig
    privacy: PrivacyConfig
    output: OutputConfig
