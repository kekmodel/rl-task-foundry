from pathlib import Path

from rl_task_foundry.config import load_config


def test_load_config_uses_solver_replicas_source_of_truth():
    config = load_config(Path("rl_task_foundry.yaml"))
    assert config.models.total_solver_replicas == 6
    assert config.output.run_db_path.name == "run.db"
    assert config.database.dsn == "postgresql://sakila:sakila@127.0.0.1:5433/sakila"
    assert config.budget.max_gpu_hours is None
    assert "codex_oauth" in config.providers
    assert "local_server" in config.providers
    assert config.providers["local_server"].base_url == "http://127.0.0.1:8000/v1"
    assert config.atomic_tools.max_tool_count == 256
    assert config.atomic_tools.bounded_result_limit == 100
    assert config.atomic_tools.max_batch_values == 128
    assert config.atomic_tools.float_precision == 2
    assert config.verification.shadow_sample_rate == 0.1
    assert config.synthesis.runtime.max_turns == 8
    assert config.synthesis.runtime.explicit_memory_window == 8
    assert config.synthesis.runtime.max_self_consistency_iterations == 5
    assert config.synthesis.runtime.max_difficulty_cranks == 6
    assert config.synthesis.runtime.max_consecutive_category_discards == 3
    assert config.synthesis.runtime.category_backoff_duration_s == 3600
    assert config.synthesis.coverage_planner.target_count_per_band == 3
    assert config.synthesis.coverage_planner.include_unset_band is False
    assert config.synthesis.registration_workers.worker_count == 2
    assert config.synthesis.registration_workers.max_db_connections == 4
    assert config.estimated_total_db_connections == 44


def test_load_config_applies_runtime_model_overrides():
    config = load_config(
        Path("rl_task_foundry.yaml"),
        composer_provider="local_server",
        solver_backbone_provider="local_server",
        solver_provider="local_server",
        solver_model="local-gpt",
    )
    assert config.models.composer.provider == "local_server"
    assert config.models.solver_backbone.provider == "local_server"
    assert all(solver.provider == "local_server" for solver in config.models.solvers)
    assert all(solver.model == "local-gpt" for solver in config.models.solvers)
