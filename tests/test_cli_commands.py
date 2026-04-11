import json
from pathlib import Path

from typer.testing import CliRunner

from rl_task_foundry.cli import app
from rl_task_foundry.infra.storage import bootstrap_run_db, connect_run_db, record_accepted_example, record_event, record_run, record_task, record_verification_result
from rl_task_foundry.pipeline.orchestrator import RunSummary
from rl_task_foundry.tasks.models import TaskSpec


def test_cli_validate_config_command():
    result = CliRunner().invoke(app, ["validate-config"])
    normalized = result.stdout.replace("\n", "")
    assert result.exit_code == 0
    assert "solver_replicas=6" in normalized
    assert "composer=codex_oauth/gpt-5.4-mini" in normalized
    assert "label_tier=A" in normalized
    assert "selected_tool_level=1" in normalized
    assert "negative_outcome_ratio=0.2" in normalized
    assert "float_precision=6" in normalized
    assert "shadow_sample_rate=0.1" in normalized
    assert "registration_lane=" in normalized
    assert "workers=2" in normalized
    assert "connections_per_worker=2" in normalized
    assert "max_db_connections=4" in normalized
    assert "mode=persistent_subprocess_pool" in normalized
    assert "db_access=worker_owned_pool" in normalized
    assert "solver_lane=main_process=True,per_tool_subprocess=False" in normalized
    assert "registration_guards=timeout_s=30,memory_limit_mb=256,call_count_limit=256" in normalized
    assert "estimated_total_db_connections=44" in normalized
    assert "registration_policy_adr=docs/adr/0001-custom-ast-preflight.md" in normalized


def test_cli_generate_task_specs_writes_jsonl(monkeypatch, tmp_path):
    output_path = tmp_path / "tasks.jsonl"

    async def _fake_introspect(self):
        return object()

    class _DummyFactory:
        def __init__(self, **_kwargs):
            pass

        async def generate(self, graph, catalog, *, limit, path_ids=None):
            assert graph is not None
            assert catalog is not None
            assert limit == 1
            assert path_ids == ["customer.address.city"]
            return [
                TaskSpec.model_validate(
                    {
                        "task_id": "task_1",
                        "anchor_table": "customer",
                        "anchor_pk_column": "customer_id",
                        "anchor_pk_value": "1",
                        "domain": "customer_support",
                        "language": "ko",
                        "label_tier": "A",
                        "question_family": "status_lookup",
                        "question": "연결된 city 값을 확인해줘.",
                        "outcome_type": "answer",
                        "answer_schema": {
                            "fields": [
                                {
                                    "name": "city",
                                    "type": "string",
                                    "canonicalizer": "lower_trim",
                                    "source_columns": ["city.city"],
                                }
                            ]
                        },
                        "selected_path_id": "customer.address.city",
                        "required_hops": 2,
                        "tool_level": 1,
                        "tool_bundle_id": "customer.address.city::canonical::L1",
                        "sensitivity_policy": "default",
                    }
                )
            ]

    monkeypatch.setattr("rl_task_foundry.cli.PostgresSchemaIntrospector.introspect", _fake_introspect)
    monkeypatch.setattr("rl_task_foundry.cli.build_path_catalog", lambda graph, max_hops: object())
    monkeypatch.setattr("rl_task_foundry.cli.TierATaskFactory", _DummyFactory)

    result = CliRunner().invoke(
        app,
        [
            "generate-task-specs",
            str(output_path),
            "--limit",
            "1",
            "--path-id",
            "customer.address.city",
        ],
    )

    assert result.exit_code == 0
    assert "generated task specs" in result.stdout
    assert "tasks=1" in result.stdout
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["task_id"] == "task_1"


def test_cli_generate_review_pack_writes_outputs(monkeypatch, tmp_path):
    output_dir = tmp_path / "review"

    class _DummyReviewPackBuilder:
        def __init__(self, _config):
            pass

        async def build_entries(self, *, limit, path_ids=None, task_specs=None):
            assert limit == 1
            assert path_ids == ["customer.address.city"]
            assert task_specs is None
            return [
                {
                    "task_id": "task_review_1",
                    "review_surface": {
                        "question": "제 주소 기준 도시 정보를 알려주세요.",
                        "answer_schema": {"fields": []},
                        "tool_bundle": {"tools": []},
                    },
                    "review_notes": {
                        "seed_question": "연결된 city 값을 확인해줘.",
                        "question_strategy": "model_generated",
                        "label_tier": "A",
                        "tool_level": 1,
                        "question_family": "status_lookup",
                        "outcome_type": "answer",
                        "selected_path_id": "customer.address.city",
                        "required_hops": 2,
                        "difficulty_features": {},
                        "presentation_generation_metadata": {},
                    },
                    "answer_key": {"canonical_answer": {"city": "sasebo"}},
                }
            ]

        def write(self, output_dir, entries):
            output_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = output_dir / "review_pack.jsonl"
            markdown_path = output_dir / "review_pack.md"
            jsonl_path.write_text(json.dumps(entries[0], ensure_ascii=False) + "\n", encoding="utf-8")
            markdown_path.write_text("# Review Pack\n", encoding="utf-8")
            return jsonl_path, markdown_path

    monkeypatch.setattr("rl_task_foundry.cli.ReviewPackBuilder", _DummyReviewPackBuilder)

    result = CliRunner().invoke(
        app,
        [
            "generate-review-pack",
            str(output_dir),
            "--limit",
            "1",
            "--path-id",
            "customer.address.city",
        ],
    )

    assert result.exit_code == 0
    assert "generated review pack" in result.stdout
    assert "entries=1" in result.stdout
    assert (output_dir / "review_pack.jsonl").exists()
    assert (output_dir / "review_pack.md").exists()


def test_cli_generate_review_pack_defaults_to_timestamped_archive_dir(monkeypatch, tmp_path):
    expected_dir = tmp_path / "review_packs" / "20260411-120000"

    class _DummyReviewPackBuilder:
        def __init__(self, _config):
            pass

        async def build_entries(self, *, limit, path_ids=None, task_specs=None):
            assert limit == 1
            assert path_ids is None
            assert task_specs is None
            return [
                {
                    "task_id": "task_review_1",
                    "review_surface": {
                        "question": "질문",
                        "answer_schema": {"fields": []},
                        "tool_bundle": {"tools": []},
                    },
                    "review_notes": {
                        "seed_question": "시드 질문",
                        "question_strategy": "model_generated",
                        "label_tier": "A",
                        "tool_level": 1,
                        "question_family": "status_lookup",
                        "outcome_type": "answer",
                        "selected_path_id": "customer.address.city",
                        "required_hops": 2,
                        "difficulty_features": {},
                        "presentation_generation_metadata": {},
                    },
                    "answer_key": {"canonical_answer": {"city": "sasebo"}},
                }
            ]

        def write(self, output_dir, entries):
            output_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = output_dir / "review_pack.jsonl"
            markdown_path = output_dir / "review_pack.md"
            jsonl_path.write_text(json.dumps(entries[0], ensure_ascii=False) + "\n", encoding="utf-8")
            markdown_path.write_text("# Review Pack\n", encoding="utf-8")
            return jsonl_path, markdown_path

    monkeypatch.setattr("rl_task_foundry.cli.ReviewPackBuilder", _DummyReviewPackBuilder)
    monkeypatch.setattr("rl_task_foundry.cli._default_review_pack_dir", lambda: expected_dir)

    result = CliRunner().invoke(
        app,
        [
            "generate-review-pack",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "20260411-120000" in result.stdout
    assert (expected_dir / "review_pack.jsonl").exists()
    assert (expected_dir / "review_pack.md").exists()


def test_cli_validate_config_applies_runtime_overrides():
    result = CliRunner().invoke(
        app,
        [
            "validate-config",
            "--composer-provider",
            "local_server",
            "--solver-backbone-provider",
            "local_server",
            "--solver-provider",
            "local_server",
            "--solver-model",
            "local-gpt",
        ],
    )
    assert result.exit_code == 0
    assert "composer=local_server/gpt-5.4-mini" in result.stdout
    assert "solver_backbone=local_server/gpt-5.4-mini" in result.stdout
    assert "gpt54m_replica=local_server/local-gptx4" in result.stdout


def test_cli_preview_task_package_uses_selected_tool_level_from_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        Path("rl_task_foundry.yaml")
        .read_text(encoding="utf-8")
        .replace("selected_tool_level: 1", "selected_tool_level: 2"),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "preview-task-package",
            "customer.address.city",
            "현재 고객의 주소 기준 도시 정보를 확인해줘",
            "--config-path",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    assert "level=L2" in result.stdout
    assert "tool_level_config=selected=2" in result.stdout


def test_cli_evaluate_tool_naming_command():
    result = CliRunner().invoke(
        app,
        [
            "evaluate-tool-naming",
            "customer.address.city",
            "--tool-level",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "compiled path=customer.address.city level=L2" in result.stdout
    assert "opacity=" in result.stdout


def test_cli_run_command_loads_task_specs_and_invokes_orchestrator(monkeypatch, tmp_path):
    task_spec_path = tmp_path / "tasks.jsonl"
    task_spec_path.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "anchor_table": "customer",
                "anchor_pk_column": "customer_id",
                "anchor_pk_value": "1",
                "domain": "customer_support",
                "language": "ko",
                "label_tier": "A",
                "question_family": "status_lookup",
                "question": "현재 고객의 주소 기준 도시 정보를 확인해줘",
                "outcome_type": "answer",
                "answer_schema": {
                    "fields": [
                        {
                            "name": "city",
                            "type": "string",
                            "canonicalizer": "lower_trim",
                            "source_columns": ["city.city"],
                        }
                    ]
                },
                "selected_path_id": "customer.address.city",
                "required_hops": 2,
                "tool_level": 1,
                "tool_bundle_id": "customer.address.city.L1",
                "sensitivity_policy": "default",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeOrchestrator:
        def __init__(self, config):
            captured["config"] = config

        async def run_tasks(self, tasks):
            captured["tasks"] = tasks
            return RunSummary(
                run_id="run_test_123",
                total_tasks=1,
                accepted_tasks=1,
                rejected_tasks=0,
                skipped_tasks=0,
                verification_results=2,
                accepted_jsonl_path=Path("accepted.jsonl"),
                rejected_jsonl_path=Path("rejected.jsonl"),
            )

    monkeypatch.setattr("rl_task_foundry.cli.Orchestrator", FakeOrchestrator)

    result = CliRunner().invoke(app, ["run", str(task_spec_path)])

    assert result.exit_code == 0
    assert "run complete" in result.stdout
    assert "run_id=run_test_123" in result.stdout
    assert len(captured["tasks"]) == 1
    assert captured["tasks"][0].task_id == "task_1"


def test_cli_run_summary_reads_run_db(tmp_path):
    config_path = tmp_path / "config.yaml"
    run_db_path = tmp_path / "artifacts" / "run.db"
    config_path.write_text(
        Path("rl_task_foundry.yaml")
        .read_text(encoding="utf-8")
        .replace("./artifacts/run.db", str(run_db_path))
        .replace("./artifacts/accepted.jsonl", str(tmp_path / "artifacts" / "accepted.jsonl"))
        .replace("./artifacts/rejected.jsonl", str(tmp_path / "artifacts" / "rejected.jsonl"))
        .replace("./artifacts/events.jsonl", str(tmp_path / "artifacts" / "events.jsonl"))
        .replace("./artifacts/traces", str(tmp_path / "artifacts" / "traces")),
        encoding="utf-8",
    )

    bootstrap_run_db(run_db_path)
    with connect_run_db(run_db_path) as conn:
        record_run(conn, run_id="run_test_456", config_hash="abc", created_at="2026-04-11T00:00:00+00:00")
        record_task(conn, run_id="run_test_456", task_id="task_1", status="accepted", payload={"task_id": "task_1"})
        record_task(conn, run_id="run_test_456", task_id="task_2", status="rejected", payload={"task_id": "task_2"})
        record_verification_result(
            conn,
            run_id="run_test_456",
            task_id="task_1",
            solver_id="solver_a",
            payload={"pass_exact": True},
        )
        record_verification_result(
            conn,
            run_id="run_test_456",
            task_id="task_2",
            solver_id="solver_a",
            payload={"pass_exact": False},
        )
        record_accepted_example(
            conn,
            run_id="run_test_456",
            task_id="task_1",
            payload={"task_id": "task_1"},
        )
        record_event(conn, run_id="run_test_456", event_type="run_started", payload={"task_count": 2})
        conn.commit()

    result = CliRunner().invoke(
        app,
        ["run-summary", "run_test_456", "--config-path", str(config_path)],
    )

    assert result.exit_code == 0
    assert "run_id=run_test_456" in result.stdout
    assert "tasks=2" in result.stdout
    assert "accepted=1" in result.stdout
    assert "rejected=1" in result.stdout
    assert "skipped=0" in result.stdout
