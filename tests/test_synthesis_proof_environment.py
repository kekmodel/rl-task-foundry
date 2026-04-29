from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig
from rl_task_foundry.synthesis.proof_environment import (
    PROOF_ANCHOR_ENTITY,
    PROOF_CANONICAL_ANSWER,
    PROOF_DB_ID,
    PROOF_TASK_TOPIC,
    build_proof_composer_script,
    build_proof_question,
    build_proof_schema_graph,
    run_proof_task,
)
from rl_task_foundry.synthesis.real_db_trial import RealDbTrialStatus
from rl_task_foundry.synthesis.task_registry import TaskRegistryCommitStatus


def _proof_config(tmp_path: Path):
    config = load_config("rl_task_foundry.yaml")
    output = OutputConfig(
        run_db_path=tmp_path / "run.db",
        traces_dir=tmp_path / "traces",
    )
    return config.model_copy(update={"output": output}, deep=True)


def test_build_proof_schema_graph_uses_given_schema_name() -> None:
    graph = build_proof_schema_graph("proof_trial_demo")

    assert {table.table_name for table in graph.tables} == {
        "proof_anchors",
        "proof_cities",
        "proof_city_links",
        "proof_lodgings",
        "proof_activities",
    }
    for table in graph.tables:
        assert table.schema_name == "proof_trial_demo"
    for edge in graph.edges:
        assert edge.source_schema == "proof_trial_demo"
        assert edge.target_schema == "proof_trial_demo"


def test_build_proof_question_returns_user_request_body() -> None:
    question = build_proof_question()

    assert not question.startswith("<entity>\n")
    assert "봄 시즌 3일 출장" in question


def test_build_proof_composer_script_payload_matches_canonical_answer() -> None:
    script = build_proof_composer_script()

    assert script.submit_payload.topic == PROOF_TASK_TOPIC
    assert script.submit_payload.parsed_entity == PROOF_ANCHOR_ENTITY
    assert script.submit_payload.canonical_answer == PROOF_CANONICAL_ANSWER
    assert not script.submit_payload.user_request.startswith("<entity>\n")
    assert "봄 시즌 3일 출장" in script.submit_payload.user_request
    tool_names = [call.tool_name for call in script.atomic_tool_calls]
    assert "sample" in tool_names
    # Every answer string must appear in at least one tool-call result so the
    # controller's grounding check can see it.
    canonical_strings = {
        "Seoul", "Suwon", "Incheon",
        "Seoul Station Stay", "Suwon Fortress Hotel", "Incheon Harbor Inn",
        "Han River Night Walk", "Fortress Loop Tour", "Harbor Sunset Ferry",
    }
    observed: set[str] = set()
    for call in script.atomic_tool_calls:
        observed.update(_collect_strings(call.result))
    missing = canonical_strings - observed
    assert missing == set(), f"tool-call results missing grounded strings: {missing}"


def _collect_strings(value: object) -> set[str]:
    collected: set[str] = set()
    if isinstance(value, str):
        collected.add(value)
        return collected
    if isinstance(value, dict):
        for item in value.values():
            collected.update(_collect_strings(item))
        return collected
    if isinstance(value, list):
        for item in value:
            collected.update(_collect_strings(item))
    return collected


def test_proof_environment_module_has_no_legacy_imports() -> None:
    module_path = Path("src/rl_task_foundry/synthesis/proof_environment.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # Proof path must not short-circuit past the real_db_trial
            # runner or import any of the hand-coded-draft helpers that
            # were removed in the Follow-up 1 rewrite.
            assert node.module != "rl_task_foundry.synthesis.quality_gate"
            assert node.module != "rl_task_foundry.synthesis.rendered_prompt_builder"
            assert node.module != "rl_task_foundry.synthesis.canonicalize"


def test_proof_topic_is_itinerary_taxonomy() -> None:
    assert PROOF_TASK_TOPIC == "itinerary"


@pytest.mark.asyncio
async def test_run_proof_task_commits_and_exports_bundle(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Integration test: requires the Postgres instance described by
    # ``rl_task_foundry.yaml`` to be reachable. asyncpg raises when it
    # isn't, which surfaces as a test failure — consistent with the
    # other PG-backed integration suites.
    config = _proof_config(tmp_path)
    summary = await run_proof_task(
        config,
        output_root=tmp_path / "proof_output",
    )

    assert summary.db_id == PROOF_DB_ID
    assert summary.trial_status is RealDbTrialStatus.ACCEPTED
    assert summary.quality_gate_status == "accept"
    assert summary.task_id is not None
    assert summary.flow_id is not None
    assert summary.registry_status is TaskRegistryCommitStatus.COMMITTED
    assert summary.bundle_root is not None
    assert summary.phase_monitor_log_path is not None
    assert (summary.bundle_root / "databases" / PROOF_DB_ID / "schema_snapshot.json").exists()
    assert (summary.bundle_root / "databases" / PROOF_DB_ID / "tooling_version.json").exists()
    assert summary.task_id is not None
    assert (summary.bundle_root / "tasks" / summary.task_id / "task.yaml").exists()
    assert "anchor candidate seeding failed" not in caplog.text

    phase_monitor_lines = [
        json.loads(line)
        for line in summary.phase_monitor_log_path.read_text(encoding="utf-8").splitlines()
    ]
    phases = [line["phase"] for line in phase_monitor_lines]
    assert "trial" in phases
    assert "registry_commit" in phases
    assert "bundle_export" in phases
