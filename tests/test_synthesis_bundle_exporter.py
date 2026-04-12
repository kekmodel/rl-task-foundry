from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from rl_task_foundry.synthesis.bundle_exporter import EnvironmentBundleExporter
from rl_task_foundry.synthesis.environment_registry import EnvironmentRegistryWriter
from tests.test_synthesis_environment_registry import _sample_draft


def test_environment_bundle_exporter_writes_label_only_layout(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "registry" / "environments",
        index_db_path=tmp_path / "registry" / "environment_registry.db",
    )
    draft = _sample_draft()
    writer.commit_draft(draft)

    exporter = EnvironmentBundleExporter(
        registry=writer,
        materializer=writer.atomic_tool_materializer,
    )
    summary = exporter.export_bundle(tmp_path / "bundle")

    assert summary.database_count == 1
    assert summary.environment_count == 1
    database_dir = summary.bundle_root / "databases" / "sakila"
    environment_dir = summary.bundle_root / "environments" / draft.environment.env_id

    assert (database_dir / "atomic_tools.py").exists()
    assert (database_dir / "atomic_tool_definitions.json").exists()
    assert (environment_dir / "environment.yaml").exists()
    assert (environment_dir / "instances.jsonl").exists()
    assert (environment_dir / "canonical_answers.jsonl").exists()
    assert not (environment_dir / "audit").exists()
    assert not (environment_dir / "tools.py").exists()

    payload = yaml.safe_load((environment_dir / "environment.yaml").read_text(encoding="utf-8"))
    assert payload["env_id"] == draft.environment.env_id
    assert payload["db_id"] == "sakila"

    exported_answers = (
        environment_dir / "canonical_answers.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(exported_answers[0])["instance_id"] == "instance_0001"

    module_path = database_dir / "atomic_tools.py"
    spec = importlib.util.spec_from_file_location("exported_atomic_tools", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "get_assignments")


def test_environment_bundle_exporter_filters_by_topic(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "registry" / "environments",
        index_db_path=tmp_path / "registry" / "environment_registry.db",
    )
    assignment_draft = _sample_draft(tmp_env_id="env_assignment_a")
    itinerary_draft = _sample_draft(
        tmp_env_id="env_itinerary_b",
        topic="itinerary",
        task_signature="sha256:task_itinerary",
    )
    writer.commit_draft(assignment_draft)
    writer.commit_draft(itinerary_draft)

    exporter = EnvironmentBundleExporter(
        registry=writer,
        materializer=writer.atomic_tool_materializer,
    )
    summary = exporter.export_bundle(
        tmp_path / "bundle",
        topic="assignment",
    )

    assert summary.environment_count == 1
    assert summary.env_ids == ("env_assignment_a",)
    assert not (summary.bundle_root / "environments" / "env_itinerary_b").exists()
