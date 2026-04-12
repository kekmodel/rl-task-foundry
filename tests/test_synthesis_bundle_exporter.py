from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from rl_task_foundry.synthesis.bundle_exporter import EnvironmentBundleExporter
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.environment_registry import EnvironmentRegistryWriter
from tests.test_synthesis_environment_registry import _sample_draft


def test_environment_bundle_exporter_writes_api_server_layout(tmp_path: Path) -> None:
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
    assert summary.db_ids == ("sakila",)
    assert summary.env_ids == (draft.environment.env_id,)

    database_dir = summary.bundle_root / "databases" / "sakila"
    environment_dir = summary.bundle_root / "environments" / draft.environment.env_id
    audit_dir = environment_dir / "audit"

    assert (database_dir / "atomic_tools.py").exists()
    assert (database_dir / "atomic_tool_definitions.json").exists()
    assert (environment_dir / "environment.yaml").exists()
    assert (environment_dir / "instances.jsonl").exists()
    assert (environment_dir / "canonical_answers.jsonl").exists()
    assert (audit_dir / "solution.py").exists()
    assert (audit_dir / "verifier.py").exists()
    assert (audit_dir / "shadow_verifier.py").exists()
    assert not (environment_dir / "tools.py").exists()

    environment_payload = yaml.safe_load((environment_dir / "environment.yaml").read_text(encoding="utf-8"))
    assert environment_payload["env_id"] == draft.environment.env_id
    assert environment_payload["db_id"] == "sakila"

    exported_instances = (environment_dir / "instances.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(exported_instances) == 1
    assert json.loads(exported_instances[0])["instance_id"] == "instance_0001"

    exported_answers = (
        environment_dir / "canonical_answers.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    assert len(exported_answers) == 1
    assert json.loads(exported_answers[0])["instance_id"] == "instance_0001"

    module_path = database_dir / "atomic_tools.py"
    spec = importlib.util.spec_from_file_location("exported_atomic_tools", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "get_assignments")


def test_environment_bundle_exporter_filters_by_category(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "registry" / "environments",
        index_db_path=tmp_path / "registry" / "environment_registry.db",
    )
    assignment_draft = _sample_draft(tmp_env_id="env_assignment_a")
    itinerary_draft = _sample_draft(
        tmp_env_id="env_itinerary_b",
        category=CategoryTaxonomy.ITINERARY,
    )
    writer.commit_draft(assignment_draft)
    writer.commit_draft(itinerary_draft)

    exporter = EnvironmentBundleExporter(
        registry=writer,
        materializer=writer.atomic_tool_materializer,
    )

    summary = exporter.export_bundle(
        tmp_path / "bundle",
        category=CategoryTaxonomy.ASSIGNMENT,
    )

    assert summary.environment_count == 1
    assert summary.env_ids == ("env_assignment_a",)
    assert not (summary.bundle_root / "environments" / "env_itinerary_b").exists()


def test_environment_bundle_exporter_filters_by_env_id(tmp_path: Path) -> None:
    writer = EnvironmentRegistryWriter(
        root_dir=tmp_path / "registry" / "environments",
        index_db_path=tmp_path / "registry" / "environment_registry.db",
        near_dup_enabled=False,
    )
    first = _sample_draft(tmp_env_id="env_assignment_a")
    second = _sample_draft(
        tmp_env_id="env_assignment_b",
        task_signature="sha256:task_b",
    )
    writer.commit_draft(first)
    writer.commit_draft(second)

    exporter = EnvironmentBundleExporter(
        registry=writer,
        materializer=writer.atomic_tool_materializer,
    )

    summary = exporter.export_bundle(
        tmp_path / "bundle",
        env_id="env_assignment_b",
    )

    assert summary.environment_count == 1
    assert summary.env_ids == ("env_assignment_b",)
    assert not (summary.bundle_root / "environments" / "env_assignment_a").exists()
