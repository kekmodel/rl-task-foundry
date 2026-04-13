from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from rl_task_foundry.synthesis.bundle_exporter import TaskBundleExporter
from rl_task_foundry.synthesis.task_registry import TaskRegistryWriter
from tests.test_synthesis_task_registry import _sample_draft


def test_task_bundle_exporter_writes_single_task_layout(tmp_path: Path) -> None:
    writer = TaskRegistryWriter(
        root_dir=tmp_path / "registry" / "tasks",
        index_db_path=tmp_path / "registry" / "task_registry.db",
    )
    draft = _sample_draft()
    writer.commit_draft(draft)

    exporter = TaskBundleExporter(
        registry=writer,
        materializer=writer.atomic_tool_materializer,
    )
    summary = exporter.export_bundle(tmp_path / "bundle")

    assert summary.database_count == 1
    assert summary.task_count == 1
    database_dir = summary.bundle_root / "databases" / "sakila"
    task_dir = summary.bundle_root / "tasks" / draft.task_bundle.task_id

    assert (database_dir / "atomic_tools.py").exists()
    assert (database_dir / "atomic_tool_definitions.json").exists()
    assert (task_dir / "task.yaml").exists()
    assert (task_dir / "task.json").exists()
    assert (task_dir / "instance.json").exists()
    assert (task_dir / "canonical_answer.json").exists()

    payload = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    assert payload["task_id"] == draft.task_bundle.task_id
    assert payload["db_id"] == "sakila"

    exported_answer = json.loads((task_dir / "canonical_answer.json").read_text(encoding="utf-8"))
    assert exported_answer["label_signature"] == draft.label_signature

    module_path = database_dir / "atomic_tools.py"
    spec = importlib.util.spec_from_file_location("exported_atomic_tools", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "get_assignments")


def test_task_bundle_exporter_filters_by_topic(tmp_path: Path) -> None:
    writer = TaskRegistryWriter(
        root_dir=tmp_path / "registry" / "tasks",
        index_db_path=tmp_path / "registry" / "task_registry.db",
    )
    assignment_draft = _sample_draft(tmp_task_id="task_assignment_a")
    itinerary_draft = _sample_draft(
        tmp_task_id="task_itinerary_b",
        topic="itinerary",
        task_signature="sha256:task_itinerary",
    )
    writer.commit_draft(assignment_draft)
    writer.commit_draft(itinerary_draft)

    exporter = TaskBundleExporter(
        registry=writer,
        materializer=writer.atomic_tool_materializer,
    )
    summary = exporter.export_bundle(
        tmp_path / "bundle",
        topic="assignment",
    )

    assert summary.task_count == 1
    assert summary.task_ids == ("task_assignment_a",)
    assert not (summary.bundle_root / "tasks" / "task_itinerary_b").exists()
