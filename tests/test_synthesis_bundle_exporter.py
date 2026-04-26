from __future__ import annotations

import json
from pathlib import Path

import yaml

from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.synthesis.bundle_exporter import TaskBundleExporter
from rl_task_foundry.synthesis.snapshot_materializer import (
    TOOLING_VERSION,
    SchemaSnapshotMaterializer,
)
from rl_task_foundry.synthesis.task_registry import TaskRegistryWriter
from rl_task_foundry.tooling.common import snapshot_from_graph
from tests.test_synthesis_task_registry import _sample_draft


def _toy_snapshot():
    customer = TableProfile(
        schema_name="public",
        table_name="customer",
        columns=[
            ColumnProfile(
                schema_name="public",
                table_name="customer",
                column_name="customer_id",
                data_type="integer",
                ordinal_position=1,
                is_nullable=False,
                visibility="user_visible",
                is_primary_key=True,
            )
        ],
        primary_key=("customer_id",),
    )
    rental = TableProfile(
        schema_name="public",
        table_name="rental",
        columns=[
            ColumnProfile(
                schema_name="public",
                table_name="rental",
                column_name="rental_id",
                data_type="integer",
                ordinal_position=1,
                is_nullable=False,
                visibility="user_visible",
                is_primary_key=True,
            ),
            ColumnProfile(
                schema_name="public",
                table_name="rental",
                column_name="customer_id",
                data_type="integer",
                ordinal_position=2,
                is_nullable=False,
                visibility="user_visible",
                is_foreign_key=True,
            ),
        ],
        primary_key=("rental_id",),
    )
    return snapshot_from_graph(
        SchemaGraph(
            tables=[customer, rental],
            edges=[
                ForeignKeyEdge(
                    constraint_name="rental_customer",
                    source_schema="public",
                    source_table="rental",
                    source_columns=("customer_id",),
                    target_schema="public",
                    target_table="customer",
                    target_columns=("customer_id",),
                )
            ],
        )
    )


def _build_exporter(tmp_path: Path) -> tuple[TaskBundleExporter, SchemaSnapshotMaterializer]:
    writer = TaskRegistryWriter(
        root_dir=tmp_path / "registry" / "tasks",
        index_db_path=tmp_path / "registry" / "task_registry.db",
    )
    materializer = SchemaSnapshotMaterializer(
        root_dir=tmp_path / "registry" / "databases"
    )
    exporter = TaskBundleExporter(
        registry=writer, snapshot_materializer=materializer
    )
    return exporter, materializer


def test_task_bundle_exporter_writes_single_task_layout(tmp_path: Path) -> None:
    exporter, materializer = _build_exporter(tmp_path)
    draft = _sample_draft()
    exporter.registry.commit_draft(draft)
    materializer.materialize(db_id=draft.db_id, snapshot=_toy_snapshot())

    summary = exporter.export_bundle(tmp_path / "bundle")

    assert summary.database_count == 1
    assert summary.task_count == 1
    database_dir = summary.bundle_root / "databases" / "sakila"
    task_dir = summary.bundle_root / "tasks" / draft.task_bundle.task_id

    assert (database_dir / "schema_snapshot.json").exists()
    assert (database_dir / "tooling_version.json").exists()
    assert (task_dir / "task.yaml").exists()
    assert (task_dir / "task.json").exists()
    assert (task_dir / "instance.json").exists()
    assert (task_dir / "canonical_answer.json").exists()

    snapshot_payload = json.loads(
        (database_dir / "schema_snapshot.json").read_text(encoding="utf-8")
    )
    assert [t["name"] for t in snapshot_payload["tables"]] == ["customer", "rental"]
    assert snapshot_payload["edges"][0]["target_table"] == "customer"

    version_payload = json.loads(
        (database_dir / "tooling_version.json").read_text(encoding="utf-8")
    )
    assert version_payload["tooling_version"] == TOOLING_VERSION
    assert "schema_map" in version_payload["composer_tools"]
    assert "create_record_set" in version_payload["atomic_primitives"]
    assert "group_top" not in version_payload["atomic_primitives"]

    payload = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    assert payload["task_id"] == draft.task_bundle.task_id
    assert payload["db_id"] == "sakila"

    exported_answer = json.loads((task_dir / "canonical_answer.json").read_text(encoding="utf-8"))
    assert exported_answer["label_signature"] == draft.label_signature


def test_task_bundle_exporter_filters_by_topic(tmp_path: Path) -> None:
    exporter, materializer = _build_exporter(tmp_path)
    assignment_draft = _sample_draft(tmp_task_id="task_assignment_a")
    itinerary_draft = _sample_draft(
        tmp_task_id="task_itinerary_b",
        topic="itinerary",
        task_signature="sha256:task_itinerary",
    )
    exporter.registry.commit_draft(assignment_draft)
    exporter.registry.commit_draft(itinerary_draft)
    materializer.materialize(db_id=assignment_draft.db_id, snapshot=_toy_snapshot())

    summary = exporter.export_bundle(
        tmp_path / "bundle",
        topic="assignment",
    )

    assert summary.task_count == 1
    assert summary.task_ids == ("task_assignment_a",)
    assert not (summary.bundle_root / "tasks" / "task_itinerary_b").exists()
