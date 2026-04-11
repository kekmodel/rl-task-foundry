import asyncio
import json
from pathlib import Path

from rl_task_foundry.config import load_config
from rl_task_foundry.pipeline.orchestrator import ReviewArtifact
from rl_task_foundry.pipeline.review_pack import ReviewPackBuilder
from rl_task_foundry.schema.path_catalog import PathCatalog, PathSpec
from rl_task_foundry.tasks.models import PresentedToolBundle, PresentedToolSpec, TaskPackage, TaskSpec
from rl_task_foundry.tools.models import ToolBundle, ToolParameter, ToolSpec
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema, GroundTruth


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="task_review_1",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question_family="status_lookup",
        question="연결된 도시 값을 확인해줘.",
        outcome_type="answer",
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="city",
                    type="string",
                    canonicalizer="lower_trim",
                    source_columns=["addresses.city"],
                )
            ]
        ),
        selected_path_id="orders.addresses",
        required_hops=1,
        tool_level=2,
        tool_bundle_id="orders.addresses::canonical::L1",
        sensitivity_policy="default",
    )


def test_review_pack_builder_writes_jsonl_and_markdown(monkeypatch, tmp_path):
    config = load_config(Path("rl_task_foundry.yaml"))
    builder = ReviewPackBuilder(config)
    task = _task()

    path = PathSpec(
        path_id="orders.addresses",
        root_table="orders",
        tables=["orders", "addresses"],
        edges=[],
        hop_count=1,
        difficulty_features={"required_hops": 1, "fanout_product": 1.0, "shortcut_count": 0},
    )
    catalog = PathCatalog(paths=[path])

    canonical_bundle = ToolBundle(
        bundle_id="orders.addresses::canonical::L1",
        path_id="orders.addresses",
        tool_level=1,
        tools=[
            ToolSpec(
                name="get_city_for_order",
                description="Get city for the selected order.",
                semantic_key="orders.addresses:lookup:city",
                kind="lookup",
                parameters=[
                    ToolParameter(
                        name="anchor_order_id",
                        json_type="integer",
                        description="Selected order id.",
                        required=True,
                    )
                ],
                output_fields=["city"],
                sql_template="SELECT city",
                path_id="orders.addresses",
                tool_level=1,
                name_source="rule_based",
            )
        ],
    )

    package = TaskPackage(
        task=task.model_copy(
            update={
                "question": "지금 제 주문이 연결된 지역 정보가 어디인지 알려주세요.",
                "question_source": "model_generated",
                "question_generation_metadata": {"status": "accepted", "attempts": 1},
                "presented_tool_bundle_id": "orders.addresses::task::task_review_1",
            }
        ),
        presented_tool_bundle=PresentedToolBundle(
            bundle_id="orders.addresses::task::task_review_1",
            canonical_bundle_id=canonical_bundle.bundle_id,
            path_id="orders.addresses",
            tool_level=2,
            question_family="status_lookup",
            outcome_type="answer",
            generation_metadata={"presentation_strategy": "task_context_model_generated"},
            tools=[
                PresentedToolSpec(
                    name="inspect_region_info",
                    description="Check the region information tied to the current order.",
                    semantic_key="orders.addresses:lookup:city",
                    kind="lookup",
                    parameter_names=["anchor_order_id"],
                    output_fields=["city"],
                    name_source="model_generated",
                )
            ],
        ),
        presentation_options=[],
    )

    truth = GroundTruth(
        task_id=task.task_id,
        verification_sql="SELECT city",
        canonical_answer={"city": "sasebo"},
        answer_schema_version="v1",
        row_context=[{"order_reference": "A-100", "shipping_window": "tomorrow", "city": "sasebo"}],
    )

    class _FakeOrchestrator:
        async def load_graph_and_catalog(self):
            return object(), catalog

        async def build_review_artifact(self, raw_task, *, graph=None, catalog=None):
            assert graph is not None
            assert catalog is not None
            assert raw_task.task_id == task.task_id
            return ReviewArtifact(
                task=task,
                path=path,
                package=package,
                canonical_bundle=canonical_bundle,
                ground_truth=truth,
                question_context={"forbidden_markers": [], "row_context": [{"order_reference": "A-100"}]},
            )

        async def aclose(self):
            return None

    monkeypatch.setattr("rl_task_foundry.pipeline.review_pack.Orchestrator", lambda _config: _FakeOrchestrator())

    entries = asyncio.run(builder.build_entries(limit=1, task_specs=[task]))

    assert len(entries) == 1
    entry = entries[0]
    assert entry["review_surface"]["question"] == "지금 제 주문이 연결된 지역 정보가 어디인지 알려주세요."
    assert entry["review_notes"]["seed_question"] == "연결된 도시 값을 확인해줘."
    assert entry["review_notes"]["question_strategy"] == "model_generated"
    assert entry["answer_key"]["canonical_answer"] == {"city": "sasebo"}

    jsonl_path, markdown_path = builder.write(tmp_path, entries)
    json_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert json_rows[0]["task_id"] == "task_review_1"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Review Pack" in markdown
    assert "inspect_region_info" in markdown
    assert "지금 제 주문이 연결된 지역 정보가 어디인지 알려주세요." in markdown
    assert "<summary>Answer Key</summary>" in markdown


def test_review_pack_builder_skips_seed_fallback_entries(monkeypatch, tmp_path):
    config = load_config(Path("rl_task_foundry.yaml"))
    builder = ReviewPackBuilder(config)
    good_task = _task()
    fallback_task = _task().model_copy(update={"task_id": "task_review_2"})

    path = PathSpec(
        path_id="orders.addresses",
        root_table="orders",
        tables=["orders", "addresses"],
        edges=[],
        hop_count=1,
        difficulty_features={"required_hops": 1, "fanout_product": 1.0, "shortcut_count": 0},
    )
    catalog = PathCatalog(paths=[path])

    canonical_bundle = ToolBundle(
        bundle_id="orders.addresses::canonical::L1",
        path_id="orders.addresses",
        tool_level=1,
        tools=[],
    )

    def _artifact(task: TaskSpec, *, question_source: str) -> ReviewArtifact:
        package = TaskPackage(
            task=task.model_copy(
                update={
                    "question": f"{task.task_id} question",
                    "question_source": question_source,
                    "question_generation_metadata": {"status": "accepted"},
                    "presented_tool_bundle_id": f"{task.task_id}::bundle",
                }
            ),
            presented_tool_bundle=PresentedToolBundle(
                bundle_id=f"{task.task_id}::bundle",
                canonical_bundle_id=canonical_bundle.bundle_id,
                path_id="orders.addresses",
                tool_level=2,
                question_family="status_lookup",
                outcome_type="answer",
                generation_metadata={"presentation_strategy": "task_context_model_generated"},
                tools=[],
            ),
            presentation_options=[],
        )
        truth = GroundTruth(
            task_id=task.task_id,
            verification_sql="SELECT 1",
            canonical_answer={"city": "sasebo"},
            answer_schema_version="v1",
        )
        return ReviewArtifact(
            task=task,
            path=path,
            package=package,
            canonical_bundle=canonical_bundle,
            ground_truth=truth,
            question_context={},
        )

    class _FakeOrchestrator:
        async def load_graph_and_catalog(self):
            return object(), catalog

        async def build_review_artifact(self, raw_task, *, graph=None, catalog=None):
            del graph, catalog
            if raw_task.task_id == "task_review_2":
                return _artifact(raw_task, question_source="seed_fallback")
            return _artifact(raw_task, question_source="model_generated")

        async def aclose(self):
            return None

    monkeypatch.setattr("rl_task_foundry.pipeline.review_pack.Orchestrator", lambda _config: _FakeOrchestrator())

    entries = asyncio.run(builder.build_entries(limit=2, task_specs=[good_task, fallback_task]))

    assert len(entries) == 1
    assert entries[0]["task_id"] == "task_review_1"
