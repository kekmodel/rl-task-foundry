"""Human-review pack generation for qualitative dataset inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.pipeline.orchestrator import Orchestrator, ReviewArtifact
from rl_task_foundry.tasks.factory import TierATaskFactory
from rl_task_foundry.tasks.models import TaskSpec


@dataclass(slots=True)
class ReviewPackBuilder:
    config: AppConfig

    async def build_entries(
        self,
        *,
        limit: int,
        path_ids: list[str] | None = None,
        task_specs: list[TaskSpec] | None = None,
    ) -> list[dict[str, Any]]:
        orchestrator = Orchestrator(self.config)
        try:
            graph, catalog = await orchestrator.load_graph_and_catalog()
            if task_specs is None:
                factory = TierATaskFactory(
                    database=self.config.database,
                    domain=self.config.domain,
                    task_config=self.config.task_composer,
                    tool_compiler=self.config.tool_compiler,
                    verification=self.config.verification,
                )
                task_specs = await factory.generate(
                    graph,
                    catalog,
                    limit=max(limit * 3, limit),
                    path_ids=path_ids,
                )
            else:
                task_specs = list(task_specs)

            entries: list[dict[str, Any]] = []
            for raw_task in task_specs:
                if len(entries) >= limit:
                    break
                artifact = await orchestrator.build_review_artifact(
                    raw_task,
                    graph=graph,
                    catalog=catalog,
                )
                if artifact.package.task.question_source == "seed_fallback":
                    continue
                entries.append(
                    self._entry(artifact)
                )
            return entries
        finally:
            await orchestrator.aclose()

    def write(
        self,
        output_dir: Path,
        entries: list[dict[str, Any]],
    ) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "review_pack.jsonl"
        markdown_path = output_dir / "review_pack.md"

        with jsonl_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str))
                handle.write("\n")

        markdown_path.write_text(self._render_markdown(entries), encoding="utf-8")
        return jsonl_path, markdown_path

    def _entry(self, artifact: ReviewArtifact) -> dict[str, Any]:
        seed_task = artifact.task
        package = artifact.package
        canonical_bundle = artifact.canonical_bundle
        ground_truth = artifact.ground_truth
        question_context = artifact.question_context
        final_task = package.task
        question_strategy = final_task.question_source
        return {
            "task_id": final_task.task_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "review_surface": {
                "question": final_task.question,
                "answer_schema": final_task.answer_schema.model_dump(mode="json"),
                "tool_bundle": package.presented_tool_bundle.model_dump(mode="json"),
            },
            "review_notes": {
                "seed_question": seed_task.question,
                "question_strategy": question_strategy,
                "question_generation_metadata": final_task.question_generation_metadata,
                "label_tier": final_task.label_tier,
                "tool_level": final_task.tool_level,
                "question_family": final_task.question_family,
                "outcome_type": final_task.outcome_type,
                "selected_path_id": final_task.selected_path_id,
                "required_hops": final_task.required_hops,
                "difficulty_features": final_task.difficulty_features,
                "anchor": {
                    "table": final_task.anchor_table,
                    "pk_column": final_task.anchor_pk_column,
                    "pk_value": final_task.anchor_pk_value,
                },
                "question_context": question_context,
                "presented_tool_bundle_id": final_task.presented_tool_bundle_id,
                "canonical_tool_bundle_id": canonical_bundle.bundle_id,
                "presentation_generation_metadata": package.presented_tool_bundle.generation_metadata,
            },
            "answer_key": ground_truth.model_dump(mode="json"),
        }

    def _render_markdown(self, entries: list[dict[str, Any]]) -> str:
        lines = [
            "# Review Pack",
            "",
            "이 파일은 사람 눈으로 task 품질을 검토하기 위한 snapshot이다.",
            "",
        ]
        for index, entry in enumerate(entries, start=1):
            surface = entry["review_surface"]
            notes = entry["review_notes"]
            answer_key = entry["answer_key"]
            tool_bundle = surface["tool_bundle"]
            lines.extend(
                [
                    f"## {index}. {entry['task_id']}",
                    "",
                    f"- family: `{notes['question_family']}`",
                    f"- outcome: `{notes['outcome_type']}`",
                    f"- tier/tool level: `{notes['label_tier']}` / `L{notes['tool_level']}`",
                    f"- path: `{notes['selected_path_id']}`",
                    f"- hops: `{notes['required_hops']}`",
                    f"- question strategy: `{notes['question_strategy']}`",
                    "",
                    "### Question",
                    "",
                    surface["question"],
                    "",
                    "### Submit Result Format",
                    "",
                    "```json",
                    json.dumps(surface["answer_schema"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                    "### Tool Set",
                    "",
                ]
            )
            for tool in tool_bundle["tools"]:
                lines.append(
                    f"- `{tool['name']}` ({tool['kind']}, {tool['presentation_role']}): "
                    f"{tool['description']}"
                )
            lines.extend(
                [
                    "",
                    "### Review Notes",
                    "",
                    f"- seed question: {notes['seed_question']}",
                    f"- presentation strategy: "
                    f"{notes['presentation_generation_metadata'].get('presentation_strategy', 'unknown')}",
                    f"- difficulty features: `{json.dumps(notes['difficulty_features'], ensure_ascii=False, sort_keys=True)}`",
                    "",
                    "<details>",
                    "<summary>Answer Key</summary>",
                    "",
                    "```json",
                    json.dumps(answer_key, ensure_ascii=False, indent=2),
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )
        return "\n".join(lines)
