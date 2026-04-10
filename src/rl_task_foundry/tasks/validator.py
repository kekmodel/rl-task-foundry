"""Task validation rules."""

from __future__ import annotations

from dataclasses import dataclass

from rl_task_foundry.tasks.models import TaskSpec


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str


class TaskValidator:
    """Apply fast deterministic task-quality checks before solving."""

    def validate(self, task: TaskSpec) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not task.question.strip():
            issues.append(ValidationIssue(code="empty_question", message="question must not be empty"))
        if not task.answer_schema.fields:
            issues.append(
                ValidationIssue(code="missing_answer_schema", message="answer_schema must define fields")
            )
        if not task.selected_path_id:
            issues.append(ValidationIssue(code="missing_path", message="selected_path_id is required"))
        return issues
