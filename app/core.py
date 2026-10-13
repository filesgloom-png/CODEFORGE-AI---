"""Small, deterministic foundation for the CODEFORGE AI agent.

The first milestone intentionally does not execute arbitrary code or call an
LLM. It validates incoming tasks and creates a bounded plan that later agent
adapters can execute safely.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskKind(str, Enum):
    CREATE = "create"
    FIX = "fix"
    EXPLAIN = "explain"
    TEST = "test"


@dataclass(frozen=True)
class TaskRequest:
    description: str
    repository: str | None = None
    kind: TaskKind = TaskKind.CREATE


@dataclass(frozen=True)
class TaskPlan:
    summary: str
    steps: tuple[str, ...]
    requires_repository: bool


class TaskValidationError(ValueError):
    """Raised when a task request is invalid."""


def validate_task(request: TaskRequest) -> TaskRequest:
    """Validate user-controlled task data before planning."""
    description = request.description.strip()
    if not description:
        raise TaskValidationError("Task description must not be empty.")
    if len(description) > 4000:
        raise TaskValidationError("Task description must be 4000 characters or fewer.")

    repository = request.repository.strip() if request.repository else None
    if repository == "":
        repository = None
    if repository and ("/" not in repository or repository.startswith("/") or repository.endswith("/")):
        raise TaskValidationError("Repository must use the owner/name format.")

    return TaskRequest(
        description=description,
        repository=repository,
        kind=request.kind,
    )


def plan_task(request: TaskRequest) -> TaskPlan:
    """Return a safe, deterministic execution outline for a validated task."""
    task = validate_task(request)
    steps = [
        "Inspect the project structure and current branch HEAD.",
        "Translate the request into small, verifiable changes.",
    ]

    if task.kind == TaskKind.EXPLAIN:
        steps.append("Explain the relevant code and trade-offs without modifying files.")
    elif task.kind == TaskKind.TEST:
        steps.extend([
            "Identify existing test and validation commands.",
            "Run the relevant checks and summarize their output.",
        ])
    elif task.kind == TaskKind.FIX:
        steps.extend([
            "Reproduce or isolate the reported failure using logs and tests.",
            "Propose the smallest targeted fix on a dedicated branch.",
            "Run the relevant checks and inspect the results.",
        ])
    else:
        steps.extend([
            "Create an implementation plan and identify affected files.",
            "Implement the change on a dedicated branch.",
            "Run available tests and build checks.",
        ])

    steps.append("Report changed files, checks, and any remaining limitations.")
    return TaskPlan(
        summary=task.description,
        steps=tuple(steps),
        requires_repository=task.kind != TaskKind.EXPLAIN or task.repository is not None,
    )
