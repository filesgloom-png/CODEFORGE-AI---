"""Controlled orchestration for inspect -> validate -> isolated branch.

This module returns structured plans/results only. It never executes shell commands,
project code, or model-supplied instructions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core import TaskKind, TaskRequest, plan_task
from app.diagnostics import diagnose_log, next_repair_action
from app.github_adapter import GitHubReadOnlyAdapter
from app.github_writer import GitHubPatchWriter
from app.memory import ProjectMemory
from app.patches import validate_changes, PatchValidationError


@dataclass(frozen=True)
class WorkflowResult:
    status: str
    repository: str | None
    summary: str
    details: dict[str, Any]


class ControlledAgentWorkflow:
    """Coordinates read-only inspection and validated branch preparation."""

    def __init__(
        self,
        repository: str,
        token: str | None = None,
        reader: GitHubReadOnlyAdapter | None = None,
        writer: GitHubPatchWriter | None = None,
        memory: ProjectMemory | None = None,
    ) -> None:
        self.repository = repository
        self.reader = reader or GitHubReadOnlyAdapter(repository, token=token)
        self.writer = writer or GitHubPatchWriter(repository, token=token)
        self.memory = memory if memory is not None else ProjectMemory()

    def recall_memory(self, query: str = "", limit: int = 20) -> WorkflowResult:
        """Recall explicitly saved project facts and decisions."""
        items = self.memory.recall(self.repository, query=query, limit=limit)
        return WorkflowResult(
            "memory_recalled", self.repository,
            f"Retrieved {len(items)} saved project memory item(s).",
            {"items": items},
        )

    def remember_fact(self, category: str, content: str) -> WorkflowResult:
        """Save an explicit preference, decision or project fact; secrets are rejected."""
        item = self.memory.remember(self.repository, category, content)
        return WorkflowResult(
            "memory_saved", self.repository, "Project memory saved.",
            {"item": item},
        )

    def inspect(self, task: TaskRequest, ref: str = "main") -> WorkflowResult:
        """Build a task plan plus a bounded repository and CI snapshot."""
        if task.repository and task.repository != self.repository:
            raise ValueError("Task repository does not match this workflow.")
        plan = plan_task(TaskRequest(task.description, self.repository, task.kind))
        tree = self.reader.get_tree(ref=ref, recursive=True)
        runs = self.reader.list_workflow_runs(branch=ref, limit=5)
        # Persist only repository metadata, never the raw task prompt.
        self.memory.remember(
            self.repository, "workflow_event",
            json.dumps({"event": "inspect", "ref": ref, "tree_sha": tree.get("sha")}),
        )
        return WorkflowResult(
            status="inspected",
            repository=self.repository,
            summary=plan.summary,
            details={
                "ref": ref,
                "plan": {"steps": list(plan.steps), "requires_repository": plan.requires_repository},
                "tree": tree,
                "recent_runs": runs,
                "memory_items": len(self.memory.recall(self.repository, limit=100)),
            },
        )

    def validate_patch(self, raw_patch: str, ref: str = "main") -> WorkflowResult:
        """Validate a model-proposed patch against a fresh repository tree."""
        tree = self.reader.get_tree(ref=ref, recursive=True)
        if tree.get("truncated"):
            return WorkflowResult(
                "rejected", self.repository,
                "Repository tree was truncated; patch cannot be safely validated.",
                {"ref": ref, "reason": "truncated_tree"},
            )
        existing = {
            item["path"] for item in tree["tree"]
            if item.get("type") == "blob" and isinstance(item.get("path"), str)
        }
        try:
            changes = validate_changes(raw_patch, existing_paths=existing)
        except PatchValidationError as exc:
            return WorkflowResult(
                "rejected", self.repository, "Patch validation failed.",
                {"ref": ref, "reason": str(exc)},
            )
        return WorkflowResult(
            "validated", self.repository, "Patch passed structural and path validation.",
            {"ref": ref, "changes": [
                {"path": change.path, "action": change.action,
                 "bytes": len(change.content.encode("utf-8")) if change.content is not None else 0}
                for change in changes
            ]},
        )

    def prepare_branch(
        self,
        raw_patch: str,
        base_branch: str = "main",
        description: str = "validated CODEFORGE AI patch",
    ) -> WorkflowResult:
        """Validate first, then publish only to a dedicated branch."""
        checked = self.validate_patch(raw_patch, ref=base_branch)
        if checked.status != "validated":
            return checked
        published = self.writer.publish_patch(
            raw_patch, base_branch=base_branch, description=description
        )
        # Save only non-secret branch metadata. No merge or CI result is inferred.
        self.memory.remember(
            self.repository, "workflow_event",
            json.dumps({
                "event": "branch_prepared",
                "base_branch": published.get("base_branch"),
                "branch": published.get("branch"),
                "commit_sha": published.get("commit_sha"),
                "changed_files": published.get("changed_files", []),
                "checks_run": bool(published.get("checks_run", False)),
            }),
        )
        return WorkflowResult(
            "branch_prepared", self.repository,
            "Validated patch committed to an isolated branch; no merge or build was run.",
            published,
        )

    def diagnose_job(self, job_id: int, attempts_used: int = 0) -> WorkflowResult:
        """Fetch and safely diagnose one Actions job log without executing its content."""
        raw_log = self.reader.fetch_job_logs(job_id)
        diagnosis = diagnose_log(raw_log)
        decision = next_repair_action(diagnosis, attempts_used)
        return WorkflowResult(
            status="diagnosed",
            repository=self.repository,
            summary="CI log analyzed as untrusted text; no commands were executed.",
            details={
                "job_id": job_id,
                "diagnosis": diagnosis.status,
                "findings": list(diagnosis.findings),
                "excerpts": list(diagnosis.excerpts),
                "truncated": diagnosis.truncated,
                "attempts_used": attempts_used,
                "next_action": decision,
            },
        )

    @staticmethod
    def decode_model_patch(model_text: str) -> str:
        """Accept only a strict JSON envelope; never interpret natural language as commands."""
        try:
            value = json.loads(model_text)
        except (TypeError, json.JSONDecodeError):
            raise ValueError("Model output must be a JSON patch envelope.") from None
        if not isinstance(value, dict) or set(value) != {"changes"}:
            raise ValueError("Model output must contain only the changes array.")
        return json.dumps(value, ensure_ascii=False)
