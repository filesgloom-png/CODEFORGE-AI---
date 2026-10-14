"""Durable, project-scoped memory for CODEFORGE AI.

Only explicit, bounded text facts are stored. Secrets and credential-like values
are rejected. SQLite persistence survives process restarts when the configured
database path is on persistent storage.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ALLOWED_CATEGORIES = {"preference", "decision", "project_fact", "workflow_event"}
_SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b", re.IGNORECASE),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", re.IGNORECASE),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
_MAX_CONTENT = 4000


class MemoryValidationError(ValueError):
    """Raised when memory input is invalid or appears to contain a secret."""


class ProjectMemory:
    """Small SQLite memory store partitioned by repository/project."""

    def __init__(self, database_path: str | os.PathLike[str] | None = None) -> None:
        if database_path is None:
            database_path = os.getenv(
                "CODEFORGE_MEMORY_DB",
                str(Path.home() / ".codeforge" / "memory.sqlite3"),
            )
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            resolved = Path(self.database_path).expanduser()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.database_path = str(resolved)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_project_id ON memories(project, id DESC)"
            )
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS repair_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project TEXT NOT NULL,
                    pr_number INTEGER NOT NULL,
                    source_run_id INTEGER NOT NULL,
                    source_sha TEXT NOT NULL,
                    attempts_used INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    current_branch TEXT,
                    current_sha TEXT,
                    current_run_id INTEGER,
                    last_conclusion TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project, pr_number, source_run_id)
                )"""
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_repair_cycles_project_state ON repair_cycles(project, state, id DESC)"
            )

    @staticmethod
    def _validate_project(project: str) -> str:
        if not isinstance(project, str) or not project.strip() or len(project) > 200:
            raise MemoryValidationError("project must be a non-empty identifier up to 200 characters.")
        return project.strip()

    @staticmethod
    def _validate_content(content: str) -> str:
        if not isinstance(content, str) or not content.strip():
            raise MemoryValidationError("memory content must be non-empty text.")
        content = content.strip()
        if len(content) > _MAX_CONTENT:
            raise MemoryValidationError(f"memory content exceeds {_MAX_CONTENT} characters.")
        if any(pattern.search(content) for pattern in _SECRET_PATTERNS):
            raise MemoryValidationError("memory rejected content that resembles a credential or secret.")
        return content

    def remember(self, project: str, category: str, content: str) -> dict[str, Any]:
        """Persist one explicit fact/event; raw credentials are not accepted."""
        project = self._validate_project(project)
        if category not in _ALLOWED_CATEGORIES:
            raise MemoryValidationError("unsupported memory category.")
        content = self._validate_content(content)
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "INSERT INTO memories(project, category, content, created_at) VALUES (?, ?, ?, ?)",
                (project, category, content, created_at),
            )
            memory_id = cursor.lastrowid
        return {"id": memory_id, "project": project, "category": category,
                "content": content, "created_at": created_at}

    def recall(self, project: str, query: str = "", limit: int = 20) -> list[dict[str, Any]]:
        """Return newest project memories; optional case-insensitive substring filter."""
        project = self._validate_project(project)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("query must be text up to 200 characters.")
        sql = "SELECT id, project, category, content, created_at FROM memories WHERE project = ?"
        args: list[Any] = [project]
        if query.strip():
            sql += " AND lower(content) LIKE lower(?)"
            args.append("%" + query.strip() + "%")
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._connection.execute(sql, args).fetchall()
        return [dict(row) for row in rows]

    def forget(self, project: str, memory_id: int) -> bool:
        """Delete a single memory item only from the selected project."""
        project = self._validate_project(project)
        if not isinstance(memory_id, int) or isinstance(memory_id, bool) or memory_id <= 0:
            raise ValueError("memory_id must be a positive integer.")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM memories WHERE project = ? AND id = ?", (project, memory_id)
            )
        return cursor.rowcount == 1

    def create_repair_cycle(
        self, project: str, pr_number: int, source_run_id: int, source_sha: str
    ) -> dict[str, Any]:
        """Create an idempotent repair cycle keyed by project, PR and originating run."""
        project = self._validate_project(project)
        for name, value in (("pr_number", pr_number), ("source_run_id", source_run_id)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", source_sha):
            raise ValueError("source_sha must be a full Git commit SHA.")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT OR IGNORE INTO repair_cycles
                (project, pr_number, source_run_id, source_sha, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                (project, pr_number, source_run_id, source_sha.lower(), now, now),
            )
            row = self._connection.execute(
                """SELECT * FROM repair_cycles
                WHERE project = ? AND pr_number = ? AND source_run_id = ?""",
                (project, pr_number, source_run_id),
            ).fetchone()
        return dict(row)

    def find_repair_cycle(
        self, project: str, pr_number: int, source_run_id: int
    ) -> dict[str, Any] | None:
        """Find an existing cycle by its idempotency key."""
        project = self._validate_project(project)
        for name, value in (("pr_number", pr_number), ("source_run_id", source_run_id)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM repair_cycles
                WHERE project = ? AND pr_number = ? AND source_run_id = ?""",
                (project, pr_number, source_run_id),
            ).fetchone()
        return dict(row) if row else None

    def get_repair_cycle(self, project: str, cycle_id: int) -> dict[str, Any] | None:
        project = self._validate_project(project)
        if not isinstance(cycle_id, int) or isinstance(cycle_id, bool) or cycle_id <= 0:
            raise ValueError("cycle_id must be a positive integer.")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM repair_cycles WHERE project = ? AND id = ?",
                (project, cycle_id),
            ).fetchone()
        return dict(row) if row else None

    def record_repair_attempt(
        self, project: str, cycle_id: int, branch: str, commit_sha: str, run_id: int | None = None
    ) -> dict[str, Any]:
        """Atomically record one unique published repair, capped at two attempts."""
        project = self._validate_project(project)
        if not isinstance(branch, str) or not re.fullmatch(r"codeforge/[A-Za-z0-9._/-]{1,140}", branch):
            raise ValueError("repair branch must use the codeforge/ namespace.")
        if ".." in branch.split("/") or branch.startswith("/"):
            raise ValueError("repair branch contains an unsafe segment.")
        if not isinstance(commit_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit_sha):
            raise ValueError("commit_sha must be a full Git commit SHA.")
        if run_id is not None and (not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0):
            raise ValueError("run_id must be a positive integer when provided.")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM repair_cycles WHERE project = ? AND id = ?",
                (project, cycle_id),
            ).fetchone()
            if row is None:
                raise ValueError("repair cycle not found in this project.")
            current = dict(row)
            if current["state"] in {"completed", "stop_and_report"}:
                raise ValueError("repair cycle is terminal.")
            if current["attempts_used"] >= 2:
                self._connection.execute(
                    "UPDATE repair_cycles SET state='stop_and_report', updated_at=? WHERE project=? AND id=?",
                    (now, project, cycle_id),
                )
                raise ValueError("repair cycle has reached the two-attempt limit.")
            if current["current_sha"] == commit_sha.lower():
                return current
            if current["current_branch"] == branch:
                raise ValueError("a repair branch cannot be reused for a new attempt.")
            self._connection.execute(
                """UPDATE repair_cycles SET attempts_used=attempts_used+1, state='awaiting_ci',
                current_branch=?, current_sha=?, current_run_id=?, last_conclusion=NULL, updated_at=?
                WHERE project=? AND id=? AND attempts_used < 2
                AND state NOT IN ('completed', 'stop_and_report')""",
                (branch, commit_sha.lower(), run_id, now, project, cycle_id),
            )
            updated = self._connection.execute(
                "SELECT * FROM repair_cycles WHERE project = ? AND id = ?",
                (project, cycle_id),
            ).fetchone()
        return dict(updated)

    def attach_repair_run(
        self, project: str, cycle_id: int, commit_sha: str, run_id: int
    ) -> dict[str, Any]:
        """Attach the first verified Actions run for the current repair SHA, idempotently."""
        project = self._validate_project(project)
        if not isinstance(commit_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit_sha):
            raise ValueError("commit_sha must be a full Git commit SHA.")
        if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
            raise ValueError("run_id must be a positive integer.")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM repair_cycles WHERE project = ? AND id = ?",
                (project, cycle_id),
            ).fetchone()
            if row is None:
                raise ValueError("repair cycle not found in this project.")
            current = dict(row)
            if current["current_sha"] != commit_sha.lower():
                raise ValueError("run SHA does not match the current repair SHA.")
            if current["state"] in {"completed", "stop_and_report"}:
                raise ValueError("repair cycle is terminal.")
            if current["current_run_id"] not in (None, run_id):
                raise ValueError("a different Actions run is already attached to this repair SHA.")
            self._connection.execute(
                "UPDATE repair_cycles SET current_run_id=?, updated_at=? WHERE project=? AND id=?",
                (run_id, now, project, cycle_id),
            )
            updated = self._connection.execute(
                "SELECT * FROM repair_cycles WHERE project = ? AND id = ?",
                (project, cycle_id),
            ).fetchone()
        return dict(updated)

    def update_repair_cycle_status(
        self, project: str, cycle_id: int, state: str, conclusion: str | None = None
    ) -> dict[str, Any]:
        """Advance a repair cycle without allowing terminal states to reopen."""
        project = self._validate_project(project)
        if state not in {"pending", "awaiting_ci", "completed", "stop_and_report"}:
            raise ValueError("unsupported repair cycle state.")
        if conclusion is not None and conclusion not in {"success", "failure", "cancelled", "timed_out", "action_required"}:
            raise ValueError("unsupported CI conclusion.")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM repair_cycles WHERE project = ? AND id = ?",
                (project, cycle_id),
            ).fetchone()
            if row is None:
                raise ValueError("repair cycle not found in this project.")
            current = dict(row)
            if current["state"] in {"completed", "stop_and_report"} and state != current["state"]:
                raise ValueError("terminal repair cycle cannot be reopened.")
            if state == "completed" and conclusion != "success":
                raise ValueError("a repair cycle completes only after CI success.")
            if state == "awaiting_ci" and current["attempts_used"] == 0:
                raise ValueError("cannot await CI before recording a repair attempt.")
            self._connection.execute(
                "UPDATE repair_cycles SET state=?, last_conclusion=?, updated_at=? WHERE project=? AND id=?",
                (state, conclusion, now, project, cycle_id),
            )
            updated = self._connection.execute(
                "SELECT * FROM repair_cycles WHERE project = ? AND id = ?",
                (project, cycle_id),
            ).fetchone()
        return dict(updated)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
