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

    def close(self) -> None:
        with self._lock:
            self._connection.close()
