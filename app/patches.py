"""Validate model-proposed file changes before any GitHub write operation.

This module deliberately validates data only: it does not access GitHub, execute
commands, or write to disk. A later executor can consume only validated changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


MAX_CHANGES = 20
MAX_FILE_BYTES = 200_000
MAX_TOTAL_BYTES = 1_000_000
_BLOCKED_PARTS = {".git", ".env", ".venv", "venv", "__pycache__"}
_BLOCKED_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "secrets.json",
}


class PatchValidationError(ValueError):
    """Raised when a proposed patch is malformed or outside safety limits."""


@dataclass(frozen=True)
class FileChange:
    """A validated, single-file operation."""

    path: str
    action: str
    content: str | None = None


def _validate_path(path: object) -> str:
    if not isinstance(path, str) or not path or path.strip() != path:
        raise PatchValidationError("Every change must have a non-empty relative path.")
    if len(path) > 240 or path.startswith("/") or "\\" in path:
        raise PatchValidationError("A change contains an invalid path.")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PatchValidationError("A change contains an unsafe path segment.")
    if any(part.lower() in _BLOCKED_PARTS for part in parts):
        raise PatchValidationError("Changes to protected paths are not allowed.")
    name = parts[-1].lower()
    if name in _BLOCKED_NAMES or name.endswith((".pem", ".key", ".p12", ".pfx")):
        raise PatchValidationError("Changes to credential-like files are not allowed.")
    if name.startswith(".env."):
        raise PatchValidationError("Changes to environment files are not allowed.")
    if parts[:2] == [".github", "workflows"]:
        raise PatchValidationError("Generated patches cannot modify CI workflow files.")
    return path


def validate_changes(
    raw: str,
    existing_paths: Iterable[str] | None = None,
    *,
    max_changes: int = MAX_CHANGES,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> tuple[FileChange, ...]:
    """Parse and validate a JSON patch envelope without applying it.

    Expected shape:
      {"changes": [{"path": "app/example.py", "action": "create",
                    "content": "..." }]}
    Actions are create, update, and delete. When existing_paths is supplied,
    create/update/delete semantics are checked against that known snapshot.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise PatchValidationError("Patch response must be non-empty JSON.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise PatchValidationError("Patch response is not valid JSON.") from None
    if not isinstance(payload, dict) or set(payload) != {"changes"}:
        raise PatchValidationError("Patch response must contain only a changes array.")
    items = payload["changes"]
    if not isinstance(items, list) or not items:
        raise PatchValidationError("Changes must be a non-empty array.")
    if len(items) > max_changes:
        raise PatchValidationError(f"A patch may contain at most {max_changes} files.")

    known = set(existing_paths) if existing_paths is not None else None
    seen: set[str] = set()
    result: list[FileChange] = []
    total_bytes = 0

    for item in items:
        if not isinstance(item, dict):
            raise PatchValidationError("Each change must be an object.")
        action = item.get("action")
        if action not in {"create", "update", "delete"}:
            raise PatchValidationError("Each change needs a supported action.")
        expected_keys = {"path", "action"} if action == "delete" else {"path", "action", "content"}
        if set(item) != expected_keys:
            raise PatchValidationError("A change has missing or unexpected fields.")
        path = _validate_path(item.get("path"))
        if path in seen:
            raise PatchValidationError("A patch cannot change the same path twice.")
        seen.add(path)

        if known is not None:
            exists = path in known
            if action == "create" and exists:
                raise PatchValidationError("A create operation targets an existing file.")
            if action in {"update", "delete"} and not exists:
                raise PatchValidationError("An update/delete operation targets an unknown file.")

        if action == "delete":
            result.append(FileChange(path=path, action=action))
            continue

        content = item.get("content")
        if not isinstance(content, str):
            raise PatchValidationError("File content must be a string.")
        size = len(content.encode("utf-8"))
        if size > max_file_bytes:
            raise PatchValidationError(f"A file exceeds the {max_file_bytes}-byte limit.")
        total_bytes += size
        if total_bytes > max_total_bytes:
            raise PatchValidationError(f"A patch exceeds the {max_total_bytes}-byte total limit.")
        result.append(FileChange(path=path, action=action, content=content))

    return tuple(result)
