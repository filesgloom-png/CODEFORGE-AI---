"""Read-only GitHub REST adapter for CODEFORGE AI.

The adapter never writes to repositories and never executes project code.
A token is optional for public repositories and is read only from the environment.
"""
from __future__ import annotations

import base64
import json
import os
import re
import io
import zipfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class GitHubAdapterError(RuntimeError):
    """Raised when GitHub data cannot be retrieved safely."""


_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MAX_RESPONSE_BYTES = 2_000_000
_MAX_FILE_BYTES = 500_000


class GitHubReadOnlyAdapter:
    """Bounded read-only access to repository trees, files, Actions runs and jobs."""

    def __init__(
        self,
        repository: str,
        token: str | None = None,
        api_base: str = "https://api.github.com",
        timeout: float = 20.0,
    ) -> None:
        if not isinstance(repository, str) or not _REPO_RE.fullmatch(repository):
            raise ValueError("repository must use owner/name format.")
        if api_base.rstrip("/") != "https://api.github.com":
            raise ValueError("Only the official GitHub API host is supported.")
        self.repository = repository
        self._token = (token if token is not None else os.getenv("GITHUB_TOKEN", "")).strip()
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout

    def _get_json(self, path: str) -> Any:
        if not path.startswith("/") or path.startswith("//") or ".." in path.split("/"):
            raise GitHubAdapterError("Invalid GitHub API path.")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "CODEFORGE-AI-readonly-adapter",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(self._api_base + path, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise GitHubAdapterError(f"GitHub API returned HTTP {exc.code}.") from None
        except (URLError, TimeoutError, OSError):
            raise GitHubAdapterError("Could not reach the GitHub API.") from None
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise GitHubAdapterError("GitHub response exceeded the size limit.")
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise GitHubAdapterError("GitHub returned an invalid JSON response.") from None

    def get_tree(self, ref: str = "main", recursive: bool = True) -> dict[str, Any]:
        """Return bounded tree metadata, excluding file contents."""
        if not isinstance(ref, str) or not ref or len(ref) > 200:
            raise ValueError("ref must be a non-empty branch, tag, or commit.")
        endpoint = f"/repos/{self.repository}/git/trees/{quote(ref, safe='')}"
        if recursive:
            endpoint += "?recursive=1"
        payload = self._get_json(endpoint)
        if not isinstance(payload, dict) or not isinstance(payload.get("tree"), list):
            raise GitHubAdapterError("GitHub tree response has an unexpected shape.")
        entries = payload["tree"]
        if len(entries) > 20_000:
            raise GitHubAdapterError("Repository tree exceeds the 20,000-entry limit.")
        safe_entries = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            kind = item.get("type")
            sha = item.get("sha")
            if isinstance(path, str) and kind in {"blob", "tree", "commit"}:
                safe_entries.append({"path": path, "type": kind, "sha": sha})
        return {"sha": payload.get("sha"), "truncated": bool(payload.get("truncated")), "tree": safe_entries}

    def get_file(self, path: str, ref: str = "main") -> str:
        """Read a UTF-8 text file through the Contents API with strict size checks."""
        if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
            raise ValueError("path must be a safe repository-relative path.")
        if any(part in {"", ".", ".."} for part in path.split("/")):
            raise ValueError("path contains an unsafe segment.")
        endpoint = (
            f"/repos/{self.repository}/contents/{quote(path, safe='/')}"
            f"?ref={quote(ref, safe='')}"
        )
        payload = self._get_json(endpoint)
        if not isinstance(payload, dict) or payload.get("type") != "file":
            raise GitHubAdapterError("The requested path is not a file.")
        if payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
            raise GitHubAdapterError("GitHub did not return a supported text-file payload.")
        encoded = payload["content"].replace("\n", "")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error):
            raise GitHubAdapterError("GitHub returned invalid file encoding.") from None
        if len(data) > _MAX_FILE_BYTES:
            raise GitHubAdapterError("File exceeds the 500,000-byte limit.")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise GitHubAdapterError("Binary or non-UTF-8 files are not supported.") from None

    def list_workflow_runs(self, branch: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """List recent Actions runs with a bounded result count."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50.")
        query = f"?per_page={limit}"
        if branch:
            query += f"&branch={quote(branch, safe='')}"
        payload = self._get_json(f"/repos/{self.repository}/actions/runs{query}")
        runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
        if not isinstance(runs, list):
            raise GitHubAdapterError("GitHub Actions response has an unexpected shape.")
        return [
            {
                "id": run.get("id"),
                "run_number": run.get("run_number"),
                "name": run.get("name"),
                "head_branch": run.get("head_branch"),
                "head_sha": run.get("head_sha"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "html_url": run.get("html_url"),
            }
            for run in runs[:limit]
            if isinstance(run, dict)
        ]

    def get_workflow_run(self, run_id: int) -> dict[str, Any]:
        """Read a single run summary; run_id must be a positive integer."""
        if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
            raise ValueError("run_id must be a positive integer.")
        payload = self._get_json(f"/repos/{self.repository}/actions/runs/{run_id}")
        if not isinstance(payload, dict):
            raise GitHubAdapterError("GitHub run response has an unexpected shape.")
        fields = ("id", "run_number", "name", "status", "conclusion", "head_branch",
                  "head_sha", "html_url", "created_at", "updated_at")
        return {field: payload.get(field) for field in fields}

    def list_workflow_runs_for_sha(self, commit_sha: str, limit: int = 20) -> list[dict[str, Any]]:
        """Read recent Actions runs filtered by an exact commit SHA."""
        if not isinstance(commit_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit_sha):
            raise ValueError("commit_sha must be a full Git commit SHA.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50.")
        payload = self._get_json(
            f"/repos/{self.repository}/actions/runs?head_sha={quote(commit_sha, safe='')}&per_page={limit}"
        )
        runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
        if not isinstance(runs, list):
            raise GitHubAdapterError("GitHub Actions response has an unexpected shape.")
        return [{
            "id": run.get("id"), "name": run.get("name"), "head_sha": run.get("head_sha"),
            "status": run.get("status"), "conclusion": run.get("conclusion"),
            "html_url": run.get("html_url"),
        } for run in runs[:limit] if isinstance(run, dict) and run.get("head_sha") == commit_sha]

    def get_pull_request(self, pr_number: int) -> dict[str, Any]:
        """Read bounded metadata for one pull request without exposing credentials."""
        if not isinstance(pr_number, int) or isinstance(pr_number, bool) or pr_number <= 0:
            raise ValueError("pr_number must be a positive integer.")
        payload = self._get_json(f"/repos/{self.repository}/pulls/{pr_number}")
        if not isinstance(payload, dict):
            raise GitHubAdapterError("GitHub pull request response has an unexpected shape.")
        head = payload.get("head") if isinstance(payload.get("head"), dict) else {}
        base = payload.get("base") if isinstance(payload.get("base"), dict) else {}
        return {
            "number": payload.get("number"), "state": payload.get("state"),
            "draft": payload.get("draft"), "merged": payload.get("merged"),
            "title": payload.get("title"), "html_url": payload.get("html_url"),
            "head_branch": head.get("ref"), "head_sha": head.get("sha"),
            "base_branch": base.get("ref"), "base_sha": base.get("sha"),
            "mergeable": payload.get("mergeable"),
        }

    def list_pull_request_runs(self, pr_number: int, limit: int = 20) -> list[dict[str, Any]]:
        """Read recent Actions runs associated with a PR head SHA; never starts runs."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50.")
        pr = self.get_pull_request(pr_number)
        head_sha = pr.get("head_sha")
        if not isinstance(head_sha, str) or not head_sha:
            raise GitHubAdapterError("Pull request has no readable head SHA.")
        payload = self._get_json(
            f"/repos/{self.repository}/actions/runs?head_sha={quote(head_sha, safe='')}&per_page={limit}"
        )
        runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
        if not isinstance(runs, list):
            raise GitHubAdapterError("GitHub Actions response has an unexpected shape.")
        return [{
            "id": run.get("id"), "name": run.get("name"),
            "head_sha": run.get("head_sha"), "status": run.get("status"),
            "conclusion": run.get("conclusion"), "html_url": run.get("html_url"),
        } for run in runs[:limit] if isinstance(run, dict)]

    def fetch_job_logs(self, job_id: int, max_bytes: int = 5_000_000) -> str:
        """Download a job log archive and return bounded UTF-8 text files only."""
        if not isinstance(job_id, int) or isinstance(job_id, bool) or job_id <= 0:
            raise ValueError("job_id must be a positive integer.")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or not 1 <= max_bytes <= 5_000_000:
            raise ValueError("max_bytes must be between 1 and 5,000,000.")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "CODEFORGE-AI-readonly-adapter",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(
            f"{self._api_base}/repos/{self.repository}/actions/jobs/{job_id}/logs",
            headers=headers,
            method="GET",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read(max_bytes + 1)
        except HTTPError as exc:
            raise GitHubAdapterError(f"GitHub log endpoint returned HTTP {exc.code}.") from None
        except (URLError, TimeoutError, OSError):
            raise GitHubAdapterError("Could not download GitHub job logs.") from None
        if len(raw) > max_bytes:
            raise GitHubAdapterError("Job log archive exceeded the configured size limit.")
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                parts = []
                total = 0
                for info in archive.infolist():
                    name = info.filename
                    if info.is_dir() or not name.lower().endswith(".txt"):
                        continue
                    if name.startswith("/") or ".." in name.replace("\\\\", "/").split("/"):
                        raise GitHubAdapterError("Log archive contains an unsafe path.")
                    if info.file_size > max_bytes or total + info.file_size > max_bytes:
                        raise GitHubAdapterError("Extracted logs exceed the configured size limit.")
                    data = archive.read(info)
                    parts.append(data.decode("utf-8", errors="replace"))
                    total += len(data)
                if not parts:
                    raise GitHubAdapterError("Log archive contains no text log files.")
                return "\\n".join(parts)
        except zipfile.BadZipFile:
            # Some compatible API gateways may return a plain text payload.
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                raise GitHubAdapterError("GitHub returned an unsupported log format.") from None

    def list_run_jobs(self, run_id: int, limit: int = 50) -> list[dict[str, Any]]:
        """Read step/job status summaries for a run (not raw log archives)."""
        if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
            raise ValueError("run_id must be a positive integer.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        payload = self._get_json(
            f"/repos/{self.repository}/actions/runs/{run_id}/jobs?per_page={limit}"
        )
        jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(jobs, list):
            raise GitHubAdapterError("GitHub jobs response has an unexpected shape.")
        result = []
        for job in jobs[:limit]:
            if not isinstance(job, dict):
                continue
            steps = job.get("steps", [])
            result.append({
                "id": job.get("id"),
                "name": job.get("name"),
                "status": job.get("status"),
                "conclusion": job.get("conclusion"),
                "html_url": job.get("html_url"),
                "steps": [
                    {"name": step.get("name"), "status": step.get("status"),
                     "conclusion": step.get("conclusion")}
                    for step in steps if isinstance(step, dict)
                ] if isinstance(steps, list) else [],
            })
        return result
