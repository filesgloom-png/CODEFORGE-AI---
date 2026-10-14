"""Create isolated GitHub branches from validated file-change envelopes.

This writer is opt-in and requires a server-side GitHub token with repository
contents write permission. It never pushes to the target branch or runs code.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.github_adapter import GitHubReadOnlyAdapter, GitHubAdapterError
from app.patches import validate_changes, PatchValidationError


class GitHubWriteError(RuntimeError):
    """Raised when a safe isolated patch cannot be published."""


class GitHubPatchWriter:
    """Publish validated changes on a newly created branch, never on main."""

    def __init__(
        self,
        repository: str,
        token: str | None = None,
        api_base: str = "https://api.github.com",
        timeout: float = 20.0,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repository or ""):
            raise ValueError("repository must use owner/name format.")
        if api_base != "https://api.github.com":
            raise ValueError("Only the official GitHub API host is supported.")
        self.repository = repository
        self._token = (token if token is not None else os.getenv("GITHUB_TOKEN", "")).strip()
        self._api_base = api_base
        self._timeout = timeout

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        if not self._token:
            raise GitHubWriteError("GITHUB_TOKEN is missing; no repository changes were made.")
        if not path.startswith("/") or path.startswith("//") or ".." in path.split("/"):
            raise GitHubWriteError("Invalid GitHub API path.")
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self._api_base + path,
            data=data,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "CODEFORGE-AI-patch-writer",
            },
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read(1_000_001)
        except HTTPError as exc:
            raise GitHubWriteError(
                f"GitHub rejected the operation with HTTP {exc.code}; the patch was not confirmed."
            ) from None
        except (URLError, TimeoutError, OSError):
            raise GitHubWriteError("GitHub could not be reached; inspect the branch before retrying.") from None
        if len(raw) > 1_000_000:
            raise GitHubWriteError("GitHub response exceeded the size limit.")
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise GitHubWriteError("GitHub returned an invalid JSON response.") from None

    def publish_patch(
        self,
        raw_patch: str,
        base_branch: str = "main",
        description: str = "validated CODEFORGE AI patch",
    ) -> dict[str, Any]:
        """Validate a JSON patch, create a branch, commit changes, and return its URL."""
        if not re.fullmatch(r"[A-Za-z0-9._/-]{1,120}", base_branch or ""):
            raise ValueError("base_branch contains unsupported characters.")
        if base_branch.startswith("/") or ".." in base_branch.split("/"):
            raise ValueError("base_branch contains an unsafe path segment.")
        reader = GitHubReadOnlyAdapter(self.repository, token=self._token)
        try:
            base_ref = reader._get_json(
                f"/repos/{self.repository}/git/ref/heads/{quote(base_branch, safe='')}"
            )
            base_sha = base_ref["object"]["sha"]
            base_commit = reader._get_json(
                f"/repos/{self.repository}/git/commits/{quote(base_sha, safe='')}"
            )
            base_tree_sha = base_commit["tree"]["sha"]
            tree_snapshot = reader.get_tree(ref=base_sha, recursive=True)
        except (GitHubAdapterError, KeyError, TypeError):
            raise GitHubWriteError("Could not inspect the selected base branch safely.") from None

        existing = {
            item["path"] for item in tree_snapshot["tree"]
            if item.get("type") == "blob" and isinstance(item.get("path"), str)
        }
        try:
            changes = validate_changes(raw_patch, existing_paths=existing)
        except PatchValidationError as exc:
            raise GitHubWriteError(f"Patch rejected before any write: {exc}") from None

        slug = re.sub(r"[^a-z0-9]+", "-", description.lower()).strip("-")[:36] or "patch"
        branch = f"codeforge/{slug}-{base_sha[:7]}"
        self._request_json("POST", f"/repos/{self.repository}/git/refs", {
            "ref": f"refs/heads/{branch}", "sha": base_sha,
        })

        tree_elements = []
        for change in changes:
            if change.action == "delete":
                tree_elements.append({"path": change.path, "mode": "100644", "type": "blob", "sha": None})
                continue
            blob = self._request_json("POST", f"/repos/{self.repository}/git/blobs", {
                "content": change.content, "encoding": "utf-8",
            })
            blob_sha = blob.get("sha")
            if not isinstance(blob_sha, str):
                raise GitHubWriteError("GitHub did not confirm a file blob; inspect the created branch.")
            tree_elements.append({
                "path": change.path, "mode": "100644", "type": "blob", "sha": blob_sha,
            })

        tree = self._request_json("POST", f"/repos/{self.repository}/git/trees", {
            "base_tree": base_tree_sha, "tree": tree_elements,
        })
        tree_sha = tree.get("sha")
        if not isinstance(tree_sha, str):
            raise GitHubWriteError("GitHub did not confirm the patch tree; inspect the created branch.")
        commit = self._request_json("POST", f"/repos/{self.repository}/git/commits", {
            "message": f"codeforge: {slug}", "tree": tree_sha, "parents": [base_sha],
        })
        commit_sha = commit.get("sha")
        if not isinstance(commit_sha, str):
            raise GitHubWriteError("GitHub did not confirm the patch commit; inspect the created branch.")
        self._request_json(
            "PATCH",
            f"/repos/{self.repository}/git/refs/heads/{quote(branch, safe='/')}",
            {"sha": commit_sha, "force": False},
        )
        return {
            "repository": self.repository,
            "base_branch": base_branch,
            "base_sha": base_sha,
            "branch": branch,
            "commit_sha": commit_sha,
            "changed_files": [change.path for change in changes],
            "branch_url": f"https://github.com/{self.repository}/tree/{quote(branch, safe='/')}",
            "commit_url": f"https://github.com/{self.repository}/commit/{commit_sha}",
            "pull_request_created": False,
            "checks_run": False,
        }

    def create_pull_request(
        self,
        head_branch: str,
        base_branch: str = "main",
        title: str = "CODEFORGE AI proposed change",
        body: str = "Created from a validated isolated CODEFORGE AI branch. Review the diff and CI before merging.",
        draft: bool = True,
    ) -> dict[str, Any]:
        """Create a draft PR only from a dedicated codeforge/* branch; never merge it."""
        if not isinstance(head_branch, str) or not re.fullmatch(r"codeforge/[A-Za-z0-9._/-]{1,100}", head_branch):
            raise ValueError("head_branch must be a dedicated codeforge/* branch.")
        if ".." in head_branch.split("/") or head_branch.startswith("/"):
            raise ValueError("head_branch contains an unsafe path segment.")
        if not isinstance(base_branch, str) or not re.fullmatch(r"[A-Za-z0-9._/-]{1,120}", base_branch):
            raise ValueError("base_branch contains unsupported characters.")
        if ".." in base_branch.split("/") or base_branch.startswith("/") or head_branch == base_branch:
            raise ValueError("base_branch is unsafe or identical to head_branch.")
        if not isinstance(title, str) or not title.strip() or len(title) > 200:
            raise ValueError("title must contain 1..200 characters.")
        if not isinstance(body, str) or len(body) > 20_000:
            raise ValueError("body must be text no longer than 20,000 characters.")
        payload = self._request_json("POST", f"/repos/{self.repository}/pulls", {
            "title": title.strip(), "body": body,
            "head": head_branch, "base": base_branch, "draft": bool(draft),
            "maintainer_can_modify": False,
        })
        number = payload.get("number")
        url = payload.get("html_url")
        if not isinstance(number, int) or not isinstance(url, str):
            raise GitHubWriteError("GitHub did not confirm the pull request; inspect the repository before retrying.")
        return {
            "number": number, "url": url, "title": payload.get("title"),
            "state": payload.get("state"), "draft": payload.get("draft", bool(draft)),
            "head_branch": head_branch, "base_branch": base_branch,
            "merged": False, "merge_performed": False,
        }
