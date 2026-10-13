"""Offline tests for the read-only GitHub adapter."""
import base64
import io
import json
import unittest
from unittest.mock import patch

from app.github_adapter import GitHubAdapterError, GitHubReadOnlyAdapter


def response(payload):
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


class GitHubAdapterTests(unittest.TestCase):
    def test_rejects_invalid_repository_and_non_github_host(self):
        with self.assertRaises(ValueError):
            GitHubReadOnlyAdapter("../bad")
        with self.assertRaises(ValueError):
            GitHubReadOnlyAdapter("owner/repo", api_base="https://example.com")

    @patch("app.github_adapter.urlopen")
    def test_tree_returns_only_bounded_metadata(self, mocked):
        mocked.return_value = response({
            "sha": "tree-sha", "truncated": False,
            "tree": [{"path": "app/main.py", "type": "blob", "sha": "blob-sha",
                      "url": "should-not-leak"}]
        })
        result = GitHubReadOnlyAdapter("owner/repo").get_tree()
        self.assertEqual(result["tree"], [
            {"path": "app/main.py", "type": "blob", "sha": "blob-sha"}
        ])
        self.assertIn("/git/trees/main?recursive=1", mocked.call_args.args[0].full_url)

    @patch("app.github_adapter.urlopen")
    def test_reads_base64_utf8_file(self, mocked):
        content = base64.b64encode("print('привет')\n".encode()).decode()
        mocked.return_value = response({"type": "file", "encoding": "base64", "content": content})
        self.assertEqual(GitHubReadOnlyAdapter("owner/repo").get_file("app/main.py"),
                         "print('привет')\n")

    def test_rejects_unsafe_file_paths_before_network(self):
        client = GitHubReadOnlyAdapter("owner/repo")
        for path in ("../secret", "/absolute", r"app\secret"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                client.get_file(path)

    @patch("app.github_adapter.urlopen")
    def test_lists_sanitized_workflow_run_summaries(self, mocked):
        mocked.return_value = response({"workflow_runs": [{
            "id": 7, "run_number": 3, "name": "CI", "head_branch": "main",
            "head_sha": "abc", "status": "completed", "conclusion": "success",
            "html_url": "https://github.com/owner/repo/actions/runs/7",
            "token": "must-not-be-returned"
        }]})
        runs = GitHubReadOnlyAdapter("owner/repo").list_workflow_runs(limit=2)
        self.assertEqual(len(runs), 1)
        self.assertNotIn("token", runs[0])
        self.assertEqual(runs[0]["conclusion"], "success")

    @patch("app.github_adapter.urlopen")
    def test_reads_run_job_and_step_statuses(self, mocked):
        mocked.return_value = response({"jobs": [{
            "id": 8, "name": "test", "status": "completed", "conclusion": "success",
            "steps": [{"name": "unit tests", "status": "completed", "conclusion": "success"}]
        }]})
        jobs = GitHubReadOnlyAdapter("owner/repo").list_run_jobs(7)
        self.assertEqual(jobs[0]["steps"][0]["name"], "unit tests")

    @patch("app.github_adapter.urlopen")
    def test_http_errors_are_sanitized(self, mocked):
        from urllib.error import HTTPError
        mocked.side_effect = HTTPError(
            "https://api.github.com", 403, "Forbidden", {}, io.BytesIO(b"private details")
        )
        with self.assertRaises(GitHubAdapterError) as caught:
            GitHubReadOnlyAdapter("owner/repo").get_tree()
        self.assertIn("HTTP 403", str(caught.exception))
        self.assertNotIn("private details", str(caught.exception))

    def test_bounds_limits(self):
        client = GitHubReadOnlyAdapter("owner/repo")
        with self.assertRaises(ValueError):
            client.list_workflow_runs(limit=51)
        with self.assertRaises(ValueError):
            client.list_run_jobs(1, limit=101)


if __name__ == "__main__":
    unittest.main()
