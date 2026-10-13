"""Offline tests for isolated GitHub patch publication."""
import io
import json
import unittest
from unittest.mock import patch

from app.github_writer import GitHubPatchWriter, GitHubWriteError


class GitHubWriterTests(unittest.TestCase):
    def test_requires_token_before_any_write(self):
        writer = GitHubPatchWriter("owner/repo", token="")
        with self.assertRaises(GitHubWriteError):
            writer._request_json("POST", "/repos/owner/repo/git/refs", {})
    
    def test_rejects_invalid_repo_and_api_host(self):
        with self.assertRaises(ValueError):
            GitHubPatchWriter("../repo", token="token")
        with self.assertRaises(ValueError):
            GitHubPatchWriter("owner/repo", token="token", api_base="https://api.github.com.evil.invalid")

    def test_rejects_invalid_base_branch(self):
        writer = GitHubPatchWriter("owner/repo", token="token")
        with self.assertRaises(ValueError):
            writer.publish_patch('{"changes":[]}', base_branch="../main")

    @patch("app.github_writer.urlopen")
    def test_write_request_uses_bearer_token_and_json(self, mocked):
        mocked.return_value = io.BytesIO(json.dumps({"sha": "new-sha"}).encode())
        writer = GitHubPatchWriter("owner/repo", token="secret-token")
        result = writer._request_json("POST", "/repos/owner/repo/git/blobs", {
            "content": "hello", "encoding": "utf-8"
        })
        self.assertEqual(result["sha"], "new-sha")
        request = mocked.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertEqual(json.loads(request.data.decode()), {
            "content": "hello", "encoding": "utf-8"
        })

    @patch("app.github_writer.urlopen")
    def test_http_error_does_not_leak_response_body(self, mocked):
        from urllib.error import HTTPError
        mocked.side_effect = HTTPError(
            "https://api.github.com", 403, "Forbidden", {}, io.BytesIO(b"private-token")
        )
        writer = GitHubPatchWriter("owner/repo", token="secret-token")
        with self.assertRaises(GitHubWriteError) as caught:
            writer._request_json("POST", "/repos/owner/repo/git/refs", {})
        self.assertIn("HTTP 403", str(caught.exception))
        self.assertNotIn("private-token", str(caught.exception))


    @patch("app.github_writer.GitHubReadOnlyAdapter")
    def test_mocked_publish_integration_creates_only_isolated_branch(self, reader_class):
        """Exercise validation -> branch -> blob/tree/commit -> branch ref, offline."""
        reader = reader_class.return_value
        reader._get_json.side_effect = [
            {"object": {"sha": "base-sha-1234567"}},
            {"tree": {"sha": "base-tree-sha"}},
        ]
        reader.get_tree.return_value = {
            "truncated": False,
            "tree": [{"path": "README.md", "type": "blob"}],
        }
        writer = GitHubPatchWriter("owner/repo", token="fake-test-token")
        responses = [
            {},  # create refs/heads/codeforge/...
            {"sha": "new-blob-sha"},
            {"sha": "new-tree-sha"},
            {"sha": "new-commit-sha"},
            {},  # update isolated branch ref
        ]
        with patch.object(writer, "_request_json", side_effect=responses) as request:
            patch_json = json.dumps({"changes": [{
                "path": "notes/integration-test.txt",
                "action": "create",
                "content": "mock-only integration test\\n",
            }]})
            result = writer.publish_patch(patch_json, description="mock integration")

        self.assertEqual(result["base_branch"], "main")
        self.assertEqual(result["base_sha"], "base-sha-1234567")
        self.assertEqual(result["commit_sha"], "new-commit-sha")
        self.assertEqual(result["changed_files"], ["notes/integration-test.txt"])
        self.assertFalse(result["checks_run"])
        self.assertFalse(result["pull_request_created"])
        self.assertTrue(result["branch"].startswith("codeforge/"))
        self.assertEqual(request.call_count, 5)

        calls = request.call_args_list
        self.assertEqual(calls[0].args[0], "POST")
        self.assertEqual(calls[0].args[1], "/repos/owner/repo/git/refs")
        self.assertEqual(calls[0].args[2]["ref"], f"refs/heads/{result['branch']}")
        self.assertEqual(calls[1].args[1], "/repos/owner/repo/git/blobs")
        self.assertEqual(calls[2].args[1], "/repos/owner/repo/git/trees")
        self.assertEqual(calls[3].args[1], "/repos/owner/repo/git/commits")
        self.assertEqual(calls[4].args[0], "PATCH")
        self.assertEqual(
            calls[4].args[1],
            f"/repos/owner/repo/git/refs/heads/{result['branch']}",
        )
        self.assertNotIn("/git/refs/heads/main", [call.args[1] for call in calls])

    @patch("app.github_writer.GitHubReadOnlyAdapter")
    def test_invalid_patch_is_rejected_before_any_github_write(self, reader_class):
        reader = reader_class.return_value
        reader._get_json.side_effect = [
            {"object": {"sha": "base-sha-1234567"}},
            {"tree": {"sha": "base-tree-sha"}},
        ]
        reader.get_tree.return_value = {"truncated": False, "tree": []}
        writer = GitHubPatchWriter("owner/repo", token="fake-test-token")
        with patch.object(writer, "_request_json") as request:
            with self.assertRaises(GitHubWriteError):
                writer.publish_patch(json.dumps({"changes": [{
                    "path": "../escape.txt", "action": "create", "content": "nope"
                }]}))
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
