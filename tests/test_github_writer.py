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


if __name__ == "__main__":
    unittest.main()
