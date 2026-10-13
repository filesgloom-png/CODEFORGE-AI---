"""Tests for model-proposed patch validation."""
import json
import unittest

from app.patches import FileChange, PatchValidationError, validate_changes


def envelope(*changes):
    return json.dumps({"changes": list(changes)})


class ValidateChangesTests(unittest.TestCase):
    def test_accepts_create_and_preserves_unicode_content(self):
        result = validate_changes(envelope({
            "path": "app/hello.py", "action": "create", "content": "print('привет')\n"
        }))
        self.assertEqual(result, (FileChange("app/hello.py", "create", "print('привет')\n"),))

    def test_accepts_delete_without_content(self):
        result = validate_changes(envelope({"path": "app/old.py", "action": "delete"}))
        self.assertEqual(result[0].action, "delete")
        self.assertIsNone(result[0].content)

    def test_rejects_parent_traversal_and_absolute_paths(self):
        for path in ("../secret.txt", "app/../../secret.txt", "/tmp/x.py", r"app\x.py"):
            with self.subTest(path=path), self.assertRaises(PatchValidationError):
                validate_changes(envelope({
                    "path": path, "action": "create", "content": "x"
                }))

    def test_rejects_credentials_and_workflow_changes(self):
        for path in (".env", "config/credentials.json", "keys/private.pem",
                     ".github/workflows/ci.yml"):
            with self.subTest(path=path), self.assertRaises(PatchValidationError):
                validate_changes(envelope({
                    "path": path, "action": "create", "content": "x"
                }))

    def test_rejects_duplicate_paths(self):
        change = {"path": "app/a.py", "action": "create", "content": "x"}
        with self.assertRaises(PatchValidationError):
            validate_changes(envelope(change, change))

    def test_rejects_unknown_fields_and_malformed_envelopes(self):
        with self.assertRaises(PatchValidationError):
            validate_changes(json.dumps({"changes": [], "command": "rm -rf /"}))
        with self.assertRaises(PatchValidationError):
            validate_changes("not json")

    def test_checks_create_update_delete_against_known_snapshot(self):
        known = {"app/existing.py"}
        with self.assertRaises(PatchValidationError):
            validate_changes(envelope({
                "path": "app/existing.py", "action": "create", "content": "x"
            }), known)
        with self.assertRaises(PatchValidationError):
            validate_changes(envelope({
                "path": "app/missing.py", "action": "update", "content": "x"
            }), known)
        with self.assertRaises(PatchValidationError):
            validate_changes(envelope({
                "path": "app/missing.py", "action": "delete"
            }), known)

    def test_rejects_too_many_files_and_oversized_content(self):
        changes = [
            {"path": f"src/file_{i}.py", "action": "create", "content": ""}
            for i in range(3)
        ]
        with self.assertRaises(PatchValidationError):
            validate_changes(envelope(*changes), max_changes=2)
        with self.assertRaises(PatchValidationError):
            validate_changes(envelope({
                "path": "src/big.py", "action": "create", "content": "x" * 11
            }), max_file_bytes=10)

    def test_rejects_empty_content_type_and_empty_change_list(self):
        with self.assertRaises(PatchValidationError):
            validate_changes(envelope({"path": "src/a.py", "action": "create", "content": None}))
        with self.assertRaises(PatchValidationError):
            validate_changes(json.dumps({"changes": []}))


if __name__ == "__main__":
    unittest.main()
