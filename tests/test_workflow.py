"""End-to-end orchestration and safe CI log diagnosis tests."""
import json
import unittest
from unittest.mock import Mock

from app.core import TaskKind, TaskRequest
from app.diagnostics import diagnose_log, next_repair_action
from app.workflow import ControlledAgentWorkflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.reader = Mock()
        self.writer = Mock()
        self.reader.get_tree.return_value = {
            "sha": "tree-sha", "truncated": False,
            "tree": [{"path": "app/existing.py", "type": "blob", "sha": "blob-sha"}],
        }
        self.reader.list_workflow_runs.return_value = [
            {"run_number": 12, "status": "completed", "conclusion": "success"}
        ]
        self.reader.fetch_job_logs.return_value = "AssertionError: expected result"
        self.workflow = ControlledAgentWorkflow(
            "owner/repo", token="test-token", reader=self.reader, writer=self.writer
        )

    def test_inspection_integrates_plan_tree_and_actions(self):
        result = self.workflow.inspect(TaskRequest("Fix the issue", "owner/repo", TaskKind.FIX))
        self.assertEqual(result.status, "inspected")
        self.assertEqual(result.details["tree"]["tree"][0]["path"], "app/existing.py")
        self.assertEqual(result.details["recent_runs"][0]["conclusion"], "success")

    def test_invalid_patch_never_calls_writer(self):
        result = self.workflow.prepare_branch(json.dumps({"changes": [
            {"path": "../bad.py", "action": "create", "content": "x"}
        ]}))
        self.assertEqual(result.status, "rejected")
        self.writer.publish_patch.assert_not_called()

    def test_valid_patch_calls_writer_only_after_validation(self):
        self.writer.publish_patch.return_value = {
            "branch": "codeforge/fix-abc1234", "commit_sha": "commit-sha",
            "changed_files": ["app/new.py"], "checks_run": False,
        }
        patch = json.dumps({"changes": [
            {"path": "app/new.py", "action": "create", "content": "print('ok')\\n"}
        ]})
        result = self.workflow.prepare_branch(patch, description="fix")
        self.assertEqual(result.status, "branch_prepared")
        self.writer.publish_patch.assert_called_once_with(
            patch, base_branch="main", description="fix"
        )

    def test_truncated_tree_blocks_validation(self):
        self.reader.get_tree.return_value = {"sha": "tree", "truncated": True, "tree": []}
        result = self.workflow.validate_patch(json.dumps({"changes": [
            {"path": "app/new.py", "action": "create", "content": "x"}
        ]}))
        self.assertEqual(result.status, "rejected")
        self.writer.publish_patch.assert_not_called()

    def test_model_output_must_be_strict_json_envelope(self):
        with self.assertRaises(ValueError):
            self.workflow.decode_model_patch("run rm -rf /")
        self.assertEqual(
            json.loads(self.workflow.decode_model_patch('{"changes": []}')),
            {"changes": []},
        )

    def test_diagnose_job_uses_safe_log_diagnosis_and_bounded_policy(self):
        result = self.workflow.diagnose_job(123, attempts_used=2)
        self.assertEqual(result.status, "diagnosed")
        self.assertEqual(result.details["next_action"], "stop_and_report")
        self.assertEqual(result.details["findings"], ["python_assertion"])


class DiagnosticsTests(unittest.TestCase):
    def test_detects_failure_and_redacts_tokens(self):
        diagnosis = diagnose_log(
            "Traceback (most recent call last):\\n"
            "token=ghp_abcdefghijklmnopqrstuvwxyz123456\\n"
            "ModuleNotFoundError: No module named 'missing'\\n"
        )
        self.assertEqual(diagnosis.status, "failure_signals")
        self.assertIn("python_traceback", diagnosis.findings)
        self.assertIn("python_import_error", diagnosis.findings)
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz123456", "\\n".join(diagnosis.excerpts))
        self.assertIn("[REDACTED]", "\\n".join(diagnosis.excerpts))

    def test_bounded_repair_policy_stops_after_two_attempts(self):
        diagnosis = diagnose_log("AssertionError: expected result")
        self.assertEqual(next_repair_action(diagnosis, 0), "request_targeted_patch")
        self.assertEqual(next_repair_action(diagnosis, 2), "stop_and_report")
        self.assertEqual(next_repair_action(diagnose_log("all checks passed"), 0), "review_manually")

    def test_rejects_invalid_attempt_counts(self):
        diagnosis = diagnose_log("FAILED (failures=1)")
        with self.assertRaises(ValueError):
            next_repair_action(diagnosis, -1)
        with self.assertRaises(ValueError):
            next_repair_action(diagnosis, 0, max_attempts=3)

    def test_limits_large_logs(self):
        diagnosis = diagnose_log("x" * 250_000)
        self.assertTrue(diagnosis.truncated)


if __name__ == "__main__":
    unittest.main()
