"""End-to-end orchestration and safe CI log diagnosis tests."""
import json
import unittest
from unittest.mock import Mock

from app.core import TaskKind, TaskRequest
from app.diagnostics import diagnose_log, next_repair_action, redact_secrets
from app.memory import ProjectMemory
from app.workflow import ControlledAgentWorkflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.reader = Mock()
        self.writer = Mock()
        self.memory = ProjectMemory(":memory:")
        self.reader.get_tree.return_value = {
            "sha": "tree-sha", "truncated": False,
            "tree": [{"path": "app/existing.py", "type": "blob", "sha": "blob-sha"}],
        }
        self.reader.list_workflow_runs.return_value = [
            {"run_number": 12, "status": "completed", "conclusion": "success"}
        ]
        self.reader.fetch_job_logs.return_value = "AssertionError: expected result"
        self.workflow = ControlledAgentWorkflow(
            "owner/repo", token="test-token", reader=self.reader, writer=self.writer,
            memory=self.memory,
        )

    def tearDown(self):
        self.memory.close()

    def test_inspection_integrates_plan_tree_and_actions(self):
        result = self.workflow.inspect(TaskRequest("Fix the issue", "owner/repo", TaskKind.FIX))
        self.assertEqual(result.status, "inspected")
        self.assertEqual(result.details["tree"]["tree"][0]["path"], "app/existing.py")
        self.assertEqual(result.details["recent_runs"][0]["conclusion"], "success")
        memories = self.memory.recall("owner/repo")
        self.assertEqual(memories[0]["category"], "workflow_event")
        self.assertNotIn("Fix the issue", memories[0]["content"])

    def test_explicit_memory_rejects_secrets_and_can_be_recalled(self):
        saved = self.workflow.remember_fact("decision", "Use Python 3.12 for the core.")
        self.assertEqual(saved.status, "memory_saved")
        recalled = self.workflow.recall_memory(query="Python")
        self.assertEqual(len(recalled.details["items"]), 1)
        with self.assertRaises(ValueError):
            self.workflow.remember_fact("decision", "token=ghp_abcdefghijklmnopqrstuvwxyz123456")

    def test_invalid_patch_never_calls_writer(self):
        result = self.workflow.prepare_branch(json.dumps({"changes": [
            {"path": "../bad.py", "action": "create", "content": "x"}
        ]}))
        self.assertEqual(result.status, "rejected")
        self.writer.publish_patch.assert_not_called()

    def test_valid_patch_calls_writer_only_after_validation_and_records_branch(self):
        self.writer.publish_patch.return_value = {
            "branch": "codeforge/fix-abc1234", "commit_sha": "commit-sha",
            "changed_files": ["app/new.py"], "checks_run": False, "base_branch": "main",
        }
        patch = json.dumps({"changes": [
            {"path": "app/new.py", "action": "create", "content": "print('ok')\n"}
        ]})
        result = self.workflow.prepare_branch(patch, description="fix")
        self.assertEqual(result.status, "branch_prepared")
        self.writer.publish_patch.assert_called_once_with(
            patch, base_branch="main", description="fix"
        )
        memories = self.memory.recall("owner/repo")
        self.assertIn("branch_prepared", memories[0]["content"])

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


    def test_model_proposal_receives_project_memory_and_validates_without_writing(self):
        model = Mock()
        model.generate.return_value = json.dumps({"changes": [
            {"path": "app/existing.py", "action": "update", "content": "VALUE = 2\\n"}
        ]})
        self.workflow.model = model
        self.workflow.remember_fact("decision", "Keep the core dependency-free.")
        result = self.workflow.propose_patch(
            TaskRequest("Update the existing value", "owner/repo", TaskKind.FIX)
        )
        self.assertEqual(result.status, "patch_proposed")
        self.assertEqual(result.details["memory_items_used"], 1)
        self.assertFalse(result.details["written_to_github"])
        self.assertFalse(result.details["checks_run"])
        prompt = model.generate.call_args.args[0]
        self.assertIn("Keep the core dependency-free.", prompt)
        self.assertIn("app/existing.py", prompt)
        self.writer.publish_patch.assert_not_called()

    def test_model_proposal_rejects_unsafe_path_before_any_write(self):
        model = Mock()
        model.generate.return_value = json.dumps({"changes": [
            {"path": "../escape.py", "action": "create", "content": "x\\n"}
        ]})
        self.workflow.model = model
        result = self.workflow.propose_patch(
            TaskRequest("Create a file", "owner/repo", TaskKind.CREATE)
        )
        self.assertEqual(result.status, "rejected")
        self.writer.publish_patch.assert_not_called()

    def test_model_proposal_rejects_malformed_output(self):
        model = Mock()
        model.generate.return_value = "Here is the code: print('hello')"
        self.workflow.model = model
        result = self.workflow.propose_patch(
            TaskRequest("Create a file", "owner/repo", TaskKind.CREATE)
        )
        self.assertEqual(result.status, "rejected")
        self.writer.publish_patch.assert_not_called()


    def test_memory_can_be_deleted_only_from_own_project(self):
        item = self.workflow.remember_fact("preference", "Prefer short summaries.")
        memory_id = item.details["item"]["id"]
        self.assertEqual(self.workflow.forget_memory(memory_id).status, "memory_deleted")
        self.assertEqual(self.workflow.forget_memory(memory_id).status, "memory_not_found")
        self.workflow.remember_fact("preference", "Another project-specific preference.")
        self.assertEqual(self.memory.recall("other/repo"), [])

    def test_propose_and_prepare_commits_only_after_valid_model_proposal(self):
        model = Mock()
        model.generate.return_value = json.dumps({"changes": [
            {"path": "app/existing.py", "action": "update", "content": "VALUE = 3\\n"}
        ]})
        self.workflow.model = model
        self.writer.publish_patch.return_value = {
            "branch": "codeforge/ai-proposed-abc1234", "commit_sha": "commit-sha",
            "changed_files": ["app/existing.py"], "checks_run": False, "base_branch": "main",
        }
        result = self.workflow.propose_and_prepare_branch(
            TaskRequest("Update value", "owner/repo", TaskKind.FIX)
        )
        self.assertEqual(result.status, "branch_prepared")
        self.assertFalse(result.details["pull_request_created"])
        self.assertFalse(result.details["checks_run"])
        self.writer.publish_patch.assert_called_once()
        self.assertEqual(result.details["proposal"]["memory_items_used"], 0)

    def test_propose_and_prepare_does_not_write_when_model_output_is_invalid(self):
        model = Mock()
        model.generate.return_value = "not JSON"
        self.workflow.model = model
        result = self.workflow.propose_and_prepare_branch(
            TaskRequest("Update value", "owner/repo", TaskKind.FIX)
        )
        self.assertEqual(result.status, "rejected")
        self.writer.publish_patch.assert_not_called()

    def test_inspect_run_returns_run_and_job_statuses(self):
        self.reader.get_workflow_run.return_value = {
            "id": 456, "status": "completed", "conclusion": "success",
        }
        self.reader.list_run_jobs.return_value = [{"id": 789, "conclusion": "success"}]
        result = self.workflow.inspect_run(456)
        self.assertEqual(result.status, "run_inspected")
        self.assertEqual(result.details["jobs"][0]["id"], 789)

    def test_diagnose_run_uses_failed_job_and_stops_after_two_attempts(self):
        self.reader.get_workflow_run.return_value = {
            "id": 456, "status": "completed", "conclusion": "failure",
        }
        self.reader.list_run_jobs.return_value = [{"id": 789, "conclusion": "failure"}]
        result = self.workflow.diagnose_run(456, attempts_used=2)
        self.assertEqual(result.status, "diagnosed")
        self.assertEqual(result.details["next_action"], "stop_and_report")
        self.reader.fetch_job_logs.assert_called_once_with(789)


    def test_create_pr_requires_isolated_branch_and_creates_draft_only(self):
        rejected = self.workflow.create_pr_from_branch("main")
        self.assertEqual(rejected.status, "rejected")
        self.writer.create_pull_request.assert_not_called()
        self.writer.create_pull_request.return_value = {
            "number": 17, "url": "https://github.com/owner/repo/pull/17",
            "draft": True, "merged": False,
        }
        created = self.workflow.create_pr_from_branch("codeforge/fix-abc1234")
        self.assertEqual(created.status, "pull_request_created")
        self.assertTrue(created.details["draft"])
        self.assertFalse(created.details["merge_performed"])
        self.writer.create_pull_request.assert_called_once_with(
            "codeforge/fix-abc1234", base_branch="main",
            title="CODEFORGE AI proposed change",
            body="Validated isolated branch. Review all changes and CI results before merging.",
            draft=True,
        )

    def test_inspect_pr_returns_linked_run_jobs_without_mutations(self):
        self.reader.get_pull_request.return_value = {
            "number": 7, "state": "open", "draft": True, "head_sha": "abc",
        }
        self.reader.list_pull_request_runs.return_value = [
            {"id": 456, "status": "completed", "conclusion": "failure"},
        ]
        self.reader.list_run_jobs.return_value = [{"id": 789, "conclusion": "failure"}]
        result = self.workflow.inspect_pull_request(7)
        self.assertEqual(result.status, "pull_request_inspected")
        self.assertEqual(result.details["runs"][0]["jobs"][0]["id"], 789)
        self.assertFalse(result.details["merge_performed"])

    def test_repair_proposal_is_validated_and_does_not_publish_or_execute(self):
        self.reader.get_workflow_run.return_value = {
            "id": 456, "status": "completed", "conclusion": "failure",
        }
        self.reader.list_run_jobs.return_value = [{"id": 789, "conclusion": "failure"}]
        model = Mock()
        model.generate.return_value = json.dumps({"changes": [
            {"path": "app/existing.py", "action": "update", "content": "VALUE = 4\\n"}
        ]})
        self.workflow.model = model
        result = self.workflow.propose_repair_patch(456, "main", attempts_used=0)
        self.assertEqual(result.status, "repair_patch_proposed")
        self.assertEqual(result.details["attempt_number"], 1)
        self.assertEqual(result.details["next_action"], "publish_isolated_branch")
        self.writer.publish_patch.assert_not_called()
        self.assertIn("untrusted data", model.generate.call_args.args[0])
        self.assertIn("AssertionError", model.generate.call_args.args[0])

    def test_repair_policy_stops_before_model_after_two_attempts(self):
        self.reader.get_workflow_run.return_value = {
            "id": 456, "status": "completed", "conclusion": "failure",
        }
        self.reader.list_run_jobs.return_value = [{"id": 789, "conclusion": "failure"}]
        model = Mock()
        self.workflow.model = model
        result = self.workflow.propose_repair_patch(456, "main", attempts_used=2)
        self.assertEqual(result.status, "repair_stopped")
        self.assertEqual(result.details["next_action"], "stop_and_report")
        model.generate.assert_not_called()
        self.writer.publish_patch.assert_not_called()

    def test_repair_proposal_is_not_generated_for_successful_run(self):
        self.reader.get_workflow_run.return_value = {
            "id": 456, "status": "completed", "conclusion": "success",
        }
        model = Mock()
        self.workflow.model = model
        result = self.workflow.propose_repair_patch(456, "main", attempts_used=0)
        self.assertEqual(result.status, "run_not_failed")
        model.generate.assert_not_called()

    def test_repair_attempt_count_cannot_exceed_two(self):
        with self.assertRaises(ValueError):
            self.workflow.propose_repair_patch(456, "main", attempts_used=3)


    def test_repair_cycle_start_is_idempotent_and_requires_exact_source_sha(self):
        sha = "a" * 40
        self.reader.get_pull_request.return_value = {
            "number": 7, "state": "open", "merged": False, "head_sha": sha,
        }
        self.reader.get_workflow_run.return_value = {
            "id": 456, "status": "completed", "conclusion": "failure", "head_sha": sha,
        }
        first = self.workflow.start_repair_cycle(7, 456, sha)
        again = self.workflow.start_repair_cycle(7, 456, sha)
        self.assertEqual(first.status, "repair_cycle_started")
        self.assertEqual(again.status, "repair_cycle_recovered")
        self.assertEqual(first.details["cycle"]["id"], again.details["cycle"]["id"])
        self.reader.get_workflow_run.return_value["head_sha"] = "b" * 40
        mismatch = self.workflow.start_repair_cycle(7, 456, sha)
        self.assertEqual(mismatch.status, "rejected")

    def test_repair_attempt_requires_run_sha_match_and_duplicate_is_not_counted_twice(self):
        cycle = self.memory.create_repair_cycle("owner/repo", 7, 456, "a" * 40)
        self.reader.get_workflow_run.return_value = {
            "id": 500, "status": "queued", "conclusion": None, "head_sha": "b" * 40,
        }
        mismatch = self.workflow.record_repair_attempt(
            cycle["id"], "codeforge/fix-one", "c" * 40, 500
        )
        self.assertEqual(mismatch.status, "rejected")
        self.reader.get_workflow_run.return_value["head_sha"] = "c" * 40
        recorded = self.workflow.record_repair_attempt(
            cycle["id"], "codeforge/fix-one", "c" * 40, 500
        )
        duplicate = self.workflow.record_repair_attempt(
            cycle["id"], "codeforge/fix-one", "c" * 40, 500
        )
        self.assertEqual(recorded.details["cycle"]["attempts_used"], 1)
        self.assertEqual(duplicate.details["cycle"]["attempts_used"], 1)

    def test_advance_cycle_completes_only_on_exact_sha_success_and_is_terminal(self):
        cycle = self.memory.create_repair_cycle("owner/repo", 7, 456, "a" * 40)
        self.reader.get_workflow_run.return_value = {
            "id": 456, "status": "completed", "conclusion": "success", "head_sha": "b" * 40,
        }
        mismatch = self.workflow.advance_repair_cycle(cycle["id"])
        self.assertEqual(mismatch.status, "rejected")
        self.reader.get_workflow_run.return_value["head_sha"] = "a" * 40
        completed = self.workflow.advance_repair_cycle(cycle["id"])
        self.assertEqual(completed.status, "completed")
        repeated = self.workflow.advance_repair_cycle(cycle["id"])
        self.assertEqual(repeated.status, "completed")
        self.assertTrue(repeated.details["duplicate_or_terminal"])

    def test_advance_cycle_stops_after_two_failed_repair_attempts(self):
        cycle = self.memory.create_repair_cycle("owner/repo", 7, 456, "a" * 40)
        self.memory.record_repair_attempt(
            "owner/repo", cycle["id"], "codeforge/fix-one", "b" * 40, 500
        )
        self.memory.record_repair_attempt(
            "owner/repo", cycle["id"], "codeforge/fix-two", "c" * 40, 501
        )
        self.reader.get_workflow_run.return_value = {
            "id": 501, "status": "completed", "conclusion": "failure", "head_sha": "c" * 40,
        }
        stopped = self.workflow.advance_repair_cycle(cycle["id"])
        self.assertEqual(stopped.status, "stop_and_report")
        self.assertEqual(stopped.details["cycle"]["attempts_used"], 2)
        repeated = self.workflow.advance_repair_cycle(cycle["id"])
        self.assertEqual(repeated.status, "stop_and_report")
        self.assertTrue(repeated.details["duplicate_or_terminal"])

    def test_advance_cycle_waits_for_exact_sha_run_to_finish(self):
        cycle = self.memory.create_repair_cycle("owner/repo", 7, 456, "a" * 40)
        self.reader.get_workflow_run.return_value = {
            "id": 456, "status": "in_progress", "conclusion": None, "head_sha": "a" * 40,
        }
        result = self.workflow.advance_repair_cycle(cycle["id"])
        self.assertEqual(result.status, "awaiting_ci")
        self.assertEqual(result.details["checked_sha"], "a" * 40)


    def test_publish_repair_revalidates_and_creates_only_isolated_draft_pr(self):
        source_sha = "a" * 40
        cycle = self.memory.create_repair_cycle("owner/repo", 7, 456, source_sha)
        self.reader.get_pull_request.return_value = {
            "number": 7, "state": "open", "merged": False, "head_branch": "codeforge/original",
            "head_sha": source_sha,
        }
        self.writer.publish_patch.return_value = {
            "branch": "codeforge/repair-one", "commit_sha": "b" * 40,
            "changed_files": ["app/existing.py"], "checks_run": False, "base_branch": "codeforge/original",
        }
        self.writer.create_pull_request.return_value = {
            "number": 8, "url": "https://github.com/owner/repo/pull/8", "draft": True,
            "merged": False,
        }
        patch = json.dumps({"changes": [
            {"path": "app/existing.py", "action": "update", "content": "VALUE = 9\\n"}
        ]})
        result = self.workflow.publish_repair_patch(cycle["id"], patch)
        self.assertEqual(result.status, "repair_published")
        self.assertEqual(result.details["commit_sha"], "b" * 40)
        self.assertEqual(result.details["cycle"]["attempts_used"], 1)
        self.writer.publish_patch.assert_called_once()
        self.writer.create_pull_request.assert_called_once()
        self.assertEqual(self.writer.create_pull_request.call_args.args[0], "codeforge/repair-one")
        self.assertEqual(self.writer.create_pull_request.call_args.kwargs["base_branch"], "codeforge/original")
        self.assertTrue(self.writer.create_pull_request.call_args.kwargs["draft"])
        repeated = self.workflow.publish_repair_patch(cycle["id"], patch)
        self.assertEqual(repeated.status, "rejected")
        self.assertEqual(self.writer.publish_patch.call_count, 1)

    def test_publish_repair_rejects_workflow_path_before_any_write(self):
        cycle = self.memory.create_repair_cycle("owner/repo", 7, 456, "a" * 40)
        self.reader.get_pull_request.return_value = {
            "number": 7, "state": "open", "merged": False, "head_branch": "codeforge/original",
            "head_sha": "a" * 40,
        }
        patch = json.dumps({"changes": [
            {"path": ".github/workflows/ci.yml", "action": "update", "content": "unsafe"}
        ]})
        result = self.workflow.publish_repair_patch(cycle["id"], patch)
        self.assertEqual(result.status, "rejected")
        self.writer.publish_patch.assert_not_called()
        self.writer.create_pull_request.assert_not_called()

    def test_advance_cycle_discovers_ci_run_by_exact_repair_sha(self):
        cycle = self.memory.create_repair_cycle("owner/repo", 7, 456, "a" * 40)
        self.memory.record_repair_attempt(
            "owner/repo", cycle["id"], "codeforge/fix-one", "b" * 40, run_id=None
        )
        self.reader.list_workflow_runs_for_sha.return_value = [
            {"id": 501, "head_sha": "c" * 40, "status": "completed", "conclusion": "success"},
            {"id": 502, "head_sha": "b" * 40, "status": "completed", "conclusion": "success"},
        ]
        self.reader.get_workflow_run.return_value = {
            "id": 502, "head_sha": "b" * 40, "status": "completed", "conclusion": "success",
        }
        result = self.workflow.advance_repair_cycle(cycle["id"])
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.details["checked_sha"], "b" * 40)
        saved = self.memory.get_repair_cycle("owner/repo", cycle["id"])
        self.assertEqual(saved["current_run_id"], 502)


class DiagnosticsTests(unittest.TestCase):
    def test_detects_failure_and_redacts_tokens(self):
        diagnosis = diagnose_log(
            "Traceback (most recent call last):\n"
            "token=ghp_abcdefghijklmnopqrstuvwxyz123456\n"
            "ModuleNotFoundError: No module named 'missing'\n"
        )
        self.assertEqual(diagnosis.status, "failure_signals")
        self.assertIn("python_traceback", diagnosis.findings)
        self.assertIn("python_import_error", diagnosis.findings)
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz123456", "\n".join(diagnosis.excerpts))
        self.assertIn("[REDACTED]", redact_secrets("token=ghp_abcdefghijklmnopqrstuvwxyz123456"))

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
