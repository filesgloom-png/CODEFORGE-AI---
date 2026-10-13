"""Tests for the initial CODEFORGE AI task-planning core."""
import unittest

from app.core import (
    TaskKind,
    TaskRequest,
    TaskValidationError,
    plan_task,
    validate_task,
)


class ValidateTaskTests(unittest.TestCase):
    def test_trims_description_and_repository(self):
        result = validate_task(TaskRequest("  Add tests  ", " filesgloom-png/demo "))
        self.assertEqual(result.description, "Add tests")
        self.assertEqual(result.repository, "filesgloom-png/demo")

    def test_rejects_empty_description(self):
        with self.assertRaises(TaskValidationError):
            validate_task(TaskRequest("   "))

    def test_rejects_oversized_description(self):
        with self.assertRaises(TaskValidationError):
            validate_task(TaskRequest("x" * 4001))

    def test_rejects_invalid_repository_format(self):
        with self.assertRaises(TaskValidationError):
            validate_task(TaskRequest("Fix bug", "not-a-repository"))


class PlanTaskTests(unittest.TestCase):
    def test_fix_plan_includes_diagnosis_and_validation(self):
        plan = plan_task(TaskRequest("Fix failing Android build", kind=TaskKind.FIX))
        self.assertTrue(any("failure" in step for step in plan.steps))
        self.assertTrue(any("checks" in step for step in plan.steps))

    def test_explain_plan_does_not_request_file_modification(self):
        plan = plan_task(TaskRequest("Explain this function", kind=TaskKind.EXPLAIN))
        self.assertTrue(any("without modifying files" in step for step in plan.steps))

    def test_create_plan_requires_repository(self):
        plan = plan_task(TaskRequest("Create a small API", kind=TaskKind.CREATE))
        self.assertTrue(plan.requires_repository)


if __name__ == "__main__":
    unittest.main()
