"""Safe environment configuration status tests."""
import unittest

from app.config import configuration_status


class ConfigurationStatusTests(unittest.TestCase):
    def test_reports_presence_without_disclosing_values(self):
        status = configuration_status({
            "GITHUB_TOKEN": "ghp_secret_should_never_appear",
            "GEMINI_API_KEY": "private-model-key",
        })
        serialized = repr(status)
        self.assertTrue(status["github_token_configured"])
        self.assertTrue(status["gemini_api_key_configured"])
        self.assertTrue(status["github_write_ready"])
        self.assertTrue(status["model_ready"])
        self.assertFalse(status["secrets_exposed"])
        self.assertNotIn("ghp_secret_should_never_appear", serialized)
        self.assertNotIn("private-model-key", serialized)

    def test_reports_missing_secrets_without_failing(self):
        status = configuration_status({})
        self.assertFalse(status["github_token_configured"])
        self.assertFalse(status["gemini_api_key_configured"])
        self.assertFalse(status["github_write_ready"])
        self.assertFalse(status["model_ready"])


if __name__ == "__main__":
    unittest.main()
