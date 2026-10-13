"""Unit tests for the Gemini adapter; no live API key is needed."""
import io
import json
import unittest
from unittest.mock import patch

from app.model import (
    GeminiClient,
    ModelConfigurationError,
    ModelProviderError,
    _extract_text,
)


class ModelAdapterTests(unittest.TestCase):
    def test_requires_api_key(self):
        client = GeminiClient(api_key="", model="test-model")
        with self.assertRaises(ModelConfigurationError):
            client.generate("Write a function")

    def test_rejects_empty_prompt(self):
        client = GeminiClient(api_key="test-key", model="test-model")
        with self.assertRaises(ValueError):
            client.generate("  ")

    def test_extracts_generated_text(self):
        payload = {
            "candidates": [{
                "content": {"parts": [
                    {"text": "Implement "},
                    {"text": "the function."},
                ]}
            }]
        }
        self.assertEqual(_extract_text(payload), "Implement \nthe function.")

    def test_rejects_response_without_text(self):
        with self.assertRaises(ModelProviderError):
            _extract_text({"candidates": []})

    @patch("app.model.urlopen")
    def test_sends_model_prompt_and_returns_text(self, mocked_urlopen):
        mocked_urlopen.return_value = io.BytesIO(
            json.dumps({
                "candidates": [{
                    "content": {"parts": [{"text": "Implement the function."}]}
                }]
            }).encode("utf-8")
        )
        client = GeminiClient(
            api_key="test-key",
            model="test-model",
            base_url="https://example.invalid/v1beta",
        )
        result = client.generate("Create a function", instructions="Be concise")
        self.assertEqual(result, "Implement the function.")

        request = mocked_urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            body["contents"][0]["parts"][0]["text"], "Create a function"
        )
        self.assertEqual(
            body["systemInstruction"]["parts"][0]["text"], "Be concise"
        )
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key")
        self.assertIn("/models/test-model:generateContent", request.full_url)
        mocked_urlopen.assert_called_once()

    @patch("app.model.urlopen")
    def test_provider_failure_does_not_expose_response_body(self, mocked_urlopen):
        from urllib.error import HTTPError

        mocked_urlopen.side_effect = HTTPError(
            "https://example.invalid", 401, "Unauthorized", {}, io.BytesIO(b"secret details")
        )
        client = GeminiClient(api_key="test-key", model="test-model")
        with self.assertRaises(ModelProviderError) as caught:
            client.generate("Hello")
        self.assertIn("HTTP 401", str(caught.exception))
        self.assertNotIn("secret details", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
