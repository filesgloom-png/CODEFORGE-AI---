"""Unit tests for the hosted model adapter; no live API key is needed."""
import io
import json
import unittest
from unittest.mock import patch

from app.model import (
    ModelConfigurationError,
    ModelProviderError,
    OpenAIResponsesClient,
    _extract_text,
)


class ModelAdapterTests(unittest.TestCase):
    def test_requires_api_key(self):
        client = OpenAIResponsesClient(api_key="", model="test-model")
        with self.assertRaises(ModelConfigurationError):
            client.generate("Write a function")

    def test_rejects_empty_prompt(self):
        client = OpenAIResponsesClient(api_key="test-key", model="test-model")
        with self.assertRaises(ValueError):
            client.generate("  ")

    def test_extracts_top_level_output_text(self):
        self.assertEqual(_extract_text({"output_text": "  Hello  "}), "Hello")

    def test_extracts_text_from_responses_output_array(self):
        payload = {
            "output": [
                {"type": "message", "content": [
                    {"type": "output_text", "text": "First"},
                    {"type": "output_text", "text": "Second"},
                ]}
            ]
        }
        self.assertEqual(_extract_text(payload), "First\nSecond")

    def test_rejects_response_without_text(self):
        with self.assertRaises(ModelProviderError):
            _extract_text({"output": []})

    @patch("app.model.urlopen")
    def test_sends_model_prompt_and_returns_text(self, mocked_urlopen):
        mocked_urlopen.return_value = io.BytesIO(
            json.dumps({"output_text": "Implement the function."}).encode("utf-8")
        )
        client = OpenAIResponsesClient(
            api_key="test-key",
            model="test-model",
            base_url="https://example.invalid/v1/responses",
        )
        result = client.generate("Create a function", instructions="Be concise")
        self.assertEqual(result, "Implement the function.")

        request = mocked_urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["input"], "Create a function")
        self.assertEqual(body["instructions"], "Be concise")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        mocked_urlopen.assert_called_once()

    @patch("app.model.urlopen")
    def test_provider_failure_does_not_expose_response_body(self, mocked_urlopen):
        from urllib.error import HTTPError

        mocked_urlopen.side_effect = HTTPError(
            "https://example.invalid", 401, "Unauthorized", {}, io.BytesIO(b"secret details")
        )
        client = OpenAIResponsesClient(api_key="test-key", model="test-model")
        with self.assertRaises(ModelProviderError) as caught:
            client.generate("Hello")
        self.assertIn("HTTP 401", str(caught.exception))
        self.assertNotIn("secret details", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
