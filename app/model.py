"""Server-side adapter for the OpenAI Responses API.

The API key is read from the process environment. Never pass it from the
Android client or commit it to the repository.
"""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelConfigurationError(RuntimeError):
    """Raised when required model configuration is missing."""


class ModelProviderError(RuntimeError):
    """Raised when the configured model provider cannot answer safely."""


def _extract_text(payload: dict[str, Any]) -> str:
    """Extract generated text from a Responses API JSON payload."""
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    output = payload.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") in {"output_text", "text"}
                    and isinstance(block.get("text"), str)
                ):
                    parts.append(block["text"])
    text = "\n".join(part for part in parts if part.strip()).strip()
    if not text:
        raise ModelProviderError("The model response contained no text output.")
    return text


class OpenAIResponsesClient:
    """Minimal dependency-free client for the OpenAI Responses API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        self._model = (model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")).strip()
        self._base_url = (
            base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/responses")
        ).rstrip("/")
        self._timeout = timeout

    def generate(self, prompt: str, instructions: str | None = None) -> str:
        """Send a prompt to the configured model and return its text response."""
        if not self._api_key:
            raise ModelConfigurationError(
                "OPENAI_API_KEY is missing. Configure it as a server-side secret."
            )
        if not self._model:
            raise ModelConfigurationError("OPENAI_MODEL must not be empty.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        body: dict[str, Any] = {
            "model": self._model,
            "input": prompt.strip(),
        }
        if instructions and instructions.strip():
            body["instructions"] = instructions.strip()

        request = Request(
            self._base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            # Do not include response bodies or request headers in exceptions:
            # provider responses can contain sensitive diagnostic information.
            raise ModelProviderError(
                f"Model provider returned HTTP {exc.code}."
            ) from None
        except (URLError, TimeoutError, OSError):
            raise ModelProviderError(
                "Could not reach the model provider. Check network access and try again."
            ) from None

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ModelProviderError("The model provider returned invalid JSON.") from None
        if not isinstance(payload, dict):
            raise ModelProviderError("The model provider returned an unexpected response.")
        return _extract_text(payload)
