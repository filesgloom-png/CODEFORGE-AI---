"""Dependency-free client for Google's free-tier Gemini API.

The API key is read from the server-side environment. Never send it from an
Android client or commit it to the repository.
"""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class ModelConfigurationError(RuntimeError):
    """Raised when required model configuration is missing."""


class ModelProviderError(RuntimeError):
    """Raised when the configured model provider cannot answer safely."""


def _extract_text(payload: dict[str, Any]) -> str:
    """Extract generated text from a Gemini generateContent response."""
    candidates = payload.get("candidates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content", {})
            if not isinstance(content, dict):
                continue
            parts = content.get("parts", [])
            if not isinstance(parts, list):
                continue
            texts = [
                part["text"]
                for part in parts
                if isinstance(part, dict)
                and isinstance(part.get("text"), str)
                and part["text"].strip()
            ]
            if texts:
                return "\n".join(texts).strip()
    raise ModelProviderError("The model response contained no text output.")


class GeminiClient:
    """Minimal client for Gemini API; free-tier usage is subject to provider limits."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self._model = (model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
        self._base_url = (
            base_url or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self._timeout = timeout

    def generate(self, prompt: str, instructions: str | None = None) -> str:
        """Send a prompt to Gemini and return generated text."""
        if not self._api_key:
            raise ModelConfigurationError(
                "GEMINI_API_KEY is missing. Configure a free-tier provider key as a server-side secret."
            )
        if not self._model:
            raise ModelConfigurationError("GEMINI_MODEL must not be empty.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt.strip()}]}],
            "generationConfig": {"temperature": 0.2},
        }
        if instructions and instructions.strip():
            body["systemInstruction"] = {"parts": [{"text": instructions.strip()}]}

        endpoint = f"{self._base_url}/models/{quote(self._model, safe='')}:generateContent"
        request = Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "x-goog-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            # Do not include response bodies or request headers in exceptions.
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
