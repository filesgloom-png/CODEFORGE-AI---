"""Safe configuration status helpers. Never return secret values."""
from __future__ import annotations

import os
from typing import Mapping


def configuration_status(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Report whether required values exist without exposing their contents."""
    values = os.environ if environ is None else environ
    github_token = bool(values.get("GITHUB_TOKEN", "").strip())
    gemini_key = bool(values.get("GEMINI_API_KEY", "").strip())
    return {
        "github_token_configured": github_token,
        "gemini_api_key_configured": gemini_key,
        "github_write_ready": github_token,
        "model_ready": gemini_key,
        "secrets_exposed": False,
        "notes": [
            "Token presence does not verify its permissions.",
            "No secret values are returned or logged.",
            "Model access is optional; do not enable billing if the project must remain free.",
        ],
    }
