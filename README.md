# CODEFORGE AI

An Android-controlled, GitHub-first autonomous coding agent.

## Current milestone

The repository has a deterministic task-planning core and a dependency-free adapter for the Gemini generateContent API. Adapter tests use mocked HTTP responses, so CI never needs a real API key. The agent does not yet edit external repositories or autonomously execute generated code.

## Architecture

- **Android client (planned):** submit tasks, follow progress, inspect results.
- **Agent core:** task validation, planning, bounded tool execution, project memory, and repair loop.
- **Model adapter:** server-side Gemini API client in `app/model.py`.
- **GitHub adapter (planned):** inspect files, work on a branch, commit changes, and read Actions logs.
- **CI gate:** compile Python and run unit tests after every push and pull request.

## Free model configuration

Gemini API offers a free tier for supported models, subject to changing model-specific quotas and rate limits. Do not enable billing if the requirement is strictly $0. A free-tier key can be created in [Google AI Studio](https://aistudio.google.com/apikey).

Configure these as environment variables on a trusted server or deployment runtime:

- `GEMINI_API_KEY` — free-tier API key; store it as a secret, never in source code.
- `GEMINI_MODEL` — optional model name; defaults to `gemini-2.5-flash`.

The Android app must never contain the provider key. Never print the key in logs or expose it to pull-request code. The current CI intentionally does not call the live model, and adding a repository secret is a separate settings step. Free-tier quotas may change or be exhausted; the client must handle provider errors without exposing secrets.

## Safety defaults

- Never put API keys in source code, logs, or Android app resources.
- Work on a dedicated branch; do not push generated changes directly to the target project's main branch.
- Limit repair attempts and require successful checks before proposing changes for merge.
- Treat repository contents and build logs as untrusted input.
- Do not execute model-generated shell commands without explicit allowlisting and isolation.

## Development

Python 3.12 is used for the core.

```bash
python -m compileall -q app tests
python -m unittest discover -s tests -v
```

See the GitHub Actions **CODEFORGE AI CI** workflow for automated validation.
