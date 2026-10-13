# CODEFORGE AI

An Android-controlled, GitHub-first autonomous coding agent.

## Current milestone

The repository has a deterministic task-planning core and a dependency-free adapter for the OpenAI Responses API. Adapter tests use mocked HTTP responses, so CI never needs a real API key. The agent does not yet edit external repositories or autonomously execute generated code.

## Architecture

- **Android client (planned):** submit tasks, follow progress, inspect results.
- **Agent core:** task validation, planning, bounded tool execution, project memory, and repair loop.
- **Model adapter:** server-side OpenAI Responses API client in `app/model.py`.
- **GitHub adapter (planned):** inspect files, work on a branch, commit changes, and read Actions logs.
- **CI gate:** compile Python and run unit tests after every push and pull request.

## Model configuration

Configure these as environment variables on a trusted server or deployment runtime:

- `OPENAI_API_KEY` — required API key; store it as a secret, never in source code.
- `OPENAI_MODEL` — optional model name; defaults to `gpt-4.1-mini`.
- `OPENAI_BASE_URL` — optional API endpoint override.

The Android app must never contain the provider key. A GitHub Actions secret is available only to workflows that explicitly receive it; do not expose it to pull-request code or print it in logs. The current CI intentionally does not call the live model, and adding a repository secret is a separate manual settings step.

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
