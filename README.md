# CODEFORGE AI

An Android-controlled, GitHub-first coding agent foundation.

## Current milestone

The Python core includes task planning, a read-only GitHub REST adapter, bounded patch validation, isolated-branch patch publishing, a controlled workflow, safe CI-log diagnosis, project-scoped SQLite memory, and secret-safe configuration status checks. CI tests use mocked HTTP; no live provider or GitHub write token is required to pass tests.

The memory store persists explicit project preferences, decisions, facts, and workflow events across process restarts when CODEFORGE_MEMORY_DB points to persistent storage. It rejects common credential-like patterns, supports project-scoped recall/deletion, and deliberately does not store raw task prompts. This is durable application memory, not a guarantee that every conversation is automatically remembered; the application must explicitly save useful facts.

The system does not execute model-generated shell commands or project code. The workflow can inspect and validate patches, and the writer can publish a validated patch to a dedicated codeforge/* branch when explicitly invoked with a server-side GitHub token. It never updates the base branch ref or merges automatically. Creating a branch does not mean the project build was run or the patch is safe to merge.

## Components

- `app/core.py`: deterministic task validation and planning.
- `app/model.py`: dependency-free Gemini API adapter (live API use requires a key and may be subject to free-tier quotas).
- `app/config.py`: reports whether GitHub/model credentials are configured without returning their values.
- `app/memory.py`: SQLite project memory with bounded content, project scoping, recall/deletion, and credential-like text rejection.
- `app/github_adapter.py`: bounded read-only repository, file and GitHub Actions metadata access.
- `app/patches.py`: strict JSON patch validation and path/size safeguards.
- `app/github_writer.py`: writes a validated patch to a dedicated branch; never merges into the target branch.
- `app/workflow.py`: orchestrates inspection, memory, validation and branch preparation.
- `app/diagnostics.py`: redacts common secrets and detects known CI failure signatures from bounded text logs.

## Configuration

For public repositories, read-only inspection can work without a token, subject to GitHub rate limits. Branch writing requires a server-side `GITHUB_TOKEN` with repository contents write permission. Never put tokens in source code, client apps, logs, or model prompts. Prefer a fine-grained token limited to the intended test repository.

Memory defaults to `~/.codeforge/memory.sqlite3`; set `CODEFORGE_MEMORY_DB` to choose a database path. The host must provide persistent storage for memory to survive host replacement/redeployment. Do not commit the database to GitHub.

The Gemini adapter is optional and is not called by CI. If the requirement is strictly $0, do not enable billing; provider free-tier availability and quotas may change.

## Safety boundaries

- Generated patches are strict JSON data, not shell commands.
- Repository tree truncation blocks patch validation.
- Patch writes go to a dedicated branch; no automatic merge is performed.
- CI log diagnosis treats logs as untrusted text and redacts common token patterns.
- Memory is explicit, project-scoped, size-limited, and rejects common secret patterns; it is not a substitute for a secure secrets manager.
- Repair policy emits a decision label only; it does not execute commands. At most two repair attempts are recommended before stopping.
- A branch being created is not proof of a successful project build. Build checks and human review are still required.

## Supported languages

The agent core itself is Python. The intended coding agent can work across languages when the model can generate them and the repository has suitable tests/build tooling. Initial CI validates only this Python core; project-specific language support and build checks must be added deliberately rather than assumed.

## Development

Python 3.12:

```bash
python -m compileall -q app tests
python -m unittest discover -s tests -v
```

See GitHub Actions **CODEFORGE AI CI** for validation.
