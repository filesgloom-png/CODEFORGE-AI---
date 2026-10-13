# CODEFORGE AI

An Android-controlled, GitHub-first coding agent foundation.

## Current milestone

The Python core now includes task planning, a read-only GitHub REST adapter, bounded patch validation, isolated-branch patch publishing, a controlled workflow, and safe CI-log diagnosis. CI tests use mocked HTTP; no live provider or GitHub write token is required to pass tests.

The system does not execute model-generated shell commands or project code. The workflow can inspect and validate patches, and the writer can publish a validated patch to a dedicated branch when explicitly invoked with a server-side GitHub token. Creating a branch does not mean the project build was run or the patch is safe to merge.

## Components

- `app/core.py`: deterministic task validation and planning.
- `app/model.py`: dependency-free Gemini API adapter (live API use requires a key and may be subject to free-tier quotas).
- `app/github_adapter.py`: bounded read-only repository, file and GitHub Actions metadata access.
- `app/patches.py`: strict JSON patch validation and path/size safeguards.
- `app/github_writer.py`: writes a validated patch to a dedicated branch; never merges into the target branch.
- `app/workflow.py`: orchestrates inspection, validation and branch preparation.
- `app/diagnostics.py`: redacts common secrets and detects known CI failure signatures from bounded text logs.

## Configuration

For public repositories, read-only inspection can work without a token, subject to GitHub rate limits. Branch writing requires a server-side `GITHUB_TOKEN` with repository contents write permission. Never put tokens in source code, client apps, logs, or model prompts. Prefer a fine-grained token limited to the intended test repository.

The Gemini adapter is optional and is not called by CI. If the requirement is strictly $0, do not enable billing; provider free-tier availability and quotas may change.

## Safety boundaries

- Generated patches are strict JSON data, not shell commands.
- Repository tree truncation blocks patch validation.
- Patch writes go to a dedicated branch; no automatic merge is performed.
- CI log diagnosis treats logs as untrusted text and redacts common token patterns.
- Repair policy emits a decision label only; it does not execute commands. At most two repair attempts are recommended before stopping.
- A branch being created is not proof of a successful project build. Build checks and human review are still required.

## Development

Python 3.12:

```bash
python -m compileall -q app tests
python -m unittest discover -s tests -v
```

See GitHub Actions **CODEFORGE AI CI** for validation.
