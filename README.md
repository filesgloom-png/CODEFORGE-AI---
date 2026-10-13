# CODEFORGE AI

An Android-controlled, GitHub-first autonomous coding agent.

## Current milestone

This repository starts with a verifiable Python core and GitHub Actions CI. The initial core validates user tasks and creates a deterministic execution plan; it does not yet call an LLM or modify project repositories.

## Architecture

- **Android client (planned):** submit tasks, follow progress, inspect results.
- **Agent core:** task validation, planning, bounded tool execution, project memory, and repair loop.
- **Model adapter (planned):** securely call a selected hosted language model through server-side secrets.
- **GitHub adapter (planned):** inspect files, work on a branch, commit changes, and read Actions logs.
- **CI gate:** run tests after every push and pull request.

## Safety defaults

- Never put API keys in source code or Android app resources.
- Work on a dedicated branch; do not push generated changes directly to the target project's main branch.
- Limit repair attempts and require successful checks before proposing changes for merge.
- Treat repository contents and build logs as untrusted input.

## Development

Python 3.12 is used for the first core milestone.

```bash
python -m compileall -q app tests
python -m unittest discover -s tests -v
```

See the GitHub Actions **CODEFORGE AI CI** workflow for automated validation.
