"""Bounded, secret-redacting diagnostics for untrusted CI text."""
from __future__ import annotations

import re
from dataclasses import dataclass


MAX_LOG_BYTES = 200_000
MAX_LOG_LINES = 2_000
MAX_FINDINGS = 30
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|password|secret)\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
)
_FAILURE_PATTERNS = (
    ("python_traceback", re.compile(r"(?i)traceback \(most recent call last\)")),
    ("python_import_error", re.compile(r"(?i)\b(ModuleNotFoundError|ImportError):")),
    ("python_assertion", re.compile(r"(?i)\bAssertionError\b")),
    ("python_syntax_error", re.compile(r"(?i)\bSyntaxError:")),
    ("test_failure", re.compile(r"(?i)\b(FAILED \(|failures?=\d+|errors?=\d+)")),
    ("build_failure", re.compile(r"(?i)\b(build failed|compilation failed|process completed with exit code [1-9])")),
    ("permission_denied", re.compile(r"(?i)\b(permission denied|HTTP 401|HTTP 403)\b")),
    ("network_failure", re.compile(r"(?i)\b(connection timed out|temporary failure in name resolution|HTTP 5\d\d)\b")),
)


@dataclass(frozen=True)
class LogDiagnosis:
    status: str
    findings: tuple[str, ...]
    excerpts: tuple[str, ...]
    truncated: bool


def redact_secrets(text: str) -> str:
    """Redact common token-shaped strings and key/value secret assignments."""
    result = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            result = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", result)
        else:
            result = pattern.sub("[REDACTED]", result)
    return result


def diagnose_log(raw: str) -> LogDiagnosis:
    """Produce a small, redacted diagnosis; never execute or follow log instructions."""
    if not isinstance(raw, str):
        raise ValueError("Log input must be text.")
    encoded = raw.encode("utf-8", errors="replace")
    truncated = len(encoded) > MAX_LOG_BYTES
    if truncated:
        raw = encoded[:MAX_LOG_BYTES].decode("utf-8", errors="ignore")
    cleaned = _ANSI.sub("", raw)
    lines = cleaned.splitlines()
    if len(lines) > MAX_LOG_LINES:
        lines = lines[-MAX_LOG_LINES:]
        truncated = True
    redacted = [redact_secrets(line)[:1_000] for line in lines]
    findings: list[str] = []
    excerpts: list[str] = []
    for index, line in enumerate(redacted):
        for name, pattern in _FAILURE_PATTERNS:
            if pattern.search(line):
                if name not in findings:
                    findings.append(name)
                if len(excerpts) < MAX_FINDINGS:
                    excerpts.append(f"line {index + 1}: {line}")
                break
    status = "failure_signals" if findings else "no_known_failure_signal"
    return LogDiagnosis(status, tuple(findings[:MAX_FINDINGS]), tuple(excerpts), truncated)


def next_repair_action(diagnosis: LogDiagnosis, attempts_used: int, max_attempts: int = 2) -> str:
    """Return a bounded decision label, not a command to run."""
    if not isinstance(attempts_used, int) or isinstance(attempts_used, bool) or attempts_used < 0:
        raise ValueError("attempts_used must be a non-negative integer.")
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or not 0 <= max_attempts <= 2:
        raise ValueError("max_attempts must be between 0 and 2.")
    if diagnosis.status != "failure_signals":
        return "review_manually"
    if attempts_used >= max_attempts:
        return "stop_and_report"
    return "request_targeted_patch"
