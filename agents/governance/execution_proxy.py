"""
Execution proxy focused on shell-injection hardening for tests.

The proxy does not execute commands in LIVE mode (to keep tests hermetic);
it validates commands against an allowlist, deny patterns, and shell
metacharacter checks, then returns an ExecutionResult describing the decision.
"""

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Pattern, Set


class ExecutionMode(Enum):
    """Proxy execution modes."""

    LIVE = "live"
    DRY_RUN = "dry_run"
    MOCK = "mock"


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable execution context for audit trails."""

    constraint_hash: str
    plan_id: str
    persona_id: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def validate(self) -> bool:
        """Context is valid when all identifiers are present."""

        return all(
            bool(value)
            for value in (self.constraint_hash, self.plan_id, self.persona_id)
        )

    def to_dict(self) -> Dict[str, str]:
        """Dictionary representation for logging."""

        return {
            "constraint_hash": self.constraint_hash,
            "plan_id": self.plan_id,
            "persona_id": self.persona_id,
            "session_id": self.session_id,
        }


def create_execution_context(
    constraint_hash: str,
    plan_id: str,
    persona_id: str,
    session_id: Optional[str] = None,
) -> ExecutionContext:
    """Helper factory to align with tests."""

    return ExecutionContext(
        constraint_hash=constraint_hash,
        plan_id=plan_id,
        persona_id=persona_id,
        session_id=session_id or uuid.uuid4().hex[:8],
    )


@dataclass
class ExecutionResult:
    """Result of executing (or blocking) a command."""

    correlation_id: str
    command: str
    mode: ExecutionMode
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    blocked: bool = False
    block_reason: Optional[str] = None
    constraint_hash: Optional[str] = None
    plan_id: Optional[str] = None
    persona_id: Optional[str] = None

    def to_audit_record(self) -> Dict[str, object]:
        """Convert to audit-friendly record."""

        return {
            "correlation_id": self.correlation_id,
            "command": self.command,
            "mode": self.mode.value,
            "exit_code": self.exit_code,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "constraint_hash": self.constraint_hash,
            "plan_id": self.plan_id,
            "persona_id": self.persona_id,
            "timestamp": int(time.time() * 1000),
        }


class ExecutionProxy:
    """Shell execution proxy with denylist and allowlist enforcement."""

    _DENY_PATTERNS: List[Pattern[str]] = [
        re.compile(r";"),  # command chaining
        re.compile(r"\|\|"),  # logical or
        re.compile(r"&&"),  # logical and
        re.compile(r"\$\("),  # subshell substitution
        re.compile(r"`"),  # backticks
        re.compile(r":\(\)\s*\{\s*:?\|:?&\s*;\s*\}:\s*"),  # fork bomb
    ]

    _DENY_REDIRECT = re.compile(r">\s*(/etc|/bin|/usr)/")
    _DENY_PIPE_TO_SHELL = re.compile(r"\|\s*(bash|sh)\b")
    _BLOCKED_COMMANDS: Set[str] = {"rm"}

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.LIVE,
        allowlist: Optional[Set[str]] = None,
        execution_context: Optional[ExecutionContext] = None,
    ) -> None:
        self.mode = mode
        self.allowlist: Set[str] = allowlist or {"ls", "cat", "echo", "grep", "find"}
        self.execution_context = execution_context
        self._friction: Dict[str, int] = {}
        self._mocks: List[tuple[Pattern[str], ExecutionResult]] = []

    def register_mock(self, pattern: str, result: ExecutionResult) -> None:
        """Register a regex pattern for mock responses."""

        self._mocks.append((re.compile(pattern), result))

    def _record_friction(self, command_token: str) -> None:
        """Increment friction counts for blocked commands."""

        self._friction[command_token] = self._friction.get(command_token, 0) + 1

    def _blocked_result(self, command: str, reason: str, ctx: Optional[ExecutionContext]) -> ExecutionResult:
        """Create a blocked execution result."""

        token = self._primary_token(command)
        if token:
            self._record_friction(token)

        return ExecutionResult(
            correlation_id=uuid.uuid4().hex[:8],
            command=command,
            mode=self.mode,
            exit_code=-1,
            stdout="",
            stderr="",
            duration_ms=0,
            blocked=True,
            block_reason=reason,
            constraint_hash=getattr(ctx, "constraint_hash", None),
            plan_id=getattr(ctx, "plan_id", None),
            persona_id=getattr(ctx, "persona_id", None),
        )

    def _primary_token(self, command: str) -> str:
        """Extract the first token of a command string."""

        return command.strip().split()[0] if command.strip() else ""

    def _tokenize_pipeline(self, command: str) -> List[str]:
        """Split a command on pipeline and chaining operators."""

        # Split on |, ;, &&, ||
        segments = re.split(r"\|\||&&|;|\|", command)
        tokens: List[str] = []
        for segment in segments:
            token = self._primary_token(segment)
            if token:
                tokens.append(token)
        return tokens

    def _validate_allowlist(self, command: str) -> Optional[str]:
        """Ensure all command tokens are within the allowlist."""

        tokens = self._tokenize_pipeline(command)
        for token in tokens:
            if token not in self.allowlist:
                return f"{token} not in allowlist"
        return None

    def _matches_denylist(self, command: str) -> Optional[str]:
        """Check denylist patterns and redirection rules."""

        if self._DENY_REDIRECT.search(command):
            return "redirect to protected system path"
        if self._DENY_PIPE_TO_SHELL.search(command):
            return "pipe to shell denied"
        for pattern in self._DENY_PATTERNS:
            if pattern.search(command):
                return "denylist metacharacter"
        return None

    def _is_blocked_command(self, command: str) -> Optional[str]:
        """Block known dangerous commands outright."""

        token = self._primary_token(command)
        if "rm -rf /" in command:
            return "denylist dangerous deletion"
        if token in self._BLOCKED_COMMANDS:
            return "denylist command"
        return None

    def execute(
        self,
        command: str,
        execution_context: Optional[ExecutionContext] = None,
    ) -> ExecutionResult:
        """Validate and (virtually) execute a command."""

        ctx = execution_context or self.execution_context

        # Handle registered mocks first
        for pattern, result in self._mocks:
            if pattern.match(command):
                return ExecutionResult(
                    correlation_id=result.correlation_id,
                    command=result.command,
                    mode=result.mode,
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    duration_ms=result.duration_ms,
                    blocked=result.blocked,
                    block_reason=result.block_reason,
                    constraint_hash=getattr(ctx, "constraint_hash", None) or result.constraint_hash,
                    plan_id=getattr(ctx, "plan_id", None) or result.plan_id,
                    persona_id=getattr(ctx, "persona_id", None) or result.persona_id,
                )

        if not command.strip():
            return self._blocked_result(command, "empty command", ctx)

        # Validate against deny patterns and allowlist
        deny_reason = self._matches_denylist(command) or self._is_blocked_command(command)
        if deny_reason:
            return self._blocked_result(command, deny_reason, ctx)

        allow_reason = self._validate_allowlist(command)
        if allow_reason:
            return self._blocked_result(command, allow_reason, ctx)

        start = time.time()
        # We do not execute commands; we just simulate success.
        stdout = ""
        if self.mode == ExecutionMode.DRY_RUN:
            stdout = f"[DRY RUN] {command}"
        elif self.mode == ExecutionMode.MOCK:
            stdout = f"[MOCK] {command}"
        else:
            stdout = "[LIVE EXECUTION SKIPPED]"

        duration_ms = int((time.time() - start) * 1000)

        return ExecutionResult(
            correlation_id=uuid.uuid4().hex[:8],
            command=command,
            mode=self.mode,
            exit_code=0,
            stdout=stdout,
            stderr="",
            duration_ms=duration_ms,
            blocked=False,
            constraint_hash=getattr(ctx, "constraint_hash", None),
            plan_id=getattr(ctx, "plan_id", None),
            persona_id=getattr(ctx, "persona_id", None),
        )

    def get_friction_report(self) -> Dict[str, int]:
        """Return counts of blocked command tokens."""

        return dict(self._friction)
