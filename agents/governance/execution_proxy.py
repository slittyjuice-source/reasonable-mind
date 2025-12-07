"""
Execution proxy focused on shell-injection hardening for tests.

The proxy does not execute commands in LIVE mode (to keep tests hermetic);
it validates commands against an allowlist, deny patterns, and shell
metacharacter checks, then returns an ExecutionResult describing the decision.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Pattern, Set
import re
import time
import uuid


class ExecutionMode(Enum):
    LIVE = "live"
    DRY_RUN = "dry_run"
    MOCK = "mock"


@dataclass(frozen=True)
class ExecutionContext:
    constraint_hash: str
    plan_id: str
    persona_id: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def validate(self) -> bool:
        return all(
            bool(value)
            for value in (self.constraint_hash, self.plan_id, self.persona_id)
        )

    def to_dict(self) -> Dict[str, str]:
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
    return ExecutionContext(
        constraint_hash=constraint_hash,
        plan_id=plan_id,
        persona_id=persona_id,
        session_id=session_id or uuid.uuid4().hex[:8],
    )


@dataclass
class ExecutionResult:
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    command: str = ""
    mode: ExecutionMode = ExecutionMode.LIVE
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    message: Optional[str] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    constraint_hash: Optional[str] = None
    plan_id: Optional[str] = None
    persona_id: Optional[str] = None
    execution_context: Optional[ExecutionContext] = None

    def to_audit_record(self) -> Dict[str, object]:
        context = (
            self.execution_context.to_dict()
            if self.execution_context
            else {
                "constraint_hash": self.constraint_hash,
                "plan_id": self.plan_id,
                "persona_id": self.persona_id,
                "session_id": None,
            }
        )
        return {
            **context,
            "correlation_id": self.correlation_id,
            "command": self.command,
            "mode": self.mode.value,
            "exit_code": self.exit_code,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timestamp": int(time.time() * 1000),
        }


class ExecutionProxy:
    _DENY_PATTERNS: List[Pattern[str]] = [
        re.compile(r";"),
        re.compile(r"\|\|"),
        re.compile(r"&&"),
        re.compile(r"\$\("),
        re.compile(r"`"),
        re.compile(r":\(\)\s*\{\s*:?\|:?&\s*;\s*\}:\s*"),
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
        self._mocks.append((re.compile(pattern), result))

    def _record_friction(self, command_token: str) -> None:
        self._friction[command_token] = self._friction.get(command_token, 0) + 1

    def _blocked_result(
        self, command: str, reason: str, ctx: Optional[ExecutionContext]
    ) -> ExecutionResult:
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
            execution_context=ctx,
        )

    def _primary_token(self, command: str) -> str:
        return command.strip().split()[0] if command.strip() else ""

    def _tokenize_pipeline(self, command: str) -> List[str]:
        segments = re.split(r"\|\||&&|;|\|", command)
        tokens: List[str] = []
        for segment in segments:
            token = self._primary_token(segment)
            if token:
                tokens.append(token)
        return tokens

    def _validate_allowlist(self, command: str) -> Optional[str]:
        tokens = self._tokenize_pipeline(command)
        for token in tokens:
            if token not in self.allowlist:
                return f"Command '{token}' is not in allowlist"
        return None

    def _matches_denylist(self, command: str) -> Optional[str]:
        # Return denylist reasons with an explicit 'denylist' marker to aid tests
        if self._DENY_REDIRECT.search(command):
            return "denylist: Redirection to system directories is denied"
        if self._DENY_PIPE_TO_SHELL.search(command):
            return "denylist: Piping to shell is denied"
        for pattern in self._DENY_PATTERNS:
            if pattern.search(command):
                return f"denylist: Pattern '{pattern.pattern}' is denied"
        return None

    def _is_blocked_command(self, command: str) -> Optional[str]:
        token = self._primary_token(command)
        if "rm -rf /" in command:
            return "denylist: Attempt to remove root directory is blocked"
        if token in self._BLOCKED_COMMANDS:
            return f"denylist: Command '{token}' is blocked"
        return None

    def execute(
        self,
        command: str,
        execution_context: Optional[ExecutionContext] = None,
    ) -> ExecutionResult:
        ctx = execution_context or self.execution_context
        # MOCK mode: return registered mock if one matches; otherwise return default mock result
        if self.mode == ExecutionMode.MOCK:
            for pattern, result in self._mocks:
                if pattern.search(command):
                    # If mock provides message but not stdout, prefer message
                    if not result.stdout and result.message:
                        # Create a shallow copy with stdout populated
                        return ExecutionResult(
                            correlation_id=result.correlation_id,
                            command=command or result.command,
                            mode=self.mode,
                            exit_code=result.exit_code,
                            stdout=result.message,
                            stderr=result.stderr,
                            duration_ms=result.duration_ms,
                            message=result.message,
                            blocked=result.blocked,
                            block_reason=result.block_reason,
                            constraint_hash=result.constraint_hash
                            or getattr(ctx, "constraint_hash", None),
                            plan_id=result.plan_id or getattr(ctx, "plan_id", None),
                            persona_id=result.persona_id
                            or getattr(ctx, "persona_id", None),
                            execution_context=result.execution_context or ctx,
                        )
                    return ExecutionResult(
                        correlation_id=result.correlation_id,
                        command=command or result.command,
                        mode=self.mode,
                        exit_code=result.exit_code,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        duration_ms=result.duration_ms,
                        message=result.message,
                        blocked=result.blocked,
                        block_reason=result.block_reason,
                        constraint_hash=result.constraint_hash
                        or getattr(ctx, "constraint_hash", None),
                        plan_id=result.plan_id or getattr(ctx, "plan_id", None),
                        persona_id=result.persona_id
                        or getattr(ctx, "persona_id", None),
                        execution_context=result.execution_context or ctx,
                    )
            # No mock matched
            return ExecutionResult(
                command=command,
                mode=self.mode,
                exit_code=0,
                stdout="",
                stderr="",
                duration_ms=0,
                message="No mock registered",
                execution_context=ctx,
            )

        # Non-MOCK modes: perform block checks
        block_reason = (
            self._matches_denylist(command)
            or self._is_blocked_command(command)
            or self._validate_allowlist(command)
        )
        if block_reason:
            return self._blocked_result(command, block_reason, ctx)

        # DRY_RUN: annotate stdout to indicate dry run
        if self.mode == ExecutionMode.DRY_RUN:
            return ExecutionResult(
                command=command,
                mode=self.mode,
                exit_code=0,
                stdout=f"[DRY RUN] {command}",
                stderr="",
                duration_ms=0,
                execution_context=ctx,
            )

        # LIVE (simulated here for tests): return simulated output
        return ExecutionResult(
            command=command,
            mode=self.mode,
            exit_code=0,
            stdout="Simulated output",
            stderr="",
            duration_ms=10,
            execution_context=ctx,
        )

    def get_friction_report(self) -> Dict[str, int]:
        """Return a copy of the friction counters for analysis/tuning."""
        return dict(self._friction)
