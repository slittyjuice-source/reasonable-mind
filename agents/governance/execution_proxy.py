
"""
Execution proxy focused on shell-injection hardening for tests.

The proxy does not execute commands in LIVE mode (to keep tests hermetic);
it validates commands against an allowlist, deny patterns, and shell

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
import uuid
import re
import time

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

    def _blocked_result(self, command: str, reason: str, ctx: Optional[ExecutionContext]) -> ExecutionResult:
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
        if self._DENY_REDIRECT.search(command):
            return "Redirection to system directories is denied"
        if self._DENY_PIPE_TO_SHELL.search(command):
            return "Piping to shell is denied"
        for pattern in self._DENY_PATTERNS:
            if pattern.search(command):
                return f"Pattern '{pattern.pattern}' is denied"
        return None

    def _is_blocked_command(self, command: str) -> Optional[str]:
        token = self._primary_token(command)
        if "rm -rf /" in command:
            return "Attempt to remove root directory is blocked"
        if token in self._BLOCKED_COMMANDS:
            return f"Command '{token}' is blocked"
        return None

    def execute(
        self,
        command: str,
        execution_context: Optional[ExecutionContext] = None,
    ) -> ExecutionResult:
        ctx = execution_context or self.execution_context
        block_reason = (
            self._validate_allowlist(command)
            or self._matches_denylist(command)
            or self._is_blocked_command(command)
        )
        if block_reason:
            return self._blocked_result(command, block_reason, ctx)
        return ExecutionResult(
            correlation_id=uuid.uuid4().hex[:8],
            command=command,
            mode=self.mode,
            exit_code=0,
            stdout="Simulated output",
            stderr="",
            duration_ms=10,
            blocked=False,
            block_reason=None,
            constraint_hash=getattr(ctx, "constraint_hash", None),
            plan_id=getattr(ctx, "plan_id", None),
            persona_id=getattr(ctx, "persona_id", None),
            execution_context=ctx,
        )
            mode=self.mode,
