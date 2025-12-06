"""Security-focused execution proxy used by governance tests."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Dict, List, Optional, Pattern, Tuple


class ExecutionMode(Enum):
    LIVE = "live"
    DRY_RUN = "dry_run"
    MOCK = "mock"


@dataclass(frozen=True)
class ExecutionContext:
    constraint_hash: str
    plan_id: str
    persona_id: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def validate(self) -> bool:
        return all([self.constraint_hash, self.plan_id, self.persona_id])

    def to_dict(self) -> dict:
        return {
            "constraint_hash": self.constraint_hash,
            "plan_id": self.plan_id,
            "persona_id": self.persona_id,
            "session_id": self.session_id,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


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
    timestamp: datetime = field(default_factory=_now)

    def to_audit_record(self) -> dict:
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
            "timestamp": self.timestamp.isoformat(),
        }


class ExecutionProxy:
    """Minimal shell injection guard used by the tests."""

    DENYLIST_PATTERNS: List[Pattern[str]] = [
        re.compile(r";"),
        re.compile(r"\|\|"),
        re.compile(r"&&"),
        re.compile(r"(^|\s)&(\s|$)"),
        re.compile(r"`.*?`"),
        re.compile(r"\$\(.*?\)"),
        re.compile(r">>\s*/(etc|bin|usr)/"),
        re.compile(r">\s*/(etc|bin|usr)/"),
        re.compile(r":\(\)\{\s*:\|:&\s*\};:"),
        re.compile(r"\|\s*(mail|sendmail)\b"),
        re.compile(r"\|\s*(sh|bash)\b"),
        re.compile(r"(;|\||&&)\s*(rm\s+-rf|curl|wget)"),
    ]

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.LIVE,
        allowlist: Optional[set[str]] = None,
        execution_context: Optional[ExecutionContext] = None,
    ):
        self.mode = mode
        self.allowlist = allowlist
        self.execution_context = execution_context
        self._friction_log: Dict[str, int] = {}
        self._mocks: List[Tuple[Pattern[str], ExecutionResult]] = []

    def register_mock(self, pattern: str, result: ExecutionResult) -> None:
        self._mocks.append((re.compile(pattern), result))

    def _base_command(self, command: str) -> str:
        return command.strip().split()[0] if command.strip() else ""

    def _strip_literals(self, command: str) -> str:
        return re.sub(r"'[^']*'|\"[^\"]*\"", "", command)

    def _matches_denylist(self, command: str) -> Optional[str]:
        sanitized = self._strip_literals(command)
        for pattern in self.DENYLIST_PATTERNS:
            if pattern.search(sanitized):
                return pattern.pattern
        # Special-case pipes: allow if all segments are allowlisted
        if "|" in sanitized:
            segments = [seg.strip() for seg in command.split("|")]
            bases = [self._base_command(seg) for seg in segments if seg.strip()]
            if self.allowlist and all(b in self.allowlist for b in bases):
                return None
            return "pipe not allowed"
        return None

    def _apply_context(self, result: ExecutionResult, context: Optional[ExecutionContext]) -> ExecutionResult:
        if context:
            result.constraint_hash = context.constraint_hash
            result.plan_id = context.plan_id
            result.persona_id = context.persona_id
        return result

    def _record_friction(self, command: str) -> None:
        base = self._base_command(command)
        if base:
            self._friction_log[base] = self._friction_log.get(base, 0) + 1

    def execute(self, command: str, execution_context: Optional[ExecutionContext] = None) -> ExecutionResult:
        ctx = execution_context or self.execution_context
        correlation_id = str(uuid.uuid4())[:8]

        # Immediate block for destructive rm -rf patterns
        if re.match(r"^rm\s+-rf\b", command.strip()):
            self._record_friction(command)
            return self._apply_context(
                ExecutionResult(
                    correlation_id=correlation_id,
                    command=command,
                    mode=self.mode,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    duration_ms=0,
                    blocked=True,
                    block_reason="blocked by denylist: rm -rf",
                ),
                ctx,
            )

        deny_reason = self._matches_denylist(command)
        if deny_reason:
            self._record_friction(command)
            return self._apply_context(
                ExecutionResult(
                    correlation_id=correlation_id,
                    command=command,
                    mode=self.mode,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    duration_ms=0,
                    blocked=True,
                    block_reason=f"blocked by denylist: {deny_reason}",
                ),
                ctx,
            )

        # allowlist validation
        base = self._base_command(command)
        if self.allowlist is not None and base not in self.allowlist:
            self._record_friction(command)
            return self._apply_context(
                ExecutionResult(
                    correlation_id=correlation_id,
                    command=command,
                    mode=self.mode,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    duration_ms=0,
                    blocked=True,
                    block_reason="not in allowlist",
                ),
                ctx,
            )

        # mock mode
        if self.mode == ExecutionMode.MOCK:
            for pattern, result in self._mocks:
                if pattern.search(command):
                    mocked = ExecutionResult(**{**result.__dict__})
                    mocked.correlation_id = correlation_id
                    mocked.command = command
                    mocked.mode = ExecutionMode.MOCK
                    return self._apply_context(mocked, ctx)

        stdout = ""
        if self.mode == ExecutionMode.DRY_RUN:
            stdout = f"[DRY RUN] {command}"
            exit_code = 0
        else:
            exit_code = 0
        res = ExecutionResult(
            correlation_id=correlation_id,
            command=command,
            mode=self.mode,
            exit_code=exit_code,
            stdout=stdout,
            stderr="",
            duration_ms=0,
            blocked=False,
        )
        return self._apply_context(res, ctx)

    def get_friction_report(self) -> Dict[str, int]:
        report: Dict[str, int] = {}
        for cmd, count in self._friction_log.items():
            base = self._base_command(cmd)
            report[base] = report.get(base, 0) + count
        return report


def create_execution_context(
    constraint_hash: str,
    plan_id: str,
    persona_id: str,
    session_id: Optional[str] = None,
) -> ExecutionContext:
    """Factory with validation used by the tests."""

    context = ExecutionContext(
        constraint_hash=constraint_hash,
        plan_id=plan_id,
        persona_id=persona_id,
        session_id=session_id or str(uuid.uuid4())[:8],
    )
    if not context.validate():
        raise ValueError("ExecutionContext requires non-empty fields")
    return context
