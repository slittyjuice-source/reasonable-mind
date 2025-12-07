from __future__ import annotations

import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
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
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def validate(self) -> bool:
        return all([self.constraint_hash, self.plan_id, self.persona_id, self.session_id])

    def to_dict(self) -> Dict[str, str]:
        return {
            "constraint_hash": self.constraint_hash,
            "plan_id": self.plan_id,
            "persona_id": self.persona_id,
            "session_id": self.session_id,
        }


def create_execution_context(constraint_hash: str, plan_id: str, persona_id: str) -> ExecutionContext:
    return ExecutionContext(constraint_hash=constraint_hash, plan_id=plan_id, persona_id=persona_id)


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    mode: ExecutionMode
    exit_code: int
    message: str = ""
    block_reason: Optional[str] = None
    duration: float = 0.0
    blocked: bool = False
    execution_context: Optional[ExecutionContext] = None
    correlation_id: Optional[str] = None
    command: Optional[str] = None
    duration_ms: Optional[float] = None

    @property
    def constraint_hash(self) -> Optional[str]:
        return self.execution_context.constraint_hash if self.execution_context else None

    @property
    def plan_id(self) -> Optional[str]:
        return self.execution_context.plan_id if self.execution_context else None

    @property
    def persona_id(self) -> Optional[str]:
        return self.execution_context.persona_id if self.execution_context else None

    def to_audit_record(self) -> Dict[str, object]:
        context = self.execution_context.to_dict() if self.execution_context else {
            "constraint_hash": None,
            "plan_id": None,
            "persona_id": None,
            "session_id": None,
        }
        return {
            **context,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "block_reason": self.block_reason,
            "mode": self.mode.value,
            "exit_code": self.exit_code,
            "blocked": self.blocked,
            "timestamp": time.time(),
        }


class ExecutionProxy:
    """Validates and executes shell commands with governance checks."""

    DEFAULT_DENYLIST: Tuple[Pattern[str], ...] = (
        re.compile(r"rm\s+-rf\s+/"),
        re.compile(r":\(\)\s*\{:\|:&;.*"),
        re.compile(r"shutdown"),
    )
    SYSTEM_REDIRECT = re.compile(r">>?.*/(etc|bin|usr)/")
    BACKGROUND_OP = re.compile(r"\s&\s")
    SUBSTITUTION = re.compile(r"\$\(|`")

    def __init__(
        self,
        allowlist: Optional[set[str]] = None,
        mode: ExecutionMode = ExecutionMode.LIVE,
        execution_context: Optional[ExecutionContext] = None,
    ) -> None:
        self.allowlist = allowlist or set()
        self.mode = mode
        self.execution_context = execution_context
        self._mocks: List[Tuple[Pattern[str], ExecutionResult]] = []
        self._friction: Dict[str, int] = {}

    @staticmethod
    def _strip_quoted(command: str) -> str:
        return re.sub(r"(['\"]).*?\1", "", command)

    def register_mock(self, pattern: str, result: ExecutionResult) -> None:
        self._mocks.append((re.compile(pattern), result))

    def get_friction_report(self) -> Dict[str, int]:
        return dict(self._friction)

    def _update_friction(self, command: str) -> None:
        key = (command.split()[0] if command.split() else "") or command.strip()
        self._friction[key] = self._friction.get(key, 0) + 1

    def _is_allowlisted_base(self, base_cmd: str) -> bool:
        if not self.allowlist:
            return True
        return base_cmd in self.allowlist

    def _validate_pipeline_allowlist(self, command: str) -> bool:
        segments = [seg.strip() for seg in command.split("|")]
        for seg in segments:
            if not seg:
                continue
            tokens = seg.split()
            if not tokens:
                continue
            if not self._is_allowlisted_base(tokens[0]):
                return False
        return True

    def _is_denied(self, command: str) -> Optional[str]:
        cleaned = self._strip_quoted(command)
        if not cleaned.strip():
            return None
        normalized = cleaned.replace(" ", "")
        if ":(){:|:&};:" in normalized:
            return "Command matches denylist pattern"
        for pattern in self.DEFAULT_DENYLIST:
            if pattern.search(cleaned):
                return "Command matches denylist pattern"
        if any(op in cleaned for op in [";", "&&", "||"]):
            return "Command chaining blocked"
        if self.BACKGROUND_OP.search(cleaned):
            return "Background execution blocked"
        if self.SUBSTITUTION.search(cleaned):
            return "Command substitution blocked: contains $() or backticks"
        if self.SYSTEM_REDIRECT.search(cleaned):
            return "Redirection to system directory blocked"
        if "|" in cleaned and not self._validate_pipeline_allowlist(cleaned):
            return "Pipeline contains non-allowlisted command"
        return None

    def _is_allowlisted(self, command: str) -> bool:
        if not self.allowlist:
            return True
        if "|" in command:
            return self._validate_pipeline_allowlist(command)
        tokens = command.split()
        if not tokens:
            return True
        return self._is_allowlisted_base(tokens[0])

    def execute(self, command: str, execution_context: Optional[ExecutionContext] = None) -> ExecutionResult:
        context = execution_context or self.execution_context
        deny_reason = self._is_denied(command)
        if deny_reason:
            self._update_friction(command)
            return ExecutionResult(
                stdout="",
                stderr="",
                mode=self.mode,
                exit_code=-1,
                block_reason=deny_reason,
                blocked=True,
                execution_context=context,
                command=command,
            )

        if not self._is_allowlisted(command):
            self._update_friction(command)
            return ExecutionResult(
                stdout="",
                stderr="",
                mode=self.mode,
                exit_code=-1,
                block_reason="Command not in allowlist",
                blocked=True,
                execution_context=context,
                command=command,
            )

        if self.mode == ExecutionMode.MOCK:
            for pattern, result in self._mocks:
                if pattern.search(command):
                    result.execution_context = context
                    if not result.stdout and result.message:
                        result.stdout = result.message
                    return result
            return ExecutionResult(
                "",
                "",
                ExecutionMode.MOCK,
                0,
                message="No mock registered",
                execution_context=context,
                command=command,
            )

        if self.mode == ExecutionMode.DRY_RUN:
            return ExecutionResult(
                stdout=f"[DRY RUN] {command}",
                stderr="",
                mode=ExecutionMode.DRY_RUN,
                exit_code=0,
                execution_context=context,
                command=command,
            )

        start = time.time()
        completed = subprocess.run(command, shell=True, capture_output=True, text=True)
        duration = time.time() - start
        return ExecutionResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            mode=ExecutionMode.LIVE,
            exit_code=completed.returncode,
            duration=duration,
            execution_context=context,
            command=command,
        )
