"""
Execution proxy - single point of control for shell/process access.

Validates commands against allowlist/denylist, enforces resource limits,
logs with correlation IDs, and supports dry-run/mock modes for testing.
"""

import subprocess
import time
import re
import uuid
from typing import Optional, Dict, Set, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter


class ExecutionMode(Enum):
    LIVE = "live"
    DRY_RUN = "dry_run"
    MOCK = "mock"


@dataclass
class ExecutionResult:
    """Result of a proxied execution."""
    correlation_id: str
    command: str
    mode: ExecutionMode
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    blocked: bool = False
    block_reason: Optional[str] = None
    
    def to_audit_record(self) -> Dict:
        return {
            "correlation_id": self.correlation_id,
            "command": self.command,
            "mode": self.mode.value,
            "exit_code": self.exit_code,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "duration_ms": self.duration_ms
        }


class ExecutionProxy:
    """
    Strict mediator for shell/process access.
    
    Usage:
        proxy = ExecutionProxy()
        result = proxy.execute("ls -la")
        if result.blocked:
            print(f"Blocked: {result.block_reason}")
    """
    
    DEFAULT_ALLOWLIST: Set[str] = {
        "ls", "cat", "head", "tail", "grep", "find", "echo",
        "pwd", "mkdir", "touch", "cp", "mv", "wc", "sort", "uniq",
        "python", "python3", "pip", "pytest",
        "git", "npm", "node", "npx",
    }
    
    DENYLIST_PATTERNS: List[str] = [
        r"rm\s+-rf\s+/",
        r"rm\s+-rf\s+~",
        r"rm\s+-rf\s+\*",
        r"mkfs\.",
        r"dd\s+if=.*of=/dev/",
        r">\s*/dev/sd",
        r"chmod\s+-R\s+777\s+/",
        r"curl.*\|\s*(ba)?sh",
        r"wget.*\|\s*(ba)?sh",
        r":[(][)][{].*[|].*&.*[}];:",  # Fork bomb
    ]
    
    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.LIVE,
        allowlist: Optional[Set[str]] = None,
        timeout_seconds: int = 300,
        max_output_bytes: int = 1_000_000,
        audit_callback: Optional[Callable] = None
    ):
        self.mode = mode
        self.allowlist = allowlist if allowlist is not None else self.DEFAULT_ALLOWLIST
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.audit_callback = audit_callback
        self._mock_responses: Dict[str, ExecutionResult] = {}
        self._metrics: Dict[str, List[str]] = {"blocked": [], "allowed": [], "timeout": []}
    
    def register_mock(self, command_pattern: str, result: ExecutionResult) -> None:
        """Register a mock response for testing."""
        self._mock_responses[command_pattern] = result
    
    def _extract_base_command(self, command: str) -> str:
        """Extract the base command from a command string."""
        parts = command.strip().split()
        if not parts:
            return ""
        base = parts[0]
        if base in ("sudo", "env", "nohup", "time"):
            return parts[1] if len(parts) > 1 else ""
        return base.split("/")[-1]  # Handle full paths
    
    def _check_denylist(self, command: str) -> Optional[str]:
        """Check if command matches any denylist pattern."""
        for pattern in self.DENYLIST_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return f"Matches denylist: {pattern}"
        return None
    
    def _check_allowlist(self, command: str) -> bool:
        """Check if base command is in allowlist."""
        base = self._extract_base_command(command)
        return base in self.allowlist
    
    def validate(self, command: str) -> tuple[bool, Optional[str]]:
        """Validate a command. Returns (is_valid, rejection_reason)."""
        deny_reason = self._check_denylist(command)
        if deny_reason:
            return False, deny_reason
        
        if not self._check_allowlist(command):
            base = self._extract_base_command(command)
            return False, f"Command '{base}' not in allowlist"
        
        return True, None
    
    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None
    ) -> ExecutionResult:
        """Execute a command through the proxy."""
        correlation_id = correlation_id or str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Validation
        is_valid, rejection_reason = self.validate(command)
        if not is_valid:
            self._metrics["blocked"].append(self._extract_base_command(command))
            result = ExecutionResult(
                correlation_id=correlation_id,
                command=command,
                mode=self.mode,
                exit_code=-1,
                stdout="",
                stderr=f"BLOCKED: {rejection_reason}",
                duration_ms=0,
                blocked=True,
                block_reason=rejection_reason
            )
            self._audit(result)
            return result
        
        self._metrics["allowed"].append(self._extract_base_command(command))
        
        # Mode-specific execution
        if self.mode == ExecutionMode.MOCK:
            result = self._execute_mock(command, correlation_id)
        elif self.mode == ExecutionMode.DRY_RUN:
            result = ExecutionResult(
                correlation_id=correlation_id,
                command=command,
                mode=ExecutionMode.DRY_RUN,
                exit_code=0,
                stdout=f"[DRY RUN] Would execute: {command}",
                stderr="",
                duration_ms=0
            )
        else:
            result = self._execute_live(command, cwd, env, correlation_id, start_time)
        
        self._audit(result)
        return result
    
    def _execute_live(
        self, command: str, cwd: Optional[str], env: Optional[Dict[str, str]],
        correlation_id: str, start_time: float
    ) -> ExecutionResult:
        """Execute command in live mode."""
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=env,
                capture_output=True,
                timeout=self.timeout_seconds,
                text=True
            )
            
            stdout = proc.stdout[:self.max_output_bytes] if proc.stdout else ""
            stderr = proc.stderr[:self.max_output_bytes] if proc.stderr else ""
            
            return ExecutionResult(
                correlation_id=correlation_id,
                command=command,
                mode=ExecutionMode.LIVE,
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_ms=(time.time() - start_time) * 1000
            )
        except subprocess.TimeoutExpired:
            self._metrics["timeout"].append(self._extract_base_command(command))
            return ExecutionResult(
                correlation_id=correlation_id,
                command=command,
                mode=ExecutionMode.LIVE,
                exit_code=-1,
                stdout="",
                stderr=f"TIMEOUT: Exceeded {self.timeout_seconds}s",
                duration_ms=self.timeout_seconds * 1000,
                blocked=True,
                block_reason="timeout"
            )
    
    def _execute_mock(self, command: str, correlation_id: str) -> ExecutionResult:
        """Execute in mock mode for testing."""
        for pattern, mock_result in self._mock_responses.items():
            if re.search(pattern, command):
                return ExecutionResult(
                    correlation_id=correlation_id,
                    command=command,
                    mode=ExecutionMode.MOCK,
                    exit_code=mock_result.exit_code,
                    stdout=mock_result.stdout,
                    stderr=mock_result.stderr,
                    duration_ms=0
                )
        
        return ExecutionResult(
            correlation_id=correlation_id,
            command=command,
            mode=ExecutionMode.MOCK,
            exit_code=0,
            stdout=f"[MOCK] {command}",
            stderr="",
            duration_ms=0
        )
    
    def _audit(self, result: ExecutionResult) -> None:
        """Send result to audit callback if configured."""
        if self.audit_callback:
            self.audit_callback(result.to_audit_record())
    
    def get_friction_report(self) -> Dict[str, int]:
        """Returns commands frequently blocked for profile tuning."""
        return dict(Counter(self._metrics["blocked"]))
    
    def get_metrics(self) -> Dict[str, int]:
        """Get execution metrics summary."""
        return {
            "total_blocked": len(self._metrics["blocked"]),
            "total_allowed": len(self._metrics["allowed"]),
            "total_timeout": len(self._metrics["timeout"]),
        }
