"""Execution proxy - single point of control for shell/process access.

Validates commands against allowlist/denylist, enforces resource limits,
logs with correlation IDs, and supports dry-run/mock modes for testing.

Constitution v1.2 Compliance:
- §6.1: ExecutionContext binds constraint_hash, plan_id, persona_id
- §7.3: Audit records include full context for traceability
"""

import subprocess
import time
import re
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Set, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter


def _utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class ExecutionMode(Enum):
    LIVE = "live"
    DRY_RUN = "dry_run"
    MOCK = "mock"


@dataclass(frozen=True)
class ExecutionContext:
    """
    Immutable context binding for execution (Constitution §6.1).
    
    Every execution should reference the active constraint profile,
    plan, and persona for complete audit traceability.
    
    Attributes:
        constraint_hash: SHA-256 hash of active constraint profile
        plan_id: ID of the currently executing plan
        persona_id: Agent ID from PersonaContext
        session_id: Unique session identifier (auto-generated if not provided)
    """
    constraint_hash: str
    plan_id: str
    persona_id: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def validate(self) -> bool:
        """Ensure all required bindings are present and non-empty."""
        return bool(self.constraint_hash and self.plan_id and self.persona_id)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for serialization."""
        return {
            "constraint_hash": self.constraint_hash,
            "plan_id": self.plan_id,
            "persona_id": self.persona_id,
            "session_id": self.session_id
        }


@dataclass
class ExecutionResult:
    """
    Result of a proxied execution with full audit context.
    
    Constitution §7.3 requires: constraint_hash, plan_id, persona_id
    for complete audit traceability.
    """
    correlation_id: str
    command: str
    mode: ExecutionMode
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    blocked: bool = False
    block_reason: Optional[str] = None
    # Constitution §7.3: Audit compliance fields
    constraint_hash: Optional[str] = None
    plan_id: Optional[str] = None
    persona_id: Optional[str] = None
    timestamp: datetime = field(default_factory=_utc_now)
    
    def to_audit_record(self) -> Dict:
        """Convert to audit record with full context per Constitution §7.3."""
        return {
            "correlation_id": self.correlation_id,
            "command": self.command,
            "mode": self.mode.value,
            "exit_code": self.exit_code,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "duration_ms": self.duration_ms,
            # Constitution §7.3: Required audit fields
            "constraint_hash": self.constraint_hash,
            "plan_id": self.plan_id,
            "persona_id": self.persona_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
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
        audit_callback: Optional[Callable] = None,
        execution_context: Optional[ExecutionContext] = None
    ):
        """
        Initialize execution proxy.
        
        Args:
            mode: Execution mode (LIVE, DRY_RUN, MOCK)
            allowlist: Set of allowed base commands
            timeout_seconds: Max execution time
            max_output_bytes: Max stdout/stderr size
            audit_callback: Callback for audit records
            execution_context: Context binding for Constitution §6.1 compliance
        """
        self.mode = mode
        self.allowlist = allowlist if allowlist is not None else self.DEFAULT_ALLOWLIST
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.audit_callback = audit_callback
        self.execution_context = execution_context
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
        correlation_id: Optional[str] = None,
        execution_context: Optional[ExecutionContext] = None
    ) -> ExecutionResult:
        """
        Execute a command through the proxy.
        
        Args:
            command: Shell command to execute
            cwd: Working directory
            env: Environment variables
            correlation_id: Unique ID for this execution
            execution_context: Override context (uses self.execution_context if None)
        
        Returns:
            ExecutionResult with full audit context per Constitution §7.3
        """
        correlation_id = correlation_id or str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Get context: per-call override > instance default > None
        ctx = execution_context or self.execution_context
        constraint_hash = ctx.constraint_hash if ctx else None
        plan_id = ctx.plan_id if ctx else None
        persona_id = ctx.persona_id if ctx else None
        
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
                block_reason=rejection_reason,
                # Constitution §7.3: Include context even for blocked commands
                constraint_hash=constraint_hash,
                plan_id=plan_id,
                persona_id=persona_id
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
                duration_ms=0,
                constraint_hash=constraint_hash,
                plan_id=plan_id,
                persona_id=persona_id
            )
        else:
            result = self._execute_live(
                command, cwd, env, correlation_id, start_time,
                constraint_hash, plan_id, persona_id
            )
        
        self._audit(result)
        return result
    
    def _execute_live(
        self, command: str, cwd: Optional[str], env: Optional[Dict[str, str]],
        correlation_id: str, start_time: float,
        constraint_hash: Optional[str] = None,
        plan_id: Optional[str] = None,
        persona_id: Optional[str] = None
    ) -> ExecutionResult:
        """Execute command in live mode with full context."""
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
                duration_ms=(time.time() - start_time) * 1000,
                constraint_hash=constraint_hash,
                plan_id=plan_id,
                persona_id=persona_id
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
                block_reason="timeout",
                constraint_hash=constraint_hash,
                plan_id=plan_id,
                persona_id=persona_id
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


def create_execution_context(
    constraint_loader: "ConstraintLoader",
    persona_lock: "PersonaLock", 
    plan_validator: "PlanValidator",
    session_id: Optional[str] = None
) -> ExecutionContext:
    """
    Create execution context from active governance components.
    
    Factory function that wires up the active constraint profile,
    persona, and plan for Constitution §6.1 compliance.
    
    Args:
        constraint_loader: Loader with active profile (has active_hash property)
        persona_lock: Lock with active persona (has active_persona property)
        plan_validator: Validator with active plan (has get_active_plan() method)
        session_id: Optional session ID override
    
    Returns:
        ExecutionContext with all bindings populated
        
    Raises:
        ValueError: If any required component is not active
        
    Example:
        loader = ConstraintLoader(Path("governance/"))
        loader.load("coding_agent_profile")
        
        lock = PersonaLock(sandbox_root)
        lock.lock_persona(persona)
        
        validator = PlanValidator()
        validator.load_plan(plan)
        
        context = create_execution_context(loader, lock, validator)
        proxy = ExecutionProxy(execution_context=context)
    """
    # Validate constraint loader
    if not hasattr(constraint_loader, 'active_hash') or not constraint_loader.active_hash:
        raise ValueError("No constraint profile loaded (constraint_loader.active_hash is None)")
    
    # Validate persona lock
    if not hasattr(persona_lock, 'active_persona') or not persona_lock.active_persona:
        raise ValueError("No persona active (persona_lock.active_persona is None)")
    
    # Validate plan validator
    if not hasattr(plan_validator, 'get_active_plan'):
        raise ValueError("PlanValidator missing get_active_plan() method")
    
    plan = plan_validator.get_active_plan()
    if not plan:
        raise ValueError("No plan active (plan_validator.get_active_plan() returned None)")
    
    if not hasattr(plan, 'plan_id') or not plan.plan_id:
        raise ValueError("Active plan missing plan_id")
    
    return ExecutionContext(
        constraint_hash=constraint_loader.active_hash,
        plan_id=plan.plan_id,
        persona_id=persona_lock.active_persona.agent_id,
        session_id=session_id or str(uuid.uuid4())[:8]
    )