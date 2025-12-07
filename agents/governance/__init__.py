"""Governance utilities for process, execution, and plan validation.

This package exposes a small surface used by the test-suite. Keep imports
at module top-level so linters and static analyzers see public symbols.
"""

from .process_gate import ProcessGate, ProcessStage, StageResult
from .registry import ConstraintRegistry, ConstraintProfile
from .execution_proxy import (
    ExecutionProxy,
    ExecutionMode,
    ExecutionResult,
    ExecutionContext,
    create_execution_context,
)
from .plan_validator import (
    PlanValidator,
    Plan,
    PlanStep,
    ViolationType,
    ValidationResult,
    Violation,
)

__all__ = [
    "ProcessGate",
    "ProcessStage",
    "StageResult",
    "ConstraintRegistry",
    "ConstraintProfile",
    "ExecutionProxy",
    "ExecutionMode",
    "ExecutionResult",
    "ExecutionContext",
    "create_execution_context",
    "PlanValidator",
    "Plan",
    "PlanStep",
    "ViolationType",
    "ValidationResult",
    "Violation",
]
