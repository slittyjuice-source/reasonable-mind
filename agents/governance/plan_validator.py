"""
Plan-to-Action validator ensuring every action ties to an approved plan step.

Supports contingencies, open questions, and enforces small revisable plans.
Emits machine-readable violations for higher-layer handling.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class ViolationType(Enum):
    UNPLANNED_ACTION = "unplanned_action"
    DESTRUCTIVE_OPERATION = "destructive_operation"
    MISSING_PLAN_CITATION = "missing_plan_citation"
    PLAN_STEP_MISMATCH = "plan_step_mismatch"
    PLAN_TOO_LARGE = "plan_too_large"
    CAPABILITY_EXCEEDED = "capability_exceeded"


@dataclass
class Violation:
    """Machine-readable policy violation."""
    violation_type: ViolationType
    action: str
    plan_step: Optional[str]
    message: str
    severity: str  # "error" | "warning"
    suggested_remedy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type.value,
            "action": self.action,
            "plan_step": self.plan_step,
            "message": self.message,
            "severity": self.severity,
            "remedy": self.suggested_remedy
        }


@dataclass
class ValidationResult:
    """Result of plan-to-action validation."""
    is_valid: bool
    violations: List[Violation] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.is_valid,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations]
        }


@dataclass
class PlanStep:
    """A single step in a plan with metadata."""
    id: str
    goal: str
    dependencies: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    allowed_actions: List[str] = field(default_factory=list)
    contingencies: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    requires_approval: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "dependencies": self.dependencies,
            "acceptance_criteria": self.acceptance_criteria,
            "allowed_actions": self.allowed_actions,
            "contingencies": self.contingencies,
            "open_questions": self.open_questions,
            "requires_approval": self.requires_approval
        }


@dataclass
class Plan:
    """A plan with steps, constraints, and size limits."""
    plan_id: str
    steps: List[PlanStep]
    constraint_profile: str
    persona_id: str
    max_steps: int = 5
    
    def validate_size(self) -> bool:
        """Enforce small, revisable plans."""
        return len(self.steps) <= self.max_steps
    
    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def has_open_questions(self) -> bool:
        """Check if any step has unresolved questions."""
        return any(step.open_questions for step in self.steps)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "steps": [s.to_dict() for s in self.steps],
            "constraint_profile": self.constraint_profile,
            "persona_id": self.persona_id,
            "max_steps": self.max_steps
        }


class PlanValidator:
    """
    Validates actions against the current plan.
    
    Usage:
        validator = PlanValidator()
        validator.load_plan(plan)
        result = validator.validate_action("create_file", {"path": "x.py"}, "step-1")
    """
    
    DESTRUCTIVE_ACTIONS: Set[str] = {
        "delete_file", "remove_directory", "drop_table",
        "truncate", "format", "overwrite", "reset", "rm", "unlink"
    }
    
    def __init__(
        self,
        allow_unplanned: bool = False,
        allow_destructive_without_approval: bool = False,
        max_plan_steps: int = 5
    ):
        self.allow_unplanned = allow_unplanned
        self.allow_destructive = allow_destructive_without_approval
        self.max_plan_steps = max_plan_steps
        self._active_plan: Optional[Plan] = None
    
    def load_plan(self, plan: Plan) -> ValidationResult:
        """Load a plan for validation. Validates plan structure first."""
        violations = []
        
        # Check plan size
        if not plan.validate_size():
            violations.append(Violation(
                violation_type=ViolationType.PLAN_TOO_LARGE,
                action="load_plan",
                plan_step=None,
                message=f"Plan has {len(plan.steps)} steps, max is {plan.max_steps}",
                severity="error",
                suggested_remedy="Break into smaller sub-plans"
            ))
        
        if violations:
            return ValidationResult(is_valid=False, violations=violations)
        
        self._active_plan = plan
        return ValidationResult(is_valid=True)
    
    def validate_action(
        self,
        action: str,
        action_params: Dict[str, Any],
        cited_plan_step: Optional[str] = None
    ) -> ValidationResult:
        """Validate an action against the current plan."""
        violations = []
        
        # Check plan loaded
        if not self._active_plan:
            if not self.allow_unplanned:
                violations.append(Violation(
                    violation_type=ViolationType.UNPLANNED_ACTION,
                    action=action,
                    plan_step=None,
                    message="No plan loaded",
                    severity="error",
                    suggested_remedy="Load a plan before executing actions"
                ))
                return ValidationResult(is_valid=False, violations=violations)
            return ValidationResult(is_valid=True)
        
        # Check citation required
        if not cited_plan_step and not self.allow_unplanned:
            violations.append(Violation(
                violation_type=ViolationType.MISSING_PLAN_CITATION,
                action=action,
                plan_step=None,
                message=f"Action '{action}' does not cite a plan step",
                severity="error",
                suggested_remedy="Add plan_step parameter"
            ))
        
        # Check cited step exists
        if cited_plan_step:
            step = self._active_plan.get_step(cited_plan_step)
            if not step:
                violations.append(Violation(
                    violation_type=ViolationType.UNPLANNED_ACTION,
                    action=action,
                    plan_step=cited_plan_step,
                    message=f"Plan step '{cited_plan_step}' not found",
                    severity="error",
                    suggested_remedy="Request re-plan"
                ))
            elif step.allowed_actions and action not in step.allowed_actions:
                violations.append(Violation(
                    violation_type=ViolationType.PLAN_STEP_MISMATCH,
                    action=action,
                    plan_step=cited_plan_step,
                    message=f"Action '{action}' not in allowed actions for step",
                    severity="warning"
                ))
        
        # Check destructive operations
        if self._is_destructive(action, action_params) and not self.allow_destructive:
            violations.append(Violation(
                violation_type=ViolationType.DESTRUCTIVE_OPERATION,
                action=action,
                plan_step=cited_plan_step,
                message=f"Destructive action '{action}' requires approval",
                severity="error",
                suggested_remedy="Get operator approval or set allow_destructive=True"
            ))
        
        has_errors = any(v.severity == "error" for v in violations)
        return ValidationResult(is_valid=not has_errors, violations=violations)
    
    def _is_destructive(self, action: str, params: Dict[str, Any]) -> bool:
        """Check if an action is destructive."""
        if action in self.DESTRUCTIVE_ACTIONS:
            return True
        for value in params.values():
            if isinstance(value, str):
                if any(d in value.lower() for d in ["delete", "remove", "drop", "truncate"]):
                    return True
        return False
    
    def amend_plan(self, new_step: PlanStep) -> ValidationResult:
        """Amend the current plan with a new step (if within limits)."""
        if not self._active_plan:
            return ValidationResult(
                is_valid=False,
                violations=[Violation(
                    violation_type=ViolationType.UNPLANNED_ACTION,
                    action="amend_plan",
                    plan_step=None,
                    message="No plan to amend",
                    severity="error"
                )]
            )
        
        if len(self._active_plan.steps) >= self._active_plan.max_steps:
            return ValidationResult(
                is_valid=False,
                violations=[Violation(
                    violation_type=ViolationType.PLAN_TOO_LARGE,
                    action="amend_plan",
                    plan_step=None,
                    message="Cannot add step: plan at max size",
                    severity="error",
                    suggested_remedy="Complete existing steps or create new plan"
                )]
            )
        
        self._active_plan.steps.append(new_step)
        return ValidationResult(is_valid=True)
    
    def get_active_plan(self) -> Optional[Plan]:
        """Get the currently loaded plan."""
        return self._active_plan
    
    def clear_plan(self) -> None:
        """Clear the active plan."""
        self._active_plan = None
