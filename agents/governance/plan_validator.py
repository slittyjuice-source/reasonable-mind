"""
Minimal plan validator for governance tests.

The validator enforces:
- An active plan must be loaded before validating actions
- Actions must cite a plan step and be present in that step's allowlist
- Destructive operations are rejected without explicit approval
- Plans cannot exceed configured max_steps
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class ViolationType(Enum):
    """Types of validation violations."""

    UNPLANNED_ACTION = auto()
    MISSING_PLAN_CITATION = auto()
    DESTRUCTIVE_OPERATION = auto()
    PLAN_TOO_LARGE = auto()


@dataclass
class Violation:
    """Represents a validation violation."""

    violation_type: ViolationType
    message: str = ""


@dataclass
class ValidationResult:
    """Result of validating a plan or action."""

    is_valid: bool
    violations: List[Violation] = field(default_factory=list)


@dataclass
class PlanStep:
    """Single step within a plan."""

    id: str
    goal: str
    allowed_actions: List[str] = field(default_factory=list)


@dataclass
class Plan:
    """Plan definition used for validation."""

    plan_id: str
    steps: List[PlanStep]
    constraint_profile: str
    persona_id: str
    max_steps: Optional[int] = None


class PlanValidator:
    """Validate actions against a loaded plan."""

    def __init__(self) -> None:
        self._active_plan: Optional[Plan] = None
        self._step_index: Dict[str, PlanStep] = {}

    def load_plan(self, plan: Plan) -> ValidationResult:
        """Load a plan after verifying max_steps."""

        if plan.max_steps is not None and len(plan.steps) > plan.max_steps:
            return ValidationResult(
                is_valid=False,
                violations=[Violation(ViolationType.PLAN_TOO_LARGE, "Plan exceeds max steps")],
            )

        self._active_plan = plan
        self._step_index = {step.id: step for step in plan.steps}
        return ValidationResult(is_valid=True)

    def get_active_plan(self) -> Optional[Plan]:
        """Return the currently loaded plan."""

        return self._active_plan

    def amend_plan(self, step: PlanStep) -> ValidationResult:
        """Add a step if it does not exceed max_steps."""

        if not self._active_plan:
            return ValidationResult(
                is_valid=False,
                violations=[Violation(ViolationType.UNPLANNED_ACTION, "No plan loaded")],
            )

        max_steps = self._active_plan.max_steps
        if max_steps is not None and len(self._active_plan.steps) >= max_steps:
            return ValidationResult(
                is_valid=False,
                violations=[Violation(ViolationType.PLAN_TOO_LARGE, "Plan exceeds max steps")],
            )

        self._active_plan.steps.append(step)
        self._step_index[step.id] = step
        return ValidationResult(is_valid=True)

    def validate_action(
        self,
        action: str,
        parameters: Dict[str, object],
        citation: Optional[str] = None,
    ) -> ValidationResult:
        """Validate a requested action against the active plan."""

        if not self._active_plan:
            return ValidationResult(
                is_valid=False,
                violations=[Violation(ViolationType.UNPLANNED_ACTION, "No active plan")],
            )

        if citation is None:
            return ValidationResult(
                is_valid=False,
                violations=[Violation(ViolationType.MISSING_PLAN_CITATION, "No plan citation provided")],
            )

        step = self._step_index.get(citation)
        if not step:
            return ValidationResult(
                is_valid=False,
                violations=[Violation(ViolationType.UNPLANNED_ACTION, "Plan step not found")],
            )

        if action not in step.allowed_actions:
            return ValidationResult(
                is_valid=False,
                violations=[Violation(ViolationType.UNPLANNED_ACTION, "Action not in allowed list")],
            )

        if self._is_destructive(action, parameters):
            return ValidationResult(
                is_valid=False,
                violations=[Violation(ViolationType.DESTRUCTIVE_OPERATION, "Destructive operation blocked")],
            )

        return ValidationResult(is_valid=True)

    @staticmethod
    def _is_destructive(action: str, parameters: Dict[str, object]) -> bool:
        """Basic heuristic for destructive actions."""

        destructive_keywords = ("delete", "rm", "destroy", "drop", "erase")
        if any(keyword in action for keyword in destructive_keywords):
            return True
        return False
