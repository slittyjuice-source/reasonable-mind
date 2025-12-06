from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class ViolationType(Enum):
    UNPLANNED_ACTION = auto()
    MISSING_PLAN_CITATION = auto()
    DESTRUCTIVE_OPERATION = auto()
    PLAN_TOO_LARGE = auto()


@dataclass
class Violation:
    violation_type: ViolationType
    detail: str


@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[Violation] = field(default_factory=list)


@dataclass
class PlanStep:
    id: str
    goal: str
    allowed_actions: List[str] = field(default_factory=list)


@dataclass
class Plan:
    plan_id: str
    steps: List[PlanStep]
    constraint_profile: str
    persona_id: str
    max_steps: int = 20


class PlanValidator:
    """Validates that actions align with approved plans."""

    def __init__(self) -> None:
        self._active_plan: Optional[Plan] = None
        self._destructive_actions = {"delete_file", "rm", "shutdown"}

    def load_plan(self, plan: Plan) -> ValidationResult:
        if len(plan.steps) > plan.max_steps:
            return ValidationResult(
                is_valid=False,
                violations=[Violation(ViolationType.PLAN_TOO_LARGE, "Plan exceeds maximum steps")],
            )
        self._active_plan = plan
        return ValidationResult(True)

    def amend_plan(self, step: PlanStep) -> ValidationResult:
        if not self._active_plan:
            return ValidationResult(False, [Violation(ViolationType.UNPLANNED_ACTION, "No active plan")])

        if len(self._active_plan.steps) >= self._active_plan.max_steps:
            return ValidationResult(False, [Violation(ViolationType.PLAN_TOO_LARGE, "Plan exceeds maximum steps")])

        self._active_plan.steps.append(step)
        return ValidationResult(True)

    def validate_action(self, action_name: str, params: dict, plan_step_id: Optional[str] = None) -> ValidationResult:
        if not self._active_plan:
            return ValidationResult(False, [Violation(ViolationType.UNPLANNED_ACTION, "No plan loaded")])

        if plan_step_id is None:
            return ValidationResult(False, [Violation(ViolationType.MISSING_PLAN_CITATION, "Missing plan citation")])

        matching = next((s for s in self._active_plan.steps if s.id == plan_step_id), None)
        if not matching:
            return ValidationResult(False, [Violation(ViolationType.UNPLANNED_ACTION, "Plan step not found")])

        if action_name not in matching.allowed_actions:
            return ValidationResult(False, [Violation(ViolationType.UNPLANNED_ACTION, "Action not in allowed actions")])

        if action_name in self._destructive_actions and not params.get("approved", False):
            return ValidationResult(False, [Violation(ViolationType.DESTRUCTIVE_OPERATION, "Approval required")])

        return ValidationResult(True)

    def get_active_plan(self) -> Optional[Plan]:
        return self._active_plan
