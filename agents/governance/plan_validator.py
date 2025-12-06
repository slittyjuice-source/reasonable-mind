"""Simple plan validator for governance tests."""
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
    max_steps: int = 10


@dataclass
class PlanViolation:
    violation_type: ViolationType
    message: str


@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[PlanViolation] = field(default_factory=list)


class PlanValidator:
    """Validates actions against a loaded plan."""

    def __init__(self) -> None:
        self._plan: Optional[Plan] = None

    def load_plan(self, plan: Plan) -> ValidationResult:
        if len(plan.steps) > plan.max_steps:
            return ValidationResult(
                False, [PlanViolation(ViolationType.PLAN_TOO_LARGE, "plan exceeds max steps")]
            )
        self._plan = plan
        return ValidationResult(True, [])

    def get_active_plan(self) -> Optional[Plan]:
        return self._plan

    def validate_action(self, action: str, parameters: dict, plan_step_id: Optional[str] = None) -> ValidationResult:
        if not self._plan:
            return ValidationResult(
                False, [PlanViolation(ViolationType.UNPLANNED_ACTION, "no plan loaded")]
            )

        if not plan_step_id:
            return ValidationResult(
                False, [PlanViolation(ViolationType.MISSING_PLAN_CITATION, "missing plan step id")]
            )

        step = next((s for s in self._plan.steps if s.id == plan_step_id), None)
        if not step or (step.allowed_actions and action not in step.allowed_actions):
            return ValidationResult(
                False, [PlanViolation(ViolationType.UNPLANNED_ACTION, "action not in plan")]
            )

        if "delete" in action or action.startswith("rm"):
            return ValidationResult(
                False, [PlanViolation(ViolationType.DESTRUCTIVE_OPERATION, "requires approval")]
            )

        return ValidationResult(True, [])

    def amend_plan(self, step: PlanStep) -> ValidationResult:
        if not self._plan:
            return ValidationResult(
                False, [PlanViolation(ViolationType.UNPLANNED_ACTION, "no plan loaded")]
            )

        if len(self._plan.steps) >= self._plan.max_steps:
            return ValidationResult(
                False, [PlanViolation(ViolationType.PLAN_TOO_LARGE, "cannot exceed max steps")]
            )

        self._plan.steps.append(step)
        return ValidationResult(True, [])
