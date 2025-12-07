"""Shared pytest fixtures for agents test suite."""

from __future__ import annotations

import pytest

from agents.governance import (
    ConstraintRegistry,
    Plan,
    PlanStep,
    PlanValidator,
    ProcessGate,
    ProcessStage,
    StageResult,
)


# =============================================================================
# PlanValidator Fixtures
# =============================================================================


@pytest.fixture
def plan_validator() -> PlanValidator:
    """Return a fresh PlanValidator instance."""
    return PlanValidator()


@pytest.fixture
def sample_plan() -> Plan:
    """Return a minimal Plan for testing."""
    return Plan(
        plan_id="plan-001",
        steps=[
            PlanStep(
                id="step-1",
                goal="Read input file",
                allowed_actions=["read_file", "list_dir"],
            ),
            PlanStep(
                id="step-2",
                goal="Process data",
                allowed_actions=["compute", "transform"],
            ),
        ],
        constraint_profile="default",
        persona_id="test-persona",
        max_steps=10,
    )


@pytest.fixture
def loaded_validator(plan_validator: PlanValidator, sample_plan: Plan) -> PlanValidator:
    """Return a PlanValidator with a plan already loaded."""
    plan_validator.load_plan(sample_plan)
    return plan_validator


@pytest.fixture
def full_plan() -> Plan:
    """Return a Plan at maximum capacity (20 steps)."""
    return Plan(
        plan_id="plan-full",
        steps=[
            PlanStep(id=f"step-{i}", goal=f"Step {i}", allowed_actions=["action"])
            for i in range(20)
        ],
        constraint_profile="default",
        persona_id="test-persona",
        max_steps=20,
    )


# =============================================================================
# ProcessGate Fixtures
# =============================================================================


@pytest.fixture
def process_gate() -> ProcessGate:
    """Return a fresh ProcessGate instance."""
    return ProcessGate(min_confidence=0.6)


@pytest.fixture
def completed_gate(process_gate: ProcessGate) -> ProcessGate:
    """Return a ProcessGate with all stages completed successfully."""
    for stage in ProcessStage:
        result = StageResult(stage=stage, passed=True, confidence=0.9, evidence=["ok"])
        process_gate.record_stage(result)
    return process_gate


@pytest.fixture
def partial_gate(process_gate: ProcessGate) -> ProcessGate:
    """Return a ProcessGate with first 3 stages completed."""
    stages = list(ProcessStage)[:3]
    for stage in stages:
        result = StageResult(stage=stage, passed=True, confidence=0.8, evidence=["ok"])
        process_gate.record_stage(result)
    return process_gate


# =============================================================================
# ConstraintRegistry Fixtures
# =============================================================================


@pytest.fixture
def constraint_registry() -> ConstraintRegistry:
    """Return a fresh ConstraintRegistry instance."""
    return ConstraintRegistry()


@pytest.fixture
def loaded_registry(constraint_registry: ConstraintRegistry) -> ConstraintRegistry:
    """Return a ConstraintRegistry with a profile already loaded."""
    constraint_registry.load_from_dict(
        "test-profile",
        {"max_depth": 5, "allowed_operations": ["read", "write"]},
    )
    return constraint_registry
